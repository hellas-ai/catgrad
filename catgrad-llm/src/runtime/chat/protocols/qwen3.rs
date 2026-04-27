//! Qwen3 / Qwen3-MoE tool-call protocol.
//!
//! Wire format:
//!
//! ```text
//! preamble text<tool_call>{"name": "x", "arguments": {...}}</tool_call>
//! more text<tool_call>{"name": "y", "arguments": {...}}</tool_call>
//! ```
//!
//! Each `<tool_call>...</tool_call>` block carries a single JSON object
//! with at minimum a `name` field; arguments arrive under either
//! `"arguments"` or `"parameters"` (the chat templates in the wild use
//! both).
//!
//! # Strict gating
//!
//! The historical `parse_qwen3_tool_calls` in `helpers/tool_calls.rs`
//! also accepted bare JSON without a sentinel wrapper as a fallback.
//! This implementation deliberately drops that fallback: a model output
//! that contains a JSON object resembling `{"name": "x", ...}` but
//! without the `<tool_call>` wrapper is plain text, not a tool call.
//!
//! # Per-call atomic emission
//!
//! `<tool_call>` opens a buffering mode; only when `</tool_call>`
//! arrives do we parse, validate, and emit the
//! `ToolCallStart` + `ToolCallArgsDelta` + `ToolCallEnd` triple as one
//! atomic unit. This is the unit of streaming: call N is delivered to
//! the client as soon as its closing sentinel is seen, even while the
//! model is still generating call N+1. True per-token argument
//! streaming is out of scope (would require a partial-JSON parser and
//! a wire-level "rollback" concept that neither OpenAI nor Anthropic
//! SSE provides — validation cannot happen mid-args without it).
//!
//! # Residual false-positive risk
//!
//! When tools are bound, the parser cannot distinguish a model that
//! genuinely wants to call a tool from a model that is illustrating a
//! tool call inside a Markdown code fence (e.g. answering "show me an
//! example tool call"). Schema validation reduces blast radius — random
//! model output rarely satisfies a typed schema — but does not
//! eliminate the class. The structural fix lives upstream in the model
//! / tokenizer (sandboxed special tokens that user text cannot
//! produce). When tools are NOT bound,
//! [`ChatTurn::make_parser`](crate::runtime::chat::ChatTurn::make_parser)
//! returns the passthrough parser and this protocol is never
//! instantiated, so the no-tools case is structurally clean.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::runtime::chat::{
    DecodeEvent, IncrementalToolCallParser, ParserError, SentinelMatcher, StopReason,
    ToolDirectory, ToolSpec,
};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Maximum bytes buffered between `<tool_call>` and `</tool_call>`
/// before the parser fails the call as oversized. Larger than any
/// plausible structured tool call (typical: <2 KiB; pathological:
/// nested JSON of a few KiB) and small enough that a runaway
/// generation cannot exhaust gateway memory.
const MAX_TOOL_CALL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Construct a Qwen3 parser bound to the given tool directory.
///
/// The parser owns the `Arc<ToolDirectory>`, so the returned
/// `Box<dyn IncrementalToolCallParser>` is `'static`. Callers may hold
/// it alongside the `ChatTurn` it came from without a self-referential
/// borrow.
pub fn make_parser(directory: Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser> {
    Box::new(Qwen3Parser::new(directory))
}

/// Render the bound tool list into the JSON shape the Qwen3 chat
/// templates expect — the OpenAI-style `[{"type": "function",
/// "function": {...}}, ...]` envelope. The chat template iterates over
/// `tools` and reads `tool.function.name`, `tool.function.description`,
/// `tool.function.parameters`.
pub fn render_tools(specs: &[ToolSpec]) -> JsonValue {
    JsonValue::Array(
        specs
            .iter()
            .map(|spec| {
                let mut function = JsonMap::new();
                function.insert("name".to_string(), JsonValue::String(spec.name.clone()));
                if let Some(description) = &spec.description {
                    function.insert(
                        "description".to_string(),
                        JsonValue::String(description.clone()),
                    );
                }
                function.insert("parameters".to_string(), spec.parameters.clone());
                let mut wrapper = JsonMap::new();
                wrapper.insert(
                    "type".to_string(),
                    JsonValue::String("function".to_string()),
                );
                wrapper.insert("function".to_string(), JsonValue::Object(function));
                JsonValue::Object(wrapper)
            })
            .collect(),
    )
}

struct Qwen3Parser {
    directory: Arc<ToolDirectory>,
    state: State,
    next_index: usize,
}

enum State {
    /// Outside any tool-call block. Watching for `<tool_call>`.
    Outside { matcher: SentinelMatcher },
    /// Inside a tool-call block. Watching for `</tool_call>`; the
    /// matcher's internal buffer is the call's payload.
    Inside { matcher: SentinelMatcher },
    /// A fatal protocol error has been emitted. `feed` and `finish`
    /// return empty from this point — see [`DecodeEvent`] terminal-event
    /// docs.
    Terminated,
}

impl Qwen3Parser {
    fn new(directory: Arc<ToolDirectory>) -> Self {
        Self {
            directory,
            state: State::Outside {
                matcher: SentinelMatcher::new(TOOL_CALL_OPEN),
            },
            next_index: 0,
        }
    }
}

impl IncrementalToolCallParser for Qwen3Parser {
    fn feed(&mut self, text: &str) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        let mut remaining = text.to_string();
        loop {
            match &mut self.state {
                State::Outside { matcher } => {
                    matcher.push(&remaining);
                    remaining.clear();
                    if let Some((before, after)) = matcher.try_match() {
                        if !before.is_empty() {
                            events.push(DecodeEvent::TextDelta(before));
                        }
                        self.state = State::Inside {
                            matcher: SentinelMatcher::new(TOOL_CALL_CLOSE),
                        };
                        remaining = after;
                        if remaining.is_empty() {
                            break;
                        }
                    } else {
                        let safe = matcher.flush_safe_text();
                        if !safe.is_empty() {
                            events.push(DecodeEvent::TextDelta(safe));
                        }
                        break;
                    }
                }
                State::Inside { matcher } => {
                    matcher.push(&remaining);
                    remaining.clear();
                    // Hard cap: oversized payloads are fatal — likely
                    // a runaway generation, not a real call.
                    if matcher.buffered_bytes() > MAX_TOOL_CALL_PAYLOAD_BYTES {
                        return self.fatal(DecodeEvent::ParseError {
                            sentinel: TOOL_CALL_OPEN,
                            source: ParserError::PayloadTooLarge {
                                limit_bytes: MAX_TOOL_CALL_PAYLOAD_BYTES,
                            },
                        });
                    }
                    if let Some((payload, after)) = matcher.try_match() {
                        let index = self.next_index;
                        match parse_payload(&payload, index, &self.directory) {
                            PayloadOutcome::Call(call_events) => {
                                self.next_index += 1;
                                events.extend(call_events);
                                self.state = State::Outside {
                                    matcher: SentinelMatcher::new(TOOL_CALL_OPEN),
                                };
                                remaining = after;
                                if remaining.is_empty() {
                                    break;
                                }
                            }
                            PayloadOutcome::Fatal(error_event) => {
                                events.extend(self.fatal(error_event));
                                return events;
                            }
                        }
                    } else {
                        // Inside, no close sentinel yet — keep buffering.
                        break;
                    }
                }
                State::Terminated => {
                    // Reached if state was set to Terminated mid-loop.
                    break;
                }
            }
        }
        events
    }

    fn finish(&mut self, reason: StopReason) -> Vec<DecodeEvent> {
        if matches!(self.state, State::Terminated) {
            return Vec::new();
        }
        let mut events = Vec::new();
        match &mut self.state {
            State::Outside { matcher } => {
                let leftover = matcher.finish();
                if !leftover.is_empty() {
                    events.push(DecodeEvent::TextDelta(leftover));
                }
                events.push(DecodeEvent::Stop { reason });
                // Outside-finish is normal termination; not a fatal
                // state transition — but no subsequent feed should
                // arrive after finish anyway.
            }
            State::Inside { .. } => {
                // Open call never closed: fatal.
                events.extend(self.fatal(DecodeEvent::ParseError {
                    sentinel: TOOL_CALL_OPEN,
                    source: ParserError::Unterminated,
                }));
            }
            State::Terminated => unreachable!("checked above"),
        }
        events
    }
}

impl Qwen3Parser {
    /// Emit a fatal error event followed by `Stop { ProtocolError }`,
    /// then transition to [`State::Terminated`]. All subsequent calls
    /// to `feed` / `finish` return empty.
    fn fatal(&mut self, error_event: DecodeEvent) -> Vec<DecodeEvent> {
        self.state = State::Terminated;
        vec![
            error_event,
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError,
            },
        ]
    }
}

enum PayloadOutcome {
    /// Validated call — emit `Start`, `ArgsDelta`, `End` contiguously.
    Call(Vec<DecodeEvent>),
    /// Anything that should not become a call: unknown name,
    /// schema-invalid args, or a parse failure. Caller wraps with
    /// `Stop { ProtocolError }` and terminates.
    Fatal(DecodeEvent),
}

fn parse_payload(payload: &str, index: usize, directory: &ToolDirectory) -> PayloadOutcome {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::Malformed("empty tool-call payload".into()),
        });
    }
    let value: JsonValue = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(err) => {
            return PayloadOutcome::Fatal(DecodeEvent::ParseError {
                sentinel: TOOL_CALL_OPEN,
                source: ParserError::from(err),
            });
        }
    };
    let Some(object) = value.as_object() else {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::Malformed("tool-call payload is not a JSON object".into()),
        });
    };
    let Some(name) = object.get("name").and_then(JsonValue::as_str) else {
        return PayloadOutcome::Fatal(DecodeEvent::ParseError {
            sentinel: TOOL_CALL_OPEN,
            source: ParserError::MissingField("name"),
        });
    };
    let args = object
        .get("arguments")
        .or_else(|| object.get("parameters"))
        .cloned()
        .unwrap_or_else(|| JsonValue::Object(JsonMap::new()));

    if directory.lookup(name).is_none() {
        return PayloadOutcome::Fatal(DecodeEvent::UnknownTool {
            name: name.to_string(),
            raw_args: args,
        });
    }
    let errors = directory.validate_args(name, &args);
    if !errors.is_empty() {
        return PayloadOutcome::Fatal(DecodeEvent::InvalidArgs {
            name: name.to_string(),
            args,
            errors,
        });
    }

    let args_text = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
    PayloadOutcome::Call(vec![
        DecodeEvent::ToolCallStart {
            index,
            name: name.to_string(),
        },
        DecodeEvent::ToolCallArgsDelta {
            index,
            delta: args_text,
        },
        DecodeEvent::ToolCallEnd { index, args },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::ToolSpec;
    use serde_json::json;

    fn add_tool() -> ToolSpec {
        ToolSpec::new(
            "add",
            Some("add two numbers".into()),
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
                "additionalProperties": false,
            }),
        )
    }

    fn directory_with_add() -> Arc<ToolDirectory> {
        Arc::new(ToolDirectory::new(vec![add_tool()]).unwrap())
    }

    fn run(parser: &mut dyn IncrementalToolCallParser, chunks: &[&str]) -> Vec<DecodeEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.feed(chunk));
        }
        events.extend(parser.finish(StopReason::EndOfText));
        events
    }

    fn last_stop_reason(events: &[DecodeEvent]) -> StopReason {
        events
            .iter()
            .rev()
            .find_map(|e| match e {
                DecodeEvent::Stop { reason } => Some(*reason),
                _ => None,
            })
            .expect("expected a Stop event")
    }

    #[test]
    fn plain_text_passes_through_as_text_delta() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(&mut p, &["hello world"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::TextDelta(s) if s == "hello world"));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn valid_call_emits_start_args_end() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#],
        );
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { index: 0, name } if name == "add"));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { index: 0, .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { index: 0, .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    /// The streaming-claim test: a single call must be fully emitted
    /// when its closing sentinel arrives, before any `finish()` call.
    /// This proves per-call streaming (not just end-of-stream batch).
    #[test]
    fn call_emitted_atomically_when_close_sentinel_arrives() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        // Start, ArgsDelta, End — all here, before finish().
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[1], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::Stop { .. })));
    }

    #[test]
    fn unknown_tool_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"<tool_call>{"name":"delete_db","arguments":{}}</tool_call>"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "delete_db"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn schema_invalid_args_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":"one"}}</tool_call>"#],
        );
        assert_eq!(events.len(), 2);
        let DecodeEvent::InvalidArgs { name, errors, .. } = &events[0] else {
            panic!("expected InvalidArgs, got {events:?}");
        };
        assert_eq!(name, "add");
        assert!(!errors.is_empty());
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn malformed_json_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(&mut p, &["<tool_call>not json at all</tool_call>"]);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn missing_name_field_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(&mut p, &[r#"<tool_call>{"arguments":{}}</tool_call>"#]);
        assert_eq!(events.len(), 2);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(source, ParserError::MissingField("name")));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn parameters_key_is_accepted_as_arguments() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"<tool_call>{"name":"add","parameters":{"a":1,"b":2}}</tool_call>"#],
        );
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn raw_json_without_sentinel_is_plain_text() {
        // Critical: bare JSON resembling a tool call must NOT be parsed.
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#],
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == r#"Here is some JSON: {"name":"add","arguments":{"a":1,"b":2}}"#
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_open_sentinel_split_across_feeds() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let mut events = Vec::new();
        events.extend(p.feed("preamble <tool_c"));
        assert!(events.iter().all(|e| match e {
            DecodeEvent::TextDelta(s) => !s.contains('<'),
            _ => true,
        }));
        events.extend(p.feed(r#"all>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> done"#));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "preamble "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " done"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn partial_close_sentinel_split_across_feeds() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let mut events = Vec::new();
        events.extend(p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_"#));
        assert!(events.is_empty(), "got events: {events:?}");
        events.extend(p.feed("call>"));
        events.extend(p.finish(StopReason::EndOfText));
        assert!(matches!(&events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallEnd { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn sentinel_prefix_that_doesnt_resolve_emits_as_text() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let mut events = Vec::new();
        events.extend(p.feed("hello <to"));
        events.extend(p.feed("morrow"));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "hello <tomorrow");
    }

    #[test]
    fn utf8_multibyte_split_across_feeds_does_not_panic() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let mut events = Vec::new();
        events.extend(p.feed("héllo "));
        events.extend(p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#));
        events.extend(p.finish(StopReason::EndOfText));
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            }
        }
        assert_eq!(text, "héllo ");
    }

    #[test]
    fn multiple_valid_tool_calls_in_sequence() {
        let mul = ToolSpec::new(
            "mul",
            None,
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
            }),
        );
        let dir = Arc::new(ToolDirectory::new(vec![add_tool(), mul]).unwrap());
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[
                r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
                r#"<tool_call>{"name":"mul","arguments":{"a":3,"b":4}}</tool_call>"#,
            ],
        );
        // Start(0,add), ArgsDelta(0), End(0), Start(1,mul), ArgsDelta(1), End(1), Stop
        assert_eq!(events.len(), 7);
        assert!(matches!(
            &events[0],
            DecodeEvent::ToolCallStart { index: 0, name } if name == "add"
        ));
        assert!(matches!(
            &events[3],
            DecodeEvent::ToolCallStart { index: 1, name } if name == "mul"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn text_then_call_then_text_in_single_feed() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"first <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> last"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::TextDelta(s) if s == "first "
        ));
        assert!(matches!(&events[1], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(&events[2], DecodeEvent::ToolCallArgsDelta { .. }));
        assert!(matches!(&events[3], DecodeEvent::ToolCallEnd { .. }));
        assert!(matches!(
            &events[4],
            DecodeEvent::TextDelta(s) if s == " last"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::EndOfText);
    }

    #[test]
    fn unterminated_tool_call_is_terminal_with_protocol_error() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"<tool_call>{"name":"add","arguments":{"a":1"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::Unterminated,
                ..
            }
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_directory_makes_every_call_terminal_unknown() {
        let dir = Arc::new(ToolDirectory::new(vec![]).unwrap());
        let mut p = Qwen3Parser::new(dir);
        let events = run(
            &mut p,
            &[r#"<tool_call>{"name":"add","arguments":{}}</tool_call>"#],
        );
        assert!(matches!(
            &events[0],
            DecodeEvent::UnknownTool { name, .. } if name == "add"
        ));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn empty_payload_is_terminal_parse_error() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let events = run(&mut p, &["<tool_call></tool_call>"]);
        assert!(matches!(&events[0], DecodeEvent::ParseError { .. }));
        assert_eq!(last_stop_reason(&events), StopReason::ProtocolError);
    }

    #[test]
    fn after_fatal_error_subsequent_feed_and_finish_return_empty() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        // Trigger fatal via unknown tool.
        let first = p.feed(r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#);
        assert!(matches!(&first[0], DecodeEvent::UnknownTool { .. }));
        assert!(matches!(
            &first[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        // Subsequent feed: empty.
        let after_feed = p.feed("any further text");
        assert!(after_feed.is_empty(), "got events: {after_feed:?}");
        let after_more =
            p.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        assert!(after_more.is_empty(), "got events: {after_more:?}");
        // Subsequent finish: empty.
        let after_finish = p.finish(StopReason::EndOfText);
        assert!(after_finish.is_empty(), "got events: {after_finish:?}");
    }

    #[test]
    fn payload_over_limit_without_close_is_terminal() {
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        // Open the sentinel, then push a single oversized chunk (just
        // bytes; not real JSON — the size check fires before parse).
        let oversize = "x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1);
        let mut events = Vec::new();
        events.extend(p.feed("<tool_call>"));
        events.extend(p.feed(&oversize));
        // Fatal error must have been emitted by now (no close arrived).
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError, got {events:?}");
        };
        assert!(matches!(
            source,
            ParserError::PayloadTooLarge { limit_bytes }
                if *limit_bytes == MAX_TOOL_CALL_PAYLOAD_BYTES
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        // Confirm subsequent feed/finish return empty.
        assert!(p.feed("more").is_empty());
        assert!(p.finish(StopReason::EndOfText).is_empty());
    }

    #[test]
    fn payload_over_limit_with_close_in_same_feed_still_fatal() {
        // Even if the close sentinel is present, an oversized payload is
        // suspicious and must fail. The size check is eager — fires
        // before try_match consults the close sentinel.
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let mut chunk = String::from("<tool_call>");
        chunk.push_str(&"x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES + 1));
        chunk.push_str("</tool_call>");
        let events = p.feed(&chunk);
        assert!(matches!(
            &events[0],
            DecodeEvent::ParseError {
                source: ParserError::PayloadTooLarge { .. },
                ..
            }
        ));
        assert!(matches!(
            &events[1],
            DecodeEvent::Stop {
                reason: StopReason::ProtocolError
            }
        ));
        // Did not parse out a tool call.
        assert!(!events.iter().any(|e| matches!(e, DecodeEvent::ToolCallStart { .. })));
    }

    #[test]
    fn error_message_does_not_include_oversized_payload() {
        // Operator-facing message must not echo the (potentially huge,
        // potentially user-controlled) payload bytes.
        let dir = directory_with_add();
        let mut p = Qwen3Parser::new(dir);
        let secret = "SUPER_SECRET_TOKEN_THAT_SHOULD_NOT_LEAK";
        let mut chunk = String::from("<tool_call>");
        chunk.push_str(secret);
        chunk.push_str(&"x".repeat(MAX_TOOL_CALL_PAYLOAD_BYTES));
        let events = p.feed(&chunk);
        let DecodeEvent::ParseError { source, .. } = &events[0] else {
            panic!("expected ParseError");
        };
        let msg = source.to_string();
        assert!(
            !msg.contains(secret),
            "oversized payload error message must not include payload bytes; got: {msg}"
        );
    }
}

#[cfg(test)]
mod proptests {
    //! Chunk-invariance: feeding the same model output as one string
    //! versus split across chunk boundaries produces the same final
    //! decoded turn (or the same DecodeFailure). This is the high-
    //! signal version of dozens of "split at byte N" scenario tests
    //! — the property holds for ANY split point.

    use super::*;
    use crate::runtime::chat::protocols::test_util;
    use proptest::prelude::*;

    /// Hand-curated set of inputs that exercise the protocol's
    /// interesting cases: plain text, a single valid call, a call
    /// embedded in surrounding text, multiple calls, sentinel-shaped
    /// text outside a sentinel, malformed content. Sourced from the
    /// scenario tests above so the property is tested over the same
    /// surface they cover.
    fn interesting_inputs() -> Vec<&'static str> {
        vec![
            // plain text
            "hello world",
            // single valid call
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
            // call surrounded by text
            r#"prefix <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> suffix"#,
            // two calls in sequence
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call><tool_call>{"name":"add","arguments":{"a":3,"b":4}}</tool_call>"#,
            // sentinel-shaped text that isn't a sentinel
            "the docs say <tool_call> but it's just text",
            // unknown tool (should produce a fatal event;
            // chunk-invariance still holds — same failure either way)
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
            // call with text suffix only
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> done"#,
        ]
    }

    proptest! {
        /// For each interesting input, feeding it as one chunk vs
        /// arbitrary 2-way splits yields the same final result.
        #[test]
        fn two_way_split_is_invariant(
            input_idx in 0_usize..7,
            split in 0_usize..200,
        ) {
            let inputs = interesting_inputs();
            let text = inputs[input_idx];
            let whole = test_util::decode_whole(make_parser, text);
            let chunked = test_util::decode_chunked(make_parser, text, &[split]);
            // Compare via Debug — covers both Ok-with-equal-turn and
            // Err-with-equal-failure cases.
            prop_assert_eq!(format!("{:?}", whole), format!("{:?}", chunked));
        }

        /// Random N-way splits also preserve invariance. Three splits
        /// covers most byte-boundary edge cases (sentinel boundary,
        /// JSON boundary, args boundary).
        #[test]
        fn n_way_split_is_invariant(
            input_idx in 0_usize..7,
            mut splits in prop::collection::vec(0_usize..200, 1..5),
        ) {
            let inputs = interesting_inputs();
            let text = inputs[input_idx];
            splits.sort_unstable();
            let whole = test_util::decode_whole(make_parser, text);
            let chunked = test_util::decode_chunked(make_parser, text, &splits);
            prop_assert_eq!(format!("{:?}", whole), format!("{:?}", chunked));
        }
    }
}
