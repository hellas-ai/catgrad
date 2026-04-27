//! Wire-neutral accumulator that collects a stream of [`DecodeEvent`]s
//! into an **ordered** assistant turn.
//!
//! [`AssistantTurnAccumulator`] is the **single place** where event-
//! sequence rules are enforced. The wire mappers in [`super::wire`]
//! and the in-process consumers (e.g. the llama tool-dispatch loop)
//! are all built on top of it. Each rule (atomic Start/ArgsDelta/End,
//! no Stop after data, args/args_json agreement, etc.) is written
//! and tested **once**, instead of once per consumer.
//!
//! # Ordered model
//!
//! [`DecodedAssistantTurn::parts`] is a `Vec<DecodedPart>` that
//! interleaves text runs and validated tool calls in **emission
//! order**. This single ordered list is the canonical source of
//! truth: the OpenAI mapper concatenates `Text` parts into a single
//! content string; the Anthropic mapper maps each part to a
//! `content_block`; the llama example filters `ToolCall` parts for
//! programmatic dispatch. No mapper needs to maintain a parallel
//! ordered-block state.
//!
//! # Two entry points
//!
//! The streaming-first form is the type itself:
//! [`AssistantTurnAccumulator::feed`] consumes one event,
//! [`AssistantTurnAccumulator::snapshot`] / [`AssistantTurnAccumulator::into_turn`]
//! produce the final turn. [`DecodedAssistantTurn::from_events`] is
//! a convenience wrapper for in-process consumers that just want to
//! drain a finite event stream and read the result.
//!
//! # Sequence rules enforced
//!
//! Each rule below produces
//! [`DecodeFailure::InternalSequence`](super::wire::DecodeFailure::InternalSequence)
//! and **poisons** the accumulator. Subsequent `feed` / `snapshot` /
//! `into_turn` return the same failure.
//!
//! - `ToolCallStart` while a call is already in flight (calls are
//!   per-parser-contract atomic `Start` / `ArgsDelta` / `End`
//!   triples — nothing else may interleave).
//! - `ToolCallStart` reusing an already-completed `parser_index`.
//! - `TextDelta` while a tool call is in flight.
//! - `ToolCallArgsDelta` or `ToolCallEnd` whose `parser_index`
//!   doesn't match the currently in-flight call.
//! - `ToolCallEnd::args` disagrees with the concatenation of the
//!   preceding `ToolCallArgsDelta` payloads (after parsing the
//!   concatenated string as JSON).
//! - **Any event arriving after a `Stop` event** — Stop is genuinely
//!   terminal in the accumulator, just as it is in the parser
//!   contract. Includes a duplicate `Stop`.
//!
//! Plus the existing terminal events the wire mappers also surface:
//! `UnknownTool`, `InvalidArgs`, `ParseError`.
//!
//! # `snapshot` / `into_turn` preconditions
//!
//! Both require: not poisoned, a `Stop` was observed, the observed
//! `StopReason` is not `ProtocolError`, no tool call still in
//! flight. `snapshot` borrows; `into_turn` consumes.

use serde_json::Value as JsonValue;

use super::event::{DecodeEvent, StopReason};
use super::wire::DecodeFailure;

/// One ordered part of an assistant turn — either a contiguous text
/// run or one validated tool call. Successive `TextDelta` events
/// (with no intervening tool call) coalesce into a single `Text`
/// part; each completed tool call becomes one `ToolCall`.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedPart {
    Text(String),
    ToolCall(DecodedToolCall),
}

/// Buffered assistant turn — what the model actually said, in a
/// wire-neutral, **ordered** shape.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAssistantTurn {
    /// Text runs and tool calls in emission order. Single source of
    /// ordering for all consumers.
    pub parts: Vec<DecodedPart>,
    /// Why generation stopped, as the parser observed it. Never
    /// `ProtocolError` here (that path errors out of `snapshot` /
    /// `into_turn`).
    pub stop_reason: StopReason,
}

impl DecodedAssistantTurn {
    /// Concatenated text from all `Text` parts, in emission order.
    /// Convenience for surfaces (e.g. OpenAI) whose wire format wants
    /// a single content string rather than ordered parts.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            if let DecodedPart::Text(s) = part {
                out.push_str(s);
            }
        }
        out
    }

    /// Validated tool calls in emission order. Convenience for
    /// programmatic consumers that don't care about interleaved text.
    pub fn tool_calls(&self) -> impl Iterator<Item = &DecodedToolCall> + '_ {
        self.parts.iter().filter_map(|p| match p {
            DecodedPart::ToolCall(c) => Some(c),
            _ => None,
        })
    }

    /// Drain a finite [`DecodeEvent`] sequence into a buffered turn.
    pub fn from_events(
        events: impl IntoIterator<Item = DecodeEvent>,
    ) -> Result<Self, DecodeFailure> {
        let mut acc = AssistantTurnAccumulator::new();
        for event in events {
            acc.feed(event)?;
        }
        acc.into_turn()
    }
}

/// One validated tool call from a [`DecodedAssistantTurn`].
///
/// Carries both `args` (the parsed semantic value, for execution)
/// and `args_json` (the raw JSON-encoded string, for wire-format
/// message history that uses the OpenAI "arguments-as-string"
/// convention). The accumulator asserts these two agree at
/// `ToolCallEnd` time.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedToolCall {
    /// 0-based index assigned by the parser within this turn.
    pub parser_index: usize,
    pub name: String,
    /// Parsed and schema-validated argument value. Use for execution.
    pub args: JsonValue,
    /// Raw JSON-encoded string form of `args`, taken from the
    /// concatenation of `ToolCallArgsDelta` payloads.
    pub args_json: String,
}

/// Streaming-first builder for a [`DecodedAssistantTurn`].
#[derive(Debug, Default)]
pub struct AssistantTurnAccumulator {
    /// Ordered parts so far. The contract guarantees that
    /// `parts.last()` (when it's `Text`) is the open text run we
    /// extend on subsequent `TextDelta`s.
    parts: Vec<DecodedPart>,
    /// `Some` while a tool call is in flight. Per parser contract
    /// only one call can be open at a time, so `Option` is the
    /// truthful shape (not a `HashMap`).
    in_progress: Option<InProgressCall>,
    /// Set once a fatal event (terminal parser variant or sequence
    /// violation) poisons the accumulator. All subsequent
    /// `feed` / `snapshot` / `into_turn` return a clone of this
    /// failure.
    failure: Option<DecodeFailure>,
    /// Captured from the `Stop` event. Once `Some`, the accumulator
    /// is **terminal**: any further event (including another `Stop`)
    /// is a sequence violation.
    stop_reason: Option<StopReason>,
}

#[derive(Debug)]
struct InProgressCall {
    parser_index: usize,
    name: String,
    args_text: String,
}

impl AssistantTurnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the poisoned failure if any. Wire mappers that compose
    /// with this accumulator use it to short-circuit their own
    /// `finish` / `snapshot` paths without keeping a duplicate copy
    /// of the failure state.
    pub fn failure(&self) -> Option<&DecodeFailure> {
        self.failure.as_ref()
    }

    /// True once a `Stop` event has been observed. After this, the
    /// accumulator is terminal — `feed` rejects any further event.
    pub fn is_stopped(&self) -> bool {
        self.stop_reason.is_some()
    }

    /// Consume one event. Returns `Err` for terminal parser events
    /// (poisoning the accumulator) and for sequence-rule violations.
    pub fn feed(&mut self, event: DecodeEvent) -> Result<(), DecodeFailure> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone());
        }
        if self.stop_reason.is_some() {
            // Stop is genuinely terminal — no further event of any
            // kind is acceptable.
            return self.poison(DecodeFailure::internal_sequence(
                "event arrived after Stop — accumulator is already terminal",
            ));
        }
        match event {
            DecodeEvent::TextDelta(s) => {
                if let Some(open) = &self.in_progress {
                    return self.poison(DecodeFailure::internal_sequence(format!(
                        "TextDelta arrived while tool call {} is still open; \
                         parser must emit ToolCallStart/ArgsDelta/End atomically",
                        open.parser_index
                    )));
                }
                // Coalesce contiguous text into the trailing Text part.
                if let Some(DecodedPart::Text(text)) = self.parts.last_mut() {
                    text.push_str(&s);
                } else {
                    self.parts.push(DecodedPart::Text(s));
                }
                Ok(())
            }
            DecodeEvent::ToolCallStart { index, name } => {
                if let Some(open) = &self.in_progress {
                    return self.poison(DecodeFailure::internal_sequence(format!(
                        "ToolCallStart for index {index} arrived while \
                         call {} is still open",
                        open.parser_index
                    )));
                }
                if self.parts.iter().any(|p| matches!(p, DecodedPart::ToolCall(c) if c.parser_index == index))
                {
                    return self.poison(DecodeFailure::internal_sequence(format!(
                        "ToolCallStart for index {index} but a call at \
                         that index already completed"
                    )));
                }
                self.in_progress = Some(InProgressCall {
                    parser_index: index,
                    name,
                    args_text: String::new(),
                });
                Ok(())
            }
            DecodeEvent::ToolCallArgsDelta { index, delta } => {
                // Two-step borrow split avoids a `&mut self` overlap
                // when the wrong-index branch needs `poison` while
                // also borrowing `self.in_progress`.
                let open_index = self.in_progress.as_ref().map(|c| c.parser_index);
                match open_index {
                    None => self.poison(DecodeFailure::internal_sequence(format!(
                        "ToolCallArgsDelta for index {index} with no \
                         preceding ToolCallStart"
                    ))),
                    Some(open) if open != index => {
                        self.poison(DecodeFailure::internal_sequence(format!(
                            "ToolCallArgsDelta for index {index} but the \
                             in-flight call is index {open}"
                        )))
                    }
                    Some(_) => {
                        self.in_progress
                            .as_mut()
                            .expect("checked above")
                            .args_text
                            .push_str(&delta);
                        Ok(())
                    }
                }
            }
            DecodeEvent::ToolCallEnd { index, args } => {
                let Some(call) = self.in_progress.take() else {
                    return self.poison(DecodeFailure::internal_sequence(format!(
                        "ToolCallEnd for index {index} with no \
                         preceding ToolCallStart"
                    )));
                };
                if call.parser_index != index {
                    let actual = call.parser_index;
                    // Restore in_progress so the failure is observable
                    // in further `feed` calls if the caller ignores Err.
                    self.in_progress = Some(call);
                    return self.poison(DecodeFailure::internal_sequence(format!(
                        "ToolCallEnd for index {index} but the in-flight \
                         call is index {actual}"
                    )));
                }
                if let Err(failure) = check_args_agree(index, &call.args_text, &args) {
                    return self.poison(failure);
                }
                self.parts.push(DecodedPart::ToolCall(DecodedToolCall {
                    parser_index: call.parser_index,
                    name: call.name,
                    args,
                    args_json: call.args_text,
                }));
                Ok(())
            }
            DecodeEvent::Stop { reason } => {
                self.stop_reason = Some(reason);
                Ok(())
            }
            DecodeEvent::UnknownTool { name, raw_args } => {
                self.poison(DecodeFailure::unknown_tool(name, raw_args))
            }
            DecodeEvent::InvalidArgs {
                name,
                args,
                errors,
            } => self.poison(DecodeFailure::invalid_args(name, args, errors)),
            DecodeEvent::ParseError { sentinel, source } => {
                self.poison(DecodeFailure::parse_error(sentinel, source))
            }
        }
    }

    /// Snapshot the buffered turn. Borrowing variant — wire mappers
    /// call this to build their per-surface snapshot frames.
    ///
    /// Fallible: returns `Err` if the accumulator is poisoned, no
    /// `Stop` has been observed, the observed `StopReason` is
    /// `ProtocolError`, or any tool call is still in flight.
    pub fn snapshot(&self) -> Result<DecodedAssistantTurn, DecodeFailure> {
        self.assert_clean_close()?;
        let stop_reason = self.stop_reason.expect("assert_clean_close ensures Stop seen");
        Ok(DecodedAssistantTurn {
            parts: self.parts.clone(),
            stop_reason,
        })
    }

    /// Consume the accumulator and produce a [`DecodedAssistantTurn`].
    /// Same preconditions as [`Self::snapshot`].
    pub fn into_turn(self) -> Result<DecodedAssistantTurn, DecodeFailure> {
        self.assert_clean_close()?;
        let stop_reason = self.stop_reason.expect("assert_clean_close ensures Stop seen");
        Ok(DecodedAssistantTurn {
            parts: self.parts,
            stop_reason,
        })
    }

    /// Postcondition check shared by `snapshot` / `into_turn` and
    /// also exposed for wire mappers that need to validate before
    /// emitting their own terminal frames (without consuming the
    /// accumulator).
    pub fn assert_clean_close(&self) -> Result<(), DecodeFailure> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone());
        }
        let Some(stop_reason) = self.stop_reason else {
            return Err(DecodeFailure::internal_sequence(
                "snapshot/into_turn called before any Stop event was observed",
            ));
        };
        if matches!(stop_reason, StopReason::ProtocolError) {
            return Err(DecodeFailure::internal_sequence(
                "snapshot/into_turn called after Stop { ProtocolError } — \
                 a fatal event preceded it",
            ));
        }
        if let Some(open) = &self.in_progress {
            return Err(DecodeFailure::internal_sequence(format!(
                "snapshot/into_turn called with tool call {} still open",
                open.parser_index
            )));
        }
        Ok(())
    }

    /// Assert that no tool call is currently in flight. Used by
    /// wire mappers' streaming `finish` to enforce the per-call
    /// atomicity rule without requiring the full `snapshot`
    /// preconditions (Stop may not have been observed yet at that
    /// moment in the wire mapper's lifecycle).
    pub fn assert_no_open_calls(&mut self) -> Result<(), DecodeFailure> {
        if let Some(failure) = &self.failure {
            return Err(failure.clone());
        }
        if let Some(open) = &self.in_progress {
            let idx = open.parser_index;
            return self.poison(DecodeFailure::internal_sequence(format!(
                "finish called with tool call {idx} still open"
            )));
        }
        Ok(())
    }

    fn poison(&mut self, failure: DecodeFailure) -> Result<(), DecodeFailure> {
        self.failure = Some(failure.clone());
        Err(failure)
    }
}

/// Verify the concatenated `ArgsDelta` payloads agree with
/// `ToolCallEnd::args`. Empty `args_text` is accepted — current
/// parsers always emit an `ArgsDelta`, but the contract permits a
/// future parser to omit it.
fn check_args_agree(
    index: usize,
    args_text: &str,
    args: &JsonValue,
) -> Result<(), DecodeFailure> {
    if args_text.is_empty() {
        return Ok(());
    }
    match serde_json::from_str::<JsonValue>(args_text) {
        Ok(parsed) if &parsed == args => Ok(()),
        Ok(parsed) => Err(DecodeFailure::internal_sequence(format!(
            "ToolCallEnd args disagree with concatenated ArgsDelta \
             payloads for index {index}: deltas parsed to {parsed}, \
             end carried {args}"
        ))),
        Err(err) => Err(DecodeFailure::internal_sequence(format!(
            "concatenated ArgsDelta payloads for index {index} \
             failed to parse as JSON: {err}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::SchemaError;
    use crate::runtime::chat::event::ParserError;
    use crate::runtime::chat::wire::DecodeFailure;
    use serde_json::json;

    fn stop(reason: StopReason) -> DecodeEvent {
        DecodeEvent::Stop { reason }
    }

    // --- Happy paths ---

    #[test]
    fn text_only_yields_one_text_part_and_concatenated_text() {
        let turn = DecodedAssistantTurn::from_events(vec![
            DecodeEvent::TextDelta("foo ".into()),
            DecodeEvent::TextDelta("bar".into()),
            stop(StopReason::EndOfText),
        ])
        .unwrap();
        assert_eq!(turn.parts.len(), 1);
        assert_eq!(turn.parts[0], DecodedPart::Text("foo bar".into()));
        assert_eq!(turn.text(), "foo bar");
        assert_eq!(turn.tool_calls().count(), 0);
        assert_eq!(turn.stop_reason, StopReason::EndOfText);
    }

    #[test]
    fn complete_tool_call_lands_with_args_and_args_json_in_agreement() {
        let turn = DecodedAssistantTurn::from_events(vec![
            DecodeEvent::ToolCallStart {
                index: 0,
                name: "add".into(),
            },
            DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: r#"{"a":1,"b":2}"#.into(),
            },
            DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"a": 1, "b": 2}),
            },
            stop(StopReason::EndOfText),
        ])
        .unwrap();
        assert_eq!(turn.parts.len(), 1);
        let DecodedPart::ToolCall(call) = &turn.parts[0] else {
            panic!("expected ToolCall, got {:?}", turn.parts[0]);
        };
        assert_eq!(call.parser_index, 0);
        assert_eq!(call.name, "add");
        assert_eq!(call.args, json!({"a": 1, "b": 2}));
        assert_eq!(call.args_json, r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn text_then_call_then_text_yields_three_parts_in_order() {
        let turn = DecodedAssistantTurn::from_events(vec![
            DecodeEvent::TextDelta("preamble ".into()),
            DecodeEvent::ToolCallStart {
                index: 0,
                name: "lookup".into(),
            },
            DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: "{}".into(),
            },
            DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({}),
            },
            DecodeEvent::TextDelta("postamble".into()),
            stop(StopReason::EndOfText),
        ])
        .unwrap();
        assert_eq!(turn.parts.len(), 3);
        assert_eq!(turn.parts[0], DecodedPart::Text("preamble ".into()));
        assert!(matches!(turn.parts[1], DecodedPart::ToolCall(_)));
        assert_eq!(turn.parts[2], DecodedPart::Text("postamble".into()));
        // Convenience accessors:
        assert_eq!(turn.text(), "preamble postamble");
        assert_eq!(turn.tool_calls().count(), 1);
    }

    #[test]
    fn multiple_args_deltas_concatenate_into_single_args_json() {
        let turn = DecodedAssistantTurn::from_events(vec![
            DecodeEvent::ToolCallStart {
                index: 0,
                name: "add".into(),
            },
            DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: r#"{"a":"#.into(),
            },
            DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: r#"1,"b":2}"#.into(),
            },
            DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"a": 1, "b": 2}),
            },
            stop(StopReason::EndOfText),
        ])
        .unwrap();
        let DecodedPart::ToolCall(call) = &turn.parts[0] else {
            panic!()
        };
        assert_eq!(call.args_json, r#"{"a":1,"b":2}"#);
    }

    // --- Sequence-rule violations ---

    fn assert_internal_sequence(failure: &DecodeFailure, hint: &str) {
        match failure {
            DecodeFailure::InternalSequence { reason } => {
                assert!(
                    reason.contains(hint),
                    "expected reason to mention `{hint}`; got `{reason}`"
                );
            }
            other => panic!("expected InternalSequence, got {other:?}"),
        }
    }

    #[test]
    fn text_delta_while_tool_call_open_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::TextDelta("rogue".into()))
            .unwrap_err();
        assert_internal_sequence(&err, "TextDelta arrived while tool call");
    }

    #[test]
    fn tool_call_start_while_another_open_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::ToolCallStart {
                index: 1,
                name: "b".into(),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "still open");
    }

    #[test]
    fn duplicate_tool_call_index_after_close_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        acc.feed(DecodeEvent::ToolCallEnd {
            index: 0,
            args: json!({}),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "a-again".into(),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "already completed");
    }

    #[test]
    fn args_delta_for_unknown_index_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        let err = acc
            .feed(DecodeEvent::ToolCallArgsDelta {
                index: 7,
                delta: "{}".into(),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "no preceding ToolCallStart");
    }

    #[test]
    fn args_delta_for_wrong_index_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::ToolCallArgsDelta {
                index: 1,
                delta: "{}".into(),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "in-flight call is index 0");
    }

    #[test]
    fn end_for_unknown_index_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        let err = acc
            .feed(DecodeEvent::ToolCallEnd {
                index: 7,
                args: json!({}),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "no preceding ToolCallStart");
    }

    #[test]
    fn end_for_wrong_index_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::ToolCallEnd {
                index: 1,
                args: json!({}),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "in-flight call is index 0");
    }

    #[test]
    fn args_json_disagreement_with_end_args_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        acc.feed(DecodeEvent::ToolCallArgsDelta {
            index: 0,
            delta: r#"{"x":1}"#.into(),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"y": 2}),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "disagree");
    }

    #[test]
    fn args_json_invalid_json_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        acc.feed(DecodeEvent::ToolCallArgsDelta {
            index: 0,
            delta: "{not json".into(),
        })
        .unwrap();
        let err = acc
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({}),
            })
            .unwrap_err();
        assert_internal_sequence(&err, "failed to parse");
    }

    // --- Stop is terminal ---

    #[test]
    fn event_after_stop_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(stop(StopReason::EndOfText)).unwrap();
        let err = acc
            .feed(DecodeEvent::TextDelta("after stop".into()))
            .unwrap_err();
        assert_internal_sequence(&err, "after Stop");
    }

    #[test]
    fn duplicate_stop_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(stop(StopReason::EndOfText)).unwrap();
        let err = acc.feed(stop(StopReason::EndOfText)).unwrap_err();
        assert_internal_sequence(&err, "after Stop");
    }

    // --- Poison contract ---

    #[test]
    fn fatal_event_poisons_subsequent_feed_and_into_turn() {
        let mut acc = AssistantTurnAccumulator::new();
        let first = acc
            .feed(DecodeEvent::UnknownTool {
                name: "missing".into(),
                raw_args: json!({}),
            })
            .unwrap_err();
        let again = acc
            .feed(DecodeEvent::TextDelta("anything".into()))
            .unwrap_err();
        assert_eq!(first.to_string(), again.to_string());
        let into_err = acc.into_turn().unwrap_err();
        assert_eq!(first.to_string(), into_err.to_string());
    }

    #[test]
    fn parse_error_clone_preserves_variant_and_message() {
        // Regression: previously ParserError::Json cloned into
        // ParserError::Malformed. The poison contract requires
        // the same failure on every subsequent call.
        let mut acc = AssistantTurnAccumulator::new();
        let bad_json: serde_json::Error =
            serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let first = acc
            .feed(DecodeEvent::ParseError {
                sentinel: "<tool_call>",
                source: ParserError::from(bad_json),
            })
            .unwrap_err();
        let again = acc
            .feed(DecodeEvent::TextDelta("hi".into()))
            .unwrap_err();
        assert_eq!(first.to_string(), again.to_string());
    }

    #[test]
    fn unknown_tool_event_returns_decode_failure() {
        let mut acc = AssistantTurnAccumulator::new();
        let err = acc
            .feed(DecodeEvent::UnknownTool {
                name: "delete_db".into(),
                raw_args: json!({}),
            })
            .unwrap_err();
        assert!(err.to_string().contains("delete_db"));
    }

    #[test]
    fn invalid_args_event_returns_decode_failure() {
        let mut acc = AssistantTurnAccumulator::new();
        let err = acc
            .feed(DecodeEvent::InvalidArgs {
                name: "add".into(),
                args: json!({"a": 1}),
                errors: vec![SchemaError {
                    path: "/b".into(),
                    message: "is required".into(),
                }],
            })
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("add"));
        assert!(msg.contains("is required"));
    }

    // --- snapshot / into_turn preconditions ---

    #[test]
    fn snapshot_without_stop_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        let err = acc.snapshot().unwrap_err();
        assert_internal_sequence(&err, "before any Stop event");
    }

    #[test]
    fn snapshot_after_protocol_error_stop_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        acc.feed(stop(StopReason::ProtocolError)).unwrap();
        let err = acc.snapshot().unwrap_err();
        assert_internal_sequence(&err, "ProtocolError");
    }

    #[test]
    fn snapshot_with_open_call_is_internal_sequence() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::ToolCallStart {
            index: 0,
            name: "a".into(),
        })
        .unwrap();
        acc.feed(stop(StopReason::EndOfText)).unwrap();
        let err = acc.snapshot().unwrap_err();
        assert_internal_sequence(&err, "still open");
    }

    #[test]
    fn snapshot_after_clean_stop_is_ok_and_borrows() {
        let mut acc = AssistantTurnAccumulator::new();
        acc.feed(DecodeEvent::TextDelta("foo".into())).unwrap();
        acc.feed(stop(StopReason::EndOfText)).unwrap();
        // Snapshot multiple times (borrows, so doesn't consume).
        let snap1 = acc.snapshot().unwrap();
        let snap2 = acc.snapshot().unwrap();
        assert_eq!(snap1, snap2);
        assert_eq!(snap1.text(), "foo");
    }
}

#[cfg(test)]
mod proptests {
    //! Property tests for the accumulator's contract. These cover
    //! the bug class that motivated the recent correctness fixes:
    //! cross-component contracts that no single scenario test
    //! exercised. Each property maps to one of the contract rules
    //! documented at the top of the module.

    use super::*;
    use proptest::prelude::*;

    /// Strategy for any single non-Stop, non-fatal `DecodeEvent`.
    fn arb_non_terminal_event() -> impl Strategy<Value = DecodeEvent> {
        prop_oneof![
            "[a-z ]{0,16}".prop_map(DecodeEvent::TextDelta),
            (0_usize..3, "[a-z]{1,8}").prop_map(|(i, name)| DecodeEvent::ToolCallStart {
                index: i,
                name,
            }),
            (0_usize..3).prop_map(|i| DecodeEvent::ToolCallArgsDelta {
                index: i,
                delta: r#"{"x":1}"#.into(),
            }),
            (0_usize..3).prop_map(|i| DecodeEvent::ToolCallEnd {
                index: i,
                args: serde_json::json!({"x": 1}),
            }),
        ]
    }

    fn arb_fatal_event() -> impl Strategy<Value = DecodeEvent> {
        prop_oneof![
            "[a-z]{1,8}".prop_map(|name| DecodeEvent::UnknownTool {
                name,
                raw_args: serde_json::json!({}),
            }),
            "[a-z]{1,8}".prop_map(|name| DecodeEvent::InvalidArgs {
                name,
                args: serde_json::json!({}),
                errors: vec![],
            }),
        ]
    }

    proptest! {
        /// After Stop, every further event is an InternalSequence
        /// error. Holds regardless of the pre-Stop sequence.
        #[test]
        fn stop_is_terminal(
            pre_stop in prop::collection::vec(arb_non_terminal_event(), 0..10),
            after in arb_non_terminal_event(),
        ) {
            let mut acc = AssistantTurnAccumulator::new();
            for ev in pre_stop {
                let _ = acc.feed(ev);  // Invalid sequences poison; OK here.
            }
            if acc.failure().is_some() {
                return Ok(());
            }
            acc.feed(DecodeEvent::Stop { reason: StopReason::EndOfText }).unwrap();

            let err = acc.feed(after).unwrap_err();
            prop_assert!(
                matches!(&err, DecodeFailure::InternalSequence { reason } if reason.contains("after Stop")),
                "expected InternalSequence after Stop, got {err:?}"
            );
        }

        /// Once a fatal event poisons the accumulator, every
        /// subsequent feed/snapshot/into_turn returns the SAME
        /// failure. Bit-equal (Debug-formatted) — the assertion
        /// shape that catches the ParserError::Json clone-lossy bug.
        #[test]
        fn fatal_event_poison_is_sticky(
            pre in prop::collection::vec(arb_non_terminal_event(), 0..5),
            fatal in arb_fatal_event(),
            after in prop::collection::vec(arb_non_terminal_event(), 0..5),
        ) {
            let mut acc = AssistantTurnAccumulator::new();
            for ev in pre {
                let _ = acc.feed(ev);
            }
            let first = acc.feed(fatal).unwrap_err();
            for ev in after {
                let again = acc.feed(ev).unwrap_err();
                prop_assert_eq!(format!("{:?}", first), format!("{:?}", again));
            }
            let snap_err = acc.snapshot().unwrap_err();
            prop_assert_eq!(format!("{:?}", first), format!("{:?}", snap_err));
            let into_err = acc.into_turn().unwrap_err();
            prop_assert_eq!(format!("{:?}", first), format!("{:?}", into_err));
        }

        /// snapshot() before any Stop event must error. Either with
        /// "before any Stop" or with a pre-existing poison from an
        /// invalid sequence — both acceptable.
        #[test]
        fn snapshot_before_stop_always_errors(
            events in prop::collection::vec(arb_non_terminal_event(), 0..15),
        ) {
            let mut acc = AssistantTurnAccumulator::new();
            for ev in events {
                let _ = acc.feed(ev);
            }
            let _ = acc.snapshot().unwrap_err();
        }

        /// On a clean event sequence, into_turn() and snapshot()
        /// agree. They're two ways to read the same canonical state.
        #[test]
        fn into_turn_and_snapshot_agree_on_clean_streams(
            events in prop::collection::vec(arb_non_terminal_event(), 0..20),
        ) {
            let mut acc1 = AssistantTurnAccumulator::new();
            let mut acc2 = AssistantTurnAccumulator::new();
            let mut poisoned = false;
            for ev in events {
                if acc1.feed(ev.clone()).is_err() {
                    poisoned = true;
                    break;
                }
                acc2.feed(ev).unwrap();
            }
            if poisoned { return Ok(()); }
            acc1.feed(DecodeEvent::Stop { reason: StopReason::EndOfText }).unwrap();
            acc2.feed(DecodeEvent::Stop { reason: StopReason::EndOfText }).unwrap();
            // The generator can produce sequences ending with an open
            // tool call (Start without End). Both snapshot and
            // into_turn correctly reject those — skip those test
            // cases (they're covered by `snapshot_with_open_call_*`
            // scenario tests).
            let Ok(snap) = acc1.snapshot() else { return Ok(()); };
            let Ok(turn) = acc2.into_turn() else { return Ok(()); };
            prop_assert_eq!(snap, turn);
        }

        /// Multiple TextDelta events between tool calls coalesce
        /// into a single Text part. Feeding "ab"+"cd" produces the
        /// same `parts` as feeding "abcd".
        #[test]
        fn text_part_coalescing_is_associative(
            chunks in prop::collection::vec("[a-z ]{0,12}", 0..8),
        ) {
            let split = {
                let mut a = AssistantTurnAccumulator::new();
                for c in &chunks {
                    a.feed(DecodeEvent::TextDelta(c.clone())).unwrap();
                }
                a.feed(DecodeEvent::Stop { reason: StopReason::EndOfText }).unwrap();
                a.into_turn().unwrap()
            };
            let merged = {
                let mut a = AssistantTurnAccumulator::new();
                let combined: String = chunks.iter().map(String::as_str).collect();
                if !combined.is_empty() {
                    a.feed(DecodeEvent::TextDelta(combined)).unwrap();
                }
                a.feed(DecodeEvent::Stop { reason: StopReason::EndOfText }).unwrap();
                a.into_turn().unwrap()
            };
            prop_assert_eq!(split.text(), merged.text());
            prop_assert!(split.parts.len() <= 1);
            prop_assert!(merged.parts.len() <= 1);
        }
    }
}
