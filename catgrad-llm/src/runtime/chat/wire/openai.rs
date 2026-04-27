//! OpenAI chat-completions stream mapper.
//!
//! Single per-request mapper that owns OpenAI chat semantics. Consumers
//! drive it the same way regardless of whether the HTTP request asked
//! for streaming:
//!
//! ```ignore
//! let mut mapper = OpenAiStreamMapper::new(|prefix| my_id_allocator.next(prefix));
//! for event in parser_event_stream {
//!     let frames = mapper.feed(event)?;        // streaming sink writes; non-streaming discards
//! }
//! let tail = mapper.finish(parser_stop_reason)?;
//! // streaming: write `tail` frames as SSE
//! // non-streaming: ignore, then `mapper.snapshot()` for the response payload
//! ```
//!
//! All [`DecodeEvent`] terminal variants ([`DecodeEvent::UnknownTool`],
//! [`DecodeEvent::InvalidArgs`], [`DecodeEvent::ParseError`]) become
//! `Err(DecodeFailure)` from [`Self::feed`] — both surfaces (OpenAI,
//! Anthropic) treat them identically and the caller renders an error
//! frame.

use serde_json::{Value as JsonValue, json};
use std::collections::HashMap;

use super::{DecodeFailure, WireMapper};
use crate::runtime::chat::{
    AssistantTurnAccumulator, DecodeEvent, DecodedToolCall, StopReason,
};
use crate::types::openai;

/// One frame the streaming sink should write to the wire. The caller
/// wraps each frame's `delta` and optional `finish_reason` into a
/// [`openai::ChatCompletionChunk`] with its own `id` / `created` /
/// `model` / `usage` — those are response metadata and stay outside
/// the mapper.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiStreamFrame {
    pub delta: openai::ChatDelta,
    pub finish_reason: Option<openai::FinishReason>,
}

/// Buffered assistant payload for the non-streaming endpoint. The
/// caller wraps this in [`openai::ChatCompletionResponse`] (stamping
/// its own `id` / `created` / `model` / `usage`).
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiSnapshot {
    pub message: openai::ChatMessage,
    pub finish_reason: openai::FinishReason,
}

type IdMinter = Box<dyn FnMut(&str) -> String + Send>;

pub struct OpenAiStreamMapper {
    id_minter: IdMinter,
    /// Wire-neutral accumulator owns: text, tool_calls, sequence
    /// rules, terminal-event poisoning, args/args_json agreement.
    /// Mapper just adds wire IDs on top.
    accumulator: AssistantTurnAccumulator,
    /// `parser_index` → minted `call_X` wire ID. Populated on
    /// `ToolCallStart`; consulted at `snapshot` time to wrap the
    /// accumulator's `DecodedToolCall`s into OpenAI tool_call JSON.
    wire_ids: HashMap<usize, String>,
    /// Whether ANY tool call has been emitted. Drives finish_reason.
    /// Could be derived from `accumulator.snapshot().tool_calls`,
    /// but a flat bool avoids a clone in the hot path.
    saw_tool_call: bool,
    final_finish_reason: Option<openai::FinishReason>,
    /// Set by the first `finish`. Subsequent `finish` calls return
    /// `Ok(empty)`; subsequent `feed` calls also return `Ok(empty)`.
    finished: bool,
}

impl std::fmt::Debug for OpenAiStreamMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiStreamMapper")
            .field("accumulator", &self.accumulator)
            .field("wire_ids", &self.wire_ids)
            .field("saw_tool_call", &self.saw_tool_call)
            .field("final_finish_reason", &self.final_finish_reason)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl OpenAiStreamMapper {
    /// Build a mapper. `id_minter("call")` is called once per
    /// [`DecodeEvent::ToolCallStart`] — the mapper does NOT use a
    /// per-instance counter (that would let two concurrent requests
    /// hand the same `call_0` to two different clients). Wire to
    /// whatever process-wide allocator the host already exposes
    /// (e.g. `next_id` in node, the shared `NEXT_ID` atomic in
    /// `examples/serve.rs`).
    pub fn new<F>(id_minter: F) -> Self
    where
        F: FnMut(&str) -> String + Send + 'static,
    {
        Self {
            id_minter: Box::new(id_minter),
            accumulator: AssistantTurnAccumulator::new(),
            wire_ids: HashMap::new(),
            saw_tool_call: false,
            final_finish_reason: None,
            finished: false,
        }
    }

    /// Consume one [`DecodeEvent`] and update mapper state. Returns the
    /// wire frames a streaming sink should emit. Non-streaming sinks
    /// discard the return value.
    ///
    /// `Err(DecodeFailure)` is returned when `event` is one of the
    /// parser's terminal variants OR the event sequence violates the
    /// parser contract (e.g. `ToolCallEnd` for an index that was
    /// never started). The mapper is then **poisoned**: every
    /// subsequent `feed` / `finish` call returns the same failure.
    ///
    /// After [`Self::finish`] has been called once, further `feed`
    /// calls return `Ok(empty)` (the stream is over).
    pub fn feed(
        &mut self,
        event: DecodeEvent,
    ) -> Result<Vec<OpenAiStreamFrame>, DecodeFailure> {
        if let Some(failure) = self.accumulator.failure() {
            return Err(failure.clone());
        }
        if self.finished {
            return Ok(Vec::new());
        }

        // Validate via the accumulator first. If it returns Err, no
        // wire state has been mutated yet — error propagates cleanly,
        // and the accumulator's poison flag is now set so subsequent
        // calls hit the early-return above.
        self.accumulator.feed(event.clone())?;

        match event {
            DecodeEvent::TextDelta(s) => Ok(vec![OpenAiStreamFrame {
                delta: openai::ChatDelta {
                    content: Some(s),
                    ..Default::default()
                },
                finish_reason: None,
            }]),
            DecodeEvent::ToolCallStart { index, name } => {
                self.saw_tool_call = true;
                let wire_id = (self.id_minter)("call");
                self.wire_ids.insert(index, wire_id.clone());
                Ok(vec![OpenAiStreamFrame {
                    delta: openai::ChatDelta {
                        tool_calls: Some(vec![json!({
                            "index": index,
                            "id": wire_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": "",
                            },
                        })]),
                        ..Default::default()
                    },
                    finish_reason: None,
                }])
            }
            DecodeEvent::ToolCallArgsDelta { index, delta } => Ok(vec![OpenAiStreamFrame {
                delta: openai::ChatDelta {
                    tool_calls: Some(vec![json!({
                        "index": index,
                        "function": { "arguments": delta },
                    })]),
                    ..Default::default()
                },
                finish_reason: None,
            }]),
            // No frame for End: Start + ArgsDelta already carried the
            // call. Accumulator persisted the closed call.
            DecodeEvent::ToolCallEnd { .. } => Ok(Vec::new()),
            // The mapper owns the finish frame (emitted from `finish`),
            // not the parser's informational `Stop`.
            DecodeEvent::Stop { .. } => Ok(Vec::new()),
            // Terminal events: accumulator returned Err above.
            DecodeEvent::UnknownTool { .. }
            | DecodeEvent::InvalidArgs { .. }
            | DecodeEvent::ParseError { .. } => {
                unreachable!("accumulator returned Err for terminal event")
            }
        }
    }

    /// Tell the mapper the stream has ended. Returns the final wire
    /// frame (an empty delta carrying `finish_reason`) for the
    /// streaming sink. Non-streaming sinks discard the return value
    /// and read the resolved `finish_reason` from [`Self::snapshot`].
    ///
    /// One-shot: subsequent `finish` calls return `Ok(empty)`.
    /// Returns `Err` (with the same poisoned failure) if a prior
    /// `feed` poisoned the mapper, or if any tool call was started
    /// but never ended.
    pub fn finish(
        &mut self,
        parser_stop: StopReason,
    ) -> Result<Vec<OpenAiStreamFrame>, DecodeFailure> {
        if let Some(failure) = self.accumulator.failure() {
            return Err(failure.clone());
        }
        if self.finished {
            return Ok(Vec::new());
        }
        // Same postcondition as `AssistantTurnAccumulator::into_turn`
        // — but expressed without consuming, so the accumulator can
        // still serve `snapshot` after.
        self.accumulator.assert_no_open_calls()?;
        self.finished = true;
        if matches!(parser_stop, StopReason::ProtocolError) {
            // Reachable only if the caller ignored a prior `feed` Err
            // and asked us to close anyway. Emit nothing — the caller
            // should already have rendered an error frame.
            return Ok(Vec::new());
        }
        let reason = self.resolve_finish_reason(parser_stop);
        self.final_finish_reason = Some(reason);
        Ok(vec![OpenAiStreamFrame {
            delta: openai::ChatDelta::default(),
            finish_reason: Some(reason),
        }])
    }

    /// Snapshot the buffered assistant payload. The caller wraps the
    /// `message` and `finish_reason` into a
    /// [`openai::ChatCompletionResponse`] for the non-streaming
    /// endpoint.
    ///
    /// **Fallible:** returns `Err` if the mapper is poisoned or
    /// [`Self::finish`] hasn't been called yet. Calling `snapshot`
    /// before `finish` would silently return a partial-state response
    /// — the type system now forbids that.
    pub fn snapshot(&self) -> Result<OpenAiSnapshot, DecodeFailure> {
        if let Some(failure) = self.accumulator.failure() {
            return Err(failure.clone());
        }
        if !self.finished {
            return Err(DecodeFailure::internal_sequence(
                "snapshot called before finish — mapper has not been finalized",
            ));
        }
        let turn = self.accumulator.snapshot()?;
        let text = turn.text();
        // Collect-into-Result short-circuits on the first
        // missing-wire-id failure.
        let tool_calls: Vec<JsonValue> = turn
            .tool_calls()
            .map(|call| self.tool_call_to_wire_json(call))
            .collect::<Result<_, _>>()?;
        let content = if text.is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(openai::MessageContent::Text(text))
        };
        let message = openai::ChatMessage::builder()
            .role("assistant".to_string())
            .content(content)
            .tool_calls(if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            })
            .build();
        // `finish(ProtocolError)` marks `finished = true` but leaves
        // `final_finish_reason` unset — there's no well-defined
        // success terminal in that case. Surface a structured error
        // here mirroring `AnthropicStreamMapper::snapshot`'s treatment
        // of the same condition.
        let Some(finish_reason) = self.final_finish_reason else {
            return Err(DecodeFailure::internal_sequence(
                "snapshot called after finish(ProtocolError) — \
                 the response is meaningless after a fatal error",
            ));
        };
        Ok(OpenAiSnapshot {
            message,
            finish_reason,
        })
    }

    /// Wrap one accumulator-emitted [`DecodedToolCall`] into the
    /// OpenAI tool_call JSON shape, looking up the wire id we minted
    /// at `ToolCallStart` time. The accumulator's `args_json` is the
    /// concatenation of `ToolCallArgsDelta` payloads — exactly what
    /// the OpenAI wire format wants for `function.arguments` (which
    /// is a JSON-encoded string, not an object).
    ///
    /// Returns `Err(DecodeFailure)` if no wire id was minted for this
    /// `parser_index`. That's structurally impossible if the mapper's
    /// state machine matches the accumulator (id is minted on
    /// `ToolCallStart`, accumulator records the call on
    /// `ToolCallEnd`), but if it ever happens, surfacing the
    /// inconsistency as `InternalSequence` is strictly better than
    /// emitting an empty-string `tool_call.id` to the wire.
    fn tool_call_to_wire_json(
        &self,
        call: &DecodedToolCall,
    ) -> Result<JsonValue, DecodeFailure> {
        let Some(wire_id) = self.wire_ids.get(&call.parser_index) else {
            return Err(DecodeFailure::internal_sequence(format!(
                "snapshot built without a wire id for tool call \
                 parser_index={}; mapper state machine diverged from \
                 accumulator",
                call.parser_index
            )));
        };
        Ok(json!({
            "id": wire_id,
            "type": "function",
            "function": {
                "name": &call.name,
                "arguments": &call.args_json,
            },
        }))
    }

    fn resolve_finish_reason(&self, parser_stop: StopReason) -> openai::FinishReason {
        if self.saw_tool_call {
            return openai::FinishReason::ToolCalls;
        }
        match parser_stop {
            StopReason::EndOfText | StopReason::StopSequence => openai::FinishReason::Stop,
            StopReason::MaxTokens => openai::FinishReason::Length,
            // Caught by the early return above; included so this match
            // stays exhaustive without a wildcard.
            StopReason::ProtocolError => openai::FinishReason::Stop,
        }
    }
}

impl WireMapper for OpenAiStreamMapper {
    type Frame = OpenAiStreamFrame;
    type Snapshot = OpenAiSnapshot;

    fn feed(&mut self, event: DecodeEvent) -> Result<Vec<Self::Frame>, DecodeFailure> {
        OpenAiStreamMapper::feed(self, event)
    }
    fn finish(&mut self, parser_stop: StopReason) -> Result<Vec<Self::Frame>, DecodeFailure> {
        OpenAiStreamMapper::finish(self, parser_stop)
    }
    fn snapshot(&self) -> Result<Self::Snapshot, DecodeFailure> {
        OpenAiStreamMapper::snapshot(self)
    }
    fn close_for_error(&mut self) -> Vec<Self::Frame> {
        // OpenAI streaming has no content-block bracketing — error
        // frames close the stream directly. No cleanup frames needed.
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn deterministic_minter() -> impl FnMut(&str) -> String + Send + 'static {
        let mut n = 0u64;
        move |prefix: &str| {
            n += 1;
            format!("{prefix}_{n}")
        }
    }

    #[test]
    fn text_only_stream_yields_one_delta_per_event_then_finish() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let frames = mapper
            .feed(DecodeEvent::TextDelta("hello".to_string()))
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].delta.content.as_deref(), Some("hello"));
        let tail = mapper.finish(StopReason::EndOfText).unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].finish_reason, Some(openai::FinishReason::Stop));
        assert!(tail[0].delta.content.is_none());
    }

    #[test]
    fn snapshot_after_text_only_carries_full_content_and_stop() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("foo ".into())).unwrap();
        mapper.feed(DecodeEvent::TextDelta("bar".into())).unwrap();
        // Real consumers feed Stop via `parser.finish()`'s event
        // stream before calling mapper.finish(); tests do it manually.
        mapper
            .feed(DecodeEvent::Stop {
                reason: StopReason::EndOfText,
            })
            .unwrap();
        mapper.finish(StopReason::EndOfText).unwrap();
        let snap = mapper.snapshot().unwrap();
        assert_eq!(snap.finish_reason, openai::FinishReason::Stop);
        assert_eq!(
            snap.message.content,
            Some(openai::MessageContent::Text("foo bar".into()))
        );
        assert!(snap.message.tool_calls.is_none());
    }

    #[test]
    fn tool_call_stream_emits_start_args_delta_and_no_end_frame() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let start = mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "add".into(),
            })
            .unwrap();
        assert_eq!(start.len(), 1);
        let tool_call = &start[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tool_call["id"], "call_1");
        assert_eq!(tool_call["function"]["name"], "add");

        let args = mapper
            .feed(DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: r#"{"a":1,"b":2}"#.into(),
            })
            .unwrap();
        assert_eq!(args.len(), 1);
        assert_eq!(
            args[0].delta.tool_calls.as_ref().unwrap()[0]["function"]["arguments"],
            r#"{"a":1,"b":2}"#
        );

        let end = mapper
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"a": 1, "b": 2}),
            })
            .unwrap();
        assert!(end.is_empty(), "ToolCallEnd is bookkeeping-only on the wire");

        let tail = mapper.finish(StopReason::EndOfText).unwrap();
        assert_eq!(tail[0].finish_reason, Some(openai::FinishReason::ToolCalls));
    }

    #[test]
    fn snapshot_after_tool_call_carries_one_aggregated_call() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "add".into(),
            })
            .unwrap();
        mapper
            .feed(DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: r#"{"a":1,"b":2}"#.into(),
            })
            .unwrap();
        mapper
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"a": 1, "b": 2}),
            })
            .unwrap();
        mapper
            .feed(DecodeEvent::Stop {
                reason: StopReason::EndOfText,
            })
            .unwrap();
        mapper.finish(StopReason::EndOfText).unwrap();
        let snap = mapper.snapshot().unwrap();
        assert_eq!(snap.finish_reason, openai::FinishReason::ToolCalls);
        let calls = snap.message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "add");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"a":1,"b":2}"#);
        // Empty content + tool calls => no content field on the message.
        assert!(snap.message.content.is_none());
    }

    #[test]
    fn unknown_tool_event_returns_decode_failure() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let err = mapper
            .feed(DecodeEvent::UnknownTool {
                name: "missing".into(),
                raw_args: json!({"a": 1}),
            })
            .unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn invalid_args_event_returns_decode_failure() {
        use crate::runtime::chat::SchemaError;
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let err = mapper
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

    #[test]
    fn finish_after_protocol_error_emits_nothing() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let frames = mapper.finish(StopReason::ProtocolError).unwrap();
        assert!(frames.is_empty());
    }

    /// Regression: previously the snapshot's `tool_call_to_wire_json`
    /// helper fell back to `unwrap_or("")` if a parser-emitted call
    /// had no minted wire id. That path is structurally unreachable
    /// in normal flow (id is minted on `ToolCallStart`, accumulator
    /// records the call on `ToolCallEnd`), but if the mapper state
    /// machine ever diverges from the accumulator, emitting an empty
    /// `tool_call.id` to the wire is the worst possible outcome.
    /// Now it returns `InternalSequence`.
    #[test]
    fn tool_call_to_wire_json_missing_id_is_internal_sequence() {
        let mapper = OpenAiStreamMapper::new(deterministic_minter());
        let call = DecodedToolCall {
            parser_index: 42,
            name: "phantom".into(),
            args: json!({}),
            args_json: "{}".into(),
        };
        let err = mapper.tool_call_to_wire_json(&call).unwrap_err();
        match err {
            DecodeFailure::InternalSequence { ref reason } => {
                assert!(reason.contains("parser_index=42"));
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected InternalSequence, got {other:?}"),
        }
    }

    /// Regression: previously `snapshot` after `finish(ProtocolError)`
    /// panicked because `final_finish_reason` is None on that path
    /// but the snapshot did `expect()` on it. Now it must return a
    /// structured `InternalSequence` failure (matching Anthropic).
    ///
    /// The misuse pattern that triggered the panic: consumer fed all
    /// events through to a clean Stop (so the accumulator is happy),
    /// THEN called `finish(ProtocolError)` (so the mapper's
    /// `final_finish_reason` was never set), THEN called `snapshot`.
    #[test]
    fn snapshot_after_finish_protocol_error_is_internal_sequence_error() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        mapper
            .feed(DecodeEvent::Stop {
                reason: StopReason::EndOfText,
            })
            .unwrap();
        mapper.finish(StopReason::ProtocolError).unwrap();
        let err = mapper.snapshot().unwrap_err();
        match err {
            DecodeFailure::InternalSequence { ref reason } => {
                assert!(
                    reason.contains("ProtocolError"),
                    "expected reason to mention ProtocolError, got `{reason}`"
                );
            }
            other => panic!("expected InternalSequence, got {other:?}"),
        }
    }

    #[test]
    fn max_tokens_stop_reason_maps_to_length() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("partial".into())).unwrap();
        mapper
            .feed(DecodeEvent::Stop {
                reason: StopReason::MaxTokens,
            })
            .unwrap();
        let tail = mapper.finish(StopReason::MaxTokens).unwrap();
        assert_eq!(tail[0].finish_reason, Some(openai::FinishReason::Length));
        assert_eq!(
            mapper.snapshot().unwrap().finish_reason,
            openai::FinishReason::Length
        );
    }

    // --- Poison + idempotency + sequence-error tests (P3 corrections) ---

    #[test]
    fn fatal_event_poisons_subsequent_feed_and_finish() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let first = mapper
            .feed(DecodeEvent::UnknownTool {
                name: "missing".into(),
                raw_args: json!({}),
            })
            .unwrap_err();
        let again = mapper
            .feed(DecodeEvent::TextDelta("anything".into()))
            .unwrap_err();
        assert_eq!(first.to_string(), again.to_string());
        let finish_err = mapper.finish(StopReason::EndOfText).unwrap_err();
        assert_eq!(first.to_string(), finish_err.to_string());
    }

    #[test]
    fn finish_is_idempotent() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        let first = mapper.finish(StopReason::EndOfText).unwrap();
        assert_eq!(first.len(), 1);
        let second = mapper.finish(StopReason::EndOfText).unwrap();
        assert!(second.is_empty(), "second finish must emit no frames");
        let third = mapper
            .feed(DecodeEvent::TextDelta("after".into()))
            .unwrap();
        assert!(third.is_empty(), "feed after finish must be a no-op");
    }

    #[test]
    fn tool_call_args_delta_for_unknown_index_is_internal_sequence_error() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let err = mapper
            .feed(DecodeEvent::ToolCallArgsDelta {
                index: 7,
                delta: "{}".into(),
            })
            .unwrap_err();
        assert!(
            matches!(err, DecodeFailure::InternalSequence { .. }),
            "expected InternalSequence, got {:?}",
            err
        );
        // Subsequent calls return the same poisoned failure.
        let again = mapper.feed(DecodeEvent::TextDelta("x".into())).unwrap_err();
        assert_eq!(err.to_string(), again.to_string());
    }

    #[test]
    fn tool_call_end_for_unknown_index_is_internal_sequence_error() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        let err = mapper
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({}),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            DecodeFailure::InternalSequence { .. }
        ));
    }

    #[test]
    fn duplicate_tool_call_start_is_internal_sequence_error() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "a".into(),
            })
            .unwrap();
        let err = mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "b".into(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            DecodeFailure::InternalSequence { .. }
        ));
    }

    #[test]
    fn finish_with_open_tool_call_is_internal_sequence_error() {
        let mut mapper = OpenAiStreamMapper::new(deterministic_minter());
        mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "a".into(),
            })
            .unwrap();
        let err = mapper.finish(StopReason::EndOfText).unwrap_err();
        assert!(matches!(
            err,
            DecodeFailure::InternalSequence { .. }
        ));
    }

    #[test]
    fn id_minter_is_called_per_tool_call_start() {
        // Two complete tool-call triples in sequence — confirms the
        // mapper invokes the injected minter once per ToolCallStart.
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let calls_inner = std::sync::Arc::clone(&calls);
        let mut mapper = OpenAiStreamMapper::new(move |prefix: &str| {
            calls_inner.lock().unwrap().push(prefix.to_string());
            format!("{prefix}_x")
        });
        for index in [0_usize, 1] {
            mapper
                .feed(DecodeEvent::ToolCallStart {
                    index,
                    name: "tool".into(),
                })
                .unwrap();
            mapper
                .feed(DecodeEvent::ToolCallEnd {
                    index,
                    args: json!({}),
                })
                .unwrap();
        }
        let prefixes = calls.lock().unwrap().clone();
        assert_eq!(prefixes, vec!["call".to_string(), "call".to_string()]);
    }
}
