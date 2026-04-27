//! Anthropic Messages API stream mapper.
//!
//! Layered over [`AssistantTurnAccumulator`]: the accumulator owns
//! sequence-rule enforcement, terminal-event poisoning, and the
//! ordered `Vec<DecodedPart>` of text runs / tool calls. The mapper
//! adds:
//!
//! - tool-call wire IDs (`toolu_X`),
//! - per-block-emission `block_index` allocation and tracking of
//!   which content block is currently open on the wire,
//! - mapping of [`DecodedPart`]s to Anthropic [`anthropic::ContentBlock`]s
//!   at snapshot time.
//!
//! There is no parallel ordered-block state — the accumulator's
//! `parts` is the single source of order. Streaming `feed` emits
//! `content_block_*` frames as parts open / close; non-streaming
//! `snapshot` derives the final block list from the same parts.
//!
//! Caller still owns:
//!
//! - `message_start` (response id, role, model, input-tokens usage).
//! - `message_stop`.
//! - The `message_delta` envelope shape — the mapper's terminal
//!   `Stop` frame carries the resolved `stop_reason`; the caller
//!   stamps its own `output_tokens` usage.
//! - HTTP transport / SSE framing.

use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::HashMap;

use super::{DecodeFailure, WireMapper};
use crate::runtime::chat::{AssistantTurnAccumulator, DecodeEvent, DecodedPart, StopReason};
use crate::types::anthropic;

/// One frame the streaming sink should write. The mapper produces
/// content-block-level events plus the terminal stop signal;
/// `message_start`, `message_stop`, and the `message_delta` envelope
/// are caller-owned (response id / model / usage live there).
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicStreamFrame {
    BlockStart {
        index: u32,
        block: anthropic::ContentBlock,
    },
    BlockDelta {
        index: u32,
        delta: anthropic::ContentBlockDelta,
    },
    BlockStop {
        index: u32,
    },
    /// Resolved final stop reason. The caller wraps this in a
    /// `message_delta` event with its own `output_tokens` usage.
    Stop(anthropic::StopReason),
}

/// Buffered assistant payload for the non-streaming endpoint. The
/// caller wraps this in [`anthropic::MessageResponse`] with its own
/// `id` / `model` / `usage`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSnapshot {
    pub blocks: Vec<anthropic::ContentBlock>,
    pub stop_reason: anthropic::StopReason,
}

type IdMinter = Box<dyn FnMut(&str) -> String + Send>;

pub struct AnthropicStreamMapper {
    id_minter: IdMinter,
    accumulator: AssistantTurnAccumulator,
    next_block_index: u32,
    /// Block index of the currently-open wire block, plus what kind
    /// it is (text or tool_use). `None` means no block is open.
    /// We don't need to remember the *contents* — those live in the
    /// accumulator and are reconstructed at snapshot time.
    open_block: Option<OpenBlock>,
    /// Per parser-tool-call: the wire id (`toolu_X`) we minted at
    /// `ToolCallStart` time, indexed by the parser's tool-call
    /// `index`. Used both during streaming (BlockStart for tool_use
    /// carries the id) and at snapshot time (each tool_use content
    /// block must carry the same id).
    wire_ids: HashMap<usize, String>,
    /// Whether ANY tool call has been emitted. Drives `stop_reason`.
    saw_tool_call: bool,
    /// Set by `finish`. `snapshot` requires this is `Some`.
    final_stop_reason: Option<anthropic::StopReason>,
    finished: bool,
}

#[derive(Debug, Clone, Copy)]
struct OpenBlock {
    index: u32,
    kind: BlockKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Tool,
}

impl std::fmt::Debug for AnthropicStreamMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicStreamMapper")
            .field("accumulator", &self.accumulator)
            .field("next_block_index", &self.next_block_index)
            .field("open_block", &self.open_block)
            .field("wire_ids", &self.wire_ids)
            .field("saw_tool_call", &self.saw_tool_call)
            .field("final_stop_reason", &self.final_stop_reason)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl AnthropicStreamMapper {
    /// Build a mapper. `id_minter("toolu")` is called once per
    /// [`DecodeEvent::ToolCallStart`] to produce the `id` of the
    /// `tool_use` content block. Wire to a process-wide allocator —
    /// see [`OpenAiStreamMapper::new`](super::openai::OpenAiStreamMapper::new)
    /// for the rationale.
    pub fn new<F>(id_minter: F) -> Self
    where
        F: FnMut(&str) -> String + Send + 'static,
    {
        Self {
            id_minter: Box::new(id_minter),
            accumulator: AssistantTurnAccumulator::new(),
            next_block_index: 0,
            open_block: None,
            wire_ids: HashMap::new(),
            saw_tool_call: false,
            final_stop_reason: None,
            finished: false,
        }
    }

    pub fn feed(
        &mut self,
        event: DecodeEvent,
    ) -> Result<Vec<AnthropicStreamFrame>, DecodeFailure> {
        if let Some(failure) = self.accumulator.failure() {
            return Err(failure.clone());
        }
        if self.finished {
            return Ok(Vec::new());
        }

        // Validate via the accumulator first. After this returns Ok,
        // the wire-level bookkeeping below can assume a clean state
        // (no TextDelta-while-tool-open, no duplicate Start, etc.).
        self.accumulator.feed(event.clone())?;

        match event {
            DecodeEvent::TextDelta(s) => {
                let mut frames = Vec::new();
                let block_index = self.ensure_text_block_open(&mut frames)?;
                frames.push(AnthropicStreamFrame::BlockDelta {
                    index: block_index,
                    delta: anthropic::ContentBlockDelta::TextDelta { text: s },
                });
                Ok(frames)
            }
            DecodeEvent::ToolCallStart {
                index: parser_index,
                name,
            } => {
                self.saw_tool_call = true;
                let mut frames = Vec::new();
                // Close any open text block before opening the tool
                // block. Accumulator guarantees no other tool block
                // is open here (atomicity rule).
                self.close_open_block_into(&mut frames);
                let block_index = self.allocate_index();
                let wire_id = (self.id_minter)("toolu");
                self.wire_ids.insert(parser_index, wire_id.clone());
                self.open_block = Some(OpenBlock {
                    index: block_index,
                    kind: BlockKind::Tool,
                });
                frames.push(AnthropicStreamFrame::BlockStart {
                    index: block_index,
                    block: anthropic::ContentBlock::ToolUse {
                        id: wire_id,
                        name,
                        input: JsonValue::Object(JsonMap::new()),
                    },
                });
                Ok(frames)
            }
            DecodeEvent::ToolCallArgsDelta { delta, .. } => {
                let block_index = self
                    .open_block
                    .expect("accumulator validated tool call is in flight")
                    .index;
                Ok(vec![AnthropicStreamFrame::BlockDelta {
                    index: block_index,
                    delta: anthropic::ContentBlockDelta::InputJsonDelta {
                        partial_json: delta,
                    },
                }])
            }
            DecodeEvent::ToolCallEnd { .. } => {
                let block_index = self
                    .open_block
                    .take()
                    .expect("accumulator validated tool call is in flight")
                    .index;
                Ok(vec![AnthropicStreamFrame::BlockStop { index: block_index }])
            }
            DecodeEvent::Stop { .. } => Ok(Vec::new()),
            DecodeEvent::UnknownTool { .. }
            | DecodeEvent::InvalidArgs { .. }
            | DecodeEvent::ParseError { .. } => {
                unreachable!("accumulator returned Err for terminal event")
            }
        }
    }

    /// Close any open text block, then emit the terminal `Stop` frame
    /// carrying the resolved `stop_reason`. Caller wraps the stop in
    /// a `message_delta` envelope with its own usage.
    ///
    /// One-shot: subsequent `finish` calls return `Ok(empty)`.
    /// Returns `Err` (with the same poisoned failure) if a prior
    /// `feed` poisoned the mapper, or if a tool call is still
    /// in flight at finish time.
    pub fn finish(
        &mut self,
        parser_stop: StopReason,
    ) -> Result<Vec<AnthropicStreamFrame>, DecodeFailure> {
        if let Some(failure) = self.accumulator.failure() {
            return Err(failure.clone());
        }
        if self.finished {
            return Ok(Vec::new());
        }
        self.accumulator.assert_no_open_calls()?;
        self.finished = true;

        let mut frames = Vec::new();
        // A text block may still be open here — close it for wire
        // bracketing. Tool block open is impossible (assert above).
        self.close_open_block_into(&mut frames);

        if matches!(parser_stop, StopReason::ProtocolError) {
            // Caller already emitted an error frame. We've supplied
            // any required block-close frames; no terminal Stop.
            // We deliberately don't set `final_stop_reason` here —
            // `snapshot` would reject this state anyway.
            return Ok(frames);
        }
        let stop_reason = self.resolve_stop_reason(parser_stop);
        self.final_stop_reason = Some(stop_reason);
        frames.push(AnthropicStreamFrame::Stop(stop_reason));
        Ok(frames)
    }

    /// Snapshot the buffered assistant payload. The caller wraps the
    /// `blocks` and `stop_reason` into [`anthropic::MessageResponse`]
    /// for the non-streaming endpoint.
    ///
    /// **Fallible:** returns `Err` if the mapper is poisoned or
    /// [`Self::finish`] hasn't been called yet (or was called with
    /// `ProtocolError`, which leaves `final_stop_reason` unset).
    pub fn snapshot(&self) -> Result<AnthropicSnapshot, DecodeFailure> {
        if let Some(failure) = self.accumulator.failure() {
            return Err(failure.clone());
        }
        if !self.finished {
            return Err(DecodeFailure::internal_sequence(
                "snapshot called before finish — mapper has not been finalized",
            ));
        }
        let Some(stop_reason) = self.final_stop_reason else {
            return Err(DecodeFailure::internal_sequence(
                "snapshot called after finish(ProtocolError) — \
                 the response is meaningless after a fatal error",
            ));
        };
        let turn = self.accumulator.snapshot()?;
        // Collect-into-Result short-circuits on the first
        // missing-wire-id failure.
        let mut blocks: Vec<anthropic::ContentBlock> = turn
            .parts
            .iter()
            .map(|part| self.part_to_block(part))
            .collect::<Result<_, _>>()?;
        // Anthropic clients reject responses with zero content blocks.
        // Emit a placeholder empty text block when the model produced
        // nothing user-visible.
        if blocks.is_empty() {
            blocks.push(anthropic::ContentBlock::Text {
                text: String::new(),
            });
        }
        Ok(AnthropicSnapshot {
            blocks,
            stop_reason,
        })
    }

    /// Cleanup-only frame emission for the error path. If a content
    /// block is open on the wire, emit its `BlockStop` so the
    /// `error` event arrives in a well-bracketed stream. Idempotent.
    /// Does NOT emit a terminal `Stop` — that's `finish`'s job.
    pub fn close_for_error(&mut self) -> Vec<AnthropicStreamFrame> {
        let mut frames = Vec::new();
        self.close_open_block_into(&mut frames);
        frames
    }

    /// Build one [`anthropic::ContentBlock`] from a decoded part.
    /// Returns `Err(DecodeFailure)` if a `ToolCall` part has no wire
    /// id minted for it — same rationale as OpenAI's
    /// [`OpenAiStreamMapper::tool_call_to_wire_json`]: structurally
    /// impossible if the mapper's state machine matches the
    /// accumulator, but a structured failure is strictly better than
    /// emitting an empty `tool_use.id` to the wire.
    fn part_to_block(
        &self,
        part: &DecodedPart,
    ) -> Result<anthropic::ContentBlock, DecodeFailure> {
        match part {
            DecodedPart::Text(s) => Ok(anthropic::ContentBlock::Text { text: s.clone() }),
            DecodedPart::ToolCall(call) => {
                let Some(wire_id) = self.wire_ids.get(&call.parser_index) else {
                    return Err(DecodeFailure::internal_sequence(format!(
                        "snapshot built without a wire id for tool call \
                         parser_index={}; mapper state machine diverged \
                         from accumulator",
                        call.parser_index
                    )));
                };
                Ok(anthropic::ContentBlock::ToolUse {
                    id: wire_id.clone(),
                    name: call.name.clone(),
                    input: call.args.clone(),
                })
            }
        }
    }

    /// Open a fresh text block (allocating its index) if none is
    /// open. If a text block IS already open, return its index.
    /// Returns `Err(InternalSequence)` if a tool block is open —
    /// the accumulator should already have rejected the inbound
    /// `TextDelta`, so this is defense-in-depth for mapper-state
    /// divergence (no panic).
    fn ensure_text_block_open(
        &mut self,
        frames: &mut Vec<AnthropicStreamFrame>,
    ) -> Result<u32, DecodeFailure> {
        match self.open_block {
            Some(OpenBlock {
                kind: BlockKind::Text,
                index,
            }) => Ok(index),
            Some(OpenBlock {
                kind: BlockKind::Tool,
                index,
            }) => Err(DecodeFailure::internal_sequence(format!(
                "ensure_text_block_open found tool_use block {index} open; \
                 mapper state machine diverged from accumulator (which \
                 guarantees TextDelta cannot arrive while tool_use is open)"
            ))),
            None => {
                let new_index = self.allocate_index();
                self.open_block = Some(OpenBlock {
                    index: new_index,
                    kind: BlockKind::Text,
                });
                frames.push(AnthropicStreamFrame::BlockStart {
                    index: new_index,
                    block: anthropic::ContentBlock::Text {
                        text: String::new(),
                    },
                });
                Ok(new_index)
            }
        }
    }

    /// Emit `BlockStop` for the currently-open block (text or tool)
    /// and clear the open-block state. No-op if nothing is open.
    /// Used by `feed(ToolCallStart)` (text → tool transition),
    /// `finish` (final close), and `close_for_error`.
    fn close_open_block_into(&mut self, frames: &mut Vec<AnthropicStreamFrame>) {
        if let Some(open) = self.open_block.take() {
            frames.push(AnthropicStreamFrame::BlockStop { index: open.index });
        }
    }

    fn allocate_index(&mut self) -> u32 {
        let i = self.next_block_index;
        self.next_block_index += 1;
        i
    }

    fn resolve_stop_reason(&self, parser_stop: StopReason) -> anthropic::StopReason {
        if self.saw_tool_call {
            return anthropic::StopReason::ToolUse;
        }
        match parser_stop {
            StopReason::EndOfText | StopReason::StopSequence => anthropic::StopReason::EndTurn,
            StopReason::MaxTokens => anthropic::StopReason::MaxTokens,
            StopReason::ProtocolError => anthropic::StopReason::EndTurn,
        }
    }
}

impl WireMapper for AnthropicStreamMapper {
    type Frame = AnthropicStreamFrame;
    type Snapshot = AnthropicSnapshot;

    fn feed(&mut self, event: DecodeEvent) -> Result<Vec<Self::Frame>, DecodeFailure> {
        AnthropicStreamMapper::feed(self, event)
    }
    fn finish(&mut self, parser_stop: StopReason) -> Result<Vec<Self::Frame>, DecodeFailure> {
        AnthropicStreamMapper::finish(self, parser_stop)
    }
    fn snapshot(&self) -> Result<Self::Snapshot, DecodeFailure> {
        AnthropicStreamMapper::snapshot(self)
    }
    fn close_for_error(&mut self) -> Vec<Self::Frame> {
        AnthropicStreamMapper::close_for_error(self)
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

    fn stop(reason: StopReason) -> DecodeEvent {
        DecodeEvent::Stop { reason }
    }

    // --- Streaming wire-frame shape ---

    #[test]
    fn text_only_stream_brackets_one_text_block() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        let f1 = mapper
            .feed(DecodeEvent::TextDelta("hello".into()))
            .unwrap();
        assert_eq!(f1.len(), 2);
        assert!(matches!(
            f1[0],
            AnthropicStreamFrame::BlockStart {
                index: 0,
                block: anthropic::ContentBlock::Text { .. }
            }
        ));
        assert!(matches!(
            &f1[1],
            AnthropicStreamFrame::BlockDelta {
                index: 0,
                delta: anthropic::ContentBlockDelta::TextDelta { text }
            } if text == "hello"
        ));
        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        let tail = mapper.finish(StopReason::EndOfText).unwrap();
        assert_eq!(tail.len(), 2);
        assert!(matches!(
            tail[0],
            AnthropicStreamFrame::BlockStop { index: 0 }
        ));
        assert!(matches!(
            tail[1],
            AnthropicStreamFrame::Stop(anthropic::StopReason::EndTurn)
        ));
    }

    #[test]
    fn tool_call_stream_brackets_tool_use_block_with_minted_id() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        let f1 = mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "add".into(),
            })
            .unwrap();
        assert_eq!(f1.len(), 1);
        let AnthropicStreamFrame::BlockStart {
            index: 0,
            block: anthropic::ContentBlock::ToolUse { id, name, input },
        } = &f1[0]
        else {
            panic!("expected tool_use BlockStart, got {:?}", f1[0])
        };
        assert_eq!(id, "toolu_1");
        assert_eq!(name, "add");
        assert_eq!(input, &json!({}));

        let f2 = mapper
            .feed(DecodeEvent::ToolCallArgsDelta {
                index: 0,
                delta: r#"{"a":1,"b":2}"#.into(),
            })
            .unwrap();
        assert!(matches!(
            &f2[0],
            AnthropicStreamFrame::BlockDelta {
                index: 0,
                delta: anthropic::ContentBlockDelta::InputJsonDelta { partial_json }
            } if partial_json == r#"{"a":1,"b":2}"#
        ));

        let f3 = mapper
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"a": 1, "b": 2}),
            })
            .unwrap();
        assert!(matches!(
            f3[0],
            AnthropicStreamFrame::BlockStop { index: 0 }
        ));

        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        let tail = mapper.finish(StopReason::EndOfText).unwrap();
        // No open blocks left → just the terminal Stop.
        assert_eq!(tail.len(), 1);
        assert!(matches!(
            tail[0],
            AnthropicStreamFrame::Stop(anthropic::StopReason::ToolUse)
        ));
    }

    #[test]
    fn text_then_tool_then_text_brackets_three_blocks_in_order() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("preamble ".into())).unwrap();
        mapper
            .feed(DecodeEvent::ToolCallStart {
                index: 0,
                name: "add".into(),
            })
            .unwrap();
        mapper
            .feed(DecodeEvent::ToolCallEnd {
                index: 0,
                args: json!({"a": 1}),
            })
            .unwrap();
        mapper.feed(DecodeEvent::TextDelta("postamble".into())).unwrap();
        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        mapper.finish(StopReason::EndOfText).unwrap();
        let snap = mapper.snapshot().unwrap();
        assert_eq!(snap.stop_reason, anthropic::StopReason::ToolUse);
        assert_eq!(snap.blocks.len(), 3);
        assert!(matches!(
            &snap.blocks[0],
            anthropic::ContentBlock::Text { text } if text == "preamble "
        ));
        assert!(matches!(
            &snap.blocks[1],
            anthropic::ContentBlock::ToolUse { id, name, input }
                if id == "toolu_1" && name == "add" && input == &json!({"a": 1})
        ));
        assert!(matches!(
            &snap.blocks[2],
            anthropic::ContentBlock::Text { text } if text == "postamble"
        ));
    }

    // --- Snapshot is fallible ---

    #[test]
    fn snapshot_before_finish_is_internal_sequence_error() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        // No mapper.finish() → snapshot rejects.
        let err = mapper.snapshot().unwrap_err();
        assert!(err.to_string().contains("not been finalized"));
    }

    #[test]
    fn snapshot_after_finish_protocol_error_is_internal_sequence_error() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        // Skip Stop event; call finish(ProtocolError) for cleanup.
        mapper.finish(StopReason::ProtocolError).unwrap();
        let err = mapper.snapshot().unwrap_err();
        assert!(err.to_string().contains("ProtocolError"));
    }

    #[test]
    fn snapshot_after_clean_finish_emits_blocks_and_stop() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        mapper.finish(StopReason::EndOfText).unwrap();
        let snap = mapper.snapshot().unwrap();
        assert_eq!(snap.stop_reason, anthropic::StopReason::EndTurn);
        assert_eq!(
            snap.blocks,
            vec![anthropic::ContentBlock::Text { text: "hi".into() }]
        );
    }

    #[test]
    fn empty_stream_snapshot_emits_one_empty_text_block() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        mapper.finish(StopReason::EndOfText).unwrap();
        let snap = mapper.snapshot().unwrap();
        assert_eq!(
            snap.blocks,
            vec![anthropic::ContentBlock::Text {
                text: String::new(),
            }],
            "Anthropic clients reject zero-block content"
        );
    }

    // --- close_for_error: terminal cleanup before error frame ---

    #[test]
    fn close_for_error_emits_block_stop_for_open_text_block() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper
            .feed(DecodeEvent::TextDelta("partial".into()))
            .unwrap();
        // Simulate a fatal event mid-stream; mapper is now in a state
        // where caller would render an error frame. Cleanup is needed.
        let cleanup = mapper.close_for_error();
        assert_eq!(cleanup.len(), 1);
        assert!(matches!(
            cleanup[0],
            AnthropicStreamFrame::BlockStop { index: 0 }
        ));
    }

    #[test]
    fn close_for_error_is_idempotent() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper
            .feed(DecodeEvent::TextDelta("partial".into()))
            .unwrap();
        let _ = mapper.close_for_error();
        let second = mapper.close_for_error();
        assert!(second.is_empty());
    }

    /// Regression: previously the snapshot's `part_to_block` helper
    /// fell back to `unwrap_or_default()` (empty string) for a
    /// missing wire id. Same shape as the OpenAI sibling test —
    /// structurally unreachable in normal flow, but if the mapper
    /// diverges from the accumulator we want a structured error,
    /// not a `tool_use.id = ""` on the wire.
    #[test]
    fn part_to_block_missing_tool_id_is_internal_sequence() {
        use crate::runtime::chat::DecodedToolCall;
        let mapper = AnthropicStreamMapper::new(deterministic_minter());
        let part = DecodedPart::ToolCall(DecodedToolCall {
            parser_index: 42,
            name: "phantom".into(),
            args: json!({}),
            args_json: "{}".into(),
        });
        let err = mapper.part_to_block(&part).unwrap_err();
        match err {
            DecodeFailure::InternalSequence { ref reason } => {
                assert!(reason.contains("parser_index=42"));
                assert!(reason.contains("diverged"));
            }
            other => panic!("expected InternalSequence, got {other:?}"),
        }
    }

    #[test]
    fn close_for_error_with_no_open_block_is_empty() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        let cleanup = mapper.close_for_error();
        assert!(cleanup.is_empty());
    }

    // --- Poison + idempotency ---

    #[test]
    fn fatal_event_poisons_subsequent_feed_and_finish() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
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
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("hi".into())).unwrap();
        mapper.feed(stop(StopReason::EndOfText)).unwrap();
        let first = mapper.finish(StopReason::EndOfText).unwrap();
        assert_eq!(first.len(), 2); // BlockStop + Stop
        let second = mapper.finish(StopReason::EndOfText).unwrap();
        assert!(second.is_empty());
        let after = mapper
            .feed(DecodeEvent::TextDelta("after".into()))
            .unwrap();
        assert!(after.is_empty());
    }

    #[test]
    fn max_tokens_stop_reason_maps_to_max_tokens() {
        let mut mapper = AnthropicStreamMapper::new(deterministic_minter());
        mapper.feed(DecodeEvent::TextDelta("partial".into())).unwrap();
        mapper.feed(stop(StopReason::MaxTokens)).unwrap();
        let tail = mapper.finish(StopReason::MaxTokens).unwrap();
        assert!(matches!(
            tail[1],
            AnthropicStreamFrame::Stop(anthropic::StopReason::MaxTokens)
        ));
        let snap = mapper.snapshot().unwrap();
        assert_eq!(snap.stop_reason, anthropic::StopReason::MaxTokens);
    }
}
