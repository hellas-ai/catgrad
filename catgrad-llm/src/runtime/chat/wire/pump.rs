//! Generic helpers that drive parser events through a [`WireMapper`]
//! and return the resulting frames as a `Vec`.
//!
//! Replaces the
//! ```ignore
//! for event in parser.feed(text) {
//!     match mapper.feed(event) {
//!         Ok(frames) => for f in frames { sink(f)?; },
//!         Err(failure) => return Err(failure),
//!     }
//! }
//! ```
//! pattern that consumer code used to write 8 times (4 sites × 2
//! streaming/non-streaming branches). With these helpers each site
//! becomes a single `let frames = pump_text(...)?;` plus a tiny `for
//! frame in frames { ... }` loop that writes whatever wire envelope
//! the surface needs.
//!
//! The returned-`Vec` shape (rather than a callback) is deliberate so
//! the helpers compose with both sync sinks (e.g. `tiny_http` SSE
//! channels) AND async-stream generator macros (e.g. axum's
//! `stream!`, where `yield` only works at the top level of the
//! generator body, not inside an arbitrary closure).

use super::DecodeFailure;
use crate::runtime::chat::{DecodeEvent, IncrementalToolCallParser, StopReason};

/// Failure shape returned by [`pump_text`] / [`pump_finish`].
///
/// Carries the structured `failure` AND the wire-bracketing
/// `cleanup` frames the caller MUST emit before sending its error
/// frame, so each per-surface call site doesn't have to remember to
/// invoke [`WireMapper::close_for_error`] separately. The pump
/// already drained those frames from the mapper — calling
/// `mapper.close_for_error()` again would return an empty Vec.
///
/// `cleanup` is `Vec<F>` (parameterized by the mapper's frame type).
/// For OpenAI it's always empty; for Anthropic it carries any
/// `BlockStop` frames needed to bracket an open content block. The
/// pump emits the `cleanup` frames in the same order the mapper
/// produced them — caller writes them to the wire as-is, then
/// renders its surface-specific error frame.
#[derive(Debug)]
pub struct PumpError<F> {
    pub failure: DecodeFailure,
    pub cleanup: Vec<F>,
}

impl<F> std::fmt::Display for PumpError<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.failure, f)
    }
}

impl<F: std::fmt::Debug> std::error::Error for PumpError<F> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.failure)
    }
}

/// Surface-specific mapper that consumes [`DecodeEvent`]s and emits
/// per-event wire frames plus a buffered snapshot.
///
/// Implemented by [`super::openai::OpenAiStreamMapper`] and
/// [`super::anthropic::AnthropicStreamMapper`]. The
/// [`pump_text`] / [`pump_finish`] free functions are generic over
/// this trait so consumers don't have to repeat the
/// "drain parser → drive mapper → collect frames" loop on every
/// surface and every (streaming, non-streaming) branch.
pub trait WireMapper {
    /// Per-event wire frame. Streaming sinks serialize each one; non-
    /// streaming sinks discard them and read [`Self::snapshot`] at
    /// the end.
    type Frame;
    /// Buffered assistant payload — what a non-streaming HTTP response
    /// envelope wraps. Caller stamps response id / model / usage on
    /// top of this.
    type Snapshot;

    fn feed(&mut self, event: DecodeEvent) -> Result<Vec<Self::Frame>, DecodeFailure>;
    fn finish(&mut self, parser_stop: StopReason) -> Result<Vec<Self::Frame>, DecodeFailure>;
    /// **Fallible.** Returns `Err` if the mapper is poisoned or
    /// `finish` hasn't been called yet — surfaces the same
    /// `DecodeFailure` consumers handle on `feed` / `finish`.
    fn snapshot(&self) -> Result<Self::Snapshot, DecodeFailure>;

    /// Emit any cleanup frames required to leave the wire in a
    /// well-bracketed state before sending an error frame. For
    /// surfaces with no bracketing (e.g. OpenAI), this is a no-op.
    /// Idempotent — second call returns empty. **Does not** emit
    /// terminal frames (no Stop / finish_reason); use `finish` for
    /// those.
    fn close_for_error(&mut self) -> Vec<Self::Frame>;
}

/// Drive one text chunk through the parser, then route every event
/// through the mapper, returning the collected wire frames.
///
/// The returned Vec is empty for a chunk that produces no events
/// (e.g. text accumulating inside a sentinel). On the first
/// `Err(DecodeFailure)` from the mapper, the pump:
/// 1. captures the failure,
/// 2. calls [`WireMapper::close_for_error`] to drain any cleanup
///    frames the surface needs to keep its wire well-bracketed,
/// 3. returns `Err(PumpError { failure, cleanup })`.
///
/// The caller emits `cleanup` to the wire (in order) **before**
/// rendering its surface-specific error frame, then closes the
/// stream. Do **not** call `pump_finish` after a failure — terminal
/// handling is synchronous with the failure.
pub fn pump_text<M>(
    parser: &mut dyn IncrementalToolCallParser,
    mapper: &mut M,
    text: &str,
) -> Result<Vec<M::Frame>, PumpError<M::Frame>>
where
    M: WireMapper,
{
    let mut frames = Vec::new();
    for event in parser.feed(text) {
        match mapper.feed(event) {
            Ok(more) => frames.extend(more),
            Err(failure) => {
                let cleanup = mapper.close_for_error();
                return Err(PumpError { failure, cleanup });
            }
        }
    }
    Ok(frames)
}

/// Drain the parser's `finish` events through the mapper, then call
/// `mapper.finish` and append its terminal frame(s). Returns the
/// combined collected frames, or [`PumpError`] (with cleanup frames
/// already drained) on the first failure.
pub fn pump_finish<M>(
    parser: &mut dyn IncrementalToolCallParser,
    mapper: &mut M,
    parser_stop: StopReason,
) -> Result<Vec<M::Frame>, PumpError<M::Frame>>
where
    M: WireMapper,
{
    let mut frames = Vec::new();
    for event in parser.finish(parser_stop) {
        match mapper.feed(event) {
            Ok(more) => frames.extend(more),
            Err(failure) => {
                let cleanup = mapper.close_for_error();
                return Err(PumpError { failure, cleanup });
            }
        }
    }
    match mapper.finish(parser_stop) {
        Ok(tail) => frames.extend(tail),
        Err(failure) => {
            let cleanup = mapper.close_for_error();
            return Err(PumpError { failure, cleanup });
        }
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    //! End-to-end integration: a real `IncrementalToolCallParser` (the
    //! Qwen3 protocol) feeding a real `WireMapper` (OpenAI) through
    //! the pump helpers.

    use super::*;
    use crate::runtime::chat::ToolDirectory;
    use crate::runtime::chat::ToolSpec;
    use crate::runtime::chat::protocols::qwen3;
    use crate::runtime::chat::wire::openai::OpenAiStreamMapper;
    use serde_json::json;
    use std::sync::Arc;

    fn calculator_directory() -> Arc<ToolDirectory> {
        Arc::new(
            ToolDirectory::new(vec![ToolSpec::new(
                "add",
                None,
                json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "number" },
                        "b": { "type": "number" },
                    },
                    "required": ["a", "b"],
                }),
            )])
            .unwrap(),
        )
    }

    fn test_id_minter() -> impl FnMut(&str) -> String + Send + 'static {
        let mut n = 0u64;
        move |prefix: &str| {
            n += 1;
            format!("{prefix}-{n}")
        }
    }

    #[test]
    fn pump_text_then_finish_yields_complete_call_frames_then_finish_frame() {
        let mut parser = qwen3::make_parser(calculator_directory());
        let mut mapper = OpenAiStreamMapper::new(test_id_minter());

        let body_frames = pump_text(
            &mut *parser,
            &mut mapper,
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#,
        )
        .unwrap();
        // Two streaming frames: ToolCallStart (with name + id), then
        // ToolCallArgsDelta (with the args string).
        assert_eq!(body_frames.len(), 2);
        let start_value = serde_json::to_value(&body_frames[0].delta).unwrap();
        assert_eq!(start_value["tool_calls"][0]["function"]["name"], "add");

        let tail_frames = pump_finish(&mut *parser, &mut mapper, StopReason::EndOfText).unwrap();
        // One terminal frame: empty delta + finish_reason=ToolCalls.
        assert_eq!(tail_frames.len(), 1);
        assert_eq!(
            tail_frames[0].finish_reason,
            Some(crate::types::openai::FinishReason::ToolCalls)
        );

        let snap = mapper.snapshot().unwrap();
        assert_eq!(snap.finish_reason, crate::types::openai::FinishReason::ToolCalls);
    }

    #[test]
    fn pump_text_returns_decode_failure_on_unknown_tool() {
        let mut parser = qwen3::make_parser(calculator_directory());
        let mut mapper = OpenAiStreamMapper::new(test_id_minter());

        let err = pump_text(
            &mut *parser,
            &mut mapper,
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
        )
        .unwrap_err();
        assert!(err.failure.to_string().contains("missing"));
        assert!(err.cleanup.is_empty(), "OpenAI has no cleanup frames");
    }

    /// Regression for the "open block + error" wire bug. With the
    /// Anthropic mapper, fatal events arriving mid-text-block must
    /// produce a `BlockStop` cleanup frame in `PumpError.cleanup` so
    /// the caller can emit it before the surface's error frame —
    /// without consumers having to remember to call
    /// `mapper.close_for_error()` separately.
    #[test]
    fn pump_text_failure_carries_anthropic_cleanup_frames() {
        use crate::runtime::chat::wire::anthropic::{
            AnthropicStreamFrame, AnthropicStreamMapper,
        };
        let mut parser = qwen3::make_parser(calculator_directory());
        let mut mapper = AnthropicStreamMapper::new(test_id_minter());

        // Open a text block, then trigger UnknownTool. The pump must
        // bundle a BlockStop cleanup frame with the failure.
        pump_text(&mut *parser, &mut mapper, "hello ").unwrap();
        let err = pump_text(
            &mut *parser,
            &mut mapper,
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
        )
        .unwrap_err();
        assert!(err.failure.to_string().contains("missing"));
        assert_eq!(err.cleanup.len(), 1, "expected one BlockStop cleanup");
        assert!(matches!(
            err.cleanup[0],
            AnthropicStreamFrame::BlockStop { index: 0 }
        ));
    }

    /// After the pump returns a failure, `mapper.close_for_error()`
    /// must be a no-op (the pump already drained it). This proves
    /// the contract: callers should NOT redundantly call
    /// `close_for_error` after handling a `PumpError`.
    #[test]
    fn pump_text_drains_close_for_error() {
        use crate::runtime::chat::wire::anthropic::AnthropicStreamMapper;
        let mut parser = qwen3::make_parser(calculator_directory());
        let mut mapper = AnthropicStreamMapper::new(test_id_minter());

        pump_text(&mut *parser, &mut mapper, "hello ").unwrap();
        let _ = pump_text(
            &mut *parser,
            &mut mapper,
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
        );
        assert!(
            crate::runtime::chat::wire::WireMapper::close_for_error(&mut mapper).is_empty(),
            "pump should have drained close_for_error"
        );
    }
}

#[cfg(test)]
mod golden_traces {
    //! Exact frame-sequence tests for the wire-error scenarios that
    //! the surrounding code is most prone to regressing on. Few in
    //! number, high in signal: the trace IS the contract.
    //!
    //! The Anthropic transport-error trace lives in
    //! node `crates/cli/src/commands/gateway/anthropic.rs` because
    //! transport errors are a node concern (require the executor
    //! Stream type and aren't reproducible here). It will land
    //! together with the A3-tests SSE harness extraction.
    use super::*;
    use crate::runtime::chat::wire::anthropic::{AnthropicStreamFrame, AnthropicStreamMapper};
    use crate::runtime::chat::wire::openai::OpenAiStreamMapper;
    use crate::runtime::chat::{ToolDirectory, ToolSpec, protocols::qwen3};
    use crate::types::anthropic;
    use serde_json::json;
    use std::sync::Arc;

    fn calculator_directory() -> Arc<ToolDirectory> {
        Arc::new(
            ToolDirectory::new(vec![ToolSpec::new(
                "add",
                None,
                json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "number" },
                        "b": { "type": "number" },
                    },
                    "required": ["a", "b"],
                }),
            )])
            .unwrap(),
        )
    }

    fn id_minter() -> impl FnMut(&str) -> String + Send + 'static {
        let mut n = 0u64;
        move |prefix: &str| {
            n += 1;
            format!("{prefix}_{n}")
        }
    }

    /// Anthropic: open text block + parser fatal event must produce
    /// exactly `[BlockStop(0)]` cleanup before the caller's error
    /// frame. This is the wire-bracketing fix from FEEDBACK #1.
    #[test]
    fn anthropic_text_then_parse_error_emits_block_stop_then_failure() {
        let mut parser = qwen3::make_parser(calculator_directory());
        let mut mapper = AnthropicStreamMapper::new(id_minter());

        // Open text block.
        let frames = pump_text(&mut *parser, &mut mapper, "hi ").unwrap();
        // Expected: BlockStart(0, Text), BlockDelta(0, "hi ").
        assert_eq!(frames.len(), 2);
        assert!(matches!(
            frames[0],
            AnthropicStreamFrame::BlockStart { index: 0, .. }
        ));

        // Trigger UnknownTool.
        let err = pump_text(
            &mut *parser,
            &mut mapper,
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
        )
        .unwrap_err();

        // The exact wire-bracketing trace:
        assert_eq!(
            err.cleanup,
            vec![AnthropicStreamFrame::BlockStop { index: 0 }],
            "must emit exactly one BlockStop for the open text block"
        );
        assert!(err.failure.to_string().contains("missing"));

        // Calling close_for_error after a PumpError must be a no-op
        // (pump already drained it).
        assert!(crate::runtime::chat::wire::WireMapper::close_for_error(&mut mapper).is_empty());
    }

    /// OpenAI: open tool call + parser fatal event produces NO
    /// cleanup frames (no wire bracketing on this surface).
    #[test]
    fn openai_tool_start_then_parse_error_emits_no_cleanup() {
        let mut parser = qwen3::make_parser(calculator_directory());
        let mut mapper = OpenAiStreamMapper::new(id_minter());

        // Drive a tool call mid-stream that will fail validation
        // (unknown tool). The Qwen3 parser emits the failure on the
        // closing sentinel.
        let err = pump_text(
            &mut *parser,
            &mut mapper,
            r#"<tool_call>{"name":"missing","arguments":{}}</tool_call>"#,
        )
        .unwrap_err();
        assert!(
            err.cleanup.is_empty(),
            "OpenAI streaming has no content-block bracketing — no cleanup frames expected, got {:?}",
            err.cleanup
        );
        assert!(err.failure.to_string().contains("missing"));
    }
}

#[cfg(test)]
mod equivalence {
    //! Streaming/non-streaming equivalence: a wire mapper drained
    //! through `pump_text`+`pump_finish` must produce a stream of
    //! frames whose accumulated state matches `mapper.snapshot()`.
    //!
    //! Plus structural invariants:
    //! - Anthropic streaming frames are well-bracketed (every
    //!   `BlockStart` has a matching `BlockStop`, indices contiguous).
    //! - Each surface emits exactly one terminal frame from `finish`.
    //! - `tool_calls` finish reason wins on any tool call (OpenAI).
    //! - `ToolUse` stop reason wins on any tool call (Anthropic).
    //!
    //! Inputs are random sequences of "interesting" chunks fed to a
    //! real Qwen3 parser — so the property ranges over the same
    //! event distributions production code sees.
    //!
    //! These tests encode the user's "non-streaming is a thin wrapper
    //! over streaming" rule as a property.

    use super::*;
    use crate::runtime::chat::wire::anthropic::{AnthropicStreamFrame, AnthropicStreamMapper};
    use crate::runtime::chat::wire::openai::{OpenAiStreamFrame, OpenAiStreamMapper};
    use crate::runtime::chat::{ToolDirectory, ToolSpec, protocols::qwen3};
    use crate::types::{anthropic, openai};
    use proptest::prelude::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn calculator_directory() -> Arc<ToolDirectory> {
        Arc::new(
            ToolDirectory::new(vec![ToolSpec::new(
                "add",
                None,
                json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "number" },
                        "b": { "type": "number" },
                    },
                    "required": ["a", "b"],
                }),
            )])
            .unwrap(),
        )
    }

    fn deterministic_minter() -> impl FnMut(&str) -> String + Send + 'static {
        let n = AtomicU64::new(0);
        move |prefix: &str| {
            let i = n.fetch_add(1, Ordering::Relaxed) + 1;
            format!("{prefix}_{i}")
        }
    }

    /// Hand-curated chunks that, concatenated, form a model output.
    /// Each chunk is a small, valid string the parser can absorb.
    fn arb_chunks() -> impl Strategy<Value = Vec<&'static str>> {
        prop::collection::vec(
            prop_oneof![
                Just("hello "),
                Just("world "),
                Just("more text"),
                Just(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#),
                Just(r#"<tool_call>{"name":"add","arguments":{"a":3,"b":4}}</tool_call>"#),
            ],
            0..6,
        )
    }

    proptest! {
        /// OpenAI: streaming frames replayed back into a synthetic
        /// snapshot equal `mapper.snapshot()`.
        #[test]
        fn openai_stream_replay_equals_snapshot(chunks in arb_chunks()) {
            let mut parser = qwen3::make_parser(calculator_directory());
            let mut mapper = OpenAiStreamMapper::new(deterministic_minter());

            let mut all_frames: Vec<OpenAiStreamFrame> = Vec::new();
            for chunk in &chunks {
                let frames = match pump_text(&mut *parser, &mut mapper, chunk) {
                    Ok(f) => f,
                    Err(_) => return Ok(()),  // skip fatal-event paths
                };
                all_frames.extend(frames);
            }
            let tail = match pump_finish(
                &mut *parser,
                &mut mapper,
                crate::runtime::chat::StopReason::EndOfText,
            ) {
                Ok(f) => f,
                Err(_) => return Ok(()),
            };
            all_frames.extend(tail);

            let snap = mapper.snapshot().unwrap();

            // Replay frames into a synthetic snapshot:
            // - text content = concat of delta.content
            // - finish_reason = the single tail-frame's finish_reason
            let mut content = String::new();
            let mut frame_finish = None;
            for frame in &all_frames {
                if let Some(c) = &frame.delta.content {
                    content.push_str(c);
                }
                if let Some(fr) = frame.finish_reason {
                    prop_assert!(
                        frame_finish.is_none(),
                        "exactly one frame must carry finish_reason"
                    );
                    frame_finish = Some(fr);
                }
            }
            let snap_content = match &snap.message.content {
                Some(openai::MessageContent::Text(t)) => t.clone(),
                _ => String::new(),
            };
            prop_assert_eq!(content, snap_content);
            prop_assert_eq!(frame_finish, Some(snap.finish_reason));
        }

        /// Anthropic: streaming frames must be well-bracketed AND
        /// the closed text/tool_use blocks reconstructed from them
        /// must equal `mapper.snapshot().blocks`.
        #[test]
        fn anthropic_stream_brackets_and_replay_equals_snapshot(
            chunks in arb_chunks(),
        ) {
            let mut parser = qwen3::make_parser(calculator_directory());
            let mut mapper = AnthropicStreamMapper::new(deterministic_minter());

            let mut all_frames: Vec<AnthropicStreamFrame> = Vec::new();
            for chunk in &chunks {
                let frames = match pump_text(&mut *parser, &mut mapper, chunk) {
                    Ok(f) => f,
                    Err(_) => return Ok(()),
                };
                all_frames.extend(frames);
            }
            let tail = match pump_finish(
                &mut *parser,
                &mut mapper,
                crate::runtime::chat::StopReason::EndOfText,
            ) {
                Ok(f) => f,
                Err(_) => return Ok(()),
            };
            all_frames.extend(tail);

            // Bracketing: walk frames, track open blocks, every Start
            // must have a matching Stop, indices are 0-based and
            // contiguous.
            let mut next_expected_index: u32 = 0;
            let mut open_index: Option<u32> = None;
            let mut stop_count = 0_usize;
            let mut closed_blocks: Vec<anthropic::ContentBlock> = Vec::new();
            let mut current_text = String::new();
            // For an open tool_use block: the wire id, name, and the
            // accumulated `input_json_delta` payloads. On BlockStop,
            // parse the accumulated JSON to reconstruct `input`.
            let mut current_tool: Option<(String, String, String)> = None;
            for frame in &all_frames {
                match frame {
                    AnthropicStreamFrame::BlockStart { index, block } => {
                        prop_assert!(open_index.is_none(),
                            "BlockStart while another block open: {open_index:?}");
                        prop_assert_eq!(*index, next_expected_index);
                        open_index = Some(*index);
                        next_expected_index += 1;
                        match block {
                            anthropic::ContentBlock::Text { .. } => {
                                current_text.clear();
                            }
                            anthropic::ContentBlock::ToolUse { id, name, .. } => {
                                current_tool = Some((id.clone(), name.clone(), String::new()));
                            }
                            _ => {}
                        }
                    }
                    AnthropicStreamFrame::BlockDelta { index, delta } => {
                        prop_assert_eq!(open_index, Some(*index));
                        match delta {
                            anthropic::ContentBlockDelta::TextDelta { text } => {
                                current_text.push_str(text);
                            }
                            anthropic::ContentBlockDelta::InputJsonDelta { partial_json } => {
                                if let Some((_, _, args)) = current_tool.as_mut() {
                                    args.push_str(partial_json);
                                }
                            }
                        }
                    }
                    AnthropicStreamFrame::BlockStop { index } => {
                        prop_assert_eq!(open_index, Some(*index));
                        open_index = None;
                        if let Some((id, name, args_json)) = current_tool.take() {
                            let input = if args_json.is_empty() {
                                serde_json::Value::Object(Default::default())
                            } else {
                                serde_json::from_str(&args_json).unwrap_or(
                                    serde_json::Value::Object(Default::default()),
                                )
                            };
                            closed_blocks.push(anthropic::ContentBlock::ToolUse { id, name, input });
                        } else {
                            closed_blocks.push(anthropic::ContentBlock::Text {
                                text: std::mem::take(&mut current_text),
                            });
                        }
                    }
                    AnthropicStreamFrame::Stop(_) => {
                        stop_count += 1;
                    }
                }
            }
            prop_assert_eq!(stop_count, 1, "exactly one Stop frame expected");
            prop_assert!(open_index.is_none(), "stream ended with an open block");

            // Replay matches snapshot. Snapshot fills empty content
            // with a single empty-text-block; do the same for our
            // replay so they agree.
            let snap = mapper.snapshot().unwrap();
            if closed_blocks.is_empty() {
                closed_blocks.push(anthropic::ContentBlock::Text { text: String::new() });
            }
            prop_assert_eq!(closed_blocks, snap.blocks);
        }

        /// Cross-surface property: if the chunks contain ANY tool
        /// call, the OpenAI snapshot's finish_reason MUST be
        /// `ToolCalls` (regardless of the parser stop). And the
        /// Anthropic snapshot's stop_reason MUST be `ToolUse`.
        #[test]
        fn finish_reason_reflects_any_emitted_tool_call(chunks in arb_chunks()) {
            let combined: String = chunks.join("");
            let had_tool_call = combined.contains("<tool_call>")
                && !combined.contains(r#""name":"missing""#);
            // Run OpenAI:
            let mut p1 = qwen3::make_parser(calculator_directory());
            let mut m1 = OpenAiStreamMapper::new(deterministic_minter());
            for chunk in &chunks {
                if pump_text(&mut *p1, &mut m1, chunk).is_err() {
                    return Ok(());
                }
            }
            if pump_finish(&mut *p1, &mut m1, crate::runtime::chat::StopReason::EndOfText).is_err() {
                return Ok(());
            }
            let s1 = m1.snapshot().unwrap();
            if had_tool_call {
                prop_assert_eq!(s1.finish_reason, openai::FinishReason::ToolCalls);
            }
            // Run Anthropic:
            let mut p2 = qwen3::make_parser(calculator_directory());
            let mut m2 = AnthropicStreamMapper::new(deterministic_minter());
            for chunk in &chunks {
                if pump_text(&mut *p2, &mut m2, chunk).is_err() {
                    return Ok(());
                }
            }
            if pump_finish(&mut *p2, &mut m2, crate::runtime::chat::StopReason::EndOfText).is_err() {
                return Ok(());
            }
            let s2 = m2.snapshot().unwrap();
            if had_tool_call {
                prop_assert_eq!(s2.stop_reason, anthropic::StopReason::ToolUse);
            }
        }
    }
}
