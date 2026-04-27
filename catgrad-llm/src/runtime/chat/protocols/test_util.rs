//! Shared test-only helpers for parser proptests. Lives at the
//! protocol-module level so each per-architecture parser
//! (`qwen3`, future `qwen3_5`, `qwen_xml`, etc.) can reuse the
//! chunk-invariance machinery without copy-paste.
//!
//! Each protocol module supplies:
//! - a `make_parser` constructor (already part of the production
//!   API),
//! - a `ToolDirectory` shaped for the tools its sample inputs use,
//! - a list of "interesting" inputs for the parser dialect.
//!
//! The helpers below take both as parameters and run the
//! chunk-invariance property over them.

#![cfg(test)]

use crate::runtime::chat::wire::DecodeFailure;
use crate::runtime::chat::{
    AssistantTurnAccumulator, DecodedAssistantTurn, IncrementalToolCallParser,
    StopReason as ParserStopReason, ToolDirectory,
};
use std::sync::Arc;

/// Decode `text` through a fresh parser as a single `feed` call,
/// then `finish`. Drains the events into an accumulator and returns
/// the snapshot result. Skips invalid sequences via the accumulator's
/// own `Err` propagation.
pub fn decode_whole<F>(make: F, text: &str) -> Result<DecodedAssistantTurn, DecodeFailure>
where
    F: FnOnce(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    F: 'static,
{
    let mut parser = make(make_dummy_directory());
    let mut events = parser.feed(text);
    events.extend(parser.finish(ParserStopReason::EndOfText));
    let mut acc = AssistantTurnAccumulator::new();
    for ev in events {
        acc.feed(ev)?;
    }
    acc.into_turn()
}

/// Decode `text` through a fresh parser, but split it into chunks at
/// the given byte offsets. Offsets are clamped to char boundaries.
/// Same return shape as `decode_whole`.
pub fn decode_chunked<F>(
    make: F,
    text: &str,
    splits: &[usize],
) -> Result<DecodedAssistantTurn, DecodeFailure>
where
    F: FnOnce(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,
    F: 'static,
{
    let mut parser = make(make_dummy_directory());
    let mut acc = AssistantTurnAccumulator::new();
    let mut last = 0;
    for &raw_split in splits {
        let split = clamp_to_char_boundary(text, raw_split.min(text.len()));
        if split <= last {
            continue;
        }
        for ev in parser.feed(&text[last..split]) {
            acc.feed(ev)?;
        }
        last = split;
    }
    if last < text.len() {
        for ev in parser.feed(&text[last..]) {
            acc.feed(ev)?;
        }
    }
    for ev in parser.finish(ParserStopReason::EndOfText) {
        acc.feed(ev)?;
    }
    acc.into_turn()
}

fn clamp_to_char_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Lazily-constructed default `ToolDirectory` for proptests that
/// don't need a specific tool surface. Per-protocol tests should
/// pass their own directory if their sample inputs require specific
/// tool names.
fn make_dummy_directory() -> Arc<ToolDirectory> {
    use crate::runtime::chat::ToolSpec;
    Arc::new(
        ToolDirectory::new(vec![ToolSpec::new(
            "add",
            None,
            serde_json::json!({
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
