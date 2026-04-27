//! Per-surface stream mappers that turn the wire-neutral
//! [`DecodeEvent`](super::DecodeEvent) stream into surface-specific
//! assistant payloads.
//!
//! There is **one execution kind** in the system: a streaming
//! [`DecodeEvent`] sequence emitted by the per-architecture parser.
//! Each wire surface (OpenAI, Anthropic) gets exactly **one** mapper
//! that owns surface semantics. Both streaming and non-streaming
//! consumers feed the same event sequence through the same mapper:
//!
//! - **Streaming sinks** read the per-event `feed`/`finish` return
//!   values and write each frame to SSE as it arrives.
//! - **Non-streaming sinks** discard the per-event return values and
//!   call `snapshot()` once at the end to read the buffered assistant
//!   payload.
//!
//! Non-streaming is therefore not a different parser, collector, or
//! semantic path — it is a *buffered readout of the same surface
//! mapper state*.
//!
//! # What the mapper owns
//!
//! - Surface-specific accumulation (text, tool calls, content blocks).
//! - Tool-call ID minting via an injected allocator (so process-wide
//!   uniqueness is the caller's responsibility — the mapper never
//!   uses a per-instance counter).
//! - The mapping from [`DecodeEvent`] terminal variants to
//!   [`DecodeFailure`].
//!
//! # What the mapper does NOT own
//!
//! - HTTP envelopes: response id, model name, created timestamp,
//!   usage, transport. Those stay with the caller.
//! - Anything outside chat semantics for its surface.

mod failure;
pub mod openai;
pub mod anthropic;
mod pump;

pub use failure::DecodeFailure;
pub use pump::{PumpError, WireMapper, pump_finish, pump_text};
