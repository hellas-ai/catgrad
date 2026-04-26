//! Chat-aware decoding layer that binds the chat template, the tool
//! directory, and the per-architecture parser into one value carried
//! across a single chat turn.
//!
//! See `docs/`-equivalent design notes (in the consuming repo) for the
//! contract this module enforces. The short version: tool specs that
//! flowed into the chat template at render time also flow into the
//! streaming parser at parse time, so calls to unknown names or with
//! schema-invalid arguments are surfaced as structured events rather
//! than silently passed through.
//!
//! # Submodules
//!
//! - `event` — [`DecodeEvent`], [`StopReason`], [`SchemaError`],
//!   [`ParserError`].
//! - `tool_spec` — [`ToolSpec`], [`ToolDirectory`] (with pre-compiled
//!   JSON-Schema validators).
//! - `parser` — [`IncrementalToolCallParser`] trait plus shared helpers
//!   (sentinel matcher, no-tools passthrough).
//!
//! Per-architecture parser implementations live in
//! `runtime::chat::protocols::*` (added in subsequent patches).

mod event;
mod parser;
mod protocol;
pub mod protocols;
mod tool_spec;
mod turn;

pub use event::{DecodeEvent, ParserError, SchemaError, StopReason};
pub use parser::{IncrementalToolCallParser, SentinelMatcher};
pub use protocol::{ToolCallProtocol, tool_protocol_for};
pub use tool_spec::{ToolDirectory, ToolSpec};
pub use turn::{ChatOptions, ChatTurn};

pub(crate) use parser::PassthroughParser;
