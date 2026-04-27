//! Structured terminal failures emitted by the per-surface stream
//! mappers when the parser produces one of its terminal events
//! ([`DecodeEvent::UnknownTool`], [`DecodeEvent::InvalidArgs`],
//! [`DecodeEvent::ParseError`]) or when an internal sequence
//! invariant is violated.
//!
//! Mapper [`feed`](super::openai::OpenAiStreamMapper::feed) /
//! [`finish`](super::openai::OpenAiStreamMapper::finish) calls return
//! `Err(DecodeFailure)`; consumers render the appropriate wire error
//! frame and close the stream. Both surfaces (OpenAI, Anthropic)
//! treat the model-output variants as 502-shaped errors, and the
//! `InternalSequence` variant as 500-shaped.
//!
//! After a `DecodeFailure` is returned, the mapper is in a poisoned
//! state — subsequent `feed` / `finish` calls also return the same
//! failure (cloned). The variants are designed to all be cheap to
//! clone (no `serde_json::Error` etc.; see
//! [`crate::runtime::chat::ParserError`] which stringifies its JSON
//! variant for exactly this reason).
//!
//! [`DecodeEvent::UnknownTool`]: super::super::DecodeEvent::UnknownTool
//! [`DecodeEvent::InvalidArgs`]: super::super::DecodeEvent::InvalidArgs
//! [`DecodeEvent::ParseError`]: super::super::DecodeEvent::ParseError

use crate::runtime::chat::event::{ParserError, SchemaError};
use serde_json::Value as JsonValue;
use thiserror::Error;

/// Terminal mapper failure: discriminated cause for a fatal event or
/// sequence violation. Caller renders this into the per-surface wire
/// error envelope.
#[derive(Debug, Clone, Error)]
pub enum DecodeFailure {
    #[error("model called unknown tool `{name}`")]
    UnknownTool { name: String, raw_args: JsonValue },

    #[error("model called `{name}` with arguments that don't match the schema: {}", format_schema_errors(.errors))]
    InvalidArgs {
        name: String,
        args: JsonValue,
        errors: Vec<SchemaError>,
    },

    #[error("model emitted malformed tool call within `{sentinel}`: {source}")]
    ParseError {
        sentinel: &'static str,
        #[source]
        source: ParserError,
    },

    /// The mapper observed a `DecodeEvent` sequence the parser
    /// contract forbids (e.g. `ToolCallEnd` for an index that was
    /// never started, or `TextDelta` while a tool call was open).
    /// This signals a bug in the parser or in the mapper's state
    /// tracking — not bad model output. Callers should surface as
    /// a 500-class error rather than the 502 they use for the
    /// model-output variants.
    #[error("internal mapper sequence error: {reason}")]
    InternalSequence { reason: String },
}

impl DecodeFailure {
    pub fn unknown_tool(name: impl Into<String>, raw_args: JsonValue) -> Self {
        Self::UnknownTool {
            name: name.into(),
            raw_args,
        }
    }

    pub fn invalid_args(
        name: impl Into<String>,
        args: JsonValue,
        errors: Vec<SchemaError>,
    ) -> Self {
        Self::InvalidArgs {
            name: name.into(),
            args,
            errors,
        }
    }

    pub fn parse_error(sentinel: &'static str, source: ParserError) -> Self {
        Self::ParseError { sentinel, source }
    }

    pub fn internal_sequence(reason: impl Into<String>) -> Self {
        Self::InternalSequence {
            reason: reason.into(),
        }
    }
}

fn format_schema_errors(errors: &[SchemaError]) -> String {
    errors
        .iter()
        .map(SchemaError::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}
