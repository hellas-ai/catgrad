//! Streaming output of a chat-aware decoding session.
//!
//! A [`ChatTurn`](super::ChatTurn) parser feeds detokenized text deltas
//! through an [`IncrementalToolCallParser`](super::IncrementalToolCallParser)
//! and the parser emits a stream of [`DecodeEvent`]s. The variants carry
//! enough information that a caller can render them directly to the
//! OpenAI / Anthropic / plain-SSE wire formats without re-parsing.

use serde_json::Value as JsonValue;
use thiserror::Error;

/// One unit of progress observed by an [`IncrementalToolCallParser`].
///
/// # Per-call atomicity
///
/// `ToolCallStart`, `ToolCallArgsDelta`, and `ToolCallEnd` for a single
/// call are emitted contiguously and atomically. `ToolCallStart` is not
/// "the model just emitted `<tool_call>`" — it is "the parser has
/// completed and validated a call, and is now emitting its wire-level
/// frames." Validation (name lookup, schema check) happens before any
/// of the three are emitted; partial / invalid calls take the
/// [`Self::UnknownTool`] / [`Self::InvalidArgs`] / [`Self::ParseError`]
/// path instead and never produce a `ToolCallStart` for that call.
///
/// Per-call streaming (the second tool call in a response is delivered
/// as soon as its closing sentinel arrives, even while later output is
/// still being generated) is the wire-level streaming guarantee. True
/// per-token argument streaming would require partial-JSON parsing and
/// is out of scope.
///
/// # Terminal events
///
/// `UnknownTool`, `InvalidArgs`, and `ParseError` are **terminal**.
/// After emitting any of them, the parser MUST also emit
/// [`StopReason::ProtocolError`] and then return empty from any
/// subsequent `feed` / `finish` call. This means the gateway never has
/// to render both an error and a "success" finish for the same
/// generation — the parser commits to the error and is done.
#[derive(Debug, Clone)]
pub enum DecodeEvent {
    /// Plain user-visible text. Safe to forward to the client immediately.
    TextDelta(String),

    /// A validated tool call is being emitted. The name is verified
    /// against the bound [`ToolDirectory`](super::ToolDirectory) and
    /// the args have been schema-checked; this event is the *first* of
    /// the contiguous `Start` / `ArgsDelta` / `End` triple, never a
    /// preview of a call that might later be rejected.
    ToolCallStart {
        /// Zero-based index within this generation. Stable across the
        /// `ToolCallStart` / `ToolCallArgsDelta` / `ToolCallEnd` triple
        /// that names a single call.
        index: usize,
        name: String,
    },

    /// Argument JSON for the call named by `index`. Current parsers
    /// emit exactly one `ToolCallArgsDelta` per call, immediately
    /// between `ToolCallStart` and `ToolCallEnd`, carrying the full
    /// serialized arguments. The variant exists in the enum to leave
    /// room for future per-token argument streaming; consumers MUST
    /// already handle the multi-delta case (concatenate by `index`).
    ToolCallArgsDelta { index: usize, delta: String },

    /// Tool call complete. `args` is the fully assembled, parsed, and
    /// schema-validated argument object.
    ToolCallEnd { index: usize, args: JsonValue },

    /// Generation is over. See [`StopReason`] for the meaning of each
    /// reason; in particular [`StopReason::ProtocolError`] is the
    /// internal terminal marker and must NOT be rendered as a client
    /// "success" finish reason — the gateway maps it to an error
    /// response.
    Stop { reason: StopReason },

    /// Parsed call referenced a name not in the bound directory.
    /// **Terminal**: this event is followed by `Stop { ProtocolError }`
    /// and the parser ignores all subsequent input.
    UnknownTool { name: String, raw_args: JsonValue },

    /// Parsed call's args failed JSON-schema validation. No
    /// `ToolCallStart` / `End` is emitted for this call. **Terminal**:
    /// followed by `Stop { ProtocolError }`; subsequent input ignored.
    InvalidArgs {
        name: String,
        args: JsonValue,
        errors: Vec<SchemaError>,
    },

    /// The protocol detected a sentinel but failed to extract a usable
    /// call. **Terminal**: followed by `Stop { ProtocolError }`;
    /// subsequent input ignored.
    ParseError {
        sentinel: &'static str,
        source: ParserError,
    },
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Model emitted an end-of-text token.
    EndOfText,
    /// Caller's `max_new_tokens` budget was exhausted.
    MaxTokens,
    /// A configured stop sequence was matched.
    StopSequence,
    /// Internal terminal marker emitted after a fatal parser error
    /// (`UnknownTool`, `InvalidArgs`, `ParseError`). The gateway must
    /// NOT render this as an OpenAI / Anthropic "success" `finish_reason`
    /// — it is a signal that the preceding error event is the truthful
    /// end of the response and the connection should close with an
    /// error frame.
    ProtocolError,
}

/// One JSON-schema validation failure.
///
/// Operators read these in error responses and audit logs; the format
/// mirrors what `jsonschema` produces but is owned data so consumers
/// don't need a `'a` lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    /// JSON Pointer-style path into the failing instance (e.g. `/foo/0/bar`).
    pub path: String,
    /// Human-readable description of the failure.
    pub message: String,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Reasons a sentinel-detected payload failed to become a structured call.
#[derive(Debug, Error)]
pub enum ParserError {
    /// The payload between sentinels did not parse as valid JSON / Python
    /// / whatever the architecture's protocol expects.
    #[error("malformed tool-call payload: {0}")]
    Malformed(String),

    /// A sentinel was opened but never closed before the stream ended.
    #[error("unterminated tool-call payload (missing closing sentinel)")]
    Unterminated,

    /// The payload parsed but was missing a required field (e.g. `name`).
    #[error("tool-call payload missing required field `{0}`")]
    MissingField(&'static str),

    /// JSON deserialization error inside the payload.
    #[error("invalid JSON in tool-call payload: {0}")]
    Json(#[from] serde_json::Error),

    /// The payload between an open and close sentinel exceeded the
    /// per-call byte limit. The payload itself is intentionally NOT
    /// included in the message — operators see only the static fact
    /// that the limit was exceeded.
    #[error("tool-call payload exceeded maximum size of {limit_bytes} bytes")]
    PayloadTooLarge { limit_bytes: usize },
}

impl Clone for ParserError {
    fn clone(&self) -> Self {
        match self {
            Self::Malformed(msg) => Self::Malformed(msg.clone()),
            Self::Unterminated => Self::Unterminated,
            Self::MissingField(field) => Self::MissingField(field),
            Self::Json(err) => Self::Malformed(err.to_string()),
            Self::PayloadTooLarge { limit_bytes } => Self::PayloadTooLarge {
                limit_bytes: *limit_bytes,
            },
        }
    }
}
