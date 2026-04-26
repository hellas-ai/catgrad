//! Per-architecture tool-call capability registry.
//!
//! Each entry describes how to render the bound tool list for the chat
//! template and how to construct an incremental parser bound to a
//! [`ToolDirectory`].
//!
//! [`tool_protocol_for`] is the single entry point. Architectures
//! missing from the table cannot serve tool-enabled chat requests:
//! [`ChatTurn::new`](super::ChatTurn::new) consults this registry and
//! rejects the turn at construction time when tools are bound but no
//! protocol is registered.
//!
//! Adding support for a new architecture means writing one
//! [`IncrementalToolCallParser`] state machine, a `render_tools` shaping
//! function, and one row in [`tool_protocol_for`].

use std::sync::Arc;

use super::protocols;
use super::{IncrementalToolCallParser, ToolDirectory, ToolSpec};
use serde_json::Value as JsonValue;

/// Capability descriptor for one model architecture's tool-calling
/// dialect. Stored as a `&'static` to allow callers to compare protocol
/// identity by pointer when useful.
#[derive(Debug)]
pub struct ToolCallProtocol {
    /// Shape the bound tool list into the JSON value the chat template
    /// expects. Output is what gets bound to the template's `tools`
    /// variable; shape is architecture-specific.
    pub render_tools: fn(&[ToolSpec]) -> JsonValue,

    /// Construct an incremental parser that owns its tool directory.
    /// The returned parser is `'static`, so a caller can hold both the
    /// `ChatTurn` and the parser together (e.g. on a per-request struct
    /// in a gateway) without a self-referential borrow.
    pub make_parser: fn(Arc<ToolDirectory>) -> Box<dyn IncrementalToolCallParser>,

    /// Whether the model can emit multiple tool calls in a single
    /// generation. Surfaces to clients via the gateway as the
    /// `parallel_tool_calls` capability.
    pub supports_parallel_calls: bool,
}

const QWEN3: ToolCallProtocol = ToolCallProtocol {
    render_tools: protocols::qwen3::render_tools,
    make_parser: protocols::qwen3::make_parser,
    supports_parallel_calls: true,
};

/// Lookup table from HF `architectures[0]` string to the architecture's
/// tool-call protocol. Returns `None` for architectures that do not
/// support tool calling (or that have not yet been ported to the
/// incremental parser model).
pub fn tool_protocol_for(arch: &str) -> Option<&'static ToolCallProtocol> {
    match arch {
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => Some(&QWEN3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_architectures_resolve_to_protocol() {
        assert!(tool_protocol_for("Qwen3ForCausalLM").is_some());
        assert!(tool_protocol_for("Qwen3MoeForCausalLM").is_some());
    }

    #[test]
    fn unknown_architecture_returns_none() {
        assert!(tool_protocol_for("Qwen3_5ForConditionalGeneration").is_none());
        assert!(tool_protocol_for("Lfm2ForCausalLM").is_none());
        assert!(tool_protocol_for("Olmo3ForCausalLM").is_none());
        assert!(tool_protocol_for("LlamaForCausalLM").is_none());
        assert!(tool_protocol_for("").is_none());
    }
}
