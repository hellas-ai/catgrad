//! Per-architecture tool-call protocol implementations.
//!
//! Each submodule provides a `make_parser(&ToolDirectory)` constructor
//! returning a boxed [`IncrementalToolCallParser`](super::IncrementalToolCallParser).
//! Architectures missing here cannot serve tool-enabled chat requests:
//! [`ChatTurn::new`](super::ChatTurn::new) (added in a subsequent
//! patch) consults the [`tool_protocol_for`](super::tool_protocol_for)
//! registry and rejects the turn at construction time.

pub mod qwen3;

#[cfg(test)]
pub(crate) mod test_util;
