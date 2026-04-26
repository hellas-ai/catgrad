//! Free-function constructor that builds a text-generation [`Program`]
//! from a HuggingFace-style config JSON, dispatching to the right model
//! implementation by architecture string.

use crate::Result;
use crate::utils::get_model;
use catgrad::category::core::Dtype;
use catgrad::runtime::Program;

/// Build a text-generation `Program` from a HuggingFace-style config JSON.
///
/// Picks the model implementation by architecture string, then constructs
/// a `Program` with the appropriate state schema and per-step nat hints.
pub fn text_program_from_config(
    config_json: &serde_json::Value,
    max_sequence_length: usize,
    dtype: Dtype,
) -> Result<Program> {
    let model = get_model(config_json, max_sequence_length, None, dtype)?;
    let module_path = model.path();
    let empty_state_type = model.empty_state_type();
    let extra_nat_chunk_size = model.extra_nat_chunk_size();
    Ok(Program::from_module(
        model.as_ref(),
        module_path,
        empty_state_type,
        max_sequence_length,
        extra_nat_chunk_size,
    )?)
}
