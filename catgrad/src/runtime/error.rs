//! Error type for the runtime layer and the module's `Result` alias.

use thiserror::Error;

/// Errors produced when constructing or driving a [`super::Program`],
/// [`super::Inputs`], [`super::BoundProgram`], or [`super::Session`].
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("invalid program: {0}")]
    InvalidProgram(String),

    #[error("invalid parameter `{path}`: {reason}")]
    InvalidParameter { path: String, reason: String },

    #[error("incompatible snapshot: inputs or program id mismatch")]
    IncompatibleSnapshot,

    #[error("execution error: {0}")]
    ExecutionError(String),

    #[error("unexpected program output: {0}")]
    UnexpectedProgramOutput(String),
}

pub type Result<T, E = RuntimeError> = std::result::Result<T, E>;
