//! LLM-specific code like tokenization and kv-cache logic which (currently) has to live outside
//! the model graph.
pub mod config;
mod error;
pub mod helpers;
pub mod legacy;
pub mod models;
pub mod run;
pub mod serve;
pub mod utils;

pub use error::LLMError;
pub type Result<T, E = error::LLMError> = std::result::Result<T, E>;
