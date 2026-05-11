//! LLM-specific runtime code like tokenization and kv-cache logic which has to live outside
//! the model graph.
mod error;
pub mod run;
pub mod types;
pub mod utils;

#[cfg(test)]
mod helpers_tests;

pub use catgrad_llm_models::{ModelError, config, helpers, model_media, model_utils, models};
pub use error::LLMError;
pub use utils::{Detokenizer, PreparedPrompt, detokenize_tokens};
pub type Result<T, E = error::LLMError> = std::result::Result<T, E>;
