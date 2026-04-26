//! Tokenization, prompt preparation, kv-cache logic, and a text-causal
//! session layer that compose over catgrad's generic runtime to do
//! end-to-end LLM inference.

pub mod config;
mod error;
pub mod helpers;
pub mod models;
pub mod runtime;
pub mod types;
pub mod utils;

pub use error::LLMError;
pub use utils::{Detokenizer, PreparedPrompt, detokenize_tokens};
pub type Result<T, E = error::LLMError> = std::result::Result<T, E>;
