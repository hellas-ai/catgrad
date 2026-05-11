pub mod config;
mod error;
pub mod helpers;
pub mod models;
pub mod utils;

pub use error::ModelError;
pub use error::ModelError as LLMError;
pub type Result<T, E = error::ModelError> = std::result::Result<T, E>;
