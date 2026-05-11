pub mod config;
mod error;
pub mod helpers;
pub mod model_media;
pub mod model_utils;
pub mod models;

pub use error::ModelError;
pub use error::ModelError as LLMError;
pub type Result<T, E = error::ModelError> = std::result::Result<T, E>;
