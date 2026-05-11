use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("IO error occurred: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unsupported tensor dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Unsupported model architecture: {0}")]
    UnsupportedModel(String),

    #[error("Invalid model config: {0}")]
    InvalidModelConfig(String),

    #[error("Failed to decode JSON at path `{path}`: {source}")]
    JsonError {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Unsupported wire-format conversion: {0}")]
    UnsupportedWireConversion(String),
}

impl From<serde_json::Error> for ModelError {
    fn from(err: serde_json::Error) -> Self {
        ModelError::JsonError {
            path: "$".to_string(),
            source: err,
        }
    }
}
