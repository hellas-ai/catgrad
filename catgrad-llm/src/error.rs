use thiserror::Error;

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("IO error occurred: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unsupported tensor dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Unsupported model architecture: {0}")]
    UnsupportedModel(String),

    #[error("Invalid model config: {0}")]
    InvalidModelConfig(String),

    #[error("Failed to parse JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Failed to deserialize safetensors: {0}")]
    SafetensorsError(#[from] safetensors::SafeTensorError),

    #[error("Tokenizer Error: {0}")]
    TokenizerError(String),

    #[error("Unknown error occurred: {0}")]
    HuggingFaceAPIError(#[from] hf_hub::api::sync::ApiError),

    #[error("Template rendering error: {0}")]
    TemplateError(#[from] minijinja::Error),
}

// iirc we didn't want to expose the `tokenizers` crate's error type directly (why?)
impl From<tokenizers::Error> for LLMError {
    fn from(err: tokenizers::Error) -> Self {
        LLMError::TokenizerError(err.to_string())
    }
}
