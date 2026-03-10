//! OpenAI /v1/completions wire format.
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use typed_builder::TypedBuilder;

use super::openai::{FinishReason, Usage};

/// Text-completions request.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct CompletionRequest {
    #[builder(!default)]
    pub model: String,
    #[builder(!default)]
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
}

/// One completion choice (used in both non-streaming and streaming responses).
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    #[builder(default)]
    pub finish_reason: Option<FinishReason>,
}

/// Text-completions response.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[builder(default)]
    pub usage: Option<Usage>,
}

/// Streaming completion chunk.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn completion_request_requires_string_prompt() {
        let parsed: CompletionRequest = serde_json::from_value(json!({
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Hello"
        }))
        .unwrap();
        assert_eq!(parsed.prompt, "Hello");

        let err = serde_json::from_value::<CompletionRequest>(json!({
            "model": "gpt-3.5-turbo-instruct",
            "prompt": [1, 2, 3]
        }))
        .unwrap_err();
        assert!(err.to_string().contains("string"));
    }

    #[test]
    fn completion_response_round_trip_serde() {
        let sample = json!({
            "id": "cmpl-abc",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo-instruct",
            "choices": [
                {
                    "index": 0,
                    "text": "Hello world",
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        });

        let parsed: CompletionResponse = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }

    #[test]
    fn completion_request_ignores_unsupported_fields() {
        let sample = json!({
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Hello",
            "max_tokens": 16,
            "stream": false,
            "presence_penalty": 0.25,
        });

        let parsed: CompletionRequest = serde_json::from_value(sample).unwrap();
        assert_eq!(
            parsed,
            CompletionRequest::builder()
                .model("gpt-3.5-turbo-instruct".to_string())
                .prompt("Hello".to_string())
                .max_tokens(Some(16))
                .stream(Some(false))
                .build()
        );
    }
}
