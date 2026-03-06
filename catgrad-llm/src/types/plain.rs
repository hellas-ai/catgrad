//! OpenAI /v1/completions wire format.
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use typed_builder::TypedBuilder;

use super::openai::{FinishReason, Stop, Usage};

/// Completion prompt supports string and tokenized forms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Single(String),
    Multiple(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

/// Logprobs payload for plain completions.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Default)]
#[builder(field_defaults(default))]
pub struct CompletionLogprobs {
    pub tokens: Option<Vec<String>>,
    pub token_logprobs: Option<Vec<f32>>,
    pub top_logprobs: Option<Vec<HashMap<String, f32>>>,
    pub text_offset: Option<Vec<u32>>,
}

/// Text-completions request.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct CompletionRequest {
    #[builder(!default)]
    pub model: String,
    #[builder(!default)]
    pub prompt: CompletionPrompt,
    pub best_of: Option<u32>,
    pub echo: Option<bool>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, i32>>,
    pub logprobs: Option<u32>,
    pub max_tokens: Option<u32>,
    pub n: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub stop: Option<Stop>,
    pub stream: Option<bool>,
    pub suffix: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub user: Option<String>,
}

/// One completion choice (used in both non-streaming and streaming responses).
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    #[builder(default)]
    pub logprobs: Option<CompletionLogprobs>,
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
    #[builder(default)]
    pub system_fingerprint: Option<String>,
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
    #[builder(default)]
    pub system_fingerprint: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prompt_deserializes_string_and_token_forms() {
        let text: CompletionPrompt = serde_json::from_value(json!("Hello")).unwrap();
        assert_eq!(text, CompletionPrompt::Single("Hello".to_string()));

        let tokens: CompletionPrompt = serde_json::from_value(json!([1, 2, 3])).unwrap();
        assert_eq!(tokens, CompletionPrompt::Tokens(vec![1, 2, 3]));

        let token_batches: CompletionPrompt =
            serde_json::from_value(json!([[1, 2], [3, 4]])).unwrap();
        assert_eq!(
            token_batches,
            CompletionPrompt::TokenBatches(vec![vec![1, 2], vec![3, 4]])
        );
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
                    "logprobs": {
                        "tokens": ["Hello", "world"],
                        "token_logprobs": [-0.5, -0.25],
                        "text_offset": [0, 6]
                    },
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
}
