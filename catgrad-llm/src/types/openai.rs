//! OpenAI chat-completions wire format.
use crate::LLMError;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use typed_builder::TypedBuilder;

use super::JsonSchema;

/// Stop sequence(s) for generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Stop {
    Single(String),
    Multiple(Vec<String>),
}

/// A chat message's content payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A typed content part in chat messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlPart },
    InputAudio { input_audio: InputAudioPart },
    Refusal { refusal: String },
    File { file: FilePart },
}

/// `image_url` content part payload.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct ImageUrlPart {
    pub url: String,
    #[builder(default)]
    pub detail: Option<String>,
}

/// `input_audio` content part payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputAudioPart {
    pub data: String,
    pub format: String,
}

/// `file` content part payload.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct FilePart {
    pub file_id: Option<String>,
    pub file_data: Option<String>,
    pub filename: Option<String>,
}

/// Chat message.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct ChatMessage {
    #[builder(!default)]
    pub role: String,
    pub content: Option<MessageContent>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<MessageToolCall>>,
    pub function_call: Option<FunctionCall>,
    pub refusal: Option<String>,
    pub audio: Option<MessageAudio>,
}

impl ChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self::builder()
            .role(role.into())
            .content(Some(MessageContent::Text(content.into())))
            .build()
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::text("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text("assistant", content)
    }
}

/// Assistant audio metadata attached to a message.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct MessageAudio {
    pub id: Option<String>,
    pub data: Option<String>,
    pub transcript: Option<String>,
    pub expires_at: Option<i64>,
}

/// Tool call entry inside an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageToolCall {
    Function { id: String, function: FunctionCall },
    Custom { id: String, custom: CustomToolCall },
}

/// Function call payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Custom tool call payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomToolCall {
    pub name: String,
    pub input: String,
}

/// Tool definition for chat-completions requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    Function { function: FunctionDefinition },
    Custom { custom: CustomToolDefinition },
}

/// Function-tool schema.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct FunctionDefinition {
    #[builder(!default)]
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<JsonSchema>,
    pub strict: Option<bool>,
}

/// Custom-tool schema.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct CustomToolDefinition {
    pub name: String,
    #[builder(default)]
    pub description: Option<String>,
}

/// Tool-choice control for chat-completions requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Object(ToolChoiceObject),
}

/// Object-form tool choice.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
#[builder(field_defaults(default))]
pub struct ToolChoiceObject {
    #[builder(!default)]
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: Option<ToolChoiceFunction>,
    pub custom: Option<ToolChoiceCustom>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolChoiceCustom {
    pub name: String,
}

/// Stream options.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

/// Structured response format options.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: ResponseJsonSchema },
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct ResponseJsonSchema {
    #[builder(!default)]
    pub name: String,
    pub description: Option<String>,
    pub schema: Option<JsonSchema>,
    pub strict: Option<bool>,
}

/// Request audio output configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioRequest {
    pub format: String,
    pub voice: String,
}

/// Chat-completions request.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct ChatCompletionRequest {
    #[builder(!default)]
    pub model: String,
    #[builder(!default)]
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Stop>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub logit_bias: Option<HashMap<String, i32>>,
    pub seed: Option<i64>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<ResponseFormat>,
    pub modalities: Option<Vec<String>>,
    pub audio: Option<AudioRequest>,
    pub user: Option<String>,
}

/// Token usage details.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[builder(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[builder(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

impl Usage {
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
    pub audio_tokens: Option<u32>,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    FunctionCall,
}

/// Token-level logprob item.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    #[builder(default)]
    pub bytes: Option<Vec<u8>>,
    #[builder(default)]
    pub top_logprobs: Option<Vec<TokenLogprobEntry>>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct TokenLogprobEntry {
    pub token: String,
    pub logprob: f32,
    #[builder(default)]
    pub bytes: Option<Vec<u8>>,
}

/// Choice-level logprobs payload.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Default)]
#[builder(field_defaults(default))]
pub struct ChoiceLogprobs {
    pub content: Option<Vec<TokenLogprob>>,
    pub refusal: Option<Vec<TokenLogprob>>,
}

/// Chat-completions response.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[builder(default)]
    pub usage: Option<Usage>,
    #[builder(default)]
    pub system_fingerprint: Option<String>,
}

/// One chat-completion choice.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[builder(default)]
    pub finish_reason: Option<FinishReason>,
    #[builder(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Streaming chat-completions chunk.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
    #[builder(default)]
    pub usage: Option<Usage>,
    #[builder(default)]
    pub system_fingerprint: Option<String>,
}

/// One streaming chat-completion choice.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[builder(default)]
    pub finish_reason: Option<FinishReason>,
    #[builder(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Delta payload for streaming response chunks.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Default)]
#[builder(field_defaults(default))]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<DeltaToolCall>>,
    pub function_call: Option<DeltaFunctionCall>,
}

/// Partial tool call emitted in chat streaming.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct DeltaToolCall {
    #[builder(!default)]
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<DeltaFunctionCall>,
    pub custom: Option<DeltaCustomToolCall>,
}

/// Partial function call emitted in chat streaming.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct DeltaFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Partial custom tool call emitted in chat streaming.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct DeltaCustomToolCall {
    pub name: Option<String>,
    pub input: Option<String>,
}

impl From<ChatMessage> for super::Message {
    fn from(value: ChatMessage) -> Self {
        super::Message::OpenAI(Box::new(value))
    }
}

impl TryFrom<super::Message> for ChatMessage {
    type Error = LLMError;

    fn try_from(value: super::Message) -> Result<Self, Self::Error> {
        match value {
            super::Message::OpenAI(msg) => Ok(*msg),
            super::Message::Anthropic(_) => Err(LLMError::UnsupportedWireConversion(
                "Cannot convert Anthropic message variant into OpenAI ChatMessage".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finish_reason_serializes_to_snake_case() {
        let value = serde_json::to_value(FinishReason::ToolCalls).unwrap();
        assert_eq!(value, json!("tool_calls"));
    }

    #[test]
    fn content_deserializes_from_text_and_parts() {
        let text: MessageContent = serde_json::from_value(json!("hello")).unwrap();
        assert_eq!(text, MessageContent::Text("hello".to_string()));

        let parts: MessageContent = serde_json::from_value(json!([
            {"type":"text", "text":"A"},
            {"type":"refusal", "refusal":"B"}
        ]))
        .unwrap();
        assert_eq!(
            parts,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "A".to_string()
                },
                ContentPart::Refusal {
                    refusal: "B".to_string()
                }
            ])
        );
    }

    #[test]
    fn chat_message_converts_to_and_from_internal_message() {
        let openai = ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello".to_string())),
            name: Some("alice".to_string()),
            tool_call_id: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            audio: None,
        };
        let internal: super::super::Message = openai.clone().into();
        assert_eq!(
            internal,
            super::super::Message::OpenAI(Box::new(openai.clone()))
        );

        let back = ChatMessage::try_from(internal).unwrap();
        assert_eq!(back, openai);
    }

    #[test]
    fn chat_completion_request_round_trip_serde() {
        let sample = json!({
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are terse."
                },
                {
                    "role": "user",
                    "content": [
                        {"type":"text","text":"Describe this image"},
                        {"type":"image_url","image_url":{"url":"https://example.com/cat.png","detail":"high"}}
                    ]
                }
            ],
            "max_tokens": 64,
            "temperature": 0.5,
            "tools": [
                {
                    "type":"function",
                    "function":{
                        "name":"lookup_weather",
                        "description":"Get weather",
                        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}
                    }
                }
            ],
            "tool_choice":"auto",
            "parallel_tool_calls": true,
            "stream": false,
            "stop": ["\n"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_reply",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"}
                        },
                        "required": ["summary"]
                    }
                }
            },
            "modalities": ["text", "audio"],
            "audio": {"format":"wav","voice":"alloy"},
            "user": "u-123"
        });
        let parsed: ChatCompletionRequest = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }

    #[test]
    fn stream_chunk_round_trip_with_tool_delta() {
        let sample = json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": "{\"city\":\"Zurich\"}"
                                }
                            }
                        ]
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "Zur",
                                "logprob": -0.25,
                                "bytes": [90, 117, 114],
                                "top_logprobs": [
                                    {"token":"Zur","logprob":-0.25,"bytes":[90,117,114]}
                                ]
                            }
                        ]
                    }
                }
            ]
        });

        let parsed: ChatCompletionChunk = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }
}
