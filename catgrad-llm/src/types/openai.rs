//! OpenAI chat-completions wire format.
use crate::LLMError;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use typed_builder::TypedBuilder;

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
}

/// Chat message.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[builder(default)]
    pub content: Option<MessageContent>,
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

/// Stream options.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
pub struct StreamOptions {
    #[builder(default)]
    pub include_usage: Option<bool>,
}

/// Chat-completions request.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[builder(default)]
    pub max_tokens: Option<u32>,
    #[builder(default)]
    pub stream: Option<bool>,
    #[builder(default)]
    pub stream_options: Option<StreamOptions>,
}

/// Token usage details.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self::builder()
            .prompt_tokens(prompt_tokens)
            .completion_tokens(completion_tokens)
            .total_tokens(prompt_tokens + completion_tokens)
            .build()
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
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
}

/// One chat-completion choice.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[builder(default)]
    pub finish_reason: Option<FinishReason>,
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
}

/// One streaming chat-completion choice.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[builder(default)]
    pub finish_reason: Option<FinishReason>,
}

/// Delta payload for streaming response chunks.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Default)]
#[builder(field_defaults(default))]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
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
    fn content_deserializes_from_text_and_parts() {
        let text: MessageContent = serde_json::from_value(json!("hello")).unwrap();
        assert_eq!(text, MessageContent::Text("hello".to_string()));

        let parts: MessageContent = serde_json::from_value(json!([
            {"type":"text", "text":"A"},
            {"type":"text", "text":"B"}
        ]))
        .unwrap();
        assert_eq!(
            parts,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "A".to_string()
                },
                ContentPart::Text {
                    text: "B".to_string()
                }
            ])
        );
    }

    #[test]
    fn chat_message_converts_to_and_from_internal_message() {
        let openai = ChatMessage::user("hello");
        let internal: super::super::Message = openai.clone().into();
        assert_eq!(
            internal,
            super::super::Message::OpenAI(Box::new(openai.clone()))
        );
        let back = ChatMessage::try_from(internal).unwrap();
        assert_eq!(back, openai);
    }

    #[test]
    fn chat_completion_request_ignores_unsupported_fields() {
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
                        {"type":"text","text":" in one sentence"}
                    ],
                    "tool_call_id": "call_1",
                    "tool_calls": [],
                    "audio": {"id":"audio_123","data":"..."},
                    "refusal": "no"
                }
            ],
            "max_tokens": 64,
            "stream": false,
            "stream_options": {"include_usage": true},
            "tools": [],
            "temperature": 0.5,
            "top_p": 0.75,
            "n": 2,
            "stop": ["\n"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.25,
            "logprobs": true,
            "top_logprobs": 3,
            "logit_bias": {"10": 1},
            "seed": 123,
            "tool_choice":"auto",
            "parallel_tool_calls": true,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_reply",
                    "strict": true
                }
            },
            "modalities": ["text", "audio"],
            "audio": {"format":"wav","voice":"alloy"},
            "user": "u-123"
        });
        let parsed: ChatCompletionRequest = serde_json::from_value(sample).unwrap();
        assert_eq!(
            parsed,
            ChatCompletionRequest::builder()
                .model("gpt-4o-mini".to_string())
                .messages(vec![
                    ChatMessage::system("You are terse."),
                    ChatMessage::builder()
                        .role("user".to_string())
                        .content(Some(MessageContent::Parts(vec![
                            ContentPart::Text {
                                text: "Describe this image".to_string(),
                            },
                            ContentPart::Text {
                                text: " in one sentence".to_string(),
                            },
                        ])))
                        .build(),
                ])
                .max_tokens(Some(64))
                .stream(Some(false))
                .stream_options(Some(StreamOptions {
                    include_usage: Some(true),
                }))
                .build()
        );
    }

    #[test]
    fn stream_chunk_round_trip_with_content_delta() {
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
                        "content": "Zurich"
                    }
                }
            ]
        });

        let parsed: ChatCompletionChunk = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }
}
