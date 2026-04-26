//! Anthropic Messages API wire format.
use crate::LLMError;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serde_with::skip_serializing_none;
use typed_builder::TypedBuilder;

/// Message content payload (string or typed block list).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Top-level message entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

impl AnthropicMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(content.into()),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text("assistant", content)
    }
}

/// System prompt can be plain text or block list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemTextBlock>),
}

/// System text block.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct SystemTextBlock {
    #[serde(rename = "type")]
    pub block_type: SystemBlockType,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SystemBlockType {
    Text,
}

/// Messages API request.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
#[builder(field_defaults(default))]
pub struct MessageRequest {
    #[builder(!default)]
    pub model: String,
    #[builder(!default)]
    pub messages: Vec<AnthropicMessage>,
    #[builder(!default)]
    pub max_tokens: u32,
    pub system: Option<SystemPrompt>,
    pub stream: Option<bool>,
    pub tools: Option<Vec<JsonValue>>,
}

/// Typed message content blocks.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: JsonValue,
    },
    ToolResult {
        tool_use_id: String,
        content: JsonValue,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
}

/// Usage details for Messages API.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl AnthropicUsage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self::builder()
            .input_tokens(input_tokens)
            .output_tokens(output_tokens)
            .build()
    }
}

/// Messages API response.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct MessageResponse {
    pub id: String,
    #[builder(default)]
    #[serde(rename = "type")]
    pub message_type: Option<String>,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    #[builder(default)]
    pub stop_reason: Option<StopReason>,
    pub usage: AnthropicUsage,
}

/// Streaming events emitted by the Anthropic Messages API.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageStreamEvent {
    MessageStart {
        message: MessageResponse,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: StreamMessageDelta,
        usage: AnthropicUsage,
    },
    MessageStop,
    Error {
        error: StreamError,
    },
}

/// Streaming update for message-level fields.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Default)]
#[builder(field_defaults(default))]
pub struct StreamMessageDelta {
    pub stop_reason: Option<StopReason>,
}

/// Streaming delta for a content block.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

/// Error event payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl From<AnthropicMessage> for super::Message {
    fn from(value: AnthropicMessage) -> Self {
        super::Message::Anthropic(value)
    }
}

impl TryFrom<super::Message> for AnthropicMessage {
    type Error = LLMError;

    fn try_from(value: super::Message) -> Result<Self, Self::Error> {
        match value {
            super::Message::Anthropic(msg) => Ok(msg),
            super::Message::OpenAI(_) => Err(LLMError::UnsupportedWireConversion(
                "Cannot convert OpenAI message variant into AnthropicMessage".to_string(),
            )),
        }
    }
}

impl From<&SystemPrompt> for MessageContent {
    fn from(value: &SystemPrompt) -> Self {
        match value {
            SystemPrompt::Text(text) => MessageContent::Text(text.clone()),
            SystemPrompt::Blocks(blocks) => MessageContent::Blocks(
                blocks
                    .iter()
                    .cloned()
                    .map(|block| ContentBlock::Text { text: block.text })
                    .collect(),
            ),
        }
    }
}

impl From<&MessageRequest> for Vec<super::Message> {
    fn from(value: &MessageRequest) -> Self {
        let mut out =
            Vec::with_capacity(value.messages.len() + usize::from(value.system.is_some()));
        if let Some(system) = &value.system {
            out.push(super::Message::Anthropic(AnthropicMessage {
                role: "system".to_string(),
                content: system.into(),
            }));
        }
        for message in value.messages.iter().cloned() {
            out.push(super::Message::Anthropic(message));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_message_converts_to_and_from_internal_message() {
        let anthropic = AnthropicMessage::user("hello");
        let internal: super::super::Message = anthropic.clone().into();
        let back = AnthropicMessage::try_from(internal).unwrap();
        assert_eq!(back, anthropic);
    }

    #[test]
    fn request_conversion_includes_system_blocks() {
        let req = MessageRequest::builder()
            .model("claude-3-5-sonnet".to_string())
            .messages(vec![AnthropicMessage::user("Hi")])
            .max_tokens(16)
            .system(Some(SystemPrompt::Blocks(vec![
                SystemTextBlock::builder()
                    .block_type(SystemBlockType::Text)
                    .text("Be brief".to_string())
                    .build(),
            ])))
            .build();

        let out = Vec::<super::super::Message>::from(&req);
        assert_eq!(
            out,
            vec![
                super::super::Message::anthropic(AnthropicMessage {
                    role: "system".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::Text {
                        text: "Be brief".to_string(),
                    }]),
                }),
                super::super::Message::anthropic(AnthropicMessage::user("Hi")),
            ]
        );
    }

    #[test]
    fn message_request_ignores_unsupported_fields() {
        let sample = json!({
            "model": "claude-3-5-sonnet",
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text":"Hello"},
                    {"type":"text","text":" again"}
                ]
            }],
            "max_tokens": 64,
            "system": "You are concise.",
            "stream": false,
            "tools": [{
                "name":"lookup_weather",
                "description":"Get weather",
                "input_schema":{"type":"object","properties":{"city":{"type":"string"}}
                }
            }],
            "temperature": 0.5,
            "top_p": 0.5,
            "top_k": 50,
            "stop_sequences": ["\n\n"],
            "tool_choice":{"type":"auto"},
            "metadata":{"user_id":"u-123"},
            "thinking":{"type":"enabled","budget_tokens":1024}
        });
        let parsed: MessageRequest = serde_json::from_value(sample).unwrap();
        assert_eq!(
            parsed,
            MessageRequest::builder()
                .model("claude-3-5-sonnet".to_string())
                .messages(vec![AnthropicMessage {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![
                        ContentBlock::Text {
                            text: "Hello".to_string(),
                        },
                        ContentBlock::Text {
                            text: " again".to_string(),
                        },
                    ]),
                }])
                .max_tokens(64)
                .system(Some(SystemPrompt::Text("You are concise.".to_string())))
                .stream(Some(false))
                .tools(Some(vec![json!({
                    "name": "lookup_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                })]))
                .build()
        );
    }

    #[test]
    fn stream_event_round_trip_serde() {
        let sample = json!({
            "type":"content_block_delta",
            "index":0,
            "delta":{
                "type":"text_delta",
                "text":"Zurich"
            }
        });

        let parsed: MessageStreamEvent = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }

    #[test]
    fn stream_error_uses_nested_type_field() {
        let sample = json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Please retry"
            }
        });
        let parsed: MessageStreamEvent = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }
}
