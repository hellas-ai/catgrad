//! Anthropic Messages API wire format.
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use typed_builder::TypedBuilder;

use super::{JsonMap, JsonSchema};

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
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct SystemTextBlock {
    #[serde(rename = "type")]
    pub block_type: SystemBlockType,
    pub text: String,
    #[builder(default)]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SystemBlockType {
    Text,
}

/// Cache control block option.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
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
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stream: Option<bool>,
    pub stop_sequences: Option<Vec<String>>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    pub metadata: Option<RequestMetadata>,
    pub thinking: Option<ThinkingConfig>,
}

/// Tool definition.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    #[builder(default)]
    pub description: Option<String>,
    pub input_schema: JsonSchema,
    #[builder(default)]
    pub cache_control: Option<CacheControl>,
}

/// Tool selection behavior.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
#[builder(field_defaults(default))]
pub struct ToolChoice {
    #[builder(!default)]
    #[serde(rename = "type")]
    pub choice_type: ToolChoiceType,
    pub name: Option<String>,
    pub disable_parallel_tool_use: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    Tool,
}

/// Request metadata.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq, Default)]
#[builder(field_defaults(default))]
pub struct RequestMetadata {
    pub user_id: Option<String>,
}

/// Thinking budget/configuration.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: ThinkingType,
    #[builder(default)]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingType {
    Enabled,
    Disabled,
    Adaptive,
}

/// Typed message content blocks.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        citations: Option<Vec<Citation>>,
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: JsonMap,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<MessageContent>,
        is_error: Option<bool>,
        cache_control: Option<CacheControl>,
    },
    Document {
        source: DocumentSource,
        title: Option<String>,
        context: Option<String>,
        citations: Option<Vec<Citation>>,
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
}

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Citation {
    CharLocation {
        cited_text: String,
        document_index: u32,
        start_char_index: u32,
        end_char_index: u32,
        document_title: Option<String>,
    },
    PageLocation {
        document_index: u32,
        start_page_number: u32,
        end_page_number: u32,
        document_title: Option<String>,
    },
    ContentBlockLocation {
        document_index: u32,
        start_block_index: u32,
        end_block_index: u32,
        document_title: Option<String>,
    },
}

/// Image source payload.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// Document source payload.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 { media_type: String, data: String },
    Text { text: String },
    Url { url: String },
    File { file_id: String },
    Content { content: Vec<ContentBlock> },
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
}

/// Usage details for Messages API.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[builder(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[builder(default)]
    pub cache_read_input_tokens: Option<u32>,
}

impl AnthropicUsage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        }
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
    #[builder(default)]
    pub stop_sequence: Option<String>,
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
    Ping,
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
    pub stop_sequence: Option<String>,
}

/// Streaming delta for a content block.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
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

impl MessageRequest {
    /// Converts request content into the internal chat-message representation.
    pub fn to_messages(&self) -> Vec<super::Message> {
        let mut out = Vec::with_capacity(self.messages.len() + usize::from(self.system.is_some()));
        if let Some(system) = &self.system {
            let content = match system {
                SystemPrompt::Text(text) => MessageContent::Text(text.clone()),
                SystemPrompt::Blocks(blocks) => MessageContent::Blocks(
                    blocks
                        .iter()
                        .cloned()
                        .map(|block| ContentBlock::Text {
                            text: block.text,
                            citations: None,
                            cache_control: block.cache_control,
                        })
                        .collect(),
                ),
            };
            out.push(super::Message::Anthropic(AnthropicMessage {
                role: "system".to_string(),
                content,
            }));
        }
        for message in self.messages.iter().cloned() {
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
    fn stop_reason_serializes_to_snake_case() {
        let value = serde_json::to_value(StopReason::PauseTurn).unwrap();
        assert_eq!(value, json!("pause_turn"));
    }

    #[test]
    fn anthropic_message_converts_to_internal_message() {
        let input = AnthropicMessage {
            role: "user".to_string(),
            content: MessageContent::Text("hello".to_string()),
        };
        let converted: super::super::Message = input.into();
        assert_eq!(
            converted,
            super::super::Message::Anthropic(AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            })
        );
    }

    #[test]
    fn anthropic_message_converts_blocks_losslessly() {
        let input = AnthropicMessage {
            role: "user".to_string(),
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: "hello".to_string(),
                citations: None,
                cache_control: None,
            }]),
        };
        let converted: super::super::Message = input.into();
        assert!(matches!(converted, super::super::Message::Anthropic(_)));
    }

    #[test]
    fn to_messages_includes_system_blocks() {
        let req = MessageRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Hi".to_string()),
            }],
            max_tokens: 16,
            system: Some(SystemPrompt::Blocks(vec![SystemTextBlock {
                block_type: SystemBlockType::Text,
                text: "Be brief".to_string(),
                cache_control: None,
            }])),
            temperature: None,
            top_p: None,
            top_k: None,
            stream: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
        };

        let out = req.to_messages();
        assert_eq!(out.len(), 2);
        assert!(matches!(&out[0], super::super::Message::Anthropic(_)));
        assert!(matches!(&out[1], super::super::Message::Anthropic(_)));
    }

    #[test]
    fn message_request_round_trip_serde() {
        let sample = json!({
            "model": "claude-3-5-sonnet",
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text":"Hello"},
                    {"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}
                ]
            }],
            "max_tokens": 64,
            "system": "You are concise.",
            "temperature": 0.5,
            "top_p": 0.5,
            "top_k": 50,
            "stream": false,
            "stop_sequences": ["\n\n"],
            "tools": [{
                "name":"lookup_weather",
                "description":"Get weather",
                "input_schema":{"type":"object","properties":{"city":{"type":"string"}}
                }
            }],
            "tool_choice":{"type":"auto"},
            "metadata":{"user_id":"u-123"},
            "thinking":{"type":"enabled","budget_tokens":1024}
        });
        let parsed: MessageRequest = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }

    #[test]
    fn thinking_type_accepts_all_known_values() {
        let enabled = json!({"type":"enabled","budget_tokens":1024});
        let disabled = json!({"type":"disabled"});
        let adaptive = json!({"type":"adaptive"});

        let enabled_cfg: ThinkingConfig = serde_json::from_value(enabled).unwrap();
        let disabled_cfg: ThinkingConfig = serde_json::from_value(disabled).unwrap();
        let adaptive_cfg: ThinkingConfig = serde_json::from_value(adaptive).unwrap();

        assert_eq!(enabled_cfg.thinking_type, ThinkingType::Enabled);
        assert_eq!(disabled_cfg.thinking_type, ThinkingType::Disabled);
        assert_eq!(adaptive_cfg.thinking_type, ThinkingType::Adaptive);
    }

    #[test]
    fn stream_event_round_trip_serde() {
        let sample = json!({
            "type":"content_block_delta",
            "index":0,
            "delta":{
                "type":"input_json_delta",
                "partial_json":"{\"city\":\"Zurich\""
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
                "type": "overloaded_error",
                "message": "Please retry"
            }
        });
        let parsed: MessageStreamEvent = serde_json::from_value(sample.clone()).unwrap();
        let serialized = serde_json::to_value(parsed).unwrap();
        assert_eq!(serialized, sample);
    }
}
