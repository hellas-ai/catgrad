//! OpenAI wire-format types.
use crate::LLMError;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
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
    ImageUrl { image_url: ImageUrl },
}

/// Image payload for OpenAI-style image_url content parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
}

/// Chat message.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[builder(default)]
    pub content: Option<MessageContent>,
    #[builder(default)]
    pub tool_calls: Option<Vec<JsonValue>>,
    #[builder(default)]
    pub tool_call_id: Option<String>,
    #[builder(default)]
    pub name: Option<String>,
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

/// Requested reasoning level for chat-completions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
}

/// Chat-completions request.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[builder(default)]
    pub tools: Option<Vec<JsonValue>>,
    #[builder(default)]
    pub max_tokens: Option<u32>,
    #[builder(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
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
            {"type":"image_url", "image_url":{"url":"https://example.com/cat.png","detail":"high"}},
            {"type":"text", "text":"B"}
        ]))
        .unwrap();
        assert_eq!(
            parts,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "A".to_string()
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/cat.png".to_string(),
                    },
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
                        {"type":"image_url","image_url":{"url":"file:///tmp/cat.png","detail":"low"}},
                        {"type":"text","text":" in one sentence"}
                    ],
                    "tool_call_id": "call_1",
                    "tool_calls": [],
                    "audio": {"id":"audio_123","data":"..."},
                    "refusal": "no"
                }
            ],
            "max_tokens": 64,
            "reasoning_effort": "medium",
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
                            ContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: "file:///tmp/cat.png".to_string(),
                                },
                            },
                            ContentPart::Text {
                                text: " in one sentence".to_string(),
                            },
                        ])))
                        .tool_calls(Some(vec![]))
                        .tool_call_id(Some("call_1".to_string()))
                        .build(),
                ])
                .tools(Some(vec![]))
                .max_tokens(Some(64))
                .reasoning_effort(Some(ReasoningEffort::Medium))
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

pub mod responses {
    use crate::LLMError;
    use serde::{Deserialize, Serialize};
    use serde_json::Value as JsonValue;
    use serde_with::skip_serializing_none;
    use typed_builder::TypedBuilder;

    /// Responses API request.
    #[skip_serializing_none]
    #[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq)]
    pub struct ResponseRequest {
        pub model: String,
        pub input: ResponseInput,
        #[builder(default)]
        pub instructions: Option<String>,
        #[builder(default)]
        pub tools: Option<Vec<JsonValue>>,
        #[builder(default)]
        pub max_output_tokens: Option<u32>,
        #[builder(default)]
        pub stream: Option<bool>,
    }

    /// Top-level response input payload.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(untagged)]
    pub enum ResponseInput {
        Text(String),
        Items(Vec<ResponseInputItem>),
    }

    /// A single item in a structured Responses API input array.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(untagged)]
    pub enum ResponseInputItem {
        Message(ResponseInputMessageItem),
        FunctionCallOutput(ResponseInputFunctionCallOutputItem),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct ResponseInputMessageItem {
        #[serde(rename = "type", default = "default_response_input_message_type")]
        pub item_type: ResponseInputMessageItemType,
        pub role: String,
        pub content: ResponseInputMessageContent,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct ResponseInputFunctionCallOutputItem {
        #[serde(rename = "type")]
        pub item_type: ResponseInputFunctionCallOutputItemType,
        pub call_id: String,
        pub output: String,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum ResponseInputMessageItemType {
        Message,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum ResponseInputFunctionCallOutputItemType {
        FunctionCallOutput,
    }

    fn default_response_input_message_type() -> ResponseInputMessageItemType {
        ResponseInputMessageItemType::Message
    }

    /// Content payload for a structured input message.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(untagged)]
    pub enum ResponseInputMessageContent {
        Text(String),
        Parts(Vec<ResponseInputContentPart>),
    }

    /// A supported content part in structured response input.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ResponseInputContentPart {
        InputText { text: String },
        OutputText { text: String },
        InputImage { image_url: super::ImageUrl },
    }

    impl ResponseRequest {
        pub fn to_messages(&self) -> Result<Vec<super::super::Message>, LLMError> {
            let mut messages = Vec::with_capacity(
                match &self.input {
                    ResponseInput::Text(_) => 1,
                    ResponseInput::Items(items) => items.len(),
                } + usize::from(self.instructions.is_some()),
            );

            if let Some(instructions) = &self.instructions {
                messages.push(super::super::Message::openai(super::ChatMessage::system(
                    instructions.clone(),
                )));
            }

            match &self.input {
                ResponseInput::Text(text) => {
                    messages.push(super::super::Message::openai(super::ChatMessage::user(
                        text.clone(),
                    )));
                }
                ResponseInput::Items(items) => {
                    for item in items {
                        let message: super::ChatMessage = item.clone().try_into()?;
                        messages.push(super::super::Message::openai(message));
                    }
                }
            }

            Ok(messages)
        }
    }

    impl TryFrom<ResponseInputItem> for super::ChatMessage {
        type Error = LLMError;

        fn try_from(value: ResponseInputItem) -> Result<Self, Self::Error> {
            match value {
                ResponseInputItem::Message(ResponseInputMessageItem { role, content, .. }) => {
                    Ok(super::ChatMessage::builder()
                        .role(role)
                        .content(Some(content.try_into()?))
                        .build())
                }
                ResponseInputItem::FunctionCallOutput(ResponseInputFunctionCallOutputItem {
                    call_id,
                    output,
                    ..
                }) => Ok(super::ChatMessage::builder()
                    .role("tool".to_string())
                    .content(Some(super::MessageContent::Text(output)))
                    .tool_call_id(Some(call_id))
                    .build()),
            }
        }
    }

    impl TryFrom<ResponseInputMessageContent> for super::MessageContent {
        type Error = LLMError;

        fn try_from(value: ResponseInputMessageContent) -> Result<Self, Self::Error> {
            match value {
                ResponseInputMessageContent::Text(text) => Ok(Self::Text(text)),
                ResponseInputMessageContent::Parts(parts) => Ok(Self::Parts(
                    parts
                        .into_iter()
                        .map(TryInto::try_into)
                        .collect::<Result<Vec<_>, _>>()?,
                )),
            }
        }
    }

    impl TryFrom<ResponseInputContentPart> for super::ContentPart {
        type Error = LLMError;

        fn try_from(value: ResponseInputContentPart) -> Result<Self, Self::Error> {
            match value {
                ResponseInputContentPart::InputText { text }
                | ResponseInputContentPart::OutputText { text } => Ok(Self::Text { text }),
                ResponseInputContentPart::InputImage { image_url } => {
                    Ok(Self::ImageUrl { image_url })
                }
            }
        }
    }

    /// Overall response status.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum ResponseStatus {
        Queued,
        InProgress,
        Completed,
    }

    /// Responses API token usage details.
    #[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
    pub struct ResponseUsage {
        pub input_tokens: u32,
        pub output_tokens: u32,
        pub total_tokens: u32,
    }

    impl ResponseUsage {
        pub fn from_counts(input_tokens: u32, output_tokens: u32) -> Self {
            Self::builder()
                .input_tokens(input_tokens)
                .output_tokens(output_tokens)
                .total_tokens(input_tokens + output_tokens)
                .build()
        }
    }

    /// Text content item in an assistant response message.
    #[skip_serializing_none]
    #[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
    pub struct ResponseOutputText {
        #[serde(rename = "type")]
        pub content_type: String,
        pub text: String,
        #[builder(default)]
        pub annotations: Option<Vec<JsonValue>>,
    }

    /// Assistant message item in the response output array.
    #[skip_serializing_none]
    #[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
    pub struct ResponseOutputMessage {
        pub id: String,
        #[serde(rename = "type")]
        pub item_type: String,
        pub status: ResponseStatus,
        pub role: String,
        pub content: Vec<ResponseOutputText>,
        #[builder(default)]
        pub annotations: Option<Vec<JsonValue>>,
    }

    /// Responses API response.
    #[skip_serializing_none]
    #[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder, PartialEq, Eq)]
    pub struct Response {
        pub id: String,
        pub object: String,
        pub created_at: i64,
        pub status: ResponseStatus,
        pub model: String,
        pub output: Vec<ResponseOutputMessage>,
        #[builder(default)]
        pub usage: Option<ResponseUsage>,
    }

    /// Streaming event payloads for the Responses API.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(tag = "type")]
    pub enum ResponseStreamEvent {
        #[serde(rename = "response.created")]
        Created {
            sequence_number: u64,
            response: Response,
        },
        #[serde(rename = "response.in_progress")]
        InProgress {
            sequence_number: u64,
            response: Response,
        },
        #[serde(rename = "response.output_item.added")]
        OutputItemAdded {
            sequence_number: u64,
            output_index: u32,
            item: ResponseOutputMessage,
        },
        #[serde(rename = "response.content_part.added")]
        ContentPartAdded {
            sequence_number: u64,
            item_id: String,
            output_index: u32,
            content_index: u32,
            part: ResponseOutputText,
        },
        #[serde(rename = "response.output_text.delta")]
        OutputTextDelta {
            sequence_number: u64,
            item_id: String,
            output_index: u32,
            content_index: u32,
            delta: String,
        },
        #[serde(rename = "response.output_text.done")]
        OutputTextDone {
            sequence_number: u64,
            item_id: String,
            output_index: u32,
            content_index: u32,
            text: String,
        },
        #[serde(rename = "response.content_part.done")]
        ContentPartDone {
            sequence_number: u64,
            item_id: String,
            output_index: u32,
            content_index: u32,
            part: ResponseOutputText,
        },
        #[serde(rename = "response.output_item.done")]
        OutputItemDone {
            sequence_number: u64,
            output_index: u32,
            item: ResponseOutputMessage,
        },
        #[serde(rename = "response.completed")]
        Completed {
            sequence_number: u64,
            response: Response,
        },
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::types::Message;
        use crate::types::openai::{ChatMessage, ContentPart, ImageUrl, MessageContent};
        use serde_json::json;

        #[test]
        fn response_request_parses_minimal_string_input() {
            let parsed: ResponseRequest = serde_json::from_value(json!({
                "model": "gpt-4.1-mini",
                "input": "Hello"
            }))
            .unwrap();

            assert_eq!(
                parsed,
                ResponseRequest::builder()
                    .model("gpt-4.1-mini".to_string())
                    .input(ResponseInput::Text("Hello".to_string()))
                    .build()
            );
        }

        #[test]
        fn response_request_supports_message_list_instructions_and_tools() {
            let parsed: ResponseRequest = serde_json::from_value(json!({
                "model": "gpt-4.1-mini",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {"type": "input_text", "text": "You are helpful."},
                            {"type": "input_text", "text": "Answer briefly."}
                        ]
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Tell a story about a unicorn."}
                        ]
                    }
                ],
                "instructions": "You are a nice LLM",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_weather",
                        "parameters": {"type": "object"}
                    }
                ],
                "max_output_tokens": 64,
                "stream": false,
                "temperature": 0.2
            }))
            .unwrap();

            assert_eq!(
                parsed,
                ResponseRequest::builder()
                    .model("gpt-4.1-mini".to_string())
                    .input(ResponseInput::Items(vec![
                        ResponseInputItem::Message(ResponseInputMessageItem {
                            item_type: ResponseInputMessageItemType::Message,
                            role: "developer".to_string(),
                            content: ResponseInputMessageContent::Parts(vec![
                                ResponseInputContentPart::InputText {
                                    text: "You are helpful.".to_string(),
                                },
                                ResponseInputContentPart::InputText {
                                    text: "Answer briefly.".to_string(),
                                },
                            ]),
                        }),
                        ResponseInputItem::Message(ResponseInputMessageItem {
                            item_type: ResponseInputMessageItemType::Message,
                            role: "user".to_string(),
                            content: ResponseInputMessageContent::Parts(vec![
                                ResponseInputContentPart::InputText {
                                    text: "Tell a story about a unicorn.".to_string(),
                                },
                            ]),
                        }),
                    ]))
                    .instructions(Some("You are a nice LLM".to_string()))
                    .tools(Some(vec![json!({
                        "type": "function",
                        "name": "lookup_weather",
                        "parameters": {"type": "object"}
                    })]))
                    .max_output_tokens(Some(64))
                    .stream(Some(false))
                    .build()
            );
        }

        #[test]
        fn response_round_trips_through_serde() {
            let sample = json!({
                "id": "resp-1",
                "object": "response",
                "created_at": 1234567890,
                "status": "completed",
                "model": "gpt-4.1-mini",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hello world"
                            }
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "total_tokens": 7
                }
            });

            let parsed: Response = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_created_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-1",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "queued",
                    "model": "gpt-4.1-mini",
                    "output": []
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_in_progress_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.in_progress",
                "sequence_number": 1,
                "response": {
                    "id": "resp-1",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "in_progress",
                    "model": "gpt-4.1-mini",
                    "output": []
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_output_item_added_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "id": "msg-1",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                    "annotations": []
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_content_part_added_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.content_part.added",
                "sequence_number": 3,
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": []
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_output_text_delta_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.output_text.delta",
                "sequence_number": 4,
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hello"
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_output_text_done_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.output_text.done",
                "sequence_number": 5,
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "text": "Hello world"
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_content_part_done_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.content_part.done",
                "sequence_number": 6,
                "item_id": "msg-1",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "Hello world",
                    "annotations": []
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_output_item_done_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.output_item.done",
                "sequence_number": 7,
                "output_index": 0,
                "item": {
                    "id": "msg-1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello world",
                            "annotations": []
                        }
                    ],
                    "annotations": []
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_completed_event_round_trips_through_serde() {
            let sample = json!({
                "type": "response.completed",
                "sequence_number": 8,
                "response": {
                    "id": "resp-1",
                    "object": "response",
                    "created_at": 1234567890,
                    "status": "completed",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "id": "msg-1",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Hello world",
                                    "annotations": []
                                }
                            ],
                            "annotations": []
                        }
                    ],
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "total_tokens": 7
                    }
                }
            });

            let parsed: ResponseStreamEvent = serde_json::from_value(sample.clone()).unwrap();
            let serialized = serde_json::to_value(parsed).unwrap();
            assert_eq!(serialized, sample);
        }

        #[test]
        fn response_request_accepts_output_text_input_parts() {
            let request = ResponseRequest::builder()
                .model("gpt-4.1-mini".to_string())
                .input(ResponseInput::Items(vec![ResponseInputItem::Message(
                    ResponseInputMessageItem {
                        item_type: ResponseInputMessageItemType::Message,
                        role: "assistant".to_string(),
                        content: ResponseInputMessageContent::Parts(vec![
                            ResponseInputContentPart::OutputText {
                                text: "Previous answer".to_string(),
                            },
                        ]),
                    },
                )]))
                .build();

            let messages = request.to_messages().unwrap();
            assert_eq!(
                messages,
                vec![Message::openai(
                    ChatMessage::builder()
                        .role("assistant".to_string())
                        .content(Some(MessageContent::Parts(vec![ContentPart::Text {
                            text: "Previous answer".to_string(),
                        }])))
                        .build(),
                )]
            );
        }

        #[test]
        fn response_request_accepts_message_items_without_type() {
            let parsed: ResponseRequest = serde_json::from_value(json!({
                "model": "gpt-4.1-mini",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Hello"}
                        ]
                    }
                ]
            }))
            .unwrap();

            assert_eq!(
                parsed,
                ResponseRequest::builder()
                    .model("gpt-4.1-mini".to_string())
                    .input(ResponseInput::Items(vec![ResponseInputItem::Message(
                        ResponseInputMessageItem {
                            item_type: ResponseInputMessageItemType::Message,
                            role: "user".to_string(),
                            content: ResponseInputMessageContent::Parts(vec![
                                ResponseInputContentPart::InputText {
                                    text: "Hello".to_string(),
                                },
                            ]),
                        },
                    )]))
                    .build()
            );
        }

        #[test]
        fn response_request_accepts_input_image_parts() {
            let parsed: ResponseRequest = serde_json::from_value(json!({
                "model": "gpt-4.1-mini",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {"type": "input_image", "image_url": {"url": "https://example.com/image.png"}}
                        ]
                    }
                ]
            }))
            .unwrap();

            assert_eq!(
                parsed,
                ResponseRequest::builder()
                    .model("gpt-4.1-mini".to_string())
                    .input(ResponseInput::Items(vec![ResponseInputItem::Message(
                        ResponseInputMessageItem {
                            item_type: ResponseInputMessageItemType::Message,
                            role: "user".to_string(),
                            content: ResponseInputMessageContent::Parts(vec![
                                ResponseInputContentPart::InputText {
                                    text: "Describe this image".to_string(),
                                },
                                ResponseInputContentPart::InputImage {
                                    image_url: ImageUrl {
                                        url: "https://example.com/image.png".to_string(),
                                    },
                                },
                            ]),
                        },
                    )]))
                    .build()
            );

            let messages = parsed.to_messages().unwrap();
            assert_eq!(
                messages,
                vec![Message::openai(
                    ChatMessage::builder()
                        .role("user".to_string())
                        .content(Some(MessageContent::Parts(vec![
                            ContentPart::Text {
                                text: "Describe this image".to_string(),
                            },
                            ContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: "https://example.com/image.png".to_string(),
                                },
                            },
                        ])))
                        .build(),
                )]
            );
        }

        #[test]
        fn response_request_converts_to_chat_messages() {
            let request = ResponseRequest::builder()
                .model("gpt-4.1-mini".to_string())
                .input(ResponseInput::Items(vec![ResponseInputItem::Message(
                    ResponseInputMessageItem {
                        item_type: ResponseInputMessageItemType::Message,
                        role: "user".to_string(),
                        content: ResponseInputMessageContent::Parts(vec![
                            ResponseInputContentPart::InputText {
                                text: "Hello".to_string(),
                            },
                            ResponseInputContentPart::InputText {
                                text: " world".to_string(),
                            },
                        ]),
                    },
                )]))
                .instructions(Some("Be concise.".to_string()))
                .build();

            let messages = request.to_messages().unwrap();
            assert_eq!(
                messages,
                vec![
                    Message::openai(ChatMessage::system("Be concise.")),
                    Message::openai(
                        ChatMessage::builder()
                            .role("user".to_string())
                            .content(Some(MessageContent::Parts(vec![
                                ContentPart::Text {
                                    text: "Hello".to_string()
                                },
                                ContentPart::Text {
                                    text: " world".to_string()
                                },
                            ])))
                            .build(),
                    ),
                ]
            );
        }

        #[test]
        fn response_request_supports_function_call_output_items() {
            let parsed: ResponseRequest = serde_json::from_value(json!({
                "model": "gpt-4.1-mini",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_123",
                        "output": "{\"result\":\"ok\"}"
                    }
                ]
            }))
            .unwrap();

            assert_eq!(
                parsed,
                ResponseRequest::builder()
                    .model("gpt-4.1-mini".to_string())
                    .input(ResponseInput::Items(vec![
                        ResponseInputItem::FunctionCallOutput(
                            ResponseInputFunctionCallOutputItem {
                                item_type:
                                    ResponseInputFunctionCallOutputItemType::FunctionCallOutput,
                                call_id: "call_123".to_string(),
                                output: "{\"result\":\"ok\"}".to_string(),
                            }
                        )
                    ]))
                    .build()
            );

            let messages = parsed.to_messages().unwrap();
            assert_eq!(
                messages,
                vec![Message::openai(
                    ChatMessage::builder()
                        .role("tool".to_string())
                        .content(Some(MessageContent::Text(
                            "{\"result\":\"ok\"}".to_string()
                        )))
                        .tool_call_id(Some("call_123".to_string()))
                        .build(),
                )]
            );
        }
    }
}
