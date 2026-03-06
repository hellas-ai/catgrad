//! LLM serving interfaces and API wire-format types.
//!
//! ## Spec coverage
//! - OpenAI chat-completions:
//!   - message content as text or typed content parts
//!   - tool definitions/choices, assistant tool calls, and streaming deltas
//! - Anthropic messages:
//!   - message/system content as string or typed blocks
//!   - tool metadata and SSE event stream types
//! - Plain completions:
//!   - prompt unions (string(s) and tokenized forms)
//!   - non-stream and stream chunk response shapes
//!
//! ## Notes
//! - JSON Schema payloads use `schemars::Schema`.
//! - Arbitrary runtime JSON arguments use `serde_json::Value`.
use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::skip_serializing_none;

pub mod anthropic;
pub mod openai;
pub mod plain;

pub type JsonData = Value;
pub type JsonMap = Map<String, JsonData>;
pub type JsonSchema = schemars::Schema;

/// Internal, protocol-neutral tool schema used by tokenizer/template rendering.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolFunctionSpec {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<JsonSchema>,
    pub strict: Option<bool>,
}

/// Internal, protocol-neutral custom tool schema.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCustomSpec {
    pub name: String,
    pub description: Option<String>,
}

/// Internal tool definition (independent from OpenAI/Anthropic wire structs).
///
/// Serialized shape intentionally matches OpenAI tool objects because most chat templates expect:
/// `{ "type": "function", "function": { ... } }`.
#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolSpec {
    Function { function: ToolFunctionSpec },
    Custom { custom: ToolCustomSpec },
}

impl From<openai::ToolDefinition> for ToolSpec {
    fn from(value: openai::ToolDefinition) -> Self {
        match value {
            openai::ToolDefinition::Function { function } => ToolSpec::Function {
                function: ToolFunctionSpec {
                    name: function.name,
                    description: function.description,
                    parameters: function.parameters,
                    strict: function.strict,
                },
            },
            openai::ToolDefinition::Custom { custom } => ToolSpec::Custom {
                custom: ToolCustomSpec {
                    name: custom.name,
                    description: custom.description,
                },
            },
        }
    }
}

impl From<anthropic::ToolDefinition> for ToolSpec {
    fn from(value: anthropic::ToolDefinition) -> Self {
        ToolSpec::Function {
            function: ToolFunctionSpec {
                name: value.name,
                description: value.description,
                parameters: Some(value.input_schema),
                strict: None,
            },
        }
    }
}

/// Internal message container used by chat tokenizers.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    OpenAI(Box<openai::ChatMessage>),
    Anthropic(anthropic::AnthropicMessage),
}

impl Message {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self::OpenAI(Box::new(openai::ChatMessage {
            role: role.into(),
            content: Some(openai::MessageContent::Text(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            function_call: None,
            refusal: None,
            audio: None,
        }))
    }
}

/// A language model has a settable internal context from which it can generate new tokens.
pub trait LM<T>: Iterator<Item = T> {
    fn set_context(&mut self, context: Vec<T>);

    fn complete(&mut self, context: Vec<T>) -> impl Iterator<Item = T> {
        self.set_context(context);
        self
    }
}

/// A *loader* is conceptually a pair of language model and supporting code (tokenizers, ChatML
/// templates, etc.)
pub trait Loader<Token, L: LM<Token>, T: Tokenizer<Token>> {
    fn load_runner(&self) -> Result<L>;
    fn load_tokenizer(&self) -> Result<T>;
}

/// A [`Tokenizer`] translates between tokens and strings
pub trait Tokenizer<Token> {
    fn encode(&self, content: String) -> Result<Vec<Token>>;
    fn decode(&self, tokens: Vec<Token>) -> Result<String>;
}

/// A [`Tokenizer`] which is aware of message structure
pub trait ChatTokenizer<Token> {
    /// `tools` is an internal protocol-neutral schema. Convert wire-format tool payloads (OpenAI,
    /// Anthropic, etc.) into [`ToolSpec`] before calling.
    fn encode_messages(&self, messages: Vec<Message>, tools: Vec<ToolSpec>) -> Result<Vec<Token>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::json_schema;

    #[test]
    fn openai_tool_definition_converts_to_tool_spec() {
        let openai_tool = openai::ToolDefinition::Function {
            function: openai::FunctionDefinition::builder()
                .name("lookup".to_string())
                .description(Some("lookup data".to_string()))
                .build(),
        };
        let spec: ToolSpec = openai_tool.into();
        assert!(matches!(spec, ToolSpec::Function { .. }));
    }

    #[test]
    fn anthropic_tool_definition_converts_to_tool_spec() {
        let anthropic_tool = anthropic::ToolDefinition::builder()
            .name("lookup".to_string())
            .input_schema(json_schema!({
                "type": "object",
            }))
            .build();
        let spec: ToolSpec = anthropic_tool.into();
        assert!(matches!(spec, ToolSpec::Function { .. }));
    }
}
