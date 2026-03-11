//! LLM serving interfaces and API wire-format types.
//!
//! ## Spec coverage
//! - OpenAI chat-completions:
//!   - message content as text or supported typed content parts
//! - Anthropic messages:
//!   - message/system content as string or typed text blocks
//! - Plain completions:
//!   - string prompt requests
//!   - non-stream and stream chunk response shapes
//!
pub mod anthropic;
pub mod openai;
pub mod plain;

/// Internal message container used by chat tokenizers.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    OpenAI(Box<openai::ChatMessage>),
    Anthropic(anthropic::AnthropicMessage),
}

impl Message {
    pub fn openai(message: impl Into<openai::ChatMessage>) -> Self {
        Self::OpenAI(Box::new(message.into()))
    }

    pub fn anthropic(message: impl Into<anthropic::AnthropicMessage>) -> Self {
        Self::Anthropic(message.into())
    }
}
