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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingPolicy {
    #[default]
    Default,
    Disabled,
    Enabled,
    Effort(openai::ReasoningEffort),
    BudgetTokens(u32),
}

impl ThinkingPolicy {
    pub(crate) fn enables_template_thinking(self) -> bool {
        !matches!(self, Self::Default | Self::Disabled)
    }
}

impl From<bool> for ThinkingPolicy {
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

impl From<Option<openai::ReasoningEffort>> for ThinkingPolicy {
    fn from(effort: Option<openai::ReasoningEffort>) -> Self {
        effort.map_or(Self::Default, |effort| match effort {
            openai::ReasoningEffort::None => Self::Disabled,
            effort => Self::Effort(effort),
        })
    }
}

impl From<Option<anthropic::ThinkingConfig>> for ThinkingPolicy {
    fn from(config: Option<anthropic::ThinkingConfig>) -> Self {
        config.map_or(Self::Default, |config| match config {
            anthropic::ThinkingConfig::Enabled { budget_tokens } => {
                Self::BudgetTokens(budget_tokens)
            }
            anthropic::ThinkingConfig::Disabled => Self::Disabled,
        })
    }
}

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
