use crate::{Result, types};
use chrono::Local;
use minijinja::{Environment, Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Map as JsonMap, Value as JsonValue};
use tokenizers::tokenizer::Tokenizer;

/// Tokenized model input and its stop tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub input_ids: Vec<i32>,
    pub stop_token_ids: Vec<i32>,
}

impl PreparedPrompt {
    /// Builds a prepared prompt from token ids and stop ids.
    pub fn new(input_ids: Vec<i32>, stop_token_ids: Vec<i32>) -> Self {
        Self {
            input_ids,
            stop_token_ids,
        }
    }

    /// Tokenizes a raw prompt string into model input ids.
    pub fn from_prompt(
        tokenizer: &Tokenizer,
        prompt: &str,
        stop_token_ids: &[i32],
    ) -> Result<Self> {
        let encoding = tokenizer.encode(prompt, true)?;
        let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        Ok(Self::new(input_ids, stop_token_ids.to_vec()))
    }

    /// Renders chat messages through the template and tokenizes the result.
    pub fn from_messages(
        tokenizer: &Tokenizer,
        chat_template: &str,
        tokenizer_config: &JsonValue,
        messages: &[types::Message],
        stop_token_ids: &[i32],
    ) -> Result<Self> {
        let prompt = render_chat_prompt(chat_template, tokenizer_config, messages)?;
        Self::from_prompt(tokenizer, &prompt, stop_token_ids)
    }
}

// Keep a single message as the source of truth and derive template-facing fields eagerly.
pub(crate) fn message_to_template_context(message: &types::Message) -> Result<Value> {
    let mut map = JsonMap::new();

    match message {
        types::Message::OpenAI(msg) => {
            map.insert("role".to_string(), JsonValue::String(msg.role.clone()));
            map.insert(
                "content".to_string(),
                JsonValue::String(match msg.content.as_ref() {
                    Some(content) => openai_content_to_template_string(content)?,
                    None => String::new(),
                }),
            );
        }
        types::Message::Anthropic(msg) => {
            map.insert("role".to_string(), JsonValue::String(msg.role.clone()));
            map.insert(
                "content".to_string(),
                JsonValue::String(anthropic_content_to_template_string(&msg.content)?),
            );
            map.insert(
                "content_blocks".to_string(),
                serde_json::to_value(anthropic_content_to_template_blocks(&msg.content))?,
            );
        }
    }

    Ok(Value::from_serialize(map))
}

fn render_chat_prompt(
    chat_template: &str,
    tokenizer_config: &JsonValue,
    messages: &[types::Message],
) -> Result<String> {
    let messages: Vec<_> = messages
        .iter()
        .map(message_to_template_context)
        .collect::<Result<_>>()?;
    render_chat_messages(chat_template, tokenizer_config, messages, false)
}

fn strftime_now(format_str: String) -> String {
    Local::now().format(&format_str).to_string()
}

fn render_chat_messages(
    chat_template: &str,
    tokenizer_config: &JsonValue,
    messages: Vec<Value>,
    enable_thinking: bool,
) -> Result<String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("strftime_now", strftime_now);
    env.add_template("chat", chat_template)?;
    let tmpl = env.get_template("chat")?;
    let bos_token = tokenizer_config
        .get("bos_token")
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    let prompt = tmpl.render(context!(
        messages => messages,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        bos_token => bos_token
    ))?;

    Ok(prompt)
}

// Used by the llm example app, clean up
pub fn render_chat_template(
    chat_template: &str,
    tokenizer_config: &serde_json::Value,
    prompt: &str,
    has_image: bool,
    enable_thinking: bool,
) -> Result<String> {
    let messages = if has_image {
        let content = vec![
            context!(type => "text", text => prompt),
            context!(type => "image"),
        ];
        vec![context!(role => "user",content => content)]
    } else {
        vec![context!(role => "user",content => prompt)]
    };
    render_chat_messages(chat_template, tokenizer_config, messages, enable_thinking)
}

// HF chat templates generally expect message.content to be a plain string.
// Wire-format content can be structured; flatten it to text for template rendering.
fn openai_content_to_template_string(content: &types::openai::MessageContent) -> Result<String> {
    match content {
        types::openai::MessageContent::Text(text) => Ok(text.clone()),
        types::openai::MessageContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    types::openai::ContentPart::Text { text } => out.push_str(text),
                }
            }
            Ok(out)
        }
    }
}

fn anthropic_content_to_template_string(
    content: &types::anthropic::MessageContent,
) -> Result<String> {
    match content {
        types::anthropic::MessageContent::Text(text) => Ok(text.clone()),
        types::anthropic::MessageContent::Blocks(blocks) => {
            anthropic_blocks_to_template_string(blocks)
        }
    }
}

fn anthropic_content_to_template_blocks(
    content: &types::anthropic::MessageContent,
) -> Vec<types::anthropic::ContentBlock> {
    match content {
        types::anthropic::MessageContent::Text(text) => {
            vec![types::anthropic::ContentBlock::Text { text: text.clone() }]
        }
        types::anthropic::MessageContent::Blocks(blocks) => blocks.clone(),
    }
}

fn anthropic_blocks_to_template_string(
    blocks: &[types::anthropic::ContentBlock],
) -> Result<String> {
    let mut out = String::new();
    for block in blocks {
        match block {
            types::anthropic::ContentBlock::Text { text } => out.push_str(text),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_message_template(message: &types::Message, template: &str) -> Result<String> {
        let mut env = Environment::new();
        env.add_template("test", template).unwrap();
        let context = message_to_template_context(message)?;
        env.get_template("test")
            .unwrap()
            .render(context!(message => context))
            .map_err(Into::into)
    }

    #[test]
    fn openai_parts_flatten_to_template_text() {
        let content = types::openai::MessageContent::Parts(vec![
            types::openai::ContentPart::Text {
                text: "hello".to_string(),
            },
            types::openai::ContentPart::Text {
                text: " world".to_string(),
            },
        ]);
        assert_eq!(
            openai_content_to_template_string(&content).unwrap(),
            "hello world".to_string()
        );
    }

    #[test]
    fn anthropic_blocks_flatten_to_template_text() {
        let content = types::anthropic::MessageContent::Blocks(vec![
            types::anthropic::ContentBlock::Text {
                text: "alpha".to_string(),
            },
            types::anthropic::ContentBlock::Text {
                text: "beta".to_string(),
            },
        ]);
        assert_eq!(
            anthropic_content_to_template_string(&content).unwrap(),
            "alphabeta".to_string()
        );
    }

    #[test]
    fn openai_message_context_exposes_flattened_content() {
        let message =
            types::Message::OpenAI(Box::new(types::openai::ChatMessage::assistant("hello")));
        let rendered = render_message_template(&message, "{{ message.content }}").unwrap();

        assert_eq!(rendered, "hello");
    }

    #[test]
    fn anthropic_message_context_exposes_structured_blocks_without_breaking_content() {
        let message = types::Message::Anthropic(types::anthropic::AnthropicMessage {
            role: "assistant".to_string(),
            content: types::anthropic::MessageContent::Blocks(vec![
                types::anthropic::ContentBlock::Text {
                    text: "alpha".to_string(),
                },
                types::anthropic::ContentBlock::Text {
                    text: "beta".to_string(),
                },
            ]),
        });

        let rendered = render_message_template(
            &message,
            "{{ message.content }}|{{ message.content_blocks|length }}|{{ message.content_blocks[1].text }}",
        )
        .unwrap();

        assert_eq!(rendered, "alphabeta|2|beta");
    }
}
