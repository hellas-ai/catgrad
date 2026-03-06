use crate::error::LLMError;
use crate::{Result, types};
use minijinja::{Environment, Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use tokenizers::tokenizer::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub prompt: String,
    pub expected_output: ExpectedAssistantOutput,
    pub stop_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPrompt {
    pub input_ids: Vec<i32>,
    pub prompt_tokens: u32,
    pub expected_output: ExpectedAssistantOutput,
    pub stop_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedAssistantOutput {
    Text,
    ToolCalls { syntax: ToolCallSyntax },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCallSyntax {
    QwenXml,
}

const OPENAI_TEMPLATE_MESSAGE_FIELDS: &[&str] = &[
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "function_call",
    "refusal",
    "audio",
    "content_parts",
];
const ANTHROPIC_TEMPLATE_MESSAGE_FIELDS: &[&str] = &["role", "content", "content_blocks"];

pub fn detect_tool_call_syntax(chat_template: &str) -> Option<ToolCallSyntax> {
    if chat_template.contains("<tool_call>") && chat_template.contains("</tool_call>") {
        Some(ToolCallSyntax::QwenXml)
    } else {
        None
    }
}

pub(crate) fn render_messages(
    chat_template: &str,
    stop_token_ids: &[i32],
    messages: &[types::Message],
    tools: &[types::ToolSpec],
) -> Result<RenderedPrompt> {
    let expected_output = detect_expected_output(chat_template, !tools.is_empty())?;

    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_template("chat", chat_template)?;
    let tmpl = env.get_template("chat")?;
    let message_context: Vec<_> = messages.iter().map(message_to_template_context).collect();
    let tools = if tools.is_empty() { None } else { Some(tools) };
    let prompt = tmpl.render(context!(
        messages => message_context,
        tools => tools,
        add_generation_prompt => true,
        enable_thinking => false
    ))?;

    Ok(RenderedPrompt {
        prompt,
        expected_output,
        stop_token_ids: stop_token_ids.to_vec(),
    })
}

pub(crate) fn prepare_messages(
    tokenizer: &Tokenizer,
    chat_template: &str,
    stop_token_ids: &[i32],
    messages: &[types::Message],
    tools: &[types::ToolSpec],
) -> Result<PreparedPrompt> {
    let rendered = render_messages(chat_template, stop_token_ids, messages, tools)?;
    prepare_text_with_expected_output(
        tokenizer,
        &rendered.prompt,
        rendered.expected_output,
        rendered.stop_token_ids,
    )
}

pub(crate) fn prepare_text(
    tokenizer: &Tokenizer,
    stop_token_ids: &[i32],
    prompt: &str,
) -> Result<PreparedPrompt> {
    prepare_text_with_expected_output(
        tokenizer,
        prompt,
        ExpectedAssistantOutput::Text,
        stop_token_ids.to_vec(),
    )
}

fn prepare_text_with_expected_output(
    tokenizer: &Tokenizer,
    prompt: &str,
    expected_output: ExpectedAssistantOutput,
    stop_token_ids: Vec<i32>,
) -> Result<PreparedPrompt> {
    let encoding = tokenizer.encode(prompt, true)?;
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    Ok(PreparedPrompt {
        prompt_tokens: input_ids.len() as u32,
        input_ids,
        expected_output,
        stop_token_ids,
    })
}

fn detect_expected_output(chat_template: &str, has_tools: bool) -> Result<ExpectedAssistantOutput> {
    if !has_tools {
        return Ok(ExpectedAssistantOutput::Text);
    }

    let Some(syntax) = detect_tool_call_syntax(chat_template) else {
        return Err(LLMError::UnsupportedTemplateFeature(
            "Tools were provided, but the active chat template does not expose a supported tool-call syntax.".to_string(),
        ));
    };

    Ok(ExpectedAssistantOutput::ToolCalls { syntax })
}

// Keep a single message as the source of truth and derive template-facing fields lazily.
pub(crate) fn message_to_template_context(message: &types::Message) -> Value {
    Value::make_object_map(
        message.clone(),
        enumerate_template_message_fields,
        lookup_template_message_field,
    )
}

fn enumerate_template_message_fields(
    message: &types::Message,
) -> Box<dyn Iterator<Item = Value> + Send + Sync + '_> {
    let fields = match message {
        types::Message::OpenAI(_) => OPENAI_TEMPLATE_MESSAGE_FIELDS,
        types::Message::Anthropic(_) => ANTHROPIC_TEMPLATE_MESSAGE_FIELDS,
    };
    Box::new(fields.iter().copied().map(Value::from))
}

fn lookup_template_message_field(message: &types::Message, key: &Value) -> Option<Value> {
    let key = key.as_str()?;
    match message {
        types::Message::OpenAI(msg) => lookup_openai_message_field(msg, key),
        types::Message::Anthropic(msg) => lookup_anthropic_message_field(msg, key),
    }
}

fn lookup_openai_message_field(msg: &types::openai::ChatMessage, key: &str) -> Option<Value> {
    match key {
        "role" => Some(Value::from(msg.role.clone())),
        "content" => Some(Value::from(openai_content_to_template_string(
            msg.content.as_ref(),
        ))),
        "name" => msg.name.clone().map(Value::from),
        "tool_call_id" => msg.tool_call_id.clone().map(Value::from),
        "tool_calls" => msg.tool_calls.as_ref().map(Value::from_serialize),
        "function_call" => msg.function_call.as_ref().map(Value::from_serialize),
        "refusal" => msg.refusal.clone().map(Value::from),
        "audio" => msg.audio.as_ref().map(Value::from_serialize),
        "content_parts" => Some(Value::from_serialize(openai_content_to_template_parts(
            msg.content.as_ref(),
        ))),
        _ => None,
    }
}

fn lookup_anthropic_message_field(
    msg: &types::anthropic::AnthropicMessage,
    key: &str,
) -> Option<Value> {
    match key {
        "role" => Some(Value::from(msg.role.clone())),
        "content" => Some(Value::from(anthropic_content_to_template_string(
            &msg.content,
        ))),
        "content_blocks" => Some(Value::from_serialize(anthropic_content_to_template_blocks(
            &msg.content,
        ))),
        _ => None,
    }
}

// HF chat templates generally expect message.content to be a plain string.
// Wire-format content can be structured; flatten it to text for template rendering.
fn openai_content_to_template_string(content: Option<&types::openai::MessageContent>) -> String {
    match content {
        None => String::new(),
        Some(types::openai::MessageContent::Text(text)) => text.clone(),
        Some(types::openai::MessageContent::Parts(parts)) => parts
            .iter()
            .map(|part| match part {
                types::openai::ContentPart::Text { text } => text.clone(),
                types::openai::ContentPart::Refusal { refusal } => refusal.clone(),
                types::openai::ContentPart::ImageUrl { .. }
                | types::openai::ContentPart::InputAudio { .. }
                | types::openai::ContentPart::File { .. } => String::new(),
            })
            .collect(),
    }
}

fn openai_content_to_template_parts(
    content: Option<&types::openai::MessageContent>,
) -> Vec<types::openai::ContentPart> {
    match content {
        None => Vec::new(),
        Some(types::openai::MessageContent::Text(text)) => {
            vec![types::openai::ContentPart::Text { text: text.clone() }]
        }
        Some(types::openai::MessageContent::Parts(parts)) => parts.clone(),
    }
}

fn anthropic_content_to_template_string(content: &types::anthropic::MessageContent) -> String {
    match content {
        types::anthropic::MessageContent::Text(text) => text.clone(),
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
            vec![types::anthropic::ContentBlock::Text {
                text: text.clone(),
                citations: None,
                cache_control: None,
            }]
        }
        types::anthropic::MessageContent::Blocks(blocks) => blocks.clone(),
    }
}

fn anthropic_blocks_to_template_string(blocks: &[types::anthropic::ContentBlock]) -> String {
    blocks
        .iter()
        .map(|block| match block {
            types::anthropic::ContentBlock::Text { text, .. } => text.clone(),
            types::anthropic::ContentBlock::ToolResult { content, .. } => content
                .as_ref()
                .map(anthropic_content_to_template_string)
                .unwrap_or_default(),
            types::anthropic::ContentBlock::Document { source, .. } => match source {
                types::anthropic::DocumentSource::Text { text } => text.clone(),
                types::anthropic::DocumentSource::Content { content } => {
                    anthropic_blocks_to_template_string(content)
                }
                types::anthropic::DocumentSource::Base64 { .. }
                | types::anthropic::DocumentSource::Url { .. }
                | types::anthropic::DocumentSource::File { .. } => String::new(),
            },
            types::anthropic::ContentBlock::Thinking { thinking, .. } => thinking.clone(),
            types::anthropic::ContentBlock::Image { .. }
            | types::anthropic::ContentBlock::ToolUse { .. }
            | types::anthropic::ContentBlock::RedactedThinking { .. } => String::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_message_template(message: &types::Message, template: &str) -> String {
        let mut env = Environment::new();
        env.add_template("test", template).unwrap();
        env.get_template("test")
            .unwrap()
            .render(context!(message => message_to_template_context(message)))
            .unwrap()
    }

    #[test]
    fn openai_parts_flatten_to_template_text() {
        let content = types::openai::MessageContent::Parts(vec![
            types::openai::ContentPart::Text {
                text: "hello".to_string(),
            },
            types::openai::ContentPart::ImageUrl {
                image_url: types::openai::ImageUrlPart::builder()
                    .url("https://example.com/img.png".to_string())
                    .build(),
            },
            types::openai::ContentPart::Refusal {
                refusal: "no".to_string(),
            },
        ]);
        assert_eq!(
            openai_content_to_template_string(Some(&content)),
            "hellono".to_string()
        );
    }

    #[test]
    fn anthropic_blocks_flatten_to_template_text() {
        let content = types::anthropic::MessageContent::Blocks(vec![
            types::anthropic::ContentBlock::Text {
                text: "alpha".to_string(),
                citations: None,
                cache_control: None,
            },
            types::anthropic::ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: Some(types::anthropic::MessageContent::Text("beta".to_string())),
                is_error: None,
                cache_control: None,
            },
            types::anthropic::ContentBlock::Document {
                source: types::anthropic::DocumentSource::Text {
                    text: "gamma".to_string(),
                },
                title: None,
                context: None,
                citations: None,
                cache_control: None,
            },
            types::anthropic::ContentBlock::Thinking {
                thinking: "delta".to_string(),
                signature: None,
            },
        ]);
        assert_eq!(
            anthropic_content_to_template_string(&content),
            "alphabetagammadelta".to_string()
        );
    }

    #[test]
    fn openai_message_context_exposes_normalized_parts() {
        let message =
            types::Message::OpenAI(Box::new(types::openai::ChatMessage::assistant("hello")));
        let rendered = render_message_template(
            &message,
            "{{ message.content }}|{{ message.content_parts|length }}|{{ message.content_parts[0].text }}",
        );

        assert_eq!(rendered, "hello|1|hello");
    }

    #[test]
    fn anthropic_message_context_exposes_structured_blocks_without_breaking_content() {
        let message = types::Message::Anthropic(types::anthropic::AnthropicMessage {
            role: "assistant".to_string(),
            content: types::anthropic::MessageContent::Blocks(vec![
                types::anthropic::ContentBlock::Text {
                    text: "alpha".to_string(),
                    citations: None,
                    cache_control: None,
                },
                types::anthropic::ContentBlock::ToolUse {
                    id: "toolu_1".to_string(),
                    name: "lookup".to_string(),
                    input: types::JsonMap::new(),
                },
                types::anthropic::ContentBlock::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: Some(types::anthropic::MessageContent::Text("beta".to_string())),
                    is_error: None,
                    cache_control: None,
                },
                types::anthropic::ContentBlock::Thinking {
                    thinking: "delta".to_string(),
                    signature: None,
                },
            ]),
        });

        let rendered = render_message_template(
            &message,
            "{{ message.content }}|{{ message.content_blocks|length }}|{{ message.content_blocks[1].name }}|{{ message.content_blocks[2].tool_use_id }}|{{ message.content_blocks[3].thinking }}",
        );

        assert_eq!(rendered, "alphabetadelta|4|lookup|toolu_1|delta");
    }

    #[test]
    fn detects_qwen_tool_call_syntax() {
        let template = "{{ '<tool_call>' }}{{ '</tool_call>' }}";
        assert_eq!(
            detect_tool_call_syntax(template),
            Some(ToolCallSyntax::QwenXml)
        );
    }

    #[test]
    fn tools_require_supported_template_syntax() {
        let err = render_messages(
            "{{ messages[0].content }}",
            &[],
            &[types::Message::text("user", "hello")],
            &[types::ToolSpec::Function {
                function: types::ToolFunctionSpec {
                    name: "lookup".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            }],
        )
        .unwrap_err();

        assert!(matches!(err, LLMError::UnsupportedTemplateFeature(_)));
    }
}
