//! A stripped-down version of ModelRunner from catgrad examples, intended for serving
use crate::Result;
use crate::legacy::models::utils::{Cache, Config, ModelBuilder, get_model};
use crate::legacy::nn::layers::{argmax, cast, reshape};
use crate::types;
use crate::utils::{
    from_json_str, get_model_chat_template, get_model_files, read_safetensors_multiple,
};
use catgrad_legacy::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{Dtype, NdArrayType, Shape, Var},
};
use minijinja::{Environment, Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tokenizers::tokenizer::Tokenizer;

/// Load model
pub struct ModelLoader {
    config: Config,
    model_paths: Vec<PathBuf>,
    tokenizer_path: PathBuf,
    chat_template: String,
    use_kv_cache: bool,
}

fn read_to_value<V: for<'a> serde::Deserialize<'a>>(path: impl AsRef<Path>) -> Result<V> {
    let config_str = &std::fs::read_to_string(path)?;
    from_json_str(config_str)
}

impl ModelLoader {
    pub fn new(model_name: &str, use_kv_cache: bool) -> Result<Self> {
        let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, "main")?;
        let chat_template = get_model_chat_template(model_name, "main")?;
        let config: Config = read_to_value(config_path)?;

        Ok(Self {
            config,
            model_paths,
            tokenizer_path,
            chat_template,
            use_kv_cache,
        })
    }
}

pub struct ModelTokenizer {
    pub tokenizer: Tokenizer,
    pub chat_template: String,
}

impl ModelTokenizer {
    fn new(tokenizer_path: PathBuf, chat_template: String) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // Modify the loaded chat template so it can be parsed by Minijinja
        // These non-standard tags are only used while training some models.
        let chat_template = chat_template
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");

        Ok(Self {
            tokenizer,
            chat_template,
        })
    }

    fn render_context(
        &self,
        messages: &[types::Message],
        tools: Option<&[types::ToolSpec]>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template("chat", &self.chat_template)?;
        let tmpl = env.get_template("chat")?;
        let message_context: Vec<_> = messages.iter().map(message_to_template_context).collect();
        Ok(tmpl.render(context!(
            messages => message_context,
            tools => tools,
            add_generation_prompt => true,
            enable_thinking => false
        ))?)
    }
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

// Keep a single message as the source of truth and derive template-facing fields lazily.
fn message_to_template_context(message: &types::Message) -> Value {
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

pub struct ModelRunner {
    pub tensors: Rc<HashMap<String, TaggedNdArray>>,
    pub state: Option<EvalState>,
    pub model: Box<dyn ModelBuilder>,
    pub use_kv_cache: bool,
    pub kv_cache: Vec<TaggedNdArray>,
    pub total_tokens: usize,
    pub config: Config,
    pub context: Vec<i32>,
    pub tokens: Vec<i32>, // new, unprocessed tokens
}

impl ModelRunner {
    fn initial_kv_cache(config: &Config) -> Vec<TaggedNdArray> {
        let v = TaggedNdArray::F32(NdArray::new_empty(Shape(vec![
            1,
            config.get_num_kv_heads(),
            0,
            config.get_head_dim(),
        ])));
        vec![v; 2 * config.num_hidden_layers]
    }

    pub fn new(
        model_paths: Vec<PathBuf>,
        config: Config,
        use_kv_cache: bool,
    ) -> Result<ModelRunner> {
        let arch = &config.architectures[0];

        let mut model = get_model(arch)?;
        let mut tensors = read_safetensors_multiple(model_paths, false)?;
        model.post_load(&mut tensors);

        let kv_cache = if use_kv_cache {
            Self::initial_kv_cache(&config)
        } else {
            vec![]
        };

        Ok(Self {
            tensors: Rc::new(tensors),
            state: None, // TODO?
            model,
            use_kv_cache,
            config,
            context: vec![],
            tokens: vec![],
            total_tokens: 0,
            kv_cache,
        })
    }

    fn next_token(&self, builder: &Builder, logits: Var) -> Var {
        let batches = logits.label.shape.0[0];
        let am = argmax(builder, logits);
        let am = reshape(builder, Shape(vec![batches, 1]), am);
        cast(builder, Dtype::I32, am)
    }

    fn build(&mut self, num_tokens: usize) {
        let batches = 1;
        let in_type = NdArrayType::new(Shape(vec![batches, num_tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let mut cache = Cache::init(
                builder,
                &self.config,
                self.total_tokens + num_tokens,
                self.use_kv_cache,
            );

            if self.use_kv_cache {
                // Shape of KV cache entries up to current sequence length
                let kv_cache_type = NdArrayType::new(
                    Shape(vec![
                        batches,
                        self.config.get_num_kv_heads(),
                        self.total_tokens,
                        self.config.get_head_dim(),
                    ]),
                    Dtype::F32,
                );

                for layer_id in 0..self.config.num_hidden_layers {
                    cache.in_kv_cache[layer_id] = (
                        Var::new(builder.clone(), kv_cache_type.clone()),
                        Var::new(builder.clone(), kv_cache_type.clone()),
                    );
                }
            }

            let result = self.model.build(
                builder,
                &self.config,
                &mut cache,
                self.total_tokens,
                x.clone(),
            );

            // Input most recently generated token and current kv_cache
            let mut sources_vec = vec![x];

            if self.use_kv_cache {
                for layer_id in 0..self.config.num_hidden_layers {
                    sources_vec.push(cache.in_kv_cache[layer_id].0.clone());
                    sources_vec.push(cache.in_kv_cache[layer_id].1.clone());
                }
            }

            // Output new token and updated kv_cache
            let new_token = self.next_token(builder, result);
            let mut targets_vec = vec![new_token];

            if self.use_kv_cache {
                let out_kv_cache: Vec<_> = cache
                    .out_kv_cache
                    .into_iter()
                    .flat_map(|(a, b)| vec![a, b])
                    .collect();

                targets_vec.extend(out_kv_cache);
            }

            (sources_vec, targets_vec)
        });

        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors));
    }

    // Make a forward pass given a list of tokens
    fn run(&mut self, x: &NdArray<i32>) -> TaggedNdArray {
        let mut sources = vec![x.clone().into()];

        if self.use_kv_cache {
            // Add kv_cache to the inputs
            sources.extend(self.kv_cache.clone());
        }

        let result = self.state.as_mut().unwrap().eval_with(sources);

        if self.use_kv_cache {
            // Save kv_cache to feed into next iteration
            self.kv_cache = result[1..].iter().map(|&tensor| tensor.clone()).collect();
        }

        result[0].clone()
    }

    fn generate(&mut self, tokens: Vec<i32>) -> Option<i32> {
        let num_tokens = tokens.len();
        let batches = 1;
        let input = NdArray::new(tokens, Shape(vec![batches, num_tokens / batches]));

        self.build(num_tokens);
        log::debug!("Model graph built...");
        let result = self.run(&input);

        let token = result.data()[0] as i32;
        if self.config.get_eos_token_ids().contains(&token) {
            // don't emit EOS tokens
            return None;
        }

        self.total_tokens += num_tokens;
        Some(token)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Trait impls

impl Iterator for ModelRunner {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        // get tokens to process; replace field with default
        let tokens = std::mem::take(&mut self.tokens);

        let next_token = self.generate(tokens);
        if let Some(token) = next_token {
            // next token to process
            self.tokens.push(token);
        }
        next_token
    }
}

fn longest_common_prefix<T: Eq>(x: &[T], y: &[T]) -> usize {
    let mut n = 0;
    for (a, b) in x.iter().zip(y.iter()) {
        if a == b {
            n += 1;
        } else {
            break;
        }
    }
    n
}

impl types::LM<i32> for ModelRunner {
    fn set_context(&mut self, context: Vec<i32>) {
        let n = longest_common_prefix(&self.context, &context);
        if n < self.context.len() {
            // PERFORMANCE: just *truncate* the context instead of fully resetting it.
            self.kv_cache = Self::initial_kv_cache(&self.config);
            self.total_tokens = 0;
        }
        self.context = context.to_vec();
        self.tokens = context.to_vec();
    }
}

impl types::Tokenizer<i32> for ModelTokenizer {
    fn encode(&self, content: String) -> Result<Vec<i32>> {
        let tokens = self.tokenizer.encode(content, true)?;
        Ok(tokens.get_ids().iter().map(|&x| x as i32).collect())
    }

    fn decode(&self, tokens: Vec<i32>) -> Result<String> {
        // TODO: efficiency?
        // TODO: support u32 in interpreter to remove try_into().unwrap().
        let tokens_u32: Vec<u32> = tokens.into_iter().map(|i| i.try_into().unwrap()).collect();
        Ok(self.tokenizer.decode(&tokens_u32, false)?)
    }
}

impl types::ChatTokenizer<i32> for ModelTokenizer {
    fn encode_messages(
        &self,
        messages: Vec<types::Message>,
        tools: Vec<types::ToolSpec>,
    ) -> Result<Vec<i32>> {
        let tools = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };
        // initialize context
        let content = self.render_context(&messages, tools)?;
        use types::Tokenizer;
        self.encode(content)
    }
}

impl types::Loader<i32, ModelRunner, ModelTokenizer> for ModelLoader {
    fn load_runner(&self) -> Result<ModelRunner> {
        ModelRunner::new(
            self.model_paths.clone(),
            self.config.clone(),
            self.use_kv_cache,
        )
    }

    fn load_tokenizer(&self) -> Result<ModelTokenizer> {
        ModelTokenizer::new(self.tokenizer_path.clone(), self.chat_template.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::{Environment, context};

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
        let message = types::Message::OpenAI(types::openai::ChatMessage::assistant("hello"));
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
}
