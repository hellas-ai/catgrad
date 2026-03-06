//! A stripped-down version of ModelRunner from catgrad examples, intended for serving
use crate::Result;
use crate::legacy::models::utils::{Cache, Config, ModelBuilder, get_model};
use crate::legacy::nn::layers::{argmax, cast, reshape};
use crate::prompt;
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
    pub eos_token_ids: Vec<i32>,
}

impl ModelTokenizer {
    fn new(
        tokenizer_path: PathBuf,
        chat_template: String,
        eos_token_ids: Vec<i32>,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // Modify the loaded chat template so it can be parsed by Minijinja
        // These non-standard tags are only used while training some models.
        let chat_template = chat_template
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");

        Ok(Self {
            tokenizer,
            chat_template,
            eos_token_ids,
        })
    }

    pub fn stop_token_ids(&self) -> &[i32] {
        &self.eos_token_ids
    }

    pub fn render_messages(
        &self,
        messages: &[types::Message],
        tools: &[types::ToolSpec],
    ) -> Result<prompt::RenderedPrompt> {
        prompt::prepare::render_messages(&self.chat_template, &self.eos_token_ids, messages, tools)
    }

    pub fn prepare_messages(
        &self,
        messages: &[types::Message],
        tools: &[types::ToolSpec],
    ) -> Result<prompt::PreparedPrompt> {
        prompt::prepare::prepare_messages(
            &self.tokenizer,
            &self.chat_template,
            &self.eos_token_ids,
            messages,
            tools,
        )
    }

    pub fn prepare_prompt(&self, prompt: &str) -> Result<prompt::PreparedPrompt> {
        prompt::prepare::prepare_text(&self.tokenizer, &self.eos_token_ids, prompt)
    }
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
        Ok(self.prepare_messages(&messages, &tools)?.input_ids)
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
        ModelTokenizer::new(
            self.tokenizer_path.clone(),
            self.chat_template.clone(),
            self.config.get_eos_token_ids(),
        )
    }
}
