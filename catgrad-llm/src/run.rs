//! A stripped-down version of ModelRunner from catgrad examples, intended for serving
use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::nn::layers::{argmax, cast, reshape},
    core::{Dtype, NdArrayType, Shape, Var},
};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use tokenizers::tokenizer::{Result, Tokenizer};

use crate::models::utils::{Cache, Config, ModelBuilder, get_model};

use crate::utils::{get_model_chat_template, get_model_files, read_safetensors_multiple};

use crate::serve;

/// Load model
pub struct ModelLoader {
    config: Config,
    model_paths: Vec<PathBuf>,
    tokenizer_path: PathBuf,
    chat_template: String,
    use_kv_cache: bool,
}

fn read_to_value<V: for<'a> serde::Deserialize<'a>>(path: PathBuf) -> Result<V> {
    let config_str = &std::fs::read_to_string(path).map_err(|e| serve::Error::IO(e.to_string()))?;
    let result: V =
        serde_json::from_str(config_str).map_err(|e| serve::Error::IO(e.to_string()))?;
    Ok(result)
}

impl ModelLoader {
    pub fn new(model_name: &str, use_kv_cache: bool) -> serve::Result<Self> {
        let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, "main");

        let chat_template = get_model_chat_template(model_name, "main");
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
    fn new(tokenizer_path: PathBuf, chat_template: String) -> serve::Result<Self> {
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

    fn render_context(&self, messages: &[serve::Message]) -> String {
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template("chat", &self.chat_template).unwrap();
        let tmpl = env.get_template("chat").unwrap();
        let message_context: Vec<_> = messages
            .iter()
            .map(|msg| context!(role => msg.role, content => msg.content))
            .collect();
        tmpl.render(context!(
            messages => message_context,
            add_generation_prompt => true,
            enable_thinking => false
        ))
        .expect("template failed to render")
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
    pub fn new(
        model_paths: Vec<PathBuf>,
        config: Config,
        use_kv_cache: bool,
    ) -> Result<ModelRunner> {
        let arch = &config.architectures[0];

        let mut model = get_model(arch)?;
        let mut tensors = read_safetensors_multiple(model_paths);
        model.post_load(&mut tensors);

        let kv_cache = if use_kv_cache {
            let v = TaggedNdArray::F32(NdArray::new_empty(Shape(vec![
                1,
                config.get_num_kv_heads(),
                0,
                config.get_head_dim(),
            ])));
            vec![v; 2 * config.num_hidden_layers]
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

impl serve::LM<i32> for ModelRunner {
    fn set_context(&mut self, context: Vec<i32>) {
        let n = longest_common_prefix(&self.context, &context);
        // TODO: if n < self.context.len, just restart (smarter: try to keep as much of kv cache as possible)
        assert_eq!(n, self.context.len());
        self.context = context[..n].to_vec();
        self.tokens = context[n..].to_vec();
    }
}

impl serve::Tokenizer<i32> for ModelTokenizer {
    fn encode(&self, content: String) -> serve::Result<Vec<i32>> {
        let tokens = self.tokenizer.encode(content, true)?;
        Ok(tokens.get_ids().iter().map(|&x| x as i32).collect())
    }

    fn decode(&self, tokens: Vec<i32>) -> serve::Result<String> {
        // TODO: efficiency?
        // TODO: support u32 in interpreter to remove try_into().unwrap().
        let tokens_u32: Vec<u32> = tokens.into_iter().map(|i| i.try_into().unwrap()).collect();
        Ok(self.tokenizer.decode(&tokens_u32, false)?)
    }
}

impl serve::ChatTokenizer<i32> for ModelTokenizer {
    fn encode_messages(&self, messages: Vec<serve::Message>) -> serve::Result<Vec<i32>> {
        // initialize context
        let content = self.render_context(&messages);
        use serve::Tokenizer;
        self.encode(content)
    }
}

impl serve::Loader<i32, ModelRunner, ModelTokenizer> for ModelLoader {
    fn load_runner(&self) -> serve::Result<ModelRunner> {
        Ok(ModelRunner::new(
            self.model_paths.clone(),
            self.config.clone(),
            self.use_kv_cache,
        )?)
    }

    fn load_tokenizer(&self) -> serve::Result<ModelTokenizer> {
        ModelTokenizer::new(self.tokenizer_path.clone(), self.chat_template.clone())
    }
}
