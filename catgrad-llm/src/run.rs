//! Greedy local-generation runtime.
//!
//! A [`ModelEngine`] owns the immutable model assets for one model: weights, config, tokenizer,
//! chat template, and stop tokens. Engines are cheap to clone and are intended to be reused across
//! multiple requests on the same thread.
//!
//! Request-local execution state does not live in the engine. Each
//! [`ModelEngine::generate_from_prepared`] call creates a fresh internal runner, so KV-cache state,
//! token position, and generated text do not leak across requests. If you want prior conversation
//! to influence generation, include that history in the prepared prompt or message list.
use crate::legacy::models::utils::{Cache, Config, ModelBuilder, get_model};
use crate::legacy::nn::layers::{argmax, cast, reshape};
use crate::types;
use crate::utils::{
    from_json_str, get_model_chat_template, get_model_files, read_safetensors_multiple,
};
use crate::{Detokenizer, PreparedPrompt, Result};
use catgrad_legacy::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{Dtype, NdArrayType, Shape, Var},
};
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use tokenizers::tokenizer::Tokenizer;

fn read_to_value<V: for<'a> serde::Deserialize<'a>>(path: impl AsRef<Path>) -> Result<V> {
    let config_str = &std::fs::read_to_string(path)?;
    from_json_str(config_str)
}

struct ModelEngineInner {
    tensors: Rc<HashMap<String, TaggedNdArray>>,
    arch: String,
    config: Config,
    tokenizer: Tokenizer,
    chat_template: String,
    eos_token_ids: Vec<i32>,
    use_kv_cache: bool,
}

/// Reusable local inference entry point for one model.
///
/// A `ModelEngine` owns immutable model assets and can be reused across many requests. It does not
/// retain per-request decode state; each generation call creates a fresh internal runner.
///
/// `ModelEngine` uses `Rc` internally, so it is cheap to clone but is not `Send` or `Sync`.
///
/// # Example
///
/// ```no_run
/// use catgrad_llm::run::ModelEngine;
/// use catgrad_llm::types::Message;
/// use catgrad_llm::types::openai::ChatMessage;
///
/// let engine = ModelEngine::new("Qwen/Qwen3-0.6B", true)?;
///
/// let prompt = engine.prepare_messages(&[
///     Message::openai(ChatMessage::system("You are concise.")),
///     Message::openai(ChatMessage::user("What is 2+2?")),
/// ])?;
/// let first = engine.generate_from_prepared(&prompt, 64, |_| Ok(()))?;
///
/// let prompt = engine.prepare_messages(&[
///     Message::openai(ChatMessage::system("You are concise.")),
///     Message::openai(ChatMessage::user("What is 4+4?")),
/// ])?;
/// let second = engine.generate_from_prepared(&prompt, 64, |_| Ok(()))?;
///
/// assert!(first.completion_tokens <= 64);
/// assert!(second.completion_tokens <= 64);
/// # Ok::<_, catgrad_llm::LLMError>(())
/// ```
#[derive(Clone)]
pub struct ModelEngine {
    inner: Rc<ModelEngineInner>,
}

// Internal per-request decode state. A fresh runner is created for each generation call.
struct ModelRunner {
    engine: ModelEngine,
    model: Box<dyn ModelBuilder>,
    kv_cache: Vec<TaggedNdArray>,
    total_tokens: usize,
}

/// Final text and token counts from a local generation.
pub struct GenerationOutput {
    /// Fully decoded generated text.
    pub text: String,
    /// Number of prompt tokens fed into the model.
    pub prompt_tokens: u32,
    /// Number of new tokens produced during generation.
    pub completion_tokens: u32,
    /// Why generation stopped.
    pub termination: GenerationTermination,
}

/// Why local generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTermination {
    /// Generation reached a model or configured stop token.
    Stop,
    /// Generation stopped because the requested token budget was exhausted.
    MaxTokens,
}

impl From<GenerationTermination> for types::openai::FinishReason {
    fn from(value: GenerationTermination) -> Self {
        match value {
            GenerationTermination::Stop => Self::Stop,
            GenerationTermination::MaxTokens => Self::Length,
        }
    }
}

impl From<GenerationTermination> for types::anthropic::StopReason {
    fn from(value: GenerationTermination) -> Self {
        match value {
            GenerationTermination::Stop => Self::EndTurn,
            GenerationTermination::MaxTokens => Self::MaxTokens,
        }
    }
}

impl ModelEngine {
    /// Loads model weights, tokenizer, and chat template from Hugging Face.
    ///
    /// Set `use_kv_cache` to reuse KV-cache state between decode steps within a single request.
    /// The cache does not persist across separate generation calls.
    pub fn new(model_name: &str, use_kv_cache: bool) -> Result<Self> {
        let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, "main")?;
        let chat_template = get_model_chat_template(model_name, "main")?;
        let config: Config = read_to_value(config_path)?;
        let arch = config.architectures[0].clone();

        let mut model = get_model(&arch)?;
        let mut tensors = read_safetensors_multiple(model_paths, false)?;
        model.post_load(&mut tensors);

        let tokenizer = Tokenizer::from_file(tokenizer_path)?;
        let chat_template = chat_template
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");
        let eos_token_ids = config.get_eos_token_ids();

        Ok(Self {
            inner: Rc::new(ModelEngineInner {
                tensors: Rc::new(tensors),
                arch,
                config,
                tokenizer,
                chat_template,
                eos_token_ids,
                use_kv_cache,
            }),
        })
    }

    /// Renders chat messages with the model chat template and tokenizes the result.
    ///
    /// The engine does not retain chat history. To continue a conversation, pass the full message
    /// history you want rendered into the prompt.
    pub fn prepare_messages(&self, messages: &[types::Message]) -> Result<PreparedPrompt> {
        PreparedPrompt::from_messages(
            &self.inner.tokenizer,
            &self.inner.chat_template,
            messages,
            &self.inner.eos_token_ids,
        )
    }

    /// Tokenizes a raw prompt string without applying a chat template.
    ///
    /// Use this for plain-completion style prompts where you already control the final prompt text.
    pub fn prepare_prompt(&self, prompt: &str) -> Result<PreparedPrompt> {
        PreparedPrompt::from_prompt(&self.inner.tokenizer, prompt, &self.inner.eos_token_ids)
    }

    /// Runs greedy local generation from a prepared prompt and streams text deltas.
    ///
    /// Each call creates a fresh internal runner, so generation state does not persist between
    /// calls. Reuse the same engine for many requests; encode request-specific history in
    /// `prepared`.
    pub fn generate_from_prepared<F>(
        &self,
        prepared: &PreparedPrompt,
        max_tokens: u32,
        mut on_text_delta: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let mut runner = ModelRunner::new(self.clone())?;
        let mut next_token = runner.generate_next_token(prepared.input_ids.clone());
        let mut decoder =
            Detokenizer::from_tokenizer(&self.inner.tokenizer, &prepared.stop_token_ids);

        let mut completion_tokens = 0u32;
        let mut termination = GenerationTermination::MaxTokens;
        for _ in 0..max_tokens {
            let Some(token) = next_token else {
                termination = GenerationTermination::Stop;
                break;
            };

            let delta = decoder.push_tokens(&[token])?;
            if decoder.is_stopped() {
                termination = GenerationTermination::Stop;
                break;
            }

            completion_tokens += 1;
            if !delta.is_empty() {
                on_text_delta(&delta)?;
            }

            next_token = runner.generate_next_token(vec![token]);
        }

        Ok(GenerationOutput {
            text: decoder.finish(),
            prompt_tokens: prepared.input_ids.len() as u32,
            completion_tokens,
            termination,
        })
    }

    fn use_kv_cache(&self) -> bool {
        self.inner.use_kv_cache
    }

    fn config(&self) -> &Config {
        &self.inner.config
    }

    fn arch(&self) -> &str {
        &self.inner.arch
    }

    fn tensors(&self) -> Rc<HashMap<String, TaggedNdArray>> {
        Rc::clone(&self.inner.tensors)
    }

    fn initial_kv_cache(&self) -> Vec<TaggedNdArray> {
        let config = self.config();
        let v = TaggedNdArray::F32(NdArray::new_empty(Shape(vec![
            1,
            config.get_num_kv_heads(),
            0,
            config.get_head_dim(),
        ])));
        vec![v; 2 * config.num_hidden_layers]
    }

    fn next_token(builder: &Builder, logits: Var) -> Var {
        let batches = logits.label.shape.0[0];
        let am = argmax(builder, logits);
        let am = reshape(builder, Shape(vec![batches, 1]), am);
        cast(builder, Dtype::I32, am)
    }

    fn eval_for_step(
        &self,
        model: &dyn ModelBuilder,
        total_tokens: usize,
        num_tokens: usize,
    ) -> EvalState {
        let config = self.config();
        let use_kv_cache = self.use_kv_cache();
        let batches = 1;
        let in_type = NdArrayType::new(Shape(vec![batches, num_tokens]), Dtype::I32);

        let mut state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let mut cache = Cache::init(builder, config, total_tokens + num_tokens, use_kv_cache);

            if use_kv_cache {
                let kv_cache_type = NdArrayType::new(
                    Shape(vec![
                        batches,
                        config.get_num_kv_heads(),
                        total_tokens,
                        config.get_head_dim(),
                    ]),
                    Dtype::F32,
                );

                for layer_id in 0..config.num_hidden_layers {
                    cache.in_kv_cache[layer_id] = (
                        Var::new(builder.clone(), kv_cache_type.clone()),
                        Var::new(builder.clone(), kv_cache_type.clone()),
                    );
                }
            }

            let result = model.build(builder, config, &mut cache, total_tokens, x.clone());

            let mut sources_vec = vec![x];
            if use_kv_cache {
                for layer_id in 0..config.num_hidden_layers {
                    sources_vec.push(cache.in_kv_cache[layer_id].0.clone());
                    sources_vec.push(cache.in_kv_cache[layer_id].1.clone());
                }
            }

            let next_token = Self::next_token(builder, result);
            let mut targets_vec = vec![next_token];
            if use_kv_cache {
                let out_kv_cache: Vec<_> = cache
                    .out_kv_cache
                    .into_iter()
                    .flat_map(|(a, b)| vec![a, b])
                    .collect();
                targets_vec.extend(out_kv_cache);
            }

            (sources_vec, targets_vec)
        });

        state.set_parameters(self.tensors());
        state
    }
}

impl ModelRunner {
    fn new(engine: ModelEngine) -> Result<Self> {
        Ok(Self {
            model: get_model(engine.arch())?,
            kv_cache: if engine.use_kv_cache() {
                engine.initial_kv_cache()
            } else {
                vec![]
            },
            total_tokens: 0,
            engine,
        })
    }

    fn run(&mut self, state: &mut EvalState, x: &NdArray<i32>) -> TaggedNdArray {
        let mut sources = vec![x.clone().into()];
        if self.engine.use_kv_cache() {
            sources.extend(self.kv_cache.clone());
        }

        let result = state.eval_with(sources);
        if self.engine.use_kv_cache() {
            self.kv_cache = result[1..].iter().map(|&tensor| tensor.clone()).collect();
        }

        result[0].clone()
    }

    fn generate_next_token(&mut self, tokens: Vec<i32>) -> Option<i32> {
        if tokens.is_empty() {
            return None;
        }

        let num_tokens = tokens.len();
        let input = NdArray::new(tokens, Shape(vec![1, num_tokens]));

        let mut state = self
            .engine
            .eval_for_step(&*self.model, self.total_tokens, num_tokens);
        log::debug!("Model graph built...");
        let result = self.run(&mut state, &input);

        let token = result.data()[0] as i32;
        if self.engine.config().get_eos_token_ids().contains(&token) {
            return None;
        }

        self.total_tokens += num_tokens;
        Some(token)
    }
}
