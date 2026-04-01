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
use crate::helpers::LLMModel;
use crate::types;
use crate::utils::{
    empty_state_cache, get_model, get_model_chat_template, load_model, post_process_model_weights,
};
use crate::{Detokenizer, LLMError, PreparedPrompt, Result};
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::{self, Backend};
use catgrad::prelude::{Dtype, Shape, TypedTerm, stdlib, to_load_ops};
use std::rc::Rc;
use tokenizers::tokenizer::Tokenizer;

struct ModelEngineInner {
    backend: CandleBackend,
    parameter_values: interpreter::Parameters<CandleBackend>,
    parameter_types: catgrad::typecheck::Parameters,
    config_json: serde_json::Value,
    tokenizer: Tokenizer,
    chat_template: String,
    eos_token_ids: Vec<i32>,
    dtype: Dtype,
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
/// let engine = ModelEngine::new("Qwen/Qwen3-0.6B", true, catgrad::prelude::Dtype::F32)?;
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
    model: Box<dyn LLMModel>,
    typed_term: TypedTerm,
    interpreter: interpreter::Interpreter<CandleBackend>,
    state_cache: Vec<interpreter::Value<CandleBackend>>,
    max_sequence_length: usize,
    use_kv_cache: bool,
    eos_token_ids: Vec<i32>,
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
    pub fn new(model_name: &str, use_kv_cache: bool, dtype: Dtype) -> Result<Self> {
        let backend = CandleBackend::new();
        let (parameter_values, parameter_types, config_json, tokenizer, _) =
            load_model(model_name, "main", &backend, dtype.clone())?;
        let model = get_model(&config_json, 1, None, dtype.clone())?;
        let chat_template = get_model_chat_template(model_name, "main")?;
        let chat_template = chat_template
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");
        let eos_token_ids = model.config().get_eos_token_ids();

        Ok(Self {
            inner: Rc::new(ModelEngineInner {
                backend,
                parameter_values,
                parameter_types,
                config_json,
                tokenizer,
                chat_template,
                eos_token_ids,
                dtype,
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
        let prompt_token_ids = token_ids_to_u32(&prepared.input_ids);
        let max_sequence_length = prepared.input_ids.len() + max_tokens as usize;
        let mut runner = ModelRunner::new(self.clone(), max_sequence_length)?;
        let mut decoder =
            Detokenizer::from_tokenizer(&self.inner.tokenizer, &prepared.stop_token_ids);
        let mut step_tokens = prompt_token_ids;

        let mut completion_tokens = 0u32;
        let mut termination = GenerationTermination::MaxTokens;
        for _ in 0..max_tokens {
            let Some((token, next_input_token)) = runner.generate_next_token(&step_tokens)? else {
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

            if self.use_kv_cache() {
                step_tokens.clear();
                step_tokens.push(next_input_token);
            } else {
                step_tokens.push(next_input_token);
            }
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
}

impl ModelRunner {
    fn new(engine: ModelEngine, max_sequence_length: usize) -> Result<Self> {
        let backend = engine.inner.backend.clone();
        let mut parameter_values = engine.inner.parameter_values.clone();
        let mut parameter_types = engine.inner.parameter_types.clone();
        let model = get_model(
            &engine.inner.config_json,
            max_sequence_length,
            None,
            engine.inner.dtype.clone(),
        )?;
        post_process_model_weights(
            model.as_ref(),
            &backend,
            &mut parameter_values,
            &mut parameter_types,
        )?;

        let typed_term = model.term().ok_or_else(|| {
            LLMError::InvalidModelConfig("Failed to create typed term".to_string())
        })?;
        let mut env = stdlib();
        env.declarations
            .extend(to_load_ops(model.path(), parameter_types.keys()));
        let interpreter = interpreter::Interpreter::new(backend.clone(), env, parameter_values);
        let state_cache = empty_state_cache(&backend, model.as_ref())?;

        Ok(Self {
            model,
            typed_term,
            interpreter,
            state_cache,
            max_sequence_length,
            use_kv_cache: engine.use_kv_cache(),
            eos_token_ids: engine.inner.eos_token_ids.clone(),
        })
    }

    fn generate_next_token(&mut self, tokens: &[u32]) -> Result<Option<(i32, u32)>> {
        if tokens.is_empty() {
            return Ok(None);
        }

        let mut inputs = Vec::with_capacity(self.state_cache.len() + 3);
        inputs.push(token_tensor(&self.interpreter, tokens)?);
        inputs.extend(self.state_cache.iter().cloned());
        inputs.push(interpreter::Value::Nat(self.max_sequence_length));
        if let Some(extra_nat) = self.model.extra_nat_input(tokens.len()) {
            inputs.push(interpreter::Value::Nat(extra_nat));
        }

        let mut results = self
            .interpreter
            .run(self.typed_term.term.clone(), inputs)
            .map_err(|err| {
                LLMError::InvalidModelConfig(format!("Failed to run inference: {err}"))
            })?;
        if results.is_empty() {
            return Err(LLMError::InvalidModelConfig(
                "model returned no outputs".to_string(),
            ));
        }

        let updated_state_cache = if results.len() > 1 {
            results.split_off(1)
        } else {
            Vec::new()
        };
        let output = results.remove(0);
        let token = match output {
            interpreter::Value::Tensor(arr) => match self.interpreter.backend.to_vec(arr) {
                interpreter::TaggedVec::U32(v) => v.last().copied().ok_or_else(|| {
                    LLMError::InvalidModelConfig("token output tensor was empty".to_string())
                })?,
                _ => {
                    return Err(LLMError::InvalidModelConfig(
                        "unexpected output dtype".to_string(),
                    ));
                }
            },
            value => {
                return Err(LLMError::InvalidModelConfig(format!(
                    "output was not a tensor: {value:?}"
                )));
            }
        };

        if self.use_kv_cache {
            self.state_cache = updated_state_cache;
        }

        let token_i32 = token as i32;
        if self.eos_token_ids.contains(&token_i32) {
            return Ok(None);
        }

        Ok(Some((token_i32, token)))
    }
}

fn token_tensor(
    interpreter: &interpreter::Interpreter<CandleBackend>,
    input_tokens: &[u32],
) -> Result<interpreter::Value<CandleBackend>> {
    interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, input_tokens.len()]),
        input_tokens.to_vec(),
    )
    .map_err(|err| LLMError::InvalidModelConfig(format!("input tensor error: {err:?}")))
}

fn token_ids_to_u32(token_ids: &[i32]) -> Vec<u32> {
    token_ids.iter().map(|&token_id| token_id as u32).collect()
}
