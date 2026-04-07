use catgrad::prelude::Dtype;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

use crate::config::LLMConfig;
use crate::helpers::{LLMModel, WeightPostProcess};
use crate::models;
use crate::{LLMError, Result};

mod detokenize;
pub use detokenize::{Detokenizer, detokenize_tokens};
mod prompt;
pub use prompt::PreparedPrompt;

mod images;
pub use images::*;

fn build_hf_api() -> Result<hf_hub::api::sync::Api> {
    let mut builder = ApiBuilder::from_env();
    let env_token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty());
    if let Some(token) = env_token {
        builder = builder.with_token(Some(token));
    }
    Ok(builder.build()?)
}

/// Deserialize a type from JSON while preserving the failing field path in errors.
pub fn from_json_str<T: DeserializeOwned>(json: &str) -> Result<T> {
    let mut deserializer = serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize(&mut deserializer).map_err(LLMError::from)
}

/// Deserialize a type from JSON bytes while preserving the failing field path in errors.
pub fn from_json_slice<T: DeserializeOwned>(json: &[u8]) -> Result<T> {
    let mut deserializer = serde_json::Deserializer::from_slice(json);
    serde_path_to_error::deserialize(&mut deserializer).map_err(LLMError::from)
}

/// Deserialize a type from a JSON reader while preserving the failing field path in errors.
pub fn from_json_reader<T: DeserializeOwned, R: Read>(reader: R) -> Result<T> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    serde_path_to_error::deserialize(&mut deserializer).map_err(LLMError::from)
}

pub fn get_model_files(
    model: &str,
    revision: &str,
) -> Result<(Vec<PathBuf>, PathBuf, PathBuf, PathBuf)> {
    let api = build_hf_api()?;
    let repo = api.repo(Repo::with_revision(
        model.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Get the model.safetensor file(s)
    let m = if let Ok(index) = repo.get("model.safetensors.index.json") {
        let index = std::fs::File::open(index)?;
        let json: serde_json::Value = from_json_reader(index)?;

        let mut weight_files = HashSet::new();
        if let Some(weight_map) = json
            .get("weight_map")
            .ok_or(LLMError::InvalidModelConfig(
                "Missing field `weight_map`".to_string(),
            ))?
            .as_object()
        {
            for v in weight_map.values() {
                let filename = v.as_str().ok_or(LLMError::InvalidModelConfig(
                    "Weight map contained non-string values".to_string(),
                ))?;
                let contents = repo.get(filename)?;
                weight_files.insert(contents);
            }
        }
        weight_files.into_iter().collect()
    } else {
        vec![repo.get("model.safetensors")?]
    };

    let c = repo.get("config.json")?;
    let t = repo.get("tokenizer.json")?;
    let tc = repo.get("tokenizer_config.json")?;

    Ok((m, c, t, tc))
}

// Try getting the model's chat template from the repository
pub fn get_model_chat_template(model: &str, revision: &str) -> Result<String> {
    let api = build_hf_api()?;
    let repo = api.repo(Repo::with_revision(
        model.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    if let Ok(ct) = repo.get("chat_template.jinja") {
        Ok(std::fs::read_to_string(ct)?)
    } else {
        let tc_path = repo.get("tokenizer_config.json")?;
        let tc = std::fs::read_to_string(tc_path)?;
        let tokenizer_config: serde_json::Value = from_json_str(&tc)?;
        Ok(tokenizer_config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or(LLMError::InvalidModelConfig(
                "Missing or invalid `chat_template` in tokenizer config".to_string(),
            ))?
            .to_string())
    }
}

use chrono::Local;
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

fn strftime_now(format_str: String) -> String {
    Local::now().format(&format_str).to_string()
}

pub fn render_chat_template(
    chat_template: &str,
    tokenizer_config: &serde_json::Value,
    prompt: &str,
    has_image: bool,
    enable_thinking: bool,
) -> Result<String, minijinja::Error> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("strftime_now", strftime_now);
    env.add_template("chat", chat_template)?;
    let tmpl = env.get_template("chat")?;
    let bos_token = tokenizer_config
        .get("bos_token")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let messages = if has_image {
        let content = vec![
            context!(type => "text", text => prompt),
            context!(type => "image"),
        ];
        vec![context!(role => "user",content => content)]
    } else {
        vec![context!(role => "user",content => prompt)]
    };
    tmpl.render(context!(
        messages => messages,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        bos_token => bos_token
    ))
}

#[derive(Debug, Clone)]
pub struct PreparedImageInput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum ModelRuntimeContext {
    Qwen3_5Vision(models::qwen3_5::Qwen3_5RuntimeVisionConfig),
}

#[derive(Debug, Clone, Default)]
pub struct PreparedMultimodalInput {
    pub image: Option<PreparedImageInput>,
    pub runtime_context: Option<ModelRuntimeContext>,
}

pub fn get_model_architecture(config_json: &serde_json::Value) -> Result<&str> {
    config_json["architectures"][0]
        .as_str()
        .ok_or(LLMError::InvalidModelConfig(
            "Missing architectures field".to_string(),
        ))
}

pub fn prepare_multimodal_input(
    config_json: &serde_json::Value,
    image_path: Option<&Path>,
) -> Result<PreparedMultimodalInput> {
    let Some(image_path) = image_path else {
        return Ok(PreparedMultimodalInput::default());
    };

    match get_model_architecture(config_json)? {
        "Qwen3_5ForConditionalGeneration" => {
            let prepared = models::qwen3_5::prepare_qwen3_5_image_input(image_path, config_json)?;
            Ok(PreparedMultimodalInput {
                image: Some(PreparedImageInput {
                    data: prepared.pixels,
                    shape: prepared.shape,
                }),
                runtime_context: Some(ModelRuntimeContext::Qwen3_5Vision(prepared.runtime_vision)),
            })
        }
        _ => Ok(PreparedMultimodalInput::default()),
    }
}

pub fn interpolate_multimodal_prompt(
    config_json: &serde_json::Value,
    runtime_context: Option<&ModelRuntimeContext>,
    prompt: &str,
) -> Result<String> {
    let arch = get_model_architecture(config_json)?;
    let interpolated = match arch {
        "Qwen3_5ForConditionalGeneration" => models::qwen3_5::interpolate_qwen3_5_prompt(
            config_json,
            match runtime_context {
                Some(ModelRuntimeContext::Qwen3_5Vision(runtime_vision)) => Some(runtime_vision),
                _ => None,
            },
            prompt,
        )?,
        _ => {
            let model = get_model(config_json, 1, runtime_context, Dtype::F32)?;
            if !model.is_multimodal() {
                return Err(LLMError::InvalidModelConfig(format!(
                    "Model architecture {arch} does not support image input"
                )));
            }
            model.multimodal_interpolate_prompt(prompt)
        }
    };

    interpolated.ok_or(LLMError::InvalidModelConfig(format!(
        "Model architecture {arch} did not provide multimodal prompt interpolation"
    )))
}

pub fn get_model(
    config_json: &serde_json::Value,
    max_sequence_length: usize,
    runtime_context: Option<&ModelRuntimeContext>,
    dtype: Dtype,
) -> Result<Box<dyn LLMModel>> {
    let arch = get_model_architecture(config_json)?;

    let model: Box<dyn LLMModel> = match arch {
        "Gemma2ForCausalLM" | "Gemma3ForCausalLM" => Box::new(models::gemma3::Gemma3Model::new(
            "model",
            config_json,
            dtype,
        )?),
        "Gemma3ForConditionalGeneration" | "PaliGemmaForConditionalGeneration" => Box::new(
            models::gemma3::Gemma3Model::new("language_model.model", config_json, dtype)?,
        ),
        "Gemma4ForConditionalGeneration" => Box::new(models::gemma4::Gemma4Model::new(
            "model.language_model",
            config_json,
            dtype,
        )?),
        "Mistral3ForConditionalGeneration" => Box::new(models::mistral3::Mistral3Model::new(
            "language_model",
            config_json,
            dtype,
        )?),
        "MistralForCausalLM" | "LlamaForCausalLM" | "SmolLM3ForCausalLM" => {
            Box::new(models::llama::LlamaModel::new("", config_json, dtype)?)
        }
        "NemotronHForCausalLM" => {
            Box::new(models::nemotron::NemotronModel::new(config_json, dtype)?)
        }
        "SmolVLMForConditionalGeneration" => {
            Box::new(models::smolvlm2::SmolVLM2Model::new(config_json, dtype)?)
        }
        "Phi3ForCausalLM" | "Phi4MMForCausalLM" => {
            Box::new(models::phi3::Phi3Model::new(config_json, dtype)?)
        }
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" | "OlmoHybridForCausalLM" => Box::new(
            models::olmo::OlmoModel::new(config_json, max_sequence_length, dtype)?,
        ),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => {
            Box::new(models::qwen3::Qwen3Model::new(config_json, dtype)?)
        }
        "Qwen3_5ForConditionalGeneration" => Box::new(models::qwen3_5::Qwen3_5Model::new(
            config_json,
            max_sequence_length,
            match runtime_context {
                Some(ModelRuntimeContext::Qwen3_5Vision(runtime_vision)) => Some(runtime_vision),
                _ => None,
            },
            dtype,
        )?),
        "GraniteForCausalLM" | "GraniteMoeForCausalLM" | "GraniteMoeHybridForCausalLM" => {
            Box::new(models::granite::GraniteModel::new(config_json, dtype)?)
        }
        "DeepseekV3ForCausalLM" => {
            Box::new(models::deepseek::DeepSeekModel::new(config_json, dtype)?)
        }
        "GptOssForCausalLM" => Box::new(models::gpt_oss::GPTOssModel::new(config_json, dtype)?),
        "Lfm2ForCausalLM" => Box::new(models::lfm2::Lfm2Model::new(config_json, dtype)?),
        "GPT2LMHeadModel" => Box::new(models::gpt2::GPT2Model::new(config_json, dtype)?),
        _ => {
            return Err(LLMError::InvalidModelConfig(format!(
                "Unsupported model architecture: {}",
                arch
            )));
        }
    };
    Ok(model)
}

use catgrad::interpreter;
use catgrad::prelude::path;
use catgrad::typecheck;

// Concatenates MoE expert weights from separate tensors into single tensors per layer
// to avoid the need for dynamic parameter names
pub(crate) fn concat_moe_experts<B: interpreter::Backend>(
    config: &dyn LLMConfig,
    num_local_experts: usize,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    use catgrad::typecheck::*;

    let proj_names = ["down_proj", "gate_proj", "up_proj"];

    for layer_idx in 0..config.num_hidden_layers() {
        for proj_name in &proj_names {
            // Collect all expert tensors for this layer and projection
            let mut expert_tensors = Vec::new();
            let mut expert_keys = Vec::new();

            for expert_idx in 0..num_local_experts {
                let key_str = format!(
                    "model.layers.{}.mlp.experts.{}.{}.weight",
                    layer_idx, expert_idx, proj_name
                );
                let key = path(key_str.split(".").collect()).expect("invalid param path");

                // Check if this expert exists in the parameter maps
                if let Some(interpreter::Value::Tensor(tensor)) = parameter_values.0.get(&key) {
                    expert_tensors.push(tensor.clone());
                    expert_keys.push(key);
                }
            }

            if expert_tensors.is_empty() {
                continue;
            }

            if expert_tensors.len() != num_local_experts {
                return Err(LLMError::InvalidModelConfig(format!(
                    "Expected {} experts for layer {} {}, found {}",
                    num_local_experts,
                    layer_idx,
                    proj_name,
                    expert_tensors.len()
                )));
            }

            let original_shape = expert_tensors[0].shape();
            let original_dims = original_shape.0.clone();

            let mut new_shape_dims = vec![num_local_experts];
            new_shape_dims.extend(original_dims.clone());

            let mut reshaped_tensors = Vec::new();
            for tensor in expert_tensors {
                let mut reshape_dims = vec![1];
                reshape_dims.extend(original_dims.clone());
                let reshaped = backend.reshape(tensor, interpreter::Shape(reshape_dims));
                reshaped_tensors.push(reshaped);
            }

            // Concatenate all reshaped tensors along dimension 0
            // TODO: this is naive and slow. Either preallocate or fuse this with the safetensors loading code.
            let mut concatenated = reshaped_tensors[0].clone();
            for tensor in &reshaped_tensors[1..] {
                concatenated = backend.concat(concatenated, tensor.clone(), 0);
            }

            let new_key_str = format!(
                "model.layers.{}.mlp.experts.{}.weight",
                layer_idx, proj_name
            );
            let new_key = path(new_key_str.split(".").collect()).expect("invalid param path");

            parameter_values
                .0
                .insert(new_key.clone(), interpreter::Value::Tensor(concatenated));

            let vne: Vec<NatExpr> = new_shape_dims.into_iter().map(NatExpr::Constant).collect();
            let tensor_type = typecheck::Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vne),
            }));
            parameter_types.0.insert(new_key, tensor_type);

            // Remove original experts
            for key in expert_keys {
                parameter_values.0.remove(&key);
                parameter_types.0.remove(&key);
            }
        }
    }

    Ok(())
}

pub fn post_process_weights<B: interpreter::Backend>(
    post_process: WeightPostProcess,
    config: &dyn LLMConfig,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    match post_process {
        WeightPostProcess::None => Ok(()),
        WeightPostProcess::ConcatMoeExperts { num_local_experts } => concat_moe_experts(
            config,
            num_local_experts,
            backend,
            parameter_values,
            parameter_types,
        ),
    }
}

pub fn post_process_model_weights<B: interpreter::Backend>(
    model: &dyn LLMModel,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    post_process_weights(
        model.weight_post_process(),
        model.config(),
        backend,
        parameter_values,
        parameter_types,
    )
}

pub fn load_model_weights<B: interpreter::Backend>(
    model_paths: Vec<PathBuf>,
    backend: &B,
    dtype: Dtype,
) -> Result<(interpreter::Parameters<B>, typecheck::Parameters, usize)> {
    if dtype != Dtype::F32 {
        return Err(LLMError::UnsupportedDtype(format!("{dtype:?}")));
    }

    // Read each tensor
    let mut type_map = HashMap::new();
    let mut data_map = HashMap::new();
    let mut total_params = 0;

    for file_path in model_paths {
        let file = std::fs::File::open(file_path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&data)?;

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let tensor_data = view.data();

            use catgrad::typecheck::*;
            // Convert dtype and load tensor data
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => tensor_data
                    .par_chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                safetensors::Dtype::BF16 => tensor_data
                    .par_chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                _ => {
                    return Err(LLMError::UnsupportedDtype(format!("{:?}", view.dtype())));
                }
            };
            total_params += data.len();

            let tensor = interpreter::tensor(backend, interpreter::Shape(shape.clone()), data)
                .map_err(|err| {
                    LLMError::InvalidModelConfig(format!("failed to create tensor {name}: {err:?}"))
                })?;
            let key = path(name.split(".").collect()).map_err(|err| {
                LLMError::InvalidModelConfig(format!("invalid param path {name}: {}", err.0))
            })?;
            data_map.insert(key.clone(), tensor);

            let vne = shape.into_iter().map(NatExpr::Constant).collect();
            let tensor_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vne),
            }));
            type_map.insert(key, tensor_type);
        }
    }

    let parameter_values = interpreter::Parameters::from(data_map);
    let parameter_types = typecheck::Parameters::from(type_map);

    Ok((parameter_values, parameter_types, total_params))
}

pub fn load_model<B: interpreter::Backend>(
    model_name: &str,
    revision: &str,
    backend: &B,
    dtype: Dtype,
) -> Result<(
    interpreter::Parameters<B>,
    typecheck::Parameters,
    serde_json::Value,
    Tokenizer,
    serde_json::Value,
    usize,
)> {
    let (model_paths, config_path, tokenizer_path, tokenizer_config_path) =
        get_model_files(model_name, revision)?;
    let config_json: serde_json::Value = from_json_str(&std::fs::read_to_string(&config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| LLMError::TokenizerError(format!("tokenizer load error {:?}", err)))?;
    let tokenizer_config_json: serde_json::Value =
        from_json_str(&std::fs::read_to_string(&tokenizer_config_path)?)?;

    let (parameter_values, parameter_types, total_params) =
        load_model_weights(model_paths, backend, dtype)?;

    Ok((
        parameter_values,
        parameter_types,
        config_json,
        tokenizer,
        tokenizer_config_json,
        total_params,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn print_bench_table(
    model_name: &str,
    size_gib: f64,
    params_m: f64,
    backend: &str,
    pp: usize,
    elapsed_pp: std::time::Duration,
    tg: usize,
    elapsed_tg: std::time::Duration,
) {
    println!(
        "| model                                    | size       | params     | backend    |            test |                  t/s |"
    );
    println!(
        "| ---------------------------------------- | ---------- | ---------- | ---------- | --------------- | -------------------- |"
    );

    let tps_pp = pp as f64 / elapsed_pp.as_secs_f64();
    println!(
        "| {:<40} | {:>6.2} GiB | {:>8.2} M | {:<10} | {:>15} | {:>20.2} |",
        model_name,
        size_gib,
        params_m,
        backend,
        format!("pp{}", pp),
        tps_pp
    );

    let tps_tg = tg as f64 / elapsed_tg.as_secs_f64();
    println!(
        "| {:<40} | {:>6.2} GiB | {:>8.2} M | {:<10} | {:>15} | {:>20.2} |",
        model_name,
        size_gib,
        params_m,
        backend,
        format!("tg{}", tg),
        tps_tg
    );
}

// Model-specific empty state cache.
// Usually just KV-cache but for hybrid models it can include additional state from the linear layers.
pub fn empty_state_cache<B: interpreter::Backend>(
    backend: &B,
    model: &dyn LLMModel,
) -> Result<Vec<interpreter::Value<B>>> {
    let typ = model.empty_state_type();

    typ.iter()
        .map(|(dtype, shape)| match dtype {
            Dtype::F32 => {
                let data = vec![0.0f32; shape.0.iter().product()];
                interpreter::tensor(backend, shape.clone(), data).map_err(|err| {
                    LLMError::InvalidModelConfig(format!("state tensor error: {:?}", err))
                })
            }
            Dtype::U32 => {
                let data = vec![0u32; shape.0.iter().product()];
                interpreter::tensor(backend, shape.clone(), data).map_err(|err| {
                    LLMError::InvalidModelConfig(format!("state tensor error: {:?}", err))
                })
            }
        })
        .collect()
}
