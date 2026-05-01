use catgrad::prelude::Dtype;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

use crate::helpers::LLMModel;
use crate::models;
use crate::{LLMError, Result};

mod detokenize;
pub use detokenize::{Detokenizer, detokenize_tokens};

mod prompt;
pub(crate) use prompt::render_chat_prompt_with_options;
pub use prompt::{
    PreparedPrompt, RenderChatTemplateOptions, render_chat_template, render_chat_template_values,
};

mod images;
pub(crate) use images::convert_image_to_patches;
pub use images::*;

mod audio;
pub use audio::*;

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

    let chat_template = if let Ok(ct) = repo.get("chat_template.jinja") {
        std::fs::read_to_string(ct)?
    } else {
        let tc_path = repo.get("tokenizer_config.json")?;
        let tc = std::fs::read_to_string(tc_path)?;
        let tokenizer_config: serde_json::Value = from_json_str(&tc)?;
        tokenizer_config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or(LLMError::InvalidModelConfig(
                "Missing or invalid `chat_template` in tokenizer config".to_string(),
            ))?
            .to_string()
    };
    // Some chat templates contain these tags that are not used for inference.
    // If more variants show up a regex may be needed later on.
    Ok(chat_template
        .replace("{% generation %}", "")
        .replace("{%- generation -%}", "")
        .replace("{% endgeneration %}", "")
        .replace("{%- endgeneration -%}", ""))
}

#[derive(Debug, Clone)]
pub struct PreparedImageInput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum ModelRuntimeContext {
    Qwen3_5Vision(models::qwen3_5::Qwen3_5RuntimeVisionConfig),
    Gemma4Vision(models::gemma4::Gemma4RuntimeVisionConfig),
    Gemma4Audio(models::gemma4::Gemma4RuntimeAudioConfig),
    Lfm2Vision(models::lfm2::Lfm2RuntimeVisionConfig),
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
    let image = load_image(image_path)?;
    prepare_multimodal_input_from_image(config_json, Some(&image))
}

pub fn prepare_multimodal_input_from_bytes(
    config_json: &serde_json::Value,
    image_bytes: &[u8],
) -> Result<PreparedMultimodalInput> {
    let image = load_image_from_bytes(image_bytes)?;
    prepare_multimodal_input_from_image(config_json, Some(&image))
}

fn prepare_multimodal_input_from_image(
    config_json: &serde_json::Value,
    image: Option<&image::DynamicImage>,
) -> Result<PreparedMultimodalInput> {
    let Some(image) = image else {
        return Ok(PreparedMultimodalInput::default());
    };

    match get_model_architecture(config_json)? {
        "Gemma3ForConditionalGeneration" | "PaliGemmaForConditionalGeneration" => {
            let (data, shape) = models::gemma3::prepare_gemma3_image_input(image, config_json)?;
            Ok(PreparedMultimodalInput {
                image: Some(PreparedImageInput { data, shape }),
                runtime_context: None,
            })
        }
        "SmolVLMForConditionalGeneration" => {
            let (data, shape) = models::smolvlm2::prepare_smolvlm2_image_input(image, config_json)?;
            Ok(PreparedMultimodalInput {
                image: Some(PreparedImageInput { data, shape }),
                runtime_context: None,
            })
        }
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
            let prepared = models::qwen3_5::prepare_qwen3_5_image_input(image, config_json)?;
            Ok(PreparedMultimodalInput {
                image: Some(PreparedImageInput {
                    data: prepared.pixels,
                    shape: prepared.shape,
                }),
                runtime_context: Some(ModelRuntimeContext::Qwen3_5Vision(prepared.runtime_vision)),
            })
        }
        "Gemma4ForConditionalGeneration" => {
            let prepared = models::gemma4::prepare_gemma4_image_input(image, config_json)?;
            Ok(PreparedMultimodalInput {
                image: Some(PreparedImageInput {
                    data: prepared.patches,
                    shape: prepared.shape,
                }),
                runtime_context: Some(ModelRuntimeContext::Gemma4Vision(prepared.runtime_vision)),
            })
        }
        "Lfm2VlForConditionalGeneration" => {
            let prepared = models::lfm2::prepare_lfm2_image_input(image, config_json)?;
            Ok(PreparedMultimodalInput {
                image: Some(PreparedImageInput {
                    data: prepared.patches,
                    shape: prepared.shape,
                }),
                runtime_context: Some(ModelRuntimeContext::Lfm2Vision(prepared.runtime_vision)),
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
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
            models::qwen3_5::interpolate_qwen3_5_prompt(
                config_json,
                match runtime_context {
                    Some(ModelRuntimeContext::Qwen3_5Vision(runtime_vision)) => {
                        Some(runtime_vision)
                    }
                    _ => None,
                },
                prompt,
            )?
        }
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

/// Split a multimodal token sequence into the text before and after the placeholder-token span.
pub fn split_placeholder_tokens(
    input_tokens: &[u32],
    mm_token_index: usize,
) -> Result<(&[u32], &[u32])> {
    let mm_token = mm_token_index as u32;
    let first_image_token_index = input_tokens
        .iter()
        .position(|&token| token == mm_token)
        .ok_or_else(|| {
            LLMError::InvalidModelConfig(format!(
                "multimodal prompt is missing image or audio token {mm_token_index}"
            ))
        })?;
    let last_mm_token_index = input_tokens
        .iter()
        .rposition(|&token| token == mm_token)
        .expect("mm token not found when searching for last occurence");

    Ok((
        &input_tokens[..first_image_token_index],
        &input_tokens[last_mm_token_index + 1..],
    ))
}

pub fn get_model(
    config_json: &serde_json::Value,
    max_sequence_length: usize,
    runtime_context: Option<&ModelRuntimeContext>,
    dtype: Dtype,
) -> Result<Box<dyn LLMModel>> {
    let arch = get_model_architecture(config_json)?;

    let model: Box<dyn LLMModel> =
        match arch {
            "Gemma2ForCausalLM" | "Gemma3ForCausalLM" => Box::new(
                models::gemma3::Gemma3Model::new("model", config_json, dtype)?,
            ),
            "Gemma3ForConditionalGeneration" | "PaliGemmaForConditionalGeneration" => Box::new(
                models::gemma3::Gemma3Model::new("language_model.model", config_json, dtype)?,
            ),
            "Gemma4ForConditionalGeneration" => Box::new(models::gemma4::Gemma4Model::new(
                "model.language_model",
                config_json,
                match runtime_context {
                    Some(ModelRuntimeContext::Gemma4Vision(runtime_vision)) => Some(runtime_vision),
                    _ => None,
                },
                match runtime_context {
                    Some(ModelRuntimeContext::Gemma4Audio(runtime_audio)) => Some(runtime_audio),
                    _ => None,
                },
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
            "Qwen3_5ForConditionalGeneration" | "Qwen3_5MoeForConditionalGeneration" => {
                Box::new(models::qwen3_5::Qwen3_5Model::new(
                    config_json,
                    max_sequence_length,
                    match runtime_context {
                        Some(ModelRuntimeContext::Qwen3_5Vision(runtime_vision)) => {
                            Some(runtime_vision)
                        }
                        _ => None,
                    },
                    dtype,
                )?)
            }
            "GraniteForCausalLM" | "GraniteMoeForCausalLM" | "GraniteMoeHybridForCausalLM" => {
                Box::new(models::granite::GraniteModel::new(config_json, dtype)?)
            }
            "DeepseekV3ForCausalLM" => {
                Box::new(models::deepseek::DeepSeekModel::new(config_json, dtype)?)
            }
            "GptOssForCausalLM" => Box::new(models::gpt_oss::GPTOssModel::new(config_json, dtype)?),
            "Lfm2ForCausalLM" | "Lfm2MoeForCausalLM" => Box::new(models::lfm2::Lfm2Model::new(
                "model",
                config_json,
                None,
                dtype,
            )?),
            "Lfm2VlForConditionalGeneration" => Box::new(models::lfm2::Lfm2Model::new(
                "model.language_model",
                config_json,
                match runtime_context {
                    Some(ModelRuntimeContext::Lfm2Vision(runtime_vision)) => Some(runtime_vision),
                    _ => None,
                },
                dtype,
            )?),
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

fn decode_f32_into(tensor_data: &[u8], dst: &mut [f32]) {
    assert_eq!(tensor_data.len() % 4, 0);
    assert_eq!(dst.len(), tensor_data.len() / 4);

    dst.par_iter_mut()
        .zip(tensor_data.par_chunks_exact(4))
        .for_each(|(out, bytes)| {
            *out = f32::from_le_bytes(bytes.try_into().unwrap());
        });
}

fn decode_bf16_to_f32_into(tensor_data: &[u8], dst: &mut [f32]) {
    assert_eq!(tensor_data.len() % 2, 0);
    assert_eq!(dst.len(), tensor_data.len() / 2);

    dst.par_iter_mut()
        .zip(tensor_data.par_chunks_exact(2))
        .for_each(|(out, bytes)| {
            *out = half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32();
        });
}

fn decode_f32_to_f16_into(tensor_data: &[u8], dst: &mut [half::f16]) {
    assert_eq!(tensor_data.len() % 4, 0);
    assert_eq!(dst.len(), tensor_data.len() / 4);

    dst.par_iter_mut()
        .zip(tensor_data.par_chunks_exact(4))
        .for_each(|(out, bytes)| {
            *out = half::f16::from_f32(f32::from_le_bytes(bytes.try_into().unwrap()));
        });
}

fn decode_bf16_to_f16_into(tensor_data: &[u8], dst: &mut [half::f16]) {
    assert_eq!(tensor_data.len() % 2, 0);
    assert_eq!(dst.len(), tensor_data.len() / 2);

    dst.par_iter_mut()
        .zip(tensor_data.par_chunks_exact(2))
        .for_each(|(out, bytes)| {
            *out =
                half::f16::from_f32(half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32());
        });
}

fn decode_f32_to_bf16_into(tensor_data: &[u8], dst: &mut [half::bf16]) {
    assert_eq!(tensor_data.len() % 4, 0);
    assert_eq!(dst.len(), tensor_data.len() / 4);

    dst.par_iter_mut()
        .zip(tensor_data.par_chunks_exact(4))
        .for_each(|(out, bytes)| {
            *out = half::bf16::from_f32(f32::from_le_bytes(bytes.try_into().unwrap()));
        });
}

fn decode_bf16_into(tensor_data: &[u8], dst: &mut [half::bf16]) {
    assert_eq!(tensor_data.len() % 2, 0);
    assert_eq!(dst.len(), tensor_data.len() / 2);

    dst.par_iter_mut()
        .zip(tensor_data.par_chunks_exact(2))
        .for_each(|(out, bytes)| {
            *out = half::bf16::from_le_bytes(bytes.try_into().unwrap());
        });
}

#[derive(Debug)]
struct PendingMoeTensor<T> {
    data: Vec<T>,
    expert_shape: Vec<usize>,
    loaded_experts: usize,
}

fn get_num_experts(config_json: &serde_json::Value) -> Option<usize> {
    ["num_experts", "n_routed_experts", "num_local_experts"]
        .into_iter()
        .find_map(|key| {
            config_json
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .filter(|count| *count > 0)
                .map(|count| count as usize)
        })
}

fn get_packed_moe_key(name: &str) -> Result<Option<(catgrad::prelude::Path, usize)>> {
    let components: Vec<&str> = name.split('.').collect();
    let Some(layer_idx) = components
        .iter()
        .position(|component| *component == "layers")
    else {
        return Ok(None);
    };
    if components
        .get(layer_idx + 1)
        .and_then(|component| component.parse::<usize>().ok())
        .is_none()
    {
        return Ok(None);
    }

    let Some(experts_rel_idx) = components[layer_idx + 2..]
        .iter()
        .position(|component| *component == "experts")
    else {
        return Ok(None);
    };
    let experts_idx = layer_idx + 2 + experts_rel_idx;
    let Some(expert_idx) = components
        .get(experts_idx + 1)
        .and_then(|component| component.parse::<usize>().ok())
    else {
        return Ok(None);
    };

    let mut packed_components = Vec::with_capacity(components.len() - 1);
    packed_components.extend_from_slice(&components[..experts_idx + 1]);
    packed_components.extend_from_slice(&components[experts_idx + 2..]);
    let key = path(packed_components).map_err(|err| {
        LLMError::InvalidModelConfig(format!("invalid packed MoE param path {name}: {}", err.0))
    })?;
    Ok(Some((key, expert_idx)))
}

fn insert_tensor_type(
    key: catgrad::prelude::Path,
    shape: Vec<usize>,
    dtype: Dtype,
    type_map: &mut BTreeMap<catgrad::prelude::Path, typecheck::Type>,
) {
    use catgrad::typecheck::*;

    let tensor_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape: ShapeExpr::Shape(shape.into_iter().map(NatExpr::Constant).collect()),
    }));
    type_map.insert(key, tensor_type);
}

fn insert_tensor_value<B: interpreter::Backend, T: interpreter::IntoTagged<B, 1>>(
    backend: &B,
    key: catgrad::prelude::Path,
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<T>,
    data_map: &mut BTreeMap<catgrad::prelude::Path, interpreter::Value<B>>,
    type_map: &mut BTreeMap<catgrad::prelude::Path, typecheck::Type>,
) -> Result<()> {
    let tensor =
        interpreter::tensor(backend, interpreter::Shape(shape.clone()), data).map_err(|err| {
            LLMError::InvalidModelConfig(format!("failed to create tensor {key}: {err:?}"))
        })?;
    data_map.insert(key.clone(), tensor);
    insert_tensor_type(key, shape, dtype, type_map);
    Ok(())
}

fn load_model_weights_for_dtype<B, T, DecodeF32, DecodeBF16>(
    model_paths: Vec<PathBuf>,
    backend: &B,
    dtype: Dtype,
    expert_count: Option<usize>,
    zero: T,
    decode_f32: DecodeF32,
    decode_bf16: DecodeBF16,
) -> Result<(interpreter::Parameters<B>, typecheck::Parameters, usize)>
where
    B: interpreter::Backend,
    T: interpreter::IntoTagged<B, 1> + Clone,
    DecodeF32: Fn(&[u8], &mut [T]),
    DecodeBF16: Fn(&[u8], &mut [T]),
{
    let mut type_map = BTreeMap::new();
    let mut data_map = BTreeMap::new();
    let mut pending_moe_tensors: BTreeMap<catgrad::prelude::Path, PendingMoeTensor<T>> =
        BTreeMap::new();
    let mut total_params = 0;

    for file_path in model_paths {
        let file = std::fs::File::open(file_path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&data)?;

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let tensor_data = view.data();
            let elements = match view.dtype() {
                safetensors::Dtype::F32 => tensor_data.len() / 4,
                safetensors::Dtype::BF16 => tensor_data.len() / 2,
                _ => {
                    return Err(LLMError::UnsupportedDtype(format!("{:?}", view.dtype())));
                }
            };

            if let Some(num_experts) = expert_count
                && let Some((packed_key, expert_idx)) = get_packed_moe_key(&name)?
            {
                let pending = pending_moe_tensors
                    .entry(packed_key.clone())
                    .or_insert_with(|| PendingMoeTensor {
                        data: vec![zero; num_experts * elements],
                        expert_shape: shape.clone(),
                        loaded_experts: 0,
                    });
                let start = expert_idx * elements;
                let end = start + elements;
                match view.dtype() {
                    safetensors::Dtype::F32 => {
                        decode_f32(tensor_data, &mut pending.data[start..end])
                    }
                    safetensors::Dtype::BF16 => {
                        decode_bf16(tensor_data, &mut pending.data[start..end])
                    }
                    _ => unreachable!(),
                }
                pending.loaded_experts += 1;
                total_params += elements;

                if pending.loaded_experts == num_experts {
                    let pending = pending_moe_tensors
                        .remove(&packed_key)
                        .expect("completed MoE tensor missing");
                    let mut packed_shape = vec![num_experts];
                    packed_shape.extend(pending.expert_shape);
                    insert_tensor_value(
                        backend,
                        packed_key,
                        packed_shape,
                        dtype,
                        pending.data,
                        &mut data_map,
                        &mut type_map,
                    )?;
                }
                continue;
            }

            let mut tensor = vec![zero; elements];
            match view.dtype() {
                safetensors::Dtype::F32 => decode_f32(tensor_data, &mut tensor),
                safetensors::Dtype::BF16 => decode_bf16(tensor_data, &mut tensor),
                _ => unreachable!(),
            }
            total_params += tensor.len();

            let key = path(name.split('.').collect()).map_err(|err| {
                LLMError::InvalidModelConfig(format!("invalid param path {name}: {}", err.0))
            })?;
            insert_tensor_value(
                backend,
                key,
                shape,
                dtype,
                tensor,
                &mut data_map,
                &mut type_map,
            )?;
        }
    }

    debug_assert!(pending_moe_tensors.is_empty());

    let parameter_values = interpreter::Parameters::from(data_map);
    let parameter_types = typecheck::Parameters::from(type_map);

    Ok((parameter_values, parameter_types, total_params))
}

pub fn load_model_weights<B: interpreter::Backend>(
    model_paths: Vec<PathBuf>,
    backend: &B,
    dtype: Dtype,
    expert_count: Option<usize>,
) -> Result<(interpreter::Parameters<B>, typecheck::Parameters, usize)> {
    if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
        return Err(LLMError::UnsupportedDtype(format!("{dtype:?}")));
    }

    match dtype {
        Dtype::F32 => load_model_weights_for_dtype(
            model_paths,
            backend,
            dtype,
            expert_count,
            0.0f32,
            decode_f32_into,
            decode_bf16_to_f32_into,
        ),
        Dtype::F16 => load_model_weights_for_dtype(
            model_paths,
            backend,
            dtype,
            expert_count,
            half::f16::from_f32(0.0),
            decode_f32_to_f16_into,
            decode_bf16_to_f16_into,
        ),
        Dtype::BF16 => load_model_weights_for_dtype(
            model_paths,
            backend,
            dtype,
            expert_count,
            half::bf16::from_f32(0.0),
            decode_f32_to_bf16_into,
            decode_bf16_into,
        ),
        Dtype::U32 => unreachable!(),
    }
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
        load_model_weights(model_paths, backend, dtype, get_num_experts(&config_json))?;

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
        .map(|(dtype, shape)| {
            Ok(interpreter::Value::Tensor(
                backend.zeros(shape.clone(), *dtype),
            ))
        })
        .collect()
}
