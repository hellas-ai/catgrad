use catgrad::prelude::Dtype;
use std::path::Path;

use crate::helpers::LLMModel;
use crate::models;
use crate::utils::{load_image, load_image_from_bytes};
use crate::{LLMError, Result};

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
            "NemotronHForCausalLM" => Box::new(models::nemotron::NemotronModel::new(
                "",
                config_json,
                dtype,
            )?),
            "NemotronH_Nano_Omni_Reasoning_V3" => Box::new(models::nemotron::NemotronModel::new(
                "language_model",
                config_json,
                dtype,
            )?),
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
