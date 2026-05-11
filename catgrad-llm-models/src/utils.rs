use catgrad::prelude::Dtype;

use crate::helpers::LLMModel;
use crate::models;
use crate::{LLMError, Result};

pub const AUDIO_FEATURE_SIZE: usize = 128;

#[derive(Debug, Clone)]
pub struct PreparedAudioFeatures {
    pub features: Vec<f32>,
    pub feature_shape: Vec<usize>,
    pub mask: Vec<f32>,
    pub mask_shape: Vec<usize>,
    pub num_mel_frames: usize,
    pub valid_mel_frames: usize,
}

#[derive(Debug, Clone)]
pub struct PatchedImageInput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub patch_grid_height: usize,
    pub patch_grid_width: usize,
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

pub fn prepare_multimodal_image_input(
    config_json: &serde_json::Value,
    image: &image::DynamicImage,
) -> Result<PreparedMultimodalInput> {
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

pub fn load_and_preprocess_dynamic_image(
    img: &image::DynamicImage,
    image_size: usize,
    patch_size: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let num_channels = 3;
    let resized_img = img.resize_to_fill(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb_img = resized_img.to_rgb8();
    let img = rgb_img.into_raw();

    let pixels: Vec<f32> = img.iter().map(|&x| x as f32 * (2. / 255.0) - 1.).collect();
    let aligned_image_size = (image_size / patch_size) * patch_size;
    let mut patches = vec![0.0; num_channels * aligned_image_size * aligned_image_size];
    for row in 0..aligned_image_size {
        for col in 0..aligned_image_size {
            for chan in 0..num_channels {
                patches[chan * aligned_image_size * aligned_image_size
                    + row * aligned_image_size
                    + col] = pixels[(row * image_size + col) * num_channels + chan];
            }
        }
    }
    Ok((
        patches,
        vec![1, num_channels, aligned_image_size, aligned_image_size],
    ))
}

fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let total_px = height * width;
    let target_px = max_patches * patch_size * patch_size;
    let factor = (target_px as f64 / total_px as f64).sqrt();
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_height = ((factor * height as f64).floor() as usize / side_mult) * side_mult;
    let mut target_width = ((factor * width as f64).floor() as usize / side_mult) * side_mult;

    if target_height == 0 && target_width == 0 {
        return Err(LLMError::InvalidModelConfig(format!(
            "gemma4 resized image would be 0x0 for patch_size {patch_size} and pooling_kernel_size {pooling_kernel_size}"
        )));
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        target_width = (((width as f64 / height as f64).floor() as usize).max(1) * side_mult)
            .min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height = (((height as f64 / width as f64).floor() as usize).max(1) * side_mult)
            .min(max_side_length);
    }

    if target_height * target_width > target_px {
        return Err(LLMError::InvalidModelConfig(format!(
            "gemma4 resized image {target_height}x{target_width} exceeds patch budget {max_patches}"
        )));
    }

    Ok((target_height, target_width))
}

pub fn convert_image_to_patches(
    image: &[f32],
    height: usize,
    width: usize,
    patch_size: usize,
) -> Vec<f32> {
    let num_channels = 3;
    let num_patches_height = height / patch_size;
    let num_patches_width = width / patch_size;
    let patch_dim = num_channels * patch_size * patch_size;
    let mut patches = vec![0.0; num_patches_height * num_patches_width * patch_dim];

    for patch_row in 0..num_patches_height {
        for patch_col in 0..num_patches_width {
            let patch_idx = patch_row * num_patches_width + patch_col;
            for inner_row in 0..patch_size {
                for inner_col in 0..patch_size {
                    for chan in 0..num_channels {
                        let src_row = patch_row * patch_size + inner_row;
                        let src_col = patch_col * patch_size + inner_col;
                        let src_idx = chan * height * width + src_row * width + src_col;
                        let dst_idx = patch_idx * patch_dim
                            + (inner_row * patch_size + inner_col) * num_channels
                            + chan;
                        patches[dst_idx] = image[src_idx];
                    }
                }
            }
        }
    }

    patches
}

#[derive(Debug, Clone)]
struct ResampleWeights {
    left: usize,
    weights: Vec<f64>,
}

fn catmull_rom_kernel(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.5 * x * x * x - 2.5 * x * x + 1.0
    } else if x < 2.0 {
        -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
    } else {
        0.0
    }
}

fn build_resample_weights(input_len: usize, output_len: usize) -> Vec<ResampleWeights> {
    let ratio = input_len as f64 / output_len as f64;
    let scale = ratio.max(1.0);
    let support = 2.0 * scale;

    (0..output_len)
        .map(|out_idx| {
            let sample_center = (out_idx as f64 + 0.5) * ratio;
            let left = ((sample_center - support).floor() as isize)
                .clamp(0, input_len.saturating_sub(1) as isize) as usize;
            let right = ((sample_center + support).ceil() as isize)
                .clamp((left + 1) as isize, input_len as isize) as usize;
            let sample_center = sample_center - 0.5;
            let mut weights: Vec<f64> = (left..right)
                .map(|src_idx| catmull_rom_kernel((src_idx as f64 - sample_center) / scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            ResampleWeights { left, weights }
        })
        .collect()
}

fn round_to_u8(x: f64) -> u8 {
    x.clamp(0.0, 255.0).round() as u8
}

fn resize_rgb_bicubic(
    image: &image::RgbImage,
    new_width: usize,
    new_height: usize,
) -> image::RgbImage {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let src = image.as_raw();
    let x_weights = build_resample_weights(width, new_width);
    let y_weights = build_resample_weights(height, new_height);

    let mut horizontal = vec![0u8; height * new_width * 3];
    for row in 0..height {
        for (out_col, weights) in x_weights.iter().enumerate() {
            let mut rgb = [0.0; 3];
            for (idx, &weight) in weights.weights.iter().enumerate() {
                let src_col = weights.left + idx;
                let src_base = (row * width + src_col) * 3;
                for chan in 0..3 {
                    rgb[chan] += src[src_base + chan] as f64 * weight;
                }
            }
            let out_base = (row * new_width + out_col) * 3;
            for chan in 0..3 {
                horizontal[out_base + chan] = round_to_u8(rgb[chan]);
            }
        }
    }

    let mut output = vec![0u8; new_width * new_height * 3];
    for (out_row, weights) in y_weights.iter().enumerate() {
        for col in 0..new_width {
            let mut rgb = [0.0; 3];
            for (idx, &weight) in weights.weights.iter().enumerate() {
                let src_row = weights.left + idx;
                let src_base = (src_row * new_width + col) * 3;
                for chan in 0..3 {
                    rgb[chan] += horizontal[src_base + chan] as f64 * weight;
                }
            }
            let out_base = (out_row * new_width + col) * 3;
            for chan in 0..3 {
                output[out_base + chan] = round_to_u8(rgb[chan]);
            }
        }
    }

    image::RgbImage::from_raw(new_width as u32, new_height as u32, output).unwrap()
}

pub fn load_and_patchify_dynamic_image(
    img: &image::DynamicImage,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<PatchedImageInput> {
    let (width, height) = (img.width() as usize, img.height() as usize);
    let (target_height, target_width) = get_aspect_ratio_preserving_size(
        height,
        width,
        patch_size,
        max_soft_tokens,
        pooling_kernel_size,
    )?;
    let rgb = resize_rgb_bicubic(&img.to_rgb8(), target_width, target_height).into_raw();
    let pixels: Vec<f32> = rgb.iter().map(|&x| x as f32 / 255.0).collect();

    let mut chw = vec![0.0; 3 * target_height * target_width];
    for row in 0..target_height {
        for col in 0..target_width {
            for chan in 0..3 {
                chw[chan * target_height * target_width + row * target_width + col] =
                    pixels[(row * target_width + col) * 3 + chan];
            }
        }
    }

    let patch_grid_height = target_height / patch_size;
    let patch_grid_width = target_width / patch_size;
    let data = convert_image_to_patches(&chw, target_height, target_width, patch_size);
    Ok(PatchedImageInput {
        shape: vec![
            1,
            patch_grid_height * patch_grid_width,
            3 * patch_size * patch_size,
        ],
        data,
        patch_grid_height,
        patch_grid_width,
    })
}
