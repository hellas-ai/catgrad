#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use crate::models::siglip::{SiglipVisionBackbone, SiglipVisionConfig};
use crate::utils::convert_image_to_patches;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize)]
struct LFM2RopeParameters {
    rope_theta: f32,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct Lfm2Config {
    hidden_size: usize,
    intermediate_size: Option<usize>,
    block_ff_dim: Option<usize>,
    block_ffn_dim_multiplier: f32,
    block_multiple_of: usize,
    block_auto_adjust_ff_dim: bool,
    full_attn_idxs: Vec<usize>,
    norm_eps: f32,
    conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
    rope_parameters: Option<LFM2RopeParameters>,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
    layer_types: Vec<String>,
}

impl Lfm2Config {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
            .or(self.block_ff_dim)
            .unwrap_or_default()
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.full_attn_idxs.contains(&layer_id)
            || (self.layer_types.len() == self.num_hidden_layers
                && self.layer_types[layer_id] == "full_attention")
    }
}

impl LLMConfig for Lfm2Config {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&layer_id| self.is_full_attention_layer(layer_id))
            .count()
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map_or(self.rope_theta, |p| p.rope_theta)
    }

    fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct Lfm2VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    layer_norm_eps: f32,
    patch_size: usize,
    num_patches: usize,
}

fn default_lfm2_vision_hidden_size() -> usize {
    768
}

fn default_lfm2_vision_intermediate_size() -> usize {
    3072
}

fn default_lfm2_vision_hidden_layers() -> usize {
    12
}

fn default_lfm2_vision_num_attention_heads() -> usize {
    12
}

fn default_lfm2_vision_patch_size() -> usize {
    16
}

fn default_lfm2_vision_num_patches() -> usize {
    256
}

fn default_lfm2_projector_hidden_size() -> usize {
    2560
}

fn default_lfm2_projector_bias() -> bool {
    true
}

fn default_lfm2_projector_use_layernorm() -> bool {
    true
}

fn default_lfm2_downsample_factor() -> usize {
    2
}

fn default_lfm2_max_image_tokens() -> usize {
    256
}

fn default_lfm2_min_image_tokens() -> usize {
    64
}

impl Lfm2VisionConfig {
    fn patches_per_side(&self) -> crate::Result<usize> {
        let side = (self.num_patches as f64).sqrt() as usize;
        if side * side != self.num_patches {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "lfm2_vl vision num_patches must be a square, got {}",
                self.num_patches
            )));
        }
        Ok(side)
    }

    fn image_size(&self) -> crate::Result<usize> {
        Ok(self.patches_per_side()? * self.patch_size)
    }

    fn to_backbone_config(&self) -> crate::Result<SiglipVisionConfig> {
        Ok(SiglipVisionConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            layer_norm_eps: self.layer_norm_eps,
            patch_size: self.patch_size,
            image_size: self.image_size()?,
            projection_dim: 0,
            num_image_tokens: 0,
        })
    }
}

impl Default for Lfm2VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_lfm2_vision_hidden_size(),
            intermediate_size: default_lfm2_vision_intermediate_size(),
            num_hidden_layers: default_lfm2_vision_hidden_layers(),
            num_attention_heads: default_lfm2_vision_num_attention_heads(),
            layer_norm_eps: 1e-6,
            patch_size: default_lfm2_vision_patch_size(),
            num_patches: default_lfm2_vision_num_patches(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Lfm2RuntimeVisionConfig {
    pub image_height: usize,
    pub image_width: usize,
    pub patch_grid_height: usize,
    pub patch_grid_width: usize,
    pub num_soft_tokens_per_image: usize,
}

#[derive(Debug, Clone)]
pub struct Lfm2PreparedImageInput {
    pub patches: Vec<f32>,
    pub shape: Vec<usize>,
    pub runtime_vision: Lfm2RuntimeVisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Lfm2ModelConfig {
    Vlm {
        text_config: Lfm2Config,
        vision_config: Lfm2VisionConfig,
        image_token_id: usize,
        #[serde(default = "default_lfm2_projector_hidden_size")]
        projector_hidden_size: usize,
        #[serde(default = "default_lfm2_projector_bias")]
        projector_bias: bool,
        #[serde(default = "default_lfm2_projector_use_layernorm")]
        projector_use_layernorm: bool,
        #[serde(default = "default_lfm2_downsample_factor")]
        downsample_factor: usize,
        #[serde(default = "default_lfm2_max_image_tokens")]
        max_image_tokens: usize,
    },
    Text(Lfm2Config),
}

#[derive(Debug, Clone)]
struct Lfm2MultimodalConfig {
    vision_config: SiglipVisionConfig,
    base_patch_grid_side: usize,
    image_token_index: usize,
    image_height: usize,
    image_width: usize,
    patch_grid_height: usize,
    patch_grid_width: usize,
    mm_tokens_per_image: usize,
    projector_hidden_size: usize,
    projector_bias: bool,
    projector_use_layernorm: bool,
    downsample_factor: usize,
}

impl Lfm2MultimodalConfig {
    fn new(
        vision_config: Lfm2VisionConfig,
        image_token_index: usize,
        projector_hidden_size: usize,
        projector_bias: bool,
        projector_use_layernorm: bool,
        downsample_factor: usize,
        max_image_tokens: usize,
        runtime_vision: Option<&Lfm2RuntimeVisionConfig>,
    ) -> crate::Result<Self> {
        if downsample_factor == 0 {
            return Err(crate::LLMError::InvalidModelConfig(
                "lfm2_vl downsample_factor must be > 0".to_string(),
            ));
        }
        if projector_hidden_size == 0 {
            return Err(crate::LLMError::InvalidModelConfig(
                "lfm2_vl projector_hidden_size must be > 0".to_string(),
            ));
        }

        let base_patch_grid_side = vision_config.patches_per_side()?;
        let projected_side = (max_image_tokens as f64).sqrt() as usize;
        if projected_side * projected_side != max_image_tokens {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "lfm2_vl max_image_tokens must be a square, got {max_image_tokens}"
            )));
        }
        let default_patch_grid_side = projected_side * downsample_factor;
        let runtime_vision = runtime_vision.cloned().unwrap_or(Lfm2RuntimeVisionConfig {
            image_height: default_patch_grid_side * vision_config.patch_size,
            image_width: default_patch_grid_side * vision_config.patch_size,
            patch_grid_height: default_patch_grid_side,
            patch_grid_width: default_patch_grid_side,
            num_soft_tokens_per_image: max_image_tokens,
        });

        if runtime_vision.patch_grid_height == 0 || runtime_vision.patch_grid_width == 0 {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "lfm2_vl patch grid must be non-zero, got {}x{}",
                runtime_vision.patch_grid_height, runtime_vision.patch_grid_width
            )));
        }
        if !runtime_vision
            .patch_grid_height
            .is_multiple_of(downsample_factor)
            || !runtime_vision
                .patch_grid_width
                .is_multiple_of(downsample_factor)
        {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "lfm2_vl patch grid {}x{} must be divisible by downsample_factor {downsample_factor}",
                runtime_vision.patch_grid_height, runtime_vision.patch_grid_width
            )));
        }
        let image_height = if runtime_vision.image_height == 0 {
            runtime_vision.patch_grid_height * vision_config.patch_size
        } else {
            runtime_vision.image_height
        };
        let image_width = if runtime_vision.image_width == 0 {
            runtime_vision.patch_grid_width * vision_config.patch_size
        } else {
            runtime_vision.image_width
        };
        let mm_tokens_per_image = if runtime_vision.num_soft_tokens_per_image == 0 {
            (runtime_vision.patch_grid_height / downsample_factor)
                * (runtime_vision.patch_grid_width / downsample_factor)
        } else {
            runtime_vision.num_soft_tokens_per_image
        };

        Ok(Self {
            vision_config: vision_config.to_backbone_config()?,
            base_patch_grid_side,
            image_token_index,
            image_height,
            image_width,
            patch_grid_height: runtime_vision.patch_grid_height,
            patch_grid_width: runtime_vision.patch_grid_width,
            mm_tokens_per_image,
            projector_hidden_size,
            projector_bias,
            projector_use_layernorm,
            downsample_factor,
        })
    }
}

fn lfm2_smart_resize(
    height: usize,
    width: usize,
    patch_size: usize,
    downsample_factor: usize,
    min_image_tokens: usize,
    max_image_tokens: usize,
) -> crate::Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(crate::LLMError::InvalidModelConfig(
            "lfm2_vl image dimensions must be non-zero".to_string(),
        ));
    }
    let total_factor = patch_size * downsample_factor;
    let min_pixels = min_image_tokens * patch_size * patch_size * downsample_factor.pow(2);
    let max_pixels = max_image_tokens * patch_size * patch_size * downsample_factor.pow(2);

    let mut resized_height =
        total_factor.max(((height + total_factor / 2) / total_factor) * total_factor);
    let mut resized_width =
        total_factor.max(((width + total_factor / 2) / total_factor) * total_factor);

    if resized_height * resized_width > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        resized_height = total_factor
            .max((((height as f64 / beta).floor() as usize) / total_factor) * total_factor);
        resized_width = total_factor
            .max((((width as f64 / beta).floor() as usize) / total_factor) * total_factor);
    } else if resized_height * resized_width < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        resized_height =
            ((height as f64 * beta).ceil() as usize).div_ceil(total_factor) * total_factor;
        resized_width =
            ((width as f64 * beta).ceil() as usize).div_ceil(total_factor) * total_factor;
    }

    Ok((resized_height, resized_width))
}

pub fn prepare_lfm2_image_input(
    image: &image::DynamicImage,
    config_json: &serde_json::Value,
) -> crate::Result<Lfm2PreparedImageInput> {
    let (vision_config, downsample_factor, max_image_tokens) =
        match serde_json::from_value(config_json.clone())? {
            Lfm2ModelConfig::Vlm {
                vision_config,
                downsample_factor,
                max_image_tokens,
                ..
            } => (vision_config, downsample_factor, max_image_tokens),
            Lfm2ModelConfig::Text(_) => {
                return Err(crate::LLMError::InvalidModelConfig(
                    "lfm2 missing multimodal configuration".to_string(),
                ));
            }
        };

    let (image_height, image_width) = lfm2_smart_resize(
        image.height() as usize,
        image.width() as usize,
        vision_config.patch_size,
        downsample_factor,
        default_lfm2_min_image_tokens(),
        max_image_tokens,
    )?;
    let resized = image.resize_exact(
        image_width as u32,
        image_height as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8().into_raw();
    let pixels: Vec<f32> = rgb
        .iter()
        .map(|&x| x as f32 * (2.0 / 255.0) - 1.0)
        .collect();

    let patch_grid_height = image_height / vision_config.patch_size;
    let patch_grid_width = image_width / vision_config.patch_size;
    let patch_dim = 3 * vision_config.patch_size * vision_config.patch_size;
    let mut chw = vec![0.0; 3 * image_height * image_width];
    for row in 0..image_height {
        for col in 0..image_width {
            for chan in 0..3 {
                chw[chan * image_height * image_width + row * image_width + col] =
                    pixels[(row * image_width + col) * 3 + chan];
            }
        }
    }
    let patches =
        convert_image_to_patches(&chw, image_height, image_width, vision_config.patch_size);

    Ok(Lfm2PreparedImageInput {
        shape: vec![1, patch_grid_height * patch_grid_width, patch_dim],
        patches,
        runtime_vision: Lfm2RuntimeVisionConfig {
            image_height,
            image_width,
            patch_grid_height,
            patch_grid_width,
            num_soft_tokens_per_image: (patch_grid_height / downsample_factor)
                * (patch_grid_width / downsample_factor),
        },
    })
}

fn interpolate_lfm2_prompt(mm: &Lfm2MultimodalConfig, prompt: &str) -> String {
    let image_tokens = "<image>".repeat(mm.mm_tokens_per_image);
    let wrapped = format!("<|image_start|>{image_tokens}<|image_end|>");
    if prompt.contains("<|image_start|><image><|image_end|>") {
        prompt.replace("<|image_start|><image><|image_end|>", &wrapped)
    } else if prompt.contains("<|image_start|>") || prompt.contains("<|image_end|>") {
        prompt.replace("<image>", &image_tokens)
    } else {
        prompt.replace("<image>", &wrapped)
    }
}

#[derive(Debug, Clone)]
pub struct Lfm2Model {
    root: String,
    config: Lfm2Config,
    layer_to_cache_id: Vec<Option<usize>>,
    layer_to_linear_id: Vec<Option<usize>>,
    num_linear_layers: usize,
    dtype: Dtype,
    multimodal: Option<Lfm2MultimodalConfig>,
}

impl LLMModel for Lfm2Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn multimodal_metadata(&self) -> Option<MultimodalMetadata> {
        let mm = self.multimodal.as_ref()?;
        Some(MultimodalMetadata {
            image_token_index: mm.image_token_index,
            mm_tokens_per_image: mm.mm_tokens_per_image,
            hidden_size: self.config.hidden_size,
            image_size: mm.image_height.max(mm.image_width),
            patch_size: mm.vision_config.patch_size,
        })
    }

    fn multimodal_vision_module(&self) -> Option<Box<dyn DynModule>> {
        let mm = self.multimodal.as_ref()?;
        Some(Box::new(Lfm2VisionModel {
            vision_config: mm.vision_config.clone(),
            base_patch_grid_side: mm.base_patch_grid_side,
            patch_grid_height: mm.patch_grid_height,
            patch_grid_width: mm.patch_grid_width,
            text_hidden_size: self.config.hidden_size,
            projector_hidden_size: mm.projector_hidden_size,
            projector_bias: mm.projector_bias,
            projector_use_layernorm: mm.projector_use_layernorm,
            downsample_factor: mm.downsample_factor,
        }))
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        self.multimodal.as_ref()?;
        Some(Box::new(Lfm2MultimodalModel {
            language_model: self.clone(),
        }))
    }

    fn multimodal_interpolate_prompt(&self, prompt: &str) -> Option<String> {
        let mm = self.multimodal.as_ref()?;
        Some(interpolate_lfm2_prompt(mm, prompt))
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let dtype = self.dtype();
        vec![
            (
                dtype,
                Shape(vec![
                    self.config.num_hidden_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_head_dim(),
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.config.num_hidden_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_head_dim(),
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.config.hidden_size,
                    self.config.conv_l_cache,
                ]),
            ),
        ]
    }
}

impl Lfm2Model {
    pub fn new(
        root: &str,
        config_json: &serde_json::Value,
        runtime_vision: Option<&Lfm2RuntimeVisionConfig>,
        dtype: Dtype,
    ) -> crate::Result<Self> {
        let (config, multimodal) = match serde_json::from_value(config_json.clone())? {
            Lfm2ModelConfig::Text(config) => (config, None),
            Lfm2ModelConfig::Vlm {
                text_config,
                vision_config,
                image_token_id,
                projector_hidden_size,
                projector_bias,
                projector_use_layernorm,
                downsample_factor,
                max_image_tokens,
            } => (
                text_config,
                Some(Lfm2MultimodalConfig::new(
                    vision_config,
                    image_token_id,
                    projector_hidden_size,
                    projector_bias,
                    projector_use_layernorm,
                    downsample_factor,
                    max_image_tokens,
                    runtime_vision,
                )?),
            ),
        };
        assert!(config.conv_l_cache > 0, "lfm2 conv_l_cache must be > 0");
        let mut layer_to_cache_id = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_to_linear_id = Vec::with_capacity(config.num_hidden_layers);
        let mut next_cache_id = 0;
        let mut next_linear_id = 0;
        for layer_id in 0..config.num_hidden_layers {
            if config.is_full_attention_layer(layer_id) {
                layer_to_cache_id.push(Some(next_cache_id));
                layer_to_linear_id.push(None);
                next_cache_id += 1;
            } else {
                layer_to_cache_id.push(None);
                layer_to_linear_id.push(Some(next_linear_id));
                next_linear_id += 1;
            }
        }

        Ok(Self {
            root: root.to_string(),
            config,
            layer_to_cache_id,
            layer_to_linear_id,
            num_linear_layers: next_linear_id,
            dtype,
            multimodal,
        })
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.config.is_full_attention_layer(layer_id)
    }

    fn attention(
        &self,
        builder: &Builder,
        cache_layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.hidden_size / num_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());
        let k = linear_no_bias(
            builder,
            dim,
            dim / rep,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = linear_no_bias(builder, dim, dim / rep, p.extend(["v_proj"]).unwrap(), x);

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = rmsnorm::<4>(
            builder,
            self.config.norm_eps,
            p.extend(["q_layernorm"]).unwrap(),
            q,
        );
        let k = rmsnorm::<4>(
            builder,
            self.config.norm_eps,
            p.extend(["k_layernorm"]).unwrap(),
            k,
        );

        let q = apply_rope_embedding(
            builder,
            pos.clone(),
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            q,
        );
        let k = apply_rope_embedding(
            builder,
            pos,
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            k,
        );

        let (k, v) = cache.update_kv_cache(builder, cache_layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let denom = cast(builder, denom, dtype(builder, attn.clone()));
        let mut attn = attn / denom;

        let attention_mask = cast(builder, attention_mask, dtype(builder, attn.clone()));
        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["out_proj"]).unwrap(), attn)
    }

    fn short_conv(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let in_proj = linear_no_bias(
            builder,
            self.config.hidden_size,
            3 * self.config.hidden_size,
            p.extend(["in_proj"]).unwrap(),
            x,
        );
        let bcx = transpose(builder, 1, 2, in_proj);

        let bcx = chunk(builder, 1, 3, self.config.hidden_size, bcx);
        let b = bcx[0].clone();
        let c = bcx[1].clone();
        let x = bcx[2].clone();

        let bx = b * x;

        let [batch_size, hidden_dim, s] = unpack::<3>(builder, shape(builder, bx.clone()));
        let cache_len = self.config.conv_l_cache;
        let linear_layer_id =
            self.layer_to_linear_id[layer_id].expect("short-conv layer missing linear state index");
        let conv_state = cache
            .linear_cache
            .as_ref()
            .expect("lfm2 short_conv requires linear cache")[linear_layer_id][0]
            .clone();

        // HF: `cache_position = cache_position.clamp(0, self.L_cache - 1)`
        let sh_pos = shape(builder, nat_to_u32(builder, pos.clone()));
        let pos_u32 = nat_to_u32(builder, pos.clone());
        let zero_pos = constant(builder, 0u32, &sh_pos);
        let pos_f32 = cast(builder, pos_u32, Dtype::F32);
        let pos_f32 = clamp(builder, pos_f32, 0.0, (cache_len - 1) as f32);
        let u32_dtype = dtype_constant(builder, Dtype::U32);
        let pos_clamped_u32 = cast(builder, pos_f32, u32_dtype);

        // HF `if cache_position[0] > 0: ... else: ...`
        let is_decode = gt(builder, nat_to_u32(builder, pos), zero_pos);

        let results = cond(
            builder,
            is_decode,
            |b, args: Vec<Var>| {
                let [bx, s, _batch_size, _hidden_dim, conv_state, pos_clamped_u32] =
                    args.try_into().unwrap();

                // `conv_state = conv_state.roll(shifts=-1, dims=-1)`
                let rolled_state = {
                    let tail = slice(b, 2, 1, cache_len - 1, conv_state.clone());
                    let head = slice(b, 2, 0, 1, conv_state);
                    concat(b, 2, tail, head)
                };

                // HF equivalent of indexing at `cache_position`: build one-hot over cache axis.
                // `conv_state[:, :, cache_position] = Bx`
                let sh_k = shape!(b, cache_len);
                let positions = arange(b, cache_len);
                let pos_vec = broadcast(b, sh_k, pos_clamped_u32);
                let one_hot = eq(b, positions, pos_vec);

                let sh_state = shape(b, rolled_state.clone());

                let bx_decode = slice(b, 2, 0, 1, bx);
                let bx_decode = broadcast(b, sh_state, bx_decode);
                let out_linear_state_decode = where_broadcast(b, one_hot, bx_decode, rolled_state);

                // Use the helper for decoding: pass out_linear_state_decode with 0 padding.
                let conv_out_decode = padded_depthwise_conv1d_no_bias(
                    b,
                    p.extend(["conv"]).unwrap(),
                    cache_len,
                    out_linear_state_decode.clone(),
                    s,
                );

                vec![conv_out_decode, out_linear_state_decode]
            },
            |b, args: Vec<Var>| {
                let [bx, s, batch_size, hidden_dim, _conv_state, _pos_clamped_u32] =
                    args.try_into().unwrap();

                // Use the helper for prefill: pass bx with causal padding.
                let conv_out_prefill = depthwise_conv1d_no_bias(
                    b,
                    p.extend(["conv"]).unwrap(),
                    cache_len,
                    bx.clone(),
                    cache_len - 1,
                );

                // `conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))`
                let zeros_prefill_state = zeros(
                    b,
                    &shape!(b, batch_size, hidden_dim, cache_len),
                    dtype(b, bx.clone()),
                );
                let x_padded_prefill_state = concat(b, 2, zeros_prefill_state, bx);
                let out_linear_state_prefill = slice(b, 2, s, cache_len, x_padded_prefill_state);

                vec![conv_out_prefill, out_linear_state_prefill]
            },
            vec![bx, s, batch_size, hidden_dim, conv_state, pos_clamped_u32],
        );

        let conv_out = results[0].clone();
        let out_linear_state = results[1].clone();

        cache
            .linear_cache
            .as_mut()
            .expect("lfm2 short_conv requires mutable linear cache")[linear_layer_id][0] =
            out_linear_state;

        let y = c * conv_out;
        let y = transpose(builder, 1, 2, y);

        linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["out_proj"]).unwrap(),
            y,
        )
    }

    fn feed_forward(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let mut intermediate_size = self.config.intermediate_size();

        if self.config.block_auto_adjust_ff_dim {
            intermediate_size = 2 * intermediate_size / 3;
            intermediate_size =
                (self.config.block_ffn_dim_multiplier * intermediate_size as f32) as usize;
            intermediate_size = self.config.block_multiple_of
                * intermediate_size.div_ceil(self.config.block_multiple_of);
        }

        let gated = linear_no_bias(
            builder,
            self.config.hidden_size,
            intermediate_size,
            p.extend(["w1"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            self.config.hidden_size,
            intermediate_size,
            p.extend(["w3"]).unwrap(),
            x,
        );
        let x = silu(builder, gated) * up;

        linear_no_bias(
            builder,
            intermediate_size,
            self.config.hidden_size,
            p.extend(["w2"]).unwrap(),
            x,
        )
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.norm_eps,
            p.extend(["operator_norm"]).unwrap(),
            x,
        );
        let x = if self.is_full_attention_layer(layer_id) {
            let cache_layer_id = self.layer_to_cache_id[layer_id]
                .expect("full-attention layer missing KV cache index");
            self.attention(
                builder,
                cache_layer_id,
                attention_mask,
                cache,
                pos,
                p.extend(["self_attn"]).unwrap(),
                x,
            )
        } else {
            self.short_conv(
                builder,
                layer_id,
                cache,
                pos,
                p.extend(["conv"]).unwrap(),
                x,
            )
        };

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.norm_eps,
            p.extend(["ffn_norm"]).unwrap(),
            x,
        );
        let x = self.feed_forward(builder, p.extend(["feed_forward"]).unwrap(), x);
        x + res
    }

    fn embed_tokens(&self, builder: &Builder, root: Path, x: Var) -> Var {
        embeddings(builder, root.extend(["embed_tokens"]).unwrap(), x)
    }

    fn root_path(&self, prefix: Path) -> Path {
        if self.root.is_empty() {
            prefix
        } else {
            prefix.extend(self.root.split('.')).unwrap()
        }
    }

    fn forward_embeddings(
        &self,
        builder: &Builder,
        root: Path,
        mut x: Var,
        in_k: Var,
        in_v: Var,
        in_conv: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let mut cache = Cache::init(
            builder,
            &self.config,
            max_positions.clone(),
            max_positions,
            in_k.clone(),
            in_v,
        );

        cache.linear_cache = Some(
            (0..self.num_linear_layers)
                .map(|layer_id| {
                    let layer = slice(builder, 0, layer_id, 1, in_conv.clone());
                    let conv_state = squeeze::<4, 3>(builder, 0, layer);
                    vec![conv_state]
                })
                .collect(),
        );
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k));
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s, pos.clone());

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                pos.clone(),
                root.extend(["layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm::<3>(
            builder,
            self.config.norm_eps,
            root.extend(["embedding_norm"]).unwrap(),
            x,
        );

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(["embed_tokens"]).unwrap(),
            x,
        );
        x = argmax(builder, x);

        let (out_k, out_v) = cache.get_kv_cache(builder);
        let out_conv = {
            let states = cache
                .linear_cache
                .as_ref()
                .expect("lfm2 cache missing output linear cache");
            let mut iter = states.iter();
            let first = iter.next().expect("lfm2 cache linear cache missing")[0].clone();
            let mut out = unsqueeze::<3, 4>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<3, 4>(builder, 0, state[0].clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        vec![x, out_k, out_v, out_conv]
    }
}

impl DynModule for Lfm2Model {
    fn path(&self) -> Path {
        path(vec!["lfm2"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, in_conv, max_positions]: [Var; 5] =
            args.try_into().expect("expected 5 inputs");
        let root = self.root_path(self.path());
        let x = self.embed_tokens(builder, root.clone(), x);
        self.forward_embeddings(builder, root, x, in_k, in_v, in_conv, max_positions)
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let (mut source, mut target) = llm_type(&self.config, self.dtype());
        let max_positions = source.pop().expect("lfm2 missing max_positions nat input");
        let batch_size = NatExpr::Var(0);
        let num_linear_layers = NatExpr::Constant(self.num_linear_layers);
        let hidden_size = NatExpr::Constant(self.config.hidden_size);
        let conv_l_cache = NatExpr::Constant(self.config.conv_l_cache);
        let t_conv = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers,
                batch_size,
                hidden_size,
                conv_l_cache,
            ]),
        }));
        source.push(t_conv.clone());
        source.push(max_positions);
        target.push(t_conv);
        (source, target)
    }
}

fn lfm2_pixel_unshuffle(
    builder: &Builder,
    hidden_size: usize,
    patch_grid_height: usize,
    patch_grid_width: usize,
    downsample_factor: usize,
    x: Var,
) -> Var {
    let [b, _, _] = unpack::<3>(builder, shape(builder, x.clone()));
    let factor_sq = downsample_factor.pow(2);
    let output_tokens =
        (patch_grid_height / downsample_factor) * (patch_grid_width / downsample_factor);

    let x = reshape(
        builder,
        shape!(builder, b, patch_grid_height, patch_grid_width, hidden_size),
        x,
    );
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            patch_grid_height,
            patch_grid_width / downsample_factor,
            hidden_size * downsample_factor
        ),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            patch_grid_width / downsample_factor,
            patch_grid_height / downsample_factor,
            hidden_size * factor_sq
        ),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    reshape(
        builder,
        shape!(builder, b, output_tokens, hidden_size * factor_sq),
        x,
    )
}

fn lfm2_projector_linear(
    builder: &Builder,
    input_dim: usize,
    output_dim: usize,
    bias: bool,
    p: Path,
    x: Var,
) -> Var {
    if bias {
        linear(builder, input_dim, output_dim, p, x)
    } else {
        linear_no_bias(builder, input_dim, output_dim, p, x)
    }
}

fn lfm2_multi_modal_projector(
    builder: &Builder,
    vision_config: &SiglipVisionConfig,
    patch_grid_height: usize,
    patch_grid_width: usize,
    text_hidden_size: usize,
    projector_hidden_size: usize,
    projector_bias: bool,
    projector_use_layernorm: bool,
    downsample_factor: usize,
    p: Path,
    x: Var,
) -> Var {
    let in_channels = vision_config.hidden_size * downsample_factor.pow(2);
    let x = lfm2_pixel_unshuffle(
        builder,
        vision_config.hidden_size,
        patch_grid_height,
        patch_grid_width,
        downsample_factor,
        x,
    );
    let x = if projector_use_layernorm {
        layernorm(builder, 1e-5, p.extend(["layer_norm"]).unwrap(), x)
    } else {
        x
    };
    let x = lfm2_projector_linear(
        builder,
        in_channels,
        projector_hidden_size,
        projector_bias,
        p.extend(["linear_1"]).unwrap(),
        x,
    );
    let x = gelu(builder, x);
    lfm2_projector_linear(
        builder,
        projector_hidden_size,
        text_hidden_size,
        projector_bias,
        p.extend(["linear_2"]).unwrap(),
        x,
    )
}

#[derive(Debug, Clone)]
struct Lfm2VisionModel {
    vision_config: SiglipVisionConfig,
    base_patch_grid_side: usize,
    patch_grid_height: usize,
    patch_grid_width: usize,
    text_hidden_size: usize,
    projector_hidden_size: usize,
    projector_bias: bool,
    projector_use_layernorm: bool,
    downsample_factor: usize,
}

impl DynModule for Lfm2VisionModel {
    fn path(&self) -> Path {
        path(vec!["Lfm2VlVision"]).expect("invalid model path")
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;

        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [pixels]: [Var; 1] = args.try_into().expect("expected 1 input");
        let backbone = SiglipVisionBackbone {};
        let x = backbone.vision_model_from_patches(
            builder,
            &self.vision_config,
            self.base_patch_grid_side,
            self.patch_grid_height,
            self.patch_grid_width,
            path(vec!["model", "vision_tower", "vision_model"]).unwrap(),
            pixels,
        );
        let x = lfm2_multi_modal_projector(
            builder,
            &self.vision_config,
            self.patch_grid_height,
            self.patch_grid_width,
            self.text_hidden_size,
            self.projector_hidden_size,
            self.projector_bias,
            self.projector_use_layernorm,
            self.downsample_factor,
            path(vec!["model", "multi_modal_projector"]).unwrap(),
            x,
        );
        vec![x]
    }
}

#[derive(Debug, Clone)]
struct Lfm2MultimodalModel {
    language_model: Lfm2Model,
}

impl Lfm2MultimodalModel {
    fn forward_image_and_texts(
        &self,
        builder: &Builder,
        text_before: Var,
        image: Var,
        text_after: Var,
        in_k: Var,
        in_v: Var,
        in_conv: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let language_root = self.language_model.root_path(Path::empty());
        let text_before =
            self.language_model
                .embed_tokens(builder, language_root.clone(), text_before);
        let text_after =
            self.language_model
                .embed_tokens(builder, language_root.clone(), text_after);
        let embeddings = concat(builder, 1, text_before, image);
        let embeddings = concat(builder, 1, embeddings, text_after);
        self.language_model.forward_embeddings(
            builder,
            language_root,
            embeddings,
            in_k,
            in_v,
            in_conv,
            max_positions,
        )
    }
}

impl DynModule for Lfm2MultimodalModel {
    fn path(&self) -> Path {
        path(vec!["Lfm2Vl"]).expect("invalid model path")
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::{NatExpr, TypeExpr};
        let t = Type::Tensor(TypeExpr::Var(0));
        (
            vec![
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                Type::Nat(NatExpr::Var(3)),
            ],
            vec![t.clone(), t.clone(), t.clone(), t],
        )
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [
            text_before,
            image,
            text_after,
            in_k,
            in_v,
            in_conv,
            max_positions,
        ]: [Var; 7] = args.try_into().expect("expected 7 inputs");
        self.forward_image_and_texts(
            builder,
            text_before,
            image,
            text_after,
            in_k,
            in_v,
            in_conv,
            max_positions,
        )
    }
}
