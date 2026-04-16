#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Qwen3_5Config {
    text_config: Qwen3_5TextConfig,
    #[serde(default)]
    pub vision_config: Qwen3_5VisionConfig,
    #[serde(default)]
    pub image_token_id: Option<usize>,
    #[serde(default)]
    pub vision_start_token_id: Option<usize>,
    #[serde(default)]
    pub vision_end_token_id: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Qwen3_5VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default)]
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub out_hidden_size: usize,
}

#[derive(Debug, Clone)]
struct Qwen3_5MultimodalConfig {
    vision_config: Qwen3_5VisionConfig,
    image_token_index: usize,
    image_height: usize,
    image_width: usize,
    base_grid_size: usize,
    grid_t: usize,
    raw_grid_height: usize,
    raw_grid_width: usize,
    merged_grid_height: usize,
    merged_grid_width: usize,
    mm_tokens_per_image: usize,
    rope_delta: f32,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Qwen3_5RuntimeVisionConfig {
    pub resized_height: usize,
    pub resized_width: usize,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen3_5PreparedImageInput {
    pub pixels: Vec<f32>,
    pub shape: Vec<usize>,
    pub runtime_vision: Qwen3_5RuntimeVisionConfig,
}

// Qwen's full theoretical range reaches 4-16384 visual tokens per image, but the
// recommended practical range for inference is 256-1280 to avoid excessive memory use.
const QWEN3_5_IMAGE_MIN_PIXELS: usize = 256 * 32 * 32;
const QWEN3_5_IMAGE_MAX_PIXELS: usize = 1280 * 32 * 32;
const QWEN3_5_IMAGE_PATCH_SIZE: usize = 16;
const QWEN3_5_IMAGE_TEMPORAL_PATCH_SIZE: usize = 2;
const QWEN3_5_IMAGE_MERGE_SIZE: usize = 2;
const QWEN3_5_IMAGE_MEAN: f32 = 0.5;
const QWEN3_5_IMAGE_STD: f32 = 0.5;

fn default_vision_patch_size() -> usize {
    QWEN3_5_IMAGE_PATCH_SIZE
}

fn default_temporal_patch_size() -> usize {
    QWEN3_5_IMAGE_TEMPORAL_PATCH_SIZE
}

fn default_in_channels() -> usize {
    3
}

fn default_spatial_merge_size() -> usize {
    QWEN3_5_IMAGE_MERGE_SIZE
}

impl Qwen3_5VisionConfig {
    fn is_configured(&self) -> bool {
        self.hidden_size > 0
            && self.intermediate_size > 0
            && self.depth > 0
            && self.num_heads > 0
            && self.num_position_embeddings > 0
    }

    fn base_grid_size(&self) -> crate::Result<usize> {
        let side = (self.num_position_embeddings as f64).sqrt() as usize;
        if side * side != self.num_position_embeddings {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "qwen3_5 num_position_embeddings must be a square, got {}",
                self.num_position_embeddings
            )));
        }
        Ok(side)
    }
}

fn qwen3_5_smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> crate::Result<(usize, usize)> {
    let aspect = height.max(width) as f64 / height.min(width) as f64;
    if aspect > 200.0 {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "qwen3_5 image aspect ratio must be <= 200, got {aspect}"
        )));
    }

    let mut resized_height = ((height + factor / 2) / factor) * factor;
    let mut resized_width = ((width + factor / 2) / factor) * factor;

    if resized_height * resized_width > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        resized_height = factor.max((((height as f64 / beta) as usize) / factor) * factor);
        resized_width = factor.max((((width as f64 / beta) as usize) / factor) * factor);
    } else if resized_height * resized_width < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        resized_height = ((height as f64 * beta).ceil() as usize).div_ceil(factor) * factor;
        resized_width = ((width as f64 * beta).ceil() as usize).div_ceil(factor) * factor;
    }

    Ok((resized_height, resized_width))
}

pub fn prepare_qwen3_5_image_input(
    image: &image::DynamicImage,
    config_json: &serde_json::Value,
) -> crate::Result<Qwen3_5PreparedImageInput> {
    let config: Qwen3_5Config = serde_json::from_value(config_json.clone())?;
    let vision_config = config.vision_config;
    if vision_config.patch_size != 0 && vision_config.patch_size != QWEN3_5_IMAGE_PATCH_SIZE {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "qwen3_5 patch_size {} did not match expected {}",
            vision_config.patch_size, QWEN3_5_IMAGE_PATCH_SIZE
        )));
    }
    if vision_config.temporal_patch_size != 0
        && vision_config.temporal_patch_size != QWEN3_5_IMAGE_TEMPORAL_PATCH_SIZE
    {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "qwen3_5 temporal_patch_size {} did not match expected {}",
            vision_config.temporal_patch_size, QWEN3_5_IMAGE_TEMPORAL_PATCH_SIZE
        )));
    }
    if vision_config.spatial_merge_size != 0
        && vision_config.spatial_merge_size != QWEN3_5_IMAGE_MERGE_SIZE
    {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "qwen3_5 spatial_merge_size {} did not match expected {}",
            vision_config.spatial_merge_size, QWEN3_5_IMAGE_MERGE_SIZE
        )));
    }

    let (width, height) = (image.width() as usize, image.height() as usize);
    let factor = QWEN3_5_IMAGE_PATCH_SIZE * QWEN3_5_IMAGE_MERGE_SIZE;
    let (resized_height, resized_width) = qwen3_5_smart_resize(
        height,
        width,
        factor,
        QWEN3_5_IMAGE_MIN_PIXELS,
        QWEN3_5_IMAGE_MAX_PIXELS,
    )?;
    let resized = image.resize_exact(
        resized_width as u32,
        resized_height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let rgb = resized.to_rgb8().into_raw();
    let pixels: Vec<f32> = rgb
        .iter()
        .map(|&x| ((x as f32 / 255.0) - QWEN3_5_IMAGE_MEAN) / QWEN3_5_IMAGE_STD)
        .collect();

    let mut chw = vec![0.0; 3 * resized_height * resized_width];
    for row in 0..resized_height {
        for col in 0..resized_width {
            for chan in 0..3 {
                chw[chan * resized_height * resized_width + row * resized_width + col] =
                    pixels[(row * resized_width + col) * 3 + chan];
            }
        }
    }

    if resized_height % QWEN3_5_IMAGE_PATCH_SIZE != 0
        || resized_width % QWEN3_5_IMAGE_PATCH_SIZE != 0
    {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "qwen3_5 resized image {resized_height}x{resized_width} must be divisible by patch_size {}",
            QWEN3_5_IMAGE_PATCH_SIZE
        )));
    }

    let grid_h = resized_height / QWEN3_5_IMAGE_PATCH_SIZE;
    let grid_w = resized_width / QWEN3_5_IMAGE_PATCH_SIZE;
    if !grid_h.is_multiple_of(QWEN3_5_IMAGE_MERGE_SIZE)
        || !grid_w.is_multiple_of(QWEN3_5_IMAGE_MERGE_SIZE)
    {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "qwen3_5 grid {grid_h}x{grid_w} must be divisible by merge_size {}",
            QWEN3_5_IMAGE_MERGE_SIZE
        )));
    }

    Ok(Qwen3_5PreparedImageInput {
        pixels: chw,
        shape: vec![1, 3, resized_height, resized_width],
        runtime_vision: Qwen3_5RuntimeVisionConfig {
            resized_height,
            resized_width,
            grid_t: 1,
            grid_h,
            grid_w,
        },
    })
}

impl Qwen3_5MultimodalConfig {
    fn new(
        text_config: &Qwen3_5TextConfig,
        config: &Qwen3_5Config,
        runtime_vision: Option<&Qwen3_5RuntimeVisionConfig>,
    ) -> crate::Result<Option<Self>> {
        let Some(image_token_index) = config.image_token_id else {
            return Ok(None);
        };
        if config.vision_start_token_id.is_none() || config.vision_end_token_id.is_none() {
            return Ok(None);
        }

        let mut vision_config = config.vision_config.clone();
        if !vision_config.is_configured() {
            return Ok(None);
        }

        if vision_config.out_hidden_size == 0 {
            vision_config.out_hidden_size = text_config.hidden_size;
        }
        if vision_config.out_hidden_size != text_config.hidden_size {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "qwen3_5 vision out_hidden_size {} does not match text hidden_size {}",
                vision_config.out_hidden_size, text_config.hidden_size
            )));
        }

        let base_grid_size = vision_config.base_grid_size()?;
        let runtime_vision = runtime_vision.cloned().unwrap_or_default();
        let grid_t = runtime_vision.grid_t.max(1);
        let raw_grid_height = if runtime_vision.grid_h == 0 {
            base_grid_size
        } else {
            runtime_vision.grid_h
        };
        let raw_grid_width = if runtime_vision.grid_w == 0 {
            base_grid_size
        } else {
            runtime_vision.grid_w
        };
        if raw_grid_height % vision_config.spatial_merge_size != 0
            || raw_grid_width % vision_config.spatial_merge_size != 0
        {
            return Err(crate::LLMError::InvalidModelConfig(format!(
                "qwen3_5 raw grid {}x{} must be divisible by spatial_merge_size {}",
                raw_grid_height, raw_grid_width, vision_config.spatial_merge_size
            )));
        }

        let merged_grid_height = raw_grid_height / vision_config.spatial_merge_size;
        let merged_grid_width = raw_grid_width / vision_config.spatial_merge_size;
        let mm_tokens_per_image = grid_t * merged_grid_height * merged_grid_width;
        let image_height = if runtime_vision.resized_height == 0 {
            raw_grid_height * vision_config.patch_size
        } else {
            runtime_vision.resized_height
        };
        let image_width = if runtime_vision.resized_width == 0 {
            raw_grid_width * vision_config.patch_size
        } else {
            runtime_vision.resized_width
        };
        Ok(Some(Self {
            image_height,
            image_width,
            image_token_index,
            base_grid_size,
            grid_t,
            raw_grid_height,
            raw_grid_width,
            merged_grid_height,
            merged_grid_width,
            mm_tokens_per_image,
            rope_delta: merged_grid_height.max(merged_grid_width) as f32
                - mm_tokens_per_image as f32,
            vision_config,
        }))
    }
}

fn interpolate_qwen3_5_prompt_impl(mm: &Qwen3_5MultimodalConfig, prompt: &str) -> String {
    let image_span = format!(
        "<|vision_start|>{}<|vision_end|>",
        "<|image_pad|>".repeat(mm.mm_tokens_per_image)
    );
    if prompt.contains("<|vision_start|><|image_pad|><|vision_end|>") {
        prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", &image_span)
    } else if prompt.contains("<|vision_start|>") || prompt.contains("<|vision_end|>") {
        prompt.to_string()
    } else {
        let interpolated = prompt.replace("<image>", &image_span);
        if interpolated == prompt {
            format!("{prompt}{image_span}")
        } else {
            interpolated
        }
    }
}

pub fn interpolate_qwen3_5_prompt(
    config_json: &serde_json::Value,
    runtime_vision: Option<&Qwen3_5RuntimeVisionConfig>,
    prompt: &str,
) -> crate::Result<Option<String>> {
    let config: Qwen3_5Config = serde_json::from_value(config_json.clone())?;
    let multimodal = Qwen3_5MultimodalConfig::new(&config.text_config, &config, runtime_vision)?;
    Ok(multimodal
        .as_ref()
        .map(|mm| interpolate_qwen3_5_prompt_impl(mm, prompt)))
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Qwen3_5RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_mrope_interleaved")]
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct Qwen3_5TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_full_attention_interval")]
    full_attention_interval: usize,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_value_head_dim: usize,
    linear_num_value_heads: usize,
    #[serde(default)]
    rope_parameters: Qwen3_5RopeParameters,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
}

impl Qwen3_5TextConfig {
    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        if self.layer_types.len() == self.num_hidden_layers {
            self.layer_types[layer_id] == "full_attention"
        } else {
            (layer_id + 1).is_multiple_of(self.full_attention_interval)
        }
    }
}

fn default_full_attention_interval() -> usize {
    4
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

fn default_mrope_section() -> Vec<usize> {
    vec![11, 11, 10]
}

fn default_mrope_interleaved() -> bool {
    true
}

fn qwen3_5_interleaved_mrope_axis(idx: usize, sections: [usize; 3]) -> usize {
    let h_limit = sections[1] * 3;
    let w_limit = sections[2] * 3;
    if idx % 3 == 1 && idx < h_limit {
        1
    } else if idx % 3 == 2 && idx < w_limit {
        2
    } else {
        0
    }
}

fn nat_to_f32(builder: &Builder, n: impl IntoNatVar) -> Var {
    cast(builder, nat_to_u32(builder, n.to_nat(builder)), Dtype::F32)
}

fn position_range_from_f32(builder: &Builder, start: Var, len: Var) -> Var {
    let range = arange(builder, len.clone());
    let range = cast(builder, range, Dtype::F32);
    let range = reshape(builder, shape!(builder, 1, len), range);
    let start = broadcast(builder, shape!(builder, 1, len), start);
    start + range
}

fn concat_many(builder: &Builder, dim: u32, xs: Vec<Var>) -> Var {
    let mut iter = xs.into_iter();
    let mut out = iter
        .next()
        .expect("concat_many requires at least one tensor");
    for x in iter {
        out = concat(builder, dim, out, x);
    }
    out
}

fn reorder_tokens_block_major(
    builder: &Builder,
    raw_grid_height: usize,
    raw_grid_width: usize,
    merge: usize,
    x: Var,
) -> Var {
    let [b, _, dim] = unpack::<3>(builder, shape(builder, x.clone()));
    let merged_grid_height = raw_grid_height / merge;
    let merged_grid_width = raw_grid_width / merge;
    let x = reshape(
        builder,
        shape!(builder, b, raw_grid_height, raw_grid_width, dim),
        x,
    );
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            merged_grid_height,
            merge,
            merged_grid_width,
            merge,
            dim
        ),
        x,
    );
    let x = transpose(builder, 2, 3, x);
    reshape(
        builder,
        shape!(builder, b, raw_grid_height * raw_grid_width, dim),
        x,
    )
}

fn reorder_position_embeddings_block_major(
    builder: &Builder,
    raw_grid_height: usize,
    raw_grid_width: usize,
    merge: usize,
    x: Var,
) -> Var {
    let [_, dim] = unpack::<2>(builder, shape(builder, x.clone()));
    let merged_grid_height = raw_grid_height / merge;
    let merged_grid_width = raw_grid_width / merge;
    let x = reshape(
        builder,
        shape!(builder, raw_grid_height, raw_grid_width, dim),
        x,
    );
    let x = reshape(
        builder,
        shape!(
            builder,
            merged_grid_height,
            merge,
            merged_grid_width,
            merge,
            dim
        ),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    reshape(
        builder,
        shape!(builder, raw_grid_height * raw_grid_width, dim),
        x,
    )
}

fn reorder_axis_block_major(
    builder: &Builder,
    raw_grid_height: usize,
    raw_grid_width: usize,
    merge: usize,
    x: Var,
) -> Var {
    let merged_grid_height = raw_grid_height / merge;
    let merged_grid_width = raw_grid_width / merge;
    let x = reshape(
        builder,
        shape!(builder, 1, raw_grid_height, raw_grid_width),
        x,
    );
    let x = reshape(
        builder,
        shape!(
            builder,
            1,
            merged_grid_height,
            merge,
            merged_grid_width,
            merge
        ),
        x,
    );
    let x = transpose(builder, 2, 3, x);
    reshape(
        builder,
        shape!(builder, 1, raw_grid_height * raw_grid_width),
        x,
    )
}

fn rotate_half_rank4(builder: &Builder, head_dim: usize, x: Var) -> Var {
    let parts = chunk(builder, 3, 2, head_dim / 2, x);
    concat(builder, 3, -parts[1].clone(), parts[0].clone())
}

fn apply_rope_with_tables(builder: &Builder, cos: Var, sin: Var, head_dim: usize, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let cos = broadcast(builder, sh.clone(), cos);
    let sin = broadcast(builder, sh, sin);
    let rotated = rotate_half_rank4(builder, head_dim, x.clone());
    cos * x + sin * rotated
}

impl LLMConfig for Qwen3_5TextConfig {
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
        self.rope_parameters.rope_theta
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters.partial_rotary_factor
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5Model {
    config: Qwen3_5TextConfig,
    layer_to_cache_id: Vec<Option<usize>>,
    layer_to_linear_id: Vec<Option<usize>>,
    num_linear_layers: usize,
    dtype: Dtype,
    pub max_sequence_length: usize,
    multimodal: Option<Qwen3_5MultimodalConfig>,
}

impl LLMModel for Qwen3_5Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn extra_nat_input(&self, seq_len: usize) -> Option<usize> {
        Some(seq_len.div_ceil(GATED_DELTA_CHUNK_SIZE))
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let dtype = self.dtype();
        vec![
            (
                dtype,
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_qk_head_dim(),
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_v_head_dim(),
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.linear_conv_dim(),
                    self.config.linear_conv_kernel_dim,
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.config.linear_num_value_heads,
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                ]),
            ),
            (dtype, Shape(vec![1])),
        ]
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
        Some(Box::new(Qwen3_5VisionModel {
            vision_config: mm.vision_config.clone(),
            base_grid_size: mm.base_grid_size,
            raw_grid_height: mm.raw_grid_height,
            raw_grid_width: mm.raw_grid_width,
            merged_grid_height: mm.merged_grid_height,
            merged_grid_width: mm.merged_grid_width,
        }))
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        self.multimodal.as_ref()?;
        Some(Box::new(Qwen3_5MultimodalModel {
            language_model: self.clone(),
        }))
    }

    fn multimodal_interpolate_prompt(&self, prompt: &str) -> Option<String> {
        let mm = self.multimodal.as_ref()?;
        Some(interpolate_qwen3_5_prompt_impl(mm, prompt))
    }
}

impl Qwen3_5Model {
    pub fn new(
        config_json: &serde_json::Value,
        max_sequence_length: usize,
        runtime_vision: Option<&Qwen3_5RuntimeVisionConfig>,
        dtype: Dtype,
    ) -> crate::Result<Self> {
        let config: Qwen3_5Config = serde_json::from_value(config_json.clone())?;
        let multimodal =
            Qwen3_5MultimodalConfig::new(&config.text_config, &config, runtime_vision)?;
        let config = config.text_config;
        assert!(
            config.linear_conv_kernel_dim > 0,
            "qwen3_5 linear_conv_kernel_dim must be > 0"
        );
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
            config,
            layer_to_cache_id,
            layer_to_linear_id,
            num_linear_layers: next_linear_id,
            dtype,
            max_sequence_length,
            multimodal,
        })
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.config.is_full_attention_layer(layer_id)
    }

    fn linear_conv_dim(&self) -> usize {
        let key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim;
        let value_dim = self.config.linear_num_value_heads * self.config.linear_value_head_dim;
        key_dim * 2 + value_dim
    }

    fn rotary_dim(&self, head_dim: usize) -> usize {
        let mut rotary_dim = (head_dim as f32 * self.config.partial_rotary_factor()) as usize;
        let mrope_half_dim = self
            .config
            .rope_parameters
            .mrope_section
            .iter()
            .copied()
            .sum::<usize>();
        if mrope_half_dim > 0 {
            rotary_dim = mrope_half_dim * 2;
        }
        if rotary_dim == 0 || rotary_dim > head_dim {
            rotary_dim = head_dim;
        }
        rotary_dim -= rotary_dim % 2;
        if rotary_dim == 0 {
            head_dim
        } else {
            rotary_dim
        }
    }

    fn apply_rope_partial(&self, builder: &Builder, pos: Var, cache: &Cache, x: Var) -> Var {
        let head_dim = self.config.get_head_dim();
        let rotary_dim = self.rotary_dim(head_dim);

        // Full Qwen3.5 mRoPE interleaving uses 3D position IDs (T/H/W). In this text-only path,
        // all axes collapse to the same scalar position so partial RoPE is equivalent here.
        apply_rope_embedding_partial(
            builder,
            pos,
            rotary_dim,
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            x,
        )
    }

    fn apply_rope_partial_with_tables(&self, builder: &Builder, cos: Var, sin: Var, x: Var) -> Var {
        let head_dim = self.config.get_head_dim();
        let rotary_dim = self.rotary_dim(head_dim);
        if rotary_dim >= head_dim {
            return apply_rope_with_tables(builder, cos, sin, head_dim, x);
        }

        let split = split(builder, 3, &[rotary_dim, head_dim - rotary_dim], x);
        let x_rot = apply_rope_with_tables(builder, cos, sin, rotary_dim, split[0].clone());
        concat(builder, 3, x_rot, split[1].clone())
    }

    fn multimodal_delta_tensor(&self, builder: &Builder) -> Var {
        let mm = self
            .multimodal
            .as_ref()
            .expect("qwen3_5 multimodal delta requires multimodal config");
        constant(builder, mm.rope_delta, &shape!(builder, 1))
    }

    fn text_axes_from_start(&self, builder: &Builder, start: Var, len: Var) -> [Var; 3] {
        let pos = position_range_from_f32(builder, start, len);
        [pos.clone(), pos.clone(), pos]
    }

    fn image_axes_from_start(&self, builder: &Builder, start: Var) -> [Var; 3] {
        let mm = self
            .multimodal
            .as_ref()
            .expect("qwen3_5 image axes require multimodal config");
        let merged_height = mm.merged_grid_height;
        let merged_width = mm.merged_grid_width;
        let tokens = mm.grid_t * merged_height * merged_width;

        let row = arange(builder, merged_height);
        let row = cast(builder, row, Dtype::F32);
        let row = reshape(builder, shape!(builder, merged_height, 1), row);
        let row = broadcast(builder, shape!(builder, merged_height, merged_width), row);
        let row = reshape(builder, shape!(builder, 1, tokens), row);

        let col = arange(builder, merged_width);
        let col = cast(builder, col, Dtype::F32);
        let col = reshape(builder, shape!(builder, 1, merged_width), col);
        let col = broadcast(builder, shape!(builder, merged_height, merged_width), col);
        let col = reshape(builder, shape!(builder, 1, tokens), col);

        let start = broadcast(builder, shape!(builder, 1, tokens), start);
        let t = start.clone();
        let h = start.clone() + row;
        let w = start + col;
        [t, h, w]
    }

    fn multimodal_rope_tables(&self, builder: &Builder, axes: [Var; 3]) -> (Var, Var) {
        let rotary_dim = self.rotary_dim(self.config.get_head_dim());
        let half_dim = rotary_dim / 2;

        let inv_idx = arange(builder, half_dim);
        let inv_idx = cast(builder, inv_idx, Dtype::F32);
        let sh = shape(builder, inv_idx.clone());
        let scale = constant(builder, 2.0 / (rotary_dim as f32), &sh);
        let theta = constant(builder, self.config.rope_theta(), &sh);
        let inv_freq = inverse(builder, pow(builder, theta, inv_idx * scale));

        let axis_freqs = axes.map(|axis| {
            let [_, seq_len] = unpack::<2>(builder, shape(builder, axis.clone()));
            let axis = reshape(builder, shape!(builder, 1, seq_len, 1), axis);
            let sh = shape!(builder, 1, seq_len, half_dim);
            let axis = broadcast(builder, sh.clone(), axis);
            let inv_freq = broadcast(builder, sh, inv_freq.clone());
            axis * inv_freq
        });

        let first = self
            .config
            .rope_parameters
            .mrope_section
            .first()
            .copied()
            .unwrap_or(half_dim)
            .min(half_dim);
        let second = self
            .config
            .rope_parameters
            .mrope_section
            .get(1)
            .copied()
            .unwrap_or(0)
            .min(half_dim - first);
        let third = self
            .config
            .rope_parameters
            .mrope_section
            .get(2)
            .copied()
            .unwrap_or(0)
            .min(half_dim - first - second);
        let sections = [first, second, third];
        let remainder = half_dim - sections.into_iter().sum::<usize>();

        let interleaved = if self.config.rope_parameters.mrope_interleaved {
            let mut cols = Vec::with_capacity(half_dim);
            for idx in 0..half_dim {
                let axis = qwen3_5_interleaved_mrope_axis(idx, sections);
                cols.push(slice(builder, 2, idx, 1, axis_freqs[axis].clone()));
            }
            concat_many(builder, 2, cols)
        } else {
            let mut parts = Vec::new();
            if sections[0] > 0 {
                parts.push(slice(builder, 2, 0, sections[0], axis_freqs[0].clone()));
            }
            if sections[1] > 0 {
                parts.push(slice(builder, 2, 0, sections[1], axis_freqs[1].clone()));
            }
            if sections[2] > 0 {
                parts.push(slice(builder, 2, 0, sections[2], axis_freqs[2].clone()));
            }
            if remainder > 0 {
                parts.push(slice(
                    builder,
                    2,
                    sections[0],
                    remainder,
                    axis_freqs[0].clone(),
                ));
            }
            concat_many(builder, 2, parts)
        };

        let emb = concat(builder, 2, interleaved.clone(), interleaved);
        (cos(builder, emb.clone()), sin(builder, emb))
    }

    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let gate = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            p.extend(["gate_proj"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            p.extend(["up_proj"]).unwrap(),
            x,
        );
        let x = silu(builder, gate) * up;
        linear_no_bias(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    fn full_attention(
        &self,
        builder: &Builder,
        cache_layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        custom_rope: Option<(Var, Var)>,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.get_head_dim();

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim * 2,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let q = reshape(builder, shape!(builder, b, s, num_heads, head_dim * 2), q);
        let q = chunk(builder, 3, 2, head_dim, q);
        let sh = shape!(builder, b, s, num_heads * head_dim);
        let q_states = q[0].clone();
        let q_gate = q[1].clone();
        let q_states = reshape(builder, sh.clone(), q_states);
        let q_gate = reshape(builder, sh, q_gate);

        let k = linear_no_bias(
            builder,
            dim,
            num_kv_heads * head_dim,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );

        let v = linear_no_bias(
            builder,
            dim,
            num_kv_heads * head_dim,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q_states);
        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = rmsnorm_gemma::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm_gemma::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = if let Some((cos, sin)) = custom_rope.clone() {
            self.apply_rope_partial_with_tables(builder, cos, sin, q)
        } else {
            self.apply_rope_partial(builder, pos.clone(), cache, q)
        };
        let k = if let Some((cos, sin)) = custom_rope {
            self.apply_rope_partial_with_tables(builder, cos, sin, k)
        } else {
            self.apply_rope_partial(builder, pos, cache, k)
        };

        let (k, v) = cache.update_kv_cache(builder, cache_layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, num_heads * head_dim);
        let mut attn = reshape(builder, sh, attn);

        attn = attn * sigmoid(builder, q_gate);

        linear_no_bias(
            builder,
            num_heads * head_dim,
            dim,
            p.extend(["o_proj"]).unwrap(),
            attn,
        )
    }

    fn linear_attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        num_chunks: Var,
        pos: Var,
        p: Path,
        hidden_states: Var,
    ) -> Var {
        let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, hidden_states.clone()));

        let num_k_heads = self.config.linear_num_key_heads;
        let num_v_heads = self.config.linear_num_value_heads;
        let head_k_dim = self.config.linear_key_head_dim;
        let head_v_dim = self.config.linear_value_head_dim;
        let max_num_chunks = self.max_sequence_length.div_ceil(GATED_DELTA_CHUNK_SIZE);
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let cache_len = self.config.linear_conv_kernel_dim;
        let linear_layer_id =
            self.layer_to_linear_id[layer_id].expect("linear-attention layer missing state index");
        let conv_state = cache
            .linear_cache
            .as_ref()
            .expect("qwen3_5 linear attention requires linear cache")[linear_layer_id][0]
            .clone();
        let recurrent_state = cache
            .linear_cache
            .as_ref()
            .expect("qwen3_5 linear attention requires linear cache")[linear_layer_id][1]
            .clone();

        let mixed_qkv = linear_no_bias(
            builder,
            self.config.hidden_size,
            conv_dim,
            p.extend(["in_proj_qkv"]).unwrap(),
            hidden_states.clone(),
        );
        let mixed_qkv = transpose(builder, 1, 2, mixed_qkv);

        let pos_u32 = nat_to_u32(builder, pos);
        let zero_pos = constant(builder, 0u32, &shape(builder, pos_u32.clone()));
        let is_decode = gt(builder, pos_u32, zero_pos);

        let results = cond(
            builder,
            is_decode.clone(),
            |b, args: Vec<Var>| {
                let [mixed_qkv, conv_state, _batch_size, seq_len] = args.try_into().unwrap();
                let hidden_states_new = concat(b, 2, conv_state, mixed_qkv);
                let out_conv_state_decode =
                    slice(b, 2, seq_len.clone(), cache_len, hidden_states_new);
                let mixed_qkv_decode = silu(
                    b,
                    padded_depthwise_conv1d_no_bias(
                        b,
                        p.extend(["conv1d"]).unwrap(),
                        cache_len,
                        out_conv_state_decode.clone(),
                        seq_len,
                    ),
                );
                vec![mixed_qkv_decode, out_conv_state_decode]
            },
            |b, args: Vec<Var>| {
                let [mixed_qkv, _conv_state, batch_size, seq_len] = args.try_into().unwrap();
                let mixed_qkv_prefill = silu(
                    b,
                    depthwise_conv1d_no_bias(
                        b,
                        p.extend(["conv1d"]).unwrap(),
                        cache_len,
                        mixed_qkv.clone(),
                        cache_len - 1,
                    ),
                );

                let zeros_prefill_state = zeros(b, &shape!(b, batch_size, conv_dim, cache_len));
                let x_padded_prefill_state = concat(b, 2, zeros_prefill_state, mixed_qkv);
                let out_conv_state_prefill =
                    slice(b, 2, seq_len, cache_len, x_padded_prefill_state);
                vec![mixed_qkv_prefill, out_conv_state_prefill]
            },
            vec![mixed_qkv, conv_state, batch_size.clone(), seq_len.clone()],
        );

        let mixed_qkv = results[0].clone();
        let out_conv_state = results[1].clone();

        cache
            .linear_cache
            .as_mut()
            .expect("qwen3_5 linear attention requires mutable linear cache")[linear_layer_id][0] =
            out_conv_state;

        let mixed_qkv = transpose(builder, 1, 2, mixed_qkv);

        let qkv = split(builder, 2, &[key_dim, key_dim, value_dim], mixed_qkv);
        let query = qkv[0].clone();
        let key = qkv[1].clone();
        let value = qkv[2].clone();

        let sh = shape!(builder, batch_size, seq_len, num_k_heads, head_k_dim);
        let mut query = reshape(builder, sh.clone(), query);
        let mut key = reshape(builder, sh, key);
        let sh = shape!(builder, batch_size, seq_len, num_v_heads, head_v_dim);
        let value = reshape(builder, sh.clone(), value);

        let z = linear_no_bias(
            builder,
            self.config.hidden_size,
            value_dim,
            p.extend(["in_proj_z"]).unwrap(),
            hidden_states.clone(),
        );
        let z = reshape(builder, sh, z);

        let b = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_v_heads,
            p.extend(["in_proj_b"]).unwrap(),
            hidden_states.clone(),
        );
        let a = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_v_heads,
            p.extend(["in_proj_a"]).unwrap(),
            hidden_states,
        );

        let beta = sigmoid(builder, b);
        let dt_bias = param(builder, &p.extend(["dt_bias"]).unwrap());
        let dt_bias = unsqueeze::<1, 2>(builder, 0, dt_bias);
        let dt_bias = broadcast(builder, shape(builder, a.clone()), dt_bias);
        let a = a + dt_bias;

        let a_log = param(builder, &p.extend(["A_log"]).unwrap());
        let a_log = unsqueeze::<1, 2>(builder, 0, a_log);
        let a_log = broadcast(builder, shape(builder, a.clone()), a_log);
        let g = -exp(builder, a_log) * softplus(builder, a);

        let rep = num_v_heads / num_k_heads;
        if rep > 1 {
            query = repeat_interleave(builder, 2, rep, query);
            key = repeat_interleave(builder, 2, rep, key);
        }

        let results = cond(
            builder,
            is_decode,
            |b, args: Vec<Var>| {
                let [query, key, value, g, beta, recurrent_state, _num_chunks] =
                    args.try_into().unwrap();
                let (core_attn_out_decode, out_recurrent_state_decode) = recurrent_gated_delta_rule(
                    b,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                    head_k_dim,
                );
                vec![core_attn_out_decode, out_recurrent_state_decode]
            },
            |b, args: Vec<Var>| {
                let [query, key, value, g, beta, _recurrent_state, num_chunks] =
                    args.try_into().unwrap();
                let (core_attn_out_prefill, out_recurrent_state_prefill) = chunk_gated_delta_rule(
                    b,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    head_k_dim,
                    num_chunks,
                    max_num_chunks,
                );
                vec![core_attn_out_prefill, out_recurrent_state_prefill]
            },
            vec![query, key, value, g, beta, recurrent_state, num_chunks],
        );

        let core_attn_out = results[0].clone();
        let out_recurrent_state = results[1].clone();

        cache
            .linear_cache
            .as_mut()
            .expect("qwen3_5 linear attention requires mutable linear cache")[linear_layer_id][1] =
            out_recurrent_state;

        let flat = shape!(
            builder,
            batch_size.clone() * seq_len.clone() * num_v_heads.to_nat(builder),
            head_v_dim
        );
        let core_attn_out = reshape(builder, flat.clone(), core_attn_out);
        let z = reshape(builder, flat, z);

        // Gated RMSNorm
        let core_attn_out = rmsnorm::<2>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm"]).unwrap(),
            core_attn_out,
        ) * silu(builder, z);

        let sh = shape!(builder, batch_size, seq_len, value_dim);
        let core_attn_out = reshape(builder, sh, core_attn_out);

        linear_no_bias(
            builder,
            value_dim,
            self.config.hidden_size,
            p.extend(["out_proj"]).unwrap(),
            core_attn_out,
        )
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        num_chunks: Var,
        pos: Var,
        custom_rope: Option<(Var, Var)>,
        p: Path,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = if self.is_full_attention_layer(layer_id) {
            let cache_layer_id = self.layer_to_cache_id[layer_id]
                .expect("full-attention layer missing KV cache index");
            self.full_attention(
                builder,
                cache_layer_id,
                attention_mask,
                cache,
                pos,
                custom_rope,
                p.extend(["self_attn"]).unwrap(),
                x,
            )
        } else {
            self.linear_attention(
                builder,
                layer_id,
                cache,
                num_chunks,
                pos,
                p.extend(["linear_attn"]).unwrap(),
                x,
            )
        };

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        x + res
    }

    fn init_cache(
        &self,
        builder: &Builder,
        in_k: Var,
        in_v: Var,
        in_conv: Var,
        in_recurrent: Var,
        max_positions: Var,
    ) -> Cache {
        let mut cache = Cache::init(
            builder,
            &self.config,
            max_positions.clone(),
            max_positions,
            in_k,
            in_v,
        );
        cache.linear_cache = Some(
            (0..self.num_linear_layers)
                .map(|layer_id| {
                    let conv_layer = slice(builder, 0, layer_id, 1, in_conv.clone());
                    let conv_state = squeeze::<4, 3>(builder, 0, conv_layer);
                    let recurrent_layer = slice(builder, 0, layer_id, 1, in_recurrent.clone());
                    let recurrent_state = squeeze::<5, 4>(builder, 0, recurrent_layer);
                    vec![conv_state, recurrent_state]
                })
                .collect(),
        );
        cache
    }

    fn collect_linear_states(
        &self,
        builder: &Builder,
        cache: &Cache,
        in_conv: Var,
        in_recurrent: Var,
    ) -> (Var, Var) {
        let out_conv = if self.num_linear_layers == 0 {
            in_conv
        } else {
            let states = cache
                .linear_cache
                .as_ref()
                .expect("qwen3_5 cache missing linear cache");
            let mut iter = states.iter();
            let first = iter.next().expect("qwen3_5 conv state missing")[0].clone();
            let mut out = unsqueeze::<3, 4>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<3, 4>(builder, 0, state[0].clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        let out_recurrent = if self.num_linear_layers == 0 {
            in_recurrent
        } else {
            let states = cache
                .linear_cache
                .as_ref()
                .expect("qwen3_5 cache missing linear cache");
            let mut iter = states.iter();
            let first = iter.next().expect("qwen3_5 recurrent state missing")[1].clone();
            let mut out = unsqueeze::<4, 5>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<4, 5>(builder, 0, state[1].clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        (out_conv, out_recurrent)
    }

    fn forward_with_embeddings(
        &self,
        builder: &Builder,
        root: Path,
        embeddings: Var,
        in_k: Var,
        in_v: Var,
        in_conv: Var,
        in_recurrent: Var,
        max_positions: Var,
        num_chunks: Var,
        custom_rope: Option<(Var, Var)>,
    ) -> Vec<Var> {
        let mut cache = self.init_cache(
            builder,
            in_k.clone(),
            in_v,
            in_conv.clone(),
            in_recurrent.clone(),
            max_positions,
        );
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k));
        let language_root = root.extend(["model", "language_model"]).unwrap();

        let [_b, s, _] = unpack::<3>(builder, shape(builder, embeddings.clone()));
        let attention_mask = causal_mask(builder, s, pos.clone());

        let mut x = embeddings;
        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                num_chunks.clone(),
                pos.clone(),
                custom_rope.clone(),
                language_root.extend(["layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            language_root.extend(["norm"]).unwrap(),
            x,
        );

        let lm_head = if self.config.tie_word_embeddings {
            root.extend(["model", "language_model", "embed_tokens"])
                .unwrap()
        } else {
            root.extend(["lm_head"]).unwrap()
        };
        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            lm_head,
            x,
        );
        x = argmax(builder, x);

        let (out_k, out_v) = cache.get_kv_cache(builder);
        let (out_conv, out_recurrent) =
            self.collect_linear_states(builder, &cache, in_conv, in_recurrent);
        vec![x, out_k, out_v, out_conv, out_recurrent]
    }
}

impl DynModule for Qwen3_5Model {
    fn path(&self) -> Path {
        path(vec!["qwen3_5"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [
            x,
            in_k,
            in_v,
            in_conv,
            in_recurrent,
            in_mm_delta,
            max_positions,
            num_chunks,
        ]: [Var; 8] = args.try_into().expect("expected 8 inputs");
        let token_embeddings = embeddings(
            builder,
            self.path()
                .extend(["model", "language_model", "embed_tokens"])
                .unwrap(),
            x,
        );
        let mut outputs = self.forward_with_embeddings(
            builder,
            self.path(),
            token_embeddings,
            in_k,
            in_v,
            in_conv,
            in_recurrent,
            max_positions,
            num_chunks,
            None,
        );
        outputs.push(in_mm_delta);
        outputs
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let (mut source, mut target) = llm_type(&self.config, self.dtype());
        let max_positions = source
            .pop()
            .expect("qwen3_5 missing max_positions nat input");
        let batch_size = NatExpr::Var(0);
        let num_linear_layers = NatExpr::Constant(self.num_linear_layers);
        let conv_dim = NatExpr::Constant(self.linear_conv_dim());
        let conv_cache_len = NatExpr::Constant(self.config.linear_conv_kernel_dim);
        let num_v_heads = NatExpr::Constant(self.config.linear_num_value_heads);
        let head_k_dim = NatExpr::Constant(self.config.linear_key_head_dim);
        let head_v_dim = NatExpr::Constant(self.config.linear_value_head_dim);

        let t_conv = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers.clone(),
                batch_size.clone(),
                conv_dim,
                conv_cache_len,
            ]),
        }));
        let t_recurrent = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers,
                batch_size,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            ]),
        }));
        let t_mm_delta = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![NatExpr::Constant(1)]),
        }));

        source.push(t_conv.clone());
        source.push(t_recurrent.clone());
        source.push(t_mm_delta.clone());
        source.push(max_positions);
        source.push(Type::Nat(NatExpr::Var(4)));
        target.push(t_conv);
        target.push(t_recurrent);
        target.push(t_mm_delta);
        (source, target)
    }
}

fn qwen_vision_rotary_tables(
    builder: &Builder,
    raw_grid_height: usize,
    raw_grid_width: usize,
    merge: usize,
    head_dim: usize,
) -> (Var, Var) {
    let tokens = raw_grid_height * raw_grid_width;
    let row = arange(builder, raw_grid_height);
    let row = cast(builder, row, Dtype::F32);
    let row = reshape(builder, shape!(builder, raw_grid_height, 1), row);
    let row = broadcast(
        builder,
        shape!(builder, raw_grid_height, raw_grid_width),
        row,
    );
    let row = reshape(builder, shape!(builder, 1, tokens), row);
    let row = reorder_axis_block_major(builder, raw_grid_height, raw_grid_width, merge, row);

    let col = arange(builder, raw_grid_width);
    let col = cast(builder, col, Dtype::F32);
    let col = reshape(builder, shape!(builder, 1, raw_grid_width), col);
    let col = broadcast(
        builder,
        shape!(builder, raw_grid_height, raw_grid_width),
        col,
    );
    let col = reshape(builder, shape!(builder, 1, tokens), col);
    let col = reorder_axis_block_major(builder, raw_grid_height, raw_grid_width, merge, col);

    let rotary_input_dim = head_dim / 2;
    let axis_dim = rotary_input_dim / 2;
    let idx = arange(builder, axis_dim);
    let idx = cast(builder, idx, Dtype::F32);
    let sh = shape(builder, idx.clone());
    let scale = constant(builder, 2.0 / (rotary_input_dim as f32), &sh);
    let theta = constant(builder, 10_000.0, &sh);
    let inv_freq = inverse(builder, pow(builder, theta, idx * scale));

    let row = reshape(builder, shape!(builder, 1, tokens, 1), row);
    let col = reshape(builder, shape!(builder, 1, tokens, 1), col);
    let sh = shape!(builder, 1, tokens, axis_dim);
    let row = broadcast(builder, sh.clone(), row);
    let col = broadcast(builder, sh.clone(), col);
    let inv_freq = broadcast(builder, sh, inv_freq);
    let row = row * inv_freq.clone();
    let col = col * inv_freq;
    let emb = concat(builder, 2, row, col);
    let emb = concat(builder, 2, emb.clone(), emb);
    (cos(builder, emb.clone()), sin(builder, emb))
}

fn qwen_vision_interp_axis(builder: &Builder, target_len: usize, base_len: usize) -> Var {
    let axis = cast(builder, arange(builder, target_len), Dtype::F32);
    if target_len <= 1 || base_len <= 1 {
        let zero = constant(builder, 0.0, &shape(builder, axis.clone()));
        axis * zero
    } else {
        let scale = constant(
            builder,
            (base_len - 1) as f32 / (target_len - 1) as f32,
            &shape(builder, axis.clone()),
        );
        axis * scale
    }
}

fn qwen_vision_interpolate_pos_embed(
    builder: &Builder,
    base_grid_size: usize,
    target_grid_height: usize,
    target_grid_width: usize,
    pos: Var,
) -> Var {
    let [_, dim] = unpack::<2>(builder, shape(builder, pos.clone()));
    if target_grid_height == base_grid_size && target_grid_width == base_grid_size {
        return pos;
    }

    let row = qwen_vision_interp_axis(builder, target_grid_height, base_grid_size);
    let col = qwen_vision_interp_axis(builder, target_grid_width, base_grid_size);
    let row_floor = floor(builder, row.clone());
    let col_floor = floor(builder, col.clone());
    let row_plus_one = constant(builder, 1.0, &shape(builder, row_floor.clone()));
    let col_plus_one = constant(builder, 1.0, &shape(builder, col_floor.clone()));
    let row_ceil = clamp(
        builder,
        row_floor.clone() + row_plus_one,
        0.0,
        (base_grid_size - 1) as f32,
    );
    let col_ceil = clamp(
        builder,
        col_floor.clone() + col_plus_one,
        0.0,
        (base_grid_size - 1) as f32,
    );
    let row_frac = row - row_floor.clone();
    let col_frac = col - col_floor.clone();

    let row_floor_u32 = cast(builder, row_floor, Dtype::U32);
    let col_floor_u32 = cast(builder, col_floor, Dtype::U32);
    let row_ceil_u32 = cast(builder, row_ceil, Dtype::U32);
    let col_ceil_u32 = cast(builder, col_ceil, Dtype::U32);

    let source_width = constant(
        builder,
        base_grid_size as u32,
        &shape(builder, row_floor_u32.clone()),
    );
    let row_floor_base = row_floor_u32 * source_width.clone();
    let row_ceil_base = row_ceil_u32 * source_width;

    let row_floor_base = reshape(
        builder,
        shape!(builder, target_grid_height, 1),
        row_floor_base,
    );
    let row_ceil_base = reshape(
        builder,
        shape!(builder, target_grid_height, 1),
        row_ceil_base,
    );
    let col_floor_u32 = reshape(
        builder,
        shape!(builder, 1, target_grid_width),
        col_floor_u32,
    );
    let col_ceil_u32 = reshape(builder, shape!(builder, 1, target_grid_width), col_ceil_u32);

    let idx_shape = shape!(builder, target_grid_height, target_grid_width);
    let idx00 = broadcast(builder, idx_shape.clone(), row_floor_base.clone())
        + broadcast(builder, idx_shape.clone(), col_floor_u32.clone());
    let idx01 = broadcast(builder, idx_shape.clone(), row_floor_base)
        + broadcast(builder, idx_shape.clone(), col_ceil_u32.clone());
    let idx10 = broadcast(builder, idx_shape.clone(), row_ceil_base.clone())
        + broadcast(builder, idx_shape.clone(), col_floor_u32);
    let idx11 = broadcast(builder, idx_shape.clone(), row_ceil_base)
        + broadcast(builder, idx_shape.clone(), col_ceil_u32);

    let target_tokens = target_grid_height * target_grid_width;
    let idx00 = reshape(builder, shape!(builder, target_tokens), idx00);
    let idx01 = reshape(builder, shape!(builder, target_tokens), idx01);
    let idx10 = reshape(builder, shape!(builder, target_tokens), idx10);
    let idx11 = reshape(builder, shape!(builder, target_tokens), idx11);

    let one_row = constant(builder, 1.0, &shape(builder, row_frac.clone()));
    let one_col = constant(builder, 1.0, &shape(builder, col_frac.clone()));
    let row_floor_weight = one_row - row_frac.clone();
    let row_ceil_weight = row_frac;
    let col_floor_weight = one_col - col_frac.clone();
    let col_ceil_weight = col_frac;

    let row_floor_weight = reshape(
        builder,
        shape!(builder, target_grid_height, 1),
        row_floor_weight,
    );
    let row_ceil_weight = reshape(
        builder,
        shape!(builder, target_grid_height, 1),
        row_ceil_weight,
    );
    let col_floor_weight = reshape(
        builder,
        shape!(builder, 1, target_grid_width),
        col_floor_weight,
    );
    let col_ceil_weight = reshape(
        builder,
        shape!(builder, 1, target_grid_width),
        col_ceil_weight,
    );

    let w00 = broadcast(builder, idx_shape.clone(), row_floor_weight.clone())
        * broadcast(builder, idx_shape.clone(), col_floor_weight.clone());
    let w01 = broadcast(builder, idx_shape.clone(), row_floor_weight)
        * broadcast(builder, idx_shape.clone(), col_ceil_weight.clone());
    let w10 = broadcast(builder, idx_shape.clone(), row_ceil_weight.clone())
        * broadcast(builder, idx_shape.clone(), col_floor_weight);
    let w11 = broadcast(builder, idx_shape, row_ceil_weight)
        * broadcast(
            builder,
            shape!(builder, target_grid_height, target_grid_width),
            col_ceil_weight,
        );

    let weight_shape = shape!(builder, target_tokens, 1);
    let w00 = reshape(
        builder,
        weight_shape.clone(),
        reshape(builder, shape!(builder, target_tokens), w00),
    );
    let w01 = reshape(
        builder,
        weight_shape.clone(),
        reshape(builder, shape!(builder, target_tokens), w01),
    );
    let w10 = reshape(
        builder,
        weight_shape.clone(),
        reshape(builder, shape!(builder, target_tokens), w10),
    );
    let w11 = reshape(
        builder,
        weight_shape,
        reshape(builder, shape!(builder, target_tokens), w11),
    );

    let w00 = broadcast(builder, shape!(builder, target_tokens, dim), w00);
    let w01 = broadcast(builder, shape!(builder, target_tokens, dim), w01);
    let w10 = broadcast(builder, shape!(builder, target_tokens, dim), w10);
    let w11 = broadcast(builder, shape!(builder, target_tokens, dim), w11);

    index(builder, 0, idx00, pos.clone()) * w00
        + index(builder, 0, idx01, pos.clone()) * w01
        + index(builder, 0, idx10, pos.clone()) * w10
        + index(builder, 0, idx11, pos) * w11
}

fn qwen_vision_embeddings(
    builder: &Builder,
    config: &Qwen3_5VisionConfig,
    base_grid_size: usize,
    raw_grid_height: usize,
    raw_grid_width: usize,
    p: Path,
    x: Var,
) -> Var {
    let [b, _, _, _] = unpack::<4>(builder, shape(builder, x.clone()));
    let x = unsqueeze::<4, 5>(builder, 2, x);
    let x = broadcast(
        builder,
        shape!(
            builder,
            b,
            config.in_channels,
            config.temporal_patch_size,
            raw_grid_height * config.patch_size,
            raw_grid_width * config.patch_size
        ),
        x,
    );
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            config.in_channels,
            config.temporal_patch_size,
            raw_grid_height,
            config.patch_size,
            raw_grid_width,
            config.patch_size
        ),
        x,
    );
    let x = transpose(builder, 1, 3, x);
    let x = transpose(builder, 2, 5, x);
    let x = transpose(builder, 4, 5, x);
    let patch_dim =
        config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
    let x = reshape(
        builder,
        shape!(builder, b, raw_grid_height * raw_grid_width, patch_dim),
        x,
    );
    let x = reorder_tokens_block_major(
        builder,
        raw_grid_height,
        raw_grid_width,
        config.spatial_merge_size,
        x,
    );

    let weight = param(
        builder,
        &p.extend(["patch_embed", "proj", "weight"]).unwrap(),
    );
    let weight = reshape(
        builder,
        shape!(builder, config.hidden_size, patch_dim),
        weight,
    );
    let weight = transpose(builder, 0, 1, weight);
    let weight = broadcast(
        builder,
        shape!(builder, b, patch_dim, config.hidden_size),
        weight,
    );
    let x = matmul(builder, x, weight);

    let bias = param(builder, &p.extend(["patch_embed", "proj", "bias"]).unwrap());
    let x = x.clone() + broadcast(builder, shape(builder, x), bias);

    let pos = param(builder, &p.extend(["pos_embed", "weight"]).unwrap());
    let pos = qwen_vision_interpolate_pos_embed(
        builder,
        base_grid_size,
        raw_grid_height,
        raw_grid_width,
        pos,
    );
    let pos = reorder_position_embeddings_block_major(
        builder,
        raw_grid_height,
        raw_grid_width,
        config.spatial_merge_size,
        pos,
    );
    x.clone() + broadcast(builder, shape(builder, x), pos)
}

fn qwen_vision_attention(
    builder: &Builder,
    config: &Qwen3_5VisionConfig,
    cos: Var,
    sin: Var,
    p: Path,
    x: Var,
) -> Var {
    let dim = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = dim / num_heads;
    let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

    let qkv = linear(builder, dim, dim * 3, p.extend(["qkv"]).unwrap(), x);
    let qkv = reshape(builder, shape!(builder, b, s, 3, num_heads, head_dim), qkv);
    let q = squeeze::<5, 4>(builder, 2, slice(builder, 2, 0, 1, qkv.clone()));
    let k = squeeze::<5, 4>(builder, 2, slice(builder, 2, 1, 1, qkv.clone()));
    let v = squeeze::<5, 4>(builder, 2, slice(builder, 2, 2, 1, qkv));

    let q = transpose(builder, 1, 2, q);
    let k = transpose(builder, 1, 2, k);
    let v = transpose(builder, 1, 2, v);
    let q = apply_rope_with_tables(builder, cos.clone(), sin.clone(), head_dim, q);
    let k = apply_rope_with_tables(builder, cos, sin, head_dim, k);

    let tk = transpose(builder, 2, 3, k);
    let attn = matmul(builder, q, tk);
    let denom = constant(
        builder,
        (head_dim as f32).sqrt(),
        &shape(builder, attn.clone()),
    );
    let attn = softmax(builder, attn / denom);
    let attn = matmul(builder, attn, v);
    let attn = transpose(builder, 1, 2, attn);
    let attn = reshape(builder, shape!(builder, b, s, dim), attn);
    linear(builder, dim, dim, p.extend(["proj"]).unwrap(), attn)
}

fn qwen_vision_mlp(builder: &Builder, config: &Qwen3_5VisionConfig, p: Path, x: Var) -> Var {
    let x = linear(
        builder,
        config.hidden_size,
        config.intermediate_size,
        p.extend(["linear_fc1"]).unwrap(),
        x,
    );
    let x = gelu(builder, x);
    linear(
        builder,
        config.intermediate_size,
        config.hidden_size,
        p.extend(["linear_fc2"]).unwrap(),
        x,
    )
}

fn qwen_vision_block(
    builder: &Builder,
    config: &Qwen3_5VisionConfig,
    cos: Var,
    sin: Var,
    p: Path,
    x: Var,
) -> Var {
    let res = x.clone();
    let x = layernorm(builder, 1e-6, p.extend(["norm1"]).unwrap(), x);
    let x = qwen_vision_attention(builder, config, cos, sin, p.extend(["attn"]).unwrap(), x);
    let x = x + res;

    let res = x.clone();
    let x = layernorm(builder, 1e-6, p.extend(["norm2"]).unwrap(), x);
    let x = qwen_vision_mlp(builder, config, p.extend(["mlp"]).unwrap(), x);
    x + res
}

fn qwen_vision_patch_merger(
    builder: &Builder,
    config: &Qwen3_5VisionConfig,
    merged_grid_height: usize,
    merged_grid_width: usize,
    p: Path,
    x: Var,
) -> Var {
    let [b, _, _] = unpack::<3>(builder, shape(builder, x.clone()));
    let merge_sq = config.spatial_merge_size * config.spatial_merge_size;
    let x = layernorm(builder, 1e-6, p.extend(["norm"]).unwrap(), x);
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            merged_grid_height * merged_grid_width,
            config.hidden_size * merge_sq
        ),
        x,
    );
    let x = linear(
        builder,
        config.hidden_size * merge_sq,
        config.hidden_size * merge_sq,
        p.extend(["linear_fc1"]).unwrap(),
        x,
    );
    let x = gelu(builder, x);
    linear(
        builder,
        config.hidden_size * merge_sq,
        config.out_hidden_size,
        p.extend(["linear_fc2"]).unwrap(),
        x,
    )
}

#[derive(Debug, Clone)]
pub struct Qwen3_5VisionModel {
    vision_config: Qwen3_5VisionConfig,
    base_grid_size: usize,
    raw_grid_height: usize,
    raw_grid_width: usize,
    merged_grid_height: usize,
    merged_grid_width: usize,
}

impl DynModule for Qwen3_5VisionModel {
    fn path(&self) -> Path {
        path(vec!["Qwen3_5Vision"]).expect("invalid model path")
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;

        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [pixels]: [Var; 1] = args.try_into().expect("expected 1 input");
        let visual_root = path(vec!["model", "visual"]).unwrap();
        let mut x = qwen_vision_embeddings(
            builder,
            &self.vision_config,
            self.base_grid_size,
            self.raw_grid_height,
            self.raw_grid_width,
            visual_root.clone(),
            pixels,
        );
        let (cos, sin) = qwen_vision_rotary_tables(
            builder,
            self.raw_grid_height,
            self.raw_grid_width,
            self.vision_config.spatial_merge_size,
            self.vision_config.hidden_size / self.vision_config.num_heads,
        );
        for layer_id in 0..self.vision_config.depth {
            x = qwen_vision_block(
                builder,
                &self.vision_config,
                cos.clone(),
                sin.clone(),
                visual_root
                    .extend(["blocks", &layer_id.to_string()])
                    .unwrap(),
                x,
            );
        }
        x = qwen_vision_patch_merger(
            builder,
            &self.vision_config,
            self.merged_grid_height,
            self.merged_grid_width,
            visual_root.extend(["merger"]).unwrap(),
            x,
        );
        vec![x]
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5MultimodalModel {
    language_model: Qwen3_5Model,
}

impl Qwen3_5MultimodalModel {
    fn multimodal_rope_and_delta(
        &self,
        builder: &Builder,
        text1_len: Var,
        image_len: Var,
        text2_len: Var,
        cache_pos: Var,
        in_mm_delta: Var,
    ) -> ((Var, Var), Var) {
        let image_len_u32 = nat_to_u32(builder, image_len.clone());
        let zero = constant(builder, 0u32, &shape(builder, image_len_u32.clone()));
        let use_image = gt(builder, image_len_u32, zero);

        let text1_axes = self.language_model.text_axes_from_start(
            builder,
            nat_to_f32(builder, cache_pos.clone()),
            text1_len.clone(),
        );

        let image_start = nat_to_f32(builder, cache_pos.clone() + text1_len.clone());
        let image_axes = self
            .language_model
            .image_axes_from_start(builder, image_start)
            .map(|axis| slice(builder, 1, 0, image_len.clone(), axis));

        let mm = self
            .language_model
            .multimodal
            .as_ref()
            .expect("qwen3_5 multimodal rope requires multimodal config");
        let delta_shape = shape(builder, in_mm_delta.clone());
        let prefill_text2_start = broadcast(
            builder,
            delta_shape.clone(),
            nat_to_f32(
                builder,
                cache_pos.clone()
                    + text1_len
                    + mm.merged_grid_height
                        .max(mm.merged_grid_width)
                        .to_nat(builder),
            ),
        );
        let decode_text2_start =
            broadcast(builder, delta_shape, nat_to_f32(builder, cache_pos)) + in_mm_delta.clone();
        let text2_start = where_broadcast(
            builder,
            use_image.clone(),
            prefill_text2_start,
            decode_text2_start,
        );
        let text2_axes = self
            .language_model
            .text_axes_from_start(builder, text2_start, text2_len);

        let axes = [
            concat_many(
                builder,
                1,
                vec![
                    text1_axes[0].clone(),
                    image_axes[0].clone(),
                    text2_axes[0].clone(),
                ],
            ),
            concat_many(
                builder,
                1,
                vec![
                    text1_axes[1].clone(),
                    image_axes[1].clone(),
                    text2_axes[1].clone(),
                ],
            ),
            concat_many(
                builder,
                1,
                vec![
                    text1_axes[2].clone(),
                    image_axes[2].clone(),
                    text2_axes[2].clone(),
                ],
            ),
        ];
        let rope = self.language_model.multimodal_rope_tables(builder, axes);
        let out_mm_delta = where_broadcast(
            builder,
            use_image,
            self.language_model.multimodal_delta_tensor(builder),
            in_mm_delta,
        );
        (rope, out_mm_delta)
    }

    fn forward_image_and_texts(
        &self,
        builder: &Builder,
        text1: Var,
        image: Var,
        text2: Var,
        in_k: Var,
        in_v: Var,
        in_conv: Var,
        in_recurrent: Var,
        in_mm_delta: Var,
        max_positions: Var,
        num_chunks: Var,
    ) -> Vec<Var> {
        let embed_path = path(vec!["model", "language_model", "embed_tokens"]).unwrap();
        let text1_embeddings = embeddings(builder, embed_path.clone(), text1.clone());
        let text2_embeddings = embeddings(builder, embed_path, text2.clone());
        let embeddings = concat(builder, 1, text1_embeddings, image.clone());
        let embeddings = concat(builder, 1, embeddings, text2_embeddings);

        let [_, text1_len] = unpack::<2>(builder, shape(builder, text1));
        let [_, text2_len] = unpack::<2>(builder, shape(builder, text2));
        let [_, image_len, _] = unpack::<3>(builder, shape(builder, image));
        let [_, _, _, cache_pos, _] = unpack::<5>(builder, shape(builder, in_k.clone()));
        let (rope, out_mm_delta) = self.multimodal_rope_and_delta(
            builder,
            text1_len,
            image_len,
            text2_len,
            cache_pos,
            in_mm_delta,
        );

        let mut outputs = self.language_model.forward_with_embeddings(
            builder,
            Path::empty(),
            embeddings,
            in_k,
            in_v,
            in_conv,
            in_recurrent,
            max_positions,
            num_chunks,
            Some(rope),
        );
        outputs.push(out_mm_delta);
        outputs
    }
}

impl DynModule for Qwen3_5MultimodalModel {
    fn path(&self) -> Path {
        path(vec!["Qwen3_5VLM"]).expect("invalid model path")
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;

        let t = Type::Tensor(TypeExpr::Var(0));
        let (mut source, target) = self.language_model.ty();
        source.remove(0);
        source.insert(0, t.clone());
        source.insert(0, t.clone());
        source.insert(0, t);
        (source, target)
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [
            text1,
            image,
            text2,
            in_k,
            in_v,
            in_conv,
            in_recurrent,
            in_mm_delta,
            max_positions,
            num_chunks,
        ]: [Var; 10] = args.try_into().expect("expected 10 inputs");
        self.forward_image_and_texts(
            builder,
            text1,
            image,
            text2,
            in_k,
            in_v,
            in_conv,
            in_recurrent,
            in_mm_delta,
            max_positions,
            num_chunks,
        )
    }
}
