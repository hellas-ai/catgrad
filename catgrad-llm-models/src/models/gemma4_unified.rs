#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use crate::utils::load_and_patchify_dynamic_image;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct Gemma4UnifiedConfig {
    text_config: Gemma4UnifiedTextConfig,
    vision_config: Gemma4UnifiedVisionConfig,
    audio_config: Gemma4UnifiedAudioConfig,
    image_token_id: usize,
    audio_token_id: usize,
    eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4UnifiedRopeTypeConfig {
    rope_theta: f32,
    partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4UnifiedRopeParameters {
    full_attention: Gemma4UnifiedRopeTypeConfig,
    sliding_attention: Gemma4UnifiedRopeTypeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4UnifiedTextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_global_key_value_heads: Option<usize>,
    head_dim: usize,
    global_head_dim: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    layer_types: Vec<String>,
    final_logit_softcapping: Option<f32>,
    vocab_size: usize,
    hidden_size_per_layer_input: usize,
    attention_bias: bool,
    attention_k_eq_v: bool,
    num_kv_shared_layers: usize,
    enable_moe_block: bool,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    top_k_experts: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    use_double_wide_mlp: bool,
    tie_word_embeddings: bool,
    rope_parameters: Gemma4UnifiedRopeParameters,
    pad_token_id: usize,
    eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4UnifiedVisionConfig {
    mm_embed_dim: usize,
    model_patch_size: usize,
    num_soft_tokens: usize,
    output_proj_dims: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    rms_norm_eps: f32,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Gemma4UnifiedRuntimeVisionConfig {
    pub num_soft_tokens_per_image: usize,
}

#[derive(Debug, Clone)]
pub struct Gemma4UnifiedPreparedImageInput {
    pub patches: Vec<f32>,
    pub shape: Vec<usize>,
    pub runtime_vision: Gemma4UnifiedRuntimeVisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4UnifiedAudioConfig {
    audio_samples_per_token: usize,
    rms_norm_eps: f32,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Gemma4UnifiedRuntimeAudioConfig {
    pub num_audio_samples: usize,
    pub num_soft_tokens_per_audio: usize,
}

#[derive(Debug, Clone)]
pub struct Gemma4UnifiedPreparedAudioInput {
    pub features: Vec<f32>,
    pub shape: Vec<usize>,
    pub mask: Vec<f32>,
    pub mask_shape: Vec<usize>,
    pub runtime_audio: Gemma4UnifiedRuntimeAudioConfig,
}

pub fn prepare_gemma4_unified_audio_input_from_waveform(
    waveform: &[f32],
    config_json: &serde_json::Value,
) -> crate::Result<Gemma4UnifiedPreparedAudioInput> {
    let config: Gemma4UnifiedConfig = serde_json::from_value(config_json.clone())?;
    let samples_per_token = config.audio_config.audio_samples_per_token;
    if waveform.is_empty() {
        return Err(crate::LLMError::UnsupportedWireConversion(
            "audio input contained no samples".to_string(),
        ));
    }

    let num_soft_tokens_per_audio = waveform.len().div_ceil(samples_per_token);
    let padded_samples = num_soft_tokens_per_audio * samples_per_token;
    let mut features = vec![0.0; padded_samples];
    features[..waveform.len()].copy_from_slice(waveform);

    Ok(Gemma4UnifiedPreparedAudioInput {
        features,
        shape: vec![1, num_soft_tokens_per_audio, samples_per_token],
        mask: vec![1.0; num_soft_tokens_per_audio],
        mask_shape: vec![1, num_soft_tokens_per_audio],
        runtime_audio: Gemma4UnifiedRuntimeAudioConfig {
            num_audio_samples: waveform.len(),
            num_soft_tokens_per_audio,
        },
    })
}

fn merge_gemma4_unified_teacher_patches(
    teacher_patches: &[f32],
    teacher_patch_grid_height: usize,
    teacher_patch_grid_width: usize,
    teacher_patch_size: usize,
    pooling_kernel_size: usize,
) -> Vec<f32> {
    let patch_dim = 3 * teacher_patch_size * teacher_patch_size;
    let model_patch_size = teacher_patch_size * pooling_kernel_size;
    let model_patch_dim = 3 * model_patch_size * model_patch_size;
    let model_patch_grid_height = teacher_patch_grid_height / pooling_kernel_size;
    let model_patch_grid_width = teacher_patch_grid_width / pooling_kernel_size;
    let num_model_patches = model_patch_grid_height * model_patch_grid_width;
    let mut merged = vec![0.0; num_model_patches * model_patch_dim];

    for model_row in 0..model_patch_grid_height {
        for model_col in 0..model_patch_grid_width {
            let model_patch_index = model_row * model_patch_grid_width + model_col;
            let model_patch_offset = model_patch_index * model_patch_dim;

            for kernel_row in 0..pooling_kernel_size {
                for kernel_col in 0..pooling_kernel_size {
                    let teacher_row = model_row * pooling_kernel_size + kernel_row;
                    let teacher_col = model_col * pooling_kernel_size + kernel_col;
                    let teacher_patch_index = teacher_row * teacher_patch_grid_width + teacher_col;
                    let teacher_patch_offset = teacher_patch_index * patch_dim;

                    for inner_row in 0..teacher_patch_size {
                        for inner_col in 0..teacher_patch_size {
                            let dst_row = kernel_row * teacher_patch_size + inner_row;
                            let dst_col = kernel_col * teacher_patch_size + inner_col;
                            let dst_pixel_offset =
                                model_patch_offset + (dst_row * model_patch_size + dst_col) * 3;
                            let src_pixel_offset = teacher_patch_offset
                                + (inner_row * teacher_patch_size + inner_col) * 3;
                            merged[dst_pixel_offset..dst_pixel_offset + 3].copy_from_slice(
                                &teacher_patches[src_pixel_offset..src_pixel_offset + 3],
                            );
                        }
                    }
                }
            }
        }
    }

    merged
}

pub fn prepare_gemma4_unified_image_input(
    image: &image::DynamicImage,
    config_json: &serde_json::Value,
) -> crate::Result<Gemma4UnifiedPreparedImageInput> {
    let config: Gemma4UnifiedConfig = serde_json::from_value(config_json.clone())?;
    let patched = load_and_patchify_dynamic_image(
        image,
        config.vision_config.patch_size,
        config.vision_config.num_soft_tokens,
        config.vision_config.pooling_kernel_size,
    )?;
    let model_patch_grid_height =
        patched.patch_grid_height / config.vision_config.pooling_kernel_size;
    let model_patch_grid_width =
        patched.patch_grid_width / config.vision_config.pooling_kernel_size;
    let num_soft_tokens_per_image = model_patch_grid_height * model_patch_grid_width;
    let merged_patches = merge_gemma4_unified_teacher_patches(
        &patched.data,
        patched.patch_grid_height,
        patched.patch_grid_width,
        config.vision_config.patch_size,
        config.vision_config.pooling_kernel_size,
    );
    let model_patch_dim =
        3 * config.vision_config.model_patch_size * config.vision_config.model_patch_size;
    let packed_patch_dim = model_patch_dim + 2;
    let mut packed = vec![0.0; num_soft_tokens_per_image * packed_patch_dim];

    for patch_index in 0..num_soft_tokens_per_image {
        let src_offset = patch_index * model_patch_dim;
        let dst_offset = patch_index * packed_patch_dim;
        packed[dst_offset..dst_offset + model_patch_dim]
            .copy_from_slice(&merged_patches[src_offset..src_offset + model_patch_dim]);
        packed[dst_offset + model_patch_dim] = (patch_index % model_patch_grid_width) as f32;
        packed[dst_offset + model_patch_dim + 1] = (patch_index / model_patch_grid_width) as f32;
    }

    Ok(Gemma4UnifiedPreparedImageInput {
        patches: packed,
        shape: vec![1, num_soft_tokens_per_image, packed_patch_dim],
        runtime_vision: Gemma4UnifiedRuntimeVisionConfig {
            num_soft_tokens_per_image,
        },
    })
}

impl Gemma4UnifiedTextConfig {
    fn is_sliding_attention_layer(&self, layer_id: usize) -> bool {
        self.layer_types[layer_id] == "sliding_attention"
    }

    fn is_shared_kv_layer(&self, layer_id: usize) -> bool {
        self.num_kv_shared_layers > 0
            && layer_id >= self.num_hidden_layers - self.num_kv_shared_layers
    }

    fn full_num_key_value_heads(&self) -> usize {
        if self.attention_k_eq_v {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    fn head_dim_for_layer(&self, layer_id: usize) -> usize {
        if self.is_sliding_attention_layer(layer_id) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    fn num_key_value_heads_for_layer(&self, layer_id: usize) -> usize {
        if self.is_sliding_attention_layer(layer_id) {
            self.num_key_value_heads
        } else {
            self.full_num_key_value_heads()
        }
    }

    fn use_alternative_attention(&self, layer_id: usize) -> bool {
        self.attention_k_eq_v && !self.is_sliding_attention_layer(layer_id)
    }

    fn mlp_intermediate_size(&self, layer_id: usize) -> usize {
        if self.use_double_wide_mlp && self.is_shared_kv_layer(layer_id) {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    fn full_partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .full_attention
            .partial_rotary_factor
            .expect("gemma4 unified full_attention partial_rotary_factor missing")
    }
}

impl LLMConfig for Gemma4UnifiedTextConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_layers(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
            .max(self.full_num_key_value_heads())
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .full_attention
            .rope_theta
            .max(self.rope_parameters.sliding_attention.rope_theta)
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.full_partial_rotary_factor()
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim.max(self.global_head_dim)
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4UnifiedAttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy)]
enum Gemma4UnifiedCacheSource {
    Own(usize),
    Shared(usize),
}

#[derive(Debug, Clone, Copy)]
struct Gemma4UnifiedLayerPlan {
    kind: Gemma4UnifiedAttentionKind,
    cache: Gemma4UnifiedCacheSource,
}

#[derive(Debug, Clone)]
enum Gemma4UnifiedMultimodalConfig {
    Vision {
        vision_config: Gemma4UnifiedVisionConfig,
        image_token_index: usize,
        runtime_vision: Gemma4UnifiedRuntimeVisionConfig,
    },
    Audio {
        audio_config: Gemma4UnifiedAudioConfig,
        audio_token_index: usize,
        runtime_audio: Gemma4UnifiedRuntimeAudioConfig,
    },
}

#[derive(Debug, Clone)]
pub struct Gemma4UnifiedModel {
    root: String,
    config: Gemma4UnifiedTextConfig,
    dtype: Dtype,
    layer_plans: Vec<Gemma4UnifiedLayerPlan>,
    sliding_cache_layers: usize,
    full_cache_layers: usize,
    multimodal: Option<Gemma4UnifiedMultimodalConfig>,
}

impl LLMModel for Gemma4UnifiedModel {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn multimodal_metadata(&self) -> Option<MultimodalMetadata> {
        let mm = self.multimodal.as_ref()?;
        match mm {
            Gemma4UnifiedMultimodalConfig::Vision {
                vision_config,
                image_token_index,
                runtime_vision,
            } => Some(MultimodalMetadata {
                image_token_index: *image_token_index,
                mm_tokens_per_image: runtime_vision.num_soft_tokens_per_image,
                hidden_size: self.config.hidden_size,
                image_size: 0,
                patch_size: vision_config.model_patch_size,
            }),
            Gemma4UnifiedMultimodalConfig::Audio {
                audio_token_index,
                runtime_audio,
                ..
            } => Some(MultimodalMetadata {
                image_token_index: *audio_token_index,
                mm_tokens_per_image: runtime_audio.num_soft_tokens_per_audio,
                hidden_size: self.config.hidden_size,
                image_size: 0,
                patch_size: 0,
            }),
        }
    }

    fn multimodal_vision_module(&self) -> Option<Box<dyn DynModule>> {
        let mm = self.multimodal.as_ref()?;
        match mm {
            Gemma4UnifiedMultimodalConfig::Vision {
                vision_config,
                runtime_vision,
                ..
            } => Some(Box::new(Gemma4UnifiedVisionEmbeddings {
                vision_config: vision_config.clone(),
                runtime_vision: runtime_vision.clone(),
                text_hidden_size: self.config.hidden_size,
            })),
            Gemma4UnifiedMultimodalConfig::Audio {
                audio_config,
                runtime_audio,
                ..
            } => Some(Box::new(Gemma4UnifiedAudioEmbeddings {
                audio_config: audio_config.clone(),
                runtime_audio: runtime_audio.clone(),
                text_hidden_size: self.config.hidden_size,
            })),
        }
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        self.multimodal.as_ref()?;
        Some(Box::new(Gemma4UnifiedMultimodalModel {
            language_model: self.clone(),
        }))
    }

    fn multimodal_interpolate_prompt(&self, prompt: &str) -> Option<String> {
        let mm = self.multimodal.as_ref()?;
        match mm {
            Gemma4UnifiedMultimodalConfig::Vision { runtime_vision, .. } => Some(prompt.replace(
                "<|image|>",
                &format!(
                    "{}{}{}",
                    "<|image>",
                    "<|image|>".repeat(runtime_vision.num_soft_tokens_per_image),
                    "<image|>"
                ),
            )),
            Gemma4UnifiedMultimodalConfig::Audio { runtime_audio, .. } => Some(prompt.replace(
                "<|audio|>",
                &format!(
                    "{}{}{}",
                    "<|audio>",
                    "<|audio|>".repeat(runtime_audio.num_soft_tokens_per_audio),
                    "<audio|>"
                ),
            )),
        }
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let dtype = self.dtype();
        vec![
            (
                dtype,
                Shape(vec![
                    self.sliding_cache_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.head_dim,
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.sliding_cache_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.head_dim,
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.full_cache_layers,
                    1,
                    self.config.full_num_key_value_heads(),
                    0,
                    self.config.global_head_dim,
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.full_cache_layers,
                    1,
                    self.config.full_num_key_value_heads(),
                    0,
                    self.config.global_head_dim,
                ]),
            ),
        ]
    }
}

impl Gemma4UnifiedModel {
    pub fn new(
        root: &str,
        config_json: &serde_json::Value,
        runtime_vision: Option<&Gemma4UnifiedRuntimeVisionConfig>,
        runtime_audio: Option<&Gemma4UnifiedRuntimeAudioConfig>,
        dtype: Dtype,
    ) -> crate::Result<Self> {
        let Gemma4UnifiedConfig {
            mut text_config,
            vision_config,
            image_token_id,
            audio_config,
            audio_token_id,
            eos_token_id,
            ..
        }: Gemma4UnifiedConfig = serde_json::from_value(config_json.clone())?;

        if let Some(eos_token_id) = eos_token_id {
            text_config.eos_token_id = Some(eos_token_id);
        }

        let multimodal = match runtime_vision {
            Some(runtime_vision) => Some(Gemma4UnifiedMultimodalConfig::Vision {
                vision_config,
                image_token_index: image_token_id,
                runtime_vision: runtime_vision.clone(),
            }),
            None => runtime_audio.map(|runtime_audio| Gemma4UnifiedMultimodalConfig::Audio {
                audio_config,
                audio_token_index: audio_token_id,
                runtime_audio: runtime_audio.clone(),
            }),
        };

        let first_shared_layer = text_config.num_hidden_layers - text_config.num_kv_shared_layers;
        let mut layer_plans = Vec::with_capacity(text_config.num_hidden_layers);
        let mut sliding_cache_layers = 0;
        let mut full_cache_layers = 0;
        let mut last_sliding_cache_id = None;
        let mut last_full_cache_id = None;

        for layer_id in 0..text_config.num_hidden_layers {
            let kind = if text_config.is_sliding_attention_layer(layer_id) {
                Gemma4UnifiedAttentionKind::Sliding
            } else {
                Gemma4UnifiedAttentionKind::Full
            };

            let cache = if layer_id < first_shared_layer {
                match kind {
                    Gemma4UnifiedAttentionKind::Sliding => {
                        let cache_id = sliding_cache_layers;
                        sliding_cache_layers += 1;
                        last_sliding_cache_id = Some(cache_id);
                        Gemma4UnifiedCacheSource::Own(cache_id)
                    }
                    Gemma4UnifiedAttentionKind::Full => {
                        let cache_id = full_cache_layers;
                        full_cache_layers += 1;
                        last_full_cache_id = Some(cache_id);
                        Gemma4UnifiedCacheSource::Own(cache_id)
                    }
                }
            } else {
                match kind {
                    Gemma4UnifiedAttentionKind::Sliding => Gemma4UnifiedCacheSource::Shared(
                        last_sliding_cache_id.ok_or_else(|| {
                            crate::LLMError::InvalidModelConfig(
                                "gemma4 unified sliding shared KV layer had no source layer"
                                    .to_string(),
                            )
                        })?,
                    ),
                    Gemma4UnifiedAttentionKind::Full => {
                        Gemma4UnifiedCacheSource::Shared(last_full_cache_id.ok_or_else(|| {
                            crate::LLMError::InvalidModelConfig(
                                "gemma4 unified full shared KV layer had no source layer"
                                    .to_string(),
                            )
                        })?)
                    }
                }
            };

            layer_plans.push(Gemma4UnifiedLayerPlan { kind, cache });
        }

        Ok(Self {
            root: root.to_string(),
            config: text_config,
            dtype,
            layer_plans,
            sliding_cache_layers,
            full_cache_layers,
            multimodal,
        })
    }

    fn scaled_embeddings(&self, builder: &Builder, p: Path, x: Var, scale: f32) -> Var {
        let x = embeddings(builder, p, x);
        let scale = constant(builder, scale, &shape(builder, x.clone()));
        let scale = cast(builder, scale, dtype(builder, x.clone()));
        x * scale
    }

    fn get_per_layer_inputs(&self, builder: &Builder, p: Path, input_ids: Var) -> Option<Var> {
        if self.config.hidden_size_per_layer_input == 0 {
            return None;
        }

        let pli = self.scaled_embeddings(
            builder,
            p.extend(["embed_tokens_per_layer"]).unwrap(),
            input_ids,
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
        );
        let [b, s, _] = unpack::<3>(builder, shape(builder, pli.clone()));
        let pli = reshape(
            builder,
            shape!(
                builder,
                b,
                s,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input
            ),
            pli,
        );
        Some(pli)
    }

    fn project_per_layer_inputs(
        &self,
        builder: &Builder,
        p: Path,
        inputs_embeds: Var,
        per_layer_inputs: Option<Var>,
    ) -> Option<Var> {
        if self.config.hidden_size_per_layer_input == 0 {
            return None;
        }

        let [b, s, _] = unpack::<3>(builder, shape(builder, inputs_embeds.clone()));
        let projection = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.num_hidden_layers * self.config.hidden_size_per_layer_input,
            p.extend(["per_layer_model_projection"]).unwrap(),
            inputs_embeds,
        );
        let scale = constant(
            builder,
            (self.config.hidden_size as f32).powf(-0.5),
            &shape(builder, projection.clone()),
        );
        let scale = cast(builder, scale, dtype(builder, projection.clone()));
        let projection = projection * scale;
        let projection = reshape(
            builder,
            shape!(
                builder,
                b,
                s,
                self.config.num_hidden_layers,
                self.config.hidden_size_per_layer_input
            ),
            projection,
        );
        let projection = rmsnorm::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["per_layer_projection_norm"]).unwrap(),
            projection,
        );

        if per_layer_inputs.is_none() {
            return Some(projection);
        }

        let scale = constant(
            builder,
            2.0f32.powf(-0.5),
            &shape(builder, projection.clone()),
        );
        let scale = cast(builder, scale, dtype(builder, projection.clone()));
        Some((projection + per_layer_inputs.unwrap()) * scale)
    }

    fn mlp(&self, builder: &Builder, layer_id: usize, p: Path, x: Var) -> Var {
        let intermediate_size = self.config.mlp_intermediate_size(layer_id);
        let gate = linear_no_bias(
            builder,
            self.config.hidden_size,
            intermediate_size,
            p.extend(["gate_proj"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            self.config.hidden_size,
            intermediate_size,
            p.extend(["up_proj"]).unwrap(),
            x,
        );
        let x = gelu(builder, gate) * up;
        linear_no_bias(
            builder,
            intermediate_size,
            self.config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    fn moe(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let num_experts = self
            .config
            .num_experts
            .expect("gemma4 unified enable_moe_block missing num_experts");
        let top_k_experts = self
            .config
            .top_k_experts
            .expect("gemma4 unified enable_moe_block missing top_k_experts");
        let moe_intermediate_size = self
            .config
            .moe_intermediate_size
            .expect("gemma4 unified enable_moe_block missing moe_intermediate_size");
        let [batch_size, seq_len, hidden_size] = unpack::<3>(builder, shape(builder, x.clone()));
        let routed = rmsnorm_raw::<3>(builder, self.config.rms_norm_eps, x.clone());
        let routed_scale = param(builder, &p.extend(["router", "scale"]).unwrap());
        let routed_scale = broadcast(builder, shape(builder, routed.clone()), routed_scale);
        let routed_scalar = constant(
            builder,
            (self.config.hidden_size as f32).powf(-0.5),
            &shape(builder, routed.clone()),
        );
        let routed_scalar = cast(builder, routed_scalar, dtype(builder, routed.clone()));
        let routed = routed * routed_scale * routed_scalar;
        let routed = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_experts,
            p.extend(["router", "proj"]).unwrap(),
            routed,
        );
        let routed = softmax(builder, routed);

        let num_tokens = batch_size.clone() * seq_len.clone();
        let (mut top_k_weights, top_k_index) = topk(builder, top_k_experts, routed);
        let top_k_shape = shape!(builder, num_tokens, top_k_experts);
        top_k_weights = reshape(builder, top_k_shape.clone(), top_k_weights);
        let top_k_index = reshape(builder, top_k_shape, top_k_index);
        let top_k_shape = shape(builder, top_k_weights.clone());
        let top_k_sum = sum(builder, top_k_weights.clone());
        let top_k_sum = broadcast(builder, top_k_shape, top_k_sum);
        top_k_weights = top_k_weights / top_k_sum;

        let expert_input = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_feedforward_layernorm_2"]).unwrap(),
            x,
        );
        let fullx_shape = shape!(builder, num_tokens, 1, hidden_size);
        let fullx = reshape(builder, fullx_shape.clone(), expert_input);

        let gate_up_proj = param(builder, &p.extend(["experts", "gate_up_proj"]).unwrap());
        let down_proj = param(builder, &p.extend(["experts", "down_proj"]).unwrap());
        let per_expert_scale = param(builder, &p.extend(["router", "per_expert_scale"]).unwrap());
        let mut sumk = constant(builder, 0.0, &fullx_shape);
        sumk = cast(builder, sumk, dtype(builder, fullx.clone()));

        for i in 0..top_k_experts {
            let idx = get(builder, 1, i, top_k_index.clone());
            let idx = squeeze::<2, 1>(builder, 1, idx);
            let mut val = get(builder, 1, i, top_k_weights.clone());

            let expert_scale = index(builder, 0, idx.clone(), per_expert_scale.clone());
            let expert_scale = unsqueeze::<1, 2>(builder, 1, expert_scale);
            val = val * expert_scale;

            let gate_up = index(builder, 0, idx.clone(), gate_up_proj.clone());
            let gate_up = transpose(builder, 1, 2, gate_up);
            let down = index(builder, 0, idx, down_proj.clone());
            let down = transpose(builder, 1, 2, down);

            let gate_up = matmul(builder, fullx.clone(), gate_up);
            let [gate, up]: [Var; 2] = chunk(builder, 2, 2, moe_intermediate_size, gate_up)
                .try_into()
                .unwrap();
            let x = gelu(builder, gate) * up;
            let x = matmul(builder, x, down);

            let val = unsqueeze::<2, 3>(builder, 2, val);
            let val = broadcast(builder, shape(builder, x.clone()), val);
            sumk = sumk + x * val;
        }

        let x = reshape(
            builder,
            shape!(builder, batch_size, seq_len, hidden_size),
            sumk,
        );
        rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm_2"]).unwrap(),
            x,
        )
    }

    fn attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        sliding_cache: &mut KVCache,
        full_cache: &mut KVCache,
        pos: Var,
        cos: Var,
        sin: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads_for_layer(layer_id);
        let head_dim = self.config.head_dim_for_layer(layer_id);
        let rep = num_heads / num_kv_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_b(
            builder,
            self.config.hidden_size,
            num_heads * head_dim,
            self.config.attention_bias,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let k = linear_b(
            builder,
            self.config.hidden_size,
            num_kv_heads * head_dim,
            self.config.attention_bias,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = if self.config.use_alternative_attention(layer_id) {
            k.clone()
        } else {
            linear_b(
                builder,
                self.config.hidden_size,
                num_kv_heads * head_dim,
                self.config.attention_bias,
                p.extend(["v_proj"]).unwrap(),
                x,
            )
        };

        let q = reshape(builder, shape!(builder, b, s, num_heads, head_dim), q);
        let k = reshape(builder, shape!(builder, b, s, num_kv_heads, head_dim), k);
        let v = reshape(builder, shape!(builder, b, s, num_kv_heads, head_dim), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = rmsnorm::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );
        let v = rmsnorm_raw::<4>(builder, self.config.rms_norm_eps, v);

        let q = apply_rope_embedding(builder, pos.clone(), head_dim, cos.clone(), sin.clone(), q);
        let k = apply_rope_embedding(builder, pos, head_dim, cos, sin, k);

        let plan = self.layer_plans[layer_id];
        let (cache, cache_id) = match plan.kind {
            Gemma4UnifiedAttentionKind::Sliding => (
                sliding_cache,
                match plan.cache {
                    Gemma4UnifiedCacheSource::Own(cache_id)
                    | Gemma4UnifiedCacheSource::Shared(cache_id) => cache_id,
                },
            ),
            Gemma4UnifiedAttentionKind::Full => (
                full_cache,
                match plan.cache {
                    Gemma4UnifiedCacheSource::Own(cache_id)
                    | Gemma4UnifiedCacheSource::Shared(cache_id) => cache_id,
                },
            ),
        };
        let (k, v) = match plan.cache {
            Gemma4UnifiedCacheSource::Own(_) => cache.update(builder, cache_id, k, v),
            Gemma4UnifiedCacheSource::Shared(_) => cache.get(cache_id),
        };

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let mut attn = matmul(builder, q, transpose(builder, 2, 3, k));
        let sh = shape(builder, attn.clone());
        let attention_mask = cast(builder, attention_mask, dtype(builder, attn.clone()));
        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let attn = transpose(builder, 1, 2, attn);
        let attn = reshape(builder, shape!(builder, b, s, num_heads * head_dim), attn);

        linear_b(
            builder,
            num_heads * head_dim,
            self.config.hidden_size,
            self.config.attention_bias,
            p.extend(["o_proj"]).unwrap(),
            attn,
        )
    }

    fn layer_per_layer_input(
        &self,
        builder: &Builder,
        layer_id: usize,
        per_layer_inputs: Var,
    ) -> Var {
        squeeze::<4, 3>(builder, 2, slice(builder, 2, layer_id, 1, per_layer_inputs))
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        per_layer_input: Option<Var>,
        sliding_cache: &mut KVCache,
        full_cache: &mut KVCache,
        pos: Var,
        cos: Var,
        sin: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let residual = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = self.attention(
            builder,
            layer_id,
            attention_mask,
            sliding_cache,
            full_cache,
            pos,
            cos,
            sin,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = residual + x;

        let residual = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_feedforward_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, layer_id, p.extend(["mlp"]).unwrap(), x);
        let x = if self.config.enable_moe_block {
            let x_dense = rmsnorm::<3>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["post_feedforward_layernorm_1"]).unwrap(),
                x,
            );
            x_dense + self.moe(builder, p.clone(), residual.clone())
        } else {
            x
        };
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm"]).unwrap(),
            x,
        );
        let mut x = residual + x;

        if let Some(per_layer_input) = per_layer_input {
            let residual = x.clone();
            let x_per_layer = linear_no_bias(
                builder,
                self.config.hidden_size,
                self.config.hidden_size_per_layer_input,
                p.extend(["per_layer_input_gate"]).unwrap(),
                x,
            );
            let x_per_layer = gelu(builder, x_per_layer) * per_layer_input;
            let x_per_layer = linear_no_bias(
                builder,
                self.config.hidden_size_per_layer_input,
                self.config.hidden_size,
                p.extend(["per_layer_projection"]).unwrap(),
                x_per_layer,
            );
            let x_per_layer = rmsnorm::<3>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["post_per_layer_input_norm"]).unwrap(),
                x_per_layer,
            );
            x = residual + x_per_layer;
        }

        let layer_scalar = param(builder, &p.extend(["layer_scalar"]).unwrap());
        let layer_scalar = broadcast(builder, shape(builder, x.clone()), layer_scalar);
        x * layer_scalar
    }

    fn forward_embeddings(
        &self,
        builder: &Builder,
        p: Path,
        language_root: Path,
        full_attention_mask: Var,
        bidirectional_span: Option<(Var, usize)>,
        inputs_embeds: Var,
        per_layer_inputs: Option<Var>,
        in_sliding_k: Var,
        in_sliding_v: Var,
        in_full_k: Var,
        in_full_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let [_, s, _] = unpack::<3>(builder, shape(builder, inputs_embeds.clone()));
        let [_, _, _, sliding_pos, _] = unpack::<5>(builder, shape(builder, in_sliding_k.clone()));
        let [_, _, _, full_pos, _] = unpack::<5>(builder, shape(builder, in_full_k.clone()));

        let sliding_attention_mask = sliding_window_mask(
            builder,
            s.clone(),
            sliding_pos.clone(),
            self.config.sliding_window,
        );
        let (sliding_attention_mask, full_attention_mask) = if let Some((span_start, span_size)) =
            bidirectional_span
        {
            let full_pos_u32 = nat_to_u32(builder, full_pos.clone());
            let zero_pos = constant(builder, 0u32, &shape(builder, full_pos_u32.clone()));
            let is_decode = gt(builder, full_pos_u32, zero_pos);
            let masks = cond(
                builder,
                is_decode,
                |_b, args: Vec<Var>| {
                    let [sliding_attention_mask, full_attention_mask, _s, _span_start] =
                        args.try_into().unwrap();
                    vec![sliding_attention_mask, full_attention_mask]
                },
                |b, args: Vec<Var>| {
                    let [sliding_attention_mask, full_attention_mask, s, span_start] =
                        args.try_into().unwrap();
                    let mask = Gemma4UnifiedMultimodalModel::bidirectional_mask(
                        b, s, span_start, span_size,
                    );
                    vec![
                        sliding_attention_mask * mask.clone(),
                        full_attention_mask * mask,
                    ]
                },
                vec![
                    sliding_attention_mask,
                    full_attention_mask,
                    s.clone(),
                    span_start,
                ],
            );
            let [sliding_attention_mask, full_attention_mask]: [Var; 2] = masks.try_into().unwrap();
            (sliding_attention_mask, full_attention_mask)
        } else {
            (sliding_attention_mask, full_attention_mask)
        };

        let (sliding_cos, sliding_sin) = rope_tables_default(
            builder,
            self.config.rope_parameters.sliding_attention.rope_theta,
            max_positions.clone(),
            self.config.head_dim,
            1.0,
        );
        let (full_cos, full_sin) = rope_tables_proportional(
            builder,
            self.config.rope_parameters.full_attention.rope_theta,
            max_positions,
            self.config.global_head_dim,
            self.config.full_partial_rotary_factor(),
        );

        let mut sliding_cache = KVCache::init(
            builder,
            self.sliding_cache_layers,
            in_sliding_k,
            in_sliding_v,
        );
        let mut full_cache = KVCache::init(builder, self.full_cache_layers, in_full_k, in_full_v);

        let per_layer_inputs = self.project_per_layer_inputs(
            builder,
            language_root.clone(),
            inputs_embeds.clone(),
            per_layer_inputs,
        );
        let mut x = inputs_embeds;
        for layer_id in 0..self.config.num_hidden_layers {
            let is_sliding = self.config.is_sliding_attention_layer(layer_id);
            let attention_mask = if is_sliding {
                sliding_attention_mask.clone()
            } else {
                full_attention_mask.clone()
            };
            let (pos, cos, sin) = if is_sliding {
                (
                    sliding_pos.clone(),
                    sliding_cos.clone(),
                    sliding_sin.clone(),
                )
            } else {
                (full_pos.clone(), full_cos.clone(), full_sin.clone())
            };
            let per_layer_input = per_layer_inputs
                .as_ref()
                .map(|inputs| self.layer_per_layer_input(builder, layer_id, inputs.clone()));
            x = self.layer(
                builder,
                layer_id,
                attention_mask,
                per_layer_input,
                &mut sliding_cache,
                &mut full_cache,
                pos,
                cos,
                sin,
                language_root
                    .clone()
                    .extend(["layers", &layer_id.to_string()])
                    .unwrap(),
                x,
            );
        }

        x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            language_root.extend(["norm"]).unwrap(),
            x,
        );

        let lm_head_weights = if self.config.tie_word_embeddings {
            language_root.extend(["embed_tokens"]).unwrap()
        } else {
            p.extend(["lm_head"]).unwrap()
        };
        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            lm_head_weights,
            x,
        );

        if let Some(softcap) = self.config.final_logit_softcapping {
            let s = constant(builder, softcap, &shape(builder, x.clone()));
            let s = cast(builder, s, dtype(builder, x.clone()));
            x = x / s.clone();
            x = tanh(builder, x);
            x = x * s;
        }

        x = argmax(builder, x);
        let (out_sliding_k, out_sliding_v) = sliding_cache.output(builder, s.clone());
        let (out_full_k, out_full_v) = full_cache.output(builder, s);
        vec![x, out_sliding_k, out_sliding_v, out_full_k, out_full_v]
    }

    pub fn forward(
        &self,
        builder: &Builder,
        p: Path,
        language_root: Path,
        x: Var,
        in_sliding_k: Var,
        in_sliding_v: Var,
        in_full_k: Var,
        in_full_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let inputs_embeds = self.scaled_embeddings(
            builder,
            language_root.extend(["embed_tokens"]).unwrap(),
            x.clone(),
            (self.config.hidden_size as f32).sqrt(),
        );
        let per_layer_inputs = self.get_per_layer_inputs(builder, language_root.clone(), x);
        let [_, s, _] = unpack::<3>(builder, shape(builder, inputs_embeds.clone()));
        let [_, _, _, full_pos, _] = unpack::<5>(builder, shape(builder, in_full_k.clone()));
        let full_attention_mask = causal_mask(builder, s, full_pos);

        self.forward_embeddings(
            builder,
            p,
            language_root,
            full_attention_mask,
            None,
            inputs_embeds,
            per_layer_inputs,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        )
    }
}

pub struct Gemma4UnifiedVisionEmbeddings {
    vision_config: Gemma4UnifiedVisionConfig,
    runtime_vision: Gemma4UnifiedRuntimeVisionConfig,
    text_hidden_size: usize,
}

impl Gemma4UnifiedVisionEmbeddings {
    fn patch_embedder(&self, builder: &Builder, pixel_values_and_positions: Var) -> Var {
        let patch_dim =
            3 * self.vision_config.model_patch_size * self.vision_config.model_patch_size;
        let pixels = slice(builder, 2, 0, patch_dim, pixel_values_and_positions.clone());
        let hidden_states = layernorm(
            builder,
            self.vision_config.rms_norm_eps,
            path(vec!["model", "vision_embedder", "patch_ln1"]).unwrap(),
            pixels,
        );
        let hidden_states = linear_b(
            builder,
            patch_dim,
            self.vision_config.mm_embed_dim,
            true,
            path(vec!["model", "vision_embedder", "patch_dense"]).unwrap(),
            hidden_states,
        );
        let mut hidden_states = layernorm(
            builder,
            self.vision_config.rms_norm_eps,
            path(vec!["model", "vision_embedder", "patch_ln2"]).unwrap(),
            hidden_states,
        );

        let [b, s, _] = unpack::<3>(builder, shape(builder, hidden_states.clone()));
        let x_positions = slice(builder, 2, patch_dim, 1, pixel_values_and_positions.clone());
        let x_positions = squeeze::<3, 2>(builder, 2, x_positions);
        let x_positions = squeeze::<2, 1>(builder, 0, x_positions);
        let x_positions = cast(builder, x_positions, Dtype::U32);

        let y_positions = slice(builder, 2, patch_dim + 1, 1, pixel_values_and_positions);
        let y_positions = squeeze::<3, 2>(builder, 2, y_positions);
        let y_positions = squeeze::<2, 1>(builder, 0, y_positions);
        let y_positions = cast(builder, y_positions, Dtype::U32);

        let position_embedding = param(
            builder,
            &path(vec!["model", "vision_embedder", "pos_embedding"]).unwrap(),
        );
        let x_table = squeeze::<3, 2>(
            builder,
            1,
            slice(builder, 1, 0, 1, position_embedding.clone()),
        );
        let y_table = squeeze::<3, 2>(builder, 1, slice(builder, 1, 1, 1, position_embedding));
        let x_pos = index(builder, 0, x_positions, x_table);
        let y_pos = index(builder, 0, y_positions, y_table);
        let pos = reshape(
            builder,
            shape!(builder, 1, s, self.vision_config.mm_embed_dim),
            x_pos + y_pos,
        );
        hidden_states = hidden_states
            + broadcast(
                builder,
                shape!(builder, b, s, self.vision_config.mm_embed_dim),
                pos,
            );

        layernorm(
            builder,
            self.vision_config.rms_norm_eps,
            path(vec!["model", "vision_embedder", "pos_norm"]).unwrap(),
            hidden_states,
        )
    }
}

impl DynModule for Gemma4UnifiedVisionEmbeddings {
    fn path(&self) -> Path {
        path(vec!["Gemma4UnifiedVisionEmbeddings"]).unwrap()
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;
        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [pixel_values_and_positions]: [Var; 1] = args.try_into().expect("expected 1 input");
        let x = self.patch_embedder(builder, pixel_values_and_positions);
        let x = rmsnorm_raw::<3>(builder, self.vision_config.rms_norm_eps, x);
        let x = linear_no_bias(
            builder,
            self.vision_config.output_proj_dims,
            self.text_hidden_size,
            path(vec!["model", "embed_vision", "embedding_projection"]).unwrap(),
            x,
        );
        let x = slice(
            builder,
            1,
            0,
            self.runtime_vision.num_soft_tokens_per_image,
            x,
        );
        vec![x]
    }
}

pub struct Gemma4UnifiedAudioEmbeddings {
    audio_config: Gemma4UnifiedAudioConfig,
    runtime_audio: Gemma4UnifiedRuntimeAudioConfig,
    text_hidden_size: usize,
}

impl DynModule for Gemma4UnifiedAudioEmbeddings {
    fn path(&self) -> Path {
        path(vec!["Gemma4UnifiedAudioEmbeddings"]).unwrap()
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;
        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone(), t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [features, _mask]: [Var; 2] = args.try_into().expect("expected 2 inputs");
        let x = rmsnorm_raw::<3>(builder, self.audio_config.rms_norm_eps, features);
        let x = linear_no_bias(
            builder,
            self.audio_config.audio_samples_per_token,
            self.text_hidden_size,
            path(vec!["model", "embed_audio", "embedding_projection"]).unwrap(),
            x,
        );
        let x = slice(
            builder,
            1,
            0,
            self.runtime_audio.num_soft_tokens_per_audio,
            x,
        );
        vec![x]
    }
}

pub struct Gemma4UnifiedMultimodalModel {
    language_model: Gemma4UnifiedModel,
}

impl Gemma4UnifiedMultimodalModel {
    fn bidirectional_mask(
        builder: &Builder,
        size: Var,
        modality_start: Var,
        modality_size: usize,
    ) -> Var {
        let row = arange(builder, size.clone());
        let sh = shape(builder, row.clone());

        let modality_end = modality_start.clone() + modality_size.to_nat(builder);

        let modality_start = nat_to_u32(builder, modality_start);
        let modality_start = broadcast(builder, sh.clone(), modality_start);

        let modality_end = nat_to_u32(builder, modality_end);
        let modality_end = broadcast(builder, sh, modality_end);

        let modality_mask_1 = gte(builder, row.clone(), modality_start);
        let modality_mask_2 = lt(builder, row, modality_end);
        let row = modality_mask_1 * modality_mask_2;

        let sh = pack::<2>(builder, [size.clone(), size]);
        let row = broadcast(builder, sh.clone(), row);
        let col = transpose(builder, 0, 1, row.clone());
        let mask = row * col;
        let mask = cast(builder, mask, Dtype::F32);
        let one = constant(builder, 1.0, &sh);
        one - mask
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_modality_and_texts(
        &self,
        builder: &Builder,
        text_before: Var,
        modality: Var,
        text_after: Var,
        in_sliding_k: Var,
        in_sliding_v: Var,
        in_full_k: Var,
        in_full_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let p = Path::empty();
        let language_root = path(vec!["model", "language_model"]).unwrap();
        let embed_tokens = language_root.extend(["embed_tokens"]).unwrap();
        let scale = (self.language_model.config.hidden_size as f32).sqrt();

        let text_before_embeds = self.language_model.scaled_embeddings(
            builder,
            embed_tokens.clone(),
            text_before.clone(),
            scale,
        );
        let text_after_embeds =
            self.language_model
                .scaled_embeddings(builder, embed_tokens, text_after.clone(), scale);
        let [_b, modality_start, _d] =
            unpack::<3>(builder, shape(builder, text_before_embeds.clone()));
        let inputs_embeds = concat(builder, 1, text_before_embeds, modality.clone());
        let inputs_embeds = concat(builder, 1, inputs_embeds, text_after_embeds);
        let bidirectional_span = match self.language_model.multimodal.as_ref() {
            Some(Gemma4UnifiedMultimodalConfig::Vision { runtime_vision, .. }) => {
                Some((modality_start, runtime_vision.num_soft_tokens_per_image))
            }
            _ => None,
        };

        let per_layer_inputs = if self.language_model.config.hidden_size_per_layer_input == 0 {
            None
        } else {
            let [b, modality_len, _] = unpack::<3>(builder, shape(builder, modality));
            let pad_ids = constant(
                builder,
                self.language_model.config.pad_token_id as u32,
                &shape!(builder, b, modality_len),
            );
            let text_before_pli = self.language_model.get_per_layer_inputs(
                builder,
                language_root.clone(),
                text_before,
            );
            let modality_pli =
                self.language_model
                    .get_per_layer_inputs(builder, language_root.clone(), pad_ids);
            let text_after_pli = self.language_model.get_per_layer_inputs(
                builder,
                language_root.clone(),
                text_after,
            );
            Some(concat(
                builder,
                1,
                concat(
                    builder,
                    1,
                    text_before_pli.expect("gemma4 unified text-before per-layer inputs missing"),
                    modality_pli.expect("gemma4 unified modality per-layer inputs missing"),
                ),
                text_after_pli.expect("gemma4 unified text-after per-layer inputs missing"),
            ))
        };

        let [_, s, _] = unpack::<3>(builder, shape(builder, inputs_embeds.clone()));
        let [_, _, _, full_pos, _] = unpack::<5>(builder, shape(builder, in_full_k.clone()));
        let full_attention_mask = causal_mask(builder, s, full_pos);

        self.language_model.forward_embeddings(
            builder,
            p,
            language_root,
            full_attention_mask,
            bidirectional_span,
            inputs_embeds,
            per_layer_inputs,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        )
    }
}

impl DynModule for Gemma4UnifiedMultimodalModel {
    fn path(&self) -> Path {
        path(vec!["Gemma4UnifiedVLM"]).unwrap()
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
                t.clone(),
                Type::Nat(NatExpr::Var(4)),
            ],
            vec![t.clone(), t.clone(), t.clone(), t.clone(), t],
        )
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [
            text_before,
            modality,
            text_after,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        ]: [Var; 8] = args.try_into().expect("expected 8 inputs");
        self.forward_modality_and_texts(
            builder,
            text_before,
            modality,
            text_after,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        )
    }
}

impl DynModule for Gemma4UnifiedModel {
    fn path(&self) -> Path {
        path(vec!["gemma4_unified"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [
            x,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        ]: [Var; 6] = args.try_into().expect("expected 6 inputs");
        let p = self.path();
        let language_root = if self.root.is_empty() {
            p.clone()
        } else {
            p.extend(self.root.split('.')).unwrap()
        };
        self.forward(
            builder,
            p,
            language_root,
            x,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        )
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let batch_size = NatExpr::Var(0);
        let seq_len = NatExpr::Var(1);
        let sliding_cache_len = NatExpr::Var(2);
        let full_cache_len = NatExpr::Var(3);
        let max_positions = NatExpr::Var(4);

        let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len.clone()]),
        }));
        let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![batch_size.clone(), NatExpr::Constant(1)]),
        }));

        let sliding_heads = NatExpr::Constant(self.config.num_key_value_heads);
        let full_heads = NatExpr::Constant(self.config.full_num_key_value_heads());
        let sliding_layers = NatExpr::Constant(self.sliding_cache_layers);
        let full_layers = NatExpr::Constant(self.full_cache_layers);
        let sliding_dim = NatExpr::Constant(self.config.head_dim);
        let full_dim = NatExpr::Constant(self.config.global_head_dim);
        let sliding_out_len = NatExpr::Add(vec![sliding_cache_len.clone(), seq_len.clone()]);
        let full_out_len = NatExpr::Add(vec![full_cache_len.clone(), seq_len]);

        let t_sliding_k = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                sliding_layers.clone(),
                batch_size.clone(),
                sliding_heads.clone(),
                sliding_cache_len.clone(),
                sliding_dim.clone(),
            ]),
        }));
        let t_sliding_v = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                sliding_layers.clone(),
                batch_size.clone(),
                sliding_heads.clone(),
                sliding_cache_len,
                sliding_dim,
            ]),
        }));
        let t_full_k = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                full_layers.clone(),
                batch_size.clone(),
                full_heads.clone(),
                full_cache_len.clone(),
                full_dim.clone(),
            ]),
        }));
        let t_full_v = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                full_layers.clone(),
                batch_size.clone(),
                full_heads.clone(),
                full_cache_len,
                full_dim,
            ]),
        }));

        let t_sliding_k_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                sliding_layers,
                batch_size.clone(),
                sliding_heads.clone(),
                sliding_out_len.clone(),
                NatExpr::Constant(self.config.head_dim),
            ]),
        }));
        let t_sliding_v_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                NatExpr::Constant(self.sliding_cache_layers),
                batch_size.clone(),
                sliding_heads,
                sliding_out_len,
                NatExpr::Constant(self.config.head_dim),
            ]),
        }));
        let t_full_k_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                full_layers,
                batch_size.clone(),
                full_heads.clone(),
                full_out_len.clone(),
                NatExpr::Constant(self.config.global_head_dim),
            ]),
        }));
        let t_full_v_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                NatExpr::Constant(self.full_cache_layers),
                batch_size,
                full_heads,
                full_out_len,
                NatExpr::Constant(self.config.global_head_dim),
            ]),
        }));

        (
            vec![
                t_x,
                t_sliding_k,
                t_sliding_v,
                t_full_k,
                t_full_v,
                Type::Nat(max_positions),
            ],
            vec![
                t_y,
                t_sliding_k_out,
                t_sliding_v_out,
                t_full_k_out,
                t_full_v_out,
            ],
        )
    }
}
