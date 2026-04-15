#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use crate::utils::load_and_patchify_dynamic_image;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct Gemma4Config {
    text_config: Gemma4TextConfig,
    vision_config: Option<Gemma4VisionConfig>,
    image_token_id: Option<usize>,
    vision_soft_tokens_per_image: Option<usize>,
    eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4RopeTypeConfig {
    rope_theta: f32,
    partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma4RopeParameters {
    full_attention: Gemma4RopeTypeConfig,
    sliding_attention: Gemma4RopeTypeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
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
    rope_parameters: Gemma4RopeParameters,
    pad_token_id: usize,
    eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionRopeParameters {
    rope_theta: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
    position_embedding_size: usize,
    rms_norm_eps: f32,
    use_clipped_linears: bool,
    rope_parameters: Gemma4VisionRopeParameters,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Gemma4RuntimeVisionConfig {
    pub patch_grid_height: usize,
    pub patch_grid_width: usize,
    pub num_soft_tokens_per_image: usize,
}

#[derive(Debug, Clone)]
pub struct Gemma4PreparedImageInput {
    pub patches: Vec<f32>,
    pub shape: Vec<usize>,
    pub runtime_vision: Gemma4RuntimeVisionConfig,
}

impl Gemma4TextConfig {
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
            .expect("gemma4 full_attention partial_rotary_factor missing")
    }
}

impl LLMConfig for Gemma4TextConfig {
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

pub fn prepare_gemma4_image_input(
    image: &image::DynamicImage,
    config_json: &serde_json::Value,
) -> crate::Result<Gemma4PreparedImageInput> {
    let config: Gemma4Config = serde_json::from_value(config_json.clone())?;
    let vision_config = config.vision_config.ok_or_else(|| {
        crate::LLMError::InvalidModelConfig("gemma4 missing vision_config".to_string())
    })?;
    let max_soft_tokens = config.vision_soft_tokens_per_image.ok_or_else(|| {
        crate::LLMError::InvalidModelConfig(
            "gemma4 missing vision_soft_tokens_per_image".to_string(),
        )
    })?;
    let patched = load_and_patchify_dynamic_image(
        image,
        vision_config.patch_size,
        max_soft_tokens,
        vision_config.pooling_kernel_size,
    )?;
    let num_patches = patched.patch_grid_height * patched.patch_grid_width;
    let pooling_area = vision_config.pooling_kernel_size * vision_config.pooling_kernel_size;
    if !num_patches.is_multiple_of(pooling_area) {
        return Err(crate::LLMError::InvalidModelConfig(format!(
            "gemma4 image patch count {num_patches} was not divisible by pooling area {pooling_area}"
        )));
    }
    Ok(Gemma4PreparedImageInput {
        patches: patched.data,
        shape: patched.shape,
        runtime_vision: Gemma4RuntimeVisionConfig {
            patch_grid_height: patched.patch_grid_height,
            patch_grid_width: patched.patch_grid_width,
            num_soft_tokens_per_image: num_patches / pooling_area,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4AttentionKind {
    Sliding,
    Full,
}

#[derive(Debug, Clone, Copy)]
enum Gemma4CacheSource {
    Own(usize),
    Shared(usize),
}

#[derive(Debug, Clone, Copy)]
struct Gemma4LayerPlan {
    kind: Gemma4AttentionKind,
    cache: Gemma4CacheSource,
}

#[derive(Debug, Clone)]
struct Gemma4MultimodalConfig {
    vision_config: Gemma4VisionConfig,
    image_token_index: usize,
    runtime_vision: Gemma4RuntimeVisionConfig,
}

#[derive(Debug, Clone)]
pub struct Gemma4Model {
    root: String,
    config: Gemma4TextConfig,
    dtype: Dtype,
    layer_plans: Vec<Gemma4LayerPlan>,
    sliding_cache_layers: usize,
    full_cache_layers: usize,
    multimodal: Option<Gemma4MultimodalConfig>,
}

impl LLMModel for Gemma4Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let dtype = self.dtype();
        vec![
            (
                dtype.clone(),
                Shape(vec![
                    self.sliding_cache_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.head_dim,
                ]),
            ),
            (
                dtype.clone(),
                Shape(vec![
                    self.sliding_cache_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.head_dim,
                ]),
            ),
            (
                dtype.clone(),
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

    fn multimodal_metadata(&self) -> Option<MultimodalMetadata> {
        let mm = self.multimodal.as_ref()?;
        Some(MultimodalMetadata {
            image_token_index: mm.image_token_index,
            mm_tokens_per_image: mm.runtime_vision.num_soft_tokens_per_image,
            hidden_size: self.config.hidden_size,
            image_size: mm
                .runtime_vision
                .patch_grid_height
                .max(mm.runtime_vision.patch_grid_width)
                * mm.vision_config.patch_size,
            patch_size: mm.vision_config.patch_size,
        })
    }

    fn multimodal_vision_module(&self) -> Option<Box<dyn DynModule>> {
        let mm = self.multimodal.as_ref()?;
        Some(Box::new(Gemma4VisionEmbeddings {
            vision_config: mm.vision_config.clone(),
            runtime_vision: mm.runtime_vision.clone(),
            text_hidden_size: self.config.hidden_size,
        }))
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        self.multimodal.as_ref()?;
        Some(Box::new(Gemma4MultimodalModel {
            language_model: self.clone(),
        }))
    }

    fn multimodal_interpolate_prompt(&self, prompt: &str) -> Option<String> {
        let mm = self.multimodal.as_ref()?;
        Some(prompt.replace(
            "<|image|>",
            &format!(
                "{}{}{}",
                "<|image>",
                "<|image|>".repeat(mm.runtime_vision.num_soft_tokens_per_image),
                "<image|>"
            ),
        ))
    }
}

impl Gemma4Model {
    pub fn new(
        root: &str,
        config_json: &serde_json::Value,
        runtime_vision: Option<&Gemma4RuntimeVisionConfig>,
        dtype: Dtype,
    ) -> crate::Result<Self> {
        let Gemma4Config {
            mut text_config,
            vision_config,
            image_token_id,
            eos_token_id,
            ..
        }: Gemma4Config = serde_json::from_value(config_json.clone())?;

        if let Some(eos_token_id) = eos_token_id {
            text_config.eos_token_id = Some(eos_token_id);
        }

        let multimodal = match (vision_config, image_token_id, runtime_vision) {
            (Some(vision_config), Some(image_token_index), Some(runtime_vision)) => {
                Some(Gemma4MultimodalConfig {
                    vision_config,
                    image_token_index,
                    runtime_vision: runtime_vision.clone(),
                })
            }
            _ => None,
        };

        let first_shared_layer = text_config.num_hidden_layers - text_config.num_kv_shared_layers;
        let mut layer_plans = Vec::with_capacity(text_config.num_hidden_layers);
        let mut sliding_cache_layers = 0;
        let mut full_cache_layers = 0;
        let mut last_sliding_cache_id = None;
        let mut last_full_cache_id = None;

        for layer_id in 0..text_config.num_hidden_layers {
            let kind = if text_config.is_sliding_attention_layer(layer_id) {
                Gemma4AttentionKind::Sliding
            } else {
                Gemma4AttentionKind::Full
            };

            let cache = if layer_id < first_shared_layer {
                match kind {
                    Gemma4AttentionKind::Sliding => {
                        let cache_id = sliding_cache_layers;
                        sliding_cache_layers += 1;
                        last_sliding_cache_id = Some(cache_id);
                        Gemma4CacheSource::Own(cache_id)
                    }
                    Gemma4AttentionKind::Full => {
                        let cache_id = full_cache_layers;
                        full_cache_layers += 1;
                        last_full_cache_id = Some(cache_id);
                        Gemma4CacheSource::Own(cache_id)
                    }
                }
            } else {
                match kind {
                    Gemma4AttentionKind::Sliding => {
                        Gemma4CacheSource::Shared(last_sliding_cache_id.ok_or_else(|| {
                            crate::LLMError::InvalidModelConfig(
                                "gemma4 sliding shared KV layer had no source layer".to_string(),
                            )
                        })?)
                    }
                    Gemma4AttentionKind::Full => {
                        Gemma4CacheSource::Shared(last_full_cache_id.ok_or_else(|| {
                            crate::LLMError::InvalidModelConfig(
                                "gemma4 full shared KV layer had no source layer".to_string(),
                            )
                        })?)
                    }
                }
            };

            layer_plans.push(Gemma4LayerPlan { kind, cache });
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
            .expect("gemma4 enable_moe_block missing num_experts");
        let top_k_experts = self
            .config
            .top_k_experts
            .expect("gemma4 enable_moe_block missing top_k_experts");
        let moe_intermediate_size = self
            .config
            .moe_intermediate_size
            .expect("gemma4 enable_moe_block missing moe_intermediate_size");
        let [batch_size, seq_len, hidden_size] = unpack::<3>(builder, shape(builder, x.clone()));
        let routed = rmsnorm_raw::<3>(builder, self.config.rms_norm_eps, x.clone());
        let routed_scale = param(builder, &p.extend(["router", "scale"]).unwrap());
        let routed_scale = broadcast(builder, shape(builder, routed.clone()), routed_scale);
        let routed_scalar = constant(
            builder,
            (self.config.hidden_size as f32).powf(-0.5),
            &shape(builder, routed.clone()),
        );
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
            Gemma4AttentionKind::Sliding => (
                sliding_cache,
                match plan.cache {
                    Gemma4CacheSource::Own(cache_id) | Gemma4CacheSource::Shared(cache_id) => {
                        cache_id
                    }
                },
            ),
            Gemma4AttentionKind::Full => (
                full_cache,
                match plan.cache {
                    Gemma4CacheSource::Own(cache_id) | Gemma4CacheSource::Shared(cache_id) => {
                        cache_id
                    }
                },
            ),
        };
        let (k, v) = match plan.cache {
            Gemma4CacheSource::Own(_) => cache.update(builder, cache_id, k, v),
            Gemma4CacheSource::Shared(_) => cache.get(cache_id),
        };

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let mut attn = matmul(builder, q, transpose(builder, 2, 3, k));
        let sh = shape(builder, attn.clone());
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

fn clippable_linear_no_bias(
    builder: &Builder,
    use_clipped_linears: bool,
    in_features: usize,
    out_features: usize,
    p: Path,
    x: Var,
) -> Var {
    let x = if use_clipped_linears {
        let sh = shape(builder, x.clone());
        let input_min = broadcast(
            builder,
            sh.clone(),
            param(builder, &p.extend(["input_min"]).unwrap()),
        );
        let input_max = broadcast(
            builder,
            sh,
            param(builder, &p.extend(["input_max"]).unwrap()),
        );
        clamp_with_tensors(builder, x, input_min, input_max)
    } else {
        x
    };
    let x = linear_no_bias(
        builder,
        in_features,
        out_features,
        p.extend(["linear"]).unwrap(),
        x,
    );
    if use_clipped_linears {
        let sh = shape(builder, x.clone());
        let output_min = broadcast(
            builder,
            sh.clone(),
            param(builder, &p.extend(["output_min"]).unwrap()),
        );
        let output_max = broadcast(
            builder,
            sh,
            param(builder, &p.extend(["output_max"]).unwrap()),
        );
        clamp_with_tensors(builder, x, output_min, output_max)
    } else {
        x
    }
}

fn gemma4_vision_positions(
    builder: &Builder,
    patch_grid_height: usize,
    patch_grid_width: usize,
) -> (Var, Var) {
    let tokens = patch_grid_height * patch_grid_width;
    let idx = cast(builder, arange(builder, tokens), Dtype::F32);
    let sh = shape(builder, idx.clone());
    let width = constant(builder, patch_grid_width as f32, &sh);
    let row = floor(builder, idx.clone() / width.clone());
    let col = idx - row.clone() * width;
    (
        cast(builder, col, Dtype::U32),
        cast(builder, row, Dtype::U32),
    )
}

fn gemma4_apply_2d_rope(
    builder: &Builder,
    col_positions: Var,
    row_positions: Var,
    head_dim: usize,
    cos_x: Var,
    sin_x: Var,
    cos_y: Var,
    sin_y: Var,
    x: Var,
) -> Var {
    let [x_part, y_part] = chunk(builder, 3, 2, head_dim / 2, x).try_into().unwrap();
    let x_part =
        apply_rope_embedding_positions(builder, col_positions, head_dim / 2, cos_x, sin_x, x_part);
    let y_part =
        apply_rope_embedding_positions(builder, row_positions, head_dim / 2, cos_y, sin_y, y_part);
    concat(builder, 3, x_part, y_part)
}

pub struct Gemma4VisionEmbeddings {
    vision_config: Gemma4VisionConfig,
    runtime_vision: Gemma4RuntimeVisionConfig,
    text_hidden_size: usize,
}

impl Gemma4VisionEmbeddings {
    fn patch_embedder(&self, builder: &Builder, pixels: Var) -> Var {
        debug_assert!(
            self.runtime_vision.patch_grid_height <= self.vision_config.position_embedding_size
        );
        debug_assert!(
            self.runtime_vision.patch_grid_width <= self.vision_config.position_embedding_size
        );
        let [b, s, _] = unpack::<3>(builder, shape(builder, pixels.clone()));
        let pixels = pixels.clone() * constant(builder, 2.0f32, &shape(builder, pixels.clone()))
            - constant(builder, 1.0f32, &shape(builder, pixels));
        let hidden_states = linear_no_bias(
            builder,
            3 * self.vision_config.patch_size * self.vision_config.patch_size,
            self.vision_config.hidden_size,
            path(vec![
                "model",
                "vision_tower",
                "patch_embedder",
                "input_proj",
            ])
            .unwrap(),
            pixels,
        );

        let (col_positions, row_positions) = gemma4_vision_positions(
            builder,
            self.runtime_vision.patch_grid_height,
            self.runtime_vision.patch_grid_width,
        );
        let position_embedding_table = param(
            builder,
            &path(vec![
                "model",
                "vision_tower",
                "patch_embedder",
                "position_embedding_table",
            ])
            .unwrap(),
        );
        let x_table = squeeze::<3, 2>(
            builder,
            0,
            slice(builder, 0, 0, 1, position_embedding_table.clone()),
        );
        let y_table = squeeze::<3, 2>(
            builder,
            0,
            slice(builder, 0, 1, 1, position_embedding_table),
        );
        let x_pos = index(builder, 0, col_positions, x_table);
        let y_pos = index(builder, 0, row_positions, y_table);
        let pos = reshape(
            builder,
            shape!(builder, 1, s, self.vision_config.hidden_size),
            x_pos + y_pos,
        );
        hidden_states
            + broadcast(
                builder,
                shape!(builder, b, s, self.vision_config.hidden_size),
                pos,
            )
    }

    fn attention(
        &self,
        builder: &Builder,
        attention_mask: Var,
        col_positions: Var,
        row_positions: Var,
        cos_x: Var,
        sin_x: Var,
        cos_y: Var,
        sin_y: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let num_heads = self.vision_config.num_attention_heads;
        let num_kv_heads = self.vision_config.num_key_value_heads;
        let head_dim = self.vision_config.head_dim;
        let rep = num_heads / num_kv_heads;
        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            self.vision_config.hidden_size,
            num_heads * head_dim,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let k = clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            self.vision_config.hidden_size,
            num_kv_heads * head_dim,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            self.vision_config.hidden_size,
            num_kv_heads * head_dim,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let q = reshape(builder, shape!(builder, b, s, num_heads, head_dim), q);
        let k = reshape(builder, shape!(builder, b, s, num_kv_heads, head_dim), k);
        let v = reshape(builder, shape!(builder, b, s, num_kv_heads, head_dim), v);

        let q = rmsnorm::<4>(
            builder,
            self.vision_config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm::<4>(
            builder,
            self.vision_config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );
        let v = rmsnorm_raw::<4>(builder, self.vision_config.rms_norm_eps, v);

        let q = gemma4_apply_2d_rope(
            builder,
            col_positions.clone(),
            row_positions.clone(),
            head_dim,
            cos_x.clone(),
            sin_x.clone(),
            cos_y.clone(),
            sin_y.clone(),
            q,
        );
        let k = gemma4_apply_2d_rope(
            builder,
            col_positions,
            row_positions,
            head_dim,
            cos_x,
            sin_x,
            cos_y,
            sin_y,
            k,
        );

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);
        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let mut attn = matmul(builder, q, transpose(builder, 2, 3, k));
        let sh = shape(builder, attn.clone());
        attn = attn + broadcast(builder, sh, attention_mask);
        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let attn = transpose(builder, 1, 2, attn);
        let attn = reshape(builder, shape!(builder, b, s, num_heads * head_dim), attn);

        clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            num_heads * head_dim,
            self.vision_config.hidden_size,
            p.extend(["o_proj"]).unwrap(),
            attn,
        )
    }

    fn layer(
        &self,
        builder: &Builder,
        attention_mask: Var,
        col_positions: Var,
        row_positions: Var,
        cos_x: Var,
        sin_x: Var,
        cos_y: Var,
        sin_y: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let residual = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.vision_config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = self.attention(
            builder,
            attention_mask,
            col_positions,
            row_positions,
            cos_x,
            sin_x,
            cos_y,
            sin_y,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = rmsnorm::<3>(
            builder,
            self.vision_config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = residual + x;

        let residual = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.vision_config.rms_norm_eps,
            p.extend(["pre_feedforward_layernorm"]).unwrap(),
            x,
        );
        let gate = clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            self.vision_config.hidden_size,
            self.vision_config.intermediate_size,
            p.extend(["mlp", "gate_proj"]).unwrap(),
            x.clone(),
        );
        let up = clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            self.vision_config.hidden_size,
            self.vision_config.intermediate_size,
            p.extend(["mlp", "up_proj"]).unwrap(),
            x,
        );
        let x = gelu(builder, gate) * up;
        let x = clippable_linear_no_bias(
            builder,
            self.vision_config.use_clipped_linears,
            self.vision_config.intermediate_size,
            self.vision_config.hidden_size,
            p.extend(["mlp", "down_proj"]).unwrap(),
            x,
        );
        let x = rmsnorm::<3>(
            builder,
            self.vision_config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm"]).unwrap(),
            x,
        );
        residual + x
    }

    fn vision_model(&self, builder: &Builder, pixels: Var) -> Var {
        let mut x = self.patch_embedder(builder, pixels);
        let [_, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = constant(builder, 0.0f32, &shape!(builder, s.clone(), s));
        let (col_positions, row_positions) = gemma4_vision_positions(
            builder,
            self.runtime_vision.patch_grid_height,
            self.runtime_vision.patch_grid_width,
        );
        let (cos_x, sin_x) = rope_tables_default(
            builder,
            self.vision_config.rope_parameters.rope_theta,
            self.runtime_vision.patch_grid_width,
            self.vision_config.head_dim / 2,
            1.0,
        );
        let (cos_y, sin_y) = rope_tables_default(
            builder,
            self.vision_config.rope_parameters.rope_theta,
            self.runtime_vision.patch_grid_height,
            self.vision_config.head_dim / 2,
            1.0,
        );

        for layer_id in 0..self.vision_config.num_hidden_layers {
            let layer_path = path(vec!["model", "vision_tower", "encoder", "layers"])
                .unwrap()
                .extend([&layer_id.to_string()])
                .unwrap();
            x = self.layer(
                builder,
                attention_mask.clone(),
                col_positions.clone(),
                row_positions.clone(),
                cos_x.clone(),
                sin_x.clone(),
                cos_y.clone(),
                sin_y.clone(),
                layer_path,
                x,
            );
        }

        let x = transpose(builder, 1, 2, x);
        let x = reshape(
            builder,
            shape!(
                builder,
                1,
                self.vision_config.hidden_size,
                self.runtime_vision.patch_grid_height,
                self.runtime_vision.patch_grid_width
            ),
            x,
        );
        let x = avgpool2d_rect(
            builder,
            self.vision_config.hidden_size,
            self.runtime_vision.patch_grid_height,
            self.runtime_vision.patch_grid_width,
            self.vision_config.pooling_kernel_size,
            x,
        );
        let x = reshape(
            builder,
            shape!(
                builder,
                1,
                self.vision_config.hidden_size,
                self.runtime_vision.num_soft_tokens_per_image
            ),
            x,
        );
        let x = transpose(builder, 1, 2, x);
        let scale = constant(
            builder,
            (self.vision_config.hidden_size as f32).sqrt(),
            &shape(builder, x.clone()),
        );
        x * scale
    }
}

impl DynModule for Gemma4VisionEmbeddings {
    fn path(&self) -> Path {
        path(vec!["Gemma4VisionEmbeddings"]).unwrap()
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;
        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [pixels]: [Var; 1] = args.try_into().expect("expected 1 input");
        let x = self.vision_model(builder, pixels);
        let x = rmsnorm_raw::<3>(builder, self.vision_config.rms_norm_eps, x);
        let x = linear_no_bias(
            builder,
            self.vision_config.hidden_size,
            self.text_hidden_size,
            path(vec!["model", "embed_vision", "embedding_projection"]).unwrap(),
            x,
        );
        vec![x]
    }
}

pub struct Gemma4MultimodalModel {
    language_model: Gemma4Model,
}

impl Gemma4MultimodalModel {
    #[allow(clippy::too_many_arguments)]
    fn forward_image_and_texts(
        &self,
        builder: &Builder,
        text_before: Var,
        image: Var,
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
        let inputs_embeds = concat(builder, 1, text_before_embeds, image.clone());
        let inputs_embeds = concat(builder, 1, inputs_embeds, text_after_embeds);

        let per_layer_inputs = if self.language_model.config.hidden_size_per_layer_input == 0 {
            None
        } else {
            let [b, image_len, _] = unpack::<3>(builder, shape(builder, image));
            let pad_ids = constant(
                builder,
                self.language_model.config.pad_token_id as u32,
                &shape!(builder, b, image_len),
            );
            let text_before_pli = self.language_model.get_per_layer_inputs(
                builder,
                language_root.clone(),
                text_before,
            );
            let image_pli =
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
                    text_before_pli.expect("gemma4 text-before per-layer inputs missing"),
                    image_pli.expect("gemma4 image per-layer inputs missing"),
                ),
                text_after_pli.expect("gemma4 text-after per-layer inputs missing"),
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

impl DynModule for Gemma4MultimodalModel {
    fn path(&self) -> Path {
        path(vec!["Gemma4VLM"]).unwrap()
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
            image,
            text_after,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        ]: [Var; 8] = args.try_into().expect("expected 8 inputs");
        self.forward_image_and_texts(
            builder,
            text_before,
            image,
            text_after,
            in_sliding_k,
            in_sliding_v,
            in_full_k,
            in_full_v,
            max_positions,
        )
    }
}

impl DynModule for Gemma4Model {
    fn path(&self) -> Path {
        path(vec!["gemma4"]).expect("invalid model path")
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
