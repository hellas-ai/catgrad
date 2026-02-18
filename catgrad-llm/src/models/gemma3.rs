#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig, RopeScaling};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub enum GemmaConfig {
    #[serde(untagged)]
    VLM {
        text_config: GemmaTextConfig,
        image_token_index: usize,
        #[serde(default)]
        mm_tokens_per_image: usize,
    },
    #[serde(untagged)]
    Text(GemmaTextConfig),
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LinearRopeScaling {
    pub factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GemmaTextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_sliding_window_pattern")]
    #[serde(alias = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,

    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<LinearRopeScaling>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_query_pre_attn_scalar() -> usize {
    256
}

fn default_sliding_window_pattern() -> usize {
    6
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_rope_local_base_freq() -> f32 {
    10000.0
}

fn default_rope_theta() -> f32 {
    1000000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_num_attention_heads() -> usize {
    8
}

fn default_num_key_value_heads() -> usize {
    4
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_head_dim() -> usize {
    256
}

fn default_vocab_size() -> usize {
    262208
}

impl LLMConfig for GemmaTextConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }
    fn num_local_experts(&self) -> usize {
        0
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn rope_scaling(&self) -> Option<RopeScaling> {
        None
    }
    fn partial_rotary_factor(&self) -> f32 {
        self.partial_rotary_factor
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }
    fn get_qk_head_dim(&self) -> usize {
        self.head_dim
    }
    fn get_v_head_dim(&self) -> usize {
        self.head_dim
    }
    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct Gemma3Model {
    pub root: String,
    pub config: GemmaTextConfig,
    pub max_sequence_length: usize,
}

impl LLMModel for Gemma3Model {}

// Gemma uses a non-standard RMSNorm implementation.
// Generic because of unpack needing the last dimension and it is being called
// with ranks 2 and 3 too.
fn rmsnorm_raw_gemma<const N: usize>(builder: &Builder, eps: f32, x: Var) -> Var {
    let x_shape = shape(builder, x.clone());
    let u = unpack::<N>(builder, x_shape.clone());
    let n = u[N - 1].clone();
    let s = sum(builder, x.clone() * x.clone());

    let constn = nat_to_u32(builder, n);
    let constn = cast(builder, constn, dtype(builder, x.clone()));
    let sh = shape(builder, s.clone());
    let constn = broadcast(builder, constn, sh);

    let mean = s / constn;

    let epsilon = constant(builder, eps, &shape(builder, mean.clone()));
    let rms = sqrt(builder, mean + epsilon);
    let denom = broadcast(builder, rms, x_shape);
    x / denom
}

pub fn rmsnorm_gemma<const N: usize>(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = rmsnorm_raw_gemma::<N>(builder, eps, x);
    let lr_shape = shape(builder, lr.clone());
    let gamma = broadcast(builder, gamma, lr_shape);
    let sh = shape(builder, gamma.clone());
    let one = constant(builder, 1.0, &sh);
    lr * (one + gamma)
}

/// Multi-modal projector for Gemma 3
pub fn multi_modal_projector(builder: &Builder, p: Path, x: Var) -> Var {
    // SigLIP parameters used in Gemma 3
    let hidden_size = 1152;
    let tokens_per_image = 256;
    let image_size = 896;
    let patch_size = 14;
    let patches = image_size / patch_size;

    let x = transpose(builder, 1, 2, x);
    let sh = shape!(builder, 1, hidden_size, patches, patches);
    let x = reshape(builder, sh, x);

    let x = avgpool2d(builder, hidden_size, patches, 4, x);
    let x = reshape(
        builder,
        shape!(builder, 1, hidden_size, tokens_per_image),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    let x = rmsnorm_gemma::<3>(
        builder,
        1e-6,
        p.extend(vec!["mm_soft_emb_norm"]).unwrap(),
        x,
    );
    let proj = param(
        builder,
        &p.extend(vec!["mm_input_projection_weight"]).unwrap(),
    );
    let proj = unsqueeze::<2, 3>(builder, 0, proj);
    matmul(builder, x, proj)
}

impl Gemma3Model {
    pub fn new(root: &str, config: GemmaTextConfig, max_sequence_length: usize) -> Self {
        Gemma3Model {
            root: root.to_string(),
            config,
            max_sequence_length,
        }
    }

    fn softcap(&self, builder: &Builder, softcap: f32, x: Var) -> Var {
        let sh = shape(builder, x.clone());
        let s = constant(builder, softcap, &sh);
        let x = x / s.clone();
        let x = tanh(builder, x);
        x * s
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
        let x = gelu(builder, gate) * up;
        linear_no_bias(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    fn attention(
        &self,
        builder: &Builder,
        layer_id: usize,
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
        let head_dim = self.config.head_dim;

        let is_gemma3 = self.config.model_type == "gemma3_text";
        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );

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
        let q = reshape(builder, sh, q);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let mut q = transpose(builder, 1, 2, q);
        let mut k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        // Norm
        if is_gemma3 {
            let sh = shape!(
                builder,
                b.clone() * s.clone() * num_heads.to_nat(builder),
                head_dim
            );
            q = reshape(builder, sh, q);

            let sh = shape!(
                builder,
                b.clone() * s.clone() * num_kv_heads.to_nat(builder),
                head_dim
            );

            k = reshape(builder, sh, k);

            q = rmsnorm_gemma::<2>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["q_norm"]).unwrap(),
                q,
            );
            k = rmsnorm_gemma::<2>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["k_norm"]).unwrap(),
                k,
            );
        };

        let sh = shape!(builder, b, num_heads, s, head_dim);
        let q = reshape(builder, sh, q);
        let sh = shape!(builder, b, num_kv_heads, s, head_dim);
        let k = reshape(builder, sh, k);

        // Every 6th layer of Gemma3 uses global attention, otherwise local attention, with different rope frequencies
        let is_local_attention =
            is_gemma3 && !(layer_id + 1).is_multiple_of(self.config.sliding_window_pattern);

        let theta = if is_local_attention {
            self.config.rope_local_base_freq
        } else {
            self.config.rope_theta
        };

        let factor = if is_local_attention {
            1.0
        } else {
            self.config
                .rope_scaling
                .as_ref()
                .map_or(1.0, |rs| rs.factor)
        };

        let q = rope(builder, theta, pos.clone(), &s, head_dim, factor, q);
        let k = rope(builder, theta, pos, &s, head_dim, factor, k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(
            builder,
            f32::sqrt(self.config.query_pre_attn_scalar as f32),
            &sh,
        );
        let mut attn = attn / denom;

        if let Some(softcap) = self.config.attn_logit_softcapping {
            attn = self.softcap(builder, softcap, attn);
        }

        let mask = broadcast(builder, attention_mask, sh);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, num_heads * head_dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(
            builder,
            num_heads * head_dim,
            dim,
            p.extend(["o_proj"]).unwrap(),
            attn,
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
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = self.attention(
            builder,
            layer_id,
            attention_mask,
            cache,
            pos,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = res + x;
        let res = x.clone();
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_feedforward_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm"]).unwrap(),
            x,
        );
        x + res
    }

    // Forward pass with text tokens as input
    pub fn forward(&self, builder: &Builder, p: Path, x: Var, in_k: Var, in_v: Var) -> [Var; 3] {
        let x = embeddings(builder, p.extend(vec!["embed_tokens"]).unwrap(), x);

        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s);

        self.forward_embeddings(builder, p, attention_mask, x, in_k, in_v)
    }

    // Forward pass with embeddings available.
    pub fn forward_embeddings(
        &self,
        builder: &Builder,
        p: Path,
        attention_mask: Var,
        x: Var,
        in_k: Var,
        in_v: Var,
    ) -> [Var; 3] {
        let mut cache = Cache::init(
            builder,
            &self.config,
            self.max_sequence_length,
            in_k.clone(),
            in_v,
        );
        let [_, _, _, cache_len, _] = unpack::<5>(builder, shape(builder, in_k));

        let sh = shape(builder, x.clone());
        let normalizer = constant(builder, f32::sqrt(self.config.hidden_size as f32), &sh);

        let mut x = x * normalizer;

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                cache_len.clone(),
                p.extend(["layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm"]).unwrap(),
            x,
        );

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            p.extend(["embed_tokens"]).unwrap(),
            x,
        );

        if let Some(softcap) = self.config.final_logit_softcapping {
            x = self.softcap(builder, softcap, x);
        }

        x = argmax(builder, x);
        let (out_k, out_v) = cache.get_kv_cache(builder);
        [x, out_k, out_v]
    }
}

impl Module<3, 3> for Gemma3Model {
    fn path(&self) -> Path {
        path(vec!["gemma3"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x, in_k, in_v]: [Var; 3]) -> [Var; 3] {
        let mut root = self.path();
        if !self.root.is_empty() {
            root = root
                .extend(self.root.split('.').collect::<Vec<&str>>())
                .unwrap();
        }
        self.forward(builder, root, x, in_k, in_v)
    }

    fn ty(&self) -> ([Type; 3], [Type; 3]) {
        llm_type(&self.config)
    }
}
