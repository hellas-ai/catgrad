#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig, RopeScaling};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct OlmoConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f32,
    rope_scaling: Option<RopeScaling>,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

impl LLMConfig for OlmoConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn rope_scaling(&self) -> Option<RopeScaling> {
        self.rope_scaling.clone()
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.partial_rotary_factor
    }

    fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct OlmoModel {
    config: OlmoConfig,
    pub max_sequence_length: usize,
}

impl LLMModel for OlmoModel {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }
}

impl OlmoModel {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: OlmoConfig = serde_json::from_value(config_json.clone())?;
        Ok(Self {
            config,
            max_sequence_length,
        })
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
        let head_dim = self.config.hidden_size / num_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());

        let k = linear_no_bias(
            builder,
            dim,
            dim * num_kv_heads / num_heads,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );

        let v = linear_no_bias(
            builder,
            dim,
            dim * num_kv_heads / num_heads,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let q = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

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

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = broadcast(builder, attention_mask, sh);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["o_proj"]).unwrap(), attn)
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
        let x = self.attention(
            builder,
            layer_id,
            attention_mask,
            cache,
            pos,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );

        let x = res + x;
        let res = x.clone();

        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);

        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm"]).unwrap(),
            x,
        );
        x + res
    }
}

impl Module<3, 3> for OlmoModel {
    fn path(&self) -> Path {
        path(vec!["olmo"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x, in_k, in_v]: [Var; 3]) -> [Var; 3] {
        let root = self.path();

        let mut cache = Cache::init(
            builder,
            &self.config,
            self.max_sequence_length,
            in_k.clone(),
            in_v,
        );
        let [_, _, _, cache_len, _] = unpack::<5>(builder, shape(builder, in_k));

        let mut x = embeddings(
            builder,
            root.extend(vec!["model", "embed_tokens"]).unwrap(),
            x,
        );
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s);

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                cache_len.clone(),
                root.extend(["model", "layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            root.extend(["model", "norm"]).unwrap(),
            x,
        );

        let lm_head_weights = if self.config.tie_word_embeddings {
            vec!["model", "embed_tokens"]
        } else {
            vec!["lm_head"]
        };

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(lm_head_weights).unwrap(),
            x,
        );

        x = argmax(builder, x);
        let (out_k, out_v) = cache.get_kv_cache(builder);
        [x, out_k, out_v]
    }

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 3], [Type; 3]) {
        llm_type(&self.config)
    }
}
