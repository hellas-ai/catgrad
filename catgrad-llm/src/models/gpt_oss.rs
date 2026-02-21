#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig, RopeScaling};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct GPTOssConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    num_experts_per_tok: usize,
    num_local_experts: usize,
    rope_theta: f32,
    rope_scaling: Option<RopeScaling>,
    sliding_window: usize,
    rms_norm_eps: f32,
    swiglu_limit: f32,
    tie_word_embeddings: bool,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
}

impl LLMConfig for GPTOssConfig {
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

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct GPTOssModel {
    config: GPTOssConfig,
    pub max_sequence_length: usize,
}

impl LLMModel for GPTOssModel {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }
}

fn gptoss_clamp(builder: &Builder, x: Var, min_val: f32, max_val: f32) -> Var {
    let sh = shape(builder, x.clone());
    let x_dtype = dtype(builder, x.clone());

    let min_t = cast(builder, constant(builder, min_val, &sh), x_dtype.clone());
    let max_t = cast(builder, constant(builder, max_val, &sh), x_dtype.clone());
    let one = cast(builder, constant(builder, 1.0, &sh), x_dtype.clone());

    let mask_min = lt(builder, x.clone(), min_t.clone());
    let mask_min = cast(builder, mask_min, x_dtype.clone());
    let x = mask_min.clone() * min_t + (one.clone() - mask_min) * x;

    let mask_max = lt(builder, max_t.clone(), x.clone());
    let mask_max = cast(builder, mask_max, x_dtype);
    mask_max.clone() * max_t + (one - mask_max) * x
}

// Build linear layer from weight and bias params without W transpose.
fn gptoss_linear(builder: &Builder, weight: Var, bias: Var, x: Var) -> Var {
    let m = matmul(builder, x, weight);
    let bias = unsqueeze::<2, 3>(builder, 1, bias);
    let bias = broadcast(builder, bias, shape(builder, m.clone()));
    m + bias
}

impl GPTOssModel {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: GPTOssConfig = serde_json::from_value(config_json.clone())?;
        Ok(Self {
            config,
            max_sequence_length,
        })
    }

    fn gptoss_swiglu(&self, builder: &Builder, gate_up: Var) -> Var {
        let original_dtype = dtype(builder, gate_up.clone());
        let x = cast(builder, gate_up, Dtype::F32);

        let idx = arange(builder, self.config.intermediate_size);
        let idx_sh = shape(builder, idx.clone());
        let two = constant(builder, 2u32, &idx_sh);
        let one = constant(builder, 1u32, &idx_sh);
        let idx_gate = idx * two;
        let idx_up = idx_gate.clone() + one;

        let gate = index(builder, 2, idx_gate, x.clone());
        let up = index(builder, 2, idx_up, x);
        let gate = gptoss_clamp(builder, gate, f32::MIN, self.config.swiglu_limit);
        let up = gptoss_clamp(
            builder,
            up,
            -self.config.swiglu_limit,
            self.config.swiglu_limit,
        );

        let alpha = constant(builder, 1.702, &shape(builder, gate.clone()));
        let glu = gate.clone() * sigmoid(builder, alpha * gate);
        let one = constant(builder, 1.0, &shape(builder, up.clone()));
        let glu = (up + one) * glu;

        cast(builder, glu, original_dtype)
    }

    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let gate_up_proj = param(builder, &p.extend(["experts", "gate_up_proj"]).unwrap());
        let gate_up_proj_bias = param(
            builder,
            &p.extend(["experts", "gate_up_proj_bias"]).unwrap(),
        );
        let down_proj = param(builder, &p.extend(["experts", "down_proj"]).unwrap());
        let down_proj_bias = param(builder, &p.extend(["experts", "down_proj_bias"]).unwrap());

        let routed = linear(
            builder,
            self.config.hidden_size,
            self.config.num_local_experts,
            p.extend(["router"]).unwrap(),
            x.clone(),
        );

        let (values, indices) = topk(builder, self.config.num_experts_per_tok, routed);
        let [_, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let sh = shape!(builder, seq_len, self.config.num_experts_per_tok);
        let mut values = reshape(builder, sh.clone(), values);
        let indices = reshape(builder, sh, indices);
        values = softmax(builder, values);

        let fullx = transpose(builder, 0, 1, x);
        let mut sumk = constant(builder, 0.0, &shape(builder, fullx.clone()));

        for i in 0..self.config.num_experts_per_tok {
            let idx = get(builder, 1, i, indices.clone());
            let idx = squeeze::<2, 1>(builder, 1, idx);
            let val = get(builder, 1, i, values.clone());

            let gate_up_w = index(builder, 0, idx.clone(), gate_up_proj.clone());
            let gate_up_b = index(builder, 0, idx.clone(), gate_up_proj_bias.clone());
            let down_w = index(builder, 0, idx.clone(), down_proj.clone());
            let down_b = index(builder, 0, idx, down_proj_bias.clone());

            let gate_up = gptoss_linear(builder, gate_up_w, gate_up_b, fullx.clone());
            let glu = self.gptoss_swiglu(builder, gate_up);
            let x = gptoss_linear(builder, down_w, down_b, glu);

            let val = unsqueeze::<2, 3>(builder, 2, val);
            let val = broadcast(builder, val, shape(builder, x.clone()));
            sumk = sumk + x * val;
        }

        transpose(builder, 0, 1, sumk)
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
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads();
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.get_head_dim();

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let q = linear(
            builder,
            self.config.hidden_size,
            num_heads * head_dim,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let k = linear(
            builder,
            self.config.hidden_size,
            num_kv_heads * head_dim,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = linear(
            builder,
            self.config.hidden_size,
            num_kv_heads * head_dim,
            p.extend(["v_proj"]).unwrap(),
            x,
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

        let sinks = param(builder, &p.extend(["sinks"]).unwrap());
        let sinks = reshape(builder, shape!(builder, 1, num_heads, 1, 1), sinks);
        let sinks = broadcast(builder, sinks, shape!(builder, b, num_heads, s, 1));
        let attn = concat(builder, 3, attn, sinks);

        let attn = softmax(builder, attn);
        let [_, _, kv_len, _] = unpack::<4>(builder, shape(builder, v.clone()));
        let attn = slice(builder, 3, 0, kv_len, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, num_heads * head_dim);
        let attn = reshape(builder, sh, attn);

        linear(
            builder,
            num_heads * head_dim,
            self.config.hidden_size,
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
        let x = rmsnorm(
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
        let x = res + x;

        let res = x.clone();
        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        res + x
    }
}

impl Module<3, 3> for GPTOssModel {
    fn path(&self) -> Path {
        path(vec!["gpt_oss"]).expect("invalid model path")
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

    fn ty(&self) -> ([Type; 3], [Type; 3]) {
        llm_type(&self.config)
    }
}
