#![allow(clippy::too_many_arguments)]
use crate::config::{Config, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;

pub struct Qwen3Model {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl LLMModel for Qwen3Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }
}

impl Qwen3Model {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: Config = serde_json::from_value(config_json.clone())?;
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
        let x = silu(builder, gate) * up; // SwiGLU
        linear_no_bias(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    pub fn rmsnorm_raw<const N: usize>(&self, builder: &Builder, eps: f32, x: Var) -> Var {
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

    fn rmsnorm<const N: usize>(&self, builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
        let gamma = param(builder, &p.extend(["weight"]).unwrap());
        let lr = self.rmsnorm_raw::<N>(builder, eps, x);
        let lr_shape = shape(builder, lr.clone());
        let gamma = broadcast(builder, gamma, lr_shape);
        lr * gamma
    }

    fn moe(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let [_, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let gate_all = param(
            builder,
            &p.extend(vec!["experts", "gate_proj", "weight"]).unwrap(),
        );
        let up_all = param(
            builder,
            &p.extend(vec!["experts", "up_proj", "weight"]).unwrap(),
        );
        let down_all = param(
            builder,
            &p.extend(vec!["experts", "down_proj", "weight"]).unwrap(),
        );

        let routed = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.num_local_experts,
            p.extend(["gate"]).unwrap(),
            x.clone(),
        );

        let routed = softmax(builder, routed);

        let vi = topk(builder, self.config.num_experts_per_tok, routed);
        let values = vi.0;
        let indices = vi.1;

        let sh = shape!(builder, seq_len, self.config.num_experts_per_tok);

        let indices = reshape(builder, sh.clone(), indices);

        let mut values = reshape(builder, sh.clone(), values);

        if self.config.norm_topk_prob {
            let sv = sum(builder, values.clone());
            let sv = broadcast(builder, sv, sh);
            values = values / sv;
        }

        let fullx = transpose(builder, 0, 1, x);
        let mut sumk = constant(builder, 0., &shape(builder, fullx.clone()));

        for i in 0..self.config.num_experts_per_tok {
            let idx = get(builder, 1, i, indices.clone());
            let val = get(builder, 1, i, values.clone());

            let gate = index(builder, 0, idx.clone(), gate_all.clone());
            let up = index(builder, 0, idx.clone(), up_all.clone());
            let down = index(builder, 0, idx.clone(), down_all.clone());

            let gate = transpose(builder, 1, 2, gate);
            let up = transpose(builder, 1, 2, up);
            let down = transpose(builder, 1, 2, down);

            let gate = matmul(builder, fullx.clone(), gate);
            let up = matmul(builder, fullx.clone(), up);
            let x = silu(builder, gate) * up; // SwiGLU

            let x = matmul(builder, x, down);

            let v = unsqueeze::<2, 3>(builder, 2, val);
            let v = broadcast(builder, v, shape(builder, x.clone()));
            sumk = sumk + x * v;
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
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.head_dim;

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

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        // Norm
        let sh = shape!(
            builder,
            b.clone() * s.clone() * num_heads.to_nat(builder),
            head_dim
        );
        let q = reshape(builder, sh, q);
        let sh = shape!(
            builder,
            b.clone() * s.clone() * num_kv_heads.to_nat(builder),
            head_dim
        );
        let k = reshape(builder, sh, k);

        let q = self.rmsnorm::<2>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = self.rmsnorm::<2>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );

        let sh = shape!(builder, b, num_heads, s, head_dim);
        let q = reshape(builder, sh, q);
        let sh = shape!(builder, b, num_kv_heads, s, head_dim);
        let k = reshape(builder, sh, k);

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
        let x = self.rmsnorm::<3>(
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
        let x = self.rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let moe_layer = self.config.num_local_experts > 1
            && (layer_id + 1).is_multiple_of(self.config.decoder_sparse_step);
        let x = if moe_layer {
            self.moe(builder, p.extend(["mlp"]).unwrap(), x)
        } else {
            self.mlp(builder, p.extend(["mlp"]).unwrap(), x)
        };
        x + res
    }
}

impl Module<3, 3> for Qwen3Model {
    fn path(&self) -> Path {
        path(vec!["qwen3"]).expect("invalid model path")
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

        x = self.rmsnorm::<3>(
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
