use super::helpers::*;
use crate::{Cache, llm_type};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad_llm::models::utils::Config;
use nn::*;
pub struct Gemma3Model {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl Gemma3Model {
    pub fn embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let wte = param(
            builder,
            &p.extend(vec!["model", "embed_tokens", "weight"]).unwrap(),
        );
        let te = index(builder, 0, x, wte);

        unsqueeze::<2, 3>(builder, 0, te)
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

    // Gemma uses a non-standard RMSNorm implementation.
    // Generic because of unpack needing the last dimension and it is being called
    // with ranks 2 and 3 too.
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
        let sh = shape(builder, gamma.clone());
        let one = constant(builder, 1.0, &sh);
        lr * (one + gamma)
    }

    fn attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        _cache: &mut Cache,
        pos: usize,
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

        // Every 6th layer uses global attention, otherwise local attention, with different rope frequencies
        let theta = if !(layer_id + 1).is_multiple_of(self.config.sliding_window_pattern) {
            self.config.rope_local_base_freq
        } else {
            self.config.rope_theta
        };
        let q = rope(builder, theta, pos, &s, head_dim, q);
        let k = rope(builder, theta, pos, &s, head_dim, k);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = causal_mask(builder, s.clone());
        let mask = broadcast(builder, mask, sh);
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
        cache: &mut Cache,
        pos: usize,
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
            cache,
            pos,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = self.rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = res + x;
        let res = x.clone();
        let x = self.rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_feedforward_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        let x = self.rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm"]).unwrap(),
            x,
        );
        x + res
    }
}

impl Module<1, 1> for Gemma3Model {
    fn path(&self) -> Path {
        path(vec!["gemma3"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let root = self.path();

        let mut cache = Cache::init(builder, &self.config, self.max_sequence_length);

        let mut x = self.embeddings(builder, root.clone(), x);

        let sh = shape(builder, x.clone());
        let normalizer = constant(builder, f32::sqrt(self.config.hidden_size as f32), &sh);

        x = x * normalizer;

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                &mut cache,
                0,
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

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(["model", "embed_tokens"]).unwrap(),
            x,
        );

        x = argmax(builder, x);
        [x]
    }

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        llm_type()
    }
}
