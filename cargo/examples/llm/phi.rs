// Phi-3 model description

use super::{Cache, Config, ModelBuilder};
use catgrad::backend::cpu::eval::Builder;
use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Shape, Var};

pub struct Model;

impl Model {
    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType::new(
            Shape(vec![config.vocab_size, config.hidden_size]),
            Dtype::F32,
        );
        let weights = parameter(builder, t, "model.embed_tokens.weight".to_string());
        embedding(builder, x, weights)
    }

    pub fn attention(
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = config.hidden_size / num_heads;
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let qkv = linear_no_bias(builder, dim, 3 * dim, &format!("{name}.qkv_proj"), x);

        let q = narrow(builder, 2, 0, num_heads * head_dim, qkv.clone());
        let k = narrow(
            builder,
            2,
            num_heads * head_dim,
            num_kv_heads * head_dim,
            qkv.clone(),
        );
        let v = narrow(
            builder,
            2,
            (num_heads + num_kv_heads) * head_dim,
            num_kv_heads * head_dim,
            qkv,
        );

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let k = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let attn = attn / denom;

        let mask = causal_mask(builder, s);
        let mask = expand(builder, attn.label.shape.clone(), mask);
        let attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape(builder, Shape(vec![b, s, dim]), x);
        let o_proj = linear_no_bias(builder, dim, dim, &format!("{name}.o_proj"), x);
        o_proj
    }

    pub fn mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let gate_up = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size * 2,
            &format!("{name}.gate_up_proj"),
            x,
        );

        let gate_up = split(builder, 2, 2, gate_up);

        let gate = gate_up[0].clone();
        let up = gate_up[1].clone();

        let x = silu(builder, gate) * up; // SwiGLU
        let x = linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.down_proj"),
            x,
        );
        x
    }

    pub fn layer(
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.input_layernorm"),
            x,
        );
        let x = Model::attention(builder, config, cache, pos, &format!("{name}.self_attn"), x);
        let x = res + x;
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.post_attention_layernorm"),
            x,
        );
        let x = Model::mlp(builder, config, &format!("{name}.mlp"), x);
        x + res
    }
}

impl ModelBuilder for Model {
    fn build(
        &self,
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        x: Var,
    ) -> Var {
        let tokens = x.label.shape.0[1];

        let emb = Model::embeddings(builder, config, x);

        let mut result = emb;

        for i in 0..config.num_hidden_layers {
            result = Model::layer(
                builder,
                config,
                cache,
                pos,
                &format!("model.layers.{i}"),
                result,
            );
        }

        // Get the logits for the last token only
        if tokens > 1 {
            result = narrow(builder, 1, tokens - 1, 1, result);
        }

        result = rmsnorm(builder, config.rms_norm_eps, "model.norm", result);

        let lm_head_weights = if config.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        };
        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            lm_head_weights,
            result,
        )
    }
}
