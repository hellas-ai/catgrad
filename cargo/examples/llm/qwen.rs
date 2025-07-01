// Qwen-3 model description

use super::{Cache, Config, ModelBuilder};
use catgrad::backend::cpu::eval::Builder;
use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Shape, Var};

pub struct Model;

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
                i,
                config,
                cache,
                pos,
                &format!("model.layers.{i}"),
                result,
            );
        }

        result = rmsnorm(builder, config.rms_norm_eps, "model.norm", result);

        // Get the logits for the last token only
        if tokens > 1 {
            result = narrow(builder, 1, tokens - 1, 1, result);
        }

        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            "lm_head",
            result,
        )
    }
}

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
        layer_id: usize,
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
        let head_dim = config.head_dim;
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim,
            &format!("{name}.q_proj"),
            x.clone(),
        );
        let k = linear_no_bias(
            builder,
            dim,
            num_kv_heads * head_dim,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_no_bias(
            builder,
            dim,
            num_kv_heads * head_dim,
            &format!("{name}.v_proj"),
            x,
        );

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let mut v = transpose(builder, 1, 2, v);

        // Norm
        let q = reshape(builder, Shape(vec![b * s * num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b * s * num_kv_heads, head_dim]), k);
        let q = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.q_norm"), q);
        let k = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.k_norm"), k);
        let q = reshape(builder, Shape(vec![b, num_heads, s, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, num_kv_heads, s, head_dim]), k);

        // Rope embeddings
        let q = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let mut k = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

        if cache.use_kv_cache {
            if let Some((cached_k, cached_v)) = &cache.kv_cache[layer_id] {
                k = concat(builder, 2, cached_k.clone(), k.clone());
                v = concat(builder, 2, cached_v.clone(), v.clone());
            }
            cache.kv_cache[layer_id] = Some((k.clone(), v.clone()));
        };

        // GQA repeat
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
        let x = reshape(builder, Shape(vec![b, s, num_heads * head_dim]), x);
        let o_proj = linear_no_bias(
            builder,
            num_heads * head_dim,
            dim,
            &format!("{name}.o_proj"),
            x,
        );
        o_proj
    }

    pub fn mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let gated = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size,
            &format!("{name}.gate_proj"),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size,
            &format!("{name}.up_proj"),
            x,
        );
        let x = silu(builder, gated) * up; // SwiGLU
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
        layer_id: usize,
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
        let x = Model::attention(
            builder,
            layer_id,
            config,
            cache,
            pos,
            &format!("{name}.self_attn"),
            x,
        );
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
