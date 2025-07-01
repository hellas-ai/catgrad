// GPT-2 model description

use super::{Cache, Config, ModelBuilder};
use catgrad::backend::cpu::eval::Builder;
use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Shape, Var};

pub struct Model;

impl Model {
    // The original GPT2 checkpoints use a Conv1D layer instead of linear,
    // equivalent to a linear layer with weights in transposed order
    fn gpt_linear(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
        let w_type = NdArrayType::new(Shape(vec![in_dim, out_dim]), x.label.dtype);
        let b_type = NdArrayType::new(Shape(vec![out_dim]), x.label.dtype);

        let w = parameter(builder, w_type, format!("{name}.weight"));
        let b = parameter(builder, b_type, format!("{name}.bias"));

        // w is already transposed in GPT-2 checkpoints
        let mut w_t = w;

        if x.label.shape.0.len() == 3 {
            let batch_size = x.label.shape.0[0];
            w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
        }

        let m = mat_mul(builder, x, w_t);
        let bb = expand(builder, m.label.shape.clone(), b);
        m + bb
    }

    pub fn embeddings(builder: &Builder, config: &Config, x: Var, pos: usize) -> Var {
        let t = NdArrayType::new(
            Shape(vec![config.vocab_size, config.hidden_size]),
            Dtype::F32,
        );
        let weights = parameter(builder, t, "wte.weight".to_string());
        let we = embedding(builder, x.clone(), weights);

        let t = NdArrayType::new(
            Shape(vec![config.max_position_embeddings, config.hidden_size]),
            Dtype::F32,
        );
        let pos = range_indices(builder, pos, pos + x.label.size());
        let pos = expand(builder, x.label.shape, pos);
        let weights = parameter(builder, t, "wpe.weight".to_string());
        let pe = embedding(builder, pos, weights);

        we + pe
    }

    pub fn attention(
        builder: &Builder,
        layer_id: usize,
        config: &Config,
        cache: &mut Cache,
        name: &str,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = dim / num_heads;

        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let c_attn = Model::gpt_linear(builder, dim, 3 * dim, &format!("{name}.c_attn"), x);

        let a = split(builder, 2, 3, c_attn);
        let q = a[0].clone();
        let k = a[1].clone();
        let v = a[2].clone();

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let mut k = transpose(builder, 1, 2, k);
        let mut v = transpose(builder, 1, 2, v);

        if cache.use_kv_cache {
            if let Some((cached_k, cached_v)) = &cache.kv_cache[layer_id] {
                k = concat(builder, 2, cached_k.clone(), k.clone());
                v = concat(builder, 2, cached_v.clone(), v.clone());
            }
            cache.kv_cache[layer_id] = Some((k.clone(), v.clone()));
        };

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let mut attn = attn / denom;

        if s > 1 {
            let mask = causal_mask(builder, s);
            let mask = expand(builder, attn.label.shape.clone(), mask);
            attn = attn + mask;
        }

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let attn = reshape(builder, Shape(vec![b, s, dim]), attn);

        let c_proj = Model::gpt_linear(builder, dim, dim, &format!("{name}.c_proj"), attn);
        c_proj
    }

    pub fn mlp(builder: &Builder, dim: usize, name: &str, x: Var) -> Var {
        let x = Model::gpt_linear(builder, dim, dim * 4, &format!("{name}.c_fc"), x);
        let x = gelu(builder, x);
        let x = Model::gpt_linear(builder, dim * 4, dim, &format!("{name}.c_proj"), x);
        x
    }

    pub fn layer(
        builder: &Builder,
        layer_id: usize,
        config: &Config,
        cache: &mut Cache,
        name: &str,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_epsilon,
            &format!("{name}.ln_1"),
            x,
        );
        let x = Model::attention(builder, layer_id, config, cache, &format!("{name}.attn"), x);
        let x = res + x;
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_epsilon,
            &format!("{name}.ln_2"),
            x,
        );
        let x = Model::mlp(builder, config.hidden_size, &format!("{name}.mlp"), x);
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

        let emb = Model::embeddings(builder, config, x, pos);

        let mut result = emb;

        for layer_id in 0..config.num_hidden_layers {
            result = Model::layer(
                builder,
                layer_id,
                config,
                cache,
                &format!("h.{layer_id}"),
                result,
            );
        }

        result = layernorm(builder, config.layer_norm_epsilon, "ln_f", result);

        // Get the logits for the last token only
        if tokens > 1 {
            result = narrow(builder, 1, tokens - 1, 1, result);
        }

        // GPT-2 uses weight tying so lm_head is the same as wte
        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            "wte",
            result,
        )
    }
}
