// GPT-OSS model description

use super::utils::{Cache, Config, ModelBuilder};
use crate::legacy::nn::layers::*;
use crate::legacy::nn::rope::apply_rope_embedding;
use catgrad_legacy::backend::cpu::eval::Builder;
use catgrad_legacy::core::{Dtype, NdArrayType, Shape, Var};

/// GPT-OSS specific SwiGLU
pub fn gptoss_swiglu(builder: &Builder, alpha: f32, limit: f32, gate_up: Var) -> Var {
    let mut x = gate_up;
    let xdtype = x.label.dtype;
    if xdtype != Dtype::F32 {
        x = cast(builder, Dtype::F32, x);
    }
    let w = x.label.shape.0[2];
    let idx_gate = range_indices(builder, 0, w, 2);
    let idx_up = range_indices(builder, 1, w + 1, 2);

    let gate = index(builder, 2, x.clone(), idx_gate);
    let up = index(builder, 2, x, idx_up);
    let gate = clamp(builder, gate, f32::NEG_INFINITY, limit);
    let up = clamp(builder, up, -limit, limit);

    let alpha = constant(builder, gate.label.clone(), alpha);
    let mut glu = gate.clone() * sigmoid(builder, alpha * gate);
    glu = increment(builder, up) * glu;
    if xdtype != Dtype::F32 {
        glu = cast(builder, xdtype, glu);
    }
    glu
}

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

        let lm_head_weights = if config.tie_word_embeddings {
            "model.embed_tokens"
        } else {
            "lm_head"
        };

        // Use weight tying
        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            lm_head_weights,
            result,
        )
    }
}

// Build linear layer from weight and bias params without W transpose
fn gptoss_linear(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    weight: Var,
    bias: Var,
    x: Var,
) -> Var {
    let mut w_t = weight;
    if x.label.shape.0.len() == 3 {
        let batch_size = x.label.shape.0[0];
        w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
    }

    let m = mat_mul(builder, x, w_t);
    let bias = expand(builder, m.label.shape.clone(), bias);
    m + bias
}

impl Model {
    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType::new(
            Shape(vec![config.vocab_size, config.hidden_size]),
            config.dtype,
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
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = config.get_head_dim();
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear(
            builder,
            config.hidden_size,
            num_heads * head_dim,
            &format!("{name}.q_proj"),
            x.clone(),
        );
        let k = linear(
            builder,
            config.hidden_size,
            num_kv_heads * head_dim,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear(
            builder,
            config.hidden_size,
            num_kv_heads * head_dim,
            &format!("{name}.v_proj"),
            x,
        );

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let k = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let attn = attn / denom;

        let mask = causal_mask(builder, s, attn.label.dtype);
        let mask = expand(builder, attn.label.shape.clone(), mask);
        let attn = attn + mask;

        let sinks_type = NdArrayType::new(Shape(vec![num_heads]), attn.label.dtype);
        let sinks = parameter(builder, sinks_type, format!("{name}.sinks"));
        let sinks = reshape(builder, Shape(vec![1, num_heads, 1, 1]), sinks);
        let sinks = expand(builder, Shape(vec![b, num_heads, s, 1]), sinks);

        let attn = concat(builder, 3, attn, sinks);

        let attn = softmax(builder, attn);
        let attn = narrow(builder, 3, 0, attn.label.shape.0[3] - 1, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape(builder, Shape(vec![b, s, num_heads * head_dim]), x);
        linear(
            builder,
            num_heads * head_dim,
            config.hidden_size,
            &format!("{name}.o_proj"),
            x,
        )
    }

    pub fn mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let gate_up_proj_type = NdArrayType::new(
            Shape(vec![
                config.num_local_experts,
                config.hidden_size,
                2 * config.intermediate_size,
            ]),
            x.label.dtype,
        );
        let gate_up_proj = parameter(
            builder,
            gate_up_proj_type,
            format!("{name}.experts.gate_up_proj"),
        );

        let gate_up_proj_bias_type = NdArrayType::new(
            Shape(vec![config.num_local_experts, 2 * config.intermediate_size]),
            x.label.dtype,
        );
        let gate_up_proj_bias = parameter(
            builder,
            gate_up_proj_bias_type,
            format!("{name}.experts.gate_up_proj_bias"),
        );

        let down_proj_type = NdArrayType::new(
            Shape(vec![
                config.num_local_experts,
                config.hidden_size,
                config.hidden_size,
            ]),
            x.label.dtype,
        );
        let down_proj = parameter(builder, down_proj_type, format!("{name}.experts.down_proj"));

        let down_proj_bias_type = NdArrayType::new(
            Shape(vec![config.num_local_experts, config.hidden_size]),
            x.label.dtype,
        );
        let down_proj_bias = parameter(
            builder,
            down_proj_bias_type,
            format!("{name}.experts.down_proj_bias"),
        );

        let routed = linear(
            builder,
            config.hidden_size,
            config.num_local_experts,
            &format!("{name}.router"),
            x.clone(),
        );

        let vi = topk(builder, config.num_experts_per_tok, routed);
        let values = vi[0].clone();
        let indices = vi[1].clone();

        let seq_len = x.label.shape.0[1];
        let indices = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            indices,
        );

        let values = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            values,
        );

        let values = softmax(builder, values);
        let mut xs = x.label.shape.0.clone();
        xs[1] = 0;

        let sumk_type = NdArrayType::new(Shape(xs), x.label.dtype);
        let mut sumk_all = Var::new(builder.clone(), sumk_type);
        let fullx = x;
        for s in 0..seq_len {
            let x = get(builder, 1, s, fullx.clone());
            let mut sumk = constant(builder, x.label.clone(), 0.0);

            let idx = get(builder, 0, s, indices.clone());
            let idx = squeeze(builder, 0, idx);
            let val = get(builder, 0, s, values.clone());

            for i in 0..config.num_experts_per_tok {
                let n = get(builder, 0, i, idx.clone());
                let gate_up = index(builder, 0, gate_up_proj.clone(), n.clone());
                let gate_up_bias = index(builder, 0, gate_up_proj_bias.clone(), n.clone());
                let down = index(builder, 0, down_proj.clone(), n.clone());
                let down_bias = index(builder, 0, down_proj_bias.clone(), n.clone());

                let gate_up = gptoss_linear(
                    builder,
                    config.hidden_size,
                    config.intermediate_size * 2,
                    gate_up.clone(),
                    gate_up_bias.clone(),
                    x.clone(),
                );
                let glu = gptoss_swiglu(builder, 1.702, 7.0, gate_up);

                let x = gptoss_linear(
                    builder,
                    config.intermediate_size,
                    config.hidden_size,
                    down,
                    down_bias,
                    glu,
                );

                let v = select(builder, 1, i, val.clone());
                let v = expand(builder, x.label.shape.clone(), v);
                let s = x * v;
                sumk = sumk + s;
            }
            sumk_all = concat(builder, 1, sumk_all, sumk);
        }
        sumk_all
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
        res + x
    }
}
