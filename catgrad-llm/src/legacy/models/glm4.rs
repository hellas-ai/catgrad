// GLM4 model description

use super::utils::{Cache, Config, ModelBuilder};
use crate::legacy::nn::layers::*;
use catgrad_legacy::backend::cpu::eval::Builder;
use catgrad_legacy::core::{Dtype, NdArrayType, Shape, Var};

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

fn rotate_half(builder: &Builder, x: Var) -> Var {
    let w = x.label.shape.0[3];

    let x1 = range_indices(builder, 0, w, 2);
    let x1 = index(builder, 3, x.clone(), x1);

    let x2 = range_indices(builder, 1, w + 1, 2);
    let x2 = index(builder, 3, x, x2);

    let x1 = unsqueeze(builder, -1, x1);
    let x2 = unsqueeze(builder, -1, x2);
    let r = concat(builder, 4, -x2, x1);
    let mut sh = r.label.shape.0.clone();
    sh[3] *= sh[4];
    let sh = Shape(sh[..4].to_vec());

    reshape(builder, sh, r)
}

fn apply_rope_embedding_glm4(builder: &Builder, pos: usize, cos: Var, sin: Var, x: Var) -> Var {
    let seq_len = x.label.shape.0[2];
    let cos = narrow(builder, 0, pos, seq_len, cos);
    let sin = narrow(builder, 0, pos, seq_len, sin);

    let cos = unsqueeze(builder, 0, cos);
    let cos = unsqueeze(builder, 0, cos);
    let sin = unsqueeze(builder, 0, sin);
    let sin = unsqueeze(builder, 0, sin);

    let sh_orig = cos.label.shape.clone();
    let cos = narrow(builder, 3, 0, 32, cos);
    let sin = narrow(builder, 3, 0, 32, sin);

    let sh = cos.label.shape.clone();
    let ns = sh.concatenate(&Shape(vec![2]));
    let cos = unsqueeze(builder, -1, cos);
    let cos = expand(builder, ns.clone(), cos);
    let cos = reshape(builder, sh_orig.clone(), cos);

    let sin = unsqueeze(builder, -1, sin);
    let sin = expand(builder, ns, sin);
    let sin = reshape(builder, sh_orig, sin);

    let xdtype = x.label.dtype;
    let x = chunk(builder, -1, 2, x);
    let x_rot = x[0].clone();
    let x_pass = x[1].clone();

    let mut x = x_rot;
    if xdtype != Dtype::F32 {
        x = cast(builder, Dtype::F32, x);
    }
    let rotated_x = rotate_half(builder, x.clone());

    let cos = expand(builder, x.label.shape.clone(), cos);
    let sin = expand(builder, rotated_x.label.shape.clone(), sin);
    let mut r = cos * x + sin * rotated_x;
    if xdtype != Dtype::F32 {
        r = cast(builder, xdtype, r);
    }

    concat(builder, 3, r, x_pass)
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
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let rep = num_heads / num_kv_heads;
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear_b(
            builder,
            dim,
            num_heads * head_dim,
            config.attention_bias,
            &format!("{name}.q_proj"),
            x.clone(),
        );
        let k = linear_b(
            builder,
            dim,
            num_kv_heads * head_dim,
            config.attention_bias,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_b(
            builder,
            dim,
            num_kv_heads * head_dim,
            config.attention_bias,
            &format!("{name}.v_proj"),
            x,
        );

        let mut q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let mut k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        if config.use_qk_norm {
            q = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.q_norm"), q);
            k = rmsnorm(builder, config.rms_norm_eps, &format!("{name}.k_norm"), k);
        }

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = apply_rope_embedding_glm4(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let k = apply_rope_embedding_glm4(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

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

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape_flex(builder, &[b as isize, s as isize, -1], x);

        linear_no_bias(
            builder,
            num_heads * head_dim,
            dim,
            &format!("{name}.o_proj"),
            x,
        )
    }

    pub fn mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let gate_up = linear_no_bias(
            builder,
            config.hidden_size,
            config.intermediate_size * 2,
            &format!("{name}.gate_up_proj"),
            x,
        );

        let gate_up = chunk(builder, 2, 2, gate_up);

        let gate = gate_up[0].clone();
        let up = gate_up[1].clone();

        let x = silu(builder, gate) * up; // SwiGLU

        linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.down_proj"),
            x,
        )
    }

    pub fn moe_mlp(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
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

        linear_no_bias(
            builder,
            config.intermediate_size,
            config.hidden_size,
            &format!("{name}.down_proj"),
            x,
        )
    }

    pub fn expert(builder: &Builder, config: &Config, name: &str, n: Var, x: Var) -> Var {
        let gate_up_type = NdArrayType::new(
            Shape(vec![config.moe_intermediate_size, config.hidden_size]),
            x.label.dtype,
        );

        let gate_p = parameter_dynamic(
            builder,
            gate_up_type.clone(),
            n.clone(),
            format!("{name}.gate_proj.weight"),
        );
        let gate = linear_no_bias_param(
            builder,
            config.hidden_size,
            config.moe_intermediate_size,
            gate_p,
            x.clone(),
        );

        let up_p = parameter_dynamic(
            builder,
            gate_up_type,
            n.clone(),
            format!("{name}.up_proj.weight"),
        );
        let up = linear_no_bias_param(
            builder,
            config.hidden_size,
            config.moe_intermediate_size,
            up_p,
            x,
        );
        let x = silu(builder, gate) * up; // SwiGLU

        let down_type = NdArrayType::new(
            Shape(vec![config.hidden_size, config.moe_intermediate_size]),
            x.label.dtype,
        );

        let down_p = parameter_dynamic(builder, down_type, n, format!("{name}.down_proj.weight"));
        linear_no_bias_param(
            builder,
            config.moe_intermediate_size,
            config.hidden_size,
            down_p,
            x,
        )
    }

    fn moe_topk_router(builder: &Builder, config: &Config, name: &str, x: Var) -> Vec<Var> {
        let seq_len = x.label.shape.0[1];

        let routed = linear_no_bias(
            builder,
            config.hidden_size,
            config.num_local_experts,
            name,
            x.clone(),
        );
        let routed = sigmoid(builder, routed);

        let bias_type = NdArrayType::new(Shape(vec![config.num_local_experts]), x.label.dtype);
        let bias = parameter(
            builder,
            bias_type,
            format!("{name}.e_score_correction_bias"),
        );
        let bias = expand(builder, routed.label.shape.clone(), bias);
        let scores_for_choice = routed + bias;

        let group_scores = reshape_flex(
            builder,
            &[
                -1,
                config.n_group as isize,
                (config.num_local_experts / config.n_group) as isize,
            ],
            scores_for_choice.clone(),
        );

        let group_scores = topk(builder, 2, group_scores)[0].clone();
        let group_scores = sum(builder, group_scores);
        let group_scores = squeeze(builder, 2, group_scores);
        let group_idx = topk(builder, config.topk_group, group_scores)[1].clone();

        // group_mask.scatter_(1, group_idx, 1)

        let cols = arange(builder, config.n_group, Dtype::I32);
        let cols = unsqueeze(builder, 0, cols);
        let cols = unsqueeze(builder, 0, cols);
        let mask_shape = Shape(vec![
            seq_len,
            config.num_local_experts / config.n_group,
            config.n_group,
        ]);
        let cols = expand(builder, mask_shape.clone(), cols);
        let group_idx = unsqueeze(builder, 2, group_idx);
        let group_idx = expand(builder, mask_shape, group_idx);
        let matches = eq(builder, cols, group_idx);
        let matches = transpose(builder, 1, 2, matches);
        let mask = sum(builder, matches);

        let mask = expand(
            builder,
            Shape(vec![
                seq_len,
                config.n_group,
                config.num_local_experts / config.n_group,
            ]),
            mask,
        );
        let mask = reshape(builder, scores_for_choice.label.shape.clone(), mask);
        let mask = cast(builder, config.dtype, mask);
        let scores_for_choice = mask * scores_for_choice;

        let vi = topk(builder, config.num_experts_per_tok, scores_for_choice);
        let values = vi[0].clone();
        let indices = vi[1].clone();

        let indices = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            indices,
        );

        let mut values = reshape(
            builder,
            Shape(vec![seq_len, config.num_experts_per_tok]),
            values,
        );

        if config.norm_topk_prob {
            let sv = sum(builder, values.clone());
            let eps = constant(builder, sv.label.clone(), 1e-20);
            let sv = sv + eps;
            let sv = expand(builder, values.label.shape.clone(), sv);
            values = values / sv;
            let scale = constant(builder, values.label.clone(), config.routed_scaling_factor);
            values = values * scale;
        }

        vec![values, indices]
    }

    pub fn moe(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let res = x.clone();
        let seq_len = x.label.shape.0[1];

        let vi = Model::moe_topk_router(builder, config, &format!("{name}.gate"), x.clone());
        let values = vi[0].clone();
        let indices = vi[1].clone();

        let mut xs = x.label.shape.0.clone();
        xs[1] = 0;
        let sumk_type = NdArrayType::new(Shape(xs), x.label.dtype);
        let mut sumk_all = Var::new(builder.clone(), sumk_type);
        let fullx = x;
        for s in 0..seq_len {
            let x = get(builder, 1, s, fullx.clone());
            let mut sumk = constant(builder, x.label.clone(), 0.0);
            for i in 0..config.num_experts_per_tok {
                let n = select(builder, 1, i, indices.clone());
                let n = get(builder, 0, s, n);
                let x = Model::expert(
                    builder,
                    config,
                    &format!("{name}.experts.{{}}"),
                    n,
                    x.clone(),
                );
                let v = select(builder, 1, i, values.clone());
                let v = get(builder, 0, s, v);
                let v = expand(builder, x.label.shape.clone(), v);
                sumk = sumk + x * v;
            }
            sumk_all = concat(builder, 1, sumk_all, sumk);
        }
        let shared = Model::moe_mlp(builder, config, &format!("{name}.shared_experts"), res);
        sumk_all + shared
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
        let mut x = Model::attention(
            builder,
            layer_id,
            config,
            cache,
            pos,
            &format!("{name}.self_attn"),
            x,
        );

        if config.model_type == "glm4" {
            x = rmsnorm(
                builder,
                config.rms_norm_eps,
                &format!("{name}.post_self_attn_layernorm"),
                x,
            );
        }

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.post_attention_layernorm"),
            x,
        );

        let x = if config.model_type == "glm4_moe" {
            if layer_id >= config.first_k_dense_replace {
                Model::moe(builder, config, &format!("{name}.mlp"), x)
            } else {
                Model::moe_mlp(builder, config, &format!("{name}.mlp"), x)
            }
        } else {
            let x = Model::mlp(builder, config, &format!("{name}.mlp"), x);
            rmsnorm(
                builder,
                config.rms_norm_eps,
                &format!("{name}.post_mlp_layernorm"),
                x,
            )
        };

        res + x
    }
}
