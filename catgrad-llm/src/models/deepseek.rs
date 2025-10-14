// DeepSeekV3 model description

use super::utils::{Cache, Config, ModelBuilder};
use crate::nn::layers::*;
use crate::nn::rope::apply_rope_embedding;
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
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
        let q = linear_no_bias(
            builder,
            config.hidden_size,
            config.q_lora_rank,
            &format!("{name}.q_a_proj"),
            x.clone(),
        );
        let q = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.q_a_layernorm"),
            q,
        );
        let q = linear_no_bias(
            builder,
            config.q_lora_rank,
            qk_head_dim * num_heads,
            &format!("{name}.q_b_proj"),
            q,
        );

        let q = reshape_flex(
            builder,
            &[b as isize, s as isize, -1, qk_head_dim as isize],
            q,
        );
        let q = transpose(builder, 1, 2, q);

        let q_split = split(
            builder,
            -1,
            &[config.qk_nope_head_dim, config.qk_rope_head_dim],
            q,
        );

        let q_pass = q_split[0].clone();
        let q_rot = q_split[1].clone();

        let compressed_kv = linear_no_bias(
            builder,
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            &format!("{name}.kv_a_proj_with_mqa"),
            x,
        );

        let kp = split(
            builder,
            -1,
            &[config.kv_lora_rank, config.qk_rope_head_dim],
            compressed_kv,
        );

        let k_pass = kp[0].clone();
        let k_rot = kp[1].clone();

        let k_pass = rmsnorm(
            builder,
            config.rms_norm_eps,
            &format!("{name}.kv_a_layernorm"),
            k_pass,
        );

        let k_pass = linear_no_bias(
            builder,
            config.kv_lora_rank,
            num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            &format!("{name}.kv_b_proj"),
            k_pass,
        );

        let k_pass = reshape_flex(
            builder,
            &[
                b as isize,
                s as isize,
                -1,
                (config.qk_nope_head_dim + config.v_head_dim) as isize,
            ],
            k_pass,
        );
        let k_pass = transpose(builder, 1, 2, k_pass);

        let kp = split(
            builder,
            -1,
            &[config.qk_nope_head_dim, config.v_head_dim],
            k_pass,
        );

        let k_pass = kp[0].clone();
        let v = kp[1].clone();

        let k_rot = reshape(
            builder,
            Shape(vec![b, 1, s, config.qk_rope_head_dim]),
            k_rot,
        );

        let q_rot = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q_rot);
        let k_rot = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k_rot);

        let mut new_shape = k_pass.label.shape.0.clone();
        new_shape[3] = k_rot.label.shape.0[3];
        let k_rot = expand(builder, Shape(new_shape), k_rot);
        let q = concat(builder, 3, q_pass, q_rot);
        let k = concat(builder, 3, k_pass, k_rot);
        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(qk_head_dim as f32));
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
            num_heads * config.v_head_dim,
            config.hidden_size,
            &format!("{name}.o_proj"),
            x,
        )
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
            config.n_routed_experts,
            name,
            x.clone(),
        );
        let routed = sigmoid(builder, routed);

        let bias_type = NdArrayType::new(Shape(vec![config.n_routed_experts]), x.label.dtype);
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
                (config.n_routed_experts / config.n_group) as isize,
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
            config.n_routed_experts / config.n_group,
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
                config.n_routed_experts / config.n_group,
            ]),
            mask,
        );
        let mask = reshape(builder, scores_for_choice.label.shape.clone(), mask);
        let mask = cast(builder, Dtype::F32, mask);
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
        let shared = Model::mlp(builder, config, &format!("{name}.shared_experts"), res);
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
        let x = if layer_id >= config.first_k_dense_replace {
            Model::moe(builder, config, &format!("{name}.mlp"), x)
        } else {
            Model::mlp(builder, config, &format!("{name}.mlp"), x)
        };
        res + x
    }
}
