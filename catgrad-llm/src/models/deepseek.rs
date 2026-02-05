use crate::config::Config;
use crate::helpers::*;
use catgrad::category::lang::eq;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;

pub struct DeepSeekModel {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl LLMModel for DeepSeekModel {}

impl DeepSeekModel {
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

    fn moe_topk_router(&self, builder: &Builder, p: Path, x: Var) -> (Var, Var) {
        let [_, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let routed = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.num_local_experts,
            p.clone(),
            x,
        );
        let routed = sigmoid(builder, routed);

        let bias = param(builder, &p.extend(["e_score_correction_bias"]).unwrap());
        let routed_sh = shape(builder, routed.clone());
        let bias = broadcast(builder, bias, routed_sh.clone());
        let scores_for_choice = routed + bias;

        let n_groups = self.config.n_group;
        let experts_per_group = self.config.num_local_experts / n_groups;

        let sh = shape!(builder, seq_len, n_groups, experts_per_group);
        let group_scores = reshape(builder, sh, scores_for_choice.clone());

        let group_scores = topk(builder, 2, group_scores).0;
        let group_scores = sum(builder, group_scores);
        let sh = shape!(builder, seq_len, n_groups);
        let group_scores = reshape(builder, sh, group_scores);

        let group_idx = topk(builder, self.config.topk_group, group_scores).1;
        let sh = shape!(builder, seq_len, experts_per_group, 1);
        let group_idx = reshape(builder, sh, group_idx);

        // group_mask.scatter_(1, group_idx, 1)
        //
        let cols = arange(builder, n_groups);
        let sh = shape!(builder, seq_len, experts_per_group, n_groups);
        let cols = broadcast(builder, cols, sh.clone());

        let group_idx = broadcast(builder, group_idx, sh);

        let matches = eq(builder, cols, group_idx);
        let matches = transpose(builder, 1, 2, matches);
        let mask = sum(builder, matches);

        let sh = shape!(builder, seq_len, n_groups, experts_per_group);
        let mask = broadcast(builder, mask, sh);
        let mask = reshape(builder, routed_sh, mask);
        let mask = cast(builder, mask, Dtype::F32);
        let scores_for_choice = mask * scores_for_choice;

        let vi = topk(builder, self.config.num_experts_per_tok, scores_for_choice);
        let values = vi.0;
        let indices = vi.1;

        let sh = shape!(builder, seq_len, self.config.num_experts_per_tok);
        let indices = reshape(builder, sh.clone(), indices);
        let mut values = reshape(builder, sh.clone(), values);

        if self.config.norm_topk_prob {
            let sv = sum(builder, values.clone());
            let sv_sh = shape(builder, sv.clone());
            let eps = constant(builder, 1e-20, &sv_sh);
            let sv = sv + eps;
            let sv = broadcast(builder, sv, sh);
            values = values / sv;
            let scale_sh = shape(builder, values.clone());
            let scale = constant(builder, self.config.routed_scaling_factor, &scale_sh);
            values = values * scale;
        }

        (values, indices)
    }

    fn moe(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let res = x.clone();
        let [_, _seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let (values, indices) =
            self.moe_topk_router(builder, p.extend(["gate"]).unwrap(), x.clone());

        let gate_all = param(
            builder,
            &p.extend(["experts", "gate_proj", "weight"]).unwrap(),
        );
        let up_all = param(
            builder,
            &p.extend(["experts", "up_proj", "weight"]).unwrap(),
        );
        let down_all = param(
            builder,
            &p.extend(["experts", "down_proj", "weight"]).unwrap(),
        );

        let fullx = transpose(builder, 0, 1, x);
        let sh = shape(builder, fullx.clone());
        let mut sumk = constant(builder, 0.0, &sh);

        for i in 0..self.config.num_experts_per_tok {
            let idx = get(builder, 1, i, indices.clone());
            let val = get(builder, 1, i, values.clone());

            let gate = index(builder, 0, idx.clone(), gate_all.clone());
            let up = index(builder, 0, idx.clone(), up_all.clone());
            let down = index(builder, 0, idx, down_all.clone());

            // Transpose to get [hidden_size, moe_intermediate_size] shape
            let gate = transpose(builder, 1, 2, gate);
            let up = transpose(builder, 1, 2, up);
            let down = transpose(builder, 1, 2, down);

            // Apply the expert transformations
            let gate = matmul(builder, fullx.clone(), gate);
            let up = matmul(builder, fullx.clone(), up);
            let x = silu(builder, gate) * up;

            let x = matmul(builder, x, down);

            let v = unsqueeze::<2, 3>(builder, 2, val);
            let v = broadcast(builder, v, shape(builder, x.clone()));
            sumk = sumk + x * v;
        }

        let sumk = transpose(builder, 0, 1, sumk);

        // Add shared experts
        let shared = self.mlp(builder, p.extend(["shared_experts"]).unwrap(), res);
        sumk + shared
    }

    fn attention(
        &self,
        builder: &Builder,
        _layer_id: usize,
        cache: &mut Cache,
        pos: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let qk_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        // Query projection with LoRA
        let q = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.q_lora_rank,
            p.extend(["q_a_proj"]).unwrap(),
            x.clone(),
        );
        let q = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_a_layernorm"]).unwrap(),
            q,
        );
        let q = linear_no_bias(
            builder,
            self.config.q_lora_rank,
            qk_head_dim * num_heads,
            p.extend(["q_b_proj"]).unwrap(),
            q,
        );

        let sh = shape!(builder, b, s, num_heads, qk_head_dim);
        let q = reshape(builder, sh, q);
        let q = transpose(builder, 1, 2, q);

        // Split q into pass and rope parts
        let q_split = split(
            builder,
            3,
            &[self.config.qk_nope_head_dim, self.config.qk_rope_head_dim],
            q,
        );
        let q_pass = q_split[0].clone();
        let q_rot = q_split[1].clone();

        // KV projection with compression
        let compressed_kv = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            p.extend(["kv_a_proj_with_mqa"]).unwrap(),
            x,
        );

        let kp_split = split(
            builder,
            2,
            &[self.config.kv_lora_rank, self.config.qk_rope_head_dim],
            compressed_kv,
        );
        let k_pass = kp_split[0].clone();
        let k_rot = kp_split[1].clone();

        let k_pass = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["kv_a_layernorm"]).unwrap(),
            k_pass,
        );

        let k_pass = linear_no_bias(
            builder,
            self.config.kv_lora_rank,
            num_heads * (self.config.qk_nope_head_dim + self.config.v_head_dim),
            p.extend(["kv_b_proj"]).unwrap(),
            k_pass,
        );

        let sh = shape!(
            builder,
            b,
            s,
            num_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim
        );
        let k_pass = reshape(builder, sh, k_pass);
        let k_pass = transpose(builder, 1, 2, k_pass);

        let kv_chunks = split(
            builder,
            3,
            &[self.config.qk_nope_head_dim, self.config.v_head_dim],
            k_pass,
        );
        let k_pass = kv_chunks[0].clone();
        let v = kv_chunks[1].clone();

        let sh = shape!(builder, b, 1, s, self.config.qk_rope_head_dim);
        let k_rot = reshape(builder, sh, k_rot);

        // Apply RoPE
        let q_rot = apply_rope_embedding(
            builder,
            pos,
            self.config.qk_rope_head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            q_rot,
        );
        let k_rot = apply_rope_embedding(
            builder,
            pos,
            self.config.qk_rope_head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            k_rot,
        );

        // Broadcast k_rot to match k_pass shape
        let sh = shape!(builder, b, num_heads, s, self.config.qk_rope_head_dim);
        let k_rot = broadcast(builder, k_rot, sh);

        let q = concat(builder, 3, q_pass, q_rot);
        let k = concat(builder, 3, k_pass, k_rot);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(qk_head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = causal_mask(builder, s.clone());
        let mask = broadcast(builder, mask, sh);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, num_heads * self.config.v_head_dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(
            builder,
            num_heads * self.config.v_head_dim,
            self.config.hidden_size,
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
        let x = rmsnorm(
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

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = if layer_id >= self.config.first_k_dense_replace {
            self.moe(builder, p.extend(["mlp"]).unwrap(), x)
        } else {
            self.mlp(builder, p.extend(["mlp"]).unwrap(), x)
        };
        res + x
    }
}

impl Module<1, 1> for DeepSeekModel {
    fn path(&self) -> Path {
        path(vec!["deepseek"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let root = self.path();

        let mut cache = Cache::init(builder, &self.config, self.max_sequence_length);

        let mut x = embeddings(
            builder,
            root.extend(vec!["model", "embed_tokens"]).unwrap(),
            x,
        );

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
        [x]
    }

    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        llm_type()
    }
}
