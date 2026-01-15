use crate::helpers::*;
use crate::legacy::models::utils::Config;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
pub struct Qwen3Model {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl Qwen3Model {
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
        _layer_id: usize,
        cache: &mut Cache,
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

        let q = rmsnorm::<2>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm::<2>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );

        let sh = shape!(builder, b, num_heads, s, head_dim);
        let q = reshape(builder, sh, q);
        let sh = shape!(builder, b, num_kv_heads, s, head_dim);
        let k = reshape(builder, sh, k);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let q = apply_rope_embedding(
            builder,
            pos,
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
        let x = rmsnorm::<3>(
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
        let x = rmsnorm::<3>(
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

impl Module<1, 1> for Qwen3Model {
    fn path(&self) -> Path {
        path(vec!["qwen3"]).expect("invalid model path")
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

        x = rmsnorm::<3>(
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

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        llm_type()
    }
}
