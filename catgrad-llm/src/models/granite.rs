use crate::config::Config;
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;

pub struct GraniteModel {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl LLMModel for GraniteModel {}

impl GraniteModel {
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

    fn moe(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let [_, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let moe_input = param(builder, &p.extend(vec!["input_linear", "weight"]).unwrap());
        let moe_output = param(builder, &p.extend(vec!["output_linear", "weight"]).unwrap());
        let routed = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.num_local_experts,
            p.extend(["router", "layer"]).unwrap(),
            x.clone(),
        );

        let vi = topk(builder, self.config.num_experts_per_tok, routed);
        let values = vi.0;
        let indices = vi.1;

        let sh = shape!(builder, seq_len, self.config.num_experts_per_tok);

        let indices = reshape(builder, sh.clone(), indices);

        let values = reshape(builder, sh, values);

        let values = softmax(builder, values);

        let fullx = transpose(builder, 0, 1, x);
        let mut sumk = constant(builder, 0., &shape(builder, fullx.clone()));

        for i in 0..self.config.num_experts_per_tok {
            let idx = get(builder, 1, i, indices.clone());
            let val = get(builder, 1, i, values.clone());

            let gate_up = index(builder, 0, idx.clone(), moe_input.clone());
            let out = index(builder, 0, idx.clone(), moe_output.clone());

            let gate_up = chunk(builder, 1, 2, self.config.intermediate_size, gate_up);
            let gate = transpose(builder, 1, 2, gate_up[0].clone());
            let up = transpose(builder, 1, 2, gate_up[1].clone());

            let gate = matmul(builder, fullx.clone(), gate);
            let up = matmul(builder, fullx.clone(), up);
            let x = silu(builder, gate) * up; // SwiGLU

            let out = transpose(builder, 1, 2, out);
            let x = matmul(builder, x, out);

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
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.hidden_size / num_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());

        let k = linear_no_bias(
            builder,
            dim,
            dim / rep,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );

        let v = linear_no_bias(builder, dim, dim / rep, p.extend(["v_proj"]).unwrap(), x);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);

        let v = reshape(builder, sh, v);
        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

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
        let mul = constant(builder, self.config.attention_multiplier, &sh);
        let mut attn = attn * mul;

        let mask = causal_mask(builder, s.clone());
        let mask = broadcast(builder, mask, sh);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["o_proj"]).unwrap(), attn)
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        pos: Var,
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

        let sh = shape(builder, x.clone());
        let mul = constant(builder, self.config.residual_multiplier, &sh);

        let x = res + x * mul.clone();
        let res = x.clone();
        let x = rmsnorm(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = if self.config.num_experts_per_tok == 0 {
            self.mlp(builder, p.extend(["mlp"]).unwrap(), x)
        } else {
            self.moe(builder, p.extend(["block_sparse_moe"]).unwrap(), x)
        };
        x * mul + res
    }
}

impl Module<3, 3> for GraniteModel {
    fn path(&self) -> Path {
        path(vec!["granite"]).expect("invalid model path")
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

        let emb = embeddings(
            builder,
            root.extend(vec!["model", "embed_tokens"]).unwrap(),
            x,
        );

        let sh = shape(builder, emb.clone());
        let mul = constant(builder, self.config.embedding_multiplier, &sh);
        let mut x = mul * emb;

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                &mut cache,
                cache_len.clone(),
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

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(["model", "embed_tokens"]).unwrap(),
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
