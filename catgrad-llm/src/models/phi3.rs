use crate::helpers::*;
use crate::legacy::models::utils::Config;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;

pub struct Phi3Model {
    pub config: Config,
    pub max_sequence_length: usize,
}

impl Phi3Model {
    fn get_proj(&self, p: &Path, name: &str) -> Path {
        if self.config.model_type == "phi4mm" {
            p.extend([name, "base_layer"]).unwrap()
        } else {
            p.extend([name]).unwrap()
        }
    }

    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let gate_up = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size * 2,
            self.get_proj(&p, "gate_up_proj"),
            x,
        );
        let gate = slice(
            builder,
            2,
            0,
            self.config.intermediate_size,
            gate_up.clone(),
        );
        let up = slice(
            builder,
            2,
            self.config.intermediate_size,
            self.config.intermediate_size,
            gate_up,
        );
        let x = silu(builder, gate) * up;
        linear_no_bias(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            self.get_proj(&p, "down_proj"),
            x,
        )
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
        let head_dim = self.config.hidden_size / num_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let op_size = head_dim * (num_heads + 2 * num_kv_heads);

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let qkv = linear_no_bias(builder, dim, op_size, self.get_proj(&p, "qkv_proj"), x);

        let q = slice(builder, 2, 0, num_heads * head_dim, qkv.clone());
        let k = slice(
            builder,
            2,
            num_heads * head_dim,
            num_kv_heads * head_dim,
            qkv.clone(),
        );

        let v = slice(
            builder,
            2,
            (num_heads + num_kv_heads) * head_dim,
            num_kv_heads * head_dim,
            qkv,
        );

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);

        let v = reshape(builder, sh, v);
        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

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
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, self.get_proj(&p, "o_proj"), attn)
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
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        x + res
    }
}

impl Module<1, 1> for Phi3Model {
    fn path(&self) -> Path {
        path(vec!["phi3"]).expect("invalid model path")
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

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        llm_type()
    }
}
