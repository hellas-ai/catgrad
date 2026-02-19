#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use serde::Deserialize;

use nn::*;

#[derive(Debug, Clone, Deserialize)]
struct GPT2Config {
    #[serde(alias = "n_embd")]
    hidden_size: usize,
    #[serde(alias = "n_layer")]
    num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    num_attention_heads: usize,
    layer_norm_epsilon: f32,
    vocab_size: usize,
    eos_token_id: Option<EosTokenId>,
}

impl LLMConfig for GPT2Config {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct GPT2Model {
    config: GPT2Config,
    pub max_sequence_length: usize,
}

impl LLMModel for GPT2Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }
}

impl GPT2Model {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: GPT2Config = serde_json::from_value(config_json.clone())?;
        Ok(Self {
            config,
            max_sequence_length,
        })
    }

    pub fn embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let wte = param(builder, &p.extend(["wte", "weight"]).unwrap());

        //flatten the input tensor as that is how index expects it
        let [b, s] = unpack::<2>(builder, shape(builder, x.clone()));
        let sh = shape!(builder, b * s);
        let x = reshape(builder, sh, x);
        let te = index(builder, 0, x, wte);

        // add back batch size dim
        let sh = shape(builder, te.clone());
        let [seq_len, hidden_dim] = unpack::<2>(builder, sh);
        let sh = shape!(builder, 1, seq_len, hidden_dim);

        let te = reshape(builder, sh.clone(), te);

        let wpe = param(builder, &p.extend(["wpe", "weight"]).unwrap());
        let r = arange(builder, seq_len);
        let pe = index(builder, 0, r, wpe);
        let pe = reshape(builder, sh, pe);
        te + pe
    }

    fn gpt_linear(
        &self,
        builder: &Builder,
        _in_dim: usize,
        _out_dim: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let w = param(builder, &p.extend(["weight"]).unwrap());
        let b = param(builder, &p.extend(["bias"]).unwrap());

        // w is already transposed in GPT-2 checkpoints
        let w_t = w;

        let w_t = unsqueeze::<2, 3>(builder, 0, w_t);
        let m = matmul(builder, x, w_t);
        let sh = shape(builder, m.clone());
        let bb = broadcast(builder, b, sh);
        m + bb
    }

    fn mlp(&self, builder: &Builder, dim: usize, p: Path, x: Var) -> Var {
        let x = self.gpt_linear(builder, dim, dim * 4, p.extend(["c_fc"]).unwrap(), x);
        // let x = gelu(builder, x);
        let x = Gelu.call(builder, [x]);
        self.gpt_linear(builder, dim * 4, dim, p.extend(["c_proj"]).unwrap(), x)
    }

    fn attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = dim / num_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let c_attn = self.gpt_linear(builder, dim, 3 * dim, p.extend(["c_attn"]).unwrap(), x);

        let a = chunk(builder, 2, 3, self.config.hidden_size, c_attn);
        let q = a[0].clone();
        let k = a[1].clone();
        let v = a[2].clone();

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh.clone(), q);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = broadcast(builder, attention_mask, sh);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        self.gpt_linear(builder, dim, dim, p.extend(["c_proj"]).unwrap(), attn)
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        p: Path,
        x: Var,
    ) -> Var {
        // Params
        let ln_1 = p.extend(["ln_1"]).unwrap();
        let attn = p.extend(["attn"]).unwrap();
        let ln_2 = p.extend(["ln_2"]).unwrap();
        let mlp = p.extend(["mlp"]).unwrap();

        // layers
        let res = x.clone();
        let x = layernorm(builder, self.config.layer_norm_epsilon, ln_1, x);
        let x = self.attention(builder, layer_id, attention_mask, cache, attn, x);
        let x = res + x;

        let res = x.clone();
        let x = layernorm(builder, self.config.layer_norm_epsilon, ln_2, x);
        let x = self.mlp(builder, self.config.hidden_size, mlp, x);
        x + res
    }
}

// Implement `Def`: this is like torch's `Module`.
impl Module<3, 3> for GPT2Model {
    fn path(&self) -> Path {
        path(vec!["gpt2"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x, in_k, in_v]: [Var; 3]) -> [Var; 3] {
        let root = self.path();

        let mut cache = Cache::init(builder, &self.config, self.max_sequence_length, in_k, in_v);
        let mut x = self.embeddings(builder, root.clone(), x);
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s);

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                root.extend(["h", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = layernorm(
            builder,
            self.config.layer_norm_epsilon,
            root.extend(["ln_f"]).unwrap(),
            x,
        );

        // weight tied lm_head
        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(["wte"]).unwrap(),
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
