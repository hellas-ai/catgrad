#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig, RopeScaling, YarnRopeScaling};
use crate::helpers::*;
use crate::models::siglip::SiglipVisionConfig;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Mistral3Config {
    pub text_config: Mistral3TextConfig,
    pub vision_config: Option<SiglipVisionConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Mistral3TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_parameters: YarnRopeScaling,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub eos_token_id: Option<EosTokenId>,
    pub vocab_size: usize,
}

impl LLMConfig for Mistral3TextConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }

    fn rope_scaling(&self) -> Option<RopeScaling> {
        Some(RopeScaling::Yarn(self.rope_parameters.clone()))
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct Mistral3Model {
    pub root: String,
    config: Mistral3TextConfig,
    dtype: Dtype,
    pub max_sequence_length: usize,
}

impl LLMModel for Mistral3Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }
}

impl Mistral3Model {
    pub fn new(
        root: &str,
        config_json: &serde_json::Value,
        max_sequence_length: usize,
        dtype: Dtype,
    ) -> crate::Result<Self> {
        let config: Mistral3Config = serde_json::from_value(config_json.clone())?;
        Ok(Self {
            root: root.to_string(),
            config: config.text_config,
            dtype,
            max_sequence_length,
        })
    }

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

    fn attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.get_head_dim();
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;

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

        let mut q = apply_rope_embedding(
            builder,
            pos.clone(),
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            q,
        );
        let k = apply_rope_embedding(
            builder,
            pos.clone(),
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            k,
        );

        // Llama 4 style scaling
        let beta = self.config.rope_parameters.llama_4_scaling_beta;
        if beta != 0.0 {
            let orig_max = self.config.rope_parameters.original_max_position_embeddings;
            let r = arange(builder, s.clone());
            let sh = shape(builder, r.clone());

            let pos = nat_to_u32(builder, pos);
            let pos = broadcast(builder, sh.clone(), pos);
            let pos = r + pos;
            let pos = cast(builder, pos, Dtype::F32);
            let orig_max = constant(builder, orig_max as f32, &sh);

            let ratio = floor(builder, pos / orig_max);
            let one = constant(builder, 1.0, &sh);
            let scale = one.clone() + constant(builder, beta, &sh) * log(builder, one + ratio);
            let scale = unsqueeze::<1, 2>(builder, 1, scale);
            let scale = broadcast(builder, shape(builder, q.clone()), scale);

            q = q * scale;
        }

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = broadcast(builder, sh, attention_mask);
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
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
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
            attention_mask,
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
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        x + res
    }
}

impl DynModule for Mistral3Model {
    fn path(&self) -> Path {
        path(vec!["mistral3"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, max_positions]: [Var; 4] = args.try_into().expect("expected 4 inputs");
        let mut root = self.path();
        if !self.root.is_empty() {
            root = root.extend(self.root.split('.')).unwrap();
        }

        let mut cache = Cache::init(builder, &self.config, max_positions, in_k.clone(), in_v);
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k));

        let mut x = embeddings(builder, root.extend(["model", "embed_tokens"]).unwrap(), x);

        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s, pos.clone());

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                pos.clone(),
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
        let (out_k, out_v) = cache.get_kv_cache(builder);
        vec![x, out_k, out_v]
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        llm_type(&self.config, self.dtype())
    }
}
