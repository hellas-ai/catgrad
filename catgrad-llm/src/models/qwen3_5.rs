#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Qwen3_5Config {
    text_config: Qwen3_5TextConfig,
    pub vision_config: Qwen3_5VisionConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Qwen3_5VisionConfig {}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Qwen3_5RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_mrope_interleaved")]
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct Qwen3_5TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default = "default_full_attention_interval")]
    full_attention_interval: usize,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_value_head_dim: usize,
    linear_num_value_heads: usize,
    #[serde(default)]
    rope_parameters: Qwen3_5RopeParameters,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
}

impl Qwen3_5TextConfig {
    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        if self.layer_types.len() == self.num_hidden_layers {
            self.layer_types[layer_id] == "full_attention"
        } else {
            (layer_id + 1).is_multiple_of(self.full_attention_interval)
        }
    }
}

fn default_full_attention_interval() -> usize {
    4
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

fn default_mrope_section() -> Vec<usize> {
    vec![11, 11, 10]
}

fn default_mrope_interleaved() -> bool {
    true
}

impl LLMConfig for Qwen3_5TextConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&layer_id| self.is_full_attention_layer(layer_id))
            .count()
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters.partial_rotary_factor
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct Qwen3_5Model {
    config: Qwen3_5TextConfig,
    layer_to_cache_id: Vec<Option<usize>>,
    layer_to_linear_id: Vec<Option<usize>>,
    num_linear_layers: usize,
    pub max_sequence_length: usize,
}

impl LLMModel for Qwen3_5Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        vec![
            (
                Dtype::F32,
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_qk_head_dim(),
                ]),
            ),
            (
                Dtype::F32,
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_v_head_dim(),
                ]),
            ),
            (
                Dtype::F32,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.linear_conv_dim(),
                    self.config.linear_conv_kernel_dim,
                ]),
            ),
            (
                Dtype::F32,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.config.linear_num_value_heads,
                    self.config.linear_key_head_dim,
                    self.config.linear_value_head_dim,
                ]),
            ),
        ]
    }
}

impl Qwen3_5Model {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: Qwen3_5Config = serde_json::from_value(config_json.clone())?;
        let config = config.text_config;
        assert!(
            config.linear_conv_kernel_dim > 0,
            "qwen3_5 linear_conv_kernel_dim must be > 0"
        );
        let mut layer_to_cache_id = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_to_linear_id = Vec::with_capacity(config.num_hidden_layers);
        let mut next_cache_id = 0;
        let mut next_linear_id = 0;
        for layer_id in 0..config.num_hidden_layers {
            if config.is_full_attention_layer(layer_id) {
                layer_to_cache_id.push(Some(next_cache_id));
                layer_to_linear_id.push(None);
                next_cache_id += 1;
            } else {
                layer_to_cache_id.push(None);
                layer_to_linear_id.push(Some(next_linear_id));
                next_linear_id += 1;
            }
        }

        Ok(Self {
            config,
            layer_to_cache_id,
            layer_to_linear_id,
            num_linear_layers: next_linear_id,
            max_sequence_length,
        })
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.config.is_full_attention_layer(layer_id)
    }

    fn linear_conv_dim(&self) -> usize {
        let key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim;
        let value_dim = self.config.linear_num_value_heads * self.config.linear_value_head_dim;
        key_dim * 2 + value_dim
    }

    fn rotary_dim(&self, head_dim: usize) -> usize {
        let mut rotary_dim = (head_dim as f32 * self.config.partial_rotary_factor()) as usize;
        let mrope_half_dim = self
            .config
            .rope_parameters
            .mrope_section
            .iter()
            .copied()
            .sum::<usize>();
        if mrope_half_dim > 0 {
            rotary_dim = mrope_half_dim * 2;
        }
        if rotary_dim == 0 || rotary_dim > head_dim {
            rotary_dim = head_dim;
        }
        rotary_dim -= rotary_dim % 2;
        if rotary_dim == 0 {
            head_dim
        } else {
            rotary_dim
        }
    }

    fn apply_rope_partial(&self, builder: &Builder, pos: Var, cache: &Cache, x: Var) -> Var {
        let head_dim = self.config.get_head_dim();
        let rotary_dim = self.rotary_dim(head_dim);

        // Full Qwen3.5 mRoPE interleaving uses 3D position IDs (T/H/W). In this text-only path,
        // all axes collapse to the same scalar position so partial RoPE is equivalent here.
        if rotary_dim >= head_dim {
            return apply_rope_embedding(
                builder,
                pos,
                head_dim,
                cache.cos.clone(),
                cache.sin.clone(),
                x,
            );
        }

        let split = split(builder, 3, &[rotary_dim, head_dim - rotary_dim], x);
        let x_rot = split[0].clone();
        let x_pass = split[1].clone();
        let x_rot = apply_rope_embedding(
            builder,
            pos,
            rotary_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            x_rot,
        );
        concat(builder, 3, x_rot, x_pass)
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

    fn full_attention(
        &self,
        builder: &Builder,
        cache_layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.get_head_dim();

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim * 2,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let q = reshape(builder, shape!(builder, b, s, num_heads, head_dim * 2), q);
        let q = chunk(builder, 3, 2, head_dim, q);
        let sh = shape!(builder, b, s, num_heads * head_dim);
        let q_states = q[0].clone();
        let q_gate = q[1].clone();
        let q_states = reshape(builder, sh.clone(), q_states);
        let q_gate = reshape(builder, sh, q_gate);

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
        let q = reshape(builder, sh, q_states);
        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = rmsnorm_gemma::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm_gemma::<4>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = self.apply_rope_partial(builder, pos.clone(), cache, q);
        let k = self.apply_rope_partial(builder, pos, cache, k);

        let (k, v) = cache.update_kv_cache(builder, cache_layer_id, k, v);

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
        let mut attn = reshape(builder, sh, attn);

        attn = attn * sigmoid(builder, q_gate);

        linear_no_bias(
            builder,
            num_heads * head_dim,
            dim,
            p.extend(["o_proj"]).unwrap(),
            attn,
        )
    }

    fn linear_attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        hidden_states: Var,
    ) -> Var {
        let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, hidden_states.clone()));

        let num_k_heads = self.config.linear_num_key_heads;
        let num_v_heads = self.config.linear_num_value_heads;
        let head_k_dim = self.config.linear_key_head_dim;
        let head_v_dim = self.config.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let cache_len = self.config.linear_conv_kernel_dim;
        let linear_layer_id =
            self.layer_to_linear_id[layer_id].expect("linear-attention layer missing state index");
        let conv_state = cache
            .linear_state
            .as_ref()
            .expect("qwen3_5 linear attention requires conv state")[linear_layer_id]
            .clone();
        let recurrent_state = cache
            .recurrent_state
            .as_ref()
            .expect("qwen3_5 linear attention requires recurrent state")[linear_layer_id]
            .clone();

        let mixed_qkv = linear_no_bias(
            builder,
            self.config.hidden_size,
            conv_dim,
            p.extend(["in_proj_qkv"]).unwrap(),
            hidden_states.clone(),
        );
        let mixed_qkv = transpose(builder, 1, 2, mixed_qkv);

        let pos_u32 = nat_to_u32(builder, pos);
        let zero_pos = constant(builder, 0u32, &shape(builder, pos_u32.clone()));
        let is_decode = gt(builder, pos_u32, zero_pos);

        let results = cond(
            builder,
            is_decode.clone(),
            |b, args: Vec<Var>| {
                let [mixed_qkv, conv_state, _batch_size, seq_len] = args.try_into().unwrap();
                let hidden_states_new = concat(b, 2, conv_state, mixed_qkv);
                let out_conv_state_decode =
                    slice(b, 2, seq_len.clone(), cache_len, hidden_states_new);
                let mixed_qkv_decode = silu(
                    b,
                    padded_depthwise_conv1d_no_bias(
                        b,
                        p.extend(["conv1d"]).unwrap(),
                        cache_len,
                        out_conv_state_decode.clone(),
                        seq_len,
                    ),
                );
                vec![mixed_qkv_decode, out_conv_state_decode]
            },
            |b, args: Vec<Var>| {
                let [mixed_qkv, _conv_state, batch_size, seq_len] = args.try_into().unwrap();
                let mixed_qkv_prefill = silu(
                    b,
                    depthwise_conv1d_no_bias(
                        b,
                        p.extend(["conv1d"]).unwrap(),
                        cache_len,
                        mixed_qkv.clone(),
                        cache_len - 1,
                    ),
                );

                let zeros_prefill_state = zeros(b, &shape!(b, batch_size, conv_dim, cache_len));
                let x_padded_prefill_state = concat(b, 2, zeros_prefill_state, mixed_qkv);
                let out_conv_state_prefill =
                    slice(b, 2, seq_len, cache_len, x_padded_prefill_state);
                vec![mixed_qkv_prefill, out_conv_state_prefill]
            },
            vec![mixed_qkv, conv_state, batch_size.clone(), seq_len.clone()],
        );

        let mixed_qkv = results[0].clone();
        let out_conv_state = results[1].clone();

        cache
            .linear_state
            .as_mut()
            .expect("qwen3_5 linear attention requires mutable conv state")[linear_layer_id] =
            out_conv_state;

        let mixed_qkv = transpose(builder, 1, 2, mixed_qkv);

        let qkv = split(builder, 2, &[key_dim, key_dim, value_dim], mixed_qkv);
        let query = qkv[0].clone();
        let key = qkv[1].clone();
        let value = qkv[2].clone();

        let sh = shape!(builder, batch_size, seq_len, num_k_heads, head_k_dim);
        let mut query = reshape(builder, sh.clone(), query);
        let mut key = reshape(builder, sh, key);
        let sh = shape!(builder, batch_size, seq_len, num_v_heads, head_v_dim);
        let value = reshape(builder, sh.clone(), value);

        let z = linear_no_bias(
            builder,
            self.config.hidden_size,
            value_dim,
            p.extend(["in_proj_z"]).unwrap(),
            hidden_states.clone(),
        );
        let z = reshape(builder, sh, z);

        let b = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_v_heads,
            p.extend(["in_proj_b"]).unwrap(),
            hidden_states.clone(),
        );
        let a = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_v_heads,
            p.extend(["in_proj_a"]).unwrap(),
            hidden_states,
        );

        let beta = sigmoid(builder, b);
        let dt_bias = param(builder, &p.extend(["dt_bias"]).unwrap());
        let dt_bias = unsqueeze::<1, 2>(builder, 0, dt_bias);
        let dt_bias = broadcast(builder, shape(builder, a.clone()), dt_bias);
        let a = a + dt_bias;

        let a_log = param(builder, &p.extend(["A_log"]).unwrap());
        let a_log = unsqueeze::<1, 2>(builder, 0, a_log);
        let a_log = broadcast(builder, shape(builder, a.clone()), a_log);
        let g = -exp(builder, a_log) * softplus(builder, a);

        let rep = num_v_heads / num_k_heads;
        if rep > 1 {
            query = repeat_interleave(builder, 2, rep, query);
            key = repeat_interleave(builder, 2, rep, key);
        }

        let results = cond(
            builder,
            is_decode,
            |b, args: Vec<Var>| {
                let [query, key, value, g, beta, recurrent_state] = args.try_into().unwrap();
                let (core_attn_out_decode, out_recurrent_state_decode) = recurrent_gated_delta_rule(
                    b,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    recurrent_state,
                    head_k_dim,
                );
                vec![core_attn_out_decode, out_recurrent_state_decode]
            },
            |b, args: Vec<Var>| {
                let [query, key, value, g, beta, _recurrent_state] = args.try_into().unwrap();
                let (core_attn_out_prefill, out_recurrent_state_prefill) = chunk_gated_delta_rule(
                    b,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    head_k_dim,
                    GATED_DELTA_CHUNK_SIZE,
                );
                vec![core_attn_out_prefill, out_recurrent_state_prefill]
            },
            vec![query, key, value, g, beta, recurrent_state],
        );

        let core_attn_out = results[0].clone();
        let out_recurrent_state = results[1].clone();

        cache
            .recurrent_state
            .as_mut()
            .expect("qwen3_5 linear attention requires mutable recurrent state")[linear_layer_id] =
            out_recurrent_state;

        let flat = shape!(
            builder,
            batch_size.clone() * seq_len.clone() * num_v_heads.to_nat(builder),
            head_v_dim
        );
        let core_attn_out = reshape(builder, flat.clone(), core_attn_out);
        let z = reshape(builder, flat, z);

        // Gated RMSNorm
        let core_attn_out = rmsnorm::<2>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm"]).unwrap(),
            core_attn_out,
        ) * silu(builder, z);

        let sh = shape!(builder, batch_size, seq_len, value_dim);
        let core_attn_out = reshape(builder, sh, core_attn_out);

        linear_no_bias(
            builder,
            value_dim,
            self.config.hidden_size,
            p.extend(["out_proj"]).unwrap(),
            core_attn_out,
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
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = if self.is_full_attention_layer(layer_id) {
            let cache_layer_id = self.layer_to_cache_id[layer_id]
                .expect("full-attention layer missing KV cache index");
            self.full_attention(
                builder,
                cache_layer_id,
                attention_mask,
                cache,
                pos,
                p.extend(["self_attn"]).unwrap(),
                x,
            )
        } else {
            self.linear_attention(
                builder,
                layer_id,
                cache,
                pos,
                p.extend(["linear_attn"]).unwrap(),
                x,
            )
        };

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        x + res
    }
}

impl DynModule for Qwen3_5Model {
    fn path(&self) -> Path {
        path(vec!["qwen3_5"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, in_conv, in_recurrent]: [Var; 5] =
            args.try_into().expect("expected 5 inputs");
        let root = self.path();

        let mut cache = Cache::init(
            builder,
            &self.config,
            self.max_sequence_length,
            in_k.clone(),
            in_v,
        );
        cache.linear_state = Some(
            (0..self.num_linear_layers)
                .map(|layer_id| {
                    let layer = slice(builder, 0, layer_id, 1, in_conv.clone());
                    squeeze::<4, 3>(builder, 0, layer)
                })
                .collect(),
        );
        cache.recurrent_state = Some(
            (0..self.num_linear_layers)
                .map(|layer_id| {
                    let layer = slice(builder, 0, layer_id, 1, in_recurrent.clone());
                    squeeze::<5, 4>(builder, 0, layer)
                })
                .collect(),
        );
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k));

        let language_root = root.extend(["model", "language_model"]).unwrap();

        let mut x = embeddings(builder, language_root.extend(["embed_tokens"]).unwrap(), x);
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s, pos.clone());

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                pos.clone(),
                language_root.extend(["layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            language_root.extend(["norm"]).unwrap(),
            x,
        );

        let lm_head_weights = if self.config.tie_word_embeddings {
            vec!["model", "language_model", "embed_tokens"]
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
        let out_conv = if self.num_linear_layers == 0 {
            in_conv
        } else {
            let states = cache
                .linear_state
                .as_ref()
                .expect("qwen3_5 cache missing conv state");
            let mut iter = states.iter();
            let first = iter.next().expect("qwen3_5 conv state missing").clone();
            let mut out = unsqueeze::<3, 4>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<3, 4>(builder, 0, state.clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        let out_recurrent = if self.num_linear_layers == 0 {
            in_recurrent
        } else {
            let states = cache
                .recurrent_state
                .as_ref()
                .expect("qwen3_5 cache missing recurrent state");
            let mut iter = states.iter();
            let first = iter
                .next()
                .expect("qwen3_5 recurrent state missing")
                .clone();
            let mut out = unsqueeze::<4, 5>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<4, 5>(builder, 0, state.clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        vec![x, out_k, out_v, out_conv, out_recurrent]
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let (mut source, mut target) = llm_type(&self.config);
        let batch_size = NatExpr::Var(0);
        let num_linear_layers = NatExpr::Constant(self.num_linear_layers);
        let conv_dim = NatExpr::Constant(self.linear_conv_dim());
        let conv_cache_len = NatExpr::Constant(self.config.linear_conv_kernel_dim);
        let num_v_heads = NatExpr::Constant(self.config.linear_num_value_heads);
        let head_k_dim = NatExpr::Constant(self.config.linear_key_head_dim);
        let head_v_dim = NatExpr::Constant(self.config.linear_value_head_dim);

        let t_conv = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers.clone(),
                batch_size.clone(),
                conv_dim,
                conv_cache_len,
            ]),
        }));
        let t_recurrent = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers,
                batch_size,
                num_v_heads,
                head_k_dim,
                head_v_dim,
            ]),
        }));

        source.push(t_conv.clone());
        source.push(t_recurrent.clone());
        target.push(t_conv);
        target.push(t_recurrent);
        (source, target)
    }
}
