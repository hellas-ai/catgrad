#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig, RopeScaling};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct OlmoConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
    rope_scaling: Option<RopeScaling>,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    model_type: String,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_value_head_dim: usize,
    linear_num_value_heads: usize,
    #[serde(default)]
    linear_allow_neg_eigval: bool,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
}

impl OlmoConfig {
    fn is_hybrid(&self) -> bool {
        self.model_type == "olmo_hybrid"
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        if self.is_hybrid() && self.layer_types.len() == self.num_hidden_layers {
            self.layer_types[layer_id] == "full_attention"
        } else {
            true
        }
    }
}

impl LLMConfig for OlmoConfig {
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
        self.rope_theta
    }

    fn rope_scaling(&self) -> Option<RopeScaling> {
        self.rope_scaling.clone()
    }

    fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct OlmoModel {
    config: OlmoConfig,
    layer_to_cache_id: Vec<Option<usize>>,
    layer_to_linear_id: Vec<Option<usize>>,
    num_linear_layers: usize,
    dtype: Dtype,
    pub max_sequence_length: usize,
}

impl LLMModel for OlmoModel {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn extra_nat_input(&self, seq_len: usize) -> Option<usize> {
        Some(seq_len.div_ceil(GATED_DELTA_CHUNK_SIZE))
    }

    fn extra_nat_chunk_size(&self) -> Option<usize> {
        Some(GATED_DELTA_CHUNK_SIZE)
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let dtype = self.dtype();
        vec![
            (
                dtype,
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_head_dim(),
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_head_dim(),
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.linear_conv_dim(),
                    self.config.linear_conv_kernel_dim,
                ]),
            ),
            (
                dtype,
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

impl OlmoModel {
    pub fn new(
        config_json: &serde_json::Value,
        max_sequence_length: usize,
        dtype: Dtype,
    ) -> crate::Result<Self> {
        let config: OlmoConfig = serde_json::from_value(config_json.clone())?;
        if config.is_hybrid() {
            assert!(
                config.linear_conv_kernel_dim > 0,
                "olmo_hybrid linear_conv_kernel_dim must be > 0"
            );
        }
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
            dtype,
            max_sequence_length,
        })
    }

    fn linear_conv_dim(&self) -> usize {
        let key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim;
        let value_dim = self.config.linear_num_value_heads * self.config.linear_value_head_dim;
        key_dim * 2 + value_dim
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.config.is_full_attention_layer(layer_id)
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
        cache_layer_id: usize,
        attention_mask: Var,
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
            dim * num_kv_heads / num_heads,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );

        let v = linear_no_bias(
            builder,
            dim,
            dim * num_kv_heads / num_heads,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let q = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["q_norm"]).unwrap(),
            q,
        );
        let k = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["k_norm"]).unwrap(),
            k,
        );

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let mut q = transpose(builder, 1, 2, q);
        let mut k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        if self.config.rope_theta != 0.0 {
            q = apply_rope_embedding(
                builder,
                pos.clone(),
                head_dim,
                cache.cos.clone(),
                cache.sin.clone(),
                q,
            );
            k = apply_rope_embedding(
                builder,
                pos,
                head_dim,
                cache.cos.clone(),
                cache.sin.clone(),
                k,
            );
        };

        let (k, v) = cache.update_kv_cache(builder, cache_layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let denom = cast(builder, denom, dtype(builder, attn.clone()));
        let mut attn = attn / denom;

        let attention_mask = cast(builder, attention_mask, dtype(builder, attn.clone()));
        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["o_proj"]).unwrap(), attn)
    }

    fn linear_attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        num_chunks: Var,
        pos: Var,
        p: Path,
        hidden_states: Var,
    ) -> Var {
        let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, hidden_states.clone()));
        let linear_layer_id =
            self.layer_to_linear_id[layer_id].expect("linear-attention layer missing state index");

        let num_k_heads = self.config.linear_num_key_heads;
        let num_v_heads = self.config.linear_num_value_heads;
        let head_k_dim = self.config.linear_key_head_dim;
        let head_v_dim = self.config.linear_value_head_dim;
        let max_num_chunks = self.max_sequence_length.div_ceil(GATED_DELTA_CHUNK_SIZE);
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let cache_len = self.config.linear_conv_kernel_dim;

        let conv_state = cache
            .linear_cache
            .as_ref()
            .expect("olmo linear attention requires linear cache")[linear_layer_id][0]
            .clone();
        let recurrent_state = cache
            .linear_cache
            .as_ref()
            .expect("olmo linear attention requires linear cache")[linear_layer_id][1]
            .clone();

        let q = linear_no_bias(
            builder,
            self.config.hidden_size,
            key_dim,
            p.extend(["q_proj"]).unwrap(),
            hidden_states.clone(),
        );
        let k = linear_no_bias(
            builder,
            self.config.hidden_size,
            key_dim,
            p.extend(["k_proj"]).unwrap(),
            hidden_states.clone(),
        );
        let v = linear_no_bias(
            builder,
            self.config.hidden_size,
            value_dim,
            p.extend(["v_proj"]).unwrap(),
            hidden_states.clone(),
        );

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q_conv_weight = param(builder, &p.extend(["q_conv1d", "weight"]).unwrap());
        let k_conv_weight = param(builder, &p.extend(["k_conv1d", "weight"]).unwrap());
        let v_conv_weight = param(builder, &p.extend(["v_conv1d", "weight"]).unwrap());

        let conv_states = split(builder, 1, &[key_dim, key_dim, value_dim], conv_state);
        let q_state = conv_states[0].clone();
        let k_state = conv_states[1].clone();
        let v_state = conv_states[2].clone();

        let pos_u32 = nat_to_u32(builder, pos);
        let zero_pos = constant(builder, 0u32, &shape(builder, pos_u32.clone()));
        let is_decode = gt(builder, pos_u32, zero_pos);

        let results = cond(
            builder,
            is_decode.clone(),
            |b, args: Vec<Var>| {
                let [
                    q,
                    k,
                    v,
                    q_state,
                    k_state,
                    v_state,
                    q_w,
                    k_w,
                    v_w,
                    _batch_size,
                    seq_len,
                ] = args.try_into().unwrap();

                let q_new = concat(b, 2, q_state, q);
                let q_state_out = slice(b, 2, seq_len.clone(), cache_len, q_new);
                let q_out = silu(
                    b,
                    padded_depthwise_conv1d_no_bias_param(
                        b,
                        q_w,
                        cache_len,
                        q_state_out.clone(),
                        seq_len.clone(),
                    ),
                );

                let k_new = concat(b, 2, k_state, k);
                let k_state_out = slice(b, 2, seq_len.clone(), cache_len, k_new);
                let k_out = silu(
                    b,
                    padded_depthwise_conv1d_no_bias_param(
                        b,
                        k_w,
                        cache_len,
                        k_state_out.clone(),
                        seq_len.clone(),
                    ),
                );

                let v_new = concat(b, 2, v_state, v);
                let v_state_out = slice(b, 2, seq_len.clone(), cache_len, v_new);
                let v_out = silu(
                    b,
                    padded_depthwise_conv1d_no_bias_param(
                        b,
                        v_w,
                        cache_len,
                        v_state_out.clone(),
                        seq_len,
                    ),
                );

                vec![q_out, k_out, v_out, q_state_out, k_state_out, v_state_out]
            },
            |b, args: Vec<Var>| {
                let [
                    q,
                    k,
                    v,
                    _q_state,
                    _k_state,
                    _v_state,
                    q_w,
                    k_w,
                    v_w,
                    batch_size,
                    seq_len,
                ] = args.try_into().unwrap();

                let q_out = silu(
                    b,
                    depthwise_conv1d_no_bias_param(b, q_w, cache_len, cache_len - 1, q.clone()),
                );
                let k_out = silu(
                    b,
                    depthwise_conv1d_no_bias_param(b, k_w, cache_len, cache_len - 1, k.clone()),
                );
                let v_out = silu(
                    b,
                    depthwise_conv1d_no_bias_param(b, v_w, cache_len, cache_len - 1, v.clone()),
                );

                let q_zeros = zeros(
                    b,
                    &shape!(b, batch_size, key_dim, cache_len),
                    dtype(b, q.clone()),
                );
                let q_state_out = slice(b, 2, seq_len.clone(), cache_len, concat(b, 2, q_zeros, q));

                let k_zeros = zeros(
                    b,
                    &shape!(b, batch_size, key_dim, cache_len),
                    dtype(b, k.clone()),
                );
                let k_state_out = slice(b, 2, seq_len.clone(), cache_len, concat(b, 2, k_zeros, k));

                let v_zeros = zeros(
                    b,
                    &shape!(b, batch_size, value_dim, cache_len),
                    dtype(b, v.clone()),
                );
                let v_state_out = slice(b, 2, seq_len, cache_len, concat(b, 2, v_zeros, v));

                vec![q_out, k_out, v_out, q_state_out, k_state_out, v_state_out]
            },
            vec![
                q,
                k,
                v,
                q_state,
                k_state,
                v_state,
                q_conv_weight,
                k_conv_weight,
                v_conv_weight,
                batch_size.clone(),
                seq_len.clone(),
            ],
        );

        let q = results[0].clone();
        let k = results[1].clone();
        let v = results[2].clone();
        let q_state_out = results[3].clone();
        let k_state_out = results[4].clone();
        let v_state_out = results[5].clone();

        let out_conv_state = concat(builder, 1, q_state_out, k_state_out);
        let out_conv_state = concat(builder, 1, out_conv_state, v_state_out);

        cache
            .linear_cache
            .as_mut()
            .expect("olmo linear attention requires mutable linear cache")[linear_layer_id][0] =
            out_conv_state;

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let sh = shape!(builder, batch_size, seq_len, num_k_heads, head_k_dim);
        let mut query = reshape(builder, sh.clone(), q);
        let mut key = reshape(builder, sh, k);
        let sh = shape!(builder, batch_size, seq_len, num_v_heads, head_v_dim);
        let value = reshape(builder, sh, v);

        let b = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_v_heads,
            p.extend(["b_proj"]).unwrap(),
            hidden_states.clone(),
        );
        let a = linear_no_bias(
            builder,
            self.config.hidden_size,
            num_v_heads,
            p.extend(["a_proj"]).unwrap(),
            hidden_states.clone(),
        );

        let mut beta = sigmoid(builder, b);
        if self.config.linear_allow_neg_eigval {
            let scale = constant(builder, 2.0, &shape(builder, beta.clone()));
            let scale = cast(builder, scale, dtype(builder, beta.clone()));
            beta = beta * scale;
        }
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
                let [query, key, value, g, beta, recurrent_state, _num_chunks] =
                    args.try_into().unwrap();
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
                let [query, key, value, g, beta, _recurrent_state, num_chunks] =
                    args.try_into().unwrap();
                let (core_attn_out_prefill, out_recurrent_state_prefill) = chunk_gated_delta_rule(
                    b,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    head_k_dim,
                    num_chunks,
                    max_num_chunks,
                );
                vec![core_attn_out_prefill, out_recurrent_state_prefill]
            },
            vec![query, key, value, g, beta, recurrent_state, num_chunks],
        );

        let core_attn_out = results[0].clone();
        let out_recurrent_state = results[1].clone();

        cache
            .linear_cache
            .as_mut()
            .expect("olmo linear attention requires mutable linear cache")[linear_layer_id][1] =
            out_recurrent_state;

        let flat = shape!(
            builder,
            batch_size.clone() * seq_len.clone() * num_v_heads.to_nat(builder),
            head_v_dim
        );
        let core_attn_out = reshape(builder, flat.clone(), core_attn_out);

        // Gated RMSNorm
        let gate = linear_no_bias(
            builder,
            self.config.hidden_size,
            value_dim,
            p.extend(["g_proj"]).unwrap(),
            hidden_states,
        );
        let gate = reshape(builder, flat, gate);
        let core_attn_out =
            rmsnorm::<2>(builder, 1e-5, p.extend(["o_norm"]).unwrap(), core_attn_out)
                * silu(builder, gate);

        let sh = shape!(builder, batch_size, seq_len, value_dim);
        let core_attn_out = reshape(builder, sh, core_attn_out);

        linear_no_bias(
            builder,
            value_dim,
            self.config.hidden_size,
            p.extend(["o_proj"]).unwrap(),
            core_attn_out,
        )
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        num_chunks: Var,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        if self.is_full_attention_layer(layer_id) {
            let res = x.clone();
            let cache_layer_id = self.layer_to_cache_id[layer_id]
                .expect("full-attention layer missing KV cache index");
            let x = self.attention(
                builder,
                cache_layer_id,
                attention_mask,
                cache,
                pos,
                p.extend(["self_attn"]).unwrap(),
                x,
            );
            let x = rmsnorm::<3>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["post_attention_layernorm"]).unwrap(),
                x,
            );

            let x = res + x;
            let res = x.clone();

            let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);

            let x = rmsnorm::<3>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["post_feedforward_layernorm"]).unwrap(),
                x,
            );
            x + res
        } else {
            let res = x.clone();
            let x = rmsnorm::<3>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["input_layernorm"]).unwrap(),
                x,
            );
            let x = self.linear_attention(
                builder,
                layer_id,
                cache,
                num_chunks,
                pos,
                p.extend(["linear_attn"]).unwrap(),
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
}

impl DynModule for OlmoModel {
    fn path(&self) -> Path {
        path(vec!["olmo"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [
            x,
            in_k,
            in_v,
            in_conv,
            in_recurrent,
            max_positions,
            num_chunks,
        ]: [Var; 7] = args.try_into().expect("expected 7 inputs");
        let root = self.path();

        let mut cache = Cache::init(
            builder,
            &self.config,
            max_positions.clone(),
            max_positions,
            in_k.clone(),
            in_v,
        );
        cache.linear_cache = Some(
            (0..self.num_linear_layers)
                .map(|layer_id| {
                    let conv_layer = slice(builder, 0, layer_id, 1, in_conv.clone());
                    let conv_state = squeeze::<4, 3>(builder, 0, conv_layer);
                    let recurrent_layer = slice(builder, 0, layer_id, 1, in_recurrent.clone());
                    let recurrent_state = squeeze::<5, 4>(builder, 0, recurrent_layer);
                    vec![conv_state, recurrent_state]
                })
                .collect(),
        );
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
                num_chunks.clone(),
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
        let out_conv = if self.num_linear_layers == 0 {
            in_conv
        } else {
            let states = cache
                .linear_cache
                .as_ref()
                .expect("olmo cache missing linear cache");
            let mut iter = states.iter();
            let first = iter.next().expect("olmo conv state missing")[0].clone();
            let mut out = unsqueeze::<3, 4>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<3, 4>(builder, 0, state[0].clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        let out_recurrent = if self.num_linear_layers == 0 {
            in_recurrent
        } else {
            let states = cache
                .linear_cache
                .as_ref()
                .expect("olmo cache missing linear cache");
            let mut iter = states.iter();
            let first = iter.next().expect("olmo recurrent state missing")[1].clone();
            let mut out = unsqueeze::<4, 5>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<4, 5>(builder, 0, state[1].clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        vec![x, out_k, out_v, out_conv, out_recurrent]
    }

    // This should return the *detailed* type of the model
    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let (mut source, mut target) = llm_type(&self.config, self.dtype());
        let max_positions = source.pop().expect("olmo missing max_positions nat input");
        let batch_size = NatExpr::Var(0);
        let num_linear_layers = NatExpr::Constant(self.num_linear_layers);
        let conv_dim = NatExpr::Constant(self.linear_conv_dim());
        let conv_cache_len = NatExpr::Constant(self.config.linear_conv_kernel_dim);
        let num_v_heads = NatExpr::Constant(self.config.linear_num_value_heads);
        let head_k_dim = NatExpr::Constant(self.config.linear_key_head_dim);
        let head_v_dim = NatExpr::Constant(self.config.linear_value_head_dim);

        let t_conv = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers.clone(),
                batch_size.clone(),
                conv_dim,
                conv_cache_len,
            ]),
        }));
        let t_recurrent = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
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
        source.push(max_positions);
        source.push(Type::Nat(NatExpr::Var(4)));
        target.push(t_conv);
        target.push(t_recurrent);
        (source, target)
    }
}
