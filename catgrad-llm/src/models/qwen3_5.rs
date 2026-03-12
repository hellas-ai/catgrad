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

const GATED_DELTA_CHUNK_SIZE: usize = 64;

fn pad_sequence<const N: usize>(builder: &Builder, x: Var, padded_seq_len: usize) -> Var {
    let mut sh = unpack::<N>(builder, shape(builder, x.clone()));
    sh[2] = padded_seq_len.to_nat(builder);
    let sh = pack(builder, sh);
    let zeros = zeros(builder, &sh);
    let x = concat(builder, 2, x, zeros);
    slice(builder, 2, 0, padded_seq_len, x)
}

fn softplus(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let one = constant(builder, 1.0, &sh);
    log(builder, one + exp(builder, x))
}

fn l2norm(builder: &Builder, x: Var, eps: f32) -> Var {
    let sh = shape(builder, x.clone());
    let denom = sum(builder, x.clone() * x.clone());
    let eps = constant(builder, eps, &shape(builder, denom.clone()));
    let denom = sqrt(builder, denom + eps);
    let denom = broadcast(builder, sh, denom);
    x / denom
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

    // Delta rule for prefill stage
    fn chunk_gated_delta_rule(
        &self,
        builder: &Builder,
        query: Var,
        key: Var,
        value: Var,
        g: Var,
        beta: Var,
        head_k_dim: usize,
        chunk_size: usize,
    ) -> (Var, Var) {
        let query = l2norm(builder, query, 1e-6);
        let key = l2norm(builder, key, 1e-6);

        let query = transpose(builder, 1, 2, query);
        let key = transpose(builder, 1, 2, key);
        let value = transpose(builder, 1, 2, value);
        let beta = transpose(builder, 1, 2, beta);
        let g = transpose(builder, 1, 2, g);

        let [_, _, original_seq_len, _] = unpack::<4>(builder, shape(builder, query.clone()));
        // Temporary prefill-only assumption: sequence length <= chunk_size (64). FIXME
        let padded_seq_len = chunk_size;

        let query = pad_sequence::<4>(builder, query, padded_seq_len);
        let key = pad_sequence::<4>(builder, key, padded_seq_len);
        let value = pad_sequence::<4>(builder, value, padded_seq_len);
        let beta = pad_sequence::<3>(builder, beta, padded_seq_len);
        let g = pad_sequence::<3>(builder, g, padded_seq_len);

        let q_shape = shape(builder, query.clone());
        let q_scale = constant(builder, 1.0 / (head_k_dim as f32).sqrt(), &q_shape);
        let query = query * q_scale;

        let beta = unsqueeze::<3, 4>(builder, 3, beta);

        let v_beta = broadcast(builder, shape(builder, value.clone()), beta.clone());
        let v_beta = value.clone() * v_beta;

        let k_beta = broadcast(builder, shape(builder, key.clone()), beta);
        let k_beta = key.clone() * k_beta;

        let [batch_size, num_heads, padded_seq_len_nat, k_head_dim] =
            unpack::<4>(builder, shape(builder, query.clone()));

        let sh = shape!(builder, batch_size, num_heads, 1, chunk_size, k_head_dim);
        let query = reshape(builder, sh.clone(), query);
        let key = reshape(builder, sh.clone(), key);
        let value = reshape(builder, sh.clone(), value);
        let k_beta = reshape(builder, sh.clone(), k_beta);
        let v_beta = reshape(builder, sh, v_beta);
        let g = reshape(
            builder,
            shape!(builder, batch_size, num_heads, 1, chunk_size,),
            g,
        );

        let g = cumsum::<4>(builder, g);

        let decay_shape = shape!(builder, batch_size, num_heads, 1, chunk_size, chunk_size);
        let g_i = unsqueeze::<4, 5>(builder, 4, g.clone());
        let g_i = broadcast(builder, decay_shape.clone(), g_i);
        let g_j = unsqueeze::<4, 5>(builder, 3, g.clone());
        let g_j = broadcast(builder, decay_shape.clone(), g_j);

        let decay_mask = exp(builder, g_i - g_j);
        let lower_mask = tril_mask(builder, padded_seq_len_nat.clone(), 0);
        let lower_mask = broadcast(builder, decay_shape.clone(), lower_mask);
        let decay_mask = where_cond(
            builder,
            lower_mask,
            decay_mask,
            zeros(builder, &decay_shape),
        );

        let query = squeeze::<5, 4>(builder, 2, query);
        let key = squeeze::<5, 4>(builder, 2, key);
        let _value = squeeze::<5, 4>(builder, 2, value);
        let k_beta = squeeze::<5, 4>(builder, 2, k_beta);
        let v_beta = squeeze::<5, 4>(builder, 2, v_beta);
        let decay_mask = squeeze::<5, 4>(builder, 2, decay_mask);
        let padded_seq_len = padded_seq_len_nat;

        let tk = transpose(builder, 2, 3, key.clone());
        let mut attn = matmul(builder, k_beta.clone(), tk.clone());
        attn = -(attn * decay_mask.clone());

        let mask_diag0 = triu_mask(builder, padded_seq_len.clone(), 0);
        let mask_diag0 = broadcast(builder, shape(builder, attn.clone()), mask_diag0);
        attn = masked_fill(builder, mask_diag0, 0.0, attn);

        for i in 1..GATED_DELTA_CHUNK_SIZE {
            let row = slice(builder, 2, i, 1, attn.clone());
            let row_prefix = slice(builder, 3, 0, i, row.clone());
            let sub = slice(builder, 2, 0, i, attn.clone());
            let sub = slice(builder, 3, 0, i, sub);

            let update = matmul(builder, row_prefix.clone(), sub);
            let new_row_prefix = row_prefix + update;

            let new_row = if i < GATED_DELTA_CHUNK_SIZE {
                let row_suffix = slice(builder, 3, i, GATED_DELTA_CHUNK_SIZE - i, row);
                concat(builder, 3, new_row_prefix, row_suffix)
            } else {
                new_row_prefix
            };

            let top = slice(builder, 2, 0, i, attn.clone());
            let updated = concat(builder, 2, top, new_row);
            attn = if i + 1 < GATED_DELTA_CHUNK_SIZE {
                let bottom = slice(builder, 2, i + 1, GATED_DELTA_CHUNK_SIZE - (i + 1), attn);
                concat(builder, 2, updated, bottom)
            } else {
                updated
            };
        }

        let eye = eye(builder, padded_seq_len.clone(), Dtype::F32);
        let eye = unsqueeze::<2, 3>(builder, 0, eye);
        let eye = unsqueeze::<3, 4>(builder, 0, eye);
        let eye = broadcast(builder, shape(builder, attn.clone()), eye);
        attn = attn + eye;

        let value = matmul(builder, attn.clone(), v_beta);

        let g_chunk = squeeze::<4, 3>(builder, 2, g);
        let g_exp = exp(builder, g_chunk.clone());
        let g_exp = unsqueeze::<3, 4>(builder, 3, g_exp);
        let g_exp = broadcast(builder, shape(builder, k_beta.clone()), g_exp);
        let _k_cumdecay = matmul(builder, attn, k_beta * g_exp);

        let [_, _, _, value_head_dim] = unpack::<4>(builder, shape(builder, value.clone()));
        let zero_recurrent_state = zeros(
            builder,
            &shape!(builder, batch_size, num_heads, k_head_dim, value_head_dim),
        );

        let mut attn = matmul(builder, query, tk);
        attn = attn * decay_mask;

        let mask_diag1 = triu_mask(builder, padded_seq_len, 1);
        let mask_diag1 = broadcast(builder, shape(builder, attn.clone()), mask_diag1);
        attn = masked_fill(builder, mask_diag1, 0.0, attn);

        // with the current single-chunk prefill assumption, the initial recurrent state is zero,
        // so `v_prime` and `attn_inter` vanish and the chunk update reduces to `attn @ value`.
        let core_attn_out = matmul(builder, attn, value.clone());

        // last_recurrent_state = last_recurrent_state * g[..., -1].exp()
        //     + (k_i * (g[..., -1, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
        // The first term is zero here because the prefill branch always starts from zeros.
        let g_last = slice(builder, 2, chunk_size - 1, 1, g_chunk.clone());
        let g_last = broadcast(builder, shape(builder, g_chunk.clone()), g_last);
        let state_decay = exp(builder, g_last - g_chunk);
        let state_decay = unsqueeze::<3, 4>(builder, 3, state_decay);
        let state_decay = broadcast(builder, shape(builder, key.clone()), state_decay);
        let weighted_key = key * state_decay;
        let last_recurrent_state =
            zero_recurrent_state + matmul(builder, transpose(builder, 2, 3, weighted_key), value);

        let core_attn_out = transpose(builder, 1, 2, core_attn_out);
        let core_attn_out = slice(builder, 1, 0, original_seq_len, core_attn_out);
        (core_attn_out, last_recurrent_state)
    }

    // Delta rule for recurrent stage
    fn recurrent_gated_delta_rule(
        &self,
        builder: &Builder,
        query: Var,
        key: Var,
        value: Var,
        g: Var,
        beta: Var,
        initial_state: Var,
        head_k_dim: usize,
    ) -> (Var, Var) {
        let query = l2norm(builder, query, 1e-6);
        let key = l2norm(builder, key, 1e-6);

        let query = transpose(builder, 1, 2, query);
        let key = transpose(builder, 1, 2, key);
        let value = transpose(builder, 1, 2, value);
        let beta = transpose(builder, 1, 2, beta);
        let g = transpose(builder, 1, 2, g);

        let [batch_size, num_heads, sequence_length, _] =
            unpack::<4>(builder, shape(builder, key.clone()));
        let [_, _, _, value_head_dim] = unpack::<4>(builder, shape(builder, value.clone()));
        let q_shape = shape(builder, query.clone());
        let q_scale = constant(builder, 1.0 / (head_k_dim as f32).sqrt(), &q_shape);
        let query = query * q_scale;

        let core_attn_out = zeros(
            builder,
            &shape!(
                builder,
                batch_size,
                num_heads,
                sequence_length,
                value_head_dim
            ),
        );
        let mut last_recurrent_state = initial_state;

        // for i in range(sequence_length):
        //     q_t = query[:, :, i]
        //     k_t = key[:, :, i]
        //     v_t = value[:, :, i]
        //     g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        //     beta_t = beta[:, :, i].unsqueeze(-1)
        //
        // Decode with KV cache only selects this branch when `seq_len == 1`, so we build the
        // single-token `i = 0` recurrence directly and keep the full-length output shape only so
        // it can be merged with the prefill branch using `where_broadcast`.
        let q_t = slice(builder, 2, 0, 1, query);
        let q_t = squeeze::<4, 3>(builder, 2, q_t);
        let k_t = slice(builder, 2, 0, 1, key);
        let k_t = squeeze::<4, 3>(builder, 2, k_t);
        let v_t = slice(builder, 2, 0, 1, value);
        let v_t = squeeze::<4, 3>(builder, 2, v_t);
        let g_t = slice(builder, 2, 0, 1, g);
        let g_t = squeeze::<3, 2>(builder, 2, g_t);
        let beta_t = slice(builder, 2, 0, 1, beta);
        let beta_t = squeeze::<3, 2>(builder, 2, beta_t);

        // last_recurrent_state = last_recurrent_state * g_t
        let g_t = exp(builder, g_t);
        let g_t = unsqueeze::<2, 3>(builder, 2, g_t);
        let g_t = unsqueeze::<3, 4>(builder, 3, g_t);
        let g_t = broadcast(builder, shape(builder, last_recurrent_state.clone()), g_t);
        last_recurrent_state = last_recurrent_state * g_t;

        // kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        let k_t_u = unsqueeze::<3, 4>(builder, 2, k_t.clone());
        let kv_mem = matmul(builder, k_t_u, last_recurrent_state.clone());
        let kv_mem = squeeze::<4, 3>(builder, 2, kv_mem);

        // delta = (v_t - kv_mem) * beta_t
        let beta_t = unsqueeze::<2, 3>(builder, 2, beta_t);
        let beta_t = broadcast(builder, shape(builder, v_t.clone()), beta_t);
        let delta = (v_t - kv_mem) * beta_t;

        // last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        let k_t_u = unsqueeze::<3, 4>(builder, 3, k_t);
        let k_t_u = broadcast(builder, shape(builder, last_recurrent_state.clone()), k_t_u);
        let delta_u = unsqueeze::<3, 4>(builder, 2, delta);
        let delta_u = broadcast(
            builder,
            shape(builder, last_recurrent_state.clone()),
            delta_u,
        );
        last_recurrent_state = last_recurrent_state + k_t_u * delta_u;

        // core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        let q_t_u = unsqueeze::<3, 4>(builder, 2, q_t);
        let core_attn_step = matmul(builder, q_t_u, last_recurrent_state.clone());
        let core_attn_step = squeeze::<4, 3>(builder, 2, core_attn_step);
        let core_attn_step = unsqueeze::<3, 4>(builder, 2, core_attn_step);
        let core_attn_step = broadcast(
            builder,
            shape(builder, core_attn_out.clone()),
            core_attn_step,
        );

        // core_attn_out = core_attn_out.transpose(1, 2).contiguous()
        let positions = arange(builder, sequence_length);
        let first_pos = constant(builder, 0u32, &shape(builder, positions.clone()));
        let first_pos = eq(builder, positions, first_pos);
        let first_pos = unsqueeze::<1, 2>(builder, 0, first_pos);
        let first_pos = unsqueeze::<2, 3>(builder, 0, first_pos);
        let first_pos = unsqueeze::<3, 4>(builder, 3, first_pos);
        let core_attn_out = where_broadcast(builder, first_pos, core_attn_step, core_attn_out);
        let core_attn_out = transpose(builder, 1, 2, core_attn_out);
        (core_attn_out, last_recurrent_state)
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
                let (core_attn_out_decode, out_recurrent_state_decode) = self
                    .recurrent_gated_delta_rule(
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
                let (core_attn_out_prefill, out_recurrent_state_prefill) = self
                    .chunk_gated_delta_rule(
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
