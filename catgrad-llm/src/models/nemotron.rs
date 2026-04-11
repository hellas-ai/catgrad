#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct NemotronConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    hybrid_override_pattern: String,
    mlp_bias: bool,
    layer_norm_epsilon: f32,
    tie_word_embeddings: bool,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
    ssm_state_size: usize,
    mamba_num_heads: usize,
    n_groups: usize,
    mamba_head_dim: usize,
    conv_kernel: usize,
    chunk_size: usize,
    time_step_min: f32,
    use_bias: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    Mamba,
    Attention,
    Mlp,
}

// Move to helpers
fn relu(builder: &Builder, x: Var) -> Var {
    let x = clamp(builder, x, 0.0, f32::MAX);
    x.clone() * x
}

fn grouped_rmsnorm(
    builder: &Builder,
    eps: f32,
    num_groups: usize,
    group_size: usize,
    p: Path,
    x: Var,
) -> Var {
    let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));
    let x = reshape(
        builder,
        shape!(builder, batch_size, seq_len, num_groups, group_size),
        x,
    );
    let x = rmsnorm_raw::<4>(builder, eps, x);
    let x = reshape(
        builder,
        shape!(builder, batch_size, seq_len, num_groups * group_size),
        x,
    );
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let gamma = broadcast(builder, shape(builder, x.clone()), gamma);
    x * gamma
}

impl NemotronConfig {
    fn num_attention_layers(&self) -> usize {
        self.hybrid_override_pattern
            .chars()
            .filter(|&c| c == '*')
            .count()
    }

    fn mamba_intermediate_size(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    fn mamba_conv_dim(&self) -> usize {
        self.mamba_intermediate_size() + 2 * self.n_groups * self.ssm_state_size
    }

    fn mamba_group_size(&self) -> usize {
        self.mamba_intermediate_size() / self.n_groups
    }
}

impl LLMConfig for NemotronConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_kv_layers(&self) -> usize {
        self.num_attention_layers()
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct NemotronModel {
    config: NemotronConfig,
    layer_kinds: Vec<LayerKind>,
    layer_to_cache_id: Vec<Option<usize>>,
    layer_to_mamba_id: Vec<Option<usize>>,
    num_mamba_layers: usize,
    dtype: Dtype,
}

impl LLMModel for NemotronModel {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let dtype = self.dtype();
        vec![
            (
                dtype.clone(),
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.head_dim,
                ]),
            ),
            (
                dtype.clone(),
                Shape(vec![
                    self.config.num_kv_layers(),
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.head_dim,
                ]),
            ),
            (
                dtype.clone(),
                Shape(vec![
                    self.num_mamba_layers,
                    1,
                    self.config.mamba_conv_dim(),
                    self.config.conv_kernel,
                ]),
            ),
            (
                dtype,
                Shape(vec![
                    self.num_mamba_layers,
                    1,
                    self.config.mamba_num_heads,
                    self.config.mamba_head_dim,
                    self.config.ssm_state_size,
                ]),
            ),
        ]
    }
}

impl NemotronModel {
    pub fn new(config_json: &serde_json::Value, dtype: Dtype) -> crate::Result<Self> {
        let config: NemotronConfig = serde_json::from_value(config_json.clone())?;
        let mut layer_kinds = Vec::with_capacity(config.num_hidden_layers);
        for ch in config.hybrid_override_pattern.chars() {
            let kind = match ch {
                'M' => LayerKind::Mamba,
                '*' => LayerKind::Attention,
                '-' => LayerKind::Mlp,
                _ => {
                    return Err(crate::LLMError::InvalidModelConfig(format!(
                        "nemotron_h hybrid_override_pattern contained unsupported character {ch:?}"
                    )));
                }
            };
            layer_kinds.push(kind);
        }

        let mut layer_to_cache_id = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_to_mamba_id = Vec::with_capacity(config.num_hidden_layers);
        let mut next_cache_id = 0;
        let mut next_mamba_id = 0;
        for kind in &layer_kinds {
            match kind {
                LayerKind::Attention => {
                    layer_to_cache_id.push(Some(next_cache_id));
                    layer_to_mamba_id.push(None);
                    next_cache_id += 1;
                }
                LayerKind::Mamba => {
                    layer_to_cache_id.push(None);
                    layer_to_mamba_id.push(Some(next_mamba_id));
                    next_mamba_id += 1;
                }
                LayerKind::Mlp => {
                    layer_to_cache_id.push(None);
                    layer_to_mamba_id.push(None);
                }
            }
        }

        Ok(Self {
            config,
            layer_kinds,
            layer_to_cache_id,
            layer_to_mamba_id,
            num_mamba_layers: next_mamba_id,
            dtype,
        })
    }

    fn add_bias_3d(&self, builder: &Builder, bias: Var, x: Var) -> Var {
        let bias = unsqueeze::<1, 2>(builder, 0, bias);
        let bias = unsqueeze::<2, 3>(builder, 2, bias);
        let bias = broadcast(builder, shape(builder, x.clone()), bias);
        x + bias
    }

    fn broadcast_d_param_prefill(&self, builder: &Builder, d_param: Var, target: Var) -> Var {
        let d_param = unsqueeze::<1, 2>(builder, 0, d_param);
        let d_param = unsqueeze::<2, 3>(builder, 1, d_param);
        let d_param = unsqueeze::<3, 4>(builder, 3, d_param);
        let d_param = unsqueeze::<4, 5>(builder, 4, d_param);
        broadcast(builder, shape(builder, target), d_param)
    }

    fn repeat_mamba_groups(&self, builder: &Builder, x: Var) -> Var {
        let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let heads_per_group = self.config.mamba_num_heads / self.config.n_groups;
        let x = reshape(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.n_groups,
                self.config.ssm_state_size
            ),
            x,
        );
        let x = unsqueeze::<4, 5>(builder, 3, x);
        let x = broadcast(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.n_groups,
                heads_per_group,
                self.config.ssm_state_size
            ),
            x,
        );
        reshape(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.mamba_num_heads,
                self.config.ssm_state_size
            ),
            x,
        )
    }

    fn attention(
        &self,
        builder: &Builder,
        cache_layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        p: Path,
        x: Var,
    ) -> Var {
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let rep = self.config.num_attention_heads / self.config.num_key_value_heads;

        let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let q = linear_no_bias(
            builder,
            self.config.hidden_size,
            q_dim,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let k = linear_no_bias(
            builder,
            self.config.hidden_size,
            kv_dim,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = linear_no_bias(
            builder,
            self.config.hidden_size,
            kv_dim,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let q = reshape(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.head_dim
            ),
            q,
        );
        let k = reshape(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim
            ),
            k,
        );
        let v = reshape(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim
            ),
            v,
        );

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let (k, v) = cache.update_kv_cache(builder, cache_layer_id, k, v);
        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, (self.config.head_dim as f32).sqrt(), &sh);
        let mut attn = attn / denom;
        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;
        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let attn = transpose(builder, 1, 2, attn);
        let attn = reshape(builder, shape!(builder, batch_size, seq_len, q_dim), attn);

        linear_no_bias(
            builder,
            q_dim,
            self.config.hidden_size,
            p.extend(["o_proj"]).unwrap(),
            attn,
        )
    }

    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let x = linear_b(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.mlp_bias,
            p.extend(["up_proj"]).unwrap(),
            x,
        );
        let x = relu(builder, x);
        linear_b(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            self.config.mlp_bias,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
    }

    fn mamba(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        hidden_states: Var,
    ) -> Var {
        let input_dtype = dtype(builder, hidden_states.clone());
        let mamba_dim = self.config.mamba_intermediate_size();
        let conv_dim = self.config.mamba_conv_dim();
        let projection_size = mamba_dim + conv_dim + self.config.mamba_num_heads;
        let mamba_layer_id =
            self.layer_to_mamba_id[layer_id].expect("mamba layer missing state index");
        let conv_state = cache
            .linear_cache
            .as_ref()
            .expect("nemotron_h mamba requires linear cache")[mamba_layer_id][0]
            .clone();
        let recurrent_state = cache
            .linear_cache
            .as_ref()
            .expect("nemotron_h mamba requires linear cache")[mamba_layer_id][1]
            .clone();

        let projected = linear_b(
            builder,
            self.config.hidden_size,
            projection_size,
            self.config.use_bias,
            p.extend(["in_proj"]).unwrap(),
            hidden_states.clone(),
        );
        let projected = split(
            builder,
            2,
            &[mamba_dim, conv_dim, self.config.mamba_num_heads],
            projected,
        );
        let gate = projected[0].clone();
        let hidden_bc = transpose(builder, 1, 2, projected[1].clone());
        let dt = projected[2].clone();

        let conv_weight = param(builder, &p.extend(["conv1d", "weight"]).unwrap());
        let conv_bias = param(builder, &p.extend(["conv1d", "bias"]).unwrap());
        let [batch_size, seq_len, _] = unpack::<3>(builder, shape(builder, hidden_states));
        let pos_u32 = nat_to_u32(builder, pos);
        let zero_pos = constant(builder, 0u32, &shape(builder, pos_u32.clone()));
        let is_decode = gt(builder, pos_u32, zero_pos);

        let conv_results = cond(
            builder,
            is_decode.clone(),
            |b, args: Vec<Var>| {
                let [
                    hidden_bc,
                    conv_state,
                    conv_weight,
                    conv_bias,
                    _batch_size,
                    seq_len,
                ] = args.try_into().unwrap();
                let hidden_with_state = concat(b, 2, conv_state, hidden_bc);
                let out_conv_state = slice(
                    b,
                    2,
                    seq_len.clone(),
                    self.config.conv_kernel,
                    hidden_with_state,
                );
                let mut conv_out = padded_depthwise_conv1d_no_bias_param(
                    b,
                    conv_weight,
                    self.config.conv_kernel,
                    out_conv_state.clone(),
                    seq_len,
                );
                conv_out = self.add_bias_3d(b, conv_bias, conv_out);
                let conv_out = silu(b, conv_out);
                vec![conv_out, out_conv_state]
            },
            |b, args: Vec<Var>| {
                let [
                    hidden_bc,
                    _conv_state,
                    conv_weight,
                    conv_bias,
                    batch_size,
                    seq_len,
                ] = args.try_into().unwrap();
                let mut conv_out = depthwise_conv1d_no_bias_param(
                    b,
                    conv_weight,
                    self.config.conv_kernel,
                    self.config.conv_kernel - 1,
                    hidden_bc.clone(),
                );
                conv_out = self.add_bias_3d(b, conv_bias, conv_out);
                let conv_out = silu(b, conv_out);

                let zeros_state =
                    zeros(b, &shape!(b, batch_size, conv_dim, self.config.conv_kernel));
                let out_conv_state = slice(
                    b,
                    2,
                    seq_len,
                    self.config.conv_kernel,
                    concat(b, 2, zeros_state, hidden_bc),
                );
                vec![conv_out, out_conv_state]
            },
            vec![
                hidden_bc,
                conv_state,
                conv_weight,
                conv_bias,
                batch_size.clone(),
                seq_len.clone(),
            ],
        );

        let conv_out = transpose(builder, 1, 2, conv_results[0].clone());
        cache
            .linear_cache
            .as_mut()
            .expect("nemotron_h mamba requires mutable linear cache")[mamba_layer_id][0] =
            conv_results[1].clone();

        let conv_out = split(
            builder,
            2,
            &[
                mamba_dim,
                self.config.n_groups * self.config.ssm_state_size,
                self.config.n_groups * self.config.ssm_state_size,
            ],
            conv_out,
        );
        let hidden = reshape(
            builder,
            shape!(
                builder,
                batch_size,
                seq_len,
                self.config.mamba_num_heads,
                self.config.mamba_head_dim
            ),
            cast(builder, conv_out[0].clone(), Dtype::F32),
        );
        let b_proj =
            self.repeat_mamba_groups(builder, cast(builder, conv_out[1].clone(), Dtype::F32));
        let c_proj =
            self.repeat_mamba_groups(builder, cast(builder, conv_out[2].clone(), Dtype::F32));

        let dt_bias = param(builder, &p.extend(["dt_bias"]).unwrap());
        let dt_bias = unsqueeze::<1, 2>(builder, 0, dt_bias);
        let dt_bias = broadcast(builder, shape(builder, dt.clone()), dt_bias);
        let dt = cast(builder, dt, Dtype::F32);
        let dt_bias = cast(builder, dt_bias, Dtype::F32);
        let dt = clamp(
            builder,
            softplus(builder, dt + dt_bias),
            self.config.time_step_min,
            f32::MAX,
        );
        let d_param = cast(
            builder,
            param(builder, &p.extend(["D"]).unwrap()),
            Dtype::F32,
        );
        let a_log = cast(
            builder,
            param(builder, &p.extend(["A_log"]).unwrap()),
            Dtype::F32,
        );

        let ssm_results = cond(
            builder,
            is_decode,
            |b, args: Vec<Var>| {
                let [hidden, dt, b_proj, c_proj, recurrent_state, d_param, a_log] =
                    args.try_into().unwrap();
                let [batch_size, seq_len, num_heads, _head_dim] =
                    unpack::<4>(b, shape(b, hidden.clone()));
                let state_shape = shape!(
                    b,
                    batch_size,
                    seq_len,
                    num_heads,
                    self.config.mamba_head_dim,
                    self.config.ssm_state_size
                );

                let a = -exp(b, a_log);
                let dt = unsqueeze::<3, 4>(b, 3, dt);
                let dt = unsqueeze::<4, 5>(b, 4, dt);
                let dt = broadcast(b, state_shape.clone(), dt);

                let a = unsqueeze::<1, 2>(b, 0, a);
                let a = unsqueeze::<2, 3>(b, 0, a);
                let a = unsqueeze::<3, 4>(b, 3, a);
                let a = unsqueeze::<4, 5>(b, 4, a);
                let a = broadcast(b, state_shape.clone(), a);
                let d_a = exp(b, dt.clone() * a);

                let recurrent_state = unsqueeze::<4, 5>(b, 1, recurrent_state);
                let recurrent_state = broadcast(b, state_shape.clone(), recurrent_state);
                let d_b = unsqueeze::<4, 5>(b, 3, b_proj);
                let d_b = broadcast(b, state_shape.clone(), d_b);
                let hidden_b = unsqueeze::<4, 5>(b, 4, hidden.clone());
                let hidden_b = broadcast(b, state_shape.clone(), hidden_b);

                let next_state = recurrent_state * d_a + hidden_b * (dt * d_b);
                let c_proj = unsqueeze::<4, 5>(b, 3, c_proj);
                let c_proj = broadcast(b, state_shape, c_proj);
                let mut y = sum(b, next_state.clone() * c_proj);

                let d_param = self.broadcast_d_param_prefill(b, d_param, y.clone());
                let hidden = unsqueeze::<4, 5>(b, 4, hidden);
                y = y + hidden * d_param;
                let y = squeeze::<5, 4>(b, 4, y);
                let final_state = slice(b, 1, 0, 1, next_state);
                let final_state = squeeze::<5, 4>(b, 1, final_state);

                vec![y, final_state]
            },
            |b, args: Vec<Var>| {
                let [hidden, dt, b_proj, c_proj, _recurrent_state, d_param, a_log] =
                    args.try_into().unwrap();
                let [batch_size, seq_len, num_heads, _head_dim] =
                    unpack::<4>(b, shape(b, hidden.clone()));

                let a = -exp(b, a_log);
                let a = unsqueeze::<1, 2>(b, 0, a);
                let a = unsqueeze::<2, 3>(b, 0, a);
                let a = broadcast(b, shape(b, dt.clone()), a);
                let g = dt.clone() * a;
                let g = transpose(b, 1, 2, g);
                let g = cumsum::<3>(b, g);
                let decay_shape = shape!(b, batch_size, num_heads, seq_len, seq_len);
                let g_q = unsqueeze::<3, 4>(b, 3, g.clone());
                let g_q = broadcast(b, decay_shape.clone(), g_q);
                let g_k = unsqueeze::<3, 4>(b, 2, g.clone());
                let g_k = broadcast(b, decay_shape.clone(), g_k);
                let lower_mask = tril_mask(b, seq_len.clone(), 0);
                let lower_mask = broadcast(b, decay_shape.clone(), lower_mask);
                let neg_inf = constant(b, f32::MIN, &decay_shape);
                let decay = exp(b, where_broadcast(b, lower_mask, g_q - g_k, neg_inf));

                let dt = transpose(b, 1, 2, dt);
                let dt_src = unsqueeze::<3, 4>(b, 2, dt.clone());
                let dt_src = broadcast(b, decay_shape, dt_src);

                let b_proj = transpose(b, 1, 2, b_proj);
                let c_proj = transpose(b, 1, 2, c_proj);
                let flat_bh = batch_size.clone() * num_heads.clone();
                let proj_shape = shape!(b, flat_bh, seq_len, self.config.ssm_state_size);
                let c_proj_flat = reshape(b, proj_shape.clone(), c_proj);
                let b_proj_flat = reshape(b, proj_shape, b_proj);
                let pair_bc = matmul(b, c_proj_flat, transpose(b, 1, 2, b_proj_flat.clone()));
                let pair_bc = reshape(
                    b,
                    shape!(b, batch_size, num_heads, seq_len, seq_len),
                    pair_bc,
                );

                let weights = decay * dt_src * pair_bc;
                let hidden_seq = hidden;
                let hidden_heads = transpose(b, 1, 2, hidden_seq.clone());
                let weights = reshape(b, shape!(b, flat_bh, seq_len.clone(), seq_len), weights);
                let hidden_flat = reshape(
                    b,
                    shape!(b, flat_bh, seq_len, self.config.mamba_head_dim),
                    hidden_heads.clone(),
                );
                let mut y = matmul(b, weights, hidden_flat);
                y = reshape(
                    b,
                    shape!(
                        b,
                        batch_size,
                        num_heads,
                        seq_len,
                        self.config.mamba_head_dim
                    ),
                    y,
                );
                y = transpose(b, 1, 2, y);
                let mut y = unsqueeze::<4, 5>(b, 4, y);

                let d_param = self.broadcast_d_param_prefill(b, d_param, y.clone());
                let hidden = unsqueeze::<4, 5>(b, 4, hidden_seq);
                y = y + hidden * d_param;
                let y = squeeze::<5, 4>(b, 4, y);

                let zero_g = zeros(b, &shape!(b, batch_size, num_heads, 1));
                let g_last = slice(b, 2, seq_len.clone(), 1, concat(b, 2, zero_g, g.clone()));
                let g_last = broadcast(b, shape(b, g.clone()), g_last);
                let state_decay = exp(b, g_last - g);
                let state_coeff = state_decay * dt;
                let state_coeff = unsqueeze::<3, 4>(b, 3, state_coeff);
                let state_coeff = broadcast(b, shape(b, hidden_heads.clone()), state_coeff);
                let hidden_state = transpose(b, 2, 3, hidden_heads * state_coeff);
                let hidden_state = reshape(
                    b,
                    shape!(b, flat_bh, self.config.mamba_head_dim, seq_len),
                    hidden_state,
                );
                let final_state = matmul(b, hidden_state, b_proj_flat);
                let final_state = reshape(
                    b,
                    shape!(
                        b,
                        batch_size,
                        num_heads,
                        self.config.mamba_head_dim,
                        self.config.ssm_state_size
                    ),
                    final_state,
                );
                vec![y, final_state]
            },
            vec![hidden, dt, b_proj, c_proj, recurrent_state, d_param, a_log],
        );

        cache
            .linear_cache
            .as_mut()
            .expect("nemotron_h mamba requires mutable linear cache")[mamba_layer_id][1] =
            ssm_results[1].clone();

        let y = reshape(
            builder,
            shape!(builder, batch_size, seq_len, mamba_dim),
            ssm_results[0].clone(),
        );
        let gate = silu(builder, cast(builder, gate, Dtype::F32));
        let y = y * gate;
        let y = grouped_rmsnorm(
            builder,
            self.config.layer_norm_epsilon,
            self.config.n_groups,
            self.config.mamba_group_size(),
            p.extend(["norm"]).unwrap(),
            y,
        );
        let y = cast(builder, y, input_dtype);

        linear_b(
            builder,
            mamba_dim,
            self.config.hidden_size,
            self.config.use_bias,
            p.extend(["out_proj"]).unwrap(),
            y,
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
        let residual = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.layer_norm_epsilon,
            p.extend(["norm"]).unwrap(),
            x,
        );
        let x = match self.layer_kinds[layer_id] {
            LayerKind::Attention => {
                let cache_layer_id = self.layer_to_cache_id[layer_id]
                    .expect("attention layer missing KV cache index");
                self.attention(
                    builder,
                    cache_layer_id,
                    attention_mask,
                    cache,
                    p.extend(["mixer"]).unwrap(),
                    x,
                )
            }
            LayerKind::Mlp => self.mlp(builder, p.extend(["mixer"]).unwrap(), x),
            LayerKind::Mamba => self.mamba(
                builder,
                layer_id,
                cache,
                pos,
                p.extend(["mixer"]).unwrap(),
                x,
            ),
        };
        residual + x
    }

    fn init_cache(
        &self,
        builder: &Builder,
        in_k: Var,
        in_v: Var,
        in_conv: Var,
        in_ssm: Var,
        max_positions: Var,
    ) -> Cache {
        let mut cache = Cache::init(
            builder,
            &self.config,
            max_positions.clone(),
            max_positions,
            in_k,
            in_v,
        );
        cache.linear_cache = Some(
            (0..self.num_mamba_layers)
                .map(|layer_id| {
                    let conv_layer = slice(builder, 0, layer_id, 1, in_conv.clone());
                    let conv_state = squeeze::<4, 3>(builder, 0, conv_layer);
                    let ssm_layer = slice(builder, 0, layer_id, 1, in_ssm.clone());
                    let ssm_state = squeeze::<5, 4>(builder, 0, ssm_layer);
                    vec![conv_state, ssm_state]
                })
                .collect(),
        );
        cache
    }

    fn collect_mamba_states(&self, builder: &Builder, cache: &Cache) -> (Var, Var) {
        let states = cache
            .linear_cache
            .as_ref()
            .expect("nemotron_h cache missing mamba cache");
        let mut iter = states.iter();
        let first = iter.next().expect("nemotron_h conv state missing")[0].clone();
        let mut out_conv = unsqueeze::<3, 4>(builder, 0, first);
        for state in iter {
            let state = unsqueeze::<3, 4>(builder, 0, state[0].clone());
            out_conv = concat(builder, 0, out_conv, state);
        }

        let states = cache
            .linear_cache
            .as_ref()
            .expect("nemotron_h cache missing mamba cache");
        let mut iter = states.iter();
        let first = iter.next().expect("nemotron_h ssm state missing")[1].clone();
        let mut out_ssm = unsqueeze::<4, 5>(builder, 0, first);
        for state in iter {
            let state = unsqueeze::<4, 5>(builder, 0, state[1].clone());
            out_ssm = concat(builder, 0, out_ssm, state);
        }

        (out_conv, out_ssm)
    }
}

impl DynModule for NemotronModel {
    fn path(&self) -> Path {
        path(vec!["nemotron_h"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, in_conv, in_ssm, max_positions]: [Var; 6] =
            args.try_into().expect("expected 6 inputs");
        let root = self.path();
        let in_v_out = in_v.clone();
        let mut cache =
            self.init_cache(builder, in_k.clone(), in_v, in_conv, in_ssm, max_positions);
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k.clone()));

        let mut x = embeddings(builder, root.extend(["backbone", "embeddings"]).unwrap(), x);
        let [_, seq_len, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, seq_len, pos.clone());

        for layer_id in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                layer_id,
                attention_mask.clone(),
                &mut cache,
                pos.clone(),
                root.extend(["backbone", "layers", &layer_id.to_string()])
                    .unwrap(),
                x,
            );
        }

        x = rmsnorm::<3>(
            builder,
            self.config.layer_norm_epsilon,
            root.extend(["backbone", "norm_f"]).unwrap(),
            x,
        );

        let lm_head = if self.config.tie_word_embeddings {
            root.extend(["backbone", "embeddings"]).unwrap()
        } else {
            root.extend(["lm_head"]).unwrap()
        };
        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            lm_head,
            x,
        );

        x = argmax(builder, x);
        let (out_k, out_v) = if self.config.num_kv_layers() == 0 {
            (in_k, in_v_out)
        } else {
            cache.get_kv_cache(builder)
        };
        let (out_conv, out_ssm) = self.collect_mamba_states(builder, &cache);
        vec![x, out_k, out_v, out_conv, out_ssm]
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let (mut source, mut target) = llm_type(&self.config, self.dtype());
        let max_positions = source
            .pop()
            .expect("nemotron_h missing max_positions nat input");
        let batch_size = NatExpr::Var(0);
        let num_mamba_layers = NatExpr::Constant(self.num_mamba_layers);
        let conv_dim = NatExpr::Constant(self.config.mamba_conv_dim());
        let conv_kernel = NatExpr::Constant(self.config.conv_kernel);
        let num_mamba_heads = NatExpr::Constant(self.config.mamba_num_heads);
        let mamba_head_dim = NatExpr::Constant(self.config.mamba_head_dim);
        let ssm_state_size = NatExpr::Constant(self.config.ssm_state_size);

        let t_conv = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                num_mamba_layers.clone(),
                batch_size.clone(),
                conv_dim,
                conv_kernel,
            ]),
        }));
        let t_ssm = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(self.dtype()),
            shape: ShapeExpr::Shape(vec![
                num_mamba_layers,
                batch_size,
                num_mamba_heads,
                mamba_head_dim,
                ssm_state_size,
            ]),
        }));

        source.push(t_conv.clone());
        source.push(t_ssm.clone());
        source.push(max_positions);
        target.push(t_conv);
        target.push(t_ssm);
        (source, target)
    }
}
