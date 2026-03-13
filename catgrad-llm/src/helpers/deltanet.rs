use crate::helpers::tensors::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;

pub const GATED_DELTA_CHUNK_SIZE: usize = 64;

pub fn softplus(builder: &Builder, x: Var) -> Var {
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

fn pad_sequence<const N: usize>(builder: &Builder, x: Var, padded_seq_len: usize) -> Var {
    let mut sh = unpack::<N>(builder, shape(builder, x.clone()));
    sh[2] = padded_seq_len.to_nat(builder);
    let sh = pack(builder, sh);
    let zeros = zeros(builder, &sh);
    let x = concat(builder, 2, x, zeros);
    slice(builder, 2, 0, padded_seq_len, x)
}

// Delta rule for prefill stage
#[allow(clippy::too_many_arguments)]
pub fn chunk_gated_delta_rule(
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
    let v_head_dim = {
        let [_b, _h, _s, v_dim] = unpack::<4>(builder, shape(builder, value.clone()));
        v_dim
    };

    let sh = shape!(builder, batch_size, num_heads, 1, chunk_size, k_head_dim);
    let query = reshape(builder, sh.clone(), query);
    let key = reshape(builder, sh.clone(), key);
    let k_beta = reshape(builder, sh, k_beta);
    let value = reshape(
        builder,
        shape!(builder, batch_size, num_heads, 1, chunk_size, v_head_dim),
        value,
    );
    let v_beta = reshape(
        builder,
        shape!(builder, batch_size, num_heads, 1, chunk_size, v_head_dim),
        v_beta,
    );
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
#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta_rule(
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
