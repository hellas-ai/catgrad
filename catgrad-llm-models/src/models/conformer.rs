// Conformer model used by Gemma4 Audio
#![allow(clippy::too_many_arguments)]
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_act: String,
    pub subsampling_conv_channels: Vec<usize>,
    pub conv_kernel_size: usize,
    pub residual_weight: f32,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    pub attention_logit_cap: f32,
    pub attention_invalid_logits_value: f32,
    pub use_clipped_linears: bool,
    pub rms_norm_eps: f32,
    pub gradient_clipping: f32,
    pub output_proj_dims: usize,
}

impl Gemma4AudioConfig {
    pub fn num_soft_tokens_for_frames(&self, frames: usize) -> usize {
        let mut frames = frames;
        for _ in 0..2 {
            frames = frames.div_ceil(2);
        }
        frames.min(750)
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4AudioTower {
    pub config: Gemma4AudioConfig,
    pub input_time_steps: usize,
    pub input_feature_bins: usize,
}

impl Gemma4AudioTower {
    pub fn audio_model(&self, builder: &Builder, features: Var, mask: Var) -> Var {
        let mut x = reshape(
            builder,
            shape!(
                builder,
                1,
                1,
                self.input_time_steps,
                self.input_feature_bins
            ),
            features,
        );
        let mut mask = reshape(builder, shape!(builder, 1, self.input_time_steps), mask);
        (x, mask) = self.sscp_block(
            builder,
            path(vec![
                "model",
                "audio_tower",
                "subsample_conv_projection",
                "layer0",
            ])
            .unwrap(),
            1,
            self.config.subsampling_conv_channels[0],
            self.input_time_steps,
            self.input_feature_bins,
            x,
            mask,
        );
        let time_after_layer0 = self.input_time_steps.div_ceil(2);
        let freq_after_layer0 = self.input_feature_bins.div_ceil(2);
        (x, mask) = self.sscp_block(
            builder,
            path(vec![
                "model",
                "audio_tower",
                "subsample_conv_projection",
                "layer1",
            ])
            .unwrap(),
            self.config.subsampling_conv_channels[0],
            self.config.subsampling_conv_channels[1],
            time_after_layer0,
            freq_after_layer0,
            x,
            mask,
        );

        let time_after_layer1 = time_after_layer0.div_ceil(2);
        let freq_after_layer1 = freq_after_layer0.div_ceil(2);
        let x = transpose(builder, 1, 2, x);
        let x = transpose(builder, 2, 3, x);
        let x = reshape(
            builder,
            shape!(
                builder,
                1,
                time_after_layer1,
                freq_after_layer1 * self.config.subsampling_conv_channels[1]
            ),
            x,
        );
        let mut x = linear_no_bias(
            builder,
            freq_after_layer1 * self.config.subsampling_conv_channels[1],
            self.config.hidden_size,
            path(vec![
                "model",
                "audio_tower",
                "subsample_conv_projection",
                "input_proj_linear",
            ])
            .unwrap(),
            x,
        );
        let position_embeddings = relative_position_embeddings(
            builder,
            self.config.hidden_size,
            self.config.attention_context_left.saturating_sub(1)
                + self.config.attention_context_right
                + 1,
        );
        let attention_mask = blocked_bidirectional_attention_mask(
            builder,
            time_after_layer1,
            self.config.attention_chunk_size,
            self.config.attention_context_left.saturating_sub(1),
            self.config.attention_context_right,
            mask,
        );

        for layer_id in 0..self.config.num_hidden_layers {
            let layer_path = path(vec!["model", "audio_tower", "layers"])
                .unwrap()
                .extend([&layer_id.to_string()])
                .unwrap();
            x = self.conformer_block(
                builder,
                layer_path,
                time_after_layer1,
                attention_mask.clone(),
                position_embeddings.clone(),
                x,
            );
        }

        linear(
            builder,
            self.config.hidden_size,
            self.config.output_proj_dims,
            path(vec!["model", "audio_tower", "output_proj"]).unwrap(),
            x,
        )
    }

    fn conformer_block(
        &self,
        builder: &Builder,
        p: Path,
        seq_len: usize,
        attention_mask: Var,
        position_embeddings: Var,
        x: Var,
    ) -> Var {
        let x = self.feed_forward(builder, p.extend(["feed_forward1"]).unwrap(), x);
        let residual = x.clone();
        let x = clamp(
            builder,
            x,
            -self.config.gradient_clipping,
            self.config.gradient_clipping,
        );
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm_pre_attn"]).unwrap(),
            x,
        );
        let x = self.attention_block(
            builder,
            p.clone(),
            seq_len,
            attention_mask,
            position_embeddings,
            x,
        );
        let x = clamp(
            builder,
            x,
            -self.config.gradient_clipping,
            self.config.gradient_clipping,
        );
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm_post_attn"]).unwrap(),
            x,
        );
        let x = residual + x;
        let x = self.light_conv(builder, p.extend(["lconv1d"]).unwrap(), x);
        let x = self.feed_forward(builder, p.extend(["feed_forward2"]).unwrap(), x);
        let x = clamp(
            builder,
            x,
            -self.config.gradient_clipping,
            self.config.gradient_clipping,
        );
        rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm_out"]).unwrap(),
            x,
        )
    }

    fn feed_forward(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let residual = x.clone();
        let x = clamp(
            builder,
            x,
            -self.config.gradient_clipping,
            self.config.gradient_clipping,
        );
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_layer_norm"]).unwrap(),
            x,
        );
        let x = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size * 4,
            p.extend(["ffw_layer_1"]).unwrap(),
            x,
        );
        let x = activation(builder, &self.config.hidden_act, x);
        let x = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size * 4,
            self.config.hidden_size,
            p.extend(["ffw_layer_2"]).unwrap(),
            x,
        );
        let x = clamp(
            builder,
            x,
            -self.config.gradient_clipping,
            self.config.gradient_clipping,
        );
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_layer_norm"]).unwrap(),
            x,
        );
        let scale = constant(
            builder,
            self.config.residual_weight,
            &shape(builder, x.clone()),
        );
        let scale = cast(builder, scale, dtype(builder, x.clone()));
        residual + x * scale
    }

    fn attention_block(
        &self,
        builder: &Builder,
        p: Path,
        seq_len: usize,
        attention_mask: Var,
        position_embeddings: Var,
        x: Var,
    ) -> Var {
        let x = self.local_attention(
            builder,
            p.extend(["self_attn"]).unwrap(),
            seq_len,
            attention_mask,
            position_embeddings,
            x,
        );
        let x = reshape(
            builder,
            shape!(builder, 1, seq_len, self.config.hidden_size),
            x,
        );
        clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["self_attn", "post"]).unwrap(),
            x,
        )
    }

    fn local_attention(
        &self,
        builder: &Builder,
        p: Path,
        seq_len: usize,
        attention_mask: Var,
        position_embeddings: Var,
        x: Var,
    ) -> Var {
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.hidden_size / num_heads;
        let chunk_size = self.config.attention_chunk_size;
        let max_past = self.config.attention_context_left.saturating_sub(1);
        let max_future = self.config.attention_context_right;
        let context_size = chunk_size + max_past + max_future;
        let num_blocks = seq_len.div_ceil(chunk_size);
        let q_scale = head_dim as f32;
        let q_scale = q_scale.powf(-0.5) / std::f32::consts::LN_2;
        let k_scale = (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;

        let q = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );
        let k = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let q = reshape(builder, shape!(builder, 1, seq_len, num_heads, head_dim), q);
        let k = reshape(builder, shape!(builder, 1, seq_len, num_heads, head_dim), k);
        let v = reshape(builder, shape!(builder, 1, seq_len, num_heads, head_dim), v);

        let q_scale = constant(builder, q_scale, &shape(builder, q.clone()));
        let q_scale = cast(builder, q_scale, dtype(builder, q.clone()));
        let q = q * q_scale;
        let per_dim_scale = param(builder, &p.extend(["per_dim_scale"]).unwrap());
        let per_dim_scale = softplus(builder, per_dim_scale);
        let per_dim_scale = reshape(builder, shape!(builder, 1, 1, 1, head_dim), per_dim_scale);
        let q = q.clone() * broadcast(builder, shape(builder, q), per_dim_scale);
        let k_scale = constant(builder, k_scale, &shape(builder, k.clone()));
        let k_scale = cast(builder, k_scale, dtype(builder, k.clone()));
        let k = k * k_scale;

        let q_blocks = blockify_4d(builder, seq_len, chunk_size, num_heads, head_dim, q);
        let k_blocks = extract_block_context_4d(
            builder,
            seq_len,
            max_past,
            max_future,
            chunk_size,
            context_size,
            num_heads,
            head_dim,
            k,
        );
        let v_blocks = extract_block_context_4d(
            builder,
            seq_len,
            max_past,
            max_future,
            chunk_size,
            context_size,
            num_heads,
            head_dim,
            v,
        );

        let mut logits = relative_attention_logits(
            builder,
            p,
            position_embeddings,
            num_blocks,
            chunk_size,
            context_size,
            num_heads,
            head_dim,
            q_blocks,
            k_blocks,
        );
        let softcap = constant(
            builder,
            self.config.attention_logit_cap,
            &shape(builder, logits.clone()),
        );
        let softcap = cast(builder, softcap, dtype(builder, logits.clone()));
        logits = tanh(builder, logits / softcap.clone()) * softcap;

        let logits = masked_fill(
            builder,
            broadcast(builder, shape(builder, logits.clone()), attention_mask),
            self.config.attention_invalid_logits_value,
            logits,
        );
        let attn = softmax(builder, logits);
        let mut head_outputs = Vec::with_capacity(num_heads);
        for head in 0..num_heads {
            let attn_head = slice(builder, 1, head, 1, attn.clone());
            let attn_head = squeeze::<5, 4>(builder, 1, attn_head);
            let value_head = slice(builder, 3, head, 1, v_blocks.clone());
            let value_head = squeeze::<5, 4>(builder, 3, value_head);

            let mut block_outputs = Vec::with_capacity(num_blocks);
            for block in 0..num_blocks {
                let attn_block = slice(builder, 1, block, 1, attn_head.clone());
                let attn_block = squeeze::<4, 3>(builder, 1, attn_block);
                let value_block = slice(builder, 1, block, 1, value_head.clone());
                let value_block = squeeze::<4, 3>(builder, 1, value_block);
                let out = matmul(builder, attn_block, value_block);
                block_outputs.push(unsqueeze::<3, 4>(builder, 1, out));
            }

            let head_output = block_outputs
                .into_iter()
                .reduce(|acc, item| concat(builder, 1, acc, item))
                .unwrap();
            head_outputs.push(unsqueeze::<4, 5>(builder, 3, head_output));
        }

        let attn = head_outputs
            .into_iter()
            .reduce(|acc, item| concat(builder, 3, acc, item))
            .unwrap();
        let attn = reshape(
            builder,
            shape!(builder, 1, num_blocks * chunk_size, num_heads * head_dim),
            attn,
        );
        slice(builder, 1, 0, seq_len, attn)
    }

    fn light_conv(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let residual = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_layer_norm"]).unwrap(),
            x,
        );
        let x = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size * 2,
            p.extend(["linear_start"]).unwrap(),
            x,
        );
        let [value, gate] = chunk(builder, 2, 2, self.config.hidden_size, x)
            .try_into()
            .unwrap();
        let x = value * sigmoid(builder, gate);
        let x = transpose(builder, 1, 2, x);
        let x = depthwise_conv1d_no_bias(
            builder,
            p.extend(["depthwise_conv1d"]).unwrap(),
            self.config.conv_kernel_size,
            x,
            self.config.conv_kernel_size - 1,
        );
        let x = transpose(builder, 1, 2, x);
        let x = clamp(
            builder,
            x,
            -self.config.gradient_clipping,
            self.config.gradient_clipping,
        );
        let x = rmsnorm::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["conv_norm"]).unwrap(),
            x,
        );
        let x = activation(builder, &self.config.hidden_act, x);
        let x = clippable_linear_no_bias(
            builder,
            self.config.use_clipped_linears,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["linear_end"]).unwrap(),
            x,
        );
        residual + x
    }

    fn sscp_block(
        &self,
        builder: &Builder,
        p: Path,
        in_channels: usize,
        out_channels: usize,
        time_steps: usize,
        freq_bins: usize,
        x: Var,
        mask: Var,
    ) -> (Var, Var) {
        let out_time = time_steps.div_ceil(2);
        let out_freq = freq_bins.div_ceil(2);
        let x_dtype = dtype(builder, x.clone());
        let zero = constant(builder, 0.0, &shape(builder, mask.clone()));
        let valid = eq(builder, mask.clone(), zero);
        let valid = cast(builder, valid, x_dtype);
        let valid = reshape(builder, shape!(builder, 1, 1, time_steps, 1), valid);
        let valid = broadcast(
            builder,
            shape!(builder, 1, in_channels, time_steps, freq_bins),
            valid,
        );
        let x = x * valid;
        let x = conv2d_stride2_square_no_bias(
            builder,
            p.extend(["conv"]).unwrap(),
            in_channels,
            out_channels,
            time_steps,
            freq_bins,
            x,
        );
        let mask = subsample_mask(builder, time_steps, out_time, mask);
        let x = transpose(builder, 1, 2, x);
        let x = transpose(builder, 2, 3, x);
        let x = reshape(
            builder,
            shape!(builder, 1, out_time * out_freq, out_channels),
            x,
        );
        let x = layernorm_weight_only(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm"]).unwrap(),
            x,
        );
        let x = reshape(
            builder,
            shape!(builder, 1, out_time, out_freq, out_channels),
            x,
        );
        let x = relu(builder, x);
        let x = transpose(builder, 2, 3, x);
        (transpose(builder, 1, 2, x), mask)
    }
}

fn activation(builder: &Builder, hidden_act: &str, x: Var) -> Var {
    match hidden_act {
        "silu" => silu(builder, x),
        other => panic!("unsupported Gemma4 audio activation `{other}`"),
    }
}

fn relu(builder: &Builder, x: Var) -> Var {
    clamp(builder, x, 0.0, f32::MAX)
}

fn layernorm_weight_only(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let x = layernorm_raw(builder, eps, x);
    let weight = param(builder, &p.extend(["weight"]).unwrap());
    x.clone() * broadcast(builder, shape(builder, x), weight)
}

fn clippable_linear_no_bias(
    builder: &Builder,
    use_clipped_linears: bool,
    in_features: usize,
    out_features: usize,
    p: Path,
    x: Var,
) -> Var {
    let x = if use_clipped_linears {
        let sh = shape(builder, x.clone());
        let input_min = broadcast(
            builder,
            sh.clone(),
            param(builder, &p.extend(["input_min"]).unwrap()),
        );
        let input_max = broadcast(
            builder,
            sh,
            param(builder, &p.extend(["input_max"]).unwrap()),
        );
        clamp_with_tensors(builder, x, input_min, input_max)
    } else {
        x
    };
    let x = linear_no_bias(
        builder,
        in_features,
        out_features,
        p.extend(["linear"]).unwrap(),
        x,
    );
    if use_clipped_linears {
        let sh = shape(builder, x.clone());
        let output_min = broadcast(
            builder,
            sh.clone(),
            param(builder, &p.extend(["output_min"]).unwrap()),
        );
        let output_max = broadcast(
            builder,
            sh,
            param(builder, &p.extend(["output_max"]).unwrap()),
        );
        clamp_with_tensors(builder, x, output_min, output_max)
    } else {
        x
    }
}

fn relative_attention_logits(
    builder: &Builder,
    p: Path,
    position_embeddings: Var,
    num_blocks: usize,
    chunk_size: usize,
    context_size: usize,
    num_heads: usize,
    head_dim: usize,
    q_blocks: Var,
    k_blocks: Var,
) -> Var {
    let rel_k = linear_no_bias(
        builder,
        num_heads * head_dim,
        num_heads * head_dim,
        p.extend(["relative_k_proj"]).unwrap(),
        position_embeddings,
    );
    let rel_count = context_size - chunk_size + 1;
    let rel_k = reshape(
        builder,
        shape!(builder, 1, rel_count, num_heads, head_dim),
        rel_k,
    );
    let rel_k = transpose(builder, 1, 2, rel_k);
    let rel_k = transpose(builder, 2, 3, rel_k);

    let q_ac = transpose(builder, 1, 3, q_blocks);
    let q_ac = transpose(builder, 2, 3, q_ac);
    let k_ac = transpose(builder, 1, 3, k_blocks);
    let k_ac = transpose(builder, 2, 3, k_ac);
    let k_ac = transpose(builder, 3, 4, k_ac);

    let mut head_terms = Vec::with_capacity(num_heads);
    for head in 0..num_heads {
        let q_head = slice(builder, 1, head, 1, q_ac.clone());
        let q_head = squeeze::<5, 4>(builder, 1, q_head);
        let k_head = slice(builder, 1, head, 1, k_ac.clone());
        let k_head = squeeze::<5, 4>(builder, 1, k_head);
        let rel_head = slice(builder, 1, head, 1, rel_k.clone());
        let rel_head = squeeze::<4, 3>(builder, 1, rel_head);

        let mut block_terms = Vec::with_capacity(num_blocks);
        for block in 0..num_blocks {
            let q_block = slice(builder, 1, block, 1, q_head.clone());
            let q_block = squeeze::<4, 3>(builder, 1, q_block);
            let k_block = slice(builder, 1, block, 1, k_head.clone());
            let k_block = squeeze::<4, 3>(builder, 1, k_block);
            let term_ac = matmul(builder, q_block.clone(), k_block);

            let q_block_flat = reshape(builder, shape!(builder, 1, chunk_size, head_dim), q_block);
            let term_bd = matmul(builder, q_block_flat, rel_head.clone());
            let term_bd = reshape(builder, shape!(builder, 1, chunk_size, rel_count), term_bd);
            let term_bd = relative_shift_single_head(builder, chunk_size, context_size, term_bd);

            let term = term_ac + term_bd;
            block_terms.push(unsqueeze::<3, 4>(builder, 1, term));
        }

        let head_term = block_terms
            .into_iter()
            .reduce(|acc, item| concat(builder, 1, acc, item))
            .unwrap();
        head_terms.push(unsqueeze::<4, 5>(builder, 1, head_term));
    }

    head_terms
        .into_iter()
        .reduce(|acc, item| concat(builder, 1, acc, item))
        .unwrap()
}

fn relative_shift_single_head(
    builder: &Builder,
    chunk_size: usize,
    context_size: usize,
    term_bd: Var,
) -> Var {
    let rel_count = context_size - chunk_size + 1;
    let pad = context_size + 1 - rel_count;
    let term_bd = if pad > 0 {
        let zeros = zeros(
            builder,
            &shape!(builder, 1, chunk_size, pad),
            dtype(builder, term_bd.clone()),
        );
        concat(builder, 2, term_bd, zeros)
    } else {
        term_bd
    };
    let term_bd = reshape(
        builder,
        shape!(builder, 1, chunk_size * (context_size + 1)),
        term_bd,
    );
    let term_bd = slice(builder, 1, 0, chunk_size * context_size, term_bd);
    reshape(
        builder,
        shape!(builder, 1, chunk_size, context_size),
        term_bd,
    )
}

fn relative_position_embeddings(builder: &Builder, hidden_size: usize, count: usize) -> Var {
    let positions = cast(builder, arange(builder, count), Dtype::F32);
    let positions = constant(
        builder,
        (count.saturating_sub(1)) as f32,
        &shape(builder, positions.clone()),
    ) - positions;
    let positions = reshape(builder, shape!(builder, 1, count, 1), positions);

    let num_timescales = hidden_size / 2;
    let min_timescale = 1.0f32;
    let max_timescale = 10_000.0f32;
    let log_increment =
        (max_timescale / min_timescale).ln() / num_timescales.saturating_sub(1).max(1) as f32;
    let inv_idx = cast(builder, arange(builder, num_timescales), Dtype::F32);
    let inv_idx = constant(builder, -log_increment, &shape(builder, inv_idx.clone())) * inv_idx;
    let inv_timescales = exp(builder, inv_idx);
    let inv_timescales =
        inv_timescales.clone() * constant(builder, min_timescale, &shape(builder, inv_timescales));
    let inv_timescales = reshape(
        builder,
        shape!(builder, 1, 1, num_timescales),
        inv_timescales,
    );
    let scaled_time_shape = shape!(builder, 1, count, num_timescales);
    let positions = broadcast(builder, scaled_time_shape.clone(), positions);
    let inv_timescales = broadcast(builder, scaled_time_shape, inv_timescales);
    let scaled_time = positions * inv_timescales;
    concat(
        builder,
        2,
        sin(builder, scaled_time.clone()),
        cos(builder, scaled_time),
    )
}

fn sliding_window_valid_mask(
    builder: &Builder,
    seq_len: usize,
    max_past: usize,
    max_future: usize,
) -> Var {
    let idx = cast(builder, arange(builder, seq_len), Dtype::F32);
    let row = reshape(builder, shape!(builder, seq_len, 1), idx.clone());
    let row = broadcast(builder, shape!(builder, seq_len, seq_len), row);
    let col = reshape(builder, shape!(builder, 1, seq_len), idx);
    let col = broadcast(builder, shape!(builder, seq_len, seq_len), col);
    let past_ok = lte(
        builder,
        row.clone(),
        col.clone() + constant(builder, max_past as f32, &shape!(builder, seq_len, seq_len)),
    );
    let future_ok = lte(
        builder,
        col,
        row + constant(
            builder,
            max_future as f32,
            &shape!(builder, seq_len, seq_len),
        ),
    );
    cast(builder, past_ok, Dtype::F32) * cast(builder, future_ok, Dtype::F32)
}

fn subsample_mask(builder: &Builder, in_len: usize, out_len: usize, mask: Var) -> Var {
    let items = (0..out_len)
        .map(|idx| {
            let start = (idx * 2).min(in_len.saturating_sub(1));
            slice(builder, 1, start, 1, mask.clone())
        })
        .collect::<Vec<_>>();
    items
        .into_iter()
        .reduce(|acc, item| concat(builder, 1, acc, item))
        .unwrap()
}

fn pad_time_4d(
    builder: &Builder,
    num_heads: usize,
    head_dim: usize,
    left: usize,
    right: usize,
    x: Var,
) -> Var {
    let x_dtype = dtype(builder, x.clone());
    let left_pad = zeros(
        builder,
        &shape!(builder, 1, left, num_heads, head_dim),
        x_dtype.clone(),
    );
    let right_pad = zeros(
        builder,
        &shape!(builder, 1, right, num_heads, head_dim),
        x_dtype,
    );
    let x = concat(builder, 1, left_pad, x);
    concat(builder, 1, x, right_pad)
}

fn pad_time_2d(builder: &Builder, left: usize, right: usize, x: Var) -> Var {
    let x_dtype = dtype(builder, x.clone());
    let left_pad = zeros(builder, &shape!(builder, 1, left), x_dtype.clone());
    let right_pad = zeros(builder, &shape!(builder, 1, right), x_dtype);
    let x = concat(builder, 1, left_pad, x);
    concat(builder, 1, x, right_pad)
}

fn blockify_4d(
    builder: &Builder,
    seq_len: usize,
    chunk_size: usize,
    num_heads: usize,
    head_dim: usize,
    x: Var,
) -> Var {
    let num_blocks = seq_len.div_ceil(chunk_size);
    let pad = num_blocks * chunk_size - seq_len;
    let x = if pad > 0 {
        pad_time_4d(builder, num_heads, head_dim, 0, pad, x)
    } else {
        x
    };
    reshape(
        builder,
        shape!(builder, 1, num_blocks, chunk_size, num_heads, head_dim),
        x,
    )
}

fn extract_block_context_4d(
    builder: &Builder,
    seq_len: usize,
    max_past: usize,
    max_future: usize,
    chunk_size: usize,
    context_size: usize,
    num_heads: usize,
    head_dim: usize,
    x: Var,
) -> Var {
    let num_blocks = seq_len.div_ceil(chunk_size);
    let x = pad_time_4d(
        builder,
        num_heads,
        head_dim,
        max_past,
        max_future + chunk_size - 1,
        x,
    );
    let items = (0..num_blocks)
        .map(|idx| {
            let start = idx * chunk_size;
            let window = slice(builder, 1, start, context_size, x.clone());
            unsqueeze::<4, 5>(builder, 1, window)
        })
        .collect::<Vec<_>>();
    items
        .into_iter()
        .reduce(|acc, item| concat(builder, 1, acc, item))
        .unwrap()
}

fn blocked_bidirectional_attention_mask(
    builder: &Builder,
    seq_len: usize,
    chunk_size: usize,
    max_past: usize,
    max_future: usize,
    mask: Var,
) -> Var {
    let num_blocks = seq_len.div_ceil(chunk_size);
    let padded_seq_len = num_blocks * chunk_size;
    let pad_amount = padded_seq_len - seq_len;
    let valid_1d = cast(
        builder,
        eq(
            builder,
            mask,
            constant(builder, 0.0, &shape!(builder, 1, seq_len)),
        ),
        Dtype::F32,
    );
    let valid_1d = if pad_amount > 0 {
        pad_time_2d(builder, 0, pad_amount, valid_1d)
    } else {
        valid_1d
    };
    let query_valid = reshape(
        builder,
        shape!(builder, 1, 1, padded_seq_len, 1),
        valid_1d.clone(),
    );
    let query_valid = broadcast(
        builder,
        shape!(builder, 1, 1, padded_seq_len, padded_seq_len),
        query_valid,
    );
    let key_valid = reshape(builder, shape!(builder, 1, 1, 1, padded_seq_len), valid_1d);
    let key_valid = broadcast(
        builder,
        shape!(builder, 1, 1, padded_seq_len, padded_seq_len),
        key_valid,
    );
    let sliding = sliding_window_valid_mask(builder, padded_seq_len, max_past, max_future);
    let sliding = unsqueeze::<2, 3>(builder, 0, sliding);
    let sliding = unsqueeze::<3, 4>(builder, 0, sliding);
    let sliding = broadcast(
        builder,
        shape!(builder, 1, 1, padded_seq_len, padded_seq_len),
        sliding,
    );
    let valid_4d = query_valid * key_valid * sliding;
    let valid_5d = convert_4d_mask_to_blocked_5d(
        builder,
        padded_seq_len,
        chunk_size,
        max_past,
        max_future,
        valid_4d,
    );
    let one = constant(builder, 1.0, &shape(builder, valid_5d.clone()));
    one - valid_5d
}

fn convert_4d_mask_to_blocked_5d(
    builder: &Builder,
    padded_seq_len: usize,
    chunk_size: usize,
    max_past: usize,
    max_future: usize,
    mask_4d: Var,
) -> Var {
    let num_blocks = padded_seq_len.div_ceil(chunk_size);
    let context_size = chunk_size + max_past + max_future;
    let mask_5d = reshape(
        builder,
        shape!(builder, 1, 1, num_blocks, chunk_size, padded_seq_len),
        mask_4d,
    );
    let mask_5d = pad_kv_5d(builder, max_past, max_future, mask_5d);
    let items = (0..num_blocks)
        .map(|block_idx| {
            let start = block_idx * chunk_size;
            let block = slice(builder, 2, block_idx, 1, mask_5d.clone());
            let block = squeeze::<5, 4>(builder, 2, block);
            let block = slice(builder, 3, start, context_size, block);
            unsqueeze::<4, 5>(builder, 2, block)
        })
        .collect::<Vec<_>>();
    items
        .into_iter()
        .reduce(|acc, item| concat(builder, 2, acc, item))
        .unwrap()
}

fn pad_kv_5d(builder: &Builder, left: usize, right: usize, x: Var) -> Var {
    let x_dtype = dtype(builder, x.clone());
    let [b, one, num_blocks, chunk_size, _seq_len] =
        unpack::<5>(builder, shape(builder, x.clone()));
    let left_pad = zeros(
        builder,
        &shape!(builder, b, one, num_blocks, chunk_size, left),
        x_dtype.clone(),
    );
    let right_pad = zeros(
        builder,
        &shape!(builder, b, one, num_blocks, chunk_size, right),
        x_dtype,
    );
    let x = concat(builder, 4, left_pad, x);
    concat(builder, 4, x, right_pad)
}

fn conv2d_stride2_square_no_bias(
    builder: &Builder,
    p: Path,
    in_channels: usize,
    out_channels: usize,
    time_steps: usize,
    freq_bins: usize,
    x: Var,
) -> Var {
    let out_time = time_steps.div_ceil(2);
    let out_freq = freq_bins.div_ceil(2);
    let x = pad_4d(builder, x, in_channels, time_steps, freq_bins, 1, 2, 1, 2);

    let mut windows = Vec::with_capacity(9);
    for kernel_t in 0..3 {
        let xt = slice(builder, 2, kernel_t, 2 * out_time, x.clone());
        let xt = reshape(
            builder,
            shape!(builder, 1, in_channels, out_time, 2, freq_bins + 3),
            xt,
        );
        let xt = squeeze::<5, 4>(builder, 3, slice(builder, 3, 0, 1, xt));
        for kernel_f in 0..3 {
            let xf = slice(builder, 3, kernel_f, 2 * out_freq, xt.clone());
            let xf = reshape(
                builder,
                shape!(builder, 1, in_channels, out_time, out_freq, 2),
                xf,
            );
            let xf = squeeze::<5, 4>(builder, 4, slice(builder, 4, 0, 1, xf));
            let xf = transpose(builder, 1, 2, xf);
            let xf = transpose(builder, 2, 3, xf);
            windows.push(xf);
        }
    }

    let x = windows
        .into_iter()
        .map(|item| unsqueeze::<4, 5>(builder, 4, item))
        .reduce(|acc, item| concat(builder, 4, acc, item))
        .unwrap();
    let x = reshape(
        builder,
        shape!(builder, 1, out_time * out_freq, in_channels * 9),
        x,
    );
    let weight = param(builder, &p.extend(["weight"]).unwrap());
    let weight = reshape(
        builder,
        shape!(builder, out_channels, in_channels * 9),
        weight,
    );
    let weight = transpose(builder, 0, 1, weight);
    let weight = broadcast(
        builder,
        shape!(builder, 1, in_channels * 9, out_channels),
        weight,
    );
    let x = matmul(builder, x, weight);
    let x = reshape(
        builder,
        shape!(builder, 1, out_time, out_freq, out_channels),
        x,
    );
    let x = transpose(builder, 2, 3, x);
    transpose(builder, 1, 2, x)
}

fn pad_4d(
    builder: &Builder,
    x: Var,
    channels: usize,
    time_steps: usize,
    freq_bins: usize,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) -> Var {
    let time_pad = zeros(
        builder,
        &shape!(builder, 1, channels, top + time_steps + bottom, left),
        dtype(builder, x.clone()),
    );
    let time_pad_right = zeros(
        builder,
        &shape!(builder, 1, channels, top + time_steps + bottom, right),
        dtype(builder, x.clone()),
    );
    let top_pad = zeros(
        builder,
        &shape!(builder, 1, channels, top, freq_bins),
        dtype(builder, x.clone()),
    );
    let bottom_pad = zeros(
        builder,
        &shape!(builder, 1, channels, bottom, freq_bins),
        dtype(builder, x.clone()),
    );
    let x = concat(builder, 2, top_pad, x);
    let x = concat(builder, 2, x, bottom_pad);
    let x = concat(builder, 3, time_pad, x);
    concat(builder, 3, x, time_pad_right)
}
