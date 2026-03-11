use crate::config::{Llama3RopeScaling, YarnRopeScaling};
use crate::helpers::tensors::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

// Generate rope tables. This part is usually precomputed
pub fn rope_tables(
    builder: &Builder,
    theta: f32,
    seq_len: impl IntoNatVar,
    head_dim: usize,
    factor: f32,
) -> (Var, Var) {
    let half_dim = head_dim / 2;

    let f = arange(builder, half_dim);
    let f = cast(builder, f, Dtype::F32);
    let sh = shape(builder, f.clone());
    let two = constant(builder, 2.0 / (head_dim as f32), &sh);
    let f = f * two;
    let theta = constant(builder, theta, &sh);
    let freq = pow(builder, theta, f);
    let inv_freq = inverse(builder, freq);

    let seq_len = seq_len.to_nat(builder);
    let sh = shape!(builder, seq_len, half_dim);
    let inv_freq = broadcast(builder, sh.clone(), inv_freq);

    let factor = constant(builder, factor, &sh);
    let inv_freq = inv_freq / factor;

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, sh, pos);
    let pos = pos * inv_freq;
    let cos = cos(builder, pos.clone());
    let sin = sin(builder, pos);

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

pub fn rope_tables_llama3(
    builder: &Builder,
    theta: f32,
    rope_scaling: &Llama3RopeScaling,
    seq_len: impl IntoNatVar,
    head_dim: usize,
) -> (Var, Var) {
    let half_dim = head_dim / 2;

    let f = arange(builder, half_dim);
    let f = cast(builder, f, Dtype::F32);
    let sh = shape(builder, f.clone());
    let two = constant(builder, 2.0 / (head_dim as f32), &sh);
    let f = f * two;
    let theta_val = constant(builder, theta, &sh);
    let freq = pow(builder, theta_val, f);
    let inv_freq = inverse(builder, freq);

    let low_freq_wavelength =
        rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
    let high_freq_wavelength =
        rope_scaling.original_max_position_embeddings as f32 / rope_scaling.high_freq_factor;

    let sh = shape(builder, inv_freq.clone());
    let low_freq_wavelength = constant(builder, low_freq_wavelength, &sh);
    let high_freq_wavelength = constant(builder, high_freq_wavelength, &sh);
    let factor = constant(builder, rope_scaling.factor, &sh);
    let low_freq_factor = constant(builder, rope_scaling.low_freq_factor, &sh);
    let high_freq_factor = constant(builder, rope_scaling.high_freq_factor, &sh);
    let old_context_len = constant(
        builder,
        rope_scaling.original_max_position_embeddings as f32,
        &sh,
    );
    let pi_2 = constant(builder, 2. * std::f32::consts::PI, &sh);

    let wavelen = pi_2 / inv_freq.clone();

    let low_freqs_mask = lt(builder, low_freq_wavelength, wavelen.clone());
    let low_freqs_mask = cast(builder, low_freqs_mask, Dtype::F32);
    let one = constant(builder, 1.0, &sh);

    let inv_freq_scaled = inv_freq.clone() / factor.clone();
    let inv_freq = low_freqs_mask.clone() * inv_freq_scaled
        + (one.clone() - low_freqs_mask.clone()) * inv_freq;

    let high_freqs_mask = lt(builder, wavelen.clone(), high_freq_wavelength);
    let high_freqs_mask = cast(builder, high_freqs_mask, Dtype::F32);

    let not_high = one.clone() - high_freqs_mask;
    let not_low = one.clone() - low_freqs_mask;
    let mid_freqs_mask = not_high * not_low;

    let smooth_factor = (old_context_len / wavelen - low_freq_factor.clone())
        / (high_freq_factor - low_freq_factor);

    let smoothed_inv_freq = smooth_factor.clone() * inv_freq.clone()
        + (one.clone() - smooth_factor) * inv_freq.clone() / factor;

    let inv_freq = mid_freqs_mask.clone() * smoothed_inv_freq + (one - mid_freqs_mask) * inv_freq;

    let seq_len = seq_len.to_nat(builder);
    let sh = shape!(builder, seq_len, half_dim);
    let inv_freq = broadcast(builder, sh, inv_freq);

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, sh, pos);
    let pos = pos * inv_freq;
    let cos = cos(builder, pos.clone());
    let sin = sin(builder, pos);

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

fn rope_yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        return 1.0;
    }
    0.1 * mscale * scale.ln() + 1.0
}

fn find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32
        * (max_position_embeddings as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
        / (2. * base.ln())
}

fn find_correction_range(
    low: f32,
    high: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> (f32, f32) {
    let low = find_correction_dim(low, dim, base, max_position_embeddings);
    let high = find_correction_dim(high, dim, base, max_position_embeddings);
    (low, high)
}

fn linear_ramp_factor(builder: &Builder, min: f32, max: f32, dim: usize) -> Var {
    let r = arange(builder, dim);
    let r = cast(builder, r, Dtype::F32);
    let sh = shape(builder, r.clone());
    let d = constant(builder, max - min, &sh);
    let min_val = constant(builder, min, &sh);
    let r = r - min_val;
    let r = r / d;
    clamp(builder, r, 0.0, 1.0)
}

pub fn rope_tables_yarn(
    builder: &Builder,
    theta: f32,
    rope_scaling: &YarnRopeScaling,
    seq_len: impl IntoNatVar,
    head_dim: usize,
) -> (Var, Var) {
    let half_dim = head_dim / 2;

    let (low, high) = find_correction_range(
        rope_scaling.beta_fast,
        rope_scaling.beta_slow,
        head_dim,
        theta,
        rope_scaling.original_max_position_embeddings,
    );

    let f = arange(builder, half_dim);
    let f = cast(builder, f, Dtype::F32);
    let sh = shape(builder, f.clone());
    let two = constant(builder, 2.0 / (head_dim as f32), &sh);
    let f = f * two;
    let theta_val = constant(builder, theta, &sh);
    let freq = pow(builder, theta_val, f);

    let inv_freq_extrapolation = inverse(builder, freq);
    let inv_freq_extrapolation_factor = linear_ramp_factor(builder, low, high, half_dim);
    let sh = shape(builder, inv_freq_extrapolation_factor.clone());
    let one = constant(builder, 1.0, &sh);
    let inv_freq_extrapolation_factor = one.clone() - inv_freq_extrapolation_factor;

    let factor = constant(builder, rope_scaling.factor, &sh);
    let inv_freq_interpolation = inv_freq_extrapolation.clone() / factor;

    let inv_freq = inv_freq_interpolation * (one - inv_freq_extrapolation_factor.clone())
        + inv_freq_extrapolation * inv_freq_extrapolation_factor;

    let seq_len = seq_len.to_nat(builder);
    let sh = shape!(builder, seq_len, half_dim);
    let inv_freq = broadcast(builder, sh, inv_freq);

    let scale = if rope_scaling.mscale != 0.0 && rope_scaling.mscale_all_dim != 0.0 {
        rope_yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale)
            / rope_yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale_all_dim)
    } else {
        rope_yarn_get_mscale(rope_scaling.factor, 1.0)
    };
    let sh = shape!(builder, 1);
    let scale = constant(builder, scale, &sh);

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, sh.clone(), pos);
    let pos = pos * inv_freq;
    let scale = broadcast(builder, sh, scale);
    let cos = cos(builder, pos.clone());
    let cos = cos * scale.clone();
    let sin = sin(builder, pos);
    let sin = sin * scale;

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

fn rotate_half(builder: &Builder, head_dim: usize, x: Var) -> Var {
    let v = chunk(builder, 3, 2, head_dim / 2, x);

    concat(builder, 3, -v[1].clone(), v[0].clone())
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by reusing calculated tables
pub fn apply_rope_embedding(
    builder: &Builder,
    pos: impl IntoNatVar,
    head_dim: usize,
    cos: Var,
    sin: Var,
    x: Var,
) -> Var {
    let sh = shape(builder, x.clone());
    let [_, _, seq_len, _] = unpack::<4>(builder, sh.clone());
    let pos = pos.to_nat(builder);
    let cos = slice(builder, 0, pos.clone(), seq_len.clone(), cos);
    let sin = slice(builder, 0, pos, seq_len, sin);
    let cos = broadcast(builder, sh.clone(), cos);
    let sin = broadcast(builder, sh, sin);

    let rotated_x = rotate_half(builder, head_dim, x.clone());

    cos * x + sin * rotated_x
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by calculating the tables
pub fn rope(
    builder: &Builder,
    theta: f32,
    pos: impl IntoNatVar,
    seq_len: &impl IntoNatVar,
    head_dim: usize,
    factor: f32,
    x: Var,
) -> Var {
    let pos = pos.to_nat(builder);
    let seq_len = seq_len.to_nat(builder);
    let (cos, sin) = rope_tables(builder, theta, pos.clone() + seq_len, head_dim, factor);

    apply_rope_embedding(builder, pos, head_dim, cos, sin, x)
}
