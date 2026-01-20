use super::layers::*;
use crate::legacy::models::utils::{Llama3RopeScaling, YarnRopeScaling};
use catgrad_legacy::backend::cpu::eval::Builder;
use catgrad_legacy::core::{Dtype, Shape, Var};
use std::f32::consts::PI;

// Generate rope tables. This part is usually precomputed
pub fn rope_tables(builder: &Builder, theta: f32, seq_len: usize, head_dim: usize) -> (Var, Var) {
    let half_dim = head_dim / 2;

    let f = arange(builder, half_dim, Dtype::F32);
    let two = constant(builder, f.label.clone(), 2.0 / (head_dim as f32));
    let f = f * two;
    let theta = constant(builder, f.label.clone(), theta);
    let freq = power(builder, theta, f);
    let inv_freq = inverse(builder, freq);
    let inv_freq = expand(builder, Shape(vec![seq_len, half_dim]), inv_freq);

    let pos = arange(builder, seq_len, Dtype::F32);
    let pos = reshape(builder, Shape(vec![seq_len, 1]), pos);
    let pos = expand(builder, inv_freq.label.shape.clone(), pos);
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
    seq_len: usize,
    head_dim: usize,
) -> (Var, Var) {
    let half_dim = head_dim / 2;

    let f = arange(builder, half_dim, Dtype::F32);
    let two = constant(builder, f.label.clone(), 2.0 / (head_dim as f32));
    let f = f * two;
    let theta = constant(builder, f.label.clone(), theta);
    let freq = power(builder, theta, f);
    let inv_freq = inverse(builder, freq);
    let inv_freq = expand(builder, Shape(vec![seq_len, half_dim]), inv_freq);

    let low_freq_wavelength =
        rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
    let high_freq_wavelength =
        rope_scaling.original_max_position_embeddings as f32 / rope_scaling.high_freq_factor;
    let low_freq_wavelength = constant(builder, inv_freq.label.clone(), low_freq_wavelength);
    let high_freq_wavelength = constant(builder, inv_freq.label.clone(), high_freq_wavelength);
    let factor = constant(builder, inv_freq.label.clone(), rope_scaling.factor);
    let low_freq_factor = constant(
        builder,
        inv_freq.label.clone(),
        rope_scaling.low_freq_factor,
    );
    let high_freq_factor = constant(
        builder,
        inv_freq.label.clone(),
        rope_scaling.high_freq_factor,
    );
    let old_context_len = constant(
        builder,
        inv_freq.label.clone(),
        rope_scaling.original_max_position_embeddings as f32,
    );

    let wavelen = constant(builder, inv_freq.label.clone(), 2. * PI) / inv_freq.clone();

    // if wavelen > low_freq_wavelength scale by factor
    let low_freqs_mask = gt(builder, wavelen.clone(), low_freq_wavelength);
    let inv_freq = cond(
        builder,
        low_freqs_mask.clone(),
        inv_freq.clone() / factor.clone(),
        inv_freq,
    );

    // if high_freq_wavelength < wavelen < low_freq_wavelength use a smooth interpolation
    let high_freqs_mask = lt(builder, wavelen.clone(), high_freq_wavelength);
    // Multiplication of masks is equivalent to logical AND
    let mid_freqs_mask = !high_freqs_mask * !low_freqs_mask;
    let one = constant(builder, inv_freq.label.clone(), 1.);
    let smooth_factor = (old_context_len / wavelen - low_freq_factor.clone())
        / (high_freq_factor - low_freq_factor);
    let smoothed_inv_freq = smooth_factor.clone() * inv_freq.clone()
        + (one - smooth_factor) * inv_freq.clone() / factor;
    let inv_freq = cond(builder, mid_freqs_mask, smoothed_inv_freq, inv_freq);

    let pos = arange(builder, seq_len, Dtype::F32);
    let pos = reshape(builder, Shape(vec![seq_len, 1]), pos);
    let pos = expand(builder, inv_freq.label.shape.clone(), pos);
    let pos = pos * inv_freq;
    let cos = cos(builder, pos.clone());
    let sin = sin(builder, pos);

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

fn rope_yarn_get_mscale(scale: f32) -> f32 {
    if scale <= 1.0 {
        return 1.0;
    }
    0.1 * scale.ln() + 1.0
}

fn find_correction_dim(
    num_rotations: f32,
    dim: usize,
    base: f32,
    max_position_embeddings: usize,
) -> f32 {
    (dim as f32 * (max_position_embeddings as f32 / (num_rotations * 2.0 * PI)).ln())
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
    let r = arange(builder, dim, Dtype::F32);
    let d = constant(builder, r.label.clone(), max - min);
    let min = constant(builder, r.label.clone(), min);
    let r = r - min;
    let r = r / d;
    clamp(builder, r, 0.0, 1.0)
}

pub fn rope_tables_yarn(
    builder: &Builder,
    theta: f32,
    rope_scaling: &YarnRopeScaling,
    seq_len: usize,
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

    let f = arange(builder, half_dim, Dtype::F32);
    let two = constant(builder, f.label.clone(), 2.0 / (head_dim as f32));
    let f = f * two;
    let theta = constant(builder, f.label.clone(), theta);
    let freq = power(builder, theta, f);

    let inv_freq_extrapolation = inverse(builder, freq);
    let inv_freq_extrapolation_factor = linear_ramp_factor(builder, low, high, half_dim);
    let one = constant(builder, inv_freq_extrapolation_factor.label.clone(), 1.);
    let inv_freq_extrapolation_factor = one.clone() - inv_freq_extrapolation_factor;

    let factor = constant(
        builder,
        inv_freq_extrapolation.label.clone(),
        rope_scaling.factor,
    );
    let inv_freq_interpolation = inv_freq_extrapolation.clone() / factor;

    let inv_freq = inv_freq_interpolation * (one - inv_freq_extrapolation_factor.clone())
        + inv_freq_extrapolation * inv_freq_extrapolation_factor;
    let inv_freq = expand(builder, Shape(vec![seq_len, half_dim]), inv_freq);

    let scale = rope_yarn_get_mscale(rope_scaling.factor);
    let scale = scalar(builder, Dtype::F32, scale);

    let pos = arange(builder, seq_len, Dtype::F32);
    let pos = reshape(builder, Shape(vec![seq_len, 1]), pos);
    let pos = expand(builder, inv_freq.label.shape.clone(), pos);
    let pos = pos * inv_freq;
    let scale = expand(builder, pos.label.shape.clone(), scale);
    let cos = cos(builder, pos.clone());
    let cos = cos * scale.clone();
    let sin = sin(builder, pos);
    let sin = sin * scale;

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

fn rotate_half(builder: &Builder, x: Var) -> Var {
    let v = chunk(builder, 3, 2, x);

    concat(builder, 3, -v[1].clone(), v[0].clone())
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by reusing calculated tables
pub fn apply_rope_embedding(builder: &Builder, pos: usize, cos: Var, sin: Var, x: Var) -> Var {
    let seq_len = x.label.shape.0[2];
    let cos = narrow(builder, 0, pos, seq_len, cos);
    let sin = narrow(builder, 0, pos, seq_len, sin);
    let cos = expand(builder, x.label.shape.clone(), cos);
    let sin = expand(builder, x.label.shape.clone(), sin);

    let xdtype = x.label.dtype;
    let mut x = x;
    if xdtype != Dtype::F32 {
        x = cast(builder, Dtype::F32, x);
    }
    let rotated_x = rotate_half(builder, x.clone());

    let mut r = cos * x + sin * rotated_x;
    if xdtype != Dtype::F32 {
        r = cast(builder, xdtype, r);
    }
    r
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by reusing calculated tables
/// in interleaved format
pub fn apply_rope_embedding_interleaved(
    builder: &Builder,
    pos: usize,
    cos: Var,
    sin: Var,
    x: Var,
) -> Var {
    // b, h, s, d = q.shape
    // q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    let x = reshape(
        builder,
        Shape(vec![
            x.label.shape.0[0],
            x.label.shape.0[1],
            x.label.shape.0[2],
            x.label.shape.0[3] / 2,
            2,
        ]),
        x,
    );
    let x = transpose(builder, 4, 3, x);
    let x = reshape(
        builder,
        Shape(vec![
            x.label.shape.0[0],
            x.label.shape.0[1],
            x.label.shape.0[2],
            x.label.shape.0[4] * 2,
        ]),
        x,
    );
    let seq_len = x.label.shape.0[2];
    let cos = narrow(builder, 0, pos, seq_len, cos);
    let sin = narrow(builder, 0, pos, seq_len, sin);
    let cos = expand(builder, x.label.shape.clone(), cos);
    let sin = expand(builder, x.label.shape.clone(), sin);

    let xdtype = x.label.dtype;
    let mut x = x;
    if xdtype != Dtype::F32 {
        x = cast(builder, Dtype::F32, x);
    }
    let rotated_x = rotate_half(builder, x.clone());

    let mut r = cos * x + sin * rotated_x;
    if xdtype != Dtype::F32 {
        r = cast(builder, xdtype, r);
    }
    r
}

/// Apply RoPE (Rotary Positional Embedding) to the input tensor by calculating the tables
pub fn rope(builder: &Builder, theta: f32, pos: usize, seq_len: usize, x: Var) -> Var {
    let head_dim = x.label.shape.0[3];
    let (cos, sin) = rope_tables(builder, theta, pos + seq_len, head_dim);

    apply_rope_embedding(builder, pos, cos, sin, x)
}
