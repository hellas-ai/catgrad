mod tensors;
pub use tensors::*;

use crate::config::{LLMConfig, Llama3RopeScaling, RopeScaling, YarnRopeScaling};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

/// Type signature for LLM Modules
pub fn llm_type(config: &dyn LLMConfig) -> (Vec<Type>, Vec<Type>) {
    use catgrad::typecheck::*;
    let batch_size = NatExpr::Var(0);
    let seq_len = NatExpr::Var(1);
    let cache_len = NatExpr::Var(2);

    // Input shape B×S
    let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len.clone()]),
    }));

    // Output shape B×1
    let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), NatExpr::Constant(1)]),
    }));

    let num_layers = NatExpr::Constant(config.num_kv_layers());
    let num_kv_heads = NatExpr::Constant(config.num_key_value_heads());
    let qk_head_dim = NatExpr::Constant(config.get_qk_head_dim());
    let v_head_dim = NatExpr::Constant(config.get_v_head_dim());

    let t_k = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            cache_len.clone(),
            qk_head_dim,
        ]),
    }));

    let t_v = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            cache_len.clone(),
            v_head_dim,
        ]),
    }));

    let out_cache_len = NatExpr::Add(vec![cache_len, seq_len]);
    let t_k_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            out_cache_len.clone(),
            NatExpr::Constant(config.get_qk_head_dim()),
        ]),
    }));

    let t_v_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers,
            batch_size,
            num_kv_heads,
            out_cache_len,
            NatExpr::Constant(config.get_v_head_dim()),
        ]),
    }));

    (vec![t_x, t_k, t_v], vec![t_y, t_k_out, t_v_out])
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub in_kv_cache: Vec<(Var, Var)>,
    pub out_kv_cache: Vec<(Var, Var)>,
    pub linear_state: Option<Vec<Var>>,
}

impl Cache {
    pub fn init(
        builder: &Builder,
        config: &dyn LLMConfig,
        positions: usize,
        in_k: Var,
        in_v: Var,
    ) -> Self {
        let (cos, sin) = match config.rope_scaling() {
            Some(RopeScaling::Yarn(params)) => rope_tables_yarn(
                builder,
                config.rope_theta(),
                &params,
                positions,
                config.get_head_dim(),
            ),
            Some(RopeScaling::Llama3(params)) => rope_tables_llama3(
                builder,
                config.rope_theta(),
                &params,
                positions,
                config.get_head_dim(),
            ),
            _ => rope_tables(
                builder,
                config.rope_theta(),
                positions,
                ((config.get_head_dim() as f32) * config.partial_rotary_factor()) as usize,
                1.0,
            ),
        };

        let num_kv_layers = config.num_kv_layers();
        let mut in_kv_cache = Vec::with_capacity(num_kv_layers);
        let mut out_kv_cache = Vec::with_capacity(num_kv_layers);
        for layer_id in 0..num_kv_layers {
            let k = slice(builder, 0, layer_id, 1, in_k.clone());
            let v = slice(builder, 0, layer_id, 1, in_v.clone());
            let k = squeeze::<5, 4>(builder, 0, k);
            let v = squeeze::<5, 4>(builder, 0, v);
            in_kv_cache.push((k.clone(), v.clone()));
            out_kv_cache.push((k, v));
        }

        Self {
            cos,
            sin,
            in_kv_cache,
            out_kv_cache,
            linear_state: None,
        }
    }

    pub fn update_kv_cache(
        &mut self,
        builder: &Builder,
        layer_id: usize,
        k: Var,
        v: Var,
    ) -> (Var, Var) {
        let cached_k = self.in_kv_cache[layer_id].0.clone();
        let cached_v = self.in_kv_cache[layer_id].1.clone();

        let k = concat(builder, 2, cached_k, k);
        let v = concat(builder, 2, cached_v, v);

        self.out_kv_cache[layer_id] = (k.clone(), v.clone());
        (k, v)
    }

    pub fn get_kv_cache(&self, builder: &Builder) -> (Var, Var) {
        let mut iter = self.out_kv_cache.iter();
        let (k0, v0) = iter.next().expect("no KV cache layers");
        let mut out_k = unsqueeze::<4, 5>(builder, 0, k0.clone());
        let mut out_v = unsqueeze::<4, 5>(builder, 0, v0.clone());
        for (k, v) in iter {
            let k = unsqueeze::<4, 5>(builder, 0, k.clone());
            let v = unsqueeze::<4, 5>(builder, 0, v.clone());
            out_k = concat(builder, 0, out_k, k);
            out_v = concat(builder, 0, out_v, v);
        }
        (out_k, out_v)
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPostProcess {
    None,
    ConcatMoeExperts { num_local_experts: usize },
}

pub trait LLMModel: DynModule {
    fn config(&self) -> &dyn LLMConfig;

    // Return empty KV-cache shape by default
    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let config = self.config();
        let k_shape = Shape(vec![
            config.num_kv_layers(),
            1,
            config.num_key_value_heads(),
            0,
            config.get_qk_head_dim(),
        ]);
        let v_shape = Shape(vec![
            config.num_kv_layers(),
            1,
            config.num_key_value_heads(),
            0,
            config.get_v_head_dim(),
        ]);
        vec![(Dtype::F32, k_shape), (Dtype::F32, v_shape)]
    }

    fn weight_post_process(&self) -> WeightPostProcess {
        WeightPostProcess::None
    }
}
