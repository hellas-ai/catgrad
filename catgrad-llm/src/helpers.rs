use crate::config::{LLMConfig, Llama3RopeScaling, RopeScaling, YarnRopeScaling};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;

/// Type signature for LLM Modules
pub fn llm_type(config: &dyn LLMConfig) -> ([Type; 3], [Type; 3]) {
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

    let num_layers = NatExpr::Constant(config.num_hidden_layers());
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

    ([t_x, t_k, t_v], [t_y, t_k_out, t_v_out])
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub in_kv_cache: Vec<(Var, Var)>,
    pub out_kv_cache: Vec<(Var, Var)>,
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
                positions.to_nat(builder),
                ((config.get_head_dim() as f32) * config.partial_rotary_factor()) as usize,
                1.0,
            ),
        };

        let mut in_kv_cache = Vec::with_capacity(config.num_hidden_layers());
        let mut out_kv_cache = Vec::with_capacity(config.num_hidden_layers());
        for layer_id in 0..config.num_hidden_layers() {
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

pub fn chunk(builder: &Builder, dim: isize, chunks: usize, chunk_size: usize, x: Var) -> Vec<Var> {
    let mut outputs = vec![];
    for i in 0..chunks {
        let s = slice(builder, dim as u32, i * chunk_size, chunk_size, x.clone());
        outputs.push(s);
    }

    outputs
}

pub fn split(builder: &Builder, dim: isize, sizes: &[usize], x: Var) -> Vec<Var> {
    let mut outputs = vec![];
    let mut offset = 0;
    for &size in sizes {
        let s = slice(builder, dim as u32, offset, size, x.clone());
        outputs.push(s);
        offset += size;
    }

    outputs
}

#[allow(dead_code)]
pub fn squeeze<const N: usize, const M: usize>(builder: &Builder, dim: usize, x: Var) -> Var {
    assert_eq!(N, M + 1);
    let x_shape = shape(builder, x.clone());
    let mut s = unpack::<N>(builder, x_shape).to_vec();
    s.remove(dim);
    let new_shape = pack::<M>(builder, s.try_into().unwrap());
    reshape(builder, new_shape, x)
}

pub fn unsqueeze<const N: usize, const M: usize>(builder: &Builder, dim: usize, x: Var) -> Var {
    assert_eq!(N + 1, M);
    let x_shape = shape(builder, x.clone());
    let mut s = unpack::<N>(builder, x_shape).to_vec();
    s.insert(dim, 1.to_nat(builder));
    let new_shape = pack::<M>(builder, s.try_into().unwrap());
    reshape(builder, new_shape, x)
}

pub fn layernorm_raw(builder: &Builder, eps: f32, x: Var) -> Var {
    let x_shape = shape(builder, x.clone());
    let [_, _, n] = unpack::<3>(builder, x_shape.clone());
    let s = sum(builder, x.clone());

    let constn = nat_to_u32(builder, n);
    let constn = cast(builder, constn, dtype(builder, x.clone()));
    let sh = shape(builder, s.clone());
    let constn = broadcast(builder, constn, sh);

    let mean = s / constn.clone();
    let nom = x - broadcast(builder, mean, x_shape.clone());

    let var = sum(builder, nom.clone() * nom.clone()) / constn;
    let sh = shape(builder, var.clone());
    let epsilon = constant(builder, eps, &sh);
    let stddev = sqrt(builder, var + epsilon);
    let denom = broadcast(builder, stddev, x_shape);

    nom / denom
}

pub fn layernorm(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = layernorm_raw(builder, eps, x);
    let lr_shape = shape(builder, lr.clone());
    let gamma = broadcast(builder, gamma, lr_shape.clone());
    let lr = lr * gamma;

    let beta = param(builder, &p.extend(["bias"]).unwrap());
    let beta = broadcast(builder, beta, lr_shape);
    lr + beta
}

pub fn rmsnorm_raw(builder: &Builder, eps: f32, x: Var) -> Var {
    let x_shape = shape(builder, x.clone());
    let [_, _, n] = unpack::<3>(builder, x_shape.clone());
    let s = sum(builder, x.clone() * x.clone());

    let constn = nat_to_u32(builder, n);
    let constn = cast(builder, constn, dtype(builder, x.clone()));
    let sh = shape(builder, s.clone());
    let constn = broadcast(builder, constn, sh);

    let mean = s / constn;

    let epsilon = constant(builder, eps, &shape(builder, mean.clone()));
    let rms = sqrt(builder, mean + epsilon);
    let denom = broadcast(builder, rms, x_shape);
    x / denom
}

// rmsnorm(x) = x / √(E[x²] + ε) × γ
pub fn rmsnorm(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = rmsnorm_raw(builder, eps, x);
    let lr_shape = shape(builder, lr.clone());
    let gamma = broadcast(builder, gamma, lr_shape);
    lr * gamma
}

pub fn repeat_kv(builder: &Builder, rep: usize, x: Var) -> Var {
    let shape = shape(builder, x.clone());
    let [b, num_kv_heads, s, head_dim] = unpack::<4>(builder, shape);

    let sh = shape!(builder, b, num_kv_heads, 1, s, head_dim);
    // equivalent of torch.repeat_interleave across dim 1
    let x = reshape(builder, sh, x);
    let sh = shape!(builder, b, num_kv_heads, rep, s, head_dim);

    let x = broadcast(builder, x, sh);

    let rnkv = num_kv_heads * rep.to_nat(builder);
    let sh = shape!(builder, b, rnkv, s, head_dim);
    reshape(builder, sh, x)
}

/// Average pooling over a square 2D grid.
pub fn avgpool2d(builder: &Builder, dim: usize, side: usize, k: usize, x: Var) -> Var {
    let windows = side / k;
    let x = reshape(builder, shape!(builder, 1, dim, windows, k, windows, k), x);
    let x = transpose(builder, 3, 4, x);
    let x = reshape(
        builder,
        shape!(builder, 1, dim, windows * windows, k * k),
        x,
    );

    let x = sum(builder, x);
    let sh = shape(builder, x.clone());
    let d = constant(builder, 1.0 / ((k * k) as f32), &sh);
    x * d
}

// Generate rope tables. This part is usually precomputed
pub fn rope_tables(
    builder: &Builder,
    theta: f32,
    seq_len: Var,
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

    let sh = shape!(builder, seq_len, half_dim);
    let inv_freq = broadcast(builder, inv_freq, sh.clone());

    let factor = constant(builder, factor, &sh);
    let inv_freq = inv_freq / factor;

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, pos, sh);
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
    let inv_freq = broadcast(builder, inv_freq, sh);

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, pos, sh);
    let pos = pos * inv_freq;
    let cos = cos(builder, pos.clone());
    let sin = sin(builder, pos);

    let cos = concat(builder, 1, cos.clone(), cos);
    let sin = concat(builder, 1, sin.clone(), sin);

    (cos, sin)
}

fn clamp(builder: &Builder, x: Var, min_val: f32, max_val: f32) -> Var {
    let sh = shape(builder, x.clone());
    let min_t = constant(builder, min_val, &sh);
    let max_t = constant(builder, max_val, &sh);
    let one = constant(builder, 1.0, &sh);

    let mask_min = lt(builder, x.clone(), min_t.clone());
    let mask_min = cast(builder, mask_min, Dtype::F32);
    let x = mask_min.clone() * min_t + (one.clone() - mask_min) * x;

    let mask_max = lt(builder, max_t.clone(), x.clone());
    mask_max.clone() * max_t + (one - mask_max) * x
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
    let inv_freq = broadcast(builder, inv_freq, sh);

    let scale = rope_yarn_get_mscale(rope_scaling.factor);
    let sh = shape!(builder, 1);
    let scale = constant(builder, scale, &sh);

    let pos = arange(builder, seq_len.clone());
    let pos = cast(builder, pos, Dtype::F32);
    let sh = shape!(builder, seq_len, 1);
    let pos = reshape(builder, sh, pos);
    let sh = shape(builder, inv_freq.clone());
    let pos = broadcast(builder, pos, sh.clone());
    let pos = pos * inv_freq;
    let scale = broadcast(builder, scale, sh);
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
    let cos = broadcast(builder, cos, sh.clone());
    let sin = broadcast(builder, sin, sh);

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

pub fn causal_mask(builder: &Builder, size: Var) -> Var {
    let i = arange(builder, size.clone());
    let sh = pack::<2>(builder, [size.clone(), size.clone()]);
    let i = broadcast(builder, i, sh.clone());

    let one = 1.to_nat(builder);
    let shr = pack::<2>(builder, [size.clone(), one]);
    let j = arange(builder, size);
    let j = reshape(builder, shr, j);
    let j = broadcast(builder, j, sh);

    let mask = lt(builder, j, i);

    let mask = cast(builder, mask, Dtype::F32);
    let sh = shape(builder, mask.clone());
    let ninf = constant(builder, f32::MIN, &sh);

    mask * ninf
}

pub fn embeddings(builder: &Builder, p: Path, x: Var) -> Var {
    let wte = param(builder, &p.extend(vec!["weight"]).unwrap());

    //flatten the input tensor as that is how index expects it
    let [b, s] = unpack::<2>(builder, shape(builder, x.clone()));
    let sh = shape!(builder, b * s);
    let x = reshape(builder, sh, x);

    //index into the weight tensor
    let te = index(builder, 0, x, wte);

    unsqueeze::<2, 3>(builder, 0, te)
}

pub trait LLMModel: Module<3, 3> {}
