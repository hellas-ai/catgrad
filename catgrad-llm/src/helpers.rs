use crate::legacy::models::utils::Config;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;

/// Type signature for LLM Modules
pub fn llm_type() -> ([Type; 1], [Type; 1]) {
    use catgrad::typecheck::*;
    let batch_size = NatExpr::Var(0);
    let seq_len = NatExpr::Var(1);

    // Input shape B×S
    let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len]),
    }));

    // Output shape B×1
    let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(1)]),
    }));

    ([t_x], [t_y])
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
}

impl Cache {
    pub fn init(builder: &Builder, config: &Config, positions: usize) -> Self {
        let (cos, sin) = rope_tables(
            builder,
            config.rope_theta,
            positions.to_nat(builder),
            config.get_head_dim(),
        );

        Self { cos, sin }
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

pub fn rmsnorm_raw<const N: usize>(builder: &Builder, eps: f32, x: Var) -> Var {
    let x_shape = shape(builder, x.clone());
    let u = unpack::<N>(builder, x_shape.clone());
    let n = u[N - 1].clone();
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
pub fn rmsnorm<const N: usize>(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = rmsnorm_raw::<N>(builder, eps, x);
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

// Generate rope tables. This part is usually precomputed
pub fn rope_tables(builder: &Builder, theta: f32, seq_len: Var, head_dim: usize) -> (Var, Var) {
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
    x: Var,
) -> Var {
    let pos = pos.to_nat(builder);
    let seq_len = seq_len.to_nat(builder);
    let (cos, sin) = rope_tables(builder, theta, pos.clone() + seq_len, head_dim);

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
