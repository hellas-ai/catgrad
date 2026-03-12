use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;

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
    let constn = broadcast(builder, sh, constn);

    let mean = s / constn.clone();
    let nom = x - broadcast(builder, x_shape.clone(), mean);

    let var = sum(builder, nom.clone() * nom.clone()) / constn;
    let sh = shape(builder, var.clone());
    let epsilon = constant(builder, eps, &sh);
    let stddev = sqrt(builder, var + epsilon);
    let denom = broadcast(builder, x_shape, stddev);

    nom / denom
}

pub fn layernorm(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = layernorm_raw(builder, eps, x);
    let lr_shape = shape(builder, lr.clone());
    let gamma = broadcast(builder, lr_shape.clone(), gamma);
    let lr = lr * gamma;

    let beta = param(builder, &p.extend(["bias"]).unwrap());
    let beta = broadcast(builder, lr_shape, beta);
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
    let constn = broadcast(builder, sh, constn);

    let mean = s / constn;

    let epsilon = constant(builder, eps, &shape(builder, mean.clone()));
    let rms = sqrt(builder, mean + epsilon);
    let denom = broadcast(builder, x_shape, rms);
    x / denom
}

// rmsnorm(x) = x / √(E[x²] + ε) × γ
pub fn rmsnorm<const N: usize>(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = rmsnorm_raw::<N>(builder, eps, x);
    let sh = shape(builder, lr.clone());
    let gamma = broadcast(builder, sh, gamma);
    lr * gamma
}

// A variant of RMSNorm used by Gemma 3 and Qwen 3.5
pub fn rmsnorm_gemma<const N: usize>(builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
    let gamma = param(builder, &p.extend(["weight"]).unwrap());
    let lr = rmsnorm_raw::<N>(builder, eps, x);
    let sh = shape(builder, lr.clone());
    let one = constant(builder, 1.0, &sh);
    let gamma = broadcast(builder, sh, gamma);
    lr * (gamma + one)
}

pub fn repeat_kv(builder: &Builder, rep: usize, x: Var) -> Var {
    let shape = shape(builder, x.clone());
    let [b, num_kv_heads, s, head_dim] = unpack::<4>(builder, shape);

    let sh = shape!(builder, b, num_kv_heads, 1, s, head_dim);
    // equivalent of torch.repeat_interleave across dim 1
    let x = reshape(builder, sh, x);
    let sh = shape!(builder, b, num_kv_heads, rep, s, head_dim);

    let x = broadcast(builder, sh, x);

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

pub fn clamp(builder: &Builder, x: Var, min_val: f32, max_val: f32) -> Var {
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

pub fn causal_mask(builder: &Builder, seq_len: Var, pos: Var) -> Var {
    let size = seq_len.clone() + pos.clone();
    let i = arange(builder, size.clone());
    let sh = shape!(builder, size, size);
    let i = broadcast(builder, sh.clone(), i);

    let shr = shape!(builder, size, 1);
    let j = arange(builder, size);
    let j = reshape(builder, shr, j);
    let j = broadcast(builder, sh.clone(), j);

    let mask = lt(builder, j, i);
    let mask = cast(builder, mask, Dtype::F32);
    let ninf = constant(builder, f32::MIN, &sh);
    let mask = mask * ninf;

    slice(builder, 0, pos, seq_len, mask)
}

pub fn embeddings(builder: &Builder, p: Path, x: Var) -> Var {
    let wte = param(builder, &p.extend(["weight"]).unwrap());

    //flatten the input tensor as that is how index expects it
    let [b, s] = unpack::<2>(builder, shape(builder, x.clone()));
    let sh = shape!(builder, b.clone() * s.clone());
    let x = reshape(builder, sh, x);

    let sh = shape(builder, wte.clone());
    let [_vocab_size, hidden_dim] = unpack::<2>(builder, sh);

    //index into the weight tensor
    let te = index(builder, 0, x, wte);

    // add back batch dimension
    let sh = shape!(builder, b, s, hidden_dim);
    reshape(builder, sh, te)
}

// Select values from `x` where `mask` is 1, otherwise from `y`.
// Wrap the where_cond op but broadcast and cast the mask to match `x`'s dtype.
pub fn where_broadcast(builder: &Builder, mask: Var, x: Var, y: Var) -> Var {
    let sh = shape(builder, x.clone());
    let x_dtype = dtype(builder, x.clone());
    let mask = broadcast(builder, sh, mask);
    let mask = cast(builder, mask, x_dtype);
    where_cond(builder, mask, x, y)
}

// Fill `x` with `fill` where `mask` is 1, otherwise leave `x` unchanged.
pub fn masked_fill(builder: &Builder, mask: Var, fill: f32, x: Var) -> Var {
    let fill = constant(builder, fill, &shape(builder, x.clone()));
    where_cond(builder, mask, fill, x)
}

// Make a diagonal matrix with ones on the diagonal and zeros elsewhere.
pub fn eye(builder: &Builder, size: Var, dtype: Dtype) -> Var {
    let row = arange(builder, size.clone());
    let col = arange(builder, size.clone());

    let row = reshape(builder, shape!(builder, size, 1), row);
    let row = broadcast(builder, shape!(builder, size.clone(), size), row);

    let col = reshape(builder, shape!(builder, 1, size), col);
    let col = broadcast(builder, shape!(builder, size.clone(), size), col);

    let eye = eq(builder, row, col);
    cast(builder, eye, dtype)
}

fn tri_mask(builder: &Builder, size: Var, diagonal: u32, triu: bool) -> Var {
    let sh = shape!(builder, size, size);
    let diagonal = constant(builder, diagonal, &sh);
    let idx = arange(builder, size.clone());
    let row = reshape(builder, shape!(builder, size, 1), idx.clone());
    let col = reshape(builder, shape!(builder, 1, size), idx);

    let row = broadcast(builder, sh.clone(), row);
    let col = broadcast(builder, sh, col);

    let row = row + diagonal;
    // Upper-triangular entries satisfy `col >= row + diagonal`.
    // Lower-triangular entries satisfy `col <= row + diagonal`.
    let mask = if triu {
        gte(builder, col, row)
    } else {
        lte(builder, col, row)
    };
    cast(builder, mask, Dtype::F32)
}

// Upper-riangular mask
pub fn triu_mask(builder: &Builder, size: Var, diagonal: u32) -> Var {
    tri_mask(builder, size, diagonal, true)
}

// Lower-triangular mask
pub fn tril_mask(builder: &Builder, size: Var, diagonal: u32) -> Var {
    tri_mask(builder, size, diagonal, false)
}

// Cumsum along the last dimension of a tensor
pub fn cumsum<const N: usize>(builder: &Builder, x: Var) -> Var {
    let sh = unpack::<N>(builder, shape(builder, x.clone()));
    let ldim = sh[N - 1].clone();

    let mut new_sh = sh;
    new_sh[N - 2] = ldim.clone();

    let new_sh = pack::<N>(builder, new_sh);
    let lower = tril_mask(builder, ldim, 0);
    let lower = transpose(builder, 0, 1, lower);

    let lower = broadcast(builder, new_sh, lower);

    matmul(builder, x, lower)
}

pub fn zeros(builder: &Builder, shape: &Var) -> Var {
    constant(builder, 0.0, shape)
}
