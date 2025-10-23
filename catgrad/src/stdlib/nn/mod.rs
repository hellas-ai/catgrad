use crate::prelude::{ops::*, *};
use std::f32::consts::{E, PI};

use crate::typecheck::*;

// A module with 1 input and 1 output used in nn
trait NNModule {
    fn name(&self) -> &str;
    fn definition(&self, builder: &Builder, x: [Var; 1]) -> [Var; 1];
}

impl<T: NNModule> Module<1, 1> for T {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Var(0),
            shape: ShapeExpr::Var(1),
        }));
        ([ty.clone()], [ty])
    }

    fn path(&self) -> Path {
        path(vec!["nn", self.name()]).unwrap()
    }
    fn def(&self, builder: &Builder, x: [Var; 1]) -> [Var; 1] {
        self.definition(builder, x)
    }
}

////////////////////////////////////////
// Sigmoid

pub struct Sigmoid;

impl NNModule for Sigmoid {
    fn name(&self) -> &str {
        "sigmoid"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [sigmoid(builder, x)]
    }
}

////////////////////////////////////////
// Exp

pub struct Exp;

impl NNModule for Exp {
    fn name(&self) -> &str {
        "exp"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [exp(builder, x)]
    }
}

pub struct Sqrt;

impl NNModule for Sqrt {
    fn name(&self) -> &str {
        "sqrt"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [sqrt(builder, x)]
    }
}

pub struct Gelu;

impl NNModule for Gelu {
    fn name(&self) -> &str {
        "gelu"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [gelu(builder, x)]
    }
}

pub struct Tanh;
impl NNModule for Tanh {
    fn name(&self) -> &str {
        "tanh"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [tanh(builder, x)]
    }
}

pub struct Silu;
impl NNModule for Silu {
    fn name(&self) -> &str {
        "silu"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [silu(builder, x)]
    }
}

pub struct Softmax;
impl NNModule for Softmax {
    fn name(&self) -> &str {
        "softmax"
    }
    fn definition(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [softmax(builder, x)]
    }
}

// Maybe turn these into Modules eventually

pub fn sqrt(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let e = constant(builder, 0.5, &sh);
    pow(builder, x, e)
}

pub fn exp(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let e = constant(builder, E, &sh);
    let e = cast(builder, e, dtype(builder, x.clone()));
    pow(builder, e, x)
}

pub fn sigmoid(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let one = constant(builder, 1.0, &sh);
    let one = cast(builder, one, dtype(builder, x.clone()));

    one.clone() / (one + exp(builder, -x))
}

pub fn tanh(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let one = constant(builder, 1.0, &sh);
    let two = constant(builder, 2.0, &sh);

    two.clone() * sigmoid(builder, two * x) - one
}

pub fn silu(builder: &Builder, x: Var) -> Var {
    x.clone() * sigmoid(builder, x)
}

// approx GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let c = constant(builder, f32::sqrt(2. / PI), &sh);
    let one = constant(builder, 1.0, &sh);
    let three = constant(builder, 3.0, &sh);
    let half = constant(builder, 0.5, &sh);
    let k = constant(builder, 0.044715, &sh);

    half * x.clone() * (one + tanh(builder, c * (x.clone() + k * (pow(builder, x, three)))))
}

pub fn softmax(builder: &Builder, x: Var) -> Var {
    let x_shape = shape(builder, x.clone());
    let m = max(builder, x.clone());
    let bmax = broadcast(builder, m, x_shape.clone());
    let x = x - bmax;
    let ex = exp(builder, x);
    let s = sum(builder, ex.clone());
    let bsum = broadcast(builder, s, x_shape);
    ex / bsum
}

pub fn chunk(builder: &Builder, dim: isize, chunks: usize, chunk_size: usize, x: Var) -> Vec<Var> {
    let d = lit(builder, nat(chunk_size as u32));
    let ddim = lit(builder, nat(dim as u32));
    let mut outputs = vec![];
    for i in 0..chunks {
        let id = lit(builder, nat(i as u32)) * d.clone();
        let s = slice(builder, ddim.clone(), id, d.clone(), x.clone());
        outputs.push(s);
    }

    outputs
}

pub fn causal_mask(builder: &Builder, size: Var) -> Var {
    let i = arange(builder, size.clone());
    let sh = pack::<2>(builder, [size.clone(), size.clone()]);
    let i = broadcast(builder, i, sh.clone());

    let one = lit(builder, nat(1));
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

pub fn linear_no_bias(builder: &Builder, _in_dim: usize, _out_dim: usize, p: Path, x: Var) -> Var {
    let w = param(builder, &p.extend(["weight"]).unwrap());

    let dim0 = lit(builder, nat(0));
    let dim1 = lit(builder, nat(1));
    let w_t = transpose(builder, dim0, dim1, w);

    // hack batch size
    let sh = shape(builder, w_t.clone());
    let [seq_len, hidden_dim] = unpack::<2>(builder, sh);
    let batch_size = lit(builder, nat(1));
    let sh = pack::<3>(builder, [batch_size, seq_len, hidden_dim]);

    let w_t = reshape(builder, sh, w_t);

    matmul(builder, x, w_t)
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

/// Add an additional dimension of extent 1 to a tensor
pub fn unsqueeze<const N: usize, const M: usize>(builder: &Builder, dim: usize, x: Var) -> Var {
    let x_shape = shape(builder, x.clone());
    let mut s = unpack::<N>(builder, x_shape).to_vec();
    s.insert(dim, lit(builder, nat(1)));
    let new_shape = pack::<M>(builder, s.try_into().unwrap());
    reshape(builder, new_shape, x)
}
