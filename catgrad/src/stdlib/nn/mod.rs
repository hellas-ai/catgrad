use crate::prelude::{ops::*, *};

use crate::typecheck::*;
use std::f32::consts::{E, PI};

////////////////////////////////////////
// Sigmoid

pub struct Sigmoid;

impl Module<1, 1> for Sigmoid {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        // TODO: allow any dtype; cast constants in exp.
        let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Var(0),
        }));
        ([ty.clone()], [ty])
    }

    // Name of the op
    fn path(&self) -> Path {
        path(vec!["nn", "sigmoid"]).unwrap()
    }

    // Shape-polymorphic sigmoid
    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let s = shape(builder, x.clone());
        let c1 = constant(builder, 1.0, &s);
        let r = c1.clone() / (c1 + Exp.call(builder, [-x]));
        [r]
    }
}

////////////////////////////////////////
// Exp

pub struct Exp;

impl Module<1, 1> for Exp {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Var(0),
            shape: ShapeExpr::Var(1),
        }));
        ([ty.clone()], [ty])
    }

    // Name of the op
    fn path(&self) -> Path {
        path(vec!["nn", "exp"]).unwrap()
    }

    // def
    fn def(&self, graph: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        // we'll cast e to whatever dtype x has.
        let e = lit(graph, std::f32::consts::E);
        let e = cast(graph, e, dtype(graph, x.clone()));
        let s = shape(graph, x.clone());
        let e = broadcast(graph, e, s);
        [pow(graph, e, x)]
    }
}

pub struct Sqrt;

impl Module<1, 1> for Sqrt {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Var(0),
            shape: ShapeExpr::Var(1),
        }));
        ([ty.clone()], [ty])
    }

    // Name of the op
    fn path(&self) -> Path {
        path(vec!["nn", "sqrt"]).unwrap()
    }

    // def
    fn def(&self, graph: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let sh = shape(graph, x.clone());
        let e = constant(graph, 0.5, &sh);
        [pow(graph, x, e)]
    }
}

pub struct Gelu;

impl Module<1, 1> for Gelu {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let ty = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Var(0),
            shape: ShapeExpr::Var(1),
        }));
        ([ty.clone()], [ty])
    }

    // Name of the op
    fn path(&self) -> Path {
        path(vec!["nn", "gelu"]).unwrap()
    }

    // def
    fn def(&self, graph: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        [gelu(graph, x)]
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
    pow(builder, e, x)
}

pub fn sigmoid(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let one = constant(builder, 1.0, &sh);

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

pub fn repeat_kv(builder: &Builder, rep: usize, x: Var) -> Var {
    let shape = shape(builder, x.clone());
    let [b, num_kv_heads, s, head_dim] = unpack::<4>(builder, shape);

    let sh = shape!(builder, b, num_kv_heads, 1, s, head_dim);
    // equivalent of torch.repeat_interleave across dim 1
    let x = reshape(builder, sh, x);
    let sh = shape!(builder, b, num_kv_heads, rep, s, head_dim);

    let x = broadcast(builder, x, sh);

    let rnkv = num_kv_heads * lit(builder, nat(rep as u32));
    let sh = shape!(builder, b, rnkv, s, head_dim);
    reshape(builder, sh, x)
}
