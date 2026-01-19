//! Standard library of definitions and APIs for neural networks and LLMs
//! These are still in flux and driven by model code using them.
//! The APIs may change and parts are likely to be moved elsewhere.
//! Some are lower level components like linear layers and activations, others are
//! higher level LLM building blocks or APIs.

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

/// Generic linear layer with optional bias with already loaded parameters given as vars
pub fn linear_b_param(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    weight: Var,
    bias: Option<Var>,
    x: Var,
) -> Var {
    let w_t = transpose(builder, 0, 1, weight);

    let sh = shape(builder, x.clone());
    let [batch_size, _, _] = unpack::<3>(builder, sh);
    let in_dim = in_dim.to_nat(builder);
    let out_dim = out_dim.to_nat(builder);
    let sh = pack::<3>(
        builder,
        [1.to_nat(builder), in_dim.clone(), out_dim.clone()],
    );

    let w_t = reshape(builder, sh, w_t);
    let sh = pack::<3>(builder, [batch_size, in_dim, out_dim]);
    let w_t = broadcast(builder, w_t, sh);

    let m = matmul(builder, x, w_t);
    if let Some(b) = bias {
        let sh = shape(builder, m.clone());
        let bb = broadcast(builder, b, sh);
        return m + bb;
    }
    m
}

pub fn linear_no_bias_param(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    weight: Var,
    x: Var,
) -> Var {
    linear_b_param(builder, in_dim, out_dim, weight, None, x)
}

pub fn linear_param(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    weight: Var,
    bias: Var,
    x: Var,
) -> Var {
    linear_b_param(builder, in_dim, out_dim, weight, Some(bias), x)
}

/// Generic linear layer with optional bias with given parameter names
pub fn linear_b(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    p: Path,
    x: Var,
) -> Var {
    let weight = param(builder, &p.extend(["weight"]).unwrap());
    let bias = if bias {
        Some(param(builder, &p.extend(["bias"]).unwrap()))
    } else {
        None
    };
    linear_b_param(builder, in_dim, out_dim, weight, bias, x)
}

pub fn linear_no_bias(builder: &Builder, in_dim: usize, out_dim: usize, p: Path, x: Var) -> Var {
    linear_b(builder, in_dim, out_dim, false, p, x)
}

pub fn linear(builder: &Builder, in_dim: usize, out_dim: usize, p: Path, x: Var) -> Var {
    linear_b(builder, in_dim, out_dim, true, p, x)
}
