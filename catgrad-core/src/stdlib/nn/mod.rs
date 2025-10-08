use crate::category::lang::*;
use crate::stdlib::def::*;
use std::f32::consts::{E, PI};

////////////////////////////////////////
// Sigmoid

pub struct Sigmoid;

impl Module<1, 1> for Sigmoid {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        // TODO: allow any dtype; cast constants in exp.
        use crate::check::*;
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

    // def
    fn def(&self, graph: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let c1 = constant_f32(graph, 1.0);
        let s = shape(graph, x.clone());
        let c1 = broadcast_to(graph, c1, s);

        let r = c1.clone() / (c1 + Exp.call(graph, [-x]));
        [r]
    }
}

////////////////////////////////////////
// Exp

pub struct Exp;

impl Module<1, 1> for Exp {
    // Type maps
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use crate::check::*;
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
        let e = constant_f32(graph, std::f32::consts::E);
        let e = cast(graph, e, dtype(graph, x.clone()));
        let s = shape(graph, x.clone());
        let e = broadcast_to(graph, e, s);
        [pow(graph, e, x)]
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
    let bmax = broadcast_to(builder, m, x_shape.clone());
    let x = x - bmax;
    let ex = exp(builder, x);
    let s = sum(builder, ex.clone());
    let bsum = broadcast_to(builder, s, x_shape);
    ex / bsum
}

pub fn chunk(builder: &Builder, dim: isize, chunks: usize, x: Var) -> Vec<Var> {
    let x_shape = shape(builder, x.clone());

    let sh = unpack::<3>(builder, x_shape);

    let d = sh[dim as usize].clone() / constant_nat(builder, chunks as u32);

    let ddim = constant_nat(builder, dim as u32);
    let mut outputs = vec![];
    for i in 0..chunks {
        let id = constant_nat(builder, i as u32) * d.clone();
        let s = slice(builder, ddim.clone(), id, d.clone(), x.clone());
        outputs.push(s);
    }

    outputs
}

pub fn causal_mask(builder: &Builder, size: Var) -> Var {
    let i = arange(builder, size.clone());
    let sh = pack::<2>(builder, [size.clone(), size.clone()]);
    let i = broadcast_to(builder, i, sh.clone());

    let one = constant_nat(builder, 1);
    let shr = pack::<2>(builder, [size.clone(), one]);
    let j = arange(builder, size);
    let j = reshape(builder, shr, j);
    let j = broadcast_to(builder, j, sh);

    let mask = lt(builder, j, i);

    let mask = cast(builder, mask, dtype_constant(builder, Dtype::F32));
    let sh = shape(builder, mask.clone());
    let ninf = constant(builder, f32::MIN, &sh);

    mask * ninf
}
