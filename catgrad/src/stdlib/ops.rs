/// Helpers wrapping `crate::category::lang::ops` methods
use crate::category::lang::{Literal, ops, ops::IntoNatVar};
use crate::prelude::{Builder, Var};

// re-export lang ops
pub use ops::{
    arange, argmax, broadcast, cast, concat, cos, dtype, dtype_constant, index, lt, matmul, max,
    nat_to_u32, pack, param, pow, reshape, shape, sin, slice, sum, topk, transpose, unpack,
};

pub fn get(builder: &Builder, dim: impl IntoNatVar, start: impl IntoNatVar, x: Var) -> Var {
    ops::slice(builder, dim, start, 1, x)
}

/// Language literals
pub fn lit<T: Into<Literal>>(builder: &Builder, x: T) -> Var {
    ops::lit(builder, x.into())
}

pub fn constant<T: Into<Literal>>(builder: &Builder, x: T, s: &Var) -> Var {
    let x = lit(builder, x);
    ops::broadcast(builder, x, s.clone())
}

pub fn inverse(builder: &Builder, x: Var) -> Var {
    let shape = shape(builder, x.clone());
    let one = constant(builder, 1.0, &shape);
    one / x
}
