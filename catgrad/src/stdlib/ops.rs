/// Helpers wrapping `crate::category::lang::ops` methods
use crate::category::lang::{Literal, ops};
use crate::prelude::{Builder, Var};

// re-export lang ops
pub use ops::{
    argmax, broadcast, cos, dtype, dtype_constant, lt, matmul, max, nat_to_u32, pack, param, pow,
    reshape, shape, sin, sum, unpack,
};

pub fn cast(builder: &Builder, x: Var, d: impl IntoDtypeVar) -> Var {
    ops::cast(builder, x, d.to_nat(builder))
}

pub fn arange(builder: &Builder, end: impl IntoNatVar) -> Var {
    ops::arange(builder, end.to_nat(builder))
}

pub fn concat(builder: &Builder, dim: impl IntoNatVar, x: Var, y: Var) -> Var {
    ops::concat(builder, dim.to_nat(builder), x, y)
}

pub fn topk(builder: &Builder, k: impl IntoNatVar, x: Var) -> (Var, Var) {
    ops::topk(builder, k.to_nat(builder), x)
}

pub fn index(builder: &Builder, dim: impl IntoNatVar, idx: Var, x: Var) -> Var {
    ops::index(builder, dim.to_nat(builder), idx, x)
}

pub fn slice(
    builder: &Builder,
    dim: impl IntoNatVar,
    start: impl IntoNatVar,
    len: impl IntoNatVar,
    x: Var,
) -> Var {
    ops::slice(
        builder,
        dim.to_nat(builder),
        start.to_nat(builder),
        len.to_nat(builder),
        x,
    )
}

pub fn get(builder: &Builder, dim: impl IntoNatVar, start: impl IntoNatVar, x: Var) -> Var {
    slice(builder, dim, start, 1, x)
}

/// Language literals
pub fn lit<T: Into<Literal>>(builder: &Builder, x: T) -> Var {
    ops::lit(builder, x.into())
}

/// Make a nat from a u32
pub fn nat(x: u32) -> Literal {
    Literal::Nat(x)
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

// TODO: helper to make a Nat into a tensor + cast
// hmmm

/// Transpose a tensor using either symbolic (Var) or static (u32) dims
pub fn transpose(builder: &Builder, a: impl IntoNatVar, b: impl IntoNatVar, x: Var) -> Var {
    ops::transpose(builder, a.to_nat(builder), b.to_nat(builder), x)
}

////////////////////////////////////////////////////////////////////////////////
// Types convertible to a Var representing a Nat

pub trait IntoNatVar {
    fn to_nat(&self, builder: &Builder) -> Var;
}

impl IntoNatVar for Var {
    fn to_nat(&self, _builder: &Builder) -> Var {
        self.clone()
    }
}

impl IntoNatVar for u32 {
    fn to_nat(&self, builder: &Builder) -> Var {
        lit(builder, nat(*self))
    }
}

impl IntoNatVar for i32 {
    fn to_nat(&self, builder: &Builder) -> Var {
        lit(builder, nat((*self).try_into().unwrap()))
    }
}

impl IntoNatVar for usize {
    fn to_nat(&self, builder: &Builder) -> Var {
        lit(builder, nat((*self).try_into().unwrap()))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Types convertible to a Var representing a Dtype

pub trait IntoDtypeVar {
    fn to_nat(&self, builder: &Builder) -> Var;
}

impl IntoDtypeVar for crate::category::core::Dtype {
    fn to_nat(&self, builder: &Builder) -> Var {
        dtype_constant(builder, crate::category::core::Dtype::F32)
    }
}

impl IntoDtypeVar for Var {
    fn to_nat(&self, _builder: &Builder) -> Var {
        self.clone()
    }
}
