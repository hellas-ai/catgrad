use crate::abstract_interpreter::{CoreSSA, InterpreterError, Result};
use crate::category::core::Dtype;

// For now, type expressions are either completely opaque, or *concrete* lists of nat exprs.
// This means concat is partial: if any Var appears, we cannot handle it.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeExpr {
    Var(usize),
    NdArrayType(NdArrayType),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeExpr {
    Var(usize),
    OfType(usize), // shape of a *type* variable
    Shape(Vec<NatExpr>),
}

/// A symbolic type value
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NdArrayType {
    pub dtype: DtypeExpr,
    pub shape: ShapeExpr,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NatExpr {
    Var(usize),
    Constant(usize),
    Mul(Vec<NatExpr>),
    Add(Vec<NatExpr>),
}

// DtypeExpr::Var is allowed, but not as a top-level free variable in the program;
// it must resolve to a concrete value during shapechecking.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DtypeExpr {
    Var(usize),
    OfType(usize), // dtype of a *type* variable
    Constant(Dtype),
}

////////////////////////////////////////////////////////////////////////////////
// Helper impls

impl NatExpr {
    pub(crate) fn nf(&self) -> Self {
        use super::isomorphism::normalize;
        normalize(self)
    }
}

impl From<usize> for NatExpr {
    fn from(n: usize) -> Self {
        NatExpr::Constant(n)
    }
}

impl ShapeExpr {
    pub(crate) fn nf(&self) -> Self {
        match self {
            ShapeExpr::Var(_) => self.clone(),
            ShapeExpr::OfType(_) => self.clone(),
            ShapeExpr::Shape(nat_exprs) => {
                ShapeExpr::Shape(nat_exprs.iter().map(|m| m.nf()).collect())
            }
        }
    }
}

impl From<Vec<usize>> for ShapeExpr {
    fn from(dims: Vec<usize>) -> Self {
        ShapeExpr::Shape(dims.into_iter().map(Into::into).collect())
    }
}

impl DtypeExpr {
    pub(crate) fn nf(&self) -> Self {
        self.clone()
    }
}

impl From<Dtype> for DtypeExpr {
    fn from(value: Dtype) -> Self {
        DtypeExpr::Constant(value)
    }
}

impl TypeExpr {
    pub(crate) fn into_ndarraytype(self, ssa: &CoreSSA) -> Result<NdArrayType> {
        match self {
            Self::NdArrayType(t) => Ok(t),
            _ => Err(InterpreterError::TypeError(ssa.edge_id)),
        }
    }

    pub(crate) fn into_shapeexpr_dtype(self, ssa: &CoreSSA) -> Result<(ShapeExpr, DtypeExpr)> {
        match self {
            Self::NdArrayType(NdArrayType { shape, dtype }) => Ok((shape, dtype)),
            _ => Err(InterpreterError::TypeError(ssa.edge_id)),
        }
    }

    // Compute a normal form
    pub(crate) fn nf(&self) -> Self {
        match self {
            TypeExpr::Var(_) => todo!(),
            TypeExpr::NdArrayType(NdArrayType { dtype, shape }) => {
                TypeExpr::NdArrayType(NdArrayType {
                    dtype: dtype.nf(),
                    shape: shape.nf(),
                })
            }
        }
    }
}
