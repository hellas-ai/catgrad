//! Abstract syntax for catgrad's surface language.

use open_hypergraphs::lax::*;
use std::fmt;

pub use crate::category::core::{Dtype, Object};
pub use crate::typecheck::Type;

use crate::path::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    F32(f32),
    U32(u32),
    Nat(u32),
    Dtype(Dtype),
}

// Operations are the core shape operations (and operation schemas like 'Constant') extended
// with "Definitions".
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Operations declared to exist, but without a definition (externally interpreted)
    Declaration(Path),

    /// Extended set of base operations with definitions
    Definition(Path),

    /// Literals (floats, ints, dtypes, nats, etc.) are elements (maps with unit domain)
    Literal(Literal),
}

pub type Term = OpenHypergraph<Object, Operation>;
pub type Var = open_hypergraphs::lax::var::Var<Object, Operation>;

/// A TypedTerm is one with source and target type specified as 'type maps' (TODO!)
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct TypedTerm {
    pub term: Term,
    pub source_type: Vec<Type>,
    pub target_type: Vec<Type>,
}

use std::cell::RefCell;
use std::rc::Rc;
pub type Builder = Rc<RefCell<Term>>;

////////////////////////////////////////////////////////////////////////////////
// Instances

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::F32(v) => write!(f, "{v}"),
            Literal::U32(v) => write!(f, "{v}"),
            Literal::Nat(v) => write!(f, "{v}"),
            Literal::Dtype(dtype) => write!(f, "{dtype:?}"),
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Declaration(path) => write!(f, "{path}"),
            Operation::Definition(path) => write!(f, "{path}"),
            Operation::Literal(literal) => write!(f, "{literal}"),
        }
    }
}

impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Literal::F32(value)
    }
}

impl From<u32> for Literal {
    fn from(value: u32) -> Self {
        Literal::U32(value)
    }
}

impl From<Dtype> for Literal {
    fn from(value: Dtype) -> Self {
        Literal::Dtype(value)
    }
}
