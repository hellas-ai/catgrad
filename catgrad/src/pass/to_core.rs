//! # Compiler pass: Lang to Core
//!
//! Functions for mapping [`lang`] terms and operations into [`core`].
use std::collections::HashMap;

use crate::category::{core, lang};
use crate::definition::Def;
use crate::path::Path;
use std::fmt::Debug;

/// [`Environment`] tells the [`to_core`] pass how to translate
/// from `lang` to `core`. It has two dicts:
///
/// - `declarations`: a list of *declared* ops which map directly to a [`core::Operation`]
/// - `definitions`: a list of *defined* ops, expressed in terms of another [`lang::TypedTerm`]
///
/// Together these let us map either a [`lang::Operation`] or [`lang::Term`] to a [`core::Term`].
#[derive(Clone, Debug)]
pub struct Environment {
    pub declarations: HashMap<Path, core::Operation>,
    pub definitions: HashMap<Path, lang::TypedTerm>,
}

impl Environment {
    pub fn op_to_core(&self, op: lang::Operation) -> Def<Path, core::Operation> {
        op_to_core(op, &self.declarations)
    }

    pub fn to_core(&self, term: lang::Term) -> core::Term {
        to_core(term, &self.declarations)
    }
}

/// Lower a `lang::Term` to a `core::Term`.
/// If a Definition or Declaration was not a core operation, it is assumed to be a definition in
/// [`core`].
pub fn to_core(term: lang::Term, core_ops: &HashMap<lang::Path, core::Operation>) -> core::Term {
    // **NOTE: this function *must* preserve node and edge IDs of input term**
    // In future, if we need to map a lang::Path to *multiple* core operations, preserving edge IDs
    // must be done by definitions; we can assume map_edges and map_nodes don't change IDs.
    // This allows us to map errors and type inference back up to the original input term.
    term.map_edges(|e| op_to_core(e, core_ops))
}

/// Map a lang::Operation to a core::Operation (plus definitions)
fn op_to_core(
    op: lang::Operation,
    core_ops: &HashMap<lang::Path, core::Operation>,
) -> Def<Path, core::Operation> {
    match op {
        lang::Operation::Definition(path) => Def::Def(path),
        lang::Operation::Declaration(path) => match core_ops.get(&path) {
            Some(op) => Def::Arr(op.clone()),
            None => Def::Def(path.clone()),
        },
        lang::Operation::Literal(lit) => Def::Arr(match lit {
            lang::Literal::F32(x) => {
                core::Operation::Tensor(core::TensorOp::Scalar(core::Scalar::F32(x)))
            }
            lang::Literal::U32(x) => {
                core::Operation::Tensor(core::TensorOp::Scalar(core::Scalar::U32(x)))
            }
            lang::Literal::Nat(n) => core::Operation::Nat(core::NatOp::Constant(n as usize)),
            lang::Literal::Dtype(d) => core::Operation::DtypeConstant(d),
        }),
    }
}

/// Lower an entire stdlib::Environment to the core, discarding type maps.
pub fn env_to_core(env: Environment) -> Environment {
    env
}

////////////////////////////////////////////////////////////////////////////////
// NOTE: below is duplicated from stdlib/ops.rs.
// These should eventually replace that duplicate code!

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        $crate::path::path(vec![$($x),*]).expect("invalid operation name")
    };
}

/// Interpretations of declared operations
pub(crate) fn core_declarations() -> HashMap<lang::Path, core::Operation> {
    use crate::category::core::{NatOp, Operation, ScalarOp::*, TensorOp::*, TypeOp};
    HashMap::from([
        (path!["cartesian", "copy"], Operation::Copy),
        // tensor ops (which actually affect tensor data)
        (path!["tensor", "add"], Operation::Tensor(Map(Add))),
        (path!["tensor", "sub"], Operation::Tensor(Map(Sub))),
        (path!["tensor", "neg"], Operation::Tensor(Map(Neg))),
        (path!["tensor", "mul"], Operation::Tensor(Map(Mul))),
        (path!["tensor", "div"], Operation::Tensor(Map(Div))),
        (path!["tensor", "pow"], Operation::Tensor(Map(Pow))),
        (path!["tensor", "sin"], Operation::Tensor(Map(Sin))),
        (path!["tensor", "cos"], Operation::Tensor(Map(Cos))),
        (path!["tensor", "lt"], Operation::Tensor(Map(LT))),
        (path!["tensor", "eq"], Operation::Tensor(Map(EQ))),
        (path!["tensor", "matmul"], Operation::Tensor(MatMul)),
        (path!["tensor", "reshape"], Operation::Tensor(Reshape)),
        (path!["tensor", "transpose"], Operation::Tensor(Transpose)),
        (path!["tensor", "broadcast"], Operation::Tensor(Broadcast)),
        (path!["tensor", "cast"], Operation::Tensor(Cast)),
        (path!["tensor", "index"], Operation::Tensor(Index)),
        (path!["tensor", "slice"], Operation::Tensor(Slice)),
        (path!["tensor", "sum"], Operation::Tensor(Sum)),
        (path!["tensor", "max"], Operation::Tensor(Max)),
        (path!["tensor", "argmax"], Operation::Tensor(Argmax)),
        (path!["tensor", "topk"], Operation::Tensor(TopK)),
        (path!["tensor", "arange"], Operation::Tensor(Arange)),
        (path!["tensor", "concat"], Operation::Tensor(Concat)),
        (path!["tensor", "nat_to_u32"], Operation::Tensor(NatToU32)),
        // Mixed Tensor/Type ops
        (path!["tensor", "shape"], Operation::Type(TypeOp::Shape)),
        (path!["tensor", "dtype"], Operation::Type(TypeOp::Dtype)),
        // Shape ops
        (path!["shape", "pack"], Operation::Type(TypeOp::Pack)),
        (path!["shape", "unpack"], Operation::Type(TypeOp::Unpack)),
        (path!["nat", "add"], Operation::Nat(NatOp::Add)),
        (path!["nat", "mul"], Operation::Nat(NatOp::Mul)),
    ])
}
