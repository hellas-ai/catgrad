use super::Parameters;
use super::display::{format_values, type_error};
use super::value_types::*;

use crate::category::{core, lang};
use crate::{
    abstract_interpreter,
    abstract_interpreter::{CoreSSA, InterpreterError, eval},
};

use crate::pass::to_core::Environment;

use super::tensor_op::tensor_op;

pub(crate) type Value = abstract_interpreter::Value<Interpreter>;
pub type ResultValues = abstract_interpreter::ResultValues<Interpreter>;
pub type Type = abstract_interpreter::Value<Interpreter>;

/// Compute the normal form for a [`Type`]
pub fn normalize(v: Type) -> Type {
    match v {
        Type::Nat(n) => Type::Nat(n.nf()),
        Type::Dtype(d) => Type::Dtype(d.nf()),
        Type::Shape(s) => Type::Shape(s.nf()),
        Type::Type(t) => Type::Type(t.nf()),
        Type::Tensor(t) => Type::Tensor(t.nf()),
    }
}

#[derive(Clone, std::fmt::Debug)]
pub struct Interpreter {
    pub(crate) environment: Environment,
    pub(crate) parameters: Parameters,
}

impl Interpreter {
    pub(crate) fn new(environment: Environment, parameters: Parameters) -> Self {
        Interpreter {
            environment,
            parameters,
        }
    }

    pub fn check_with(&self, term: core::Term, source_values: Vec<Value>) -> ResultValues {
        eval(self, term, source_values)
    }
}

impl abstract_interpreter::Interpreter for Interpreter {
    type Nat = NatExpr;
    type Dtype = DtypeExpr;
    type Shape = ShapeExpr;
    type NdArrayType = TypeExpr;
    type Tensor = TypeExpr;

    fn pack(dims: Vec<Self::Nat>) -> Self::Shape {
        ShapeExpr::Shape(dims)
    }

    fn unpack(shape: Self::Shape) -> Option<Vec<Self::Nat>> {
        match shape {
            ShapeExpr::Var(_) => None,
            ShapeExpr::OfType(_) => None,
            ShapeExpr::Shape(nat_exprs) => Some(nat_exprs),
        }
    }

    fn shape(tensor: Self::Tensor) -> Option<Self::Shape> {
        match tensor {
            TypeExpr::Var(_) => None,
            TypeExpr::NdArrayType(nd_array_type) => Some(nd_array_type.shape),
        }
    }

    fn dtype(tensor: Self::Tensor) -> Option<Self::Dtype> {
        match tensor {
            TypeExpr::Var(_) => None,
            TypeExpr::NdArrayType(nd_array_type) => Some(nd_array_type.dtype),
        }
    }

    fn dtype_constant(d: core::Dtype) -> Self::Dtype {
        DtypeExpr::Constant(d)
    }

    fn nat_constant(nat: usize) -> Self::Nat {
        NatExpr::Constant(nat)
    }

    fn nat_add(a: Self::Nat, b: Self::Nat) -> Self::Nat {
        NatExpr::Add(vec![a, b])
    }

    fn nat_mul(a: Self::Nat, b: Self::Nat) -> Self::Nat {
        NatExpr::Mul(vec![a, b])
    }

    fn handle_load(&self, _ssa: &CoreSSA, path: &crate::prelude::Path) -> Option<Vec<Value>> {
        self.parameters.0.get(path).map(|t| vec![t.clone()])
    }

    fn handle_definition(
        &self,
        _ssa: &CoreSSA,
        args: Vec<abstract_interpreter::Value<Self>>,
        path: &crate::prelude::Path,
    ) -> abstract_interpreter::ResultValues<Self> {
        let source_values = args.to_vec();
        let lang::TypedTerm { term, .. } = self
            .environment
            .definitions
            .get(path)
            .unwrap_or_else(|| panic!("definition {path} not found"));
        // TODO: can we remove this clone?
        let term = self.environment.to_core(term.clone());
        self.check_with(term, source_values)
    }

    fn tensor_op(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value>,
        op: &core::TensorOp,
    ) -> abstract_interpreter::ResultValues<Self> {
        tensor_op(ssa, args, op)
    }

    fn handle_if(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value>,
        then_branch: &core::Term,
        else_branch: &core::Term,
    ) -> abstract_interpreter::ResultValues<Self> {
        let inputs = args.clone();
        let (_cond, branch_args) = args
            .split_first()
            .ok_or(InterpreterError::ArityError(ssa.edge_id))?;

        // check both branches
        let branch_args_vec: Vec<Value> = branch_args.to_vec();
        let then_res = self.check_with(then_branch.clone(), branch_args_vec.clone())?;
        let else_res = self.check_with(else_branch.clone(), branch_args_vec)?;

        // normalize and compare
        let then_norm: Vec<_> = then_res.into_iter().map(normalize).collect();
        let else_norm: Vec<_> = else_res.into_iter().map(normalize).collect();

        if then_norm == else_norm {
            Ok(then_norm)
        } else {
            Err(type_error(
                ssa,
                "if",
                &inputs,
                format!(
                    "if branches return different types: then [{}], else [{}]",
                    format_values(&then_norm),
                    format_values(&else_norm)
                ),
            ))
        }
    }
}
