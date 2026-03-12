use super::backend::Backend;
use super::tensor_op::tensor_op;
use super::types::*;

use crate::category::core::{Dtype, Shape, TensorOp};
use crate::category::{core, lang};
use crate::pass::to_core::Environment;
use crate::{
    abstract_interpreter,
    abstract_interpreter::{CoreSSA, InterpreterError, eval},
};

#[derive(Clone, std::fmt::Debug)]
pub struct Interpreter<B: Backend> {
    pub backend: B,
    pub environment: Environment,
    pub parameters: Parameters<B>,
}

impl<B: Backend> Interpreter<B> {
    pub fn new(backend: B, environment: Environment, parameters: Parameters<B>) -> Self {
        Interpreter {
            backend,
            environment,
            parameters,
        }
    }

    pub fn eval(&self, term: core::Term, args: Vec<Value<B>>) -> ResultValues<B> {
        eval(self, term, args)
    }

    pub fn run(&self, term: lang::Term, args: Vec<Value<B>>) -> ResultValues<B> {
        let term = self.environment.to_core(term);
        self.eval(term, args)
    }
}

#[rustfmt::skip]
impl<B: Backend> abstract_interpreter::Interpreter for Interpreter<B> {
    type Nat = usize;
    type Dtype = Dtype;
    type Shape = Shape;
    type NdArrayType = core::NdArrayType;
    type Tensor = TaggedTensor<B>;

    fn pack(dims: Vec<Self::Nat>) -> Self::Shape            { Shape(dims)          }
    fn unpack(shape: Self::Shape) -> Option<Vec<Self::Nat>> { Some(shape.0)        }
    fn shape(tensor: Self::Tensor) -> Option<Self::Shape>   { Some(tensor.shape()) }
    fn dtype(tensor: Self::Tensor) -> Option<Self::Dtype>   { Some(tensor.dtype()) }
    fn dtype_constant(d: core::Dtype) -> Self::Dtype        { d                    }
    fn nat_constant(nat: usize) -> Self::Nat                { nat                  }
    fn nat_add(a: Self::Nat, b: Self::Nat) -> Self::Nat     { a + b                }
    fn nat_mul(a: Self::Nat, b: Self::Nat) -> Self::Nat     { a * b                }

    fn handle_load(&self, _ssa: &CoreSSA, path: &crate::prelude::Path) -> Option<Vec<Value<B>>> {
        // TODO: remove clone?
        self.parameters.0.get(path).map(|t| vec![t.clone()])
    }

    fn handle_definition(
        &self,
        _ssa: &CoreSSA,
        args: Vec<abstract_interpreter::Value<Self>>,
        path: &crate::prelude::Path,
    ) -> abstract_interpreter::ResultValues<Self> {
        let source_values = args.to_vec();
        let lang::TypedTerm { term, .. } = self.environment.definitions.get(path).unwrap_or_else(|| {
            panic!("Definition not found for path: {:?}", path);
        });
        // TODO: can we remove this clone?
        let term = self.environment.to_core(term.clone());
        self.eval(term, source_values)
    }

    fn tensor_op(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value<B>>,
        op: &TensorOp,
    ) -> abstract_interpreter::ResultValues<Self> {
        tensor_op(&self.backend, ssa, args, op)
    }

    fn handle_if(
        &self,
        ssa: &CoreSSA,
        args: Vec<Value<B>>,
        then_branch: &core::Term,
        else_branch: &core::Term,
    ) -> abstract_interpreter::ResultValues<Self> {
        let (cond_val, branch_args) = args
            .split_first()
            .ok_or(InterpreterError::ArityError(ssa.edge_id))?;
        let cond_tensor = match cond_val {
            Value::Tensor(t) => t.clone(),
            _ => return Err(InterpreterError::TypeError(ssa.edge_id)),
        };

        let branch_args_vec: Vec<Value<B>> = branch_args.to_vec();

        if self.backend.to_bool(cond_tensor) {
            self.eval(then_branch.clone(), branch_args_vec)
        } else {
            self.eval(else_branch.clone(), branch_args_vec)
        }
    }
}
