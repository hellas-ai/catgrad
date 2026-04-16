//! Abstract evaluation
use super::types::*;

use crate::category::core::*;
use crate::definition::Def;
use crate::ssa::parallel_ssa;

use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

/// Evaluate a [`Term`] using an abstract [`Interpreter`]
pub fn eval<I: Interpreter>(interpreter: &I, term: Term, values: Vec<Value<I>>) -> ResultValues<I> {
    eval_with(interpreter, term, values, |_, _| ())
}

/// Evaluate a [`Term`] using an abstract [`Interpreter`], and call the `on_write` callback after
/// every node has an intermediate value assigned.
pub fn eval_with<I: Interpreter, F: FnMut(NodeId, &Value<I>)>(
    interpreter: &I,
    term: Term,
    values: Vec<Value<I>>,
    mut on_write: F,
) -> ResultValues<I> {
    // TODO: replace with Err
    assert_eq!(values.len(), term.sources.len());

    // create initial state by moving argument values into state
    let mut state = HashMap::<NodeId, Value<I>>::new();
    for (node_id, value) in term.sources.iter().zip(values) {
        on_write(*node_id, &value);
        state.insert(*node_id, value);
    }

    // Save target nodes before moving term
    let target_nodes = term.targets.clone();

    // Iterate through partially-ordered SSA ops
    for par in parallel_ssa(term.to_strict())? {
        // PERFORMANCE: we can do these ops in parallel. Does it get speedups?
        for ssa in par {
            // get args: Vec<Value> by popping each id in op.sources from state - take
            // ownership.
            let mut args = Vec::new();
            for (node_id, _) in &ssa.sources {
                match state.remove(node_id) {
                    Some(value) => args.push(value),
                    None => return Err(InterpreterError::MultipleRead(*node_id)),
                }
            }

            // Dispatch: ops are either definitions or core ops.
            let results = match &ssa.op {
                Def::Def(path) => interpreter.handle_definition(&ssa, args, path),
                Def::Arr(op) => apply_op(interpreter, &ssa, args, op),
            }?;

            // write each result into state at op.targets ids
            for ((node_id, _), result) in ssa.targets.iter().zip(results) {
                on_write(*node_id, &result);
                if state.insert(*node_id, result).is_some() {
                    return Err(InterpreterError::MultipleWrite(*node_id));
                }
            }
        }
    }

    // Extract target values and return them
    let mut target_values = Vec::new();
    for target_node in &target_nodes {
        match state.remove(target_node) {
            Some(value) => target_values.push(value),
            None => return Err(InterpreterError::MultipleRead(*target_node)),
        }
    }

    Ok(target_values)
}

fn apply_op<I: Interpreter>(
    interpreter: &I,
    ssa: &CoreSSA,
    args: Vec<Value<I>>,
    op: &Operation,
) -> ResultValues<I> {
    match op {
        Operation::Type(type_op) => apply_type_op(ssa, args, type_op),
        Operation::Nat(nat_op) => apply_nat_op(ssa, args, nat_op),
        Operation::DtypeConstant(dtype) => Ok(vec![Value::Dtype(I::dtype_constant(*dtype))]),
        Operation::Tensor(tensor_op) => interpreter.tensor_op(ssa, args, tensor_op),
        Operation::Copy => apply_copy(ssa, args),
        Operation::If(then_branch, else_branch) => {
            interpreter.handle_if(ssa, args, then_branch, else_branch)
        }
        Operation::Load(path) => interpreter
            .handle_load(ssa, path)
            .ok_or(InterpreterError::Load(ssa.edge_id, path.clone())),
    }
}

////////////////////////////////////////////////////////////////////////////////
// Handlers for each possible op type.
// General convention is apply_<typename>

////////////////////////////////////////
// Copy

fn apply_copy<V: Interpreter>(ssa: &CoreSSA, args: Vec<Value<V>>) -> Result<Vec<Value<V>>> {
    let [v] = get_exact_arity(ssa, args)?;
    let n = ssa.targets.len();
    let mut result = Vec::with_capacity(n);
    result.push(v);
    for _ in 1..n {
        result.push(result[0].clone())
    }
    Ok(result)
}

use super::util::{get_exact_arity, to_nat, to_shape, to_tensor};
////////////////////////////////////////
// Type ops

fn apply_type_op<V: Interpreter>(
    ssa: &CoreSSA,
    args: Vec<Value<V>>,
    type_op: &TypeOp,
) -> ResultValues<V> {
    match type_op {
        // Pack dimensions into a shape
        TypeOp::Pack => {
            // Get all args (dims) and pack into result shape.
            let dims: Result<Vec<V::Nat>> = args.into_iter().map(|v| to_nat(ssa, v)).collect();
            Ok(vec![Value::Shape(V::pack(dims?))])
        }
        // Unpack a shape into dimensions
        TypeOp::Unpack => {
            // Get exactly 1 argument...
            let [arg] = get_exact_arity(ssa, args)?;
            // ... which is a shape ...
            let shape = to_shape(ssa, arg)?;
            // .. and unpack it into its constituent dimensions
            Ok(V::unpack(shape)
                .ok_or(InterpreterError::TypeError(ssa.edge_id))?
                .into_iter()
                .map(|dim| Value::Nat(dim))
                .collect())
        }
        // Map a tensor to its shape
        TypeOp::Shape => {
            // Get exactly 1 tensor argument
            let [arg] = get_exact_arity(ssa, args)?;
            let tensor = to_tensor(ssa, arg)?;
            Ok(vec![Value::Shape(
                V::shape(tensor).ok_or(InterpreterError::TypeError(ssa.edge_id))?,
            )])
        }
        // Map a tensor to its dtype
        TypeOp::Dtype => {
            let [arg] = get_exact_arity(ssa, args)?;
            let tensor = to_tensor(ssa, arg)?;
            Ok(vec![Value::Dtype(
                V::dtype(tensor).ok_or(InterpreterError::TypeError(ssa.edge_id))?,
            )])
        }
    }
}

////////////////////////////////////////
// Nat ops

fn apply_nat_op<I: Interpreter>(ssa: &CoreSSA, args: Vec<Value<I>>, op: &NatOp) -> ResultValues<I> {
    // Ensure all args are nats.
    let args: Result<Vec<I::Nat>> = args.into_iter().map(|n| to_nat(ssa, n)).collect();
    match op {
        NatOp::Constant(n) => {
            let [] = get_exact_arity(ssa, args?)?;
            Ok(vec![Value::Nat(I::nat_constant(*n))])
        }
        NatOp::Add => {
            let [a, b] = get_exact_arity(ssa, args?)?;
            Ok(vec![Value::Nat(I::nat_add(a, b))])
        }
        NatOp::Mul => {
            let [a, b] = get_exact_arity(ssa, args?)?;
            Ok(vec![Value::Nat(I::nat_mul(a, b))])
        }
    }
}
