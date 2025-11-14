use super::interpreter::{ResultValues, Value};
use super::value_types::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr};

use crate::abstract_interpreter::{
    CoreSSA, InterpreterError, Result,
    util::{ensure_profile, get_exact_arity, to_dtype, to_nat, to_shape, to_tensor},
};
use crate::category::core::{Dtype, Scalar, ScalarOp, TensorOp};

pub(crate) fn tensor_op(ssa: &CoreSSA, args: Vec<Value>, op: &TensorOp) -> ResultValues {
    match op {
        TensorOp::Map(scalar_op) => tensor_map(ssa, args, scalar_op),
        TensorOp::NatToU32 => tensor_nat_to_u32(ssa, args),
        TensorOp::Cast => tensor_cast(ssa, args),
        TensorOp::MatMul => tensor_matmul(ssa, args),
        TensorOp::Scalar(c) => tensor_constant(ssa, args, c.clone()),
        TensorOp::Sum | TensorOp::Max | TensorOp::Argmax => tensor_reduce(ssa, args),
        TensorOp::TopK => tensor_topk(ssa, args),
        TensorOp::Broadcast => tensor_broadcast(ssa, args),
        TensorOp::Reshape => tensor_reshape(ssa, args),
        TensorOp::Transpose => tensor_transpose(ssa, args),
        TensorOp::Slice => tensor_slice(ssa, args),
        TensorOp::Concat => tensor_concat(ssa, args),
        TensorOp::Arange => tensor_arange(ssa, args),
        TensorOp::Index => tensor_index(ssa, args),
    }
}

fn tensor_map(ssa: &CoreSSA, args: Vec<Value>, op: &ScalarOp) -> ResultValues {
    // FIXME: do Sin/Cos work on non-floating types? Are LT/EQ supposed to return U32 or F32?
    let (arity, coarity) = op.profile();
    let args = ensure_profile(ssa, args, arity, coarity)?;

    // clippy is wrong, it's always better to check <= 0 instead of = 0.
    #[allow(clippy::absurd_extreme_comparisons)]
    if arity <= 0 {
        panic!("Map cannot support ScalarOps of arity 0");
    }

    // check all args are tensors
    let types = args
        .into_iter()
        .map(|t| to_tensor(ssa, t))
        .collect::<Result<Vec<_>>>()?;

    // normal form for all types
    let types: Vec<_> = types.into_iter().map(|t| t.nf()).collect();
    if types.iter().all(|t| *t == types[0]) {
        Ok((0..coarity)
            .map(|_| Value::Tensor(types[0].clone()))
            .collect())
    } else {
        Err(InterpreterError::TypeError(ssa.edge_id))
    }
}

fn tensor_nat_to_u32(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [n] = get_exact_arity(ssa, args)?;
    let _ = to_nat(ssa, n)?; // ensure arg is a nat
    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![]),
    }))])
}

fn tensor_cast(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [tensor, dtype] = get_exact_arity(ssa, args)?;
    let (tensor, dtype) = (to_tensor(ssa, tensor)?, to_dtype(ssa, dtype)?);
    let type_expr = match tensor {
        // Shape of v, but dtype
        TypeExpr::Var(v) => TypeExpr::NdArrayType(NdArrayType {
            dtype,
            shape: ShapeExpr::OfType(v),
        }),

        // Replace dtype of existing NdArrayType
        TypeExpr::NdArrayType(s) => TypeExpr::NdArrayType(NdArrayType {
            dtype,
            shape: s.shape,
        }),
    };
    Ok(vec![Value::Tensor(type_expr)])
}

fn tensor_matmul(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [t, u] = get_exact_arity(ssa, args)?;
    let (t, u) = match (to_tensor(ssa, t)?.nf(), to_tensor(ssa, u)?.nf()) {
        (TypeExpr::NdArrayType(t), TypeExpr::NdArrayType(u)) if t.dtype == u.dtype => Ok((t, u)),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;
    let dtype = t.dtype.clone();

    if let (ShapeExpr::Shape(t), ShapeExpr::Shape(u)) = (t.shape, u.shape) {
        // Ensure equal ranks, at least 2 dims, and contraction dimension must match
        if t.len() != u.len() || t.len() < 2 || t[t.len() - 1] != u[u.len() - 2] {
            return Err(InterpreterError::TypeError(ssa.edge_id));
        }

        // check prefix dimensions match - e.g. N0..Nk for (N0, ..., Nk, A, B) and (N0..Nk, B, C)
        let prefix_len = t.len() - 2;
        if t[..prefix_len] != u[..prefix_len] {
            return Err(InterpreterError::TypeError(ssa.edge_id));
        }

        // Result shape: all but last element of t, then append final element of u
        let mut shape = t[..t.len() - 1].to_vec();
        shape.push(u[u.len() - 1].clone());

        Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype,
            shape: ShapeExpr::Shape(shape),
        }))])
    } else {
        Err(InterpreterError::TypeError(ssa.edge_id))
    }
}

fn tensor_constant(ssa: &CoreSSA, args: Vec<Value>, c: Scalar) -> ResultValues {
    let [] = get_exact_arity(ssa, args)?; // ensure 0 args
    let d = match c {
        Scalar::F32(_) => Dtype::F32,
        Scalar::U32(_) => Dtype::U32,
    };
    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(d),
        shape: ShapeExpr::Shape(vec![]),
    }))])
}

fn tensor_reduce(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [tensor] = get_exact_arity(ssa, args)?;
    let type_expr = match to_tensor(ssa, tensor)? {
        TypeExpr::NdArrayType(n) => match n.shape {
            ShapeExpr::Shape(mut shape) => {
                let k = shape.len();
                shape[k - 1] = NatExpr::Constant(1);
                TypeExpr::NdArrayType(NdArrayType {
                    dtype: n.dtype,
                    shape: ShapeExpr::Shape(shape),
                })
            }
            _ => return Err(InterpreterError::TypeError(ssa.edge_id)),
        },
        TypeExpr::Var(_) => return Err(InterpreterError::TypeError(ssa.edge_id)),
    };

    Ok(vec![Value::Tensor(type_expr)])
}

fn tensor_topk(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [tensor, k] = get_exact_arity(ssa, args)?;
    let (tensor, k) = (
        to_tensor(ssa, tensor)?.into_ndarraytype(ssa)?,
        to_nat(ssa, k)?,
    );

    match tensor.shape {
        ShapeExpr::Shape(mut shape) if !shape.is_empty() => {
            let last_idx = shape.len() - 1;
            shape[last_idx] = k.nf();
            let values = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: tensor.dtype.clone(),
                shape: ShapeExpr::Shape(shape.clone()),
            }));
            let indices = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::U32),
                shape: ShapeExpr::Shape(shape),
            }));
            Ok(vec![values, indices])
        }
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

// TODO: return normalized, broadcasted result (y) instead,
// and use it in tensor_broadcast?
fn is_broadcastable(x: &[NatExpr], y: &[NatExpr]) -> bool {
    // x must be a suffix of y
    let d = y.len() as isize - x.len() as isize;
    if d < 0 {
        return false;
    }
    let d = d as usize;

    // Compute normal forms on aligned dimensions, e.g.
    //      x =        (d₀  32+32)
    //      y = (1  9   d₀  64)
    //                  ^
    //                  |-- compares only last two dims
    let x = x.iter().map(|x| x.nf());
    let y = y[d..].iter().map(|x| x.nf());

    // check all normal forms pointwise equal, or x is 1
    for (x, y) in x.zip(y) {
        if x != y && x != NatExpr::Constant(1) {
            return false;
        }
    }
    true
}

fn tensor_broadcast(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [t, s] = get_exact_arity(ssa, args)?;
    let (t, s) = (to_tensor(ssa, t)?, to_shape(ssa, s)?);

    // Ensure t has a known shape
    let (t_shape, dtype) = match t {
        TypeExpr::NdArrayType(NdArrayType { shape, dtype }) => Ok((shape, dtype)),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    let shape = match (t_shape, &s) {
        // unit () is always broadcastable
        (ShapeExpr::Shape(ts), ShapeExpr::Var(_)) if ts.is_empty() => Ok(s),
        // otherwise check compatibility
        (ShapeExpr::Shape(ts), ShapeExpr::Shape(ss)) if is_broadcastable(&ts, ss) => Ok(s),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    let result_type = TypeExpr::NdArrayType(NdArrayType { shape, dtype });
    Ok(vec![Value::Tensor(result_type)])
}

fn tensor_transpose(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [t, dim0, dim1] = get_exact_arity(ssa, args)?;
    let (t, dim0, dim1) = (to_tensor(ssa, t)?, to_nat(ssa, dim0)?, to_nat(ssa, dim1)?);

    let (dim0, dim1) = match (dim0.nf(), dim1.nf()) {
        (NatExpr::Constant(dim0), NatExpr::Constant(dim1)) => Ok((dim0, dim1)),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    let input: NdArrayType = match t {
        TypeExpr::NdArrayType(input) => Ok(input),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    match input.shape {
        ShapeExpr::Shape(mut shape) => {
            shape.swap(dim0, dim1);
            Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: input.dtype,
                shape: ShapeExpr::Shape(shape),
            }))])
        }
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

fn tensor_concat(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [a, b, dim] = get_exact_arity(ssa, args)?;
    let (a, b, dim) = (
        to_tensor(ssa, a)?.into_ndarraytype(ssa)?,
        to_tensor(ssa, b)?.into_ndarraytype(ssa)?,
        to_nat(ssa, dim)?,
    );

    let dim = match dim.nf() {
        NatExpr::Constant(dim) => Ok(dim),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    match (a.shape, b.shape) {
        (ShapeExpr::Shape(shape_a), ShapeExpr::Shape(shape_b)) => {
            let mut shape = shape_a.clone();
            shape[dim] = NatExpr::Add(vec![shape_a[dim].clone(), shape_b[dim].clone()]);
            Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: a.dtype,
                shape: ShapeExpr::Shape(shape),
            }))])
        }
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

fn tensor_slice(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [input, dim, _start, len] = get_exact_arity(ssa, args)?;
    let (input, dim, len) = (
        to_tensor(ssa, input)?.into_ndarraytype(ssa)?,
        to_nat(ssa, dim)?,
        to_nat(ssa, len)?,
    );

    let dim = match dim.nf() {
        NatExpr::Constant(dim) => Ok(dim),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    match input.shape {
        ShapeExpr::Shape(mut shape) => {
            shape[dim] = len;
            Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: input.dtype,
                shape: ShapeExpr::Shape(shape),
            }))])
        }
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

fn tensor_arange(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [n] = get_exact_arity(ssa, args)?;
    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![to_nat(ssa, n)?]),
    }))])
}

fn tensor_index(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [input, n, idx] = get_exact_arity(ssa, args)?;
    let (input, n, idx) = (
        to_tensor(ssa, input)?.into_ndarraytype(ssa)?,
        to_nat(ssa, n)?,
        to_tensor(ssa, idx)?.into_ndarraytype(ssa)?,
    );

    let n = match n.nf() {
        NatExpr::Constant(n) => Ok(n),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }?;

    match (input.shape, idx.shape) {
        (ShapeExpr::Shape(mut input_shape), ShapeExpr::Shape(idx_shape)) => {
            input_shape[n] = idx_shape[0].clone();
            Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: input.dtype,
                shape: ShapeExpr::Shape(input_shape),
            }))])
        }
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

fn tensor_reshape(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [target_shape, tensor] = get_exact_arity(ssa, args)?;
    let (target_shape, (shape, dtype)) = (
        to_shape(ssa, target_shape)?,
        to_tensor(ssa, tensor)?.into_shapeexpr_dtype(ssa)?,
    );

    if !shapes_isomorphic(&shape, &target_shape) {
        // FIXME: better error type here
        return Err(InterpreterError::TypeError(ssa.edge_id));
    }

    let target_type = NdArrayType {
        shape: target_shape,
        dtype,
    };
    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(target_type))])
}

// Return normalized shapes for s, t
// TODO: return ApplyResult
fn shapes_isomorphic(s: &ShapeExpr, t: &ShapeExpr) -> bool {
    match (s, t) {
        (ShapeExpr::Var(v), ShapeExpr::Var(u)) => v == u,
        (ShapeExpr::OfType(v), ShapeExpr::OfType(u)) => v == u,
        (ShapeExpr::Shape(s), ShapeExpr::Shape(t)) => {
            super::isomorphism::isomorphic(s.clone(), t.clone())
        }
        _ => false,
    }
}
