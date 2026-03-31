use super::display::type_error;
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
        TensorOp::Sum | TensorOp::Max => tensor_reduce(ssa, args),
        TensorOp::Argmax => tensor_argmax(ssa, args),
        TensorOp::TopK => tensor_topk(ssa, args),
        TensorOp::Probe(_) => tensor_probe(ssa, args),
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
    let inputs = args.clone();
    let op_name = format!("map({op:?})");
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
        Err(type_error(
            ssa,
            &op_name,
            &inputs,
            "map requires all tensor inputs to have the same type",
        ))
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
    let inputs = args.clone();
    let [lhs, rhs] = get_exact_arity(ssa, args)?;
    let (lhs, rhs) = match (to_tensor(ssa, lhs)?.nf(), to_tensor(ssa, rhs)?.nf()) {
        (TypeExpr::NdArrayType(lhs), TypeExpr::NdArrayType(rhs)) => (lhs, rhs),
        _ => {
            return Err(type_error(
                ssa,
                "matmul",
                &inputs,
                "matmul requires tensors with concrete ndarray types",
            ));
        }
    };

    if lhs.dtype != rhs.dtype {
        return Err(type_error(
            ssa,
            "matmul",
            &inputs,
            format!(
                "matmul requires matching dtypes, got lhs {} and rhs {}",
                lhs.dtype, rhs.dtype
            ),
        ));
    }
    let dtype = lhs.dtype.clone();

    if let (ShapeExpr::Shape(lhs_shape), ShapeExpr::Shape(rhs_shape)) = (&lhs.shape, &rhs.shape) {
        // Ensure equal ranks, at least 2 dims, and contraction dimension must match
        if lhs_shape.len() != rhs_shape.len() {
            return Err(type_error(
                ssa,
                "matmul",
                &inputs,
                format!(
                    "matmul requires tensors with the same rank, got lhs {} and rhs {}",
                    lhs.shape, rhs.shape
                ),
            ));
        }
        if lhs_shape.len() < 2 {
            return Err(type_error(
                ssa,
                "matmul",
                &inputs,
                format!(
                    "matmul requires rank >= 2 tensors, got lhs {} and rhs {}",
                    lhs.shape, rhs.shape
                ),
            ));
        }
        if lhs_shape[lhs_shape.len() - 1] != rhs_shape[rhs_shape.len() - 2] {
            return Err(type_error(
                ssa,
                "matmul",
                &inputs,
                format!(
                    "matmul contraction dimensions do not match: lhs {} vs rhs {}",
                    lhs_shape[lhs_shape.len() - 1],
                    rhs_shape[rhs_shape.len() - 2]
                ),
            ));
        }

        // check prefix dimensions match - e.g. N0..Nk for (N0, ..., Nk, A, B) and (N0..Nk, B, C)
        let prefix_len = lhs_shape.len() - 2;
        if lhs_shape[..prefix_len] != rhs_shape[..prefix_len] {
            return Err(type_error(
                ssa,
                "matmul",
                &inputs,
                format!(
                    "matmul batch dimensions do not match: lhs {} vs rhs {}",
                    lhs.shape, rhs.shape
                ),
            ));
        }

        // Result shape: all but last element of lhs, then append final element of rhs
        let mut shape = lhs_shape[..lhs_shape.len() - 1].to_vec();
        shape.push(rhs_shape[rhs_shape.len() - 1].clone());

        Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype,
            shape: ShapeExpr::Shape(shape),
        }))])
    } else {
        Err(type_error(
            ssa,
            "matmul",
            &inputs,
            "matmul requires tensors with concrete shapes",
        ))
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

fn tensor_argmax(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [tensor] = get_exact_arity(ssa, args)?;
    let type_expr = match to_tensor(ssa, tensor)? {
        TypeExpr::NdArrayType(n) => match n.shape {
            ShapeExpr::Shape(mut shape) => {
                let k = shape.len();
                shape[k - 1] = NatExpr::Constant(1);
                TypeExpr::NdArrayType(NdArrayType {
                    dtype: Dtype::U32.into(),
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
    let inputs = args.clone();
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
        ShapeExpr::Shape(_) => Err(type_error(
            ssa,
            "topk",
            &inputs,
            "topk requires a tensor with rank >= 1",
        )),
        _ => Err(InterpreterError::TypeError(ssa.edge_id)),
    }
}

fn tensor_probe(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let [tensor] = get_exact_arity(ssa, args)?;
    let _ = to_tensor(ssa, tensor)?;
    Ok(vec![])
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
    let inputs = args.clone();
    let [t, s] = get_exact_arity(ssa, args)?;
    let (t, s) = (to_tensor(ssa, t)?, to_shape(ssa, s)?);

    // Ensure t has a known shape
    let (t_shape, dtype) = match t {
        TypeExpr::NdArrayType(NdArrayType { shape, dtype }) => (shape, dtype),
        _ => return Err(InterpreterError::TypeError(ssa.edge_id)),
    };

    let shape = match (t_shape, &s) {
        // unit () is always broadcastable
        (ShapeExpr::Shape(ts), ShapeExpr::Var(_)) if ts.is_empty() => Ok(s),
        // otherwise check compatibility
        (ShapeExpr::Shape(ts), ShapeExpr::Shape(ss)) if is_broadcastable(&ts, ss) => Ok(s),
        _ => Err(type_error(
            ssa,
            "broadcast",
            &inputs,
            "broadcast target shape is not compatible with the tensor shape",
        )),
    }?;

    let result_type = TypeExpr::NdArrayType(NdArrayType { shape, dtype });
    Ok(vec![Value::Tensor(result_type)])
}

fn tensor_transpose(ssa: &CoreSSA, args: Vec<Value>) -> ResultValues {
    let inputs = args.clone();
    let [t, dim0, dim1] = get_exact_arity(ssa, args)?;
    let (t, dim0, dim1) = (to_tensor(ssa, t)?, to_nat(ssa, dim0)?, to_nat(ssa, dim1)?);

    let (dim0, dim1) = match (dim0.nf(), dim1.nf()) {
        (NatExpr::Constant(dim0), NatExpr::Constant(dim1)) => (dim0, dim1),
        _ => {
            return Err(type_error(
                ssa,
                "transpose",
                &inputs,
                "transpose requires constant dimension indices",
            ));
        }
    };

    let input: NdArrayType = match t {
        TypeExpr::NdArrayType(input) => input,
        _ => return Err(InterpreterError::TypeError(ssa.edge_id)),
    };

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
    let inputs = args.clone();
    let [a, b, dim] = get_exact_arity(ssa, args)?;
    let (a, b, dim) = (
        to_tensor(ssa, a)?.into_ndarraytype(ssa)?,
        to_tensor(ssa, b)?.into_ndarraytype(ssa)?,
        to_nat(ssa, dim)?,
    );

    let dim = match dim.nf() {
        NatExpr::Constant(dim) => dim,
        _ => {
            return Err(type_error(
                ssa,
                "concat",
                &inputs,
                "concat requires a constant dimension index",
            ));
        }
    };

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
    let inputs = args.clone();
    let [input, dim, _start, len] = get_exact_arity(ssa, args)?;
    let (input, dim, len) = (
        to_tensor(ssa, input)?.into_ndarraytype(ssa)?,
        to_nat(ssa, dim)?,
        to_nat(ssa, len)?,
    );

    let dim = match dim.nf() {
        NatExpr::Constant(dim) => dim,
        _ => {
            return Err(type_error(
                ssa,
                "slice",
                &inputs,
                "slice requires a constant dimension index",
            ));
        }
    };

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
    let inputs = args.clone();
    let [input, n, idx] = get_exact_arity(ssa, args)?;
    let (input, n, idx) = (
        to_tensor(ssa, input)?.into_ndarraytype(ssa)?,
        to_nat(ssa, n)?,
        to_tensor(ssa, idx)?.into_ndarraytype(ssa)?,
    );

    let n = match n.nf() {
        NatExpr::Constant(n) => n,
        _ => {
            return Err(type_error(
                ssa,
                "index",
                &inputs,
                "index requires a constant dimension index",
            ));
        }
    };

    match (input.shape, idx.shape) {
        (ShapeExpr::Shape(mut input_shape), ShapeExpr::Shape(idx_shape)) => {
            if idx_shape.len() != 1 {
                return Err(type_error(
                    ssa,
                    "index",
                    &inputs,
                    format!(
                        "index expects a 1-D index tensor, got shape {}",
                        ShapeExpr::Shape(idx_shape)
                    ),
                ));
            }
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
    let inputs = args.clone();
    let [target_shape, tensor] = get_exact_arity(ssa, args)?;
    let (target_shape, (shape, dtype)) = (
        to_shape(ssa, target_shape)?,
        to_tensor(ssa, tensor)?.into_shapeexpr_dtype(ssa)?,
    );

    if !shapes_isomorphic(&shape, &target_shape) {
        return Err(type_error(
            ssa,
            "reshape",
            &inputs,
            format!("reshape requires isomorphic shapes, got {shape} and {target_shape}"),
        ));
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
