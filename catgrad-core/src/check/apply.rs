use super::types::*;
use crate::category::lang::Path;
use crate::category::{
    core::{Dtype, NatOp, Operation, ScalarOp, TensorOp, TypeOp},
    lang,
};
use crate::ssa::SSA;
use crate::stdlib::Environment;

// Get a value for each resulting NodeId.
pub fn s_apply(
    env: &Environment,
    params: &Parameters,
    op: &Operation,
    args: &[Value],
    ssa: &SSA<lang::Object, lang::Operation>,
) -> ApplyResult {
    match &op {
        Operation::Load(path) => apply_load(env, params, path, args, ssa),
        Operation::Type(op) => type_op(op, args),
        Operation::DtypeConstant(d) => Ok(vec![Value::Dtype(DtypeExpr::Constant(d.clone()))]),
        Operation::Nat(op) => nat_op(op, args),
        Operation::Tensor(op) => tensor_op(op, args),
        Operation::Copy => apply_copy(args, ssa),
    }
}

fn apply_load(
    _env: &Environment,
    params: &Parameters,
    path: &Path,
    args: &[Value],
    _ssa: &SSA<lang::Object, lang::Operation>,
) -> ApplyResult {
    if !args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let result = params
        .0
        .get(path)
        .ok_or(ApplyError::UnknownOp(path.clone()))?;
    Ok(vec![result.clone()])
}

fn apply_copy(args: &[Value], ssa: &SSA<lang::Object, lang::Operation>) -> ApplyResult {
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    Ok(vec![args[0].clone(); ssa.targets.len()])
}

////////////////////////////////////////
// TypeOp application + helpers

fn type_op(op: &TypeOp, args: &[Value]) -> ApplyResult {
    match op {
        TypeOp::Pack => type_pack(args),
        TypeOp::Unpack => type_unpack(args),
        TypeOp::Shape => type_shape(args),
        TypeOp::Dtype => type_dtype(args),
    }
}

fn type_shape(args: &[Value]) -> ApplyResult {
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    match &args[0] {
        Value::Tensor(TypeExpr::Var(v)) => Ok(vec![Value::Shape(ShapeExpr::OfType(*v))]),
        Value::Tensor(TypeExpr::NdArrayType(s)) => Ok(vec![Value::Shape(s.shape.clone())]),
        _ => Err(ApplyError::TypeError),
    }
}

fn type_dtype(args: &[Value]) -> ApplyResult {
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    match &args[0] {
        Value::Tensor(TypeExpr::Var(v)) => Ok(vec![Value::Dtype(DtypeExpr::OfType(*v))]),
        Value::Tensor(TypeExpr::NdArrayType(s)) => Ok(vec![Value::Dtype(s.dtype.clone())]),
        _ => Err(ApplyError::TypeError),
    }
}

fn type_pack(args: &[Value]) -> ApplyResult {
    // type_pack should have n nat args
    // Creates an NdArrayType from a dtype and individual nat dimensions
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let mut shape = Vec::new();
    for arg in args {
        match arg {
            Value::Nat(n) => shape.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    Ok(vec![Value::Shape(ShapeExpr::Shape(shape))])
}

fn type_unpack(args: &[Value]) -> ApplyResult {
    // type_unpack should have exactly 1 NdArrayType arg
    // Returns dtype + individual nat dimensions
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    }

    match &args[0] {
        Value::Shape(s) => {
            match &s {
                ShapeExpr::Var(_) => Err(ApplyError::TypeError),
                ShapeExpr::OfType(_) => Err(ApplyError::TypeError),
                ShapeExpr::Shape(nat_exprs) => {
                    let mut result = Vec::new();

                    // Return each dimension as individual nats
                    for dim in nat_exprs {
                        result.push(Value::Nat(dim.clone()));
                    }

                    Ok(result)
                }
            }
        }
        _ => {
            println!("here?");
            Err(ApplyError::TypeError)
        }
    }
}

////////////////////////////////////////
// Nat op application + helpers

fn nat_op(op: &NatOp, args: &[Value]) -> ApplyResult {
    match op {
        NatOp::Constant(n) => {
            if !args.is_empty() {
                return Err(ApplyError::ArityError);
            }
            Ok(vec![Value::Nat(NatExpr::Constant(*n))])
        }
        NatOp::Mul => nat_mul(args),
        NatOp::Add => nat_add(args),
    }
}

fn nat_mul(args: &[Value]) -> ApplyResult {
    // Multiply n natural numbers together
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let mut nat_exprs = Vec::new();
    for arg in args {
        match arg {
            Value::Nat(n) => nat_exprs.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    // If there's only one argument, return it directly
    if nat_exprs.len() == 1 {
        Ok(vec![Value::Nat(nat_exprs.into_iter().next().unwrap())])
    } else {
        Ok(vec![Value::Nat(NatExpr::Mul(nat_exprs))])
    }
}

fn nat_add(args: &[Value]) -> ApplyResult {
    // Multiply n natural numbers together
    if args.is_empty() {
        return Err(ApplyError::ArityError);
    }

    let mut nat_exprs = Vec::new();
    for arg in args {
        match arg {
            Value::Nat(n) => nat_exprs.push(n.clone()),
            _ => return Err(ApplyError::TypeError),
        }
    }

    // If there's only one argument, return it directly
    if nat_exprs.len() == 1 {
        Ok(vec![Value::Nat(nat_exprs.into_iter().next().unwrap())])
    } else {
        Ok(vec![Value::Nat(NatExpr::Add(nat_exprs))])
    }
}

////////////////////////////////////////
// Tensor op application & helpers

fn tensor_op(op: &TensorOp, args: &[Value]) -> ApplyResult {
    match op {
        TensorOp::Reshape => tensor_reshape(args),
        TensorOp::MatMul => tensor_matmul(args),
        TensorOp::Map(scalar_op) => tensor_map(scalar_op, args),
        TensorOp::Cast => tensor_cast(args),
        TensorOp::Sum => tensor_sum(args),
        TensorOp::Max => tensor_max(args),
        TensorOp::Broadcast => tensor_broadcast(args),
        TensorOp::Index => tensor_index(args),
        TensorOp::Arange => tensor_arange(args),
        TensorOp::Scalar => tensor_scalar(args),
        TensorOp::Transpose => tensor_transpose(args),
        op => todo!("operation {op:?}"),
    }
}

fn tensor_map(scalar_op: &ScalarOp, args: &[Value]) -> ApplyResult {
    let (arity, coarity) = scalar_op.profile();

    // Check argument count matches arity
    if args.len() != arity {
        return Err(ApplyError::ArityError);
    }

    // Check all arguments are tensors with the same type
    let mut tensor_type: Option<&TypeExpr> = None;
    for arg in args {
        match arg {
            Value::Tensor(type_expr) => {
                if let Some(existing_type) = tensor_type {
                    if existing_type != type_expr {
                        return Err(ApplyError::TypeError);
                    }
                } else {
                    tensor_type = Some(type_expr);
                }
            }
            _ => return Err(ApplyError::TypeError),
        }
    }

    let tensor_type = tensor_type.unwrap(); // Safe because we checked arity > 0

    // Return coarity copies of the same tensor type
    Ok((0..coarity)
        .map(|_| Value::Tensor(tensor_type.clone()))
        .collect())
}

fn tensor_cast(args: &[Value]) -> ApplyResult {
    if args.len() != 2 {
        return Err(ApplyError::ArityError);
    };

    let [Value::Tensor(tensor), Value::Dtype(dtype)] = args else {
        return Err(ApplyError::TypeError);
    };

    let type_expr = match tensor {
        // Shape of v, but dtype
        TypeExpr::Var(v) => TypeExpr::NdArrayType(NdArrayType {
            dtype: dtype.clone(),
            shape: ShapeExpr::OfType(*v),
        }),

        // Replace dtype of existing NdArrayType
        TypeExpr::NdArrayType(s) => TypeExpr::NdArrayType(NdArrayType {
            dtype: dtype.clone(),
            shape: s.shape.clone(),
        }),
    };

    Ok(vec![Value::Tensor(type_expr)])
}

fn tensor_reduce(args: &[Value]) -> ApplyResult {
    if args.len() != 1 {
        return Err(ApplyError::ArityError);
    };

    let [Value::Tensor(tensor)] = args else {
        return Err(ApplyError::TypeError);
    };

    let type_expr = match tensor {
        TypeExpr::Var(_) => return Err(ApplyError::TypeError),
        TypeExpr::NdArrayType(n) => match &n.shape {
            ShapeExpr::Shape(input_shape) => {
                let out_shape = input_shape[..input_shape.len() - 1].to_vec();
                TypeExpr::NdArrayType(NdArrayType {
                    dtype: n.dtype.clone(),
                    shape: ShapeExpr::Shape(out_shape),
                })
            }
            _ => return Err(ApplyError::TypeError),
        },
    };

    Ok(vec![Value::Tensor(type_expr)])
}

fn tensor_sum(args: &[Value]) -> ApplyResult {
    tensor_reduce(args)
}

fn tensor_max(args: &[Value]) -> ApplyResult {
    tensor_reduce(args)
}

fn tensor_broadcast(args: &[Value]) -> ApplyResult {
    match (&args[0], &args[1]) {
        (Value::Tensor(TypeExpr::NdArrayType(t)), Value::Shape(shape)) => {
            match (&t.shape, &shape) {
                (ShapeExpr::Shape(_), ShapeExpr::Shape(s2)) => {
                    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                        dtype: t.dtype.clone(),
                        shape: ShapeExpr::Shape(s2.clone()),
                    }))])
                }
                (ShapeExpr::Shape(s), ShapeExpr::Var(v)) if s.is_empty() => {
                    Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                        dtype: t.dtype.clone(),
                        shape: ShapeExpr::Var(*v),
                    }))])
                }
                _ => Err(ApplyError::TypeError),
            }
        }
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_transpose(args: &[Value]) -> ApplyResult {
    if args.len() != 3 {
        return Err(ApplyError::ArityError);
    }
    match (&args[0], &args[1], &args[2]) {
        (
            Value::Tensor(TypeExpr::NdArrayType(input)),
            Value::Nat(NatExpr::Constant(dim0)),
            Value::Nat(NatExpr::Constant(dim1)),
        ) => match &input.shape {
            ShapeExpr::Shape(input_shape) => {
                let mut out_shape = input_shape.clone();
                out_shape[*dim0] = input_shape[*dim1].clone();
                out_shape[*dim1] = input_shape[*dim0].clone();
                Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                    dtype: input.dtype.clone(),
                    shape: ShapeExpr::Shape(out_shape),
                }))])
            }
            _ => Err(ApplyError::TypeError),
        },
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_index(args: &[Value]) -> ApplyResult {
    if args.len() != 3 {
        return Err(ApplyError::ArityError);
    }
    match (&args[0], &args[1], &args[2]) {
        (
            Value::Tensor(TypeExpr::NdArrayType(input)),
            Value::Nat(NatExpr::Constant(n)),
            Value::Tensor(TypeExpr::NdArrayType(idx)),
        ) => match (&input.shape, &idx.shape) {
            (ShapeExpr::Shape(input_shape), ShapeExpr::Shape(idx_shape)) => {
                let mut out_shape = input_shape.clone();
                out_shape[*n] = idx_shape[0].clone();
                Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                    dtype: input.dtype.clone(),
                    shape: ShapeExpr::Shape(out_shape),
                }))])
            }
            _ => Err(ApplyError::TypeError),
        },
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_arange(args: &[Value]) -> ApplyResult {
    match &args[0] {
        Value::Nat(n) => Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![n.clone()]),
        }))]),
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_scalar(args: &[Value]) -> ApplyResult {
    match &args[0] {
        Value::Nat(_) => Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![]),
        }))]),
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_matmul(args: &[Value]) -> ApplyResult {
    // reshape takes 2 args: target type and input tensor
    if args.len() != 2 {
        return Err(ApplyError::ArityError);
    }

    match (&args[0], &args[1]) {
        (Value::Tensor(TypeExpr::NdArrayType(t)), Value::Tensor(TypeExpr::NdArrayType(u))) => {
            if t.dtype != u.dtype {
                return Err(ApplyError::TypeError);
            }

            let dtype = t.dtype.clone();

            if let (ShapeExpr::Shape(t), ShapeExpr::Shape(u)) = (t.shape.clone(), u.shape.clone()) {
                // Same rank
                if t.len() != u.len() {
                    return Err(ApplyError::TypeError);
                }

                // need at least 2 dims in each to matmul
                if t.len() < 2 {
                    return Err(ApplyError::TypeError);
                }

                // Check contraction dimension matches
                if t[t.len() - 1] != u[u.len() - 2] {
                    return Err(ApplyError::TypeError);
                }

                // check prefix dimensions match - e.g. for (N0, ..., Nk, A, B) and (N0..Nk, B, C), N0..Nk
                // should match.
                let prefix_len = t.len() - 2;
                if t[..prefix_len] != u[..prefix_len] {
                    return Err(ApplyError::TypeError);
                }

                // Result shape: all but last element of t, then append final element of u
                let mut shape = t[..t.len() - 1].to_vec();
                shape.push(u[u.len() - 1].clone());

                Ok(vec![Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                    dtype,
                    shape: ShapeExpr::Shape(shape),
                }))])
            } else {
                Err(ApplyError::TypeError)
            }
        }
        _ => Err(ApplyError::TypeError),
    }
}

fn tensor_reshape(args: &[Value]) -> ApplyResult {
    // reshape takes 2 args: target type and input tensor
    if args.len() != 2 {
        return Err(ApplyError::ArityError);
    }

    match (&args[0], &args[1]) {
        (
            Value::Shape(target_shape),
            Value::Tensor(TypeExpr::NdArrayType(NdArrayType { dtype, shape })),
        ) => {
            if !shapes_isomorphic(shape, target_shape) {
                return Err(ApplyError::ShapeMismatch(
                    shape.clone(),
                    target_shape.clone(),
                ));
            }

            let target_type = NdArrayType {
                dtype: dtype.clone(),
                shape: target_shape.clone(),
            };

            Ok(vec![Value::Tensor(TypeExpr::NdArrayType(target_type))])
        }
        _ => Err(ApplyError::TypeError),
    }
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
