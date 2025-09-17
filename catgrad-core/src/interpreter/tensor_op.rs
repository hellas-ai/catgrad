//! Tensor operation implementations for the interpreter

use super::backend::*;
use super::{ApplyError, ApplyErrorKind, TaggedNdArray, TaggedNdArrayTuple, Value};
use crate::category::core::{Dtype, ScalarOp, TensorOp};
use crate::category::lang::{Object, Operation};
use crate::ssa::SSA;

/// Apply a Tensor operation
pub(crate) fn apply_tensor_op<B: Backend>(
    backend: &B,
    tensor_op: &TensorOp,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    match tensor_op {
        TensorOp::MatMul => binop(backend, args, ssa, B::matmul),
        TensorOp::Constant(_constant) => todo!("constant"),
        TensorOp::Sum => tensor_sum(backend, args, ssa),
        TensorOp::Max => tensor_max(backend, args, ssa),
        TensorOp::Arange => tensor_arange(backend, args, ssa),
        TensorOp::Argmax => todo!("argmax"),
        TensorOp::Broadcast => tensor_broadcast(backend, args, ssa),
        TensorOp::Reshape => tensor_reshape(backend, args, ssa),
        TensorOp::Map(ScalarOp::Add) => binop(backend, args, ssa, B::add),
        TensorOp::Map(ScalarOp::Sub) => binop(backend, args, ssa, B::sub),
        TensorOp::Map(ScalarOp::Pow) => binop(backend, args, ssa, B::pow),
        TensorOp::Map(ScalarOp::Neg) => unary_op(backend, args, ssa, B::neg),
        TensorOp::Map(ScalarOp::Mul) => binop(backend, args, ssa, B::mul),
        TensorOp::Map(ScalarOp::Div) => binop(backend, args, ssa, B::div),
        TensorOp::Map(scalar_op) => todo!("unimplemented scalar op {:?}", scalar_op),
        TensorOp::Cast => tensor_cast(backend, args, ssa),
        TensorOp::Scalar => tensor_scalar(backend, args, ssa),
        TensorOp::Concat => tensor_concat(backend, args, ssa),
        TensorOp::Index => tensor_index(backend, args, ssa),
        TensorOp::Slice => tensor_slice(backend, args, ssa),
        TensorOp::Copy => todo!("copy"),
    }
}

fn err<K: Into<ApplyErrorKind>>(kind: K, ssa: &SSA<Object, Operation>) -> Box<ApplyError> {
    Box::new(ApplyError {
        kind: kind.into(),
        ssa: ssa.clone(),
    })
}

pub(crate) fn tensor_scalar<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(err(ApplyErrorKind::ArityError, ssa));
    }

    let value: u32 = match args[0] {
        Value::Nat(n) => n
            .try_into()
            .map_err(|_| err(ApplyErrorKind::NatOverflow, ssa))?,
        _ => return Err(err(ApplyErrorKind::TypeError, ssa)),
    };

    let tensor = backend
        .ndarray_from_slice(&[value], super::Shape(vec![]))
        .map_err(|e| err(e, ssa))?;

    let result = TaggedNdArrayTuple::U32([tensor]);
    Ok(vec![Value::NdArray(result)])
}

fn tensor_cast<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];
    let target_dtype = &args[1];

    let Value::Dtype(target_dtype) = target_dtype else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.cast(x.clone(), target_dtype.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_sum<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.sum(x.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_max<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let tensor = &args[0];

    let Value::NdArray(x) = tensor else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.max(x.clone());
    Ok(vec![Value::NdArray(result)])
}

fn tensor_reshape<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [new_shape, tensor] - reshape(builder, new_shape, tensor)
    if let (Value::Shape(new_shape), Value::NdArray(x)) = (args.remove(0), args.remove(0)) {
        let result = backend.reshape(x, new_shape);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_broadcast<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 2 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    if let (Value::Shape(shape_prefix), Value::NdArray(x)) = (args.remove(1), args.remove(0)) {
        let result = backend.broadcast(x, shape_prefix);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_arange<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let Value::Nat(end) = args[0] else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = backend.arange(end);
    Ok(vec![Value::NdArray(result)])
}

fn tensor_index<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 3 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [input, indices]
    if let (Value::NdArray(input), Value::Nat(dim), Value::NdArray(indices)) =
        (args.remove(0), args.remove(0), args.remove(0))
    {
        let result = backend.index(input, dim, indices);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_concat<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 3 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [tensor, tensor, dim]
    if let (Value::NdArray(a), Value::NdArray(b), Value::Nat(dim)) =
        (args.remove(0), args.remove(0), args.remove(0))
    {
        let result = backend.concat(a, b, dim);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

fn tensor_slice<B: Backend>(
    backend: &B,
    mut args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 4 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    // Args are: [input, dim, start, end]
    if let (Value::NdArray(input), Value::Nat(dim), Value::Nat(start), Value::Nat(len)) = (
        args.remove(0),
        args.remove(0),
        args.remove(0),
        args.remove(0),
    ) {
        let result = backend.slice(input, dim, start, len);
        Ok(vec![Value::NdArray(result)])
    } else {
        Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }))
    }
}

#[allow(type_alias_bounds)]
type Binop<B: Backend> = fn(&B, TaggedNdArrayTuple<B, 2>) -> TaggedNdArrayTuple<B, 1>;

#[allow(type_alias_bounds)]
type Unaryop<B: Backend> = fn(&B, TaggedNdArray<B>) -> TaggedNdArray<B>;

fn binop<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    callback: Binop<B>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    let args = try_into_tagged_ndarrays::<B, 2>(args, ssa)?;
    let result = callback(backend, args);
    Ok(vec![Value::NdArray(result)])
}

fn unary_op<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    callback: Unaryop<B>,
) -> Result<Vec<Value<B>>, Box<ApplyError>> {
    if args.len() != 1 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let Value::NdArray(x) = &args[0] else {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        }));
    };

    let result = callback(backend, x.clone());
    Ok(vec![Value::NdArray(result)])
}

// TODO: binop has some boilerplate- have to unpack/repack a lot. can we fix that using run_op &
// into() on taggedndarraytuple?
#[allow(dead_code)]
/// Run an M → 1 op taking M NdArray values of the same dtype, producing an NdArray.
fn run_op<B: Backend, F, const M: usize>(
    args: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
    f: F,
) -> Result<Vec<Value<B>>, Box<ApplyError>>
where
    F: Fn(TaggedNdArrayTuple<B, M>) -> TaggedNdArrayTuple<B, 1>,
{
    Ok(vec![Value::NdArray(f(try_into_tagged_ndarrays::<B, M>(
        args, ssa,
    )?))])
}

/// Convert a Vec<Value<B>> into TaggedNdArrays<B, N> with compile-time length checking
pub(crate) fn try_into_tagged_ndarrays<B: Backend, const N: usize>(
    values: Vec<Value<B>>,
    ssa: &SSA<Object, Operation>,
) -> Result<TaggedNdArrayTuple<B, N>, Box<ApplyError>> {
    let n = values.len();

    // clippy is WRONG! what if someone changes the type of n, then what, huh!?
    // i will die on this hill
    #[allow(clippy::absurd_extreme_comparisons)]
    if n != N || n <= 0 {
        return Err(Box::new(ApplyError {
            kind: ApplyErrorKind::ArityError,
            ssa: ssa.clone(),
        }));
    }

    let mut f32_arrays = Vec::new();
    let mut u32_arrays = Vec::new();
    let mut dtype_kind: Option<Dtype> = None;

    for value in values {
        match value {
            Value::NdArray(TaggedNdArray::F32(arr)) => {
                if let Some(ref existing) = dtype_kind {
                    if *existing != Dtype::F32 {
                        return Err(Box::new(ApplyError {
                            kind: ApplyErrorKind::TypeError,
                            ssa: ssa.clone(),
                        }));
                    }
                } else {
                    dtype_kind = Some(Dtype::F32);
                }
                f32_arrays.push(arr[0].clone());
            }
            Value::NdArray(TaggedNdArray::U32(arr)) => {
                if let Some(ref existing) = dtype_kind {
                    if *existing != Dtype::U32 {
                        return Err(Box::new(ApplyError {
                            kind: ApplyErrorKind::TypeError,
                            ssa: ssa.clone(),
                        }));
                    }
                } else {
                    dtype_kind = Some(Dtype::U32);
                }
                u32_arrays.push(arr[0].clone());
            }
            _ => {
                return Err(Box::new(ApplyError {
                    kind: ApplyErrorKind::TypeError,
                    ssa: ssa.clone(),
                }));
            }
        }
    }

    let result = match dtype_kind {
        Some(Dtype::F32) => f32_arrays.try_into().ok().map(TaggedNdArrayTuple::F32),
        Some(Dtype::U32) => u32_arrays.try_into().ok().map(TaggedNdArrayTuple::U32),
        _ => None,
    };

    result.ok_or_else(|| {
        Box::new(ApplyError {
            kind: ApplyErrorKind::TypeError,
            ssa: ssa.clone(),
        })
    })
}
