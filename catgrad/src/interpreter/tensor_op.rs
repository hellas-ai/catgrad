//! Tensor operation implementations for the interpreter
use super::backend::*;
use super::{ResultValues, TaggedTensor, TaggedTensorTuple, Value};
use crate::abstract_interpreter::util::{get_exact_arity, to_dtype, to_nat, to_shape, to_tensor};
use crate::abstract_interpreter::{CoreSSA, InterpreterError, Result};
use crate::category::core::{Dtype, Scalar, ScalarOp, TensorOp};

/// Apply a Tensor operation
pub(crate) fn tensor_op<B: Backend>(
    backend: &B,
    ssa: &CoreSSA,
    args: Vec<Value<B>>,
    tensor_op: &TensorOp,
) -> ResultValues<B> {
    match tensor_op {
        TensorOp::Map(ScalarOp::Add) => binop(backend, args, ssa, B::add),
        TensorOp::Map(ScalarOp::Sub) => binop(backend, args, ssa, B::sub),
        TensorOp::Map(ScalarOp::Pow) => binop(backend, args, ssa, B::pow),
        TensorOp::Map(ScalarOp::Sin) => unary_op(backend, args, ssa, B::sin),
        TensorOp::Map(ScalarOp::Cos) => unary_op(backend, args, ssa, B::cos),
        TensorOp::Map(ScalarOp::Neg) => unary_op(backend, args, ssa, B::neg),
        TensorOp::Map(ScalarOp::Mul) => binop(backend, args, ssa, B::mul),
        TensorOp::Map(ScalarOp::Div) => binop(backend, args, ssa, B::div),
        TensorOp::Map(ScalarOp::LT) => binop(backend, args, ssa, B::lt),
        TensorOp::Map(ScalarOp::EQ) => binop(backend, args, ssa, B::eq),
        TensorOp::NatToU32 => tensor_nat_to_u32(backend, args, ssa),
        TensorOp::Cast => tensor_cast(backend, args, ssa),
        TensorOp::MatMul => binop(backend, args, ssa, B::matmul),
        TensorOp::Scalar(c) => tensor_constant(backend, args, ssa, c),
        TensorOp::Sum => tensor_sum(backend, args, ssa),
        TensorOp::Max => tensor_max(backend, args, ssa),
        TensorOp::Argmax => tensor_argmax(backend, args, ssa),
        TensorOp::TopK => tensor_topk(backend, args, ssa),
        TensorOp::Broadcast => tensor_broadcast(backend, args, ssa),
        TensorOp::Reshape => tensor_reshape(backend, args, ssa),
        TensorOp::Transpose => tensor_transpose(backend, args, ssa),
        TensorOp::Slice => tensor_slice(backend, args, ssa),
        TensorOp::Concat => tensor_concat(backend, args, ssa),
        TensorOp::Arange => tensor_arange(backend, args, ssa),
        TensorOp::Index => tensor_index(backend, args, ssa),
    }
}

fn tensor<B: Backend, T: super::IntoTagged<B, 1>>(
    backend: &B,
    ssa: &CoreSSA,
    shape: super::Shape,
    data: &[T],
) -> ResultValues<B> {
    let value = TaggedTensor::from_slice(backend, data, shape)
        .map_err(|_| InterpreterError::ApplyError(ssa.edge_id))?;
    Ok(vec![Value::Tensor(value)])
}

pub fn tensor_constant<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>, // must be empty
    ssa: &CoreSSA,
    c: &Scalar,
) -> ResultValues<B> {
    let [] = get_exact_arity(ssa, args)?; // get 0 args
    match c {
        Scalar::F32(x) => tensor(backend, ssa, super::Shape(vec![]), &[*x]),
        Scalar::U32(x) => tensor(backend, ssa, super::Shape(vec![]), &[*x]),
    }
}

fn tensor_nat_to_u32<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> ResultValues<B> {
    let [value] = get_exact_arity(ssa, args)?;
    let value: u32 = to_nat(ssa, value)?
        .try_into()
        .map_err(|_| InterpreterError::ApplyError(ssa.edge_id))?;
    tensor(backend, ssa, super::Shape(vec![]), &[value])
}

fn tensor_cast<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [tensor, target_dtype] = get_exact_arity(ssa, args)?;
    let (x, target_dtype) = (to_tensor(ssa, tensor)?, to_dtype(ssa, target_dtype)?);
    Ok(vec![Value::Tensor(backend.cast(x, target_dtype))])
}

fn tensor_sum<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [x] = get_exact_arity(ssa, args)?;
    let x = to_tensor(ssa, x)?;
    Ok(vec![Value::Tensor(backend.sum(x))])
}

fn tensor_max<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [x] = get_exact_arity(ssa, args)?;
    let x = to_tensor(ssa, x)?;
    Ok(vec![Value::Tensor(backend.max(x))])
}

fn tensor_argmax<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [x] = get_exact_arity(ssa, args)?;
    let x = to_tensor(ssa, x)?;
    Ok(vec![Value::Tensor(backend.argmax(x))])
}

fn tensor_topk<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [x, k] = get_exact_arity(ssa, args)?;
    let (x, k) = (to_tensor(ssa, x)?, to_nat(ssa, k)?);
    let (values, indices) = backend.topk(x, k);
    Ok(vec![Value::Tensor(values), Value::Tensor(indices)])
}

fn tensor_broadcast<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> ResultValues<B> {
    let [x, s] = get_exact_arity(ssa, args)?;
    let (x, shape_prefix) = (to_tensor(ssa, x)?, to_shape(ssa, s)?);
    Ok(vec![Value::Tensor(backend.broadcast(x, shape_prefix))])
}

fn tensor_reshape<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [s, x] = get_exact_arity(ssa, args)?;
    let (shape, x) = (to_shape(ssa, s)?, to_tensor(ssa, x)?);
    Ok(vec![Value::Tensor(backend.reshape(x, shape))])
}

fn tensor_transpose<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
) -> ResultValues<B> {
    let [x, dim0, dim1] = get_exact_arity(ssa, args)?;
    let (x, dim0, dim1) = (to_tensor(ssa, x)?, to_nat(ssa, dim0)?, to_nat(ssa, dim1)?);
    Ok(vec![Value::Tensor(backend.transpose(x, dim0, dim1))])
}

fn tensor_slice<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [input, d, s, l] = get_exact_arity(ssa, args)?;
    let input = to_tensor(ssa, input)?;
    let (dim, start, len) = (to_nat(ssa, d)?, to_nat(ssa, s)?, to_nat(ssa, l)?);
    Ok(vec![Value::Tensor(backend.slice(input, dim, start, len))])
}

fn tensor_concat<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [a, b, dim] = get_exact_arity(ssa, args)?;
    let (a, b, dim) = (to_tensor(ssa, a)?, to_tensor(ssa, b)?, to_nat(ssa, dim)?);
    Ok(vec![Value::Tensor(backend.concat(a, b, dim))])
}

fn tensor_arange<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [end] = get_exact_arity(ssa, args)?;
    Ok(vec![Value::Tensor(backend.arange(to_nat(ssa, end)?))])
}

fn tensor_index<B: Backend>(backend: &B, args: Vec<Value<B>>, ssa: &CoreSSA) -> ResultValues<B> {
    let [x, d, ix] = get_exact_arity(ssa, args)?;
    let (input, dim, indices) = (to_tensor(ssa, x)?, to_nat(ssa, d)?, to_tensor(ssa, ix)?);
    Ok(vec![Value::Tensor(backend.index(input, dim, indices))])
}

#[allow(type_alias_bounds)]
type Binop<B: Backend> = fn(&B, TaggedTensorTuple<B, 2>) -> TaggedTensorTuple<B, 1>;

#[allow(type_alias_bounds)]
type Unaryop<B: Backend> = fn(&B, TaggedTensor<B>) -> TaggedTensor<B>;

fn binop<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
    callback: Binop<B>,
) -> ResultValues<B> {
    let args = try_into_tagged_ndarrays::<B, 2>(args, ssa)?;
    let result = callback(backend, args);
    Ok(vec![Value::Tensor(result)])
}

fn unary_op<B: Backend>(
    backend: &B,
    args: Vec<Value<B>>,
    ssa: &CoreSSA,
    callback: Unaryop<B>,
) -> ResultValues<B> {
    let [x] = get_exact_arity(ssa, args)?;
    let result = callback(backend, to_tensor(ssa, x)?);
    Ok(vec![Value::Tensor(result)])
}

/// Convert a Vec<Value<B>> into TaggedNdArrays<B, N> with compile-time length checking
pub(crate) fn try_into_tagged_ndarrays<B: Backend, const N: usize>(
    values: Vec<Value<B>>, // TODO: rename args
    ssa: &CoreSSA,
) -> Result<TaggedTensorTuple<B, N>> {
    // If no args, type is ambiguous, but this is a programmer error.
    if N == 0 {
        panic!("try_into_tagged_ndarrays is undefined for N <= 0");
    }

    // Get exactly N tensors
    let tensors: Vec<TaggedTensor<B>> = get_exact_arity::<N, _>(ssa, values)?
        .into_iter()
        .map(|x| to_tensor(ssa, x))
        .collect::<Result<_>>()?;
    let dtype = tensors[0].dtype();

    // Collect each tag into its own typed array
    let mut f32_arrays = Vec::new();
    let mut u32_arrays = Vec::new();
    for x in tensors {
        match x {
            TaggedTensorTuple::F32([x]) => f32_arrays.push(x),
            TaggedTensorTuple::U32([x]) => u32_arrays.push(x),
        }

        // early exit: if one dtype didn't match, we bail. (only happens when product is nonzeroD)
        if f32_arrays.len().min(u32_arrays.len()) > 0 {
            return Err(InterpreterError::TypeError(ssa.edge_id));
        }
    }

    Ok(match dtype {
        Dtype::F32 => f32_arrays.try_into().ok().map(TaggedTensorTuple::F32),
        Dtype::U32 => u32_arrays.try_into().ok().map(TaggedTensorTuple::U32),
    }
    .unwrap()) // unwrap OK: we already checked arity!
}
