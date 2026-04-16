//! Interpretation of core terms
pub mod types;
pub use types::*;

#[allow(clippy::module_inception)]
pub mod interpreter;
pub use interpreter::*;

pub mod backend;
//pub mod parameters;
pub mod tensor_op;

pub use crate::category::core::{Dtype, Shape};
pub use backend::{Backend, BackendError};
use half::{bf16, f16};

#[cfg(all(test, feature = "ndarray-backend"))]
mod tests;

// Create a tensor

pub fn tensor<B: Backend, T: IntoTagged<B, 1>>(
    backend: &B,
    shape: Shape,
    data: Vec<T>,
) -> Result<Value<B>, BackendError> {
    if shape.size() != data.len() {
        return Err(BackendError::ShapeError);
    }
    let tagged = TaggedTensor::from_vec(backend, data, shape)?;
    Ok(Value::Tensor(tagged))
}

pub fn float_tensor<B: Backend>(
    backend: &B,
    shape: Shape,
    data: Vec<f32>,
    dtype: Dtype,
) -> Result<Value<B>, BackendError> {
    match dtype {
        Dtype::F32 => tensor(backend, shape, data),
        Dtype::F16 => tensor(
            backend,
            shape,
            data.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
        ),
        Dtype::BF16 => tensor(
            backend,
            shape,
            data.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        ),
        Dtype::U32 => panic!("float_tensor requires a floating-point dtype"),
    }
}
