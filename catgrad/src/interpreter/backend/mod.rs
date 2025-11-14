use super::types::*;
use crate::category::core::{Dtype, Shape};
use std::fmt::Debug;

/// Default backend: only perform shape computations.
pub mod shape_only;

#[cfg(feature = "ndarray-backend")]
pub mod ndarray;

#[cfg(feature = "candle-backend")]
pub mod candle;

/// Backends implement this trait.
///
/// # Conventions
///
/// - Methods take a `TaggedNdArrayTuple<Self; N>`: a tuple of arrays of the *same dtype*
/// - A method of this signature is expected to work for *any dtype*
/// - Kernels *never* implicitly broadcast their arguments. Shapes must be an exact match, or error.
/// - Reductions preserve rank. For example, sum tensor shape `[2,3,4]` gives `[2,3,1]` instead of `[2,3]`.
pub trait Backend: Clone + Debug {
    /// Representation of tensor values. (e.g., device ptrs, Vec, etc.)
    type BackendTensor: BackendTensorOps;

    fn zeros(&self, shape: Shape, target_dtype: Dtype) -> TaggedTensor<Self>;

    fn ndarray_from_slice_f32(
        &self,
        data: &[f32],
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError>;

    fn ndarray_from_slice_u32(
        &self,
        data: &[u32],
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError>;

    fn cast(&self, x: TaggedTensor<Self>, target_dtype: Dtype) -> TaggedTensor<Self>;
    fn matmul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn add(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn sub(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn mul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn div(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn pow(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn lt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self>;
    fn sin(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self>;
    fn cos(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self>;
    fn neg(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self>;
    fn broadcast(&self, x: TaggedTensor<Self>, shape: Shape) -> TaggedTensor<Self>;
    fn reshape(&self, x: TaggedTensor<Self>, new_shape: Shape) -> TaggedTensor<Self>;
    fn transpose(&self, x: TaggedTensor<Self>, dim0: usize, dim1: usize) -> TaggedTensor<Self>;
    fn max(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self>;
    fn sum(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self>;
    fn argmax(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self>;
    fn topk(&self, x: TaggedTensor<Self>, k: usize) -> (TaggedTensor<Self>, TaggedTensor<Self>);
    fn compare(&self, x: TaggedTensorTuple<Self, 2>) -> bool;
    fn concat(
        &self,
        x: TaggedTensor<Self>,
        y: TaggedTensor<Self>,
        dim: usize,
    ) -> TaggedTensor<Self>;
    fn index(
        &self,
        x: TaggedTensor<Self>,
        dim: usize,
        indices: TaggedTensor<Self>,
    ) -> TaggedTensor<Self>;
    fn slice(
        &self,
        x: TaggedTensor<Self>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> TaggedTensor<Self>;
    fn arange(&self, end: usize) -> TaggedTensor<Self>;

    fn to_vec(&self, vec: TaggedTensor<Self>) -> TaggedVec;
}

pub trait BackendTensorOps: Send + Sync + Clone + Debug {
    fn shape(&self) -> Shape;
}

#[derive(Debug, Clone)]
pub enum BackendError {
    /// The size of a shape did not match the number of elements in a Tensor
    ShapeError,
}
