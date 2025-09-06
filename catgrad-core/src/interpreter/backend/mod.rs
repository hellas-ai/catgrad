use super::types::*;
use crate::category::core::{Dtype, Shape};
use std::fmt::Debug;

#[cfg(feature = "ndarray-backend")]
pub mod ndarray;

/// Backends implement this trait.
///
/// # Conventions
///
/// - Methods take a `TaggedNdArrayTuple<Self; N>`: a tuple of arrays of the *same dtype*
/// - A method of this signature is expected to work for *any dtype*
/// -
pub trait Backend: Send + Sync + Clone + Debug {
    /// Representation of tensor values. (e.g., device ptrs, Vec, etc.)
    type NdArray<D: HasDtype>: NdArray<D, Backend = Self>;

    // Generic helper functions to create ndarrays.
    fn scalar<D: HasDtype>(&self, d: D) -> Self::NdArray<D>;
    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D>;

    fn ndarray_from_slice<D: HasDtype>(
        &self,
        data: &[D],
        shape: Shape,
    ) -> Result<Self::NdArray<D>, BackendError>;

    fn cast(&self, x: TaggedNdArray<Self>, target_dtype: Dtype) -> TaggedNdArray<Self>;
    fn matmul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self>;
    fn add(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self>;
    fn mul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self>;
    fn div(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self>;
    fn pow(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self>;
    fn neg(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self>;
    fn broadcast(&self, x: TaggedNdArray<Self>, shape_prefix: Shape) -> TaggedNdArray<Self>;
    fn reshape(&self, x: TaggedNdArray<Self>, new_shape: Shape) -> TaggedNdArray<Self>;
}

pub trait NdArray<D: HasDtype>: Send + Sync + Clone + Debug + PartialEq {
    type Backend: Backend;
    fn shape(&self) -> Shape;
}

#[derive(PartialEq, Debug, Clone)]
pub enum BackendError {
    /// The size of a shape did not match the number of elements in a Tensor
    ShapeError,
}
