/// A backend which only does shape computations, not tensor ones.
use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError};

#[derive(Clone, Debug)]
pub struct ShapeOnlyBackend;

#[derive(Clone, Debug)]
pub struct ShapeOnly(Shape);

impl ShapeOnly {
    pub fn shape(&self) -> Shape {
        self.0.clone()
    }
}

impl<D: HasDtype> crate::interpreter::backend::NdArray<D> for ShapeOnly {
    type Backend = ShapeOnlyBackend;

    fn shape(&self) -> Shape {
        self.0.clone()
    }
}

impl Backend for ShapeOnlyBackend {
    type NdArray<D: HasDtype> = ShapeOnly;

    fn scalar<D: HasDtype>(&self, _d: D) -> Self::NdArray<D> {
        ShapeOnly(Shape(vec![]))
    }

    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        ShapeOnly(shape)
    }

    fn ndarray_from_slice<D: HasDtype>(
        &self,
        _data: &[D],
        shape: Shape,
    ) -> Result<Self::NdArray<D>, BackendError> {
        Ok(ShapeOnly(shape))
    }

    fn cast(&self, x: TaggedNdArray<Self>, _target_dtype: Dtype) -> TaggedNdArray<Self> {
        x
    }

    fn matmul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::matmul_shape(x, y)]),
            U32([x, y]) => U32([Self::matmul_shape(x, y)]),
        }
    }

    fn add(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn sub(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn mul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn div(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn pow(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn neg(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        x
    }

    fn broadcast(&self, x: TaggedNdArray<Self>, shape_prefix: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::broadcast_with_prefix(arr, shape_prefix)]),
            U32([arr]) => U32([Self::broadcast_with_prefix(arr, shape_prefix)]),
        }
    }

    fn reshape(&self, _x: TaggedNdArray<Self>, new_shape: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        let arr = ShapeOnly(new_shape);
        match _x {
            F32(_) => F32([arr]),
            U32(_) => U32([arr]),
        }
    }

    fn max(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn sum(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn compare(&self, x: TaggedNdArrayTuple<Self, 2>) -> bool {
        use TaggedNdArrayTuple::*;
        match x {
            F32([a, b]) => a.0 == b.0,
            U32([a, b]) => a.0 == b.0,
        }
    }

    fn arange(&self, end: usize) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        U32([ShapeOnly(Shape(vec![end]))])
    }

    fn concat(
        &self,
        x: TaggedNdArray<Self>,
        y: TaggedNdArray<Self>,
        dim: usize,
    ) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match (x, y) {
            (F32([ShapeOnly(a)]), F32([ShapeOnly(b)])) => {
                let mut s = a.clone();
                s[dim] = a[dim] + b[dim];
                F32([ShapeOnly(s)])
            }
            (U32([ShapeOnly(a)]), U32([ShapeOnly(b)])) => {
                let mut s = a.clone();
                s[dim] = a[dim] + b[dim];
                U32([ShapeOnly(s)])
            }
            _ => panic!("Incompatible types for concatenation"),
        }
    }

    fn index(
        &self,
        x: TaggedNdArray<Self>,
        dim: usize,
        indices: TaggedNdArray<Self>,
    ) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        let shape = &match indices {
            F32([shape]) => shape,
            U32([shape]) => shape,
        }
        .0;
        assert_eq!(shape.rank(), 1);
        let n = shape[0]; // first dim

        match x {
            F32([ShapeOnly(mut s)]) => {
                s[dim] = n;
                F32([ShapeOnly(s)])
            }
            U32([ShapeOnly(mut s)]) => {
                s[dim] = n;
                U32([ShapeOnly(s)])
            }
        }
    }

    fn slice(
        &self,
        x: TaggedNdArray<Self>,
        dim: usize,
        _start: usize,
        len: usize,
    ) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([ShapeOnly(mut s)]) => {
                s[dim] = len;
                F32([ShapeOnly(s)])
            }
            U32([ShapeOnly(mut s)]) => {
                s[dim] = len;
                U32([ShapeOnly(s)])
            }
        }
    }
}

impl ShapeOnlyBackend {
    fn matmul_shape(lhs: ShapeOnly, rhs: ShapeOnly) -> ShapeOnly {
        let lhs_shape = &lhs.0.0;
        let rhs_shape = &rhs.0.0;

        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            panic!("matmul: both operands must have at least 2 dimensions");
        }

        let lhs_batch = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch = &rhs_shape[..rhs_shape.len() - 2];
        let lhs_m = lhs_shape[lhs_shape.len() - 2];
        let lhs_k = lhs_shape[lhs_shape.len() - 1];
        let rhs_k = rhs_shape[rhs_shape.len() - 2];
        let rhs_n = rhs_shape[rhs_shape.len() - 1];

        if lhs_k != rhs_k {
            panic!("matmul: inner dimensions must match");
        }

        if lhs_batch != rhs_batch {
            panic!("matmul: batch dimensions must match");
        }

        let mut result_shape = lhs_batch.to_vec();
        result_shape.push(lhs_m);
        result_shape.push(rhs_n);

        ShapeOnly(Shape(result_shape))
    }

    fn exact_shape_match(x: ShapeOnly, y: ShapeOnly) -> ShapeOnly {
        if x.0 != y.0 {
            panic!("Shape mismatch: {:?} vs {:?}", x.0, y.0);
        }
        x
    }

    fn broadcast_with_prefix(arr: ShapeOnly, shape_prefix: Shape) -> ShapeOnly {
        let mut result_shape = shape_prefix.0;
        result_shape.extend_from_slice(&arr.0.0);
        ShapeOnly(Shape(result_shape))
    }

    fn reduce_last_dim(arr: ShapeOnly) -> ShapeOnly {
        let shape = &arr.0.0;
        if shape.is_empty() {
            return arr;
        }

        let mut result_shape = shape.clone();
        let last_idx = result_shape.len() - 1;
        result_shape[last_idx] = 1;

        ShapeOnly(Shape(result_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::core::Shape;

    #[test]
    fn test_scalar() {
        let backend = ShapeOnlyBackend;
        let scalar_f32 = backend.scalar(1.0f32);
        let scalar_u32 = backend.scalar(42u32);

        assert_eq!(scalar_f32.0, Shape(vec![]));
        assert_eq!(scalar_u32.0, Shape(vec![]));
    }

    #[test]
    fn test_zeros() {
        let backend = ShapeOnlyBackend;
        let shape = Shape(vec![2, 3, 4]);
        let zeros_f32: ShapeOnly = backend.zeros::<f32>(shape.clone());
        let zeros_u32: ShapeOnly = backend.zeros::<u32>(shape.clone());

        assert_eq!(zeros_f32.0, shape);
        assert_eq!(zeros_u32.0, shape);
    }

    #[test]
    fn test_ndarray_from_slice() {
        let backend = ShapeOnlyBackend;
        let shape = Shape(vec![2, 3]);
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = backend.ndarray_from_slice(&data, shape.clone()).unwrap();
        assert_eq!(result.0, shape);
    }

    #[test]
    fn test_exact_shape_match_same_shapes() {
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![2, 3]));
        let result = ShapeOnlyBackend::exact_shape_match(x, y);
        assert_eq!(result.0, Shape(vec![2, 3]));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_exact_shape_match_different_shapes() {
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![3, 2]));
        ShapeOnlyBackend::exact_shape_match(x, y);
    }

    #[test]
    fn test_add_same_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![2, 3]));
        let lhs = TaggedNdArrayTuple::F32([x, y]);

        let result = backend.add(lhs);
        assert_eq!(result.shape(), Shape(vec![2, 3]));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_add_different_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![3, 2]));
        let lhs = TaggedNdArrayTuple::F32([x, y]);

        backend.add(lhs);
    }

    #[test]
    fn test_matmul_2d() {
        let lhs = ShapeOnly(Shape(vec![3, 4]));
        let rhs = ShapeOnly(Shape(vec![4, 5]));
        let result = ShapeOnlyBackend::matmul_shape(lhs, rhs);
        assert_eq!(result.shape(), Shape(vec![3, 5]));
    }

    #[test]
    fn test_matmul_batched() {
        let lhs = ShapeOnly(Shape(vec![2, 3, 4]));
        let rhs = ShapeOnly(Shape(vec![2, 4, 5]));
        let result = ShapeOnlyBackend::matmul_shape(lhs, rhs);
        assert_eq!(result.shape(), Shape(vec![2, 3, 5]));
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match")]
    fn test_matmul_incompatible() {
        let lhs = ShapeOnly(Shape(vec![3, 4]));
        let rhs = ShapeOnly(Shape(vec![5, 6]));
        ShapeOnlyBackend::matmul_shape(lhs, rhs);
    }

    #[test]
    fn test_broadcast_with_prefix() {
        let arr = ShapeOnly(Shape(vec![3, 4]));
        let prefix = Shape(vec![2, 5]);
        let result = ShapeOnlyBackend::broadcast_with_prefix(arr, prefix);
        assert_eq!(result.shape(), Shape(vec![2, 5, 3, 4]));
    }

    #[test]
    fn test_reduce_last_dim() {
        let arr = ShapeOnly(Shape(vec![2, 3, 4]));
        let result = ShapeOnlyBackend::reduce_last_dim(arr);
        assert_eq!(result.shape(), Shape(vec![2, 3, 1]));
    }

    #[test]
    fn test_reduce_last_dim_1d() {
        let arr = ShapeOnly(Shape(vec![5]));
        let result = ShapeOnlyBackend::reduce_last_dim(arr);
        assert_eq!(result.shape(), Shape(vec![1]));
    }

    #[test]
    fn test_reduce_last_dim_scalar() {
        let arr = ShapeOnly(Shape(vec![]));
        let result = ShapeOnlyBackend::reduce_last_dim(arr);
        assert_eq!(result.shape(), Shape(vec![]));
    }

    #[test]
    fn test_reshape() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let tagged_x = TaggedNdArrayTuple::F32([x]);
        let new_shape = Shape(vec![6]);

        let result = backend.reshape(tagged_x, new_shape.clone());
        assert_eq!(result.shape(), new_shape);
    }

    #[test]
    fn test_neg() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let tagged_x = TaggedNdArrayTuple::F32([x]);

        let result = backend.neg(tagged_x);
        assert_eq!(result.shape(), Shape(vec![2, 3]));
    }

    #[test]
    fn test_sum() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3, 4]));
        let tagged_x = TaggedNdArrayTuple::F32([x]);

        let result = backend.sum(tagged_x);
        assert_eq!(result.shape(), Shape(vec![2, 3, 1]));
    }

    #[test]
    fn test_max() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3, 4]));
        let tagged_x = TaggedNdArrayTuple::F32([x]);

        let result = backend.max(tagged_x);
        assert_eq!(result.shape(), Shape(vec![2, 3, 1]));
    }

    #[test]
    fn test_compare_same_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![2, 3]));
        let lhs = TaggedNdArrayTuple::F32([x, y]);

        assert!(backend.compare(lhs));
    }

    #[test]
    fn test_compare_different_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![3, 2]));
        let lhs = TaggedNdArrayTuple::F32([x, y]);

        assert!(!backend.compare(lhs));
    }

    #[test]
    fn test_cast() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let tagged_x = TaggedNdArrayTuple::F32([x]);

        let result = backend.cast(tagged_x, Dtype::U32);
        assert_eq!(result.shape(), Shape(vec![2, 3]));
    }
}
