/// A backend which only does shape computations, not tensor ones.
use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, BackendTensorOps};

#[derive(Clone, Debug)]
pub struct ShapeOnlyBackend;

#[derive(Clone, Debug)]
pub struct ShapeOnly(Shape);

impl ShapeOnly {
    pub fn shape(&self) -> Shape {
        self.0.clone()
    }
}

impl BackendTensorOps for ShapeOnly {
    fn shape(&self) -> Shape {
        self.0.clone()
    }
}

impl Backend for ShapeOnlyBackend {
    type BackendTensor = ShapeOnly;

    fn zeros(&self, shape: Shape, target_dtype: Dtype) -> TaggedTensor<Self> {
        match target_dtype {
            Dtype::F32 => TaggedTensor::F32([ShapeOnly(shape)]),
            Dtype::U32 => TaggedTensor::U32([ShapeOnly(shape)]),
        }
    }

    fn ndarray_from_slice_f32(
        &self,
        _data: &[f32],
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        Ok(TaggedTensor::F32([ShapeOnly(shape)]))
    }

    fn ndarray_from_slice_u32(
        &self,
        _data: &[u32],
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        Ok(TaggedTensor::U32([ShapeOnly(shape)]))
    }

    fn cast(&self, x: TaggedTensor<Self>, _target_dtype: Dtype) -> TaggedTensor<Self> {
        x
    }

    fn matmul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::matmul_shape(x, y)]),
            U32([x, y]) => U32([Self::matmul_shape(x, y)]),
        }
    }

    fn add(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn sub(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn mul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn div(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn pow(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn lt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        self.exact_match(lhs)
    }

    fn sin(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        x
    }

    fn cos(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        x
    }

    fn neg(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        x
    }

    fn broadcast(&self, x: TaggedTensor<Self>, shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::broadcast(arr, shape)]),
            U32([arr]) => U32([Self::broadcast(arr, shape)]),
        }
    }

    fn reshape(&self, _x: TaggedTensor<Self>, new_shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        let arr = ShapeOnly(new_shape);
        match _x {
            F32(_) => F32([arr]),
            U32(_) => U32([arr]),
        }
    }

    fn transpose(&self, x: TaggedTensor<Self>, dim0: usize, dim1: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        let mut shape = x.shape();
        shape.0.swap(dim0, dim1);

        match x {
            F32(_) => F32([ShapeOnly(shape)]),
            U32(_) => U32([ShapeOnly(shape)]),
        }
    }

    fn max(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn sum(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn argmax(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => U32([Self::reduce_last_dim(arr)]),
            U32([arr]) => U32([Self::reduce_last_dim(arr)]),
        }
    }

    fn topk(&self, x: TaggedTensor<Self>, k: usize) -> (TaggedTensor<Self>, TaggedTensor<Self>) {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let new_shape = Self::topk(arr, k);
                let values = TaggedTensor::F32([new_shape.clone()]);
                let indices = TaggedTensor::U32([new_shape]);
                (values, indices)
            }
            _ => panic!("Unsupported type for topk"),
        }
    }

    fn compare(&self, x: TaggedTensorTuple<Self, 2>) -> bool {
        use TaggedTensorTuple::*;
        match x {
            F32([a, b]) => a.0 == b.0,
            U32([a, b]) => a.0 == b.0,
        }
    }

    fn arange(&self, end: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        U32([ShapeOnly(Shape(vec![end]))])
    }

    fn concat(
        &self,
        x: TaggedTensor<Self>,
        y: TaggedTensor<Self>,
        dim: usize,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
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
        x: TaggedTensor<Self>,
        dim: usize,
        indices: TaggedTensor<Self>,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
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
        x: TaggedTensor<Self>,
        dim: usize,
        _start: usize,
        len: usize,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
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

    fn to_vec(&self, _vec: TaggedTensor<Self>) -> TaggedVec {
        panic!("not supported");
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

    fn exact_match(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::exact_shape_match(x, y)]),
            U32([x, y]) => U32([Self::exact_shape_match(x, y)]),
        }
    }

    fn exact_shape_match(x: ShapeOnly, y: ShapeOnly) -> ShapeOnly {
        if x.0 != y.0 {
            panic!("Shape mismatch: {:?} vs {:?}", x.0, y.0);
        }
        x
    }

    fn broadcast(_arr: ShapeOnly, shape: Shape) -> ShapeOnly {
        ShapeOnly(Shape(shape.0))
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

    fn topk(arr: ShapeOnly, k: usize) -> ShapeOnly {
        let mut shape = arr.0.0;
        let last_idx = shape.len() - 1;
        shape[last_idx] = k;
        ShapeOnly(Shape(shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let backend = ShapeOnlyBackend;
        let shape = Shape(vec![2, 3, 4]);
        let zeros_f32 = backend.zeros(shape.clone(), Dtype::F32);
        let zeros_u32 = backend.zeros(shape.clone(), Dtype::U32);

        assert_eq!(zeros_f32.shape(), shape);
        assert_eq!(zeros_u32.shape(), shape);
        assert_eq!(zeros_f32.dtype(), Dtype::F32);
        assert_eq!(zeros_u32.dtype(), Dtype::U32);
    }

    #[test]
    fn test_ndarray_from_slice() {
        let backend = ShapeOnlyBackend;
        let shape = Shape(vec![2, 3]);
        let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_u32 = vec![1u32, 2, 3, 4, 5, 6];

        let result_f32 = backend
            .ndarray_from_slice_f32(&data_f32, shape.clone())
            .unwrap();
        assert_eq!(result_f32.shape(), shape);
        assert_eq!(result_f32.dtype(), Dtype::F32);

        let result_u32 = backend
            .ndarray_from_slice_u32(&data_u32, shape.clone())
            .unwrap();
        assert_eq!(result_u32.shape(), shape);
        assert_eq!(result_u32.dtype(), Dtype::U32);
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
        let lhs = TaggedTensorTuple::F32([x, y]);

        let result = backend.add(lhs);
        assert_eq!(result.shape(), Shape(vec![2, 3]));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_add_different_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![3, 2]));
        let lhs = TaggedTensorTuple::F32([x, y]);

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
        let shape = Shape(vec![2, 5, 3, 4]);
        let result = ShapeOnlyBackend::broadcast(arr, shape);
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
        let tagged_x = TaggedTensorTuple::F32([x]);
        let new_shape = Shape(vec![6]);

        let result = backend.reshape(tagged_x, new_shape.clone());
        assert_eq!(result.shape(), new_shape);
    }

    #[test]
    fn test_neg() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let tagged_x = TaggedTensorTuple::F32([x]);

        let result = backend.neg(tagged_x);
        assert_eq!(result.shape(), Shape(vec![2, 3]));
    }

    #[test]
    fn test_sum() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3, 4]));
        let tagged_x = TaggedTensorTuple::F32([x]);

        let result = backend.sum(tagged_x);
        assert_eq!(result.shape(), Shape(vec![2, 3, 1]));
    }

    #[test]
    fn test_max() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3, 4]));
        let tagged_x = TaggedTensorTuple::F32([x]);

        let result = backend.max(tagged_x);
        assert_eq!(result.shape(), Shape(vec![2, 3, 1]));
    }

    #[test]
    fn test_compare_same_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![2, 3]));
        let lhs = TaggedTensorTuple::F32([x, y]);

        assert!(backend.compare(lhs));
    }

    #[test]
    fn test_compare_different_shapes() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let y = ShapeOnly(Shape(vec![3, 2]));
        let lhs = TaggedTensorTuple::F32([x, y]);

        assert!(!backend.compare(lhs));
    }

    #[test]
    fn test_cast() {
        let backend = ShapeOnlyBackend;
        let x = ShapeOnly(Shape(vec![2, 3]));
        let tagged_x = TaggedTensorTuple::F32([x]);

        let result = backend.cast(tagged_x, Dtype::U32);
        assert_eq!(result.shape(), Shape(vec![2, 3]));
    }

    #[test]
    fn test_topk_shapes() {
        let backend = ShapeOnlyBackend;
        let input = TaggedTensor::F32([ShapeOnly(Shape(vec![2, 5]))]);
        let (values, indices) = backend.topk(input, 3);
        assert_eq!(values.shape(), Shape(vec![2, 3]));
        assert_eq!(indices.shape(), Shape(vec![2, 3]));
        assert_eq!(values.dtype(), Dtype::F32);
        assert_eq!(indices.dtype(), Dtype::U32);
    }
}
