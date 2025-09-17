use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, NdArray};
use ndarray::{ArrayD, IxDyn};

#[derive(Clone, Debug)]
pub struct NdArrayBackend;

impl Backend for NdArrayBackend {
    type NdArray<D: HasDtype> = ArrayD<D>;

    fn scalar<D: HasDtype>(&self, d: D) -> Self::NdArray<D> {
        ArrayD::from_elem(IxDyn(&[]), d)
    }

    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        let dims: Vec<usize> = shape.0;
        ArrayD::from_elem(IxDyn(&dims), D::default())
    }

    fn ndarray_from_slice<D: HasDtype>(
        &self,
        data: &[D],
        shape: Shape,
    ) -> Result<Self::NdArray<D>, BackendError> {
        let dims: Vec<usize> = shape.0;
        ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).map_err(|_| BackendError::ShapeError)
    }

    fn arange(&self, end: usize) -> TaggedNdArray<Self> {
        let result = ndarray::Array::range(0.0, end as f32, 1.0).into_dyn();
        let result = TaggedNdArray::F32([result]);
        self.cast(result, Dtype::U32)
    }

    fn cast(&self, x: TaggedNdArray<Self>, target_dtype: Dtype) -> TaggedNdArray<Self> {
        match (&x, target_dtype) {
            (TaggedNdArray::F32(arr), Dtype::U32) => {
                let data: Vec<u32> = arr[0].iter().map(|&val| val as u32).collect();
                let result = ArrayD::from_shape_vec(arr[0].raw_dim(), data).unwrap();
                TaggedNdArray::U32([result])
            }
            (TaggedNdArray::U32(arr), Dtype::F32) => {
                let data: Vec<f32> = arr[0].iter().map(|&val| val as f32).collect();
                let result = ArrayD::from_shape_vec(arr[0].raw_dim(), data).unwrap();
                TaggedNdArray::F32([result])
            }
            (TaggedNdArray::F32(_), Dtype::F32) => x,
            (TaggedNdArray::U32(_), Dtype::U32) => x,
        }
    }

    fn matmul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::batched_matmul(x, y)]),
            U32([x, y]) => U32([Self::batched_matmul(x, y)]),
        }
    }

    fn add(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::add(x, y)]),
            U32([x, y]) => U32([Self::add(x, y)]),
        }
    }

    fn sub(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::sub(x, y)]),
            U32([x, y]) => U32([Self::sub(x, y)]),
        }
    }

    fn mul(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::mul(x, y)]),
            U32([x, y]) => U32([Self::mul(x, y)]),
        }
    }

    fn div(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::div(x, y)]),
            U32([x, y]) => U32([Self::div(x, y)]),
        }
    }

    fn pow(&self, lhs: TaggedNdArrayTuple<Self, 2>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::pow_f32(x, y)]),
            U32([x, y]) => U32([Self::pow_u32(x, y)]),
        }
    }

    fn neg(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::neg_f32(arr)]),
            U32([arr]) => U32([Self::neg_u32(arr)]),
        }
    }

    fn max(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::max_f32(arr)]),
            U32([arr]) => U32([Self::max_u32(arr)]),
        }
    }

    fn sum(&self, x: TaggedNdArray<Self>) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::sum(arr)]),
            U32([arr]) => U32([Self::sum(arr)]),
        }
    }

    fn broadcast(&self, x: TaggedNdArray<Self>, shape_prefix: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::broadcast_ndarray(arr, shape_prefix)]),
            U32([arr]) => U32([Self::broadcast_ndarray(arr, shape_prefix)]),
        }
    }

    fn index(
        &self,
        x: TaggedNdArray<Self>,
        dim: usize,
        indices: TaggedNdArray<Self>,
    ) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match (x, indices) {
            (F32([arr]), U32([indices])) => F32([Self::index_ndarray(arr, dim, indices)]),
            (U32([arr]), U32([indices])) => U32([Self::index_ndarray(arr, dim, indices)]),
            _ => panic!("Invalid input types for indexing"),
        }
    }

    fn slice(
        &self,
        x: TaggedNdArray<Self>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::slice_ndarray(arr, dim, start, len)]),
            U32([arr]) => U32([Self::slice_ndarray(arr, dim, start, len)]),
        }
    }

    fn reshape(&self, x: TaggedNdArray<Self>, new_shape: Shape) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match x {
            F32([arr]) => F32([Self::reshape_ndarray(arr, new_shape)]),
            U32([arr]) => U32([Self::reshape_ndarray(arr, new_shape)]),
        }
    }

    fn concat(
        &self,
        x: TaggedNdArray<Self>,
        y: TaggedNdArray<Self>,
        dim: usize,
    ) -> TaggedNdArray<Self> {
        use TaggedNdArrayTuple::*;
        match (x, y) {
            (F32([a]), F32([b])) => F32([Self::concat_ndarray(a, b, dim)]),
            (U32([a]), U32([b])) => U32([Self::concat_ndarray(a, b, dim)]),
            _ => panic!("Incompatible types for concatenation"),
        }
    }

    fn compare(&self, x: TaggedNdArrayTuple<Self, 2>) -> bool {
        use TaggedNdArrayTuple::*;
        match x {
            F32([a, b]) => a == b,
            U32([a, b]) => a == b,
        }
    }
}

impl NdArrayBackend {
    fn reshape_ndarray<D: HasDtype>(arr: ArrayD<D>, new_shape: Shape) -> ArrayD<D> {
        let new_dims = ndarray::IxDyn(&new_shape.0);
        arr.to_shape(new_dims).unwrap().to_owned()
    }

    fn broadcast_ndarray<D: HasDtype + Clone>(arr: ArrayD<D>, shape_prefix: Shape) -> ArrayD<D> {
        let current_shape = arr.shape().to_vec();
        let mut new_shape = shape_prefix.0;
        new_shape.extend_from_slice(&current_shape);

        // Use ndarray's broadcast to expand dimensions
        let broadcasted = arr.broadcast(ndarray::IxDyn(&new_shape)).unwrap();
        broadcasted.to_owned()
    }

    fn index_ndarray<D: HasDtype>(arr: ArrayD<D>, dim: usize, indices: ArrayD<u32>) -> ArrayD<D> {
        let idx = indices.iter().map(|&i| i as usize).collect::<Vec<_>>();

        arr.select(ndarray::Axis(dim), &idx)
    }

    fn slice_ndarray<D: HasDtype>(
        arr: ArrayD<D>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> ArrayD<D> {
        let r = arr.slice_axis(ndarray::Axis(dim), (start..start + len).into());
        r.to_owned()
    }

    fn concat_ndarray<D: HasDtype>(a: ArrayD<D>, b: ArrayD<D>, dim: usize) -> ArrayD<D> {
        ndarray::concatenate(ndarray::Axis(dim), &[a.view(), b.view()]).unwrap()
    }

    fn add<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        // PERFORMANCE does ndarray reuse an x/y buffer if possible? If not, can we improve things
        // using in-place updates? That is, use `x += y` if x is contiguous.
        x + y
    }

    fn sub<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        // PERFORMANCE does ndarray reuse an x/y buffer if possible? If not, can we improve things
        // using in-place updates? That is, use `x -= y` if x is contiguous.
        x - y
    }

    fn mul<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        x * y
    }

    fn div<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        x / y
    }

    fn neg_f32(x: ArrayD<f32>) -> ArrayD<f32> {
        x.map(|&v| -v)
    }

    fn neg_u32(x: ArrayD<u32>) -> ArrayD<u32> {
        // For u32, negation doesn't make much sense, but we'll implement it anyway
        // using wrapping negation
        x.map(|&v| v.wrapping_neg())
    }

    fn pow_f32(x: ArrayD<f32>, y: ArrayD<f32>) -> ArrayD<f32> {
        ndarray::Zip::from(&x)
            .and(&y)
            .map_collect(|&a, &b| a.powf(b))
    }

    fn pow_u32(x: ArrayD<u32>, y: ArrayD<u32>) -> ArrayD<u32> {
        ndarray::Zip::from(&x)
            .and(&y)
            .map_collect(|&a, &b| a.pow(b))
    }

    fn max_f32(x: ArrayD<f32>) -> ArrayD<f32> {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.fold_axis(ndarray::Axis(axis), f32::MIN, |acc, &x| acc.max(x))
    }

    fn max_u32(x: ArrayD<u32>) -> ArrayD<u32> {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.fold_axis(ndarray::Axis(axis), u32::MIN, |acc, x| *acc.max(x))
    }

    fn sum<D>(x: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.sum_axis(ndarray::Axis(axis))
    }

    fn matmul_generic<D>(lhs: ArrayD<D>, rhs: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        // For now, only handle rank 2 case
        assert_eq!(lhs.ndim(), 2, "matmul: self must be rank 2");
        assert_eq!(rhs.ndim(), 2, "matmul: rhs must be rank 2");

        // Convert ArrayD to Array2 for dot product
        // NOTE: ndarray needs to know we have 2d arrays statically to use .dot():
        // https://stackoverflow.com/questions/79035190/
        let self_2d = lhs.into_dimensionality::<ndarray::Ix2>().unwrap();
        let rhs_2d = rhs.into_dimensionality::<ndarray::Ix2>().unwrap();

        // Perform matrix multiplication and convert back to ArrayD
        self_2d.dot(&rhs_2d).into_dyn()
    }

    pub fn batched_matmul<D>(lhs: ArrayD<D>, rhs: ArrayD<D>) -> ArrayD<D>
    where
        D: HasDtype + ndarray::LinalgScalar,
    {
        // PERFORMANCE: Haven't checked fast/slow paths in this code; use rayon to parallelise?
        assert!(
            lhs.ndim() >= 2,
            "batched_matmul: lhs must be at least rank 2"
        );
        assert!(
            rhs.ndim() >= 2,
            "batched_matmul: rhs must be at least rank 2"
        );

        let lhs_shape = lhs.shape().to_vec();
        let rhs_shape = rhs.shape().to_vec();

        if lhs.ndim() == 2 && rhs.ndim() == 2 {
            // Regular matrix multiplication
            return Self::matmul_generic(lhs, rhs);
        }

        // Get the batch dimensions and matrix dimensions
        let lhs_batch_dims = &lhs_shape[..lhs_shape.len() - 2];
        let rhs_batch_dims = &rhs_shape[..rhs_shape.len() - 2];
        let lhs_matrix_dims = &lhs_shape[lhs_shape.len() - 2..];
        let rhs_matrix_dims = &rhs_shape[rhs_shape.len() - 2..];

        // Check matrix dimensions compatibility
        assert_eq!(
            lhs_matrix_dims[1], rhs_matrix_dims[0],
            "batched_matmul: incompatible matrix dimensions"
        );

        // For simplicity, require batch dimensions to match exactly
        assert_eq!(
            lhs_batch_dims, rhs_batch_dims,
            "batched_matmul: batch dimensions must match"
        );

        let batch_size: usize = lhs_batch_dims.iter().product();
        let lhs_m = lhs_matrix_dims[0];
        let lhs_k = lhs_matrix_dims[1];
        let rhs_k = rhs_matrix_dims[0];
        let rhs_n = rhs_matrix_dims[1];

        // Reshape to (batch_size, m, k) and (batch_size, k, n)
        let lhs_reshaped = lhs.to_shape((batch_size, lhs_m, lhs_k)).unwrap();
        let rhs_reshaped = rhs.to_shape((batch_size, rhs_k, rhs_n)).unwrap();

        // Perform batched matrix multiplication
        let mut result_data = Vec::with_capacity(batch_size * lhs_m * rhs_n);

        for b in 0..batch_size {
            let lhs_batch = lhs_reshaped.slice(ndarray::s![b, .., ..]).to_owned();
            let rhs_batch = rhs_reshaped.slice(ndarray::s![b, .., ..]).to_owned();
            let batch_result = Self::matmul_generic(lhs_batch.into_dyn(), rhs_batch.into_dyn());
            result_data.extend_from_slice(batch_result.as_slice().unwrap());
        }

        // Reshape result back to proper batch shape
        let mut result_shape = lhs_batch_dims.to_vec();
        result_shape.push(lhs_m);
        result_shape.push(rhs_n);

        ArrayD::from_shape_vec(ndarray::IxDyn(&result_shape), result_data).unwrap()
    }
}

impl<D: HasDtype> NdArray<D> for ArrayD<D> {
    type Backend = NdArrayBackend;

    fn shape(&self) -> Shape {
        Shape(self.shape().to_vec())
    }
}

#[test]
fn test_batched_matmul() {
    use ndarray::ArrayD;

    // Test with 2 batch dimensions: [2, 3, 2, 2] × [2, 3, 2, 1] = [2, 3, 2, 1]
    let lhs_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // batch 0,0
        5.0, 6.0, 7.0, 8.0, // batch 0,1
        9.0, 10.0, 11.0, 12.0, // batch 0,2
        13.0, 14.0, 15.0, 16.0, // batch 1,0
        17.0, 18.0, 19.0, 20.0, // batch 1,1
        21.0, 22.0, 23.0, 24.0, // batch 1,2
    ];
    let lhs = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 2, 2]), lhs_data).unwrap();

    let rhs_data = vec![
        1.0f32, 2.0, // batch 0,0
        3.0, 4.0, // batch 0,1
        5.0, 6.0, // batch 0,2
        7.0, 8.0, // batch 1,0
        9.0, 10.0, // batch 1,1
        11.0, 12.0, // batch 1,2
    ];
    let rhs = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 2, 1]), rhs_data).unwrap();

    let result = NdArrayBackend::batched_matmul(lhs, rhs);

    // Expected shape: [2, 3, 2, 1]
    assert_eq!(result.shape(), &[2, 3, 2, 1]);

    let expected = [
        5.0f32, 11.0, // batch 0,0
        39.0, 53.0, // batch 0,1
        105.0, 127.0, // batch 0,2
        203.0, 233.0, // batch 1,0
        333.0, 371.0, // batch 1,1
        495.0, 541.0, // batch 1,2
    ];

    let result_flat = result.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_add() {
    use ndarray::ArrayD;

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), x_data).unwrap();

    let y_data = vec![5.0f32, 6.0, 7.0, 8.0];
    let y = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), y_data).unwrap();

    let result = NdArrayBackend::add(x, y);

    let expected = [6.0f32, 8.0, 10.0, 12.0];
    let result_flat = result.as_slice().unwrap();

    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_sub() {
    use ndarray::ArrayD;

    let x_data = vec![10.0f32, 8.0, 6.0, 4.0];
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), x_data).unwrap();

    let y_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let y = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), y_data).unwrap();

    let result = NdArrayBackend::sub(x, y);

    let expected = [9.0f32, 6.0, 3.0, 0.0];
    let result_flat = result.as_slice().unwrap();

    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_sum() {
    use ndarray::ArrayD;

    // Test summing across last dimension: [2, 3] -> [2]
    let x_data = vec![1u32, 2, 3, 4, 5, 6];
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_data).unwrap();

    let result = NdArrayBackend::sum(x);

    // Expected: [1+2+3, 4+5+6] = [6, 15]
    let expected = [6u32, 15];
    assert_eq!(result.shape(), &[2]);

    let result_flat = result.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    // Test with 3D tensor: [2, 2, 3] -> [2, 2]
    let x_data_3d = vec![
        1.0f32, 2.0, 3.0, // [0,0,:]
        4.0, 5.0, 6.0, // [0,1,:]
        7.0, 8.0, 9.0, // [1,0,:]
        10.0, 11.0, 12.0, // [1,1,:]
    ];
    let x_3d = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2, 3]), x_data_3d).unwrap();

    let result_3d = NdArrayBackend::sum(x_3d);

    // Expected: [[1+2+3, 4+5+6], [7+8+9, 10+11+12]] = [[6, 15], [24, 33]]
    let expected_3d = [6.0f32, 15.0, 24.0, 33.0];
    assert_eq!(result_3d.shape(), &[2, 2]);

    let result_3d_flat = result_3d.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_3d_flat.iter().zip(expected_3d.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}
#[test]
fn test_max() {
    use ndarray::ArrayD;

    // Test max across last dimension: [2, 3] -> [2]
    let x_data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0];
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_data).unwrap();

    let result = NdArrayBackend::max_f32(x);

    // Expected: [max(1,5,3), max(2,8,4)] = [5, 8]
    let expected = [5.0f32, 8.0];
    assert_eq!(result.shape(), &[2]);

    let result_flat = result.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    // Test with u32 array: [2, 2] -> [2]
    let x_data_u32 = vec![1u32, 5, 3, 2];
    let x_u32 = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), x_data_u32).unwrap();

    let result_u32 = NdArrayBackend::max_u32(x_u32);

    // Expected: [max(1,5), max(3,2)] = [5, 3]
    let expected_u32 = [5u32, 3];
    assert_eq!(result_u32.shape(), &[2]);

    let result_u32_flat = result_u32.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_u32_flat.iter().zip(expected_u32.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    // Test with 3D tensor: [2, 2, 3] -> [2, 2]
    let x_data_3d = vec![
        1.0f32, 2.0, 3.0, // [0,0,:]
        4.0, 5.0, 6.0, // [0,1,:]
        7.0, 8.0, 9.0, // [1,0,:]
        10.0, 11.0, 12.0, // [1,1,:]
    ];
    let x_3d = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2, 3]), x_data_3d).unwrap();

    let result_3d = NdArrayBackend::max_f32(x_3d);

    // Expected: [[max(1,2,3), max(4,5,6)], [max(7,8,9), max(10,11,12)]] = [[3, 6], [9, 12]]
    let expected_3d = [3.0f32, 6.0, 9.0, 12.0];
    assert_eq!(result_3d.shape(), &[2, 2]);

    let result_3d_flat = result_3d.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_3d_flat.iter().zip(expected_3d.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}
