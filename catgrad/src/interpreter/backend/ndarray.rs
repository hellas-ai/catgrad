use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, BackendTensorOps};
use ndarray::{ArcArray, ArrayD, Axis, IxDyn};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct NdArrayBackend;

// We can't really handle HKTs properly in rust, so that means the ndarray backend suffers through
// some unfortunate "double tagging" of values.
// this has some unsafe methods (to_<dtype>), but panics here are always programmer errors.
#[derive(Clone, Debug, PartialEq)]
pub enum TaggedArrayD {
    F32(ArcArray<f32, IxDyn>),
    U32(ArcArray<u32, IxDyn>),
}

impl TaggedArrayD {
    fn unwrap_f32(self) -> ArrayD<f32> {
        match self {
            TaggedArrayD::F32(x) => x.into_owned(),
            _ => panic!("Not f32 array"),
        }
    }

    fn unwrap_u32(self) -> ArrayD<u32> {
        match self {
            TaggedArrayD::U32(x) => x.into_owned(),
            _ => panic!("Not u32 array"),
        }
    }
}

fn from_f32(x: ArrayD<f32>) -> TaggedTensor<NdArrayBackend> {
    TaggedTensor::F32([TaggedArrayD::F32(x.into_shared())])
}

fn from_u32(x: ArrayD<u32>) -> TaggedTensor<NdArrayBackend> {
    TaggedTensor::U32([TaggedArrayD::U32(x.into_shared())])
}

impl Backend for NdArrayBackend {
    type BackendTensor = TaggedArrayD;

    fn to_vec(&self, vec: TaggedTensor<Self>) -> TaggedVec {
        match vec {
            TaggedTensor::F32([x]) => TaggedVec::F32(x.unwrap_f32().as_slice().unwrap().to_vec()),
            TaggedTensor::U32([x]) => TaggedVec::U32(x.unwrap_u32().as_slice().unwrap().to_vec()),
        }
    }

    fn zeros(&self, shape: Shape, target_dtype: Dtype) -> TaggedTensor<Self> {
        let dims: Vec<usize> = shape.0;
        match target_dtype {
            Dtype::F32 => {
                let arr = TaggedArrayD::F32(ArcArray::from_elem(IxDyn(&dims), 0.0f32));
                TaggedTensor::F32([arr])
            }
            Dtype::U32 => {
                let arr = TaggedArrayD::U32(ArcArray::from_elem(IxDyn(&dims), 0u32));
                TaggedTensor::U32([arr])
            }
        }
    }

    fn ndarray_from_slice_f32(
        &self,
        data: &[f32],
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: Vec<usize> = shape.0;
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|_| BackendError::ShapeError)?;

        Ok(from_f32(arr))
    }

    fn ndarray_from_slice_u32(
        &self,
        data: &[u32],
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: Vec<usize> = shape.0;
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|_| BackendError::ShapeError)?;
        Ok(from_u32(arr))
    }

    fn arange(&self, end: usize) -> TaggedTensor<Self> {
        // TODO: use from_iter instead?
        let result = from_f32(ndarray::Array::range(0.0, end as f32, 1.0).into_dyn());
        self.cast(result, Dtype::U32)
    }

    fn cast(&self, x: TaggedTensor<Self>, target_dtype: Dtype) -> TaggedTensor<Self> {
        match (x, target_dtype) {
            (TaggedTensor::F32([arr]), Dtype::U32) => {
                let arr = arr.unwrap_f32();
                let data: Vec<u32> = arr.iter().map(|&val| val as u32).collect();
                let result = ArrayD::from_shape_vec(arr.raw_dim(), data).unwrap();
                from_u32(result)
            }
            (TaggedTensor::U32([arr]), Dtype::F32) => {
                let arr = arr.unwrap_u32();
                let data: Vec<f32> = arr.iter().map(|&val| val as f32).collect();
                let result = ArrayD::from_shape_vec(arr.raw_dim(), data).unwrap();
                from_f32(result)
            }
            (x @ TaggedTensor::F32(_), Dtype::F32) => x,
            (x @ TaggedTensor::U32(_), Dtype::U32) => x,
        }
    }

    fn matmul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => from_f32(Self::batched_matmul(x.unwrap_f32(), y.unwrap_f32())),
            U32([x, y]) => from_u32(Self::batched_matmul(x.unwrap_u32(), y.unwrap_u32())),
        }
    }

    fn add(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => from_f32(Self::add(x.unwrap_f32(), y.unwrap_f32())),
            U32([x, y]) => from_u32(Self::add(x.unwrap_u32(), y.unwrap_u32())),
        }
    }

    fn sub(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => from_f32(Self::sub(x.unwrap_f32(), y.unwrap_f32())),
            U32([x, y]) => from_u32(Self::sub(x.unwrap_u32(), y.unwrap_u32())),
        }
    }

    fn mul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => from_f32(Self::mul(x.unwrap_f32(), y.unwrap_f32())),
            U32([x, y]) => from_u32(Self::mul(x.unwrap_u32(), y.unwrap_u32())),
        }
    }

    fn div(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => from_f32(Self::div(x.unwrap_f32(), y.unwrap_f32())),
            U32([x, y]) => from_u32(Self::div(x.unwrap_u32(), y.unwrap_u32())),
        }
    }

    fn pow(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => from_f32(Self::pow_f32(x.unwrap_f32(), y.unwrap_f32())),
            U32([x, y]) => from_u32(Self::pow_u32(x.unwrap_u32(), y.unwrap_u32())),
        }
    }

    fn lt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => {
                let x = x.unwrap_f32();
                let y = y.unwrap_f32();
                let res = ndarray::Zip::from(&x).and(&y).map_collect(|&x, &y| x < y);
                from_f32(res.mapv(|x| x as u32 as f32))
            }
            U32([x, y]) => {
                let x = x.unwrap_u32();
                let y = y.unwrap_u32();
                let res = ndarray::Zip::from(&x).and(&y).map_collect(|&x, &y| x < y);
                from_u32(res.mapv(|x| x as u32))
            }
        }
    }

    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => {
                let x = x.unwrap_f32();
                let y = y.unwrap_f32();
                let res = ndarray::Zip::from(&x).and(&y).map_collect(|&x, &y| x == y);
                from_f32(res.mapv(|x| x as u32 as f32))
            }
            U32([x, y]) => {
                let x = x.unwrap_u32();
                let y = y.unwrap_u32();
                let res = ndarray::Zip::from(&x).and(&y).map_collect(|&x, &y| x == y);
                from_u32(res.mapv(|x| x as u32))
            }
        }
    }

    fn neg(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::neg_f32(arr.unwrap_f32())),
            U32([arr]) => from_u32(Self::neg_u32(arr.unwrap_u32())),
        }
    }

    fn sin(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(arr.unwrap_f32().sin()),
            _ => panic!("Invalid input types for sin"),
        }
    }

    fn cos(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(arr.unwrap_f32().cos()),
            _ => panic!("Invalid input types for cos"),
        }
    }

    fn max(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::max_f32(arr.unwrap_f32())),
            U32([arr]) => from_u32(Self::max_u32(arr.unwrap_u32())),
        }
    }

    fn sum(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::sum(arr.unwrap_f32())),
            U32([arr]) => from_u32(Self::sum(arr.unwrap_u32())),
        }
    }

    fn argmax(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_u32(Self::argmax_f32(arr.unwrap_f32())),
            U32([arr]) => from_u32(Self::argmax_u32(arr.unwrap_u32())),
        }
    }

    fn topk(&self, x: TaggedTensor<Self>, k: usize) -> (TaggedTensor<Self>, TaggedTensor<Self>) {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let (values, indices) = Self::topk_f32(arr.unwrap_f32(), k);
                (from_f32(values), from_u32(indices))
            }
            _ => panic!("Unsupported type for topk"),
        }
    }

    fn broadcast(&self, x: TaggedTensor<Self>, shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::broadcast_ndarray(arr.unwrap_f32(), shape)),
            U32([arr]) => from_u32(Self::broadcast_ndarray(arr.unwrap_u32(), shape)),
        }
    }

    fn transpose(&self, x: TaggedTensor<Self>, dim0: usize, dim1: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::transpose_ndarray(arr.unwrap_f32(), dim0, dim1)),
            U32([arr]) => from_u32(Self::transpose_ndarray(arr.unwrap_u32(), dim0, dim1)),
        }
    }

    fn index(
        &self,
        x: TaggedTensor<Self>,
        dim: usize,
        indices: TaggedTensor<Self>,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match (x, indices) {
            (F32([arr]), U32([indices])) => from_f32(Self::index_ndarray(
                arr.unwrap_f32(),
                dim,
                indices.unwrap_u32(),
            )),
            (U32([arr]), U32([indices])) => from_u32(Self::index_ndarray(
                arr.unwrap_u32(),
                dim,
                indices.unwrap_u32(),
            )),
            _ => panic!("Invalid input types for indexing"),
        }
    }

    fn slice(
        &self,
        x: TaggedTensor<Self>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::slice_ndarray(arr.unwrap_f32(), dim, start, len)),
            U32([arr]) => from_u32(Self::slice_ndarray(arr.unwrap_u32(), dim, start, len)),
        }
    }

    fn reshape(&self, x: TaggedTensor<Self>, new_shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => from_f32(Self::reshape_ndarray(arr.unwrap_f32(), new_shape)),
            U32([arr]) => from_u32(Self::reshape_ndarray(arr.unwrap_u32(), new_shape)),
        }
    }

    fn concat(
        &self,
        x: TaggedTensor<Self>,
        y: TaggedTensor<Self>,
        dim: usize,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match (x, y) {
            (F32([a]), F32([b])) => {
                from_f32(Self::concat_ndarray(a.unwrap_f32(), b.unwrap_f32(), dim))
            }
            (U32([a]), U32([b])) => {
                from_u32(Self::concat_ndarray(a.unwrap_u32(), b.unwrap_u32(), dim))
            }
            _ => panic!("Incompatible types for concatenation"),
        }
    }

    fn compare(&self, x: TaggedTensorTuple<Self, 2>) -> bool {
        use TaggedTensorTuple::*;
        match x {
            F32([a, b]) => a == b,
            U32([a, b]) => a == b,
        }
    }
}

impl NdArrayBackend {
    fn reshape_ndarray<D: Copy + Send + Sync + Debug>(
        arr: ArrayD<D>,
        new_shape: Shape,
    ) -> ArrayD<D> {
        let new_dims = ndarray::IxDyn(&new_shape.0);
        arr.to_shape(new_dims).unwrap().to_owned()
    }

    fn broadcast_ndarray<D: Copy + Send + Sync + Debug + Clone>(
        arr: ArrayD<D>,
        shape: Shape,
    ) -> ArrayD<D> {
        let broadcasted = arr.broadcast(ndarray::IxDyn(&shape.0)).unwrap();
        broadcasted.to_owned()
    }

    fn transpose_ndarray<D: Copy + Send + Sync + Debug>(
        arr: ArrayD<D>,
        dim0: usize,
        dim1: usize,
    ) -> ArrayD<D> {
        let mut res = arr.to_owned();
        res.swap_axes(dim0, dim1);
        res
    }

    fn index_ndarray<D: Copy + Send + Sync + Debug>(
        arr: ArrayD<D>,
        dim: usize,
        indices: ArrayD<u32>,
    ) -> ArrayD<D> {
        let idx = indices.iter().map(|&i| i as usize).collect::<Vec<_>>();

        arr.select(Axis(dim), &idx)
    }

    fn slice_ndarray<D: Copy + Send + Sync + Debug>(
        arr: ArrayD<D>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> ArrayD<D> {
        let r = arr.slice_axis(Axis(dim), (start..start + len).into());
        r.to_owned()
    }

    fn concat_ndarray<D: Copy + Send + Sync + Debug>(
        a: ArrayD<D>,
        b: ArrayD<D>,
        dim: usize,
    ) -> ArrayD<D> {
        ndarray::concatenate(Axis(dim), &[a.view(), b.view()]).unwrap()
    }

    fn add<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: ndarray::LinalgScalar,
    {
        // PERFORMANCE does ndarray reuse an x/y buffer if possible? If not, can we improve things
        // using in-place updates? That is, use `x += y` if x is contiguous.
        x + y
    }

    fn sub<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: ndarray::LinalgScalar,
    {
        // PERFORMANCE does ndarray reuse an x/y buffer if possible? If not, can we improve things
        // using in-place updates? That is, use `x -= y` if x is contiguous.
        x - y
    }

    fn mul<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: ndarray::LinalgScalar,
    {
        x * y
    }

    fn div<D>(x: ArrayD<D>, y: ArrayD<D>) -> ArrayD<D>
    where
        D: ndarray::LinalgScalar,
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
        x.fold_axis(Axis(axis), f32::MIN, |acc, &x| acc.max(x))
            .insert_axis(Axis(axis))
    }

    fn max_u32(x: ArrayD<u32>) -> ArrayD<u32> {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.fold_axis(Axis(axis), u32::MIN, |acc, x| *acc.max(x))
            .insert_axis(Axis(axis))
    }

    fn argmax_f32(x: ArrayD<f32>) -> ArrayD<u32> {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.map_axis(Axis(axis), |view| {
            view.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx as u32)
                .unwrap()
        })
        .insert_axis(Axis(axis))
    }

    fn argmax_u32(x: ArrayD<u32>) -> ArrayD<u32> {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.map_axis(Axis(axis), |view| {
            view.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.cmp(b))
                .map(|(idx, _)| idx as u32)
                .unwrap()
        })
        .insert_axis(Axis(axis))
    }

    fn topk_f32(x: ArrayD<f32>, k: usize) -> (ArrayD<f32>, ArrayD<u32>) {
        let mut dims = x.shape().to_vec();

        let last_idx = dims.len() - 1;
        let last_dim = dims[last_idx];

        let total_outer = x.len() / last_dim;
        let mut values: Vec<f32> = Vec::with_capacity(total_outer * k);
        let mut indices: Vec<u32> = Vec::with_capacity(total_outer * k);

        for lane in x.lanes(Axis(last_idx)) {
            let mut idxs: Vec<usize> = (0..lane.len()).collect();
            idxs.select_nth_unstable_by(k, |&i, &j| lane[j].total_cmp(&lane[i]));
            idxs.truncate(k);

            for i in idxs {
                values.push(lane[i]);
                indices.push(i as u32);
            }
        }

        dims[last_idx] = k;
        let values = ArrayD::from_shape_vec(IxDyn(&dims), values).unwrap();
        let indices = ArrayD::from_shape_vec(IxDyn(&dims), indices).unwrap();
        (values, indices)
    }

    fn sum<D>(x: ArrayD<D>) -> ArrayD<D>
    where
        D: ndarray::LinalgScalar,
    {
        // across the last dimension
        let axis = x.ndim() - 1;
        x.sum_axis(Axis(axis)).insert_axis(Axis(axis))
    }

    fn matmul_generic<D>(lhs: ArrayD<D>, rhs: ArrayD<D>) -> ArrayD<D>
    where
        D: ndarray::LinalgScalar,
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
        D: ndarray::LinalgScalar,
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

impl BackendTensorOps for TaggedArrayD {
    fn shape(&self) -> Shape {
        Shape(
            match self {
                TaggedArrayD::F32(x) => x.shape(),
                TaggedArrayD::U32(x) => x.shape(),
            }
            .to_vec(),
        )
    }
}

#[test]
fn test_batched_matmul() {
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
    // Test summing across last dimension: [2, 3] -> [2]
    let x_data = vec![1u32, 2, 3, 4, 5, 6];
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_data).unwrap();

    let result = NdArrayBackend::sum(x);

    // Expected: [1+2+3, 4+5+6] = [[6], [15]]
    let expected = [6u32, 15];
    assert_eq!(result.shape(), &[2, 1]);

    let result_flat = result.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_flat.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    // Test with 3D tensor: [2, 2, 3] -> [2, 2, 1]
    let x_data_3d = vec![
        1.0f32, 2.0, 3.0, // [0,0,:]
        4.0, 5.0, 6.0, // [0,1,:]
        7.0, 8.0, 9.0, // [1,0,:]
        10.0, 11.0, 12.0, // [1,1,:]
    ];
    let x_3d = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2, 3]), x_data_3d).unwrap();

    let result_3d = NdArrayBackend::sum(x_3d);

    // Expected: [[1+2+3, 4+5+6], [7+8+9, 10+11+12]] = [[6], [15], [24], [33]]
    let expected_3d = [6.0f32, 15.0, 24.0, 33.0];
    assert_eq!(result_3d.shape(), &[2, 2, 1]);

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
    // Test max across last dimension: [2, 3] -> [2, 1]
    let x_data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0];
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_data).unwrap();

    let result = NdArrayBackend::max_f32(x);

    // Expected: [max(1,5,3), max(2,8,4)] = [[5], [8]]
    let expected = [5.0f32, 8.0];
    assert_eq!(result.shape(), &[2, 1]);

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

    // Expected: [max(1,5), max(3,2)] = [[5], [3]]
    let expected_u32 = [5u32, 3];
    assert_eq!(result_u32.shape(), &[2, 1]);

    let result_u32_flat = result_u32.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_u32_flat.iter().zip(expected_u32.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }

    // Test with 3D tensor: [2, 2, 3] -> [2, 2, 1]
    let x_data_3d = vec![
        1.0f32, 2.0, 3.0, // [0,0,:]
        4.0, 5.0, 6.0, // [0,1,:]
        7.0, 8.0, 9.0, // [1,0,:]
        10.0, 11.0, 12.0, // [1,1,:]
    ];
    let x_3d = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2, 3]), x_data_3d).unwrap();

    let result_3d = NdArrayBackend::max_f32(x_3d);

    // Expected: [[max(1,2,3), max(4,5,6)], [max(7,8,9), max(10,11,12)]] = [[[3], [6]], [[9], [12]]]
    let expected_3d = [3.0f32, 6.0, 9.0, 12.0];
    assert_eq!(result_3d.shape(), &[2, 2, 1]);

    let result_3d_flat = result_3d.as_slice().unwrap();
    for (i, (&actual, &expected)) in result_3d_flat.iter().zip(expected_3d.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}
