use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, NdArray};
use candle_core::{D, DType, Device, Tensor};

// ============================================================================
// CANDLE BACKEND ARCHITECTURE EXPLANATION
// ============================================================================
//
// This backend follows a 2-layer architecture pattern common in Rust:
//
// 1. **`CandleTensor` - The Data Container**
//    - Wrapper around `candle_core::Tensor`
//    - Implements the `NdArray<D>` trait (required by the Backend trait)
//    - Provides type safety and API consistency with other backends
//
// 2. **`CandleBackend` - The Operations Provider**
//    - Manages device state (CPU, GPU, Metal, etc.)
//    - Implements the `Backend` trait with all operations (add, mul, matmul, etc.)
//
// ARCHITECTURE PATTERN:
//
// Backend (Operations) ──→ Tensor (Data)
//      ↓                      ↓
// CandleBackend         CandleTensor
//   - device              - Tensor
//   - configuration       - NdArray trait
//
// ============================================================================

#[derive(Clone, Debug)]
pub struct CandleTensor(pub Tensor);

#[derive(Clone, Debug)]
pub struct CandleBackend {
    device: Device,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    pub fn with_device(device: Device) -> Self {
        Self { device }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CandleBackend {
    type NdArray<D: HasDtype> = CandleTensor;

    fn scalar<D: HasDtype>(&self, d: D) -> Self::NdArray<D> {
        // Use unsafe transmute as a workaround for type erasure
        // This is not ideal but necessary due to trait constraints
        // (Candle's Tensor::new() requires knowing the specific
        // data type at compile time)
        // TODO: Issue #188 refactors the backend to fix this.
        if std::mem::size_of::<D>() == std::mem::size_of::<f32>() {
            let val = unsafe { std::mem::transmute_copy::<D, f32>(&d) };
            // Create a true scalar with shape [] by using Tensor::from_slice with empty shape
            CandleTensor(Tensor::from_slice(&[val], (), &self.device).unwrap())
        } else if std::mem::size_of::<D>() == std::mem::size_of::<u32>() {
            let val = unsafe { std::mem::transmute_copy::<D, u32>(&d) };
            // Create a true scalar with shape [] by using Tensor::from_slice with empty shape
            CandleTensor(Tensor::from_slice(&[val], (), &self.device).unwrap())
        } else {
            panic!("Unsupported dtype for scalar creation");
        }
    }

    fn zeros<D: HasDtype + Default>(&self, shape: Shape) -> Self::NdArray<D> {
        let dims: &[usize] = &shape.0;
        if std::mem::size_of::<D>() == std::mem::size_of::<f32>() {
            CandleTensor(Tensor::zeros(dims, DType::F32, &self.device).unwrap())
        } else if std::mem::size_of::<D>() == std::mem::size_of::<u32>() {
            CandleTensor(Tensor::zeros(dims, DType::U32, &self.device).unwrap())
        } else {
            panic!("Unsupported dtype for zeros creation");
        }
    }

    fn ndarray_from_slice<D: HasDtype>(
        &self,
        data: &[D],
        shape: Shape,
    ) -> Result<Self::NdArray<D>, BackendError> {
        let dims: &[usize] = &shape.0;
        if std::mem::size_of::<D>() == std::mem::size_of::<f32>() {
            let data_f32: &[f32] = unsafe { std::mem::transmute(data) };
            let tensor = Tensor::new(data_f32, &self.device)
                .map_err(|_| BackendError::ShapeError)?
                .reshape(dims)
                .map_err(|_| BackendError::ShapeError)?;
            Ok(CandleTensor(tensor))
        } else if std::mem::size_of::<D>() == std::mem::size_of::<u32>() {
            let data_u32: &[u32] = unsafe { std::mem::transmute(data) };
            let tensor = Tensor::new(data_u32, &self.device)
                .map_err(|_| BackendError::ShapeError)?
                .reshape(dims)
                .map_err(|_| BackendError::ShapeError)?;
            Ok(CandleTensor(tensor))
        } else {
            panic!("Unsupported dtype for slice creation");
        }
    }

    fn cast(&self, x: TaggedTensor<Self>, target_dtype: Dtype) -> TaggedTensor<Self> {
        match (&x, target_dtype) {
            (TaggedTensor::F32(arr), Dtype::U32) => {
                let result = arr[0].0.to_dtype(DType::U32).unwrap();
                TaggedTensor::U32([CandleTensor(result)])
            }
            (TaggedTensor::U32(arr), Dtype::F32) => {
                let result = arr[0].0.to_dtype(DType::F32).unwrap();
                TaggedTensor::F32([CandleTensor(result)])
            }
            (TaggedTensor::F32(_), Dtype::F32) => x,
            (TaggedTensor::U32(_), Dtype::U32) => x,
        }
    }

    fn matmul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([CandleTensor(Self::batched_matmul(x.0, y.0))]),
            U32([x, y]) => U32([CandleTensor(Self::batched_matmul(x.0, y.0))]),
        }
    }

    fn add(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::add(x, y)]),
            U32([x, y]) => U32([Self::add(x, y)]),
        }
    }

    fn sub(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::sub(x, y)]),
            U32([x, y]) => U32([Self::sub(x, y)]),
        }
    }

    fn mul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::mul(x, y)]),
            U32([x, y]) => U32([Self::mul(x, y)]),
        }
    }

    fn div(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::div(x, y)]),
            U32([x, y]) => U32([Self::div(x, y)]),
        }
    }

    fn lt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::lt(x, y)]),
            U32([x, y]) => U32([Self::lt(x, y)]),
        }
    }

    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::eq(x, y)]),
            U32([x, y]) => U32([Self::eq(x, y)]),
        }
    }

    fn pow(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::pow(x, y)]),
            U32([x, y]) => U32([Self::pow(x, y)]),
        }
    }

    fn neg(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::neg(arr)]),
            U32([arr]) => U32([Self::neg(arr)]),
        }
    }

    fn sin(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::sin(arr)]),
            _ => panic!("Invalid type for sin"),
        }
    }

    fn cos(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::cos(arr)]),
            _ => panic!("Invalid type for cos"),
        }
    }

    fn max(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::max(arr)]),
            U32([arr]) => U32([Self::max(arr)]),
        }
    }

    fn sum(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::sum(arr)]),
            U32([arr]) => U32([Self::sum(arr)]),
        }
    }

    fn argmax(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => U32([Self::argmax(arr)]),
            U32([arr]) => U32([Self::argmax(arr)]),
        }
    }

    fn broadcast(&self, x: TaggedTensor<Self>, shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([CandleTensor(Self::broadcast_tensor(arr.0, shape))]),
            U32([arr]) => U32([CandleTensor(Self::broadcast_tensor(arr.0, shape))]),
        }
    }

    fn reshape(&self, x: TaggedTensor<Self>, new_shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([CandleTensor(Self::reshape_tensor(arr.0, new_shape))]),
            U32([arr]) => U32([CandleTensor(Self::reshape_tensor(arr.0, new_shape))]),
        }
    }

    fn transpose(&self, x: TaggedTensor<Self>, dim0: usize, dim1: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([CandleTensor(Self::transpose_tensor(arr.0, dim0, dim1))]),
            U32([arr]) => U32([CandleTensor(Self::transpose_tensor(arr.0, dim0, dim1))]),
        }
    }
    fn arange(&self, end: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        let r = Tensor::arange(0, end as u32, &self.device).unwrap();
        U32([CandleTensor(r)])
    }

    fn index(
        &self,
        x: TaggedTensor<Self>,
        dim: usize,
        indices: TaggedTensor<Self>,
    ) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match (x, indices) {
            (F32([arr]), U32([indices])) => {
                F32([CandleTensor(Self::index_tensor(arr.0, dim, indices.0))])
            }
            (U32([arr]), U32([indices])) => {
                U32([CandleTensor(Self::index_tensor(arr.0, dim, indices.0))])
            }
            _ => panic!("Invalid index type"),
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
            F32([arr]) => F32([CandleTensor(Self::slice_tensor(arr.0, dim, start, len))]),
            U32([arr]) => U32([CandleTensor(Self::slice_tensor(arr.0, dim, start, len))]),
        }
    }

    fn compare(&self, x: TaggedTensorTuple<Self, 2>) -> bool {
        use TaggedTensorTuple::*;
        match x {
            F32([a, b]) => Self::compare_tensors(&a.0, &b.0),
            U32([a, b]) => Self::compare_tensors(&a.0, &b.0),
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
            (F32([a]), F32([b])) => F32([CandleTensor(Self::concat_tensors(&a.0, &b.0, dim))]),
            (U32([a]), U32([b])) => U32([CandleTensor(Self::concat_tensors(&a.0, &b.0, dim))]),
            _ => panic!("Incompatible types for concatenation"),
        }
    }
}

impl CandleBackend {
    // ============================================================================
    //              TENSOR COMPARISON: CANDLE vs NDARRAY DESIGN DIFFERENCES
    // ============================================================================
    //
    // **NDARRAY:**
    // - CPU-only, Rust-native data structures
    // - `a == b` automatically handles shape checking + element-wise comparison
    //
    // **CANDLE:**
    // - GPU/CPU computation with device memory management
    // - Explicit error handling (GPU operations can fail)
    // - Element-wise operations return tensors, not scalars
    //
    // **Why Candle's approach:**
    // 1. `.eq()` returns Result<Tensor, Error> (not Result<bool, Error>)
    // 2. Returns U8 boolean tensor where 1=equal, 0=not equal
    // 3. Need `min_all()` to check if ALL elements are true (equal)
    // 4. Must handle device errors explicitly
    // 5. More efficient than converting to Vec for comparison
    //
    // ============================================================================

    fn compare_tensors(a: &Tensor, b: &Tensor) -> bool {
        if a.dims() != b.dims() {
            return false;
        }

        a.eq(b)
            .ok()
            .and_then(|eq_tensor| eq_tensor.min_all().ok())
            .and_then(|min_val| min_val.to_scalar::<u8>().ok())
            .map(|min_scalar| min_scalar == 1)
            .unwrap_or(false)
    }

    fn concat_tensors(a: &Tensor, b: &Tensor, dim: usize) -> Tensor {
        Tensor::cat(&[a, b], dim).unwrap()
    }

    fn reshape_tensor(tensor: Tensor, new_shape: Shape) -> Tensor {
        tensor.reshape(&*new_shape.0).unwrap()
    }

    fn transpose_tensor(tensor: Tensor, dim0: usize, dim1: usize) -> Tensor {
        tensor.transpose(dim0, dim1).unwrap()
    }

    fn index_tensor(tensor: Tensor, dim: usize, indices: Tensor) -> Tensor {
        let idx = indices.flatten_all().unwrap();
        tensor.index_select(&idx, dim).unwrap()
    }

    fn slice_tensor(tensor: Tensor, dim: usize, start: usize, len: usize) -> Tensor {
        tensor.narrow(dim, start, len).unwrap()
    }

    fn broadcast_tensor(tensor: Tensor, shape: Shape) -> Tensor {
        tensor.broadcast_as(shape.0).unwrap()
    }

    fn add(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }
        CandleTensor((&x.0 + &y.0).unwrap())
    }

    fn sub(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }
        CandleTensor((&x.0 - &y.0).unwrap())
    }

    fn mul(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }
        CandleTensor((&x.0 * &y.0).unwrap())
    }

    fn div(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }
        CandleTensor((&x.0 / &y.0).unwrap())
    }

    fn lt(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor((x.0.lt(&y.0)).unwrap())
    }

    fn eq(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor((x.0.eq(&y.0)).unwrap())
    }

    fn neg(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.neg().unwrap())
    }

    fn sin(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.sin().unwrap())
    }

    fn cos(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.cos().unwrap())
    }

    fn pow(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }
        CandleTensor(x.0.pow(&y.0).unwrap())
    }

    fn sum(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.sum_keepdim(D::Minus1).unwrap())
    }

    fn max(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.max_keepdim(D::Minus1).unwrap())
    }

    fn argmax(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.argmax_keepdim(D::Minus1).unwrap())
    }

    fn matmul_generic(lhs: Tensor, rhs: Tensor) -> Tensor {
        // For now, only handle rank 2 case
        assert_eq!(lhs.dims().len(), 2, "matmul: lhs must be rank 2");
        assert_eq!(rhs.dims().len(), 2, "matmul: rhs must be rank 2");

        lhs.matmul(&rhs).unwrap()
    }

    pub fn batched_matmul(lhs: Tensor, rhs: Tensor) -> Tensor {
        // For now, handle the simple 2D case
        if lhs.dims().len() == 2 && rhs.dims().len() == 2 {
            return Self::matmul_generic(lhs, rhs);
        }

        // Same as ndarray: require exact batch dimension match, panic if not
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();

        if lhs_dims.len() >= 2 && rhs_dims.len() >= 2 {
            // Get batch dimensions and matrix dimensions
            let lhs_batch_dims = &lhs_dims[..lhs_dims.len() - 2];
            let rhs_batch_dims = &rhs_dims[..rhs_dims.len() - 2];
            let lhs_matrix_dims = &lhs_dims[lhs_dims.len() - 2..];
            let rhs_matrix_dims = &rhs_dims[rhs_dims.len() - 2..];

            // Check matrix dimensions compatibility
            assert_eq!(
                lhs_matrix_dims[1], rhs_matrix_dims[0],
                "batched_matmul: incompatible matrix dimensions"
            );

            // Require batch dimensions to match exactly (same as ndarray)
            assert_eq!(
                lhs_batch_dims, rhs_batch_dims,
                "batched_matmul: batch dimensions must match exactly"
            );

            let batch_size: usize = lhs_batch_dims.iter().product();
            let lhs_m = lhs_matrix_dims[0];
            let lhs_k = lhs_matrix_dims[1];
            let rhs_k = rhs_matrix_dims[0];
            let rhs_n = rhs_matrix_dims[1];

            // Reshape to (batch_size, m, k) and (batch_size, k, n)
            let lhs_reshaped = lhs.reshape(&[batch_size, lhs_m, lhs_k]).unwrap();
            let rhs_reshaped = rhs.reshape(&[batch_size, rhs_k, rhs_n]).unwrap();

            // Perform batched matrix multiplication
            let mut results = Vec::new();
            for b in 0..batch_size {
                let lhs_batch = lhs_reshaped.get(b).unwrap();
                let rhs_batch = rhs_reshaped.get(b).unwrap();
                let batch_result = lhs_batch.matmul(&rhs_batch).unwrap();
                results.push(batch_result);
            }

            // Concatenate results and reshape back to proper batch shape
            let mut result_shape = lhs_batch_dims.to_vec();
            result_shape.push(lhs_m);
            result_shape.push(rhs_n);

            Tensor::cat(&results, 0)
                .unwrap()
                .reshape(&*result_shape)
                .unwrap()
        } else {
            // Fallback to regular matmul
            lhs.matmul(&rhs).unwrap()
        }
    }
}

impl<D: HasDtype> NdArray<D> for CandleTensor {
    type Backend = CandleBackend;

    fn shape(&self) -> Shape {
        Shape(self.0.dims().to_vec())
    }
}

#[test]
fn test_batched_matmul() {
    use candle_core::Tensor;

    // Test with 2 batch dimensions: [2, 3, 2, 2] × [2, 3, 2, 1] = [2, 3, 2, 1]
    let lhs_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // batch 0,0
        5.0, 6.0, 7.0, 8.0, // batch 0,1
        9.0, 10.0, 11.0, 12.0, // batch 0,2
        13.0, 14.0, 15.0, 16.0, // batch 1,0
        17.0, 18.0, 19.0, 20.0, // batch 1,1
        21.0, 22.0, 23.0, 24.0, // batch 1,2
    ];
    let lhs = Tensor::new(&*lhs_data, &candle_core::Device::Cpu)
        .unwrap()
        .reshape(&[2, 3, 2, 2])
        .unwrap();

    let rhs_data = vec![
        1.0f32, 2.0, // batch 0,0
        3.0, 4.0, // batch 0,1
        5.0, 6.0, // batch 0,2
        7.0, 8.0, // batch 1,0
        9.0, 10.0, // batch 1,1
        11.0, 12.0, // batch 1,2
    ];
    let rhs = Tensor::new(&*rhs_data, &candle_core::Device::Cpu)
        .unwrap()
        .reshape(&[2, 3, 2, 1])
        .unwrap();

    let result = CandleBackend::batched_matmul(lhs, rhs);

    // Expected shape: [2, 3, 2, 1]
    assert_eq!(result.dims(), &[2, 3, 2, 1]);

    let expected = [
        5.0f32, 11.0, // batch 0,0
        39.0, 53.0, // batch 0,1
        105.0, 127.0, // batch 0,2
        203.0, 233.0, // batch 1,0
        333.0, 371.0, // batch 1,1
        495.0, 541.0, // batch 1,2
    ];

    // Flatten the result to compare with expected values
    let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
    for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Mismatch at index {i}: got {actual}, expected {expected}"
        );
    }
}
