use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, BackendTensorOps};
use candle_core::{
    D, DType, Device, Tensor,
    utils::{cuda_is_available, metal_is_available},
};

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

    pub fn new_accel(accel: bool) -> Self {
        let device = if !accel {
            Device::Cpu
        } else if cuda_is_available() {
            Device::new_cuda(0).unwrap()
        } else if metal_is_available() {
            Device::new_metal(0).unwrap()
        } else {
            Device::Cpu
        };
        Self { device }
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
    type BackendTensor = CandleTensor;

    fn to_vec(&self, vec: TaggedTensor<Self>) -> TaggedVec {
        match vec {
            TaggedTensor::F32([x]) => TaggedVec::F32(x.0.flatten_all().unwrap().to_vec1().unwrap()),
            TaggedTensor::U32([x]) => TaggedVec::U32(x.0.flatten_all().unwrap().to_vec1().unwrap()),
        }
    }

    fn format_tensor(&self, tensor: &TaggedTensor<Self>) -> String {
        match tensor {
            TaggedTensor::F32([x]) => format!("{}", &x.0),
            TaggedTensor::U32([x]) => format!("{}", &x.0),
        }
    }

    fn zeros(&self, shape: Shape, target_dtype: Dtype) -> TaggedTensor<Self> {
        let dims: &[usize] = &shape.0;
        match target_dtype {
            Dtype::F32 => {
                let tensor = Tensor::zeros(dims, DType::F32, &self.device).unwrap();
                TaggedTensor::F32([CandleTensor(tensor)])
            }
            Dtype::U32 => {
                let tensor = Tensor::zeros(dims, DType::U32, &self.device).unwrap();
                TaggedTensor::U32([CandleTensor(tensor)])
            }
        }
    }

    fn ndarray_from_vec_f32(
        &self,
        data: Vec<f32>,
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: &[usize] = &shape.0;
        let tensor =
            Tensor::from_vec(data, dims, &self.device).map_err(|_| BackendError::ShapeError)?;
        Ok(TaggedTensor::F32([CandleTensor(tensor)]))
    }

    fn ndarray_from_vec_u32(
        &self,
        data: Vec<u32>,
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: &[usize] = &shape.0;
        let tensor =
            Tensor::from_vec(data, dims, &self.device).map_err(|_| BackendError::ShapeError)?;
        Ok(TaggedTensor::U32([CandleTensor(tensor)]))
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

    fn gt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::gt(x, y)]),
            U32([x, y]) => U32([Self::gt(x, y)]),
        }
    }

    fn gte(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::gte(x, y)]),
            U32([x, y]) => U32([Self::gte(x, y)]),
        }
    }

    fn lte(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::lte(x, y)]),
            U32([x, y]) => U32([Self::lte(x, y)]),
        }
    }

    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::eq(x, y)]),
            U32([x, y]) => U32([Self::eq(x, y)]),
        }
    }

    fn where_cond(&self, args: TaggedTensorTuple<Self, 3>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match args {
            F32([mask, x, y]) => F32([Self::where_cond(mask, x, y)]),
            U32([mask, x, y]) => U32([Self::where_cond(mask, x, y)]),
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

    fn log(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::log(arr)]),
            _ => panic!("Invalid type for log"),
        }
    }

    fn floor(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::floor(arr)]),
            _ => panic!("Invalid type for floor"),
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

    fn topk(&self, x: TaggedTensor<Self>, k: usize) -> (TaggedTensor<Self>, TaggedTensor<Self>) {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let (values, indices) = Self::topk_f32(arr.0, k);
                (F32([CandleTensor(values)]), U32([CandleTensor(indices)]))
            }
            _ => panic!("Unsupported type for topk"),
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

    fn to_bool(&self, x: TaggedTensor<Self>) -> bool {
        match x {
            TaggedTensor::F32([x]) => {
                x.0.gt(0.0)
                    .ok()
                    .and_then(|t| t.max_all().ok())
                    .and_then(|m| m.to_scalar::<u8>().ok())
                    .map(|s| s == 1)
                    .unwrap_or(false)
            }
            TaggedTensor::U32([x]) => {
                x.0.ne(0u32)
                    .ok()
                    .and_then(|t| t.max_all().ok())
                    .and_then(|m| m.to_scalar::<u8>().ok())
                    .map(|s| s == 1)
                    .unwrap_or(false)
            }
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
        let dims_s = tensor.dims();
        let dims_t = new_shape.0.clone();

        // This is the special but very common case of [A, B] -> [1, A, B]
        // Use unsqueeze() since it does not copy, whereas Candle reshape()
        // copies when the source is not contiguous
        if dims_t[0] == 1 && dims_t[1..] == *dims_s {
            tensor.unsqueeze(0).unwrap()
        } else {
            tensor.reshape(&*new_shape.0).unwrap()
        }
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
        let dtype = x.0.dtype();
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor(x.0.lt(&y.0).unwrap().to_dtype(dtype).unwrap())
    }

    fn gt(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        let dtype = x.0.dtype();
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor(x.0.gt(&y.0).unwrap().to_dtype(dtype).unwrap())
    }

    fn gte(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        let dtype = x.0.dtype();
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor(x.0.ge(&y.0).unwrap().to_dtype(dtype).unwrap())
    }

    fn lte(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        let dtype = x.0.dtype();
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor(x.0.le(&y.0).unwrap().to_dtype(dtype).unwrap())
    }

    fn eq(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        CandleTensor((x.0.eq(&y.0)).unwrap())
    }

    fn where_cond(mask: CandleTensor, x: CandleTensor, y: CandleTensor) -> CandleTensor {
        let mask = match mask.0.dtype() {
            DType::F32 => mask.0.gt(0.).unwrap(),
            DType::U32 => mask.0.ne(0u32).unwrap(),
            _ => mask.0, // already U8 (boolean) or other type
        };

        CandleTensor(mask.where_cond(&x.0, &y.0).unwrap())
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

    fn log(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.log().unwrap())
    }

    fn floor(x: CandleTensor) -> CandleTensor {
        CandleTensor(x.0.floor().unwrap())
    }

    // Candle's pow function does not support negative base and silently generates NaNs
    // so we do element-wise powf https://github.com/huggingface/candle/issues/1640
    fn pow(x: CandleTensor, y: CandleTensor) -> CandleTensor {
        if x.0.dims() != y.0.dims() {
            panic!("Shape mismatch in operation");
        }

        // Convert tensors to vectors for element-wise powf operation
        let x_vec: Vec<f32> = x.0.flatten_all().unwrap().to_vec1().unwrap();
        let y_vec: Vec<f32> = y.0.flatten_all().unwrap().to_vec1().unwrap();

        // Perform element-wise powf
        let result_vec: Vec<f32> = x_vec
            .iter()
            .zip(y_vec.iter())
            .map(|(a, b)| a.powf(*b))
            .collect();

        // Convert back to tensor with original shape
        let shape = x.0.dims();
        let result_tensor = Tensor::from_vec(result_vec, shape, x.0.device()).unwrap();
        CandleTensor(result_tensor)
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

    fn topk_f32(tensor: Tensor, k: usize) -> (Tensor, Tensor) {
        let (values, indices) = tensor.sort_last_dim(false).unwrap();
        let topk_indices = indices.narrow(D::Minus1, 0, k).unwrap();
        let topk_values = values.narrow(D::Minus1, 0, k).unwrap();
        (topk_values, topk_indices)
    }

    pub fn batched_matmul(lhs: Tensor, rhs: Tensor) -> Tensor {
        match lhs.matmul(&rhs) {
            Ok(result) => result,
            // On error retry with contiguous inputs
            Err(_) => {
                let lhs = lhs.contiguous().unwrap();
                let rhs = rhs.contiguous().unwrap();
                lhs.matmul(&rhs).unwrap()
            }
        }
    }
}

impl BackendTensorOps for CandleTensor {
    fn shape(&self) -> Shape {
        Shape(self.0.dims().to_vec())
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
