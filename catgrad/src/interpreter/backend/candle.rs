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
struct DeferredIndex0 {
    indices: Tensor,
    dim: usize,
}

#[derive(Clone, Debug)]
pub struct CandleTensor {
    tensor: Tensor,
    deferred_index0: Option<DeferredIndex0>,
}

impl CandleTensor {
    fn from_materialized(tensor: Tensor) -> Self {
        Self {
            tensor,
            deferred_index0: None,
        }
    }

    fn from_indexed_select(tensor: &Tensor, dim: usize, indices: &Tensor) -> Self {
        Self {
            tensor: tensor.clone(),
            deferred_index0: Some(DeferredIndex0 {
                indices: indices.flatten_all().unwrap(),
                dim,
            }),
        }
    }

    pub fn materialize(&self) -> Tensor {
        match &self.deferred_index0 {
            None => self.tensor.clone(),
            Some(DeferredIndex0 { indices, dim }) => {
                CandleBackend::index_tensor_materialized(&self.tensor, *dim, indices)
            }
        }
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let tensor = self.tensor.transpose(dim0, dim1).unwrap();
        let deferred_index0 =
            self.deferred_index0
                .as_ref()
                .map(|DeferredIndex0 { indices, dim }| DeferredIndex0 {
                    indices: indices.clone(),
                    dim: if *dim == dim0 {
                        dim1
                    } else if *dim == dim1 {
                        dim0
                    } else {
                        *dim
                    },
                });
        Self {
            tensor,
            deferred_index0,
        }
    }

    fn slice(&self, dim: usize, start: usize, len: usize) -> Self {
        match &self.deferred_index0 {
            None => Self::from_materialized(CandleBackend::slice_tensor_materialized(
                &self.tensor,
                dim,
                start,
                len,
            )),
            Some(DeferredIndex0 {
                indices,
                dim: indexed_dim,
            }) if dim == *indexed_dim => Self {
                tensor: self.tensor.clone(),
                deferred_index0: Some(DeferredIndex0 {
                    indices: indices.narrow(0, start, len).unwrap(),
                    dim: *indexed_dim,
                }),
            },
            Some(DeferredIndex0 {
                indices,
                dim: indexed_dim,
            }) => Self {
                tensor: self.tensor.narrow(dim, start, len).unwrap(),
                deferred_index0: Some(DeferredIndex0 {
                    indices: indices.clone(),
                    dim: *indexed_dim,
                }),
            },
        }
    }

    fn shape(&self) -> Shape {
        match &self.deferred_index0 {
            None => Shape(self.tensor.dims().to_vec()),
            Some(DeferredIndex0 { indices, dim }) => {
                let mut dims = self.tensor.dims().to_vec();
                dims[*dim] = indices.dims1().unwrap();
                Shape(dims)
            }
        }
    }
}

impl From<Tensor> for CandleTensor {
    fn from(value: Tensor) -> Self {
        Self::from_materialized(value)
    }
}

#[derive(Clone, Debug)]
pub struct CandleBackend {
    device: Device,
}

impl CandleBackend {
    fn ensure_dtype_supported(&self, dtype: DType) {
        if dtype == DType::BF16 && !self.device.supports_bf16() {
            panic!("BF16 is only supported by Candle on CUDA/Metal devices");
        }
    }

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
            TaggedTensor::F32([x]) => {
                let x = x.materialize();
                TaggedVec::F32(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::F16([x]) => {
                let x = x.materialize();
                TaggedVec::F16(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::BF16([x]) => {
                let x = x.materialize();
                TaggedVec::BF16(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::U32([x]) => {
                let x = x.materialize();
                TaggedVec::U32(x.flatten_all().unwrap().to_vec1().unwrap())
            }
        }
    }

    fn format_tensor(&self, tensor: &TaggedTensor<Self>) -> String {
        match tensor {
            TaggedTensor::F32([x]) => format!("{}", x.materialize()),
            TaggedTensor::F16([x]) => format!("{}", x.materialize()),
            TaggedTensor::BF16([x]) => format!("{}", x.materialize()),
            TaggedTensor::U32([x]) => format!("{}", x.materialize()),
        }
    }

    fn zeros(&self, shape: Shape, target_dtype: Dtype) -> TaggedTensor<Self> {
        let dims: &[usize] = &shape.0;
        match target_dtype {
            Dtype::F32 => {
                let tensor = Tensor::zeros(dims, DType::F32, &self.device).unwrap();
                TaggedTensor::F32([tensor.into()])
            }
            Dtype::F16 => {
                let tensor = Tensor::zeros(dims, DType::F16, &self.device).unwrap();
                TaggedTensor::F16([tensor.into()])
            }
            Dtype::BF16 => {
                self.ensure_dtype_supported(DType::BF16);
                let tensor = Tensor::zeros(dims, DType::BF16, &self.device).unwrap();
                TaggedTensor::BF16([tensor.into()])
            }
            Dtype::U32 => {
                let tensor = Tensor::zeros(dims, DType::U32, &self.device).unwrap();
                TaggedTensor::U32([tensor.into()])
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
        Ok(TaggedTensor::F32([tensor.into()]))
    }

    fn ndarray_from_vec_f16(
        &self,
        data: Vec<half::f16>,
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: &[usize] = &shape.0;
        let tensor =
            Tensor::from_vec(data, dims, &self.device).map_err(|_| BackendError::ShapeError)?;
        Ok(TaggedTensor::F16([tensor.into()]))
    }

    fn ndarray_from_vec_bf16(
        &self,
        data: Vec<half::bf16>,
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        self.ensure_dtype_supported(DType::BF16);
        let dims: &[usize] = &shape.0;
        let tensor =
            Tensor::from_vec(data, dims, &self.device).map_err(|_| BackendError::ShapeError)?;
        Ok(TaggedTensor::BF16([tensor.into()]))
    }

    fn ndarray_from_vec_u32(
        &self,
        data: Vec<u32>,
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: &[usize] = &shape.0;
        let tensor =
            Tensor::from_vec(data, dims, &self.device).map_err(|_| BackendError::ShapeError)?;
        Ok(TaggedTensor::U32([tensor.into()]))
    }

    fn cast(&self, x: TaggedTensor<Self>, target_dtype: Dtype) -> TaggedTensor<Self> {
        if x.dtype() == target_dtype {
            return x;
        }

        let tensor = match x {
            TaggedTensor::F32([arr])
            | TaggedTensor::F16([arr])
            | TaggedTensor::BF16([arr])
            | TaggedTensor::U32([arr]) => arr.materialize(),
        };
        let target_dtype = match target_dtype {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::U32 => DType::U32,
        };
        self.ensure_dtype_supported(target_dtype);
        let tensor: CandleTensor = tensor.to_dtype(target_dtype).unwrap().into();
        match target_dtype {
            DType::F32 => TaggedTensor::F32([tensor]),
            DType::F16 => TaggedTensor::F16([tensor]),
            DType::BF16 => TaggedTensor::BF16([tensor]),
            DType::U32 => TaggedTensor::U32([tensor]),
            _ => unreachable!(),
        }
    }

    fn matmul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::matmul_tensors(x, y)]),
            F16([x, y]) => F16([Self::matmul_tensors(x, y)]),
            BF16([x, y]) => BF16([Self::matmul_tensors(x, y)]),
            U32([x, y]) => U32([Self::matmul_tensors(x, y)]),
        }
    }

    fn add(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::add)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::add)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::add)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::add)]),
        }
    }

    fn sub(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::sub)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::sub)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::sub)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::sub)]),
        }
    }

    fn mul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::mul)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::mul)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::mul)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::mul)]),
        }
    }

    fn div(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::div)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::div)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::div)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::div)]),
        }
    }

    fn lt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::lt)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::lt)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::lt)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::lt)]),
        }
    }

    fn gt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::gt)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::gt)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::gt)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::gt)]),
        }
    }

    fn gte(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::gte)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::gte)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::gte)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::gte)]),
        }
    }

    fn lte(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::lte)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::lte)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::lte)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::lte)]),
        }
    }

    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::eq)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::eq)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::eq)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::eq)]),
        }
    }

    fn where_cond(&self, args: TaggedTensorTuple<Self, 3>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match args {
            F32([mask, x, y]) => F32([Self::ternary_eager(mask, x, y, Self::where_cond)]),
            F16([mask, x, y]) => F16([Self::ternary_eager(mask, x, y, Self::where_cond)]),
            BF16([mask, x, y]) => BF16([Self::ternary_eager(mask, x, y, Self::where_cond)]),
            U32([mask, x, y]) => U32([Self::ternary_eager(mask, x, y, Self::where_cond)]),
        }
    }

    fn pow(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, Self::pow)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, Self::pow)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, Self::pow)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, Self::pow)]),
        }
    }

    fn neg(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::neg)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::neg)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::neg)]),
            U32([arr]) => U32([Self::unary_eager(arr, Self::neg)]),
        }
    }

    fn sin(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::sin)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::sin)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::sin)]),
            _ => panic!("Invalid type for sin"),
        }
    }

    fn cos(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::cos)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::cos)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::cos)]),
            _ => panic!("Invalid type for cos"),
        }
    }

    fn log(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::log)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::log)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::log)]),
            _ => panic!("Invalid type for log"),
        }
    }

    fn floor(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::floor)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::floor)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::floor)]),
            _ => panic!("Invalid type for floor"),
        }
    }

    fn max(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::max)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::max)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::max)]),
            U32([arr]) => U32([Self::unary_eager(arr, Self::max)]),
        }
    }

    fn sum(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, Self::sum)]),
            F16([arr]) => F16([Self::unary_eager(arr, Self::sum)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, Self::sum)]),
            U32([arr]) => U32([Self::unary_eager(arr, Self::sum)]),
        }
    }

    fn argmax(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => U32([Self::unary_eager(arr, Self::argmax)]),
            F16([arr]) => U32([Self::unary_eager(arr, Self::argmax)]),
            BF16([arr]) => U32([Self::unary_eager(arr, Self::argmax)]),
            U32([arr]) => U32([Self::unary_eager(arr, Self::argmax)]),
        }
    }

    fn topk(&self, x: TaggedTensor<Self>, k: usize) -> (TaggedTensor<Self>, TaggedTensor<Self>) {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let arr = arr.materialize();
                let (values, indices) = Self::topk_f32(&arr, k);
                (F32([values.into()]), U32([indices.into()]))
            }
            F16([arr]) => {
                let arr = arr.materialize();
                let (values, indices) = Self::topk_f32(&arr, k);
                (F16([values.into()]), U32([indices.into()]))
            }
            BF16([arr]) => {
                let arr = arr.materialize();
                let (values, indices) = Self::topk_f32(&arr, k);
                (BF16([values.into()]), U32([indices.into()]))
            }
            _ => panic!("Unsupported type for topk"),
        }
    }

    fn broadcast(&self, x: TaggedTensor<Self>, shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let arr = arr.materialize();
                F32([Self::broadcast_tensor(&arr, shape).into()])
            }
            F16([arr]) => {
                let arr = arr.materialize();
                F16([Self::broadcast_tensor(&arr, shape).into()])
            }
            BF16([arr]) => {
                let arr = arr.materialize();
                BF16([Self::broadcast_tensor(&arr, shape).into()])
            }
            U32([arr]) => {
                let arr = arr.materialize();
                U32([Self::broadcast_tensor(&arr, shape).into()])
            }
        }
    }

    fn reshape(&self, x: TaggedTensor<Self>, new_shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let arr = arr.materialize();
                F32([Self::reshape_tensor(&arr, new_shape).into()])
            }
            F16([arr]) => {
                let arr = arr.materialize();
                F16([Self::reshape_tensor(&arr, new_shape).into()])
            }
            BF16([arr]) => {
                let arr = arr.materialize();
                BF16([Self::reshape_tensor(&arr, new_shape).into()])
            }
            U32([arr]) => {
                let arr = arr.materialize();
                U32([Self::reshape_tensor(&arr, new_shape).into()])
            }
        }
    }

    fn transpose(&self, x: TaggedTensor<Self>, dim0: usize, dim1: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([arr.transpose(dim0, dim1)]),
            F16([arr]) => F16([arr.transpose(dim0, dim1)]),
            BF16([arr]) => BF16([arr.transpose(dim0, dim1)]),
            U32([arr]) => U32([arr.transpose(dim0, dim1)]),
        }
    }
    fn arange(&self, end: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        let r = Tensor::arange(0, end as u32, &self.device).unwrap();
        U32([r.into()])
    }

    fn to_bool(&self, x: TaggedTensor<Self>) -> bool {
        match x {
            TaggedTensor::F32([x]) => x
                .materialize()
                .gt(0.0)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::F16([x]) => x
                .materialize()
                .gt(0.0)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::BF16([x]) => x
                .materialize()
                .gt(0.0)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::U32([x]) => x
                .materialize()
                .ne(0u32)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
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
            (F32([arr]), U32([indices])) => F32([Self::index_tensor(arr, dim, indices)]),
            (F16([arr]), U32([indices])) => F16([Self::index_tensor(arr, dim, indices)]),
            (BF16([arr]), U32([indices])) => BF16([Self::index_tensor(arr, dim, indices)]),
            (U32([arr]), U32([indices])) => U32([Self::index_tensor(arr, dim, indices)]),
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
            F32([arr]) => F32([arr.slice(dim, start, len)]),
            F16([arr]) => F16([arr.slice(dim, start, len)]),
            BF16([arr]) => BF16([arr.slice(dim, start, len)]),
            U32([arr]) => U32([arr.slice(dim, start, len)]),
        }
    }

    fn compare(&self, x: TaggedTensorTuple<Self, 2>) -> bool {
        use TaggedTensorTuple::*;
        match x {
            F32([a, b]) => Self::compare_tensors(&a.materialize(), &b.materialize()),
            F16([a, b]) => Self::compare_tensors(&a.materialize(), &b.materialize()),
            BF16([a, b]) => Self::compare_tensors(&a.materialize(), &b.materialize()),
            U32([a, b]) => Self::compare_tensors(&a.materialize(), &b.materialize()),
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
                F32([Self::concat_tensors(&a.materialize(), &b.materialize(), dim).into()])
            }
            (F16([a]), F16([b])) => {
                F16([Self::concat_tensors(&a.materialize(), &b.materialize(), dim).into()])
            }
            (BF16([a]), BF16([b])) => {
                BF16([Self::concat_tensors(&a.materialize(), &b.materialize(), dim).into()])
            }
            (U32([a]), U32([b])) => {
                U32([Self::concat_tensors(&a.materialize(), &b.materialize(), dim).into()])
            }
            _ => panic!("Incompatible types for concatenation"),
        }
    }
}

impl CandleBackend {
    fn float_tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
        let tensor = match tensor.dtype() {
            DType::F32 => tensor.clone(),
            DType::F16 | DType::BF16 => tensor.to_dtype(DType::F32).unwrap(),
            dtype => panic!("Unsupported float tensor dtype {dtype:?}"),
        };

        tensor.flatten_all().unwrap().to_vec1().unwrap()
    }

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

    fn reshape_tensor(tensor: &Tensor, new_shape: Shape) -> Tensor {
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

    fn unary_eager(x: CandleTensor, op: fn(&Tensor) -> CandleTensor) -> CandleTensor {
        let x = x.materialize();
        op(&x)
    }

    fn binary_eager(
        x: CandleTensor,
        y: CandleTensor,
        op: fn(&Tensor, &Tensor) -> CandleTensor,
    ) -> CandleTensor {
        let x = x.materialize();
        let y = y.materialize();
        op(&x, &y)
    }

    fn ternary_eager(
        x: CandleTensor,
        y: CandleTensor,
        z: CandleTensor,
        op: fn(&Tensor, &Tensor, &Tensor) -> CandleTensor,
    ) -> CandleTensor {
        let x = x.materialize();
        let y = y.materialize();
        let z = z.materialize();
        op(&x, &y, &z)
    }

    fn matmul_tensors(lhs: CandleTensor, rhs: CandleTensor) -> CandleTensor {
        match (&lhs.deferred_index0, &rhs.deferred_index0) {
            (None, Some(DeferredIndex0 { indices, dim })) if *dim == 0 => {
                Self::indexed_batched_matmul_rhs(&lhs.tensor, &rhs.tensor, indices).into()
            }
            _ => {
                let lhs = lhs.materialize();
                let rhs = rhs.materialize();
                Self::batched_matmul(&lhs, &rhs).into()
            }
        }
    }

    fn indexed_batched_matmul_rhs(lhs: &Tensor, rhs: &Tensor, indices: &Tensor) -> Tensor {
        let num_experts = rhs.dim(0).unwrap();
        let mut positions_by_expert = vec![Vec::<u32>::new(); num_experts];

        for (position, expert_id) in indices
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .into_iter()
            .enumerate()
        {
            positions_by_expert[expert_id as usize].push(position as u32);
        }

        let mut out_dims = lhs.dims().to_vec();
        let rhs_out_dim = *rhs.dims().last().unwrap();
        *out_dims.last_mut().unwrap() = rhs_out_dim;
        let mut out = Tensor::zeros(&*out_dims, lhs.dtype(), lhs.device()).unwrap();

        for (expert_id, positions) in positions_by_expert.into_iter().enumerate() {
            if positions.is_empty() {
                continue;
            }

            let positions_len = positions.len();
            let positions = Tensor::from_vec(positions, positions_len, lhs.device()).unwrap();
            let lhs_chunk = lhs.index_select(&positions, 0).unwrap();
            let rhs_chunk = rhs
                .narrow(0, expert_id, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .broadcast_as(vec![positions_len, rhs.dims()[1], rhs_out_dim])
                .unwrap();
            let result_chunk = Self::batched_matmul(&lhs_chunk, &rhs_chunk);
            out = out.index_add(&positions, &result_chunk, 0).unwrap();
        }

        out
    }

    fn index_tensor(input: CandleTensor, dim: usize, indices: CandleTensor) -> CandleTensor {
        let indices = indices.materialize();
        if dim == 0 && input.deferred_index0.is_none() {
            CandleTensor::from_indexed_select(&input.tensor, dim, &indices)
        } else {
            let tensor = input.materialize();
            Self::index_tensor_materialized(&tensor, dim, &indices).into()
        }
    }

    fn index_tensor_materialized(tensor: &Tensor, dim: usize, indices: &Tensor) -> Tensor {
        let idx = indices.flatten_all().unwrap();
        tensor.index_select(&idx, dim).unwrap()
    }

    fn slice_tensor_materialized(tensor: &Tensor, dim: usize, start: usize, len: usize) -> Tensor {
        tensor.narrow(dim, start, len).unwrap()
    }

    fn broadcast_tensor(tensor: &Tensor, shape: Shape) -> Tensor {
        tensor.broadcast_as(shape.0).unwrap()
    }

    fn add(x: &Tensor, y: &Tensor) -> CandleTensor {
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }
        ((x + y).unwrap()).into()
    }

    fn sub(x: &Tensor, y: &Tensor) -> CandleTensor {
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }
        ((x - y).unwrap()).into()
    }

    fn mul(x: &Tensor, y: &Tensor) -> CandleTensor {
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }
        ((x * y).unwrap()).into()
    }

    fn div(x: &Tensor, y: &Tensor) -> CandleTensor {
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }
        ((x / y).unwrap()).into()
    }

    fn lt(x: &Tensor, y: &Tensor) -> CandleTensor {
        let dtype = x.dtype();
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }

        x.lt(y).unwrap().to_dtype(dtype).unwrap().into()
    }

    fn gt(x: &Tensor, y: &Tensor) -> CandleTensor {
        let dtype = x.dtype();
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }

        x.gt(y).unwrap().to_dtype(dtype).unwrap().into()
    }

    fn gte(x: &Tensor, y: &Tensor) -> CandleTensor {
        let dtype = x.dtype();
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }

        x.ge(y).unwrap().to_dtype(dtype).unwrap().into()
    }

    fn lte(x: &Tensor, y: &Tensor) -> CandleTensor {
        let dtype = x.dtype();
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }

        x.le(y).unwrap().to_dtype(dtype).unwrap().into()
    }

    fn eq(x: &Tensor, y: &Tensor) -> CandleTensor {
        let dtype = x.dtype();
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }

        x.eq(y).unwrap().to_dtype(dtype).unwrap().into()
    }

    fn where_cond(mask: &Tensor, x: &Tensor, y: &Tensor) -> CandleTensor {
        let mask = match mask.dtype() {
            DType::F32 | DType::F16 | DType::BF16 => mask.gt(0.).unwrap(),
            DType::U32 => mask.ne(0u32).unwrap(),
            _ => mask.clone(), // already U8 (boolean) or other type
        };

        mask.where_cond(x, y).unwrap().into()
    }

    fn neg(x: &Tensor) -> CandleTensor {
        x.neg().unwrap().into()
    }

    fn sin(x: &Tensor) -> CandleTensor {
        x.sin().unwrap().into()
    }

    fn cos(x: &Tensor) -> CandleTensor {
        x.cos().unwrap().into()
    }

    fn log(x: &Tensor) -> CandleTensor {
        x.log().unwrap().into()
    }

    fn floor(x: &Tensor) -> CandleTensor {
        x.floor().unwrap().into()
    }

    // Candle's pow function does not support negative base and silently generates NaNs
    // so we do element-wise powf https://github.com/huggingface/candle/issues/1640
    fn pow(x: &Tensor, y: &Tensor) -> CandleTensor {
        if x.dims() != y.dims() {
            panic!("Shape mismatch in operation");
        }

        let dtype = x.dtype();
        let shape = x.dims().to_vec();
        let device = x.device().clone();

        // Convert tensors to vectors for element-wise powf operation
        let x_vec = Self::float_tensor_to_f32_vec(x);
        let y_vec = Self::float_tensor_to_f32_vec(y);

        // Perform element-wise powf
        let result_vec: Vec<f32> = x_vec
            .iter()
            .zip(y_vec.iter())
            .map(|(a, b)| a.powf(*b))
            .collect();

        // Convert back to tensor with original shape
        let result_tensor = Tensor::from_vec(result_vec, shape, &device).unwrap();
        let result_tensor = if dtype == DType::F32 {
            result_tensor
        } else {
            result_tensor.to_dtype(dtype).unwrap()
        };
        result_tensor.into()
    }

    fn sum(x: &Tensor) -> CandleTensor {
        x.sum_keepdim(D::Minus1).unwrap().into()
    }

    fn max(x: &Tensor) -> CandleTensor {
        x.max_keepdim(D::Minus1).unwrap().into()
    }

    fn argmax(x: &Tensor) -> CandleTensor {
        x.argmax_keepdim(D::Minus1).unwrap().into()
    }

    fn topk_f32(tensor: &Tensor, k: usize) -> (Tensor, Tensor) {
        let (values, indices) = tensor.sort_last_dim(false).unwrap();
        let topk_indices = indices.narrow(D::Minus1, 0, k).unwrap();
        let topk_values = values.narrow(D::Minus1, 0, k).unwrap();
        (topk_values, topk_indices)
    }

    fn contiguous_if_needed(tensor: &Tensor) -> Tensor {
        if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous().unwrap()
        }
    }

    pub fn batched_matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        match lhs.matmul(rhs) {
            Ok(result) => result,
            // On error retry with contiguous inputs.
            Err(_) => {
                let lhs = Self::contiguous_if_needed(lhs);
                let rhs = Self::contiguous_if_needed(rhs);
                lhs.matmul(&rhs).unwrap()
            }
        }
    }
}

impl BackendTensorOps for CandleTensor {
    fn shape(&self) -> Shape {
        self.shape()
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

    let result = CandleBackend::batched_matmul(&lhs, &rhs);

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

#[test]
fn test_indexed_select_to_vec_matches_materialized_gather() {
    let tensor = Tensor::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .reshape(&[3, 2])
    .unwrap();
    let indices = Tensor::new(&[2u32, 0], &candle_core::Device::Cpu).unwrap();

    let expected = CandleBackend::index_tensor_materialized(&tensor, 0, &indices);
    let actual = CandleTensor::from_indexed_select(&tensor, 0, &indices).materialize();

    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
}

#[test]
fn test_indexed_select_slice_matches_materialized_gather() {
    let tensor = Tensor::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .reshape(&[4, 2])
    .unwrap();
    let indices = Tensor::new(&[3u32, 1, 2], &candle_core::Device::Cpu).unwrap();

    let expected = CandleBackend::index_tensor_materialized(&tensor, 0, &indices)
        .narrow(0, 1, 2)
        .unwrap();
    let actual = CandleTensor::from_indexed_select(&tensor, 0, &indices)
        .slice(0, 1, 2)
        .materialize();

    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
}

#[test]
fn test_indexed_select_rhs_matmul_matches_materialized_gather() {
    let lhs = Tensor::new(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .reshape(&[3, 1, 2])
    .unwrap();
    let rhs = Tensor::new(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .reshape(&[3, 2, 2])
    .unwrap();
    let indices = Tensor::new(&[2u32, 0, 2], &candle_core::Device::Cpu).unwrap();

    let expected_rhs = CandleBackend::index_tensor_materialized(&rhs, 0, &indices);
    let expected = CandleBackend::batched_matmul(&lhs, &expected_rhs);
    let actual = CandleBackend::matmul_tensors(
        lhs.into(),
        CandleTensor::from_indexed_select(&rhs, 0, &indices),
    )
    .materialize();

    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
}
