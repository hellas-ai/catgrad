use super::super::types::*;
use crate::category::core::{Dtype, Shape};
use crate::interpreter::backend::{Backend, BackendError, BackendTensorOps};
use candle_core::{
    D, DType, Device, Module, Tensor,
    quantized::{QMatMul, QTensor},
    utils::{cuda_is_available, metal_is_available},
};
use float8::F8E4M3;
use std::sync::Arc;

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
struct DenseTensor {
    tensor: Tensor,
    deferred_index0: Option<DeferredIndex0>,
}

#[derive(Clone, Debug)]
struct QuantizedTensor {
    qtensor: Arc<QTensor>,
    qmatmul: QMatMul,
    shape: Shape,
    transposed: bool,
}

#[derive(Clone, Debug)]
enum CandleTensorInner {
    Dense(DenseTensor),
    Quantized(QuantizedTensor),
}

#[derive(Clone, Debug)]
pub struct CandleTensor {
    inner: CandleTensorInner,
}

impl CandleTensor {
    fn from_materialized(tensor: Tensor) -> Self {
        Self {
            inner: CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0: None,
            }),
        }
    }

    fn from_indexed_select(tensor: &Tensor, dim: usize, indices: &Tensor) -> Self {
        Self {
            inner: CandleTensorInner::Dense(DenseTensor {
                tensor: tensor.clone(),
                deferred_index0: Some(DeferredIndex0 {
                    indices: indices.flatten_all().unwrap(),
                    dim,
                }),
            }),
        }
    }

    fn from_quantized(qtensor: QTensor) -> Self {
        let qtensor = Arc::new(qtensor);
        let qmatmul = QMatMul::from_arc(qtensor.clone()).unwrap();
        let shape = Shape(qtensor.shape().dims().to_vec());
        Self {
            inner: CandleTensorInner::Quantized(QuantizedTensor {
                qtensor,
                qmatmul,
                shape,
                transposed: false,
            }),
        }
    }

    fn can_use_quantized_matmul(&self) -> bool {
        matches!(
            self.inner,
            CandleTensorInner::Quantized(QuantizedTensor {
                transposed: true,
                ..
            })
        )
    }

    pub fn materialize(&self) -> Tensor {
        self.materialize_for_dtype(DType::F32)
    }

    pub fn materialize_for_dtype(&self, dtype: DType) -> Tensor {
        match &self.inner {
            CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0,
            }) => {
                let tensor = match deferred_index0 {
                    None => tensor.clone(),
                    Some(DeferredIndex0 { indices, dim }) => {
                        CandleBackend::index_tensor_materialized(tensor, *dim, indices)
                    }
                };
                if tensor.dtype() == dtype {
                    tensor
                } else {
                    tensor.to_dtype(dtype).unwrap()
                }
            }
            CandleTensorInner::Quantized(QuantizedTensor {
                qtensor,
                shape,
                transposed,
                ..
            }) => {
                let mut tensor = match dtype {
                    DType::F16 => qtensor.dequantize_f16(&qtensor.device()).unwrap(),
                    DType::BF16 => qtensor
                        .dequantize(&qtensor.device())
                        .unwrap()
                        .to_dtype(DType::BF16)
                        .unwrap(),
                    DType::F32 => qtensor.dequantize(&qtensor.device()).unwrap(),
                    DType::F8E4M3 => qtensor
                        .dequantize(&qtensor.device())
                        .unwrap()
                        .to_dtype(DType::F8E4M3)
                        .unwrap(),
                    DType::U32 => panic!("cannot materialize quantized tensor as u32"),
                    _ => panic!("unsupported candle tensor dtype for quantized materialization"),
                };
                if *transposed {
                    tensor = tensor.transpose(0, 1).unwrap();
                }
                if tensor.dims() != shape.0 {
                    tensor = tensor.broadcast_as(shape.0.clone()).unwrap();
                }
                if tensor.dtype() != dtype {
                    tensor.to_dtype(dtype).unwrap()
                } else {
                    tensor
                }
            }
        }
    }

    fn transpose(&self, dtype: DType, dim0: usize, dim1: usize) -> Self {
        match &self.inner {
            CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0,
            }) => {
                let tensor = tensor.transpose(dim0, dim1).unwrap();
                let deferred_index0 =
                    deferred_index0
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
                    inner: CandleTensorInner::Dense(DenseTensor {
                        tensor,
                        deferred_index0,
                    }),
                }
            }
            CandleTensorInner::Quantized(quantized) => {
                if quantized.shape.rank() == 2 && dim0 == 0 && dim1 == 1 {
                    let mut quantized = quantized.clone();
                    quantized.transposed = !quantized.transposed;
                    quantized.shape.0.swap(0, 1);
                    Self {
                        inner: CandleTensorInner::Quantized(quantized),
                    }
                } else {
                    let tensor = self
                        .materialize_for_dtype(dtype)
                        .transpose(dim0, dim1)
                        .unwrap();
                    Self::from_materialized(tensor)
                }
            }
        }
    }

    fn slice(&self, dtype: DType, dim: usize, start: usize, len: usize) -> Self {
        match &self.inner {
            CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0,
            }) => match deferred_index0 {
                None => Self::from_materialized(CandleBackend::slice_tensor_materialized(
                    tensor, dim, start, len,
                )),
                Some(DeferredIndex0 {
                    indices,
                    dim: indexed_dim,
                }) if dim == *indexed_dim => Self {
                    inner: CandleTensorInner::Dense(DenseTensor {
                        tensor: tensor.clone(),
                        deferred_index0: Some(DeferredIndex0 {
                            indices: indices.narrow(0, start, len).unwrap(),
                            dim: *indexed_dim,
                        }),
                    }),
                },
                Some(DeferredIndex0 {
                    indices,
                    dim: indexed_dim,
                }) => Self {
                    inner: CandleTensorInner::Dense(DenseTensor {
                        tensor: tensor.narrow(dim, start, len).unwrap(),
                        deferred_index0: Some(DeferredIndex0 {
                            indices: indices.clone(),
                            dim: *indexed_dim,
                        }),
                    }),
                },
            },
            CandleTensorInner::Quantized(_) => {
                let tensor = CandleBackend::slice_tensor_materialized(
                    &self.materialize_for_dtype(dtype),
                    dim,
                    start,
                    len,
                );
                Self::from_materialized(tensor)
            }
        }
    }

    fn shape(&self) -> Shape {
        match &self.inner {
            CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0,
            }) => match deferred_index0 {
                None => Shape(tensor.dims().to_vec()),
                Some(DeferredIndex0 { indices, dim }) => {
                    let mut dims = tensor.dims().to_vec();
                    dims[*dim] = indices.dims1().unwrap();
                    Shape(dims)
                }
            },
            CandleTensorInner::Quantized(QuantizedTensor { shape, .. }) => shape.clone(),
        }
    }

    fn quantized_matmul(&self, lhs: &Tensor, output_dtype: DType) -> Tensor {
        let CandleTensorInner::Quantized(QuantizedTensor {
            qmatmul,
            transposed,
            ..
        }) = &self.inner
        else {
            panic!("quantized_matmul called on dense tensor");
        };
        assert!(
            *transposed,
            "quantized matmul requires a transposed weight view"
        );
        let compute_dtype = match output_dtype {
            DType::BF16 => DType::F16,
            DType::F32 | DType::F16 => output_dtype,
            _ => panic!("unsupported output dtype for quantized matmul: {output_dtype:?}"),
        };
        let lhs = if lhs.dtype() == compute_dtype {
            lhs.clone()
        } else {
            lhs.to_dtype(compute_dtype).unwrap()
        };
        let out = qmatmul.forward(&lhs).unwrap();
        if out.dtype() == output_dtype {
            out
        } else {
            out.to_dtype(output_dtype).unwrap()
        }
    }

    fn dense_parts(&self) -> Option<(&Tensor, Option<&DeferredIndex0>)> {
        match &self.inner {
            CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0,
            }) => Some((tensor, deferred_index0.as_ref())),
            CandleTensorInner::Quantized(_) => None,
        }
    }

    fn index_tensor(&self, dtype: DType, dim: usize, indices: CandleTensor) -> Self {
        let indices = indices.materialize_for_dtype(DType::U32);
        match &self.inner {
            CandleTensorInner::Dense(DenseTensor {
                tensor,
                deferred_index0,
            }) if dim == 0 && deferred_index0.is_none() => {
                CandleTensor::from_indexed_select(tensor, dim, &indices)
            }
            _ => {
                let tensor = self.materialize_for_dtype(dtype);
                CandleBackend::index_tensor_materialized(&tensor, dim, &indices).into()
            }
        }
    }

    fn broadcast_to(&self, dtype: DType, shape: Shape) -> Self {
        match &self.inner {
            CandleTensorInner::Dense(_) => {
                let tensor = self.materialize_for_dtype(dtype);
                Self::from_materialized(CandleBackend::broadcast_tensor(&tensor, shape))
            }
            CandleTensorInner::Quantized(quantized) => {
                let current = &quantized.shape.0;
                if shape.0.len() >= current.len()
                    && shape.0[shape.0.len() - current.len()..] == current[..]
                {
                    let mut quantized = quantized.clone();
                    quantized.shape = shape;
                    Self {
                        inner: CandleTensorInner::Quantized(quantized),
                    }
                } else {
                    let tensor = self.materialize_for_dtype(dtype);
                    Self::from_materialized(CandleBackend::broadcast_tensor(&tensor, shape))
                }
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
    fn candle_dtype(dtype: Dtype) -> DType {
        match dtype {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::F8 => DType::F8E4M3,
            Dtype::U32 => DType::U32,
        }
    }

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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tagged_tensor_from_tensor(&self, tensor: Tensor) -> TaggedTensor<Self> {
        match tensor.dtype() {
            DType::F32 => TaggedTensor::F32([tensor.into()]),
            DType::F16 => TaggedTensor::F16([tensor.into()]),
            DType::BF16 => TaggedTensor::BF16([tensor.into()]),
            DType::F8E4M3 => TaggedTensor::FP8([tensor.into()]),
            DType::U32 => TaggedTensor::U32([tensor.into()]),
            dtype => panic!("unsupported candle tensor dtype for catgrad import: {dtype:?}"),
        }
    }

    pub fn tagged_tensor_from_qtensor(
        &self,
        qtensor: QTensor,
        logical_dtype: Dtype,
    ) -> TaggedTensor<Self> {
        let tensor = CandleTensor::from_quantized(qtensor);
        match logical_dtype {
            Dtype::F32 => TaggedTensor::F32([tensor]),
            Dtype::F16 => TaggedTensor::F16([tensor]),
            Dtype::BF16 => TaggedTensor::BF16([tensor]),
            Dtype::F8 | Dtype::U32 => {
                panic!("unsupported logical dtype for quantized candle import: {logical_dtype:?}")
            }
        }
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
                let x = x.materialize_for_dtype(DType::F32);
                TaggedVec::F32(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::F16([x]) => {
                let x = x.materialize_for_dtype(DType::F16);
                TaggedVec::F16(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::BF16([x]) => {
                let x = x.materialize_for_dtype(DType::BF16);
                TaggedVec::BF16(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::FP8([x]) => {
                let x = x.materialize_for_dtype(DType::F8E4M3);
                TaggedVec::FP8(x.flatten_all().unwrap().to_vec1().unwrap())
            }
            TaggedTensor::U32([x]) => {
                let x = x.materialize_for_dtype(DType::U32);
                TaggedVec::U32(x.flatten_all().unwrap().to_vec1().unwrap())
            }
        }
    }

    fn format_tensor(&self, tensor: &TaggedTensor<Self>) -> String {
        match tensor {
            TaggedTensor::F32([x]) => format!("{}", x.materialize_for_dtype(DType::F32)),
            TaggedTensor::F16([x]) => format!("{}", x.materialize_for_dtype(DType::F16)),
            TaggedTensor::BF16([x]) => format!("{}", x.materialize_for_dtype(DType::BF16)),
            TaggedTensor::FP8([x]) => format!("{}", x.materialize_for_dtype(DType::F8E4M3)),
            TaggedTensor::U32([x]) => format!("{}", x.materialize_for_dtype(DType::U32)),
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
            Dtype::F8 => {
                let tensor = Tensor::zeros(dims, DType::F8E4M3, &self.device).unwrap();
                TaggedTensor::FP8([tensor.into()])
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

    fn ndarray_from_vec_fp8(
        &self,
        data: Vec<F8E4M3>,
        shape: Shape,
    ) -> Result<TaggedTensor<Self>, BackendError> {
        let dims: &[usize] = &shape.0;
        let tensor =
            Tensor::from_vec(data, dims, &self.device).map_err(|_| BackendError::ShapeError)?;
        Ok(TaggedTensor::FP8([tensor.into()]))
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

        let source_dtype = Self::candle_dtype(x.dtype());
        let tensor = match x {
            TaggedTensor::F32([arr])
            | TaggedTensor::F16([arr])
            | TaggedTensor::BF16([arr])
            | TaggedTensor::FP8([arr])
            | TaggedTensor::U32([arr]) => arr.materialize_for_dtype(source_dtype),
        };
        let target_dtype = Self::candle_dtype(target_dtype);
        self.ensure_dtype_supported(target_dtype);
        let tensor: CandleTensor = tensor.to_dtype(target_dtype).unwrap().into();
        match target_dtype {
            DType::F32 => TaggedTensor::F32([tensor]),
            DType::F16 => TaggedTensor::F16([tensor]),
            DType::BF16 => TaggedTensor::BF16([tensor]),
            DType::F8E4M3 => TaggedTensor::FP8([tensor]),
            DType::U32 => TaggedTensor::U32([tensor]),
            _ => unreachable!(),
        }
    }

    fn matmul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::matmul_tensors(x, y, DType::F32)]),
            F16([x, y]) => F16([Self::matmul_tensors(x, y, DType::F16)]),
            BF16([x, y]) => BF16([Self::matmul_tensors(x, y, DType::BF16)]),
            FP8([x, y]) => FP8([Self::matmul_tensors(x, y, DType::F8E4M3)]),
            U32([x, y]) => U32([Self::matmul_tensors(x, y, DType::U32)]),
        }
    }

    fn add(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::add)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::add)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::add)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::add)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::add)]),
        }
    }

    fn sub(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::sub)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::sub)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::sub)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::sub)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::sub)]),
        }
    }

    fn mul(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::mul)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::mul)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::mul)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::mul)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::mul)]),
        }
    }

    fn div(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::div)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::div)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::div)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::div)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::div)]),
        }
    }

    fn lt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::lt)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::lt)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::lt)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::lt)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::lt)]),
        }
    }

    fn gt(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::gt)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::gt)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::gt)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::gt)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::gt)]),
        }
    }

    fn gte(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::gte)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::gte)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::gte)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::gte)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::gte)]),
        }
    }

    fn lte(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::lte)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::lte)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::lte)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::lte)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::lte)]),
        }
    }

    fn eq(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::eq)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::eq)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::eq)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::eq)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::eq)]),
        }
    }

    fn where_cond(&self, args: TaggedTensorTuple<Self, 3>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match args {
            F32([mask, x, y]) => F32([Self::ternary_eager(
                mask,
                x,
                y,
                DType::F32,
                Self::where_cond,
            )]),
            F16([mask, x, y]) => F16([Self::ternary_eager(
                mask,
                x,
                y,
                DType::F16,
                Self::where_cond,
            )]),
            BF16([mask, x, y]) => BF16([Self::ternary_eager(
                mask,
                x,
                y,
                DType::BF16,
                Self::where_cond,
            )]),
            FP8([mask, x, y]) => FP8([Self::ternary_eager(
                mask,
                x,
                y,
                DType::F8E4M3,
                Self::where_cond,
            )]),
            U32([mask, x, y]) => U32([Self::ternary_eager(
                mask,
                x,
                y,
                DType::U32,
                Self::where_cond,
            )]),
        }
    }

    fn pow(&self, lhs: TaggedTensorTuple<Self, 2>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match lhs {
            F32([x, y]) => F32([Self::binary_eager(x, y, DType::F32, Self::pow)]),
            F16([x, y]) => F16([Self::binary_eager(x, y, DType::F16, Self::pow)]),
            BF16([x, y]) => BF16([Self::binary_eager(x, y, DType::BF16, Self::pow)]),
            FP8([x, y]) => FP8([Self::binary_eager(x, y, DType::F8E4M3, Self::pow)]),
            U32([x, y]) => U32([Self::binary_eager(x, y, DType::U32, Self::pow)]),
        }
    }

    fn neg(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::neg)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::neg)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::neg)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::neg)]),
            U32([arr]) => U32([Self::unary_eager(arr, DType::U32, Self::neg)]),
        }
    }

    fn sin(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::sin)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::sin)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::sin)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::sin)]),
            _ => panic!("Invalid type for sin"),
        }
    }

    fn cos(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::cos)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::cos)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::cos)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::cos)]),
            _ => panic!("Invalid type for cos"),
        }
    }

    fn exp(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::exp)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::exp)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::exp)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::exp)]),
            _ => panic!("Invalid type for exp"),
        }
    }

    fn log(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::log)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::log)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::log)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::log)]),
            _ => panic!("Invalid type for log"),
        }
    }

    fn floor(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::floor)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::floor)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::floor)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::floor)]),
            _ => panic!("Invalid type for floor"),
        }
    }

    fn max(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::max)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::max)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::max)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::max)]),
            U32([arr]) => U32([Self::unary_eager(arr, DType::U32, Self::max)]),
        }
    }

    fn sum(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([Self::unary_eager(arr, DType::F32, Self::sum)]),
            F16([arr]) => F16([Self::unary_eager(arr, DType::F16, Self::sum)]),
            BF16([arr]) => BF16([Self::unary_eager(arr, DType::BF16, Self::sum)]),
            FP8([arr]) => FP8([Self::unary_eager(arr, DType::F8E4M3, Self::sum)]),
            U32([arr]) => U32([Self::unary_eager(arr, DType::U32, Self::sum)]),
        }
    }

    fn argmax(&self, x: TaggedTensor<Self>) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => U32([Self::unary_eager(arr, DType::F32, Self::argmax)]),
            F16([arr]) => U32([Self::unary_eager(arr, DType::F16, Self::argmax)]),
            BF16([arr]) => U32([Self::unary_eager(arr, DType::BF16, Self::argmax)]),
            FP8([arr]) => U32([Self::unary_eager(arr, DType::F8E4M3, Self::argmax)]),
            U32([arr]) => U32([Self::unary_eager(arr, DType::U32, Self::argmax)]),
        }
    }

    fn topk(&self, x: TaggedTensor<Self>, k: usize) -> (TaggedTensor<Self>, TaggedTensor<Self>) {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let arr = arr.materialize_for_dtype(DType::F32);
                let (values, indices) = Self::topk_f32(&arr, k);
                (F32([values.into()]), U32([indices.into()]))
            }
            F16([arr]) => {
                let arr = arr.materialize_for_dtype(DType::F16);
                let (values, indices) = Self::topk_f32(&arr, k);
                (F16([values.into()]), U32([indices.into()]))
            }
            BF16([arr]) => {
                let arr = arr.materialize_for_dtype(DType::BF16);
                let (values, indices) = Self::topk_f32(&arr, k);
                (BF16([values.into()]), U32([indices.into()]))
            }
            FP8([arr]) => {
                let arr = arr.materialize_for_dtype(DType::F8E4M3);
                let (values, indices) = Self::topk_f32(&arr, k);
                (FP8([values.into()]), U32([indices.into()]))
            }
            _ => panic!("Unsupported type for topk"),
        }
    }

    fn broadcast(&self, x: TaggedTensor<Self>, shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([arr.broadcast_to(DType::F32, shape)]),
            F16([arr]) => F16([arr.broadcast_to(DType::F16, shape)]),
            BF16([arr]) => BF16([arr.broadcast_to(DType::BF16, shape)]),
            FP8([arr]) => FP8([arr.broadcast_to(DType::F8E4M3, shape)]),
            U32([arr]) => U32([arr.broadcast_to(DType::U32, shape)]),
        }
    }

    fn reshape(&self, x: TaggedTensor<Self>, new_shape: Shape) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => {
                let arr = arr.materialize_for_dtype(DType::F32);
                F32([Self::reshape_tensor(&arr, new_shape).into()])
            }
            F16([arr]) => {
                let arr = arr.materialize_for_dtype(DType::F16);
                F16([Self::reshape_tensor(&arr, new_shape).into()])
            }
            BF16([arr]) => {
                let arr = arr.materialize_for_dtype(DType::BF16);
                BF16([Self::reshape_tensor(&arr, new_shape).into()])
            }
            FP8([arr]) => {
                let arr = arr.materialize_for_dtype(DType::F8E4M3);
                FP8([Self::reshape_tensor(&arr, new_shape).into()])
            }
            U32([arr]) => {
                let arr = arr.materialize_for_dtype(DType::U32);
                U32([Self::reshape_tensor(&arr, new_shape).into()])
            }
        }
    }

    fn transpose(&self, x: TaggedTensor<Self>, dim0: usize, dim1: usize) -> TaggedTensor<Self> {
        use TaggedTensorTuple::*;
        match x {
            F32([arr]) => F32([arr.transpose(DType::F32, dim0, dim1)]),
            F16([arr]) => F16([arr.transpose(DType::F16, dim0, dim1)]),
            BF16([arr]) => BF16([arr.transpose(DType::BF16, dim0, dim1)]),
            FP8([arr]) => FP8([arr.transpose(DType::F8E4M3, dim0, dim1)]),
            U32([arr]) => U32([arr.transpose(DType::U32, dim0, dim1)]),
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
                .materialize_for_dtype(DType::F32)
                .gt(0.0)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::F16([x]) => x
                .materialize_for_dtype(DType::F16)
                .gt(0.0)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::BF16([x]) => x
                .materialize_for_dtype(DType::BF16)
                .gt(0.0)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::FP8([x]) => x
                .materialize_for_dtype(DType::F8E4M3)
                .gt(F8E4M3::ZERO)
                .ok()
                .and_then(|t| t.max_all().ok())
                .and_then(|m| m.to_scalar::<u8>().ok())
                .map(|s| s == 1)
                .unwrap_or(false),
            TaggedTensor::U32([x]) => x
                .materialize_for_dtype(DType::U32)
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
            (F32([arr]), U32([indices])) => F32([arr.index_tensor(DType::F32, dim, indices)]),
            (F16([arr]), U32([indices])) => F16([arr.index_tensor(DType::F16, dim, indices)]),
            (BF16([arr]), U32([indices])) => BF16([arr.index_tensor(DType::BF16, dim, indices)]),
            (FP8([arr]), U32([indices])) => FP8([arr.index_tensor(DType::F8E4M3, dim, indices)]),
            (U32([arr]), U32([indices])) => U32([arr.index_tensor(DType::U32, dim, indices)]),
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
            F32([arr]) => F32([arr.slice(DType::F32, dim, start, len)]),
            F16([arr]) => F16([arr.slice(DType::F16, dim, start, len)]),
            BF16([arr]) => BF16([arr.slice(DType::BF16, dim, start, len)]),
            FP8([arr]) => FP8([arr.slice(DType::F8E4M3, dim, start, len)]),
            U32([arr]) => U32([arr.slice(DType::U32, dim, start, len)]),
        }
    }

    fn compare(&self, x: TaggedTensorTuple<Self, 2>) -> bool {
        use TaggedTensorTuple::*;
        match x {
            F32([a, b]) => Self::compare_tensors(
                &a.materialize_for_dtype(DType::F32),
                &b.materialize_for_dtype(DType::F32),
            ),
            F16([a, b]) => Self::compare_tensors(
                &a.materialize_for_dtype(DType::F16),
                &b.materialize_for_dtype(DType::F16),
            ),
            BF16([a, b]) => Self::compare_tensors(
                &a.materialize_for_dtype(DType::BF16),
                &b.materialize_for_dtype(DType::BF16),
            ),
            FP8([a, b]) => Self::compare_tensors(
                &a.materialize_for_dtype(DType::F8E4M3),
                &b.materialize_for_dtype(DType::F8E4M3),
            ),
            U32([a, b]) => Self::compare_tensors(
                &a.materialize_for_dtype(DType::U32),
                &b.materialize_for_dtype(DType::U32),
            ),
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
            (F32([a]), F32([b])) => F32([Self::concat_tensors(
                &a.materialize_for_dtype(DType::F32),
                &b.materialize_for_dtype(DType::F32),
                dim,
            )
            .into()]),
            (F16([a]), F16([b])) => F16([Self::concat_tensors(
                &a.materialize_for_dtype(DType::F16),
                &b.materialize_for_dtype(DType::F16),
                dim,
            )
            .into()]),
            (BF16([a]), BF16([b])) => BF16([Self::concat_tensors(
                &a.materialize_for_dtype(DType::BF16),
                &b.materialize_for_dtype(DType::BF16),
                dim,
            )
            .into()]),
            (FP8([a]), FP8([b])) => FP8([Self::concat_tensors(
                &a.materialize_for_dtype(DType::F8E4M3),
                &b.materialize_for_dtype(DType::F8E4M3),
                dim,
            )
            .into()]),
            (U32([a]), U32([b])) => U32([Self::concat_tensors(
                &a.materialize_for_dtype(DType::U32),
                &b.materialize_for_dtype(DType::U32),
                dim,
            )
            .into()]),
            _ => panic!("Incompatible types for concatenation"),
        }
    }
}

impl CandleBackend {
    fn float_tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
        let tensor = match tensor.dtype() {
            DType::F32 => tensor.clone(),
            DType::F16 | DType::BF16 | DType::F8E4M3 => tensor.to_dtype(DType::F32).unwrap(),
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

    fn unary_eager(x: CandleTensor, dtype: DType, op: fn(&Tensor) -> CandleTensor) -> CandleTensor {
        let x = x.materialize_for_dtype(dtype);
        op(&x)
    }

    fn binary_eager(
        x: CandleTensor,
        y: CandleTensor,
        dtype: DType,
        op: fn(&Tensor, &Tensor) -> CandleTensor,
    ) -> CandleTensor {
        let x = x.materialize_for_dtype(dtype);
        let y = y.materialize_for_dtype(dtype);
        op(&x, &y)
    }

    fn ternary_eager(
        x: CandleTensor,
        y: CandleTensor,
        z: CandleTensor,
        dtype: DType,
        op: fn(&Tensor, &Tensor, &Tensor) -> CandleTensor,
    ) -> CandleTensor {
        let x = x.materialize_for_dtype(dtype);
        let y = y.materialize_for_dtype(dtype);
        let z = z.materialize_for_dtype(dtype);
        op(&x, &y, &z)
    }

    fn matmul_tensors(lhs: CandleTensor, rhs: CandleTensor, output_dtype: DType) -> CandleTensor {
        if rhs.can_use_quantized_matmul() {
            let lhs = lhs.materialize_for_dtype(output_dtype);
            return rhs.quantized_matmul(&lhs, output_dtype).into();
        }

        match (lhs.dense_parts(), rhs.dense_parts()) {
            (
                Some((lhs_tensor, None)),
                Some((rhs_tensor, Some(DeferredIndex0 { indices, dim }))),
            ) if *dim == 0 => {
                Self::indexed_batched_matmul_rhs(lhs_tensor, rhs_tensor, indices).into()
            }
            _ => {
                let lhs = lhs.materialize_for_dtype(output_dtype);
                let rhs = rhs.materialize_for_dtype(output_dtype);
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
            DType::F8E4M3 => mask.gt(F8E4M3::ZERO).unwrap(),
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

    fn exp(x: &Tensor) -> CandleTensor {
        x.exp().unwrap().into()
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
        .slice(DType::F32, 0, 1, 2)
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
        DType::F32,
    )
    .materialize();

    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(
        actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    );
}
