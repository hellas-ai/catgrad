//! DAG-CBOR canonical encoding helpers and tensor blob primitives.
//!
//! Whole module is gated on the `dag-cbor` feature. `blake3` and
//! `serde_ipld_dagcbor` only enter the dependency graph here.

use half::{bf16, f16};

use super::typed::Cid;
use crate::category::core::{Dtype, Shape};
use crate::interpreter;

const TENSOR_SCHEMA: &str = "hellas.tensor.v1";

impl<T> Cid<T> {
    /// Hash arbitrary blob bytes (the canonical DAG-CBOR encoding of `T`)
    /// with blake3, producing a typed CID. The hash is the same 32-byte
    /// blake3 output that `iroh-blobs` uses for content addressing.
    pub fn from_dag_cbor_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes(*blake3::hash(bytes).as_bytes())
    }

    /// Encode `value` via `serde_ipld_dagcbor` and return its CID. Useful
    /// for ad-hoc commitment objects whose serde derive is already
    /// canonical-friendly. For complex objects, prefer building bytes via
    /// [`DagCborEncoder`] explicitly.
    pub fn from_dag_cbor_serialize<S>(
        value: &S,
    ) -> Result<Self, serde_ipld_dagcbor::EncodeError<std::collections::TryReserveError>>
    where
        S: serde::Serialize + ?Sized,
    {
        let bytes = dag_cbor_bytes(value)?;
        Ok(Self::from_dag_cbor_bytes(&bytes))
    }
}

/// Serialize `value` into DAG-CBOR bytes via `serde_ipld_dagcbor`.
pub fn dag_cbor_bytes<S>(
    value: &S,
) -> Result<Vec<u8>, serde_ipld_dagcbor::EncodeError<std::collections::TryReserveError>>
where
    S: serde::Serialize + ?Sized,
{
    serde_ipld_dagcbor::to_vec(value)
}

/// Minimal strict DAG-CBOR encoder for commitment blobs that should not rely
/// on serde field representation (e.g. when the blob's structure must be
/// stable independent of any future serde-derive change).
pub struct DagCborEncoder {
    bytes: Vec<u8>,
}

impl DagCborEncoder {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn array(&mut self, len: u64) {
        self.header(4, len);
    }

    pub fn bytes(&mut self, value: &[u8]) {
        self.header(2, value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    pub fn str(&mut self, value: &str) {
        self.header(3, value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.header(0, value);
    }

    pub fn i64(&mut self, value: i64) {
        if value >= 0 {
            self.header(0, value as u64);
        } else {
            self.header(1, (-1_i128 - value as i128) as u64);
        }
    }

    fn header(&mut self, major: u8, value: u64) {
        let major = major << 5;
        match value {
            0..=23 => self.bytes.push(major | value as u8),
            24..=0xff => self.bytes.extend_from_slice(&[major | 24, value as u8]),
            0x100..=0xffff => {
                self.bytes.push(major | 25);
                self.bytes.extend_from_slice(&(value as u16).to_be_bytes());
            }
            0x1_0000..=0xffff_ffff => {
                self.bytes.push(major | 26);
                self.bytes.extend_from_slice(&(value as u32).to_be_bytes());
            }
            _ => {
                self.bytes.push(major | 27);
                self.bytes.extend_from_slice(&value.to_be_bytes());
            }
        }
    }
}

impl Default for DagCborEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical tensor blob schema.
///
/// `Cid<Tensor>` hashes dtype, shape, and little-endian data bytes. It is
/// intentionally independent of any specific backend tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor;

/// Marker type for [`Cid<SnapshotBundle>`]: single content hash over a
/// [`super::super::runtime::Snapshot`]'s state tensor CIDs in
/// program-defined order. Used as the receipt's `final_state` field —
/// single CID (privacy on layer count) rather than a per-tensor list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotBundle;

pub fn tensor_cid(dtype: Dtype, shape: &Shape, data: &[u8]) -> Cid<Tensor> {
    Cid::from_dag_cbor_bytes(&tensor_dag_cbor_bytes(dtype, shape, data))
}

pub fn tensor_dag_cbor_bytes(dtype: Dtype, shape: &Shape, data: &[u8]) -> Vec<u8> {
    let mut encoder = DagCborEncoder::new();
    encoder.array(4);
    encoder.str(TENSOR_SCHEMA);
    encoder.str(dtype_name(dtype));
    encoder.array(shape.rank() as u64);
    for dim in &shape.0 {
        encoder.u64(*dim as u64);
    }
    encoder.bytes(data);
    encoder.into_bytes()
}

pub fn tensor_cid_from_backend_tensor<B: interpreter::Backend>(
    backend: &B,
    tensor: &interpreter::TaggedTensor<B>,
) -> Cid<Tensor> {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let bytes = tagged_vec_to_le_bytes(backend.to_vec(tensor.clone()));
    tensor_cid(dtype, &shape, &bytes)
}

pub fn u32_tensor_cid(shape: Shape, data: &[u32]) -> Cid<Tensor> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(data));
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    tensor_cid(Dtype::U32, &shape, &bytes)
}

fn tagged_vec_to_le_bytes(values: interpreter::TaggedVec) -> Vec<u8> {
    match values {
        interpreter::TaggedVec::F32(values) => le_bytes_u32(values, f32::to_bits),
        interpreter::TaggedVec::F16(values) => le_bytes_u16(values, f16::to_bits),
        interpreter::TaggedVec::BF16(values) => le_bytes_u16(values, bf16::to_bits),
        interpreter::TaggedVec::U32(values) => le_bytes_u32(values, |value| value),
    }
}

fn le_bytes_u32<T>(values: Vec<T>, to_bits: impl Fn(T) -> u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for value in values {
        bytes.extend_from_slice(&to_bits(value).to_le_bytes());
    }
    bytes
}

fn le_bytes_u16<T>(values: Vec<T>, to_bits: impl Fn(T) -> u16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
    for value in values {
        bytes.extend_from_slice(&to_bits(value).to_le_bytes());
    }
    bytes
}

const fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "f32",
        Dtype::F16 => "f16",
        Dtype::BF16 => "bf16",
        Dtype::U32 => "u32",
    }
}

#[cfg(test)]
mod tests {
    use super::u32_tensor_cid;
    use crate::category::core::Shape;

    #[test]
    fn tensor_cid_includes_shape() {
        let a = u32_tensor_cid(Shape(vec![2]), &[1, 2]);
        let b = u32_tensor_cid(Shape(vec![1, 2]), &[1, 2]);
        assert_ne!(a, b);
    }
}
