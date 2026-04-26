//! A captured state vector that can resume a [`super::Session`].
//!
//! Snapshots are live runtime state, not content-addressed blobs. They carry
//! the program id and parameter tensor CIDs needed for resume compatibility,
//! but they deliberately do not hash state tensors at construction. Hashing or
//! exporting state is an explicit higher-level operation because state can be
//! huge and backend-resident.

use std::sync::Arc;

use crate::category::core::Dtype;
use crate::cid::{Cid, DagCborEncoder, SnapshotBundle, Tensor, tensor_cid_from_backend_tensor};
use crate::interpreter::{self, Backend};

use super::Program;

const SNAPSHOT_BUNDLE_SCHEMA: &str = "hellas.snapshot.v1";

#[derive(Clone, Debug)]
pub struct Snapshot<B: Backend> {
    program_id: Cid<Program>,
    parameters: Arc<[Cid<Tensor>]>,
    state: Vec<interpreter::Value<B>>,
}

impl<B: Backend> Snapshot<B> {
    pub(crate) fn new(
        program_id: Cid<Program>,
        parameters: Arc<[Cid<Tensor>]>,
        state: Vec<interpreter::Value<B>>,
    ) -> Self {
        Self {
            program_id,
            parameters,
            state,
        }
    }

    pub fn program_id(&self) -> Cid<Program> {
        self.program_id
    }

    pub fn parameters(&self) -> &Arc<[Cid<Tensor>]> {
        &self.parameters
    }

    pub(crate) fn into_state(self) -> Vec<interpreter::Value<B>> {
        self.state
    }

    pub(crate) fn state(&self) -> &[interpreter::Value<B>] {
        &self.state
    }

    /// Per-tensor content CIDs in program-defined order. One `Cid<Tensor>`
    /// per state slot. Cost: hashes each backend tensor's bytes — for
    /// large KV caches this is non-trivial but typically a one-time cost
    /// per snapshot.
    pub fn tensor_cids(&self, backend: &B) -> Vec<Cid<Tensor>> {
        self.state
            .iter()
            .map(|value| match value {
                interpreter::Value::Tensor(tensor) => {
                    tensor_cid_from_backend_tensor(backend, tensor)
                }
                other => panic!("snapshot state contained non-tensor value: {other:?}"),
            })
            .collect()
    }

    /// Single content hash over this snapshot's state tensor CIDs in
    /// program-defined order. Used as the `final_state` field on
    /// `TextReceipt` — one CID hides per-tensor structural metadata
    /// (layer count, shape arity) at the receipt level. Per-tensor CIDs
    /// are still derivable via [`Self::tensor_cids`] for callers that
    /// need them (auth-gated availability checks, etc.).
    pub fn cid(&self, backend: &B) -> Cid<SnapshotBundle> {
        let tensor_cids = self.tensor_cids(backend);
        let mut encoder = DagCborEncoder::new();
        encoder.array(2);
        encoder.str(SNAPSHOT_BUNDLE_SCHEMA);
        encoder.array(tensor_cids.len() as u64);
        for tensor_cid in &tensor_cids {
            encoder.bytes(tensor_cid.as_bytes());
        }
        Cid::from_dag_cbor_bytes(&encoder.into_bytes())
    }

    /// Total bytes allocated by state tensors.
    pub fn allocated(&self) -> usize {
        self.state
            .iter()
            .map(|value| match value {
                interpreter::Value::Tensor(tensor) => tensor
                    .shape()
                    .size()
                    .saturating_mul(dtype_size(tensor.dtype())),
                _ => 0,
            })
            .sum()
    }
}

const fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 | Dtype::U32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
    }
}

#[cfg(all(test, feature = "ndarray-backend"))]
mod tests {
    use super::{Program, Snapshot};
    use crate::cid::Cid;
    use crate::interpreter::backend::ndarray::NdArrayBackend;
    use crate::interpreter::{self, Backend};
    use std::sync::Arc;

    #[test]
    fn allocated_uses_state_tensor_shapes() {
        let backend = NdArrayBackend;
        let state = vec![
            interpreter::Value::Tensor(backend.zeros(
                crate::prelude::Shape(vec![2, 3]),
                crate::prelude::Dtype::F32,
            )),
            interpreter::Value::Tensor(
                backend.zeros(crate::prelude::Shape(vec![5]), crate::prelude::Dtype::U32),
            ),
        ];
        let snapshot = Snapshot::new(
            Cid::<Program>::from_bytes([0; 32]),
            Arc::from(Vec::<_>::new().into_boxed_slice()),
            state,
        );
        assert_eq!(snapshot.allocated(), (2 * 3 + 5) * 4);
    }
}
