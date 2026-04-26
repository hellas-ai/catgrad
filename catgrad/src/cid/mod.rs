//! Typed content identifiers and canonical blob encodings.
//!
//! Layered into three modules so feature gates are coarse:
//!
//! - [`typed`]: always available. The `Cid<T>` type, hex Display, and the
//!   shared hex-parsing helper.
//! - [`dag_cbor`] (gated on `dag-cbor`): canonical encoding helpers,
//!   `DagCborEncoder`, the `Tensor` blob marker, and tensor CID utilities.
//!   Pulls in `blake3` and `serde_ipld_dagcbor` only here.
//! - [`serde_impls`] (gated on `serde`): `Serialize`/`Deserialize` impls for
//!   `Cid<T>` (lowercase hex string).

mod typed;

#[cfg(feature = "dag-cbor")]
mod dag_cbor;

#[cfg(feature = "serde")]
mod serde_impls;

pub use typed::Cid;

#[cfg(feature = "dag-cbor")]
pub use dag_cbor::{
    DagCborEncoder, SnapshotBundle, Tensor, dag_cbor_bytes, tensor_cid,
    tensor_cid_from_backend_tensor, tensor_dag_cbor_bytes, u32_tensor_cid,
};
