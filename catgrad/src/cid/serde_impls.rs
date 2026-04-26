//! `Serialize` / `Deserialize` for [`Cid<T>`] as a 64-char lowercase hex string.
//!
//! The whole module is gated behind the `serde` feature; no per-item cfgs.
//!
//! Note: this is the *string* representation used by JSON / log / metadata
//! formats. Inside DAG-CBOR commitment blobs, CIDs are encoded as raw 32-byte
//! byte strings via [`super::DagCborEncoder`]; do not rely on this serde
//! representation for canonical hashing.

use std::marker::PhantomData;

use serde::de::{Error as DeError, Visitor};

use super::typed::{Cid, parse_hex_32};

impl<T> serde::Serialize for Cid<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de, T> serde::Deserialize<'de> for Cid<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CidVisitor<T>(PhantomData<T>);

        impl<T> Visitor<'_> for CidVisitor<T> {
            type Value = Cid<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a 64-character lowercase hex CID")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                let bytes = parse_hex_32(value).map_err(E::custom)?;
                Ok(Cid::from_bytes(bytes))
            }
        }

        deserializer.deserialize_str(CidVisitor(PhantomData))
    }
}
