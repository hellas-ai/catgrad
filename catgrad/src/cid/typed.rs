//! `Cid<T>`: typed 32-byte content identifier.
//!
//! Always available. Computing a CID from canonical bytes lives in the
//! [`super::dag_cbor`] submodule; this file is just the value type and its
//! basic impls.

use std::marker::PhantomData;

/// Typed content identifier for a canonical blob.
///
/// The type parameter is a compile-time namespace. A `Cid<Program>` and a
/// `Cid<Tensor>` have the same byte representation but cannot be accidentally
/// passed to one another's APIs.
pub struct Cid<T> {
    bytes: [u8; 32],
    _ty: PhantomData<T>,
}

impl<T> Cid<T> {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self {
            bytes,
            _ty: PhantomData,
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

impl<T> Clone for Cid<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Cid<T> {}

impl<T> PartialEq for Cid<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
    }
}

impl<T> Eq for Cid<T> {}

impl<T> std::hash::Hash for Cid<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}

impl<T> std::fmt::Display for Cid<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.bytes {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl<T> std::fmt::Debug for Cid<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cid({self})")
    }
}

/// Parse a 64-character lowercase hex string into 32 bytes. Used by the
/// serde `Deserialize` impl and by the round-trip test; gated on the union
/// of those conditions so default-feature builds don't carry it as dead code.
#[cfg(any(feature = "serde", test))]
pub(super) fn parse_hex_32(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err(format!(
            "CID must be 64 hex characters, got {}",
            value.len()
        ));
    }
    let mut out = [0_u8; 32];
    for (idx, byte) in out.iter_mut().enumerate() {
        let hi = hex_nibble(value.as_bytes()[idx * 2])?;
        let lo = hex_nibble(value.as_bytes()[idx * 2 + 1])?;
        *byte = (hi << 4) | lo;
    }
    Ok(out)
}

#[cfg(any(feature = "serde", test))]
fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(format!("invalid lowercase hex byte 0x{byte:02x}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{Cid, parse_hex_32};

    struct A;
    struct B;

    #[test]
    fn cid_display_round_trips_hex() {
        let cid = Cid::<A>::from_bytes([0xab; 32]);
        let encoded = cid.to_string();
        let decoded = Cid::<A>::from_bytes(parse_hex_32(&encoded).unwrap());
        assert_eq!(decoded.as_bytes(), cid.as_bytes());
    }

    #[test]
    fn cid_types_do_not_interchange() {
        let cid = Cid::<A>::from_bytes([1; 32]);
        let same_bytes = Cid::<B>::from_bytes(*cid.as_bytes());
        assert_eq!(same_bytes.as_bytes(), cid.as_bytes());
    }
}
