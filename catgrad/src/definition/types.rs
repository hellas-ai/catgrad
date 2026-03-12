//! # Categories with *definitions*
//!
//! Allows extending a set of ops with some additional definitions.
//! For example, extend the set of ops `{add, negate}` with `sub := x y ↦  add(x, negate(y))`.
//!
//! Formally, given an symmetric monoidal theory `T = (Σ₀, Σ₁, Σ₂)`.
//! Def(T) is the theory presented by `(Σ₀, Σ₁ + K, Σ₂ + R)` where:
//! - `K` is a set of adjoined operations - (e.g. `{sub}`)
//! - `R` is a set of rewrites, one for each `k ∈ K`, expanding $k$ to its definition, e.g., `sub ⇝  x y ↦  add(x, negate(y))`.

/// A set of operations Arr extended with some additional operations in the set K.
/// `Def<K, Arr> ~= Σ₁ + K`
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug)]
pub enum Def<K, Arr> {
    Def(K),
    Arr(Arr),
}

use open_hypergraphs::lax::var::HasVar;

impl<K, A: HasVar> HasVar for Def<K, A> {
    fn var() -> Self {
        Def::Arr(A::var())
    }
}
