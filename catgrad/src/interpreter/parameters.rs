//! Materialized (tensor-only) parameters for the runtime interpreter.
//!
//! The abstract [`abstract_interpreter::parameters::Parameters<I>`] is
//! heterogeneous: values are `Value<I>`, which for typechecking are types
//! and for the runtime are `Value<B>` (tensor *or* nat / dtype / shape /
//! …). That generality is right for the abstract layer.
//!
//! At the runtime layer, parameters are always tensors — model loading
//! produces tensors, and the program's `Load(path)` ops resolve to
//! tensor values. [`Parameters<B>`] makes that fact part of the type:
//! a path-keyed map of [`TaggedTensor<B>`], no other variants admitted.
//! When the interpreter needs to surface a parameter as a `Value<B>`, it
//! wraps the tensor in `Value::Tensor` at the lookup site.
//!
//! These parameters are *materialized* — actual tensor data on a backend,
//! not symbolic types or expressions. Loading a model produces a
//! [`Parameters<B>`]; binding a [`crate::runtime::Program`] consumes one.
//!
//! [`abstract_interpreter::parameters::Parameters<I>`]: crate::abstract_interpreter::parameters::Parameters

use super::{Backend, TaggedTensor};
use crate::path::Path;
use std::collections::BTreeMap;

/// Materialized parameters: path → tensor map.
///
/// Construct via [`From<BTreeMap<Path, TaggedTensor<B>>>`] or
/// [`Self::default`]; iteration order is path-sorted (intrinsic to
/// [`BTreeMap`]).
#[derive(Clone, Debug)]
pub struct Parameters<B: Backend>(pub BTreeMap<Path, TaggedTensor<B>>);

impl<B: Backend> Default for Parameters<B> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<B: Backend> From<BTreeMap<Path, TaggedTensor<B>>> for Parameters<B> {
    fn from(map: BTreeMap<Path, TaggedTensor<B>>) -> Self {
        Self(map)
    }
}

impl<B: Backend, const N: usize> From<[(Path, TaggedTensor<B>); N]> for Parameters<B> {
    fn from(arr: [(Path, TaggedTensor<B>); N]) -> Self {
        Self(BTreeMap::from(arr))
    }
}

impl<B: Backend> Parameters<B> {
    pub fn keys(&self) -> std::collections::btree_map::Keys<'_, Path, TaggedTensor<B>> {
        self.0.keys()
    }
}
