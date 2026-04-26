use super::{Interpreter, Value};
use crate::path::Path;
use std::collections::btree_map::{BTreeMap,Keys};

#[derive(Clone, Debug)]
pub struct Parameters<I: Interpreter>(pub BTreeMap<Path, Value<I>>);

// Needed so Backend doesn't have to implement Default
impl<I: Interpreter> Default for Parameters<I> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<I: Interpreter> From<BTreeMap<Path, Value<I>>> for Parameters<I> {
    fn from(map: BTreeMap<Path, Value<I>>) -> Self {
        Parameters(map)
    }
}

impl<const N: usize, I: Interpreter> From<[(Path, Value<I>); N]> for Parameters<I> {
    fn from(arr: [(Path, Value<I>); N]) -> Self {
        Parameters(BTreeMap::from(arr))
    }
}

impl<'a, I: Interpreter> IntoIterator for &'a Parameters<I> {
    type Item = &'a Path;
    type IntoIter = Keys<'a, Path, Value<I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.keys()
    }
}

impl<I: Interpreter> Parameters<I> {
    pub fn keys(&self) -> Keys<'_, Path, Value<I>> {
        self.0.keys()
    }
}
