use catgrad::category::lang::Operation;
use catgrad::prelude::*;

use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, dyn_functor::define_map_arrow},
};

use std::collections::HashMap;

#[derive(Clone)]
pub struct Inline<O> {
    env: HashMap<Path, OpenHypergraph<O, Operation>>,
}

// Def<String, Copy> → lang::Term
// Copy → Copy

// TODO: generalise definition?
impl<O: Clone + PartialEq> Functor<O, Operation, O, Operation> for Inline<O> {
    fn map_object(&self, o: &O) -> impl ExactSizeIterator<Item = O> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Operation,
        source: &[O],
        target: &[O],
    ) -> OpenHypergraph<O, Operation> {
        match a {
            Operation::Definition(k) => match self.env.get(k) {
                Some(term) => term.clone(),
                None => panic!("TODO"), // OpenHypergraph::singleton(None, source.to_vec(), target.to_vec()),
            },
            a => OpenHypergraph::singleton(a.clone(), source.to_vec(), target.to_vec()),
        }
    }

    fn map_arrow(&self, f: &OpenHypergraph<O, Operation>) -> OpenHypergraph<O, Operation> {
        define_map_arrow(self, f)
    }
}

// TODO: error with missing definitions!
pub fn inline<O: Clone + PartialEq>(
    env: HashMap<Path, OpenHypergraph<O, Operation>>,
    term: OpenHypergraph<O, Operation>,
) -> OpenHypergraph<O, Operation> {
    Inline { env }.map_arrow(&term)
}
