use open_hypergraphs::lax::functor::*;
use open_hypergraphs::lax::*;

use catgrad::category::lang::Operation;
use catgrad::prelude::*;

pub fn forget_identity_casts(
    f: &OpenHypergraph<Type, Operation>,
) -> OpenHypergraph<Type, Operation> {
    ForgetIdentityCasts.map_arrow(f)
}

// not public: no use for this except via forget_monogamous
#[derive(Clone)]
struct ForgetIdentityCasts;

fn proj0(t0: &Type, t1: &Type) -> OpenHypergraph<Type, Operation> {
    let id = OpenHypergraph::identity(vec![t0.clone()]);
    let copy = Path::new(vec!["cartesian", "copy"]).unwrap();
    let discard = OpenHypergraph::singleton(Operation::Declaration(copy), vec![t1.clone()], vec![]);
    &id | &discard
}

impl Functor<Type, Operation, Type, Operation> for ForgetIdentityCasts {
    // Identity-on-objects
    fn map_object(&self, o: &Type) -> impl ExactSizeIterator<Item = Type> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Operation,
        source: &[Type],
        target: &[Type],
    ) -> OpenHypergraph<Type, Operation> {
        let cast = Path::new(vec!["tensor", "cast"]).unwrap();
        if (Operation::Declaration(cast) == *a) && (source[0] == target[0]) {
            proj0(&source[0], &source[1])
        } else {
            OpenHypergraph::singleton(a.clone(), source.to_vec(), target.to_vec())
        }
    }

    fn map_arrow(&self, f: &OpenHypergraph<Type, Operation>) -> OpenHypergraph<Type, Operation> {
        dyn_functor::define_map_arrow(self, f)
    }
}
