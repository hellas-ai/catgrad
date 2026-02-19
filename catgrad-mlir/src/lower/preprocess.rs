use catgrad::category::lang;
use catgrad::prelude::*;
use open_hypergraphs::lax::OpenHypergraph;

// TODO: PERFORMANCE: this is inefficient when called multiple times: inlining should be done "bottom-up" in
// topological dependency order so we don't repeat work.
// FIX: pass a *vec* of names, then inline the whole environment bottom-up, and emit MLIR for each
// of the passed names.
/// Render a TypedTerm and its associated environment of definitions to MLIR
pub fn preprocess(
    env: &Environment,
    params: &typecheck::Parameters,
    path: Path,
) -> (Vec<Path>, OpenHypergraph<Type, lang::Operation>) {
    let mut typed_term = env.definitions.get(&path).unwrap().clone();

    // Inline all operations
    // TODO: use catgrad's inline; refactor lang to use Def?
    let defs = env
        .definitions
        .iter()
        .map(|(path, typed_term)| (path.clone(), typed_term.term.clone()))
        .collect();
    let term = super::inline::inline(defs, typed_term.term);

    // Forget extraneous Copy operations
    typed_term.term = open_hypergraphs::lax::var::forget::forget_monogamous(&term);

    // TODO: fix use of unwrap()
    // Typecheck `term` and get an open hypergraph annotated with *normalized* types
    let node_annotations: Vec<_> = typecheck::check(env, params, typed_term.clone())
        .unwrap()
        .into_iter()
        .map(typecheck::normalize)
        .collect();

    // Verify no free Dtype variables (which cannot be compiled in MLIR) by checking there were no
    // non-constant Dtypes in the entrypoint term's node annotations
    // TODO: replace assertion with a Result
    assert!(node_annotations.iter().all(is_dtype_monomorphic));

    // Create a term with normalized types for node labels
    let checked_term = typed_term.term.with_nodes(|_| node_annotations).unwrap();

    // Map type-level ops to identities at runtime
    let checked_term = super::functor::forget_identity_casts(&checked_term);

    // Forget all copy ops for MLIR (it's harder to deal with)
    let checked_term = open_hypergraphs::lax::var::forget::forget(&checked_term);

    // Factor out parameters
    let param_ops: std::collections::HashMap<Path, ()> =
        params.keys().map(|k| (path.concat(k), ())).collect();

    let (params, checked_term) = super::factor(&checked_term, |op| match op {
        lang::Operation::Declaration(dec_path) => param_ops.contains_key(dec_path),
        _ => false,
    });

    // Panics below should never happen; factor above only pulls declarations.
    let param_paths = params.hypergraph.edges.into_iter().map(|e| match e {
        lang::Operation::Declaration(path) => path,
        lang::Operation::Definition(_path) => panic!("impossible definition found in params!"),
        lang::Operation::Literal(_literal) => panic!("impossible literal found in params!"),
        lang::Operation::Probe(_label) => panic!("impossible probe found in params!"),
    });

    // Return all param paths in order of use
    (param_paths.collect(), checked_term)

    // TODO: Instead of inlining, produce MLIR for independent functions in the environment.
    // This requires:
    //
    //  - Monomorphization (generate variants for each dtype argument)
    //  - Deparametrization (modify each term in env. to explicitly pass params)
}

/// Ensure the boundaries of the entry point term do not have free Dtype vars
fn is_dtype_monomorphic(ty: &Type) -> bool {
    match &ty {
        Type::Dtype(d) => match d {
            typecheck::DtypeExpr::Constant(_) => true,
            // Non-constant dtypes not permitted
            _ => false,
        },
        _ => true,
    }
}
