//! Trait helpers for creating the stdlib and defining models/layers/etc.
use crate::category::lang::*;
use crate::typecheck::Type;
use crate::util::build_typed;

use open_hypergraphs::lax::var;

////////////////////////////////////////////////////////////////////////////////
// Generic interface

/// A type implementing [`Module`] defines a *typed term* with additional metadata.
/// Analogous to a `nn.Module` in PyTorch.
/// In total, a `T: Module` defines:
///
/// 1. A *unique global name* (`path`)
/// 2. A *type* (`ty`)
/// 3. A *definition* (`inline`) - an open hypergraph representing
///
/// Note that definitions have fixed arity/coarity, but types can vary.
pub trait Module<const A: usize, const B: usize> {
    /// The *type* of this definition, used to construct a [`TypedTerm`]
    fn ty(&self) -> ([Type; A], [Type; B]);

    /// Unique global name in the stdlib/environment
    fn path(&self) -> Path;

    /// The *definition* of this term, as a function which mutably inlines it into the provided
    /// Builder.
    fn def(&self, builder: &Builder, args: [Var; A]) -> [Var; B];

    ////////////////////////////////////////
    // Derived functions

    /// The *sort* of this type are the *object labels* of the sources/targets of its definition.
    fn sort(&self) -> ([Object; A], [Object; B]) {
        let (v1, v2) = self.ty();
        (v1.map(to_sort), v2.map(to_sort))
    }

    /// alias for `def` which is clearer to use in context, e.g. Sigmoid::inline();
    fn inline(&self, builder: &Builder, args: [Var; A]) -> [Var; B] {
        self.def(builder, args)
    }

    /// Create a single `Definition` operation in the graph with name `self.path()`.
    fn op(&self, builder: &Builder, args: [Var; A]) -> [Var; B] {
        let result_types = self.sort().1.to_vec();
        var::operation(
            builder,
            &args,
            result_types,
            Operation::Definition(self.path()),
        )
        .try_into()
        .unwrap() // guaranteed to work: size fixed by result_types
    }

    /// Construct a standalone OpenHypergraph for this definition
    /// Returns None when `self.inline` returned vars with sorts different to those declared in
    /// `self.ty`.
    fn term(&self) -> Option<TypedTerm> {
        let (source_type, target_type) = self.ty();
        let source_object = source_type.clone().map(to_sort);

        // TODO: err handling
        let term = build_typed(source_object, |builder, args| {
            self.inline(builder, args).to_vec()
        })
        .unwrap();

        use open_hypergraphs::category::*; // TODO: remove this trait import when it's auto-exported by OpenHypergraph
        if term.target() != target_type.clone().map(to_sort) {
            None
        } else {
            Some(TypedTerm {
                term,
                source_type: source_type.to_vec(),
                target_type: target_type.to_vec(),
            })
        }
    }
}

/// Like [`Module`], but with dynamic arity/coarity using vectors.
pub trait DynModule {
    /// The *type* of this definition, used to construct a [`TypedTerm`].
    fn ty(&self) -> (Vec<Type>, Vec<Type>);

    /// Unique global name in the stdlib/environment.
    fn path(&self) -> Path;

    /// The *definition* of this term, as a function which mutably inlines it into the provided
    /// Builder.
    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var>;

    ////////////////////////////////////////
    // Derived functions

    /// The *sort* of this type are the *object labels* of the sources/targets of its definition.
    fn sort(&self) -> (Vec<Object>, Vec<Object>) {
        let (source_type, target_type) = self.ty();
        (
            source_type.into_iter().map(to_sort).collect(),
            target_type.into_iter().map(to_sort).collect(),
        )
    }

    /// Alias for `def` which is clearer to use in context.
    fn inline(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        self.def(builder, args)
    }

    /// Create a single `Definition` operation in the graph with name `self.path()`.
    fn op(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let result_types = self.sort().1;
        var::operation(
            builder,
            &args,
            result_types,
            Operation::Definition(self.path()),
        )
    }

    /// Construct a standalone OpenHypergraph for this definition.
    fn term(&self) -> Option<TypedTerm> {
        let (source_type, target_type) = self.ty();
        let source_object: Vec<Object> = source_type.iter().cloned().map(to_sort).collect();

        let term = var::build(|builder| {
            let args: Vec<Var> = source_object
                .iter()
                .cloned()
                .map(|o| Var::new(builder.clone(), o))
                .collect();
            let sources = args.clone();
            let targets = self.inline(builder, args);
            (sources, targets)
        })
        .ok()?;

        use open_hypergraphs::category::*;
        let target_object: Vec<Object> = target_type.iter().cloned().map(to_sort).collect();
        if term.target() != target_object {
            None
        } else {
            Some(TypedTerm {
                term,
                source_type,
                target_type,
            })
        }
    }
}

/// Get the corresponding [`Object`] (sort) for a given [`Type`]
// TODO: move?
fn to_sort(value: Type) -> Object {
    match value {
        Type::Type(_) => Object::NdArrayType,
        Type::Shape(_) => Object::Shape,
        Type::Nat(_) => Object::Nat,
        Type::Dtype(_) => Object::Dtype,
        Type::Tensor(_) => Object::Tensor,
    }
}

////////////////////////////////////////////////////////////////////////////////
// "Function" definitions just return a single Var- this makes it easier to call them.

/// A FnModule is a "Function Module": a `Module` with a single output var.
/// The `call` method is a helper for getting the single output of self.op.
pub trait FnModule<const N: usize>: Module<N, 1> {
    /// Like [`Module::op`] for coarity 1.
    fn call(&self, builder: &Builder, args: [Var; N]) -> Var {
        let [r] = self.op(builder, args);
        r
    }

    // TODO: call_inline?
}

impl<const N: usize, T: Module<N, 1>> FnModule<N> for T {}
