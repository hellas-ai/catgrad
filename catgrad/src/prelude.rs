//! Default imports
// Constructing model graphs
pub use crate::category::core::Shape;
pub use crate::category::lang::{
    Builder, Dtype, Term, Type, TypedTerm, Var, ops::IntoDtypeVar, ops::IntoNatVar,
};
pub use crate::stdlib::{FnModule, Module, nn, ops, stdlib, to_load_ops};

// Interpreting and compiling
pub use crate::interpreter;
pub use crate::pass::to_core::Environment;
pub use crate::typecheck;

// Utilities and Macros
pub use crate::path::{Path, path};
pub use crate::shape;
