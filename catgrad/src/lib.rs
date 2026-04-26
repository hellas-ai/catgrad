#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
pub mod category;
pub mod cid;
pub mod stdlib;

// Compiler passes
pub mod pass;

// path::Path is a type of dot-separated strings used to name definitions and lang ops.
pub mod path;

// Shapechecking & Evaluation
pub mod interpreter;
#[cfg(feature = "runtime")]
pub mod runtime;
pub mod ssa;
pub mod typecheck;

// general compiler tools
pub mod abstract_interpreter;
pub mod definition;

// Utilities
#[cfg(feature = "svg")]
pub mod svg;

pub(crate) mod util;

// entry point
pub mod prelude;

////////////////////////////////////////////////////////////////////////////////
// Macros

/// Syntax sugar for creating a shape
#[macro_export]
macro_rules! shape {
    ($b:expr, $($x:expr),+ $(,)?) => {{
        let dims = [ $( $crate::prelude::IntoNatVar::to_nat((&$x), $b) ),+ ];
        pack($b, dims)
    }};
}
