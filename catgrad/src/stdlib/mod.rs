//! Standard library of declarations and definitions for catgrad's `lang` category.

// Actual definitions and the `to_load_ops` helper.
mod stdlib_definitions;
pub use stdlib_definitions::{stdlib, to_load_ops};

// The [`Module`] helper for creating language definitions
mod module;
pub use module::{DynModule, FnModule, Module};

// Basic ops like reshape, transpose, matmul
pub mod ops;

// Neural network layers (e.g., layernorm, sigmoid, matmul)
pub mod nn;
