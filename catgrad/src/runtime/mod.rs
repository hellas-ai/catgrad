//! Execute catgrad [`Program`]s against materialized parameter tensors.
//!
//! Three pieces fit together to take a typechecked computation graph all
//! the way to a running session:
//!
//! 1. A [`Program`] is a content-addressed, typechecked term plus the
//!    metadata needed to evaluate it (state schema, sequence-length budget,
//!    optional per-step nat hint).
//! 2. A [`crate::interpreter::Parameters`] is a path → tensor map (see its
//!    docs for the "materialized" framing). Loaded once, shared across
//!    many binds.
//! 3. A [`BoundProgram`] is the pairing of a `Program` with materialized
//!    parameters; constructed via [`BoundProgram::bind`], which hashes
//!    every parameter tensor to derive its content CIDs. Sessions are
//!    spawned from a `BoundProgram`.
//! 4. A [`Session`] is a live evaluation of a [`BoundProgram`]. State
//!    flows through successive `run` calls and can be captured into a
//!    [`Snapshot`] for resumption.
//!
//! Typical flow:
//!
//! ```ignore
//! let bound       = BoundProgram::bind(&parameters, &backend, program)?;
//! let bound       = Arc::new(bound);
//! let mut session = bound.clone().start(bound.empty_snapshot())?;
//! let outputs     = session.run(args)?;
//! let snap        = session.into_snapshot();
//! ```
//!
//! Domain-specific layers (text causal stepping, multimodal prefill, …)
//! compose over [`BoundProgram`] via extension traits rather than living
//! inside this module.
//!
//! # Submodules
//!
//! - `program`   — the typechecked computation graph; content-addressed
//!   via canonical DAG-CBOR encoding.
//! - `bound`     — `Program` bound against a parameter set; owns the
//!   per-tensor parameter CIDs derived at bind time.
//! - `session`   — a running `BoundProgram`, owning the live state vector.
//! - `snapshot`  — a captured state vector, resume-compatible with the
//!   `(Program, Parameters)` pair that produced it.
//! - `error`     — `RuntimeError` and the module's `Result` alias.

mod bound;
mod error;
mod program;
mod session;
mod snapshot;

pub use bound::BoundProgram;
pub use error::{Result, RuntimeError};
pub use program::{Program, ProgramSpec};
pub use session::Session;
pub use snapshot::Snapshot;
