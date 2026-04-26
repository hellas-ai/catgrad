//! Output of [`Self::bind`]: a [`super::Program`] paired with the parameter
//! tensor CIDs it loaded against and the [`Interpreter`] ready to run it.
//!
//! A `BoundProgram` is the unit you spawn [`Session`]s from. It's
//! internally a value type with one `Arc` (the parameter CID list, the
//! natural type for an immutable shared sequence) — but [`Self::start`]
//! takes `Arc<Self>` because the session takes shared ownership of the
//! bind. Multi-session-per-bind is visible at call sites: `Arc::clone`
//! the handle and call `start` again.

use std::sync::Arc;

use crate::cid::{Cid, Tensor, tensor_cid_from_backend_tensor};
use crate::interpreter::{self, Backend, Interpreter};
use crate::prelude::{stdlib, to_load_ops};

use super::error::Result;
use super::{Program, Session, Snapshot};

/// A program loaded against a specific set of parameter tensors.
///
/// Carries:
/// - `program` — the catgrad [`Program`] (graph + types + sequence
///   length and other shape metadata). Held via [`Arc`] so multiple
///   binds against the same program share a single allocation; `Program`
///   itself is a deep value (the typed-term hypergraph), so cloning by
///   value would cost real time.
/// - `parameters` — content CIDs of the parameter tensors the program
///   was bound to, in canonical (path-sorted) order. Per-tensor list,
///   not a single aggregate, so callers can check tensor-level blob
///   availability directly.
/// - `interpreter` — the runtime that actually executes the program;
///   owns the backend handle and parameter values.
///
/// Construct via [`Self::bind`]; spawn sessions via [`Self::start`].
#[derive(Debug)]
pub struct BoundProgram<B: interpreter::Backend> {
    program: Arc<Program>,
    parameters: Arc<[Cid<Tensor>]>,
    interpreter: Interpreter<B>,
}

impl<B: Backend> BoundProgram<B> {
    /// Bind `program` against the materialized parameters `params`,
    /// hashing every tensor up front to derive the parameter CIDs.
    ///
    /// `params` is taken by reference so a single materialized set can
    /// back many bound programs; the tensor map is cloned (shallow —
    /// concrete backend tensors are Arc-wrapped, so this is cheap).
    /// `program` is accepted as either an owned `Program` or an
    /// `Arc<Program>` via [`Into`] — multi-bind callers should pass
    /// `Arc::clone(&shared)` to avoid the deep-clone cost of
    /// `Program::clone`.
    pub fn bind(
        params: &interpreter::Parameters<B>,
        backend: &B,
        program: impl Into<Arc<Program>>,
    ) -> Result<Self> {
        let program = program.into();

        let mut env = stdlib();
        env.declarations
            .extend(to_load_ops(program.module_path().clone(), params.keys()));

        let cids: Arc<[Cid<Tensor>]> = params
            .0
            .values()
            .map(|tensor| tensor_cid_from_backend_tensor(backend, tensor))
            .collect::<Vec<_>>()
            .into();

        let interpreter = Interpreter::new(backend.clone(), env, params.clone());
        Ok(Self {
            program,
            parameters: cids,
            interpreter,
        })
    }


    /// The bound [`Program`].
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Per-tensor content CIDs of this bind's parameters.
    pub fn parameters(&self) -> &Arc<[Cid<Tensor>]> {
        &self.parameters
    }

    /// The underlying interpreter (owns the backend handle and
    /// parameter values). Exposed for extension traits and integration
    /// code; most callers want [`Self::start`] instead.
    pub fn interpreter(&self) -> &Interpreter<B> {
        &self.interpreter
    }

    /// The empty starting state for this program — zero tensors of the
    /// shapes declared in `program.empty_state_type()`. Used as the
    /// cold-start snapshot when no prior execution has produced state
    /// for this bind.
    pub fn empty_snapshot(&self) -> Snapshot<B> {
        let state = self
            .program
            .empty_state_type()
            .iter()
            .map(|(dtype, shape)| {
                interpreter::Value::Tensor(self.interpreter.backend.zeros(shape.clone(), *dtype))
            })
            .collect();
        Snapshot::new(self.program.id(), Arc::clone(&self.parameters), state)
    }

    /// Spawn a [`Session`] from this bind, resuming the supplied snapshot.
    ///
    /// Consumes `Arc<Self>` because the session takes shared ownership
    /// of the bound program. To spawn multiple sessions from one bind,
    /// `Arc::clone` the handle before each call.
    pub fn start(self: Arc<Self>, snapshot: Snapshot<B>) -> Result<Session<B>> {
        Session::from_bound(self, snapshot)
    }
}
