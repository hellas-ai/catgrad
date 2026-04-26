//! A live execution of a [`super::BoundProgram`].
//!
//! Owns the state vector that flows through successive `run` calls.
//! Holds an `Arc<BoundProgram>` so the bind can be shared across multiple
//! sessions; everything else (program, interpreter, parameter tensor CIDs)
//! is reached through that one Arc.
//!
//! State is captured into a [`super::Snapshot`] for later resumption via
//! [`Self::snapshot`] (clone) or [`Self::into_snapshot`] (consuming).

use std::sync::Arc;

use crate::interpreter;

use super::Snapshot;
use super::bound::BoundProgram;
use super::error::{Result, RuntimeError};

#[derive(Debug)]
pub struct Session<B: interpreter::Backend> {
    bound: Arc<BoundProgram<B>>,
    state: Vec<interpreter::Value<B>>,
}

impl<B: interpreter::Backend> Session<B> {
    pub(crate) fn from_bound(bound: Arc<BoundProgram<B>>, snapshot: Snapshot<B>) -> Result<Self> {
        if snapshot.program_id() != bound.program().id()
            || **snapshot.parameters() != **bound.parameters()
        {
            return Err(RuntimeError::IncompatibleSnapshot);
        }
        let expected_arity = bound.program().empty_state_type().len();
        if snapshot.state().len() != expected_arity {
            return Err(RuntimeError::InvalidProgram(format!(
                "snapshot state arity {} did not match expected {expected_arity}",
                snapshot.state().len()
            )));
        }

        Ok(Self {
            bound,
            state: snapshot.into_state(),
        })
    }

    /// Current state values (KV-cache, etc).
    ///
    /// The caller is responsible for placing these at the correct position
    /// in the input vector passed to [`Self::run`].
    pub fn state(&self) -> &[interpreter::Value<B>] {
        &self.state
    }

    pub fn snapshot(&self) -> Snapshot<B> {
        Snapshot::new(
            self.bound.program().id(),
            Arc::clone(self.bound.parameters()),
            self.state.clone(),
        )
    }

    pub fn into_snapshot(self) -> Snapshot<B> {
        Snapshot::new(
            self.bound.program().id(),
            Arc::clone(self.bound.parameters()),
            self.state,
        )
    }

    /// Run the program with the complete input vector.
    ///
    /// The caller must build the full input list including state from
    /// [`Self::state`] at the position the model expects. After execution,
    /// trailing state outputs are captured back into the session and the
    /// remaining outputs are returned.
    pub fn run(
        &mut self,
        inputs: Vec<interpreter::Value<B>>,
    ) -> Result<Vec<interpreter::Value<B>>> {
        let term = self.bound.program().typed_term().term.clone();
        let mut results = self
            .bound
            .interpreter()
            .run(term, inputs)
            .map_err(|err| RuntimeError::ExecutionError(err.to_string()))?;

        let state_arity = self.bound.program().empty_state_type().len();
        if results.len() < state_arity {
            return Err(RuntimeError::UnexpectedProgramOutput(format!(
                "program returned {} results for state arity {}",
                results.len(),
                state_arity,
            )));
        }

        let state_index = results.len() - state_arity;
        self.state = results.split_off(state_index);
        Ok(results)
    }
}
