//! Text generation: live runtime state, the decoder typestate, and the
//! [`BoundProgramText`] extension trait that lets a [`BoundProgram`]
//! produce both.
//!
//! # Two layers
//!
//! - [`TextState<B>`] is the live runtime anchor. It holds a snapshot
//!   (live backend tensors) plus the [`TextReceipt`] that names this
//!   state and the receipt's CID. Returned by `genesis_text_state` for
//!   cold starts, and by [`TextDecoder::into_text_state`] at end of
//!   execution.
//!
//! - [`TextDecoder<B>`] is the only thing you can do *during* a text
//!   generation. There is no public way to construct it — the only path
//!   is through [`BoundProgramText::prefill`], which consumes a
//!   `TextState` and returns a decoder. From that point only single-
//!   token operations are available; the typestate forbids multi-token
//!   mid-decode calls.
//!
//! # Why typestate
//!
//! The pre-prefill state ("session exists, no tokens consumed") and the
//! post-prefill state ("can advance one token at a time") are different
//! operating regimes with disjoint allowed APIs. Encoding that as one
//! type with a runtime flag is a code smell — easy to call wrong, easy
//! to introduce bugs. The replacement is two facts at the type level:
//!
//! 1. Multi-token input only happens at session start, via `prefill`.
//! 2. The only thing returned by `prefill` is a `TextDecoder`, which
//!    has no `prefill` method.
//!
//! So calling prefill twice is a type error. Calling advance before
//! prefill is structurally impossible (no decoder yet).
//!
//! # No `TextSession`
//!
//! The previous design had a `TextSession` that started empty and held a
//! mutable `prefill_done: bool`. Both purposes are now covered by the
//! types above. `TextSession`, `TextSnapshot`, `TextStepOutput`, and the
//! `CausalStepper` trait are gone.

use std::sync::Arc;

use crate::{LLMError, Result};
use catgrad::category::core::Shape;
use catgrad::cid::{Cid, SnapshotBundle, Tensor, u32_tensor_cid};
use catgrad::interpreter::{self, Backend};
use catgrad::runtime::{BoundProgram, RuntimeError, Session, Snapshot};

use super::text_execution::{TextExecution, TextPolicy, TextReceipt};

/// Live runtime anchor: a snapshot plus the receipt naming it.
///
/// Holds:
/// - `receipt_id` — `Cid<TextReceipt>` of `receipt`. Cached so anchored
///   follow-ups can reference this state without re-hashing.
/// - `receipt` — content commitment naming what this state is.
/// - `snapshot` — live backend tensors for resuming a session.
#[derive(Debug, Clone)]
pub struct TextState<B: Backend> {
    receipt_id: Cid<TextReceipt>,
    receipt: TextReceipt,
    snapshot: Snapshot<B>,
}

impl<B: Backend> TextState<B> {
    pub(crate) fn new(receipt: TextReceipt, snapshot: Snapshot<B>) -> Self {
        Self {
            receipt_id: receipt.id(),
            receipt,
            snapshot,
        }
    }

    pub fn receipt_id(&self) -> Cid<TextReceipt> {
        self.receipt_id
    }

    pub fn receipt(&self) -> &TextReceipt {
        &self.receipt
    }

    pub fn position(&self) -> usize {
        self.receipt.position
    }

    pub fn allocated(&self) -> usize {
        self.snapshot.allocated()
    }

    pub(crate) fn snapshot(&self) -> &Snapshot<B> {
        &self.snapshot
    }
}

/// Post-prefill text decoder.
///
/// The only constructor is [`BoundProgramText::prefill`]. From that
/// point only [`Self::next_token`], [`Self::commit_next`],
/// [`Self::advance_one`], [`Self::position`], and
/// [`Self::into_text_state`] are available — multi-token input is
/// structurally impossible.
#[derive(Debug)]
pub struct TextDecoder<B: Backend> {
    session: Session<B>,
    backend: B,
    initial_receipt_id: Cid<TextReceipt>,
    position: usize,
    max_sequence_length: usize,
    extra_nat_chunk_size: Option<usize>,
    /// The model's predicted next token at the current position. Set by
    /// `prefill` (over the input) and updated after each step. Always
    /// matches what the model would produce for `position`.
    cached_next_token: u32,
}

impl<B: Backend> TextDecoder<B> {
    /// Peek at the predicted next token without committing. Useful for
    /// stop-token checking before deciding whether to call
    /// [`Self::commit_next`].
    pub const fn next_token(&self) -> u32 {
        self.cached_next_token
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    /// Live state tensors for diagnostic / testing use. Production
    /// callers should consume the decoder via [`Self::into_text_state`]
    /// and use the resulting receipt's `final_state` CID instead.
    pub fn state(&self) -> &[interpreter::Value<B>] {
        self.session.state()
    }

    /// CID of the [`TextReceipt`] this decoder was prefilled against.
    /// Useful for chain auditing.
    pub fn initial_receipt_id(&self) -> Cid<TextReceipt> {
        self.initial_receipt_id
    }

    /// Emit the cached predicted token and advance state by it.
    /// Returns the emitted token (the same value [`Self::next_token`]
    /// would have returned just before this call). After return, the
    /// decoder is fully receipt-aligned: state has consumed the emitted
    /// token, `cached_next_token` is the new prediction.
    ///
    /// This is the autoregressive workhorse for the decode loop.
    pub fn commit_next(&mut self) -> Result<u32> {
        let emitted = self.cached_next_token;
        self.advance_one(emitted)?;
        Ok(emitted)
    }

    /// Advance state by an arbitrary token (teacher-forced). Updates
    /// `cached_next_token` to the model's prediction at the new
    /// position. Used by tests and any speculative-decoding path that
    /// might land later.
    ///
    /// For autoregressive generation, prefer [`Self::commit_next`].
    pub fn advance_one(&mut self, token: u32) -> Result<u32> {
        self.cached_next_token = self.step_tokens(&[token])?;
        Ok(self.cached_next_token)
    }

    /// Consume the decoder into a [`TextState`] suitable for anchoring a
    /// follow-up execution. `commitment_id` is the
    /// [`Cid<TextExecution>`] of the commitment that drove this decoder
    /// (computed by the caller via `TextExecution::new` before
    /// `prefill`). `output_tokens` is the slice of emitted tokens
    /// accumulated by the runner.
    pub fn into_text_state(
        self,
        commitment_id: Cid<TextExecution>,
        output_tokens: &[u32],
    ) -> Result<TextState<B>> {
        let final_state = self.session.snapshot().cid(&self.backend);
        let output_tokens_cid =
            u32_tensor_cid(Shape(vec![1, output_tokens.len()]), output_tokens);
        let receipt = TextReceipt {
            commitment_id,
            position: self.position,
            final_state,
            output_tokens: output_tokens_cid,
        };
        Ok(TextState::new(receipt, self.session.into_snapshot()))
    }

    fn step_tokens(&mut self, tokens: &[u32]) -> Result<u32> {
        debug_assert!(!tokens.is_empty(), "step_tokens with empty input");

        let next_position = self.position.checked_add(tokens.len()).ok_or_else(|| {
            LLMError::InvalidTextSessionState("text session position overflow".to_string())
        })?;
        if next_position > self.max_sequence_length {
            return Err(LLMError::InvalidTextSessionState(format!(
                "text session position {next_position} exceeds max_sequence_length {}",
                self.max_sequence_length
            )));
        }

        let token_tensor =
            interpreter::tensor(&self.backend, Shape(vec![1, tokens.len()]), tokens.to_vec())
                .map_err(|error| {
                    LLMError::Runtime(RuntimeError::ExecutionError(format!(
                        "failed to build text input tensor: {error:?}"
                    )))
                })?;

        let mut inputs = vec![token_tensor];
        inputs.extend(self.session.state().iter().cloned());
        inputs.push(interpreter::Value::Nat(self.max_sequence_length));
        if let Some(chunk_size) = self.extra_nat_chunk_size {
            inputs.push(interpreter::Value::Nat(tokens.len().div_ceil(chunk_size)));
        }

        let mut outputs = self.session.run(inputs)?;
        let token = extract_next_token(&self.backend, &mut outputs)?;
        self.position = next_position;
        Ok(token)
    }
}

fn extract_next_token<B: Backend>(
    backend: &B,
    outputs: &mut Vec<interpreter::Value<B>>,
) -> Result<u32> {
    if outputs.len() != 1 {
        return Err(RuntimeError::UnexpectedProgramOutput(format!(
            "text program returned {} non-state outputs, expected 1",
            outputs.len()
        ))
        .into());
    }

    match outputs.remove(0) {
        interpreter::Value::Tensor(tensor) => match backend.to_vec(tensor) {
            interpreter::TaggedVec::U32(tokens) => tokens.last().copied().ok_or_else(|| {
                RuntimeError::UnexpectedProgramOutput(
                    "text program returned an empty token tensor".to_string(),
                )
                .into()
            }),
            other => Err(RuntimeError::UnexpectedProgramOutput(format!(
                "text program returned {other:?}, expected u32 token tensor"
            ))
            .into()),
        },
        other => Err(RuntimeError::UnexpectedProgramOutput(format!(
            "text program returned {other:?}, expected token tensor"
        ))
        .into()),
    }
}

/// Extension trait that adds text-generation operations to a
/// [`BoundProgram`]. Two methods, both with single concerns:
///
/// - [`Self::genesis_text_state`] materializes the cold-start anchor:
///   the live state for "I haven't run anything against this bind yet."
/// - [`Self::prefill`] starts a generation: consumes a starting state
///   and an input tensor, runs the prefill in one batched call, returns
///   a [`TextDecoder`] for any single-token continuation.
pub trait BoundProgramText<B: Backend>: Sized {
    fn genesis_text_state(&self) -> TextState<B>;
    fn prefill(
        self: Arc<Self>,
        initial_state: &TextState<B>,
        input_tensor: &interpreter::Value<B>,
    ) -> Result<TextDecoder<B>>;
}

impl<B: Backend> BoundProgramText<B> for BoundProgram<B> {
    fn genesis_text_state(&self) -> TextState<B> {
        let snapshot = self.empty_snapshot();
        let final_state: Cid<SnapshotBundle> = snapshot.cid(&self.interpreter().backend);
        let output_tokens: Cid<Tensor> = u32_tensor_cid(Shape(vec![1, 0]), &[]);
        let commitment_id = TextExecution::genesis(
            self.program().id(),
            self.parameters().to_vec(),
        )
        .id();
        let receipt = TextReceipt {
            commitment_id,
            position: 0,
            final_state,
            output_tokens,
        };
        TextState::new(receipt, snapshot)
    }

    fn prefill(
        self: Arc<Self>,
        initial_state: &TextState<B>,
        input_tensor: &interpreter::Value<B>,
    ) -> Result<TextDecoder<B>> {
        let backend = self.interpreter().backend.clone();
        let (_shape, tokens_vec) = validate_token_input(input_tensor, &backend)?;

        let starting_position = initial_state.position();
        let max_sequence_length = self.program().max_sequence_length();
        let extra_nat_chunk_size = self.program().extra_nat_chunk_size();
        let initial_receipt_id = initial_state.receipt_id();
        let session = self.start(initial_state.snapshot().clone())?;

        let mut decoder = TextDecoder {
            session,
            backend,
            initial_receipt_id,
            position: starting_position,
            max_sequence_length,
            extra_nat_chunk_size,
            cached_next_token: 0,
        };

        decoder.cached_next_token = decoder.step_tokens(&tokens_vec)?;
        Ok(decoder)
    }
}

impl TextExecution {
    /// Build a [`TextExecution::Step`] commitment from a bound program
    /// and the live values that will drive a `prefill`. Computes the
    /// CIDs for `parameters` and `input_tokens` from the bound program
    /// and the supplied tensor; the policy CID is computed via
    /// [`TextPolicy::id`].
    ///
    /// `initial_state` is the receipt id naming the state to resume
    /// from — callers with a live [`TextState<B>`] obtain it via
    /// [`TextState::receipt_id`]; callers driving execution remotely
    /// (where there's no live state in process) pass the
    /// [`Cid<TextReceipt>`] directly.
    ///
    /// Validates `input_tensor` to the same contract `prefill` enforces
    /// (u32 tensor, shape `[1, n]` with `n > 0`); errors with
    /// [`LLMError::InvalidTextSessionState`] if violated, so callers
    /// fail at commitment time rather than later at run time.
    pub fn new<B: Backend>(
        bound: &BoundProgram<B>,
        initial_state: Cid<TextReceipt>,
        input_tensor: &interpreter::Value<B>,
        policy: &TextPolicy,
    ) -> Result<Self> {
        let backend = &bound.interpreter().backend;
        let (shape, tokens_vec) = validate_token_input(input_tensor, backend)?;
        let input_tokens = u32_tensor_cid(shape, &tokens_vec);
        Ok(Self::Step {
            program: bound.program().id(),
            parameters: bound.parameters().to_vec(),
            initial_state,
            input_tokens,
            policy: policy.id(),
        })
    }
}

/// Shared validation for the input tensor handed to both
/// [`BoundProgramText::prefill`] and [`TextExecution::new`]. Decodes
/// to the raw `Vec<u32>` so callers can hash or step it without re-
/// invoking `backend.to_vec`.
fn validate_token_input<B: Backend>(
    input_tensor: &interpreter::Value<B>,
    backend: &B,
) -> Result<(Shape, Vec<u32>)> {
    let interpreter::Value::Tensor(tensor) = input_tensor else {
        return Err(LLMError::InvalidTextSessionState(
            "input must be a tensor value".to_string(),
        ));
    };
    if tensor.dtype() != catgrad::category::core::Dtype::U32 {
        return Err(LLMError::InvalidTextSessionState(format!(
            "input tensor must be u32, got {:?}",
            tensor.dtype()
        )));
    }
    let shape = tensor.shape();
    if shape.0.len() != 2 || shape.0[0] != 1 {
        return Err(LLMError::InvalidTextSessionState(format!(
            "input tensor must have shape [1, n], got {:?}",
            shape
        )));
    }
    if shape.0[1] == 0 {
        return Err(LLMError::InvalidTextSessionState(
            "input requires at least one token".to_string(),
        ));
    }
    let tokens_vec: Vec<u32> = match backend.to_vec(tensor.clone()) {
        interpreter::TaggedVec::U32(values) => values,
        other => {
            return Err(LLMError::InvalidTextSessionState(format!(
                "input tensor decoded to non-u32: {other:?}"
            )));
        }
    };
    Ok((shape, tokens_vec))
}
