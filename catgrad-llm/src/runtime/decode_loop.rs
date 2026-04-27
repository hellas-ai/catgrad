//! Generic decode loop that drives a [`TextDecoder`] until a stop
//! condition is reached.
//!
//! Replaces the hand-rolled "for `max_new_tokens`, peek, check stop,
//! commit, callback" pattern that consumers were writing in three
//! places (`examples/serve.rs::Engine::generate`, node executor's
//! `runner::run_decode_loop`, and the older `examples/llama` decode
//! path). Each consumer now layers its own concerns — detokenization,
//! cancellation, batching — through the per-token callback.

use std::ops::ControlFlow;
use thiserror::Error;

use catgrad::interpreter::backend::Backend;

use super::text_decoder::TextDecoder;

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeOutcome {
    /// The model emitted a token in the stop-tokens list. The decoder
    /// is positioned just before that token (it was peeked but not
    /// committed) so the caller can append a `into_text_state`
    /// receipt without including the stop token.
    EndOfSequence,
    /// The `max_new_tokens` budget was exhausted.
    MaxTokens,
    /// The per-token callback returned `ControlFlow::Break(BreakReason::StopSequence)`
    /// — the callback detected a stop condition (e.g. detokenizer
    /// matched a configured stop string) that the parser-level
    /// stop-token list couldn't catch.
    StopSequence,
    /// The per-token callback returned `ControlFlow::Break(BreakReason::Cancelled)`
    /// — usually because an external cancellation signal fired.
    Cancelled,
}

/// Reason the per-token callback wants to break the loop. Distinguishes
/// stop-sequence detection (a normal end of generation) from
/// cancellation (an external interrupt). The mapping to `DecodeOutcome`
/// is one-to-one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakReason {
    StopSequence,
    Cancelled,
}

/// Errors `run_decode` can produce. Distinguishes the decoder's own
/// failures from the callback's so callers can route each
/// appropriately.
#[derive(Debug, Error)]
pub enum DecodeLoopError<E> {
    /// The decoder itself failed (out-of-range position, runtime
    /// error, etc.). Caller treats as 5xx-class.
    #[error(transparent)]
    Decoder(crate::LLMError),
    /// The per-token callback returned an error. Caller decides what
    /// it means in their domain (e.g. SSE channel closed).
    #[error(transparent)]
    Sink(E),
}

/// Drive one autoregressive generation step at a time, yielding each
/// emitted token to `on_token`, until a stop condition is reached.
///
/// The callback returns:
/// - `Ok(ControlFlow::Continue(()))` to keep going,
/// - `Ok(ControlFlow::Break(BreakReason::StopSequence))` for a
///   stop condition detected by the callback (e.g. a Detokenizer
///   matched a configured stop string),
/// - `Ok(ControlFlow::Break(BreakReason::Cancelled))` for external
///   cancellation,
/// - `Err(E)` to abort with a sink error.
///
/// Returns `(generated, outcome)`. `generated` is the count of
/// tokens passed to `on_token` — does NOT include any stop token
/// detected at peek time via `stop_tokens` (those are not committed
/// and not yielded).
pub fn run_decode<B, F, E>(
    decoder: &mut TextDecoder<B>,
    max_new_tokens: u32,
    stop_tokens: &[i32],
    mut on_token: F,
) -> std::result::Result<(u32, DecodeOutcome), DecodeLoopError<E>>
where
    B: Backend,
    F: FnMut(u32) -> std::result::Result<ControlFlow<BreakReason>, E>,
{
    let mut generated = 0u32;
    for _ in 0..max_new_tokens {
        let predicted = decoder.next_token();
        if is_stop_token(predicted, stop_tokens) {
            return Ok((generated, DecodeOutcome::EndOfSequence));
        }
        let emitted = decoder
            .commit_next()
            .map_err(DecodeLoopError::Decoder)?;
        debug_assert_eq!(emitted, predicted);
        generated += 1;
        match on_token(emitted).map_err(DecodeLoopError::Sink)? {
            ControlFlow::Continue(()) => {}
            ControlFlow::Break(BreakReason::StopSequence) => {
                return Ok((generated, DecodeOutcome::StopSequence));
            }
            ControlFlow::Break(BreakReason::Cancelled) => {
                return Ok((generated, DecodeOutcome::Cancelled));
            }
        }
    }
    Ok((generated, DecodeOutcome::MaxTokens))
}

fn is_stop_token(token: u32, stop_tokens: &[i32]) -> bool {
    i32::try_from(token)
        .ok()
        .is_some_and(|t| stop_tokens.contains(&t))
}
