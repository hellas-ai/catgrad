//! Text-generation extensions composed over catgrad's generic runtime.
//!
//! Catgrad's runtime knows how to bind a [`Program`] to a set of
//! materialized [`Parameters`] and run it as a [`Session`]. This module adds the layer
//! needed for autoregressive text generation: a typestate-enforced
//! decoder, a live runtime anchor with a content-addressed receipt,
//! and the commitment / receipt types that name an execution and its
//! outcome.
//!
//! # Submodules
//!
//! - `text_decoder`   — [`TextState`], [`TextDecoder`], and the
//!   [`BoundProgramText`] extension trait. The decoder is the only
//!   public post-prefill type; it has no `prefill` method, so
//!   multi-token mid-decode calls are structurally impossible.
//! - `text_execution` — [`TextPolicy`], [`TextExecution`] (the
//!   commitment, with `Genesis` and `Step` variants), and
//!   [`TextReceipt`] (the post-execution content commitment).
//! - `text_program`   — free-function constructor that builds a
//!   text-generation [`Program`] from a HuggingFace-style config JSON.
//!
//! # Vocabulary
//!
//! - **Commitment** = `Cid<TextExecution>`. Input-addressed (computable
//!   before running). Plays the role of a Nix `.drv` hash.
//! - **Receipt** = `Cid<TextReceipt>`. Content-addressed over the
//!   executor's actual output bytes (final state CID + output tokens
//!   CID + position + commitment id). The executor's binding promise.
//!
//! Both can ride the wire as 32-byte CIDs without revealing pre-images;
//! resolution is auth-gated.
//!
//! [`Program`]: catgrad::runtime::Program
//! [`Parameters`]: catgrad::interpreter::Parameters
//! [`Session`]: catgrad::runtime::Session

mod text_decoder;
mod text_execution;
mod text_program;

pub use text_decoder::{BoundProgramText, TextDecoder, TextState};
pub use text_execution::{TextExecution, TextPolicy, TextReceipt};
pub use text_program::text_program_from_config;
