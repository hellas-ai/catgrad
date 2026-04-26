//! Text-generation commitment + receipt types.
//!
//! Three content-addressed objects, each playing a distinct role:
//!
//! - [`TextPolicy`] — call-time generation behaviour (max new tokens,
//!   stop token ids), normalized so semantically equal policies hash to
//!   the same bytes.
//!
//! - [`TextExecution`] — the request *commitment*. A content-addressed
//!   recipe naming exactly what is to be (or was) executed: program,
//!   parameter bundle, starting receipt, input tokens, policy. Two
//!   variants distinguish the chain root from interior nodes:
//!     * `Genesis { program, parameters }` — the trivial no-op execution
//!       that produces a program's empty starting state. No input, no
//!       policy. Deterministic per `(program, parameters)`.
//!     * `Step { program, parameters, initial_state, input_tokens, policy }`
//!       — every real execution. Names its starting state by receipt
//!       CID; chain navigates upward via `initial_state`.
//!
//!   `Cid<TextExecution>` is the *commitment* — input-addressed,
//!   computable before any model code runs. Used as the exact-replay
//!   cache key and as the audit anchor ("the executor commits to having
//!   run exactly this").
//!
//! - [`TextReceipt`] — the executor's *post-execution* commitment to
//!   what they actually produced: a flat struct over commitment_id,
//!   position, final state CID, output tokens CID. One struct, no
//!   variants — Genesis-vs-Step is encoded structurally in the
//!   referenced commitment.
//!
//!   `Cid<TextReceipt>` is what follow-up requests reference as their
//!   `initial_state`. Receipts are the executor's binding promise:
//!   "I ran commitment X and produced these specific outputs." Under FP
//!   non-determinism two honest executors may produce different
//!   receipts for the same commitment; both are valid records of their
//!   respective executions.
//!
//! Both objects reveal only CIDs externally — pre-images (actual
//! tensors, token bytes) are auth-gated and resolved separately when
//! callers need them.

use catgrad::cid::{Cid, DagCborEncoder, SnapshotBundle, Tensor};
use catgrad::runtime::Program;

const TEXT_POLICY_SCHEMA: &str = "hellas.text_policy.v1";
const TEXT_EXECUTION_GENESIS_SCHEMA: &str = "hellas.text_execution.genesis.v1";
const TEXT_EXECUTION_STEP_SCHEMA: &str = "hellas.text_execution.step.v1";
const TEXT_RECEIPT_SCHEMA: &str = "hellas.text_receipt.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPolicy {
    max_new_tokens: u32,
    stop_token_ids: Vec<i32>,
}

impl TextPolicy {
    pub fn new(max_new_tokens: u32, mut stop_token_ids: Vec<i32>) -> Self {
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        Self {
            max_new_tokens,
            stop_token_ids,
        }
    }

    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    pub fn stop_token_ids(&self) -> &[i32] {
        &self.stop_token_ids
    }

    pub fn id(&self) -> Cid<TextPolicy> {
        Cid::from_dag_cbor_bytes(&self.to_dag_cbor_bytes())
    }

    pub fn to_dag_cbor_bytes(&self) -> Vec<u8> {
        let mut encoder = DagCborEncoder::new();
        encoder.array(3);
        encoder.str(TEXT_POLICY_SCHEMA);
        encoder.u64(self.max_new_tokens as u64);
        encoder.array(self.stop_token_ids.len() as u64);
        for token in &self.stop_token_ids {
            encoder.i64(*token as i64);
        }
        encoder.into_bytes()
    }
}

/// Request commitment. See module docs for the role distinction
/// between commitment and receipt.
///
/// `parameters` is the per-tensor list (not an aggregate hash) so that
/// auditors / blob-availability checkers can read the individual CIDs
/// without a pre-image fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextExecution {
    /// The trivial no-op execution that produces a program's empty
    /// starting state. Deterministic per `(program, parameters)` — any
    /// honest implementor synthesizes the same CID.
    Genesis {
        program: Cid<Program>,
        parameters: Vec<Cid<Tensor>>,
    },
    /// Every real execution. Anchored on a previous receipt (or, for
    /// cold starts at the gateway level, on the synthesized genesis
    /// receipt for `(program, parameters)`).
    Step {
        program: Cid<Program>,
        parameters: Vec<Cid<Tensor>>,
        initial_state: Cid<TextReceipt>,
        input_tokens: Cid<Tensor>,
        policy: Cid<TextPolicy>,
    },
}

impl TextExecution {
    /// Build the [`TextExecution::Genesis`] commitment for a
    /// `(program, parameters)` pair. Result is deterministic — two
    /// honest implementors with the same inputs produce the same CID.
    pub fn genesis(program: Cid<Program>, parameters: Vec<Cid<Tensor>>) -> Self {
        Self::Genesis {
            program,
            parameters,
        }
    }

    /// Build a [`TextExecution::Step`] commitment from already-computed
    /// CIDs. Higher-level builders (see `BoundProgramText`) handle the
    /// CID computation from live values.
    pub fn step(
        program: Cid<Program>,
        parameters: Vec<Cid<Tensor>>,
        initial_state: Cid<TextReceipt>,
        input_tokens: Cid<Tensor>,
        policy: Cid<TextPolicy>,
    ) -> Self {
        Self::Step {
            program,
            parameters,
            initial_state,
            input_tokens,
            policy,
        }
    }

    pub fn id(&self) -> Cid<TextExecution> {
        Cid::from_dag_cbor_bytes(&self.to_dag_cbor_bytes())
    }

    pub fn to_dag_cbor_bytes(&self) -> Vec<u8> {
        let mut encoder = DagCborEncoder::new();
        match self {
            Self::Genesis {
                program,
                parameters,
            } => {
                encoder.array(3);
                encoder.str(TEXT_EXECUTION_GENESIS_SCHEMA);
                encoder.bytes(program.as_bytes());
                encoder.array(parameters.len() as u64);
                for tensor_cid in parameters {
                    encoder.bytes(tensor_cid.as_bytes());
                }
            }
            Self::Step {
                program,
                parameters,
                initial_state,
                input_tokens,
                policy,
            } => {
                encoder.array(6);
                encoder.str(TEXT_EXECUTION_STEP_SCHEMA);
                encoder.bytes(program.as_bytes());
                encoder.array(parameters.len() as u64);
                for tensor_cid in parameters {
                    encoder.bytes(tensor_cid.as_bytes());
                }
                encoder.bytes(initial_state.as_bytes());
                encoder.bytes(input_tokens.as_bytes());
                encoder.bytes(policy.as_bytes());
            }
        }
        encoder.into_bytes()
    }
}

/// Post-execution commitment: the executor's binding promise of what
/// state and output they produced for a specific [`TextExecution`].
///
/// Flat struct, no variants. The Genesis-vs-Step distinction lives on
/// the referenced commitment; from a receipt CID alone you cannot tell
/// (without resolving the commitment) whether it's a genesis receipt or
/// a real-execution receipt — which is the privacy-preserving default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextReceipt {
    pub commitment_id: Cid<TextExecution>,
    pub position: usize,
    pub final_state: Cid<SnapshotBundle>,
    pub output_tokens: Cid<Tensor>,
}

impl TextReceipt {
    pub fn id(&self) -> Cid<TextReceipt> {
        Cid::from_dag_cbor_bytes(&self.to_dag_cbor_bytes())
    }

    pub fn to_dag_cbor_bytes(&self) -> Vec<u8> {
        let mut encoder = DagCborEncoder::new();
        encoder.array(5);
        encoder.str(TEXT_RECEIPT_SCHEMA);
        encoder.bytes(self.commitment_id.as_bytes());
        encoder.u64(self.position as u64);
        encoder.bytes(self.final_state.as_bytes());
        encoder.bytes(self.output_tokens.as_bytes());
        encoder.into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::{TextExecution, TextPolicy, TextReceipt};
    use catgrad::cid::{Cid, SnapshotBundle, Tensor};
    use catgrad::runtime::Program;

    fn dummy_program() -> Cid<Program> {
        Cid::<Program>::from_bytes([7; 32])
    }
    fn dummy_params() -> Vec<Cid<Tensor>> {
        vec![Cid::<Tensor>::from_bytes([3; 32])]
    }
    fn dummy_receipt() -> Cid<TextReceipt> {
        Cid::<TextReceipt>::from_bytes([5; 32])
    }
    fn dummy_tensor(b: u8) -> Cid<Tensor> {
        Cid::<Tensor>::from_bytes([b; 32])
    }
    fn dummy_snapshot(b: u8) -> Cid<SnapshotBundle> {
        Cid::<SnapshotBundle>::from_bytes([b; 32])
    }

    #[test]
    fn text_policy_canonicalizes_stop_ids() {
        let a = TextPolicy::new(16, vec![2, 1, 2]);
        let b = TextPolicy::new(16, vec![1, 2]);
        assert_eq!(a.stop_token_ids(), &[1, 2]);
        assert_eq!(a.id(), b.id());
    }

    #[test]
    fn genesis_commitment_is_deterministic_per_program_and_parameters() {
        let a = TextExecution::genesis(dummy_program(), dummy_params()).id();
        let b = TextExecution::genesis(dummy_program(), dummy_params()).id();
        assert_eq!(a, b);
    }

    #[test]
    fn genesis_and_step_with_matching_program_have_distinct_commitments() {
        let policy = TextPolicy::new(1, vec![]).id();
        let genesis = TextExecution::genesis(dummy_program(), dummy_params()).id();
        let step = TextExecution::step(
            dummy_program(),
            dummy_params(),
            dummy_receipt(),
            dummy_tensor(1),
            policy,
        )
        .id();
        assert_ne!(genesis, step);
    }

    #[test]
    fn step_commitment_changes_when_input_tokens_change() {
        let policy = TextPolicy::new(1, vec![]).id();
        let a = TextExecution::step(
            dummy_program(),
            dummy_params(),
            dummy_receipt(),
            dummy_tensor(1),
            policy,
        )
        .id();
        let b = TextExecution::step(
            dummy_program(),
            dummy_params(),
            dummy_receipt(),
            dummy_tensor(2),
            policy,
        )
        .id();
        assert_ne!(a, b);
    }

    #[test]
    fn receipt_id_changes_when_output_tokens_change() {
        let commitment = TextExecution::genesis(dummy_program(), dummy_params()).id();
        let final_state = dummy_snapshot(9);
        let a = TextReceipt {
            commitment_id: commitment,
            position: 0,
            final_state,
            output_tokens: dummy_tensor(1),
        }
        .id();
        let b = TextReceipt {
            commitment_id: commitment,
            position: 0,
            final_state,
            output_tokens: dummy_tensor(2),
        }
        .id();
        assert_ne!(a, b);
    }
}
