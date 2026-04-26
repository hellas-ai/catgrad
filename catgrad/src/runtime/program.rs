//! The typechecked computation graph plus the metadata an interpreter
//! needs to evaluate it: state schema, max sequence-length budget, and an
//! optional per-step nat hint for chunked-recurrence models.
//!
//! [`ProgramSpec`] is the pure-data form — what gets serialized and
//! hashed. [`Program`] wraps a `ProgramSpec` with its cached canonical
//! id; the wrapper is immutable so the id can never go stale.

use crate::category::core::{Dtype, Shape};
use crate::category::lang::TypedTerm;
use crate::cid::{Cid, dag_cbor_bytes};
use crate::path::Path;
use crate::prelude::DynModule;

use super::error::{Result, RuntimeError};

/// Runtime ABI/schema tag included in every `Program` CID.
///
/// Bump this when runtime interpretation changes in a way that can affect
/// deterministic output without changing the serialized program fields. That
/// includes input ordering, load-op binding rules, state-output ordering,
/// empty-state initialization semantics, `extra_nat_chunk_size` meaning, or the
/// canonical encoding itself.
const PROGRAM_SCHEMA: &str = "hellas.program.v1";

/// The serializable, hashable content of a [`Program`]. Construct directly
/// when you have all fields in hand, then build a [`Program`] via
/// `program_spec.into()` (or [`Program::from`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgramSpec {
    pub typed_term: TypedTerm,
    pub module_path: Path,
    pub empty_state_type: Vec<(Dtype, Shape)>,
    pub max_sequence_length: usize,
    /// For models that need an extra per-step nat input computed as
    /// `seq_len.div_ceil(chunk_size)` (gated-delta chunking).
    /// `None` for models that don't need such an input.
    #[serde(default)]
    pub extra_nat_chunk_size: Option<usize>,
}

/// Schema-tagged envelope used both for hashing and for ser/de of
/// [`Program`]. Generic over the body so the field list lives in exactly
/// one place ([`ProgramSpec`]); the schema constant is added on the
/// wire so an ABI bump produces a different CID for the same body.
#[derive(serde::Serialize, serde::Deserialize)]
struct Tagged<T> {
    schema: String,
    #[serde(flatten)]
    body: T,
}

/// A bound computation graph plus the runtime metadata needed to interpret it.
///
/// `Program` is content-addressed via [`Program::id`], which hashes the
/// canonical DAG-CBOR encoding of the wrapped [`ProgramSpec`] together
/// with the [`PROGRAM_SCHEMA`] tag. The id is computed once at
/// construction (or at deserialize time) and is free to read.
#[derive(Debug, Clone)]
pub struct Program {
    spec: ProgramSpec,
    canonical_id: Cid<Program>,
}

impl From<ProgramSpec> for Program {
    fn from(spec: ProgramSpec) -> Self {
        let canonical_id = compute_canonical_id(&spec);
        Self { spec, canonical_id }
    }
}

impl Program {
    /// Build a [`Program`] from a module by extracting its typed term and
    /// wrapping it together with the supplied runtime metadata. Errors if
    /// the module fails to produce a typed term.
    pub fn from_module(
        module: &dyn DynModule,
        module_path: Path,
        empty_state_type: Vec<(Dtype, Shape)>,
        max_sequence_length: usize,
        extra_nat_chunk_size: Option<usize>,
    ) -> Result<Self> {
        let typed_term = module.term().ok_or_else(|| {
            RuntimeError::InvalidProgram("failed to build typed term from module".to_string())
        })?;
        Ok(ProgramSpec {
            typed_term,
            module_path,
            empty_state_type,
            max_sequence_length,
            extra_nat_chunk_size,
        }
        .into())
    }

    /// Canonical content id for this program. Free — computed once at
    /// construction. This identifies the graph/program only; it is not the
    /// full execution commitment (which a higher-level crate composes from
    /// `(Program::id, weights, inputs, generation policy, ...)`).
    pub fn id(&self) -> Cid<Program> {
        self.canonical_id
    }

    /// The underlying [`ProgramSpec`]. For ad-hoc field access; the
    /// common fields also have direct accessors (see [`Self::module_path`]
    /// etc.).
    pub fn spec(&self) -> &ProgramSpec {
        &self.spec
    }

    pub fn typed_term(&self) -> &TypedTerm {
        &self.spec.typed_term
    }

    pub fn module_path(&self) -> &Path {
        &self.spec.module_path
    }

    pub fn empty_state_type(&self) -> &[(Dtype, Shape)] {
        &self.spec.empty_state_type
    }

    pub fn max_sequence_length(&self) -> usize {
        self.spec.max_sequence_length
    }

    pub fn extra_nat_chunk_size(&self) -> Option<usize> {
        self.spec.extra_nat_chunk_size
    }

    pub fn to_dag_cbor_bytes(&self) -> Result<Vec<u8>> {
        dag_cbor_bytes(&Tagged {
            schema: PROGRAM_SCHEMA.to_string(),
            body: &self.spec,
        })
        .map_err(|err| {
            RuntimeError::InvalidProgram(format!("program DAG-CBOR encoding failed: {err}"))
        })
    }
}

fn compute_canonical_id(spec: &ProgramSpec) -> Cid<Program> {
    let bytes = dag_cbor_bytes(&Tagged {
        schema: PROGRAM_SCHEMA.to_string(),
        body: spec,
    })
    .expect("ProgramSpec is structural; DAG-CBOR encoding cannot fail");
    Cid::from_dag_cbor_bytes(&bytes)
}

impl serde::Serialize for Program {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Tagged {
            schema: PROGRAM_SCHEMA.to_string(),
            body: &self.spec,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Program {
    /// Deserialize a Program, validate its schema tag, and compute the
    /// canonical id eagerly. The schema must match [`PROGRAM_SCHEMA`];
    /// otherwise we'd silently load a v1 blob into v2 code (or vice
    /// versa) and produce meaningless results.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let Tagged::<ProgramSpec> { schema, body } = Tagged::deserialize(deserializer)?;
        if schema != PROGRAM_SCHEMA {
            return Err(serde::de::Error::custom(format!(
                "unknown program schema {schema:?}, expected {PROGRAM_SCHEMA:?}"
            )));
        }
        Ok(Program::from(body))
    }
}
