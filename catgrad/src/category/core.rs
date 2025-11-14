//! Core operations on shapes, natural numbers, and tensors.
//! A simple, portable IR.

use crate::definition::Def;
use crate::path::Path;
use open_hypergraphs::lax::OpenHypergraph;

////////////////////////////////////////////////////////////////////////////////
// Basic types.

// a core::Term is an open hypergraph with adjoined definitions named by Paths
pub type Term = OpenHypergraph<Object, Def<Path, Operation>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdArrayType {
    pub dtype: Dtype,
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    U32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Product of extents
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    /// Compute contiguous strides for a shape
    pub fn contiguous_strides(&self) -> Vec<isize> {
        let mut strides: Vec<isize> = vec![1];
        for dim in self.0.iter().skip(1).rev() {
            strides.push(strides.last().unwrap() * (*dim as isize));
        }
        strides.reverse();
        strides
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

////////////////////////////////////////////////////////////////////////////////
// objects

/// Objects of the category.
/// Note that Nat and Rank-1 shapes are only isomorphic so we can safely index by naturals.
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum Object {
    Nat, // natural numbers
    Dtype,
    NdArrayType,
    Shape,
    Tensor,
}

impl std::fmt::Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

////////////////////////////////////////////////////////////////////////////////
// Operations

use crate::category::lang;

/// Operations are those of core, extended with operations on shapes
#[derive(Debug, PartialEq, Clone)]
pub enum Operation {
    Type(TypeOp),
    Nat(NatOp),
    DtypeConstant(Dtype),
    Tensor(TensorOp),
    Copy,

    /// Load a tensor from the environment.
    // TODO: remove!
    Load(lang::Path),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum NatOp {
    Constant(usize),

    // Multiply n naturals
    Mul,

    // Add n naturals
    Add,
}

/// Operations involving shapes and dtypes
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum TypeOp {
    /// Pack k Nats into a shape
    /// Pack : Nat^k → Type
    Pack,

    /// Split a shape into nat dimensions
    /// Unpack : Type → Nat^k
    Unpack,

    /// Get the shape of a tensor (not its dtype!)
    /// Tensor → Shape
    Shape,

    /// Get the dtype of a tensor
    /// Tensor → Dtype
    Dtype,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    F32(f32),
    U32(u32),
}

/// Generating tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    /// Lift a scalar operation `f : m → n` to `m` input and `n` output arrays.
    /// `Map_f : S₀ ● ..m.. ● S_m → S₀ ● ..n.. ● Sn`
    Map(ScalarOp),

    /// `NatToU32 : Nat → Tensor ()` turns a Nat into a (scalar) tensor.
    NatToU32,

    /// Cast a tensor to a dtype
    /// `Tensor × Dtype → Tensor`
    Cast,

    /// Batch matrix multiplication
    /// `MatMul : (N, A, B) ● (N, B, C) → (N, A, C)`
    MatMul,

    /// A tensor with shape () having a single value.
    Scalar(Scalar),

    /// Sum last dimension of a tensor
    /// `Sum : Tensor → Tensor`
    Sum,

    /// Max last dimension of a tensor
    /// `Max : Tensor → Tensor`
    Max,

    /// Argmax last dimension of a tensor
    /// `Argmax : Tensor → Tensor`
    Argmax,

    /// Return the top-k values and their indices along the last dimension
    /// `TopK : Tensor × Nat → Tensor × Tensor`
    TopK,

    // broadcast a Tensor to a given shape
    Broadcast,

    /// Reshape a tensor into an isomorphic shape
    Reshape,

    /// Transpose two axes of the input
    /// `Transpose : Tensor × Dim × Dim → Tensor`
    Transpose,

    /// Slice a tensor along a dimension
    /// `Slice : Tensor × Dim × Start × Len → Tensor`
    Slice,

    /// Concatenate two tensors along a dimension
    /// `Concat : Tensor × Tensor × Dim → Tensor`
    Concat,

    /// Create a 1-D tensor with values from 0 to end (exclusive)
    /// `Arange : End → Tensor`
    Arange,

    // Slice using an index along a dimension
    // `Index: Tensor × Dim × Indices → Tensor`
    // Tensor: input tensor
    // Dim: dimension to slice along
    // Indices: 1-D tensor of indices to pick, they can be unordered and repeated
    // Output will be a tensor with the shape same as the input's except for the dimension being sliced where it is the length of the indices tensor
    Index,
}

/// For now, we assume that every Dtype defines a ring & has comparisons
/// TODO: constants, comparisons
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum ScalarOp {
    Add, // 2 → 1
    Sub, // 2 → 1
    Mul, // 2 → 1
    Div, // 2 → 1
    Neg, // 1 → 1
    Pow, // 2 → 1
    LT,  // 2 → 1
    EQ,  // 2 → 1
    Cos, // 1 → 1
    Sin, // 1 → 1
}

impl ScalarOp {
    /// The *profile* of an operation is the pair of its arity and coarity.
    pub fn profile(&self) -> (usize, usize) {
        match self {
            ScalarOp::Add => (2, 1),
            ScalarOp::Sub => (2, 1),
            ScalarOp::Mul => (2, 1),
            ScalarOp::Div => (2, 1),
            ScalarOp::Neg => (1, 1),
            ScalarOp::Pow => (2, 1),
            ScalarOp::LT => (2, 1),
            ScalarOp::EQ => (2, 1),
            ScalarOp::Cos => (1, 1),
            ScalarOp::Sin => (1, 1),
        }
    }
}
