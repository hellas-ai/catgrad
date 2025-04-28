use open_hypergraphs::prelude::*;

use super::object::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    /// Pointwise composition of N matrices `x_i : A ⇒ B` with `y_i : B ⇒ C`
    /// for `i ∈ N`.
    ///
    MatrixMultiply,

    /// Const value
    Const(f32),

    /// Max value across last dimension
    Max,

    /// Sum value across last dimension
    Sum,

    /// Broadcast a value to one of shape n+x.
    Broadcast(Shape),

    /// Reshape a value
    Reshape,

    /// Transpose (swap) two dimensions of a tensor
    Transpose { dim0: usize, dim1: usize },

    /// Create a copy
    Copy,

    /// Pointwise addition of two values of similar shapes
    Add,

    /// Pointwise subtraction of two values of similar shapes
    Sub,

    /// Pointwise multiplication of two values of similar shapes
    Mul,

    /// Pointwise division of two values of similar shapes
    Div,

    /// Pointwise raising to power of two values of similar shapes
    Pow,

    /// Pointwise negation of value
    Negate,

    /// Inputs injected at runtime (model parameters)
    Parameter(String),

    /// Embedding lookup
    Embedding,

    /// Arange
    Arange,
}

pub type Term = open_hypergraphs::lax::OpenHypergraph<PrimitiveType, Operation>;
pub type StrictTerm = OpenHypergraph<PrimitiveType, Operation>;

impl Operation {
    // Make an OpenHypergraph from an operation, sources and targets
    pub fn term(op: Operation, s: Vec<NdArrayType>, t: Vec<NdArrayType>) -> Term {
        open_hypergraphs::lax::OpenHypergraph::singleton(op, s, t)
    }

    pub fn identity(t: Type) -> Term {
        open_hypergraphs::lax::OpenHypergraph::identity(t)
    }

    // Make an OpenHypergraph for the MatrixMultiply operation
    pub fn matmul(n: Shape, a: usize, b: usize, c: usize, dtype: Dtype) -> Term {
        let source0 = NdArrayType {
            shape: &n + &a + &b,
            dtype,
        };

        let source1 = NdArrayType {
            shape: &n + &b + &c,
            dtype,
        };

        let target = NdArrayType {
            shape: &n + &a + &c,
            dtype,
        };

        Operation::term(
            Operation::MatrixMultiply,
            vec![source0, source1],
            vec![target],
        )
    }

    // Make an OpenHypergraph for the Broadcast operation
    pub fn broadcast(x: NdArrayType, shape: Shape) -> Term {
        let source = x.clone();
        let target = NdArrayType {
            shape: shape.clone(),
            dtype: x.dtype,
        };
        let op = Operation::Broadcast(shape);
        Operation::term(op, vec![source], vec![target])
    }

    // Make an OpenHypergraph for the Transpose operation
    pub fn transpose(x: NdArrayType, dim0: usize, dim1: usize) -> Term {
        assert!(
            dim0 < x.shape.0.len(),
            "Transpose dimension dim0 invalid: {dim0}"
        );
        assert!(
            dim1 < x.shape.0.len(),
            "Transpose dimension dim1 invalid: {dim0}"
        );
        let source = x.clone();

        // Create new shape with swapped dimensions
        let mut new_shape = x.shape.0.clone();
        new_shape.swap(dim0, dim1);

        let target = NdArrayType {
            shape: Shape(new_shape),
            dtype: x.dtype,
        };

        let op = Operation::Transpose { dim0, dim1 };
        Operation::term(op, vec![source], vec![target])
    }

    // Make an OpenHypergraph for the Reshape operation
    pub fn reshape(x: NdArrayType, shape: Shape) -> Term {
        assert_eq!(
            x.size(),
            shape.size(),
            "Reshape from {:?} to {:?} must preserve total size.",
            x.shape,
            shape
        );
        let source = x.clone();
        let target = NdArrayType {
            shape: shape.clone(),
            dtype: x.dtype,
        };
        let op = Operation::Reshape;
        Operation::term(op, vec![source], vec![target])
    }

    // Make an OpenHypergraph for the Copy operation
    pub fn copy(x: NdArrayType) -> Term {
        let op = Operation::Copy;
        Operation::term(op, vec![x.clone()], vec![x.clone(), x.clone()])
    }

    // Make an OpenHypergraph for the given operation
    fn reduceop(x: NdArrayType, op: Operation) -> Term {
        let source = x.clone();
        let target = NdArrayType {
            shape: Shape(x.shape.0[..x.shape.0.len() - 1].to_vec()),
            dtype: x.dtype,
        };
        Operation::term(op, vec![source], vec![target])
    }

    // Make an OpenHypergraph for the given operation
    fn unop(x: NdArrayType, op: Operation) -> Term {
        Operation::term(op, vec![x.clone()], vec![x.clone()])
    }

    // Make an OpenHypergraph for the Parameter operation
    pub fn parameter(x: NdArrayType, name: &str) -> Term {
        let op = Operation::Parameter(name.to_string());
        Operation::term(op, vec![], vec![x.clone()])
    }

    // Make an OpenHypergraph for the Const operation
    pub fn constop(x: NdArrayType, k: f32) -> Term {
        let op = Operation::Const(k);
        Operation::term(op, vec![], vec![x.clone()])
    }

    // Make an OpenHypergraph for the Negate operation
    pub fn negate(x: NdArrayType) -> Term {
        Operation::unop(x, Operation::Negate)
    }

    // Make an OpenHypergraph for the given binary operation
    fn binop(x: NdArrayType, op: Operation) -> Term {
        Operation::term(op, vec![x.clone(), x.clone()], vec![x.clone()])
    }

    // Make an OpenHypergraph for the Add operation
    pub fn add(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Add)
    }

    // Make an OpenHypergraph for the Sub operation
    pub fn sub(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Sub)
    }

    // Make an OpenHypergraph for the Mul operation
    pub fn mul(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Mul)
    }

    // Make an OpenHypergraph for the Div operation
    pub fn div(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Div)
    }

    // Make an OpenHypergraph for the Pow operation
    pub fn pow(x: NdArrayType) -> Term {
        Operation::binop(x, Operation::Pow)
    }

    // Make an OpenHypergraph for a Sum operation
    pub fn sum(x: NdArrayType) -> Term {
        Operation::reduceop(x, Operation::Sum)
    }

    // Make an OpenHypergraph for a Max operation
    pub fn max(x: NdArrayType) -> Term {
        Operation::reduceop(x, Operation::Max)
    }
}

use open_hypergraphs::lax::var;
pub type Var = var::Var<PrimitiveType, Operation>;

impl var::HasVar for Operation {
    fn var() -> Self {
        Operation::Copy
    }
}

impl var::HasAdd<PrimitiveType, Operation> for Operation {
    fn add(_lhs: PrimitiveType, rhs: PrimitiveType) -> (PrimitiveType, Operation) {
        (rhs, Operation::Add)
    }
}

impl var::HasSub<PrimitiveType, Operation> for Operation {
    fn sub(_lhs: PrimitiveType, rhs: PrimitiveType) -> (PrimitiveType, Operation) {
        (rhs, Operation::Sub)
    }
}

impl var::HasMul<PrimitiveType, Operation> for Operation {
    fn mul(_lhs: PrimitiveType, rhs: PrimitiveType) -> (PrimitiveType, Operation) {
        (rhs, Operation::Mul)
    }
}

impl var::HasDiv<PrimitiveType, Operation> for Operation {
    fn div(_lhs: PrimitiveType, rhs: PrimitiveType) -> (PrimitiveType, Operation) {
        (rhs, Operation::Div)
    }
}

impl var::HasNeg<PrimitiveType, Operation> for Operation {
    fn neg(x: PrimitiveType) -> (PrimitiveType, Operation) {
        (x, Operation::Negate)
    }
}
