use super::types::*;
use crate::path::Path;
use open_hypergraphs::lax::var;

macro_rules! path{
    [$($x:expr),* $(,)?] => {
        $crate::path::path(vec![$($x),*]).expect("invalid operation name")
    };
}

macro_rules! op {
    [$($x:expr),* $(,)?] => {
        Operation::Declaration(path![$($x),*])
    };
}

////////////////////////////////////////////////////////////////////////////////
// Types

// Copy lets us use HasVar
impl var::HasVar for Operation {
    fn var() -> Self {
        op!["cartesian", "copy"]
    }
}

impl var::HasAdd<Object, Operation> for Operation {
    fn add(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        match lhs {
            Object::Nat => (lhs, op!["nat", "add"]),
            Object::Tensor => (lhs, op!["tensor", "add"]),
            _ => panic!("addition undefined for {lhs:?}"),
        }
    }
}

impl var::HasSub<Object, Operation> for Operation {
    fn sub(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        match lhs {
            Object::Tensor => (lhs, op!["tensor", "sub"]),
            _ => panic!("subtraction undefined for {lhs:?}"),
        }
    }
}

impl var::HasMul<Object, Operation> for Operation {
    fn mul(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        // NOTE: this is a bit of a hack- we explicitly treat nat/tensor muls differently.
        // Would be better to have proper polymorphic ops
        match lhs {
            Object::Nat => (lhs, op!["nat", "mul"]),
            Object::Tensor => (lhs, op!["tensor", "mul"]),
            _ => panic!("multiply undefined for {lhs:?}"),
        }
    }
}

impl var::HasDiv<Object, Operation> for Operation {
    fn div(lhs: Object, rhs: Object) -> (Object, Operation) {
        assert_eq!(lhs, rhs);
        (lhs, op!["tensor", "div"])
    }
}

impl var::HasNeg<Object, Operation> for Operation {
    fn neg(operand_type: Object) -> (Object, Operation) {
        (operand_type, op!["tensor", "neg"])
    }
}

pub fn pow(builder: &Builder, value: Var, exponent: Var) -> Var {
    var::fn_operation(
        builder,
        &[value, exponent],
        Object::Tensor,
        op!["tensor", "pow"],
    )
}

pub fn sin(builder: &Builder, value: Var) -> Var {
    var::fn_operation(builder, &[value], Object::Tensor, op!["tensor", "sin"])
}

pub fn cos(builder: &Builder, value: Var) -> Var {
    var::fn_operation(builder, &[value], Object::Tensor, op!["tensor", "cos"])
}

////////////////////////////////////////////////////////////////////////////////
// Declarations

/// For a param named a.b.c, create an op named param.a.b.c.
pub fn param(builder: &Builder, path: &Path) -> Var {
    var::fn_operation(
        builder,
        &[],
        Object::Tensor,
        Operation::Declaration(path.clone()),
    )
}

pub fn lit(builder: &Builder, lit: Literal) -> Var {
    let obj = match lit {
        Literal::Nat(_) => Object::Nat,
        Literal::Dtype(_) => Object::Nat,
        Literal::U32(_) | Literal::F32(_) => Object::Tensor,
    };
    var::fn_operation(builder, &[], obj, Operation::Literal(lit))
}

/// Pack a fixed number of Nat values into a specific shape
pub fn pack<const N: usize>(builder: &Builder, extents: [Var; N]) -> Var {
    for x in &extents {
        assert_eq!(x.label, Object::Nat);
    }

    var::fn_operation(builder, &extents, Object::Shape, op!["shape", "pack"])
}

/// Unpack a shape into a dtype and its constituent Nat dimensions
pub fn unpack<const N: usize>(builder: &Builder, x: Var) -> [Var; N] {
    assert_eq!(x.label, Object::Shape);

    let ty = vec![Object::Nat; N];
    let elements = var::operation(builder, &[x], ty, op!["shape", "unpack"]);

    crate::util::iter_to_array(elements.into_iter()).expect("N elements")
}

// Tensor → Shape
pub fn shape(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::Shape, op!["tensor", "shape"])
}

pub fn dtype_constant(builder: &Builder, dtype: Dtype) -> Var {
    var::fn_operation(
        builder,
        &[],
        Object::Dtype,
        Operation::Literal(Literal::Dtype(dtype)),
    )
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Helpers

pub fn broadcast(builder: &Builder, x: Var, s: Var) -> Var {
    var::fn_operation(builder, &[x, s], Object::Tensor, op!["tensor", "broadcast"])
}

pub fn expand(builder: &Builder, x: Var, s: Var) -> Var {
    broadcast(builder, x, s)
}

pub fn reshape(builder: &Builder, t: Var, x: Var) -> Var {
    var::fn_operation(builder, &[t, x], Object::Tensor, op!["tensor", "reshape"])
}

pub fn transpose(builder: &Builder, dim0: Var, dim1: Var, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x, dim0, dim1],
        Object::Tensor,
        op!["tensor", "transpose"],
    )
}

pub fn index(builder: &Builder, dim: Var, idx: Var, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x, dim, idx],
        Object::Tensor,
        op!["tensor", "index"],
    )
}

pub fn slice(builder: &Builder, dim: Var, start: Var, len: Var, x: Var) -> Var {
    var::fn_operation(
        builder,
        &[x, dim, start, len],
        Object::Tensor,
        op!["tensor", "slice"],
    )
}

pub fn concat(builder: &Builder, dim: Var, x: Var, y: Var) -> Var {
    var::fn_operation(
        builder,
        &[x, y, dim],
        Object::Tensor,
        op!["tensor", "concat"],
    )
}

pub fn arange(builder: &Builder, end: Var) -> Var {
    var::fn_operation(builder, &[end], Object::Tensor, op!["tensor", "arange"])
}

pub fn max(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::Tensor, op!["tensor", "max"])
}

pub fn sum(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::Tensor, op!["tensor", "sum"])
}

pub fn argmax(builder: &Builder, x: Var) -> Var {
    var::fn_operation(builder, &[x], Object::Tensor, op!["tensor", "argmax"])
}

pub fn topk(builder: &Builder, k: Var, x: Var) -> (Var, Var) {
    assert_eq!(k.label, Object::Nat);
    assert_eq!(x.label, Object::Tensor);

    let outputs = var::operation(
        builder,
        &[x, k],
        vec![Object::Tensor, Object::Tensor],
        op!["tensor", "topk"],
    );
    let values = outputs[0].clone();
    let indices = outputs[1].clone();
    (values, indices)
}

pub fn nat_to_u32(builder: &Builder, nat: Var) -> Var {
    var::fn_operation(builder, &[nat], Object::Tensor, op!["tensor", "nat_to_u32"])
}

pub fn lt(builder: &Builder, x: Var, y: Var) -> Var {
    var::fn_operation(builder, &[x, y], Object::Tensor, op!["tensor", "lt"])
}

pub fn eq(builder: &Builder, x: Var, y: Var) -> Var {
    var::fn_operation(builder, &[x, y], Object::Tensor, op!["tensor", "eq"])
}

/// Batch matmul
pub fn matmul(builder: &Builder, f: Var, g: Var) -> Var {
    // checked during shapechecking, but errors easier to follow here.
    assert_eq!(f.label, Object::Tensor);
    assert_eq!(g.label, Object::Tensor);

    var::fn_operation(builder, &[f, g], Object::Tensor, op!["tensor", "matmul"])
}

pub fn cast(builder: &Builder, x: Var, dtype: Var) -> Var {
    assert_eq!(x.label, Object::Tensor);
    assert_eq!(dtype.label, Object::Dtype);
    var::fn_operation(builder, &[x, dtype], Object::Tensor, op!["tensor", "cast"])
}

pub fn dtype(builder: &Builder, x: Var) -> Var {
    assert_eq!(x.label, Object::Tensor);
    var::fn_operation(builder, &[x], Object::Dtype, op!["tensor", "dtype"])
}
