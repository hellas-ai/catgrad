#![cfg(any())]
//! Test that operations can be lowered to text and mlir-opt will accept them.
use catgrad::prelude::*;
use catgrad::stdlib::ops::IntoDtypeVar;
use catgrad::typecheck::*;
use catgrad::{category::lang, stdlib::ops};

use catgrad_mlir::compile::CompiledModel;

use open_hypergraphs::lax::{
    OpenHypergraph,
    var::{BuildResult, build},
};

//use crate::lower::preprocess;

use std::cell::RefCell;
use std::rc::Rc;

fn run_test(term: TypedTerm) {
    let symbol = path(vec!["test", "term"]).unwrap();

    // Make environment
    let mut env = stdlib();
    env.definitions.extend([(symbol.clone(), term)]);

    let params = typecheck::Parameters::from([]);

    // TODO: CompiledModel::new crashes on failure- fix!
    let _ = CompiledModel::new(&env, &params, symbol);
}

#[test]
fn test_shape_op() {
    // shape of *input* tensor
    let x_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(vec![3.into(), 1.into(), 4.into()]),
        dtype: Dtype::F32.into(),
    }));
    run_test(
        build_typed_term(
            [x_type],
            [shape_type(&[3, 1, 4])], // rank 3
            |builder, [x]| vec![ops::shape(builder, x)],
        )
        .unwrap(),
    );
}

#[test]
fn test_cast_u32_f32() {
    let s = vec![3, 1, 4];
    run_test(
        build_typed_term(
            [tensor_type(&s, Dtype::U32)],
            [tensor_type(&s, Dtype::F32)],
            |builder, [x]| {
                let d = Dtype::F32.to_dtype(builder);
                vec![ops::cast(builder, x, d)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_arange() {
    let s = vec![10];
    run_test(
        build_typed_term([], [tensor_type(&s, Dtype::U32)], |builder, []| {
            vec![ops::arange(builder, 10)]
        })
        .unwrap(),
    );
}

#[test]
fn test_shape_pack() {
    run_test(
        build_typed_term([], [shape_type(&[2, 3, 4])], |builder, []| {
            let n1 = 2.to_nat(builder);
            let n2 = 3.to_nat(builder);
            let n3 = 4.to_nat(builder);
            vec![ops::pack(builder, [n1, n2, n3])]
        })
        .unwrap(),
    );
}

#[test]
fn test_tensor_index() {
    let input_shape = vec![4, 3]; // 4x3 tensor
    let indices_shape = vec![2]; // 2 indices
    let output_shape = vec![2, 3]; // result: 2x3 tensor (indexed along dim 0)

    run_test(
        build_typed_term(
            [
                tensor_type(&input_shape, Dtype::F32),
                tensor_type(&indices_shape, Dtype::U32),
            ],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor, indices_tensor]| {
                let dim = 0.to_nat(builder); // Index along dimension 0
                vec![ops::index(builder, dim, indices_tensor, input_tensor)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_transpose() {
    let input_shape = vec![2, 3, 4]; // 2x3x4 tensor
    let output_shape = vec![2, 4, 3]; // transposed: 2x4x3 (dims 1 and 2 swapped)

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [x]| vec![ops::transpose(builder, 1, 2, x)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_reshape() {
    let input_shape = vec![2, 6]; // 2x6 tensor
    let output_shape = vec![3, 4]; // reshaped to 3x4 tensor

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor]| {
                // Create shape [3, 4]
                let dim1 = 3.to_nat(builder);
                let dim2 = 4.to_nat(builder);
                let new_shape = ops::pack(builder, [dim1, dim2]);
                vec![ops::reshape(builder, new_shape, input_tensor)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_shape_unpack() {
    run_test(
        build_typed_term(
            [shape_type(&[3, 1, 4])], // input: shape with rank 3
            [
                Type::Nat(3.into()),
                Type::Nat(1.into()),
                Type::Nat(4.into()),
            ], // output: 3 individual Nat values
            |builder, [shape]| ops::unpack::<3>(builder, shape).into(),
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_sum() {
    let input_shape = vec![2, 3, 4]; // 2x3x4 tensor
    let output_shape = vec![2, 3, 1]; // summed over final dim: 2x3x1 tensor (catgrad keeps rank)

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor]| vec![ops::sum(builder, input_tensor)],
        )
        .unwrap(),
    );
}

#[test]
fn test_nat_to_u32() {
    run_test(
        build_typed_term(
            [Type::Nat(3.into())],          // input: Nat value 3
            [tensor_type(&[], Dtype::U32)], // output: scalar tensor (empty shape)
            |builder, [nat_value]| vec![ops::nat_to_u32(builder, nat_value)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_sin() {
    let shape = vec![2, 3]; // 2x3 tensor

    run_test(
        build_typed_term(
            [tensor_type(&shape, Dtype::F32)],
            [tensor_type(&shape, Dtype::F32)],
            |builder, [input_tensor]| vec![ops::sin(builder, input_tensor)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_cos() {
    let shape = vec![2, 3]; // 2x3 tensor

    run_test(
        build_typed_term(
            [tensor_type(&shape, Dtype::F32)],
            [tensor_type(&shape, Dtype::F32)],
            |builder, [input_tensor]| vec![ops::cos(builder, input_tensor)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_concat() {
    let tensor1_shape = vec![2, 3]; // 2x3 tensor
    let tensor2_shape = vec![2, 4]; // 2x4 tensor
    let output_shape = vec![2, 7]; // concatenated along dim 1: 2x7 tensor

    run_test(
        build_typed_term(
            [
                tensor_type(&tensor1_shape, Dtype::F32),
                tensor_type(&tensor2_shape, Dtype::F32),
            ],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [tensor1, tensor2]| {
                let dim = 1.to_nat(builder); // Concatenate along dimension 1
                vec![ops::concat(builder, dim, tensor1, tensor2)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_matmul() {
    let lhs_shape = vec![4, 3, 5]; // (N=4, A=3, B=5)
    let rhs_shape = vec![4, 5, 7]; // (N=4, B=5, C=7)
    let output_shape = vec![4, 3, 7]; // (N=4, A=3, C=7)

    run_test(
        build_typed_term(
            [
                tensor_type(&lhs_shape, Dtype::F32),
                tensor_type(&rhs_shape, Dtype::F32),
            ],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [lhs, rhs]| vec![ops::matmul(builder, lhs, rhs)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_lt() {
    let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(vec![2.into(), NatExpr::Var(0), 3.into(), NatExpr::Var(0)]),
        dtype: Dtype::F32.into(),
    }));

    run_test(
        build_typed_term([ty.clone(), ty.clone()], [ty], |builder, [lhs, rhs]| {
            vec![ops::lt(builder, lhs, rhs)]
        })
        .unwrap(),
    );
}

#[test]
fn test_tensor_slice() {
    let input_shape = vec![2, 6, 4]; // 2x6x4 tensor
    let output_shape = vec![2, 3, 4]; // slice dim 1, start=2, len=3: 2x3x4 tensor

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor]| {
                let dim = 1.to_nat(builder); // Slice along dimension 1
                let start = 2.to_nat(builder); // Start at index 2
                let len = 3.to_nat(builder); // Take 3 elements
                vec![ops::slice(builder, dim, start, len, input_tensor)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_nat_mul() {
    run_test(
        build_typed_term(
            [Type::Nat(3.into()), Type::Nat(4.into())], // input: Nat values 3 and 4
            [Type::Nat(12.into())],                     // output: Nat value 12
            |_builder, [a, b]| vec![a * b],             // Use multiplication operator
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_max() {
    let input_shape = vec![2, 3, 4]; // 2x3x4 tensor
    let output_shape = vec![2, 3, 1]; // max over final dim: 2x3x1 tensor

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor]| vec![ops::max(builder, input_tensor)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_argmax() {
    let input_shape = vec![2, 3, 4]; // 2x3x4 tensor
    let output_shape = vec![2, 3, 1]; // argmax over final dim: 2x3x1 tensor (indices)

    run_test(
        build_typed_term(
            [tensor_type(&input_shape, Dtype::F32)],
            [tensor_type(&output_shape, Dtype::U32)], // output is indices (U32)
            |builder, [input_tensor]| vec![ops::argmax(builder, input_tensor)],
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_broadcast() {
    let input_shape = vec![3, 1]; // 3x1 tensor (to be broadcasted)
    let output_shape = vec![10, 3, 4]; // 10x3x4 tensor (broadcasted result)

    run_test(
        build_typed_term(
            [
                tensor_type(&input_shape, Dtype::F32),
                shape_type(&output_shape), // target shape
            ],
            [tensor_type(&output_shape, Dtype::F32)],
            |builder, [input_tensor, target_shape]| {
                vec![ops::broadcast(builder, input_tensor, target_shape)]
            },
        )
        .unwrap(),
    );
}

#[test]
fn test_tensor_broadcast_dynamic() {
    // Test broadcast with dynamic dimensions to verify tensor.empty() gets proper args
    let input_shape = vec![1]; // 1D tensor (to be broadcasted)

    let output_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(vec![NatExpr::Var(0), 4.into()]), // Dynamic x 4 tensor
        dtype: Dtype::F32.into(),
    }));

    run_test(
        build_typed_term(
            [
                tensor_type(&input_shape, Dtype::F32),
                shape_type(&[2, 4]), // target shape: 2x4
            ],
            [output_type],
            |builder, [input_tensor, target_shape]| {
                vec![ops::broadcast(builder, input_tensor, target_shape)]
            },
        )
        .unwrap(),
    );
}

////////////////////////////////////////////////////////////////////////////////
// Type helpers

fn tensor_type(shape: &[usize], dtype: Dtype) -> Type {
    Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: shape.to_owned().into(),
        dtype: dtype.into(),
    }))
}

fn shape_type(dims: &[usize]) -> Type {
    Type::Shape(dims.to_owned().into())
}

////////////////////////////////////////////////////////////////////////////////
// Building terms

// Forget type details (for constructing terms)
fn type_to_kind(t: Type) -> lang::Object {
    match t {
        Type::Nat(_) => lang::Object::Nat,
        Type::Dtype(_) => lang::Object::Dtype,
        Type::Shape(_) => lang::Object::Shape,
        Type::Type(_) => lang::Object::NdArrayType,
        Type::Tensor(_) => lang::Object::Tensor,
    }
}

fn build_typed_term<const ARITY: usize, const COARITY: usize, F>(
    source_type: [Type; ARITY],
    target_type: [Type; COARITY],
    f: F,
) -> Option<TypedTerm>
where
    F: Fn(&Rc<RefCell<OpenHypergraph<lang::Object, lang::Operation>>>, [Var; ARITY]) -> Vec<Var>,
{
    let source_kind = source_type.clone().map(type_to_kind);
    let term = build_typed(source_kind, f).ok()?;
    Some(TypedTerm {
        term,
        source_type: source_type.to_vec(),
        target_type: target_type.to_vec(),
    })
}

fn build_typed<const ARITY: usize, F>(
    source_types: [lang::Object; ARITY],
    f: F,
) -> BuildResult<lang::Object, lang::Operation>
where
    F: Fn(&Rc<RefCell<OpenHypergraph<lang::Object, lang::Operation>>>, [Var; ARITY]) -> Vec<Var>,
{
    use std::array;
    build(move |state| {
        // use from_fn to avoid having to clone source_types
        let sources: [Var; ARITY] =
            array::from_fn(|i| Var::new(state.clone(), source_types[i].clone()));

        let sources_vec = sources.iter().cloned().collect();
        let targets = f(state, sources);
        (sources_vec, targets)
    })
}
