use crate::core::{NdArrayType, Operation, PrimitiveType, Shape, Term, Var};
use open_hypergraphs::lax::var::operation;
use std::cell::RefCell;
use std::rc::Rc;

type Builder = Rc<RefCell<Term>>;

fn mat_mul_output_type(f: &PrimitiveType, g: &PrimitiveType) -> PrimitiveType {
    assert_eq!(f.dtype, g.dtype);
    let n = f.shape.0.len();
    let m = g.shape.0.len();
    assert_eq!(f.shape.0[n - 1], g.shape.0[m - 2]);

    let mut shape = f.shape.0[..n - 1].to_vec();
    shape.push(g.shape.0[m - 1]);
    NdArrayType {
        shape: Shape(shape),
        dtype: f.dtype,
    }
}

/// Batch matrix multiply two batches of matrices
///
/// - `f : N × A × B`
/// - `g : N × B × C`
/// - `mat_mul(builder, f, g) : N × A × C`
pub fn mat_mul(builder: &Builder, f: Var, g: Var) -> Var {
    let output_type: PrimitiveType = mat_mul_output_type(&f.label, &g.label);

    let n = f.label.shape.0.len();
    let m = g.label.shape.0.len();
    assert_eq!(n, m);

    // let batch = f.label.shape.0[..n - 2].to_vec();
    // let a = f.label.shape.0[n - 2];
    let b = f.label.shape.0[n - 1];
    let b_prime = g.label.shape.0[m - 2];
    // let c = g.label.shape.0[m - 1];

    assert_eq!(b, b_prime);

    let op: Operation = Operation::MatrixMultiply;
    operation(builder, &[f, g], output_type, op)
}

#[test]
fn test_matmul() {
    use crate::backend::cpu::eval::EvalState;
    use crate::backend::cpu::ndarray::{NdArray, TaggedNdArray};
    use crate::core::Dtype;
    let type_a = NdArrayType {
        shape: Shape(vec![1, 2]),
        dtype: Dtype::F32,
    };

    let type_b = NdArrayType {
        shape: Shape(vec![2, 3]),
        dtype: Dtype::F32,
    };

    let state = Rc::new(RefCell::new(Term::empty()));

    {
        let a = Var::new(state.clone(), type_a.clone());
        let b = Var::new(state.clone(), type_b.clone());

        let c = mat_mul(&state, a.clone(), b.clone());

        state.borrow_mut().sources = vec![a.new_source(), b.new_source()];
        state.borrow_mut().targets = vec![c.new_target()];
    }
    let f = Rc::try_unwrap(state).unwrap().into_inner();

    // a (1×2) matrix
    let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));
    // a (2×3) matrix
    let y = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));
    // result should be a 1×3 result
    let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));
    crate::backend::cpu::kernel::batch_matmul::<f32>(&x, &y, &mut expected);

    let mut state = EvalState::from_lax(f);

    let [actual] = state.eval_with(vec![x.into(), y.into()])[..] else {
        panic!("unexpected coarity at eval time")
    };

    let tagged: TaggedNdArray = expected.into();
    assert_eq!(&tagged, actual);
}
