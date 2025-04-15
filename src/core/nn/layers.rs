use crate::core::{Dtype, NdArrayType, Operation, PrimitiveType, Shape, Term, Var};
use open_hypergraphs::lax::var::operation;
use std::cell::RefCell;
use std::f32::consts::{E, PI};
use std::rc::Rc;

pub type Builder = Rc<RefCell<Term>>;

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

    let op = Operation::MatrixMultiply;
    operation(builder, &[f, g], output_type, op)
}

pub fn parameter(builder: &Builder, param_type: NdArrayType, name: String) -> Var {
    let op = Operation::Parameter(name);
    operation(builder, &[], param_type, op)
}

pub fn constant(builder: &Builder, param_type: NdArrayType, k: f32) -> Var {
    let op = Operation::Const(k);
    operation(builder, &[], param_type, op)
}

pub fn broadcast(builder: &Builder, n: Shape, x: Var) -> Var {
    let in_t = x.label.clone();
    let out_t = &n + &in_t;
    let op = Operation::Broadcast(n);
    operation(builder, &[x.clone()], out_t, op)
}

pub fn power(builder: &Builder, base: Var, power: Var) -> Var {
    let op = Operation::Pow;
    operation(builder, &[base.clone(), power.clone()], base.label, op)
}

pub fn reduceop(builder: &Builder, op: Operation, x: Var) -> Var {
    let source = x.label.clone();
    let target = NdArrayType {
        shape: Shape(source.shape.0[..source.shape.0.len() - 1].to_vec()),
        dtype: source.dtype,
    };
    operation(builder, &[x.clone()], target, op)
}

pub fn sum(builder: &Builder, x: Var) -> Var {
    reduceop(builder, Operation::Sum, x)
}

pub fn max(builder: &Builder, x: Var) -> Var {
    reduceop(builder, Operation::Max, x)
}

pub fn transpose(builder: &Builder, dim0: usize, dim1: usize, x: Var) -> Var {
    let in_t = x.label.clone();

    // Create new shape with swapped dimensions
    let mut new_shape = in_t.shape.0.clone();
    new_shape.swap(dim0, dim1);

    let out_t = NdArrayType {
        shape: Shape(new_shape),
        dtype: in_t.dtype,
    };
    let op = Operation::Transpose { dim0, dim1 };
    operation(builder, &[x.clone()], out_t, op)
}

pub fn linear(
    builder: &Builder,
    input_features: usize,
    output_features: usize,
    dtype: Dtype,
    name: &str,
    x: Var,
) -> Var {
    // let batch_size = 1;
    let w_type = NdArrayType {
        shape: Shape(vec![output_features, input_features]),
        dtype,
    };
    // Bias
    let b_type = NdArrayType {
        shape: Shape(vec![output_features]),
        dtype,
    };

    let w = parameter(builder, w_type.clone(), format!("{name}.weight"));
    let b = parameter(builder, b_type.clone(), format!("{name}.bias"));

    let b_b = broadcast(builder, Shape(vec![1]), b);
    let w_t = transpose(builder, 0, 1, w);
    mat_mul(builder, x, w_t) + b_b
}

pub fn sigmoid(builder: &Builder, x: Var) -> Var {
    let one = constant(builder, x.label.clone(), 1.0);
    let e = constant(builder, x.label.clone(), E);

    one.clone() / (one + power(builder, e, -x))
}

pub fn tanh(builder: &Builder, x: Var) -> Var {
    let one = constant(builder, x.label.clone(), 1.0);
    let two = constant(builder, x.label.clone(), 2.0);

    two.clone() * sigmoid(builder, two * x) - one
}

// approx GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(builder: &Builder, x: Var) -> Var {
    let c = constant(builder, x.label.clone(), f32::sqrt(2. / PI));
    let one = constant(builder, x.label.clone(), 1.0);
    let three = constant(builder, x.label.clone(), 3.0);
    let half = constant(builder, x.label.clone(), 0.5);
    let k = constant(builder, x.label.clone(), 0.044715);

    half * x.clone() * (one + tanh(builder, c * (x.clone() + k * (power(builder, x, three)))))
}

#[cfg(test)]
mod test {
    use super::{gelu, linear, mat_mul, sigmoid, tanh, Builder};
    use crate::backend::cpu::eval::EvalState;
    use crate::backend::cpu::ndarray::{NdArray, TaggedNdArray};
    use crate::core::{Dtype, NdArrayType, Shape, Term, Var};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn test_activation<F>(x: &[f32], exp: &[f32], act: F)
    where
        F: Fn(&Builder, Var) -> Var,
    {
        let shape = Shape(vec![1, x.len()]);
        let in_type = NdArrayType {
            shape: shape.clone(),
            dtype: Dtype::F32,
        };

        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            let result = act(&builder, x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let x = NdArray::new(x.to_vec(), shape);

        let f = Rc::try_unwrap(builder).unwrap().into_inner();
        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual.approx(6), exp);
    }

    #[test]
    fn test_tanh() {
        test_activation(&[1.0, 2.0, 3.0], &[0.761594, 0.964028, 0.995055], tanh);
    }

    #[test]
    fn test_gelu() {
        test_activation(&[1.0, 2.0, 3.0], &[0.841192, 1.954598, 2.996363], gelu);
    }

    #[test]
    fn test_sigmoid() {
        test_activation(&[1.0, 2.0, 3.0], &[0.731059, 0.880797, 0.952574], sigmoid);
    }

    #[test]
    fn test_linear() {
        let in_type = NdArrayType {
            shape: Shape(vec![1, 3]),
            dtype: Dtype::F32,
        };
        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            // Run linear layer (x * w^T + b)
            let result = linear(&builder, 3, 2, Dtype::F32, "l", x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        let mut state = EvalState::from_lax(f);

        // Create test input data
        let x = NdArray::new(vec![1.0, 2.0, 3.0], Shape(vec![1, 3]));

        // Parameter values
        let w = NdArray::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Shape(vec![2, 3]));
        let b = NdArray::new(vec![0.1, 0.1], Shape(vec![2]));

        let mut parameters = HashMap::new();
        parameters.insert("l.weight".to_string(), w.into());
        parameters.insert("l.bias".to_string(), b.into());

        state.set_parameters(parameters);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        // Check result shape
        match actual {
            TaggedNdArray::F32(arr) => {
                assert_eq!(arr.shape.0, vec![1, 2]);
                assert_eq!(arr.data, vec![1.5, 3.3]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_matmul() {
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
}
