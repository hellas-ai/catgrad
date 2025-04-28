use crate::core::{NdArrayType, Operation, PrimitiveType, Shape, Term, Var};
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

pub fn embedding(builder: &Builder, indices: Var, weights: Var) -> Var {
    let mut shape = indices.label.shape.0.clone();
    shape.push(weights.label.shape.0[1]);
    let out_type = NdArrayType {
        shape: Shape(shape),
        dtype: weights.label.dtype,
    };
    let op = Operation::Embedding;
    operation(builder, &[indices, weights], out_type, op)
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

pub fn expand(builder: &Builder, shape: Shape, x: Var) -> Var {
    let out_t = NdArrayType {
        shape: shape.clone(),
        dtype: x.label.dtype,
    };
    let op = Operation::Broadcast(shape);
    operation(builder, &[x.clone()], out_t, op)
}

pub fn reshape(builder: &Builder, shape: Shape, x: Var) -> Var {
    let out_t = NdArrayType {
        shape,
        dtype: x.label.dtype,
    };
    let op = Operation::Reshape;
    operation(builder, &[x.clone()], out_t, op)
}

pub fn power(builder: &Builder, base: Var, power: Var) -> Var {
    let op = Operation::Pow;
    operation(builder, &[base.clone(), power.clone()], base.label, op)
}

pub fn sqrt(builder: &Builder, x: Var) -> Var {
    let mh = constant(builder, x.label.clone(), 0.5);
    power(builder, x, mh)
}

pub fn exp(builder: &Builder, x: Var) -> Var {
    let e = constant(builder, x.label.clone(), E);
    power(builder, e, x)
}

pub fn reduceop(builder: &Builder, op: Operation, x: Var) -> Var {
    let source = x.label.clone();

    // keep the last dimension, set it to 1
    let mut target_shape = source.shape.0.clone();
    target_shape[source.shape.0.len() - 1] = 1;
    let target = NdArrayType {
        shape: Shape(target_shape),
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
    name: &str,
    x: Var,
) -> Var {
    let batch_size = x.label.shape.0[0];
    let w_type = NdArrayType {
        shape: Shape(vec![output_features, input_features]),
        dtype: x.label.dtype,
    };
    // Bias
    let b_type = NdArrayType {
        shape: Shape(vec![output_features]),
        dtype: x.label.dtype,
    };

    let w = parameter(builder, w_type.clone(), format!("{name}.weight"));
    let b = parameter(builder, b_type.clone(), format!("{name}.bias"));

    let b_b = broadcast(builder, Shape(vec![batch_size]), b);
    // let b_b = expand(builder, Shape(vec![batch_size, output_features]), b);
    let w_t = transpose(builder, 0, 1, w);
    mat_mul(builder, x, w_t) + b_b
}

pub fn sigmoid(builder: &Builder, x: Var) -> Var {
    let one = constant(builder, x.label.clone(), 1.0);

    one.clone() / (one + exp(builder, -x))
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

fn layernorm_raw(builder: &Builder, x: Var) -> Var {
    let n = x.label.shape.0[x.label.shape.0.len() - 1];

    let s = sum(builder, x.clone());
    let constn = constant(builder, s.label.clone(), n as f32);
    let mean = sum(builder, x.clone()) / constn.clone();
    let nom = x.clone() - expand(builder, x.label.shape.clone(), mean.clone());

    let var = sum(builder, nom.clone() * nom.clone()) / constn;
    let epsilon = constant(builder, var.label.clone(), 1e-5);
    let stddev = sqrt(builder, var + epsilon);
    let denom = expand(builder, x.label.shape, stddev);

    nom / denom
}

pub fn layernorm(builder: &Builder, name: &str, x: Var) -> Var {
    let shape = vec![x.label.shape.0[x.label.shape.0.len() - 1]];
    let t = NdArrayType {
        shape: Shape(shape),
        dtype: x.label.dtype,
    };
    let gamma = parameter(builder, t.clone(), format!("{name}.weight"));
    let beta = parameter(builder, t, format!("{name}.bias"));
    let lr = layernorm_raw(builder, x);
    let gamma = expand(builder, lr.label.shape.clone(), gamma);
    let beta = expand(builder, lr.label.shape.clone(), beta);
    lr * gamma + beta
}

fn rmsnorm_raw(builder: &Builder, x: Var) -> Var {
    let n = x.label.shape.0[x.label.shape.0.len() - 1];
    let s = sum(builder, x.clone() * x.clone());
    let constn = constant(builder, s.label.clone(), n as f32);
    let ms = sum(builder, x.clone() * x.clone()) / constn;
    let epsilon = constant(builder, ms.label.clone(), 1e-5);
    let rms = sqrt(builder, ms + epsilon);
    let b = expand(builder, x.label.shape.clone(), rms);

    x / b
}

// rmsnorm(x) = x / √(E[x²] + ε) × γ
pub fn rmsnorm(builder: &Builder, name: &str, x: Var) -> Var {
    let shape = vec![x.label.shape.0[x.label.shape.0.len() - 1]];
    let t = NdArrayType {
        shape: Shape(shape),
        dtype: x.label.dtype,
    };
    let gamma = parameter(builder, t.clone(), format!("{name}.weight"));
    let lr = rmsnorm_raw(builder, x);
    let gamma = expand(builder, lr.label.shape.clone(), gamma);
    lr * gamma
}

pub fn softmax(builder: &Builder, x: Var) -> Var {
    let m = max(builder, x.clone());
    let bmax = expand(builder, x.label.shape.clone(), m);
    let x = x - bmax;
    let ex = exp(builder, x.clone());
    let s = sum(builder, ex.clone());
    let bsum = expand(builder, x.label.shape.clone(), s);
    ex / bsum
}

#[cfg(test)]
mod test {
    use super::{
        gelu, layernorm_raw, linear, mat_mul, rmsnorm_raw, sigmoid, softmax, tanh, Builder,
    };
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
    fn test_softmax() {
        test_activation(&[1.0, 2.0, 3.0], &[0.090031, 0.244728, 0.665241], softmax);
        test_activation(
            &[100.1, 100.2, 100.3],
            &[0.300609, 0.332224, 0.367167],
            softmax,
        );
    }

    #[test]
    fn test_rmsnorm() {
        test_activation(
            &[0., 1., 2., 3., 4.],
            &[0.0, 0.408248, 0.816496, 1.224744, 1.632992],
            rmsnorm_raw,
        )
    }

    #[test]
    fn test_layernorm() {
        test_activation(
            &[0., 1., 2., 3., 4.],
            &[-1.414210, -0.707105, 0.000000, 0.707105, 1.414210],
            layernorm_raw,
        )
    }

    #[test]
    fn test_linear() {
        let in_type = NdArrayType {
            shape: Shape(vec![2, 3]),
            dtype: Dtype::F32,
        };
        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            // Run linear layer (x * w^T + b)
            let result = linear(&builder, 3, 2, "l", x.clone());

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        let mut state = EvalState::from_lax(f);

        // Create test input data
        let x = NdArray::new(vec![1.0, 2.0, 3.0, 2.0, 3.0, 1.0], Shape(vec![2, 3]));

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

        assert_eq!(actual.approx(6), &[1.5, 3.3, 1.2, 3.0]);
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
