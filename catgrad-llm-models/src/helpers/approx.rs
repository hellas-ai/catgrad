/// Transcendental functions built via polynomial approximation from primitive ops
/// These are mostly examples of possible deterministic implementations that do not rely
/// on native math libraries with platform-dependent non-reproducible outputs
///
/// softmax and sincos can be dropped in existing attention and RoPE code but in this
/// unoptimized unfused form incur a performance penalty.
use super::where_broadcast;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

/// Constants used by polynomial approximation and range reduction
const HALF_PI: f32 = std::f32::consts::PI / 2.0;
const HALF_PI_HEAD: f32 = f32::from_bits(0x3fc90000);
const HALF_PI_TAIL: f32 = f32::from_bits(0x39fdaa22);
const SIN_C3: f32 = -1.0 / 6.0;
const SIN_C5: f32 = 1.0 / 120.0;
const SIN_C7: f32 = -1.0 / 5040.0;
const COS_C2: f32 = -1.0 / 2.0;
const COS_C4: f32 = 1.0 / 24.0;
const COS_C6: f32 = -1.0 / 720.0;
const INV_LN2: f32 = std::f32::consts::LOG2_E;
const LN2_HEAD: f32 = 0.693_145_75;
const LN2_TAIL: f32 = 1.428_606_8e-6;
const EXP_OVERFLOW_CUTOFF: f32 = 88.722_84;
const EXP_UNDERFLOW_CUTOFF: f32 = -103.972_08;
const EXP_C2: f32 = 1.0 / 2.0;
const EXP_C3: f32 = 1.0 / 6.0;
const EXP_C4: f32 = 1.0 / 24.0;
const EXP_C5: f32 = 1.0 / 120.0;
const EXP_C6: f32 = 1.0 / 720.0;
const EXP_C7: f32 = 1.0 / 5040.0;
const EXP_C8: f32 = 1.0 / 40320.0;

// Simple Taylor expansion for sin()
fn sin_taylor(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let x2 = x.clone() * x.clone();

    let one = constant(builder, 1.0, &sh);
    let c3 = constant(builder, SIN_C3, &sh);
    let c5 = constant(builder, SIN_C5, &sh);
    let c7 = constant(builder, SIN_C7, &sh);

    let poly = one + x2.clone() * (c3 + x2.clone() * (c5 + x2 * c7));
    x * poly
}

// Simple Taylor expansion for cos()
fn cos_taylor(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let x2 = x.clone() * x;

    let one = constant(builder, 1.0, &sh);
    let c2 = constant(builder, COS_C2, &sh);
    let c4 = constant(builder, COS_C4, &sh);
    let c6 = constant(builder, COS_C6, &sh);

    one + x2.clone() * (c2 + x2.clone() * (c4 + x2 * c6))
}

// Cody-Waite reduction for sin/cos
fn reduce_half_pi(builder: &Builder, x: Var) -> (Var, Var) {
    let sh = shape(builder, x.clone());
    let p = constant(builder, HALF_PI, &sh);
    let q = round(builder, x.clone() / p);

    let p0 = constant(builder, HALF_PI_HEAD, &sh);
    let p1 = constant(builder, HALF_PI_TAIL, &sh);

    let xr = x - q.clone() * p0;
    let xr = xr - q.clone() * p1;

    let four = constant(builder, 4.0, &sh);
    let q_mod4 = q.clone() - four.clone() * floor(builder, q / four);
    (xr, q_mod4)
}

// Taylor series approximation for exp() on a reduced range
// A minimax polynomial would be more appropriate
fn exp_taylor(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let one = constant(builder, 1.0, &sh);
    let c2 = constant(builder, EXP_C2, &sh);
    let c3 = constant(builder, EXP_C3, &sh);
    let c4 = constant(builder, EXP_C4, &sh);
    let c5 = constant(builder, EXP_C5, &sh);
    let c6 = constant(builder, EXP_C6, &sh);
    let c7 = constant(builder, EXP_C7, &sh);
    let c8 = constant(builder, EXP_C8, &sh);

    one.clone()
        + x.clone()
            * (one
                + x.clone()
                    * (c2
                        + x.clone()
                            * (c3
                                + x.clone()
                                    * (c4
                                        + x.clone()
                                            * (c5
                                                + x.clone() * (c6 + x.clone() * (c7 + x * c8)))))))
}

fn reduce_ln2(builder: &Builder, x: Var) -> (Var, Var) {
    let sh = shape(builder, x.clone());
    let inv_ln2 = constant(builder, INV_LN2, &sh);
    let k = round(builder, x.clone() * inv_ln2);

    let ln2_head = constant(builder, LN2_HEAD, &sh);
    let ln2_tail = constant(builder, LN2_TAIL, &sh);

    let r = x - k.clone() * ln2_head;
    let r = r - k.clone() * ln2_tail;
    (r, k)
}

fn reconstruct_sin(builder: &Builder, q_mod4: Var, sin_r: Var, cos_r: Var) -> Var {
    let sh = shape(builder, q_mod4.clone());
    let q0 = eq(builder, q_mod4.clone(), constant(builder, 0.0, &sh));
    let q1 = eq(builder, q_mod4.clone(), constant(builder, 1.0, &sh));
    let q2 = eq(builder, q_mod4, constant(builder, 2.0, &sh));

    where_broadcast(
        builder,
        q0,
        sin_r.clone(),
        where_broadcast(
            builder,
            q1,
            cos_r.clone(),
            where_broadcast(builder, q2, -sin_r, -cos_r),
        ),
    )
}

fn reconstruct_cos(builder: &Builder, q_mod4: Var, sin_r: Var, cos_r: Var) -> Var {
    let sh = shape(builder, q_mod4.clone());
    let q0 = eq(builder, q_mod4.clone(), constant(builder, 0.0, &sh));
    let q1 = eq(builder, q_mod4.clone(), constant(builder, 1.0, &sh));
    let q2 = eq(builder, q_mod4, constant(builder, 2.0, &sh));

    where_broadcast(
        builder,
        q0,
        cos_r.clone(),
        where_broadcast(
            builder,
            q1,
            -sin_r.clone(),
            where_broadcast(builder, q2, -cos_r, sin_r),
        ),
    )
}

// sincos() returning both sin and cos since they reuse the reduction phase
// and are usually used in pairs in positional embeddings
pub fn sincos_approx(builder: &Builder, x: Var) -> (Var, Var) {
    let (xr, q_mod4) = reduce_half_pi(builder, x);
    let sin_r = sin_taylor(builder, xr.clone());
    let cos_r = cos_taylor(builder, xr);
    let sin = reconstruct_sin(builder, q_mod4.clone(), sin_r.clone(), cos_r.clone());
    let cos = reconstruct_cos(builder, q_mod4, sin_r, cos_r);
    (sin, cos)
}

pub fn sin_approx(builder: &Builder, x: Var) -> Var {
    let (sin, _) = sincos_approx(builder, x);
    sin
}

pub fn cos_approx(builder: &Builder, x: Var) -> Var {
    let (_, cos) = sincos_approx(builder, x);
    cos
}

// exp() approximation via range reduction, approximation and reconstruction
pub fn exp_approx(builder: &Builder, x: Var) -> Var {
    let sh = shape(builder, x.clone());
    let zero = constant(builder, 0.0, &sh);
    let one = constant(builder, 1.0, &sh);
    let pos_inf = constant(builder, f32::INFINITY, &sh);
    let neg_inf = constant(builder, f32::NEG_INFINITY, &sh);
    let is_pos_inf = eq(builder, x.clone(), pos_inf.clone());
    let is_neg_inf = eq(builder, x.clone(), neg_inf);
    let overflow_cutoff = constant(builder, EXP_OVERFLOW_CUTOFF, &sh);
    let underflow_cutoff = constant(builder, EXP_UNDERFLOW_CUTOFF, &sh);
    let overflows = gt(builder, x.clone(), overflow_cutoff);
    let underflows = lt(builder, x.clone(), underflow_cutoff);
    let is_positive_special = where_cond(builder, is_pos_inf, one.clone(), overflows);
    let is_negative_special = where_cond(builder, is_neg_inf, one, underflows);
    let safe_x = where_cond(
        builder,
        is_negative_special.clone(),
        zero.clone(),
        where_cond(builder, is_positive_special.clone(), zero.clone(), x),
    );

    let (r, k) = reduce_ln2(builder, safe_x);
    let exp_r = exp_taylor(builder, r);
    let two = constant(builder, 2.0, &sh);
    let approx = pow(builder, two, k) * exp_r;
    where_cond(
        builder,
        is_negative_special,
        zero,
        where_cond(builder, is_positive_special, pos_inf, approx),
    )
}

pub fn softmax_approx(builder: &Builder, x: Var) -> Var {
    let x_dtype = dtype(builder, x.clone());
    let x = cast(builder, x, Dtype::F32);
    let x_shape = shape(builder, x.clone());
    let m = max(builder, x.clone());
    let bmax = broadcast(builder, x_shape.clone(), m);
    let x = x - bmax;
    let ex = exp_approx(builder, x);
    let s = sum(builder, ex.clone());
    let bsum = broadcast(builder, x_shape, s);
    cast(builder, ex / bsum, x_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catgrad::interpreter::backend::Backend;
    use catgrad::interpreter::backend::ndarray::NdArrayBackend;
    use catgrad::interpreter::{Interpreter, Parameters, TaggedVec, Value, tensor};
    use catgrad::prelude::{Path, Shape, path};
    use catgrad::stdlib::{Module, stdlib};
    use catgrad::typecheck::{self, DtypeExpr, NatExpr, NdArrayType, ShapeExpr, Type, TypeExpr};

    const TEST_POINTS: [f32; 8] = [
        0.0,
        std::f32::consts::PI / 6.0,
        std::f32::consts::PI / 2.0,
        std::f32::consts::PI,
        3.0 * std::f32::consts::PI / 2.0,
        -std::f32::consts::PI / 2.0,
        -std::f32::consts::PI / 6.0,
        2.0,
    ];

    fn sin_taylor_scalar(x: f32) -> f32 {
        let x2 = x * x;
        x * (1.0 + x2 * (SIN_C3 + x2 * (SIN_C5 + x2 * SIN_C7)))
    }

    fn cos_taylor_scalar(x: f32) -> f32 {
        let x2 = x * x;
        1.0 + x2 * (COS_C2 + x2 * (COS_C4 + x2 * COS_C6))
    }

    fn reduce_half_pi_scalar(x: f32) -> (f32, f32) {
        let q = (x / HALF_PI).round_ties_even();
        let xr = x - q * HALF_PI_HEAD - q * HALF_PI_TAIL;
        let q_mod4 = q - 4.0 * (q / 4.0).floor();
        (xr, q_mod4)
    }

    fn reconstruct_sin_scalar(q_mod4: f32, sin_r: f32, cos_r: f32) -> f32 {
        if q_mod4 == 0.0 {
            sin_r
        } else if q_mod4 == 1.0 {
            cos_r
        } else if q_mod4 == 2.0 {
            -sin_r
        } else {
            -cos_r
        }
    }

    fn reconstruct_cos_scalar(q_mod4: f32, sin_r: f32, cos_r: f32) -> f32 {
        if q_mod4 == 0.0 {
            cos_r
        } else if q_mod4 == 1.0 {
            -sin_r
        } else if q_mod4 == 2.0 {
            -cos_r
        } else {
            sin_r
        }
    }

    fn sincos_approx_scalar(x: f32) -> (f32, f32) {
        let (xr, q_mod4) = reduce_half_pi_scalar(x);
        let sin_r = sin_taylor_scalar(xr);
        let cos_r = cos_taylor_scalar(xr);
        (
            reconstruct_sin_scalar(q_mod4, sin_r, cos_r),
            reconstruct_cos_scalar(q_mod4, sin_r, cos_r),
        )
    }

    fn sin_approx_scalar(x: f32) -> f32 {
        sincos_approx_scalar(x).0
    }

    fn cos_approx_scalar(x: f32) -> f32 {
        sincos_approx_scalar(x).1
    }

    fn exp_taylor_scalar(x: f32) -> f32 {
        1.0 + x
            * (1.0
                + x * (EXP_C2
                    + x * (EXP_C3
                        + x * (EXP_C4 + x * (EXP_C5 + x * (EXP_C6 + x * (EXP_C7 + x * EXP_C8)))))))
    }

    fn reduce_ln2_scalar(x: f32) -> (f32, f32) {
        let k = (x * INV_LN2).round_ties_even();
        let r = x - k * LN2_HEAD - k * LN2_TAIL;
        (r, k)
    }

    fn exp_approx_scalar(x: f32) -> f32 {
        if x.is_infinite() {
            return if x.is_sign_negative() {
                0.0
            } else {
                f32::INFINITY
            };
        }
        if x > EXP_OVERFLOW_CUTOFF {
            return f32::INFINITY;
        }
        if x < EXP_UNDERFLOW_CUTOFF {
            return 0.0;
        }
        let (r, k) = reduce_ln2_scalar(x);
        2.0f32.powf(k) * exp_taylor_scalar(r)
    }

    fn softmax_approx_scalar(xs: &[f32]) -> Vec<f32> {
        let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = xs.iter().map(|&x| exp_approx_scalar(x - max)).collect();
        let sum: f32 = exps.iter().sum();
        exps.into_iter().map(|x| x / sum).collect()
    }

    #[derive(Debug, Clone, Copy)]
    struct SinApproxModule;

    impl Module<1, 1> for SinApproxModule {
        fn ty(&self) -> ([Type; 1], [Type; 1]) {
            let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![NatExpr::Constant(TEST_POINTS.len())]),
            }));
            ([ty.clone()], [ty])
        }

        fn path(&self) -> Path {
            path(vec!["test", "sin_approx"]).unwrap()
        }

        fn def(&self, builder: &Builder, args: [Var; 1]) -> [Var; 1] {
            let [x] = args;
            [sin_approx(builder, x)]
        }
    }

    #[test]
    fn test_sincos_approx_scalar_matches_std_trig() {
        for x in TEST_POINTS {
            let (actual_sin, actual_cos) = sincos_approx_scalar(x);
            let expected_sin = x.sin();
            let expected_cos = x.cos();
            assert!(
                (actual_sin - expected_sin).abs() < 1e-4,
                "sin x={x}: got {actual_sin}, expected {expected_sin}"
            );
            assert!(
                (actual_cos - expected_cos).abs() < 1e-4,
                "cos x={x}: got {actual_cos}, expected {expected_cos}"
            );
        }
    }

    #[test]
    fn test_sin_approx_graph_matches_scalar_reference() {
        let typed = SinApproxModule.term().unwrap();
        let env = stdlib();

        typecheck::check_with(
            &env,
            &typecheck::Parameters::default(),
            typed.term.clone(),
            typed.source_type.clone(),
        )
        .unwrap();

        let backend = NdArrayBackend;
        let interpreter = Interpreter::new(backend.clone(), env, Parameters::default());
        let input = tensor(
            &backend,
            Shape(vec![TEST_POINTS.len()]),
            TEST_POINTS.to_vec(),
        )
        .unwrap();
        let result = interpreter.run(typed.term, vec![input]).unwrap();

        let actual = match result.into_iter().next().unwrap() {
            Value::Tensor(tensor) => match interpreter.backend.to_vec(tensor) {
                TaggedVec::F32(values) => values,
                _ => panic!("expected f32 tensor output"),
            },
            _ => panic!("expected tensor output"),
        };

        let expected: Vec<f32> = TEST_POINTS.iter().copied().map(sin_approx_scalar).collect();
        for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "index {i}: got {actual}, expected {expected}"
            );
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct SinCosApproxModule;

    impl Module<1, 2> for SinCosApproxModule {
        fn ty(&self) -> ([Type; 1], [Type; 2]) {
            let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![NatExpr::Constant(TEST_POINTS.len())]),
            }));
            ([ty.clone()], [ty.clone(), ty])
        }

        fn path(&self) -> Path {
            path(vec!["test", "sincos_approx"]).unwrap()
        }

        fn def(&self, builder: &Builder, args: [Var; 1]) -> [Var; 2] {
            let [x] = args;
            let (sin, cos) = sincos_approx(builder, x);
            [sin, cos]
        }
    }

    #[test]
    fn test_sincos_approx_graph_matches_scalar_reference() {
        let typed = SinCosApproxModule.term().unwrap();
        let env = stdlib();

        typecheck::check_with(
            &env,
            &typecheck::Parameters::default(),
            typed.term.clone(),
            typed.source_type.clone(),
        )
        .unwrap();

        let backend = NdArrayBackend;
        let interpreter = Interpreter::new(backend.clone(), env, Parameters::default());
        let input = tensor(
            &backend,
            Shape(vec![TEST_POINTS.len()]),
            TEST_POINTS.to_vec(),
        )
        .unwrap();
        let result = interpreter.run(typed.term, vec![input]).unwrap();

        let actual = result
            .into_iter()
            .map(|value| match value {
                Value::Tensor(tensor) => match interpreter.backend.to_vec(tensor) {
                    TaggedVec::F32(values) => values,
                    _ => panic!("expected f32 tensor output"),
                },
                _ => panic!("expected tensor output"),
            })
            .collect::<Vec<_>>();

        let expected_sin: Vec<f32> = TEST_POINTS.iter().copied().map(sin_approx_scalar).collect();
        let expected_cos: Vec<f32> = TEST_POINTS.iter().copied().map(cos_approx_scalar).collect();
        for (i, (&actual, &expected)) in actual[0].iter().zip(expected_sin.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "sin index {i}: got {actual}, expected {expected}"
            );
        }
        for (i, (&actual, &expected)) in actual[1].iter().zip(expected_cos.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "cos index {i}: got {actual}, expected {expected}"
            );
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct ExpApproxModule;

    impl Module<1, 1> for ExpApproxModule {
        fn ty(&self) -> ([Type; 1], [Type; 1]) {
            let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![NatExpr::Constant(TEST_POINTS.len())]),
            }));
            ([ty.clone()], [ty])
        }

        fn path(&self) -> Path {
            path(vec!["test", "exp_approx"]).unwrap()
        }

        fn def(&self, builder: &Builder, args: [Var; 1]) -> [Var; 1] {
            let [x] = args;
            [exp_approx(builder, x)]
        }
    }

    #[test]
    fn test_exp_approx_scalar_matches_std_exp() {
        let points = [
            -2.0,
            -std::f32::consts::LN_2,
            -0.5 * std::f32::consts::LN_2,
            0.0,
            0.5 * std::f32::consts::LN_2,
            std::f32::consts::LN_2,
            1.0,
            2.0,
        ];

        for x in points {
            let actual = exp_approx_scalar(x);
            let expected = x.exp();
            assert!(
                (actual - expected).abs() < 5e-4,
                "x={x}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_exp_approx_graph_matches_scalar_reference() {
        let points = vec![
            -2.0,
            -std::f32::consts::LN_2,
            -0.5 * std::f32::consts::LN_2,
            0.0,
            0.5 * std::f32::consts::LN_2,
            std::f32::consts::LN_2,
            1.0,
            2.0,
        ];

        let typed = ExpApproxModule.term().unwrap();
        let env = stdlib();

        typecheck::check_with(
            &env,
            &typecheck::Parameters::default(),
            typed.term.clone(),
            typed.source_type.clone(),
        )
        .unwrap();

        let backend = NdArrayBackend;
        let interpreter = Interpreter::new(backend.clone(), env, Parameters::default());
        let input = tensor(&backend, Shape(vec![points.len()]), points.clone()).unwrap();
        let result = interpreter.run(typed.term, vec![input]).unwrap();

        let actual = match result.into_iter().next().unwrap() {
            Value::Tensor(tensor) => match interpreter.backend.to_vec(tensor) {
                TaggedVec::F32(values) => values,
                _ => panic!("expected f32 tensor output"),
            },
            _ => panic!("expected tensor output"),
        };

        let expected: Vec<f32> = points.iter().copied().map(exp_approx_scalar).collect();
        for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "index {i}: got {actual}, expected {expected}"
            );
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct SoftmaxApproxModule;

    impl Module<1, 1> for SoftmaxApproxModule {
        fn ty(&self) -> ([Type; 1], [Type; 1]) {
            let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![NatExpr::Constant(4)]),
            }));
            ([ty.clone()], [ty])
        }

        fn path(&self) -> Path {
            path(vec!["test", "softmax_approx"]).unwrap()
        }

        fn def(&self, builder: &Builder, args: [Var; 1]) -> [Var; 1] {
            let [x] = args;
            [softmax_approx(builder, x)]
        }
    }

    #[test]
    fn test_softmax_approx_graph_matches_scalar_reference() {
        let points = vec![1.0, 0.0, -1.0, -3.4028e38];
        let typed = SoftmaxApproxModule.term().unwrap();
        let env = stdlib();

        typecheck::check_with(
            &env,
            &typecheck::Parameters::default(),
            typed.term.clone(),
            typed.source_type.clone(),
        )
        .unwrap();

        let backend = NdArrayBackend;
        let interpreter = Interpreter::new(backend.clone(), env, Parameters::default());
        let input = tensor(&backend, Shape(vec![points.len()]), points.clone()).unwrap();
        let result = interpreter.run(typed.term, vec![input]).unwrap();

        let actual = match result.into_iter().next().unwrap() {
            Value::Tensor(tensor) => match interpreter.backend.to_vec(tensor) {
                TaggedVec::F32(values) => values,
                _ => panic!("expected f32 tensor output"),
            },
            _ => panic!("expected tensor output"),
        };

        let expected = softmax_approx_scalar(&points);
        for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "index {i}: got {actual}, expected {expected}"
            );
        }
    }
}
