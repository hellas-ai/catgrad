use catgrad::category::lang::Object;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::interpreter::{Backend, Interpreter, Parameters, Value};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use open_hypergraphs::lax::var;

#[test]
fn test_if_op_true() {
    let backend = NdArrayBackend;
    let env = catgrad::stdlib::stdlib();
    let parameters = Parameters::default();
    let interpreter = Interpreter::new(backend, env, parameters);

    let term = var::build(|builder: &Builder| {
        let x = Var::new(builder.clone(), Object::Tensor);
        let cond_val = Var::new(builder.clone(), Object::Tensor);

        let res = cond(
            builder,
            cond_val.clone(),
            |_b, args: Vec<Var>| {
                let [x] = args.try_into().unwrap();
                vec![x.clone() + x] // 2*x
            },
            |_b, args: Vec<Var>| {
                let [x] = args.try_into().unwrap();
                vec![x] // x
            },
            vec![x.clone()],
        )
        .pop()
        .unwrap();

        (vec![x, cond_val], vec![res])
    })
    .unwrap();

    // Input x = 5.0, cond = 1.0 (true)
    let x_val =
        catgrad::interpreter::tensor(&interpreter.backend, Shape(vec![]), vec![5.0f32]).unwrap();
    let cond_true =
        catgrad::interpreter::tensor(&interpreter.backend, Shape(vec![]), vec![1.0f32]).unwrap();

    let results = interpreter.run(term, vec![x_val, cond_true]).unwrap();
    let res = results[0].clone();

    if let Value::Tensor(t) = res {
        let vec = interpreter.backend.to_vec(t);
        if let catgrad::interpreter::TaggedVec::F32(v) = vec {
            assert_eq!(v[0], 10.0);
        } else {
            panic!("Expected F32 result");
        }
    } else {
        panic!("Expected tensor result");
    }
}

#[test]
fn test_if_op_false() {
    let backend = NdArrayBackend;
    let env = catgrad::stdlib::stdlib();
    let parameters = Parameters::default();
    let interpreter = Interpreter::new(backend, env, parameters);

    let term = var::build(|builder: &Builder| {
        let x = Var::new(builder.clone(), Object::Tensor);
        let cond_val = Var::new(builder.clone(), Object::Tensor);

        let res = cond(
            builder,
            cond_val.clone(),
            |_b, args: Vec<Var>| {
                let [x] = args.try_into().unwrap();
                vec![x.clone() + x] // 2*x
            },
            |_b, args: Vec<Var>| {
                let [x] = args.try_into().unwrap();
                vec![x] // x
            },
            vec![x.clone()],
        )
        .pop()
        .unwrap();

        (vec![x, cond_val], vec![res])
    })
    .unwrap();

    // Input x = 5.0, cond = 0.0 (false)
    let x_val =
        catgrad::interpreter::tensor(&interpreter.backend, Shape(vec![]), vec![5.0f32]).unwrap();
    let cond_false =
        catgrad::interpreter::tensor(&interpreter.backend, Shape(vec![]), vec![0.0f32]).unwrap();

    let results = interpreter.run(term, vec![x_val, cond_false]).unwrap();
    let res = results[0].clone();

    if let Value::Tensor(t) = res {
        let vec = interpreter.backend.to_vec(t);
        if let catgrad::interpreter::TaggedVec::F32(v) = vec {
            assert_eq!(v[0], 5.0);
        } else {
            panic!("Expected F32 result");
        }
    } else {
        panic!("Expected tensor result");
    }
}
