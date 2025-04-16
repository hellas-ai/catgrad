// Example NN model inference
// Terms built using the var API

use std::rc::Rc;
use std::{cell::RefCell, f64::NAN};

use catgrad::{
    backend::cpu::{
        eval::EvalState,
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{
        nn::{
            layers::{gelu, layernorm, linear, rmsnorm, softmax, tanh, Builder},
            utils::read_safetensors,
        },
        Dtype, NdArrayType, Shape, Term, Var,
    },
};

#[allow(unused)]
fn show(name: &str, var: &Var) {
    println!("{name} label: {:?}", var.label,);
}

#[derive(Debug)]
struct Model {
    pub term: Term,
}

pub fn layer(
    builder: &Builder,
    input_features: usize,
    output_features: usize,
    dtype: Dtype,
    name: &str,
    x: Var,
) -> Var {
    let res = x.clone();
    let x = mlp_layer(
        builder,
        input_features,
        output_features,
        dtype,
        &format!("{name}.mlp"),
        x,
    );
    x + res
}

pub fn mlp_layer(
    builder: &Builder,
    input_features: usize,
    output_features: usize,
    dtype: Dtype,
    name: &str,
    x: Var,
) -> Var {
    // let res = x.clone();
    let l1 = linear(
        builder,
        input_features,
        output_features,
        dtype,
        &format!("{name}.lin1"),
        x,
    );
    let a = tanh(builder, l1);
    // let a = gelu(builder, l1);
    let l2 = linear(
        builder,
        output_features,
        input_features,
        dtype,
        &format!("{name}.lin2"),
        a,
    );
    l2
}

impl Model {
    pub fn build(in_dim: usize, out_dim: usize) -> Self {
        let in_type = NdArrayType {
            shape: Shape(vec![1, in_dim]),
            dtype: Dtype::F32,
        };

        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            let mut result = x.clone();
            result = layernorm(&builder, "prenorm", result);
            for i in 0..4 {
                result = layer(
                    &builder,
                    in_dim,
                    out_dim,
                    Dtype::F32,
                    &format!("layers.{i}"),
                    result,
                );
            }
            result = layernorm(&builder, "postnorm", result);
            result = softmax(&builder, result);

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        Self { term: f }
    }

    pub fn run(&self, x: &NdArray<f32>) -> TaggedNdArray {
        let mut state = EvalState::from_lax(self.term.clone());
        let tensors = read_safetensors("model.safetensors");
        state.set_parameters(tensors);
        let [result] = state.eval_with(vec![x.clone().into()])[..] else {
            panic!("unexpected result")
        };

        result.clone()
    }
}

pub fn main() {
    let input = NdArray::new(vec![1.0; 8], Shape(vec![1, 8]));
    let model = Model::build(8, 16);
    // println!("Model {:#?}", &model);
    let result = model.run(&input);
    println!("input {:?}", input);
    println!("Result: {:?}", result);
}
