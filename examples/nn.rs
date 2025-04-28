// Example NN model inference
// Terms built using the var API

use std::cell::RefCell;
use std::rc::Rc;

use catgrad::{
    backend::cpu::{
        eval::EvalState,
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{
        nn::{
            layers::{
                constant, embedding, gelu, layernorm, linear, mat_mul, parameter, rmsnorm, softmax,
                tanh, transpose, Builder,
            },
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

pub fn layer(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    let res = x.clone();
    let x = rmsnorm(builder, &format!("{name}.prenorm"), x);
    let x = attention(builder, in_dim, &format!("{name}.attention"), x);
    let x = rmsnorm(builder, &format!("{name}.postnorm"), x);
    let x = mlp(builder, in_dim, out_dim, &format!("{name}.mlp"), x);
    x + res
}

pub fn token_embeddings(
    builder: &Builder,
    vocab_size: usize,
    dim: usize,
    name: &str,
    x: Var,
) -> Var {
    let t = NdArrayType {
        shape: Shape(vec![vocab_size, dim]),
        dtype: Dtype::F32,
    };
    let weights = parameter(builder, t, format!("{name}.weight"));
    embedding(builder, x, weights)
}

pub fn attention(builder: &Builder, dim: usize, name: &str, x: Var) -> Var {
    let k = linear(builder, dim, dim, &format!("{name}.key"), x.clone());
    let q = linear(builder, dim, dim, &format!("{name}.query"), x.clone());
    let v = linear(builder, dim, dim, &format!("{name}.value"), x.clone());

    let tk = transpose(builder, 1, 2, k); // TODO: dims
    let attn = mat_mul(builder, q.clone(), tk);
    let denom = constant(builder, attn.label.clone(), f32::sqrt(dim as f32));
    let attn = attn / denom;
    let attn = softmax(builder, attn);
    let attn = mat_mul(builder, attn, v);
    let o = linear(builder, dim, dim, &format!("{name}.proj"), attn);
    o
}

pub fn mlp(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    let x = linear(builder, in_dim, out_dim, &format!("{name}.lin1"), x);
    let x = tanh(builder, x);
    let x = linear(builder, out_dim, in_dim, &format!("{name}.lin2"), x);
    let x = gelu(builder, x);
    x
}

impl Model {
    pub fn build(
        batches: usize,
        tokens: usize,
        vocab_size: usize,
        layers: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Self {
        let in_type = NdArrayType {
            shape: Shape(vec![batches, tokens]),
            dtype: Dtype::I32,
        };

        let builder = Rc::new(RefCell::new(Term::empty()));
        {
            let x = Var::new(builder.clone(), in_type.clone());
            let mut result = x.clone();
            result = token_embeddings(&builder, vocab_size, in_dim, "token_embeddings", result);
            result = layernorm(&builder, "prenorm", result);
            for i in 0..layers {
                result = layer(&builder, in_dim, out_dim, &format!("layers.{i}"), result);
            }
            result = layernorm(&builder, "postnorm", result);
            result = softmax(&builder, result);

            builder.borrow_mut().sources = vec![x.new_source()];
            builder.borrow_mut().targets = vec![result.new_target()];
        }

        let f = Rc::try_unwrap(builder).unwrap().into_inner();

        Self { term: f }
    }

    pub fn run(&self, x: &NdArray<i32>) -> TaggedNdArray {
        let mut state = EvalState::from_lax(self.term.clone());
        let tensors = read_safetensors("model.safetensors");
        state.set_parameters(tensors);
        let [result] = state.eval_with(vec![x.clone().into()])[..] else {
            panic!("unexpected result")
        };

        result.clone()
    }
}

struct OptParse(Vec<String>);

impl OptParse {
    fn new() -> Self {
        Self(std::env::args().collect())
    }

    fn get(&self, name: &str, default: usize) -> usize {
        for p in self.0.windows(2) {
            if p[0] == name {
                return p[1].parse::<usize>().unwrap();
            }
        }
        return default;
    }
}

pub fn main() {
    let args = OptParse::new();
    let batches = args.get("-b", 1);
    let dim = args.get("-d", 8);
    let exp = args.get("-e", 2);
    let layers = args.get("-l", 4);
    let vocab_size = args.get("-v", 128);
    let tokens = args.get("-t", 9);
    let fill = args.get("-f", 1);

    let input = NdArray::new(
        vec![fill as i32; batches * tokens],
        Shape(vec![batches, tokens]),
    );
    let model = Model::build(batches, tokens, vocab_size, layers, dim, dim * exp);
    println!("Model built...");
    let result = model.run(&input);
    println!("input {:?}", input);
    println!("Result: {:?}", result);
}
