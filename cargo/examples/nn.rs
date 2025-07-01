// Example NN model inference
// Terms built using the var API

use clap::Parser;

use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{nn::layers::*, Dtype, NdArrayType, Shape, Var},
};

mod utils;
use utils::read_safetensors;

#[allow(unused)]
fn show(name: &str, var: &Var) {
    println!("{name} label: {:?}", var.label,);
}

#[derive(Debug)]
struct Model {
    pub state: EvalState,
}

pub fn layer(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    let res = x.clone();
    let x = rmsnorm(builder, 1e-05, &format!("{name}.prenorm"), x);
    let x = attention(builder, in_dim, &format!("{name}.attention"), x);
    let x = rmsnorm(builder, 1e-05, &format!("{name}.postnorm"), x);
    let x = mlp(builder, in_dim, out_dim, &format!("{name}.mlp"), x);
    x + res
}

pub fn embeddings(builder: &Builder, size: usize, dim: usize, name: &str, x: Var) -> Var {
    let t = NdArrayType::new(Shape(vec![size, dim]), Dtype::F32);
    let weights = parameter(builder, t, format!("{name}.weight"));
    embedding(builder, x, weights)
}

pub fn attention(builder: &Builder, dim: usize, name: &str, x: Var) -> Var {
    let num_heads = 4;
    let head_dim = dim / num_heads;
    let b = x.label.shape.0[0];
    let s = x.label.shape.0[1];

    let k = linear(builder, dim, dim, &format!("{name}.key"), x.clone());
    let q = linear(builder, dim, dim, &format!("{name}.query"), x.clone());
    let v = linear(builder, dim, dim, &format!("{name}.value"), x);

    let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
    let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
    let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

    let q = transpose(builder, 1, 2, q);
    let k = transpose(builder, 1, 2, k);
    let v = transpose(builder, 1, 2, v);

    let tk = transpose(builder, 2, 3, k);
    let attn = mat_mul(builder, q, tk);
    let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
    let attn = attn / denom;
    let attn = softmax(builder, attn);
    let attn = mat_mul(builder, attn, v);
    let x = transpose(builder, 1, 2, attn);
    let x = reshape(builder, Shape(vec![b, s, dim]), x);
    let o = linear(builder, dim, dim, &format!("{name}.proj"), x);
    o
}

pub fn mlp(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    let x = linear(builder, in_dim, out_dim, &format!("{name}.lin1"), x);
    let x = tanh(builder, x);
    let x = linear(builder, out_dim, in_dim, &format!("{name}.lin2"), x);

    gelu(builder, x)
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
        let in_type = NdArrayType::new(Shape(vec![batches, tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let tok_emb = embeddings(builder, vocab_size, in_dim, "token_embeddings", x.clone());
            // TODO: fix hardcoded max_seq_len
            let max_seq_len = 16;
            let pos = arange(builder, x.label.size(), Dtype::I32);
            let pos = expand(builder, x.label.shape.clone(), pos);
            let pos_emb = embeddings(builder, max_seq_len, in_dim, "position_embeddings", pos);
            let emb = tok_emb + pos_emb;

            let mut result = layernorm(builder, 1e-05, "prenorm", emb);
            for i in 0..layers {
                result = layer(builder, in_dim, out_dim, &format!("layers.{i}"), result);
            }
            result = layernorm(builder, 1e-05, "postnorm", result);
            result = softmax(builder, result);
            (vec![x], vec![result])
        });

        Self { state }
    }

    pub fn run(&self, x: &NdArray<i32>, model_path: &str) -> TaggedNdArray {
        let mut state = self.state.clone();
        let tensors = read_safetensors(model_path);
        state.set_parameters(std::rc::Rc::new(tensors));
        let [result] = state.eval_with(vec![x.clone().into()])[..] else {
            panic!("unexpected result")
        };

        result.clone()
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to the safetensors model file
    #[arg(short = 'p', long, default_value = "model.safetensors")]
    model_path: String,

    /// Dimension size
    #[arg(short = 'd', long, default_value_t = 8)]
    dim: usize,

    /// MLP expansion factor
    #[arg(short = 'e', long, default_value_t = 2)]
    exp: usize,

    /// Number of layers
    #[arg(short = 'l', long, default_value_t = 4)]
    layers: usize,

    /// Vocab size
    #[arg(short = 'v', long, default_value_t = 128)]
    vocab_size: usize,

    /// Number of batches
    #[arg(short = 'b', long, default_value_t = 1)]
    batches: usize,

    /// Number of tokens per sequence
    #[arg(short = 't', long, default_value_t = 1)]
    tokens: usize,

    /// Value to fill input tensor with
    #[arg(short = 'f', long, default_value_t = 0)]
    fill: usize,
}

pub fn main() {
    let args = Args::parse();
    let batches = args.batches;
    let dim = args.dim;
    let exp = args.exp;
    let layers = args.layers;
    let vocab_size = args.vocab_size;
    let tokens = args.tokens;
    let fill = args.fill;

    let iv = if fill != 0 {
        vec![fill as i32; batches * tokens]
    } else {
        (0..batches)
            .flat_map(|_| 0..tokens)
            .map(|x| x as i32)
            .collect()
    };

    let input = NdArray::new(iv, Shape(vec![batches, tokens]));
    let model = Model::build(batches, tokens, vocab_size, layers, dim, dim * exp);
    println!("Model built...");
    let result = model.run(&input, &args.model_path);
    println!("input {input:?}");
    println!("Result: {result:?}");
}
