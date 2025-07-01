// Example for inference on a BERT compatible model
// While the architecture is the same across these models, the weight names can be slightly different
// and require some adjustments while loading the weights
// Tested with https://huggingface.co/BAAI/bge-base-en-v1.5
use clap::Parser;
use std::path::PathBuf;
use tokenizers::tokenizer::{Result, Tokenizer};

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

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub vocab_size: usize,
}

#[derive(Debug)]
struct Model {
    pub state: EvalState,
}

pub fn layer(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
    let att = attention(builder, config, &format!("{name}.attention"), x);
    let x = intermediate(
        builder,
        config.hidden_size,
        config.intermediate_size,
        &format!("{name}.intermediate"),
        att.clone(),
    );
    let x = output(
        builder,
        config.intermediate_size,
        config.hidden_size,
        config.layer_norm_eps,
        &format!("{name}.output"),
        x,
        att,
    );
    x
}

pub fn embeddings(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
    let t = NdArrayType::new(
        Shape(vec![config.vocab_size, config.hidden_size]),
        Dtype::F32,
    );
    let weights = parameter(builder, t, format!("{name}.word_embeddings.weight"));
    let we = embedding(builder, x.clone(), weights);

    let t = NdArrayType::new(
        Shape(vec![config.max_position_embeddings, config.hidden_size]),
        Dtype::F32,
    );
    let pos = arange(builder, x.label.size(), Dtype::I32);
    let pos = expand(builder, x.label.shape.clone(), pos);
    let weights = parameter(builder, t, format!("{name}.position_embeddings.weight"));
    let pe = embedding(builder, pos, weights);

    let t = NdArrayType::new(Shape(vec![2, config.hidden_size]), Dtype::F32);
    let weights = parameter(builder, t, format!("{name}.token_type_embeddings.weight"));
    let typ = constant(builder, x.label, 0.);
    let te = embedding(builder, typ, weights);

    let norm = layernorm(
        builder,
        config.layer_norm_eps,
        &format!("{name}.LayerNorm"),
        we + pe + te,
    );

    norm
}

pub fn attention(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
    let dim = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let head_dim = dim / num_heads;
    let b = x.label.shape.0[0];
    let s = x.label.shape.0[1];

    let k = linear(builder, dim, dim, &format!("{name}.self.key"), x.clone());
    let q = linear(builder, dim, dim, &format!("{name}.self.query"), x.clone());
    let v = linear(builder, dim, dim, &format!("{name}.self.value"), x.clone());

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
    let attn = transpose(builder, 1, 2, attn);
    let attn = reshape(builder, Shape(vec![b, s, dim]), attn);
    let o = output(
        builder,
        dim,
        dim,
        config.layer_norm_eps,
        &format!("{name}.output"),
        attn,
        x,
    );
    o
}

pub fn intermediate(builder: &Builder, in_dim: usize, out_dim: usize, name: &str, x: Var) -> Var {
    let x = linear(builder, in_dim, out_dim, &format!("{name}.dense"), x);

    gelu(builder, x)
}

pub fn output(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    eps: f32,
    name: &str,
    x: Var,
    input: Var,
) -> Var {
    let x = linear(builder, in_dim, out_dim, &format!("{name}.dense"), x);
    layernorm(builder, eps, &format!("{name}.LayerNorm"), x + input)
}

impl Model {
    pub fn build(batches: usize, tokens: usize, config: &Config) -> Self {
        let in_type = NdArrayType::new(Shape(vec![batches, tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let emb = embeddings(builder, config, "embeddings", x.clone());

            let mut result = emb;

            for i in 0..config.num_hidden_layers {
                result = layer(builder, config, &format!("encoder.layer.{i}"), result);
            }
            (vec![x], vec![result])
        });

        Self { state }
    }

    pub fn run(&mut self, x: &NdArray<i32>, model_path: &str) -> TaggedNdArray {
        let tensors = read_safetensors(model_path);
        println!("Model weights loaded...");
        self.state.set_parameters(std::rc::Rc::new(tensors));
        let [result] = self.state.eval_with(vec![x.clone().into()])[..] else {
            panic!("unexpected result")
        };

        result.clone()
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to the safetensors model file
    #[arg(short = 'm', long, default_value = "bert.safetensors")]
    model_path: String,

    /// Number of batches
    #[arg(short = 'b', long, default_value_t = 1)]
    batches: usize,

    /// Number of tokens per sequence
    #[arg(short = 't', long, default_value_t = 1)]
    tokens: usize,

    /// Value to fill input tensor with for testing
    #[arg(short = 'f', long, default_value_t = 0)]
    fill: usize,

    /// Input text to embed
    #[arg(short = 'p', long)]
    prompt: Option<String>,
}

fn get_config(model_path: &str) -> Config {
    let mut model_path = PathBuf::from(model_path);
    if let Ok(link) = model_path.read_link() {
        model_path = link;
    }

    let config_dir = model_path.parent().unwrap();
    let config_path = config_dir.join("config.json");
    serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap()
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let config = get_config(&args.model_path);

    let mut batches = args.batches;
    let mut tokens = args.tokens;
    let fill = args.fill;

    let iv = if fill != 0 {
        vec![fill as i32; batches * tokens]
    } else {
        (0..batches)
            .flat_map(|_| 0..tokens)
            .map(|x| x as i32)
            .collect()
    };

    let mut input = NdArray::new(iv, Shape(vec![batches, tokens]));
    if let Some(prompt) = args.prompt {
        let tokenizer = Tokenizer::from_pretrained("BAAI/bge-base-en-v1.5", None)?;
        let encoding = tokenizer.encode(prompt, true)?;
        // println!("{:?}", encoding.get_tokens());

        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        tokens = ids.len();
        batches = 1;
        input = NdArray::new(ids, Shape(vec![1, tokens]));
    }

    println!("Input tokens {:?}", &input);
    let mut model = Model::build(batches, tokens, &config);
    println!("Model graph built...");
    let result = model.run(&input, &args.model_path);
    println!("Result: {result:?}");
    Ok(())
}
