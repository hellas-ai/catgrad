use catgrad::{
    backend::cpu::{
        eval::EvalState,
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{Shape, Var},
};
use clap::Parser;
use hf_hub::api::sync::Api;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;
use tokenizers::tokenizer::{Result, Tokenizer};

#[path = "../utils/mod.rs"]
mod utils;
use utils::read_safetensors_multiple;

#[allow(unused)]
fn show(name: &str, var: &Var) {
    println!("{name} label: {:?}", var.label,);
}

// This configuration contains the union of relevant fields from all supported models.
// Models ignore fields they don't need. The aliases are for GPT-2 alternative names.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub sliding_window_pattern: usize,
    pub rope_local_base_freq: f32,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    pub architectures: Vec<String>,
}

mod gemma;
mod gpt2;
mod llama;
mod olmo;
mod phi;
mod qwen;

use gemma::Model as GemmaModel;
use gpt2::Model as GPT2Model;
use llama::Model as LlamaModel;
use olmo::Model as OlmoModel;
use phi::Model as PhiModel;
use qwen::Model as QwenModel;

struct ModelRunner {
    pub tensors: Rc<HashMap<String, TaggedNdArray>>,
    pub state: Option<EvalState>,
    pub model: Box<dyn ModelBuilder>,
}

// Trait for model builders for various architectures (llama, qwen, gpt2, etc.)
pub trait ModelBuilder {
    // Build the model architecture graph for a given input shape
    fn build(&mut self, batches: usize, tokens: usize, config: &Config) -> EvalState;
    // Optional post-processing of loaded weights (renaming, reshaping, etc.)
    fn post_load(&mut self, _tensors: &mut HashMap<String, TaggedNdArray>) {}
}

impl ModelRunner {
    fn build(&mut self, batches: usize, tokens: usize, config: &Config) {
        let state = self.model.build(batches, tokens, config);
        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors)); // TODO ^
    }

    // Create model and load weights from file
    pub fn new(model_paths: Vec<PathBuf>, arch: &str) -> Self {
        let mut model: Box<dyn ModelBuilder> = match arch {
            "LlamaForCausalLM" => Box::new(LlamaModel {}),
            "Olmo2ForCausalLM" => Box::new(OlmoModel {}),
            "Qwen3ForCausalLM" => Box::new(QwenModel {}),
            "Gemma3ForCausalLM" => Box::new(GemmaModel {}),
            "Phi3ForCausalLM" => Box::new(PhiModel {}),
            "GPT2LMHeadModel" => Box::new(GPT2Model {}),
            _ => panic!("Unknown architecture {arch}"),
        };

        let mut tensors = read_safetensors_multiple(model_paths);
        model.post_load(&mut tensors);
        println!("Model weights loaded...");

        Self {
            tensors: Rc::new(tensors),
            state: None,
            model,
        }
    }

    // Make a forward pass given a list of tokens
    pub fn run(&mut self, x: &NdArray<i32>) -> TaggedNdArray {
        let [result] = self
            .state
            .as_mut()
            .unwrap()
            .eval_with(vec![x.clone().into()])[..]
        else {
            panic!("unexpected result")
        };

        result.clone()
    }

    fn generate(&mut self, batches: usize, tokens: Vec<i32>, config: &Config) -> i32 {
        let l = tokens.len();
        let input = NdArray::new(tokens, Shape(vec![batches, l / batches]));

        self.build(batches, l, config);
        log::debug!("Model graph built...");
        let result = self.run(&input);

        let v = config.vocab_size;

        let r = result.data();

        // Greedy (temp = 0) sampling
        argmax(&r[r.len() - v..])
    }
}

fn argmax(v: &[f32]) -> i32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap() as i32
}

#[derive(Parser, Debug)]
struct Args {
    /// Model name on Huggingface Hub
    #[arg(
        short = 'm',
        long,
        default_value = "HuggingFaceTB/SmolLM2-135M-Instruct"
    )]
    model_name: String,

    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Hello world")]
    prompt: String,

    /// Number of tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    seq_len: usize,
}

fn get_model_files(model: &str) -> (Vec<PathBuf>, PathBuf, PathBuf) {
    let api = Api::new().unwrap();

    let repo = api.model(model.to_string());

    // Get the model.safetensor file(s)
    let m = if let Ok(index) = repo.get("model.safetensors.index.json") {
        let index = std::fs::File::open(index).unwrap();
        let json: serde_json::Value = serde_json::from_reader(&index).unwrap();
        let mut set = std::collections::HashSet::new();
        if let Some(weight_map) = json.get("weight_map").unwrap().as_object() {
            for v in weight_map.values() {
                set.insert(v.as_str().unwrap().to_string());
            }
        }
        set.iter().map(|p| repo.get(p).unwrap()).collect()
    } else {
        vec![repo.get("model.safetensors").unwrap()]
    };

    let c = repo.get("config.json").unwrap();
    let t = repo.get("tokenizer.json").unwrap();

    (m, c, t)
}

pub fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let (model_paths, config_path, tokenizer_path) = get_model_files(&args.model_name);
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();

    let encoding = tokenizer.encode(args.prompt.clone(), true)?;

    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    let tokens = token_ids.len();
    let batches = 1;
    let input = NdArray::new(token_ids, Shape(vec![batches, tokens]));

    log::info!("Input tokens {:?}", &input);
    let mut model_runner = ModelRunner::new(model_paths, &config.architectures[0]);
    let mut input_tokens = input.data.borrow_mut();

    print!("{}", args.prompt);

    let start_gen = std::time::Instant::now();

    let mut total_tokens = 0;
    for _ in 0..args.seq_len {
        let next_token_id = model_runner.generate(batches, input_tokens.clone(), &config);
        print!(
            "{}",
            tokenizer.decode(&[next_token_id as u32], false).unwrap()
        );
        std::io::stdout().flush().unwrap();
        input_tokens.push(next_token_id);
        total_tokens += 1;
    }

    let elapsed = start_gen.elapsed();
    println!(
        "\n{total_tokens} tokens generated in {} seconds. ({:.2} tokens/sec)",
        elapsed.as_secs(),
        total_tokens as f64 / elapsed.as_secs_f64(),
    );

    Ok(())
}
