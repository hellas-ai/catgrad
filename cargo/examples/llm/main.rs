use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::nn::layers::{argmax, cast, concat, reshape, rope_tables, side_effect},
    core::{Callback, Dtype, NdArrayType, Shape, Var},
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
impl Config {
    // Sometimes the head_dim fields is missing
    fn get_head_dim(&self) -> usize {
        if self.head_dim == 0 {
            self.hidden_size / self.num_attention_heads
        } else {
            self.head_dim
        }
    }
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub use_kv_cache: bool,
    pub kv_cache: Vec<Option<(Var, Var)>>,
}

impl Cache {
    pub fn init(builder: &Builder, config: &Config, positions: usize, use_kv_cache: bool) -> Self {
        let (cos, sin) = rope_tables(builder, config.rope_theta, positions, config.get_head_dim());
        Self {
            cos,
            sin,
            use_kv_cache,
            kv_cache: vec![None; config.num_hidden_layers],
        }
    }
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
    pub tokenizer: Tokenizer,
    pub use_kv_cache: bool,
}

// Trait for model builders for various architectures (llama, qwen, gpt2, etc.)
pub trait ModelBuilder {
    // Build the model architecture graph for a given input shape
    fn build(
        &self,
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        x: Var,
    ) -> Var;
    // Optional post-processing of loaded weights (renaming, reshaping, etc.)
    fn post_load(&mut self, _tensors: &mut HashMap<String, TaggedNdArray>) {}
}

impl ModelRunner {
    fn next_token(&self, builder: &Builder, logits: Var) -> Var {
        let batches = logits.label.shape.0[0];
        let am = argmax(builder, logits);
        let am = reshape(builder, Shape(vec![batches, 1]), am);
        cast(builder, Dtype::I32, am)
    }

    fn unroll(&self, builder: &Builder, config: &Config, x: Var, seq_len: usize) -> Var {
        let mut input = x;

        let il = input.label.shape.0[1];
        let mut cache = Cache::init(builder, config, il + seq_len, self.use_kv_cache);
        for i in 0..seq_len {
            let pos = if i == 0 || !self.use_kv_cache {
                0
            } else {
                il + i
            };
            let result = self
                .model
                .build(builder, config, &mut cache, pos, input.clone());
            let new_token = self.next_token(builder, result);
            if cache.use_kv_cache {
                input = new_token.clone();
                let tokenizer = self.tokenizer.clone();
                side_effect(
                    builder,
                    Callback::new(move |a: &TaggedNdArray| {
                        let token = tokenizer.decode(&[a.get(&[0, 0]) as u32], false).unwrap();
                        print!("{token}");
                        std::io::stdout().flush().unwrap();
                    }),
                    &new_token,
                );
            } else {
                input = concat(builder, 1, input, new_token);
            }
        }

        input
    }

    fn build_unrolled(
        &mut self,
        batches: usize,
        tokens: usize,
        config: &Config,
        max_new_tokens: usize,
    ) {
        let in_type = NdArrayType::new(Shape(vec![batches, tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let result = self.unroll(builder, config, x.clone(), max_new_tokens);

            (vec![x], vec![result])
        });

        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors));
    }

    fn build(&mut self, batches: usize, tokens: usize, config: &Config) {
        let in_type = NdArrayType::new(Shape(vec![batches, tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let positions = x.label.shape.0[1];
            let mut cache = Cache::init(builder, config, positions, self.use_kv_cache);
            let result = self.model.build(builder, config, &mut cache, 0, x.clone());
            let new_token = self.next_token(builder, result);
            (vec![x], vec![new_token])
        });

        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors));
    }

    pub fn new(arch: &str, tokenizer: Tokenizer, use_kv_cache: bool) -> Self {
        let model: Box<dyn ModelBuilder> = match arch {
            "LlamaForCausalLM" => Box::new(LlamaModel {}),
            "Olmo2ForCausalLM" => Box::new(OlmoModel {}),
            "Qwen3ForCausalLM" => Box::new(QwenModel {}),
            "Gemma3ForCausalLM" => Box::new(GemmaModel {}),
            "Phi3ForCausalLM" => Box::new(PhiModel {}),
            "GPT2LMHeadModel" => Box::new(GPT2Model {}),
            _ => panic!("Unknown architecture {arch}"),
        };
        Self {
            tensors: Rc::new(HashMap::new()),
            state: None,
            model,
            tokenizer,
            use_kv_cache,
        }
    }

    pub fn load(&mut self, model_paths: Vec<PathBuf>) {
        let mut tensors = read_safetensors_multiple(model_paths);
        self.model.post_load(&mut tensors);

        self.tensors = Rc::new(tensors);
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

        let r = result.data();

        r[0] as i32
    }

    fn generate_all(
        &mut self,
        batches: usize,
        tokens: Vec<i32>,
        config: &Config,
        max_new_tokens: usize,
    ) -> Vec<u32> {
        let l = tokens.len();
        let input = NdArray::new(tokens, Shape(vec![batches, l / batches]));

        self.build_unrolled(batches, l, config, max_new_tokens);
        log::debug!("Model graph built...");
        let result = self.run(&input);

        let r = result.data();

        r.iter().map(|&x| x as u32).collect()
    }

    fn save_dot(&mut self, config: &Config, path: &PathBuf) {
        use graphviz_rust::{print, printer::PrinterContext};

        let in_type = NdArrayType::new(Shape(vec![1, 1]), Dtype::I32);

        let term = EvalState::build_lax(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let positions = x.label.shape.0[1];
            let mut cache = Cache::init(builder, config, positions, self.use_kv_cache);
            let result = self.model.build(builder, config, &mut cache, 0, x.clone());

            (vec![x], vec![result])
        });

        let dot_graph = open_hypergraphs_dot::generate_dot(&term);
        let dot_string = print(dot_graph, &mut PrinterContext::default());
        let _ = std::fs::write(path, dot_string);
    }
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

    /// Path to save Graphviz dot file
    #[arg(long, default_value=None)]
    save_dot: Option<PathBuf>,

    /// Build unrolled graph
    #[arg(short = 'u', long)]
    unrolled: bool,

    /// Use KV-cache
    #[arg(short = 'k', long)]
    kv_cache: bool,
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
    let models = HashMap::from([
        ("gpt", "openai-community/gpt2"),
        ("smol", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        ("llama", "meta-llama/Llama-3.2-1B"),
        ("gemma", "google/gemma-3-1b-pt"),
        ("qwen", "Qwen/Qwen3-0.6B"),
        ("olmo", "allenai/OLMo-2-0425-1B"),
        ("phi", "microsoft/Phi-4-mini-instruct"),
    ]);

    let model_name = models
        .get(args.model_name.as_str())
        .copied()
        .unwrap_or(&args.model_name);

    let (model_paths, config_path, tokenizer_path) = get_model_files(model_name);
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();

    let encoding = tokenizer.encode(args.prompt.clone(), true)?;

    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    let tokens = token_ids.len();
    let batches = 1;

    let mut model_runner = ModelRunner::new(&config.architectures[0], tokenizer, args.kv_cache);

    if let Some(ref path) = args.save_dot {
        model_runner.save_dot(&config, path);
        println!(
            "Model graph for {} written to {}",
            model_name,
            path.display()
        );
        return Ok(());
    }

    model_runner.load(model_paths);
    println!("Model weights loaded for {model_name}");

    let input = NdArray::new(token_ids, Shape(vec![batches, tokens]));
    log::info!("Input tokens {:?}", &input);
    let mut input_tokens = input.data.borrow_mut();

    let start_gen = std::time::Instant::now();

    if args.unrolled {
        if model_runner.use_kv_cache {
            print!("{}", args.prompt);
        }
        let next_tokens =
            model_runner.generate_all(batches, input_tokens.clone(), &config, args.seq_len);
        if !model_runner.use_kv_cache {
            print!(
                "{}",
                model_runner.tokenizer.decode(&next_tokens, false).unwrap()
            );
        }
    } else {
        print!("{}", args.prompt);
        for _ in 0..args.seq_len {
            let next_token_id = model_runner.generate(batches, input_tokens.clone(), &config);
            print!(
                "{}",
                model_runner
                    .tokenizer
                    .decode(&[next_token_id as u32], false)
                    .unwrap()
            );
            std::io::stdout().flush().unwrap();
            input_tokens.push(next_token_id);
        }
    }

    let elapsed = start_gen.elapsed();
    println!(
        "\n{} tokens generated in {} seconds. ({:.2} tokens/sec)",
        args.seq_len,
        elapsed.as_secs(),
        args.seq_len as f64 / elapsed.as_secs_f64(),
    );

    Ok(())
}
