use catgrad_legacy::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{Dtype, NdArrayType, Shape, Var},
};
use clap::Parser;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;
use tokenizers::tokenizer::{Result, Tokenizer};

use catgrad_llm::utils::{
    get_model_chat_template, get_model_files, read_safetensors_multiple, render_chat_template,
};

use catgrad_llm::legacy::models::utils::{Cache, Config, ModelBuilder, get_model};
use catgrad_llm::legacy::nn::layers::{argmax, cast, reshape};

struct ModelRunner {
    pub tensors: Rc<HashMap<String, TaggedNdArray>>,
    pub state: Option<EvalState>,
    pub model: Box<dyn ModelBuilder>,
    pub tokenizer: Tokenizer,
    pub use_kv_cache: bool,
    pub kv_cache: Vec<TaggedNdArray>,
    pub total_tokens: usize,
    pub use_fp16: bool,
}

impl ModelRunner {
    fn next_token(&self, builder: &Builder, logits: Var) -> Var {
        let batches = logits.label.shape.0[0];
        let mut logits = logits;
        if logits.label.dtype != Dtype::F32 {
            logits = cast(builder, Dtype::F32, logits);
        }
        let am = argmax(builder, logits);
        let am = reshape(builder, Shape(vec![batches, 1]), am);
        cast(builder, Dtype::I32, am)
    }

    fn build(&mut self, batches: usize, num_tokens: usize, config: &Config) {
        let in_type = NdArrayType::new(Shape(vec![batches, num_tokens]), Dtype::I32);

        let state = EvalState::build(|builder| {
            let x = Var::new(builder.clone(), in_type.clone());
            let mut cache = Cache::init(
                builder,
                config,
                self.total_tokens + num_tokens,
                self.use_kv_cache,
            );

            if self.use_kv_cache {
                // Shape of KV cache entries up to current sequence length
                let k_cache_type = NdArrayType::new(
                    Shape(vec![
                        batches,
                        config.get_num_kv_heads(),
                        self.total_tokens,
                        config.get_qk_head_dim(),
                    ]),
                    config.dtype,
                );

                let v_cache_type = NdArrayType::new(
                    Shape(vec![
                        batches,
                        config.get_num_kv_heads(),
                        self.total_tokens,
                        config.get_v_head_dim(),
                    ]),
                    config.dtype,
                );
                for layer_id in 0..config.num_hidden_layers {
                    cache.in_kv_cache[layer_id] = (
                        Var::new(builder.clone(), k_cache_type.clone()),
                        Var::new(builder.clone(), v_cache_type.clone()),
                    );
                }
            }

            let start_pos = if self.use_kv_cache {
                self.total_tokens
            } else {
                0
            };

            let result = self
                .model
                .build(builder, config, &mut cache, start_pos, x.clone());

            // Input most recently generated token and current kv_cache
            let mut sources_vec = vec![x];

            if self.use_kv_cache {
                for layer_id in 0..config.num_hidden_layers {
                    sources_vec.push(cache.in_kv_cache[layer_id].0.clone());
                    sources_vec.push(cache.in_kv_cache[layer_id].1.clone());
                }
            }

            // Output new token and updated kv_cache
            let new_token = self.next_token(builder, result);
            let mut targets_vec = vec![new_token];

            if self.use_kv_cache {
                let out_kv_cache: Vec<_> = cache
                    .out_kv_cache
                    .into_iter()
                    .flat_map(|(a, b)| vec![a, b])
                    .collect();

                targets_vec.extend(out_kv_cache);
            }

            (sources_vec, targets_vec)
        });

        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors));
    }

    fn new(config: &Config, tokenizer: Tokenizer, use_kv_cache: bool, use_fp16: bool) -> Self {
        let arch = config.architectures[0].as_str();
        let model = get_model(arch).expect("Unknown architecture {arch}");
        let kv_cache = if use_kv_cache {
            let shape = Shape(vec![1, config.get_num_kv_heads(), 0, config.get_head_dim()]);

            let v = if use_fp16 {
                TaggedNdArray::F16(NdArray::new_empty(shape))
            } else {
                TaggedNdArray::F32(NdArray::new_empty(shape))
            };

            vec![v; 2 * config.num_hidden_layers]
        } else {
            vec![]
        };
        Self {
            tensors: Rc::new(HashMap::new()),
            state: None,
            model,
            tokenizer,
            use_kv_cache,
            kv_cache,
            use_fp16,
            total_tokens: 0,
        }
    }

    fn load(&mut self, model_paths: Vec<PathBuf>) {
        let mut tensors =
            read_safetensors_multiple(model_paths, self.use_fp16).expect("loading model weights");
        self.model.post_load(&mut tensors);

        self.tensors = Rc::new(tensors);
    }

    // Make a forward pass given a list of tokens
    fn run(&mut self, x: &NdArray<i32>) -> TaggedNdArray {
        let mut sources = vec![x.clone().into()];

        if self.use_kv_cache {
            // Add kv_cache to the inputs
            sources.extend(self.kv_cache.clone());
        }

        let result = self.state.as_mut().unwrap().eval_with(sources);

        if self.use_kv_cache {
            // Save kv_cache to feed into next iteration
            self.kv_cache = result[1..].iter().map(|&tensor| tensor.clone()).collect();
        }

        result[0].clone()
    }

    fn generate(&mut self, batches: usize, tokens: Vec<i32>, config: &Config) -> i32 {
        let num_tokens = tokens.len();
        let input = NdArray::new(tokens, Shape(vec![batches, num_tokens / batches]));

        self.build(batches, num_tokens, config);
        log::debug!("Model graph built...");
        let result = self.run(&input);

        let r = result.data();

        self.total_tokens += num_tokens;
        r[0] as i32
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
    #[arg(short = 'm', long, default_value = "smollm2")]
    model_name: String,

    /// Model revision on Huggingface Hub
    #[arg(long, default_value = "main")]
    revision: String,

    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Category theory is")]
    prompt: String,

    /// Pass raw prompt without chat template
    #[arg(short = 'r', long)]
    raw: bool,

    /// Number of tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    seq_len: usize,

    /// Path to save Graphviz dot file
    #[arg(long, default_value=None)]
    save_dot: Option<PathBuf>,

    /// Use KV-cache
    #[arg(short = 'k', long)]
    kv_cache: bool,

    /// Enable thinking
    #[arg(short = 't', long)]
    thinking: bool,

    /// Use F16 weights
    #[arg(short = 'f', long)]
    use_fp16: bool,

    /// Benchmark
    #[arg(
       short = 'b',
           long,
           num_args = 2,
           value_names = ["PP", "TG"]
       )]
    bench: Option<Vec<usize>>,
}

pub fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let models = HashMap::from([
        ("gpt2", "openai-community/gpt2"),
        ("gptoss", "unsloth/gpt-oss-20b-BF16"),
        ("smollm2", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        ("smollm3", "HuggingFaceTB/SmolLM3-3B"),
        ("llama", "meta-llama/Llama-3.2-1B-Instruct"),
        ("mistral", "mistralai/Ministral-8B-Instruct-2410"),
        ("gemma", "google/gemma-3-1b-it"),
        ("qwen3", "Qwen/Qwen3-0.6B"),
        ("qwen2", "Qwen/Qwen2.5-0.5B"),
        ("lfm", "LiquidAI/LFM2-350M"),
        ("qwenmoe", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        ("deepseek", "tiny-random/deepseek-v3.1"),
        ("granitemoe", "ibm-granite/granite-3.1-1b-a400m-instruct"),
        ("granite", "ibm-granite/granite-3.3-2b-instruct"),
        ("granite4", "ibm-granite/granite-4.0-micro"),
        ("olmo2", "allenai/OLMo-2-0425-1B-Instruct"),
        ("olmo3", "allenai/Olmo-3-7B-Instruct"),
        ("phi", "microsoft/Phi-4-mini-instruct"),
        ("modernbert", "jhu-clsp/ettin-decoder-17m"),
    ]);

    let model_name = models
        .get(args.model_name.as_str())
        .copied()
        .unwrap_or(&args.model_name);

    let (model_paths, config_path, tokenizer_path, _) =
        get_model_files(model_name, &args.revision).expect("loading model files");
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let mut config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    if args.use_fp16 {
        config.dtype = Dtype::F16;
    }
    let chat_template = get_model_chat_template(model_name, &args.revision).unwrap_or_default();

    // SmolLM3 template specific hack, move to lib.
    let chat_template = chat_template
        .replace("{% generation %}", "")
        .replace("{% endgeneration %}", "");

    let mut seq_len = args.seq_len;

    let benchmarking = args.bench.is_some();

    let mut pp = 0;
    let mut tg = 0;
    let prompt = if let Some(bench) = args.bench {
        pp = bench[0];
        tg = bench[1];
        println!(
            "Benchmarking {} with prefill size {} and sequence length {}",
            model_name, pp, tg
        );
        let prompt = "The".repeat(pp);
        seq_len = tg;
        prompt
    } else if chat_template.is_empty() || args.raw {
        args.prompt.clone()
    } else {
        render_chat_template(&chat_template, &args.prompt, false, args.thinking).unwrap()
    };

    let encoding = tokenizer.encode(prompt.clone(), true)?;

    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    let tokens = token_ids.len();
    let batches = 1;

    let mut model_runner = ModelRunner::new(&config, tokenizer, args.kv_cache, args.use_fp16);

    if let Some(ref path) = args.save_dot {
        model_runner.save_dot(&config, path);
        println!(
            "Model graph for {} written to {}",
            model_name,
            path.display()
        );
        return Ok(());
    }

    let start_load = std::time::Instant::now();
    model_runner.load(model_paths);
    let elapsed_load = start_load.elapsed();
    println!(
        "Model weights loaded for {model_name} in {:.2} seconds",
        elapsed_load.as_secs_f64()
    );

    let input = NdArray::new(token_ids, Shape(vec![batches, tokens]));
    log::info!("Input tokens {:?}", &input);
    let mut input_tokens = input.data.borrow().clone();

    let mut start_gen = std::time::Instant::now();
    let mut elapsed_pp = std::time::Duration::ZERO;

    let mut generated_tokens = 0;

    if !benchmarking {
        print!("{prompt}");
    }
    for i in 0..seq_len {
        let next_token_id = model_runner.generate(batches, input_tokens.clone(), &config);
        if i == 0 {
            elapsed_pp = start_gen.elapsed();
            start_gen = std::time::Instant::now();
        }
        generated_tokens += 1;
        if config.get_eos_token_ids().contains(&next_token_id) && !benchmarking {
            break;
        }
        if !benchmarking {
            print!(
                "{}",
                model_runner
                    .tokenizer
                    .decode(&[next_token_id as u32], false)?
            );
            std::io::stdout().flush()?;
        }
        if model_runner.use_kv_cache {
            input_tokens = vec![next_token_id];
        } else {
            input_tokens.push(next_token_id);
        }
    }

    let elapsed_gen = start_gen.elapsed();
    println!(
        "\n{} tokens generated in {} seconds. ({:.2} tps)",
        generated_tokens,
        (elapsed_pp + elapsed_gen).as_secs(),
        generated_tokens as f64 / (elapsed_pp + elapsed_gen).as_secs_f64(),
    );

    if benchmarking {
        println!(
            "PP {pp} in {} ms {:.2} tps",
            elapsed_pp.as_millis(),
            pp as f64 / elapsed_pp.as_secs_f64()
        );
        println!(
            "TG {tg} in {} ms {:.2} tps",
            elapsed_gen.as_millis(),
            tg as f64 / elapsed_gen.as_secs_f64()
        );
    }

    Ok(())
}
