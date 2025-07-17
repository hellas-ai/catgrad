use catgrad::{
    backend::cpu::{
        eval::{Builder, EvalState},
        ndarray::{NdArray, TaggedNdArray},
    },
    core::nn::layers::{argmax, cast, reshape},
    core::{Dtype, NdArrayType, Shape, Var},
};
use chrono::Local;
use clap::Parser;
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;
use tokenizers::tokenizer::{Result, Tokenizer};

use catgrad_llm::utils::{get_model_chat_template, get_model_files, read_safetensors_multiple};

use catgrad_llm::models::gemma::Model as GemmaModel;
use catgrad_llm::models::gpt2::Model as GPT2Model;
use catgrad_llm::models::llama::Model as LlamaModel;
use catgrad_llm::models::olmo::Model as OlmoModel;
use catgrad_llm::models::phi::Model as PhiModel;
use catgrad_llm::models::qwen::Model as QwenModel;
use catgrad_llm::models::smollm3::Model as SmolLM3Model;
use catgrad_llm::models::utils::{Cache, Config, ModelBuilder};

struct ModelRunner {
    pub tensors: Rc<HashMap<String, TaggedNdArray>>,
    pub state: Option<EvalState>,
    pub model: Box<dyn ModelBuilder>,
    pub tokenizer: Tokenizer,
    pub use_kv_cache: bool,
    pub kv_cache: Vec<TaggedNdArray>,
    pub total_tokens: usize,
}

impl ModelRunner {
    fn next_token(&self, builder: &Builder, logits: Var) -> Var {
        let batches = logits.label.shape.0[0];
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
                let kv_cache_type = NdArrayType::new(
                    Shape(vec![
                        batches,
                        config.get_num_kv_heads(),
                        self.total_tokens,
                        config.get_head_dim(),
                    ]),
                    Dtype::F32,
                );

                for layer_id in 0..config.num_hidden_layers {
                    cache.in_kv_cache[layer_id] = (
                        Var::new(builder.clone(), kv_cache_type.clone()),
                        Var::new(builder.clone(), kv_cache_type.clone()),
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

    fn new(config: &Config, tokenizer: Tokenizer, use_kv_cache: bool) -> Self {
        let arch = config.architectures[0].as_str();
        let model: Box<dyn ModelBuilder> = match arch {
            "LlamaForCausalLM" => Box::new(LlamaModel {}),
            "Olmo2ForCausalLM" => Box::new(OlmoModel {}),
            "Qwen3ForCausalLM" => Box::new(QwenModel {}),
            "Gemma3ForCausalLM" => Box::new(GemmaModel {}),
            "Phi3ForCausalLM" => Box::new(PhiModel {}),
            "SmolLM3ForCausalLM" => Box::new(SmolLM3Model {}),
            "GPT2LMHeadModel" => Box::new(GPT2Model {}),
            _ => panic!("Unknown architecture {arch}"),
        };
        let kv_cache = if use_kv_cache {
            let v = TaggedNdArray::F32(NdArray::new_empty(Shape(vec![
                1,
                config.get_num_kv_heads(),
                0,
                config.get_head_dim(),
            ])));
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
            total_tokens: 0,
        }
    }

    fn load(&mut self, model_paths: Vec<PathBuf>) {
        let mut tensors = read_safetensors_multiple(model_paths);
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

    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Hello world")]
    prompt: String,

    /// Pass raw prompt without chat template
    #[arg(short = 'r', long)]
    raw_prompt: bool,

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
}

fn strftime_now(format_str: String) -> String {
    Local::now().format(&format_str).to_string()
}

pub fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let models = HashMap::from([
        ("gpt", "openai-community/gpt2"),
        ("smollm2", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        ("smollm3", "HuggingFaceTB/SmolLM3-3B-Base"),
        ("llama", "meta-llama/Llama-3.2-1B-Instruct"),
        ("gemma", "google/gemma-3-1b-it"),
        ("qwen", "Qwen/Qwen3-0.6B"),
        ("olmo", "allenai/OLMo-2-0425-1B-Instruct"),
        ("phi", "microsoft/Phi-4-mini-instruct"),
    ]);

    let model_name = models
        .get(args.model_name.as_str())
        .copied()
        .unwrap_or(&args.model_name);

    let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name);
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    let chat_template = get_model_chat_template(model_name);

    // SmolLM3 template specific hack, move to lib.
    let chat_template = chat_template
        .replace("{% generation %}", "")
        .replace("{% endgeneration %}", "");

    let prompt = if chat_template.is_empty() || args.raw_prompt {
        args.prompt.clone()
    } else {
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_function("strftime_now", strftime_now);
        env.add_template("chat", &chat_template).unwrap();
        let tmpl = env.get_template("chat").unwrap();
        tmpl.render(
                context!(messages => vec![ context!(role => "user",content => args.prompt)], add_generation_prompt => true, enable_thinking=>args.thinking)
            )?
    };

    let encoding = tokenizer.encode(prompt.clone(), true)?;

    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
    let tokens = token_ids.len();
    let batches = 1;

    let mut model_runner = ModelRunner::new(&config, tokenizer, args.kv_cache);

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
    let mut input_tokens = input.data.borrow().clone();

    let start_gen = std::time::Instant::now();

    let mut generated_tokens = 0;

    print!("{prompt}");
    for _ in 0..args.seq_len {
        let next_token_id = model_runner.generate(batches, input_tokens.clone(), &config);
        generated_tokens += 1;
        if config.get_eos_token_ids().contains(&next_token_id) {
            break;
        }
        print!(
            "{}",
            model_runner
                .tokenizer
                .decode(&[next_token_id as u32], false)?
        );
        std::io::stdout().flush()?;
        if model_runner.use_kv_cache {
            input_tokens = vec![next_token_id];
        } else {
            input_tokens.push(next_token_id);
        }
    }

    let elapsed = start_gen.elapsed();
    println!(
        "\n{} tokens generated in {} seconds. ({:.2} tokens/sec)",
        generated_tokens,
        elapsed.as_secs(),
        generated_tokens as f64 / elapsed.as_secs_f64(),
    );

    Ok(())
}
