use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use std::io::Write;

use std::collections::HashMap;

use catgrad_llm::models::utils::Config;
use catgrad_llm::utils::get_model_files;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tokenizers::tokenizer::Tokenizer;

mod gemma3;
mod gpt2;
mod granite;
mod helpers;
mod llama;
mod qwen3;

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
    #[arg(short = 'p', long, default_value = "Category theory is")]
    prompt: String,
    /// Tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    seq_len: usize,
    /// Enable typecheck
    #[arg(short = 't', long)]
    typecheck: bool,
    /// Backend to use
    #[arg(short = 'b', long, value_enum, default_value_t = BackendChoice::Ndarray)]
    backend: BackendChoice,
    /// Enable Candle backend acceleration
    #[arg(short = 'a', long)]
    accel: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Candle,
}

/// Construct, shapecheck, and interpret the a given LLM using the selected backend.
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    match args.backend {
        BackendChoice::Ndarray => run_with_backend(&args, NdArrayBackend),
        BackendChoice::Candle => run_with_backend(&args, CandleBackend::new_accel(args.accel)),
    }
}

fn run_with_backend<B: interpreter::Backend>(args: &Args, backend: B) -> Result<()> {
    let (interpreter_params, parameters, config, tokenizer) =
        load_model(&args.model_name, &backend)?;

    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    let model: Box<dyn Module<1, 1>> = match config.architectures[0].as_str() {
        "LlamaForCausalLM" => Box::new(llama::LlamaModel {
            config: config.clone(),
            max_sequence_length: args.seq_len + token_ids.len(),
        }),
        "Gemma3ForCausalLM" => Box::new(gemma3::Gemma3Model {
            config: config.clone(),
            max_sequence_length: args.seq_len + token_ids.len(),
        }),
        "Qwen3ForCausalLM" => Box::new(qwen3::Qwen3Model {
            config: config.clone(),
            max_sequence_length: args.seq_len + token_ids.len(),
        }),
        "GraniteForCausalLM" | "GraniteMoeForCausalLM" => Box::new(granite::GraniteModel {
            config: config.clone(),
            max_sequence_length: args.seq_len + token_ids.len(),
        }),
        "GPT2LMHeadModel" => Box::new(gpt2::GPT2Model {
            config: config.clone(),
        }),
        _ => panic!("Unsupported model architecture {}", config.architectures[0]),
    };

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Shapecheck the model
    if args.typecheck {
        typecheck::check(&env, &parameters, typed_term.clone())
            .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;
    }

    print!("{}", args.prompt);
    let start_gen = std::time::Instant::now();
    let interpreter = interpreter::Interpreter::new(backend, env, interpreter_params);
    // Run interpreter
    for _ in 0..args.seq_len {
        let next_token_id = run_interpreter(&typed_term, &interpreter, &token_ids)?;
        if config.get_eos_token_ids().contains(&(next_token_id as i32)) {
            break;
        }
        let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
        token_ids.push(next_token_id);
        print!("{}", decoded_token);
        std::io::stdout().flush()?;
    }

    let elapsed_gen = start_gen.elapsed();
    let generated_tokens = args.seq_len;
    println!(
        "\n{} tokens generated in {} seconds. ({:.2} tps)",
        generated_tokens,
        elapsed_gen.as_secs(),
        generated_tokens as f64 / elapsed_gen.as_secs_f64(),
    );
    Ok(())
}

fn run_interpreter<B: interpreter::Backend>(
    typed_term: &TypedTerm,
    interpreter: &interpreter::Interpreter<B>,
    input_data: &[u32],
) -> Result<u32> {
    let input_tensor = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, input_data.len()]),
        input_data,
    )
    .expect("Failed to create input tensor");

    // Run the model
    let mut results = interpreter
        .run(typed_term.term.clone(), vec![input_tensor])
        .expect("Failed to run inference");

    // Print info about the main output (should be the last one)
    if let Some(output) = results.pop() {
        match output {
            interpreter::Value::Tensor(arr) => match interpreter.backend.to_vec(arr) {
                interpreter::TaggedVec::U32(v) => Ok(v[v.len() - 1]),
                _ => Err(anyhow::anyhow!("Unexpected output dtype")),
            },
            t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
        }
    } else {
        Err(anyhow::anyhow!("No result"))
    }
}

/// Type signature for LLM Modules
fn llm_type() -> ([Type; 1], [Type; 1]) {
    use catgrad::typecheck::*;
    let batch_size = NatExpr::Var(0);
    let seq_len = NatExpr::Var(1);

    // Input shape B×S
    let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len]),
    }));

    // Output shape B×1
    let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(1)]),
    }));

    ([t_x], [t_y])
}

struct Cache {
    pub cos: Var,
    pub sin: Var,
}

impl Cache {
    pub fn init(builder: &Builder, config: &Config, positions: usize) -> Self {
        let (cos, sin) = helpers::rope_tables(
            builder,
            config.rope_theta,
            positions.to_nat(builder),
            config.get_head_dim(),
        );

        Self { cos, sin }
    }
}

fn load_model<B: interpreter::Backend>(
    model_name: &str,
    backend: &B,
) -> Result<(
    interpreter::Parameters<B>,
    typecheck::Parameters,
    Config,
    Tokenizer,
)> {
    let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, "main")?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| anyhow::anyhow!("tokenizer load error {:?}", err))?;

    // Read each tensor
    let mut type_map = HashMap::new();
    let mut data_map = HashMap::new();

    for file_path in model_paths {
        let file = std::fs::File::open(file_path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&data)?;

        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let tensor_data = view.data();

            use catgrad::typecheck::*;
            // Convert dtype and load tensor data
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => tensor_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                safetensors::Dtype::BF16 => tensor_data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                _ => {
                    panic!("Unsupported dtype: {:?}", view.dtype());
                }
            };

            let tensor = interpreter::tensor(backend, Shape(shape.clone()), &data)
                .expect("failed to create tensor");
            let key = path(name.split(".").collect()).expect("invalid param path");
            data_map.insert(key.clone(), tensor);

            let vne = shape.into_iter().map(NatExpr::Constant).collect();
            let tensor_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vne),
            }));
            type_map.insert(key, tensor_type);
        }
    }

    Ok((
        interpreter::Parameters::from(data_map),
        typecheck::Parameters::from(type_map),
        config,
        tokenizer,
    ))
}
