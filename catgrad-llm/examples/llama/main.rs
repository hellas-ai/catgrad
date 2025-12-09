use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::*;
use std::io::Write;
use std::path::PathBuf;

use std::collections::HashMap;

use rayon::prelude::*;

use catgrad_llm::legacy::models::utils::Config;
use catgrad_llm::utils::get_model_files;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tokenizers::tokenizer::Tokenizer;

use catgrad_llm::models::*;

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
    max_seq_len: usize,
    /// Enable typecheck
    #[arg(short = 't', long)]
    typecheck: bool,
    /// Backend to use
    #[arg(short = 'b', long, value_enum, default_value_t = BackendChoice::Ndarray)]
    backend: BackendChoice,
    /// Enable Candle backend acceleration
    #[arg(short = 'a', long)]
    accel: bool,
    /// Dump the constructed graph to this JSON file then exit.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Load model from a previously dumped JSON graph
    #[arg(long)]
    load: Option<PathBuf>,
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
    let (parameter_values, parameter_types, config, tokenizer) =
        load_model(&args.model_name, &backend)?;

    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    let max_sequence_length = args.max_seq_len + token_ids.len();
    let model: Box<dyn Module<1, 1>> = match config.architectures[0].as_str() {
        "LlamaForCausalLM" => Box::new(llama::LlamaModel {
            config: config.clone(),
            max_sequence_length,
        }),
        "Gemma3ForCausalLM" => Box::new(gemma3::Gemma3Model {
            config: config.clone(),
            max_sequence_length,
        }),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => Box::new(qwen3::Qwen3Model {
            config: config.clone(),
            max_sequence_length,
        }),
        "GraniteForCausalLM" | "GraniteMoeForCausalLM" => Box::new(granite::GraniteModel {
            config: config.clone(),
            max_sequence_length,
        }),
        "DeepseekV3ForCausalLM" => Box::new(deepseek::DeepSeekModel {
            config: config.clone(),
            max_sequence_length,
        }),
        "GPT2LMHeadModel" => Box::new(gpt2::GPT2Model {
            config: config.clone(),
            max_sequence_length,
        }),
        _ => panic!("Unsupported model architecture {}", config.architectures[0]),
    };

    let typed_term = if let Some(load_path) = &args.load {
        let file = std::fs::File::open(load_path)?;
        serde_json::from_reader(file)?
    } else {
        model.term().expect("Failed to create typed term")
    };

    if let Some(dump_path) = &args.dump {
        let file = std::fs::File::create(dump_path)?;
        serde_json::to_writer_pretty(file, &typed_term)?;
        println!(
            "Graph for {} and max_seq_length of {max_sequence_length} dumped to {}",
            model.path(),
            dump_path.display()
        );
        return Ok(());
    }

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameter_types.keys()));

    // Shapecheck the model
    if args.typecheck {
        typecheck::check(&env, &parameter_types, typed_term.clone())
            .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;
    }

    print!("{}", args.prompt);
    let start_gen = std::time::Instant::now();
    let interpreter = interpreter::Interpreter::new(backend, env, parameter_values);
    // Run interpreter
    for _ in 0..args.max_seq_len {
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
    let generated_tokens = args.max_seq_len;
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
        input_data.to_vec(),
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

// Concatenates MoE expert weights from separate tensors into single tensors per layer
// to avoid the need for dynamic parameter names
fn concat_moe_experts<B: interpreter::Backend>(
    config: &Config,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    use catgrad::typecheck::*;

    let proj_names = ["down_proj", "gate_proj", "up_proj"];

    for layer_idx in 0..config.num_hidden_layers {
        for proj_name in &proj_names {
            // Collect all expert tensors for this layer and projection
            let mut expert_tensors = Vec::new();
            let mut expert_keys = Vec::new();

            for expert_idx in 0..config.num_local_experts {
                let key_str = format!(
                    "model.layers.{}.mlp.experts.{}.{}.weight",
                    layer_idx, expert_idx, proj_name
                );
                let key = path(key_str.split(".").collect()).expect("invalid param path");

                // Check if this expert exists in the parameter maps
                if let Some(interpreter::Value::Tensor(tensor)) = parameter_values.0.get(&key) {
                    expert_tensors.push(tensor.clone());
                    expert_keys.push(key);
                }
            }

            if expert_tensors.is_empty() {
                continue;
            }

            if expert_tensors.len() != config.num_local_experts {
                return Err(anyhow::anyhow!(
                    "Expected {} experts for layer {} {}, found {}",
                    config.num_local_experts,
                    layer_idx,
                    proj_name,
                    expert_tensors.len()
                ));
            }

            let original_shape = expert_tensors[0].shape();
            let original_dims = original_shape.0.clone();

            let mut new_shape_dims = vec![config.num_local_experts];
            new_shape_dims.extend(original_dims.clone());

            let mut reshaped_tensors = Vec::new();
            for tensor in expert_tensors {
                let mut reshape_dims = vec![1];
                reshape_dims.extend(original_dims.clone());
                let reshaped = backend.reshape(tensor, Shape(reshape_dims));
                reshaped_tensors.push(reshaped);
            }

            // Concatenate all reshaped tensors along dimension 0
            // TODO: this is naive and slow. Either preallocate or fuse this with the safetensors loading code.
            let mut concatenated = reshaped_tensors[0].clone();
            for tensor in &reshaped_tensors[1..] {
                concatenated = backend.concat(concatenated, tensor.clone(), 0);
            }

            let new_key_str = format!(
                "model.layers.{}.mlp.experts.{}.weight",
                layer_idx, proj_name
            );
            let new_key = path(new_key_str.split(".").collect()).expect("invalid param path");

            parameter_values
                .0
                .insert(new_key.clone(), interpreter::Value::Tensor(concatenated));

            let vne: Vec<NatExpr> = new_shape_dims.into_iter().map(NatExpr::Constant).collect();
            let tensor_type = typecheck::Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vne),
            }));
            parameter_types.0.insert(new_key, tensor_type);

            // Remove original experts
            for key in expert_keys {
                parameter_values.0.remove(&key);
                parameter_types.0.remove(&key);
            }
        }
    }

    Ok(())
}

fn post_process_weights<B: interpreter::Backend>(
    config: &Config,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    if config.num_local_experts == 0 {
        return Ok(());
    }

    concat_moe_experts(config, backend, parameter_values, parameter_types)
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

    let start_load = std::time::Instant::now();
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
                    .par_chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                safetensors::Dtype::BF16 => tensor_data
                    .par_chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                _ => {
                    panic!("Unsupported dtype: {:?}", view.dtype());
                }
            };

            let tensor = interpreter::tensor(backend, Shape(shape.clone()), data)
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

    let mut parameter_values = interpreter::Parameters::from(data_map);
    let mut parameter_types = typecheck::Parameters::from(type_map);

    let elapsed_load = start_load.elapsed();
    log::info!(
        "Model weights loaded for {} in {:.2} seconds",
        model_name,
        elapsed_load.as_secs_f64()
    );
    post_process_weights(
        &config,
        backend,
        &mut parameter_values,
        &mut parameter_types,
    )?;

    Ok((parameter_values, parameter_types, config, tokenizer))
}
