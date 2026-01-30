use catgrad::prelude::*;
use catgrad_mlir::{compile::CompiledModel, runtime::LlvmRuntime};
use std::io::Write;

use anyhow::Result;
use std::collections::HashMap;

use catgrad_llm::utils::{get_model, get_model_files};
use clap::Parser;
use tokenizers::tokenizer::Tokenizer;

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
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    let (param_values, parameters, config_json, tokenizer) = load_model(&args.model_name, "main")?;

    let encoding = tokenizer
        .encode(args.prompt.clone(), true)
        .map_err(|err| anyhow::anyhow!("tokenizer error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    let max_sequence_length = args.max_seq_len + token_ids.len();
    let (model, config) = get_model(&config_json, max_sequence_length)?;

    let typed_term = model.term().expect("Failed to create typed term");

    // TODO: this is a lot of work for the user...?
    let mut env = stdlib();
    env.definitions.extend([(model.path(), typed_term)]);
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    let prefix = model.path();
    let param_values = param_values
        .into_iter()
        .map(|(k, v)| (prefix.concat(&k), v))
        .collect();

    ////////////////////////////////////////
    // Compile and set up runtime with compiled code
    let compiled_model = CompiledModel::new(&env, &parameters, model.path());

    print!("{}", args.prompt);
    let start_gen = std::time::Instant::now();

    for _ in 0..args.max_seq_len {
        let input_tensor = LlvmRuntime::tensor_u32(token_ids.clone(), vec![1, token_ids.len()]);
        let results = compiled_model.call(model.path(), &param_values, vec![input_tensor])?;

        if let Some(output) = results.last() {
            let (ov, _) = output.to_vec_f32();
            let next_token_id = ov.last().unwrap();
            let next_token_id = next_token_id.to_bits();
            if config.get_eos_token_ids().contains(&(next_token_id as i32)) {
                break;
            }
            token_ids.push(next_token_id);
            let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
            print!("{}", decoded_token);
            std::io::stdout().flush()?;
        } else {
            break;
        }
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

fn load_model(
    model_name: &str,
    revision: &str,
) -> Result<(
    HashMap<Path, catgrad_mlir::runtime::MlirValue>,
    typecheck::Parameters,
    serde_json::Value,
    Tokenizer,
)> {
    let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, revision)?;
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
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

            let tensor = LlvmRuntime::tensor(data, shape.clone());
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
        data_map,
        typecheck::Parameters::from(type_map),
        config_json,
        tokenizer,
    ))
}
