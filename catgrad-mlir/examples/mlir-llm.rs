use catgrad::prelude::*;
use catgrad_mlir::{compile::CompiledModel, runtime::LlvmRuntime};

use anyhow::Result;
use std::collections::HashMap;

use catgrad_llm::legacy::models::utils::Config;
use catgrad_llm::models::*;
use catgrad_llm::utils::get_model_files;
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
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    let (param_values, parameters, config, _tokenizer) = load_model(&args.model_name)?;

    ////////////////////////////////////////
    // Setup model and environment

    let max_sequence_length = 1;

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

    let typed_term = model.term().expect("Failed to create typed term");

    // TODO: this is a lot of work for the user...?
    let mut env = stdlib();
    env.definitions.extend([(model.path(), typed_term)]);
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    ////////////////////////////////////////
    // Compile and set up runtime with compiled code
    println!("Compiling {}...", model.path());
    let compiled_model = CompiledModel::new(&env, &parameters, model.path());

    ////////////////////////////////////////
    // Execute with example data
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 10.0, 11.0, 12.0, // 2
    ];

    let input_tensor = LlvmRuntime::tensor_with_strides(input_data, vec![3, 4], vec![4, 1]);
    println!("Input tensor: {}", input_tensor);

    // Call the function using the CompiledModel API
    let prefix = model.path();
    let param_values = param_values
        .into_iter()
        .map(|(k, v)| (prefix.concat(&k), v))
        .collect();

    println!("calling...");
    let results = compiled_model.call(model.path(), &param_values, vec![input_tensor])?;

    // Print each result using Display
    for (i, result) in results.iter().enumerate() {
        println!("Output tensor {}: {}", i, result);
    }

    Ok(())
}

fn load_model(
    model_name: &str,
) -> Result<(
    HashMap<Path, catgrad_mlir::runtime::MlirValue>,
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
        config,
        tokenizer,
    ))
}
