use crate::{LLMError, Result};
use catgrad::prelude::Module;
use catgrad_legacy::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad_legacy::core::Shape;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::path::PathBuf;

use crate::models::*;
use rayon::prelude::*;

pub fn read_safetensors_file(
    path: impl AsRef<Path>,
    use_fp16: bool,
) -> Result<HashMap<String, TaggedNdArray>> {
    let file = std::fs::File::open(path)?;
    let data = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    // Read each tensor
    let mut map = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape = Shape(view.shape().to_vec());
        let tensor_data = view.data();

        // Convert dtype and load tensor data
        match view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = tensor_data
                    .par_chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                map.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            // cast BF16 to F16 or F32
            safetensors::Dtype::BF16 => {
                if use_fp16 {
                    let data: Vec<half::f16> = tensor_data
                        .par_chunks_exact(2)
                        .map(|b| {
                            let f = half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32();
                            half::f16::from_f32(f)
                        })
                        .collect();
                    map.insert(
                        name.to_string(),
                        TaggedNdArray::F16(NdArray::new(data, shape)),
                    );
                } else {
                    let data: Vec<f32> = tensor_data
                        .par_chunks_exact(2)
                        .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                        .collect();
                    map.insert(
                        name.to_string(),
                        TaggedNdArray::F32(NdArray::new(data, shape)),
                    );
                }
            }
            safetensors::Dtype::I32 => {
                let data: Vec<i32> = tensor_data
                    .par_chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                map.insert(
                    name.to_string(),
                    TaggedNdArray::I32(NdArray::new(data, shape)),
                );
            }
            safetensors::Dtype::I64 => {
                let data: Vec<i32> = tensor_data
                    .par_chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as i32)
                    .collect();
                map.insert(
                    name.to_string(),
                    TaggedNdArray::I32(NdArray::new(data, shape)),
                );
            }
            // Add other dtype conversions as needed
            _ => {
                return Err(LLMError::UnsupportedDtype(format!("{:?}", view.dtype())));
            }
        }
    }

    Ok(map)
}

pub fn read_safetensors_multiple(
    paths: impl IntoIterator<Item = impl AsRef<Path>>,
    use_fp16: bool,
) -> Result<HashMap<String, TaggedNdArray>> {
    let mut map = HashMap::new();
    for path in paths {
        let file_map = read_safetensors_file(path, use_fp16)?;
        map.extend(file_map);
    }
    Ok(map)
}

pub fn get_model_files(
    model: &str,
    revision: &str,
) -> Result<(Vec<PathBuf>, PathBuf, PathBuf, PathBuf)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Get the model.safetensor file(s)
    let m = if let Ok(index) = repo.get("model.safetensors.index.json") {
        let index = std::fs::File::open(index)?;
        let json: serde_json::Value = serde_json::from_reader(&index)?;

        let mut weight_files = HashSet::new();
        if let Some(weight_map) = json
            .get("weight_map")
            .ok_or(LLMError::InvalidModelConfig(
                "Missing field `weight_map`".to_string(),
            ))?
            .as_object()
        {
            for v in weight_map.values() {
                let filename = v.as_str().ok_or(LLMError::InvalidModelConfig(
                    "Weight map contained non-string values".to_string(),
                ))?;
                let contents = repo.get(filename)?;
                weight_files.insert(contents);
            }
        }
        weight_files.into_iter().collect()
    } else {
        vec![repo.get("model.safetensors")?]
    };

    let c = repo.get("config.json")?;
    let t = repo.get("tokenizer.json")?;
    let tc = repo.get("tokenizer_config.json")?;

    Ok((m, c, t, tc))
}

// Try getting the model's chat template from the repository
pub fn get_model_chat_template(model: &str, revision: &str) -> Result<String> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    if let Ok(ct) = repo.get("chat_template.jinja") {
        Ok(std::fs::read_to_string(ct)?)
    } else {
        let tc_path = repo.get("tokenizer_config.json")?;
        let tc = std::fs::read_to_string(tc_path)?;
        let tokenizer_config: serde_json::Value = serde_json::from_str(&tc)?;
        Ok(tokenizer_config
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or(LLMError::InvalidModelConfig(
                "Missing or invalid `chat_template` in tokenizer config".to_string(),
            ))?
            .to_string())
    }
}

use crate::legacy::models::utils::Config;

pub fn get_model(config: &Config, max_sequence_length: usize) -> Result<Box<dyn Module<1, 1>>> {
    let arch = config.architectures[0].as_str();
    match arch {
        "LlamaForCausalLM" => Ok(Box::new(llama::LlamaModel {
            config: config.clone(),
            max_sequence_length,
        })),
        "Phi3ForCausalLM" | "Phi4MMForCausalLM" => Ok(Box::new(phi3::Phi3Model {
            config: config.clone(),
            max_sequence_length,
        })),
        "Gemma3ForCausalLM" => Ok(Box::new(gemma3::Gemma3Model {
            config: config.clone(),
            max_sequence_length,
        })),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => Ok(Box::new(qwen3::Qwen3Model {
            config: config.clone(),
            max_sequence_length,
        })),
        "GraniteForCausalLM" | "GraniteMoeForCausalLM" => Ok(Box::new(granite::GraniteModel {
            config: config.clone(),
            max_sequence_length,
        })),
        "DeepseekV3ForCausalLM" => Ok(Box::new(deepseek::DeepSeekModel {
            config: config.clone(),
            max_sequence_length,
        })),
        "GPT2LMHeadModel" => Ok(Box::new(gpt2::GPT2Model {
            config: config.clone(),
            max_sequence_length,
        })),
        _ => Err(LLMError::UnsupportedModel(arch.to_string())),
    }
}

use catgrad::interpreter;
use catgrad::prelude::path;
use catgrad::typecheck;

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
                return Err(LLMError::InvalidModelConfig(format!(
                    "Expected {} experts for layer {} {}, found {}",
                    config.num_local_experts,
                    layer_idx,
                    proj_name,
                    expert_tensors.len()
                )));
            }

            let original_shape = expert_tensors[0].shape();
            let original_dims = original_shape.0.clone();

            let mut new_shape_dims = vec![config.num_local_experts];
            new_shape_dims.extend(original_dims.clone());

            let mut reshaped_tensors = Vec::new();
            for tensor in expert_tensors {
                let mut reshape_dims = vec![1];
                reshape_dims.extend(original_dims.clone());
                let reshaped = backend.reshape(tensor, interpreter::Shape(reshape_dims));
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

pub fn post_process_weights<B: interpreter::Backend>(
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

use catgrad::prelude::Dtype;
use tokenizers::tokenizer::Tokenizer;

pub fn load_model<B: interpreter::Backend>(
    model_name: &str,
    revision: &str,
    backend: &B,
) -> Result<(
    interpreter::Parameters<B>,
    typecheck::Parameters,
    Config,
    Tokenizer,
)> {
    let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, revision)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| LLMError::TokenizerError(format!("tokenizer load error {:?}", err)))?;

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

            let tensor = interpreter::tensor(backend, interpreter::Shape(shape.clone()), data)
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
