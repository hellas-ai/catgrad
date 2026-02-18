use catgrad::prelude::Dtype;
use catgrad_legacy::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad_legacy::core::Shape;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokenizers::tokenizer::Tokenizer;

use crate::config::{Config, LLMConfig};
use crate::helpers::LLMModel;
use crate::models::gemma3::{Gemma3Model, GemmaConfig, GemmaTextConfig};
use crate::models::*;
use crate::{LLMError, Result};

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
                if use_fp16 {
                    let data: Vec<half::f16> =
                        data.iter().map(|&x| half::f16::from_f32(x)).collect();
                    map.insert(
                        name.to_string(),
                        TaggedNdArray::F16(NdArray::new(data, shape)),
                    );
                } else {
                    map.insert(
                        name.to_string(),
                        TaggedNdArray::F32(NdArray::new(data, shape)),
                    );
                }
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
    let api = ApiBuilder::from_env().build()?;
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
    let api = ApiBuilder::from_env().build()?;
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

use chrono::Local;
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

fn strftime_now(format_str: String) -> String {
    Local::now().format(&format_str).to_string()
}

pub fn render_chat_template(
    chat_template: &str,
    prompt: &str,
    has_image: bool,
    enable_thinking: bool,
) -> Result<String, minijinja::Error> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("strftime_now", strftime_now);
    env.add_template("chat", chat_template).unwrap();
    let tmpl = env.get_template("chat").unwrap();
    let messages = if has_image {
        let content = vec![
            context!(type => "text", text => prompt),
            context!(type => "image"),
        ];
        vec![context!(role => "user",content => content)]
    } else {
        vec![context!(role => "user",content => prompt)]
    };
    tmpl.render(
            context!(messages => messages, add_generation_prompt => true, enable_thinking => enable_thinking)
            )
}

pub fn parse_config(config_json: &serde_json::Value) -> Result<Box<dyn LLMConfig>> {
    let arch = config_json["architectures"][0]
        .as_str()
        .ok_or(LLMError::InvalidModelConfig(
            "Missing architectures field".to_string(),
        ))?;

    match arch {
        "Gemma2ForCausalLM" | "Gemma3ForCausalLM" => {
            let config: GemmaTextConfig = serde_json::from_value(config_json.clone())?;
            Ok(Box::new(config))
        }
        "Gemma3ForConditionalGeneration" => {
            let config: GemmaConfig = serde_json::from_value(config_json.clone())?;
            match config {
                GemmaConfig::Text(config) => Ok(Box::new(config)),
                GemmaConfig::VLM { text_config, .. } => Ok(Box::new(text_config)),
            }
        }
        _ => {
            let config: Config = serde_json::from_value(config_json.clone())?;
            Ok(Box::new(config))
        }
    }
}

pub fn get_model(
    config_json: &serde_json::Value,
    max_sequence_length: usize,
) -> Result<(Box<dyn LLMModel>, Box<dyn LLMConfig>)> {
    let arch = config_json["architectures"][0]
        .as_str()
        .ok_or(LLMError::InvalidModelConfig(
            "Missing architectures field".to_string(),
        ))?;

    let config: Config = serde_json::from_value(config_json.clone())?;

    let model: Box<dyn LLMModel> = match arch {
        "Gemma2ForCausalLM" | "Gemma3ForCausalLM" => {
            let config: GemmaTextConfig = serde_json::from_value(config_json.clone())?;
            let model = Box::new(gemma3::Gemma3Model::new(
                "model",
                config.clone(),
                max_sequence_length,
            ));
            return Ok((model, Box::new(config)));
        }
        "Gemma3ForConditionalGeneration" => {
            let GemmaConfig::VLM { text_config, .. } = serde_json::from_value(config_json.clone())?
            else {
                unreachable!()
            };
            let model = Box::new(Gemma3Model::new(
                "language_model.model",
                text_config.clone(),
                max_sequence_length,
            ));
            return Ok((model, Box::new(text_config)));
        }
        "MistralForCausalLM" | "LlamaForCausalLM" => Box::new(llama::LlamaModel {
            root: "".to_string(),
            config: config.clone(),
            max_sequence_length,
        }),
        "Phi3ForCausalLM" | "Phi4MMForCausalLM" => Box::new(phi3::Phi3Model {
            config: config.clone(),
            max_sequence_length,
        }),
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => Box::new(olmo::OlmoModel {
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
        _ => panic!("Unsupported model architecture: {}", arch),
    };
    Ok((model, Box::new(config)))
}

use catgrad::interpreter;
use catgrad::prelude::path;
use catgrad::typecheck;

// Concatenates MoE expert weights from separate tensors into single tensors per layer
// to avoid the need for dynamic parameter names
fn concat_moe_experts<B: interpreter::Backend>(
    config: &dyn LLMConfig,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    use catgrad::typecheck::*;

    let proj_names = ["down_proj", "gate_proj", "up_proj"];

    for layer_idx in 0..config.num_hidden_layers() {
        for proj_name in &proj_names {
            // Collect all expert tensors for this layer and projection
            let mut expert_tensors = Vec::new();
            let mut expert_keys = Vec::new();

            for expert_idx in 0..config.num_local_experts() {
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

            if expert_tensors.len() != config.num_local_experts() {
                return Err(LLMError::InvalidModelConfig(format!(
                    "Expected {} experts for layer {} {}, found {}",
                    config.num_local_experts(),
                    layer_idx,
                    proj_name,
                    expert_tensors.len()
                )));
            }

            let original_shape = expert_tensors[0].shape();
            let original_dims = original_shape.0.clone();

            let mut new_shape_dims = vec![config.num_local_experts()];
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
    config: &dyn LLMConfig,
    backend: &B,
    parameter_values: &mut interpreter::Parameters<B>,
    parameter_types: &mut typecheck::Parameters,
) -> Result<()> {
    if config.num_local_experts() == 0 {
        return Ok(());
    }

    concat_moe_experts(config, backend, parameter_values, parameter_types)
}

pub fn load_model_weights<B: interpreter::Backend>(
    model_paths: Vec<PathBuf>,
    backend: &B,
) -> Result<(interpreter::Parameters<B>, typecheck::Parameters, usize)> {
    // Read each tensor
    let mut type_map = HashMap::new();
    let mut data_map = HashMap::new();
    let mut total_params = 0;

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
            total_params += data.len();

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

    let parameter_values = interpreter::Parameters::from(data_map);
    let parameter_types = typecheck::Parameters::from(type_map);

    Ok((parameter_values, parameter_types, total_params))
}

pub fn load_model<B: interpreter::Backend>(
    model_name: &str,
    revision: &str,
    backend: &B,
) -> Result<(
    interpreter::Parameters<B>,
    typecheck::Parameters,
    serde_json::Value,
    Tokenizer,
    usize,
)> {
    let (model_paths, config_path, tokenizer_path, _) = get_model_files(model_name, revision)?;
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|err| LLMError::TokenizerError(format!("tokenizer load error {:?}", err)))?;

    let start_load = std::time::Instant::now();

    let (mut parameter_values, mut parameter_types, total_params) =
        load_model_weights(model_paths, backend)?;

    let elapsed_load = start_load.elapsed();

    log::info!(
        "Model weights loaded for {} in {:.2} seconds",
        model_name,
        elapsed_load.as_secs_f64()
    );

    let config = parse_config(&config_json)?;

    post_process_weights(
        config.as_ref(),
        backend,
        &mut parameter_values,
        &mut parameter_types,
    )?;

    Ok((
        parameter_values,
        parameter_types,
        config_json,
        tokenizer,
        total_params,
    ))
}

// Loads the image and returns flattened data + shape
pub fn load_and_preprocess_image(
    image_path: &PathBuf,
    image_size: usize,
    patch_size: usize,
) -> (Vec<f32>, Vec<usize>) {
    let num_channels = 3;

    let img = image::open(image_path).unwrap();
    let resized_img = img.resize_to_fill(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb_img = resized_img.to_rgb8();
    let img = rgb_img.into_raw();

    let pixels: Vec<f32> = img.iter().map(|&x| x as f32 * (2. / 255.0) - 1.).collect();
    // For image sizes 384x384 we need to truncate to 378x378 so it's a multiple of patch size.
    let aligned_image_size = (image_size / patch_size) * patch_size;
    let mut patches = vec![0.0; num_channels * aligned_image_size * aligned_image_size];
    for row in 0..aligned_image_size {
        for col in 0..aligned_image_size {
            for chan in 0..num_channels {
                patches[chan * aligned_image_size * aligned_image_size
                    + row * aligned_image_size
                    + col] = pixels[(row * image_size + col) * num_channels + chan];
            }
        }
    }
    (
        patches,
        vec![1, num_channels, aligned_image_size, aligned_image_size],
    )
}
