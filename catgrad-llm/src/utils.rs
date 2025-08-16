use crate::{LLMError, Result};
use catgrad::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad::core::Shape;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::path::PathBuf;

fn read_safetensors_file(path: impl AsRef<Path>) -> Result<HashMap<String, TaggedNdArray>> {
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
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                map.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            // cast BF16 to F32 until we support BF16
            safetensors::Dtype::BF16 => {
                let data: Vec<f32> = tensor_data
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect();
                map.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            safetensors::Dtype::I64 => {
                log::warn!("Ignoring I64 tensor: {name}");
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
) -> Result<HashMap<String, TaggedNdArray>> {
    let mut map = HashMap::new();
    for path in paths {
        let file_map = read_safetensors_file(path)?;
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
