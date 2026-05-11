use crate::model_media::PatchedImageInput;
use crate::model_utils::{PreparedMultimodalInput, prepare_multimodal_image_input};
use crate::{LLMError, Result};
use std::path::{Path, PathBuf};
use std::{fs::File, io::Read};

// SigLIP-style image preprocessing: square resize/crop to CHW with pixels in [-1, 1].
pub fn load_and_preprocess_image(
    image_path: &Path,
    image_size: usize,
    patch_size: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let img = load_image(image_path)?;
    load_and_preprocess_dynamic_image(&img, image_size, patch_size)
}

pub fn load_image(image_path: &Path) -> Result<image::DynamicImage> {
    image::open(image_path).map_err(|err| LLMError::IoError(std::io::Error::other(err)))
}

pub fn load_image_from_bytes(image_bytes: &[u8]) -> Result<image::DynamicImage> {
    image::load_from_memory(image_bytes)
        .map_err(|err| LLMError::IoError(std::io::Error::other(err)))
}

pub fn load_and_preprocess_dynamic_image(
    img: &image::DynamicImage,
    image_size: usize,
    patch_size: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    Ok(crate::model_media::load_and_preprocess_dynamic_image(
        img, image_size, patch_size,
    )?)
}

pub fn prepare_multimodal_input(
    config_json: &serde_json::Value,
    image_path: Option<&Path>,
) -> Result<PreparedMultimodalInput> {
    let Some(image_path) = image_path else {
        return Ok(PreparedMultimodalInput::default());
    };
    let image = load_image(image_path)?;
    Ok(prepare_multimodal_image_input(config_json, &image)?)
}

pub fn prepare_multimodal_input_from_bytes(
    config_json: &serde_json::Value,
    image_bytes: &[u8],
) -> Result<PreparedMultimodalInput> {
    let image = load_image_from_bytes(image_bytes)?;
    Ok(prepare_multimodal_image_input(config_json, &image)?)
}

// Gemma4-style preprocessing: aspect-ratio-preserving resize, [0, 1] pixels, then patchify.
pub fn load_and_patchify_image(
    image_path: &Path,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<PatchedImageInput> {
    let img = load_image(image_path)?;
    load_and_patchify_dynamic_image(&img, patch_size, max_soft_tokens, pooling_kernel_size)
}

pub fn load_and_patchify_dynamic_image(
    img: &image::DynamicImage,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<PatchedImageInput> {
    Ok(crate::model_media::load_and_patchify_dynamic_image(
        img,
        patch_size,
        max_soft_tokens,
        pooling_kernel_size,
    )?)
}

// Sanitize the model name to use in cache file names.
fn sanitize(model_name: &str) -> String {
    model_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

// Make a cache file name based on the model name, image name, and a simple image data checksum.
pub fn cache_path_for_embeddings(
    model_name: &str,
    image_name: &str,
    image_data: &[f32],
) -> PathBuf {
    let cache_dir = std::env::var("CATGRAD_CACHE").unwrap_or_else(|_| ".cache".to_string());
    let checksum = image_data
        .iter()
        .fold(0u32, |acc, x| acc.wrapping_add(x.to_bits()));
    let filename = format!(
        "{}-{}-{:08x}.bin",
        sanitize(model_name),
        sanitize(image_name),
        checksum
    );
    PathBuf::from(cache_dir).join(filename)
}

// Load cached image embeddings from a file to speed up processing.
pub fn load_cached_embeddings(path: &Path) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    assert!(bytes.len() % 4 == 0);

    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

// Save image embeddings to a cache file for future use.
pub fn save_cached_embeddings(path: &Path, data: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|err| LLMError::IoError(std::io::Error::other(err)))
}
