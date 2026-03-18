use crate::{LLMError, Result};
use std::path::{Path, PathBuf};
use std::{fs::File, io::Read};

// Loads the image and returns flattened data + shape
pub fn load_and_preprocess_image(
    image_path: &Path,
    image_size: usize,
    patch_size: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let num_channels = 3;

    let img =
        image::open(image_path).map_err(|err| LLMError::IoError(std::io::Error::other(err)))?;
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
    Ok((
        patches,
        vec![1, num_channels, aligned_image_size, aligned_image_size],
    ))
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
    let checksum: u32 = image_data.iter().map(|x| x.to_bits()).sum();
    let filename = format!(
        "{}-{}-{:08x}.bin",
        sanitize(model_name),
        sanitize(image_name),
        checksum
    );
    PathBuf::from(cache_dir).join(filename)
}

pub fn load_cached_embeddings(path: &Path) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    assert!(bytes.len() % 4 == 0);

    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

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
