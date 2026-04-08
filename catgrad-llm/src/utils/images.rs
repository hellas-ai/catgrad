use crate::{LLMError, Result};
use std::path::{Path, PathBuf};
use std::{fs::File, io::Read};

#[derive(Debug, Clone)]
pub struct PatchedImageInput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub patch_grid_height: usize,
    pub patch_grid_width: usize,
}

// SigLIP-style image preprocessing: square resize/crop to CHW with pixels in [-1, 1].
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

fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let total_px = height * width;
    let target_px = max_patches * patch_size * patch_size;
    let factor = (target_px as f64 / total_px as f64).sqrt();
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_height = ((factor * height as f64).floor() as usize / side_mult) * side_mult;
    let mut target_width = ((factor * width as f64).floor() as usize / side_mult) * side_mult;

    if target_height == 0 && target_width == 0 {
        return Err(LLMError::InvalidModelConfig(format!(
            "gemma4 resized image would be 0x0 for patch_size {patch_size} and pooling_kernel_size {pooling_kernel_size}"
        )));
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        target_width = (((width as f64 / height as f64).floor() as usize).max(1) * side_mult)
            .min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height = (((height as f64 / width as f64).floor() as usize).max(1) * side_mult)
            .min(max_side_length);
    }

    if target_height * target_width > target_px {
        return Err(LLMError::InvalidModelConfig(format!(
            "gemma4 resized image {target_height}x{target_width} exceeds patch budget {max_patches}"
        )));
    }

    Ok((target_height, target_width))
}

fn convert_image_to_patches(
    image: &[f32],
    height: usize,
    width: usize,
    patch_size: usize,
) -> Vec<f32> {
    let num_channels = 3;
    let num_patches_height = height / patch_size;
    let num_patches_width = width / patch_size;
    let patch_dim = num_channels * patch_size * patch_size;
    let mut patches = vec![0.0; num_patches_height * num_patches_width * patch_dim];

    for patch_row in 0..num_patches_height {
        for patch_col in 0..num_patches_width {
            let patch_idx = patch_row * num_patches_width + patch_col;
            for inner_row in 0..patch_size {
                for inner_col in 0..patch_size {
                    for chan in 0..num_channels {
                        let src_row = patch_row * patch_size + inner_row;
                        let src_col = patch_col * patch_size + inner_col;
                        let src_idx = chan * height * width + src_row * width + src_col;
                        let dst_idx = patch_idx * patch_dim
                            + (inner_row * patch_size + inner_col) * num_channels
                            + chan;
                        patches[dst_idx] = image[src_idx];
                    }
                }
            }
        }
    }

    patches
}

#[derive(Debug, Clone)]
struct ResampleWeights {
    left: usize,
    weights: Vec<f64>,
}

fn catmull_rom_kernel(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.5 * x * x * x - 2.5 * x * x + 1.0
    } else if x < 2.0 {
        -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
    } else {
        0.0
    }
}

fn build_resample_weights(input_len: usize, output_len: usize) -> Vec<ResampleWeights> {
    let ratio = input_len as f64 / output_len as f64;
    let scale = ratio.max(1.0);
    let support = 2.0 * scale;

    (0..output_len)
        .map(|out_idx| {
            let sample_center = (out_idx as f64 + 0.5) * ratio;
            let left = ((sample_center - support).floor() as isize)
                .clamp(0, input_len.saturating_sub(1) as isize) as usize;
            let right = ((sample_center + support).ceil() as isize)
                .clamp((left + 1) as isize, input_len as isize) as usize;
            let sample_center = sample_center - 0.5;
            let mut weights: Vec<f64> = (left..right)
                .map(|src_idx| catmull_rom_kernel((src_idx as f64 - sample_center) / scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            ResampleWeights { left, weights }
        })
        .collect()
}

fn round_to_u8(x: f64) -> u8 {
    x.clamp(0.0, 255.0).round() as u8
}

fn resize_rgb_bicubic(
    image: &image::RgbImage,
    new_width: usize,
    new_height: usize,
) -> image::RgbImage {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let src = image.as_raw();
    let x_weights = build_resample_weights(width, new_width);
    let y_weights = build_resample_weights(height, new_height);

    let mut horizontal = vec![0u8; height * new_width * 3];
    for row in 0..height {
        for (out_col, weights) in x_weights.iter().enumerate() {
            let mut rgb = [0.0; 3];
            for (idx, &weight) in weights.weights.iter().enumerate() {
                let src_col = weights.left + idx;
                let src_base = (row * width + src_col) * 3;
                for chan in 0..3 {
                    rgb[chan] += src[src_base + chan] as f64 * weight;
                }
            }
            let out_base = (row * new_width + out_col) * 3;
            for chan in 0..3 {
                horizontal[out_base + chan] = round_to_u8(rgb[chan]);
            }
        }
    }

    let mut output = vec![0u8; new_width * new_height * 3];
    for (out_row, weights) in y_weights.iter().enumerate() {
        for col in 0..new_width {
            let mut rgb = [0.0; 3];
            for (idx, &weight) in weights.weights.iter().enumerate() {
                let src_row = weights.left + idx;
                let src_base = (src_row * new_width + col) * 3;
                for chan in 0..3 {
                    rgb[chan] += horizontal[src_base + chan] as f64 * weight;
                }
            }
            let out_base = (out_row * new_width + col) * 3;
            for chan in 0..3 {
                output[out_base + chan] = round_to_u8(rgb[chan]);
            }
        }
    }

    image::RgbImage::from_raw(new_width as u32, new_height as u32, output).unwrap()
}

// Gemma4-style preprocessing: aspect-ratio-preserving resize, [0, 1] pixels, then patchify.
pub fn load_and_patchify_image(
    image_path: &Path,
    patch_size: usize,
    max_soft_tokens: usize,
    pooling_kernel_size: usize,
) -> Result<PatchedImageInput> {
    let img =
        image::open(image_path).map_err(|err| LLMError::IoError(std::io::Error::other(err)))?;
    let (width, height) = (img.width() as usize, img.height() as usize);
    let (target_height, target_width) = get_aspect_ratio_preserving_size(
        height,
        width,
        patch_size,
        max_soft_tokens,
        pooling_kernel_size,
    )?;
    let rgb = resize_rgb_bicubic(&img.to_rgb8(), target_width, target_height).into_raw();
    let pixels: Vec<f32> = rgb.iter().map(|&x| x as f32 / 255.0).collect();

    let mut chw = vec![0.0; 3 * target_height * target_width];
    for row in 0..target_height {
        for col in 0..target_width {
            for chan in 0..3 {
                chw[chan * target_height * target_width + row * target_width + col] =
                    pixels[(row * target_width + col) * 3 + chan];
            }
        }
    }

    let patch_grid_height = target_height / patch_size;
    let patch_grid_width = target_width / patch_size;
    let data = convert_image_to_patches(&chw, target_height, target_width, patch_size);
    Ok(PatchedImageInput {
        shape: vec![
            1,
            patch_grid_height * patch_grid_width,
            3 * patch_size * patch_size,
        ],
        data,
        patch_grid_height,
        patch_grid_width,
    })
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
