use crate::{LLMError, Result};
use std::path::Path;

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
