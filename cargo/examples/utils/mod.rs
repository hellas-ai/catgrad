use catgrad::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad::core::Shape;
use std::collections::HashMap;
use std::path::PathBuf;

fn read_safetensors_file(path: PathBuf, map: &mut HashMap<String, TaggedNdArray>) {
    let file = std::fs::File::open(path).unwrap();
    let data = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

    // Read each tensor
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
                panic!("Unsupported dtype {:?}", view.dtype())
            }
        }
    }
}

#[allow(dead_code)]
pub fn read_safetensors(path: &str) -> HashMap<String, TaggedNdArray> {
    let mut map = HashMap::new();
    read_safetensors_file(PathBuf::from(path), &mut map);
    map
}

#[allow(dead_code)]
pub fn read_safetensors_multiple(path: Vec<PathBuf>) -> HashMap<String, TaggedNdArray> {
    let mut map = HashMap::new();
    for path in path {
        read_safetensors_file(path, &mut map);
    }
    map
}
