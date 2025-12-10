use catgrad_mlir::runtime::{Entrypoint, LlvmRuntime, MlirType};
use std::ffi::CString;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let func_name = CString::new("negate_f32")?;

    // Create runtime with negate_f32 entrypoint
    let entrypoint = Entrypoint {
        func_name: func_name.clone(),
        source_types: vec![MlirType::Memref(3)], // Single 3D memref input
        target_types: vec![MlirType::Memref(3), MlirType::Memref(3)], // Two 3D memref outputs
    };

    // Initialize runtime for shared object file
    let runtime = LlvmRuntime::new(std::path::Path::new("./main.so"), vec![entrypoint])?;

    // Create input tensor using the runtime helper
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // First slice
        5.0, 6.0, 7.0, 8.0, // Second slice
        9.0, 10.0, 11.0, 12.0, // Third slice
    ];
    let input_tensor = LlvmRuntime::tensor(input_data, vec![3, 1, 4]);

    // Print the input tensor using Display
    println!("input: {}", input_tensor);

    // Call the function using the safe runtime API
    let results = runtime.call(&func_name, vec![input_tensor])?;

    // Print each result using Display
    for (i, result) in results.iter().enumerate() {
        println!("result {}: {}", i, result);
    }

    Ok(())
}
