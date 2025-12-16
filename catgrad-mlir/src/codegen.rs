use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use tempdir::TempDir;

#[derive(Debug)]
pub enum CodegenError {
    IoError(std::io::Error),
    ProcessError(String),
    Utf8Error(std::string::FromUtf8Error),
    MlirOptError(std::io::Error),
}

impl From<std::io::Error> for CodegenError {
    fn from(err: std::io::Error) -> Self {
        CodegenError::IoError(err)
    }
}

impl From<std::string::FromUtf8Error> for CodegenError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        CodegenError::Utf8Error(err)
    }
}

/// Transform MLIR text into a shared library (.so)
///
/// 1. Lower MLIR through the LLVM pipeline (`mlir-opt`)
/// 2. Translate to LLVM IR (`mlir-translate`)
/// 3. Compile to object file (llc)
/// 4. Link to shared library (clang)
pub fn codegen<P: AsRef<Path>>(mlir_text: &str, output_so: P) -> Result<(), CodegenError> {
    #[cfg(feature = "codegen-script")]
    if let Ok(path) = std::env::var("CATGRAD_MLIR_CODEGEN_SCRIPT") {
        return codegen_with_script(mlir_text, &path, output_so);
    }

    let lowered_mlir = lower_mlir(mlir_text)?;
    let llvm_ir = mlir_to_llvm_ir(&lowered_mlir)?;
    compile_to_shared_lib(&llvm_ir, output_so)?;
    Ok(())
}

/// Transform MLIR text into a shared library (.so)
/// using an external script
#[cfg(feature = "codegen-script")]
fn codegen_with_script<P: AsRef<Path>>(
    mlir_text: &str,
    script: &str,
    output_so: P,
) -> Result<(), CodegenError> {
    let mut child = Command::new(script)
        .arg(output_so.as_ref())
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| {
            CodegenError::ProcessError(format!("Failed to spawn codegen script {}: {}", script, e))
        })?;

    child
        .stdin
        .take()
        .unwrap()
        .write_all(mlir_text.as_bytes())?;

    let status = child.wait()?;
    if !status.success() {
        return Err(CodegenError::ProcessError(format!(
            "Codegen script ({}): {}",
            script, status
        )));
    }

    Ok(())
}

/// Lower MLIR through the LLVM pipeline
fn lower_mlir(mlir_text: &str) -> Result<String, CodegenError> {
    let mut opt_cmd = Command::new("mlir-opt")
        .arg("-")
        .arg("--mlir-print-debuginfo")
        .arg("--convert-elementwise-to-linalg")
        .arg("--linalg-fuse-elementwise-ops")
        .arg("--one-shot-bufferize=bufferize-function-boundaries")
        .arg("--convert-linalg-to-loops")
        .arg("--convert-scf-to-cf")
        .arg("--expand-strided-metadata")
        .arg("--lower-affine")
        .arg("--finalize-memref-to-llvm")
        .arg("--convert-math-to-llvm")
        .arg("--convert-arith-to-llvm")
        .arg("--convert-func-to-llvm")
        .arg("--convert-cf-to-llvm")
        .arg("--ensure-debug-info-scope-on-llvm-func")
        .arg("--reconcile-unrealized-casts")
        .arg("--remove-dead-values")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(CodegenError::MlirOptError)?;

    if let Some(mut stdin) = opt_cmd.stdin.take() {
        stdin.write_all(mlir_text.as_bytes())?;
    }

    let lowered_output = opt_cmd.wait_with_output()?;

    if !lowered_output.status.success() {
        return Err(CodegenError::ProcessError(format!(
            "mlir-opt failed: {}",
            String::from_utf8(lowered_output.stderr)?
        )));
    }

    let lowered_mlir = String::from_utf8(lowered_output.stdout)?;
    Ok(lowered_mlir)
}

/// Translate lowered MLIR to LLVM IR
fn mlir_to_llvm_ir(lowered_mlir: &str) -> Result<String, CodegenError> {
    let mut translate_cmd = Command::new("mlir-translate")
        .arg("--mlir-to-llvmir")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = translate_cmd.stdin.take() {
        stdin.write_all(lowered_mlir.as_bytes())?;
    }

    let translate_result = translate_cmd.wait_with_output()?;

    if !translate_result.status.success() {
        return Err(CodegenError::ProcessError(format!(
            "mlir-translate failed: {}",
            String::from_utf8(translate_result.stderr)?
        )));
    }

    let llvm_ir = String::from_utf8(translate_result.stdout)?;
    Ok(llvm_ir)
}

/// Compile LLVM IR to shared library
fn compile_to_shared_lib<P: AsRef<Path>>(llvm_ir: &str, output_so: P) -> Result<(), CodegenError> {
    let output_so = output_so.as_ref();

    // Create temporary directory
    let temp_dir = TempDir::new("catgrad-mlir")?;

    // Write LLVM IR to temporary file for llc
    let base_name = output_so.with_extension("");
    let temp_ll = temp_dir
        .path()
        .join(format!("{}.ll", base_name.to_string_lossy()));
    fs::write(&temp_ll, llvm_ir)?;

    // Create a temporary file for the object
    let temp_obj = temp_dir
        .path()
        .join(format!("{}.o", base_name.to_string_lossy()));

    // Compile to object file
    let llc_output = Command::new("llc")
        .arg("-filetype=obj")
        .arg("-relocation-model=pic")
        .arg(&temp_ll)
        .arg("-o")
        .arg(&temp_obj)
        .output()?;

    if !llc_output.status.success() {
        return Err(CodegenError::ProcessError(format!(
            "llc failed: {}",
            String::from_utf8(llc_output.stderr)?
        )));
    }

    // Link to shared library
    let clang_output = Command::new("clang")
        .arg("-shared")
        .arg("-fPIC")
        .arg(&temp_obj)
        .arg("-o")
        .arg(output_so)
        .arg("-lm")
        // Below are needed for memrefCopy, which itself is needed because of the reshape op lowering
        // which uses non-contiguous copies.
        .arg("-lmlir_c_runner_utils")
        .output()?;

    if !clang_output.status.success() {
        return Err(CodegenError::ProcessError(format!(
            "clang failed: {}",
            String::from_utf8(clang_output.stderr)?
        )));
    }

    Ok(())
}
