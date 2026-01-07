//! # MLIR LLVM Runtime
//!
//! Provides safe RAII wrappers around:
//!
//! - dlopen/dlclose and function symbol resolution (TODO: and calling).
//! - Simplified MLIR type creation/destruction
use libffi::middle::CodePtr;
use libffi::raw;
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::Path;
use std::rc::Rc;

////////////////////////////////////////////////////////////////////////////////
// Runtime

#[derive(Debug)]
pub enum RuntimeError {
    LibraryLoadError(String),
    SymbolNotFound(String),
    TypeError(String),
    InvalidPath(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::LibraryLoadError(msg) => write!(f, "Library load error: {}", msg),
            RuntimeError::SymbolNotFound(msg) => write!(f, "Symbol not found: {}", msg),
            RuntimeError::InvalidPath(msg) => write!(f, "Invalid path: {}", msg),
            RuntimeError::TypeError(msg) => write!(f, "Type error: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// A loaded .so file which was compiled from a catgrad model lowered through MLIR.
/// Abstractly, one can think of this as a dictionary of functions which can be called.
pub struct LlvmRuntime {
    lib_handle: *mut c_void,
    entrypoints: HashMap<CString, (CodePtr, Entrypoint)>,
}

pub struct Entrypoint {
    pub func_name: CString,
    pub source_types: Vec<MlirType>,
    pub target_types: Vec<MlirType>,
}

impl Drop for LlvmRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.lib_handle.is_null() {
                libc::dlclose(self.lib_handle);
            }
        }
    }
}

impl LlvmRuntime {
    /// Create a new runtime from a path to a .so file.
    /// Loads all specified entrypoints and verifies they exist.
    pub fn new(so_path: &Path, entrypoints: Vec<Entrypoint>) -> Result<LlvmRuntime, RuntimeError> {
        // Convert path to C string
        let path_str = so_path
            .to_str()
            .ok_or_else(|| RuntimeError::InvalidPath("Path contains invalid UTF-8".to_string()))?;
        let lib_path = CString::new(path_str)
            .map_err(|_| RuntimeError::InvalidPath("Path contains null bytes".to_string()))?;

        unsafe {
            // Load the shared library, resolve missing symbols now
            let handle = libc::dlopen(lib_path.as_ptr(), libc::RTLD_NOW);
            if handle.is_null() {
                let error = libc::dlerror();
                let error_msg = if !error.is_null() {
                    std::ffi::CStr::from_ptr(error)
                        .to_string_lossy()
                        .to_string()
                } else {
                    "Unknown dlopen error".to_string()
                };
                return Err(RuntimeError::LibraryLoadError(error_msg));
            }

            // Load and verify all entrypoints
            let mut loaded_entrypoints = HashMap::new();
            for entrypoint in entrypoints {
                let func_ptr = libc::dlsym(handle, entrypoint.func_name.as_ptr());
                if func_ptr.is_null() {
                    let error = libc::dlerror();
                    let error_msg = if !error.is_null() {
                        std::ffi::CStr::from_ptr(error)
                            .to_string_lossy()
                            .to_string()
                    } else {
                        format!(
                            "Symbol '{}' not found",
                            entrypoint.func_name.to_string_lossy()
                        )
                    };
                    libc::dlclose(handle); // Clean up on error
                    return Err(RuntimeError::SymbolNotFound(error_msg));
                }

                let code_ptr = CodePtr(func_ptr);
                loaded_entrypoints.insert(entrypoint.func_name.clone(), (code_ptr, entrypoint));
            }

            Ok(LlvmRuntime {
                lib_handle: handle,
                entrypoints: loaded_entrypoints,
            })
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![1];
        for dim in shape.iter().skip(1).rev() {
            strides.push(strides.last().unwrap() * dim);
        }
        strides.reverse();
        strides
    }

    /// Create a tensor from an f32 buffer with separate extents and strides
    ///
    /// This takes ownership of the data vector and manages it safely.
    pub fn tensor(data: Vec<f32>, extents: Vec<usize>) -> MlirValue {
        let strides = LlvmRuntime::compute_strides(&extents);
        LlvmRuntime::tensor_with_strides(data, extents, strides)
    }

    pub fn tensor_with_strides(
        data: Vec<f32>,
        extents: Vec<usize>,
        strides: Vec<usize>,
    ) -> MlirValue {
        let (data_ptr, len, capacity) = into_raw_parts(data);

        let sizes: Vec<i64> = extents.into_iter().map(|s| s as i64).collect();
        let strides: Vec<i64> = strides.into_iter().map(|s| s as i64).collect();

        MlirValue::MlirTensorF32(MlirTensor {
            allocated: MlirBuffer::Rust(Rc::new(data_ptr), len, capacity),
            aligned: data_ptr, // For user-created tensors, allocated == aligned
            offset: 0,
            sizes,
            strides,
        })
    }

    pub fn tensor_u32(data: Vec<u32>, extents: Vec<usize>) -> MlirValue {
        let strides = LlvmRuntime::compute_strides(&extents);
        LlvmRuntime::tensor_with_strides_u32(data, extents, strides)
    }

    pub fn tensor_with_strides_u32(
        data: Vec<u32>,
        extents: Vec<usize>,
        strides: Vec<usize>,
    ) -> MlirValue {
        let (data_ptr, len, capacity) = into_raw_parts(data);

        let sizes: Vec<i64> = extents.into_iter().map(|s| s as i64).collect();
        let strides: Vec<i64> = strides.into_iter().map(|s| s as i64).collect();

        MlirValue::MlirTensorU32(MlirTensor {
            allocated: MlirBuffer::Rust(Rc::new(data_ptr), len, capacity),
            aligned: data_ptr, // For user-created tensors, allocated == aligned
            offset: 0,
            sizes,
            strides,
        })
    }

    pub fn call(
        &self,
        name: &CString,
        args: Vec<MlirValue>,
    ) -> Result<Vec<MlirValue>, RuntimeError> {
        // Get the entrypoint and verify it exists
        let (func_ptr, entrypoint) = self.get_entrypoint(name).ok_or_else(|| {
            RuntimeError::SymbolNotFound(format!("Function '{}' not found", name.to_string_lossy()))
        })?;

        // Verify argument types match
        // NOTE: this only checks ranks; we must also check that the statically required dimensions
        // of the typed term match the provided data dimensions!
        let actual_source_types: Vec<MlirType> = args.iter().map(|a| a.to_type()).collect();

        let m = actual_source_types.len();
        let n = entrypoint.source_types.len();
        if m != n {
            return Err(RuntimeError::TypeError(format!(
                "Function '{}' arity mismatch: expected {:?}, got {:?}",
                name.to_string_lossy(),
                entrypoint.source_types,
                args.iter().map(|a| a.to_type()),
            )));
        }

        for (i, (expected, actual)) in entrypoint
            .source_types
            .iter()
            .zip(actual_source_types.iter())
            .enumerate()
        {
            if actual != expected {
                return Err(RuntimeError::TypeError(format!(
                    "Function '{}' argument {i} type mismatch: expected {expected:?}, got {actual:?}",
                    name.to_string_lossy(),
                )));
            }
        }

        // Call the internal helper function
        let result_tensors = call(func_ptr, args, entrypoint.target_types.clone());

        // Convert MlirTensor results back to MlirValue
        let results: Vec<MlirValue> = result_tensors
            .into_iter()
            .map(MlirValue::MlirTensorF32)
            .collect();

        Ok(results)
    }

    /// Get a function pointer and its signature by name
    pub fn get_entrypoint(&self, name: &CString) -> Option<(CodePtr, &Entrypoint)> {
        self.entrypoints.get(name).map(|(ptr, ep)| (*ptr, ep))
    }

    /// List all loaded entrypoint names
    pub fn entrypoint_names(&self) -> Vec<&CString> {
        self.entrypoints.keys().collect()
    }
}

////////////////////////////////////////////////////////////////////////////////
use libffi::middle::*;

/// Memory management for tensor buffers with shared ownership
#[derive(Debug, Clone)]
pub enum MlirBuffer<T> {
    Rust(Rc<*mut T>, usize, usize), // Shared Rust pointer + separate len/capacity
    Malloc(Rc<*mut T>),             // Shared malloc'd pointer
}

impl<T> Drop for MlirBuffer<T> {
    fn drop(&mut self) {
        match self {
            MlirBuffer::Rust(ptr_rc, len, capacity) => {
                // Only reconstruct and drop the Vec if this is the last reference
                if Rc::strong_count(ptr_rc) == 1 {
                    unsafe {
                        let _vec = Vec::from_raw_parts(**ptr_rc, *len, *capacity);
                        // Vec drop handles cleanup
                    }
                }
            }
            MlirBuffer::Malloc(ptr_rc) => {
                // Only free if this is the last reference
                if Rc::strong_count(ptr_rc) == 1 {
                    unsafe {
                        // Check if this is not actually a malloc'd pointer but statically allocated by LLVM.
                        // When the allocated memref field is 0xdeadbeef only the aligned field is valid.
                        // TODO: make this cleaner/more robust maybe by introducing a new enum variant for static allocation.
                        if **ptr_rc as usize != 0xdeadbeef {
                            libc::free(**ptr_rc as *mut c_void);
                        }
                    }
                }
            }
        }
    }
}

/// Helper struct to marshal bytes into the Memref format expected by LLVM
/// TODO: remove pub fields!
#[derive(Debug, Clone)]
pub struct MlirTensor<T> {
    pub allocated: MlirBuffer<T>,
    pub aligned: *mut T,
    pub offset: i64, // offset into ptr
    pub sizes: Vec<i64>,
    pub strides: Vec<i64>,
}

impl<T> std::fmt::Display for MlirTensor<T>
where
    T: std::fmt::Display + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MlirTensor<{}> [", std::any::type_name::<T>())?;

        // Print dimensions
        for (i, &size) in self.sizes.iter().enumerate() {
            if i > 0 {
                write!(f, "x")?;
            }
            write!(f, "{}", size)?;
        }
        write!(f, "] ")?;

        // Print some data if it's a reasonable size
        if self.sizes.iter().product::<i64>() <= 20 {
            unsafe {
                write!(f, "[")?;
                let total_elements = self.sizes.iter().product::<i64>();
                for i in 0..total_elements {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    let ptr = self.aligned.add((self.offset + i) as usize) as *const T;
                    write!(f, "{}", *ptr)?;
                }
                write!(f, "]")?;
            }
        } else {
            write!(f, "[{} elements]", self.sizes.iter().product::<i64>())?;
        }

        Ok(())
    }
}

impl<T> MlirTensor<T> {
    /// Construct a `libffi::middle::Arg` for each field as if of the below C struct.
    /// ```c
    /// #[repr(C)]
    /// struct Memref {
    ///     allocated: *mut f32,
    ///     aligned: *mut f32,
    ///     offset: i64,
    ///     sizes: [i64; N],
    ///     strides: [i64; N],
    /// }
    /// ```
    fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        let mut result = Vec::with_capacity(3 + 2 * self.sizes.len());

        // Get reference to the raw pointer from the buffer
        match &self.allocated {
            MlirBuffer::Rust(ptr_rc, _, _) => {
                result.push(Arg::new(ptr_rc.as_ref()));
            }
            MlirBuffer::Malloc(ptr_rc) => {
                result.push(Arg::new(ptr_rc.as_ref()));
            }
        }

        result.push(Arg::new(&self.aligned));
        result.push(Arg::new(&self.offset));
        for size in &self.sizes {
            result.push(Arg::new(size));
        }
        for stride in &self.strides {
            result.push(Arg::new(stride));
        }
        result
    }
}

/// Runtime representations of catgrad types which can be used with the LLVM Runtime
#[derive(Debug, Clone)]
pub enum MlirValue {
    MlirTensorF32(MlirTensor<f32>), // f32 tensors
    MlirTensorU32(MlirTensor<u32>), // u32 tensors
    I64(i64),
}

impl std::fmt::Display for MlirValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlirValue::MlirTensorF32(tensor) => write!(f, "{}", tensor),
            MlirValue::MlirTensorU32(tensor) => write!(f, "{}", tensor),
            MlirValue::I64(value) => write!(f, "I64({})", value),
        }
    }
}

impl MlirValue {
    pub fn to_type(&self) -> MlirType {
        match self {
            MlirValue::MlirTensorF32(tensor) => MlirType::Memref(tensor.sizes.len()),
            MlirValue::MlirTensorU32(tensor) => MlirType::Memref(tensor.sizes.len()),
            MlirValue::I64(_) => MlirType::I32,
        }
    }

    pub fn to_args<'a>(&'a self) -> Vec<Arg<'a>> {
        match self {
            MlirValue::MlirTensorF32(tensor) => tensor.to_args(),
            MlirValue::MlirTensorU32(tensor) => tensor.to_args(),
            MlirValue::I64(val) => vec![Arg::new(val)],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirType {
    Memref(usize), // ranked memref
    I32,
}

impl MlirType {
    // Get individual field types
    pub fn to_fields(&self) -> Vec<Type> {
        match self {
            MlirType::Memref(rank) => {
                let mut fields = vec![
                    Type::pointer(), // allocated
                    Type::pointer(), // aligned
                    Type::i64(),     // offset
                ];
                // Add rank number of sizes
                for _ in 0..*rank {
                    fields.push(Type::i64());
                }
                // Add rank number of strides
                for _ in 0..*rank {
                    fields.push(Type::i64());
                }
                fields
            }
            MlirType::I32 => vec![Type::i32()],
        }
    }

    // Get the struct type (wraps fields in a struct)
    pub fn to_type(&self) -> Type {
        let fields = self.to_fields();
        if fields.len() == 1 {
            fields[0].clone()
        } else {
            Type::structure(fields)
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility to call the function pointer with specified source/target types

// Unchecked call a raw ptr with MlirValue args and return MlirTensor results.
// TODO: return MlirValues!
fn call(
    ptr: CodePtr,
    source_values: Vec<MlirValue>,
    target_types: Vec<MlirType>,
) -> Vec<MlirTensor<f32>> {
    // Derive source types from the source values (flat map to get individual arg types)
    let source_types: Vec<Type> = source_values
        .iter()
        .flat_map(|v| v.to_type().to_fields())
        .collect();

    // Build args from source values (flatten all args)
    let args: Vec<Arg> = source_values.iter().flat_map(|v| v.to_args()).collect();

    // Build return type from target types
    let result_fields: Vec<Type> = target_types.iter().map(|vt| vt.to_type()).collect();
    let return_type = Type::structure(result_fields);

    unsafe {
        // Make Cif from source/return types
        let cif = Cif::new(source_types, return_type);

        // Calculate result buffer size dynamically
        let result_size = calculate_result_size(&target_types);
        let mut result = vec![0u8; result_size];

        // Call using raw ffi_call directly
        // TODO: sandbox this by running in its own process in case of segfault?
        raw::ffi_call(
            cif.as_raw_ptr(),
            Some(*ptr.as_safe_fun()),
            result.as_mut_ptr().cast::<c_void>(),
            args.as_ptr() as *mut *mut c_void,
        );

        // Collect input buffers to detect aliasing with outputs
        let input_buffers: std::collections::HashMap<*mut f32, MlirBuffer<f32>> = source_values
            .iter()
            .filter_map(|val| match val {
                MlirValue::MlirTensorF32(tensor) => match &tensor.allocated {
                    MlirBuffer::Rust(ptr_rc, _, _) => Some((**ptr_rc, tensor.allocated.clone())),
                    MlirBuffer::Malloc(ptr_rc) => Some((**ptr_rc, tensor.allocated.clone())),
                },
                _ => None,
            })
            .collect();

        // Parse results dynamically from the buffer
        parse_results_from_buffer(&result, &target_types, input_buffers)
    }
}

fn calculate_result_size(target_types: &[MlirType]) -> usize {
    // TODO: this probably makes some non-portable assumptions about the memref struct layout
    target_types
        .iter()
        .map(|vt| match vt {
            MlirType::Memref(rank) => {
                // Size = 2 pointers + 1 i64 offset + rank sizes + rank strides
                2 * std::mem::size_of::<*mut f32>()
                    + std::mem::size_of::<i64>()
                    + 2 * rank * std::mem::size_of::<i64>()
            }
            MlirType::I32 => std::mem::size_of::<i64>(),
        })
        .sum()
}

fn parse_results_from_buffer(
    buffer: &[u8],
    target_types: &[MlirType],
    input_buffers: std::collections::HashMap<*mut f32, MlirBuffer<f32>>,
) -> Vec<MlirTensor<f32>> {
    unsafe {
        let mut offset = 0;
        let mut results = Vec::new();
        // Start with input buffers, then add malloc'd ones as we encounter them
        let mut all_buffers: HashMap<*mut f32, MlirBuffer<f32>> = input_buffers;

        for target_type in target_types {
            let tensor = match target_type {
                MlirType::Memref(rank) => {
                    // Parse the memref struct fields manually using actual rank
                    let allocated = std::ptr::read(buffer.as_ptr().add(offset) as *const *mut f32);
                    offset += std::mem::size_of::<*mut f32>();

                    let aligned = std::ptr::read(buffer.as_ptr().add(offset) as *const *mut f32);
                    offset += std::mem::size_of::<*mut f32>();

                    let memref_offset = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                    offset += std::mem::size_of::<i64>();

                    // Read the rank number of sizes
                    let mut sizes = Vec::with_capacity(*rank);
                    for _ in 0..*rank {
                        let size = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                        sizes.push(size);
                        offset += std::mem::size_of::<i64>();
                    }

                    // Read the rank number of strides
                    let mut strides = Vec::with_capacity(*rank);
                    for _ in 0..*rank {
                        let stride = std::ptr::read(buffer.as_ptr().add(offset) as *const i64);
                        strides.push(stride);
                        offset += std::mem::size_of::<i64>();
                    }

                    // Get or create buffer for this pointer
                    let allocated_buffer =
                        if let Some(existing_buffer) = all_buffers.get(&allocated) {
                            // Reuse existing buffer (input or already seen malloc)
                            existing_buffer.clone()
                        } else {
                            // New malloc'd pointer
                            let new_buffer = MlirBuffer::Malloc(Rc::new(allocated));
                            all_buffers.insert(allocated, new_buffer.clone());
                            new_buffer
                        };

                    MlirTensor {
                        allocated: allocated_buffer,
                        aligned,
                        offset: memref_offset,
                        sizes,
                        strides,
                    }
                }
                MlirType::I32 => {
                    panic!("I64 not handled yet")
                }
            };
            results.push(tensor);
        }

        results
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unsafe util functions

// TODO: remove this when `Vec::into_raw_parts` is
// [merged](https://github.com/rust-lang/rust/issues/65816)
pub fn into_raw_parts<T>(vec: Vec<T>) -> (*mut T, usize, usize) {
    use std::mem::ManuallyDrop;
    let mut me = ManuallyDrop::new(vec);
    (me.as_mut_ptr(), me.len(), me.capacity())
}
