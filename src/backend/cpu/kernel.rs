//! Array kernels for CPU
use super::ndarray::*;
use crate::core::object::Shape;
use core::fmt::Debug;
use gemm::{gemm, Parallelism};

fn matmul<T: Numeric + 'static>(
    a: &NdArraySlice<T>,
    b: &NdArraySlice<T>,
    c: &mut NdArrayMutSlice<T>,
) {
    // Extract dimensions from input slices
    // For matrix multiplication: (m×k) × (k×n) → (m×n)
    if a.shape.0.len() != 2 || b.shape.0.len() != 2 || c.shape.0.len() != 2 {
        panic!("Matrix multiplication requires 2D arrays");
    }

    let m = a.shape.0[0];
    let k = a.shape.0[1];
    let n = b.shape.0[1];

    // Check compatibility
    if b.shape.0[0] != k {
        panic!("Incompatible dimensions for matrix multiplication");
    }
    if c.shape.0[0] != m || c.shape.0[1] != n {
        panic!("Output matrix has incorrect dimensions");
    }

    // Call gemm with the correct parameters
    // gemm(m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
    // where:
    // - m, k, n are dimensions (m×k matrix A, k×n matrix B, m×n matrix C)
    // - alpha, beta are scaling factors for A*B and C

    // For row-major matrices:
    // - Row stride = distance between rows (number of elements in a row)
    // - Column stride = 1 (adjacent elements in memory)

    unsafe {
        gemm(
            m,                   // m: rows in matrices A and C
            n,                   // n: cols in matrices B and C
            k,                   // k: cols in A, rows in B
            c.data.as_mut_ptr(), // c: pointer to result matrix C
            c.strides[1],        // column stride for C
            c.strides[0],        // row stride for C
            false,               // read C
            a.data.as_ptr(),     // a: pointer to first matrix A
            a.strides[1],        // column stride for A
            a.strides[0],        // row stride for A
            b.data.as_ptr(),     // b: pointer to second matrix B
            b.strides[1],        // column stride for B
            b.strides[0],        // row stride for B
            T::zero(),           // alpha scaling factor for A*B
            T::one(),            // beta scaling factor for C
            false,               // conj C
            false,               // conj A
            false,               // conj B
            Parallelism::None,
        )
    }
}

/// Batch matrix multiply (compose) `f : N×...×A×B` with `g : N×...×B×C`, writing the result to `h :
/// N×...×A×C`. Works with arrays of dimension 2 or greater.
pub fn batch_matmul<T: Numeric + 'static>(f: &NdArray<T>, g: &NdArray<T>, h: &mut NdArray<T>) {
    // Assert that shapes of f, g, h have at least 2 dimensions
    if f.shape.0.len() < 2 || g.shape.0.len() < 2 || h.shape.0.len() < 2 {
        panic!("Matrix multiplication requires at least 2D arrays");
    }

    // If arrays are exactly 2D, perform a single matrix multiplication
    if f.shape.0.len() == 2 && g.shape.0.len() == 2 && h.shape.0.len() == 2 {
        let a = f.shape.0[0];
        let b = f.shape.0[1];
        let c = g.shape.0[1];

        // Check compatibility for 2D case
        if g.shape.0[0] != b {
            panic!("Incompatible dimensions for matrix multiplication");
        }
        if h.shape.0[0] != a || h.shape.0[1] != c {
            panic!("Output matrix has incorrect dimensions");
        }

        let f_slice = f.slice(&[]);
        let g_slice = g.slice(&[]);
        let mut h_slice = h.slice_mut(&[]);

        matmul(&f_slice, &g_slice, &mut h_slice);
        return;
    }

    // For arrays with dimension > 2, treat all but the last 2 dimensions as batch dimensions
    // All arrays must have the same number of dimensions
    if f.shape.0.len() != g.shape.0.len() || f.shape.0.len() != h.shape.0.len() {
        panic!(
            "All arrays must have the same number of dimensions for batch matrix multiplication"
        );
    }

    let ndims = f.shape.0.len();
    let batch_dims = ndims - 2;

    // For ND arrays, the last two dimensions are the matrix dimensions
    let a = f.shape.0[ndims - 2];
    let b = f.shape.0[ndims - 1];
    let c = g.shape.0[ndims - 1];

    // Check compatibility for the matrix dimensions
    if g.shape.0[ndims - 2] != b {
        panic!("Incompatible dimensions for matrix multiplication");
    }
    if h.shape.0[ndims - 2] != a || h.shape.0[ndims - 1] != c {
        panic!("Output matrix has incorrect dimensions");
    }

    // Check that batch dimensions are compatible
    for i in 0..batch_dims {
        if f.shape.0[i] != g.shape.0[i] || f.shape.0[i] != h.shape.0[i] {
            panic!("Batch dimensions must match for all arrays");
        }
    }

    // If there's only one batch dimension, we can use the simple loop
    if batch_dims == 1 {
        let batch_size = f.shape.0[0];

        // Loop over batch dimension calling matmul
        for i in 0..batch_size {
            let f_slice = f.slice(&[i]);
            let g_slice = g.slice(&[i]);
            let mut h_slice = h.slice_mut(&[i]);

            matmul(&f_slice, &g_slice, &mut h_slice);
        }
    } else {
        // For multiple batch dimensions, we need to iterate through all combinations
        // Calculate total number of batches
        let mut total_batches = 1;
        for i in 0..batch_dims {
            total_batches *= f.shape.0[i];
        }

        // For each batch, compute the indices for that batch
        for batch_idx in 0..total_batches {
            let mut indices = Vec::with_capacity(batch_dims);
            let mut remaining = batch_idx;

            // Convert flat batch index to multidimensional indices
            for i in (0..batch_dims).rev() {
                let dim_size = f.shape.0[i];
                indices.insert(0, remaining % dim_size);
                remaining /= dim_size;
            }

            let f_slice = f.slice(&indices);
            let g_slice = g.slice(&indices);
            let mut h_slice = h.slice_mut(&indices);

            matmul(&f_slice, &g_slice, &mut h_slice);
        }
    }
}

pub trait Numeric:
    num_traits::Num + num_traits::Bounded + std::ops::Neg<Output = Self> + Copy + Debug
{
}
impl<T> Numeric for T where
    T: num_traits::Num + num_traits::Bounded + std::ops::Neg<Output = Self> + Copy + Debug
{
}

pub trait BinOp<T: Numeric> {
    fn apply(&self, a: &NdArray<T>, b: &NdArray<T>, c: &mut NdArray<T>);
}

pub struct AddOp;
impl<T: Numeric> BinOp<T> for AddOp {
    fn apply(&self, a: &NdArray<T>, b: &NdArray<T>, c: &mut NdArray<T>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i] + b.data[i];
        }
    }
}

pub struct SubOp;
impl<T: Numeric> BinOp<T> for SubOp {
    fn apply(&self, a: &NdArray<T>, b: &NdArray<T>, c: &mut NdArray<T>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i] - b.data[i];
        }
    }
}

pub struct MulOp;
impl<T: Numeric> BinOp<T> for MulOp {
    fn apply(&self, a: &NdArray<T>, b: &NdArray<T>, c: &mut NdArray<T>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i] * b.data[i];
        }
    }
}

pub struct DivOp;
impl<T: Numeric> BinOp<T> for DivOp {
    fn apply(&self, a: &NdArray<T>, b: &NdArray<T>, c: &mut NdArray<T>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i] / b.data[i];
        }
    }
}

pub struct PowOp;

// TODO: Maybe this can be done with less duplication by using num_traits::Pow?
impl BinOp<i32> for PowOp {
    fn apply(&self, a: &NdArray<i32>, b: &NdArray<i32>, c: &mut NdArray<i32>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i].pow(b.data[i] as u32);
        }
    }
}

use num_traits::Float;
impl BinOp<half::f16> for PowOp {
    fn apply(&self, a: &NdArray<half::f16>, b: &NdArray<half::f16>, c: &mut NdArray<half::f16>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i].powf(b.data[i]);
        }
    }
}

impl BinOp<f32> for PowOp {
    fn apply(&self, a: &NdArray<f32>, b: &NdArray<f32>, c: &mut NdArray<f32>) {
        for i in 0..a.data.len() {
            c.data[i] = a.data[i].powf(b.data[i]);
        }
    }
}

pub struct MatMulOp;
impl<T: Numeric + 'static> BinOp<T> for MatMulOp {
    fn apply(&self, a: &NdArray<T>, b: &NdArray<T>, c: &mut NdArray<T>) {
        batch_matmul(a, b, c);
    }
}

pub trait UnaryOp<T: Numeric> {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>);
}

pub struct NegOp;
impl<T: Numeric> UnaryOp<T> for NegOp {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>) {
        for i in 0..a.data.len() {
            b.data[i] = -a.data[i];
        }
    }
}

pub struct ReshapeOp;

impl<T: Numeric> UnaryOp<T> for ReshapeOp {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>) {
        assert_eq!(
            a.shape.size(),
            b.shape.size(),
            "ReshapeOp: input shape must be compatible with target shape"
        );
        b.data = a.data.clone(); //TODO: reuse vec instead of copy
    }
}

// Max and Sum could have a more generic FoldOp trait but for now
// they work well as UnaryOps.

pub struct MaxOp;

impl<T: Numeric + PartialOrd> UnaryOp<T> for MaxOp {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>) {
        let last_dim = a.shape.0[a.shape.0.len() - 1];
        for (i, chunk) in a.data.chunks(last_dim).enumerate() {
            b.data[i] = chunk
                .iter()
                .fold(T::min_value(), |acc, &x| if acc > x { acc } else { x })
        }
    }
}

pub struct SumOp;

impl<T: Numeric + PartialOrd + std::iter::Sum> UnaryOp<T> for SumOp {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>) {
        let last_dim = a.shape.0[a.shape.0.len() - 1];
        for (i, chunk) in a.data.chunks(last_dim).enumerate() {
            b.data[i] = chunk.iter().copied().sum();
        }
    }
}

/// Broadcast input across a new shape.
/// Ex: Input of shape [4, 5] broadcasted with shape n = [2, 3]
/// will result in output of shape [2,3,4,5] where the input is repeated 2x3 times.
pub struct BroadcastOp {
    pub n: Shape,
}

impl<T: Numeric> UnaryOp<T> for BroadcastOp {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>) {
        //TODO: Add proper compatibility checks
        // assert_eq!(
        //     a.shape.0.len(),
        //     self.n.0.len(),
        //     // b.shape.0.len(),
        //     "BroadcastOp: input and output shapes must have compatible dimensions"
        // );
        // assert_eq!(
        //     a.shape.size() * self.n.size(),
        //     b.shape.size(),
        //     "BroadcastOp: output size must be multiple of input size"
        // );
        b.data = a.data.clone(); //TODO: reuse vec instead of copy

        // Set strides to 0 for the broadcasting dimensions
        b.strides[..self.n.0.len()].fill(0);
    }
}

pub struct TransposeOp {
    pub dim0: usize,
    pub dim1: usize,
}

impl<T: Numeric> UnaryOp<T> for TransposeOp {
    fn apply(&self, a: &NdArray<T>, b: &mut NdArray<T>) {
        // Validate dimensions
        assert!(
            self.dim0 < a.shape.0.len() && self.dim1 < a.shape.0.len(),
            "TransposeOp: dimensions must be valid for input shape"
        );

        // Create new shape with swapped dimensions
        let mut new_shape = a.shape.0.clone();
        new_shape.swap(self.dim0, self.dim1);

        b.shape = Shape(new_shape);

        // Create new strides with swapped dimensions
        let mut new_strides = a.strides.clone();
        new_strides.swap(self.dim0, self.dim1);
        b.strides = new_strides;

        // Copy the data - we won't actually move anything in memory,
        // just change how we index into it
        b.data = a.data.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32() {
        // Create matrices for multiplication
        // A: 2x3 matrix
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(vec![2, 3]));
        // B: 3x2 matrix
        let b = NdArray::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], Shape(vec![3, 2]));

        // C: 2x2 matrix for result
        let mut c = NdArray::new(vec![0.0; 4], Shape(vec![2, 2]));

        // Expected result:
        // [1 2 3] × [7  8]  = [58  64]
        // [4 5 6]   [9 10]    [139 154]
        //          [11 12]

        let a_slice = a.slice(&[]);
        let b_slice = b.slice(&[]);
        let mut c_slice = c.slice_mut(&[]);

        // Call matmul
        matmul(&a_slice, &b_slice, &mut c_slice);

        // Check the result
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_batch_matmul_f32() {
        // Create 3D arrays for batch matrix multiplication
        // 2 batches of 2x3 matrices
        let f = NdArray::new(
            vec![
                // Batch 0: 2x3 matrix
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1: 2x3 matrix
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            Shape(vec![2, 2, 3]),
        );

        // 2 batches of 3x2 matrices
        let g = NdArray::new(
            vec![
                // Batch 0: 3x2 matrix
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1: 3x2 matrix
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            Shape(vec![2, 3, 2]),
        );

        // Output: 2 batches of 2x2 matrices
        let mut h = NdArray::new(
            vec![0.0; 8], // 2 batches * 2 * 2 = 8 elements
            Shape(vec![2, 2, 2]),
        );

        // Expected results:
        // Batch 0: [1 2 3] × [1 2]  = [22 28]
        //          [4 5 6]   [3 4]    [49 64]
        //                    [5 6]
        //
        // Batch 1: [7  8  9] × [7  8]   = [220 244]
        //          [10 11 12]   [9  10]    [301 334]
        //                       [11 12]

        // Call batch_matmul
        batch_matmul(&f, &g, &mut h);

        // Check the results for both batches
        assert_eq!(
            h.data,
            vec![
                // Batch 0
                22.0, 28.0, 49.0, 64.0, // Batch 1
                220.0, 244.0, 301.0, 334.0
            ]
        );
    }
}
