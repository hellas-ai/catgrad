use super::kernel::Numeric;
use crate::core::object::*;
use half::f16;
use num_traits::Zero;
use std::cell::RefCell;
use std::rc::Rc;

/// N-dimensional arrays of elements T with a given shape.
#[derive(PartialEq, Debug, Clone)]
pub struct NdArray<T> {
    pub data: Rc<RefCell<Vec<T>>>, // raw data of the array
    pub shape: Shape,              // shape information (erasable?)
    pub strides: Vec<isize>,       // strides for each dimension
    pub offset: usize,             // offset for the array
}

fn compute_strides(shape: &Shape) -> Vec<isize> {
    let mut strides: Vec<isize> = vec![1];
    for dim in shape.0.iter().skip(1).rev() {
        strides.push(strides.last().unwrap() * (*dim as isize));
    }
    strides.reverse();
    strides
}

impl<T: Numeric> NdArray<T> {
    pub fn new(data: Vec<T>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.size(),
            "Data length must match shape size"
        );
        Self {
            data: Rc::new(RefCell::new(data)),
            strides: compute_strides(&shape),
            shape,
            offset: 0,
        }
    }

    /// Create a new empty NdArray with the given shape.
    pub fn new_empty(shape: Shape) -> Self {
        Self {
            data: Rc::new(RefCell::new(vec![])),
            strides: compute_strides(&shape),
            shape,
            offset: 0,
        }
    }

    /// Allocate memory for the NdArray's data field.
    pub fn allocate(&mut self) {
        self.data.borrow_mut().resize(self.shape.size(), T::zero());
    }

    /// Deallocate memory for the NdArray's data field.
    pub fn deallocate(&mut self) {
        self.data = Rc::new(RefCell::new(vec![]));
    }

    pub fn is_empty(&self) -> bool {
        self.shape.size() == 0
    }

    pub fn len(&self) -> usize {
        self.shape.size()
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Compute slice indices for fixed first n dimension.
    /// Returns (start_index, slice_shape)
    fn calculate_slice_info(&self, indices: &[usize]) -> (usize, Shape, Vec<isize>) {
        if indices.len() > self.shape.0.len() {
            panic!("Too many indices provided");
        }

        // Check that indices are within bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.0[i] {
                panic!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, i, self.shape.0[i]
                );
            }
        }

        // If no indices provided, use the whole array
        if indices.is_empty() {
            return (0, self.shape.clone(), self.strides.clone());
        }

        let mut start_index = 0;

        // Start with the innermost dimension
        for i in 0..self.shape.0.len() {
            if i >= indices.len() {
                // This dimension is part of the slice, we start at index 0
                // No need to add anything to start_index
            } else {
                // This dimension is fixed by the provided indices
                start_index += indices[i] * (self.strides[i] as usize);
            }
        }

        // Compute the shape of the resulting slice
        let slice_shape = Shape(self.shape.0[indices.len()..].to_vec());
        let slice_strides = self.strides[indices.len()..].to_vec();

        (start_index, slice_shape, slice_strides)
    }

    pub fn slice(&self, indices: &[usize]) -> NdArray<T> {
        let (start_index, slice_shape, slice_strides) = self.calculate_slice_info(indices);

        NdArray {
            data: Rc::clone(&self.data),
            shape: slice_shape,
            strides: slice_strides,
            offset: self.offset + start_index,
        }
    }

    /// Calculates the flat index in the data vector for a given multi-dimensional index.
    /// Panics if the index is out of bounds.
    fn calculate_flat_index(&self, index: &[usize]) -> usize {
        if index.len() != self.shape.0.len() {
            panic!(
                "Index dimension mismatch: expected {}, got {}",
                self.shape.0.len(),
                index.len()
            );
        }

        let mut flat_index = self.offset;
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape.0[i] {
                panic!(
                    "Index {} out of bounds for dimension {} with size {}",
                    idx, i, self.shape.0[i]
                );
            }
            flat_index += idx * (self.strides[i] as usize);
        }
        flat_index
    }
}

impl<T: Numeric> NdArray<T> {
    /// Copy data from another NdArray into this one.
    /// Panics if the shapes don't match.
    pub fn copy_from(&mut self, other: &NdArray<T>) {
        // Check that shapes match
        if self.shape != other.shape {
            panic!(
                "Shape mismatch in copy_from: destination shape {:?} != source shape {:?}",
                self.shape, other.shape
            );
        }

        if self.strides == other.strides {
            let mut data = self.data.borrow_mut();
            data.clone_from_slice(&other.data.borrow());
            return;
        }

        // For arrays with different strides, we need to copy element by element
        // using the proper indexing for each array

        other.shape.for_each_index(|_, indices| {
            self.set(indices, other.get(indices));
        });
    }

    pub fn get(&self, index: &[usize]) -> T {
        let flat_index = self.calculate_flat_index(index);
        self.data.borrow()[flat_index]
    }

    pub fn set(&mut self, index: &[usize], value: T) {
        let flat_index = self.calculate_flat_index(index);
        self.data.borrow_mut()[flat_index] = value;
    }
}

impl<T: Numeric + Zero> NdArray<T> {
    pub fn from_shape(shape: Shape) -> Self {
        // TODO: don't really need to initialize to zero; is there a better way here? bytemuck?
        log::debug!("New NdArray {:?} {:?}", shape, shape.size());
        NdArray::new(vec![T::zero(); shape.size()], shape)
    }

    pub fn fill(&mut self, value: T) {
        let mut data = self.data.borrow_mut();
        for i in 0..data.len() {
            data[i] = value;
        }
    }
}

impl<T: std::fmt::Display + Numeric> NdArray<T> {
    /// Pretty print the array in PyTorch-like format
    pub fn pretty_print(&self) -> String {
        self.pretty_print_with_options(4, 4)
    }

    pub fn pretty_print_with_options(&self, edge_items: usize, precision: usize) -> String {
        self.format_recursive(&[], 0, edge_items, precision)
    }

    fn format_recursive(
        &self,
        indices: &[usize],
        depth: usize,
        edge_items: usize,
        precision: usize,
    ) -> String {
        if indices.len() == self.shape.0.len() {
            return format!("{:.*}", precision, self.get(indices));
        }

        let indent = " ".repeat(depth);
        // Newline after closing ]
        let comma = if indices.len() < self.shape.0.len() - 1 {
            ",\n"
        } else {
            ","
        };

        let dim_size = self.shape.0[indices.len()];
        let mut result = String::new();

        result.push('\n');
        result.push_str(&indent);
        result.push('[');

        for i in 0..dim_size {
            if i == edge_items && dim_size > edge_items * 2 {
                result.push_str(comma);
                result.push_str(" ...");
            }
            if i >= edge_items && i + edge_items < dim_size {
                continue;
            }
            if i > 0 {
                result.push_str(comma);
                result.push(' ');
            }

            let mut new_indices = indices.to_vec();
            new_indices.push(i);
            result.push_str(&self.format_recursive(&new_indices, depth + 1, edge_items, precision));
        }
        result.push(']');
        result
    }
}

impl<T: std::fmt::Display + Numeric> std::fmt::Display for NdArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pretty_print())
    }
}

/// A disjoint union of typed arrays
/// intuition: the "values" assigned to each node in a [`Term`].
#[derive(PartialEq, Debug, Clone)]
pub enum TaggedNdArray {
    F16(NdArray<f16>),
    F32(NdArray<f32>),
    I32(NdArray<i32>),
}

impl TaggedNdArray {
    pub fn is_empty(&self) -> bool {
        match self {
            TaggedNdArray::F16(vec) => vec.is_empty(),
            TaggedNdArray::F32(vec) => vec.is_empty(),
            TaggedNdArray::I32(vec) => vec.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TaggedNdArray::F16(vec) => vec.len(),
            TaggedNdArray::F32(vec) => vec.len(),
            TaggedNdArray::I32(vec) => vec.len(),
        }
    }

    pub fn data_len(&self) -> usize {
        match self {
            TaggedNdArray::F16(a) => a.data.borrow().len(),
            TaggedNdArray::F32(a) => a.data.borrow().len(),
            TaggedNdArray::I32(a) => a.data.borrow().len(),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            TaggedNdArray::F16(a) => a.is_contiguous(),
            TaggedNdArray::F32(a) => a.is_contiguous(),
            TaggedNdArray::I32(a) => a.is_contiguous(),
        }
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        match self {
            TaggedNdArray::F16(arr) => arr.get(index).to_f32(),
            TaggedNdArray::F32(arr) => arr.get(index),
            TaggedNdArray::I32(arr) => arr.get(index) as f32,
        }
    }

    pub fn from_type(t: &NdArrayType) -> Self {
        match t.dtype {
            Dtype::F16 => TaggedNdArray::F16(NdArray::from_shape(t.shape.clone())),
            Dtype::F32 => TaggedNdArray::F32(NdArray::from_shape(t.shape.clone())),
            Dtype::I32 => TaggedNdArray::I32(NdArray::from_shape(t.shape.clone())),
        }
    }

    pub fn from_type_empty(t: &NdArrayType) -> Self {
        match t.dtype {
            Dtype::F16 => TaggedNdArray::F16(NdArray::new_empty(t.shape.clone())),
            Dtype::F32 => TaggedNdArray::F32(NdArray::new_empty(t.shape.clone())),
            Dtype::I32 => TaggedNdArray::I32(NdArray::new_empty(t.shape.clone())),
        }
    }

    pub fn data(&self) -> Vec<f32> {
        match self {
            TaggedNdArray::F16(vec) => vec.data.borrow().iter().map(|&x| x.into()).collect(),
            TaggedNdArray::F32(vec) => vec.data.borrow().to_vec(),
            TaggedNdArray::I32(vec) => vec.data.borrow().iter().map(|&x| x as f32).collect(),
        }
    }

    pub fn shape(&self) -> Shape {
        match self {
            TaggedNdArray::F16(vec) => vec.shape.clone(),
            TaggedNdArray::F32(vec) => vec.shape.clone(),
            TaggedNdArray::I32(vec) => vec.shape.clone(),
        }
    }

    pub fn strides(&self) -> Vec<isize> {
        match self {
            TaggedNdArray::F16(vec) => vec.strides.clone(),
            TaggedNdArray::F32(vec) => vec.strides.clone(),
            TaggedNdArray::I32(vec) => vec.strides.clone(),
        }
    }
    pub fn allocate(&mut self) {
        match self {
            TaggedNdArray::F16(a) => a.allocate(),
            TaggedNdArray::F32(a) => a.allocate(),
            TaggedNdArray::I32(a) => a.allocate(),
        }
    }

    pub fn deallocate(&mut self) {
        match self {
            TaggedNdArray::F16(a) => a.deallocate(),
            TaggedNdArray::F32(a) => a.deallocate(),
            TaggedNdArray::I32(a) => a.deallocate(),
        }
    }

    /// Check if data is close to another slice within a tolerance
    pub fn allclose(&self, other: &[f32], rtol: f32, atol: f32) -> bool {
        if self.len() != other.len() {
            return false; // Vectors must have the same length
        }

        std::iter::zip(self.data().as_slice(), other)
            .all(|(x, &y)| (x - y).abs() <= atol + rtol * y.abs())
    }

    /// Approximate data to a given number of decimal places
    pub fn approx(&self, digits: i32) -> Vec<f32> {
        let b = 10f32.powi(digits);
        self.data().iter().map(|x| (x * b).round() / b).collect()
    }
}

impl TaggedNdArray {
    /// Pretty print the tagged array in PyTorch-like format
    pub fn pretty_print(&self) -> String {
        match self {
            TaggedNdArray::F16(arr) => arr.pretty_print(),
            TaggedNdArray::F32(arr) => arr.pretty_print(),
            TaggedNdArray::I32(arr) => arr.pretty_print(),
        }
    }

    /// Pretty print with custom options
    pub fn pretty_print_with_options(&self, edge_items: usize, precision: usize) -> String {
        match self {
            TaggedNdArray::F16(arr) => arr.pretty_print_with_options(edge_items, precision),
            TaggedNdArray::F32(arr) => arr.pretty_print_with_options(edge_items, precision),
            TaggedNdArray::I32(arr) => arr.pretty_print_with_options(edge_items, precision),
        }
    }
}

impl std::fmt::Display for TaggedNdArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pretty_print())
    }
}

impl From<NdArray<f16>> for TaggedNdArray {
    fn from(value: NdArray<f16>) -> Self {
        TaggedNdArray::F16(value)
    }
}
impl From<NdArray<f32>> for TaggedNdArray {
    fn from(value: NdArray<f32>) -> Self {
        TaggedNdArray::F32(value)
    }
}

impl From<NdArray<i32>> for TaggedNdArray {
    fn from(value: NdArray<i32>) -> Self {
        TaggedNdArray::I32(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_log::test;

    #[test]
    fn test_strides() {
        let shape = Shape(vec![2, 3, 4]);
        let strides = compute_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);

        let shape = Shape(vec![6]);
        let strides = compute_strides(&shape);
        assert_eq!(strides, vec![1]);
    }

    #[test]
    fn test_slice() {
        // Create a 2x3x4 array filled with zeros
        let v = (0..24).map(|i| i as f32).collect();
        let array = NdArray::new(v, Shape(vec![2, 3, 4]));

        // Calculate the expected linear indices for a 2x3x4 array with row-major ordering
        // For [0,1,...] we should get indices [4,5,6,7]

        // Get a slice at [0, 1, ...]
        let mut slice = array.slice(&[0, 1]);

        // Check the shape of the slice
        assert_eq!(slice.shape, Shape(vec![4]));

        // Check the data in the slice
        assert_eq!(slice.get(&[0]), 4.0);
        assert_eq!(slice.get(&[1]), 5.0);
        assert_eq!(slice.get(&[2]), 6.0);
        assert_eq!(slice.get(&[3]), 7.0);

        // Modify the slice and check that it affects the original array
        slice.set(&[2], 99.0);
        assert_eq!(array.get(&[0, 1, 2]), 99.0);

        // Test with different indices [1, 2, ...]
        let slice2 = array.slice(&[1, 2]);
        assert_eq!(slice2.shape, Shape(vec![4]));
        assert_eq!(slice2.get(&[0]), 20.0);
        assert_eq!(slice2.get(&[1]), 21.0);
        assert_eq!(slice2.get(&[2]), 22.0);
        assert_eq!(slice2.get(&[3]), 23.0);

        // Test slicing with no indices (should return the whole array)
        let slice_all = array.slice(&[]);
        assert_eq!(slice_all.shape, Shape(vec![2, 3, 4]));
        assert_eq!(slice_all.len(), 24);

        // Test slicing a 4D array
        let v = (0..120).map(|i| i as f32).collect();
        let array4d = NdArray::new(v, Shape(vec![2, 3, 4, 5]));

        let slice4d = array4d.slice(&[0, 0]);
        assert_eq!(slice4d.shape, Shape(vec![4, 5]));
        assert_eq!(slice4d.len(), 20);

        // Check the data in the 2D slice
        for i in 0..4 {
            for j in 0..5 {
                assert_eq!(slice4d.get(&[i, j]), (i * 5 + j) as f32);
            }
        }
    }

    #[test]
    fn test_indexing() {
        let v = (0..24).map(|i| i as f32).collect();
        let mut array = NdArray::new(v, Shape(vec![2, 3, 4]));

        // Test reading
        assert_eq!(array.get(&[0, 0, 0]), 0.0);
        assert_eq!(array.get(&[0, 1, 2]), 6.0); // 0*12 + 1*4 + 2*1 = 6
        assert_eq!(array.get(&[1, 0, 0]), 12.0); // 1*12 + 0*4 + 0*1 = 12
        assert_eq!(array.get(&[1, 2, 3]), 23.0); // 1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23

        // Test writing
        array.set(&[0, 1, 2], 99.0);
        assert_eq!(array.data.borrow()[6], 99.0);
        assert_eq!(array.get(&[0, 1, 2]), 99.0);

        array.set(&[1, 2, 3], -1.0);
        assert_eq!(array.data.borrow()[23], -1.0);
        assert_eq!(array.get(&[1, 2, 3]), -1.0);
    }

    #[test]
    #[should_panic]
    fn test_indexing_out_of_bounds_dim() {
        let array = NdArray::new(vec![0.0; 24], Shape(vec![2, 3, 4]));
        let _ = array.get(&[0, 0, 4]); // Index 4 is out of bounds for the last dimension
    }

    #[test]
    #[should_panic]
    fn test_indexing_out_of_bounds_mut() {
        let mut array = NdArray::new(vec![0.0; 24], Shape(vec![2, 3, 4]));
        array.set(&[2, 0, 0], 5.0); // Index 2 is out of bounds for the first dimension
    }

    #[test]
    #[should_panic]
    fn test_indexing_wrong_ndim() {
        let array = NdArray::new(vec![0.0; 24], Shape(vec![2, 3, 4]));
        let _ = array.get(&[0, 0]); // Incorrect number of dimensions
    }

    #[test]
    #[should_panic]
    fn test_indexing_wrong_ndim_mut() {
        let mut array = NdArray::new(vec![0.0; 24], Shape(vec![2, 3, 4]));
        array.set(&[1, 1], 5.0); // Incorrect number of dimensions
    }
}
