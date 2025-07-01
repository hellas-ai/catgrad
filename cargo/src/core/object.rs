use std::ops::Add;

/// A type alias for natural numbers
pub type Nat = usize;

////////////////////////////////////////////////////////////////////////////////
// Generating Objects

/// Dtypes supported by N-dimensional arrays.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Dtype {
    F16,
    F32,
    I32,
}

/// A rank-N shape is a length-N array of natural numbers
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(pub Vec<Nat>);

impl Shape {
    pub fn size(&self) -> Nat {
        self.0.iter().product()
    }

    pub fn empty() -> Self {
        Shape(vec![])
    }

    pub fn concatenate(&self, other: &Self) -> Self {
        Shape(self.0.iter().chain(other.0.iter()).cloned().collect())
    }

    pub fn for_each_index<F>(&self, mut f: F)
    where
        F: FnMut(usize, &[usize]),
    {
        let mut indices = vec![0; self.0.len()];
        let total_elements = self.size();

        for i in 0..total_elements {
            f(i, &indices);

            // Increment indices with carry
            let mut d = indices.len() - 1;
            loop {
                indices[d] += 1;
                if indices[d] < self.0[d] || d == 0 {
                    break;
                }
                indices[d] = 0;
                d -= 1;
            }
        }
    }
}

/// The type of an NdArray is defined by shape and dtype.
/// These are the objects of the category.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NdArrayType {
    pub shape: Shape,
    pub dtype: Dtype,
}

impl NdArrayType {
    pub fn new(shape: Shape, dtype: Dtype) -> Self {
        NdArrayType { shape, dtype }
    }

    pub fn size(&self) -> Nat {
        self.shape.size()
    }

    pub fn concatenate(&self, other: &Self) -> Option<Self> {
        if self.dtype != other.dtype {
            return None;
        }

        Some(NdArrayType::new(
            self.shape.concatenate(&other.shape),
            self.dtype,
        ))
    }
}

/// Primitive types in catgrad core.
///
/// These form the generating objects of the category of "core" array programs.
pub type PrimitiveType = NdArrayType;

/// The type of a catgrad core term
///
/// "Functions" in catgrad naturally have multiple inputs and outputs, so *types* of "functions"
/// are always *lists* of types.
///
/// These are the *objects* of the category of "core" array programs.
pub type Type = Vec<PrimitiveType>;

/// The input and output type of a function (i.e., an arrow in the category).
pub type Interface = (Type, Type);

////////////////////////////////////////////////////////////////////////////////
// Convenience instances

// Add Nat to Shape (reference version)
impl<'a> Add<&'a Nat> for &'a Shape {
    type Output = Shape;

    fn add(self, other: &'a Nat) -> Self::Output {
        let mut result = self.0.clone();
        result.push(*other);
        Shape(result)
    }
}

// Add Nat to Shape (owned version)
impl<'a> Add<&'a Nat> for Shape {
    type Output = Shape;

    fn add(self, other: &'a Nat) -> Self::Output {
        let mut result = self.0;
        result.push(*other);
        Shape(result)
    }
}

// Add Shape to Shape (reference version)
impl<'a> Add<&'a Shape> for &'a Shape {
    type Output = Shape;

    fn add(self, other: &'a Shape) -> Self::Output {
        self.concatenate(other)
    }
}

// Add Shape to Shape (owned version)
impl<'a> Add<&'a Shape> for Shape {
    type Output = Shape;

    fn add(self, other: &'a Shape) -> Self::Output {
        Shape(self.0.iter().chain(other.0.iter()).cloned().collect())
    }
}

// Add Shape to NdArrayType (reference version)
impl<'a> Add<&'a NdArrayType> for &'a Shape {
    type Output = NdArrayType;

    fn add(self, other: &'a NdArrayType) -> Self::Output {
        NdArrayType::new(self.concatenate(&other.shape), other.dtype)
    }
}

// Add Shape to NdArrayType (owned version)
impl<'a> Add<&'a NdArrayType> for Shape {
    type Output = NdArrayType;

    fn add(self, other: &'a NdArrayType) -> Self::Output {
        NdArrayType::new(
            Shape(self.0.iter().chain(other.shape.0.iter()).cloned().collect()),
            other.dtype,
        )
    }
}

// Add NdArrayType to NdArrayType (reference version)
impl<'a> Add<&'a NdArrayType> for &'a NdArrayType {
    type Output = NdArrayType;

    fn add(self, other: &'a NdArrayType) -> Self::Output {
        if let Some(result) = self.concatenate(other) {
            result
        } else {
            panic!("Cannot add NdArrayTypes with different dtypes")
        }
    }
}
