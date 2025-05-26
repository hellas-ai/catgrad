use super::ndarray::*;
use crate::backend::cpu::kernel;
use crate::core::{Operation, StrictTerm, Term, Var};
use half::f16;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use Operation::*;
use TaggedNdArray::*;

use log;

// TODO: this convenience method should live in open_hypergraphs
use open_hypergraphs::strict::layer::*;
use open_hypergraphs::strict::*;
fn layered_operations(f: &StrictTerm) -> Vec<Vec<usize>> {
    let (order, _unvisited) = layer(f);
    // TODO: check not unvisited.any(|x| x == 1).
    let c = converse(&IndexedCoproduct::elements(order));
    c.into_iter().map(|x| x.table.0).collect()
}

/// For each node in the term `f`, allocate an `NdArray` whose size corresponds to the given
/// shape.
/// NOTE: some operations (like broadcasting) don't actually need to allocate a new array; they are
/// really just "renamings" of the same underlying data. So this allocation strategy should be
/// improved in future.
fn allocate(f: &StrictTerm) -> Vec<TaggedNdArray> {
    // Loop over all nodes in the term, allocate an array according to the size/dtype of its
    // labeled NdArrayType.
    let mut result = Vec::with_capacity(f.h.w.len());

    for t in f.h.w.0.iter() {
        let t = TaggedNdArray::from_type(t);
        result.push(t);
    }

    result
}

// Like get_disjoint_mut but i and j can be the same.
fn get_refs_one_mut<T>(data: &mut [T], i: usize, j: usize, k: usize) -> (&T, &T, &mut T) {
    let a = &data[i] as *const T;
    let b = &data[j] as *const T;
    let c = &mut data[k] as *mut T;

    unsafe { (&*a, &*b, &mut *c) }
}

pub type Builder = Rc<RefCell<Term>>;

/// Evaluator state for a single term.
#[derive(Debug, Clone)]
pub struct EvalState {
    term: StrictTerm,
    data: Vec<TaggedNdArray>,
    parameters: Option<HashMap<String, TaggedNdArray>>,
}

impl EvalState {
    /// Preallocate arrays for each node in a term
    pub fn new(f: StrictTerm) -> Self {
        Self {
            data: allocate(&f),
            term: f,
            parameters: None,
        }
    }

    pub fn from_lax(f: Term) -> Self {
        EvalState::new(f.to_open_hypergraph())
    }

    pub fn set_parameters(&mut self, parameters: HashMap<String, TaggedNdArray>) {
        self.parameters = Some(parameters);
    }

    fn apply_binary_operation(
        &mut self,
        sources: &[usize],
        targets: &[usize],
        operation: &Operation,
    ) {
        let (i, j) = (sources[0], sources[1]);
        let k = targets[0];

        match get_refs_one_mut(&mut self.data, i, j, k) {
            (F16(a), F16(b), F16(c)) => {
                let op: Box<dyn kernel::BinOp<f16>> = match operation {
                    Add => Box::new(kernel::AddOp),
                    Sub => Box::new(kernel::SubOp),
                    Mul => Box::new(kernel::MulOp),
                    Div => Box::new(kernel::DivOp),
                    Pow => Box::new(kernel::PowOp),
                    LT => Box::new(kernel::LTOp),
                    EQ => Box::new(kernel::EQOp),
                    MatrixMultiply => Box::new(kernel::MatMulOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(a, b, c);
            }
            (F32(a), F32(b), F32(c)) => {
                let op: Box<dyn kernel::BinOp<f32>> = match operation {
                    Add => Box::new(kernel::AddOp),
                    Sub => Box::new(kernel::SubOp),
                    Mul => Box::new(kernel::MulOp),
                    Div => Box::new(kernel::DivOp),
                    Pow => Box::new(kernel::PowOp),
                    LT => Box::new(kernel::LTOp),
                    EQ => Box::new(kernel::EQOp),
                    MatrixMultiply => Box::new(kernel::MatMulOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(a, b, c);
            }
            (I32(a), I32(b), I32(c)) => {
                let op: Box<dyn kernel::BinOp<i32>> = match operation {
                    Add => Box::new(kernel::AddOp),
                    Sub => Box::new(kernel::SubOp),
                    Mul => Box::new(kernel::MulOp),
                    Div => Box::new(kernel::DivOp),
                    LT => Box::new(kernel::LTOp),
                    EQ => Box::new(kernel::EQOp),
                    Pow => Box::new(kernel::PowOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(a, b, c);
            }
            t => panic!("invalid type: {t:?}"),
        }
    }

    fn apply_unary_operation(
        &mut self,
        sources: &[usize],
        targets: &[usize],
        operation: &Operation,
    ) {
        match self.data[..].get_disjoint_mut([sources[0], targets[0]]) {
            Ok([F16(a), F16(b)]) => {
                let op: Box<dyn kernel::UnaryOp<f16>> = match operation {
                    Negate => Box::new(kernel::NegOp),
                    Not => Box::new(kernel::NotOp),
                    Reshape => Box::new(kernel::ReshapeOp),
                    Broadcast(n) => Box::new(kernel::BroadcastOp { n: n.clone() }),
                    Transpose { dim0, dim1 } => Box::new(kernel::TransposeOp {
                        dim0: *dim0,
                        dim1: *dim1,
                    }),
                    Max => Box::new(kernel::MaxOp),
                    Sum => Box::new(kernel::SumOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, b);
            }
            Ok([F32(a), F32(b)]) => {
                let op: Box<dyn kernel::UnaryOp<f32>> = match operation {
                    Negate => Box::new(kernel::NegOp),
                    Not => Box::new(kernel::NotOp),
                    Reshape => Box::new(kernel::ReshapeOp),
                    Broadcast(n) => Box::new(kernel::BroadcastOp { n: n.clone() }),
                    Transpose { dim0, dim1 } => Box::new(kernel::TransposeOp {
                        dim0: *dim0,
                        dim1: *dim1,
                    }),
                    Max => Box::new(kernel::MaxOp),
                    Sum => Box::new(kernel::SumOp),
                    Sin => Box::new(kernel::SinOp),
                    Cos => Box::new(kernel::CosOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, b);
            }
            Ok([I32(a), I32(b)]) => {
                let op: Box<dyn kernel::UnaryOp<i32>> = match operation {
                    Negate => Box::new(kernel::NegOp),
                    Not => Box::new(kernel::NotOp),
                    Reshape => Box::new(kernel::ReshapeOp),
                    Broadcast(n) => Box::new(kernel::BroadcastOp { n: n.clone() }),
                    Transpose { dim0, dim1 } => Box::new(kernel::TransposeOp {
                        dim0: *dim0,
                        dim1: *dim1,
                    }),
                    Max => Box::new(kernel::MaxOp),
                    Sum => Box::new(kernel::SumOp),
                    _ => panic!("invalid operation"),
                };

                op.apply(&*a, b);
            }
            t => panic!("invalid type: {t:?}"),
        }
    }

    /// Apply an operation to specified sources and target arrays in self.data.
    pub fn apply(&mut self, op: &Operation, sources: &[usize], targets: &[usize]) {
        match op {
            Add | Sub | Mul | Div | Pow | MatrixMultiply | LT | EQ => {
                self.apply_binary_operation(sources, targets, op);
            }
            Sum
            | Max
            | Sin
            | Cos
            | Negate
            | Not
            | Reshape
            | Broadcast { .. }
            | Transpose { .. } => {
                self.apply_unary_operation(sources, targets, op);
            }
            Copy => {
                assert_eq!(sources.len(), 1);
                for t in targets {
                    match self.data[..].get_disjoint_mut([sources[0], *t]) {
                        Ok([F32(a), F32(b)]) => {
                            b.copy_from(a);
                        }
                        Ok([F16(a), F16(b)]) => {
                            b.copy_from(a);
                        }
                        Ok([I32(a), I32(b)]) => {
                            b.copy_from(a);
                        }
                        _ => panic!("invalid types"),
                    }
                }
            }
            Const(k) => match self.data.get_mut(targets[0]) {
                Some(F16(a)) => a.fill(f16::from_f32(*k)),
                Some(F32(a)) => a.fill(*k),
                Some(I32(a)) => a.fill(*k as i32),
                _ => panic!("invalid type"),
            },

            Arange => match self.data.get_mut(targets[0]) {
                Some(I32(a)) => {
                    for (i, x) in a.data.iter_mut().enumerate() {
                        *x = i as i32;
                    }
                }
                Some(F32(a)) => {
                    for (i, x) in a.data.iter_mut().enumerate() {
                        *x = i as f32;
                    }
                }
                _ => panic!("invalid type"),
            },

            Print(name, verbose) => {
                let s = self.data.get(sources[0]).unwrap();
                println!("{}: shape: {:?} stride: {:?}", name, s.shape(), s.strides());

                fn format_slice<T: std::fmt::Debug>(
                    data: &[T],
                    start: usize,
                    end: usize,
                ) -> String {
                    (start..end)
                        .map(|i| format!("{:?}", data[i]))
                        .collect::<Vec<String>>()
                        .join(", ")
                }

                if *verbose {
                    let data = &s.data();
                    let total = s.data().len();
                    let max_per_line = 5;
                    let o = if total <= max_per_line * 2 {
                        format_slice(data, 0, total)
                    } else {
                        format!(
                            "{}, ..., {}",
                            format_slice(data, 0, max_per_line),
                            format_slice(data, total - max_per_line, total)
                        )
                    };
                    println!("{}: {}", name, o);
                }
            }
            Parameter(name) => {
                // TODO:
                // - The matching here is very ugly and incomplete
                // - The parameters are being copied instead of referenced.
                if let Some(parameters) = self.parameters.as_mut() {
                    match self.data.get_mut(targets[0]) {
                        Some(F32(a)) => {
                            let p = parameters.get(name);
                            if let Some(TaggedNdArray::F32(x)) = p {
                                a.copy_from(x);
                            } else {
                                panic!("Parameters loaded, parameter '{name}'::F32 not found.")
                            }
                        }
                        _ => panic!("Invalid type for parameter '{name}'"),
                    }
                } else {
                    panic!("Parameters not loaded, requested parameter '{name}'");
                }
            }
            Embedding => {
                // The first source is the indices and second source is the embedding table
                let i = sources[0]; // indices
                let j = sources[1]; // embedding weights
                let k = targets[0]; // output

                match self.data[..].get_disjoint_mut([i, j, k]) {
                    Ok([I32(indices), F32(weights), F32(output)]) => {
                        let embedding_dim = weights.shape.0[1];

                        // Flatten indices for processing
                        let flat_indices: Vec<_> = indices.data.to_vec();
                        let mut flat_index = 0;

                        // For each index, look up the corresponding embedding vector
                        for &idx in &flat_indices {
                            if idx < 0 || idx as usize >= weights.shape.0[0] {
                                panic!("Embedding index out of bounds");
                            }

                            // Copy embedding vector for this index
                            for j in 0..embedding_dim {
                                // Get the vector at index idx
                                let src_offset = (idx as usize) * embedding_dim + j;
                                output.data[flat_index + j] = weights.data[src_offset];
                            }
                            flat_index += embedding_dim;
                        }
                    }
                    // Similar implementations for other numeric types
                    _ => panic!("invalid type for embedding operation"),
                }
            }
        }
    }

    /// mutably evaluate self with args, returning a reference to output arrays.
    pub fn eval_with(&mut self, args: Vec<TaggedNdArray>) -> Vec<&TaggedNdArray> {
        let sources = &self.term.s.table;

        assert_eq!(
            args.len(),
            sources.len(),
            "Expected {} arguments but got {}.",
            sources.len(),
            args.len()
        );
        for (i, arg) in args.iter().enumerate() {
            self.data[sources[i]] = arg.clone();
        }

        self.eval()
    }

    /// mutably evaluate self, returning a reference to output arrays.
    pub fn eval(&mut self) -> Vec<&TaggedNdArray> {
        // unpack the segmented array of sources into a vec of vecs.
        // TODO: this clones the value - provide non-cloning iter, or one that returns slices?
        #[rustfmt::skip]
        let sources: Vec<Vec<usize>> = self.term.h.s.clone().into_iter().map(|x| x.table.0).collect();
        #[rustfmt::skip]
        let targets: Vec<Vec<usize>> = self.term.h.t.clone().into_iter().map(|x| x.table.0).collect();

        for (l, ops) in layered_operations(&self.term).iter().enumerate() {
            // each layer has any number of ops. TODO: evaluate these in parallel!
            for i in ops {
                let op = self.term.h.x.0[*i].clone();
                log::debug!("{l} OP: {:?}", &op);
                self.apply(&op, &sources[*i], &targets[*i]);
            }
        }

        // Return result array ptrs
        self.term.t.table.0.iter().map(|i| &self.data[*i]).collect()
    }

    pub fn build_lax<F>(f: F) -> Term
    where
        F: Fn(&Builder) -> (Vec<Var>, Vec<Var>),
    {
        let state = Rc::new(RefCell::new(Term::empty()));
        {
            let (s, t) = f(&state);
            state.borrow_mut().sources = s.iter().map(|x| x.new_source()).collect();
            state.borrow_mut().targets = t.iter().map(|x| x.new_target()).collect();
        }
        Rc::try_unwrap(state).unwrap().into_inner()
    }

    pub fn build<F>(f: F) -> EvalState
    where
        F: Fn(&Builder) -> (Vec<Var>, Vec<Var>),
    {
        let term = EvalState::build_lax(f);
        EvalState::from_lax(term)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::operation::Var;
    use crate::core::{Dtype, NdArrayType, Operation, Shape};
    use test_log::test;

    fn test_unarynop_generic<T>(op: Term, x_data: Vec<T>, expected_data: Vec<T>)
    where
        TaggedNdArray: From<NdArray<T>>,
    {
        let x = NdArray::new(x_data, Shape(vec![2, 2]));
        let expected = NdArray::new(expected_data, Shape(vec![2, 2]));

        let mut state = EvalState::from_lax(op);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(tagged.approx(6), actual.approx(6));
    }

    #[test]
    fn test_neg() {
        test_unarynop_generic::<f16>(
            Operation::negate(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F16,
            }),
            vec![1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![-1.0, -2.0, -3.0, -4.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        );
        test_unarynop_generic::<f32>(
            Operation::negate(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-1.0, -2.0, -3.0, -4.0],
        );
        test_unarynop_generic::<i32>(
            Operation::negate(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![1, 2, 3, 4],
            vec![-1, -2, -3, -4],
        );
    }

    #[test]
    fn test_not() {
        test_unarynop_generic::<i32>(
            Operation::not(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![1, 0, -1, 2],
            vec![0, 1, 0, 0],
        );
    }

    #[test]
    fn test_sin_cos() {
        test_unarynop_generic::<f32>(
            Operation::sin(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![0., 1., 2., 3.],
            vec![0.0000, 0.8414709, 0.909297, 0.141120],
        );

        test_unarynop_generic::<f32>(
            Operation::cos(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![0., 1., 2., 3.],
            vec![1.0000, 0.540302, -0.416147, -0.989993],
        );
    }

    fn test_binop_generic<T>(op: Term, x_data: Vec<T>, y_data: Vec<T>, expected_data: Vec<T>)
    where
        TaggedNdArray: From<NdArray<T>>,
    {
        // Get binary operand shape
        let shape = op.hypergraph.nodes[0].shape.clone();

        let x = NdArray::new(x_data, shape.clone());
        let y = NdArray::new(y_data, shape.clone());
        let expected = NdArray::new(expected_data, shape);

        let mut state = EvalState::from_lax(op);

        let [actual] = state.eval_with(vec![x.into(), y.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    #[should_panic]
    fn test_eval_with_argcount() {
        let f = Operation::add(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        });

        let x = NdArray::new(vec![0.; 4], Shape(vec![2, 2]));
        let mut state = EvalState::from_lax(f);

        // Passing a single argument into a binary
        state.eval_with(vec![x.into()]);
    }

    #[test]
    fn test_add() {
        test_binop_generic::<f16>(
            Operation::add(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F16,
            }),
            vec![1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![10.0, 20.0, 30.0, 40.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![11.0, 22.0, 33.0, 44.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        );

        test_binop_generic::<f32>(
            Operation::add(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![11.0, 22.0, 33.0, 44.0],
        );

        // Test for I32
        test_binop_generic::<i32>(
            Operation::add(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![1, 2, 3, 4],
            vec![10, 20, 30, 40],
            vec![11, 22, 33, 44],
        );
    }

    #[test]
    fn test_sub() {
        // Test subtraction with F32
        test_binop_generic::<f32>(
            Operation::sub(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![10.0, 20.0, 30.0, 40.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![9.0, 18.0, 27.0, 36.0],
        );

        // Test subtraction with I32
        test_binop_generic::<i32>(
            Operation::sub(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![10, 20, 30, 40],
            vec![1, 2, 3, 4],
            vec![9, 18, 27, 36],
        );
    }

    #[test]
    fn test_mul() {
        // Test multiplication with F32
        test_binop_generic::<f32>(
            Operation::mul(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![20.0, 60.0, 120.0, 200.0],
        );

        // Test multiplication with I32
        test_binop_generic::<i32>(
            Operation::mul(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![2, 3, 4, 5],
            vec![10, 20, 30, 40],
            vec![20, 60, 120, 200],
        );
    }

    #[test]
    fn test_div() {
        // Test division with F32
        test_binop_generic::<f32>(
            Operation::div(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![2.0, 4.0, 6.0, 8.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![1.0, 2.0, 3.0, 4.0],
        );

        // Test division with I32
        test_binop_generic::<i32>(
            Operation::div(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![2, 4, 6, 8],
            vec![2, 2, 2, 2],
            vec![1, 2, 3, 4],
        );
    }

    #[test]
    fn test_pow() {
        // Test raising to a power with F32
        test_binop_generic::<f32>(
            Operation::pow(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F32,
            }),
            vec![2.0, 4.0, 6.0, 8.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![4.0, 16.0, 36.0, 64.0],
        );

        // Test raising to a power with F16
        test_binop_generic::<f16>(
            Operation::pow(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::F16,
            }),
            vec![2.0, 4.0, 6.0, 8.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![2.0, 2.0, 2.0, 2.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
            vec![4.0, 16.0, 36.0, 64.0]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect(),
        );

        // Test raising to a power with I32
        test_binop_generic::<i32>(
            Operation::pow(NdArrayType {
                shape: Shape(vec![2, 2]),
                dtype: Dtype::I32,
            }),
            vec![2, 4, 6, 8],
            vec![2, 2, 2, 2],
            vec![4, 16, 36, 64],
        );
    }

    #[test]
    fn test_less_than() {
        test_binop_generic::<i32>(
            Operation::lt(NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::I32,
            }),
            vec![1, 2, 3, 4, 5, -6],
            vec![1, 0, 4, -1, 5, 6],
            vec![0, 0, 1, 0, 0, 1],
        );
        test_binop_generic::<f32>(
            Operation::lt(NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::F32,
            }),
            vec![1., 2., 3., 4., 5., -6.],
            vec![1., 0., 4., -1., 5., 6.],
            vec![0., 0., 1., 0., 0., 1.],
        );
    }

    #[test]
    fn test_matmul() {
        let f = Operation::matmul(Shape::empty(), 1, 2, 3, Dtype::F32);

        // a (1×2) matrix
        let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));

        // a (2×3) matrix
        let m = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));

        // result should be a 1×3 result
        let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));

        kernel::batch_matmul::<f32>(&x, &m, &mut expected);

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into(), m.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_matmul_transposed() {
        let f = Operation::matmul(Shape::empty(), 1, 2, 3, Dtype::F32);

        // a (1×2) matrix
        let x = NdArray::new(vec![2., 4.], Shape(vec![1, 2]));

        // a (2×3) matrix
        let m = NdArray::new(vec![1., 2., 3., 4., 5., 6.], Shape(vec![2, 3]));

        // Transposed equivalent of m
        let mut mt = NdArray::new(vec![1., 4., 2., 5., 3., 6.], Shape(vec![3, 2]));
        mt.shape = Shape(vec![2, 3]);
        mt.strides = vec![1, 2];

        // result should be a 1×3 result
        let mut expected = NdArray::new(vec![0.; 3], Shape(vec![1, 3]));

        kernel::batch_matmul::<f32>(&x, &m, &mut expected);

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into(), mt.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = expected.into();
        assert_eq!(&tagged, actual);
    }

    #[test]
    fn test_const() {
        let f = Operation::constop(
            NdArrayType {
                shape: Shape(vec![4, 3]),
                dtype: Dtype::F32,
            },
            2.3,
        );

        let expected = NdArray::new(vec![2.3; 12], Shape(vec![4, 3]));

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    fn test_const_add() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        };

        let const_a = Operation::constop(typ.clone(), 1.0);
        let const_b = Operation::constop(typ.clone(), 2.0);
        let add = Operation::add(typ.clone());

        let expected = NdArray::new(vec![3.0, 3.0, 3.0, 3.0], Shape(vec![2, 2]));

        let f = (&(&const_a | &const_b) >> &add).unwrap();
        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    fn test_parameter() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        };

        let param_a = Operation::parameter(typ.clone(), "param_a");
        let param_b = Operation::parameter(typ.clone(), "param_b");

        let add = Operation::add(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        });

        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], Shape(vec![2, 2]));
        let b = NdArray::new(vec![-1.0, 2.0, -3.0, 4.0], Shape(vec![2, 2]));

        let mut parameters = HashMap::new();
        parameters.insert("param_a".to_string(), a.into());
        parameters.insert("param_b".to_string(), b.into());

        let expected = NdArray::new(vec![0., 4., 0., 8.], Shape(vec![2, 2]));

        let f = (&(&param_a | &param_b) >> &add).unwrap();
        let mut state = EvalState::from_lax(f);
        state.set_parameters(parameters);

        let [actual] = state.eval()[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    #[should_panic]
    fn test_missing_parameter() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F16,
        };

        let param_a = Operation::parameter(typ.clone(), "param_a");

        let mut state = EvalState::from_lax(param_a);
        let parameters = HashMap::new();
        state.set_parameters(parameters);

        state.eval();
    }

    #[test]
    #[should_panic]
    fn test_missing_parameters() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        };

        let param_a = Operation::parameter(typ.clone(), "param_a");

        let mut state = EvalState::from_lax(param_a);

        state.eval();
    }

    #[test]
    fn test_reshape() {
        let f = Operation::reshape(
            NdArrayType {
                shape: Shape(vec![4, 3]),
                dtype: Dtype::I32,
            },
            Shape(vec![2, 6]),
        );

        let x = NdArray::new((0..12).collect(), Shape(vec![4, 3]));

        let expected = NdArray::new((0..12).collect(), Shape(vec![2, 6]));

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    #[should_panic]
    fn test_reshape_invalid_shape() {
        let f = Operation::reshape(
            NdArrayType {
                shape: Shape(vec![4, 3]),
                dtype: Dtype::I32,
            },
            Shape(vec![2, 4]),
        );

        let x = NdArray::new((0..12).collect(), Shape(vec![4, 3]));

        let mut state = EvalState::from_lax(f);

        state.eval_with(vec![x.into()]);
    }

    #[test]
    fn test_broadcast_left() {
        let f = Operation::broadcast(
            NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::I32,
            },
            Shape(vec![2, 1, 2, 3]),
        );

        let x = NdArray::new((30..36).collect(), Shape(vec![2, 3]));

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        // check that array is broadcasted across two new dimensions
        if let TaggedNdArray::I32(actual) = actual {
            assert_eq!(actual.shape.0, &[2, 1, 2, 3]);
            assert_eq!(actual.strides, [0, 0, 3, 1]);

            assert_eq!(actual[&[0, 0, 0, 0]], 30);
            assert_eq!(actual[&[1, 0, 0, 0]], 30);
            assert_eq!(actual[&[0, 0, 0, 2]], 32);
            assert_eq!(actual[&[1, 0, 0, 2]], 32);
            assert_eq!(actual[&[0, 0, 1, 2]], 35);
            assert_eq!(actual[&[1, 0, 1, 2]], 35);
        }
    }

    #[test]
    fn test_broadcast_right() {
        let f = Operation::broadcast(
            NdArrayType {
                shape: Shape(vec![4, 1]),
                dtype: Dtype::I32,
            },
            Shape(vec![4, 3]),
        );

        let x = NdArray::new((30..34).collect(), Shape(vec![4, 1]));

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        // check that array is broadcasted across two new dimensions
        if let TaggedNdArray::I32(actual) = actual {
            assert_eq!(actual.shape.0, &[4, 3]);
            assert_eq!(actual.strides, [1, 0]);

            assert_eq!(actual[&[0, 0]], 30);
            assert_eq!(actual[&[0, 1]], 30);
            assert_eq!(actual[&[0, 2]], 30);
            assert_eq!(actual[&[1, 0]], 31);
            assert_eq!(actual[&[2, 1]], 32);
            assert_eq!(actual[&[3, 2]], 33);
        }
    }
    #[test]
    fn test_transpose() {
        let f = Operation::transpose(
            NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::F32,
            },
            0,
            1,
        );

        // Create a 2x3 matrix
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(vec![2, 3]));

        let mut state = EvalState::from_lax(f);
        let [actual] = state.eval_with(vec![input.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };
        match actual {
            TaggedNdArray::F32(array) => {
                assert_eq!(array.shape, Shape(vec![3, 2]));
                assert_eq!(array[&[0, 0]], 1.0);
                assert_eq!(array[&[0, 1]], 4.0);
                assert_eq!(array[&[1, 0]], 2.0);
                assert_eq!(array[&[1, 1]], 5.0);
                assert_eq!(array[&[2, 0]], 3.0);
                assert_eq!(array[&[2, 1]], 6.0);
            }
            _ => panic!("Expected F32 array"),
        }
    }

    #[test]
    #[should_panic]
    fn test_transpose_invalid_dim() {
        let f = Operation::transpose(
            NdArrayType {
                shape: Shape(vec![2, 3]),
                dtype: Dtype::F32,
            },
            0,
            2,
        );

        // Create a 2x3 matrix
        let input = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(vec![2, 3]));

        let mut state = EvalState::from_lax(f);
        state.eval_with(vec![input.into()]);
    }

    #[test]
    fn test_copy_from() {
        let f = Operation::copy(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        });

        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], Shape(vec![2, 2]));

        let tagged: TaggedNdArray = x.into();

        let mut state = EvalState::from_lax(f);

        let [copy1, copy2] = state.eval_with(vec![tagged.clone()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(copy1, &tagged);
        assert_eq!(copy2, &tagged);
    }

    #[test]
    fn test_max() {
        let f = Operation::max(NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        });

        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], Shape(vec![2, 2]));

        let expected = NdArray::new(vec![2.0, 4.0], Shape(vec![2]));

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    #[test]
    fn test_sum() {
        let f = Operation::sum(NdArrayType {
            shape: Shape(vec![2, 3]),
            dtype: Dtype::F32,
        });

        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape(vec![2, 3]));

        let expected = NdArray::new(vec![6.0, 15.0], Shape(vec![2]));

        let mut state = EvalState::from_lax(f);

        let [actual] = state.eval_with(vec![x.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        assert_eq!(actual, &expected.into());
    }

    // Var interface test
    #[test]
    fn test_var_add() {
        let typ = NdArrayType {
            shape: Shape(vec![2, 2]),
            dtype: Dtype::F32,
        };

        let mut state = EvalState::build(|builder| {
            let a = Var::new(builder.clone(), typ.clone());
            let b = Var::new(builder.clone(), typ.clone());

            let c = a.clone() + b.clone();
            (vec![a, b], vec![c])
        });

        let x = NdArray::new(vec![1., 2., 3., 4.], Shape(vec![2, 2]));
        let y = NdArray::new(vec![1., 1., 1., 1.], Shape(vec![2, 2]));
        let exp = NdArray::new(vec![2., 3., 4., 5.], Shape(vec![2, 2]));

        let [actual] = state.eval_with(vec![x.into(), y.into()])[..] else {
            panic!("unexpected coarity at eval time")
        };

        let tagged: TaggedNdArray = exp.into();
        assert_eq!(&tagged, actual);
    }
}
