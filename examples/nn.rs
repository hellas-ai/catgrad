use open_hypergraphs::prelude::Arrow;
use safetensors;
use std::collections::HashMap;

use catgrad::{
    backend::cpu::{
        eval::EvalState,
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{Dtype, NdArrayType, Operation, Shape, Term},
};

#[cfg(test)]
fn chained(x: &NdArray<f32>, y: &NdArray<f32>) -> TaggedNdArray {
    let typ = NdArrayType {
        shape: x.shape.clone(),
        dtype: Dtype::F32,
    };

    let ones = Operation::constop(typ.clone(), 1.0);
    let neg = Operation::negate(typ.clone());
    let add = Operation::add(typ.clone());
    let mul = Operation::mul(typ.clone());

    let id = Operation::identity(vec![typ.clone()]);

    // -(x+1)*(y+1)
    let op = (&(&(&(&ones | &id) | &(&ones | &id)) >> &(&add | &add)).unwrap() >> &mul).unwrap();
    let op = (&op >> &neg).unwrap();
    let mut state = EvalState::from_lax(op);
    let [result] = state.eval_with(vec![x.clone().into(), y.clone().into()])[..] else {
        panic!("unexpected result")
    };

    result.clone()
}

#[cfg(test)]
fn sigmoid(x: &NdArray<f32>) -> TaggedNdArray {
    let typ = NdArrayType {
        shape: x.shape.clone(),
        dtype: Dtype::F32,
    };

    let sl = sigmoid_layer(typ);
    let mut state = EvalState::from_lax(sl);
    let [result] = state.eval_with(vec![x.clone().into()])[..] else {
        panic!("unexpected neg result")
    };

    result.clone()
}

fn sigmoid_layer(typ: NdArrayType) -> Term {
    let one = Operation::constop(typ.clone(), 1.0);
    let e = Operation::constop(typ.clone(), std::f32::consts::E);

    let pow = Operation::pow(typ.clone());
    let neg = Operation::negate(typ.clone());
    let add = Operation::add(typ.clone());
    let div = Operation::div(typ.clone());

    let f = (&(&e | &neg) >> &pow).unwrap();
    let f = (&(&one | &f) >> &add).unwrap();
    let f = (&(&one | &f) >> &div).unwrap();

    f
}

fn tanh_layer(typ: NdArrayType) -> Term {
    let one = Operation::constop(typ.clone(), 1.0);
    let two = Operation::constop(typ.clone(), 2.0);

    let id = Operation::identity(vec![typ.clone()]);
    let mul = Operation::mul(typ.clone());
    let sub = Operation::sub(typ.clone());

    let f = (&(&id | &two) >> &mul).unwrap(); // 2*x
    let f = (&f >> &sigmoid_layer(typ)).unwrap(); // sigmoid(2*x)
    let f = (&(&f | &two) >> &mul).unwrap(); // 2*sigmoid(2*x)
    let f = (&(&f | &one) >> &sub).unwrap();

    f
}

fn tanh(x: &NdArray<f32>) -> TaggedNdArray {
    let typ = NdArrayType {
        shape: x.shape.clone(),
        dtype: Dtype::F32,
    };

    let tl = tanh_layer(typ);
    let mut state = EvalState::from_lax(tl);
    let [result] = state.eval_with(vec![x.clone().into()])[..] else {
        panic!("unexpected sub result")
    };

    // tanh(x) = 2*sigmoid(2*x) - 1
    result.clone()
}

#[allow(unused)]
fn show(name: &str, term: &Term) {
    println!(
        "{name} sources: {:?}\n{name} targets: {:?}",
        term.source(),
        term.target()
    );
}

fn linear_layer(
    // batch_size: usize,
    input_features: usize,
    output_features: usize,
    dtype: Dtype,
) -> Term {
    let batch_size = 1;

    // Input
    let x_type = NdArrayType {
        shape: Shape(vec![batch_size, input_features]),
        dtype: dtype.clone(),
    };
    // Weights
    let w_type = NdArrayType {
        shape: Shape(vec![output_features, input_features]),
        dtype: dtype.clone(),
    };
    // Result
    let out_type = NdArrayType {
        shape: Shape(vec![batch_size, output_features]),
        dtype: dtype.clone(),
    };

    let id_x = Operation::identity(vec![x_type.clone()]);
    let id_b = Operation::identity(vec![out_type.clone()]);

    let transpose = Operation::transpose(w_type.clone(), 0, 1);

    let matmul = Operation::matmul(
        Shape::empty(),
        batch_size,
        input_features,
        output_features,
        dtype.clone(),
    );

    let add = Operation::add(out_type.clone());

    let step1 = &(&id_x | &transpose) | &id_b;
    let step2 = &matmul | &id_b;
    let step3 = add;

    let term = (&(&step1 >> &step2).unwrap() >> &step3).unwrap();

    term
}

fn linear(x: &NdArray<f32>, w: &NdArray<f32>, b: &NdArray<f32>) -> TaggedNdArray {
    let lin = linear_layer(x.shape.0[1], w.shape.0[0], Dtype::F32);
    // Evaluate add step
    let mut add_state = EvalState::from_lax(lin);
    let [result] =
        add_state.eval_with(vec![x.clone().into(), w.clone().into(), b.clone().into()])[..]
    else {
        panic!("unexpected linear layer result")
    };

    result.clone()
}

// Linear layer function
#[allow(unused)]
fn linear_with_evals(x: &NdArray<f32>, w: &NdArray<f32>, b: &NdArray<f32>) -> TaggedNdArray {
    // Create operations for nn linear layer
    let matmul = Operation::matmul(
        Shape::empty(),
        x.shape.0[0],
        x.shape.0[1],
        w.shape.0[0],
        Dtype::F32,
    );

    let add = Operation::add(NdArrayType {
        shape: Shape(vec![x.shape.0[0], w.shape.0[0]]),
        dtype: Dtype::F32,
    });

    let transpose = Operation::transpose(
        NdArrayType {
            shape: Shape(vec![w.shape.0[0], w.shape.0[1]]),
            dtype: Dtype::F32,
        },
        0,
        1,
    );

    // Transpose w
    let mut state = EvalState::from_lax(transpose);
    let [wt] = state.eval_with(vec![w.clone().into()])[..] else {
        panic!("unexpected transpose result")
    };

    // Evaluate matmul step
    let mut state = EvalState::from_lax(matmul);
    let [result] = state.eval_with(vec![x.clone().into(), wt.clone()])[..] else {
        panic!("unexpected matmul result")
    };

    // Evaluate add step
    let mut add_state = EvalState::from_lax(add);
    let [result] = add_state.eval_with(vec![result.clone(), b.clone().into()])[..] else {
        panic!("unexpected add result")
    };

    result.clone()
}

struct Model {
    pub tensors: HashMap<String, TaggedNdArray>,
}

impl Model {
    pub fn load(path: &str) -> Self {
        let tensors = read_safetensors(&path);
        Self { tensors }
    }

    fn get(&self, name: &str) -> &NdArray<f32> {
        let t = self
            .tensors
            .get(name)
            .expect(&format!("No parameter found for name {name}"));
        if let TaggedNdArray::F32(t) = t {
            t
        } else {
            panic!("No tensor");
        }
    }

    fn linear(&self, x: &NdArray<f32>, name: &str) -> TaggedNdArray {
        let weight = self.get(&format!("{name}.weight"));
        let bias = self.get(&format!("{name}.bias"));
        if weight.shape.0[1] != x.shape.0[1] {
            panic!(
                "Input features ({}) don't match weight dimensions ({})",
                x.shape.0[1], weight.shape.0[1]
            );
        }
        linear(x, weight, bias)
    }

    pub fn layer(&self, x: &NdArray<f32>, name: &str) -> TaggedNdArray {
        let res = self.linear(x, &format!("{name}.lin1"));
        let x = if let TaggedNdArray::F32(r) = res {
            r
        } else {
            panic!("invalid");
        };

        let res = tanh(&x);

        let x = if let TaggedNdArray::F32(r) = res {
            r
        } else {
            panic!("invalid");
        };
        self.linear(&x, &format!("{name}.lin2"))
    }

    pub fn run(&self, x: &NdArray<f32>) -> TaggedNdArray {
        let mut x = x.clone();
        for idx in 0..4 {
            let res = self.layer(&x, &format!("layers.{idx}"));
            x = if let TaggedNdArray::F32(r) = res {
                r
            } else {
                panic!("invalid");
            };
        }
        x.into()
    }
}

// Read tensor data from safetensors file
fn read_safetensors(path: &str) -> HashMap<String, TaggedNdArray> {
    // Load file
    let data = std::fs::read(path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

    // Initialize result map
    let mut result = HashMap::new();

    // Read each tensor
    for name in tensors.names() {
        let view = tensors.tensor(name).unwrap();
        let shape = Shape(view.shape().to_vec());

        // Convert dtype and load tensor data
        match view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = view
                    .data()
                    .chunks(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                result.insert(
                    name.to_string(),
                    TaggedNdArray::F32(NdArray::new(data, shape)),
                );
            }
            // Add other dtype conversions as needed
            _ => panic!("Unsupported dtype"),
        }
    }

    result
}

pub fn main() {
    let input = NdArray::new(vec![1.0; 8], Shape(vec![1, 8]));
    let model = Model::load("model.safetensors");
    let result = model.run(&input);
    println!("input {:?}", input);
    println!("Result: {:?}", result);
}

#[test]
fn test_chained() {
    // Create test input data
    let x = NdArray::new(vec![1.0, 2.0, 3.0], Shape(vec![1, 3]));
    let y = NdArray::new(vec![1.0, 2.0, 3.0], Shape(vec![1, 3]));

    // Calculate -(x+1)*(y+1)
    let result = chained(&x, &y);

    // Check result shape
    match result {
        TaggedNdArray::F32(arr) => {
            assert_eq!(arr.shape.0, vec![1, 3]);
            assert_eq!(arr.data, vec![-4.0, -9.0, -16.0]);
        }
        _ => panic!("wrong type"),
    }
}

#[cfg(test)]
fn allclose<T>(a: &[T], b: &[T], rtol: f32, atol: f32) -> bool
where
    T: Into<f32> + Copy,
{
    if a.len() != b.len() {
        return false; // Vectors must have the same length
    }

    std::iter::zip(a, b).all(|(&x, &y)| {
        let (x, y) = (x.into(), y.into());
        (x - y).abs() <= atol + rtol * y.abs()
    })
}

#[test]
fn test_linear() {
    // Create test input data
    let x = NdArray::new(vec![1.0, 2.0, 3.0], Shape(vec![1, 3]));
    let w = NdArray::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], Shape(vec![2, 3]));
    let b = NdArray::new(vec![0.1, 0.1], Shape(vec![1, 2]));

    // Run linear layer (x * w^T + b)
    let result = linear(&x, &w, &b);

    // Check result shape
    match result {
        TaggedNdArray::F32(arr) => {
            assert_eq!(arr.shape.0, vec![1, 2]);
            assert_eq!(arr.data, vec![1.5, 3.3]);
        }
        _ => panic!("wrong type"),
    }
}

#[test]
fn test_sigmoid() {
    let x = NdArray::new(vec![1.0; 10], Shape(vec![2, 5]));
    let result = sigmoid(&x);

    // Check result shape
    match result {
        TaggedNdArray::F32(arr) => {
            assert_eq!(arr.shape, Shape(vec![2, 5]));
            assert_eq!(arr.data, vec![0.7310586; 10]);
            assert!(allclose(
                arr.data.as_slice(),
                vec![0.731058; 10].as_slice(),
                1e-6,
                1e-6
            ));
        }
        _ => panic!("wrong type"),
    }
}

#[test]
fn test_tanh() {
    let x = NdArray::new(vec![1.0; 10], Shape(vec![2, 5]));
    let result = tanh(&x);

    // Check result shape
    match result {
        TaggedNdArray::F32(arr) => {
            assert_eq!(arr.shape, Shape(vec![2, 5]));
            assert!(allclose(
                arr.data.as_slice(),
                vec![0.761594; 10].as_slice(),
                1e-6,
                1e-6
            ));
        }
        _ => panic!("wrong type"),
    }
}
