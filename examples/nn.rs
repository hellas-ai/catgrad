use safetensors;
use std::collections::HashMap;

use catgrad::{
    backend::cpu::{
        eval::EvalState,
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{Dtype, NdArrayType, Operation, Shape},
};

fn sigmoid(x: &NdArray<f32>) -> TaggedNdArray {
    let len = x.shape.size();
    let typ = NdArrayType {
        shape: x.shape.clone(),
        dtype: Dtype::F32,
    };

    let e = std::f32::consts::E;
    let es = NdArray::new(vec![e; len], Shape(vec![len]));
    let ones = NdArray::new(vec![1.; len], Shape(vec![len]));

    let pow = Operation::Pow(typ.clone()).term();
    let neg = Operation::Negate(typ.clone()).term();
    let add = Operation::Add(typ.clone()).term();
    let div = Operation::Div(typ.clone()).term();

    let mut state = EvalState::new(neg);
    let [result] = state.eval_with(vec![x.clone().into()])[..] else {
        panic!("unexpected neg result")
    };

    let mut state = EvalState::new(pow);
    let [result] = state.eval_with(vec![es.into(), result.clone()])[..] else {
        panic!("unexpected pow result")
    };

    let mut state = EvalState::new(add);
    let [result] = state.eval_with(vec![ones.clone().into(), result.clone()])[..] else {
        panic!("unexpected add result")
    };

    let mut state = EvalState::new(div);
    let [result] = state.eval_with(vec![ones.clone().into(), result.clone()])[..] else {
        panic!("unexpected div result")
    };

    result.clone()
}

fn tanh(x: &NdArray<f32>) -> TaggedNdArray {
    let len = x.shape.size();

    let typ = NdArrayType {
        shape: x.shape.clone(),
        dtype: Dtype::F32,
    };

    let twos = NdArray::new(vec![2.; len], Shape(vec![len]));

    let mul = Operation::Mul(typ.clone()).term();
    let sub = Operation::Sub(typ.clone()).term();

    let mut state = EvalState::new(mul.clone());
    let [result] = state.eval_with(vec![twos.clone().into(), x.clone().into()])[..] else {
        panic!("unexpected mul result")
    };

    let result = match result {
        TaggedNdArray::F32(arr) => arr,
        _ => panic!("Unimplemented"),
    };

    // sigmoid(2*x)
    let sig = sigmoid(result);

    let mut state = EvalState::new(mul);
    let [result] = state.eval_with(vec![twos.clone().into(), sig.clone()])[..] else {
        panic!("unexpected mul result")
    };

    let ones = NdArray::new(vec![1.; len], Shape(vec![len]));

    let mut state = EvalState::new(sub);
    let [result] = state.eval_with(vec![result.clone(), ones.clone().into()])[..] else {
        panic!("unexpected sub result")
    };

    // tanh(x) = 2*sigmoid(2*x) - 1
    result.clone()
}

// Linear layer function
fn linear(x: &NdArray<f32>, w: &NdArray<f32>, b: &NdArray<f32>) -> TaggedNdArray {
    // Create operations for nn linear layer
    let matmul = Operation::MatrixMultiply {
        n: Shape::empty(),
        a: x.shape.0[0],
        b: x.shape.0[1],
        c: w.shape.0[0],
        dtype: Dtype::F32,
    }
    .term();

    let add = Operation::Add(NdArrayType {
        shape: Shape(vec![x.shape.0[0], w.shape.0[0]]),
        dtype: Dtype::F32,
    })
    .term();

    let transpose = Operation::Transpose {
        x: NdArrayType {
            shape: Shape(vec![x.shape.0[0], w.shape.0[1]]),
            dtype: Dtype::F32,
        },
        dim0: 0,
        dim1: 1,
    }
    .term();

    // Transpose w
    let mut state = EvalState::new(transpose);
    let [wt] = state.eval_with(vec![w.clone().into()])[..] else {
        panic!("unexpected transpose result")
    };

    // Evaluate matmul step
    let mut state = EvalState::new(matmul);
    let [result] = state.eval_with(vec![x.clone().into(), wt.clone()])[..] else {
        panic!("unexpected matmul result")
    };

    // Evaluate add step
    let mut add_state = EvalState::new(add);
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
