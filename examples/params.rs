use open_hypergraphs::prelude::Arrow;
use safetensors;
use std::collections::HashMap;

use catgrad::{
    backend::cpu::{
        eval::EvalState,
        ndarray::{NdArray, TaggedNdArray},
    },
    core::{identity, Dtype, NdArrayType, Operation, Shape, Term},
};

fn sigmoid_layer(typ: NdArrayType) -> Term {
    let one = Operation::Const {
        x: typ.clone(),
        k: 1.0,
    }
    .term();

    let e = Operation::Const {
        x: typ.clone(),
        k: std::f32::consts::E,
    }
    .term();

    let pow = Operation::Pow(typ.clone()).term();
    let neg = Operation::Negate(typ.clone()).term();
    let add = Operation::Add(typ.clone()).term();
    let div = Operation::Div(typ.clone()).term();

    let f = (&(&e | &neg) >> &pow).unwrap();
    let f = (&(&one | &f) >> &add).unwrap();
    let f = (&(&one | &f) >> &div).unwrap();

    f
}

fn tanh_layer(typ: NdArrayType) -> Term {
    let one = Operation::Const {
        x: typ.clone(),
        k: 1.0,
    }
    .term();

    let two = Operation::Const {
        x: typ.clone(),
        k: 2.0,
    }
    .term();

    let id = identity(vec![typ.clone()]);
    let mul = Operation::Mul(typ.clone()).term();
    let sub = Operation::Sub(typ.clone()).term();

    let f = (&(&id | &two) >> &mul).unwrap(); // 2*x
    let f = (&f >> &sigmoid_layer(typ)).unwrap(); // sigmoid(2*x)
    let f = (&(&f | &two) >> &mul).unwrap(); // 2*sigmoid(2*x)
    let f = (&(&f | &one) >> &sub).unwrap();

    f
}

#[allow(unused)]
fn show(name: &str, term: &Term) {
    println!(
        "{name} sources: {:?}\n{name} targets: {:?}",
        term.source(),
        term.target()
    );
}

// linear + tanh + linear
fn mlp_layer(input_features: usize, output_features: usize, dtype: Dtype, name: &str) -> Term {
    let typ = NdArrayType {
        shape: Shape(vec![1, output_features]),
        dtype: dtype.clone(),
    };

    let l1 = (&linear_layer(
        input_features,
        output_features,
        dtype.clone(),
        format!("{name}.lin1").as_str(),
    ) >> &tanh_layer(typ))
        .unwrap();
    return (&l1
        >> &linear_layer(
            output_features,
            input_features,
            dtype,
            format!("{name}.lin2").as_str(),
        ))
        .unwrap();
}

fn linear_layer(
    // batch_size: usize,
    input_features: usize,
    output_features: usize,
    dtype: Dtype,
    name: &str,
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

    let id_x = identity(vec![x_type.clone()]);

    let param_w = Operation::Parameter {
        x: w_type.clone(),
        name: format!("{name}.weight"),
    }
    .term();

    let param_b = Operation::Parameter {
        x: out_type.clone(),
        name: format!("{name}.bias"),
    }
    .term();

    let transpose = Operation::Transpose {
        x: w_type.clone(),
        dim0: 0,
        dim1: 1,
    }
    .term();

    let matmul = Operation::MatrixMultiply {
        n: Shape::empty(),
        a: batch_size,
        b: input_features,
        c: output_features,
        dtype: dtype.clone(),
    }
    .term();

    let add = Operation::Add(out_type.clone()).term();

    let trans = (&param_w >> &transpose).unwrap();
    let mm = (&(&id_x | &trans) >> &matmul).unwrap();
    let term = (&(&param_b | &mm) >> &add).unwrap();

    term
}

struct Model {
    pub term: Term,
}

impl Model {
    pub fn build(in_dim: usize, out_dim: usize) -> Self {
        let term = mlp_layer(in_dim, out_dim, Dtype::F32, "layers.0");
        let term = (&term >> &mlp_layer(in_dim, out_dim, Dtype::F32, "layers.1")).unwrap();
        let term = (&term >> &mlp_layer(in_dim, out_dim, Dtype::F32, "layers.2")).unwrap();
        let term = (&term >> &mlp_layer(in_dim, out_dim, Dtype::F32, "layers.3")).unwrap();

        Self { term }
    }

    pub fn run(&self, x: &NdArray<f32>) -> TaggedNdArray {
        let mut state = EvalState::new(self.term.clone());
        let tensors = read_safetensors("model.safetensors");
        state.set_parameters(tensors);
        let [result] = state.eval_with(vec![x.clone().into()])[..] else {
            panic!("unexpected result")
        };

        result.clone()
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
    let model = Model::build(8, 24);
    let result = model.run(&input);
    println!("input {:?}", input);
    println!("Result: {:?}", result);
}
