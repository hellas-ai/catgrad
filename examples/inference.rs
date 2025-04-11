// Example NN model inference
// Terms built using the algebraic API

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
    let type_in = NdArrayType {
        shape: Shape(vec![1, input_features]),
        dtype,
    };
    let type_out = NdArrayType {
        shape: Shape(vec![1, output_features]),
        dtype,
    };

    let copy = Operation::copy(type_in.clone());
    let add = Operation::add(type_in.clone());
    let id_x = Operation::identity(vec![type_in.clone()]);

    let l1 = (&linear_layer(
        input_features,
        output_features,
        dtype,
        &format!("{name}.lin1"),
    ) >> &tanh_layer(type_out))
        .unwrap();
    let l2 = (&l1
        >> &linear_layer(
            output_features,
            input_features,
            dtype,
            &format!("{name}.lin2"),
        ))
        .unwrap();

    let term = (&copy >> &(&id_x | &l2)).unwrap();
    let term = (&term >> &add).unwrap();

    term
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
        dtype,
    };
    // Weights
    let w_type = NdArrayType {
        shape: Shape(vec![output_features, input_features]),
        dtype,
    };
    // Bias
    let b_type = NdArrayType {
        shape: Shape(vec![output_features]),
        dtype,
    };
    // Result
    let out_type = NdArrayType {
        shape: Shape(vec![batch_size, output_features]),
        dtype,
    };

    let id_x = Operation::identity(vec![x_type.clone()]);

    let param_w = Operation::parameter(w_type.clone(), &format!("{name}.weight"));
    let param_b = Operation::parameter(b_type.clone(), &format!("{name}.bias"));

    let transpose = Operation::transpose(w_type.clone(), 0, 1);
    let broadcast = Operation::broadcast(b_type.clone(), Shape(vec![1]));

    let matmul = Operation::matmul(
        Shape::empty(),
        batch_size,
        input_features,
        output_features,
        dtype,
    );

    let add = Operation::add(out_type.clone());

    let transposed_w = (&param_w >> &transpose).unwrap();
    let broadcasted_bias = (&param_b >> &broadcast).unwrap();
    let mm = (&(&id_x | &transposed_w) >> &matmul).unwrap();
    let term = (&(&broadcasted_bias | &mm) >> &add).unwrap();

    term
}

#[derive(Debug)]
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
        let mut state = EvalState::from_lax(self.term.clone());
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
    let model = Model::build(8, 16);
    // println!("Model {:#?}", &model);
    let result = model.run(&input);
    println!("input {:?}", input);
    println!("Result: {:?}", result);
}
