use catgrad::prelude::ops::*;
use catgrad::prelude::*;

use std::collections::BTreeMap;

/// Construct, shapecheck, and interpret the `SimpleMNISTModel` using the ndarray backend.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = SimpleMNISTModel;

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");
    save_svg(&typed_term.term, &format!("{}.svg", model.path()))?;

    // Create parameters for the model
    let parameters = load_param_types();

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Shapecheck the model
    let check_result =
        typecheck::check(&env, &parameters, typed_term.clone()).expect("typecheck failed");

    // Diagram of term with shapes inferred
    let labeled_term = typed_term.term.clone().with_nodes(|_| check_result);
    let filename = &format!("{}_typed.svg", model.path());
    save_svg(&labeled_term.unwrap(), filename)?;

    // Choose a backend from available features
    let backend = select_backend()?;

    // Run the interpreter with the selected backend
    let results = run_interpreter(&backend, &typed_term, env)?;

    // Print the `Value`s returned by the interpreter.
    for value in results {
        println!("{value:?}");
    }

    Ok(())
}

fn run_interpreter<B: interpreter::Backend>(
    backend: &B,
    typed_term: &TypedTerm,
    env: Environment,
) -> Result<Vec<interpreter::Value<B>>, Box<dyn std::error::Error>> {
    // Create sample input data: batch of 2 MNIST-like images (28x28)
    let input_data: Vec<f32> = (0..2 * 28 * 28)
        .map(|i| (i as f32 * 0.001) % 1.0) // Simple pattern: values between 0 and 1
        .collect();

    let interpreter_params = load_param_data(backend);
    let interpreter = interpreter::Interpreter::new(backend.clone(), env, interpreter_params);

    let input_tensor = interpreter::tensor(
        &interpreter.backend,
        interpreter::Shape(vec![2, 28, 28]),
        input_data,
    )
    .expect("Failed to create input tensor");

    let results = interpreter
        .run(typed_term.term.clone(), vec![input_tensor])
        .expect("Failed to run inference");

    Ok(results)
}

/// Pick a backend depending on what features are available
fn select_backend() -> Result<impl interpreter::Backend, Box<dyn std::error::Error>> {
    #[cfg(feature = "candle-backend")]
    {
        println!("selected candle backend...");
        use catgrad::interpreter::backend::candle::CandleBackend;
        #[allow(clippy::needless_return)]
        return Ok(CandleBackend::new());
    }

    #[cfg(all(feature = "ndarray-backend", not(feature = "candle-backend")))]
    {
        println!("selected ndarray backend...");
        use catgrad::interpreter::backend::ndarray::NdArrayBackend;
        #[allow(clippy::needless_return)]
        return Ok(NdArrayBackend);
    }

    #[cfg(not(any(feature = "candle-backend", feature = "ndarray-backend")))]
    {
        println!("selected ShapeOnly backend (no tensors computed)");
        return Ok(interpreter::backend::shape_only::ShapeOnlyBackend);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Define the SimpleMNISTModel model

pub struct SimpleMNISTModel;

// Implement `Def`: this is like torch's `Module`.
impl Module<1, 1> for SimpleMNISTModel {
    // Model name
    // TODO: NOTE: it's not clear how user is supposed to know how to choose this name!
    fn path(&self) -> Path {
        Path::new(["model", "hidden"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        // Flatten input from B×28×28 to B×784
        let [batch_size, h, w] = unpack::<3>(builder, shape(builder, x.clone()));
        let flat_size = h * w;
        let flat_shape = pack::<2>(builder, [batch_size, flat_size]);
        let x = reshape(builder, flat_shape, x);

        let root = self.path();

        let p = param(builder, &root.extend(["0", "weights"]).unwrap());

        // layer 1: B×784 @ 784×100 = B×100
        let x = matmul(builder, x, p);
        let x = nn::Sigmoid.call(builder, [x]);

        // layer 2: B×100 @ 100×10 = B×10
        let p = param(builder, &root.extend(["1", "weights"]).unwrap());
        let x = matmul(builder, x, p);
        let x = nn::Sigmoid.call(builder, [x]);

        // result
        [x]
    }

    // This should return the *detailed* type of the model
    // TODO: NOTE: API for writing types is still WIP. Lots of boilerplate here!
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use catgrad::typecheck::*;

        let batch_size = NatExpr::Var(0);

        // Input shape B×28×28
        let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![
                batch_size.clone(),
                NatExpr::Constant(28),
                NatExpr::Constant(28),
            ]),
        }));

        // Output shape B×10
        let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(10)]),
        }));

        ([t_x], [t_y])
    }
}

////////////////////////////////////////////////////////////////////////////////
// Parameter loading boilerplate
// NOTE: in reality, this would be done by loading e.g. a safetensors file.

// NOTE: you would normally create this data by reading the safetensors file!
fn load_param_types() -> typecheck::Parameters {
    use catgrad::category::core::Dtype;
    use catgrad::typecheck::value_types::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr};

    let mut map = BTreeMap::new();

    // Layer 1: (28*28) → 100
    let layer1_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            NatExpr::Mul(vec![NatExpr::Constant(28), NatExpr::Constant(28)]),
            NatExpr::Constant(100),
        ]),
    }));
    map.insert(
        path(vec!["0", "weights"]).expect("invalid param path"),
        layer1_type,
    );

    // Layer 2: 100 → 10
    let layer2_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![NatExpr::Constant(100), NatExpr::Constant(10)]),
    }));
    map.insert(
        path(vec!["1", "weights"]).expect("invalid param path"),
        layer2_type,
    );

    typecheck::Parameters::from(map)
}

// NOTE: you would normally create this data by reading the safetensors file!
fn load_param_data<B: interpreter::Backend>(backend: &B) -> interpreter::Parameters<B> {
    use catgrad::category::core::Shape;
    use std::collections::BTreeMap;

    let mut map = BTreeMap::new();

    // Layer 1 weights: [784, 100] - initialize with small random-ish values
    let layer1_data: Vec<f32> = (0..784 * 100)
        .map(|i| (i as f32 * 0.01 % 2.0) - 1.0) // Simple pattern: values between -1 and 1
        .collect();
    let layer1_tensor = interpreter::tensor(backend, Shape(vec![784, 100]), layer1_data)
        .expect("failed to create tensor");
    map.insert(
        path(vec!["0", "weights"]).expect("invalid param path"),
        layer1_tensor,
    );

    // Layer 2 weights: [100, 10]
    let layer2_data: Vec<f32> = (0..100 * 10)
        .map(|i| (i as f32 * 0.01 % 2.0) - 1.0)
        .collect();
    let layer2_tensor = interpreter::tensor(backend, Shape(vec![100, 10]), layer2_data)
        .expect("failed to create tensor");
    map.insert(
        path(vec!["1", "weights"]).expect("invalid param path"),
        layer2_tensor,
    );

    interpreter::Parameters::from(map)
}

#[cfg(feature = "svg")]
pub fn save_svg<
    O: PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
    A: PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
>(
    term: &open_hypergraphs::lax::OpenHypergraph<O, A>,
    filename: &str,
) -> Result<(), std::io::Error> {
    use catgrad::svg::to_svg;
    let bytes = match to_svg(term) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to generate SVG: {e}");
            return Ok(());
        }
    };

    let output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("images");

    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln!("Failed to create directory {output_dir:?}: {e}");
        return Ok(());
    }

    let output_path = output_dir.join(filename);
    println!("saving svg to {output_path:?}");

    if let Err(e) = std::fs::write(&output_path, bytes) {
        eprintln!("Failed to write SVG file {output_path:?}: {e}");
    }

    Ok(())
}

#[cfg(not(feature = "svg"))]
pub fn save_svg<O, A>(
    _term: &open_hypergraphs::lax::OpenHypergraph<O, A>,
    _filename: &str,
) -> Result<(), std::io::Error> {
    println!("SVG feature not enabled, skipping diagram generation");
    Ok(())
}

// include this as a test
#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main().unwrap();
    }
}
