use catgrad::prelude::*;
use catgrad::typecheck::*;
use catgrad_mlir::{compile::CompiledModel, runtime::LlvmRuntime};
use std::collections::HashMap;
use std::env;

/// Construct, shapecheck, and lower an `Exp` function to MLIR, then codegen and run.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    ////////////////////////////////////////
    // Parse arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <output.so>", args[0]);
        std::process::exit(1);
    }

    ////////////////////////////////////////
    // Setup model and environment
    let model = ConcreteSigmoidPlusConst;
    let typed_term = model.term().expect("Failed to create typed term");

    // Create parameters map with the constant parameter
    let param_type = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![].into_iter().map(NatExpr::Constant).collect()), // scalar
    }));
    let param_name = path(vec!["offset"]).unwrap();
    let parameters = typecheck::Parameters::from([(param_name.clone(), param_type)]);

    // TODO: this is a lot of work for the user...?
    let mut env = stdlib();
    env.definitions.extend([(model.path(), typed_term)]);
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    ////////////////////////////////////////
    // Compile and set up runtime with compiled code
    println!("Compiling {}...", model.path());
    let compiled_model = CompiledModel::new(&env, &parameters, model.path());

    // Set the scalar parameter (value 1.0)
    let scalar_data = vec![1.0f32];
    let param_tensor = LlvmRuntime::tensor(scalar_data, vec![]); // scalar tensor
    let param_values = HashMap::from([(model.path().concat(&param_name), param_tensor)]);

    ////////////////////////////////////////
    // Execute with example data
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 10.0, 11.0, 12.0, // 2
    ];
    let input_tensor = LlvmRuntime::tensor(input_data, vec![3, 1, 4]);
    println!("Input tensor: {}", input_tensor);

    // Call the function using the CompiledModel API
    let results = compiled_model.call(model.path(), &param_values, vec![input_tensor])?;

    // Print each result using Display
    for (i, result) in results.iter().enumerate() {
        println!("Output tensor {}: {}", i, result);
    }

    Ok(())
}

pub struct ConcreteSigmoid;

// Basic sigmoid example with a concrete type
impl Module<1, 1> for ConcreteSigmoid {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![3, 1, 4].into_iter().map(NatExpr::Constant).collect()),
        }));
        ([ty.clone()], [ty])
    }

    fn path(&self) -> Path {
        path(vec!["test", "sigmoid"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let sh = ops::shape(builder, x.clone());
        let one = ops::constant(builder, 1.0, &sh);
        let one = ops::cast(builder, one, ops::dtype(builder, x.clone()));

        [one.clone() / (one + nn::Exp.call(builder, [-x]))]
    }
}

pub struct ConcreteSigmoidPlusConst;

// Sigmoid plus constant parameter example
impl Module<1, 1> for ConcreteSigmoidPlusConst {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let ty = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![3, 1, 4].into_iter().map(NatExpr::Constant).collect()),
        }));
        ([ty.clone()], [ty])
    }

    fn path(&self) -> Path {
        path(vec!["test", "sigmoid_plus_const"]).unwrap()
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        let sh = ops::shape(builder, x.clone());
        let one = ops::constant(builder, 1.0, &sh);
        let one = ops::cast(builder, one, ops::dtype(builder, x.clone()));

        // Compute sigmoid
        let sigmoid = one.clone() / (one + nn::Exp.call(builder, [-x]));

        // Get the scalar parameter and broadcast it to match input shape
        let offset = ops::param(builder, &self.path().extend(["offset"]).unwrap());
        let offset_broadcasted = ops::broadcast(builder, offset, sh);

        // Add the broadcasted constant to sigmoid
        [sigmoid + offset_broadcasted]
    }
}
