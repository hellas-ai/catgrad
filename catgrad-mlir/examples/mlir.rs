use catgrad::prelude::*;
use catgrad::typecheck::*;
use catgrad_mlir::lower;

/// Construct, shapecheck, and lower an `Exp` function to MLIR
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = ConcreteSigmoid;

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");

    // Create parameters for the model
    let parameters = typecheck::Parameters::from([]);

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.definitions.extend([(model.path(), typed_term)]);
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Convert model to MLIR
    let (_, term) = lower::preprocess(&env, &parameters, model.path());
    let mlir = lower::term_to_func(&model.path().to_string(), term);

    // print MLIR
    println!("{}", mlir);

    Ok(())
}

pub struct ConcreteSigmoid;

// Redo the sigmoid example with a concrete type
impl Module<1, 1> for ConcreteSigmoid {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        // TODO: try replacing shape with ShapeExpr::Var(0)
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
