use catgrad_core::category::core::Dtype;
use catgrad_core::category::core::Shape;
use catgrad_core::check::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, TypeExpr, Value};
use catgrad_core::interpreter::backend::ndarray::NdArrayBackend;
use catgrad_core::prelude::*;

use catgrad_core::interpreter;

use std::collections::HashMap;

use catgrad_llm::models::utils::Config;
use catgrad_llm::utils::get_model_files;

use anyhow::Result;

/// Construct, shapecheck, and interpret the `GPT2Model` using the ndarray backend.
fn main() -> Result<()> {
    // Create parameters for the model
    let backend = NdArrayBackend;
    let (interpreter_params, parameters, config) = load_model("openai-community/gpt2", &backend)?;

    let model = GPT2Model { config };

    // Get the model as a typed term
    let typed_term = model.term().expect("Failed to create typed term");
    save_svg(&typed_term.term, &format!("{}.svg", model.path()))?;

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameters.keys()));

    // Shapecheck the model
    let check_result = check::check(&env, &parameters, typed_term.clone())
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    // Diagram of term with shapes inferred
    let labeled_term = typed_term.term.clone().with_nodes(|_| check_result);
    let filename = &format!("{}_typed.svg", model.path());
    save_svg(&labeled_term.unwrap(), filename)?;

    // Run interpreter
    run_interpreter(&typed_term, env, interpreter_params)?;

    Ok(())
}

fn run_interpreter(
    typed_term: &TypedTerm,
    env: Environment,
    interpreter_params: interpreter::Parameters<NdArrayBackend>,
) -> Result<()> {
    let backend = NdArrayBackend;

    // Create interpreter
    let interpreter = interpreter::Interpreter::new(backend, env, interpreter_params);

    // GPT-2 encoding for 'Category theory is'
    let input_data = [27313, 4583, 318];
    let input_tensor = interpreter::tensor(&interpreter.backend, Shape(vec![1, 3]), &input_data)
        .expect("Failed to create input tensor");

    // Run the model
    let results = interpreter
        .run(typed_term.term.clone(), vec![input_tensor])
        .expect("Failed to run inference");

    // Print info about the main output (should be the last one)
    if let Some(output) = results.last() {
        use catgrad_core::interpreter::{TaggedNdArray, Value};
        match output {
            Value::NdArray(TaggedNdArray::U32([arr])) => {
                println!("Output shape: {:?}", arr.shape());
                println!(
                    "Output sample: {:?}",
                    &arr.as_slice().unwrap()[..10.min(arr.len())]
                );
            }
            _ => println!("Unexpected output type: {:?}", output),
        }
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////
// Define the GPT2Model model

pub struct GPT2Model {
    config: Config,
}

impl GPT2Model {
    pub fn info(&self) {
        println!("Config: {:#?}", self.config);
    }

    fn layernorm_raw(&self, builder: &Builder, eps: f32, x: Var) -> Var {
        let x_shape = shape(builder, x.clone());
        let [_, _, n] = unpack::<3>(builder, x_shape.clone());
        let s = sum(builder, x.clone());

        let constn = scalar(builder, n);
        let constn = cast(builder, constn, dtype(builder, x.clone()));
        let sh = shape(builder, s.clone());
        let constn = broadcast_to(builder, constn, sh);

        let mean = s / constn.clone();
        let nom = x - broadcast_to(builder, mean, x_shape.clone());

        let var = sum(builder, nom.clone() * nom.clone()) / constn;
        let epsilon = constant_f32(builder, eps);
        let sh = shape(builder, var.clone());
        let epsilon = broadcast_to(builder, epsilon, sh);
        let stddev = nn::sqrt(builder, var + epsilon);
        let denom = broadcast_to(builder, stddev, x_shape);

        nom / denom
    }

    pub fn layernorm(&self, builder: &Builder, eps: f32, p: Path, x: Var) -> Var {
        let gamma = param(
            builder,
            &p.concat(&path(vec!["weight"]).expect("invalid param path")),
        );
        let lr = self.layernorm_raw(builder, eps, x);
        let lr_shape = shape(builder, lr.clone());
        let gamma = broadcast_to(builder, gamma, lr_shape.clone());
        let lr = lr * gamma;

        let beta = param(
            builder,
            &p.concat(&path(vec!["bias"]).expect("invalid param path")),
        );
        let beta = broadcast_to(builder, beta, lr_shape);
        lr + beta
    }

    pub fn embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let wte = param(
            builder,
            &p.concat(&path(vec!["wte", "weight"]).expect("invalid param path")),
        );
        let dim = constant_nat(builder, 0);
        let te = index(builder, wte, dim.clone(), x);

        // add back batch size dim
        let sh = shape(builder, te.clone());
        let [seq_len, hidden_dim] = unpack::<2>(builder, sh);
        let batch_size = constant_nat(builder, 1);
        let sh = pack::<3>(builder, [batch_size, seq_len.clone(), hidden_dim]);

        let te = reshape(builder, sh.clone(), te);

        let wpe = param(
            builder,
            &p.concat(&path(vec!["wpe", "weight"]).expect("invalid param path")),
        );

        let r = arange(builder, seq_len);
        let pe = index(builder, wpe, dim, r);
        let pe = reshape(builder, sh, pe);
        te + pe
    }

    // fn gpt_linear(builder: &Builder, in_dim: usize, out_dim: usize, p: Path, x: Var) -> Var {
    //     let w = param(
    //         builder,
    //         &p.concat(&path(vec!["weight"]).expect("invalid param path")),
    //     );
    //     let b = param(
    //         builder,
    //         &p.concat(&path(vec!["bias"]).expect("invalid param path")),
    //     );

    //     // w is already transposed in GPT-2 checkpoints
    //     let mut w_t = w;

    //     // if x.label.shape.0.len() == 3 {
    //     //     let batch_size = x.label.shape.0[0];
    //     //     w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
    //     // }

    //     let m = matmul(builder, x, w_t);
    //     // let bb = broadcast(builder, m.label.shape.clone(), b);
    //     // m +bb
    //     m
    // }
}

// Implement `Def`: this is like torch's `Module`.
impl Module<1, 1> for GPT2Model {
    fn path(&self) -> Path {
        path(vec!["gpt2"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, [x]: [Var; 1]) -> [Var; 1] {
        // let [_batch_size, _seq] = unpack::<2>(builder, shape(builder, x.clone()));

        let root = self.path();

        // self.info();

        let x = self.embeddings(builder, root, x);

        // let _ln_f = self.layernorm(
        //     builder,
        //     self.config.layer_norm_epsilon,
        //     root.concat(&path(vec!["ln_f"]).expect("invalid param path")),
        //     x.clone(),
        // );
        // let _ln_f = self.layernorm_raw(builder, self.config.layer_norm_epsilon, x.clone());
        [x]
    }

    // This should return the *detailed* type of the model
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        use catgrad_core::check::*;

        let batch_size = NatExpr::Var(0);
        let seq_len = NatExpr::Var(1);

        // Input shape B×S
        let t_x = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len]),
        }));

        // Output shape B×1
        let t_y = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::U32),
            shape: ShapeExpr::Shape(vec![batch_size, NatExpr::Constant(1)]),
        }));

        ([t_x], [t_y])
    }
}

fn load_model<B: interpreter::Backend>(
    model_name: &str,
    backend: &B,
) -> Result<(interpreter::Parameters<B>, check::Parameters, Config)> {
    let (model_paths, config_path, _tokenizer_path, _) = get_model_files(model_name, "main")?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let file = std::fs::File::open(&model_paths[0])?;
    let data = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    // Read each tensor
    let mut type_map = HashMap::new();
    let mut data_map = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();
        let tensor_data = view.data();

        // Convert dtype and load tensor data
        match view.dtype() {
            safetensors::Dtype::F32 => {
                let data: Vec<f32> = tensor_data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let tensor =
                    interpreter::TaggedNdArray::from_slice(backend, &data, Shape(shape.clone()))
                        .expect("Failed to create tensor");
                let key = path(name.split(".").collect()).expect("invalid param path");
                data_map.insert(key.clone(), tensor);

                let vne = shape.into_iter().map(NatExpr::Constant).collect();
                let tensor_type = Value::Tensor(TypeExpr::NdArrayType(NdArrayType {
                    dtype: DtypeExpr::Constant(Dtype::F32),
                    shape: ShapeExpr::Shape(vne),
                }));
                type_map.insert(key, tensor_type);
            }
            // Add other dtype conversions as needed
            _ => {
                panic!("Unsupported dtype: {:?}", view.dtype());
            }
        }
    }
    Ok((
        interpreter::Parameters::from(data_map),
        check::Parameters::from(type_map),
        config,
    ))
}

pub fn save_svg<
    O: PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
    A: PartialEq + Clone + std::fmt::Display + std::fmt::Debug,
>(
    term: &open_hypergraphs::lax::OpenHypergraph<O, A>,
    filename: &str,
) -> Result<()> {
    use catgrad_core::svg::to_svg;
    let bytes = to_svg(term)?;
    let output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("images");
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join(filename);
    println!("saving svg to {output_path:?}");
    std::fs::write(output_path, bytes).expect("write diagram file");
    Ok(())
}
