use crate::helpers::tensors::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

// Depthwise 1D convolution without bias
pub fn depthwise_conv1d_no_bias(
    builder: &Builder,
    weight_path: Path,
    kernel_size: usize,
    x: Var,
    padding_size: usize,
) -> Var {
    let conv_weight = param(builder, &weight_path.extend(["weight"]).unwrap());
    depthwise_conv1d_no_bias_param(builder, conv_weight, kernel_size, padding_size, x)
}

// Depthwise 1D convolution without bias for already padded inputs.
// `x_padded` is expected to be padded with `K-1` leading zeros; `output_len` is the desired
// unpadded sequence length. This variant is required because we cannot subtract Nats to get the
// unpadded sequence length directly.
pub fn padded_depthwise_conv1d_no_bias(
    builder: &Builder,
    weight_path: Path,
    kernel_size: usize,
    x_padded: Var,
    output_len: Var,
) -> Var {
    let conv_weight = param(builder, &weight_path.extend(["weight"]).unwrap());
    padded_depthwise_conv1d_no_bias_param(builder, conv_weight, kernel_size, x_padded, output_len)
}

// Parameterized depthwise 1D convolution with optional bias.
// `conv_weight` is expected to have shape `H x 1 x K`.
// `conv_bias` (optional) is expected to have shape `H`.
pub fn depthwise_conv1d_param(
    builder: &Builder,
    conv_weight: Var,
    conv_bias: Option<Var>,
    kernel_size: usize,
    x: Var,
    padding_size: usize,
) -> Var {
    let [b, h, s] = unpack::<3>(builder, shape(builder, x.clone()));

    let x_padded = if padding_size > 0 {
        let pad_shape = shape!(builder, b, h, padding_size);
        let pad = constant(builder, 0.0, &pad_shape);
        concat(builder, 2, pad, x)
    } else {
        x
    };

    depthwise_conv1d_param_padded(builder, conv_weight, conv_bias, kernel_size, x_padded, s)
}

// Parameterized depthwise 1D convolution without bias.
// `conv_weight` is expected to have shape `H x 1 x K`.
pub fn depthwise_conv1d_no_bias_param(
    builder: &Builder,
    conv_weight: Var,
    kernel_size: usize,
    padding_size: usize,
    x: Var,
) -> Var {
    depthwise_conv1d_param(builder, conv_weight, None, kernel_size, x, padding_size)
}

// Parameterized depthwise 1D convolution without bias for already padded inputs.
pub fn padded_depthwise_conv1d_no_bias_param(
    builder: &Builder,
    conv_weight: Var,
    kernel_size: usize,
    x_padded: Var,
    output_len: Var,
) -> Var {
    depthwise_conv1d_param_padded(
        builder,
        conv_weight,
        None,
        kernel_size,
        x_padded,
        output_len,
    )
}

// Helper function for depthwise 1D convolution with bias for already padded inputs.
fn depthwise_conv1d_param_padded(
    builder: &Builder,
    conv_weight: Var,
    conv_bias: Option<Var>,
    kernel_size: usize,
    x_padded: Var,
    output_len: Var,
) -> Var {
    let conv_weight = squeeze::<3, 2>(builder, 1, conv_weight);

    let mut conv_out: Option<Var> = None;
    for offset in 0..kernel_size {
        let x_slice = slice(builder, 2, offset, output_len.clone(), x_padded.clone());
        let w_slice = slice(builder, 1, offset, 1, conv_weight.clone());
        let w_slice = unsqueeze::<2, 3>(builder, 0, w_slice);
        let w_slice = broadcast(builder, shape(builder, x_slice.clone()), w_slice);
        let term = x_slice * w_slice;
        conv_out = Some(match conv_out {
            Some(acc) => acc + term,
            None => term,
        });
    }
    let mut conv_out = conv_out.expect("kernel_size must be positive");

    if let Some(bias) = conv_bias {
        let bias = unsqueeze::<1, 2>(builder, 0, bias);
        let bias = unsqueeze::<2, 3>(builder, 2, bias);
        let bias = broadcast(builder, shape(builder, conv_out.clone()), bias);
        conv_out = conv_out + bias;
    }

    conv_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use catgrad::abstract_interpreter::Value as TypeValue;
    use catgrad::category::core::Shape;
    use catgrad::interpreter::backend::Backend;
    use catgrad::interpreter::backend::ndarray::NdArrayBackend;
    use catgrad::interpreter::{
        Interpreter, Parameters, TaggedTensor, TaggedTensorTuple, Value, tensor,
    };
    use catgrad::stdlib::{Module, stdlib};
    use catgrad::typecheck::value_types::*;

    struct DepthwiseConv1dTest;

    impl Module<2, 1> for DepthwiseConv1dTest {
        fn ty(&self) -> ([Type; 2], [Type; 1]) {
            let t_x = TypeValue::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![
                    NatExpr::Constant(1),
                    NatExpr::Constant(2),
                    NatExpr::Constant(4),
                ]),
            }));
            let t_w = TypeValue::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![
                    NatExpr::Constant(2),
                    NatExpr::Constant(1),
                    NatExpr::Constant(3),
                ]),
            }));
            let t_y = TypeValue::Tensor(TypeExpr::NdArrayType(NdArrayType {
                dtype: DtypeExpr::Constant(Dtype::F32),
                shape: ShapeExpr::Shape(vec![
                    NatExpr::Constant(1),
                    NatExpr::Constant(2),
                    NatExpr::Constant(4),
                ]),
            }));
            ([t_x, t_w], [t_y])
        }

        fn path(&self) -> Path {
            path(vec!["test", "depthwise_conv1d"]).unwrap()
        }

        fn def(&self, builder: &Builder, [x, w]: [Var; 2]) -> [Var; 1] {
            [depthwise_conv1d_no_bias_param(builder, w, 3, 2, x)]
        }
    }

    #[test]
    fn test_depthwise_conv1d_no_bias_param_matches_reference_values() {
        let typed_term = DepthwiseConv1dTest.term().unwrap();
        let backend = NdArrayBackend;
        let interpreter = Interpreter::new(backend, stdlib(), Parameters::default());

        let x = tensor(
            &interpreter.backend,
            Shape(vec![1, 2, 4]),
            vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let w = tensor(
            &interpreter.backend,
            Shape(vec![2, 1, 3]),
            vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 2.0],
        )
        .unwrap();

        let mut outputs = interpreter.run(typed_term.term, vec![x, w]).unwrap();
        let y = outputs.pop().expect("missing output");

        let expected = tensor(
            &interpreter.backend,
            Shape(vec![1, 2, 4]),
            vec![3.0f32, 8.0, 14.0, 20.0, 20.0, 30.0, 45.0, 60.0],
        )
        .unwrap();

        match (y, expected) {
            (
                Value::Tensor(TaggedTensor::F32([actual])),
                Value::Tensor(TaggedTensor::F32([exp])),
            ) => {
                assert!(
                    interpreter
                        .backend
                        .compare(TaggedTensorTuple::F32([actual, exp])),
                    "depthwise conv output should match expected reference values"
                );
            }
            _ => panic!("expected f32 tensor outputs"),
        }
    }
}
