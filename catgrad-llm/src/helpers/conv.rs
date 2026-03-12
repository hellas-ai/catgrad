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
