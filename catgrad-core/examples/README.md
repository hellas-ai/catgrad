# Examples

This directory contains example programs demonstrating how to use the catgrad library.


<br>

## hidden.rs

The `hidden.rs` example demonstrates a simple neural network model for MNIST digit classification. It shows how to:

- Define a neural network model using catgrad's DSL
- Perform shape checking and type inference
- Run inference using different backends
- Generate SVG diagrams of the computation graph

<br>

### Running the Example

The example requires one of the available backends to be enabled.

#### Using the ndarray backend for tensor operations:
```bash
cargo run --example hidden --features ndarray-backend
```

#### Using the Candle backend for tensor operations:
```bash
cargo run --example hidden --features candle-backend
```

#### With SVG diagram generation showing the computation graph:
```bash
cargo run --example hidden --features ndarray-backend,svg
```

or

```bash
cargo run --example hidden --features candle-backend,svg
```

<br>

### What the Example Does

1. **Model Definition**: Creates a simple 2-layer neural network:
   - Input: 28×28 MNIST-like images (flattened to 784 dimensions)
   - Layer 1: 784 → 100 neurons with sigmoid activation
   - Layer 2: 100 → 10 neurons with sigmoid activation (output classes)

2. **Shape Checking**: Performs static type checking to verify tensor shapes and operations

3. **Inference**: Runs the model on sample input data and displays the output shape and sample values

4. **Visualization**: Generates SVG diagrams of the computation graph (when `svg` feature is enabled)

<br>

### Output

The example will:
- Print the output tensor shape and sample values
- Generate SVG files in `examples/images/` directory (if `svg` feature is enabled):
  - `model.hidden.svg`: Basic computation graph
  - `model.hidden_typed.svg`: Graph with inferred types and shapes

<br>

### Notes

- The example uses synthetic parameter data for demonstration purposes
- In a real application, you would load actual model weights from a file (e.g., safetensors format)
- The model processes a batch of 2 sample images for demonstration
