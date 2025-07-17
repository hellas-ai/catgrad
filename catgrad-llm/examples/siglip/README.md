# SigLIP Example

This example demonstrates zero-shot image classification using the SigLIP model.
It takes an image and a list of text labels as input, and outputs the probability of each label applying to the image.

## Usage

You need to provide a path to an image and at least one label.

```bash
cargo run --release --example siglip -- -i cats.jpg -l "two cats" "a dog" "a bicycle race"
```

The default model is `google/siglip-base-patch16-224`. You can specify a different SigLIP model from the Hub using the `-m` flag
such as `google/siglip2-base-patch16-224`.

## Known problems

* The image size is hardcoded for now to 224x224 so models with names not ending in -224
will not work correctly. For larger images the positional embeddings need to be interpolated.

* `config.json` files for image models are incomplete on HF Hub so many model specific values need to be set in the app.
