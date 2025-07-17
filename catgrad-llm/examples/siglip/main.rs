use catgrad::backend::cpu::eval::{Builder, EvalState};
use catgrad::backend::cpu::ndarray::{NdArray, TaggedNdArray};
use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Shape, Var};
use catgrad_llm::utils::{get_model_files, read_safetensors_multiple};
use clap::Parser;
use image::imageops::FilterType;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    /// Model name on Huggingface Hub
    #[arg(short = 'm', long, default_value = "google/siglip-base-patch16-224")]
    model_name: String,

    /// Path to an image file
    #[arg(short = 'i', long)]
    image: PathBuf,

    /// Labels to classify the image against
    #[arg(short = 'l', long, num_args = 1.., required = true)]
    labels: Vec<String>,
}

fn default_hidden_size() -> usize {
    768
}

fn default_intermediate_size() -> usize {
    3072
}

fn default_num_hidden_layers() -> usize {
    12
}

fn default_num_attention_heads() -> usize {
    12
}

fn default_layer_norm_eps() -> f32 {
    1e-6
}

fn default_max_position_embeddings() -> usize {
    64
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TransformerConfig {
    #[serde(default = "default_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TextConfig {
    #[serde(flatten)]
    transformer: TransformerConfig,
    vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VisionConfig {
    #[serde(flatten)]
    transformer: TransformerConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SiglipConfig {
    text_config: TextConfig,
    vision_config: VisionConfig,
}

fn attention(builder: &Builder, config: &TransformerConfig, name: &str, x: Var) -> Var {
    let dim = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let b = x.label.shape.0[0];
    let s = x.label.shape.0[1];

    let q = linear(builder, dim, dim, &format!("{name}.q_proj"), x.clone());
    let k = linear(builder, dim, dim, &format!("{name}.k_proj"), x.clone());
    let v = linear(builder, dim, dim, &format!("{name}.v_proj"), x);

    let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
    let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
    let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

    let q = transpose(builder, 1, 2, q);
    let k = transpose(builder, 1, 2, k);
    let v = transpose(builder, 1, 2, v);

    let tk = transpose(builder, 2, 3, k);
    let attn = mat_mul(builder, q, tk);
    let denom = constant(builder, attn.label.clone(), (head_dim as f32).sqrt());
    let attn = attn / denom;

    let attn = softmax(builder, attn);
    let attn = mat_mul(builder, attn, v);
    let x = transpose(builder, 1, 2, attn);
    let x = reshape(builder, Shape(vec![b, s, dim]), x);

    linear(builder, dim, dim, &format!("{name}.out_proj"), x)
}

fn mlp(builder: &Builder, config: &TransformerConfig, name: &str, x: Var) -> Var {
    let x = linear(
        builder,
        config.hidden_size,
        config.intermediate_size,
        &format!("{name}.fc1"),
        x,
    );
    let x = gelu(builder, x);
    linear(
        builder,
        config.intermediate_size,
        config.hidden_size,
        &format!("{name}.fc2"),
        x,
    )
}

fn encoder_layer(builder: &Builder, config: &TransformerConfig, name: &str, x: Var) -> Var {
    let res = x.clone();
    let x = layernorm(
        builder,
        config.layer_norm_eps,
        &format!("{name}.layer_norm1"),
        x,
    );
    let x = attention(builder, config, &format!("{name}.self_attn"), x);
    let x = x + res;

    let res = x.clone();
    let x = layernorm(
        builder,
        config.layer_norm_eps,
        &format!("{name}.layer_norm2"),
        x,
    );
    let x = mlp(builder, config, &format!("{name}.mlp"), x);
    x + res
}

// Instead of using a convolutional layer to extract patches, we use a linear layer.
fn vision_embeddings(builder: &Builder, config: &VisionConfig, name: &str, x: Var) -> Var {
    let patch_size = 16;
    let image_size = 224;
    let num_patches = (image_size / patch_size) * (image_size / patch_size);
    let num_channels = 3;
    let hidden_size = config.transformer.hidden_size;

    // Patch embeddings from convolution
    // The weight is for a conv2d, so we need to reshape it for matmul.
    let patch_weight_t = NdArrayType::new(
        Shape(vec![hidden_size, num_channels, patch_size, patch_size]),
        Dtype::F32,
    );
    let patch_weight = parameter(
        builder,
        patch_weight_t,
        format!("{name}.patch_embedding.weight"),
    );

    let patch_weight_permuted = transpose(builder, 1, 2, patch_weight);
    let patch_weight_permuted = transpose(builder, 2, 3, patch_weight_permuted);

    let patch_weight_flat = reshape(
        builder,
        Shape(vec![hidden_size, num_channels * patch_size * patch_size]),
        patch_weight_permuted,
    );

    let patch_weight_flat_t = transpose(builder, 0, 1, patch_weight_flat);
    let patch_weight_flat_t = expand(builder, Shape(vec![1, 768, 768]), patch_weight_flat_t);
    let mut patch_embeddings = mat_mul(builder, x, patch_weight_flat_t);

    let bias_t = NdArrayType::new(Shape(vec![hidden_size]), Dtype::F32);
    let bias = parameter(builder, bias_t, format!("{name}.patch_embedding.bias"));
    let bias_expanded = expand(builder, patch_embeddings.label.shape.clone(), bias);
    patch_embeddings = patch_embeddings + bias_expanded;

    // Position embeddings
    let t = NdArrayType::new(
        Shape(vec![num_patches, config.transformer.hidden_size]),
        Dtype::F32,
    );
    let weights = parameter(builder, t, format!("{name}.position_embedding.weight"));
    let pe = expand(
        builder,
        Shape(vec![1, num_patches, config.transformer.hidden_size]),
        weights,
    );

    patch_embeddings + pe
}

// Build linear layer from weight and bias parameters
fn linear_wb(
    builder: &Builder,
    in_dim: usize,
    out_dim: usize,
    weight: Var,
    bias: Var,
    x: Var,
) -> Var {
    let mut w_t = transpose(builder, 0, 1, weight);

    if x.label.shape.0.len() == 3 {
        let batch_size = x.label.shape.0[0];
        w_t = expand(builder, Shape(vec![batch_size, in_dim, out_dim]), w_t);
    }

    let m = mat_mul(builder, x, w_t);
    let bias = expand(builder, m.label.shape.clone(), bias);
    m + bias
}

fn vision_head_attention(
    builder: &Builder,
    config: &TransformerConfig,
    name: &str,
    probe: Var,
    x: Var,
) -> Var {
    let dim = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let b = x.label.shape.0[0];
    let s = x.label.shape.0[1];

    let w_type = NdArrayType::new(Shape(vec![3 * dim, dim]), x.label.dtype);
    let weight = parameter(builder, w_type, format!("{name}.in_proj_weight"));
    let b_type = NdArrayType::new(Shape(vec![3 * dim]), x.label.dtype);
    let bias = parameter(builder, b_type, format!("{name}.in_proj_bias"));

    let ws = split(builder, 0, 3, weight);
    let bs = split(builder, 0, 3, bias);

    let q = linear_wb(builder, dim, dim, ws[0].clone(), bs[0].clone(), probe);
    let k = linear_wb(builder, dim, dim, ws[1].clone(), bs[1].clone(), x.clone());
    let v = linear_wb(builder, dim, dim, ws[2].clone(), bs[2].clone(), x);

    let q = reshape(builder, Shape(vec![b, 1, num_heads, head_dim]), q);
    let k = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), k);
    let v = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), v);

    let q = transpose(builder, 1, 2, q);
    let k = transpose(builder, 1, 2, k);
    let v = transpose(builder, 1, 2, v);

    let tk = transpose(builder, 2, 3, k);
    let attn = mat_mul(builder, q, tk);
    let denom = constant(builder, attn.label.clone(), (head_dim as f32).sqrt());
    let attn = attn / denom;

    let attn = softmax(builder, attn);
    let attn = mat_mul(builder, attn, v);
    let x = transpose(builder, 1, 2, attn);
    let x = reshape(builder, Shape(vec![b, 1, dim]), x);

    linear(builder, dim, dim, &format!("{name}.out_proj"), x)
}

fn vision_head(builder: &Builder, config: &VisionConfig, name: &str, x: Var) -> Var {
    let config = &config.transformer;
    let probe_type = NdArrayType::new(Shape(vec![1, 1, config.hidden_size]), Dtype::F32);
    let probe = parameter(builder, probe_type, format!("{name}.probe"));
    let x = vision_head_attention(builder, config, &format!("{name}.attention"), probe, x);
    let res = x.clone();
    let x = layernorm(
        builder,
        config.layer_norm_eps,
        &format!("{name}.layernorm"),
        x,
    );

    let x = mlp(builder, config, &format!("{name}.mlp"), x);
    let res = x + res;
    narrow(builder, 1, 0, 1, res)
}

fn vision_model(builder: &Builder, config: &VisionConfig, x: Var) -> Var {
    let mut x = vision_embeddings(builder, config, "vision_model.embeddings", x);
    let vconfig = config;
    let config = &config.transformer;
    for i in 0..config.num_hidden_layers {
        x = encoder_layer(
            builder,
            config,
            &format!("vision_model.encoder.layers.{i}"),
            x,
        );
    }
    let x = layernorm(
        builder,
        config.layer_norm_eps,
        "vision_model.post_layernorm",
        x,
    );

    vision_head(builder, vconfig, "vision_model.head", x)
}

fn text_embeddings(builder: &Builder, config: &TextConfig, name: &str, x: Var) -> Var {
    let t = NdArrayType::new(
        Shape(vec![config.vocab_size, config.transformer.hidden_size]),
        Dtype::F32,
    );
    let weights = parameter(builder, t, format!("{name}.token_embedding.weight"));
    let we = embedding(builder, x.clone(), weights);

    let t = NdArrayType::new(
        Shape(vec![
            config.max_position_embeddings,
            config.transformer.hidden_size,
        ]),
        Dtype::F32,
    );
    let pos = arange(builder, x.label.shape.0[1], Dtype::I32);
    let weights = parameter(builder, t, format!("{name}.position_embedding.weight"));
    let pe = embedding(builder, pos, weights);
    let pe = expand(builder, we.label.shape.clone(), pe);

    we + pe
}

fn text_model(builder: &Builder, config: &TextConfig, x: Var) -> Var {
    let mut x = text_embeddings(builder, config, "text_model.embeddings", x);
    let config = &config.transformer;
    for i in 0..config.num_hidden_layers {
        x = encoder_layer(
            builder,
            config,
            &format!("text_model.encoder.layers.{i}"),
            x,
        );
    }
    let x = layernorm(
        builder,
        config.layer_norm_eps,
        "text_model.final_layer_norm",
        x,
    );
    let x = narrow(builder, 1, 63, 1, x);

    linear(
        builder,
        config.hidden_size,
        config.hidden_size,
        "text_model.head",
        x,
    )
}

fn div_l2_norm(builder: &Builder, x: Var) -> Var {
    let sqr = x.clone() * x.clone();
    let l2_norm = sum(builder, sqr);
    let l2_norm = sqrt(builder, l2_norm);
    let l2_norm = expand(builder, x.label.shape.clone(), l2_norm);
    x / l2_norm
}

fn siglip(builder: &Builder, config: &SiglipConfig, text: Var, image: Var) -> (Var, Var) {
    let scalar_type = NdArrayType::new(Shape(vec![1]), Dtype::F32);

    let dim = config.vision_config.transformer.hidden_size;
    let n_text = text.label.shape.0[0];
    let n_img = image.label.shape.0[0];
    let text_features = text_model(builder, &config.text_config, text);
    let text_features = div_l2_norm(builder, text_features);
    let text_features = reshape(builder, Shape(vec![n_text, dim]), text_features);

    let image_features = vision_model(builder, &config.vision_config, image);
    let image_features = div_l2_norm(builder, image_features);
    let image_features = reshape(builder, Shape(vec![n_img, dim]), image_features);

    let rt = transpose(builder, 0, 1, image_features);
    let logits_per_text = mat_mul(builder, text_features, rt);

    let logit_scale = parameter(builder, scalar_type.clone(), "logit_scale".to_string());
    let logit_bias = parameter(builder, scalar_type, "logit_bias".to_string());
    let logit_scale = exp(builder, logit_scale);

    let logit_scale = expand(builder, logits_per_text.label.shape.clone(), logit_scale);
    let logit_bias = expand(builder, logits_per_text.label.shape.clone(), logit_bias);
    let logits_per_text = logits_per_text * logit_scale + logit_bias;
    let logits_per_image = transpose(builder, 0, 1, logits_per_text.clone());
    (logits_per_text, logits_per_image)
}

struct ModelRunner {
    pub state: Option<EvalState>,
    pub tensors: Rc<HashMap<String, TaggedNdArray>>,
    pub config: SiglipConfig,
}

impl ModelRunner {
    fn new(config: SiglipConfig) -> Self {
        Self {
            tensors: Rc::new(HashMap::new()),
            state: None,
            config,
        }
    }

    fn load(&mut self, model_paths: Vec<PathBuf>) {
        let tensors = read_safetensors_multiple(model_paths);
        self.tensors = Rc::new(tensors);
    }

    fn build_graph(&mut self, batches: usize, num_tokens: usize) {
        let text_in_type = NdArrayType::new(Shape(vec![batches, num_tokens]), Dtype::I32);
        let image_in_type = NdArrayType::new(Shape(vec![1, 196, 768]), Dtype::F32);

        let state = EvalState::build(|builder| {
            let text = Var::new(builder.clone(), text_in_type.clone());
            let image = Var::new(builder.clone(), image_in_type.clone());
            let (_logits_per_text, logits_per_image) =
                siglip(builder, &self.config, text.clone(), image.clone());

            // Get the probabilities of each label matching this image
            let probs = softmax(builder, logits_per_image);
            let sources_vec = vec![text, image];

            let targets_vec = vec![probs];
            (sources_vec, targets_vec)
        });

        self.state = Some(state);
        self.state
            .as_mut()
            .unwrap()
            .set_parameters(Rc::clone(&self.tensors));
    }

    fn run(&mut self, text: &NdArray<i32>, image: &NdArray<f32>) -> Vec<&TaggedNdArray> {
        let sources = vec![text.clone().into(), image.clone().into()];

        self.state.as_mut().unwrap().eval_with(sources)
    }
}

// Loads the image and returns it as a sequence of flattened 16x16 patches
// corresponding to the image tokens going into the model
fn load_and_preprocess_image(image_path: &PathBuf) -> NdArray<f32> {
    let image_size = 224;
    let patch_size = 16;
    let num_patches: usize = (image_size / patch_size) * (image_size / patch_size);
    let num_channels = 3;

    let img = image::open(image_path).unwrap();
    let resized_img =
        img.resize_to_fill(image_size as u32, image_size as u32, FilterType::Triangle);
    let rgb_img = resized_img.to_rgb8();
    let img = rgb_img.into_raw();

    let pixels: Vec<f32> = img.iter().map(|&x| x as f32 * (2. / 255.0) - 1.).collect();
    let mut patches = vec![0.0; num_patches * patch_size * patch_size * num_channels];
    for i in 0..num_patches {
        let row = (i / (image_size / patch_size)) * patch_size;
        let col = (i % (image_size / patch_size)) * patch_size;
        for r in 0..patch_size {
            for c in 0..patch_size {
                for ch in 0..num_channels {
                    patches[i * patch_size * patch_size * num_channels
                        + (r * patch_size + c) * num_channels
                        + ch] = pixels[((row + r) * image_size + col + c) * num_channels + ch];
                }
            }
        }
    }
    NdArray::new(
        patches,
        Shape(vec![1, num_patches, patch_size * patch_size * num_channels]),
    )
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    let (model_paths, config_path, tokenizer_path, _) = get_model_files(&args.model_name);

    let config: SiglipConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let mut runner = ModelRunner::new(config.clone());
    runner.load(model_paths);
    println!("SigLIP model {} loaded successfully.", args.model_name);

    let image_tensor = load_and_preprocess_image(&args.image);

    let labels = args.labels;
    let mut encodings = vec![];

    let max_len = config.text_config.max_position_embeddings;
    let pad_token_id = 1;

    for label in labels.clone() {
        let e = tokenizer
            .encode(label, true)
            .map_err(|e| format!("Failed to encode labels: {e}"))?;
        let token_ids: Vec<i32> = e.get_ids().iter().map(|&x| x as i32).collect();
        encodings.extend(&token_ids);
        encodings.extend(vec![pad_token_id; max_len - token_ids.len()]);
    }

    let batches = encodings.len() / max_len;
    let input = NdArray::new(encodings, Shape(vec![batches, max_len]));
    runner.build_graph(batches, max_len);
    let result = runner.run(&input, &image_tensor);

    for (i, label) in labels.iter().enumerate() {
        println!("{label}: {:.4}%", result[0].get(&[0, i]) * 100.);
    }
    Ok(())
}
