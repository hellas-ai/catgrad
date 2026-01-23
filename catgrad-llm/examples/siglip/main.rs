use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::{self, Backend, Interpreter};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;
use catgrad::typecheck::TypeExpr;
use catgrad_llm::helpers::*;
use catgrad_llm::utils::get_model_files;
use clap::Parser;
use image::imageops::FilterType;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
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

fn default_patch_size() -> usize {
    16
}

fn default_image_size() -> usize {
    224
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
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VisionConfig {
    #[serde(flatten)]
    transformer: TransformerConfig,
    #[serde(default = "default_patch_size")]
    patch_size: usize,
    #[serde(default = "default_image_size")]
    image_size: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SiglipConfig {
    text_config: TextConfig,
    vision_config: VisionConfig,
}

struct SiglipVisionBackbone {
    config: VisionConfig,
}

struct SiglipModel {
    config: SiglipConfig,
    vision_backbone: SiglipVisionBackbone,
}

impl SiglipVisionBackbone {
    fn attention(&self, builder: &Builder, config: &TransformerConfig, p: Path, x: Var) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());
        let k = linear(builder, dim, dim, p.extend(["k_proj"]).unwrap(), x.clone());
        let v = linear(builder, dim, dim, p.extend(["v_proj"]).unwrap(), x);

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh.clone(), q);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);

        let head_dim_float = head_dim as f32;
        let sh_attn = shape(builder, attn.clone());
        let denom = constant(builder, head_dim_float.sqrt(), &sh_attn);
        let attn = attn / denom;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let x = reshape(builder, sh, x);

        linear(builder, dim, dim, p.extend(["out_proj"]).unwrap(), x)
    }
    fn mlp(&self, builder: &Builder, config: &TransformerConfig, p: Path, x: Var) -> Var {
        let x = linear(
            builder,
            config.hidden_size,
            config.intermediate_size,
            p.extend(["fc1"]).unwrap(),
            x,
        );
        let x = gelu(builder, x);
        linear(
            builder,
            config.intermediate_size,
            config.hidden_size,
            p.extend(["fc2"]).unwrap(),
            x,
        )
    }

    fn encoder_layer(&self, builder: &Builder, config: &TransformerConfig, p: Path, x: Var) -> Var {
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["layer_norm1"]).unwrap(),
            x,
        );
        let x = self.attention(builder, config, p.extend(["self_attn"]).unwrap(), x);
        let x = x + res;

        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["layer_norm2"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, config, p.extend(["mlp"]).unwrap(), x);
        x + res
    }

    fn vision_embeddings(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let patch_size = config.patch_size;
        let image_size = config.image_size;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let num_channels = 3;
        let hidden_size = config.transformer.hidden_size;

        let patch_weight = param(builder, &p.extend(["patch_embedding", "weight"]).unwrap());

        let patch_weight_permuted = transpose(builder, 1, 2, patch_weight);
        let patch_weight_permuted = transpose(builder, 2, 3, patch_weight_permuted);

        let sh_flat = shape!(builder, hidden_size, num_channels * patch_size * patch_size);
        let patch_weight_flat = reshape(builder, sh_flat, patch_weight_permuted);

        let patch_weight_flat_t = transpose(builder, 0, 1, patch_weight_flat);

        let sh_expand = shape!(
            builder,
            1,
            num_channels * patch_size * patch_size,
            hidden_size
        );
        let patch_weight_flat_t = broadcast(builder, patch_weight_flat_t, sh_expand);

        let mut patch_embeddings = matmul(builder, x, patch_weight_flat_t);

        let bias = param(builder, &p.extend(["patch_embedding", "bias"]).unwrap());
        let sh_emb = shape(builder, patch_embeddings.clone());
        let bias_expanded = broadcast(builder, bias, sh_emb);
        patch_embeddings = patch_embeddings + bias_expanded;

        let weights = param(
            builder,
            &p.extend(["position_embedding", "weight"]).unwrap(),
        );

        let sh_pe = shape!(builder, 1, num_patches, config.transformer.hidden_size);
        let pe = broadcast(builder, weights, sh_pe);

        patch_embeddings + pe
    }

    fn vision_model(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let mut x = self.vision_embeddings(builder, config, p.extend(["embeddings"]).unwrap(), x);
        let config = &config.transformer;
        for i in 0..config.num_hidden_layers {
            x = self.encoder_layer(
                builder,
                config,
                p.extend(["encoder", "layers", &i.to_string()]).unwrap(),
                x,
            );
        }
        layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["post_layernorm"]).unwrap(),
            x,
        )
    }
}

impl SiglipModel {
    fn vision_head_attention(
        &self,
        builder: &Builder,
        config: &TransformerConfig,
        p: Path,
        probe: Var,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let weight = param(builder, &p.extend(["in_proj_weight"]).unwrap());
        let bias = param(builder, &p.extend(["in_proj_bias"]).unwrap());

        let ws = chunk(builder, 0, 3, dim, weight);
        let bs = chunk(builder, 0, 3, dim, bias);

        let q = linear_param(builder, dim, dim, ws[0].clone(), bs[0].clone(), probe);
        let k = linear_param(builder, dim, dim, ws[1].clone(), bs[1].clone(), x.clone());
        let v = linear_param(builder, dim, dim, ws[2].clone(), bs[2].clone(), x);

        let sh_q = shape!(builder, b, 1, num_heads, head_dim);
        let sh_kv = shape!(builder, b, s, num_heads, head_dim);

        let q = reshape(builder, sh_q, q);
        let k = reshape(builder, sh_kv.clone(), k);
        let v = reshape(builder, sh_kv, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let head_dim_float = head_dim as f32;
        let sh_attn = shape(builder, attn.clone());
        let denom = constant(builder, head_dim_float.sqrt(), &sh_attn);
        let attn = attn / denom;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let sh_out = shape!(builder, b, 1, dim);
        let x = reshape(builder, sh_out, x);

        linear(builder, dim, dim, p.extend(["out_proj"]).unwrap(), x)
    }

    fn vision_head(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let t_config = &config.transformer;
        let probe = param(builder, &p.extend(["probe"]).unwrap());

        let sh_probe = shape!(builder, 1, 1, t_config.hidden_size);
        let probe = reshape(builder, sh_probe, probe);

        let x = self.vision_head_attention(
            builder,
            t_config,
            p.extend(["attention"]).unwrap(),
            probe,
            x,
        );
        let res = x.clone();
        let x = layernorm(
            builder,
            t_config.layer_norm_eps,
            p.extend(["layernorm"]).unwrap(),
            x,
        );

        let x = self
            .vision_backbone
            .mlp(builder, t_config, p.extend(["mlp"]).unwrap(), x);
        let res = x + res;

        slice(builder, 1, 0, 1, res)
    }

    fn vision_model_with_head(
        &self,
        builder: &Builder,
        config: &VisionConfig,
        p: Path,
        x: Var,
    ) -> Var {
        let x = self
            .vision_backbone
            .vision_model(builder, config, p.clone(), x);
        self.vision_head(builder, config, p.extend(["head"]).unwrap(), x)
    }

    fn text_embeddings(&self, builder: &Builder, config: &TextConfig, p: Path, x: Var) -> Var {
        let [b, s] = unpack::<2>(builder, shape(builder, x.clone()));
        let sh = shape!(builder, b, s, config.transformer.hidden_size);

        let wte = param(builder, &p.extend(["token_embedding", "weight"]).unwrap());
        let we = index(builder, 0, x, wte);
        let we = reshape(builder, sh.clone(), we);

        let pos = arange(builder, s);

        let pe_weights = param(
            builder,
            &p.extend(["position_embedding", "weight"]).unwrap(),
        );
        let pe = index(builder, 0, pos, pe_weights);

        let pe = broadcast(builder, pe, sh);

        we + pe
    }

    fn text_model(&self, builder: &Builder, config: &TextConfig, p: Path, x: Var) -> Var {
        let mut x = self.text_embeddings(builder, config, p.extend(["embeddings"]).unwrap(), x);
        let config_t = &config.transformer;
        for i in 0..config_t.num_hidden_layers {
            x = self.vision_backbone.encoder_layer(
                builder,
                config_t,
                p.extend(["encoder", "layers", &i.to_string()]).unwrap(),
                x,
            );
        }
        let x = layernorm(
            builder,
            config_t.layer_norm_eps,
            p.extend(["final_layer_norm"]).unwrap(),
            x,
        );

        let x = slice(builder, 1, 63, 1, x);

        linear(
            builder,
            config.transformer.hidden_size,
            config.transformer.hidden_size,
            p.extend(["head"]).unwrap(),
            x,
        )
    }

    fn div_l2_norm(&self, builder: &Builder, x: Var) -> Var {
        let sqr = x.clone() * x.clone();
        let l2_norm = sum(builder, sqr);
        let l2_norm = sqrt(builder, l2_norm);
        let sh_x = shape(builder, x.clone());
        let l2_norm = broadcast(builder, l2_norm, sh_x);
        x / l2_norm
    }
}

impl Module<1, 1> for SiglipVisionBackbone {
    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let t = Type::Tensor(TypeExpr::Var(0));
        ([t.clone()], [t])
    }

    fn path(&self) -> Path {
        path(vec!["siglip_vision_backbone"]).unwrap()
    }

    fn def(&self, builder: &Builder, [image]: [Var; 1]) -> [Var; 1] {
        let image_features = self.vision_model(
            builder,
            &self.config,
            path(vec!["vision_model"]).unwrap(),
            image,
        );
        [image_features]
    }
}

impl Module<2, 2> for SiglipModel {
    fn ty(&self) -> ([Type; 2], [Type; 2]) {
        let t = Type::Tensor(TypeExpr::Var(0));
        ([t.clone(), t.clone()], [t.clone(), t])
    }

    fn path(&self) -> Path {
        path(vec!["siglip"]).unwrap()
    }

    fn def(&self, builder: &Builder, [text, image]: [Var; 2]) -> [Var; 2] {
        let dim = self.config.vision_config.transformer.hidden_size;

        let [n_text, _] = unpack::<2>(builder, shape(builder, text.clone()));

        let text_features = self.text_model(
            builder,
            &self.config.text_config,
            path(vec!["text_model"]).unwrap(),
            text,
        );
        let text_features = self.div_l2_norm(builder, text_features);
        let sh_text = shape!(builder, n_text, dim);
        let text_features = reshape(builder, sh_text, text_features);

        let [n_img, _, _] = unpack::<3>(builder, shape(builder, image.clone()));

        let image_features = self.vision_model_with_head(
            builder,
            &self.config.vision_config,
            path(vec!["vision_model"]).unwrap(),
            image,
        );
        let image_features = self.div_l2_norm(builder, image_features);
        let sh_img = shape!(builder, n_img, dim);
        let image_features = reshape(builder, sh_img, image_features);

        let rt = transpose(builder, 0, 1, image_features);
        let logits_per_text = matmul(builder, text_features, rt);

        let logit_scale = param(builder, &path(vec!["logit_scale"]).unwrap());
        let logit_bias = param(builder, &path(vec!["logit_bias"]).unwrap());
        let logit_scale = exp(builder, logit_scale);

        let sh_logits = shape(builder, logits_per_text.clone());
        let logit_scale = broadcast(builder, logit_scale, sh_logits.clone());
        let logit_bias = broadcast(builder, logit_bias, sh_logits);

        let logits_per_text = logits_per_text * logit_scale + logit_bias;
        let logits_per_image = transpose(builder, 0, 1, logits_per_text.clone());

        [logits_per_text, logits_per_image]
    }
}

// Loads the image and returns flattened data + shape
fn load_and_preprocess_image(
    image_path: &PathBuf,
    image_size: usize,
    patch_size: usize,
) -> (Vec<f32>, Shape) {
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
    (
        patches,
        Shape(vec![1, num_patches, patch_size * patch_size * num_channels]),
    )
}

fn load_tensors(
    paths: Vec<PathBuf>,
    backend: &CandleBackend,
) -> Result<interpreter::Parameters<CandleBackend>> {
    let mut map = HashMap::new();

    for file_path in paths {
        let file = std::fs::File::open(file_path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&data)?;

        for (name, view) in tensors.tensors() {
            let shape = Shape(view.shape().to_vec());
            let tensor_data = view.data();

            // Load as F32
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => tensor_data
                    .par_chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                safetensors::Dtype::BF16 => tensor_data
                    .par_chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                _ => panic!("Unsupported dtype: {:?}", view.dtype()),
            };

            let tensor = interpreter::tensor(backend, shape, data)
                .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;
            let key = path(name.split(".").collect()).expect("invalid param path");
            map.insert(key, tensor);
        }
    }

    Ok(interpreter::Parameters::from(map))
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    let (model_paths, config_path, tokenizer_path, _) = get_model_files(&args.model_name, "main")?;

    let config: SiglipConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let backend = CandleBackend::new();
    let parameters = load_tensors(model_paths, &backend)?;
    println!("SigLIP model {} loaded successfully.", args.model_name);

    let (image_data, image_shape) = load_and_preprocess_image(
        &args.image,
        config.vision_config.image_size,
        config.vision_config.patch_size,
    );
    let image_tensor = interpreter::tensor(&backend, image_shape, image_data)
        .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;

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
    let input_shape = Shape(vec![batches, max_len]);

    // Convert encodings (i32) to u32 for interpreter
    let encodings_u32: Vec<u32> = encodings.iter().map(|&x| x as u32).collect();
    let input_tensor = interpreter::tensor(&backend, input_shape, encodings_u32)
        .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;

    // Build model
    let model = SiglipModel {
        config: config.clone(),
        vision_backbone: SiglipVisionBackbone {
            config: config.vision_config,
        },
    };
    let typed_term = model.term().expect("failed to build model term");

    let mut env = catgrad::stdlib::stdlib();
    let param_keys: Vec<Path> = parameters.0.keys().cloned().collect();
    env.declarations
        .extend(catgrad::stdlib::to_load_ops(Path::empty(), &param_keys));

    let interp = Interpreter::new(backend, env, parameters);

    let results = interp.eval(
        interp.environment.to_core(typed_term.term),
        vec![input_tensor, image_tensor],
    )?;
    let result_tensor = match &results[1] {
        interpreter::Value::Tensor(t) => t,
        _ => panic!("Expected tensor output"),
    };

    let vec_res = interp.backend.to_vec(result_tensor.clone());

    let probs = match vec_res {
        interpreter::TaggedVec::F32(v) => v,
        _ => panic!("Expected F32 data"),
    };

    let max_val = probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_probs: Vec<f32> = probs.iter().map(|x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exp_probs.iter().sum();
    let softmax_probs: Vec<f32> = exp_probs.iter().map(|x| x / sum_exp).collect();

    for (i, label) in labels.iter().enumerate() {
        println!("{label}: {:.4}%", softmax_probs[i] * 100.);
    }

    Ok(())
}
