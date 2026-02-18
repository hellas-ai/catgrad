use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::{self, Backend, Interpreter};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;
use catgrad_llm::config::LLMConfig;
use catgrad_llm::helpers::*;

use catgrad::typecheck::TypeExpr;
use catgrad_llm::models::gemma3::{Gemma3Model, GemmaTextConfig, multi_modal_projector};
use catgrad_llm::models::siglip::{SiglipVisionBackbone, VisionConfig};
use catgrad_llm::utils::{
    get_model_chat_template, get_model_files, load_and_preprocess_image, load_model_weights,
    render_chat_template,
};
use clap::Parser;
use std::path::PathBuf;
use std::{fs::File, io::Read, io::Write};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    /// Model name on Huggingface Hub
    #[arg(short = 'm', long, default_value = "google/paligemma2-3b-mix-224")]
    model_name: String,

    /// Path to an image file
    #[arg(short = 'i', long)]
    image: PathBuf,

    #[arg(short = 'p', long, default_value = "")]
    prompt: String,

    /// Pass raw prompt without chat template
    #[arg(long)]
    raw: bool,

    /// Tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    max_seq_len: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct VLMConfig {
    text_config: GemmaTextConfig,
    vision_config: VisionConfig,
    #[serde(default = "default_mm_tokens_per_image")]
    mm_tokens_per_image: usize,
    image_token_index: usize,
}

fn default_mm_tokens_per_image() -> usize {
    256
}

type KvCache<B> = (interpreter::Value<B>, interpreter::Value<B>);

fn empty_kv_cache<B: interpreter::Backend>(
    backend: &B,
    config: &dyn LLMConfig,
) -> Result<KvCache<B>> {
    let k_shape = Shape(vec![
        config.num_hidden_layers(),
        1,
        config.num_key_value_heads(),
        0,
        config.get_qk_head_dim(),
    ]);
    let v_shape = Shape(vec![
        config.num_hidden_layers(),
        1,
        config.num_key_value_heads(),
        0,
        config.get_v_head_dim(),
    ]);
    let k = interpreter::tensor(backend, k_shape, Vec::<f32>::new())
        .map_err(|err| anyhow::anyhow!("kv cache tensor error: {:?}", err))?;
    let v = interpreter::tensor(backend, v_shape, Vec::<f32>::new())
        .map_err(|err| anyhow::anyhow!("kv cache tensor error: {:?}", err))?;
    Ok((k, v))
}

struct VLMModel {
    language_model: Box<Gemma3Model>,
}
impl VLMModel {
    pub fn bidirectional_mask(
        &self,
        builder: &Builder,
        size: Var,
        img_start: Var,
        img_size: Var,
    ) -> Var {
        let row = arange(builder, size.clone());
        let sh = shape(builder, row.clone());

        let img_end = img_start.clone() + img_size.to_nat(builder);

        let img_start = nat_to_u32(builder, img_start);
        let img_start = broadcast(builder, img_start, sh.clone());

        let img_end = nat_to_u32(builder, img_end);
        let img_end = broadcast(builder, img_end, sh);

        let img_mask_1 = gte(builder, row.clone(), img_start);
        let img_mask_2 = lt(builder, row, img_end);
        let row = img_mask_1 * img_mask_2;

        let sh = pack::<2>(builder, [size.clone(), size]);
        let row = broadcast(builder, row, sh.clone());
        let col = transpose(builder, 0, 1, row.clone());
        let mask = row * col;
        let mask = cast(builder, mask, Dtype::F32);
        let one = constant(builder, 1.0, &sh);
        one - mask
    }

    // Forward pass with image embeddings and text tokens as input
    #[allow(clippy::too_many_arguments)]
    pub fn forward_image_and_texts(
        &self,
        builder: &Builder,
        p: Path,
        text1: Var,
        image: Var,
        text2: Var,
        in_k: Var,
        in_v: Var,
    ) -> [Var; 3] {
        let text1 = self.language_model.scaled_embeddings(
            builder,
            p.extend(vec!["embed_tokens"]).unwrap(),
            text1,
        );
        let text2 = self.language_model.scaled_embeddings(
            builder,
            p.extend(vec!["embed_tokens"]).unwrap(),
            text2,
        );

        let [_b, img_start, _] = unpack::<3>(builder, shape(builder, text1.clone()));
        let embeddings = concat(builder, 1, text1, image);
        let embeddings = concat(builder, 1, embeddings, text2);

        let [_b, s, _] = unpack::<3>(builder, shape(builder, embeddings.clone()));

        let is_paligemma = self.language_model.config.model_type == "gemma2";

        let attention_mask = if is_paligemma {
            let sh = shape!(builder, s.clone(), s);
            constant(builder, 0.0, &sh)
        } else {
            let attention_mask = causal_mask(builder, s.clone());

            // hardcoded 256 tokens per image
            let image_mask = self.bidirectional_mask(builder, s, img_start, 256.to_nat(builder));

            attention_mask * image_mask
        };

        let [x, out_k, out_v] = self.language_model.forward_embeddings(
            builder,
            p,
            attention_mask,
            embeddings,
            in_k,
            in_v,
        );
        [x, out_k, out_v]
    }
}

// VLM model taking an image and a prompt, generating text
impl Module<5, 3> for VLMModel {
    fn path(&self) -> Path {
        path(vec!["VLM"]).unwrap()
    }

    fn ty(&self) -> ([Type; 5], [Type; 3]) {
        let t = Type::Tensor(TypeExpr::Var(0));
        (
            [t.clone(), t.clone(), t.clone(), t.clone(), t.clone()],
            [t.clone(), t.clone(), t],
        )
    }

    fn def(&self, builder: &Builder, [text1, image, text2, in_k, in_v]: [Var; 5]) -> [Var; 3] {
        self.forward_image_and_texts(
            builder,
            path(vec!["language_model", "model"]).unwrap(),
            text1,
            image,
            text2,
            in_k,
            in_v,
        )
    }
}

pub struct VisionEmbeddings {
    paligemma: bool,
    pub config: VisionConfig,
    pub vision_tower: SiglipVisionBackbone,
}

impl VisionEmbeddings {
    pub fn new(paligemma: bool, config: VisionConfig) -> Self {
        Self {
            paligemma,
            config,
            vision_tower: SiglipVisionBackbone {},
        }
    }
}

// VisionEmbeddings model generating embeddings from an image
impl Module<1, 1> for VisionEmbeddings {
    fn path(&self) -> Path {
        path(vec!["VisionEmbeddings"]).unwrap()
    }

    fn ty(&self) -> ([Type; 1], [Type; 1]) {
        let t = Type::Tensor(TypeExpr::Var(0));
        ([t.clone()], [t])
    }

    fn def(&self, builder: &Builder, [pixels]: [Var; 1]) -> [Var; 1] {
        let x = self.vision_tower.vision_model(
            builder,
            &self.config,
            path(vec!["vision_tower", "vision_model"]).unwrap(),
            pixels,
        );

        // Gemma3 case
        if !self.paligemma {
            let x = multi_modal_projector(builder, path(vec!["multi_modal_projector"]).unwrap(), x);
            return [x];
        }

        let x = linear(
            builder,
            self.config.hidden_size,
            self.config.hidden_size * 2,
            path(vec!["multi_modal_projector", "linear"]).unwrap(),
            x,
        );
        let scale = (self.config.projection_dim as f32).sqrt();
        let sh = shape(builder, x.clone());
        let scale = constant(builder, scale, &sh);
        let x = x / scale;
        [x]
    }
}

fn sanitize(model_name: &str) -> String {
    model_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

// Make a cache file name based on the model name, image name, and a simple image data checksum.
fn cache_path_for_embeddings(model_name: &str, image_name: &str, image_data: &[f32]) -> PathBuf {
    let cache_dir = std::env::var("CATGRAD_CACHE").unwrap_or_else(|_| ".cache".to_string());
    let checksum: u32 = image_data.iter().map(|x| x.to_bits()).sum();
    let filename = format!(
        "{}-{}-{:08x}.bin",
        sanitize(model_name),
        sanitize(image_name),
        checksum
    );
    PathBuf::from(cache_dir).join(filename)
}

fn load_cached_embeddings(path: &std::path::Path) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;

    assert!(bytes.len() % 4 == 0);

    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn save_cached_embeddings(path: &std::path::Path, data: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|err| anyhow::anyhow!("check error {:?}", err))
}

fn to_f32_vec(
    backend: &CandleBackend,
    value: &interpreter::Value<CandleBackend>,
) -> Result<Vec<f32>> {
    match value.clone() {
        interpreter::Value::Tensor(arr) => match backend.to_vec(arr) {
            interpreter::TaggedVec::F32(v) => Ok(v),
            _ => Err(anyhow::anyhow!("Unexpected output dtype")),
        },
        t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    let (model_paths, config_path, tokenizer_path, _) = get_model_files(&args.model_name, "main")?;

    let mut config: VLMConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    let is_paligemma = config.text_config.model_type == "gemma2";

    if is_paligemma {
        config.text_config.max_position_embeddings = 8192;
        config.text_config.rope_theta = 10000.0;
        config.mm_tokens_per_image = config.vision_config.num_image_tokens;
    }

    // println!("config: {:?}", config);

    let chat_template = get_model_chat_template(&args.model_name, "main").unwrap_or_default();

    // println!("chat_template: {}", chat_template);
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    let prompt = if chat_template.is_empty() || args.raw {
        args.prompt.clone()
    } else {
        render_chat_template(&chat_template, &args.prompt, true, false).unwrap()
    };

    // We insert 256 image tokens to mirror what the HF Transformers processor does
    // even though we replace them with image features later (so a single sentinel token would be enough)
    // What must be kept in the token stream are the start/end/bos/eol tokens.
    let prompt = if is_paligemma {
        let img_seq = format!("{}<bos>", "<image>".repeat(256));
        let p = prompt.replace("<image>", &img_seq);
        format!("{}\n", p)
    } else {
        // Gemma3 special image token sequence
        let img_seq = format!(
            "\n\n<start_of_image>{}<end_of_image>\n\n",
            "<image_soft_token>".repeat(256)
        );
        prompt.replace("<start_of_image>", &img_seq)
    };

    // println!("prompt: {}", prompt);
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    println!("token_ids: {:?}", token_ids);

    let backend = CandleBackend::new();
    let (parameters, _, _) = load_model_weights(model_paths, &backend)?;
    println!("VLM model {} loaded successfully.", args.model_name);

    // Initialize interpreter
    let mut env = catgrad::stdlib::stdlib();
    let param_keys: Vec<Path> = parameters.0.keys().cloned().collect();
    env.declarations
        .extend(catgrad::stdlib::to_load_ops(Path::empty(), &param_keys));

    let interpreter = Interpreter::new(backend.clone(), env, parameters);

    let (image_data, image_shape) = load_and_preprocess_image(
        &args.image,
        config.vision_config.image_size,
        config.vision_config.patch_size,
    );

    let cache_path =
        cache_path_for_embeddings(&args.model_name, args.image.to_str().unwrap(), &image_data);

    // Cache the expensive computation of visual embeddings
    let visual_embeddings_tensor =
        if let Ok(visual_embeddings) = load_cached_embeddings(&cache_path) {
            println!("Loading cached image features from: {:?}", cache_path);
            interpreter::tensor(
                &backend,
                Shape(vec![
                    1,
                    config.mm_tokens_per_image,
                    config.text_config.hidden_size,
                ]),
                visual_embeddings,
            )
            .unwrap()
        } else {
            let image_tensor = interpreter::tensor(&backend, Shape(image_shape), image_data)
                .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;

            // Get embeddings from the image
            let vision_model = VisionEmbeddings::new(is_paligemma, config.vision_config.clone());
            let vision_term = vision_model
                .term()
                .expect("failed to build vision model term");
            let results = interpreter.run(vision_term.term, vec![image_tensor])?;
            let visual_embeddings_tensor = results[0].clone();
            let v = to_f32_vec(&backend, &visual_embeddings_tensor)?;
            save_cached_embeddings(&cache_path, &v)?;
            visual_embeddings_tensor
        };

    println!("Visual embeddings shape: {:?}", visual_embeddings_tensor);
    let vlm_model = VLMModel {
        language_model: Box::new(Gemma3Model {
            root: "language_model.model".to_string(),
            config: config.text_config.clone(),
            max_sequence_length: 1000, //TODO
        }),
    };

    let language_term = vlm_model
        .term()
        .expect("failed to build language model term");

    let max_seq_len = args.max_seq_len;
    let empty_cache = empty_kv_cache(&interpreter.backend, &config.text_config)?;
    let mut kv_cache = empty_cache;
    let mut use_image_embeddings = true;

    // Run text generation loop
    for _i in 0..max_seq_len {
        let (next_token_id, new_cache) = run_interpreter(
            &language_term,
            &interpreter,
            &token_ids,
            &config,
            &visual_embeddings_tensor,
            &kv_cache,
            use_image_embeddings,
        )?;
        if config
            .text_config
            .clone()
            .get_eos_token_ids()
            .contains(&(next_token_id as i32))
        {
            break;
        }
        kv_cache = new_cache;
        // After the prefill stage the image embeddings are no longer needed, they are in the KV cache
        use_image_embeddings = false;
        token_ids = vec![next_token_id];
        let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
        print!("{}", decoded_token);
        std::io::stdout().flush()?;
    }
    Ok(())
}

fn run_interpreter(
    typed_term: &TypedTerm,
    interpreter: &interpreter::Interpreter<CandleBackend>,
    text_tokens: &[u32],
    config: &VLMConfig,
    visual_embeddings_tensor: &interpreter::Value<CandleBackend>,
    kv_cache: &KvCache<CandleBackend>,
    use_image_embeddings: bool,
) -> Result<(u32, KvCache<CandleBackend>)> {
    let empty_image_embeddings = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, 0, config.text_config.hidden_size]),
        Vec::<f32>::new(),
    )
    .expect("Failed to create empty image embeddings");

    let (text_before_tokens, text_after_tokens, image_embeddings) = if use_image_embeddings {
        let first_image_token_index = text_tokens
            .iter()
            .position(|&x| x == config.image_token_index as u32)
            .unwrap_or(0);

        let last_image_token_index = text_tokens
            .iter()
            .rposition(|&x| x == config.image_token_index as u32)
            .unwrap_or(0);

        (
            &text_tokens[..first_image_token_index],
            &text_tokens[last_image_token_index + 1..],
            visual_embeddings_tensor.clone(),
        )
    } else {
        (&[][..], text_tokens, empty_image_embeddings)
    };

    let text_before = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, text_before_tokens.len()]),
        text_before_tokens.to_vec(),
    )
    .expect("Failed to create input tensor");

    let text_after = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, text_after_tokens.len()]),
        text_after_tokens.to_vec(),
    )
    .expect("Failed to create input tensor");

    let mut results = interpreter
        .run(
            typed_term.term.clone(),
            vec![
                text_before,
                image_embeddings,
                text_after,
                kv_cache.0.clone(),
                kv_cache.1.clone(),
            ],
        )
        .expect("Failed to run inference");

    let out_v = results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("No KV cache V output"))?;
    let out_k = results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("No KV cache K output"))?;
    // Print info about the main output (should be the last one)
    if let Some(output) = results.pop() {
        match output {
            interpreter::Value::Tensor(arr) => match interpreter.backend.to_vec(arr) {
                interpreter::TaggedVec::U32(v) => Ok((v[v.len() - 1], (out_k, out_v))),
                _ => Err(anyhow::anyhow!("Unexpected output dtype")),
            },
            t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
        }
    } else {
        Err(anyhow::anyhow!("No result"))
    }
}
