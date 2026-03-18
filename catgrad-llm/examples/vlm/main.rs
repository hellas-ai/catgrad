use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::{self, Backend, Interpreter};
use catgrad::prelude::*;
use catgrad_llm::config::LLMConfig;
use catgrad_llm::utils::{
    cache_path_for_embeddings, get_model, get_model_chat_template, load_and_preprocess_image,
    load_cached_embeddings, load_model, post_process_model_weights, render_chat_template,
    save_cached_embeddings,
};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;

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

type KvCache<B> = (interpreter::Value<B>, interpreter::Value<B>);

fn empty_kv_cache<B: interpreter::Backend>(
    backend: &B,
    config: &dyn LLMConfig,
) -> Result<KvCache<B>> {
    let k_shape = Shape(vec![
        config.num_kv_layers(),
        1,
        config.num_key_value_heads(),
        0,
        config.get_qk_head_dim(),
    ]);
    let v_shape = Shape(vec![
        config.num_kv_layers(),
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

    let backend = CandleBackend::new();
    let (mut parameter_values, mut parameter_types, config_json, tokenizer, _) =
        load_model(&args.model_name, "main", &backend)?;

    let chat_template = get_model_chat_template(&args.model_name, "main").unwrap_or_default();

    let prompt = if chat_template.is_empty() || args.raw {
        args.prompt.clone()
    } else {
        render_chat_template(&chat_template, &args.prompt, true, false).unwrap()
    };

    let bootstrap_model = get_model(&config_json, 1)?;
    let prompt = bootstrap_model
        .multimodal_interpolate_prompt(&prompt)
        .ok_or_else(|| anyhow::anyhow!("Model does not provide multimodal prompt interpolation"))?;

    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;
    let mut token_ids = encoding.get_ids().to_vec();
    println!("token_ids: {:?}", token_ids);

    let max_sequence_length = args.max_seq_len + token_ids.len();
    let model = get_model(&config_json, max_sequence_length)?;
    let mm = model
        .multimodal_metadata()
        .ok_or_else(|| anyhow::anyhow!("Model is not multimodal"))?;

    let vision_model = model
        .multimodal_vision_module()
        .ok_or_else(|| anyhow::anyhow!("Model does not provide a multimodal vision module"))?;
    let language_model = model
        .multimodal_language_module()
        .ok_or_else(|| anyhow::anyhow!("Model does not provide a multimodal language module"))?;

    post_process_model_weights(
        model.as_ref(),
        &backend,
        &mut parameter_values,
        &mut parameter_types,
    )?;

    // Initialize interpreter
    let mut env = catgrad::stdlib::stdlib();
    env.declarations.extend(catgrad::stdlib::to_load_ops(
        Path::empty(),
        parameter_types.keys(),
    ));

    let interpreter = Interpreter::new(backend.clone(), env, parameter_values);

    let (image_data, image_shape) =
        load_and_preprocess_image(&args.image, mm.image_size, mm.patch_size)?;

    let cache_path =
        cache_path_for_embeddings(&args.model_name, args.image.to_str().unwrap(), &image_data);

    // Cache the expensive computation of visual embeddings
    let visual_embeddings_tensor =
        if let Ok(visual_embeddings) = load_cached_embeddings(&cache_path) {
            println!("Loading cached image features from: {:?}", cache_path);
            interpreter::tensor(
                &backend,
                Shape(vec![1, mm.mm_tokens_per_image, mm.hidden_size]),
                visual_embeddings,
            )
            .unwrap()
        } else {
            let image_tensor = interpreter::tensor(&backend, Shape(image_shape), image_data)
                .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;

            // Get embeddings from the image
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
    let language_term = language_model
        .term()
        .expect("failed to build language model term");

    let max_seq_len = args.max_seq_len;
    let empty_cache = empty_kv_cache(&interpreter.backend, model.config())?;
    let mut kv_cache = empty_cache;
    let mut use_image_embeddings = true;
    let eos_token_ids = model.config().get_eos_token_ids();

    // Run text generation loop
    for _i in 0..max_seq_len {
        let (next_token_id, new_cache) = run_interpreter(
            &language_term,
            &interpreter,
            &token_ids,
            mm.hidden_size,
            mm.image_token_index,
            &visual_embeddings_tensor,
            &kv_cache,
            use_image_embeddings,
        )?;
        if eos_token_ids.contains(&(next_token_id as i32)) {
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

#[allow(clippy::too_many_arguments)]
fn run_interpreter(
    typed_term: &TypedTerm,
    interpreter: &interpreter::Interpreter<CandleBackend>,
    text_tokens: &[u32],
    hidden_size: usize,
    image_token_index: usize,
    visual_embeddings_tensor: &interpreter::Value<CandleBackend>,
    kv_cache: &KvCache<CandleBackend>,
    use_image_embeddings: bool,
) -> Result<(u32, KvCache<CandleBackend>)> {
    let empty_image_embeddings = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, 0, hidden_size]),
        Vec::<f32>::new(),
    )
    .expect("Failed to create empty image embeddings");

    let (text_before_tokens, text_after_tokens, image_embeddings) = if use_image_embeddings {
        let first_image_token_index = text_tokens
            .iter()
            .position(|&x| x == image_token_index as u32)
            .unwrap_or(0);

        let last_image_token_index = text_tokens
            .iter()
            .rposition(|&x| x == image_token_index as u32)
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
