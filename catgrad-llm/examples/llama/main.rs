use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::*;
use catgrad_llm::helpers::LLMModel;
use catgrad_llm::utils::{
    cache_path_for_embeddings, empty_state_cache, get_model, get_model_chat_template,
    interpolate_multimodal_prompt, load_cached_embeddings, load_model, post_process_model_weights,
    prepare_multimodal_input, print_bench_table, render_chat_template, save_cached_embeddings,
    split_image_tokens,
};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Model name on Huggingface Hub
    #[arg(
        short = 'm',
        long,
        default_value = "HuggingFaceTB/SmolLM2-135M-Instruct"
    )]
    model_name: String,
    /// Model revision (branch, tag, or commit)
    #[arg(short = 'r', long, default_value = "main")]
    revision: String,
    /// TOML config file overriding model aliases.
    #[arg(short = 'c', long, value_name = "PATH")]
    config_file: Option<PathBuf>,
    /// List configured model aliases and exit
    #[arg(long)]
    list_models: bool,
    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Category theory is")]
    prompt: String,
    /// Optional image input for multimodal-capable models
    #[arg(short = 'i', long)]
    image: Option<PathBuf>,
    /// Pass raw prompt without chat template
    #[arg(long)]
    raw: bool,
    /// Thinking mode
    #[arg(long)]
    thinking: bool,
    /// Tokens to generate
    #[arg(short = 's', long, default_value_t = 1)]
    max_seq_len: usize,
    /// Use KV-cache
    #[arg(short = 'k', long)]
    kv_cache: bool,
    /// Enable typecheck
    #[arg(short = 't', long)]
    typecheck: bool,
    /// Backend to use
    #[arg(short = 'b', long, value_enum, default_value_t = BackendChoice::Candle)]
    backend: BackendChoice,
    /// Disable Candle backend acceleration
    #[arg(long = "no-accel")]
    no_accel: bool,
    /// Dump the constructed graph to this JSON file then exit.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Load model from a previously dumped JSON graph
    #[arg(long)]
    load: Option<PathBuf>,
    /// Benchmark
    #[arg(
        long,
        num_args = 2,
        value_names = ["PP", "TG"]
    )]
    bench: Option<Vec<usize>>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendChoice {
    Ndarray,
    Candle,
}

impl BackendChoice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ndarray => "Ndarray",
            Self::Candle => "Candle",
        }
    }
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    aliases: HashMap<String, String>,
}

fn parse_config(contents: &str, source: &str) -> Result<AppConfig> {
    let config: AppConfig = toml::from_str(contents)
        .map_err(|e| anyhow::anyhow!("invalid config file {source}: {e}"))?;
    Ok(config)
}

fn merge_config_file(app_config: &mut AppConfig, path: &Path) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read alias file {}: {e}", path.display()))?;
    let cfg = parse_config(&contents, &path.display().to_string())?;
    app_config.aliases.extend(cfg.aliases);
    Ok(())
}

// The app config currently contains only model aliases.
// The hardcoded ones can be overridden by user config files.
fn get_app_config(args: &Args) -> Result<AppConfig> {
    let default_config = include_str!("llm_config.default.toml");
    let mut app_config = parse_config(default_config, "embedded defaults")?;

    let local_alias_path = Path::new("llm_config.toml");
    if local_alias_path.exists() {
        merge_config_file(&mut app_config, local_alias_path)?;
    }

    if let Some(config_file) = &args.config_file {
        merge_config_file(&mut app_config, config_file)?;
    }
    Ok(app_config)
}

/// Construct, shapecheck, and interpret the a given LLM using the selected backend.
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let app_config = get_app_config(&args)?;
    if args.list_models {
        for (model, alias) in get_models(&app_config) {
            println!("{model} ({alias})");
        }
        return Ok(());
    }
    match args.backend {
        BackendChoice::Ndarray => run_with_backend(&args, &app_config, NdArrayBackend),
        BackendChoice::Candle => {
            run_with_backend(&args, &app_config, CandleBackend::new_accel(!args.no_accel))
        }
    }
}

fn get_model_name(args: &Args, app_config: &AppConfig) -> String {
    app_config
        .aliases
        .get(args.model_name.as_str())
        .cloned()
        .unwrap_or_else(|| args.model_name.clone())
}

fn get_models(app_config: &AppConfig) -> Vec<(&str, &str)> {
    let mut models: Vec<(&str, &str)> = app_config
        .aliases
        .iter()
        .map(|(alias, model)| (model.as_str(), alias.as_str()))
        .collect();
    models.sort_unstable_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    models
}

fn run_with_backend<B: interpreter::Backend>(
    args: &Args,
    app_config: &AppConfig,
    backend: B,
) -> Result<()> {
    let model_name = get_model_name(args, app_config);
    let model_dtype = Dtype::F32;

    let start_load = std::time::Instant::now();
    let (
        mut parameter_values,
        mut parameter_types,
        config_json,
        tokenizer,
        tokenizer_config,
        total_params,
    ) = load_model(&model_name, &args.revision, &backend, model_dtype)?;
    let elapsed_load = start_load.elapsed();

    eprintln!(
        "Model weights loaded for {} in {:.2} seconds",
        model_name,
        elapsed_load.as_secs_f64()
    );

    let chat_template = get_model_chat_template(&model_name, &args.revision).unwrap_or_default();

    let benchmarking = args.bench.is_some();
    let mut pp = 0;
    let mut tg = 0;
    let mut max_seq_len = args.max_seq_len;
    let use_image = args.image.is_some() && !benchmarking;

    let prompt = if let Some(bench) = &args.bench {
        pp = bench[0];
        tg = bench[1];
        max_seq_len = tg;
        eprintln!(
            "Benchmarking {} with prefill size {} and sequence length {}",
            &model_name, pp, tg
        );
        "The".repeat(pp)
    } else if chat_template.is_empty() || args.raw {
        args.prompt.clone()
    } else {
        render_chat_template(
            &chat_template,
            &tokenizer_config,
            &args.prompt,
            use_image,
            args.thinking,
        )?
    };

    let prepared_multimodal = if use_image {
        prepare_multimodal_input(&config_json, args.image.as_deref())?
    } else {
        Default::default()
    };
    let runtime_context = prepared_multimodal.runtime_context.as_ref();

    if !benchmarking && !use_image {
        print!("{prompt}");
    }
    let prompt = if use_image {
        interpolate_multimodal_prompt(&config_json, runtime_context, &prompt)?
    } else {
        prompt
    };

    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();
    let max_sequence_length = max_seq_len + token_ids.len();
    let model = get_model(
        &config_json,
        max_sequence_length,
        runtime_context,
        model_dtype,
    )?;
    post_process_model_weights(
        model.as_ref(),
        &backend,
        &mut parameter_values,
        &mut parameter_types,
    )?;

    let mm_metadata = if use_image {
        Some(
            model
                .multimodal_metadata()
                .ok_or_else(|| anyhow::anyhow!("Model {} is not multimodal", model_name))?,
        )
    } else {
        None
    };

    let typed_term = if let Some(load_path) = &args.load {
        let file = std::fs::File::open(load_path)?;
        serde_json::from_reader(file)?
    } else if use_image {
        let language_model = model.multimodal_language_module().ok_or_else(|| {
            anyhow::anyhow!(
                "Model {} does not provide multimodal language module",
                model_name
            )
        })?;
        language_model
            .term()
            .ok_or_else(|| anyhow::anyhow!("Failed to create multimodal typed term"))?
    } else {
        model.term().expect("Failed to create typed term")
    };

    if let Some(dump_path) = &args.dump {
        let file = std::fs::File::create(dump_path)?;
        serde_json::to_writer_pretty(file, &typed_term)?;
        eprintln!(
            "Graph for {} and max_seq_length of {max_sequence_length} dumped to {}",
            model.path(),
            dump_path.display()
        );
        return Ok(());
    }

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    let load_prefix = if use_image {
        catgrad::prelude::Path::empty()
    } else {
        model.path()
    };
    env.declarations
        .extend(to_load_ops(load_prefix, parameter_types.keys()));

    // Shapecheck the model
    if args.typecheck {
        typecheck::check(&env, &parameter_types, typed_term.clone()).map_err(anyhow::Error::new)?;
    }

    let mut generated_tokens = 0;
    let mut start_gen = std::time::Instant::now();
    let mut elapsed_pp = std::time::Duration::ZERO;
    let interpreter = interpreter::Interpreter::new(backend, env, parameter_values);

    let mut multimodal_ctx: Option<MultimodalRuntime<B>> = None;
    if let Some(mm) = mm_metadata {
        let vision_model = model.multimodal_vision_module().ok_or_else(|| {
            anyhow::anyhow!("Model {} does not provide vision module", model_name)
        })?;
        let image_path = args
            .image
            .as_ref()
            .expect("image existence already checked");
        let prepared_image = prepared_multimodal.image.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Model {} did not provide prepared image input for {}",
                model_name,
                image_path.display()
            )
        })?;
        let image_data = prepared_image.data.clone();
        let image_shape = prepared_image.shape.clone();
        let cache_path =
            cache_path_for_embeddings(&model_name, &image_path.to_string_lossy(), &image_data);
        let visual_embeddings = if let Ok(cached) = load_cached_embeddings(&cache_path) {
            eprintln!(
                "Loading cached image features from: {}",
                cache_path.display()
            );
            interpreter::tensor(
                &interpreter.backend,
                Shape(vec![1, mm.mm_tokens_per_image, mm.hidden_size]),
                cached,
            )
            .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?
        } else {
            let image_tensor =
                interpreter::tensor(&interpreter.backend, Shape(image_shape), image_data)
                    .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;
            let vision_term = vision_model
                .term()
                .ok_or_else(|| anyhow::anyhow!("failed to build vision model term"))?;
            let results = interpreter.run(vision_term.term, vec![image_tensor])?;
            let visual_embeddings = results
                .first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Vision model returned no outputs"))?;
            let flattened = to_f32_vec(&interpreter.backend, &visual_embeddings)?;
            save_cached_embeddings(&cache_path, &flattened)?;
            eprintln!("Saved image features to: {}", cache_path.display());
            visual_embeddings
        };

        multimodal_ctx = Some(MultimodalRuntime {
            hidden_size: mm.hidden_size,
            image_token_index: mm.image_token_index,
            visual_embeddings,
        });
    }

    let eos_token_ids = model.config().get_eos_token_ids();

    let mut state_cache = empty_state_cache(&interpreter.backend, model.as_ref())?;
    let use_kv_cache = args.kv_cache || use_image;
    let mut use_image_embeddings = use_image;

    // Run inference loop
    for i in 0..max_seq_len {
        let decode_inputs = if let Some(ctx) = multimodal_ctx.as_ref() {
            DecodeInputs::Multimodal {
                input_tokens: &token_ids,
                hidden_size: ctx.hidden_size,
                image_token_index: ctx.image_token_index,
                visual_embeddings: &ctx.visual_embeddings,
                use_image_embeddings,
            }
        } else {
            DecodeInputs::Text {
                input_tokens: &token_ids,
            }
        };
        let (next_token_id, updated_state_cache) = run_interpreter(
            model.as_ref(),
            &typed_term,
            &interpreter,
            decode_inputs,
            &state_cache,
            max_sequence_length,
        )?;
        if i == 0 {
            elapsed_pp = start_gen.elapsed();
            start_gen = std::time::Instant::now();
        }
        generated_tokens += 1;
        if eos_token_ids.contains(&(next_token_id as i32)) && !benchmarking {
            break;
        }
        if use_kv_cache {
            state_cache = updated_state_cache;
            token_ids = vec![next_token_id];
        } else {
            token_ids.push(next_token_id);
        }
        if use_image && use_kv_cache {
            use_image_embeddings = false;
        }
        if !benchmarking {
            let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
            print!("{decoded_token}");
            std::io::stdout().flush()?;
        }
    }

    let elapsed_gen = start_gen.elapsed();
    if benchmarking {
        // hardcode size multiplier as 4.0 since we only load in F32
        let size_gib = (total_params as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        let params_m = total_params as f64 / 1_000_000.0;
        print_bench_table(
            &model_name,
            size_gib,
            params_m,
            args.backend.as_str(),
            pp,
            elapsed_pp,
            tg,
            elapsed_gen,
        );
    } else {
        println!();
        eprintln!(
            "{} tokens generated in {} seconds. ({:.2} tps)",
            generated_tokens,
            (elapsed_pp + elapsed_gen).as_secs(),
            generated_tokens as f64 / (elapsed_pp + elapsed_gen).as_secs_f64(),
        );
    }
    Ok(())
}

fn to_f32_vec<B: interpreter::Backend>(
    backend: &B,
    value: &interpreter::Value<B>,
) -> Result<Vec<f32>> {
    match value.clone() {
        interpreter::Value::Tensor(arr) => match backend.to_vec(arr) {
            interpreter::TaggedVec::F32(v) => Ok(v),
            _ => Err(anyhow::anyhow!("Unexpected output dtype")),
        },
        t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
    }
}

struct MultimodalRuntime<B: interpreter::Backend> {
    hidden_size: usize,
    image_token_index: usize,
    visual_embeddings: interpreter::Value<B>,
}

enum DecodeInputs<'a, B: interpreter::Backend> {
    Text {
        input_tokens: &'a [u32],
    },
    Multimodal {
        input_tokens: &'a [u32],
        hidden_size: usize,
        image_token_index: usize,
        visual_embeddings: &'a interpreter::Value<B>,
        use_image_embeddings: bool,
    },
}

fn token_tensor<B: interpreter::Backend>(
    interpreter: &interpreter::Interpreter<B>,
    label: &str,
    input_tokens: &[u32],
) -> Result<interpreter::Value<B>> {
    interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, input_tokens.len()]),
        input_tokens.to_vec(),
    )
    .map_err(|err| anyhow::anyhow!("{label} tensor error: {:?}", err))
}

fn run_interpreter<B: interpreter::Backend>(
    model: &dyn LLMModel,
    typed_term: &TypedTerm,
    interpreter: &interpreter::Interpreter<B>,
    decode_inputs: DecodeInputs<'_, B>,
    state_cache: &[interpreter::Value<B>],
    max_sequence_length: usize,
) -> Result<(u32, Vec<interpreter::Value<B>>)> {
    let mut inputs = Vec::with_capacity(state_cache.len() + 4);
    let input_seq_len;

    match decode_inputs {
        DecodeInputs::Text { input_tokens } => {
            input_seq_len = input_tokens.len();
            inputs.push(token_tensor(interpreter, "input", input_tokens)?);
        }
        DecodeInputs::Multimodal {
            input_tokens,
            hidden_size,
            image_token_index,
            visual_embeddings,
            use_image_embeddings,
        } => {
            input_seq_len = input_tokens.len();
            let empty_image_embeddings = interpreter::tensor(
                &interpreter.backend,
                Shape(vec![1, 0, hidden_size]),
                Vec::<f32>::new(),
            )
            .map_err(|err| anyhow::anyhow!("empty image tensor error: {:?}", err))?;

            let (text_before_tokens, text_after_tokens) = if use_image_embeddings {
                split_image_tokens(input_tokens, image_token_index)?
            } else {
                (&[][..], input_tokens)
            };

            let text_before = token_tensor(interpreter, "text_before", text_before_tokens)?;
            let text_after = token_tensor(interpreter, "text_after", text_after_tokens)?;
            let image_embeddings = if use_image_embeddings {
                visual_embeddings.clone()
            } else {
                empty_image_embeddings
            };

            inputs.push(text_before);
            inputs.push(image_embeddings);
            inputs.push(text_after);
        }
    }

    inputs.extend(state_cache.iter().cloned());
    inputs.push(interpreter::Value::Nat(max_sequence_length));
    // Workaround: push extra nat input needed for gated delta decoding, representing chunk_number.
    // It is seq_len dependent and cannot be computed in-graph as it is seq_len % chunk_size.
    if let Some(extra_nat) = model.extra_nat_input(input_seq_len) {
        inputs.push(interpreter::Value::Nat(extra_nat));
    }

    // Run the model
    let mut results = interpreter
        .run(typed_term.term.clone(), inputs)
        .expect("Failed to run inference");

    if results.is_empty() {
        return Err(anyhow::anyhow!("model returned no outputs"));
    }
    let updated_state_cache = if results.len() > 1 {
        results.split_off(1)
    } else {
        Vec::new()
    };
    let output = results.remove(0);

    match output {
        interpreter::Value::Tensor(arr) => match interpreter.backend.to_vec(arr) {
            interpreter::TaggedVec::U32(v) => {
                let token = v
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("token output tensor was empty"))?;
                Ok((token, updated_state_cache))
            }
            _ => Err(anyhow::anyhow!("Unexpected output dtype")),
        },
        t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
    }
}
