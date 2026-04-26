use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::*;
use catgrad::runtime::{BoundProgram, Program, ProgramSpec, Session};
use std::sync::Arc;
use catgrad_llm::helpers::{LLMModel, ToolCall, ToolUseStep};
use catgrad_llm::utils::*;
use clap::{Parser, ValueEnum};
use minijinja::{Value, context};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

mod tools;

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
    /// Floating-point dtype to use for model weights and activations
    #[arg(long, default_value = "f32", value_parser = parse_model_dtype)]
    dtype: Dtype,
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
    /// Enable simple tool use for models whose chat template exposes a supported format.
    #[arg(long)]
    tool_use: bool,
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

fn parse_model_dtype(s: &str) -> Result<Dtype, String> {
    let dtype: Dtype = s.parse()?;
    match dtype {
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => Ok(dtype),
        Dtype::U32 => Err("model dtype must be f32, f16, or bf16".to_string()),
    }
}

fn dtype_size_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => 4,
        Dtype::F16 => 2,
        Dtype::BF16 => 2,
        Dtype::U32 => 4,
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
    if matches!(
        (args.backend, args.dtype),
        (BackendChoice::Ndarray, Dtype::F16 | Dtype::BF16)
    ) {
        anyhow::bail!("--dtype f16 and bf16 currently require the candle backend");
    }
    if args.tool_use && args.raw {
        anyhow::bail!("--tool-use does not support --raw");
    }
    if args.tool_use && args.bench.is_some() {
        anyhow::bail!("--tool-use does not support --bench");
    }

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
    models.sort_unstable_by_key(|a| a.0.to_lowercase());
    models
}

fn user_message(prompt: &str, has_image: bool) -> Value {
    if has_image {
        let content = vec![
            context!(type => "text", text => prompt),
            context!(type => "image"),
        ];
        context!(role => "user", content => content)
    } else {
        context!(role => "user", content => prompt)
    }
}

fn assistant_message(content: String) -> Value {
    context!(role => "assistant", content => content)
}

fn tool_call_id(index: usize) -> String {
    format!("call_{}", index + 1)
}

fn assistant_tool_message(content: String, tool_calls: &[ToolCall]) -> Value {
    let tool_calls: Vec<_> = tool_calls
        .iter()
        .enumerate()
        .map(|(index, tool_call)| {
            context!(
                id => tool_call_id(index),
                type => "function",
                function => context!(
                    name => &tool_call.name,
                    arguments => &tool_call.arguments,
                )
            )
        })
        .collect();
    context!(role => "assistant", content => content, tool_calls => tool_calls)
}

fn tool_message(index: usize, tool_call: &ToolCall, content: String) -> Value {
    context!(
        role => "tool",
        name => &tool_call.name,
        tool_call_id => tool_call_id(index),
        content => content,
    )
}

fn render_tool_prompt(
    chat_template: &str,
    tokenizer_config: &serde_json::Value,
    prompt: &str,
    tools: &[serde_json::Value],
    enable_thinking: bool,
) -> Result<String> {
    Ok(render_chat_template_values(
        chat_template,
        tokenizer_config,
        &[user_message(prompt, false)],
        RenderChatTemplateOptions {
            enable_thinking,
            tools: Some(tools),
        },
    )?)
}

fn render_tool_follow_up_prompt(
    chat_template: &str,
    tokenizer_config: &serde_json::Value,
    prompt: &str,
    tools: &[serde_json::Value],
    tool_use_step: &ToolUseStep,
    tool_responses: &[String],
    enable_thinking: bool,
) -> Result<String> {
    let assistant_message = if let Some(raw_tool_calls) = &tool_use_step.assistant_tool_calls_text {
        assistant_message(format!(
            "{}{}",
            tool_use_step.assistant_content, raw_tool_calls
        ))
    } else {
        assistant_tool_message(
            tool_use_step.assistant_content.clone(),
            &tool_use_step.tool_calls,
        )
    };

    let mut messages = vec![user_message(prompt, false), assistant_message];
    messages.extend(
        tool_use_step
            .tool_calls
            .iter()
            .zip(tool_responses.iter())
            .enumerate()
            .map(|(index, (tool_call, response))| tool_message(index, tool_call, response.clone())),
    );
    Ok(render_chat_template_values(
        chat_template,
        tokenizer_config,
        &messages,
        RenderChatTemplateOptions {
            enable_thinking,
            tools: Some(tools),
        },
    )?)
}

fn run_with_backend<B: interpreter::Backend>(
    args: &Args,
    app_config: &AppConfig,
    backend: B,
) -> Result<()> {
    let model_name = get_model_name(args, app_config);
    let model_dtype = args.dtype;

    let start_load = std::time::Instant::now();
    let (parameter_values, _parameter_types, config_json, tokenizer, tokenizer_config, total_params) =
        load_model(&model_name, &args.revision, &backend, model_dtype)?;
    let elapsed_load = start_load.elapsed();

    eprintln!(
        "Model weights loaded for {} in {:.2} seconds",
        model_name,
        elapsed_load.as_secs_f64()
    );

    let chat_template = match get_model_chat_template(&model_name, &args.revision) {
        Ok(template) => template,
        Err(err) if args.tool_use => return Err(err.into()),
        Err(_) => String::new(),
    };
    let tool_schemas: Vec<serde_json::Value> = if args.tool_use {
        tools::tool_schemas()
    } else {
        Vec::new()
    };

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
    } else if args.tool_use {
        render_tool_prompt(
            &chat_template,
            &tokenizer_config,
            &args.prompt,
            &tool_schemas,
            args.thinking,
        )?
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

    if !benchmarking && !use_image && !args.tool_use {
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

    // Workaround: remove duplicate BOS token if present, seen in LFM2-VL
    if let Some(bos_token_id) = tokenizer_config
        .get("bos_token")
        .and_then(|bos_token| {
            bos_token
                .as_str()
                .or_else(|| bos_token.get("content").and_then(serde_json::Value::as_str))
        })
        .and_then(|bos_token| tokenizer.token_to_id(bos_token))
    {
        if token_ids.starts_with(&[bos_token_id, bos_token_id]) {
            token_ids.remove(0);
        }
    }

    let max_sequence_length = if args.tool_use {
        token_ids.len() + (2 * max_seq_len) + 256
    } else {
        max_seq_len + token_ids.len()
    };
    let model = get_model(
        &config_json,
        max_sequence_length,
        runtime_context,
        model_dtype,
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

    let module_path = if use_image {
        catgrad::prelude::Path::empty()
    } else {
        model.path()
    };
    let spec = ProgramSpec {
        typed_term: typed_term.clone(),
        module_path,
        empty_state_type: model.empty_state_type(),
        max_sequence_length,
        extra_nat_chunk_size: model.extra_nat_chunk_size(),
    };
    let bound_program =
        Arc::new(BoundProgram::bind(&parameter_values, &backend, Program::from(spec))?);

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
            interpreter::float_tensor(
                &backend,
                Shape(vec![1, mm.mm_tokens_per_image, mm.hidden_size]),
                cached,
                model_dtype,
            )
            .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?
        } else {
            let image_tensor = interpreter::float_tensor(
                &backend,
                Shape(image_shape),
                image_data,
                model_dtype,
            )
            .map_err(|e| anyhow::anyhow!("BackendError: {:?}", e))?;
            let vision_program = Program::from_module(
                vision_model.as_ref(),
                catgrad::prelude::Path::empty(),
                vec![],
                0,
                None,
            )?;
            let bound_vision = std::sync::Arc::new(BoundProgram::bind(
                &parameter_values,
                &backend,
                vision_program,
            )?);
            let snap = bound_vision.empty_snapshot();
            let mut vision_session = std::sync::Arc::clone(&bound_vision).start(snap)?;
            let mut results = vision_session.run(vec![image_tensor])?;
            if results.is_empty() {
                return Err(anyhow::anyhow!("Vision model returned no outputs"));
            }
            let visual_embeddings = results.remove(0);
            let flattened = to_f32_vec(&backend, &visual_embeddings)?;
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

    let use_kv_cache = args.kv_cache || use_image;
    if args.tool_use {
        let (first_text, first_tokens, first_elapsed_pp, first_elapsed_gen) = generate_stream(
            model.as_ref(),
            &bound_program,
            &tokenizer,
            token_ids,
            GenerationConfig {
                max_seq_len: args.max_seq_len,
                max_sequence_length,
                use_kv_cache: true,
                benchmarking: false,
                stream_output: false,
                multimodal_ctx: None,
            },
        )?;
        if let Some(tool_use_step) = model.parse_tool_calls(&first_text)? {
            let mut tool_responses = Vec::with_capacity(tool_use_step.tool_calls.len());
            for tool_call in &tool_use_step.tool_calls {
                let tool_response = tools::execute_tool_call(tool_call)?;
                eprintln!(
                    "Tool {}({}) -> {}",
                    tool_call.name,
                    serde_json::Value::Object(tool_call.arguments.clone()),
                    tool_response
                );
                tool_responses.push(tool_response);
            }

            let follow_up_prompt = render_tool_follow_up_prompt(
                &chat_template,
                &tokenizer_config,
                &args.prompt,
                &tool_schemas,
                &tool_use_step,
                &tool_responses,
                args.thinking,
            )?;
            let follow_up_ids = tokenizer
                .encode(follow_up_prompt.as_str(), true)
                .map_err(|err| anyhow::anyhow!("check error {:?}", err))?
                .get_ids()
                .to_vec();
            let (_, second_tokens, second_elapsed_pp, second_elapsed_gen) = generate_stream(
                model.as_ref(),
                &bound_program,
                &tokenizer,
                follow_up_ids,
                GenerationConfig {
                    max_seq_len: args.max_seq_len,
                    max_sequence_length,
                    use_kv_cache: true,
                    benchmarking: false,
                    stream_output: true,
                    multimodal_ctx: None,
                },
            )?;
            println!();
            let total_tokens = first_tokens + second_tokens;
            let total_elapsed =
                first_elapsed_pp + first_elapsed_gen + second_elapsed_pp + second_elapsed_gen;
            eprintln!(
                "{} tokens generated in {} seconds. ({:.2} tps)",
                total_tokens,
                total_elapsed.as_secs(),
                total_tokens as f64 / total_elapsed.as_secs_f64(),
            );
        } else {
            print!("{first_text}");
            std::io::stdout().flush()?;
            println!();
            let total_elapsed = first_elapsed_pp + first_elapsed_gen;
            eprintln!(
                "{} tokens generated in {} seconds. ({:.2} tps)",
                first_tokens,
                total_elapsed.as_secs(),
                first_tokens as f64 / total_elapsed.as_secs_f64(),
            );
        }
    } else {
        let (_, generated_tokens, elapsed_pp, elapsed_gen) = generate_stream(
            model.as_ref(),
            &bound_program,
            &tokenizer,
            token_ids,
            GenerationConfig {
                max_seq_len,
                max_sequence_length,
                use_kv_cache,
                benchmarking,
                stream_output: !benchmarking,
                multimodal_ctx: multimodal_ctx.as_ref(),
            },
        )?;
        if benchmarking {
            let size_gib = (total_params as f64 * dtype_size_bytes(args.dtype) as f64)
                / (1024.0 * 1024.0 * 1024.0);
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
            interpreter::TaggedVec::F16(v) => Ok(v.into_iter().map(|x| x.to_f32()).collect()),
            interpreter::TaggedVec::BF16(v) => Ok(v.into_iter().map(|x| x.to_f32()).collect()),
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

enum DecodeParameterBundle<'a, B: interpreter::Backend> {
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

struct GenerationConfig<'a, B: interpreter::Backend> {
    max_seq_len: usize,
    max_sequence_length: usize,
    use_kv_cache: bool,
    benchmarking: bool,
    stream_output: bool,
    multimodal_ctx: Option<&'a MultimodalRuntime<B>>,
}

fn generate_stream<B: interpreter::Backend>(
    model: &dyn LLMModel,
    bound_program: &std::sync::Arc<BoundProgram<B>>,
    tokenizer: &tokenizers::Tokenizer,
    mut token_ids: Vec<u32>,
    config: GenerationConfig<'_, B>,
) -> Result<(String, usize, std::time::Duration, std::time::Duration)> {
    let eos_token_ids = model.config().get_eos_token_ids();
    let empty_snapshot = bound_program.empty_snapshot();
    let mut snapshot = empty_snapshot.clone();
    let mut use_image_embeddings = config.multimodal_ctx.is_some();
    let mut output = String::new();
    let mut generated_tokens = 0;
    let mut start_gen = std::time::Instant::now();
    let mut elapsed_pp = std::time::Duration::ZERO;

    for i in 0..config.max_seq_len {
        let decode_inputs = if let Some(ctx) = config.multimodal_ctx {
            DecodeParameterBundle::Multimodal {
                input_tokens: &token_ids,
                hidden_size: ctx.hidden_size,
                image_token_index: ctx.image_token_index,
                visual_embeddings: &ctx.visual_embeddings,
                use_image_embeddings,
            }
        } else {
            DecodeParameterBundle::Text {
                input_tokens: &token_ids,
            }
        };
        let mut session = std::sync::Arc::clone(bound_program).start(snapshot)?;
        let next_token_id = run_session(
            model,
            &**bound_program,
            &mut session,
            decode_inputs,
            config.max_sequence_length,
        )?;
        let next_snapshot = session.into_snapshot();
        if i == 0 {
            elapsed_pp = start_gen.elapsed();
            start_gen = std::time::Instant::now();
        }
        generated_tokens += 1;
        if eos_token_ids.contains(&(next_token_id as i32)) && !config.benchmarking {
            break;
        }
        if config.use_kv_cache {
            snapshot = next_snapshot;
            token_ids = vec![next_token_id];
        } else {
            snapshot = empty_snapshot.clone();
            token_ids.push(next_token_id);
        }
        if config.multimodal_ctx.is_some() && config.use_kv_cache {
            use_image_embeddings = false;
        }
        let decoded_token = tokenizer.decode(&[next_token_id], true).unwrap();
        output.push_str(&decoded_token);
        if config.stream_output {
            print!("{decoded_token}");
            std::io::stdout().flush()?;
        }
    }

    Ok((output, generated_tokens, elapsed_pp, start_gen.elapsed()))
}

fn run_session<B: interpreter::Backend>(
    model: &dyn LLMModel,
    bound_program: &BoundProgram<B>,
    session: &mut Session<B>,
    decode_inputs: DecodeParameterBundle<'_, B>,
    max_sequence_length: usize,
) -> Result<u32> {
    let backend = &bound_program.interpreter().backend;
    let mut inputs = Vec::with_capacity(session.state().len() + 4);
    let input_seq_len;

    match decode_inputs {
        DecodeParameterBundle::Text { input_tokens } => {
            input_seq_len = input_tokens.len();
            inputs.push(token_tensor_b(backend, "input", input_tokens)?);
        }
        DecodeParameterBundle::Multimodal {
            input_tokens,
            hidden_size,
            image_token_index,
            visual_embeddings,
            use_image_embeddings,
        } => {
            input_seq_len = input_tokens.len();
            let empty_image_embeddings = interpreter::float_tensor(
                backend,
                Shape(vec![1, 0, hidden_size]),
                Vec::<f32>::new(),
                model.dtype(),
            )
            .map_err(|err| anyhow::anyhow!("empty image tensor error: {:?}", err))?;

            let (text_before_tokens, text_after_tokens) = if use_image_embeddings {
                split_image_tokens(input_tokens, image_token_index)?
            } else {
                (&[][..], input_tokens)
            };

            let text_before = token_tensor_b(backend, "text_before", text_before_tokens)?;
            let text_after = token_tensor_b(backend, "text_after", text_after_tokens)?;
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

    inputs.extend(session.state().iter().cloned());
    inputs.push(interpreter::Value::Nat(max_sequence_length));
    // Workaround: push extra nat input needed for gated delta decoding, representing chunk_number.
    if let Some(extra_nat) = model.extra_nat_input(input_seq_len) {
        inputs.push(interpreter::Value::Nat(extra_nat));
    }

    let mut outputs = session.run(inputs)?;
    if outputs.is_empty() {
        return Err(anyhow::anyhow!("model returned no outputs"));
    }
    let output = outputs.remove(0);

    match output {
        interpreter::Value::Tensor(arr) => match backend.to_vec(arr) {
            interpreter::TaggedVec::U32(v) => v
                .last()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("token output tensor was empty")),
            _ => Err(anyhow::anyhow!("Unexpected output dtype")),
        },
        t => Err(anyhow::anyhow!("Output was not a tensor: {:?}", t)),
    }
}

fn token_tensor_b<B: interpreter::Backend>(
    backend: &B,
    label: &str,
    input_tokens: &[u32],
) -> Result<interpreter::Value<B>> {
    interpreter::tensor(
        backend,
        Shape(vec![1, input_tokens.len()]),
        input_tokens.to_vec(),
    )
    .map_err(|err| anyhow::anyhow!("{label} tensor error: {:?}", err))
}
