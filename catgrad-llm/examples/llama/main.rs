use anyhow::Result;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::backend::ndarray::NdArrayBackend;
use catgrad::prelude::*;
use clap::{Parser, ValueEnum};
use std::io::Write;
use std::path::PathBuf;

use catgrad_llm::config::LLMConfig;
use catgrad_llm::utils::{get_model, get_model_chat_template, load_model, render_chat_template};

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
    /// Initial prompt
    #[arg(short = 'p', long, default_value = "Category theory is")]
    prompt: String,
    /// Pass raw prompt without chat template
    #[arg(long)]
    raw: bool,
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
    /// Enable Candle backend acceleration
    #[arg(short = 'a', long)]
    accel: bool,
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

/// Construct, shapecheck, and interpret the a given LLM using the selected backend.
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    match args.backend {
        BackendChoice::Ndarray => run_with_backend(&args, NdArrayBackend),
        BackendChoice::Candle => run_with_backend(&args, CandleBackend::new_accel(args.accel)),
    }
}

fn run_with_backend<B: interpreter::Backend>(args: &Args, backend: B) -> Result<()> {
    let (parameter_values, parameter_types, config_json, tokenizer, total_params) =
        load_model(&args.model_name, &args.revision, &backend)?;

    let chat_template =
        get_model_chat_template(&args.model_name, &args.revision).unwrap_or_default();

    let benchmarking = args.bench.is_some();
    let mut pp = 0;
    let mut tg = 0;
    let mut max_seq_len = args.max_seq_len;

    let prompt = if let Some(bench) = &args.bench {
        pp = bench[0];
        tg = bench[1];
        max_seq_len = tg;
        println!(
            "Benchmarking {} with prefill size {} and sequence length {}",
            args.model_name, pp, tg
        );
        "The".repeat(pp)
    } else if chat_template.is_empty() || args.raw {
        args.prompt.clone()
    } else {
        render_chat_template(&chat_template, &args.prompt, false, false)?
    };

    let encoding = tokenizer
        .encode(prompt.clone(), true)
        .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;

    let mut token_ids = encoding.get_ids().to_vec();

    let max_sequence_length = max_seq_len + token_ids.len();
    let (model, config) = get_model(&config_json, max_sequence_length)?;

    let typed_term = if let Some(load_path) = &args.load {
        let file = std::fs::File::open(load_path)?;
        serde_json::from_reader(file)?
    } else {
        model.term().expect("Failed to create typed term")
    };

    if let Some(dump_path) = &args.dump {
        let file = std::fs::File::create(dump_path)?;
        serde_json::to_writer_pretty(file, &typed_term)?;
        println!(
            "Graph for {} and max_seq_length of {max_sequence_length} dumped to {}",
            model.path(),
            dump_path.display()
        );
        return Ok(());
    }

    // Get stdlib environment and extend with parameter declarations
    let mut env = stdlib();
    env.declarations
        .extend(to_load_ops(model.path(), parameter_types.keys()));

    // Shapecheck the model
    if args.typecheck {
        typecheck::check(&env, &parameter_types, typed_term.clone())
            .map_err(|err| anyhow::anyhow!("check error {:?}", err))?;
    }

    let mut generated_tokens = 0;
    if !benchmarking {
        print!("{}", prompt);
    }
    let mut start_gen = std::time::Instant::now();
    let mut elapsed_pp = std::time::Duration::ZERO;
    let interpreter = interpreter::Interpreter::new(backend, env, parameter_values);
    let empty_cache = empty_kv_cache(&interpreter.backend, config.as_ref())?;
    let mut kv_cache = empty_cache.clone();

    // Run inference loop
    for i in 0..max_seq_len {
        let cache = if args.kv_cache {
            &kv_cache
        } else {
            &empty_cache
        };
        let (next_token_id, new_cache) =
            run_interpreter(&typed_term, &interpreter, &token_ids, cache)?;
        if i == 0 {
            elapsed_pp = start_gen.elapsed();
            start_gen = std::time::Instant::now();
        }
        generated_tokens += 1;
        if config.get_eos_token_ids().contains(&(next_token_id as i32)) && !benchmarking {
            break;
        }
        if args.kv_cache {
            kv_cache = new_cache;
            token_ids = vec![next_token_id];
        } else {
            token_ids.push(next_token_id);
        }
        if !benchmarking {
            let decoded_token = tokenizer.decode(&[next_token_id], false).unwrap();
            print!("{}", decoded_token);
            std::io::stdout().flush()?;
        }
    }

    let elapsed_gen = start_gen.elapsed();
    if benchmarking {
        // hardcode size multiplier as 4.0 since we only load in F32
        let size_gib = (total_params as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        let params_m = total_params as f64 / 1_000_000.0;
        let b_str = match args.backend {
            BackendChoice::Ndarray => "Ndarray",
            BackendChoice::Candle => "Candle",
        };

        println!(
            "| model                                    | size       | params     | backend    |            test |                  t/s |"
        );
        println!(
            "| ---------------------------------------- | ---------- | ---------- | ---------- | --------------- | -------------------- |"
        );

        let tps_pp = pp as f64 / elapsed_pp.as_secs_f64();
        println!(
            "| {:<40} | {:>6.2} GiB | {:>8.2} M | {:<10} | {:>15} | {:>20.2} |",
            args.model_name,
            size_gib,
            params_m,
            b_str,
            format!("pp{}", pp),
            tps_pp
        );

        let tps_tg = tg as f64 / elapsed_gen.as_secs_f64();
        println!(
            "| {:<40} | {:>6.2} GiB | {:>8.2} M | {:<10} | {:>15} | {:>20.2} |",
            args.model_name,
            size_gib,
            params_m,
            b_str,
            format!("tg{}", tg),
            tps_tg
        );
    } else {
        println!(
            "\n{} tokens generated in {} seconds. ({:.2} tps)",
            generated_tokens,
            (elapsed_pp + elapsed_gen).as_secs(),
            generated_tokens as f64 / (elapsed_pp + elapsed_gen).as_secs_f64(),
        );
    }
    Ok(())
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

fn run_interpreter<B: interpreter::Backend>(
    typed_term: &TypedTerm,
    interpreter: &interpreter::Interpreter<B>,
    input_data: &[u32],
    kv_cache: &KvCache<B>,
) -> Result<(u32, KvCache<B>)> {
    let input_tensor = interpreter::tensor(
        &interpreter.backend,
        Shape(vec![1, input_data.len()]),
        input_data.to_vec(),
    )
    .expect("Failed to create input tensor");

    // Run the model
    let mut results = interpreter
        .run(
            typed_term.term.clone(),
            vec![input_tensor, kv_cache.0.clone(), kv_cache.1.clone()],
        )
        .expect("Failed to run inference");

    let out_v = results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("No KV cache V output"))?;
    let out_k = results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("No KV cache K output"))?;
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
