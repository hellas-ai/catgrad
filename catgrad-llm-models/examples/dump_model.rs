use anyhow::Result;
use catgrad::prelude::Dtype;
use catgrad_llm_models::utils::get_model;
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

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
    /// Total sequence length encoded into the dumped graph
    #[arg(short = 's', long, default_value_t = 1024)]
    max_seq_len: usize,
    /// Floating-point dtype to use for model weights and activations
    #[arg(long, default_value = "f32", value_parser = parse_model_dtype)]
    dtype: Dtype,
    /// Dump the constructed graph to this JSON file.
    #[arg(long)]
    dump: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct AppConfig {
    aliases: HashMap<String, String>,
}

fn parse_model_dtype(s: &str) -> Result<Dtype, String> {
    let dtype: Dtype = s.parse()?;
    match dtype {
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => Ok(dtype),
        Dtype::F8 | Dtype::U32 => Err("model dtype must be f32, f16, or bf16".to_string()),
    }
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

fn get_app_config(args: &Args) -> Result<AppConfig> {
    let default_config = include_str!("../../catgrad-llm/examples/llama/llm_config.default.toml");
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

fn build_hf_api() -> Result<hf_hub::api::sync::Api> {
    let mut builder = ApiBuilder::from_env();
    let env_token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty());
    if let Some(token) = env_token {
        builder = builder.with_token(Some(token));
    }
    Ok(builder.build()?)
}

fn load_model_config(model: &str, revision: &str) -> Result<serde_json::Value> {
    let api = build_hf_api()?;
    let repo = api.repo(Repo::with_revision(
        model.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let config_path = repo.get("config.json")?;
    Ok(serde_json::from_reader(File::open(config_path)?)?)
}

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

    let dump_path = args
        .dump
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--dump is required unless --list-models"))?;
    let model_name = get_model_name(&args, &app_config);
    let config_json = load_model_config(&model_name, &args.revision)?;
    let model = get_model(&config_json, args.max_seq_len, None, args.dtype)?;
    let typed_term = model
        .term()
        .ok_or_else(|| anyhow::anyhow!("Failed to create typed term"))?;

    serde_json::to_writer_pretty(File::create(dump_path)?, &typed_term)?;
    eprintln!(
        "Graph for {} and max_seq_len of {} dumped to {}",
        model.path(),
        args.max_seq_len,
        dump_path.display()
    );
    Ok(())
}
