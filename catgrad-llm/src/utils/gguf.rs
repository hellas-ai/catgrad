use candle_core::{
    DType, Tensor,
    quantized::{GgmlDType, gguf_file, tokenizer::TokenizerFromGguf},
};
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::prelude::Dtype;
use catgrad::prelude::path;
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::tokenizer::Tokenizer;

use crate::{LLMError, Result};

fn sanitize_chat_template(chat_template: String) -> String {
    chat_template
        .replace("{% generation %}", "")
        .replace("{%- generation -%}", "")
        .replace("{% endgeneration %}", "")
        .replace("{%- endgeneration -%}", "")
}

fn gguf_metadata_value(
    content: &gguf_file::Content,
    key: impl AsRef<str>,
) -> Result<&gguf_file::Value> {
    let key = key.as_ref();
    content
        .metadata
        .get(key)
        .ok_or(LLMError::InvalidModelConfig(format!(
            "missing GGUF metadata key `{key}`"
        )))
}

fn gguf_value_to_u64(value: &gguf_file::Value) -> Result<u64> {
    value
        .to_u64()
        .or_else(|_| value.to_u32().map(u64::from))
        .or_else(|_| value.to_i32().map(|v| v as u64))
        .or_else(|_| value.to_i64().map(|v| v as u64))
        .map_err(|_| {
            LLMError::InvalidModelConfig(format!(
                "expected integer GGUF metadata value, got {value:?}"
            ))
        })
}

fn gguf_value_to_f32(value: &gguf_file::Value) -> Result<f32> {
    value
        .to_f32()
        .or_else(|_| value.to_f64().map(|v| v as f32))
        .map_err(|_| {
            LLMError::InvalidModelConfig(format!(
                "expected floating-point GGUF metadata value, got {value:?}"
            ))
        })
}

fn gguf_value_to_usizes(value: &gguf_file::Value) -> Result<Vec<usize>> {
    value
        .to_vec()
        .map_err(|_| {
            LLMError::InvalidModelConfig(format!("expected GGUF metadata array, got {value:?}"))
        })?
        .iter()
        .map(|entry| gguf_value_to_u64(entry).map(|value| value as usize))
        .collect()
}

fn gguf_architecture(content: &gguf_file::Content) -> Result<String> {
    gguf_metadata_value(content, "general.architecture")?
        .to_string()
        .cloned()
        .map_err(|_| {
            LLMError::InvalidModelConfig(
                "GGUF general.architecture metadata is not a string".to_string(),
            )
        })
}

fn gguf_tokens(content: &gguf_file::Content) -> Result<Vec<String>> {
    gguf_metadata_value(content, "tokenizer.ggml.tokens")?
        .to_vec()
        .map_err(|_| {
            LLMError::InvalidModelConfig(
                "GGUF tokenizer tokens metadata is not an array".to_string(),
            )
        })?
        .iter()
        .map(|value| {
            value.to_string().cloned().map_err(|_| {
                LLMError::InvalidModelConfig(
                    "GGUF tokenizer tokens metadata contains a non-string entry".to_string(),
                )
            })
        })
        .collect()
}

fn gguf_token_id(content: &gguf_file::Content, token: &str) -> Result<Option<u64>> {
    Ok(gguf_tokens(content)?
        .iter()
        .position(|candidate| candidate == token)
        .map(|id| id as u64))
}

fn gguf_context_length(content: &gguf_file::Content, architecture: &str) -> Result<Option<u64>> {
    content
        .metadata
        .get(&format!("{architecture}.context_length"))
        .map(gguf_value_to_u64)
        .transpose()
}

fn gguf_name(content: &gguf_file::Content) -> Result<Option<String>> {
    content
        .metadata
        .get("general.name")
        .map(|value| {
            value.to_string().cloned().map_err(|_| {
                LLMError::InvalidModelConfig(
                    "GGUF general.name metadata is not a string".to_string(),
                )
            })
        })
        .transpose()
}

fn gguf_multiple_eos_token_ids(content: &gguf_file::Content, tokens: &[&str]) -> Result<Vec<u64>> {
    let mut eos_token_ids = Vec::new();
    for token in tokens {
        if let Some(token_id) = gguf_token_id(content, token)?
            && !eos_token_ids.contains(&token_id)
        {
            eos_token_ids.push(token_id);
        }
    }
    Ok(eos_token_ids)
}

fn gguf_supported_quantized_dtype(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q8_0
            | GgmlDType::Q6K
            | GgmlDType::Q5K
            | GgmlDType::Q4K
            | GgmlDType::Q3K
            | GgmlDType::Q5_1
            | GgmlDType::Q5_0
            | GgmlDType::Q4_1
            | GgmlDType::Q4_0
    )
}

fn gguf_is_quantized_dtype(dtype: GgmlDType) -> bool {
    gguf_supported_quantized_dtype(dtype)
}

fn gguf_has_quantized_tied_output(content: &gguf_file::Content) -> bool {
    !content.tensor_infos.contains_key("output.weight")
        && content
            .tensor_infos
            .get("token_embd.weight")
            .is_some_and(|tensor| gguf_is_quantized_dtype(tensor.ggml_dtype))
}

fn gguf_tie_word_embeddings(content: &gguf_file::Content) -> bool {
    !content.tensor_infos.contains_key("output.weight") && !gguf_has_quantized_tied_output(content)
}

pub fn build_gguf_tokenizer(content: &gguf_file::Content) -> Result<Tokenizer> {
    if let Some(hf_json) = content.metadata.get("tokenizer.huggingface.json") {
        let json = hf_json.to_string().map_err(|_| {
            LLMError::InvalidModelConfig(
                "GGUF tokenizer.huggingface.json metadata is not a string".to_string(),
            )
        })?;
        return super::from_json_str(json);
    }

    let tokenizer = <Tokenizer as TokenizerFromGguf>::from_gguf(content)
        .map_err(|err| LLMError::TokenizerError(format!("GGUF tokenizer load error: {err}")))?;
    let tokenizer_json = serde_json::to_string(&tokenizer).map_err(|err| {
        LLMError::TokenizerError(format!("failed to serialize GGUF tokenizer: {err}"))
    })?;
    super::from_json_str(&tokenizer_json)
}

pub fn build_gguf_tokenizer_config_json(content: &gguf_file::Content) -> Result<serde_json::Value> {
    let tokens = gguf_tokens(content)?;
    let bos_token = content
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .map(gguf_value_to_u64)
        .transpose()?
        .and_then(|id| tokens.get(id as usize).cloned());
    let chat_template = content
        .metadata
        .get("tokenizer.chat_template")
        .map(|value| {
            value.to_string().cloned().map_err(|_| {
                LLMError::InvalidModelConfig(
                    "GGUF tokenizer.chat_template metadata is not a string".to_string(),
                )
            })
        })
        .transpose()?;

    let mut map = serde_json::Map::new();
    if let Some(bos_token) = bos_token {
        map.insert(
            "bos_token".to_string(),
            serde_json::Value::String(bos_token),
        );
    }
    if let Some(chat_template) = chat_template {
        map.insert(
            "chat_template".to_string(),
            serde_json::Value::String(sanitize_chat_template(chat_template)),
        );
    }
    Ok(serde_json::Value::Object(map))
}

pub fn gguf_chat_template(content: &gguf_file::Content) -> Result<String> {
    let Some(value) = content.metadata.get("tokenizer.chat_template") else {
        return Ok(String::new());
    };
    let chat_template = value.to_string().map_err(|_| {
        LLMError::InvalidModelConfig(
            "GGUF tokenizer.chat_template metadata is not a string".to_string(),
        )
    })?;
    Ok(sanitize_chat_template(chat_template.clone()))
}

pub fn build_llama_config_from_gguf(content: &gguf_file::Content) -> Result<serde_json::Value> {
    let architecture = gguf_architecture(content)?;
    if architecture.as_str() != "llama" {
        return Err(LLMError::UnsupportedModel(format!(
            "GGUF architecture {architecture}"
        )));
    }

    let hidden_size = gguf_value_to_u64(gguf_metadata_value(content, "llama.embedding_length")?)?;
    let intermediate_size =
        gguf_value_to_u64(gguf_metadata_value(content, "llama.feed_forward_length")?)?;
    let num_hidden_layers = gguf_value_to_u64(gguf_metadata_value(content, "llama.block_count")?)?;
    let num_attention_heads =
        gguf_value_to_u64(gguf_metadata_value(content, "llama.attention.head_count")?)?;
    let num_key_value_heads = content
        .metadata
        .get("llama.attention.head_count_kv")
        .map(gguf_value_to_u64)
        .transpose()?
        .unwrap_or(num_attention_heads);
    let max_position_embeddings = gguf_context_length(content, "llama")?;
    let rope_theta = content
        .metadata
        .get("llama.rope.freq_base")
        .map(gguf_value_to_f32)
        .transpose()?
        .unwrap_or(10000.0);
    let rms_norm_eps = gguf_value_to_f32(gguf_metadata_value(
        content,
        "llama.attention.layer_norm_rms_epsilon",
    )?)?;
    let vocab_size = content
        .metadata
        .get("llama.vocab_size")
        .map(gguf_value_to_u64)
        .transpose()?
        .unwrap_or(gguf_tokens(content)?.len() as u64);
    let eos_token_ids = {
        let eos_token_ids =
            gguf_multiple_eos_token_ids(content, &["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"])?;
        if eos_token_ids.is_empty() {
            content
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .map(gguf_value_to_u64)
                .transpose()?
                .into_iter()
                .collect()
        } else {
            eos_token_ids
        }
    };
    let bos_token_id = content
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .map(gguf_value_to_u64)
        .transpose()?;
    let pad_token_id = content
        .metadata
        .get("tokenizer.ggml.padding_token_id")
        .map(gguf_value_to_u64)
        .transpose()?;
    let tie_word_embeddings = gguf_tie_word_embeddings(content);
    let partial_rotary_factor = content
        .metadata
        .get("llama.rope.dimension_count")
        .map(gguf_value_to_u64)
        .transpose()?
        .map(|rotary_dim| rotary_dim as f64 / (hidden_size as f64 / num_attention_heads as f64))
        .unwrap_or(1.0);
    let rope_scaling = max_position_embeddings
        .filter(|&context_length| context_length > 8192)
        .and_then(|context_length| {
            let model_name = gguf_name(content).ok().flatten().unwrap_or_default();
            model_name.contains("Llama-3").then(|| {
                serde_json::json!({
                    "factor": (context_length as f32) / 8192.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                })
            })
        });

    let mut map = serde_json::Map::new();
    map.insert(
        "architectures".to_string(),
        serde_json::json!(["LlamaForCausalLM"]),
    );
    map.insert("model_type".to_string(), serde_json::json!("llama"));
    map.insert("hidden_size".to_string(), serde_json::json!(hidden_size));
    map.insert(
        "intermediate_size".to_string(),
        serde_json::json!(intermediate_size),
    );
    map.insert(
        "num_hidden_layers".to_string(),
        serde_json::json!(num_hidden_layers),
    );
    map.insert(
        "num_attention_heads".to_string(),
        serde_json::json!(num_attention_heads),
    );
    map.insert(
        "num_key_value_heads".to_string(),
        serde_json::json!(num_key_value_heads),
    );
    map.insert("rope_theta".to_string(), serde_json::json!(rope_theta));
    map.insert("rms_norm_eps".to_string(), serde_json::json!(rms_norm_eps));
    map.insert(
        "tie_word_embeddings".to_string(),
        serde_json::json!(tie_word_embeddings),
    );
    map.insert("vocab_size".to_string(), serde_json::json!(vocab_size));
    map.insert(
        "partial_rotary_factor".to_string(),
        serde_json::json!(partial_rotary_factor),
    );
    if let Some(max_position_embeddings) = max_position_embeddings {
        map.insert(
            "max_position_embeddings".to_string(),
            serde_json::json!(max_position_embeddings),
        );
    }
    if let Some(rope_scaling) = rope_scaling {
        map.insert("rope_scaling".to_string(), rope_scaling);
    }
    if let Some(bos_token_id) = bos_token_id {
        map.insert("bos_token_id".to_string(), serde_json::json!(bos_token_id));
    }
    if let Some(pad_token_id) = pad_token_id {
        map.insert("pad_token_id".to_string(), serde_json::json!(pad_token_id));
    }
    match eos_token_ids.as_slice() {
        [] => {}
        [eos_token_id] => {
            map.insert("eos_token_id".to_string(), serde_json::json!(eos_token_id));
        }
        _ => {
            map.insert("eos_token_id".to_string(), serde_json::json!(eos_token_ids));
        }
    }
    Ok(serde_json::Value::Object(map))
}

pub fn build_qwen35_config_from_gguf(content: &gguf_file::Content) -> Result<serde_json::Value> {
    let architecture = gguf_architecture(content)?;
    if architecture.as_str() != "qwen35" {
        return Err(LLMError::UnsupportedModel(format!(
            "GGUF architecture {architecture}"
        )));
    }

    let hidden_size = gguf_value_to_u64(gguf_metadata_value(content, "qwen35.embedding_length")?)?;
    let intermediate_size =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.feed_forward_length")?)?;
    let num_hidden_layers = gguf_value_to_u64(gguf_metadata_value(content, "qwen35.block_count")?)?;
    let num_attention_heads =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.attention.head_count")?)?;
    let num_key_value_heads = gguf_value_to_u64(gguf_metadata_value(
        content,
        "qwen35.attention.head_count_kv",
    )?)?;
    let head_dim = gguf_value_to_u64(gguf_metadata_value(content, "qwen35.attention.key_length")?)?;
    let rope_theta = gguf_value_to_f32(gguf_metadata_value(content, "qwen35.rope.freq_base")?)?;
    let rope_sections = gguf_value_to_usizes(gguf_metadata_value(
        content,
        "qwen35.rope.dimension_sections",
    )?)?;
    let rms_norm_eps = gguf_value_to_f32(gguf_metadata_value(
        content,
        "qwen35.attention.layer_norm_rms_epsilon",
    )?)?;
    let vocab_size = content
        .metadata
        .get("qwen35.vocab_size")
        .map(gguf_value_to_u64)
        .transpose()?
        .unwrap_or(gguf_tokens(content)?.len() as u64);
    let tie_word_embeddings = gguf_tie_word_embeddings(content);
    let max_position_embeddings = gguf_context_length(content, "qwen35")?;
    let full_attention_interval = content
        .metadata
        .get("qwen35.full_attention_interval")
        .map(gguf_value_to_u64)
        .transpose()?
        .unwrap_or(4);
    let linear_conv_kernel_dim =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.conv_kernel")?)?;
    let linear_key_head_dim =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.state_size")?)?;
    let linear_num_key_heads =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.group_count")?)?;
    let linear_value_dim =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.inner_size")?)?;
    let linear_num_value_heads =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.time_step_rank")?)?;
    let linear_value_head_dim = linear_value_dim / linear_num_value_heads;
    let partial_rotary_factor = content
        .metadata
        .get("qwen35.rope.dimension_count")
        .map(gguf_value_to_u64)
        .transpose()?
        .map(|rotary_dim| rotary_dim as f64 / head_dim as f64)
        .unwrap_or(0.25);
    let eos_token_ids = {
        let mut eos_token_ids = Vec::new();
        if let Some(eos_token_id) = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .map(gguf_value_to_u64)
            .transpose()?
        {
            eos_token_ids.push(eos_token_id);
        }
        if let Some(end_of_text_id) = gguf_token_id(content, "<|endoftext|>")?
            && !eos_token_ids.contains(&end_of_text_id)
        {
            eos_token_ids.push(end_of_text_id);
        }
        eos_token_ids
    };
    let layer_types: Vec<_> = (0..num_hidden_layers)
        .map(|layer_id| {
            if ((layer_id + 1) % full_attention_interval) == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();

    let mut rope_parameters = serde_json::Map::new();
    rope_parameters.insert("rope_type".to_string(), serde_json::json!("default"));
    rope_parameters.insert("rope_theta".to_string(), serde_json::json!(rope_theta));
    rope_parameters.insert(
        "partial_rotary_factor".to_string(),
        serde_json::json!(partial_rotary_factor),
    );
    rope_parameters.insert(
        "mrope_section".to_string(),
        serde_json::json!(rope_sections),
    );
    rope_parameters.insert("mrope_interleaved".to_string(), serde_json::json!(true));

    let mut text_config = serde_json::Map::new();
    text_config.insert("hidden_size".to_string(), serde_json::json!(hidden_size));
    text_config.insert(
        "intermediate_size".to_string(),
        serde_json::json!(intermediate_size),
    );
    text_config.insert(
        "num_hidden_layers".to_string(),
        serde_json::json!(num_hidden_layers),
    );
    text_config.insert(
        "num_attention_heads".to_string(),
        serde_json::json!(num_attention_heads),
    );
    text_config.insert(
        "num_key_value_heads".to_string(),
        serde_json::json!(num_key_value_heads),
    );
    text_config.insert("head_dim".to_string(), serde_json::json!(head_dim));
    text_config.insert("layer_types".to_string(), serde_json::json!(layer_types));
    text_config.insert(
        "full_attention_interval".to_string(),
        serde_json::json!(full_attention_interval),
    );
    text_config.insert(
        "linear_conv_kernel_dim".to_string(),
        serde_json::json!(linear_conv_kernel_dim),
    );
    text_config.insert(
        "linear_key_head_dim".to_string(),
        serde_json::json!(linear_key_head_dim),
    );
    text_config.insert(
        "linear_num_key_heads".to_string(),
        serde_json::json!(linear_num_key_heads),
    );
    text_config.insert(
        "linear_value_head_dim".to_string(),
        serde_json::json!(linear_value_head_dim),
    );
    text_config.insert(
        "linear_num_value_heads".to_string(),
        serde_json::json!(linear_num_value_heads),
    );
    text_config.insert("rms_norm_eps".to_string(), serde_json::json!(rms_norm_eps));
    text_config.insert(
        "tie_word_embeddings".to_string(),
        serde_json::json!(tie_word_embeddings),
    );
    text_config.insert("vocab_size".to_string(), serde_json::json!(vocab_size));
    text_config.insert("model_type".to_string(), serde_json::json!("qwen3_5_text"));
    text_config.insert(
        "rope_parameters".to_string(),
        serde_json::Value::Object(rope_parameters),
    );
    if let Some(max_position_embeddings) = max_position_embeddings {
        text_config.insert(
            "max_position_embeddings".to_string(),
            serde_json::json!(max_position_embeddings),
        );
    }
    match eos_token_ids.as_slice() {
        [] => {}
        [eos_token_id] => {
            text_config.insert("eos_token_id".to_string(), serde_json::json!(eos_token_id));
        }
        _ => {
            text_config.insert("eos_token_id".to_string(), serde_json::json!(eos_token_ids));
        }
    }

    let mut map = serde_json::Map::new();
    map.insert(
        "architectures".to_string(),
        serde_json::json!(["Qwen3_5ForConditionalGeneration"]),
    );
    map.insert("model_type".to_string(), serde_json::json!("qwen3_5"));
    map.insert(
        "tie_word_embeddings".to_string(),
        serde_json::json!(tie_word_embeddings),
    );
    map.insert(
        "text_config".to_string(),
        serde_json::Value::Object(text_config),
    );
    Ok(serde_json::Value::Object(map))
}

pub fn build_granite_config_from_gguf(content: &gguf_file::Content) -> Result<serde_json::Value> {
    let architecture = gguf_architecture(content)?;
    if architecture.as_str() != "granite" {
        return Err(LLMError::UnsupportedModel(format!(
            "GGUF architecture {architecture}"
        )));
    }

    let hidden_size = gguf_value_to_u64(gguf_metadata_value(content, "granite.embedding_length")?)?;
    let intermediate_size =
        gguf_value_to_u64(gguf_metadata_value(content, "granite.feed_forward_length")?)?;
    let num_hidden_layers =
        gguf_value_to_u64(gguf_metadata_value(content, "granite.block_count")?)?;
    let num_attention_heads = gguf_value_to_u64(gguf_metadata_value(
        content,
        "granite.attention.head_count",
    )?)?;
    let num_key_value_heads = gguf_value_to_u64(gguf_metadata_value(
        content,
        "granite.attention.head_count_kv",
    )?)?;
    let attention_multiplier =
        gguf_value_to_f32(gguf_metadata_value(content, "granite.attention.scale")?)?;
    let embedding_multiplier =
        gguf_value_to_f32(gguf_metadata_value(content, "granite.embedding_scale")?)?;
    let residual_multiplier =
        gguf_value_to_f32(gguf_metadata_value(content, "granite.residual_scale")?)?;
    let rope_theta = gguf_value_to_f32(gguf_metadata_value(content, "granite.rope.freq_base")?)?;
    let rms_norm_eps = gguf_value_to_f32(gguf_metadata_value(
        content,
        "granite.attention.layer_norm_rms_epsilon",
    )?)?;
    let vocab_size = content
        .metadata
        .get("granite.vocab_size")
        .map(gguf_value_to_u64)
        .transpose()?
        .unwrap_or(gguf_tokens(content)?.len() as u64);
    let eos_token_id = content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .map(gguf_value_to_u64)
        .transpose()?;
    let partial_rotary_factor = content
        .metadata
        .get("granite.rope.dimension_count")
        .map(gguf_value_to_u64)
        .transpose()?
        .map(|rotary_dim| rotary_dim as f64 / (hidden_size as f64 / num_attention_heads as f64))
        .unwrap_or(1.0);
    let max_position_embeddings = gguf_context_length(content, "granite")?;

    let mut map = serde_json::Map::new();
    map.insert(
        "architectures".to_string(),
        serde_json::json!(["GraniteForCausalLM"]),
    );
    map.insert("model_type".to_string(), serde_json::json!("granite"));
    map.insert("hidden_size".to_string(), serde_json::json!(hidden_size));
    map.insert(
        "intermediate_size".to_string(),
        serde_json::json!(intermediate_size),
    );
    map.insert(
        "num_hidden_layers".to_string(),
        serde_json::json!(num_hidden_layers),
    );
    map.insert(
        "num_attention_heads".to_string(),
        serde_json::json!(num_attention_heads),
    );
    map.insert(
        "num_key_value_heads".to_string(),
        serde_json::json!(num_key_value_heads),
    );
    map.insert(
        "attention_multiplier".to_string(),
        serde_json::json!(attention_multiplier),
    );
    map.insert(
        "embedding_multiplier".to_string(),
        serde_json::json!(embedding_multiplier),
    );
    map.insert(
        "residual_multiplier".to_string(),
        serde_json::json!(residual_multiplier),
    );
    map.insert("rope_theta".to_string(), serde_json::json!(rope_theta));
    map.insert("rms_norm_eps".to_string(), serde_json::json!(rms_norm_eps));
    map.insert(
        "partial_rotary_factor".to_string(),
        serde_json::json!(partial_rotary_factor),
    );
    map.insert(
        "tie_word_embeddings".to_string(),
        serde_json::json!(gguf_tie_word_embeddings(content)),
    );
    map.insert("vocab_size".to_string(), serde_json::json!(vocab_size));
    if let Some(max_position_embeddings) = max_position_embeddings {
        map.insert(
            "max_position_embeddings".to_string(),
            serde_json::json!(max_position_embeddings),
        );
    }
    if let Some(eos_token_id) = eos_token_id {
        map.insert("eos_token_id".to_string(), serde_json::json!(eos_token_id));
    }
    Ok(serde_json::Value::Object(map))
}

fn gguf_path(name: &str, components: Vec<&str>) -> Result<catgrad::prelude::Path> {
    path(components).map_err(|err| {
        LLMError::InvalidModelConfig(format!("invalid GGUF param path for `{name}`: {}", err.0))
    })
}

fn map_llama_gguf_tensor_name(name: &str) -> Result<Option<(catgrad::prelude::Path, bool)>> {
    if name == "rope_freqs.weight" {
        return Ok(None);
    }
    if name == "token_embd.weight" {
        return Ok(Some((
            gguf_path(name, vec!["model", "embed_tokens", "weight"])?,
            false,
        )));
    }
    if name == "output_norm.weight" {
        return Ok(Some((
            gguf_path(name, vec!["model", "norm", "weight"])?,
            false,
        )));
    }
    if name == "output.weight" {
        return Ok(Some((gguf_path(name, vec!["lm_head", "weight"])?, false)));
    }

    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() != 4 || parts[0] != "blk" {
        return Err(LLMError::InvalidModelConfig(format!(
            "unsupported GGUF llama tensor name `{name}`"
        )));
    }
    let layer = parts[1];
    let suffix = match (parts[2], parts[3]) {
        ("attn_norm", "weight") => vec!["input_layernorm", "weight"],
        ("ffn_norm", "weight") => vec!["post_attention_layernorm", "weight"],
        ("attn_q", "weight") => vec!["self_attn", "q_proj", "weight"],
        ("attn_k", "weight") => vec!["self_attn", "k_proj", "weight"],
        ("attn_v", "weight") => vec!["self_attn", "v_proj", "weight"],
        ("attn_output", "weight") => vec!["self_attn", "o_proj", "weight"],
        ("ffn_gate", "weight") => vec!["mlp", "gate_proj", "weight"],
        ("ffn_up", "weight") => vec!["mlp", "up_proj", "weight"],
        ("ffn_down", "weight") => vec!["mlp", "down_proj", "weight"],
        _ => {
            return Err(LLMError::InvalidModelConfig(format!(
                "unsupported GGUF llama tensor name `{name}`"
            )));
        }
    };

    let mut components = vec!["model", "layers", layer];
    components.extend(suffix);
    Ok(Some((gguf_path(name, components)?, false)))
}

fn map_qwen35_gguf_tensor_name(name: &str) -> Result<Option<(catgrad::prelude::Path, bool)>> {
    if name == "rope_freqs.weight" {
        return Ok(None);
    }
    if name == "token_embd.weight" {
        return Ok(Some((
            gguf_path(
                name,
                vec!["model", "language_model", "embed_tokens", "weight"],
            )?,
            false,
        )));
    }
    if name == "output_norm.weight" {
        return Ok(Some((
            gguf_path(name, vec!["model", "language_model", "norm", "weight"])?,
            false,
        )));
    }
    if name == "output.weight" {
        return Ok(Some((gguf_path(name, vec!["lm_head", "weight"])?, false)));
    }

    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 3 || parts[0] != "blk" {
        return Err(LLMError::InvalidModelConfig(format!(
            "unsupported GGUF qwen35 tensor name `{name}`"
        )));
    }
    let layer = parts[1];
    let suffix = match parts[2..].as_ref() {
        ["attn_norm", "weight"] => vec!["input_layernorm", "weight"],
        ["post_attention_norm", "weight"] => vec!["post_attention_layernorm", "weight"],
        ["ffn_gate", "weight"] => vec!["mlp", "gate_proj", "weight"],
        ["ffn_up", "weight"] => vec!["mlp", "up_proj", "weight"],
        ["ffn_down", "weight"] => vec!["mlp", "down_proj", "weight"],
        ["attn_q", "weight"] => vec!["self_attn", "q_proj", "weight"],
        ["attn_k", "weight"] => vec!["self_attn", "k_proj", "weight"],
        ["attn_v", "weight"] => vec!["self_attn", "v_proj", "weight"],
        ["attn_output", "weight"] => vec!["self_attn", "o_proj", "weight"],
        ["attn_q_norm", "weight"] => vec!["self_attn", "q_norm", "weight"],
        ["attn_k_norm", "weight"] => vec!["self_attn", "k_norm", "weight"],
        ["attn_qkv", "weight"] => vec!["linear_attn", "in_proj_qkv", "weight"],
        ["attn_gate", "weight"] => vec!["linear_attn", "in_proj_z", "weight"],
        ["ssm_alpha", "weight"] => vec!["linear_attn", "in_proj_a", "weight"],
        ["ssm_beta", "weight"] => vec!["linear_attn", "in_proj_b", "weight"],
        ["ssm_dt", "bias"] => vec!["linear_attn", "dt_bias"],
        ["ssm_a"] => vec!["linear_attn", "A_log"],
        ["ssm_norm", "weight"] => vec!["linear_attn", "norm", "weight"],
        ["ssm_out", "weight"] => vec!["linear_attn", "out_proj", "weight"],
        ["ssm_conv1d", "weight"] => vec!["linear_attn", "conv1d", "weight"],
        _ => {
            return Err(LLMError::InvalidModelConfig(format!(
                "unsupported GGUF qwen35 tensor name `{name}`"
            )));
        }
    };

    let mut components = vec!["model", "language_model", "layers", layer];
    components.extend(suffix);
    Ok(Some((gguf_path(name, components)?, false)))
}

fn qwen35_gguf_uses_gemma_rmsnorm_offset(name: &str) -> bool {
    name == "output_norm.weight"
        || name.ends_with("attn_norm.weight")
        || name.ends_with("post_attention_norm.weight")
        || name.ends_with("attn_q_norm.weight")
        || name.ends_with("attn_k_norm.weight")
}

#[derive(Clone, Copy, Debug)]
struct Qwen35LinearAttentionLayout {
    key_dim: usize,
    value_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    num_v_per_k: usize,
    head_v_dim: usize,
}

fn qwen35_linear_attention_layout(
    content: &gguf_file::Content,
) -> Result<Option<Qwen35LinearAttentionLayout>> {
    if gguf_architecture(content)?.as_str() != "qwen35" {
        return Ok(None);
    }

    let head_k_dim = gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.state_size")?)?;
    let num_k_heads = gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.group_count")?)?;
    let value_dim = gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.inner_size")?)?;
    let num_v_heads =
        gguf_value_to_u64(gguf_metadata_value(content, "qwen35.ssm.time_step_rank")?)?;
    let num_v_per_k = num_v_heads / num_k_heads;

    Ok(Some(Qwen35LinearAttentionLayout {
        key_dim: (head_k_dim * num_k_heads) as usize,
        value_dim: value_dim as usize,
        num_k_heads: num_k_heads as usize,
        num_v_heads: num_v_heads as usize,
        num_v_per_k: num_v_per_k as usize,
        head_v_dim: (value_dim / num_v_heads) as usize,
    }))
}

fn qwen35_needs_grouped_v_head_layout_fixup(
    layout: Option<Qwen35LinearAttentionLayout>,
    name: &str,
) -> bool {
    let Some(layout) = layout else {
        return false;
    };
    if layout.num_k_heads == layout.num_v_heads {
        return false;
    }
    name.ends_with("attn_qkv.weight")
        || name.ends_with("attn_gate.weight")
        || name.ends_with("ssm_alpha.weight")
        || name.ends_with("ssm_beta.weight")
        || name.ends_with("ssm_dt.bias")
        || name.ends_with("ssm_a")
        || name.ends_with("ssm_conv1d.weight")
        || name.ends_with("ssm_out.weight")
}

fn qwen35_inverse_reorder_v_rows(
    tensor: Tensor,
    layout: Qwen35LinearAttentionLayout,
    head_dim: usize,
    name: &str,
) -> Result<Tensor> {
    let dims = tensor.dims().to_vec();
    let [rows, rest @ ..] = dims.as_slice() else {
        return Err(LLMError::InvalidModelConfig(format!(
            "expected qwen35 tensor `{name}` to have rank >= 1, got shape {dims:?}"
        )));
    };
    let expected_rows = layout.num_v_heads * head_dim;
    if *rows != expected_rows {
        return Err(LLMError::InvalidModelConfig(format!(
            "unexpected qwen35 tensor `{name}` shape {dims:?}, expected leading dim {expected_rows}"
        )));
    }

    let mut shape = vec![layout.num_v_per_k, layout.num_k_heads, head_dim];
    shape.extend_from_slice(rest);
    let mut flat_shape = vec![expected_rows];
    flat_shape.extend_from_slice(rest);

    tensor
        .reshape(&*shape)
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to reshape qwen35 tensor `{name}` for grouped V-head layout: {err}"
            ))
        })?
        .transpose(0, 1)
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to transpose qwen35 tensor `{name}` for grouped V-head layout: {err}"
            ))
        })?
        .contiguous()
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to make qwen35 tensor `{name}` contiguous after grouped V-head layout fix: {err}"
            ))
        })?
        .reshape(&*flat_shape)
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to restore qwen35 tensor `{name}` shape after grouped V-head layout fix: {err}"
            ))
        })
}

fn qwen35_inverse_reorder_out_proj_columns(
    tensor: Tensor,
    layout: Qwen35LinearAttentionLayout,
    name: &str,
) -> Result<Tensor> {
    let dims = tensor.dims().to_vec();
    let [rows, cols] = dims.as_slice() else {
        return Err(LLMError::InvalidModelConfig(format!(
            "expected qwen35 out_proj tensor `{name}` to be rank-2, got shape {dims:?}"
        )));
    };
    if *cols != layout.value_dim {
        return Err(LLMError::InvalidModelConfig(format!(
            "unexpected qwen35 out_proj tensor `{name}` shape {dims:?}, expected trailing dim {}",
            layout.value_dim
        )));
    }

    tensor
        .reshape((
            *rows,
            layout.num_v_per_k,
            layout.num_k_heads,
            layout.head_v_dim,
        ))
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to reshape qwen35 tensor `{name}` for out_proj layout fix: {err}"
            ))
        })?
        .transpose(1, 2)
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to transpose qwen35 tensor `{name}` for out_proj layout fix: {err}"
            ))
        })?
        .contiguous()
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to make qwen35 tensor `{name}` contiguous after out_proj layout fix: {err}"
            ))
        })?
        .reshape((*rows, *cols))
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to restore qwen35 tensor `{name}` shape after out_proj layout fix: {err}"
            ))
        })
}

fn qwen35_inverse_grouped_v_head_layout(
    tensor: Tensor,
    layout: Qwen35LinearAttentionLayout,
    name: &str,
) -> Result<Tensor> {
    if name.ends_with("attn_qkv.weight") {
        let q = tensor.narrow(0, 0, layout.key_dim).map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to split qwen35 tensor `{name}` q rows: {err}"
            ))
        })?;
        let k = tensor
            .narrow(0, layout.key_dim, layout.key_dim)
            .map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to split qwen35 tensor `{name}` k rows: {err}"
                ))
            })?;
        let v = tensor
            .narrow(0, layout.key_dim * 2, layout.value_dim)
            .map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to split qwen35 tensor `{name}` v rows: {err}"
                ))
            })?;
        let v = qwen35_inverse_reorder_v_rows(v, layout, layout.head_v_dim, name)?;
        return Tensor::cat(&[q, k, v], 0).map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to reassemble qwen35 tensor `{name}` after grouped V-head layout fix: {err}"
            ))
        });
    }
    if name.ends_with("attn_gate.weight") {
        return qwen35_inverse_reorder_v_rows(tensor, layout, layout.head_v_dim, name);
    }
    if name.ends_with("ssm_alpha.weight") || name.ends_with("ssm_beta.weight") {
        return qwen35_inverse_reorder_v_rows(tensor, layout, 1, name);
    }
    if name.ends_with("ssm_dt.bias") || name.ends_with("ssm_a") {
        return qwen35_inverse_reorder_v_rows(tensor, layout, 1, name);
    }
    if name.ends_with("ssm_conv1d.weight") {
        let qk = tensor.narrow(0, 0, layout.key_dim * 2).map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to split qwen35 conv tensor `{name}` qk rows: {err}"
            ))
        })?;
        let v = tensor
            .narrow(0, layout.key_dim * 2, layout.value_dim)
            .map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to split qwen35 conv tensor `{name}` v rows: {err}"
                ))
            })?;
        let v = qwen35_inverse_reorder_v_rows(v, layout, layout.head_v_dim, name)?;
        return Tensor::cat(&[qk, v], 0).map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to reassemble qwen35 conv tensor `{name}` after grouped V-head layout fix: {err}"
            ))
        });
    }
    if name.ends_with("ssm_out.weight") {
        return qwen35_inverse_reorder_out_proj_columns(tensor, layout, name);
    }
    Ok(tensor)
}

fn gguf_llama_like_rope_unpermute_heads(
    content: &gguf_file::Content,
    architecture: &str,
    name: &str,
) -> Result<Option<usize>> {
    let metadata_prefix = match architecture {
        "llama" => "llama",
        "granite" => "granite",
        _ => return Ok(None),
    };
    let key = if name.ends_with("attn_q.weight") {
        Some(format!("{metadata_prefix}.attention.head_count"))
    } else if name.ends_with("attn_k.weight") {
        Some(format!("{metadata_prefix}.attention.head_count_kv"))
    } else {
        None
    };
    key.map(|key| {
        gguf_value_to_u64(gguf_metadata_value(content, &key)?).map(|value| value as usize)
    })
    .transpose()
}

fn unpermute_llama_like_attention_tensor(
    tensor: candle_core::Tensor,
    num_heads: usize,
    name: &str,
) -> Result<candle_core::Tensor> {
    let dims = tensor.dims().to_vec();
    let [output_dim, input_dim] = dims.as_slice() else {
        return Err(LLMError::InvalidModelConfig(format!(
            "expected a rank-2 tensor for `{name}`, got shape {dims:?}"
        )));
    };
    let head_dim = output_dim / num_heads;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(LLMError::InvalidModelConfig(format!(
            "invalid RoPE head_dim {head_dim} for GGUF tensor `{name}`"
        )));
    }
    tensor
        .reshape((num_heads, head_dim / 2, 2, *input_dim))
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to reshape GGUF tensor `{name}` for RoPE unpermute: {err}"
            ))
        })?
        .transpose(1, 2)
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to transpose GGUF tensor `{name}` for RoPE unpermute: {err}"
            ))
        })?
        .reshape((*output_dim, *input_dim))
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to restore GGUF tensor `{name}` shape after RoPE unpermute: {err}"
            ))
        })?
        .contiguous()
        .map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to make GGUF tensor `{name}` contiguous after RoPE unpermute: {err}"
            ))
        })
}

fn map_granite_gguf_tensor_name(name: &str) -> Result<Option<(catgrad::prelude::Path, bool)>> {
    if name == "rope_freqs.weight" {
        return Ok(None);
    }
    if name == "token_embd.weight" {
        return Ok(Some((
            gguf_path(name, vec!["model", "embed_tokens", "weight"])?,
            false,
        )));
    }
    if name == "output_norm.weight" {
        return Ok(Some((
            gguf_path(name, vec!["model", "norm", "weight"])?,
            false,
        )));
    }
    if name == "output.weight" {
        return Ok(Some((gguf_path(name, vec!["lm_head", "weight"])?, false)));
    }

    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() != 4 || parts[0] != "blk" {
        return Err(LLMError::InvalidModelConfig(format!(
            "unsupported GGUF granite tensor name `{name}`"
        )));
    }
    let layer = parts[1];
    let suffix = match (parts[2], parts[3]) {
        ("attn_norm", "weight") => vec!["input_layernorm", "weight"],
        ("ffn_norm", "weight") => vec!["post_attention_layernorm", "weight"],
        ("attn_q", "weight") => vec!["self_attn", "q_proj", "weight"],
        ("attn_k", "weight") => vec!["self_attn", "k_proj", "weight"],
        ("attn_v", "weight") => vec!["self_attn", "v_proj", "weight"],
        ("attn_output", "weight") => vec!["self_attn", "o_proj", "weight"],
        ("ffn_gate", "weight") => vec!["mlp", "gate_proj", "weight"],
        ("ffn_up", "weight") => vec!["mlp", "up_proj", "weight"],
        ("ffn_down", "weight") => vec!["mlp", "down_proj", "weight"],
        _ => {
            return Err(LLMError::InvalidModelConfig(format!(
                "unsupported GGUF granite tensor name `{name}`"
            )));
        }
    };

    let mut components = vec!["model", "layers", layer];
    components.extend(suffix);
    Ok(Some((gguf_path(name, components)?, false)))
}

fn gguf_architecture_dispatch(
    architecture: &str,
    name: &str,
) -> Result<Option<(catgrad::prelude::Path, bool)>> {
    match architecture {
        "llama" => map_llama_gguf_tensor_name(name),
        "qwen35" => map_qwen35_gguf_tensor_name(name),
        "granite" => map_granite_gguf_tensor_name(name),
        _ => Err(LLMError::UnsupportedModel(format!(
            "GGUF architecture {architecture}"
        ))),
    }
}

fn gguf_tied_lm_head_path(architecture: &str) -> Result<catgrad::prelude::Path> {
    match architecture {
        "llama" | "qwen35" | "granite" => gguf_path("lm_head.weight", vec!["lm_head", "weight"]),
        _ => Err(LLMError::UnsupportedModel(format!(
            "GGUF architecture {architecture}"
        ))),
    }
}

fn gguf_is_dense_only_tensor(architecture: &str, name: &str) -> bool {
    if name == "token_embd.weight" || name.ends_with("norm.weight") {
        return true;
    }

    match architecture {
        "llama" | "granite" => name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight"),
        "qwen35" => {
            name.ends_with("ssm_dt.bias")
                || name.ends_with("ssm_a")
                || name.ends_with("ssm_conv1d.weight")
        }
        _ => true,
    }
}

fn gguf_can_keep_quantized(architecture: &str, name: &str) -> bool {
    !gguf_is_dense_only_tensor(architecture, name)
}

pub fn load_gguf_weights(
    model_path: &Path,
    backend: &CandleBackend,
    dtype: Dtype,
) -> Result<(
    catgrad::interpreter::Parameters<CandleBackend>,
    catgrad::typecheck::Parameters,
    usize,
)> {
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|err| {
        LLMError::InvalidModelConfig(format!("failed to read GGUF metadata: {err}"))
    })?;

    let target_dtype = match dtype {
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        _ => return Err(LLMError::UnsupportedDtype(format!("{dtype:?}"))),
    };

    let mut data_map = BTreeMap::new();
    let mut type_map = BTreeMap::new();
    let mut total_params = 0usize;
    let architecture = gguf_architecture(&content)?;
    let tied_quantized_output = gguf_has_quantized_tied_output(&content);
    let qwen35_layout = qwen35_linear_attention_layout(&content)?;

    for (name, tensor_info) in &content.tensor_infos {
        let Some((key, transpose)) = gguf_architecture_dispatch(&architecture, name)? else {
            continue;
        };
        let is_quantized = gguf_is_quantized_dtype(tensor_info.ggml_dtype);
        if !matches!(
            tensor_info.ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) && !gguf_supported_quantized_dtype(tensor_info.ggml_dtype)
        {
            return Err(LLMError::UnsupportedDtype(format!(
                "GGUF tensor `{name}` has unsupported dtype {:?}",
                tensor_info.ggml_dtype
            )));
        }

        let qtensor = content
            .tensor(&mut file, name, backend.device())
            .map_err(|err| {
                LLMError::InvalidModelConfig(format!("failed to read GGUF tensor `{name}`: {err}"))
            })?;
        total_params += tensor_info.shape.elem_count();
        let keep_quantized = is_quantized
            && gguf_can_keep_quantized(&architecture, name)
            && !qwen35_needs_grouped_v_head_layout_fixup(qwen35_layout, name);

        if keep_quantized && transpose {
            return Err(LLMError::InvalidModelConfig(format!(
                "quantized GGUF tensor `{name}` requires transpose but quantized transpose is unsupported"
            )));
        }

        if keep_quantized {
            let shape = tensor_info.shape.dims().to_vec();
            data_map.insert(
                key.clone(),
                catgrad::interpreter::Value::Tensor(
                    backend.tagged_tensor_from_qtensor(qtensor, dtype),
                ),
            );
            let tensor_type = catgrad::typecheck::Type::Tensor(
                catgrad::typecheck::TypeExpr::NdArrayType(catgrad::typecheck::NdArrayType {
                    dtype: catgrad::typecheck::DtypeExpr::Constant(dtype),
                    shape: catgrad::typecheck::ShapeExpr::Shape(
                        shape
                            .into_iter()
                            .map(catgrad::typecheck::NatExpr::Constant)
                            .collect(),
                    ),
                }),
            );
            type_map.insert(key, tensor_type);

            if tied_quantized_output && name == "token_embd.weight" {
                let lm_head_key = gguf_tied_lm_head_path(&architecture)?;
                data_map.insert(
                    lm_head_key.clone(),
                    catgrad::interpreter::Value::Tensor(
                        backend.tagged_tensor_from_qtensor(
                            content
                                .tensor(&mut file, name, backend.device())
                                .map_err(|err| {
                                    LLMError::InvalidModelConfig(format!(
                                        "failed to reread GGUF tensor `{name}`: {err}"
                                    ))
                                })?,
                            dtype,
                        ),
                    ),
                );
                let tensor_type = catgrad::typecheck::Type::Tensor(
                    catgrad::typecheck::TypeExpr::NdArrayType(catgrad::typecheck::NdArrayType {
                        dtype: catgrad::typecheck::DtypeExpr::Constant(dtype),
                        shape: catgrad::typecheck::ShapeExpr::Shape(
                            tensor_info
                                .shape
                                .dims()
                                .to_vec()
                                .into_iter()
                                .map(catgrad::typecheck::NatExpr::Constant)
                                .collect(),
                        ),
                    }),
                );
                type_map.insert(lm_head_key, tensor_type);
            }
            continue;
        }

        let mut tensor = qtensor.dequantize(backend.device()).map_err(|err| {
            LLMError::InvalidModelConfig(format!(
                "failed to materialize GGUF tensor `{name}`: {err}"
            ))
        })?;
        if transpose {
            tensor = tensor.transpose(0, 1).map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to transpose GGUF tensor `{name}`: {err}"
                ))
            })?;
            tensor = tensor.contiguous().map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to make transposed GGUF tensor `{name}` contiguous: {err}"
                ))
            })?;
        }
        if architecture == "qwen35" && name.ends_with("ssm_conv1d.weight") {
            tensor = tensor.unsqueeze(1).map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to reshape GGUF tensor `{name}` for qwen35 conv1d: {err}"
                ))
            })?;
        }
        if architecture == "qwen35" && name.ends_with("ssm_a") {
            tensor = tensor
                .neg()
                .map_err(|err| {
                    LLMError::InvalidModelConfig(format!(
                        "failed to negate GGUF tensor `{name}`: {err}"
                    ))
                })?
                .log()
                .map_err(|err| {
                    LLMError::InvalidModelConfig(format!(
                        "failed to convert GGUF tensor `{name}` into A_log: {err}"
                    ))
                })?;
        }
        if architecture == "qwen35" && qwen35_needs_grouped_v_head_layout_fixup(qwen35_layout, name)
        {
            tensor = qwen35_inverse_grouped_v_head_layout(
                tensor,
                qwen35_layout.expect("qwen35 layout missing"),
                name,
            )?;
        }
        if let Some(num_heads) =
            gguf_llama_like_rope_unpermute_heads(&content, &architecture, name)?
        {
            tensor = unpermute_llama_like_attention_tensor(tensor, num_heads, name)?;
        }
        if architecture == "qwen35"
            && tensor_info.ggml_dtype == GgmlDType::F32
            && qwen35_gguf_uses_gemma_rmsnorm_offset(name)
        {
            let one = tensor.ones_like().map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to create ones tensor for GGUF tensor `{name}`: {err}"
                ))
            })?;
            tensor = tensor.sub(&one).map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to convert GGUF tensor `{name}` from gamma to gamma_minus_one: {err}"
                ))
            })?;
        }

        let preserve_source_dtype = tensor_info.ggml_dtype == GgmlDType::F32;
        let tensor_dtype = if preserve_source_dtype {
            match tensor.dtype() {
                DType::F32 => Dtype::F32,
                DType::F16 => Dtype::F16,
                DType::BF16 => Dtype::BF16,
                dense_dtype => {
                    return Err(LLMError::UnsupportedDtype(format!(
                        "unsupported dense GGUF tensor dtype for `{name}`: {dense_dtype:?}"
                    )));
                }
            }
        } else {
            dtype
        };
        if !preserve_source_dtype && tensor.dtype() != target_dtype {
            tensor = tensor.to_dtype(target_dtype).map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "failed to cast GGUF tensor `{name}` to {target_dtype:?}: {err}"
                ))
            })?;
        }

        let shape = tensor.dims().to_vec();
        data_map.insert(
            key.clone(),
            catgrad::interpreter::Value::Tensor(backend.tagged_tensor_from_tensor(tensor)),
        );
        let tensor_type = catgrad::typecheck::Type::Tensor(
            catgrad::typecheck::TypeExpr::NdArrayType(catgrad::typecheck::NdArrayType {
                dtype: catgrad::typecheck::DtypeExpr::Constant(tensor_dtype),
                shape: catgrad::typecheck::ShapeExpr::Shape(
                    shape
                        .into_iter()
                        .map(catgrad::typecheck::NatExpr::Constant)
                        .collect(),
                ),
            }),
        );
        type_map.insert(key, tensor_type);

        if tied_quantized_output && name == "token_embd.weight" {
            let lm_head_key = gguf_tied_lm_head_path(&architecture)?;
            data_map.insert(
                lm_head_key.clone(),
                catgrad::interpreter::Value::Tensor(
                    backend.tagged_tensor_from_qtensor(
                        content
                            .tensor(&mut file, name, backend.device())
                            .map_err(|err| {
                                LLMError::InvalidModelConfig(format!(
                                    "failed to reread GGUF tensor `{name}`: {err}"
                                ))
                            })?,
                        dtype,
                    ),
                ),
            );
            let tensor_type = catgrad::typecheck::Type::Tensor(
                catgrad::typecheck::TypeExpr::NdArrayType(catgrad::typecheck::NdArrayType {
                    dtype: catgrad::typecheck::DtypeExpr::Constant(dtype),
                    shape: catgrad::typecheck::ShapeExpr::Shape(
                        tensor_info
                            .shape
                            .dims()
                            .to_vec()
                            .into_iter()
                            .map(catgrad::typecheck::NatExpr::Constant)
                            .collect(),
                    ),
                }),
            );
            type_map.insert(lm_head_key, tensor_type);
        }
    }

    Ok((
        catgrad::interpreter::Parameters::from(data_map),
        catgrad::typecheck::Parameters::from(type_map),
        total_params,
    ))
}

type LoadedGgufModelArtifacts = (
    catgrad::interpreter::Parameters<CandleBackend>,
    catgrad::typecheck::Parameters,
    serde_json::Value,
    Tokenizer,
    serde_json::Value,
    String,
    usize,
);

pub fn load_gguf_model(
    model_name: &str,
    revision: &str,
    gguf_filename: &str,
    backend: &CandleBackend,
    dtype: Dtype,
) -> Result<LoadedGgufModelArtifacts> {
    let repo = super::get_model_repo(model_name, revision)?;
    let model_path = repo.get(gguf_filename)?;

    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|err| {
        LLMError::InvalidModelConfig(format!("failed to read GGUF metadata: {err}"))
    })?;

    let config_json = match gguf_architecture(&content)?.as_str() {
        "llama" => build_llama_config_from_gguf(&content)?,
        "qwen35" => build_qwen35_config_from_gguf(&content)?,
        "granite" => build_granite_config_from_gguf(&content)?,
        architecture => {
            return Err(LLMError::UnsupportedModel(format!(
                "GGUF architecture {architecture}"
            )));
        }
    };
    let tokenizer = build_gguf_tokenizer(&content)?;
    let tokenizer_config_json = build_gguf_tokenizer_config_json(&content)?;
    let chat_template = {
        let template = gguf_chat_template(&content)?;
        if template.is_empty() {
            String::new()
        } else {
            template
        }
    };
    let (parameter_values, parameter_types, total_params) =
        load_gguf_weights(&model_path, backend, dtype)?;

    Ok((
        parameter_values,
        parameter_types,
        config_json,
        tokenizer,
        tokenizer_config_json,
        chat_template,
        total_params,
    ))
}
