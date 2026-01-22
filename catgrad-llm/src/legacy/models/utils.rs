use crate::legacy::nn::layers::concat;
use crate::legacy::nn::rope::{rope_tables, rope_tables_llama3, rope_tables_yarn};
use catgrad_legacy::{
    backend::cpu::{eval::Builder, ndarray::TaggedNdArray},
    core::{Dtype, NdArrayType, Shape, Var},
};

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(i32),
    Multiple(Vec<i32>),
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct YarnRopeScaling {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub truncate: bool,
    pub mscale: f32,
    pub mscale_all_dim: f32,
    pub original_max_position_embeddings: usize,
    #[serde(alias = "type")]
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RopeScaling {
    #[serde(alias = "llama3")]
    Llama3(Llama3RopeScaling),
    #[serde(alias = "yarn")]
    Yarn(YarnRopeScaling),
}

use super::lfm2::Lfm2Config;

// This configuration contains the union of relevant fields from all supported models.
// Models ignore fields they don't need. The aliases are for GPT-2 alternative names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub attention_bias: bool,
    pub head_dim: usize,
    pub decoder_sparse_step: usize,
    pub num_experts_per_tok: usize,
    #[serde(alias = "num_experts", alias = "n_routed_experts")]
    pub num_local_experts: usize,
    pub moe_intermediate_size: usize,
    pub first_k_dense_replace: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub routed_scaling_factor: f32,
    pub n_group: usize,
    pub topk_group: usize,
    pub n_shared_experts: usize,
    pub v_head_dim: usize,
    pub norm_topk_prob: bool,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub local_rope_theta: f32,
    pub global_rope_theta: f32,
    #[serde(alias = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub global_attn_every_n_layers: usize,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_local_base_freq: f32,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    pub no_rope_layer_interval: usize,
    pub layer_norm_epsilon: f32,
    pub layer_norm_eps: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub use_qk_norm: bool,
    pub eos_token_id: Option<EosTokenId>,
    pub vocab_size: usize,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub layer_types: Vec<String>,
    // This is set by the app to determine whether to use f16 or f32 for weights at runtime
    #[serde(skip)]
    pub dtype: Dtype,
    #[serde(flatten)]
    pub lfm2: Lfm2Config,
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

impl Config {
    // Sometimes the head_dim fields is missing
    pub fn get_head_dim(&self) -> usize {
        if self.qk_rope_head_dim != 0 {
            self.qk_rope_head_dim
        } else if self.head_dim == 0 {
            self.hidden_size / self.num_attention_heads
        } else {
            self.head_dim
        }
    }

    // DeepSeek Multihead Latent Attention uses different head dimensions for queries, keys and values
    pub fn get_qk_head_dim(&self) -> usize {
        let qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        if qk_head_dim != 0 {
            qk_head_dim
        } else {
            self.get_head_dim()
        }
    }

    pub fn get_v_head_dim(&self) -> usize {
        if self.v_head_dim != 0 {
            self.v_head_dim
        } else {
            self.get_head_dim()
        }
    }

    pub fn get_num_kv_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }

    pub fn get_eos_token_ids(&self) -> Vec<i32> {
        match self.eos_token_id {
            Some(EosTokenId::Single(id)) => vec![id],
            Some(EosTokenId::Multiple(ref ids)) => ids.clone(),
            None => vec![],
        }
    }
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub use_kv_cache: bool,
    pub in_kv_cache: Vec<(Var, Var)>,
    pub out_kv_cache: Vec<(Var, Var)>,
}

impl Cache {
    pub fn init(builder: &Builder, config: &Config, positions: usize, use_kv_cache: bool) -> Self {
        let (cos, sin) = match &config.rope_scaling {
            Some(RopeScaling::Llama3(params)) => rope_tables_llama3(
                builder,
                config.rope_theta,
                params,
                positions,
                config.get_head_dim(),
            ),
            Some(RopeScaling::Yarn(params)) => rope_tables_yarn(
                builder,
                config.rope_theta,
                params,
                positions,
                config.get_head_dim(),
            ),
            _ => rope_tables(
                builder,
                config.rope_theta,
                positions,
                ((config.get_head_dim() as f32) * config.partial_rotary_factor) as usize,
            ),
        };

        let kv_cache_type = NdArrayType::new(Shape(vec![]), Dtype::F32);
        // Just a placeholder
        let empty = Var::new(builder.clone(), kv_cache_type);
        Self {
            cos,
            sin,
            use_kv_cache,
            in_kv_cache: vec![(empty.clone(), empty.clone()); config.num_hidden_layers],
            out_kv_cache: vec![(empty.clone(), empty); config.num_hidden_layers],
        }
    }

    pub fn update_kv_cache(
        &mut self,
        builder: &Builder,
        layer_id: usize,
        k: Var,
        v: Var,
    ) -> (Var, Var) {
        let (mut k, mut v) = (k, v);
        if self.use_kv_cache {
            let cached_k = self.in_kv_cache[layer_id].0.clone();
            let cached_v = self.in_kv_cache[layer_id].1.clone();

            k = concat(builder, 2, cached_k, k);
            v = concat(builder, 2, cached_v, v);

            self.out_kv_cache[layer_id] = (k.clone(), v.clone());
        }
        (k, v)
    }
}

// Trait for model builders for various architectures (llama, qwen, gpt2, etc.)
pub trait ModelBuilder {
    // Build the model architecture graph for a given input shape
    fn build(
        &self,
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        x: Var,
    ) -> Var;
    // Optional post-processing of loaded weights (renaming, reshaping, etc.)
    fn post_load(&mut self, _tensors: &mut HashMap<String, TaggedNdArray>) {}
}

use super::deepseek::Model as DeepSeekV3Model;
use super::gemma::Model as GemmaModel;
use super::glm4::Model as GLM4Model;
use super::gpt_oss::Model as GPTOssModel;
use super::gpt2::Model as GPT2Model;
use super::granite::Model as GraniteModel;
use super::lfm2::Model as LFM2Model;
use super::llama::Model as LlamaModel;
use super::modernbert::Model as ModernBertDecoderModel;
use super::olmo::Model as OlmoModel;
use super::phi::Model as PhiModel;
use super::qwen::Model as QwenModel;
use super::smollm3::Model as SmolLM3Model;

pub fn get_model(arch: &str) -> crate::Result<Box<dyn ModelBuilder>> {
    match arch {
        "LlamaForCausalLM" => Ok(Box::new(LlamaModel {})),
        "MistralForCausalLM" => Ok(Box::new(LlamaModel {})),
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => Ok(Box::new(OlmoModel {})),
        "Qwen2ForCausalLM" => Ok(Box::new(QwenModel {})),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => Ok(Box::new(QwenModel {})),
        "Gemma3ForCausalLM" => Ok(Box::new(GemmaModel {
            root: "model".to_string(),
        })),
        "DeepseekV3ForCausalLM" => Ok(Box::new(DeepSeekV3Model {})),
        "GraniteForCausalLM" => Ok(Box::new(GraniteModel {})),
        "GraniteMoeForCausalLM" => Ok(Box::new(GraniteModel {})),
        "GraniteMoeHybridForCausalLM" => Ok(Box::new(GraniteModel {})),
        "GptOssForCausalLM" => Ok(Box::new(GPTOssModel {})),
        "Glm4MoeForCausalLM" | "Glm4ForCausalLM" => Ok(Box::new(GLM4Model {})),
        // GLM-4.7 Flash seems to be a DeepSeekV3 model
        "Glm4MoeLiteForCausalLM" => Ok(Box::new(DeepSeekV3Model {})),
        "Lfm2ForCausalLM" => Ok(Box::new(LFM2Model {})),
        "ModernBertDecoderForCausalLM" => Ok(Box::new(ModernBertDecoderModel {})),
        "Phi3ForCausalLM" => Ok(Box::new(PhiModel {})),
        "SmolLM3ForCausalLM" => Ok(Box::new(SmolLM3Model {})),
        "GPT2LMHeadModel" => Ok(Box::new(GPT2Model {})),
        _ => Err(crate::LLMError::UnsupportedModel(format!(
            "Unsupported model architecture: {arch}"
        ))),
    }
}
