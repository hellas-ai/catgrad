use serde::Deserialize;

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

pub trait LLMConfig {
    fn num_hidden_layers(&self) -> usize;
    fn num_key_value_heads(&self) -> usize;
    fn rope_theta(&self) -> f32 {
        10000.0
    }
    fn rope_scaling(&self) -> Option<RopeScaling> {
        None
    }
    fn partial_rotary_factor(&self) -> f32 {
        1.0
    }

    fn get_head_dim(&self) -> usize;
    fn get_qk_head_dim(&self) -> usize {
        self.get_head_dim()
    }
    fn get_v_head_dim(&self) -> usize {
        self.get_head_dim()
    }
    fn eos_token_id(&self) -> Option<EosTokenId>;
    fn get_eos_token_ids(&self) -> Vec<i32> {
        match self.eos_token_id() {
            Some(EosTokenId::Single(id)) => vec![id],
            Some(EosTokenId::Multiple(ref ids)) => ids.clone(),
            None => vec![],
        }
    }
}

// This configuration contains the union of relevant fields from all supported models.
// Models ignore fields they don't need. The aliases are for GPT-2 alternative names.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub attention_bias: bool,
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
    pub global_attn_every_n_layers: usize,
    pub rope_scaling: Option<RopeScaling>,
    pub no_rope_layer_interval: usize,
    pub layer_norm_eps: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub use_qk_norm: bool,
    pub eos_token_id: Option<EosTokenId>,
    pub vocab_size: usize,
    pub architectures: Vec<String>,
    pub layer_types: Vec<String>,
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

impl LLMConfig for Config {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn num_key_value_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }
    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }
    fn rope_scaling(&self) -> Option<RopeScaling> {
        self.rope_scaling.clone()
    }
    fn partial_rotary_factor(&self) -> f32 {
        self.partial_rotary_factor
    }

    // Sometimes the head_dim fields is missing
    fn get_head_dim(&self) -> usize {
        if self.qk_rope_head_dim != 0 {
            self.qk_rope_head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    // DeepSeek Multihead Latent Attention uses different head dimensions for queries, keys and values
    fn get_qk_head_dim(&self) -> usize {
        let qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        if qk_head_dim != 0 {
            qk_head_dim
        } else {
            self.get_head_dim()
        }
    }

    fn get_v_head_dim(&self) -> usize {
        if self.v_head_dim != 0 {
            self.v_head_dim
        } else {
            self.get_head_dim()
        }
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}
