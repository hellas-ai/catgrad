use serde::Deserialize;
use std::collections::BTreeMap;

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
    pub llama_4_scaling_beta: f32,
    pub original_max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LongropeRopeScaling {
    pub short_factor: Vec<f32>,
    pub long_factor: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RopeScaling {
    #[serde(alias = "llama3")]
    Llama3(Llama3RopeScaling),
    Longrope(LongropeRopeScaling),
    #[serde(alias = "yarn")]
    Yarn(YarnRopeScaling),
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QuantizationScheme {
    pub strategy: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QuantizationConfigGroup {
    pub format: Option<String>,
    pub input_activations: Option<QuantizationScheme>,
    pub weights: Option<QuantizationScheme>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct QuantizationConfig {
    pub format: Option<String>,
    pub ignore: Vec<String>,
    pub config_groups: BTreeMap<String, QuantizationConfigGroup>,
}

impl QuantizationConfig {
    pub fn is_float_quantized(&self) -> bool {
        self.format.as_deref() == Some("float-quantized")
    }

    pub fn input_strategy(&self) -> Option<&str> {
        self.config_groups.values().find_map(|group| {
            group
                .input_activations
                .as_ref()
                .and_then(|scheme| scheme.strategy.as_deref())
        })
    }

    pub fn weight_strategy(&self) -> Option<&str> {
        self.config_groups.values().find_map(|group| {
            group
                .weights
                .as_ref()
                .and_then(|scheme| scheme.strategy.as_deref())
        })
    }

    pub fn uses_dynamic_fp8(&self) -> bool {
        self.is_float_quantized()
            && self.input_strategy() == Some("token")
            && self.weight_strategy() == Some("channel")
    }

    pub fn is_ignored(&self, module_name: &str) -> bool {
        self.ignore.iter().any(|ignored| ignored == module_name)
    }

    pub fn uses_dynamic_fp8_for_module(&self, module_name: &str) -> bool {
        self.uses_dynamic_fp8() && !self.is_ignored(module_name)
    }
}

pub trait LLMConfig {
    fn num_hidden_layers(&self) -> usize;
    fn num_kv_layers(&self) -> usize {
        self.num_hidden_layers()
    }
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
    fn max_position_embeddings(&self) -> usize {
        0
    }
    fn original_max_position_embeddings(&self) -> usize {
        self.max_position_embeddings()
    }
    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        None
    }
    fn uses_dynamic_fp8(&self) -> bool {
        self.quantization_config()
            .is_some_and(QuantizationConfig::uses_dynamic_fp8)
    }
    fn uses_dynamic_fp8_for_module(&self, module_name: &str) -> bool {
        self.quantization_config()
            .is_some_and(|q| q.uses_dynamic_fp8_for_module(module_name))
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
