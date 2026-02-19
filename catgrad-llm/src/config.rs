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
