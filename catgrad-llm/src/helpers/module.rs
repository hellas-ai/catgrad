use crate::config::LLMConfig;
use catgrad::prelude::*;

/// Type signature for LLM Modules
pub fn llm_type(config: &dyn LLMConfig, dtype: Dtype) -> (Vec<Type>, Vec<Type>) {
    use catgrad::typecheck::*;
    let batch_size = NatExpr::Var(0);
    let seq_len = NatExpr::Var(1);
    let cache_len = NatExpr::Var(2);
    let max_positions = NatExpr::Var(3);

    // Input shape B×S
    let t_x = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), seq_len.clone()]),
    }));

    // Output shape B×1
    let t_y = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::U32),
        shape: ShapeExpr::Shape(vec![batch_size.clone(), NatExpr::Constant(1)]),
    }));

    let num_layers = NatExpr::Constant(config.num_kv_layers());
    let num_kv_heads = NatExpr::Constant(config.num_key_value_heads());
    let qk_head_dim = NatExpr::Constant(config.get_qk_head_dim());
    let v_head_dim = NatExpr::Constant(config.get_v_head_dim());

    let t_k = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            cache_len.clone(),
            qk_head_dim,
        ]),
    }));

    let t_v = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            cache_len.clone(),
            v_head_dim,
        ]),
    }));

    let out_cache_len = NatExpr::Add(vec![cache_len, seq_len]);
    let t_k_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            out_cache_len.clone(),
            NatExpr::Constant(config.get_qk_head_dim()),
        ]),
    }));

    let t_v_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape: ShapeExpr::Shape(vec![
            num_layers,
            batch_size,
            num_kv_heads,
            out_cache_len,
            NatExpr::Constant(config.get_v_head_dim()),
        ]),
    }));

    (
        vec![t_x, t_k, t_v, Type::Nat(max_positions)],
        vec![t_y, t_k_out, t_v_out],
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPostProcess {
    None,
    ConcatMoeExperts { num_local_experts: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultimodalMetadata {
    pub image_token_index: usize,
    pub mm_tokens_per_image: usize,
    pub hidden_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

pub trait LLMModel: DynModule {
    fn config(&self) -> &dyn LLMConfig;

    fn dtype(&self) -> Dtype;

    fn extra_nat_input(&self, _seq_len: usize) -> Option<usize> {
        None
    }

    // Return empty KV-cache shape by default
    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let config = self.config();
        let dtype = self.dtype();
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
        vec![(dtype, k_shape), (dtype, v_shape)]
    }

    fn weight_post_process(&self) -> WeightPostProcess {
        WeightPostProcess::None
    }

    fn is_multimodal(&self) -> bool {
        self.multimodal_metadata().is_some()
    }

    fn multimodal_metadata(&self) -> Option<MultimodalMetadata> {
        None
    }

    fn multimodal_vision_module(&self) -> Option<Box<dyn DynModule>> {
        None
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        None
    }

    fn multimodal_interpolate_prompt(&self, _prompt: &str) -> Option<String> {
        None
    }
}
