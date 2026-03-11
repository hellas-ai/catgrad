mod tensors;
pub use tensors::*;

mod rope;
pub use rope::*;

use crate::config::{LLMConfig, RopeScaling};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

/// Type signature for LLM Modules
pub fn llm_type(config: &dyn LLMConfig) -> (Vec<Type>, Vec<Type>) {
    use catgrad::typecheck::*;
    let batch_size = NatExpr::Var(0);
    let seq_len = NatExpr::Var(1);
    let cache_len = NatExpr::Var(2);

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
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            cache_len.clone(),
            qk_head_dim,
        ]),
    }));

    let t_v = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
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
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers.clone(),
            batch_size.clone(),
            num_kv_heads.clone(),
            out_cache_len.clone(),
            NatExpr::Constant(config.get_qk_head_dim()),
        ]),
    }));

    let t_v_out = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(Dtype::F32),
        shape: ShapeExpr::Shape(vec![
            num_layers,
            batch_size,
            num_kv_heads,
            out_cache_len,
            NatExpr::Constant(config.get_v_head_dim()),
        ]),
    }));

    (vec![t_x, t_k, t_v], vec![t_y, t_k_out, t_v_out])
}

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub in_kv_cache: Vec<(Var, Var)>,
    pub out_kv_cache: Vec<(Var, Var)>,
    pub linear_state: Option<Vec<Var>>,
}

impl Cache {
    pub fn init(
        builder: &Builder,
        config: &dyn LLMConfig,
        positions: usize,
        in_k: Var,
        in_v: Var,
    ) -> Self {
        let (cos, sin) = match config.rope_scaling() {
            Some(RopeScaling::Yarn(params)) => rope_tables_yarn(
                builder,
                config.rope_theta(),
                &params,
                positions,
                config.get_head_dim(),
            ),
            Some(RopeScaling::Llama3(params)) => rope_tables_llama3(
                builder,
                config.rope_theta(),
                &params,
                positions,
                config.get_head_dim(),
            ),
            _ => rope_tables(
                builder,
                config.rope_theta(),
                positions,
                ((config.get_head_dim() as f32) * config.partial_rotary_factor()) as usize,
                1.0,
            ),
        };

        let num_kv_layers = config.num_kv_layers();
        let mut in_kv_cache = Vec::with_capacity(num_kv_layers);
        let mut out_kv_cache = Vec::with_capacity(num_kv_layers);
        for layer_id in 0..num_kv_layers {
            let k = slice(builder, 0, layer_id, 1, in_k.clone());
            let v = slice(builder, 0, layer_id, 1, in_v.clone());
            let k = squeeze::<5, 4>(builder, 0, k);
            let v = squeeze::<5, 4>(builder, 0, v);
            in_kv_cache.push((k.clone(), v.clone()));
            out_kv_cache.push((k, v));
        }

        Self {
            cos,
            sin,
            in_kv_cache,
            out_kv_cache,
            linear_state: None,
        }
    }

    pub fn update_kv_cache(
        &mut self,
        builder: &Builder,
        layer_id: usize,
        k: Var,
        v: Var,
    ) -> (Var, Var) {
        let cached_k = self.in_kv_cache[layer_id].0.clone();
        let cached_v = self.in_kv_cache[layer_id].1.clone();

        let k = concat(builder, 2, cached_k, k);
        let v = concat(builder, 2, cached_v, v);

        self.out_kv_cache[layer_id] = (k.clone(), v.clone());
        (k, v)
    }

    pub fn get_kv_cache(&self, builder: &Builder) -> (Var, Var) {
        let mut iter = self.out_kv_cache.iter();
        let (k0, v0) = iter.next().expect("no KV cache layers");
        let mut out_k = unsqueeze::<4, 5>(builder, 0, k0.clone());
        let mut out_v = unsqueeze::<4, 5>(builder, 0, v0.clone());
        for (k, v) in iter {
            let k = unsqueeze::<4, 5>(builder, 0, k.clone());
            let v = unsqueeze::<4, 5>(builder, 0, v.clone());
            out_k = concat(builder, 0, out_k, k);
            out_v = concat(builder, 0, out_v, v);
        }
        (out_k, out_v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightPostProcess {
    None,
    ConcatMoeExperts { num_local_experts: usize },
}

pub trait LLMModel: DynModule {
    fn config(&self) -> &dyn LLMConfig;

    // Return empty KV-cache shape by default
    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        let config = self.config();
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
        vec![(Dtype::F32, k_shape), (Dtype::F32, v_shape)]
    }

    fn weight_post_process(&self) -> WeightPostProcess {
        WeightPostProcess::None
    }
}
