mod tensors;
pub use tensors::*;

mod rope;
pub use rope::*;

mod module;
pub use module::*;

use crate::config::{LLMConfig, RopeScaling};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

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
