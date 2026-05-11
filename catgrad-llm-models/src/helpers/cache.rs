use crate::config::LLMConfig;
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;

pub struct KVCache {
    in_k: Var,
    in_v: Var,
    out_kv_cache: Vec<(Var, Var)>,
}

impl KVCache {
    pub fn init(builder: &Builder, num_layers: usize, in_k: Var, in_v: Var) -> Self {
        let mut out_kv_cache = Vec::with_capacity(num_layers);
        for layer_id in 0..num_layers {
            let k = slice(builder, 0, layer_id, 1, in_k.clone());
            let v = slice(builder, 0, layer_id, 1, in_v.clone());
            let k = squeeze::<5, 4>(builder, 0, k);
            let v = squeeze::<5, 4>(builder, 0, v);
            out_kv_cache.push((k, v));
        }

        Self {
            in_k,
            in_v,
            out_kv_cache,
        }
    }

    pub fn update(&mut self, builder: &Builder, layer_id: usize, k: Var, v: Var) -> (Var, Var) {
        let cached_k = self.out_kv_cache[layer_id].0.clone();
        let cached_v = self.out_kv_cache[layer_id].1.clone();

        let k = concat(builder, 2, cached_k, k);
        let v = concat(builder, 2, cached_v, v);

        self.out_kv_cache[layer_id] = (k.clone(), v.clone());
        (k, v)
    }

    pub fn get(&self, layer_id: usize) -> (Var, Var) {
        self.out_kv_cache[layer_id].clone()
    }

    pub fn output(&self, builder: &Builder, seq_len: Var) -> (Var, Var) {
        if self.out_kv_cache.is_empty() {
            let [_, batch, heads, cache_len, dim_k] =
                unpack::<5>(builder, shape(builder, self.in_k.clone()));
            let [_, _, _, _, dim_v] = unpack::<5>(builder, shape(builder, self.in_v.clone()));
            let out_len = cache_len + seq_len;
            let out_k = reshape(
                builder,
                shape!(builder, 0, batch, heads, out_len, dim_k),
                self.in_k.clone(),
            );
            let out_v = reshape(
                builder,
                shape!(builder, 0, batch, heads, out_len, dim_v),
                self.in_v.clone(),
            );
            return (out_k, out_v);
        }

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

pub struct Cache {
    pub cos: Var,
    pub sin: Var,
    pub kv_cache: KVCache,
    pub linear_cache: Option<Vec<Vec<Var>>>,
}

impl Cache {
    pub fn init(
        builder: &Builder,
        config: &dyn LLMConfig,
        table_len: impl IntoNatVar,
        current_context_len: impl IntoNatVar,
        in_k: Var,
        in_v: Var,
    ) -> Self {
        let (cos, sin) = init_rope_tables(builder, config, table_len, current_context_len);
        let kv_cache = KVCache::init(builder, config.num_kv_layers(), in_k, in_v);

        Self {
            cos,
            sin,
            kv_cache,
            linear_cache: None,
        }
    }

    pub fn update_kv_cache(
        &mut self,
        builder: &Builder,
        layer_id: usize,
        k: Var,
        v: Var,
    ) -> (Var, Var) {
        self.kv_cache.update(builder, layer_id, k, v)
    }

    pub fn get_kv_cache(&self, builder: &Builder) -> (Var, Var) {
        self.kv_cache.output(builder, 0.to_nat(builder))
    }
}
