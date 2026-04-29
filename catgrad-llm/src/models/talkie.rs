//! Talkie 13B — a decoder-only transformer with the standard Llama backbone
//! plus four small departures:
//!
//! 1. RMSNorm everywhere is **unweighted** (`F.rms_norm(x, ..)` with no γ),
//!    including a norm immediately after the embedding.
//! 2. **QK-norm** — RMSNorm is applied to Q and K *after* RoPE.
//! 3. **Per-head and per-layer learned gains** — `head_gain` (shape `[H]`)
//!    on Q after QK-norm, and scalar `attn_gain` / `mlp_gain` on the
//!    attention and MLP residual branches.
//! 4. **Embedding skip connection** — the post-input-norm activations are
//!    threaded through every block as `e_x` and added back via a scalar
//!    `embed_skip` gain.
//!
//! The lm_head is an untied `[V, D]` parameter (not a `Linear`) and is
//! scaled by a learned scalar (`lm_head_gain.w_g`) before the final matmul.
//!
//! RoPE convention differs from catgrad's default by a sign on `sin`
//! (talkie: `y1 = x1·cos + x2·sin`, catgrad: `y1 = x1·cos - x2·sin`); we
//! negate `cache.sin` once after init to match.
//!
//! Parameter naming follows the HF-style port at
//! `lewtun/talkie-1930-13b-it-hf` — the decoder stack lives under
//! `model.{embed,blocks.…}` while `lm_head` and `lm_head_gain.w_g` are at
//! the root (matching `TalkieForCausalLM` having `self.model = TalkieModel(…)`
//! and the head as direct attributes).

#![allow(clippy::too_many_arguments)]

use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TalkieConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    eos_token_id: Option<EosTokenId>,
}

impl LLMConfig for TalkieConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_key_value_heads(&self) -> usize {
        // Talkie has no GQA — kv heads == attention heads.
        self.num_attention_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

impl TalkieConfig {
    /// Talkie's MLP intermediate width: `round((8/3)·D / 128) · 128`.
    /// Hardcoded in `src/talkie/model.py`'s `MLP.__init__`; not stored in
    /// the checkpoint or any config file.
    fn intermediate_size(&self) -> usize {
        let h = self.hidden_size as f32;
        (((8.0 / 3.0) * h / 128.0).round() as usize) * 128
    }
}

#[derive(Debug, Clone)]
pub struct TalkieModel {
    config: TalkieConfig,
    dtype: Dtype,
}

impl LLMModel for TalkieModel {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }
}

impl TalkieModel {
    pub fn new(config_json: &serde_json::Value, dtype: Dtype) -> crate::Result<Self> {
        let config: TalkieConfig = serde_json::from_value(config_json.clone())?;
        Ok(Self { config, dtype })
    }

    fn forward(
        &self,
        builder: &Builder,
        p: Path,
        x: Var,
        in_k: Var,
        in_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let eps = self.config.rms_norm_eps;
        // The HF-style port wraps the decoder stack in a `TalkieModel` that
        // sits under the `TalkieForCausalLM`'s `self.model`; lm_head and
        // lm_head_gain stay at the root.
        let m = p.extend(["model"]).unwrap();

        // Embed → input RMSNorm (unweighted) → save as e_x for embed-skip.
        let x = embeddings(builder, m.extend(["embed"]).unwrap(), x);
        let x = rmsnorm_raw::<3>(builder, eps, x);
        let e_x = x.clone();

        let [_, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k.clone()));
        let attention_mask = causal_mask(builder, s, pos.clone());

        let mut cache = Cache::init(
            builder,
            &self.config,
            max_positions.clone(),
            max_positions,
            in_k,
            in_v,
        );

        // Talkie's RoPE has the opposite sin convention from catgrad's. Negate
        // the sin table once here; everything downstream uses it as-is.
        let neg = constant(builder, -1.0, &shape(builder, cache.sin.clone()));
        cache.sin = cache.sin.clone() * neg;

        let mut x = x;
        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                pos.clone(),
                e_x.clone(),
                m.extend(["blocks", &i.to_string()]).unwrap(),
                x,
            );
        }

        let x = rmsnorm_raw::<3>(builder, eps, x);

        // lm_head with WeightGain: scale the [V, D] weight by a scalar
        // before the matmul. lm_head is a bare Parameter, not a Linear,
        // so we can't go through `linear_no_bias` (which expects `<p>.weight`).
        let lm_head = param(builder, &p.extend(["lm_head"]).unwrap());
        let w_g = param(builder, &p.extend(["lm_head_gain", "w_g"]).unwrap());
        let lm_sh = shape(builder, lm_head.clone());
        let w_g = broadcast(builder, lm_sh, w_g);
        let lm_head = lm_head * w_g;
        let x = linear_no_bias_param(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            lm_head,
            x,
        );

        let x = argmax(builder, x);
        let (out_k, out_v) = cache.get_kv_cache(builder);
        vec![x, out_k, out_v]
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        e_x: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let eps = self.config.rms_norm_eps;

        // Pre-attn norm (unweighted) → attn → scalar attn_gain → residual.
        let res = x.clone();
        let x = rmsnorm_raw::<3>(builder, eps, x);
        let x = self.attention(
            builder,
            layer_id,
            attention_mask,
            cache,
            pos,
            p.extend(["attn"]).unwrap(),
            x,
        );
        let x = scale(builder, p.extend(["attn_gain", "a_g"]).unwrap(), x);
        let x = res + x;

        // Pre-mlp norm (unweighted) → mlp → scalar mlp_gain → residual.
        let res = x.clone();
        let x = rmsnorm_raw::<3>(builder, eps, x);
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        let x = scale(builder, p.extend(["mlp_gain", "a_g"]).unwrap(), x);
        let x = res + x;

        // Embedding-skip residual: x += embed_skip * e_x.
        let skip = scale(builder, p.extend(["embed_skip", "a_g"]).unwrap(), e_x);
        x + skip
    }

    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let h = self.config.hidden_size;
        let i = self.config.intermediate_size();
        let gate = linear_no_bias(builder, h, i, p.extend(["mlp_gate"]).unwrap(), x.clone());
        let up = linear_no_bias(builder, h, i, p.extend(["mlp_linear"]).unwrap(), x);
        let x = silu(builder, gate) * up;
        linear_no_bias(builder, i, h, p.extend(["mlp_resid"]).unwrap(), x)
    }

    fn attention(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let eps = self.config.rms_norm_eps;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(
            builder,
            dim,
            dim,
            p.extend(["attn_query"]).unwrap(),
            x.clone(),
        );
        let k = linear_no_bias(
            builder,
            dim,
            dim,
            p.extend(["attn_key"]).unwrap(),
            x.clone(),
        );
        let v = linear_no_bias(builder, dim, dim, p.extend(["attn_value"]).unwrap(), x);

        let qkv_sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, qkv_sh.clone(), q);
        let k = reshape(builder, qkv_sh.clone(), k);
        let v = reshape(builder, qkv_sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        // RoPE (cache.sin already sign-flipped at init).
        let q = apply_rope_embedding(
            builder,
            pos.clone(),
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            q,
        );
        let k = apply_rope_embedding(
            builder,
            pos,
            head_dim,
            cache.cos.clone(),
            cache.sin.clone(),
            k,
        );

        // QK-norm (RMSNorm with no learned weight, over last dim).
        let q = rmsnorm_raw::<4>(builder, eps, q);
        let k = rmsnorm_raw::<4>(builder, eps, k);

        // Per-head gain on Q only. head_g: [H] → broadcast to [B, H, S, D].
        let head_g = param(builder, &p.extend(["head_gain", "head_g"]).unwrap());
        let head_g = reshape(builder, shape!(builder, 1, num_heads, 1, 1), head_g);
        let q_sh = shape(builder, q.clone());
        let head_g = broadcast(builder, q_sh, head_g);
        let q = q * head_g;

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let attn_sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &attn_sh);
        let denom = cast(builder, denom, dtype(builder, attn.clone()));
        let mut attn = attn / denom;

        let mask = cast(builder, attention_mask, dtype(builder, attn.clone()));
        let mask = broadcast(builder, attn_sh, mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let attn = reshape(builder, shape!(builder, b, s, dim), attn);

        linear_no_bias(builder, dim, dim, p.extend(["attn_resid"]).unwrap(), attn)
    }
}

/// Multiply `x` by a scalar parameter at `p` (shape `[1]`), broadcasting
/// over `x`'s shape. Used for `attn_gain`, `mlp_gain`, `embed_skip`.
fn scale(builder: &Builder, p: Path, x: Var) -> Var {
    let g = param(builder, &p);
    let sh = shape(builder, x.clone());
    let g = broadcast(builder, sh, g);
    x * g
}

impl DynModule for TalkieModel {
    fn path(&self) -> Path {
        path(vec!["talkie"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, max_positions]: [Var; 4] = args.try_into().expect("expected 4 inputs");
        self.forward(builder, self.path(), x, in_k, in_v, max_positions)
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        llm_type(&self.config, self.dtype())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_typechecks() {
        // Tiny shape that still exercises every novel piece (QK-norm,
        // head_gain, attn_gain, mlp_gain, embed_skip, lm_head_gain) and the
        // n_mlp formula `round((8/3)·H/128)·128` (here: 192 → 512).
        let cfg = serde_json::json!({
            "vocab_size": 64,
            "hidden_size": 192,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "head_dim": 96,
            "max_position_embeddings": 16,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "eos_token_id": 63,
        });
        let model = TalkieModel::new(&cfg, Dtype::BF16).expect("model construction");
        assert_eq!(model.config.intermediate_size(), 512);
        model
            .term()
            .expect("term construction failed (sort/type mismatch)");
    }
}
