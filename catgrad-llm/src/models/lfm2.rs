#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct Lfm2Config {
    hidden_size: usize,
    intermediate_size: Option<usize>,
    block_ff_dim: Option<usize>,
    block_ffn_dim_multiplier: f32,
    block_multiple_of: usize,
    block_auto_adjust_ff_dim: bool,
    full_attn_idxs: Vec<usize>,
    norm_eps: f32,
    conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
    eos_token_id: Option<EosTokenId>,
    vocab_size: usize,
    layer_types: Vec<String>,
}

impl Lfm2Config {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
            .or(self.block_ff_dim)
            .unwrap_or_default()
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.full_attn_idxs.contains(&layer_id)
            || (self.layer_types.len() == self.num_hidden_layers
                && self.layer_types[layer_id] == "full_attention")
    }

    fn num_kv_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&layer_id| self.is_full_attention_layer(layer_id))
            .count()
    }
}

impl LLMConfig for Lfm2Config {
    // This returns the number of layers for which to cache KV values.
    fn num_hidden_layers(&self) -> usize {
        self.num_kv_layers()
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn get_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

pub struct Lfm2Model {
    config: Lfm2Config,
    layer_to_cache_id: Vec<Option<usize>>,
    layer_to_linear_id: Vec<Option<usize>>,
    num_linear_layers: usize,
    pub max_sequence_length: usize,
}

impl LLMModel for Lfm2Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn empty_state_type(&self) -> Vec<(Dtype, Shape)> {
        vec![
            (
                Dtype::F32,
                Shape(vec![
                    self.config.num_hidden_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_head_dim(),
                ]),
            ),
            (
                Dtype::F32,
                Shape(vec![
                    self.config.num_hidden_layers,
                    1,
                    self.config.num_key_value_heads,
                    0,
                    self.config.get_head_dim(),
                ]),
            ),
            (
                Dtype::F32,
                Shape(vec![
                    self.num_linear_layers,
                    1,
                    self.config.hidden_size,
                    self.config.conv_l_cache,
                ]),
            ),
        ]
    }
}

// Select values from `x` where `mask` is 1, otherwise from `y`.
fn cond(builder: &Builder, mask: Var, x: Var, y: Var) -> Var {
    let sh = shape(builder, x.clone());
    let x_dtype = dtype(builder, x.clone());
    let mask = broadcast(builder, sh.clone(), mask);
    let mask = cast(builder, mask, x_dtype);
    let one = constant(builder, 1.0, &sh);
    x * mask.clone() + y * (one - mask)
}

impl Lfm2Model {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: Lfm2Config = serde_json::from_value(config_json.clone())?;
        assert!(config.conv_l_cache > 0, "lfm2 conv_l_cache must be > 0");
        let mut layer_to_cache_id = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_to_linear_id = Vec::with_capacity(config.num_hidden_layers);
        let mut next_cache_id = 0;
        let mut next_linear_id = 0;
        for layer_id in 0..config.num_hidden_layers {
            if config.is_full_attention_layer(layer_id) {
                layer_to_cache_id.push(Some(next_cache_id));
                layer_to_linear_id.push(None);
                next_cache_id += 1;
            } else {
                layer_to_cache_id.push(None);
                layer_to_linear_id.push(Some(next_linear_id));
                next_linear_id += 1;
            }
        }

        Ok(Self {
            config,
            layer_to_cache_id,
            layer_to_linear_id,
            num_linear_layers: next_linear_id,
            max_sequence_length,
        })
    }

    fn is_full_attention_layer(&self, layer_id: usize) -> bool {
        self.config.is_full_attention_layer(layer_id)
    }

    fn attention(
        &self,
        builder: &Builder,
        cache_layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let dim = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.hidden_size / num_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());
        let k = linear_no_bias(
            builder,
            dim,
            dim / rep,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );
        let v = linear_no_bias(builder, dim, dim / rep, p.extend(["v_proj"]).unwrap(), x);

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = rmsnorm::<4>(
            builder,
            self.config.norm_eps,
            p.extend(["q_layernorm"]).unwrap(),
            q,
        );
        let k = rmsnorm::<4>(
            builder,
            self.config.norm_eps,
            p.extend(["k_layernorm"]).unwrap(),
            k,
        );

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

        let (k, v) = cache.update_kv_cache(builder, cache_layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(builder, f32::sqrt(head_dim as f32), &sh);
        let mut attn = attn / denom;

        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(builder, dim, dim, p.extend(["out_proj"]).unwrap(), attn)
    }

    fn depthwise_conv1d(&self, builder: &Builder, p: &Path, x_padded: Var, s: Var) -> Var {
        let k = self.config.conv_l_cache;

        let conv_weight = param(builder, &p.extend(["conv", "weight"]).unwrap());
        let conv_weight = squeeze::<3, 2>(builder, 1, conv_weight);

        let mut conv_out: Option<Var> = None;
        for offset in 0..k {
            let x_slice = slice(builder, 2, offset, s.clone(), x_padded.clone());
            let w_slice = slice(builder, 1, offset, 1, conv_weight.clone());
            let w_slice = unsqueeze::<2, 3>(builder, 0, w_slice);
            let w_slice = broadcast(builder, shape(builder, x_slice.clone()), w_slice);
            let term = x_slice * w_slice;
            conv_out = Some(match conv_out {
                Some(acc) => acc + term,
                None => term,
            });
        }
        let mut conv_out = conv_out.expect("conv_l_cache must be positive");

        if self.config.conv_bias {
            let conv_bias = param(builder, &p.extend(["conv", "bias"]).unwrap());
            let conv_bias = unsqueeze::<1, 2>(builder, 0, conv_bias);
            let conv_bias = unsqueeze::<2, 3>(builder, 2, conv_bias);
            let conv_bias = broadcast(builder, shape(builder, conv_out.clone()), conv_bias);
            conv_out = conv_out + conv_bias;
        }

        conv_out
    }

    fn short_conv(
        &self,
        builder: &Builder,
        layer_id: usize,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let in_proj = linear_no_bias(
            builder,
            self.config.hidden_size,
            3 * self.config.hidden_size,
            p.extend(["in_proj"]).unwrap(),
            x,
        );
        let bcx = transpose(builder, 1, 2, in_proj);

        let bcx = chunk(builder, 1, 3, self.config.hidden_size, bcx);
        let b = bcx[0].clone();
        let c = bcx[1].clone();
        let x = bcx[2].clone();

        let bx = b * x;

        let [batch_size, hidden_dim, s] = unpack::<3>(builder, shape(builder, bx.clone()));
        let cache_len = self.config.conv_l_cache;
        let linear_layer_id =
            self.layer_to_linear_id[layer_id].expect("short-conv layer missing linear state index");
        let conv_state = cache
            .linear_state
            .as_ref()
            .expect("lfm2 short_conv requires linear state")[linear_layer_id]
            .clone();

        // HF prefill:
        // `conv_out = self.conv(Bx)[..., :seqlen]`
        let zeros_prefill_conv = constant(
            builder,
            0.0,
            &shape!(builder, batch_size, hidden_dim, cache_len - 1),
        );
        let x_padded_prefill_conv = concat(builder, 2, zeros_prefill_conv, bx.clone());
        let conv_out_prefill = self.depthwise_conv1d(builder, &p, x_padded_prefill_conv, s.clone());
        // HF prefill:
        // `conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))`
        let zeros_prefill_state = constant(
            builder,
            0.0,
            &shape!(builder, batch_size, hidden_dim, cache_len),
        );
        let x_padded_prefill_state = concat(builder, 2, zeros_prefill_state, bx.clone());
        let out_linear_state_prefill =
            slice(builder, 2, s.clone(), cache_len, x_padded_prefill_state);
        // HF prefill cache write:
        // `past_key_values.conv_cache[self.layer_idx].copy_(conv_state)`
        // where `conv_state` is exactly the padded/truncated `Bx`.

        // HF decode:
        // `conv_state = conv_state.roll(shifts=-1, dims=-1)`
        let rolled_state = {
            let tail = slice(builder, 2, 1, cache_len - 1, conv_state.clone());
            let head = slice(builder, 2, 0, 1, conv_state);
            concat(builder, 2, tail, head)
        };

        // HF: `cache_position = cache_position.clamp(0, self.L_cache - 1)`
        let sh_pos = shape(builder, nat_to_u32(builder, pos.clone()));
        let pos_u32 = nat_to_u32(builder, pos.clone());
        let zero_pos = constant(builder, 0u32, &sh_pos);
        let pos_f32 = cast(builder, pos_u32, Dtype::F32);
        let pos_f32 = clamp(builder, pos_f32, 0.0, (cache_len - 1) as f32);
        let u32_dtype = dtype_constant(builder, Dtype::U32);
        let pos_clamped_u32 = cast(builder, pos_f32, u32_dtype);

        // HF equivalent of indexing at `cache_position`: build one-hot over cache axis.
        // `conv_state[:, :, cache_position] = Bx`
        let sh_k = shape!(builder, cache_len.to_nat(builder));
        let positions = arange(builder, cache_len);
        let pos_vec = broadcast(builder, sh_k, pos_clamped_u32);
        let one_hot = eq(builder, positions, pos_vec);

        let sh_state = shape(builder, rolled_state.clone());

        // HF decode writes a single-token `Bx` into the selected cache slot.
        let bx_decode = slice(builder, 2, 0, 1, bx);
        let bx_decode = broadcast(builder, sh_state, bx_decode);
        let out_linear_state_decode = cond(builder, one_hot, bx_decode, rolled_state);

        // HF decode:
        // `conv_out = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1).unsqueeze(-1)`
        // Build a padded decode input with output length `s` so both branches have identical shape.
        let zeros_decode_tail = constant(builder, 0.0, &shape!(builder, batch_size, hidden_dim, s));
        let x_padded_decode = concat(
            builder,
            2,
            out_linear_state_decode.clone(),
            zeros_decode_tail,
        );
        let conv_out_decode = self.depthwise_conv1d(builder, &p, x_padded_decode, s);

        // HF branch condition: `if cache_position[0] > 0: ... else: ...`
        let is_decode = gt(builder, nat_to_u32(builder, pos), zero_pos);
        let conv_out = cond(
            builder,
            is_decode.clone(),
            conv_out_decode,
            conv_out_prefill,
        );
        let out_linear_state = cond(
            builder,
            is_decode,
            out_linear_state_decode,
            out_linear_state_prefill,
        );

        cache
            .linear_state
            .as_mut()
            .expect("lfm2 short_conv requires mutable linear state")[linear_layer_id] =
            out_linear_state;

        // HF: `y = C * conv_out; y = y.transpose(-1, -2).contiguous(); y = self.out_proj(y)`
        let y = c * conv_out;
        let y = transpose(builder, 1, 2, y);

        linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.hidden_size,
            p.extend(["out_proj"]).unwrap(),
            y,
        )
    }

    fn feed_forward(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let mut intermediate_size = self.config.intermediate_size();

        if self.config.block_auto_adjust_ff_dim {
            intermediate_size = 2 * intermediate_size / 3;
            intermediate_size =
                (self.config.block_ffn_dim_multiplier * intermediate_size as f32) as usize;
            intermediate_size = self.config.block_multiple_of
                * intermediate_size.div_ceil(self.config.block_multiple_of);
        }

        let gated = linear_no_bias(
            builder,
            self.config.hidden_size,
            intermediate_size,
            p.extend(["w1"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            self.config.hidden_size,
            intermediate_size,
            p.extend(["w3"]).unwrap(),
            x,
        );
        let x = silu(builder, gated) * up;

        linear_no_bias(
            builder,
            intermediate_size,
            self.config.hidden_size,
            p.extend(["w2"]).unwrap(),
            x,
        )
    }

    fn layer(
        &self,
        builder: &Builder,
        layer_id: usize,
        attention_mask: Var,
        cache: &mut Cache,
        pos: Var,
        p: Path,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.norm_eps,
            p.extend(["operator_norm"]).unwrap(),
            x,
        );
        let x = if self.is_full_attention_layer(layer_id) {
            let cache_layer_id = self.layer_to_cache_id[layer_id]
                .expect("full-attention layer missing KV cache index");
            self.attention(
                builder,
                cache_layer_id,
                attention_mask,
                cache,
                pos,
                p.extend(["self_attn"]).unwrap(),
                x,
            )
        } else {
            self.short_conv(
                builder,
                layer_id,
                cache,
                pos,
                p.extend(["conv"]).unwrap(),
                x,
            )
        };

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm::<3>(
            builder,
            self.config.norm_eps,
            p.extend(["ffn_norm"]).unwrap(),
            x,
        );
        let x = self.feed_forward(builder, p.extend(["feed_forward"]).unwrap(), x);
        x + res
    }
}

impl DynModule for Lfm2Model {
    fn path(&self) -> Path {
        path(vec!["lfm2"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, in_conv]: [Var; 4] = args.try_into().expect("expected 4 inputs");
        let root = self.path();

        let mut cache = Cache::init(
            builder,
            &self.config,
            self.max_sequence_length,
            in_k.clone(),
            in_v,
        );

        // initialize linear state
        cache.linear_state = Some(
            (0..self.num_linear_layers)
                .map(|layer_id| {
                    let layer = slice(builder, 0, layer_id, 1, in_conv.clone());
                    squeeze::<4, 3>(builder, 0, layer)
                })
                .collect(),
        );
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k));

        let mut x = embeddings(
            builder,
            root.extend(vec!["model", "embed_tokens"]).unwrap(),
            x,
        );
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let attention_mask = causal_mask(builder, s, pos.clone());

        for i in 0..self.config.num_hidden_layers {
            x = self.layer(
                builder,
                i,
                attention_mask.clone(),
                &mut cache,
                pos.clone(),
                root.extend(["model", "layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm::<3>(
            builder,
            self.config.norm_eps,
            root.extend(["model", "embedding_norm"]).unwrap(),
            x,
        );

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            root.extend(["model", "embed_tokens"]).unwrap(),
            x,
        );

        x = argmax(builder, x);
        let (out_k, out_v) = cache.get_kv_cache(builder);
        let out_conv = {
            let states = cache
                .linear_state
                .as_ref()
                .expect("lfm2 cache missing output linear state");
            let mut iter = states.iter();
            let first = iter
                .next()
                .expect("lfm2 cache linear state missing")
                .clone();
            let mut out = unsqueeze::<3, 4>(builder, 0, first);
            for state in iter {
                let state = unsqueeze::<3, 4>(builder, 0, state.clone());
                out = concat(builder, 0, out, state);
            }
            out
        };
        vec![x, out_k, out_v, out_conv]
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::*;

        let (mut source, mut target) = llm_type(&self.config);
        let batch_size = NatExpr::Var(0);
        let num_linear_layers = NatExpr::Constant(self.num_linear_layers);
        let hidden_size = NatExpr::Constant(self.config.hidden_size);
        let conv_l_cache = NatExpr::Constant(self.config.conv_l_cache);
        let t_conv = Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
            dtype: DtypeExpr::Constant(Dtype::F32),
            shape: ShapeExpr::Shape(vec![
                num_linear_layers,
                batch_size,
                hidden_size,
                conv_l_cache,
            ]),
        }));
        source.push(t_conv.clone());
        target.push(t_conv);
        (source, target)
    }
}
