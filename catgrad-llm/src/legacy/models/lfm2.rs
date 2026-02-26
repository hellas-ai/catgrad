// LFM2 model description
use super::utils::{Cache, Config, ModelBuilder};
use crate::legacy::nn::layers::*;
use crate::legacy::nn::rope::apply_rope_embedding;
use catgrad_legacy::backend::cpu::eval::Builder;
use catgrad_legacy::core::{NdArrayType, Shape, Var};

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
pub struct Lfm2Config {
    #[serde(alias = "block_ff_dim")]
    pub intermediate_size: usize,
    pub block_ffn_dim_multiplier: f32,
    pub block_multiple_of: usize,
    pub block_norm_eps: f32,
    pub block_auto_adjust_ff_dim: bool,
    pub full_attn_idxs: Vec<usize>,
    pub norm_eps: f32,
    pub conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_dim: usize,
    pub conv_dim_out: usize,
    pub num_dense_layers: usize,
}

pub struct Model;

impl ModelBuilder for Model {
    fn build(
        &self,
        builder: &Builder,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        x: Var,
    ) -> Var {
        let tokens = x.label.shape.0[1];
        let emb = Model::embeddings(builder, config, x);

        let mut result = emb;

        for i in 0..config.num_hidden_layers {
            result = Model::layer(
                builder,
                i,
                config,
                cache,
                pos,
                &format!("model.layers.{i}"),
                result,
            );
        }

        result = rmsnorm(
            builder,
            config.lfm2.norm_eps,
            "model.embedding_norm",
            result,
        );

        // Get the logits for the last token only
        if tokens > 1 {
            result = narrow(builder, 1, tokens - 1, 1, result);
        }

        linear_no_bias(
            builder,
            config.hidden_size,
            config.vocab_size,
            "model.embed_tokens",
            result,
        )
    }
}

impl Model {
    pub fn embeddings(builder: &Builder, config: &Config, x: Var) -> Var {
        let t = NdArrayType::new(
            Shape(vec![config.vocab_size, config.hidden_size]),
            config.dtype,
        );
        let weights = parameter(builder, t, "model.embed_tokens.weight".to_string());
        embedding(builder, x, weights)
    }

    pub fn attention(
        builder: &Builder,
        layer_id: usize,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = config.hidden_size / num_heads;
        let b = x.label.shape.0[0];
        let s = x.label.shape.0[1];

        let q = linear_no_bias(builder, dim, dim, &format!("{name}.q_proj"), x.clone());
        let k = linear_no_bias(
            builder,
            dim,
            dim / rep,
            &format!("{name}.k_proj"),
            x.clone(),
        );
        let v = linear_no_bias(builder, dim, dim / rep, &format!("{name}.v_proj"), x);

        let q = reshape(builder, Shape(vec![b, s, num_heads, head_dim]), q);
        let k = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), k);
        let v = reshape(builder, Shape(vec![b, s, num_kv_heads, head_dim]), v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let q = rmsnorm(
            builder,
            config.lfm2.norm_eps,
            &format!("{name}.q_layernorm"),
            q,
        );
        let k = rmsnorm(
            builder,
            config.lfm2.norm_eps,
            &format!("{name}.k_layernorm"),
            k,
        );

        let q = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), q);
        let k = apply_rope_embedding(builder, pos, cache.cos.clone(), cache.sin.clone(), k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = mat_mul(builder, q, tk);
        let denom = constant(builder, attn.label.clone(), f32::sqrt(head_dim as f32));
        let attn = attn / denom;

        let mask = causal_mask(builder, s, attn.label.dtype);
        let mask = expand(builder, attn.label.shape.clone(), mask);
        let attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = mat_mul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let x = reshape(builder, Shape(vec![b, s, dim]), x);

        linear_no_bias(builder, dim, dim, &format!("{name}.out_proj"), x)
    }

    pub fn depthwise_conv1d(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let b = x.label.shape.0[0];
        let h = x.label.shape.0[1];
        let s = x.label.shape.0[2];
        let k = config.lfm2.conv_l_cache;
        if k == 0 {
            return x;
        }

        // Equivalent to Conv1d(..., groups=hidden, kernel_size=k, padding=k-1)[..., :seqlen]
        let x_padded = if k > 1 {
            let pad_t = NdArrayType::new(Shape(vec![b, h, k - 1]), x.label.dtype);
            let pad = constant(builder, pad_t, 0.0);
            concat(builder, 2, pad, x)
        } else {
            x
        };

        // Depthwise kernel: [hidden, 1, k] -> [hidden, k]
        let w_t = NdArrayType::new(Shape(vec![config.hidden_size, 1, k]), config.dtype);
        let conv_weight = parameter(builder, w_t, format!("{name}.conv.weight"));
        let conv_weight = squeeze(builder, 1, conv_weight);

        let x_slice = narrow(builder, 2, 0, s, x_padded.clone());
        let w_slice = narrow(builder, 1, 0, 1, conv_weight.clone());
        let w_slice = unsqueeze(builder, 0, w_slice);
        let w_slice = expand(builder, x_slice.label.shape.clone(), w_slice);
        let mut conv_out = x_slice * w_slice;

        for offset in 1..k {
            let x_slice = narrow(builder, 2, offset, s, x_padded.clone());
            let w_slice = narrow(builder, 1, offset, 1, conv_weight.clone());
            let w_slice = unsqueeze(builder, 0, w_slice);
            let w_slice = expand(builder, x_slice.label.shape.clone(), w_slice);
            conv_out = conv_out + x_slice * w_slice;
        }

        if config.lfm2.conv_bias {
            let b_t = NdArrayType::new(Shape(vec![config.hidden_size]), config.dtype);
            let conv_bias = parameter(builder, b_t, format!("{name}.conv.bias"));
            let conv_bias = unsqueeze(builder, 0, conv_bias);
            let conv_bias = unsqueeze(builder, 2, conv_bias);
            let conv_bias = expand(builder, conv_out.label.shape.clone(), conv_bias);
            conv_out = conv_out + conv_bias;
        }

        conv_out
    }

    // class Lfm2ShortConv(nn.Module):
    //     def __init__(
    //         self,
    //         config: Lfm2Config,
    //         layer_idx: int,
    //     ):
    //         super().__init__()
    //         self.config = config
    //         self.layer_idx = layer_idx
    //         self.L_cache = config.conv_L_cache
    //         self.bias = config.conv_bias

    //         self.conv = nn.Conv1d(
    //             in_channels=config.hidden_size,
    //             out_channels=config.hidden_size,
    //             kernel_size=self.L_cache,
    //             groups=config.hidden_size,
    //             bias=self.bias,
    //             padding=self.L_cache - 1,
    //         )
    //         self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
    //         self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    #[allow(non_snake_case)]
    pub fn short_conv(
        builder: &Builder,
        _layer_id: usize,
        config: &Config,
        _cache: &mut Cache,
        _pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let _seqlen = x.label.shape.0[1];

        let in_proj = linear_no_bias(
            builder,
            config.hidden_size,
            3 * config.hidden_size,
            &format!("{name}.in_proj"),
            x,
        );
        let BCx = transpose(builder, -1, -2, in_proj);
        let BCx = chunk(builder, -2, 3, BCx);
        let B = BCx[0].clone();
        let C = BCx[1].clone();
        let x = BCx[2].clone();

        let Bx = B * x;

        // TODO: cache conv state like KV cache is
        let conv_out = Model::depthwise_conv1d(builder, config, name, Bx);
        let y = C * conv_out;
        let y = transpose(builder, -1, -2, y);
        linear_no_bias(
            builder,
            config.hidden_size,
            config.hidden_size,
            &format!("{name}.out_proj"),
            y,
        )
    }

    pub fn feed_forward(builder: &Builder, config: &Config, name: &str, x: Var) -> Var {
        let mut intermediate_size = config.lfm2.intermediate_size;

        if config.lfm2.block_auto_adjust_ff_dim {
            intermediate_size = 2 * intermediate_size / 3;
            intermediate_size =
                (config.lfm2.block_ffn_dim_multiplier * intermediate_size as f32) as usize;
            intermediate_size = config.lfm2.block_multiple_of
                * intermediate_size.div_ceil(config.lfm2.block_multiple_of);
        }

        let gated = linear_no_bias(
            builder,
            config.hidden_size,
            intermediate_size,
            &format!("{name}.w1"),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            config.hidden_size,
            intermediate_size,
            &format!("{name}.w3"),
            x,
        );
        let x = silu(builder, gated) * up; // SwiGLU

        linear_no_bias(
            builder,
            intermediate_size,
            config.hidden_size,
            &format!("{name}.w2"),
            x,
        )
    }

    pub fn layer(
        builder: &Builder,
        layer_id: usize,
        config: &Config,
        cache: &mut Cache,
        pos: usize,
        name: &str,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.lfm2.norm_eps,
            &format!("{name}.operator_norm"),
            x,
        );
        let x = if config.lfm2.full_attn_idxs.contains(&layer_id)
            || (config.layer_types.len() == config.num_hidden_layers
                && config.layer_types[layer_id] == "full_attention")
        {
            Model::attention(
                builder,
                layer_id,
                config,
                cache,
                pos,
                &format!("{name}.self_attn"),
                x,
            )
        } else {
            Model::short_conv(
                builder,
                layer_id,
                config,
                cache,
                pos,
                &format!("{name}.conv"),
                x,
            )
        };

        let x = res + x;
        let res = x.clone();
        let x = rmsnorm(
            builder,
            config.lfm2.norm_eps,
            &format!("{name}.ffn_norm"),
            x,
        );
        let x = Model::feed_forward(builder, config, &format!("{name}.feed_forward"), x);
        x + res
    }
}
