use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default)]
    pub projection_dim: usize,
    #[serde(default)]
    pub num_image_tokens: usize,
}

fn default_patch_size() -> usize {
    16
}

fn default_image_size() -> usize {
    224
}

fn default_layer_norm_eps() -> f32 {
    1e-6
}

pub struct SiglipVisionBackbone {}

impl SiglipVisionBackbone {
    fn attention(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear(builder, dim, dim, p.extend(["q_proj"]).unwrap(), x.clone());
        let k = linear(builder, dim, dim, p.extend(["k_proj"]).unwrap(), x.clone());
        let v = linear(builder, dim, dim, p.extend(["v_proj"]).unwrap(), x);

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh.clone(), q);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let q = transpose(builder, 1, 2, q);
        let k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);

        let head_dim_float = head_dim as f32;
        let sh_attn = shape(builder, attn.clone());
        let denom = constant(builder, head_dim_float.sqrt(), &sh_attn);
        let attn = attn / denom;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);
        let x = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, dim);
        let x = reshape(builder, sh, x);

        linear(builder, dim, dim, p.extend(["out_proj"]).unwrap(), x)
    }

    pub fn mlp(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let x = linear(
            builder,
            config.hidden_size,
            config.intermediate_size,
            p.extend(["fc1"]).unwrap(),
            x,
        );
        let x = gelu(builder, x);
        linear(
            builder,
            config.intermediate_size,
            config.hidden_size,
            p.extend(["fc2"]).unwrap(),
            x,
        )
    }

    pub fn encoder_layer(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["layer_norm1"]).unwrap(),
            x,
        );
        let x = self.attention(builder, config, p.extend(["self_attn"]).unwrap(), x);
        let x = x + res;

        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["layer_norm2"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, config, p.extend(["mlp"]).unwrap(), x);
        x + res
    }

    fn vision_embeddings(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let patch_size = config.patch_size;
        let image_size = config.image_size;
        let num_patches = image_size / patch_size;
        let num_channels = 3;
        let hidden_size = config.hidden_size;

        let dim = num_channels * patch_size * patch_size;
        let x = reshape(
            builder,
            shape!(
                builder,
                1,
                num_channels,
                num_patches,
                patch_size,
                num_patches,
                patch_size
            ),
            x,
        );

        let x = transpose(builder, 1, 2, x);
        let x = transpose(builder, 2, 4, x);
        let x = transpose(builder, 3, 4, x);

        let x = reshape(
            builder,
            shape!(builder, 1, num_patches * num_patches, dim),
            x,
        );

        let weight = param(builder, &p.extend(["patch_embedding", "weight"]).unwrap());

        let weight = reshape(builder, shape!(builder, hidden_size, dim), weight);
        let weight = transpose(builder, 0, 1, weight);

        let sh = shape!(builder, 1, dim, hidden_size);
        let weight = broadcast(builder, weight, sh);

        let emb = matmul(builder, x, weight);

        let bias = param(builder, &p.extend(["patch_embedding", "bias"]).unwrap());
        let sh = shape(builder, emb.clone());
        let bias = broadcast(builder, bias, sh.clone());
        let emb = emb + bias;

        let pe = param(
            builder,
            &p.extend(["position_embedding", "weight"]).unwrap(),
        );

        let pe = broadcast(builder, pe, sh);

        emb + pe
    }

    pub fn vision_model(&self, builder: &Builder, config: &VisionConfig, p: Path, x: Var) -> Var {
        let mut x = self.vision_embeddings(builder, config, p.extend(["embeddings"]).unwrap(), x);
        for i in 0..config.num_hidden_layers {
            x = self.encoder_layer(
                builder,
                config,
                p.extend(["encoder", "layers", &i.to_string()]).unwrap(),
                x,
            );
        }
        layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["post_layernorm"]).unwrap(),
            x,
        )
    }
}
