use crate::helpers::*;
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SiglipVisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
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

fn default_vision_hidden_size() -> usize {
    1152
}
fn default_vision_intermediate_size() -> usize {
    3072
}

fn default_vision_hidden_layers() -> usize {
    12
}

fn default_vision_num_attention_heads() -> usize {
    16
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
    fn interp_axis(builder: &Builder, target_len: usize, base_len: usize) -> Var {
        let axis = cast(builder, arange(builder, target_len), Dtype::F32);
        if target_len <= 1 || base_len <= 1 {
            let zero = constant(builder, 0.0, &shape(builder, axis.clone()));
            axis * zero
        } else {
            let scale = constant(
                builder,
                (base_len - 1) as f32 / (target_len - 1) as f32,
                &shape(builder, axis.clone()),
            );
            axis * scale
        }
    }

    fn position_embeddings(
        &self,
        builder: &Builder,
        base_patch_grid_side: usize,
        target_patch_grid_height: usize,
        target_patch_grid_width: usize,
        p: Path,
    ) -> Var {
        let pos = param(
            builder,
            &p.extend(["position_embedding", "weight"]).unwrap(),
        );
        let pos_dtype = dtype(builder, pos.clone());
        let [_, dim] = unpack::<2>(builder, shape(builder, pos.clone()));
        if target_patch_grid_height == base_patch_grid_side
            && target_patch_grid_width == base_patch_grid_side
        {
            return reshape(
                builder,
                shape!(
                    builder,
                    1,
                    target_patch_grid_height * target_patch_grid_width,
                    dim
                ),
                pos,
            );
        }
        let pos = cast(builder, pos, Dtype::F32);

        let row = Self::interp_axis(builder, target_patch_grid_height, base_patch_grid_side);
        let col = Self::interp_axis(builder, target_patch_grid_width, base_patch_grid_side);
        let row_floor = floor(builder, row.clone());
        let col_floor = floor(builder, col.clone());
        let row_ceil = clamp(
            builder,
            row_floor.clone() + constant(builder, 1.0, &shape(builder, row_floor.clone())),
            0.0,
            (base_patch_grid_side - 1) as f32,
        );
        let col_ceil = clamp(
            builder,
            col_floor.clone() + constant(builder, 1.0, &shape(builder, col_floor.clone())),
            0.0,
            (base_patch_grid_side - 1) as f32,
        );
        let row_frac = row - row_floor.clone();
        let col_frac = col - col_floor.clone();

        let row_floor_u32 = cast(builder, row_floor, Dtype::U32);
        let col_floor_u32 = cast(builder, col_floor, Dtype::U32);
        let row_ceil_u32 = cast(builder, row_ceil, Dtype::U32);
        let col_ceil_u32 = cast(builder, col_ceil, Dtype::U32);

        let source_width = constant(
            builder,
            base_patch_grid_side as u32,
            &shape(builder, row_floor_u32.clone()),
        );
        let row_floor_base = reshape(
            builder,
            shape!(builder, target_patch_grid_height, 1),
            row_floor_u32 * source_width.clone(),
        );
        let row_ceil_base = reshape(
            builder,
            shape!(builder, target_patch_grid_height, 1),
            row_ceil_u32 * source_width,
        );
        let col_floor_u32 = reshape(
            builder,
            shape!(builder, 1, target_patch_grid_width),
            col_floor_u32,
        );
        let col_ceil_u32 = reshape(
            builder,
            shape!(builder, 1, target_patch_grid_width),
            col_ceil_u32,
        );

        let idx_shape = shape!(builder, target_patch_grid_height, target_patch_grid_width);
        let idx00 = broadcast(builder, idx_shape.clone(), row_floor_base.clone())
            + broadcast(builder, idx_shape.clone(), col_floor_u32.clone());
        let idx01 = broadcast(builder, idx_shape.clone(), row_floor_base)
            + broadcast(builder, idx_shape.clone(), col_ceil_u32.clone());
        let idx10 = broadcast(builder, idx_shape.clone(), row_ceil_base.clone())
            + broadcast(builder, idx_shape.clone(), col_floor_u32);
        let idx11 = broadcast(builder, idx_shape.clone(), row_ceil_base)
            + broadcast(builder, idx_shape.clone(), col_ceil_u32);

        let target_tokens = target_patch_grid_height * target_patch_grid_width;
        let idx00 = reshape(builder, shape!(builder, target_tokens), idx00);
        let idx01 = reshape(builder, shape!(builder, target_tokens), idx01);
        let idx10 = reshape(builder, shape!(builder, target_tokens), idx10);
        let idx11 = reshape(builder, shape!(builder, target_tokens), idx11);

        let row_floor_weight = reshape(
            builder,
            shape!(builder, target_patch_grid_height, 1),
            constant(builder, 1.0, &shape(builder, row_frac.clone())) - row_frac.clone(),
        );
        let row_ceil_weight = reshape(
            builder,
            shape!(builder, target_patch_grid_height, 1),
            row_frac,
        );
        let col_floor_weight = reshape(
            builder,
            shape!(builder, 1, target_patch_grid_width),
            constant(builder, 1.0, &shape(builder, col_frac.clone())) - col_frac.clone(),
        );
        let col_ceil_weight = reshape(
            builder,
            shape!(builder, 1, target_patch_grid_width),
            col_frac,
        );

        let w00 = broadcast(builder, idx_shape.clone(), row_floor_weight.clone())
            * broadcast(builder, idx_shape.clone(), col_floor_weight.clone());
        let w01 = broadcast(builder, idx_shape.clone(), row_floor_weight)
            * broadcast(builder, idx_shape.clone(), col_ceil_weight.clone());
        let w10 = broadcast(builder, idx_shape.clone(), row_ceil_weight.clone())
            * broadcast(builder, idx_shape.clone(), col_floor_weight);
        let w11 = broadcast(builder, idx_shape, row_ceil_weight)
            * broadcast(
                builder,
                shape!(builder, target_patch_grid_height, target_patch_grid_width),
                col_ceil_weight,
            );

        let weight_shape = shape!(builder, target_tokens, 1);
        let w00 = broadcast(
            builder,
            shape!(builder, target_tokens, dim),
            reshape(
                builder,
                weight_shape.clone(),
                reshape(builder, shape!(builder, target_tokens), w00),
            ),
        );
        let w01 = broadcast(
            builder,
            shape!(builder, target_tokens, dim),
            reshape(
                builder,
                weight_shape.clone(),
                reshape(builder, shape!(builder, target_tokens), w01),
            ),
        );
        let w10 = broadcast(
            builder,
            shape!(builder, target_tokens, dim),
            reshape(
                builder,
                weight_shape.clone(),
                reshape(builder, shape!(builder, target_tokens), w10),
            ),
        );
        let w11 = broadcast(
            builder,
            shape!(builder, target_tokens, dim),
            reshape(
                builder,
                weight_shape,
                reshape(builder, shape!(builder, target_tokens), w11),
            ),
        );

        let pos = index(builder, 0, idx00, pos.clone()) * w00
            + index(builder, 0, idx01, pos.clone()) * w01
            + index(builder, 0, idx10, pos.clone()) * w10
            + index(builder, 0, idx11, pos) * w11;
        cast(
            builder,
            reshape(builder, shape!(builder, 1, target_tokens, dim), pos),
            pos_dtype,
        )
    }

    fn attention(builder: &Builder, config: &SiglipVisionConfig, p: Path, x: Var) -> Var {
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

    pub fn mlp(&self, builder: &Builder, config: &SiglipVisionConfig, p: Path, x: Var) -> Var {
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

    pub fn encoder_layer(
        &self,
        builder: &Builder,
        config: &SiglipVisionConfig,
        p: Path,
        x: Var,
    ) -> Var {
        let res = x.clone();
        let x = layernorm(
            builder,
            config.layer_norm_eps,
            p.extend(["layer_norm1"]).unwrap(),
            x,
        );
        let x = Self::attention(builder, config, p.extend(["self_attn"]).unwrap(), x);
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

    fn vision_embeddings(builder: &Builder, config: &SiglipVisionConfig, p: Path, x: Var) -> Var {
        let patch_size = config.patch_size;
        let image_size = config.image_size;
        let num_patches = image_size / patch_size;
        let num_channels = 3;
        let hidden_size = config.hidden_size;
        let [b, _, _, _] = unpack::<4>(builder, shape(builder, x.clone()));

        let dim = num_channels * patch_size * patch_size;
        let x = reshape(
            builder,
            shape!(
                builder,
                b,
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
            shape!(builder, b, num_patches * num_patches, dim),
            x,
        );

        let weight = param(builder, &p.extend(["patch_embedding", "weight"]).unwrap());

        let weight = reshape(builder, shape!(builder, hidden_size, dim), weight);
        let weight = transpose(builder, 0, 1, weight);

        let sh = shape!(builder, b, dim, hidden_size);
        let weight = broadcast(builder, sh, weight);

        let emb = matmul(builder, x, weight);

        let bias = param(builder, &p.extend(["patch_embedding", "bias"]).unwrap());
        let sh = shape(builder, emb.clone());
        let bias = broadcast(builder, sh.clone(), bias);
        let emb = emb + bias;

        let pe = param(
            builder,
            &p.extend(["position_embedding", "weight"]).unwrap(),
        );

        let pe = broadcast(builder, sh, pe);

        emb + pe
    }

    #[allow(clippy::too_many_arguments)]
    fn vision_embeddings_from_patches(
        &self,
        builder: &Builder,
        config: &SiglipVisionConfig,
        base_patch_grid_side: usize,
        target_patch_grid_height: usize,
        target_patch_grid_width: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let hidden_size = config.hidden_size;
        let dim = 3 * config.patch_size * config.patch_size;
        let [b, _, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let weight = param(builder, &p.extend(["patch_embedding", "weight"]).unwrap());
        let weight = reshape(builder, shape!(builder, hidden_size, dim), weight);
        let weight = transpose(builder, 0, 1, weight);
        let weight = broadcast(builder, shape!(builder, b, dim, hidden_size), weight);
        let emb = matmul(builder, x, weight);

        let bias = param(builder, &p.extend(["patch_embedding", "bias"]).unwrap());
        let sh = shape(builder, emb.clone());
        let bias = broadcast(builder, sh.clone(), bias);
        let pe = self.position_embeddings(
            builder,
            base_patch_grid_side,
            target_patch_grid_height,
            target_patch_grid_width,
            p,
        );
        let pe = broadcast(builder, sh, pe);

        emb + bias + pe
    }

    pub fn vision_model(
        &self,
        builder: &Builder,
        config: &SiglipVisionConfig,
        p: Path,
        x: Var,
    ) -> Var {
        let mut x = Self::vision_embeddings(builder, config, p.extend(["embeddings"]).unwrap(), x);
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

    #[allow(clippy::too_many_arguments)]
    pub fn vision_model_from_patches(
        &self,
        builder: &Builder,
        config: &SiglipVisionConfig,
        base_patch_grid_side: usize,
        target_patch_grid_height: usize,
        target_patch_grid_width: usize,
        p: Path,
        x: Var,
    ) -> Var {
        let mut x = self.vision_embeddings_from_patches(
            builder,
            config,
            base_patch_grid_side,
            target_patch_grid_height,
            target_patch_grid_width,
            p.extend(["embeddings"]).unwrap(),
            x,
        );
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
