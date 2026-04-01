#![allow(clippy::too_many_arguments)]
use crate::config::{EosTokenId, LLMConfig};
use crate::helpers::*;
use crate::models::siglip::{SiglipVisionBackbone, SiglipVisionConfig};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use catgrad::stdlib::nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub enum GemmaConfig {
    #[serde(untagged)]
    VLM {
        text_config: GemmaTextConfig,
        vision_config: SiglipVisionConfig,
        image_token_index: usize,
        #[serde(default = "default_mm_tokens_per_image")]
        mm_tokens_per_image: usize,
    },
    #[serde(untagged)]
    Text(GemmaTextConfig),
}

fn default_mm_tokens_per_image() -> usize {
    256
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LinearRopeScaling {
    pub factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GemmaTextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_sliding_window_pattern")]
    #[serde(alias = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub sliding_window: usize,
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,

    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<LinearRopeScaling>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_query_pre_attn_scalar() -> usize {
    256
}

fn default_sliding_window_pattern() -> usize {
    6
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_rope_local_base_freq() -> f32 {
    10000.0
}

fn default_rope_theta() -> f32 {
    1000000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_num_attention_heads() -> usize {
    8
}

fn default_num_key_value_heads() -> usize {
    4
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_head_dim() -> usize {
    256
}

fn default_vocab_size() -> usize {
    262208
}

impl LLMConfig for GemmaTextConfig {
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    fn partial_rotary_factor(&self) -> f32 {
        self.partial_rotary_factor
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }

    fn eos_token_id(&self) -> Option<EosTokenId> {
        self.eos_token_id.clone()
    }
}

#[derive(Debug, Clone)]
struct GemmaMultimodalConfig {
    vision_config: SiglipVisionConfig,
    image_token_index: usize,
    mm_tokens_per_image: usize,
    is_paligemma: bool,
}

#[derive(Debug, Clone)]
pub struct Gemma3Model {
    pub root: String,
    pub config: GemmaTextConfig,
    pub max_sequence_length: usize,
    multimodal: Option<GemmaMultimodalConfig>,
}

impl LLMModel for Gemma3Model {
    fn config(&self) -> &dyn LLMConfig {
        &self.config
    }

    fn multimodal_metadata(&self) -> Option<MultimodalMetadata> {
        let mm = self.multimodal.as_ref()?;
        Some(MultimodalMetadata {
            image_token_index: mm.image_token_index,
            mm_tokens_per_image: mm.mm_tokens_per_image,
            hidden_size: self.config.hidden_size,
            image_size: mm.vision_config.image_size,
            patch_size: mm.vision_config.patch_size,
        })
    }

    fn multimodal_vision_module(&self) -> Option<Box<dyn DynModule>> {
        let mm = self.multimodal.as_ref()?;
        Some(Box::new(VisionEmbeddings::new(
            mm.is_paligemma,
            mm.vision_config.clone(),
        )))
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        let mm = self.multimodal.as_ref()?;
        Some(Box::new(Gemma3MultimodalModel {
            language_model: self.clone(),
            mm_tokens_per_image: mm.mm_tokens_per_image,
            is_paligemma: mm.is_paligemma,
        }))
    }

    fn multimodal_interpolate_prompt(&self, prompt: &str) -> Option<String> {
        let mm = self.multimodal.as_ref()?;
        if mm.is_paligemma {
            let img_seq = format!("{}<bos>", "<image>".repeat(mm.mm_tokens_per_image));
            let p = prompt.replace("<image>", &img_seq);
            Some(format!("{}\n", p))
        } else {
            let img_seq = format!(
                "\n\n<start_of_image>{}<end_of_image>\n\n",
                "<image_soft_token>".repeat(mm.mm_tokens_per_image)
            );
            Some(prompt.replace("<start_of_image>", &img_seq))
        }
    }
}

/// Multi-modal projector for Gemma 3
pub fn multi_modal_projector(builder: &Builder, p: Path, x: Var) -> Var {
    // SigLIP parameters used in Gemma 3
    let hidden_size = 1152;
    let tokens_per_image = 256;
    let image_size = 896;
    let patch_size = 14;
    let patches = image_size / patch_size;

    let x = transpose(builder, 1, 2, x);
    let sh = shape!(builder, 1, hidden_size, patches, patches);
    let x = reshape(builder, sh, x);

    let x = avgpool2d(builder, hidden_size, patches, 4, x);
    let x = reshape(
        builder,
        shape!(builder, 1, hidden_size, tokens_per_image),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    let x = rmsnorm_gemma::<3>(builder, 1e-6, p.extend(["mm_soft_emb_norm"]).unwrap(), x);
    let proj = param(builder, &p.extend(["mm_input_projection_weight"]).unwrap());
    let proj = unsqueeze::<2, 3>(builder, 0, proj);
    matmul(builder, x, proj)
}

pub struct VisionEmbeddings {
    paligemma: bool,
    pub config: SiglipVisionConfig,
    pub vision_tower: SiglipVisionBackbone,
}

impl VisionEmbeddings {
    pub fn new(paligemma: bool, config: SiglipVisionConfig) -> Self {
        Self {
            paligemma,
            config,
            vision_tower: SiglipVisionBackbone {},
        }
    }
}

impl DynModule for VisionEmbeddings {
    fn path(&self) -> Path {
        path(vec!["VisionEmbeddings"]).unwrap()
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;

        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [pixels]: [Var; 1] = args.try_into().expect("expected 1 input");
        let x = self.vision_tower.vision_model(
            builder,
            &self.config,
            path(vec!["vision_tower", "vision_model"]).unwrap(),
            pixels,
        );

        // Gemma3 case.
        if !self.paligemma {
            let x = multi_modal_projector(builder, path(vec!["multi_modal_projector"]).unwrap(), x);
            return vec![x];
        }

        let x = linear(
            builder,
            self.config.hidden_size,
            self.config.hidden_size * 2,
            path(vec!["multi_modal_projector", "linear"]).unwrap(),
            x,
        );
        let scale = (self.config.projection_dim as f32).sqrt();
        let sh = shape(builder, x.clone());
        let scale = constant(builder, scale, &sh);
        let x = x / scale;
        vec![x]
    }
}

pub struct Gemma3MultimodalModel {
    language_model: Gemma3Model,
    mm_tokens_per_image: usize,
    is_paligemma: bool,
}

impl Gemma3MultimodalModel {
    fn bidirectional_mask(
        &self,
        builder: &Builder,
        size: Var,
        img_start: Var,
        img_size: Var,
    ) -> Var {
        let row = arange(builder, size.clone());
        let sh = shape(builder, row.clone());

        let img_end = img_start.clone() + img_size.to_nat(builder);

        let img_start = nat_to_u32(builder, img_start);
        let img_start = broadcast(builder, sh.clone(), img_start);

        let img_end = nat_to_u32(builder, img_end);
        let img_end = broadcast(builder, sh, img_end);

        let img_mask_1 = gte(builder, row.clone(), img_start);
        let img_mask_2 = lt(builder, row, img_end);
        let row = img_mask_1 * img_mask_2;

        let sh = pack::<2>(builder, [size.clone(), size]);
        let row = broadcast(builder, sh.clone(), row);
        let col = transpose(builder, 0, 1, row.clone());
        let mask = row * col;
        let mask = cast(builder, mask, Dtype::F32);
        let one = constant(builder, 1.0, &sh);
        one - mask
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_image_and_texts(
        &self,
        builder: &Builder,
        p: Path,
        text1: Var,
        image: Var,
        text2: Var,
        in_k: Var,
        in_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let text1 = self.language_model.scaled_embeddings(
            builder,
            p.extend(["embed_tokens"]).unwrap(),
            text1,
        );
        let text2 = self.language_model.scaled_embeddings(
            builder,
            p.extend(["embed_tokens"]).unwrap(),
            text2,
        );

        let [_b, img_start, _] = unpack::<3>(builder, shape(builder, text1.clone()));
        let embeddings = concat(builder, 1, text1, image);
        let embeddings = concat(builder, 1, embeddings, text2);

        let [_b, s, _] = unpack::<3>(builder, shape(builder, embeddings.clone()));

        let attention_mask = if self.is_paligemma {
            let sh = shape!(builder, s.clone(), s);
            constant(builder, 0.0, &sh)
        } else {
            let attention_mask = causal_mask(builder, s.clone(), 0.to_nat(builder));
            let image_mask = self.bidirectional_mask(
                builder,
                s,
                img_start,
                self.mm_tokens_per_image.to_nat(builder),
            );
            attention_mask * image_mask
        };

        self.language_model.forward_embeddings(
            builder,
            p,
            attention_mask,
            embeddings,
            in_k,
            in_v,
            max_positions,
        )
    }
}

impl DynModule for Gemma3MultimodalModel {
    fn path(&self) -> Path {
        path(vec!["VLM"]).unwrap()
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::{NatExpr, TypeExpr};
        let t = Type::Tensor(TypeExpr::Var(0));
        (
            vec![
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                t.clone(),
                Type::Nat(NatExpr::Var(3)),
            ],
            vec![t.clone(), t.clone(), t],
        )
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [text1, image, text2, in_k, in_v, max_positions]: [Var; 6] =
            args.try_into().expect("expected 6 inputs");

        let mut root = Path::empty();
        if !self.language_model.root.is_empty() {
            root = root.extend(self.language_model.root.split('.')).unwrap();
        }
        self.forward_image_and_texts(
            builder,
            root,
            text1,
            image,
            text2,
            in_k,
            in_v,
            max_positions,
        )
    }
}

impl Gemma3Model {
    pub fn new(
        root: &str,
        config_json: &serde_json::Value,
        max_sequence_length: usize,
    ) -> crate::Result<Self> {
        let (config, multimodal) = match serde_json::from_value(config_json.clone())? {
            GemmaConfig::Text(config) => (config, None),
            GemmaConfig::VLM {
                mut text_config,
                vision_config,
                image_token_index,
                mut mm_tokens_per_image,
            } => {
                let is_paligemma = text_config.model_type == "gemma2";
                if is_paligemma {
                    text_config.max_position_embeddings = 8192;
                    text_config.rope_theta = 10000.0;
                    mm_tokens_per_image = vision_config.num_image_tokens;
                }
                (
                    text_config,
                    Some(GemmaMultimodalConfig {
                        vision_config,
                        image_token_index,
                        mm_tokens_per_image,
                        is_paligemma,
                    }),
                )
            }
        };
        Ok(Gemma3Model {
            root: root.to_string(),
            config,
            max_sequence_length,
            multimodal,
        })
    }

    fn is_gemma3(&self) -> bool {
        self.config.model_type == "gemma3_text"
    }

    fn is_local_attention_layer(&self, layer_id: usize) -> bool {
        self.is_gemma3() && !(layer_id + 1).is_multiple_of(self.config.sliding_window_pattern)
    }

    fn normalize(&self, builder: &Builder, x: Var) -> Var {
        let sh = shape(builder, x.clone());
        let normalizer = constant(builder, f32::sqrt(self.config.hidden_size as f32), &sh);
        x * normalizer
    }

    fn softcap(&self, builder: &Builder, softcap: f32, x: Var) -> Var {
        let sh = shape(builder, x.clone());
        let s = constant(builder, softcap, &sh);
        let x = x / s.clone();
        let x = tanh(builder, x);
        x * s
    }
    fn mlp(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let gate = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            p.extend(["gate_proj"]).unwrap(),
            x.clone(),
        );
        let up = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.intermediate_size,
            p.extend(["up_proj"]).unwrap(),
            x,
        );
        let x = gelu(builder, gate) * up;
        linear_no_bias(
            builder,
            self.config.intermediate_size,
            self.config.hidden_size,
            p.extend(["down_proj"]).unwrap(),
            x,
        )
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
        let num_kv_heads = self.config.num_key_value_heads;
        let rep = num_heads / num_kv_heads;
        let head_dim = self.config.head_dim;

        let [b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));

        let q = linear_no_bias(
            builder,
            dim,
            num_heads * head_dim,
            p.extend(["q_proj"]).unwrap(),
            x.clone(),
        );

        let k = linear_no_bias(
            builder,
            dim,
            num_kv_heads * head_dim,
            p.extend(["k_proj"]).unwrap(),
            x.clone(),
        );

        let v = linear_no_bias(
            builder,
            dim,
            num_kv_heads * head_dim,
            p.extend(["v_proj"]).unwrap(),
            x,
        );

        let sh = shape!(builder, b, s, num_heads, head_dim);
        let q = reshape(builder, sh, q);

        let sh = shape!(builder, b, s, num_kv_heads, head_dim);
        let k = reshape(builder, sh.clone(), k);
        let v = reshape(builder, sh, v);

        let mut q = transpose(builder, 1, 2, q);
        let mut k = transpose(builder, 1, 2, k);
        let v = transpose(builder, 1, 2, v);

        // Norm
        if self.is_gemma3() {
            let sh = shape!(
                builder,
                b.clone() * s.clone() * num_heads.to_nat(builder),
                head_dim
            );
            q = reshape(builder, sh, q);

            let sh = shape!(
                builder,
                b.clone() * s.clone() * num_kv_heads.to_nat(builder),
                head_dim
            );

            k = reshape(builder, sh, k);

            q = rmsnorm_gemma::<2>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["q_norm"]).unwrap(),
                q,
            );
            k = rmsnorm_gemma::<2>(
                builder,
                self.config.rms_norm_eps,
                p.extend(["k_norm"]).unwrap(),
                k,
            );
        }

        let sh = shape!(builder, b, num_heads, s, head_dim);
        let q = reshape(builder, sh, q);
        let sh = shape!(builder, b, num_kv_heads, s, head_dim);
        let k = reshape(builder, sh, k);

        // Every Nth layer of Gemma3 uses global attention, otherwise local attention.
        let is_local_attention = self.is_local_attention_layer(layer_id);

        let theta = if is_local_attention {
            self.config.rope_local_base_freq
        } else {
            self.config.rope_theta
        };

        let factor = if is_local_attention {
            1.0
        } else {
            self.config
                .rope_scaling
                .as_ref()
                .map_or(1.0, |rs| rs.factor)
        };

        let q = rope(builder, theta, pos.clone(), &s, head_dim, factor, q);
        let k = rope(builder, theta, pos, &s, head_dim, factor, k);

        let (k, v) = cache.update_kv_cache(builder, layer_id, k, v);

        let k = repeat_kv(builder, rep, k);
        let v = repeat_kv(builder, rep, v);

        let tk = transpose(builder, 2, 3, k);
        let attn = matmul(builder, q, tk);
        let sh = shape(builder, attn.clone());
        let denom = constant(
            builder,
            f32::sqrt(self.config.query_pre_attn_scalar as f32),
            &sh,
        );
        let mut attn = attn / denom;

        if let Some(softcap) = self.config.attn_logit_softcapping {
            attn = self.softcap(builder, softcap, attn);
        }

        let mask = broadcast(builder, sh, attention_mask);
        attn = attn + mask;

        let attn = softmax(builder, attn);
        let attn = matmul(builder, attn, v);

        let attn = transpose(builder, 1, 2, attn);
        let sh = shape!(builder, b, s, num_heads * head_dim);
        let attn = reshape(builder, sh, attn);

        linear_no_bias(
            builder,
            num_heads * head_dim,
            dim,
            p.extend(["o_proj"]).unwrap(),
            attn,
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
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["input_layernorm"]).unwrap(),
            x,
        );
        let x = self.attention(
            builder,
            layer_id,
            attention_mask,
            cache,
            pos,
            p.extend(["self_attn"]).unwrap(),
            x,
        );
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_attention_layernorm"]).unwrap(),
            x,
        );
        let x = res + x;
        let res = x.clone();
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["pre_feedforward_layernorm"]).unwrap(),
            x,
        );
        let x = self.mlp(builder, p.extend(["mlp"]).unwrap(), x);
        let x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["post_feedforward_layernorm"]).unwrap(),
            x,
        );
        x + res
    }

    pub fn scaled_embeddings(&self, builder: &Builder, p: Path, x: Var) -> Var {
        let x = embeddings(builder, p, x);

        // Gemma3 uses scaled embeddings for text only, Gemma2 scales all inputs later
        if self.is_gemma3() {
            self.normalize(builder, x)
        } else {
            x
        }
    }

    // Forward pass with text tokens as input
    pub fn forward(
        &self,
        builder: &Builder,
        p: Path,
        x: Var,
        in_k: Var,
        in_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let x = self.scaled_embeddings(builder, p.extend(["embed_tokens"]).unwrap(), x);
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k.clone()));
        let attention_mask = causal_mask(builder, s, pos);

        self.forward_embeddings(builder, p, attention_mask, x, in_k, in_v, max_positions)
    }

    // Forward pass with embeddings available.
    pub fn forward_embeddings(
        &self,
        builder: &Builder,
        p: Path,
        attention_mask: Var,
        x: Var,
        in_k: Var,
        in_v: Var,
        max_positions: Var,
    ) -> Vec<Var> {
        let mut cache = Cache::init(builder, &self.config, max_positions, in_k.clone(), in_v);
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k));
        let [_b, s, _] = unpack::<3>(builder, shape(builder, x.clone()));
        let sliding_attention_mask =
            sliding_window_mask(builder, s, pos.clone(), self.config.sliding_window);

        let mut x = if self.is_gemma3() {
            x
        } else {
            self.normalize(builder, x)
        };

        for i in 0..self.config.num_hidden_layers {
            let layer_mask = if self.is_local_attention_layer(i) {
                sliding_attention_mask.clone()
            } else {
                attention_mask.clone()
            };
            x = self.layer(
                builder,
                i,
                layer_mask,
                &mut cache,
                pos.clone(),
                p.extend(["layers", &i.to_string()]).unwrap(),
                x,
            );
        }

        x = rmsnorm_gemma::<3>(
            builder,
            self.config.rms_norm_eps,
            p.extend(["norm"]).unwrap(),
            x,
        );

        x = linear_no_bias(
            builder,
            self.config.hidden_size,
            self.config.vocab_size,
            p.extend(["embed_tokens"]).unwrap(),
            x,
        );

        if let Some(softcap) = self.config.final_logit_softcapping {
            x = self.softcap(builder, softcap, x);
        }

        x = argmax(builder, x);
        let (out_k, out_v) = cache.get_kv_cache(builder);
        vec![x, out_k, out_v]
    }
}

impl DynModule for Gemma3Model {
    fn path(&self) -> Path {
        path(vec!["gemma3"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v, max_positions]: [Var; 4] = args.try_into().expect("expected 4 inputs");
        let mut root = self.path();
        if !self.root.is_empty() {
            root = root.extend(self.root.split('.')).unwrap();
        }
        self.forward(builder, root, x, in_k, in_v, max_positions)
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        llm_type(&self.config, self.dtype())
    }
}
