#![allow(clippy::too_many_arguments)]
use crate::config::LLMConfig;
use crate::helpers::*;
use crate::models::llama::{LlamaConfig, LlamaModel};
use crate::models::siglip::{SiglipVisionBackbone, SiglipVisionConfig};
use catgrad::prelude::ops::*;
use catgrad::prelude::*;
use nn::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SmolVLM2Config {
    pub text_config: LlamaConfig,
    pub vision_config: SiglipVisionConfig,
    pub image_token_id: usize,
    #[serde(default)]
    #[allow(dead_code)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_scale_factor")]
    pub scale_factor: usize,
}

fn default_scale_factor() -> usize {
    4
}

#[derive(Debug, Clone)]
struct SmolVLM2MultimodalConfig {
    vision_config: SiglipVisionConfig,
    scale_factor: usize,
    text_hidden_size: usize,
    image_token_index: usize,
    mm_tokens_per_image: usize,
}

#[derive(Debug, Clone)]
pub struct SmolVLM2Model {
    language_model: LlamaModel,
    multimodal: SmolVLM2MultimodalConfig,
}

impl LLMModel for SmolVLM2Model {
    fn config(&self) -> &dyn LLMConfig {
        self.language_model.config()
    }

    fn multimodal_metadata(&self) -> Option<MultimodalMetadata> {
        Some(MultimodalMetadata {
            image_token_index: self.multimodal.image_token_index,
            mm_tokens_per_image: self.multimodal.mm_tokens_per_image,
            hidden_size: self.multimodal.text_hidden_size,
            image_size: self.multimodal.vision_config.image_size,
            patch_size: self.multimodal.vision_config.patch_size,
        })
    }

    fn multimodal_vision_module(&self) -> Option<Box<dyn DynModule>> {
        Some(Box::new(SmolVLM2VisionModel {
            vision_config: self.multimodal.vision_config.clone(),
            scale_factor: self.multimodal.scale_factor,
            text_hidden_size: self.multimodal.text_hidden_size,
        }))
    }

    fn multimodal_language_module(&self) -> Option<Box<dyn DynModule>> {
        Some(Box::new(SmolVLM2MultimodalModel {
            language_model: self.language_model.clone(),
        }))
    }

    fn multimodal_interpolate_prompt(&self, prompt: &str) -> Option<String> {
        let mm = &self.multimodal;
        let image_prompt = format!(
            "<fake_token_around_image><global-img>{}<fake_token_around_image>",
            "<image>".repeat(mm.mm_tokens_per_image)
        );
        Some(prompt.replace("<image>", &image_prompt))
    }
}

fn pixel_shuffle(
    builder: &Builder,
    hidden_size: usize,
    patches_per_side: usize,
    scale_factor: usize,
    x: Var,
) -> Var {
    let [b, _, _] = unpack::<3>(builder, shape(builder, x.clone()));
    let height = patches_per_side;
    let width = patches_per_side;
    let scale_sq = scale_factor * scale_factor;
    let output_tokens = (patches_per_side * patches_per_side) / scale_sq;
    let output_hidden_size = hidden_size * scale_sq;

    let x = reshape(builder, shape!(builder, b, height, width, hidden_size), x);
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            height,
            width / scale_factor,
            hidden_size * scale_factor
        ),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    let x = reshape(
        builder,
        shape!(
            builder,
            b,
            width / scale_factor,
            height / scale_factor,
            output_hidden_size
        ),
        x,
    );
    let x = transpose(builder, 1, 2, x);
    reshape(
        builder,
        shape!(builder, b, output_tokens, output_hidden_size),
        x,
    )
}

fn connector(
    builder: &Builder,
    p: Path,
    vision_config: &SiglipVisionConfig,
    scale_factor: usize,
    text_hidden_size: usize,
    x: Var,
) -> Var {
    let patches_per_side = vision_config.image_size / vision_config.patch_size;
    let x = pixel_shuffle(
        builder,
        vision_config.hidden_size,
        patches_per_side,
        scale_factor,
        x,
    );
    linear_no_bias(
        builder,
        vision_config.hidden_size * scale_factor * scale_factor,
        text_hidden_size,
        p.extend(["modality_projection", "proj"]).unwrap(),
        x,
    )
}

#[derive(Debug, Clone)]
pub struct SmolVLM2VisionModel {
    vision_config: SiglipVisionConfig,
    scale_factor: usize,
    text_hidden_size: usize,
}

impl DynModule for SmolVLM2VisionModel {
    fn path(&self) -> Path {
        path(vec!["SmolVLM2Vision"]).expect("invalid model path")
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;

        let t = Type::Tensor(TypeExpr::Var(0));
        (vec![t.clone()], vec![t])
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [pixels]: [Var; 1] = args.try_into().expect("expected 1 input");
        let backbone = SiglipVisionBackbone {};
        let x = backbone.vision_model(
            builder,
            &self.vision_config,
            path(vec!["model", "vision_model"]).unwrap(),
            pixels,
        );
        let x = connector(
            builder,
            path(vec!["model", "connector"]).unwrap(),
            &self.vision_config,
            self.scale_factor,
            self.text_hidden_size,
            x,
        );
        vec![x]
    }
}

#[derive(Debug, Clone)]
pub struct SmolVLM2MultimodalModel {
    language_model: LlamaModel,
}

impl SmolVLM2MultimodalModel {
    fn forward_image_and_texts(
        &self,
        builder: &Builder,
        prefix: Path,
        text1: Var,
        image: Var,
        text2: Var,
        in_k: Var,
        in_v: Var,
    ) -> Vec<Var> {
        let text1 = embeddings(
            builder,
            prefix
                .extend(["model", "text_model", "embed_tokens"])
                .unwrap(),
            text1,
        );
        let text2 = embeddings(
            builder,
            prefix
                .extend(["model", "text_model", "embed_tokens"])
                .unwrap(),
            text2,
        );
        let embeddings = concat(builder, 1, text1, image);
        let embeddings = concat(builder, 1, embeddings, text2);
        let [_b, s, _] = unpack::<3>(builder, shape(builder, embeddings.clone()));
        let [_, _, _, pos, _] = unpack::<5>(builder, shape(builder, in_k.clone()));
        let attention_mask = causal_mask(builder, s, pos);

        self.language_model.forward_embeddings(
            builder,
            prefix,
            attention_mask,
            embeddings,
            in_k,
            in_v,
        )
    }
}

impl DynModule for SmolVLM2MultimodalModel {
    fn path(&self) -> Path {
        path(vec!["SmolVLM2VLM"]).expect("invalid model path")
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        use catgrad::typecheck::TypeExpr;
        let t = Type::Tensor(TypeExpr::Var(0));
        (
            vec![t.clone(), t.clone(), t.clone(), t.clone(), t.clone()],
            vec![t.clone(), t.clone(), t],
        )
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [text1, image, text2, in_k, in_v]: [Var; 5] =
            args.try_into().expect("expected 5 inputs");
        self.forward_image_and_texts(builder, Path::empty(), text1, image, text2, in_k, in_v)
    }
}

impl SmolVLM2Model {
    pub fn new(config_json: &serde_json::Value, max_sequence_length: usize) -> crate::Result<Self> {
        let config: SmolVLM2Config = serde_json::from_value(config_json.clone())?;
        let text_config = config.text_config;
        let text_hidden_size = text_config.hidden_size();
        let vision_config = config.vision_config;
        let mm_tokens_per_image = (vision_config.image_size / vision_config.patch_size).pow(2)
            / config.scale_factor.pow(2);

        Ok(Self {
            language_model: LlamaModel::from_config_with_roots(
                text_config,
                max_sequence_length,
                "model.text_model",
            ),
            multimodal: SmolVLM2MultimodalConfig {
                vision_config,
                scale_factor: config.scale_factor,
                text_hidden_size,
                image_token_index: config.image_token_id,
                mm_tokens_per_image,
            },
        })
    }
}

impl DynModule for SmolVLM2Model {
    fn path(&self) -> Path {
        path(vec!["smolvlm2"]).expect("invalid model path")
    }

    fn def(&self, builder: &Builder, args: Vec<Var>) -> Vec<Var> {
        let [x, in_k, in_v]: [Var; 3] = args.try_into().expect("expected 3 inputs");
        self.language_model
            .forward(builder, self.path(), x, in_k, in_v)
    }

    fn ty(&self) -> (Vec<Type>, Vec<Type>) {
        llm_type(self.config())
    }
}
