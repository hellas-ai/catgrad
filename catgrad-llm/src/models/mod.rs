// `deepseek` is intentionally not declared. The DeepSeek-V3 model required a
// `WeightPostProcess::ConcatMoeExperts` weight transform that no other model
// uses; rather than carry that machinery in the runtime contract for one
// model, we removed it. The source is left in place at `models/deepseek.rs`
// as a reference for future re-introduction.
pub mod gemma3;
pub mod gemma4;
pub mod gpt2;
pub mod gpt_oss;
pub mod granite;
pub mod lfm2;
pub mod llama;
pub mod mistral3;
pub mod nemotron;
pub mod olmo;
pub mod phi3;
pub mod qwen3;
pub mod qwen3_5;
pub mod siglip;
pub mod smolvlm2;
