pub mod decode;
pub mod parse;
pub mod prepare;

pub use decode::{StreamingDecoder, TokenDecoder};
pub use parse::{AssistantOutput, ToolCall, parse_assistant_output};
pub use prepare::{
    ExpectedAssistantOutput, PreparedPrompt, RenderedPrompt, ToolCallSyntax,
    detect_tool_call_syntax,
};
