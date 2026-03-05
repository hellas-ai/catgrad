use crate::{Result, types};
use serde::Deserialize;

use super::prepare::{ExpectedAssistantOutput, ToolCallSyntax};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantOutput {
    Text(String),
    ToolCalls {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
}

#[derive(Debug, Deserialize)]
struct ToolCallPayload {
    name: String,
    arguments: serde_json::Value,
}

pub fn parse_assistant_output(
    raw_text: &str,
    expected: &ExpectedAssistantOutput,
) -> Result<AssistantOutput> {
    Ok(match expected {
        ExpectedAssistantOutput::Text => AssistantOutput::Text(raw_text.to_string()),
        ExpectedAssistantOutput::ToolCalls { syntax } => match syntax {
            ToolCallSyntax::QwenXml => parse_qwen_tool_calls(raw_text),
        },
    })
}

impl ToolCall {
    pub fn to_openai_tool_call(&self) -> types::openai::MessageToolCall {
        types::openai::MessageToolCall::Function {
            id: self.id.clone(),
            function: types::openai::FunctionCall {
                name: self.name.clone(),
                arguments: self.arguments.clone(),
            },
        }
    }

    pub fn arguments_json_map(&self) -> types::JsonMap {
        match serde_json::from_str::<types::JsonData>(&self.arguments) {
            Ok(types::JsonData::Object(map)) => map,
            Ok(other) => {
                let mut map = types::JsonMap::new();
                map.insert("_value".to_string(), other);
                map
            }
            Err(_) => {
                let mut map = types::JsonMap::new();
                map.insert(
                    "_raw".to_string(),
                    types::JsonData::String(self.arguments.clone()),
                );
                map
            }
        }
    }

    pub fn to_anthropic_tool_use_block(&self) -> types::anthropic::ContentBlock {
        types::anthropic::ContentBlock::ToolUse {
            id: self.id.clone(),
            name: self.name.clone(),
            input: self.arguments_json_map(),
        }
    }
}

fn parse_qwen_tool_calls(raw: &str) -> AssistantOutput {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut remainder = raw;
    let mut visible_text = String::new();
    let mut payloads = Vec::new();

    while let Some(start) = remainder.find(OPEN) {
        visible_text.push_str(&remainder[..start]);
        let after_open = &remainder[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            return AssistantOutput::Text(raw.to_string());
        };
        payloads.push(after_open[..end].trim().to_string());
        remainder = &after_open[end + CLOSE.len()..];
    }
    visible_text.push_str(remainder);

    if payloads.is_empty() {
        return AssistantOutput::Text(raw.to_string());
    }

    let mut tool_calls = Vec::with_capacity(payloads.len());
    for (index, payload) in payloads.into_iter().enumerate() {
        let Ok(parsed) = serde_json::from_str::<ToolCallPayload>(&payload) else {
            return AssistantOutput::Text(raw.to_string());
        };
        let arguments = if let Some(arguments) = parsed.arguments.as_str() {
            arguments.to_string()
        } else {
            parsed.arguments.to_string()
        };
        tool_calls.push(ToolCall {
            id: format!("call_{index}"),
            name: parsed.name,
            arguments,
        });
    }

    let content = {
        let trimmed = visible_text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    };

    AssistantOutput::ToolCalls {
        content,
        tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_tool_calls_parse_into_generic_output() {
        let raw =
            r#"Before<tool_call>{"name":"lookup","arguments":{"city":"Paris"}}</tool_call>After"#;
        let parsed = parse_assistant_output(
            raw,
            &ExpectedAssistantOutput::ToolCalls {
                syntax: ToolCallSyntax::QwenXml,
            },
        )
        .unwrap();

        assert_eq!(
            parsed,
            AssistantOutput::ToolCalls {
                content: Some("BeforeAfter".to_string()),
                tool_calls: vec![ToolCall {
                    id: "call_0".to_string(),
                    name: "lookup".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                }],
            }
        );
    }

    #[test]
    fn malformed_tool_payload_falls_back_to_text() {
        let raw = r#"<tool_call>{"name":"lookup","arguments":</tool_call>"#;
        let parsed = parse_assistant_output(
            raw,
            &ExpectedAssistantOutput::ToolCalls {
                syntax: ToolCallSyntax::QwenXml,
            },
        )
        .unwrap();

        assert_eq!(parsed, AssistantOutput::Text(raw.to_string()));
    }

    #[test]
    fn tool_arguments_map_preserves_non_object_arguments() {
        let call = ToolCall {
            id: "call_0".to_string(),
            name: "lookup".to_string(),
            arguments: r#""Paris""#.to_string(),
        };
        let map = call.arguments_json_map();
        assert_eq!(
            map.get("_value"),
            Some(&types::JsonData::String("Paris".to_string()))
        );
    }
}
