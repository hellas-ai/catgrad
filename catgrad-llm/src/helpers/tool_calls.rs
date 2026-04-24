use crate::{LLMError, Result};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct ToolUseStep {
    pub assistant_content: String,
    pub assistant_tool_calls_text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

struct ExtractedPayloads<'a> {
    assistant_content: String,
    raw_tool_calls_text: String,
    payloads: Vec<&'a str>,
}

pub fn parse_qwen3_tool_calls(output: &str) -> Result<Option<ToolUseStep>> {
    let Some(extracted) = find_repeated_payloads(output, "<tool_call>", "</tool_call>")? else {
        return Ok(parse_raw_json_tool_call(output));
    };
    build_tool_use_step(
        extracted.assistant_content,
        None,
        extracted.payloads,
        |payload| {
            parse_json_tool_call_payload(payload)
                .map(|tool_call| tool_call.map(|tool_call| vec![tool_call]))
        },
    )
}

pub fn parse_qwen3_5_tool_calls(output: &str) -> Result<Option<ToolUseStep>> {
    let Some(extracted) = find_repeated_payloads(output, "<tool_call>", "</tool_call>")? else {
        return Ok(None);
    };
    build_tool_use_step(
        extracted.assistant_content,
        None,
        extracted.payloads,
        |payload| {
            parse_function_tool_call_payload(payload)
                .map(|tool_call| tool_call.map(|tool_call| vec![tool_call]))
        },
    )
}

pub fn parse_lfm2_tool_calls(output: &str) -> Result<Option<ToolUseStep>> {
    parse_python_tool_calls(output, "<|tool_call_start|>", "<|tool_call_end|>")
}

pub fn parse_olmo3_tool_calls(output: &str) -> Result<Option<ToolUseStep>> {
    parse_python_tool_calls(output, "<function_calls>", "</function_calls>")
}

fn build_tool_use_step<'a, F>(
    assistant_content: String,
    assistant_tool_calls_text: Option<String>,
    payloads: Vec<&'a str>,
    mut parse_payload: F,
) -> Result<Option<ToolUseStep>>
where
    F: FnMut(&'a str) -> Result<Option<Vec<ToolCall>>>,
{
    let mut tool_calls = Vec::new();
    for payload in payloads {
        let Some(mut parsed_payload) = parse_payload(payload)? else {
            return Ok(None);
        };
        tool_calls.append(&mut parsed_payload);
    }
    if tool_calls.is_empty() {
        return Ok(None);
    }
    Ok(Some(ToolUseStep {
        assistant_content,
        assistant_tool_calls_text,
        tool_calls,
    }))
}

fn parse_raw_json_tool_call(output: &str) -> Option<ToolUseStep> {
    parse_json_tool_call_payload(output.trim())
        .ok()
        .flatten()
        .map(|tool_call| ToolUseStep {
            assistant_content: String::new(),
            assistant_tool_calls_text: None,
            tool_calls: vec![tool_call],
        })
}

fn parse_json_tool_call_payload(payload: &str) -> Result<Option<ToolCall>> {
    let Ok(tool_call) = serde_json::from_str::<Value>(payload) else {
        return Ok(None);
    };
    match tool_call {
        Value::Array(tool_calls) => {
            if tool_calls.len() != 1 {
                return Err(LLMError::UnsupportedWireConversion(
                    "json tool call arrays are only supported inside repeated <tool_call> blocks"
                        .to_string(),
                ));
            }
            parse_json_tool_call_value(&tool_calls[0])
        }
        other => parse_json_tool_call_value(&other),
    }
}

fn parse_function_tool_call_payload(payload: &str) -> Result<Option<ToolCall>> {
    let Some((name, function_body)) =
        parse_named_block(payload, "<function=", "</function>", "function")?
    else {
        return Ok(None);
    };

    let mut arguments = Map::new();
    let mut rest = function_body;
    while let Some(parameter_start) = rest.find("<parameter=") {
        let parameter_block = &rest[parameter_start..];
        let Some((parameter_name, parameter_value, consumed)) = parse_named_block_at_start(
            parameter_block,
            "<parameter=",
            "</parameter>",
            "parameter",
        )?
        else {
            return Ok(None);
        };
        arguments.insert(parameter_name, parse_scalar(parameter_value));
        rest = &parameter_block[consumed..];
    }

    Ok(Some(ToolCall { name, arguments }))
}

fn parse_json_arguments(value: &Value) -> Result<Map<String, Value>> {
    match value {
        Value::Object(arguments) => Ok(arguments.clone()),
        Value::String(arguments) => {
            let parsed: Value = serde_json::from_str(arguments)?;
            let Some(arguments) = parsed.as_object() else {
                return Err(LLMError::UnsupportedWireConversion(
                    "json tool arguments must decode to an object".to_string(),
                ));
            };
            Ok(arguments.clone())
        }
        _ => Err(LLMError::UnsupportedWireConversion(
            "json tool arguments must be an object or encoded object string".to_string(),
        )),
    }
}

fn parse_scalar(text: &str) -> Value {
    let text = text.trim();
    serde_json::from_str(text).unwrap_or_else(|_| json!(text))
}

fn parse_json_tool_call_value(tool_call: &Value) -> Result<Option<ToolCall>> {
    let Some(object) = tool_call.as_object() else {
        return Ok(None);
    };
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Ok(None);
    };
    let arguments = object
        .get("arguments")
        .or_else(|| object.get("parameters"))
        .map(parse_json_arguments)
        .transpose()?
        .unwrap_or_default();
    Ok(Some(ToolCall {
        name: name.to_string(),
        arguments,
    }))
}

fn parse_named_block<'a>(
    text: &'a str,
    open_prefix: &str,
    close_tag: &str,
    block_kind: &str,
) -> Result<Option<(String, &'a str)>> {
    let Some(start) = text.find(open_prefix) else {
        return Ok(None);
    };
    let Some((name, body, _consumed)) =
        parse_named_block_at_start(&text[start..], open_prefix, close_tag, block_kind)?
    else {
        return Ok(None);
    };
    Ok(Some((name, body)))
}

fn parse_named_block_at_start<'a>(
    text: &'a str,
    open_prefix: &str,
    close_tag: &str,
    block_kind: &str,
) -> Result<Option<(String, &'a str, usize)>> {
    if !text.starts_with(open_prefix) {
        return Ok(None);
    }

    let header = &text[open_prefix.len()..];
    let name_end = header.find('>').ok_or_else(|| {
        LLMError::UnsupportedWireConversion(format!("unterminated <{block_kind}=...> tag"))
    })?;
    let name = header[..name_end].trim().to_string();
    let body = &header[name_end + 1..];
    let body_end = body.find(close_tag).ok_or_else(|| {
        LLMError::UnsupportedWireConversion(format!("missing {close_tag} in tool call"))
    })?;

    Ok(Some((
        name,
        &body[..body_end],
        open_prefix.len() + name_end + 1 + body_end + close_tag.len(),
    )))
}

fn parse_python_tool_calls(output: &str, start: &str, end: &str) -> Result<Option<ToolUseStep>> {
    let Some(extracted) = find_repeated_payloads(output, start, end)? else {
        return Ok(None);
    };
    build_tool_use_step(
        extracted.assistant_content,
        Some(extracted.raw_tool_calls_text),
        extracted.payloads,
        |payload| parse_python_tool_call_block(payload).map(Some),
    )
}

fn parse_python_tool_call_block(payload: &str) -> Result<Vec<ToolCall>> {
    let payload = payload.trim();
    let inner = payload
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
        .unwrap_or(payload)
        .trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }

    split_top_level(inner, ',')?
        .into_iter()
        .map(parse_python_tool_call)
        .collect()
}

fn parse_python_tool_call(text: &str) -> Result<ToolCall> {
    let text = text.trim();
    let open_paren = text
        .find('(')
        .ok_or_else(|| LLMError::UnsupportedWireConversion(format!("invalid tool call: {text}")))?;
    let close_paren = text
        .rfind(')')
        .ok_or_else(|| LLMError::UnsupportedWireConversion(format!("invalid tool call: {text}")))?;
    if close_paren <= open_paren {
        return Err(LLMError::UnsupportedWireConversion(format!(
            "invalid tool call: {text}"
        )));
    }

    let name = text[..open_paren].trim();
    if name.is_empty() {
        return Err(LLMError::UnsupportedWireConversion(format!(
            "invalid tool call: {text}"
        )));
    }

    let args_text = text[open_paren + 1..close_paren].trim();
    let mut arguments = Map::new();
    if !args_text.is_empty() {
        for arg in split_top_level(args_text, ',')? {
            let eq = find_top_level_char(arg, '=').ok_or_else(|| {
                LLMError::UnsupportedWireConversion(format!("invalid tool argument: {arg}"))
            })?;
            let key = arg[..eq].trim();
            let value = arg[eq + 1..].trim();
            if key.is_empty() {
                return Err(LLMError::UnsupportedWireConversion(format!(
                    "invalid tool argument: {arg}"
                )));
            }
            arguments.insert(key.to_string(), parse_python_value(value)?);
        }
    }

    Ok(ToolCall {
        name: name.to_string(),
        arguments,
    })
}

fn parse_python_value(text: &str) -> Result<Value> {
    let text = text.trim();
    if text.len() >= 2
        && ((text.starts_with('"') && text.ends_with('"'))
            || (text.starts_with('\'') && text.ends_with('\'')))
    {
        return Ok(Value::String(parse_python_string(text)?));
    }

    match text {
        "True" => Ok(Value::Bool(true)),
        "False" => Ok(Value::Bool(false)),
        "None" => Ok(Value::Null),
        _ => serde_json::from_str(text).or_else(|_| Ok(Value::String(text.to_string()))),
    }
}

fn parse_python_string(text: &str) -> Result<String> {
    let quote = text
        .chars()
        .next()
        .ok_or_else(|| LLMError::UnsupportedWireConversion("empty python string".to_string()))?;
    let inner = &text[1..text.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let Some(escaped) = chars.next() else {
                return Err(LLMError::UnsupportedWireConversion(
                    "unterminated python escape".to_string(),
                ));
            };
            out.push(match escaped {
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            });
        } else {
            out.push(ch);
        }
    }
    if !text.ends_with(quote) {
        return Err(LLMError::UnsupportedWireConversion(
            "unterminated python string".to_string(),
        ));
    }
    Ok(out)
}

fn split_top_level(text: &str, separator: char) -> Result<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth_paren = 0usize;
    let mut depth_bracket = 0usize;
    let mut depth_brace = 0usize;
    let mut in_quote: Option<char> = None;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if let Some(quote) = in_quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == quote {
                in_quote = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => in_quote = Some(ch),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            _ if ch == separator && depth_paren == 0 && depth_bracket == 0 && depth_brace == 0 => {
                let part = text[start..idx].trim();
                if !part.is_empty() {
                    parts.push(part);
                }
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if in_quote.is_some() || depth_paren != 0 || depth_bracket != 0 || depth_brace != 0 {
        return Err(LLMError::UnsupportedWireConversion(format!(
            "unterminated tool-call expression: {text}"
        )));
    }

    let part = text[start..].trim();
    if !part.is_empty() {
        parts.push(part);
    }
    Ok(parts)
}

fn find_top_level_char(text: &str, needle: char) -> Option<usize> {
    let mut depth_paren = 0usize;
    let mut depth_bracket = 0usize;
    let mut depth_brace = 0usize;
    let mut in_quote: Option<char> = None;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if let Some(quote) = in_quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == quote {
                in_quote = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => in_quote = Some(ch),
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            '[' => depth_bracket += 1,
            ']' => depth_bracket = depth_bracket.saturating_sub(1),
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            _ if ch == needle && depth_paren == 0 && depth_bracket == 0 && depth_brace == 0 => {
                return Some(idx);
            }
            _ => {}
        }
    }

    None
}

fn find_repeated_payloads<'a>(
    output: &'a str,
    start_token: &str,
    end_token: &str,
) -> Result<Option<ExtractedPayloads<'a>>> {
    let Some(first_start) = output.find(start_token) else {
        return Ok(None);
    };
    let assistant_content = output[..first_start].trim().to_string();
    let mut payloads = Vec::new();
    let mut cursor = first_start;

    while let Some(relative_start) = output[cursor..].find(start_token) {
        let block_start = cursor + relative_start;
        let content_start = block_start + start_token.len();
        let Some(relative_end) = output[content_start..].find(end_token) else {
            return Err(LLMError::UnsupportedWireConversion(format!(
                "unterminated wrapped payload starting with {start_token}"
            )));
        };
        let content_end = content_start + relative_end;
        let block_end = content_end + end_token.len();
        payloads.push(output[content_start..content_end].trim());
        cursor = block_end;
    }

    Ok(Some(ExtractedPayloads {
        assistant_content,
        raw_tool_calls_text: output[first_start..cursor].to_string(),
        payloads,
    }))
}
