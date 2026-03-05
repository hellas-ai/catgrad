use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use catgrad_llm::run::{ModelLoader, ModelRunner, ModelTokenizer};
use catgrad_llm::types::{self, ChatTokenizer, LM, Loader, Tokenizer, anthropic, openai, plain};
use catgrad_llm::utils::from_json_slice;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Parser, Debug)]
struct Args {
    /// Host interface to bind
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Hugging Face model id
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,

    /// Enable KV cache reuse
    #[arg(long, default_value_t = true)]
    use_kv_cache: bool,

    /// Fallback max new tokens when request omits max_tokens
    #[arg(long, default_value_t = 128)]
    default_max_tokens: u32,
}

struct InferenceEngine {
    loader: ModelLoader,
    tokenizer: ModelTokenizer,
    default_max_tokens: u32,
}

struct GenerationOutput {
    text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl InferenceEngine {
    fn new(model: &str, use_kv_cache: bool, default_max_tokens: u32) -> anyhow::Result<Self> {
        let loader = ModelLoader::new(model, use_kv_cache)?;
        let tokenizer = loader.load_tokenizer()?;
        Ok(Self {
            loader,
            tokenizer,
            default_max_tokens,
        })
    }

    fn generate_from_context<F>(
        &self,
        context: Vec<i32>,
        max_tokens: Option<u32>,
        mut on_text_delta: F,
    ) -> anyhow::Result<GenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let prompt_tokens = context.len() as u32;
        let max_new_tokens = max_tokens.unwrap_or(self.default_max_tokens) as usize;
        let mut runner: ModelRunner = self.loader.load_runner()?;

        let mut generated = Vec::new();
        let mut decoded = String::new();
        for token in runner.complete(context).take(max_new_tokens) {
            generated.push(token);

            let next_decoded = self.tokenizer.decode(generated.clone())?;
            let delta = next_decoded
                .strip_prefix(decoded.as_str())
                .unwrap_or(next_decoded.as_str());
            if !delta.is_empty() {
                on_text_delta(delta)?;
            }
            decoded = next_decoded;
        }

        Ok(GenerationOutput {
            text: decoded,
            prompt_tokens,
            completion_tokens: generated.len() as u32,
        })
    }

    fn generate_from_messages<F>(
        &self,
        messages: Vec<types::Message>,
        tools: Vec<types::ToolSpec>,
        max_tokens: Option<u32>,
        on_text_delta: F,
    ) -> anyhow::Result<GenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let context = self.encode_messages(messages, tools)?;
        self.generate_from_context(context, max_tokens, on_text_delta)
    }

    fn generate_from_prompt<F>(
        &self,
        prompt: String,
        max_tokens: Option<u32>,
        on_text_delta: F,
    ) -> anyhow::Result<GenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let context = self.tokenizer.encode(prompt)?;
        self.generate_from_context(context, max_tokens, on_text_delta)
    }

    fn encode_messages(
        &self,
        messages: Vec<types::Message>,
        tools: Vec<types::ToolSpec>,
    ) -> anyhow::Result<Vec<i32>> {
        Ok(self.tokenizer.encode_messages(messages, tools)?)
    }
}

struct ChannelReader {
    rx: Receiver<Vec<u8>>,
    current: Cursor<Vec<u8>>,
    done: bool,
}

impl ChannelReader {
    fn new(rx: Receiver<Vec<u8>>) -> Self {
        Self {
            rx,
            current: Cursor::new(Vec::new()),
            done: false,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let read = self.current.read(buf)?;
            if read > 0 {
                return Ok(read);
            }

            if self.done {
                return Ok(0);
            }

            match self.rx.recv() {
                Ok(next) => {
                    self.current = Cursor::new(next);
                }
                Err(_) => {
                    self.done = true;
                    return Ok(0);
                }
            }
        }
    }
}

struct SseSender {
    tx: Sender<Vec<u8>>,
}

impl SseSender {
    fn new(tx: Sender<Vec<u8>>) -> Self {
        Self { tx }
    }

    fn send_data<T: Serialize>(&self, payload: &T) -> anyhow::Result<()> {
        let mut frame = b"data: ".to_vec();
        frame.extend(serde_json::to_vec(payload)?);
        frame.extend(b"\n\n");
        self.tx
            .send(frame)
            .map_err(|_| anyhow::anyhow!("stream receiver disconnected"))
    }

    fn send_event_data<T: Serialize>(&self, event: &str, payload: &T) -> anyhow::Result<()> {
        let mut frame = b"event: ".to_vec();
        frame.extend(event.as_bytes());
        frame.extend(b"\n");
        frame.extend(b"data: ");
        frame.extend(serde_json::to_vec(payload)?);
        frame.extend(b"\n\n");
        self.tx
            .send(frame)
            .map_err(|_| anyhow::anyhow!("stream receiver disconnected"))
    }

    fn send_done(&self) -> anyhow::Result<()> {
        self.tx
            .send(b"data: [DONE]\n\n".to_vec())
            .map_err(|_| anyhow::anyhow!("stream receiver disconnected"))
    }
}

fn sse_response(rx: Receiver<Vec<u8>>) -> Response<ChannelReader> {
    let headers = vec![
        Header::from_bytes("Content-Type", "text/event-stream")
            .expect("failed to construct content-type header"),
        Header::from_bytes("Cache-Control", "no-cache")
            .expect("failed to construct cache-control header"),
        Header::from_bytes("Connection", "keep-alive")
            .expect("failed to construct connection header"),
    ];

    Response::new(StatusCode(200), headers, ChannelReader::new(rx), None, None)
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn next_id(prefix: &str) -> String {
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{n}")
}

fn json_response<T: Serialize>(status: u16, value: &T) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let mut response = Response::from_data(body).with_status_code(StatusCode(status));
    response.add_header(
        Header::from_bytes("Content-Type", "application/json")
            .expect("failed to construct content-type header"),
    );
    response
}

fn error_response(status: u16, message: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(
        status,
        &json!({
            "error": {
                "message": message
            }
        }),
    )
}

fn request_body(request: &mut Request) -> Result<Vec<u8>, std::io::Error> {
    let mut body = Vec::new();
    request.as_reader().read_to_end(&mut body)?;
    Ok(body)
}

fn u32_tokens_to_i32(tokens: Vec<u32>) -> anyhow::Result<Vec<i32>> {
    tokens
        .into_iter()
        .map(|token| {
            i32::try_from(token).map_err(|_| anyhow::anyhow!("token id {token} overflows i32"))
        })
        .collect()
}

#[derive(Debug)]
enum ParsedAssistantOutput {
    Text(String),
    ToolCalls {
        content: Option<String>,
        tool_calls: Vec<openai::MessageToolCall>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallFormat {
    QwenTags,
    AnthropicBlocks,
}

#[derive(Debug, Deserialize)]
struct ToolCallPayload {
    name: String,
    arguments: serde_json::Value,
}

fn parse_qwen_tool_calls(raw: &str) -> ParsedAssistantOutput {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut remainder = raw;
    let mut visible_text = String::new();
    let mut payloads = Vec::new();

    while let Some(start) = remainder.find(OPEN) {
        visible_text.push_str(&remainder[..start]);
        let after_open = &remainder[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            return ParsedAssistantOutput::Text(raw.to_string());
        };
        payloads.push(after_open[..end].trim().to_string());
        remainder = &after_open[end + CLOSE.len()..];
    }
    visible_text.push_str(remainder);

    if payloads.is_empty() {
        return ParsedAssistantOutput::Text(raw.to_string());
    }

    let mut tool_calls = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let Ok(parsed) = serde_json::from_str::<ToolCallPayload>(&payload) else {
            return ParsedAssistantOutput::Text(raw.to_string());
        };
        let arguments = if let Some(arguments) = parsed.arguments.as_str() {
            arguments.to_string()
        } else {
            parsed.arguments.to_string()
        };
        tool_calls.push(openai::MessageToolCall::Function {
            id: next_id("call"),
            function: openai::FunctionCall {
                name: parsed.name,
                arguments,
            },
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

    ParsedAssistantOutput::ToolCalls {
        content,
        tool_calls,
    }
}

fn parse_anthropic_tool_calls(raw: &str) -> ParsedAssistantOutput {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return ParsedAssistantOutput::Text(String::new());
    }

    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return ParsedAssistantOutput::Text(raw.to_string());
    };

    let blocks: Vec<&serde_json::Value> = match &value {
        serde_json::Value::Array(items) => items.iter().collect(),
        serde_json::Value::Object(map) => {
            if let Some(content) = map.get("content").and_then(|v| v.as_array()) {
                content.iter().collect()
            } else {
                vec![&value]
            }
        }
        _ => return ParsedAssistantOutput::Text(raw.to_string()),
    };

    let mut text = String::new();
    let mut saw_text_block = false;
    let mut tool_calls = Vec::new();

    for block in blocks {
        let Some(obj) = block.as_object() else {
            return ParsedAssistantOutput::Text(raw.to_string());
        };
        let Some(block_type) = obj.get("type").and_then(|v| v.as_str()) else {
            return ParsedAssistantOutput::Text(raw.to_string());
        };
        match block_type {
            "text" => {
                if let Some(block_text) = obj.get("text").and_then(|v| v.as_str()) {
                    text.push_str(block_text);
                    saw_text_block = true;
                }
            }
            "tool_use" => {
                let Some(name) = obj.get("name").and_then(|v| v.as_str()) else {
                    return ParsedAssistantOutput::Text(raw.to_string());
                };
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| next_id("call"));
                let input = obj
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
                let arguments = if let Some(raw_arguments) = input.as_str() {
                    raw_arguments.to_string()
                } else {
                    input.to_string()
                };
                tool_calls.push(openai::MessageToolCall::Function {
                    id,
                    function: openai::FunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
            _ => {}
        }
    }

    if tool_calls.is_empty() {
        if saw_text_block {
            ParsedAssistantOutput::Text(text)
        } else {
            ParsedAssistantOutput::Text(raw.to_string())
        }
    } else {
        let content = {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        };
        ParsedAssistantOutput::ToolCalls {
            content,
            tool_calls,
        }
    }
}

fn parse_tool_calls(raw: &str, format: ToolCallFormat) -> ParsedAssistantOutput {
    match format {
        ToolCallFormat::QwenTags => parse_qwen_tool_calls(raw),
        ToolCallFormat::AnthropicBlocks => parse_anthropic_tool_calls(raw),
    }
}

fn to_delta_tool_call(index: u32, call: &openai::MessageToolCall) -> openai::DeltaToolCall {
    match call {
        openai::MessageToolCall::Function { id, function } => openai::DeltaToolCall::builder()
            .index(index)
            .id(Some(id.clone()))
            .tool_type(Some("function".to_string()))
            .function(Some(
                openai::DeltaFunctionCall::builder()
                    .name(Some(function.name.clone()))
                    .arguments(Some(function.arguments.clone()))
                    .build(),
            ))
            .build(),
        openai::MessageToolCall::Custom { id, custom } => openai::DeltaToolCall::builder()
            .index(index)
            .id(Some(id.clone()))
            .tool_type(Some("custom".to_string()))
            .custom(Some(
                openai::DeltaCustomToolCall::builder()
                    .name(Some(custom.name.clone()))
                    .input(Some(custom.input.clone()))
                    .build(),
            ))
            .build(),
    }
}

fn parse_tool_arguments_map(arguments: &str) -> types::JsonMap {
    match serde_json::from_str::<types::JsonData>(arguments) {
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
                types::JsonData::String(arguments.to_string()),
            );
            map
        }
    }
}

fn to_anthropic_tool_blocks(
    tool_calls: &[openai::MessageToolCall],
) -> Vec<anthropic::ContentBlock> {
    tool_calls
        .iter()
        .map(|call| match call {
            openai::MessageToolCall::Function { id, function } => {
                anthropic::ContentBlock::ToolUse {
                    id: id.clone(),
                    name: function.name.clone(),
                    input: parse_tool_arguments_map(&function.arguments),
                }
            }
            openai::MessageToolCall::Custom { id, custom } => anthropic::ContentBlock::ToolUse {
                id: id.clone(),
                name: custom.name.clone(),
                input: parse_tool_arguments_map(&custom.input),
            },
        })
        .collect()
}

fn template_supports_qwen_tool_calls(chat_template: &str) -> bool {
    chat_template.contains("<tool_call>") && chat_template.contains("</tool_call>")
}

fn template_supports_anthropic_tool_calls(chat_template: &str) -> bool {
    chat_template.contains("tool_use")
}

fn detect_tool_call_format(chat_template: &str) -> Option<ToolCallFormat> {
    if template_supports_qwen_tool_calls(chat_template) {
        Some(ToolCallFormat::QwenTags)
    } else if template_supports_anthropic_tool_calls(chat_template) {
        Some(ToolCallFormat::AnthropicBlocks)
    } else {
        None
    }
}

fn respond_with_sse_stream<F>(request: Request, f: F)
where
    F: FnOnce(SseSender) -> anyhow::Result<()>,
{
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let sender = SseSender::new(tx);
    let response = sse_response(rx);

    let handle = thread::spawn(move || {
        let _ = request.respond(response);
    });

    let _ = f(sender);
    let _ = handle.join();
}

fn serve_openai(request: Request, engine: &InferenceEngine, req: openai::ChatCompletionRequest) {
    let model = req.model;
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);
    let stream_include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let tools: Vec<types::ToolSpec> = req
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    let has_tools = !tools.is_empty();
    let tool_call_format = if has_tools {
        detect_tool_call_format(&engine.tokenizer.chat_template)
    } else {
        None
    };
    if has_tools && tool_call_format.is_none() {
        let _ = request.respond(error_response(
            400,
            "Tools were provided, but the active chat template does not expose a supported tool-call format (<tool_call> tags or Anthropic tool_use blocks).",
        ));
        return;
    }
    let messages: Vec<types::Message> = req
        .messages
        .into_iter()
        .map(types::Message::OpenAI)
        .collect();

    if stream {
        respond_with_sse_stream(request, |sse| {
            let id = next_id("chatcmpl");
            let created = now_unix();

            let start_chunk = openai::ChatCompletionChunk::builder()
                .id(id.clone())
                .object("chat.completion.chunk".to_string())
                .created(created)
                .model(model.clone())
                .choices(vec![
                    openai::ChatStreamChoice::builder()
                        .index(0)
                        .delta(openai::ChatDelta {
                            role: Some("assistant".to_string()),
                            ..Default::default()
                        })
                        .build(),
                ])
                .build();
            sse.send_data(&start_chunk)?;

            let generated = if tool_call_format.is_some() {
                engine.generate_from_messages(messages, tools.clone(), max_tokens, |_| Ok(()))?
            } else {
                engine.generate_from_messages(messages, tools.clone(), max_tokens, |delta| {
                    let chunk = openai::ChatCompletionChunk::builder()
                        .id(id.clone())
                        .object("chat.completion.chunk".to_string())
                        .created(created)
                        .model(model.clone())
                        .choices(vec![
                            openai::ChatStreamChoice::builder()
                                .index(0)
                                .delta(openai::ChatDelta {
                                    content: Some(delta.to_string()),
                                    ..Default::default()
                                })
                                .build(),
                        ])
                        .build();
                    sse.send_data(&chunk)
                })?
            };

            let mut finish_reason = openai::FinishReason::Stop;
            if let Some(format) = tool_call_format {
                match parse_tool_calls(&generated.text, format) {
                    ParsedAssistantOutput::Text(text) => {
                        if !text.is_empty() {
                            let text_chunk = openai::ChatCompletionChunk::builder()
                                .id(id.clone())
                                .object("chat.completion.chunk".to_string())
                                .created(created)
                                .model(model.clone())
                                .choices(vec![
                                    openai::ChatStreamChoice::builder()
                                        .index(0)
                                        .delta(openai::ChatDelta {
                                            content: Some(text),
                                            ..Default::default()
                                        })
                                        .build(),
                                ])
                                .build();
                            sse.send_data(&text_chunk)?;
                        }
                    }
                    ParsedAssistantOutput::ToolCalls {
                        content,
                        tool_calls,
                    } => {
                        if let Some(content) = content {
                            let text_chunk = openai::ChatCompletionChunk::builder()
                                .id(id.clone())
                                .object("chat.completion.chunk".to_string())
                                .created(created)
                                .model(model.clone())
                                .choices(vec![
                                    openai::ChatStreamChoice::builder()
                                        .index(0)
                                        .delta(openai::ChatDelta {
                                            content: Some(content),
                                            ..Default::default()
                                        })
                                        .build(),
                                ])
                                .build();
                            sse.send_data(&text_chunk)?;
                        }

                        let delta_calls: Vec<openai::DeltaToolCall> = tool_calls
                            .iter()
                            .enumerate()
                            .map(|(index, call)| to_delta_tool_call(index as u32, call))
                            .collect();
                        let tool_chunk = openai::ChatCompletionChunk::builder()
                            .id(id.clone())
                            .object("chat.completion.chunk".to_string())
                            .created(created)
                            .model(model.clone())
                            .choices(vec![
                                openai::ChatStreamChoice::builder()
                                    .index(0)
                                    .delta(openai::ChatDelta {
                                        tool_calls: Some(delta_calls),
                                        ..Default::default()
                                    })
                                    .build(),
                            ])
                            .build();
                        sse.send_data(&tool_chunk)?;
                        finish_reason = openai::FinishReason::ToolCalls;
                    }
                }
            }

            let final_chunk = openai::ChatCompletionChunk::builder()
                .id(id.clone())
                .object("chat.completion.chunk".to_string())
                .created(created)
                .model(model.clone())
                .choices(vec![
                    openai::ChatStreamChoice::builder()
                        .index(0)
                        .delta(openai::ChatDelta::default())
                        .finish_reason(Some(finish_reason))
                        .build(),
                ])
                .build();
            sse.send_data(&final_chunk)?;

            if stream_include_usage {
                let usage_chunk = openai::ChatCompletionChunk::builder()
                    .id(id)
                    .object("chat.completion.chunk".to_string())
                    .created(created)
                    .model(model)
                    .choices(vec![])
                    .usage(Some(openai::Usage::from_counts(
                        generated.prompt_tokens,
                        generated.completion_tokens,
                    )))
                    .build();
                sse.send_data(&usage_chunk)?;
            }

            sse.send_done()?;
            Ok(())
        });
        return;
    }

    let generated = match engine.generate_from_messages(messages, tools, max_tokens, |_| Ok(())) {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let (message, finish_reason) = if let Some(format) = tool_call_format {
        match parse_tool_calls(&generated.text, format) {
            ParsedAssistantOutput::Text(text) => (
                openai::ChatMessage::assistant(text),
                openai::FinishReason::Stop,
            ),
            ParsedAssistantOutput::ToolCalls {
                content,
                tool_calls,
            } => (
                openai::ChatMessage::builder()
                    .role("assistant".to_string())
                    .content(content.map(openai::MessageContent::Text))
                    .tool_calls(Some(tool_calls))
                    .build(),
                openai::FinishReason::ToolCalls,
            ),
        }
    } else {
        (
            openai::ChatMessage::assistant(generated.text),
            openai::FinishReason::Stop,
        )
    };

    let response = openai::ChatCompletionResponse::builder()
        .id(next_id("chatcmpl"))
        .object("chat.completion".to_string())
        .created(now_unix())
        .model(model)
        .choices(vec![
            openai::ChatChoice::builder()
                .index(0)
                .message(message)
                .finish_reason(Some(finish_reason))
                .build(),
        ])
        .usage(Some(openai::Usage::from_counts(
            generated.prompt_tokens,
            generated.completion_tokens,
        )))
        .build();

    let _ = request.respond(json_response(200, &response));
}

fn serve_anthropic(request: Request, engine: &InferenceEngine, req: anthropic::MessageRequest) {
    let model = req.model.clone();
    let max_tokens = Some(req.max_tokens);
    let stream = req.stream == Some(true);
    let tools: Vec<types::ToolSpec> = req
        .tools
        .as_ref()
        .map(|tools| tools.iter().cloned().map(Into::into).collect())
        .unwrap_or_default();
    let has_tools = !tools.is_empty();
    let tool_call_format = if has_tools {
        detect_tool_call_format(&engine.tokenizer.chat_template)
    } else {
        None
    };
    if has_tools && tool_call_format.is_none() {
        let _ = request.respond(error_response(
            400,
            "Tools were provided, but the active chat template does not expose a supported tool-call format (<tool_call> tags or Anthropic tool_use blocks).",
        ));
        return;
    }
    let messages = req.to_messages();

    if stream {
        respond_with_sse_stream(request, |sse| {
            let id = next_id("msg");
            let context = engine.encode_messages(messages, tools.clone())?;
            let created_usage = anthropic::AnthropicUsage::new(context.len() as u32, 0);
            let message_start = anthropic::MessageStreamEvent::MessageStart {
                message: anthropic::MessageResponse::builder()
                    .id(id.clone())
                    .message_type(Some("message".to_string()))
                    .role("assistant".to_string())
                    .content(vec![])
                    .model(model.clone())
                    .usage(created_usage)
                    .build(),
            };
            sse.send_event_data("message_start", &message_start)?;

            let generated = if tool_call_format.is_some() {
                engine.generate_from_context(context, max_tokens, |_| Ok(()))?
            } else {
                let block_start = anthropic::MessageStreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: anthropic::ContentBlock::Text {
                        text: String::new(),
                        citations: None,
                        cache_control: None,
                    },
                };
                sse.send_event_data("content_block_start", &block_start)?;

                let generated = engine.generate_from_context(context, max_tokens, |delta| {
                    let event = anthropic::MessageStreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: anthropic::ContentBlockDelta::TextDelta {
                            text: delta.to_string(),
                        },
                    };
                    sse.send_event_data("content_block_delta", &event)
                })?;

                sse.send_event_data(
                    "content_block_stop",
                    &anthropic::MessageStreamEvent::ContentBlockStop { index: 0 },
                )?;
                generated
            };

            let stop_reason = if let Some(format) = tool_call_format {
                match parse_tool_calls(&generated.text, format) {
                    ParsedAssistantOutput::Text(text) => {
                        if !text.is_empty() {
                            sse.send_event_data(
                                "content_block_start",
                                &anthropic::MessageStreamEvent::ContentBlockStart {
                                    index: 0,
                                    content_block: anthropic::ContentBlock::Text {
                                        text: String::new(),
                                        citations: None,
                                        cache_control: None,
                                    },
                                },
                            )?;
                            sse.send_event_data(
                                "content_block_delta",
                                &anthropic::MessageStreamEvent::ContentBlockDelta {
                                    index: 0,
                                    delta: anthropic::ContentBlockDelta::TextDelta { text },
                                },
                            )?;
                            sse.send_event_data(
                                "content_block_stop",
                                &anthropic::MessageStreamEvent::ContentBlockStop { index: 0 },
                            )?;
                        }
                        anthropic::StopReason::EndTurn
                    }
                    ParsedAssistantOutput::ToolCalls {
                        content,
                        tool_calls,
                    } => {
                        let mut index = 0u32;
                        if let Some(content) = content {
                            sse.send_event_data(
                                "content_block_start",
                                &anthropic::MessageStreamEvent::ContentBlockStart {
                                    index,
                                    content_block: anthropic::ContentBlock::Text {
                                        text: String::new(),
                                        citations: None,
                                        cache_control: None,
                                    },
                                },
                            )?;
                            sse.send_event_data(
                                "content_block_delta",
                                &anthropic::MessageStreamEvent::ContentBlockDelta {
                                    index,
                                    delta: anthropic::ContentBlockDelta::TextDelta {
                                        text: content,
                                    },
                                },
                            )?;
                            sse.send_event_data(
                                "content_block_stop",
                                &anthropic::MessageStreamEvent::ContentBlockStop { index },
                            )?;
                            index += 1;
                        }

                        for block in to_anthropic_tool_blocks(&tool_calls) {
                            sse.send_event_data(
                                "content_block_start",
                                &anthropic::MessageStreamEvent::ContentBlockStart {
                                    index,
                                    content_block: block,
                                },
                            )?;
                            sse.send_event_data(
                                "content_block_stop",
                                &anthropic::MessageStreamEvent::ContentBlockStop { index },
                            )?;
                            index += 1;
                        }
                        anthropic::StopReason::ToolUse
                    }
                }
            } else {
                anthropic::StopReason::EndTurn
            };

            let message_delta = anthropic::MessageStreamEvent::MessageDelta {
                delta: anthropic::StreamMessageDelta {
                    stop_reason: Some(stop_reason),
                    ..Default::default()
                },
                usage: anthropic::AnthropicUsage::new(
                    generated.prompt_tokens,
                    generated.completion_tokens,
                ),
            };
            sse.send_event_data("message_delta", &message_delta)?;
            sse.send_event_data("message_stop", &anthropic::MessageStreamEvent::MessageStop)?;
            Ok(())
        });
        return;
    }

    let generated = match engine.generate_from_messages(messages, tools, max_tokens, |_| Ok(())) {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let (content, stop_reason) = if let Some(format) = tool_call_format {
        match parse_tool_calls(&generated.text, format) {
            ParsedAssistantOutput::Text(text) => (
                vec![anthropic::ContentBlock::Text {
                    text,
                    citations: None,
                    cache_control: None,
                }],
                anthropic::StopReason::EndTurn,
            ),
            ParsedAssistantOutput::ToolCalls {
                content,
                tool_calls,
            } => {
                let mut blocks = Vec::new();
                if let Some(content) = content {
                    blocks.push(anthropic::ContentBlock::Text {
                        text: content,
                        citations: None,
                        cache_control: None,
                    });
                }
                blocks.extend(to_anthropic_tool_blocks(&tool_calls));
                (blocks, anthropic::StopReason::ToolUse)
            }
        }
    } else {
        (
            vec![anthropic::ContentBlock::Text {
                text: generated.text,
                citations: None,
                cache_control: None,
            }],
            anthropic::StopReason::EndTurn,
        )
    };

    let response = anthropic::MessageResponse::builder()
        .id(next_id("msg"))
        .message_type(Some("message".to_string()))
        .role("assistant".to_string())
        .content(content)
        .model(model)
        .stop_reason(Some(stop_reason))
        .usage(anthropic::AnthropicUsage::new(
            generated.prompt_tokens,
            generated.completion_tokens,
        ))
        .build();

    let _ = request.respond(json_response(200, &response));
}

fn serve_plain(request: Request, engine: &InferenceEngine, req: plain::CompletionRequest) {
    let model = req.model;
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);

    let generate_plain = |engine: &InferenceEngine,
                          on_delta: &mut dyn FnMut(&str) -> anyhow::Result<()>|
     -> anyhow::Result<GenerationOutput> {
        match req.prompt.clone() {
            plain::CompletionPrompt::Single(prompt) => {
                engine.generate_from_prompt(prompt, max_tokens, on_delta)
            }
            plain::CompletionPrompt::Multiple(prompts) => {
                engine.generate_from_prompt(prompts.join("\n"), max_tokens, on_delta)
            }
            plain::CompletionPrompt::Tokens(tokens) => {
                let ctx = u32_tokens_to_i32(tokens)?;
                engine.generate_from_context(ctx, max_tokens, on_delta)
            }
            plain::CompletionPrompt::TokenBatches(mut batches) => {
                if batches.is_empty() {
                    engine.generate_from_context(Vec::new(), max_tokens, on_delta)
                } else if batches.len() == 1 {
                    let ctx = u32_tokens_to_i32(batches.remove(0))?;
                    engine.generate_from_context(ctx, max_tokens, on_delta)
                } else {
                    Err(anyhow::anyhow!(
                        "multiple token batches are not supported in this demo server"
                    ))
                }
            }
        }
    };

    if stream {
        respond_with_sse_stream(request, |sse| {
            let id = next_id("cmpl");
            let created = now_unix();

            let mut on_delta = |delta: &str| {
                let chunk = plain::CompletionChunk::builder()
                    .id(id.clone())
                    .object("text_completion".to_string())
                    .created(created)
                    .model(model.clone())
                    .choices(vec![
                        plain::CompletionStreamChoice::builder()
                            .index(0)
                            .text(delta.to_string())
                            .build(),
                    ])
                    .build();
                sse.send_data(&chunk)
            };

            let _generated = generate_plain(engine, &mut on_delta)?;

            let final_chunk = plain::CompletionChunk::builder()
                .id(id)
                .object("text_completion".to_string())
                .created(created)
                .model(model)
                .choices(vec![
                    plain::CompletionStreamChoice::builder()
                        .index(0)
                        .text(String::new())
                        .finish_reason(Some(openai::FinishReason::Stop))
                        .build(),
                ])
                .build();
            sse.send_data(&final_chunk)?;
            sse.send_done()?;
            Ok(())
        });
        return;
    }

    let mut ignore = |_delta: &str| Ok(());
    let generated = match generate_plain(engine, &mut ignore) {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(
                400,
                &format!("Invalid completion prompt: {err}"),
            ));
            return;
        }
    };

    let response = plain::CompletionResponse::builder()
        .id(next_id("cmpl"))
        .object("text_completion".to_string())
        .created(now_unix())
        .model(model)
        .choices(vec![
            plain::CompletionChoice::builder()
                .index(0)
                .text(generated.text)
                .finish_reason(Some(openai::FinishReason::Stop))
                .build(),
        ])
        .usage(Some(openai::Usage::from_counts(
            generated.prompt_tokens,
            generated.completion_tokens,
        )))
        .build();

    let _ = request.respond(json_response(200, &response));
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Loading model `{}` (this can take a while)...", args.model);
    let engine = InferenceEngine::new(&args.model, args.use_kv_cache, args.default_max_tokens)?;

    let addr = format!("{}:{}", args.host, args.port);
    let server =
        Server::http(&addr).map_err(|err| anyhow::anyhow!("failed to bind {addr}: {err}"))?;

    println!("catgrad demo API server listening on http://{addr}");
    println!("POST /v1/chat/completions (OpenAI)");
    println!("POST /v1/messages (Anthropic)");
    println!("POST /v1/completions (plain)");

    for mut request in server.incoming_requests() {
        if request.method() != &Method::Post {
            let _ = request.respond(error_response(405, "Only POST is supported."));
            continue;
        }

        let path = request
            .url()
            .split('?')
            .next()
            .unwrap_or(request.url())
            .to_string();

        let body = match request_body(&mut request) {
            Ok(body) => body,
            Err(_) => {
                let _ = request.respond(error_response(400, "Failed to read request body."));
                continue;
            }
        };

        match path.as_str() {
            "/v1/chat/completions" => match from_json_slice::<openai::ChatCompletionRequest>(&body)
            {
                Ok(req) => serve_openai(request, &engine, req),
                Err(err) => {
                    let _ = request.respond(error_response(
                        400,
                        &format!("Invalid OpenAI request: {err}"),
                    ));
                }
            },
            "/v1/messages" => match from_json_slice::<anthropic::MessageRequest>(&body) {
                Ok(req) => serve_anthropic(request, &engine, req),
                Err(err) => {
                    let _ = request.respond(error_response(
                        400,
                        &format!("Invalid Anthropic request: {err}"),
                    ));
                }
            },
            "/v1/completions" => match from_json_slice::<plain::CompletionRequest>(&body) {
                Ok(req) => serve_plain(request, &engine, req),
                Err(err) => {
                    let _ = request.respond(error_response(
                        400,
                        &format!("Invalid completion request: {err}"),
                    ));
                }
            },
            _ => {
                let _ = request.respond(error_response(404, "Unknown route."));
            }
        }
    }

    Ok(())
}
