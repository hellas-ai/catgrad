use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use catgrad_llm::LLMError;
use catgrad_llm::prompt::{
    self, AssistantOutput, ExpectedAssistantOutput, PreparedPrompt, StreamingDecoder,
};
use catgrad_llm::run::{ModelLoader, ModelRunner, ModelTokenizer};
use catgrad_llm::types::{self, LM, Loader, anthropic, openai, plain};
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
        prompt_tokens: u32,
        stop_token_ids: &[i32],
        max_tokens: Option<u32>,
        mut on_text_delta: F,
    ) -> anyhow::Result<GenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        let max_new_tokens = max_tokens.unwrap_or(self.default_max_tokens) as usize;
        let mut runner: ModelRunner = self.loader.load_runner()?;
        let mut decoder = StreamingDecoder::new(&self.tokenizer, stop_token_ids);

        let mut completion_tokens = 0u32;
        for token in runner.complete(context).take(max_new_tokens) {
            let delta = decoder.push_tokens(&[token])?;
            if decoder.is_stopped() {
                break;
            }
            completion_tokens += 1;
            if !delta.is_empty() {
                on_text_delta(&delta)?;
            }
        }

        Ok(GenerationOutput {
            text: decoder.finish(),
            prompt_tokens,
            completion_tokens,
        })
    }

    fn generate_from_prepared<F>(
        &self,
        prepared: &PreparedPrompt,
        max_tokens: Option<u32>,
        on_text_delta: F,
    ) -> anyhow::Result<GenerationOutput>
    where
        F: FnMut(&str) -> anyhow::Result<()>,
    {
        self.generate_from_context(
            prepared.input_ids.clone(),
            prepared.prompt_tokens,
            &prepared.stop_token_ids,
            max_tokens,
            on_text_delta,
        )
    }

    fn prepare_messages(
        &self,
        messages: &[types::Message],
        tools: &[types::ToolSpec],
    ) -> catgrad_llm::Result<PreparedPrompt> {
        self.tokenizer.prepare_messages(messages, tools)
    }

    fn prepare_prompt(&self, prompt: &str) -> catgrad_llm::Result<PreparedPrompt> {
        self.tokenizer.prepare_prompt(prompt)
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

fn to_openai_tool_calls(tool_calls: &[prompt::ToolCall]) -> Vec<openai::MessageToolCall> {
    tool_calls
        .iter()
        .map(prompt::ToolCall::to_openai_tool_call)
        .collect()
}

fn to_anthropic_tool_blocks(tool_calls: &[prompt::ToolCall]) -> Vec<anthropic::ContentBlock> {
    tool_calls
        .iter()
        .map(prompt::ToolCall::to_anthropic_tool_use_block)
        .collect()
}

fn parse_generated_output(
    text: &str,
    expected_output: &ExpectedAssistantOutput,
) -> anyhow::Result<AssistantOutput> {
    Ok(prompt::parse_assistant_output(text, expected_output)?)
}

fn respond_llm_error(request: Request, err: LLMError) {
    let status = match err {
        LLMError::UnsupportedTemplateFeature(_) => 400,
        _ => 500,
    };
    let message = match status {
        400 => err.to_string(),
        _ => format!("Inference error: {err}"),
    };
    let _ = request.respond(error_response(status, &message));
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
    let messages: Vec<types::Message> = req
        .messages
        .into_iter()
        .map(|m| types::Message::OpenAI(Box::new(m)))
        .collect();
    let prepared = match engine.prepare_messages(&messages, &tools) {
        Ok(prepared) => prepared,
        Err(err) => {
            respond_llm_error(request, err);
            return;
        }
    };

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

            let generated = match prepared.expected_output {
                ExpectedAssistantOutput::Text => {
                    engine.generate_from_prepared(&prepared, max_tokens, |delta| {
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
                }
                ExpectedAssistantOutput::ToolCalls { .. } => {
                    engine.generate_from_prepared(&prepared, max_tokens, |_| Ok(()))?
                }
            };

            let parsed = parse_generated_output(&generated.text, &prepared.expected_output)?;
            let mut finish_reason = openai::FinishReason::Stop;

            match parsed {
                AssistantOutput::Text(text) => {
                    if !text.is_empty()
                        && !matches!(prepared.expected_output, ExpectedAssistantOutput::Text)
                    {
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
                AssistantOutput::ToolCalls {
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
                        .map(|(index, call)| call.to_openai_delta_tool_call(index as u32))
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

    let generated = match engine.generate_from_prepared(&prepared, max_tokens, |_| Ok(())) {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let parsed = match parse_generated_output(&generated.text, &prepared.expected_output) {
        Ok(parsed) => parsed,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let (message, finish_reason) = match parsed {
        AssistantOutput::Text(text) => (
            openai::ChatMessage::assistant(text),
            openai::FinishReason::Stop,
        ),
        AssistantOutput::ToolCalls {
            content,
            tool_calls,
        } => (
            openai::ChatMessage::builder()
                .role("assistant".to_string())
                .content(content.map(openai::MessageContent::Text))
                .tool_calls(Some(to_openai_tool_calls(&tool_calls)))
                .build(),
            openai::FinishReason::ToolCalls,
        ),
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
    let messages = req.to_messages();
    let prepared = match engine.prepare_messages(&messages, &tools) {
        Ok(prepared) => prepared,
        Err(err) => {
            respond_llm_error(request, err);
            return;
        }
    };

    if stream {
        respond_with_sse_stream(request, |sse| {
            let id = next_id("msg");
            let created_usage = anthropic::AnthropicUsage::new(prepared.prompt_tokens, 0);
            let message_start = anthropic::MessageStreamEvent::MessageStart {
                message: anthropic::MessageResponse::builder()
                    .id(id)
                    .message_type(Some("message".to_string()))
                    .role("assistant".to_string())
                    .content(vec![])
                    .model(model.clone())
                    .usage(created_usage)
                    .build(),
            };
            sse.send_event_data("message_start", &message_start)?;

            let generated = match prepared.expected_output {
                ExpectedAssistantOutput::Text => {
                    let block_start = anthropic::MessageStreamEvent::ContentBlockStart {
                        index: 0,
                        content_block: anthropic::ContentBlock::Text {
                            text: String::new(),
                            citations: None,
                            cache_control: None,
                        },
                    };
                    sse.send_event_data("content_block_start", &block_start)?;

                    let generated =
                        engine.generate_from_prepared(&prepared, max_tokens, |delta| {
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
                }
                ExpectedAssistantOutput::ToolCalls { .. } => {
                    engine.generate_from_prepared(&prepared, max_tokens, |_| Ok(()))?
                }
            };

            let parsed = parse_generated_output(&generated.text, &prepared.expected_output)?;

            let stop_reason = match parsed {
                AssistantOutput::Text(text) => {
                    if !text.is_empty()
                        && !matches!(prepared.expected_output, ExpectedAssistantOutput::Text)
                    {
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
                AssistantOutput::ToolCalls {
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
                                delta: anthropic::ContentBlockDelta::TextDelta { text: content },
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

    let generated = match engine.generate_from_prepared(&prepared, max_tokens, |_| Ok(())) {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let parsed = match parse_generated_output(&generated.text, &prepared.expected_output) {
        Ok(parsed) => parsed,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let (content, stop_reason) = match parsed {
        AssistantOutput::Text(text) => (
            vec![anthropic::ContentBlock::Text {
                text,
                citations: None,
                cache_control: None,
            }],
            anthropic::StopReason::EndTurn,
        ),
        AssistantOutput::ToolCalls {
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
                let prepared = engine.prepare_prompt(&prompt)?;
                engine.generate_from_prepared(&prepared, max_tokens, on_delta)
            }
            plain::CompletionPrompt::Multiple(prompts) => {
                let prepared = engine.prepare_prompt(&prompts.join("\n"))?;
                engine.generate_from_prepared(&prepared, max_tokens, on_delta)
            }
            plain::CompletionPrompt::Tokens(tokens) => {
                let ctx = u32_tokens_to_i32(tokens)?;
                engine.generate_from_context(
                    ctx.clone(),
                    ctx.len() as u32,
                    engine.tokenizer.stop_token_ids(),
                    max_tokens,
                    on_delta,
                )
            }
            plain::CompletionPrompt::TokenBatches(mut batches) => {
                if batches.is_empty() {
                    engine.generate_from_context(
                        Vec::new(),
                        0,
                        engine.tokenizer.stop_token_ids(),
                        max_tokens,
                        on_delta,
                    )
                } else if batches.len() == 1 {
                    let ctx = u32_tokens_to_i32(batches.remove(0))?;
                    engine.generate_from_context(
                        ctx.clone(),
                        ctx.len() as u32,
                        engine.tokenizer.stop_token_ids(),
                        max_tokens,
                        on_delta,
                    )
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
