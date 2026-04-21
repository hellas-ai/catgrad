use catgrad::prelude::Dtype;
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
use catgrad_llm::run::ModelEngine;
use catgrad_llm::types::{anthropic, openai, plain};
use catgrad_llm::utils::from_json_slice;

// Known limitations of this demo server:
// - User-provided stop strings are ignored; only model-native EOS stopping is supported.
// - /v1/completions only supports string prompts.

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
    engine: ModelEngine,
    model_name: String,
    default_max_tokens: u32,
}

impl InferenceEngine {
    fn new(model: &str, use_kv_cache: bool, default_max_tokens: u32) -> anyhow::Result<Self> {
        Ok(Self {
            engine: ModelEngine::new(model, use_kv_cache, Dtype::F32)?,
            model_name: model.to_string(),
            default_max_tokens,
        })
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

            if let Ok(next) = self.rx.recv() {
                self.current = Cursor::new(next);
            } else {
                self.done = true;
                return Ok(0);
            }
        }
    }
}

#[derive(Clone)]
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

    fn send_error(&self, message: &str) -> anyhow::Result<()> {
        self.send_event_data(
            "error",
            &json!({
                "error": {
                    "message": message
                }
            }),
        )
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
        .map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
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
    let error_sender = sender.clone();
    let response = sse_response(rx);

    let handle = thread::spawn(move || {
        let _ = request.respond(response);
    });

    if let Err(err) = f(sender) {
        log::error!("streaming response failed: {err}");
        let _ = error_sender.send_error(&format!("{err:#}"));
    }
    let _ = handle.join();
}

fn serve_openai(request: Request, engine: &InferenceEngine, req: openai::ChatCompletionRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);
    let stream_include_usage = req
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let prepared = match engine.engine.prepare_openai_chat_request(&req) {
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

            let generated = engine.engine.generate_from_prepared(
                &prepared,
                max_tokens.unwrap_or(engine.default_max_tokens),
                |delta| {
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
                    sse.send_data(&chunk).map_err(std::io::Error::other)?;
                    Ok(())
                },
            )?;

            let final_chunk = openai::ChatCompletionChunk::builder()
                .id(id.clone())
                .object("chat.completion.chunk".to_string())
                .created(created)
                .model(model.clone())
                .choices(vec![
                    openai::ChatStreamChoice::builder()
                        .index(0)
                        .delta(openai::ChatDelta::default())
                        .finish_reason(Some(generated.termination.into()))
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

    let generated = match engine.engine.generate_from_prepared(
        &prepared,
        max_tokens.unwrap_or(engine.default_max_tokens),
        |_| Ok(()),
    ) {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let response = openai::ChatCompletionResponse::builder()
        .id(next_id("chatcmpl"))
        .object("chat.completion".to_string())
        .created(now_unix())
        .model(model)
        .choices(vec![
            openai::ChatChoice::builder()
                .index(0)
                .message(openai::ChatMessage::assistant(generated.text))
                .finish_reason(Some(generated.termination.into()))
                .build(),
        ])
        .usage(Some(openai::Usage::from_counts(
            generated.prompt_tokens,
            generated.completion_tokens,
        )))
        .build();

    let _ = request.respond(json_response(200, &response));
}

fn serve_anthropic(request: Request, engine: &InferenceEngine, req: &anthropic::MessageRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);
    let messages: Vec<_> = req.into();
    let prepared = match engine.engine.prepare_messages(&messages) {
        Ok(prepared) => prepared,
        Err(err) => {
            respond_llm_error(request, err);
            return;
        }
    };

    if stream {
        respond_with_sse_stream(request, |sse| {
            let id = next_id("msg");
            let created_usage = anthropic::AnthropicUsage::new(prepared.input_ids.len() as u32, 0);
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

            let block_start = anthropic::MessageStreamEvent::ContentBlockStart {
                index: 0,
                content_block: anthropic::ContentBlock::Text {
                    text: String::new(),
                },
            };
            sse.send_event_data("content_block_start", &block_start)?;

            let generated =
                engine
                    .engine
                    .generate_from_prepared(&prepared, max_tokens, |delta| {
                        let event = anthropic::MessageStreamEvent::ContentBlockDelta {
                            index: 0,
                            delta: anthropic::ContentBlockDelta::TextDelta {
                                text: delta.to_string(),
                            },
                        };
                        sse.send_event_data("content_block_delta", &event)
                            .map_err(std::io::Error::other)?;
                        Ok(())
                    })?;

            sse.send_event_data(
                "content_block_stop",
                &anthropic::MessageStreamEvent::ContentBlockStop { index: 0 },
            )?;

            let message_delta = anthropic::MessageStreamEvent::MessageDelta {
                delta: anthropic::StreamMessageDelta {
                    stop_reason: Some(generated.termination.into()),
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

    let generated = match engine
        .engine
        .generate_from_prepared(&prepared, max_tokens, |_| Ok(()))
    {
        Ok(out) => out,
        Err(err) => {
            let _ = request.respond(error_response(500, &format!("Inference error: {err}")));
            return;
        }
    };

    let response = anthropic::MessageResponse::builder()
        .id(next_id("msg"))
        .message_type(Some("message".to_string()))
        .role("assistant".to_string())
        .content(vec![anthropic::ContentBlock::Text {
            text: generated.text,
        }])
        .model(model)
        .stop_reason(Some(generated.termination.into()))
        .usage(anthropic::AnthropicUsage::new(
            generated.prompt_tokens,
            generated.completion_tokens,
        ))
        .build();

    let _ = request.respond(json_response(200, &response));
}

fn serve_plain(request: Request, engine: &InferenceEngine, req: &plain::CompletionRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);
    let input = match engine.engine.prepare_prompt(&req.prompt) {
        Ok(input) => input,
        Err(err) => {
            let _ = request.respond(error_response(
                400,
                &format!("Invalid completion prompt: {err}"),
            ));
            return;
        }
    };

    if stream {
        respond_with_sse_stream(request, |sse| {
            let id = next_id("cmpl");
            let created = now_unix();

            let generated = engine.engine.generate_from_prepared(
                &input,
                max_tokens.unwrap_or(engine.default_max_tokens),
                |delta| {
                    let chunk = plain::CompletionChunk::builder()
                        .id(id.clone())
                        .object("text_completion".to_string())
                        .created(created)
                        .model(model.clone())
                        .choices(vec![
                            plain::CompletionChoice::builder()
                                .index(0)
                                .text(delta.to_string())
                                .build(),
                        ])
                        .build();
                    sse.send_data(&chunk).map_err(std::io::Error::other)?;
                    Ok(())
                },
            )?;

            let final_chunk = plain::CompletionChunk::builder()
                .id(id)
                .object("text_completion".to_string())
                .created(created)
                .model(model)
                .choices(vec![
                    plain::CompletionChoice::builder()
                        .index(0)
                        .text(String::new())
                        .finish_reason(Some(generated.termination.into()))
                        .build(),
                ])
                .build();
            sse.send_data(&final_chunk)?;
            sse.send_done()?;
            Ok(())
        });
        return;
    }

    let generated = match engine.engine.generate_from_prepared(
        &input,
        max_tokens.unwrap_or(engine.default_max_tokens),
        |_| Ok(()),
    ) {
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
                .finish_reason(Some(generated.termination.into()))
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
            .unwrap_or_else(|| request.url())
            .to_string();

        let Ok(body) = request_body(&mut request) else {
            let _ = request.respond(error_response(400, "Failed to read request body."));
            continue;
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
                Ok(req) => serve_anthropic(request, &engine, &req),
                Err(err) => {
                    let _ = request.respond(error_response(
                        400,
                        &format!("Invalid Anthropic request: {err}"),
                    ));
                }
            },
            "/v1/completions" => match from_json_slice::<plain::CompletionRequest>(&body) {
                Ok(req) => serve_plain(request, &engine, &req),
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
