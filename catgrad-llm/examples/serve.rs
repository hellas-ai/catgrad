use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::prelude::*;
use catgrad::runtime::{BoundProgram, RuntimeError};
use catgrad_llm::runtime::{BoundProgramText, text_program_from_config};
use catgrad_llm::types::{self, anthropic, openai, plain};
use catgrad_llm::utils::{from_json_slice, get_model, get_model_chat_template, load_model};
use catgrad_llm::{Detokenizer, LLMError, PreparedPrompt};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    #[arg(long, default_value_t = 128)]
    default_max_tokens: u32,
    #[arg(long, default_value_t = 4096)]
    max_sequence_length: usize,
}

type Backend = CandleBackend;

struct Engine {
    bound: std::sync::Arc<BoundProgram<Backend>>,
    tokenizer: tokenizers::Tokenizer,
    chat_template: String,
    tokenizer_config: serde_json::Value,
    eos_token_ids: Vec<i32>,
    model_name: String,
    default_max_tokens: u32,
}

#[derive(Clone, Copy, Debug)]
enum StopReason {
    Stop,
    MaxTokens,
}

impl From<StopReason> for openai::FinishReason {
    fn from(v: StopReason) -> Self {
        match v {
            StopReason::Stop => Self::Stop,
            StopReason::MaxTokens => Self::Length,
        }
    }
}

impl From<StopReason> for anthropic::StopReason {
    fn from(v: StopReason) -> Self {
        match v {
            StopReason::Stop => Self::EndTurn,
            StopReason::MaxTokens => Self::MaxTokens,
        }
    }
}

impl Engine {
    fn new(
        model: &str,
        default_max_tokens: u32,
        max_sequence_length: usize,
    ) -> anyhow::Result<Self> {
        let backend = CandleBackend::new();
        let (pv, _pt, config_json, tokenizer, tokenizer_config, _) =
            load_model(model, "main", &backend, Dtype::F32)?;
        let chat_template = get_model_chat_template(model, "main").unwrap_or_default();
        let mdl = get_model(&config_json, max_sequence_length, None, Dtype::F32)?;
        let eos_token_ids = mdl.config().get_eos_token_ids();

        let spec = text_program_from_config(&config_json, max_sequence_length, Dtype::F32)?;
        let bound = std::sync::Arc::new(BoundProgram::bind(&pv, &backend, spec)?);

        Ok(Self {
            bound,
            tokenizer,
            chat_template,
            tokenizer_config,
            eos_token_ids,
            model_name: model.to_string(),
            default_max_tokens,
        })
    }

    fn prepare_chat_messages(
        &self,
        messages: &[types::Message],
    ) -> Result<PreparedPrompt, LLMError> {
        if self.chat_template.is_empty() {
            return Err(LLMError::InvalidModelConfig(
                "model has no chat template".to_string(),
            ));
        }
        let prompt = PreparedPrompt::from_messages(
            &self.tokenizer,
            &self.chat_template,
            &self.tokenizer_config,
            messages,
            &self.eos_token_ids,
        )?;
        Ok(prompt)
    }

    fn prepare_plain(&self, prompt: &str) -> Result<PreparedPrompt, LLMError> {
        PreparedPrompt::from_prompt(&self.tokenizer, prompt, &self.eos_token_ids)
    }

    fn generate(
        &self,
        prepared: &PreparedPrompt,
        max_tokens: u32,
        mut on_delta: impl FnMut(&str) -> Result<(), LLMError>,
    ) -> anyhow::Result<(u32, StopReason)> {
        let token_ids: Vec<u32> = prepared.input_ids.iter().map(|&id| id as u32).collect();
        let initial_state = self.bound.genesis_text_state();
        let input_tensor = catgrad::interpreter::tensor(
            &self.bound.interpreter().backend,
            catgrad::category::core::Shape(vec![1, token_ids.len()]),
            token_ids,
        )
        .map_err(|e| anyhow::anyhow!("failed to build input tensor: {e:?}"))?;
        let mut session = std::sync::Arc::clone(&self.bound)
            .prefill(&initial_state, &input_tensor)?;
        let mut detok = Detokenizer::from_tokenizer(&self.tokenizer, &prepared.stop_token_ids);

        let mut generated = 0u32;
        let mut reason = StopReason::MaxTokens;
        for _ in 0..max_tokens {
            let next = session.next_token();
            let delta = detok.push_tokens(&[next as i32])?;
            if detok.is_stopped() {
                reason = StopReason::Stop;
                break;
            }
            generated += 1;
            if !delta.is_empty() {
                on_delta(&delta)?;
            }
            session.commit_next()?;
        }
        Ok((generated, reason))
    }
}

// --- SSE helpers ---

struct SseSender(Sender<Vec<u8>>);

impl SseSender {
    fn data<T: Serialize>(&self, payload: &T) -> anyhow::Result<()> {
        let mut frame = b"data: ".to_vec();
        frame.extend(serde_json::to_vec(payload)?);
        frame.extend(b"\n\n");
        self.0
            .send(frame)
            .map_err(|_| anyhow::anyhow!("disconnected"))
    }

    fn event_data<T: Serialize>(&self, event: &str, payload: &T) -> anyhow::Result<()> {
        let mut frame = format!("event: {event}\ndata: ").into_bytes();
        frame.extend(serde_json::to_vec(payload)?);
        frame.extend(b"\n\n");
        self.0
            .send(frame)
            .map_err(|_| anyhow::anyhow!("disconnected"))
    }

    fn done(&self) -> anyhow::Result<()> {
        self.0
            .send(b"data: [DONE]\n\n".to_vec())
            .map_err(|_| anyhow::anyhow!("disconnected"))
    }
}

struct ChannelReader {
    rx: Receiver<Vec<u8>>,
    cur: Cursor<Vec<u8>>,
    done: bool,
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let n = self.cur.read(buf)?;
            if n > 0 {
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            match self.rx.recv() {
                Ok(next) => self.cur = Cursor::new(next),
                Err(_) => {
                    self.done = true;
                    return Ok(0);
                }
            }
        }
    }
}

fn sse_response(rx: Receiver<Vec<u8>>) -> Response<ChannelReader> {
    let reader = ChannelReader {
        rx,
        cur: Cursor::new(Vec::new()),
        done: false,
    };
    let headers = vec![
        Header::from_bytes("Content-Type", "text/event-stream").unwrap(),
        Header::from_bytes("Cache-Control", "no-cache").unwrap(),
    ];
    Response::new(StatusCode(200), headers, reader, None, None)
}

fn stream_response<F>(request: Request, f: F)
where
    F: FnOnce(&SseSender) -> anyhow::Result<()>,
{
    let (tx, rx) = mpsc::channel();
    let sse = SseSender(tx);
    let handle = thread::spawn(move || {
        let _ = request.respond(sse_response(rx));
    });
    if let Err(e) = f(&sse) {
        log::error!("stream error: {e}");
    }
    // Drop the sender first so the spawned response thread's
    // `ChannelReader::read` sees `rx.recv() == Err`, returns 0, and
    // tiny_http's `io::copy` can complete and emit the chunked
    // terminator. Without this, `sse` outlives `handle.join()`, the
    // response thread blocks forever on the channel, and join
    // deadlocks the request handler.
    drop(sse);
    let _ = handle.join();
}

fn json_response<T: Serialize>(status: u16, value: &T) -> Response<Cursor<Vec<u8>>> {
    let body = serde_json::to_vec(value).unwrap_or_default();
    let mut r = Response::from_data(body).with_status_code(StatusCode(status));
    r.add_header(Header::from_bytes("Content-Type", "application/json").unwrap());
    r
}

fn error_response(status: u16, msg: &str) -> Response<Cursor<Vec<u8>>> {
    json_response(status, &json!({"error": {"message": msg}}))
}

fn next_id(prefix: &str) -> String {
    format!("{prefix}-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// --- Route handlers ---

fn serve_openai(request: Request, engine: &Engine, req: openai::ChatCompletionRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens.unwrap_or(engine.default_max_tokens);
    let stream = req.stream == Some(true);
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let messages: Vec<types::Message> =
        req.messages.into_iter().map(types::Message::from).collect();
    let prepared = match engine.prepare_chat_messages(&messages) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        stream_response(request, |sse| {
            let id = next_id("chatcmpl");
            let created = now_unix();
            sse.data(
                &openai::ChatCompletionChunk::builder()
                    .id(id.clone())
                    .object("chat.completion.chunk".into())
                    .created(created)
                    .model(model.clone())
                    .choices(vec![
                        openai::ChatStreamChoice::builder()
                            .index(0)
                            .delta(openai::ChatDelta {
                                role: Some("assistant".into()),
                                ..Default::default()
                            })
                            .build(),
                    ])
                    .build(),
            )?;
            let (completion_tokens, reason) = engine.generate(&prepared, max_tokens, |delta| {
                sse.data(
                    &openai::ChatCompletionChunk::builder()
                        .id(id.clone())
                        .object("chat.completion.chunk".into())
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
                        .build(),
                )
                .map_err(|e| LLMError::Runtime(RuntimeError::ExecutionError(e.to_string())))
            })?;
            sse.data(
                &openai::ChatCompletionChunk::builder()
                    .id(id.clone())
                    .object("chat.completion.chunk".into())
                    .created(created)
                    .model(model.clone())
                    .choices(vec![
                        openai::ChatStreamChoice::builder()
                            .index(0)
                            .delta(openai::ChatDelta::default())
                            .finish_reason(Some(reason.into()))
                            .build(),
                    ])
                    .build(),
            )?;
            if include_usage {
                sse.data(
                    &openai::ChatCompletionChunk::builder()
                        .id(id)
                        .object("chat.completion.chunk".into())
                        .created(created)
                        .model(model)
                        .choices(vec![])
                        .usage(Some(openai::Usage::from_counts(
                            prompt_tokens,
                            completion_tokens,
                        )))
                        .build(),
                )?;
            }
            sse.done()
        });
    } else {
        let mut text = String::new();
        let result = engine.generate(&prepared, max_tokens, |d| {
            text.push_str(d);
            Ok(())
        });
        match result {
            Ok((completion_tokens, reason)) => {
                let _ = request.respond(json_response(
                    200,
                    &openai::ChatCompletionResponse::builder()
                        .id(next_id("chatcmpl"))
                        .object("chat.completion".into())
                        .created(now_unix())
                        .model(model)
                        .choices(vec![
                            openai::ChatChoice::builder()
                                .index(0)
                                .message(openai::ChatMessage::assistant(text))
                                .finish_reason(Some(reason.into()))
                                .build(),
                        ])
                        .usage(Some(openai::Usage::from_counts(
                            prompt_tokens,
                            completion_tokens,
                        )))
                        .build(),
                ));
            }
            Err(e) => drop(request.respond(error_response(500, &e.to_string()))),
        }
    }
}

fn serve_anthropic(request: Request, engine: &Engine, req: &anthropic::MessageRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);
    let messages: Vec<_> = req.into();
    let prepared = match engine.prepare_chat_messages(&messages) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        stream_response(request, |sse| {
            let id = next_id("msg");
            sse.event_data(
                "message_start",
                &anthropic::MessageStreamEvent::MessageStart {
                    message: anthropic::MessageResponse::builder()
                        .id(id)
                        .message_type(Some("message".into()))
                        .role("assistant".into())
                        .content(vec![])
                        .model(model.clone())
                        .usage(anthropic::AnthropicUsage::new(prompt_tokens, 0))
                        .build(),
                },
            )?;
            sse.event_data(
                "content_block_start",
                &anthropic::MessageStreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: anthropic::ContentBlock::Text {
                        text: String::new(),
                    },
                },
            )?;
            let (completion_tokens, reason) = engine.generate(&prepared, max_tokens, |delta| {
                sse.event_data(
                    "content_block_delta",
                    &anthropic::MessageStreamEvent::ContentBlockDelta {
                        index: 0,
                        delta: anthropic::ContentBlockDelta::TextDelta {
                            text: delta.to_string(),
                        },
                    },
                )
                .map_err(|e| LLMError::Runtime(RuntimeError::ExecutionError(e.to_string())))
            })?;
            sse.event_data(
                "content_block_stop",
                &anthropic::MessageStreamEvent::ContentBlockStop { index: 0 },
            )?;
            sse.event_data(
                "message_delta",
                &anthropic::MessageStreamEvent::MessageDelta {
                    delta: anthropic::StreamMessageDelta {
                        stop_reason: Some(reason.into()),
                    },
                    usage: anthropic::AnthropicUsage::new(prompt_tokens, completion_tokens),
                },
            )?;
            sse.event_data("message_stop", &anthropic::MessageStreamEvent::MessageStop)
        });
    } else {
        let mut text = String::new();
        let result = engine.generate(&prepared, max_tokens, |d| {
            text.push_str(d);
            Ok(())
        });
        match result {
            Ok((completion_tokens, reason)) => {
                let _ = request.respond(json_response(
                    200,
                    &anthropic::MessageResponse::builder()
                        .id(next_id("msg"))
                        .message_type(Some("message".into()))
                        .role("assistant".into())
                        .content(vec![anthropic::ContentBlock::Text { text }])
                        .model(model)
                        .stop_reason(Some(reason.into()))
                        .usage(anthropic::AnthropicUsage::new(
                            prompt_tokens,
                            completion_tokens,
                        ))
                        .build(),
                ));
            }
            Err(e) => drop(request.respond(error_response(500, &e.to_string()))),
        }
    }
}

fn serve_plain(request: Request, engine: &Engine, req: &plain::CompletionRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens.unwrap_or(engine.default_max_tokens);
    let stream = req.stream == Some(true);
    let prepared = match engine.prepare_plain(&req.prompt) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        stream_response(request, |sse| {
            let id = next_id("cmpl");
            let created = now_unix();
            let (_completion_tokens, reason) = engine.generate(&prepared, max_tokens, |delta| {
                sse.data(
                    &plain::CompletionChunk::builder()
                        .id(id.clone())
                        .object("text_completion".into())
                        .created(created)
                        .model(model.clone())
                        .choices(vec![
                            plain::CompletionChoice::builder()
                                .index(0)
                                .text(delta.to_string())
                                .build(),
                        ])
                        .build(),
                )
                .map_err(|e| LLMError::Runtime(RuntimeError::ExecutionError(e.to_string())))
            })?;
            sse.data(
                &plain::CompletionChunk::builder()
                    .id(id)
                    .object("text_completion".into())
                    .created(created)
                    .model(model)
                    .choices(vec![
                        plain::CompletionChoice::builder()
                            .index(0)
                            .text(String::new())
                            .finish_reason(Some(reason.into()))
                            .build(),
                    ])
                    .build(),
            )?;
            sse.done()
        });
    } else {
        let mut text = String::new();
        let result = engine.generate(&prepared, max_tokens, |d| {
            text.push_str(d);
            Ok(())
        });
        match result {
            Ok((completion_tokens, reason)) => {
                let _ = request.respond(json_response(
                    200,
                    &plain::CompletionResponse::builder()
                        .id(next_id("cmpl"))
                        .object("text_completion".into())
                        .created(now_unix())
                        .model(model)
                        .choices(vec![
                            plain::CompletionChoice::builder()
                                .index(0)
                                .text(text)
                                .finish_reason(Some(reason.into()))
                                .build(),
                        ])
                        .usage(Some(openai::Usage::from_counts(
                            prompt_tokens,
                            completion_tokens,
                        )))
                        .build(),
                ));
            }
            Err(e) => drop(request.respond(error_response(500, &e.to_string()))),
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    eprintln!("Loading model `{}` ...", args.model);
    let engine = Engine::new(
        &args.model,
        args.default_max_tokens,
        args.max_sequence_length,
    )?;

    let addr = format!("{}:{}", args.host, args.port);
    let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;

    eprintln!("Listening on http://{addr}");
    eprintln!("  POST /v1/chat/completions  (OpenAI)");
    eprintln!("  POST /v1/messages          (Anthropic)");
    eprintln!("  POST /v1/completions       (plain)");

    for mut request in server.incoming_requests() {
        if request.method() != &Method::Post {
            let _ = request.respond(error_response(405, "POST only"));
            continue;
        }
        let path = request
            .url()
            .split('?')
            .next()
            .unwrap_or(request.url())
            .to_string();
        let mut body = Vec::new();
        if request.as_reader().read_to_end(&mut body).is_err() {
            let _ = request.respond(error_response(400, "bad body"));
            continue;
        }
        match path.as_str() {
            "/v1/chat/completions" => match from_json_slice::<openai::ChatCompletionRequest>(&body)
            {
                Ok(req) => serve_openai(request, &engine, req),
                Err(e) => drop(request.respond(error_response(400, &e.to_string()))),
            },
            "/v1/messages" => match from_json_slice::<anthropic::MessageRequest>(&body) {
                Ok(req) => serve_anthropic(request, &engine, &req),
                Err(e) => drop(request.respond(error_response(400, &e.to_string()))),
            },
            "/v1/completions" => match from_json_slice::<plain::CompletionRequest>(&body) {
                Ok(req) => serve_plain(request, &engine, &req),
                Err(e) => drop(request.respond(error_response(400, &e.to_string()))),
            },
            _ => drop(request.respond(error_response(404, "not found"))),
        }
    }
    Ok(())
}
