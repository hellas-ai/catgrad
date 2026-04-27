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
use catgrad_llm::runtime::chat::wire::anthropic::{AnthropicStreamFrame, AnthropicStreamMapper};
use catgrad_llm::runtime::chat::wire::openai::{OpenAiStreamFrame, OpenAiStreamMapper};
use catgrad_llm::runtime::chat::wire::{PumpError, pump_finish, pump_text};
use catgrad_llm::runtime::chat::{
    ChatOptions, ChatTurn, ChatTurnConfigError, DecodeFailure,
    StopReason as ParserStopReason, ToolDirectory,
};
use catgrad_llm::runtime::{
    BoundProgramText, BreakReason, DecodeOutcome, run_decode, text_program_from_config,
};
use catgrad_llm::types::{self, anthropic, openai, plain};
use catgrad_llm::utils::{
    from_json_slice, get_model, get_model_architecture, get_model_chat_template, load_model,
};
use catgrad_llm::{Detokenizer, LLMError, PreparedPrompt};
use std::sync::Arc;

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
    bound: Arc<BoundProgram<Backend>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    chat_template: Arc<str>,
    tokenizer_config: Arc<serde_json::Value>,
    config: serde_json::Value,
    eos_token_ids: Arc<[i32]>,
    model_name: String,
    default_max_tokens: u32,
}

/// Errors `Engine::chat_turn` produces. Mapped to HTTP status by the
/// surface handlers (all are 400 today). Wire-tools shape errors
/// are caught earlier by `ToolDirectory::from_*_tools` at the surface
/// edge.
#[derive(Debug)]
enum ChatTurnError {
    NoChatTemplate,
    ChatTurnConfig(ChatTurnConfigError),
    Other(LLMError),
}

impl From<ChatTurnConfigError> for ChatTurnError {
    fn from(e: ChatTurnConfigError) -> Self {
        Self::ChatTurnConfig(e)
    }
}

impl std::fmt::Display for ChatTurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoChatTemplate => write!(f, "model has no chat template"),
            Self::ChatTurnConfig(e) => write!(f, "{e}"),
            Self::Other(e) => write!(f, "{e}"),
        }
    }
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
        let chat_template = get_model_chat_template(model, "main").ok();
        let mdl = get_model(&config_json, max_sequence_length, None, Dtype::F32)?;
        let eos_token_ids = mdl.config().get_eos_token_ids();

        let spec = text_program_from_config(&config_json, max_sequence_length, Dtype::F32)?;
        let bound = Arc::new(BoundProgram::bind(&pv, &backend, spec)?);

        Ok(Self {
            bound,
            tokenizer: Arc::new(tokenizer),
            chat_template: Arc::from(chat_template.unwrap_or_default()),
            tokenizer_config: Arc::new(tokenizer_config),
            config: config_json,
            eos_token_ids: Arc::from(eos_token_ids.as_slice()),
            model_name: model.to_string(),
            default_max_tokens,
        })
    }

    fn prepare_plain(&self, prompt: &str) -> Result<PreparedPrompt, LLMError> {
        PreparedPrompt::from_prompt(&self.tokenizer, prompt, &self.eos_token_ids)
    }

    /// Build a `ChatTurn` for one chat-completion request. The caller
    /// supplies an already-built [`ToolDirectory`] (or `None` for no
    /// tools) — wire-shape conversion happens at the gateway edge via
    /// `ToolDirectory::from_openai_tools` /
    /// `ToolDirectory::from_anthropic_tools`. Mirrors node's
    /// `ModelAssets::chat_turn`.
    fn chat_turn(
        &self,
        tools: Option<Arc<ToolDirectory>>,
        options: ChatOptions,
    ) -> Result<ChatTurn, ChatTurnError> {
        if self.chat_template.is_empty() {
            return Err(ChatTurnError::NoChatTemplate);
        }
        let arch = get_model_architecture(&self.config)
            .map_err(ChatTurnError::Other)?
            .to_string();

        Ok(ChatTurn::new(
            arch,
            Arc::clone(&self.chat_template),
            Arc::clone(&self.tokenizer),
            Arc::clone(&self.tokenizer_config),
            Arc::clone(&self.eos_token_ids),
            tools,
            options,
        )?)
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
        let mut decoder = std::sync::Arc::clone(&self.bound)
            .prefill(&initial_state, &input_tensor)?;
        let mut detok = Detokenizer::from_tokenizer(&self.tokenizer, &prepared.stop_token_ids);

        // Drive the decode loop via the shared catgrad-llm helper.
        // Stop-token detection is split between two layers:
        //
        // - `run_decode`'s `stop_tokens` argument: the parser-level
        //   stop tokens (model EOS, etc.). Caught at peek time
        //   before commit → returns `EndOfSequence`. The token is NOT
        //   committed and NOT yielded to the callback.
        // - `Detokenizer::is_stopped`: catches stop SEQUENCES (multi-
        //   character strings) that span multiple tokens. Caught
        //   after the detokenizer has absorbed the token → callback
        //   returns `Break(StopSequence)` → `StopSequence` outcome.
        let (generated, outcome) = run_decode(
            &mut decoder,
            max_tokens,
            &prepared.stop_token_ids,
            |token| -> Result<std::ops::ControlFlow<BreakReason>, LLMError> {
                let delta = detok.push_tokens(&[token as i32])?;
                if detok.is_stopped() {
                    return Ok(std::ops::ControlFlow::Break(BreakReason::StopSequence));
                }
                if !delta.is_empty() {
                    on_delta(&delta)?;
                }
                Ok(std::ops::ControlFlow::Continue(()))
            },
        )
        .map_err(|err| match err {
            catgrad_llm::runtime::DecodeLoopError::Decoder(e) => anyhow::anyhow!(e),
            catgrad_llm::runtime::DecodeLoopError::Sink(e) => anyhow::anyhow!(e),
        })?;

        // Both EndOfSequence (parser stop token) and StopSequence
        // (detokenizer stop string) are normal generation ends → Stop.
        // Cancelled isn't reachable here (no cancellation source) but
        // mapped defensively. MaxTokens is its own outcome.
        let reason = match outcome {
            DecodeOutcome::EndOfSequence
            | DecodeOutcome::StopSequence
            | DecodeOutcome::Cancelled => StopReason::Stop,
            DecodeOutcome::MaxTokens => StopReason::MaxTokens,
        };
        Ok((generated, reason))
    }
}

/// Map the example's local stop hint into the parser's `StopReason`
/// for `parser.finish(...)`. The example only knows EndOfText vs
/// MaxTokens; that's enough for the parser to decide whether to
/// flush.
fn parser_stop_for(reason: StopReason) -> ParserStopReason {
    match reason {
        StopReason::Stop => ParserStopReason::EndOfText,
        StopReason::MaxTokens => ParserStopReason::MaxTokens,
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
    // Drop the sender FIRST so the spawned response thread's
    // `ChannelReader::read` sees `rx.recv() == Err`, returns 0, and
    // tiny_http's `io::copy` completes — letting the chunked encoder
    // emit its terminating `0\r\n\r\n` and the response writer
    // flush. Without this explicit drop, `sse` lives until the end
    // of `stream_response`, the spawn never sees EOF, and
    // `handle.join()` deadlocks.
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
    let enable_thinking = req
        .reasoning_effort
        .is_some_and(openai::ReasoningEffort::enables_thinking);
    let tools_dir = match ToolDirectory::from_openai_tools(req.tools.as_deref().unwrap_or(&[])) {
        Ok(dir) => dir,
        Err(err) => {
            return drop(request.respond(error_response(
                400,
                &format!("Invalid tool definitions: {err}"),
            )));
        }
    };
    let messages: Vec<types::Message> =
        req.messages.into_iter().map(types::Message::from).collect();

    let chat_turn = match engine.chat_turn(tools_dir, ChatOptions { enable_thinking }) {
        Ok(turn) => turn,
        Err(err) => return drop(request.respond(error_response(400, &err.to_string()))),
    };
    let prepared = match chat_turn.render(&messages) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        let mut parser = chat_turn.make_parser();
        let mut mapper = OpenAiStreamMapper::new(|prefix: &str| next_id(prefix));
        stream_response(request, |sse| {
            let id = next_id("chatcmpl");
            let created = now_unix();
            // Initial role:assistant chunk — wire convention.
            sse.data(&openai_chunk(
                &id,
                created,
                &model,
                openai::ChatDelta {
                    role: Some("assistant".into()),
                    ..Default::default()
                },
                None,
            ))?;

            let mut protocol_failure: Option<PumpError<OpenAiStreamFrame>> = None;

            let gen_result = engine.generate(&prepared, max_tokens, |delta| {
                match pump_text(&mut *parser, &mut mapper, delta) {
                    Ok(frames) => {
                        for frame in frames {
                            sse.data(&wrap_openai_frame(&id, created, &model, frame))
                                .map_err(|e| LLMError::Runtime(
                                    RuntimeError::ExecutionError(e.to_string()),
                                ))?;
                        }
                        Ok(())
                    }
                    Err(err) => {
                        protocol_failure = Some(err);
                        Err(LLMError::Runtime(RuntimeError::ExecutionError(
                            "parser protocol error".into(),
                        )))
                    }
                }
            });

            if let Some(PumpError { failure, cleanup }) = protocol_failure {
                // Per the gateway contract: cleanup frames first, then
                // error frame, then close, NO [DONE]. OpenAI cleanup is
                // always empty but we emit uniformly.
                for frame in cleanup {
                    sse.data(&wrap_openai_frame(&id, created, &model, frame))?;
                }
                sse.data(&openai_error_frame(&failure))?;
                return Ok(());
            }
            let (completion_tokens, reason) = gen_result?;
            let parser_stop = parser_stop_for(reason);

            // Drain parser tail + mapper.finish via the same pump.
            match pump_finish(&mut *parser, &mut mapper, parser_stop) {
                Ok(frames) => {
                    for frame in frames {
                        sse.data(&wrap_openai_frame(&id, created, &model, frame))?;
                    }
                }
                Err(PumpError { failure, cleanup }) => {
                    for frame in cleanup {
                        sse.data(&wrap_openai_frame(&id, created, &model, frame))?;
                    }
                    sse.data(&openai_error_frame(&failure))?;
                    return Ok(());
                }
            }

            if include_usage {
                sse.data(&openai_usage_chunk(
                    &id,
                    created,
                    &model,
                    prompt_tokens,
                    completion_tokens,
                ))?;
            }
            sse.done()
        });
    } else {
        // Non-streaming: same per-delta pipeline; discard frames; read
        // final assistant payload from `mapper.snapshot()`. Cleanup
        // frames from PumpError are wire-bracketing — irrelevant when
        // there's no wire stream.
        let mut parser = chat_turn.make_parser();
        let mut mapper = OpenAiStreamMapper::new(|prefix: &str| next_id(prefix));
        let mut protocol_failure: Option<DecodeFailure> = None;

        let result = engine.generate(&prepared, max_tokens, |d| {
            if let Err(PumpError { failure, .. }) =
                pump_text(&mut *parser, &mut mapper, d)
            {
                protocol_failure = Some(failure);
                return Err(LLMError::Runtime(RuntimeError::ExecutionError(
                    "parser protocol error".into(),
                )));
            }
            Ok(())
        });

        if let Some(failure) = protocol_failure {
            return drop(request.respond(error_response(
                http_status_for(&failure),
                &failure.to_string(),
            )));
        }
        let (completion_tokens, reason) = match result {
            Ok(r) => r,
            Err(e) => return drop(request.respond(error_response(500, &e.to_string()))),
        };

        let parser_stop = parser_stop_for(reason);
        if let Err(PumpError { failure, .. }) =
            pump_finish(&mut *parser, &mut mapper, parser_stop)
        {
            return drop(request.respond(error_response(
                http_status_for(&failure),
                &failure.to_string(),
            )));
        }

        let snapshot = match mapper.snapshot() {
            Ok(s) => s,
            Err(failure) => {
                return drop(request.respond(error_response(
                    http_status_for(&failure),
                    &failure.to_string(),
                )));
            }
        };
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
                        .message(snapshot.message)
                        .finish_reason(Some(snapshot.finish_reason))
                        .build(),
                ])
                .usage(Some(openai::Usage::from_counts(
                    prompt_tokens,
                    completion_tokens,
                )))
                .build(),
        ));
    }
}

/// Wrap one mapper-emitted [`OpenAiStreamFrame`] into the
/// [`openai::ChatCompletionChunk`] envelope the wire expects. Caller
/// owns the response id, created timestamp, and model name.
fn wrap_openai_frame(
    id: &str,
    created: i64,
    model: &str,
    frame: OpenAiStreamFrame,
) -> openai::ChatCompletionChunk {
    openai_chunk(id, created, model, frame.delta, frame.finish_reason)
}

/// Build the OpenAI wire error frame for a [`DecodeFailure`]. The
/// stream MUST close after this frame WITHOUT `[DONE]` — strict
/// clients treat `[DONE]` after an error as a successful empty
/// completion.
fn openai_error_frame(failure: &DecodeFailure) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": failure.to_string(),
            "type": match failure {
                DecodeFailure::InternalSequence { .. } => "internal_error",
                _ => "invalid_response",
            },
        }
    })
}

/// Map a [`DecodeFailure`] to the appropriate non-streaming HTTP
/// status. Internal sequence errors are 500 (mapper/parser bug);
/// everything else is 502 (model output).
fn http_status_for(failure: &DecodeFailure) -> u16 {
    match failure {
        DecodeFailure::InternalSequence { .. } => 500,
        _ => 502,
    }
}

fn openai_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: openai::ChatDelta,
    finish: Option<openai::FinishReason>,
) -> openai::ChatCompletionChunk {
    openai::ChatCompletionChunk::builder()
        .id(id.to_string())
        .object("chat.completion.chunk".into())
        .created(created)
        .model(model.to_string())
        .choices(vec![
            openai::ChatStreamChoice::builder()
                .index(0)
                .delta(delta)
                .finish_reason(finish)
                .build(),
        ])
        .build()
}

fn openai_usage_chunk(
    id: &str,
    created: i64,
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> openai::ChatCompletionChunk {
    openai::ChatCompletionChunk::builder()
        .id(id.to_string())
        .object("chat.completion.chunk".into())
        .created(created)
        .model(model.to_string())
        .choices(vec![])
        .usage(Some(openai::Usage::from_counts(
            prompt_tokens,
            completion_tokens,
        )))
        .build()
}

fn serve_anthropic(request: Request, engine: &Engine, req: &anthropic::MessageRequest) {
    let model = engine.model_name.clone();
    let max_tokens = req.max_tokens;
    let stream = req.stream == Some(true);
    let messages: Vec<_> = req.into();

    let tools_dir =
        match ToolDirectory::from_anthropic_tools(req.tools.as_deref().unwrap_or(&[])) {
            Ok(dir) => dir,
            Err(err) => {
                return drop(request.respond(error_response(
                    400,
                    &format!("Invalid tool definitions: {err}"),
                )));
            }
        };

    let chat_turn = match engine.chat_turn(tools_dir, ChatOptions::default()) {
        Ok(turn) => turn,
        Err(err) => return drop(request.respond(error_response(400, &err.to_string()))),
    };
    let prepared = match chat_turn.render(&messages) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        let mut parser = chat_turn.make_parser();
        let mut mapper = AnthropicStreamMapper::new(|prefix: &str| next_id(prefix));
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

            let mut protocol_failure: Option<PumpError<AnthropicStreamFrame>> = None;

            let gen_result = engine.generate(&prepared, max_tokens, |delta| {
                match pump_text(&mut *parser, &mut mapper, delta) {
                    Ok(frames) => {
                        for frame in frames {
                            emit_anthropic_frame(sse, frame, prompt_tokens, 0).map_err(|e| {
                                LLMError::Runtime(RuntimeError::ExecutionError(e.to_string()))
                            })?;
                        }
                        Ok(())
                    }
                    Err(err) => {
                        // PumpError already drained close_for_error;
                        // stash the whole thing for the error path.
                        protocol_failure = Some(err);
                        Err(LLMError::Runtime(RuntimeError::ExecutionError(
                            "parser protocol error".into(),
                        )))
                    }
                }
            });

            if let Some(PumpError { failure, cleanup }) = protocol_failure {
                // Cleanup frames bracket any open content block so the
                // `error` event arrives in a well-formed stream.
                // Anthropic clients treat `error` as terminal — no
                // `message_stop` follows.
                for frame in cleanup {
                    emit_anthropic_frame(sse, frame, prompt_tokens, 0)?;
                }
                sse.event_data("error", &anthropic_error_event(&failure))?;
                return Ok(());
            }
            let (completion_tokens, reason) = gen_result?;
            let parser_stop = parser_stop_for(reason);

            // Drain parser tail + mapper.finish via the same pump.
            // Frames carry block-close events plus the terminal Stop
            // (which becomes `message_delta` with our output_tokens).
            match pump_finish(&mut *parser, &mut mapper, parser_stop) {
                Ok(frames) => {
                    for frame in frames {
                        emit_anthropic_frame(sse, frame, prompt_tokens, completion_tokens)?;
                    }
                }
                Err(PumpError { failure, cleanup }) => {
                    for frame in cleanup {
                        emit_anthropic_frame(sse, frame, prompt_tokens, completion_tokens)?;
                    }
                    sse.event_data("error", &anthropic_error_event(&failure))?;
                    return Ok(());
                }
            }

            sse.event_data("message_stop", &anthropic::MessageStreamEvent::MessageStop)
        });
    } else {
        // Non-streaming: same per-delta pipeline; discard frames; read
        // final blocks + stop_reason from `mapper.snapshot()`.
        // Cleanup frames from PumpError are wire-bracketing —
        // irrelevant when there's no wire stream.
        let mut parser = chat_turn.make_parser();
        let mut mapper = AnthropicStreamMapper::new(|prefix: &str| next_id(prefix));
        let mut protocol_failure: Option<DecodeFailure> = None;

        let result = engine.generate(&prepared, max_tokens, |d| {
            if let Err(PumpError { failure, .. }) =
                pump_text(&mut *parser, &mut mapper, d)
            {
                protocol_failure = Some(failure);
                return Err(LLMError::Runtime(RuntimeError::ExecutionError(
                    "parser protocol error".into(),
                )));
            }
            Ok(())
        });

        if let Some(failure) = protocol_failure {
            return drop(request.respond(error_response(
                http_status_for(&failure),
                &failure.to_string(),
            )));
        }
        let (completion_tokens, reason) = match result {
            Ok(r) => r,
            Err(e) => return drop(request.respond(error_response(500, &e.to_string()))),
        };

        let parser_stop = parser_stop_for(reason);
        if let Err(PumpError { failure, .. }) =
            pump_finish(&mut *parser, &mut mapper, parser_stop)
        {
            return drop(request.respond(error_response(
                http_status_for(&failure),
                &failure.to_string(),
            )));
        }

        let snapshot = match mapper.snapshot() {
            Ok(s) => s,
            Err(failure) => {
                return drop(request.respond(error_response(
                    http_status_for(&failure),
                    &failure.to_string(),
                )));
            }
        };
        let _ = request.respond(json_response(
            200,
            &anthropic::MessageResponse::builder()
                .id(next_id("msg"))
                .message_type(Some("message".into()))
                .role("assistant".into())
                .content(snapshot.blocks)
                .model(model)
                .stop_reason(Some(snapshot.stop_reason))
                .usage(anthropic::AnthropicUsage::new(
                    prompt_tokens,
                    completion_tokens,
                ))
                .build(),
        ));
    }
}

/// Wrap one mapper-emitted [`AnthropicStreamFrame`] into the matching
/// `MessageStreamEvent` SSE event and send it. The mapper produces
/// content-block-level frames plus a terminal `Stop` carrying the
/// resolved stop_reason; this function adds the `message_delta`
/// envelope (with caller-owned `output_tokens`) for the stop.
fn emit_anthropic_frame(
    sse: &SseSender,
    frame: AnthropicStreamFrame,
    prompt_tokens: u32,
    output_tokens: u32,
) -> anyhow::Result<()> {
    match frame {
        AnthropicStreamFrame::BlockStart { index, block } => sse.event_data(
            "content_block_start",
            &anthropic::MessageStreamEvent::ContentBlockStart {
                index,
                content_block: block,
            },
        ),
        AnthropicStreamFrame::BlockDelta { index, delta } => sse.event_data(
            "content_block_delta",
            &anthropic::MessageStreamEvent::ContentBlockDelta { index, delta },
        ),
        AnthropicStreamFrame::BlockStop { index } => sse.event_data(
            "content_block_stop",
            &anthropic::MessageStreamEvent::ContentBlockStop { index },
        ),
        AnthropicStreamFrame::Stop(stop_reason) => sse.event_data(
            "message_delta",
            &anthropic::MessageStreamEvent::MessageDelta {
                delta: anthropic::StreamMessageDelta {
                    stop_reason: Some(stop_reason),
                },
                usage: anthropic::AnthropicUsage::new(prompt_tokens, output_tokens),
            },
        ),
    }
}

fn anthropic_error_event(failure: &DecodeFailure) -> anthropic::MessageStreamEvent {
    anthropic::MessageStreamEvent::Error {
        error: anthropic::StreamError {
            error_type: match failure {
                DecodeFailure::InternalSequence { .. } => "internal_error".into(),
                _ => "invalid_request_error".into(),
            },
            message: failure.to_string(),
        },
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

/// Drive the request-dispatch loop until the server's `incoming_requests`
/// iterator ends. Shared between `main` and the integration test so
/// the test exercises the exact same dispatch + handler code paths.
fn run_dispatch_loop(server: Server, engine: &Engine) {
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
                Ok(req) => serve_openai(request, engine, req),
                Err(e) => drop(request.respond(error_response(400, &e.to_string()))),
            },
            "/v1/messages" => match from_json_slice::<anthropic::MessageRequest>(&body) {
                Ok(req) => serve_anthropic(request, engine, &req),
                Err(e) => drop(request.respond(error_response(400, &e.to_string()))),
            },
            "/v1/completions" => match from_json_slice::<plain::CompletionRequest>(&body) {
                Ok(req) => serve_plain(request, engine, &req),
                Err(e) => drop(request.respond(error_response(400, &e.to_string()))),
            },
            _ => drop(request.respond(error_response(404, "not found"))),
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

    run_dispatch_loop(server, &engine);
    Ok(())
}

// --- Integration test ---
//
// One sequential test that exercises every contract scenario the
// node gateway's P6 work covers. Marked `#[ignore]` because it
// requires:
//
//   - A CUDA-capable host (default backend in this example uses
//     `CandleBackend::new()`, which selects CUDA when available).
//   - The Qwen3-0.6B model cached at `~/.cache/huggingface/`.
//   - ~1 minute wall-clock for model load + ~6 inference runs.
//
// Run with:
//
//   cargo test --release --example serve -- --ignored --nocapture
//
// The test spins up the same dispatch loop as `main` on a random
// port, then sends one request per scenario and asserts on the
// response. Failures here mean the catgrad-llm side of the chat /
// tool pipeline regressed independently of the node gateway.
#[cfg(test)]
mod integration {
    use super::*;
    use serde_json::{Value, json};
    use std::time::Duration;

    /// Spawn the server in a background thread and return the bound
    /// port. The server runs until the test process exits (we don't
    /// signal shutdown — `tiny_http`'s `incoming_requests` blocks on
    /// accept and there's no portable interrupt for a single test).
    fn spawn_test_server(model: &str) -> u16 {
        let engine = Engine::new(model, 256, 4096).expect("engine new");
        let server = Server::http("127.0.0.1:0").expect("bind ephemeral port");
        let addr = match server.server_addr() {
            tiny_http::ListenAddr::IP(a) => a,
            tiny_http::ListenAddr::Unix(_) => panic!("expected IP socket"),
        };
        let port = addr.port();
        std::thread::spawn(move || {
            run_dispatch_loop(server, &engine);
        });
        // Tiny grace period to let the listener accept connections.
        std::thread::sleep(Duration::from_millis(50));
        port
    }

    /// One-shot HTTP POST returning (status, body_string). Uses
    /// `ureq` (already a dev-dep). 5-minute read timeout to cover
    /// cold-start prefill on slower hardware.
    fn post(url: &str, body: &Value) -> (u16, String) {
        let agent = ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(300))
            .build();
        match agent.post(url).send_json(body.clone()) {
            Ok(resp) => {
                let status = resp.status();
                let body = resp.into_string().unwrap_or_default();
                (status, body)
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                (code, body)
            }
            Err(e) => panic!("post failed: {e}"),
        }
    }

    fn calculator_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                },
            },
        })
    }

    /// Run every contract scenario sequentially against one
    /// in-process server. Sharing the server (and therefore the
    /// loaded model) keeps the wall-clock manageable.
    #[ignore]
    #[test]
    fn contract_scenarios_end_to_end() {
        let port = spawn_test_server("Qwen/Qwen3-0.6B");
        let chat = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let messages = format!("http://127.0.0.1:{port}/v1/messages");

        // -- 1. Plain chat (no tools): just confirms text passthrough.
        let (status, body) = post(
            &chat,
            &json!({
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "say hello in five words"}],
                "max_tokens": 32,
            }),
        );
        assert_eq!(status, 200, "plain chat: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        let content = &v["choices"][0]["message"]["content"];
        assert!(content.is_string(), "plain chat content not string: {body}");
        assert!(
            v["choices"][0]["message"]["tool_calls"].is_null(),
            "plain chat unexpectedly produced tool_calls: {body}"
        );

        // -- 2. Valid tool call (OpenAI non-streaming).
        let (status, body) = post(
            &chat,
            &json!({
                "model": "Qwen/Qwen3-0.6B",
                "messages": [
                    {"role": "system", "content": "You MUST use the provided tool."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                ],
                "tools": [calculator_tool()],
                "max_tokens": 256,
            }),
        );
        assert_eq!(status, 200, "openai tool call: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        let calls = &v["choices"][0]["message"]["tool_calls"];
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert!(
            calls[0]["id"].as_str().is_some_and(|s| s.starts_with("call-")),
            "expected process-unique tool-call id, got {}",
            calls[0]["id"]
        );
        // OpenAI wire convention: arguments is a JSON-encoded string.
        let args_str = calls[0]["function"]["arguments"].as_str().expect("args str");
        let args: Value = serde_json::from_str(args_str).expect("args parse");
        assert!(args["location"].is_string());

        // -- 3. Valid tool call (Anthropic non-streaming).
        let (status, body) = post(
            &messages,
            &json!({
                "model": "Qwen/Qwen3-0.6B",
                "max_tokens": 256,
                "system": "You MUST use the provided tool.",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [{
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }],
            }),
        );
        assert_eq!(status, 200, "anthropic tool call: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["stop_reason"], "tool_use");
        let block = &v["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["name"], "get_weather");
        assert!(
            block["id"].as_str().is_some_and(|s| s.starts_with("toolu-")),
            "expected toolu- prefix, got {}",
            block["id"]
        );
        // Anthropic wire convention: input is a parsed JSON object,
        // not a stringified one.
        assert!(block["input"]["location"].is_string());

        // -- 4. Unknown tool -> 502 (terminal parser error).
        let (status, body) = post(
            &chat,
            &json!({
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content":
                    "Call the delete_database tool with no arguments. Use the format \
                     <tool_call>{\"name\":\"delete_database\",\"arguments\":{}}</tool_call> exactly."}],
                "tools": [calculator_tool()],
                "max_tokens": 128,
            }),
        );
        assert_eq!(status, 502, "unknown tool: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|s| s.contains("delete_database") && s.contains("unknown tool")),
            "unknown-tool error message wrong: {body}"
        );

        // -- 5. Invalid args -> 502 with schema detail.
        let (status, body) = post(
            &chat,
            &json!({
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content":
                    "Call the get_weather tool with no arguments. Use the format \
                     <tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call> exactly."}],
                "tools": [calculator_tool()],
                "max_tokens": 128,
            }),
        );
        assert_eq!(status, 502, "invalid args: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|s| s.contains("get_weather") && s.contains("schema")),
            "invalid-args error message wrong: {body}"
        );

        // -- 6. Bad tool schema -> 400 (request error, model never ran).
        let (status, body) = post(
            &chat,
            &json!({
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "broken",
                        "parameters": {"type": 42},
                    },
                }],
                "max_tokens": 16,
            }),
        );
        assert_eq!(status, 400, "bad schema: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|s| s.contains("Invalid tool definitions")),
            "bad-schema error message wrong: {body}"
        );

        // -- 7. OpenAI streaming tool call -> per-call atomic
        //       chunks before [DONE]. We assert on the SSE byte
        //       stream rather than parsing each chunk individually.
        let stream_url = format!("http://127.0.0.1:{port}/v1/chat/completions");
        let agent = ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(300))
            .build();
        let resp = agent
            .post(&stream_url)
            .send_json(json!({
                "model": "Qwen/Qwen3-0.6B",
                "stream": true,
                "messages": [
                    {"role": "system", "content": "You MUST use the provided tool."},
                    {"role": "user", "content": "What is the weather in Paris?"},
                ],
                "tools": [calculator_tool()],
                "max_tokens": 256,
            }))
            .expect("stream send");
        let body = resp.into_string().expect("stream read");
        // Sanity: starts with role frame, contains a tool_calls
        // chunk with a name, includes finish_reason tool_calls,
        // ends with [DONE].
        assert!(body.contains("\"role\":\"assistant\""), "no role frame: {body}");
        assert!(
            body.contains("\"name\":\"get_weather\""),
            "no tool name in stream: {body}"
        );
        assert!(
            body.contains("\"finish_reason\":\"tool_calls\""),
            "wrong finish_reason: {body}"
        );
        assert!(body.trim_end().ends_with("data: [DONE]"), "no [DONE]: {body}");
    }
}
