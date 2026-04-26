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
use catgrad_llm::runtime::chat::{
    ChatOptions, ChatTurn, DecodeEvent, StopReason as ParserStopReason, ToolDirectory, ToolSpec,
};
use catgrad_llm::runtime::{BoundProgramText, text_program_from_config};
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
/// surface handlers: `InvalidToolDirectory` and `ToolsUnsupportedForModel`
/// are request errors (400); the others surface as 500 (no chat
/// template) or get rejected at parse-time before reaching here.
#[derive(Debug)]
enum ChatTurnError {
    NoChatTemplate,
    InvalidToolDirectory(LLMError),
    ToolsUnsupportedForModel { arch: String },
    Other(LLMError),
}

impl std::fmt::Display for ChatTurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoChatTemplate => write!(f, "model has no chat template"),
            Self::InvalidToolDirectory(e) => write!(f, "Invalid tool definitions: {e}"),
            Self::ToolsUnsupportedForModel { arch } => {
                write!(f, "Model architecture `{arch}` does not support tool calling")
            }
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

    /// Build a `ChatTurn` for one chat-completion request. Wire-tools
    /// → typed `ToolSpec` conversion + `ToolDirectory` build + protocol
    /// lookup all happen here, mirroring node's `ModelAssets::chat_turn`.
    /// The wire-tools input is always the OpenAI-shaped envelope
    /// (the Anthropic surface converts to it before calling).
    fn chat_turn(
        &self,
        wire_tools: Option<&[serde_json::Value]>,
        options: ChatOptions,
    ) -> Result<ChatTurn, ChatTurnError> {
        if self.chat_template.is_empty() {
            return Err(ChatTurnError::NoChatTemplate);
        }
        let arch = get_model_architecture(&self.config)
            .map_err(ChatTurnError::Other)?
            .to_string();

        // Empty wire-tools list normalized to None at the edge — same
        // semantics as ChatTurn::new's internal normalization, kept
        // explicit here so the wire meaning ("user sent []") is
        // visible in one place.
        let directory = match wire_tools {
            None => None,
            Some(specs) if specs.is_empty() => None,
            Some(specs) => {
                let typed = wire_tools_to_specs(specs).map_err(ChatTurnError::InvalidToolDirectory)?;
                let dir =
                    ToolDirectory::new(typed).map_err(ChatTurnError::InvalidToolDirectory)?;
                Some(Arc::new(dir))
            }
        };

        ChatTurn::new(
            arch.clone(),
            Arc::clone(&self.chat_template),
            Arc::clone(&self.tokenizer),
            Arc::clone(&self.tokenizer_config),
            Arc::clone(&self.eos_token_ids),
            directory,
            options,
        )
        .map_err(|err| match err {
            LLMError::UnsupportedModel(_) => ChatTurnError::ToolsUnsupportedForModel { arch },
            other => ChatTurnError::Other(other),
        })
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

/// Convert an OpenAI-shaped wire tool list (`[{type, function:{name,
/// description, parameters}}, ...]`) into typed [`ToolSpec`]s. The
/// Anthropic surface goes through `anthropic_tool_to_openai_form`
/// before reaching here. Strict shape — a missing `name` is a request
/// error, since the schema can't be applied without one.
fn wire_tools_to_specs(wire_tools: &[serde_json::Value]) -> Result<Vec<ToolSpec>, LLMError> {
    let mut out = Vec::with_capacity(wire_tools.len());
    for (idx, entry) in wire_tools.iter().enumerate() {
        let function = entry.get("function").ok_or_else(|| {
            LLMError::InvalidModelConfig(format!("tool[{idx}] is missing the `function` wrapper"))
        })?;
        let name = function
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                LLMError::InvalidModelConfig(format!(
                    "tool[{idx}].function is missing required `name`"
                ))
            })?
            .to_string();
        let description = function
            .get("description")
            .and_then(serde_json::Value::as_str)
            .map(|s| s.to_string());
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
        out.push(ToolSpec::new(name, description, parameters));
    }
    Ok(out)
}

/// Translate an Anthropic-shaped tool (`{name, description,
/// input_schema}`) into the OpenAI-shaped envelope, so the wire
/// converter has one input shape.
fn anthropic_tool_to_openai_form(t: &serde_json::Value) -> serde_json::Value {
    let mut function = serde_json::Map::new();
    if let Some(name) = t.get("name") {
        function.insert("name".to_string(), name.clone());
    }
    if let Some(desc) = t.get("description") {
        function.insert("description".to_string(), desc.clone());
    }
    if let Some(schema) = t.get("input_schema") {
        function.insert("parameters".to_string(), schema.clone());
    }
    let mut wrapper = serde_json::Map::new();
    wrapper.insert("type".to_string(), serde_json::Value::String("function".into()));
    wrapper.insert("function".to_string(), serde_json::Value::Object(function));
    serde_json::Value::Object(wrapper)
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

/// OpenAI `finish_reason` mapping that honors `saw_tool_call` over
/// the inference stop reason — clients use this signal to dispatch.
fn openai_finish_reason(stop: StopReason, saw_tool_call: bool) -> openai::FinishReason {
    if saw_tool_call {
        return openai::FinishReason::ToolCalls;
    }
    stop.into()
}

/// Anthropic `stop_reason` mapping with the same `saw_tool_call`
/// override.
fn anthropic_stop_reason(stop: StopReason, saw_tool_call: bool) -> anthropic::StopReason {
    if saw_tool_call {
        return anthropic::StopReason::ToolUse;
    }
    stop.into()
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
    let tools = req.tools.clone();
    let messages: Vec<types::Message> =
        req.messages.into_iter().map(types::Message::from).collect();

    let chat_turn = match engine.chat_turn(tools.as_deref(), ChatOptions { enable_thinking }) {
        Ok(turn) => turn,
        Err(err @ ChatTurnError::InvalidToolDirectory(_))
        | Err(err @ ChatTurnError::ToolsUnsupportedForModel { .. }) => {
            return drop(request.respond(error_response(400, &err.to_string())));
        }
        Err(err) => return drop(request.respond(error_response(400, &err.to_string()))),
    };
    let prepared = match chat_turn.render(&messages) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        let mut parser = chat_turn.make_parser();
        stream_response(request, |sse| {
            let id = next_id("chatcmpl");
            let created = now_unix();
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

            let mut saw_tool_call = false;
            let mut in_progress: std::collections::HashMap<usize, OpenAiCall> =
                std::collections::HashMap::new();
            let mut protocol_error: Option<String> = None;

            let gen_result = engine.generate(&prepared, max_tokens, |delta| {
                let events = parser.feed(delta);
                for event in events {
                    let outcome = openai_apply_stream_event(
                        event,
                        &id,
                        created,
                        &model,
                        sse,
                        &mut saw_tool_call,
                        &mut in_progress,
                    )
                    .map_err(|e| {
                        LLMError::Runtime(RuntimeError::ExecutionError(e.to_string()))
                    })?;
                    if let Some(msg) = outcome {
                        protocol_error = Some(msg);
                        return Err(LLMError::Runtime(RuntimeError::ExecutionError(
                            "parser protocol error".into(),
                        )));
                    }
                }
                Ok(())
            });

            if let Some(message) = protocol_error {
                // Per the gateway contract: error frame, then close,
                // NO [DONE]. Strict OpenAI clients treat [DONE] after
                // an error as a successful empty completion.
                sse.data(&serde_json::json!({
                    "error": {
                        "message": message,
                        "type": "invalid_response",
                    }
                }))?;
                return Ok(());
            }
            // Propagate any non-protocol generation error.
            let (completion_tokens, reason) = gen_result?;

            // Drain parser tail events.
            let mut tail_protocol_error: Option<String> = None;
            for event in parser.finish(parser_stop_for(reason)) {
                if let Some(msg) = openai_apply_stream_event(
                    event,
                    &id,
                    created,
                    &model,
                    sse,
                    &mut saw_tool_call,
                    &mut in_progress,
                )? {
                    tail_protocol_error = Some(msg);
                    break;
                }
            }
            if let Some(message) = tail_protocol_error {
                sse.data(&serde_json::json!({
                    "error": {
                        "message": message,
                        "type": "invalid_response",
                    }
                }))?;
                return Ok(());
            }

            sse.data(&openai_chunk(
                &id,
                created,
                &model,
                openai::ChatDelta::default(),
                Some(openai_finish_reason(reason, saw_tool_call)),
            ))?;
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
        // Non-streaming: collect text, then parse all at once.
        let mut text = String::new();
        let result = engine.generate(&prepared, max_tokens, |d| {
            text.push_str(d);
            Ok(())
        });
        let (completion_tokens, reason) = match result {
            Ok(r) => r,
            Err(e) => return drop(request.respond(error_response(500, &e.to_string()))),
        };

        let mut parser = chat_turn.make_parser();
        let mut events = parser.feed(&text);
        events.extend(parser.finish(parser_stop_for(reason)));

        let mut content = String::new();
        let mut tool_calls: Vec<serde_json::Value> = Vec::new();
        let mut saw_tool_call = false;
        let mut in_progress: std::collections::HashMap<usize, OpenAiCall> =
            std::collections::HashMap::new();
        for event in events {
            match openai_apply_event(
                event,
                &mut content,
                &mut tool_calls,
                &mut saw_tool_call,
                &mut in_progress,
            ) {
                Ok(()) => {}
                Err(message) => {
                    return drop(request.respond(error_response(502, &message)));
                }
            }
        }

        let message_content = if content.is_empty() {
            None
        } else {
            Some(openai::MessageContent::Text(content))
        };
        let assistant = openai::ChatMessage::builder()
            .role("assistant".into())
            .content(message_content)
            .tool_calls(if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            })
            .build();
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
                        .message(assistant)
                        .finish_reason(Some(openai_finish_reason(reason, saw_tool_call)))
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

/// One in-flight tool call, keyed by parser index. Wire ID minted at
/// `ToolCallStart` and reused for matching `ArgsDelta` / `End`.
struct OpenAiCall {
    wire_id: String,
    name: String,
    arguments: String,
}

/// Apply one parser event to the non-streaming OpenAI response
/// accumulators. Returns `Err(message)` for terminal parser events
/// (caller maps to HTTP 502).
fn openai_apply_event(
    event: DecodeEvent,
    content: &mut String,
    tool_calls: &mut Vec<serde_json::Value>,
    saw_tool_call: &mut bool,
    in_progress: &mut std::collections::HashMap<usize, OpenAiCall>,
) -> Result<(), String> {
    match event {
        DecodeEvent::TextDelta(s) => content.push_str(&s),
        DecodeEvent::ToolCallStart { index, name } => {
            *saw_tool_call = true;
            in_progress.insert(
                index,
                OpenAiCall {
                    wire_id: next_id("call"),
                    name,
                    arguments: String::new(),
                },
            );
        }
        DecodeEvent::ToolCallArgsDelta { index, delta } => {
            if let Some(c) = in_progress.get_mut(&index) {
                c.arguments.push_str(&delta);
            }
        }
        DecodeEvent::ToolCallEnd { index, .. } => {
            if let Some(c) = in_progress.remove(&index) {
                tool_calls.push(serde_json::json!({
                    "id": c.wire_id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.arguments,
                    },
                }));
            }
        }
        DecodeEvent::Stop { .. } => {}
        DecodeEvent::UnknownTool { name, .. } => {
            return Err(format!("model called unknown tool `{name}`"));
        }
        DecodeEvent::InvalidArgs { name, errors, .. } => {
            let detail = errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(format!(
                "model called `{name}` with arguments that don't match the schema: {detail}"
            ));
        }
        DecodeEvent::ParseError { sentinel, source } => {
            return Err(format!(
                "model emitted malformed tool call within `{sentinel}`: {source}"
            ));
        }
    }
    Ok(())
}

/// Apply one parser event to the streaming OpenAI surface. Returns
/// `Ok(Some(error_msg))` for terminal parser events (caller emits an
/// error frame and closes WITHOUT `[DONE]`); `Ok(None)` for normal
/// progress; `Err(_)` only for SSE-channel send failures.
fn openai_apply_stream_event(
    event: DecodeEvent,
    id: &str,
    created: i64,
    model: &str,
    sse: &SseSender,
    saw_tool_call: &mut bool,
    in_progress: &mut std::collections::HashMap<usize, OpenAiCall>,
) -> anyhow::Result<Option<String>> {
    match event {
        DecodeEvent::TextDelta(s) => {
            sse.data(&openai_chunk(
                id,
                created,
                model,
                openai::ChatDelta {
                    content: Some(s),
                    ..Default::default()
                },
                None,
            ))?;
        }
        DecodeEvent::ToolCallStart { index, name } => {
            *saw_tool_call = true;
            let wire_id = next_id("call");
            in_progress.insert(
                index,
                OpenAiCall {
                    wire_id: wire_id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                },
            );
            sse.data(&openai_chunk(
                id,
                created,
                model,
                openai::ChatDelta {
                    tool_calls: Some(vec![serde_json::json!({
                        "index": index,
                        "id": wire_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": "",
                        },
                    })]),
                    ..Default::default()
                },
                None,
            ))?;
        }
        DecodeEvent::ToolCallArgsDelta { index, delta } => {
            if let Some(c) = in_progress.get_mut(&index) {
                c.arguments.push_str(&delta);
            }
            sse.data(&openai_chunk(
                id,
                created,
                model,
                openai::ChatDelta {
                    tool_calls: Some(vec![serde_json::json!({
                        "index": index,
                        "function": { "arguments": delta },
                    })]),
                    ..Default::default()
                },
                None,
            ))?;
        }
        DecodeEvent::ToolCallEnd { index, .. } => {
            in_progress.remove(&index);
        }
        DecodeEvent::Stop { .. } => {}
        DecodeEvent::UnknownTool { name, .. } => {
            return Ok(Some(format!("model called unknown tool `{name}`")));
        }
        DecodeEvent::InvalidArgs { name, errors, .. } => {
            let detail = errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Ok(Some(format!(
                "model called `{name}` with arguments that don't match the schema: {detail}"
            )));
        }
        DecodeEvent::ParseError { sentinel, source } => {
            return Ok(Some(format!(
                "model emitted malformed tool call within `{sentinel}`: {source}"
            )));
        }
    }
    Ok(None)
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

    // Anthropic wire tools → OpenAI shape → typed ToolSpec via the
    // shared converter.
    let tools: Option<Vec<serde_json::Value>> = req.tools.as_ref().map(|tools| {
        tools.iter().map(anthropic_tool_to_openai_form).collect()
    });

    let chat_turn = match engine.chat_turn(tools.as_deref(), ChatOptions::default()) {
        Ok(turn) => turn,
        Err(err @ ChatTurnError::InvalidToolDirectory(_))
        | Err(err @ ChatTurnError::ToolsUnsupportedForModel { .. }) => {
            return drop(request.respond(error_response(400, &err.to_string())));
        }
        Err(err) => return drop(request.respond(error_response(400, &err.to_string()))),
    };
    let prepared = match chat_turn.render(&messages) {
        Ok(p) => p,
        Err(e) => return drop(request.respond(error_response(400, &e.to_string()))),
    };
    let prompt_tokens = prepared.input_ids.len() as u32;

    if stream {
        let mut parser = chat_turn.make_parser();
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

            let mut blocks = AnthropicBlocks::new();
            let mut saw_tool_call = false;
            let mut protocol_error: Option<String> = None;

            let gen_result = engine.generate(&prepared, max_tokens, |delta| {
                let events = parser.feed(delta);
                for event in events {
                    let outcome = anthropic_apply_stream_event(
                        event,
                        sse,
                        &mut blocks,
                        &mut saw_tool_call,
                    )
                    .map_err(|e| {
                        LLMError::Runtime(RuntimeError::ExecutionError(e.to_string()))
                    })?;
                    if let Some(msg) = outcome {
                        protocol_error = Some(msg);
                        return Err(LLMError::Runtime(RuntimeError::ExecutionError(
                            "parser protocol error".into(),
                        )));
                    }
                }
                Ok(())
            });

            if let Some(message) = protocol_error {
                // Anthropic clients treat `error` as terminal — no
                // `message_stop` follows. Close any open block first.
                blocks.close_open(sse)?;
                sse.event_data(
                    "error",
                    &anthropic::MessageStreamEvent::Error {
                        error: anthropic::StreamError {
                            error_type: "invalid_request_error".into(),
                            message,
                        },
                    },
                )?;
                return Ok(());
            }
            let (completion_tokens, reason) = gen_result?;

            // Drain parser tail.
            let mut tail_protocol_error: Option<String> = None;
            for event in parser.finish(parser_stop_for(reason)) {
                if let Some(msg) = anthropic_apply_stream_event(
                    event,
                    sse,
                    &mut blocks,
                    &mut saw_tool_call,
                )? {
                    tail_protocol_error = Some(msg);
                    break;
                }
            }
            if let Some(message) = tail_protocol_error {
                blocks.close_open(sse)?;
                sse.event_data(
                    "error",
                    &anthropic::MessageStreamEvent::Error {
                        error: anthropic::StreamError {
                            error_type: "invalid_request_error".into(),
                            message,
                        },
                    },
                )?;
                return Ok(());
            }

            blocks.close_open(sse)?;
            sse.event_data(
                "message_delta",
                &anthropic::MessageStreamEvent::MessageDelta {
                    delta: anthropic::StreamMessageDelta {
                        stop_reason: Some(anthropic_stop_reason(reason, saw_tool_call)),
                    },
                    usage: anthropic::AnthropicUsage::new(prompt_tokens, completion_tokens),
                },
            )?;
            sse.event_data("message_stop", &anthropic::MessageStreamEvent::MessageStop)
        });
    } else {
        // Non-streaming: collect text, parse, walk events into blocks.
        let mut text = String::new();
        let result = engine.generate(&prepared, max_tokens, |d| {
            text.push_str(d);
            Ok(())
        });
        let (completion_tokens, reason) = match result {
            Ok(r) => r,
            Err(e) => return drop(request.respond(error_response(500, &e.to_string()))),
        };

        let mut parser = chat_turn.make_parser();
        let mut events = parser.feed(&text);
        events.extend(parser.finish(parser_stop_for(reason)));

        let mut content_blocks: Vec<anthropic::ContentBlock> = Vec::new();
        let mut current_text = String::new();
        let mut saw_tool_call = false;
        let mut in_progress: std::collections::HashMap<usize, (String, String)> =
            std::collections::HashMap::new();
        for event in events {
            match event {
                DecodeEvent::TextDelta(s) => current_text.push_str(&s),
                DecodeEvent::ToolCallStart { index, name } => {
                    saw_tool_call = true;
                    if !current_text.is_empty() {
                        content_blocks.push(anthropic::ContentBlock::Text {
                            text: std::mem::take(&mut current_text),
                        });
                    }
                    in_progress.insert(index, (next_id("toolu"), name));
                }
                DecodeEvent::ToolCallArgsDelta { .. } => {}
                DecodeEvent::ToolCallEnd { index, args } => {
                    if let Some((wire_id, name)) = in_progress.remove(&index) {
                        content_blocks.push(anthropic::ContentBlock::ToolUse {
                            id: wire_id,
                            name,
                            input: args,
                        });
                    }
                }
                DecodeEvent::Stop { .. } => {}
                DecodeEvent::UnknownTool { name, .. } => {
                    return drop(request.respond(error_response(
                        502,
                        &format!("model called unknown tool `{name}`"),
                    )));
                }
                DecodeEvent::InvalidArgs { name, errors, .. } => {
                    let detail = errors
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join("; ");
                    return drop(request.respond(error_response(
                        502,
                        &format!(
                            "model called `{name}` with arguments that don't match the schema: {detail}"
                        ),
                    )));
                }
                DecodeEvent::ParseError { sentinel, source } => {
                    return drop(request.respond(error_response(
                        502,
                        &format!(
                            "model emitted malformed tool call within `{sentinel}`: {source}"
                        ),
                    )));
                }
            }
        }
        if !current_text.is_empty() {
            content_blocks.push(anthropic::ContentBlock::Text { text: current_text });
        }
        if content_blocks.is_empty() {
            content_blocks.push(anthropic::ContentBlock::Text { text: String::new() });
        }

        let _ = request.respond(json_response(
            200,
            &anthropic::MessageResponse::builder()
                .id(next_id("msg"))
                .message_type(Some("message".into()))
                .role("assistant".into())
                .content(content_blocks)
                .model(model)
                .stop_reason(Some(anthropic_stop_reason(reason, saw_tool_call)))
                .usage(anthropic::AnthropicUsage::new(
                    prompt_tokens,
                    completion_tokens,
                ))
                .build(),
        ));
    }
}

/// Anthropic streaming block tracker. Anthropic requires
/// `content_block_start` / `delta`* / `stop` to be perfectly
/// bracketed and forbids interleaving deltas from different blocks,
/// so transitions between text and tool_use blocks must close-then-
/// open. The tracker maintains its own block-index counter — the
/// parser's tool-call `index` is NOT the same as the Anthropic
/// content-block index.
struct AnthropicBlocks {
    next_index: u32,
    open: AnthropicOpen,
    in_progress: std::collections::HashMap<usize, u32>,
}

enum AnthropicOpen {
    None,
    Text { index: u32 },
    ToolUse { block_index: u32 },
}

impl AnthropicBlocks {
    fn new() -> Self {
        Self {
            next_index: 0,
            open: AnthropicOpen::None,
            in_progress: std::collections::HashMap::new(),
        }
    }

    fn alloc(&mut self) -> u32 {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    /// Close any currently-open block. Used before terminal frames
    /// or before opening a different block kind.
    fn close_open(&mut self, sse: &SseSender) -> anyhow::Result<()> {
        let to_close = match self.open {
            AnthropicOpen::None => None,
            AnthropicOpen::Text { index } => Some(index),
            AnthropicOpen::ToolUse { block_index } => Some(block_index),
        };
        if let Some(index) = to_close {
            sse.event_data(
                "content_block_stop",
                &anthropic::MessageStreamEvent::ContentBlockStop { index },
            )?;
            self.open = AnthropicOpen::None;
        }
        Ok(())
    }
}

fn anthropic_apply_stream_event(
    event: DecodeEvent,
    sse: &SseSender,
    blocks: &mut AnthropicBlocks,
    saw_tool_call: &mut bool,
) -> anyhow::Result<Option<String>> {
    match event {
        DecodeEvent::TextDelta(s) => {
            let block_index = match blocks.open {
                AnthropicOpen::Text { index } => index,
                AnthropicOpen::ToolUse { .. } | AnthropicOpen::None => {
                    blocks.close_open(sse)?;
                    let idx = blocks.alloc();
                    sse.event_data(
                        "content_block_start",
                        &anthropic::MessageStreamEvent::ContentBlockStart {
                            index: idx,
                            content_block: anthropic::ContentBlock::Text {
                                text: String::new(),
                            },
                        },
                    )?;
                    blocks.open = AnthropicOpen::Text { index: idx };
                    idx
                }
            };
            sse.event_data(
                "content_block_delta",
                &anthropic::MessageStreamEvent::ContentBlockDelta {
                    index: block_index,
                    delta: anthropic::ContentBlockDelta::TextDelta { text: s },
                },
            )?;
        }
        DecodeEvent::ToolCallStart { index, name } => {
            *saw_tool_call = true;
            blocks.close_open(sse)?;
            let block_index = blocks.alloc();
            blocks.in_progress.insert(index, block_index);
            blocks.open = AnthropicOpen::ToolUse { block_index };
            sse.event_data(
                "content_block_start",
                &anthropic::MessageStreamEvent::ContentBlockStart {
                    index: block_index,
                    content_block: anthropic::ContentBlock::ToolUse {
                        id: next_id("toolu"),
                        name,
                        input: serde_json::Value::Object(serde_json::Map::new()),
                    },
                },
            )?;
        }
        DecodeEvent::ToolCallArgsDelta { index, delta } => {
            if let Some(&block_index) = blocks.in_progress.get(&index) {
                sse.event_data(
                    "content_block_delta",
                    &anthropic::MessageStreamEvent::ContentBlockDelta {
                        index: block_index,
                        delta: anthropic::ContentBlockDelta::InputJsonDelta {
                            partial_json: delta,
                        },
                    },
                )?;
            }
        }
        DecodeEvent::ToolCallEnd { index, .. } => {
            if let Some(block_index) = blocks.in_progress.remove(&index) {
                sse.event_data(
                    "content_block_stop",
                    &anthropic::MessageStreamEvent::ContentBlockStop { index: block_index },
                )?;
                blocks.open = AnthropicOpen::None;
            }
        }
        DecodeEvent::Stop { .. } => {}
        DecodeEvent::UnknownTool { name, .. } => {
            return Ok(Some(format!("model called unknown tool `{name}`")));
        }
        DecodeEvent::InvalidArgs { name, errors, .. } => {
            let detail = errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Ok(Some(format!(
                "model called `{name}` with arguments that don't match the schema: {detail}"
            )));
        }
        DecodeEvent::ParseError { sentinel, source } => {
            return Ok(Some(format!(
                "model emitted malformed tool call within `{sentinel}`: {source}"
            )));
        }
    }
    Ok(None)
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
