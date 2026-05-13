//! Demo HTTP server exposing OpenAI and Anthropic chat surfaces over a
//! single in-process `ModelEngine`.
//!
//! `ModelEngine` is `!Send` (Rc-backed), so the engine lives on a dedicated
//! worker thread. Handlers post closures into the worker via a
//! `tokio::sync::mpsc` channel; the closure runs on the worker thread with
//! `&Worker` and writes its result back through a `oneshot` or SSE channel.
//!
//! Known limitation: user-provided stop strings are ignored; only
//! model-native EOS stopping is supported.

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use catgrad::prelude::Dtype;
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use catgrad_llm::run::{GenerationOutput, ModelEngine};
use catgrad_llm::types::{anthropic, openai};
use catgrad_llm::{LLMError, Result as LlmResult};

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

/// SSE events flow back as `Result<Event, Infallible>` so the receiver
/// stream plugs straight into `Sse::new(...)` with no mapping.
type SseSink = mpsc::UnboundedSender<Result<Event, Infallible>>;
type Job = Box<dyn FnOnce(&Worker) + Send + 'static>;

#[derive(Clone)]
struct AppState {
    jobs: mpsc::UnboundedSender<Job>,
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn next_id(prefix: &str) -> String {
    format!("{prefix}-{}", NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

fn data_event<T: Serialize>(payload: &T) -> Event {
    Event::default().data(serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()))
}

fn named_event<T: Serialize>(name: &str, payload: &T) -> Event {
    Event::default()
        .event(name)
        .data(serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()))
}

/// Emit an event when an SSE sink is attached. Lazy: the closure only
/// runs in streaming mode. Returns `Err(IoError)` when the client has
/// disconnected, so the generation loop can short-circuit.
fn emit_maybe<F>(sse: Option<&SseSink>, event: F) -> LlmResult<()>
where
    F: FnOnce() -> Event,
{
    match sse {
        Some(sse) => sse
            .send(Ok(event()))
            .map_err(|_| std::io::Error::other("client disconnected").into()),
        None => Ok(()),
    }
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

fn llm_error_response(err: LLMError) -> Response {
    let status = match err {
        LLMError::UnsupportedTemplateFeature(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error_response(status, err.to_string())
}

/// Dispatch a request to the worker, returning either a single typed
/// JSON response (non-streaming) or an SSE stream the worker fills as
/// it goes. The worker closure sees `Option<&SseSink>` and emits
/// events through `emit_maybe` when one is attached.
async fn respond<T, F>(jobs: &mpsc::UnboundedSender<Job>, streaming: bool, f: F) -> Response
where
    T: Serialize + Send + 'static,
    F: FnOnce(&Worker, Option<&SseSink>) -> LlmResult<T> + Send + 'static,
{
    let worker_dead = || {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference worker terminated",
        )
    };
    if streaming {
        let (tx, rx) = mpsc::unbounded_channel();
        if jobs
            .send(Box::new(move |w| {
                if let Err(err) = f(w, Some(&tx)) {
                    let _ = tx.send(Ok(named_event(
                        "error",
                        &json!({ "message": err.to_string() }),
                    )));
                }
            }))
            .is_err()
        {
            return worker_dead();
        }
        Sse::new(UnboundedReceiverStream::new(rx)).into_response()
    } else {
        let (tx, rx) = oneshot::channel();
        if jobs
            .send(Box::new(move |w| {
                let _ = tx.send(f(w, None));
            }))
            .is_err()
        {
            return worker_dead();
        }
        match rx.await {
            Ok(Ok(t)) => Json(t).into_response(),
            Ok(Err(err)) => llm_error_response(err),
            Err(_) => worker_dead(),
        }
    }
}

// --- Handlers ---

async fn handle_openai(
    State(state): State<AppState>,
    Json(req): Json<openai::ChatCompletionRequest>,
) -> Response {
    respond(&state.jobs, req.stream == Some(true), move |w, sse| {
        w.serve_openai(req, sse)
    })
    .await
}

async fn handle_anthropic(
    State(state): State<AppState>,
    Json(req): Json<anthropic::MessageRequest>,
) -> Response {
    respond(&state.jobs, req.stream == Some(true), move |w, sse| {
        w.serve_anthropic(req, sse)
    })
    .await
}

// --- Worker ---

fn openai_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: openai::ChatDelta,
    finish_reason: Option<openai::FinishReason>,
) -> openai::ChatCompletionChunk {
    openai::ChatCompletionChunk::builder()
        .id(id.into())
        .object("chat.completion.chunk".into())
        .created(created)
        .model(model.into())
        .choices(vec![
            openai::ChatStreamChoice::builder()
                .index(0)
                .delta(delta)
                .finish_reason(finish_reason)
                .build(),
        ])
        .build()
}

fn openai_response(
    id: String,
    created: i64,
    model: String,
    g: GenerationOutput,
) -> openai::ChatCompletionResponse {
    openai::ChatCompletionResponse::builder()
        .id(id)
        .object("chat.completion".into())
        .created(created)
        .model(model)
        .choices(vec![
            openai::ChatChoice::builder()
                .index(0)
                .message(openai::ChatMessage::assistant(g.text))
                .finish_reason(Some(g.termination.into()))
                .build(),
        ])
        .usage(Some(openai::Usage::from_counts(
            g.prompt_tokens,
            g.completion_tokens,
        )))
        .build()
}

fn anthropic_response(
    id: String,
    model: String,
    g: GenerationOutput,
) -> anthropic::MessageResponse {
    anthropic::MessageResponse::builder()
        .id(id)
        .message_type(Some("message".into()))
        .role("assistant".into())
        .content(vec![anthropic::ContentBlock::Text { text: g.text }])
        .model(model)
        .stop_reason(Some(g.termination.into()))
        .usage(anthropic::AnthropicUsage::new(
            g.prompt_tokens,
            g.completion_tokens,
        ))
        .build()
}

struct Worker {
    engine: ModelEngine,
    model_name: String,
    default_max_tokens: u32,
}

impl Worker {
    fn new(model: &str, use_kv_cache: bool, default_max_tokens: u32) -> anyhow::Result<Self> {
        Ok(Self {
            engine: ModelEngine::new(model, use_kv_cache, Dtype::F32)?,
            model_name: model.to_string(),
            default_max_tokens,
        })
    }

    fn run(self, mut jobs: mpsc::UnboundedReceiver<Job>) {
        while let Some(job) = jobs.blocking_recv() {
            job(&self);
        }
    }

    fn serve_openai(
        &self,
        req: openai::ChatCompletionRequest,
        sse: Option<&SseSink>,
    ) -> LlmResult<openai::ChatCompletionResponse> {
        let max_tokens = req.max_tokens.unwrap_or(self.default_max_tokens);
        let include_usage = req
            .stream_options
            .as_ref()
            .and_then(|o| o.include_usage)
            .unwrap_or(false);
        let prepared = self.engine.prepare_openai(&req)?;

        let id = next_id("chatcmpl");
        let created = now_unix();
        let model = self.model_name.clone();

        emit_maybe(sse, || {
            data_event(&openai_chunk(
                &id,
                created,
                &model,
                openai::ChatDelta {
                    role: Some("assistant".into()),
                    ..Default::default()
                },
                None,
            ))
        })?;

        let g = self
            .engine
            .generate_from_prepared(&prepared, max_tokens, |delta| {
                emit_maybe(sse, || {
                    data_event(&openai_chunk(
                        &id,
                        created,
                        &model,
                        openai::ChatDelta {
                            content: Some(delta.into()),
                            ..Default::default()
                        },
                        None,
                    ))
                })
            })?;

        emit_maybe(sse, || {
            data_event(&openai_chunk(
                &id,
                created,
                &model,
                openai::ChatDelta::default(),
                Some(g.termination.into()),
            ))
        })?;

        if include_usage {
            emit_maybe(sse, || {
                let chunk = openai::ChatCompletionChunk::builder()
                    .id(id.clone())
                    .object("chat.completion.chunk".into())
                    .created(created)
                    .model(model.clone())
                    .choices(vec![])
                    .usage(Some(openai::Usage::from_counts(
                        g.prompt_tokens,
                        g.completion_tokens,
                    )))
                    .build();
                data_event(&chunk)
            })?;
        }
        emit_maybe(sse, || Event::default().data("[DONE]"))?;

        Ok(openai_response(id, created, model, g))
    }

    fn serve_anthropic(
        &self,
        req: anthropic::MessageRequest,
        sse: Option<&SseSink>,
    ) -> LlmResult<anthropic::MessageResponse> {
        use anthropic::MessageStreamEvent::*;
        let prepared = self.engine.prepare_anthropic(&req)?;
        let id = next_id("msg");
        let model = self.model_name.clone();
        let emit =
            |name, ev: anthropic::MessageStreamEvent| emit_maybe(sse, || named_event(name, &ev));

        emit(
            "message_start",
            MessageStart {
                message: anthropic::MessageResponse::builder()
                    .id(id.clone())
                    .message_type(Some("message".into()))
                    .role("assistant".into())
                    .content(vec![])
                    .model(model.clone())
                    .usage(anthropic::AnthropicUsage::new(
                        prepared.input_ids.len() as u32,
                        0,
                    ))
                    .build(),
            },
        )?;
        emit(
            "content_block_start",
            ContentBlockStart {
                index: 0,
                content_block: anthropic::ContentBlock::Text {
                    text: String::new(),
                },
            },
        )?;

        let g = self
            .engine
            .generate_from_prepared(&prepared, req.max_tokens, |delta| {
                emit(
                    "content_block_delta",
                    ContentBlockDelta {
                        index: 0,
                        delta: anthropic::ContentBlockDelta::TextDelta { text: delta.into() },
                    },
                )
            })?;

        emit("content_block_stop", ContentBlockStop { index: 0 })?;
        emit(
            "message_delta",
            MessageDelta {
                delta: anthropic::StreamMessageDelta {
                    stop_reason: Some(g.termination.into()),
                },
                usage: anthropic::AnthropicUsage::new(g.prompt_tokens, g.completion_tokens),
            },
        )?;
        emit("message_stop", MessageStop)?;

        Ok(anthropic_response(id, model, g))
    }
}

// --- main ---

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let (jobs_tx, jobs_rx) = mpsc::unbounded_channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

    // ModelEngine is `!Send` (Rc-backed). Construct it on the worker
    // thread, signal readiness, then run the dispatch loop.
    let model = args.model.clone();
    let use_kv_cache = args.use_kv_cache;
    let default_max_tokens = args.default_max_tokens;
    let worker_handle = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            println!("Loading model `{model}` (this can take a while)...");
            let worker = match Worker::new(&model, use_kv_cache, default_max_tokens) {
                Ok(w) => w,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            worker.run(jobs_rx);
        })?;
    ready_rx.recv()??;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()?;
    rt.block_on(serve(args, jobs_tx))?;
    let _ = worker_handle.join();
    Ok(())
}

async fn serve(args: Args, jobs: mpsc::UnboundedSender<Job>) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_openai))
        .route("/v1/messages", post(handle_anthropic))
        .with_state(AppState { jobs });
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("catgrad demo API server listening on http://{addr}");
    println!("POST /v1/chat/completions (OpenAI)");
    println!("POST /v1/messages (Anthropic)");
    axum::serve(listener, app).await?;
    Ok(())
}
