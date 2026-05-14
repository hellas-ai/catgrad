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

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use catgrad::prelude::Dtype;
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use catgrad_llm::api::{self, ApiContext, EndpointResult};
use catgrad_llm::run::ModelEngine;
use catgrad_llm::types::{anthropic, openai};
use catgrad_llm::{LLMError, Result as LlmResult};

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
    api: ApiContext,
}

fn data_event<T: Serialize>(payload: &T) -> Event {
    Event::default().data(serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()))
}

fn named_event<T: Serialize>(name: &str, payload: &T) -> Event {
    Event::default()
        .event(name)
        .data(serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()))
}

fn done_event() -> Event {
    Event::default().data("[DONE]")
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
    F: FnOnce(&Worker, Option<&SseSink>) -> LlmResult<EndpointResult<T>> + Send + 'static,
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
            .send(Box::new(move |w| match f(w, Some(&tx)) {
                Ok(EndpointResult::Streamed) => {}
                Ok(EndpointResult::Json(_)) => {
                    let _ = tx.send(Ok(named_event(
                        "error",
                        &json!({ "message": "handler returned JSON for a streaming request" }),
                    )));
                }
                Err(err) => {
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
        return Sse::new(UnboundedReceiverStream::new(rx)).into_response();
    }

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
        Ok(Ok(EndpointResult::Json(t))) => Json(t).into_response(),
        Ok(Ok(EndpointResult::Streamed)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "handler unexpectedly streamed a non-streaming request",
        ),
        Ok(Err(err)) => llm_error_response(err),
        Err(_) => worker_dead(),
    }
}

// --- Handlers ---
async fn handle_models(State(state): State<AppState>) -> Response {
    Json(api::openai_models_response(&state.api)).into_response()
}

async fn handle_openai(
    State(state): State<AppState>,
    Json(req): Json<openai::ChatCompletionRequest>,
) -> Response {
    respond(&state.jobs, req.stream == Some(true), move |w, sse| {
        w.serve_openai(req, sse)
    })
    .await
}

async fn handle_openai_responses(
    State(state): State<AppState>,
    Json(req): Json<openai::responses::ResponseRequest>,
) -> Response {
    respond(&state.jobs, req.stream == Some(true), move |w, sse| {
        w.serve_openai_responses(req, sse)
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
struct Worker {
    engine: ModelEngine,
    api: ApiContext,
}

impl Worker {
    fn new(model: &str, api: ApiContext, use_kv_cache: bool) -> anyhow::Result<Self> {
        Ok(Self {
            engine: ModelEngine::new(model, use_kv_cache, Dtype::F32)?,
            api,
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
    ) -> LlmResult<EndpointResult<openai::ChatCompletionResponse>> {
        let result = api::handle_openai_chat(&self.engine, &self.api, &req, |chunk| {
            emit_maybe(sse, move || data_event(&chunk))
        })?;
        if matches!(result, EndpointResult::Streamed) {
            emit_maybe(sse, done_event)?;
        }
        Ok(result)
    }

    fn serve_openai_responses(
        &self,
        req: openai::responses::ResponseRequest,
        sse: Option<&SseSink>,
    ) -> LlmResult<EndpointResult<openai::responses::Response>> {
        let result = api::handle_openai_responses(&self.engine, &self.api, &req, |event| {
            emit_maybe(sse, move || data_event(&event))
        })?;
        if matches!(result, EndpointResult::Streamed) {
            emit_maybe(sse, done_event)?;
        }
        Ok(result)
    }

    fn serve_anthropic(
        &self,
        req: anthropic::MessageRequest,
        sse: Option<&SseSink>,
    ) -> LlmResult<EndpointResult<anthropic::MessageResponse>> {
        api::handle_anthropic_messages(&self.engine, &self.api, &req, |event| {
            emit_maybe(sse, move || named_event(event.event, &event.payload))
        })
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let api = ApiContext::new(
        &args.model,
        vec![args.model.clone()],
        args.default_max_tokens,
    );

    let (jobs_tx, jobs_rx) = mpsc::unbounded_channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

    // ModelEngine is `!Send` (Rc-backed). Construct it on the worker
    // thread, signal readiness, then run the dispatch loop.
    let model = args.model.clone();
    let use_kv_cache = args.use_kv_cache;
    let api_for_worker = api.clone();
    let worker_handle = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            println!("Loading model `{model}` (this can take a while)...");
            let worker = match Worker::new(&model, api_for_worker, use_kv_cache) {
                Ok(worker) => worker,
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
    rt.block_on(serve(args, AppState { jobs: jobs_tx, api }))?;
    let _ = worker_handle.join();
    Ok(())
}

async fn serve(args: Args, state: AppState) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_openai))
        .route("/v1/responses", post(handle_openai_responses))
        .route("/v1/messages", post(handle_anthropic))
        .with_state(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("catgrad demo API server listening on http://{addr}");
    println!("GET /v1/models (OpenAI)");
    println!("POST /v1/chat/completions (OpenAI)");
    println!("POST /v1/responses (OpenAI)");
    println!("POST /v1/messages (Anthropic)");
    axum::serve(listener, app).await?;
    Ok(())
}
