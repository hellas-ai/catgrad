//! Transport-agnostic API serving helpers built on top of [`crate::run::ModelEngine`].
use crate::Result;
use crate::run::ModelEngine;
use crate::types::{anthropic, openai};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Shared serving configuration that is independent from any HTTP transport.
#[derive(Debug, Clone)]
pub struct ApiContext {
    model_name: String,
    served_model_ids: Vec<String>,
    default_max_tokens: u32,
}

impl ApiContext {
    pub fn new(
        model_name: impl Into<String>,
        served_model_ids: Vec<String>,
        default_max_tokens: u32,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            served_model_ids,
            default_max_tokens,
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn served_model_ids(&self) -> &[String] {
        &self.served_model_ids
    }

    pub fn default_max_tokens(&self) -> u32 {
        self.default_max_tokens
    }
}

/// Result of handling an API request before transport framing is applied.
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointResult<T> {
    Json(T),
    Streamed,
}

/// Named SSE event payload for APIs that use explicit event names.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedEvent<T> {
    pub event: &'static str,
    pub payload: T,
}

pub fn openai_models_response(context: &ApiContext) -> Value {
    json!({
        "object": "list",
        "data": context.served_model_ids()
            .iter()
            .map(|model_id| {
                json!({
                    "id": model_id,
                    "object": "model"
                })
            })
            .collect::<Vec<_>>()
    })
}

pub fn handle_openai_chat<F>(
    engine: &ModelEngine,
    context: &ApiContext,
    request: &openai::ChatCompletionRequest,
    mut on_chunk: F,
) -> Result<EndpointResult<openai::ChatCompletionResponse>>
where
    F: FnMut(openai::ChatCompletionChunk) -> Result<()>,
{
    let prepared = engine.prepare_openai_chat_request(request)?;
    let max_tokens = request.max_tokens.unwrap_or(context.default_max_tokens());
    let model = context.model_name().to_string();

    if request.stream == Some(true) {
        let id = next_id("chatcmpl");
        let created = now_unix();

        on_chunk(
            openai::ChatCompletionChunk::builder()
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
                .build(),
        )?;

        let generated = engine.generate_from_prepared(&prepared, max_tokens, |delta| {
            on_chunk(
                openai::ChatCompletionChunk::builder()
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
                    .build(),
            )
        })?;

        on_chunk(
            openai::ChatCompletionChunk::builder()
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
                .build(),
        )?;

        let include_usage = request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false);
        if include_usage {
            on_chunk(
                openai::ChatCompletionChunk::builder()
                    .id(id)
                    .object("chat.completion.chunk".to_string())
                    .created(created)
                    .model(model)
                    .choices(vec![])
                    .usage(Some(openai::Usage::from_counts(
                        generated.prompt_tokens,
                        generated.completion_tokens,
                    )))
                    .build(),
            )?;
        }

        return Ok(EndpointResult::Streamed);
    }

    let generated = engine.generate_from_prepared(&prepared, max_tokens, |_| Ok(()))?;
    Ok(EndpointResult::Json(
        openai::ChatCompletionResponse::builder()
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
            .build(),
    ))
}

pub fn handle_openai_responses<F>(
    engine: &ModelEngine,
    context: &ApiContext,
    request: &openai::responses::ResponseRequest,
    mut on_event: F,
) -> Result<EndpointResult<openai::responses::Response>>
where
    F: FnMut(openai::responses::ResponseStreamEvent) -> Result<()>,
{
    let prepared = engine.prepare_openai_response_request(request)?;
    let max_tokens = request
        .max_output_tokens
        .unwrap_or(context.default_max_tokens());
    let model = context.model_name().to_string();

    if request.stream == Some(true) {
        let response_id = next_id("resp");
        let message_id = next_id("msg");
        let created = now_unix();
        let mut sequence_number = 0u64;

        on_event(openai::responses::ResponseStreamEvent::Created {
            sequence_number,
            response: build_openai_response(
                &response_id,
                created,
                &model,
                openai::responses::ResponseStatus::Queued,
                vec![],
                None,
            ),
        })?;
        sequence_number += 1;

        on_event(openai::responses::ResponseStreamEvent::InProgress {
            sequence_number,
            response: build_openai_response(
                &response_id,
                created,
                &model,
                openai::responses::ResponseStatus::InProgress,
                vec![],
                None,
            ),
        })?;
        sequence_number += 1;

        on_event(openai::responses::ResponseStreamEvent::OutputItemAdded {
            sequence_number,
            output_index: 0,
            item: build_openai_response_message(
                &message_id,
                openai::responses::ResponseStatus::InProgress,
                vec![],
            ),
        })?;
        sequence_number += 1;

        on_event(openai::responses::ResponseStreamEvent::ContentPartAdded {
            sequence_number,
            item_id: message_id.clone(),
            output_index: 0,
            content_index: 0,
            part: build_openai_response_text(String::new()),
        })?;
        sequence_number += 1;

        let mut text = String::new();
        let generated = engine.generate_from_prepared(&prepared, max_tokens, |delta| {
            text.push_str(delta);
            on_event(openai::responses::ResponseStreamEvent::OutputTextDelta {
                sequence_number,
                item_id: message_id.clone(),
                output_index: 0,
                content_index: 0,
                delta: delta.to_string(),
            })?;
            sequence_number += 1;
            Ok(())
        })?;

        on_event(openai::responses::ResponseStreamEvent::OutputTextDone {
            sequence_number,
            item_id: message_id.clone(),
            output_index: 0,
            content_index: 0,
            text: text.clone(),
        })?;
        sequence_number += 1;

        let completed_part = build_openai_response_text(text);
        on_event(openai::responses::ResponseStreamEvent::ContentPartDone {
            sequence_number,
            item_id: message_id.clone(),
            output_index: 0,
            content_index: 0,
            part: completed_part.clone(),
        })?;
        sequence_number += 1;

        let completed_item = build_openai_response_message(
            &message_id,
            openai::responses::ResponseStatus::Completed,
            vec![completed_part],
        );
        on_event(openai::responses::ResponseStreamEvent::OutputItemDone {
            sequence_number,
            output_index: 0,
            item: completed_item.clone(),
        })?;
        sequence_number += 1;

        on_event(openai::responses::ResponseStreamEvent::Completed {
            sequence_number,
            response: build_openai_response(
                &response_id,
                created,
                &model,
                openai::responses::ResponseStatus::Completed,
                vec![completed_item],
                Some(openai::responses::ResponseUsage::from_counts(
                    generated.prompt_tokens,
                    generated.completion_tokens,
                )),
            ),
        })?;

        return Ok(EndpointResult::Streamed);
    }

    let generated = engine.generate_from_prepared(&prepared, max_tokens, |_| Ok(()))?;
    Ok(EndpointResult::Json(build_openai_response(
        &next_id("resp"),
        now_unix(),
        &model,
        openai::responses::ResponseStatus::Completed,
        vec![build_openai_response_message(
            &next_id("msg"),
            openai::responses::ResponseStatus::Completed,
            vec![build_openai_response_text(generated.text)],
        )],
        Some(openai::responses::ResponseUsage::from_counts(
            generated.prompt_tokens,
            generated.completion_tokens,
        )),
    )))
}

pub fn handle_anthropic_messages<F>(
    engine: &ModelEngine,
    context: &ApiContext,
    request: &anthropic::MessageRequest,
    mut on_event: F,
) -> Result<EndpointResult<anthropic::MessageResponse>>
where
    F: FnMut(NamedEvent<anthropic::MessageStreamEvent>) -> Result<()>,
{
    let prepared = engine.prepare_anthropic(request)?;
    let model = context.model_name().to_string();

    if request.stream == Some(true) {
        let id = next_id("msg");
        on_event(NamedEvent {
            event: "message_start",
            payload: anthropic::MessageStreamEvent::MessageStart {
                message: anthropic::MessageResponse::builder()
                    .id(id)
                    .message_type(Some("message".to_string()))
                    .role("assistant".to_string())
                    .content(vec![])
                    .model(model)
                    .usage(anthropic::AnthropicUsage::new(
                        prepared.input_ids.len() as u32,
                        0,
                    ))
                    .build(),
            },
        })?;
        on_event(NamedEvent {
            event: "content_block_start",
            payload: anthropic::MessageStreamEvent::ContentBlockStart {
                index: 0,
                content_block: anthropic::ContentBlock::Text {
                    text: String::new(),
                },
            },
        })?;

        let generated = engine.generate_from_prepared(&prepared, request.max_tokens, |delta| {
            on_event(NamedEvent {
                event: "content_block_delta",
                payload: anthropic::MessageStreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: anthropic::ContentBlockDelta::TextDelta {
                        text: delta.to_string(),
                    },
                },
            })
        })?;

        on_event(NamedEvent {
            event: "content_block_stop",
            payload: anthropic::MessageStreamEvent::ContentBlockStop { index: 0 },
        })?;
        on_event(NamedEvent {
            event: "message_delta",
            payload: anthropic::MessageStreamEvent::MessageDelta {
                delta: anthropic::StreamMessageDelta {
                    stop_reason: Some(generated.termination.into()),
                },
                usage: anthropic::AnthropicUsage::new(
                    generated.prompt_tokens,
                    generated.completion_tokens,
                ),
            },
        })?;
        on_event(NamedEvent {
            event: "message_stop",
            payload: anthropic::MessageStreamEvent::MessageStop,
        })?;

        return Ok(EndpointResult::Streamed);
    }

    let generated = engine.generate_from_prepared(&prepared, request.max_tokens, |_| Ok(()))?;
    Ok(EndpointResult::Json(
        anthropic::MessageResponse::builder()
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
            .build(),
    ))
}

fn build_openai_response(
    response_id: &str,
    created_at: i64,
    model: &str,
    status: openai::responses::ResponseStatus,
    output: Vec<openai::responses::ResponseOutputMessage>,
    usage: Option<openai::responses::ResponseUsage>,
) -> openai::responses::Response {
    openai::responses::Response::builder()
        .id(response_id.to_string())
        .object("response".to_string())
        .created_at(created_at)
        .status(status)
        .model(model.to_string())
        .output(output)
        .usage(usage)
        .build()
}

fn build_openai_response_message(
    message_id: &str,
    status: openai::responses::ResponseStatus,
    content: Vec<openai::responses::ResponseOutputText>,
) -> openai::responses::ResponseOutputMessage {
    openai::responses::ResponseOutputMessage::builder()
        .id(message_id.to_string())
        .item_type("message".to_string())
        .status(status)
        .role("assistant".to_string())
        .content(content)
        .annotations(Some(vec![]))
        .build()
}

fn build_openai_response_text(text: String) -> openai::responses::ResponseOutputText {
    openai::responses::ResponseOutputText::builder()
        .content_type("output_text".to_string())
        .text(text)
        .annotations(Some(vec![]))
        .build()
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn next_id(prefix: &str) -> String {
    let value = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{value}")
}
