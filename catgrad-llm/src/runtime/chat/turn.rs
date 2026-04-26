//! [`ChatTurn`] — the single value that carries the binding established
//! at chat-template render time through to tool-call parse time.
//!
//! Construction binds `(architecture, tools, protocol)`. Render and
//! parser construction both consult that binding, so the false-positive
//! and unknown-tool defect classes become structurally impossible:
//!
//! - With no tools bound, [`Self::make_parser`] returns a passthrough
//!   parser that emits only `TextDelta` and `Stop`. The per-architecture
//!   parser is never instantiated, so no sentinel scan ever runs.
//! - With tools bound, [`Self::make_parser`] returns the architecture's
//!   parser bound to the same [`ToolDirectory`] used at render time, so
//!   parsed call names are validated against exactly the tools that
//!   were offered.

use std::sync::Arc;

use serde_json::Value as JsonValue;
use tokenizers::tokenizer::Tokenizer;

use crate::error::LLMError;
use crate::types;
use crate::utils::{PreparedPrompt, RenderChatTemplateOptions};
use crate::{Result, runtime::chat::PassthroughParser};

use super::protocol::{ToolCallProtocol, tool_protocol_for};
use super::{IncrementalToolCallParser, ToolDirectory};

/// Per-turn options that aren't request-content (messages) or
/// per-architecture (protocol).
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatOptions {
    /// Forwarded to the chat template as the `enable_thinking` variable.
    /// Models that support a "thinking" decoding mode read this to
    /// decide whether to insert thinking tags.
    pub enable_thinking: bool,
}

/// One chat-turn binding: model resources + tools + options. Hold this
/// across the executor RPC await; consult it again at parse time.
///
/// All shared resources are `Arc`'d so a `ChatTurn` is `Send + Sync`
/// and can move freely across `.await` points.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    arch: String,
    chat_template: Arc<str>,
    tokenizer: Arc<Tokenizer>,
    tokenizer_config: Arc<JsonValue>,
    stop_token_ids: Arc<[i32]>,
    tools: Option<Arc<ToolDirectory>>,
    /// Some iff `tools` is some — `None` means either no tools were
    /// bound, or the architecture has no protocol but tools weren't
    /// requested. The `(Some(tools), None)` combination is rejected at
    /// construction time, so this invariant holds at use sites.
    protocol: Option<&'static ToolCallProtocol>,
    options: ChatOptions,
}

impl ChatTurn {
    /// Build a turn. Returns an error iff a non-empty `tools`
    /// directory is bound for an architecture with no registered
    /// [`ToolCallProtocol`].
    ///
    /// `Some(empty_directory)` is normalized to `None` before any
    /// protocol lookup — an empty tools list is semantically identical
    /// to "no tools requested" and must not subject the request to the
    /// protocol-required path. This means a client that sends
    /// `tools: []` against an unsupported architecture is accepted as
    /// plain chat, and a client that sends `tools: []` against a
    /// supported architecture gets passthrough parsing (sentinel-shaped
    /// model output remains plain text).
    pub fn new(
        arch: impl Into<String>,
        chat_template: Arc<str>,
        tokenizer: Arc<Tokenizer>,
        tokenizer_config: Arc<JsonValue>,
        stop_token_ids: Arc<[i32]>,
        tools: Option<Arc<ToolDirectory>>,
        options: ChatOptions,
    ) -> Result<Self> {
        let arch = arch.into();
        let tools = tools.filter(|dir| !dir.is_empty());
        let protocol = match (&tools, tool_protocol_for(&arch)) {
            (None, _) => None,
            (Some(_), Some(p)) => Some(p),
            (Some(_), None) => {
                return Err(LLMError::UnsupportedModel(format!(
                    "model `{arch}` does not support tool calling \
                     (no tool-call protocol registered for this architecture)"
                )));
            }
        };
        Ok(Self {
            arch,
            chat_template,
            tokenizer,
            tokenizer_config,
            stop_token_ids,
            tools,
            protocol,
            options,
        })
    }

    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn tools(&self) -> Option<&Arc<ToolDirectory>> {
        self.tools.as_ref()
    }

    pub fn options(&self) -> ChatOptions {
        self.options
    }

    pub fn protocol(&self) -> Option<&'static ToolCallProtocol> {
        self.protocol
    }

    /// Render the chat template and tokenize. Pure: no inference, no
    /// network. The single tool-enabled render path in the crate —
    /// architecture-specific shaping happens here via the bound
    /// protocol's `render_tools`, and the result is fed into a
    /// crate-private `PreparedPrompt::from_messages_with_tools`. No
    /// other code path can inject a `tools` variable into the chat
    /// template.
    pub fn render(&self, messages: &[types::Message]) -> Result<PreparedPrompt> {
        // `tools` here is None unless ChatTurn was constructed with a
        // non-empty ToolDirectory AND the architecture has a registered
        // protocol. Both conditions are enforced by `Self::new`.
        let shaped_tools = if let Some(tools) = &self.tools {
            let render_fn = self
                .protocol
                .expect("invariant: tools=Some implies protocol=Some")
                .render_tools;
            Some(render_fn(tools.specs()))
        } else {
            None
        };
        PreparedPrompt::from_messages_with_tools(
            &self.tokenizer,
            &self.chat_template,
            &self.tokenizer_config,
            messages,
            &self.stop_token_ids,
            RenderChatTemplateOptions {
                enable_thinking: self.options.enable_thinking,
            },
            shaped_tools.as_ref(),
        )
    }

    /// Construct an incremental parser bound to this turn's tool
    /// directory. With no tools bound, returns a passthrough that
    /// emits only `TextDelta` and `Stop` — the per-arch parser is
    /// never instantiated, so sentinel-shaped text in plain chat
    /// cannot become a false-positive tool call.
    ///
    /// The returned parser is `'static`: it owns its tool directory
    /// via `Arc::clone`. Callers may store it alongside the
    /// `ChatTurn` it came from without a self-referential borrow.
    pub fn make_parser(&self) -> Box<dyn IncrementalToolCallParser> {
        match (&self.tools, self.protocol) {
            (Some(tools), Some(protocol)) => (protocol.make_parser)(Arc::clone(tools)),
            _ => Box::new(PassthroughParser),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::ToolSpec;
    use crate::runtime::chat::event::{DecodeEvent, StopReason};
    use serde_json::json;

    fn dummy_tokenizer() -> Arc<Tokenizer> {
        // Build a minimal Tokenizer from a tiny JSON config so we don't
        // need a real model on disk for these tests.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"a": 0, "b": 1, "c": 2},
                "unk_token": "a"
            }
        }"#;
        Arc::new(Tokenizer::from_bytes(json.as_bytes()).unwrap())
    }

    fn dummy_template() -> Arc<str> {
        Arc::from("{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}")
    }

    fn dummy_config() -> Arc<JsonValue> {
        Arc::new(json!({"bos_token": ""}))
    }

    fn dummy_stop_ids() -> Arc<[i32]> {
        Arc::from(vec![0_i32].as_slice())
    }

    fn calculator_directory() -> Arc<ToolDirectory> {
        Arc::new(
            ToolDirectory::new(vec![ToolSpec::new(
                "add",
                None,
                json!({
                    "type": "object",
                    "properties": {
                        "a": { "type": "number" },
                        "b": { "type": "number" },
                    },
                    "required": ["a", "b"],
                }),
            )])
            .unwrap(),
        )
    }

    #[test]
    fn no_tools_bound_constructs_for_any_architecture() {
        let turn = ChatTurn::new(
            "LlamaForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            None,
            ChatOptions::default(),
        );
        assert!(turn.is_ok());
    }

    #[test]
    fn tools_bound_with_supported_arch_constructs() {
        let turn = ChatTurn::new(
            "Qwen3ForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            Some(calculator_directory()),
            ChatOptions::default(),
        );
        assert!(turn.is_ok());
        assert!(turn.unwrap().protocol().is_some());
    }

    #[test]
    fn tools_bound_with_unsupported_arch_errors() {
        let err = ChatTurn::new(
            "LlamaForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            Some(calculator_directory()),
            ChatOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not support tool calling"));
        assert!(err.to_string().contains("LlamaForCausalLM"));
    }

    #[test]
    fn make_parser_with_no_tools_is_passthrough() {
        let turn = ChatTurn::new(
            "Qwen3ForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            None,
            ChatOptions::default(),
        )
        .unwrap();
        let mut parser = turn.make_parser();
        // Even sentinel-shaped text must pass through as plain text
        // when no tools are bound.
        let events =
            parser.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        let mut text = String::new();
        for ev in &events {
            if let DecodeEvent::TextDelta(s) = ev {
                text.push_str(s);
            } else {
                panic!("passthrough must not emit non-text events: {ev:?}");
            }
        }
        assert_eq!(
            text,
            r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#
        );
        let stop = parser.finish(StopReason::EndOfText);
        assert!(matches!(stop[0], DecodeEvent::Stop { .. }));
    }

    #[test]
    fn make_parser_with_tools_uses_qwen3() {
        let turn = ChatTurn::new(
            "Qwen3ForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            Some(calculator_directory()),
            ChatOptions::default(),
        )
        .unwrap();
        let mut parser = turn.make_parser();
        let events =
            parser.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        assert!(matches!(events[0], DecodeEvent::ToolCallStart { .. }));
        assert!(matches!(events[2], DecodeEvent::ToolCallEnd { .. }));
    }

    #[test]
    fn empty_tools_directory_on_unsupported_arch_is_accepted() {
        // Some(empty) must behave exactly like None: the protocol-required
        // path is never taken, so unsupported architectures accept the turn.
        let empty = Arc::new(ToolDirectory::new(vec![]).unwrap());
        let turn = ChatTurn::new(
            "LlamaForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            Some(empty),
            ChatOptions::default(),
        )
        .expect("empty tools must be accepted on unsupported arch");
        assert!(turn.tools().is_none(), "empty tools should normalize to None");
        assert!(turn.protocol().is_none());
    }

    #[test]
    fn empty_tools_directory_on_supported_arch_uses_passthrough() {
        // Some(empty) on a supported architecture must also normalize to
        // None — the per-arch parser must not be instantiated. Sentinel-
        // shaped text in the model output stays as plain text.
        let empty = Arc::new(ToolDirectory::new(vec![]).unwrap());
        let turn = ChatTurn::new(
            "Qwen3ForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            Some(empty),
            ChatOptions::default(),
        )
        .expect("empty tools must be accepted on supported arch");
        assert!(turn.tools().is_none(), "empty tools should normalize to None");
        let mut parser = turn.make_parser();
        let events =
            parser.feed(r#"<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>"#);
        for ev in &events {
            assert!(
                matches!(ev, DecodeEvent::TextDelta(_)),
                "passthrough must emit only text; got {ev:?}"
            );
        }
    }

    #[test]
    fn render_with_no_tools_omits_tools_from_template() {
        let turn = ChatTurn::new(
            "Qwen3ForCausalLM",
            dummy_template(),
            dummy_tokenizer(),
            dummy_config(),
            dummy_stop_ids(),
            None,
            ChatOptions::default(),
        )
        .unwrap();
        let messages = vec![types::Message::OpenAI(Box::new(
            crate::types::openai::ChatMessage::user("hi"),
        ))];
        let prepared = turn.render(&messages).unwrap();
        // Just check that render produces something — the dummy template
        // doesn't read `tools`, so we mainly want to confirm no error.
        assert!(!prepared.input_ids.is_empty());
    }
}
