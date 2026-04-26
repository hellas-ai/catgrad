//! Typed tool inventory bound to a chat turn.
//!
//! A [`ToolDirectory`] is constructed once per chat request from the
//! caller's wire-format tool list. Schemas are pre-compiled at
//! construction so per-call validation in the streaming parser is a
//! single hashtable lookup plus a `validate` traversal — no
//! per-invocation parse cost.

use crate::Result;
use crate::error::LLMError;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use super::event::SchemaError;

/// One offered tool: a name, a free-text description, and a JSON-schema
/// describing the accepted argument shape.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    /// JSON-Schema for the `arguments` object the model is expected to
    /// emit. Validated once at [`ToolDirectory::new`] time.
    pub parameters: JsonValue,
}

impl ToolSpec {
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        parameters: JsonValue,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            parameters,
        }
    }
}

/// Bound set of tools for a single chat turn.
///
/// Holds the original [`ToolSpec`] list (for chat-template rendering) and
/// a parallel map of pre-compiled `jsonschema` validators (for argument
/// validation inside the streaming parser).
#[derive(Debug)]
pub struct ToolDirectory {
    specs: Vec<ToolSpec>,
    validators: HashMap<String, jsonschema::Validator>,
}

impl ToolDirectory {
    /// Compile the parameter schema for each tool. Returns an error on
    /// duplicate names or on any schema that is not itself a valid
    /// JSON-Schema document.
    pub fn new(specs: Vec<ToolSpec>) -> Result<Self> {
        let mut validators = HashMap::with_capacity(specs.len());
        for spec in &specs {
            if validators.contains_key(&spec.name) {
                return Err(LLMError::InvalidModelConfig(format!(
                    "duplicate tool name in directory: `{}`",
                    spec.name
                )));
            }
            let validator = jsonschema::validator_for(&spec.parameters).map_err(|err| {
                LLMError::InvalidModelConfig(format!(
                    "tool `{}` parameters are not a valid JSON Schema: {err}",
                    spec.name
                ))
            })?;
            validators.insert(spec.name.clone(), validator);
        }
        Ok(Self { specs, validators })
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// Look up a tool by the name a model emitted. Returns `None` for
    /// hallucinated names — caller emits
    /// [`DecodeEvent::UnknownTool`](super::DecodeEvent::UnknownTool).
    pub fn lookup(&self, name: &str) -> Option<(&ToolSpec, &jsonschema::Validator)> {
        let spec = self.specs.iter().find(|s| s.name == name)?;
        let validator = self.validators.get(name)?;
        Some((spec, validator))
    }

    /// Validate `args` against the named tool's schema. Returns the list
    /// of failures suitable for direct inclusion in
    /// [`DecodeEvent::InvalidArgs`](super::DecodeEvent::InvalidArgs).
    ///
    /// Returns an empty `Vec` if the tool name is unknown — callers
    /// should have already gated on [`Self::lookup`] before reaching
    /// validation. The empty return is therefore a "passed" result, but
    /// it is the caller's responsibility not to call this on an unknown
    /// tool.
    pub fn validate_args(&self, name: &str, args: &JsonValue) -> Vec<SchemaError> {
        let Some(validator) = self.validators.get(name) else {
            return Vec::new();
        };
        validator
            .iter_errors(args)
            .map(|err| SchemaError {
                path: err.instance_path.to_string(),
                message: err.to_string(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn calculator_spec() -> ToolSpec {
        ToolSpec::new(
            "add",
            Some("add two numbers".into()),
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" },
                },
                "required": ["a", "b"],
                "additionalProperties": false,
            }),
        )
    }

    #[test]
    fn directory_compiles_schemas_and_looks_up_by_name() {
        let dir = ToolDirectory::new(vec![calculator_spec()]).unwrap();
        let (spec, _validator) = dir.lookup("add").expect("add tool present");
        assert_eq!(spec.name, "add");
        assert!(dir.lookup("subtract").is_none());
    }

    #[test]
    fn directory_rejects_duplicate_tool_names() {
        let err = ToolDirectory::new(vec![calculator_spec(), calculator_spec()]).unwrap_err();
        assert!(err.to_string().contains("duplicate tool name"));
    }

    #[test]
    fn directory_rejects_invalid_schema() {
        let bad = ToolSpec::new("x", None, json!({ "type": 42 }));
        let err = ToolDirectory::new(vec![bad]).unwrap_err();
        assert!(err.to_string().contains("not a valid JSON Schema"));
    }

    #[test]
    fn validate_args_accepts_well_formed_input() {
        let dir = ToolDirectory::new(vec![calculator_spec()]).unwrap();
        let errors = dir.validate_args("add", &json!({ "a": 1, "b": 2 }));
        assert!(errors.is_empty(), "expected no errors, got {errors:?}");
    }

    #[test]
    fn validate_args_reports_missing_required_field() {
        let dir = ToolDirectory::new(vec![calculator_spec()]).unwrap();
        let errors = dir.validate_args("add", &json!({ "a": 1 }));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("\"b\""));
    }

    #[test]
    fn validate_args_reports_wrong_type() {
        let dir = ToolDirectory::new(vec![calculator_spec()]).unwrap();
        let errors = dir.validate_args("add", &json!({ "a": "one", "b": 2 }));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].path.contains("a"));
    }

    #[test]
    fn validate_args_rejects_extra_fields_when_schema_forbids() {
        let dir = ToolDirectory::new(vec![calculator_spec()]).unwrap();
        let errors = dir.validate_args("add", &json!({ "a": 1, "b": 2, "c": 3 }));
        assert!(!errors.is_empty());
    }
}
