//! Single-tool demo (a calculator) for the llama example's
//! `--tool-use` flag. Provides a typed [`ToolSpec`] for the model to
//! see and a local executor that turns parsed arguments into a
//! result string.
//!
//! Used only by the `--tool-use` path in `main.rs`. Adding more tools
//! means returning a longer list from [`tool_specs`] and adding a
//! `name` arm to [`execute`].

use anyhow::{Result, anyhow};
use catgrad_llm::runtime::chat::ToolSpec;
use serde_json::{Value, json};

pub fn tool_specs() -> Vec<ToolSpec> {
    vec![calculator_spec()]
}

/// Dispatch a parsed tool call by name. Returns the result as a JSON
/// string (the gateway / chat template treats `tool` role messages as
/// opaque text content).
pub fn execute(name: &str, args: &Value) -> Result<String> {
    match name {
        "calculator" => {
            let result = calculator(args)?;
            Ok(serde_json::to_string(&json!({ "result": result }))?)
        }
        other => Err(anyhow!("unsupported tool call: {other}")),
    }
}

fn calculator_spec() -> ToolSpec {
    ToolSpec::new(
        "calculator",
        Some("Calculate a result from two numbers using +, -, *, or /.".into()),
        json!({
            "type": "object",
            "properties": {
                "lhs": { "type": "number", "description": "The left-hand number." },
                "rhs": { "type": "number", "description": "The right-hand number." },
                "op": {
                    "type": "string",
                    "enum": ["add", "sub", "mul", "div"],
                    "description": "The operation to apply.",
                },
            },
            "required": ["lhs", "rhs", "op"],
        }),
    )
}

fn calculator(args: &Value) -> Result<f64> {
    let lhs = args
        .get("lhs")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("calculator: missing or non-numeric `lhs`"))?;
    let rhs = args
        .get("rhs")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("calculator: missing or non-numeric `rhs`"))?;
    let op = args
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("calculator: missing `op`"))?;
    match op {
        "add" => Ok(lhs + rhs),
        "sub" => Ok(lhs - rhs),
        "mul" => Ok(lhs * rhs),
        "div" if rhs != 0.0 => Ok(lhs / rhs),
        "div" => Err(anyhow!("division by zero")),
        other => Err(anyhow!("unsupported operator: {other}")),
    }
}
