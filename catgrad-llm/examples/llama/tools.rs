use anyhow::{Result, anyhow};
use catgrad_llm::helpers::ToolCall;
use minijinja::{Value, context};
use serde_json::{Map, Value as JsonValue};

pub fn tool_schemas() -> Vec<Value> {
    vec![calculator_schema()]
}

pub fn execute_tool_call(tool_call: &ToolCall) -> Result<String> {
    match tool_call.name.as_str() {
        "calculator" => {
            let result = calculator(&tool_call.arguments)?;
            Ok(serde_json::to_string(
                &serde_json::json!({ "result": result }),
            )?)
        }
        other => Err(anyhow!("unsupported tool call: {other}")),
    }
}

fn calculator(arguments: &Map<String, JsonValue>) -> Result<f64> {
    let left = required_number(arguments, &["lhs", "left"])?;
    let right = required_number(arguments, &["rhs", "right"])?;
    let operator = required_string(arguments, &["op", "operator"])?;

    match operator {
        "add" => Ok(left + right),
        "sub" | "subtract" => Ok(left - right),
        "mul" | "multiply" => Ok(left * right),
        "div" | "divide" => {
            if right == 0.0 {
                Err(anyhow!("division by zero"))
            } else {
                Ok(left / right)
            }
        }
        _ => Err(anyhow!("unsupported calculator operator: {operator}")),
    }
}

fn required_number(arguments: &Map<String, JsonValue>, keys: &[&str]) -> Result<f64> {
    keys.iter()
        .find_map(|key| arguments.get(*key).and_then(JsonValue::as_f64))
        .ok_or_else(|| anyhow!("tool argument `{}` must be a number", keys[0]))
}

fn required_string<'a>(arguments: &'a Map<String, JsonValue>, keys: &[&str]) -> Result<&'a str> {
    keys.iter()
        .find_map(|key| arguments.get(*key).and_then(JsonValue::as_str))
        .ok_or_else(|| anyhow!("tool argument `{}` must be a string", keys[0]))
}

fn calculator_schema() -> Value {
    context!(
        type => "function",
        function => context!(
            name => "calculator",
            description => "Calc lhs op rhs.",
            parameters => context!(
                type => "object",
                properties => context!(
                    lhs => context!(type => "number"),
                    rhs => context!(type => "number"),
                    op => context!(type => "string", enum => ["add", "sub", "mul", "div"]),
                ),
                required => ["lhs", "rhs", "op"],
            )
        )
    )
}
