use crate::abstract_interpreter::{CoreSSA, InterpreterError};
use crate::category::core::Dtype;
use std::fmt::{Display, Formatter, Result as FmtResult};

use super::interpreter::Value;
use super::value_types::*;

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Value::Type(type_expr) => write!(f, "{type_expr}"),
            Value::Shape(shape_expr) => write!(f, "{shape_expr}"),
            Value::Nat(nat_expr) => write!(f, "{nat_expr}"),
            Value::Dtype(dtype_expr) => write!(f, "{dtype_expr}"),
            Value::Tensor(type_expr) => write!(f, "{type_expr}"),
        }
    }
}

impl Display for TypeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TypeExpr::Var(n) => write!(f, "v{n}"),
            TypeExpr::NdArrayType(array_type) => write!(f, "{array_type}"),
        }
    }
}

impl Display for ShapeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ShapeExpr::Var(i) => write!(f, "v{i}"),
            ShapeExpr::OfType(i) => write!(f, "shape_of(v{i})"),
            ShapeExpr::Shape(shape) => {
                write!(f, "[")?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dim}")?;
                }
                write!(f, "]")
            }
        }
    }
}

impl Display for NdArrayType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match &self.shape {
            ShapeExpr::Var(i) => write!(f, "v{i} : {}", self.dtype),
            ShapeExpr::OfType(i) => write!(f, "shape_of(v{i}) : {}", self.dtype),
            ShapeExpr::Shape(shape) => {
                write!(f, "{}[", self.dtype)?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{dim}")?;
                }
                write!(f, "]")
            }
        }
    }
}

impl Display for NatExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            NatExpr::Var(n) => write!(f, "v{n}"),
            NatExpr::Constant(c) => write!(f, "{c}"),
            NatExpr::Mul(terms) => {
                if terms.is_empty() {
                    write!(f, "1")
                } else if terms.len() == 1 {
                    write!(f, "{}", terms[0])
                } else {
                    write!(f, "(")?;
                    for (i, term) in terms.iter().enumerate() {
                        if i > 0 {
                            write!(f, "*")?;
                        }
                        write!(f, "{term}")?;
                    }
                    write!(f, ")")
                }
            }
            NatExpr::Add(terms) => {
                if terms.is_empty() {
                    write!(f, "0")
                } else if terms.len() == 1 {
                    write!(f, "{}", terms[0])
                } else {
                    write!(f, "(")?;
                    for (i, term) in terms.iter().enumerate() {
                        if i > 0 {
                            write!(f, "+")?;
                        }
                        write!(f, "{term}")?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}

impl Display for DtypeExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            DtypeExpr::Var(n) => write!(f, "v{n}"),
            DtypeExpr::OfType(n) => write!(f, "dtype(v{n})"),
            DtypeExpr::Constant(dtype) => write!(f, "{dtype}"),
        }
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Dtype::F32 => write!(f, "f32"),
            Dtype::F16 => write!(f, "f16"),
            Dtype::BF16 => write!(f, "bf16"),
            Dtype::U32 => write!(f, "u32"),
        }
    }
}

pub(crate) fn type_error(
    ssa: &CoreSSA,
    op_name: &str,
    inputs: &[Value],
    detail: impl Into<String>,
) -> InterpreterError {
    InterpreterError::Typecheck(format_error(ssa, op_name, inputs, detail.into()))
}

pub(crate) fn format_values(values: &[Value]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_error(ssa: &CoreSSA, op_name: &str, inputs: &[Value], detail: String) -> String {
    let mut message = format!(
        "typecheck failed at {}: {detail}",
        format_edge(ssa, op_name)
    );

    if !inputs.is_empty() {
        message.push_str("\ninputs:");
        for ((node_id, _), value) in ssa.sources.iter().zip(inputs) {
            message.push_str(&format!("\n  v{}: {value}", node_id.0));
        }
    }

    message
}

fn format_edge(ssa: &CoreSSA, op_name: &str) -> String {
    let targets = ssa
        .targets
        .iter()
        .map(|(node_id, _)| format!("v{}", node_id.0))
        .collect::<Vec<_>>()
        .join(", ");
    let sources = ssa
        .sources
        .iter()
        .map(|(node_id, _)| format!("v{}", node_id.0))
        .collect::<Vec<_>>()
        .join(", ");

    if targets.is_empty() {
        format!("edge {} {op_name}({sources})", ssa.edge_id.0)
    } else {
        format!("edge {} {targets} = {op_name}({sources})", ssa.edge_id.0)
    }
}
