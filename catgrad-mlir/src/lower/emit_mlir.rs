//! Lower a catgrad::category::lang::Term to MLIR
use catgrad::category::lang;
use catgrad::prelude::*;
use catgrad::ssa::{SSA, ssa};

use open_hypergraphs::lax::{EdgeId, OpenHypergraph};

use super::grammar;
use super::util::*;

pub type Term = OpenHypergraph<typecheck::Type, lang::Operation>;

/// Lower a [`Term`] to MLIR text, giving it the specified name.
pub fn term_to_func(name: &str, term: Term) -> grammar::Func {
    let parameters = get_mlir_parameters(&term);
    let (return_stmt, return_type) = get_mlir_returns(&term);
    let body = get_mlir_body(term);

    grammar::Func {
        name: name.to_string(),
        parameters,
        return_type,
        body,
        return_stmt,
    }
}

/// Convert source nodes of a `Term` to MLIR function parameters
fn get_mlir_parameters(term: &Term) -> Vec<grammar::Parameter> {
    term.sources
        .iter()
        .map(|&source_id| {
            let source_type = &term.hypergraph.nodes[source_id.0];
            grammar::Parameter {
                name: format!("v{}", source_id.0),
                param_type: core_type_to_mlir(source_type),
            }
        })
        .collect()
}

/// Convert a term's target nodes and types into the *return type* and *return statement* for a
/// MLIR function.
pub fn get_mlir_returns(term: &Term) -> (grammar::Return, Vec<grammar::Type>) {
    let typed_ids = term
        .targets
        .iter()
        .map(|t| {
            let catgrad_ty = &term.hypergraph.nodes[t.0];
            let id = grammar::Identifier::Node(*t);
            let ty = core_type_to_mlir(catgrad_ty);
            grammar::TypedIdentifier { id, ty }
        })
        .collect::<Vec<_>>();

    let types = typed_ids.iter().map(|tid| tid.ty.clone()).collect();
    (grammar::Return(typed_ids), types)
}

pub fn get_mlir_body(term: Term) -> Vec<grammar::Statement> {
    let ops = ssa(term.to_strict()).expect("FIXME: unable to decompose input term");
    ops.iter().flat_map(to_statements).collect()
}

////////////////////////////////////////////////////////////////////////////////
// Map individual ops
// TODO: own module?
// TODO: generate a unique kernel for each <name, source, target> type?

/// Make a list of assignment statements from a single op
fn to_statements(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    // Get LHS of assignment and types
    let (result, return_types): (Vec<grammar::Identifier>, Vec<grammar::Type>) = ssa
        .targets
        .iter()
        .map(|(i, t)| (grammar::Identifier::Node(*i), core_type_to_mlir(t)))
        .unzip();

    let mut statements = vec![
        grammar::Statement::Custom(format!("// {:?}", ssa.op)),
        grammar::Statement::Custom(format!("// {:?}", ssa.sources)),
        grammar::Statement::Custom(format!("// {:?}", ssa.targets)),
    ];

    statements.extend(match &ssa.op {
        // Declarations lower to explicit snippets
        lang::Operation::Declaration(path) => lower_operation(path, ssa),

        // Definitions always lower to *func* calls.
        lang::Operation::Definition(path) => {
            let ins = ssa.sources.iter().map(to_typed_identifier).collect();

            let expr = grammar::Call {
                name: path.to_string(),
                args: ins,
                return_type: return_types,
            }
            .into();

            vec![grammar::Assignment { result, expr }.into()]
        }
        lang::Operation::Literal(lit) => literal_to_statements(ssa.edge_id, lit, result),
        // Do not emit probes
        lang::Operation::Probe(_label) => vec![],
    });

    statements
}

// This is an awful hack
fn as_floating(x: String) -> String {
    if x.contains(".") { x } else { format!("{x}.0") }
}

fn literal_to_statements(
    edge_id: EdgeId,
    lit: &lang::Literal,
    result: Vec<grammar::Identifier>,
) -> Vec<grammar::Statement> {
    match lit {
        lang::Literal::F32(x) => make_scalar_tensor_statements(
            edge_id,
            result[0].clone(),
            as_floating(x.to_string()),
            "f32".to_string(),
        ),
        lang::Literal::U32(x) => make_scalar_tensor_statements(
            edge_id,
            result[0].clone(),
            x.to_string(),
            "i32".to_string(),
        ),
        lang::Literal::Nat(x) => {
            let expr = grammar::Expr::Constant(grammar::Constant {
                name: "arith.constant".to_string(),
                value: Some(x.to_string()),
                ty: Some(grammar::Type::Index),
            });
            vec![grammar::Assignment { result, expr }.into()]
        }
        lang::Literal::Dtype(_) => {
            let expr = grammar::Expr::Constant(grammar::Constant {
                name: "arith.constant".to_string(),
                value: Some("false".to_string()),
                ty: None,
            });
            vec![grammar::Assignment { result, expr }.into()]
        }
    }
}

fn make_scalar_tensor_statements(
    edge_id: EdgeId,
    target_id: grammar::Identifier,
    value: String,
    dtype: String,
) -> Vec<grammar::Statement> {
    let base = grammar::Identifier::Edge(edge_id);
    let scalar_constant = grammar::Statement::Custom(format!(
        "  {base}_scalar = arith.constant {value} : {dtype}",
    ));

    let tensor_from_elements = grammar::Statement::Custom(format!(
        "  {target_id} = tensor.from_elements {base}_scalar : tensor<{dtype}>",
    ));

    vec![scalar_constant, tensor_from_elements]
}

fn lower_operation(path: &Path, ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    use super::ops;
    match path.to_string().as_str() {
        "cartesian.copy" => panic!("Copy ops must be forgotten before MLIR"),
        "tensor.shape" => ops::shape(ssa),
        "tensor.dtype" => vec![],
        "tensor.neg" => ops::neg(ssa).into_iter().map(Into::into).collect(),
        "tensor.broadcast" => ops::broadcast(ssa),
        "tensor.cast" => ops::cast(ssa).into_iter().map(Into::into).collect(),
        "tensor.add" => ops::add(ssa).into_iter().map(Into::into).collect(),
        "tensor.sub" => ops::sub(ssa).into_iter().map(Into::into).collect(),
        "tensor.mul" => ops::mul(ssa).into_iter().map(Into::into).collect(),
        "tensor.pow" => ops::pow(ssa).into_iter().map(Into::into).collect(),
        "tensor.div" => ops::div(ssa).into_iter().map(Into::into).collect(),
        "tensor.sin" => ops::sin(ssa).into_iter().map(Into::into).collect(),
        "tensor.cos" => ops::cos(ssa).into_iter().map(Into::into).collect(),
        "tensor.lt" => ops::lt(ssa),
        "tensor.arange" => ops::arange(ssa),
        "shape.pack" => ops::shape_pack(ssa).into_iter().map(Into::into).collect(),
        "shape.unpack" => ops::shape_unpack(ssa),
        "tensor.index" => ops::tensor_index(ssa),
        "tensor.transpose" => ops::tensor_transpose(ssa),
        "tensor.reshape" => ops::tensor_reshape(ssa),
        "tensor.sum" => ops::tensor_sum(ssa),
        "tensor.nat_to_u32" => ops::nat_to_u32(ssa),
        "tensor.concat" => ops::tensor_concat(ssa),
        "tensor.matmul" => ops::tensor_matmul(ssa),
        "tensor.slice" => ops::tensor_slice(ssa),
        "nat.mul" => ops::nat_mul(ssa),
        "nat.add" => ops::nat_add(ssa),
        "tensor.max" => ops::tensor_max(ssa),
        "tensor.argmax" => ops::tensor_argmax(ssa),
        op => todo!("unknown op {op:?}"),
    }
}
