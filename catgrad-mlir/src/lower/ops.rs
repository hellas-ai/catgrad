//! # Lower catgrad ops to MLIR
//
// TODO:
//
// - Intermediate variables by convention use MLIR SSA id {base}_{suffix}.
//   This uses source[0] as base, but *SHOULD* use edge_id which won't break when source[0] is
//   copied.
// - hardcoded 'index' type everywhere: should get this from code (in case we change runtime rep
//   of Nat)
// - max reduction uses neginf as initial value. This is not ideal, won't work with other dtypes.
// - matmul should support arbitrary ranked batches
// - arange, broadcast both use hacky boilerplate for generating dynamic dim args: fix?
use catgrad::ssa::SSA;
use catgrad::{
    category::lang,
    prelude::Dtype,
    typecheck::{DtypeExpr, NatExpr, NdArrayType, ShapeExpr, Type, TypeExpr},
};

use super::util::*;
use super::{grammar, grammar::Identifier};

// Represent a `Shape` as MLIR `tensor<rank x index>`.
//
// Generates assignments like
// ```
//  %shape_dim0 = tensor.dim %input, %c0 : tensor<?x?x?xf32>
//  %shape_dim1 = tensor.dim %input, %c1 : tensor<?x?x?xf32>
//  %shape_dim2 = tensor.dim %input, %c2 : tensor<?x?x?xf32>
//  %shape = tensor.from_elements %id_dim0, %id_dim1, %id_dim2 : tensor<3 x index>
// ```
//
// Where `shape` is the SSA node id of the result.
pub fn shape(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    // Extract the shape from the input tensor type
    let dims =
        require_known_shape(ssa.sources[0].1.clone()).expect("shape operation needs known shape");

    let rank = dims.len();

    let base = Identifier::Edge(ssa.edge_id);
    let source_ssa = Identifier::Node(ssa.sources[0].0);
    let target_ssa = Identifier::Node(ssa.targets[0].0);

    let source_type = core_type_to_mlir(&ssa.sources[0].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // List of statements to generate
    let mut statements = vec![];

    // Generate constant expressions
    // {base}_c{i} = arith.constant {i} : index
    statements.extend((0..rank).map(|i| format!("{base}_c{i} = arith.constant {i} : index")));

    // Generate assignments
    // {base}_d{i} = tensor.dim {}, %c0
    statements
        .extend((0..rank).map(|i| {
            format!("{base}_d{i} = tensor.dim {source_ssa}, {base}_c{i} : {source_type}")
        }));

    // Comma separated vars `{base}_d{i}`.
    let dim_vars = (0..rank)
        .map(|i| format!("{base}_d{i}"))
        .collect::<Vec<_>>()
        .join(", ");

    //  %shape = tensor.from_elements %v0_d0, %v0_d1, %v0_d2 : tensor<3 x index>
    statements.push(format!(
        "{target_ssa} = tensor.from_elements {dim_vars} : {target_type}"
    ));

    statements
        .into_iter()
        .map(grammar::Statement::Custom)
        .collect()
}

pub fn broadcast(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let in_shape = require_known_shape(ssa.sources[0].1.clone())
        .expect("broadcast operation needs known input shape");

    let out_shape = require_known_shape(ssa.targets[0].1.clone())
        .expect("broadcast operation needs known output shape");

    let base = Identifier::Edge(ssa.edge_id);
    let tensor_id = Identifier::Node(ssa.sources[0].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    // Generate indexing maps for broadcasting
    let out_rank = out_shape.len();

    // Make affine map for *broadcasted* tensor
    //      (d_0..d_{out_rank}) -> (trailing dims, with 1s replaced with literal 0)
    let input_dims: Vec<_> = in_shape
        .iter()
        .map(|nat| match nat {
            NatExpr::Constant(n) => Some(*n),
            _ => None,
        })
        .collect();

    let input_affine_map = make_affine_map(input_dims, out_rank);

    // make affine map for *output* tensor
    //      (d_0..d_{out_rank}) → (d_0..d_{out_rank})
    let output_affine_map = make_affine_map(vec![None; out_rank], out_rank);

    let attrs = format!(
        "indexing_maps = [{}, {}], iterator_types = [{}]",
        input_affine_map,
        output_affine_map,
        vec!["\"parallel\""; out_rank].join(", "),
    );

    let tensor_type = core_type_to_mlir(&ssa.sources[0].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // Extract element type for the linalg.generic operation
    let dtype = require_known_dtype(ssa.sources[0].1.clone())
        .expect("broadcast requires tensor with known dtype");
    let element_type = element_type_name(dtype);

    let mut statements = vec![];

    // Extract dimensions from the target shape tensor for dynamic dimensions
    let shape_id = Identifier::Node(ssa.sources[1].0); // The target shape tensor
    let mut dim_args = vec![];

    for (i, dim) in out_shape.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.extract {shape_id}[{base}_c{i}] : tensor<{}xindex>",
                out_shape.len()
            )));
            dim_args.push(format!("{base}_d{i}"));
        }
    }

    let empty_expr = if dim_args.is_empty() {
        format!("tensor.empty() : {target_type}")
    } else {
        format!("tensor.empty({}) : {target_type}", dim_args.join(", "))
    };

    statements.push(grammar::Statement::Custom(format!(
        "  {base}_out = {empty_expr}"
    )));
    statements.push(grammar::Statement::Custom(format!(
        r#"  {target_id} = linalg.generic {{{attrs}}} ins({tensor_id} : {tensor_type})
                                    outs({base}_out : {target_type}) {{
                                        ^bb0(%in: {element_type}, %out: {element_type}):
                                        linalg.yield %in : {element_type}
                                    }} -> {target_type}"#
    )));

    statements
}

fn make_affine_map(dims: Vec<Option<usize>>, out_rank: usize) -> String {
    // Assume dim names are d{i}
    let params: Vec<String> = (0..out_rank).map(|i| format!("d{}", i)).collect();

    // For broadcasting, input dimensions align with trailing output dimensions
    let input_rank = dims.len();
    let offset = out_rank - input_rank; // How many leading dims in output have no input correspondence

    let outputs: Vec<String> = dims
        .iter()
        .enumerate()
        .map(|(i, d)| match d {
            Some(1) => "0".to_string(), // Broadcasted dimension maps to constant 0
            _ => format!("d{}", i + offset), // Map to corresponding trailing output dimension
        })
        .collect();

    format!(
        "affine_map<({}) -> ({})>",
        params.join(", "),
        outputs.join(", ")
    )
}

pub fn shape_pack(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // Pack k Nats into a shape: Nat^k → Type
    // Creates a tensor<k x index> from k natural number inputs
    assert!(ssa.targets.len() == 1);

    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // Get the input identifiers (natural numbers)
    let input_ids: Vec<String> = ssa
        .sources
        .iter()
        .map(|(source_node, _)| format!("%v{}", source_node.0))
        .collect();

    vec![
        grammar::Expr::Custom(format!(
            "tensor.from_elements {} : {}",
            input_ids.join(", "),
            target_type
        ))
        .into_assignment(ssa),
    ]
}

// Shape → Nat .. Nat
// Example rendering:
// %c0 = arith.constant 0 : index
// %c1 = arith.constant 1 : index
// %c2 = arith.constant 2 : index
// %e0 = tensor.extract %t[%c0] : tensor<3xi32>
// %e1 = tensor.extract %t[%c1] : tensor<3xi32>
// %e2 = tensor.extract %t[%c2] : tensor<3xi32>
pub fn shape_unpack(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    let base = Identifier::Edge(ssa.edge_id);
    let x = &Identifier::Node(ssa.sources[0].0); // tensor<Nxi32>
    let x_type = core_type_to_mlir(&ssa.sources[0].1);

    let rank = ssa.targets.len();

    let mut statements = vec![];

    // Generate constant expressions
    // {base}_c{i} = arith.constant {i} : index
    statements.extend((0..rank).map(|i| format!("{base}_c{i} = arith.constant {i} : index")));

    // Generate assignments: one for each target!
    // {base}_d{i} = tensor.dim {base}, %c0
    statements.extend((0..rank).map(|i| {
        let target = Identifier::Node(ssa.targets[i].0);
        format!("{target} = tensor.extract {x}[{base}_c{i}] : {x_type}",)
    }));

    statements
        .into_iter()
        .map(grammar::Statement::Custom)
        .collect()
}

// Index : Sources × Dim × Indexes → Values
//
// Lowers to something like this (in this example, dim = 0)
// ```
// func.func @example(
//   %src: tensor<?x?x?xf32>,
//   %idx: tensor<?xindex>
// ) -> tensor<?x?x?xf32> {
//   // Constants for each dimension
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c2 = arith.constant 2 : index
//
//   // Size of indexes tensor
//   %n = tensor.dim %idx, %c0 : tensor<?xindex>
//
//   // Dims of input tensor
//   %d1 = tensor.dim %src, %c1 : tensor<?x?x?xf32>
//   %d2 = tensor.dim %src, %c2 : tensor<?x?x?xf32>
//
//   // Result tensor
//   %empty = tensor.empty(%n, %d1, %d2) : tensor<?x?x?xf32>
//
//   %res = linalg.generic
//     {
//       indexing_maps = [
//         affine_map<(d0, d1, d2) -> (d0)>,
//         affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//       ],
//       iterator_types = ["parallel", "parallel", "parallel"]
//     }
//     ins(%idx : tensor<?xindex>)
//     outs(%empty : tensor<?x?x?xf32>) {
//       ^bb0(%idx_elem: index, %out_elem: f32):
//         %idx_1 = linalg.index 1 : index
//         %idx_2 = linalg.index 2 : index
//         %val = tensor.extract %src[%idx_elem, %idx_1, %idx_2] : tensor<?x?x?xf32>
//         linalg.yield %val : f32
//     } -> tensor<?x?x?xf32>
//   return %res : tensor<?x?x?xf32>
// }
// ```
pub fn tensor_index(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 3);
    assert!(ssa.targets.len() == 1);

    // Verify that dim is statically known to be 0
    let Type::Nat(NatExpr::Constant(0)) = &ssa.sources[1].1 else {
        panic!("tensor.index currently only supports dimension 0")
    };

    let source_type = core_type_to_mlir(&ssa.sources[0].1);
    let index_type = core_type_to_mlir(&ssa.sources[2].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // Extract target shape to determine which dimensions are statically known
    let target_shape_dims =
        require_known_shape(ssa.targets[0].1.clone()).expect("Index operation needs known shape");

    // List of statements to be returned
    let mut statements = vec![];

    // Extract the rank from the source tensor type
    let shape_dims =
        require_known_shape(ssa.sources[0].1.clone()).expect("Index operation needs known shape");

    let rank = shape_dims.len();

    // Base name prefix used for intermediate variables that don't appear in the original hypergraph
    let source_ssa = Identifier::Node(ssa.sources[0].0);
    let index_ssa = Identifier::Node(ssa.sources[2].0);
    let base = Identifier::Edge(ssa.edge_id);
    let target_id = Identifier::Node(ssa.targets[0].0);

    // Generate constant statements for each dimension
    //  %c0 = arith.constant 0 : index
    for i in 0..rank {
        statements.push(grammar::Statement::Custom(format!(
            "  {base}_c{i} = arith.constant {i} : index"
        )));
    }

    // Generate size of indexes tensor
    //  %n = tensor.dim %idx, %c0 : tensor<?xindex>
    statements.push(grammar::Statement::Custom(format!(
        "  {base}_d0 = tensor.dim {index_ssa}, {base}_c0 : {index_type}",
    )));

    // Generate dimension extraction statements for dims 1..rank
    //  %d1 = tensor.dim %src, %c1 : tensor<?x?x?xf32>
    let src_id = Identifier::Node(ssa.sources[0].0);
    for i in 1..rank {
        statements.push(grammar::Statement::Custom(format!(
            "  {base}_d{i} = tensor.dim {source_ssa}, {base}_c{i} : {source_type}",
        )));
    }

    // Generate empty result tensor - only include dynamic dimensions
    let empty_expr = to_empty_expr(&base, &target_type, &target_shape_dims);

    // Generate affine map inputs: d0, d1, ..., d_{rank-1}
    let parallel_stmts = (0..rank)
        .map(|_| "\"parallel\"")
        .collect::<Vec<_>>()
        .join(", ");
    let affine_map_inputs = (0..rank)
        .map(|i| format!("d{i}"))
        .collect::<Vec<_>>()
        .join(", ");

    // Generate index variables: %idx_elem, %idx_1, %idx_2, ...
    let mut index_vars = vec!["%idx_elem_index".to_string()];
    for i in 1..rank {
        index_vars.push(format!("%idx_{i}"));
    }
    let index_vars_str = index_vars.join(", ");

    // Generate index assignment statements for dimensions 1..rank
    //  %idx_1 = linalg.index 1 : index
    let index_assignments = (1..rank)
        .map(|i| format!("              %idx_{i} = linalg.index {i} : index"))
        .collect::<Vec<_>>()
        .join("\n");

    let linalg_generic = grammar::Statement::Custom(format!(
        r#"
        {base}_empty = {empty_expr}
        {target_id} = linalg.generic
          {{
            indexing_maps = [
              affine_map<({affine_map_inputs}) -> (d0)>,                    // idx
              affine_map<({affine_map_inputs}) -> ({affine_map_inputs})>    // output
            ],
            iterator_types = [{parallel_stmts}]
          }}
          ins({index_ssa} : {index_type})
          outs({base}_empty : {target_type}) {{
            ^bb0(%idx_elem: i32, %out_elem: f32):
              %idx_elem_index = arith.index_cast %idx_elem : i32 to index
              {index_assignments}
              %val = tensor.extract {src_id}[{index_vars_str}] : {source_type}
              linalg.yield %val : f32
          }} -> {target_type}
    "#
    ));

    statements.push(linalg_generic);
    statements
}

// Transpose : Tensor × Nat × Nat → Tensor
// Transposes two dimensions (assumed to be statically known as NatExpr::Constant) in the input tensor.
// Example:
// ```
// %transposed = linalg.transpose ins(%0 : tensor<?x?x?x?xf32>)
//                                outs(%init : tensor<?x?x?x?xf32>)
//                                permutation = [0, 2, 1, 3]
// ```
pub fn tensor_transpose(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 3);
    assert!(ssa.targets.len() == 1);

    // Extract the two dimension indices (must be constants)
    let Type::Nat(NatExpr::Constant(dim1)) = &ssa.sources[1].1 else {
        panic!("transpose dim1 must be a constant")
    };
    let Type::Nat(NatExpr::Constant(dim2)) = &ssa.sources[2].1 else {
        panic!("transpose dim2 must be a constant")
    };

    // Get tensor type info
    let shape_dims = require_known_shape(ssa.sources[0].1.clone())
        .expect("Transpose operation needs known shape");

    let rank = shape_dims.len();

    let source_type = core_type_to_mlir(&ssa.sources[0].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    let tensor_id = Identifier::Node(ssa.sources[0].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    // Create permutation array: identity with dim1 and dim2 swapped
    let mut permutation: Vec<usize> = (0..rank).collect();
    permutation.swap(*dim1, *dim2);

    // Get target shape for empty tensor creation
    let mut target_shape_dims = require_known_shape(ssa.targets[0].1.clone())
        .expect("Transpose operation needs known shape");
    target_shape_dims.swap(*dim1, *dim2);

    let mut statements = vec![];

    // Generate dimension extraction for dynamic dimensions
    let base = Identifier::Edge(ssa.edge_id);
    for (i, dim) in shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            // This is a dynamic dimension, extract it
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.dim {tensor_id}, {base}_c{i} : {source_type}"
            )));
        }
    }

    // Create empty tensor for the result
    let empty_expr = to_empty_expr(&base, &target_type, &target_shape_dims);

    // Generate the transpose operation
    let permutation_str = permutation
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let transpose_stmt = grammar::Statement::Custom(format!(
        "  {base}_empty = {empty_expr}\n  {target_id} = linalg.transpose ins({tensor_id} : {source_type}) outs({base}_empty : {target_type}) permutation = [{permutation_str}]"
    ));

    statements.push(transpose_stmt);
    statements
}

// Reshape : Shape × Tensor → Tensor
// Reshape(s, x) -> y
//
//%reshaped = tensor.reshape %input(%shape) : (tensor<3x4xf32>, tensor<2xindex>) -> tensor<2x6xf32>
pub fn tensor_reshape(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    let s = &Identifier::Node(ssa.sources[0].0);
    let x = &Identifier::Node(ssa.sources[1].0);
    let y = &Identifier::Node(ssa.targets[0].0);

    let s_type = core_type_to_mlir(&ssa.sources[0].1);
    let x_type = core_type_to_mlir(&ssa.sources[1].1);
    let y_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![grammar::Statement::Custom(format!(
        "{y} = tensor.reshape {x}({s}) : ({x_type}, {s_type}) -> {y_type}"
    ))]
}

// Sum : Tensor → Tensor
// Sums over the final dimension of the tensor.
// Example:
// Input: tensor<2x3x4xf32> -> Output: tensor<2x3x1xf32>
//
// Generates MLIR like:
// %empty = tensor.empty() : tensor<2x3xf32>
// %c0 = arith.constant 0.0 : f32
// %fill = linalg.fill ins(%c0 : f32) outs(%empty : tensor<2x3xf32>) -> tensor<2x3xf32>
// %sum = linalg.generic {
//   indexing_maps = [
//     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,  // input
//     affine_map<(d0, d1, d2) -> (d0, d1)>       // output (reduced d2)
//   ],
//   iterator_types = ["parallel", "parallel", "reduction"]
// } ins(%input : tensor<2x3x4xf32>) outs(%fill : tensor<2x3xf32>) {
//   ^bb0(%in: f32, %out: f32):
//     %add = arith.addf %in, %out : f32
//     linalg.yield %add : f32
// } -> tensor<2x3xf32>
pub fn tensor_sum(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    // Get input and output types
    let input_shape = require_known_shape(ssa.sources[0].1.clone())
        .expect("Sum operation needs known input shape");
    let output_shape = require_known_shape(ssa.targets[0].1.clone())
        .expect("Sum operation needs known output shape");

    let rank = input_shape.len();
    let output_rank = output_shape.len();

    // Verify that output rank is same as input rank (catgrad reductions keep rank)
    assert_eq!(
        output_rank, rank,
        "Sum should keep same rank, reducing final dimension to size 1"
    );

    // Verify final dimension is reduced to 1
    if let NatExpr::Constant(final_dim_size) = &output_shape[rank - 1] {
        assert_eq!(
            *final_dim_size, 1,
            "Sum should reduce final dimension to size 1"
        );
    }

    let input_type = core_type_to_mlir(&ssa.sources[0].1);
    let output_type = core_type_to_mlir(&ssa.targets[0].1);

    let input_id = Identifier::Node(ssa.sources[0].0);
    let target_id = Identifier::Node(ssa.targets[0].0);
    let base = Identifier::Edge(ssa.edge_id);

    let mut statements = vec![];

    // Generate dimension extraction for dynamic dimensions in output
    for (i, dim) in output_shape.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.dim {input_id}, {base}_c{i} : {input_type}"
            )));
        }
    }

    // Create empty tensor for the result
    let empty_expr = to_empty_expr(&base, &output_type, &output_shape);

    // Create zero constant for initialization
    statements.push(grammar::Statement::Custom(format!(
        "  {base}_zero = arith.constant 0.0 : f32"
    )));

    // Fill the empty tensor with zeros
    statements.push(grammar::Statement::Custom(format!(
        "  {base}_empty = {empty_expr}"
    )));

    statements.push(grammar::Statement::Custom(format!(
        "  {base}_fill = linalg.fill ins({base}_zero : f32) outs({base}_empty : {output_type}) -> {output_type}"
    )));

    // Generate indexing maps
    let input_dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();

    // Output map: all dims except final are the same, final dim maps to 0 (reduced)
    let mut output_dims: Vec<String> = (0..(rank - 1)).map(|i| format!("d{i}")).collect();
    output_dims.push("0".to_string()); // Final dimension reduced to constant 0

    let input_map = format!(
        "affine_map<({}) -> ({})>",
        input_dims.join(", "),
        input_dims.join(", ")
    );
    let output_map = format!(
        "affine_map<({}) -> ({})>",
        input_dims.join(", "),
        output_dims.join(", ")
    );

    // Generate iterator types: parallel for all dims except last which is reduction
    let mut iterator_types = vec!["\"parallel\""; rank - 1];
    iterator_types.push("\"reduction\"");
    let iterator_types_str = iterator_types.join(", ");

    // Generate the linalg.generic operation for reduction
    let linalg_stmt = grammar::Statement::Custom(format!(
        r#"  {target_id} = linalg.generic {{
    indexing_maps = [{input_map}, {output_map}],
    iterator_types = [{iterator_types_str}]
  }} ins({input_id} : {input_type}) outs({base}_fill : {output_type}) {{
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  }} -> {output_type}"#
    ));

    statements.push(linalg_stmt);
    statements
}

// NatToU32 : Nat → Tensor ()
// Converts a Nat value into a scalar tensor.
// Example:
// Input: %nat (index type) -> Output: tensor<i32> (scalar tensor)
//
// Generates MLIR like:
// %cast = arith.index_cast %nat : index to i32
// %scalar = tensor.from_elements %cast : tensor<i32>
pub fn nat_to_u32(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    let base = Identifier::Edge(ssa.edge_id);
    let nat_id = Identifier::Node(ssa.sources[0].0);
    let target_id = Identifier::Node(ssa.targets[0].0);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![
        // Cast from index to i32
        grammar::Statement::Custom(format!(
            "  {base}_cast = arith.index_cast {nat_id} : index to i32"
        )),
        // Create scalar tensor from the cast value
        grammar::Statement::Custom(format!(
            "  {target_id} = tensor.from_elements {base}_cast : {target_type}"
        )),
    ]
}

// NatMul : Nat × Nat → Nat
// Multiplies two natural numbers.
// Example:
// Input: %a, %b (index types) -> Output: %result (index type)
//
// Generates MLIR like:
// %result = arith.muli %a, %b : index
pub fn nat_mul(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let a_id = Identifier::Node(ssa.sources[0].0);
    let b_id = Identifier::Node(ssa.sources[1].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    vec![grammar::Statement::Custom(format!(
        "  {target_id} = arith.muli {a_id}, {b_id} : index"
    ))]
}

// NatAdd : Nat × Nat → Nat
// Adds two natural numbers.
// Example:
// Input: %a, %b (index types) -> Output: %result (index type)
//
// Generates MLIR like:
// %result = arith.addi %a, %b : index
pub fn nat_add(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let a_id = Identifier::Node(ssa.sources[0].0);
    let b_id = Identifier::Node(ssa.sources[1].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    vec![grammar::Statement::Custom(format!(
        "  {target_id} = arith.addi {a_id}, {b_id} : index"
    ))]
}

// Max : Tensor → Tensor
// Finds maximum over the final dimension of the tensor.
// Example:
// Input: tensor<2x3x4xf32> -> Output: tensor<2x3x1xf32>
//
// Generates MLIR similar to sum but with max operation
pub fn tensor_max(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    let input_shape = require_known_shape(ssa.sources[0].1.clone())
        .expect("Max operation needs known input shape");
    let output_shape = require_known_shape(ssa.targets[0].1.clone())
        .expect("Max operation needs known output shape");

    let rank = input_shape.len();
    let output_rank = output_shape.len();

    assert_eq!(
        output_rank, rank,
        "Max should keep same rank, reducing final dimension to size 1"
    );

    if let NatExpr::Constant(final_dim_size) = &output_shape[rank - 1] {
        assert_eq!(
            *final_dim_size, 1,
            "Max should reduce final dimension to size 1"
        );
    }

    let input_type = core_type_to_mlir(&ssa.sources[0].1);
    let output_type = core_type_to_mlir(&ssa.targets[0].1);

    let input_id = Identifier::Node(ssa.sources[0].0);
    let target_id = Identifier::Node(ssa.targets[0].0);
    let base = Identifier::Edge(ssa.edge_id);

    let mut statements = vec![];

    // Generate dimension extraction for dynamic dimensions in output
    for (i, dim) in output_shape.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.dim {input_id}, {base}_c{i} : {input_type}"
            )));
        }
    }

    // Create empty tensor for the result
    let empty_expr = to_empty_expr(&base, &output_type, &output_shape);

    // Create negative infinity constant for initialization
    statements.push(grammar::Statement::Custom(format!(
        "  {base}_ninf = arith.constant -3.40282347e+38 : f32"
    )));

    statements.push(grammar::Statement::Custom(format!(
        "  {base}_empty = {empty_expr}"
    )));

    statements.push(grammar::Statement::Custom(format!(
        "  {base}_fill = linalg.fill ins({base}_ninf : f32) outs({base}_empty : {output_type}) -> {output_type}"
    )));

    // Generate indexing maps (same as sum)
    let input_dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let mut output_dims: Vec<String> = (0..(rank - 1)).map(|i| format!("d{i}")).collect();
    output_dims.push("0".to_string());

    let input_map = format!(
        "affine_map<({}) -> ({})>",
        input_dims.join(", "),
        input_dims.join(", ")
    );
    let output_map = format!(
        "affine_map<({}) -> ({})>",
        input_dims.join(", "),
        output_dims.join(", ")
    );

    let mut iterator_types = vec!["\"parallel\""; rank - 1];
    iterator_types.push("\"reduction\"");
    let iterator_types_str = iterator_types.join(", ");

    // Generate the linalg.generic operation for max reduction
    let linalg_stmt = grammar::Statement::Custom(format!(
        r#"  {target_id} = linalg.generic {{
    indexing_maps = [{input_map}, {output_map}],
    iterator_types = [{iterator_types_str}]
  }} ins({input_id} : {input_type}) outs({base}_fill : {output_type}) {{
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %out : f32
    linalg.yield %max : f32
  }} -> {output_type}"#
    ));

    statements.push(linalg_stmt);
    statements
}

// ArgMax : Tensor → Tensor
// Finds indices of maximum values over the final dimension of the tensor.
// Example:
//
// ```
//  %c0 = arith.constant 0 : index
//  %min_val = arith.constant 0xFF800000 : f32  // -inf
//
//  %indices_buf = tensor.empty() : tensor<4xindex>
//  %max_buf = tensor.empty() : tensor<4xf32>
//
//  %init_indices = linalg.fill ins(%c0 : index) outs(%indices_buf : tensor<4xindex>) -> tensor<4xindex>
//  %init_maxes = linalg.fill ins(%min_val : f32) outs(%max_buf : tensor<4xf32>) -> tensor<4xf32>
//
//  %results:2 = linalg.generic {
//    indexing_maps = [
//      affine_map<(d0, d1) -> (d0, d1)>,  // input
//      affine_map<(d0, d1) -> (d0)>,       // output indices
//      affine_map<(d0, d1) -> (d0)>        // output max values
//    ],
//    iterator_types = ["parallel", "reduction"]
//  }
//  ins(%x : tensor<4x8xf32>)
//  outs(%init_indices, %init_maxes : tensor<4xindex>, tensor<4xf32>) {
//  ^bb0(%in: f32, %out_idx: index, %out_max: f32):
//    %curr_idx = linalg.index 1 : index
//    %cmp = arith.cmpf ogt, %in, %out_max : f32
//    %new_max = arith.select %cmp, %in, %out_max : f32
//    %new_idx = arith.select %cmp, %curr_idx, %out_idx : index
//    linalg.yield %new_idx, %new_max : index, f32
//  } -> (tensor<4xindex>, tensor<4xf32>)
//
//  return %results#0 : tensor<4xindex>
// ```
pub fn tensor_argmax(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    let rank = require_known_shape(ssa.sources[0].1.clone())
        .expect("ArgMax operation needs known input shape")
        .len();

    let x = Identifier::Node(ssa.sources[0].0);
    let y = Identifier::Node(ssa.targets[0].0);
    let base = Identifier::Edge(ssa.edge_id);

    let x_type = core_type_to_mlir(&ssa.sources[0].1);
    let y_type = core_type_to_mlir(&ssa.targets[0].1);

    let mut statements: Vec<String> = vec![];

    // Generate input/output affine maps
    let input_dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let mut output_dims: Vec<String> = (0..(rank - 1)).map(|i| format!("d{i}")).collect();
    output_dims.push("0".to_string()); // Final dimension reduced to constant 0

    let input_map = format!(
        "affine_map<({}) -> ({})>",
        input_dims.join(", "),
        input_dims.join(", ")
    );
    let output_map = format!(
        "affine_map<({}) -> ({})>",
        input_dims.join(", "),
        output_dims.join(", ")
    );

    // Generate iterator types: parallel for all dims except last which is reduction
    let mut iterator_types = vec!["\"parallel\""; rank - 1];
    iterator_types.push("\"reduction\"");
    let iterator_types_str = iterator_types.join(", ");

    // Create helper tensor for tracking max values during reduction
    let max_buf_type = y_type.clone().with_dtype("f32".to_string());
    let indices_buf_type = y_type.with_dtype("i32".to_string());

    // Extract dimensions for dynamic empty tensor creation
    let target_shape_dims = require_known_shape(ssa.targets[0].1.clone())
        .expect("ArgMax operation needs known target shape");

    let mut dim_args = vec![];
    for (i, dim) in target_shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            statements.push(format!("{base}_c{i} = arith.constant {i} : index"));
            statements.push(format!(
                "{base}_d{i} = tensor.dim {x}, {base}_c{i} : {x_type}"
            ));
            dim_args.push(format!("{base}_d{i}"));
        }
    }

    let empty_args = if dim_args.is_empty() {
        "()".to_string()
    } else {
        format!("({})", dim_args.join(", "))
    };

    // Constants and initialization
    statements.extend([
        format!("{base}_c0 = arith.constant 0 : i32"),
        format!("{base}_min_val = arith.constant 0xFF800000 : f32"),
        format!("{base}_indices_buf = tensor.empty{empty_args} : {indices_buf_type}"),
        format!("{base}_max_buf = tensor.empty{empty_args} : {max_buf_type}"),
        format!("{base}_init_indices = linalg.fill ins({base}_c0 : i32) outs({base}_indices_buf : {indices_buf_type}) -> {indices_buf_type}"),
        format!("{base}_init_maxes = linalg.fill ins({base}_min_val : f32) outs({base}_max_buf : {max_buf_type}) -> {max_buf_type}"),
    ]);

    let linalg_stmt = format!(
        r#"{base}_results:2 = linalg.generic {{
          indexing_maps = [
            {input_map},
            {output_map},
            {output_map}
          ],
          iterator_types = [{iterator_types_str}]
        }}
        ins({x} : {x_type})
        outs({base}_init_indices, {base}_init_maxes : {indices_buf_type}, {max_buf_type}) {{
        ^bb0(%in: f32, %out_idx: i32, %out_max: f32):
          %curr_idx = linalg.index {last_dim} : index
          %cmp = arith.cmpf ogt, %in, %out_max : f32
          %new_max = arith.select %cmp, %in, %out_max : f32
          %out_idx_index = arith.index_cast %out_idx: i32 to index
          %new_idx = arith.select %cmp, %curr_idx, %out_idx_index : index
          %new_idx_i32 = arith.index_cast %new_idx : index to i32
          linalg.yield %new_idx_i32, %new_max : i32, f32
        }} -> ({indices_buf_type}, {max_buf_type})"#,
        last_dim = rank - 1
    );

    statements.push(linalg_stmt);
    statements.extend([
        format!("{base}_result = tensor.empty{empty_args} : {indices_buf_type}"),
        format!("{y} = linalg.copy ins({base}_results#0 : {indices_buf_type}) outs({base}_result: {indices_buf_type}) -> {indices_buf_type}"),
    ]);

    statements
        .into_iter()
        .map(grammar::Statement::Custom)
        .collect()
}

// Concat : Tensor × Tensor × Dim → Tensor
// Concatenates two tensors along a specified dimension.
// Example:
// Input: tensor<2x3xf32>, tensor<2x4xf32>, dim=1 -> Output: tensor<2x7xf32>
//
// Generates MLIR like:
// %result = tensor.concat dim(1) %tensor1, %tensor2 : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
pub fn tensor_concat(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 3);
    assert!(ssa.targets.len() == 1);

    // Sources are: [tensor1, tensor2, dim]
    let tensor1_id = Identifier::Node(ssa.sources[0].0);
    let tensor2_id = Identifier::Node(ssa.sources[1].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    // Extract the dimension to concatenate along
    let dim =
        require_known_nat(ssa.sources[2].1.clone()).expect("concat dimension must be constant");

    let tensor1_type = core_type_to_mlir(&ssa.sources[0].1);
    let tensor2_type = core_type_to_mlir(&ssa.sources[1].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![grammar::Statement::Custom(format!(
        "  {target_id} = tensor.concat dim({dim}) {tensor1_id}, {tensor2_id} : ({tensor1_type}, {tensor2_type}) -> {target_type}"
    ))]
}

// TODO: matmul needs to support arbitrary batch ranks
// MatMul : (N, A, B) × (N, B, C) → (N, A, C)
// Batched matrix multiplication operation.
// Example:
// Input: tensor<4x3x5xf32>, tensor<4x5x7xf32> -> Output: tensor<4x3x7xf32>
//
// Generates MLIR like:
// %empty = tensor.empty() : tensor<4x3x7xf32>
// %result = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x3x5xf32>, tensor<4x5x7xf32>)
//                             outs(%empty : tensor<4x3x7xf32>) -> tensor<4x3x7xf32>
pub fn tensor_matmul(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let base = Identifier::Edge(ssa.edge_id);
    let lhs_id = Identifier::Node(ssa.sources[0].0);
    let rhs_id = Identifier::Node(ssa.sources[1].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    let lhs_type = core_type_to_mlir(&ssa.sources[0].1);
    let rhs_type = core_type_to_mlir(&ssa.sources[1].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // Extract element type
    let dtype = require_known_dtype(ssa.sources[0].1.clone())
        .expect("matmul requires tensor with known dtype");
    let element_type = element_type_name(dtype);

    // Get shapes
    let target_shape_dims = require_known_shape(ssa.targets[0].1.clone())
        .expect("MatMul operation needs known target shape");

    let rank = target_shape_dims.len();
    let mut statements = vec![];

    // Generate dimension extraction for dynamic dimensions in target
    for (i, dim) in target_shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.dim {lhs_id}, {base}_c{i} : {lhs_type}"
            )));
        }
    }

    // Create empty tensor for the result
    let empty_expr = to_empty_expr(&base, &target_type, &target_shape_dims);

    statements.push(grammar::Statement::Custom(format!(
        "  {base}_empty = {empty_expr}"
    )));

    // Generate indexing maps for matmul
    // We need (rank + 1) dimensions: batch_dims + M + N + K
    // For input A: (...batch_dims, M, K)
    // For input B: (...batch_dims, K, N)
    // For output: (...batch_dims, M, N)
    let total_dims = rank + 1; // Add one for the K dimension
    let batch_dims: Vec<String> = (0..rank - 2).map(|i| format!("d{i}")).collect();
    let m_dim = format!("d{}", rank - 2);
    let n_dim = format!("d{}", rank - 1);
    let k_dim = format!("d{}", rank); // K is the extra reduction dimension

    let all_dims: Vec<String> = (0..total_dims).map(|i| format!("d{i}")).collect();

    let lhs_map = format!(
        "affine_map<({}) -> ({})>",
        all_dims.join(", "),
        [batch_dims.clone(), vec![m_dim.clone(), k_dim.clone()]]
            .concat()
            .join(", ")
    );
    let rhs_map = format!(
        "affine_map<({}) -> ({})>",
        all_dims.join(", "),
        [batch_dims.clone(), vec![k_dim, n_dim.clone()]]
            .concat()
            .join(", ")
    );
    let out_map = format!(
        "affine_map<({}) -> ({})>",
        all_dims.join(", "),
        [batch_dims, vec![m_dim, n_dim]].concat().join(", ")
    );

    let iterator_types = [
        vec!["\"parallel\""; rank - 2], // batch dimensions
        vec!["\"parallel\"", "\"parallel\"", "\"reduction\""], // M, N, K
    ]
    .concat()
    .join(", ");

    // Generate linalg.generic for matmul
    statements.push(grammar::Statement::Custom(format!(
        r#"  {target_id} = linalg.generic {{
    indexing_maps = [{lhs_map}, {rhs_map}, {out_map}],
    iterator_types = [{iterator_types}]
  }} ins({lhs_id}, {rhs_id} : {lhs_type}, {rhs_type}) outs({base}_empty : {target_type}) {{
  ^bb0(%a: {element_type}, %b: {element_type}, %acc: {element_type}):
    %prod = arith.mulf %a, %b : {element_type}
    %sum = arith.addf %acc, %prod : {element_type}
    linalg.yield %sum : {element_type}
  }} -> {target_type}"#
    )));

    statements
}

// Slice : Tensor × Dim × Start × Len → Tensor
// Slices a tensor along a specified dimension.
// Example:
// ```
// %c0 = arith.constant 0 : index
// %c1 = arith.constant 1 : index
//
// // dim₀ = size along dimension 0 (kept fully)
// %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
//
// // syntax: [offsets] [sizes] [strides]
// %slice_dyn = tensor.extract_slice %input[0, %start] [%dim0, %len] [1, 1]
//            : tensor<?x?xf32> to tensor<?x?xf32>
// // last step recovers any lost static information
// %slice = tensor.cast %v4_dyn : tensor<?x?x?xf32> to tensor<2x3x4xf32>
// ```
pub fn tensor_slice(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    let base = Identifier::Edge(ssa.edge_id);
    let x = Identifier::Node(ssa.sources[0].0);
    let x_type = core_type_to_mlir(&ssa.sources[0].1);

    let y = Identifier::Node(ssa.targets[0].0);
    let y_type = core_type_to_mlir(&ssa.targets[0].1);

    let start = Identifier::Node(ssa.sources[2].0);
    let len = Identifier::Node(ssa.sources[3].0);

    let rank = require_known_shape(ssa.sources[0].1.clone())
        .expect("slice op requires known rank")
        .len();

    let mut statements = vec![];

    // Generate constants 0..rank-1
    statements.extend((0..rank).map(|i| format!("{base}_c{i} = arith.constant {i} : index")));

    // Get all dimensions of input variable.
    statements
        .extend((0..rank).map(|i| format!("{base}_d{i} = tensor.dim {x}, {base}_c{i} : {x_type}")));

    // TODO: we can do non-static dim, but requires an additional block of vars assigned:
    // offset_i = if i == dim then start else 0
    let static_dim =
        require_known_nat(ssa.sources[1].1.clone()).expect("slice requires dim statically known");

    // 0, 0, ..., start, ..., 0
    let offset_vars = (0..rank)
        .map(|i| {
            if i == static_dim {
                format!("{start}")
            } else {
                format!("{base}_c{i}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    // d0, d1, ..., len, .., dn
    let size_vars = (0..rank)
        .map(|i| {
            if i == static_dim {
                format!("{len}")
            } else {
                format!("{base}_d{i}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    // Repeat 1s
    let strides = vec![format!("{base}_c1"); rank].join(",");

    // Dynamically typed result
    let y_type_dyn = y_type.clone().into_dynamic(); // same as y_type, but with all dynamic

    statements.extend([
        format!("{base}_dyn = tensor.extract_slice {x} [{offset_vars}] [{size_vars}] [{strides}] : {x_type} to {y_type_dyn}"),
        format!("{y} = tensor.cast {base}_dyn : {y_type_dyn} to {y_type}"),
    ]);

    statements
        .into_iter()
        .map(grammar::Statement::Custom)
        .collect()
}

////////////////////////////////////////////////////////////////////////////////
// NOTE: below here are essentially unfinished!

// TODO: FIXME: this just uses arith.sitofp, which is not always correct
pub fn cast(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let tensor_id = Identifier::Node(ssa.sources[0].0);
    let source_type = core_type_to_mlir(&ssa.sources[0].1);

    // Extract dtype from second source
    let Type::Dtype(catgrad::typecheck::DtypeExpr::Constant(target_dtype)) = &ssa.sources[1].1
    else {
        panic!("cast dtype must be a constant")
    };

    let source_dtype =
        require_known_dtype(ssa.sources[0].1.clone()).expect("cast must have known dtype");
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    let op_name = match (source_dtype, target_dtype) {
        (lang::Dtype::F32, lang::Dtype::F32) => panic!("Invalid cast F32 → F32"),
        (lang::Dtype::F32, lang::Dtype::U32) => todo!("Cast F32 → U32"),
        (lang::Dtype::U32, lang::Dtype::F32) => "arith.uitofp",
        (lang::Dtype::U32, lang::Dtype::U32) => panic!("Invalid cast U32 → U32"),
    };

    vec![
        grammar::Expr::Custom(format!(
            "{} {} : {} to {}",
            op_name, tensor_id, source_type, target_type
        ))
        .into_assignment(ssa),
    ]
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise ops

fn elementwise(op_name: &str, ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    let operands: Vec<grammar::Identifier> = ssa
        .sources
        .iter()
        .map(|(source_node, _)| Identifier::Node(*source_node))
        .collect();
    let result_type = core_type_to_mlir(&ssa.targets[0].1);

    vec![
        grammar::Expr::Elementwise(grammar::Elementwise {
            name: op_name.to_string(),
            operands,
            ty: result_type,
        })
        .into_assignment(ssa),
    ]
}

pub fn neg(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    // typechecked ssa should never break this invariant
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.negf", ssa)
}

pub fn add(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.addf", ssa)
}

pub fn sub(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.subf", ssa)
}

pub fn mul(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.mulf", ssa)
}

pub fn pow(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("math.powf", ssa)
}

pub fn div(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);
    elementwise("arith.divf", ssa)
}

pub fn cos(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    elementwise("math.cos", ssa)
}

pub fn sin(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Assignment> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);
    elementwise("math.sin", ssa)
}

// TODO: support non-f32 dtypes
// `LT : Tensor × Tensor → Tensor`
//
// Example:
//
// ```
// %init = tensor.empty() : tensor<2x3xf32>
// %result = linalg.generic {
//     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//                      affine_map<(d0, d1) -> (d0, d1)>,
//                      affine_map<(d0, d1) -> (d0, d1)>],
//     iterator_types = ["parallel", "parallel"]
// } ins(%v0, %v1 : tensor<2x3xf32>, tensor<2x3xf32>)
//   outs(%init : tensor<2x3xf32>) {
//     ^bb0(%a: f32, %b: f32, %out: f32):
//         %cmp = arith.cmpf olt, %a, %b : f32
//         %c0 = arith.constant 0.0 : f32
//         %c1 = arith.constant 1.0 : f32
//         %sel = arith.select %cmp, %c1, %c0 : f32
//         linalg.yield %sel : f32
// } -> tensor<2x3xf32>
// ```
pub fn lt(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 2);
    assert!(ssa.targets.len() == 1);

    let base = Identifier::Edge(ssa.edge_id);
    let lhs_id = Identifier::Node(ssa.sources[0].0);
    let rhs_id = Identifier::Node(ssa.sources[1].0);
    let target_id = Identifier::Node(ssa.targets[0].0);

    let lhs_type = core_type_to_mlir(&ssa.sources[0].1);
    let rhs_type = core_type_to_mlir(&ssa.sources[1].1);
    let target_type = core_type_to_mlir(&ssa.targets[0].1);

    // Extract element type for the linalg.generic operation
    let dtype =
        require_known_dtype(ssa.sources[0].1.clone()).expect("lt requires tensor with known dtype");
    let element_type = element_type_name(dtype.clone());

    // Get target shape for empty tensor creation
    let target_shape_dims = require_known_shape(ssa.targets[0].1.clone())
        .expect("Lt operation needs known target shape");

    let rank = target_shape_dims.len();
    let mut statements = vec![];

    // Generate dimension extraction for dynamic dimensions
    for (i, dim) in target_shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_c{i} = arith.constant {i} : index"
            )));
            statements.push(grammar::Statement::Custom(format!(
                "  {base}_d{i} = tensor.dim {lhs_id}, {base}_c{i} : {lhs_type}"
            )));
        }
    }

    // Create empty tensor for the result
    let mut dim_args = vec![];
    for (i, dim) in target_shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            dim_args.push(format!("{base}_d{i}"));
        }
    }
    let empty_expr = format!("tensor.empty({}) : {target_type}", dim_args.join(", "));

    statements.push(grammar::Statement::Custom(format!(
        "  {base}_init = {empty_expr}"
    )));

    // Generate indexing maps for elementwise operation
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let affine_map = format!("affine_map<({}) -> ({})>", dims.join(", "), dims.join(", "));
    let iterator_types = vec!["\"parallel\""; rank].join(", ");

    // Generate the linalg.generic operation for comparison
    let (cmp_op, zero_val, one_val) = match dtype {
        Dtype::F32 => ("arith.cmpf olt", "0.0", "1.0"),
        Dtype::U32 => ("arith.cmpi ult", "0", "1"),
    };

    let linalg_stmt = grammar::Statement::Custom(format!(
        r#"  {target_id} = linalg.generic {{
    indexing_maps = [{affine_map}, {affine_map}, {affine_map}],
    iterator_types = [{iterator_types}]
  }} ins({lhs_id}, {rhs_id} : {lhs_type}, {rhs_type}) outs({base}_init : {target_type}) {{
  ^bb0(%a: {element_type}, %b: {element_type}, %out: {element_type}):
    %cmp = {cmp_op}, %a, %b : {element_type}
    %c0 = arith.constant {zero_val} : {element_type}
    %c1 = arith.constant {one_val} : {element_type}
    %sel = arith.select %cmp, %c1, %c0 : {element_type}
    linalg.yield %sel : {element_type}
  }} -> {target_type}"#
    ));

    statements.push(linalg_stmt);
    statements
}

pub fn arange(ssa: &SSA<Type, lang::Operation>) -> Vec<grammar::Statement> {
    assert!(ssa.sources.len() == 1);
    assert!(ssa.targets.len() == 1);

    let target_type = core_type_to_mlir(&ssa.targets[0].1);
    let target_shape_dims = require_known_shape(ssa.targets[0].1.clone())
        .expect("Arange operation needs known target shape");

    let end_id = Identifier::Node(ssa.sources[0].0); // The 'end' parameter
    let target_id = Identifier::Node(ssa.targets[0].0);

    let mut statements = vec![];
    let mut dim_args = vec![];

    // For arange, the size should come from the 'end' parameter (source[0])
    for (i, dim) in target_shape_dims.iter().enumerate() {
        if !matches!(dim, NatExpr::Constant(_)) {
            // For arange, typically only the first dimension is dynamic and equals the 'end' parameter
            if i == 0 {
                dim_args.push(end_id.to_string());
            } else {
                // If other dimensions are dynamic, we'd need more complex logic
                panic!("Arange with dynamic dimensions beyond the first is not supported");
            }
        }
    }

    let generate_expr = if dim_args.is_empty() {
        format!(
            "tensor.generate {{\n^bb0(%i0: index):\n  %idx = arith.index_cast %i0 : index to i32\n  tensor.yield %idx : i32\n}} : {target_type}"
        )
    } else {
        format!(
            "tensor.generate {} {{\n^bb0(%i0: index):\n  %idx = arith.index_cast %i0 : index to i32\n  tensor.yield %idx : i32\n}} : {target_type}",
            dim_args.join(", ")
        )
    };

    statements.push(grammar::Statement::Custom(format!(
        "  {target_id} = {generate_expr}"
    )));
    statements
}

////////////////////////////////////////////////////////////////////////////////
// Utils

fn require_known_shape(t: Type) -> Option<Vec<NatExpr>> {
    // Extract the shape from the input tensor type
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        shape: ShapeExpr::Shape(dims),
        ..
    })) = t
    else {
        return None;
    };
    Some(dims)
}

fn require_known_dtype(t: Type) -> Option<Dtype> {
    // Extract the shape from the input tensor type
    let Type::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        ..
    })) = t
    else {
        return None;
    };
    Some(dtype)
}

fn require_known_nat(t: Type) -> Option<usize> {
    let Type::Nat(NatExpr::Constant(n)) = t else {
        return None;
    };
    Some(n)
}

/// Generate a `tensor.empty` expression from a list of NatExpr representing the dims.
fn to_empty_expr(
    base: &grammar::Identifier,
    target_type: &grammar::Type,
    dims: &[NatExpr],
) -> String {
    let rank = dims.len();

    let is_known_dimension: Vec<bool> = dims
        .iter()
        .map(|dim| matches!(dim, NatExpr::Constant(_)))
        .collect();

    let mut dim_args = vec![];

    #[allow(clippy::needless_range_loop)]
    for i in 0..rank {
        if !is_known_dimension[i] {
            dim_args.push(format!("{base}_d{i}"));
        }
    }
    let empty_expr = format!("tensor.empty({}) : {target_type}", dim_args.join(", "));
    empty_expr
}

fn element_type_name(dtype: Dtype) -> String {
    match dtype {
        Dtype::F32 => "f32",
        Dtype::U32 => "i32",
    }
    .to_string()
}
