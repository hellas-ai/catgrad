# A MLIR backend for [catgrad](https://catgrad.com)

Main modules:

- `lower` handles preprocessing and rendering the graph to MLIR text
    - `lower::preprocess` ...
        - typechecks
        - removes no-ops like identity casts
        - "deparametrises" the graph, so that parameters become function inputs
    - `lower::grammar` is a semi-structured representation of MLIR text
    - `lower::ops` contains lowerings of individual ops to MLIR
        - NOTE: there are likely bugs here!
- `codegen` calls MLIR tools to map MLIR text into a shared object file. If `CATGRAD_MLIR_CODEGEN_SCRIPT` is set, it will call that script instead.
- `runtime` is the low-level runtime interface to run the generated code
- `compile::CompiledModel` is a higher-level interface which also handles parameter passing

# Lowering ops

A quick note on op lowering (`lower::ops`):

- In general, the ith hypergraph node becomes an SSA variable like %vi
- However, many ops require intermediate SSA variables which do not correspond to hypergraph nodes
- To avoid name shadowing, intermediates are named using the *hyperedge* id as prefix, e.g. %e1_{suffix}
- This is typically named "base" in each lowering
- For example, the `shape_unpack` op needs to create a number of constant index intermediates:

```rust
let base = Identifier::Edge(ssa.edge_id);
..
let mut statements = vec![];

// Generate constant expressions
// {base}_c{i} = arith.constant {i} : index
statements.extend((0..rank).map(|i| format!("{base}_c{i} = arith.constant {i} : index")));

statements
    .into_iter()
    .map(grammar::Statement::Custom)
    .collect()
```

Final note: the "semi-structured" grammar turned out to be mostly useless below
the level of statements. It probably makes sense to delete `Assignment`,
`Expr`, and their contained types, but for now just avoid using them.

# Examples

Aside from the llama example, see also the `e2e` example which gives a self
contained (and working) example of running code.
