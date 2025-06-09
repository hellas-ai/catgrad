use std::collections::HashMap;
use std::sync::{Arc, Mutex}; // Mutex for global stores if primitives are static

use egglog::{
    ast::{Literal as EgglogLiteral, Sexp, Symbol as EgglogSymbol},
    sort::Sort as EgglogSort,
    EGraph as EgglogEGraph, ExtractReport as EgglogExtractReport,
    PrimitiveLike as EgglogPrimitiveLike, Value as EgglogValue,
};

// Catgrad types
use crate::backend::cpu::ndarray::{
    NdArray as CatGradNdArray, TaggedNdArray as CatGradTaggedNdArray,
};
use crate::core::{
    Dtype as CatGradDtype,
    NdArrayType as CatGradNdArrayType,
    Operation as CatGradOperation,
    Shape as CatGradShape,
    Term as CatGradTerm, // This is LaxOpenHypergraph
    Type as CatGradType, // This is Vec<CatGradNdArrayType>
};
use open_hypergraphs::lax::{NodeId as LaxNodeId, OpenHypergraph as LaxOpenHypergraph};

const EGG_PROGRAM_RULES: &str = r#"
(sort ShapeId)      ;; Opaque ID for CatGradShape
(sort DTypeId)      ;; Opaque ID for CatGradDtype
(sort TensorDataId) ;; Opaque ID for CatGradTaggedNdArray (constant data)

(datatype CatGradNode
    (Input String ShapeId DTypeId)
    (Param String ShapeId DTypeId)
    (Const TensorDataId ShapeId DTypeId)
    (MatrixMultiply CatGradNode CatGradNode) ;; FIXME: generalize to match catgrad
    (Const F32)
    (Max CatGradNode)
    (Sum CatGradNode)
    (Broadcast CatGradNode ShapeId) ;; broadcast a value to one of shape n+x
    (Reshape CatGradNode)
    (Transpose CatGradNode usize usize)
    (Embedding CatGradNode)
    (Index CatGradNode usize)
    (Concat CatGradNode usize)
    (Arange CatGradNode)
    (Not CatGradNode)
    (LT CatGradNode CatGradNode) ;; Less than
    (EQ CatGradNode CatGradNode) ;; Equality
    (Sin CatGradNode)
    (Cos CatGradNode)
    ;; all the below are pointwise
    (Copy CatGradNode)
    (Add CatGradNode CatGradNode)
    (Sub CatGradNode CatGradNode)
    (Mul CatGradNode CatGradNode)
    (Div CatGradNode CatGradNode)
    (Pow CatGradNode CatGradNode)
    (Negate CatGradNode)
)

;; Cost function
(function cost (CatGradNode) i64
  :merge (min old new)
  :default 1000000)

(rule ((= node (Input _ _ _))) ((set (cost node) 0)))
(rule ((= node (Param _ _ _))) ((set (cost node) 0)))
(rule ((= node (Const _ _ _))) ((set (cost node) 0)))

;; Primitives (Rust functions)
(primitive get_node_shape (CatGradNode) ShapeId)
(primitive get_node_dtype (CatGradNode) DTypeId)
(primitive get_node_tensor_data (CatGradNode) TensorDataId) ;; Only for (Const ...) nodes

(primitive is_zero_tensor (TensorDataId) ()) ;; Succeeds if tensor data is all zeros
(primitive is_one_tensor (TensorDataId) ())  ;; Succeeds if tensor data is all ones

(primitive eval_const_add (TensorDataId TensorDataId ShapeId DTypeId) TensorDataId)
(primitive eval_const_mul (TensorDataId TensorDataId ShapeId DTypeId) TensorDataId)

(primitive get_add_op_cost (ShapeId ShapeId) i64)
(primitive get_mul_op_cost (ShapeId ShapeId) i64)


(rule ((= node (Add lhs rhs))
       (= c_lhs (cost lhs))
       (= c_rhs (cost rhs))
       (= s_lhs (get_node_shape lhs))
       (= s_rhs (get_node_shape rhs))
       (= op_cost (get_add_op_cost s_lhs s_rhs)))
      ((set (cost node) (+ c_lhs c_rhs op_cost))))

(rule ((= node (Mul lhs rhs))
       (= c_lhs (cost lhs))
       (= c_rhs (cost rhs))
       (= s_lhs (get_node_shape lhs))
       (= s_rhs (get_node_shape rhs))
       (= op_cost (get_mul_op_cost s_lhs s_rhs)))
      ((set (cost node) (+ c_lhs c_rhs op_cost))))

;; Rewrite Rules
;; x + 0 = x
(rule ((= term (Add lhs (Const zero_data s_zero dt_zero)))
       (is_zero_tensor zero_data)
       (= s_lhs (get_node_shape lhs))
       (= dt_lhs (get_node_dtype lhs))
       ;; Assuming shapes and dtypes must exactly match for this specific 0-identity
       ;; For a real system, shape/dtype compatibility would be more nuanced.
       (= s_lhs s_zero)
       (= dt_lhs dt_zero))
      ((union term lhs)))

(rule ((= term (Add (Const zero_data s_zero dt_zero) rhs))
       (is_zero_tensor zero_data)
       (= s_rhs (get_node_shape rhs))
       (= dt_rhs (get_node_dtype rhs))
       (= s_rhs s_zero)
       (= dt_rhs dt_zero))
      ((union term rhs)))

;; x * 1 = x
(rule ((= term (Mul lhs (Const one_data s_one dt_one)))
       (is_one_tensor one_data)
       (= s_lhs (get_node_shape lhs))
       (= dt_lhs (get_node_dtype lhs))
       (= s_lhs s_one)
       (= dt_lhs dt_one))
      ((union term lhs)))

(rule ((= term (Mul (Const one_data s_one dt_one) rhs))
       (is_one_tensor one_data)
       (= s_rhs (get_node_shape rhs))
       (= dt_rhs (get_node_dtype rhs))
       (= s_rhs s_one)
       (= dt_rhs dt_one))
      ((union term rhs)))

;; Constant Folding: const1 + const2
(rewrite (Add (Const d1 s dt) (Const d2 s dt))
         (Const (eval_const_add d1 d2 s dt) s dt))

;; Constant Folding: const1 * const2
(rewrite (Mul (Const d1 s dt) (Const d2 s dt))
         (Const (eval_const_mul d1 d2 s dt) s dt))
"#;

// --- Stores for CatGrad objects and their Egglog IDs ---
// These need to be accessible by the primitives. Using Mutex for global-like access.
// A better approach for a real system would be to pass a context Rc<RefCell<Stores>>
// to the EGraph or have primitives capture it.
lazy_static::lazy_static! {
    static ref SHAPE_STORE: Mutex<ShapeStore> = Mutex::new(ShapeStore::default());
    static ref DTYPE_STORE: Mutex<DtypeStore> = Mutex::new(DtypeStore::default());
    static ref TENSOR_DATA_STORE: Mutex<TensorDataStore> = Mutex::new(TensorDataStore::default());
}

#[derive(Clone, Default)]
struct ShapeStore {
    shapes: Vec<CatGradShape>,
    id_to_idx: HashMap<String, usize>,
    // For quick reverse lookup if a shape already exists, to reuse IDs.
    // CatGradShape needs to be Hash + Eq for this.
    shape_to_idx: HashMap<CatGradShape, usize>,
    next_id_val: usize,
}

impl ShapeStore {
    fn assign_id(&mut self, shape: CatGradShape) -> String {
        if let Some(idx) = self.shape_to_idx.get(&shape) {
            return format!("s{}", idx);
        }
        let idx = self.shapes.len();
        let id = format!("s{}", idx);
        self.shapes.push(shape.clone());
        self.id_to_idx.insert(id.clone(), idx);
        self.shape_to_idx.insert(shape, idx);
        id
    }
    fn get_shape(&self, id_str: &str) -> Option<&CatGradShape> {
        self.id_to_idx
            .get(id_str)
            .and_then(|idx| self.shapes.get(*idx))
    }
}

#[derive(Clone, Default)]
struct DtypeStore {
    dtypes: Vec<CatGradDtype>,
    id_to_idx: HashMap<String, usize>,
    dtype_to_idx: HashMap<CatGradDtype, usize>, // CatGradDtype is Copy, Eq, Hash
}
impl DtypeStore {
    fn assign_id(&mut self, dtype: CatGradDtype) -> String {
        if let Some(idx) = self.dtype_to_idx.get(&dtype) {
            return format!("dt{}", idx);
        }
        let idx = self.dtypes.len();
        let id = format!("dt{}", idx);
        self.dtypes.push(dtype);
        self.id_to_idx.insert(id.clone(), idx);
        self.dtype_to_idx.insert(dtype, idx);
        id
    }
    fn get_dtype(&self, id_str: &str) -> Option<&CatGradDtype> {
        self.id_to_idx
            .get(id_str)
            .and_then(|idx| self.dtypes.get(*idx))
    }
}

#[derive(Clone, Default)]
struct TensorDataStore {
    tensor_data_vec: Vec<CatGradTaggedNdArray>,
    id_to_idx: HashMap<String, usize>,
    // No easy way to hash CatGradTaggedNdArray for reverse lookup to reuse IDs for identical constants
    // For prototype, always assign a new ID.
    // A real system might hash the data bytes.
}

impl TensorDataStore {
    fn assign_id(&mut self, data: CatGradTaggedNdArray) -> String {
        let idx = self.tensor_data_vec.len();
        let id = format!("td{}", idx);
        self.tensor_data_vec.push(data);
        self.id_to_idx.insert(id.clone(), idx);
        id
    }
    fn get_data(&self, id_str: &str) -> Option<&CatGradTaggedNdArray> {
        self.id_to_idx
            .get(id_str)
            .and_then(|idx| self.tensor_data_vec.get(*idx))
    }
}

// --- CatGrad Term to Egglog S-expression Converter ---
struct CatGradToEgglogConverter {
    // These are local to one conversion pass
    wire_to_egglog_var: HashMap<usize, String>, // LaxNodeId.0 to egglog var name
    s_expressions: Vec<String>,
    fresh_var_counter: usize,
    // To collect (Input ...) and (Param ...) for top-level definition
    top_level_defines: HashMap<String, String>, // egglog_var_name -> s_expression_string
}

impl CatGradToEgglogConverter {
    fn new() -> Self {
        Self {
            wire_to_egglog_var: HashMap::new(),
            s_expressions: Vec::new(),
            fresh_var_counter: 0,
            top_level_defines: HashMap::new(),
        }
    }

    fn fresh_egglog_var(&mut self, prefix: &str) -> String {
        let name = format!("_{}_{}", prefix, self.fresh_var_counter);
        self.fresh_var_counter += 1;
        name
    }

    fn translate_wire_to_egglog_var(
        &mut self,
        wire_id: LaxNodeId,
        term: &LaxOpenHypergraph<CatGradNdArrayType, CatGradOperation>,
        processed_ops: &mut HashMap<usize, Vec<String>>, // EdgeId.0 -> output egglog_vars
    ) -> Result<String, String> {
        let wire_idx = wire_id.0;
        if let Some(egglog_var) = self.wire_to_egglog_var.get(&wire_idx) {
            return Ok(egglog_var.clone());
        }

        // Is this wire a graph input?
        if let Some(input_idx) = term
            .sources
            .iter()
            .position(|&src_wire| src_wire == wire_id)
        {
            let egglog_var = self.fresh_egglog_var("input");
            let node_label = &term.hypergraph.nodes[wire_idx]; // This is CatGradNdArrayType
            let shape_id = SHAPE_STORE
                .lock()
                .unwrap()
                .assign_id(node_label.shape.clone());
            let dtype_id = DTYPE_STORE.lock().unwrap().assign_id(node_label.dtype);
            let input_op_name = format!("graph_input_{}", input_idx); // Or use a lookup for actual names if available

            let sexp_str = format!("(Input \"{}\" {} {})", input_op_name, shape_id, dtype_id);
            self.top_level_defines
                .entry(egglog_var.clone())
                .or_insert(sexp_str);
            self.wire_to_egglog_var.insert(wire_idx, egglog_var.clone());
            return Ok(egglog_var);
        }

        // Find the operation (edge) that produces this wire_id
        for (edge_idx, hyperedge) in term.hypergraph.adjacency.iter().enumerate() {
            if let Some(output_port_idx) = hyperedge
                .targets
                .iter()
                .position(|&target_wire| target_wire == wire_id)
            {
                // This edge produces our wire. Ensure this edge is processed.
                let output_egglog_vars = self.process_catgrad_op(edge_idx, term, processed_ops)?;
                return Ok(output_egglog_vars[output_port_idx].clone());
            }
        }
        Err(format!(
            "Wire ID {} is not a graph input and not an output of any operation.",
            wire_idx
        ))
    }

    fn process_catgrad_op(
        &mut self,
        edge_idx: usize,
        term: &LaxOpenHypergraph<CatGradNdArrayType, CatGradOperation>,
        processed_ops: &mut HashMap<usize, Vec<String>>, // EdgeId.0 -> output egglog_vars
    ) -> Result<Vec<String>, String> {
        if let Some(vars) = processed_ops.get(&edge_idx) {
            return Ok(vars.clone());
        }

        let catgrad_op = &term.hypergraph.edges[edge_idx];
        let hyperedge = &term.hypergraph.adjacency[edge_idx];

        let mut input_egglog_vars = Vec::new();
        for input_wire_id in &hyperedge.sources {
            input_egglog_vars.push(self.translate_wire_to_egglog_var(
                *input_wire_id,
                term,
                processed_ops,
            )?);
        }

        let (egglog_constructor, mut current_s_exp) = match catgrad_op {
            CatGradOperation::Const(val_f32) => {
                // For prototype, handle only F32 consts.
                // A real version would use CatGradTaggedNdArray and the TensorDataStore.
                // Here, we'll make a fake TensorDataId from the float value for simplicity.
                // Assume shape and dtype are known or can be inferred for constants.
                // For the prototype, we might need to hardcode or simplify this.
                // Let's assume a constant implies a scalar F32 for now.
                let scalar_shape = CatGradShape(vec![]);
                let f32_dtype = CatGradDtype::F32;
                let shape_id = SHAPE_STORE.lock().unwrap().assign_id(scalar_shape);
                let dtype_id = DTYPE_STORE.lock().unwrap().assign_id(f32_dtype);

                // Create a dummy TaggedNdArray for the store
                let ndarray = CatGradNdArray::new(vec![*val_f32], CatGradShape(vec![]));
                let data_id = TENSOR_DATA_STORE
                    .lock()
                    .unwrap()
                    .assign_id(CatGradTaggedNdArray::F32(ndarray));

                (
                    "Const".to_string(),
                    format!("({} {} {})", data_id, shape_id, dtype_id),
                )
            }
            CatGradOperation::Parameter(name) => {
                // Assume parameter shape/dtype are known from the output wire's label
                // This is a simplification. In reality, ParamOps are leaves.
                let output_wire_label = &term.hypergraph.nodes[hyperedge.targets[0].0];
                let shape_id = SHAPE_STORE
                    .lock()
                    .unwrap()
                    .assign_id(output_wire_label.shape.clone());
                let dtype_id = DTYPE_STORE
                    .lock()
                    .unwrap()
                    .assign_id(output_wire_label.dtype);
                let sexp_str = format!("(Param \"{}\" {} {})", name, shape_id, dtype_id);
                // Param is a leaf, define it at top level
                let egglog_var =
                    self.fresh_egglog_var(&format!("param_{}", name.replace(".", "_")));
                self.top_level_defines
                    .entry(egglog_var.clone())
                    .or_insert(sexp_str);

                // Special handling: process_catgrad_op should return the var directly defined
                processed_ops.insert(edge_idx, vec![egglog_var.clone()]);
                return Ok(vec![egglog_var]);
            }
            CatGradOperation::Add => (
                "Add".to_string(),
                format!("(Add {} {})", input_egglog_vars[0], input_egglog_vars[1]),
            ),
            CatGradOperation::Mul => (
                "Mul".to_string(),
                format!("(Mul {} {})", input_egglog_vars[0], input_egglog_vars[1]),
            ),
            // TODO: Map other CatGradOperations
            _ => {
                return Err(format!(
                    "Unsupported CatGradOperation for egglog translation: {:?}",
                    catgrad_op
                ))
            }
        };

        // For non-leaf ops, define their outputs
        let mut output_vars = vec![];
        for (i, output_wire_id) in hyperedge.targets.iter().enumerate() {
            let output_egglog_var = self.fresh_egglog_var(&format!(
                "{}_op{}_out{}",
                egglog_constructor.to_lowercase(),
                edge_idx,
                i
            ));
            self.s_expressions
                .push(format!("(define {} {})", output_egglog_var, current_s_exp));
            self.wire_to_egglog_var
                .insert(output_wire_id.0, output_egglog_var.clone());
            output_vars.push(output_egglog_var);
        }

        processed_ops.insert(edge_idx, output_vars.clone());
        Ok(output_vars)
    }

    pub fn convert(
        mut self,
        catgrad_term: &LaxOpenHypergraph<CatGradNdArrayType, CatGradOperation>,
        output_wires: &[LaxNodeId], // The specific output wires of the graph we care about
    ) -> Result<String, String> {
        let mut program_parts = vec![EGG_PROGRAM_RULES.to_string()];
        let mut processed_ops = HashMap::new();

        let mut final_output_egglog_vars = Vec::new();
        for output_wire_id in output_wires {
            let egglog_var = self.translate_wire_to_egglog_var(
                *output_wire_id,
                catgrad_term,
                &mut processed_ops,
            )?;
            final_output_egglog_vars.push(egglog_var);
        }

        // Add top-level defines (Inputs, Params)
        for (var_name, sexp_str) in self.top_level_defines {
            program_parts.push(format!("(define {} {})", var_name, sexp_str));
        }

        program_parts.extend(self.s_expressions);

        // Add extraction command for the final output(s)
        // For prototype, assume single output for simplicity
        if final_output_egglog_vars.len() == 1 {
            program_parts.push(format!("(extract {})", final_output_egglog_vars[0]));
        } else {
            // Handle multiple outputs if necessary, e.g., by creating a tuple or extracting one by one
            return Err(
                "Multiple outputs not yet fully supported for extraction in prototype".to_string(),
            );
        }
        program_parts.push("(run-schedule (saturate my_rules))".to_string()); // Assume a ruleset `my_rules`

        Ok(program_parts.join("\n"))
    }
}

// --- Egglog S-expression to CatGrad Term Converter ---
struct EgglogToCatGradConverter {
    shape_store: ShapeStore,
    dtype_store: DtypeStore,
    tensor_data_store: TensorDataStore,
    egglog_var_to_catgrad_wire: HashMap<String, LaxNodeId>,
    catgrad_term_builder: LaxOpenHypergraph<CatGradNdArrayType, CatGradOperation>,
    // To handle (let ((_v0 (Input ...))) (_v0)) from egglog extract
    // This maps let-bound vars to their defining CatGradNode Sexp
    let_bindings: HashMap<EgglogSymbol, egglog::ast::Sexp>,
}

#[derive(Debug)]
enum BuildFromSexpError {
    Egglog(egglog::ast::ParseError),
    UnknownVar(String),
    CatGrad(String),
    BadInput,
}

impl From<egglog::ast::ParseError> for BuildFromSexpError {
    fn from(err: egglog::ast::ParseError) -> Self {
        BuildFromSexpError::Egglog(err)
    }
}

impl From<String> for BuildFromSexpError {
    fn from(err: String) -> Self {
        BuildFromSexpError::CatGrad(err)
    }
}

impl From<&str> for BuildFromSexpError {
    fn from(err: &str) -> Self {
        BuildFromSexpError::CatGrad(err.to_string())
    }
}

impl std::fmt::Display for BuildFromSexpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildFromSexpError::Egglog(err) => write!(f, "Egglog parse error: {}", err),
            BuildFromSexpError::CatGrad(err) => write!(f, "CatGrad conversion error: {}", err),
            BuildFromSexpError::UnknownVar(var) => write!(f, "Unknown variable: {}", var),
            BuildFromSexpError::BadInput => write!(f, "Bad input: not enough S-expressions"),
        }
    }
}
impl std::error::Error for BuildFromSexpError {}

impl EgglogToCatGradConverter {
    fn new(s_store: ShapeStore, d_store: DtypeStore, td_store: TensorDataStore) -> Self {
        Self {
            shape_store: s_store,
            dtype_store: d_store,
            tensor_data_store: td_store,
            egglog_var_to_catgrad_wire: HashMap::new(),
            catgrad_term_builder: LaxOpenHypergraph::empty(),
            let_bindings: HashMap::new(),
        }
    }

    // Parses egglog's Sexp AST and builds CatGradTerm
    // Returns the CatGrad NodeId for the output wire of the created operation
    fn build_from_sexp(
        &mut self,
        sexp: &egglog::ast::Sexp,
    ) -> Result<LaxNodeId, BuildFromSexpError> {
        match sexp {
            egglog::ast::Sexp::Atom(sym, _) => {
                // This is a variable, look it up. If it's let-bound, parse its definition.
                if let Some(defined_sexp) = self.let_bindings.get(sym).cloned() {
                    // If already processed this var via let, return its wire
                    if let Some(wire_id) = self.egglog_var_to_catgrad_wire.get(&sym.to_string()) {
                        return Ok(*wire_id);
                    }
                    // Otherwise, process the definition now
                    let wire_id = self.build_from_sexp(&defined_sexp)?;
                    self.egglog_var_to_catgrad_wire
                        .insert(sym.to_string(), wire_id);
                    return Ok(wire_id);
                }
                // If not let-bound, it must have been defined earlier (e.g. by (define ...))
                // and should already be in wire_to_egglog_var from the forward pass,
                // or it's an error.
                // For parsing extracted term, this direct var usage means it was likely defined via (define)
                // and should be in wire_to_egglog_var.
                // However, our CatGradToEgglogConverter creates (define) for *all* nodes.
                // The (extract) output might simplify this.
                // If egglog (extract) gives (Input "name" s_id dt_id), this branch is hit.
                // For now, if it's an atom, assume it's a var that should be in egglog_var_to_catgrad_wire
                // This part of logic might need refinement based on actual (extract) output.
                self.egglog_var_to_catgrad_wire
                    .get(&sym.to_string())
                    .cloned()
                    .ok_or_else(|| BuildFromSexpError::UnknownVar(sym.to_string()))
            }
            egglog::ast::Sexp::List(items, list_span) => {
                if items.is_empty() {
                    return Err(BuildFromSexpError::BadInput);
                }
                let head_sexp = &items[0];
                let args_sexps = &items[1..];

                match head_sexp {
                    egglog::ast::Sexp::Atom(head_sym, _) => {
                        let op_name = head_sym.as_str();
                        match op_name {
                            "Input" => {
                                let name = args_sexps[0].expect_string("Input name")?;
                                let shape_id_str =
                                    args_sexps[1].expect_atom("ShapeID for Input")?.to_string();
                                let dtype_id_str =
                                    args_sexps[2].expect_atom("DTypeID for Input")?.to_string();

                                let shape = self
                                    .shape_store
                                    .get_shape(&shape_id_str)
                                    .ok_or("Invalid ShapeID")?
                                    .clone();
                                let dtype = *self
                                    .dtype_store
                                    .get_dtype(&dtype_id_str)
                                    .ok_or("Invalid DTypeID")?;

                                // In CatGrad, an Input is just a wire with a type.
                                // It doesn't correspond to an "Operation" in the same way.
                                // We create a new node (wire) in the graph for it.
                                let node_label = CatGradNdArrayType::new(shape, dtype);
                                let wire_id =
                                    self.catgrad_term_builder.hypergraph.new_node(node_label);
                                // This wire is now a source of the new graph.
                                self.catgrad_term_builder.sources.push(wire_id);
                                Ok(wire_id)
                            }
                            "Param" => {
                                let name = args_sexps[0].expect_string("Param name")?;
                                let shape_id_str =
                                    args_sexps[1].expect_atom("ShapeID for Param")?.to_string();
                                let dtype_id_str =
                                    args_sexps[2].expect_atom("DTypeID for Param")?.to_string();

                                let shape = self
                                    .shape_store
                                    .get_shape(&shape_id_str)
                                    .ok_or("Invalid ShapeID for Param")?
                                    .clone();
                                let dtype = *self
                                    .dtype_store
                                    .get_dtype(&dtype_id_str)
                                    .ok_or("Invalid DTypeID for Param")?;
                                let node_label = CatGradNdArrayType::new(shape, dtype);

                                let cg_op = CatGradOperation::Parameter(name);
                                let (_edge_id, interface) = self
                                    .catgrad_term_builder
                                    .hypergraph
                                    .new_operation(cg_op, vec![], vec![node_label]);
                                Ok(interface.1[0]) // Param has one output wire
                            }
                            "Const" => {
                                let data_id_str = args_sexps[0]
                                    .expect_atom("TensorDataID for Const")?
                                    .to_string();
                                let shape_id_str =
                                    args_sexps[1].expect_atom("ShapeID for Const")?.to_string();
                                let dtype_id_str =
                                    args_sexps[2].expect_atom("DTypeID for Const")?.to_string();

                                let data = self
                                    .tensor_data_store
                                    .get_data(&data_id_str)
                                    .ok_or("Invalid TensorDataID for Const")?;
                                let shape = self
                                    .shape_store
                                    .get_shape(&shape_id_str)
                                    .ok_or("Invalid ShapeID for Const")?
                                    .clone();
                                let dtype = *self
                                    .dtype_store
                                    .get_dtype(&dtype_id_str)
                                    .ok_or("Invalid DTypeID for Const")?;
                                let node_label = CatGradNdArrayType::new(shape, dtype);

                                // Extract f32 val for CatGradOperation::Const (prototype limitation)
                                let val_f32 = match data {
                                    CatGradTaggedNdArray::F32(nd_arr) => {
                                        if nd_arr.shape.0.is_empty() && nd_arr.data.len() == 1 {
                                            nd_arr.data[0]
                                        } else {
                                            return Err(BuildFromSexpError::CatGrad(
                                                "Const data for prototype must be scalar F32"
                                                    .to_string(),
                                            ));
                                        }
                                    }
                                    _ => {
                                        return Err(BuildFromSexpError::CatGrad(
                                            "Const data for prototype must be F32".to_string(),
                                        ));
                                    }
                                };

                                let cg_op = CatGradOperation::Const(val_f32);
                                let (_edge_id, interface) = self
                                    .catgrad_term_builder
                                    .hypergraph
                                    .new_operation(cg_op, vec![], vec![node_label]);
                                Ok(interface.1[0]) // Const has one output wire
                            }
                            "Add" | "Mul" => {
                                let lhs_wire = self.build_from_sexp(&args_sexps[0])?;
                                let rhs_wire = self.build_from_sexp(&args_sexps[1])?;

                                let lhs_label =
                                    self.catgrad_term_builder.hypergraph.nodes[lhs_wire.0].clone();
                                // TODO: Check compatibility or derive output type properly
                                let output_label = lhs_label.clone();

                                let cg_op = if op_name == "Add" {
                                    CatGradOperation::Add
                                } else {
                                    CatGradOperation::Mul
                                };
                                let (_edge_id, interface) =
                                    self.catgrad_term_builder.hypergraph.new_operation(
                                        cg_op,
                                        vec![lhs_label, output_label.clone()],
                                        vec![output_label],
                                    );
                                self.catgrad_term_builder
                                    .hypergraph
                                    .unify(interface.0[0], lhs_wire);
                                self.catgrad_term_builder
                                    .hypergraph
                                    .unify(interface.0[1], rhs_wire);
                                Ok(interface.1[0])
                            }
                            "let" => {
                                // (let ((var1 def1) (var2 def2) ...) body_expr)
                                if let Sexp::List(bindings, _) = &args_sexps[0] {
                                    let mut new_bindings = self.let_bindings.clone();
                                    for binding_sexp in bindings {
                                        if let Sexp::List(pair, _) = binding_sexp {
                                            if pair.len() == 2 {
                                                if let Sexp::Atom(var_sym, _) = &pair[0] {
                                                    new_bindings.insert(*var_sym, pair[1].clone());
                                                } else {
                                                    return Err("Let binding var not atom".into());
                                                }
                                            } else {
                                                return Err("Let binding not a pair".into());
                                            }
                                        } else {
                                            return Err("Let binding not a list".into());
                                        }
                                    }
                                    // Process body with these bindings
                                    let mut sub_converter = EgglogToCatGradConverter {
                                        shape_store: self.shape_store.clone(),
                                        dtype_store: self.dtype_store.clone(),
                                        tensor_data_store: self.tensor_data_store.clone(),
                                        egglog_var_to_catgrad_wire: self
                                            .egglog_var_to_catgrad_wire
                                            .clone(),
                                        catgrad_term_builder: LaxOpenHypergraph::empty(), // build sub-graph
                                        let_bindings: new_bindings,
                                    };
                                    let body_wire =
                                        sub_converter.build_from_sexp(&args_sexps[1])?;
                                    // Now merge sub_converter's term_builder into self.catgrad_term_builder
                                    // This is complex: involves remapping NodeIds, etc.
                                    // For prototype, (let) might be tricky if extract doesn't inline everything.
                                    // Assuming extract produces a tree or DAG without new lets for now.
                                    // If extract produces lets, we need to handle them properly.
                                    // This simplistic handling assumes the let-bound vars are used in the body
                                    // and their corresponding CatGrad wires will be created.
                                    self.catgrad_term_builder.hypergraph.coproduct_assign(
                                        sub_converter.catgrad_term_builder.hypergraph,
                                    );
                                    // The body_wire is relative to sub_converter. Needs remapping.
                                    // This is a placeholder, real merging is harder.
                                    self.egglog_var_to_catgrad_wire
                                        .extend(sub_converter.egglog_var_to_catgrad_wire);
                                    return Ok(body_wire); // This wire ID might be from a different graph if not careful
                                }
                                Err("Invalid let structure".into())
                            }
                            _ => Err(format!(
                                "Unsupported egglog constructor in output: {}",
                                op_name
                            )
                            .into()),
                        }
                    }
                    _ => Err("Egglog output head is not an atom".into()),
                }
            }
            _ => Err("Egglog output is not a list or atom".into()),
        }
    }

    pub fn convert(
        mut self,
        extracted_sexp_str: &str,
        // Original catgrad term's output types, to set for the new term
        original_output_types: CatGradType,
    ) -> Result<CatGradTerm, BuildFromSexpError> {
        let sexps_vec = egglog::ast::all_sexps(egglog::ast::Context::new(None, extracted_sexp_str))
            .map_err(|e| e.to_string())?;
        if sexps_vec.is_empty() {
            return Err("Empty extraction result from egglog".into());
        }
        let root_sexp = &sexps_vec[0]; // Assume extract gives one top-level term expression

        let final_output_wire = self.build_from_sexp(root_sexp)?;
        self.catgrad_term_builder.targets = vec![final_output_wire];

        // Ensure output wire types match the original
        // This is simplified. A real system would verify or adjust.
        if self.catgrad_term_builder.targets.len() == original_output_types.len() {
            for (i, target_wire_id) in self.catgrad_term_builder.targets.iter().enumerate() {
                self.catgrad_term_builder.hypergraph.nodes[target_wire_id.0] =
                    original_output_types[i].clone();
            }
        } else {
            // Mismatch in number of outputs, could be an error or require complex handling
            return Err(format!(
                "Output arity mismatch: expected {}, got {}",
                original_output_types.len(),
                self.catgrad_term_builder.targets.len()
            )
            .into());
        }

        // The builder now contains the new CatGradTerm
        // The quotient map might need to be applied if `build_from_sexp` didn't handle all sharing correctly.
        // For now, assume build_from_sexp tries to build a minimal graph.
        self.catgrad_term_builder.hypergraph.quotient();

        Ok(self.catgrad_term_builder)
    }
}

/// Main public function for optimization
pub fn optimize_with_egglog(
    catgrad_term: &CatGradTerm,
    output_wires_indices: &[usize], // Indices into catgrad_term.targets
) -> Result<CatGradTerm, String> {
    // 1. Clear global stores for a fresh run (or manage them per call if not global)
    SHAPE_STORE.lock().unwrap().shapes.clear();
    SHAPE_STORE.lock().unwrap().id_to_idx.clear();
    SHAPE_STORE.lock().unwrap().shape_to_idx.clear();
    SHAPE_STORE.lock().unwrap().next_id_val = 0;
    DTYPE_STORE.lock().unwrap().dtypes.clear();
    DTYPE_STORE.lock().unwrap().id_to_idx.clear();
    DTYPE_STORE.lock().unwrap().dtype_to_idx.clear();
    TENSOR_DATA_STORE.lock().unwrap().tensor_data_vec.clear();
    TENSOR_DATA_STORE.lock().unwrap().id_to_idx.clear();

    // 2. Translate CatGradTerm to egglog s-expression
    let converter = CatGradToEgglogConverter::new();
    let original_target_node_ids: Vec<LaxNodeId> = output_wires_indices
        .iter()
        .map(|&idx| catgrad_term.targets[idx])
        .collect();

    let egglog_program_string = converter
        .convert(catgrad_term, &original_target_node_ids)
        .map_err(|e| format!("CatGrad to Egglog conversion error: {}", e))?;

    println!("--- Generated Egglog Program ---");
    println!("{}", egglog_program_string);
    println!("--------------------------------");

    // 3. Run egglog
    let mut egraph = EgglogEGraph::default();
    // Register primitives (this needs more elaborate setup matching egglog's API)
    register_catgrad_primitives(&mut egraph)?;

    let msgs = egraph
        .parse_and_run_program(None, &egglog_program_string)
        .map_err(|e| format!("Egglog execution error: {}", e))?;

    println!("--- Egglog Messages ---");
    for msg in msgs {
        println!("{}", msg);
    }
    println!("-----------------------");

    // 4. Extract optimized term
    // The extraction result comes from the last `(extract ...)` command's side effect on `egraph.extract_report`.
    let extracted_s_expr = match egraph.get_extract_report() {
        Some(EgglogExtractReport::Best {
            termdag,
            cost: _,
            term,
        }) => {
            println!(
                "Extracted egglog term (cost ...): {} (internal)",
                termdag.to_string(term)
            );
            termdag.to_string(term) // This is what we need to parse back
        }
        Some(EgglogExtractReport::Variants { termdag, terms }) => {
            if terms.is_empty() {
                return Err("Egglog extraction returned no variants.".to_string());
            }
            println!("Extracted egglog variants. Using the first one.");
            termdag.to_string(&terms[0])
        }
        None => return Err("No extraction report from egglog. Did (extract) run?".to_string()),
    };

    println!("--- Extracted Egglog S-expression ---");
    println!("{}", extracted_s_expr);
    println!("-------------------------------------");

    // 5. Translate egglog s-expression back to CatGradTerm
    let shape_s = SHAPE_STORE.lock().unwrap().clone();
    let dtype_s = DTYPE_STORE.lock().unwrap().clone();
    let tensor_s = TENSOR_DATA_STORE.lock().unwrap().clone();
    let back_converter = EgglogToCatGradConverter::new(shape_s, dtype_s, tensor_s);

    // Collect original output types
    let original_output_types: CatGradType = original_target_node_ids
        .iter()
        .map(|node_id| catgrad_term.hypergraph.nodes[node_id.0].clone())
        .collect();

    let optimized_catgrad_term = back_converter
        .convert(&extracted_s_expr, original_output_types)
        .map_err(|e| format!("Egglog to CatGrad conversion error: {}", e))?;

    Ok(optimized_catgrad_term)
}

// --- Primitive Implementations ---
// Helper macro to create EgglogPrimitive wrappers
macro_rules! make_primitive {
    ($name:ident, $rust_fn:ident, $arity:expr) => {
        #[derive(Debug)]
        struct $name;
        impl EgglogPrimitiveLike for $name {
            fn name(&self) -> EgglogSymbol {
                EgglogSymbol::from(stringify!($name).to_lowercase())
            }
            fn get_type_constraints(
                &self,
                _span: &egglog::ast::Span,
            ) -> Box<dyn egglog::constraint::TypeConstraint> {
                // For prototype, assume any type for args, specific type for output
                // This needs to be more robust, using egglog::constraint types.
                // This is a placeholder, proper type constraints are complex.
                // We'd use SimpleTypeConstraint or AllEqualTypeConstraint from egglog.
                // For now, let this be dynamically checked or loosely typed.
                Box::new(egglog::constraint::SimpleTypeConstraint::new(
                    self.name(),
                    vec![],                   // This is incorrect, needs actual arg/ret sorts
                    egglog::ast::Span::Panic, // Placeholder
                ))
            }
            fn apply(
                &self,
                values: &[EgglogValue],
                // sorts: (&[EgglogArcSort], &EgglogArcSort), // Correct signature might involve sorts
                _typeinfo_sorts: (&[Arc<dyn EgglogSort>], &Arc<dyn EgglogSort>), // Placeholder for actual signature
                _egraph: Option<&mut EgglogEGraph>,
            ) -> Option<EgglogValue> {
                if values.len() != $arity {
                    // egglog should enforce arity based on (primitive ...) declaration
                    // but good to double check.
                    eprintln!(
                        "Primitive {} called with {} args, expected {}",
                        stringify!($name),
                        values.len(),
                        $arity
                    );
                    return None;
                }
                $rust_fn(values)
            }
        }
    };
}

fn get_string_arg(val: &EgglogValue, store_name: &str) -> Result<String, String> {
    // Egglog Values are u64. Primitives for String, ShapeId etc. will pass Symbols (interned strings) as u64.
    // We need to convert this back. This assumes Symbol::from(NonZeroU32) then .bits as u64.
    let bits32 = val.bits as u32;
    if let Some(nonzero_bits32) = std::num::NonZeroU32::new(bits32) {
        Ok(EgglogSymbol::from(nonzero_bits32).to_string())
    } else {
        Err(format!("Invalid ID format for {} in primitive", store_name))
    }
}

// Primitive: (get_node_shape <CatGradNode_val>) -> ShapeId_val
fn prim_get_node_shape(args: &[EgglogValue]) -> Option<EgglogValue> {
    // Arg0 is a CatGradNode (represented by its ID in the egraph).
    // This primitive would need to query the egraph to find what (ShapeId ...) is associated with that node.
    // This requires egraph access and a way to query attached data, which is complex.
    // For the prototype, this might be hard to implement fully without deeper egraph interaction.
    // A simplification: If CatGradNode itself stores its shape_id, we can extract it.
    // But egglog terms are unified, so we need canonical representative.
    // Placeholder: return a dummy ShapeId.
    Some(EgglogValue::from(EgglogSymbol::from(
        "dummy_shape_id_from_prim",
    )))
}
make_primitive!(GetNodeShapePrim, prim_get_node_shape, 1);

fn prim_get_node_dtype(args: &[EgglogValue]) -> Option<EgglogValue> {
    Some(EgglogValue::from(EgglogSymbol::from(
        "dummy_dtype_id_from_prim",
    )))
}
make_primitive!(GetNodeDtypePrim, prim_get_node_dtype, 1);

fn prim_get_node_tensor_data(args: &[EgglogValue]) -> Option<EgglogValue> {
    Some(EgglogValue::from(EgglogSymbol::from(
        "dummy_tensor_data_id_from_prim",
    )))
}
make_primitive!(GetNodeTensorDataPrim, prim_get_node_tensor_data, 1);

fn prim_is_zero_tensor(args: &[EgglogValue]) -> Option<EgglogValue> {
    let data_id_str = get_string_arg(&args[0], "TensorDataId").ok()?;
    let store = TENSOR_DATA_STORE.lock().unwrap();
    let data = store.get_data(&data_id_str)?;
    match data {
        CatGradTaggedNdArray::F32(arr) => {
            if arr.data.iter().all(|x| *x == 0.0) {
                Some(EgglogValue::unit())
            } else {
                None
            }
        }
        // Handle other dtypes
        _ => None,
    }
}
make_primitive!(IsZeroTensorPrim, prim_is_zero_tensor, 1);

fn prim_is_one_tensor(args: &[EgglogValue]) -> Option<EgglogValue> {
    let data_id_str = get_string_arg(&args[0], "TensorDataId").ok()?;
    let store = TENSOR_DATA_STORE.lock().unwrap();
    let data = store.get_data(&data_id_str)?;
    match data {
        CatGradTaggedNdArray::F32(arr) => {
            if arr.data.iter().all(|x| *x == 1.0) {
                Some(EgglogValue::unit())
            } else {
                None
            }
        }
        _ => None,
    }
}
make_primitive!(IsOneTensorPrim, prim_is_one_tensor, 1);

fn prim_eval_const_add(args: &[EgglogValue]) -> Option<EgglogValue> {
    let d1_id_str = get_string_arg(&args[0], "TensorDataId1 for Add").ok()?;
    let d2_id_str = get_string_arg(&args[1], "TensorDataId2 for Add").ok()?;
    // Shape and DType args are for type checking / ensuring compatibility, not strictly needed for op if data has type info
    // let _s_id_str = get_string_arg(&args[2], "ShapeId for Add").ok()?;
    // let _dt_id_str = get_string_arg(&args[3], "DTypeId for Add").ok()?;

    let mut store = TENSOR_DATA_STORE.lock().unwrap();
    let d1 = store.get_data(&d1_id_str)?.clone(); // Clone to operate
    let d2 = store.get_data(&d2_id_str)?.clone();

    match (d1, d2) {
        (CatGradTaggedNdArray::F32(arr1), CatGradTaggedNdArray::F32(arr2)) => {
            if arr1.shape != arr2.shape {
                return None;
            } // Basic check
            let mut result_data = Vec::with_capacity(arr1.data.len());
            for (x, y) in arr1.data.iter().zip(arr2.data.iter()) {
                result_data.push(x + y);
            }
            let result_arr = CatGradNdArray::new(result_data, arr1.shape.clone());
            let new_data_id = store.assign_id(CatGradTaggedNdArray::F32(result_arr));
            Some(EgglogValue::from(EgglogSymbol::from(new_data_id.as_str())))
        }
        _ => None, // Unsupported types for add
    }
}
make_primitive!(EvalConstAddPrim, prim_eval_const_add, 4);

fn prim_eval_const_mul(args: &[EgglogValue]) -> Option<EgglogValue> {
    let d1_id_str = get_string_arg(&args[0], "TensorDataId1 for Mul").ok()?;
    let d2_id_str = get_string_arg(&args[1], "TensorDataId2 for Mul").ok()?;

    let mut store = TENSOR_DATA_STORE.lock().unwrap();
    let d1 = store.get_data(&d1_id_str)?.clone();
    let d2 = store.get_data(&d2_id_str)?.clone();

    match (d1, d2) {
        (CatGradTaggedNdArray::F32(arr1), CatGradTaggedNdArray::F32(arr2)) => {
            if arr1.shape != arr2.shape {
                return None;
            }
            let mut result_data = Vec::with_capacity(arr1.data.len());
            for (x, y) in arr1.data.iter().zip(arr2.data.iter()) {
                result_data.push(x * y);
            }
            let result_arr = CatGradNdArray::new(result_data, arr1.shape.clone());
            let new_data_id = store.assign_id(CatGradTaggedNdArray::F32(result_arr));
            Some(EgglogValue::from(EgglogSymbol::from(new_data_id.as_str())))
        }
        _ => None,
    }
}
make_primitive!(EvalConstMulPrim, prim_eval_const_mul, 4);

fn prim_get_add_op_cost(args: &[EgglogValue]) -> Option<EgglogValue> {
    // Placeholder: actual cost depends on shapes.
    // Args are ShapeIds.
    Some(EgglogValue::from(10i64)) // Example cost
}
make_primitive!(GetAddOpCostPrim, prim_get_add_op_cost, 2);

fn prim_get_mul_op_cost(args: &[EgglogValue]) -> Option<EgglogValue> {
    Some(EgglogValue::from(10i64))
}
make_primitive!(GetMulOpCostPrim, prim_get_mul_op_cost, 2);

fn register_catgrad_primitives(egraph: &mut EgglogEGraph) -> Result<(), String> {
    // This registration mechanism depends on egglog's exact API.
    // egglog::EGraph::add_primitive takes `impl Into<Primitive>`
    // The make_primitive macro defines structs that impl PrimitiveLike.
    // We need to ensure they are correctly wrapped.
    egraph.add_primitive(GetNodeShapePrim);
    egraph.add_primitive(GetNodeDtypePrim);
    egraph.add_primitive(GetNodeTensorDataPrim);
    egraph.add_primitive(IsZeroTensorPrim);
    egraph.add_primitive(IsOneTensorPrim);
    egraph.add_primitive(EvalConstAddPrim);
    egraph.add_primitive(EvalConstMulPrim);
    egraph.add_primitive(GetAddOpCostPrim);
    egraph.add_primitive(GetMulOpCostPrim);
    Ok(())
}

// Helper for Sexp parsing (simplified for prototype)
// Real parsing should use egglog's parser or a proper Sexp library.
trait SexpExt {
    fn expect_atom(&self, context: &str) -> Result<EgglogSymbol, String>;
    fn expect_string(&self, context: &str) -> Result<String, String>;
}

impl SexpExt for egglog::ast::Sexp {
    fn expect_atom(&self, context: &str) -> Result<EgglogSymbol, String> {
        if let egglog::ast::Sexp::Atom(s, _) = self {
            Ok(*s)
        } else {
            Err(format!(
                "Expected atom for {} but got (something else?)",
                context
            ))
        }
    }
    fn expect_string(&self, context: &str) -> Result<String, String> {
        if let egglog::ast::Sexp::Literal(EgglogLiteral::String(s), _) = self {
            Ok(s.to_string())
        } else {
            Err(format!(
                "Expected string literal for {} but got (something else?)",
                context
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Dtype, NdArrayType, Operation, Shape};
    use open_hypergraphs::lax::Hyperedge;

    #[test]
    fn test_prototype_optimization_add_zero() {
        // Build a simple CatGradTerm: input + 0
        let mut term = LaxOpenHypergraph::<NdArrayType, Operation>::empty();

        let shape = Shape(vec![2, 2]);
        let dtype = Dtype::F32;
        let nd_type = NdArrayType::new(shape.clone(), dtype);

        // Input wire
        let input_wire = term.hypergraph.new_node(nd_type.clone());
        term.sources.push(input_wire);

        // Const 0 wire
        // let zero_data = NdArray::new(vec![0.0; 4], shape.clone());
        // let zero_tagged = TaggedNdArray::F32(zero_data);
        let const_zero_op_node_label = nd_type.clone();
        let (_const_edge, const_interface) = term.hypergraph.new_operation(
            Operation::Const(0.0),
            vec![],
            vec![const_zero_op_node_label],
        );
        let zero_wire = const_interface.1[0];

        // Add operation
        let add_op_output_node_label = nd_type.clone();
        let (_add_edge, add_interface) = term.hypergraph.new_operation(
            Operation::Add,
            vec![nd_type.clone(), nd_type.clone()],
            vec![add_op_output_node_label],
        );
        // Connect inputs to Add
        term.hypergraph.unify(add_interface.0[0], input_wire);
        term.hypergraph.unify(add_interface.0[1], zero_wire);

        let output_wire = add_interface.1[0];
        term.targets.push(output_wire);

        println!("Original CatGrad Term (structure):");
        // A simple way to inspect, not a full print
        println!("  Sources: {:?}", term.sources);
        println!("  Targets: {:?}", term.targets);
        println!(
            "  Nodes: {} (labels omitted for brevity)",
            term.hypergraph.nodes.len()
        );
        println!(
            "  Edges: {} (ops: {:?})",
            term.hypergraph.edges.len(),
            term.hypergraph.edges
        );
        for (i, adj) in term.hypergraph.adjacency.iter().enumerate() {
            println!("    Edge {}: {:?} -> {:?}", i, adj.sources, adj.targets);
        }

        match optimize_with_egglog(&term, &[0]) {
            Ok(optimized_term) => {
                println!("\nOptimized CatGrad Term (structure):");
                println!("  Sources: {:?}", optimized_term.sources);
                println!("  Targets: {:?}", optimized_term.targets);
                println!("  Nodes: {}", optimized_term.hypergraph.nodes.len());
                println!(
                    "  Edges: {} (ops: {:?})",
                    optimized_term.hypergraph.edges.len(),
                    optimized_term.hypergraph.edges
                );
                for (i, adj) in optimized_term.hypergraph.adjacency.iter().enumerate() {
                    println!("    Edge {}: {:?} -> {:?}", i, adj.sources, adj.targets);
                }

                // Expected: graph simplifies to just the input wire
                // - Sources should still contain the original input wire (or its equivalent after re-numbering)
                // - Targets should point to that same input wire.
                // - Number of operations (edges) should be 0.
                // - Number of nodes might be 1 (just the input wire).
                assert_eq!(
                    optimized_term.hypergraph.edges.len(),
                    0,
                    "Optimized graph should have no operations for input + 0"
                );
                assert_eq!(
                    optimized_term.hypergraph.nodes.len(),
                    1,
                    "Optimized graph should have one node (the input)"
                );
                assert_eq!(
                    optimized_term.sources.len(),
                    1,
                    "Optimized graph should have one source"
                );
                assert_eq!(
                    optimized_term.targets.len(),
                    1,
                    "Optimized graph should have one target"
                );
                assert_eq!(
                    optimized_term.sources[0], optimized_term.targets[0],
                    "Output should be the input wire"
                );
            }
            Err(e) => panic!("Optimization failed: {}", e),
        }
    }
}
