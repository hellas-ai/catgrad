use super::*;
use catgrad::abstract_interpreter::Value as TypeValue;
use catgrad::interpreter::backend::candle::CandleBackend;
use catgrad::interpreter::{
    Backend, Interpreter, Parameters, TaggedTensor, TaggedTensorTuple, Value, tensor,
};
use catgrad::prelude::*;
use catgrad::typecheck::value_types::*;

fn tensor_type<const N: usize>(dtype: Dtype, shape: [usize; N]) -> Type {
    let shape = ShapeExpr::Shape(shape.into_iter().map(NatExpr::Constant).collect());
    TypeValue::Tensor(TypeExpr::NdArrayType(NdArrayType {
        dtype: DtypeExpr::Constant(dtype),
        shape,
    }))
}

fn f32_type<const N: usize>(shape: [usize; N]) -> Type {
    tensor_type(Dtype::F32, shape)
}

struct SimpleModule<const A: usize, const B: usize> {
    name: &'static str,
    in_types: [Type; A],
    out_types: [Type; B],
    build: fn(&Builder, [Var; A]) -> [Var; B],
}

impl<const A: usize, const B: usize> SimpleModule<A, B> {
    fn new(
        name: &'static str,
        in_types: [Type; A],
        out_types: [Type; B],
        build: fn(&Builder, [Var; A]) -> [Var; B],
    ) -> Self {
        Self {
            name,
            in_types,
            out_types,
            build,
        }
    }
}

impl<const A: usize, const B: usize> Module<A, B> for SimpleModule<A, B> {
    fn ty(&self) -> ([Type; A], [Type; B]) {
        (self.in_types.clone(), self.out_types.clone())
    }

    fn path(&self) -> Path {
        path(vec!["test", "helpers", "tensors", self.name]).unwrap()
    }

    fn def(&self, builder: &Builder, args: [Var; A]) -> [Var; B] {
        (self.build)(builder, args)
    }
}

struct TestCtx {
    backend: CandleBackend,
    interpreter: Interpreter<CandleBackend>,
}

impl TestCtx {
    fn new() -> Self {
        let backend = CandleBackend::new();
        let interpreter = Interpreter::new(backend.clone(), stdlib(), Parameters::default());
        Self {
            backend,
            interpreter,
        }
    }

    fn f32<const N: usize>(&self, shape: [usize; N], data: Vec<f32>) -> Value<CandleBackend> {
        tensor(&self.backend, Shape(shape.to_vec()), data).expect("tensor creation failed")
    }

    fn run<const A: usize, const B: usize>(
        &self,
        module: SimpleModule<A, B>,
        inputs: Vec<Value<CandleBackend>>,
    ) -> Vec<Value<CandleBackend>> {
        let typed_term = module.term().expect("failed to build typed term");
        self.interpreter
            .run(typed_term.term, inputs)
            .expect("interpreter run failed")
    }

    fn run1<const A: usize>(
        &self,
        module: SimpleModule<A, 1>,
        inputs: Vec<Value<CandleBackend>>,
    ) -> Value<CandleBackend> {
        let mut outputs = self.run(module, inputs);
        outputs.pop().expect("missing output")
    }

    fn assert_f32_eq(&self, actual: Value<CandleBackend>, expected: Value<CandleBackend>) {
        match (actual, expected) {
            (
                Value::Tensor(TaggedTensor::F32([actual])),
                Value::Tensor(TaggedTensor::F32([expected])),
            ) => {
                assert!(
                    self.backend
                        .compare(TaggedTensorTuple::F32([actual, expected])),
                    "tensor output did not match expected values"
                );
            }
            _ => panic!("expected f32 tensor outputs"),
        }
    }
}

#[test]
fn split_slices_along_dim() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "split_dim1",
        [f32_type([1, 4])],
        [f32_type([1, 2]), f32_type([1, 2])],
        |builder, [x]| {
            let parts = split(builder, 1, &[2, 2], x);
            [parts[0].clone(), parts[1].clone()]
        },
    );

    let x = ctx.f32([1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let outputs = ctx.run(module, vec![x]);
    let [left, right]: [Value<CandleBackend>; 2] = outputs.try_into().expect("expected 2 outputs");

    ctx.assert_f32_eq(left, ctx.f32([1, 2], vec![1.0, 2.0]));
    ctx.assert_f32_eq(right, ctx.f32([1, 2], vec![3.0, 4.0]));
}

#[test]
fn chunk_slices_along_dim() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "chunk_dim1",
        [f32_type([1, 4])],
        [f32_type([1, 2]), f32_type([1, 2])],
        |builder, [x]| {
            let parts = chunk(builder, 1, 2, 2, x);
            [parts[0].clone(), parts[1].clone()]
        },
    );

    let x = ctx.f32([1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let outputs = ctx.run(module, vec![x]);
    let [left, right]: [Value<CandleBackend>; 2] = outputs.try_into().expect("expected 2 outputs");

    ctx.assert_f32_eq(left, ctx.f32([1, 2], vec![1.0, 2.0]));
    ctx.assert_f32_eq(right, ctx.f32([1, 2], vec![3.0, 4.0]));
}

#[test]
fn squeeze_removes_singleton_dim() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "squeeze_dim1",
        [f32_type([2, 1, 3])],
        [f32_type([2, 3])],
        |builder, [x]| [squeeze::<3, 2>(builder, 1, x)],
    );

    let x = ctx.f32([2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(y, ctx.f32([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
}

#[test]
fn unsqueeze_inserts_singleton_dim() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "unsqueeze_dim1",
        [f32_type([2, 3])],
        [f32_type([2, 1, 3])],
        |builder, [x]| [unsqueeze::<2, 3>(builder, 1, x)],
    );

    let x = ctx.f32([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(y, ctx.f32([2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
}

#[test]
fn repeat_kv_repeats_heads() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "repeat_kv",
        [f32_type([1, 2, 1, 2])],
        [f32_type([1, 6, 1, 2])],
        |builder, [x]| [repeat_kv(builder, 3, x)],
    );

    let x = ctx.f32([1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(
        y,
        ctx.f32(
            [1, 6, 1, 2],
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        ),
    );
}

#[test]
fn repeat_interleave_repeats_dim() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "repeat_interleave_dim2",
        [f32_type([1, 1, 2, 2])],
        [f32_type([1, 1, 4, 2])],
        |builder, [x]| [repeat_interleave(builder, 2, 2, x)],
    );

    let x = ctx.f32([1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(
        y,
        ctx.f32([1, 1, 4, 2], vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]),
    );
}

#[test]
fn clamp_limits_values() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new("clamp", [f32_type([4])], [f32_type([4])], |builder, [x]| {
        [clamp(builder, x, -1.0, 1.5)]
    });

    let x = ctx.f32([4], vec![-2.0, -1.0, 0.5, 2.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(y, ctx.f32([4], vec![-1.0, -1.0, 0.5, 1.5]));
}

#[test]
fn avgpool2d_pools_blocks() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "avgpool2d_2x2",
        [f32_type([1, 1, 2, 2])],
        [f32_type([1, 1, 1, 1])],
        |builder, [x]| [avgpool2d(builder, 1, 2, 2, x)],
    );

    let x = ctx.f32([1, 1, 2, 2], vec![1.0, 3.0, 5.0, 7.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(y, ctx.f32([1, 1, 1, 1], vec![4.0]));
}

#[test]
fn where_broadcast_selects_values() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "where_broadcast",
        [f32_type([1, 1]), f32_type([1, 2]), f32_type([1, 2])],
        [f32_type([1, 2])],
        |builder, [mask, x, y]| [where_broadcast(builder, mask, x, y)],
    );

    let mask = ctx.f32([1, 1], vec![0.0]);
    let x = ctx.f32([1, 2], vec![1.0, 2.0]);
    let y = ctx.f32([1, 2], vec![5.0, 6.0]);
    let out = ctx.run1(module, vec![mask, x, y]);

    ctx.assert_f32_eq(out, ctx.f32([1, 2], vec![5.0, 6.0]));
}

#[test]
fn masked_fill_replaces_masked() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "masked_fill",
        [f32_type([2]), f32_type([2])],
        [f32_type([2])],
        |builder, [mask, x]| [masked_fill(builder, mask, 9.0, x)],
    );

    let mask = ctx.f32([2], vec![1.0, 0.0]);
    let x = ctx.f32([2], vec![1.0, 2.0]);
    let y = ctx.run1(module, vec![mask, x]);

    ctx.assert_f32_eq(y, ctx.f32([2], vec![9.0, 2.0]));
}

#[test]
fn cumsum_last_dim() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "cumsum_last_dim",
        [f32_type([1, 1, 1, 3])],
        [f32_type([1, 1, 1, 3])],
        |builder, [x]| [cumsum::<4>(builder, x)],
    );

    let x = ctx.f32([1, 1, 1, 3], vec![1.0, 2.0, 3.0]);
    let y = ctx.run1(module, vec![x]);

    ctx.assert_f32_eq(y, ctx.f32([1, 1, 1, 3], vec![1.0, 3.0, 6.0]));
}

#[test]
fn depthwise_conv1d_no_bias_param_matches_reference_values() {
    let ctx = TestCtx::new();
    let module = SimpleModule::new(
        "depthwise_conv1d_no_bias_param",
        [f32_type([1, 2, 4]), f32_type([2, 1, 3])],
        [f32_type([1, 2, 4])],
        |builder, [x, w]| [depthwise_conv1d_no_bias_param(builder, w, 3, 2, x)],
    );

    let x = ctx.f32([1, 2, 4], vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
    let w = ctx.f32([2, 1, 3], vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.0]);

    let y = ctx.run1(module, vec![x, w]);

    ctx.assert_f32_eq(
        y,
        ctx.f32(
            [1, 2, 4],
            vec![3.0, 8.0, 14.0, 20.0, 20.0, 30.0, 45.0, 60.0],
        ),
    );
}
