#![cfg(feature = "candle-backend")]

use catgrad_core::category::core::Shape;
use catgrad_core::interpreter::backend::Backend;
use catgrad_core::interpreter::backend::candle::CandleBackend;
use catgrad_core::interpreter::{TaggedNdArray, TaggedNdArrayTuple, Value};

// ============================================================================
// CANDLE BACKEND UNIT TESTS
// ============================================================================
// These tests verify all Backend trait methods for the Candle backend

#[test]
fn test_candle_backend_basic_operations() {
    let backend = CandleBackend::new();

    // Test scalar creation
    let scalar: <CandleBackend as Backend>::NdArray<f32> = backend.scalar(42.0f32);
    // Note: Candle now creates a tensor with shape [] for scalars
    assert_eq!(scalar.0.shape().dims(), &[] as &[usize]);

    // Test zeros creation
    let zeros: <CandleBackend as Backend>::NdArray<f32> = backend.zeros::<f32>(Shape(vec![2, 3]));
    assert_eq!(zeros.0.shape().dims(), &[2, 3]);

    // Test tensor creation from slice
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();
    assert_eq!(tensor.0.shape().dims(), &[2, 2]);
}

#[test]
fn test_candle_backend_arithmetic() {
    let backend = CandleBackend::new();

    // Create two tensors
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
    let data2 = vec![2.0f32, 3.0, 4.0, 5.0];

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 2]))
        .unwrap();

    // Test addition
    let result = backend.add(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test multiplication
    let data3 = vec![1.0f32, 2.0, 3.0, 4.0];
    let data4 = vec![2.0f32, 3.0, 4.0, 5.0];
    let tensor3: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data3, Shape(vec![2, 2]))
        .unwrap();
    let tensor4: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data4, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.mul(TaggedNdArrayTuple::F32([tensor3, tensor4]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

#[test]
fn test_candle_backend_subtraction() {
    let backend = CandleBackend::new();

    // Test F32 subtraction
    let data1 = vec![10.0f32, 8.0, 6.0, 4.0];
    let data2 = vec![1.0f32, 2.0, 3.0, 4.0];

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.sub(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 subtraction
    let data3 = vec![10u32, 8, 6, 4];
    let data4 = vec![1u32, 2, 3, 4];

    let tensor3: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data3, Shape(vec![2, 2]))
        .unwrap();
    let tensor4: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data4, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.sub(TaggedNdArrayTuple::U32([tensor3, tensor4]));
    match result {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_max() {
    let backend = CandleBackend::new();

    // Test F32 max
    let data = vec![1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 3]))
        .unwrap();

    let result = backend.max(TaggedNdArray::F32([tensor]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 1]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 max
    let data_u32 = vec![1u32, 5, 3, 2];
    let tensor_u32: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data_u32, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.max(TaggedNdArray::U32([tensor_u32]));
    match result {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 1]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_sum() {
    let backend = CandleBackend::new();

    // Test F32 sum
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 3]))
        .unwrap();

    let result = backend.sum(TaggedNdArray::F32([tensor]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 1]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 sum
    let data_u32 = vec![1u32, 2, 3, 4, 5, 6];
    let tensor_u32: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data_u32, Shape(vec![2, 3]))
        .unwrap();

    let result = backend.sum(TaggedNdArray::U32([tensor_u32]));
    match result {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 1]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_matmul() {
    let backend = CandleBackend::new();

    // Create matrices for matmul: [2, 3] × [3, 2] = [2, 2]
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let data2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 3]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![3, 2]))
        .unwrap();

    let result = backend.matmul(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

#[test]
fn test_candle_backend_reshape() {
    let backend = CandleBackend::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 3]))
        .unwrap();

    let reshaped = backend.reshape(TaggedNdArray::F32([tensor]), Shape(vec![3, 2]));
    match reshaped {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[3, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

#[test]
fn test_candle_backend_cast() {
    let backend = CandleBackend::new();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();

    // Cast F32 to U32
    let casted = backend.cast(
        TaggedNdArray::F32([tensor]),
        catgrad_core::category::core::Dtype::U32,
    );
    match casted {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_division() {
    let backend = CandleBackend::new();

    // Test F32 division
    let data1 = vec![6.0f32, 8.0, 10.0, 12.0];
    let data2 = vec![2.0f32, 4.0, 5.0, 3.0];

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.div(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 division
    let data3 = vec![6u32, 8, 10, 12];
    let data4 = vec![2u32, 4, 5, 3];

    let tensor3: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data3, Shape(vec![2, 2]))
        .unwrap();
    let tensor4: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data4, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.div(TaggedNdArrayTuple::U32([tensor3, tensor4]));
    match result {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_power() {
    let backend = CandleBackend::new();

    // Test F32 power
    let data1 = vec![2.0f32, 3.0, 4.0, 5.0];
    let data2 = vec![2.0f32, 2.0, 2.0, 2.0];

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.pow(TaggedNdArrayTuple::F32([tensor1, tensor2]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 power
    let data3 = vec![2u32, 3, 4, 5];
    let data4 = vec![2u32, 2, 2, 2];

    let tensor3: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data3, Shape(vec![2, 2]))
        .unwrap();
    let tensor4: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data4, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.pow(TaggedNdArrayTuple::U32([tensor3, tensor4]));
    match result {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_negation() {
    let backend = CandleBackend::new();

    // Test F32 negation
    let data = vec![1.0f32, -2.0, 3.0, -4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.neg(TaggedNdArray::F32([tensor]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 negation
    let data_u32 = vec![1u32, 2, 3, 4];
    let tensor_u32: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data_u32, Shape(vec![2, 2]))
        .unwrap();

    let result = backend.neg(TaggedNdArray::U32([tensor_u32]));
    match result {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_broadcast() {
    let backend = CandleBackend::new();

    // Test F32 broadcasting
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();

    // Broadcast to add a dimension at the front: [2, 2] -> [1, 2, 2]
    let broadcasted = backend.broadcast(TaggedNdArray::F32([tensor]), Shape(vec![1, 2, 2]));
    match broadcasted {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[1, 2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Broadcast to expand the first dimension: [1, 2, 2] -> [5, 2, 2]
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![1, 2, 2]))
        .unwrap();
    let broadcasted = backend.broadcast(TaggedNdArray::F32([tensor]), Shape(vec![5, 2, 2]));
    match broadcasted {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[5, 2, 2]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test U32 broadcasting
    let data_u32 = vec![1u32, 2, 3, 4];
    let tensor_u32: <CandleBackend as Backend>::NdArray<u32> = backend
        .ndarray_from_slice(&data_u32, Shape(vec![2, 2]))
        .unwrap();

    // Broadcast to add multiple dimensions: [2, 2] -> [2, 1, 2, 2]
    let broadcasted_u32 =
        backend.broadcast(TaggedNdArray::U32([tensor_u32]), Shape(vec![2, 1, 2, 2]));
    match broadcasted_u32 {
        TaggedNdArray::U32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[2, 1, 2, 2]);
        }
        _ => panic!("Expected U32 result"),
    }
}

#[test]
fn test_candle_backend_broadcast_scalar() {
    let backend = CandleBackend::new();

    // Test broadcasting a scalar
    let scalar: <CandleBackend as Backend>::NdArray<f32> = backend.scalar(5.0f32);

    // Broadcast scalar to [3, 2] (concatenating [] and [3, 2])
    let broadcasted = backend.broadcast(TaggedNdArray::F32([scalar]), Shape(vec![3, 2]));
    match broadcasted {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[3, 2]);
        }
        _ => panic!("Expected F32 result"),
    }
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================
// These tests verify that shape mismatches are properly handled

#[test]
#[should_panic(expected = "Shape mismatch in operation")]
fn test_candle_backend_shape_mismatch_error() {
    let backend = CandleBackend::new();

    // Create tensors with different shapes
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let data2 = vec![1.0f32, 2.0, 3.0]; // [3] - different shape

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> =
        backend.ndarray_from_slice(&data2, Shape(vec![3])).unwrap();

    // This should panic due to shape mismatch
    let _result = backend.add(TaggedNdArrayTuple::F32([tensor1, tensor2]));
}

#[test]
#[should_panic(expected = "Shape mismatch in operation")]
fn test_candle_backend_scalar_tensor_mismatch_error() {
    let backend = CandleBackend::new();

    // Create a scalar and a tensor with different shapes
    let scalar: <CandleBackend as Backend>::NdArray<f32> = backend.scalar(5.0f32); // shape [1]
    let data = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![2, 2]))
        .unwrap();

    // This should panic due to shape mismatch - scalar [1] vs tensor [2, 2]
    let _result = backend.add(TaggedNdArrayTuple::F32([scalar, tensor]));
}

#[test]
#[should_panic(expected = "Shape mismatch in operation")]
fn test_candle_backend_multiplication_shape_mismatch_error() {
    let backend = CandleBackend::new();

    // Create tensors with different shapes
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let data2 = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2] - different shape

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 3]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 2]))
        .unwrap();

    // This should panic due to shape mismatch
    let _result = backend.mul(TaggedNdArrayTuple::F32([tensor1, tensor2]));
}

#[test]
#[should_panic(expected = "Shape mismatch in operation")]
fn test_candle_backend_division_shape_mismatch_error() {
    let backend = CandleBackend::new();

    // Create tensors with different shapes
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let data2 = vec![1.0f32, 2.0]; // [2] - different shape

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> =
        backend.ndarray_from_slice(&data2, Shape(vec![2])).unwrap();

    // This should panic due to shape mismatch
    let _result = backend.div(TaggedNdArrayTuple::F32([tensor1, tensor2]));
}

#[test]
#[should_panic(expected = "Shape mismatch in operation")]
fn test_candle_backend_power_shape_mismatch_error() {
    let backend = CandleBackend::new();

    // Create tensors with different shapes
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let data2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3] - different shape

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data2, Shape(vec![2, 3]))
        .unwrap();

    // This should panic due to shape mismatch
    let _result = backend.pow(TaggedNdArrayTuple::F32([tensor1, tensor2]));
}

#[test]
#[should_panic(expected = "Shape mismatch in operation")]
fn test_candle_backend_subtraction_shape_mismatch_error() {
    let backend = CandleBackend::new();

    // Create tensors with different shapes
    let data1 = vec![1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
    let data2 = vec![1.0f32, 2.0]; // [2] - different shape

    let tensor1: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data1, Shape(vec![2, 2]))
        .unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> =
        backend.ndarray_from_slice(&data2, Shape(vec![2])).unwrap();

    // This should panic due to shape mismatch
    let _result = backend.sub(TaggedNdArrayTuple::F32([tensor1, tensor2]));
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================
// These tests verify edge cases and boundary conditions

#[test]
fn test_candle_backend_empty_tensor() {
    let backend = CandleBackend::new();

    // Test zeros with empty shape (scalar)
    let scalar: <CandleBackend as Backend>::NdArray<f32> = backend.zeros::<f32>(Shape(vec![]));
    assert_eq!(scalar.0.shape().dims(), &[] as &[usize]);

    // Test zeros with single element
    let single: <CandleBackend as Backend>::NdArray<f32> = backend.zeros::<f32>(Shape(vec![1]));
    assert_eq!(single.0.shape().dims(), &[1]);
}

#[test]
fn test_candle_backend_single_element_operations() {
    let backend = CandleBackend::new();

    // Test operations on single-element tensors
    let data1 = vec![5.0f32];
    let data2 = vec![3.0f32];

    let tensor1: <CandleBackend as Backend>::NdArray<f32> =
        backend.ndarray_from_slice(&data1, Shape(vec![1])).unwrap();
    let tensor2: <CandleBackend as Backend>::NdArray<f32> =
        backend.ndarray_from_slice(&data2, Shape(vec![1])).unwrap();

    // Test all operations on single elements
    let add_result = backend.add(TaggedNdArrayTuple::F32([tensor1.clone(), tensor2.clone()]));
    let sub_result = backend.sub(TaggedNdArrayTuple::F32([tensor1.clone(), tensor2.clone()]));
    let mul_result = backend.mul(TaggedNdArrayTuple::F32([tensor1.clone(), tensor2.clone()]));
    let div_result = backend.div(TaggedNdArrayTuple::F32([tensor1.clone(), tensor2.clone()]));
    let pow_result = backend.pow(TaggedNdArrayTuple::F32([tensor1.clone(), tensor2]));

    // All results should have shape [1]
    for (name, result) in [
        ("add", add_result),
        ("sub", sub_result),
        ("mul", mul_result),
        ("div", div_result),
        ("pow", pow_result),
    ] {
        match result {
            TaggedNdArray::F32([arr]) => {
                assert_eq!(
                    arr.0.shape().dims(),
                    &[1],
                    "{} result should have shape [1]",
                    name
                );
            }
            _ => panic!("Expected F32 result for {}", name),
        }
    }

    // Test unary operations
    let neg_result = backend.neg(TaggedNdArray::F32([tensor1.clone()]));
    let max_result = backend.max(TaggedNdArray::F32([tensor1.clone()]));
    let sum_result = backend.sum(TaggedNdArray::F32([tensor1]));

    // Test negation (preserves shape)
    match neg_result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(
                arr.0.shape().dims(),
                &[1],
                "neg result should have shape [1]"
            );
        }
        _ => panic!("Expected F32 result for neg"),
    }

    // Test max and sum
    match max_result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(
                arr.0.shape().dims(),
                &[1],
                "max result should have shape []"
            );
        }
        _ => panic!("Expected F32 result for max"),
    }

    match sum_result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(
                arr.0.shape().dims(),
                &[1],
                "sum result should have shape []"
            );
        }
        _ => panic!("Expected F32 result for sum"),
    }
}

#[test]
fn test_candle_backend_large_tensor() {
    let backend = CandleBackend::new();

    // Test with a larger tensor to ensure operations scale
    let size = 100;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let tensor: <CandleBackend as Backend>::NdArray<f32> = backend
        .ndarray_from_slice(&data, Shape(vec![10, 10]))
        .unwrap();

    // Test that operations work on larger tensors
    let result = backend.add(TaggedNdArrayTuple::F32([tensor.clone(), tensor.clone()]));
    match result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[10, 10]);
        }
        _ => panic!("Expected F32 result"),
    }

    // Test reduction operations
    let sum_result = backend.sum(TaggedNdArray::F32([tensor.clone()]));
    let max_result = backend.max(TaggedNdArray::F32([tensor]));

    match sum_result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[10, 1]);
        }
        _ => panic!("Expected F32 result for sum"),
    }

    match max_result {
        TaggedNdArray::F32([arr]) => {
            assert_eq!(arr.0.shape().dims(), &[10, 1]);
        }
        _ => panic!("Expected F32 result for max"),
    }
}

// ============================================================================
// CANDLE INTERPRETER TESTS
// ============================================================================
// These tests verify that the Candle backend works correctly through the
// interpreter, including higher-level operations and model execution.

use catgrad_core::category::lang::*;
use catgrad_core::interpreter::{Interpreter, Parameters, tensor};
use catgrad_core::stdlib::nn::Exp;
use catgrad_core::stdlib::*;
use catgrad_core::{check, check::*};

pub mod test_models;
use test_models::{Add, BatchMatMul};

fn run_candle_test_with_inputs<F>(
    TypedTerm {
        term, source_type, ..
    }: TypedTerm,
    build_inputs: F,
) -> Vec<catgrad_core::interpreter::Value<CandleBackend>>
where
    F: FnOnce(&CandleBackend) -> Vec<catgrad_core::interpreter::Value<CandleBackend>>,
{
    // Get stdlib / environment
    let env = catgrad_core::stdlib::stdlib();

    // Typecheck
    let _result = check_with(
        &env,
        &check::Parameters::default(),
        term.clone(),
        source_type,
    )
    .unwrap();

    // Run interpreter
    let backend = CandleBackend::new();
    let interpreter: Interpreter<CandleBackend> =
        Interpreter::new(backend, env, Parameters::default());

    let values = build_inputs(&interpreter.backend);
    interpreter.run(term, values).unwrap()
}

#[test]
fn test_candle_interpreter_add() {
    let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // Shape (2, 1, 3)
    let result = run_candle_test_with_inputs(Add.term().unwrap(), |backend| {
        let input = tensor(backend, Shape(vec![2, 1, 3]), &data).unwrap();
        vec![input.clone(), input]
    });

    println!("Candle Interpreter result: {result:?}");

    // Create expected result (double the input data)
    let expected_data: Vec<u32> = data.iter().map(|&x| x * 2).collect();
    let backend = CandleBackend::new();
    let expected = tensor(&backend, Shape(vec![2, 1, 3]), &expected_data).unwrap();
    let backend = CandleBackend::new();
    match (&result[0], &expected) {
        (
            Value::NdArray(TaggedNdArray::U32([actual])),
            Value::NdArray(TaggedNdArray::U32([exp])),
        ) => {
            assert!(
                backend.compare(TaggedNdArrayTuple::U32([actual.clone(), exp.clone()])),
                "Result should be double the input data"
            );
        }
        _ => panic!("Expected U32 tensors"),
    }
}

#[test]
fn test_candle_interpreter_batch_matmul() {
    // Construct batch matmul inputs with shapes [2, 2, 2] × [2, 2, 1] = [2, 2, 1]
    // Batch 0: [[1, 2], [3, 4]] × [[1], [2]] = [[5], [11]]
    // Batch 1: [[5, 6], [7, 8]] × [[3], [4]] = [[39], [53]]
    let x0_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // batch 0
        5.0, 6.0, 7.0, 8.0, // batch 1
    ];
    let x1_data: Vec<f32> = vec![
        1.0, 2.0, // batch 0
        3.0, 4.0, // batch 1
    ];

    let result = run_candle_test_with_inputs(BatchMatMul.term().unwrap(), |backend| {
        let x0 = tensor(backend, Shape(vec![2, 2, 2]), &x0_data).unwrap();
        let x1 = tensor(backend, Shape(vec![2, 2, 1]), &x1_data).unwrap();
        vec![x0, x1]
    });

    let backend = CandleBackend::new();
    // Create expected result
    let expected_data: Vec<f32> = vec![
        5.0, 11.0, // batch 0: [1*1+2*2, 3*1+4*2]
        39.0, 53.0, // batch 1: [5*3+6*4, 7*3+8*4]
    ];
    let expected = tensor(&backend, Shape(vec![2, 2, 1]), &expected_data).unwrap();
    let backend = CandleBackend::new();
    match (&result[0], &expected) {
        (
            Value::NdArray(TaggedNdArray::F32([actual])),
            Value::NdArray(TaggedNdArray::F32([exp])),
        ) => {
            assert!(
                backend.compare(TaggedNdArrayTuple::F32([actual.clone(), exp.clone()])),
                "Batch matmul result should match expected output"
            );
        }
        _ => panic!("Expected F32 tensors"),
    }
}

// ============================================================================
// MISSING TEST COVERAGE - EXP FUNCTION
// ============================================================================
// Add exp function testing to match ndarray backend coverage

fn allclose_f32(a: &[f32], b: &[f32], rtol: f32, atol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&x, &y)| {
        let diff = (x - y).abs();
        diff <= atol + rtol * y.abs()
    })
}

#[test]
fn test_candle_interpreter_exp() {
    // Test with a simpler shape first to debug the issue
    let data: Vec<f32> = vec![0.0, 1.0]; // Shape (2,)
    let result = run_candle_test_with_inputs(Exp.term().unwrap(), |backend| {
        vec![tensor(backend, Shape(vec![2]), &data).unwrap()]
    });

    // make sure actual result is a single F32 array
    use catgrad_core::interpreter::{TaggedNdArray, Value};
    let actual = match &result[..] {
        [Value::NdArray(TaggedNdArray::F32([actual]))] => actual,
        xs => panic!("wrong output type: {xs:?}"),
    };

    // Create expected result (e^x for each element)
    let expected: Vec<f32> = data.iter().map(|&x| x.exp()).collect();

    assert!(
        allclose_f32(
            &actual.0.flatten_all().unwrap().to_vec1().unwrap(),
            &expected,
            1e-5,
            1e-8
        ),
        "actual should be close to expected!"
    );
}
