//! Integration tests for reduction operations.
//!
//! Cross-API error-and-panic contract verification per 13-reduction.md §8.5.

use xenon::dimension::{Axis, Dimension, Ix1, Ix2, IxDyn};
use xenon::error::XenonError;
use xenon::tensor::{Tensor, Tensor1};

// ── Axis error contract — integration layer ──

#[test]
fn test_sum_axis_invalid_axis_integration() {
    let x = Tensor::<i32, Ix2>::zeros((2, 3)).expect("valid test input");
    assert!(matches!(
        x.sum_axis(Axis(2)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

#[test]
fn test_sum_axis_keepdims_invalid_axis_integration() {
    let x = Tensor::<i32, Ix2>::zeros((2, 3)).expect("valid test input");
    assert!(matches!(
        x.sum_axis_keepdims(Axis(2)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

// ── Integer overflow panic — all three public APIs ──

/// 13-reduction §10 line 501: integer overflow on global `sum` must panic.
#[test]
#[should_panic(expected = "integer overflow in reduction sum")]
fn test_sum_overflow_panic() {
    let x = Tensor1::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).expect("valid test input");
    let _ = x.sum();
}

/// 13-reduction §10 line 501: integer overflow on `sum_axis` must panic
/// with the same operation+element-index diagnostic as global `sum`.
#[test]
#[should_panic(expected = "integer overflow in reduction sum_axis")]
fn test_sum_axis_overflow_panic() {
    // Shape (2, 1): single column, two rows summed along axis 0.
    // F-order: element (0,0)=MAX, (1,0)=1 → sum_axis(0) adds both → overflow.
    let x =
        Tensor::<i32, Ix2>::from_shape_vec((2, 1), vec![i32::MAX, 1]).expect("valid test input");
    let _ = x.sum_axis(Axis(0));
}

/// 13-reduction §10 line 501: integer overflow on `sum_axis_keepdims`
/// must panic with the same contract; keepdims retains rank but shares
/// the accumulation path with `sum_axis`.
#[test]
#[should_panic(expected = "integer overflow in reduction sum_axis_keepdims")]
fn test_sum_axis_keepdims_overflow_panic() {
    let x =
        Tensor::<i32, Ix2>::from_shape_vec((2, 1), vec![i32::MAX, 1]).expect("valid test input");
    let _ = x.sum_axis_keepdims(Axis(0));
}

// ── IEEE 754 non-finite propagation ──

/// 13-reduction §8.2: Inf inputs follow IEEE 754, do not trigger panic.
#[test]
fn test_sum_inf() {
    let x = Tensor1::from_shape_vec(Ix1(3), vec![1.0_f64, f64::INFINITY, 2.0])
        .expect("valid test input");
    assert_eq!(x.sum(), f64::INFINITY);

    let neg = Tensor1::from_shape_vec(Ix1(2), vec![f64::INFINITY, f64::NEG_INFINITY])
        .expect("valid test input");
    // Inf + (-Inf) = NaN per IEEE 754.
    assert!(neg.sum().is_nan());
}

// ── High-rank IxDyn shape semantics ──

/// 13-reduction §8.2 line 407 / §8.3 line 420: high-rank IxDyn inputs
/// produce correct shape projection and keepdims semantics.
#[test]
fn test_sum_high_rank_ixdyn() {
    let dim = IxDyn::from_slice(&[2, 1, 3, 1, 1, 4]);
    let n = dim.checked_size().expect("valid test input");
    let data: Vec<i32> = (0..n as i32).collect();
    let x = Tensor::<i32, IxDyn>::from_shape_vec(dim.clone(), data).expect("valid test input");

    // sum_axis removes the reduced axis from the output shape.
    let reduced = x.sum_axis(Axis(5)).expect("valid test input");
    assert_eq!(reduced.shape(), &[2, 1, 3, 1, 1]);

    // sum_axis_keepdims preserves rank with axis length 1.
    let kept = x.sum_axis_keepdims(Axis(5)).expect("valid test input");
    assert_eq!(kept.shape(), &[2, 1, 3, 1, 1, 1]);

    // Element value sanity: reduced[0..] should equal sum along axis 5.
    let s: i32 = x.sum();
    let r: i32 = reduced.sum();
    assert_eq!(s, r, "sum-of-reduced must equal global sum");
}

// ── Large tensor parallel-path tolerance ──

/// 13-reduction §8.2 line 406: 10^7-class tensor parallel-path tolerance.
/// §6.3 line 266-275: `abs(actual - expected) <= max(4 * EPS * n * max_abs_input,
///                                                   4 * MIN_POSITIVE)`.
#[cfg(feature = "parallel")]
#[test]
fn test_sum_large_tensor_parallel_threshold() {
    let n: usize = 10_000_000;
    // Use small magnitudes to keep max_abs_input bounded and the tolerance tight.
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 1.0e-9).collect();
    let max_abs_input = data.last().copied().unwrap_or(0.0).abs();

    let x = Tensor1::from_shape_vec(Ix1(n), data.clone()).expect("valid test input");
    let parallel_result = x.sum();

    // Independent serial baseline (computed outside the library) for cross-check.
    let serial_baseline: f64 = data.iter().sum();

    // §6.3 tolerance scaled by 2x for parallel chunking: the split-accumulate
    // tree may introduce additional reordering beyond the serial baseline.
    let tol = (8.0 * f64::EPSILON * (n as f64) * max_abs_input).max(8.0 * f64::MIN_POSITIVE);
    assert!(
        (parallel_result - serial_baseline).abs() <= tol,
        "parallel sum {parallel_result} vs baseline {serial_baseline} exceeds §6.3 tolerance {tol}"
    );
}

// ── Additional reduction integration tests ──

use xenon::tensor::Tensor2;

/// Global sum of a 2D tensor.
#[test]
fn test_sum_global() {
    let x = Tensor2::<i32>::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    assert_eq!(x.sum(), 21);
}

/// Sum along a specific axis, removing that axis.
#[test]
fn test_sum_axis() {
    // F-order: shape (2,3) → matrix [[1,3,5],[2,4,6]].
    // sum_axis(0) sums rows: [3, 7, 11]
    let x = Tensor2::<i32>::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    let y = x.sum_axis(Axis(0)).expect("valid test input");
    assert_eq!(y.shape(), &[3]);
    assert_eq!(y.as_slice().expect("contiguous"), &[3, 7, 11]);
}

/// sum_axis_keepdims preserves the reduced axis with length 1.
#[test]
fn test_sum_keepdims() {
    let x = Tensor2::<i32>::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    let y = x.sum_axis_keepdims(Axis(1)).expect("valid test input");
    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(y.as_slice().expect("contiguous"), &[9, 12]);
}

/// Empty tensors return additive identity (zero).
#[test]
fn test_sum_empty() {
    let x = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid test input");
    assert_eq!(x.sum(), 0);
    let x2 = Tensor::<i32, Ix2>::from_shape_vec((0, 3), vec![]).expect("valid test input");
    assert_eq!(x2.sum(), 0);
}

/// NaN propagates through sum for float types.
#[test]
fn test_sum_nan() {
    let x = Tensor1::<f64>::from_shape_vec(Ix1(3), vec![1.0, f64::NAN, 2.0]).expect("valid test input");
    assert!(x.sum().is_nan());
}

/// Integer overflow in sum panics.
#[test]
#[should_panic(expected = "overflow")]
fn test_integer_sum_overflow_panics() {
    let x = Tensor1::<i32>::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).expect("valid test input");
    let _ = x.sum();
}
