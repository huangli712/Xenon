// tests/test_matrix.rs
//
// Integration coverage for `matrix::dot` per 12-matrix §8.5.
// W17T7 fulfils §8.2 high-priority tests at the integration layer
// where `pub(crate)` access is not required.

use xenon::XenonError;
use xenon::complex::Complex;
use xenon::dimension::{Ix1, Ix2};
use xenon::tensor::{Tensor, Tensor1};

/// 12-matrix §10.1 line 574-575: f64 dot tolerance.
/// Use 36× to accommodate SIMD accumulation order differences.
fn f64_dot_tolerance(n: usize, max_abs_a: f64, max_abs_b: f64) -> f64 {
    let ulp_term = 36.0 * f64::EPSILON * (n as f64) * max_abs_a * max_abs_b;
    let floor = 4.0 * f64::MIN_POSITIVE;
    ulp_term.max(floor)
}

/// 12-matrix §10.1 line 573: f32 dot tolerance.
#[allow(dead_code)]
fn f32_dot_tolerance(n: usize, max_abs_a: f32, max_abs_b: f32) -> f32 {
    let ulp_term = 8.0 * f32::EPSILON * (n as f32) * max_abs_a * max_abs_b;
    let floor = 4.0 * f32::MIN_POSITIVE;
    ulp_term.max(floor)
}

// ── Public API baseline ──

#[test]
fn test_dot_basic() {
    // §5.3 line 199-202 worked example: 1*4 + 2*5 + 3*6 = 32.
    let a = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(3), vec![4_i32, 5, 6]).expect("valid construction");
    assert_eq!(a.dot(&b).expect("valid construction"), 32_i32);
}

#[test]
fn test_dot_complex() {
    // §5.2 line 184-189 worked example: conj(1+2i) * (3+4i) = 11 - 2i.
    let a = Tensor1::from_shape_vec(Ix1(1), vec![Complex::<f64>::new(1.0, 2.0)])
        .expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(1), vec![Complex::<f64>::new(3.0, 4.0)])
        .expect("valid construction");
    assert_eq!(
        a.dot(&b).expect("valid construction"),
        Complex::<f64>::new(11.0, -2.0)
    );
}

// ── Error paths ──

#[test]
fn test_dot_shape_mismatch() {
    let a = Tensor1::from_shape_vec(Ix1(2), vec![1_i32, 2]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).expect("valid construction");
    let err = a.dot(&b).expect_err("must return error");
    assert!(matches!(err, XenonError::ShapeMismatch { .. }));
}

#[test]
fn test_dot_high_rank_invalid_argument() {
    // §8.3 line 484: high-rank input → InvalidArgument with full
    // diagnostic fields. Use 2-D to keep construction minimal.
    let a = Tensor::<i32, Ix2>::from_shape_vec((1, 1), vec![1_i32]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(1), vec![1_i32]).expect("valid construction");
    let err = a.dot(&b).expect_err("must return error");
    assert!(matches!(err, XenonError::InvalidArgument { .. }));
}

// ── Integer overflow panic ──

#[test]
#[should_panic(expected = "dot: integer overflow during multiplication")]
fn test_dot_int_overflow_mul() {
    let a = Tensor1::from_shape_vec(Ix1(1), vec![i32::MAX]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(1), vec![2_i32]).expect("valid construction");
    let _ = a.dot(&b).expect("valid construction");
}

#[test]
#[should_panic(expected = "dot: integer overflow during accumulation")]
fn test_dot_int_overflow_add() {
    // First mul fits; the running sum overflows on the second add.
    let a = Tensor1::from_shape_vec(Ix1(3), vec![i32::MAX, 1_i32, 1]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, i32::MAX, 1]).expect("valid construction");
    let _ = a.dot(&b).expect("valid construction");
}

// ── Boundary scenarios ──

#[test]
fn test_dot_empty() {
    let a = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
    let b = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
    assert_eq!(a.dot(&b).expect("valid construction"), 0.0_f64);
}

#[test]
fn test_dot_single_element() {
    let a = Tensor1::from_shape_vec(Ix1(1), vec![7_i32]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(1), vec![6_i32]).expect("valid construction");
    assert_eq!(a.dot(&b).expect("valid construction"), 42_i32);
}

// ── Non-finite values ──

#[test]
fn test_dot_nan_input() {
    // §10.2 line 604: any NaN element → NaN result.
    let a = Tensor1::from_shape_vec(Ix1(2), vec![1.0_f64, f64::NAN]).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(2), vec![2.0_f64, 3.0]).expect("valid construction");
    assert!(a.dot(&b).expect("valid construction").is_nan());
}

// ── Cross-path float tolerance ──

#[test]
fn test_dot_float_tolerance_across_paths() {
    // §10.1 line 574-575 + §8.6 line 511-512: regardless of feature
    // configuration, the observable result must fall within the
    // documented tolerance relative to a single-precision-fold scalar
    // baseline.
    let n: usize = 8192;
    let a_vals: Vec<f64> = (0..n).map(|i| ((i % 17) as f64) * 0.1 + 1.0).collect();
    let b_vals: Vec<f64> = (0..n).map(|i| ((i % 19) as f64) * 0.07 + 0.5).collect();
    let a = Tensor1::from_shape_vec(Ix1(n), a_vals.clone()).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(n), b_vals.clone()).expect("valid construction");

    let expected: f64 = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(0.0_f64, |acc, (x, y)| acc + x * y);

    let actual = a.dot(&b).expect("valid construction");

    let max_abs_a = a_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let max_abs_b = b_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = f64_dot_tolerance(n, max_abs_a, max_abs_b);

    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "dot diverges from scalar baseline: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
    );
}

// ── Feature-gate consistency ──

#[cfg(all(feature = "simd", feature = "parallel"))]
#[test]
fn test_dot_simd_parallel_combined_consistency() {
    // §8.2 line 472: SIMD + parallel combined path matches scalar
    // baseline within §10.1 tolerance.
    let n: usize = 16_384;
    let a_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1.0).collect();
    let b_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 2.0).collect();
    let a = Tensor1::from_shape_vec(Ix1(n), a_vals.clone()).expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(n), b_vals.clone()).expect("valid construction");

    let actual = a.dot(&b).expect("valid construction");
    let expected: f64 = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(0.0_f64, |acc, (x, y)| acc + x * y);

    let max_abs_a = a_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let max_abs_b = b_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = f64_dot_tolerance(n, max_abs_a, max_abs_b);

    assert!((actual - expected).abs() <= tol);
}

#[cfg(feature = "parallel")]
#[test]
fn test_dot_parallel_threshold_boundary() {
    // §8.2 line 473 + §8.3 line 486: threshold-adjacent inputs
    // maintain correct path selection and result semantics.
    for &n in &[4095_usize, 4096, 4097] {
        let a_vals: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) + 1.0).collect();
        let b_vals: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) + 1.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), a_vals.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), b_vals.clone()).expect("valid construction");

        let actual = a.dot(&b).expect("valid construction");
        let expected: f64 = a_vals
            .iter()
            .zip(b_vals.iter())
            .fold(0.0_f64, |acc, (x, y)| acc + x * y);

        let max_abs_a = a_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs_b = b_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance(n, max_abs_a, max_abs_b);

        assert!(
            (actual - expected).abs() <= tol,
            "threshold-boundary n={n}: actual={actual}, expected={expected}, tol={tol}"
        );
    }
}

// ── Additional integration tests for matrix::dot ──

/// Basic dot product: i32 3-element vectors (different arrangement from
/// the existing test_dot_basic, verifying consistent results).
#[test]
fn test_dot_product() {
    let a = Tensor1::from_shape_vec(Ix1(4), vec![2_i32, 3, 5, 7])
        .expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(4), vec![11_i32, 13, 17, 19])
        .expect("valid construction");
    // 2*11 + 3*13 + 5*17 + 7*19 = 22 + 39 + 85 + 133 = 279
    assert_eq!(a.dot(&b).expect("valid construction"), 279_i32);
}

/// i32 overflow panics: multiplication of large values triggers panic.
#[test]
#[should_panic(expected = "dot: integer overflow during multiplication")]
fn test_i32_dot_overflow_panics() {
    let a = Tensor1::from_shape_vec(Ix1(2), vec![i32::MAX, 1_i32])
        .expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(2), vec![2_i32, 1_i32])
        .expect("valid construction");
    let _ = a.dot(&b).expect("construction");
}

/// i64 overflow panics: accumulation overflow.
#[test]
#[should_panic(expected = "dot: integer overflow during accumulation")]
fn test_i64_dot_overflow_panics() {
    let a = Tensor1::from_shape_vec(Ix1(3), vec![i64::MAX, 1_i64, 1])
        .expect("valid construction");
    let b = Tensor1::from_shape_vec(Ix1(3), vec![1_i64, i64::MAX, 1])
        .expect("valid construction");
    let _ = a.dot(&b).expect("construction");
}
