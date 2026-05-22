//! Integration tests for `xenon::overload` (19-overload §8.2).
//!
//! Covers: broadcast combos, scalar combos, type combos, and deep-copy
//! ownership verification. Owned tensor path is exercised here; `TensorView`
//! integration is validated transitively through unit tests in
//! `src/overload/arithmetic.rs` (W23T9 / W23T10).

use xenon::complex::Complex;
use xenon::overload::Scalar;
use xenon::tensor::Tensor;

// ─────────────────────────────────────────────────────────────
// Same-shape and broadcast combos
// ─────────────────────────────────────────────────────────────

#[test]
fn test_add_same_shape() {
    let left = Tensor::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    let right = Tensor::from_shape_vec([2, 3], vec![10, 20, 30, 40, 50, 60]).expect("valid test input");
    let result = (left + right).expect("broadcast succeeds");
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(
        result.as_slice().expect("contiguous"),
        &[11, 22, 33, 44, 55, 66]
    );
}

// Xenon stores tensors in F-order (column-major) per
// `00-coding.md §14 决策 1`. For shape=[2,3] data=[1,2,3,4,5,6]:
//   col 0 = [1,2], col 1 = [3,4], col 2 = [5,6]
//   logical matrix: [[1,3,5], [2,4,6]]
// Broadcasting [3] data=[10,20,30] places right[j] at every (i, j):
//   (0,0)=1+10=11, (1,0)=2+10=12,
//   (0,1)=3+20=23, (1,1)=4+20=24,
//   (0,2)=5+30=35, (1,2)=6+30=36
// F-order memory layout: [11, 12, 23, 24, 35, 36].
#[test]
fn test_add_broadcast() {
    let left = Tensor::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    let right = Tensor::from_shape_vec([3], vec![10, 20, 30]).expect("valid test input");
    let result = (&left + &right).expect("broadcast succeeds");
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(
        result.as_slice().expect("contiguous"),
        &[11, 12, 23, 24, 35, 36]
    );
}

#[test]
fn test_add_ref_ref() {
    let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
    let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
    let result = (&left + &right).expect("broadcast succeeds");
    assert_eq!(result.as_slice().expect("c"), &[4, 6]);
    assert_eq!(left.as_slice().expect("c"), &[1, 2]);
    assert_eq!(right.as_slice().expect("c"), &[3, 4]);
}

#[test]
fn test_broadcast_incompatible() {
    let left = Tensor::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    let right = Tensor::from_shape_vec([4], vec![10, 20, 30, 40]).expect("valid test input");
    assert!((&left + &right).is_err());
}

// ─────────────────────────────────────────────────────────────
// Scalar combos
// ─────────────────────────────────────────────────────────────

#[test]
fn test_right_scalar_combo() {
    let tensor = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
    assert_eq!((&tensor + 5.0f64).as_slice().expect("c"), &[6.0, 7.0]);
}

#[test]
fn test_scalar_wrapper_combo() {
    let tensor = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
    assert_eq!((Scalar(5.0) + &tensor).as_slice().expect("c"), &[6.0, 7.0]);
    // Owned + wrapper: create fresh tensor for ownership test.
    let t2 = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
    assert_eq!((Scalar(5.0) + t2).as_slice().expect("c"), &[6.0, 7.0]);
}

#[test]
fn test_native_left_scalar_combo_f64() {
    let tensor = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
    assert_eq!((5.0f64 + &tensor).as_slice().expect("c"), &[6.0, 7.0]);
    let t2 = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
    assert_eq!((5.0f64 + t2).as_slice().expect("c"), &[6.0, 7.0]);
}

#[test]
fn test_native_left_scalar_combo_i32() {
    let tensor = Tensor::from_shape_vec([2], vec![1i32, 2]).expect("valid test input");
    assert_eq!((5i32 + &tensor).as_slice().expect("c"), &[6, 7]);
    let t2 = Tensor::from_shape_vec([2], vec![1i32, 2]).expect("valid test input");
    assert_eq!((5i32 + t2).as_slice().expect("c"), &[6, 7]);
}

// ─────────────────────────────────────────────────────────────
// Non-commutative left-scalar verification (Sub / Div)
// ─────────────────────────────────────────────────────────────

#[test]
fn test_left_scalar_sub_noncommutative() {
    let t1 = Tensor::from_shape_vec([2], vec![3.0f64, 7.0]).expect("valid test input");
    let t2 = Tensor::from_shape_vec([2], vec![3.0f64, 7.0]).expect("valid test input");
    assert_eq!((Scalar(10.0) - t1).as_slice().expect("c"), &[7.0, 3.0]);
    assert_eq!((10.0f64 - t2).as_slice().expect("c"), &[7.0, 3.0]);
}

#[test]
fn test_left_scalar_div_noncommutative() {
    let t1 = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
    let t2 = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
    assert_eq!((Scalar(8.0) / t1).as_slice().expect("c"), &[4.0, 2.0]);
    assert_eq!((8.0f64 / t2).as_slice().expect("c"), &[4.0, 2.0]);
}

// ─────────────────────────────────────────────────────────────
// Type combos
// ─────────────────────────────────────────────────────────────

#[test]
fn test_i32_tensor() {
    let left = Tensor::from_shape_vec([2], vec![1i32, 2i32]).expect("valid test input");
    let right = Tensor::from_shape_vec([2], vec![3i32, 4i32]).expect("valid test input");
    let result = (&left + &right).expect("broadcast succeeds");
    assert_eq!(result.as_slice().expect("c"), &[4i32, 6i32]);
}

#[test]
fn test_complex_tensor() {
    let left = Tensor::from_shape_vec([2], vec![Complex::new(1.0f64, 0.0), Complex::new(2.0, 0.0)]).expect("valid test input");
    let right =
        Tensor::from_shape_vec([2], vec![Complex::new(3.0f64, 0.0), Complex::new(4.0, 0.0)]).expect("valid test input");
    let result = (&left + &right).expect("broadcast succeeds");
    assert_eq!(
        result.as_slice().expect("c"),
        &[Complex::new(4.0, 0.0), Complex::new(6.0, 0.0)]
    );
}

// ─────────────────────────────────────────────────────────────
// Sub / Mul / Div sanity
// ─────────────────────────────────────────────────────────────

#[test]
fn test_sub_mul_div_basic() {
    let a = Tensor::from_shape_vec([2], vec![8.0f64, 9.0]).expect("valid test input");
    let b = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");

    assert_eq!((&a - &b).expect("broadcast succeeds").as_slice().expect("c"), &[6.0, 6.0]);
    assert_eq!((&a * &b).expect("broadcast succeeds").as_slice().expect("c"), &[16.0, 27.0]);
    assert_eq!((&a / &b).expect("broadcast succeeds").as_slice().expect("c"), &[4.0, 3.0]);
}

// ─────────────────────────────────────────────────────────────
// Deep-copy verification
// ─────────────────────────────────────────────────────────────

#[test]
fn test_result_ownership_no_alias() {
    let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
    let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
    let result = (&left + &right).expect("broadcast succeeds");
    // Result tensor is independent of inputs
    assert_eq!(left.as_slice().expect("c"), &[1, 2]);
    assert_eq!(right.as_slice().expect("c"), &[3, 4]);
    let _ = result;
    // Both inputs remain accessible
    assert_eq!(left.as_slice().expect("c"), &[1, 2]);
}
