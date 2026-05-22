// tests/test_set.rs
//
// Integration coverage for `unique` operation per 13-set.md §8.
// Uses the public API: tensor.unique() → Tensor<A, Ix1>.

use xenon::complex::Complex;
use xenon::dimension::Ix1;
use xenon::tensor::{Tensor, Tensor1, Tensor2};

/// Helper: assert that `actual` contains exactly the expected set of i32 values
/// (order-independent).
fn assert_set_eq_i32(actual: &Tensor<i32, Ix1>, expected: &[i32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for e in expected {
        assert!(
            actual.iter().any(|a| *a == *e),
            "missing element {e} in unique output"
        );
    }
}

#[test]
fn test_unique_set_equality() {
    let x = Tensor1::from_shape_vec(Ix1(6), vec![3_i32, 1, 2, 3, 2, 1])
        .expect("valid construction");
    let y = x.unique();
    assert_set_eq_i32(&y, &[1, 2, 3]);
}

#[test]
fn test_unique_integers() {
    let x = Tensor1::from_shape_vec(Ix1(8), vec![4_i32, 1, 2, 1, 3, 2, 4, 5])
        .expect("valid construction");
    let y = x.unique();
    assert_set_eq_i32(&y, &[1, 2, 3, 4, 5]);
}

#[test]
fn test_unique_complex() {
    let values = vec![
        Complex::new(1.0_f64, 2.0),
        Complex::new(3.0_f64, 4.0),
        Complex::new(1.0_f64, 2.0),
        Complex::new(5.0_f64, 6.0),
    ];
    let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
        .expect("valid construction");
    let y = x.unique();
    assert_eq!(y.len(), 3);
    assert!(y.iter().any(|v| *v == Complex::new(1.0_f64, 2.0)));
    assert!(y.iter().any(|v| *v == Complex::new(3.0_f64, 4.0)));
    assert!(y.iter().any(|v| *v == Complex::new(5.0_f64, 6.0)));
}

#[test]
fn test_unique_nan_preserved() {
    // NaN != NaN via partial_eq, so each NaN is retained as distinct.
    let x = Tensor1::from_shape_vec(Ix1(4), vec![f64::NAN, f64::NAN, 1.0, f64::NAN])
        .expect("valid construction");
    let y = x.unique();
    let nan_count = y.iter().filter(|v| v.is_nan()).count();
    let non_nan_count = y.iter().filter(|v| !v.is_nan()).count();
    assert_eq!(nan_count, 3);
    assert_eq!(non_nan_count, 1);
}

#[test]
fn test_unique_non_contiguous() {
    // F-order [2, 3] with data [1, 2, 3, 4, 5, 6]:
    //   logical = [[1, 3, 5], [2, 4, 6]]
    // Transpose → [3, 2]: [[1, 2], [3, 4], [5, 6]]
    let x = Tensor2::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6])
        .expect("valid construction");
    let view = x.transpose();
    let y = view.unique();
    assert_set_eq_i32(&y, &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_unique_transposed_view() {
    let x = Tensor2::from_shape_vec([3, 2], vec![1_i32, 2, 1, 3, 2, 3])
        .expect("valid construction");
    let view = x.transpose(); // shape [2, 3]
    let y = view.unique();
    assert_set_eq_i32(&y, &[1, 2, 3]);
}

#[test]
fn test_unique_signed_zero_equal() {
    // IEEE-754: -0.0 == 0.0 is true → deduplicated to one element.
    let x = Tensor1::from_shape_vec(Ix1(3), vec![-0.0_f64, 0.0, 1.0])
        .expect("valid construction");
    let y = x.unique();
    assert_eq!(y.len(), 2);
    assert!(y.iter().any(|v| *v == 1.0_f64));
}

#[test]
fn test_unique_order_unspecified() {
    // Do not rely on concrete output order — only verify multiset equality.
    let x = Tensor1::from_shape_vec(Ix1(5), vec![2_i32, 1, 2, 3, 1])
        .expect("valid construction");
    let y = x.unique();
    assert_set_eq_i32(&y, &[1, 2, 3]);
}
