//! Deduplication for tensor element values.
//!
//! `UniqueElement` is a sealed marker trait that enables the [`unique`]
//! operation on [`TensorBase`]. The supported element types are `i32`,
//! `i64`, `f32`, `f64`, [`Complex`]\<f32\>, and [`Complex`]\<f64\>.

use crate::dimension::{Dimension, Ix1};
use crate::complex::Complex;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Sealed trait for types that support the `unique` operation.
///
/// Implemented only inside this crate for supported element types;
/// `bool` does not implement this trait.
pub trait UniqueElement: crate::private::Sealed + Element {
    /// Equality check used by `unique`.
    fn unique_eq(&self, other: &Self) -> bool;
}

impl UniqueElement for i32 {
    /// Equality via normalised integer comparison.
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for i64 {
    /// Equality via normalised integer comparison.
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for f32 {
    /// Equality via IEEE 754 comparison
    /// (treats `-0.0` equal to `0.0`, `NaN` unequal to itself).
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for f64 {
    /// Equality via IEEE 754 comparison
    /// (treats `-0.0` equal to `0.0`, `NaN` unequal to itself).
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for Complex<f32> {
    /// Component-wise equality (NaN components cause inequality).
    fn unique_eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl UniqueElement for Complex<f64> {
    /// Component-wise equality (NaN components cause inequality).
    fn unique_eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

/// Internal deduplication logic: iterates element values, collecting
/// those that do not compare equal to any previously seen element.
/// Returns a 1D owned tensor of unique elements in encounter order.
pub(crate) fn unique_impl<S, D, A>(tensor: &TensorBase<S, D>) -> Tensor<A, Ix1>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: UniqueElement,
{
    let mut out = Vec::new();
    for value in tensor.iter().copied() {
        if !out.iter().any(|seen| value.unique_eq(seen)) {
            out.push(value);
        }
    }
    Tensor::from_shape_vec(Ix1(out.len()), out)
        .expect("unique output shape matches data length")
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: UniqueElement,
{
    /// Returns unique elements as a 1D owned tensor.
    ///
    /// See [`UniqueElement`] for supported types and equality semantics.
    pub fn unique(&self) -> Tensor<A, Ix1> {
        unique_impl(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2, IxDyn};
    use crate::tensor::{Tensor, Tensor1};

    /// Verifies `unique_eq` for `i32` — equal values return `true`,
    /// unequal return `false`.
    #[test]
    fn test_unique_eq_basic_i32() {
        assert!(42_i32.unique_eq(&42));
        assert!(!0_i32.unique_eq(&1));
    }

    /// Verifies `unique_eq` treats `-0.0` equal to `0.0` for `f32`.
    #[test]
    fn test_unique_eq_signed_zero_f32() {
        assert!((-0.0_f32).unique_eq(&0.0));
    }

    /// Verifies `unique_eq` treats `NaN` as unequal to itself for `f32`.
    #[test]
    fn test_unique_eq_nan_f32() {
        let nan = f32::NAN;
        assert!(!nan.unique_eq(&nan));
    }

    // -- unique_eq trait-level tests -----------------------------------------

    /// Verifies `unique` on a 1D `i32` tensor returns deduplicated elements.
    #[test]
    fn test_unique_basic_i32() {
        let x = Tensor1::from_shape_vec(Ix1(6), vec![3_i32, 1, 2, 1, 3, 2])
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 3);
        assert!(y.iter().any(|v| *v == 1));
        assert!(y.iter().any(|v| *v == 2));
        assert!(y.iter().any(|v| *v == 3));
    }

    /// Verifies `unique` on an empty tensor returns an empty tensor.
    #[test]
    fn test_unique_empty() {
        let x = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![])
            .expect("test input shape matches data length");
        assert_eq!(unique_impl(&x).len(), 0);
    }

    /// Verifies `unique` on `f32` tensor with `NaN` retains
    /// each distinct `NaN` entry.
    #[test]
    fn test_unique_nan_preserved_f32() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![f32::NAN, f32::NAN, 1.0])
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.iter().filter(|v| v.is_nan()).count(), 2);
        assert_eq!(y.iter().filter(|v| !v.is_nan()).count(), 1);
    }

    /// Verifies `unique` on `f32` tensor deduplicates `-0.0` and `0.0`
    /// as a single element.
    #[test]
    fn test_unique_signed_zero_equal_f32() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![-0.0_f32, 0.0])
            .expect("test input shape matches data length");
        assert_eq!(unique_impl(&x).len(), 1);
    }

    /// Verifies `unique` on `f64` tensor with `NaN` retains each distinct
    /// `NaN` entry.
    #[test]
    fn test_unique_nan_preserved_f64() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![f64::NAN, f64::NAN, 1.0])
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.iter().filter(|v| v.is_nan()).count(), 2);
        assert_eq!(y.iter().filter(|v| !v.is_nan()).count(), 1);
    }

    /// Verifies `unique` on `f64` tensor deduplicates `-0.0` and `0.0` as
    /// a single element.
    #[test]
    fn test_unique_signed_zero_equal_f64() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![-0.0_f64, 0.0])
            .expect("test input shape matches data length");
        assert_eq!(unique_impl(&x).len(), 1);
    }

    /// Verifies `unique` on `Complex<f64>` tensor with duplicate values
    /// returns deduplicated result.
    #[test]
    fn test_unique_basic_complex() {
        let values = vec![
            Complex::new(1.0_f64, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(1.0, 2.0),
        ];
        let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 2);
        assert!(y.iter().any(|v| *v == Complex::new(1.0_f64, 2.0)));
        assert!(y.iter().any(|v| *v == Complex::new(3.0_f64, 4.0)));
    }

    /// Verifies `unique` deduplicates `Complex` values with signed-zero
    /// components as equal.
    #[test]
    fn test_unique_complex_componentwise() {
        let values = vec![Complex::new(0.0_f64, -0.0), Complex::new(-0.0_f64, 0.0)];
        let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 1);
    }

    /// Verifies `unique` on `Complex` with `NaN` components retains all
    /// entries because `NaN` components cause inequality.
    #[test]
    fn test_unique_complex_nan_preserved() {
        let values = vec![Complex::new(f64::NAN, 1.0), Complex::new(f64::NAN, 1.0)];
        let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 2);
        assert_eq!(y.iter().filter(|v| v.re.is_nan()).count(), 2);
    }

    /// Verifies `unique` on `Complex<f32>` tensor with duplicate values
    /// returns deduplicated result.
    #[test]
    fn test_unique_basic_complex_f32() {
        let values = vec![
            Complex::new(1.0_f32, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(1.0, 2.0),
        ];
        let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 2);
        assert!(y.iter().any(|v| *v == Complex::new(1.0_f32, 2.0)));
        assert!(y.iter().any(|v| *v == Complex::new(3.0_f32, 4.0)));
    }

    // -- unique_impl internal tests ------------------------------------------

    /// Asserts that `actual` contains exactly the elements in `expected`
    /// as a multiset, regardless of order.
    fn assert_set_eq_i32<S, D>(actual: &TensorBase<S, D>, expected: &[i32])
    where
        S: Storage<Elem = i32>,
        D: Dimension,
    {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for e in expected {
            assert!(
                actual.iter().any(|a| a == e),
                "missing element {e} in unique output"
            );
        }
    }

    // -- unique() public API tests -------------------------------------------

    /// Verifies `unique` on a 2D tensor flattens to 1D with
    /// deduplicated elements.
    #[test]
    fn test_unique_2d() {
        let x = Tensor::<i32, Ix2>::from_shape_vec((2, 3), vec![1, 2, 1, 3, 2, 3])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.ndim(), 1);
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

    /// Verifies `unique` output contains the correct multiset of elements
    /// without depending on output order.
    #[test]
    fn test_unique_order_unspecified() {
        let x = Tensor1::from_shape_vec(Ix1(5), vec![2_i32, 1, 2, 3, 1])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

    /// Verifies `unique` on a tensor with all elements repeated produces
    /// the correct distinct set.
    #[test]
    fn test_unique_set_equality() {
        let x = Tensor1::from_shape_vec(Ix1(6), vec![3_i32, 1, 2, 3, 2, 1])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

    /// Verifies `unique` on `i64` tensor returns deduplicated elements.
    #[test]
    fn test_unique_basic_i64() {
        let x = Tensor1::from_shape_vec(Ix1(5), vec![1_i64, 2, 1, 3, 2])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.len(), 3);
        for e in [1_i64, 2, 3] {
            assert!(y.iter().any(|v| *v == e));
        }
    }

    /// Verifies `unique` on `f32` tensor returns deduplicated elements.
    #[test]
    fn test_unique_basic_f32() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![1.0_f32, 2.0, 1.0])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.len(), 2);
        for e in [1.0_f32, 2.0] {
            assert!(y.iter().any(|v| *v == e));
        }
    }

    /// Verifies `unique` on `f64` tensor returns deduplicated elements.
    #[test]
    fn test_unique_basic_f64() {
        let x = Tensor1::from_shape_vec(Ix1(4), vec![1.0_f64, 2.0, 1.0, 2.0])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.len(), 2);
        for e in [1.0_f64, 2.0] {
            assert!(y.iter().any(|v| *v == e));
        }
    }

    /// Verifies `unique` on a single-element tensor returns that element.
    #[test]
    fn test_unique_single() {
        let x = Tensor1::from_shape_vec(Ix1(1), vec![42_i32])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[42]);
    }

    /// Verifies `unique` on a tensor where all elements are identical
    /// returns one element.
    #[test]
    fn test_unique_all_same() {
        let x = Tensor1::from_shape_vec(Ix1(5), vec![7_i32; 5])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[7]);
    }

    /// Verifies `unique` on a 5D `IxDyn` tensor flattens to 1D with
    /// deduplicated elements.
    #[test]
    fn test_unique_high_rank_ixdyn() {
        let shape = vec![2_usize, 1, 2, 1, 2];
        let data: Vec<i32> = vec![1, 2, 1, 2, 3, 1, 3, 2];
        let x =
            Tensor::<i32, IxDyn>::from_shape_vec(IxDyn::from_slice(&shape), data)
                .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.ndim(), 1);
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

    /// Verifies `unique` on `i64` tensor with `MIN` and `MAX` values
    /// returns deduplicated result.
    #[test]
    fn test_unique_extreme_i64_values() {
        let x =
            Tensor1::from_shape_vec(Ix1(5), vec![i64::MIN, i64::MAX, 0_i64, i64::MIN, i64::MAX])
                .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.len(), 3);
        assert!(y.iter().any(|v| *v == i64::MIN));
        assert!(y.iter().any(|v| *v == i64::MAX));
        assert!(y.iter().any(|v| *v == 0));
    }

    /// Verifies `unique` on a large tensor with high duplication returns
    /// the correct distinct set.
    #[test]
    fn test_unique_large_tensor_high_dup() {
        let n = 1024_usize;
        let data: Vec<i32> = (0..n as i32).map(|i| i % 4).collect();
        let x = Tensor1::from_shape_vec(Ix1(n), data)
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[0, 1, 2, 3]);
    }

    /// Verifies `unique` on a tensor view returns deduplicated elements.
    #[test]
    fn test_unique_on_view() {
        let x = Tensor1::from_shape_vec(Ix1(4), vec![1_i32, 2, 1, 2])
            .expect("test input shape matches data length");
        let v = x.view();
        let y = v.unique();
        assert_set_eq_i32(&y, &[1, 2]);
    }
}
