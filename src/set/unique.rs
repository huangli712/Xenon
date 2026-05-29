//! Deduplication for tensor element values.
//!
//! `UniqueElement` is a sealed marker trait that enables the [`unique`]
//! operation on [`TensorBase`]. The supported element types are `i32`,
//! `i64`, `f32`, `f64`, [`Complex`]<f32>, and [`Complex`]<f64>.

use crate::dimension::{Dimension, Ix1};
use crate::complex::Complex;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Sealed trait for types that support the `unique` operation.
///
/// Reuses the shared `crate::private::Sealed` infrastructure
/// (defined in `03-element.md §5.8`). Implemented only inside
/// this crate for supported element types; `bool` does not
/// implement this trait.
pub trait UniqueElement: crate::private::Sealed + Element {
    /// Equality check used by `unique`.
    fn unique_eq(&self, other: &Self) -> bool;
}

impl UniqueElement for i32 {
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for i64 {
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for f32 {
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for f64 {
    fn unique_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl UniqueElement for Complex<f32> {
    fn unique_eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

impl UniqueElement for Complex<f64> {
    fn unique_eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

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
    Tensor::from_shape_vec(Ix1(out.len()), out).expect("unique output shape matches data length")
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
    use crate::set::UniqueElement as PubUniqueElement;

    #[test]
    fn test_set_module_exports_unique_element() {
        fn assert_unique<T: PubUniqueElement>() {}
        assert_unique::<i32>();
        assert_unique::<f64>();
    }

    #[test]
    fn test_unique_eq_basic_i32() {
        assert!(42_i32.unique_eq(&42));
        assert!(!0_i32.unique_eq(&1));
    }

    #[test]
    fn test_unique_eq_signed_zero_f32() {
        // Trait-method level check: `-0.0 == 0.0` via `UniqueElement::unique_eq`.
        // The §8.2 operation-level counterpart `test_unique_signed_zero_equal_f32`
        // is owned by W19T4.
        assert!((-0.0_f32).unique_eq(&0.0));
    }

    #[test]
    fn test_unique_eq_nan_f32() {
        // Trait-method level check: `NaN != NaN` via `UniqueElement::unique_eq`.
        // The §8.2 operation-level counterpart `test_unique_nan_preserved_f32`
        // is owned by W19T4.
        let nan = f32::NAN;
        assert!(!nan.unique_eq(&nan));
    }

    // ── W19T3: unique_impl operation-level tests ──

    use crate::dimension::{Ix1, Ix2, IxDyn};
    use crate::tensor::{Tensor, Tensor1};

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

    #[test]
    fn test_unique_empty() {
        let x = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![])
            .expect("test input shape matches data length");
        assert_eq!(unique_impl(&x).len(), 0);
    }

    // ── W19T4: operation-level float NaN / ±0.0 tests ──

    #[test]
    fn test_unique_nan_preserved_f32() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![f32::NAN, f32::NAN, 1.0])
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.iter().filter(|v| v.is_nan()).count(), 2);
        assert_eq!(y.iter().filter(|v| !v.is_nan()).count(), 1);
    }

    #[test]
    fn test_unique_signed_zero_equal_f32() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![-0.0_f32, 0.0])
            .expect("test input shape matches data length");
        assert_eq!(unique_impl(&x).len(), 1);
    }

    #[test]
    fn test_unique_nan_preserved_f64() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![f64::NAN, f64::NAN, 1.0])
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.iter().filter(|v| v.is_nan()).count(), 2);
        assert_eq!(y.iter().filter(|v| !v.is_nan()).count(), 1);
    }

    #[test]
    fn test_unique_signed_zero_equal_f64() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![-0.0_f64, 0.0])
            .expect("test input shape matches data length");
        assert_eq!(unique_impl(&x).len(), 1);
    }

    // ── W19T5: Complex<f32>/<f64> component-wise tests ──

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

    #[test]
    fn test_unique_complex_componentwise() {
        // `0+(-0)i` vs `(-0)+0i`: both components `==` as `0+0i`, deduplicated to one.
        let values = vec![Complex::new(0.0_f64, -0.0), Complex::new(-0.0_f64, 0.0)];
        let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 1);
    }

    #[test]
    fn test_unique_complex_nan_preserved() {
        // Covers §8.3 boundary test: complex `[1+NaNi, 1+NaNi]` returns length-2
        // result (because NaN components are unequal).
        // Any NaN component makes the complex values compare unequal; both retained.
        let values = vec![Complex::new(f64::NAN, 1.0), Complex::new(f64::NAN, 1.0)];
        let x = Tensor1::from_shape_vec(Ix1(values.len()), values)
            .expect("test input shape matches data length");
        let y = unique_impl(&x);
        assert_eq!(y.len(), 2);
        assert_eq!(y.iter().filter(|v| v.re.is_nan()).count(), 2);
    }

    // ── W19T6: remaining §8.2 in-module tests ──

    fn assert_set_eq_i32<S, D>(actual: &crate::tensor::TensorBase<S, D>, expected: &[i32])
    where
        S: crate::storage::Storage<Elem = i32>,
        D: crate::dimension::Dimension,
    {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for e in expected {
            assert!(
                actual.iter().any(|a| a == e),
                "missing element {e} in unique output"
            );
        }
    }

    #[test]
    fn test_unique_2d() {
        let x = Tensor::<i32, Ix2>::from_shape_vec((2, 3), vec![1, 2, 1, 3, 2, 3])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.ndim(), 1);
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

    #[test]
    fn test_unique_order_unspecified() {
        // Do not rely on concrete output order — only verify multiset equality.
        let x = Tensor1::from_shape_vec(Ix1(5), vec![2_i32, 1, 2, 3, 1])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

    #[test]
    fn test_unique_set_equality() {
        let x = Tensor1::from_shape_vec(Ix1(6), vec![3_i32, 1, 2, 3, 2, 1])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

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

    #[test]
    fn test_unique_single() {
        let x = Tensor1::from_shape_vec(Ix1(1), vec![42_i32])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[42]);
    }

    #[test]
    fn test_unique_all_same() {
        let x = Tensor1::from_shape_vec(Ix1(5), vec![7_i32; 5])
            .expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[7]);
    }

    #[test]
    fn test_unique_high_rank_ixdyn() {
        // 5D IxDyn input should still be logically flattened to 1D.
        let shape = vec![2_usize, 1, 2, 1, 2];
        let data: Vec<i32> = vec![1, 2, 1, 2, 3, 1, 3, 2];
        let x =
            Tensor::<i32, IxDyn>::from_shape_vec(crate::dimension::IxDyn::from_slice(&shape), data)
                .expect("test input shape matches data length");
        let y = x.unique();
        assert_eq!(y.ndim(), 1);
        assert_set_eq_i32(&y, &[1, 2, 3]);
    }

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

    #[test]
    fn test_unique_large_tensor_high_dup() {
        // §8.2 `10^7` scale belongs to W29 performance/stress tests; here we
        // run a lightweight variant (1024 elements, high duplication), verifying
        // semantic correctness of the main path under duplicated input without
        // slowing down unit tests.
        let n = 1024_usize;
        let data: Vec<i32> = (0..n as i32).map(|i| i % 4).collect();
        let x =
            Tensor1::from_shape_vec(Ix1(n), data).expect("test input shape matches data length");
        let y = x.unique();
        assert_set_eq_i32(&y, &[0, 1, 2, 3]);
    }
}
