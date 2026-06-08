//! Public API implementations for reduction operations.
//!
//! This file contains the `impl TensorBase` blocks that define the public
//! reduction methods, delegating to the private implementations in `sum.rs`.

use crate::dimension::{Axis, Dimension, RemoveAxis};
use crate::element::Numeric;
use crate::error::XenonError;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

// ── Public API: sum / sum_axis_keepdims (D: Dimension) ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    /// Returns the sum of all logical elements. See `13-reduction.md §5.1`.
    ///
    /// Empty arrays return the additive identity `A::zero()`.
    /// Rank-0 (scalar) tensors return their single element.
    /// Integer overflow is unrecoverable and panics.
    pub fn sum(&self) -> A {
        crate::reduction::sum_all(self)
    }

    /// Reduces along `axis` and keeps the reduced axis with length 1.
    /// See `13-reduction.md §5.1`. For 0D tensors, every `axis` returns
    /// `XenonError::InvalidAxis` (no axis is valid at rank 0).
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    pub fn sum_axis_keepdims(&self, axis: Axis) -> Result<Tensor<A, D>, XenonError> {
        crate::reduction::sum_axis_keepdims_impl(self, axis)
    }
}

// ── Public API: sum_axis (D: Dimension + RemoveAxis) ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + RemoveAxis,
    A: Numeric + Copy + 'static,
{
    /// Reduces along `axis` and removes that axis from the output shape.
    /// See `13-reduction.md §5.1`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<A, D::Smaller>, XenonError> {
        crate::reduction::sum_axis_impl(self, axis)
    }
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use crate::complex::Complex;
    use crate::dimension::{Axis, Ix0, Ix1, Ix2};
    use crate::error::XenonError;
    #[cfg(any(feature = "simd", feature = "parallel"))]
    use crate::reduction::sum;
    use crate::tensor::{Tensor, Tensor1};

    // ── W18T2 tests ──

    #[test]
    fn test_sum_i32() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).expect("valid test input");
        assert_eq!(x.sum(), 6);
    }

    #[test]
    fn test_sum_empty() {
        let x = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid test input");
        assert_eq!(x.sum(), 0);
    }

    #[test]
    fn test_sum_scalar_rank0() {
        let x = Tensor::<i32, Ix0>::from_shape_vec(Ix0, vec![9]).expect("valid test input");
        assert_eq!(x.sum(), 9);
    }

    /// 13-reduction §7 Wave 1 T2 / §8.2: float NaN propagates per IEEE 754.
    #[test]
    fn test_sum_nan() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![1.0_f64, f64::NAN, 2.0])
            .expect("valid test input");
        assert!(x.sum().is_nan());
    }

    /// 13-reduction §7 Wave 1 T2 / §8.2: complex NaN propagates per component.
    /// §6.3 line 280-284: existence-only NaN check via `is_nan()`, not bit pattern.
    #[test]
    fn test_sum_complex_nan() {
        let x = Tensor1::from_shape_vec(
            Ix1(2),
            vec![
                Complex::<f64>::new(1.0, 2.0),
                Complex::<f64>::new(f64::NAN, 3.0),
            ],
        )
        .expect("valid test input");
        let result = x.sum();
        // Real component is NaN because one input had NaN real part.
        assert!(result.re.is_nan());
        // Imaginary component sum is 2 + 3 = 5, finite.
        assert_eq!(result.im, 5.0);
    }

    // ── W18T3 tests ──

    #[test]
    fn test_sum_axis_2d() {
        // F-order layout: axis-0 varies fastest; data[i + j*nrows].
        // Shape (2, 3): [[1,3,5], [2,4,6]] -> sum_axis(1) sums columns.
        let x = Tensor::<i32, Ix2>::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6])
            .expect("valid test input");
        let y = x.sum_axis(Axis(1)).expect("valid test input");
        assert_eq!(y.shape(), &[2]);
        assert_eq!(y.as_slice().expect("contiguous tensor"), &[9, 12]);
    }

    #[test]
    fn test_sum_axis_invalid_axis() {
        let x = Tensor::<i32, Ix2>::zeros((2, 3)).expect("valid test input");
        assert!(matches!(
            x.sum_axis(Axis(2)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }

    /// 13-reduction §7 T3 / §8.3 boundary row: zero-length reduced axis
    /// must produce a result whose output slots are all `A::zero()`.
    #[test]
    fn test_sum_axis_zero_len_axis() {
        let x = Tensor::<i32, Ix2>::from_shape_vec((0, 3), vec![]).expect("valid test input");
        let y = x.sum_axis(Axis(0)).expect("valid test input");
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.as_slice().expect("contiguous tensor"), &[0, 0, 0]);
    }

    // ── W18T4 tests ──

    #[test]
    fn test_sum_axis_keepdims() {
        // F-order: shape (2, 3) -> sum along axis 1 keeps dim 1 with length 1.
        // [[1,3,5], [2,4,6]] -> keepdims(1): [[9], [12]]
        let x = Tensor::<i32, Ix2>::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6])
            .expect("valid test input");
        let y = x.sum_axis_keepdims(Axis(1)).expect("valid test input");
        assert_eq!(y.shape(), &[2, 1]);
        assert_eq!(y.as_slice().expect("contiguous tensor"), &[9, 12]);
    }

    #[test]
    fn test_sum_axis_keepdims_invalid_axis() {
        let x = Tensor::<i32, Ix2>::zeros((2, 3)).expect("valid test input");
        assert!(matches!(
            x.sum_axis_keepdims(Axis(2)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }

    /// 13-reduction §7 T4 / §8.3: zero-length reduced axis preserves
    /// rank with the reduced axis length 1; all output slots are zero.
    #[test]
    fn test_sum_axis_keepdims_zero_len_axis() {
        let x = Tensor::<i32, Ix2>::from_shape_vec((0, 3), vec![]).expect("valid test input");
        let y = x.sum_axis_keepdims(Axis(0)).expect("valid test input");
        assert_eq!(y.shape(), &[1, 3]);
        assert_eq!(y.as_slice().expect("contiguous tensor"), &[0, 0, 0]);
    }
    // ── W18T5 dispatch consistency tests ──

    /// 13-reduction §6.3 line 264-275: f32 finite-value tolerance.
    #[cfg(any(feature = "simd", feature = "parallel"))]
    fn approx_eq_f32(actual: f32, expected: f32, n: usize, max_abs_input: f32) -> bool {
        let tol = (4.0 * f32::EPSILON * (n as f32) * max_abs_input).max(4.0 * f32::MIN_POSITIVE);
        (actual - expected).abs() <= tol
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_sum_simd_consistency() {
        // Length above the W14 SIMD threshold (1024) so the SIMD facade
        // actually executes rather than rejecting on short input.
        let n = 2048;
        let data: Vec<f32> = (0..n).map(|v| (v as f32) * 0.5).collect();
        let max_abs = data.last().copied().unwrap_or(0.0).abs();
        let x = Tensor1::from_shape_vec(Ix1(n), data).expect("valid test input");
        let dispatched = x.sum();
        let serial = sum::sum_serial(&x);
        assert!(
            approx_eq_f32(dispatched, serial, n, max_abs),
            "SIMD path {dispatched} vs serial {serial} exceeds §6.3 tolerance"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sum_parallel_consistency() {
        // Use f32 (not i32/i64) to bypass `force_scalar_for_integers` so the
        // parallel dispatch path actually executes. Length above
        // PARALLEL_THRESHOLD (65536) per 30-dispatch §5.6.
        let n = 100_000;
        let data: Vec<f32> = (0..n).map(|v| (v as f32) * 1e-3).collect();
        let max_abs = data.last().copied().unwrap_or(0.0).abs();
        let x = Tensor1::from_shape_vec(Ix1(n), data).expect("valid test input");
        let dispatched = x.sum();
        let serial = sum::sum_serial(&x);
        assert!(
            approx_eq_f32(dispatched, serial, n, max_abs),
            "parallel path {dispatched} vs serial {serial} exceeds §6.3 tolerance"
        );
    }
}
