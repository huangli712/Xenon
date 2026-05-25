//! BLAS layout compatibility queries.
//!
//! See `docs/design/23-ffi.md` §5.10–§5.12 and §6.4 for the
//! authoritative semantics; `26-error.md` §5.1 defines the structured
//! error payloads.

use std::borrow::Cow;

use super::types::BlasInfo;
use crate::dimension::Dimension;
use crate::error::{FfiBackend, FfiErrorCategory, XenonError};
use crate::storage::Storage;
use crate::tensor::TensorBase;

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Checks whether the memory layout is BLAS-compatible.
    ///
    /// # BLAS Compatibility Conditions
    ///
    /// | Condition | Description |
    /// |-----------|-------------|
    /// | Contiguity | F-contiguous (Xenon only supports F-order) |
    /// | Positive strides | All strides > 0 |
    /// | No zero strides | No broadcast dimensions |
    ///
    /// # Returns
    ///
    /// `true` if the layout matches Xenon's BLAS memory-layout contract;
    /// `false` if a copy is needed first.
    ///
    /// This method **checks layout only**. Callers must still verify
    /// `ndim() == 2` and convert `rows`, `cols`, and `lda` to the
    /// BLAS/LAPACK backend integer type expected by the target
    /// implementation, typically by calling `blas_info()` and then
    /// `BlasInfo::as_blas_int()` on the exported metadata.
    pub fn is_blas_layout_compatible(&self) -> bool {
        self.is_f_contiguous() && !self.has_zero_stride()
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns BLAS layout identifier and parameter information.
    ///
    /// # Returns
    ///
    /// - `Ok(BlasInfo<A>)` — compatibility conditions met; `rows` /
    ///   `cols` / `leading_dim` are returned as raw `usize` metadata.
    /// - `Err(XenonError::Ffi { .. })` — returned when the tensor is not
    ///   2D, not BLAS compatible, or has zero rows (`shape[0] == 0`).
    ///
    /// Backend integer width conversion is NOT performed here; use
    /// `BlasInfo::as_blas_int::<I>(value)` to convert any of the returned
    /// `usize` sizes to `i32` / `i64` etc.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::Ffi` (`backend: FfiBackend::Blas`) with:
    /// - `FfiErrorCategory::InvalidRank { expected: 2, actual }` if the
    ///   tensor is not 2-D.
    /// - `FfiErrorCategory::BlasIncompatibleLayout { shape, strides }` if
    ///   the tensor is not F-contiguous, has a zero stride, or has zero
    ///   rows (`shape[0] == 0`, which would violate BLAS `lda >= max(1, rows)`).
    pub fn blas_info(&self) -> Result<BlasInfo<A>, XenonError> {
        // Gate 1: rank must be 2.
        if self.ndim() != 2 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::blas_info"),
                category: FfiErrorCategory::InvalidRank {
                    expected: 2,
                    actual: self.ndim(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        // Gate 2: layout must be F-contiguous without zero strides.
        if !self.is_blas_layout_compatible() {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::blas_info"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        // Gate 3: BLAS requires `lda >= max(1, rows)`. For F-order shape
        // [0, n] (zero-row matrix), `product(shape) == 0` ⇒ the layout
        // is still classified F_CONTIGUOUS by `06-layout.md` §5.11's
        // HAS_ZERO_STRIDE rule, and `is_blas_layout_compatible` returns
        // true. The naturally computed F-order `strides[1]` equals
        // `rows == 0`, which violates BLAS's `lda >= 1`. Reject zero-row
        // matrices here as a separate gate so the exported `leading_dim`
        // is always `>= max(1, rows)`.
        if rows == 0 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::blas_info"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        // Post `rows > 0` gate: F-order `strides[1] == rows >= 1`, so
        // `leading_dim` always satisfies BLAS's `lda >= max(1, rows)`.
        Ok(BlasInfo {
            data_ptr: self.as_ptr(),
            leading_dim: self.strides()[1],
            rows,
            cols,
        })
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns the leading dimension (only meaningful for 2D arrays).
    ///
    /// For F-order matrix `A[M, N]`, `LDA = stride[1]`.
    /// For zero-column matrices (`cols == 0 && rows > 0`), returns
    /// `stride[1]` (= rows for F-order) so that `lda >= max(1, rows)`
    /// is still satisfied.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)` — LDA of a BLAS-compatible 2D array.
    /// - `Err(XenonError::Ffi { .. })` — returned for non-BLAS-compatible
    ///   input, non-2D input, or zero-row matrices.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::Ffi` (`backend: FfiBackend::Blas`) with:
    /// - `FfiErrorCategory::InvalidRank { expected: 2, actual }` if the
    ///   tensor is not 2-D.
    /// - `FfiErrorCategory::BlasIncompatibleLayout { shape, strides }` if
    ///   the tensor is not F-contiguous, has a zero stride, or has zero
    ///   rows (`shape[0] == 0`).
    pub fn lda(&self) -> Result<usize, XenonError> {
        if self.ndim() != 2 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::lda"),
                category: FfiErrorCategory::InvalidRank {
                    expected: 2,
                    actual: self.ndim(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        if !self.is_blas_layout_compatible() {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::lda"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        // Mirror `blas_info()` rows-gate.
        if self.shape()[0] == 0 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::lda"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        Ok(self.strides()[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};
    use crate::layout::Strides;
    use crate::tensor::TensorView;

    /// Helper: build a TensorView<Ix2> via W8T7 `from_raw_parts`. Reusable
    /// in-test constructor that avoids depending on W22 (`Tensor::zeros` /
    /// `Tensor::from_shape_vec`); shape + canonical F-order strides only.
    fn make_view_f64_ix2<'a>(data: &'a [f64], shape: [usize; 2]) -> TensorView<'a, f64, Ix2> {
        // Canonical F-order strides for shape [m, n] are [1, m].
        let strides = [1_usize, shape[0]];
        // SAFETY: F-order canonical layout fits within data.len() = product(shape).
        unsafe {
            TensorView::<'a, f64, Ix2>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix2(shape[0], shape[1]),
                Strides::from_slice(&strides).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order layout")
    }

    fn make_view_f64_ix1<'a>(data: &'a [f64]) -> TensorView<'a, f64, Ix1> {
        // SAFETY: F-order 1-D over data.
        unsafe {
            TensorView::<'a, f64, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(data.len()),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order 1-D")
    }

    /// §8.2 test_is_blas_layout_compatible — F-order 2D tensor passes.
    #[test]
    fn test_is_blas_layout_compatible_f_order_passes() {
        let data = [0.0_f64; 12]; // 3 * 4 = 12
        let t = make_view_f64_ix2(&data, [3, 4]);
        assert!(t.is_blas_layout_compatible());
    }

    /// `is_blas_layout_compatible` must check layout only — `ndim != 2`
    /// alone does NOT make it return false; ndim is a caller-side check.
    #[test]
    fn test_is_blas_layout_compatible_layout_only_no_rank_check() {
        let data = vec![0.0_f64; 5];
        let t1 = make_view_f64_ix1(&data);
        // 1-D F-contiguous tensor passes the layout-only check.
        assert!(t1.is_blas_layout_compatible());
    }

    /// §8.2 test_blas_info_f_order — successful path returns correct
    /// rows/cols/leading_dim.
    #[test]
    fn test_blas_info_f_order_returns_correct_metadata() {
        let data = [0.0_f64; 12];
        let t = make_view_f64_ix2(&data, [3, 4]);
        let info = t.blas_info().expect("F-order 2D is BLAS compatible");
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 4);
        assert_eq!(info.leading_dim, 3); // F-order strides[1] == rows
        assert_eq!(info.data_ptr, t.as_ptr());
    }

    /// Gate 1: non-2D input ⇒ `InvalidRank`.
    #[test]
    fn test_blas_info_invalid_rank_for_1d() {
        let data = vec![0.0_f64; 5];
        let t = make_view_f64_ix1(&data);
        let Err(err) = t.blas_info() else {
            panic!("expected InvalidRank error, got Ok");
        };
        match err {
            XenonError::Ffi {
                category, backend, ..
            } => {
                assert!(matches!(backend, FfiBackend::Blas));
                match category {
                    FfiErrorCategory::InvalidRank { expected, actual } => {
                        assert_eq!(expected, 2);
                        assert_eq!(actual, 1);
                    },
                    other => panic!("unexpected category: {other:?}"),
                }
            },
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    /// Gate 3: zero-row matrix ⇒ `BlasIncompatibleLayout`.
    /// Empty storage + shape [0, 4] is a valid empty F-order layout that
    /// passes `validate_access_range` (len=0 early-return) but fails the
    /// rows==0 gate in `blas_info`.
    #[test]
    fn test_blas_info_zero_rows_rejected() {
        let data = Vec::<f64>::new();
        // SAFETY: shape=[0,4] product=0 → validate_access_range early-returns.
        // Strides = canonical F-order for [0,4] = [1, 0] (rows=0).
        let t = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                data.as_ptr(),
                0,                                                          // storage_len
                Ix2(0, 4),                                                  // shape
                Strides::from_slice(&[1_usize, 0]).expect("valid strides"), // strides
                0,                                                          // offset
            )
        }
        .expect("empty F-order [0, 4] should pass validation");
        let Err(err) = t.blas_info() else {
            panic!("expected BlasIncompatibleLayout error, got Ok");
        };
        match err {
            XenonError::Ffi {
                category: FfiErrorCategory::BlasIncompatibleLayout { .. },
                ..
            } => {},
            other => panic!("expected BlasIncompatibleLayout, got {other:?}"),
        }
    }

    /// §8.2 test_lda_f_order — F-order [3, 4] returns 3.
    #[test]
    fn test_lda_f_order() {
        let data = [0.0_f64; 12];
        let t = make_view_f64_ix2(&data, [3, 4]);
        assert_eq!(t.lda().expect("F-order BLAS-compatible"), 3);
    }

    /// Gate 2 placeholder: non-F-contiguous 2D should trigger
    /// `BlasIncompatibleLayout`. Constructed manually via `from_raw_parts`
    /// with C-order strides instead of relying on `transpose()` (W20T3) or
    /// `slice()` (W21T6).
    #[test]
    fn test_lda_non_f_contiguous_returns_incompatible() {
        let data = [0.0_f64; 12];
        // shape [3, 4] with C-order strides [4, 1] — NOT F-order.
        let strides = [4_usize, 1_usize];
        // SAFETY: shape product 12 == data.len(); strides cover [0, 11].
        let t = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix2(3, 4),
                Strides::from_slice(&strides).expect("valid strides"),
                0,
            )
        }
        .expect("valid layout (just non-F-order)");
        // is_blas_layout_compatible returns false because is_f_contiguous() is false.
        assert!(!t.is_blas_layout_compatible());
        // lda() must return BlasIncompatibleLayout (Gate 2).
        let Err(err) = t.lda() else {
            panic!("expected BlasIncompatibleLayout error, got Ok");
        };
        match err {
            XenonError::Ffi {
                category: FfiErrorCategory::BlasIncompatibleLayout { .. },
                ..
            } => {},
            other => panic!("expected BlasIncompatibleLayout, got {other:?}"),
        }
    }

    /// Gate 1: non-2D input to `lda()` triggers `InvalidRank` (mirrors
    /// `test_blas_info_invalid_rank_for_1d`).
    #[test]
    fn test_lda_non_2d_returns_invalid_rank() {
        let data = vec![0.0_f64; 5];
        let t = make_view_f64_ix1(&data);
        let Err(err) = t.lda() else {
            panic!("expected InvalidRank error, got Ok");
        };
        match err {
            XenonError::Ffi {
                category:
                    FfiErrorCategory::InvalidRank {
                        expected, actual, ..
                    },
                ..
            } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            },
            other => panic!("expected InvalidRank, got {other:?}"),
        }
    }
}
