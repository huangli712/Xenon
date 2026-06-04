//! FFI pointer-export API, multi-dimensional index → element offset /
//! pointer helpers, and BLAS layout compatibility queries.
//!
//! All entry points are inherent methods on `TensorBase`: `export` /
//! `export_mut` (raw descriptor export), `try_offset_of` / `try_ptr_at`
//! (index → offset / pointer), and `is_blas_layout_compatible` /
//! `blas_info` / `lda` (BLAS layout queries).

use core::ptr::NonNull;
use std::borrow::Cow;

use crate::error::{FfiBackend, FfiErrorCategory};
use crate::error::{InvalidLayoutReason, StorageKindTag, XenonError};
use crate::dimension::Dimension;
use crate::element::{Element, element_type_of};
use crate::storage::{Storage, StorageMut};
use crate::tensor::{StorageKind, StorageSemantics};

use super::types::{BlasInfo, TensorExportMutRaw, TensorExportRaw};

// `TensorBase` is imported so the `impl TensorBase { ... }` blocks below
// resolve. The public FFI re-export of `TensorBase` / `OwnedRawParts`
// (stable `crate::ffi::*` path) lives in `mod.rs`, sourced directly from
// `crate::tensor`. The raw-parts constructors (`from_raw_parts` /
// `from_raw_parts_mut` / `into_raw_parts` / `from_raw_parts_owned`) are
// inherent methods on `TensorBase` and need no separate import here.
use crate::tensor::TensorBase;

// ── Index → offset / pointer ─────────────────────────────────

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A> + StorageSemantics,
    D: Dimension,
{
    /// Converts a multi-dimensional index to an element offset relative
    /// to the **logical first element pointer** (`self.as_ptr()`).
    ///
    /// `offset = Σ(stride[i] * index[i])` for all `i` in `[0, ndim)`.
    ///
    /// Both multiplication and accumulation use checked arithmetic, and
    /// any overflow is reported as a recoverable error rather than panic
    /// or wraparound.
    ///
    /// # Errors
    ///
    /// | Failure | Variant |
    /// |---------|---------|
    /// | `index.len() != self.ndim()` | `XenonError::DimensionMismatch` |
    /// | `index[i] >= shape[i]` | `XenonError::IndexOutOfBounds` |
    /// | `strides[i] * index[i]` or accumulator overflows `usize` | `XenonError::InvalidLayout { reason: AccessRangeExceedsStorage }` |
    pub fn try_offset_of(&self, index: &[usize]) -> Result<usize, XenonError> {
        if index.len() != self.ndim() {
            return Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("ffi::try_offset_of"),
                expected: self.ndim(),
                actual: index.len(),
            });
        }
        let shape = self.shape();
        let strides = self.strides();
        // Build storage_kind tag once: tensor's StorageKind maps 1:1 to
        // StorageKindTag and is needed by every InvalidLayout branch.
        let storage_kind: StorageKindTag = match self.storage_kind() {
            StorageKind::Owned => StorageKindTag::Owned,
            StorageKind::View => StorageKindTag::View,
            StorageKind::ViewMut => StorageKindTag::ViewMut,
            StorageKind::Shared => StorageKindTag::Shared,
        };
        let mut offset: usize = 0;
        for axis in 0..self.ndim() {
            if index[axis] >= shape[axis] {
                return Err(XenonError::IndexOutOfBounds {
                    operation: Cow::Borrowed("ffi::try_offset_of"),
                    attempted_index: index.to_vec(),
                    axis,
                    shape: shape.to_vec(),
                });
            }
            let term = strides[axis].checked_mul(index[axis]).ok_or_else(|| {
                XenonError::InvalidLayout {
                    operation: Cow::Borrowed("ffi::try_offset_of"),
                    storage_kind,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    offset: self.offset(),
                    storage_len: self.storage_len(),
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                }
            })?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| XenonError::InvalidLayout {
                    operation: Cow::Borrowed("ffi::try_offset_of"),
                    storage_kind,
                    shape: shape.to_vec(),
                    strides: strides.to_vec(),
                    offset: self.offset(),
                    storage_len: self.storage_len(),
                    reason: InvalidLayoutReason::AccessRangeExceedsStorage,
                })?;
        }
        Ok(offset)
    }

    /// Converts a multi-dimensional index to a raw pointer to the
    /// corresponding element.
    ///
    /// # Errors
    ///
    /// Propagates every error returned by [`try_offset_of`]:
    /// - `XenonError::DimensionMismatch` if `index.len() != self.ndim()`.
    /// - `XenonError::IndexOutOfBounds` if `index[i] >= shape[i]` for any axis.
    /// - `XenonError::InvalidLayout { reason: AccessRangeExceedsStorage }` if
    ///   `strides[i] * index[i]` or the accumulator overflows `usize`.
    ///
    /// [`try_offset_of`]: Self::try_offset_of
    ///
    /// # Safety
    ///
    /// Returns a raw pointer; the caller is responsible for ensuring the
    /// pointer is dereferenced only while `self` remains valid. The
    /// offset is validated to be within the tensor's metadata range
    /// (no zero-stride / no overflow / no out-of-bounds index), but
    /// pointer dereference safety remains the caller's responsibility
    /// at the FFI boundary.
    pub fn try_ptr_at(&self, index: &[usize]) -> Result<*const A, XenonError> {
        let offset = self.try_offset_of(index)?;
        // SAFETY: `try_offset_of` validates `index < shape` per axis;
        // for a well-constructed tensor the resulting offset always lies
        // within the storage range covered by `as_ptr() ..
        // as_ptr().add(storage_len - self.offset())`.
        Ok(unsafe { self.as_ptr().add(offset) })
    }
}

// ── BLAS layout ───────────────────────────────────────────────

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
                
            });
        }
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        // Gate 3: BLAS requires `lda >= max(1, rows)`. For F-order shape
        // [0, n] (zero-row matrix), `product(shape) == 0`, so the layout
        // is still classified F-contiguous and `is_blas_layout_compatible`
        // returns true. The naturally computed F-order `strides[1]` equals
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
                
            });
        }
        Ok(self.strides()[1])
    }
}

// ── Export API ────────────────────────────────────────────────

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Export tensor data as a C-compatible raw structure.
    ///
    /// The returned `TensorExportRaw` borrows the tensor's data and
    /// metadata. The consumer must ensure the tensor outlives the export.
    /// This method does not fail; it always returns a valid export.
    ///
    /// `data` always carries the **storage base pointer**; the logical
    /// first element address is derived from `data.add(offset)` for
    /// non-empty tensors. Empty tensors (`len() == 0`) get a valid
    /// aligned `dangling` pointer that must not be dereferenced;
    /// `shape`, `strides`, and `offset` still describe the empty
    /// tensor metadata.
    ///
    /// **Return type**: this is the public FFI entry returning
    /// `TensorExportRaw`, the C-visible non-generic raw descriptor, built
    /// directly from the source tensor.
    pub fn export(&self) -> TensorExportRaw {
        let data = if self.is_empty() {
            // Empty tensor: a valid aligned non-dereferenceable pointer.
            // Do NOT call as_storage_ptr() — the backing storage may be
            // empty or even unallocated (e.g. zero-cap Vec).
            NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            self.as_storage_ptr()
        };
        TensorExportRaw {
            data: data.cast(),
            element_type: element_type_of::<A>() as u8,
            ndim: self.ndim(),
            shape: self.shape().as_ptr(),
            strides: self.strides().as_ptr(),
            storage_len: self.storage_len(),
            offset: self.offset(),
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Export tensor data with mutable access.
    ///
    /// **Return type**: public FFI entry returning `TensorExportMutRaw`,
    /// the C-visible non-generic raw descriptor, built directly from the
    /// source tensor.
    ///
    /// This API is only implemented for writable storage, so read-only
    /// storage modes are rejected at the trait boundary rather than at
    /// runtime. No additional fallible validation is performed beyond the
    /// existing `&mut self` + `S: StorageMut` exclusivity boundary.
    pub fn export_mut(&mut self) -> TensorExportMutRaw {
        let data = if self.is_empty() {
            NonNull::<A>::dangling().as_ptr()
        } else {
            self.as_storage_mut_ptr()
        };
        // Capture metadata after computing `data` (raw pointers hold no borrow).
        let element_type = element_type_of::<A>() as u8;
        let ndim = self.ndim();
        let shape = self.shape().as_ptr();
        let strides = self.strides().as_ptr();
        let storage_len = self.storage_len();
        let offset = self.offset();
        TensorExportMutRaw {
            data: data.cast(),
            element_type,
            ndim,
            shape,
            strides,
            storage_len,
            offset,
        }
    }
}

/// Doctest enforcing that `export_mut` cannot be called on a read-only view.
///
/// ```compile_fail
/// use xenon::tensor::Tensor;
/// use xenon::dimension::Ix1;
/// let t: Tensor<i32, Ix1> = Tensor::from_shape_vec([2], vec![1, 2]).unwrap();
/// let v = t.view();
/// // `view()` returns a `ViewRepr`-backed TensorBase, which is `Storage`
/// // but NOT `StorageMut`. `export_mut` must therefore fail to compile.
/// let _ = (&v).export_mut();
/// ```
fn _doctest_export_mut_rejects_view() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};
    use crate::element::ElementType;
    use crate::layout::Strides;
    use crate::tensor::{TensorView, TensorViewMut};

    // ── Helpers ─────────────────────────────────────────────────

    /// Helper: build a `TensorView<Ix2>` over a slice via `from_raw_parts`.
    fn make_view_ix2<'a>(
        data: &'a [i32],
        shape: [usize; 2],
        strides: [usize; 2],
    ) -> TensorView<'a, i32, Ix2> {
        // SAFETY: shape/strides describe an F-order layout fully within data.len().
        unsafe {
            TensorView::<'a, i32, Ix2>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix2(shape[0], shape[1]),
                Strides::from_slice(&strides).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order layout")
    }

    /// Helper: build a `TensorView<Ix2>` via `from_raw_parts` from a shape
    /// plus canonical F-order strides (no owning-constructor dependency).
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

    /// Helper: build a 1-D `TensorView<f64, Ix1>` over a slice with unit stride.
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

    // ── BLAS tests ──────────────────────────────────────────────

    /// F-order 2D tensor passes the BLAS layout check.
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

    /// Success path returns correct rows / cols / leading_dim.
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

    /// F-order [3, 4] returns lda 3.
    #[test]
    fn test_lda_f_order() {
        let data = [0.0_f64; 12];
        let t = make_view_f64_ix2(&data, [3, 4]);
        assert_eq!(t.lda().expect("F-order BLAS-compatible"), 3);
    }

    /// Gate 2: non-F-contiguous 2D triggers `BlasIncompatibleLayout`.
    /// Constructed manually via `from_raw_parts` with C-order strides.
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

    // ── Offset/pointer tests ────────────────────────────────────

    /// F-order [2, 3] tensor, strides=[1,2]; index [1, 2] yields
    /// 1*1 + 2*2 = 5.
    #[test]
    fn test_try_offset_of_various() {
        let data = vec![1_i32, 2, 3, 4, 5, 6];
        let t = make_view_ix2(&data, [2, 3], [1, 2]);
        assert_eq!(t.try_offset_of(&[0, 0]).expect("valid index"), 0);
        assert_eq!(t.try_offset_of(&[1, 2]).expect("valid index"), 5);
        // Out-of-bounds along axis 0.
        let Err(err) = t.try_offset_of(&[2, 0]) else {
            panic!("expected IndexOutOfBounds error, got Ok");
        };
        match err {
            XenonError::IndexOutOfBounds {
                axis,
                attempted_index,
                shape,
                ..
            } => {
                assert_eq!(axis, 0);
                assert_eq!(attempted_index, vec![2, 0]);
                assert_eq!(shape, vec![2, 3]);
            },
            other => panic!("expected IndexOutOfBounds, got {other:?}"),
        }
    }

    /// Rank mismatch ⇒ `DimensionMismatch`.
    #[test]
    fn test_try_offset_of_rank_mismatch() {
        let data = vec![1_i32, 2, 3, 4, 5, 6];
        let t = make_view_ix2(&data, [2, 3], [1, 2]);
        let Err(err) = t.try_offset_of(&[0]) else {
            panic!("expected DimensionMismatch error, got Ok");
        };
        match err {
            XenonError::DimensionMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            },
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    /// Checked arithmetic on `strides * index` overflow must yield
    /// `InvalidLayout { reason: AccessRangeExceedsStorage }`, never panic.
    ///
    /// Construction relies on `validate_access_range` early-returning for
    /// an empty tensor (`shape=[3, 0]`, len 0), which lets the view carry
    /// `stride[0] = usize::MAX` past construction. `try_offset_of(&[2, 0])`
    /// then computes `usize::MAX * 2` on axis 0, overflowing `checked_mul`.
    #[test]
    fn test_try_offset_of_checked_overflow() {
        let data = Vec::<i32>::new(); // empty storage
        // SAFETY: shape=[3,0] has product 0 → validate_access_range early-returns
        // at the len==0 gate, bypassing the stride>isize::MAX check.
        // The huge stride[0] propagates into the constructed view, where
        // try_offset_of will hit the checked_mul overflow path.
        let t = unsafe {
            TensorView::<i32, Ix2>::from_raw_parts(
                data.as_ptr(),                                                 // ptr: *const i32
                data.len(), // storage_len: usize = 0
                Ix2(3, 0),  // shape: Ix2
                Strides::from_slice(&[usize::MAX, 1]).expect("valid strides"), // strides: Strides<Ix2>
                0,                                                             // offset: usize
            )
        }
        .expect(
            "shape=[3, 0] with size=0 should pass \
             validate_access_range's empty-tensor early-return path",
        );

        // Now try_offset_of with index=[2, 0]:
        //   axis 0: 2 < 3 ✓ → term = usize::MAX * 2 → checked_mul OVERFLOW
        //   Expected: InvalidLayout { reason: AccessRangeExceedsStorage }
        let Err(err) = t.try_offset_of(&[2, 0]) else {
            panic!("expected InvalidLayout error, got Ok");
        };
        match err {
            XenonError::InvalidLayout { reason, .. } => {
                assert!(matches!(
                    reason,
                    InvalidLayoutReason::AccessRangeExceedsStorage
                ));
            },
            other => panic!("expected InvalidLayout, got {other:?}"),
        }
    }

    /// Pointer dereference yields the element at the indexed position.
    #[test]
    fn test_try_ptr_at_various() {
        let data = [10_i32, 20, 30];
        // SAFETY: F-order [3] view over data.
        let t = unsafe {
            TensorView::<i32, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(3),
                Strides::from_slice(&[1]).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order [3]");
        let ptr0 = t.try_ptr_at(&[0]).expect("valid index");
        let ptr1 = t.try_ptr_at(&[1]).expect("valid index");
        let ptr2 = t.try_ptr_at(&[2]).expect("valid index");
        // SAFETY: pointers borrow t for its lifetime; data outlives t.
        unsafe {
            assert_eq!(*ptr0, 10);
            assert_eq!(*ptr1, 20);
            assert_eq!(*ptr2, 30);
        }
        // Out-of-bounds propagates `IndexOutOfBounds`.
        let Err(err) = t.try_ptr_at(&[3]) else {
            panic!("expected IndexOutOfBounds error, got Ok");
        };
        assert!(matches!(err, XenonError::IndexOutOfBounds { .. }));
    }

    // ── Export tests ────────────────────────────────────────────

    /// Exports match source metadata, and crucially `data ==
    /// as_storage_ptr()` (NOT `as_ptr()`).
    #[test]
    fn test_export_contract_data_is_storage_base() {
        let data = [10_i32, 20, 30];
        // SAFETY: F-order [3] view over data; offset = 0.
        let tensor = unsafe {
            TensorView::<i32, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(3),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order [3] view");
        let raw = tensor.export();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.storage_len, 3);
        assert_eq!(raw.element_type, ElementType::I32 as u8);
        // The critical contract: `data` carries the storage
        // base pointer, not the logical first. For offset = 0 views the two
        // are equal; the offset != 0 case is tested in
        // `test_export_contract_data_storage_base_with_nonzero_offset`.
        assert_eq!(raw.data as *const i32, tensor.as_storage_ptr());
    }

    /// Critical: `data == as_storage_ptr()` AND `data != as_ptr()` when
    /// `offset != 0`. This is the gate guarding against the historical
    /// regression where `data` was set to `as_ptr()` (logical first)
    /// instead of `as_storage_ptr()` (storage base).
    #[test]
    fn test_export_contract_data_storage_base_with_nonzero_offset() {
        let data = [10_i32, 20, 30, 40, 50];
        // SAFETY: F-order [2] view starting at offset 2 over data;
        // storage_base = data.as_ptr(); logical_first = data.as_ptr() + 2.
        let tensor = unsafe {
            TensorView::<i32, Ix1>::from_raw_parts(
                data.as_ptr(), // ptr = storage base
                data.len(),    // storage_len = 5
                Ix1(2),        // shape
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                2, // offset ≠ 0
            )
        }
        .expect("F-order [2] view starting at offset 2");
        let raw = tensor.export();
        // Storage-base contract: `data` field equals storage base.
        assert_eq!(raw.data as *const i32, tensor.as_storage_ptr());
        // Offset preserved.
        assert_eq!(raw.offset, 2);
        // The logical first pointer is `data + offset` (NOT just `data`).
        // SAFETY: offset < storage_len ensures pointer stays within object.
        let logical_first = unsafe { (raw.data as *const i32).add(raw.offset) };
        assert_eq!(logical_first, tensor.as_ptr());
        // And it differs from `data` because offset ≠ 0.
        assert_ne!(raw.data as *const i32, tensor.as_ptr());
    }

    /// Empty tensors must export a dangling but valid aligned pointer.
    #[test]
    fn test_export_empty_tensor_dangling() {
        let data = Vec::<f64>::new();
        // SAFETY: shape [0] product 0 → validate_access_range early-returns;
        // storage_len = 0; F-order strides [1].
        let tensor = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                data.as_ptr(),
                0,
                Ix1(0),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("empty F-order [0] view");
        let raw = tensor.export();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.storage_len, 0);
        // dangling pointers are non-null and properly aligned.
        assert!(!raw.data.is_null());
        assert_eq!((raw.data as usize) % core::mem::align_of::<f64>(), 0);
    }

    /// `data` is writable; only callable on `StorageMut`. We verify
    /// writability by storing through the raw pointer and reading back
    /// via the original tensor.
    #[test]
    fn test_export_mut_contract_writable() {
        let mut data = vec![1_i32, 2];
        // SAFETY: F-order [2] mutable view over data; offset = 0.
        let mut tensor = unsafe {
            TensorViewMut::<i32, Ix1>::from_raw_parts_mut(
                data.as_mut_ptr(),
                data.len(),
                Ix1(2),
                Strides::from_slice(&[1_usize]).expect("valid strides"),
                0,
            )
        }
        .expect("valid F-order [2] mutable view");
        let storage_base_before = tensor.as_storage_ptr() as usize;
        let raw = tensor.export_mut();
        assert_eq!(raw.ndim, 1);
        assert_eq!(raw.data as usize, storage_base_before);
        // SAFETY: raw.data is the storage base; storage_len=2; we write
        // within the borrowed range while no other reference exists
        // (export_mut consumed `&mut tensor` for its lifetime).
        unsafe {
            (raw.data as *mut i32).add(0).write(99);
        }
        // Reading back through `tensor` is sound after `raw` is dropped
        // because `TensorExportMutRaw` does not own the lifetime —
        // the borrow ends at end-of-expression for `raw`.
        let _ = raw;
        let observed = tensor.as_slice().expect("tensor is non-empty")[0];
        assert_eq!(observed, 99);
    }
}
