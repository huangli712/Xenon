//! Multi-dimensional index → element offset / pointer helpers.
//!
//! See `docs/design/23-ffi.md` §5.13 (`try_offset_of` / `try_ptr_at`)
//! and §6.2 (overflow-safety contract). Error variants follow
//! `26-error.md` §5.1 (`IndexOutOfBounds`, `DimensionMismatch`,
//! `InvalidLayout` + `InvalidLayoutReason::AccessRangeExceedsStorage`).

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::error::{InvalidLayoutReason, StorageKindTag, XenonError};
use crate::storage::Storage;
use crate::tensor::{StorageKind, StorageSemantics, TensorBase};

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
    /// or wraparound (`23-ffi.md` §6.2).
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
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A> + StorageSemantics,
    D: Dimension,
{
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
    /// at the FFI boundary (`23-ffi.md` §5.13 / §1.2 minimum-constraint
    /// principle).
    pub fn try_ptr_at(&self, index: &[usize]) -> Result<*const A, XenonError> {
        let offset = self.try_offset_of(index)?;
        // SAFETY: `try_offset_of` validates `index < shape` per axis;
        // for a well-constructed tensor the resulting offset always lies
        // within the storage range covered by `as_ptr() ..
        // as_ptr().add(storage_len - self.offset())`.
        Ok(unsafe { self.as_ptr().add(offset) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};
    use crate::layout::Strides;
    use crate::tensor::TensorView;

    /// Helper: build a TensorView<Ix2> on top of a Vec via W8T7 `from_raw_parts`.
    /// Returns the (view, vec) pair; the Vec owner must outlive the view.
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

    /// §8.2 test_try_offset_of_various — F-order [2, 3] tensor,
    /// shape=[2,3], strides=[1,2]; index [1, 2] → 1*1 + 2*2 = 5.
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

    /// §8.2 test_try_offset_of_checked_overflow — checked arithmetic on
    /// `strides * index` overflow must yield `InvalidLayout { reason:
    /// AccessRangeExceedsStorage }`, never panic.
    ///
    /// **Construction path analysis**:
    ///   - `validate_access_range` structure (W8T7 Step 2 / 07-tensor.md §6.2):
    ///     Step 1: shape.checked_size() → len
    ///     Step 2: if len == 0: early-return Ok(())  ← this test relies on this path
    ///     Step 3: stride > isize::MAX  → StrideExceedsIsizeMax
    ///     Step 4-5: span/max_offset computation
    ///   - shape=[3, 0] → len=0 → Step 2 early-returns, **bypassing Step 3's
    ///     isize::MAX check**. This lets the carrier tensor carry
    ///     `stride[0] = usize::MAX` through construction-time validation.
    ///   - try_offset_of(&[2, 0]): axis 0 computes `2 < 3 ✓ → term = usize::MAX * 2`,
    ///     **checked_mul overflows** → InvalidLayout { reason: AccessRangeExceedsStorage }.
    ///   - axis 1 (index=0) does not participate in the overflow, not affecting
    ///     the test semantics.
    #[test]
    fn test_try_offset_of_checked_overflow() {
        let data = Vec::<i32>::new(); // empty storage
        // SAFETY: shape=[3,0] has product 0 → validate_access_range early-returns
        // at the len==0 gate (W8T7 Step 2), bypassing the stride>isize::MAX check.
        // The huge stride[0] propagates into the constructed view, where
        // try_offset_of will hit the checked_mul overflow path.
        let t = unsafe {
            crate::tensor::TensorView::<i32, Ix2>::from_raw_parts(
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

    /// §8.2 test_try_ptr_at_various — pointer dereference yields the
    /// element at the indexed position.
    #[test]
    fn test_try_ptr_at_various() {
        let data = [10_i32, 20, 30];
        // SAFETY: F-order [3] view over data.
        let t = unsafe {
            crate::tensor::TensorView::<i32, Ix1>::from_raw_parts(
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
}
