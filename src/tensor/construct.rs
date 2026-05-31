//! Internal constructors, validators, and raw-parts entry points.

use crate::Result;
use crate::dimension::Dimension;
use crate::error::{InvalidLayoutReason, StorageKindTag, XenonError};
use crate::layout::{LayoutFlags, Strides, compute_layout_flags};
use crate::storage::{Owned, RawStorage, ViewMutRepr, ViewRepr};
use std::borrow::Cow;
// ── new_unchecked ──

impl<S, D> super::TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Canonical unchecked tensor metadata assembly.
    ///
    /// # Safety
    /// Caller must guarantee shape/strides/offset/flags mutual consistency,
    /// validated access range, and correct `derived_from_view_mut`.
    pub(crate) unsafe fn new_unchecked(
        storage: S,
        shape: D,
        strides: Strides<D>,
        offset: usize,
        flags: LayoutFlags,
        derived_from_view_mut: bool,
    ) -> Self {
        Self {
            storage,
            shape,
            strides,
            offset,
            flags,
            derived_from_view_mut,
        }
    }
}

// ── validate_access_range ──

/// Validates that the logical access range defined by shape/strides/offset
/// fits within the given storage length.
///
/// Returns an error for: shape product overflow, stride exceeding
/// `isize::MAX`, stride span overflow, and out-of-bounds access.
/// Zero-length tensors are accepted as long as the offset does not exceed
/// the storage length.
pub(crate) fn validate_access_range<D: Dimension>(
    shape: &D,
    strides: &Strides<D>,
    offset: usize,
    storage_len: usize,
    op_name: &'static str,
    kind: StorageKindTag,
) -> Result<()> {
    let len = match shape.checked_size() {
        Ok(l) => l,
        Err(_) => {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::ShapeProductOverflow,
            });
        },
    };

    if len == 0 {
        if offset > storage_len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::EmptyTensorOffsetExceedsStorage,
            });
        }
        return Ok(());
    }

    for &stride in strides.as_slice() {
        if stride > isize::MAX as usize {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::StrideExceedsIsizeMax,
            });
        }
    }

    let mut max_offset = offset;
    for (&dim, &stride) in shape.slice().iter().zip(strides.as_slice()) {
        if dim == 0 {
            continue;
        }
        let span = match (dim - 1).checked_mul(stride) {
            Some(s) => s,
            None => {
                return Err(XenonError::InvalidLayout {
                    operation: Cow::Borrowed(op_name),
                    storage_kind: kind,
                    shape: shape.slice().to_vec(),
                    strides: strides.as_slice().to_vec(),
                    offset,
                    storage_len,
                    reason: InvalidLayoutReason::StrideSpanOverflow,
                });
            },
        };
        max_offset = match max_offset.checked_add(span) {
            Some(m) => m,
            None => {
                return Err(XenonError::InvalidLayout {
                    operation: Cow::Borrowed(op_name),
                    storage_kind: kind,
                    shape: shape.slice().to_vec(),
                    strides: strides.as_slice().to_vec(),
                    offset,
                    storage_len,
                    reason: InvalidLayoutReason::AccessRangeOverflow,
                });
            },
        };
    }

    if max_offset >= storage_len {
        return Err(XenonError::InvalidLayout {
            operation: Cow::Borrowed(op_name),
            storage_kind: kind,
            shape: shape.slice().to_vec(),
            strides: strides.as_slice().to_vec(),
            offset,
            storage_len,
            reason: InvalidLayoutReason::AccessRangeExceedsStorage,
        });
    }

    Ok(())
}

// ── validate_non_overlapping_layout ──

/// Validates that a mutable view's layout has no ambiguous element overlap.
///
/// Rejects zero-stride axes on non-singleton dimensions (which would
/// cause multiple logical indices to map to the same memory) and layouts
/// where different index tuples alias the same storage address. Singleton
/// dimensions and empty tensors are accepted.
pub(crate) fn validate_non_overlapping_layout<D: Dimension>(
    shape: &D,
    strides: &Strides<D>,
    offset: usize,
    storage_len: usize,
) -> Result<()> {
    let len = shape.checked_size().unwrap_or(0);
    if len <= 1 {
        return Ok(());
    }

    for (&dim, &stride) in shape.slice().iter().zip(strides.as_slice()) {
        if dim > 1 && stride == 0 {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::validate_non_overlapping_layout"),
                storage_kind: StorageKindTag::ViewMut,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::ZeroStrideRejectedForViewMut,
            });
        }
    }

    let mut axes: Vec<(usize, usize)> = shape
        .slice()
        .iter()
        .zip(strides.as_slice())
        .filter(|(dim, _)| **dim > 1)
        .map(|(&dim, &stride)| (dim, stride))
        .collect();
    axes.sort_by_key(|&(_, stride)| stride);

    let mut covered_max_offset: usize = 0;
    for (dim, stride) in axes {
        if stride <= covered_max_offset {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::validate_non_overlapping_layout"),
                storage_kind: StorageKindTag::ViewMut,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::AmbiguousOverlap,
            });
        }
        let span = match (dim - 1).checked_mul(stride) {
            Some(s) => s,
            None => {
                return Err(XenonError::InvalidLayout {
                    operation: Cow::Borrowed("tensor::validate_non_overlapping_layout"),
                    storage_kind: StorageKindTag::ViewMut,
                    shape: shape.slice().to_vec(),
                    strides: strides.as_slice().to_vec(),
                    offset,
                    storage_len,
                    reason: InvalidLayoutReason::StrideSpanOverflow,
                });
            },
        };
        covered_max_offset = match covered_max_offset.checked_add(span) {
            Some(m) => m,
            None => {
                return Err(XenonError::InvalidLayout {
                    operation: Cow::Borrowed("tensor::validate_non_overlapping_layout"),
                    storage_kind: StorageKindTag::ViewMut,
                    shape: shape.slice().to_vec(),
                    strides: strides.as_slice().to_vec(),
                    offset,
                    storage_len,
                    reason: InvalidLayoutReason::StrideSpanOverflow,
                });
            },
        };
    }

    Ok(())
}

// ── from_raw_parts (immutable view) ──

impl<'a, A, D> super::TensorBase<ViewRepr<'a, A>, D>
where
    A: crate::element::Element,
    D: Dimension,
{
    /// Constructs an immutable view from raw parts.
    ///
    /// # Safety
    ///
    /// - `ptr` is the non-null storage base pointer, valid for lifetime `'a`.
    ///   Empty tensors must still pass a non-null sentinel such as
    ///   `NonNull::<A>::dangling().as_ptr()`.
    /// - The byte range `[ptr, ptr + storage_len * size_of::<A>())` belongs
    ///   to a single allocated object and stays valid for lifetime `'a`.
    /// - `ptr` is aligned to `align_of::<A>()`.
    /// - Every logical element address derived from shape/strides/offset
    ///   points to an initialized `A` value (for non-empty tensors).
    /// - No live `&mut` reference to overlapping memory exists during `'a`.
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidLayout)` for shape product overflow,
    /// stride > `isize::MAX`, stride span overflow, or access range
    /// out of bounds.
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self> {
        validate_access_range(
            &shape,
            &strides,
            offset,
            storage_len,
            "TensorView::from_raw_parts",
            StorageKindTag::View,
        )?;

        let storage = unsafe { ViewRepr::from_raw_parts(ptr, storage_len) };

        let logical_first: *const A = if shape.checked_size().unwrap_or(0) == 0 {
            core::ptr::NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            unsafe { ptr.add(offset) }
        };
        let flags = compute_layout_flags::<A, D>(&shape, &strides, logical_first);

        Ok(unsafe { Self::new_unchecked(storage, shape, strides, offset, flags, false) })
    }
}

// ── from_raw_parts_mut (mutable view) ──

impl<'a, A, D> super::TensorBase<ViewMutRepr<'a, A>, D>
where
    A: crate::element::Element,
    D: Dimension,
{
    /// Constructs a mutable view from raw parts.
    ///
    /// # Safety
    ///
    /// Inherits all caller obligations from [`from_raw_parts`] plus:
    /// - `ptr` is non-null; empty tensors must still pass a non-null sentinel.
    /// - Caller holds exclusive write access to `[ptr, ptr + storage_len)`
    ///   for lifetime `'a`.
    /// - No other reference (shared or mutable) to overlapping memory may be
    ///   alive during `'a`.
    /// - The layout itself is non-overlapping (no two logical indices map to
    ///   the same address).
    ///
    /// # Errors
    ///
    /// Same as [`from_raw_parts`], plus rejects zero-stride on non-singleton
    /// axes and ambiguous-overlap layouts.
    ///
    /// [`from_raw_parts`]: crate::tensor::TensorBase::from_raw_parts
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self> {
        validate_access_range(
            &shape,
            &strides,
            offset,
            storage_len,
            "TensorViewMut::from_raw_parts_mut",
            StorageKindTag::ViewMut,
        )?;
        validate_non_overlapping_layout(&shape, &strides, offset, storage_len)?;

        let storage = unsafe { ViewMutRepr::from_raw_parts_mut(ptr, storage_len) };

        let logical_first: *const A = if shape.checked_size().unwrap_or(0) == 0 {
            core::ptr::NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            unsafe { (ptr as *const A).add(offset) }
        };
        let flags = compute_layout_flags::<A, D>(&shape, &strides, logical_first);

        Ok(unsafe { Self::new_unchecked(storage, shape, strides, offset, flags, false) })
    }
}

// ── from_raw_vec_unchecked ──

impl<A, D> super::TensorBase<Owned<A>, D>
where
    A: crate::element::Element,
    D: Dimension,
{
    /// Construct an Owned tensor from a Vec, skipping all consistency checks.
    ///
    /// # Safety
    ///
    /// - `data.as_ptr()` remains valid for construction.
    /// - `shape.checked_size()` was previously validated (no overflow).
    /// - `data.len() == shape.checked_size()` — mismatch is undefined behaviour.
    ///
    /// # Panics
    ///
    /// Panics if `Strides::f_contiguous(&shape)` returns an error (shape product
    /// overflow), or if `Owned::from_vec(data)` returns an error (allocation
    /// failure or byte-size overflow). Both are unreachable when the caller
    /// upholds the `# Safety` precondition that `shape.checked_size()` was
    /// previously validated.
    pub unsafe fn from_raw_vec_unchecked(data: Vec<A>, shape: D) -> Self {
        let strides = crate::layout::Strides::f_contiguous(&shape).expect("caller-proved valid shape");
        let storage = Owned::from_vec(data).expect("caller-proved valid vec");
        let flags = compute_layout_flags::<A, D>(&shape, &strides, storage.as_ptr());
        unsafe { Self::new_unchecked(storage, shape, strides, 0, flags, false) }
    }
}

// ── OwnedRawParts ──
#[cfg(test)]
mod tests {
    use super::{validate_access_range, validate_non_overlapping_layout};
    use crate::dimension::{Ix0, Ix2};
    use crate::error::StorageKindTag;
    use crate::layout::Strides;

    /// Validates access range for a 2×2 F-order layout with sufficient storage.
    #[test]
    fn test_validate_access_range_valid() {
        let r = validate_access_range(
            &Ix2(2, 2),
            &Strides::new(Ix2(1, 2)),
            0,
            4,
            "test",
            StorageKindTag::View,
        );
        assert!(r.is_ok());
    }

    /// Access range with storage_len 3 on a 2×2 layout should be rejected.
    #[test]
    fn test_validate_access_range_out_of_bounds() {
        let r = validate_access_range(
            &Ix2(2, 2),
            &Strides::new(Ix2(1, 2)),
            0,
            3,
            "test",
            StorageKindTag::View,
        );
        assert!(r.is_err());
    }

    /// Empty tensor (any axis = 0) with offset 0 should pass validation.
    #[test]
    fn test_validate_access_range_empty_offset_ok() {
        let r = validate_access_range(
            &Ix2(0, 3),
            &Strides::new(Ix2(1, 1)),
            0,
            0,
            "test",
            StorageKindTag::View,
        );
        assert!(r.is_ok());
    }

    /// Dense 2×3 F-order layout should be non-overlapping.
    #[test]
    fn test_validate_non_overlap_dense_prefix_ok() {
        let r = validate_non_overlapping_layout(&Ix2(2, 3), &Strides::new(Ix2(1, 2)), 0, 6);
        assert!(r.is_ok());
    }

    /// Zero-stride axis on a 2×3 layout should be rejected.
    #[test]
    fn test_validate_non_overlap_zero_stride_rejected() {
        let r = validate_non_overlapping_layout(&Ix2(2, 3), &Strides::new(Ix2(0, 1)), 0, 6);
        assert!(r.is_err());
    }

    /// Ambiguous overlap (stride [1, 1] for 2×2) should be rejected.
    #[test]
    fn test_validate_non_overlap_ambiguous_rejected() {
        let r = validate_non_overlapping_layout(&Ix2(2, 2), &Strides::new(Ix2(1, 1)), 0, 4);
        assert!(r.is_err());
    }

    /// `from_raw_vec_unchecked` with 4-element vec and shape [2, 2]
    /// produces a valid F-contiguous tensor.
    #[test]
    fn test_from_raw_vec_unchecked_valid() {
        let tensor = unsafe {
            super::super::TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4], Ix2(2, 2))
        };
        assert_eq!(tensor.len(), 4);
        assert!(tensor.is_f_contiguous());
    }

    /// `from_raw_vec_unchecked` with empty vec and shape [0, 3]
    /// produces an empty F-contiguous tensor.
    #[test]
    fn test_from_raw_vec_unchecked_empty() {
        let tensor = unsafe {
            super::super::TensorBase::from_raw_vec_unchecked(Vec::<i32>::new(), Ix2(0, 3))
        };
        assert_eq!(tensor.len(), 0);
        assert!(tensor.is_f_contiguous());
    }

    /// `from_raw_vec_unchecked` with a 0-dimensional shape should succeed.
    #[test]
    fn test_from_raw_vec_unchecked_zero_dim() {
        let tensor = unsafe { super::super::TensorBase::from_raw_vec_unchecked(vec![42_i32], Ix0) };
        assert_eq!(tensor.ndim(), 0);
        assert_eq!(tensor.len(), 1);
    }

}
