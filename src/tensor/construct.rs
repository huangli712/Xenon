//! Internal constructors, validators, and raw-parts entry points.

use core::mem::ManuallyDrop;
use core::ptr::NonNull;

use crate::Result;
use crate::dimension::Dimension;
use crate::error::{InvalidLayoutReason, StorageKindTag, XenonError};
use crate::layout::{LayoutFlags, Strides, compute_layout_flags};
use crate::storage::{Owned, RawStorage, StorageOwned, ViewMutRepr, ViewRepr};
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

/// Decomposition of an owned tensor into raw pointer + allocator metadata.
///
/// # ABI Note
///
/// `OwnedRawParts<A, D>` is **not** a stable C-ABI type. The `D` and
/// `Strides<D>` fields are Rust generics whose layout is not specified
/// by `#[repr(C)]` (especially for `IxDyn`, which contains a `Vec<usize>`).
/// FFI consumers MUST NOT decode this struct from C code. It exists solely
/// as a Rust-internal round-trip carrier for `into_raw_parts` /
/// `from_raw_parts_owned`. C-facing interop must use the dedicated
/// `TensorExportRaw` / `TensorExportMutRaw` types, which are explicitly
/// designed for a stable C ABI.
#[expect(
    missing_debug_implementations,
    reason = "OwnedRawParts carries a raw pointer; Debug is misleading"
)]
pub struct OwnedRawParts<A, D>
where
    D: Dimension,
{
    /// Pointer to the storage base. Ownership transferred to the consumer
    /// upon `into_raw_parts`; reclaimed by `from_raw_parts_owned`.
    pub ptr: *mut A,
    /// Logical length in elements (matches `shape.checked_size()`).
    pub len: usize,
    /// Allocator capacity in elements (`cap >= len`).
    pub cap: usize,
    /// Allocator alignment in bytes (power of two, `>= align_of::<A>()`).
    pub align: usize,
    /// Owned tensor shape; for `into_raw_parts` always satisfies
    /// `shape.checked_size().expect("test input valid") == len`.
    pub shape: D,
    /// Canonical F-order strides matching `shape`.
    pub strides: Strides<D>,
    /// Logical offset in element units; **must be 0** for owned raw parts
    /// (enforced by `from_raw_parts_owned`).
    pub offset: usize,
}

impl<A, D> super::TensorBase<Owned<A>, D>
where
    A: crate::element::Element + Clone,
    D: Dimension + Clone,
{
    /// Consumes the tensor, returning owned raw parts.
    ///
    /// # Returns
    ///
    /// An `OwnedRawParts<A, D>` snapshot containing the pointer plus the
    /// allocator metadata required to reconstruct Xenon's aligned owned
    /// storage.
    ///
    /// # Ownership Transfer
    ///
    /// This method consumes `self`. The caller MUST eventually reconstruct
    /// the tensor via `from_raw_parts_owned` and let `Drop` reclaim the
    /// memory, or else memory is leaked. Calling system `free` or any
    /// foreign allocator on `raw.ptr` is UB because Xenon's aligned
    /// allocator uses a specific (cap, align) pair recorded only in the
    /// returned `OwnedRawParts`.
    pub fn into_raw_parts(self) -> OwnedRawParts<A, D> {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is a valid owned tensor; the storage base pointer
        // is exactly the pointer whose ownership is being transferred to the
        // caller as part of the returned raw parts.
        let ptr = this.storage.as_mut_ptr_unchecked();
        OwnedRawParts {
            ptr,
            len: this.storage.len(),
            cap: this.storage.capacity(),
            align: this.storage.alignment(),
            shape: this.shape.clone(),
            strides: this.strides.clone(),
            offset: this.offset,
        }
    }
}

impl<A, D> super::TensorBase<Owned<A>, D>
where
    A: crate::element::Element,
    D: Dimension + Clone + PartialEq,
{
    /// Reconstructs an owned tensor from raw parts obtained via
    /// `into_raw_parts`. Takes ownership of memory allocated by Xenon's
    /// aligned allocator.
    ///
    /// # Safety
    ///
    /// - `raw.ptr` must point to memory allocated by Xenon's aligned
    ///   allocator with the recorded `(cap, align)` pair.
    /// - `raw.len`, `raw.cap`, and `raw.align` must be the original allocator
    ///   metadata (as returned by `into_raw_parts`).
    /// - `raw.shape` and `raw.strides` must describe a valid, non-overlapping
    ///   canonical F-order layout.
    /// - `raw.offset` must be 0.
    /// - The caller transfers ownership; do NOT free `raw.ptr` separately.
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidLayout { reason, .. })` when directly
    /// checkable metadata validation fails. The memory/pointer guarantees
    /// must be upheld by the caller as they cannot be checked from metadata
    /// alone.
    pub unsafe fn from_raw_parts_owned(raw: OwnedRawParts<A, D>) -> Result<Self> {
        // offset must be zero for owned raw parts.
        if raw.offset != 0 {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::OwnedRequiresZeroOffset,
            });
        }

        // shape product must be representable AND equal raw.len.
        let expected_len = raw
            .shape
            .checked_size()
            .map_err(|_| XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::ShapeProductOverflow,
            })?;
        if raw.len != expected_len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::LenShapeMismatch,
            });
        }

        // capacity must cover len.
        if raw.cap < raw.len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::CapacityBelowLen,
            });
        }

        // align must be a valid power of two and at least align_of::<A>().
        if !raw.align.is_power_of_two() || raw.align < core::mem::align_of::<A>() {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::AlignmentInvalid,
            });
        }

        // strides must equal canonical F-order strides.
        let expected_strides = Strides::f_contiguous(&raw.shape)?;
        if raw.strides.as_slice() != expected_strides.as_slice() {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::OwnedRequiresCanonicalFOrder,
            });
        }

        // SAFETY: Caller's # Safety contract guarantees raw.ptr is valid
        // memory allocated by Xenon's aligned allocator with the recorded
        // (len, cap, align) metadata. Ownership transfer is part of the
        // contract; raw.ptr must not be freed externally.
        let storage = unsafe { Owned::from_raw_parts(raw.ptr, raw.len, raw.cap, raw.align) };

        let logical_ptr: *const A = if raw.len == 0 {
            // Empty tensors: use a well-defined non-dereferenceable sentinel
            // rather than a potentially dangling storage pointer.
            NonNull::<A>::dangling().as_ptr()
        } else {
            // offset == 0 already verified; raw.ptr IS the logical first element.
            raw.ptr
        };
        let flags = compute_layout_flags::<A, D>(&raw.shape, &raw.strides, logical_ptr);

        // SAFETY (new_unchecked invariant):
        //   (1) shape was overflow-checked above;
        //   (2) strides were verified canonical F-order;
        //   (3) offset == 0 was verified;
        //   (4) flags were just produced by compute_layout_flags for the
        //       same shape/strides/logical_ptr;
        //   (5) the logical access range [0, raw.len) lies within storage
        //       because raw.len == shape.checked_size() and
        //       raw.cap >= raw.len;
        //   (6) derived_from_view_mut: false — from_raw_parts_owned is an
        //       Owned reconstruction, NOT a ViewMut downgrade.
        Ok(unsafe {
            super::TensorBase::new_unchecked(storage, raw.shape, raw.strides, 0, flags, false)
        })
    }
}
#[cfg(test)]
mod tests {
    use super::{validate_access_range, validate_non_overlapping_layout};
    use crate::dimension::{Dimension, Ix0, Ix1, Ix2};
    use crate::error::StorageKindTag;
    use crate::layout::Strides;
    use crate::tensor::Tensor;

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

    // ── OwnedRawParts round-trip tests ──

    /// `into_raw_parts` → `from_raw_parts_owned` round-trip preserves shape,
    /// strides, offset, and element contents.
    #[test]
    fn test_into_raw_parts_roundtrip_2d() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let raw = original.into_raw_parts();
        assert_eq!(raw.len, 6);
        assert!(raw.cap >= 6);
        assert_eq!(raw.offset, 0);
        assert_eq!(raw.shape.slice(), &[2, 3]);
        // SAFETY: raw came directly from into_raw_parts; no external mutation.
        let restored: Tensor<i32, Ix2> =
            unsafe { Tensor::from_raw_parts_owned(raw) }.expect("round-trip must succeed");
        assert_eq!(restored.shape(), &[2, 3]);
        assert_eq!(
            restored.as_slice().expect("test input valid"),
            &[1, 2, 3, 4, 5, 6]
        );
    }

    /// Round-trip for 1D tensors via the from_vec convenience path.
    #[test]
    fn test_into_raw_parts_roundtrip_1d() {
        let original =
            Tensor::<f64, Ix1>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).expect("test input valid");
        let raw = original.into_raw_parts();
        let restored: Tensor<f64, Ix1> =
            unsafe { Tensor::from_raw_parts_owned(raw) }.expect("test input valid");
        assert_eq!(
            restored.as_slice().expect("test input valid"),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }

    /// Empty tensor round-trip — len=0 path must use dangling sentinel for
    /// compute_layout_flags input.
    #[test]
    fn test_into_raw_parts_roundtrip_empty() {
        let original =
            Tensor::<i32, Ix1>::from_shape_vec([0], Vec::new()).expect("test input valid");
        let raw = original.into_raw_parts();
        assert_eq!(raw.len, 0);
        let restored: Tensor<i32, Ix1> =
            unsafe { Tensor::from_raw_parts_owned(raw) }.expect("test input valid");
        assert_eq!(restored.len(), 0);
    }

    /// Gate 1: from_raw_parts_owned rejects non-zero offset.
    #[test]
    fn test_from_raw_parts_owned_rejects_nonzero_offset() {
        let original = Tensor::<i32, Ix1>::from_vec(vec![1_i32, 2, 3]).expect("test input valid");
        let mut raw = original.into_raw_parts();
        raw.offset = 1; // Tamper with the offset.
        let err = unsafe { Tensor::<i32, Ix1>::from_raw_parts_owned(raw) }
            .expect_err("tampered raw parts");
        assert!(matches!(
            err,
            crate::error::XenonError::InvalidLayout {
                reason: crate::error::InvalidLayoutReason::OwnedRequiresZeroOffset,
                ..
            }
        ));
    }

    /// Gate 2 (LenShapeMismatch): tampered shape produces wrong expected length.
    #[test]
    fn test_from_raw_parts_owned_rejects_len_shape_mismatch() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let mut raw = original.into_raw_parts();
        // Tamper: claim shape [3, 3] (expected_len=9) but len stays at 6.
        // Gate 2 (LenShapeMismatch) fires before Gate 5 because the check
        // order in `from_raw_parts_owned` is offset → shape → cap → align → strides.
        raw.shape = crate::dimension::Ix2(3, 3);
        let err = unsafe { Tensor::<i32, Ix2>::from_raw_parts_owned(raw) }
            .expect_err("tampered raw parts");
        assert!(matches!(
            err,
            crate::error::XenonError::InvalidLayout {
                reason: crate::error::InvalidLayoutReason::LenShapeMismatch,
                ..
            }
        ));
    }

    /// Gate 5 (OwnedRequiresCanonicalFOrder): tampered strides reject.
    #[test]
    fn test_from_raw_parts_owned_rejects_non_canonical_strides() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let mut raw = original.into_raw_parts();
        // Tamper strides — replace canonical [1, 2] with C-order [3, 1].
        raw.strides = crate::layout::Strides::from_slice(&[3, 1]).expect("test input valid");
        let err = unsafe { Tensor::<i32, Ix2>::from_raw_parts_owned(raw) }
            .expect_err("tampered raw parts");
        assert!(matches!(
            err,
            crate::error::XenonError::InvalidLayout {
                reason: crate::error::InvalidLayoutReason::OwnedRequiresCanonicalFOrder,
                ..
            }
        ));
    }
}
