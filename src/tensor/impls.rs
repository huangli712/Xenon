//! Implementations for [`TensorBase`]: validation, query, construction,
//! views, raw‑parts, pointer access, slice extraction, semantic dispatch,
//! and associated tests.

use core::slice;
use core::mem::ManuallyDrop;
use core::ptr::NonNull;
use std::borrow::Cow;

use super::{TensorBase, OwnedRawParts};
use super::{DataLocation, StorageKind, AccessSemantics, AliasClass};
use super::StorageSemantics;

use crate::error::{InvalidLayoutReason, StorageKindTag, XenonError};
use crate::Result;
use crate::dimension::Dimension;
use crate::element::Element;
use crate::layout::{LayoutFlags, LayoutState, Strides, compute_layout_flags};
use crate::storage::{Owned, ViewRepr, ViewMutRepr, ArcRepr};
use crate::storage::{RawStorage, Storage, StorageMut, StorageOwned};

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
    // Compute shape product — overflow means metadata is corrupt.
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

    // Zero-length tensors are valid if offset stays within storage.
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

    // Every stride must fit in isize for safe pointer arithmetic.
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

    // Compute max logical offset: start from base, add span of each axis.
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

    // Check that the computed max offset does not exceed storage.
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
    // Trivially non-overlapping: scalar (0D or 1-element) or empty.
    let len = shape.checked_size().unwrap_or(0);
    if len <= 1 {
        return Ok(());
    }

    // Reject zero-stride axes on dimensions larger than 1.
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

    // Greedy overlap detection: sort axes by stride ascending, then verify
    // each axis span starts beyond the region already covered.
    let mut axes: Vec<(usize, usize)> = shape
        .slice()
        .iter()
        .zip(strides.as_slice())
        .filter(|(dim, _)| **dim > 1)
        .map(|(&dim, &stride)| (dim, stride))
        .collect();
    axes.sort_by_key(|&(_, stride)| stride);

    // Walk sorted axes: the gap between the previous axis's covered region
    // and the current axis's first element must be positive (non-overlap).
    // If a stride falls inside covered region, two index tuples alias.
    let mut covered_max_offset: usize = 0;
    for (dim, stride) in axes {
        // Ambiguous overlap: current stride falls inside previously covered region.
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
        // Span of this axis: (dim - 1) × stride elements past the first.
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
        // Extend covered region by this axis's span.
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

// ---------- Basic query, layout & construction ------------------------------

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Axis lengths. Zero-copy delegation to `Dimension::slice()`.
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }

    /// Strides in element units (usize, may be 0 for broadcast dims).
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    /// Number of dimensions (compile-time const for `Ix0`–`Ix6`,
    /// runtime for `IxDyn`).
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Total logical element count = product of all axis lengths.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `shape.checked_size()` succeeds for every
    /// tensor that has been constructed through the safe constructors, which
    /// validate the axis-product fits in `usize` at construction time. A panic
    /// here would indicate a violated construction-time invariant.
    pub fn len(&self) -> usize {
        self.shape.checked_size().expect("validated shape")
    }

    /// Non-negative offset from storage base to logical first element.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Full layout flags (`F_CONTIGUOUS` / `ALIGNED` / `HAS_ZERO_STRIDE`).
    pub fn flags(&self) -> LayoutFlags {
        self.flags
    }

    /// Physical data location. Currently always [`DataLocation::Cpu`].
    pub fn data_location(&self) -> DataLocation {
        DataLocation::Cpu
    }

    /// Returns the layout-state classification (`FContiguous` /
    /// `NonContiguous` / `BroadcastView`).
    pub fn layout_state(&self) -> LayoutState {
        self.flags.classify()
    }

    /// Underlying storage buffer length in elements.
    ///
    /// Distinct from [`len`](Self::len) which returns the logical element
    /// count (product of axis dimensions). The storage buffer may be larger
    /// than the logical count for views into larger allocations.
    pub fn storage_len(&self) -> usize {
        self.storage.len()
    }

    /// Returns `true` if the data is F-order contiguous.
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.is_f_contiguous()
    }

    /// Returns `true` if the data is at least 64-byte aligned.
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }

    /// `true` iff any axis length is 0 (i.e. logical element count is 0).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` iff `LayoutFlags::HAS_ZERO_STRIDE` is set (broadcast-
    /// induced zero stride).
    pub fn has_zero_stride(&self) -> bool {
        self.flags.has_zero_stride()
    }

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

// ---------- Dimension-bound query -------------------------------------------

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension + Clone,
{
    /// Clone of the dimension descriptor. Requires `D: Clone`.
    pub fn raw_dim(&self) -> D {
        self.shape.clone()
    }
}

// ---------- Immutable storage access ----------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Raw pointer to the logical first element.
    ///
    /// For empty tensors returns `NonNull::dangling().as_ptr()`. Otherwise
    /// returns `storage.as_ptr().add(offset)`.
    pub fn as_ptr(&self) -> *const A {
        if self.is_empty() {
            NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            unsafe { self.storage.as_ptr().add(self.offset) }
        }
    }

    /// Raw storage base pointer (does NOT add `offset`).
    pub fn as_storage_ptr(&self) -> *const A {
        self.storage.as_ptr()
    }

    /// Returns `Some(&[A])` for F-contiguous non-broadcast non-empty tensors,
    /// or for empty tensors. Returns `None` otherwise.
    pub fn as_slice(&self) -> Option<&[A]> {
        if self.is_empty() {
            return Some(unsafe { slice::from_raw_parts(self.as_ptr(), 0) });
        }
        if !self.flags.is_f_contiguous() || self.flags.has_zero_stride() {
            return None;
        }
        Some(unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) })
    }
}

// ---------- Mutable storage access ------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Raw mutable pointer to the logical first element.
    ///
    /// For empty tensors returns `NonNull::dangling().as_ptr()`.
    pub fn as_mut_ptr(&mut self) -> *mut A {
        if self.is_empty() {
            NonNull::<A>::dangling().as_ptr()
        } else {
            unsafe { self.storage.as_mut_ptr().add(self.offset) }
        }
    }

    /// Raw mutable storage base pointer (does NOT add `offset`).
    pub fn as_storage_mut_ptr(&mut self) -> *mut A {
        self.storage.as_mut_ptr()
    }

    /// Returns `Some(&mut [A])` for F-contiguous non-broadcast non-empty
    /// tensors, or for empty tensors. Returns `None` otherwise.
    pub fn as_mut_slice(&mut self) -> Option<&mut [A]> {
        if self.is_empty() {
            return Some(unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), 0) });
        }
        if !self.flags.is_f_contiguous() || self.flags.has_zero_stride() {
            return None;
        }
        let len = self.len();
        let ptr = self.as_mut_ptr();
        Some(unsafe { slice::from_raw_parts_mut(ptr, len) })
    }
}

// ---------- Semantic dispatch -----------------------------------------------

impl<S, D> TensorBase<S, D>
where
    S: StorageSemantics,
    D: Dimension,
{
    /// Returns the storage-representation [`StorageKind`] of this tensor.
    pub fn storage_kind(&self) -> StorageKind {
        S::KIND
    }

    /// Returns the [`AccessSemantics`] of this tensor.
    pub fn access_semantics(&self) -> AccessSemantics {
        S::access_semantics(self.flags, self.derived_from_view_mut)
    }

    /// Returns the precise [`AliasClass`] for this tensor.
    pub fn alias_class(&self) -> AliasClass {
        S::alias_class(self.flags, self.derived_from_view_mut)
    }
}

// ---------- Owned storage methods -------------------------------------------

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element + Clone,
    D: Dimension + Clone + PartialEq,
{
    /// Creates an immutable view sharing the underlying storage.
    ///
    /// `derived_from_view_mut` is set to `false` for Owned-backed views.
    pub fn view(&self) -> TensorBase<ViewRepr<'_, A>, D> {
        // SAFETY: storage exposes valid base pointer + len
        let storage = unsafe {
            ViewRepr::from_raw_parts(
                self.storage.as_ptr(),
                self.storage.len()
            )
        };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        }
    }

    /// Creates a mutable view sharing the underlying storage.
    pub fn view_mut(&mut self) -> TensorBase<ViewMutRepr<'_, A>, D> {
        let storage = unsafe {
            ViewMutRepr::from_raw_parts_mut(
                self.storage.as_ptr() as *mut A, self.storage.len()
            )
        };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        }
    }

    /// Zero-copy conversion from `Tensor<A, D>` to `ArcTensor<A, D>`.
    ///
    /// Wraps the storage-layer `Owned<A>::into_shared(self) -> ArcRepr<A>`.
    /// Shape, strides, offset, and layout flags are preserved;
    /// `derived_from_view_mut` is `false` since `Owned`-backed tensors are
    /// never derived from a `ViewMut`.
    pub fn into_shared(self) -> TensorBase<ArcRepr<A>, D> {
        let storage = self.storage.into_shared();
        // SAFETY: shape/strides/offset/flags are inherited verbatim from a
        // previously validated Owned tensor; underlying memory is preserved
        // by the zero-copy `Owned::into_shared` move. derived_from_view_mut
        // is false because Owned tensors cannot be derived from a ViewMut.
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape,
                self.strides,
                self.offset,
                self.flags,
                false,
            )
        }
    }

    /// Consumes the tensor, returning owned raw parts.
    ///
    /// This method consumes `self`. The caller must eventually reconstruct
    /// the tensor via `from_raw_parts_owned` and let `Drop` reclaim the
    /// memory, or else memory is leaked.
    pub fn into_raw_parts(self) -> OwnedRawParts<A, D> {
        let this = ManuallyDrop::new(self);
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

    /// Reconstructs an owned tensor from raw parts obtained via
    /// `into_raw_parts`, completing the round-trip ownership transfer.
    ///
    /// Together with `into_raw_parts`, this method forms the zero-copy
    /// ownership bridge: `into_raw_parts` decomposes a `Tensor<A, D>` into
    /// [`OwnedRawParts`], and `from_raw_parts_owned` reassembles it, taking
    /// back ownership of memory allocated by Xenon's aligned allocator.
    ///
    /// # Safety
    ///
    /// - `raw.ptr` must point to memory allocated by Xenon's aligned
    ///   allocator with the recorded `(cap, align)` pair.
    /// - `raw.len`, `raw.cap`, and `raw.align` must be the original allocator
    ///   metadata as returned by `into_raw_parts`.
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
        // Owned tensors always have offset 0.
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

        // Shape product must be representable AND equal raw.len.
        let expected_len = raw.shape.checked_size().map_err(|_| {
            XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::ShapeProductOverflow,
            }
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

        // Capacity must cover len.
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

        // Alignment must be a valid power of two and at least align_of::<A>().
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

        // Strides must equal canonical F-order strides.
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

        // All validations passed — transfer ownership back to Xenon's allocator.
        let storage = unsafe {
            Owned::from_raw_parts(raw.ptr, raw.len, raw.cap, raw.align)
        };

        // Empty tensors use a dangling sentinel; otherwise raw.ptr is the
        // logical first element (offset was validated to be 0 above).
        let logical_ptr: *const A = if raw.len == 0 {
            NonNull::<A>::dangling().as_ptr()
        } else {
            raw.ptr
        };

        // Compute layout flags from the validated shape/strides/logical_ptr.
        let flags = compute_layout_flags::<A, D>(
            &raw.shape,
            &raw.strides,
            logical_ptr
        );

        Ok(TensorBase {
            storage,
            shape: raw.shape,
            strides: raw.strides,
            offset: 0,
            flags,
            derived_from_view_mut: false,
        })
    }
}

// ---------- Owned construction ----------------------------------------------

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
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

impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D>
where
    D: Dimension + Clone,
{
    /// Creates an immutable view, propagating `derived_from_view_mut` from
    /// the source (may be `true` if the source was already a demoted ViewMut).
    pub fn view(&self) -> TensorBase<ViewRepr<'_, A>, D> {
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                self.derived_from_view_mut,
            )
        }
    }
}

// ---------- from_raw_parts (immutable view) ---------------------------------

impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D>
where
    A: Element,
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
            NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            unsafe { ptr.add(offset) }
        };
        let flags = compute_layout_flags::<A, D>(&shape, &strides, logical_first);

        Ok(unsafe { Self::new_unchecked(storage, shape, strides, offset, flags, false) })
    }
}

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    A: Element,
    D: Dimension + Clone,
{
    /// Demotes a mutable view to an immutable view with ViewMut provenance.
    ///
    /// Sets `derived_from_view_mut = true` so that `access_semantics()` and
    /// `alias_class()` correctly report the ViewMut origin.
    pub fn view(&self) -> TensorBase<ViewRepr<'_, A>, D> {
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                true,
            )
        }
    }
}

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    D: Dimension + Clone,
{
    /// Creates a reborrowed mutable view. Does NOT set the ViewMut provenance bit.
    pub fn view_mut(&mut self) -> TensorBase<ViewMutRepr<'_, A>, D> {
        let storage = unsafe {
            ViewMutRepr::from_raw_parts_mut(self.storage.as_ptr() as *mut A, self.storage.len())
        };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        }
    }
}

// ---------- from_raw_parts_mut (mutable view) -------------------------------

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    A: Element,
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
            NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            unsafe { (ptr as *const A).add(offset) }
        };
        let flags = compute_layout_flags::<A, D>(&shape, &strides, logical_first);

        Ok(unsafe { Self::new_unchecked(storage, shape, strides, offset, flags, false) })
    }
}

impl<A, D> TensorBase<ArcRepr<A>, D>
where
    A: Element,
    D: Dimension + Clone,
{
    /// Creates an immutable view sharing the underlying Arc storage.
    pub fn view(&self) -> TensorBase<ViewRepr<'_, A>, D> {
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Dimension, Ix0, Ix1, Ix2};
    use crate::layout::{LayoutFlags, LayoutState, Strides};
    use crate::storage::Owned;
    use crate::tensor::AccessSemantics;
    use crate::tensor::Tensor;

    fn make_owned(
        shape: Ix2,
        data: Vec<f64>,
        flags: LayoutFlags,
        derived: bool,
        offset: usize,
    ) -> TensorBase<Owned<f64>, Ix2> {
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(data).expect("valid vec");
        TensorBase {
            storage,
            shape,
            strides,
            offset,
            flags,
            derived_from_view_mut: derived,
        }
    }

    fn f_contig(data: Vec<i32>, shape: Ix2, offset: usize) -> TensorBase<Owned<i32>, Ix2> {
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(data).expect("valid vec");
        TensorBase {
            storage,
            shape,
            strides,
            offset,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        }
    }

    fn f_contig_i32(data: Vec<i32>, shape: Ix2) -> TensorBase<Owned<i32>, Ix2> {
        f_contig(data, shape, 0)
    }


    /// Verify shape and strides are correctly returned for a 3×4 tensor.
    #[test]
    fn test_tensor_shape_strides() {
        let t = f_contig_i32(vec![1; 12], Ix2(3, 4));
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.strides(), &[1_usize, 3]);
    }
    /// Verify len and ndim are correctly computed for a 3×4 tensor.
    #[test]
    fn test_tensor_len_and_ndim() {
        let t = f_contig_i32(vec![1; 12], Ix2(3, 4));
        assert_eq!(t.len(), 12);
        assert_eq!(t.ndim(), 2);
    }
    /// Verify offset and flags after direct struct construction.
    #[test]
    fn test_tensor_offset_and_flags() {
        let strides = Strides::f_contiguous(&Ix1(5)).expect("valid");
        let storage = Owned::from_vec(vec![1_i32; 5]).expect("valid");
        let t = TensorBase::<Owned<i32>, Ix1> {
            storage,
            shape: Ix1(5),
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };
        assert_eq!(t.offset(), 0);
        assert!(t.flags().is_f_contiguous());
    }
    /// Verify raw_dim clones the dimension descriptor correctly.
    #[test]
    fn test_tensor_raw_dim() {
        let t = f_contig_i32(vec![1; 4], Ix2(2, 2));
        assert_eq!(t.raw_dim().slice(), &[2, 2]);
    }
    /// Verify is_empty returns true when an axis is zero.
    #[test]
    fn test_tensor_is_empty() {
        let t = f_contig_i32(Vec::<i32>::new(), Ix2(0, 3));
        assert!(t.is_empty());
    }


    /// Verify layout_state returns FContiguous for F_CONTIGUOUS flags.
    #[test]
    fn test_layout_state_f_contiguous() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::F_CONTIGUOUS, false, 0);
        assert!(t.is_f_contiguous());
        assert_eq!(t.layout_state(), LayoutState::FContiguous);
    }
    /// Verify layout_state returns NonContiguous for EMPTY flags.
    #[test]
    fn test_layout_state_non_contiguous() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::EMPTY, false, 0);
        assert!(!t.is_f_contiguous());
        assert_eq!(t.layout_state(), LayoutState::NonContiguous);
    }
    /// Verify is_aligned returns true when the aligned flag is set.
    #[test]
    fn test_flags_aligned() {
        let t = make_owned(
            Ix2(2, 3),
            vec![0.0; 6],
            LayoutFlags::F_CONTIGUOUS.set_aligned(true),
            false,
            0,
        );
        assert!(t.is_aligned());
    }
    /// Verify is_aligned returns false when the aligned flag is not set.
    #[test]
    fn test_not_aligned() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::EMPTY, false, 0);
        assert!(!t.is_aligned());
    }
    /// Verify has_zero_stride returns true when HAS_ZERO_STRIDE is set.
    #[test]
    fn test_has_zero_stride_set() {
        let t = make_owned(
            Ix2(2, 3),
            vec![0.0; 6],
            LayoutFlags::HAS_ZERO_STRIDE,
            false,
            0,
        );
        assert!(t.has_zero_stride());
    }
    /// Verify has_zero_stride returns false for F-contiguous layout.
    #[test]
    fn test_has_zero_stride_clear() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::F_CONTIGUOUS, false, 0);
        assert!(!t.has_zero_stride());
    }


    /// Verify as_storage_ptr returns a non-null pointer.
    #[test]
    fn test_as_storage_ptr() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert!(!t.as_storage_ptr().is_null());
    }
    /// Verify storage_len matches the underlying storage length.
    #[test]
    fn test_storage_len() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert_eq!(t.storage_len(), 4);
    }
    /// Verify as_ptr returns a non-null pointer equal to as_storage_ptr
    /// (offset = 0).
    #[test]
    fn test_as_ptr() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let ptr = t.as_ptr();
        assert!(!ptr.is_null());
        assert_eq!(ptr, t.as_storage_ptr());
    }
    /// Verify as_mut_ptr returns a non-null pointer through which
    /// element mutation is visible.
    #[test]
    fn test_as_mut_ptr() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let ptr = t.as_mut_ptr();
        assert!(!ptr.is_null());
        unsafe {
            *ptr = 99;
        }
        assert_eq!(t.as_slice().expect("F-contiguous")[0], 99);
    }
    /// Verify as_storage_mut_ptr returns a non-null pointer.
    #[test]
    fn test_as_storage_mut_ptr() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert!(!t.as_storage_mut_ptr().is_null());
    }
    /// Verify as_slice returns the expected contiguous data.
    #[test]
    fn test_as_slice() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert_eq!(t.as_slice().expect("F-contiguous"), &[1, 2, 3, 4]);
    }
    /// Verify as_mut_slice allows in-place mutation visible through as_slice.
    #[test]
    fn test_as_mut_slice() {
        let mut t = f_contig_i32(vec![1, 2, 3], Ix2(1, 3));
        t.as_mut_slice().expect("F-contiguous")[1] = 9;
        assert_eq!(t.as_slice().expect("F-contiguous"), &[1, 9, 3]);
    }
    /// Verify as_slice returns an empty slice for an empty tensor.
    #[test]
    fn test_as_slice_empty() {
        let t = f_contig_i32(Vec::new(), Ix2(0, 3));
        assert!(t.as_slice().expect("empty").is_empty());
    }
    /// Verify as_mut_slice returns an empty slice for an empty tensor.
    #[test]
    fn test_as_mut_slice_empty() {
        let mut t = f_contig_i32(Vec::new(), Ix2(0, 3));
        assert!(t.as_mut_slice().expect("empty").is_empty());
    }
    /// Verify as_slice returns None for non-contiguous layout.
    #[test]
    fn test_as_slice_non_contiguous() {
        let shape = Ix2(2, 2);
        let strides = Strides::f_contiguous(&shape).expect("valid");
        let storage = Owned::from_vec(vec![1_i32, 2, 3, 4]).expect("valid");
        let t: TensorBase<Owned<i32>, Ix2> = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::EMPTY,
            derived_from_view_mut: false,
        };
        assert!(t.as_slice().is_none());
    }


    /// Verify view shares data with source; mutation visible through both.
    #[test]
    fn test_view_data_shared() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let v = t.view();
        assert_eq!(v.shape(), t.shape());
        assert_eq!(v.as_slice().expect("F-contiguous"), &[1, 2, 3, 4]);
        let _ = v;
        t.as_mut_slice().expect("F-contiguous")[0] = 99;
        let v2 = t.view();
        assert_eq!(v2.as_slice().expect("F-contiguous")[0], 99);
    }
    /// Verify view_mut allows mutable access to the underlying data.
    #[test]
    fn test_view_mut_writable() {
        let mut t = f_contig_i32(vec![1, 2], Ix2(1, 2));
        t.view_mut().as_mut_slice().expect("F-contiguous")[0] = 9;
        assert_eq!(t.as_slice().expect("F-contiguous")[0], 9);
    }
    /// Verify view() on ViewMut sets derived_from_view_mut and reports
    /// SharedReadOnly semantics.
    #[test]
    fn test_view_from_view_mut_derived_flag() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let vm = t.view_mut();
        let downgraded = vm.view();
        assert!(downgraded.derived_from_view_mut);
        assert_eq!(
            downgraded.access_semantics(),
            AccessSemantics::SharedReadOnly
        );
    }
    /// Verify view_mut reborrow does not set derived_from_view_mut and
    /// reports Writable semantics.
    #[test]
    fn test_view_mut_reborrow_no_flag() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let mut vm1 = t.view_mut();
        let vm2 = vm1.view_mut();
        assert!(!vm2.derived_from_view_mut);
        assert_eq!(vm2.access_semantics(), AccessSemantics::Writable);
    }
    /// Verify view() on Owned reports ReadOnly and does not set
    /// derived_from_view_mut.
    #[test]
    fn test_view_from_owned_read_only() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let v = t.view();
        assert!(!v.derived_from_view_mut);
        assert_eq!(v.access_semantics(), AccessSemantics::ReadOnly);
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

    /// Empty tensor round-trip uses dangling sentinel for compute_layout_flags.
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

    /// from_raw_parts_owned rejects non-zero offset.
    #[test]
    fn test_from_raw_parts_owned_rejects_nonzero_offset() {
        let original = Tensor::<i32, Ix1>::from_vec(vec![1_i32, 2, 3]).expect("test input valid");
        let mut raw = original.into_raw_parts();
        raw.offset = 1;
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

    /// Tampered shape produces LenShapeMismatch error.
    #[test]
    fn test_from_raw_parts_owned_rejects_len_shape_mismatch() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let mut raw = original.into_raw_parts();
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

    /// Tampered strides reject with OwnedRequiresCanonicalFOrder.
    #[test]
    fn test_from_raw_parts_owned_rejects_non_canonical_strides() {
        let original = Tensor::<i32, Ix2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("test input valid");
        let mut raw = original.into_raw_parts();
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

    // ── construct (new_unchecked / validate / from_raw_parts) tests ──

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
            TensorBase::from_raw_vec_unchecked(vec![1_i32, 2, 3, 4], Ix2(2, 2))
        };
        assert_eq!(tensor.len(), 4);
        assert!(tensor.is_f_contiguous());
    }

    /// `from_raw_vec_unchecked` with empty vec and shape [0, 3]
    /// produces an empty F-contiguous tensor.
    #[test]
    fn test_from_raw_vec_unchecked_empty() {
        let tensor = unsafe {
            TensorBase::from_raw_vec_unchecked(Vec::<i32>::new(), Ix2(0, 3))
        };
        assert_eq!(tensor.len(), 0);
        assert!(tensor.is_f_contiguous());
    }

    /// `from_raw_vec_unchecked` with a 0-dimensional shape should succeed.
    #[test]
    fn test_from_raw_vec_unchecked_zero_dim() {
        let tensor = unsafe { TensorBase::from_raw_vec_unchecked(vec![42_i32], Ix0) };
        assert_eq!(tensor.ndim(), 0);
        assert_eq!(tensor.len(), 1);
    }

    // ── Semantic dispatch tests ──

    /// Verify storage_kind returns Owned for Owned-backed tensors.
    #[test]
    fn test_tensor_storage_kind_owned() {
        let shape = Ix2(1, 1);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(vec![1_i32; 1]).expect("valid vec");

        let t = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };
        assert_eq!(t.storage_kind(), StorageKind::Owned);
    }

    /// Verify access_semantics returns Owned for Owned-backed tensors.
    #[test]
    fn test_tensor_access_semantics_owned() {
        let shape = Ix2(1, 1);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(vec![1_i32; 1]).expect("valid vec");

        let t = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };
        assert_eq!(t.access_semantics(), AccessSemantics::Owned);
    }

    /// Verify alias_class returns Unique for F-contiguous Owned tensors.
    #[test]
    fn test_alias_class_unique_owned() {
        let shape = Ix2(1, 1);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(vec![1_i32; 1]).expect("valid vec");

        let t = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };
        assert_eq!(t.alias_class(), AliasClass::Unique);
    }

    /// Verify alias_class returns BroadcastAlias for zero-stride tensors.
    #[test]
    fn test_alias_class_broadcast() {
        let shape = Ix2(1, 1);
        let strides = Strides::new(Ix2(0, 1));
        let storage = Owned::from_vec(vec![1_i32; 1]).expect("valid vec");

        let t = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::HAS_ZERO_STRIDE,
            derived_from_view_mut: false,
        };
        assert_eq!(t.alias_class(), AliasClass::BroadcastAlias);
    }

    /// Verify alias_class returns ViewMutDerived when derived_from_view_mut
    /// is true.
    #[test]
    fn test_alias_class_view_mut_derived() {
        let shape = Ix2(1, 1);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(vec![1_i32; 1]).expect("valid vec");

        let t = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: true,
        };
        assert_eq!(t.alias_class(), AliasClass::ViewMutDerived);
    }
}
