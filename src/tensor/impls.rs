//! Query methods for [`TensorBase`]: shape/strides/ndim/len, layout flags,
//! storage kind, access semantics, alias classification, pointer access,
//! contiguous slices, and view creation.
//!
//! See `07-tensor.md §5.3–§5.5` for the authoritative API definitions.

use super::TensorBase;
use crate::element::Element;
use crate::storage::{ArcRepr, Owned, RawStorage, ViewMutRepr, ViewRepr};
use crate::storage::{Storage, StorageMut};

// ── Semantic query enums ──

/// Physical data location of the tensor payload.
///
/// Current version only supports CPU memory; GPU support deferred to later waves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocation {
    /// Data resides in CPU memory.
    Cpu,
}

/// Storage-representation classification returned by [`TensorBase::storage_kind`].
///
/// Reports the underlying storage *representation type*, not high-level access
/// semantics. See [`AccessSemantics`] for the caller-facing access model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    /// Owned storage (`Owned<A>`).
    Owned,
    /// Immutable borrowed view (`ViewRepr<'a, A>`).
    View,
    /// Mutable borrowed view (`ViewMutRepr<'a, A>`).
    ViewMut,
    /// Reference-counted shared storage (`ArcRepr<A>`).
    Shared,
}

/// Access semantics returned by [`TensorBase::access_semantics`].
///
/// Authoritative 4-variant set per `07-tensor.md §5.3` L378-383.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessSemantics {
    /// Plain non-broadcast read-only view.
    ReadOnly,
    /// Arc shared / broadcast / ViewMut-demoted view.
    SharedReadOnly,
    /// Exclusive mutable view.
    Writable,
    /// Owned storage.
    Owned,
}

// ── Basic query methods ──

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: crate::dimension::Dimension,
{
    /// Axis lengths. Zero-copy delegation to `Dimension::slice()`.
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }

    /// Strides in element units (usize, may be 0 for broadcast dims).
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    /// Number of dimensions (compile-time const for `Ix0`–`Ix6`, runtime for `IxDyn`).
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

    /// `true` iff any axis length is 0 (i.e. logical element count is 0).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Non-negative offset from storage base to logical first element.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Full layout flags (`F_CONTIGUOUS` / `ALIGNED` / `HAS_ZERO_STRIDE`).
    pub fn flags(&self) -> crate::layout::LayoutFlags {
        self.flags
    }

    /// Physical data location. Currently always [`DataLocation::Cpu`].
    pub fn data_location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: crate::dimension::Dimension + Clone,
{
    /// Clone of the dimension descriptor. Requires `D: Clone`.
    pub fn raw_dim(&self) -> D {
        self.shape.clone()
    }
}

// ── Layout query delegation ──

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: crate::dimension::Dimension,
{
    /// Returns the layout-state classification (`FContiguous` / `NonContiguous` / `BroadcastView`).
    pub fn layout_state(&self) -> crate::layout::LayoutState {
        self.flags.classify()
    }

    /// Returns `true` if the data is F-order contiguous.
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.is_f_contiguous()
    }

    /// Returns `true` if the data is at least 64-byte aligned.
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }

    /// Returns `true` iff `LayoutFlags::HAS_ZERO_STRIDE` is set (broadcast-induced zero stride).
    pub fn has_zero_stride(&self) -> bool {
        self.flags.has_zero_stride()
    }
}

// ── Dispatched storage_kind / access_semantics / alias_class ──
//
// Uses a sealed `StorageSemantics` trait to provide generic dispatch across
// all four concrete storage representations, so code generic over
// `S: RawStorage + StorageSemantics` can query storage kind, access
// semantics, and alias class without knowing the concrete `S`.
//
// Prior design used per‑type inherent impls (Owned / ViewRepr /
// ViewMutRepr / ArcRepr). That approach locked every new storage type
// into 3 extra impl blocks and made generic functions unable to call
// these methods. This trait‑based design eliminates both issues.

/// Sealed helper trait for callers writing generic helpers over
/// `TensorBase<S, D>`.
///
/// This trait is intentionally public so downstream code can name the bound
/// required by [`TensorBase::storage_kind`], [`TensorBase::access_semantics`],
/// and [`TensorBase::alias_class`] on generic `S`. It remains sealed because
/// [`RawStorage`] is sealed, so external crates cannot implement it for custom
/// storage types.
pub trait StorageSemantics: RawStorage {
    /// The [`StorageKind`] for this storage representation.
    const KIND: StorageKind;

    /// Compute [`AccessSemantics`] for the given layout flags and provenance
    /// state (the `derived_from_view_mut` flag on `TensorBase`).
    fn access_semantics(
        flags: crate::layout::LayoutFlags,
        derived_from_view_mut: bool,
    ) -> AccessSemantics;

    /// Compute [`AliasClass`] for the given layout flags and provenance state.
    fn alias_class(flags: crate::layout::LayoutFlags, derived_from_view_mut: bool) -> AliasClass;
}

// ── Implementations for the four sealed storage types ──

impl<A> StorageSemantics for Owned<A> {
    const KIND: StorageKind = StorageKind::Owned;
    fn access_semantics(_: crate::layout::LayoutFlags, _: bool) -> AccessSemantics {
        AccessSemantics::Owned
    }
    fn alias_class(flags: crate::layout::LayoutFlags, derived: bool) -> AliasClass {
        if flags.has_zero_stride() {
            AliasClass::BroadcastAlias
        } else if derived {
            AliasClass::ViewMutDerived
        } else {
            AliasClass::Unique
        }
    }
}

impl<A> StorageSemantics for ViewRepr<'_, A> {
    const KIND: StorageKind = StorageKind::View;
    fn access_semantics(flags: crate::layout::LayoutFlags, derived: bool) -> AccessSemantics {
        if flags.has_zero_stride() || derived {
            AccessSemantics::SharedReadOnly
        } else {
            AccessSemantics::ReadOnly
        }
    }
    fn alias_class(flags: crate::layout::LayoutFlags, derived: bool) -> AliasClass {
        if flags.has_zero_stride() {
            AliasClass::BroadcastAlias
        } else if derived {
            AliasClass::ViewMutDerived
        } else {
            AliasClass::Unique
        }
    }
}

impl<A> StorageSemantics for ViewMutRepr<'_, A> {
    const KIND: StorageKind = StorageKind::ViewMut;
    fn access_semantics(_: crate::layout::LayoutFlags, _: bool) -> AccessSemantics {
        AccessSemantics::Writable
    }
    fn alias_class(flags: crate::layout::LayoutFlags, _: bool) -> AliasClass {
        if flags.has_zero_stride() {
            AliasClass::BroadcastAlias
        } else {
            AliasClass::Unique
        }
    }
}

impl<A: Element> StorageSemantics for ArcRepr<A> {
    const KIND: StorageKind = StorageKind::Shared;
    fn access_semantics(_: crate::layout::LayoutFlags, _: bool) -> AccessSemantics {
        AccessSemantics::SharedReadOnly
    }
    fn alias_class(_: crate::layout::LayoutFlags, _: bool) -> AliasClass {
        AliasClass::ArcShared
    }
}

// ── Generic dispatch methods ──

impl<S, D> TensorBase<S, D>
where
    S: StorageSemantics,
    D: crate::dimension::Dimension,
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

// ── Pointer access & slice ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: crate::dimension::Dimension,
{
    /// Raw storage base pointer (does NOT add `offset`).
    pub fn as_storage_ptr(&self) -> *const A {
        self.storage.as_ptr()
    }

    /// Underlying storage buffer length in elements.
    pub fn storage_len(&self) -> usize {
        self.storage.len()
    }

    /// Raw pointer to the logical first element.
    ///
    /// For empty tensors returns `NonNull::dangling().as_ptr()`. Otherwise
    /// returns `storage.as_ptr().add(offset)`.
    pub fn as_ptr(&self) -> *const A {
        if self.is_empty() {
            core::ptr::NonNull::<A>::dangling().as_ptr() as *const A
        } else {
            unsafe { self.storage.as_ptr().add(self.offset) }
        }
    }

    /// Returns `Some(&[A])` for F-contiguous non-broadcast non-empty tensors,
    /// or for empty tensors. Returns `None` otherwise.
    pub fn as_slice(&self) -> Option<&[A]> {
        if self.is_empty() {
            return Some(unsafe { core::slice::from_raw_parts(self.as_ptr(), 0) });
        }
        if !self.flags.is_f_contiguous() || self.flags.has_zero_stride() {
            return None;
        }
        Some(unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) })
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: crate::dimension::Dimension,
{
    /// Raw mutable storage base pointer (does NOT add `offset`).
    pub fn as_storage_mut_ptr(&mut self) -> *mut A {
        self.storage.as_mut_ptr()
    }

    /// Raw mutable pointer to the logical first element.
    ///
    /// For empty tensors returns `NonNull::dangling().as_ptr()`.
    pub fn as_mut_ptr(&mut self) -> *mut A {
        if self.is_empty() {
            core::ptr::NonNull::<A>::dangling().as_ptr()
        } else {
            unsafe { self.storage.as_mut_ptr().add(self.offset) }
        }
    }

    /// Returns `Some(&mut [A])` for F-contiguous non-broadcast non-empty
    /// tensors, or for empty tensors. Returns `None` otherwise.
    pub fn as_mut_slice(&mut self) -> Option<&mut [A]> {
        if self.is_empty() {
            return Some(unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), 0) });
        }
        if !self.flags.is_f_contiguous() || self.flags.has_zero_stride() {
            return None;
        }
        let len = self.len();
        let ptr = self.as_mut_ptr();
        Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
    }
}

// ── AliasClass enum + alias_class() ──

/// Precise alias classification returned by [`TensorBase::alias_class`].
///
/// Unlike [`AccessSemantics::SharedReadOnly`] which merges three semantically
/// distinct categories, `AliasClass` splits them so callers can pattern-match
/// on alias origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasClass {
    /// No aliases: source is Owned or exclusive ViewMut.
    Unique,
    /// Arc shared ownership: multiple `ArcTensor` instances share a `SharedBuf`.
    ArcShared,
    /// Broadcast zero-stride alias: same physical element accessed by multiple
    /// logical indices.
    BroadcastAlias,
    /// Read-only view demoted from ViewMut (`derived_from_view_mut == true`).
    ViewMutDerived,
}

// ── view() / view_mut() ──
//
// view() is implemented per concrete storage type to avoid Rust's method
// resolution ambiguity between generic and specific impl blocks.

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: crate::dimension::Dimension + Clone,
{
    /// Creates an immutable view sharing the underlying storage.
    ///
    /// `derived_from_view_mut` is set to `false` for Owned-backed views.
    pub fn view(&self) -> TensorBase<ViewRepr<'_, A>, D> {
        // SAFETY: storage exposes valid base pointer + len
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

    /// Zero-copy conversion from `Tensor<A, D>` to `ArcTensor<A, D>`.
    ///
    /// Wraps the storage-layer `Owned<A>::into_shared(self) -> ArcRepr<A>`
    /// per `05-storage.md §5.11.3` line 628. Shape, strides, offset, and
    /// layout flags are preserved; `derived_from_view_mut` is `false` since
    /// `Owned`-backed tensors are never derived from a `ViewMut`.
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
}

impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D>
where
    D: crate::dimension::Dimension + Clone,
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

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    A: Element,
    D: crate::dimension::Dimension + Clone,
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

impl<A, D> TensorBase<ArcRepr<A>, D>
where
    A: Element,
    D: crate::dimension::Dimension + Clone,
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

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: crate::dimension::Dimension + Clone,
{
    /// Creates a mutable view sharing the underlying storage.
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

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    D: crate::dimension::Dimension + Clone,
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

// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Dimension, Ix1, Ix2};
    use crate::layout::{LayoutFlags, LayoutState, Strides};
    use crate::storage::Owned;

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

    // ── W8T4 tests ──

    #[test]
    fn test_tensor_shape_strides() {
        let t = f_contig_i32(vec![1; 12], Ix2(3, 4));
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.strides(), &[1_usize, 3]);
    }
    #[test]
    fn test_tensor_len_and_ndim() {
        let t = f_contig_i32(vec![1; 12], Ix2(3, 4));
        assert_eq!(t.len(), 12);
        assert_eq!(t.ndim(), 2);
    }
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
    #[test]
    fn test_tensor_raw_dim() {
        let t = f_contig_i32(vec![1; 4], Ix2(2, 2));
        assert_eq!(t.raw_dim().slice(), &[2, 2]);
    }
    #[test]
    fn test_tensor_is_empty() {
        let t = f_contig_i32(Vec::<i32>::new(), Ix2(0, 3));
        assert!(t.is_empty());
    }
    #[test]
    fn test_tensor_data_location() {
        let t = f_contig_i32(vec![1], Ix2(1, 1));
        assert_eq!(t.data_location(), DataLocation::Cpu);
    }
    #[test]
    fn test_tensor_storage_kind_owned() {
        let t = f_contig_i32(vec![1; 4], Ix2(2, 2));
        assert_eq!(t.storage_kind(), StorageKind::Owned);
    }
    #[test]
    fn test_tensor_access_semantics_owned() {
        let t = f_contig_i32(vec![1; 4], Ix2(2, 2));
        assert_eq!(t.access_semantics(), AccessSemantics::Owned);
    }

    // ── W8T5 tests ──

    #[test]
    fn test_layout_state_f_contiguous() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::F_CONTIGUOUS, false, 0);
        assert!(t.is_f_contiguous());
        assert_eq!(t.layout_state(), LayoutState::FContiguous);
    }
    #[test]
    fn test_layout_state_non_contiguous() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::EMPTY, false, 0);
        assert!(!t.is_f_contiguous());
        assert_eq!(t.layout_state(), LayoutState::NonContiguous);
    }
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
    #[test]
    fn test_not_aligned() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::EMPTY, false, 0);
        assert!(!t.is_aligned());
    }
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
    #[test]
    fn test_has_zero_stride_clear() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::F_CONTIGUOUS, false, 0);
        assert!(!t.has_zero_stride());
    }
    #[test]
    fn test_alias_class_unique_owned() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::F_CONTIGUOUS, false, 0);
        assert_eq!(t.alias_class(), AliasClass::Unique);
    }
    #[test]
    fn test_alias_class_broadcast() {
        let t = make_owned(
            Ix2(2, 3),
            vec![0.0; 6],
            LayoutFlags::HAS_ZERO_STRIDE,
            false,
            0,
        );
        assert_eq!(t.alias_class(), AliasClass::BroadcastAlias);
    }
    #[test]
    fn test_alias_class_view_mut_derived() {
        let t = make_owned(Ix2(2, 3), vec![0.0; 6], LayoutFlags::F_CONTIGUOUS, true, 0);
        assert_eq!(t.alias_class(), AliasClass::ViewMutDerived);
    }

    // ── W8T6 tests ──

    #[test]
    fn test_as_storage_ptr() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert!(!t.as_storage_ptr().is_null());
    }
    #[test]
    fn test_storage_len() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert_eq!(t.storage_len(), 4);
    }
    #[test]
    fn test_as_ptr() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let ptr = t.as_ptr();
        assert!(!ptr.is_null());
        assert_eq!(ptr, t.as_storage_ptr());
    }
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
    #[test]
    fn test_as_storage_mut_ptr() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert!(!t.as_storage_mut_ptr().is_null());
    }
    #[test]
    fn test_as_slice() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        assert_eq!(t.as_slice().expect("F-contiguous"), &[1, 2, 3, 4]);
    }
    #[test]
    fn test_as_mut_slice() {
        let mut t = f_contig_i32(vec![1, 2, 3], Ix2(1, 3));
        t.as_mut_slice().expect("F-contiguous")[1] = 9;
        assert_eq!(t.as_slice().expect("F-contiguous"), &[1, 9, 3]);
    }
    #[test]
    fn test_as_slice_empty() {
        let t = f_contig_i32(Vec::new(), Ix2(0, 3));
        assert!(t.as_slice().expect("empty").is_empty());
    }
    #[test]
    fn test_as_mut_slice_empty() {
        let mut t = f_contig_i32(Vec::new(), Ix2(0, 3));
        assert!(t.as_mut_slice().expect("empty").is_empty());
    }
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

    // ── W8T9 tests ──

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
    #[test]
    fn test_view_mut_writable() {
        let mut t = f_contig_i32(vec![1, 2], Ix2(1, 2));
        t.view_mut().as_mut_slice().expect("F-contiguous")[0] = 9;
        assert_eq!(t.as_slice().expect("F-contiguous")[0], 9);
    }
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
    #[test]
    fn test_view_mut_reborrow_no_flag() {
        let mut t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let mut vm1 = t.view_mut();
        let vm2 = vm1.view_mut();
        assert!(!vm2.derived_from_view_mut);
        assert_eq!(vm2.access_semantics(), AccessSemantics::Writable);
    }
    #[test]
    fn test_view_from_owned_read_only() {
        let t = f_contig_i32(vec![1, 2, 3, 4], Ix2(2, 2));
        let v = t.view();
        assert!(!v.derived_from_view_mut);
        assert_eq!(v.access_semantics(), AccessSemantics::ReadOnly);
    }
}
