//! Public types of the tensor module.
//!
//! This module defines [`TensorBase<S, D>`] — the central n‑dimensional array
//! — together with its raw‑parts decomposition [`OwnedRawParts`] and the
//! semantic query enums [`DataLocation`], [`StorageKind`], [`AccessSemantics`],
//! and [`AliasClass`].

use crate::dimension::Dimension;
use crate::layout::{LayoutFlags, Strides};
use crate::storage::RawStorage;

/// N-dimensional array with type-level storage and dimension descriptors.
///
/// `TensorBase<S, D>` is the central type of the tensor module. It pairs a
/// storage representation `S` (one of [`Owned`], [`ViewRepr`], [`ViewMutRepr`],
/// or [`ArcRepr`]) with a dimension descriptor `D` (one of [`Ix0`]–[`Ix6`] or
/// [`IxDyn`]), and carries six metadata fields: [`storage`], [`shape`],
/// [`strides`], [`offset`], [`flags`], and [`derived_from_view_mut`].
///
/// The struct intentionally does **not** implement `Debug`; field-level
/// introspection is provided through query methods on [`TensorBase`]. The
/// internal field layout is not part of the public API and may change across
/// minor versions.
///
/// [`Owned`]: crate::storage::Owned
/// [`ViewRepr`]: crate::storage::ViewRepr
/// [`ViewMutRepr`]: crate::storage::ViewMutRepr
/// [`ArcRepr`]: crate::storage::ArcRepr
/// [`Ix0`]: crate::dimension::Ix0
/// [`Ix6`]: crate::dimension::Ix6
/// [`IxDyn`]: crate::dimension::IxDyn
/// [`storage`]: #structfield.storage
/// [`shape`]: #structfield.shape
/// [`strides`]: #structfield.strides
/// [`offset`]: #structfield.offset
/// [`flags`]: #structfield.flags
/// [`derived_from_view_mut`]: #structfield.derived_from_view_mut
pub struct TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    // SAFETY INVARIANT: the six fields below together encode the tensor
    // layout contract. Direct field mutation (possible via `pub(crate)`
    // visibility) can violate the shape/strides/offset/flags mutual-
    // consistency invariant OR the provenance invariant encoded by
    // `derived_from_view_mut`. ANY constructor path within the crate
    // MUST route through `new_unchecked` (or one of the validated public
    // constructors) which is the single internal entry point for tensor
    // metadata assembly. Tests may construct directly because their
    // invariants are locally obvious; production code MUST NOT.

    /// Opaque storage handle. Private to the type; exposed through query API.
    pub(crate) storage: S,

    /// Axis lengths. Zero-copy exposed via [`TensorBase::shape`].
    pub(crate) shape: D,
    
    /// Strides in element units, may be zero for broadcast axes.
    pub(crate) strides: Strides<D>,
    
    /// Offset from storage base to logical first element.
    pub(crate) offset: usize,
    
    /// Layout flags (F-contiguous, aligned, zero-stride).
    pub(crate) flags: LayoutFlags,
    
    /// `true` iff this view was demoted from a [`ViewMutRepr`] via `view()`.
    ///
    /// Enables [`access_semantics`] and [`alias_class`] to correctly report
    /// `SharedReadOnly` / `ViewMutDerived` for ViewMut-demoted views.
    /// **Do NOT mutate this field directly**; always set it through
    /// [`new_unchecked`] or the provenance-aware constructors.
    pub(crate) derived_from_view_mut: bool,
}

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
    /// Pointer to the storage base.
    pub ptr: *mut A,

    /// Logical length in elements.
    pub len: usize,

    /// Allocator capacity in elements.
    pub cap: usize,
    
    /// Allocator alignment in bytes.
    pub align: usize,
    
    /// Owned tensor shape.
    pub shape: D,
    
    /// Canonical F-order strides matching shape.
    pub strides: Strides<D>,
    
    /// Logical offset in element units.
    pub offset: usize,
}

/// Physical data location of the tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocation {
    /// Data resides in CPU memory.
    Cpu,
}

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
    
    /// Broadcast zero-stride alias.
    BroadcastAlias,
    
    /// Read-only view demoted from ViewMut.
    ViewMutDerived,
}

/// Access semantics returned by [`TensorBase::access_semantics`].
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

#[cfg(test)]
mod tests {
    use super::{AccessSemantics, AliasClass, DataLocation, OwnedRawParts, StorageKind, TensorBase};
    use crate::dimension::{Dimension as _, Ix2};
    use crate::layout::{LayoutFlags, Strides};
    use crate::storage::Owned;

    /// Verify `TensorBase` struct fields hold the values assigned to them
    /// after direct construction.
    #[test]
    fn test_tensorbase_struct_fields() {
        let data = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = Ix2(2, 3);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(data).expect("valid vec");

        let tensor = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };

        // Verify pub(crate) fields hold the values assigned
        assert_eq!(tensor.offset, 0);
        assert!(!tensor.derived_from_view_mut);
        assert!(tensor.flags.is_f_contiguous());
    }

    /// Verify data_location returns Cpu.
    #[test]
    fn test_tensor_data_location() {
        let data = vec![1_i32];
        let shape = Ix2(1, 1);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let storage = Owned::from_vec(data).expect("valid vec");

        let t = TensorBase {
            storage,
            shape,
            strides,
            offset: 0,
            flags: LayoutFlags::F_CONTIGUOUS,
            derived_from_view_mut: false,
        };
        assert_eq!(t.data_location(), DataLocation::Cpu);
    }
    /// Verify `DataLocation::Cpu` can be constructed and derives Debug/Clone/Copy/PartialEq/Eq.
    #[test]
    fn test_data_location_variants() {
        let loc = DataLocation::Cpu;
        let _copy = loc; // Copy
        assert_eq!(loc, _copy); // PartialEq
        assert_eq!(format!("{:?}", loc), "Cpu"); // Debug
    }

    /// Verify `StorageKind` variants can be constructed and derives work.
    #[test]
    fn test_storage_kind_variants() {
        let v = StorageKind::Owned;
        let _c = v; // Copy
        assert_eq!(v, StorageKind::Owned); // PartialEq
        assert!(!matches!(v, StorageKind::View)); // pattern match
        let owned = StorageKind::View;
        let shared = StorageKind::Shared;
        let viewmut = StorageKind::ViewMut;
        assert_eq!(format!("{:?}", owned), "View");
        assert_eq!(format!("{:?}", shared), "Shared");
        assert_eq!(format!("{:?}", viewmut), "ViewMut");
    }

    /// Verify `AccessSemantics` variants can be constructed and derives work.
    #[test]
    fn test_access_semantics_variants() {
        let ro = AccessSemantics::ReadOnly;
        let _c = ro; // Copy
        let owned = AccessSemantics::Owned;
        assert_eq!(ro, AccessSemantics::ReadOnly);
        assert_eq!(owned, AccessSemantics::Owned);
        assert_ne!(AccessSemantics::ReadOnly, AccessSemantics::Writable);
        assert_eq!(format!("{:?}", AccessSemantics::SharedReadOnly), "SharedReadOnly");
    }

    /// Verify `AliasClass` variants can be constructed and derives work.
    #[test]
    fn test_alias_class_variants() {
        let u = AliasClass::Unique;
        let _c = u; // Copy
        assert_eq!(u, AliasClass::Unique);
        assert_ne!(u, AliasClass::BroadcastAlias);
        assert_eq!(format!("{:?}", AliasClass::ViewMutDerived), "ViewMutDerived");
    }

    /// Verify `OwnedRawParts` fields can be constructed and are pub-crate accessible.
    #[test]
    fn test_owned_raw_parts_fields() {
        let shape = Ix2(2, 3);
        let strides = Strides::f_contiguous(&shape).expect("valid shape");
        let raw: OwnedRawParts<f64, Ix2> = OwnedRawParts {
            ptr: std::ptr::null_mut::<f64>(),
            len: 6,
            cap: 8,
            align: 64,
            shape,
            strides,
            offset: 0,
        };
        assert_eq!(raw.len, 6);
        assert_eq!(raw.cap, 8);
        assert!(raw.align.is_power_of_two());
        assert_eq!(raw.offset, 0);
        assert_eq!(raw.shape.slice(), &[2, 3]);
    }
}
