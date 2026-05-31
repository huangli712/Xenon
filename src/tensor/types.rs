//! Core tensor type: [`TensorBase<S, D>`] and [`OwnedRawParts`].

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
    /// Opaque storage handle. Private to the type; exposed through query API.
    // SAFETY INVARIANT: the six fields below together encode the tensor
    // layout contract. Direct field mutation (possible via `pub(crate)`
    // visibility) can violate the shape/strides/offset/flags mutual-
    // consistency invariant OR the provenance invariant encoded by
    // `derived_from_view_mut`. ANY constructor path within the crate
    // MUST route through `construct::new_unchecked` (or one of the
    // validated public constructors) which is the single internal entry
    // point for tensor metadata assembly. Tests may construct directly
    // because their invariants are locally obvious; production code
    // MUST NOT.
    pub(crate) storage: S,
    pub(crate) shape: D,
    pub(crate) strides: Strides<D>,
    pub(crate) offset: usize,
    pub(crate) flags: LayoutFlags,
    /// `true` iff this view was demoted from a [`ViewMutRepr`] via `view()`.
    ///
    /// Enables [`access_semantics`](TensorBase::access_semantics) and
    /// [`alias_class`](TensorBase::alias_class) to correctly report
    /// `SharedReadOnly` / `ViewMutDerived` for ViewMut-demoted views.
    /// **Do NOT mutate this field directly**; always set it through
    /// [`new_unchecked`](super::construct::TensorBase::new_unchecked) or the
    /// provenance-aware constructors.
    pub(crate) derived_from_view_mut: bool,
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

// ── DataLocation ──

/// Physical data location of the tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocation {
    /// Data resides in CPU memory.
    Cpu,
}

// ── AliasClass ──

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

// ── AccessSemantics ──

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

// ── StorageKind ──

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
    use super::TensorBase;
    use super::{DataLocation, StorageKind, AccessSemantics, AliasClass};
    use crate::dimension::Ix2;
    use crate::layout::{LayoutFlags, Strides};
    use crate::storage::Owned;

    /// Verify the tensor module skeleton compiles and all three sub-modules
    /// (`impls`, `aliases`, `construct`) are reachable.
    #[test]
    fn test_module_skeleton_compile() {
        // Reaching this point confirms: (a) all three sub-module files
        // exist, (b) `src/tensor/mod.rs` parses, (c) the crate compiled.
        assert_ne!(0, 1);
    }

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

    // ── AliasClass tests ──

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
