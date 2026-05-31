//! Storage semantics trait: [`StorageSemantics`] and related dispatch methods.

use super::TensorBase;
use super::{AccessSemantics, AliasClass, StorageKind};
use crate::dimension::Dimension;
use crate::element::Element;
use crate::layout::LayoutFlags;
use crate::storage::{ArcRepr, Owned, RawStorage, ViewMutRepr, ViewRepr};

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
        flags: LayoutFlags,
        derived_from_view_mut: bool,
    ) -> AccessSemantics;

    /// Compute [`AliasClass`] for the given layout flags and provenance state.
    fn alias_class(flags: LayoutFlags, derived_from_view_mut: bool) -> AliasClass;
}

// ── Implementations for the four sealed storage types ──

impl<A> StorageSemantics for Owned<A> {
    const KIND: StorageKind = StorageKind::Owned;
    fn access_semantics(_: LayoutFlags, _: bool) -> AccessSemantics {
        AccessSemantics::Owned
    }
    fn alias_class(flags: LayoutFlags, derived: bool) -> AliasClass {
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
    fn access_semantics(flags: LayoutFlags, derived: bool) -> AccessSemantics {
        if flags.has_zero_stride() || derived {
            AccessSemantics::SharedReadOnly
        } else {
            AccessSemantics::ReadOnly
        }
    }
    fn alias_class(flags: LayoutFlags, derived: bool) -> AliasClass {
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
    fn access_semantics(_: LayoutFlags, _: bool) -> AccessSemantics {
        AccessSemantics::Writable
    }
    fn alias_class(flags: LayoutFlags, _: bool) -> AliasClass {
        if flags.has_zero_stride() {
            AliasClass::BroadcastAlias
        } else {
            AliasClass::Unique
        }
    }
}

impl<A: Element> StorageSemantics for ArcRepr<A> {
    const KIND: StorageKind = StorageKind::Shared;
    fn access_semantics(_: LayoutFlags, _: bool) -> AccessSemantics {
        AccessSemantics::SharedReadOnly
    }
    fn alias_class(_: LayoutFlags, _: bool) -> AliasClass {
        AliasClass::ArcShared
    }
}

// ── Generic dispatch methods ──

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix2;
    use crate::layout::{LayoutFlags, Strides};
    use crate::storage::Owned;

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
