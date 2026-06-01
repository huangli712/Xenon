//! Sealed dispatch trait for storage-type-dependent queries.
//!
//! [`StorageSemantics`] maps each concrete storage representation to its
//! [`StorageKind`], [`AccessSemantics`], and [`AliasClass`]. Code generic
//! over `S: StorageSemantics` can call [`TensorBase::storage_kind`],
//! [`TensorBase::access_semantics`], and [`TensorBase::alias_class`] without
//! knowing the concrete `S`.

use crate::element::Element;
use crate::layout::LayoutFlags;
use crate::storage::{ArcRepr, Owned, RawStorage, ViewMutRepr, ViewRepr};
use super::{AccessSemantics, AliasClass, StorageKind};

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
        derived_from_view_mut: bool
    ) -> AccessSemantics;

    /// Compute [`AliasClass`] for the given layout flags and provenance state.
    fn alias_class(
        flags: LayoutFlags,
        derived_from_view_mut: bool
    ) -> AliasClass;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutFlags;

    // ── Owned ──

    /// Owned storage kind is Owned.
    #[test]
    fn test_owned_kind() {
        assert_eq!(<Owned<i32>>::KIND, StorageKind::Owned);
    }

    /// Owned access_semantics always returns Owned regardless of
    /// flags/provenance.
    #[test]
    fn test_owned_access_semantics() {
        assert_eq!(
            <Owned<i32>>::access_semantics(LayoutFlags::F_CONTIGUOUS, false),
            AccessSemantics::Owned
        );
        assert_eq!(
            <Owned<i32>>::access_semantics(LayoutFlags::HAS_ZERO_STRIDE, true),
            AccessSemantics::Owned
        );
    }

    /// Owned alias_class returns Unique for F-contiguous, BroadcastAlias
    /// for zero-stride, ViewMutDerived when derived_from_view_mut is set.
    #[test]
    fn test_owned_alias_class() {
        assert_eq!(
            <Owned<i32>>::alias_class(LayoutFlags::F_CONTIGUOUS, false),
            AliasClass::Unique
        );
        assert_eq!(
            <Owned<i32>>::alias_class(LayoutFlags::HAS_ZERO_STRIDE, false),
            AliasClass::BroadcastAlias
        );
        assert_eq!(
            <Owned<i32>>::alias_class(LayoutFlags::F_CONTIGUOUS, true),
            AliasClass::ViewMutDerived
        );
    }

    // ── ViewRepr ──

    /// ViewRepr access_semantics: ReadOnly for plain, SharedReadOnly for
    /// zero-stride or ViewMut-derived.
    #[test]
    fn test_viewrepr_access_semantics() {
        assert_eq!(
            <ViewRepr<'_, i32>>::access_semantics(LayoutFlags::F_CONTIGUOUS, false),
            AccessSemantics::ReadOnly
        );
        assert_eq!(
            <ViewRepr<'_, i32>>::access_semantics(LayoutFlags::HAS_ZERO_STRIDE, false),
            AccessSemantics::SharedReadOnly
        );
        assert_eq!(
            <ViewRepr<'_, i32>>::access_semantics(LayoutFlags::F_CONTIGUOUS, true),
            AccessSemantics::SharedReadOnly
        );
    }

    /// ViewRepr alias_class: Unique for plain, BroadcastAlias for zero-stride,
    /// ViewMutDerived when derived_from_view_mut is set.
    #[test]
    fn test_viewrepr_alias_class() {
        assert_eq!(
            <ViewRepr<'_, i32>>::alias_class(LayoutFlags::F_CONTIGUOUS, false),
            AliasClass::Unique
        );
        assert_eq!(
            <ViewRepr<'_, i32>>::alias_class(LayoutFlags::HAS_ZERO_STRIDE, false),
            AliasClass::BroadcastAlias
        );
        assert_eq!(
            <ViewRepr<'_, i32>>::alias_class(LayoutFlags::F_CONTIGUOUS, true),
            AliasClass::ViewMutDerived
        );
    }

    // ── ViewMutRepr ──

    /// ViewMutRepr access_semantics always returns Writable.
    #[test]
    fn test_viewmutrepr_access_semantics() {
        assert_eq!(
            <ViewMutRepr<'_, i32>>::access_semantics(LayoutFlags::F_CONTIGUOUS, false),
            AccessSemantics::Writable
        );
    }

    /// ViewMutRepr alias_class: Unique for plain, BroadcastAlias for
    /// zero-stride.
    #[test]
    fn test_viewmutrepr_alias_class() {
        assert_eq!(
            <ViewMutRepr<'_, i32>>::alias_class(LayoutFlags::F_CONTIGUOUS, false),
            AliasClass::Unique
        );
        assert_eq!(
            <ViewMutRepr<'_, i32>>::alias_class(LayoutFlags::HAS_ZERO_STRIDE, false),
            AliasClass::BroadcastAlias
        );
    }

    // ── ArcRepr ──

    /// ArcRepr access_semantics always returns SharedReadOnly.
    #[test]
    fn test_arcrepr_access_semantics() {
        assert_eq!(
            <ArcRepr<i32>>::access_semantics(LayoutFlags::F_CONTIGUOUS, false),
            AccessSemantics::SharedReadOnly
        );
    }

    /// ArcRepr alias_class always returns ArcShared regardless of flags.
    #[test]
    fn test_arcrepr_alias_class() {
        assert_eq!(
            <ArcRepr<i32>>::alias_class(LayoutFlags::F_CONTIGUOUS, false),
            AliasClass::ArcShared
        );
        assert_eq!(
            <ArcRepr<i32>>::alias_class(LayoutFlags::HAS_ZERO_STRIDE, false),
            AliasClass::ArcShared
        );
    }
}
