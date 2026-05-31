//! Storage semantics trait: [`StorageSemantics`] and related dispatch methods.

use super::{AccessSemantics, AliasClass, StorageKind};
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

