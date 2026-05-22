//! Tensor core: `TensorBase<S, D>`, type aliases, query methods, and
//! raw-parts construction. See `07-tensor.md §3` for the file layout
//! rationale.
//!
//! Sub-module ownership:
//! - `impls`     — query methods, view/view_mut, semantics dispatch
//! - `aliases`   — 36 type aliases (re-exported via `pub use aliases::*;`)
//! - `construct` — crate-internal constructors and validators
//!
//! ## Public re-exports
//!
//! From `impls.rs`:
//! - `AccessSemantics`, `AliasClass`, `DataLocation`, `StorageKind`,
//!   `StorageSemantics`
//!
//! From `aliases.rs` (via `pub use aliases::*;`):
//! - 4 primary aliases: `Tensor`, `TensorView`, `TensorViewMut`, `ArcTensor`
//! - 32 convenience aliases: `Tensor0`..`Tensor6`, `TensorD` and their
//!   `View`/`ViewMut`/`Arc` counterparts
//!
//! `TensorBase<S, D>` itself is defined in this `mod.rs`.
//! The `construct` module remains private, but it hosts both the internal
//! `new_unchecked` constructor and the public raw-parts entry points exposed
//! as `TensorBase`'s `from_raw_parts` and `from_raw_parts_mut`.

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
#[expect(
    missing_debug_implementations,
    reason = "intentionally no Debug; see struct doc"
)]
pub struct TensorBase<S, D>
where
    S: crate::storage::RawStorage,
    D: crate::dimension::Dimension,
{
    /// Opaque storage handle. Private to the type; exposed through query API.
    // SAFETY INVARIANT: the six fields below together establish the tensor
    // layout contract described in `07-tensor.md §5`. Direct field mutation
    // (possible via `pub(crate)` visibility) can violate the shape/strides/
    // offset/flags mutual-consistency invariant OR the provenance invariant
    // encoded by `derived_from_view_mut`. ANY constructor path within the
    // crate MUST route through `construct::new_unchecked` (or one of the
    // validated public constructors) which is the single internal entry
    // point for tensor metadata assembly. Tests may construct directly
    // because their invariants are locally obvious; production code MUST NOT.
    pub(crate) storage: S,
    pub(crate) shape: D,
    pub(crate) strides: crate::layout::Strides<D>,
    pub(crate) offset: usize,
    pub(crate) flags: crate::layout::LayoutFlags,
    /// `true` iff this view was demoted from a [`ViewMutRepr`] via `view()`.
    ///
    /// Enables [`access_semantics`](TensorBase::access_semantics) and
    /// [`alias_class`](TensorBase::alias_class) to correctly report
    /// `SharedReadOnly` / `ViewMutDerived` for ViewMut-demoted views.
    /// **Do NOT mutate this field directly**; always set it through
    /// [`new_unchecked`](construct::TensorBase::new_unchecked) or the
    /// provenance-aware constructors.
    pub(crate) derived_from_view_mut: bool,
}

mod aliases;
mod construct;
mod impls;

pub use aliases::*;
pub use construct::OwnedRawParts;
pub use impls::AliasClass;
pub use impls::StorageSemantics;
pub use impls::{AccessSemantics, DataLocation, StorageKind};

// Re-exports are added incrementally by downstream tasks; see W8T2..W8T9.

#[cfg(test)]
mod tests {
    use super::TensorBase;
    use crate::dimension::Ix2;
    use crate::layout::{LayoutFlags, compute_f_strides};
    use crate::storage::Owned;

    /// Verify the tensor module skeleton compiles and all three sub-modules
    /// are reachable. If the placeholder `.rs` files in Step 1 are missing,
    /// the outer `mod impls;` / `mod aliases;` / `mod construct;` declarations
    /// fail at `cargo check`, and this test will never even be invoked.
    #[test]
    fn test_module_skeleton_compile() {
        // Reaching this point means: (a) all three sub-module files exist,
        // (b) `src/tensor/mod.rs` parses, (c) the crate compiled. The
        // behavioural tests for query/view methods are added in W8T4+ once
        // the underlying APIs exist.
        assert_ne!(0, 1);
    }

    #[test]
    fn test_tensorbase_struct_fields() {
        let data = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = Ix2(2, 3);
        let strides = compute_f_strides(&shape).expect("valid shape");
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
}
