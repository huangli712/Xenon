//! Tensor core: `TensorBase<S, D>`, type aliases, query methods, and
//! raw-parts construction.
//!
//! Sub-module ownership:
//! - `impls`     — query methods, view/view_mut, semantics dispatch
//! - `aliases`   — 36 type aliases (re-exported via `pub use aliases::*;`)
//! - `types`     — `TensorBase<S, D>` definition
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
//! `TensorBase<S, D>` is defined in [`types`] and re-exported here.
//! The `construct` module remains private, but it hosts both the internal
//! `new_unchecked` constructor and the public raw-parts entry points exposed
//! as `TensorBase`'s `from_raw_parts` and `from_raw_parts_mut`.

mod aliases;
mod impls;
mod types;

pub use aliases::*;
pub use types::OwnedRawParts;
pub use types::StorageSemantics;
pub use types::{DataLocation, StorageKind, AccessSemantics, AliasClass};
pub use types::TensorBase;
