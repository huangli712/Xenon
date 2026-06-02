//! Type conversion: `cast()`, `to_owned()`, and `into_owned()`.
//!
//! ## Submodules
//!
//! * `cast` — `pub(crate) CastTo` trait and its 36 tier-based impls
//!   covering the full 6×6 numeric element matrix.
//! * `impls` — Tensor-level `cast()`, `to_owned()`, and `into_owned()`
//!   methods on `TensorBase` with helpers.
//! * `types` — `pub trait CastElement`, the sealed compile-time gate
//!   that excludes `bool` and other non-numeric types from conversion.
//!
//! ## Public API
//!
//! Only `CastElement` is re-exported.  `CastTo` is `pub(crate)`; users
//! interact with conversion through `TensorBase::cast()`.

mod types;
mod cast;
mod impls;

pub use types::CastElement;
