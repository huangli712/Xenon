//! Type conversion: `cast()`, `to_owned()`, and `into_owned()`.
//!
//! ## Submodules
//!
//! * `cast` — `pub(crate) CastTo` trait and its 36 tier-based impls
//!   covering the full 6×6 numeric element matrix.
//! * `impls` — Tensor-level `cast()`, `to_owned()`, and `into_owned()`
//!   methods on `TensorBase` with helpers.
//!
//! ## Public API
//!
//! `CastElement` (defined in `crate::element`) is re-exported here for path
//! stability.  `CastTo` is `pub(crate)`; users interact with conversion
//! through `TensorBase::cast()`.

mod cast;
mod impls;
