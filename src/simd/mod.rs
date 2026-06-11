//! SIMD vectorized computation backend.
//!
//! This module is only compiled when the `simd` feature is enabled.
//! All items are `pub(crate)` — no public API exposure.
//!
//! ## Architecture
//!
//! - `get_arch()` caches a `pulp::Arch` singleton via `OnceLock`.

#[cfg(feature = "simd")]
mod types;

#[cfg(feature = "simd")]
mod driver;

pub(crate) use types::{BinaryOp, UnaryOp};

#[allow(unused_imports)]
pub(crate) use driver::{
    get_arch,
    simd_vector_width,
};
