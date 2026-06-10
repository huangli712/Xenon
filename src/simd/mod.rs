//! SIMD vectorized computation backend.
//!
//! This module is only compiled when the `simd` feature is enabled.
//! All items are `pub(crate)` — no public API exposure.
//!
//! ## Architecture
//!
//! - Facade functions (`dispatch_vector_*_op`)
//!   admit SIMD execution and return `bool`/`Option<A>` to signal
//!   acceptance. The caller **must** run its own scalar fallback on
//!   rejection.
//! - `get_arch()` caches a `pulp::Arch` singleton via `OnceLock`.

#[cfg(feature = "simd")]
mod types;

#[cfg(feature = "simd")]
mod binary;

#[cfg(feature = "simd")]
mod unary;

#[cfg(feature = "simd")]
mod driver;

pub(crate) use types::{BinaryOp, UnaryOp};

#[allow(unused_imports)]
pub(crate) use driver::{
    get_arch,
    simd_vector_width,
    dispatch_vector_binary_op,
    dispatch_vector_unary_op,
};
