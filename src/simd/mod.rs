//! SIMD vectorized computation backend.
//!
//! This module is only compiled when the `simd` feature is enabled.
//! All items are `pub(crate)` — no public API exposure.
//!
//! ## Architecture
//!
//! - [`SimdElement`]: sealed marker trait for types with SIMD lane support.
//! - Facade functions (`dispatch_vector_*_op`, `try_sum_*`, `try_dot_*`)
//!   admit SIMD execution and return `bool`/`Option<A>` to signal
//!   acceptance. The caller **must** run its own scalar fallback on
//!   rejection.
//! - `get_arch()` caches a `pulp::Arch` singleton via `OnceLock`.

#[cfg(feature = "simd")]
mod binary;
#[cfg(feature = "simd")]
mod dot;
#[cfg(feature = "simd")]
mod sum;
#[cfg(feature = "simd")]
mod driver;
#[cfg(feature = "simd")]
mod types;
#[cfg(feature = "simd")]
mod unary;
#[cfg(feature = "simd")]
mod vector;

// ---------------------------------------------------------------------------
// Re-exports so existing imports still resolve
// ---------------------------------------------------------------------------

pub(crate) use driver::{
    dispatch_vector_binary_op,
    dispatch_vector_unary_op,
    get_arch,
    simd_vector_width,
    try_dot_complex_f32,
    try_dot_complex_f64,
    try_dot_f32,
    try_dot_f64,
    try_dot_i32,
    try_sum_complex_f32,
    try_sum_complex_f64,
    try_sum_f32,
    try_sum_f64,
    try_sum_i32,
};
pub(crate) use types::{BinaryOp, SimdElement, UnaryOp};


