//! Facade entry points for SIMD dot product.
//!
//! Each function admits SIMD execution and returns `Option<A>` to signal
//! acceptance. The caller **must** run its own scalar fallback on
//! rejection.

use crate::complex::Complex;
use super::dot_simd::{
    try_dot_complex_f32_impl,
    try_dot_complex_f64_impl,
    try_dot_f32_impl,
    try_dot_f64_impl,
};

// ----------------------------------------------------------------------------
// Facade entry points — dot (inner product)
// ----------------------------------------------------------------------------

/// Stub: i32 dot has no SIMD path (i32 widening unavailable).
/// Always returns `None` so callers fall back to scalar.
#[allow(dead_code, reason = "i32 dot stub — no SIMD widening available")]
pub(crate) fn try_dot_i32(lhs: &[i32], rhs: &[i32]) -> Option<i32> {
    assert_eq!(lhs.len(), rhs.len());
    None
}

/// Dispatches to SIMD f32 dot product; panics if lengths differ.
pub(crate) fn try_dot_f32(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    assert_eq!(lhs.len(), rhs.len());
    try_dot_f32_impl(lhs, rhs)
}

/// Dispatches to SIMD f64 dot product; panics if lengths differ.
pub(crate) fn try_dot_f64(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    assert_eq!(lhs.len(), rhs.len());
    try_dot_f64_impl(lhs, rhs)
}

/// Dispatches to SIMD `Complex<f32>` dot product (BLAS xdotc).
pub(crate) fn try_dot_complex_f32(
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
) -> Option<Complex<f32>> {
    assert_eq!(lhs.len(), rhs.len());
    try_dot_complex_f32_impl(lhs, rhs)
}

/// Dispatches to SIMD `Complex<f64>` dot product (BLAS xdotc).
pub(crate) fn try_dot_complex_f64(
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
) -> Option<Complex<f64>> {
    assert_eq!(lhs.len(), rhs.len());
    try_dot_complex_f64_impl(lhs, rhs)
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    /// Empty slices fall below the SIMD threshold and must return `None`.
    #[test]
    fn test_dot_empty_array() {
        let lhs: [f32; 0] = [];
        let rhs: [f32; 0] = [];
        assert_eq!(try_dot_f32(&lhs, &rhs), None);
    }
}
