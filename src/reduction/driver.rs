//! Facade entry points for SIMD sum reduction.
//!
//! Each function admits SIMD execution and returns `Option<A>` to signal
//! acceptance. The caller **must** run its own scalar fallback on
//! rejection.

use super::sum_simd;
use crate::complex::Complex;

// ----------------------------------------------------------------------------
// Facade entry points — sum (reduction)
// ----------------------------------------------------------------------------

/// Stub: i32 sum has no SIMD path (i32 widening unavailable).
/// Always returns `None` so callers fall back to scalar.
#[allow(dead_code, reason = "i32 sum stub — no SIMD widening available")]
pub(crate) fn try_sum_i32(data: &[i32]) -> Option<i32> {
    let _ = data;
    None
}

/// Dispatches to SIMD f32 sum; returns `None` if below threshold.
pub(crate) fn try_sum_f32(data: &[f32]) -> Option<f32> {
    sum_simd::try_sum_f32_impl(data)
}

/// Dispatches to SIMD f64 sum; returns `None` if below threshold.
pub(crate) fn try_sum_f64(data: &[f64]) -> Option<f64> {
    sum_simd::try_sum_f64_impl(data)
}

/// Dispatches to SIMD `Complex<f32>` sum; returns `None` if below threshold.
pub(crate) fn try_sum_complex_f32(
    data: &[Complex<f32>]
) -> Option<Complex<f32>> {
    sum_simd::try_sum_complex_f32_impl(data)
}

/// Dispatches to SIMD `Complex<f64>` sum; returns `None` if below threshold.
pub(crate) fn try_sum_complex_f64(
    data: &[Complex<f64>]
) -> Option<Complex<f64>> {
    sum_simd::try_sum_complex_f64_impl(data)
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    /// Empty slices fall below the SIMD threshold and must return `None`.
    #[test]
    fn test_sum_empty_array() {
        let data: [f32; 0] = [];
        assert_eq!(try_sum_f32(&data), None);
    }
}
