//! Sealed marker trait for SIMD-capable element types.
//!
//! `SimdElement` lives in `element` rather than `simd` so the bound is
//! available in every feature configuration: the `BinaryArith` /
//! `UnaryArith` dispatch traits in `crate::math` reference it
//! unconditionally, while only the actual SIMD kernels under `crate::simd`
//! stay gated behind the `simd` feature.

use crate::complex::Complex;
use crate::private::Sealed;

/// Sealed marker trait for types that support SIMD lane operations.
///
/// Implemented for 6 concrete types:
/// `f32`, `f64`, `i32`, `i64`, `Complex<f32>`, `Complex<f64>`.
///
/// `Sealed` prevents downstream crates from adding new implementations.
/// Use `core::mem::size_of::<A>()` / `core::mem::align_of::<A>()` for
/// per-type size/alignment metadata — the compiler exposes the same values
/// without requiring trait-level redeclaration.
pub(crate) trait SimdElement: Sealed + Copy + Clone + Send + Sync + 'static {}

impl SimdElement for f32 {}
impl SimdElement for f64 {}
impl SimdElement for i32 {}
impl SimdElement for i64 {}
impl SimdElement for Complex<f32> {}
impl SimdElement for Complex<f64> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time verification that all 6 numeric element types
    /// implement the `SimdElement` marker trait.
    #[test]
    fn test_marker_trait_impls() {
        fn assert_simd<T: SimdElement>() {}
        assert_simd::<f32>();
        assert_simd::<f64>();
        assert_simd::<i32>();
        assert_simd::<i64>();
        assert_simd::<Complex<f32>>();
        assert_simd::<Complex<f64>>();
    }
}
