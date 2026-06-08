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
