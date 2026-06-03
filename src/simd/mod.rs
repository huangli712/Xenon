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

use crate::complex::Complex;
use crate::private::Sealed;

#[cfg(feature = "simd")]
mod binary;
#[cfg(feature = "simd")]
mod dot;
#[cfg(feature = "simd")]
mod sum;
#[cfg(feature = "simd")]
mod unary;
#[cfg(feature = "simd")]
mod vector;
#[cfg(feature = "simd")]
use pulp::Arch;

// ---------------------------------------------------------------------------
// Arch cache
// ---------------------------------------------------------------------------

/// Returns a reference to the lazily-initialized static `pulp::Arch`.
///
/// The `OnceLock` is placed inside the function body so that external
/// code cannot bypass the accessor to read the cache directly.
#[cfg(feature = "simd")]
pub(crate) fn get_arch() -> &'static Arch {
    static ARCH: std::sync::OnceLock<Arch> = std::sync::OnceLock::new();
    ARCH.get_or_init(Arch::new)
}

// ---------------------------------------------------------------------------
// SimdElement — sealed marker trait
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Operation enums
// ---------------------------------------------------------------------------

/// Binary element-wise operation selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unary element-wise operation selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum UnaryOp {
    Neg,
    // Future: Abs, Square.
}

// ---------------------------------------------------------------------------
// Facade entry points — element-wise
// ---------------------------------------------------------------------------

/// Dispatches a binary element-wise operation to the SIMD backend.
///
/// Returns `true` if SIMD executed and wrote the result into `dst`.
/// Returns `false` if SIMD was rejected (too few elements, unsupported
/// type/ISA, etc.). The caller **must** run its own scalar fallback on
/// rejection.
///
/// # Panics
///
/// Panics if `lhs.len() != rhs.len()` or `lhs.len() != dst.len()`.
pub(crate) fn dispatch_vector_binary_op<A>(
    op: BinaryOp,
    lhs: &[A],
    rhs: &[A],
    dst: &mut [A],
) -> bool
where
    A: SimdElement,
{
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), dst.len());

    let tid = std::any::TypeId::of::<A>();
    // W14T2: f32/f64 element-wise dispatch via concrete kernels.
    if tid == std::any::TypeId::of::<f32>() {
        let lhs = unsafe { std::slice::from_raw_parts(lhs.as_ptr() as *const f32, lhs.len()) };
        let rhs = unsafe { std::slice::from_raw_parts(rhs.as_ptr() as *const f32, rhs.len()) };
        let dst =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len()) };
        return binary::dispatch_binary_f32(op, lhs, rhs, dst);
    }
    if tid == std::any::TypeId::of::<f64>() {
        let lhs = unsafe { std::slice::from_raw_parts(lhs.as_ptr() as *const f64, lhs.len()) };
        let rhs = unsafe { std::slice::from_raw_parts(rhs.as_ptr() as *const f64, rhs.len()) };
        let dst =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len()) };
        return binary::dispatch_binary_f64(op, lhs, rhs, dst);
    }
    // Complex<f32>/Complex<f64> element-wise handled by W14T11.
    if tid == std::any::TypeId::of::<Complex<f32>>() {
        let lhs =
            unsafe { std::slice::from_raw_parts(lhs.as_ptr() as *const Complex<f32>, lhs.len()) };
        let rhs =
            unsafe { std::slice::from_raw_parts(rhs.as_ptr() as *const Complex<f32>, rhs.len()) };
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f32>, dst.len())
        };
        return binary::dispatch_binary_complex_f32(op, lhs, rhs, dst);
    }
    if tid == std::any::TypeId::of::<Complex<f64>>() {
        let lhs =
            unsafe { std::slice::from_raw_parts(lhs.as_ptr() as *const Complex<f64>, lhs.len()) };
        let rhs =
            unsafe { std::slice::from_raw_parts(rhs.as_ptr() as *const Complex<f64>, rhs.len()) };
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f64>, dst.len())
        };
        return binary::dispatch_binary_complex_f64(op, lhs, rhs, dst);
    }
    // i32/i64 element-wise not supported (returns false).
    false
}

/// Dispatches a unary element-wise operation to the SIMD backend.
///
/// Semantics are the same as [`dispatch_vector_binary_op`]: `true` means
/// SIMD wrote `dst`, `false` means rejected.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
pub(crate) fn dispatch_vector_unary_op<A>(op: UnaryOp, src: &[A], dst: &mut [A]) -> bool
where
    A: SimdElement,
{
    assert_eq!(src.len(), dst.len());

    let tid = std::any::TypeId::of::<A>();
    // W14T2: f32/f64 element-wise Neg dispatch.
    if tid == std::any::TypeId::of::<f32>() {
        let src = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len()) };
        let dst =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len()) };
        return unary::dispatch_unary_f32(op, src, dst);
    }
    if tid == std::any::TypeId::of::<f64>() {
        let src = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f64, src.len()) };
        let dst =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len()) };
        return unary::dispatch_unary_f64(op, src, dst);
    }
    // Complex element-wise Neg handled by W14T11.
    if tid == std::any::TypeId::of::<Complex<f32>>() {
        let src =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const Complex<f32>, src.len()) };
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f32>, dst.len())
        };
        return unary::dispatch_unary_complex_f32(op, src, dst);
    }
    if tid == std::any::TypeId::of::<Complex<f64>>() {
        let src =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const Complex<f64>, src.len()) };
        let dst = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f64>, dst.len())
        };
        return unary::dispatch_unary_complex_f64(op, src, dst);
    }
    false
}

// ---------------------------------------------------------------------------
// Facade entry points — sum (reduction)
// ---------------------------------------------------------------------------

pub(crate) fn try_sum_f32(data: &[f32]) -> Option<f32> {
    vector::try_sum_f32_impl(data)
}

pub(crate) fn try_sum_f64(data: &[f64]) -> Option<f64> {
    vector::try_sum_f64_impl(data)
}

pub(crate) fn try_sum_complex_f32(data: &[Complex<f32>]) -> Option<Complex<f32>> {
    vector::try_sum_complex_f32_impl(data)
}

pub(crate) fn try_sum_complex_f64(data: &[Complex<f64>]) -> Option<Complex<f64>> {
    vector::try_sum_complex_f64_impl(data)
}

#[allow(
    dead_code,
    reason = "08-simd §6.6 (W14T4) capability stub paired with try_dot_i32 — \
              i32 sum via SIMD is unavailable in pulp 0.22 (no i32->i64 \
              widening). Always returns None so callers fall back to scalar \
              checked_add. Test in vector.rs verifies the contract \
              (admission path is prepared even though no production caller \
              wires through yet). (`allow` rather than `expect` because \
              dead_code only fires without `--tests`; test-mode use \
              suppresses the lint.)"
)]
pub(crate) fn try_sum_i32(data: &[i32]) -> Option<i32> {
    // W14T0 spike: i32->i64 widening is unavailable in pulp 0.22.
    // No SIMD path exists; always returns None so caller uses
    // scalar checked_add path (08-simd §6.6, W14T4).
    let _ = data;
    None
}

// ---------------------------------------------------------------------------
// Facade entry points — dot (inner product)
// ---------------------------------------------------------------------------

pub(crate) fn try_dot_f32(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    assert_eq!(lhs.len(), rhs.len());
    vector::try_dot_f32_impl(lhs, rhs)
}

pub(crate) fn try_dot_f64(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    assert_eq!(lhs.len(), rhs.len());
    vector::try_dot_f64_impl(lhs, rhs)
}

pub(crate) fn try_dot_complex_f32(
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
) -> Option<Complex<f32>> {
    assert_eq!(lhs.len(), rhs.len());
    vector::try_dot_complex_f32_impl(lhs, rhs)
}

pub(crate) fn try_dot_complex_f64(
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
) -> Option<Complex<f64>> {
    assert_eq!(lhs.len(), rhs.len());
    vector::try_dot_complex_f64_impl(lhs, rhs)
}

#[allow(
    dead_code,
    reason = "08-simd capability stub paired with try_sum_i32 — i32 dot via \
              SIMD is unavailable in pulp 0.22 (no i32->i64 widening). \
              Always returns None so callers fall back to scalar checked_mul. \
              Test in vector.rs verifies the contract (admission path is \
              prepared even though no production caller wires through yet). \
              (`allow` rather than `expect` because dead_code only fires \
              without `--tests`; test-mode use suppresses the lint.)"
)]
pub(crate) fn try_dot_i32(lhs: &[i32], rhs: &[i32]) -> Option<i32> {
    assert_eq!(lhs.len(), rhs.len());
    None
}

// ---------------------------------------------------------------------------
// Capability query
// ---------------------------------------------------------------------------

/// Returns the SIMD lane width for `T` on the current platform.
///
/// `Some(width > 1)` means the platform exposes a usable SIMD lane width.
/// `None` means the feature is disabled, the type is unsupported, or no
/// suitable ISA is available.
///
/// Per [`08-simd §5.12`], callers use this to decide whether to attempt
/// SIMD dispatch at all.
#[allow(
    dead_code,
    reason = "08-simd §5.12 capability-query skeleton — returns None until \
              ISA-specific implementations are wired in by later W14 tasks. \
              Test in this module verifies the skeleton contract. \
              (`allow` rather than `expect` because dead_code only fires \
              without `--tests`; test-mode use suppresses the lint.)"
)]
pub(crate) fn simd_vector_width<T: SimdElement>() -> Option<usize> {
    // Skeleton: returns None until ISA-specific implementations are filled in
    // by later W14 tasks. See 08-simd §5.12 for the capability-query contract.
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    #[test]
    fn test_empty_array() {
        let lhs: [f32; 0] = [];
        let rhs: [f32; 0] = [];
        let mut dst: [f32; 0] = [];

        assert!(!dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!dispatch_vector_unary_op(UnaryOp::Neg, &lhs, &mut dst));
        assert_eq!(try_sum_f32(&lhs), None);
        assert_eq!(try_dot_f32(&lhs, &rhs), None);
    }

    #[test]
    fn test_single_element() {
        let lhs = [2.0_f32];
        let rhs = [3.0_f32];
        let mut dst = [99.0_f32];

        assert!(!dispatch_vector_binary_op(
            BinaryOp::Mul,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert_eq!(dst, [99.0]);

        assert!(!dispatch_vector_unary_op(UnaryOp::Neg, &lhs, &mut dst));
        assert_eq!(dst, [99.0]);
    }

    #[test]
    fn test_simd_vector_width_skeleton_returns_none() {
        // Skeleton stage: capability query returns None for every supported
        // SimdElement type until later W14 tasks wire ISA lane widths.
        assert_eq!(simd_vector_width::<f32>(), None);
        assert_eq!(simd_vector_width::<f64>(), None);
        assert_eq!(simd_vector_width::<i32>(), None);
        assert_eq!(simd_vector_width::<i64>(), None);
        assert_eq!(simd_vector_width::<Complex<f32>>(), None);
        assert_eq!(simd_vector_width::<Complex<f64>>(), None);
    }
}
