//! Facade entry points for SIMD dispatch.
//!
//! Each function admits SIMD execution and returns `bool` / `Option<A>`
//! to signal acceptance. The caller **must** run its own scalar fallback
//! on rejection.

use pulp::Arch;

use std::slice;
use std::any::TypeId;
use std::sync::OnceLock;

use super::{binary, dot, sum, unary};
use crate::complex::Complex;
use crate::simd::{BinaryOp, SimdElement, UnaryOp};

// ----------------------------------------------------------------------------
// Arch cache
// ----------------------------------------------------------------------------

/// Returns a reference to the lazily-initialized static `pulp::Arch`.
///
/// The `OnceLock` is placed inside the function body so that external
/// code cannot bypass the accessor to read the cache directly.
pub(crate) fn get_arch() -> &'static Arch {
    static ARCH: OnceLock<Arch> = OnceLock::new();
    ARCH.get_or_init(Arch::new)
}

// ----------------------------------------------------------------------------
// Facade entry points — element-wise
// ----------------------------------------------------------------------------

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

    let tid = TypeId::of::<A>();
    if tid == TypeId::of::<f32>() {
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let lhs = unsafe {
            slice::from_raw_parts(lhs.as_ptr() as *const f32, lhs.len())
        };
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let rhs = unsafe {
            slice::from_raw_parts(rhs.as_ptr() as *const f32, rhs.len())
        };
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len())
        };
        return binary::dispatch_binary_f32(op, lhs, rhs, dst);
    }
    if tid == TypeId::of::<f64>() {
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let lhs = unsafe {
            slice::from_raw_parts(lhs.as_ptr() as *const f64, lhs.len())
        };
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let rhs = unsafe {
            slice::from_raw_parts(rhs.as_ptr() as *const f64, rhs.len())
        };
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len())
        };
        return binary::dispatch_binary_f64(op, lhs, rhs, dst);
    }
    if tid == TypeId::of::<Complex<f32>>() {
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is
        // identical to [T; 2]. The cast through raw pointers preserves
        // provenance and the length 2*n is correct.
        let lhs = unsafe {
            slice::from_raw_parts(lhs.as_ptr() as *const Complex<f32>, lhs.len())
        };
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is
        // identical to [T; 2]. The cast through raw pointers preserves
        // provenance and the length 2*n is correct.
        let rhs = unsafe {
            slice::from_raw_parts(rhs.as_ptr() as *const Complex<f32>, rhs.len())
        };
        // SAFETY: dst has the same layout guarantee as lhs/rhs — Complex<T>
        // is repr(C) with two T fields.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f32>, dst.len())
        };
        return binary::dispatch_binary_complex_f32(op, lhs, rhs, dst);
    }
    if tid == TypeId::of::<Complex<f64>>() {
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is
        // identical to [T; 2]. The cast through raw pointers preserves
        // provenance and the length 2*n is correct.
        let lhs = unsafe {
            slice::from_raw_parts(lhs.as_ptr() as *const Complex<f64>, lhs.len())
        };
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is
        // identical to [T; 2]. The cast through raw pointers preserves
        // provenance and the length 2*n is correct.
        let rhs = unsafe {
            slice::from_raw_parts(rhs.as_ptr() as *const Complex<f64>, rhs.len())
        };
        // SAFETY: dst has the same layout guarantee as lhs/rhs — Complex<T>
        // is repr(C) with two T fields.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f64>, dst.len())
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
pub(crate) fn dispatch_vector_unary_op<A>(
    op: UnaryOp,
    src: &[A],
    dst: &mut [A]
) -> bool
where
    A: SimdElement,
{
    assert_eq!(src.len(), dst.len());

    let tid = TypeId::of::<A>();
    if tid == TypeId::of::<f32>() {
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let src = unsafe {
            slice::from_raw_parts(src.as_ptr() as *const f32, src.len())
        };
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f32, dst.len())
        };
        return unary::dispatch_unary_f32(op, src, dst);
    }
    if tid == TypeId::of::<f64>() {
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let src = unsafe {
            slice::from_raw_parts(src.as_ptr() as *const f64, src.len())
        };
        // SAFETY: TypeId check confirmed the concrete type; the unsized
        // coercion from &[A] to &[T] through raw pointers is sound
        // because A == T.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, dst.len())
        };
        return unary::dispatch_unary_f64(op, src, dst);
    }
    if tid == TypeId::of::<Complex<f32>>() {
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is
        // identical to [T; 2]. The cast through raw pointers preserves
        // provenance and the length 2*n is correct.
        let src = unsafe {
            slice::from_raw_parts(src.as_ptr() as *const Complex<f32>, src.len())
        };
        // SAFETY: dst has the same layout guarantee as lhs/rhs — Complex<T>
        // is repr(C) with two T fields.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f32>, dst.len())
        };
        return unary::dispatch_unary_complex_f32(op, src, dst);
    }
    if tid == TypeId::of::<Complex<f64>>() {
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is
        // identical to [T; 2]. The cast through raw pointers preserves
        // provenance and the length 2*n is correct.
        let src = unsafe {
            slice::from_raw_parts(src.as_ptr() as *const Complex<f64>, src.len())
        };
        // SAFETY: dst has the same layout guarantee as lhs/rhs — Complex<T>
        // is repr(C) with two T fields.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut Complex<f64>, dst.len())
        };
        return unary::dispatch_unary_complex_f64(op, src, dst);
    }
    false
}

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
    sum::try_sum_f32_impl(data)
}

/// Dispatches to SIMD f64 sum; returns `None` if below threshold.
pub(crate) fn try_sum_f64(data: &[f64]) -> Option<f64> {
    sum::try_sum_f64_impl(data)
}

/// Dispatches to SIMD `Complex<f32>` sum; returns `None` if below threshold.
pub(crate) fn try_sum_complex_f32(
    data: &[Complex<f32>]
) -> Option<Complex<f32>> {
    sum::try_sum_complex_f32_impl(data)
}

/// Dispatches to SIMD `Complex<f64>` sum; returns `None` if below threshold.
pub(crate) fn try_sum_complex_f64(
    data: &[Complex<f64>]
) -> Option<Complex<f64>> {
    sum::try_sum_complex_f64_impl(data)
}

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
    dot::try_dot_f32_impl(lhs, rhs)
}

/// Dispatches to SIMD f64 dot product; panics if lengths differ.
pub(crate) fn try_dot_f64(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    assert_eq!(lhs.len(), rhs.len());
    dot::try_dot_f64_impl(lhs, rhs)
}

/// Dispatches to SIMD `Complex<f32>` dot product (BLAS xdotc).
pub(crate) fn try_dot_complex_f32(
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
) -> Option<Complex<f32>> {
    assert_eq!(lhs.len(), rhs.len());
    dot::try_dot_complex_f32_impl(lhs, rhs)
}

/// Dispatches to SIMD `Complex<f64>` dot product (BLAS xdotc).
pub(crate) fn try_dot_complex_f64(
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
) -> Option<Complex<f64>> {
    assert_eq!(lhs.len(), rhs.len());
    dot::try_dot_complex_f64_impl(lhs, rhs)
}

// ----------------------------------------------------------------------------
// Capability query
// ----------------------------------------------------------------------------

/// Returns the SIMD lane width for `T` on the current platform.
///
/// `Some(width > 1)` means the platform exposes a usable SIMD lane width.
/// `None` means the feature is disabled, the type is unsupported, or no
/// suitable ISA is available.
///
#[allow(dead_code, reason = "returns None until ISA dispatch is wired")]
pub(crate) fn simd_vector_width<T: SimdElement>() -> Option<usize> {
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    // ---- dispatch threshold rejection ---------------------------------------

    /// Verifies that slices below the element-wise threshold are
    /// rejected by both binary and unary dispatch, leaving `dst` untouched.
    #[test]
    fn test_vector_sub_mul_div_below_threshold_rejects() {
        let lhs: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let mut dst = vec![99.0f32; lhs.len()];

        // len=32 < threshold 64 — must reject
        assert!(!dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!dispatch_vector_binary_op(
            BinaryOp::Mul,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!dispatch_vector_unary_op(
            UnaryOp::Neg,
            &lhs,
            &mut dst
        ));
        // dst should remain unchanged on rejection
        for &v in &dst {
            assert_eq!(v, 99.0_f32, "dst must be untouched on SIMD rejection");
        }
    }

    // ---- empty / single element edge case -----------------------------------

    /// Empty slices must be rejected by element-wise dispatch and
    /// return `None` from sum/dot.
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

    /// Single-element slices are below threshold and must be rejected.
    /// `dst` must remain untouched on rejection.
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
}
