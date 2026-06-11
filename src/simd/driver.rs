//! Facade entry points for SIMD dispatch.
//!
//! Each function admits SIMD execution and returns `bool` / `Option<A>`
//! to signal acceptance. The caller **must** run its own scalar fallback
//! on rejection.

use pulp::Arch;

use std::slice;
use std::any::TypeId;
use std::sync::OnceLock;

use super::binary;
use crate::complex::Complex;
use crate::element::SimdElement;
use crate::simd::BinaryOp;

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

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    // ---- dispatch threshold rejection --------------------------------------

    /// Verifies that slices below the element-wise threshold are rejected by
    /// binary dispatch, leaving `dst` untouched.
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
        // dst should remain unchanged on rejection
        for &v in &dst {
            assert_eq!(v, 99.0_f32, "dst must be untouched on SIMD rejection");
        }
    }

    // ---- empty / single element edge case ----------------------------------

    /// Empty slices must be rejected by element-wise dispatch.
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
    }
}
