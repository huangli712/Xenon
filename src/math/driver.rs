//! Facade entry point for SIMD unary dispatch.
//!
//! Admits SIMD execution and returns `bool` to signal acceptance. The
//! caller **must** run its own scalar fallback on rejection.

use std::slice;
use std::any::TypeId;

use crate::complex::Complex;
use crate::element::SimdElement;
use crate::simd::UnaryOp;

use super::unary_simd;

// ----------------------------------------------------------------------------
// Facade entry point — unary element-wise
// ----------------------------------------------------------------------------

/// Dispatches a unary element-wise operation to the SIMD backend.
///
/// Returns `true` if SIMD executed and wrote the result into `dst`.
/// Returns `false` if SIMD was rejected (too few elements, unsupported
/// type/ISA, etc.). The caller **must** run its own scalar fallback on
/// rejection.
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
        return unary_simd::dispatch_unary_f32(op, src, dst);
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
        return unary_simd::dispatch_unary_f64(op, src, dst);
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
        return unary_simd::dispatch_unary_complex_f32(op, src, dst);
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
        return unary_simd::dispatch_unary_complex_f64(op, src, dst);
    }
    false
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;

    /// Slices below the element-wise threshold must be rejected by unary
    /// dispatch, leaving `dst` untouched.
    #[test]
    fn test_unary_below_threshold_rejects() {
        let src: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let mut dst = vec![99.0f32; src.len()];

        // len=32 < threshold 64 — must reject
        assert!(!dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst));
        // dst should remain unchanged on rejection
        for &v in &dst {
            assert_eq!(v, 99.0_f32, "dst must be untouched on SIMD rejection");
        }
    }

    /// Empty slices must be rejected by unary dispatch.
    #[test]
    fn test_unary_empty_array() {
        let src: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        assert!(!dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst));
    }

    /// Single-element slices are below threshold and must be rejected;
    /// `dst` must remain untouched.
    #[test]
    fn test_unary_single_element() {
        let src = [2.0_f32];
        let mut dst = [99.0_f32];
        assert!(!dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst));
        assert_eq!(dst, [99.0]);
    }
}
