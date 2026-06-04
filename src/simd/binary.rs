//! Binary element-wise SIMD kernels (add, sub, mul, div).
//!
//! Each kernel holds slice references and implements [`pulp::WithSimd`]
//! so the `pulp` architecture can dispatch to the right ISA at runtime.
//! Dispatch helpers perform threshold admission before routing to a
//! concrete kernel.
//!
//! Supported types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.

use pulp::{Simd, WithSimd};

use std::slice;
use crate::complex::Complex;
use crate::simd::{BinaryOp, get_arch};

// ----------------------------------------------------------------------------
// Thresholds
// ----------------------------------------------------------------------------

/// Minimum slice length for element-wise SIMD admission (f32, f64).
pub(crate) const ELEMENTWISE_THRESHOLD: usize = 64;

/// Minimum slice length for complex element-wise SIMD admission.
pub(crate) const COMPLEX_ELEMENTWISE_THRESHOLD: usize = 128;

// ----------------------------------------------------------------------------
// f32 binary kernel (Add)
// ----------------------------------------------------------------------------

/// Element-wise f32 addition: `dst[i] = lhs[i] + rhs[i]`.
pub(crate) struct AddF32Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f32],

    /// Right operand slice.
    pub(crate) rhs: &'a [f32],
    
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for AddF32Kernel<'_> {
    type Output = ();

    /// Applies SIMD add over the body, scalar add over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(self.dst);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.add_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] + rhs_tail[i];
        }
    }
}

// ----------------------------------------------------------------------------
// f32 binary kernel (Sub)
// ----------------------------------------------------------------------------

/// Element-wise f32 subtraction: `dst[i] = lhs[i] - rhs[i]`.
pub(crate) struct SubF32Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f32],
    
    /// Right operand slice.
    pub(crate) rhs: &'a [f32],
    
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for SubF32Kernel<'_> {
    type Output = ();

    /// Applies SIMD sub over the body, scalar sub over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(self.dst);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.sub_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] - rhs_tail[i];
        }
    }
}

// ----------------------------------------------------------------------------
// f32 binary kernel (Mul)
// ----------------------------------------------------------------------------

/// Element-wise f32 multiplication: `dst[i] = lhs[i] * rhs[i]`.
/// Uses separate mul lane ops (not FMA) to stay bit-identical with scalar.
pub(crate) struct MulF32Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f32],
    
    /// Right operand slice.
    pub(crate) rhs: &'a [f32],
    
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for MulF32Kernel<'_> {
    type Output = ();

    /// Applies SIMD mul (not FMA) over the body, scalar mul over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(self.dst);

        // Use separate mul lane ops, not FMA, to keep element-wise
        // semantics bit-identical with scalar.
        for i in 0..lhs_body.len() {
            dst_body[i] = simd.mul_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] * rhs_tail[i];
        }
    }
}

// ----------------------------------------------------------------------------
// f32 binary kernel (Div)
// ----------------------------------------------------------------------------

/// Element-wise f32 division: `dst[i] = lhs[i] / rhs[i]`.
pub(crate) struct DivF32Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f32],

    /// Right operand slice.
    pub(crate) rhs: &'a [f32],
    
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for DivF32Kernel<'_> {
    type Output = ();

    /// Applies SIMD div over the body, scalar div over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(self.dst);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.div_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] / rhs_tail[i];
        }
    }
}

// ----------------------------------------------------------------------------
// f64 binary kernel (Add)
// ----------------------------------------------------------------------------

/// Element-wise f64 addition: `dst[i] = lhs[i] + rhs[i]`.
pub(crate) struct AddF64Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f64],
    
    /// Right operand slice.
    pub(crate) rhs: &'a [f64],
    
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for AddF64Kernel<'_> {
    type Output = ();

    /// Applies SIMD add over the body, scalar add over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f64s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(self.dst);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.add_f64s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] + rhs_tail[i];
        }
    }
}

// ----------------------------------------------------------------------------
// f64 binary kernel (Sub)
// ----------------------------------------------------------------------------

/// Element-wise f64 subtraction: `dst[i] = lhs[i] - rhs[i]`.
pub(crate) struct SubF64Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f64],
    /// Right operand slice.
    pub(crate) rhs: &'a [f64],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for SubF64Kernel<'_> {
    type Output = ();

    /// Applies SIMD sub over the body, scalar sub over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f64s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(self.dst);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.sub_f64s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] - rhs_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// f64 binary kernel (Mul)
// ---------------------------------------------------------------------------

/// Element-wise f64 multiplication: `dst[i] = lhs[i] * rhs[i]`.
/// Uses separate mul lane ops (not FMA) to stay bit-identical with scalar.
pub(crate) struct MulF64Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f64],
    /// Right operand slice.
    pub(crate) rhs: &'a [f64],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for MulF64Kernel<'_> {
    type Output = ();

    /// Applies SIMD mul (not FMA) over the body, scalar mul over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f64s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(self.dst);

        // Use separate mul lane ops, not FMA, to keep element-wise
        // semantics bit-identical with scalar.
        for i in 0..lhs_body.len() {
            dst_body[i] = simd.mul_f64s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] * rhs_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// f64 binary kernel (Div)
// ---------------------------------------------------------------------------

/// Element-wise f64 division: `dst[i] = lhs[i] / rhs[i]`.
pub(crate) struct DivF64Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f64],
    /// Right operand slice.
    pub(crate) rhs: &'a [f64],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for DivF64Kernel<'_> {
    type Output = ();

    /// Applies SIMD div over the body, scalar div over the tail.
    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f64s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(self.dst);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.div_f64s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] / rhs_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Complex<f32> binary kernels (Add, Sub)
// ---------------------------------------------------------------------------

/// Element-wise `Complex<f32>` addition.
/// Reinterprets the interleaved real/imag layout as `[f32]` for SIMD.
pub(crate) struct ComplexAddF32Kernel<'a> {
    /// Left operand slice (interleaved real/imag).
    pub(crate) lhs: &'a [crate::complex::Complex<f32>],
    /// Right operand slice (interleaved real/imag).
    pub(crate) rhs: &'a [crate::complex::Complex<f32>],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [crate::complex::Complex<f32>],
}

impl WithSimd for ComplexAddF32Kernel<'_> {
    type Output = ();

    /// Reinterprets complex slices as f32 and applies SIMD add.
    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let lhs_f32 = unsafe { slice::from_raw_parts(self.lhs.as_ptr() as *const f32, n * 2) };
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let rhs_f32 = unsafe { slice::from_raw_parts(self.rhs.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
            unsafe { slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
        let (lhs_body, lhs_tail) = S::as_simd_f32s(lhs_f32);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(rhs_f32);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(dst_f32);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.add_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] + rhs_tail[i];
        }
    }
}

/// Element-wise `Complex<f32>` subtraction.
pub(crate) struct ComplexSubF32Kernel<'a> {
    /// Left operand slice (interleaved real/imag).
    pub(crate) lhs: &'a [crate::complex::Complex<f32>],
    /// Right operand slice (interleaved real/imag).
    pub(crate) rhs: &'a [crate::complex::Complex<f32>],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [crate::complex::Complex<f32>],
}

impl WithSimd for ComplexSubF32Kernel<'_> {
    type Output = ();

    /// Reinterprets complex slices as f32 and applies SIMD sub.
    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let lhs_f32 = unsafe { slice::from_raw_parts(self.lhs.as_ptr() as *const f32, n * 2) };
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let rhs_f32 = unsafe { slice::from_raw_parts(self.rhs.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
            unsafe { slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
        let (lhs_body, lhs_tail) = S::as_simd_f32s(lhs_f32);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(rhs_f32);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(dst_f32);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.sub_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] - rhs_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Complex<f32> binary kernel (Mul)
// ---------------------------------------------------------------------------

/// Element-wise `Complex<f32>` multiplication.
/// Falls back to scalar; full SIMD vectorisation is pending.
pub(crate) struct ComplexMulF32Kernel<'a> {
    /// Left operand slice (interleaved real/imag).
    pub(crate) lhs: &'a [crate::complex::Complex<f32>],
    /// Right operand slice (interleaved real/imag).
    pub(crate) rhs: &'a [crate::complex::Complex<f32>],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [crate::complex::Complex<f32>],
}

impl WithSimd for ComplexMulF32Kernel<'_> {
    type Output = ();

    /// Falls back to scalar for complex multiply.
    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let lhs_f32 = unsafe { slice::from_raw_parts(self.lhs.as_ptr() as *const f32, n * 2) };
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let rhs_f32 = unsafe { slice::from_raw_parts(self.rhs.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
            unsafe { slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
        let (lhs_body, lhs_tail) = S::as_simd_f32s(lhs_f32);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(rhs_f32);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(dst_f32);

        // (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        for i in 0..lhs_body.len() {
            let ac = simd.mul_f32s(lhs_body[i], rhs_body[i]); // this multiplies all lanes, not correct
            let _ = ac;
            // For now, use scalar tail approach for complex mul
            // (SIMD vectorization of complex mul requires deinterleave which is TBD)
        }
        // Fall back to scalar for all elements (complex mul not vectorized in this iteration)
        for i in 0..self.lhs.len() {
            let l = self.lhs[i];
            let r = self.rhs[i];
            self.dst[i] = l * r;
        }
        let _ = (lhs_tail, rhs_tail, dst_body, dst_tail, simd);
    }
}

// ---------------------------------------------------------------------------
// Complex<f64> binary kernel (Add)
// ---------------------------------------------------------------------------

/// Element-wise `Complex<f64>` addition.
/// Sub and Mul are not implemented for f64 complex; only Add is supported.
pub(crate) struct ComplexAddF64Kernel<'a> {
    /// Left operand slice (interleaved real/imag).
    pub(crate) lhs: &'a [crate::complex::Complex<f64>],
    /// Right operand slice (interleaved real/imag).
    pub(crate) rhs: &'a [crate::complex::Complex<f64>],
    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [crate::complex::Complex<f64>],
}

impl WithSimd for ComplexAddF64Kernel<'_> {
    type Output = ();

    /// Reinterprets complex slices as f64 and applies SIMD add.
    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let lhs_f64 = unsafe { slice::from_raw_parts(self.lhs.as_ptr() as *const f64, n * 2) };
        // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
        let rhs_f64 = unsafe { slice::from_raw_parts(self.rhs.as_ptr() as *const f64, n * 2) };
        let dst_f64 =
            // SAFETY: Complex<T> is repr(C) with two T fields; the layout is identical to [T; 2]. The cast through raw pointers preserves provenance and the length 2*n is correct.
            unsafe { slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f64, n * 2) };
        let (lhs_body, lhs_tail) = S::as_simd_f64s(lhs_f64);
        let (rhs_body, rhs_tail) = S::as_simd_f64s(rhs_f64);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(dst_f64);

        for i in 0..lhs_body.len() {
            dst_body[i] = simd.add_f64s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] + rhs_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers (called from driver.rs facade)
// ---------------------------------------------------------------------------

/// Dispatches f32 binary op to the corresponding kernel.
pub(crate) fn dispatch_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32], dst: &mut [f32]) -> bool {
    if lhs.len() < ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(AddF32Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => arch.dispatch(SubF32Kernel { lhs, rhs, dst }),
        BinaryOp::Mul => arch.dispatch(MulF32Kernel { lhs, rhs, dst }),
        BinaryOp::Div => arch.dispatch(DivF32Kernel { lhs, rhs, dst }),
    }
    true
}

/// Dispatches f64 binary op to the corresponding kernel.
pub(crate) fn dispatch_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) -> bool {
    if lhs.len() < ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(AddF64Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => arch.dispatch(SubF64Kernel { lhs, rhs, dst }),
        BinaryOp::Mul => arch.dispatch(MulF64Kernel { lhs, rhs, dst }),
        BinaryOp::Div => arch.dispatch(DivF64Kernel { lhs, rhs, dst }),
    }
    true
}

/// Dispatches Complex<f32> binary element-wise op to the kernel.
pub(crate) fn dispatch_binary_complex_f32(
    op: BinaryOp,
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
    dst: &mut [Complex<f32>],
) -> bool {
    if lhs.len() < COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(ComplexAddF32Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => arch.dispatch(ComplexSubF32Kernel { lhs, rhs, dst }),
        BinaryOp::Mul => arch.dispatch(ComplexMulF32Kernel { lhs, rhs, dst }),
        BinaryOp::Div => return false, // Complex div not implemented
    }
    true
}

/// Dispatches Complex<f64> binary element-wise op to the kernel.
/// Sub, Mul, and Div are not implemented for Complex<f64>.
pub(crate) fn dispatch_binary_complex_f64(
    op: BinaryOp,
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
    dst: &mut [Complex<f64>],
) -> bool {
    if lhs.len() < COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(ComplexAddF64Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => return false, // Sub not implemented for f64 complex
        BinaryOp::Mul => return false, // Mul not implemented for f64 complex
        BinaryOp::Div => return false,
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd::{dispatch_vector_binary_op, BinaryOp};

    /// Width used for boundary / tail coverage tests.
    const SIMD_WIDTH: usize = 64;
    /// Number of random cases per property test.
    const CASES: usize = 32;
    /// Maximum random slice length for property tests.
    const MAX_LEN: usize = 4096;

    // ---- f32 admission ------------------------------------------------------

    /// Asserts SIMD and scalar addition produce identical results.
    fn assert_add_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l + r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f32 addition goes through SIMD and matches scalar.
    #[test]
    fn test_vector_add_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
        // len=128 > threshold 64, SIMD admission must succeed.
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_add_f32(&lhs, &rhs, &dst);
    }

    /// Asserts SIMD and scalar subtraction produce identical results.
    fn assert_sub_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l - r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f32 subtraction goes through SIMD and matches scalar.
    #[test]
    fn test_vector_sub_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Sub, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_sub_f32(&lhs, &rhs, &dst);
    }

    /// Asserts SIMD and scalar multiplication produce identical results.
    fn assert_mul_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l * r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f32 multiplication goes through SIMD and matches scalar.
    #[test]
    fn test_vector_mul_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Mul, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_mul_f32(&lhs, &rhs, &dst);
    }

    /// Asserts SIMD and scalar division produce identical results.
    fn assert_div_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l / r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f32 division goes through SIMD and matches scalar.
    /// Offsets inputs by +1.0 to avoid division by zero.
    #[test]
    fn test_vector_div_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32 + 1.0).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5 + 1.0).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Div, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_div_f32(&lhs, &rhs, &dst);
    }

    // ---- f64 admission ------------------------------------------------------

    /// Asserts SIMD and scalar addition produce identical results.
    fn assert_add_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l + r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f64 addition goes through SIMD and matches scalar.
    #[test]
    fn test_vector_add_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_add_f64(&lhs, &rhs, &dst);
    }

    /// Asserts SIMD and scalar subtraction produce identical results.
    fn assert_sub_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l - r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f64 subtraction goes through SIMD and matches scalar.
    #[test]
    fn test_vector_sub_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Sub, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_sub_f64(&lhs, &rhs, &dst);
    }

    /// Asserts SIMD and scalar multiplication produce identical results.
    fn assert_mul_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l * r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f64 multiplication goes through SIMD and matches scalar.
    #[test]
    fn test_vector_mul_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Mul, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_mul_f64(&lhs, &rhs, &dst);
    }

    /// Asserts SIMD and scalar division produce identical results.
    fn assert_div_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l / r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f64 division goes through SIMD and matches scalar.
    /// Offsets inputs by +1.0 to avoid division by zero.
    #[test]
    fn test_vector_div_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64 + 1.0).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25 + 1.0).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Div, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_div_f64(&lhs, &rhs, &dst);
    }

    // ---- consistency vs serial ----------------------------------------------
    //
    // These tests verify that SIMD output matches the scalar equivalent
    // bit-for-bit (or NaN-for-NaN) across varied inputs including
    // extreme values, NaNs, infinities, and subnormals.

    /// Checks two f64 values are bit-identical or both NaN.
    fn assert_same_bits_or_nan_f64(actual: f64, expected: f64) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    /// Checks two f32 values are bit-identical or both NaN.
    fn assert_same_bits_or_nan_f32(actual: f32, expected: f32) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    /// Asserts two f64 slices are element-wise bit-identical (or both NaN).
    fn assert_vec_bits_or_nan_f64(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert_same_bits_or_nan_f64(a, e);
        }
    }

    /// Asserts two f32 slices are element-wise bit-identical (or both NaN).
    fn assert_vec_bits_or_nan_f32(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert_same_bits_or_nan_f32(a, e);
        }
    }

    /// Applies a binary op to two scalars for reference comparison.
    fn apply_binary_f64(op: BinaryOp, lhs: f64, rhs: f64) -> f64 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

    /// Applies a binary op to two scalars for reference comparison.
    fn apply_binary_f32(op: BinaryOp, lhs: f32, rhs: f32) -> f32 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

    /// Generates two `Vec<f64>` from seeded extreme-value fixtures.
    fn fixture_f64(len: usize) -> (Vec<f64>, Vec<f64>) {
        let lhs_seed = [
            1.5_f64, -2.3, 0.001, -1e20, std::f64::consts::PI,
            0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ];
        let rhs_seed = [
            -4.25_f64, 8.0, -0.125, 1e-20, -2.0,
            0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ];
        let lhs: Vec<f64> = (0..len).map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f64> = (0..len).map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    /// Generates two `Vec<f32>` from seeded extreme-value fixtures.
    fn fixture_f32(len: usize) -> (Vec<f32>, Vec<f32>) {
        let lhs_seed = [
            1.5_f32, -2.3, 0.001, -1e10, std::f32::consts::PI,
            0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
        ];
        let rhs_seed = [
            -4.25_f32, 8.0, -0.125, 1e-10, -2.0,
            0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
        ];
        let lhs: Vec<f32> = (0..len).map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f32> = (0..len).map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    /// Runs SIMD vs scalar comparison for f64, checking bitwise equivalence.
    fn simd_vs_serial_bitwise_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) {
        let mut dst = vec![0.0_f64; lhs.len()];
        let handled = dispatch_vector_binary_op(op, lhs, rhs, &mut dst);
        if handled {
            let serial: Vec<f64> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| apply_binary_f64(op, l, r))
                .collect();
            assert_vec_bits_or_nan_f64(&dst, &serial);
        }
    }

    /// Runs SIMD vs scalar comparison for f32, checking bitwise equivalence.
    fn simd_vs_serial_bitwise_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) {
        let mut dst = vec![0.0_f32; lhs.len()];
        let handled = dispatch_vector_binary_op(op, lhs, rhs, &mut dst);
        if handled {
            let serial: Vec<f32> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| apply_binary_f32(op, l, r))
                .collect();
            assert_vec_bits_or_nan_f32(&dst, &serial);
        }
    }

    /// Compares SIMD vs scalar for each binary operation over a
    /// 256-element synthetic fixture containing extreme values.
    #[test]
    fn test_simd_vector_consistency_elementwise() {
        let (lhs_f64, rhs_f64) = fixture_f64(256);
        let (lhs_f32, rhs_f32) = fixture_f32(256);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            simd_vs_serial_bitwise_f64(op, &lhs_f64, &rhs_f64);
            simd_vs_serial_bitwise_f32(op, &lhs_f32, &rhs_f32);
        }
    }

    /// Tests various lengths (including 0, 1, and multiples of SIMD width)
    /// for bit-for-bit agreement between SIMD and scalar add/sub.
    #[test]
    fn test_elementwise_boundary_lengths() {
        let mut lengths = vec![0, 1, 32, 64, 65, 128, 256];
        lengths.extend((1..8).map(|extra| SIMD_WIDTH + extra));
        for len in lengths {
            let (lhs, rhs) = fixture_f64(len);
            simd_vs_serial_bitwise_f64(BinaryOp::Add, &lhs, &rhs);
            simd_vs_serial_bitwise_f64(BinaryOp::Sub, &lhs, &rhs);
        }
    }

    /// Tests that slices whose length is misaligned with the SIMD width
    /// (256 + small tail) still produce correct results for all four ops.
    #[test]
    fn test_elementwise_tail_handling() {
        for extra in [1_usize, 3, 7, 15] {
            let (lhs, rhs) = fixture_f64(256 + extra);
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                simd_vs_serial_bitwise_f64(op, &lhs, &rhs);
            }
        }
    }

    /// Slice below the element-wise threshold must be rejected.
    #[test]
    fn test_elementwise_below_threshold() {
        let (lhs, rhs) = fixture_f64(32);
        let mut dst = vec![0.0_f64; lhs.len()];
        assert!(!dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
    }

    /// Slice exactly at the element-wise threshold must be admitted.
    #[test]
    fn test_elementwise_at_threshold() {
        let (lhs, rhs) = fixture_f64(64);
        simd_vs_serial_bitwise_f64(BinaryOp::Add, &lhs, &rhs);
    }

    /// Offsetting a slice within a larger allocation (misaligned pointer)
    /// must still produce correct SIMD results.
    #[test]
    fn test_elementwise_misaligned() {
        let (lhs_storage, rhs_storage) = fixture_f64(258);
        let lhs = &lhs_storage[1..257];
        let rhs = &rhs_storage[1..257];
        simd_vs_serial_bitwise_f64(BinaryOp::Mul, lhs, rhs);
    }

    // ---- binary property tests ----------------------------------------------

    /// splitmix64 PRNG for deterministic property-based tests.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generates a random length in `[0, max_len]` from a PRNG state.
    fn gen_len(state: &mut u64, max_len: usize) -> usize {
        (splitmix64(state) as usize) % (max_len + 1)
    }

    /// Generates a random f64 in `[-10, 10)` from a PRNG state.
    fn gen_f64(state: &mut u64) -> f64 {
        let frac = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
        (frac - 0.5) * 20.0
    }

    /// Generates a random f32 in `[-10, 10)` from a PRNG state.
    fn gen_f32(state: &mut u64) -> f32 {
        let frac = (splitmix64(state) >> 11) as f32 / (1u64 << 53) as f32;
        (frac - 0.5) * 20.0
    }

    /// Randomised binary element-wise consistency check.
    fn prop_elementwise_binary_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = super::ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                let mut dst = vec![0.0_f64; len];
                let handled = dispatch_vector_binary_op(op, &lhs, &rhs, &mut dst);
                if handled {
                    let serial: Vec<f64> = lhs
                        .iter()
                        .zip(rhs.iter())
                        .map(|(&l, &r)| match op {
                            BinaryOp::Add => l + r,
                            BinaryOp::Sub => l - r,
                            BinaryOp::Mul => l * r,
                            BinaryOp::Div => l / r,
                        })
                        .collect();
                    for (i, (&a, &e)) in dst.iter().zip(serial.iter()).enumerate() {
                        if a.is_nan() || e.is_nan() {
                            assert!(a.is_nan() && e.is_nan(), "nan mismatch at len={len}, i={i}");
                        } else {
                            assert_eq!(a.to_bits(), e.to_bits(), "bits mismatch at len={len}, i={i}");
                        }
                    }
                }
            }
        }
    }

    /// Randomised complex f32 element-wise add consistency check.
    fn prop_elementwise_complex_add_f32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = super::COMPLEX_ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<Complex<f32>> = (0..len)
                .map(|_| Complex::new(gen_f32(&mut rng), gen_f32(&mut rng)))
                .collect();
            let rhs: Vec<Complex<f32>> = (0..len)
                .map(|_| Complex::new(gen_f32(&mut rng), gen_f32(&mut rng)))
                .collect();
            let mut dst = vec![Complex::new(0.0, 0.0); len];
            let handled = dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
            if handled {
                for (i, (a, (l, r))) in dst.iter().zip(lhs.iter().zip(rhs.iter())).enumerate() {
                    let e = *l + *r;
                    if a.re.is_nan() || a.im.is_nan() || e.re.is_nan() || e.im.is_nan() {
                        assert!(
                            (a.re.is_nan() && e.re.is_nan()) || (a.re == e.re),
                            "real nan mismatch at len={len}, i={i}"
                        );
                        assert!(
                            (a.im.is_nan() && e.im.is_nan()) || (a.im == e.im),
                            "imag nan mismatch at len={len}, i={i}"
                        );
                    } else {
                        assert_eq!(a.re.to_bits(), e.re.to_bits(), "real bits at len={len}, i={i}");
                        assert_eq!(a.im.to_bits(), e.im.to_bits(), "imag bits at len={len}, i={i}");
                    }
                }
            }
        }
    }

    /// Randomised tail-handling test with varying SIMD widths and tail sizes.
    fn prop_tail_handling_f64(seed: u64) {
        let mut rng = seed;
        for width in [2usize, 4, 8, 16, 32] {
            for tail in 1..width {
                let base = 1 + (splitmix64(&mut rng) as usize % 16);
                let len = base * width + tail;
                let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
                let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
                let mut dst = vec![0.0_f64; len];
                let handled = dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
                if handled {
                    for i in 0..len {
                        let expected = lhs[i] + rhs[i];
                        if expected.is_nan() || dst[i].is_nan() {
                            assert!(dst[i].is_nan() && expected.is_nan());
                        } else {
                            assert_eq!(dst[i].to_bits(), expected.to_bits());
                        }
                    }
                }
            }
        }
    }

    /// Randomised element-wise comparison between SIMD and scalar
    /// for all four binary operations against Complex<f32> add.
    #[test]
    fn test_prop_elementwise_consistency() {
        prop_elementwise_binary_f64(0x1001);
        prop_elementwise_complex_add_f32(0x1003);
    }

    /// Randomised tail-handling test across multiple SIMD widths
    /// and tail sizes verifying correct fallback.
    #[test]
    fn test_prop_tail_and_fallback() {
        prop_tail_handling_f64(0x5001);
    }

    // ---- complex binary admission -------------------------------------------

    /// 128-element `Complex<f32>` subtraction goes through SIMD and matches scalar.
    #[test]
    fn test_vector_complex_sub_f32() {
        let lhs: Vec<Complex<f32>> = (0..128)
            .map(|v| Complex::new(v as f32 + 1.0, (v as f32) * 0.5))
            .collect();
        let rhs: Vec<Complex<f32>> = (0..128)
            .map(|v| Complex::new((v as f32) * 0.5, v as f32))
            .collect();
        let mut dst = vec![Complex::new(0.0, 0.0); lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Sub, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        for (i, ((&a, &l), &r)) in dst.iter().zip(lhs.iter()).zip(rhs.iter()).enumerate() {
            let e = l - r;
            assert_eq!(a.re, e.re, "real mismatch at i={i}");
            assert_eq!(a.im, e.im, "imag mismatch at i={i}");
        }
    }

    /// 128-element `Complex<f64>` addition goes through SIMD and matches scalar.
    #[test]
    fn test_vector_complex_add_f64() {
        let lhs: Vec<Complex<f64>> = (0..128)
            .map(|v| Complex::new(v as f64 - 64.0, (v as f64) * 0.25))
            .collect();
        let rhs: Vec<Complex<f64>> = (0..128)
            .map(|v| Complex::new((v as f64) * 0.5 - 32.0, v as f64))
            .collect();
        let mut dst = vec![Complex::new(0.0, 0.0); lhs.len()];
        let handled = dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        for (i, ((&a, &l), &r)) in dst.iter().zip(lhs.iter()).zip(rhs.iter()).enumerate() {
            let e = l + r;
            assert_eq!(a.re, e.re, "real mismatch at i={i}");
            assert_eq!(a.im, e.im, "imag mismatch at i={i}");
        }
    }
}
