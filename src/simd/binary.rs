//! f32/f64 binary element-wise SIMD kernels.

use pulp::{Simd, WithSimd};

// ---------------------------------------------------------------------------
// f32 binary kernel (Add)
// ---------------------------------------------------------------------------

pub(crate) struct AddF32Kernel<'a> {
    pub(crate) lhs: &'a [f32],
    pub(crate) rhs: &'a [f32],
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for AddF32Kernel<'_> {
    type Output = ();

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

// ---------------------------------------------------------------------------
// f32 binary kernel (Sub)
// ---------------------------------------------------------------------------

pub(crate) struct SubF32Kernel<'a> {
    pub(crate) lhs: &'a [f32],
    pub(crate) rhs: &'a [f32],
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for SubF32Kernel<'_> {
    type Output = ();

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

// ---------------------------------------------------------------------------
// f32 binary kernel (Mul)
// ---------------------------------------------------------------------------

pub(crate) struct MulF32Kernel<'a> {
    pub(crate) lhs: &'a [f32],
    pub(crate) rhs: &'a [f32],
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for MulF32Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(self.dst);

        // FMA is forbidden in element-wise main loop (08-simd §6.6).
        // Use separate mul lane ops, not mul_add.
        for i in 0..lhs_body.len() {
            dst_body[i] = simd.mul_f32s(lhs_body[i], rhs_body[i]);
        }
        for i in 0..lhs_tail.len() {
            dst_tail[i] = lhs_tail[i] * rhs_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// f32 binary kernel (Div)
// ---------------------------------------------------------------------------

pub(crate) struct DivF32Kernel<'a> {
    pub(crate) lhs: &'a [f32],
    pub(crate) rhs: &'a [f32],
    pub(crate) dst: &'a mut [f32],
}

impl WithSimd for DivF32Kernel<'_> {
    type Output = ();

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

// ---------------------------------------------------------------------------
// f64 binary kernel (Add)
// ---------------------------------------------------------------------------

pub(crate) struct AddF64Kernel<'a> {
    pub(crate) lhs: &'a [f64],
    pub(crate) rhs: &'a [f64],
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for AddF64Kernel<'_> {
    type Output = ();

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

// ---------------------------------------------------------------------------
// f64 binary kernel (Sub)
// ---------------------------------------------------------------------------

pub(crate) struct SubF64Kernel<'a> {
    pub(crate) lhs: &'a [f64],
    pub(crate) rhs: &'a [f64],
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for SubF64Kernel<'_> {
    type Output = ();

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

pub(crate) struct MulF64Kernel<'a> {
    pub(crate) lhs: &'a [f64],
    pub(crate) rhs: &'a [f64],
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for MulF64Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, rhs_tail) = S::as_simd_f64s(self.rhs);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(self.dst);

        // FMA is forbidden in element-wise main loop (08-simd §6.6).
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

pub(crate) struct DivF64Kernel<'a> {
    pub(crate) lhs: &'a [f64],
    pub(crate) rhs: &'a [f64],
    pub(crate) dst: &'a mut [f64],
}

impl WithSimd for DivF64Kernel<'_> {
    type Output = ();

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

pub(crate) struct ComplexAddF32Kernel<'a> {
    pub(crate) lhs: &'a [crate::complex::Complex<f32>],
    pub(crate) rhs: &'a [crate::complex::Complex<f32>],
    pub(crate) dst: &'a mut [crate::complex::Complex<f32>],
}

impl WithSimd for ComplexAddF32Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        let lhs_f32 = unsafe { std::slice::from_raw_parts(self.lhs.as_ptr() as *const f32, n * 2) };
        let rhs_f32 = unsafe { std::slice::from_raw_parts(self.rhs.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
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

pub(crate) struct ComplexSubF32Kernel<'a> {
    pub(crate) lhs: &'a [crate::complex::Complex<f32>],
    pub(crate) rhs: &'a [crate::complex::Complex<f32>],
    pub(crate) dst: &'a mut [crate::complex::Complex<f32>],
}

impl WithSimd for ComplexSubF32Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        let lhs_f32 = unsafe { std::slice::from_raw_parts(self.lhs.as_ptr() as *const f32, n * 2) };
        let rhs_f32 = unsafe { std::slice::from_raw_parts(self.rhs.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
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

pub(crate) struct ComplexMulF32Kernel<'a> {
    pub(crate) lhs: &'a [crate::complex::Complex<f32>],
    pub(crate) rhs: &'a [crate::complex::Complex<f32>],
    pub(crate) dst: &'a mut [crate::complex::Complex<f32>],
}

impl WithSimd for ComplexMulF32Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.lhs.len();
        let lhs_f32 = unsafe { std::slice::from_raw_parts(self.lhs.as_ptr() as *const f32, n * 2) };
        let rhs_f32 = unsafe { std::slice::from_raw_parts(self.rhs.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
        let (lhs_body, lhs_tail) = S::as_simd_f32s(lhs_f32);
        let (rhs_body, rhs_tail) = S::as_simd_f32s(rhs_f32);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(dst_f32);

        // (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        // FMA forbidden in element-wise main loop (08-simd §6.6).
        for i in 0..lhs_body.len() {
            // Re = a*c - b*d
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::simd::BinaryOp;

    fn assert_add_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l + r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_add_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
        // len=128 > threshold 64, SIMD admission must succeed.
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_add_f32(&lhs, &rhs, &dst);
    }

    fn assert_sub_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l - r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_sub_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Sub, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_sub_f32(&lhs, &rhs, &dst);
    }

    fn assert_mul_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l * r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_mul_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Mul, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_mul_f32(&lhs, &rhs, &dst);
    }

    fn assert_div_f32(lhs: &[f32], rhs: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = lhs.iter().zip(rhs).map(|(&l, &r)| l / r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_div_f32() {
        let lhs: Vec<f32> = (0..128).map(|v| v as f32 + 1.0).collect();
        let rhs: Vec<f32> = (0..128).map(|v| (v as f32) * 0.5 + 1.0).collect();
        let mut dst = vec![0.0f32; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Div, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_div_f32(&lhs, &rhs, &dst);
    }

    // ---- f64 ----

    fn assert_add_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l + r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_add_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_add_f64(&lhs, &rhs, &dst);
    }

    fn assert_sub_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l - r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_sub_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Sub, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_sub_f64(&lhs, &rhs, &dst);
    }

    fn assert_mul_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l * r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_mul_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Mul, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_mul_f64(&lhs, &rhs, &dst);
    }

    fn assert_div_f64(lhs: &[f64], rhs: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = lhs.iter().zip(rhs).map(|(&l, &r)| l / r).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_div_f64() {
        let lhs: Vec<f64> = (0..128).map(|v| v as f64 + 1.0).collect();
        let rhs: Vec<f64> = (0..128).map(|v| (v as f64) * 0.25 + 1.0).collect();
        let mut dst = vec![0.0f64; lhs.len()];
        let handled = crate::simd::dispatch_vector_binary_op(BinaryOp::Div, &lhs, &rhs, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_div_f64(&lhs, &rhs, &dst);
    }
}
