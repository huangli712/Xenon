//! f32/f64 unary element-wise SIMD kernels.

use pulp::{Simd, WithSimd};

use crate::complex::Complex;

// ---------------------------------------------------------------------------
// Neg kernel
// ---------------------------------------------------------------------------

pub(crate) struct NegKernel<'a, T> {
    pub(crate) src: &'a [T],
    pub(crate) dst: &'a mut [T],
    /// Implementation token for monomorphization.
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl WithSimd for NegKernel<'_, f32> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (src_body, src_tail) = S::as_simd_f32s(self.src);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(self.dst);

        for i in 0..src_body.len() {
            dst_body[i] = simd.neg_f32s(src_body[i]);
        }
        for i in 0..src_tail.len() {
            dst_tail[i] = -src_tail[i];
        }
    }
}

impl WithSimd for NegKernel<'_, f64> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let (src_body, src_tail) = S::as_simd_f64s(self.src);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(self.dst);

        for i in 0..src_body.len() {
            dst_body[i] = simd.neg_f64s(src_body[i]);
        }
        for i in 0..src_tail.len() {
            dst_tail[i] = -src_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Complex<f32> Neg kernel
// ---------------------------------------------------------------------------

pub(crate) struct ComplexNegF32Kernel<'a> {
    pub(crate) src: &'a [Complex<f32>],
    pub(crate) dst: &'a mut [Complex<f32>],
}

impl WithSimd for ComplexNegF32Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.src.len();
        let src_f32 = unsafe { std::slice::from_raw_parts(self.src.as_ptr() as *const f32, n * 2) };
        let dst_f32 =
            unsafe { std::slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2) };
        let (src_body, src_tail) = S::as_simd_f32s(src_f32);
        let (dst_body, dst_tail) = S::as_mut_simd_f32s(dst_f32);

        for i in 0..src_body.len() {
            dst_body[i] = simd.neg_f32s(src_body[i]);
        }
        for i in 0..src_tail.len() {
            dst_tail[i] = -src_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Complex<f64> Neg kernel
// ---------------------------------------------------------------------------

pub(crate) struct ComplexNegF64Kernel<'a> {
    pub(crate) src: &'a [Complex<f64>],
    pub(crate) dst: &'a mut [Complex<f64>],
}

impl WithSimd for ComplexNegF64Kernel<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.src.len();
        let src_f64 = unsafe { std::slice::from_raw_parts(self.src.as_ptr() as *const f64, n * 2) };
        let dst_f64 =
            unsafe { std::slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f64, n * 2) };
        let (src_body, src_tail) = S::as_simd_f64s(src_f64);
        let (dst_body, dst_tail) = S::as_mut_simd_f64s(dst_f64);

        for i in 0..src_body.len() {
            dst_body[i] = simd.neg_f64s(src_body[i]);
        }
        for i in 0..src_tail.len() {
            dst_tail[i] = -src_tail[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::simd::UnaryOp;

    fn assert_neg_f32(src: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = src.iter().map(|&v| -v).collect();
        assert_eq!(actual, expected.as_slice());
    }

    fn assert_neg_f64(src: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = src.iter().map(|&v| -v).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn test_vector_neg_f32() {
        let src: Vec<f32> = (0..128).map(|v| v as f32 - 64.0).collect();
        let mut dst = vec![0.0f32; src.len()];
        let handled = crate::simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_neg_f32(&src, &dst);
    }

    #[test]
    fn test_vector_neg_f64() {
        let src: Vec<f64> = (0..128).map(|v| v as f64 - 64.0).collect();
        let mut dst = vec![0.0f64; src.len()];
        let handled = crate::simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_neg_f64(&src, &dst);
    }
}
