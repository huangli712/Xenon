//! f32/f64 dot-product SIMD reduction kernels.

use pulp::{Simd, WithSimd};

use crate::complex::Complex;

// ---------------------------------------------------------------------------
// Dispatch helpers (called from mod.rs facade)
// ---------------------------------------------------------------------------

/// Dot threshold for f32/f64 (08-simd §5.8 L456).
const DOT_THRESHOLD: usize = 512;

pub(crate) fn try_dot_f32_impl(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < DOT_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(DotF32Kernel { lhs, rhs }))
}

pub(crate) fn try_dot_f64_impl(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < DOT_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(DotF64Kernel { lhs, rhs }))
}

/// Dot threshold for Complex (PLAN.md W14, derived from f32/f64 dot=512).
const COMPLEX_DOT_THRESHOLD: usize = 512;

pub(crate) fn try_dot_complex_f32_impl(
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
) -> Option<Complex<f32>> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < COMPLEX_DOT_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(ComplexDotF32Kernel { lhs, rhs }))
}

pub(crate) fn try_dot_complex_f64_impl(
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
) -> Option<Complex<f64>> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < COMPLEX_DOT_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(ComplexDotF64Kernel { lhs, rhs }))
}

// ---------------------------------------------------------------------------
// f32 dot kernel
// ---------------------------------------------------------------------------

pub(crate) struct DotF32Kernel<'a> {
    pub(crate) lhs: &'a [f32],
    pub(crate) rhs: &'a [f32],
}

impl WithSimd for DotF32Kernel<'_> {
    type Output = f32;

    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, _rhs_tail) = S::as_simd_f32s(self.rhs);

        // FMA forbidden in per-element multiply+accumulate (08-simd §6.6).
        let mut acc = simd.splat_f32s(0.0);
        for i in 0..lhs_body.len() {
            let prod = simd.mul_f32s(lhs_body[i], rhs_body[i]);
            acc = simd.add_f32s(acc, prod);
        }

        // Horizontal reduction merge (FMA allowed, tolerance documented).
        let mut scalar = simd.reduce_sum_f32s(acc);

        // Tail
        for i in 0..lhs_tail.len() {
            scalar += lhs_tail[i] * self.rhs[self.rhs.len() - lhs_tail.len() + i];
        }

        scalar
    }
}

// ---------------------------------------------------------------------------
// f64 dot kernel
// ---------------------------------------------------------------------------

pub(crate) struct DotF64Kernel<'a> {
    pub(crate) lhs: &'a [f64],
    pub(crate) rhs: &'a [f64],
}

impl WithSimd for DotF64Kernel<'_> {
    type Output = f64;

    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, _rhs_tail) = S::as_simd_f64s(self.rhs);

        let mut acc = simd.splat_f64s(0.0);
        for i in 0..lhs_body.len() {
            let prod = simd.mul_f64s(lhs_body[i], rhs_body[i]);
            acc = simd.add_f64s(acc, prod);
        }

        let mut scalar = simd.reduce_sum_f64s(acc);

        let tail_offset = self.rhs.len() - lhs_tail.len();
        for (i, &l) in lhs_tail.iter().enumerate() {
            scalar += l * self.rhs[tail_offset + i];
        }

        scalar
    }
}

// ---------------------------------------------------------------------------
// Complex<f32> dot kernel
// ---------------------------------------------------------------------------

pub(crate) struct ComplexDotF32Kernel<'a> {
    pub(crate) lhs: &'a [Complex<f32>],
    pub(crate) rhs: &'a [Complex<f32>],
}

impl WithSimd for ComplexDotF32Kernel<'_> {
    type Output = Complex<f32>;

    fn with_simd<S: Simd>(self, simd: S) -> Complex<f32> {
        // BLAS xdotc contract: dot = sum(conj(lhs_i) * rhs_i)
        // conj(lhs) * rhs = (re_l * re_r + im_l * im_r) + (re_l * im_r - im_l * re_r)i
        let mut re_acc = 0.0f32;
        let mut im_acc = 0.0f32;
        for i in 0..self.lhs.len() {
            let l = self.lhs[i];
            let r = self.rhs[i];
            // conj(l) * r
            re_acc += l.re * r.re + l.im * r.im;
            im_acc += l.re * r.im - l.im * r.re;
        }
        let _ = simd;
        Complex::new(re_acc, im_acc)
    }
}

// ---------------------------------------------------------------------------
// Complex<f64> dot kernel
// ---------------------------------------------------------------------------

pub(crate) struct ComplexDotF64Kernel<'a> {
    pub(crate) lhs: &'a [Complex<f64>],
    pub(crate) rhs: &'a [Complex<f64>],
}

impl WithSimd for ComplexDotF64Kernel<'_> {
    type Output = Complex<f64>;

    fn with_simd<S: Simd>(self, simd: S) -> Complex<f64> {
        let mut re_acc = 0.0f64;
        let mut im_acc = 0.0f64;
        for i in 0..self.lhs.len() {
            let l = self.lhs[i];
            let r = self.rhs[i];
            re_acc += l.re * r.re + l.im * r.im;
            im_acc += l.re * r.im - l.im * r.re;
        }
        let _ = simd;
        Complex::new(re_acc, im_acc)
    }
}
