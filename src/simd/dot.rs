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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd;

    fn tolerance_f64(data: &[f64]) -> f64 {
        let n = data.len();
        let max_abs_input = data.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
        (4.0 * f64::EPSILON * (n as f64) * max_abs_input).max(4.0 * f64::MIN_POSITIVE)
    }

    fn tolerance_f32(data: &[f32]) -> f32 {
        let n = data.len();
        let max_abs_input = data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        (4.0 * f32::EPSILON * (n as f32) * max_abs_input).max(4.0 * f32::MIN_POSITIVE)
    }

    fn assert_within_tolerance_f64(actual: f64, expected: f64, tol: f64) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else if expected.is_infinite() || actual.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!((actual - expected).abs() <= tol.max(4.0 * f64::MIN_POSITIVE));
        }
    }

    fn assert_within_tolerance_f32(actual: f32, expected: f32, tol: f32) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else if expected.is_infinite() || actual.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!((actual - expected).abs() <= tol.max(4.0 * f32::MIN_POSITIVE));
        }
    }

    fn data_f64(len: usize) -> Vec<f64> {
        (0..len).map(|i| ((i as f64) * 0.25).sin()).collect()
    }

    fn data_f32(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25).sin()).collect()
    }

    #[test]
    fn test_dot_tolerance_f64_within_documented_bounds() {
        let lhs = data_f64(1024);
        let rhs: Vec<f64> = data_f64(1024).into_iter().map(|v| v * -0.5).collect();
        let scalar: f64 = lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).sum();
        if let Some(simd) = simd::try_dot_f64(&lhs, &rhs) {
            let max_abs_a = lhs.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
            let max_abs_b = rhs.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
            let tol = (8.0 * f64::EPSILON * (lhs.len() as f64) * max_abs_a * max_abs_b)
                .max(4.0 * f64::MIN_POSITIVE);
            assert_within_tolerance_f64(simd, scalar, tol);
        }
    }

    #[test]
    fn test_dot_tolerance_f32_within_documented_bounds() {
        let lhs = data_f32(1024);
        let rhs: Vec<f32> = data_f32(1024).into_iter().map(|v| v * -0.5).collect();
        let scalar: f32 = lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).sum();
        if let Some(simd) = simd::try_dot_f32(&lhs, &rhs) {
            let max_abs_a = lhs.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let max_abs_b = rhs.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let tol = (8.0 * f32::EPSILON * (lhs.len() as f32) * max_abs_a * max_abs_b)
                .max(4.0 * f32::MIN_POSITIVE);
            assert_within_tolerance_f32(simd, scalar, tol);
        }
    }

    #[test]
    fn test_complex_dot_tolerance_real_imag_components() {
        let lhs: Vec<Complex<f64>> = (0..1024)
            .map(|i| Complex::new(i as f64 * 0.25, i as f64 * -0.5))
            .collect();
        let rhs: Vec<Complex<f64>> = (0..1024)
            .map(|i| Complex::new((i as f64).cos(), (i as f64).sin()))
            .collect();
        let scalar: Complex<f64> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l.conj() * *r)
            .fold(Complex::new(0.0, 0.0), |a, b| a + b);
        if let Some(simd) = simd::try_dot_complex_f64(&lhs, &rhs) {
            let max_abs_a = lhs.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
            let max_abs_b = rhs.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
            let tol = (16.0 * f64::EPSILON * (lhs.len() as f64) * max_abs_a * max_abs_b)
                .max(4.0 * f64::MIN_POSITIVE);
            assert_within_tolerance_f64(simd.re, scalar.re, tol);
            assert_within_tolerance_f64(simd.im, scalar.im, tol);
        }
    }

    #[test]
    fn test_complex_dot_tolerance_f32_real_imag_components() {
        let lhs: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new(i as f32 * 0.25, i as f32 * -0.5))
            .collect();
        let rhs: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new((i as f32).cos(), (i as f32).sin()))
            .collect();
        let scalar: Complex<f32> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l.conj() * *r)
            .fold(Complex::new(0.0, 0.0), |a, b| a + b);
        if let Some(simd) = simd::try_dot_complex_f32(&lhs, &rhs) {
            let max_abs_a = lhs.iter().map(|c| c.norm()).fold(0.0_f32, f32::max);
            let max_abs_b = rhs.iter().map(|c| c.norm()).fold(0.0_f32, f32::max);
            let tol = (16.0 * f32::EPSILON * (lhs.len() as f32) * max_abs_a * max_abs_b)
                .max(4.0 * f32::MIN_POSITIVE);
            assert_within_tolerance_f32(simd.re, scalar.re, tol);
            assert_within_tolerance_f32(simd.im, scalar.im, tol);
        }
    }

    #[test]
    fn test_dot_dispatch_simd_int_admission() {
        let lhs: Vec<i32> = (0..512).collect();
        let rhs: Vec<i32> = (0..512).map(|v| v - 128).collect();
        if let Some(simd) = simd::try_dot_i32(&lhs, &rhs) {
            let scalar_i64: i64 = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| (l as i64) * (r as i64))
                .sum();
            let scalar_i32 =
                i32::try_from(scalar_i64).expect("test fixture stays within i32 range");
            assert_eq!(simd, scalar_i32);
        }
    }
}
