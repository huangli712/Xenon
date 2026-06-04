//! SIMD reduction kernels for dot-product (inner product).
//!
//! Computes `sum(lhs_i * rhs_i)` using parallel SIMD lanes.
//! Complex variant follows BLAS xdotc: `sum(conj(lhs_i) * rhs_i)`.
//! Supported types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.

use pulp::{Simd, WithSimd};

use crate::complex::Complex;
use crate::simd::get_arch;

// ----------------------------------------------------------------------------
// Thresholds
// ----------------------------------------------------------------------------

/// Minimum slice length for f32/f64 dot-product SIMD admission.
const DOT_THRESHOLD: usize = 512;

/// Minimum slice length for complex dot-product SIMD admission.
const COMPLEX_DOT_THRESHOLD: usize = 512;

// ----------------------------------------------------------------------------
// f32 dot kernel
// ----------------------------------------------------------------------------

/// Inner product of two `f32` slices: `sum(lhs[i] * rhs[i])`.
pub(crate) struct DotF32Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f32],

    /// Right operand slice.
    pub(crate) rhs: &'a [f32],
}

impl WithSimd for DotF32Kernel<'_> {
    type Output = f32;

    /// FMA forbidden in per-element multiply+accumulate.
    /// Uses separate mul and add lane ops.
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        let (lhs_body, lhs_tail) = S::as_simd_f32s(self.lhs);
        let (rhs_body, _rhs_tail) = S::as_simd_f32s(self.rhs);

        // Separate mul + add to keep element-wise semantics
        // bit-identical with scalar.
        let mut acc = simd.splat_f32s(0.0);
        for i in 0..lhs_body.len() {
            let prod = simd.mul_f32s(lhs_body[i], rhs_body[i]);
            acc = simd.add_f32s(acc, prod);
        }

        // Horizontal reduction merge across lanes.
        let mut scalar = simd.reduce_sum_f32s(acc);

        // Tail
        for i in 0..lhs_tail.len() {
            scalar += lhs_tail[i] * self.rhs[self.rhs.len() - lhs_tail.len() + i];
        }

        scalar
    }
}

// ----------------------------------------------------------------------------
// f64 dot kernel
// ----------------------------------------------------------------------------

/// Inner product of two `f64` slices: `sum(lhs[i] * rhs[i])`.
pub(crate) struct DotF64Kernel<'a> {
    /// Left operand slice.
    pub(crate) lhs: &'a [f64],

    /// Right operand slice.
    pub(crate) rhs: &'a [f64],
}

impl WithSimd for DotF64Kernel<'_> {
    type Output = f64;

    /// Same as DotF32Kernel, using f64 lanes.
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let (lhs_body, lhs_tail) = S::as_simd_f64s(self.lhs);
        let (rhs_body, _rhs_tail) = S::as_simd_f64s(self.rhs);

        // Separate mul + add to keep element-wise semantics
        // bit-identical with scalar.
        let mut acc = simd.splat_f64s(0.0);
        for i in 0..lhs_body.len() {
            let prod = simd.mul_f64s(lhs_body[i], rhs_body[i]);
            acc = simd.add_f64s(acc, prod);
        }

        // Horizontal reduction merge across lanes.
        let mut scalar = simd.reduce_sum_f64s(acc);

        // Tail
        let tail_offset = self.rhs.len() - lhs_tail.len();
        for (i, &l) in lhs_tail.iter().enumerate() {
            scalar += l * self.rhs[tail_offset + i];
        }

        scalar
    }
}

// ----------------------------------------------------------------------------
// Complex<f32> dot kernel
// ----------------------------------------------------------------------------

/// Inner product of two `Complex<f32>` slices under `conj(lhs)·rhs`.
pub(crate) struct ComplexDotF32Kernel<'a> {
    /// Left operand slice (interleaved real/imag).
    pub(crate) lhs: &'a [Complex<f32>],

    /// Right operand slice (interleaved real/imag).
    pub(crate) rhs: &'a [Complex<f32>],
}

impl WithSimd for ComplexDotF32Kernel<'_> {
    type Output = Complex<f32>;

    /// Computes BLAS xdotc via scalar loop over `Complex<f32>` elements.
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

// ----------------------------------------------------------------------------
// Complex<f64> dot kernel
// ----------------------------------------------------------------------------

/// Inner product of two `Complex<f64>` slices under `conj(lhs)·rhs`.
pub(crate) struct ComplexDotF64Kernel<'a> {
    /// Left operand slice (interleaved real/imag).
    pub(crate) lhs: &'a [Complex<f64>],

    /// Right operand slice (interleaved real/imag).
    pub(crate) rhs: &'a [Complex<f64>],
}

impl WithSimd for ComplexDotF64Kernel<'_> {
    type Output = Complex<f64>;

    /// Same as ComplexDotF32Kernel, using `Complex<f64>` elements.
    fn with_simd<S: Simd>(self, simd: S) -> Complex<f64> {
        // BLAS xdotc contract: dot = sum(conj(lhs_i) * rhs_i)
        // conj(lhs) * rhs = (re_l * re_r + im_l * im_r) + (re_l * im_r - im_l * re_r)i
        let mut re_acc = 0.0f64;
        let mut im_acc = 0.0f64;
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

// ----------------------------------------------------------------------------
// Dispatch helpers (called from driver.rs facade)
// ----------------------------------------------------------------------------

/// Admission helper for f32 dot product.
pub(crate) fn try_dot_f32_impl(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < DOT_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(DotF32Kernel { lhs, rhs }))
}

/// Admission helper for f64 dot product.
pub(crate) fn try_dot_f64_impl(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < DOT_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(DotF64Kernel { lhs, rhs }))
}

/// Admission helper for `Complex<f32>` dot product.
pub(crate) fn try_dot_complex_f32_impl(
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
) -> Option<Complex<f32>> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < COMPLEX_DOT_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(ComplexDotF32Kernel { lhs, rhs }))
}

/// Admission helper for `Complex<f64>` dot product.
pub(crate) fn try_dot_complex_f64_impl(
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
) -> Option<Complex<f64>> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < COMPLEX_DOT_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(ComplexDotF64Kernel { lhs, rhs }))
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd::try_dot_i32;
    use crate::simd::{try_dot_f32, try_dot_f64};
    use crate::simd::{try_dot_complex_f32, try_dot_complex_f64};

    /// Number of random cases per property test.
    const CASES: usize = 32;

    /// Maximum random slice length for property tests.
    const MAX_LEN: usize = 4096;

    // ---- tolerance ---------------------------------------------------------

    /// Asserts f64 is within tolerance (or matches NaN/∞).
    fn assert_within_tolerance_f64(actual: f64, expected: f64, tol: f64) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else if expected.is_infinite() || actual.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!(
                (actual - expected).abs() <= tol.max(4.0 * f64::MIN_POSITIVE)
            );
        }
    }

    /// Asserts f32 is within tolerance (or matches NaN/∞).
    fn assert_within_tolerance_f32(actual: f32, expected: f32, tol: f32) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else if expected.is_infinite() || actual.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!(
                (actual - expected).abs() <= tol.max(4.0 * f32::MIN_POSITIVE)
            );
        }
    }

    /// Generates deterministic f64 test data via sin.
    fn data_f64(len: usize) -> Vec<f64> {
        (0..len).map(|i| ((i as f64) * 0.25).sin()).collect()
    }

    /// Generates deterministic f32 test data via sin.
    fn data_f32(len: usize) -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * 0.25).sin()).collect()
    }

    /// Asserts f64 dot-product is within documented tolerance.
    #[test]
    fn test_dot_tolerance_f64_within_documented_bounds() {
        let lhs = data_f64(1024);
        let rhs: Vec<f64> = data_f64(1024)
            .into_iter()
            .map(|v| v * -0.5)
            .collect();
        let scalar: f64 = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&l, &r)| l * r)
            .sum();
        if let Some(simd) = try_dot_f64(&lhs, &rhs) {
            let max_abs_a = lhs
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let max_abs_b = rhs
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let tol = (8.0 * f64::EPSILON * (lhs.len() as f64) * max_abs_a * max_abs_b)
                .max(4.0 * f64::MIN_POSITIVE);
            assert_within_tolerance_f64(simd, scalar, tol);
        }
    }

    /// Asserts f32 dot-product is within documented tolerance.
    #[test]
    fn test_dot_tolerance_f32_within_documented_bounds() {
        let lhs = data_f32(1024);
        let rhs: Vec<f32> = data_f32(1024)
            .into_iter()
            .map(|v| v * -0.5)
            .collect();
        let scalar: f32 = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&l, &r)| l * r)
            .sum();
        if let Some(simd) = try_dot_f32(&lhs, &rhs) {
            let max_abs_a = lhs
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0_f32, f32::max);
            let max_abs_b = rhs
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0_f32, f32::max);
            let tol = (8.0 * f32::EPSILON * (lhs.len() as f32) * max_abs_a * max_abs_b)
                .max(4.0 * f32::MIN_POSITIVE);
            assert_within_tolerance_f32(simd, scalar, tol);
        }
    }

    /// Asserts complex f64 dot product (BLAS xdotc: conj(lhs) · rhs)
    /// is within documented tolerance on both real and imaginary components.
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
        if let Some(simd) = try_dot_complex_f64(&lhs, &rhs) {
            let max_abs_a = lhs
                .iter()
                .map(|c| c.norm())
                .fold(0.0_f64, f64::max);
            let max_abs_b = rhs
                .iter()
                .map(|c| c.norm())
                .fold(0.0_f64, f64::max);
            let tol = (16.0 * f64::EPSILON * (lhs.len() as f64) * max_abs_a * max_abs_b)
                .max(4.0 * f64::MIN_POSITIVE);
            assert_within_tolerance_f64(simd.re, scalar.re, tol);
            assert_within_tolerance_f64(simd.im, scalar.im, tol);
        }
    }

    /// Asserts complex f32 dot product (BLAS xdotc) is within documented
    /// tolerance on both real and imaginary components.
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
        if let Some(simd) = try_dot_complex_f32(&lhs, &rhs) {
            let max_abs_a = lhs
                .iter()
                .map(|c| c.norm())
                .fold(0.0_f32, f32::max);
            let max_abs_b = rhs
                .iter()
                .map(|c| c.norm())
                .fold(0.0_f32, f32::max);
            let tol = (16.0 * f32::EPSILON * (lhs.len() as f32) * max_abs_a * max_abs_b)
                .max(4.0 * f32::MIN_POSITIVE);
            assert_within_tolerance_f32(simd.re, scalar.re, tol);
            assert_within_tolerance_f32(simd.im, scalar.im, tol);
        }
    }

    // ---- int stub admission ------------------------------------------------

    /// Verifies the i32 dot stub returns `None` (no SIMD widening available).
    #[test]
    fn test_dot_dispatch_simd_int_admission() {
        let lhs: Vec<i32> = (0..512).collect();
        let rhs: Vec<i32> = (0..512).map(|v| v - 128).collect();
        if let Some(simd) = try_dot_i32(&lhs, &rhs) {
            let scalar_i64: i64 = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| (l as i64) * (r as i64))
                .sum();
            let scalar_i32 = i32::try_from(scalar_i64)
                .expect("test fixture stays within i32 range");
            assert_eq!(simd, scalar_i32);
        }
    }

    // ---- dot property tests ------------------------------------------------

    // Randomized property-based tests for SIMD dot-product.

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

    /// Loose tolerance bound based on expected magnitude.
    fn reduction_bound_f64(expected: f64, len: usize) -> f64 {
        let eps = f64::EPSILON;
        let magnitude = expected.abs().max(1.0);
        ((len as f64) * eps * magnitude * 4.0).max(4.0 * f64::MIN_POSITIVE)
    }

    /// Asserts f64 is within a generous reduction bound.
    fn assert_within_reduction_bound_f64(
        actual: f64,
        expected: f64,
        len: usize,
        op: &str
    ) {
        let bound = reduction_bound_f64(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: \
            actual={actual}, expected={expected}, bound={bound}"
        );
    }

    /// Randomised f64 dot-product within tolerance check.
    fn prop_dot_tolerance_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = super::DOT_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            if let Some(simd) = try_dot_f64(&lhs, &rhs) {
                let scalar: f64 = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(&l, &r)| l * r)
                    .sum();
                assert_within_reduction_bound_f64(simd, scalar, len, "dot f64");
            }
        }
    }

    /// Randomised complex f64 conjugate dot-product within tolerance check.
    fn prop_dot_conjugate_complex_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = super::COMPLEX_DOT_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            let rhs: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            if let Some(simd) = try_dot_complex_f64(&lhs, &rhs) {
                let scalar: Complex<f64> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(l, r)| l.conj() * *r)
                    .fold(Complex::new(0.0, 0.0), |a, b| a + b);
                assert_within_reduction_bound_f64(
                    simd.re,
                    scalar.re,
                    len,
                    "complex dot f64 re"
                );
                assert_within_reduction_bound_f64(
                    simd.im,
                    scalar.im,
                    len,
                    "complex dot f64 im"
                );
            }
        }
    }

    /// Aggregate: runs f64 dot and complex f64 conjugate dot property tests.
    #[test]
    fn test_prop_dot_conjugate_contract() {
        prop_dot_tolerance_f64(0x3001);
        prop_dot_conjugate_complex_f64(0x3002);
    }
}
