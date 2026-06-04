//! Unary element-wise SIMD kernels (negation).
//!
//! Supported types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.

use pulp::{Simd, WithSimd};

use std::slice;
use crate::complex::Complex;
use crate::simd::{UnaryOp, get_arch};

// ----------------------------------------------------------------------------
// Neg kernel
// ----------------------------------------------------------------------------

/// Element-wise negation: `dst[i] = -src[i]`.
/// Generic over `T`; only `f32` and `f64` monomorphisations are implemented.
pub(crate) struct NegKernel<'a, T> {
    /// Source slice.
    pub(crate) src: &'a [T],

    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [T],

    /// Phantom token to lock monomorphisation to `f32` / `f64`.
    pub(crate) _marker: std::marker::PhantomData<T>,
}

impl WithSimd for NegKernel<'_, f32> {
    type Output = ();

    /// Applies SIMD neg over the body, scalar neg over the tail.
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

    /// Applies SIMD neg over the body, scalar neg over the tail.
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

// ----------------------------------------------------------------------------
// Complex<f32> Neg kernel
// ----------------------------------------------------------------------------

/// Element-wise `Complex<f32>` negation.
/// Reinterprets the interleaved real/imag layout as `[f32]` for SIMD.
pub(crate) struct ComplexNegF32Kernel<'a> {
    /// Source slice (interleaved real/imag).
    pub(crate) src: &'a [Complex<f32>],

    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [Complex<f32>],
}

impl WithSimd for ComplexNegF32Kernel<'_> {
    type Output = ();

    /// Reinterprets complex slices as f32 and applies SIMD neg.
    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.src.len();
        // SAFETY: Complex<f32> is repr(C) with two f32 fields, so the layout
        // is identical to [f32; 2]. Casting through raw pointers preserves
        // provenance and the resulting slice has length 2*n which is exactly
        // the f32 footprint.
        let src_f32 = unsafe {
            slice::from_raw_parts(self.src.as_ptr() as *const f32, n * 2)
        };
        // SAFETY: destination has the same layout as source.
        let dst_f32 = unsafe {
            slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f32, n * 2)
        };
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

// ----------------------------------------------------------------------------
// Complex<f64> Neg kernel
// ----------------------------------------------------------------------------

/// Element-wise `Complex<f64>` negation.
/// Reinterprets the interleaved real/imag layout as `[f64]` for SIMD.
pub(crate) struct ComplexNegF64Kernel<'a> {
    /// Source slice (interleaved real/imag).
    pub(crate) src: &'a [Complex<f64>],

    /// Destination slice (overwritten).
    pub(crate) dst: &'a mut [Complex<f64>],
}

impl WithSimd for ComplexNegF64Kernel<'_> {
    type Output = ();

    /// Reinterprets complex slices as f64 and applies SIMD neg.
    fn with_simd<S: Simd>(self, simd: S) {
        let n = self.src.len();
        // SAFETY: Complex<f64> is repr(C) with two f64 fields;
        // same reasoning as the f32 variant.
        let src_f64 = unsafe {
            slice::from_raw_parts(self.src.as_ptr() as *const f64, n * 2)
        };
        // SAFETY: destination has the same layout as source.
        let dst_f64 = unsafe {
            slice::from_raw_parts_mut(self.dst.as_mut_ptr() as *mut f64, n * 2)
        };
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

// ----------------------------------------------------------------------------
// Dispatch helpers (called from driver.rs facade)
// ----------------------------------------------------------------------------

/// Dispatches f32 unary Neg to the kernel.
pub(crate) fn dispatch_unary_f32(
    op: UnaryOp,
    src: &[f32],
    dst: &mut [f32]
) -> bool {
    if src.len() < super::binary::ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => {
            arch.dispatch(NegKernel {
                src,
                dst,
                _marker: std::marker::PhantomData,
            });
        },
    }
    true
}

/// Dispatches f64 unary Neg to the kernel.
pub(crate) fn dispatch_unary_f64(
    op: UnaryOp,
    src: &[f64],
    dst: &mut [f64]
) -> bool {
    if src.len() < super::binary::ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => {
            arch.dispatch(NegKernel {
                src,
                dst,
                _marker: std::marker::PhantomData,
            });
        },
    }
    true
}

/// Dispatches Complex<f32> unary op to the kernel.
pub(crate) fn dispatch_unary_complex_f32(
    op: UnaryOp,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
) -> bool {
    if src.len() < super::binary::COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => arch.dispatch(ComplexNegF32Kernel { src, dst }),
    }
    true
}

/// Dispatches Complex<f64> unary op to the kernel.
pub(crate) fn dispatch_unary_complex_f64(
    op: UnaryOp,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
) -> bool {
    if src.len() < super::binary::COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => arch.dispatch(ComplexNegF64Kernel { src, dst }),
    }
    true
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd::{dispatch_vector_unary_op, UnaryOp};

    /// Number of random cases per property test.
    const CASES: usize = 32;

    /// Maximum random slice length for property tests.
    const MAX_LEN: usize = 4096;

    // ---- basic correctness -------------------------------------------------

    /// Asserts SIMD and scalar negation produce identical results.
    fn assert_neg_f32(src: &[f32], actual: &[f32]) {
        let expected: Vec<f32> = src.iter().map(|&v| -v).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts SIMD and scalar negation produce identical results.
    fn assert_neg_f64(src: &[f64], actual: &[f64]) {
        let expected: Vec<f64> = src.iter().map(|&v| -v).collect();
        assert_eq!(actual, expected.as_slice());
    }

    /// Asserts 128-element f32 negation goes through SIMD and matches scalar.
    #[test]
    fn test_vector_neg_f32() {
        let src: Vec<f32> = (0..128).map(|v| v as f32 - 64.0).collect();
        let mut dst = vec![0.0f32; src.len()];
        let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_neg_f32(&src, &dst);
    }

    /// Asserts 128-element f64 negation goes through SIMD and matches scalar.
    #[test]
    fn test_vector_neg_f64() {
        let src: Vec<f64> = (0..128).map(|v| v as f64 - 64.0).collect();
        let mut dst = vec![0.0f64; src.len()];
        let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        assert_neg_f64(&src, &dst);
    }

    // ---- consistency vs serial ---------------------------------------------

    // Verifies SIMD neg matches scalar bit-for-bit (or NaN-for-NaN)
    // against a fixture containing extreme float values.

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
        let lhs: Vec<f64> = (0..len)
            .map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f64> = (0..len)
            .map(|i| rhs_seed[i % rhs_seed.len()]).collect();
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
        let lhs: Vec<f32> = (0..len)
            .map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f32> = (0..len)
            .map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    /// Compares f64 SIMD neg against serial for a fixture containing
    /// extreme values, NaNs, and infinities.
    #[test]
    fn test_simd_neg_f64_matches_serial() {
        let (src, _) = fixture_f64(256);
        let mut dst = vec![0.0_f64; src.len()];
        let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        if handled {
            let serial: Vec<f64> = src.iter().map(|&v| -v).collect();
            for (&a, &e) in dst.iter().zip(serial.iter()) {
                assert_same_bits_or_nan_f64(a, e);
            }
        }
    }

    /// Compares f32 SIMD neg against serial for a fixture containing
    /// extreme values, NaNs, and infinities.
    #[test]
    fn test_simd_neg_f32_matches_serial() {
        let (src, _) = fixture_f32(256);
        let mut dst = vec![0.0_f32; src.len()];
        let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        if handled {
            let serial: Vec<f32> = src.iter().map(|&v| -v).collect();
            for (&a, &e) in dst.iter().zip(serial.iter()) {
                assert_same_bits_or_nan_f32(a, e);
            }
        }
    }

    // ---- neg property tests -------------------------------------------------

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

    /// Randomised f64 negation consistency check.
    fn prop_elementwise_neg_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = crate::simd::binary::ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let src: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let mut dst = vec![0.0_f64; len];
            let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
            if handled {
                let serial: Vec<f64> = src.iter().map(|&v| -v).collect();
                for (&a, &e) in dst.iter().zip(serial.iter()) {
                    if a.is_nan() || e.is_nan() {
                        assert!(a.is_nan() && e.is_nan());
                    } else {
                        assert_eq!(a.to_bits(), e.to_bits());
                    }
                }
            }
        }
    }

    /// Aggregates the neg property sub-tests.
    #[test]
    fn test_prop_neg_consistency() {
        prop_elementwise_neg_f64(0x1002);
    }

    // ---- complex neg admission ----------------------------------------------

    /// 128-element `Complex<f32>` negation goes through SIMD and matches scalar.
    #[test]
    fn test_vector_complex_neg_f32() {
        let src: Vec<Complex<f32>> = (0..128)
            .map(|v| Complex::new(v as f32 - 64.0, (v as f32) * 0.5 - 32.0))
            .collect();
        let mut dst = vec![Complex::new(0.0, 0.0); src.len()];
        let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        for (&a, s) in dst.iter().zip(src.iter()) {
            assert_eq!(a.re, -s.re);
            assert_eq!(a.im, -s.im);
        }
    }

    /// 128-element `Complex<f64>` negation goes through SIMD and matches scalar.
    #[test]
    fn test_vector_complex_neg_f64() {
        let src: Vec<Complex<f64>> = (0..128)
            .map(|v| Complex::new(v as f64 - 64.0, (v as f64) * 0.25 - 32.0))
            .collect();
        let mut dst = vec![Complex::new(0.0, 0.0); src.len()];
        let handled = dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        assert!(handled, "len=128 above threshold must admit SIMD");
        for (&a, s) in dst.iter().zip(src.iter()) {
            assert_eq!(a.re, -s.re);
            assert_eq!(a.im, -s.im);
        }
    }
}
