//! SIMD reduction kernels for element-wise sum.
//!
//! Accumulates elements in parallel SIMD lanes then reduces
//! horizontally to a single scalar. Supported types: `f32`, `f64`,
//! `Complex<f32>`, `Complex<f64>`.

use pulp::{Simd, WithSimd};

use crate::complex::Complex;

/// Minimum slice length for f32/f64 sum SIMD admission.
const SUM_THRESHOLD: usize = 1024;

/// Minimum slice length for complex sum SIMD admission.
const COMPLEX_SUM_THRESHOLD: usize = 1024;

// ---------------------------------------------------------------------------
// f32 sum kernel
// ---------------------------------------------------------------------------

pub(crate) struct SumF32Kernel<'a> {
    /// Slice of f32 values to sum.
    pub(crate) data: &'a [f32],
}

impl WithSimd for SumF32Kernel<'_> {
    type Output = f32;

    /// Accumulates lane-local sums of f32 body, reduces horizontally, sums tail.
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        let (body, tail) = S::as_simd_f32s(self.data);

        // Lane-local accumulation: each lane holds its own partial sum.
        let mut acc = simd.splat_f32s(0.0);
        for &v in body {
            acc = simd.add_f32s(acc, v);
        }

        // Horizontal reduction merge: sum across lanes.
        let mut scalar = simd.reduce_sum_f32s(acc);

        // Scalar tail: remaining elements after the vector-aligned prefix.
        for &v in tail {
            scalar += v;
        }

        scalar
    }
}

// ---------------------------------------------------------------------------
// f64 sum kernel
// ---------------------------------------------------------------------------

pub(crate) struct SumF64Kernel<'a> {
    /// Slice of f64 values to sum.
    pub(crate) data: &'a [f64],
}

impl WithSimd for SumF64Kernel<'_> {
    type Output = f64;

    /// Accumulates lane-local sums of f64 body, reduces horizontally, sums tail.
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let (body, tail) = S::as_simd_f64s(self.data);

        let mut acc = simd.splat_f64s(0.0);
        for &v in body {
            acc = simd.add_f64s(acc, v);
        }

        let mut scalar = simd.reduce_sum_f64s(acc);

        for &v in tail {
            scalar += v;
        }

        scalar
    }
}

// ---------------------------------------------------------------------------
// Complex<f32> sum kernel
// ---------------------------------------------------------------------------

pub(crate) struct ComplexSumF32Kernel<'a> {
    /// Slice of `Complex<f32>` values to sum (interleaved real/imag).
    pub(crate) data: &'a [Complex<f32>],
}

impl WithSimd for ComplexSumF32Kernel<'_> {
    type Output = Complex<f32>;

    /// Accumulates interleaved real/imag lanes, deinterleaves, sums scalar tail.
    fn with_simd<S: Simd>(self, simd: S) -> Complex<f32> {
        // Reinterpret Complex<f32> as interleaved [re, im, re, im, ...] f32 slice.
        // SAFETY: Complex<f32> is #[repr(C)] with two f32 fields (complex/mod.rs:119-126).
        let f32_data: &[f32] = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.data.len() * 2)
        };
        let (body, tail) = S::as_simd_f32s(f32_data);

        // Accumulate interleaved f32 lanes (real/imag pairs).
        let mut acc = simd.splat_f32s(0.0);
        for &v in body {
            acc = simd.add_f32s(acc, v);
        }

        // Scalar tail.
        let mut re_sum = 0.0f32;
        let mut im_sum = 0.0f32;
        for chunk in tail.chunks(2) {
            re_sum += chunk[0];
            if chunk.len() > 1 {
                im_sum += chunk[1];
            }
        }

        // deinterleave accumulated SIMD vector into real-only and imag-only parts.
        // acc contains [re0, im0, re1, im1, ...]; deinterleave splits into SoA.
        self.deinterleave_and_accumulate::<S>(simd, &acc, &mut re_sum, &mut im_sum)
    }
}

impl ComplexSumF32Kernel<'_> {
    fn deinterleave_and_accumulate<S: Simd>(
        &self,
        simd: S,
        acc: &S::f32s,
        re_sum: &mut f32,
        im_sum: &mut f32,
    ) -> Complex<f32> {
        let lane_count = core::mem::size_of::<S::f32s>() / core::mem::size_of::<f32>();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(acc as *const S::f32s as *const u8, lane_count * 4)
        };
        for i in 0..lane_count / 2 {
            let re = f32::from_ne_bytes([
                bytes[i * 8],
                bytes[i * 8 + 1],
                bytes[i * 8 + 2],
                bytes[i * 8 + 3],
            ]);
            let im = f32::from_ne_bytes([
                bytes[i * 8 + 4],
                bytes[i * 8 + 5],
                bytes[i * 8 + 6],
                bytes[i * 8 + 7],
            ]);
            *re_sum += re;
            *im_sum += im;
        }
        let _ = simd;
        Complex::new(*re_sum, *im_sum)
    }
}

// ---------------------------------------------------------------------------
// Complex<f64> sum kernel
// ---------------------------------------------------------------------------

pub(crate) struct ComplexSumF64Kernel<'a> {
    /// Slice of `Complex<f64>` values to sum (interleaved real/imag).
    pub(crate) data: &'a [Complex<f64>],
}

impl WithSimd for ComplexSumF64Kernel<'_> {
    type Output = Complex<f64>;

    /// Same as ComplexSumF32Kernel, using f64 lanes.
    fn with_simd<S: Simd>(self, simd: S) -> Complex<f64> {
        let f64_data: &[f64] = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const f64, self.data.len() * 2)
        };
        let (body, tail) = S::as_simd_f64s(f64_data);

        let mut acc = simd.splat_f64s(0.0);
        for &v in body {
            acc = simd.add_f64s(acc, v);
        }

        let mut re_sum = 0.0f64;
        let mut im_sum = 0.0f64;
        for chunk in tail.chunks(2) {
            re_sum += chunk[0];
            if chunk.len() > 1 {
                im_sum += chunk[1];
            }
        }

        // Horizontal reduction merge across lanes.
        let scalar = simd.reduce_sum_f64s(acc);
        // Since we summed [re0, im0, re1, im1, ...] all together,
        // scalar = sum of all re + sum of all im. We need to split them.
        // Use bytemuck to manually extract lanes.
        let lane_count = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&acc as *const S::f64s as *const u8, lane_count * 8)
        };
        for i in 0..lane_count / 2 {
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&bytes[i * 16..i * 16 + 8]);
            im_bytes.copy_from_slice(&bytes[i * 16 + 8..i * 16 + 16]);
            re_sum += f64::from_ne_bytes(re_bytes);
            im_sum += f64::from_ne_bytes(im_bytes);
        }
        let _ = (simd, scalar);
        Complex::new(re_sum, im_sum)
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers (called from mod.rs facade)
// ---------------------------------------------------------------------------

/// Dispatches f32 sum to the SIMD kernel if the threshold is met.
pub(crate) fn try_sum_f32_impl(data: &[f32]) -> Option<f32> {
    if data.len() < SUM_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(SumF32Kernel { data }))
}

/// Dispatches f64 sum to the SIMD kernel if the threshold is met.
pub(crate) fn try_sum_f64_impl(data: &[f64]) -> Option<f64> {
    if data.len() < SUM_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(SumF64Kernel { data }))
}

pub(crate) fn try_sum_complex_f32_impl(data: &[Complex<f32>]) -> Option<Complex<f32>> {
    if data.len() < COMPLEX_SUM_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(ComplexSumF32Kernel { data }))
}

pub(crate) fn try_sum_complex_f64_impl(data: &[Complex<f64>]) -> Option<Complex<f64>> {
    if data.len() < COMPLEX_SUM_THRESHOLD {
        return None;
    }
    let arch = crate::simd::get_arch();
    Some(arch.dispatch(ComplexSumF64Kernel { data }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd;

    // ---- admission / basic correctness ----

    /// Computing tolerance as 4·ε·n·max(|input|) — a documented bound
    /// for floating-point SIMD sum accumulation.
    /// Computing tolerance as 4·ε·n·max(|input|) — a documented bound
    /// for floating-point SIMD sum accumulation.
    fn tolerance_f32(data: &[f32]) -> f32 {
        let n = data.len() as f64;
        let max_abs = data.iter().map(|v| v.abs() as f64).fold(0.0f64, f64::max);
        // Tolerance per 13-reduction.md §6.3: max(4·ε·n·max_abs_input, 4·MIN_POSITIVE)
        ((4.0 * f32::EPSILON as f64 * n * max_abs) as f32).max(4.0 * f32::MIN_POSITIVE)
    }

    /// Asserts 2048-element f32 sum enters SIMD and stays within tolerance.
    #[test]
    fn test_sum_dispatch_simd_float_f32() {
        let data: Vec<f32> = (0..2048).map(|v| v as f32 * 0.25 - 64.0).collect();
        let simd_result = simd::try_sum_f32(&data);
        assert!(
            simd_result.is_some(),
            "len >= 1024 should enter SIMD sum path when supported"
        );
        let simd = simd_result.expect("len >= 1024 should enter SIMD sum path");
        let scalar: f32 = data.iter().sum();
        let tol = tolerance_f32(&data);
        assert!(
            (simd - scalar).abs() <= tol,
            "SIMD sum {simd} deviates from scalar {scalar} beyond {tol}"
        );
    }

    /// Computing tolerance as 4·ε·n·max(|input|) for f64.
    fn tolerance_f64(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        (4.0 * f64::EPSILON * n * max_abs).max(4.0 * f64::MIN_POSITIVE)
    }

    /// Asserts 2048-element f64 sum enters SIMD and stays within tolerance.
    #[test]
    fn test_sum_dispatch_simd_float_f64() {
        let data: Vec<f64> = (0..2048).map(|v| v as f64 * 0.125 - 128.0).collect();
        let simd_result = simd::try_sum_f64(&data);
        assert!(
            simd_result.is_some(),
            "len >= 1024 should enter SIMD sum path when supported"
        );
        let simd = simd_result.expect("len >= 1024 should enter SIMD sum path");
        let scalar: f64 = data.iter().sum();
        let tol = tolerance_f64(&data);
        assert!(
            (simd - scalar).abs() <= tol,
            "SIMD sum {simd} deviates from scalar {scalar} beyond {tol}"
        );
    }

    /// Asserts f32 lengths below threshold are rejected, at threshold admitted.
    #[test]
    fn test_simd_sum_threshold_boundary() {
        let below: Vec<f32> = (0..1023).map(|v| v as f32).collect();
        assert!(
            simd::try_sum_f32(&below).is_none(),
            "len=1023 must stay below SIMD threshold"
        );

        let at_threshold: Vec<f32> = (0..1024).map(|v| v as f32).collect();
        assert!(
            simd::try_sum_f32(&at_threshold).is_some(),
            "len=1024 must be admitted when supported"
        );
    }

    // ---- tolerance bounds ----

    /// Asserts f64 is within tolerance (or matches NaN/∞).
    fn assert_within_tolerance_f64(actual: f64, expected: f64, tol: f64) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else if expected.is_infinite() || actual.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!((actual - expected).abs() <= tol.max(4.0 * f64::MIN_POSITIVE));
        }
    }

    /// Asserts f32 is within tolerance (or matches NaN/∞).
    fn assert_within_tolerance_f32(actual: f32, expected: f32, tol: f32) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else if expected.is_infinite() || actual.is_infinite() {
            assert_eq!(actual, expected);
        } else {
            assert!((actual - expected).abs() <= tol.max(4.0 * f32::MIN_POSITIVE));
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

    /// Asserts f64 sum is within tolerance when SIMD is available.
    #[test]
    fn test_sum_tolerance_f64_within_documented_bounds() {
        let data = data_f64(2048);
        let scalar: f64 = data.iter().sum();
        if let Some(simd) = simd::try_sum_f64(&data) {
            assert_within_tolerance_f64(simd, scalar, tolerance_f64(&data));
        }
    }

    /// Asserts f32 sum is within tolerance when SIMD is available.
    #[test]
    fn test_sum_tolerance_f32_within_documented_bounds() {
        let data = data_f32(2048);
        let scalar: f32 = data.iter().sum();
        if let Some(simd) = simd::try_sum_f32(&data) {
            assert_within_tolerance_f32(simd, scalar, tolerance_f32(&data));
        }
    }

    /// Asserts complex f64 sum matches scalar on both real and imag
    /// components within documented tolerance.
    #[test]
    fn test_complex_sum_tolerance_real_imag_components() {
        let data: Vec<Complex<f64>> = (0..2048)
            .map(|i| Complex::new((i as f64).sin(), (i as f64 * 0.5).cos()))
            .collect();
        let scalar: Complex<f64> = data
            .iter()
            .copied()
            .fold(Complex::new(0.0, 0.0), |a, b| a + b);
        if let Some(simd) = simd::try_sum_complex_f64(&data) {
            let real: Vec<f64> = data.iter().map(|v| v.re).collect();
            let imag: Vec<f64> = data.iter().map(|v| v.im).collect();
            assert_within_tolerance_f64(simd.re, scalar.re, tolerance_f64(&real));
            assert_within_tolerance_f64(simd.im, scalar.im, tolerance_f64(&imag));
        }
    }

    /// Asserts complex f32 sum matches scalar on both real and imag
    /// components within documented tolerance.
    #[test]
    fn test_complex_sum_tolerance_f32_real_imag_components() {
        let data: Vec<Complex<f32>> = (0..2048)
            .map(|i| Complex::new((i as f32).sin(), (i as f32 * 0.5).cos()))
            .collect();
        let scalar: Complex<f32> = data
            .iter()
            .copied()
            .fold(Complex::new(0.0, 0.0), |a, b| a + b);
        if let Some(simd) = simd::try_sum_complex_f32(&data) {
            let real: Vec<f32> = data.iter().map(|v| v.re).collect();
            let imag: Vec<f32> = data.iter().map(|v| v.im).collect();
            assert_within_tolerance_f32(simd.re, scalar.re, tolerance_f32(&real));
            assert_within_tolerance_f32(simd.im, scalar.im, tolerance_f32(&imag));
        }
    }

    /// Asserts i32 sum (stub) returns None so caller falls back to scalar.
    #[test]
    fn test_sum_dispatch_simd_int_admission() {
        let data: Vec<i32> = (0..1024).collect();
        if let Some(simd) = simd::try_sum_i32(&data) {
            let scalar_i64: i64 = data.iter().map(|&v| v as i64).sum();
            let scalar_i32 =
                i32::try_from(scalar_i64).expect("test fixture stays within i32 range");
            assert_eq!(simd, scalar_i32);
        }
    }

    /// Checks that NaN in the input propagates through the SIMD sum path.
    #[test]
    fn test_sum_nan_propagation() {
        let mut data = vec![1.0_f64; 2048];
        data[1024] = f64::NAN;
        if let Some(simd) = simd::try_sum_f64(&data) {
            assert!(simd.is_nan());
        }
    }

    /// Checks that `+∞` in the input yields `+∞` from SIMD sum.
    #[test]
    fn test_sum_inf_sign_consistency() {
        let mut positive = vec![1.0_f64; 2048];
        positive[7] = f64::INFINITY;
        if let Some(simd) = simd::try_sum_f64(&positive) {
            assert_eq!(simd, f64::INFINITY);
        }
    }

    /// Just-below-threshold is rejected; just-at-threshold is admitted
    /// and within tolerance.
    #[test]
    fn test_entry_threshold_boundary() {
        let below = vec![1.0_f64; 1023];
        assert!(simd::try_sum_f64(&below).is_none());
        let at_threshold = vec![1.0_f64; 1024];
        let simd = simd::try_sum_f64(&at_threshold).expect("len=1024 must enter f64 sum SIMD");
        assert_within_tolerance_f64(simd, 1024.0, tolerance_f64(&at_threshold));
    }

    // ---- sum property tests ----
    //
    // Randomized property-based tests that compare SIMD sum against
    // scalar across many seed-driven random inputs.

    /// Number of random cases per property test.
    const CASES: usize = 32;
    /// Maximum random slice length for property tests.
    const MAX_LEN: usize = 4096;
    /// Sum threshold used by property tests.
    const SUM_THRESHOLD: usize = 1024;
    /// Complex sum threshold used by property tests.
    const COMPLEX_SUM_THRESHOLD: usize = 1024;

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

    /// Loose tolerance bound based on expected magnitude.
    fn reduction_bound_f64(expected: f64, len: usize) -> f64 {
        let eps = f64::EPSILON;
        let magnitude = expected.abs().max(1.0);
        ((len as f64) * eps * magnitude * 4.0).max(4.0 * f64::MIN_POSITIVE)
    }

    /// Loose tolerance bound for f32.
    fn reduction_bound_f32(expected: f32, len: usize) -> f32 {
        let eps = f32::EPSILON;
        let magnitude = expected.abs().max(1.0);
        ((len as f32) * eps * magnitude * 4.0).max(4.0 * f32::MIN_POSITIVE)
    }

    /// Asserts f64 is within a generous reduction bound.
    fn assert_within_reduction_bound_f64(actual: f64, expected: f64, len: usize, op: &str) {
        let bound = reduction_bound_f64(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: actual={actual}, expected={expected}, bound={bound}"
        );
    }

    /// Asserts f32 is within a generous reduction bound.
    fn assert_within_reduction_bound_f32(actual: f32, expected: f32, len: usize, op: &str) {
        let bound = reduction_bound_f32(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: actual={actual}, expected={expected}, bound={bound}"
        );
    }

    /// Randomised f64 sum within tolerance check.
    fn prop_sum_tolerance_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = SUM_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let data: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            if let Some(simd) = simd::try_sum_f64(&data) {
                let scalar: f64 = data.iter().sum();
                assert_within_reduction_bound_f64(simd, scalar, len, "sum f64");
            }
        }
    }

    /// Randomised f32 sum within tolerance check.
    fn prop_sum_tolerance_f32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = SUM_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let data: Vec<f32> = (0..len).map(|_| gen_f32(&mut rng)).collect();
            if let Some(simd) = simd::try_sum_f32(&data) {
                let scalar: f32 = data.iter().sum();
                assert_within_reduction_bound_f32(simd, scalar, len, "sum f32");
            }
        }
    }

    /// Randomised complex f64 sum within tolerance check.
    fn prop_sum_complex_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = COMPLEX_SUM_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let data: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            if let Some(simd) = simd::try_sum_complex_f64(&data) {
                let scalar: Complex<f64> = data
                    .iter()
                    .copied()
                    .fold(Complex::new(0.0, 0.0), |a, b| a + b);
                assert_within_reduction_bound_f64(simd.re, scalar.re, len, "complex sum f64 re");
                assert_within_reduction_bound_f64(simd.im, scalar.im, len, "complex sum f64 im");
            }
        }
    }

    /// Aggregate: runs f64 sum, f32 sum, and complex sum property tests.
    #[test]
    fn prop_sum_tolerance() {
        prop_sum_tolerance_f64(0x2001);
        prop_sum_tolerance_f32(0x2002);
        prop_sum_complex_f64(0x2003);
    }

    // ---- integer stub ----

    /// Generates a random i32 that won't overflow during widening.
    fn gen_i32_no_overflow(state: &mut u64) -> i32 {
        ((splitmix64(state) % 2001) as i32) - 1000
    }

    /// Verifies the i32 sum stub never panics across random sizes.
    fn prop_integer_no_panic_i32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = gen_len(&mut rng, MAX_LEN);
            let data: Vec<i32> = (0..len).map(|_| gen_i32_no_overflow(&mut rng)).collect();
            // i32 SIMD widening is unavailable; stub always returns None.
            assert!(
                simd::try_sum_i32(&data).is_none(),
                "i32 SIMD sum should not be available (widening unavailable)"
            );
        }
    }

    /// Verifies the i32 sum stub never panics and always returns None.
    #[test]
    fn prop_integer_no_panic() {
        prop_integer_no_panic_i32(0x4001);
    }
}
