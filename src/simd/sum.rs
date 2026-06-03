//! f32/f64 sum SIMD reduction kernels.

use pulp::{Simd, WithSimd};

// ---------------------------------------------------------------------------
// f32 sum kernel
// ---------------------------------------------------------------------------

pub(crate) struct SumF32Kernel<'a> {
    pub(crate) data: &'a [f32],
}

impl WithSimd for SumF32Kernel<'_> {
    type Output = f32;

    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        let (body, tail) = S::as_simd_f32s(self.data);

        // Lane-local accumulation: each lane holds its own partial sum.
        let mut acc = simd.splat_f32s(0.0);
        for &v in body {
            acc = simd.add_f32s(acc, v);
        }

        // Horizontal reduction merge: sum across lanes.
        // FMA is allowed in this phase per 08-simd §6.6 (tolerance is documented).
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
    pub(crate) data: &'a [f64],
}

impl WithSimd for SumF64Kernel<'_> {
    type Output = f64;

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::simd;

    fn tolerance_f32(data: &[f32]) -> f32 {
        let n = data.len() as f64;
        let max_abs = data.iter().map(|v| v.abs() as f64).fold(0.0f64, f64::max);
        // Tolerance per 13-reduction.md §6.3: max(4·ε·n·max_abs_input, 4·MIN_POSITIVE)
        ((4.0 * f32::EPSILON as f64 * n * max_abs) as f32).max(4.0 * f32::MIN_POSITIVE)
    }

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

    fn tolerance_f64(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        (4.0 * f64::EPSILON * n * max_abs).max(4.0 * f64::MIN_POSITIVE)
    }

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
}
