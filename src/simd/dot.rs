//! f32/f64 dot-product SIMD reduction kernels.

use pulp::{Simd, WithSimd};

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
