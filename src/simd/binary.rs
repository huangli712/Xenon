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
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
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
}
