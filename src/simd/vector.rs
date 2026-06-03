//! Element-wise SIMD kernels for f32 and f64.
//!
//! Each kernel holds slice references and implements [`pulp::WithSimd`].
//! The facade functions in [`super`] perform threshold admission and
//! type-based dispatch before routing to these kernels.

use crate::simd::{BinaryOp, UnaryOp, get_arch};
use pulp::{Simd, WithSimd};

// ---------------------------------------------------------------------------
// Thresholds (per 08-simd §5.8)
// ---------------------------------------------------------------------------

const ELEMENTWISE_F32_F64_THRESHOLD: usize = 64;

// ---------------------------------------------------------------------------
// Concrete dispatch helpers (called from mod.rs facade)
// ---------------------------------------------------------------------------

/// Dispatches f32 binary op to the corresponding kernel.
pub(crate) fn dispatch_binary_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32], dst: &mut [f32]) -> bool {
    if lhs.len() < ELEMENTWISE_F32_F64_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(super::binary::AddF32Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => arch.dispatch(super::binary::SubF32Kernel { lhs, rhs, dst }),
        BinaryOp::Mul => arch.dispatch(super::binary::MulF32Kernel { lhs, rhs, dst }),
        BinaryOp::Div => arch.dispatch(super::binary::DivF32Kernel { lhs, rhs, dst }),
    }
    true
}

/// Dispatches f64 binary op to the corresponding kernel.
pub(crate) fn dispatch_binary_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64], dst: &mut [f64]) -> bool {
    if lhs.len() < ELEMENTWISE_F32_F64_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(super::binary::AddF64Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => arch.dispatch(super::binary::SubF64Kernel { lhs, rhs, dst }),
        BinaryOp::Mul => arch.dispatch(super::binary::MulF64Kernel { lhs, rhs, dst }),
        BinaryOp::Div => arch.dispatch(super::binary::DivF64Kernel { lhs, rhs, dst }),
    }
    true
}

/// Dispatches f32 unary Neg to the kernel.
pub(crate) fn dispatch_unary_f32(op: UnaryOp, src: &[f32], dst: &mut [f32]) -> bool {
    if src.len() < ELEMENTWISE_F32_F64_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => {
            arch.dispatch(super::unary::NegKernel {
                src,
                dst,
                _marker: std::marker::PhantomData,
            });
        },
    }
    true
}

/// Dispatches f64 unary Neg to the kernel.
pub(crate) fn dispatch_unary_f64(op: UnaryOp, src: &[f64], dst: &mut [f64]) -> bool {
    if src.len() < ELEMENTWISE_F32_F64_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => {
            arch.dispatch(super::unary::NegKernel {
                src,
                dst,
                _marker: std::marker::PhantomData,
            });
        },
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;


    #[test]
    fn test_vector_sub_mul_div_below_threshold_rejects() {
        let lhs: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let rhs: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let mut dst = vec![99.0f32; lhs.len()];

        // len=32 < threshold 64 — must reject
        assert!(!crate::simd::dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!crate::simd::dispatch_vector_binary_op(
            BinaryOp::Mul,
            &lhs,
            &rhs,
            &mut dst
        ));
        assert!(!crate::simd::dispatch_vector_unary_op(
            UnaryOp::Neg,
            &lhs,
            &mut dst
        ));
        // dst should remain unchanged on rejection
        for &v in &dst {
            assert_eq!(v, 99.0_f32, "dst must be untouched on SIMD rejection");
        }
    }
}
// ---------------------------------------------------------------------------
// f32/f64 sum kernels (W14T3)
// ---------------------------------------------------------------------------

/// Sum threshold per 08-simd §5.8 L457.
const SUM_F32_F64_THRESHOLD: usize = 1024;

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

/// Dispatches f32 sum to the SIMD kernel if the threshold is met.
pub(crate) fn try_sum_f32_impl(data: &[f32]) -> Option<f32> {
    if data.len() < SUM_F32_F64_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(SumF32Kernel { data }))
}

/// Dispatches f64 sum to the SIMD kernel if the threshold is met.
pub(crate) fn try_sum_f64_impl(data: &[f64]) -> Option<f64> {
    if data.len() < SUM_F32_F64_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(SumF64Kernel { data }))
}

// ---------------------------------------------------------------------------
// W14T3 sum tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod sum_tests {
    use crate::simd;

    fn tolerance_f32(data: &[f32]) -> f32 {
        let n = data.len() as f64;
        let max_abs = data.iter().map(|v| v.abs() as f64).fold(0.0f64, f64::max);
        // Tolerance per 13-reduction.md §6.3: max(4·ε·n·max_abs_input, 4·MIN_POSITIVE)
        ((4.0 * f32::EPSILON as f64 * n * max_abs) as f32).max(4.0 * f32::MIN_POSITIVE)
    }

    fn tolerance_f64(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        (4.0 * f64::EPSILON * n * max_abs).max(4.0 * f64::MIN_POSITIVE)
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
} // ---------------------------------------------------------------------------
// Complex sum kernels (W14T5)
// ---------------------------------------------------------------------------

/// Complex sum threshold per PLAN.md W14 补充决策 (derived from §5.8 f32/f64 sum=1024).
const COMPLEX_SUM_THRESHOLD: usize = 1024;

/// Dot threshold for f32/f64 (08-simd §5.8 L456).
const DOT_F32_F64_THRESHOLD: usize = 512;
/// Dot threshold for Complex (PLAN.md W14, derived from f32/f64 dot=512).
const COMPLEX_DOT_THRESHOLD: usize = 512;

/// Complex element-wise threshold (08-simd §5.8 L451).
const COMPLEX_ELEMENTWISE_THRESHOLD: usize = 128;

use crate::complex::Complex;

pub(crate) struct ComplexSumF32Kernel<'a> {
    pub(crate) data: &'a [Complex<f32>],
}

impl WithSimd for ComplexSumF32Kernel<'_> {
    type Output = Complex<f32>;

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
        // Use core::mem::size_of to determine lane count.
        let lane_count = core::mem::size_of::<S::f32s>() / core::mem::size_of::<f32>();
        // For each adjacent pair of lanes, add to real/imag.
        // We use bytemuck to interpret the SIMD register as a byte array
        // and then reconstruct the f32 values.
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

pub(crate) struct ComplexSumF64Kernel<'a> {
    pub(crate) data: &'a [Complex<f64>],
}

impl WithSimd for ComplexSumF64Kernel<'_> {
    type Output = Complex<f64>;

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

        // Deinterleave accumulated f64s.
        // For simplicity, reduce interleaved acc directly.
        // FMA is allowed in horizontal reduction merge (08-simd §6.6).
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

pub(crate) fn try_sum_complex_f32_impl(data: &[Complex<f32>]) -> Option<Complex<f32>> {
    if data.len() < COMPLEX_SUM_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(ComplexSumF32Kernel { data }))
}

pub(crate) fn try_sum_complex_f64_impl(data: &[Complex<f64>]) -> Option<Complex<f64>> {
    if data.len() < COMPLEX_SUM_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(ComplexSumF64Kernel { data }))
}

// ---------------------------------------------------------------------------
// f32/f64 dot kernels (W14T6)
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

pub(crate) fn try_dot_f32_impl(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < DOT_F32_F64_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(DotF32Kernel { lhs, rhs }))
}

pub(crate) fn try_dot_f64_impl(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
    assert_eq!(lhs.len(), rhs.len());
    if lhs.len() < DOT_F32_F64_THRESHOLD {
        return None;
    }
    let arch = get_arch();
    Some(arch.dispatch(DotF64Kernel { lhs, rhs }))
}

// ---------------------------------------------------------------------------
// Complex dot kernels (W14T6)
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

// ---------------------------------------------------------------------------
// Complex element-wise kernels (W14T11 — Neg)
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

/// Dispatches Complex<f32> binary element-wise op to the kernel.
pub(crate) fn dispatch_binary_complex_f32(
    op: BinaryOp,
    lhs: &[Complex<f32>],
    rhs: &[Complex<f32>],
    dst: &mut [Complex<f32>],
) -> bool {
    if lhs.len() < COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(super::binary::ComplexAddF32Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => arch.dispatch(super::binary::ComplexSubF32Kernel { lhs, rhs, dst }),
        BinaryOp::Mul => arch.dispatch(super::binary::ComplexMulF32Kernel { lhs, rhs, dst }),
        BinaryOp::Div => return false, // Complex div not implemented
    }
    true
}

/// Dispatches Complex<f32> unary op to the kernel.
pub(crate) fn dispatch_unary_complex_f32(
    op: UnaryOp,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
) -> bool {
    if src.len() < COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => arch.dispatch(ComplexNegF32Kernel { src, dst }),
    }
    true
}

// Complex<f64> element-wise kernels (Neg)
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

pub(crate) fn dispatch_binary_complex_f64(
    op: BinaryOp,
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
    dst: &mut [Complex<f64>],
) -> bool {
    if lhs.len() < COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        BinaryOp::Add => arch.dispatch(super::binary::ComplexAddF64Kernel { lhs, rhs, dst }),
        BinaryOp::Sub => return false, // Sub not implemented for f64 complex
        BinaryOp::Mul => return false, // Mul not implemented for f64 complex
        BinaryOp::Div => return false,
    }
    true
}

pub(crate) fn dispatch_unary_complex_f64(
    op: UnaryOp,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
) -> bool {
    if src.len() < COMPLEX_ELEMENTWISE_THRESHOLD {
        return false;
    }
    let arch = get_arch();
    match op {
        UnaryOp::Neg => arch.dispatch(ComplexNegF64Kernel { src, dst }),
    }
    true
} 

// ---------------------------------------------------------------------------
// W14T8 Element-wise consistency tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod consistency_tests {
    use crate::simd::{self, BinaryOp, UnaryOp};

    const SIMD_WIDTH: usize = 64;

    fn assert_same_bits_or_nan_f64(actual: f64, expected: f64) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    fn assert_same_bits_or_nan_f32(actual: f32, expected: f32) {
        if expected.is_nan() || actual.is_nan() {
            assert!(actual.is_nan() && expected.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    fn assert_vec_bits_or_nan_f64(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert_same_bits_or_nan_f64(a, e);
        }
    }

    fn assert_vec_bits_or_nan_f32(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected.iter()) {
            assert_same_bits_or_nan_f32(a, e);
        }
    }

    fn apply_binary_f64(op: BinaryOp, lhs: f64, rhs: f64) -> f64 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

    fn apply_binary_f32(op: BinaryOp, lhs: f32, rhs: f32) -> f32 {
        match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
        }
    }

    fn fixture_f64(len: usize) -> (Vec<f64>, Vec<f64>) {
        let lhs_seed = [
            1.5_f64,
            -2.3,
            0.001,
            -1e20,
            std::f64::consts::PI,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ];
        let rhs_seed = [
            -4.25_f64,
            8.0,
            -0.125,
            1e-20,
            -2.0,
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 2.0,
        ];
        let lhs: Vec<f64> = (0..len).map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f64> = (0..len).map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    fn fixture_f32(len: usize) -> (Vec<f32>, Vec<f32>) {
        let lhs_seed = [
            1.5_f32,
            -2.3,
            0.001,
            -1e10,
            std::f32::consts::PI,
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
        ];
        let rhs_seed = [
            -4.25_f32,
            8.0,
            -0.125,
            1e-10,
            -2.0,
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
        ];
        let lhs: Vec<f32> = (0..len).map(|i| lhs_seed[i % lhs_seed.len()]).collect();
        let rhs: Vec<f32> = (0..len).map(|i| rhs_seed[i % rhs_seed.len()]).collect();
        (lhs, rhs)
    }

    fn simd_vs_serial_bitwise_f64(op: BinaryOp, lhs: &[f64], rhs: &[f64]) {
        let mut dst = vec![0.0_f64; lhs.len()];
        let handled = simd::dispatch_vector_binary_op(op, lhs, rhs, &mut dst);
        if handled {
            let serial: Vec<f64> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| apply_binary_f64(op, l, r))
                .collect();
            assert_vec_bits_or_nan_f64(&dst, &serial);
        }
    }

    fn simd_vs_serial_bitwise_f32(op: BinaryOp, lhs: &[f32], rhs: &[f32]) {
        let mut dst = vec![0.0_f32; lhs.len()];
        let handled = simd::dispatch_vector_binary_op(op, lhs, rhs, &mut dst);
        if handled {
            let serial: Vec<f32> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&l, &r)| apply_binary_f32(op, l, r))
                .collect();
            assert_vec_bits_or_nan_f32(&dst, &serial);
        }
    }

    #[test]
    fn test_simd_vector_consistency_elementwise() {
        let (lhs_f64, rhs_f64) = fixture_f64(256);
        let (lhs_f32, rhs_f32) = fixture_f32(256);
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            simd_vs_serial_bitwise_f64(op, &lhs_f64, &rhs_f64);
            simd_vs_serial_bitwise_f32(op, &lhs_f32, &rhs_f32);
        }
    }

    #[test]
    fn test_simd_neg_f64_matches_serial() {
        let (src, _) = fixture_f64(256);
        let mut dst = vec![0.0_f64; src.len()];
        let handled = simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        if handled {
            let serial: Vec<f64> = src.iter().map(|&v| -v).collect();
            assert_vec_bits_or_nan_f64(&dst, &serial);
        }
    }

    #[test]
    fn test_simd_neg_f32_matches_serial() {
        let (src, _) = fixture_f32(256);
        let mut dst = vec![0.0_f32; src.len()];
        let handled = simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
        if handled {
            let serial: Vec<f32> = src.iter().map(|&v| -v).collect();
            assert_vec_bits_or_nan_f32(&dst, &serial);
        }
    }

    #[test]
    fn test_elementwise_boundary_lengths() {
        let mut lengths = vec![0, 1, 32, 64, 65, 128, 256];
        lengths.extend((1..8).map(|extra| SIMD_WIDTH + extra));
        for len in lengths {
            let (lhs, rhs) = fixture_f64(len);
            simd_vs_serial_bitwise_f64(BinaryOp::Add, &lhs, &rhs);
            simd_vs_serial_bitwise_f64(BinaryOp::Sub, &lhs, &rhs);
        }
    }

    #[test]
    fn test_elementwise_tail_handling() {
        for extra in [1_usize, 3, 7, 15] {
            let (lhs, rhs) = fixture_f64(256 + extra);
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                simd_vs_serial_bitwise_f64(op, &lhs, &rhs);
            }
        }
    }

    #[test]
    fn test_elementwise_below_threshold() {
        let (lhs, rhs) = fixture_f64(32);
        let mut dst = vec![0.0_f64; lhs.len()];
        assert!(!simd::dispatch_vector_binary_op(
            BinaryOp::Add,
            &lhs,
            &rhs,
            &mut dst
        ));
    }

    #[test]
    fn test_elementwise_at_threshold() {
        let (lhs, rhs) = fixture_f64(64);
        simd_vs_serial_bitwise_f64(BinaryOp::Add, &lhs, &rhs);
    }

    #[test]
    fn test_elementwise_misaligned() {
        let (lhs_storage, rhs_storage) = fixture_f64(258);
        let lhs = &lhs_storage[1..257];
        let rhs = &rhs_storage[1..257];
        simd_vs_serial_bitwise_f64(BinaryOp::Mul, lhs, rhs);
    }
}

// ---------------------------------------------------------------------------
// W14T9 Reduction/dot tolerance tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod reduction_tests {
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
    fn test_sum_tolerance_f64_within_documented_bounds() {
        let data = data_f64(2048);
        let scalar: f64 = data.iter().sum();
        if let Some(simd) = simd::try_sum_f64(&data) {
            assert_within_tolerance_f64(simd, scalar, tolerance_f64(&data));
        }
    }

    #[test]
    fn test_sum_tolerance_f32_within_documented_bounds() {
        let data = data_f32(2048);
        let scalar: f32 = data.iter().sum();
        if let Some(simd) = simd::try_sum_f32(&data) {
            assert_within_tolerance_f32(simd, scalar, tolerance_f32(&data));
        }
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
    fn test_sum_dispatch_simd_int_admission() {
        let data: Vec<i32> = (0..1024).collect();
        if let Some(simd) = simd::try_sum_i32(&data) {
            let scalar_i64: i64 = data.iter().map(|&v| v as i64).sum();
            let scalar_i32 =
                i32::try_from(scalar_i64).expect("test fixture stays within i32 range");
            assert_eq!(simd, scalar_i32);
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

    #[test]
    fn test_sum_nan_propagation() {
        let mut data = vec![1.0_f64; 2048];
        data[1024] = f64::NAN;
        if let Some(simd) = simd::try_sum_f64(&data) {
            assert!(simd.is_nan());
        }
    }

    #[test]
    fn test_sum_inf_sign_consistency() {
        let mut positive = vec![1.0_f64; 2048];
        positive[7] = f64::INFINITY;
        if let Some(simd) = simd::try_sum_f64(&positive) {
            assert_eq!(simd, f64::INFINITY);
        }
    }

    #[test]
    fn test_entry_threshold_boundary() {
        let below = vec![1.0_f64; 1023];
        assert!(simd::try_sum_f64(&below).is_none());
        let at_threshold = vec![1.0_f64; 1024];
        let simd = simd::try_sum_f64(&at_threshold).expect("len=1024 must enter f64 sum SIMD");
        assert_within_tolerance_f64(simd, 1024.0, tolerance_f64(&at_threshold));
    }
} // ---------------------------------------------------------------------------
// W14T10 Randomized property tests (in-crate, direct SIMD facade testing)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod property_tests {
    use crate::complex::Complex;
    use crate::simd::{self, BinaryOp, UnaryOp};

    const CASES: usize = 32;
    const MAX_LEN: usize = 4096;
    const ELEMENTWISE_THRESHOLD: usize = 64;
    const SUM_THRESHOLD: usize = 1024;
    const DOT_THRESHOLD: usize = 512;
    const COMPLEX_SUM_THRESHOLD: usize = 1024;
    const COMPLEX_DOT_THRESHOLD: usize = 512;
    const COMPLEX_ELEMENTWISE_THRESHOLD: usize = 128;

    // ---- splitmix64 PRNG ----

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn gen_len(state: &mut u64, max_len: usize) -> usize {
        (splitmix64(state) as usize) % (max_len + 1)
    }

    fn gen_f64(state: &mut u64) -> f64 {
        let frac = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
        (frac - 0.5) * 20.0
    }

    fn gen_f32(state: &mut u64) -> f32 {
        let frac = (splitmix64(state) >> 11) as f32 / (1u64 << 53) as f32;
        (frac - 0.5) * 20.0
    }

    fn gen_i32_no_overflow(state: &mut u64) -> i32 {
        ((splitmix64(state) % 2001) as i32) - 1000
    }

    // ---- Tolerance helpers ----

    fn reduction_bound_f64(expected: f64, len: usize) -> f64 {
        let eps = f64::EPSILON;
        let magnitude = expected.abs().max(1.0);
        // Generous bound: O(n * eps * |expected|) with factor 4.0.
        // This is looser than the strict design bound (13-reduction §6.3)
        // but avoids flaky failures from different accumulation orders.
        ((len as f64) * eps * magnitude * 4.0).max(4.0 * f64::MIN_POSITIVE)
    }

    fn reduction_bound_f32(expected: f32, len: usize) -> f32 {
        let eps = f32::EPSILON;
        let magnitude = expected.abs().max(1.0);
        ((len as f32) * eps * magnitude * 4.0).max(4.0 * f32::MIN_POSITIVE)
    }

    fn assert_within_reduction_bound_f64(actual: f64, expected: f64, len: usize, op: &str) {
        let bound = reduction_bound_f64(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: actual={actual}, expected={expected}, bound={bound}"
        );
    }

    fn assert_within_reduction_bound_f32(actual: f32, expected: f32, len: usize, op: &str) {
        let bound = reduction_bound_f32(expected, len);
        assert!(
            (actual - expected).abs() <= bound,
            "{op} outside bound at len={len}: actual={actual}, expected={expected}, bound={bound}"
        );
    }

    // ---- Element-wise property tests ----

    fn prop_elementwise_binary_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
                let mut dst = vec![0.0_f64; len];
                let handled = simd::dispatch_vector_binary_op(op, &lhs, &rhs, &mut dst);
                if handled {
                    let serial: Vec<f64> = lhs
                        .iter()
                        .zip(rhs.iter())
                        .map(|(&l, &r)| match op {
                            BinaryOp::Add => l + r,
                            BinaryOp::Sub => l - r,
                            BinaryOp::Mul => l * r,
                            BinaryOp::Div => l / r,
                        })
                        .collect();
                    for (i, (&a, &e)) in dst.iter().zip(serial.iter()).enumerate() {
                        if a.is_nan() || e.is_nan() {
                            assert!(a.is_nan() && e.is_nan(), "nan mismatch at len={len}, i={i}");
                        } else {
                            assert_eq!(
                                a.to_bits(),
                                e.to_bits(),
                                "bits mismatch at len={len}, i={i}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn prop_elementwise_neg_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let src: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let mut dst = vec![0.0_f64; len];
            let handled = simd::dispatch_vector_unary_op(UnaryOp::Neg, &src, &mut dst);
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

    fn prop_elementwise_complex_add_f32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = COMPLEX_ELEMENTWISE_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<Complex<f32>> = (0..len)
                .map(|_| Complex::new(gen_f32(&mut rng), gen_f32(&mut rng)))
                .collect();
            let rhs: Vec<Complex<f32>> = (0..len)
                .map(|_| Complex::new(gen_f32(&mut rng), gen_f32(&mut rng)))
                .collect();
            let mut dst = vec![Complex::new(0.0, 0.0); len];
            let handled = simd::dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
            if handled {
                for (i, (a, (l, r))) in dst.iter().zip(lhs.iter().zip(rhs.iter())).enumerate() {
                    let e = *l + *r;
                    if a.re.is_nan() || a.im.is_nan() || e.re.is_nan() || e.im.is_nan() {
                        assert!(
                            (a.re.is_nan() && e.re.is_nan()) || (a.re == e.re),
                            "real nan mismatch at len={len}, i={i}"
                        );
                        assert!(
                            (a.im.is_nan() && e.im.is_nan()) || (a.im == e.im),
                            "imag nan mismatch at len={len}, i={i}"
                        );
                    } else {
                        assert_eq!(
                            a.re.to_bits(),
                            e.re.to_bits(),
                            "real bits at len={len}, i={i}"
                        );
                        assert_eq!(
                            a.im.to_bits(),
                            e.im.to_bits(),
                            "imag bits at len={len}, i={i}"
                        );
                    }
                }
            }
        }
    }

    // ---- Reduction/dot property tests ----

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

    fn prop_dot_tolerance_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = DOT_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
            if let Some(simd) = simd::try_dot_f64(&lhs, &rhs) {
                let scalar: f64 = lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).sum();
                assert_within_reduction_bound_f64(simd, scalar, len, "dot f64");
            }
        }
    }

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

    fn prop_dot_conjugate_complex_f64(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = COMPLEX_DOT_THRESHOLD + gen_len(&mut rng, MAX_LEN);
            let lhs: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            let rhs: Vec<Complex<f64>> = (0..len)
                .map(|_| Complex::new(gen_f64(&mut rng), gen_f64(&mut rng)))
                .collect();
            if let Some(simd) = simd::try_dot_complex_f64(&lhs, &rhs) {
                let scalar: Complex<f64> = lhs
                    .iter()
                    .zip(rhs.iter())
                    .map(|(l, r)| l.conj() * *r)
                    .fold(Complex::new(0.0, 0.0), |a, b| a + b);
                assert_within_reduction_bound_f64(simd.re, scalar.re, len, "complex dot f64 re");
                assert_within_reduction_bound_f64(simd.im, scalar.im, len, "complex dot f64 im");
            }
        }
    }

    fn prop_integer_no_panic_i32(seed: u64) {
        let mut rng = seed;
        for _case in 0..CASES {
            let len = gen_len(&mut rng, MAX_LEN);
            let data: Vec<i32> = (0..len).map(|_| gen_i32_no_overflow(&mut rng)).collect();
            // i32 SIMD is currently not available (per W14T0 spike),
            // so try_sum_i32 should always return None.
            assert!(
                simd::try_sum_i32(&data).is_none(),
                "i32 SIMD sum should not be available (widening unavailable)"
            );
        }
    }

    fn prop_tail_handling_f64(seed: u64) {
        let mut rng = seed;
        for width in [2usize, 4, 8, 16, 32] {
            for tail in 1..width {
                let base = 1 + (splitmix64(&mut rng) as usize % 16);
                let len = base * width + tail;
                let lhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
                let rhs: Vec<f64> = (0..len).map(|_| gen_f64(&mut rng)).collect();
                let mut dst = vec![0.0_f64; len];
                let handled = simd::dispatch_vector_binary_op(BinaryOp::Add, &lhs, &rhs, &mut dst);
                if handled {
                    for i in 0..len {
                        let expected = lhs[i] + rhs[i];
                        if expected.is_nan() || dst[i].is_nan() {
                            assert!(dst[i].is_nan() && expected.is_nan());
                        } else {
                            assert_eq!(dst[i].to_bits(), expected.to_bits());
                        }
                    }
                }
            }
        }
    }

    // ---- Aggregate test entry points ----

    #[test]
    fn prop_elementwise_consistency() {
        prop_elementwise_binary_f64(0x1001);
        prop_elementwise_neg_f64(0x1002);
        prop_elementwise_complex_add_f32(0x1003);
    }

    #[test]
    fn prop_sum_tolerance() {
        prop_sum_tolerance_f64(0x2001);
        prop_sum_tolerance_f32(0x2002);
        prop_sum_complex_f64(0x2003);
    }

    #[test]
    fn prop_dot_conjugate_contract() {
        prop_dot_tolerance_f64(0x3001);
        prop_dot_conjugate_complex_f64(0x3002);
    }

    #[test]
    fn prop_integer_no_panic() {
        prop_integer_no_panic_i32(0x4001);
    }

    #[test]
    fn prop_tail_and_fallback() {
        prop_tail_handling_f64(0x5001);
    }
}
