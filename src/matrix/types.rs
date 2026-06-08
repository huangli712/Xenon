//! `DotAccumulate` trait — per-type dot-product accumulation step.
//!
//! Design reference: 12-matrix §5.1, §6.

use crate::complex::Complex;
use crate::element::{CheckedAdd, CheckedMul, Numeric};

// ── Per-type accumulation trait ──

/// Per-type dot-product accumulation step.
///
/// Each scalar type implements one fused multiply-add step.
/// Design reference: 12-matrix §5.1, §6.
pub trait DotAccumulate: Numeric + 'static {
    /// Compute one step of the dot-product accumulation.
    ///
    /// `acc` is the running accumulator, `x` and `y` are the current
    /// element pair, `index` and `len` are provided for diagnostic
    /// messages (e.g. overflow panics).
    fn dot_step(acc: Self, x: Self, y: Self, index: usize, len: usize) -> Self;
}

// ── Float / complex impls ──

impl DotAccumulate for f32 {
    #[inline]
    fn dot_step(acc: Self, x: Self, y: Self, _index: usize, _len: usize) -> Self {
        acc + x.conjugate() * y
    }
}

impl DotAccumulate for f64 {
    #[inline]
    fn dot_step(acc: Self, x: Self, y: Self, _index: usize, _len: usize) -> Self {
        acc + x.conjugate() * y
    }
}

impl DotAccumulate for Complex<f32> {
    #[inline]
    fn dot_step(acc: Self, x: Self, y: Self, _index: usize, _len: usize) -> Self {
        acc + x.conjugate() * y
    }
}

impl DotAccumulate for Complex<f64> {
    #[inline]
    fn dot_step(acc: Self, x: Self, y: Self, _index: usize, _len: usize) -> Self {
        acc + x.conjugate() * y
    }
}

// ── Integer impls ──

impl DotAccumulate for i32 {
    #[inline]
    fn dot_step(acc: Self, x: Self, y: Self, index: usize, len: usize) -> Self {
        let product = CheckedMul::checked_mul(x, y).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during multiplication at element {index} of shape [{len}] (type i32)"
            )
        });
        CheckedAdd::checked_add(acc, product).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during accumulation at element {index} of shape [{len}] (type i32)"
            )
        })
    }
}

impl DotAccumulate for i64 {
    #[inline]
    fn dot_step(acc: Self, x: Self, y: Self, index: usize, len: usize) -> Self {
        let product = CheckedMul::checked_mul(x, y).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during multiplication at element {index} of shape [{len}] (type i64)"
            )
        });
        CheckedAdd::checked_add(acc, product).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during accumulation at element {index} of shape [{len}] (type i64)"
            )
        })
    }
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use crate::dimension::Ix1;
    use crate::tensor::Tensor1;

    #[test]
    #[should_panic(
        expected = "dot: integer overflow during multiplication at element 0 of shape [1] (type i32)"
    )]
    fn test_dot_int_overflow_mul() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![i32::MAX]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![2_i32]).expect("valid construction");
        let _ = crate::matrix::dot(&a, &b).expect("valid construction");
    }

    #[test]
    #[should_panic(expected = "dot: integer overflow during accumulation at element")]
    fn test_dot_int_overflow_add() {
        let a = Tensor1::from_shape_vec(Ix1(3), vec![i32::MAX, 1, 1]).expect("valid construction");
        let b =
            Tensor1::from_shape_vec(Ix1(3), vec![1_i32, i32::MAX, 1]).expect("valid construction");
        let _ = crate::matrix::dot(&a, &b).expect("valid construction");
    }
}
