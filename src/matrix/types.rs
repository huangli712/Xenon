//! Per-type dot-product accumulation step.
//!
//! Design reference: 12-matrix §5.1, §6.

use core::any::TypeId;
use core::mem::transmute_copy;

use crate::element::Numeric;

// ── Per-type accumulation step ──

/// Per-step dot-product accumulation with type-aware arithmetic semantics.
///
/// - `i32`, `i64`: checked multiply then checked add; integer overflow is
///   unrecoverable and panics with element context (index, shape, type).
/// - `f32`, `f64`, `Complex<f32>`, `Complex<f64>`: conjugate-linear
///   `acc + x.conjugate() * y` via `Numeric`, preserving IEEE 754 NaN / Inf
///   propagation. `conjugate()` is the identity for real types and the true
///   conjugate for complex types.
///
/// Mirrors the `TypeId`-dispatch pattern of
/// [`checked_add_step`](crate::reduction) so the per-type accumulation logic
/// stays off the public `dot` generic bound.
///
/// # Safety
///
/// The `unsafe` reads are sound because each is gated by
/// `TypeId::of::<A>() == TypeId::of::<I>()`, which proves layout identity
/// between `A` and the integer type `I`.
#[inline]
pub(crate) fn dot_step<A>(acc: A, x: A, y: A, index: usize, len: usize) -> A
where
    A: Numeric + Copy + 'static,
{
    if TypeId::of::<A>() == TypeId::of::<i32>() {
        // SAFETY: TypeId equality proves `A == i32`, so reading `&A` through
        // a `*const i32` is sound.
        let xi: i32 = unsafe { *(&x as *const A as *const i32) };
        let yi: i32 = unsafe { *(&y as *const A as *const i32) };
        let acci: i32 = unsafe { *(&acc as *const A as *const i32) };
        let product = xi.checked_mul(yi).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during multiplication at element {index} of shape [{len}] (type i32)"
            )
        });
        let sum = acci.checked_add(product).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during accumulation at element {index} of shape [{len}] (type i32)"
            )
        });
        // SAFETY: `A == i32`; reinterpreting `i32` as `A` is identity.
        return unsafe { transmute_copy::<i32, A>(&sum) };
    }
    if TypeId::of::<A>() == TypeId::of::<i64>() {
        // SAFETY: TypeId equality proves `A == i64`.
        let xi: i64 = unsafe { *(&x as *const A as *const i64) };
        let yi: i64 = unsafe { *(&y as *const A as *const i64) };
        let acci: i64 = unsafe { *(&acc as *const A as *const i64) };
        let product = xi.checked_mul(yi).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during multiplication at element {index} of shape [{len}] (type i64)"
            )
        });
        let sum = acci.checked_add(product).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during accumulation at element {index} of shape [{len}] (type i64)"
            )
        });
        // SAFETY: `A == i64`; reinterpreting `i64` as `A` is identity.
        return unsafe { transmute_copy::<i64, A>(&sum) };
    }
    // Float / complex path: conjugate-linear `acc + x.conjugate() * y` via
    // `Numeric: Add + Mul`. `conjugate()` is the identity for f32 / f64 and
    // the true conjugate for `Complex`, so this single expression covers all
    // non-integer supported types and preserves IEEE 754 NaN / Inf semantics.
    acc + x.conjugate() * y
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
