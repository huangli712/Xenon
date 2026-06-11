//! Binary element-wise operations: arithmetic (add/sub/mul/div) and
//! the shared broadcast-aware traversal skeleton.

use core::any::TypeId;

use crate::error::XenonError;
use crate::broadcast::broadcast_with;
use crate::dispatch::{ExecPath, select_exec_path};

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::element::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub};
use crate::element::{Element, Numeric, SimdElement};
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;

#[cfg(feature = "parallel")]
use crate::parallel::binary::par_zip_checked;

use super::types::BinaryOp;
#[cfg(feature = "simd")]
use super::driver::dispatch_vector_binary_op;

// ----------------------------------------------------------------------------
// Private per-type dispatch trait for binary arithmetic
// ----------------------------------------------------------------------------

/// Per-type binary arithmetic dispatch for add / sub / mul / div.
///
/// - Integer types (`i32`, `i64`): checked arithmetic via
///   `CheckedAdd` / `CheckedSub` / `CheckedMul` / `CheckedDiv`; overflow,
///   division-by-zero, and `MIN / -1` panic with diagnostic text that
///   includes the operation, type, operand values, element index, and
///   broadcast shape.
/// - Floating types (`f32`, `f64`): ordinary `+` / `-` / `*` / `/`;
///   NaN / Inf propagation by IEEE 754.
/// - Complex types (`Complex<f32>`, `Complex<f64>`): ordinary `+` / `-`
///   / `*` / `/` from the `Add` / `Sub` / `Mul` / `Div` supertraits on
///   `Numeric`; float-driven NaN propagation.
///
/// Sealed via `Numeric: Sealed`. Consumers cannot add new impls; all six
/// concrete `Numeric` types provide their own impl in this file.
///
/// `SimdElement` is included as a supertrait so the compiler can resolve
/// `apply_arith_with_dispatch`'s `A: SimdElement` bound when the caller
/// has `A: BinaryArith` in scope. Since `SimdElement` lives in
/// `crate::element` (ungated), this bound holds in every feature
/// configuration — only the SIMD kernels under `crate::simd` are gated
/// behind the `simd` feature. All six concrete impls (i32/i64/f32/f64/
/// Complex<f32>/Complex<f64>) already implement `SimdElement`, so adding
/// it as a supertrait does not narrow the sealed set.
pub(crate) trait BinaryArith: Numeric + SimdElement + 'static {
    /// Context-aware add step. `idx` and `shape` are consumed by integer
    /// monomorphizations for overflow panic diagnostics.
    fn add_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn sub_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn mul_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn div_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
}

// --------- Integer impls (checked arithmetic with diagnostic panic) ---------

/// `BinaryArith` for `i32`: checked arithmetic. Overflow, division by
/// zero, and `i32::MIN / -1` panic with operation, type, operand values,
/// element index, and broadcast shape in the message.
impl BinaryArith for i32 {
    #[inline]
    fn add_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i32 as CheckedAdd>::checked_add(a, b).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=add, type={}, trigger=overflow, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i32",
                a,
                b,
                idx,
                shape
            )
        })
    }
    
    #[inline]
    fn sub_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i32 as CheckedSub>::checked_sub(a, b).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=sub, type={}, trigger=overflow, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i32",
                a,
                b,
                idx,
                shape
            )
        })
    }
    
    #[inline]
    fn mul_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i32 as CheckedMul>::checked_mul(a, b).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=mul, type={}, trigger=overflow, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i32",
                a,
                b,
                idx,
                shape
            )
        })
    }
   
    #[inline]
    fn div_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i32 as CheckedDiv>::checked_div(a, b).unwrap_or_else(|| {
            let trigger = if b == 0 { "div_by_zero" } else { "overflow" };
            panic!(
                "integer arithmetic error: operation=div, type={}, trigger={}, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i32",
                trigger,
                a,
                b,
                idx,
                shape
            )
        })
    }
}

/// `BinaryArith` for `i64`: checked arithmetic. Overflow, division by
/// zero, and `i64::MIN / -1` panic with operation, type, operand values,
/// element index, and broadcast shape in the message.
impl BinaryArith for i64 {
    #[inline]
    fn add_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i64 as CheckedAdd>::checked_add(a, b).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=add, type={}, trigger=overflow, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i64",
                a,
                b,
                idx,
                shape
            )
        })
    }

    #[inline]
    fn sub_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i64 as CheckedSub>::checked_sub(a, b).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=sub, type={}, trigger=overflow, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i64",
                a,
                b,
                idx,
                shape
            )
        })
    }
    
    #[inline]
    fn mul_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i64 as CheckedMul>::checked_mul(a, b).unwrap_or_else(|| {
            panic!(
                "integer overflow: operation=mul, type={}, trigger=overflow, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i64",
                a,
                b,
                idx,
                shape
            )
        })
    }
   
    #[inline]
    fn div_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
        <i64 as CheckedDiv>::checked_div(a, b).unwrap_or_else(|| {
            let trigger = if b == 0 { "div_by_zero" } else { "overflow" };
            panic!(
                "integer arithmetic error: operation=div, type={}, trigger={}, \
                 lhs={}, rhs={}, element_index={}, shape={:?}",
                "i64",
                trigger,
                a,
                b,
                idx,
                shape
            )
        })
    }
}

// ------------------- Float impls (IEEE 754, never panic) --------------------

/// `BinaryArith` for `f32`: ordinary IEEE 754 `+` / `-` / `*` / `/`;
/// never panics. NaN and Inf propagate per IEEE 754.
impl BinaryArith for f32 {
    #[inline]
    fn add_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a + b
    }

    #[inline]
    fn sub_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a - b
    }
    
    #[inline]
    fn mul_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a * b
    }
    
    #[inline]
    fn div_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a / b
    }
}

/// `BinaryArith` for `f64`: ordinary IEEE 754 `+` / `-` / `*` / `/`;
/// never panics. NaN and Inf propagate per IEEE 754.
impl BinaryArith for f64 {
    #[inline]
    fn add_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a + b
    }
    
    #[inline]
    fn sub_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a - b
    }
    
    #[inline]
    fn mul_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a * b
    }
    
    #[inline]
    fn div_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a / b
    }
}

// ------------------------------ Complex impls -------------------------------

/// `BinaryArith` for `Complex<f32>`: ordinary `+` / `-` / `*` / `/`
/// inherited from the `Numeric` supertraits; never panics. NaN
/// propagation follows the underlying `f32` IEEE 754 semantics.
impl BinaryArith for Complex<f32> {
    #[inline]
    fn add_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a + b
    }
    
    #[inline]
    fn sub_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a - b
    }
    
    #[inline]
    fn mul_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a * b
    }
    
    #[inline]
    fn div_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a / b
    }
}

/// `BinaryArith` for `Complex<f64>`: ordinary `+` / `-` / `*` / `/`
/// inherited from the `Numeric` supertraits; never panics. NaN
/// propagation follows the underlying `f64` IEEE 754 semantics.
impl BinaryArith for Complex<f64> {
    #[inline]
    fn add_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a + b
    }
    
    #[inline]
    fn sub_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a - b
    }
    
    #[inline]
    fn mul_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a * b
    }
    
    #[inline]
    fn div_step(a: Self, b: Self, _i: usize, _s: &[usize]) -> Self {
        a / b
    }
}

// ----------------------------------------------------------------------------
// Public arithmetic methods: tensor-tensor add/sub/mul/div
// ----------------------------------------------------------------------------

#[expect(private_bounds)]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: BinaryArith,
{
    /// Element-wise addition with broadcast.
    ///
    /// Integer types (i32/i64) keep the serial checked path so overflow
    /// panics can report the offending element index and broadcast shape.
    /// Float/complex types route through `apply_binary_with_dispatch` to
    /// access the SIMD and parallel paths.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn add<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        apply_binary_with_dispatch(
            self,
            other,
            |x, y, idx, shape| <A as BinaryArith>::add_step(x, y, idx, shape),
            BinaryOp::Add,
        )
    }

    /// Element-wise subtraction with broadcast.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn sub<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        apply_binary_with_dispatch(
            self,
            other,
            |x, y, idx, shape| <A as BinaryArith>::sub_step(x, y, idx, shape),
            BinaryOp::Sub,
        )
    }

    /// Element-wise multiplication with broadcast.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn mul<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        apply_binary_with_dispatch(
            self,
            other,
            |x, y, idx, shape| <A as BinaryArith>::mul_step(x, y, idx, shape),
            BinaryOp::Mul,
        )
    }

    /// Element-wise division with broadcast.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn div<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        apply_binary_with_dispatch(
            self,
            other,
            |x, y, idx, shape| <A as BinaryArith>::div_step(x, y, idx, shape),
            BinaryOp::Div,
        )
    }
}

// ----------------------------------------------------------------------------
// Scalar arithmetic variants
// ----------------------------------------------------------------------------

#[expect(private_bounds)]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
    A: BinaryArith,
{
    /// Element-wise tensor + scalar.
    ///
    /// # Panics
    ///
    /// Panics if `Tensor::<A, Ix0>::from_scalar(scalar)` fails, or if the
    /// subsequent `self.add(&other)` broadcast fails. Neither is reachable in
    /// practice: `from_scalar` cannot fail for valid `Element` types, and the
    /// `BroadcastDim<Ix0, Output = D>` bound guarantees scalar-broadcast
    /// compatibility.
    pub fn add_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails for valid element types");
        // `BroadcastDim<Ix0, Output = D>` guarantees shape compatibility;
        // broadcast can never fail here.
        self.add(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0>\
                     guarantees compatibility")
    }

    /// Element-wise tensor - scalar.
    ///
    /// # Panics
    ///
    /// Panics if `Tensor::<A, Ix0>::from_scalar(scalar)` fails, or if the
    /// subsequent `self.sub(&other)` broadcast fails. Neither is reachable in
    /// practice: `from_scalar` cannot fail for valid `Element` types, and the
    /// `BroadcastDim<Ix0, Output = D>` bound guarantees scalar-broadcast
    /// compatibility.
    pub fn sub_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.sub(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }

    /// Element-wise tensor * scalar.
    ///
    /// # Panics
    ///
    /// Panics if `Tensor::<A, Ix0>::from_scalar(scalar)` fails, or if the
    /// subsequent `self.mul(&other)` broadcast fails. Neither is reachable in
    /// practice: `from_scalar` cannot fail for valid `Element` types, and the
    /// `BroadcastDim<Ix0, Output = D>` bound guarantees scalar-broadcast
    /// compatibility.
    pub fn mul_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.mul(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0>\
                     guarantees compatibility")
    }

    /// Element-wise tensor / scalar.
    ///
    /// # Panics
    ///
    /// Panics if `Tensor::<A, Ix0>::from_scalar(scalar)` fails, or if the
    /// subsequent `self.div(&other)` broadcast fails. Neither is reachable in
    /// practice: `from_scalar` cannot fail for valid `Element` types, and the
    /// `BroadcastDim<Ix0, Output = D>` bound guarantees scalar-broadcast
    /// compatibility.
    pub fn div_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.div(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0>\
                     guarantees compatibility")
    }

    /// Element-wise `scalar - element` (left-scalar subtraction).
    ///
    /// Internal helper for non-commutative left-scalar operator dispatch.
    /// NOT part of the public API surface. Routes through 
    /// `apply_binary_with_dispatch` with swapped operands; float/complex
    /// types reach the SIMD and parallel paths, integers keep their
    /// per-element panic diagnostics.
    pub(crate) fn sub_from_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        // Swap operand order: compute `scalar - self` element-wise.
        apply_binary_with_dispatch(
            &other,
            self,
            |x, y, idx, shape| <A as BinaryArith>::sub_step(x, y, idx, shape),
            BinaryOp::Sub,
        ).expect("scalar broadcast cannot fail: BroadcastDim<Ix0>\
                  guarantees compatibility")
    }

    /// Element-wise `scalar / element` (left-scalar division).
    ///
    /// Internal helper for non-commutative left-scalar operator dispatch;
    /// NOT part of the public API surface.
    pub(crate) fn div_from_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        // Swap operand order: compute `scalar / self` element-wise.
        apply_binary_with_dispatch(
            &other,
            self,
            |x, y, idx, shape| <A as BinaryArith>::div_step(x, y, idx, shape),
            BinaryOp::Div,
        ).expect("scalar broadcast cannot fail: BroadcastDim<Ix0>\
                  guarantees compatibility")
    }
}

// ----------------------------------------------------------------------------
// Broadcast-aware binary helper with index/shape context
// ----------------------------------------------------------------------------

/// Broadcast-aware binary traversal with element-index + shape context
/// propagated into the kernel closure — needed so integer overflow /
/// div-by-zero panics can report the offending element index and the
/// broadcast output shape. Homogeneous `A -> A` (integer arithmetic).
pub(crate) fn apply_binary_checked<A, S1, S2, D1, D2, F>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    mut f: F,
) -> Result<Tensor<A, <D1 as BroadcastDim<D2>>::Output>, XenonError>
where
    A: Element,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDim<D2>,
    D2: Dimension,
    F: FnMut(A, A, usize, &[usize]) -> A,
{
    let (a_view, b_view, out_dim) = broadcast_with(a, b)?;
    let shape_slice: Vec<usize> = out_dim.slice().to_vec();
    let mut result = Tensor::<A, <D1 as BroadcastDim<D2>>::Output>::zeros(out_dim)?;
    for (idx, (dst, (a_val, b_val))) in result
        .iter_mut()
        .zip(a_view.iter().zip(b_view.iter()))
        .enumerate()
    {
        *dst = f(*a_val, *b_val, idx, &shape_slice);
    }
    Ok(result)
}

// ----------------------------------------------------------------------------
// Dispatch-aware helpers for arithmetic
// ----------------------------------------------------------------------------

/// Non-broadcasting binary traversal helper. Assumes `a` and `b` have
/// identical shapes (caller is responsible for `broadcast_to` upstream).
/// Used by dispatch helpers in their Serial and SIMD-fallback paths.
pub(crate) fn apply_binary_serial<A, O, S1, S2, D, F>(
    a: &TensorBase<S1, D>,
    b: &TensorBase<S2, D>,
    mut op: F,
) -> Tensor<O, D>
where
    A: Element,
    O: Element,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D: Dimension,
    F: FnMut(A, A) -> O,
{
    let mut result = Tensor::<O, D>::zeros(a.raw_dim())
        .expect("input dimension must be valid");
    for ((dst, &a_val), &b_val) in result
        .iter_mut()
        .zip(a.iter()).zip(b.iter())
    {
        *dst = op(a_val, b_val);
    }
    result
}

/// Unified broadcast arithmetic dispatch for `add` / `sub` / `mul` / `div`.
///
/// - Integer types (`i32`, `i64`): serial checked traversal via
///   `apply_binary_checked`, carrying the per-element index and broadcast
///   shape so overflow / div-by-zero panics keep their diagnostic context.
/// - Float / complex types: routed through the Serial / SIMD / Parallel
///   execution path chosen by `select_exec_path`.
///
/// This helper is NOT gated on the `simd` feature — only the inner SIMD
/// kernel attempt is conditional. That keeps the Parallel path reachable
/// whenever `parallel` is enabled, independent of `simd`.
///
/// The `step` closure carries `(idx, &shape)` for the integer path; the
/// float path adapts it to a context-free kernel via `|x, y| step(x, y, 0,
/// &[])`, which is zero-cost since float / complex impls ignore those
/// parameters.
pub(crate) fn apply_binary_with_dispatch<A, S1, S2, D1, D2, F>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    step: F,
    op: BinaryOp,
) -> Result<Tensor<A, <D1 as BroadcastDim<D2>>::Output>, XenonError>
where
    A: Element + SimdElement + 'static,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDim<D2>,
    D2: Dimension,
    F: Fn(A, A, usize, &[usize]) -> A + Copy + Send + Sync,
{
    // Integer carve-out: i32 / i64 must keep the per-element panic
    // diagnostic context, so they take the serial checked path.
    if TypeId::of::<A>() == TypeId::of::<i32>()
        || TypeId::of::<A>() == TypeId::of::<i64>()
    {
        return apply_binary_checked(a, b, step);
    }
    // Float / complex: drop the index/shape context and route through the
    // Serial / SIMD / Parallel execution path chosen by `select_exec_path`.
    let scalar_op = move |x, y| step(x, y, 0, &[]);
    let (a_view, b_view, out_dim) = broadcast_with(a, b)?;
    let len = out_dim.checked_size().expect("broadcast_shape validated");
    let both_contiguous = a_view.is_f_contiguous() && b_view.is_f_contiguous();
    let both_aligned = a_view.is_aligned() && b_view.is_aligned();
    let (path, guard) = select_exec_path(len, both_contiguous, both_aligned);

    let result = match path {
        ExecPath::Serial => apply_binary_serial(&a_view, &b_view, scalar_op),
        ExecPath::Simd => {
            #[cfg(feature = "simd")]
            {
                try_simd_binary(&a_view, &b_view, op)
                    .unwrap_or_else(
                        || apply_binary_serial(&a_view, &b_view, scalar_op)
                    )
            }
            #[cfg(not(feature = "simd"))]
            {
                let _ = op;
                apply_binary_serial(&a_view, &b_view, scalar_op)
            }
        },
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard
                    .expect("ExecPath::Parallel must carry a ParallelGuard");
                par_zip_checked(
                    a,
                    b,
                    &out_dim,
                    &strat, g, |a, b| Ok(scalar_op(*a, *b))
                )?
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_binary_serial(&a_view, &b_view, scalar_op)
            }
        },
    };
    Ok(result)
}

/// Homogeneous arithmetic SIMD helper. Returns `None` if the SIMD kernel
/// reports it did not handle the op, or either view is non-contiguous.
#[cfg(feature = "simd")]
fn try_simd_binary<A, S1, S2, D>(
    a: &TensorBase<S1, D>,
    b: &TensorBase<S2, D>,
    op: BinaryOp,
) -> Option<Tensor<A, D>>
where
    A: Element + SimdElement,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D: Dimension,
{
    let lhs_slice: &[A] = a.as_slice()?;
    let rhs_slice: &[A] = b.as_slice()?;
    let mut result = Tensor::<A, _>::zeros(a.raw_dim())
        .expect("input dimension must be valid");
    let dst: &mut [A] = result.as_mut_slice()?;
    if dispatch_vector_binary_op(op, lhs_slice, rhs_slice, dst) {
        Some(result)
    } else {
        None
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;
    use crate::dimension::{Ix1, Ix2};
    use crate::tensor::Tensor;

    /// Element-wise `i32` addition over rank-1 tensors produces the
    /// pairwise sums.
    #[test]
    fn test_add_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 2, 3])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![4, 5, 6])
            .expect("valid tensor shape");
        let c = a.add(&b)
            .expect("broadcast succeeds in test");
        assert_eq!(*c.get(&[0]).expect("valid index"), 5);
        assert_eq!(*c.get(&[1]).expect("valid index"), 7);
        assert_eq!(*c.get(&[2]).expect("valid index"), 9);
    }

    /// Element-wise `f64` addition over rank-1 tensors produces the
    /// pairwise sums.
    #[test]
    fn test_add_f64() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.5, -1.5])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([2], vec![0.5, 2.5])
            .expect("valid tensor shape");
        let c = a.add(&b)
            .expect("broadcast succeeds in test");
        assert!((*c.get(&[0]).expect("valid index") - 2.0).abs() < 1e-10);
        assert!((*c.get(&[1]).expect("valid index") - 1.0).abs() < 1e-10);
    }

    /// Broadcasting rank-2 inputs of shapes `[3, 1]` and `[1, 4]` yields
    /// an output of shape `[3, 4]` with element-wise sums.
    #[test]
    fn test_add_broadcast() {
        let a = Tensor::<f64, Ix2>::from_shape_vec(
            [3, 1],
            vec![1.0, 2.0, 3.0]
        ).expect("valid tensor shape");
        let b = Tensor::<f64, Ix2>::from_shape_vec(
            [1, 4],
            vec![10.0, 20.0, 30.0, 40.0]
        ).expect("valid tensor shape");
        let c = a.add(&b)
            .expect("broadcast succeeds in test");
        assert_eq!(c.shape(), &[3, 4]);
        let val = c.get(&[0, 0]).expect("valid index");
        assert!((*val - 11.0).abs() < 1e-10);
    }

    /// `mul_scalar` multiplies every element of the tensor by the given
    /// scalar.
    #[test]
    fn test_mul_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec(
            [3],
            vec![1.0, 2.0, 3.0]
        ).expect("valid tensor shape");
        let r = t.mul_scalar(2.5);
        assert!((*r.get(&[0]).expect("valid index") - 2.5).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 5.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 7.5).abs() < 1e-10);
    }

    /// `i32::MAX + 1` triggers an integer-overflow panic during
    /// element-wise add.
    #[test]
    fn test_add_i32_overflow_panic() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MAX])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([1], vec![1])
            .expect("valid tensor shape");
        let result = catch_unwind(|| a.add(&b));
        assert!(result.is_err(), "i32::MAX + 1 must panic");
    }

    /// Cross-path consistency: the dispatch-wired `equal` method produces
    /// correct results regardless of which `ExecPath` is selected internally.
    #[test]
    fn test_dispatch_path_consistency_equal() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 2.0, 3.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 2.0, 4.0])
            .expect("valid tensor shape");
        // Uses apply_compare_with_dispatch internally → routes through
        // select_exec_path → Serial/Simd/Parallel → scalar.
        let result = a.equal(&b).expect("broadcast succeeds in test");
        assert!(*result.get(&[0]).expect("valid index"));
        assert!(*result.get(&[1]).expect("valid index"));
        assert!(!*result.get(&[2]).expect("valid index"));
    }

    /// `apply_binary_serial` is the scalar fallback used inside dispatch
    /// helpers; validate it independently produces element-wise sums.
    #[test]
    fn test_apply_binary_serial() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.0, 2.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([2], vec![3.0, 4.0])
            .expect("valid tensor shape");
        let r = apply_binary_serial(&a, &b, |x, y| x + y);
        assert!((*r.get(&[0]).expect("valid index") - 4.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 6.0).abs() < 1e-10);
    }

    /// The SIMD-on `add` path produces results matching a precomputed
    /// scalar reference (within `1e-10`). Runs only when the `simd`
    /// feature is enabled.
    ///
    /// The test does not call SIMD kernels directly. Instead it compares
    /// the public `add` method (which goes through
    /// `apply_arith_with_dispatch`) against a precomputed scalar
    /// reference vector, ensuring the dispatch wiring routes through
    /// the SIMD path without breaking semantics.
    #[cfg(feature = "simd")]
    #[test]
    fn test_add_simd_vs_scalar() {
        let a = Tensor::<f64, Ix1>::from_shape_vec(
            [256],
            (0..256).map(|x| x as f64).collect()
        ).expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec(
            [256],
            (0..256).map(|x| (x * 2) as f64).collect()
        ).expect("valid tensor shape");
        let expected: Vec<f64> = (0..256).map(|x| 3.0 * x as f64).collect();
        let result = a.add(&b).expect("broadcast succeeds in test");
        for (i, (got, &exp)) in result
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-10,
                "SIMD path mismatch at index {i}: got {got} expected {exp}",
            );
        }
    }

    /// Cross-path consistency for integer add. `i32` add is exact (no
    /// float tolerance), so byte-level equality is required across the
    /// Serial / SIMD / Parallel paths. Integer types always retain
    /// `apply_binary_checked` so `element_index` panic diagnostics are
    /// preserved.
    #[test]
    fn test_add_path_consistency_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec(
            [64],
            (0..64).collect()
        ).expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec(
            [64],
            (0..64).map(|x| x * 3).collect()
        ).expect("valid tensor shape");
        let r = a.add(&b).expect("broadcast succeeds in test");
        for i in 0..64 {
            assert_eq!(
                *r.get(&[i]).expect("valid index"),
                i as i32 + i as i32 * 3
            );
        }
    }

    /// Element-wise `f64` subtraction yields the pairwise differences.
    #[test]
    fn test_sub_f64() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([3], vec![5.0, 2.0, -1.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 4.0, -3.0])
            .expect("valid tensor shape");
        let c = a.sub(&b).expect("broadcast succeeds in test");
        assert!((*c.get(&[0]).expect("valid index") - 4.0).abs() < 1e-10);
        assert!((*c.get(&[1]).expect("valid index") + 2.0).abs() < 1e-10);
        assert!((*c.get(&[2]).expect("valid index") - 2.0).abs() < 1e-10);
    }

    /// Element-wise `f64` multiplication yields the pairwise products.
    #[test]
    fn test_mul_f64() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([3], vec![2.0, 3.0, -4.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([3], vec![5.0, -2.0, 0.5])
            .expect("valid tensor shape");
        let c = a.mul(&b).expect("broadcast succeeds in test");
        assert!((*c.get(&[0]).expect("valid index") - 10.0).abs() < 1e-10);
        assert!((*c.get(&[1]).expect("valid index") + 6.0).abs() < 1e-10);
        assert!((*c.get(&[2]).expect("valid index") + 2.0).abs() < 1e-10);
    }

    /// Element-wise `f64` division yields the pairwise quotients.
    #[test]
    fn test_div_f64() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([3], vec![10.0, 9.0, -8.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([3], vec![2.0, 3.0, -4.0])
            .expect("valid tensor shape");
        let c = a.div(&b).expect("broadcast succeeds in test");
        assert!((*c.get(&[0]).expect("valid index") - 5.0).abs() < 1e-10);
        assert!((*c.get(&[1]).expect("valid index") - 3.0).abs() < 1e-10);
        assert!((*c.get(&[2]).expect("valid index") - 2.0).abs() < 1e-10);
    }

    /// Element-wise `i32` subtraction (checked path) yields exact differences.
    #[test]
    fn test_sub_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![5, 2, -1])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 4, -3])
            .expect("valid tensor shape");
        let c = a.sub(&b).expect("broadcast succeeds in test");
        assert_eq!(*c.get(&[0]).expect("valid index"), 4);
        assert_eq!(*c.get(&[1]).expect("valid index"), -2);
        assert_eq!(*c.get(&[2]).expect("valid index"), 2);
    }

    /// Element-wise `i32` multiplication (checked path) yields exact products.
    #[test]
    fn test_mul_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![2, 3, -4])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![5, -2, 6])
            .expect("valid tensor shape");
        let c = a.mul(&b).expect("broadcast succeeds in test");
        assert_eq!(*c.get(&[0]).expect("valid index"), 10);
        assert_eq!(*c.get(&[1]).expect("valid index"), -6);
        assert_eq!(*c.get(&[2]).expect("valid index"), -24);
    }

    /// Element-wise `i32` division (checked path) truncates toward zero.
    #[test]
    fn test_div_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![10, 7, -9])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![2, 3, 2])
            .expect("valid tensor shape");
        let c = a.div(&b).expect("broadcast succeeds in test");
        assert_eq!(*c.get(&[0]).expect("valid index"), 5);
        assert_eq!(*c.get(&[1]).expect("valid index"), 2);
        assert_eq!(*c.get(&[2]).expect("valid index"), -4);
    }

    /// `add_scalar` adds the scalar to every element.
    #[test]
    fn test_add_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 2.0, 3.0])
            .expect("valid tensor shape");
        let r = t.add_scalar(10.0);
        assert!((*r.get(&[0]).expect("valid index") - 11.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 12.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 13.0).abs() < 1e-10);
    }

    /// `sub_scalar` subtracts the scalar from every element (`tensor - scalar`).
    #[test]
    fn test_sub_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![10.0, 20.0, 30.0])
            .expect("valid tensor shape");
        let r = t.sub_scalar(5.0);
        assert!((*r.get(&[0]).expect("valid index") - 5.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 15.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 25.0).abs() < 1e-10);
    }

    /// `div_scalar` divides every element by the scalar (`tensor / scalar`).
    #[test]
    fn test_div_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![10.0, 20.0, 30.0])
            .expect("valid tensor shape");
        let r = t.div_scalar(2.0);
        assert!((*r.get(&[0]).expect("valid index") - 5.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 10.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 15.0).abs() < 1e-10);
    }

    /// `sub_from_scalar` computes `scalar - element` (non-commutative left
    /// scalar). Verifies the operand order is NOT the same as `sub_scalar`.
    #[test]
    fn test_sub_from_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 4.0, 10.0])
            .expect("valid tensor shape");
        // 5 - [1, 4, 10] = [4, 1, -5]
        let r = t.sub_from_scalar(5.0);
        assert!((*r.get(&[0]).expect("valid index") - 4.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 1.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") + 5.0).abs() < 1e-10);
    }

    /// `div_from_scalar` computes `scalar / element` (non-commutative left
    /// scalar). Verifies the operand order is NOT the same as `div_scalar`.
    #[test]
    fn test_div_from_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![2.0, 4.0, 10.0])
            .expect("valid tensor shape");
        // 20 / [2, 4, 10] = [10, 5, 2]
        let r = t.div_from_scalar(20.0);
        assert!((*r.get(&[0]).expect("valid index") - 10.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 5.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 2.0).abs() < 1e-10);
    }

    /// `sub_from_scalar` on `i32` keeps checked-arithmetic semantics and the
    /// swapped operand order.
    #[test]
    fn test_sub_from_scalar_i32() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 4, 10])
            .expect("valid tensor shape");
        let r = t.sub_from_scalar(5);
        assert_eq!(*r.get(&[0]).expect("valid index"), 4);
        assert_eq!(*r.get(&[1]).expect("valid index"), 1);
        assert_eq!(*r.get(&[2]).expect("valid index"), -5);
    }

    /// `i32::MIN - 1` triggers an integer-overflow panic during element-wise
    /// subtraction.
    #[test]
    fn test_sub_i32_overflow_panic() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MIN])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([1], vec![1])
            .expect("valid tensor shape");
        let result = catch_unwind(|| a.sub(&b));
        assert!(result.is_err(), "i32::MIN - 1 must panic");
    }

    /// `i32::MAX * 2` triggers an integer-overflow panic during element-wise
    /// multiplication.
    #[test]
    fn test_mul_i32_overflow_panic() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MAX])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([1], vec![2])
            .expect("valid tensor shape");
        let result = catch_unwind(|| a.mul(&b));
        assert!(result.is_err(), "i32::MAX * 2 must panic");
    }

    /// Integer division by zero triggers a `div_by_zero` panic.
    #[test]
    fn test_div_i32_by_zero_panic() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([1], vec![5])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([1], vec![0])
            .expect("valid tensor shape");
        let result = catch_unwind(|| a.div(&b));
        assert!(result.is_err(), "i32 / 0 must panic");
    }

    /// `i32::MIN / -1` overflows the checked division and panics.
    #[test]
    fn test_div_i32_min_overflow_panic() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MIN])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([1], vec![-1])
            .expect("valid tensor shape");
        let result = catch_unwind(|| a.div(&b));
        assert!(result.is_err(), "i32::MIN / -1 must panic");
    }

    /// `Complex<f64>` add: `(1+2i) + (3+4i) = 4+6i`.
    #[test]
    fn test_complex_add() {
        let a = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(1.0, 2.0)]
        ).expect("valid tensor shape");
        let b = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(3.0, 4.0)]
        ).expect("valid tensor shape");
        let c = a.add(&b).expect("broadcast succeeds in test");
        let v = c.get(&[0]).expect("valid index");
        assert!((v.re() - 4.0).abs() < 1e-10);
        assert!((v.im() - 6.0).abs() < 1e-10);
    }

    /// `Complex<f64>` sub: `(5+6i) - (1+2i) = 4+4i`.
    #[test]
    fn test_complex_sub() {
        let a = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(5.0, 6.0)]
        ).expect("valid tensor shape");
        let b = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(1.0, 2.0)]
        ).expect("valid tensor shape");
        let c = a.sub(&b).expect("broadcast succeeds in test");
        let v = c.get(&[0]).expect("valid index");
        assert!((v.re() - 4.0).abs() < 1e-10);
        assert!((v.im() - 4.0).abs() < 1e-10);
    }

    /// `Complex<f64>` mul: `(1+2i) * (3+4i) = -5+10i`.
    #[test]
    fn test_complex_mul() {
        let a = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(1.0, 2.0)]
        ).expect("valid tensor shape");
        let b = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(3.0, 4.0)]
        ).expect("valid tensor shape");
        let c = a.mul(&b).expect("broadcast succeeds in test");
        let v = c.get(&[0]).expect("valid index");
        assert!((v.re() + 5.0).abs() < 1e-10);
        assert!((v.im() - 10.0).abs() < 1e-10);
    }

    /// `Complex<f64>` div by a real value: `(4+6i) / (2+0i) = 2+3i`.
    #[test]
    fn test_complex_div() {
        let a = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(4.0, 6.0)]
        ).expect("valid tensor shape");
        let b = Tensor::<Complex<f64>, Ix1>::from_shape_vec(
            [1],
            vec![Complex::new(2.0, 0.0)]
        ).expect("valid tensor shape");
        let c = a.div(&b).expect("broadcast succeeds in test");
        let v = c.get(&[0]).expect("valid index");
        assert!((v.re() - 2.0).abs() < 1e-10);
        assert!((v.im() - 3.0).abs() < 1e-10);
    }

    /// Float `add`/`sub`/`mul`/`div` produce identical results on the Parallel
    /// path (forced via a threshold of 1) as on the Serial path (parallel
    /// disabled via the 0 sentinel). Guards the inlined `ExecPath::Parallel`
    /// arm of `apply_binary_with_dispatch`.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_arith_parallel_matches_serial_f64() {
        use crate::dispatch::ThresholdTestGuard;
        use crate::dispatch::set_parallel_threshold;

        let a = Tensor::<f64, Ix1>::from_shape_vec(
            [128],
            (0..128).map(|x| x as f64 + 1.0).collect()
        ).expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec(
            [128],
            (0..128).map(|x| (x as f64) * 0.5 + 1.0).collect()
        ).expect("valid tensor shape");

        let _guard = ThresholdTestGuard::new();
        // Serial reference (parallel disabled by the 0 sentinel).
        set_parallel_threshold(0);
        let add_serial = a.add(&b).expect("broadcast succeeds in test");
        let sub_serial = a.sub(&b).expect("broadcast succeeds in test");
        let mul_serial = a.mul(&b).expect("broadcast succeeds in test");
        let div_serial = a.div(&b).expect("broadcast succeeds in test");
        // Force the parallel path (any len >= 1 routes to Parallel).
        set_parallel_threshold(1);
        let add_par = a.add(&b).expect("broadcast succeeds in test");
        let sub_par = a.sub(&b).expect("broadcast succeeds in test");
        let mul_par = a.mul(&b).expect("broadcast succeeds in test");
        let div_par = a.div(&b).expect("broadcast succeeds in test");

        for i in 0..128 {
            let ix = [i];
            assert_eq!(
                add_par.get(&ix).expect("valid index"),
                add_serial.get(&ix).expect("valid index"),
                "add parallel/serial mismatch at {i}"
            );
            assert_eq!(
                sub_par.get(&ix).expect("valid index"),
                sub_serial.get(&ix).expect("valid index"),
                "sub parallel/serial mismatch at {i}"
            );
            assert_eq!(
                mul_par.get(&ix).expect("valid index"),
                mul_serial.get(&ix).expect("valid index"),
                "mul parallel/serial mismatch at {i}"
            );
            assert_eq!(
                div_par.get(&ix).expect("valid index"),
                div_serial.get(&ix).expect("valid index"),
                "div parallel/serial mismatch at {i}"
            );
        }
    }

    /// Integer `add` keeps exact byte-level equality between the Parallel
    /// path (forced) and the Serial checked path.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_add_parallel_matches_serial_i32() {
        use crate::dispatch::ThresholdTestGuard;
        use crate::dispatch::set_parallel_threshold;

        let a = Tensor::<i32, Ix1>::from_shape_vec(
            [128],
            (0..128).collect()
        ).expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec(
            [128],
            (0..128).map(|x| x * 2).collect()
        ).expect("valid tensor shape");

        let _guard = ThresholdTestGuard::new();
        set_parallel_threshold(0);
        let serial = a.add(&b).expect("broadcast succeeds in test");
        set_parallel_threshold(1);
        let parallel = a.add(&b).expect("broadcast succeeds in test");
        for i in 0..128 {
            assert_eq!(
                parallel.get(&[i]).expect("valid index"),
                serial.get(&[i]).expect("valid index"),
                "i32 add parallel/serial mismatch at {i}"
            );
        }
    }
}
