//! Binary element-wise operations: arithmetic (add/sub/mul/div) and
//! the shared broadcast-aware traversal skeleton.
//!
//! Implemented by W16T2 (shared skeleton) and W16T6 (arithmetic methods).

use crate::broadcast::broadcast_shape;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;
use crate::dispatch::{ExecPath, select_exec_path};
use crate::element::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, Element, Numeric};
use crate::error::XenonError;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

// ============================================================================
// Private per-type dispatch trait for binary arithmetic
// ============================================================================

/// Per-type binary arithmetic step for add / sub / mul / div.
///
/// Design reference: 11-math §5.3 (signatures) and §10 (panic fields).
/// - Integer types (`i32`, `i64`): checked arithmetic via
///   `CheckedAdd` / `CheckedSub` / `CheckedMul` / `CheckedDiv`; overflow,
///   division-by-zero, `MIN / -1` → panic with diagnostic text per §10.
/// - Floating types (`f32`, `f64`): ordinary `+` / `-` / `*` / `/`;
///   NaN / Inf propagation by IEEE 754.
/// - Complex types (`Complex<f32>`, `Complex<f64>`): ordinary `+` / `-`
///   / `*` / `/` from the `Add` / `Sub` / `Mul` / `Div` supertraits on
///   `Numeric`; float-driven NaN propagation.
///
/// Sealed via `Numeric: Sealed` (03-element §5.2). Consumers cannot add
/// new impls; all six concrete `Numeric` types provide their own impl
/// in this file.
/// Per-type binary arithmetic dispatch trait.
///
/// W16T11 (Option D): with the `simd` feature enabled, `SimdElement` is
/// included as a supertrait so the compiler can resolve
/// `apply_arith_with_dispatch`'s `A: SimdElement` bound when the caller
/// has `A: BinaryArith` in scope. All six concrete impls (i32/i64/f32/
/// f64/Complex<f32>/Complex<f64>) already implement `SimdElement` per
/// W14T1 — adding it as a supertrait does not narrow the sealed set.
#[cfg(feature = "simd")]
pub(crate) trait BinaryArith: Numeric + crate::simd::SimdElement + 'static {
    /// Context-aware add step. `idx` / `shape` consumed by integer
    /// monomorphizations for panic diagnostics per 11-math §10.
    fn add_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn sub_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn mul_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn div_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
}

#[cfg(not(feature = "simd"))]
pub(crate) trait BinaryArith: Numeric + 'static {
    /// Context-aware add step. `idx` / `shape` consumed by integer
    /// monomorphizations for panic diagnostics per 11-math §10.
    fn add_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn sub_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn mul_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
    fn div_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self;
}

// ========== Integer impls (checked arithmetic with diagnostic panic) ==========

macro_rules! impl_binary_int {
    ($t:ty) => {
        impl BinaryArith for $t {
            #[inline]
            fn add_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
                <$t as CheckedAdd>::checked_add(a, b).unwrap_or_else(|| {
                    panic!(
                        "integer overflow: operation=add, type={}, trigger=overflow, \
                         lhs={}, rhs={}, element_index={}, shape={:?}",
                        stringify!($t),
                        a,
                        b,
                        idx,
                        shape
                    )
                })
            }
            #[inline]
            fn sub_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
                <$t as CheckedSub>::checked_sub(a, b).unwrap_or_else(|| {
                    panic!(
                        "integer overflow: operation=sub, type={}, trigger=overflow, \
                         lhs={}, rhs={}, element_index={}, shape={:?}",
                        stringify!($t),
                        a,
                        b,
                        idx,
                        shape
                    )
                })
            }
            #[inline]
            fn mul_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
                <$t as CheckedMul>::checked_mul(a, b).unwrap_or_else(|| {
                    panic!(
                        "integer overflow: operation=mul, type={}, trigger=overflow, \
                         lhs={}, rhs={}, element_index={}, shape={:?}",
                        stringify!($t),
                        a,
                        b,
                        idx,
                        shape
                    )
                })
            }
            #[inline]
            fn div_step(a: Self, b: Self, idx: usize, shape: &[usize]) -> Self {
                <$t as CheckedDiv>::checked_div(a, b).unwrap_or_else(|| {
                    let trigger = if b == 0 { "div_by_zero" } else { "overflow" };
                    panic!(
                        "integer arithmetic error: operation=div, type={}, trigger={}, \
                         lhs={}, rhs={}, element_index={}, shape={:?}",
                        stringify!($t),
                        trigger,
                        a,
                        b,
                        idx,
                        shape
                    )
                })
            }
        }
    };
}

impl_binary_int!(i32);
impl_binary_int!(i64);

// ========== Float impls (IEEE 754, never panic) ==========

macro_rules! impl_binary_float {
    ($t:ty) => {
        impl BinaryArith for $t {
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
    };
}

impl_binary_float!(f32);
impl_binary_float!(f64);

// ========== Complex impls ==========

macro_rules! impl_binary_complex {
    ($t:ty) => {
        impl BinaryArith for $t {
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
    };
}

impl_binary_complex!(crate::complex::Complex<f32>);
impl_binary_complex!(crate::complex::Complex<f64>);

// ============================================================================
// Broadcast-aware binary helper with index/shape context
// ============================================================================

/// Broadcast-aware binary traversal with element-index + shape context
/// propagated into the kernel closure — needed by integer panic diagnostics
/// per 11-math §10.
///
/// `O = A` for arithmetic; kept generic to support heterogeneous output type.
fn apply_binary_indexed<A, O, S1, S2, D1, D2, F>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    mut f: F,
) -> Result<Tensor<O, <D1 as BroadcastDim<D2>>::Output>, XenonError>
where
    A: crate::element::Element,
    O: crate::element::Element,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDim<D2>,
    D2: Dimension,
    F: FnMut(A, A, usize, &[usize]) -> O,
{
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_dim = <D1 as BroadcastDim<D2>>::Output::try_from_slice(out_shape.slice())
        .expect("broadcast_shape validated the output shape");
    let a_view = a.broadcast_to(out_dim.clone())?;
    let b_view = b.broadcast_to(out_dim.clone())?;
    let shape_slice: Vec<usize> = out_dim.slice().to_vec();
    let mut result = Tensor::<O, <D1 as BroadcastDim<D2>>::Output>::zeros(out_dim)?;
    for (idx, (dst, (a_val, b_val))) in result
        .iter_mut()
        .zip(a_view.iter().zip(b_view.iter()))
        .enumerate()
    {
        *dst = f(*a_val, *b_val, idx, &shape_slice);
    }
    Ok(result)
}

// ============================================================================
// Public arithmetic methods: tensor-tensor add/sub/mul/div
// ============================================================================

#[expect(
    private_bounds,
    reason = "BinaryArith is pub(crate) but this impl block publishes methods with a pub(crate)-only bound; sealed-trait pattern"
)]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: BinaryArith,
{
    /// Element-wise addition with broadcast.
    ///
    /// W16T11 Step 7: integer types (i32/i64) retain `apply_binary_indexed`
    /// for §10 `element_index` panic diagnostics. Float/complex types route
    /// through `apply_arith_with_dispatch` to access SIMD + parallel paths.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible (see `15-broadcast.md §6.2`).
    pub fn add<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_arith_with_dispatch(
                    self,
                    other,
                    |x, y| <A as BinaryArith>::add_step(x, y, 0, &[]),
                    Some(crate::simd::BinaryOp::Add),
                );
            }
        }
        apply_binary_indexed(self, other, |x, y, idx, shape| {
            <A as BinaryArith>::add_step(x, y, idx, shape)
        })
    }

    /// Element-wise subtraction with broadcast.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible (see `15-broadcast.md §6.2`).
    pub fn sub<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_arith_with_dispatch(
                    self,
                    other,
                    |x, y| <A as BinaryArith>::sub_step(x, y, 0, &[]),
                    Some(crate::simd::BinaryOp::Sub),
                );
            }
        }
        apply_binary_indexed(self, other, |x, y, idx, shape| {
            <A as BinaryArith>::sub_step(x, y, idx, shape)
        })
    }

    /// Element-wise multiplication with broadcast.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible (see `15-broadcast.md §6.2`).
    pub fn mul<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_arith_with_dispatch(
                    self,
                    other,
                    |x, y| <A as BinaryArith>::mul_step(x, y, 0, &[]),
                    Some(crate::simd::BinaryOp::Mul),
                );
            }
        }
        apply_binary_indexed(self, other, |x, y, idx, shape| {
            <A as BinaryArith>::mul_step(x, y, idx, shape)
        })
    }

    /// Element-wise division with broadcast.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible (see `15-broadcast.md §6.2`).
    pub fn div<S2, E>(
        &self,
        other: &TensorBase<S2, E>,
    ) -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
    {
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_arith_with_dispatch(
                    self,
                    other,
                    |x, y| <A as BinaryArith>::div_step(x, y, 0, &[]),
                    Some(crate::simd::BinaryOp::Div),
                );
            }
        }
        apply_binary_indexed(self, other, |x, y, idx, shape| {
            <A as BinaryArith>::div_step(x, y, idx, shape)
        })
    }
}

// ============================================================================
// Scalar arithmetic variants
// ============================================================================

#[expect(private_bounds, reason = "BinaryArith is a private sealed trait")]
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
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility")
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
        let other = Tensor::<A, Ix0>::from_scalar(scalar).expect("from_scalar never fails");
        self.sub(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility")
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
        let other = Tensor::<A, Ix0>::from_scalar(scalar).expect("from_scalar never fails");
        self.mul(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility")
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
        let other = Tensor::<A, Ix0>::from_scalar(scalar).expect("from_scalar never fails");
        self.div(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility")
    }

    /// Element-wise `scalar - element` (left-scalar subtraction).
    /// Internal helper for `19-overload.md §5` non-commutative left-scalar
    /// operator dispatch. NOT part of the public API surface.
    ///
    /// W16T11 Step 7: float/complex route through `apply_arith_with_dispatch`
    /// for SIMD; integers retain `apply_binary_indexed` for §10 diagnostics.
    pub(crate) fn sub_from_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar).expect("from_scalar never fails");
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_arith_with_dispatch(
                    &other,
                    self,
                    |x, y| <A as BinaryArith>::sub_step(x, y, 0, &[]),
                    Some(crate::simd::BinaryOp::Sub),
                )
                .expect(
                    "scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility",
                );
            }
        }
        // Swap operand order: compute `scalar - self` element-wise.
        apply_binary_indexed(&other, self, |x, y, idx, shape| {
            <A as BinaryArith>::sub_step(x, y, idx, shape)
        })
        .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility")
    }

    /// Element-wise `scalar / element` (left-scalar division).
    /// Internal helper for `19-overload.md §5`; NOT part of the public
    /// API surface.
    pub(crate) fn div_from_scalar(&self, scalar: A) -> Tensor<A, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar).expect("from_scalar never fails");
        #[cfg(feature = "simd")]
        {
            use core::any::TypeId;
            if TypeId::of::<A>() != TypeId::of::<i32>() && TypeId::of::<A>() != TypeId::of::<i64>()
            {
                return apply_arith_with_dispatch(
                    &other,
                    self,
                    |x, y| <A as BinaryArith>::div_step(x, y, 0, &[]),
                    Some(crate::simd::BinaryOp::Div),
                )
                .expect(
                    "scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility",
                );
            }
        }
        apply_binary_indexed(&other, self, |x, y, idx, shape| {
            <A as BinaryArith>::div_step(x, y, idx, shape)
        })
        .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> guarantees compatibility")
    }
}

// ============================================================================
// W16T11: Dispatch-aware helpers for arithmetic and comparison
// ============================================================================

/// Non-broadcasting binary traversal helper. Assumes `a` and `b` have
/// identical shapes (caller is responsible for `broadcast_to` upstream).
/// Used by dispatch helpers in their Serial and SIMD-fallback paths.
pub(in crate::math) fn apply_binary_scalar<A, O, S1, S2, D, F>(
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
    let mut result = Tensor::<O, D>::zeros(a.raw_dim()).expect("input dimension must be valid");
    for ((dst, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
        *dst = op(a_val, b_val);
    }
    result
}

/// Dispatch-aware broadcast comparison helper for W16T8/T9/T10.
/// Output is always `bool`. W14 does not expose comparison SIMD kernels,
/// so `ExecPath::Simd` falls through to the scalar loop.
///
/// Parallel path delegates to [`crate::parallel::par_zip_map`] (W15T3)
/// when the `parallel` feature is enabled and `select_exec_path` returns
/// `ExecPath::Parallel`. Otherwise falls back to scalar.
pub(in crate::math) fn apply_compare_with_dispatch<A, S1, S2, D1, D2, F>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    op: F,
) -> Result<Tensor<bool, <D1 as BroadcastDim<D2>>::Output>, XenonError>
where
    A: Element,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDim<D2>,
    D2: Dimension,
    F: Fn(A, A) -> bool + Copy + Send + Sync,
{
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_dim = <D1 as BroadcastDim<D2>>::Output::try_from_slice(out_shape.slice())
        .expect("broadcast_shape validated the output shape");

    let a_view = a.broadcast_to(out_dim.clone())?;
    let b_view = b.broadcast_to(out_dim.clone())?;

    let len = out_dim.checked_size().expect("broadcast_shape validated");
    let both_contiguous = a_view.is_f_contiguous() && b_view.is_f_contiguous();
    let both_aligned = a_view.is_aligned() && b_view.is_aligned();
    let (path, guard) = select_exec_path(len, both_contiguous, both_aligned);

    let result = match path {
        ExecPath::Serial | ExecPath::Simd => apply_binary_scalar(&a_view, &b_view, op),
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                crate::parallel::map::par_zip_map(a, b, &out_dim, &strat, g, |a, b| Ok(op(*a, *b)))?
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_binary_scalar(&a_view, &b_view, op)
            }
        },
    };
    Ok(result)
}
/// Dispatch-aware broadcast arithmetic helper for W16T6 float/complex
/// types (`add`/`sub`/`mul`/`div`). Homogeneous `A → A`. SIMD path is
/// available when the `simd` feature is enabled and W14 facade covers the
/// op. Integer types must NOT use this helper — they retain
/// `apply_binary_indexed` for §10 `element_index` diagnostics.
#[cfg(feature = "simd")]
pub(in crate::math) fn apply_arith_with_dispatch<A, S1, S2, D1, D2, F>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    op: F,
    op_tag: Option<crate::simd::BinaryOp>,
) -> Result<Tensor<A, <D1 as BroadcastDim<D2>>::Output>, XenonError>
where
    A: Element + crate::simd::SimdElement + 'static,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDim<D2>,
    D2: Dimension,
    F: Fn(A, A) -> A + Copy + Send + Sync,
{
    // §10 element_index carve-out: integer types must NOT enter this
    // helper — they would lose the per-element panic diagnostic context
    // required by 11-math §10 line 785. Callers in `binary.rs` arithmetic
    // methods gate on TypeId before invoking this helper; this assertion
    // catches accidental misuse during development.
    debug_assert!(
        core::any::TypeId::of::<A>() != core::any::TypeId::of::<i32>()
            && core::any::TypeId::of::<A>() != core::any::TypeId::of::<i64>(),
        "apply_arith_with_dispatch must not be called with integer types; \
         use apply_binary_indexed instead per 11-math §10 element_index \
         requirement"
    );

    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let out_dim = <D1 as BroadcastDim<D2>>::Output::try_from_slice(out_shape.slice())
        .expect("broadcast_shape validated");
    let a_view = a.broadcast_to(out_dim.clone())?;
    let b_view = b.broadcast_to(out_dim.clone())?;

    let len = out_dim.checked_size().expect("broadcast_shape validated");
    let both_contiguous = a_view.is_f_contiguous() && b_view.is_f_contiguous();
    let both_aligned = a_view.is_aligned() && b_view.is_aligned();
    let (path, guard) = select_exec_path(len, both_contiguous, both_aligned);

    let result = match path {
        ExecPath::Serial => apply_binary_scalar(&a_view, &b_view, op),
        ExecPath::Simd => try_simd_arith(&a_view, &b_view, op_tag)
            .unwrap_or_else(|| apply_binary_scalar(&a_view, &b_view, op)),
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard.expect("ExecPath::Parallel must carry a ParallelGuard");
                crate::parallel::map::par_zip_map(a, b, &out_dim, &strat, g, |a, b| Ok(op(*a, *b)))?
            }
            #[cfg(not(feature = "parallel"))]
            {
                apply_binary_scalar(&a_view, &b_view, op)
            }
        },
    };
    Ok(result)
}

/// Homogeneous arithmetic SIMD helper. Returns `None` if op_tag is None,
/// W14 kernel returned false, or views are non-contiguous.
#[cfg(feature = "simd")]
fn try_simd_arith<A, S1, S2, D>(
    a: &TensorBase<S1, D>,
    b: &TensorBase<S2, D>,
    op_tag: Option<crate::simd::BinaryOp>,
) -> Option<Tensor<A, D>>
where
    A: Element + crate::simd::SimdElement,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D: Dimension,
{
    let tag = op_tag?;
    let lhs_slice: &[A] = a.as_slice()?;
    let rhs_slice: &[A] = b.as_slice()?;
    let mut result = Tensor::<A, _>::zeros(a.raw_dim()).expect("input dimension must be valid");
    let dst: &mut [A] = result.as_mut_slice()?;
    if crate::simd::dispatch_vector_binary_op(tag, lhs_slice, rhs_slice, dst) {
        Some(result)
    } else {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};
    use crate::tensor::Tensor;

    #[test]
    fn test_add_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 2, 3]).expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![4, 5, 6]).expect("valid tensor shape");
        let c = a.add(&b).expect("broadcast succeeds in test");
        assert_eq!(*c.get(&[0]).expect("valid index"), 5);
        assert_eq!(*c.get(&[1]).expect("valid index"), 7);
        assert_eq!(*c.get(&[2]).expect("valid index"), 9);
    }

    #[test]
    fn test_add_f64() {
        let a =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.5, -1.5]).expect("valid tensor shape");
        let b =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![0.5, 2.5]).expect("valid tensor shape");
        let c = a.add(&b).expect("broadcast succeeds in test");
        assert!((*c.get(&[0]).expect("valid index") - 2.0).abs() < 1e-10);
        assert!((*c.get(&[1]).expect("valid index") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::<f64, Ix2>::from_shape_vec([3, 1], vec![1.0, 2.0, 3.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix2>::from_shape_vec([1, 4], vec![10.0, 20.0, 30.0, 40.0])
            .expect("valid tensor shape");
        let c = a.add(&b).expect("broadcast succeeds in test");
        assert_eq!(c.shape(), &[3, 4]);
        let val = c.get(&[0, 0]).expect("valid index");
        assert!((*val - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul_scalar() {
        let t = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 2.0, 3.0])
            .expect("valid tensor shape");
        let r = t.mul_scalar(2.5);
        assert!((*r.get(&[0]).expect("valid index") - 2.5).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 5.0).abs() < 1e-10);
        assert!((*r.get(&[2]).expect("valid index") - 7.5).abs() < 1e-10);
    }

    #[test]
    fn test_add_i32_overflow_panic() {
        let a =
            Tensor::<i32, Ix1>::from_shape_vec([1], vec![i32::MAX]).expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([1], vec![1]).expect("valid tensor shape");
        let result = std::panic::catch_unwind(|| a.add(&b));
        assert!(result.is_err(), "i32::MAX + 1 must panic");
    }

    // ── W16T11: Dispatch path consistency tests ──

    /// Cross-path consistency: the dispatch-wired `equal` method produces
    /// correct results regardless of which ExecPath is selected internally.
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

    /// The `apply_binary_scalar` helper (W16T11 Step 4) is used as the
    /// scalar fallback inside dispatch helpers. Validate it independently.
    #[test]
    fn test_apply_binary_scalar() {
        let a =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.0, 2.0]).expect("valid tensor shape");
        let b =
            Tensor::<f64, Ix1>::from_shape_vec([2], vec![3.0, 4.0]).expect("valid tensor shape");
        let r = apply_binary_scalar(&a, &b, |x, y| x + y);
        assert!((*r.get(&[0]).expect("valid index") - 4.0).abs() < 1e-10);
        assert!((*r.get(&[1]).expect("valid index") - 6.0).abs() < 1e-10);
    }

    /// W16T11 Step 11: validates that the SIMD-on path produces results
    /// identical (within tolerance) to the SIMD-off baseline. Runs only
    /// when the `simd` feature is enabled.
    ///
    /// The test does not call SIMD kernels directly. Instead it compares
    /// the public `add` method (which goes through `apply_arith_with_dispatch`)
    /// against a precomputed scalar reference vector, ensuring the
    /// dispatch wiring routes through the SIMD path without breaking
    /// semantics.
    #[cfg(feature = "simd")]
    #[test]
    fn test_add_simd_vs_scalar() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([256], (0..256).map(|x| x as f64).collect())
            .expect("valid tensor shape");
        let b =
            Tensor::<f64, Ix1>::from_shape_vec([256], (0..256).map(|x| (x * 2) as f64).collect())
                .expect("valid tensor shape");
        let expected: Vec<f64> = (0..256).map(|x| 3.0 * x as f64).collect();
        let result = a.add(&b).expect("broadcast succeeds in test");
        for (i, (got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "SIMD path mismatch at index {i}: got {got} expected {exp}",
            );
        }
    }

    /// W16T11 Step 11: cross-path consistency. SIMD-off (default) and
    /// SIMD-on builds must produce the same byte-level result for
    /// integer ops, per 11-math §10 line 787 ("path consistency").
    ///
    /// i32 add is exact (no float tolerance), so byte-level equality is
    /// required across Serial / SIMD / Parallel paths. Integer types
    /// always retain `apply_binary_indexed` per the §10 element_index
    /// carve-out (W16T11 Step 7).
    #[test]
    fn test_add_path_consistency_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([64], (0..64).collect())
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([64], (0..64).map(|x| x * 3).collect())
            .expect("valid tensor shape");
        let r = a.add(&b).expect("broadcast succeeds in test");
        for i in 0..64 {
            assert_eq!(*r.get(&[i]).expect("valid index"), i as i32 + i as i32 * 3);
        }
    }
}
