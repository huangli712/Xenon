//! Arithmetic operator overloading for `Tensor` / `TensorView`.
//!
//! Delegates element-wise computation to inherent methods on
//! [`TensorBase`] (`add`, `sub`, `mul`, `div`, `*_scalar`,
//! `*_from_scalar`). Operators produce new owned tensors; inputs are
//! never mutated.

use core::ops::{Add, Div, Mul, Sub};

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::{Owned, ViewRepr};
use crate::tensor::{Tensor, TensorBase};

/// Newtype wrapper for scalar values, enabling a generic left-scalar path.
///
/// Rust orphan rules forbid blanket impls such as
/// `impl<T> Add<TensorBase<...>> for T`. Concrete primitive left-hand sides
/// like `impl Add<Tensor<f32, D>> for f32` remain legal and are provided for
/// Xenon's supported scalar set in `W23T5`–`W23T8` / `W23T10`.
///
/// Exported via `xenon::overload::Scalar` only — intentionally excluded from
/// the prelude and top-level re-exports.
#[expect(
    missing_debug_implementations,
    reason = "Scalar is a trivial newtype wrapper; Debug is not part of its public contract per 19-overload §5.3"
)]
pub struct Scalar<A>(pub A);

// ==========================================================================
// Macros for native left-scalar per-type impls (avoids repetitive boilerplate)
// ==========================================================================

/// Emits all 6 native left-scalar type impls for owned `Tensor` given
/// operator trait `$trait` (Add/Sub/Mul/Div), method `$method` and
/// delegation `$delegate`.
macro_rules! native_left_scalar_owned_all {
    ($trait:ident, $method:ident, $delegate:ident) => {
        impl<D> $trait<TensorBase<Owned<f32>, D>> for f32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f32, D>;
            fn $method(self, rhs: TensorBase<Owned<f32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<&'a TensorBase<Owned<f32>, D>> for f32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f32, D>;
            fn $method(self, rhs: &'a TensorBase<Owned<f32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<D> $trait<TensorBase<Owned<f64>, D>> for f64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f64, D>;
            fn $method(self, rhs: TensorBase<Owned<f64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<&'a TensorBase<Owned<f64>, D>> for f64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f64, D>;
            fn $method(self, rhs: &'a TensorBase<Owned<f64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<D> $trait<TensorBase<Owned<i32>, D>> for i32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i32, D>;
            fn $method(self, rhs: TensorBase<Owned<i32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<&'a TensorBase<Owned<i32>, D>> for i32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i32, D>;
            fn $method(self, rhs: &'a TensorBase<Owned<i32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<D> $trait<TensorBase<Owned<i64>, D>> for i64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i64, D>;
            fn $method(self, rhs: TensorBase<Owned<i64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<&'a TensorBase<Owned<i64>, D>> for i64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i64, D>;
            fn $method(self, rhs: &'a TensorBase<Owned<i64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<D> $trait<TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f32>, D>;
            fn $method(self, rhs: TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<&'a TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f32>, D>;
            fn $method(self, rhs: &'a TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<D> $trait<TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f64>, D>;
            fn $method(self, rhs: TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<&'a TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f64>, D>;
            fn $method(self, rhs: &'a TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
    };
}

/// Emits all 6 native left-scalar type impls for `TensorView` given
/// operator trait `$trait`, method `$method` and delegation `$delegate`.
macro_rules! native_left_scalar_view_all {
    ($trait:ident, $method:ident, $delegate:ident) => {
        impl<'a, D> $trait<TensorBase<ViewRepr<'a, f32>, D>> for f32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f32, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, 'b, D> $trait<&'b TensorBase<ViewRepr<'a, f32>, D>> for f32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f32, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<TensorBase<ViewRepr<'a, f64>, D>> for f64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f64, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, 'b, D> $trait<&'b TensorBase<ViewRepr<'a, f64>, D>> for f64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<f64, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<TensorBase<ViewRepr<'a, i32>, D>> for i32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i32, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, 'b, D> $trait<&'b TensorBase<ViewRepr<'a, i32>, D>> for i32
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i32, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<TensorBase<ViewRepr<'a, i64>, D>> for i64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i64, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, 'b, D> $trait<&'b TensorBase<ViewRepr<'a, i64>, D>> for i64
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<i64, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f32>, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, 'b, D> $trait<&'b TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f32>, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, D> $trait<TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f64>, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
        impl<'a, 'b, D> $trait<&'b TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
        where
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<Complex<f64>, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
                rhs.$delegate(self)
            }
        }
    };
}

// ==========================================================================
// Scalar bound shorthand — both directions of BroadcastDim required by
// add_scalar / sub_scalar / mul_scalar / div_scalar and *_from_scalar.
// ==========================================================================
// Add — owned tensor × owned tensor (W23T3)
// ==========================================================================

impl<A, D, E> Add<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn add(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::add(&self, &rhs)
    }
}

// ==========================================================================
// Add — ref / mixed owned tensor combos (W23T4)
// ==========================================================================

// &Tensor + &Tensor
impl<'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn add(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::add(self, rhs)
    }
}

// Tensor + &Tensor
impl<'a, A, D, E> Add<&'a TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn add(self, rhs: &'a TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::add(&self, rhs)
    }
}

// &Tensor + Tensor
impl<A, D, E> Add<TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn add(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::add(self, &rhs)
    }
}

// ==========================================================================
// Add — right scalar: Tensor + A, &Tensor + A (W23T5)
// ==========================================================================

impl<A, D> Add<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<A, D> Add<A> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.add_scalar(rhs)
    }
}

// ==========================================================================
// Add — Scalar<A> left: Scalar<A> + Tensor, Scalar<A> + &Tensor (W23T5)
// ==========================================================================

impl<A, D> Add<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}

impl<'a, A, D> Add<&'a TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}

// ==========================================================================
// Add — native left scalar per-type (W23T5)
// ==========================================================================

native_left_scalar_owned_all!(Add, add, add_scalar);

// ==========================================================================
// Sub — tensor × tensor (W23T6)
// ==========================================================================

impl<A, D, E> Sub<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn sub(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::sub(&self, &rhs)
    }
}

impl<'b, A, D, E> Sub<&'b TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn sub(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::sub(self, rhs)
    }
}

impl<'a, A, D, E> Sub<&'a TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn sub(self, rhs: &'a TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::sub(&self, rhs)
    }
}

impl<A, D, E> Sub<TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn sub(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::sub(self, &rhs)
    }
}

// ==========================================================================
// Sub — right scalar (W23T6)
// ==========================================================================

impl<A, D> Sub<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn sub(self, rhs: A) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<A, D> Sub<A> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn sub(self, rhs: A) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

// ==========================================================================
// Sub — Scalar<A> left (non-commutative → sub_from_scalar) (W23T6)
// ==========================================================================

impl<A, D> Sub<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn sub(self, rhs: TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.sub_from_scalar(self.0)
    }
}

impl<'a, A, D> Sub<&'a TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn sub(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.sub_from_scalar(self.0)
    }
}

// ==========================================================================
// Sub — native left scalar per-type (W23T6)
// ==========================================================================

native_left_scalar_owned_all!(Sub, sub, sub_from_scalar);

// ==========================================================================
// Mul — tensor × tensor (W23T7)
// ==========================================================================

impl<A, D, E> Mul<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn mul(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::mul(&self, &rhs)
    }
}

impl<'b, A, D, E> Mul<&'b TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn mul(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::mul(self, rhs)
    }
}

impl<'a, A, D, E> Mul<&'a TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn mul(self, rhs: &'a TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::mul(&self, rhs)
    }
}

impl<A, D, E> Mul<TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn mul(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::mul(self, &rhs)
    }
}

// ==========================================================================
// Mul — right scalar (W23T7)
// ==========================================================================

impl<A, D> Mul<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn mul(self, rhs: A) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<A, D> Mul<A> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn mul(self, rhs: A) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

// ==========================================================================
// Mul — Scalar<A> left (commutative) (W23T7)
// ==========================================================================

impl<A, D> Mul<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn mul(self, rhs: TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

impl<'a, A, D> Mul<&'a TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn mul(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

// ==========================================================================
// Mul — native left scalar per-type (W23T7)
// ==========================================================================

native_left_scalar_owned_all!(Mul, mul, mul_scalar);

// ==========================================================================
// Div — tensor × tensor (W23T8)
// ==========================================================================

impl<A, D, E> Div<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn div(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::div(&self, &rhs)
    }
}

impl<'b, A, D, E> Div<&'b TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn div(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::div(self, rhs)
    }
}

impl<'a, A, D, E> Div<&'a TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn div(self, rhs: &'a TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::div(&self, rhs)
    }
}

impl<A, D, E> Div<TensorBase<Owned<A>, E>> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn div(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::div(self, &rhs)
    }
}

// ==========================================================================
// Div — right scalar (W23T8)
// ==========================================================================

impl<A, D> Div<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn div(self, rhs: A) -> Self::Output {
        self.div_scalar(rhs)
    }
}

impl<A, D> Div<A> for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn div(self, rhs: A) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// ==========================================================================
// Div — Scalar<A> left (non-commutative → div_from_scalar) (W23T8)
// ==========================================================================

impl<A, D> Div<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn div(self, rhs: TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.div_from_scalar(self.0)
    }
}

impl<'a, A, D> Div<&'a TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    fn div(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.div_from_scalar(self.0)
    }
}

// ==========================================================================
// Div — native left scalar per-type (W23T8)
// ==========================================================================

native_left_scalar_owned_all!(Div, div, div_from_scalar);

// ==========================================================================
// TensorView — tensor × tensor (W23T9)
// ==========================================================================

macro_rules! view_tensor_binop {
    ($trait:ident, $method:ident) => {
        // &TensorView + &TensorView
        impl<'a, 'b, A, D, E> $trait<&'b TensorBase<ViewRepr<'b, A>, E>>
            for &'a TensorBase<ViewRepr<'a, A>, D>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<E>,
            E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
        {
            type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

            fn $method(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
                TensorBase::$method(self, rhs)
            }
        }

        // &TensorView + &Tensor
        impl<'a, 'b, A, D, E> $trait<&'b TensorBase<Owned<A>, E>>
            for &'a TensorBase<ViewRepr<'a, A>, D>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<E>,
            E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
        {
            type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

            fn $method(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
                TensorBase::$method(self, rhs)
            }
        }

        // &Tensor + &TensorView
        impl<'a, 'b, A, D, E> $trait<&'b TensorBase<ViewRepr<'b, A>, E>>
            for &'a TensorBase<Owned<A>, D>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<E>,
            E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
        {
            type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

            fn $method(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
                TensorBase::$method(self, rhs)
            }
        }
    };
}

view_tensor_binop!(Add, add);
view_tensor_binop!(Sub, sub);
view_tensor_binop!(Mul, mul);
view_tensor_binop!(Div, div);

// ==========================================================================
// TensorView — right scalar (W23T10)
// ==========================================================================

macro_rules! view_right_scalar {
    ($trait:ident, $method:ident, $delegate:ident) => {
        impl<'a, A, D> $trait<A> for TensorBase<ViewRepr<'a, A>, D>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<A, D>;
            fn $method(self, rhs: A) -> Self::Output {
                self.$delegate(rhs)
            }
        }
        impl<'a, 'b, A, D> $trait<A> for &'b TensorBase<ViewRepr<'a, A>, D>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<A, D>;
            fn $method(self, rhs: A) -> Self::Output {
                self.$delegate(rhs)
            }
        }
    };
}

view_right_scalar!(Add, add, add_scalar);
view_right_scalar!(Sub, sub, sub_scalar);
view_right_scalar!(Mul, mul, mul_scalar);
view_right_scalar!(Div, div, div_scalar);

// ==========================================================================
// TensorView — Scalar<A> left (W23T10)
// ==========================================================================

macro_rules! view_scalar_left {
    ($trait:ident, $method:ident, $delegate:ident) => {
        impl<'a, A, D> $trait<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<A, D>;
            fn $method(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
                rhs.$delegate(self.0)
            }
        }
        impl<'a, 'b, A, D> $trait<&'b TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
        where
            A: BinaryArith,
            D: Dimension + BroadcastDim<Ix0, Output = D>,
            Ix0: BroadcastDim<D, Output = D>,
        {
            type Output = Tensor<A, D>;
            fn $method(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
                rhs.$delegate(self.0)
            }
        }
    };
}

view_scalar_left!(Add, add, add_scalar);
view_scalar_left!(Sub, sub, sub_from_scalar);
view_scalar_left!(Mul, mul, mul_scalar);
view_scalar_left!(Div, div, div_from_scalar);

// ==========================================================================
// TensorView — native left scalar per-type (W23T10)
// ==========================================================================

native_left_scalar_view_all!(Add, add, add_scalar);
native_left_scalar_view_all!(Sub, sub, sub_from_scalar);
native_left_scalar_view_all!(Mul, mul, mul_scalar);
native_left_scalar_view_all!(Div, div, div_from_scalar);

// ==========================================================================
// Unit tests (W23T2–W23T10)
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ---- W23T2: Scalar wrapper construction ----
    #[test]
    fn test_scalar_wrapper_construct() {
        let scalar = Scalar(2i32);
        assert_eq!(scalar.0, 2);
    }

    // ---- W23T3: Add same-shape ----
    #[test]
    fn test_add_same_shape() {
        let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (left + right).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("contiguous"), &[4, 6]);
    }

    // ---- W23T3: Add broadcast ----
    //
    // Xenon stores tensors in F-order (column-major) per
    // `00-coding.md §14 决策 1`. For shape=[2,3] data=[1,2,3,4,5,6]:
    //   col 0 = [1,2], col 1 = [3,4], col 2 = [5,6]
    //   logical matrix: [[1,3,5], [2,4,6]]
    // Broadcasting [3] data=[10,20,30] places right[j] at every (i, j):
    //   (0,0)=1+10=11, (1,0)=2+10=12,
    //   (0,1)=3+20=23, (1,1)=4+20=24,
    //   (0,2)=5+30=35, (1,2)=6+30=36
    // F-order memory layout: [11, 12, 23, 24, 35, 36].
    //
    // The W23T3 design doc Step 2 sample expects the C-order layout
    // [11,22,33,14,25,36]; that sample is inconsistent with Xenon's
    // F-order baseline and is corrected here to the F-order value.
    #[test]
    fn test_add_broadcast() {
        let left = Tensor::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
        let right = Tensor::from_shape_vec([3], vec![10, 20, 30]).expect("valid test input");
        let result = (left + right).expect("broadcast succeeds");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().expect("contiguous"),
            &[11, 12, 23, 24, 35, 36]
        );
    }

    // ---- W23T4: ref/mixed owned tensor Add combos ----
    #[test]
    fn test_add_ref_ref() {
        let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (&left + &right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[1, 2]);
        assert_eq!(right.as_slice().expect("c"), &[3, 4]);
        assert_eq!(result.as_slice().expect("c"), &[4, 6]);
    }

    #[test]
    fn test_add_owned_ref() {
        let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (left + &right).expect("broadcast succeeds");
        assert_eq!(right.as_slice().expect("c"), &[3, 4]);
        assert_eq!(result.as_slice().expect("c"), &[4, 6]);
    }

    #[test]
    fn test_add_ref_owned() {
        let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (&left + right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[1, 2]);
        assert_eq!(result.as_slice().expect("c"), &[4, 6]);
    }

    // ---- W23T5: Add scalar paths ----
    #[test]
    fn test_add_right_scalar() {
        let tensor = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
        assert_eq!((tensor + 5).as_slice().expect("c"), &[6, 7]);
    }

    #[test]
    fn test_scalar_wrapper_add_tensor() {
        let tensor = Tensor::from_shape_vec([2], vec![1.0, 2.0]).expect("valid test input");
        assert_eq!((Scalar(5.0) + tensor).as_slice().expect("c"), &[6.0, 7.0]);
    }

    #[test]
    fn test_scalar_wrapper_add_ref_tensor() {
        let tensor = Tensor::from_shape_vec([2], vec![1.0, 2.0]).expect("valid test input");
        assert_eq!((Scalar(5.0) + &tensor).as_slice().expect("c"), &[6.0, 7.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[1.0, 2.0]);
    }

    #[test]
    fn test_native_scalar_add_tensor_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
        assert_eq!((5.0f64 + tensor).as_slice().expect("c"), &[6.0, 7.0]);
    }

    #[test]
    fn test_native_scalar_add_ref_tensor_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
        assert_eq!((5.0f64 + &tensor).as_slice().expect("c"), &[6.0, 7.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[1.0, 2.0]);
    }

    #[test]
    fn test_native_scalar_add_tensor_i32() {
        let tensor = Tensor::from_shape_vec([2], vec![1i32, 2i32]).expect("valid test input");
        assert_eq!((5i32 + tensor).as_slice().expect("c"), &[6i32, 7i32]);
    }

    // ---- W23T6: Sub ----
    #[test]
    fn test_sub_basic() {
        let left = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        assert_eq!((left - right).expect("broadcast succeeds").as_slice().expect("c"), &[2, 3]);
    }

    #[test]
    fn test_sub_ref_ref() {
        let left = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (&left - &right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[5, 7]);
        assert_eq!(result.as_slice().expect("c"), &[2, 3]);
    }

    #[test]
    fn test_sub_right_scalar() {
        let tensor = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        assert_eq!((tensor - 2).as_slice().expect("c"), &[3, 5]);
    }

    #[test]
    fn test_sub_scalar_wrapper_left_noncommutative() {
        let tensor = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        assert_eq!((Scalar(10) - tensor).as_slice().expect("c"), &[5, 3]);
    }

    #[test]
    fn test_sub_native_left_scalar_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![5.0f64, 7.0]).expect("valid test input");
        assert_eq!((10.0f64 - tensor).as_slice().expect("c"), &[5.0, 3.0]);
    }

    // ---- W23T7: Mul ----
    #[test]
    fn test_mul_basic() {
        let left = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![4, 5]).expect("valid test input");
        assert_eq!((left * right).expect("broadcast succeeds").as_slice().expect("c"), &[8, 15]);
    }

    #[test]
    fn test_mul_ref_ref() {
        let left = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![4, 5]).expect("valid test input");
        let result = (&left * &right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[2, 3]);
        assert_eq!(result.as_slice().expect("c"), &[8, 15]);
    }

    #[test]
    fn test_mul_right_scalar() {
        let tensor = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        assert_eq!((tensor * 4).as_slice().expect("c"), &[8, 12]);
    }

    #[test]
    fn test_mul_scalar_wrapper_left_commutative() {
        let tensor = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        assert_eq!((Scalar(4) * tensor).as_slice().expect("c"), &[8, 12]);
    }

    #[test]
    fn test_mul_native_left_scalar_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        assert_eq!((4.0f64 * tensor).as_slice().expect("c"), &[8.0, 12.0]);
    }

    // ---- W23T8: Div ----
    #[test]
    fn test_div_basic() {
        let left = Tensor::from_shape_vec([2], vec![8.0, 9.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![2.0, 3.0]).expect("valid test input");
        assert_eq!((left / right).expect("broadcast succeeds").as_slice().expect("c"), &[4.0, 3.0]);
    }

    #[test]
    fn test_div_ref_ref() {
        let left = Tensor::from_shape_vec([2], vec![8.0, 9.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![2.0, 3.0]).expect("valid test input");
        let result = (&left / &right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[8.0, 9.0]);
        assert_eq!(result.as_slice().expect("c"), &[4.0, 3.0]);
    }

    #[test]
    fn test_div_right_scalar() {
        let tensor = Tensor::from_shape_vec([2], vec![8.0, 9.0]).expect("valid test input");
        assert_eq!((tensor / 2.0).as_slice().expect("c"), &[4.0, 4.5]);
    }

    #[test]
    fn test_div_scalar_wrapper_left_noncommutative() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0, 4.0]).expect("valid test input");
        assert_eq!((Scalar(8.0) / tensor).as_slice().expect("c"), &[4.0, 2.0]);
    }

    #[test]
    fn test_div_native_left_scalar_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
        assert_eq!((8.0f64 / tensor).as_slice().expect("c"), &[4.0, 2.0]);
    }

    // ---- W23T9: TensorView tensor×tensor ----
    #[test]
    fn test_add_view_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![5, 6, 7, 8]).expect("valid test input");
        let lv = left.view();
        let rv = right.view();
        let result = (&lv + &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[6, 8, 10, 12]);
        assert_eq!(left.as_slice().expect("c"), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_add_view_owned() {
        let left = Tensor::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![5, 6, 7, 8]).expect("valid test input");
        let lv = left.view();
        let result = (&lv + &right).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[6, 8, 10, 12]);
    }

    #[test]
    fn test_add_owned_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![5, 6, 7, 8]).expect("valid test input");
        let rv = right.view();
        let result = (&left + &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[6, 8, 10, 12]);
    }

    #[test]
    fn test_view_sub_mul_div_sanity() {
        let a = Tensor::from_shape_vec([2], vec![8.0f64, 9.0]).expect("valid test input");
        let b = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let av = a.view();
        let bv = b.view();
        assert_eq!((&av - &bv).expect("broadcast succeeds").as_slice().expect("c"), &[6.0, 6.0]);
        assert_eq!((&av * &bv).expect("broadcast succeeds").as_slice().expect("c"), &[16.0, 27.0]);
        assert_eq!((&av / &bv).expect("broadcast succeeds").as_slice().expect("c"), &[4.0, 3.0]);
    }

    // ---- W23T10: TensorView scalar ----
    #[test]
    fn test_view_add_right_scalar() {
        let t = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v + 5.0).as_slice().expect("c"), &[6.0, 7.0]);
    }

    #[test]
    fn test_view_scalar_wrapper_add_commutative() {
        let t = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(5.0) + &v).as_slice().expect("c"), &[6.0, 7.0]);
    }

    #[test]
    fn test_view_native_left_scalar_add_f64() {
        let t = Tensor::from_shape_vec([2], vec![1.0f64, 2.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((5.0 + &v).as_slice().expect("c"), &[6.0, 7.0]);
    }

    #[test]
    fn test_view_sub_left_scalar_noncommutative() {
        let t = Tensor::from_shape_vec([2], vec![3.0f64, 7.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(10.0) - &v).as_slice().expect("c"), &[7.0, 3.0]);
        assert_eq!((10.0 - &v).as_slice().expect("c"), &[7.0, 3.0]);
    }

    #[test]
    fn test_view_div_left_scalar_noncommutative() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(8.0) / &v).as_slice().expect("c"), &[4.0, 2.0]);
        assert_eq!((8.0 / &v).as_slice().expect("c"), &[4.0, 2.0]);
    }

    #[test]
    fn test_view_mul_right_and_left() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v * 4.0).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!((Scalar(4.0) * &v).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!((4.0 * &v).as_slice().expect("c"), &[8.0, 12.0]);
    }
}
