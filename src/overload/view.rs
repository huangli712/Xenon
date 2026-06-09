//! `TensorView` operator overloading (W23T9–T10).
//!
//! Provides `Add`, `Sub`, `Mul`, `Div` implementations for pairs of
//! `TensorBase<ViewRepr<A>, D>` and cross-combinations with
//! `TensorBase<Owned<A>, D>`, as well as right-scalar, `Scalar<A>`
//! left-scalar, and native per-type left-scalar paths for views.

use core::ops::{Add, Div, Mul, Sub};

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::{Owned, ViewRepr};
use crate::tensor::{Tensor, TensorBase};

use super::scalar::Scalar;

// ==========================================================================
// Add — TensorView × tensor (W23T9)
// ==========================================================================

impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::add(self, rhs)
    }
}

impl<'a, 'b, A, D, E> Add<&'b TensorBase<Owned<A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn add(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::add(self, rhs)
    }
}

// ==========================================================================
// Sub — TensorView × tensor (W23T9)
// ==========================================================================

impl<'a, 'b, A, D, E> Sub<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn sub(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::sub(self, rhs)
    }
}

impl<'a, 'b, A, D, E> Sub<&'b TensorBase<Owned<A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D, E> Sub<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn sub(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::sub(self, rhs)
    }
}

// ==========================================================================
// Mul — TensorView × tensor (W23T9)
// ==========================================================================

impl<'a, 'b, A, D, E> Mul<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn mul(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::mul(self, rhs)
    }
}

impl<'a, 'b, A, D, E> Mul<&'b TensorBase<Owned<A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D, E> Mul<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn mul(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::mul(self, rhs)
    }
}

// ==========================================================================
// Div — TensorView × tensor (W23T9)
// ==========================================================================

impl<'a, 'b, A, D, E> Div<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn div(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::div(self, rhs)
    }
}

impl<'a, 'b, A, D, E> Div<&'b TensorBase<Owned<A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D, E> Div<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    fn div(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::div(self, rhs)
    }
}


// ==========================================================================
// Add — TensorView right scalar (W23T10)
// ==========================================================================

impl<'a, A, D> Add<A> for TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D> Add<A> for &'b TensorBase<ViewRepr<'a, A>, D>
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
// Sub — TensorView right scalar (W23T10)
// ==========================================================================

impl<'a, A, D> Sub<A> for TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D> Sub<A> for &'b TensorBase<ViewRepr<'a, A>, D>
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
// Mul — TensorView right scalar (W23T10)
// ==========================================================================

impl<'a, A, D> Mul<A> for TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D> Mul<A> for &'b TensorBase<ViewRepr<'a, A>, D>
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
// Div — TensorView right scalar (W23T10)
// ==========================================================================

impl<'a, A, D> Div<A> for TensorBase<ViewRepr<'a, A>, D>
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

impl<'a, 'b, A, D> Div<A> for &'b TensorBase<ViewRepr<'a, A>, D>
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
// Add — TensorView Scalar<A> left (W23T10)
// ==========================================================================

impl<'a, A, D> Add<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}

impl<'a, 'b, A, D> Add<&'b TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.add_scalar(self.0)
    }
}

// ==========================================================================
// Sub — TensorView Scalar<A> left (W23T10)
// ==========================================================================

impl<'a, A, D> Sub<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.sub_from_scalar(self.0)
    }
}

impl<'a, 'b, A, D> Sub<&'b TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.sub_from_scalar(self.0)
    }
}

// ==========================================================================
// Mul — TensorView Scalar<A> left (W23T10)
// ==========================================================================

impl<'a, A, D> Mul<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

impl<'a, 'b, A, D> Mul<&'b TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

// ==========================================================================
// Div — TensorView Scalar<A> left (W23T10)
// ==========================================================================

impl<'a, A, D> Div<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.div_from_scalar(self.0)
    }
}

impl<'a, 'b, A, D> Div<&'b TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.div_from_scalar(self.0)
    }
}


// ==========================================================================
// TensorView — native left scalar per-type (W23T10)
// ==========================================================================

impl<'a, D> Add<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, 'b, D> Add<&'b TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, 'b, D> Add<&'b TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, 'b, D> Add<&'b TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, 'b, D> Add<&'b TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, 'b, D> Add<&'b TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn add(self, rhs: TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, 'b, D> Add<&'b TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn add(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Sub<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, 'b, D> Sub<&'b TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, 'b, D> Sub<&'b TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, 'b, D> Sub<&'b TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, 'b, D> Sub<&'b TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, 'b, D> Sub<&'b TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn sub(self, rhs: TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, 'b, D> Sub<&'b TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Mul<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, 'b, D> Mul<&'b TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, D> Mul<TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, 'b, D> Mul<&'b TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, D> Mul<TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, 'b, D> Mul<&'b TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, D> Mul<TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, 'b, D> Mul<&'b TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, D> Mul<TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, 'b, D> Mul<&'b TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, D> Mul<TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn mul(self, rhs: TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, 'b, D> Mul<&'b TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
impl<'a, D> Div<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, 'b, D> Div<&'b TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, f32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, D> Div<TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, 'b, D> Div<&'b TensorBase<ViewRepr<'a, f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, f64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, D> Div<TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, 'b, D> Div<&'b TensorBase<ViewRepr<'a, i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, i32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, D> Div<TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, 'b, D> Div<&'b TensorBase<ViewRepr<'a, i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, i64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, D> Div<TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, 'b, D> Div<&'b TensorBase<ViewRepr<'a, Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f32>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, D> Div<TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn div(self, rhs: TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}
impl<'a, 'b, D> Div<&'b TensorBase<ViewRepr<'a, Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

// ==========================================================================
// Unit tests (W23T9–W23T10)
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

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
