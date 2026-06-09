//! Owned `Tensor` Sub operator overloading (W23T6).
//!
//! Provides `Sub` implementations for pairs of
//! `TensorBase<Owned<A>, D>` with broadcast support, right-scalar,
//! `Scalar<A>` left-scalar, and native per-type left-scalar paths.

use core::ops::Sub;

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::Owned;
use crate::tensor::{Tensor, TensorBase};

use super::scalar::Scalar;

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

impl<D> Sub<TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<&'a TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn sub(self, rhs: &'a TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<D> Sub<TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<&'a TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn sub(self, rhs: &'a TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<D> Sub<TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn sub(self, rhs: TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<&'a TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn sub(self, rhs: &'a TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<D> Sub<TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn sub(self, rhs: TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<&'a TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn sub(self, rhs: &'a TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<D> Sub<TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn sub(self, rhs: TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<&'a TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn sub(self, rhs: &'a TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<D> Sub<TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn sub(self, rhs: TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}
impl<'a, D> Sub<&'a TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn sub(self, rhs: &'a TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}

// ==========================================================================
// Unit tests (W23T6)
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

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
}
