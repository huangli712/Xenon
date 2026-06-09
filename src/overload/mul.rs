//! `Tensor` and `TensorView` multiplication operator overloading.
//!
//! Provides `Mul` implementations for owned tensors, tensor views,
//! and their cross-combinations with broadcast support, right-scalar,
//! `Scalar<A>` left-scalar, and native per-type left-scalar paths.

use core::ops::Mul;

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::{Owned, ViewRepr};
use crate::tensor::{Tensor, TensorBase};

use super::scalar::Scalar;

// ----------------------------------------------------------------------------
// Mul — tensor × tensor
// ----------------------------------------------------------------------------

impl<A, D, E> Mul<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `multiplication` between two tensors with broadcasting.
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

    /// Performs element-wise `multiplication` between two tensors with broadcasting. Both operands are borrowed.
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

    /// Performs element-wise `multiplication` with broadcasting. Borrows the right operand, consumes the left.
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

    /// Performs element-wise `multiplication` with broadcasting. Borrows the left operand, consumes the right.
    fn mul(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::mul(self, &rhs)
    }
}

// ----------------------------------------------------------------------------
// Mul — right scalar
// ----------------------------------------------------------------------------

impl<A, D> Mul<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    /// Applies `multiplication` of a scalar to each element of the tensor.
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

    /// Applies `multiplication` of a scalar to each element of the tensor.
    fn mul(self, rhs: A) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

// ----------------------------------------------------------------------------
// Mul — Scalar<A> left (commutative)
// ----------------------------------------------------------------------------

impl<A, D> Mul<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    /// Applies `multiplication` with a [`Scalar`] value as the left operand to each element of the tensor.
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

    /// Applies `multiplication` with a [`Scalar`] value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

// ----------------------------------------------------------------------------
// Mul — native left scalar per-type
// ----------------------------------------------------------------------------

impl<D> Mul<TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'a, D> Mul<&'a TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<D> Mul<TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'a, D> Mul<&'a TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<D> Mul<TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'a, D> Mul<&'a TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<D> Mul<TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'a, D> Mul<&'a TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<D> Mul<TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'a, D> Mul<&'a TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<D> Mul<TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'a, D> Mul<&'a TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor.
    fn mul(self, rhs: &'a TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// TensorView — MUL
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Mul — TensorView × tensor
// ----------------------------------------------------------------------------

impl<'a, 'b, A, D, E> Mul<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `multiplication` between two tensor views with broadcasting. Both operands are borrowed.
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

    /// Performs element-wise `multiplication` between two tensor views with broadcasting. Both operands are borrowed.
    fn mul(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::mul(self, rhs)
    }
}

impl<'b, A, D, E> Mul<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `multiplication` between two tensor views with broadcasting. Both operands are borrowed.
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::mul(self, rhs)
    }
}

// ----------------------------------------------------------------------------
// Mul — TensorView right scalar
// ----------------------------------------------------------------------------

impl<'a, A, D> Mul<A> for TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    /// Applies `multiplication` of a scalar to each element of the tensor view.
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
    /// Applies `multiplication` of a scalar to each element of the tensor view.
    fn mul(self, rhs: A) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

// ----------------------------------------------------------------------------
// Mul — TensorView Scalar<A> left
// ----------------------------------------------------------------------------

impl<'a, A, D> Mul<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    /// Applies `multiplication` with a [`Scalar`] value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with a [`Scalar`] value as the left operand to each element of the tensor view.
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.mul_scalar(self.0)
    }
}

impl<'a, D> Mul<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `multiplication` with this scalar value as the left operand to each element of the tensor view.
    fn mul(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// Unit tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ---- Scalar ------------------------------------------------------------

    #[test]
    fn test_scalar_wrapper_construct() {
        let scalar = Scalar(2i32);
        assert_eq!(scalar.0, 2);
    }

    // ---- Owned — tensor ----------------------------------------------------

    #[test]
    fn test_mul_basic() {
        let left = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![4, 5]).expect("valid test input");
        assert_eq!((left * right).expect("broadcast succeeds").as_slice().expect("c"), &[8, 15]);
    }

    #[test]
    fn test_mul_broadcast() {
        let left = Tensor::from_shape_vec([2, 3], vec![2, 3, 4, 5, 6, 7]).expect("valid test input");
        let right = Tensor::from_shape_vec([3], vec![10, 20, 30]).expect("valid test input");
        let result = (left * right).expect("broadcast succeeds");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.as_slice().expect("contiguous"), &[20, 30, 80, 100, 180, 210]);
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
    fn test_mul_owned_ref() {
        let left = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![4, 5]).expect("valid test input");
        let result = (left * &right).expect("broadcast succeeds");
        assert_eq!(right.as_slice().expect("c"), &[4, 5]);
        assert_eq!(result.as_slice().expect("c"), &[8, 15]);
    }

    #[test]
    fn test_mul_ref_owned() {
        let left = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![4, 5]).expect("valid test input");
        let result = (&left * right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[2, 3]);
        assert_eq!(result.as_slice().expect("c"), &[8, 15]);
    }

    // ---- Owned — right scalar ----------------------------------------------

    #[test]
    fn test_mul_right_scalar() {
        let tensor = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        assert_eq!((tensor * 4).as_slice().expect("c"), &[8, 12]);
    }

    #[test]
    fn test_mul_right_scalar_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        assert_eq!((&tensor * 4).as_slice().expect("c"), &[8, 12]);
    }

    // ---- Owned — Scalar left -----------------------------------------------

    #[test]
    fn test_mul_scalar_wrapper_left() {
        let tensor = Tensor::from_shape_vec([2], vec![2, 3]).expect("valid test input");
        assert_eq!((Scalar(4) * tensor).as_slice().expect("c"), &[8, 12]);
    }

    #[test]
    fn test_mul_scalar_wrapper_left_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        assert_eq!((Scalar(4.0) * &tensor).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[2.0f64, 3.0]);
    }

    // ---- Owned — per-type left scalar --------------------------------------

    #[test]
    fn test_mul_native_left_scalar_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        assert_eq!((4.0f64 * tensor).as_slice().expect("c"), &[8.0, 12.0]);
    }

    #[test]
    fn test_mul_native_left_scalar_f64_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        assert_eq!((4.0f64 * &tensor).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[2.0f64, 3.0]);
    }

    #[test]
    fn test_mul_native_left_scalar_i32() {
        let tensor = Tensor::from_shape_vec([2], vec![2i32, 3]).expect("valid test input");
        assert_eq!((4i32 * tensor).as_slice().expect("c"), &[8i32, 12i32]);
    }

    #[test]
    fn test_mul_native_left_scalar_i64() {
        let tensor = Tensor::from_shape_vec([2], vec![2i64, 3]).expect("valid test input");
        assert_eq!((4i64 * tensor).as_slice().expect("c"), &[8i64, 12i64]);
    }

    #[test]
    fn test_mul_native_left_scalar_f32() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0, 3.0f32]).expect("valid test input");
        assert_eq!((4.0f32 * tensor).as_slice().expect("c"), &[8.0f32, 12.0f32]);
    }

    // ---- View — tensor -----------------------------------------------------

    #[test]
    fn test_view_mul_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![2, 3, 4, 5]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![6, 7, 8, 9]).expect("valid test input");
        let lv = left.view();
        let rv = right.view();
        let result = (&lv * &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[12, 21, 32, 45]);
        assert_eq!(left.as_slice().expect("c"), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_view_mul_owned() {
        let left = Tensor::from_shape_vec([2, 2], vec![2, 3, 4, 5]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![6, 7, 8, 9]).expect("valid test input");
        let lv = left.view();
        let result = (&lv * &right).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[12, 21, 32, 45]);
    }

    #[test]
    fn test_view_owned_mul_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![2, 3, 4, 5]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![6, 7, 8, 9]).expect("valid test input");
        let rv = right.view();
        let result = (&left * &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[12, 21, 32, 45]);
    }

    // ---- View — right scalar -----------------------------------------------

    #[test]
    fn test_view_mul_right_scalar() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v * 4.0).as_slice().expect("c"), &[8.0, 12.0]);
    }

    // ---- View — Scalar left ------------------------------------------------

    #[test]
    fn test_view_mul_scalar_wrapper_left() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(4.0) * &v).as_slice().expect("c"), &[8.0, 12.0]);
    }

    // ---- View — per-type left scalar ---------------------------------------

    #[test]
    fn test_view_mul_native_left_scalar_f64() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((4.0 * &v).as_slice().expect("c"), &[8.0, 12.0]);
    }

    #[test]
    fn test_view_mul_combined() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v * 4.0).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!((Scalar(4.0) * &v).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!((4.0 * &v).as_slice().expect("c"), &[8.0, 12.0]);
    }
}
