//! `Tensor` and `TensorView` subtraction operator overloading.
//!
//! Provides `Sub` implementations for owned tensors, tensor views,
//! and their cross-combinations with broadcast support, right-scalar,
//! `Scalar<A>` left-scalar, and native per-type left-scalar paths.

use core::ops::Sub;

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::{Owned, ViewRepr};
use crate::tensor::{Tensor, TensorBase};

use super::scalar::Scalar;

// ----------------------------------------------------------------------------
// Sub — tensor × tensor
// ----------------------------------------------------------------------------

impl<A, D, E> Sub<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `subtraction` between two tensors with broadcasting.
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

    /// Performs element-wise `subtraction` between two tensors with broadcasting. Both operands are borrowed.
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

    /// Performs element-wise `subtraction` with broadcasting. Borrows the right operand, consumes the left.
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

    /// Performs element-wise `subtraction` with broadcasting. Borrows the left operand, consumes the right.
    fn sub(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::sub(self, &rhs)
    }
}

// ----------------------------------------------------------------------------
// Sub — right scalar
// ----------------------------------------------------------------------------

impl<A, D> Sub<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    /// Applies `subtraction` of a scalar to each element of the tensor.
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

    /// Applies `subtraction` of a scalar to each element of the tensor.
    fn sub(self, rhs: A) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

// ----------------------------------------------------------------------------
// Sub — Scalar<A> left (non-commutative → sub_from_scalar)
// ----------------------------------------------------------------------------

impl<A, D> Sub<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    /// Applies `subtraction` with a [`Scalar`] value as the left operand to each element of the tensor.
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

    /// Applies `subtraction` with a [`Scalar`] value as the left operand to each element of the tensor.
    fn sub(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.sub_from_scalar(self.0)
    }
}

// ----------------------------------------------------------------------------
// Sub — native left scalar per-type
// ----------------------------------------------------------------------------

impl<D> Sub<TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor.
    fn sub(self, rhs: &'a TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// TensorView — SUB
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Sub — TensorView × tensor
// ----------------------------------------------------------------------------

impl<'a, 'b, A, D, E> Sub<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `subtraction` between two tensor views with broadcasting. Both operands are borrowed.
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

    /// Performs element-wise `subtraction` between two tensor views with broadcasting. Both operands are borrowed.
    fn sub(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::sub(self, rhs)
    }
}

impl<'b, A, D, E> Sub<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `subtraction` between two tensor views with broadcasting. Both operands are borrowed.
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::sub(self, rhs)
    }
}

// ----------------------------------------------------------------------------
// Sub — TensorView right scalar
// ----------------------------------------------------------------------------

impl<'a, A, D> Sub<A> for TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    /// Applies `subtraction` of a scalar to each element of the tensor view.
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
    /// Applies `subtraction` of a scalar to each element of the tensor view.
    fn sub(self, rhs: A) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

// ----------------------------------------------------------------------------
// Sub — TensorView Scalar<A> left
// ----------------------------------------------------------------------------

impl<'a, A, D> Sub<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    /// Applies `subtraction` with a [`Scalar`] value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with a [`Scalar`] value as the left operand to each element of the tensor view.
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.sub_from_scalar(self.0)
    }
}

impl<'a, D> Sub<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `subtraction` with this scalar value as the left operand to each element of the tensor view.
    fn sub(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.sub_from_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// Unit tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ---- Scalar ----

#[test]
    fn test_scalar_wrapper_construct() {
        let scalar = Scalar(2i32);
        assert_eq!(scalar.0, 2);
    }

    // ---- Scalar ----

    // ---- Owned — tensor ----

#[test]
    fn test_sub_basic() {
        let left = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        assert_eq!((left - right).expect("broadcast succeeds").as_slice().expect("c"), &[2, 3]);
    }

    // ---- Owned — tensor ----

#[test]
    fn test_sub_broadcast() {
        let left = Tensor::from_shape_vec([2, 3], vec![5, 6, 7, 8, 9, 10]).expect("valid test input");
        let right = Tensor::from_shape_vec([3], vec![1, 2, 3]).expect("valid test input");
        let result = (left - right).expect("broadcast succeeds");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.as_slice().expect("contiguous"), &[4, 5, 5, 6, 6, 7]);
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
    fn test_sub_owned_ref() {
        let left = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (left - &right).expect("broadcast succeeds");
        assert_eq!(right.as_slice().expect("c"), &[3, 4]);
        assert_eq!(result.as_slice().expect("c"), &[2, 3]);
    }

#[test]
    fn test_sub_ref_owned() {
        let left = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (&left - right).expect("broadcast succeeds");
        assert_eq!(left.as_slice().expect("c"), &[5, 7]);
        assert_eq!(result.as_slice().expect("c"), &[2, 3]);
    }

    // ---- Owned — right scalar ----

#[test]
    fn test_sub_right_scalar() {
        let tensor = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        assert_eq!((tensor - 2).as_slice().expect("c"), &[3, 5]);
    }

    // ---- Owned — scalar ----

#[test]
    fn test_sub_right_scalar_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        assert_eq!((&tensor - 2).as_slice().expect("c"), &[3, 5]);
    }

    // ---- Owned — Scalar left ----

#[test]
    fn test_sub_scalar_wrapper_left() {
        let tensor = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        assert_eq!((Scalar(10) - tensor).as_slice().expect("c"), &[5, 3]);
    }

#[test]
    fn test_sub_scalar_wrapper_left_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![5, 7]).expect("valid test input");
        assert_eq!((Scalar(10) - &tensor).as_slice().expect("c"), &[5, 3]);
        assert_eq!(tensor.as_slice().expect("c"), &[5, 7]);
    }

    // ---- Owned — per-type left scalar ----

#[test]
    fn test_sub_native_left_scalar_i32() {
        let tensor = Tensor::from_shape_vec([2], vec![5i32, 7]).expect("valid test input");
        assert_eq!((10i32 - tensor).as_slice().expect("c"), &[5i32, 3i32]);
    }

#[test]
    fn test_sub_native_left_scalar_i64() {
        let tensor = Tensor::from_shape_vec([2], vec![5i64, 7]).expect("valid test input");
        assert_eq!((10i64 - tensor).as_slice().expect("c"), &[5i64, 3i64]);
    }

#[test]
    fn test_sub_native_left_scalar_f32() {
        let tensor = Tensor::from_shape_vec([2], vec![5.0, 7.0f32]).expect("valid test input");
        assert_eq!((10.0f32 - tensor).as_slice().expect("c"), &[5.0, 3.0f32]);
    }

#[test]
    fn test_sub_native_left_scalar_f64() {
        let tensor = Tensor::from_shape_vec([2], vec![5.0f64, 7.0]).expect("valid test input");
        assert_eq!((10.0f64 - tensor).as_slice().expect("c"), &[5.0, 3.0]);
    }

#[test]
    fn test_sub_native_left_scalar_f64_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![5.0f64, 7.0]).expect("valid test input");
        assert_eq!((10.0f64 - &tensor).as_slice().expect("c"), &[5.0, 3.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[5.0f64, 7.0]);
    }

    // ---- View — tensor ----

#[test]
    fn test_view_sub_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![5, 6, 7, 8]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
        let lv = left.view();
        let rv = right.view();
        let result = (&lv - &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[4, 4, 4, 4]);
        assert_eq!(left.as_slice().expect("c"), &[5, 6, 7, 8]);
    }

    // ---- View — tensor ----

#[test]
    fn test_view_sub_owned() {
        let left = Tensor::from_shape_vec([2, 2], vec![5, 6, 7, 8]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
        let lv = left.view();
        let result = (&lv - &right).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[4, 4, 4, 4]);
    }

#[test]
    fn test_view_owned_sub_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![5, 6, 7, 8]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
        let rv = right.view();
        let result = (&left - &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[4, 4, 4, 4]);
    }

    // ---- View — right scalar ----

#[test]
    fn test_view_sub_right_scalar() {
        let t = Tensor::from_shape_vec([2], vec![5.0f64, 7.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v - 2.0).as_slice().expect("c"), &[3.0, 5.0]);
    }

    // ---- View — scalar ----

    // ---- View — Scalar left ----

#[test]
    fn test_view_sub_scalar_wrapper_left() {
        let t = Tensor::from_shape_vec([2], vec![3.0f64, 7.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(10.0) - &v).as_slice().expect("c"), &[7.0, 3.0]);
    }

    // ---- View — per-type left scalar ----

#[test]
    fn test_view_sub_native_left_scalar_f64() {
        let t = Tensor::from_shape_vec([2], vec![3.0f64, 7.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((10.0 - &v).as_slice().expect("c"), &[7.0, 3.0]);
    }

#[test]
    fn test_view_sub_combined() {
        let t = Tensor::from_shape_vec([2], vec![3.0f64, 7.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(10.0) - &v).as_slice().expect("c"), &[7.0, 3.0]);
        assert_eq!((10.0 - &v).as_slice().expect("c"), &[7.0, 3.0]);
    }
}
