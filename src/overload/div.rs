//! `Tensor` and `TensorView` division operator overloading.
//!
//! Provides `Div` implementations for owned tensors, tensor views,
//! and their cross-combinations with broadcast support, right-scalar,
//! `Scalar<A>` left-scalar, and native per-type left-scalar paths.

use core::ops::Div;

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::{Owned, ViewRepr};
use crate::tensor::{Tensor, TensorBase};

use super::scalar::Scalar;

// ----------------------------------------------------------------------------
// Div — tensor × tensor
// ----------------------------------------------------------------------------

impl<A, D, E> Div<TensorBase<Owned<A>, E>> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `division` between two tensors with broadcasting.
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

    /// Performs element-wise `division` between two tensors with broadcasting. Both operands are borrowed.
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

    /// Performs element-wise `division` with broadcasting. Borrows the right operand, consumes the left.
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

    /// Performs element-wise `division` with broadcasting. Borrows the left operand, consumes the right.
    fn div(self, rhs: TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::div(self, &rhs)
    }
}

// ----------------------------------------------------------------------------
// Div — right scalar
// ----------------------------------------------------------------------------

impl<A, D> Div<A> for TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    /// Applies `division` of a scalar to each element of the tensor.
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

    /// Applies `division` of a scalar to each element of the tensor.
    fn div(self, rhs: A) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// ----------------------------------------------------------------------------
// Div — Scalar<A> left (non-commutative → div_from_scalar)
// ----------------------------------------------------------------------------

impl<A, D> Div<TensorBase<Owned<A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;

    /// Applies `division` with a [`Scalar`] value as the left operand to each element of the tensor.
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

    /// Applies `division` with a [`Scalar`] value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<A>, D>) -> Self::Output {
        rhs.div_from_scalar(self.0)
    }
}

// ----------------------------------------------------------------------------
// Div — native left scalar per-type
// ----------------------------------------------------------------------------

impl<D> Div<TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<'a, D> Div<&'a TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<D> Div<TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<'a, D> Div<&'a TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<D> Div<TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<'a, D> Div<&'a TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<D> Div<TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<'a, D> Div<&'a TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<D> Div<TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<'a, D> Div<&'a TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<D> Div<TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

impl<'a, D> Div<&'a TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor.
    fn div(self, rhs: &'a TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// TensorView — DIV
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Div — TensorView × tensor
// ----------------------------------------------------------------------------

impl<'a, 'b, A, D, E> Div<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &'a TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `division` between two tensor views with broadcasting. Both operands are borrowed.
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

    /// Performs element-wise `division` between two tensor views with broadcasting. Both operands are borrowed.
    fn div(self, rhs: &'b TensorBase<Owned<A>, E>) -> Self::Output {
        TensorBase::div(self, rhs)
    }
}

impl<'b, A, D, E> Div<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &TensorBase<Owned<A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<E>,
    E: Dimension + BroadcastDim<D, Output = <D as BroadcastDim<E>>::Output>,
{
    type Output = Result<Tensor<A, <D as BroadcastDim<E>>::Output>>;

    /// Performs element-wise `division` between two tensor views with broadcasting. Both operands are borrowed.
    fn div(self, rhs: &'b TensorBase<ViewRepr<'b, A>, E>) -> Self::Output {
        TensorBase::div(self, rhs)
    }
}

// ----------------------------------------------------------------------------
// Div — TensorView right scalar
// ----------------------------------------------------------------------------

impl<'a, A, D> Div<A> for TensorBase<ViewRepr<'a, A>, D>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    /// Applies `division` of a scalar to each element of the tensor view.
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
    /// Applies `division` of a scalar to each element of the tensor view.
    fn div(self, rhs: A) -> Self::Output {
        self.div_scalar(rhs)
    }
}

// ----------------------------------------------------------------------------
// Div — TensorView Scalar<A> left
// ----------------------------------------------------------------------------

impl<'a, A, D> Div<TensorBase<ViewRepr<'a, A>, D>> for Scalar<A>
where
    A: BinaryArith,
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<A, D>;
    /// Applies `division` with a [`Scalar`] value as the left operand to each element of the tensor view.
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
    /// Applies `division` with a [`Scalar`] value as the left operand to each element of the tensor view.
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, A>, D>) -> Self::Output {
        rhs.div_from_scalar(self.0)
    }
}

impl<'a, D> Div<TensorBase<ViewRepr<'a, f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
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
    /// Applies `division` with this scalar value as the left operand to each element of the tensor view.
    fn div(self, rhs: &'b TensorBase<ViewRepr<'a, Complex<f64>>, D>) -> Self::Output {
        rhs.div_from_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// Unit tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ---- Div ----
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
    fn test_div_owned_ref() {
        let left = Tensor::from_shape_vec([2], vec![8.0, 9.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![2.0, 3.0]).expect("valid test input");
        let result = (left / &right).expect("broadcast succeeds");
        assert_eq!(right.as_slice().expect("c"), &[2.0, 3.0]);
        assert_eq!(result.as_slice().expect("c"), &[4.0, 3.0]);
    }

    #[test]
    fn test_div_ref_owned() {
        let left = Tensor::from_shape_vec([2], vec![8.0, 9.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![2.0, 3.0]).expect("valid test input");
        let result = (&left / right).expect("broadcast succeeds");
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

    #[test]
    fn test_div_native_left_scalar_i32() {
        let tensor = Tensor::from_shape_vec([2], vec![2i32, 4]).expect("valid test input");
        assert_eq!((8i32 / tensor).as_slice().expect("c"), &[4i32, 2i32]);
    }

    // ---- TensorView ----
    #[test]
    fn test_view_div_left_scalar_noncommutative() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(8.0) / &v).as_slice().expect("c"), &[4.0, 2.0]);
        assert_eq!((8.0 / &v).as_slice().expect("c"), &[4.0, 2.0]);
    }

    #[test]
    fn test_div_right_scalar_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![8.0, 9.0]).expect("valid test input");
        assert_eq!((&tensor / 2.0).as_slice().expect("c"), &[4.0, 4.5]);
    }


    #[test]
    fn test_div_scalar_wrapper_left_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0, 4.0]).expect("valid test input");
        assert_eq!((Scalar(8.0) / &tensor).as_slice().expect("c"), &[4.0, 2.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[2.0, 4.0]);
    }


    #[test]
    fn test_div_native_left_scalar_f64_ref() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
        assert_eq!((8.0f64 / &tensor).as_slice().expect("c"), &[4.0, 2.0]);
        assert_eq!(tensor.as_slice().expect("c"), &[2.0f64, 4.0]);
    }


    #[test]
    fn test_div_native_left_scalar_f32() {
        let tensor = Tensor::from_shape_vec([2], vec![2.0, 4.0f32]).expect("valid test input");
        assert_eq!((8.0f32 / tensor).as_slice().expect("c"), &[4.0, 2.0f32]);
    }


    #[test]
    fn test_view_div_right_scalar() {
        let t = Tensor::from_shape_vec([2], vec![8.0f64, 9.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v / 2.0).as_slice().expect("c"), &[4.0, 4.5]);
    }


    #[test]
    fn test_view_div_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![8.0, 9.0, 10.0, 12.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![2.0, 3.0, 2.0, 4.0]).expect("valid test input");
        let lv = left.view();
        let rv = right.view();
        let result = (&lv / &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[4.0, 3.0, 5.0, 3.0]);
        assert_eq!(left.as_slice().expect("c"), &[8.0, 9.0, 10.0, 12.0]);
    }


    #[test]
    fn test_view_div_owned() {
        let left = Tensor::from_shape_vec([2, 2], vec![8.0, 9.0, 10.0, 12.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![2.0, 3.0, 2.0, 4.0]).expect("valid test input");
        let lv = left.view();
        let result = (&lv / &right).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[4.0, 3.0, 5.0, 3.0]);
    }


    #[test]
    fn test_view_owned_div_view() {
        let left = Tensor::from_shape_vec([2, 2], vec![8.0, 9.0, 10.0, 12.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([2, 2], vec![2.0, 3.0, 2.0, 4.0]).expect("valid test input");
        let rv = right.view();
        let result = (&left / &rv).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("c"), &[4.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn test_div_native_left_scalar_i64() {
        let tensor = Tensor::from_shape_vec([2], vec![2i64, 4]).expect("valid test input");
        assert_eq!((8i64 / tensor).as_slice().expect("c"), &[4i64, 2i64]);
    }

    #[test]
    fn test_div_broadcast() {
        let left = Tensor::from_shape_vec([2, 3], vec![20.0, 30.0, 40.0, 60.0, 60.0, 90.0]).expect("valid test input");
        let right = Tensor::from_shape_vec([3], vec![10.0, 20.0, 30.0]).expect("valid test input");
        let result = (left / right).expect("broadcast succeeds");
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.as_slice().expect("contiguous"), &[2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn test_view_div_native_left_scalar_f64() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((8.0 / &v).as_slice().expect("c"), &[4.0, 2.0]);
    }

    #[test]
    fn test_view_div_scalar_wrapper_left() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 4.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((Scalar(8.0) / &v).as_slice().expect("c"), &[4.0, 2.0]);
    }

    #[test]
    fn test_scalar_wrapper_construct() {
        let scalar = Scalar(2i32);
        assert_eq!(scalar.0, 2);
    }

}
