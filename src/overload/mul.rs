//! Owned `Tensor` Mul operator overloading (W23T7).
//!
//! Provides `Mul` implementations for pairs of
//! `TensorBase<Owned<A>, D>` with broadcast support, right-scalar,
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
// Mul — tensor × tensor (W23T7)
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
// Mul — right scalar (W23T7)
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
// Mul — Scalar<A> left (commutative) (W23T7)
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
// Mul — native left scalar per-type (W23T7)
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
// TensorView — MUL (W23T9–T10)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TensorView — MUL (W23T9–T10)
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Mul — TensorView × tensor (W23T9)
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
// Mul — TensorView right scalar (W23T10)
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
// Mul — TensorView Scalar<A> left (W23T10)
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
// Unit tests (W23T7, W23T9–T10)
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

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

    // ---- W23T9-T10: TensorView ----
    #[test]
    fn test_view_mul_right_and_left() {
        let t = Tensor::from_shape_vec([2], vec![2.0f64, 3.0]).expect("valid test input");
        let v = t.view();
        assert_eq!((&v * 4.0).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!((Scalar(4.0) * &v).as_slice().expect("c"), &[8.0, 12.0]);
        assert_eq!((4.0 * &v).as_slice().expect("c"), &[8.0, 12.0]);
    }
}
