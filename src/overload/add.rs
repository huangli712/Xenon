//! Owned `Tensor` Add operator overloading.
//!
//! Provides `Add` implementations for pairs of
//! `TensorBase<Owned<A>, D>` with broadcast support, right-scalar,
//! `Scalar<A>` left-scalar, and native per-type left-scalar paths.

use core::ops::Add;

use crate::complex::Complex;
use crate::dimension::{BroadcastDim, Dimension, Ix0};
use crate::error::Result;
use crate::math::BinaryArith;
use crate::storage::{Owned, ViewRepr};
use crate::tensor::{Tensor, TensorBase};

use super::scalar::Scalar;

// ----------------------------------------------------------------------------
// Add — owned tensor × owned tensor
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Add — ref / mixed owned tensor combos
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Add — right scalar: Tensor + A, &Tensor + A
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Add — Scalar<A> left: Scalar<A> + Tensor, Scalar<A> + &Tensor
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Add — native left scalar per-type
// ----------------------------------------------------------------------------

impl<D> Add<TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn add(self, rhs: TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<&'a TensorBase<Owned<f32>, D>> for f32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f32, D>;
    fn add(self, rhs: &'a TensorBase<Owned<f32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<D> Add<TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn add(self, rhs: TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<&'a TensorBase<Owned<f64>, D>> for f64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<f64, D>;
    fn add(self, rhs: &'a TensorBase<Owned<f64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<D> Add<TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn add(self, rhs: TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<&'a TensorBase<Owned<i32>, D>> for i32
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i32, D>;
    fn add(self, rhs: &'a TensorBase<Owned<i32>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<D> Add<TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn add(self, rhs: TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<&'a TensorBase<Owned<i64>, D>> for i64
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<i64, D>;
    fn add(self, rhs: &'a TensorBase<Owned<i64>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<D> Add<TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn add(self, rhs: TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<&'a TensorBase<Owned<Complex<f32>>, D>> for Complex<f32>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f32>, D>;
    fn add(self, rhs: &'a TensorBase<Owned<Complex<f32>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<D> Add<TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn add(self, rhs: TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}
impl<'a, D> Add<&'a TensorBase<Owned<Complex<f64>>, D>> for Complex<f64>
where
    D: Dimension + BroadcastDim<Ix0, Output = D>,
    Ix0: BroadcastDim<D, Output = D>,
{
    type Output = Tensor<Complex<f64>, D>;
    fn add(self, rhs: &'a TensorBase<Owned<Complex<f64>>, D>) -> Self::Output {
        rhs.add_scalar(self)
    }
}

// ----------------------------------------------------------------------------
// TensorView — ADD
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Add — TensorView × tensor
// ----------------------------------------------------------------------------

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

impl<'b, A, D, E> Add<&'b TensorBase<ViewRepr<'b, A>, E>>
    for &TensorBase<Owned<A>, D>
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


// ----------------------------------------------------------------------------
// Add — TensorView right scalar
// ----------------------------------------------------------------------------

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


// ----------------------------------------------------------------------------
// Add — TensorView Scalar<A> left
// ----------------------------------------------------------------------------

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


// ----------------------------------------------------------------------------
// TensorView — native left scalar per-type
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Unit tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ---- Scalar wrapper construction ----
    #[test]
    fn test_scalar_wrapper_construct() {
        let scalar = Scalar(2i32);
        assert_eq!(scalar.0, 2);
    }

    // ---- Add same-shape ----
    #[test]
    fn test_add_same_shape() {
        let left = Tensor::from_shape_vec([2], vec![1, 2]).expect("valid test input");
        let right = Tensor::from_shape_vec([2], vec![3, 4]).expect("valid test input");
        let result = (left + right).expect("broadcast succeeds");
        assert_eq!(result.as_slice().expect("contiguous"), &[4, 6]);
    }

    // ---- Add broadcast ----
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
    // The original design doc sample expects the C-order layout
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

    // ---- ref/mixed owned tensor Add combos ----
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

    // ---- Add scalar paths ----
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

    // ---- TensorView ----
    // ---- TensorView tensor×tensor ----
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

    // ---- TensorView scalar ----
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
}
