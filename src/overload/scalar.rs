//! Shared types and macros for operator overloading.
//!
//! Contains the [`Scalar`] newtype wrapper and the `native_left_scalar_*`
//! macro families used by both [`super::owned`] and [`super::view`].

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
 pub(crate) use native_left_scalar_owned_all;
 
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
 pub(crate) use native_left_scalar_view_all;
