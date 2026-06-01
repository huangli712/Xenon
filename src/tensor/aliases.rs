//! Type alias definitions for the tensor module.
//!
//! Defines 36 type aliases in four categories: 4 primary aliases (`Tensor` /
//! `TensorView` / `TensorViewMut` / `ArcTensor`) plus 8 dimension-specialized
//! convenience aliases for each of the 4 storage modes, covering `Ix0`–`Ix6`
//! and `IxDyn`.
//!
//! Primary aliases do NOT provide a default for `D`; callers must always
//! specify the dimension explicitly or use the convenience aliases below.

use super::TensorBase;
use crate::dimension::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use crate::storage::{ArcRepr, Owned, ViewMutRepr, ViewRepr};

/// Owned tensor. The default workhorse type — owns its data buffer
/// and handles allocation/deallocation.
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;

/// Immutable borrowed view. Shares the underlying storage without
/// copying and enforces read-only access.
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<'a, A>, D>;

/// Mutable borrowed view. Grants exclusive write access to the
/// underlying storage for the borrow lifetime.
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<'a, A>, D>;

/// Reference-counted shared-ownership tensor. Multiple `ArcTensor`
/// instances can share the same underlying buffer.
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;

// ------ Dimension-specialized aliases for Tensor ----------------------------

/// 0-dimensional owned tensor (scalar container).
pub type Tensor0<A> = Tensor<A, Ix0>;

/// 1-dimensional owned tensor (vector).
pub type Tensor1<A> = Tensor<A, Ix1>;

/// 2-dimensional owned tensor (matrix).
pub type Tensor2<A> = Tensor<A, Ix2>;

/// 3-dimensional owned tensor.
pub type Tensor3<A> = Tensor<A, Ix3>;

/// 4-dimensional owned tensor.
pub type Tensor4<A> = Tensor<A, Ix4>;

/// 5-dimensional owned tensor.
pub type Tensor5<A> = Tensor<A, Ix5>;

/// 6-dimensional owned tensor.
pub type Tensor6<A> = Tensor<A, Ix6>;

/// Dynamically-dimensioned owned tensor.
pub type TensorD<A> = Tensor<A, IxDyn>;

// ------ Dimension-specialized aliases for TensorView ------------------------

/// 0-dimensional immutable view (scalar view).
pub type TensorView0<'a, A> = TensorView<'a, A, Ix0>;

/// 1-dimensional immutable view (vector view).
pub type TensorView1<'a, A> = TensorView<'a, A, Ix1>;

/// 2-dimensional immutable view (matrix view).
pub type TensorView2<'a, A> = TensorView<'a, A, Ix2>;

/// 3-dimensional immutable view.
pub type TensorView3<'a, A> = TensorView<'a, A, Ix3>;

/// 4-dimensional immutable view.
pub type TensorView4<'a, A> = TensorView<'a, A, Ix4>;

/// 5-dimensional immutable view.
pub type TensorView5<'a, A> = TensorView<'a, A, Ix5>;

/// 6-dimensional immutable view.
pub type TensorView6<'a, A> = TensorView<'a, A, Ix6>;

/// Dynamically-dimensioned immutable view.
pub type TensorViewD<'a, A> = TensorView<'a, A, IxDyn>;

// ------ Dimension-specialized aliases for TensorViewMut ---------------------

/// 0-dimensional mutable view (scalar mutable borrow).
pub type TensorViewMut0<'a, A> = TensorViewMut<'a, A, Ix0>;

/// 1-dimensional mutable view (vector mutable borrow).
pub type TensorViewMut1<'a, A> = TensorViewMut<'a, A, Ix1>;

/// 2-dimensional mutable view (matrix mutable borrow).
pub type TensorViewMut2<'a, A> = TensorViewMut<'a, A, Ix2>;

/// 3-dimensional mutable view.
pub type TensorViewMut3<'a, A> = TensorViewMut<'a, A, Ix3>;

/// 4-dimensional mutable view.
pub type TensorViewMut4<'a, A> = TensorViewMut<'a, A, Ix4>;

/// 5-dimensional mutable view.
pub type TensorViewMut5<'a, A> = TensorViewMut<'a, A, Ix5>;

/// 6-dimensional mutable view.
pub type TensorViewMut6<'a, A> = TensorViewMut<'a, A, Ix6>;

/// Dynamically-dimensioned mutable view.
pub type TensorViewMutD<'a, A> = TensorViewMut<'a, A, IxDyn>;

// ------ Dimension-specialized aliases for ArcTensor -------------------------

/// 0-dimensional reference-counted tensor (scalar, shared ownership).
pub type ArcTensor0<A> = ArcTensor<A, Ix0>;

/// 1-dimensional reference-counted tensor (vector, shared ownership).
pub type ArcTensor1<A> = ArcTensor<A, Ix1>;

/// 2-dimensional reference-counted tensor (matrix, shared ownership).
pub type ArcTensor2<A> = ArcTensor<A, Ix2>;

/// 3-dimensional reference-counted tensor.
pub type ArcTensor3<A> = ArcTensor<A, Ix3>;

/// 4-dimensional reference-counted tensor.
pub type ArcTensor4<A> = ArcTensor<A, Ix4>;

/// 5-dimensional reference-counted tensor.
pub type ArcTensor5<A> = ArcTensor<A, Ix5>;

/// 6-dimensional reference-counted tensor.
pub type ArcTensor6<A> = ArcTensor<A, Ix6>;

/// Dynamically-dimensioned reference-counted tensor.
pub type ArcTensorD<A> = ArcTensor<A, IxDyn>;
