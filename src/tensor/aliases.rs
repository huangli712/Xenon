//! Type alias definitions for the tensor module.
//!
//! Defines 36 type aliases in four categories per `07-tensor.md §5.2`:
//! 4 primary aliases (`Tensor` / `TensorView` / `TensorViewMut` / `ArcTensor`)
//! plus 8 dimension-specialized convenience aliases for each of the 4 storage
//! modes, covering `Ix0`–`Ix6` and `IxDyn`.
//!
//! Primary aliases do NOT provide a default for `D`; callers must always
//! specify the dimension explicitly or use the convenience aliases below.

#![allow(missing_docs)]

use super::TensorBase;
use crate::dimension::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use crate::storage::{ArcRepr, Owned, ViewMutRepr, ViewRepr};

pub type Tensor<A, D> = TensorBase<Owned<A>, D>;
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<'a, A>, D>;
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<'a, A>, D>;
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;

pub type Tensor0<A> = Tensor<A, Ix0>;
pub type Tensor1<A> = Tensor<A, Ix1>;
pub type Tensor2<A> = Tensor<A, Ix2>;
pub type Tensor3<A> = Tensor<A, Ix3>;
pub type Tensor4<A> = Tensor<A, Ix4>;
pub type Tensor5<A> = Tensor<A, Ix5>;
pub type Tensor6<A> = Tensor<A, Ix6>;
pub type TensorD<A> = Tensor<A, IxDyn>;

pub type TensorView0<'a, A> = TensorView<'a, A, Ix0>;
pub type TensorView1<'a, A> = TensorView<'a, A, Ix1>;
pub type TensorView2<'a, A> = TensorView<'a, A, Ix2>;
pub type TensorView3<'a, A> = TensorView<'a, A, Ix3>;
pub type TensorView4<'a, A> = TensorView<'a, A, Ix4>;
pub type TensorView5<'a, A> = TensorView<'a, A, Ix5>;
pub type TensorView6<'a, A> = TensorView<'a, A, Ix6>;
pub type TensorViewD<'a, A> = TensorView<'a, A, IxDyn>;

pub type TensorViewMut0<'a, A> = TensorViewMut<'a, A, Ix0>;
pub type TensorViewMut1<'a, A> = TensorViewMut<'a, A, Ix1>;
pub type TensorViewMut2<'a, A> = TensorViewMut<'a, A, Ix2>;
pub type TensorViewMut3<'a, A> = TensorViewMut<'a, A, Ix3>;
pub type TensorViewMut4<'a, A> = TensorViewMut<'a, A, Ix4>;
pub type TensorViewMut5<'a, A> = TensorViewMut<'a, A, Ix5>;
pub type TensorViewMut6<'a, A> = TensorViewMut<'a, A, Ix6>;
pub type TensorViewMutD<'a, A> = TensorViewMut<'a, A, IxDyn>;

pub type ArcTensor0<A> = ArcTensor<A, Ix0>;
pub type ArcTensor1<A> = ArcTensor<A, Ix1>;
pub type ArcTensor2<A> = ArcTensor<A, Ix2>;
pub type ArcTensor3<A> = ArcTensor<A, Ix3>;
pub type ArcTensor4<A> = ArcTensor<A, Ix4>;
pub type ArcTensor5<A> = ArcTensor<A, Ix5>;
pub type ArcTensor6<A> = ArcTensor<A, Ix6>;
pub type ArcTensorD<A> = ArcTensor<A, IxDyn>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_aliases_compile() {
        let _: Option<Tensor<f64, Ix2>> = None;
        let _: Option<TensorView<'_, f64, Ix2>> = None;
        let _: Option<TensorViewMut<'_, f64, Ix2>> = None;
        let _: Option<ArcTensor<f64, Ix2>> = None;

        let _: Option<Tensor0<f64>> = None;
        let _: Option<Tensor1<f64>> = None;
        let _: Option<Tensor2<f64>> = None;
        let _: Option<Tensor3<f64>> = None;
        let _: Option<Tensor4<f64>> = None;
        let _: Option<Tensor5<f64>> = None;
        let _: Option<Tensor6<f64>> = None;
        let _: Option<TensorD<f64>> = None;

        let _: Option<TensorView0<'_, f64>> = None;
        let _: Option<TensorView1<'_, f64>> = None;
        let _: Option<TensorView2<'_, f64>> = None;
        let _: Option<TensorView3<'_, f64>> = None;
        let _: Option<TensorView4<'_, f64>> = None;
        let _: Option<TensorView5<'_, f64>> = None;
        let _: Option<TensorView6<'_, f64>> = None;
        let _: Option<TensorViewD<'_, f64>> = None;

        let _: Option<TensorViewMut0<'_, f64>> = None;
        let _: Option<TensorViewMut1<'_, f64>> = None;
        let _: Option<TensorViewMut2<'_, f64>> = None;
        let _: Option<TensorViewMut3<'_, f64>> = None;
        let _: Option<TensorViewMut4<'_, f64>> = None;
        let _: Option<TensorViewMut5<'_, f64>> = None;
        let _: Option<TensorViewMut6<'_, f64>> = None;
        let _: Option<TensorViewMutD<'_, f64>> = None;

        let _: Option<ArcTensor0<f64>> = None;
        let _: Option<ArcTensor1<f64>> = None;
        let _: Option<ArcTensor2<f64>> = None;
        let _: Option<ArcTensor3<f64>> = None;
        let _: Option<ArcTensor4<f64>> = None;
        let _: Option<ArcTensor5<f64>> = None;
        let _: Option<ArcTensor6<f64>> = None;
        let _: Option<ArcTensorD<f64>> = None;
    }
}
