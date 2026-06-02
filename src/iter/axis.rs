//! Axis-wise sub-view iterators — `AxisIter` and `AxisIterMut`.

use core::marker::PhantomData;
use std::borrow::Cow;

use crate::error::XenonError;
use crate::dimension::{Axis, Dimension, RemoveAxis};
use crate::element::Element;
use crate::layout::Strides;
use crate::tensor::{TensorView, TensorViewMut};

/// Read-only axis iterator.
///
/// Yields [`TensorView`] sub-views of rank `D::Smaller` by slicing along
/// the selected axis. Each sub-view spans all remaining dimensions. The
/// iterator itself requires only `D: Dimension`; the `Iterator` impl
/// further constrains to `D: RemoveAxis` so the item type can be
/// `TensorView<'a, A, D::Smaller>`.
#[expect(missing_debug_implementations)]
pub struct AxisIter<'a, A, D: Dimension> {
    /// Storage base pointer, captured from the source view at construction.
    base_ptr: *const A,

    /// Offset from storage base to the logical first element of the tensor.
    base_offset: usize,

    /// Shape of a single sub-view (all axes except the iteration axis).
    sub_shape: Vec<usize>,

    /// Strides of a single sub-view, in the same axis order as `sub_shape`.
    sub_strides: Vec<usize>,

    /// Stride (in elements) between consecutive positions along the
    /// iteration axis.
    axis_stride: usize,

    /// Number of sub-views (length of the iteration axis).
    len: usize,

    /// Current position along the iteration axis (0-based).
    pos: usize,

    /// Total number of elements in the underlying storage, used for bounds
    /// validation when constructing sub-views.
    storage_len: usize,

    /// Lifetime anchor — ties yielded references to the source borrow.
    _marker: PhantomData<&'a A>,

    /// Consumes the dimension type parameter so the struct is well-formed.
    _dim: PhantomData<D>,
}

impl<'a, A, D: Dimension> AxisIter<'a, A, D> {
    /// Construct an axis iterator for the given view and axis.
    ///
    /// Returns `InvalidAxis` if `axis` is out of range for the tensor's
    /// dimensionality.
    pub(crate) fn new(view: TensorView<'a, A, D>, axis: Axis) -> Result<Self, XenonError> {
        let ndim = view.ndim();
        if axis.0 >= ndim {
            return Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("iter::AxisIter::new"),
                axis: axis.0,
                ndim,
                shape: view.shape().to_vec(),
            });
        }
        let shape = view.shape();
        let strides = view.strides();
        let len = shape[axis.0];
        let axis_stride = strides[axis.0];
        let base_offset = view.offset();
        let base_ptr = view.as_storage_ptr();
        let storage_len = view.storage_len();

        let mut sub_shape = Vec::with_capacity(ndim - 1);
        let mut sub_strides = Vec::with_capacity(ndim - 1);
        for (i, (&s, &st)) in shape.iter().zip(strides.iter()).enumerate() {
            if i != axis.0 {
                sub_shape.push(s);
                sub_strides.push(st);
            }
        }

        Ok(Self {
            base_ptr,
            base_offset,
            sub_shape,
            sub_strides,
            axis_stride,
            len,
            pos: 0,
            storage_len,
            _marker: PhantomData,
            _dim: PhantomData,
        })
    }
}

impl<'a, A, D> Iterator for AxisIter<'a, A, D>
where
    A: Element,
    D: RemoveAxis,
{
    type Item = TensorView<'a, A, D::Smaller>;

    /// Yields the next sub-view at the current axis position, then
    /// advances `pos` by one.
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.len {
            return None;
        }
        let step_offset = self.base_offset + self.axis_stride * self.pos;
        self.pos += 1;
        let sub_dim = <D::Smaller as Dimension>::try_from_slice(&self.sub_shape)
            .expect("rank invariant: D::Smaller has ndim == D::NDIM - 1");
        let sub_strides = Strides::<D::Smaller>::from_slice(&self.sub_strides)
            .expect("rank invariant: stride rank matches reduced shape rank");
        let view = unsafe {
            TensorView::<'a, A, D::Smaller>::from_raw_parts(
                self.base_ptr,
                self.storage_len,
                sub_dim,
                sub_strides,
                step_offset,
            )
            .expect("invariants pre-validated at construction")
        };
        Some(view)
    }

    /// Returns the exact remaining sub-view count.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
where
    A: Element,
    D: RemoveAxis,
{
}
/// Mutable axis iterator.
///
/// Yields [`TensorViewMut`] sub-views of rank `D::Smaller` by slicing
/// along the selected axis. The construction-time `debug_assert!` enforces
/// that broadcast (zero-stride) layouts are rejected — these violate the
/// aliasing guarantees of `&mut`.
///
/// # Safety
///
/// Each `next()` advances `pos` by 1 monotonically; consecutive yields
/// are separated by `axis_stride` elements, so the produced `&mut`-backed
/// views cover non-overlapping logical regions.
#[expect(missing_debug_implementations)]
pub struct AxisIterMut<'a, A, D: Dimension> {
    /// Storage base pointer, captured from the mutable source view.
    base_ptr: *mut A,

    /// Offset from storage base to the logical first element of the tensor.
    base_offset: usize,

    /// Shape of a single sub-view (all axes except the iteration axis).
    sub_shape: Vec<usize>,

    /// Strides of a single sub-view, in the same axis order as `sub_shape`.
    sub_strides: Vec<usize>,

    /// Stride (in elements) between consecutive positions along the
    /// iteration axis.
    axis_stride: usize,

    /// Number of sub-views (length of the iteration axis).
    len: usize,

    /// Current position along the iteration axis (0-based).
    pos: usize,

    /// Total number of elements in the underlying storage.
    storage_len: usize,

    /// Lifetime anchor — ties yielded mutable references to the source
    /// borrow.
    _marker: PhantomData<&'a mut A>,

    /// Consumes the dimension type parameter so the struct is well-formed.
    _dim: PhantomData<D>,
}

impl<'a, A, D: Dimension> AxisIterMut<'a, A, D> {
    /// Construct a mutable axis iterator for the given view and axis.
    ///
    /// Returns `InvalidAxis` if `axis` is out of range. The
    /// `debug_assert!` rejects broadcast (zero-stride) layouts which
    /// violate `&mut` aliasing guarantees.
    pub(crate) fn new(view: TensorViewMut<'a, A, D>, axis: Axis) -> Result<Self, XenonError> {
        let ndim = view.ndim();
        if axis.0 >= ndim {
            return Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("iter::AxisIterMut::new"),
                axis: axis.0,
                ndim,
                shape: view.shape().to_vec(),
            });
        }
        debug_assert!(
            !view.has_zero_stride(),
            "AxisIterMut on zero-stride/broadcast view is unsupported"
        );

        let shape = view.shape();
        let strides = view.strides();
        let len = shape[axis.0];
        let axis_stride = strides[axis.0];
        let base_offset = view.offset();
        let storage_len = view.storage_len();

        let mut sub_shape = Vec::with_capacity(ndim - 1);
        let mut sub_strides = Vec::with_capacity(ndim - 1);
        for (i, (&s, &st)) in shape.iter().zip(strides.iter()).enumerate() {
            if i != axis.0 {
                sub_shape.push(s);
                sub_strides.push(st);
            }
        }

        let mut view = view;
        let base_ptr = view.as_storage_mut_ptr();

        Ok(Self {
            base_ptr,
            base_offset,
            sub_shape,
            sub_strides,
            axis_stride,
            len,
            pos: 0,
            storage_len,
            _marker: PhantomData,
            _dim: PhantomData,
        })
    }
}

impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
where
    A: Element,
    D: RemoveAxis,
{
    type Item = TensorViewMut<'a, A, D::Smaller>;

    /// Yields the next mutable sub-view at the current axis position,
    /// then advances `pos` by one.
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.len {
            return None;
        }
        let step_offset = self.base_offset + self.axis_stride * self.pos;
        self.pos += 1;
        let sub_dim = <D::Smaller as Dimension>::try_from_slice(&self.sub_shape)
            .expect("rank invariant: D::Smaller has ndim == D::NDIM - 1");
        let sub_strides = Strides::<D::Smaller>::from_slice(&self.sub_strides)
            .expect("rank invariant: stride rank matches reduced shape rank");
        let view = unsafe {
            TensorViewMut::<'a, A, D::Smaller>::from_raw_parts_mut(
                self.base_ptr,
                self.storage_len,
                sub_dim,
                sub_strides,
                step_offset,
            )
            .expect("invariants pre-validated at construction")
        };
        Some(view)
    }

    /// Returns the exact remaining mutable sub-view count.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D>
where
    A: Element,
    D: RemoveAxis,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Axis, IxDyn};
    use crate::tensor::TensorBase;

    unsafe fn make_tensor<A: crate::element::Element, D: Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<crate::storage::Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    /// Verifies that `AxisIter::len()` and `count()` both report the size of the
    /// selected axis (axis 0 of a 2x3 tensor yields 2 sub-views).
    #[test]
    fn test_axis_iter_count() {
        let tensor = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        let iter = AxisIter::new(tensor.view(), Axis(0)).expect("Axis(0) is valid for 2-D tensor");
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.count(), 2);
    }

    /// Verifies that each sub-view yielded by `AxisIter` over axis 0 of a 2x3
    /// tensor has the reduced shape `[3]` (the remaining axis).
    #[test]
    fn test_axis_iter_shape() {
        let tensor = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        let mut iter =
            AxisIter::new(tensor.view(), Axis(0)).expect("Axis(0) is valid for 2-D tensor");
        let sub = iter
            .next()
            .expect("Iterator should yield at least one element");
        assert_eq!(sub.shape(), &[3]);
    }

    /// Verifies behavior on a tensor with an empty axis: iterating the empty
    /// axis yields zero sub-views, while iterating a non-empty axis yields
    /// sub-views whose remaining axis is empty.
    #[test]
    fn test_axis_iter_empty_axis() {
        let tensor = unsafe { make_tensor(Vec::<f64>::new(), crate::dimension::Ix2(0, 3)) };
        let iter = AxisIter::new(tensor.view(), Axis(0)).expect("Axis(0) is valid even if empty");
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.count(), 0);

        let iter = AxisIter::new(tensor.view(), Axis(1)).expect("Axis(1) is valid");
        assert_eq!(iter.len(), 3);
        for sub in iter {
            assert_eq!(sub.shape(), &[0]);
        }
    }

    /// Verifies that constructing an `AxisIter` on a rank-0 dynamic tensor
    /// returns `InvalidAxis` because no axis exists to iterate over.
    #[test]
    fn test_axis_iter_dyn_rank0_error() {
        let scalar = unsafe { make_tensor(vec![1.0_f64], IxDyn::from_slice(&[])) };
        assert!(matches!(
            AxisIter::new(scalar.view(), Axis(0)),
            Err(XenonError::InvalidAxis {
                axis: 0,
                ndim: 0,
                ..
            })
        ));
    }

    /// Verifies that passing an out-of-range axis index (`usize::MAX`) to
    /// `AxisIter::new` returns an `InvalidAxis` error.
    #[test]
    fn test_axis_iter_large_axis_index_error() {
        let tensor = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        assert!(matches!(
            AxisIter::new(tensor.view(), Axis(usize::MAX)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }

    /// Verifies that `AxisIterMut::new` rejects an out-of-bounds axis index
    /// (axis 2 on a 2-D tensor) with an `InvalidAxis` error.
    #[test]
    fn test_axis_iter_mut_axis_out_of_bounds() {
        let mut tensor = unsafe { make_tensor(Vec::<f64>::new(), crate::dimension::Ix2(2, 3)) };
        assert!(matches!(
            AxisIterMut::new(tensor.view_mut(), Axis(2)),
            Err(XenonError::InvalidAxis {
                axis: 2,
                ndim: 2,
                ..
            })
        ));
    }
}
