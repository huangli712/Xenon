//! Tensor iterators.
//!
//! Defines the public iterator surface area as specified in `10-iterator.md §5`:
//! - `Iter` / `IterMut` (flat element iterators, §5.1, wired up in W12T3 / W12T4)
//! - `AxisIter` / `AxisIterMut` (axis-wise sub-view iterators, §5.2, wired up in W12T5)
//! - `IndexedIter` / `IndexedIterMut` (index-paired iterators, §5.4, wired up in W12T6)
//!
//! Entry methods on `TensorBase` (§5.5) are added in W12T7.

mod axis;
mod elements;
mod indexed;

pub use axis::{AxisIter, AxisIterMut};
pub use elements::Iter;
pub use elements::IterMut;
pub use indexed::{IndexedIter, IndexedIterMut};

// ── TensorBase entry methods (W12T7) ──

use crate::dimension::{Axis, Dimension, RemoveAxis};
use crate::error::XenonError;
use crate::storage::{Storage, StorageMut, ViewMutRepr, ViewRepr};
use crate::tensor::TensorBase;

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
{
    /// Construct a read-only view from any Storage-backed tensor.
    fn as_view(&self) -> crate::tensor::TensorView<'_, A, D> {
        // SAFETY: Storage guarantees valid base pointer + len.
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        }
    }

    /// Element iterator (immutable). Yields `&A` in logical F-order.
    pub fn iter(&self) -> Iter<'_, A, D> {
        Iter::new(self.as_view())
    }

    /// Indexed element iterator (immutable). Yields `(D, &A)` in F-order.
    pub fn indexed_iter(&self) -> IndexedIter<'_, A, D> {
        let shape: Vec<usize> = self.shape().to_vec();
        IndexedIter::new(Iter::new(self.as_view()), &shape)
    }

    /// Axis iterator.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` if `axis` is out of range for the
    /// tensor's dimensionality.
    pub fn axis_iter(&self, axis: Axis) -> Result<AxisIter<'_, A, D>, XenonError>
    where
        D: RemoveAxis,
    {
        AxisIter::new(self.as_view(), axis)
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension + Clone,
{
    fn as_view_mut(&mut self) -> crate::tensor::TensorViewMut<'_, A, D> {
        // SAFETY: StorageMut guarantees valid base pointer + len.
        let storage = unsafe {
            ViewMutRepr::from_raw_parts_mut(self.storage.as_mut_ptr(), self.storage.len())
        };
        unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        }
    }

    /// Mutable element iterator. Yields `&mut A` in logical F-order.
    pub fn iter_mut(&mut self) -> IterMut<'_, A, D> {
        IterMut::new(self.as_view_mut())
    }

    /// Mutable indexed iterator. Yields `(D, &mut A)` in F-order.
    pub fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, A, D> {
        let shape: Vec<usize> = self.shape().to_vec();
        IndexedIterMut::new(IterMut::new(self.as_view_mut()), &shape)
    }

    /// Mutable axis iterator. Same error semantics as `axis_iter`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` if `axis` is out of range for the
    /// tensor's dimensionality.
    pub fn axis_iter_mut(&mut self, axis: Axis) -> Result<AxisIterMut<'_, A, D>, XenonError>
    where
        D: RemoveAxis,
    {
        AxisIterMut::new(self.as_view_mut(), axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Axis, IxDyn};
    use crate::error::XenonError;

    unsafe fn make_tensor<A: crate::element::Element, D: Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<crate::storage::Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    #[test]
    fn test_tensor_iter_integration() {
        let mut tensor = unsafe { make_tensor(vec![1i32, 2, 3, 4], crate::dimension::Ix2(2, 2)) };
        assert_eq!(tensor.iter().len(), 4);

        let (idx0, _) = tensor.indexed_iter().next().expect("tensor has 4 elements");
        assert_eq!(idx0, crate::dimension::Ix2(0, 0));

        assert_eq!(
            tensor.axis_iter(Axis(1)).expect("Axis(1) is valid").len(),
            2
        );

        tensor.iter_mut().for_each(|v| *v += 1);
        let after: Vec<_> = tensor.iter().copied().collect();
        assert_eq!(after, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_axis_iter_mut_integration() {
        let mut tensor = unsafe { make_tensor(vec![1i32, 2, 3, 4], crate::dimension::Ix2(2, 2)) };
        for mut row in tensor.axis_iter_mut(Axis(0)).expect("Axis(0) is valid") {
            for value in row.iter_mut() {
                *value += 10;
            }
        }
        let mut after: Vec<_> = tensor.iter().copied().collect();
        after.sort();
        assert_eq!(after, vec![11, 12, 13, 14]);
    }

    #[test]
    fn test_axis_iter_dyn_rank0_error() {
        let scalar = unsafe { make_tensor(vec![1.0_f64], IxDyn::from_slice(&[])) };
        assert!(matches!(
            scalar.axis_iter(Axis(0)),
            Err(XenonError::InvalidAxis {
                axis: 0,
                ndim: 0,
                ..
            })
        ));
    }

    #[test]
    fn test_axis_iter_large_axis_index_error() {
        let tensor = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        assert!(matches!(
            tensor.axis_iter(Axis(usize::MAX)),
            Err(XenonError::InvalidAxis { .. })
        ));
        let mut tensor_mut = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        assert!(matches!(
            tensor_mut.axis_iter_mut(Axis(usize::MAX)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }

    #[test]
    fn test_axis_iter_out_of_bounds_invalid_axis() {
        let tensor = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        assert!(matches!(
            tensor.axis_iter(Axis(2)),
            Err(XenonError::InvalidAxis {
                axis: 2,
                ndim: 2,
                ..
            })
        ));
        assert!(matches!(
            tensor.axis_iter(Axis(5)),
            Err(XenonError::InvalidAxis {
                axis: 5,
                ndim: 2,
                ..
            })
        ));

        let mut tensor_mut = unsafe { make_tensor(vec![0.0_f64; 6], crate::dimension::Ix2(2, 3)) };
        assert!(matches!(
            tensor_mut.axis_iter_mut(Axis(2)),
            Err(XenonError::InvalidAxis {
                axis: 2,
                ndim: 2,
                ..
            })
        ));
    }

    #[test]
    fn test_elements_large_tensor_count() {
        let n0: usize = 100;
        let n1: usize = 1_000;
        let tensor = unsafe { make_tensor(vec![0_i32; 100_000], crate::dimension::Ix2(n0, n1)) };
        assert_eq!(tensor.iter().len(), n0 * n1);
        assert_eq!(tensor.iter().count(), n0 * n1);
    }

    #[test]
    fn test_iter_mut_accepts_non_broadcast_owned_tensor() {
        let mut tensor = unsafe { make_tensor((0..9).collect(), crate::dimension::Ix2(3, 3)) };
        tensor.iter_mut().for_each(|v| *v *= 2);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            vec![0, 2, 4, 6, 8, 10, 12, 14, 16]
        );
    }
}
