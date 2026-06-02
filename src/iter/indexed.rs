//! Index-paired element iterators — `IndexedIter` and `IndexedIterMut`.

use crate::dimension::Dimension;
use super::primitives::{Iter, IterMut};
use super::types::StrideState;

/// Read-only indexed iterator.
///
/// Wraps [`Iter`] and pairs each element with its multi-dimensional
/// logical index `D`. Indices increment in F-order — the inner `Iter`
/// and the separate `StrideState` advance independently but visit
/// positions in the same order, so `(index, value)` pairs stay
/// aligned without explicit synchronisation.
#[expect(missing_debug_implementations)]
pub struct IndexedIter<'a, A, D: Dimension> {
    /// Inner flat element iterator. Drives element traversal independently
    /// from the index state machine.
    iter: Iter<'a, A, D>,

    /// Index state machine that tracks the current logical position.
    state: StrideState,
}

impl<'a, A, D: Dimension> IndexedIter<'a, A, D> {
    /// Construct from an `Iter` and the source shape.
    ///
    /// The single `StrideState` drives the index produced by this wrapper;
    /// the inner `Iter` advances its own pointer/state independently. The
    /// two state machines visit logical positions in the same F-order, so
    /// the `(index, value)` pairs stay aligned without explicit synchronisation.
    pub(crate) fn new(iter: Iter<'a, A, D>, shape: &[usize]) -> Self {
        Self {
            iter,
            state: StrideState::new(shape),
        }
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedIter<'a, A, D> {
    type Item = (D, &'a A);

    /// Yields the next `(index, value)` pair. Snapshots the logical
    /// index before advancing the state machine.
    fn next(&mut self) -> Option<Self::Item> {
        // Snapshot the index for this position *before* advancing.
        let index_slice = self.state.index().to_vec();
        let value = self.iter.next()?;
        self.state.advance();
        // `Dimension::try_from_slice` lifts the runtime `&[usize]` index back
        // into the concrete dimension type `D`.
        let dim = D::try_from_slice(&index_slice)
            .expect("rank invariant: index slice len equals D rank");
        Some((dim, value))
    }

    /// Delegates to the inner element iterator.
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIter<'a, A, D> {}

/// Mutable indexed iterator.
///
/// Wraps [`IterMut`] and pairs each mutable element reference with its
/// multi-dimensional logical index `D`. The same F-order alignment
/// guarantee as [`IndexedIter`] applies.
///
/// # Safety
///
/// Each logical index maps to a distinct physical address because
/// `TensorViewMut` only admits non-broadcast, positive-stride layouts.
#[expect(missing_debug_implementations)]
pub struct IndexedIterMut<'a, A, D: Dimension> {
    /// Inner mutable flat element iterator.
    iter: IterMut<'a, A, D>,

    /// Index state machine that tracks the current logical position.
    state: StrideState,
}

impl<'a, A, D: Dimension> IndexedIterMut<'a, A, D> {
    /// Construct from an `IterMut` and the source shape.
    ///
    /// The `StrideState` is initialised from `shape` and advances in
    /// lockstep with the inner iterator, maintaining F-order alignment.
    pub(crate) fn new(iter: IterMut<'a, A, D>, shape: &[usize]) -> Self {
        Self {
            iter,
            state: StrideState::new(shape),
        }
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
    type Item = (D, &'a mut A);

    /// Yields the next `(index, &mut value)` pair.
    fn next(&mut self) -> Option<Self::Item> {
        let index_slice = self.state.index().to_vec();
        let value = self.iter.next()?;
        self.state.advance();
        let dim = D::try_from_slice(&index_slice)
            .expect("rank invariant: index slice len equals D rank");
        Some((dim, value))
    }

    /// Delegates to the inner mutable element iterator.
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIterMut<'a, A, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Dimension, Ix0, Ix2, IxDyn};
    use crate::element::Element;
    use crate::storage::Owned;
    use crate::tensor::TensorBase;

    /// Test helper: construct an owned tensor from raw data without layout
    /// validation, assuming the caller provides consistent shape and strides.
    unsafe fn make_tensor<A: Element, D: Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    /// Verifies F-order iteration over a 2x2 tensor produces indices in
    /// column-major order with their associated values.
    #[test]
    fn test_indexed_iter_order() {
        let tensor = unsafe {
            make_tensor(vec![1i32, 2, 3, 4], Ix2(2, 2))
        };
        let iter = IndexedIter::new(Iter::new(tensor.view()), tensor.shape());
        let items: Vec<(Ix2, i32)> = iter.map(|(idx, v)| (idx, *v)).collect();
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].0, Ix2(0, 0));
        assert_eq!(items[1].0, Ix2(1, 0));
        assert_eq!(items[2].0, Ix2(0, 1));
        assert_eq!(items[3].0, Ix2(1, 1));
        assert_eq!(
            items.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
    }

    /// Verifies that iterating a 0-rank scalar tensor yields exactly one
    /// (Ix0, value) pair.
    #[test]
    fn test_indexed_iter_ix0() {
        let scalar = unsafe {
            make_tensor(vec![7i32], Ix0)
        };
        let iter = IndexedIter::new(Iter::new(scalar.view()), scalar.shape());
        let items: Vec<(Ix0, i32)> = iter.map(|(idx, v)| (idx, *v)).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, Ix0);
        assert_eq!(items[0].1, 7);
    }

    /// Verifies that iterating a high-rank (7-D) IxDyn tensor yields the
    /// correct element count and correct indices at the first, second,
    /// and last positions.
    #[test]
    fn test_indexed_iter_high_rank_ixdyn() {
        let shape = IxDyn::from_slice(&[2, 2, 2, 2, 2, 2, 2]);
        let total: usize = shape.slice().iter().product();
        let tensor = unsafe {
            make_tensor((0..total as i32).collect(), shape.clone())
        };
        let iter = IndexedIter::new(Iter::new(tensor.view()), tensor.shape());
        let items: Vec<(IxDyn, i32)> = iter.map(|(idx, v)| (idx, *v)).collect();
        assert_eq!(items.len(), total);
        let zero = IxDyn::from_slice(&[0, 0, 0, 0, 0, 0, 0]);
        let one_at_axis0 = IxDyn::from_slice(&[1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(items[0].0, zero);
        assert_eq!(items[1].0, one_at_axis0);
        let last = IxDyn::from_slice(&[1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(items[total - 1].0, last);
    }

    /// `IndexedIterMut` writes through yielded references propagate back via
    /// each element's paired index.
    #[test]
    fn test_indexed_iter_mut_write() {
        let mut tensor = unsafe {
            make_tensor(vec![1i32, 2, 3, 4], Ix2(2, 2))
        };
        let shape = [2usize, 2];
        let iter = IndexedIterMut::new(IterMut::new(tensor.view_mut()), &shape);
        for (idx, value) in iter {
            // Double elements where axis-0 index is 0.
            if idx == Ix2(0, 0) || idx == Ix2(0, 1) {
                *value *= 2;
            }
        }
        let collected: Vec<_> = tensor.iter().copied().collect();
        assert_eq!(collected, vec![2, 2, 6, 4]);
    }

    /// `IndexedIter` on an empty tensor finishes immediately.
    #[test]
    fn test_indexed_iter_empty() {
        let tensor = unsafe {
            make_tensor(Vec::<f64>::new(), Ix2(0, 3))
        };
        let mut iter = IndexedIter::new(Iter::new(tensor.view()), tensor.shape());
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    /// `IndexedIterMut` on an empty tensor finishes immediately.
    #[test]
    fn test_indexed_iter_mut_empty() {
        let mut tensor = unsafe {
            make_tensor(Vec::<f64>::new(), Ix2(0, 3))
        };
        let shape = [0usize, 3];
        let mut iter =
            IndexedIterMut::new(IterMut::new(tensor.view_mut()), &shape);
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    /// `size_hint` returns the exact remaining count and decrements after
    /// each `next()`.
    #[test]
    fn test_indexed_iter_size_hint() {
        let tensor = unsafe {
            make_tensor(vec![1i32, 2, 3, 4], Ix2(2, 2))
        };
        let mut iter = IndexedIter::new(Iter::new(tensor.view()), tensor.shape());
        assert_eq!(iter.len(), 4);
        assert_eq!(iter.size_hint(), (4, Some(4)));
        iter.next();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.count(), 3);
    }

    /// `IndexedIterMut` on a rank-0 tensor yields exactly one `(Ix0, &mut A)`
    /// pair, and the mutation propagates.
    #[test]
    fn test_indexed_iter_mut_ix0() {
        let mut scalar = unsafe {
            make_tensor(vec![7i32], Ix0)
        };
        let shape = [0usize; 0];
        let iter = IndexedIterMut::new(IterMut::new(scalar.view_mut()), &shape);
        for (idx, value) in iter {
            assert_eq!(idx, Ix0);
            *value = 99;
        }
        assert_eq!(scalar.iter().copied().collect::<Vec<_>>(), vec![99]);
    }

    /// `IndexedIterMut::size_hint` returns the exact remaining count and
    /// decrements after each `next()`.
    #[test]
    fn test_indexed_iter_mut_size_hint() {
        let mut tensor = unsafe {
            make_tensor(vec![0i32; 6], Ix2(3, 2))
        };
        let shape = [3usize, 2];
        let mut iter =
            IndexedIterMut::new(IterMut::new(tensor.view_mut()), &shape);
        assert_eq!(iter.len(), 6);
        assert_eq!(iter.size_hint(), (6, Some(6)));
        iter.next();
        assert_eq!(iter.size_hint(), (5, Some(5)));
        assert_eq!(iter.count(), 5);
    }
}
