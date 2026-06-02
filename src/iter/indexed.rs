use crate::dimension::Dimension;
use crate::iter::primitives::{Iter, IterMut};
use crate::iter::types::StrideState;

/// Element iterator paired with the multi-dimensional logical index.
/// Yields `(D, &'a A)` tuples; indices increment in F-order.
#[expect(
    missing_debug_implementations,
    reason = "iterator is not meant to be introspected"
)]
pub struct IndexedIter<'a, A, D: Dimension> {
    iter: Iter<'a, A, D>,
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

    fn next(&mut self) -> Option<Self::Item> {
        // Snapshot the index for this position *before* advancing.
        let index_slice = self.state.index().to_vec();
        let value = self.iter.next()?;
        self.state.advance();
        // `Dimension::try_from_slice` lifts the runtime `&[usize]` index back
        // into the concrete dimension type `D`.
        let dim = D::try_from_slice(&index_slice)
            .expect("rank invariant: index slice rank == D rank from StrideState::new shape");
        Some((dim, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIter<'a, A, D> {}

/// Mutable variant of `IndexedIter`. Yields `(D, &'a mut A)` tuples.
///
/// # Safety
///
/// Each logical index maps to a distinct physical address because
/// `TensorViewMut` only admits non-broadcast, positive-stride layouts.
#[expect(
    missing_debug_implementations,
    reason = "iterator is not meant to be introspected"
)]
pub struct IndexedIterMut<'a, A, D: Dimension> {
    iter: IterMut<'a, A, D>,
    state: StrideState,
}

impl<'a, A, D: Dimension> IndexedIterMut<'a, A, D> {
    pub(crate) fn new(iter: IterMut<'a, A, D>, shape: &[usize]) -> Self {
        Self {
            iter,
            state: StrideState::new(shape),
        }
    }
}

impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
    type Item = (D, &'a mut A);

    fn next(&mut self) -> Option<Self::Item> {
        let index_slice = self.state.index().to_vec();
        let value = self.iter.next()?;
        self.state.advance();
        let dim = D::try_from_slice(&index_slice)
            .expect("rank invariant: index slice rank == D rank from StrideState::new shape");
        Some((dim, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIterMut<'a, A, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Dimension, Ix0, Ix2, IxDyn};
    use crate::iter::primitives::Iter;
    use crate::tensor::TensorBase;

    unsafe fn make_tensor<A: crate::element::Element, D: Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<crate::storage::Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    /// Verifies F-order iteration over a 2x2 tensor produces indices in column-major order with their associated values.
    #[test]
    fn test_indexed_iter_order() {
        let tensor = unsafe { make_tensor(vec![1i32, 2, 3, 4], Ix2(2, 2)) };
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

    /// Verifies that iterating a 0-rank scalar tensor yields exactly one (Ix0, value) pair.
    #[test]
    fn test_indexed_iter_ix0() {
        let scalar = unsafe { make_tensor(vec![7i32], Ix0) };
        let iter = IndexedIter::new(Iter::new(scalar.view()), scalar.shape());
        let items: Vec<(Ix0, i32)> = iter.map(|(idx, v)| (idx, *v)).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, Ix0);
        assert_eq!(items[0].1, 7);
    }

    /// Verifies that iterating a high-rank (7-D) IxDyn tensor yields the correct element count and correct indices at the first, second, and last positions.
    #[test]
    fn test_indexed_iter_high_rank_ixdyn() {
        let shape = IxDyn::from_slice(&[2, 2, 2, 2, 2, 2, 2]);
        let total: usize = shape.slice().iter().product();
        let tensor = unsafe { make_tensor((0..total as i32).collect(), shape.clone()) };
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
}
