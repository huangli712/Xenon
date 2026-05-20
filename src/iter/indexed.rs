use crate::dimension::Dimension;
use crate::iter::elements::{Iter, IterMut, StrideState};

/// Element iterator paired with the multi-dimensional logical index.
/// Yields `(D, &'a A)` tuples; indices increment in F-order.
/// (10-iterator §5.4)
#[expect(missing_debug_implementations, reason = "iterator is not meant to be introspected")]
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
        // into the concrete dimension type `D` (works for both static Ix* and
        // IxDyn; see 02-dimension §5.1).
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
/// (10-iterator §5.4)
///
/// # Safety
///
/// Inherits the §6.5 aliasing argument from `IterMut`: the underlying
/// `TensorViewMut` admits only no-broadcast, positive-stride layouts validated
/// at construction time, so each logical index maps to a distinct physical
/// address.
#[expect(missing_debug_implementations, reason = "iterator is not meant to be introspected")]
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
    use crate::iter::elements::Iter;
    use crate::tensor::TensorBase;

    unsafe fn make_tensor<A: crate::element::Element, D: Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<crate::storage::Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

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
        assert_eq!(items.iter().map(|(_, v)| *v).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_indexed_iter_ix0() {
        let scalar = unsafe { make_tensor(vec![7i32], Ix0) };
        let iter = IndexedIter::new(Iter::new(scalar.view()), scalar.shape());
        let items: Vec<(Ix0, i32)> = iter.map(|(idx, v)| (idx, *v)).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, Ix0);
        assert_eq!(items[0].1, 7);
    }

    #[test]
    fn test_indexed_iter_high_rank_ixdyn() {
        let shape = IxDyn::from_slice(&[2, 2, 2, 2, 2, 2, 2]);
        let total: usize = shape.slice().iter().product();
        let tensor =
            unsafe { make_tensor((0..total as i32).collect(), shape.clone()) };
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