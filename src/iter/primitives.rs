//! Flat element iterators and the F-order stride state machine that drives them.

use core::marker::PhantomData;

use crate::dimension::Dimension;
use crate::tensor::{TensorView, TensorViewMut};
use super::types::StrideState;

// ── offset_of_index helper ──

/// `pub(crate)` helper used by both `Iter` and `IterMut` slow paths.
///
/// `offset = base_offset + Σ(strides[i] * index[i])`. The function only sums
/// pre-validated stride/index pairs from a `TensorView`/`TensorViewMut`
/// constructed via the safe tensor paths; no overflow check is needed because
/// tensor construction has already verified representability.
#[inline]
pub(crate) fn offset_of_index(strides: &[usize], base_offset: usize, index: &[usize]) -> usize {
    debug_assert_eq!(strides.len(), index.len());
    let mut offset = base_offset;
    for (s, i) in strides.iter().zip(index.iter()) {
        offset += s * i;
    }
    offset
}

// ── Iter ──

/// Flat element iterator. Yields elements in logical F-order.
///
/// `'a` and `A` are anchored by the embedded `tensor: TensorView<'a, A, D>`,
/// so no extra `PhantomData<&'a A>` is required: `PhantomData` is only needed
/// when the iterator does *not* keep the view around.
#[expect(
    missing_debug_implementations,
    reason = "iterator is not meant to be introspected"
)]
pub struct Iter<'a, A, D: Dimension> {
    tensor: TensorView<'a, A, D>,
    state: StrideState,
    remaining: usize,
    /// In the fast path this is the next physical element offset; in the slow
    /// path it is unused (slow path computes the offset on each `next()`).
    next_fast_offset: usize,
    is_f_contiguous: bool,
}

impl<'a, A, D: Dimension> Iter<'a, A, D> {
    /// Construct from a view; selects fast vs slow path automatically.
    pub(crate) fn new(tensor: TensorView<'a, A, D>) -> Self {
        let remaining = tensor.len();
        let is_f_contiguous = tensor.is_f_contiguous();
        let state = StrideState::new(tensor.shape());
        let next_fast_offset = tensor.offset();
        Self {
            tensor,
            state,
            remaining,
            next_fast_offset,
            is_f_contiguous,
        }
    }
}

impl<'a, A, D: Dimension> Iterator for Iter<'a, A, D> {
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        // `as_storage_ptr()` returns the storage base pointer WITHOUT
        // adding `self.tensor.offset()`. We add the logical offset
        // explicitly on every yield — fast path seeds `next_fast_offset`
        // with `tensor.offset()` at construction, slow path passes
        // `tensor.offset()` into `offset_of_index` — so the final
        // `base_ptr.add(offset)` applies `offset` exactly once. Using
        // `as_ptr()` here would double-apply the offset.
        let base_ptr = self.tensor.as_storage_ptr();
        let offset = if self.is_f_contiguous {
            // Fast path: monotonic pointer increment.
            let off = self.next_fast_offset;
            self.next_fast_offset += 1;
            off
        } else {
            // Slow path: stride-based offset from validated metadata.
            let off = offset_of_index(
                self.tensor.strides(),
                self.tensor.offset(),
                self.state.index(),
            );
            self.state.advance();
            off
        };
        self.remaining -= 1;
        // SAFETY: `tensor` was constructed via the safe tensor paths;
        // `offset` is within the validated storage range and points to an
        // initialised element slot. `'a` lifetime is tied to
        // `self.tensor: TensorView<'a, A, D>`.
        unsafe { Some(&*base_ptr.add(offset)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for Iter<'a, A, D> {}

/// Mutable flat element iterator. Yields elements in logical F-order.
///
/// # Safety
///
/// Each call to `next()` produces a `&'a mut A` pointing at a distinct logical
/// element slot. Non-overlapping mutable references rely on the layout
/// invariants validated by `TensorViewMut` construction: no negative strides,
/// no zero-stride / broadcast layout, no padding exposure, and
/// shape/stride/offset/storage_len consistency.
#[expect(
    missing_debug_implementations,
    reason = "iterator is not meant to be introspected"
)]
pub struct IterMut<'a, A, D: Dimension> {
    /// Base pointer captured at construction time. We do **not** keep the
    /// `TensorViewMut` around because doing so would conflict with the
    /// `&'a mut A` references handed out by `next()`. Lifetime soundness is
    /// expressed via the `PhantomData<&'a mut A>` marker.
    base_ptr: *mut A,
    strides: Vec<usize>,
    base_offset: usize,
    state: StrideState,
    remaining: usize,
    /// Fast-path running offset (only valid when `is_f_contiguous == true`).
    next_fast_offset: usize,
    is_f_contiguous: bool,
    _marker: PhantomData<&'a mut A>,
    _dim: PhantomData<D>,
}

impl<'a, A, D: Dimension> IterMut<'a, A, D> {
    /// Construct from a mutable view; selects fast vs slow path automatically.
    ///
    /// The runtime `debug_assert!` on `has_zero_stride()` is a **redundant
    /// defence in depth** on top of the primary compile-time guarantee that
    /// broadcast views only return `TensorView`, which has no `iter_mut()`.
    /// It catches misuse of `from_raw_parts_mut` that bypassed the safe
    /// constructors.
    pub(crate) fn new(view: TensorViewMut<'a, A, D>) -> Self {
        debug_assert!(
            !view.has_zero_stride(),
            "IterMut on a zero-stride/broadcast view violates the \
             compile-time guarantee that broadcast views are read-only; \
             only reachable via misuse of `from_raw_parts_mut`."
        );
        let remaining = view.len();
        let is_f_contiguous = view.is_f_contiguous();
        let strides = view.strides().to_vec();
        let base_offset = view.offset();
        let state = StrideState::new(view.shape());

        // Consume the view to obtain the raw base pointer.
        let mut view = view;
        let base_ptr = view.as_storage_mut_ptr();

        Self {
            base_ptr,
            strides,
            base_offset,
            state,
            remaining,
            next_fast_offset: base_offset,
            is_f_contiguous,
            _marker: PhantomData,
            _dim: PhantomData,
        }
    }
}

impl<'a, A, D: Dimension> Iterator for IterMut<'a, A, D> {
    type Item = &'a mut A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let offset = if self.is_f_contiguous {
            // Fast path: monotonic pointer increment.
            let off = self.next_fast_offset;
            self.next_fast_offset += 1;
            off
        } else {
            // Slow path: stride-based offset, computed from validated metadata.
            let off = offset_of_index(&self.strides, self.base_offset, self.state.index());
            self.state.advance();
            off
        };
        self.remaining -= 1;
        // SAFETY: each logical index maps to a distinct, non-overlapping
        // physical address for layouts admissible to `TensorViewMut`. The
        // state machine visits each logical index at most once and
        // monotonically advances `remaining`.
        unsafe { Some(&mut *self.base_ptr.add(offset)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, A, D: Dimension> ExactSizeIterator for IterMut<'a, A, D> {}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::{Iter, IterMut};
    use crate::dimension::{Ix0, Ix2};
    use crate::tensor::TensorBase;

    unsafe fn make_tensor<A: crate::element::Element, D: crate::dimension::Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<crate::storage::Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    // ── Iter ──

    /// F-order contiguous tensor: iter order == physical layout.
    #[test]
    fn test_elements_f_contig() {
        let tensor = unsafe { make_tensor(vec![1i32, 2, 3, 4], Ix2(2, 2)) };
        let values: Vec<_> = Iter::new(tensor.view()).copied().collect();
        assert_eq!(values, vec![1, 2, 3, 4]);
    }

    /// Non-contiguous view (transpose) exercises the stride-based slow path.
    #[test]
    fn test_elements_non_contiguous() {
        let tensor = unsafe { make_tensor(vec![1i32, 2, 3, 4, 5, 6], Ix2(3, 2)) };
        let transposed = tensor.transpose();
        assert!(!transposed.is_f_contiguous());
        let values: Vec<_> = Iter::new(transposed).copied().collect();
        assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);
    }

    /// Empty array: `iter()` finishes immediately, count == 0.
    #[test]
    fn test_elements_empty() {
        let tensor = unsafe { make_tensor(Vec::<f64>::new(), Ix2(0, 3)) };
        assert_eq!(Iter::new(tensor.view()).count(), 0);
        assert_eq!(Iter::new(tensor.view()).len(), 0);
    }

    /// Ix0 / rank-0 tensor: `iter()` yields exactly 1 element.
    #[test]
    fn test_elements_ix0() {
        let scalar = unsafe { make_tensor(vec![7i32], Ix0) };
        let values: Vec<_> = Iter::new(scalar.view()).copied().collect();
        assert_eq!(values, vec![7]);
        assert_eq!(Iter::new(scalar.view()).len(), 1);
    }

    // ── IterMut ──

    /// `iter_mut()` writes propagate back through the source tensor.
    #[test]
    fn test_elements_mut_write() {
        let mut tensor = unsafe { make_tensor(vec![1i32, 2, 3], Ix2(3, 1)) };
        for value in IterMut::new(tensor.view_mut()) {
            *value *= 2;
        }
        let collected: Vec<_> = Iter::new(tensor.view()).copied().collect();
        assert_eq!(collected, vec![2, 4, 6]);
    }
}
