use crate::dimension::Dimension;
use crate::dispatch::ParallelExecStrategy;
use crate::element::Element;
use crate::storage::{ArcRepr, Owned, ViewMutRepr, ViewRepr};
use crate::tensor::{TensorBase, TensorView};

#[cfg(feature = "parallel")]
pub(crate) struct ParIter<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
    /// Logical index interval [lo, hi) currently owned by this producer.
    /// Initially [0, base.len()); rayon splits sub-ranges via Producer::split_at.
    lo: usize,
    hi: usize,
    chunk_size: Option<usize>,
    max_workers: Option<usize>,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParIter<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    /// Construct with default strategy fields. Used by `TensorBase::par_iter()`.
    pub(crate) fn new(base: TensorView<'a, A, D>) -> Self {
        let hi = base.len();
        Self {
            base,
            lo: 0,
            hi,
            chunk_size: None,
            max_workers: None,
        }
    }

    /// Override strategy fields after construction (used by par_map_checked etc.).
    /// See 09-parallel §5.6 line 272.
    pub(crate) fn with_strategy(mut self, strategy: &ParallelExecStrategy) -> Self {
        self.chunk_size = strategy.chunk_size();
        self.max_workers = strategy.max_workers();
        self
    }
}

// `Clone` is required by `par_map_checked` (W15T7) to run the two-pass
// pattern without cloning the user closure: phase 1 consumes a cloned
// ParIter via `try_for_each`, phase 2 consumes the original. The clone
// is metadata-only (TensorView + 4 small scalar fields) — no element
// data is duplicated.
//
// Manual impl reconstructs TensorView from `pub(crate)` fields.
// `TensorView: Clone` was spec'd in 07-tensor §5 but not yet implemented;
// a manual clone here is functionally equivalent and avoids adding a
// derive across the crate boundary in this task.
#[cfg(feature = "parallel")]
impl<'a, A, D> Clone for ParIter<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    fn clone(&self) -> Self {
        // SAFETY: all TensorBase fields are pub(crate), and ViewRepr is Copy.
        // Reconstructing from identical metadata produces a valid view.
        let base = unsafe {
            TensorBase::new_unchecked(
                self.base.storage,
                self.base.shape.clone(),
                self.base.strides.clone(),
                self.base.offset,
                self.base.flags,
                self.base.derived_from_view_mut,
            )
        };
        Self {
            base,
            lo: self.lo,
            hi: self.hi,
            chunk_size: self.chunk_size,
            max_workers: self.max_workers,
        }
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::ParallelIterator for ParIter<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    type Item = &'a A;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }
    fn opt_len(&self) -> Option<usize> {
        Some(self.hi - self.lo)
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IndexedParallelIterator for ParIter<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    fn len(&self) -> usize {
        self.hi - self.lo
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(self)
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::plumbing::Producer for ParIter<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    type Item = &'a A;
    type IntoIter = ParIterSeq<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        ParIterSeq::new(self.base, self.lo, self.hi)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // Disjoint coverage invariant (09-parallel §5.6 line 268):
        // produces two sub-producers covering [lo, mid) and [mid, hi) with
        // no overlap and no gap.
        debug_assert!(index <= self.hi - self.lo);
        let mid = self.lo + index;

        // Clone metadata (ViewRepr is Copy; shape/strides/offset/flags
        // are Clone/Copy). The right side takes ownership of the original
        // base; the left side gets a freshly reconstructed copy.
        // SAFETY: all TensorBase fields are pub(crate); reconstructing
        // from identical metadata produces a valid view.
        let base_left = unsafe {
            TensorBase::new_unchecked(
                self.base.storage,
                self.base.shape.clone(),
                self.base.strides.clone(),
                self.base.offset,
                self.base.flags,
                self.base.derived_from_view_mut,
            )
        };
        let left = Self {
            base: base_left,
            lo: self.lo,
            hi: mid,
            chunk_size: self.chunk_size,
            max_workers: self.max_workers,
        };
        let right = Self {
            base: self.base,
            lo: mid,
            hi: self.hi,
            chunk_size: self.chunk_size,
            max_workers: self.max_workers,
        };
        (left, right)
    }
}

/// Sequential per-chunk iterator. F-contiguous: returns a slice iterator
/// over the contiguous sub-range; non-contiguous: walks logical indices
/// via stride state machine (deferred — see note below).
#[cfg(feature = "parallel")]
pub(crate) struct ParIterSeq<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
    cursor: usize,
    end: usize,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParIterSeq<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    fn new(base: TensorView<'a, A, D>, lo: usize, hi: usize) -> Self {
        Self {
            base,
            cursor: lo,
            end: hi,
        }
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> Iterator for ParIterSeq<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    type Item = &'a A;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let i = self.cursor;
        self.cursor += 1;
        if self.base.is_f_contiguous() {
            // F-contiguous fast path: index directly into the underlying slice.
            // SAFETY: base is F-contiguous + non-broadcast, so as_slice()
            // returns Some, and i < base.len() — both guaranteed.
            let slice = self
                .base
                .as_slice()
                .expect("ParIterSeq only constructed for F-contiguous tensors");
            // SAFETY: the slice is backed by the base storage whose lifetime
            // is 'a; extending from &self borrow to 'a is sound.
            Some(unsafe { &*(&slice[i] as *const A) })
        } else {
            // Non-contiguous walk: deferred to a later wave.
            unimplemented!(
                "non-F-contiguous ParIter walk is unimplemented in W15. \
                 Callers MUST gate non-F-contiguous inputs to a serial \
                 fallback before calling par_iter() (see W15T1 Step 5 \
                 entry-side gating). 30-dispatch §5.6 admits non-contig \
                 inputs to Parallel via doubled threshold; this is a real \
                 path that callers must handle, not a dispatch bug."
            )
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.cursor;
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ExactSizeIterator for ParIterSeq<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
}

#[cfg(feature = "parallel")]
impl<'a, A, D> DoubleEndedIterator for ParIterSeq<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        self.end -= 1;
        let i = self.end;
        if self.base.is_f_contiguous() {
            let slice = self
                .base
                .as_slice()
                .expect("ParIterSeq only constructed for F-contiguous tensors");
            // SAFETY: same reasoning as next() — storage lifetime is 'a.
            Some(unsafe { &*(&slice[i] as *const A) })
        } else {
            unimplemented!(
                "non-F-contiguous ParIter walk is unimplemented in W15. \
                 Callers MUST gate non-F-contiguous inputs to a serial \
                 fallback before calling par_iter()."
            )
        }
    }
}

// ── TensorBase::par_iter() — per concrete storage type ──
//
// `view()` is only available on concrete storage types (not on the generic
// `S: Storage` bound). We mirror the existing pattern in `tensor/impls.rs`
// where `view()` is implemented per `Owned` / `ViewRepr` / `ViewMutRepr` /
// `ArcRepr`. The generic `<S: Storage>` bound was aspirational in the design
// but cannot compile with the current tensor module API. A future wave can
// replace these four impls with a single generic impl once `view()` is lifted
// to `Storage`.

use crate::storage::RawStorage;

#[cfg(feature = "parallel")]
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element + Send + Sync,
    D: Dimension + Clone,
{
    pub(crate) fn par_iter(&self) -> ParIter<'_, A, D> {
        debug_assert!(
            self.is_f_contiguous(),
            "par_iter() in W15 supports F-contiguous inputs only; \
             callers must gate non-F-contiguous inputs to serial."
        );
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        let view = unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        };
        ParIter::new(view)
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D>
where
    D: Dimension + Clone,
    A: Element + Send + Sync,
{
    pub(crate) fn par_iter(&self) -> ParIter<'_, A, D> {
        debug_assert!(
            self.is_f_contiguous(),
            "par_iter() in W15 supports F-contiguous inputs only; \
             callers must gate non-F-contiguous inputs to serial."
        );
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        let view = unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                self.derived_from_view_mut,
            )
        };
        ParIter::new(view)
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    A: Element + Send + Sync,
    D: Dimension + Clone,
{
    pub(crate) fn par_iter(&self) -> ParIter<'_, A, D> {
        debug_assert!(
            self.is_f_contiguous(),
            "par_iter() in W15 supports F-contiguous inputs only; \
             callers must gate non-F-contiguous inputs to serial."
        );
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        let view = unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                true,
            )
        };
        ParIter::new(view)
    }
}

#[cfg(feature = "parallel")]
impl<A, D> TensorBase<ArcRepr<A>, D>
where
    A: Element + Send + Sync,
    D: Dimension + Clone,
{
    pub(crate) fn par_iter(&self) -> ParIter<'_, A, D> {
        debug_assert!(
            self.is_f_contiguous(),
            "par_iter() in W15 supports F-contiguous inputs only; \
             callers must gate non-F-contiguous inputs to serial."
        );
        let storage =
            unsafe { ViewRepr::from_raw_parts(self.storage.as_ptr(), self.storage.len()) };
        let view = unsafe {
            TensorBase::new_unchecked(
                storage,
                self.shape.clone(),
                self.strides.clone(),
                self.offset,
                self.flags,
                false,
            )
        };
        ParIter::new(view)
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use crate::dimension::Ix1;
    use crate::layout::Strides;
    use crate::tensor::TensorView;
    use rayon::iter::IndexedParallelIterator;

    #[test]
    fn test_par_iter_len_matches_tensor_len() {
        // 09-parallel §8.2 line 597
        let data = vec![0.0f64; 2048];
        // SAFETY: F-order [2048] view over data; offset = 0; stride 1.
        let tensor = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(2048),
                Strides::from_slice(&[1_usize]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order [2048] view");
        let par_iter = tensor.par_iter();
        assert_eq!(par_iter.len(), 2048);
    }
}
