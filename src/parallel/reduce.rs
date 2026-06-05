//! Parallel reduction skeleton.
//!
//! Provides [`par_reduce_impl`], the generic parallel reduction primitive
//! that concrete reductions (e.g. sum) are built on top of.

use crate::dimension::Dimension;
use crate::dispatch::{ParallelExecStrategy, ParallelGuard, with_parallel_worker_context};
use crate::element::Element;
use crate::parallel::chunks::compute_safe_chunks;
use crate::storage::Storage;
use crate::tensor::TensorBase;

/// Generic parallel reduction over a tensor's elements.
///
/// Reduces all elements to a single value using `op`, seeded by `identity`
/// (also returned unchanged for an empty tensor). `op` must be associative
/// and `identity` must be its neutral element, since rayon merges partial
/// results in an unspecified order.
///
/// # Panics
///
/// Panics if `tensor` is not F-contiguous (i.e. `tensor.as_slice()` returns
/// `None`); callers must route non-contiguous / broadcast inputs to the
/// serial path before invoking this function.
#[cfg(feature = "parallel")]
pub(crate) fn par_reduce_impl<S, A, D, F, ID>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
    identity: ID,
    op: F,
) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Send + Sync + Clone,
    F: Fn(A, A) -> A + Send + Sync,
    ID: Fn() -> A + Send + Sync + Clone,
{
    let total = tensor.len();
    if total == 0 {
        return identity();
    }

    let num_threads = strategy
        .max_workers()
        .unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size()
        .unwrap_or_else(|| compute_safe_chunks(total, num_threads));

    let src_slice = tensor
        .as_slice()
        .expect("par_reduce_impl caller must ensure F-contiguous + non-broadcast");

    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .map(|element| {
            #[allow(clippy::clone_on_copy, reason = "A: Clone but not necessarily Copy")]
            with_parallel_worker_context(|| element.clone())
        })
        .reduce(identity, op)
}
