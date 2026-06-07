//! Parallel reduction skeleton.
//!
//! Provides [`par_reduce_impl`], the generic parallel reduction primitive
//! that concrete reductions (e.g. sum) are built on top of.

use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::TensorBase;

use crate::dispatch::{ParallelExecStrategy, ParallelGuard};
use crate::dispatch::{with_parallel_worker_context};
use super::chunks::compute_safe_chunks;

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
    use rayon::iter::{
        IndexedParallelIterator,
        IntoParallelRefIterator,
        ParallelIterator
    };

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

    src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .map(|element| {
            #[allow(clippy::clone_on_copy)]
            with_parallel_worker_context(|| element.clone())
        })
        .reduce(identity, op)
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    use crate::dimension::{Dimension, Ix1};
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::storage::Storage;
    use crate::tensor::{TensorBase, TensorView};

    use crate::dispatch::ThresholdTestGuard;
    use crate::dispatch::{ExecPath, select_exec_path};
    use crate::dispatch::{ParallelExecStrategy, ParallelGuard};
    use crate::dispatch::{reset_parallel_threshold, set_parallel_threshold};

    /// Force the parallel path (via `set_parallel_threshold(1)`) and return
    /// its guard, asserting the parallel path was actually selected.
    fn acquire_guard<S, D, A>(t: &TensorBase<S, D>) -> ParallelGuard
    where
        S: Storage<Elem = A>,
        D: Dimension,
        A: Element,
    {
        let (path, g) = select_exec_path(t.len(), t.is_f_contiguous(), t.is_aligned());
        assert_eq!(path, ExecPath::Parallel);
        g.expect("Parallel implies Some(guard)")
    }

    /// Build a 1-D F-order `f64` view over `data` for test inputs.
    unsafe fn view_1d_f64<'a>(data: &'a [f64]) -> TensorView<'a, f64, Ix1> {
        // SAFETY: caller ensures data is a valid F-order 1-D contiguous slice.
        unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(data.len()),
                Strides::from_slice(&[1_usize]).expect("valid F-order strides for test"),
                0,
            )
            .expect("valid F-order 1-D f64 view")
        }
    }

    /// `par_reduce_impl` works with a non-additive operator (max), proving it
    /// is not implicitly coupled to summation.
    #[test]
    fn test_par_reduce_impl_max_op() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);

        let data: Vec<f64> = (0..2048).map(|i| (i as f64 * 7.0) % 101.0).collect();
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_guard(&tensor);
        let par_max =
            par_reduce_impl(&tensor, &strategy, guard, || f64::NEG_INFINITY, |a, b| a.max(b));
        let serial_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(par_max, serial_max);

        reset_parallel_threshold();
    }

    /// `par_reduce_impl` returns the identity for an empty tensor regardless
    /// of the operator.
    #[test]
    fn test_par_reduce_impl_empty_returns_identity() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);

        let empty: Vec<f64> = Vec::new();
        let tensor_empty = unsafe { view_1d_f64(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d_f64(&one_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_guard(&one);
        let result =
            par_reduce_impl(&tensor_empty, &strategy, guard, || f64::NEG_INFINITY, |a, b| a.max(b));
        assert_eq!(result, f64::NEG_INFINITY);

        reset_parallel_threshold();
    }
}
