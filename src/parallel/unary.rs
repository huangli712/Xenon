//! Single-input parallel element-wise maps.
//!
//! W15T2: par_map — single-input parallel element-wise map.
//! W15T7: par_map_checked — two-pass checked parallel map with error + panic propagation.

use crate::dimension::Dimension;
use crate::dispatch::{ParallelExecStrategy, ParallelGuard, with_parallel_worker_context};
use crate::element::Element;
use crate::error::XenonError;
use crate::parallel::compute_safe_chunks;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Parallel element-wise map.
///
/// Test-only visibility: re-exported at crate root under `#[doc(hidden)]`.
///
/// # Panics
///
/// Panics if `tensor` is not F-contiguous (i.e. `tensor.as_slice()` returns
/// `None`). Callers must route non-contiguous / broadcast inputs to the
/// Serial path via `dispatch::select_exec_path` before invoking `par_map`.
#[cfg(feature = "parallel")]
pub fn par_map<S, A, B, D, F>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
    f: F,
) -> Tensor<B, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Send + Sync,
{
    let total = tensor.len();
    let num_threads = strategy
        .max_workers()
        .unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size()
        .unwrap_or_else(|| compute_safe_chunks(total, num_threads));

    // par_map is a single-input element-wise operation. The caller
    // (dispatch) ensures the tensor is F-contiguous before routing
    // to Parallel. We use the underlying contiguous slice directly
    // via rayon's par_iter() on slices.
    let src_slice = tensor.as_slice().expect(
        "par_map caller must ensure F-contiguous + non-broadcast; \
         dispatch gates non-contiguous inputs to Serial",
    );

    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    let mut output_data: Vec<B> = Vec::with_capacity(total);
    src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .map(|src| {
            // Worker TLS: IN_PARALLEL = true for the duration of this map call.
            // Nested select_exec_path() inside f() will fall back to Serial.
            with_parallel_worker_context(|| f(src))
        })
        .collect_into_vec(&mut output_data);

    // SAFETY (07-tensor 5 from_raw_vec_unchecked precondition):
    //   - output_data.len() == total == checked_size(tensor.raw_dim())
    //     (collect_into_vec on an IndexedParallelIterator with len() = total)
    //   - F-order index alignment guaranteed by IndexedParallelIterator
    //     (09-parallel 6.7 line 504-505)
    unsafe { Tensor::from_raw_vec_unchecked(output_data, tensor.raw_dim()) }
}

// ── W15T7: par_map_checked ──

#[cfg(feature = "parallel")]
#[allow(
    dead_code,
    reason = "W15T7 reserved API per 09-parallel.md §10 line 700 — checked \
              parallel map with error/panic propagation. Implementation and \
              tests are complete; no production caller yet (no operation \
              currently needs a fallible element mapper). Kept under the \
              W15 deliverable contract; remove this attribute when a caller \
              is wired in. (`allow` rather than `expect` because dead_code \
              only fires without `--tests`; test-mode use suppresses the \
              lint, so `expect` would be unfulfilled.)"
)]
pub(crate) fn par_map_checked<S, A, B, D, F>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
    f: F,
) -> Result<Tensor<B, D>, XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> Result<B, XenonError> + Send + Sync,
{
    let total = tensor.len();
    let num_threads = strategy
        .max_workers()
        .unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size()
        .unwrap_or_else(|| compute_safe_chunks(total, num_threads));

    let src_slice = tensor
        .as_slice()
        .expect("par_map_checked caller must ensure F-contiguous + non-broadcast");

    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    // Phase 1: error probe via try_for_each. If any element returns Err,
    // bail without scheduling the second collect pass. Rayon does NOT
    // guarantee returning the lowest-index Err; design (§10 line 700)
    // only requires that at least one Err is propagated.
    let probe: Result<(), XenonError> = src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .try_for_each(|item| with_parallel_worker_context(|| f(item).map(|_| ())));
    probe?;

    // Phase 2: success path — indexed collect into pre-sized Vec<B>.
    // IndexedParallelIterator + collect_into_vec writes results by F-order
    // logical index, regardless of worker completion order (§6.7 line 504-505).
    let mut out: Vec<B> = Vec::with_capacity(total);
    src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .map(|item| {
            with_parallel_worker_context(|| {
                f(item).expect(
                    "internal precondition violation: f returned Err on \
                     phase 2 after phase 1 probe passed; f must be \
                     deterministic + side-effect free (09-parallel 6.6)",
                )
            })
        })
        .collect_into_vec(&mut out);

    // SAFETY (07-tensor 5 from_raw_vec_unchecked precondition):
    //   - out.len() == total == checked_size(tensor.raw_dim())
    //     (IndexedParallelIterator len + collect_into_vec)
    //   - F-order alignment (§6.7 line 504-505)
    Ok(unsafe { Tensor::from_raw_vec_unchecked(out, tensor.raw_dim()) })
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::dimension::Ix1;
    use crate::dispatch::{
        ExecPath, ParallelExecStrategy, reset_parallel_threshold, select_exec_path,
        set_parallel_threshold,
    };
    use crate::dispatch::ThresholdTestGuard;
    use crate::error::InvalidArgumentKind;
    use crate::layout::Strides;
    use crate::tensor::TensorView;
    use std::borrow::Cow;

    fn acquire_parallel_guard<S, D, A>(t: &TensorBase<S, D>) -> ParallelGuard
    where
        S: Storage<Elem = A>,
        D: Dimension,
        A: Element,
    {
        let (path, g) = select_exec_path(t.len(), t.is_f_contiguous(), t.is_aligned());
        assert_eq!(path, ExecPath::Parallel);
        g.expect("Parallel implies Some(guard)")
    }

    /// SAFETY helper: build a 1-D F-order f64 view over a slice.
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

    #[test]
    fn test_par_map_parallel_path() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);

        let data = [1.0f64, 2.0, 3.0, 4.0];
        let tensor = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(4),
                Strides::from_slice(&[1_usize]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order [4] view");
        let (path, guard_opt) =
            select_exec_path(tensor.len(), tensor.is_f_contiguous(), tensor.is_aligned());
        assert_eq!(path, ExecPath::Parallel);
        let guard = guard_opt.expect("Parallel implies Some(guard) by 30-dispatch 5.5");

        let strategy = ParallelExecStrategy::auto();
        let result = par_map(&tensor, &strategy, guard, |v| v * 2.0);
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[2.0, 4.0, 6.0, 8.0]
        );

        reset_parallel_threshold();
    }

    #[test]
    fn test_parallel_feature_matrix_single_worker() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy =
            ParallelExecStrategy::new(None, Some(1)).expect("valid strategy with max_workers=1");
        let guard = acquire_parallel_guard(&tensor);
        let result = par_map(&tensor, &strategy, guard, |v| v * 2.0);
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[2.0, 4.0, 6.0, 8.0]
        );
        reset_parallel_threshold();
    }

    #[test]
    fn test_parallel_feature_matrix_default_workers() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&tensor);
        let result = par_map(&tensor, &strategy, guard, |v| v * 2.0);
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[2.0, 4.0, 6.0, 8.0]
        );
        reset_parallel_threshold();
    }

    #[test]
    fn test_par_map_checked_matches_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&tensor);
        let result = par_map_checked(&tensor, &strategy, guard, |v| Ok(v * 2.0))
            .expect("par_map_checked should succeed for valid test input");
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[2.0, 4.0, 6.0, 8.0]
        );
        reset_parallel_threshold();
    }

    #[test]
    fn test_parallel_error_propagation() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&tensor);
        let result = par_map_checked(&tensor, &strategy, guard, |v| {
            if *v == 3.0 {
                Err(XenonError::InvalidArgument {
                    operation: Cow::Borrowed("test"),
                    kind: InvalidArgumentKind::NumericOutOfRange {
                        argument: Cow::Borrowed("v"),
                        domain: Cow::Borrowed("[0, 2]"),
                        actual: Cow::Borrowed("3"),
                    },
                })
            } else {
                Ok(v * 2.0)
            }
        });
        assert!(result.is_err());
        reset_parallel_threshold();
    }

    #[test]
    #[should_panic]
    fn test_parallel_panic_propagation() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&tensor);
        let _ = par_map_checked(&tensor, &strategy, guard, |v| {
            if *v == 3.0 {
                panic!("panic in worker");
            }
            Ok(v * 2.0)
        });
        reset_parallel_threshold();
    }
}
