//! Single-input parallel element-wise maps.
//!
//! - [`par_map`] — infallible single-input element-wise map.
//! - [`par_map_checked`] — fallible variant whose closure returns `Result`,
//!   with error + panic propagation.

use crate::error::XenonError;
use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

use crate::dispatch::{ParallelExecStrategy, ParallelGuard};
use crate::dispatch::{with_parallel_worker_context};
use super::chunks::compute_safe_chunks;

/// Infallible parallel element-wise map.
///
/// Applies `f` to every element and collects the results in logical order.
/// Public (rather than `pub(crate)`) so integration tests can exercise the
/// kernel directly; re-exported through the crate prelude.
///
/// # Panics
///
/// Panics if `tensor` is not F-contiguous (i.e. `tensor.as_slice()` returns
/// `None`). Callers must route non-contiguous / broadcast inputs to the
/// serial path before invoking `par_map`.
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
    use rayon::iter::{
        IndexedParallelIterator,
        IntoParallelRefIterator,
        ParallelIterator
    };

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

    // SAFETY: from_raw_vec_unchecked requires the Vec length and F-order
    // layout to match the dimension.
    //   - output_data.len() == total == checked_size(tensor.raw_dim())
    //     (collect_into_vec on an IndexedParallelIterator with len() = total)
    //   - F-order index alignment guaranteed by IndexedParallelIterator
    unsafe {
        Tensor::from_raw_vec_unchecked(output_data, tensor.raw_dim())
    }
}

/// Fallible parallel element-wise map.
///
/// The single-input counterpart of `par_zip_checked`: `f` may return `Err`.
/// Uses a two-pass strategy — an error probe followed by an indexed collect —
/// so success results land in logical order regardless of worker completion
/// order. Public (rather than `pub(crate)`) so integration tests can exercise
/// the kernel directly; re-exported through the crate prelude.
///
/// # Errors
///
/// Returns an `Err` from `f` if any element produces one (rayon does not
/// guarantee which error is returned when multiple elements fail); no result
/// tensor is produced in that case.
///
/// # Panics
///
/// Panics if `tensor` is not F-contiguous (callers must route non-contiguous
/// or broadcast inputs to the serial path), or if a worker closure `f` panics
/// (the panic propagates out of the parallel region).
#[cfg(feature = "parallel")]
pub fn par_map_checked<S, A, B, D, F>(
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
    use rayon::iter::{
        IndexedParallelIterator,
        IntoParallelRefIterator,
        ParallelIterator
    };

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

    // Phase 1: error probe via try_for_each. If any element returns Err,
    // bail without scheduling the second collect pass. Rayon does NOT
    // guarantee returning the lowest-index Err; we only require that at
    // least one Err is propagated.
    let probe: Result<(), XenonError> = src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .try_for_each(|item| with_parallel_worker_context(|| f(item).map(|_| ())));
    probe?;

    // Phase 2: success path — indexed collect into pre-sized Vec<B>.
    // IndexedParallelIterator + collect_into_vec writes results by F-order
    // logical index, regardless of worker completion order.
    let mut out: Vec<B> = Vec::with_capacity(total);
    src_slice
        .par_iter()
        .with_min_len(chunk_size)
        .map(|item| {
            with_parallel_worker_context(|| {
                f(item).expect(
                    "internal precondition violation: f returned Err on \
                     phase 2 after phase 1 probe passed; f must be \
                     deterministic + side-effect free",
                )
            })
        })
        .collect_into_vec(&mut out);

    // SAFETY: from_raw_vec_unchecked requires the Vec length and F-order
    // layout to match the dimension.
    //   - out.len() == total == checked_size(tensor.raw_dim())
    //     (IndexedParallelIterator len + collect_into_vec)
    //   - F-order alignment guaranteed by IndexedParallelIterator
    Ok(unsafe {
        Tensor::from_raw_vec_unchecked(out, tensor.raw_dim())
    })
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    use std::borrow::Cow;

    use crate::error::InvalidArgumentKind;
    use crate::dimension::Ix1;
    use crate::layout::Strides;
    use crate::tensor::TensorView;

    use crate::dispatch::ThresholdTestGuard;
    use crate::dispatch::{ExecPath, select_exec_path};
    use crate::dispatch::ParallelExecStrategy;
    use crate::dispatch::{reset_parallel_threshold, set_parallel_threshold};
    
    /// Force the parallel path and return its guard, asserting the parallel
    /// path was actually selected.
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

    /// `par_map` runs on the parallel path and doubles every element.
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
        let guard = guard_opt.expect("Parallel implies Some(guard)");

        let strategy = ParallelExecStrategy::auto();
        let result = par_map(&tensor, &strategy, guard, |v| v * 2.0);
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[2.0, 4.0, 6.0, 8.0]
        );

        reset_parallel_threshold();
    }

    /// `par_map` produces correct results with a single worker.
    #[test]
    fn test_par_map_single_worker() {
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

    /// `par_map` produces correct results with the default worker count.
    #[test]
    fn test_par_map_default_workers() {
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

    /// `par_map` supports a type-changing closure `f64 -> i64`.
    #[test]
    fn test_par_map_type_changing_closure() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data = [1.5f64, 2.5, 3.5, 4.5];
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&tensor);
        let result = par_map(&tensor, &strategy, guard, |v| *v as i64);
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[1i64, 2, 3, 4]
        );
        reset_parallel_threshold();
    }

    /// `par_map` returns an empty tensor for an empty input.
    #[test]
    fn test_par_map_empty() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let empty: Vec<f64> = Vec::new();
        let tensor_empty = unsafe { view_1d_f64(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d_f64(&one_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&one);
        let result = par_map(&tensor_empty, &strategy, guard, |v| v * 2.0);
        assert_eq!(result.len(), 0);
        reset_parallel_threshold();
    }

    /// `par_map_checked` matches the serial result when the closure succeeds.
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

    /// `par_map_checked` propagates a closure `Err` as an overall `Err`.
    #[test]
    fn test_par_map_checked_error_propagation() {
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

    /// `par_map_checked` propagates a worker panic as a panic.
    #[test]
    #[should_panic]
    fn test_par_map_checked_panic_propagation() {
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

    /// `par_map_checked` returns an empty tensor for an empty input.
    #[test]
    fn test_par_map_checked_empty() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let empty: Vec<f64> = Vec::new();
        let tensor_empty = unsafe { view_1d_f64(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d_f64(&one_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&one);
        let result = par_map_checked(&tensor_empty, &strategy, guard, |v| Ok(v * 2.0))
            .expect("empty par_map_checked should succeed");
        assert_eq!(result.len(), 0);
        reset_parallel_threshold();
    }
}
