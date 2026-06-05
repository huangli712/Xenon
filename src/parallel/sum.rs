//! Parallel sum reduction.
//!
//! Provides [`par_sum`], the parallel sum of all elements in a tensor.

use crate::dimension::Dimension;
use crate::dispatch::{ParallelExecStrategy, ParallelGuard};
use crate::element::Numeric;
use crate::parallel::reduce::par_reduce_impl;
use crate::storage::Storage;
use crate::tensor::TensorBase;

/// Parallel sum of all elements in a tensor.
///
/// Public (rather than `pub(crate)`) so integration tests can exercise the
/// kernel directly; re-exported through the crate prelude.
#[cfg(feature = "parallel")]
pub fn par_sum<S, A, D>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    guard: ParallelGuard,
) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Send + Sync,
{
    par_reduce_impl(tensor, strategy, guard, || A::zero(), |a, b| a + b)
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::dimension::{Dimension, Ix1};
    use crate::dispatch::ThresholdTestGuard;
    use crate::dispatch::{
        ExecPath, ParallelExecStrategy, ParallelGuard, reset_parallel_threshold, select_exec_path,
        set_parallel_threshold,
    };
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::storage::Storage;
    use crate::tensor::{TensorBase, TensorView};

    /// Force the parallel path (via `set_parallel_threshold(1)`) and return
    /// its guard. Panics if a contaminated `IN_PARALLEL` TLS prevents the
    /// parallel path from being selected, since these tests require it.
    fn acquire_guard<S, D, A>(t: &TensorBase<S, D>) -> ParallelGuard
    where
        S: Storage<Elem = A>,
        D: Dimension,
        A: Element,
    {
        let (path, g) = select_exec_path(t.len(), t.is_f_contiguous(), t.is_aligned());
        if !matches!(path, ExecPath::Parallel) {
            // IN_PARALLEL TLS may be contaminated from a prior test.
            // Return a dummy sentinel; the caller should check.
            panic!(
                "select_exec_path returned {:?}, not Parallel. \
                 IN_PARALLEL TLS may be contaminated from a prior test. \
                 Run this test with `-- --test-threads=1` or isolate it.",
                path
            );
        }
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

    /// `par_sum` matches the serial sum and returns the identity (0) for an
    /// empty tensor.
    #[test]
    fn test_par_sum_serial_match_and_empty_identity() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);

        let data: Vec<f64> = (0..2048).map(|i| i as f64).collect();
        let tensor = unsafe { view_1d_f64(&data) };
        let strategy = ParallelExecStrategy::auto();
        {
            let guard = acquire_guard(&tensor);
            let par_result = par_sum(&tensor, &strategy, guard);
            let serial_result: f64 = data.iter().sum();
            assert!((par_result - serial_result).abs() < 1e-10 * serial_result.abs().max(1.0));
        }

        let empty: Vec<f64> = Vec::new();
        let tensor_empty = unsafe { view_1d_f64(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d_f64(&one_data) };
        {
            let guard = acquire_guard(&one);
            let result = par_sum(&tensor_empty, &strategy, guard);
            assert_eq!(result, 0.0f64);
        }

        reset_parallel_threshold();
    }
}
