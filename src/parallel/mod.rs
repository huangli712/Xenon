//! Parallel backend module. Only compiled with the `parallel` feature.

/// Parallel reduction skeleton (`par_reduce_impl`).
#[cfg(feature = "parallel")]
pub(crate) mod reduce;

/// Parallel sum reduction (`par_sum`).
#[cfg(feature = "parallel")]
pub(crate) mod sum;

#[cfg(all(test, feature = "parallel"))]
mod feature_matrix_tests {
    use crate::dimension::{Dimension, Ix1};
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::storage::Storage;
    use crate::tensor::{TensorBase, TensorView};

    use crate::dispatch::ThresholdTestGuard;
    use crate::dispatch::{ExecPath, select_exec_path};
    use crate::dispatch::{ParallelExecStrategy, ParallelGuard};
    use crate::dispatch::{reset_parallel_threshold, set_parallel_threshold};

    use super::sum::par_sum;

    /// Force the parallel path and return its guard, panicking if the
    /// parallel path was not selected.
    fn acquire_parallel_guard<S, D, A>(t: &TensorBase<S, D>) -> ParallelGuard
    where
        S: Storage<Elem = A>,
        D: Dimension,
        A: Element,
    {
        let (path, g) = select_exec_path(
            t.len(),
            t.is_f_contiguous(),
            t.is_aligned()
        );
        if !matches!(path, ExecPath::Parallel) {
            panic!("select_exec_path returned {:?}, not Parallel", path);
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
                Strides::from_slice(&[1_usize])
                    .expect("valid F-order strides for test"),
                0,
            ).expect("valid F-order 1-D f64 view")
        }
    }

    /// Single-worker and multi-worker `par_sum` agree with the serial sum.
    #[test]
    fn test_par_sum_single_and_multi_worker_agree() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data: Vec<f64> = (0..2048).map(|i| i as f64).collect();
        let tensor = unsafe { view_1d_f64(&data) };

        let strategy_single = ParallelExecStrategy::new(None, Some(1))
            .expect("valid strategy with max_workers=1");
        let guard_single = acquire_parallel_guard(&tensor);
        let sum_single = par_sum(&tensor, &strategy_single, guard_single);

        let strategy_multi = ParallelExecStrategy::auto();
        let guard_multi = acquire_parallel_guard(&tensor);
        let sum_multi = par_sum(&tensor, &strategy_multi, guard_multi);

        let serial: f64 = data.iter().sum();
        assert!((sum_single - serial).abs() < 1e-10 * serial.abs().max(1.0));
        assert!((sum_multi - serial).abs() < 1e-10 * serial.abs().max(1.0));
        reset_parallel_threshold();
    }
}
