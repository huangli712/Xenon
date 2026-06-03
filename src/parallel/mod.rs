//! Parallel backend module. Only compiled with `--features parallel`.
//! See `docs/design/09-parallel.md` for the full design.
//!
//! # Compile-fail test: bool does not implement Numeric (par_sum / par_dot)
//!
//! 03-element §5.2 line 707-721: `bool` implements `Element` only, NOT
//! `Numeric`. W15T4 `par_sum` and W15T5 `par_dot` require `A: Numeric`,
//! so any attempt to call them on a `bool` tensor must fail to compile.
//! This satisfies 09-parallel.md §8.7 ("type boundary / compile-time test").
//!
//! ```compile_fail
//! use xenon::tensor::TensorBase;
//! use xenon::storage::Owned;
//! use xenon::dimension::Ix1;
//! use xenon::dispatch::{select_exec_path, ParallelExecStrategy};
//! use xenon::par_sum;
//! let t: TensorBase<Owned<bool>, Ix1> = TensorBase::from_shape_vec(
//!     Ix1(2), vec![true, false],
//! ).unwrap();
//! let (_p, g) = select_exec_path(
//!     t.len(), t.is_f_contiguous(), t.is_aligned(),
//! );
//! let _ = par_sum(&t, &ParallelExecStrategy::auto(), g.unwrap());
//! // ^^^^^^^ trait bound `bool: Numeric` is not satisfied
//! ```

#[cfg(feature = "parallel")]
pub(crate) mod checked;
#[cfg(feature = "parallel")]
pub(crate) mod map;
#[cfg(feature = "parallel")]
pub(crate) mod reduce;

/// Compute a per-chunk element count for parallel splitting.
///
/// Algorithm (09-parallel §6.3):
/// - `MIN_CHUNK = 1024`           : lower bound to amortize rayon scheduling.
/// - `TARGET_CHUNKS_PER_WORKER=4` : give work-stealing slack for tail balance.
/// - `total == 0`        → returns 1 (dummy; no work will be scheduled).
/// - `total <= workers`  → returns 1 (one element per worker; rest idle).
/// - otherwise           → max(ceil_div(total, workers*4), MIN_CHUNK).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn compute_safe_chunks(total: usize, num_workers: usize) -> usize {
    const MIN_CHUNK: usize = 1024;
    const TARGET_CHUNKS_PER_WORKER: usize = 4;

    if total == 0 {
        return 1;
    }
    if total <= num_workers {
        return 1;
    }
    let target_chunks = num_workers.saturating_mul(TARGET_CHUNKS_PER_WORKER);
    let raw = total.div_ceil(target_chunks);
    raw.max(MIN_CHUNK)
}

#[cfg(test)]
mod chunk_tests {
    use super::compute_safe_chunks;

    #[test]
    fn test_compute_safe_chunks_empty_returns_one() {
        // 09-parallel §6.3 line 350: total == 0 → 1 (dummy)
        assert_eq!(compute_safe_chunks(0, 8), 1);
    }

    #[test]
    fn test_compute_safe_chunks_few_elements_returns_one() {
        // 09-parallel §6.3 line 353: total <= workers → 1
        assert_eq!(compute_safe_chunks(5, 8), 1);
    }

    #[test]
    fn test_compute_safe_chunks_respects_min_chunk() {
        // total=10_000, workers=8 → ceil(10_000 / 32) = 313 < MIN_CHUNK(1024)
        // → returns 1024
        assert_eq!(compute_safe_chunks(10_000, 8), 1024);
    }

    #[test]
    fn test_compute_safe_chunks_large_total_overrides_min() {
        // total=1_000_000, workers=8 → ceil(1_000_000 / 32) = 31_250 > 1024
        // → returns 31_250
        assert_eq!(compute_safe_chunks(1_000_000, 8), 31_250);
    }
}

#[cfg(all(test, feature = "parallel"))]
mod feature_matrix_tests {
    use crate::dimension::{Dimension, Ix1};
    use crate::dispatch::{
        ExecPath, ParallelExecStrategy, ParallelGuard, reset_parallel_threshold, select_exec_path,
        set_parallel_threshold,
    };
    use crate::dispatch::test_support::ThresholdTestGuard;
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::parallel::map::par_map;
    use crate::parallel::reduce::par_sum;
    use crate::storage::Storage;
    use crate::tensor::{TensorBase, TensorView};

    fn acquire_parallel_guard<S, D, A>(t: &TensorBase<S, D>) -> ParallelGuard
    where
        S: Storage<Elem = A>,
        D: Dimension,
        A: Element,
    {
        let (path, g) = select_exec_path(t.len(), t.is_f_contiguous(), t.is_aligned());
        if !matches!(path, ExecPath::Parallel) {
            panic!("select_exec_path returned {:?}, not Parallel", path);
        }
        g.expect("Parallel implies Some(guard)")
    }

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
    fn test_parallel_single_and_multi_worker_results_agree() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let data: Vec<f64> = (0..2048).map(|i| i as f64).collect();
        let tensor = unsafe { view_1d_f64(&data) };

        let strategy_single =
            ParallelExecStrategy::new(None, Some(1)).expect("valid strategy with max_workers=1");
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
