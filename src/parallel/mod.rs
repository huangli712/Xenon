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
pub(crate) mod binary;
#[cfg(feature = "parallel")]
pub(crate) mod chunks;
#[cfg(feature = "parallel")]
pub(crate) mod dot;
#[cfg(feature = "parallel")]
pub(crate) mod reduce;
#[cfg(feature = "parallel")]
pub(crate) mod sum;
#[cfg(feature = "parallel")]
pub(crate) mod unary;

#[cfg(all(test, feature = "parallel"))]
mod feature_matrix_tests {
    use crate::dimension::{Dimension, Ix1};
    use crate::dispatch::{
        ExecPath, ParallelExecStrategy, ParallelGuard, reset_parallel_threshold, select_exec_path,
        set_parallel_threshold,
    };
    use crate::dispatch::ThresholdTestGuard;
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::parallel::sum::par_sum;
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
