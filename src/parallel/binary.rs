//! Dual-input parallel element-wise maps.
//!
//! - [`par_zip`] — infallible dual-input broadcast element-wise map.
//! - [`par_zip_checked`] — fallible variant whose closure returns `Result`,
//!   with error + panic propagation.

use std::borrow::Cow;

use crate::error::{InvalidShapeKind, XenonError};
use crate::dimension::Dimension;
use crate::element::Element;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

use crate::dispatch::{ParallelExecStrategy, ParallelGuard};
use crate::dispatch::{with_parallel_worker_context};
use super::chunks::compute_safe_chunks;

/// Infallible dual-input broadcast element-wise parallel map.
///
/// The dual-input counterpart of `par_map`: the closure cannot fail
/// (`Fn(&A, &B) -> C`), so results are collected directly into `Vec<C>`
/// without the per-element `Result` buffering and second aggregation pass
/// that `par_zip_checked` performs. Public (rather than `pub(crate)`) so
/// integration tests can exercise the kernel directly; re-exported through
/// the crate prelude.
///
/// # Panics
///
/// Panics if `output_dim.checked_size()` overflows `usize`, or if either
/// input is not broadcast-compatible with `output_dim`. Both are caller
/// (math layer) preconditions; a violation is an internal bug.
#[cfg(feature = "parallel")]
pub fn par_zip<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
    f: F,
) -> Tensor<C, DO>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = B>,
    DL: Dimension + Clone,
    DR: Dimension + Clone,
    DO: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> C + Send + Sync,
{
    use rayon::iter::{
        IndexedParallelIterator,
        IntoParallelIterator,
        ParallelIterator
    };

    // checked_size overflow is an internal bug: the math layer validates the
    // broadcast output shape before routing here (mirrors broadcast_to below).
    let total = output_dim.checked_size().expect(
        "par_zip caller (math layer) must pass a valid output_dim; \
         shape product overflow is an internal bug",
    );

    let num_threads = strategy
        .max_workers()
        .unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size()
        .unwrap_or_else(|| compute_safe_chunks(total, num_threads));

    // Broadcast-compatible read-only views: math layer ensures both inputs
    // already broadcast against output_dim.
    let lhs_view = lhs
        .broadcast_to(output_dim.clone())
        .expect("math layer ensures broadcast compatibility; \
                 violation is an internal bug");
    let rhs_view = rhs
        .broadcast_to(output_dim.clone())
        .expect("math layer ensures broadcast compatibility; \
                 violation is an internal bug");

    // Pre-compute output shape for F-order index -> multi-dim coord conversion.
    let out_shape = output_dim.slice();
    let ndim = out_shape.len();
    let mut strides_f = vec![1usize; ndim];
    for k in 1..ndim {
        strides_f[k] = strides_f[k - 1] * out_shape[k - 1];
    }

    // Infallible: collect directly into Vec<C> (no Result buffering / second pass).
    let mut output_data: Vec<C> = Vec::with_capacity(total);
    (0..total)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|i| {
            with_parallel_worker_context(|| {
                // F-order logical index -> multi-dim coord.
                let mut coord = vec![0usize; ndim];
                let remainder = i;
                for k in 0..ndim {
                    coord[k] = remainder / strides_f[k] % out_shape[k];
                }

                // SAFETY: coord is computed from i in [0, total), where
                // total == product(out_shape). Each coord[k] is bounded by
                // out_shape[k] via modulo. lhs_view/rhs_view are broadcast-
                // compatible with output_dim (math layer precondition), so
                // coord is valid for both views.
                let a = unsafe { lhs_view.get_unchecked(&coord) };
                let b = unsafe { rhs_view.get_unchecked(&coord) };
                f(a, b)
            })
        })
        .collect_into_vec(&mut output_data);

    // SAFETY: from_raw_vec_unchecked requires the Vec length and F-order
    // layout to match the dimension.
    //   - output_data.len() == total == output_dim.checked_size() (validated above)
    //   - F-order alignment: (0..total).into_par_iter() + collect_into_vec
    //     preserves index -> slot mapping
    unsafe {
        Tensor::from_raw_vec_unchecked(output_data, output_dim.clone())
    }
}

/// Fallible dual-input broadcast element-wise parallel map.
///
/// Like `par_zip` but the closure may return `Err`. Results are buffered per
/// element and aggregated in logical order on the success path. Public
/// (rather than `pub(crate)`) so integration tests can exercise the kernel
/// directly; re-exported through the crate prelude.
///
/// # Errors
///
/// Returns [`XenonError::InvalidShape`] if `output_dim.checked_size()`
/// overflows `usize`, or an `Err` from `f` if any element produces one
/// (rayon does not guarantee which error is returned when multiple elements
/// fail); no result tensor is produced in either case.
///
/// # Panics
///
/// Panics if either input is not broadcast-compatible with `output_dim`
/// (a math-layer precondition; a violation is an internal bug), or if a
/// worker closure `f` panics (the panic propagates out of the parallel
/// region).
#[cfg(feature = "parallel")]
pub fn par_zip_checked<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
    f: F,
) -> Result<Tensor<C, DO>, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = B>,
    DL: Dimension + Clone,
    DR: Dimension + Clone,
    DO: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> Result<C, XenonError> + Send + Sync,
{
    use rayon::iter::{
        IndexedParallelIterator,
        IntoParallelIterator,
        ParallelIterator
    };

    // checked_size overflow -> InvalidShape with ProductOverflow
    let total = output_dim
        .checked_size()
        .map_err(|_| XenonError::InvalidShape {
            operation: Cow::Borrowed("par_zip_checked"),
            shape: output_dim.slice().to_vec(),
            kind: InvalidShapeKind::ProductOverflow,
            offending_dim: None,
        })?;

    let num_threads = strategy
        .max_workers()
        .unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size()
        .unwrap_or_else(|| compute_safe_chunks(total, num_threads));

    // Broadcast-compatible read-only views: math layer ensures both inputs
    // already broadcast against output_dim.
    let lhs_view = lhs
        .broadcast_to(output_dim.clone())
        .expect("math layer ensures broadcast compatibility; \
                violation is an internal bug");
    let rhs_view = rhs
        .broadcast_to(output_dim.clone())
        .expect("math layer ensures broadcast compatibility; \
                 violation is an internal bug");

    // Pre-compute output shape for F-order index -> multi-dim coord conversion.
    let out_shape = output_dim.slice();
    let ndim = out_shape.len();
    let mut strides_f = vec![1usize; ndim];
    for k in 1..ndim {
        strides_f[k] = strides_f[k - 1] * out_shape[k - 1];
    }

    let mut output_data: Vec<Result<C, XenonError>> = Vec::with_capacity(total);
    (0..total)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|i| {
            with_parallel_worker_context(|| {
                // F-order logical index -> multi-dim coord.
                let mut coord = vec![0usize; ndim];
                let remainder = i;
                for k in 0..ndim {
                    coord[k] = remainder / strides_f[k] % out_shape[k];
                }

                // SAFETY: coord is computed from i in [0, total), where
                // total == product(out_shape). Each coord[k] is bounded by
                // out_shape[k] via modulo. lhs_view/rhs_view are broadcast-
                // compatible with output_dim (math layer precondition), so
                // coord is valid for both views.
                let a = unsafe { lhs_view.get_unchecked(&coord) };
                let b = unsafe { rhs_view.get_unchecked(&coord) };
                f(a, b)
            })
        })
        .collect_into_vec(&mut output_data);

    // Aggregate: first Err observed wins; success path requires all-Ok.
    let mut succeeded: Vec<C> = Vec::with_capacity(total);
    for r in output_data {
        succeeded.push(r?);
    }

    // SAFETY: from_raw_vec_unchecked requires the Vec length and F-order
    // layout to match the dimension.
    //   - succeeded.len() == total == output_dim.checked_size() (validated above)
    //   - F-order alignment: (0..total).into_par_iter() + collect_into_vec
    //     preserves index -> slot mapping
    Ok(unsafe {
        Tensor::from_raw_vec_unchecked(succeeded, output_dim.clone())
    })
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    use crate::error::InvalidArgumentKind;
    use crate::dimension::{Ix1, Ix2};
    use crate::layout::Strides;
    use crate::tensor::TensorView;
    
    use crate::dispatch::ThresholdTestGuard;
    use crate::dispatch::{ExecPath, select_exec_path};
    use crate::dispatch::{ParallelExecStrategy};
    use crate::dispatch::{reset_parallel_threshold, set_parallel_threshold};

    /// Force the parallel path and return its guard, asserting the parallel
    /// path was actually selected.
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
        assert_eq!(path, ExecPath::Parallel);
        g.expect("Parallel implies Some(guard)")
    }

    /// Build a 1-D F-order view over `data` for test inputs.
    unsafe fn view_1d<'a, A: Element>(data: &'a [A]) -> TensorView<'a, A, Ix1> {
        // SAFETY: caller ensures data is a valid F-order 1-D contiguous slice.
        unsafe {
            TensorView::<A, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(data.len()),
                Strides::from_slice(&[1_usize])
                    .expect("valid F-order strides for test"),
                0,
            ).expect("valid F-order 1-D view")
        }
    }

    /// `par_zip` matches the serial element-wise add.
    #[test]
    fn test_par_zip_matches_serial_add() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let rhs_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let lhs = unsafe { view_1d(&lhs_data) };
        let rhs = unsafe { view_1d(&rhs_data) };
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip(
            &lhs,
            &rhs,
            &output_dim,
            &strategy,
            guard,
            |a, b| a + b
        );
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 22.0, 33.0, 44.0]
        );
        reset_parallel_threshold();
    }

    /// `par_zip` broadcasts a length-1 rhs against the lhs.
    #[test]
    fn test_par_zip_broadcast_rhs_scalar() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        // Length-1 broadcasted to length-4. The math layer normally
        // produces this via broadcast_to; here we construct an explicit
        // stride-0 view over a single-element backing buffer.
        let rhs_data = [10.0f64];
        let lhs = unsafe { view_1d(&lhs_data) };
        // SAFETY: shape [4], stride [0], storage_len 1 = broadcast view.
        let rhs = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                rhs_data.as_ptr(),
                rhs_data.len(),
                Ix1(4),
                Strides::from_slice(&[0_usize])
                    .expect("valid broadcast strides for test"),
                0,
            )
        }.expect("valid broadcast view");
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip(
            &lhs,
            &rhs,
            &output_dim,
            &strategy,
            guard,
            |a, b| a + b
        );
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 12.0, 13.0, 14.0]
        );
        reset_parallel_threshold();
    }

    /// `par_zip` exercises the F-order index -> multi-dim coord path with a
    /// 2-D broadcast: `[3,1]` + `[1,4]` -> `[3,4]`.
    #[test]
    fn test_par_zip_multidim_broadcast() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        // lhs: shape [3,1], F-order strides [1,3]
        let lhs_data = [1.0f64, 2.0, 3.0];
        let lhs = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                lhs_data.as_ptr(),
                lhs_data.len(),
                Ix2(3, 1),
                Strides::from_slice(&[1_usize, 3]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order [3,1] view");
        // rhs: shape [1,4], F-order strides [1,1]
        let rhs_data = [10.0f64, 20.0, 30.0, 40.0];
        let rhs = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                rhs_data.as_ptr(),
                rhs_data.len(),
                Ix2(1, 4),
                Strides::from_slice(&[1_usize, 1]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order [1,4] view");
        let output_dim = Ix2(3, 4);
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d(&one_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&one);
        let result = par_zip(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| a + b);
        // F-order [3,4]: slot(i,j) at i + 3*j; result[i,j] = lhs[i] + rhs[j].
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0, 42.0, 43.0]
        );
        reset_parallel_threshold();
    }

    /// `par_zip` supports a type-changing closure `(f64, f64) -> bool`.
    #[test]
    fn test_par_zip_type_changing_closure() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let rhs_data = vec![4.0f64, 3.0, 2.0, 1.0];
        let lhs = unsafe { view_1d(&lhs_data) };
        let rhs = unsafe { view_1d(&rhs_data) };
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| *a > *b);
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[false, false, true, true]
        );
        reset_parallel_threshold();
    }

    /// `par_zip` returns an empty tensor for empty inputs.
    #[test]
    fn test_par_zip_empty() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let empty: Vec<f64> = Vec::new();
        let lhs = unsafe { view_1d(&empty) };
        let rhs = unsafe { view_1d(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d(&one_data) };
        let output_dim = Ix1(0);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&one);
        let result = par_zip(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| a + b);
        assert_eq!(result.len(), 0);
        reset_parallel_threshold();
    }

    /// `par_zip_checked` matches the serial element-wise add.
    #[test]
    fn test_par_zip_checked_matches_serial_add() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let rhs_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let lhs = unsafe { view_1d(&lhs_data) };
        let rhs = unsafe { view_1d(&rhs_data) };
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip_checked(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| Ok(a + b))
            .expect("par_zip_checked should succeed for valid test input");
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 22.0, 33.0, 44.0]
        );
        reset_parallel_threshold();
    }

    /// `par_zip_checked` broadcasts a length-1 rhs against the lhs.
    #[test]
    fn test_par_zip_checked_broadcast_rhs_scalar() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        // Length-1 broadcasted to length-4. The math layer normally
        // produces this via broadcast_to; here we construct an explicit
        // stride-0 view over a single-element backing buffer.
        let rhs_data = [10.0f64];
        let lhs = unsafe { view_1d(&lhs_data) };
        // SAFETY: shape [4], stride [0], storage_len 1 = broadcast view.
        let rhs = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                rhs_data.as_ptr(),
                rhs_data.len(),
                Ix1(4),
                Strides::from_slice(&[0_usize]).expect("valid broadcast strides for test"),
                0,
            )
        }
        .expect("valid broadcast view");
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip_checked(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| Ok(a + b))
            .expect("par_zip_checked should succeed for valid test input");
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 12.0, 13.0, 14.0]
        );
        reset_parallel_threshold();
    }

    /// `par_zip_checked` exercises the same 2-D broadcast coord path.
    #[test]
    fn test_par_zip_checked_multidim_broadcast() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = [1.0f64, 2.0, 3.0];
        let lhs = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                lhs_data.as_ptr(),
                lhs_data.len(),
                Ix2(3, 1),
                Strides::from_slice(&[1_usize, 3]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order [3,1] view");
        let rhs_data = [10.0f64, 20.0, 30.0, 40.0];
        let rhs = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                rhs_data.as_ptr(),
                rhs_data.len(),
                Ix2(1, 4),
                Strides::from_slice(&[1_usize, 1]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order [1,4] view");
        let output_dim = Ix2(3, 4);
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d(&one_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&one);
        let result = par_zip_checked(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| Ok(a + b))
            .expect("par_zip_checked should succeed for valid test input");
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0, 41.0, 42.0, 43.0]
        );
        reset_parallel_threshold();
    }

    /// `par_zip_checked` propagates a closure `Err` as an overall `Err`.
    #[test]
    fn test_par_zip_checked_error_propagation() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let rhs_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let lhs = unsafe { view_1d(&lhs_data) };
        let rhs = unsafe { view_1d(&rhs_data) };
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip_checked(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| {
            if *a == 3.0 {
                Err(XenonError::InvalidArgument {
                    operation: Cow::Borrowed("test"),
                    kind: InvalidArgumentKind::NumericOutOfRange {
                        argument: Cow::Borrowed("a"),
                        domain: Cow::Borrowed("[0, 2]"),
                        actual: Cow::Borrowed("3"),
                    },
                })
            } else {
                Ok(a + b)
            }
        });
        assert!(result.is_err());
        reset_parallel_threshold();
    }

    /// `par_zip_checked` propagates a worker panic as a panic.
    #[test]
    #[should_panic]
    fn test_par_zip_checked_panic_propagation() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let rhs_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let lhs = unsafe { view_1d(&lhs_data) };
        let rhs = unsafe { view_1d(&rhs_data) };
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let _ = par_zip_checked(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| {
            if *a == 3.0 {
                panic!("panic in worker");
            }
            Ok(a + b)
        });
        reset_parallel_threshold();
    }

    /// `par_zip_checked` returns an empty tensor for empty inputs.
    #[test]
    fn test_par_zip_checked_empty() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let empty: Vec<f64> = Vec::new();
        let lhs = unsafe { view_1d(&empty) };
        let rhs = unsafe { view_1d(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d(&one_data) };
        let output_dim = Ix1(0);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&one);
        let result = par_zip_checked(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| Ok(a + b))
            .expect("empty par_zip_checked should succeed");
        assert_eq!(result.len(), 0);
        reset_parallel_threshold();
    }
}
