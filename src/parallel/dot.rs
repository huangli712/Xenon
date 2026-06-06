//! Parallel dot product.
//!
//! Provides [`par_dot`], the parallel inner product of two 1-D tensors.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::dispatch::{ParallelExecStrategy, ParallelGuard, with_parallel_worker_context};
use crate::element::Numeric;
use crate::error::{InvalidArgumentKind, XenonError};
use crate::parallel::chunks::compute_safe_chunks;
use crate::storage::Storage;
use crate::tensor::TensorBase;

/// Parallel dot product of two 1-D tensors.
///
/// Computes `sum(conj(lhs_i) * rhs_i)`. Public (rather than `pub(crate)`) so
/// integration tests can exercise the kernel directly; re-exported through
/// the crate prelude.
///
/// # Errors
///
/// Returns:
/// - [`XenonError::InvalidArgument`] when either operand is not 1-D, or when
///   either operand is not F-contiguous or carries a zero stride
///   (broadcast view).
/// - [`XenonError::ShapeMismatch`] when `lhs` and `rhs` have different shapes.
///
/// # Panics
///
/// Does not panic in practice: the `as_slice().expect(...)` calls are
/// guarded by the F-contiguous + non-broadcast checks above, which return
/// `Err(InvalidArgument)` instead. The `expect` messages document the
/// invariant for future refactors.
#[cfg(feature = "parallel")]
pub fn par_dot<SL, SR, A, DL, DR>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
) -> Result<A, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = A>,
    DL: Dimension,
    DR: Dimension,
    A: Numeric + Send + Sync,
{
    if lhs.ndim() != 1 {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("par_dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("lhs"),
                constraint: Cow::Owned(format!("must be 1-D, got {}-D", lhs.ndim())),
            },
        });
    }
    if rhs.ndim() != 1 {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("par_dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("rhs"),
                constraint: Cow::Owned(format!("must be 1-D, got {}-D", rhs.ndim())),
            },
        });
    }

    if lhs.shape() != rhs.shape() {
        return Err(XenonError::ShapeMismatch {
            operation: Cow::Borrowed("par_dot"),
            left_shape: lhs.shape().to_vec(),
            right_shape: rhs.shape().to_vec(),
        });
    }

    let total = lhs.len();
    if total == 0 {
        return Ok(A::zero());
    }

    let lhs_bad = !lhs.is_f_contiguous() || lhs.has_zero_stride();
    let rhs_bad = !rhs.is_f_contiguous() || rhs.has_zero_stride();
    if lhs_bad || rhs_bad {
        let argument = if lhs_bad { "lhs" } else { "rhs" };
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("par_dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed(argument),
                constraint: Cow::Borrowed(
                    "1-D operand must be F-contiguous and not a broadcast view",
                ),
            },
        });
    }

    let lhs_slice = lhs
        .as_slice()
        .expect("verified lhs is 1-D F-contiguous and not a broadcast view");
    let rhs_slice = rhs
        .as_slice()
        .expect("verified rhs is 1-D F-contiguous and not a broadcast view");

    let num_threads = strategy
        .max_workers()
        .unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size()
        .unwrap_or_else(|| compute_safe_chunks(total, num_threads));

    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let result = (0..total)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|i| {
            with_parallel_worker_context(|| {
                let a = lhs_slice[i];
                let b = rhs_slice[i];
                a.conjugate() * b
            })
        })
        .reduce(|| A::zero(), |x, y| x + y);

    Ok(result)
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::dimension::{Dimension, Ix1, Ix2};
    use crate::dispatch::ThresholdTestGuard;
    use crate::dispatch::{
        ExecPath, ParallelExecStrategy, ParallelGuard, reset_parallel_threshold, select_exec_path,
        set_parallel_threshold,
    };
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::storage::Storage;
    use crate::tensor::{TensorBase, TensorView};
    use crate::complex::Complex;

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

    /// Build a 1-D F-order view over `data` for test inputs (any element type).
    unsafe fn view_1d<'a, A: Element>(data: &'a [A]) -> TensorView<'a, A, Ix1> {
        // SAFETY: caller ensures data is a valid F-order 1-D contiguous slice.
        unsafe {
            TensorView::<A, Ix1>::from_raw_parts(
                data.as_ptr(),
                data.len(),
                Ix1(data.len()),
                Strides::from_slice(&[1_usize]).expect("valid F-order strides for test"),
                0,
            )
            .expect("valid F-order 1-D view")
        }
    }

    /// `par_dot` matches the serial inner product and returns the identity
    /// (0) for empty inputs.
    #[test]
    fn test_par_dot_matches_serial_and_empty_identity() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);

        let a_vec: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let b_vec: Vec<f64> = (0..256).map(|i| (255 - i) as f64).collect();
        let a = unsafe { view_1d_f64(&a_vec) };
        let b = unsafe { view_1d_f64(&b_vec) };
        let strategy = ParallelExecStrategy::auto();
        {
            let guard = acquire_guard(&a);
            let par_result = par_dot(&a, &b, &strategy, guard)
                .expect("par_dot should succeed for valid test input");
            let serial_result: f64 = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x * y).sum();
            assert!((par_result - serial_result).abs() < 1e-10 * serial_result.abs().max(1.0));
        }

        let empty: Vec<f64> = Vec::new();
        let ea = unsafe { view_1d_f64(&empty) };
        let eb = unsafe { view_1d_f64(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d_f64(&one_data) };
        {
            let guard = acquire_guard(&one);
            let result =
                par_dot(&ea, &eb, &strategy, guard).expect("empty par_dot should return identity");
            assert_eq!(result, 0.0f64);
        }

        reset_parallel_threshold();
    }

    /// `par_dot` rejects shape mismatch, non-1-D rank, and broadcast views
    /// with the appropriate errors.
    #[test]
    fn test_par_dot_error_cases() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let strategy = ParallelExecStrategy::auto();

        // Shape mismatch — needs a guard for the signature but par_dot
        // returns Err before using any parallel state.
        let a_data = vec![1.0f64, 2.0, 3.0];
        let b_data = vec![1.0f64, 2.0];
        let a = unsafe { view_1d_f64(&a_data) };
        let b = unsafe { view_1d_f64(&b_data) };
        {
            let guard = acquire_guard(&a);
            let result = par_dot(&a, &b, &strategy, guard);
            match result {
                Err(XenonError::ShapeMismatch {
                    left_shape,
                    right_shape,
                    ..
                }) => {
                    assert_eq!(left_shape, vec![3]);
                    assert_eq!(right_shape, vec![2]);
                },
                _ => panic!("expected ShapeMismatch, got {:?}", result),
            }
        }

        // Rank mismatch (2-D lhs)
        let a_2d_data = [1.0f64, 2.0, 3.0, 4.0];
        let a_2d = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                a_2d_data.as_ptr(),
                a_2d_data.len(),
                Ix2(2, 2),
                Strides::from_slice(&[1_usize, 2]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order 2x2 view");
        let b_1d_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_1d = unsafe { view_1d_f64(&b_1d_data) };
        {
            let guard = acquire_guard(&b_1d);
            let result = par_dot(&a_2d, &b_1d, &strategy, guard);
            assert!(matches!(result, Err(XenonError::InvalidArgument { .. })));
        }

        // Broadcast view rejection
        let a_bc_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a_bc = unsafe { view_1d_f64(&a_bc_data) };
        let b_backing = [10.0f64];
        let b_bc = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                b_backing.as_ptr(),
                b_backing.len(),
                Ix1(4),
                Strides::from_slice(&[0_usize]).expect("valid broadcast strides for test"),
                0,
            )
        }
        .expect("valid broadcast view");
        {
            let guard = acquire_guard(&a_bc);
            let result = par_dot(&a_bc, &b_bc, &strategy, guard);
            assert!(matches!(result, Err(XenonError::InvalidArgument { .. })));
        }

        reset_parallel_threshold();
    }

    /// `par_dot` applies the Hermitian convention `sum(conj(lhs) * rhs)` for
    /// complex inputs — the conjugate on the left operand is observable here
    /// (unlike real inputs, where conjugate is the identity).
    #[test]
    fn test_par_dot_complex_f64_conjugate() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let a_data = [Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)];
        let b_data = [Complex::new(5.0f64, 6.0), Complex::new(7.0, 8.0)];
        let a = unsafe { view_1d(&a_data) };
        let b = unsafe { view_1d(&b_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_guard(&a);
        let result =
            par_dot(&a, &b, &strategy, guard).expect("par_dot should succeed for valid input");
        // conj(1+2i)*(5+6i) + conj(3+4i)*(7+8i)
        // = (1-2i)*(5+6i) + (3-4i)*(7+8i)
        // = (17-4i) + (53-4i) = 70 - 8i
        assert_eq!(result, Complex::new(70.0f64, -8.0));
        reset_parallel_threshold();
    }

    /// `par_dot` works for `f32`.
    #[test]
    fn test_par_dot_f32() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [5.0f32, 6.0, 7.0, 8.0];
        let a = unsafe { view_1d(&a_data) };
        let b = unsafe { view_1d(&b_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_guard(&a);
        let result =
            par_dot(&a, &b, &strategy, guard).expect("par_dot should succeed for valid input");
        assert_eq!(result, 70.0f32); // 5+12+21+32
        reset_parallel_threshold();
    }

    /// `par_dot` works for `i32` (conjugate is the identity).
    #[test]
    fn test_par_dot_i32() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let a_data = [1i32, 2, 3, 4];
        let b_data = [5i32, 6, 7, 8];
        let a = unsafe { view_1d(&a_data) };
        let b = unsafe { view_1d(&b_data) };
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_guard(&a);
        let result =
            par_dot(&a, &b, &strategy, guard).expect("par_dot should succeed for valid input");
        assert_eq!(result, 70i32);
        reset_parallel_threshold();
    }

    /// `par_dot` rejects a non-1-D rhs (the symmetric branch to the 2-D lhs
    /// case), returning `InvalidArgument`.
    #[test]
    fn test_par_dot_rhs_rank_mismatch() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let strategy = ParallelExecStrategy::auto();

        let a_1d_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a_1d = unsafe { view_1d_f64(&a_1d_data) };
        let b_2d_data = [1.0f64, 2.0, 3.0, 4.0];
        let b_2d = unsafe {
            TensorView::<f64, Ix2>::from_raw_parts(
                b_2d_data.as_ptr(),
                b_2d_data.len(),
                Ix2(2, 2),
                Strides::from_slice(&[1_usize, 2]).expect("valid F-order strides for test"),
                0,
            )
        }
        .expect("valid F-order 2x2 view");
        let guard = acquire_guard(&a_1d);
        let result = par_dot(&a_1d, &b_2d, &strategy, guard);
        assert!(matches!(result, Err(XenonError::InvalidArgument { .. })));
        reset_parallel_threshold();
    }

    /// `par_dot` rejects an lhs broadcast view (the lhs_bad branch), returning
    /// `InvalidArgument`.
    #[test]
    fn test_par_dot_lhs_broadcast_rejected() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        let strategy = ParallelExecStrategy::auto();

        let a_backing = [10.0f64];
        let a_bc = unsafe {
            TensorView::<f64, Ix1>::from_raw_parts(
                a_backing.as_ptr(),
                a_backing.len(),
                Ix1(4),
                Strides::from_slice(&[0_usize]).expect("valid broadcast strides for test"),
                0,
            )
        }
        .expect("valid broadcast view");
        let b_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = unsafe { view_1d_f64(&b_data) };
        let guard = acquire_guard(&b);
        let result = par_dot(&a_bc, &b, &strategy, guard);
        assert!(matches!(result, Err(XenonError::InvalidArgument { .. })));
        reset_parallel_threshold();
    }
}
