//! Parallel reduction operations.
//!
//! W15T4: par_reduce_impl + par_sum — general reduction and sum.
//! W15T5: par_dot — parallel dot product.

use std::borrow::Cow;

use crate::dispatch::{ParallelExecStrategy, ParallelGuard, with_parallel_worker_context};
use crate::dimension::Dimension;
use crate::element::{Element, Numeric};
use crate::error::{InvalidArgumentKind, XenonError};
use crate::storage::Storage;
use crate::tensor::TensorBase;
use crate::parallel::compute_safe_chunks;

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

    let src_slice = tensor.as_slice().expect(
        "par_reduce_impl caller must ensure F-contiguous + non-broadcast"
    );

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

#[cfg(feature = "parallel")]
pub(crate) fn par_sum<S, A, D>(
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

#[cfg(feature = "parallel")]
pub(crate) fn par_dot<SL, SR, A, DL, DR>(
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

    let lhs_slice = lhs.as_slice().expect(
        "verified lhs is 1-D F-contiguous and not a broadcast view"
    );
    let rhs_slice = rhs.as_slice().expect(
        "verified rhs is 1-D F-contiguous and not a broadcast view"
    );

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
    use crate::dispatch::{
        select_exec_path, ExecPath, ParallelExecStrategy,
        set_parallel_threshold, reset_parallel_threshold, ParallelGuard,
    };
    use crate::dimension::{Dimension, Ix1, Ix2};
    use crate::element::Element;
    use crate::layout::Strides;
    use crate::storage::Storage;
    use crate::tensor::{TensorBase, TensorView};

    /// Acquire a guard from select_exec_path. Uses `set_parallel_threshold(1)`
    /// to force the parallel path. If IN_PARALLEL TLS is contaminated from a
    /// prior test, the guard will be None — tests that need a real parallel
    /// path must still get Some(guard), so we assert it.
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

    // ── W15T4 tests ──

    #[test]
    fn test_par_sum_serial_match_and_empty_identity() {
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

    // ── W15T5 tests ──

    #[test]
    fn test_par_dot_matches_serial_and_empty_identity() {
        set_parallel_threshold(1);

        let a_vec: Vec<f64> = (0..256).map(|i| i as f64).collect();
        let b_vec: Vec<f64> = (0..256).map(|i| (255 - i) as f64).collect();
        let a = unsafe { view_1d_f64(&a_vec) };
        let b = unsafe { view_1d_f64(&b_vec) };
        let strategy = ParallelExecStrategy::auto();
        {
            let guard = acquire_guard(&a);
            let par_result = par_dot(&a, &b, &strategy, guard).expect("par_dot should succeed for valid test input");
            let serial_result: f64 = a_vec.iter().zip(b_vec.iter())
                .map(|(x, y)| x * y).sum();
            assert!((par_result - serial_result).abs()
                < 1e-10 * serial_result.abs().max(1.0));
        }

        let empty: Vec<f64> = Vec::new();
        let ea = unsafe { view_1d_f64(&empty) };
        let eb = unsafe { view_1d_f64(&empty) };
        let one_data = vec![0.0f64];
        let one = unsafe { view_1d_f64(&one_data) };
        {
            let guard = acquire_guard(&one);
            let result = par_dot(&ea, &eb, &strategy, guard).expect("empty par_dot should return identity");
            assert_eq!(result, 0.0f64);
        }

        reset_parallel_threshold();
    }

    #[test]
    fn test_par_dot_error_cases() {
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
                Err(XenonError::ShapeMismatch { left_shape, right_shape, .. }) => {
                    assert_eq!(left_shape, vec![3]);
                    assert_eq!(right_shape, vec![2]);
                }
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
}