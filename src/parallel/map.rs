//! Parallel element-wise map operations.
//!
//! W15T2: par_map — single-input parallel element-wise map.
//! W15T3: par_zip_map — dual-input broadcast element-wise parallel map.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::dispatch::{ParallelExecStrategy, ParallelGuard, with_parallel_worker_context};
use crate::element::Element;
use crate::error::{InvalidShapeKind, XenonError};
use crate::parallel::compute_safe_chunks;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Parallel element-wise map.
///
/// Test-only visibility: re-exported at crate root under `#[doc(hidden)]`.
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

// ── W15T3: par_zip_map ──

#[cfg(feature = "parallel")]
pub(crate) fn par_zip_map<SL, SR, A, B, C, DL, DR, DO, F>(
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
    // checked_size overflow -> InvalidShape with ProductOverflow
    let total = output_dim
        .checked_size()
        .map_err(|_| XenonError::InvalidShape {
            operation: Cow::Borrowed("par_zip_map"),
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
    let lhs_view = lhs.broadcast_to(output_dim.clone()).expect(
        "math layer ensures broadcast compatibility; violation is an internal bug \
         (09-parallel 6.3 line 422, 30-dispatch debug_assert policy)",
    );
    let rhs_view = rhs
        .broadcast_to(output_dim.clone())
        .expect("math layer ensures broadcast compatibility; violation is an internal bug");

    // Pre-compute output shape for F-order index -> multi-dim coord conversion.
    let out_shape = output_dim.slice();
    let ndim = out_shape.len();
    let mut strides_f = vec![1usize; ndim];
    for k in 1..ndim {
        strides_f[k] = strides_f[k - 1] * out_shape[k - 1];
    }

    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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

    // SAFETY (07-tensor 5 from_raw_vec_unchecked precondition):
    //   - succeeded.len() == total == output_dim.checked_size() (validated above)
    //   - F-order alignment: (0..total).into_par_iter() + collect_into_vec
    //     preserves index -> slot mapping (09-parallel 6.7 line 504-505)
    Ok(unsafe { Tensor::from_raw_vec_unchecked(succeeded, output_dim.clone()) })
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use crate::dimension::Ix1;
    use crate::dispatch::{
        ExecPath, ParallelExecStrategy, reset_parallel_threshold, select_exec_path,
        set_parallel_threshold,
    };
    use crate::layout::Strides;
    use crate::tensor::TensorView;

    #[test]
    fn test_par_map_parallel_path() {
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

    // ── W15T3 tests ──

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

    /// SAFETY helper: build a 1-D F-order view over a Vec.
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

    #[test]
    fn test_par_zip_map_matches_serial_add() {
        set_parallel_threshold(1);
        let lhs_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let rhs_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let lhs = unsafe { view_1d(&lhs_data) };
        let rhs = unsafe { view_1d(&rhs_data) };
        let output_dim = Ix1(4);
        let strategy = ParallelExecStrategy::auto();
        let guard = acquire_parallel_guard(&lhs);
        let result = par_zip_map(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| Ok(a + b))
            .expect("par_zip_map should succeed for valid test input");
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 22.0, 33.0, 44.0]
        );
        reset_parallel_threshold();
    }

    #[test]
    fn test_par_zip_map_broadcast_rhs_scalar() {
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
        let result = par_zip_map(&lhs, &rhs, &output_dim, &strategy, guard, |a, b| Ok(a + b))
            .expect("par_zip_map should succeed for valid test input");
        assert_eq!(
            result.as_slice().expect("valid F-order test output"),
            &[11.0, 12.0, 13.0, 14.0]
        );
        reset_parallel_threshold();
    }
}
