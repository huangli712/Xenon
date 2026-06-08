//! Vector dot product entry point.
//!
//! Layered implementation across W17 tasks:
//!   * W17T1: public signature with `Ok(A::zero())` stub.
//!   * W17T2: rank + length validation returning recoverable errors.
//!   * W17T3: scalar inner product via `DotAccumulate` trait.
//!   * W17T4: dispatch wiring (`select_exec_path`).
//!   * W17T5: SIMD path integration via `simd::try_dot_*`.
//!   * W17T6 (this task): parallel path integration via `parallel::par_dot`.
//!
//! Design reference: 12-matrix §5.1, §6.

#[cfg(feature = "simd")]
use std::any::TypeId;
use std::borrow::Cow;

#[cfg(feature = "simd")]
use crate::complex::Complex;
use crate::dimension::Dimension;
#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;
use crate::dispatch::{ExecPath, select_exec_path};
use crate::element::Numeric;
use super::types::DotAccumulate;
use crate::error::{InvalidArgumentKind, XenonError};
use crate::storage::Storage;
use crate::tensor::TensorBase;

// ── Validation ──

fn validate_dot_inputs<S1, S2, A, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Result<(), XenonError>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    if a.ndim() != 1 {
        return Err(invalid_rank("a"));
    }
    if b.ndim() != 1 {
        return Err(invalid_rank("b"));
    }
    if a.len() != b.len() {
        return Err(XenonError::ShapeMismatch {
            operation: Cow::Borrowed("dot"),
            left_shape: a.shape().to_vec(),
            right_shape: b.shape().to_vec(),
        });
    }
    Ok(())
}

fn invalid_rank(argument: &'static str) -> XenonError {
    XenonError::InvalidArgument {
        operation: Cow::Borrowed("dot"),
        kind: InvalidArgumentKind::OperationSpecific {
            argument: Cow::Borrowed(argument),
            constraint: Cow::Borrowed("rank == 1"),
        },
    }
}

// ── Alignment ──

#[inline]
fn alignment_ok<S1, S2, A, D1, D2>(a: &TensorBase<S1, D1>, b: &TensorBase<S2, D2>) -> bool
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    a.is_aligned() && b.is_aligned()
}

// ── Scalar baseline ──

pub(crate) fn scalar_dot<S1, S2, A, D1, D2>(a: &TensorBase<S1, D1>, b: &TensorBase<S2, D2>) -> A
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Numeric + DotAccumulate,
    D1: Dimension,
    D2: Dimension,
{
    let len = a.len();
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .enumerate()
        .fold(A::zero(), |acc, (index, (x, y))| {
            <A as DotAccumulate>::dot_step(acc, x, y, index, len)
        })
}

#[inline]
pub(crate) fn dot_serial<S1, S2, A, D1, D2>(a: &TensorBase<S1, D1>, b: &TensorBase<S2, D2>) -> A
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Numeric + DotAccumulate,
    D1: Dimension,
    D2: Dimension,
{
    scalar_dot(a, b)
}

// ── SIMD support ──

#[cfg(feature = "simd")]
#[inline]
fn can_use_simd_dot<A: 'static, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> bool
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    if !(a.is_f_contiguous() && b.is_f_contiguous()) {
        return false;
    }
    let t = TypeId::of::<A>();
    t == TypeId::of::<f32>()
        || t == TypeId::of::<f64>()
        || t == TypeId::of::<Complex<f32>>()
        || t == TypeId::of::<Complex<f64>>()
}

#[cfg(feature = "simd")]
fn try_simd_dot_dispatch<A: 'static + Copy, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Option<A>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    let lhs = a.as_slice()?;
    let rhs = b.as_slice()?;

    let t = TypeId::of::<A>();
    if t == TypeId::of::<f32>() {
        let lhs = reinterpret_slice::<A, f32>(lhs);
        let rhs = reinterpret_slice::<A, f32>(rhs);
        return crate::simd::try_dot_f32(lhs, rhs).map(|v| reinterpret_value::<f32, A>(v));
    }
    if t == TypeId::of::<f64>() {
        let lhs = reinterpret_slice::<A, f64>(lhs);
        let rhs = reinterpret_slice::<A, f64>(rhs);
        return crate::simd::try_dot_f64(lhs, rhs).map(|v| reinterpret_value::<f64, A>(v));
    }
    if t == TypeId::of::<Complex<f32>>() {
        let lhs = reinterpret_slice::<A, Complex<f32>>(lhs);
        let rhs = reinterpret_slice::<A, Complex<f32>>(rhs);
        return crate::simd::try_dot_complex_f32(lhs, rhs)
            .map(|v| reinterpret_value::<Complex<f32>, A>(v));
    }
    if t == TypeId::of::<Complex<f64>>() {
        let lhs = reinterpret_slice::<A, Complex<f64>>(lhs);
        let rhs = reinterpret_slice::<A, Complex<f64>>(rhs);
        return crate::simd::try_dot_complex_f64(lhs, rhs)
            .map(|v| reinterpret_value::<Complex<f64>, A>(v));
    }
    None
}

#[cfg(feature = "simd")]
#[inline]
fn reinterpret_slice<A: 'static, B: 'static>(s: &[A]) -> &[B] {
    debug_assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
    // SAFETY: TypeId equality implies full type identity.
    unsafe { &*(s as *const [A] as *const [B]) }
}

#[cfg(feature = "simd")]
#[inline]
fn reinterpret_value<A: 'static + Copy, B: 'static + Copy>(v: A) -> B {
    debug_assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
    // SAFETY: TypeId equality implies same size / alignment / niche.
    unsafe { std::mem::transmute_copy::<A, B>(&v) }
}

// ── Public dot entry ──

/// Vector dot product entry point.
///
/// See 12-matrix §5.1 for the final user-visible contract.
///
/// # Errors
///
/// Returns `XenonError::DimensionMismatch` when the two tensors do not
/// have the same element count. Returns `XenonError::InvalidLayout`
/// when shape product overflow or stride validation fails.
pub(crate) fn dot<S1, S2, A, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Result<A, XenonError>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Numeric + DotAccumulate + Send + Sync,
    D1: Dimension,
    D2: Dimension,
{
    validate_dot_inputs(a, b)?;

    let is_contig = a.is_f_contiguous() && b.is_f_contiguous();
    let (path, guard) = select_exec_path(a.len(), is_contig, alignment_ok(a, b));

    match path {
        ExecPath::Serial => Ok(dot_serial(a, b)),
        ExecPath::Simd => {
            let _ = guard;
            #[cfg(feature = "simd")]
            {
                if can_use_simd_dot::<A, _, _, _, _>(a, b)
                    && let Some(v) = try_simd_dot_dispatch(a, b)
                {
                    return Ok(v);
                }
            }
            Ok(dot_serial(a, b))
        },
        ExecPath::Parallel => {
            // Dispatch invariant (30-dispatch §5.5 line 166): when
            // ExecPath::Parallel is returned, guard MUST be Some.
            let guard = match guard {
                Some(g) => g,
                None => unreachable!(
                    "dispatch returned (ExecPath::Parallel, None) — \
                     invariant violated, see 30-dispatch §5.5 line 166"
                ),
            };

            #[cfg(feature = "parallel")]
            {
                let strategy = ParallelExecStrategy::auto();
                crate::parallel::dot::par_dot::<_, _, A, _, _>(a, b, &strategy, guard)
            }

            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                Ok(dot_serial(a, b))
            }
        },
    }
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::dimension::Ix1;
    use crate::dimension::Ix2;
    #[cfg(feature = "parallel")]
    use crate::dispatch::with_parallel_worker_context;
    use crate::tensor::Tensor;
    use crate::tensor::Tensor1;

    // W17T1
    #[test]
    fn test_matrix_module_dot_skeleton_returns_zero() {
        let a = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        let b = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), 0.0_f64);
    }

    #[test]
    fn test_matrix_module_dot_skeleton_returns_zero_for_int() {
        let a = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        let b = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), 0_i32);
    }

    // W17T2
    #[test]
    fn test_dot_shape_mismatch() {
        let a = Tensor1::from_shape_vec(Ix1(2), vec![1_i32, 2]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).expect("valid construction");
        let err = dot(&a, &b).expect_err("must return error");
        match err {
            XenonError::ShapeMismatch {
                ref operation,
                ref left_shape,
                ref right_shape,
            } => {
                assert_eq!(operation.as_ref(), "dot");
                assert_eq!(left_shape.as_slice(), &[2]);
                assert_eq!(right_shape.as_slice(), &[3]);
            },
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_dot_high_rank_invalid_argument() {
        let a =
            Tensor::<i32, Ix2>::from_shape_vec((1, 1), vec![1_i32]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![1_i32]).expect("valid construction");
        let err = dot(&a, &b).expect_err("must return error");
        match err {
            XenonError::InvalidArgument {
                ref operation,
                kind:
                    InvalidArgumentKind::OperationSpecific {
                        ref argument,
                        ref constraint,
                    },
            } => {
                assert_eq!(operation.as_ref(), "dot");
                assert_eq!(argument.as_ref(), "a");
                assert_eq!(constraint.as_ref(), "rank == 1");
            },
            other => panic!("expected InvalidArgument::OperationSpecific, got {other:?}"),
        }
    }

    #[test]
    fn test_dot_rhs_high_rank_invalid_argument() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![1_i32]).expect("valid construction");
        let b =
            Tensor::<i32, Ix2>::from_shape_vec((1, 1), vec![1_i32]).expect("valid construction");
        let err = dot(&a, &b).expect_err("must return error");
        match err {
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::OperationSpecific { ref argument, .. },
                ..
            } => assert_eq!(argument.as_ref(), "b"),
            other => panic!("expected InvalidArgument::OperationSpecific for b, got {other:?}"),
        }
    }

    // W17T3
    #[test]
    fn test_dot_complex() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![Complex::<f64>::new(1.0, 2.0)])
            .expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![Complex::<f64>::new(3.0, 4.0)])
            .expect("valid construction");
        let r = dot(&a, &b).expect("valid construction");
        assert_eq!(r, Complex::<f64>::new(11.0, -2.0));
    }

    #[test]
    fn test_dot_real_conjugate_is_identity() {
        let a =
            Tensor1::from_shape_vec(Ix1(3), vec![1.0_f64, -2.0, 3.0]).expect("valid construction");
        let b =
            Tensor1::from_shape_vec(Ix1(3), vec![4.0_f64, 5.0, -6.0]).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), -24.0_f64);
    }

    // W17T4
    #[test]
    fn test_dot_empty() {
        let a = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        let b = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), 0_i32);
    }

    #[test]
    fn test_dot_single_element() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![7_i32]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![6_i32]).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), 42_i32);
    }

    #[test]
    fn test_dot_wire_dispatch_small_input_serial() {
        let a = Tensor1::from_shape_vec(Ix1(4), vec![1.0_f64, 2.0, 3.0, 4.0])
            .expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(4), vec![5.0_f64, 6.0, 7.0, 8.0])
            .expect("valid construction");
        assert_eq!(
            dot(&a, &b).expect("valid construction"),
            1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0
        );
    }

    #[test]
    fn test_dot_wire_dispatch_large_input_falls_back_to_scalar() {
        let n: usize = 4096;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 + 1.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), xs.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), ys.clone()).expect("valid construction");
        let expected: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot(&a, &b).expect("valid construction"), expected);
    }

    // W17T5: tolerance helper + SIMD tests
    #[cfg(any(feature = "simd", feature = "parallel"))]
    fn f64_dot_tolerance(n: usize, max_abs_a: f64, max_abs_b: f64) -> f64 {
        let ulp_term = 8.0 * f64::EPSILON * (n as f64) * max_abs_a * max_abs_b;
        let floor = 4.0 * f64::MIN_POSITIVE;
        ulp_term.max(floor)
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_dot_simd_path_with_feature() {
        let values: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.5).collect();
        let a =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");
        let b =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");

        let actual = dot(&a, &b).expect("valid construction");
        let expected = dot_serial(&a, &b);

        let max_abs = values.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance(values.len(), max_abs, max_abs);
        assert!(
            (actual - expected).abs() <= tol,
            "SIMD result {actual} drifts beyond §10.1 tol {tol} from scalar {expected}"
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_dot_simd_path_unsupported_type_falls_back() {
        let values: Vec<i64> = (0..2048).collect();
        let a =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(values.len()), values).expect("valid construction");
        assert_eq!(dot(&a, &b).expect("valid construction"), dot_serial(&a, &b));
    }

    // W17T6: parallel path tests
    #[cfg(feature = "parallel")]
    #[test]
    fn test_dot_parallel_path() {
        let values: Vec<i64> = (0..4096_i64).collect();
        let a =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(values.len()), values).expect("valid construction");
        // Integer dot is exact across paths.
        assert_eq!(dot(&a, &b).expect("valid construction"), dot_serial(&a, &b));
    }

    #[cfg(feature = "parallel")]
    fn f64_dot_tolerance_parallel(n: usize, max_abs_a: f64, max_abs_b: f64) -> f64 {
        // Parallel reduction can reorder floating-point accumulation,
        // requiring more headroom than §10.1 serial/SIMD tolerance.
        let ulp_term = 256.0 * f64::EPSILON * (n as f64) * max_abs_a * max_abs_b;
        let floor = 4.0 * f64::MIN_POSITIVE;
        ulp_term.max(floor)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_dot_large_vector_parallel_threshold() {
        let n: usize = 100_000;
        let xs: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) * 0.1 + 1.0).collect();
        let ys: Vec<f64> = (0..n).map(|i| ((i % 13) as f64) * 0.1 + 2.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), xs.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), ys.clone()).expect("valid construction");
        let actual = dot(&a, &b).expect("valid construction");
        let expected = dot_serial(&a, &b);
        let max_abs_a = xs.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs_b = ys.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance_parallel(n, max_abs_a, max_abs_b);
        assert!(
            (actual - expected).abs() <= tol,
            "parallel result {actual} drifts beyond tolerance {tol} from scalar {expected}"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_dot_nested_parallel_falls_back() {
        let n: usize = 16_384;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1.0).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 2.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), xs.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), ys.clone()).expect("valid construction");

        let baseline = dot_serial(&a, &b);

        let result = with_parallel_worker_context(|| dot(&a, &b).expect("valid construction"));

        let max_abs_a = xs.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs_b = ys.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance(n, max_abs_a, max_abs_b);
        assert!(
            (result - baseline).abs() <= tol,
            "nested-parallel result {result} drifts beyond §10.1 tol {tol} from baseline {baseline}"
        );
    }
}
