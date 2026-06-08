//! Vector dot product.
//!
//! Implementation dispatches through serial, SIMD, and parallel execution
//! paths depending on input size, contiguity, and alignment.

use std::any::TypeId;
use std::borrow::Cow;
use std::mem::transmute_copy;

use crate::error::{InvalidArgumentKind, XenonError};
use crate::dimension::Dimension;
use crate::element::Numeric;
use crate::storage::Storage;
use crate::tensor::TensorBase;
use crate::dispatch::{ExecPath, select_exec_path};

#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;

#[cfg(feature = "parallel")]
use crate::parallel::dot::par_dot;

#[cfg(feature = "simd")]
use crate::simd::{
    try_dot_f32,
    try_dot_f64,
    try_dot_complex_f32,
    try_dot_complex_f64,
};

#[cfg(feature = "simd")]
use crate::complex::Complex;

// --- dot_impl ---------------------------------------------------------------

/// Vector dot product with validation and dispatch.
///
/// Selects an execution path via [`select_exec_path`] based on element count,
/// contiguity, and alignment. Falls back to the serial baseline when SIMD or
/// parallel paths are unavailable or unsuitable.
///
/// # Errors
///
/// Returns `XenonError::InvalidArgument` when either tensor is not
/// 1‑dimensional. Returns `XenonError::ShapeMismatch` when the two tensors
/// have different element counts.
pub(crate) fn dot_impl<S1, S2, A, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Result<A, XenonError>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Numeric + Copy + 'static + Send + Sync,
    D1: Dimension,
    D2: Dimension,
{
    validate_dot_inputs(a, b)?;

    let is_contig = a.is_f_contiguous() && b.is_f_contiguous();
    let (path, guard) = select_exec_path(
        a.len(),
        is_contig,
        a.is_aligned() && b.is_aligned()
    );

    match path {
        ExecPath::Serial => Ok(try_dot_serial(a, b)),
        ExecPath::Simd => {
            let _ = guard;
            #[cfg(feature = "simd")]
            {
                if can_use_simd_dot::<A, _, _, _, _>(a, b)
                    && let Some(v) = try_dot_simd(a, b)
                {
                    return Ok(v);
                }
            }
            Ok(try_dot_serial(a, b))
        },
        ExecPath::Parallel => {
            // When select_exec_path returns Parallel, the guard is always Some.
            let guard = match guard {
                Some(g) => g,
                None => unreachable!(
                    "dot_impl: expected ParallelGuard on ExecPath::Parallel"
                ),
            };

            #[cfg(feature = "parallel")]
            {
                let strategy = ParallelExecStrategy::auto();
                par_dot::<_, _, A, _, _>(a, b, &strategy, guard)
            }

            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                Ok(try_dot_serial(a, b))
            }
        },
    }
}

// --- Scalar baseline --------------------------------------------------------

/// Serial dot product baseline.
///
/// Iterates element‑wise, delegating per‑step arithmetic to [`dot_step`].
/// Used as the fallback when SIMD or parallel execution is not applicable.
#[inline]
pub(crate) fn try_dot_serial<S1, S2, A, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>
) -> A
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Numeric + Copy + 'static,
    D1: Dimension,
    D2: Dimension,
{
    let len = a.len();
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .enumerate()
        .fold(A::zero(), |acc, (index, (x, y))| {
            dot_step(acc, x, y, index, len)
        })
}

// --- SIMD dispatch ----------------------------------------------------------

/// SIMD‑accelerated dot product.
///
/// Returns `None` when the element type is not supported by the SIMD
/// backend. The caller falls back to [`try_dot_serial`] on `None`.
#[cfg(feature = "simd")]
fn try_dot_simd<A: 'static + Copy, S1, S2, D1, D2>(
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

    if TypeId::of::<A>() == TypeId::of::<f32>() {
        // SAFETY: TypeId equality proves `A == f32`.
        let lhs: &[f32] = unsafe {
            &*(lhs as *const [A] as *const [f32]) 
        };
        let rhs: &[f32] = unsafe { 
            &*(rhs as *const [A] as *const [f32]) 
        };
        return try_dot_f32(lhs, rhs)
            .map(|v| unsafe { transmute_copy::<f32, A>(&v) });
    }
    if TypeId::of::<A>() == TypeId::of::<f64>() {
        // SAFETY: TypeId equality proves `A == f64`.
        let lhs: &[f64] = unsafe {
            &*(lhs as *const [A] as *const [f64])
        };
        let rhs: &[f64] = unsafe {
            &*(rhs as *const [A] as *const [f64]) 
        };
        return try_dot_f64(lhs, rhs)
            .map(|v| unsafe { transmute_copy::<f64, A>(&v) });
    }
    if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
        // SAFETY: TypeId equality proves `A == Complex<f32>`.
        let lhs: &[Complex<f32>] = unsafe {
            &*(lhs as *const [A] as *const [Complex<f32>]) 
        };
        let rhs: &[Complex<f32>] = unsafe { 
            &*(rhs as *const [A] as *const [Complex<f32>]) 
        };
        return try_dot_complex_f32(lhs, rhs)
            .map(|v| unsafe { transmute_copy::<Complex<f32>, A>(&v) });
    }
    if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
        // SAFETY: TypeId equality proves `A == Complex<f64>`.
        let lhs: &[Complex<f64>] = unsafe {
            &*(lhs as *const [A] as *const [Complex<f64>]) 
        };
        let rhs: &[Complex<f64>] = unsafe {
            &*(rhs as *const [A] as *const [Complex<f64>]) 
        };
        return try_dot_complex_f64(lhs, rhs)
            .map(|v| unsafe { transmute_copy::<Complex<f64>, A>(&v) });
    }
    None
}

// --- Validation --------------------------------------------------------------

/// Validate that both inputs are 1‑dimensional vectors of equal length.
///
/// # Errors
///
/// Returns `XenonError::InvalidArgument` if either input has rank != 1.
/// Returns `XenonError::ShapeMismatch` if the lengths differ.
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
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("a"),
                constraint: Cow::Borrowed("rank == 1"),
            },
        });
    }
    if b.ndim() != 1 {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("b"),
                constraint: Cow::Borrowed("rank == 1"),
            },
        });
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

// --- SIMD support ------------------------------------------------------------

/// Returns `true` when both vectors are SIMD‑eligible.
///
/// Requires F‑contiguous layout and a SIMD‑supported element type
/// (`f32`, `f64`, `Complex<f32>`, `Complex<f64>`).
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

/// Per‑step dot‑product accumulation with type‑aware arithmetic.
///
/// - `i32`, `i64`: checked multiply then checked add; integer overflow is
///   unrecoverable and panics with element context (index, shape, type).
/// - `f32`, `f64`, `Complex<f32>`, `Complex<f64>`: conjugate‑linear
///   `acc + x.conjugate() * y` via `Numeric`, preserving IEEE 754 NaN / Inf
///   propagation. `conjugate()` is the identity for real types and the true
///   conjugate for complex types.
///
/// Uses `TypeId` dispatch so the per‑type arithmetic logic stays off the
/// generic bounds of the public `dot` entry point.
///
/// # Safety
///
/// The `unsafe` reads are sound because each is gated by
/// `TypeId::of::<A>() == TypeId::of::<I>()`, which proves layout identity
/// between `A` and the integer type `I`.
#[inline]
fn dot_step<A>(acc: A, x: A, y: A, index: usize, len: usize) -> A
where
    A: Numeric + Copy + 'static,
{
    if TypeId::of::<A>() == TypeId::of::<i32>() {
        // SAFETY: TypeId equality proves `A == i32`, so reading `&A` through
        // a `*const i32` is sound.
        let xi: i32 = unsafe { *(&x as *const A as *const i32) };
        let yi: i32 = unsafe { *(&y as *const A as *const i32) };
        let acci: i32 = unsafe { *(&acc as *const A as *const i32) };
        let product = xi.checked_mul(yi).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during multiplication at element {index} of shape [{len}] (type i32)"
            )
        });
        let sum = acci.checked_add(product).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during accumulation at element {index} of shape [{len}] (type i32)"
            )
        });
        // SAFETY: `A == i32`; reinterpreting `i32` as `A` is identity.
        return unsafe { transmute_copy::<i32, A>(&sum) };
    }
    if TypeId::of::<A>() == TypeId::of::<i64>() {
        // SAFETY: TypeId equality proves `A == i64`.
        let xi: i64 = unsafe { *(&x as *const A as *const i64) };
        let yi: i64 = unsafe { *(&y as *const A as *const i64) };
        let acci: i64 = unsafe { *(&acc as *const A as *const i64) };
        let product = xi.checked_mul(yi).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during multiplication at element {index} of shape [{len}] (type i64)"
            )
        });
        let sum = acci.checked_add(product).unwrap_or_else(|| {
            panic!(
                "dot: integer overflow during accumulation at element {index} of shape [{len}] (type i64)"
            )
        });
        // SAFETY: `A == i64`; reinterpreting `i64` as `A` is identity.
        return unsafe { transmute_copy::<i64, A>(&sum) };
    }
    // Float / complex path: conjugate-linear `acc + x.conjugate() * y` via
    // `Numeric: Add + Mul`. `conjugate()` is the identity for f32 / f64 and
    // the true conjugate for `Complex`, so this single expression covers all
    // non-integer supported types and preserves IEEE 754 NaN / Inf semantics.
    acc + x.conjugate() * y
}

// --- Unit tests --------------------------------------------------------------

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

    // --- Helpers -------------------------------------------------------------

    /// F64 dot‑product tolerance for scalar and SIMD comparison.
    ///
    /// Based on element count and the maximum absolute values in each input.
    #[cfg(any(feature = "simd", feature = "parallel"))]
    fn f64_dot_tolerance(n: usize, max_abs_a: f64, max_abs_b: f64) -> f64 {
        let ulp_term = 8.0 * f64::EPSILON * (n as f64) * max_abs_a * max_abs_b;
        let floor = 4.0 * f64::MIN_POSITIVE;
        ulp_term.max(floor)
    }

    /// F64 dot‑product tolerance for parallel vs serial comparison.
    ///
    /// Parallel reduction reorders floating‑point accumulation, requiring
    /// a wider tolerance margin than the scalar / SIMD comparison.
    #[cfg(feature = "parallel")]
    fn f64_dot_tolerance_parallel(n: usize, max_abs_a: f64, max_abs_b: f64) -> f64 {
        let ulp_term = 256.0 * f64::EPSILON * (n as f64) * max_abs_a * max_abs_b;
        let floor = 4.0 * f64::MIN_POSITIVE;
        ulp_term.max(floor)
    }

    // --- dot_impl: basic correctness -----------------------------------------

    /// Dot product of two empty f64 tensors returns `0.0`.
    #[test]
    fn test_dot_zero_f64() {
        let a = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        let b = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), 0.0_f64);
    }

    /// Dot product of two empty i32 tensors returns `0`.
    #[test]
    fn test_dot_zero_i32() {
        let a = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        let b = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), 0_i32);
    }

    /// Dot product of two empty i32 tensors returns the additive identity.
    #[test]
    fn test_dot_empty() {
        let a = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        let b = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![]).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), 0_i32);
    }

    /// Dot product of two single‑element tensors equals the product of their elements.
    #[test]
    fn test_dot_single_element() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![7_i32]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![6_i32]).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), 42_i32);
    }

    /// Dot product of a small length‑4 f64 tensor produces the expected scalar sum.
    #[test]
    fn test_dot_small() {
        let a = Tensor1::from_shape_vec(Ix1(4), vec![1.0_f64, 2.0, 3.0, 4.0])
            .expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(4), vec![5.0_f64, 6.0, 7.0, 8.0])
            .expect("valid construction");
        assert_eq!(
            dot_impl(&a, &b).expect("valid construction"),
            1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0
        );
    }

    /// Dot product of a large (4096‑element) f64 tensor matches the scalar reference.
    #[test]
    fn test_dot_large() {
        let n: usize = 4096;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 + 1.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), xs.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), ys.clone()).expect("valid construction");
        let expected: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), expected);
    }

    /// Conjugate of a real number is the identity, so the dot product of
    /// real vectors equals the ordinary inner product.
    #[test]
    fn test_dot_real_conjugate_is_identity() {
        let a =
            Tensor1::from_shape_vec(Ix1(3), vec![1.0_f64, -2.0, 3.0]).expect("valid construction");
        let b =
            Tensor1::from_shape_vec(Ix1(3), vec![4.0_f64, 5.0, -6.0]).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), -24.0_f64);
    }

    /// Dot product of complex vectors computes conjugate‑linear inner product.
    #[test]
    fn test_dot_complex() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![Complex::<f64>::new(1.0, 2.0)])
            .expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![Complex::<f64>::new(3.0, 4.0)])
            .expect("valid construction");
        let r = dot_impl(&a, &b).expect("valid construction");
        assert_eq!(r, Complex::<f64>::new(11.0, -2.0));
    }

    // --- dot_impl: validation errors -----------------------------------------

    /// Shape mismatch between two non‑empty vectors returns `ShapeMismatch`.
    #[test]
    fn test_dot_shape_mismatch() {
        let a = Tensor1::from_shape_vec(Ix1(2), vec![1_i32, 2]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).expect("valid construction");
        let err = dot_impl(&a, &b).expect_err("must return error");
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

    /// A rank‑2 left tensor returns `InvalidArgument` with argument `"a"`.
    #[test]
    fn test_dot_rank_high_lhs() {
        let a =
            Tensor::<i32, Ix2>::from_shape_vec((1, 1), vec![1_i32]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![1_i32]).expect("valid construction");
        let err = dot_impl(&a, &b).expect_err("must return error");
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

    /// A rank‑2 right tensor returns `InvalidArgument` with argument `"b"`.
    #[test]
    fn test_dot_rank_high_rhs() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![1_i32]).expect("valid construction");
        let b =
            Tensor::<i32, Ix2>::from_shape_vec((1, 1), vec![1_i32]).expect("valid construction");
        let err = dot_impl(&a, &b).expect_err("must return error");
        match err {
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::OperationSpecific { ref argument, .. },
                ..
            } => assert_eq!(argument.as_ref(), "b"),
            other => panic!("expected InvalidArgument::OperationSpecific for b, got {other:?}"),
        }
    }

    // --- dot_impl: integer overflow ------------------------------------------

    /// Integer overflow during multiplication panics with element context.
    #[test]
    #[should_panic(
        expected = "dot: integer overflow during multiplication at element 0 of shape [1] (type i32)"
    )]
    fn test_dot_int_overflow_mul() {
        let a = Tensor1::from_shape_vec(Ix1(1), vec![i32::MAX]).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(1), vec![2_i32]).expect("valid construction");
        let _ = dot_impl(&a, &b).expect("valid construction");
    }

    /// Integer overflow during accumulation panics with element context.
    #[test]
    #[should_panic(expected = "dot: integer overflow during accumulation at element")]
    fn test_dot_int_overflow_add() {
        let a = Tensor1::from_shape_vec(Ix1(3), vec![i32::MAX, 1, 1]).expect("valid construction");
        let b =
            Tensor1::from_shape_vec(Ix1(3), vec![1_i32, i32::MAX, 1]).expect("valid construction");
        let _ = dot_impl(&a, &b).expect("valid construction");
    }

    // --- dot_impl: SIMD path -------------------------------------------------

    /// SIMD path produces a result within tolerance of the serial baseline.
    #[cfg(feature = "simd")]
    #[test]
    fn test_dot_simd() {
        let values: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.5).collect();
        let a =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");
        let b =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");

        let actual = dot_impl(&a, &b).expect("valid construction");
        let expected = try_dot_serial(&a, &b);

        let max_abs = values.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance(values.len(), max_abs, max_abs);
        assert!(
            (actual - expected).abs() <= tol,
            "SIMD result {actual} exceeds tolerance {tol} vs serial {expected}"
        );
    }

    /// Unsupported element types (i64) fall back to the serial path silently.
    #[cfg(feature = "simd")]
    #[test]
    fn test_dot_simd_unsupported() {
        let values: Vec<i64> = (0..2048).collect();
        let a =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(values.len()), values).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), try_dot_serial(&a, &b));
    }

    // --- dot_impl: parallel path ---------------------------------------------

    /// Integer dot product via the parallel path matches the serial baseline exactly.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_dot_parallel_path() {
        let values: Vec<i64> = (0..4096_i64).collect();
        let a =
            Tensor1::from_shape_vec(Ix1(values.len()), values.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(values.len()), values).expect("valid construction");
        assert_eq!(dot_impl(&a, &b).expect("valid construction"), try_dot_serial(&a, &b));
    }

    /// Large‑vector parallel dot product stays within floating‑point tolerance.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_dot_parallel_large() {
        let n: usize = 100_000;
        let xs: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) * 0.1 + 1.0).collect();
        let ys: Vec<f64> = (0..n).map(|i| ((i % 13) as f64) * 0.1 + 2.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), xs.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), ys.clone()).expect("valid construction");
        let actual = dot_impl(&a, &b).expect("valid construction");
        let expected = try_dot_serial(&a, &b);
        let max_abs_a = xs.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs_b = ys.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance_parallel(n, max_abs_a, max_abs_b);
        assert!(
            (actual - expected).abs() <= tol,
            "parallel result {actual} exceeds tolerance {tol} vs serial {expected}"
        );
    }

    /// Nested parallel dot product falls back to serial and stays within tolerance.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_dot_parallel_nested() {
        let n: usize = 16_384;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 1.0).collect();
        let ys: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 + 2.0).collect();
        let a = Tensor1::from_shape_vec(Ix1(n), xs.clone()).expect("valid construction");
        let b = Tensor1::from_shape_vec(Ix1(n), ys.clone()).expect("valid construction");

        let baseline = try_dot_serial(&a, &b);

        let result = with_parallel_worker_context(|| dot_impl(&a, &b).expect("valid construction"));

        let max_abs_a = xs.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let max_abs_b = ys.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = f64_dot_tolerance(n, max_abs_a, max_abs_b);
        assert!(
            (result - baseline).abs() <= tol,
            "nested-parallel result {result} exceeds tolerance {tol} vs baseline {baseline}"
        );
    }
}
