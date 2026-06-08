//! Sum reducer implementations.
//!
//! Three public APIs: [`TensorBase::sum`], [`TensorBase::sum_axis`],
//! [`TensorBase::sum_axis_keepdims`].

use core::any::TypeId;
use std::borrow::Cow;

use crate::dimension::{Axis, Dimension, RemoveAxis};
use crate::dispatch::select_exec_path;
use crate::element::Numeric;

#[cfg(feature = "simd")]
use crate::complex::Complex;
use crate::error::XenonError;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;

#[cfg(feature = "simd")]
use crate::simd::{try_sum_complex_f32, try_sum_complex_f64, try_sum_f32, try_sum_f64};

pub(crate) fn sum_impl<S, D, A>(tensor: &TensorBase<S, D>) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    // 13-reduction §6.3: caller-side integer gate before dispatch.
    if force_scalar_for_integers::<A>() {
        return try_sum_serial(tensor);
    }

    let (path, _guard) =
        select_exec_path(tensor.len(), tensor.is_f_contiguous(), tensor.is_aligned());

    match path {
        #[cfg(feature = "parallel")]
        crate::dispatch::ExecPath::Parallel => {
            // §5.5 line 388-393: when select returns Parallel, the guard is always Some.
            let guard = _guard.expect("Parallel path implies Some(guard) per §5.5");
            // ParallelExecStrategy is held independently by the parallel/ backend
            // (30-dispatch.md §5.3 + Wave 10 audit memo); we construct the default
            // strategy locally and pass by reference.
            let strategy = ParallelExecStrategy::auto();
            crate::parallel::sum::par_sum(tensor, &strategy, guard)
        },
        #[cfg(feature = "simd")]
        crate::dispatch::ExecPath::Simd => {
            try_sum_simd(tensor).unwrap_or_else(|| try_sum_serial(tensor))
        },
        _ => try_sum_serial(tensor),
    }
}

// ── sum_axis implementation (W18T3) ──

pub(crate) fn sum_axis_impl<S, D, A>(
    tensor: &TensorBase<S, D>,
    axis: Axis,
) -> Result<Tensor<A, D::Smaller>, XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension + RemoveAxis,
    A: Numeric + Copy + 'static,
{
    validate_axis(&tensor.raw_dim(), axis, "sum_axis")?;
    // SAFETY-of-flow: axis is validated above, so remove_axis() cannot fail here.
    let (output_dim, _removed_len) = tensor.raw_dim().remove_axis(axis)?;
    let mut output = Tensor::<A, D::Smaller>::zeros(output_dim)?;
    accumulate_axis(tensor, axis, &mut output)?;
    Ok(output)
}

// ── sum_axis_keepdims implementation (W18T4) ──

pub(crate) fn sum_axis_keepdims_impl<S, D, A>(
    tensor: &TensorBase<S, D>,
    axis: Axis,
) -> Result<Tensor<A, D>, XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    validate_axis(&tensor.raw_dim(), axis, "sum_axis_keepdims")?;
    // SAFETY-of-flow: axis is validated above; dim_with_axis_set cannot fail
    // for axis OOB here. `try_from_slice` succeeds because the slice length
    // equals `ndim`, which `Dimension::try_from_slice` accepts.
    let output_dim = dim_with_axis_set(&tensor.raw_dim(), axis, 1, "sum_axis_keepdims")?;
    let mut output = Tensor::<A, D>::zeros(output_dim)?;
    accumulate_axis_keepdims(tensor, axis, &mut output)?;
    Ok(output)
}

/// Scalar serial baseline — same body as the W18T2 version, but now living
/// as a private helper so the dispatched `sum_impl` can fall back to it.
pub(crate) fn try_sum_serial<S, D, A>(tensor: &TensorBase<S, D>) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    // Capture shape before fold so the panic closure can reference it without
    // borrowing `tensor` mutably through `iter()` (mirrors W18T2 `sum_impl`).
    let shape_snapshot = tensor.shape().to_vec();
    tensor
        .iter()
        .enumerate()
        .fold(A::zero(), |acc, (index, &value)| {
            checked_add_step(acc, value).unwrap_or_else(|| {
                panic!(
                    "integer overflow in reduction sum: element_type={}, shape={:?}, element_index={}",
                    core::any::type_name::<A>(),
                    shape_snapshot,
                    index
                )
            })
        })
}

/// Type-dispatched wrapper for the W14 SIMD facades. Returns `None` when the
/// element type is unsupported (caller falls back to `try_sum_serial`) or when the
/// W14 facade itself rejects the input (e.g. shorter than the SIMD threshold).
/// The integer gate in `sum_impl` ensures `i32`/`i64` never reach this branch.
#[cfg(feature = "simd")]
fn try_sum_simd<S, D, A>(tensor: &TensorBase<S, D>) -> Option<A>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    // SIMD facades only consume contiguous slices. When `as_slice()` returns
    // `Some`, the underlying storage is F-contiguous and the slice spans the
    // logical elements (07-tensor.md §5.4 line 544).
    let slice: &[A] = tensor.as_slice()?;

    if TypeId::of::<A>() == TypeId::of::<f32>() {
        // SAFETY: TypeId equality proves `A == f32`.
        let s: &[f32] = unsafe { &*(slice as *const [A] as *const [f32]) };
        return try_sum_f32(s).map(|r| unsafe { core::mem::transmute_copy::<f32, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<f64>() {
        let s: &[f64] = unsafe { &*(slice as *const [A] as *const [f64]) };
        return try_sum_f64(s).map(|r| unsafe { core::mem::transmute_copy::<f64, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
        let s: &[Complex<f32>] = unsafe { &*(slice as *const [A] as *const [Complex<f32>]) };
        return try_sum_complex_f32(s)
            .map(|r| unsafe { core::mem::transmute_copy::<Complex<f32>, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
        let s: &[Complex<f64>] = unsafe { &*(slice as *const [A] as *const [Complex<f64>]) };
        return try_sum_complex_f64(s)
            .map(|r| unsafe { core::mem::transmute_copy::<Complex<f64>, A>(&r) });
    }
    None
}

// ── Axis validation helper (W18T3) ──

/// `pub(crate)` so W18T4 can reuse it without duplicating the InvalidAxis logic.
pub(crate) fn validate_axis<D: Dimension>(
    shape: &D,
    axis: Axis,
    operation: &'static str,
) -> Result<(), XenonError> {
    if axis.index() >= shape.ndim() {
        return Err(XenonError::InvalidAxis {
            operation: Cow::Borrowed(operation),
            axis: axis.index(),
            ndim: shape.ndim(),
            shape: shape.slice().to_vec(),
        });
    }
    Ok(())
}

/// Per-slot accumulation over a single axis, mapping input indexed coordinates
/// onto reduced output coordinates. Uses `checked_add_step` (defined in W18T2)
/// so integer overflow panics with element context per 13-reduction §1.1, and
/// float / complex addition follows IEEE 754 via `Numeric + Add`.
fn accumulate_axis<S, D, A>(
    tensor: &TensorBase<S, D>,
    axis: Axis,
    output: &mut Tensor<A, D::Smaller>,
) -> Result<(), XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension + RemoveAxis,
    A: Numeric + Copy + 'static,
{
    for (input_index, &value) in tensor.indexed_iter() {
        // `remove_axis` returns Result<Self::Smaller, _>; axis was pre-validated
        // by `validate_axis(...)` above so this projection cannot fail.
        let (output_index, _removed_len) = input_index
            .remove_axis(axis)
            .expect("axis pre-validated; remove_axis cannot fail here");
        // `TensorBase` does not implement `IndexMut<D>`. Use `get_mut`
        // with `&[usize]` (via `Dimension::slice`) to avoid requiring
        // `NdIndex<D::Smaller>` tuple-form impls. The lookup cannot fail
        // because `output_index` derives from a coordinate that already maps
        // inside `output`'s logical range.
        let slot = output
            .get_mut(output_index.slice())
            .expect("output_index is within reduced-shape bounds");
        *slot = checked_add_step(*slot, value).unwrap_or_else(|| {
            panic!(
                "integer overflow in reduction sum_axis: element_type={}, shape={:?}, input_index={:?}",
                core::any::type_name::<A>(),
                tensor.shape(),
                input_index
            )
        });
    }
    Ok(())
}

/// Per-slot accumulation, mapping input coordinates onto keepdims output
/// coordinates by forcing the reduced axis component to `0`. Numeric
/// semantics inherit from `checked_add_step` (13-reduction §6.1):
/// integers panic on overflow, floats / complex follow IEEE 754.
fn accumulate_axis_keepdims<S, D, A>(
    tensor: &TensorBase<S, D>,
    axis: Axis,
    output: &mut Tensor<A, D>,
) -> Result<(), XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    for (input_index, &value) in tensor.indexed_iter() {
        // axis was pre-validated; this construction cannot fail.
        let output_index = dim_with_axis_set(&input_index, axis, 0, "sum_axis_keepdims")
            .expect("axis pre-validated; dim_with_axis_set cannot fail here");
        // `TensorBase` does not impl `IndexMut<D>`; use `get_mut`
        // (17-indexing.md §5.2). Index is provably in-bounds because
        // `output` has the reduced axis length 1 and `output_index[axis] = 0`.
        let slot = output
            .get_mut(output_index.slice())
            .expect("output_index is within keepdims-shape bounds");
        *slot = checked_add_step(*slot, value).unwrap_or_else(|| {
            panic!(
                "integer overflow in reduction sum_axis_keepdims: element_type={}, shape={:?}, input_index={:?}",
                core::any::type_name::<A>(),
                tensor.shape(),
                input_index
            )
        });
    }
    Ok(())
}

// ── dim_with_axis_set helper (W18T4) ──

/// Construct a new dimension whose `axis` component has been replaced with
/// `value`, leaving every other axis unchanged.
///
/// Uses only the stable `Dimension` API surface from
/// `02-dimension.md §5.1`: `slice(&self) -> &[usize]` for read-out and
/// `try_from_slice(&[usize]) -> Result<Self, _>` for reconstruction. This
/// keeps the 0D / static-rank / dynamic-rank cases on the same code path
/// and avoids extending the public trait surface.
///
/// Returns `XenonError::InvalidAxis` when `axis.index() >= dim.ndim()`,
/// matching `13-reduction.md §5.2` axis OOB error contract.
pub(crate) fn dim_with_axis_set<D: Dimension>(
    dim: &D,
    axis: Axis,
    value: usize,
    operation: &'static str,
) -> Result<D, XenonError> {
    let mut dims: Vec<usize> = dim.slice().to_vec();
    if axis.index() >= dims.len() {
        return Err(XenonError::InvalidAxis {
            operation: Cow::Borrowed(operation),
            axis: axis.index(),
            ndim: dims.len(),
            shape: dims,
        });
    }
    dims[axis.index()] = value;
    D::try_from_slice(&dims)
}

/// 13-reduction §6.3 line 291-295: integer types skip the dispatcher when no
/// verified widening SIMD implementation exists, falling directly to the
/// scalar serial path for checked arithmetic equivalence.
fn force_scalar_for_integers<A: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<i32>() || TypeId::of::<A>() == TypeId::of::<i64>()
}

/// Per-step accumulation enforcing 13-reduction §6.3 type semantics:
/// - Integer types (`i32`, `i64`): checked arithmetic via `CheckedAdd`,
///   returning `None` on overflow so the caller can panic with element context.
/// - Floating / complex types: ordinary `+` via `Numeric: Add<Output = Self>`;
///   IEEE 754 NaN / Inf propagation is preserved.
///
/// Dispatch mirrors the 13-reduction §6.3 line 291-295 SIMD type-gate pattern.
/// The `unsafe` reads are sound because each is gated by
/// `TypeId::of::<A>() == TypeId::of::<I>()`, which proves layout identity.
#[inline]
pub(crate) fn checked_add_step<A>(acc: A, value: A) -> Option<A>
where
    A: Numeric + Copy + 'static,
{
    if TypeId::of::<A>() == TypeId::of::<i32>() {
        // SAFETY: TypeId equality proves `A == i32`, so `&A` and `&i32` are
        // pointer-compatible and reading through a `*const i32` is sound.
        let a: i32 = unsafe { *(&acc as *const A as *const i32) };
        let v: i32 = unsafe { *(&value as *const A as *const i32) };
        return a
            .checked_add(v)
            // SAFETY: `A == i32`; reinterpreting `i32` as `A` is identity.
            .map(|r| unsafe { core::mem::transmute_copy::<i32, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<i64>() {
        // SAFETY: TypeId equality proves `A == i64`.
        let a: i64 = unsafe { *(&acc as *const A as *const i64) };
        let v: i64 = unsafe { *(&value as *const A as *const i64) };
        return a
            .checked_add(v)
            // SAFETY: `A == i64`; transmute is identity.
            .map(|r| unsafe { core::mem::transmute_copy::<i64, A>(&r) });
    }
    // Float / complex path: `Numeric: Add<Output = Self>` covers all remaining
    // supported element types (`f32`, `f64`, `Complex<f32>`, `Complex<f64>`).
    Some(acc + value)
}



// ── Unit tests for internal/private helpers ──

#[cfg(test)]
mod tests {
    use crate::complex::Complex;
    use crate::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, RemoveAxis};
    use crate::error::XenonError;
    use crate::tensor::{Tensor, Tensor1};

    // ── checked_add_step ──

    #[test]
    fn test_checked_add_step_i32_normal() {
        assert_eq!(super::checked_add_step(1_i32, 2), Some(3));
        assert_eq!(super::checked_add_step(-5_i32, 3), Some(-2));
    }

    #[test]
    fn test_checked_add_step_i32_overflow() {
        assert!(super::checked_add_step(i32::MAX, 1).is_none());
    }

    #[test]
    fn test_checked_add_step_i32_underflow() {
        assert!(super::checked_add_step(i32::MIN, -1).is_none());
    }

    #[test]
    fn test_checked_add_step_i64_normal() {
        assert_eq!(super::checked_add_step(100_i64, 200), Some(300));
    }

    #[test]
    fn test_checked_add_step_i64_overflow() {
        assert!(super::checked_add_step(i64::MAX, 1).is_none());
    }

    #[test]
    fn test_checked_add_step_f32_normal() {
        assert_eq!(super::checked_add_step(1.5_f32, 2.5), Some(4.0));
    }

    #[test]
    fn test_checked_add_step_f32_nan_propagates() {
        let result = super::checked_add_step(1.0_f32, f32::NAN);
        assert!(result.unwrap().is_nan());
    }

    #[test]
    fn test_checked_add_step_f64_normal() {
        assert_eq!(super::checked_add_step(1.5_f64, 2.5), Some(4.0));
    }

    #[test]
    fn test_checked_add_step_complex_f64_normal() {
        let a = Complex::<f64>::new(1.0, 2.0);
        let b = Complex::<f64>::new(3.0, 4.0);
        let result = super::checked_add_step(a, b);
        assert_eq!(result, Some(Complex::new(4.0, 6.0)));
    }

    #[test]
    fn test_checked_add_step_zero_identity() {
        // i32 zero + something = something
        assert_eq!(super::checked_add_step(0_i32, 42), Some(42));
        // f64 zero + something = something
        assert_eq!(super::checked_add_step(0.0_f64, 3.14), Some(3.14));
    }

    // ── force_scalar_for_integers ──

    #[test]
    fn test_force_scalar_i32_true() {
        assert!(super::force_scalar_for_integers::<i32>());
    }

    #[test]
    fn test_force_scalar_i64_true() {
        assert!(super::force_scalar_for_integers::<i64>());
    }

    #[test]
    fn test_force_scalar_f32_false() {
        assert!(!super::force_scalar_for_integers::<f32>());
    }

    #[test]
    fn test_force_scalar_f64_false() {
        assert!(!super::force_scalar_for_integers::<f64>());
    }

    #[test]
    fn test_force_scalar_complex_f32_false() {
        assert!(!super::force_scalar_for_integers::<Complex<f32>>());
    }

    // ── validate_axis ──

    #[test]
    fn test_validate_axis_valid() {
        let dim = Ix2(3, 4);
        assert!(super::validate_axis(&dim, Axis(0), "test_op").is_ok());
        assert!(super::validate_axis(&dim, Axis(1), "test_op").is_ok());
    }

    #[test]
    fn test_validate_axis_invalid() {
        let dim = Ix2(3, 4);
        let err = super::validate_axis(&dim, Axis(2), "test_op").unwrap_err();
        assert!(matches!(err, XenonError::InvalidAxis { .. }));
    }

    #[test]
    fn test_validate_axis_0d_always_invalid() {
        let dim = Ix0;
        let err = super::validate_axis(&dim, Axis(0), "test_op").unwrap_err();
        assert!(matches!(err, XenonError::InvalidAxis { .. }));
    }

    // ── dim_with_axis_set ──

    #[test]
    fn test_dim_with_axis_set_valid() {
        let dim = Ix2(3, 4);
        let result = super::dim_with_axis_set(&dim, Axis(1), 7, "test_op").unwrap();
        assert_eq!(result.slice(), &[3, 7]);
    }

    #[test]
    fn test_dim_with_axis_set_oob() {
        let dim = Ix2(3, 4);
        let err = super::dim_with_axis_set(&dim, Axis(2), 1, "test_op").unwrap_err();
        assert!(matches!(err, XenonError::InvalidAxis { .. }));
    }

    // ── try_sum_serial ──

    #[test]
    fn test_try_sum_serial_i32() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3]).unwrap();
        assert_eq!(super::try_sum_serial(&x), 6);
    }

    #[test]
    fn test_try_sum_serial_empty() {
        let x = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![]).unwrap();
        assert_eq!(super::try_sum_serial(&x), 0.0);
    }

    #[test]
    fn test_try_sum_serial_nan_propagates() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![1.0_f64, f64::NAN]).unwrap();
        assert!(super::try_sum_serial(&x).is_nan());
    }

    #[test]
    #[should_panic(expected = "integer overflow")]
    fn test_try_sum_serial_i32_overflow_panics() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).unwrap();
        super::try_sum_serial(&x);
    }

    // ── accumulate_axis: integer overflow panic path ──

    #[test]
    #[should_panic(expected = "integer overflow")]
    fn test_accumulate_axis_overflow_panics() {
        // sum_axis over axis 0 with shape (2,) and elements [i32::MAX, 1].
        let x = Tensor::<i32, Ix1>::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).unwrap();
        // sum_axis on a 1D tensor; remove_axis reduces to Ix0.
        let output_dim = x.raw_dim().remove_axis(Axis(0)).unwrap().0;
        let mut output = Tensor::<i32, Ix0>::zeros(output_dim).unwrap();
        super::accumulate_axis(&x, Axis(0), &mut output).unwrap();
    }

    // ── accumulate_axis_keepdims: integer overflow panic path ──

    #[test]
    #[should_panic(expected = "integer overflow")]
    fn test_accumulate_axis_keepdims_overflow_panics() {
        let x = Tensor::<i32, Ix1>::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).unwrap();
        let output_dim = super::dim_with_axis_set(&x.raw_dim(), Axis(0), 1, "test").unwrap();
        let mut output = Tensor::<i32, Ix1>::zeros(output_dim).unwrap();
        super::accumulate_axis_keepdims(&x, Axis(0), &mut output).unwrap();
    }
}
