//! Sum reducer implementations.
//!
//! Internal functions backing the public methods [`TensorBase::sum`],
//! [`TensorBase::sum_axis`], and [`TensorBase::sum_axis_keepdims`].

use core::any::{TypeId, type_name};
use core::mem::transmute_copy;
use std::borrow::Cow;

use crate::error::XenonError;
use crate::dimension::{Axis, Dimension, RemoveAxis};
use crate::element::Numeric;
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};
use crate::dispatch::{select_exec_path, ExecPath};

#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;

#[cfg(feature = "parallel")]
use crate::parallel::sum::par_sum;

#[cfg(feature = "simd")]
use crate::complex::Complex;

#[cfg(feature = "simd")]
use crate::simd::{
    try_sum_f32,
    try_sum_f64,
    try_sum_complex_f32,
    try_sum_complex_f64
};

// --- sum_impl ---------------------------------------------------------------

/// Reduce all elements to a single scalar via dispatch to serial, SIMD, or
/// parallel paths.
///
/// Integer types (`i32`, `i64`) are gated to the serial path for checked
/// arithmetic. Other types are dispatched via [`select_exec_path`].
pub(crate) fn sum_impl<S, D, A>(tensor: &TensorBase<S, D>) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    // Integer types skip the dispatcher; no verified widening SIMD
    // implementation exists for them so we use the scalar serial path.
    if force_scalar_for_integers::<A>() {
        return try_sum_serial(tensor);
    }

    let (path, _guard) = select_exec_path(
        tensor.len(),
        tensor.is_f_contiguous(),
        tensor.is_aligned()
    );

    match path {
        #[cfg(feature = "parallel")]
        ExecPath::Parallel => {
            // When select_exec_path returns Parallel, the guard is always Some.
            let guard = _guard.expect("Parallel path implies Some(guard)");
            let strategy = ParallelExecStrategy::auto();
            par_sum(tensor, &strategy, guard)
        },
        #[cfg(feature = "simd")]
        ExecPath::Simd => {
            try_sum_simd(tensor).unwrap_or_else(|| try_sum_serial(tensor))
        },
        _ => try_sum_serial(tensor),
    }
}

// --- sum_axis_impl ----------------------------------------------------------

/// Reduce along a single axis, removing that axis from the output shape.
///
/// Validates the axis, constructs a zero-filled output tensor of the reduced
/// shape, then accumulates input elements into the corresponding output slots.
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

// --- sum_axis_keepdims_impl -------------------------------------------------

/// Reduce along a single axis, keeping the reduced axis with length 1.
///
/// Validates the axis, constructs a zero-filled output tensor with the
/// reduced axis set to length 1, then accumulates input elements into the
/// corresponding output slots.
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
    let output_dim = dim_with_axis_set(
        &tensor.raw_dim(),
        axis,
        1,
        "sum_axis_keepdims"
    )?;
    let mut output = Tensor::<A, D>::zeros(output_dim)?;
    accumulate_axis_keepdims(tensor, axis, &mut output)?;
    Ok(output)
}

// --- Serial & SIMD sum ------------------------------------------------------

/// Serial fallback for sum reduction.
///
/// Iterates over all elements, accumulating with [`checked_add_step`].
/// Captures the shape snapshot before iteration so the panic closure can
/// reference it without holding a mutable borrow on the tensor.
pub(crate) fn try_sum_serial<S, D, A>(tensor: &TensorBase<S, D>) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    let shape_snapshot = tensor.shape().to_vec();
    tensor
        .iter()
        .enumerate()
        .fold(A::zero(), |acc, (index, &value)| {
            checked_add_step(acc, value).unwrap_or_else(|| {
                panic!(
                    "integer overflow in reduction sum: \
                     element_type={}, shape={:?}, element_index={}",
                    type_name::<A>(),
                    shape_snapshot,
                    index
                )
            })
        })
}

/// Type-dispatched SIMD sum.
///
/// Returns `None` when the element type is unsupported by the SIMD backend
/// or the input is too short to benefit from SIMD. The caller falls back to
/// [`try_sum_serial`] on `None`.
///
/// Integer types (`i32`, `i64`) never reach this branch; they are gated by
/// `force_scalar_for_integers` in [`sum_impl`].
#[cfg(feature = "simd")]
fn try_sum_simd<S, D, A>(tensor: &TensorBase<S, D>) -> Option<A>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy + 'static,
{
    // SIMD facades only consume contiguous slices. When `as_slice()` returns
    // `Some`, the underlying storage is F-contiguous and the slice spans the
    // logical elements.
    let slice: &[A] = tensor.as_slice()?;

    if TypeId::of::<A>() == TypeId::of::<f32>() {
        // SAFETY: TypeId equality proves `A == f32`.
        let s: &[f32] = unsafe {
            &*(slice as *const [A] as *const [f32])
        };
        return try_sum_f32(s)
            .map(|r| unsafe { transmute_copy::<f32, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<f64>() {
        let s: &[f64] = unsafe {
            &*(slice as *const [A] as *const [f64])
        };
        return try_sum_f64(s)
            .map(|r| unsafe { transmute_copy::<f64, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
        let s: &[Complex<f32>] = unsafe {
            &*(slice as *const [A] as *const [Complex<f32>])
        };
        return try_sum_complex_f32(s)
            .map(|r| unsafe { transmute_copy::<Complex<f32>, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
        let s: &[Complex<f64>] = unsafe {
            &*(slice as *const [A] as *const [Complex<f64>])
        };
        return try_sum_complex_f64(s)
            .map(|r| unsafe { transmute_copy::<Complex<f64>, A>(&r) });
    }
    None
}

// --- Axis validation --------------------------------------------------------

/// Validate that `axis` is within bounds for the given dimension shape.
///
/// Returns `Ok(())` if valid, or `Err(XenonError::InvalidAxis)` with the
/// operation name if `axis.index() >= shape.ndim()`.
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

/// Accumulate input elements into an output tensor by mapping each input
/// indexed coordinate onto a reduced output coordinate (removing one axis).
///
/// Uses [`checked_add_step`] for accumulation: integer overflow panics with
/// element context, float / complex follows IEEE 754 via `Numeric + Add`.
///
/// The caller must have pre-validated the axis via [`validate_axis`].
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
        // Axis was pre-validated; remove_axis projection cannot fail.
        let (output_index, _removed_len) = input_index
            .remove_axis(axis)
            .expect("axis pre-validated; remove_axis cannot fail here");
        let slot = output
            .get_mut(output_index.slice())
            .expect("output_index is within reduced-shape bounds");
        *slot = checked_add_step(*slot, value).unwrap_or_else(|| {
            panic!(
                "integer overflow in reduction sum_axis: \
                 element_type={}, shape={:?}, input_index={:?}",
                type_name::<A>(),
                tensor.shape(),
                input_index
            )
        });
    }
    Ok(())
}

/// Accumulate input elements into a keepdims output tensor by mapping each
/// input coordinate onto the output coordinate with the reduced axis forced
/// to `0`.
///
/// Uses [`checked_add_step`] for accumulation: integer overflow panics with
/// element context, float / complex follows IEEE 754 via `Numeric + Add`.
///
/// The caller must have pre-validated the axis via [`validate_axis`].
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
        // Axis was pre-validated; dim_with_axis_set cannot fail.
        let output_index = dim_with_axis_set(
            &input_index,
            axis,
            0,
            "sum_axis_keepdims"
        ).expect("axis pre-validated; dim_with_axis_set cannot fail here");
        let slot = output
            .get_mut(output_index.slice())
            .expect("output_index is within keepdims-shape bounds");
        *slot = checked_add_step(*slot, value).unwrap_or_else(|| {
            panic!(
                "integer overflow in reduction sum_axis_keepdims: \
                 element_type={}, shape={:?}, input_index={:?}",
                type_name::<A>(),
                tensor.shape(),
                input_index
            )
        });
    }
    Ok(())
}

// --- Internal helpers -------------------------------------------------------

/// Construct a new dimension identical to `dim` except that the component
/// at `axis` is replaced with `value`.
///
/// Uses the stable `Dimension` API (`slice` for read-out, `try_from_slice`
/// for reconstruction) so the same code path handles 0D, static-rank, and
/// dynamic-rank dimensions.
///
/// # Errors
///
/// Returns `XenonError::InvalidAxis` when `axis.index() >= dim.ndim()`.
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

/// Returns `true` for `i32` and `i64`, directing those element types to the
/// scalar serial path for checked arithmetic.
///
/// Float and complex types return `false`, allowing SIMD or parallel dispatch.
fn force_scalar_for_integers<A: 'static>() -> bool {
    let id = TypeId::of::<A>();
    id == TypeId::of::<i32>() || id == TypeId::of::<i64>()
}

/// Per-step accumulation with type-aware arithmetic semantics.
///
/// - `i32`, `i64`: checked addition via `CheckedAdd`, returning `None` on
///   overflow so the caller can panic with element context.
/// - `f32`, `f64`, `Complex<f32>`, `Complex<f64>`: ordinary `+` via
///   `Numeric: Add<Output = Self>`, preserving IEEE 754 NaN / Inf propagation.
///
/// # Safety
///
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
            .map(|r| unsafe { transmute_copy::<i32, A>(&r) });
    }
    if TypeId::of::<A>() == TypeId::of::<i64>() {
        // SAFETY: TypeId equality proves `A == i64`.
        let a: i64 = unsafe { *(&acc as *const A as *const i64) };
        let v: i64 = unsafe { *(&value as *const A as *const i64) };
        return a
            .checked_add(v)
            // SAFETY: `A == i64`; transmute is identity.
            .map(|r| unsafe { transmute_copy::<i64, A>(&r) });
    }
    // Float / complex path: `Numeric: Add<Output = Self>` covers all remaining
    // supported element types.
    Some(acc + value)
}

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::complex::Complex;
    use crate::dimension::{Ix0, Ix1, Ix2};
    use crate::dimension::{Axis, Dimension, RemoveAxis};
    use crate::tensor::{Tensor, Tensor1};

    // --- checked_add_step ---------------------------------------------------

    /// i32 normal addition returns the sum.
    #[test]
    fn test_checked_add_step_i32_normal() {
        assert_eq!(super::checked_add_step(1_i32, 2), Some(3));
        assert_eq!(super::checked_add_step(-5_i32, 3), Some(-2));
    }

    /// i32 overflow returns None.
    #[test]
    fn test_checked_add_step_i32_overflow() {
        assert!(super::checked_add_step(i32::MAX, 1).is_none());
    }

    /// i32 underflow (MIN + -1) returns None.
    #[test]
    fn test_checked_add_step_i32_underflow() {
        assert!(super::checked_add_step(i32::MIN, -1).is_none());
    }

    /// i64 normal addition returns the sum.
    #[test]
    fn test_checked_add_step_i64_normal() {
        assert_eq!(super::checked_add_step(100_i64, 200), Some(300));
    }

    /// i64 overflow returns None.
    #[test]
    fn test_checked_add_step_i64_overflow() {
        assert!(super::checked_add_step(i64::MAX, 1).is_none());
    }

    /// f32 normal addition returns the sum.
    #[test]
    fn test_checked_add_step_f32_normal() {
        assert_eq!(super::checked_add_step(1.5_f32, 2.5), Some(4.0));
    }

    /// f32 NaN propagates through addition.
    #[test]
    fn test_checked_add_step_f32_nan_propagates() {
        let result = super::checked_add_step(1.0_f32, f32::NAN);
        assert!(result.expect("NaN must produce Some").is_nan());
    }

    /// f64 normal addition returns the sum.
    #[test]
    fn test_checked_add_step_f64_normal() {
        assert_eq!(super::checked_add_step(1.5_f64, 2.5), Some(4.0));
    }

    /// Complex f64 addition sums real and imaginary parts independently.
    #[test]
    fn test_checked_add_step_complex_f64_normal() {
        let a = Complex::<f64>::new(1.0, 2.0);
        let b = Complex::<f64>::new(3.0, 4.0);
        let result = super::checked_add_step(a, b);
        assert_eq!(result, Some(Complex::new(4.0, 6.0)));
    }

    /// Adding zero is an identity operation for both integer and float types.
    #[test]
    fn test_checked_add_step_zero_identity() {
        assert_eq!(super::checked_add_step(0_i32, 42), Some(42));
        assert_eq!(super::checked_add_step(0.0_f64, 2.5), Some(2.5));
    }

    // --- force_scalar_for_integers ------------------------------------------

    /// i32 is classified as a scalar-only integer type.
    #[test]
    fn test_force_scalar_i32_true() {
        assert!(super::force_scalar_for_integers::<i32>());
    }

    /// i64 is classified as a scalar-only integer type.
    #[test]
    fn test_force_scalar_i64_true() {
        assert!(super::force_scalar_for_integers::<i64>());
    }

    /// f32 is not an integer type and may use SIMD/parallel paths.
    #[test]
    fn test_force_scalar_f32_false() {
        assert!(!super::force_scalar_for_integers::<f32>());
    }

    /// f64 is not an integer type and may use SIMD/parallel paths.
    #[test]
    fn test_force_scalar_f64_false() {
        assert!(!super::force_scalar_for_integers::<f64>());
    }

    /// Complex f32 is not an integer type and may use SIMD/parallel paths.
    #[test]
    fn test_force_scalar_complex_f32_false() {
        assert!(!super::force_scalar_for_integers::<Complex<f32>>());
    }

    // -------------------------- dim_with_axis_set ---------------------------

    /// Replacing a valid axis component produces the expected dimension.
    #[test]
    fn test_dim_with_axis_set_valid() {
        let dim = Ix2(3, 4);
        let result = super::dim_with_axis_set(&dim, Axis(1), 7, "test_op")
            .expect("valid axis");
        assert_eq!(result.slice(), &[3, 7]);
    }

    /// An out-of-bounds axis returns InvalidAxis error.
    #[test]
    fn test_dim_with_axis_set_oob() {
        let dim = Ix2(3, 4);
        let err = super::dim_with_axis_set(&dim, Axis(2), 1, "test_op")
            .expect_err("expected InvalidAxis");
        assert!(matches!(err, XenonError::InvalidAxis { .. }));
    }

    // --- validate_axis ------------------------------------------------------

    /// Valid axes within [0, ndim) return Ok(()).
    #[test]
    fn test_validate_axis_valid() {
        let dim = Ix2(3, 4);
        assert!(super::validate_axis(&dim, Axis(0), "test_op").is_ok());
        assert!(super::validate_axis(&dim, Axis(1), "test_op").is_ok());
    }

    /// An axis equal to ndim returns InvalidAxis error.
    #[test]
    fn test_validate_axis_invalid() {
        let dim = Ix2(3, 4);
        let err = super::validate_axis(&dim, Axis(2), "test_op")
            .expect_err("expected InvalidAxis");
        assert!(matches!(err, XenonError::InvalidAxis { .. }));
    }

    /// A 0D dimension has no valid axes.
    #[test]
    fn test_validate_axis_0d_always_invalid() {
        let dim = Ix0;
        let err = super::validate_axis(&dim, Axis(0), "test_op")
            .expect_err("expected InvalidAxis");
        assert!(matches!(err, XenonError::InvalidAxis { .. }));
    }

    // --- try_sum_serial -----------------------------------------------------

    /// Serial sum of i32 elements equals their arithmetic total.
    #[test]
    fn test_try_sum_serial_i32() {
        let x = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3])
            .expect("valid test input");
        assert_eq!(super::try_sum_serial(&x), 6);
    }

    /// Serial sum of an empty tensor returns the additive identity.
    #[test]
    fn test_try_sum_serial_empty() {
        let x = Tensor1::<f64>::from_shape_vec(Ix1(0), vec![])
            .expect("valid test input");
        assert_eq!(super::try_sum_serial(&x), 0.0);
    }

    /// Serial sum propagates f64 NaN per IEEE 754.
    #[test]
    fn test_try_sum_serial_nan_propagates() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![1.0_f64, f64::NAN])
            .expect("valid test input");
        assert!(super::try_sum_serial(&x).is_nan());
    }

    /// Serial sum panics on i32 overflow with an "integer overflow" message.
    #[test]
    #[should_panic(expected = "integer overflow")]
    fn test_try_sum_serial_i32_overflow_panics() {
        let x = Tensor1::from_shape_vec(Ix1(2), vec![i32::MAX, 1])
            .expect("valid test input");
        super::try_sum_serial(&x);
    }

    // ----------------------- accumulate_axis overflow -----------------------

    /// accumulate_axis panics on i32 overflow in the reduction loop.
    #[test]
    #[should_panic(expected = "integer overflow")]
    fn test_accumulate_axis_overflow_panics() {
        let x = Tensor::<i32, Ix1>::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).expect("valid test input");
        let output_dim = x.raw_dim().remove_axis(Axis(0)).expect("axis 0 is valid").0;
        let mut output = Tensor::<i32, Ix0>::zeros(output_dim).expect("valid shape");
        let _ = super::accumulate_axis(&x, Axis(0), &mut output);
    }

    // ------------------ accumulate_axis_keepdims overflow -------------------

    /// accumulate_axis_keepdims panics on i32 overflow in the reduction loop.
    #[test]
    #[should_panic(expected = "integer overflow")]
    fn test_accumulate_axis_keepdims_overflow_panics() {
        let x = Tensor::<i32, Ix1>::from_shape_vec(Ix1(2), vec![i32::MAX, 1]).expect("valid test input");
        let output_dim = super::dim_with_axis_set(&x.raw_dim(), Axis(0), 1, "test").expect("valid axis");
        let mut output = Tensor::<i32, Ix1>::zeros(output_dim).expect("valid shape");
        let _ = super::accumulate_axis_keepdims(&x, Axis(0), &mut output);
    }
}
