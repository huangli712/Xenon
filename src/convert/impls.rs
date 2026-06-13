//! `TensorBase` method impls: `cast()`, `to_owned()`, and `into_owned()`.
//!
//! Contains the three `impl TensorBase<S, D>` blocks with helpers that
//! provide tensor-level type conversion and ownership transfer.
//!
//! `CastTo` trait and tier impls are in `super::cast`.
//! `SealedElement` is defined in `crate::element`.

use std::borrow::Cow;

use crate::error::{Result, XenonError};
use crate::dimension::Dimension;
use crate::element::{Element, SealedElement};
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{RawStorage, Storage, StorageIntoOwned, Owned};
use crate::tensor::{Tensor, TensorBase};
use super::{cast::CastTo};

/// Thin wrapper around `TensorBase::new_unchecked` for zero-overhead Owned
/// construction from a validated `(shape, Vec<A>)` pair.
///
/// # Safety
///
/// Caller must prove:
/// - `shape.checked_size()` is `Ok` (no overflow)
/// - `data.len() == shape.checked_size().unwrap()`
///
/// Used by `cast()` and `to_owned()` after they have already proven
/// length / shape consistency at the call site.
#[inline]
pub(crate) unsafe fn from_shape_vec_aligned_unchecked<A, D>(
    shape: D,
    data: Vec<A>
) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    unsafe { Tensor::from_raw_vec_unchecked(data, shape) }
}

/// Rewraps a `TypeConversion` error from element-level (`"cast_to"`,
/// `element_index = None`) to tensor-level (`"cast"`, `element_index = Some(idx)`).
fn rewrap_cast_error(error: XenonError, index: usize) -> XenonError {
    match error {
        XenonError::TypeConversion {
            source_type,
            target_type,
            reason,
            ..
        } => XenonError::TypeConversion {
            operation: Cow::Borrowed("cast"),
            source_type,
            target_type,
            reason,
            element_index: Some(index),
        },
        other => other,
    }
}

/// Reconstruct a canonical F-order `Tensor<A, D>` from an `Owned<A>` storage
/// produced by `StorageIntoOwned::into_owned_storage()`.
///
/// Self-contained helper local to the convert module.
///
/// # Panics
///
/// Panics if `Strides::f_contiguous` fails on `dim`. This cannot happen because
/// `dim` originates from a valid `TensorBase` whose shape was already validated.
#[inline]
pub(crate) fn into_owned_from_owned_storage<A, D>(
    owned_storage: Owned<A>,
    dim: D
) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    let strides = Strides::f_contiguous(&dim)
        .expect("validated dim from TensorBase");
    let flags = compute_layout_flags(&dim, &strides, owned_storage.as_ptr());
    // SAFETY: `into_owned_storage()` guarantees `owned_storage.len() == product(dim)`;
    // `dim` originated from a valid `TensorBase`; canonical F-order strides
    // and flags are computed above.
    unsafe {
        TensorBase::new_unchecked(
            owned_storage,
            dim,
            strides,
            0, /* offset = */
            flags,
            false, /* derived_from_view_mut = */
        )
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: SealedElement + Copy,
{
    /// Element-wise type conversion to target element type `B`.
    ///
    /// Returns an owned `Tensor<B, D>` regardless of input storage mode.
    ///
    /// # Type Parameters
    ///
    /// * `B` - Target element type, must implement `SealedElement`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::TypeConversion` when any element cannot be
    /// converted to `B`, tagged with the failing element's index.
    #[expect(
        private_bounds,
        reason = "CastTo is pub(crate) sealed; public cast() is gated by it"
    )]
    pub fn cast<B>(&self) -> Result<Tensor<B, D>>
    where
        B: SealedElement,
        A: CastTo<B>,
    {
        let mut data: Vec<B> = Vec::with_capacity(self.len());
        for (index, value) in self.iter().copied().enumerate() {
            let converted = value
                .cast_to()
                .map_err(|error| rewrap_cast_error(error, index))?;
            data.push(converted);
        }
        // SAFETY: data.len() == self.len() == product(self.raw_dim()) because
        // the loop pushes exactly one element per F-order iteration over self.
        // self.raw_dim() was validated when self was constructed.
        Ok(unsafe { from_shape_vec_aligned_unchecked(self.raw_dim(), data) })
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Clones logical elements into a new owned tensor in canonical F-order.
    ///
    /// Skips padding and offset; always allocates fresh storage.
    pub fn to_owned(&self) -> Tensor<A, D> {
        let mut data: Vec<A> = Vec::with_capacity(self.len());
        for elem in self.iter().cloned() {
            data.push(elem);
        }
        // SAFETY: data.len() == self.len() == product(self.raw_dim()).
        unsafe { from_shape_vec_aligned_unchecked(self.raw_dim(), data) }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Consume the tensor into an owned tensor.
    ///
    /// - `Tensor` (S=`Owned<A>`): returned directly, O(1), same data
    /// - `TensorView` / `TensorViewMut` / `ArcTensor`: O(n) allocate+copy
    ///   into canonical F-order
    pub fn into_owned(self) -> Tensor<A, D> {
        let owned_storage = self.storage.into_owned_storage();
        into_owned_from_owned_storage(owned_storage, self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ConversionFailureReason;
    use crate::dimension::Ix1;
    use crate::tensor::Tensor1;

    /// Tensor-level `cast()` performs Tier-1 `i32` → `f64` conversion
    /// element-wise and returns an owned tensor.
    #[test]
    fn test_cast_i32_to_f64() {
        let tensor: Tensor1<i32> = unsafe {
            Tensor1::from_raw_vec_unchecked(vec![1_i32, 2, 3], Ix1(3))
        };
        let converted: Tensor1<f64> = tensor.cast()
            .expect("i32→f64 cast should succeed");
        let result: Vec<f64> = converted.iter().copied().collect();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    /// A failed `cast()` reports the index of the first non-convertible element.
    #[test]
    fn test_cast_reports_element_index() {
        let tensor: Tensor1<f64> = unsafe {
            Tensor1::from_raw_vec_unchecked(vec![1.0_f64, 2.0], Ix1(2))
        };
        let error = tensor.cast::<f32>()
            .expect_err("f64→f32 cast should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                element_index: Some(0),
                ..
            }
        ));
    }

    /// Casting `NaN` to an integer type fails with `FloatToInteger`.
    #[test]
    fn test_cast_nan_to_int_returns_error() {
        let tensor: Tensor1<f64> = unsafe {
            Tensor1::from_raw_vec_unchecked(vec![f64::NAN], Ix1(1))
        };
        let error = tensor.cast::<i32>()
            .expect_err("NaN→i32 cast should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                element_index: Some(0),
                ..
            }
        ));
    }

    /// Casting infinities to an integer type fails with `FloatToInteger`.
    #[test]
    fn test_cast_inf_to_int_returns_error() {
        let tensor: Tensor1<f64> = unsafe {
            Tensor1::from_raw_vec_unchecked(
                vec![f64::INFINITY, f64::NEG_INFINITY],
                Ix1(2),
            )
        };
        let error = tensor.cast::<i32>()
            .expect_err("inf→i32 cast should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                element_index: Some(0),
                ..
            }
        ));
    }

    /// `to_owned()` on a view clones logical elements into a fresh owned tensor.
    #[test]
    fn test_to_owned_from_view() {
        let tensor: Tensor1<i32> = unsafe {
            Tensor1::from_raw_vec_unchecked(vec![1, 2, 3], Ix1(3))
        };
        let view = tensor.view();
        let owned = view.to_owned();
        let result: Vec<i32> = owned.iter().copied().collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    /// `into_owned()` on an owned tensor returns its elements unchanged.
    #[test]
    fn test_into_owned_tensor() {
        let tensor: Tensor1<f64> = unsafe {
            Tensor1::from_raw_vec_unchecked(vec![4.0, 5.0], Ix1(2))
        };
        let owned = tensor.into_owned();
        let result: Vec<f64> = owned.iter().copied().collect();
        assert_eq!(result, vec![4.0, 5.0]);
    }

    /// `rewrap_cast_error` stamps the correct index on a later element failure,
    /// not just the first element.
    #[test]
    fn test_cast_reports_correct_index_for_later_element() {
        use crate::complex::Complex;
        let tensor: Tensor1<Complex<f64>> = unsafe {
            Tensor1::from_raw_vec_unchecked(
                vec![Complex::new(1.0, 0.0), Complex::new(2.0, 1.0)],
                Ix1(2),
            )
        };
        let error = tensor.cast::<f64>()
            .expect_err("non-zero imag at index 1 should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                element_index: Some(1),
                ..
            }
        ));
    }
}
