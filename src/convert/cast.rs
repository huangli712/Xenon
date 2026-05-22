//! Type conversion implementation.
//!
//! The public `CastTo` and `CastElement` traits are owned by `crate::element`;
//! this module only provides tensor-level conversion dispatch and the
//! crate-private `ConvertTo` shim.

use std::borrow::Cow;

use crate::complex::Complex;
use crate::dimension::Dimension;
use crate::element::{CastElement, CastTo, Element};
use crate::error::{ConversionFailureReason, Result, XenonError};
use crate::layout::{compute_f_strides, compute_layout_flags};
use crate::storage::{Owned, RawStorage, Storage, StorageIntoOwned};
use crate::tensor::{Tensor, TensorBase};

/// Crate-private sealed conversion dispatch trait.
///
/// Serves as the static dispatch entry point for the three-tier conversion
/// architecture (Tier-0 identity, Tier-1 lossless `From`, Tier-2/Tier-3
/// `CastTo`-based). Sealed via `CastElement: Element: Sealed`, preventing
/// external crates from extending the conversion matrix.
///
/// Tier-1 impls return `Ok(B::from(self))` directly without instantiating
/// `CastTo`; Tier-2/Tier-3 impls delegate to `<A as CastTo<B>>::cast_to(self)`.
pub(crate) trait ConvertTo<B>: CastElement
where
    B: CastElement,
{
    /// Converts `self` into `B`.
    ///
    /// Tier-1 (lossless) pairs always return `Ok`. Tier-2 (static lossy) and
    /// Tier-3 (dynamic) pairs may return `Err(XenonError::TypeConversion {..})`.
    fn convert(self) -> Result<B>;
}

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
pub(crate) unsafe fn from_shape_vec_aligned_unchecked<A, D>(shape: D, data: Vec<A>) -> Tensor<A, D>
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

/// Element-wise type conversion to target element type `B`.
///
/// Returns an owned `Tensor<B, D>` regardless of input storage mode.
///
/// # Type Parameters
///
/// * `B` - Target element type
///
/// # Errors
///
/// Returns `XenonError::TypeConversion` when any element cannot be converted
/// under the rules defined in `require.md §23`.
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: CastElement + Copy,
{
    /// Element-wise type conversion to target element type `B`.
    ///
    /// Returns an owned `Tensor<B, D>` regardless of input storage mode.
    ///
    /// # Type Parameters
    ///
    /// * `B` - Target element type, must implement `CastElement`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::TypeConversion` when any element cannot be
    /// converted under the rules defined in `require.md §23`.
    #[expect(
        private_bounds,
        reason = "ConvertTo is pub(crate) sealed; public cast() is gated by it"
    )]
    pub fn cast<B>(&self) -> Result<Tensor<B, D>>
    where
        B: CastElement,
        A: ConvertTo<B>,
    {
        let mut data: Vec<B> = Vec::with_capacity(self.len());
        for (index, value) in self.iter().copied().enumerate() {
            let converted = value
                .convert()
                .map_err(|error| rewrap_cast_error(error, index))?;
            data.push(converted);
        }
        // SAFETY: data.len() == self.len() == product(self.raw_dim()) because
        // the loop pushes exactly one element per F-order iteration over self.
        // self.raw_dim() was validated when self was constructed.
        Ok(unsafe { from_shape_vec_aligned_unchecked(self.raw_dim(), data) })
    }
}

/// Reconstruct a canonical F-order `Tensor<A, D>` from an `Owned<A>` storage
/// produced by `StorageIntoOwned::into_owned_storage()`.
///
/// Self-contained helper local to the convert module.
///
/// # Panics
///
/// Panics if `compute_f_strides` fails on `dim`. This cannot happen because
/// `dim` originates from a valid `TensorBase` whose shape was already validated.
#[inline]
pub(crate) fn into_owned_from_owned_storage<A, D>(owned_storage: Owned<A>, dim: D) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    let strides = compute_f_strides(&dim).expect("validated dim from TensorBase");
    let flags = compute_layout_flags(&dim, &strides, owned_storage.as_ptr());
    // SAFETY: `into_owned_storage()` guarantees `owned_storage.len() == product(dim)`;
    // `dim` originated from a valid `TensorBase`; canonical F-order strides
    // and flags are computed above.
    unsafe {
        TensorBase::new_unchecked(
            owned_storage,
            dim,
            strides,
            /* offset = */ 0,
            flags,
            /* derived_from_view_mut = */ false,
        )
    }
}

// ── to_owned() / into_owned() methods on TensorBase ──

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
        // SAFETY: data.len() == self.len() == product(self.raw_dim()); see §5.6.
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
    /// - `Tensor` (S=Owned<A>): returned directly, O(1), same data
    /// - `TensorView` / `TensorViewMut` / `ArcTensor`: O(n) allocate+copy
    ///   into canonical F-order
    pub fn into_owned(self) -> Tensor<A, D> {
        let owned_storage = self.storage.into_owned_storage();
        into_owned_from_owned_storage(owned_storage, self.shape)
    }
}

// ── Tier-0: Same-type identity (6 cells) ──

impl ConvertTo<i32> for i32 {
    #[inline]
    fn convert(self) -> Result<i32> {
        Ok(self)
    }
}

impl ConvertTo<i64> for i64 {
    #[inline]
    fn convert(self) -> Result<i64> {
        Ok(self)
    }
}

impl ConvertTo<f32> for f32 {
    #[inline]
    fn convert(self) -> Result<f32> {
        Ok(self)
    }
}

impl ConvertTo<f64> for f64 {
    #[inline]
    fn convert(self) -> Result<f64> {
        Ok(self)
    }
}

impl ConvertTo<Complex<f32>> for Complex<f32> {
    #[inline]
    fn convert(self) -> Result<Complex<f32>> {
        Ok(self)
    }
}

impl ConvertTo<Complex<f64>> for Complex<f64> {
    #[inline]
    fn convert(self) -> Result<Complex<f64>> {
        Ok(self)
    }
}

// ── Tier-1: std `From` arithmetic widening (3 cells) ──

impl ConvertTo<i64> for i32 {
    #[inline]
    fn convert(self) -> Result<i64> {
        Ok(i64::from(self))
    }
}

impl ConvertTo<f64> for f32 {
    #[inline]
    fn convert(self) -> Result<f64> {
        Ok(f64::from(self))
    }
}

impl ConvertTo<f64> for i32 {
    #[inline]
    fn convert(self) -> Result<f64> {
        Ok(f64::from(self))
    }
}

// ── Tier-1: real → complex zero-imaginary widening (4 cells) ──

impl ConvertTo<Complex<f32>> for f32 {
    #[inline]
    fn convert(self) -> Result<Complex<f32>> {
        Ok(Complex::new(self, 0.0))
    }
}

impl ConvertTo<Complex<f64>> for f64 {
    #[inline]
    fn convert(self) -> Result<Complex<f64>> {
        Ok(Complex::new(self, 0.0))
    }
}

impl ConvertTo<Complex<f64>> for f32 {
    #[inline]
    fn convert(self) -> Result<Complex<f64>> {
        Ok(Complex::new(f64::from(self), 0.0))
    }
}

impl ConvertTo<Complex<f64>> for i32 {
    #[inline]
    fn convert(self) -> Result<Complex<f64>> {
        Ok(Complex::new(f64::from(self), 0.0))
    }
}

// ── Tier-1: complex → complex widening (1 cell) ──

impl ConvertTo<Complex<f64>> for Complex<f32> {
    #[inline]
    fn convert(self) -> Result<Complex<f64>> {
        Ok(Complex::new(f64::from(self.re), f64::from(self.im)))
    }
}

// ── Tier-2: Lossy-by-default CastTo impls (14 cells) ──

impl CastTo<f32> for f64 {
    #[inline]
    fn cast_to(self) -> Result<f32> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <f32 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::LossyFloatNarrowing,
            element_index: None,
        })
    }
}

impl CastTo<i32> for f64 {
    #[inline]
    fn cast_to(self) -> Result<i32> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <i32 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        })
    }
}

impl CastTo<i64> for f64 {
    #[inline]
    fn cast_to(self) -> Result<i64> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        })
    }
}

impl CastTo<i32> for f32 {
    #[inline]
    fn cast_to(self) -> Result<i32> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f32 as Element>::ELEMENT_TYPE_NAME,
            target_type: <i32 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        })
    }
}

impl CastTo<i64> for f32 {
    #[inline]
    fn cast_to(self) -> Result<i64> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f32 as Element>::ELEMENT_TYPE_NAME,
            target_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        })
    }
}

impl CastTo<i32> for i64 {
    #[inline]
    fn cast_to(self) -> Result<i32> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <i32 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::LossyIntegerNarrowing,
            element_index: None,
        })
    }
}

impl CastTo<f32> for i64 {
    #[inline]
    fn cast_to(self) -> Result<f32> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <f32 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
            element_index: None,
        })
    }
}

impl CastTo<f64> for i64 {
    #[inline]
    fn cast_to(self) -> Result<f64> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <f64 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
            element_index: None,
        })
    }
}

impl CastTo<f32> for i32 {
    #[inline]
    fn cast_to(self) -> Result<f32> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i32 as Element>::ELEMENT_TYPE_NAME,
            target_type: <f32 as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
            element_index: None,
        })
    }
}

impl CastTo<Complex<f32>> for i32 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i32 as Element>::ELEMENT_TYPE_NAME,
            target_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
            element_index: None,
        })
    }
}

impl CastTo<Complex<f32>> for i64 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
            element_index: None,
        })
    }
}

impl CastTo<Complex<f64>> for i64 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
            element_index: None,
        })
    }
}

impl CastTo<Complex<f32>> for f64 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME,
            target_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::LossyFloatNarrowing,
            element_index: None,
        })
    }
}

impl CastTo<Complex<f32>> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
            target_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
            reason: ConversionFailureReason::LossyFloatNarrowing,
            element_index: None,
        })
    }
}

// ── Tier-2: ConvertTo forwarding impls (14 cells) ──

impl ConvertTo<f32> for f64 {
    #[inline]
    fn convert(self) -> Result<f32> {
        <f64 as CastTo<f32>>::cast_to(self)
    }
}

impl ConvertTo<i32> for f64 {
    #[inline]
    fn convert(self) -> Result<i32> {
        <f64 as CastTo<i32>>::cast_to(self)
    }
}

impl ConvertTo<i64> for f64 {
    #[inline]
    fn convert(self) -> Result<i64> {
        <f64 as CastTo<i64>>::cast_to(self)
    }
}

impl ConvertTo<i32> for f32 {
    #[inline]
    fn convert(self) -> Result<i32> {
        <f32 as CastTo<i32>>::cast_to(self)
    }
}

impl ConvertTo<i64> for f32 {
    #[inline]
    fn convert(self) -> Result<i64> {
        <f32 as CastTo<i64>>::cast_to(self)
    }
}

impl ConvertTo<i32> for i64 {
    #[inline]
    fn convert(self) -> Result<i32> {
        <i64 as CastTo<i32>>::cast_to(self)
    }
}

impl ConvertTo<f32> for i64 {
    #[inline]
    fn convert(self) -> Result<f32> {
        <i64 as CastTo<f32>>::cast_to(self)
    }
}

impl ConvertTo<f64> for i64 {
    #[inline]
    fn convert(self) -> Result<f64> {
        <i64 as CastTo<f64>>::cast_to(self)
    }
}

impl ConvertTo<f32> for i32 {
    #[inline]
    fn convert(self) -> Result<f32> {
        <i32 as CastTo<f32>>::cast_to(self)
    }
}

impl ConvertTo<Complex<f32>> for i32 {
    #[inline]
    fn convert(self) -> Result<Complex<f32>> {
        <i32 as CastTo<Complex<f32>>>::cast_to(self)
    }
}

impl ConvertTo<Complex<f32>> for i64 {
    #[inline]
    fn convert(self) -> Result<Complex<f32>> {
        <i64 as CastTo<Complex<f32>>>::cast_to(self)
    }
}

impl ConvertTo<Complex<f64>> for i64 {
    #[inline]
    fn convert(self) -> Result<Complex<f64>> {
        <i64 as CastTo<Complex<f64>>>::cast_to(self)
    }
}

impl ConvertTo<Complex<f32>> for f64 {
    #[inline]
    fn convert(self) -> Result<Complex<f32>> {
        <f64 as CastTo<Complex<f32>>>::cast_to(self)
    }
}

impl ConvertTo<Complex<f32>> for Complex<f64> {
    #[inline]
    fn convert(self) -> Result<Complex<f32>> {
        <Complex<f64> as CastTo<Complex<f32>>>::cast_to(self)
    }
}

// ── Tier-3: Dynamic CastTo impls (8 cells) ──

// Group A: 同精度，直接返回实部 (cells #1, #2)
impl CastTo<f32> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<f32> {
        if self.im == 0.0 {
            Ok(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
                target_type: <f32 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

impl CastTo<f64> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<f64> {
        if self.im == 0.0 {
            Ok(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
                target_type: <f64 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

// Group B: 内层 Tier-1 std From widening (cell #3 only)
// Complex<f32> → f64: im == 0 → Ok(f64::from(self.re))
impl CastTo<f64> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<f64> {
        if self.im == 0.0 {
            Ok(f64::from(self.re))
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
                target_type: <f64 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

// Group C: 内层 Tier-2 静态有损 (cells #4, #5, #6, #7, #8)
impl CastTo<f32> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<f32> {
        if self.im == 0.0 {
            <f64 as CastTo<f32>>::cast_to(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
                target_type: <f32 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

impl CastTo<i32> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<i32> {
        if self.im == 0.0 {
            <f32 as CastTo<i32>>::cast_to(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
                target_type: <i32 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

impl CastTo<i64> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<i64> {
        if self.im == 0.0 {
            <f32 as CastTo<i64>>::cast_to(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
                target_type: <i64 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

impl CastTo<i32> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<i32> {
        if self.im == 0.0 {
            <f64 as CastTo<i32>>::cast_to(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
                target_type: <i32 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

impl CastTo<i64> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<i64> {
        if self.im == 0.0 {
            <f64 as CastTo<i64>>::cast_to(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
                target_type: <i64 as Element>::ELEMENT_TYPE_NAME,
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

// ── Tier-3: ConvertTo forwarding impls (8 cells) ──

impl ConvertTo<f32> for Complex<f32> {
    #[inline]
    fn convert(self) -> Result<f32> {
        <Complex<f32> as CastTo<f32>>::cast_to(self)
    }
}

impl ConvertTo<f64> for Complex<f64> {
    #[inline]
    fn convert(self) -> Result<f64> {
        <Complex<f64> as CastTo<f64>>::cast_to(self)
    }
}

impl ConvertTo<f64> for Complex<f32> {
    #[inline]
    fn convert(self) -> Result<f64> {
        <Complex<f32> as CastTo<f64>>::cast_to(self)
    }
}

impl ConvertTo<f32> for Complex<f64> {
    #[inline]
    fn convert(self) -> Result<f32> {
        <Complex<f64> as CastTo<f32>>::cast_to(self)
    }
}

impl ConvertTo<i32> for Complex<f32> {
    #[inline]
    fn convert(self) -> Result<i32> {
        <Complex<f32> as CastTo<i32>>::cast_to(self)
    }
}

impl ConvertTo<i64> for Complex<f32> {
    #[inline]
    fn convert(self) -> Result<i64> {
        <Complex<f32> as CastTo<i64>>::cast_to(self)
    }
}

impl ConvertTo<i32> for Complex<f64> {
    #[inline]
    fn convert(self) -> Result<i32> {
        <Complex<f64> as CastTo<i32>>::cast_to(self)
    }
}

impl ConvertTo<i64> for Complex<f64> {
    #[inline]
    fn convert(self) -> Result<i64> {
        <Complex<f64> as CastTo<i64>>::cast_to(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor1;

    #[test]
    fn test_convert_to_trait_signature_accepts_cast_elements() {
        fn require_cast_element<A: CastElement>() {}
        require_cast_element::<i32>();
        require_cast_element::<f64>();
    }

    #[test]
    fn test_cast_f32_to_f64() {
        assert_eq!(
            <f32 as ConvertTo<f64>>::convert(1.5).expect("f32→f64 is tier-1 lossless"),
            1.5_f64
        );
    }

    #[test]
    fn test_cast_real_to_complex() {
        let value = <i32 as ConvertTo<Complex<f64>>>::convert(7)
            .expect("i32→Complex<f64> is tier-1 lossless");
        assert_eq!(value, Complex::new(7.0, 0.0));
    }

    #[test]
    fn test_cast_f64_to_f32_returns_error() {
        assert!(matches!(
            <f64 as ConvertTo<f32>>::convert(1.0),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::LossyFloatNarrowing,
                ..
            })
        ));
    }

    #[test]
    fn test_cast_int_narrowing_returns_error() {
        assert!(matches!(
            <i64 as ConvertTo<i32>>::convert(1),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::LossyIntegerNarrowing,
                ..
            })
        ));
    }

    #[test]
    fn test_cast_float_to_int_returns_error() {
        assert!(matches!(
            <f64 as ConvertTo<i32>>::convert(1.0),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                ..
            })
        ));
        assert!(matches!(
            <f32 as ConvertTo<i64>>::convert(1.0),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                ..
            })
        ));
    }

    #[test]
    fn test_cast_complex_to_real_requires_zero_imag() {
        let ok = Complex::new(3.0_f64, 0.0);
        assert_eq!(
            <Complex<f64> as ConvertTo<f64>>::convert(ok)
                .expect("Complex<f64>→f64 with im=0 should succeed"),
            3.0
        );

        let err = Complex::new(3.0_f64, 1.0);
        assert!(matches!(
            <Complex<f64> as ConvertTo<f64>>::convert(err),
            Err(XenonError::TypeConversion { .. })
        ));
    }

    #[test]
    fn test_cast_complex_to_int_requires_zero_imag_and_inner_success() {
        // im != 0 => NonZeroImaginaryPart (precondition fails)
        let err_im = Complex::new(1.0_f64, 2.0);
        assert!(matches!(
            <Complex<f64> as ConvertTo<i32>>::convert(err_im),
            Err(XenonError::TypeConversion { .. })
        ));

        // im == 0 but inner f64 -> i32 is lossy-by-default (Tier-2) => still Err.
        // Verifies §5.4: zero-imag is necessary but NOT sufficient.
        let err_inner = Complex::new(1.0_f64, 0.0);
        assert!(matches!(
            <Complex<f64> as ConvertTo<i32>>::convert(err_inner),
            Err(XenonError::TypeConversion { .. })
        ));
    }

    #[test]
    fn test_cast_i32_to_f64() {
        let tensor: Tensor1<i32> =
            unsafe { Tensor1::from_raw_vec_unchecked(vec![1_i32, 2, 3], crate::dimension::Ix1(3)) };
        let converted: Tensor1<f64> = tensor.cast().expect("i32→f64 cast should succeed");
        let result: Vec<f64> = converted.iter().copied().collect();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cast_reports_element_index() {
        let tensor: Tensor1<f64> = unsafe {
            Tensor1::from_raw_vec_unchecked(vec![1.0_f64, 2.0], crate::dimension::Ix1(2))
        };
        let error = tensor.cast::<f32>().expect_err("f64→f32 cast should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                element_index: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn test_cast_nan_to_int_returns_error() {
        let tensor: Tensor1<f64> =
            unsafe { Tensor1::from_raw_vec_unchecked(vec![f64::NAN], crate::dimension::Ix1(1)) };
        let error = tensor.cast::<i32>().expect_err("NaN→i32 cast should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                element_index: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn test_cast_inf_to_int_returns_error() {
        let tensor: Tensor1<f64> = unsafe {
            Tensor1::from_raw_vec_unchecked(
                vec![f64::INFINITY, f64::NEG_INFINITY],
                crate::dimension::Ix1(2),
            )
        };
        let error = tensor.cast::<i32>().expect_err("inf→i32 cast should fail");
        assert!(matches!(
            error,
            XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                element_index: Some(0),
                ..
            }
        ));
    }

    #[test]
    fn test_to_owned_from_view() {
        let tensor: Tensor1<i32> =
            unsafe { Tensor1::from_raw_vec_unchecked(vec![1, 2, 3], crate::dimension::Ix1(3)) };
        let view = tensor.view();
        let owned = view.to_owned();
        let result: Vec<i32> = owned.iter().copied().collect();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_into_owned_tensor() {
        let tensor: Tensor1<f64> =
            unsafe { Tensor1::from_raw_vec_unchecked(vec![4.0, 5.0], crate::dimension::Ix1(2)) };
        let owned = tensor.into_owned();
        let result: Vec<f64> = owned.iter().copied().collect();
        assert_eq!(result, vec![4.0, 5.0]);
    }
}
