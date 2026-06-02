//! Type conversion traits and tier-based impls.
//!
//! Defines the public `CastTo` trait, the crate-private `ConvertTo` dispatch
//! shim, and all Tier-0/1/2/3 conversion impls for the 6×6 element matrix.
//!
//! Tensor-level `cast()`, `to_owned()`, and `into_owned()` methods
//! on `TensorBase` are in `super::impls`.
//! `CastElement` is defined in `super::types`.

use std::borrow::Cow;

use crate::complex::Complex;
use crate::convert::CastElement;
use crate::element::Element;
use crate::error::{ConversionFailureReason, Result, XenonError};

/// Type conversion trait for element types.
///
/// Defines fallible conversion between element types. Lossy conversions
/// (e.g., `f64` → `i32` truncation, overflow) return
/// `Err(XenonError::TypeConversion)`.
///
/// # Sealed
///
/// Only Xenon's closed element set may implement this trait.
pub trait CastTo<T: Element>: Element {
    /// Attempts to convert `self` to type `T`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::TypeConversion` if the conversion is lossy or
    /// the value cannot be represented in the target type.
    fn cast_to(self) -> std::result::Result<T, XenonError>;
}

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


}
