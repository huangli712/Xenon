//! Per-pair type conversion dispatch for the 6×6 element matrix.
//!
//! The 36 conversion pairs among the 6 numeric types (`i32`, `i64`,
//! `f32`, `f64`, `Complex<f32>`, `Complex<f64>`) are covered by a
//! single trait — `CastTo` — organized in four tiers:
//!
//! * **Tier-0** (6 cells): same-type identity, always `Ok(self)`.
//! * **Tier-1** (8 cells): lossless widening via `std::From` or
//!   zero-imaginary complex construction, always `Ok`.
//! * **Tier-2** (14 cells): lossy-by-default, always returns a typed
//!   `Err(XenonError::TypeConversion {..})` such as
//!   `LossyFloatNarrowing` or `FloatToInteger`.
//! * **Tier-3** (8 cells): dynamic — extraction from `Complex` that
//!   succeeds only when `im == 0`, then delegates to the inner
//!   real conversion.
//!
//! `CastTo` is `pub(crate)`.  Public compile-time gating is provided
//! by `CastElement` (`super::types`), a sealed marker that excludes
//! `bool` and other non-numeric types.

use std::borrow::Cow;

use crate::complex::Complex;
use crate::element::Element;
use crate::error::{ConversionFailureReason, Result, XenonError};

use super::CastElement;

/// Crate-private sealed conversion dispatch trait.
///
/// Serves as the static dispatch entry point for the three-tier conversion
/// architecture (Tier-0 identity, Tier-1 lossless `From`, Tier-2 static lossy,
/// Tier-3 dynamic). Sealed via `CastElement: Element: Sealed`, preventing
/// external crates from extending the conversion matrix.
///
/// Tier-0/Tier-1 impls return `Ok(..)` directly; Tier-2 impls return a typed
/// `Err(XenonError::TypeConversion {..})`; Tier-3 impls perform a dynamic
/// `im == 0` check before delegating to the inner real conversion.
pub(crate) trait CastTo<B>: CastElement
where
    B: CastElement,
{
    /// Converts `self` into `B`.
    ///
    /// Tier-1 (lossless) pairs always return `Ok`. Tier-2 (static lossy) and
    /// Tier-3 (dynamic) pairs may return `Err(XenonError::TypeConversion {..})`.
    fn cast_to(self) -> Result<B>;
}

// --- Tier-0: Same-type identity (6 cells) -----------------------------------

impl CastTo<i32> for i32 {
    #[inline]
    fn cast_to(self) -> Result<i32> {
        Ok(self)
    }
}

impl CastTo<i64> for i64 {
    #[inline]
    fn cast_to(self) -> Result<i64> {
        Ok(self)
    }
}

impl CastTo<f32> for f32 {
    #[inline]
    fn cast_to(self) -> Result<f32> {
        Ok(self)
    }
}

impl CastTo<f64> for f64 {
    #[inline]
    fn cast_to(self) -> Result<f64> {
        Ok(self)
    }
}

impl CastTo<Complex<f32>> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>> {
        Ok(self)
    }
}

impl CastTo<Complex<f64>> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>> {
        Ok(self)
    }
}

// --- Tier-1: std `From` arithmetic widening (3 cells) -----------------------

impl CastTo<i64> for i32 {
    #[inline]
    fn cast_to(self) -> Result<i64> {
        Ok(i64::from(self))
    }
}

impl CastTo<f64> for f32 {
    #[inline]
    fn cast_to(self) -> Result<f64> {
        Ok(f64::from(self))
    }
}

impl CastTo<f64> for i32 {
    #[inline]
    fn cast_to(self) -> Result<f64> {
        Ok(f64::from(self))
    }
}

// --- Tier-1: real → complex zero-imaginary widening (4 cells) ---------------

impl CastTo<Complex<f32>> for f32 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>> {
        Ok(Complex::new(self, 0.0))
    }
}

impl CastTo<Complex<f64>> for f64 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>> {
        Ok(Complex::new(self, 0.0))
    }
}

impl CastTo<Complex<f64>> for f32 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>> {
        Ok(Complex::new(f64::from(self), 0.0))
    }
}

impl CastTo<Complex<f64>> for i32 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>> {
        Ok(Complex::new(f64::from(self), 0.0))
    }
}

// --- Tier-1: complex → complex widening (1 cell) ----------------------------

impl CastTo<Complex<f64>> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>> {
        Ok(Complex::new(f64::from(self.re), f64::from(self.im)))
    }
}

// --- Tier-2: Lossy-by-default conversions (14 cells) ------------------------

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

// --- Tier-3: Dynamic conversions (8 cells) ----------------------------------

// Group A: same-precision real extraction (cells #1, #2)
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

// Group B: inner Tier-1 std::From widening (cell #3 only)
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

// Group C: inner Tier-2 static lossy (cells #4, #5, #6, #7, #8)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tier-1 lossless `f32` → `f64` widening succeeds and preserves the value.
    #[test]
    fn test_cast_f32_to_f64() {
        assert_eq!(
            <f32 as CastTo<f64>>::cast_to(1.5)
                .expect("f32→f64 is tier-1 lossless"),
            1.5_f64
        );
    }

    /// Tier-1 `i32` → `Complex<f64>` widening yields a zero-imaginary complex.
    #[test]
    fn test_cast_real_to_complex() {
        let value = <i32 as CastTo<Complex<f64>>>::cast_to(7)
            .expect("i32→Complex<f64> is tier-1 lossless");
        assert_eq!(value, Complex::new(7.0, 0.0));
    }

    /// Tier-2 `f64` → `f32` narrowing is lossy-by-default and reports
    /// `LossyFloatNarrowing`.
    #[test]
    fn test_cast_f64_to_f32_returns_error() {
        assert!(matches!(
            <f64 as CastTo<f32>>::cast_to(1.0),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::LossyFloatNarrowing,
                ..
            })
        ));
    }

    /// Tier-2 `i64` → `i32` narrowing reports `LossyIntegerNarrowing`.
    #[test]
    fn test_cast_int_narrowing_returns_error() {
        assert!(matches!(
            <i64 as CastTo<i32>>::cast_to(1),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::LossyIntegerNarrowing,
                ..
            })
        ));
    }

    /// Tier-2 float → integer conversions report `FloatToInteger`.
    #[test]
    fn test_cast_float_to_int_returns_error() {
        assert!(matches!(
            <f64 as CastTo<i32>>::cast_to(1.0),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                ..
            })
        ));
        assert!(matches!(
            <f32 as CastTo<i64>>::cast_to(1.0),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::FloatToInteger,
                ..
            })
        ));
    }

    /// Tier-2 integer → float conversions report `IntegerToFloatPrecisionLoss`.
    #[test]
    fn test_cast_int_to_float_precision_loss() {
        assert!(matches!(
            <i32 as CastTo<f32>>::cast_to(1),
            Err(XenonError::TypeConversion {
                reason: ConversionFailureReason::IntegerToFloatPrecisionLoss,
                ..
            })
        ));
    }

    /// Tier-3 `Complex<f64>` → `f64` succeeds only when the imaginary part
    /// is zero, otherwise it reports a conversion error.
    #[test]
    fn test_cast_complex_to_real_requires_zero_imag() {
        let ok = Complex::new(3.0_f64, 0.0);
        assert_eq!(
            <Complex<f64> as CastTo<f64>>::cast_to(ok)
                .expect("Complex<f64>→f64 with im=0 should succeed"),
            3.0
        );

        let err = Complex::new(3.0_f64, 1.0);
        assert!(matches!(
            <Complex<f64> as CastTo<f64>>::cast_to(err),
            Err(XenonError::TypeConversion { .. })
        ));
    }

    /// Tier-3 `Complex<f64>` → `i32` requires both a zero imaginary part and
    /// a successful inner real conversion; zero-imag alone is not sufficient.
    #[test]
    fn test_cast_complex_to_int_requires_zero_imag_and_inner_success() {
        // im != 0 => NonZeroImaginaryPart (precondition fails)
        let err_im = Complex::new(1.0_f64, 2.0);
        assert!(matches!(
            <Complex<f64> as CastTo<i32>>::cast_to(err_im),
            Err(XenonError::TypeConversion { .. })
        ));

        // im == 0 but inner f64 -> i32 is lossy-by-default (Tier-2) => still Err.
        // Zero-imag is necessary but NOT sufficient.
        let err_inner = Complex::new(1.0_f64, 0.0);
        assert!(matches!(
            <Complex<f64> as CastTo<i32>>::cast_to(err_inner),
            Err(XenonError::TypeConversion { .. })
        ));
    }

    /// Tier-3 `Complex<f32>` → `f64` succeeds when `im == 0` via inner
    /// Tier-1 `std::From` widening.
    #[test]
    fn test_cast_complex_f32_to_f64_zero_imag_succeeds() {
        let ok = Complex::new(1.0_f32, 0.0);
        assert_eq!(
            <Complex<f32> as CastTo<f64>>::cast_to(ok)
                .expect("Complex<f32>→f64 with im=0 should succeed"),
            1.0_f64
        );

        let err = Complex::new(1.0_f32, 0.5);
        assert!(matches!(
            <Complex<f32> as CastTo<f64>>::cast_to(err),
            Err(XenonError::TypeConversion { .. })
        ));
    }
}
