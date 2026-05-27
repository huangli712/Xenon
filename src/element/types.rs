//! Element type hierarchy: base traits and type discriminants.
//!
//! # Supported element types
//!
//! The closed set of element types consists of 7 members:
//!
//! | Type | `Element` | `Numeric` | `RealScalar` | `ComplexScalar` |
//! |------|-----------|-----------|--------------|-----------------|
//! | `i32` | ✓ | ✓ | | |
//! | `i64` | ✓ | ✓ | | |
//! | `f32` | ✓ | ✓ | ✓ | |
//! | `f64` | ✓ | ✓ | ✓ | |
//! | `Complex<f32>` | ✓ | ✓ | | ✓ |
//! | `Complex<f64>` | ✓ | ✓ | | ✓ |
//! | `bool` | ✓ | | | |
//!
//! # `usize` is NOT an element type
//!
//! `usize` is used for indexing, shape metadata, and dimension
//! expressions only. It does not implement `Element` because:
//!
//! * It lacks an additive inverse (no negative values), which prevents
//!   it from forming the algebraic structure required by `Element`.
//! * `Element` types require `zero()` and `one()` identities; `usize`
//!   has no consistent negation semantics in this context.
//!
//! Concrete impls for primitive types are in `primitives.rs`.
//!
//! # `CastTo<T>` error semantics
//!
//! `CastTo<T>::cast_to()` returns `Err(XenonError::TypeConversion)` for
//! lossy conversions. Lossy cases include:
//!
//! * Float → integer with truncation (e.g., `1.5f64.cast_to::<i32>()`)
//! * Overflow (e.g., `i64::MAX.cast_to::<i32>()`)
//! * NaN → integer (no finite representation)
//! * Complex → real when imaginary part is non-zero
//!
//!
//! `bool` is excluded from `CastTo<T>` as both source and target.
//! This exclusion is enforced at compile time via the absence of
//! `impl CastElement for bool`.

use crate::error::XenonError;
use crate::private::Sealed;

use crate::complex::Complex;

/// Base trait for all tensor element types.
///
/// `Element` provides identity values (`zero`/`one`), a compile-time type
/// discriminant (`ELEMENT_TYPE`), and a canonical name (`ELEMENT_TYPE_NAME`)
/// for use in error messages and FFI mapping.
///
/// # Sealed
///
/// Only types within Xenon's closed element set may implement `Element`.
pub trait Element:
    Copy + Clone + PartialEq + core::fmt::Debug + core::fmt::Display + Send + Sync + Sealed
{
    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Compile-time discriminant for this element type.
    const ELEMENT_TYPE: ElementType;

    /// Canonical, stable name for this element type.
    const ELEMENT_TYPE_NAME: &'static str;
}

/// Compile-time enumerated discriminant for every supported element type.
///
/// Each variant carries an explicit `#[repr(u8)]` value for FFI use.
/// The enum is `#[non_exhaustive]` so that downstream code must handle
/// future additions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum ElementType {
    /// Boolean: `true` or `false`.
    Bool = 0,
    /// 32-bit signed integer.
    I32 = 1,
    /// 64-bit signed integer.
    I64 = 2,
    /// 32-bit IEEE-754 floating-point.
    F32 = 3,
    /// 64-bit IEEE-754 floating-point.
    F64 = 4,
    /// Single-precision complex number: `Complex<f32>`.
    Complex32 = 5,
    /// Double-precision complex number: `Complex<f64>`.
    Complex64 = 6,
}

impl ElementType {
    /// Canonical, human-readable name for each variant.
    ///
    /// These strings match `Element::ELEMENT_TYPE_NAME` for the
    /// corresponding concrete types.
    pub const fn name(self) -> &'static str {
        match self {
            ElementType::Bool => "bool",
            ElementType::I32 => "i32",
            ElementType::I64 => "i64",
            ElementType::F32 => "f32",
            ElementType::F64 => "f64",
            ElementType::Complex32 => "Complex<f32>",
            ElementType::Complex64 => "Complex<f64>",
        }
    }

    /// Returns the discriminant for the element type `A`.
    pub const fn of<A: Element>() -> Self {
        A::ELEMENT_TYPE
    }
}

impl core::fmt::Display for ElementType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Returns the `ElementType` discriminant for `A`.
pub const fn element_type_of<A: Element>() -> ElementType {
    A::ELEMENT_TYPE
}

/// Returns the canonical name for `A`.
pub const fn element_type_name_of<A: Element>() -> &'static str {
    A::ELEMENT_TYPE_NAME
}

/// Marker trait for element types that support ordered comparison.
///
/// Only `i32`, `i64`, `f32`, `f64` implement this trait.
pub trait OrderedCompareElement: Element + PartialOrd + Sealed {}

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
    fn cast_to(self) -> Result<T, XenonError>;
}

/// Internal marker for the bool element type.
///
/// Constrains operations to bool tensors only. Not part of the public API;
/// sealed via `crate::private::Sealed`.
#[allow(dead_code)]
pub(crate) trait BoolElement: Element + Sealed {}

/// Public sealed marker for element types in the cast matrix.
///
/// `cast()` public signature `where A: CastElement, T: CastElement` uses this
/// trait to exclude `bool` from conversion at compile time and narrow the
/// element set to the 6 numeric types.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait CastElement: Element {}

impl OrderedCompareElement for i32 {}
impl OrderedCompareElement for i64 {}
impl OrderedCompareElement for f32 {}
impl OrderedCompareElement for f64 {}

impl CastElement for i32 {}
impl CastElement for i64 {}
impl CastElement for f32 {}
impl CastElement for f64 {}
impl CastElement for Complex<f32> {}
impl CastElement for Complex<f64> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

    /// Verifies that the `Element` API surface (`zero`, `one`, `ELEMENT_TYPE`,
    /// `ELEMENT_TYPE_NAME`) is well-formed and reachable through a generic
    /// `A: Element` bound. Exercises all 7 closed element types.
    #[test]
    fn test_element_contract() {
        fn check<A: Element>() {
            let _ = A::zero();
            let _ = A::one();
            let _ = A::ELEMENT_TYPE;
            let _ = A::ELEMENT_TYPE_NAME;
        }
        check::<i32>();
        check::<i64>();
        check::<f32>();
        check::<f64>();
        check::<bool>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
    }

    /// Verifies ElementType::name() returns the correct string for each
    /// variant.
    #[test]
    fn test_element_type_name_round_trip() {
        assert_eq!(ElementType::Bool.name(), "bool");
        assert_eq!(ElementType::I32.name(), "i32");
        assert_eq!(ElementType::I64.name(), "i64");
        assert_eq!(ElementType::F32.name(), "f32");
        assert_eq!(ElementType::F64.name(), "f64");
        assert_eq!(ElementType::Complex32.name(), "Complex<f32>");
        assert_eq!(ElementType::Complex64.name(), "Complex<f64>");
    }

    /// Compile-time verification: `usize` must NOT implement `Element`.
    /// This test will FAIL TO COMPILE if someone accidentally adds
    /// an `Element` impl for `usize`.
    #[test]
    fn test_usize_does_not_implement_element() {
        fn _assert_element<T: Element>() {
            let _ = T::zero();
        }
        _assert_element::<i32>();
        _assert_element::<i64>();
        _assert_element::<f64>();
        // Negative bound: uncommenting the line below MUST cause a compile error.
        // _assert_element::<usize>();
    }

    /// Verifies OrderedCompareElement, CastElement, and BoolElement trait
    /// bounds for concrete types.
    #[test]
    fn test_marker_trait_impls() {
        fn assert_ordered<T: OrderedCompareElement>() {}
        assert_ordered::<i32>();
        assert_ordered::<i64>();
        assert_ordered::<f32>();
        assert_ordered::<f64>();

        fn assert_castable<T: CastElement>() {}
        assert_castable::<i32>();
        assert_castable::<i64>();
        assert_castable::<f32>();
        assert_castable::<f64>();
        assert_castable::<Complex<f32>>();
        assert_castable::<Complex<f64>>();

        fn assert_bool<T: BoolElement>() {}
        assert_bool::<bool>();
    }

    #[test]
    fn test_element_type_discriminants() {
        assert_eq!(ElementType::Bool as u8, 0);
        assert_eq!(ElementType::I32 as u8, 1);
        assert_eq!(ElementType::I64 as u8, 2);
        assert_eq!(ElementType::F32 as u8, 3);
        assert_eq!(ElementType::F64 as u8, 4);
        assert_eq!(ElementType::Complex32 as u8, 5);
        assert_eq!(ElementType::Complex64 as u8, 6);
    }

    /// Verifies ElementType::of::&lt;A&gt;() resolves to the correct variant.
    #[test]
    fn test_element_type_of_dispatch() {
        assert_eq!(ElementType::of::<i32>(), ElementType::I32);
        assert_eq!(ElementType::of::<f64>(), ElementType::F64);
        assert_eq!(ElementType::of::<bool>(), ElementType::Bool);
        assert_eq!(ElementType::of::<Complex<f64>>(), ElementType::Complex64);
    }

    /// Verifies element_type_of() and element_type_name_of() free
    /// functions.
    #[test]
    fn test_free_functions_dispatch() {
        assert_eq!(element_type_of::<f32>(), ElementType::F32);
        assert_eq!(element_type_name_of::<i64>(), "i64");
        assert_eq!(element_type_name_of::<Complex<f32>>(), "Complex<f32>");
    }

    /// Verifies Display impl for ElementType.
    #[test]
    fn test_element_type_display() {
        assert_eq!(format!("{}", ElementType::Bool), "bool");
        assert_eq!(format!("{}", ElementType::I32), "i32");
        assert_eq!(format!("{}", ElementType::F64), "f64");
        assert_eq!(format!("{}", ElementType::Complex64), "Complex<f64>");
    }

    /// Verifies Element::ELEMENT_TYPE_NAME matches ElementType::name()
    /// for each type.
    #[test]
    fn test_element_type_name_consistency() {
        assert_eq!(<i32 as Element>::ELEMENT_TYPE_NAME, ElementType::I32.name());
        assert_eq!(<i64 as Element>::ELEMENT_TYPE_NAME, ElementType::I64.name());
        assert_eq!(<f32 as Element>::ELEMENT_TYPE_NAME, ElementType::F32.name());
        assert_eq!(<f64 as Element>::ELEMENT_TYPE_NAME, ElementType::F64.name());
        assert_eq!(
            <bool as Element>::ELEMENT_TYPE_NAME,
            ElementType::Bool.name()
        );
        assert_eq!(
            <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
            ElementType::Complex32.name(),
        );
        assert_eq!(
            <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
            ElementType::Complex64.name(),
        );
    }
}
