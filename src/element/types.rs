//! Element type discriminant and free lookup functions.

use core::fmt::{Display, Formatter};
use super::Element;

/// Compile-time enumerated discriminant for every supported element type.
///
/// Each variant carries an explicit `#[repr(u8)]` value for FFI use,
/// providing a stable ABI mapping from Rust element types to C-compatible
/// integer tags. The enum is `#[non_exhaustive]` so that downstream code
/// must handle future additions (e.g., new numeric types or complex
/// precision levels).
///
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
    /// Returns the canonical, human-readable name for this variant.
    ///
    /// Each name matches the corresponding `Element::ELEMENT_TYPE_NAME`
    /// string for the concrete type.
    pub(crate) const fn name(self) -> &'static str {
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

    /// Returns the compile-time discriminant for element type `A`.
    ///
    /// This is a zero-cost generic dispatch: `ElementType::of::<A>()`
    /// resolves to the `A::ELEMENT_TYPE` associated constant and is
    /// eligible for const evaluation.
    #[allow(dead_code)]
    pub(crate) const fn of<A: Element>() -> Self {
        A::ELEMENT_TYPE
    }
}

impl Display for ElementType {
    /// Formats `ElementType` using its [`name`](ElementType::name).
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Returns the `ElementType` discriminant for `A`.
///
/// This is a free-function equivalent of [`ElementType::of`].
/// Both resolve to `A::ELEMENT_TYPE` and are zero-cost const functions.
pub(crate) const fn element_type_of<A: Element>() -> ElementType {
    A::ELEMENT_TYPE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

    /// Verifies `ElementType::name()` returns the correct string for
    /// every variant, covering all 7 closed element types.
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

    /// Verifies `#[repr(u8)]` discriminant values match the design spec:
    /// `Bool=0` through `Complex64=6`.
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

    /// Verifies `ElementType::of::<A>()` resolves to the correct variant
    /// through the `A::ELEMENT_TYPE` associated constant.
    #[test]
    fn test_element_type_of_dispatch() {
        assert_eq!(ElementType::of::<i32>(), ElementType::I32);
        assert_eq!(ElementType::of::<f64>(), ElementType::F64);
        assert_eq!(ElementType::of::<bool>(), ElementType::Bool);
        assert_eq!(ElementType::of::<Complex<f64>>(), ElementType::Complex64);
    }

    /// Verifies the free function `element_type_of::<A>()` returns the
    /// same result as `ElementType::of::<A>()`.
    #[test]
    fn test_free_functions_dispatch() {
        assert_eq!(element_type_of::<f32>(), ElementType::F32);
    }

    /// Verifies the `Display` impl delegates to [`ElementType::name`].
    #[test]
    fn test_element_type_display() {
        assert_eq!(format!("{}", ElementType::Bool), "bool");
        assert_eq!(format!("{}", ElementType::I32), "i32");
        assert_eq!(format!("{}", ElementType::F64), "f64");
        assert_eq!(format!("{}", ElementType::Complex64), "Complex<f64>");
    }
}
