//! Element type discriminants, free functions, and marker traits.
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

use core::fmt::{Display, Formatter};

use crate::element::primitives::Element;

// ── ElementType ───────────────────────────────────────────────────────────

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

impl Display for ElementType {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Returns the `ElementType` discriminant for `A`.
pub const fn element_type_of<A: Element>() -> ElementType {
    A::ELEMENT_TYPE
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

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

    /// Verifies element_type_of() free function.
    #[test]
    fn test_free_functions_dispatch() {
        assert_eq!(element_type_of::<f32>(), ElementType::F32);
    }

    /// Verifies Display impl for ElementType.
    #[test]
    fn test_element_type_display() {
        assert_eq!(format!("{}", ElementType::Bool), "bool");
        assert_eq!(format!("{}", ElementType::I32), "i32");
        assert_eq!(format!("{}", ElementType::F64), "f64");
        assert_eq!(format!("{}", ElementType::Complex64), "Complex<f64>");
    }
}
