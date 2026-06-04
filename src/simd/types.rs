//! SIMD type definitions: SimdElement trait, operation enums.

use crate::complex::Complex;
use crate::private::Sealed;

// ---------------------------------------------------------------------------
// SimdElement — sealed marker trait
// ---------------------------------------------------------------------------

/// Sealed marker trait for types that support SIMD lane operations.
///
/// Implemented for 6 concrete types:
/// `f32`, `f64`, `i32`, `i64`, `Complex<f32>`, `Complex<f64>`.
///
/// `Sealed` prevents downstream crates from adding new implementations.
/// Use `core::mem::size_of::<A>()` / `core::mem::align_of::<A>()` for
/// per-type size/alignment metadata — the compiler exposes the same values
/// without requiring trait-level redeclaration.
pub(crate) trait SimdElement: Sealed + Copy + Clone + Send + Sync + 'static {}

impl SimdElement for f32 {}
impl SimdElement for f64 {}
impl SimdElement for i32 {}
impl SimdElement for i64 {}
impl SimdElement for Complex<f32> {}
impl SimdElement for Complex<f64> {}

// ---------------------------------------------------------------------------
// Operation enums
// ---------------------------------------------------------------------------

/// Binary element-wise operation selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum BinaryOp {
    /// Element-wise addition.
    Add,

    /// Element-wise subtraction.
    Sub,

    /// Element-wise multiplication.
    Mul,
    
    /// Element-wise division.
    Div,
}

/// Unary element-wise operation selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum UnaryOp {
    /// Element-wise negation.
    Neg,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::simd::simd_vector_width;
    use super::*;

    /// Verifies that the capability query returns `None` for every
    /// supported element type (ISA lane widths not yet wired).
    #[test]
    fn test_simd_vector_width_skeleton_returns_none() {
        assert_eq!(simd_vector_width::<f32>(), None);
        assert_eq!(simd_vector_width::<f64>(), None);
        assert_eq!(simd_vector_width::<i32>(), None);
        assert_eq!(simd_vector_width::<i64>(), None);
        assert_eq!(simd_vector_width::<Complex<f32>>(), None);
        assert_eq!(simd_vector_width::<Complex<f64>>(), None);
    }
}
