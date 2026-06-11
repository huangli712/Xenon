//! Element-wise operation selectors.
//!
//! Both `BinaryOp` and `UnaryOp` are feature-independent: the ungated
//! dispatch layer in `binary.rs` / `unary.rs` names them to select an
//! element-wise operation. When `simd` is enabled they also tag the SIMD
//! kernels; without `simd` the dispatch layer simply ignores them.

// ----------------------------------------------------------------------------
// Operation enums
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(all(test, feature = "simd"))]
mod tests {
    use crate::complex::Complex;
    use crate::simd::simd_vector_width;

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
