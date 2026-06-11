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


