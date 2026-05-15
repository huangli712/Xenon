//! Xenon prelude module.
//!
//! Provides a convenient way to import the most commonly used types and traits
//! from the Xenon crate.
//!
//! # Usage
//!
//! ```
//! use xenon::prelude::*;
//! ```

// --- Core types (added incrementally as modules are implemented) ---
// Order strictly follows 01-architecture.md §7 prelude export list (lines 542-576).

// Tensor types — available after W8
// pub use crate::tensor::{TensorBase,
//                          Tensor, TensorView, TensorViewMut, ArcTensor};

// Dimension types — available after W3
// pub use crate::dimension::{Dimension, IntoDimension,
//                            Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6,
//                            IxDyn, Axis};

// Element traits — available after W4
// pub use crate::element::{Element, Numeric, RealScalar, ComplexScalar};

// Complex type — available after W5
// pub use crate::complex::Complex;

// Error types — active (W2T5)
pub use crate::error::{
    // Core types
    XenonError,
    Result,
    // Auxiliary enums
    AbiMismatchKind,
    ConversionFailureReason,
    FfiBackend,
    FfiErrorCategory,
    InvalidArgumentKind,
    InvalidLayoutReason,
    InvalidShapeKind,
    StorageConversionKind,
    StorageKindTag,
    TypedViewRejection,
    WorkspaceBorrowKind,
    WorkspaceBorrowState,
    WorkspaceErrorCategory,
};

// Construction convenience helpers — available after W22
// pub use crate::construct::{zeros, ones, eye, from_shape_vec};

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_prelude_error_imports() {
        // Verify XenonError is importable from prelude
        let e: XenonError = XenonError::InvalidLayout {
            operation: std::borrow::Cow::Borrowed("validate"),
            storage_kind: StorageKindTag::Owned,
            shape: vec![2, 3],
            strides: vec![3, 1],
            offset: 0,
            storage_len: 6,
            reason: InvalidLayoutReason::AccessRangeExceedsStorage,
        };
        let _ = format!("{}", e);
    }

    #[test]
    fn test_prelude_result_alias() {
        // Verify Result type alias is importable from prelude
        let _ok: Result<i32> = Ok(42);
    }

    #[test]
    fn test_error_is_debug_and_display() {
        let e = XenonError::DimensionMismatch {
            operation: std::borrow::Cow::Borrowed("test"),
            expected: 2,
            actual: 3,
        };
        // Must compile — confirms Debug + Display are in scope
        let _debug = format!("{:?}", e);
        let _display = format!("{}", e);
    }

    #[test]
    fn test_prelude_auxiliary_enums_importable() {
        // Verify auxiliary enums are importable from prelude
        let _cat = FfiErrorCategory::NullPointer {
            argument: std::borrow::Cow::Borrowed("data"),
        };
        let _kind = InvalidShapeKind::ProductOverflow;
        let _reason = ConversionFailureReason::FloatToInteger;
        let _backend = FfiBackend::Blas;
        let _tag = StorageKindTag::Shared;
    }

    #[test]
    fn test_xenon_error_direct_reexport_at_crate_root() {
        let _e: crate::XenonError = crate::XenonError::DimensionMismatch {
            operation: std::borrow::Cow::Borrowed("test"),
            expected: 1,
            actual: 2,
        };
    }
}
