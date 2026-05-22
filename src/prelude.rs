//! The Xenon prelude.
//!
//! Re-exports the most commonly used types so downstream code can
//! write `use xenon::prelude::*;` instead of importing each item.
//! Includes: `Tensor`, dimension aliases, `XenonError`, `Result`, and
//! the conventional element/storage marker traits required by typical
//! tensor construction and arithmetic.
//!
//! # Example
//!
//! ```rust
//! use xenon::prelude::*;
//! ```

// --- Core types (added incrementally as modules are implemented) ---
// Order strictly follows 01-architecture.md §7 prelude export list (lines 542-576).

// Tensor types — available after W8
pub use crate::tensor::{
    ArcTensor, ArcTensor0, ArcTensor1, ArcTensor2, ArcTensor3, ArcTensor4, ArcTensor5,
    ArcTensor6, ArcTensorD, Tensor, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6,
    TensorBase, TensorD, TensorView, TensorView0, TensorView1, TensorView2, TensorView3,
    TensorView4, TensorView5, TensorView6, TensorViewD, TensorViewMut, TensorViewMut0,
    TensorViewMut1, TensorViewMut2, TensorViewMut3, TensorViewMut4, TensorViewMut5,
    TensorViewMut6, TensorViewMutD,
};

// Dimension types — available after W3
pub use crate::dimension::{
    Axis, Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
};

// Element traits — available after W4
pub use crate::element::{CastElement, CastTo, ComplexScalar, Element, Numeric, RealScalar};

// Complex type — available after W5
pub use crate::complex::Complex;

// Error types — active (W2T5)
pub use crate::error::{
    // Auxiliary enums
    AbiMismatchKind,
    ConversionFailureReason,
    FfiBackend,
    FfiErrorCategory,
    InvalidArgumentKind,
    InvalidLayoutReason,
    InvalidShapeKind,
    Result,
    StorageConversionKind,
    StorageKindTag,
    TypedViewRejection,
    WorkspaceBorrowKind,
    WorkspaceBorrowState,
    WorkspaceErrorCategory,
    // Core types
    XenonError,
};

// Construction convenience helpers — available after W22
// Note: zeros/ones/eye/from_scalar are inherent methods on TensorBase,
// not free functions. They become available when Tensor1/etc. are in scope
// (via the tensor re-exports above). No separate re-export needed.

// Set operations — available after W19
pub use crate::set::UniqueElement;

// Storage types — available after W7
pub use crate::storage::{
    ArcRepr, Owned, Storage, StorageIntoOwned, StorageMut, StorageOwned, StorageShared, View,
    ViewMut, ViewMutRepr, ViewRepr,
};

// Layout types — available after W6
pub use crate::layout::{LayoutFlags, LayoutState, Strides};

// Index types — available after W17
pub use crate::index::{SliceInfo, SliceInfoElem, SliceInfoIndices};

// Operator overload types — available after W19
pub use crate::overload::Scalar;

// Workspace type — available after W24
pub use crate::workspace::Workspace;

// FFI types — available after W13
// Note: FfiBackend and FfiErrorCategory are already exported via `crate::error` above;
// no need to re-export them from `crate::ffi` to avoid name conflicts.
pub use crate::ffi::{BlasInfo, ElementType, TensorExportMutRaw, TensorExportRaw};

pub use crate::matrix::dot;

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

    #[test]
    fn test_prelude_tensor_imports() {
        let _: Option<Tensor<f64, Ix2>> = None;
        let _: Option<TensorView<'_, f64, Ix2>> = None;
        let _: Option<ArcTensor<f64, Ix2>> = None;
        let _: Option<Tensor1<f64>> = None;
        let _: Option<Tensor2<f64>> = None;
    }

    #[test]
    fn test_prelude_dimension_imports() {
        let _dim: Ix2 = Ix2(2, 3);
        let _axis: Axis = Axis(0);
    }

    #[test]
    fn test_prelude_element_trait_imports() {
        fn _check_element<T: Element>() {}
        fn _check_numeric<T: Numeric>() {}
        fn _check_real<T: RealScalar>() {}
        _check_element::<f64>();
        _check_numeric::<f64>();
        _check_real::<f64>();
    }

    #[test]
    fn test_prelude_complex_imports() {
        let _: Complex<f64> = Complex::new(1.0, 2.0);
    }

    #[test]
    fn test_prelude_construct_imports() {
        let _ = Tensor1::<f64>::zeros([3]);
        let _ = Tensor1::<f64>::ones([3]);
        let _ = Tensor2::<f64>::eye(3);
        let _ = Tensor0::from_scalar(42.0_f64);
    }

    #[test]
    fn test_prelude_storage_imports() {
        let _tag: StorageKindTag = StorageKindTag::Owned;
    }

    #[test]
    fn test_prelude_sliceinfo_imports() {
        let _elem: SliceInfoElem = SliceInfoElem::Range { start: 0, end: 1 };
    }
}