//! The Xenon prelude.
//!
//! Re-exports the most commonly used types so downstream code can
//! write `use xenon::prelude::*;` instead of importing each item
//! individually.
//!
//! # Example
//!
//! ```rust
//! use xenon::prelude::*;
//! ```

// ── Public re-exports ────────────────────────────────────────────────

// Conversion traits
pub use crate::convert::{
    CastElement,
};

// Complex number type
pub use crate::complex::Complex;

// Element traits
pub use crate::element::{
    Element,
    Numeric,
    RealScalar,
    ComplexScalar,
};

// Tensor types
//
// Construction methods — zeros, ones, eye, from_shape_vec, from_scalar
// are inherent methods on TensorBase, available via the tensor
// re-exports below; no separate `pub use` is needed.
pub use crate::tensor::{
    TensorBase,
};

pub use crate::tensor::{
    ArcTensor,
    ArcTensor0,
    ArcTensor1,
    ArcTensor2,
    ArcTensor3,
    ArcTensor4,
    ArcTensor5,
    ArcTensor6,
    ArcTensorD,
};

pub use crate::tensor::{
    Tensor,
    Tensor0,
    Tensor1,
    Tensor2,
    Tensor3,
    Tensor4,
    Tensor5,
    Tensor6,
    TensorD,
};

pub use crate::tensor::{
    TensorView,
    TensorView0,
    TensorView1,
    TensorView2,
    TensorView3,
    TensorView4,
    TensorView5,
    TensorView6,
    TensorViewD,
};

pub use crate::tensor::{
    TensorViewMut,
    TensorViewMut0,
    TensorViewMut1,
    TensorViewMut2,
    TensorViewMut3,
    TensorViewMut4,
    TensorViewMut5,
    TensorViewMut6,
    TensorViewMutD,
};

// Dimension types
pub use crate::dimension::{
    Axis,
    Dimension,
    IntoDimension,
};

pub use crate::dimension::{
    Ix0,
    Ix1,
    Ix2,
    Ix3,
    Ix4,
    Ix5,
    Ix6,
    IxDyn,
};

// Index / slicing types
pub use crate::index::{
    SliceInfo,
    SliceInfoElem,
    SliceInfoIndices,
};

// Layout types
pub use crate::layout::{
    LayoutFlags,
    LayoutState,
    Strides,
};

// Storage types
pub use crate::storage::{
    Owned,
    ViewRepr,
    View,
    ViewMutRepr,
    ViewMut,
    ArcRepr,
};

pub use crate::storage::{
    Storage,
    StorageMut,
    StorageOwned,
    StorageShared,
    StorageIntoOwned,
};

// Error types and Result alias
pub use crate::error::{
    ConversionFailureReason,
    FfiBackend,
    FfiErrorCategory,
    InvalidArgumentKind,
    InvalidLayoutReason,
    InvalidShapeKind,
    Result,
    StorageKindTag,
    TypedViewRejection,
    WorkspaceBorrowKind,
    WorkspaceBorrowState,
    WorkspaceErrorCategory,
    XenonError,
};

// Operator overload helper
pub use crate::overload::Scalar;

// Set trait
pub use crate::set::UniqueElement;

// Matrix operations
pub use crate::matrix::dot;

// Scratch workspace type
pub use crate::workspace::Workspace;

// FFI types
pub use crate::ffi::{
    BlasInfo,
    TensorExportRaw,
    TensorExportMutRaw,
};

// ── Test-only re-exports ────────────────────────────────────────────
//
// Integration tests under `tests/` are external crates and cannot reach
// `pub(crate)` items inside `dispatch` or `parallel`. The items below
// are re-exported solely so those tests can observe dispatch decisions,
// tweak thresholds, and exercise parallel kernels directly.

pub use crate::dispatch::{
    ExecPath,
    select_exec_path
};

#[cfg(any(feature = "parallel", feature = "simd"))]
pub use crate::dispatch::ThresholdTestGuard;

#[cfg(feature = "simd")]
pub use crate::dispatch::{
    reset_simd_threshold,
    set_simd_threshold
};

#[cfg(feature = "parallel")]
pub use crate::dispatch::{
    ParallelExecStrategy,
    ParallelGuard,
    reset_parallel_threshold,
    set_parallel_threshold
};

#[cfg(feature = "parallel")]
pub use crate::parallel::unary::{par_map, par_map_checked};

#[cfg(feature = "parallel")]
pub use crate::parallel::binary::{par_zip, par_zip_checked};

#[cfg(feature = "parallel")]
pub use crate::parallel::dot::par_dot;

#[cfg(feature = "parallel")]
pub use crate::parallel::sum::par_sum;

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    /// Verifies that `XenonError` and its variants (e.g., `InvalidLayout`)
    /// are importable and constructible from the prelude.
    #[test]
    fn test_prelude_error_imports() {
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

    /// Verifies that `Result<T>` alias resolves correctly from the prelude.
    #[test]
    fn test_prelude_result_alias() {
        let _ok: Result<i32> = Ok(42);
    }

    /// Verifies that `XenonError` implements `Debug` and `Display`.
    #[test]
    fn test_error_is_debug_and_display() {
        let e = XenonError::DimensionMismatch {
            operation: std::borrow::Cow::Borrowed("test"),
            expected: 2,
            actual: 3,
        };
        let _debug = format!("{:?}", e);
        let _display = format!("{}", e);
    }

    /// Verifies that auxiliary error-related enums (`FfiErrorCategory`,
    /// `InvalidShapeKind`, etc.) are importable from the prelude.
    #[test]
    fn test_prelude_auxiliary_enums_importable() {
        let _cat = FfiErrorCategory::NullPointer {
            argument: std::borrow::Cow::Borrowed("data"),
        };
        let _kind = InvalidShapeKind::ProductOverflow;
        let _reason = ConversionFailureReason::FloatToInteger;
        let _backend = FfiBackend::Blas;
        let _tag = StorageKindTag::Shared;
    }

    /// Verifies that `XenonError` is also re-exported directly at the
    /// crate root (`crate::XenonError`).
    #[test]
    fn test_xenon_error_direct_reexport_at_crate_root() {
        let _e: crate::XenonError = XenonError::DimensionMismatch {
            operation: std::borrow::Cow::Borrowed("test"),
            expected: 1,
            actual: 2,
        };
    }

    /// Verifies that `Tensor`, `TensorView`, `ArcTensor`, and typed
    /// aliases (`Tensor1`, `Tensor2`) are importable from the prelude.
    #[test]
    fn test_prelude_tensor_imports() {
        let _: Option<Tensor<f64, Ix2>> = None;
        let _: Option<TensorView<'_, f64, Ix2>> = None;
        let _: Option<ArcTensor<f64, Ix2>> = None;
        let _: Option<Tensor1<f64>> = None;
        let _: Option<Tensor2<f64>> = None;
    }

    /// Verifies that dimension types (`Ix2`, `Axis`) are importable
    /// from the prelude.
    #[test]
    fn test_prelude_dimension_imports() {
        let _dim: Ix2 = Ix2(2, 3);
        let _axis: Axis = Axis(0);
    }

    /// Verifies that element trait bounds (`Element`, `Numeric`,
    /// `RealScalar`) are checked through the prelude.
    #[test]
    fn test_prelude_element_trait_imports() {
        fn _check_element<T: Element>() {}
        fn _check_numeric<T: Numeric>() {}
        fn _check_real<T: RealScalar>() {}
        _check_element::<f64>();
        _check_numeric::<f64>();
        _check_real::<f64>();
    }

    /// Verifies that `Complex` type is importable from the prelude.
    #[test]
    fn test_prelude_complex_imports() {
        let _: Complex<f64> = Complex::new(1.0, 2.0);
    }

    /// Verifies that tensor constructors (`zeros`, `ones`, `eye`,
    /// `from_scalar`) are available from the prelude.
    #[test]
    fn test_prelude_construct_imports() {
        let _ = Tensor1::<f64>::zeros([3]);
        let _ = Tensor1::<f64>::ones([3]);
        let _ = Tensor2::<f64>::eye(3);
        let _ = Tensor0::from_scalar(42.0_f64);
    }

    /// Verifies that storage types (`StorageKindTag`) are importable
    /// from the prelude.
    #[test]
    fn test_prelude_storage_imports() {
        let _tag: StorageKindTag = StorageKindTag::Owned;
    }

    /// Verifies that `SliceInfoElem` and its variants are importable
    /// from the prelude.
    #[test]
    fn test_prelude_sliceinfo_imports() {
        let _elem: SliceInfoElem = SliceInfoElem::Range { start: 0, end: 1 };
    }
}
