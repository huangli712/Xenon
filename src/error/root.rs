//! Primary error type and crate `Result` alias.
//!
//! Defines [`XenonError`] — the unified error enum for all public Xenon APIs —
//! and the [`Result`] type alias used throughout the crate.
//! Also provides constructor helpers for commonly used error variants
//! (e.g., workspace split/boundary errors).

use core::fmt::{self, Display, Formatter};
use std::borrow::Cow;
use std::error::Error;
use std::vec::Vec;

use super::display::FmtShape;
use super::{
    ConversionFailureReason, FfiBackend, FfiErrorCategory,
    InvalidArgumentKind, InvalidLayoutReason, InvalidShapeKind,
    StorageKindTag,
    WorkspaceBorrowKind, WorkspaceBorrowState, WorkspaceErrorCategory,
};

/// Unified recoverable error type for all public Xenon APIs.
///
/// This enum is marked `#[non_exhaustive]`: downstream `match` expressions
/// MUST include a wildcard arm (`_ => ...`) and MUST NOT exhaustively pattern
/// against the listed variants. This lets future Xenon versions add new
///
/// # Examples
///
/// Access via direct re-export:
///
/// ```
/// use xenon::XenonError;
/// let _: XenonError = XenonError::DimensionMismatch {
///     operation: std::borrow::Cow::Borrowed("doc"),
///     expected: 1,
///     actual: 2,
/// };
/// ```
///
/// Access via prelude:
///
/// ```
/// use xenon::prelude::*;
/// let _: XenonError = XenonError::DimensionMismatch {
///     operation: std::borrow::Cow::Borrowed("doc"),
///     expected: 1,
///     actual: 2,
/// };
/// ```
/// top-level error categories (within the same SemVer major) without forcing
/// a breaking change on every downstream `match`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum XenonError {
    /// Two shapes are incompatible for the requested operation.
    ShapeMismatch {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Shape of the left operand.
        left_shape: Vec<usize>,
        /// Shape of the right operand.
        right_shape: Vec<usize>,
    },

    /// Broadcasting shapes are incompatible.
    BroadcastError {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Shape of the left-hand side.
        lhs_shape: Vec<usize>,
        /// Shape of the right-hand side.
        rhs_shape: Vec<usize>,
        /// Expected target shape, if one was computed.
        attempted_target_shape: Option<Vec<usize>>,
        /// Axis along which broadcasting was attempted, if applicable.
        axis: Option<usize>,
    },

    /// Invalid memory layout detected (construction, view, raw-parts, etc.).
    InvalidLayout {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Kind of storage being validated.
        storage_kind: StorageKindTag,
        /// Shape of the tensor.
        shape: Vec<usize>,
        /// Strides of the tensor.
        strides: Vec<usize>,
        /// Offset into the storage.
        offset: usize,
        /// Total length of the storage in elements.
        storage_len: usize,
        /// Detailed reason the layout was rejected.
        reason: InvalidLayoutReason,
    },

    /// Axis index is out of the valid dimension range.
    InvalidAxis {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// The axis that was out of bounds.
        axis: usize,
        /// Number of dimensions in the tensor.
        ndim: usize,
        /// Shape of the tensor.
        shape: Vec<usize>,
    },

    /// Shape value invalid for the requested operation.
    InvalidShape {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// The shape that was rejected.
        shape: Vec<usize>,
        /// Kind of shape validation failure.
        kind: InvalidShapeKind,
        /// The specific dimension that caused the failure, if identifiable.
        offending_dim: Option<usize>,
    },

    /// The number of dimensions does not match what the operation expects.
    DimensionMismatch {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        actual: usize,
    },

    /// Generic invalid argument error with structured classification.
    InvalidArgument {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Structured classification of the invalid argument.
        kind: InvalidArgumentKind,
    },

    /// Storage mode incompatible with the requested operation.
    InvalidStorageMode {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Expected storage kind.
        expected: StorageKindTag,
        /// Actual storage kind.
        actual: StorageKindTag,
        /// Shape of the tensor, if available.
        shape: Option<Vec<usize>>,
    },

    /// FFI-related error (raw-parts, BLAS/LAPACK interoperability).
    Ffi {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Structured classification of the FFI error.
        category: FfiErrorCategory,
        /// Backend involved in the FFI call.
        backend: FfiBackend,
    },

    /// Workspace operation error (alloc, borrow, split, capacity).
    Workspace {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Structured classification of the workspace error.
        category: WorkspaceErrorCategory,
    },

    /// Multi-dimensional index out of bounds.
    IndexOutOfBounds {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// The full attempted index (one component per axis).
        attempted_index: Vec<usize>,
        /// Axis along which the index was out of bounds.
        axis: usize,
        /// Shape of the tensor.
        shape: Vec<usize>,
    },

    /// Element type conversion failed (e.g. `cast`, `Complex -> Real`).
    TypeConversion {
        /// The operation that failed.
        operation: Cow<'static, str>,
        /// Name of the source type.
        source_type: &'static str,
        /// Name of the target type.
        target_type: &'static str,
        /// Reason for the conversion failure.
        reason: ConversionFailureReason,
        /// Index of the element that caused the failure, if known.
        element_index: Option<usize>,
    },
}

impl Display for XenonError {
    /// Formats the Xenon error with all structured context fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { operation, left_shape, right_shape } => {
                write!(f, "shape mismatch in `{operation}`: cannot operate on {} and {}",
                    FmtShape(left_shape),
                    FmtShape(right_shape),
                )
            },
            Self::BroadcastError { operation, lhs_shape, rhs_shape, attempted_target_shape, axis } => {
                write!(f, "broadcast error in `{operation}`: cannot broadcast {} with {}",
                    FmtShape(lhs_shape),
                    FmtShape(rhs_shape),
                )?;
                if let Some(target) = attempted_target_shape {
                    write!(f, " (attempted target: {})", FmtShape(target))?;
                }
                if let Some(ax) = axis {
                    write!(f, " (axis: {ax})")?;
                }
                Ok(())
            },
            Self::InvalidLayout { operation, storage_kind, shape, strides, offset, storage_len, reason } => {
                write!(f, "invalid layout ({reason}) in `{operation}`: storage={storage_kind}, ")?;
                write!(f, "shape={}, strides={}, offset={offset}, len={storage_len}",
                    FmtShape(shape),
                    FmtShape(strides),
                )
            },
            Self::InvalidAxis { operation, axis, ndim, shape } => {
                write!(f, "invalid axis {axis} in `{operation}`: valid range is 0..{ndim} ")?;
                write!(f, "for shape {}", FmtShape(shape))
            },
            Self::InvalidShape { operation, shape, kind, offending_dim } => {
                write!(f, "invalid shape ({kind}) in `{operation}`: shape={}", FmtShape(shape))?;
                if let Some(dim) = offending_dim {
                    write!(f, " (offending dim: {dim})")?;
                }
                Ok(())
            },
            Self::DimensionMismatch { operation, expected, actual } => {
                write!(f, "dimension mismatch in `{operation}`: expected {expected} ")?;
                write!(f, "dimensions, got {actual}")
            },
            Self::InvalidArgument { operation, kind } => {
                write!(f, "invalid argument ({kind}) in `{operation}`")
            },
            Self::InvalidStorageMode { operation, expected, actual, shape } => {
                write!(f, "invalid storage mode in `{operation}`: expected {expected}, ")?;
                write!(f, "got {actual}")?;
                if let Some(s) = shape {
                    write!(f, " for shape {}", FmtShape(s))?;
                }
                Ok(())
            },
            Self::Ffi { operation, category, backend } => {
                write!(f, "FFI error (`{category}`) in `{operation}` (backend: {backend})")
            },
            Self::Workspace { operation, category } => {
                write!(f, "workspace error (`{category}`) in `{operation}`")
            },
            Self::IndexOutOfBounds { operation, attempted_index, axis, shape } => {
                write!(f, "index out of bounds in `{operation}`: attempted {} at ", FmtShape(attempted_index))?;
                write!(f, "axis {axis} (shape: {})", FmtShape(shape))
            },
            Self::TypeConversion { operation, source_type, target_type, reason, element_index } => {
                write!(f, "type conversion failed in `{operation}`: {source_type} -> ")?;
                write!(f, "{target_type} ({reason})")?;
                if let Some(idx) = element_index {
                    write!(f, " at element index {idx}")?;
                }
                Ok(())
            },
        }
    }
}

impl Error for XenonError {
    /// All `XenonError` variants are leaf errors with no chained source.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

// Constructor helpers for common error variants.
impl XenonError {
    // --- Workspace constructor helpers ---
    //
    // Each helper preserves the `operation` field and accepts structured
    // borrow / overflow context so callers (the borrow/split/expand modules)
    // never lose diagnostic fidelity. The `operation` string is `&'static str`
    // to remain `Cow::Borrowed`-friendly with no allocation.

    /// Construct a `Workspace::SplitOutOfBounds` error.
    pub fn workspace_split_oob(operation: &'static str, mid: usize, len: usize) -> Self {
        XenonError::Workspace {
            operation: Cow::Borrowed(operation),
            category: WorkspaceErrorCategory::SplitOutOfBounds { mid, len },
        }
    }

    /// Construct a `Workspace::BorrowConflict` error.
    pub fn workspace_borrow_conflict(
        operation: &'static str,
        requested: WorkspaceBorrowKind,
        current: WorkspaceBorrowState,
    ) -> Self {
        XenonError::Workspace {
            operation: Cow::Borrowed(operation),
            category: WorkspaceErrorCategory::BorrowConflict { requested, current },
        }
    }

    /// Construct a `Workspace::GrowOverflow` error.
    pub fn workspace_grow_overflow(
        operation: &'static str,
        current_capacity: usize,
        additional: usize,
    ) -> Self {
        XenonError::Workspace {
            operation: Cow::Borrowed(operation),
            category: WorkspaceErrorCategory::GrowOverflow {
                current_capacity,
                additional,
            },
        }
    }
}

/// Canonical `Result` alias used by all public Xenon APIs.
///
/// Equivalent to `core::result::Result<T, XenonError>`.
pub type Result<T> = core::result::Result<T, XenonError>;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{
        ConversionFailureReason, FfiBackend, FfiErrorCategory,
        InvalidArgumentKind, InvalidLayoutReason, InvalidShapeKind,
        StorageKindTag, TypedViewRejection,
        WorkspaceBorrowKind, WorkspaceBorrowState, WorkspaceErrorCategory,
    };
    use std::error::Error;

    /// Verify XenonError enum is constructable with each variant.
    #[test]
    fn test_error_variants_construct() {
        // ShapeMismatch
        let e = XenonError::ShapeMismatch {
            operation: Cow::Borrowed("dot"),
            left_shape: vec![2, 3],
            right_shape: vec![3, 4],
        };
        assert!(!format!("{:?}", e).is_empty());

        // IndexOutOfBounds
        let e = XenonError::IndexOutOfBounds {
            operation: Cow::Borrowed("slice"),
            attempted_index: vec![0, 5],
            axis: 1,
            shape: vec![3, 4],
        };
        if let XenonError::IndexOutOfBounds {
            axis,
            attempted_index,
            ..
        } = &e
        {
            assert_eq!(*axis, 1);
            assert_eq!(attempted_index, &vec![0, 5]);
        } else {
            panic!("variant mismatch");
        }

        // TypeConversion
        let e = XenonError::TypeConversion {
            operation: Cow::Borrowed("cast"),
            source_type: "f64",
            target_type: "i32",
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        };
        if let XenonError::TypeConversion { source_type, .. } = &e {
            assert_eq!(*source_type, "f64");
        } else {
            panic!("variant mismatch");
        }

        // DimensionMismatch
        let e = XenonError::DimensionMismatch {
            operation: Cow::Borrowed("reshape"),
            expected: 2,
            actual: 3,
        };
        if let XenonError::DimensionMismatch { expected, .. } = &e {
            assert_eq!(*expected, 2);
        } else {
            panic!("variant mismatch");
        }

        // Ffi
        let e = XenonError::Ffi {
            operation: Cow::Borrowed("export"),
            category: FfiErrorCategory::NullPointer {
                argument: Cow::Borrowed("data"),
            },
            backend: FfiBackend::RawParts,
        };
        if let XenonError::Ffi { operation, .. } = &e {
            assert_eq!(operation, "export");
        } else {
            panic!("variant mismatch");
        }
    }

    /// Verify debug formatting does not panic for any error variant.
    #[test]
    fn test_error_debug_no_panic() {
        let errors = [
            XenonError::ShapeMismatch {
                operation: Cow::Borrowed("reshape"),
                left_shape: vec![],
                right_shape: vec![1],
            },
            XenonError::InvalidShape {
                operation: Cow::Borrowed("from_shape_vec"),
                shape: vec![2, 3],
                kind: InvalidShapeKind::ElementCountMismatch {
                    expected: 6,
                    actual: 5,
                },
                offending_dim: None,
            },
            XenonError::BroadcastError {
                operation: Cow::Borrowed("add"),
                lhs_shape: vec![2, 1],
                rhs_shape: vec![3, 1],
                attempted_target_shape: None,
                axis: None,
            },
        ];
        for e in &errors {
            let _ = format!("{:?}", e);
        }
    }

    /// Verify Clone + PartialEq roundtrip consistency.
    #[test]
    fn test_clone_eq_roundtrip() {
        let e1 = XenonError::InvalidAxis {
            operation: Cow::Borrowed("sum"),
            axis: 1,
            ndim: 2,
            shape: vec![3, 4],
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);

        let e3 = XenonError::InvalidAxis {
            operation: Cow::Borrowed("sum"),
            axis: 0,
            ndim: 2,
            shape: vec![3, 4],
        };
        assert_ne!(e1, e3);
    }

    /// Verify IndexOutOfBounds carries axis and shape context.
    #[test]
    fn test_index_error_reports_axis_and_shape() {
        let e = XenonError::IndexOutOfBounds {
            operation: Cow::Borrowed("index"),
            attempted_index: vec![2, 8],
            axis: 1,
            shape: vec![3, 4],
        };
        if let XenonError::IndexOutOfBounds {
            axis,
            shape,
            attempted_index,
            ..
        } = &e
        {
            assert_eq!(*axis, 1);
            assert_eq!(shape, &vec![3, 4]);
            assert_eq!(attempted_index, &vec![2, 8]);
        } else {
            panic!("variant mismatch");
        }
    }

    /// Verify Result type alias is usable.
    #[test]
    fn test_result_alias_usable() {
        let ok: Result<i32> = Ok(42);
        if let Ok(val) = ok {
            assert_eq!(val, 42);
        } else {
            panic!("expected Ok");
        }

        let err: Result<i32> = Err(XenonError::DimensionMismatch {
            operation: Cow::Borrowed("test"),
            expected: 1,
            actual: 2,
        });
        assert!(err.is_err());
    }

    /// Verify Display output contains operation name and shape info.
    #[test]
    fn test_display_contains_structured_info() {
        let e = XenonError::ShapeMismatch {
            operation: Cow::Borrowed("dot"),
            left_shape: vec![2, 3],
            right_shape: vec![3, 4],
        };
        let s = format!("{}", e);
        assert!(s.contains("dot"));
        assert!(s.contains("[2 × 3]"));
        assert!(s.contains("[3 × 4]"));
    }

    /// Verify IndexOutOfBounds Display includes operation, axis, and shape.
    #[test]
    fn test_display_index_out_of_bounds() {
        let e = XenonError::IndexOutOfBounds {
            operation: Cow::Borrowed("slice"),
            attempted_index: vec![0, 5],
            axis: 1,
            shape: vec![3, 4],
        };
        let s = format!("{}", e);
        assert!(s.contains("slice"));
        assert!(s.contains("axis 1"));
        assert!(s.contains("[3 × 4]"));
    }

    /// Verify TypeConversion Display includes source/target types and reason.
    #[test]
    fn test_display_type_conversion() {
        let e = XenonError::TypeConversion {
            operation: Cow::Borrowed("cast"),
            source_type: "f64",
            target_type: "i32",
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        };
        let s = format!("{}", e);
        assert!(s.contains("f64"));
        assert!(s.contains("i32"));
        assert!(s.contains("float to integer"));
    }

    /// Verify BroadcastError Display includes all shapes when present.
    #[test]
    fn test_display_broadcast_error() {
        let e = XenonError::BroadcastError {
            operation: Cow::Borrowed("add"),
            lhs_shape: vec![3, 1],
            rhs_shape: vec![1, 4],
            attempted_target_shape: Some(vec![3, 4]),
            axis: None,
        };
        let s = format!("{}", e);
        assert!(s.contains("[3 × 1]"));
        assert!(s.contains("[1 × 4]"));
        assert!(s.contains("[3 × 4]"));
    }

    /// Verify optional fields are omitted in Display output when `None`,
    /// never rendered as `Some(...)` or `None`.
    #[test]
    fn test_display_option_fields_render_any() {
        let e = XenonError::BroadcastError {
            operation: Cow::Borrowed("add"),
            lhs_shape: vec![3, 1],
            rhs_shape: vec![1, 4],
            attempted_target_shape: None,
            axis: None,
        };
        let s = format!("{}", e);
        assert!(!s.contains("Some("));
        assert!(!s.contains("None"));
        // sanity: core structured fields still present
        assert!(s.contains("[3 × 1]"));
        assert!(s.contains("[1 × 4]"));
    }

    /// Verify `TypeConversion.operation` field appears in Display output.
    #[test]
    fn test_type_conversion_carries_operation() {
        let e = XenonError::TypeConversion {
            operation: Cow::Borrowed("cast"),
            source_type: "f64",
            target_type: "i32",
            reason: ConversionFailureReason::FloatToInteger,
            element_index: Some(7),
        };
        let s = format!("{}", e);
        assert!(s.contains("cast"));
        assert!(s.contains("element index 7"));
    }

    /// Verify `source_type` / `target_type` are written directly in Display,
    /// not wrapped in `{:?}` or TypeId style.
    #[test]
    fn test_type_conversion_uses_element_type_name() {
        let e = XenonError::TypeConversion {
            operation: Cow::Borrowed("cast"),
            source_type: "Complex<f64>",
            target_type: "f64",
            reason: ConversionFailureReason::NonZeroImaginaryPart,
            element_index: None,
        };
        let s = format!("{}", e);
        // type names appear directly (not Debug-wrapped)
        assert!(s.contains("Complex<f64>"));
        assert!(s.contains("f64"));
        // reason uses Display, outputs readable text
        assert!(s.contains("non-zero imaginary part"));
    }

    /// Verify `XenonError` implements `std::error::Error`.
    #[test]
    fn test_error_trait_implemented() {
        fn assert_error<E: Error>(_: &E) {}
        let e = XenonError::IndexOutOfBounds {
            operation: Cow::Borrowed("index"),
            attempted_index: vec![0],
            axis: 0,
            shape: vec![5],
        };
        assert_error(&e);
    }

    /// Verify `source()` returns `None` for leaf (non-chained) variants.
    #[test]
    fn test_source_returns_none_for_leaf_variants() {
        let e = XenonError::IndexOutOfBounds {
            operation: Cow::Borrowed("index"),
            attempted_index: vec![0],
            axis: 0,
            shape: vec![5],
        };
        assert!(e.source().is_none());

        let e = XenonError::ShapeMismatch {
            operation: Cow::Borrowed("test"),
            left_shape: vec![],
            right_shape: vec![1],
        };
        assert!(e.source().is_none());
    }

    /// Verify `source()` returns `None` for `Ffi` and `Workspace` variants.
    #[test]
    fn test_source_returns_none_for_ffi_and_workspace() {
        let e = XenonError::Ffi {
            operation: Cow::Borrowed("check"),
            category: FfiErrorCategory::NullPointer {
                argument: Cow::Borrowed("ptr"),
            },
            backend: FfiBackend::RawParts,
        };
        assert!(e.source().is_none());

        let e = XenonError::Workspace {
            operation: Cow::Borrowed("new"),
            category: WorkspaceErrorCategory::TypedViewRejected {
                detail: TypedViewRejection::ZeroSizedType,
            },
        };
        assert!(e.source().is_none());
    }

    /// Verify `XenonError` is `Send + Sync` for use across threads.
    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<XenonError>();
    }

    /// Verify `XenonError` is usable as `Box<dyn std::error::Error>`.
    #[test]
    fn test_dyn_error_compatible() {
        let e: Box<dyn Error> = Box::new(XenonError::DimensionMismatch {
            operation: Cow::Borrowed("reshape"),
            expected: 2,
            actual: 3,
        });
        assert!(e.to_string().contains("2"));
    }

    /// Verify DimensionMismatch carries operation, expected, actual fields.
    #[test]
    fn test_dimension_mismatch_variant_fields() {
        let err = XenonError::DimensionMismatch {
            operation: Cow::Borrowed("Ix3::try_from_dyn"),
            expected: 3,
            actual: 4,
        };
        match err {
            XenonError::DimensionMismatch {
                operation,
                expected,
                actual,
            } => {
                assert_eq!(operation, "Ix3::try_from_dyn");
                assert_eq!(expected, 3);
                assert_eq!(actual, 4);
            },
            _ => panic!("not DimensionMismatch"),
        }
    }

    /// Verify Display format includes operation, expected, and actual.
    #[test]
    fn test_dimension_mismatch_display_includes_operation() {
        let err = XenonError::DimensionMismatch {
            operation: Cow::Borrowed("Ix2::try_from_dyn"),
            expected: 2,
            actual: 3,
        };
        let msg = format!("{err}");
        assert!(msg.contains("Ix2::try_from_dyn"), "msg: {msg}");
        assert!(msg.contains("expected 2"), "msg: {msg}");
        assert!(msg.contains("3"), "msg: {msg}");
    }

    // ── Workspace constructor helper tests ──

    /// Verify all 3 workspace constructor helpers carry `operation` and
    /// structured context.
    #[test]
    fn test_workspace_constructor_helpers() {
        // split_oob carries `operation`, `mid`, `len`.
        let err = XenonError::workspace_split_oob("Workspace::split_at_mut", 10, 5);
        let s = format!("{err:?}");
        assert!(s.contains("Workspace::split_at_mut"));
        assert!(s.contains("SplitOutOfBounds"));
        assert!(s.contains("mid: 10"));

        // borrow_conflict carries `operation`, `requested`, `current`.
        let err = XenonError::workspace_borrow_conflict(
            "Workspace::borrow",
            WorkspaceBorrowKind::Shared,
            WorkspaceBorrowState::Exclusive,
        );
        let s = format!("{err:?}");
        assert!(s.contains("Workspace::borrow"));
        assert!(s.contains("BorrowConflict"));
        assert!(s.contains("Shared"));
        assert!(s.contains("Exclusive"));

        // grow_overflow carries `operation`, `current_capacity`, `additional`.
        let err = XenonError::workspace_grow_overflow("Workspace::ensure_capacity", usize::MAX, 1);
        let s = format!("{err:?}");
        assert!(s.contains("Workspace::ensure_capacity"));
        assert!(s.contains("GrowOverflow"));
        assert!(s.contains("current_capacity"));
        assert!(s.contains("additional"));
    }

    // ── Display output tests for remaining XenonError variants ──

    /// Verify `InvalidLayout` variant Display includes storage kind and reason.
    #[test]
    fn test_display_invalid_layout() {
        let e = XenonError::InvalidLayout {
            operation: Cow::Borrowed("from_raw_parts"),
            storage_kind: StorageKindTag::ViewMut,
            shape: vec![2, 3],
            strides: vec![1, 2],
            offset: 0,
            storage_len: 6,
            reason: InvalidLayoutReason::AmbiguousOverlap,
        };
        let s = format!("{}", e);
        assert!(s.contains("from_raw_parts"));
        assert!(s.contains("ambiguous overlap"));
        assert!(s.contains("view mut"));
        assert!(s.contains("[2 × 3]"));
        assert!(s.contains("[1 × 2]"));
        assert!(s.contains("offset=0"));
        assert!(s.contains("len=6"));
    }

    /// Verify `InvalidAxis` variant Display includes ndim and shape.
    #[test]
    fn test_display_invalid_axis() {
        let e = XenonError::InvalidAxis {
            operation: Cow::Borrowed("sum"),
            axis: 2,
            ndim: 2,
            shape: vec![3, 4],
        };
        let s = format!("{}", e);
        assert!(s.contains("sum"));
        assert!(s.contains("axis 2"));
        assert!(s.contains("0..2"));
        assert!(s.contains("[3 × 4]"));
    }

    /// Verify `InvalidShape` variant Display includes kind and offending dim.
    #[test]
    fn test_display_invalid_shape() {
        let e = XenonError::InvalidShape {
            operation: Cow::Borrowed("from_shape_vec"),
            shape: vec![2, 3],
            kind: InvalidShapeKind::ElementCountMismatch { expected: 6, actual: 5 },
            offending_dim: Some(0),
        };
        let s = format!("{}", e);
        assert!(s.contains("from_shape_vec"));
        assert!(s.contains("element count mismatch"));
        assert!(s.contains("offending dim: 0"));
    }

    /// Verify `InvalidArgument` variant Display includes kind.
    #[test]
    fn test_display_invalid_argument() {
        let e = XenonError::InvalidArgument {
            operation: Cow::Borrowed("slice"),
            kind: InvalidArgumentKind::DuplicateOrEmpty { argument: Cow::Borrowed("axes") },
        };
        let s = format!("{}", e);
        assert!(s.contains("slice"));
        assert!(s.contains("duplicate or empty"));
    }

    /// Verify `InvalidStorageMode` variant Display includes shape when present.
    #[test]
    fn test_display_invalid_storage_mode() {
        let e = XenonError::InvalidStorageMode {
            operation: Cow::Borrowed("slice_mut"),
            expected: StorageKindTag::ViewMut,
            actual: StorageKindTag::View,
            shape: Some(vec![2, 3]),
        };
        let s = format!("{}", e);
        assert!(s.contains("slice_mut"));
        assert!(s.contains("view mut"));
        assert!(s.contains("view"));
        assert!(s.contains("[2 × 3]"));
    }

    /// Verify `Ffi` variant Display includes category and backend.
    #[test]
    fn test_display_ffi_error() {
        let e = XenonError::Ffi {
            operation: Cow::Borrowed("export"),
            category: FfiErrorCategory::InvalidRank { expected: 2, actual: 3 },
            backend: FfiBackend::Blas,
        };
        let s = format!("{}", e);
        assert!(s.contains("export"));
        assert!(s.contains("invalid rank"));
        assert!(s.contains("BLAS"));
    }

    /// Verify `Workspace` variant Display includes category.
    #[test]
    fn test_display_workspace_error() {
        let e = XenonError::Workspace {
            operation: Cow::Borrowed("new"),
            category: WorkspaceErrorCategory::AllocFailed { size: 2048, align: 128 },
        };
        let s = format!("{}", e);
        assert!(s.contains("new"));
        assert!(s.contains("allocation failed"));
        assert!(s.contains("2048"));
        assert!(s.contains("128"));
    }
}
