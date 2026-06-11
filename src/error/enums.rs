//! Structured error payloads used as [`XenonError`] variant fields.
//!
//! These enums carry the structured data for each error category —
//! shape validation, argument checks, layout validation, FFI errors,
//! workspace errors, and type conversions.

use core::fmt::{self, Debug, Display, Formatter};

use std::borrow::Cow;
use std::vec::Vec;

use super::display::FmtShape;

// --- FfiErrorCategory -------------------------------------------------------

/// FFI error category for `XenonError::Ffi`. All categories are
/// fully structured; no free-text fallback variant.
///
/// Marked `#[non_exhaustive]` to absorb future FFI-error categories without
/// breaking downstream `match` exhaustiveness within the same major version.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FfiErrorCategory {
    /// Caller passed a null raw pointer where a valid pointer was required.
    NullPointer {
        /// The argument name identifying the offending pointer.
        argument: Cow<'static, str>,
    },

    /// Rank check failed (e.g., BLAS layer expects 2D matrix).
    InvalidRank {
        /// Expected rank.
        expected: usize,
        /// Actual rank.
        actual: usize,
    },

    /// Layout cannot be expressed in the FFI ABI (e.g., non F-contiguous
    /// where BLAS layer requires column-major contiguous).
    BlasIncompatibleLayout {
        /// Shape of the tensor.
        shape: Vec<usize>,
        /// Strides of the tensor.
        strides: Vec<usize>,
    },

    /// `usize`-to-backend-integer conversion overflowed (e.g., to `i32` LDA).
    IntegerOverflow {
        /// The value that overflowed.
        value: usize,
        /// Bit-width of the target integer type.
        target_width_bits: u8,
    },
}

impl Display for FfiErrorCategory {
    /// Formats the FFI error category with structured detail fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullPointer { argument } => {
                write!(f, "null pointer for argument {argument}")
            },
            Self::InvalidRank { expected, actual } => {
                write!(f, "invalid rank: expected {expected}, actual {actual}")
            },
            Self::BlasIncompatibleLayout { shape, strides } => {
                write!(f, "BLAS-incompatible layout: shape {}, strides {}", FmtShape(shape), FmtShape(strides))
            },
            Self::IntegerOverflow { value, target_width_bits } => {
                write!(f, "integer overflow: {value} does not fit in i{target_width_bits}")
            },
        }
    }
}

// --- FfiBackend -------------------------------------------------------------

/// Backend identifier for `XenonError::Ffi.backend`.
///
/// Closed enum: any future backend must extend this enum (SemVer-tracked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiBackend {
    /// Generic raw-parts FFI (no specific backend library).
    RawParts,

    /// BLAS-compatible export.
    Blas,
}

impl Display for FfiBackend {
    /// Formats the FFI backend as a short human-readable label.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::RawParts => write!(f, "raw parts"),
            Self::Blas => write!(f, "BLAS"),
        }
    }
}

// --- WorkspaceErrorCategory -------------------------------------------------

/// Workspace error category for `XenonError::Workspace`. All categories
/// carry structured context; no free-text fallback variant.
///
/// Marked `#[non_exhaustive]` to allow new workspace-error categories in
/// future minor versions without breaking downstream `match` exhaustiveness.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkspaceErrorCategory {
    /// Underlying allocator returned failure (e.g., OOM / size==0 not
    /// allowed).
    AllocFailed {
        /// Requested allocation size in bytes.
        size: usize,
        /// Requested allocation alignment in bytes.
        align: usize,
    },

    /// Layout request violates `Layout::from_size_align` rules.
    InvalidLayout {
        /// Requested layout size in bytes.
        size: usize,
        /// Requested layout alignment in bytes.
        align: usize,
    },

    /// Borrow request conflicts with current borrow state.
    BorrowConflict {
        /// The kind of borrow that was requested.
        requested: WorkspaceBorrowKind,
        /// The current borrow state preventing the request.
        current: WorkspaceBorrowState,
    },

    /// `split_at_mut` mid index out of bounds for current view length.
    SplitOutOfBounds {
        /// The midpoint index that was out of bounds.
        mid: usize,
        /// The current view length.
        len: usize,
    },

    /// Capacity grow overflow.
    ///
    /// `current_capacity` is the currently available byte length of the
    /// region or workspace; `additional` is the requested additional
    /// bytes (always in BYTES). For typed-view `count * size_of::<T>()`
    /// overflows where `count` is in element units (not bytes), use
    /// `TypedViewRejection::TypedByteLengthOverflow` instead — see
    /// `TypedViewRejection::TypedByteLengthOverflow`.
    GrowOverflow {
        /// Current available byte capacity.
        current_capacity: usize,
        /// Requested additional bytes.
        additional: usize,
    },

    /// Typed view request rejected (e.g., ZST not supported, range not
    /// aligned for `T`, count×size_of overflow — the last via
    /// `TypedViewRejection::TypedByteLengthOverflow`).
    TypedViewRejected {
        /// Detail of the rejection.
        detail: TypedViewRejection,
    },
}

impl Display for WorkspaceErrorCategory {
    /// Formats the workspace error category with structured detail fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocFailed { size, align } => {
                write!(f, "allocation failed (size={size}, align={align})")
            },
            Self::InvalidLayout { size, align } => {
                write!(f, "invalid layout (size={size}, align={align})")
            },
            Self::BorrowConflict { requested, current } => {
                write!(f, "borrow conflict: requested {requested:?}, current {current:?}")
            },
            Self::SplitOutOfBounds { mid, len } => {
                write!(f, "split out of bounds (mid={mid}, len={len})")
            },
            Self::GrowOverflow { current_capacity, additional } => {
                write!(f, "grow overflow: capacity={current_capacity} + additional={additional}")
            },
            Self::TypedViewRejected { detail } => {
                write!(f, "typed view rejected: {detail:?}")
            },
        }
    }
}

// --- WorkspaceBorrowKind ----------------------------------------------------

/// Type of workspace borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceBorrowKind {
    /// A shared (immutable) borrow.
    Shared,

    /// An exclusive (mutable) borrow.
    Exclusive,

    /// A split (partitioned) borrow.
    Split,
}

impl Display for WorkspaceBorrowKind {
    /// Formats the workspace borrow kind as a short label.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared => f.write_str("shared"),
            Self::Exclusive => f.write_str("exclusive"),
            Self::Split => f.write_str("split"),
        }
    }
}

// --- WorkspaceBorrowState ---------------------------------------------------

/// Current borrow state tracked by the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceBorrowState {
    /// No active borrow.
    None,

    /// One or more shared borrows are active.
    Shared,

    /// An exclusive borrow is active.
    Exclusive,

    /// Multiple split borrows are active.
    SplitActive {
        /// Number of active splits.
        count: usize,
    },
}

impl Display for WorkspaceBorrowState {
    /// Formats the workspace borrow state as a short label.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Shared => f.write_str("shared"),
            Self::Exclusive => f.write_str("exclusive"),
            Self::SplitActive { count } => write!(f, "split active (count={count})"),
        }
    }
}

// --- TypedViewRejection -----------------------------------------------------

/// Identifies a typed view rejection reason.
///
/// Marked `#[non_exhaustive]` to allow new typed-view rejection kinds in
/// future minor versions without breaking downstream `match` exhaustiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypedViewRejection {
    /// `T` is a zero-sized type; typed view of ZST is rejected.
    ZeroSizedType,

    /// Buffer base address does not satisfy `align_of::<T>()`.
    AlignmentMismatch {
        /// Required alignment.
        required: usize,
        /// Actual alignment.
        actual: usize,
    },

    /// `count.checked_mul(size_of::<T>())` overflowed `usize`. We cannot
    /// represent the requested byte length, so reusing `GrowOverflow`
    /// (which expects bytes) would produce a misleading diagnostic. Carry
    /// `count` (element units) and `elem_size` (bytes per `T`) instead.
    TypedByteLengthOverflow {
        /// Requested element count.
        count: usize,
        /// Size of each element in bytes.
        elem_size: usize,
    },
}

impl Display for TypedViewRejection {
    /// Formats the typed view rejection reason with structured detail fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSizedType => {
                write!(f, "zero-sized type")
            },
            Self::AlignmentMismatch { required, actual } => {
                write!(f, "alignment mismatch: required {required}, actual {actual}")
            },
            Self::TypedByteLengthOverflow { count, elem_size } => {
                write!(f, "byte length overflow: count={count}, elem_size={elem_size}")
            },
        }
    }
}

// --- ConversionFailureReason ------------------------------------------------

/// Reason for type conversion failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionFailureReason {
    /// Integer → narrower integer where value doesn't fit.
    LossyIntegerNarrowing,

    /// Float → narrower float where value doesn't fit.
    LossyFloatNarrowing,

    /// Float → integer conversion.
    FloatToInteger,

    /// Integer → float loses precision for the specific value.
    IntegerToFloatPrecisionLoss,

    /// Complex → real attempted but imaginary part is non-zero.
    NonZeroImaginaryPart,
}

impl Display for ConversionFailureReason {
    /// Formats the conversion failure reason as a short label.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::LossyIntegerNarrowing => write!(f, "lossy integer narrowing"),
            Self::LossyFloatNarrowing => write!(f, "lossy float narrowing"),
            Self::FloatToInteger => write!(f, "float to integer"),
            Self::IntegerToFloatPrecisionLoss => write!(f, "integer to float precision loss"),
            Self::NonZeroImaginaryPart => write!(f, "non-zero imaginary part"),
        }
    }
}

// --- InvalidArgumentKind ----------------------------------------------------

/// Kind for `XenonError::InvalidArgument`.
///
/// Marked `#[non_exhaustive]` to allow new invalid-argument kinds in
/// future minor versions.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidArgumentKind {
    /// Range slice `start..end` is out of `[0, axis_len]`.
    RangeOutOfBounds {
        /// The axis index.
        axis: usize,
        /// Length of the axis.
        axis_len: usize,
        /// Start of the range (inclusive).
        start: usize,
        /// End of the range (exclusive).
        end: usize,
    },

    /// Range slice has `start > end`.
    RangeStartAfterEnd {
        /// The axis index.
        axis: usize,
        /// Start of the invalid range.
        start: usize,
        /// End of the invalid range.
        end: usize,
    },

    /// Numeric parameter outside its required domain.
    NumericOutOfRange {
        /// Name of the offending argument.
        argument: Cow<'static, str>,
        /// The required domain description.
        domain: Cow<'static, str>,
        /// The actual value received.
        actual: Cow<'static, str>,
    },

    /// Threshold / chunk-size / max-workers etc. configuration violated.
    InvalidConfig {
        /// Name of the configuration argument.
        argument: Cow<'static, str>,
        /// The constraint that was violated.
        constraint: Cow<'static, str>,
        /// The actual value provided.
        actual: Cow<'static, str>,
    },

    /// Unique-list / set parameter contained duplicate or empty groups.
    DuplicateOrEmpty {
        /// Name of the offending argument.
        argument: Cow<'static, str>,
    },

    /// Caller-provided shape parameter inconsistent with operation
    /// (e.g., `clip` min > max, `reshape` shape product mismatch but
    /// reported via `InvalidShape::ElementCountMismatch` instead — this
    /// variant covers operation-specific argument validation).
    OperationSpecific {
        /// Name of the offending argument.
        argument: Cow<'static, str>,
        /// The constraint that was violated.
        constraint: Cow<'static, str>,
    },
}

impl Display for InvalidArgumentKind {
    /// Formats the invalid argument kind with structured detail fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::RangeOutOfBounds { axis, axis_len, start, end } => {
                write!(f, "range [{start}..{end}] out of bounds for axis {axis} (len={axis_len})")
            },
            Self::RangeStartAfterEnd { axis, start, end } => {
                write!(f, "range start ({start}) after end ({end}) at axis {axis}",)
            },
            Self::NumericOutOfRange { argument, domain, actual } => {
                write!(f, "`{argument}` out of range: {domain}, got {actual}")
            },
            Self::InvalidConfig { argument, constraint, actual } => {
                write!(f, "invalid config `{argument}`: {constraint}, got {actual}",)
            },
            Self::DuplicateOrEmpty { argument } => {
                write!(f, "duplicate or empty `{argument}`")
            },
            Self::OperationSpecific { argument, constraint } => {
                write!(f, "`{argument}`: {constraint}")
            },
        }
    }
}

// --- InvalidLayoutReason ----------------------------------------------------

/// Reason for `XenonError::InvalidLayout`.
///
/// Closed enum: each reason has program-matchable semantics.
///
/// Marked `#[non_exhaustive]` to allow new layout-validation reasons in
/// future minor versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidLayoutReason {
    /// `shape.checked_size()` overflowed `usize`.
    ShapeProductOverflow,

    /// Computed `max_offset` exceeds `storage_len`.
    AccessRangeExceedsStorage,

    /// Empty tensor metadata uses `offset > storage_len`.
    EmptyTensorOffsetExceedsStorage,

    /// A stride exceeds `isize::MAX`, so pointer `.add()` arithmetic
    /// cannot be proven valid.
    StrideExceedsIsizeMax,

    /// `(shape[axis] - 1) * stride[axis]` overflowed.
    StrideSpanOverflow,

    /// Accumulating the reachable access range overflowed.
    AccessRangeOverflow,

    /// Zero stride rejected specifically for ViewMut construction: the
    /// layout passes `validate_access_range` but contains a non-singleton
    /// axis with stride == 0. The caller can switch to a read-only view
    /// instead, which does accept broadcast zero strides.
    ZeroStrideRejectedForViewMut,

    /// Logical layout cannot be conservatively proven non-overlapping
    /// for the requested mutable access.
    AmbiguousOverlap,

    /// Owned raw-parts reconstruction requires `offset == 0`.
    OwnedRequiresZeroOffset,

    /// Owned raw-parts `len` does not equal `shape.checked_size()`.
    LenShapeMismatch,

    /// Owned raw-parts `cap` is smaller than `len`.
    CapacityBelowLen,

    /// Owned raw-parts allocator alignment is invalid for the element type.
    AlignmentInvalid,

    /// Owned raw-parts strides do not match canonical F-order strides.
    OwnedRequiresCanonicalFOrder,
}

impl Display for InvalidLayoutReason {
    /// Formats the layout validation reason as a short label.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeProductOverflow => write!(f, "shape product overflow"),
            Self::AccessRangeExceedsStorage => write!(f, "access range exceeds storage"),
            Self::EmptyTensorOffsetExceedsStorage => write!(f, "empty tensor offset exceeds storage"),
            Self::StrideExceedsIsizeMax => write!(f, "stride exceeds isize::MAX"),
            Self::StrideSpanOverflow => write!(f, "stride span overflow"),
            Self::AccessRangeOverflow => write!(f, "access range overflow"),
            Self::ZeroStrideRejectedForViewMut => write!(f, "zero stride rejected for ViewMut"),
            Self::AmbiguousOverlap => write!(f, "ambiguous overlap"),
            Self::OwnedRequiresZeroOffset => write!(f, "owned requires zero offset"),
            Self::LenShapeMismatch => write!(f, "len-shape mismatch"),
            Self::CapacityBelowLen => write!(f, "capacity below len"),
            Self::AlignmentInvalid => write!(f, "alignment invalid"),
            Self::OwnedRequiresCanonicalFOrder => write!(f, "owned requires canonical F-order"),
        }
    }
}

// --- StorageKindTag ---------------------------------------------------------

/// Tag identifying which storage kind is currently in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKindTag {
    /// Owned storage (heap-allocated, exclusive ownership).
    Owned,

    /// Immutable borrowed view.
    View,

    /// Mutable borrowed view.
    ViewMut,

    /// Reference-counted shared storage.
    Shared,
}

impl Display for StorageKindTag {
    /// Formats the storage kind as a short label.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned => write!(f, "owned"),
            Self::View => write!(f, "view"),
            Self::ViewMut => write!(f, "view mut"),
            Self::Shared => write!(f, "shared"),
        }
    }
}

// --- InvalidShapeKind -------------------------------------------------------

/// Kind for `XenonError::InvalidShape`.
///
/// Marked `#[non_exhaustive]` to allow new shape-validation kinds in future
/// minor versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidShapeKind {
    /// `shape.checked_size()` overflowed `usize`. Element-count fields
    /// are intentionally absent because no finite expected/actual
    /// counts can be expressed.
    ProductOverflow,

    /// Provided element count does not equal `shape.checked_size()`.
    ElementCountMismatch {
        /// Expected element count from `shape.checked_size()`.
        expected: usize,
        /// Actual element count provided.
        actual: usize,
    },
}

impl Display for InvalidShapeKind {
    /// Formats the invalid shape kind with structured detail fields.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProductOverflow => write!(f, "product overflow"),
            Self::ElementCountMismatch { expected, actual } => {
                write!(f, "element count mismatch: expected {expected}, got {actual}")
            },
        }
    }
}

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    /// Verify FFI auxiliary enums are constructable.
    #[test]
    fn test_ffi_aux_enums_construct() {
        let _ = FfiErrorCategory::NullPointer {
            argument: Cow::Borrowed("ptr"),
        };
        let _ = FfiBackend::RawParts;
        let _ = FfiBackend::Blas;
    }

    /// Verify Workspace auxiliary enums are constructable.
    #[test]
    fn test_workspace_aux_enums_construct() {
        let _ = WorkspaceErrorCategory::AllocFailed {
            size: 4096,
            align: 64,
        };
        let _ = WorkspaceErrorCategory::BorrowConflict {
            requested: WorkspaceBorrowKind::Exclusive,
            current: WorkspaceBorrowState::Shared,
        };
        let _ = WorkspaceBorrowState::SplitActive { count: 2 };
        let _ = TypedViewRejection::TypedByteLengthOverflow {
            count: usize::MAX,
            elem_size: 8,
        };
    }

    /// Verify Conversion / Argument / Layout / Storage enums are constructable.
    #[test]
    fn test_other_aux_enums_construct() {
        let _ = ConversionFailureReason::FloatToInteger;
        let _ = InvalidArgumentKind::RangeOutOfBounds {
            axis: 0,
            axis_len: 5,
            start: 3,
            end: 10,
        };
        let _ = InvalidLayoutReason::AccessRangeExceedsStorage;
        let _ = StorageKindTag::Owned;
        let _ = InvalidShapeKind::ElementCountMismatch {
            expected: 6,
            actual: 5,
        };
    }

    /// Verify auxiliary enums are Clone + PartialEq.
    #[test]
    fn test_aux_enums_clone_eq() {
        let a = FfiBackend::Blas;
        let b = a;
        assert_eq!(a, b);

        let a = StorageKindTag::ViewMut;
        let b = a;
        assert_eq!(a, b);

        let a = InvalidShapeKind::ProductOverflow;
        let b = a;
        assert_eq!(a, b);
    }

    /// Verify Debug formatting does not panic for any aux enum.
    #[test]
    fn test_aux_enums_debug_no_panic() {
        let cases: &[&dyn Debug] = &[
            &FfiErrorCategory::IntegerOverflow {
                value: usize::MAX,
                target_width_bits: 32,
            },
            &WorkspaceErrorCategory::GrowOverflow {
                current_capacity: 1024,
                additional: usize::MAX,
            },
            &ConversionFailureReason::NonZeroImaginaryPart,
            &InvalidArgumentKind::DuplicateOrEmpty {
                argument: Cow::Borrowed("axes"),
            },
            &InvalidLayoutReason::AmbiguousOverlap,
            &StorageKindTag::Shared,
        ];
        for c in cases {
            let _ = format!("{:?}", c);
        }
    }

    /// Verify all `WorkspaceErrorCategory` variants are constructable and
    /// carry structured fields.
    #[test]
    fn test_workspace_workspace_error_category() {
        // InvalidLayout
        let cat = WorkspaceErrorCategory::InvalidLayout { size: 0, align: 3 };
        assert!(format!("{cat:?}").contains("InvalidLayout"));

        // AllocFailed
        let cat = WorkspaceErrorCategory::AllocFailed {
            size: 1024,
            align: 64,
        };
        assert!(format!("{cat:?}").contains("AllocFailed"));

        // BorrowConflict
        let cat = WorkspaceErrorCategory::BorrowConflict {
            requested: WorkspaceBorrowKind::Exclusive,
            current: WorkspaceBorrowState::SplitActive { count: 2 },
        };
        assert!(format!("{cat:?}").contains("BorrowConflict"));
        assert!(format!("{cat:?}").contains("SplitActive"));

        // SplitOutOfBounds — field name MUST be `mid` (not `split_at`).
        let cat = WorkspaceErrorCategory::SplitOutOfBounds { mid: 42, len: 10 };
        assert!(format!("{cat:?}").contains("SplitOutOfBounds"));
        assert!(format!("{cat:?}").contains("mid: 42"));

        // GrowOverflow — field names MUST be `current_capacity` and `additional`.
        let cat = WorkspaceErrorCategory::GrowOverflow {
            current_capacity: usize::MAX,
            additional: 1,
        };
        assert!(format!("{cat:?}").contains("GrowOverflow"));
        assert!(format!("{cat:?}").contains("current_capacity"));
        assert!(format!("{cat:?}").contains("additional"));

        // TypedViewRejected
        let cat = WorkspaceErrorCategory::TypedViewRejected {
            detail: TypedViewRejection::ZeroSizedType,
        };
        assert!(format!("{cat:?}").contains("TypedViewRejected"));
    }

    /// Verify all 3 `TypedViewRejection` variants carry expected fields.
    #[test]
    fn test_typed_view_rejection_variants() {
        let r = TypedViewRejection::ZeroSizedType;
        assert!(format!("{r:?}").contains("ZeroSizedType"));

        let r = TypedViewRejection::AlignmentMismatch {
            required: 8,
            actual: 1,
        };
        assert!(format!("{r:?}").contains("AlignmentMismatch"));

        let r = TypedViewRejection::TypedByteLengthOverflow {
            count: usize::MAX,
            elem_size: 4,
        };
        assert!(format!("{r:?}").contains("TypedByteLengthOverflow"));
    }

    /// Verify `FfiErrorCategory` Display output for all 4 variants.
    #[test]
    fn test_ffi_error_category_display() {
        let s = format!("{}", FfiErrorCategory::NullPointer {
            argument: Cow::Borrowed("ptr")
        });
        assert!(s.contains("null pointer"));
        assert!(s.contains("ptr"));

        let s = format!("{}", FfiErrorCategory::InvalidRank {
            expected: 2,
            actual: 3
        });
        assert!(s.contains("invalid rank"));
        assert!(s.contains("expected 2"));
        assert!(s.contains("actual 3"));

        let s = format!("{}", FfiErrorCategory::BlasIncompatibleLayout {
            shape: vec![2, 3],
            strides: vec![1, 2]
        });
        assert!(s.contains("BLAS-incompatible"));
        assert!(s.contains("[2 × 3]"));
        assert!(s.contains("[1 × 2]"));

        let s = format!("{}", FfiErrorCategory::IntegerOverflow {
            value: 5000,
            target_width_bits: 32
        });
        assert!(s.contains("5000"));
        assert!(s.contains("i32"));
    }

    /// Verify `FfiBackend` Display output.
    #[test]
    fn test_ffi_backend_display() {
        assert_eq!(format!("{}", FfiBackend::RawParts), "raw parts");
        assert_eq!(format!("{}", FfiBackend::Blas), "BLAS");
    }

    /// Verify `WorkspaceErrorCategory` Display output for all 6 variants.
    #[test]
    fn test_workspace_error_category_display() {
        let s = format!("{}", WorkspaceErrorCategory::AllocFailed {
            size: 1024,
            align: 64
        });
        assert!(s.contains("allocation failed"));
        assert!(s.contains("1024"));
        assert!(s.contains("64"));

        let s = format!("{}", WorkspaceErrorCategory::InvalidLayout {
            size: 0,
            align: 1
        });
        assert!(s.contains("invalid layout"));

        let s = format!("{}", WorkspaceErrorCategory::BorrowConflict {
            requested: WorkspaceBorrowKind::Exclusive,
            current: WorkspaceBorrowState::Shared
        });
        assert!(s.contains("borrow conflict"));
        assert!(s.contains("Exclusive"));
        assert!(s.contains("Shared"));

        let s = format!("{}", WorkspaceErrorCategory::SplitOutOfBounds {
            mid: 5,
            len: 10
        });
        assert!(s.contains("split out of bounds"));
        assert!(s.contains("5"));
        assert!(s.contains("10"));

        let s = format!("{}", WorkspaceErrorCategory::GrowOverflow {
            current_capacity: 100,
            additional: 50
        });
        assert!(s.contains("grow overflow"));
        assert!(s.contains("100"));
        assert!(s.contains("50"));

        let s = format!("{}", WorkspaceErrorCategory::TypedViewRejected {
            detail: TypedViewRejection::ZeroSizedType
        });
        assert!(s.contains("typed view rejected"));
        assert!(s.contains("ZeroSizedType"));
    }

    /// Verify `WorkspaceBorrowKind` Display output.
    #[test]
    fn test_workspace_borrow_kind_display() {
        assert_eq!(format!("{}", WorkspaceBorrowKind::Shared), "shared");
        assert_eq!(format!("{}", WorkspaceBorrowKind::Exclusive), "exclusive");
        assert_eq!(format!("{}", WorkspaceBorrowKind::Split), "split");
    }

    /// Verify `WorkspaceBorrowState` Display output for all 4 variants.
    #[test]
    fn test_workspace_borrow_state_display() {
        assert_eq!(format!("{}", WorkspaceBorrowState::None), "none");
        assert_eq!(format!("{}", WorkspaceBorrowState::Shared), "shared");
        assert_eq!(format!("{}", WorkspaceBorrowState::Exclusive), "exclusive");
        let s = format!("{}", WorkspaceBorrowState::SplitActive { count: 3 });
        assert!(s.contains("split active"));
        assert!(s.contains("3"));
    }

    /// Verify `TypedViewRejection` Display output for all 3 variants.
    #[test]
    fn test_typed_view_rejection_display() {
        assert_eq!(
            format!("{}", TypedViewRejection::ZeroSizedType),
            "zero-sized type"
        );

        let s = format!("{}", TypedViewRejection::AlignmentMismatch {
            required: 8,
            actual: 1
        });
        assert!(s.contains("alignment mismatch"));
        assert!(s.contains("8"));
        assert!(s.contains("1"));

        let s = format!("{}", TypedViewRejection::TypedByteLengthOverflow {
            count: 1024,
            elem_size: 8
        });
        assert!(s.contains("byte length overflow"));
        assert!(s.contains("1024"));
        assert!(s.contains("8"));
    }

    /// Verify `ConversionFailureReason` Display output for all 5 variants.
    #[test]
    fn test_conversion_failure_reason_display() {
        assert_eq!(
            format!("{}", ConversionFailureReason::LossyIntegerNarrowing),
            "lossy integer narrowing"
        );
        assert_eq!(
            format!("{}", ConversionFailureReason::LossyFloatNarrowing),
            "lossy float narrowing"
        );
        assert_eq!(
            format!("{}", ConversionFailureReason::FloatToInteger),
            "float to integer"
        );
        assert_eq!(
            format!("{}", ConversionFailureReason::IntegerToFloatPrecisionLoss),
            "integer to float precision loss"
        );
        assert_eq!(
            format!("{}", ConversionFailureReason::NonZeroImaginaryPart),
            "non-zero imaginary part"
        );
    }

    /// Verify `InvalidArgumentKind` Display output for all 6 variants.
    #[test]
    fn test_invalid_argument_kind_display() {
        let s = format!("{}", InvalidArgumentKind::RangeOutOfBounds {
            axis: 0,
            axis_len: 5,
            start: 3,
            end: 10
        });
        assert!(s.contains("out of bounds"));
        assert!(s.contains("axis 0"));
        assert!(s.contains("5"));

        let s = format!("{}", InvalidArgumentKind::RangeStartAfterEnd {
            axis: 1,
            start: 5,
            end: 3
        });
        assert!(s.contains("start (5) after end (3)"));
        assert!(s.contains("axis 1"));

        let s = format!("{}", InvalidArgumentKind::NumericOutOfRange {
            argument: Cow::Borrowed("n"),
            domain: Cow::Borrowed(">= 0"),
            actual: Cow::Borrowed("-1")
        });
        assert!(s.contains("`n`"));
        assert!(s.contains(">= 0"));
        assert!(s.contains("-1"));

        let s = format!("{}", InvalidArgumentKind::InvalidConfig {
            argument: Cow::Borrowed("threshold"),
            constraint: Cow::Borrowed("> 0"),
            actual: Cow::Borrowed("0")
        });
        assert!(s.contains("invalid config"));
        assert!(s.contains("threshold"));

        let s = format!("{}", InvalidArgumentKind::DuplicateOrEmpty {
            argument: Cow::Borrowed("axes")
        });
        assert!(s.contains("duplicate or empty"));
        assert!(s.contains("axes"));

        let s = format!("{}", InvalidArgumentKind::OperationSpecific {
            argument: Cow::Borrowed("min"),
            constraint: Cow::Borrowed("min > max")
        });
        assert!(s.contains("`min`"));
        assert!(s.contains("min > max"));
    }

    /// Verify `InvalidLayoutReason` Display output for a representative
    /// sampling of all 13 variants.
    #[test]
    fn test_invalid_layout_reason_display() {
        assert_eq!(
            format!("{}", InvalidLayoutReason::ShapeProductOverflow),
            "shape product overflow"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::StrideExceedsIsizeMax),
            "stride exceeds isize::MAX"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::StrideSpanOverflow),
            "stride span overflow"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::AccessRangeOverflow),
            "access range overflow"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::ZeroStrideRejectedForViewMut),
            "zero stride rejected for ViewMut"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::AmbiguousOverlap),
            "ambiguous overlap"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::OwnedRequiresZeroOffset),
            "owned requires zero offset"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::LenShapeMismatch),
            "len-shape mismatch"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::CapacityBelowLen),
            "capacity below len"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::AlignmentInvalid),
            "alignment invalid"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::OwnedRequiresCanonicalFOrder),
            "owned requires canonical F-order"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::AccessRangeExceedsStorage),
            "access range exceeds storage"
        );
        assert_eq!(
            format!("{}", InvalidLayoutReason::EmptyTensorOffsetExceedsStorage),
            "empty tensor offset exceeds storage"
        );
    }

    /// Verify `InvalidShapeKind` Display output for both variants.
    #[test]
    fn test_invalid_shape_kind_display() {
        assert_eq!(
            format!("{}", InvalidShapeKind::ProductOverflow),
            "product overflow"
        );

        let s = format!("{}", InvalidShapeKind::ElementCountMismatch {
            expected: 6,
            actual: 5
        });
        assert!(s.contains("element count mismatch"));
        assert!(s.contains("expected 6"));
        assert!(s.contains("got 5"));
    }

    /// Verify `StorageKindTag` Display output.
    #[test]
    fn test_storage_kind_tag_display() {
        assert_eq!(format!("{}", StorageKindTag::Owned), "owned");
        assert_eq!(format!("{}", StorageKindTag::View), "view");
        assert_eq!(format!("{}", StorageKindTag::ViewMut), "view mut");
        assert_eq!(format!("{}", StorageKindTag::Shared), "shared");
    }
}
