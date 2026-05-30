//! Xenon error types.
//!
//! All recoverable errors are represented by the [`XenonError`] enum.
//! The crate uses `Result<T, XenonError>` (aliased as [`Result`]) for
//! all fallible operations.

use core::fmt::{self, Debug, Display, Formatter};
use std::borrow::Cow;
use std::error::Error;
use std::vec::Vec;

/// Helper for formatting `[usize]` shape/stride slices in error messages.
///
/// Output format: `[]`、`[5]`、`[2 × 3 × 4]` — NumPy style.
struct FmtShape<'a>(&'a [usize]);

impl<'a> Display for FmtShape<'a> {
    /// Formats the shape slice in NumPy-style bracket notation.
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

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
            Self::ZeroSizedType => write!(f, "zero-sized type"),
            Self::AlignmentMismatch { required, actual } => {
                write!(f, "alignment mismatch: required {required}, actual {actual}")
            },
            Self::TypedByteLengthOverflow { count, elem_size } => {
                write!(f, "byte length overflow: count={count}, elem_size={elem_size}")
            },
        }
    }
}

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
    use std::borrow::Cow;

    /// Helper for formatting optional values in error messages.
    ///
    /// Displays `<any>` if `None`, otherwise formats the value via `Display`.
    struct OrAny<T>(Option<T>);

    impl<T: Display> Display for OrAny<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            match &self.0 {
                Some(v) => write!(f, "{v}"),
                None => write!(f, "<any>"),
            }
        }
    }

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

    /// Verify empty `FmtShape` renders as `[]`.
    #[test]
    fn test_fmt_shape_empty() {
        assert_eq!(format!("{}", FmtShape(&[])), "[]");
    }

    /// Verify single-dimension `FmtShape` renders as `[N]`.
    #[test]
    fn test_fmt_shape_1d() {
        assert_eq!(format!("{}", FmtShape(&[5])), "[5]");
    }

    /// Verify multi-dimension `FmtShape` renders as `[a × b × c]`.
    #[test]
    fn test_fmt_shape_3d() {
        assert_eq!(format!("{}", FmtShape(&[2, 3, 4])), "[2 × 3 × 4]");
    }

    /// Verify `OrAny(Some(v))` renders the inner value via Display.
    #[test]
    fn test_or_any_some() {
        assert_eq!(format!("{}", OrAny(Some(42))), "42");
    }

    /// Verify `OrAny(None)` renders as `<any>`.
    #[test]
    fn test_or_any_none() {
        assert_eq!(format!("{}", OrAny(None::<i32>)), "<any>");
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
        fn assert_error<E: std::error::Error>(_: &E) {}
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
        let e: Box<dyn std::error::Error> = Box::new(XenonError::DimensionMismatch {
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

    // ── Workspace error category + constructor helper tests ──

    /// Verify all `WorkspaceErrorCategory` variants are constructable
    /// and carry structured fields.
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
}
