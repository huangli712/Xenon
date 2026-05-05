# 错误处理模块设计

> 文档编号: 26
> 模块目录: src/error.rs
> 任务阶段: Phase 1
> 前置文档: 01-architecture.md, 07-tensor.md, 21-type.md
> 需求参考: 需求说明书 §8、§12 - §28
> 范围声明: 范围内

---

## 1. 模块定位

本文档定义 Xenon 错误处理模块的设计方案。该模块负责提供全部公开 API 的统一错误模型，包括：

- 可恢复错误如何通过 `Result` 暴露
- 不可恢复错误如何通过 panic 报告
- 错误上下文字段的最小集合
- 类型转换、索引、形状、FFI 等场景的统一诊断规则

### 1.1 职责边界

| 职责           | 包含                                                   | 不包含                       |
| -------------- | ------------------------------------------------------ | ---------------------------- |
| 可恢复错误模型 | `XenonError` 枚举、`Result<T>` 别名、结构化诊断字段    | 日志系统、遥测               |
| panic 规范     | panic 分类规则、panic 消息模板、允许 panic 的边界定义  | 运行时 panic 拦截与恢复      |
| 错误上下文     | 每个变体的最小结构化字段定义、Display 格式化要求       | 错误上报平台、序列化错误模型 |
| 错误传播       | 并行路径中的 `Err` 与 panic 传播规则                   | 跨进程序列化                 |

### 1.2 设计原则

| 原则       | 体现                                                           |
| ---------- | -------------------------------------------------------------- |
| 统一性     | 所有公开 API 使用同一 `XenonError` 枚举，不按模块拆分错误类型  |
| 结构化诊断 | 每个错误变体携带结构化字段，不依赖纯字符串消息                 |
| 可恢复优先 | 安全公开 API 对非法输入返回 `Result`，仅特定边界允许 panic     |
| 最小分配   | 诊断字段的堆分配成本为可接受的工程开销，换取跨 API 一致性      |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                     |
| -------- | ------------------------------------------------------------------------ |
| 需求映射 | 需求说明书 §8, §12 - §28                                                 |
| 范围内   | 可恢复错误返回值、panic 分类、诊断字段、类型转换失败、索引失败、FFI 失败 |
| 范围外   | 自定义日志、第三方错误包装器、跨进程序列化                               |
| 非目标   | 通过 `panic` 替代本应可恢复的用户输入错误                                |

---

## 3. 文件位置

```
src/
└── error.rs       # XenonError, Result<T>, Display impl, helper enums
```

单文件设计：错误类型高度相关且全局共享，拆分反而增加跨模块耦合复杂度。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/error.rs
├── core::fmt                   # Display implementation
├── alloc::borrow::Cow          # Stable identifier-like strings (operation/backend names)
├── alloc::boxed::Box           # Source chain for Ffi / Workspace variants
└── alloc::vec::Vec             # Heap allocation for shape, index fields
```

**`ElementType` 不在本模块定义（v3.2.0 起回到 `crate::element`，详见 `03-element.md §5.1.1`）**。`error` 模块持有所有错误状态所需类型信息时使用 `&'static str`（值由 `Element::ELEMENT_TYPE_NAME` 关联常量提供），而**不**持有 `ElementType` 枚举字段。这保持 `error`（L0）严格不依赖 `element`（L2），同时通过字符串字面量让 error 仍能完整 `Display` 类型诊断信息——L0..L6 单向依赖严格成立。

### 4.2 类型级依赖

| 来源                  | 使用的类型/trait                                                              |
| --------------------- | ----------------------------------------------------------------------------- |
| `core::fmt`           | `Display`, `Formatter`, `fmt::Result`（`Display` 实现）                       |
| `alloc::borrow`       | `Cow<'static, str>`（仅用于 `operation` 等稳定标识符字符串）                  |
| `alloc::boxed`        | `Box<XenonError>`（`Ffi` / `Workspace` 变体的 `cause` 源链字段；递归枚举不能直接包含自身，`Box` 用固定大小指针打断无限大小递归） |
| `alloc::vec`          | `Vec<usize>`（`shape`、`attempted_index` 等字段）                             |

无对其他 crate 内模块的依赖。

### 4.3 依赖合法性

| 项目           | 说明                                              |
| -------------- | ------------------------------------------------- |
| 新增第三方依赖 | 无新增依赖                                        |
| 合法性结论     | 符合最小依赖限制；`error` 严格保持 L0 无内部依赖  |
| 替代方案       | 不适用；错误模型统一由 crate 内部类型与标准库承载 |

### 4.4 依赖方向声明

依赖方向：单向向上。`error.rs` 仅依赖标准库 / 核心库类型，不依赖任何 crate 内部模块——v3.2.0 起严格成立（v1.3.x 阶段曾引入对 `ElementType` 的 owner 关系并通过 re-export 维持下游路径稳定，v3.2.0 移除了这一耦合）。

`error` 严格保持 L0 单向依赖（v3.2.0 起）：

- 不向上引用 `element` / `complex` / 任何 L1+ 模块
- `XenonError::TypeConversion` 与 `AbiMismatchKind::ElementTypeMismatch` 中的类型字段使用 `&'static str` 而非 `ElementType` 枚举（值来自 `Element::ELEMENT_TYPE_NAME`，由元素侧统一控制；详见 `03-element.md v1.4.0 §5.1.1`）
- `ElementType` 枚举本身**不再**定义在本模块；其权威定义在 `crate::element`

---

## 5. 公共 API 设计

### 5.1 公共接口草案与关键签名

```rust,ignore
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

// NOTE (v3.2.0): `ElementType` is **not** defined here.
// Authoritative definition has moved (back) to `crate::element`
// (see `03-element.md §5.1.1`). The `error` module records type-tag
// information using `&'static str` instead — values come from
// `<A as Element>::ELEMENT_TYPE_NAME`. This keeps `error` (L0) free
// of any internal-module dependency while still giving `XenonError`
// fully formed `Display` output (the strings are simply written
// directly, without going through an enum).

/// Unified recoverable error type for all public Xenon APIs.
///
/// This enum is marked `#[non_exhaustive]`: downstream `match` expressions
/// MUST include a wildcard arm (`_ => ...`) and MUST NOT exhaustively pattern
/// against the listed variants. This lets future Xenon versions add new
/// top-level error categories (within the same SemVer major) without forcing
/// a breaking change on every downstream `match`.
///
/// **SemVer policy (1.x baseline)**:
///
/// What `#[non_exhaustive]` on this top-level enum DOES allow within a
/// minor version:
/// - Adding a new top-level variant (e.g. a 14th category) is non-breaking
///   because downstream `match` already requires a wildcard arm.
///
/// What `#[non_exhaustive]` on this top-level enum DOES NOT allow:
/// - Adding a new field to an existing struct-style variant (e.g. adding
///   `axis: Option<usize>` to `ShapeMismatch`) IS still a breaking change.
///   Top-level `#[non_exhaustive]` does NOT propagate into individual
///   variants. To allow non-breaking field growth on a specific variant,
///   that variant itself would need a per-variant `#[non_exhaustive]`
///   attribute (Rust supports this on struct-like variants); we do NOT
///   apply that to current variants because their field sets are already
///   stable.
///
/// What the inner sub-enums (`FfiErrorCategory`, `AbiMismatchKind`,
/// `InvalidLayoutReason`, `InvalidShapeKind`, `WorkspaceErrorCategory`,
/// `InvalidArgumentKind`) being `#[non_exhaustive]` DOES allow:
/// - Adding a new variant inside any of those sub-enums is non-breaking
///   (e.g. a new `FfiErrorCategory` for a future LAPACK backend, a new
///   `WorkspaceErrorCategory` for a future borrow-state, or a new
///   `InvalidArgumentKind` for a new operation family). This is
///   independent of the top-level enum: top-level `#[non_exhaustive]`
///   protects future categories at the `XenonError` level; sub-enum
///   `#[non_exhaustive]` protects future categories within an existing
///   `XenonError::Ffi { category, .. }` / `XenonError::Workspace
///   { category, .. }` / `XenonError::InvalidArgument { kind, .. }`
///   payload.
///
/// (Note: `WorkspaceErrorCategory` and `InvalidArgumentKind` were added
/// to the `#[non_exhaustive]` set in v3.3.1 alongside the policy text
/// correction; the original v3.3.0 release marked only the first four
/// sub-enums.)
///
/// `FfiBackend` and `StorageKindTag` are intentionally left as closed
/// enums (no `#[non_exhaustive]`); see their definitions for rationale
/// (closed sets by design).
///
/// **Always-breaking changes** (require major bump):
/// - Removing or renaming a top-level variant.
/// - Removing or renaming a sub-enum variant.
/// - Adding a new field to an existing variant (top-level OR sub-enum)
///   that is NOT itself `#[non_exhaustive]`.
/// - Changing the type or semantic meaning of an existing field.
///
/// See `01-architecture.md §13` decision 8 for the full rationale.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum XenonError {
    ShapeMismatch {
        operation: Cow<'static, str>,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },

    BroadcastError {
        operation: Cow<'static, str>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        attempted_target_shape: Option<Vec<usize>>,
        axis: Option<usize>,
    },

    /// Reserved for future BLAS/FFI layout validation.
    /// No public API constructs this error in the current version.
    /// The variant is exposed in the public enum for SemVer stability:
    /// future BLAS/FFI integrations may construct it without an enum
    /// breaking change. `required_layout` / `actual_layout` use a
    /// stable enum-like vocabulary (e.g., `"f-contiguous"`,
    /// `"non-contiguous"`, `"broadcast-view"`), not free-form messages.
    LayoutMismatch {
        operation: Cow<'static, str>,
        required_layout: Cow<'static, str>,
        actual_layout: Cow<'static, str>,
        shape: Vec<usize>,
    },

    InvalidLayout {
        operation: Cow<'static, str>,
        storage_kind: StorageKindTag,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        storage_len: usize,
        reason: InvalidLayoutReason,
    },

    InvalidAxis {
        operation: Cow<'static, str>,
        axis: usize,
        ndim: usize,
        shape: Vec<usize>,
    },

    InvalidShape {
        operation: Cow<'static, str>,
        shape: Vec<usize>,
        kind: InvalidShapeKind,
        offending_dim: Option<usize>,
    },

    DimensionMismatch {
        operation: Cow<'static, str>,
        expected: usize,
        actual: usize,
    },

    InvalidArgument {
        operation: Cow<'static, str>,
        kind: InvalidArgumentKind,
    },

    InvalidStorageMode {
        operation: Cow<'static, str>,
        expected: StorageKindTag,
        actual: StorageKindTag,
        shape: Option<Vec<usize>>,
        conversion: Option<StorageConversionKind>,
    },

    Ffi {
        operation: Cow<'static, str>,
        category: FfiErrorCategory,
        backend: FfiBackend,
        cause: Option<Box<XenonError>>,
    },

    Workspace {
        operation: Cow<'static, str>,
        category: WorkspaceErrorCategory,
        cause: Option<Box<XenonError>>,
    },

    IndexOutOfBounds {
        operation: Cow<'static, str>,
        attempted_index: Vec<usize>,
        axis: usize,
        shape: Vec<usize>,
    },

    TypeConversion {
        operation: Cow<'static, str>,
        // `&'static str` rather than `ElementType` (v3.2.0): value should
        // be the canonical name from `<A as Element>::ELEMENT_TYPE_NAME`
        // (see `03-element.md §5.1.1`). Storing the string keeps `error`
        // free of any dependency on `element`. Construction-site
        // ergonomic: pass `<A as Element>::ELEMENT_TYPE_NAME` or
        // `crate::element::element_type_name_of::<A>()`.
        source_type: &'static str,
        target_type: &'static str,
        reason: ConversionFailureReason,
        element_index: Option<usize>,
    },
}

/// FFI error category for `XenonError::Ffi`. All categories are
/// fully structured; no free-text fallback variant.
///
/// Marked `#[non_exhaustive]` to absorb future FFI-error categories without
/// breaking downstream `match` exhaustiveness within the same major version
/// (see `XenonError`'s SemVer policy doc above).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FfiErrorCategory {
    /// Caller passed a null raw pointer where a valid pointer was required.
    NullPointer { argument: Cow<'static, str> },
    /// Pointer alignment did not satisfy the type's alignment requirement.
    AlignmentMismatch { required: usize, actual: usize },
    /// Rank check failed (e.g., BLAS layer expects 2D matrix).
    InvalidRank { expected: usize, actual: usize },
    /// Layout cannot be expressed in the FFI ABI (e.g., non F-contiguous
    /// where BLAS layer requires column-major contiguous).
    BlasIncompatibleLayout {
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    /// `usize`-to-backend-integer conversion overflowed (e.g., to `i32` LDA).
    IntegerOverflow {
        value: usize,
        target_width_bits: u8,
    },
    /// ABI shape mismatch when reconstructing tensor from raw parts.
    AbiMismatch { detail: AbiMismatchKind },
    /// `from_raw_parts_mut` rejected a layout whose disjointness cannot
    /// be conservatively proven (overlap-rejected guard).
    OverlapRejected {
        shape: Vec<usize>,
        strides: Vec<usize>,
    },
    /// Foreign allocator metadata does not match Xenon's owned-tensor
    /// invariants (e.g., element type / capacity / alignment differ).
    ForeignAllocatorMismatch { detail: AbiMismatchKind },
}

/// Backend identifier for `XenonError::Ffi.backend`. Closed enum: any
/// future backend must extend this enum (SemVer-tracked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiBackend {
    /// Generic raw-parts FFI (no specific backend library).
    RawParts,
    /// BLAS-compatible export.
    Blas,
}

/// Detail kind for ABI mismatch / foreign allocator mismatch.
///
/// Marked `#[non_exhaustive]` to allow new ABI mismatch kinds in future
/// minor versions (e.g., when supporting additional FFI metadata
/// validation).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbiMismatchKind {
    /// Element type tag mismatch (e.g. C side claims `f32` but Rust side
    /// holds `f64`). Both fields are `&'static str` (v3.2.0): pass
    /// `<A as Element>::ELEMENT_TYPE_NAME` for the Rust side and the
    /// already-validated string from the FFI tag table for the C side.
    /// See `23-ffi.md §10` for FFI-side construction ergonomics.
    ElementTypeMismatch { expected: &'static str, actual: &'static str },
    CapacityMismatch { expected: usize, actual: usize },
    AlignmentMismatch { expected: usize, actual: usize },
    ShapeProductExceedsLen { product: usize, storage_len: usize },
    StridesRankMismatch { shape_ndim: usize, strides_ndim: usize },
}

impl core::fmt::Display for FfiErrorCategory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NullPointer { argument } =>
                write!(f, "null pointer for argument {argument}"),
            Self::AlignmentMismatch { required, actual } =>
                write!(f, "alignment mismatch: required {required}, actual {actual}"),
            Self::InvalidRank { expected, actual } =>
                write!(f, "invalid rank: expected {expected}, actual {actual}"),
            Self::BlasIncompatibleLayout { shape, strides } =>
                write!(f, "BLAS-incompatible layout: shape [{}], strides [{}]",
                    FmtShape(shape), FmtShape(strides)),
            Self::IntegerOverflow { value, target_width_bits } =>
                write!(f, "integer overflow: {value} does not fit in i{target_width_bits}"),
            Self::AbiMismatch { detail } =>
                write!(f, "ABI mismatch: {detail:?}"),
            Self::OverlapRejected { shape, strides } =>
                write!(f, "potentially overlapping layout rejected: shape [{}], strides [{}]",
                    FmtShape(shape), FmtShape(strides)),
            Self::ForeignAllocatorMismatch { detail } =>
                write!(f, "foreign allocator metadata mismatch: {detail:?}"),
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
    /// Underlying allocator returned failure (e.g., OOM / size==0 not allowed).
    AllocFailed { size: usize, align: usize },
    /// Layout request violates `Layout::from_size_align` rules.
    InvalidLayout { size: usize, align: usize },
    /// Borrow request conflicts with current borrow state.
    BorrowConflict {
        requested: WorkspaceBorrowKind,
        current: WorkspaceBorrowState,
    },
    /// `split_at_mut` mid index out of bounds for current view length.
    SplitOutOfBounds { mid: usize, len: usize },
    /// Internal split-count atomic invariant was violated (e.g., underflow
    /// or leak detected in debug).
    SplitCountInvariant { detail: Cow<'static, str> },
    /// Capacity grow overflow.
    ///
    /// `current_capacity` is the currently available byte length of the
    /// region or workspace; `additional` is the requested additional
    /// bytes (always in BYTES). For typed-view `count * size_of::<T>()`
    /// overflows where `count` is in element units (not bytes), use
    /// `TypedViewRejection::TypedByteLengthOverflow` instead — see
    /// `24-workspace.md §5.6` and v3.1.1 changelog.
    GrowOverflow { current_capacity: usize, additional: usize },
    /// Typed view request rejected (e.g., ZST not supported, range not
    /// aligned for `T`, count×size_of overflow — the last via
    /// `TypedViewRejection::TypedByteLengthOverflow`).
    TypedViewRejected { detail: TypedViewRejection },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceBorrowKind {
    Shared,
    Exclusive,
    Split,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceBorrowState {
    None,
    Shared,
    Exclusive,
    SplitActive { count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypedViewRejection {
    /// `T` is a zero-sized type; typed view of ZST is rejected.
    ZeroSizedType,
    /// Buffer base address does not satisfy `align_of::<T>()`.
    AlignmentMismatch { required: usize, actual: usize },
    /// `count.checked_mul(size_of::<T>())` overflowed `usize`. We cannot
    /// represent the requested byte length, so reusing `GrowOverflow`
    /// (which expects bytes) would produce a misleading diagnostic. Carry
    /// `count` (element units) and `elem_size` (bytes per `T`) instead.
    /// Added in v3.1.1 to replace the misuse of `GrowOverflow` previously
    /// done by `24-workspace.md §5.6` typed helpers.
    TypedByteLengthOverflow { count: usize, elem_size: usize },
    // Historical note: `LengthNotMultipleOfSize` was removed in v3.1.0.
    // `24-workspace.md §5.6` typed view API only allocates by element
    // `count` (computing `count * size_of::<T>()` internally), it does
    // NOT reinterpret an arbitrary byte length, so the variant had no
    // triggering call site. The new `TypedByteLengthOverflow` (above)
    // replaces it for the `count * size_of` overflow path.
}

impl core::fmt::Display for WorkspaceErrorCategory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AllocFailed { size, align } =>
                write!(f, "allocation failed (size={size}, align={align})"),
            Self::InvalidLayout { size, align } =>
                write!(f, "invalid layout (size={size}, align={align})"),
            Self::BorrowConflict { requested, current } =>
                write!(f, "borrow conflict: requested {requested:?}, current {current:?}"),
            Self::SplitOutOfBounds { mid, len } =>
                write!(f, "split out of bounds (mid={mid}, len={len})"),
            Self::SplitCountInvariant { detail } =>
                write!(f, "split-count invariant violated: {detail}"),
            Self::GrowOverflow { current_capacity, additional } =>
                write!(f, "grow overflow: capacity={current_capacity} + additional={additional}"),
            Self::TypedViewRejected { detail } =>
                write!(f, "typed view rejected: {detail:?}"),
        }
    }
}

/// Reason for `XenonError::InvalidLayout`. Closed enum: each reason
/// has program-matchable semantics.
///
/// **Single source of truth.** This enum is the only authoritative source
/// for layout-validation failure reasons across the crate. The `tensor`,
/// `layout`, `ffi`, and `construction` modules MUST construct
/// `XenonError::InvalidLayout { reason, .. }` using the variants defined
/// here and MUST NOT introduce locally-named variants outside this enum.
/// Adding a new layout-validation case requires extending this enum first.
///
/// Marked `#[non_exhaustive]` to allow new layout-validation reasons in
/// future minor versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidLayoutReason {
    /// `shape.checked_size()` overflowed `usize`.
    ShapeProductOverflow,
    /// `strides.len() != shape.len()`.
    StridesRankMismatch,
    /// Computed `max_offset` exceeds `storage_len`.
    AccessRangeExceedsStorage,
    /// Empty tensor metadata uses `offset > storage_len`.
    EmptyTensorOffsetExceedsStorage,
    /// Stride along an axis is not allowed for the current storage kind
    /// (e.g., negative stride; not representable as `usize`) or cannot
    /// be represented for pointer arithmetic.
    UnsupportedStride,
    /// A stride exceeds `isize::MAX`, so pointer `.add()` arithmetic
    /// cannot be proven valid.
    StrideExceedsIsizeMax,
    /// `(shape[axis] - 1) * stride[axis]` overflowed.
    StrideSpanOverflow,
    /// Accumulating the reachable access range overflowed.
    AccessRangeOverflow,
    /// Zero stride observed on a non-broadcast-view storage kind.
    UnexpectedZeroStride,
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
    /// Owned raw-parts reconstruction requires canonical F-order strides.
    OwnedRequiresCanonicalFOrder,
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
    ElementCountMismatch { expected: usize, actual: usize },
    /// Provided constructor input rank exceeds the static-rank support
    /// policy (`Ix0..=Ix6`) on a non-`try_from_dyn` path — for example,
    /// when an internal `IntoDimension` / `Tensor::from_shape_vec` pipeline
    /// receives a shape vector with `provided_ndim > 6`.
    ///
    /// **Excludes** `Dimension::try_from_dyn(IxDyn(...))` rank-mismatch
    /// path, which returns `XenonError::DimensionMismatch` (see
    /// `02-dimension.md §5.4` + `§8.3` in this doc); that path is a
    /// dimension-conversion mismatch, not a constructor rank-policy
    /// violation.
    RankExceedsStaticMax { provided_ndim: usize, max_ndim: usize },
}

/// Kind for `XenonError::InvalidArgument`.
///
/// Marked `#[non_exhaustive]` to allow new invalid-argument kinds in
/// future minor versions (one per operation family).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidArgumentKind {
    /// Range slice `start..end` is out of `[0, axis_len]`.
    RangeOutOfBounds {
        axis: usize,
        axis_len: usize,
        start: usize,
        end: usize,
    },
    /// Range slice has `start > end`.
    RangeStartAfterEnd { axis: usize, start: usize, end: usize },
    /// Numeric parameter outside its required domain (e.g., `alpha < 0`).
    NumericOutOfRange {
        argument: Cow<'static, str>,
        domain: Cow<'static, str>,
        actual: Cow<'static, str>,
    },
    /// Threshold / chunk-size / max-workers etc. configuration violated.
    InvalidConfig {
        argument: Cow<'static, str>,
        constraint: Cow<'static, str>,
        actual: Cow<'static, str>,
    },
    /// Unique-list / set parameter contained duplicate or empty groups.
    DuplicateOrEmpty { argument: Cow<'static, str> },
    /// Caller-provided shape parameter inconsistent with operation
    /// (e.g., `clip` min > max, `reshape` shape product mismatch but
    /// reported via `InvalidShape::ElementCountMismatch` instead — this
    /// variant covers operation-specific argument validation).
    OperationSpecific {
        argument: Cow<'static, str>,
        constraint: Cow<'static, str>,
    },
}

/// Storage kind tag, used in error variants to identify which storage
/// model the error refers to. Closed enum aligned with the public
/// `StorageKind` enum in `07-tensor.md` (Owned / View / ViewMut / Shared).
/// `Shared` is the user-facing name for the storage mode backed by
/// `ArcRepr<A>` (see `05-storage.md`); using `Shared` here keeps error
/// diagnostics consistent with the public `StorageKind::Shared` API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKindTag {
    Owned,
    View,
    ViewMut,
    Shared, // Backed by ArcRepr<A> in storage layer.
}

/// Storage conversion kind for `InvalidStorageMode.conversion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageConversionKind {
    ToOwned,
    IntoOwned,
    Transpose,
    SliceMut,
    BroadcastTo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionFailureReason {
    LossyIntegerNarrowing,
    LossyFloatNarrowing,
    FloatToInteger,
    IntegerToFloatPrecisionLoss,
    NonZeroImaginaryPart,
}

impl core::fmt::Display for ConversionFailureReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LossyIntegerNarrowing => write!(f, "lossy integer narrowing"),
            Self::LossyFloatNarrowing => write!(f, "lossy float narrowing"),
            Self::FloatToInteger => write!(f, "float to integer"),
            Self::IntegerToFloatPrecisionLoss => write!(f, "integer to float precision loss"),
            Self::NonZeroImaginaryPart => write!(f, "non-zero imaginary part"),
        }
    }
}

pub type Result<T> = core::result::Result<T, XenonError>;
```

- **空数组的语义**：空数组（任意维度上 size==0 的张量）是 Xenon 中的
  **合法输入**而非错误条件。所有公开 API 在“形状本身合法但其中一个或
  多个维度长度为 0”的情况下都不构造可恢复错误：例如 `sum()` 返回加法
  单位元 `A::zero()`、`unique()` 返回空 1D 张量、广播规则正常应用、
  `transpose()` 返回相同形状的空视图。`XenonError` 不定义
  `EmptyArray` 变体，公开 API 也不得在“仅因为形状包含 0”这一原因下
  返回 `Err`。
- “形状本身非法”仍然返回错误：`shape` 元素总数 `checked_size()` 溢出
  `usize`、`from_shape_vec` 提供的 `Vec<A>` 长度与
  `shape.checked_size()` 不一致、动态维度 rank 超过静态最大值等场景，
  返回 `XenonError::InvalidShape`，但其 `kind` 不会被诊断为“数组为空”，
  而是 `ProductOverflow` / `ElementCountMismatch` /
  `RankExceedsStaticMax` 中的某一种。`§8.3 边界测试`中的 `shape=[0, 3]`
  对应“合法空形状”，预期为 `Ok` 而非 `InvalidShape`（除非该 API 自身
  另有合法性约束，比如要求至少 1 个轴长大于 0；此时也以
  `InvalidShape::ElementCountMismatch` 表达，不以“空”为由）。
- `XenonError` 须实现 `std::error::Error` trait，提供 `source()` 方法
  用于链式错误追踪。
- `Ffi` 和 `Workspace` 变体可携带 `cause: Option<Box<XenonError>>`
  源链：当外层公开错误是“包装”自一个更底层的可恢复错误（例如
  `Workspace` 借用失败被 FFI 路径包装为 `Ffi` 错误）时，外层变体的
  `cause` 设为 `Some(Box::new(inner))`；`source()` 据此返回内层错误的
  `&dyn Error`。其他变体的 `source()` 仍返回 `None`（叶子错误）。
- 公开 API 统一使用 prelude 导出的 `crate::error::Result`（即
  `Result<T, XenonError>` 别名）作为返回类型。
- 所有可恢复错误直接以 `XenonError` 结构化变体返回，不使用模块内部
  错误类型。
- 每个变体携带适用的结构化字段，满足 `需求说明书 §27` 对公开诊断
  信息的要求。

```rust,ignore
/// `XenonError` implements `std::error::Error` for compatibility with
/// the standard error-handling ecosystem (`?`, `anyhow`, `thiserror`,
/// etc.). Variants that wrap another recoverable error expose it via
/// `source()`; the others are leaf errors with no nested source.
impl std::error::Error for XenonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ffi { cause: Some(inner), .. } => Some(&**inner),
            Self::Workspace { cause: Some(inner), .. } => Some(&**inner),
            _ => None,
        }
    }
}
```

`XenonError` 实现 `std::error::Error` 以兼容标准错误处理生态（`?`、
`anyhow`、`thiserror` 等）。仅 `Ffi` / `Workspace` 两个变体允许通过
`cause` 字段链式包装；其余变体均为叶子错误，`source()` 返回 `None`。
源链是 SemVer 兼容扩展点：未来若新的变体需要承载内部错误源，须以新增
变体而非改变现有变体语义的方式扩展。

### 5.2 可恢复错误与 panic 的边界

| 场景                              | 处理方式                                     | 说明                                  |
| --------------------------------- | -------------------------------------------- | ------------------------------------- |
| 形状不兼容 / 广播失败             | `Result::Err(XenonError)`                    | 运行时输入决定，可恢复                |
| 轴越界 / 参数非法 / FFI 前提失败  | `Result::Err(XenonError)`                    | 调用方可修正输入并重试                |
| `cast()` 有损或前提不满足         | `Result::Err(XenonError::TypeConversion(_))` | `需求说明书 §23` 强制要求             |
| 方法型索引失败                    | `Result::Err(XenonError::IndexOutOfBounds)`  | 需返回结构化索引上下文                |
| 有符号整数算术溢出 / 除以零       | panic                                        | 仅适用于 `i32` / `i64`，见需求说明书  |
| 有符号整数算术结果不可表示        | panic                                        | 仅适用于 `i32` / `i64`，见需求说明书  |
| `sqrt(negative)`                  | IEEE 754 返回 `NaN`，不得 panic              | `f32` / `f64` 数学域边界              |
| `ln(negative)`                    | IEEE 754 返回 `NaN`，不得 panic              | `f32` / `f64` 数学域边界              |
| `ln(0)`                           | IEEE 754 返回 `-Inf`，不得 panic             | `f32` / `f64` 数学域边界              |

### 5.3 安全 API 的 panic 边界

总原则： 
- 所有安全公开 API 对非法输入须返回可恢复错误（`Result`）。
- 仅 `unsafe` 函数的前提违反和内部 helper 可使用 panic。
- Xenon 当前稳定 API **不实现** `std::ops::Index` / `std::ops::IndexMut`（见 `01-architecture.md` 决策 7、`00-coding.md §8.1`）；方括号索引语法 `tensor[i]` 在当前版本不可编译，**不是**当前的 panic 边界。所有公开安全索引错误通过 `try_at()` / `try_at_mut()` 以 `Result::Err(XenonError::IndexOutOfBounds { .. })` 返回。

除下表外，其余安全公开 API 遇到错误条件时都必须返回 `Result<_, XenonError>`，不得以 panic 代替可恢复错误；即使是 FFI convenience helper，只要属于安全公开 API，也必须遵循这一规则。

| 类别                      | 允许 panic 的边界                        | 约束                                            |
| ------------------------- | ---------------------------------------- | ----------------------------------------------- |
| 需求明确定义的算术域边界  | `i32` / `i64` 的逐元素算术、归约、内积   | 溢出、除以零、结果不可表示时 panic              |
| internal / unsafe helper 边界 | private helper、`unsafe fn` 前提检查、未对外公开的 typed helper | 仅限实现内部或不安全前提；不得作为安全公开 API 的用户输入错误出口 |

### 5.4 公开 API 边界规则

| 边界位置               | 规则                          |
| ---------------------- | ----------------------------- |
| Public API return type | `Result<_, XenonError>`       |
| 错误构造方式           | 直接构造 `XenonError` 结构化变体，不经过中间错误类型映射 |

该表为公开错误边界的唯一基线。

### 5.5 类型转换错误规范

`cast()` 的错误模型须与 `21-type.md` 保持一致：

- `cast<B>(&self)` 返回 `Result<Tensor<B, D>, XenonError>`
- 任何被 `需求说明书 §23` 判定为有损的默认转换组合，都须返回
  `XenonError::TypeConversion`
- 仅当需求显式给出附加成功前提时，满足前提后才可成功
- `Complex -> Real` 不是编译期拒绝；当 `im == 0` 时允许继续转换，
  否则返回 `XenonError::TypeConversion { reason: NonZeroImaginaryPart, ... }`
- `bool` 不参与逐元素类型转换，因此不得用 `TypeConversion` 为 `bool`
  扩大支持范围
- `TypeConversion` 必须包含 `operation: Cow<'static, str>` 字段，记录
  触发转换的高层运算名（例如 `"cast"`、`"complex_to_real"`、
  `"infer_dtype_promotion"`），与其他错误变体保持字段一致性
- 源/目标类型字段使用 `&'static str`（v3.2.0 起）。值由 `Element::ELEMENT_TYPE_NAME`
  关联常量提供（详见 `03-element.md §5.1.1`），由元素侧统一控制，避免 error 模块
  反向依赖 element。**不**使用 `core::any::TypeId`：`TypeId` 是不透明哈希，无法满足
  "结构化诊断 + 可读 Display" 要求；**也不**直接持有 `ElementType` 枚举字段，因为这
  会让 error（L0）被迫依赖 element（L2）。使用 `&'static str` 同时获得：编译期确定的
  字符串字面量（零分配、零间接）+ 直接 Display + L0 单向依赖严格成立

类型转换失败统一通过 `XenonError::TypeConversion` 返回，其中字段为
公开字段，用户可直接通过模式匹配访问。

### 5.6 结构化上下文字段要求

所有错误变体都须带“错误类别 + 适用上下文”的结构化字段；不得使用纯
字符串消息字段（除 `operation`、`backend` 等稳定标识符以及枚举内的
有限自由文本载荷外）。

| 变体                  | 最小结构化字段                                                                |
| --------------------- | ----------------------------------------------------------------------------- |
| `ShapeMismatch`       | `operation`, `left_shape`, `right_shape`                                      |
| `BroadcastError`      | `operation`, `lhs_shape`, `rhs_shape`, `attempted_target_shape?`, `axis?`     |
| `LayoutMismatch`      | `operation`, `required_layout`, `actual_layout`, `shape`                      |
| `InvalidLayout`       | `operation`, `storage_kind`, `shape`, `strides`, `offset`, `storage_len`, `reason` |
| `InvalidAxis`         | `operation`, `axis`, `ndim`, `shape`                                          |
| `InvalidShape`        | `operation`, `shape`, `kind`, `offending_dim?`                                |
| `DimensionMismatch`   | `operation`, `expected`, `actual`                                             |
| `InvalidArgument`     | `operation`, `kind`（结构化子枚举 `InvalidArgumentKind`，每个变体内部携带其专属字段，例如 `RangeOutOfBounds` 必带 `axis/axis_len/start/end`；自 v3.3.1 起 `#[non_exhaustive]`——v3.3.0 设计意图，v3.3.1 补全属性） |
| `InvalidStorageMode`  | `operation`, `expected`, `actual`, `shape?`, `conversion?`                    |
| `Ffi`                 | `operation`, `category`（结构化子枚举 `FfiErrorCategory`，每个子类携带专属结构化负载；自 v3.3.0 起 `#[non_exhaustive]`）, `backend`, `cause?` |
| `Workspace`           | `operation`, `category`（结构化子枚举 `WorkspaceErrorCategory`，每个子类携带专属结构化负载；自 v3.3.1 起 `#[non_exhaustive]`——v3.3.0 设计意图，v3.3.1 补全属性）, `cause?` |
| `IndexOutOfBounds`    | `operation`, `attempted_index`, `axis`, `shape`；`attempted_index` 表示完整多维索引 tuple，`axis` 指出首个越界维度 |
| `TypeConversion`      | `operation`, `source_type`（`&'static str`，v3.2.0 起；值来自 `Element::ELEMENT_TYPE_NAME`）, `target_type`（同前）, `reason`, `element_index?` |

分配成本说明：`attempted_index: Vec<usize>`、`shape: Vec<usize>` 以及
若干 `InvalidArgumentKind` / `FfiErrorCategory` 子变体内的 `Vec<usize>`
字段会带来少量堆分配成本；这是当前版本可接受的诊断开销，用于换取跨
公开 API 的一致结构化上下文。

**字段命名约定（跨变体一致性规则）：** 各变体字段集合允许不同，但
**同义字段必须使用同名**，新增变体或扩展字段时须遵循以下规则；目的是
让结构化诊断在跨变体程序化处理时具备稳定语义。

| 字段名                                  | 类型                                | 含义                                                       | 备注                                                                       |
| --------------------------------------- | ----------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| `operation`                             | `Cow<'static, str>`                 | 触发错误的高层运算名（如 `"sum"`、`"reshape"`、`"slice"`） | 几乎所有变体都必须携带；以一致词汇表命名；仅作为稳定标识符使用             |
| `axis`                                  | `usize`                             | 单一相关维度索引                                           | 多个相关维度时使用 `axes: Vec<usize>`                                      |
| `ndim`                                  | `usize`                             | 张量秩                                                     | 与 `axis < ndim` 配套使用                                                  |
| `shape`                                 | `Vec<usize>`                        | 完整逻辑形状                                               | 字段名固定为 `shape`；二元运算用 `lhs_shape`/`rhs_shape` 或 `left_shape`/`right_shape`，不得混用其他前缀 |
| `lhs_shape` / `rhs_shape`               | `Vec<usize>`                        | 二元运算的左右操作数形状                                   | 二元广播/算术运算优先使用此对，区别于 `left_shape`/`right_shape` 用于纯形状对比的语义场景 |
| `expected` / `actual`                   | 类型与场景相关（结构化子枚举或 `usize`）| 期望值与实际值                                         | 简单二元对比模式；不混用 `required` / `provided` 等同义词                  |
| `kind`                                  | 结构化子枚举（`InvalidShapeKind` 自 v3.3.0、`InvalidArgumentKind` 自 v3.3.1 起 `#[non_exhaustive]`）| 变体内部细分（如 `InvalidShapeKind`、`InvalidArgumentKind`）| 替代以前的自由文本 `reason: Cow<str>`                              |
| `category`                              | 结构化子枚举（`FfiErrorCategory` 自 v3.3.0、`WorkspaceErrorCategory` 自 v3.3.1 起 `#[non_exhaustive]`）| 子分类（仅 `Ffi` / `Workspace` 使用）                  | 子枚举中各变体携带专属结构化字段                                           |
| `attempted_index`                       | `Vec<usize>`                        | 多维索引 tuple                                             | `IndexOutOfBounds` 等需要完整索引上下文的变体使用                          |
| `cause`                                 | `Option<Box<XenonError>>`           | 源链 inner error                                           | 仅 `Ffi` / `Workspace` 允许；通过 `Error::source()` 暴露                   |
| `backend`                               | `FfiBackend` 封闭枚举（design-intent closed，无 `#[non_exhaustive]`）| FFI 后端标识                                               | 仅 `Ffi` 使用                                                              |
| `storage_kind` / `expected` / `actual`  | `StorageKindTag` 封闭枚举（design-intent closed，无 `#[non_exhaustive]`）| 存储模式标签                                               | `InvalidLayout` / `InvalidStorageMode` 使用                                |
| `conversion`                            | `Option<StorageConversionKind>`     | 存储模式转换种类                                           | `InvalidStorageMode` 使用                                                  |
| `source_type` / `target_type`           | `&'static str`（v3.2.0 起）         | 元素类型名（如 `"f32"`、`"Complex<f64>"`）                 | `TypeConversion` 使用；值由 `Element::ELEMENT_TYPE_NAME` 提供，详见 `03-element.md §5.1.1` |

未来新增变体须复用上表名称与字段类型；如需新字段且语义新颖，须先在
本表中扩展再使用。**字符串字段（如 `operation`）只允许作为稳定标识
符**，不得作为可变诊断载体；所有可变细节须通过结构化子枚举的子变体表达。

### 5.7 Display 与 panic 信息要求

- Display 输出和 panic 文本都必须能让调用方定位问题来源；最少应包含
  操作名、错误类别以及适用上下文。
- panic 信息必须包含 `operation` + error kind + 至少一个关键上下文
  字段（如 `axis`、`shape`、`index`、类型等）。
- 对 `Option<Vec<usize>>` 等可选结构化字段，`Display` 实现必须做
  人性化格式化；`None` 统一显示为 `<any>`，不得直接打印 `Some(...)`
  / `None` 调试文本。
- 对 `TypeConversion` 的 `source_type` / `target_type`，Display 直接
  写出该 `&'static str` 值（v3.2.0 起；值已是人类可读名称，例如 `"f32"`、
  `"Complex<f64>"`，由 `Element::ELEMENT_TYPE_NAME` 关联常量保证）。
  禁止使用 `{:?}` 或 `TypeId` 风格的不透明哈希。
- 携带 `cause` 的 `Ffi` / `Workspace` 变体须在 Display 末尾追加
  `; caused by: <inner>` 片段，使单次格式化即可显示完整错误链；
  程序化遍历仍通过 `std::error::Error::source()`。

`Display` 的具体实现伪代码（含 `OrAny<T>` / `FmtShape<'_>` 等格式化
helper）属于内部实现细节，集中在 §6.1 给出，不在公共 API 章节复述。
公共契约只规定输出必须包含上表所述字段集合。

### 5.8 统一 panic 类别

除文档已提到的归约溢出外，以下不可恢复情形都须纳入统一 panic 规范：

- 逐元素整数算术溢出
- 整数除以零
- 结果不可表示（例如 `abs(i32::MIN)`、`i32::MIN / -1`）
- 整数内积的乘积或累加溢出

推荐 panic message 模板：`"Xenon: {operation} overflow for {type} at {context}"`

| panic 类别                   | 推荐消息示例                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------ |
| 逐元素加法溢出               | `"Xenon: add overflow for i32 at element_index=7"`                             |
| 归约溢出                     | `"Xenon: sum overflow for i64 at axis=1, output_index=3"`                      |
| 内积溢出                     | `"Xenon: dot overflow for i32 at lane=12"`                                     |
| internal/unsafe helper panic | `"Xenon: ptr_at precondition violation in internal helper at axis=1, index=8"` |

### 5.9 Good / Bad 对比式代码示例

类型转换：

```rust,ignore
// Good - cast is fallible and reports the failing element.
// Note: `ConvertTo<B>` is the `pub(crate) sealed` static dispatch trait
// owned by `convert/cast.rs` (see `21-type.md §6.1.ter`); it routes
// Tier-1 lossless pairs through `From` and Tier-2/Tier-3 pairs through
// `<A as CastTo<B>>::cast_to`. cast()'s public bound MUST be
// `A: ConvertTo<B>` (NOT `A: CastTo<B>`), because Tier-1 lossless type
// pairs (e.g. f32->f64, i32->i64, i32->f64) intentionally do NOT
// implement `CastTo` per the three-tier architecture.
pub fn cast<B: CastElement>(&self) -> Result<Tensor<B, D>, XenonError>
where
    A: ConvertTo<B>,
{
    let mut out = Vec::with_capacity(self.len());
    for (index, value) in self.iter().enumerate() {
        // ConvertTo::convert dispatches: Tier-0 identity returns Ok(self),
        // Tier-1 returns Ok(B::from(self)), Tier-2/Tier-3 forward to
        // <A as CastTo<B>>::cast_to(self) (which produces the structured
        // TypeConversion error fields below for lossy / dynamic pairs).
        let converted = ConvertTo::<B>::convert(*value).map_err(|err| {
            // Preserve the conversion error, enriching with element index
            // and the high-level operation name.
            match err {
                XenonError::TypeConversion {
                    source_type, target_type, reason, ..
                } => XenonError::TypeConversion {
                    operation: Cow::Borrowed("cast"),
                    source_type,
                    target_type,
                    reason,
                    element_index: Some(index),
                },
                other => other,
            }
        })?;
        out.push(converted);
    }
    // Internal helper, not a public API.
    Ok(Tensor::from_shape_vec_aligned(self.shape().clone(), out))
}

// Bad - silently saturating or truncating; also wrongly bounds on
// `A: CastTo<B>` (which would forbid Tier-1 lossless pairs since Tier-1
// type pairs intentionally have NO `CastTo` impl per `21-type.md §6.1.bis`
// — this Bad example would not even compile for `f32 -> f64`, and the
// non-fallible signature also drops the structured TypeConversion error).
pub fn cast_bad<B: Element>(&self) -> Tensor<B, D>
where
    A: CastTo<B>, // WRONG: should be ConvertTo<B>; CastTo lacks Tier-1 impls
{
    let out = self.iter().map(|value| value.cast_to_lossy()).collect();
    // Internal helper, not a public API.
    Tensor::from_shape_vec_aligned(self.shape().clone(), out)
}
```

整数算术溢出：

```rust,ignore
// Good - checked arithmetic with explicit panic message.
let value = lhs
    .checked_mul(rhs)
    .expect("Xenon: dot overflow for i32 at lhs_index=3, rhs_index=3");

// Bad - silent wrapping in release mode.
let value = lhs * rhs;
```

---

## 6. 内部实现设计

### 6.1 算法描述

错误构造与格式化的内部实现包含两部分：枚举构造模板 + Display
实现。下列伪代码描述其骨架；实际实现位于 `src/error.rs`，对外不暴露。

```
construct_xenon_error(operation, variant, context_fields):
    1. match variant to select the appropriate XenonError enum variant
    2. populate all mandatory structured fields from context_fields
       (including closed-enum sub-variants such as InvalidLayoutReason,
       InvalidShapeKind, InvalidArgumentKind, FfiErrorCategory,
       WorkspaceErrorCategory, AbiMismatchKind, TypedViewRejection,
       StorageKindTag, StorageConversionKind, FfiBackend; type-tag
       fields use `&'static str` from `Element::ELEMENT_TYPE_NAME`,
       see `03-element.md §5.1.1`)
    3. set optional fields (e.g., offending_dim, cause, conversion) to None
       when not applicable
    4. return XenonError::{variant} { ... }
```

格式化辅助类型（私有，不属于公开 API；放置于 `src/error.rs` 的私有
模块内部）：

```rust,ignore
// Internal Display helpers. Not part of the public API.
struct OrAny<T>(Option<T>);

impl<T: fmt::Display> fmt::Display for OrAny<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(v) => v.fmt(f),
            None => write!(f, "<any>"),
        }
    }
}

struct FmtShape<'a>(&'a [usize]);

impl<'a> fmt::Display for FmtShape<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", dim)?;
        }
        Ok(())
    }
}
```

`Display for XenonError` 主体的实现策略：

```
fmt_display(error, formatter):
    1. match error variant
    2. for each variant, format:
        - operation
        - error-kind label (variant name in human form)
        - all mandatory structured fields, using FmtShape for [usize]
          and OrAny<T> for Option<T>
        - for Ffi / Workspace: append "; caused by: <inner>" if cause is Some
        - for TypeConversion: write source_type / target_type as the
          stored &'static str directly (v3.2.0); no Debug formatting
    3. write formatted string to formatter
```

未来如需新增变体或子枚举，须同步：

1. 更新 `§5.6` 字段表
2. 在 `§6.1` 的构造模板中添加该变体所需结构化字段映射
3. 在 `Display for XenonError` 中添加该变体分支
4. 在 `§8.2` 单元测试清单中添加 Display / Clone / PartialEq 用例

### 6.2 安全性论证

本模块不涉及 `unsafe` 代码。错误类型的构造、`Display` 实现与 `Clone`/`PartialEq` 派生均为安全操作。

### 6.3 性能考量表

| 方面         | 设计决策                                                                      |
| ------------ | ----------------------------------------------------------------------------- |
| 分配开销     | `Vec<usize>` 字段（`shape`、`attempted_index`）在错误构造时产生少量堆分配     |
| 零分配路径   | 错误路径本身非热路径；少量分配换取结构化诊断上下文是可接受的工程权衡          |
| Clone 成本   | `XenonError` 的 `Clone` 会复制 `Vec` 和 `Cow` 字段；仅在测试或显式需要时调用  |
| PartialEq    | 用于测试断言；`source_type` / `target_type`（v3.2.0 起为 `&'static str`）的 `PartialEq` 比较是逐字节字符串比较；`Vec` 为逐元素比较 |

---

## 7. 实现任务拆分

### Wave 1: 基础类型

- [ ] **T1**: 创建 `src/error.rs` 骨架
  - 文件: `src/error.rs`
  - 内容: 模块声明、`XenonError` 枚举定义（所有变体及字段）
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 定义辅助枚举类型
  - 文件: `src/error.rs`
  - 内容: `FfiErrorCategory`、`WorkspaceErrorCategory`、`ConversionFailureReason` 枚举，`Result<T>` 类型别名
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: Display 实现

- [ ] **T3**: 实现 `fmt::Display` for `XenonError`
  - 文件: `src/error.rs`
  - 内容: 各变体的 Display 格式化实现，包含辅助类型 `OrAny<T>`、`FmtShape<'a>`
  - 测试: `test_display_*` 系列
  - 前置: T2
  - 预计: 15 min

### Wave 3: Error trait 与导出

- [ ] **T4**: 实现 `std::error::Error` for `XenonError`
  - 文件: `src/error.rs`
  - 内容: `Error` trait 实现；对 `Ffi` / `Workspace` 的 `cause: Some(_)` 返回内层错误，其余叶子变体返回 `None`
  - 测试: `test_error_trait_source_leaf_none`, `test_error_trait_source_chain_ffi_workspace`
  - 前置: T2
  - 预计: 5 min

- [ ] **T5**: 添加 prelude 导出
  - 文件: `src/error.rs`, `src/lib.rs`（或对应 prelude 文件）
  - 内容: 公开导出 `XenonError`、`Result`、辅助枚举
  - 测试: 编译通过，外部 crate 可通过 prelude 使用
  - 前置: T4
  - 预计: 5 min

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置               | 目的                                                       |
| ----------------------- | ------------------ | ---------------------------------------------------------- |
| 单元测试                | `src/error.rs` 内  | 验证 `XenonError` 各变体的 Display、Clone、PartialEq       |
| 集成测试                | 集成测试目录       | 验证跨模块 API 的错误映射正确性                            |
| 边界测试                | 与集成测试配套     | 空形状、非法轴、越界索引、复数虚部非零、整数极值、NaN/Inf  |
| panic 测试              | 集成测试目录       | 验证逐元素整数溢出、除以零、`abs(MIN)`、dot overflow       |
| 并行测试                | 集成测试目录       | 验证 `Err` 与 panic 在并行路径中的传播一致性               |
| Feature gate / 配置测试 | 配置矩阵           | 验证可选 SIMD/并行路径与标量路径的错误类别一致             |
| 类型边界 / 编译期测试   | 编译期测试框架     | 验证子枚举（`ConversionFailureReason` / `FfiErrorCategory` / `WorkspaceErrorCategory` / `AbiMismatchKind` / `TypedViewRejection` 等）的 `match` 完备性；`source_type` / `target_type` 为 `&'static str`，无需 const 上下文枚举验证 |

### 8.2 单元测试清单

| 测试函数                                       | 测试内容                                        | 优先级 |
| ---------------------------------------------- | ----------------------------------------------- | ------ |
| `test_cast_lossy_returns_type_conversion`      | 有损转换返回 `TypeConversion`                   | 高     |
| `test_cast_reports_element_index`              | 转换失败包含 `element_index`                    | 高     |
| `test_complex_to_real_requires_zero_imag`      | 复数转实数的附加成功前提                        | 高     |
| `test_invalid_argument_has_structured_context` | `InvalidArgument` 不再只有 message              | 高     |
| `test_invalid_shape_reports_dimension_context` | `InvalidShape` 包含维度/元素数上下文            | 高     |
| `test_index_error_reports_axis_and_shape`      | 索引错误包含 `attempted_index`、`axis`、`shape` | 高     |
| `test_integer_division_by_zero_panics`         | 除以零走统一 panic                              | 高     |
| `test_dot_overflow_panics`                     | 内积溢出走统一 panic                            | 高     |
| `test_display_shape_mismatch`                  | `ShapeMismatch` 的 Display 输出格式             | 中     |
| `test_display_option_fields_render_any`        | `None` 字段显示为 `<any>`                       | 中     |
| `test_error_trait_source_leaf_none`            | 叶子变体的 `source()` 返回 `None`               | 中     |
| `test_error_trait_source_chain_ffi_workspace`  | `Ffi { cause: Some(_) }` 的 `source()` 返回内层 | 高     |
| `test_type_conversion_uses_element_type_name`  | `TypeConversion` 的源/目标字段是 `&'static str`，值与 `Element::ELEMENT_TYPE_NAME` 一致（如 `"f32"`、`"Complex<f64>"`） | 高     |
| `test_type_conversion_carries_operation`       | `TypeConversion` 的 `operation` 必须非空        | 高     |
| `test_clone_eq_roundtrip`                      | `Clone` + `PartialEq` 往返一致                  | 中     |

### 8.3 边界测试场景

| 场景                             | 预期行为                                                                                                  |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 空形状 `shape=[0, 3]`            | 合法输入；构造成功，归约/广播/`unique` 等返回单位元或空张量；不返回 `InvalidShape`                         |
| 形状乘积溢出 `shape=[usize::MAX,2]` | 返回 `InvalidShape { kind: ProductOverflow, .. }`                                                       |
| 元素数不匹配 `shape=[2,3], data.len()=5` | 返回 `InvalidShape { kind: ElementCountMismatch { expected: 6, actual: 5 }, .. }`                  |
| rank 超静态最大 `IxDyn(7) -> Ix6` 转换 | 返回 `DimensionMismatch { operation: "Dimension::try_from_dyn", expected: 6, actual: 7, .. }`，由 `02-dimension.md §5.4 try_from_dyn` 提供（rank-mismatch 路径属维度不匹配，不属 InvalidShape） |
| 静态 rank 张量构造时输入 rank 超出该维度类型最大值（非 `IxDyn`→静态转换路径，例如内部用 `IntoDimension` 构造 `Ix6` 时遇到 `provided_ndim > 6`） | 返回 `InvalidShape { kind: RankExceedsStaticMax { provided_ndim, max_ndim }, .. }`           |
| 非法轴 `axis=5, ndim=2`          | 返回 `InvalidAxis` 结构化错误                                                                              |
| 越界索引 `index=[9], shape=[4]`  | 返回 `IndexOutOfBounds` 结构化错误                                                                         |
| 复数虚部非零 `Complex(1, 2)`     | 转换为实数类型返回 `TypeConversion { reason: NonZeroImaginaryPart, source_type: "Complex<f64>", target_type: "f64", .. }` |
| 整数极值 `i32::MIN`              | `abs(i32::MIN)` 走 panic                                                                                   |
| NaN/Inf 转换                     | `f64::NaN` → `i32` 返回 `TypeConversion { reason: FloatToInteger, source_type: "f64", target_type: "i32", .. }` |
| FFI 空指针                       | 返回 `Ffi { category: NullPointer { argument: "ptr" }, backend: RawParts, cause: None, .. }`              |
| FFI 包装 workspace 错误          | 返回 `Ffi { category: ..., cause: Some(Box::new(XenonError::Workspace { .. })), .. }`，`Error::source()` 返回内层 |

### 8.4 Feature gate / 配置测试

| 配置      | 验证点                                         |
| --------- | ---------------------------------------------- |
| 默认配置  | SIMD/并行关闭时错误类别与结构化字段一致        |
| 启用 SIMD | SIMD 路径错误类别与标量路径相同                |
| 启用并行  | 并行路径错误传播与串行路径一致，不静默吞掉错误 |

### 8.5 评审要求

- 任何新增公开 API 都必须明确写出"返回 `Result` 还是 panic"的裁决
- 任何新增错误变体都必须说明结构化字段，不得只增加 `message: &'static str`
- 任何新增类型转换组合都必须同时更新 `21-type.md` 与本规范中的错误路径说明

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向    | 对方模块                       | 接口/类型                       | 约定                              |
| ------- | ------------------------------ | ------------------------------- | --------------------------------- |
| 被消费  | `tensor` / `shape` / `matrix`  | `XenonError::ShapeMismatch`     | 非广播的双输入形状冲突时构造并返回 |
| 被消费  | `index`                        | `XenonError::IndexOutOfBounds`  | 方法型索引越界时构造并返回        |
| 被消费  | `broadcast` / `math`           | `XenonError::BroadcastError`    | 广播不兼容时构造并返回            |
| 被消费  | `reduction`                    | `XenonError::InvalidAxis`       | 轴越界时构造并返回；溢出走 panic  |
| 被消费  | `convert`                      | `XenonError::TypeConversion`    | 有损转换失败时构造并返回          |
| 被消费  | `ffi`                          | `XenonError::Ffi`               | FFI 前提不满足时构造并返回        |
| 被消费  | `tensor` / `ffi`               | `XenonError::InvalidLayout`     | 元数据校验失败时构造并返回        |
| 被消费  | `index` / `math` / `overload` / `parallel` | `XenonError::InvalidArgument`   | 参数非法时构造并返回              |
| 被消费  | `construction` / `math` / `parallel` | `XenonError::InvalidShape` | 形状/长度不匹配时构造并返回       |
| 被消费  | `dimension` / `parallel` / `index` / `ffi` | `XenonError::DimensionMismatch` | 维度不匹配时构造并返回 |
| 被消费  | `storage` / `utility`          | `XenonError::InvalidStorageMode`| 存储模式不支持时构造并返回        |
| 被消费  | `workspace`                    | `XenonError::Workspace`         | 工作区分配/借用/分割失败时构造并返回 |
| 被消费  | 所有模块                       | `Result<T>`                     | 公开 API 返回类型统一使用此别名   |

### 9.2 数据流描述

```
Caller invokes public API (e.g., tensor.broadcast_to(shape))
    │
    ├── API validates input (shape, axis, type, layout, ...)
    │       │
    │       ├── Valid → proceed with computation → return Ok(result)
    │       │
    │       └── Invalid → construct XenonError::{Variant} with structured context
    │               │
    │               └── return Err(XenonError)
    │
    └── For panic-bound operations (integer overflow in checked arithmetic, internal/unsafe helper precondition violations, ...)
        // Note: Xenon does NOT implement std::ops::Index/IndexMut for tensors;
        // user-facing indexing is via fallible methods (see 17-indexing.md §5),
        // so there is no "Index syntax sugar" panic path on the public API.
            │
            ├── Triggered → panic with formatted message
            │       "Xenon: {operation} overflow for {type} at {context}"
            │
            └── In parallel path (rayon)
                    │
                    ├── Err → propagate to join point as soon as detectable
                    └── panic → propagate to join point, never silently swallow
```

### 9.3 生命周期与所有权约定

`XenonError` 为 `Clone` 类型，错误构造时所有 `Cow<'static, str>`
字段使用 `'static` 生命周期，不借用调用方的临时数据。`Vec<usize>`
字段在构造时独立拥有，错误可安全地跨线程传递和存储。

`Ffi` / `Workspace` 变体的 `cause: Option<Box<XenonError>>` 字段在
构造时通过 `Box::new(inner)` 装箱，所有权完全归属外层错误；克隆外层
错误时会递归 `Clone` 内层错误并重新装箱，相等性比较递归到内层。
源链深度无人为限制，但实际构造路径不会产生超过 2-3 层的嵌套
（公开 API 不会自身递归包装）。

---

## 10. 错误处理与语义边界

| 主题              | 需要说明的内容                                                    |
| ----------------- | ----------------------------------------------------------------- |
| Recoverable error | 本模块定义可恢复错误的统一类型，错误构造本身不会失败（infallible）|
| Panic             | 本模块不直接触发 panic；各消费模块按 panic 边界表（§5.2）触发     |
| 路径一致性        | 标量 / SIMD / 并行路径须保持相同错误类别和结构化字段              |
| 容差边界          | 不适用；错误类型不涉及浮点数值计算                                |

### 10.1 数学函数定义域边界

| 场景             | `f32` / `f64` 行为 | 约束来源           |
| ---------------- | ------------------ | ------------------ |
| `sqrt(negative)` | 返回 `NaN`         | `需求说明书 §28.3` |
| `ln(negative)`   | 返回 `NaN`         | `需求说明书 §28.3` |
| `ln(0)`          | 返回 `-Inf`        | `需求说明书 §28.3` |

这些情形遵循 IEEE 754 语义，属于数值结果边界，不属于 panic 边界。

### 10.2 并行路径与资源释放

- 并行路径中的 `Err(XenonError)` 须尽快向上传播，不得延后为"全部 worker 完成后再统一检查"
- 并行路径中的 panic 不得被吞掉或伪装为成功结果
- 所有资源释放逻辑不得再触发 panic；在 `panic = abort` 环境下允许进程级终止带来资源不回收

对 `rayon` 上下文中的"立即传播"，本文采用工程化解释：

- 任一 worker 首次观察到 panic 或 `Err` 后，须终止该 worker 的当前执行路径并向 join 点报告失败
- 其他 worker 可能在 join 检测到失败前完成自己已经领取的当前 chunk；这是 `rayon` work-stealing 调度的实际限制
- 因此 `需求说明书 §27` 中的"立即"含义是"as soon as practically detectable"，而不是"所有线程瞬时同步中止"

### 10.3 受影响模块

| 模块/能力            | 影响内容                                            |
| -------------------- | --------------------------------------------------- |
| `tensor` / `shape`   | 形状校验、布局前提、元素总数校验                    |
| `index`              | 越界索引、按轴索引、切片边界诊断                    |
| `broadcast` / `math` | 广播失败、形状不兼容、参数非法                      |
| `reduction`          | 非法轴、空输入单位元语义、整数溢出 panic            |
| `convert`            | 类型转换失败的元素索引定位                          |
| `ffi`                | FFI 前提失败与后端约束诊断                          |
| `parallel`           | panic / `Err` 的尽快传播，不得静默吞掉              |
| `storage` / `utility`| 存储模式不支持时返回 `InvalidStorageMode`           |
| `workspace`          | 工作区分配失败、布局非法、借用冲突、分割越界        |
| `dimension`          | 静态/动态维度转换不匹配时返回 `DimensionMismatch`   |
| `construction`       | 构造时形状/长度不匹配返回 `InvalidShape`            |

---

## 11. 设计决策记录

### 决策 1：统一 `XenonError` 枚举而非模块独立错误类型

| 属性     | 值                                                                              |
| -------- | ------------------------------------------------------------------------------- |
| 决策     | 所有公开 API 使用单一 `XenonError` 枚举，不按模块拆分为独立错误类型             |
| 理由     | 避免调用方为不同模块维护不同的错误处理逻辑；结构化字段可按变体精确匹配          |
| 替代方案 | 每个模块定义自己的 `XxxError` 枚举 — 放弃，增加调用方负担且难以跨模块统一诊断   |
| 替代方案 | 使用 `anyhow` / `eyre` 等动态错误类型 — 放弃，违反最小依赖约束且损失结构化匹配  |

### 决策 2：panic 边界的严格划分

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | 安全公开 API 仅在语言级语法边界和需求明确定义的算术域边界允许 panic              |
| 理由     | 需求说明书 §27 要求可恢复错误以返回值形式报告；宽泛的 panic 会破坏调用方的错误恢复能力 |
| 替代方案 | 对所有非法输入统一 panic — 放弃，违反可恢复优先原则                              |
| 替代方案 | 实现 `std::ops::Index` 把 `tensor[i]` 越界 panic 化 — 放弃，与可恢复优先原则冲突；当前稳定 API 直接不实现 `Index` / `IndexMut`，公开安全索引错误经由 `try_at()` / `try_at_mut()` 以 `Result` 返回 |

### 决策 3：结构化字段而非纯字符串消息

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | 每个错误变体携带结构化字段（`operation`、`shape`、`axis` 等），不用 `message: String` |
| 理由     | 结构化字段允许调用方按程序逻辑匹配和处理错误，满足 `需求说明书 §27` 对诊断信息的要求 |
| 替代方案 | 仅提供 `String` 消息 — 放弃，无法程序化匹配错误类别                              |
| 替代方案 | 使用 `thiserror` 派生 — 放弃，违反最小依赖约束                                   |

### 决策 4：FFI / Workspace 子分类完全结构化

| 属性     | 值                                                                                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `FfiErrorCategory` / `WorkspaceErrorCategory` 改为携带结构化负载的 closed enum，覆盖 NullPointer / AlignmentMismatch / AbiMismatch / OverlapRejected / ForeignAllocatorMismatch / BorrowConflict / SplitCountInvariant / GrowOverflow / TypedViewRejected 等具体子类 |
| 理由     | 早期评审指出：原 `Ffi { backend, precondition, actual }` 三个 `Cow<str>` 字段把关键诊断稳定为自由文本，`InvalidRank/BlasIncompatibleLayout/IntegerOverflow` 三类无法覆盖 raw-parts FFI 常见错误源；workspace 的 `AllocFailed/InvalidLayout/AlreadyBorrowed（旧模型历史名称）/SplitOutOfBounds` 同样过粗 |
| 替代方案 | 维持粗粒度子枚举 + 自由文本 — 放弃，违反“结构化诊断不依赖纯字符串消息”原则                                                                                         |
| 替代方案 | 把所有 FFI/workspace 错误打平为一级变体 — 放弃，会让 `XenonError` 顶层变体爆炸性增长且失去“按子系统聚类”的程序化匹配能力                                            |

### 决策 5：仅 `Ffi` / `Workspace` 引入源链

| 属性     | 值                                                                                                                                            |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `Ffi` / `Workspace` 携带 `cause: Option<Box<XenonError>>`；其他变体保持叶子错误。`Error::source()` 据此返回内层；其余变体返回 `None`           |
| 理由     | 前序评审 B6.c：原 `source()` 始终返回 `None` 已经封死了链式追踪的能力；FFI 包装 workspace 错误、workspace allocator 错误等真实场景需要源链支持 |
| 替代方案 | 全部变体都加 `cause` — 放弃，绝大多数变体本身就是叶子，统一加字段会增加构造复杂度与无意义 None                                                  |
| 替代方案 | 用外部 `anyhow` / `Box<dyn Error>` 包装 — 放弃，违反最小依赖约束                                                                              |

### 决策 6：`TypeConversion` 用类型名 `&'static str` 替换 `TypeId`，并补 `operation`

| 属性     | 值                                                                                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `TypeConversion` 字段为 `{ operation, source_type: &'static str, target_type: &'static str, reason, element_index? }`（v3.2.0 起；v1.3.0–v3.1.x 曾使用 `ElementType`，详见决策回滚说明）。`source_type` / `target_type` 取值来自 `Element::ELEMENT_TYPE_NAME` 关联常量（`03-element.md §5.1.1`） |
| 理由     | Round-8 评审 H-R8 / C23：`TypeId` 是不透明哈希，无法满足结构化诊断 + 可读 Display；同时补齐 `operation` 字段消除"几乎所有变体都必须携带 operation 但 TypeConversion 例外"的不一致。`&'static str` 取代 `ElementType` 枚举（v3.2.0）：避免 error 反向依赖 element，让 L0..L6 单向依赖严格成立，同时保留可读 Display；具体类型枚举仍由 `crate::element` 拥有，结构化匹配可由调用方按需 `match` 字符串字面量（受支持的全集是固定的封闭名称集合） |
| 替代方案 | 保留 `TypeId` + 在 Display 时反查类型名 — 放弃，反查机制不存在且 TypeId 不可程序化匹配封闭元素集合 |
| 替代方案 | 字段保持 `ElementType` 枚举 — 放弃（v3.2.0 反转）：会让 error 模块持有 element 模块定义的类型，破坏 L0 单向依赖（即便通过 re-export 隐藏耦合，链路上仍是 error → element） |
| 替代方案 | 自由文本字符串 `Cow<'static, str>` — 放弃：构造点缺乏统一来源会导致拼写不一致；本设计要求值必须来自 `Element::ELEMENT_TYPE_NAME` 关联常量集中管理 |

---

## 12. 性能考量

- 错误构造：O(k)，其中 k 为 `Vec<usize>` 字段中的元素总数（通常为 ndim，即 ≤ 7）
- `Clone`：与构造相同的分配开销
- `Display`：O(n) 格式化，n 为输出字符串长度

| 场景           | 行为                                                     |
| -------------- | -------------------------------------------------------- |
| 正常路径       | 错误类型不被构造，零开销                                 |
| 错误路径       | 少量堆分配（`Vec<usize>` + `Cow`），为可接受的诊断开销   |
| 热路径影响     | 无；错误构造仅在异常路径触发                             |
| 测试路径开销   | `Clone` + `PartialEq` 用于断言；非生产路径               |

---

## 13. 平台与工程约束

| 约束       | 需要说明的内容                                                          |
| ---------- | ----------------------------------------------------------------------- |
| `std` only | 方案依赖 `std`；需求说明书 §1.3 明确仅支持 `std` 环境，不讨论 `no_std`  |
| MSRV       | Rust 1.85+                                                              |
| 单 crate   | 保持单 crate 边界；错误类型定义在 crate 内部，不引入额外 crate          |
| SemVer     | 新增错误变体或修改诊断字段属于公开 API 兼容性变更，须遵循 SemVer        |
| 最小依赖   | 不引入额外第三方依赖；错误模型由标准库与 crate 内部类型承载             |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.0.6 | 2026-04-10 |
| 1.1.0 | 2026-04-14 |
| 1.1.1 | 2026-04-14 |
| 1.1.2 | 2026-04-14 |
| 1.1.3 | 2026-04-15 |
| 1.1.4 | 2026-04-15 |
| 1.1.5 | 2026-04-16 |
| 2.0.0 | 2026-04-22 |
| 3.0.0 | 2026-05-02 |
| 3.0.1 | 2026-05-03 |
| 3.1.0 | 2026-05-03 |
| 3.1.1 | 2026-05-03 |
| 3.2.0 | 2026-05-03 |
| 3.3.0 | 2026-05-05 |
| 3.3.1 | 2026-05-05 |
| 3.3.2 | 2026-05-05 |

### v3.3.2 (2026-05-05) — patch fix: SemVer policy doc 完整性 + §5.6 v3.3.0/v3.3.1 起始版本辨识

- 第三轮重审专家发现 v3.3.1 修复存在两处不完整：
  - **`XenonError` doc comment SemVer policy 段**：sub-enum 列表只列了 4 个（`FfiErrorCategory` / `AbiMismatchKind` / `InvalidLayoutReason` / `InvalidShapeKind`），遗漏了 v3.3.1 补加 `#[non_exhaustive]` 的 `WorkspaceErrorCategory` 和 `InvalidArgumentKind`，导致 doc 描述与代码属性不一致。
  - **§5.6 表格**：把 `WorkspaceErrorCategory` / `InvalidArgumentKind` 标为"自 v3.3.0 起 `#[non_exhaustive]`"，但实际是 v3.3.1 补全；起始版本溯源不准确。
- 修复：
  - SemVer policy 段 inner sub-enum 列表补全为 6 个（含 `WorkspaceErrorCategory` / `InvalidArgumentKind`），新增括号注：v3.3.0 仅标了前 4 个，v3.3.1 补全后 2 个。
  - §5.6 表格逐行精确标注：`InvalidArgumentKind` / `WorkspaceErrorCategory` 改为"自 v3.3.1 起"；混合行（`kind` / `category`）改为分别标注每个子枚举的起始版本（前者 `InvalidShapeKind` 自 v3.3.0、`InvalidArgumentKind` 自 v3.3.1；后者 `FfiErrorCategory` 自 v3.3.0、`WorkspaceErrorCategory` 自 v3.3.1）。
  - v3.3.1 changlog §5.6 段措辞同步精确化。
- 协同：纯 doc 完整性修订；属性集合、字段集合、变体集合均无变化；不影响任何引用 26-error 的下游文档（这些文档当前 pin v3.3.1 的描述仍准确）。

### v3.3.1 (2026-05-05) — patch fix: SemVer policy 文字纠正 + 补 2 个 sub-enum `#[non_exhaustive]`

- **背景**：v3.3.0 给 `XenonError` 加 `#[non_exhaustive]` 时附带的 SemVer policy doc comment 存在文字误导——把"顶层 `#[non_exhaustive]` 允许新增变体"与"变体级 `#[non_exhaustive]` 允许变体内新增字段"两件事混淆，写成"inner sub-enums...absorb future field/variant growth without bumping major"。重审专家定级为 MAJOR（误导未来 SemVer 决策）。
- **修复**：
  - `XenonError` doc comment 重写 SemVer policy 段：明确区分**顶层 `#[non_exhaustive]` 允许的**（新增顶层变体非破坏）、**顶层 `#[non_exhaustive]` 不允许的**（给已有 struct-style variant 加字段仍 breaking——需要 per-variant `#[non_exhaustive]`）、**sub-enum `#[non_exhaustive]` 独立保护的**（在已有 payload 内新增子分类）、**始终 breaking 的变更**。
  - 补加 `#[non_exhaustive]` 到 `WorkspaceErrorCategory` 与 `InvalidArgumentKind`（v3.3.0 漏标，但本意一致——它们也是 sub-enum 风格的可扩展枚举，doc comment 已说明）。
  - §5.6 表格"封闭枚举"措辞改为"结构化子枚举"，并精确标注 `#[non_exhaustive]` 起始版本——`FfiErrorCategory` / `InvalidShapeKind` 自 v3.3.0；`WorkspaceErrorCategory` / `InvalidArgumentKind` 自 v3.3.1（v3.3.0 漏标，v3.3.1 补全）。`FfiBackend` / `StorageKindTag` 标注为"封闭枚举（design-intent closed，无 `#[non_exhaustive]`）"以区分。
- **不变项**：所有变体、字段集合、字段类型、错误语义、Display 输出格式、SemVer 边界——零变化。
- **协同**：纯 doc 与属性补强；引用 26-error 的文档无需修改字段或行为描述。

### v3.3.0 (2026-05-05) — 公开错误 enum 加 `#[non_exhaustive]`（SemVer 防御性更新）

- **背景**：v3.2.0 之前 `XenonError` 是公开 exhaustive enum，未来若需新增第 14 个顶层错误类别就是破坏性变更（强制下游所有 `match` 加新 arm）。Oracle 评审认定为 MAJOR：13 顶层变体一旦冻结，1.x 内任何错误扩展都会强制下游代码改动，SemVer 压力高。
- **修复方向（方向 A，本文档单点修订）**：给 `XenonError` 与可扩展的内部 sub-enum 加 `#[non_exhaustive]` 属性。封闭集合不动（`StorageKindTag` 4 种存储模式、`FfiBackend` 仅 RawParts/Blas 两种且 doc 显式声明 closed），保留 closed enum 语义。
- **`#[non_exhaustive]` 应用清单**：
  - `XenonError` 顶层 enum——核心防御。
  - `FfiErrorCategory`——FFI 错误类别可能新增（如未来 LAPACK / GPU backend）。
  - `AbiMismatchKind`——ABI 不匹配类型可能新增。
  - `InvalidLayoutReason`——v3.x 已经 patch 多次新增 reason 变体，明确开放扩展。
  - `InvalidShapeKind`——同样高频扩展。
  - `WorkspaceErrorCategory`——工作空间错误类别可能新增。
  - `InvalidArgumentKind`——一个操作族对应一个 variant，新族需扩展。
- **不加 `#[non_exhaustive]` 的 enum**：
  - `FfiBackend`——doc comment 明确声明为 closed enum；任何新 backend 必须 SemVer-tracked。设计意图就是封闭。
  - `StorageKindTag`——对应 4 种存储模式（Owned/View/ViewMut/Shared），与 `07-tensor.md StorageKind` 一一对应；存储模式集合本身就是封闭的。
- **配套 doc 更新**：`XenonError` doc comment 新增"SemVer policy (1.x baseline)"段落，明确区分：
  - **顶层 `#[non_exhaustive]` 允许的**：新增顶层变体（如第 14 个 category）是 minor 非破坏性。
  - **顶层 `#[non_exhaustive]` 不允许的**：给已有 struct-style variant 加字段仍是 breaking——`#[non_exhaustive]` **不**沿继承到 variant 内部。后续若需要变体内部加字段，必须给该变体本身额外加 `#[non_exhaustive]`（Rust 支持 struct-like variants 的 per-variant 该属性）。
  - **sub-enum `#[non_exhaustive]` 独立保护**：内部子枚举（FfiErrorCategory / AbiMismatchKind / InvalidLayoutReason / InvalidShapeKind / WorkspaceErrorCategory / InvalidArgumentKind）加 `#[non_exhaustive]` 是另一层独立保护——保证可在 `XenonError::Ffi { category, .. }` 等已有 payload 内新增子分类。
  - **始终 breaking 的变更**：删除/重命名顶层或 sub-enum 变体；给非 `#[non_exhaustive]` 变体加字段；改变既有字段类型/语义。
  交叉引用 `01-architecture.md §13 决策 8`。
- **下游影响**：所有 `match` `XenonError` / `FfiErrorCategory` / `AbiMismatchKind` / `InvalidLayoutReason` / `InvalidShapeKind` 的代码 **MUST** 包含 `_ => ...` wildcard arm。这是 `#[non_exhaustive]` 的 Rust 标准约束。
- **协同**：本次修改不改变任何字段、变体或语义，仅增加 `#[non_exhaustive]` 属性。所有引用 `26-error.md` 的文档（`01-architecture.md §1.5`、各模块的协同基线 pin）需 bump pin 到 v3.3.0；但实际错误字段集合不变，因此引用方文档的内容无需改动。

### v3.2.0 (2026-05-03) — ElementType 字段类型改 `&'static str`（破坏性公开 API 更新；ElementType 类型回归 element）

> 本版本与 `03-element.md v1.4.0` 协同。**公开 API 破坏性变更**：`XenonError::TypeConversion::source_type` / `target_type` 与 `AbiMismatchKind::ElementTypeMismatch::expected` / `actual` 的字段类型从 `ElementType` 枚举改为 `&'static str`；同时本模块**移除** `ElementType` 枚举定义（v3.1.0–v3.1.1 中临时持有，回到 `crate::element` 拥有，详见 `03-element.md §5.1.1`）。L0 单向依赖严格成立：error 现在不依赖任何 internal 模块，仅依赖标准库。

**破坏性变更点**：
- `XenonError::TypeConversion { source_type: ElementType, target_type: ElementType, ... }` → `{ source_type: &'static str, target_type: &'static str, ... }`
- `AbiMismatchKind::ElementTypeMismatch { expected: ElementType, actual: ElementType }` → `{ expected: &'static str, actual: &'static str }`
- `crate::error::ElementType` 路径**不再存在**；唯一权威路径是 `crate::element::ElementType`，FFI 路径是 `crate::ffi::ElementType`（`element` re-export）
- 错误构造站点：`source_type: ElementType::F32` → `source_type: <f32 as Element>::ELEMENT_TYPE_NAME`（值为 `"f32"`），或等价 `crate::element::element_type_name_of::<f32>()`

**契约更新**：
- §4.1 / §4.4：error 严格 L0，无 internal 依赖；`ElementType` re-export 链断开
- §5.1：删除 `ElementType` 定义与 Display impl；保留对 `03-element.md §5.1.1` 的引用
- §5.4：`TypeConversion` / `ElementTypeMismatch` 字段类型注释更新
- §5.6 / §5.7：字段表与 Display 实现规范更新
- §11 决策 6：增加 v3.2.0 决策反转说明与新替代方案
- §13 关联类型表（PartialEq）：从枚举判别比较改为字符串字面量比较

**测试影响**：
- `test_type_conversion_uses_element_type` → `test_type_conversion_uses_element_type_name`（断言值改为字符串字面量）
- 所有错误构造期望字面量从 `ElementType::F32` 改为 `"f32"`

**未受影响**：FFI 模块 `TensorExport.element_type: ElementType` 字段保持 ABI 稳定，`element_type` 仍是 enum（C 端按 u8 discriminant 比较），只是路径从 `crate::error::ElementType` 改为 `crate::element::ElementType`（详见 `23-ffi.md`）。

### v3.1.1 (2026-05-03) — TypedByteLengthOverflow 新增

> 详见 `24-workspace.md §5.6` 协同更新。`TypedViewRejection` 增加 `TypedByteLengthOverflow { count, elem_size }` 变体替代误用的 `GrowOverflow`。

### v3.1.0 (2026-05-03) — ElementType 下沉到 L0 + OverlapRejected 启用 + LengthNotMultipleOfSize 删除（破坏性内部更新）

> 本版本恢复 `01-architecture.md §5.2` 的 L0..L6 单向依赖，并清理 v3.0 协同期中遗留的两处死变体。公开 `XenonError` 变体名称保持兼容，调用方仅在错误诊断路径才能感知差异。

- §4.1 / §4.4：`error.rs` 不再依赖 `crate::element::ElementType`。`ElementType` 枚举的权威定义下沉到 L0 `error` 模块，`element` 与 `ffi` 通过 `pub use crate::error::ElementType` 暴露上层稳定路径；`element_type_of::<A>()` 帮助函数因依赖 `A: Element` trait bound，作为 `element` 模块的 **自由函数（free function，非 inherent impl）** 留在 `element`——Rust 不允许在 `element` 模块为定义于 `error` 模块的类型添加 inherent impl（E0116），详见 `03-element.md §5.1` v1.3.1 决策记录。
- §5.1：`ElementType` 枚举完整定义（含 `Display` impl）出现在本模块，与 `XenonError` 同处一个文件。变体名（`Bool`/`I32`/`I64`/`F32`/`F64`/`Complex32`/`Complex64`）与 `repr(u8)` / derive 集合不变；`Display` 输出格式延续 03-element.md 中的稳定文本（`bool` / `i32` / `i64` / `f32` / `f64` / `Complex<f32>` / `Complex<f64>`）。
- §5.5 / §11 决策 6：`TypeConversion` 字段引用从 `crate::element::ElementType` 改为本模块 §5.1 的权威定义；语义无变化。
- C5 协同：`FfiErrorCategory::OverlapRejected { shape, strides }` 由本版本起被 `23-ffi v2.x` 的 `from_raw_parts_mut` 自别名路径正式启用，不再是文档中的死变体（具体修订见 `23-ffi.md` 最新版本）。
- C11 协同：`WorkspaceErrorCategory::TypedViewRejection::LengthNotMultipleOfSize { len_bytes, elem_size }` 子变体在本版本删除，因为 `24-workspace v2.x` 的 typed view API 仅按 `count` 申请、不存在按字节长度 reinterpret 的路径。

### v3.0.1 (2026-05-03) — Medium/Low documentation follow-up

- Explained why recursive `XenonError` source chains require `Box<XenonError>`.
- Corrected the `Error::source()` implementation task to return inner errors for `Ffi` / `Workspace` causes and `None` for leaf variants.
- Replaced an internal review placeholder with explicit historical wording and marked `AlreadyBorrowed` as an old-model historical name.

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
