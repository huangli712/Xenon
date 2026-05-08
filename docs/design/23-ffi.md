# FFI 接口模块设计

> 文档编号: 23
> 模块目录: src/ffi/
> 任务阶段: Phase 4
> 前置文档: 02-dimension.md, 03-element.md, 05-storage.md, 06-layout.md, 07-tensor.md, 26-error.md
> 需求参考: 需求说明书 §5 - §8、§25、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责            | 包含                                                                        |
| --------------- | --------------------------------------------------------------------------- |
| 原始指针 API    | `as_ptr()`/`as_mut_ptr()`                                                   |
| 裸指针构造张量  | `from_raw_parts`/`from_raw_parts_mut`                                       |
| 裸指针解构张量  | `into_raw_parts`                                                            |
| BLAS 兼容性 API | `is_blas_layout_compatible()` 与 BLAS 元数据导出（`blas_info()` / `lda()`） |
| 多维索引转换    | `try_offset_of()`/`try_ptr_at()`                                            |

| 职责            | 不包含                                              |
| --------------- | --------------------------------------------------- |
| 原始指针 API    | BLAS 绑定实现（由上游库通过 `blas-sys` crate 提供） |
| 裸指针构造张量  | GPU 内存操作                                        |
| 裸指针解构张量  | 跨进程共享内存                                      |
| BLAS 兼容性 API | 自动调用 BLAS（由上游库负责）                       |
| 多维索引转换    | 序列化/反序列化                                     |

### 1.2 设计原则

| 原则         | 体现                                        |
| ------------ | ------------------------------------------- |
| 零拷贝       | 指针 API 无数据拷贝，O(1) 开销              |
| 安全边界清晰 | 所有 unsafe 函数有详尽 Safety 文档          |
| BLAS 友好    | 提供完整的 BLAS 兼容性检查和布局查询        |
| 最小约束     | FFI 方法避免重复安全检查（调用方已 unsafe） |
| 错误结构化   | FFI 错误一律使用 `26-error.md §5.1` 封闭枚举（`FfiErrorCategory` / `FfiBackend` / `InvalidLayoutReason` / `StorageKindTag`）；禁止自由文本 `precondition`/`actual`/`reason` |
| FFI panic 边界 | Xenon 本模块**不**定义 `extern "C"` 导出函数；任何上游 C ABI wrapper 必须阻止 Rust panic 穿越 C ABI（`std::panic::catch_unwind` 或在 FFI 边界采用 `panic = "abort"`），并保证 panic 后不会继续使用已失效的 `TensorExport` 指针 |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                        |
| -------- | --------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §5 -  §8、§25、§27、§28                                          |
| 范围内   | 原始指针访问、raw-parts 往返、BLAS 兼容性查询、多维索引到偏移 / 指针转换。  |
| 范围外   | 实际 BLAS / LAPACK 例程调用、GPU 互操作、跨进程共享内存与更高层序列化协议。 |
| 非目标   | 不把 `ffi` 扩展为外部数值库绑定层，不新增第三方 FFI crate 依赖。            |

---

## 3. 文件位置

```
src/
└── ffi/
    ├── mod.rs         # Module root, re-exports
    ├── types.rs       # TensorExportRaw / TensorExportMutRaw (non-generic, C-visible);
    │                  #   BlasInfo type definitions; re-exports ElementType (from element),
    │                  #   FfiErrorCategory (from error)
    ├── ptr.rs         # Raw-pointer API wrappers (export/export_mut, re-export from tensor module)
    ├── blas.rs        # BLAS compatibility checks (is_blas_layout_compatible, blas_info, lda)
    ├── offset.rs      # Multi-dimensional index to pointer offset (try_offset_of, try_ptr_at)
    └── private.rs     # Generic Rust-only descriptors `TensorExport<'a, A>` /
                       #   `TensorExportMut<'a, A>` + `From` impls converting them to
                       #   the C-visible `TensorExportRaw` / `TensorExportMutRaw`.
                       #
                       # Marked `#[doc(hidden)]` and `pub(crate)`-exported. cbindgen
                       # treats this as internal — these generic types never appear
                       # in any `extern "C"` function signature, so they are not
                       # reachable from cbindgen's transitive emission set. This is
                       # gate #2 of the three-gate cbindgen contract (see §5.3.bis).
```

多文件设计：将 FFI 按职责拆分为多个文件，便于后期拓展和维护。`private.rs` 隔离 generic Rust-only 描述符，是 cbindgen 三道闸门契约的 gate #2（见 §5.3.bis），确保泛型类型与 C-visible raw 描述符在文件层面就分离。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/ffi/
├── mod.rs
│   └── re-exports from types, ptr, blas, offset
├── types.rs
│   ├── core
│   └── crate::element       # re-exports ElementType (defined in element module)
├── ptr.rs
│   ├── crate::tensor        # TensorBase<S, D>, offset, OwnedRawParts re-export
│   ├── crate::dimension     # Dimension trait
│   ├── crate::element       # Element trait (for element_type_of free fn)
│   └── crate::storage       # Storage, StorageMut, owned allocator metadata
├── blas.rs
│   ├── crate::tensor        # TensorBase<S, D> (as_ptr via inherent method)
│   ├── crate::storage       # Storage
│   ├── crate::layout        # is_f_contiguous, has_zero_stride (via TensorBase method → LayoutFlags)
│   ├── crate::error         # XenonError, FfiErrorCategory (blas_info/lda error construction)
│   └── super::types         # BlasInfo, FfiErrorCategory
└── offset.rs
    ├── crate::tensor        # TensorBase<S, D>
    ├── crate::dimension     # Dimension trait
    ├── crate::storage       # Storage<Elem=A>
    └── crate::error         # XenonError (try_offset_of/try_ptr_at error construction)
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `.shape()`, `.strides()`, `.offset()`                                               |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`                                                                       |
| `element`   | `Element`、`ElementType`（**权威定义在 `crate::element`**，v1.4.0 起；本模块通过 `pub use crate::element::ElementType` re-export 暴露 `crate::ffi::ElementType` 给 C 消费者）、`element_type_of::<A>()`（`pub const fn`，定义在 `crate::element`，配合 inherent `ElementType::of::<A>()` 共同提供入口；详见 `03-element.md §5.1.1` v1.4.0 决策） |
| `storage`   | `Storage<Elem=A>`, `StorageMut<Elem=A>`, owned allocator metadata（供 `OwnedRawParts<A, D>` 导出/重建） |
| `layout`    | `is_f_contiguous()`（定义于 `06-layout.md` §5.7）、`has_zero_stride()`（定义于 `06-layout.md` §5.1）；`TensorBase` 方法参见 `07-tensor.md` §5.3 |
| `error`     | `XenonError`（含 `Ffi`、`DimensionMismatch`、`IndexOutOfBounds`、`InvalidLayout` 等变体）、`FfiErrorCategory`（封闭枚举，定义于 `26-error.md` §5.1，含 `NullPointer`/`AlignmentMismatch`/`InvalidRank`/`BlasIncompatibleLayout`/`IntegerOverflow`/`AbiMismatch`/`OverlapRejected`/`ForeignAllocatorMismatch` 八个结构化子变体）、`FfiBackend`（封闭枚举：`RawParts`/`Blas`，定义于 `26-error.md` §5.1）、`InvalidLayoutReason`（封闭枚举，定义于 `26-error.md` §5.1）、`StorageKindTag`（封闭枚举：`Owned`/`View`/`ViewMut`/`Shared`，由 `ArcRepr<A>` 支撑 `Shared`；定义于 `26-error.md` §5.1） |

### 4.3 依赖合法性

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.4 依赖方向

依赖方向：单向向上。`ffi` 仅消费 `tensor`、`storage` 等核心模块，为上游库提供接口。

本文聚焦这些能力在 FFI 边界的公开形态，因此依赖表中仍把相关实现文件归入 `ffi` 模块文档范围，而不把它写成反向依赖。

---

## 5. 公共 API 设计

**inherent 方法模式：** FFI 模块中的 `export()`、`export_mut()`、`is_blas_layout_compatible()`、`blas_info()`、`lda()`、`try_offset_of()`、`try_ptr_at()` 均为 `TensorBase<S, D>` 的 inherent 方法，但代码组织在 `src/ffi/` 子目录中。这些方法需要访问 `TensorBase` 的公开接口（`shape()`、`strides()` 等），无需直接操作私有字段，因此通过 inherent impl 在 ffi 模块中定义而不影响模块边界。这遵循了 §4.4 中的 owner 约定：核心构造与解构方法（`from_raw_parts*()`、`into_raw_parts()`）保留在 tensor 模块，FFI 模块仅负责面向 FFI 消费者的查询与导出方法。

**owner 约定：** 

- `as_ptr()` / `as_mut_ptr()` 的核心定义在 `07-tensor.md`（tensor 核心层）。
- `into_raw_parts()` / `from_raw_parts_owned()` / `OwnedRawParts` 的核心实现同样在 `07-tensor.md §5.7`（`src/tensor/construct.rs`），因为它们需要访问 `TensorBase` 的私有字段。
- `ffi` 模块负责指针导出格式（`TensorExport` / `TensorExportMut`）、BLAS 辅助 API 和裸指针偏移计算（`try_offset_of` / `try_ptr_at`）。
- `ffi` 模块通过 `pub use crate::tensor::OwnedRawParts` 向 FFI 消费者 re-export tensor 模块定义的类型。`into_raw_parts()` 和 `from_raw_parts_owned()` 作为 `TensorBase` 的 inherent 方法可直接在 FFI 上下文中调用，无需额外包装。

### 5.1 辅助类型

```rust,ignore
use crate::error::{FfiErrorCategory, FfiBackend};

/// FFI-specific recoverable errors are constructed directly as
/// `XenonError::Ffi { operation, category, backend, cause }`. See
/// `26-error.md §5.1` for the authoritative field list.
///
/// - `operation: Cow<'static, str>` — operation name (e.g.
///   `Cow::Borrowed("ffi::blas_info")`).
/// - `category: FfiErrorCategory` — closed enum carrying the failure
///   class together with its **structured** payload (e.g.
///   `BlasIncompatibleLayout { shape, strides }`,
///   `IntegerOverflow { value, target_width_bits }`,
///   `InvalidRank { expected, actual }`). No free-text payload.
/// - `backend: FfiBackend` — closed enum: `RawParts` for generic raw-parts
///   FFI, `Blas` for BLAS-compatible export.
/// - `cause: Option<Box<XenonError>>` — optional source-chain pointer
///   per `26-error.md` §5.1; the chain is exposed to callers via
///   `std::error::Error::source()` (see `26-error.md` §5.1 `impl Error`).
///
/// Example source-chain construction when an FFI boundary wraps a lower-level
/// Workspace failure:
///
/// ```ignore
/// let inner = XenonError::Workspace {
///     operation: Cow::Borrowed("Workspace::borrow"),
///     category: WorkspaceErrorCategory::BorrowConflict {
///         requested: WorkspaceBorrowKind::Shared,
///         current: WorkspaceBorrowState::Exclusive,
///     },
///     cause: None,
/// };
/// let outer = XenonError::Ffi {
///     operation: Cow::Borrowed("ffi::export_workspace_buffer"),
///     category: FfiErrorCategory::AbiMismatch {
///         detail: AbiMismatchKind::CapacityMismatch { expected: 1024, actual: 512 },
///     },
///     backend: FfiBackend::RawParts,
///     cause: Some(Box::new(inner)),
/// };
/// ```
///
/// Leaf FFI errors normally use `cause: None`; wrapper errors use
/// `cause: Some(Box::new(inner))`.
///
/// FFI errors must NOT use free-text `precondition` / `actual` fields;
/// the structured payload inside `FfiErrorCategory` already carries the
/// diagnostic context required by `requirements specification §27`.
```

### 5.2 原始指针 API

**结果类型说明：** 公开 API 统一使用 `Result<T, XenonError>`，`crate::error::Result<_>` 为等价类型别名。

**核心定义归属：** `as_ptr()`、`as_mut_ptr()`、`from_raw_parts()`、`from_raw_parts_mut()` 的实现定义在 `07-tensor.md` §5.4 和 §5.7，代码位于 `src/tensor/construct.rs`。本文仅描述这些方法在 FFI 边界的语义契约和 Safety 要求；完整签名与实现参见 `07-tensor.md`。

````rust,ignore
// as_ptr() — see 07-tensor.md §5.4
//
// Returns a read-only raw pointer to the logical first element.
// - Non-empty: storage.as_ptr().add(offset)
// - Empty:     NonNull::<A>::dangling().as_ptr()
//
// impl<S, D, A> TensorBase<S, D> where S: Storage<Elem = A>, D: Dimension {
//     pub fn as_ptr(&self) -> *const A { ... }
// }

// as_mut_ptr() — see 07-tensor.md §5.4
//
// Returns a mutable raw pointer to the logical first element.
// Only available for S: StorageMut.
// - Non-empty: storage.as_mut_ptr().add(offset)
// - Empty:     NonNull::<A>::dangling().as_ptr()
//
// impl<S, D, A> TensorBase<S, D> where S: StorageMut<Elem = A>, D: Dimension {
//     pub fn as_mut_ptr(&mut self) -> *mut A { ... }
// }

// from_raw_parts() — see 07-tensor.md §5.7
//
// Constructs an immutable view from raw pointer.
// ptr = storage base pointer; offset = displacement to logical first element.
// Calls validate_access_range() internally; empty tensors use dangling sentinel.
//
// impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D> where A: Element, D: Dimension {
//     pub unsafe fn from_raw_parts(
//         ptr: *const A, storage_len: usize, shape: D,
//         strides: Strides<D>, offset: usize,
//     ) -> Result<Self, XenonError> { ... }
// }

// from_raw_parts_mut() — see 07-tensor.md §5.7
//
// Constructs a mutable view from raw pointer.
// Same as from_raw_parts, plus: rejects zero-stride non-singleton axes and
// overlapping layouts via validate_non_overlapping_layout().
//
// impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D> where A: Element, D: Dimension {
//     pub unsafe fn from_raw_parts_mut(
//         ptr: *mut A, storage_len: usize, shape: D,
//         strides: Strides<D>, offset: usize,
//     ) -> Result<Self, XenonError> { ... }
// }
````

**FFI 侧 Safety 摘要（调用方须保证）：**

| 方法 | 调用方义务 |
|------|-----------|
| `as_ptr()` | 返回指针借用源张量；源张量 drop 后立即失效 |
| `as_mut_ptr()` | 同上，且借用期间不可有其它引用 |
| `from_raw_parts()` | ptr 有效/对齐/覆盖全部可达元素；生命周期 `'a` 内可读共享 |
| `from_raw_parts_mut()` | 同上，且独占可写、逻辑元素地址不重叠 |

### 5.3 C 侧结构化导出格式

`ElementType` 枚举定义于 `element` 模块（见 `03-element.md §5.1.1`，v1.4.0 起），`ffi` 模块通过 `pub use crate::element::ElementType` re-export 以供 FFI 消费者使用稳定路径 `crate::ffi::ElementType`。此设计让 element 拥有类型枚举，让 ffi 提供 C ABI 边界稳定路径，同时 error 模块完全不依赖 ElementType（error 用 `&'static str` 记录类型诊断信息，详见 `26-error.md v3.2.0 §5.4`）。

```rust,ignore
// src/ffi/types.rs
pub use crate::element::ElementType;      // re-export from element module
pub use crate::error::FfiErrorCategory;   // re-export from error module

/// BLAS layout metadata (full definition in §5.5).
pub struct BlasInfo<A> { /* fields omitted — see §5.5 */ }

// See 03-element.md §5.1.1 for the full ElementType definition.
// Only the FFI-consumer-visible public API signature is shown here.
//
// **C ABI value pinning**: each variant has an explicit
// discriminant. These values are SemVer-pinned for `crate::ffi::ElementType`
// consumers — adding a new variant gets a new value and is a non-breaking
// change under `#[non_exhaustive]`; reordering or reusing existing values
// is a breaking change for C ABI consumers and requires a major version
// bump.
//
// #[repr(u8)] #[non_exhaustive]
// pub enum ElementType {
//     Bool      = 0,
//     I32       = 1,
//     I64       = 2,
//     F32       = 3,
//     F64       = 4,
//     Complex32 = 5,
//     Complex64 = 6,
// }
//
// impl ElementType {
//     pub const fn name(self) -> &'static str { ... }
//     pub const fn of<A: Element>() -> Self { A::ELEMENT_TYPE }
// }
```

#### 5.3.bis C 头文件可见的非泛型导出 schema（v3.0.2）

`TensorExport<'a, A>` / `TensorExportMut<'a, A>` 是 Rust 侧带生命周期与 `PhantomData` 的泛型类型，C 头文件无法直接表达"泛型 + 生命周期 + PhantomData"。`crate::ffi` 因此对外暴露**非泛型**的 C-visible 描述符，作为 cbindgen 的固定输出 schema：

```rust,ignore
/// C-visible read-only tensor descriptor.
///
/// This is the cbindgen-emitted concrete schema. Generic
/// `TensorExport<'a, A>` is converted to `TensorExportRaw` at the FFI
/// boundary by stripping the lifetime / `PhantomData` and erasing
/// `*const A` to `*const core::ffi::c_void`. C consumers cast `data`
/// to the matching pointer type using `element_type` as the discriminator.
#[repr(C)]
pub struct TensorExportRaw {
    pub data: *const core::ffi::c_void,
    pub element_type: ElementType,    // see C ABI value pinning above
    pub ndim: usize,
    pub shape: *const usize,
    pub strides: *const usize,
    pub storage_len: usize,
    pub offset: usize,
    // No PhantomData / lifetime: those are Rust-only.
}

/// C-visible mutable tensor descriptor (writable variant).
#[repr(C)]
pub struct TensorExportMutRaw {
    pub data: *mut core::ffi::c_void,
    pub element_type: ElementType,
    pub ndim: usize,
    pub shape: *const usize,
    pub strides: *const usize,
    pub storage_len: usize,
    pub offset: usize,
}

// Rust-side conversion (consumed at the FFI boundary, never crosses C).
impl<'a, A: Element> From<TensorExport<'a, A>> for TensorExportRaw {
    fn from(e: TensorExport<'a, A>) -> Self {
        TensorExportRaw {
            data: e.data as *const core::ffi::c_void,
            element_type: e.element_type,
            ndim: e.ndim,
            shape: e.shape,
            strides: e.strides,
            storage_len: e.storage_len,
            offset: e.offset,
        }
    }
}

impl<'a, A: Element> From<TensorExportMut<'a, A>> for TensorExportMutRaw {
    fn from(e: TensorExportMut<'a, A>) -> Self {
        TensorExportMutRaw {
            data: e.data as *mut core::ffi::c_void,
            element_type: e.element_type,
            ndim: e.ndim,
            shape: e.shape,
            strides: e.strides,
            storage_len: e.storage_len,
            offset: e.offset,
        }
    }
}
```

C 消费者**只能**绑定到 `TensorExportRaw` / `TensorExportMutRaw`；Rust 侧的 `TensorExport<'a, A>` / `TensorExportMut<'a, A>` 是内部表达类型，包含生命周期借用证据与类型化指针，cbindgen 不会为其生成 C 头文件条目。两类描述符通过 `From` 在 FFI 边界一次性转换。这一设计保留了 Rust 侧的借用安全（`PhantomData<&'a A>` 阻止借用越界），同时给 C 一份稳定可消费的 ABI schema。

#### cbindgen 配置合约（v3.0.2 强制）

为强制 generic Rust-only 描述符不进入 C 头文件，工程依赖 **三道闸门**协同（cbindgen 没有真正的 "exhaustive allowlist" 机制；`[export] include` 只是把那些没被 `extern "C"` 函数引用、但也想强制纳入的额外类型 *补充进来*，并不能把生成集合限制为只有列表中的项）：

1. **`extern "C"` 函数签名只引用 raw 描述符。** `crate::ffi` 的所有 `extern "C"` 函数只接受 / 返回 `TensorExportRaw` / `TensorExportMutRaw` / `ElementType` 等非泛型 C-visible 类型。这是最强约束：cbindgen 仅会生成被 `extern "C"` 函数实际依赖的类型。
2. **`#[doc(hidden)] mod private { ... }` 隔离泛型类型。** `TensorExport<'a, A>` / `TensorExportMut<'a, A>` 与 `From` 转换 impl 全部放在 `crate::ffi::private` 子模块内，让 cbindgen 的 parser 把它们视为内部细节；同时通过 `[export.exclude]` 显式列出名字作为第二道闸门防止意外暴露。
3. **`cbindgen.toml` 显式 `[export.exclude]` + 测试时头文件 grep 检查。** 三道闸门叠加，任何一道单独失效（例如未来重构把泛型类型移出 private 模块）都不会让 generic schema 泄露到 C 头。

```toml
# cbindgen.toml — repository-pinned excerpt for FFI ABI stability.
language = "C"

[export]
# Force-include items NOT referenced by any extern "C" function (rare).
# This is NOT an exhaustive allowlist — cbindgen ALWAYS emits items
# transitively reachable from extern "C" function signatures regardless
# of this list. The first gate (extern "C" only references raw types)
# is what actually constrains the output set.
include = [
    "ElementType",      # standalone enum, ABI-stable
    "FfiErrorCode",     # error code enum exposed to C
    "FfiBackend",       # backend tag enum exposed to C
]

# Second gate: explicit deny by name. Even if a future refactor accidentally
# referenced these from an extern "C" function, cbindgen would skip them.
exclude = [
    "TensorExport",
    "TensorExportMut",
]

[parse]
parse_deps = false  # do not parse dependency crates' types
```

**测试合约（28-tests）**：必须包含 `test_cbindgen_header_exports_only_raw_descriptors`，断言生成的 C 头文件：

1. 包含 `typedef ... TensorExportRaw;` / `typedef ... TensorExportMutRaw;` / `enum ElementType` 定义；
2. **不**包含 `TensorExport` / `TensorExportMut` 这两个**裸标识符**（必须使用 word-boundary 正则匹配，例如 `\bTensorExport\b` / `\bTensorExportMut\b`，**不**使用普通 substring grep；否则 `TensorExportRaw` / `TensorExportMutRaw` 会被前缀误命中）的任何 typedef / struct / enum 出现——这是三道闸门的最终验证；
3. `ElementType` 枚举值与 03-element §5.1.1 显式 discriminants 严格一致（`Bool=0..Complex64=6`）。

CI 在每次 PR 重新生成 C 头并对比预期 schema；schema 差异需 reviewer 在 PR 中显式确认。

### 5.4 指针约定对照

| API                         | 基准                 | 说明                                                                 |
| --------------------------- | -------------------- | -------------------------------------------------------------------- |
| `as_ptr()` / `as_mut_ptr()` | 逻辑首元素           | 对非空张量返回第一个逻辑元素的指针；空张量返回 dangling              |
| `TensorExport.data`         | storage base pointer | 非空张量时等于底层存储的基地址；空张量时为有效对齐但不可解引用的指针 |
| `BlasInfo.data_ptr`         | 逻辑首元素           | 等价于 `as_ptr()`                                                    |
| `try_ptr_at(indices)`       | 指定逻辑位置         | 基于 `as_ptr()` + `try_offset_of(indices)` 结果计算                  |

```rust,ignore
/// Raw tensor data export for FFI consumers.
///
/// # Safety
///
/// - All pointer fields (`data`, `shape`, `strides`) borrow the source tensor's
///   internal storage and metadata. They become invalid immediately after the
///   source tensor is dropped.
/// - C consumers must use `ndim` as the length of both the `shape` and `strides`
///   arrays. Do NOT use hardcoded lengths or any other source.
/// - For `bool` element type, interoperability with C `_Bool` / C23 `bool` is
///   only documented for explicitly supported platforms/ABIs. This does not
///   constitute a cross-language stable ABI promise across all targets.
/// - `TensorExport` is the read-only export form and uses `*const A`.
///   `TensorExportMut` is the writable export form and uses `*mut A`.
///
/// **Visibility & file location (R13 D-01):** Generic descriptors are
/// `pub(crate)` Rust-only borrowing evidence and live in
/// `src/ffi/private.rs` (see §3 file layout + §5.3.bis cbindgen gate #2:
/// generic descriptors are physically isolated and excluded from cbindgen
/// emission set). They are NOT part of any C ABI surface; the C-visible
/// raw descriptors are `TensorExportRaw` / `TensorExportMutRaw` (defined
/// above). `#[doc(hidden)]` keeps them out of public rustdoc as well.
// File: src/ffi/private.rs
#[doc(hidden)]
#[repr(C)]
pub(crate) struct TensorExport<'a, A> {
    /// Typed pointer to the storage base pointer.
    ///
    /// For non-empty tensors this points at the underlying storage base.
    /// For empty tensors (`len() == 0`), this is still a valid aligned pointer
    /// but must not be dereferenced.
    ///
    /// `strides` and `offset` use element units of `A`.
    /// C consumers must cast `data` to the matching element type and interpret
    /// both `offset` and `strides` as element counts rather than byte counts.
    /// The logical first element address is `data.add(offset)` when `len() != 0`.
    ///
    pub data: *const A,
    /// Element type identifier (matches ElementType enum).
    pub element_type: ElementType,
    /// Number of dimensions.
    ///
    /// C consumers must use this value as the length of both `shape` and `strides`
    /// arrays. Do NOT substitute with any other value.
    pub ndim: usize,
    /// Shape array (length = ndim).
    pub shape: *const usize,
    /// Stride array (length = ndim), in units of elements (not bytes).
    pub strides: *const usize,
    /// Storage length in elements for safe view reconstruction.
    pub storage_len: usize,
    /// Logical offset metadata in element units, preserved for raw-parts
    /// roundtrip/reconstruction contracts.
    pub offset: usize,
    /// Lifetime marker tying the export to the source tensor borrow.
    ///
    /// Must be the last field in this `#[repr(C)]` struct: as a ZST,
    /// it contributes 0 bytes in the C ABI, but placing it in the middle
    /// can produce unspecified behavior across compiler versions.
    /// Keeping it last ensures cross-version consistency.
    pub _marker: core::marker::PhantomData<&'a A>,
}

/// Raw mutable tensor data export for FFI consumers.
///
/// Field semantics are identical to `TensorExport` unless noted below.
/// The only differences are: `data` is `*mut A` (writable), and
/// `_marker` uses `PhantomData<&'a mut A>` (exclusive borrow).
///
/// **Visibility & file location (R13 D-01):** Same `pub(crate)` Rust-only
/// scope and `src/ffi/private.rs` location as `TensorExport`. Not part of
/// any C ABI surface; C consumers see `TensorExportMutRaw` instead.
// File: src/ffi/private.rs
#[doc(hidden)]
#[repr(C)]
pub(crate) struct TensorExportMut<'a, A> {
    /// Typed mutable pointer to the storage base pointer.
    /// Same semantics as `TensorExport::data`, but writable.
    pub data: *mut A,
    /// See `TensorExport::element_type`.
    pub element_type: ElementType,
    /// See `TensorExport::ndim`.
    pub ndim: usize,
    /// See `TensorExport::shape`.
    pub shape: *const usize,
    /// See `TensorExport::strides`.
    pub strides: *const usize,
    /// See `TensorExport::storage_len`.
    pub storage_len: usize,
    /// See `TensorExport::offset`.
    pub offset: usize,
    /// Lifetime marker; `PhantomData<&'a mut A>` enforces exclusive borrow.
    ///
    /// Must be the last field (ZST), same rationale as `TensorExport::_marker`.
    pub _marker: core::marker::PhantomData<&'a mut A>,
}

// Static layout assertions for FFI compatibility.
//
// IMPORTANT (Rust `#[repr(C)]` rules): a raw sum of `size_of::<field>` is
// NOT the struct size — `#[repr(C)]` inserts padding before each field
// so that its address satisfies the field's alignment. After
// `data: *const f64` (8B aligned) the `element_type: ElementType`
// (`#[repr(u8)]`, 1B aligned) packs at offset 8, but the next field
// `ndim: usize` (8B aligned on 64-bit) needs offset 16, so 7 padding
// bytes appear between `element_type` and `ndim`. The total size on a
// typical 64-bit ABI is therefore not the naive sum.
//
// The assertions below verify field offsets and total size with padding
// taken into account. PhantomData<&'a A> is ZST (0 bytes) and is placed
// last in `#[repr(C)]` to avoid unspecified ZST positioning behavior.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    // Field offsets (verifies the C-side header layout for cbindgen consumers).
    assert!(offset_of!(TensorExport<f64>, data)         == 0);
    // `element_type` immediately follows the 8-byte pointer on 64-bit.
    assert!(offset_of!(TensorExport<f64>, element_type) == size_of::<*const f64>());
    // `ndim` is 8B-aligned: padding inserted between `element_type` and `ndim`.
    assert!(offset_of!(TensorExport<f64>, ndim)         % align_of::<usize>() == 0);
    // PhantomData ZST is last in `#[repr(C)]`.
    // (No offset assertion on `_marker`: ZST positioning is unobservable
    // in C ABI, but its trailing position is required by §5.3 prose.)

    // Total size equals offset of the last non-ZST field plus its size,
    // rounded up to the struct's overall alignment. We verify that the
    // total size is at least the maximum-aligned monotonic layout, and
    // is consistent with the offsets above:
    assert!(size_of::<TensorExport<f64>>() >= offset_of!(TensorExport<f64>, offset) + size_of::<usize>());
    assert!(size_of::<TensorExport<f64>>() % align_of::<TensorExport<f64>>() == 0);
};

```

`PhantomData<&'a A>` 必须放在 `#[repr(C)]` 结构体的最后位置：作为 ZST，它在 C ABI 视角不占据任何字节，但放在中间可能因不同编译器版本对 ZST 字段的处理而产生未指定行为。统一放在末尾确保跨版本一致性。

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    /// Export tensor data as a raw C-compatible structure.
    ///
    /// The returned `TensorExport` borrows the tensor's data and metadata.
    /// The consumer must ensure the tensor outlives the export.
    /// This method does not fail; it always returns a valid export.
    ///
    /// `data` always carries the storage base pointer; the logical first element
    /// address is derived from `data.add(offset)` for non-empty tensors.
    /// Empty tensors are allowed: when `len() == 0`, `data` is a valid aligned
    /// pointer that must not be dereferenced. `shape`, `strides`, and `offset`
    /// still describe the empty tensor metadata.
    /// **Visibility & return type (R15 D-01 fix):** This is the public FFI
    /// entry; it returns `TensorExportRaw` (the C-visible non-generic raw
    /// descriptor; see §5.4 above). The intermediate generic descriptor
    /// `TensorExport<'_, A>` is `pub(crate)` Rust-only borrowing evidence
    /// (located in `src/ffi/private.rs`, §3 + §5.3.bis) and cannot appear
    /// in a `pub fn` return type — Rust's `private_in_public` rule
    /// (`error[E0446]`) would reject that signature. The internal generic
    /// descriptor is built within this method and immediately converted to
    /// `TensorExportRaw` via the `From<TensorExport<'_, A>>` impl in §5.4.
    pub fn export(&self) -> TensorExportRaw {
        // Build the internal `pub(crate)` generic descriptor first; then
        // convert to the C-visible raw descriptor via `From` (§5.4).
        let generic = self.export_internal();
        generic.into()
    }

    /// `pub(crate)` internal helper: produces the typed generic descriptor
    /// for in-crate borrow tracking and lifetime evidence. Not exposed to
    /// downstream consumers; use `export()` for the public FFI surface.
    pub(crate) fn export_internal(&self) -> TensorExport<'_, A> {
        TensorExport {
            data: if self.is_empty() {
                // Empty tensor: return a valid aligned non-dereferenceable pointer.
                // Do NOT call as_storage_ptr() — the backing storage may be empty
                // or even unallocated (e.g. zero-cap Vec).
                core::ptr::NonNull::<A>::dangling().as_ptr()
            } else {
                self.as_storage_ptr()
            },
            _marker: core::marker::PhantomData,
            element_type: crate::element::element_type_of::<A>(),
            ndim: self.ndim(),
            shape: self.shape().as_slice().as_ptr(),
            strides: self.strides().as_slice().as_ptr(),
            storage_len: self.storage_len(),
            offset: self.offset(),
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    A: Element,
    D: Dimension,
{

    /// Export tensor data with mutable access.
    ///
    /// **Visibility & return type (R15 D-01 fix):** Public FFI entry; returns
    /// `TensorExportMutRaw` (the C-visible non-generic raw descriptor). The
    /// intermediate generic `TensorExportMut<'_, A>` is `pub(crate)` Rust-only
    /// (located in `src/ffi/private.rs`) and cannot appear in `pub fn` return
    /// type per Rust's `private_in_public` rule.
    ///
    /// This API is only implemented for writable storage, so read-only storage
    /// modes are rejected at the trait boundary rather than at runtime.
    /// No additional fallible validation is performed beyond the existing
    /// `&mut self` + `S: StorageMut` exclusivity boundary.
    pub fn export_mut(&mut self) -> TensorExportMutRaw {
        let generic = self.export_mut_internal();
        generic.into()
    }

    /// `pub(crate)` internal helper: produces the typed mutable generic
    /// descriptor for in-crate borrow tracking and lifetime evidence.
    /// Not exposed to downstream consumers; use `export_mut()` for the public
    /// FFI surface.
    pub(crate) fn export_mut_internal(&mut self) -> TensorExportMut<'_, A> {
        TensorExportMut {
            data: if self.is_empty() {
                core::ptr::NonNull::<A>::dangling().as_ptr()
            } else {
                self.as_storage_mut_ptr()
            },
            _marker: core::marker::PhantomData,
            element_type: crate::element::element_type_of::<A>(),
            ndim: self.ndim(),
            shape: self.shape().as_slice().as_ptr(),
            strides: self.strides().as_slice().as_ptr(),
            storage_len: self.storage_len(),
            offset: self.offset(),
        }
    }
}
```

**指针语义**：`TensorExportRaw.data`（`*const c_void`，C-visible）与 `TensorExportMutRaw.data`（`*mut c_void`，C-visible）通过 `From<TensorExport<'_, A>>` / `From<TensorExportMut<'_, A>>` 转换从 generic 描述符的 typed pointer 派生而来；C 消费者必须按 `element_type` 字段识别的类型 cast 后再访问。逻辑首元素地址通过 `data + offset * size_of_element` 计算，`offset` 与 `strides` 以元素个数（非字节）计量。空张量（`len() == 0`）时 `data` 为有效对齐但不可解引用的 dangling 指针。Generic descriptor (`TensorExport<'_, A>` / `TensorExportMut<'_, A>`) 仅在 crate 内部使用，详细字段语义见 §5.4 结构体注释与上方对照表。

**导出范围与可写边界**：公开 `export()` 返回 `TensorExportRaw`（内部经 generic descriptor 中转后转换），仅要求 `S: Storage`，覆盖 Owned、View、只读共享存储及所有合法 stride 布局。公开 `export_mut()` 返回 `TensorExportMutRaw`，要求 `S: StorageMut`，通过 `&mut self` 保证独占可写访问；只读视图和共享只读存储在 trait 边界上直接被拒绝，与 `需求说明书 §6` 的存储模式转换和 `需求说明书 §25` 的零拷贝导出要求保持一致。Crate 内部如需 generic descriptor 的 borrow 证据，使用 `pub(crate)` 的 `export_internal()` / `export_mut_internal()`。

**stride 约定**：`strides` 以元素个数（非字节）表示步长。按照 `06-layout.md` §1.2 与 `需求说明书 §7`，当前版本不支持负步长。`from_raw_parts()` 允许零步长布局以表达广播只读视图；`from_raw_parts_mut()` 拒绝所有非空零步长布局（非单元素轴的 `stride == 0` 会报错）。

**生命周期与 auto trait**：导出结果不拥有底层内存，`data`、`shape`、`strides` 均借用源张量内部数据，源张量 drop 后立即失效。`TensorExport` 包含 `*const A`（裸指针），因此 Rust 自动推导为 `!Send + !Sync`。如果 FFI 消费者需要跨线程共享，必须显式包装（如 `Arc<TensorExport<...>>` + 手动 `unsafe impl`）。Xenon 不为 `TensorExport` 提供 `Send/Sync` 自动实现。

#### 5.4.1 FFI panic 边界与回调限制

`TensorExport<'a, A>` / `TensorExportMut<'a, A>` 只是借用源张量内部数据和元数据的 Rust 结构体；本模块**不**提供 `pub extern "C"` 导出函数，也不承诺替调用方管理 C ABI 边界的 panic 行为。

当上游库把 `TensorExport` 传给 C 代码时，必须满足以下额外约束：

- Rust panic **不得**穿越 `extern "C"` ABI 边界。任何由上游定义的 `extern "C"` wrapper 都必须在最外层使用 `std::panic::catch_unwind` 捕获 panic 并转换为上游 ABI 的错误码，或在该 FFI 边界采用 `panic = "abort"` 策略。
- `TensorExport` / `TensorExportMut` 的所有指针只在源张量借用仍然存活、且没有发生 unwinding 导致源张量或相关 owner 被 drop 的期间有效。
- C 代码不得在持有 `TensorExport` 指针期间 re-enter Rust 并调用可能 panic 的 callback，除非该 callback 的 Rust ABI 边界同样捕获 panic 或直接 abort。
- 如果捕获到 panic，wrapper 必须把当前导出的所有 borrowed pointer 视为失效，**不得**继续让 C 侧读取或缓存这些指针。

推荐的上游 wrapper 形态如下：

```rust,ignore
#[repr(C)]
pub enum XenonFfiStatus {
    Ok = 0,
    Error = 1,
    Panic = 2,
}

#[no_mangle]
pub extern "C" fn upstream_call_xenon(/* C ABI args */) -> XenonFfiStatus {
    match std::panic::catch_unwind(|| {
        // Build or borrow the tensor.
        // Create TensorExport only for the synchronous C call duration.
        // Do not let C store the borrowed pointers after this function returns.
    }) {
        Ok(Ok(())) => XenonFfiStatus::Ok,
        Ok(Err(_err)) => XenonFfiStatus::Error,
        Err(_panic) => XenonFfiStatus::Panic,
    }
}
```

上述 wrapper 示例属于上游集成责任；Xenon 的 `export()` / `export_mut()` 本身仍保持 O(1)、不分配、不捕获 panic。

### 5.5 Complex FFI 布局契约

```rust,ignore
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

#[repr(C)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}
```

**Complex 布局约定：** `Complex<f32>` 与 `Complex<f64>` 的 FFI 表示分别等价于 `#[repr(C)] struct { re: f32, im: f32 }` 和 `#[repr(C)] struct { re: f64, im: f64 }`。复数类型的完整定义、运算和 ABI 约定参见 `04-complex.md`。

**内存保证：** `#[repr(C)]` 保证字段顺序固定为 `re` 后接 `im`，整体对齐分别等于 `f32` / `f64` 的 C ABI 对齐要求；若目标 ABI 需要尾部 padding，则该 padding 仅作用于单个复数元素末尾，不改变数组按该结构逐元素重复排布的语义。

**导出语义：** 导出复数张量时，`TensorExport<Complex<f32>>` / `TensorExport<Complex<f64>>` 和 `TensorExportMut<Complex<f32>>` / `TensorExportMut<Complex<f64>>` 中的 `data` 仍是“复数元素指针”，`offset` 与 `strides` 仍按“复数元素个数”计量，而不是按标量 `re/im` 分量或字节计量。C 侧看到的是 `Complex32*` / `Complex64*` 加上相同的 shape/stride 元数据。

### 5.6 Bool FFI 布局契约

**Bool ABI 约束：** `bool` 与 C `_Bool` / C23 `bool` 的互操作仅在文档明确支持的平台/ABI 下成立；它用于说明当前支持目标上的对接方式，不作为跨语言、跨目标的稳定 ABI 承诺。对这些已支持平台，C 消费者应使用 `_Bool` 或 `bool`（C23）来匹配 `TensorExport<bool>` / `TensorExportMut<bool>` 中的 `data` 指针类型，并避免使用 `int`、`unsigned char` 等其它整数类型。

**导出语义：** 导出 `bool` 张量时，`TensorExport<bool>` 中的 `data` 为 `*const bool`（C 侧 `const _Bool*`），`TensorExportMut<bool>` 中的 `data` 为 `*mut bool`（C 侧 `_Bool*`），`offset` 与 `strides` 按 `bool` 元素个数计量。`strides[i] == 1` 表示相邻逻辑元素在内存中连续排列（每个占 1 字节）。

**C 侧验证说明：** Xenon 仅对文档明确支持的平台/ABI 给出 Rust `bool` 与 C `_Bool` 的互操作说明；跨语言集成时，调用方仍应在目标工具链侧通过 `sizeof(_Bool) == 1`、`_Alignof(_Bool) == 1` 等静态断言验证兼容性，不应把该文档表述解读为跨平台稳定 ABI 保证。

**测试边界说明：** 与上述 ABI 约束一致，`bool` FFI ABI 相关测试也只应在文档明确支持的 targets/ABI 上启用；其它目标上应通过 `#[cfg(...)]` 跳过，而不是把 `_Bool` 兼容性断言提升为无条件测试基线。

### 5.7 从裸指针构造张量

**核心定义归属：** `from_raw_parts()` 和 `from_raw_parts_mut()` 的完整签名、实现和 Safety 文档定义在 `07-tensor.md` §5.7，代码位于 `src/tensor/construct.rs`。本文仅描述 FFI 边界的语义要点和校验逻辑概述。

**语义契约摘要：**

```rust,ignore
// from_raw_parts() — see 07-tensor.md §5.7 for the full implementation
//
// impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D> where A: Element, D: Dimension {
//     pub unsafe fn from_raw_parts(
//         ptr: *const A,        // storage base pointer
//         storage_len: usize,
//         shape: D,
//         strides: Strides<D>,
//         offset: usize,        // displacement from storage base to logical first
//     ) -> Result<Self, XenonError>
// }
//
// Internal validation flow:
//   1. validate_access_range(&shape, &strides, offset, storage_len)
//   2. Empty tensors use NonNull::dangling() as logical_ptr
//   3. compute_layout_flags(&shape, &strides, logical_ptr)
//
// from_raw_parts_mut() — see 07-tensor.md §5.7 for the full implementation
//
// impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D> where A: Element, D: Dimension {
//     pub unsafe fn from_raw_parts_mut(
//         ptr: *mut A, storage_len: usize, shape: D,
//         strides: Strides<D>, offset: usize,
//     ) -> Result<Self, XenonError>
// }
//
// Additional validation (beyond from_raw_parts):
//   1. Reject non-empty zero-stride layouts (stride == 0 on non-singleton axes)
//   2. validate_non_overlapping_layout() conservative non-overlap check
//   3. Empty tensors use NonNull::dangling() as logical_ptr
```

**校验边界说明：** 与 `07-tensor.md` §5.7 一致，`from_raw_parts*()` 只验证库能够直接检查的元数据约束（例如 shape/stride/offset/storage*len 组合是否合法、是否溢出、是否越界），并在失败时返回 `Result<*, XenonError>`。指针有效性、对齐、实际可访问范围与生命周期仍由调用方在 `unsafe` 前提下负责。

**空张量补充：** `ptr.add(offset)` 形式的逻辑首元素地址计算只适用于非空张量；空张量路径必须跳过该指针运算，并改用 `NonNull::dangling()` 这类明确定义的非解引用哨兵值参与 flags / metadata 初始化。

**可写视图补充：** `from_raw_parts_mut()` 不仅必须拒绝所有非空零步长布局（任何非单元素轴的 `stride == 0`），还必须拒绝一切能被高效保守判定为潜在自别名的布局。实现上先用 `validate_access_range()` 验证越界与可表示性，再用 `validate_non_overlapping_layout(shape, strides, offset, storage_len)` 对受支持的正步长布局做保守非重叠判定；若布局超出该高效判定范围，也必须返回可恢复错误，而不是枚举全部可达 offset。

### 5.8 将张量解构为裸指针

**实现归属：** `OwnedRawParts` 结构体及 `into_raw_parts()` / `from_raw_parts_owned()` 方法的**核心实现**定义于 `07-tensor.md §5.7`（`src/tensor/construct.rs`）。这些方法需要直接访问 `TensorBase` 的私有字段（`storage`、`shape`、`strides`、`offset`、`flags`），因此只能在 tensor 模块内定义。本模块（`src/ffi/ptr.rs`）通过 re-export 向 FFI 消费者暴露：

```rust,ignore
// src/ffi/ptr.rs
pub use crate::tensor::{OwnedRawParts, TensorBase};

// into_raw_parts() and from_raw_parts_owned() are inherent methods on
// TensorBase<Owned<A>, D> defined in src/tensor/construct.rs.
// They are directly callable on any Owned tensor; no wrapper is needed.
```

完整 API 签名、`OwnedRawParts` 字段定义、`into_raw_parts()` 代码、验证逻辑及 `# Safety` 契约参见 `07-tensor.md §5.7 "Owned 裸指针分解与重建"`。

**设计决策：** `into_raw_parts` 仅适用于 Owned 存储，且导出的内存布局必须满足 Xenon 的 owned 不变量：F-order contiguous、`offset == 0`、canonical F-order strides。若调用方持有的是 view 或带 offset 的逻辑子视图，必须先显式物化为新的 owned contiguous tensor，再跨越 FFI 边界导出裸指针。如需将 View 转为 Owned 再解构，参见 `21-type.md` §5.5。

### 5.9 内存管理

`into_raw_parts()` 返回的是 Xenon 分配器元信息的完整快照。回收必须遵守 Xenon 的分配契约：要么通过 `Tensor::from_raw_parts_owned(raw)` 重建后交由 Xenon 的 Drop 释放，要么仅使用与该契约等价、且明确以 Xenon 分配器元数据为前提的回收路径；不得把该指针交给系统 `free`、C 侧默认释放器或其他不知晓 `cap` / `align` 的 foreign allocator。正确回收内存的方式如下：

| 规则                 | 说明                                                            |
| -------------------- | --------------------------------------------------------------- |
| ✅ 重建张量后 Drop   | 使用 `Tensor::from_raw_parts_owned(raw)` 重建，让 Drop 处理释放 |
| ❌ 直接调用系统 free | 分配器不匹配，导致 UB 或内存泄漏                                |
| ❌ 忽略返回值        | 内存泄漏                                                        |

**实现归属：** `from_raw_parts_owned()` 的核心实现（含完整验证逻辑与 `TensorBase` 构造）定义于 `07-tensor.md §5.7`（`src/tensor/construct.rs`）。本模块通过 re-export 暴露该方法。完整 `# Safety` 契约及验证步骤详见该文档。

**owned 重建校验说明：** `from_raw_parts_owned()` 虽然仍是 `unsafe`，但必须先验证所有可直接从元数据证明的约束：`offset == 0`、`strides` 等于 canonical F-order、`len == product(shape)`、`cap >= len`、`align` 是对 `A` 有效的 2 的幂对齐。只有指针真实来源、分配器匹配和初始化状态等无法由元数据单独证明的前提继续留给调用方承担。

**裸指针直接构造 Owned 张量的设计约束：** 当前版本不提供从任意裸指针直接构造 `Owned` 张量的接口。`from_raw_parts()` / `from_raw_parts_mut()` 仅构造视图（View / ViewMut），`from_raw_parts_owned()` 仅从 `into_raw_parts()` 导出的 `OwnedRawParts` 重建 Owned 张量。原因是 `Owned` 存储需要 Xenon 分配器的元数据（capacity、alignment），这些信息无法从单一裸指针推断。若调用方需要从裸指针创建 Owned 张量，须先将数据复制到 Xenon 分配的张量中（如通过 `Tensor::from_shape_vec()` 等构造方法）。

```rust,ignore
// Correct round-trip: into_raw_parts → use pointer → from_raw_parts_owned → drop
let tensor = Tensor2::<f64>::zeros([3, 4]);
let raw = tensor.into_raw_parts();

// ... use ptr in FFI code ...

// Reconstruct and let Drop handle deallocation
unsafe {
    // SAFETY: `raw` comes directly from Xenon's `into_raw_parts()` and has not
    // been modified or freed by foreign code.
    let reconstructed = Tensor::<f64, _>::from_raw_parts_owned(raw)
        .expect("owned raw parts should remain valid after round-trip");
    drop(reconstructed);  // Correctly deallocates with Xenon's aligned allocator
}
```

### 5.10 BLAS 兼容性 API

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Checks whether the memory layout is BLAS-compatible.
    ///
    /// # BLAS Compatibility Conditions
    ///
    /// | Condition | Description |
    /// |------|------|
    /// | Contiguity | F-contiguous (Xenon only supports F-order) |
    /// | Positive strides | All strides > 0 |
    /// | No zero strides | No broadcast dimensions |
    ///
    /// # Returns
    ///
    /// `true` if the layout matches Xenon's BLAS memory-layout contract;
    /// `false` if a copy is needed first.
    ///
    /// This method checks layout only. Callers must still verify `ndim() == 2`
    /// and convert `rows`, `cols`, and `lda` to the BLAS/LAPACK backend integer
    /// type expected by the target implementation, typically by calling
    /// `blas_info()` and then `as_blas_int()` on the exported metadata.
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert!(a.is_blas_layout_compatible());
    ///
    /// let info = SliceInfo::new(/* [Range { start: 0, end: 3 }, Range { start: 1, end: 3 }] */)?;
    /// let b = a.slice(info)?;
    /// assert!(!b.is_blas_layout_compatible());
    /// ```
    pub fn is_blas_layout_compatible(&self) -> bool {
        self.is_f_contiguous()      // method name: see 07-tensor.md §5.3
            && !self.has_zero_stride()
    }
}
````

### 5.11 blas_info 和 BlasInfo 结构体

````rust,ignore
use crate::error::{FfiErrorCategory, FfiBackend};

/// BLAS/LAPACK matrix metadata.
///
/// BLAS/LAPACK backends may use different integer widths. Xenon therefore keeps
/// the raw dimensions in `usize` and lets callers convert them to the backend's
/// integer type (`i32` or `i64`) at the FFI boundary.
pub struct BlasInfo<A> {
    /// Data pointer to the logical first element.
    pub data_ptr: *const A,
    /// Leading dimension (element units, raw `usize`).
    pub leading_dim: usize,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl<A> BlasInfo<A> {
    /// Convert a raw BLAS/LAPACK size parameter to the backend integer type.
    ///
    /// `target_width_bits` reports the bit width of `I` (e.g. `32` for `i32`,
    /// `64` for `i64`) so that the structured `FfiErrorCategory::IntegerOverflow`
    /// payload accurately identifies which backend integer type was unable
    /// to represent `value`.
    pub fn as_blas_int<I>(value: usize) -> Result<I, XenonError>
    where
        I: TryFrom<usize>,
    {
        value.try_into().map_err(|_| XenonError::Ffi {
            operation: Cow::Borrowed("ffi::blas_info::as_blas_int"),
            category: FfiErrorCategory::IntegerOverflow {
                value,
                target_width_bits: (core::mem::size_of::<I>() * 8) as u8,
            },
            backend: FfiBackend::Blas,
            cause: None,
        })
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns BLAS layout identifier and parameter information.
    ///
    /// # Returns
    ///
    /// - `Ok(BlasInfo<A>)`: compatibility conditions met; `rows` / `cols` /
    ///   `leading_dim` are returned as raw `usize` metadata
    /// - `Err(XenonError::Ffi { .. })`: returned when the tensor is not 2D or
    ///   not BLAS compatible
    ///
    /// BLAS/LAPACK backend integer widths vary by implementation. `blas_info()`
    /// provides `rows`/`cols`/`leading_dim` as raw `usize` values, along with
    /// an `as_blas_int()` helper to convert them to the backend's integer type
    /// (`i32` or `i64`). Callers choose the appropriate conversion based on
    /// the target backend. `blas_info()` itself does not perform integer-width
    /// conversion; the conversion that may actually fail is the subsequent
    /// `as_blas_int()`.
    ///
    /// This module also provides helpers for LAPACK integration. The leading
    /// dimension and matrix layout information required by LAPACK shares the
    /// same metadata export format as BLAS (`blas_info()` /
    /// `is_blas_layout_compatible()`). LAPACK-specific parameters (e.g.,
    /// pivot indices) are managed by upstream libraries via the raw pointer API.
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// let info = a.blas_info().expect("F-order 2D tensor should be BLAS-compatible");
    /// assert_eq!(info.rows, 3);
    /// assert_eq!(info.cols, 4);
    /// ```
    pub fn blas_info(&self) -> Result<BlasInfo<A>, XenonError> {
        if self.ndim() != 2 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::blas_info"),
                category: FfiErrorCategory::InvalidRank {
                    expected: 2,
                    actual: self.ndim(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        if !self.is_blas_layout_compatible() {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::blas_info"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }

        let rows = self.shape()[0];
        let cols = self.shape()[1];
        // BLAS requires `lda >= max(1, rows)`. For F-order shape `[0, n]`
        // (zero-row matrix), `product(shape) == 0`, so by the authoritative
        // `HAS_ZERO_STRIDE := any(stride == 0) && product(shape) > 0` rule
        // (06-layout.md §5.11) the layout is still classified F_CONTIGUOUS
        // and `is_blas_layout_compatible()` returns `true`. The naturally
        // computed F-order `strides[1]` equals `rows == 0`, which violates
        // BLAS's `lda >= 1` requirement and would induce UB in any BLAS
        // routine. Reject zero-row matrices here as a separate gate so the
        // exported `BlasInfo::leading_dim` is always `>= max(1, rows)`.
        // The mirror rule for zero-column matrices (`cols == 0 && rows > 0`)
        // is naturally satisfied because F-order `strides[1] == rows` already
        // gives `lda == rows >= 1`.
        if rows == 0 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::blas_info"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }

        let data_ptr = self.as_ptr();
        // Post `rows > 0` gate: F-order `strides[1] == rows >= 1`, so
        // `leading_dim` always satisfies BLAS's `lda >= max(1, rows)`.
        let leading_dim = self.strides()[1];

        Ok(BlasInfo {
            data_ptr,
            leading_dim,
            rows,
            cols,
        })
    }
}
````

### 5.12 LDA 查询

````rust,ignore
use crate::error::{FfiErrorCategory, FfiBackend};

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns the leading dimension (only meaningful for 2D arrays).
    ///
    /// For F-order matrix `A[M, N]`, `LDA = stride[1]`.
    /// For zero-column matrices (`cols == 0 && rows > 0`), Xenon returns
    /// `stride[1]` (= rows for F-order) so that `lda >= max(1, rows)` is
    /// still satisfied.
    ///
    /// **Note:** `lda()` is only valid for BLAS-compatible 2D tensors. For non-contiguous tensors (such as sliced views),
    /// the returned stride cannot be used directly in a BLAS call. Check `is_blas_layout_compatible()` first.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)`: LDA of a BLAS-compatible 2D array
    /// - `Err(XenonError::Ffi { .. })`: returned for non-BLAS-compatible 2D input
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert_eq!(a.lda()?, 3);  // F-order, LDA = M = 3
    /// # Ok::<(), xenon::XenonError>(())
    /// ```
    pub fn lda(&self) -> Result<usize, XenonError> {
        if self.ndim() != 2 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::lda"),
                category: FfiErrorCategory::InvalidRank {
                    expected: 2,
                    actual: self.ndim(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        if !self.is_blas_layout_compatible() {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::lda"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        // Mirror `blas_info()` rows-gate: BLAS requires `lda >= max(1, rows)`.
        // For F-order shape `[0, n]`, `product(shape) == 0` ⇒
        // `HAS_ZERO_STRIDE` stays false (06-layout.md §5.11), so the layout
        // is still F_CONTIGUOUS and `is_blas_layout_compatible()` returns
        // true; raw `strides[1] == rows == 0` would violate BLAS's `lda >= 1`.
        // Reject zero-row matrices explicitly.
        if self.shape()[0] == 0 {
            return Err(XenonError::Ffi {
                operation: Cow::Borrowed("ffi::lda"),
                category: FfiErrorCategory::BlasIncompatibleLayout {
                    shape: self.shape().to_vec(),
                    strides: self.strides().to_vec(),
                },
                backend: FfiBackend::Blas,
                cause: None,
            });
        }
        let strides = self.strides();
        // Post rows-gate: F-order `strides[1] == rows >= 1`, satisfying
        // BLAS's `lda >= max(1, rows)`.
        Ok(strides[1])
    }
}
````

### 5.13 多维索引到指针偏移

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Converts a multi-dimensional index to an element offset relative to the
    /// logical first element pointer.
    ///
    /// Offset = Σ(stride[i] * index[i]) for all i in [0, ndim)
    ///
    /// Returns a `usize` offset relative to the logical first element pointer.
    /// Both multiplication and accumulation use checked arithmetic, and any
    /// overflow is reported as a recoverable error rather than panic or wraparound.
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// // shape=[3,4], strides=[1,3], F-order
    /// // index [1, 2] → offset = 1*1 + 2*3 = 7
    /// assert_eq!(tensor.try_offset_of(&[1, 2])?, 7);
    /// # Ok::<(), xenon::XenonError>(())
    /// ```
    pub fn try_offset_of(&self, index: &[usize]) -> Result<usize, XenonError> {
        if index.len() != self.ndim() {
            return Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("ffi::try_offset_of"),
                expected: self.ndim(),
                actual: index.len(),
            });
        }
        let strides = self.strides();
        let shape = self.shape();
        // Build storage_kind tag once: tensor's StorageKind already maps
        // 1:1 to StorageKindTag (Owned/View/ViewMut/Shared), and is needed
        // by every InvalidLayout branch below.
        let storage_kind: StorageKindTag = self.storage_kind().into();
        let mut offset: usize = 0;
        for i in 0..self.ndim() {
            if index[i] >= shape[i] {
                return Err(XenonError::IndexOutOfBounds {
                    operation: Cow::Borrowed("ffi::try_offset_of"),
                    attempted_index: index.to_vec(),
                    axis: i,
                    shape: shape.to_vec(),
                });
            }
            let term = strides[i].checked_mul(index[i]).ok_or_else(|| XenonError::InvalidLayout {
                operation: Cow::Borrowed("ffi::try_offset_of"),
                storage_kind,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset: self.offset(),
                storage_len: self.storage_len(),
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })?;
            offset = offset.checked_add(term).ok_or_else(|| XenonError::InvalidLayout {
                operation: Cow::Borrowed("ffi::try_offset_of"),
                storage_kind,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset: self.offset(),
                storage_len: self.storage_len(),
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })?;
        }
        Ok(offset)
    }

    /// Converts a multi-dimensional index to a raw pointer to the corresponding element.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor1::<i32>::from_shape_vec(Ix1(4), vec![10, 20, 30, 40])?;
    /// let ptr = tensor.try_ptr_at(&[2])?;
    /// assert_eq!(unsafe { *ptr }, 30);
    /// # Ok::<(), xenon::XenonError>(())
    /// ```
    pub fn try_ptr_at(&self, index: &[usize]) -> Result<*const A, XenonError> {
        let offset = self.try_offset_of(index)?;
        // SAFETY: offset is within storage bounds as validated by dimension checks
        Ok(unsafe { self.as_ptr().add(offset) })
    }
}
````

### 5.14 Good/Bad 对比

```rust,ignore
// Good - Check BLAS layout compatibility before passing
if tensor.is_blas_layout_compatible() {
    let info = tensor.blas_info().expect("BLAS-compatible tensor should yield BlasInfo");
    unsafe {
        // SAFETY: `info` came from `blas_info()`, so layout/rank/integer checks passed.
        call_blas_dgemm(CblasColMajor, CblasNoTrans, info, ...);
    }
} else {
    let contiguous = tensor.to_contiguous();
    let info = contiguous.blas_info().expect("contiguous tensor should yield BlasInfo");
    unsafe {
        // SAFETY: `contiguous` is materialized into Xenon's BLAS-compatible layout.
        call_blas_dgemm(CblasColMajor, CblasNoTrans, info, ...);
    }
}

// Bad - Pass directly without checking BLAS layout compatibility
unsafe {
    // SAFETY: This is intentionally incorrect example code.
    call_blas_dgemm(CblasColMajor, CblasNoTrans, ...,
        tensor.as_ptr(), tensor.lda().expect("caller must prove BLAS compatibility first"),
        ...,
    );  // UB if tensor is non-contiguous!
}
```

---

## 6. 内部实现设计

### 6.1 指针有效性论证

`as_ptr()` 和 `as_mut_ptr()` 的返回值有效性由 `TensorBase` 的构造不变量保证——非空张量的 storage base pointer 保证非 null 且有效，`offset` 保证在 storage 范围内。具体来说：

- **Owned 存储**：由 Xenon 的对齐分配器分配，base pointer 保证非 null、对齐且覆盖全部元素；Owned 张量的 `offset` 始终为 0。
- **View / ViewMut 存储**：base pointer 与 offset 由安全构造路径保证在底层 storage 的可访问范围内；若通过 `from_raw_parts()` 构造，则由调用方在 `unsafe` 前提下保证指针有效性与对齐。
- **空张量**：`as_ptr()` / `as_mut_ptr()` 必须返回悬垂但非解引用的 dangling 指针；它们不对空张量做 `.add(offset)` 运算，也不泄露 storage base pointer 作为逻辑首元素指针。

`from_raw_parts` 的 Safety 由调用方保证，但会先执行可直接检查的元数据验证：若 `shape`、`strides`、`offset` 与 `storage_len` 的组合明显非法，则返回 `Err(XenonError::InvalidLayout { .. })`；只有那些库无法从元数据自行证明的指针/生命周期前提，才继续由调用方承担。空张量路径必须跳过 `ptr.add(offset)`，改用非解引用哨兵值参与 flags 计算。

### 6.2 元数据校验算法

`from_raw_parts()` / `from_raw_parts_mut()` 内部调用 `validate_access_range()` 验证元数据合法性。当前版本 stride 全为非负 `usize`，因此 `logical_min` 恒等于 `offset`。算法与 `07-tensor.md` §6.2 保持一致：

```
// `caller_storage_kind: StorageKindTag` is supplied by the caller (View /
// ViewMut / Owned). All `reason` values are closed-enum variants of
// `InvalidLayoutReason` per `26-error.md §5.1`; no free-text reason is used.
validate_access_range(shape, strides, offset, storage_len, caller_storage_kind):
    if shape.checked_size() overflows:
        return Err(XenonError::InvalidLayout {
            operation: Cow::Borrowed("validate_access_range"),
            storage_kind: caller_storage_kind,
            shape, strides, offset, storage_len,
            reason: InvalidLayoutReason::ShapeProductOverflow,
        })

    if shape.checked_size() == Ok(0):
        if offset > storage_len:
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("validate_access_range"),
                storage_kind: caller_storage_kind,
                shape, strides, offset, storage_len,
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })
        return Ok(())

    max_offset = offset

    for axis in 0..ndim:
        if shape[axis] == 0:
            return Ok(())

        span = (shape[axis] - 1).checked_mul(strides[axis])
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: Cow::Borrowed("validate_access_range"),
                storage_kind: caller_storage_kind,
                shape, strides, offset, storage_len,
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })?
        max_offset = max_offset.checked_add(span)
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: Cow::Borrowed("validate_access_range"),
                storage_kind: caller_storage_kind,
                shape, strides, offset, storage_len,
                reason: InvalidLayoutReason::AccessRangeExceedsStorage,
            })?

    if max_offset >= storage_len:
        return Err(XenonError::InvalidLayout {
            operation: Cow::Borrowed("validate_access_range"),
            storage_kind: caller_storage_kind,
            shape, strides, offset, storage_len,
            reason: InvalidLayoutReason::AccessRangeExceedsStorage,
        })

    return Ok(())
```

**溢出安全性说明**：

- `validate_access_range()` 负责在构造阶段验证 `(shape, strides, offset, storage_len)` 整体可表示且不越界。
- `try_offset_of()` 负责在查询阶段对 `stride * index` 与逐项累加执行 checked arithmetic；即使张量本身来自安全构造路径，也不得把查询过程的溢出静默提升为 panic 或 wraparound。
- 这两层校验必须同时存在：前者约束张量元数据，后者约束单次索引转换的错误语义。

### 6.3 可写布局非重叠校验

`from_raw_parts_mut()` 的非重叠校验算法定义于 `07-tensor.md §5.7`（`validate_non_overlapping_layout`）。该校验与 `validate_access_range()` 分工不同：前者解决"会不会越界"，后者解决"会不会别名写入"。两者都属于 `需求说明书 §8` 下可直接验证的安全构造前提，失败时都须返回可恢复错误。FFI 消费者通过 `from_raw_parts_mut()` 间接调用该校验，无需了解算法细节——完整算法描述和保守策略说明参见 `07-tensor.md`。
### 6.4 BLAS 兼容性检查流程

```
is_blas_layout_compatible():
    │
    ├── is_f_contiguous()? ─── No ──→ false
    │
    ├── has_zero_stride()? ── Yes ──→ false
    │
    └── All passed ────────────────→ true

Additional caller-side checks:
    ├── ndim() == 2 ?
    └── rows / cols / lda fit BLAS integer range ?
```

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/ffi/` 模块骨架和辅助类型
  - 文件: `src/ffi/mod.rs`, `src/ffi/types.rs`
  - 内容: 模块声明、re-exports、`FfiErrorCategory`、`BlasInfo` 结构体
  - 测试: `test_blas_info_f_order`, `test_xenon_error_ffi_mapping`
  - 前置: 无
  - 预计: 10 min

### Wave 2: 指针 API

- [ ] **T2**: 提供原始指针访问的 FFI 包装器，并 re-export 裸指针构造/解构 API
  - 文件: `src/ffi/ptr.rs`
  - 内容: re-export `as_ptr()`, `as_mut_ptr()`, `from_raw_parts`, `from_raw_parts_mut`, `into_raw_parts`（定义在 `tensor` 模块），以及 FFI 包装器 `export()` / `export_mut()`
  - 测试: `test_as_ptr_basic`, `test_as_mut_ptr_basic`, `test_from_raw_parts_roundtrip`, `test_into_raw_parts`
  - 前置: T1
  - 预计: 10 min

### Wave 3: BLAS 和索引

- [ ] **T3**: 实现 BLAS 兼容性 API
  - 文件: `src/ffi/blas.rs`
  - 内容: `is_blas_layout_compatible()`, `blas_info()`, `lda()`
  - 测试: `test_is_blas_layout_compatible`, `test_lda_f_order`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现多维索引到指针偏移
  - 文件: `src/ffi/offset.rs`
  - 内容: `try_offset_of()` / `try_ptr_at()` 的可恢复错误路径，以及 checked arithmetic 校验
  - 测试: `test_try_offset_of_various`, `test_try_offset_of_checked_overflow`, `test_try_ptr_at_various`
  - 前置: T1
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                       | 说明                                                             |
| -------- | ------------------------------------------ | ---------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`                   | 验证指针访问、BLAS 兼容检查与 raw-parts 语义                     |
| 集成测试 | `tests/`                                   | 验证 `ffi` 与 `tensor`、`layout`、`storage` 的协同路径           |
| 边界测试 | 同模块测试中标注                           | 覆盖空张量、广播维度、未对齐指针和 BLAS 不兼容布局等边界         |
| 属性测试 | `tests/test_ffi.rs` 或 `tests/property_tests.rs` | 验证 `try_offset_of` / `try_ptr_at` / raw-parts roundtrip 不变量 |

### 8.2 单元测试清单

| 测试函数                                 | 测试内容                                                                                                                              | 优先级 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `test_as_ptr_basic`                      | `as_ptr()` 返回有效指针                                                                                                               | 高     |
| `test_as_mut_ptr_basic`                  | `as_mut_ptr()` 返回有效可写指针                                                                                                       | 高     |
| `test_as_ptr_offset`                     | 指针考虑 offset 后指向正确元素                                                                                                        | 高     |
| `test_is_blas_layout_compatible`         | BLAS 布局兼容性主路径（含兼容/不兼容子场景）                                                                                          | 高     |
| `test_blas_info_f_order`                 | F-order 返回正确 BlasInfo                                                                                                             | 高     |
| `test_blas_info_as_blas_int_overflow`    | `BlasInfo::as_blas_int()` 对接近 `usize::MAX` 的 rows/cols/lda 返回转换错误                                                           | 高     |
| `test_lda_f_order`                       | F-order [3,4] 返回 3                                                                                                                  | 高     |
| `test_lda_non_contiguous`                | 非连续（切片）数组 lda() 返回错误                                                                                                     | 中     |
| `test_from_raw_parts_roundtrip`          | `into_raw_parts → from_raw_parts_owned` 往返一致性                                                                                    | 高     |
| `test_from_raw_parts_mut_roundtrip`      | 可变构造 → 修改 → 读取                                                                                                                | 高     |
| `test_from_raw_parts_mut_reject_overlap` | 可写 raw-parts 构造拒绝地址重叠布局                                                                                                   | 高     |
| `test_into_raw_parts`                    | Owned 张量解构后指针有效                                                                                                              | 高     |
| `test_into_raw_parts_memory_leak`        | 解构后正确释放                                                                                                                        | 中     |
| `test_export_contract`                   | `export()` 导出 `data/shape/strides/offset/ndim` 与源张量元数据一致                                                                   | 高     |
| `test_export_mut_contract`               | `export_mut()` 仅对 `StorageMut` 路径开放，且返回可写导出描述符                                                                       | 高     |
| `test_complex_ffi_abi`                   | `Complex32/Complex64` 的 `#[repr(C)]` 字段顺序、大小与对齐满足 ABI 约定                                                               | 高     |
| `test_bool_ffi_abi`                      | 仅在文档明确支持的 targets/ABI 上验证 `bool` FFI 导出匹配 `_Bool` ABI（1-byte / align 1 / 值域 0/1）；其它目标通过 `#[cfg(...)]` 跳过 | 高     |
| `test_export_empty_tensor_pointer`       | 空张量导出时返回有效对齐但不可解引用的指针，且 shape/strides/offset 仍正确                                                            | 高     |
| `test_ffi_wrapper_catches_panic_doc_example` | 文档示例级验证：上游 `extern "C"` wrapper 使用 `std::panic::catch_unwind` 阻止 panic 穿越 C ABI（参见 §5.4.1）；该测试不要求 Xenon 自身暴露 `extern "C"` 函数 | 高     |
| `test_try_offset_of_various`             | recoverable 索引转换返回正确偏移或错误                                                                                                | 高     |
| `test_try_offset_of_checked_overflow`    | 极端 stride/index 组合返回可恢复错误而非 panic                                                                                        | 高     |
| `test_try_ptr_at_various`                | recoverable 指针转换返回正确指针或错误                                                                                                | 高     |

### 8.3 边界测试场景

| 场景           | 预期行为                                                                                                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 空张量         | `as_ptr()` 对空张量不保证返回可解引用指针；raw-parts 构造需跳过 `ptr.add(offset)`；`export()` / `export_mut()` 返回有效对齐但不可解引用的指针且必须保留正确元数据 |
| 单元素张量     | `as_ptr()` 指向唯一元素                                                                                                                                           |
| 非连续切片     | `is_blas_layout_compatible()` 返回 `false`                                                                                                                        |
| 广播维度       | `is_blas_layout_compatible()` 返回 `false`                                                                                                                        |
| 自别名可写布局 | `from_raw_parts_mut()` 返回 `XenonError::Ffi { category: FfiErrorCategory::OverlapRejected{shape, strides}, backend: FfiBackend::RawParts, .. }` |
| 零尺寸矩阵     | `blas_info()` / `lda()` 在 `rows == 0` 时返回 `BlasIncompatibleLayout` 错误（由 `blas_info()` / `lda()` 内部的 `rows == 0` 显式 gate 拒绝；**注意**：空数组 `product(shape) == 0` 不会触发 `has_zero_stride`，参见 `06-layout.md §5.11` 的 `HAS_ZERO_STRIDE := any(stride == 0) && product(shape) > 0` 权威定义；因此 `is_blas_layout_compatible()` 单独无法过滤 `[0, n]`，需 `blas_info()` 自行 gate）。在 `cols == 0 && rows > 0` 时返回 `strides[1]`（= rows），满足 `lda >= max(1, rows)` |
| 1D 张量        | `lda()` 返回错误                                                                                                                                                  |
| 零维张量       | `try_offset_of(&[])` 返回 `Ok(0)`                                                                                                                                 |
| 未对齐指针     | `from_raw_parts` 的 Safety 文档需说明对齐要求                                                                                                                     |

### 8.4 属性测试不变量

| 不变量                                                                                 | 测试方法                             |
| -------------------------------------------------------------------------------------- | ------------------------------------ |
| `try_ptr_at(idx)` 返回的指针等于基于 `as_ptr()` 和 `try_offset_of(idx)` 计算的期望地址 | 在合法索引集合上逐点比对             |
| `into_raw_parts → from_raw_parts_owned` roundtrip 保持 shape/strides/offset            | 对 F-contiguous owned 张量做往返验证 |
| `is_blas_layout_compatible() == true` 且维度/整数范围合法 ⟹ `blas_info()` 成功         | 以连续二维张量为样本验证             |

### 8.5 集成测试

| 测试文件            | 测试内容                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| `tests/test_ffi.rs` | 指针 API / BLAS 兼容检查 / raw-parts roundtrip 与 `tensor`、`layout`、`storage` 的端到端协同路径 |

### 8.6 Feature gate / 配置测试

| 配置              | 验证点                                                                         |
| ----------------- | ------------------------------------------------------------------------------ |
| 默认配置          | 指针 API、BLAS 兼容性检查与 raw-parts roundtrip 在默认构建下保持既定安全边界。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。                                          |

### 8.7 类型边界 / 编译期测试

| 场景                                                                                | 测试方式                      |
| ----------------------------------------------------------------------------------- | ----------------------------- |
| `into_raw_parts()` 仅对 `Owned` 存储开放                                            | 编译期测试。                  |
| `blas_info()` / `lda()` 仅对 2D BLAS-compatible 张量成功                            | 运行时错误测试与签名检查。    |
| `export_mut()` 仅对 `S: StorageMut` 路径开放，`export()` 覆盖所有 `S: Storage` 路径 | 编译期测试 + 运行时契约断言。 |
| 实际 BLAS/LAPACK 调用与 GPU interop 不属于当前 API                                  | API 缺失断言。                |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                       | 对方模块             | 接口/类型                                | 约定                                                                    |
| -------------------------- | -------------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| `ffi → tensor`             | `tensor`             | 原始指针访问                             | 通过 `TensorBase` 的 storage 获取底层指针，参见 `07-tensor.md` §5       |
| `ffi → layout`             | `layout`             | `is_f_contiguous()` / stride 标志        | BLAS 布局兼容性检查依赖布局查询结果，参见 `06-layout.md` §5.7           |
| `ffi → storage`            | `storage`            | `OwnedRawParts` / allocator metadata     | `into_raw_parts` 导出 owned 存储的完整重建信息，参见 `05-storage.md` §5 |
| `ffi → upstream libraries` | `upstream libraries` | `blas_info()` / `lda()` / `try_ptr_at()` | 向外部 BLAS/FFI 调用方暴露零拷贝参数与可恢复索引转换                    |

### 9.2 数据流描述

```text
Upstream code calls as_ptr() / blas_info() / into_raw_parts()
    │
    ├── ffi reads raw pointer, shape, strides, and offset from tensor / storage
    ├── layout decides BLAS compatibility and leading-dimension preconditions
    ├── raw-parts roundtrip exports full allocator metadata for owned storage
    └── the module exposes zero-copy parameters to the external C / BLAS boundary
```

### 9.3 生命周期与所有权约定

| 操作                                        | 所有权/生命周期语义                                                                                                                                                                                                                    |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `as_ptr()` / `as_mut_ptr()`                 | 返回的指针借用源张量；源张量 drop 后指针立即失效。`as_mut_ptr()` 要求独占 `&mut self`，借用期间不可有其它引用。                                                                                                                        |
| `into_raw_parts()`                          | 消费源张量（`self`），将内存所有权转移给调用方。调用方须按 Xenon 分配契约回收：通过 `from_raw_parts_owned()` 重建张量并由 Drop 释放，或使用与 Xenon 分配器元数据等价的专用回收路径；不得直接调用系统 `free` 或其他 foreign allocator。 |
| `from_raw_parts()` / `from_raw_parts_mut()` | 构造的视图生命周期 `'a` 由调用方在 `unsafe` 前提下保证，须与底层内存的实际存活期一致。视图不拥有内存，drop 时不会释放。                                                                                                                |
| `from_raw_parts_owned()`                    | 接收 `OwnedRawParts` 并重建 Owned 张量，内存所有权回归 Xenon 的 Drop 管理。                                                                                                                                                            |
| `export()` / `export_mut()`                 | 返回的 `TensorExport` / `TensorExportMut` 中 `data`、`shape`、`strides` 均借用源张量内部存储；源张量 drop 后全部指针失效。若上游 C ABI wrapper 捕获到 panic，或 unwinding 可能已 drop 源张量 / owner，则这些 borrowed pointer 必须立即视为失效，C 侧不得继续读取或缓存（参见 §5.4.1）。`export_mut()` 额外要求 `&mut self` 且 `S: StorageMut`，确保独占可写访问。 |

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Recoverable error | `blas_info()` / `lda()` 在 rank 或布局非法时返回 `XenonError::Ffi { operation, category: FfiErrorCategory::InvalidRank{expected,actual} \| BlasIncompatibleLayout{shape,strides}, backend: FfiBackend::Blas, cause: None }`（封闭枚举，字段对齐 `26-error.md §5.1`，**不**使用自由文本 `precondition`/`actual`）；BLAS 整数宽度转换失败由 `BlasInfo::as_blas_int()` 返回 `FfiErrorCategory::IntegerOverflow{value, target_width_bits}`；`from_raw_parts_owned()` 在 owned 元数据非法时返回 `XenonError::InvalidLayout { reason: InvalidLayoutReason::*, storage_kind: StorageKindTag::Owned, .. }`；`try_offset_of()` / `try_ptr_at()` 在 rank 失配时返回 `XenonError::DimensionMismatch{operation, expected, actual}`，在 bounds 越界时返回 `XenonError::IndexOutOfBounds{operation, attempted_index, axis, shape}`，在 checked arithmetic 溢出时返回 `XenonError::InvalidLayout { reason: InvalidLayoutReason::AccessRangeExceedsStorage, storage_kind: 调用方实际 StorageKindTag, .. }`；`from_raw_parts_mut()` 在可写布局自别名时返回 `XenonError::Ffi { operation, category: FfiErrorCategory::OverlapRejected{shape, strides}, backend: FfiBackend::RawParts, cause: None }`（与 `26-error.md §5.1` 协同启用 `OverlapRejected` 子变体）。该路径不再使用 `InvalidLayoutReason::AmbiguousOverlap`——后者保留给非 FFI 入口的 layout 自检（如内部 layout 校验场景）。所有错误变体禁止使用 `Cow<str>` 自由文本作为诊断 payload；结构化负载由 `26-error.md §5.1` 的封闭子枚举承担。 |
| Panic             | 本模块不提供公开 panic-sugar 索引转换 API，也**不**定义 `extern "C"` 导出函数。若上游把 Xenon API 包装为 C ABI，wrapper 必须阻止 Rust panic 穿越 C ABI：使用 `std::panic::catch_unwind` 转换为上游 ABI 错误码，或采用 `panic = "abort"`（参见 §1.2 与 §5.4.1）。`from_raw_parts*()` 中那些无法直接验证的不安全前提若被违反，仍属于 unsafe UB，而非 recoverable error。 |
| 路径一致性        | 指针访问、BLAS 查询与 raw-parts roundtrip 必须共享同一 shape / strides / offset 解释；无 SIMD / 并行分支。                                                                                                                                                                                                                                                                                                               |
| 容差边界          | 不适用。                                                                                                                                                                                                                                                                                                                                                                                                                 |

**错误语义对齐：** FFI 文档仅公开 `try_offset_of()` 与 `try_ptr_at()` 这类 `Result` 接口。索引越界、维度不匹配、偏移溢出和布局自别名都属于 `需求说明书 §27` 下的可恢复错误，不再额外提供 `offset_of()` / `ptr_at()` 之类会把这些条件升级为 panic 的公开 sugar。

---

## 11. 设计决策记录

### 决策 1: BLAS 兼容 API 设计

| 属性     | 值                                                                                      |
| -------- | --------------------------------------------------------------------------------------- |
| 决策     | 提供结构化的 `BlasInfo` 查询方法，而非仅返回布尔值                                      |
| 理由     | 上游库需要完整的 BLAS 参数（data ptr、lda、rows、cols），结构体返回比单独方法调用更便捷 |
| 替代方案 | 仅返回 `bool is_blas_layout_compatible()` — 放弃，上游库需要重复获取多个参数            |
| 替代方案 | 返回 raw C 常量 — 放弃，不符合 Rust 惯例                                                |

**补充**：Xenon 的直接 BLAS 路径只接受 BLAS-compatible 的 F-order 2D 张量。转置或非连续视图必须先显式 materialize 为 `to_contiguous()` 结果，再由调用方结合导出的元数据传入对应的后端常量。

### 决策 2: Safety 独立边界

| 属性     | 值                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 决策     | `from_raw_parts` 和 `from_raw_parts_mut` 使用最小 Safety 模约束集                        |
| 理由     | 将安全责任尽可能交给调用方，库本身不做额外假设；与 `std::slice::from_raw_parts` 设计一致 |
| 替代方案 | 库内部验证所有 Safety 条件 — 放弃，运行时开销过大（O(n) 检查）                           |

### 决策 3: 性能 — 零拷贝优先

| 属性     | 值                                                                                                                          |
| -------- | --------------------------------------------------------------------------------------------------------------------------- |
| 决策     | FFI 方法只做可直接检查的元数据验证，不重复承担指针级 Safety 证明                                                            |
| 理由     | 与 `07-tensor.md` 一致：保留必要的 `shape/stride/offset/storage_len` 校验，同时避免把无法证明的内存前提伪装成库内可验证逻辑 |
| 替代方案 | 完全不校验元数据 — 放弃，会让明显非法输入延迟到 UB；对所有内存前提做深度验证 — 放弃，超出当前边界                           |

**补充**：`try_offset_of()` 在文档层明确要求 checked arithmetic；即使张量本身来自安全构造路径，也不得把索引转换错误表述为“天然不会发生，因此无需检查”。

---

## 12. 性能考量

| 操作                          | 时间复杂度 | 说明                                      |
| ----------------------------- | ---------- | ----------------------------------------- |
| `as_ptr()`                    | O(1)       | 仅指针加法                                |
| `as_mut_ptr()`                | O(1)       | 仅指针加法                                |
| `is_blas_layout_compatible()` | O(1)       | 检查布局标志                              |
| `blas_info()`                 | O(1)       | 包含 `is_blas_layout_compatible()` + 构造 |
| `lda()`                       | O(1)       | 步长查询                                  |
| `try_offset_of()`             | O(ndim)    | 逐轴计算 + 可恢复错误分支                 |
| `try_ptr_at()`                | O(ndim)    | `try_offset_of()` + 指针加法              |
| `from_raw_parts()`            | O(ndim)    | 元数据校验 + 构造视图                     |
| `into_raw_parts()`            | O(1)       | 提取字段 + `ManuallyDrop`                 |

**性能提示**:

- `as_ptr()` 和 `as_mut_ptr()` 应标注 `#[inline]`
- `try_offset_of()` / `try_ptr_at()` 在热路径中可能需要内联
- `is_blas_layout_compatible()` 检查现有 `LayoutFlags`，无需重新计算

---

## 13. 平台与工程约束

| 约束        | 说明                                                                                                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `std` only  | 当前版本仅讨论 `std` 环境下的 FFI 接口；FFI 指针操作依赖 `std` 提供的分配器与布局保证                                                                                            |
| MSRV        | Rust 1.85+                                                                                                                                                                       |
| 单 crate    | FFI 模块位于 `src/ffi/`，不引入额外 crate，保持 Xenon 单 crate 结构                                                                                                              |
| SemVer      | C ABI 稳定契约**仅**覆盖 C-visible raw descriptors：`TensorExportRaw` / `TensorExportMutRaw` 的字段布局与 `#[repr(C)]` 表示，以及 `crate::ffi::ElementType` 的显式 discriminants（参见 `03-element.md §5.1.1`）。Generic descriptors `TensorExport<'a, A>` / `TensorExportMut<'a, A>` 是 `pub(crate)` Rust-only 借用证据（位于 `src/ffi/private.rs`，参见 §3 / §5.3.bis），**不**进入 C ABI 稳定契约面，可在不破坏 SemVer 的前提下变更字段。`OwnedRawParts<A, D>` 是 owned 解构/重建的 Rust API 表面，字段布局变更须遵循 SemVer。新增公共 FFI 方法或 raw descriptor 变体属于 minor 变更 |
| 最小依赖    | 无新增第三方依赖，符合 `需求说明书 §1.3` 对最小依赖的限制                                                                                                                        |
| 索引类型    | 逻辑索引统一使用 `usize`；BLAS/LAPACK 整数参数在边界处按目标后端转换为 `i32` 或 `i64`                                                                                            |
| stride 范围 | 当前版本只接受非负 stride；负步长导入不在范围内                                                                                                                                  |
| 错误诊断    | `blas_info()` / `lda()` 返回 `Result`，保留失败原因                                                                                                                              |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
