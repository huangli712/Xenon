# FFI 接口模块设计

> 文档编号: 23 | 模块: `src/ffi/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `06-layout.md`
> 需求参考: `需求说明书 §5`, `需求说明书 §6`, `需求说明书 §7`, `需求说明书 §8`, `需求说明书 §25`, `需求说明书 §27`, `需求说明书 §28.1`, `需求说明书 §28.4`
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责            | 包含                                                                        | 不包含                                              |
| --------------- | --------------------------------------------------------------------------- | --------------------------------------------------- |
| 原始指针 API    | `as_ptr()`/`as_mut_ptr()`                                                   | BLAS 绑定实现（由上游库通过 `blas-sys` crate 提供） |
| 裸指针构造张量  | `from_raw_parts`/`from_raw_parts_mut`                                       | GPU 内存操作                                        |
| 裸指针解构张量  | `into_raw_parts`                                                            | 跨进程共享内存                                      |
| BLAS 兼容性 API | `is_blas_layout_compatible()` 与 BLAS 元数据导出（`blas_info()` / `lda()`） | 自动调用 BLAS（由上游库负责）                       |
| 多维索引转换    | `try_offset_of()`/`try_ptr_at()`                                            | 序列化/反序列化                                     |

### 1.2 设计原则

| 原则         | 体现                                        |
| ------------ | ------------------------------------------- |
| 零拷贝       | 指针 API 无数据拷贝，O(1) 开销              |
| 安全边界清晰 | 所有 unsafe 函数有详尽 Safety 文档          |
| BLAS 友好    | 提供完整的 BLAS 兼容性检查和布局查询        |
| 最小约束     | FFI 方法避免重复安全检查（调用方已 unsafe） |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: ffi  <- current module
```

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §5`, `需求说明书 §6`, `需求说明书 §7`, `需求说明书 §8`, `需求说明书 §25`, `需求说明书 §27`, `需求说明书 §28.1`, `需求说明书 §28.4` |
| 范围内   | 原始指针访问、raw-parts 往返、BLAS 兼容性查询、多维索引到偏移 / 指针转换。                                                                     |
| 范围外   | 实际 BLAS / LAPACK 例程调用、GPU 互操作、跨进程共享内存与更高层序列化协议。                                                                    |
| 非目标   | 不把 `ffi` 扩展为外部数值库绑定层，不新增第三方 FFI crate 依赖。                                                                               |

| 需求条款     | 本文承接方式                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------- |
| 需求说明书 §5 复数类型  | 明确 `Complex<f32>` / `Complex<f64>` 的稳定 `#[repr(C)]` FFI 表示。                          |
| 需求说明书 §6 存储系统  | `export()` / `export_mut()` 分别覆盖 `Storage` / `StorageMut`，保持零拷贝导出边界。          |
| 需求说明书 §7 内存布局  | 导出与导入统一使用 shape / strides / offset 元数据解释当前版本合法布局。                     |
| 需求说明书 §8 张量类型  | `from_raw_parts*()` 验证可检查的布局、范围与别名条件，失败时返回可恢复错误。                 |
| 需求说明书 §25 FFI 集成 | 提供原始指针、偏移转换、BLAS 兼容性查询和 raw-parts roundtrip。                              |
| 需求说明书 §27 错误处理 | 仅公开 `try_offset_of()` / `try_ptr_at()` 这类 `Result` API，不额外暴露 panic sugar。        |
| 需求说明书 §28.1 文档   | 所有 unsafe 入口提供 Safety 说明；关键 FFI API 提供示例，非完整上下文示例统一标记 `ignore`。 |

---

## 3. 文件位置

```
src/
└── ffi/
    ├── mod.rs         # Module root, re-exports
    ├── types.rs       # FfiError and BlasInfo type definitions
    ├── ptr.rs         # Raw-pointer APIs (as_ptr, as_mut_ptr, from_raw_parts, from_raw_parts_mut, into_raw_parts)
    ├── blas.rs        # BLAS compatibility checks (is_blas_layout_compatible, blas_info, lda)
    └── offset.rs      # Multi-dimensional index to pointer offset (try_offset_of, try_ptr_at)
```

多文件设计：将 FFI 按职责拆分为多个文件，便于后期拓展和维护。

| 文件        | 职责                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------- |
| `mod.rs`    | 模块入口，导出公共 API                                                                      |
| `types.rs`  | `FfiError` 枚举、`BlasInfo` 结构体                                                          |
| `ptr.rs`    | 原始指针访问（`as_ptr`/`as_mut_ptr`）和裸指针构造/解构（`from_raw_parts`/`into_raw_parts`） |
| `blas.rs`   | BLAS 兼容性检查和参数查询（`is_blas_layout_compatible`/`blas_info`/`lda`）                  |
| `offset.rs` | 多维索引到偏移量和指针转换（`try_offset_of`、`try_ptr_at`）                                 |

---

## 4. 依赖关系

### 4.1 依赖图

```
src/ffi/
├── mod.rs
│   └── re-exports from types, ptr, blas, offset
├── types.rs
│   └── (no external dependency, only core)
├── ptr.rs
│   ├── crate::tensor        # TensorBase<S, D>, offset
│   ├── crate::dimension     # Dimension trait
│   ├── crate::storage       # Storage, StorageMut, owned allocator metadata
│   └── crate::layout        # is_f_contiguous
├── blas.rs
│   ├── crate::tensor        # TensorBase<S, D>
│   ├── crate::storage       # Storage
│   ├── crate::layout        # is_f_contiguous, has_zero_stride
│   ├── super::types         # BlasInfo, FfiError
│   └── super::ptr           # as_ptr
└── offset.rs
    ├── crate::tensor        # TensorBase<S, D>
    ├── crate::dimension     # Dimension trait
    └── crate::storage       # Storage<Elem=A>
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                        | 参考                 | 使用者                           |
| ----------- | ------------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `.shape()`, `.strides()`, `.offset()`                                               | `07-tensor.md` §5    | `ptr.rs`, `blas.rs`, `offset.rs` |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`                                                                       | `02-dimension.md` §5 | `ptr.rs`, `offset.rs`            |
| `storage`   | `Storage<Elem=A>`, `StorageMut<Elem=A>`, owned allocator metadata（供 `OwnedRawParts<A, D>` 导出/重建） | `05-storage.md` §5   | `ptr.rs`, `blas.rs`, `offset.rs` |
| `layout`    | `is_f_contiguous()`, `has_zero_stride()`                                                                | `06-layout.md` §5    | `ptr.rs`, `blas.rs`              |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `ffi` 仅消费 `tensor`、`storage` 等核心模块，为上游库提供接口。

> **owner 约定：** `as_ptr()` / `as_mut_ptr()` 的核心定义在 `07-tensor.md`（tensor 核心层）；`ffi` 模块负责指针导出格式（`TensorExport`）、BLAS 辅助 API 和裸指针构造。本文聚焦这些能力在 FFI 边界的公开形态，因此依赖表中仍把相关实现文件归入 `ffi` 模块文档范围，而不把它写成反向依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

---

## 5. 公共 API 设计

### 5.1 辅助类型

```rust,ignore
/// Internal error classification for FFI-specific failures.
///
/// # Diagnostics Design
///
/// `FfiError` uses `&'static str` for the `operation`, `backend`, and
/// `precondition` fields to maintain zero-allocation, compile-time-known
/// structured formatting. The `actual` field uses `Cow<'static, str>` to
/// accommodate runtime dynamic context (e.g., actual ndim value, shape, etc.).
/// This design satisfies `需求说明书 §27` diagnostics requirements: the error
/// category is identified by the enum variant, and the triggering context is
/// carried by `actual`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FfiError {
    InvalidRank {
        operation: &'static str,
        backend: &'static str,
        precondition: &'static str,
        actual: alloc::borrow::Cow<'static, str>,
    },
    BlasIncompatibleLayout {
        operation: &'static str,
        backend: &'static str,
        precondition: &'static str,
        actual: alloc::borrow::Cow<'static, str>,
    },
    IntegerOverflow {
        operation: &'static str,
        backend: &'static str,
        precondition: &'static str,
        actual: alloc::borrow::Cow<'static, str>,
    },
}

/// Internal ffi-module errors are converted to the public
/// `XenonError::Ffi { reason: String }` variant before crossing
/// Xenon's public API boundary.
impl From<FfiError> for XenonError {
    fn from(value: FfiError) -> Self {
        XenonError::Ffi {
            reason: value.to_string(),
        }
    }
}

impl core::fmt::Display for FfiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for FfiError {}
```

> **公开边界说明：** `FfiError` 是 `ffi` 模块内部使用的 `pub(crate)` 错误分类，不对 crate 外 re-export。所有公共 FFI 相关错误都必须先在模块内部构造 `FfiError`，再于公开边界统一转换为 `XenonError::Ffi { reason }` 返回；公开 API 不得出现 `FfiError` 类型名或 `XenonError::Ffi(FfiError)` 这类签名。
>
> **诊断一致性说明：** `需求说明书 §27` 倾向公开恢复性错误携带结构化上下文；当前 `XenonError::Ffi { reason }` 仍是 opaque 包装，而结构化字段保留在模块内部 `FfiError`。是否把这些字段提升到公开 `XenonError` 变体属于统一错误枚举的设计变更，需与 `26-error.md` 一并评审，本次仅在文档中显式记录该边界。

### 5.2 原始指针 API

> **结果类型说明：** 公开 API 统一使用 `Result<T, XenonError>`，`crate::error::Result<_>` 为等价类型别名。

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a read-only raw pointer to the logical first element.
    ///
    /// The pointer returned here points to the first logical element.
    /// Internally, storage keeps the storage base pointer and TensorBase applies
    /// `offset` exactly once when exposing the logical-first pointer.
    /// The returned pointer is invalid after `self` is modified or dropped.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_ptr();
    /// // Can be passed to read-only C functions
    /// ```
    pub fn as_ptr(&self) -> *const A {
        if self.is_empty() {
            // For empty tensors, return a non-dereferenceable dangling pointer.
            // Do NOT call .add() on a potentially dangling base pointer.
            return core::ptr::NonNull::<A>::dangling().as_ptr();
        }
        // SAFETY: non-empty tensor guarantees storage base pointer is valid
        // and offset is within bounds by TensorBase construction invariants.
        unsafe {
            self.storage.as_ptr().add(self.offset)
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Returns a mutable raw pointer to the logical first element.
    ///
    /// Only available for writable storage (Owned, ViewMut).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_mut_ptr();
    /// // Can be passed to C functions requiring a mutable pointer
    /// ```
    pub fn as_mut_ptr(&mut self) -> *mut A {
        if self.is_empty() {
            return core::ptr::NonNull::<A>::dangling().as_ptr();
        }
        unsafe {
            self.storage.as_mut_ptr().add(self.offset)
        }
    }
}
````

#### C 侧结构化导出格式

```rust,ignore
/// Element type discriminant for FFI consumers.
///
/// Each variant corresponds to one of Xenon's supported tensor element types
/// (see `需求说明书 §4`). C consumers use this to interpret the `data` pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum ElementType {
    Bool,
    I32,
    I64,
    F32,
    F64,
    Complex32,
    Complex64,
}

impl ElementType {
    /// Returns the `ElementType` discriminant for `A`.
    ///
    /// This is determined at compile time via `Element` trait association.
    pub const fn of<A: Element>() -> Self;
}
```

> **实现基础说明：** 可在 `Element` sealed trait 中引入 `const ELEMENT_TYPE: ElementType` 关联常量作为 `ElementType::of::<A>()` 的实现基础。若当前 Rust 版本不支持所需 const 机制，可将该 API 降为普通 `fn`，保持语义不变。

### 5.2.1 指针约定对照

| API                         | 基准                 | 说明                                                                 |
| --------------------------- | -------------------- | -------------------------------------------------------------------- |
| `as_ptr()` / `as_mut_ptr()` | 逻辑首元素           | 对非空张量返回第一个逻辑元素的指针；空张量返回 dangling              |
| `TensorExport.data`         | storage base pointer | 非空张量时等于底层存储的基地址；空张量时为有效对齐但不可解引用的指针 |
| `BlasInfo.data_ptr`         | 逻辑首元素           | 等价于 `as_ptr()`                                                    |
| `try_ptr_at(indices)`       | 指定逻辑位置         | 基于 `as_ptr() + offset` 计算                                        |

> **指针语义统一**：`TensorExport.data` 和 `TensorExportMut.data` 指向底层存储的基地址（storage base pointer），与 `from_raw_parts()` 的 `ptr` 参数语义一致。逻辑首元素地址可通过 `base_ptr + offset` 计算。

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
#[repr(C)]
pub struct TensorExport<'a, A> {
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
    /// Lifetime marker tying the export to the source tensor borrow.
    pub _marker: core::marker::PhantomData<&'a A>,
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
}

/// Raw mutable tensor data export for FFI consumers.
#[repr(C)]
pub struct TensorExportMut<'a, A> {
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
    pub data: *mut A,
    /// Lifetime marker tying the export to the source tensor borrow.
    pub _marker: core::marker::PhantomData<&'a mut A>,
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
}

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
    pub fn export(&self) -> TensorExport<'_, A>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    A: Element,
    D: Dimension,
{

    /// Export tensor data with mutable access.
    ///
    /// This API is only implemented for writable storage, so read-only storage
    /// modes are rejected at the trait boundary rather than at runtime.
    /// No additional fallible validation is performed beyond the existing
    /// `&mut self` + `S: StorageMut` exclusivity boundary.
    pub fn export_mut(&mut self) -> TensorExportMut<'_, A>;
}
```

> **导出语义说明：** `TensorExport<'a, A>` / `TensorExportMut<'a, A>` 面向 C 调用方提供"指针 + shape + strides + storage_len + offset"的结构化快照，其中 `shape` 与 `strides` 指针均借用源张量内部元数据，不能在源张量释放后继续使用；生命周期参数和 `PhantomData` 明确表达“源张量必须活得比导出结构更久”。

> **导出范围说明：** `export()` 提供只读结构化导出并返回 `TensorExport<'_, A>`，适用于任意 `TensorBase<S, D>` 且仅要求 `S: Storage`，因此覆盖 Owned、View、只读共享存储以及所有合法 stride 布局。`export_mut()` 直接返回 `TensorExportMut<'_, A>`，适用于任意满足 `S: StorageMut` 的 `TensorBase<S, D>`，因此同时覆盖 Owned 与 `ViewMut` 这两类可写存储。

> **可写导出边界：** `export_mut()` 通过 `&mut self` 和 `S: StorageMut` 保证 Xenon 侧的独占可写访问；只读视图和共享只读存储则在 trait 边界上直接被拒绝。这与 `需求说明书 §6` 的存储模式转换和 `需求说明书 §25` 的零拷贝导出要求保持一致。

> **空张量约定：** 空张量上的 `as_ptr()` / `as_mut_ptr()` 必须返回有效对齐但不可解引用的 dangling 指针。导出时 `TensorExport*.data` 也遵循相同约定：它始终表示 storage base pointer；当 `len() == 0` 时该位置没有可解引用元素，因此调用方必须先基于长度判断是否可访问。

> **指针语义补充：** `TensorExport<'_, A>::data` 是 `*const A`，`TensorExportMut<'_, A>::data` 是 `*mut A`，二者语义上都始终指向 storage base pointer。对非空张量，逻辑首元素地址需通过 `data.add(offset)` 计算；当 `offset == 0` 时，它与 storage base 重合。`offset` 与 `strides` 都以"元素个数"计量，而不是字节数。

> **stride 约定：** `strides` 以"元素个数"而非字节数表示步长，类型为 `usize`。按照 `06-layout.md` §1.2 与 `需求说明书 §7`，当前版本 Xenon 不支持负步长，因此 FFI 导出格式也不保留负 stride 语义。`from_raw_parts()` 允许零步长布局以表达广播只读视图；`from_raw_parts_mut()` 则拒绝所有非空零步长布局（即任何非单元素轴的 `stride == 0` 都会报错）。

> **offset 约定：** `offset` 记录与 `07-tensor.md` raw-parts 契约一致的逻辑偏移元数据，单位始终是元素个数而不是字节数。导出结构中的 `data` 始终指向 storage base pointer，C 调用方应通过 `data + offset` 还原逻辑首元素地址；`offset` 本身仍仅用于视图重建、范围校验和与 Xenon 原始布局元数据对齐。

> **ndim 一致性约定：** C 消费者须以 `ndim` 为 `shape` 和 `strides` 数组的长度，不得以硬编码长度或其它来源替代。`TensorExport` 的构造保证 `shape` 和 `strides` 指向的数组长度均等于 `ndim`。

> **生命周期与借用语义：** 导出结果不拥有底层内存；一旦源张量被 drop，`TensorExport` 内的 `data`、`shape`、`strides` 全部立即失效。应将该导出结果视为对源张量当前元数据与指针状态的借用快照：`export()` 暴露只读快照，`export_mut()` 暴露独占可写快照；无论是否跨 FFI 边界缓存，该快照都不得超出源张量的生命周期，也不得绕过 `&mut self` 所表达的独占写语义。本文不额外指定 `TensorExport<'_, _>` / `TensorExportMut<'_, _>` 的 auto trait 组合，线程相关性质以其字段与 Rust auto-trait 推导结果为准；测试计划应通过编译期 `Send` / `Sync` 检查验证该自动推导结果。

#### 5.2.2 Complex FFI 布局契约

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

> **Complex 布局约定：** `Complex<f32>` 与 `Complex<f64>` 的 FFI 表示分别等价于 `#[repr(C)] struct { re: f32, im: f32 }` 和 `#[repr(C)] struct { re: f64, im: f64 }`。

> **内存保证：** `#[repr(C)]` 保证字段顺序固定为 `re` 后接 `im`，整体对齐分别等于 `f32` / `f64` 的 C ABI 对齐要求；若目标 ABI 需要尾部 padding，则该 padding 仅作用于单个复数元素末尾，不改变数组按该结构逐元素重复排布的语义。

> **导出语义：** 导出复数张量时，`TensorExport<Complex<f32>>` / `TensorExport<Complex<f64>>` 和 `TensorExportMut<Complex<f32>>` / `TensorExportMut<Complex<f64>>` 中的 `data` 仍是“复数元素指针”，`offset` 与 `strides` 仍按“复数元素个数”计量，而不是按标量 `re/im` 分量或字节计量。C 侧看到的是 `Complex32*` / `Complex64*` 加上相同的 shape/stride 元数据。

#### 5.2.3 Bool FFI 布局契约

> **Bool ABI 约束：** `bool` 与 C `_Bool` / C23 `bool` 的互操作仅在文档明确支持的平台/ABI 下成立；它用于说明当前支持目标上的对接方式，不作为跨语言、跨目标的稳定 ABI 承诺。对这些已支持平台，C 消费者应使用 `_Bool` 或 `bool`（C23）来匹配 `TensorExport<bool>` / `TensorExportMut<bool>` 中的 `data` 指针类型，并避免使用 `int`、`unsigned char` 等其它整数类型。

> **导出语义：** 导出 `bool` 张量时，`TensorExport<bool>` 中的 `data` 为 `*const bool`（C 侧 `const _Bool*`），`TensorExportMut<bool>` 中的 `data` 为 `*mut bool`（C 侧 `_Bool*`），`offset` 与 `strides` 按 `bool` 元素个数计量。`strides[i] == 1` 表示相邻逻辑元素在内存中连续排列（每个占 1 字节）。

> **C 侧验证说明：** Xenon 仅对文档明确支持的平台/ABI 给出 Rust `bool` 与 C `_Bool` 的互操作说明；跨语言集成时，调用方仍应在目标工具链侧通过 `sizeof(_Bool) == 1`、`_Alignof(_Bool) == 1` 等静态断言验证兼容性，不应把该文档表述解读为跨平台稳定 ABI 保证。

> **测试边界说明：** 与上述 ABI 约束一致，`bool` FFI ABI 相关测试也只应在文档明确支持的 targets/ABI 上启用；其它目标上应通过 `#[cfg(...)]` 跳过，而不是把 `_Bool` 兼容性断言提升为无条件测试基线。

### 5.3 从裸指针构造张量

````rust,ignore
impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D>
where
    D: Dimension,
{
    /// Constructs an immutable view from raw pointer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Storage base pointer (immutable)
    /// * `shape` - Length of each axis
    /// * `strides` - Strides per axis (element units, non-negative)
    /// * `offset` - Data start offset (element units)
    ///
    /// # Returns
    ///
    /// `Ok(TensorView<'a, A, D>)` when all directly checkable metadata
    /// constraints pass; otherwise `Err(XenonError::InvalidLayout { .. })`.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following:
    ///
    /// | Prerequisite | Description |
    /// |----------|------|
    /// | Pointer validity | For non-empty tensors, `ptr` must be non-null, non-dangling, and aligned to `align_of::<A>()`; empty tensors may use a non-dereferenceable sentinel pointer |
    /// | Memory range | Memory starting from the storage base pointer `ptr` must cover all accessible elements (considering offset, shape, strides) |
    /// | Lifetime | Memory must remain valid for lifetime `'a` |
    /// | Aliasing rules | Memory can be read-shared but must not be written to |
    /// | Layout consistency | `shape` and `strides` lengths must match |
    /// | Element initialization | All accessible elements must be properly initialized |
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data: [f64; 12] = [0.0; 12];
    /// let view = unsafe {
    ///     // SAFETY: `data` lives for the whole view lifetime, is properly aligned,
    ///     // and the metadata describes the full accessible range.
    ///     TensorView2::from_raw_parts(
    ///         data.as_ptr(),
    ///         data.len(),
    ///         [3, 4],
    ///         Strides::from_slice(&[1, 3]),
    ///         0,
    ///     )
    ///     .expect("metadata should describe a valid view")
    /// };
    /// ```
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self, XenonError> {
        validate_access_range(&shape, &strides, offset, storage_len)?;
        let logical_ptr = if shape.size() == 0 {
            // Empty tensors must not do pointer arithmetic on a possibly dangling
            // storage-base sentinel. Use a well-defined non-dereferenceable value.
            core::ptr::NonNull::<A>::dangling().as_ptr()
        } else {
            unsafe { ptr.add(offset) }
        };

        // SAFETY: Caller still guarantees pointer validity, alignment,
        // initialization, actual accessible range, and lifetime. The constructor
        // only rejects metadata combinations it can check directly.
        Ok(TensorBase {
            storage: ViewRepr::new(ptr, storage_len),
            shape,
            strides,
            offset,
            flags: layout::compute_layout_flags(&shape, &strides, logical_ptr),
        })
    }

}

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    D: Dimension,
{
    /// Constructs a mutable view from raw pointer.
    ///
    /// Same as `from_raw_parts`, but requires exclusive access (no other references).
    /// `ptr` is still the storage base pointer rather than the logical-first pointer.
    ///
    /// # Safety
    ///
    /// Same as `from_raw_parts`, with additional requirement: no other references to the memory,
    /// and the logical element set described by `(shape, strides, offset)` must not alias itself.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut data: [f64; 12] = [0.0; 12];
    /// let view = unsafe {
    ///     // SAFETY: `data` is uniquely borrowed for the view lifetime, properly
    ///     // aligned, and the metadata describes a non-overlapping writable range.
    ///     TensorViewMut2::from_raw_parts_mut(
    ///         data.as_mut_ptr(),
    ///         data.len(),
    ///         [3, 4],
    ///         Strides::from_slice(&[1, 3]),
    ///         0,
    ///     )
    ///     .expect("metadata should describe a valid mutable view")
    /// };
    /// ```
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self, XenonError> {
        validate_access_range(&shape, &strides, offset, storage_len)?;
        if shape.size() != 0 && shape.iter().zip(strides.iter()).any(|(&axis_len, &stride)| axis_len > 1 && stride == 0) {
            return Err(XenonError::InvalidLayout {
                operation: "ffi::from_raw_parts_mut".into(),
                storage_kind: "view_mut".into(),
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset,
                storage_len,
                reason: "mutable raw-parts view must not contain zero strides on non-singleton axes".into(),
            });
        }
        validate_non_overlapping_layout(&shape, &strides, offset, storage_len)?;
        let logical_ptr = if shape.size() == 0 {
            // Empty tensors must not do pointer arithmetic on a possibly dangling
            // storage-base sentinel. Use a well-defined non-dereferenceable value.
            core::ptr::NonNull::<A>::dangling().as_ptr()
        } else {
            unsafe { ptr.add(offset) }
        };

        // SAFETY: Caller guarantees exclusive mutable access to the memory
        // for lifetime 'a. Same validity requirements as from_raw_parts.
        Ok(TensorBase {
            storage: ViewMutRepr::new(ptr, storage_len),
            shape,
            strides,
            offset,
            flags: layout::compute_layout_flags(&shape, &strides, logical_ptr),
        })
    }
}
````

> **校验边界说明：** 与 `07-tensor.md` §5.6 一致，`from_raw_parts*()` 只验证库能够直接检查的元数据约束（例如 shape/stride/offset/storage*len 组合是否合法、是否溢出、是否越界），并在失败时返回 `Result<*, XenonError>`。指针有效性、对齐、实际可访问范围与生命周期仍由调用方在 `unsafe` 前提下负责。

> **空张量补充：** `ptr.add(offset)` 形式的逻辑首元素地址计算只适用于非空张量；空张量路径必须跳过该指针运算，并改用 `NonNull::dangling()` 这类明确定义的非解引用哨兵值参与 flags / metadata 初始化。

> **可写视图补充：** `from_raw_parts_mut()` 不仅必须拒绝所有非空零步长布局（任何非单元素轴的 `stride == 0`），还必须拒绝一切能被高效保守判定为潜在自别名的布局。实现上先用 `validate_access_range()` 验证越界与可表示性，再用 `validate_non_overlapping_layout(shape, strides, offset, storage_len)` 对受支持的正步长布局做保守非重叠判定；若布局超出该高效判定范围，也必须返回可恢复错误，而不是枚举全部可达 offset。

### 5.4 将张量解构为裸指针

````rust,ignore
pub struct OwnedRawParts<A, D> {
    pub ptr: *mut A,
    pub len: usize,
    pub cap: usize,
    pub align: usize,
    pub shape: D,
    pub strides: Strides<D>,
    pub offset: usize,
}

impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// Consumes the tensor, returning owned raw parts.
    ///
    /// # Returns
    ///
    /// An `OwnedRawParts<A, D>` snapshot containing the pointer plus the allocator
    /// metadata required to reconstruct Xenon's aligned owned storage.
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let raw = tensor.into_raw_parts();
    /// // Reconstruct with Tensor::from_raw_parts_owned(raw) and let Drop free it.
    /// ```
    pub fn into_raw_parts(self) -> OwnedRawParts<A, D> {
        let this = core::mem::ManuallyDrop::new(self);
        OwnedRawParts {
            ptr: unsafe { this.storage.as_mut_ptr() },
            len: this.storage.len(),
            cap: this.storage.capacity(),
            align: this.storage.alignment(),
            shape: this.shape.clone(),
            strides: this.strides.clone(),
            offset: this.offset,
        }
    }
}
````

> **设计决策：** `into_raw_parts` 仅适用于 Owned 存储，且导出的内存布局必须满足 Xenon 的 owned 不变量：F-order contiguous、`offset == 0`、canonical F-order strides。若调用方持有的是 view 或带 offset 的逻辑子视图，必须先显式物化为新的 owned contiguous tensor，再跨越 FFI 边界导出裸指针。如需将 View 转为 Owned 再解构，参见 `21-type.md` §5.6。

#### 内存管理

`into_raw_parts()` 返回的是 Xenon 分配器元信息的完整快照。回收必须遵守 Xenon 的分配契约：要么通过 `Tensor::from_raw_parts_owned(raw)` 重建后交由 Xenon 的 Drop 释放，要么仅使用与该契约等价、且明确以 Xenon 分配器元数据为前提的回收路径；不得把该指针交给系统 `free`、C 侧默认释放器或其他不知晓 `cap` / `align` 的 foreign allocator。正确回收内存的方式如下：

| 规则                 | 说明                                                            |
| -------------------- | --------------------------------------------------------------- |
| ✅ 重建张量后 Drop   | 使用 `Tensor::from_raw_parts_owned(raw)` 重建，让 Drop 处理释放 |
| ❌ 直接调用系统 free | 分配器不匹配，导致 UB 或内存泄漏                                |
| ❌ 忽略返回值        | 内存泄漏                                                        |

```rust,ignore
/// Reconstructs an owned tensor from raw parts obtained via `into_raw_parts`.
/// Takes ownership of memory allocated by Xenon's aligned allocator.
///
/// # Safety
///
/// - `raw.ptr` must point to memory allocated by Xenon's aligned allocator
/// - `raw.len`, `raw.cap`, and `raw.align` must be the original allocator metadata
/// - `raw.shape` and `raw.strides` must describe a valid, non-overlapping canonical F-order layout
/// - `raw.offset` must be 0 for owned raw parts
/// - The caller transfers ownership; do NOT free `raw.ptr` separately
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    pub unsafe fn from_raw_parts_owned(
        raw: OwnedRawParts<A, D>,
    ) -> Result<Self, XenonError> {
    if raw.offset != 0 {
        return Err(XenonError::InvalidLayout {
            operation: "ffi::from_raw_parts_owned".into(),
            storage_kind: "owned".into(),
            shape: raw.shape.to_vec(),
            strides: raw.strides.to_vec(),
            offset: raw.offset,
            storage_len: raw.len,
            reason: "owned raw parts must use offset == 0".into(),
        });
    }
    let expected_len = raw.shape.size();
    if raw.len != expected_len {
        return Err(XenonError::InvalidLayout {
            operation: "ffi::from_raw_parts_owned".into(),
            storage_kind: "owned".into(),
            shape: raw.shape.to_vec(),
            strides: raw.strides.to_vec(),
            offset: raw.offset,
            storage_len: raw.len,
            reason: "raw.len must equal product(shape)".into(),
        });
    }
    if raw.cap < raw.len {
        return Err(XenonError::InvalidLayout {
            operation: "ffi::from_raw_parts_owned".into(),
            storage_kind: "owned".into(),
            shape: raw.shape.to_vec(),
            strides: raw.strides.to_vec(),
            offset: raw.offset,
            storage_len: raw.len,
            reason: "raw.cap must be >= raw.len".into(),
        });
    }
    if !raw.align.is_power_of_two() || raw.align < core::mem::align_of::<A>() {
        return Err(XenonError::InvalidLayout {
            operation: "ffi::from_raw_parts_owned".into(),
            storage_kind: "owned".into(),
            shape: raw.shape.to_vec(),
            strides: raw.strides.to_vec(),
            offset: raw.offset,
            storage_len: raw.len,
            reason: "raw.align must be a valid power-of-two alignment for A".into(),
        });
    }
    let expected_strides = layout::canonical_f_strides(&raw.shape);
    if raw.strides != expected_strides {
        return Err(XenonError::InvalidLayout {
            operation: "ffi::from_raw_parts_owned".into(),
            storage_kind: "owned".into(),
            shape: raw.shape.to_vec(),
            strides: raw.strides.to_vec(),
            offset: raw.offset,
            storage_len: raw.len,
            reason: "owned raw parts must use canonical F-order strides".into(),
        });
    }

    let storage = Owned::from_raw_parts(raw.ptr, raw.len, raw.cap, raw.align);
    let flags = layout::compute_layout_flags(&raw.shape, &raw.strides, raw.ptr);
    Ok(TensorBase { storage, shape: raw.shape, strides: raw.strides, offset: raw.offset, flags })
    }
}
```

> **owned 重建校验说明：** `from_raw_parts_owned()` 虽然仍是 `unsafe`，但必须先验证所有可直接从元数据证明的约束：`offset == 0`、`strides` 等于 canonical F-order、`len == product(shape)`、`cap >= len`、`align` 是对 `A` 有效的 2 的幂对齐。只有指针真实来源、分配器匹配和初始化状态等无法由元数据单独证明的前提继续留给调用方承担。

> **裸指针直接构造 Owned 张量的设计约束：** 当前版本不提供从任意裸指针直接构造 `Owned` 张量的接口。`from_raw_parts()` / `from_raw_parts_mut()` 仅构造视图（View / ViewMut），`from_raw_parts_owned()` 仅从 `into_raw_parts()` 导出的 `OwnedRawParts` 重建 Owned 张量。原因是 `Owned` 存储需要 Xenon 分配器的元数据（capacity、alignment），这些信息无法从单一裸指针推断。若调用方需要从裸指针创建 Owned 张量，须先将数据复制到 Xenon 分配的张量中（如通过 `Tensor::from_shape_vec()` 等构造方法）。

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

### 5.5 BLAS 兼容性 API

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

### 5.6 blas_info 和 BlasInfo 结构体

````rust,ignore
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
    pub fn as_blas_int<I>(&self, value: usize) -> Result<I, XenonError>
    where
        I: TryFrom<usize>,
    {
        value.try_into().map_err(|_| FfiError::IntegerOverflow {
            operation: "ffi::blas_info",
            backend: "blas/lapack",
            precondition: "BLAS/LAPACK integer parameter must fit target backend type",
            actual: alloc::format!("value={}", value).into(),
        }.into())
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
    /// - `Err(XenonError)`: wraps internal `FfiError` when not BLAS compatible,
    ///   or not 2D
    ///
    /// BLAS/LAPACK 后端的整数宽度因实现而异。`blas_info()` 提供
    /// `rows`/`cols`/`leading_dim` 的原始 `usize` 值，并提供
    /// `as_blas_int()` 辅助方法将其转换为后端所需的整数类型（`i32` 或
    /// `i64`）。调用方根据目标后端选择合适的转换。`blas_info()` 本身
    /// 不执行该整数宽度转换；真正可能失败的是后续的 `as_blas_int()`。
    ///
    /// 本模块同时提供面向 LAPACK 集成的辅助能力。LAPACK 所需的
    /// leading dimension、矩阵布局信息与 BLAS 共享同一套 metadata 导出格式
    /// （`blas_info()` / `is_blas_layout_compatible()`）。LAPACK 特有的参数
    /// （如 pivot indices）由上游库通过 raw pointer API 自行管理。
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
            return Err(FfiError::InvalidRank {
                operation: "ffi::blas_info",
                backend: "blas",
                precondition: "tensor must be 2D",
                actual: alloc::format!("ndim={}", self.ndim()).into(),
            }.into());
        }
        if !self.is_blas_layout_compatible() {
            return Err(FfiError::BlasIncompatibleLayout {
                operation: "ffi::blas_info",
                backend: "blas",
                precondition: "F-contiguous 2D tensor without zero strides",
                actual: alloc::format!("shape={:?}, strides={:?}", self.shape(), self.strides()).into(),
            }.into());
        }

        let data_ptr = self.as_ptr();
        let lda = self.lda()?;
        let rows = self.shape()[0];
        let cols = self.shape()[1];

        Ok(BlasInfo {
            data_ptr,
            leading_dim: lda,
            rows,
            cols,
        })
    }
}
````

### 5.7 LDA 查询

````rust,ignore
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns the leading dimension (only meaningful for 2D arrays).
    ///
    /// For F-order matrix `A[M, N]`, `LDA = stride[1]`.
    /// For zero-size matrices, Xenon returns `1` so that callers can satisfy
    /// the common BLAS requirement `lda >= max(1, rows)`.
    ///
    /// **Note:** `lda()` is only valid for BLAS-compatible 2D tensors. For non-contiguous tensors (such as sliced views),
    /// the returned stride cannot be used directly in a BLAS call. Check `is_blas_layout_compatible()` first.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)`: LDA of a BLAS-compatible 2D array
    /// - `Err(XenonError)`: wraps internal `FfiError` for non-BLAS-compatible 2D input
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
            return Err(FfiError::InvalidRank {
                operation: "ffi::lda",
                backend: "blas",
                precondition: "tensor must be 2D",
                actual: alloc::format!("ndim={}", self.ndim()).into(),
            }.into());
        }
        if !self.is_blas_layout_compatible() {
            return Err(FfiError::BlasIncompatibleLayout {
                operation: "ffi::lda",
                backend: "blas",
                precondition: "F-contiguous 2D tensor without zero strides",
                actual: alloc::format!("shape={:?}, strides={:?}", self.shape(), self.strides()).into(),
            }.into());
        }
        if self.shape()[0] == 0 || self.shape()[1] == 0 {
            return Ok(1);
        }
        let strides = self.strides();
        Ok(strides[1])
    }
}
````

### 5.8 多维索引到指针偏移

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
                operation: "ffi::try_offset_of".into(),
                expected: self.ndim(),
                actual: index.len(),
            });
        }
        let strides = self.strides();
        let shape = self.shape();
        let mut offset: usize = 0;
        for i in 0..self.ndim() {
            if index[i] >= shape[i] {
                return Err(XenonError::IndexOutOfBounds {
                    operation: "ffi::try_offset_of".into(),
                    attempted_index: index.to_vec(),
                    axis: i,
                    shape: shape.to_vec(),
                });
            }
            let term = strides[i].checked_mul(index[i]).ok_or_else(|| XenonError::InvalidLayout {
                operation: "ffi::try_offset_of".into(),
                storage_kind: self.storage_kind().into(),
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset: self.offset(),
                storage_len: self.storage_len(),
                reason: "index-to-offset multiplication overflow".into(),
            })?;
            offset = offset.checked_add(term).ok_or_else(|| XenonError::InvalidLayout {
                operation: "ffi::try_offset_of".into(),
                storage_kind: self.storage_kind().into(),
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset: self.offset(),
                storage_len: self.storage_len(),
                reason: "index-to-offset accumulation overflow".into(),
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

### 5.9 Good/Bad 对比

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

### 6.2 元数据校验算法 (`validate_access_range`)

`from_raw_parts()` / `from_raw_parts_mut()` 内部调用 `validate_access_range()` 验证元数据合法性。当前版本 stride 全为非负 `usize`，因此 `logical_min` 恒等于 `offset`。算法如下：

```
validate_access_range(shape, strides, offset, storage_len):
    1. Verify len(shape) == len(strides); otherwise return Err(DimensionMismatch).
    2. Compute total_elements = product(shape) with checked multiplication;
       on overflow, return Err(IntegerOverflow).
    3. If total_elements == 0: skip pointer-range checks (empty tensor).
    4. Compute the maximum element offset that any logical element can
       reach, using checked multiplication / addition:
          For each axis i in [0, ndim):
            if shape[i] > 0:
              axis_extent = checked_mul(shape[i] - 1, strides[i])
              accumulate max contribution
          logical_min = offset
          logical_max = checked_add(offset, sum of axis_extent)
       If any checked operation fails, return Err(IntegerOverflow).
    5. If logical_max >= storage_len: return Err(InvalidLayout {
           storage_kind,
           shape,
           strides,
           offset,
           storage_len,
           reason: "access range exceeds storage",
       }).
    6. Return Ok(()).
```

**溢出安全性说明**：

- `validate_access_range()` 负责在构造阶段验证 `(shape, strides, offset, storage_len)` 整体可表示且不越界。
- `try_offset_of()` 负责在查询阶段对 `stride * index` 与逐项累加执行 checked arithmetic；即使张量本身来自安全构造路径，也不得把查询过程的溢出静默提升为 panic 或 wraparound。
- 这两层校验必须同时存在：前者约束张量元数据，后者约束单次索引转换的错误语义。

### 6.3 可写布局非重叠校验 (`validate_non_overlapping_layout`)

`from_raw_parts_mut()` 还必须拒绝会让两个不同逻辑索引映射到同一地址的可写布局。这里的“非重叠”定义为：任意两个不同逻辑索引 `i != j`，其可写目标地址 `addr(i)` 与 `addr(j)` 必须不同；换言之，逻辑元素地址集合不得重叠。该校验不得通过枚举全部可达 offset 来实现；当前版本只承诺接受可高效保守判定的正步长布局（例如 canonical F-order，以及满足同一保守判据的更一般正步长布局）。算法如下：

```
validate_non_overlapping_layout(shape, strides, offset, storage_len):
    1. If product(shape) <= 1: return Ok(()).
    2. Reject immediately if any non-singleton axis has stride == 0.
    3. Collect all non-singleton axes, sort them by stride ascending, and track
       the already-covered span of the lower-stride subspace.
    4. For each sorted axis i:
         require stride[i] >= covered_span;
         covered_span = covered_span + (shape[i] - 1) * stride[i]
       If any checked arithmetic fails or the inequality does not hold, reject.
    5. If the conservative test cannot prove non-overlap, return
       Err(InvalidLayout {
           storage_kind: "view_mut",
           shape,
           strides,
           offset,
           storage_len,
           reason: "mutable layout is not in the efficiently verifiable non-overlapping subset",
        }).
    6. Otherwise return Ok(()).
```

> 该校验与 `validate_access_range()` 分工不同：前者解决“会不会越界”，后者解决“会不会别名写入”。两者都属于 `需求说明书 §8` 下可直接验证的安全构造前提，失败时都须返回可恢复错误。该保守算法允许拒绝一部分理论上合法但无法高效证明不重叠的 exotic stride 布局；当前版本不为这类布局提供可写 raw-parts 构造承诺。

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
  - 内容: 模块声明、re-exports、`FfiError`、`BlasInfo` 结构体
  - 测试: `test_blas_info_f_order`, `test_ffi_error_mapping`
  - 前置: 无
  - 预计: 10 min

### Wave 2: 指针 API

- [ ] **T2**: 实现原始指针访问和裸指针构造/解构
  - 文件: `src/ffi/ptr.rs`
  - 内容: `as_ptr()`, `as_mut_ptr()`, `from_raw_parts`, `from_raw_parts_mut`, `into_raw_parts` 及 Safety 文档
  - 测试: `test_as_ptr_basic`, `test_as_mut_ptr_basic`, `test_from_raw_parts_roundtrip`, `test_into_raw_parts`
  - 前置: T1
  - 预计: 10 min

### Wave 3: BLAS 和索引（可并行）

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

### 并行执行图

```
Wave 1:    [T1]
             │
Wave 2:    [T2]
             │
Wave 3: ┌────┴────┐
        │         │
       [T3]      [T4]   (can run in parallel)
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                       | 说明                                                             |
| -------- | ------------------------------------------ | ---------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`                   | 验证指针访问、BLAS 兼容检查与 raw-parts 语义                     |
| 集成测试 | `tests/`                                   | 验证 `ffi` 与 `tensor`、`layout`、`storage` 的协同路径           |
| 边界测试 | 同模块测试中标注                           | 覆盖空张量、广播维度、未对齐指针和 BLAS 不兼容布局等边界         |
| 属性测试 | `tests/test_ffi.rs` 或 `tests/property.rs` | 验证 `try_offset_of` / `try_ptr_at` / raw-parts roundtrip 不变量 |

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
| 自别名可写布局 | `from_raw_parts_mut()` 返回 `InvalidLayout`                                                                                                                       |
| 零尺寸矩阵     | `lda()` 返回 `1`，供调用方满足 BLAS 最小 LDA 约束                                                                                                                 |
| 1D 张量        | `lda()` 返回错误                                                                                                                                                  |
| 零维张量       | `try_offset_of(&[])` 返回 `Ok(0)`                                                                                                                                 |
| 未对齐指针     | `from_raw_parts` 的 Safety 文档需说明对齐要求                                                                                                                     |

### 8.4 `需求说明书 §28.4` 边界测试场景

| 场景           | 说明                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------- |
| 导出结构生命周期   | `TensorExport<'a, _>` / `TensorExportMut<'a, _>` 不得逃逸源张量生命周期，借用结束后不可继续使用 |
| 广播零步长导入     | 只读零步长布局允许导入为共享视图；可写零步长布局构造统一返回错误                            |
| `storage_len` 重建 | 导出后按 `storage_len` 重建视图时覆盖空张量、offset 非零与末元素访问边界                            |

### 8.5 属性测试不变量

| 不变量                                                                                 | 测试方法                             |
| -------------------------------------------------------------------------------------- | ------------------------------------ |
| `try_ptr_at(idx)` 返回的指针等于基于 `as_ptr()` 和 `try_offset_of(idx)` 计算的期望地址 | 在合法索引集合上逐点比对             |
| `into_raw_parts → from_raw_parts_owned` roundtrip 保持 shape/strides/offset            | 对 F-contiguous owned 张量做往返验证 |
| `is_blas_layout_compatible() == true` 且维度/整数范围合法 ⟹ `blas_info()` 成功         | 以连续二维张量为样本验证             |

### 8.6 内存安全测试

| 场景                                           | 验证方式               |
| ---------------------------------------------- | ---------------------- |
| `from_raw_parts` + Drop                        | 无内存泄漏（借用语义） |
| `into_raw_parts` + `from_raw_parts_owned(raw)` | 重建后由 Drop 正确释放 |

### 8.7 集成测试

| 测试文件            | 测试内容                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| `tests/test_ffi.rs` | 指针 API / BLAS 兼容检查 / raw-parts roundtrip 与 `tensor`、`layout`、`storage` 的端到端协同路径 |

### 8.8 Feature gate / 配置测试

| 配置              | 验证点                                                                         |
| ----------------- | ------------------------------------------------------------------------------ |
| 默认配置          | 指针 API、BLAS 兼容性检查与 raw-parts roundtrip 在默认构建下保持既定安全边界。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。                                          |

### 8.9 类型边界 / 编译期测试

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
| `ffi ← layout`             | `layout`             | `is_f_contiguous()` / stride 标志        | BLAS 布局兼容性检查依赖布局查询结果，参见 `06-layout.md` §5.5           |
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
| `export()` / `export_mut()`                 | 返回的 `TensorExport` / `TensorExportMut` 中 `data`、`shape`、`strides` 均借用源张量内部存储；源张量 drop 后全部指针失效。`export_mut()` 额外要求 `&mut self` 且 `S: StorageMut`，确保独占可写访问。                                   |

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Recoverable error | `blas_info()` / `lda()` 在 rank 或布局非法时返回 `XenonError::Ffi { reason }`；BLAS 整数宽度转换失败由 `BlasInfo::as_blas_int()` 返回同一公开 opaque 变体；`from_raw_parts_owned()` 在 owned 元数据非法时返回 `XenonError::InvalidLayout`；`try_offset_of()` / `try_ptr_at()` 在 rank / bounds / checked arithmetic 非法时返回 `XenonError`；`from_raw_parts_mut()` 在可写布局自别名时返回 `XenonError::InvalidLayout`。 |
| Panic             | 不提供公开 panic-sugar 索引转换 API；`from_raw_parts*()` 中那些无法直接验证的不安全前提若被违反，仍属于 unsafe UB，而非 recoverable error。                                                                                                                                                                                                                                                                              |
| 路径一致性        | 指针访问、BLAS 查询与 raw-parts roundtrip 必须共享同一 shape / strides / offset 解释；无 SIMD / 并行分支。                                                                                                                                                                                                                                                                                                               |
| 容差边界          | 不适用。                                                                                                                                                                                                                                                                                                                                                                                                                 |

> **错误语义对齐：** FFI 文档仅公开 `try_offset_of()` 与 `try_ptr_at()` 这类 `Result` 接口。索引越界、维度不匹配、偏移溢出和布局自别名都属于 `需求说明书 §27` 下的可恢复错误，不再额外提供 `offset_of()` / `ptr_at()` 之类会把这些条件升级为 panic 的公开 sugar。

---

## 11. 设计决策记录

### 决策 1: BLAS 兼容 API 设计

| 属性     | 值                                                                                      |
| -------- | --------------------------------------------------------------------------------------- |
| 决策     | 提供结构化的 `BlasInfo` 查询方法，而非仅返回布尔值                                      |
| 理由     | 上游库需要完整的 BLAS 参数（data ptr、lda、rows、cols），结构体返回比单独方法调用更便捷 |
| 替代方案 | 仅返回 `bool is_blas_layout_compatible()` — 放弃，上游库需要重复获取多个参数            |
| 替代方案 | 返回 raw C 常量 — 放弃，不符合 Rust 惯例                                                |

> **补充**：Xenon 的直接 BLAS 路径只接受 BLAS-compatible 的 F-order 2D 张量。转置或非连续视图必须先显式 materialize 为 `to_contiguous()` 结果，再由调用方结合导出的元数据传入对应的后端常量。

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

> **补充**：`try_offset_of()` 在文档层明确要求 checked arithmetic；即使张量本身来自安全构造路径，也不得把索引转换错误表述为“天然不会发生，因此无需检查”。

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
| SemVer      | `TensorExport<'a, A>`、`TensorExportMut<'a, A>`、`OwnedRawParts<A, D>` 的字段布局和 `#[repr(C)]` 表示均为公开契约，变更须遵循 SemVer；新增公共 FFI 方法或枚举变体属于 minor 变更 |
| 最小依赖    | 无新增第三方依赖，符合 `需求说明书 §1.3` 对最小依赖的限制                                                                                                                        |
| 索引类型    | 逻辑索引统一使用 `usize`；BLAS/LAPACK 整数参数在边界处按目标后端转换为 `i32` 或 `i64`                                                                                            |
| stride 范围 | 当前版本只接受非负 stride；负步长导入不在范围内                                                                                                                                  |
| 错误诊断    | `blas_info()` / `lda()` 返回 `Result`，保留失败原因                                                                                                                              |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-10 |
| 1.1.4 | 2026-04-14 |
| 1.1.5 | 2026-04-15 |
| 1.2.0 | 2026-04-15 |
| 1.2.1 | 2026-04-15 |
| 1.2.2 | 2026-04-15 |
| 1.2.3 | 2026-04-16 |
| 1.2.4 | 2026-04-16 |
| 1.2.5 | 2026-04-16 |
| 1.2.6 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
