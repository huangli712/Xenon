# FFI 接口模块设计

> 文档编号: 23 | 模块: `src/ffi/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `06-layout.md`
> 需求参考: 需求说明书 §5, §6, §7, §8, §25, §27, §28.1
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责            | 包含                                            | 不包含                                              |
| --------------- | ----------------------------------------------- | --------------------------------------------------- |
| 原始指针 API    | `as_ptr()`/`as_mut_ptr()`                       | BLAS 绑定实现（由上游库通过 `blas-sys` crate 提供） |
| 裸指针构造张量  | `from_raw_parts`/`from_raw_parts_mut`           | GPU 内存操作                                        |
| 裸指针解构张量  | `into_raw_parts`                                | 跨进程共享内存                                      |
| BLAS 兼容性 API | `blas_layout()`/`is_blas_layout_compatible()`，以及非核心的兼容性辅助 `blas_trans()` / `BlasTrans` | 自动调用 BLAS（由上游库负责）                       |
| 多维索引转换    | `try_offset_of()`/`try_ptr_at()`                | 序列化/反序列化                                     |

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

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §5, §6, §7, §8, §25, §27, §28.1 |
| 范围内   | 原始指针访问、raw-parts 往返、BLAS 兼容性查询、多维索引到偏移 / 指针转换。 |
| 范围外   | 实际 BLAS / LAPACK 例程调用、GPU 互操作、跨进程共享内存与更高层序列化协议。 |
| 非目标   | 不把 `ffi` 扩展为外部数值库绑定层，不新增第三方 FFI crate 依赖。 |

| 需求条款 | 本文承接方式 |
| -------- | ------------ |
| §5 复数类型 | 明确 `Complex<f32>` / `Complex<f64>` 的稳定 `#[repr(C)]` FFI 表示。 |
| §6 存储系统 | `export()` / `export_mut()` 分别覆盖 `Storage` / `StorageMut`，保持零拷贝导出边界。 |
| §7 内存布局 | 导出与导入统一使用 shape / strides / offset 元数据解释当前版本合法布局。 |
| §8 张量类型 | `from_raw_parts*()` 验证可检查的布局、范围与别名条件，失败时返回可恢复错误。 |
| §25 FFI 集成 | 提供原始指针、偏移转换、BLAS 兼容性查询和 raw-parts roundtrip。 |
| §27 错误处理 | 仅公开 `try_offset_of()` / `try_ptr_at()` 这类 `Result` API，不额外暴露 panic sugar。 |
| §28.1 文档 | 所有 unsafe 入口提供 Safety 说明；关键 FFI API 提供示例，非完整上下文示例统一标记 `ignore`。 |

---

## 3. 文件位置

```
src/
└── ffi/
    ├── mod.rs         # Module root, re-exports
    ├── types.rs       # BlasLayout, BlasTrans, BlasInfo type definitions
    ├── ptr.rs         # Raw-pointer APIs (as_ptr, as_mut_ptr, from_raw_parts, from_raw_parts_mut, into_raw_parts)
    ├── blas.rs        # BLAS compatibility checks (is_blas_layout_compatible, blas_info, lda)
    └── offset.rs      # Multi-dimensional index to pointer offset (try_offset_of, try_ptr_at)
```

多文件设计：将 FFI 按职责拆分为多个文件，便于后期拓展和维护。

| 文件        | 职责                                                                                        |
| ----------- | ------------------------------------------------------------------------------------------- |
| `mod.rs`    | 模块入口，导出公共 API                                                                      |
| `types.rs`  | `BlasLayout`/`BlasTrans` 枚举、`BlasInfo` 结构体                                            |
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
│   ├── super::types         # BlasInfo, BlasLayout, compatibility-only BlasTrans
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

> **owner 约定：** `as_ptr()` / `as_mut_ptr()` / `into_raw_parts()` / `from_raw_parts_owned()` 是 ffi 模块暴露的公开 FFI API。它们消费 `tensor`/`storage` 的元数据，但不应在依赖表中反向写成“来自 tensor 模块的方法”。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 辅助类型

```rust,ignore
/// BLAS matrix layout identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasLayout {
    /// Column-major (Fortran order).
    /// Corresponds to BLAS `CblasColMajor` (102).
    ColumnMajor,
}

/// BLAS transpose identifier.
///
/// This enum is a compatibility helper for upstream BLAS call sites, not part of
/// Xenon's core tensor semantics or required public tensor surface. Xenon's direct
/// BLAS-compatible tensors are always F-order (column-major), so `blas_trans()`
/// only serves as a non-core convenience wrapper around the fixed `NoTrans`
/// choice after BLAS-compatibility checks. Upstream callers may ignore this
/// helper and pass the corresponding backend constant directly once they have
/// established an equivalent precondition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasTrans {
    /// No transpose.
    NoTrans,
    /// Transpose.
    Trans,
    /// Conjugate transpose (complex only).
    ConjTrans,
}

impl BlasLayout {
    /// Xenon's BLAS layout is always column-major.
    pub const COLUMN_MAJOR: BlasLayout = BlasLayout::ColumnMajor;
}

/// Internal error classification for FFI-specific failures.
///
/// # Diagnostics Design
///
/// `FfiError` uses `&'static str` for the `operation`, `backend`, and
/// `precondition` fields to maintain zero-allocation, compile-time-known
/// structured formatting. The `actual` field uses `Cow<'static, str>` to
/// accommodate runtime dynamic context (e.g., actual ndim value, shape, etc.).
/// This design satisfies `require.md` §27 diagnostics requirements: the error
/// category is identified by the enum variant, and the triggering context is
/// carried by `actual`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FfiError {
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

/// Public Xenon APIs wrap `FfiError` in `XenonError::Ffi` before returning.
impl From<FfiError> for XenonError {
    fn from(value: FfiError) -> Self {
        XenonError::Ffi(value)
    }
}
```

### 5.2 原始指针 API

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a read-only raw pointer to the data start.
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
    /// ```ignore
    pub fn as_ptr(&self) -> *const A {
        if self.is_empty() {
            // For empty tensors, return a non-dereferenceable dangling pointer.
            // Do NOT call .add() on a potentially dangling base pointer.
            return self.storage.as_ptr();
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
    /// Returns a mutable raw pointer to the data start.
    ///
    /// Only available for writable storage (Owned, ViewMut).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_mut_ptr();
    /// // Can be passed to C functions requiring a mutable pointer
    /// ```ignore
    pub fn as_mut_ptr(&mut self) -> *mut A {
        if self.is_empty() {
            return self.storage.as_mut_ptr();
        }
        unsafe {
            self.storage.as_mut_ptr().add(self.offset)
        }
    }
}
````

#### C 侧结构化导出格式

````rust,ignore
/// Element type discriminant for FFI consumers.
///
/// Each variant corresponds to one of Xenon's supported tensor element types
/// (see `require.md` §4). C consumers use this to interpret the `data` pointer.
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

impl<A: Element> ElementType {
    /// Returns the `ElementType` discriminant for `A`.
    ///
    /// This is determined at compile time via `Element` trait association.
    pub const fn of() -> Self;
}

/// Raw tensor data export for FFI consumers.
///
/// # Safety
///
/// - All pointer fields (`data`, `shape`, `strides`) borrow the source tensor's
///   internal storage and metadata. They become invalid immediately after the
///   source tensor is dropped.
/// - C consumers must use `ndim` as the length of both the `shape` and `strides`
///   arrays. Do NOT use hardcoded lengths or any other source.
/// - For `bool` element type, C consumers must use `_Bool` (C23 `bool`) to match
///   Rust's `bool` ABI representation. Using `int` or `unsigned char` is
///   undefined behavior.
/// - `TensorExport` is the read-only export form and uses `*const A`.
///   `TensorExportMut` is the writable export form and uses `*mut A`.
#[repr(C)]
pub struct TensorExport<A> {
    /// Typed pointer to the storage base address.
    ///
    /// `data`, `strides`, and `offset` all use element units of `A`.
    /// C consumers must cast `data` to the matching element type and interpret
    /// both `offset` and `strides` as element counts rather than byte counts.
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
    /// Offset from the storage base pointer to the logical first element,
    /// measured in elements.
    pub offset: usize,
}

/// Raw mutable tensor data export for FFI consumers.
#[repr(C)]
pub struct TensorExportMut<A> {
    /// Typed pointer to the storage base address.
    ///
    /// `data`, `strides`, and `offset` all use element units of `A`.
    /// C consumers must cast `data` to the matching element type and interpret
    /// both `offset` and `strides` as element counts rather than byte counts.
    pub data: *mut A,
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
    /// Offset from the storage base pointer to the logical first element,
    /// measured in elements.
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
    /// Empty tensors are allowed: when `len() == 0`, `data` may be a
    /// non-dereferenceable sentinel pointer. In that case `shape`, `strides`,
    /// and `offset` must still correctly describe the empty tensor metadata.
    pub fn export(&self) -> TensorExport<A>;
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
    pub fn export_mut(&mut self) -> Result<TensorExportMut<A>, XenonError>;
}
````

> **导出语义说明：** `TensorExport` 面向 C 调用方提供"指针 + shape + strides + offset"的结构化快照，其中 `shape` 与 `strides` 指针均借用源张量内部元数据，不能在源张量释放后继续使用。

> **导出范围说明：** `export()` 提供只读结构化导出并返回 `TensorExport<A>`，适用于任意 `TensorBase<S, D>` 且仅要求 `S: Storage`，因此覆盖 Owned、View、只读共享存储以及所有合法 stride 布局。`export_mut()` 返回 `TensorExportMut<A>`，适用于任意满足 `S: StorageMut` 的 `TensorBase<S, D>`，因此同时覆盖 Owned 与 `ViewMut` 这两类可写存储。

> **可写导出边界：** `export_mut()` 通过 `&mut self` 和 `S: StorageMut` 保证 Xenon 侧的独占可写访问；只读视图和共享只读存储则在 trait 边界上直接被拒绝。这与 `require.md §6` 的存储模式转换和 `§25` 的零拷贝导出要求保持一致。

> **空张量约定：** 空张量（`len() == 0`）导出时 `data` 可为悬垂但非解引用的哨兵指针，此时 `shape`、`strides`、`offset` 仍须正确反映空张量元数据。C 调用方必须先基于长度判断是否可解引用。

> **指针语义补充：** `TensorExport<A>::data` 是 `*const A`，`TensorExportMut<A>::data` 是 `*mut A`，二者语义上都始终指向 storage base pointer。`offset` 与 `strides` 都以"元素个数"计量，而不是字节数。逻辑首元素地址等于 `data.add(offset)`（仅对非空张量成立）。

> **stride 约定：** `strides` 以"元素个数"而非字节数表示步长，类型为 `usize`。按照 `06-layout.md` §1.2 与 `require.md` §7，当前版本 Xenon 不支持负步长，因此 FFI 导出格式也不保留负 stride 语义。

> **offset 约定：** `offset` 一律表示从 `data`（即导出的 storage base pointer）到逻辑首元素的**元素单位**位移，与 `07-tensor.md` 的 raw-parts 契约一致；不得将其解释为字节偏移。C 调用方必须先按元素单位应用这一次偏移，再把结果视为逻辑首元素地址。

> **ndim 一致性约定：** C 消费者须以 `ndim` 为 `shape` 和 `strides` 数组的长度，不得以硬编码长度或其它来源替代。`TensorExport` 的构造保证 `shape` 和 `strides` 指向的数组长度均等于 `ndim`。

> **生命周期与借用语义：** 导出结果不拥有底层内存；一旦源张量被 drop，`TensorExport` 内的 `data`、`shape`、`strides` 全部立即失效。应将该导出结果视为对源张量当前元数据与指针状态的借用快照：`export()` 暴露只读快照，`export_mut()` 暴露独占可写快照；无论是否跨 FFI 边界缓存，该快照都不得超出源张量的生命周期，也不得绕过 `&mut self` 所表达的独占写语义。本文不额外指定 `TensorExport<_>` / `TensorExportMut<_>` 的 auto trait 组合，线程相关性质以其字段与 Rust auto-trait 推导结果为准。

#### 5.2.1 Complex FFI layout contract

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


#### 5.2.2 Bool FFI layout contract

> **Bool ABI 要求：** Rust `bool` 在 FFI 中等价于 C 的 `_Bool`（C23 起为 `bool`），大小为 1 字节，对齐为 1，合法值为 `0`（false）和 `1`（true）。C 消费者必须使用 `_Bool` 或 `bool`（C23）来匹配 `TensorExport<bool>` / `TensorExportMut<bool>` 中的 `data` 指针类型；使用 `int`、`unsigned char` 或其它整数类型会导致未定义行为。

> **导出语义：** 导出 `bool` 张量时，`TensorExport<bool>` 中的 `data` 为 `*const bool`（C 侧 `const _Bool*`），`TensorExportMut<bool>` 中的 `data` 为 `*mut bool`（C 侧 `_Bool*`），`offset` 与 `strides` 按 `bool` 元素个数计量。`strides[i] == 1` 表示相邻逻辑元素在内存中连续排列（每个占 1 字节）。
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
    /// ```ignore
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
            storage: ViewRepr::new(ptr),
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
        if shape.size() != 0 && layout::has_zero_stride(&strides) {
            return Err(XenonError::InvalidLayout {
                operation: "ffi::from_raw_parts_mut".into(),
                storage_kind: "view_mut".into(),
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset,
                storage_len,
                reason: "mutable raw-parts view must not contain zero strides".into(),
            });
        }
        validate_non_overlapping_layout(&shape, &strides, offset)?;
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
            storage: ViewMutRepr::new(ptr),
            shape,
            strides,
            offset,
            flags: layout::compute_layout_flags(&shape, &strides, logical_ptr),
        })
    }
}
````

> **校验边界说明：** 与 `07-tensor.md` §5.6 一致，`from_raw_parts*()` 只验证库能够直接检查的元数据约束（例如 shape/stride/offset/storage_len 组合是否合法、是否溢出、是否越界），并在失败时返回 `Result<_, XenonError>`。指针有效性、对齐、实际可访问范围与生命周期仍由调用方在 `unsafe` 前提下负责。

> **空张量补充：** `ptr.add(offset)` 形式的逻辑首元素地址计算只适用于非空张量；空张量路径必须跳过该指针运算，并改用 `NonNull::dangling()` 这类明确定义的非解引用哨兵值参与 flags / metadata 初始化。

> **可写视图补充：** `from_raw_parts_mut()` 不仅必须拒绝零步长布局，还必须拒绝一切能被高效保守判定为潜在自别名的布局。实现上先用 `validate_access_range()` 验证越界与可表示性，再用 `validate_non_overlapping_layout()` 对受支持的正步长布局做保守非重叠判定；若布局超出该高效判定范围，也必须返回可恢复错误，而不是枚举全部可达 offset。

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

> **裸指针直接构造 Owned 张量的设计约束：** 当前版本不提供从任意裸指针直接构造 `Owned` 张量的接口。`from_raw_parts()` / `from_raw_parts_mut()` 仅构造视图（View / ViewMut），`from_raw_parts_owned()` 仅从 `into_raw_parts()` 导出的 `OwnedRawParts` 重建 Owned 张量。原因是 `Owned` 存储需要 Xenon 分配器的元数据（capacity、alignment），这些信息无法从单一裸指针推断。若调用方需要从裸指针创建 Owned 张量，须先将数据复制到 Xenon 分配的张量中（如通过 `Tensor::from_vec()` 等构造方法）。

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
    /// and ensure `rows`, `cols`, and `lda` fit the BLAS backend integer type
    /// (currently `i32`), typically by calling `blas_info()`.
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert!(a.is_blas_layout_compatible());
    ///
    /// let b = a.slice(s![.., 1..3]);
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
/// BLAS matrix information.
///
/// Contains all parameters needed for BLAS function calls.
pub struct BlasInfo<A> {
    /// Data pointer to the logical first element.
    pub data_ptr: *const A,
    /// Leading dimension (element units).
    pub leading_dim: i32,
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
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
    /// - `Ok(BlasInfo<A>)`: compatibility conditions met and all BLAS integer
    ///   parameters fit in `i32`
    /// - `Err(XenonError)`: wraps internal `FfiError` when not BLAS compatible,
    ///   not 2D, or any BLAS parameter
    ///   exceeds `i32::MAX`
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
        let raw_lda = self.lda()?;
        let lda = i32::try_from(raw_lda).map_err(|_| FfiError::IntegerOverflow {
            operation: "ffi::blas_info",
            backend: "blas",
            precondition: "lda must fit in i32",
            actual: alloc::format!("lda={}", raw_lda).into(),
        }).map_err(XenonError::from)?;
        let rows = i32::try_from(self.shape()[0]).map_err(|_| FfiError::IntegerOverflow {
            operation: "ffi::blas_info",
            backend: "blas",
            precondition: "rows must fit in i32",
            actual: alloc::format!("rows={}", self.shape()[0]).into(),
        }).map_err(XenonError::from)?;
        let cols = i32::try_from(self.shape()[1]).map_err(|_| FfiError::IntegerOverflow {
            operation: "ffi::blas_info",
            backend: "blas",
            precondition: "cols must fit in i32",
            actual: alloc::format!("cols={}", self.shape()[1]).into(),
        }).map_err(XenonError::from)?;

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
                    attempted_index: index[i],
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
    /// let tensor = Tensor1::<i32>::from_vec(vec![10, 20, 30, 40]);
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
// blas_trans() is only a compatibility helper queried after obtaining a
// BLAS-compatible representation. For Xenon's direct BLAS path, this is always
// a standard F-order matrix and therefore maps to BlasTrans::NoTrans.
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns the BLAS transpose identifier for this tensor.
    ///
    /// This is a compatibility-only helper, not a core tensor API commitment.
    /// For Xenon's BLAS-compatible F-order matrices it returns `BlasTrans::NoTrans`.
    /// Transposed or otherwise non-BLAS-compatible views must first be converted
    /// into an owned F-order tensor before calling BLAS; callers may also bypass
    /// this helper and pass the equivalent backend constant directly.
    pub fn blas_trans(&self) -> BlasTrans {
        BlasTrans::NoTrans
    }
}

// Good - Check BLAS layout compatibility before passing
if tensor.is_blas_layout_compatible() {
    let info = tensor.blas_info().expect("BLAS-compatible tensor should yield BlasInfo");
    unsafe {
        // SAFETY: `info` came from `blas_info()`, so layout/rank/integer checks passed.
        call_blas_dgemm(info, tensor.blas_trans(), ...);
    }
} else {
    let contiguous = tensor.to_contiguous();
    let info = contiguous.blas_info().expect("contiguous tensor should yield BlasInfo");
    unsafe {
        // SAFETY: `contiguous` is materialized into Xenon's BLAS-compatible layout.
        call_blas_dgemm(info, contiguous.blas_trans(), ...);
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
- **空张量**：storage base pointer 可能是悬垂但非解引用的哨兵值，`as_ptr()` 不对其做 `.add(offset)` 运算，直接返回 base pointer。

`from_raw_parts` 的 Safety 由调用方保证，但会先执行可直接检查的元数据验证：若 `shape`、`strides`、`offset` 与 `storage_len` 的组合明显非法，则返回 `Err(XenonError::InvalidLayout { .. })`；只有那些库无法从元数据自行证明的指针/生命周期前提，才继续由调用方承担。空张量路径必须跳过 `ptr.add(offset)`，改用非解引用哨兵值参与 flags 计算。

### 6.2 元数据校验算法 (`validate_access_range`)

`from_raw_parts()` / `from_raw_parts_mut()` 内部调用 `validate_access_range()` 验证元数据合法性。算法如下：

```
validate_access_range(shape, strides, offset, storage_len):
    1. Verify len(shape) == len(strides); otherwise return Err(DimensionMismatch).
    2. Compute total_elements = product(shape) with checked multiplication;
       on overflow, return Err(IntegerOverflow).
    3. If total_elements == 0: skip pointer-range checks (empty tensor).
    4. Compute the minimum and maximum element offsets that any logical element
       can reach, using checked subtraction / multiplication / addition:
          For each axis i in [0, ndim):
            if shape[i] > 0:
              axis_extent = checked_mul(shape[i] - 1, strides[i])
              track axis-wise min/max contribution
          logical_min = checked_add(offset, sum of min contributions)
          logical_max = checked_add(offset, sum of max contributions)
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

`from_raw_parts_mut()` 还必须拒绝会让两个不同逻辑索引映射到同一地址的可写布局。这里的“非重叠”定义为：任意两个不同逻辑索引 `i != j`，其可写目标地址 `addr(i)` 与 `addr(j)` 必须不同；换言之，逻辑元素地址集合不得重叠。该校验不得通过枚举全部可达 offset 来实现；当前版本只承诺接受可高效保守判定的正步长布局（例如 contiguous / canonical C-order / canonical F-order，以及满足同一保守判据的更一般正步长布局）。算法如下：

```
validate_non_overlapping_layout(shape, strides, offset):
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

> 该校验与 `validate_access_range()` 分工不同：前者解决“会不会越界”，后者解决“会不会别名写入”。两者都属于 `require.md §8` 下可直接验证的安全构造前提，失败时都须返回可恢复错误。该保守算法允许拒绝一部分理论上合法但无法高效证明不重叠的 exotic stride 布局；当前版本不为这类布局提供可写 raw-parts 构造承诺。

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
  - 内容: 模块声明、re-exports、`BlasLayout`、兼容性辅助 `BlasTrans`、`BlasInfo` 结构体
  - 测试: `test_blas_layout_column_major`, `test_blas_trans_helper`
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
  - 测试: `test_is_blas_layout_compatible_f_order`, `test_is_blas_layout_compatible_non_contiguous`, `test_lda_f_order`
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

| 测试函数                                 | 测试内容                                           | 优先级 |
| ---------------------------------------- | -------------------------------------------------- | ------ |
| `test_as_ptr_basic`                      | `as_ptr()` 返回有效指针                            | 高     |
| `test_as_mut_ptr_basic`                  | `as_mut_ptr()` 返回有效可写指针                    | 高     |
| `test_as_ptr_offset`                     | 指针考虑 offset 后指向正确元素                     | 高     |
| `test_is_blas_layout_compatible_f_order`        | F-order 连续数组兼容                        | 高     |
| `test_is_blas_layout_compatible_non_contiguous` | 非连续切片不兼容                            | 高     |
| `test_is_blas_layout_compatible_broadcast`      | 广播维度（零步长）不兼容                    | 高     |
| `test_blas_info_f_order`                 | F-order 返回正确 BlasInfo                          | 高     |
| `test_blas_info_overflow`                | `blas_info()` 处理接近 `usize::MAX` 的 rows/cols/lda | 高     |
| `test_lda_f_order`                       | F-order [3,4] 返回 3                               | 高     |
| `test_lda_non_contiguous`                | 非连续（切片）数组 lda() 返回错误                  | 中     |
| `test_from_raw_parts_roundtrip`          | `into_raw_parts → from_raw_parts_owned` 往返一致性 | 高     |
| `test_from_raw_parts_mut_roundtrip`      | 可变构造 → 修改 → 读取                             | 高     |
| `test_from_raw_parts_mut_reject_overlap` | 可写 raw-parts 构造拒绝地址重叠布局                | 高     |
| `test_into_raw_parts`                    | Owned 张量解构后指针有效                           | 高     |
| `test_into_raw_parts_memory_leak`        | 解构后正确释放                                     | 中     |
| `test_try_offset_of_various`             | recoverable 索引转换返回正确偏移或错误             | 高     |
| `test_try_offset_of_checked_overflow`    | 极端 stride/index 组合返回可恢复错误而非 panic     | 高     |
| `test_try_ptr_at_various`                | recoverable 指针转换返回正确指针或错误             | 高     |

### 8.3 边界测试场景

| 场景       | 预期行为                                      |
| ---------- | --------------------------------------------- |
| 空张量     | `as_ptr()` 对空张量不保证返回可解引用指针；raw-parts 构造需跳过 `ptr.add(offset)` |
| 单元素张量 | `as_ptr()` 指向唯一元素                       |
| 非连续切片 | `is_blas_layout_compatible()` 返回 `false`   |
| 广播维度   | `is_blas_layout_compatible()` 返回 `false`   |
| 自别名可写布局 | `from_raw_parts_mut()` 返回 `InvalidLayout` |
| 零尺寸矩阵 | `lda()` 返回 `1`，供调用方满足 BLAS 最小 LDA 约束 |
| 1D 张量    | `lda()` 返回错误                              |
| 零维张量   | `try_offset_of(&[])` 返回 `Ok(0)`             |
| 未对齐指针 | `from_raw_parts` 的 Safety 文档需说明对齐要求 |

### 8.4 属性测试不变量

| 不变量                                                                                 | 测试方法                             |
| -------------------------------------------------------------------------------------- | ------------------------------------ |
| `try_ptr_at(idx)` 返回的指针等于基于 `as_ptr()` 和 `try_offset_of(idx)` 计算的期望地址 | 在合法索引集合上逐点比对             |
| `into_raw_parts → from_raw_parts_owned` roundtrip 保持 shape/strides/offset            | 对 F-contiguous owned 张量做往返验证 |
| `is_blas_layout_compatible() == true` 且维度/整数范围合法 ⟹ `blas_info()` 成功         | 以连续二维张量为样本验证             |

### 8.5 内存安全测试

| 场景                                           | 验证方式               |
| ---------------------------------------------- | ---------------------- |
| `from_raw_parts` + Drop                        | 无内存泄漏（借用语义） |
| `into_raw_parts` + `from_raw_parts_owned(raw)` | 重建后由 Drop 正确释放 |
| `from_raw_parts` 野指针                        | AddressSanitizer 检测  |

### 8.6 集成测试

| 测试文件            | 测试内容                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| `tests/test_ffi.rs` | 指针 API / BLAS 兼容检查 / raw-parts roundtrip 与 `tensor`、`layout`、`storage` 的端到端协同路径 |

### 8.7 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 指针 API、BLAS 兼容性检查与 raw-parts roundtrip 在默认构建下保持既定安全边界。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.8 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `into_raw_parts()` 仅对 `Owned` 存储开放 | 编译期测试。 |
| `blas_info()` / `lda()` 仅对 2D BLAS-compatible 张量成功 | 运行时错误测试与签名检查。 |
| 实际 BLAS/LAPACK 调用与 GPU interop 不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向            | 对方模块  | 接口/类型                            | 约定                                                                    |
| --------------- | --------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `ffi → tensor`  | `tensor`  | 原始指针访问                         | 通过 `TensorBase` 的 storage 获取底层指针，参见 `07-tensor.md` §5       |
| `ffi ← layout`  | `layout`  | `is_f_contiguous()` / stride 标志    | BLAS 布局兼容性检查依赖布局查询结果，参见 `06-layout.md` §5.5           |
| `ffi → storage` | `storage` | `OwnedRawParts` / allocator metadata | `into_raw_parts` 导出 owned 存储的完整重建信息，参见 `05-storage.md` §5 |
| `ffi → upstream libraries`  | `upstream libraries`  | `blas_info()` / `lda()` / `try_ptr_at()` | 向外部 BLAS/FFI 调用方暴露零拷贝参数与可恢复索引转换                     |

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

| 操作 | 所有权/生命周期语义 |
|------|---------------------|
| `as_ptr()` / `as_mut_ptr()` | 返回的指针借用源张量；源张量 drop 后指针立即失效。`as_mut_ptr()` 要求独占 `&mut self`，借用期间不可有其它引用。 |
| `into_raw_parts()` | 消费源张量（`self`），将内存所有权转移给调用方。调用方须按 Xenon 分配契约回收：通过 `from_raw_parts_owned()` 重建张量并由 Drop 释放，或使用与 Xenon 分配器元数据等价的专用回收路径；不得直接调用系统 `free` 或其他 foreign allocator。 |
| `from_raw_parts()` / `from_raw_parts_mut()` | 构造的视图生命周期 `'a` 由调用方在 `unsafe` 前提下保证，须与底层内存的实际存活期一致。视图不拥有内存，drop 时不会释放。 |
| `from_raw_parts_owned()` | 接收 `OwnedRawParts` 并重建 Owned 张量，内存所有权回归 Xenon 的 Drop 管理。 |
| `export()` / `export_mut()` | 返回的 `TensorExport` / `TensorExportMut` 中 `data`、`shape`、`strides` 均借用源张量内部存储；源张量 drop 后全部指针失效。`export_mut()` 额外要求 `&mut self` 且 `S: StorageMut`，确保独占可写访问。 |

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `blas_info()` / `lda()` 在 rank、布局或 BLAS 整数参数非法时返回 `XenonError::Ffi`；`from_raw_parts_owned()` 在 owned 元数据非法时返回 `XenonError::InvalidLayout`；`try_offset_of()` / `try_ptr_at()` 在 rank / bounds / checked arithmetic 非法时返回 `XenonError`；`from_raw_parts_mut()` 在可写布局自别名时返回 `XenonError::InvalidLayout`。 |
| Panic | 不提供公开 panic-sugar 索引转换 API；`from_raw_parts*()` 中那些无法直接验证的不安全前提若被违反，仍属于 unsafe UB，而非 recoverable error。 |
| 路径一致性 | 指针访问、BLAS 查询与 raw-parts roundtrip 必须共享同一 shape / strides / offset 解释；无 SIMD / 并行分支。 |
| 容差边界 | 不适用。 |

> **错误语义对齐：** FFI 文档仅公开 `try_offset_of()` 与 `try_ptr_at()` 这类 `Result` 接口。索引越界、维度不匹配、偏移溢出和布局自别名都属于 `require.md §27` 下的可恢复错误，不再额外提供 `offset_of()` / `ptr_at()` 之类会把这些条件升级为 panic 的公开 sugar。

---

## 11. 设计决策记录

### 决策 1: BLAS 兼容 API 设计

| 属性     | 值                                                                                      |
| -------- | --------------------------------------------------------------------------------------- |
| 决策     | 提供结构化的 `BlasInfo` 查询方法，而非仅返回布尔值                                      |
| 理由     | 上游库需要完整的 BLAS 参数（data ptr、lda、rows、cols），结构体返回比单独方法调用更便捷 |
| 替代方案 | 仅返回 `bool is_blas_layout_compatible()` — 放弃，上游库需要重复获取多个参数            |
| 替代方案 | 返回 raw C 常量 — 放弃，不符合 Rust 惯例                                                |

> **补充**：Xenon 的直接 BLAS 路径只接受 BLAS-compatible 的 F-order 2D 张量。转置或非连续视图必须先显式 materialize 为 `to_contiguous()` 结果，再以 `BlasTrans::NoTrans` 传给上游 BLAS。

### 决策 2: Safety 独立边界

| 属性     | 值                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 决策     | `from_raw_parts` 和 `from_raw_parts_mut` 使用最小 Safety 模约束集                        |
| 理由     | 将安全责任尽可能交给调用方，库本身不做额外假设；与 `std::slice::from_raw_parts` 设计一致 |
| 替代方案 | 库内部验证所有 Safety 条件 — 放弃，运行时开销过大（O(n) 检查）                           |

### 决策 3: 性能 — 零拷贝优先

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | FFI 方法只做可直接检查的元数据验证，不重复承担指针级 Safety 证明                                     |
| 理由     | 与 `07-tensor.md` 一致：保留必要的 `shape/stride/offset/storage_len` 校验，同时避免把无法证明的内存前提伪装成库内可验证逻辑 |
| 替代方案 | 完全不校验元数据 — 放弃，会让明显非法输入延迟到 UB；对所有内存前提做深度验证 — 放弃，超出当前边界 |

> **补充**：`try_offset_of()` 在文档层明确要求 checked arithmetic；即使张量本身来自安全构造路径，也不得把索引转换错误表述为“天然不会发生，因此无需检查”。

---

## 12. 性能考量

| 操作                   | 时间复杂度 | 说明                               |
| ---------------------- | ---------- | ---------------------------------- |
| `as_ptr()`             | O(1)       | 仅指针加法                         |
| `as_mut_ptr()`         | O(1)       | 仅指针加法                         |
| `is_blas_layout_compatible()` | O(1)       | 检查布局标志                |
| `blas_info()`                 | O(1)       | 包含 `is_blas_layout_compatible()` + 构造 |
| `lda()`                | O(1)       | 步长查询                           |
| `try_offset_of()`      | O(ndim)    | 逐轴计算 + 可恢复错误分支          |
| `try_ptr_at()`         | O(ndim)    | `try_offset_of()` + 指针加法       |
| `from_raw_parts()`     | O(ndim)    | 元数据校验 + 构造视图              |
| `into_raw_parts()`     | O(1)       | 提取字段 + `ManuallyDrop`          |

**性能提示**:

- `as_ptr()` 和 `as_mut_ptr()` 应标注 `#[inline]`
- `try_offset_of()` / `try_ptr_at()` 在热路径中可能需要内联
- `is_blas_layout_compatible()` 检查现有 `LayoutFlags`，无需重新计算

---

## 13. 平台与工程约束

| 约束        | 说明                                                                                          |
| ----------- | --------------------------------------------------------------------------------------------- |
| `std` only  | 当前版本仅讨论 `std` 环境下的 FFI 接口；FFI 指针操作依赖 `std` 提供的分配器与布局保证         |
| 单 crate    | FFI 模块位于 `src/ffi/`，不引入额外 crate，保持 Xenon 单 crate 结构                           |
| SemVer      | `TensorExport<A>`、`TensorExportMut<A>`、`OwnedRawParts<A,D>` 的字段布局和 `#[repr(C)]` 表示均为公开契约，变更须遵循 SemVer；新增公共 FFI 方法或枚举变体属于 minor 变更 |
| 最小依赖    | 无新增第三方依赖，符合 `require.md` §1.3 对最小依赖的限制                                     |
| 索引类型    | 逻辑索引统一使用 `usize`；仅 BLAS 整数参数在边界处再转换为 `i32`                               |
| stride 范围 | 当前版本只接受非负 stride；负步长导入不在范围内                                                |
| 错误诊断    | `blas_info()` / `lda()` 返回 `Result`，保留失败原因                                            |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
