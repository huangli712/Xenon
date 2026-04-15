# 张量类型模块设计

> 文档编号: 07 | 模块: `src/tensor/` | 阶段: Phase 3
> 前置文档: `01-architecture.md`, `02-dimension.md`, `05-storage.md`, `06-layout.md`
> 需求参考: 需求说明书 §6、§7、§8、§10、§19、§22、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责        | 包含                                                                          | 不包含                                                   |
| ----------- | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
| 核心结构体  | `TensorBase<S, D>` 双参数泛型结构体定义                                       | 逐元素与归约逻辑（参见 `11-math.md`、`13-reduction.md`） |
| 类型别名    | `Tensor`/`TensorView`/`TensorViewMut`/`ArcTensor` 及维度便捷别名              | 广播规则（参见 `15-broadcast.md §3`）                    |
| 基础查询    | shape/ndim/len/strides/is_empty/is_f_contiguous/is_aligned/存储位置查询等方法 | 形状操作（当前仅 transpose，参见 `16-shape.md §5.1`）    |
| 安全构造    | 从形状和数据构造，验证合法性                                                  | 索引操作（参见 `17-indexing.md §1`）                     |
| unsafe 构造 | `from_raw_parts`，用于 FFI                                                    | 切片操作（参见 `17-indexing.md §5`）                     |
| 视图方法    | view/view_mut                                                                 | 集合操作（参见 `14-set.md §1`）                          |

### 1.2 设计原则

| 原则       | 体现                                         |
| ---------- | -------------------------------------------- |
| 零开销抽象 | 不同存储模式在运行时无额外开销               |
| 类型安全   | 通过泛型约束在编译期保证访问权限             |
| 统一接口   | 所有张量类型共享相同的核心 API               |
| 最小核心   | 核心结构仅包含必要字段，功能通过扩展方法提供 |
| 栈上元数据 | 静态维度的 TensorBase 元数据完全在栈上       |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent from layout)
L4: tensor (depends on storage, dimension, layout)  ← current module
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                  |
| -------- | --------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §6、§7、§8、§10、§19、§22、§27、§28                         |
| 范围内   | `TensorBase<S, D>`、类型别名、基础查询、构造校验、视图与 raw-parts 契约 |
| 范围外   | 广播、索引、reshape、归约与逐元素运算                                 |
| 非目标   | 引入运行时动态张量类型系统、隐藏存储模式差异或跳过元数据合法性校验     |

> **需求 §6 边界说明：** 存储模式转换矩阵与具体转换 API 由 `05-storage.md` 承载实现设计；本文档仅定义 `storage_kind()`、view/raw-parts 与张量查询接口，不重复展开转换细节。

---

## 3. 文件位置

```
src/tensor/
├── mod.rs             # TensorBase<S, D> struct definition and public exports
├── impls.rs           # core query method implementations
├── aliases.rs         # type alias definitions
└── construct.rs       # internal constructors (unsafe low-level construction)
```

文件划分理由：结构体定义、方法实现、类型别名、构造方法各自独立且职责清晰。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
┌─────────────────────────────────────────────────────────────┐
│                     TensorBase<S, D>                        │
│                   (src/tensor/mod.rs)                       │
└────────────────────────┬────────────────────────────────────┘
                         │ uses
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   storage     │ │   dimension   │ │    layout     │
│ - Owned<A>    │ │ - Ix0-Ix6     │ │ - LayoutFlags │
│ - ViewRepr    │ │ - IxDyn       │ │ - is_f_contig │
│ - ViewMutRepr │ │ - Dimension   │ │ - strides     │
│ - ArcRepr     │ │   trait       │ │   compute     │
│ - Storage     │ │               │ │               │
│   trait       │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `storage`   | `Owned<A>`, `ViewRepr<'a, A>`, `ViewMutRepr<'a, A>`, `ArcRepr<A>`, `Storage`, `StorageMut`, `StorageOwned`, `StorageShared`（参见 `05-storage.md §5`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `.slice()`, `.checked_size()`, `.ndim()`（参见 `02-dimension.md §5`）                                              |
| `layout`    | `LayoutFlags`, `compute_f_strides()`, `is_f_contiguous()`, `is_aligned()`（参见 `06-layout.md §5`）                                                    |

### 4.2a 依赖合法性

| 项目           | 结论                           |
| -------------- | ------------------------------ |
| 新增第三方依赖 | 无                             |
| 合法性结论     | 符合需求说明书最小依赖限制     |
| 替代方案       | 不适用                         |

> 注意：`06-layout.md` 的当前结论是不再引入公开 `Layout` 结构体；`TensorBase` 直接内联 `offset` 与 `LayoutFlags` 等布局元数据。

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `tensor/` 消费 `storage`、`dimension`、`layout` 的 trait 和类型，不被它们依赖。`math/`、`iter/` 等上层模块消费 `tensor`。

---

## 5. 公共 API 设计

### 5.1 TensorBase\<S, D\> 核心结构体

```rust,ignore
/// Core abstraction for multi-dimensional arrays.
///
/// # Type Parameters
///
/// * `S` - Storage mode, determining ownership and access rights
/// * `D` - Dimension type, determining rank and shape representation
///
/// # Memory Layout
///
/// Struct size depends on the concrete instantiation of S and D. For static dimensions (Ix0-Ix6),
/// D is a stack-allocated fixed-size array; for dynamic dimensions (IxDyn), D contains a heap-allocated Vec.
pub struct TensorBase<S, D> {
    /// Underlying data storage.
    storage: S,

    /// Length of each axis.
    shape: D,

    /// Stride of each axis (in element units).
    ///
    /// Strides are modeled separately from shape via `Strides<D>` so that
    /// zero strides remain explicit layout metadata.
    strides: Strides<D>,

    /// Non-negative displacement from the storage base pointer to the logical first
    /// element (in element units).
    ///
    /// `storage.as_ptr()` / `storage.as_mut_ptr()` always return the storage base pointer.
    /// Public raw-pointer APIs such as `TensorBase::as_ptr()` apply `offset` exactly once.
    /// View constructors keep `offset` as the non-negative displacement from the
    /// storage base pointer to the logical first element.
    offset: usize,

    /// Layout flags (u8 bitflags).
    ///
    /// Caches contiguity, alignment, and zero-stride info for O(1) queries.
    flags: LayoutFlags,
}
```

> **FFI 布局说明：** `TensorBase` 不对外承诺稳定的结构体内存布局，也不作为 FFI 边界类型。FFI 消费者应优先使用 `23-ffi.md` 的 `TensorExport`，而非直接依赖 `TensorBase` 的字段顺序或 ABI 表示。

> **设计说明：** `TensorBase` 直接嵌入 `offset` 和 `flags` 字段。
> 这是因为 `offset` 与存储指针配合进行偏移计算，属于张量实例的固有属性；直接内联这些布局元数据可以避免额外的间接层，并保持与 `06-layout.md` 的当前结论一致。

> **raw-parts 契约：** `from_raw_parts*()` 系列中的 `ptr` 一律表示 storage base pointer，
> `offset` 一律表示从 storage base 到逻辑首元素的非负位移；`TensorBase::as_ptr()` /
> `TensorBase::as_mut_ptr()` 负责应用这一次偏移。`ffi` 文档中的示例与 Safety 说明必须遵循同一语义。

> **线程安全推导**: `TensorBase<S, D>` 的 `Send`/`Sync` 由存储模式 `S` 和元素类型 `A` 共同决定：`S` 提供 `Send`/`Sync`（参见 `05-storage.md §5.3`），`A` 须满足对应的线程安全约束（参见 `25-safety.md §4`）。

| 张量存储模式 | `Send` 条件 | `Sync` 条件 | 说明 |
| ------------ | ----------- | ----------- | ---- |
| `Tensor<Owned<A>, D>` | 取决于 `Owned<A>: Send` | 取决于 `Owned<A>: Sync` | 拥有型规则与 `05-storage.md`、`25-safety.md` 一致 |
| `TensorView<'a, A, D>` | 取决于 `ViewRepr<'a, A>: Send` | 取决于 `ViewRepr<'a, A>: Sync` | 只读借用可跨线程共享的前提由 storage 层定义 |
| `TensorViewMut<'a, A, D>` | 取决于 `ViewMutRepr<'a, A>: Send` | 不成立 | 可变视图只允许独占传播 |
| `ArcTensor<A, D>` | 取决于 `ArcRepr<A>: Send` | 取决于 `ArcRepr<A>: Sync` | 共享只读线程安全前提完全继承 storage 层 |

### 5.2 Type aliases (full list)

```rust
// === Primary type aliases ===

/// Owning multi-dimensional array.
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;

/// Immutable view.
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<'a, A>, D>;

/// Mutable view.
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<'a, A>, D>;

/// Atomically reference-counted shared array.
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;

// === Owned dimension convenience aliases ===

pub type Tensor0<A> = Tensor<A, Ix0>;
pub type Tensor1<A> = Tensor<A, Ix1>;
pub type Tensor2<A> = Tensor<A, Ix2>;
pub type Tensor3<A> = Tensor<A, Ix3>;
pub type Tensor4<A> = Tensor<A, Ix4>;
pub type Tensor5<A> = Tensor<A, Ix5>;
pub type Tensor6<A> = Tensor<A, Ix6>;
pub type TensorD<A> = Tensor<A, IxDyn>;

// === View dimension convenience aliases ===

pub type TensorView0<'a, A> = TensorView<'a, A, Ix0>;
pub type TensorView1<'a, A> = TensorView<'a, A, Ix1>;
pub type TensorView2<'a, A> = TensorView<'a, A, Ix2>;
pub type TensorView3<'a, A> = TensorView<'a, A, Ix3>;
pub type TensorView4<'a, A> = TensorView<'a, A, Ix4>;
pub type TensorView5<'a, A> = TensorView<'a, A, Ix5>;
pub type TensorView6<'a, A> = TensorView<'a, A, Ix6>;
pub type TensorViewD<'a, A> = TensorView<'a, A, IxDyn>;

// === ViewMut dimension convenience aliases ===

pub type TensorViewMut0<'a, A> = TensorViewMut<'a, A, Ix0>;
pub type TensorViewMut1<'a, A> = TensorViewMut<'a, A, Ix1>;
pub type TensorViewMut2<'a, A> = TensorViewMut<'a, A, Ix2>;
pub type TensorViewMut3<'a, A> = TensorViewMut<'a, A, Ix3>;
pub type TensorViewMut4<'a, A> = TensorViewMut<'a, A, Ix4>;
pub type TensorViewMut5<'a, A> = TensorViewMut<'a, A, Ix5>;
pub type TensorViewMut6<'a, A> = TensorViewMut<'a, A, Ix6>;
pub type TensorViewMutD<'a, A> = TensorViewMut<'a, A, IxDyn>;

// === Arc dimension convenience aliases ===

pub type ArcTensor0<A> = ArcTensor<A, Ix0>;
pub type ArcTensor1<A> = ArcTensor<A, Ix1>;
pub type ArcTensor2<A> = ArcTensor<A, Ix2>;
pub type ArcTensor3<A> = ArcTensor<A, Ix3>;
pub type ArcTensor4<A> = ArcTensor<A, Ix4>;
pub type ArcTensor5<A> = ArcTensor<A, Ix5>;
pub type ArcTensor6<A> = ArcTensor<A, Ix6>;
pub type ArcTensorD<A> = ArcTensor<A, IxDyn>;
```

### 5.3 基础信息查询方法

```rust,ignore
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// Returns a slice of axis lengths.
    pub fn shape(&self) -> &[usize];

    /// Returns a slice of strides (usize, in element units).
    ///
    /// Strides may be zero for broadcast dimensions.
    pub fn strides(&self) -> &[usize];

    /// Returns the number of dimensions.
    ///
    /// For static dimensions (Ix0-Ix6), this is a compile-time constant.
    /// For dynamic dimensions (IxDyn), this is a runtime value.
    pub fn ndim(&self) -> usize;

    /// Returns the total number of logical elements (product of all dimension lengths).
    ///
    /// Current design note: this query is O(ndim). If O(1) is required in the
    /// future, `TensorBase` may cache the logical element count explicitly.
    pub fn len(&self) -> usize;

    /// Returns whether the array is empty (any dimension length is 0).
    pub fn is_empty(&self) -> bool;

    /// Returns the data start offset (in element units).
    pub fn offset(&self) -> usize;

    /// Returns the complete layout flags.
    pub fn flags(&self) -> LayoutFlags;

    /// Returns the storage-location classification of the tensor payload.
    pub fn storage_kind(&self) -> StorageKind;

    /// Returns the physical data location of the tensor payload.
    pub fn data_location(&self) -> DataLocation;

    /// Returns the layout-state classification of the logical tensor view.
    pub fn layout_state(&self) -> LayoutState;

    /// Whether the data is F-order contiguous.
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.is_f_contiguous()
    }

    /// Whether the data is 64-byte aligned.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }

    /// Whether there is a zero stride (broadcast dimension).
    #[inline]
    pub fn has_zero_stride(&self) -> bool {
        self.flags.has_zero_stride()
    }

}

impl<S, D> TensorBase<S, D>
where
    D: Dimension + Clone,
{
    /// Returns a clone of the dimension type.
    pub fn raw_dim(&self) -> D;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    Owned,
    View,
    ViewMut,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocation {
    Cpu,
}
```

> **`len` / storage 长度不变量：** `TensorBase::len()` 返回逻辑元素总数（由 `shape` 计算）；`Storage::len()` 返回底层存储的可见长度。对于视图类型，storage len 可能大于 logical len。所有 bounds check 基于 logical len，raw-parts 构造基于 storage len。

> **数据位置查询说明：** 当前版本仅支持 CPU 内存，`data_location()` 恒返回 `DataLocation::Cpu`，用于满足 `require.md §8` 的存储位置查询接口。

> **`storage_kind()` 语义说明：** `storage_kind()` 返回底层**实际存储表示类型**对应的 `Owned / View / ViewMut / Shared`，而不是高层语义分类。`OwnedRepr` 报告 `Owned`，`ViewRepr` 报告 `View`，`ViewMutRepr` 报告 `ViewMut`，`ArcRepr` 报告 `Shared`。因此广播结果若底层表示为 `ViewRepr`，其 `storage_kind()` 也必须返回 `View`，而不是 `Shared`。

> **广播语义补充：** 广播结果的只读共享语义通过 layout flags 和访问控制表达，而非通过 `storage_kind()` 伪装。详见 `15-broadcast.md`。

> **三层语义模型：**
>
> 1. **存储表示层**：`Owned` / `View` / `ViewMut` / `Shared`，即 `storage_kind()` 的返回值，描述底层 representation。
> 2. **访问语义层**：`ReadOnly` / `SharedReadOnly` / `Writable`，由存储表示与当前访问约束共同导出。
> 3. **布局状态层**：`FContiguous` / `NonContiguous` / `BroadcastView`，由 `LayoutFlags` / `LayoutState` 描述。
>
> 广播张量通常表现为“表示层 = `View`，访问语义层 = `SharedReadOnly`，布局状态层 = `BroadcastView`”。

> `LayoutState` 使用 `crate::layout::LayoutState`（参见 `06-layout.md §5`）；
> 本文档不再重复定义 `FContiguous`、`NonContiguous`、`BroadcastView` 三个变体。

### 5.4 指针访问方法

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a raw pointer to the logical first element.
    ///
    /// For empty tensors, this returns `NonNull::dangling().as_ptr()` and does
    /// not perform pointer arithmetic on the storage base pointer.
    pub fn as_ptr(&self) -> *const A;

    /// Returns the raw storage base pointer WITHOUT adding the offset.
    ///
    /// Unlike `as_ptr()` which returns `storage.as_ptr().add(offset)`,
    /// this method returns `storage.as_ptr()` directly — the raw base
    /// pointer of the storage buffer. The caller is responsible for
    /// manually accounting for `self.offset` when computing element
    /// addresses.
    ///
    /// The returned pointer does NOT point to the first logical element;
    /// use `as_ptr()` for that. Any pointer arithmetic based on this
    /// pointer must include `self.offset` to access the correct data.
    pub fn as_storage_ptr(&self) -> *const A;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Returns a mutable raw pointer to the data start.
    ///
    /// For empty tensors, this returns `NonNull::dangling().as_ptr()` and does
    /// not perform pointer arithmetic on the storage base pointer.
    pub fn as_mut_ptr(&mut self) -> *mut A;
}
```

### 5.4a 连续切片访问方法

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a shared slice when the logical tensor is F-contiguous and the
    /// logical-first pointer contract is satisfied.
    ///
    /// This is the zero-copy fast path consumed by `simd/`, `parallel/`, and
    /// convenience APIs such as `set::unique()` examples. Non-contiguous views
    /// and broadcast views return `None` and must fall
    /// back to iterator-based access. A non-zero logical offset alone does not
    /// disqualify the fast path: if `as_ptr()` already points at the logical
    /// first element and the layout is contiguous, `as_slice()` may still be
    /// returned zero-copy.
    /// Empty tensors return `Some(&[])`.
    pub fn as_slice(&self) -> Option<&[A]>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Returns a mutable slice when the logical tensor is F-contiguous, has no
    /// zero strides, and the logical-first element is uniquely writable.
    ///
    /// Broadcast results are immutable by construction and therefore can never
    /// satisfy this method's preconditions. As with `as_slice()`, a non-zero
    /// logical offset is acceptable as long as `as_mut_ptr()` points at the
    /// logical first element and the logical layout remains contiguous.
    /// Empty tensors return `Some(&mut [])`.
    pub fn as_mut_slice(&mut self) -> Option<&mut [A]>;
}
```

### 5.5 安全构造方法

> **构造责任边界：** 安全构造路径必须验证全部可验证元数据约束，至少包括 shape/stride 可表示性、元素总数计算不溢出、以及逻辑访问范围不越界。`from_shape_vec` 这类 API 不得把这些前提留给调用方；safe 构造负责兜底全部可检查元数据条件。

````rust,ignore
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Constructs an owning tensor from shape and data, validating correctness.
    ///
    /// # Arguments
    ///
    /// * `shape` - Length of each axis
    /// * `data` - Element data following logical-index correspondence semantics
    ///   defined by `require.md §19`; the input order defines which element belongs
    ///   to each logical index, rather than requiring callers to pre-arrange bytes
    ///   in a specific physical layout
    ///
    /// # Errors
    ///
    /// Returns `Err` when:
    /// - `shape.checked_size()` overflows
    /// - `data.len() != shape.checked_size()`
    /// - F-order stride derivation fails
    /// - underlying storage construction fails
    ///
    /// Error conditions include but are not limited to: shape-product overflow,
    /// mismatch between logical element count and input data length, failure to
    /// derive canonical F-order strides, and failure to construct the underlying
    /// storage. Any unmet condition returns `XenonError`.
    ///
    /// Xenon follows `require.md §19`: "the order of the input data defines the element-to-logical-index correspondence."
    /// The current version defaults to 64-byte-aligned allocation
    /// (for example via `Owned::from_vec_aligned`), consistent with `05-storage.md`.
    /// This aligned path is the default owned-storage policy; any exception must be
    /// explicitly documented by the corresponding constructor and still preserve
    /// the same logical element order. See `05-storage.md §5` and `18-construction.md §5.3`。
    /// Owned tensors constructed from shape + data also use the canonical packed
    /// F-order stride for their logical layout; any mentioned "padding" refers only
    /// to allocation-level tail capacity, not to logical tensor stride gaps. See
    /// `06-layout.md` Decision 5 for the full ADR.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = Tensor2::<f64>::from_shape_vec([3, 4], vec![1.0; 12])?;
    /// ```
    pub fn from_shape_vec(shape: D, data: Vec<A>) -> Result<Self, XenonError>;

    /// Constructor-module APIs such as `zeros()` and `ones()` also return
    /// `Result<Self, XenonError>` and must be used with `?` or `.unwrap()` in
    /// examples and calling code.
    // NOTE: the default owned-storage path uses aligned allocation.
    // The concrete alignment policy belongs to the storage module.

    /// Construct a tensor from a Vec without validating shape/stride consistency.
    ///
    /// # Safety
    /// - `data.as_ptr()` must remain valid for the duration of construction, and
    ///   `Vec<A>` must satisfy the alignment requirements of `A`
    /// - `data.len()` must equal the product of all shape dimensions
    /// - That product must be representable in `usize` without overflow
    /// - `shape` must be representable by the current dimension type
    /// - The default packed F-order stride derived from `shape` must be
    ///   representable and consistent with `require.md §7`
    /// - The constructor assumes no extra offset and therefore treats the input
    ///   buffer as the full logical tensor payload
    pub(crate) unsafe fn from_raw_vec_unchecked(data: Vec<A>, shape: D) -> Self {
        // computes F-order strides internally
        // ...
    }
}
````

### 5.6 unsafe 构造方法

> **unsafe 构造责任边界：** `from_raw_parts*()` 这类接口只验证能够基于输入元数据直接检查的条件；safe 构造会兜底验证全部可检查元数据，而 unsafe 构造仅拒绝明显非法的 shape/stride/offset/storage_len 组合。若这些元数据校验失败，构造器返回 `Err(XenonError::InvalidLayout)`（附带上下文）。调用方仍负责保证指针有效性、对齐、可访问范围和生命周期等库无法自行证明的内存前提。文档中的 `# Safety` 说明必须与这一分工保持一致。

```rust,ignore
impl<'a, A, D> TensorBase<ViewRepr<'a, A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Constructs an immutable view from raw parts.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `ptr` points to the storage base pointer of the view; for empty arrays or
    ///   ZST elements, a well-formed dangling sentinel is permitted because it is never dereferenced
    /// - For empty tensors, metadata validation treats `offset` as a bounded metadata
    ///   value only (`offset <= storage_len`) and performs no actual pointer offset
    /// - Memory remains valid for the lifetime `'a` of the returned view
    /// - Pointer alignment and initialization requirements of `A` are satisfied
    /// - The access range implied by `shape`, `strides`, and `offset` is actually accessible within the backing storage
    ///
    /// The constructor validates metadata that can be checked directly:
    /// - `shape` and `strides` are combinable for this dimension type
    /// - Element-count computation does not overflow
    /// - The logical access range implied by `shape`, `strides`, and `offset`
    ///   fits within `storage_len`
    ///
    /// If metadata validation fails, returns `Err(XenonError::InvalidLayout)`
    /// with context. The unsafe obligation is limited to pointer validity,
    /// alignment, actual accessible range, and lifetime guarantees that the
    /// library cannot verify from metadata alone.
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self, XenonError>;
}

impl<'a, A, D> TensorBase<ViewMutRepr<'a, A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Constructs a mutable view from raw parts.
    ///
    /// # Safety
    ///
    /// Same as `from_raw_parts`, including the rule that empty tensors only require
    /// `offset <= storage_len` and never perform actual pointer offsetting, with the
    /// additional requirement of exclusive access. In particular, mutable views must
    /// not describe overlapping logical elements: if multiple logical indices map to
    /// the same physical address, the constructor must reject the layout and return
    /// an error.
    ///
    /// The constructor returns `Err(XenonError::InvalidLayout)` when directly
    /// checkable metadata validation fails; the unsafe obligation remains the
    /// memory/pointer guarantees that cannot be checked by the library.
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self, XenonError>;
}
```

### 5.7 视图方法

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
{
    /// Creates an immutable view (zero-copy).
    pub fn view(&self) -> TensorView<'_, A, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension + Clone,
{
    /// Creates a mutable view (zero-copy, exclusive access).
    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D>;
}
```

### 5.8 Good/Bad 对比

```rust
// Good - Use generic constraints to accept any readable tensor
fn process<S, D, A>(tensor: &TensorBase<S, D>)
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    let ptr = tensor.as_ptr();
    // ...
}

// Bad - Hardcoded Owned type
fn process_bad<A, D>(tensor: &Tensor<A, D>)
where
    D: Dimension,
{
    let ptr = tensor.as_ptr();
    // ...
}
```

```rust
// Good - Use from_shape_vec to validate correctness
let t = Tensor2::<f64>::from_shape_vec([3, 4], vec![1.0; 12])?;

// Bad - Use unsafe from_raw_parts to skip validation
let t = unsafe {
    TensorView2::from_raw_parts(
        data.as_ptr(),
        data.len(),
        Ix2(3, 4),
        Strides::from_slice(&[1, 3]).unwrap(),
        0,
    )?
};
```

---

## 6. 内部实现设计

### 6.1 步长存储策略

> **设计决策：** `shape` 与 `strides` 分离建模：`shape` 字段类型为 `D`，`strides` 字段类型为 `Strides<D>`。
>
> **实现方案：**
>
> | 层次                 | 类型         | 说明                                                                  |
> | -------------------- | ------------ | --------------------------------------------------------------------- |
> | `TensorBase.strides` | `Strides<D>` | 与 shape 维度数一致，显式保存 stride 元数据                           |
> | `strides()` 返回值   | `&[usize]`   | 直接来自 `Strides<D>`（参见 `06-layout.md §5`）                       |
> | layout 模块计算      | `usize`      | F-order、转置与零步长布局在 layout 层计算（参见 `06-layout.md §5.3` / `§5.7`） |
>
> **权衡：**
>
> - `Strides<D>` 保证 strides 与 shape 维度数相同（编译期）
> - 静态维度使用栈分配数组（性能）
> - 当前版本仅覆盖非负步长与零步长（广播）；负步长布局不在当前版本范围内（参见 `require.md §7`）

### 6.2 offset 字段设计

```
Original array storage: [a, b, c, d, e, f, g, h]
shape: [8], strides: [1], offset: 0

After slicing [2..5]:
storage: [a, b, c, d, e, f, g, h]  // shared, no copy
shape: [3], strides: [1], offset: 2  // metadata adjustment only
Logical view: [c, d, e]
```

**安全性论证**：安全构造路径必须调用 `validate_access_range(shape, strides, offset, storage_len)` 之类的检查来计算所有逻辑索引可达的最小/最大物理偏移，并验证它们都落在底层 storage 范围内。unsafe raw-parts 路径可复用这些检查拒绝明显错误的元数据，但访问范围前提本身仍由调用方保证。只有这些前提成立后，`as_ptr()` 才能把“logical-first pointer”定义为逻辑首元素地址。

> **重要设计约定：** `TensorBase::offset` 是所有存储模式（Owned、ViewRepr、ViewMutRepr、ArcRepr）共用的唯一偏移字段。`ArcRepr` 不存储独立的 offset — 数据访问的起始位置完全由 `TensorBase::offset` 决定。这避免了双重偏移计算的 bug，并使偏移逻辑集中在一处。

> **logical-first pointer 契约：** `TensorBase::as_ptr()` / `as_mut_ptr()` 返回的是逻辑首元素指针，而不是 storage base pointer。layout 标志计算、连续切片快路径和 FFI raw-parts safety 文档都必须使用这一同一约定；若需要 storage base pointer，只能通过 storage 层 API 或 raw-parts 输入显式提供。

> **raw-parts 设计补充：** `storage_len` 是 raw-parts 视图构造的必填输入。`ViewRepr` / `ViewMutRepr` 需要保存 backing storage 的可访问元素数，`validate_access_range(...)` 也必须基于该长度执行边界校验；仅有 `ptr + shape + strides + offset` 不足以安全重建视图。

> **空张量指针说明：** 当 `len == 0` 时，元数据仍可描述一个合法的空视图，但 `as_ptr()` / `as_mut_ptr()` 不能对 storage base pointer 执行 `add(offset)`。Rust 的指针算术要求结果仍落在同一已分配对象内；对悬垂哨兵或空存储基指针做偏移计算会触发未定义行为，且对 ZST 即使执行 `add(0)` 也不应依赖这种做法。设计上因此统一采用“空张量 `offset` 仅需满足 `offset <= storage_len`，但不做实际偏移”的契约，并让指针 API 直接返回 `NonNull::dangling().as_ptr()` 快路径。

```text
validate_access_range(shape, strides, offset, storage_len):
    if shape.checked_size() overflows:
        return Err(XenonError::InvalidLayout {
            operation: "validate_access_range",
            storage_kind: "raw_parts",
            shape: shape.slice().to_vec(),
            strides: strides.as_slice().to_vec(),
            offset,
            storage_len,
            reason: "element count overflow",
        })

    if shape.checked_size() == Ok(0):
        if offset > storage_len:
            return Err(XenonError::InvalidLayout {
                operation: "validate_access_range",
                storage_kind: "raw_parts",
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: "empty tensor requires offset <= storage_len",
            })
        return Ok(())

    max_offset = offset

    for axis in 0..ndim:
        if shape[axis] == 0:
            return Ok(())

        span = (shape[axis] - 1).checked_mul(strides[axis])
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: "validate_access_range",
                storage_kind: "raw_parts",
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: "stride span overflow",
            })?
        max_offset = max_offset.checked_add(span)
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: "validate_access_range",
                storage_kind: "raw_parts",
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: "logical access range overflow",
            })?

    if max_offset >= storage_len:
        return Err(XenonError::InvalidLayout {
            operation: "validate_access_range",
            storage_kind: "raw_parts",
            shape: shape.slice().to_vec(),
            strides: strides.as_slice().to_vec(),
            offset,
            storage_len,
            reason: "logical access range exceeds backing storage",
        })

    return Ok(())
```

### 6.3 内存布局示意

```
Tensor2<f64> = TensorBase<Owned<f64>, Ix2>

┌─────────────────────────────────────────┐
│ storage: Owned<f64>                     │
│   ┌───────────────────────────────────┐ │
│   │ data: AlignedBuf<f64> (64B aligned)│ │
│   │ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]    │ │
│   └───────────────────────────────────┘ │
│ shape: Ix2(2, 3)                        │
│ strides: Strides::from_slice(&[1, 2])   │
│ offset: 0                               │
│ flags: F_CONTIGUOUS | ALIGNED           │
└─────────────────────────────────────────┘

Logical view:
  [[1.0, 3.0, 5.0],
   [2.0, 4.0, 6.0]]
```

---

## 7. 实现任务拆分

### Wave 1: 结构体定义和基础

- [ ] **T1**: 创建 `src/tensor/mod.rs` 骨架
  - 文件: `src/tensor/mod.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: storage、dimension、layout 模块完成
  - 预计: 5 min

- [ ] **T2**: 定义 `TensorBase<S, D>` 结构体
  - 文件: `src/tensor/mod.rs`
  - 内容: 结构体定义，5 个字段：storage、shape、strides、offset、flags
  - 测试: 结构体编译通过
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 定义类型别名 (aliases.rs)
  - 文件: `src/tensor/aliases.rs`
  - 内容: 4 个主类型别名 + 4×8 = 32 个维度便捷别名
  - 测试: 所有别名编译通过
  - 前置: T2
  - 预计: 10 min

### Wave 2: 核心查询方法

- [ ] **T4**: 实现形状与步长查询方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `shape()`/`strides()`/`ndim()`/`len()`/`is_empty()`/`offset()`/`raw_dim()`/`flags()`/`storage_kind()`
  - 测试: `test_shape_query`, `test_len_empty`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现布局查询委托方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `layout_state()`/`is_f_contiguous()`/`is_aligned()`/`has_zero_stride()`
  - 测试: `test_layout_flags_delegate`, `test_layout_state_classification`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6**: 实现指针访问方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `as_ptr()`/`as_storage_ptr()`/`as_mut_ptr()`
  - 测试: `test_as_ptr`, `test_as_mut_ptr`
  - 前置: T4
  - 预计: 10 min

### Wave 3: 构造和视图

- [ ] **T7**: 实现 `from_raw_parts` 系列 (construct.rs)
  - 文件: `src/tensor/construct.rs`
  - 内容: `from_raw_parts`(不可变)/`from_raw_parts_mut`(可变)，显式接收 `storage_len` 并统一走 `validate_access_range`
  - 测试: `test_from_raw_parts_view`, `test_from_raw_parts_mut`, `test_from_raw_parts_invalid_range`
  - 前置: T2
  - 预计: 10 min

- [ ] **T8**: 实现安全构造方法 (construct.rs)
  - 文件: `src/tensor/construct.rs`
  - 内容: `from_shape_vec`/`from_raw_vec_unchecked`(内部方法)
  - 测试: `test_from_shape_vec_valid`, `test_from_shape_vec_invalid`
  - 前置: T5, T7
  - 预计: 10 min

- [ ] **T9**: 实现视图创建方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `view()`/`view_mut()`
  - 测试: `test_view_create`, `test_view_mut_create`
  - 前置: T6
  - 预计: 10 min

### Wave 4: 测试和收尾

- [ ] **T10**: 集成测试和文档
  - 文件: `tests/test_tensor.rs`
  - 内容: 跨模块交互测试、边界测试、类型别名编译验证
  - 测试: 完整集成测试套件
  - 前置: T3, T9
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] → [T2] → [T3]
                ↓
Wave 2:        [T4] → [T5]
                ↓      ↓
               [T6]   [T7]
                ↓      ↓
Wave 3:       [T8] → [T9]
                ↓
Wave 4:       [T10]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型     | 位置                     | 目的                                            |
| -------- | ------------------------ | ----------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个方法                                    |
| 集成测试 | `tests/`                 | 验证跨模块交互                                  |
| 边界测试 | 集成测试中标注           | 空数组、单元素、高维                            |
| 编译测试 | `tests compile_fail`     | 验证类型约束                                    |
| 属性测试 | `tests/property/`        | 验证长度、shape/stride 与 view/raw-parts 不变量 |

### 8.2 集成测试函数列表

以下集成测试函数验证 TensorBase 跨模块边界的正确性：

| 测试函数                                 | 测试内容                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `test_tensor_cross_dim_interop`          | TensorBase 与 Dimension 模块交互：验证 Ix0~Ix6 和 IxDyn 的 shape/strides 查询         |
| `test_tensor_storage_layout_integration` | TensorBase 与 Storage/Layout 模块交互：验证 from_shape_vec 后的标志位计算和指针正确性 |
| `test_tensor_view_mut_roundtrip`         | 验证 `view_mut()` 可直接获取零拷贝可变视图并回写到底层数据                            |

### 8.3 单元测试清单

| 测试函数                            | 测试内容                                             | 优先级 |
| ----------------------------------- | ---------------------------------------------------- | ------ |
| `test_tensor_shape_2d`              | `Tensor2::from_shape_vec([3,4], data)` 后 shape 查询 | 高     |
| `test_tensor_len`                   | `len()` 返回 shape 乘积                              | 高     |
| `test_tensor_is_empty`              | 空数组 `is_empty()` 返回 true                        | 高     |
| `test_tensor_ndim_static`           | `Tensor2` 的 `ndim()` == 2                           | 高     |
| `test_tensor_ndim_dynamic`          | `TensorD` 的 `ndim()` 运行时                         | 中     |
| `test_tensor_strides_f_order`       | F-order 步长正确 `[1, shape[0], ...]`                | 高     |
| `test_tensor_layout_state`          | 按 `crate::layout::LayoutState`（参见 `06-layout.md §5`）完成分类 | 高     |
| `test_tensor_flags_f_contiguous`    | 新构造张量 F-连续                                    | 高     |
| `test_tensor_flags_aligned`         | 新构造张量对齐                                       | 高     |
| `test_tensor_as_ptr`                | 指针指向正确位置                                     | 高     |
| `test_tensor_as_mut_ptr`            | 可变指针指向正确位置                                 | 高     |
| `test_tensor_storage_kind`          | `Owned`/`View`/`ViewMut`/`Shared` 的存储位置查询正确 | 高     |
| `test_tensor_view`                  | `view()` 创建正确视图                                | 高     |
| `test_tensor_view_mut`              | `view_mut()` 创建正确可变视图                        | 高     |
| `test_from_shape_vec_valid`         | 合法构造成功                                         | 高     |
| `test_from_shape_vec_len_mismatch`  | 长度不匹配返回错误                                   | 高     |
| `test_from_raw_parts_invalid_range` | raw-parts 越界访问范围被拒绝                         | 高     |
| `test_type_aliases_compile`         | 所有类型别名编译通过                                 | 高     |
| `test_tensor0_scalar`               | 0D 标量张量 `len()==1`                               | 中     |
| `test_tensor_empty_dim`             | 含 0 维度的张量 `is_empty()`                         | 中     |

### 8.4 边界测试场景

| 场景                  | 预期行为                       |
| --------------------- | ------------------------------ |
| 空张量 `shape=[0, 3]` | `len()==0`, `is_empty()==true` |
| 单元素 `shape=[1, 1]` | `len()==1`, F-连续             |
| 标量 `Tensor0<f64>`   | `ndim()==0`, `len()==1`        |
| 高维 `Tensor6`        | `ndim()==6`, 步长正确          |
| 动态维度 `TensorD`    | `ndim()` 运行时值正确          |
| 大张量 `10^7` 元素    | 构造成功，长度与 flags 保持正确 |
| 非连续转置视图        | 可构造 `view()`，但连续切片快路径返回 `None` |
| 空张量 + 多种 offset  | 只要 `offset <= storage_len` 即合法 |
| 非法元素类型编译失败  | compile-fail 测试拒绝不满足元素约束的类型 |

### 8.4a 需求说明书 §28.4 边界测试占位

| 占位项 | 说明 |
| ------ | ---- |
| 空张量边界 | 占位：覆盖 `as_ptr()` dangling sentinel、`as_slice()` / `as_mut_slice()` 返回空切片 |
| 大张量边界 | 占位：覆盖超大 shape 的 `checked_size()`、stride 计算与构造错误传播 |
| 高维张量边界 | 占位：覆盖高维 `TensorD` / `Tensor6` 的 shape、strides 与 `layout_state()` 查询 |

### 8.5 属性测试不变量

| 不变量                                            | 测试方法                                  |
| ------------------------------------------------- | ----------------------------------------- |
| `tensor.len() == tensor.shape().iter().product()` | 随机形状                                  |
| `tensor.view().shape() == tensor.shape()`         | 随机形状和存储模式                        |
| `from_shape_vec` 后 `is_f_contiguous() == true`   | 随机合法形状                              |
| 安全构造路径在访问范围不合法时返回错误            | 随机 shape/stride/offset/storage_len 组合 |

### 8.6 集成测试

| 测试文件               | 测试内容                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `tests/test_tensor.rs` | `from_shape_vec` / `view` / `view_mut` / `as_ptr` 与 `dimension`、`storage`、`layout`、`index` 的端到端协同路径 |

### 8.7 数据流描述

```text
User calls constructors / `view()` / `view_mut()` / query APIs
    │
    ├── dimension provides shape metadata
    ├── storage provides the backing buffer and ownership model
    ├── tensor combines shape + strides + offset + flags
    ├── layout computes contiguity / alignment / zero-stride flags
    └── index / iter / math / ffi and other upper layers continue consuming `TensorBase` as the unified carrier
```

### 8.8 Feature gate / 配置测试

| 配置项 | 覆盖方式                             | 说明                                         |
| ------ | ------------------------------------ | -------------------------------------------- |
| 默认配置 | 常规单元/集成测试路径                 | 本模块无独立 feature gate，默认配置即主路径  |
| 非默认 feature | 不适用                             | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.9 类型边界 / 编译期测试

| 测试类型 | 覆盖方式                                          | 说明                                                     |
| -------- | ------------------------------------------------- | -------------------------------------------------------- |
| 存储访问边界 | compile-fail 测试只读存储不暴露可写 API             | 验证 `Storage` / `StorageMut` 约束在 `TensorBase` 上正确投影 |
| 别名边界 | 编译期验证 `Tensor{N}` / `TensorView{N}` / `ArcTensor{N}` 全部展开 | 验证便捷别名与核心类型实例化保持一致                     |
| raw-parts 边界 | 编译期与运行时测试结合验证 `Strides<D>` / `D` / `offset` 契约 | 验证元数据契约不会被类型层或构造层打破                   |

---

## 9. 与其他模块的交互

### 9.1 与 storage 模块的接口

| 接口 | 方向 | 契约 |
| ---- | ---- | ---- |
| `Storage::as_ptr()` / `StorageMut::as_mut_ptr()` | `tensor` 消费 `storage` | storage 层返回 storage base pointer；`TensorBase` 负责叠加 `offset` 并形成 logical-first pointer |
| `Owned::from_vec_aligned(data)` | `tensor` 消费 `storage` | 当前版本默认采用 64 字节对齐分配策略；若存在例外，须显式文档化，且不得改变 `require.md §19` 规定的逻辑元素顺序 |
| `Storage<Elem = A>` / `StorageMut<Elem = A>` | `tensor` 消费 `storage` trait | 元素类型、只读/可写访问能力完全由存储模式 trait 约束决定，`tensor` 不重复维护独立元素类型参数 |

```rust,ignore
// TensorBase obtains element type via Storage trait's associated type
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn as_ptr(&self) -> *const A {
        if self.is_empty() {
            return NonNull::<A>::dangling().as_ptr();
        }
        // storage.as_ptr() returns the storage base pointer; TensorBase converts it
        // to the logical-first pointer after construction invariants have been validated.
        unsafe { self.storage.as_ptr().add(self.offset) }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    pub fn as_mut_ptr(&mut self) -> *mut A {
        if self.is_empty() {
            return NonNull::<A>::dangling().as_ptr();
        }
        unsafe { self.storage.as_mut_ptr().add(self.offset) }
    }
}
```

### 9.2 与 dimension 模块的接口

| 接口 | 方向 | 契约 |
| ---- | ---- | ---- |
| `Dimension::slice()` | `tensor` 消费 `dimension` | `shape()` 返回的轴长切片与底层 `D` 保持一致，不复制逻辑元数据语义 |
| `Dimension::checked_size()` | `tensor` 消费 `dimension` | `tensor` 基于已验证 shape 暴露 `len()`；凡直接消费 `shape: D` 做构造、分配或范围校验时必须先用 `checked_size()` 避免溢出 |
| `Dimension` rank contract | `tensor` 消费 `dimension` | `shape` 与 `Strides<D>` 必须保持相同 rank，供 `simd` / `parallel` 继续消费同一布局契约 |

```rust,ignore
// Dimension trait provides shape operations
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }

    pub fn len(&self) -> usize {
        // TensorBase only reaches this query path after construction-time shape
        // validation, so checked_size() must already succeed here.
        self.shape.checked_size().expect("tensor shape must be validated before len()")
    }
}
```

> **实现约束重申：** `len()` 的返回值始终来源于逻辑 `shape`，不允许退化为读取 `storage.len()`；后者仅用于 raw-parts 与底层访问范围校验。

### 9.3 与 layout 模块的接口

| 接口 | 方向 | 契约 |
| ---- | ---- | ---- |
| `layout::compute_f_strides(&shape)` | `tensor` 消费 `layout` | 安全构造拥有型连续张量时统一按 F-order 生成 stride |
| `layout::compute_layout_flags(&shape, &strides, logical_ptr)` | `tensor` 消费 `layout` | `flags` 的计算必须基于 logical-first pointer 契约，与 `as_ptr()` / `as_slice()` 的可见语义一致 |
| `LayoutState` / layout flags queries | `simd`、`parallel` 继续消费 `tensor` 暴露结果 | 上游加速模块只通过 `TensorBase` 查询连续性、对齐和广播状态，不绕过 `tensor` 直接重建布局判断 |

```rust,ignore
// Layout module provides stride computation and contiguity checks
// TensorBase computes LayoutFlags during construction
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    pub fn from_shape_vec(shape: D, data: Vec<A>) -> Result<Self, XenonError> {
        let expected = shape.checked_size().map_err(|_| XenonError::InvalidShape {
            operation: "from_shape_vec",
            shape: shape.slice().to_vec(),
            expected_elements: 0,
            actual_elements: data.len(),
            offending_dim: None,
            reason: Some("element count overflow".into()),
        })?;
        if data.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: "from_shape_vec",
                shape: shape.slice().to_vec(),
                expected_elements: expected,
                actual_elements: data.len(),
                offending_dim: None,
                reason: Some("data length does not match shape element count".into()),
            });
        }
        let strides = layout::compute_f_strides(&shape)?;
        // The current version defaults to a fresh 64-byte aligned allocation.
        // Any exception must be explicitly documented by the corresponding
        // constructor path. See 05-storage.md §5 and 18-construction.md §5.3.
        let storage = Owned::from_vec_aligned(data)?;
        let logical_ptr = storage.as_ptr();
        let flags = layout::compute_layout_flags(&shape, &strides, logical_ptr);
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            flags,
        })
    }
}
```

---

## 10. 错误处理与语义边界

| 项目           | 内容 |
| -------------- | ---- |
| Recoverable error | `from_shape_vec()`、`from_raw_parts*()` 等构造校验失败返回 `XenonError`；上下文字段应包含操作名、shape、strides、offset、storage_len 或期望长度等元数据 |
| Panic | 本模块公开安全构造不以 panic 作为常规错误通道；仅在内部已验证快捷路径或明显违背 `unsafe` 前提的后续使用中可能出现 panic/未定义行为风险 |
| 路径一致性 | scalar、SIMD 快路径与 parallel 上游消费必须共享同一逻辑首元素与 flags 语义，不允许因路径差异改变结果 |
| 容差边界 | 不适用 |

---

## 11. 设计决策记录

### 决策 1：TensorBase\<S, D\> 双参数泛型设计

| 属性     | 值                                                                                           |
| -------- | -------------------------------------------------------------------------------------------- |
| 决策     | 使用 `TensorBase<S, D>` 双参数泛型，S 为存储模式，D 为维度类型                               |
| 理由     | 零开销（编译期单态化）；类型安全（编译期禁止只读视图写入）；统一接口（所有存储模式共享 API） |
| 替代方案 | `TensorBase<A, S, D>` 三参数 — 放弃，A 可从 S 推导，冗余                                     |
| 替代方案 | 分离类型（Tensor/TensorView 独立结构体） — 放弃，代码重复                                    |
| 替代方案 | 单一 `Tensor<A, D>` + 运行时标志 — 放弃，运行时开销                                          |

### 决策 2：步长存储策略

| 属性     | 值                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------- |
| 决策     | `strides` 字段使用 `Strides<D>` 独立类型存储                                                   |
| 理由     | 显式保留 stride 元数据；与 `shape: D` 职责分离；静态维度仍可栈分配，动态维度仍可保持维度数一致 |
| 替代方案 | `strides: Vec<isize>` — 放弃，静态维度也要堆分配                                               |
| 替代方案 | `strides: [isize; N]` — 放弃，不支持动态维度                                                   |
| 替代方案 | `strides` 复用 `D` 类型 — 放弃，无法显式表达 stride 元数据，且会混淆 shape 与 layout 的职责    |

### 决策 3：offset 字段必要性

| 属性     | 值                                                                             |
| -------- | ------------------------------------------------------------------------------ |
| 决策     | 包含 `offset: usize` 字段                                                      |
| 理由     | 切片操作 O(1)（仅修改元数据）；无数据复制；统一机制适用所有存储模式；BLAS 兼容 |
| 替代方案 | 无 offset，切片时调整 storage 指针 — 放弃，Owned 无法调整指针                  |

### 决策 4：不实现 Deref\<Target=TensorView\>

| 属性     | 值                                                                            |
| -------- | ----------------------------------------------------------------------------- |
| 决策     | 不实现 `Deref<Target = TensorView>`                                           |
| 理由     | 显式优于隐式（`.view()` 清晰表达意图）；避免隐式生命周期传播；与 ndarray 一致 |
| 替代方案 | 实现 Deref — 放弃，隐式转换可能导致意外借用                                   |

---

## 12. 性能考量

| 方面       | 设计决策                                          |
| ---------- | ------------------------------------------------- |
| 栈上元数据 | 静态维度（Ix0-Ix6）的 TensorBase 元数据完全在栈上 |
| 零成本抽象 | 不同存储模式编译为不同类型，无虚调用              |
| 查询复杂度 | `shape()`/`ndim()`/`flags()` 为 O(1)，`len()` 当前为 O(ndim) |
| 视图零拷贝 | `view()`/`view_mut()` 仅复制元数据                |
| 单态化     | Dimension + Storage trait 在泛型上下文中单态化    |

**TensorBase 大小分析（参考）**：

| 实例化             | 大小（估算） | 说明                                                                                                       |
| ------------------ | ------------ | ---------------------------------------------------------------------------------------------------------- |
| `Tensor2<f64>`     | ~72 bytes    | Owned(24) + Ix2(16) + Strides<Ix2>(16) + usize(8) + u8(1) + padding(7) = 72 bytes                          |
| `TensorView2<f64>` | ~56 bytes    | ViewRepr<'a, f64>(metadata + pointer) + Ix2(16) + Strides<Ix2>(16) + usize(8) + u8(1) + padding ≈ 56 bytes |
| `TensorD<f64>`     | ~96 bytes    | Owned(24) + IxDyn(24×2) + usize(8) + u8(1) + padding                                                       |

**性能数据（参考）**：

| 操作               | 开销         | 说明               |
| ------------------ | ------------ | ------------------ |
| `shape()`          | ~1ns         | 切片返回           |
| `len()`            | ~2ns         | 乘积计算           |
| `view()`           | ~5ns         | 元数据复制         |
| `from_shape_vec()` | ~1μs + alloc | 包含验证和步长计算 |

---

## 13. 平台与工程约束

| 约束       | 说明                                    |
| ---------- | --------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`  |
| 单 crate   | 保持单 crate 边界                       |
| SemVer     | 张量元数据字段与构造契约变更遵循 SemVer |
| 最小依赖   | 无新增第三方依赖                        |
| MSRV       | Rust 1.85+                             |

---


## 附录 A：完整类型关系图

```
                        TensorBase<S, D>
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    TensorBase<        TensorBase<          TensorBase<
      Owned<A>,         ViewRepr<'a, A>,    ViewMutRepr<'a, A>,
         D>                    D>                    D>
          │                   │                   │
          ▼                   ▼                   ▼
      Tensor<A,D>      TensorView<'a,A,D>  TensorViewMut<'a,A,D>
          │                   │                   │
    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
    │           │       │           │       │           │
 Tensor1<A> TensorD<A> TensorView1 TensorViewD TensorViewMut1 TensorViewMutD
    │           │       │           │       │           │
   ...         ...     ...         ...     ...         ...
```

## 附录 B：命名约定速查

| 模式               | 示例                    | 含义               |
| ------------------ | ----------------------- | ------------------ |
| `Tensor{N}`        | `Tensor2<A>`            | N 维拥有型数组     |
| `TensorD`          | `TensorD<A>`            | 动态维度拥有型数组 |
| `TensorView{N}`    | `TensorView2<'a, A>`    | N 维不可变视图     |
| `TensorViewMut{N}` | `TensorViewMut2<'a, A>` | N 维可变视图       |
| `ArcTensor{N}`     | `ArcTensor2<A>`         | N 维 Arc 共享数组  |

## 附录 C：数据流图

```
User calls constructor-module API `Tensor::<f64, Ix2>::zeros([3, 4])?`
    │
    ├── `Dimension::ndim()`          → 2
    ├── `Dimension::slice()`         → [3, 4]
    ├── compute element count        → 12
    ├── compute strides (F-order)    → [1, 3]
    ├── aligned allocation 12 * 8 = 96 bytes  → 64-byte aligned
    ├── compute `LayoutFlags`        → F_CONTIGUOUS | ALIGNED
    └── return `Result<TensorBase<Owned<f64>, Ix2>, XenonError>`
```

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-10 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-14 |
| 1.1.4 | 2026-04-15 |
| 1.1.5 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
