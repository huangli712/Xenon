# 张量类型模块设计

> 文档编号: 07
> 模块目录: src/tensor/
> 任务阶段: Phase 3
> 前置文档: 01-architecture.md, 02-dimension.md, 03-element.md, 05-storage.md, 06-layout.md
> 需求参考: 需求说明书 §6 - §8、§10、§19、§22、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责        | 包含                                                                          |
| ----------- | ----------------------------------------------------------------------------- |
| 核心结构体  | `TensorBase<S, D>` 双参数泛型结构体定义                                       |
| 类型别名    | `Tensor`/`TensorView`/`TensorViewMut`/`ArcTensor` 及维度便捷别名              |
| 基础查询    | shape/ndim/len/strides/is_empty/is_f_contiguous/is_aligned/存储位置查询等方法 |
| 安全构造    | 从形状和数据构造，验证合法性                                                  |
| unsafe 构造 | `from_raw_parts`，用于 FFI                                                    |
| 视图方法    | view/view_mut                                                                 |

| 职责        | 不包含                                                   |
| ----------- | -------------------------------------------------------- |
| 核心结构体  | 逐元素与归约逻辑（参见 `11-math.md`、`13-reduction.md`） |
| 类型别名    | 广播规则（参见 `15-broadcast.md §5`）                    |
| 基础查询    | 形状操作（参见 `16-shape.md §5.1`）                      |
| 安全构造    | 索引操作（参见 `17-indexing.md §5`）                     |
| unsafe 构造 | 切片操作（参见 `17-indexing.md §5`）                     |
| 视图方法    | 集合操作（参见 `14-set.md §5`）                          |

### 1.2 设计原则

| 原则       | 体现                                         |
| ---------- | -------------------------------------------- |
| 零开销抽象 | 不同存储模式在运行时无额外开销               |
| 类型安全   | 通过泛型约束在编译期保证访问权限             |
| 统一接口   | 所有张量类型共享相同的核心 API               |
| 最小核心   | 核心结构仅包含必要字段，功能通过扩展方法提供 |
| 栈上元数据 | 静态维度的 TensorBase 元数据完全在栈上       |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                     |
| -------- | -------------------------------------------------------------------------|
| 需求映射 | 需求说明书 §6 - §8、§10、§19、§22、§27、§28                              |
| 范围内   | `TensorBase<S, D>`、类型别名、基础查询、构造校验、视图与 raw-parts 契约  |
| 范围外   | 广播、索引、reshape、归约与逐元素运算                                    |
| 非目标   | 引入运行时动态张量类型系统、隐藏存储模式差异或跳过元数据合法性校验       |

- 存储模式转换矩阵与具体转换 API 由 `05-storage.md` 承载实现设计。
- 本文档仅定义 `storage_kind()`、view/raw-parts 与张量查询接口，不重复展开转换细节。
- 不引入公开 `Layout` 结构体；`TensorBase` 直接内联 `offset` 与 `LayoutFlags` 等布局元数据。

---

## 3. 文件位置

```
src/tensor/
├── mod.rs             # TensorBase<S, D> struct definition and public exports
├── impls.rs           # core query method implementations
├── aliases.rs         # type alias definitions
└── construct.rs       # internal constructors (unsafe low-level construction,
                       #   Owned raw-parts decomposition & reconstruction)
```

文件划分理由：结构体定义、方法实现、类型别名、构造方法各自独立且职责清晰。

- 公开安全构造方法（`from_shape_vec`、`zeros`、`ones`、`eye` 等）的实现位于独立的上层模块 `src/construct/`（参见 `18-construction.md`）。
- 本目录下的 `construct.rs` 负责内部 unsafe 低级构造（`from_raw_parts`、`from_raw_vec_unchecked`）以及 Owned 张量的裸指针分解与重建（`into_raw_parts`、`from_raw_parts_owned`、`OwnedRawParts`）。这些方法需要直接访问 `TensorBase` 的私有字段，因此只能在本模块内定义；FFI 模块通过公开 API 或 re-export 暴露给外部消费者。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/tensor/
|
├── mod.rs
│   └── TensorBase<S, D> struct definition and public exports
|
├── impls.rs
│   ├── crate::storage    # Owned, ViewRepr, ViewMutRepr, ArcRepr, Storage, StorageMut, StorageOwned, StorageShared
│   ├── crate::dimension  # Dimension, Ix0~Ix6, IxDyn, .slice(), .checked_size(), .ndim()
│   ├── crate::layout     # LayoutFlags, compute_f_strides(), is_f_contiguous(), is_aligned()
│   └── crate::element    # Element
|
├── aliases.rs
│   └── (no external crate dependency; references TensorBase and dimension types from mod.rs)
|
└── construct.rs
    ├── crate::storage    # Owned<A>, from_raw_parts storage access
    ├── crate::dimension  # Dimension, checked_size()
    ├── crate::layout     # Strides<D>, LayoutFlags, compute_f_strides(), compute_layout_flags()
    └── crate::error      # XenonError::InvalidLayout
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `storage`   | `Owned`, `ViewRepr`, `ViewMutRepr`, `ArcRepr`, `Storage`, `StorageMut`, `StorageOwned`, `StorageShared`（参见 `05-storage.md §5`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `IntoDimension`, `.slice()`, `.checked_size()`, `.ndim()`（参见 `02-dimension.md §5`、`§5.6`）   |
| `layout`    | `LayoutFlags`, `Strides<D>`, `compute_f_strides()`, `compute_layout_flags()`, `is_f_contiguous()`, `is_aligned()`（参见 `06-layout.md §5`） |
| `element`   | `Element`（构造方法中 `A: Element` 约束；参见 `03-element.md §5`）                                                                  |
| `error`     | `XenonError`（`InvalidLayout` 含 `InvalidLayoutReason` / `StorageKindTag` 字段、`InvalidShape` 含 `InvalidShapeKind`；构造校验与 `validate_access_range`；参见 `26-error.md §5.1`） |

### 4.3 依赖合法性

| 项目           | 结论                       |
| -------------- | -------------------------- |
| 新增第三方依赖 | 无                         |
| 合法性结论     | 符合需求说明书最小依赖限制 |
| 替代方案       | 不适用                     |

### 4.4 依赖方向声明

依赖方向：单向向上。`tensor` 消费 `storage`、`dimension`、`layout` 的 trait 和类型，不被它们依赖。`math`、`iter` 等上层模块消费 `tensor`。

---

## 5. 公共 API 设计

### 5.1 TensorBase<S, D> 核心结构体

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
    /// Authoritative bit set: `F_CONTIGUOUS / ALIGNED / HAS_ZERO_STRIDE`
    /// (see `06-layout.md §5.1`); `compute_layout_flags(shape, strides, ptr)`
    /// is the unique authoritative entry. This field never carries
    /// derivation-source markers — those live in `derived_from_view_mut`
    /// below to keep `LayoutFlags` orthogonal to construction provenance.
    flags: LayoutFlags,

    /// Internal 1-bit marker: was this tensor produced by demoting a
    /// `ViewMutRepr` to a read-only `ViewRepr` (e.g., via `view_mut().view()`)?
    ///
    /// **Visibility:** private; no public getter / setter. Read internally by
    /// `access_semantics()` (see §5.3 rule (3)). Set internally `true` ONLY
    /// when this tensor is a `ViewRepr` produced by demoting a `ViewMutRepr`
    /// source — namely: (a) `ViewMutRepr::view()` returning a read-only
    /// `ViewRepr`, and (b) slicing routes whose source is `ViewMutRepr` or a
    /// `ViewRepr` already carrying `derived_from_view_mut == true` (see
    /// `17-indexing.md §6.3` propagation rule). Re-borrows that stay
    /// `ViewMutRepr` (e.g., `view_mut()` chained on a `ViewMutRepr`) do
    /// NOT set this marker — they remain writable. All other construction
    /// paths (Owned constructors, broadcast/transpose, slicing on sources
    /// that are neither `ViewMutRepr` nor a `ViewRepr` already carrying
    /// `derived_from_view_mut == true`, `from_raw_parts*`) leave it `false`.
    ///
    /// This is a **separate** field from `LayoutFlags`: it carries
    /// construction-provenance information, not layout geometry. Reusing a
    /// `LayoutFlags` reserved bit would conflate the two concerns and break
    /// `compute_layout_flags` as the unique authoritative entry for layout-derived
    /// state (`06-layout.md §5.1 / §5.3 / §5.12`). Implementations MAY pack this
    /// bool with other private internal bools (e.g., via a private
    /// `#[repr(transparent)]` byte) to avoid padding waste, provided the
    /// semantic boundary against `LayoutFlags` is preserved.
    derived_from_view_mut: bool,
}
```

- `TensorBase` 直接嵌入 `offset` / `flags` / `derived_from_view_mut` 字段。这是因为 `offset` 与存储指针配合进行偏移计算，`flags` 缓存布局信息，`derived_from_view_mut` 跟踪 ViewMut 降级来源，三者都属于张量实例的固有属性。
- `TensorBase` 不对外承诺稳定的结构体内存布局，也不作为 FFI 边界类型。FFI 消费者应优先使用 `23-ffi.md` 的 `TensorExport`，而非直接依赖 `TensorBase` 的字段顺序或 ABI 表示。
- `from_raw_parts*()` 系列中的 `ptr` 一律表示 storage base pointer，`offset` 一律表示从 storage base 到逻辑首元素的非负位移。`TensorBase::as_ptr()` / `TensorBase::as_mut_ptr()` 负责应用这一次偏移。`ffi` 文档中的示例与 Safety 说明必须遵循同一语义。

**线程安全推导**: `TensorBase<S, D>` 的 `Send`/`Sync` 由存储模式 `S` 和元素类型 `A` 共同决定：`S` 提供 `Send`/`Sync`（参见 `05-storage.md §6.8`），`A` 须满足对应的线程安全约束（参见 `25-safety.md §5`）。

| 张量存储模式              | `Send` 条件                       | `Sync` 条件                    | 说明                                              |
| ------------------------- | --------------------------------- | ------------------------------ | ------------------------------------------------- |
| `Tensor<Owned<A>, D>`     | 取决于 `Owned<A>: Send`           | 取决于 `Owned<A>: Sync`        | 拥有型规则与 `05-storage.md`、`25-safety.md` 一致 |
| `TensorView<'a, A, D>`    | 取决于 `ViewRepr<'a, A>: Send`    | 取决于 `ViewRepr<'a, A>: Sync` | 只读借用可跨线程共享的前提由 storage 层定义       |
| `TensorViewMut<'a, A, D>` | 取决于 `ViewMutRepr<'a, A>: Send` | 永不实现 `Sync` | 可变视图只允许独占传播；`!Sync` 由 `ViewMutRepr` 内部 raw `*mut A` 字段使 auto-trait 不实现 `Sync`、且 storage 层不提供 `unsafe impl Sync` 共同保证（参见 `05-storage.md §6.8`） |
| `ArcTensor<A, D>`         | 取决于 `ArcRepr<A>: Send`         | 取决于 `ArcRepr<A>: Sync`      | 共享只读线程安全前提完全继承 storage 层           |

### 5.2 Type aliases

```rust,ignore
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

    /// Query the access semantics of this tensor's data.
    pub fn access_semantics(&self) -> AccessSemantics;

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

    /// Returns the precise alias classification for this tensor.
    ///
    /// This is the **single recommended entry point** for L4/L5 modules
    /// (FFI export, parallel chunk safety, unsafe pointer arithmetic)
    /// to determine aliasing class. Callers SHOULD NOT manually combine
    /// `storage_kind()`, `has_zero_stride()`, and `derived_from_view_mut()`
    /// flags — those flag combinations are the implementation detail of
    /// this method and are forbidden by the safety contract defined in
    /// 25-safety.md §5.
    ///
    /// `AccessSemantics::SharedReadOnly` remains a 3-way summary for
    /// general access-permission queries; `AliasClass` provides the
    /// specific provenance needed for soundness reasoning.
    pub fn alias_class(&self) -> AliasClass {
        if self.storage_kind() == StorageKind::Shared {
            AliasClass::ArcShared
        } else if self.flags().has_zero_stride() {
            // Per 06-layout.md §5.11: HAS_ZERO_STRIDE is only set
            // when product(shape) > 0, so the empty-tensor edge case
            // is already excluded.
            AliasClass::BroadcastAlias
        } else if self.derived_from_view_mut() {
            AliasClass::ViewMutDerived
        } else {
            AliasClass::Unique
        }
    }

}

impl<S, D> TensorBase<S, D>
where
    D: Dimension + Clone,
{
    /// Returns a clone of the dimension type.
    pub fn raw_dim(&self) -> D;
}

// Semantic query enums — authoritative definition resides in this module.
// 01-architecture.md §11 provides a quick-reference summary of these types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    Owned,
    View,
    ViewMut,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessSemantics {
    ReadOnly,
    SharedReadOnly,
    Writable,
    Owned,
}

/// 别名分类：`TensorBase::alias_class()` 返回的精确别名类别枚举。
///
/// 与 `AccessSemantics` 不同，此枚举提供精确的别名来源区分，
/// 用于 L4/L5 模块的安全推理（如 unsafe 指针算术、并行分块安全、
/// FFI 导出决策）。
///
/// `AccessSemantics::SharedReadOnly` 将三类语义不同的张量合并
/// 为单一变体；`AliasClass` 将其拆分为独立变体，方便调用方在
/// 需要区分别名来源时进行模式匹配，而不必手动组合
/// `storage_kind()`、`has_zero_stride()`、`derived_from_view_mut()`
/// 三个标志。
///
/// 详见 25-safety.md §5 的安全契约。
pub enum AliasClass {
    /// 张量独占其底层数据，无别名: 来源为 Owned 或独占 ViewMut。
    Unique,
    /// Arc 共享所有权: 多个 ArcTensor 实例共享底层 SharedBuf。
    ArcShared,
    /// 广播零步长别名: 同一物理元素被多个逻辑索引访问
    /// (any(stride == 0) && product(shape) > 0)。
    BroadcastAlias,
    /// ViewMut 降级而来的只读视图: derived_from_view_mut == true
    /// 且非广播、非 Arc。
    ViewMutDerived,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocation {
    Cpu,
}
```

- **`len` / storage 长度不变量：** `TensorBase::len()` 返回逻辑元素总数（由 `shape` 计算）；`Storage::len()` 返回底层存储的可见长度。对于视图类型，storage len 可能大于 logical len。所有 bounds check 基于 logical len，raw-parts 构造基于 storage len。
- **数据位置查询说明：** 当前版本仅支持 CPU 内存，`data_location()` 恒返回 `DataLocation::Cpu`，用于满足 `需求说明书 §8` 的存储位置查询接口。
- **`storage_kind()` 语义说明：** `storage_kind()` 返回底层**实际存储表示类型**对应的 `Owned / View / ViewMut / Shared`，而不是高层语义分类。`Owned` 报告 `Owned`，`ViewRepr` 报告 `View`，`ViewMutRepr` 报告 `ViewMut`，`ArcRepr` 报告 `Shared`。因此广播结果若底层表示为 `ViewRepr`，其 `storage_kind()` 也必须返回 `View`，而不是 `Shared`。
- **广播语义补充：** 广播结果的只读共享语义通过 layout flags 和访问控制表达，而非通过 `storage_kind()` 伪装。详见 `15-broadcast.md`。
- **`access_semantics()` 广播判定机制：** 当 `ViewRepr` 的 `LayoutFlags` 包含 `HAS_ZERO_STRIDE` 时，`access_semantics()` 返回 `AccessSemantics::SharedReadOnly`，以区分普通只读视图（`ReadOnly`）与广播只读视图（`SharedReadOnly`）。此判定与 `LayoutState::BroadcastView` 的分类条件一致（见 `06-layout.md §5.11`）。
- **`ViewMutRepr → ViewRepr` 零拷贝降级的来源标记（v3.0.1）：** 当 `view_mut().view()` 把可写视图降级为只读视图时，结果的 `LayoutFlags` 不一定包含 `HAS_ZERO_STRIDE`（普通 contiguous mutable view 降级后仍是 contiguous）。为了让 `access_semantics()` 在不依赖来源上下文的前提下仍能区分"普通 view 借用"与"由 ViewMut 降级而来的共享只读视图"，`TensorBase` 携带一个**独立的、私有的** 1-bit 内部标记字段 `derived_from_view_mut: bool`（结构体字段层面，**不**复用 `LayoutFlags` 任何 bit；`06-layout.md §5.1` 的 `LayoutFlags` 权威定义仅包含 `F_CONTIGUOUS / ALIGNED / HAS_ZERO_STRIDE`，且 `compute_layout_flags(shape, strides, ptr)` 是其唯一权威计算入口，与张量来源信息正交）。该字段对外不暴露 setter，仅由 `ViewMutRepr::view()` 降级为只读 `ViewRepr` 的路径，以及 `17-indexing.md §6.3` 切片传播规则（源为 `ViewMutRepr` 或源已带 `derived_from_view_mut == true`）设置；`view_mut()` 的可写 reborrow 仍返回 `ViewMutRepr`，**不**设置该标记。可以与 `TensorBase` 中其它内部 bool 通过私有 `#[repr(transparent)]` 包装位组打包以避免 padding 浪费，但其语义边界与 `LayoutFlags` 严格分离。`access_semantics()` 的判定规则因此扩展为：(1) `storage_kind() == StorageKind::Shared` → `SharedReadOnly`；(2) `storage_kind() == StorageKind::ViewMut` → `Writable`；(3) `storage_kind() == StorageKind::View` 且 `(layout_flags().has_zero_stride() || self.derived_from_view_mut)` → `SharedReadOnly`；(4) `storage_kind() == StorageKind::View` 且二者都未设置 → `ReadOnly`；(5) `storage_kind() == StorageKind::Owned` → `Owned`。这避免了 `access_semantics()` 输出与构造来源的歧义，且实现成本只有一个独立标志位，不污染 `LayoutFlags` 权威计算。
- **`SharedReadOnly` 三重含义说明：** `AccessSemantics::SharedReadOnly` 覆盖三类语义不同的张量，由不同的 (`storage_kind()`, `layout_flags().has_zero_stride()`, `derived_from_view_mut`) 组合识别：
  1. **所有权共享（`ArcRepr`）：** `storage_kind() == Shared`。多个张量句柄通过 `Arc` 共享底层存储；写访问需要先唯一化（参见 `05-storage.md §5.8`、`§11` 决策 2）。这里"共享"指存储所有权层面的共享。
  2. **同物理地址共享（广播 `ViewRepr`）：** `storage_kind() == View && layout_flags().has_zero_stride()`。多个不同逻辑索引映射到同一物理地址（典型来源：`broadcast_to` / `broadcast_with`）。这里"共享"指同一物理元素被多个逻辑索引共享读取。
  3. **来源共享（ViewMut 降级而来的 `ViewRepr`）：** `storage_kind() == View && self.derived_from_view_mut`（**与零步长无关**：普通 contiguous mutable view 降级后逻辑索引 1:1 物理索引，没有地址重叠，但仍标记为"共享只读"以反映"原始独占借用已转交，再获取写访问需要重新论证")。
  三类共享在写访问安全性上的结论相同（都不能直接可写），因此合并到同一变体；但调用方若需要区分（例如内部 CoW 唯一化路径只对 `ArcRepr` 适用，BLAS / SIMD 路径需要拒绝零步长但可接受 ViewMut 降级），必须先通过 `storage_kind()` 判断底层存储类型，再由 `flags().has_zero_stride()` 区分广播视图，最后由内部 `derived_from_view_mut` 区分降级来源。`access_semantics()` 本身不暴露这一三向区分。
- **权威约束：** 访问语义的权威查询入口是 `access_semantics()`；`storage_kind()` 只报告底层表示类型，不能替代访问语义判定。
- **`HAS_ZERO_STRIDE` 权威约束：** `HAS_ZERO_STRIDE` 标志位的定义以 **06-layout.md §5.11** 为唯一权威（`any(stride == 0) && product(shape) > 0`）；本节仅通过 `flags().has_zero_stride()` 和 `LayoutState` 查询使用该标志，不重复定义其规则。
- `LayoutState` 使用 `crate::layout::LayoutState`（参见 `06-layout.md §5`）；
- 本文档不再重复定义 `FContiguous`、`NonContiguous`、`BroadcastView` 三个变体。

- **`alias_class()` 规范入口：** `TensorBase::alias_class() -> AliasClass` 是 L4/L5 模块（FFI 导出、并行分块安全、unsafe 指针算术）判断别名类别的 **单一推荐入口**。L4/L5 模块 **不应** 手动组合 `storage_kind()`、`has_zero_stride()`、`derived_from_view_mut()` 三个标志——这些标志组合是本方法的实现细节，由 25-safety.md §5 的安全契约禁止在外部直接组合。`AccessSemantics::SharedReadOnly` 保留为三合一语义摘要，用于通用访问权限查询；`AliasClass` 提供具体的别名来源（Arc 共享 / 广播零步长 / ViewMut 降级 / 独占），为安全性论证提供精确依据。安全契约详见 25-safety.md §5。

**三层语义模型：**

1. **存储表示层**：`Owned` / `View` / `ViewMut` / `Shared`，即 `storage_kind()` 的返回值，描述底层 representation。
2. **访问语义层**：`ReadOnly` / `SharedReadOnly` / `Writable` / `Owned`，由 `access_semantics()` 返回，描述当前张量对底层数据的访问语义。
3. **布局状态层**：`FContiguous` / `NonContiguous` / `BroadcastView`，由 `LayoutFlags` / `LayoutState` 描述。

广播张量通常表现为“表示层 = `View`，访问语义层 = `SharedReadOnly`，布局状态层 = `BroadcastView`”。

| 语义分类 | 对应表示类型                                                                                      | 统一语义说明                                                           | 查询结果                          |
| -------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------- |
| 只读     | `ViewRepr<'_, A>`（普通非广播只读借用）                                                           | 只提供共享只读借用；不提供写访问；不持有底层存储                       | `AccessSemantics::ReadOnly`       |
| 共享只读 | `ArcRepr<A>`，或带共享只读语义标记的 `ViewRepr<'_, A>`（广播结果 / `ViewMutRepr` 零拷贝降级结果） | 可被多个只读视图共享；不提供安全可写访问；可共享所有权或共享借用语义   | `AccessSemantics::SharedReadOnly` |
| 可写     | `ViewMutRepr<'_, A>`                                                                              | 提供独占可写借用；同时允许读取；不得与其他可写或共享只读访问并存       | `AccessSemantics::Writable`       |
| 拥有     | `Owned<A>`                                                                                        | 持有底层存储所有权；提供可读可写访问；可零拷贝借出视图或降级为共享只读 | `AccessSemantics::Owned`          |

### 5.4 指针访问方法

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a raw pointer to the logical first element.
    ///
    /// For empty tensors, this returns `NonNull::dangling().as_ptr()` (which
    /// yields `*mut A`, implicitly coerced to `*const A`) and does not perform
    /// pointer arithmetic on the storage base pointer.
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

    /// Returns the length of the underlying storage buffer in elements.
    ///
    /// This differs from `len()` which returns the logical element count
    /// (product of shape). For views, `storage_len()` may be larger than
    /// `len()` because the backing storage can extend beyond the visible
    /// portion. For owned tensors, `storage_len()` equals `len()`.
    pub fn storage_len(&self) -> usize;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Returns a mutable raw pointer to the data start.
    ///
    /// For empty tensors, this returns `NonNull::dangling().as_ptr()` and does
    /// not perform pointer arithmetic on the storage base pointer. Note:
    /// `NonNull::dangling().as_ptr()` returns `*mut A`, matching this method's
    /// return type exactly.
    pub fn as_mut_ptr(&mut self) -> *mut A;

    /// Returns the raw mutable storage base pointer WITHOUT adding the offset.
    ///
    /// Unlike `as_mut_ptr()` which returns `storage.as_mut_ptr().add(offset)`,
    /// this method returns `storage.as_mut_ptr()` directly — the raw mutable
    /// base pointer of the storage buffer. The caller is responsible for
    /// manually accounting for `self.offset()` when computing element
    /// addresses.
    ///
    /// The returned pointer does NOT point to the first logical element;
    /// use `as_mut_ptr()` for that. Any pointer arithmetic based on this
    /// pointer must include `self.offset()` to access the correct data.
    pub fn as_storage_mut_ptr(&mut self) -> *mut A;
}
```

### 5.5 连续切片访问方法

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a shared slice when all of the following preconditions hold:
    ///
    /// 1. `flags.is_f_contiguous()` is true (F-order contiguous layout)
    /// 2. `!flags.has_zero_stride()` (no broadcast dimensions)
    /// 3. `as_ptr()` points at the logical first element (logical-first
    ///    pointer contract; see §6.2)
    ///
    /// This is the zero-copy fast path consumed by `simd/`, `parallel/`, and
    /// convenience APIs such as `set::unique()` examples. Non-contiguous views
    /// and broadcast views return `None` and must fall
    /// back to iterator-based access. A non-zero logical offset alone does not
    /// disqualify the fast path: if `as_ptr()` already points at the logical
    /// first element and the layout is contiguous, `as_slice()` may still be
    /// returned zero-copy.
    /// Empty tensors return `Some(&[])`.
    ///
    /// **ZST contract:** the closed element set in `03-element.md §5.6`
    /// contains no zero-sized types, so `size_of::<A>() > 0` always holds and
    /// `slice::from_raw_parts` does not require a special ZST branch. If the
    /// closed element set is ever extended to include ZSTs, this contract
    /// must be revisited together with `05-storage.md §5.5`.
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

### 5.6 安全构造方法

安全构造路径必须验证全部可验证元数据约束，至少包括 shape/stride 可表示性、元素总数计算不溢出、以及逻辑访问范围不越界。`from_shape_vec` 这类 API 不得把这些前提留给调用方；safe 构造负责兜底全部可检查元数据条件。

> **权威分工说明（避免双权威）：**
>
> - `TensorBase<Owned<A>, D>::from_shape_vec` 等公开安全构造方法的**完整设计**（包含算法、错误字段、对齐策略、边界场景）位于 `18-construction.md §5.3`。
> - 本节仅给出**公开签名与公开契约摘要**，不重复展开实现算法。任何与 `18-construction.md` 不一致的描述均以 `18-construction.md` 为准。
> - 本节列出 `from_shape_vec` 的目的，是说明它是 `TensorBase<Owned<A>, D>` 的固有方法签名，并固定其错误返回类型与文档化前置条件，方便 `tensor` 模块下游消费者在不阅读 `18-construction.md` 时也能理解类型签名。

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
    ///   defined by `需求说明书 §19`; the input order defines which element belongs
    ///   to each logical index, rather than requiring callers to pre-arrange bytes
    ///   in a specific physical layout
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError)` per `26-error.md §5.1`:
    ///
    /// - `XenonError::InvalidShape { kind: InvalidShapeKind::ProductOverflow, .. }`
    ///   when `shape.checked_size()` overflows
    /// - `XenonError::InvalidShape { kind: InvalidShapeKind::ElementCountMismatch, .. }`
    ///   when `data.len() != shape.checked_size()`
    /// - `XenonError::InvalidLayout { reason: InvalidLayoutReason::*, .. }`
    ///   when canonical F-order stride derivation fails
    /// - `XenonError::InvalidLayout` or storage-layer error when the underlying
    ///   storage cannot be constructed
    ///
    /// All error variants are structured per `26-error.md §5.1` (no free-text
    /// `reason` strings). The full error matrix lives with the implementation
    /// in `18-construction.md §5.3`.
    ///
    /// The current version defaults to 64-byte-aligned allocation
    /// (for example via `Owned::from_vec_aligned`), consistent with `05-storage.md`.
    /// This aligned path is the default owned-storage policy; any exception must be
    /// explicitly documented by the corresponding constructor and still preserve
    /// the same logical element order. See `05-storage.md §5` and `18-construction.md §5.3`.
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
    pub fn from_shape_vec<Sh>(shape: Sh, data: Vec<A>) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>;

    /// Constructor-module APIs such as `zeros()` and `ones()` also return
    /// `Result<Self, XenonError>` and must be used with `?` in examples and
    /// calling code.
    // NOTE: the default owned-storage path uses aligned allocation.
    // The concrete alignment policy belongs to the storage module.

    /// Construct a tensor from a `Vec` while skipping logical / physical
    /// consistency checks. This is a `pub(crate)` internal fast path used by
    /// constructor-module helpers (such as `Tensor::zeros` once shape has been
    /// pre-validated) and by parallel-write paths that already proved their
    /// outputs satisfy the canonical layout invariants.
    ///
    /// The `_unchecked` suffix means: the caller has already proved the listed
    /// safety contract holds, and this constructor performs no `Result`-returning
    /// validation. In particular, `shape.checked_size()` is **not** revalidated;
    /// the caller must guarantee it succeeded prior to invocation. Any violation
    /// is undefined behavior, not a recoverable error.
    ///
    /// # Safety
    /// - `data.as_ptr()` must remain valid for the duration of construction, and
    ///   `Vec<A>` must satisfy the alignment requirements of `A`
    /// - `shape.checked_size()` must already have been validated (no overflow)
    ///   before calling this method
    /// - `data.len()` must equal the previously validated element count
    /// - `shape` must be representable by the current dimension type
    /// - The default packed F-order stride derived from `shape` must be
    ///   representable and consistent with `需求说明书 §7`
    /// - The constructor assumes no extra offset and therefore treats the input
    ///   buffer as the full logical tensor payload
    pub(crate) unsafe fn from_raw_vec_unchecked(data: Vec<A>, shape: D) -> Self {
        // No revalidation; `shape.checked_size()` and `data.len()` are caller-proved.
        // Computes F-order strides internally and constructs flags directly.
        // ...
    }

    // `new_unchecked` intentionally NOT defined here on the
    // `Owned`-specialized impl block. A separate
    // `impl<A, D> TensorBase<Owned<A>, D> { fn new_unchecked(...) }` would
    // collide with the generic `impl<S: RawStorage, D> { fn new_unchecked }`
    // below (Rust E0592: duplicate definitions for
    // `TensorBase<Owned<A>, D>` because `Owned<A>: RawStorage`). All Owned
    // construction paths route through the generic `new_unchecked` below;
    // see that method's doc comment for the full safety contract.
}

impl<S, D> TensorBase<S, D>
where
    S: crate::storage::RawStorage,
    D: Dimension,
{
    /// Canonical unchecked constructor for tensor metadata assembly.
    ///
    /// **Single canonical `pub(crate)` unsafe entry point.** All construction
    /// paths (Owned via `zeros` / `ones` / `from_shape_vec` / `from_scalar`,
    /// View / ViewMut via slicing / transpose / broadcast / `view()` /
    /// `view_mut()`, Arc via Arc-storage construction) route through this
    /// single generic form. No Owned-specialized parallel form exists — see
    /// the comment block above for the Rust coherence reason (E0592 collision
    /// when both an Owned-specialized impl and a generic
    /// `impl<S: RawStorage, D>` impl define a method with the same name).
    ///
    /// All other internal unchecked constructors (e.g.
    /// `Tensor::from_shape_vec_aligned_unchecked` in `21-type.md §5.6`) MUST
    /// forward to this method rather than defining their own safety
    /// invariants for tensor layout fields. They build storage / strides /
    /// flags internally, then delegate to `new_unchecked` with `offset = 0`
    /// and `derived_from_view_mut = false`.
    ///
    /// # Safety
    /// - `shape`, `strides`, `offset`, and `flags` must be mutually
    ///   consistent: `flags` was produced by
    ///   `layout::compute_layout_flags::<A, D>(&shape, &strides, storage_ptr)`
    ///   for the same `shape` and `strides` actually stored, where
    ///   `storage_ptr` is the same logical-first pointer the caller will
    ///   later expose via `as_ptr()`.
    /// - The logical access range derived from `shape` / `strides` / `offset`
    ///   must lie entirely within `storage` (no overflow, no out-of-bounds).
    /// - `shape.checked_size()` must already have been validated (no
    ///   overflow) before calling this method.
    /// - When `offset == 0` and `shape` is canonical-F-contiguous w.r.t.
    ///   `strides`, the value of `flags` must reflect that (the caller
    ///   should use `compute_layout_flags` rather than fabricating bits
    ///   manually).
    /// - `derived_from_view_mut` semantics: must be `true` ONLY when this
    ///   tensor is a `ViewRepr` produced by demoting a `ViewMutRepr` source
    ///   (see §5.3 rule (3) for the `access_semantics()` consequence) or
    ///   when slicing a source that itself has `derived_from_view_mut == true`
    ///   (the tag propagates through nested slicing). For Owned construction
    ///   paths (`S = Owned<A>`), the caller MUST pass `false` (Owned tensors
    ///   never carry downgrade provenance).
    /// - `view_mut()` reborrow chained on a `ViewMutRepr` stays `ViewMutRepr`
    ///   (writable) and does NOT use this constructor with `true`.
    pub(crate) unsafe fn new_unchecked(
        storage: S,
        shape: D,
        strides: Strides<D>,
        offset: usize,
        flags: LayoutFlags,
        derived_from_view_mut: bool,
    ) -> Self {
        // No revalidation; all six constructor inputs are caller-proved
        // (storage, shape, strides, offset, flags, derived_from_view_mut).
        // `unsafe fn` is required because the # Safety contract documents
        // UB on metadata mismatch.
        TensorBase { storage, shape, strides, offset, flags, derived_from_view_mut }
    }
}
````

### 5.7 unsafe 构造方法

`from_raw_parts*()` 这类接口只验证能够基于输入元数据直接检查的条件；safe 构造会兜底验证全部可检查元数据，而 unsafe 构造仅拒绝明显非法的 shape/stride/offset/storage_len 组合。若这些元数据校验失败，构造器返回 `Err(XenonError::InvalidLayout)`（附带上下文）。调用方仍负责保证指针有效性、对齐、可访问范围和生命周期等库无法自行证明的内存前提。文档中的 `# Safety` 说明必须与这一分工保持一致。

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
    /// The caller must guarantee the following invariants. Violating any of
    /// them is undefined behavior; the library cannot detect them from the
    /// passed metadata alone.
    ///
    /// **Pointer / memory invariants (caller-only):**
    /// - `ptr` points to the storage base of the view (logical-first pointer is
    ///   computed by the constructor as `ptr.add(offset)` only when `offset != 0`
    ///   and the tensor is non-empty)
    /// - The byte range `[ptr, ptr + storage_len * size_of::<A>())` is part of
    ///   a single allocated object (same provenance) and remains valid for the
    ///   entire lifetime `'a` of the returned view
    /// - `ptr` is aligned to `align_of::<A>()` (even for ZST elements, in which
    ///   case a `NonNull::<A>::dangling().as_ptr()` sentinel satisfies this rule)
    /// - For non-empty tensors, every logical element address derived from
    ///   `shape`, `strides`, and `offset` points to an initialized `A` value
    /// - For empty tensors (`product(shape) == 0`), `ptr` may be a well-formed
    ///   dangling sentinel and is never dereferenced
    /// - No `&mut` reference to overlapping memory is alive during `'a`
    ///
    /// **Constructor-validated metadata (no caller obligation):** the
    /// constructor returns `Err(XenonError::InvalidLayout)` (see
    /// `26-error.md §5.1`, `InvalidLayoutReason`) when any of the following
    /// directly checkable conditions fail; otherwise returns `Ok(_)`.
    /// - `shape` and `strides` agree on rank for this dimension type
    /// - `shape.checked_size()` succeeds (no element-count overflow)
    /// - Every stride is representable as a non-negative `usize <= isize::MAX`
    ///   (so that pointer-offset arithmetic via `<*const A>::add` cannot
    ///   produce an `isize` overflow)
    /// - The layout family is valid for an immutable view: F-order packed,
    ///   non-contiguous F-order-derived (slice / transpose), or broadcast-style
    ///   zero-stride layouts (zero strides only allowed on read-only paths)
    /// - The logical access range computed by `validate_access_range`
    ///   (see §6.2) fits within `storage_len`
    /// - For empty tensors, the only metadata requirement is `offset <= storage_len`
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
    /// Inherits all caller obligations from `from_raw_parts` (pointer
    /// provenance, alignment, lifetime, initialization, no overlapping `&` /
    /// `&mut` aliases) **with the following additional rules**:
    ///
    /// - The caller must hold exclusive write access to the entire backing
    ///   storage range covered by `[ptr, ptr + storage_len * size_of::<A>())`
    ///   for the lifetime `'a`
    /// - No other reference (shared or mutable) to overlapping memory may be
    ///   alive during `'a`
    /// - The caller asserts that the layout itself is non-overlapping, i.e. no
    ///   two distinct logical indices map to the same physical address
    ///
    /// **Constructor-validated metadata** (returns `Err(XenonError::InvalidLayout)`
    /// on failure):
    /// - All checks performed by `from_raw_parts`
    /// - The layout family is valid for a mutable view: only F-order packed or
    ///   F-order-derived non-contiguous layouts. Broadcast-style / zero-stride
    ///   layouts are **rejected** (a zero stride on any non-singleton axis means
    ///   multiple logical indices alias the same address)
    /// - `validate_non_overlapping_layout` (see below) accepts the layout. The
    ///   library only accepts the efficiently verifiable non-overlapping subset;
    ///   exotic but theoretically valid strided layouts may be conservatively
    ///   rejected.
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        storage_len: usize,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Result<Self, XenonError>;
}
```

**可写布局非重叠校验：** `from_raw_parts_mut()` 还必须拒绝会让两个不同逻辑索引映射到同一地址的可写布局。"非重叠"定义为：任意两个不同逻辑索引 `i != j`，其可写目标地址 `addr(i)` 与 `addr(j)` 必须不同。该校验不得通过枚举全部可达 offset 来实现；当前版本只承诺接受可高效保守判定的正步长布局。

**核心不变量：** 对一个非单元素轴 `i`，该轴单独可达的 offset 集合是
`{ k_i * stride[i] | 0 <= k_i < shape[i] }`，其大小为 `(shape[i] - 1) * stride[i] + 1`
（"+1" 来自 `k_i = 0` 这一项）。因此在按 stride 升序逐轴并入时，"下一轴 stride" 必须严格大于 "已覆盖子空间最大可达 offset"，下一轴的最小非零步进 `1 * stride[next]` 才不会与已覆盖区域产生别名。

算法如下（**保守 dense-prefix 充分判定**，并非完备判定）：

```text
// Algorithm name: dense-prefix sufficient non-overlap test.
//
// Soundness: PASSING inputs are guaranteed non-overlapping (this test
// proves the property).
// Completeness: this test is CONSERVATIVE — some layouts that are
// non-overlapping but do not form a dense prefix pattern will be
// rejected. We accept that trade-off because the dense-prefix family
// covers all layouts Xenon's safe constructors and internal slicing
// API can produce (canonical F-order, transpose, slice with positive
// strides). External `from_raw_parts_mut` callers passing exotic
// non-overlapping strides will be rejected and should route through
// Xenon's internal slicing API (which carries provenance) instead.
validate_non_overlapping_layout(shape, strides, offset, storage_len):
    1. If product(shape) <= 1:
           return Ok(()).
    2. Reject immediately if any non-singleton axis has stride == 0.
    3. Collect all non-singleton axes, sort them by stride ascending.
    4. Initialize covered_max_offset = 0.
       (covered_max_offset is the maximum offset reachable by varying only
        the axes already accepted; the corresponding offset set has size
        covered_max_offset + 1. For dense-prefix layouts the offset set
        equals the contiguous integer range [0, covered_max_offset],
        which is what makes step 5's strict-greater test sufficient.
        For non-dense-prefix layouts the offset set is a SUBSET of
        [0, covered_max_offset] — step 5 then becomes conservative.)
    5. For each sorted axis i:
           // The next axis's smallest non-zero step (stride[i] * 1) must
           // exceed every offset already reachable, otherwise it aliases
           // an already-covered offset.
           require stride[i] > covered_max_offset;
           // checked_mul to detect span overflow before adding it in.
           span_i = (shape[i] - 1).checked_mul(stride[i])?;
           covered_max_offset = covered_max_offset.checked_add(span_i)?;
       If any checked arithmetic fails or the inequality does not hold, reject.
    6. If the conservative test cannot prove non-overlap, return
       Err(XenonError::InvalidLayout {
           operation: "validate_non_overlapping_layout".into(),
           storage_kind: StorageKindTag::ViewMut,
           shape: shape.slice().to_vec(),
           strides: strides.as_slice().to_vec(),
           offset,
           storage_len,
           reason: InvalidLayoutReason::AmbiguousOverlap,
       }).
    7. Otherwise return Ok(()).
```

> **示例（核对算法正确性）：** `shape = [2, 2]`, `strides = [1, 1]`。
> 排序后第一轴 stride=1，进入步骤 5：要求 `1 > 0`（成立），然后
> `covered_max_offset = 0 + (2-1)*1 = 1`。第二轴 stride=1，要求 `1 > 1`（不成立），
> 拒绝。该结果正确，因为两个轴单独可达 offset 集合 `{0, 1}` 与 `{0, 1}` 完全重叠。

该保守算法允许拒绝一部分理论上合法但无法高效证明不重叠的 exotic stride 布局；当前版本不为这类布局提供可写 raw-parts 构造承诺。FFI 文档（`23-ffi.md §6.3`）引用此算法。

#### Owned 裸指针分解与重建

上述 `from_raw_parts*()` 构造视图；以下方法专用于 Owned 张量的分解与重建，构成完整的 round-trip 对。核心实现定义于本模块（`src/tensor/construct.rs`），FFI 模块仅做薄包装——参见 `23-ffi.md §5.8`。

```rust,ignore
/// Decomposition of an owned tensor into raw pointer + allocator metadata.
///
/// **Note on ABI:** `OwnedRawParts<A, D>` is **not** a stable C-ABI type. The
/// `D` and `Strides<D>` fields are Rust generics whose layout is not specified
/// by `#[repr(C)]` (especially for `IxDyn`, which contains a `Vec<usize>`).
/// FFI consumers MUST NOT decode this struct from C code. It exists solely as
/// a Rust-internal round-trip carrier for `into_raw_parts` /
/// `from_raw_parts_owned`. C-facing interop must use the dedicated
/// `TensorExport` / `TensorExportMut` types defined in `23-ffi.md §5.4`,
/// which are explicitly designed for stable C ABI.
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
    D: Dimension + Clone,
{
    /// Consumes the tensor, returning owned raw parts.
    ///
    /// # Returns
    ///
    /// An `OwnedRawParts<A, D>` snapshot containing the pointer plus the
    /// allocator metadata required to reconstruct Xenon's aligned owned
    /// storage.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tensor = Tensor2::<f64>::zeros([3, 4])?;
    /// let raw = tensor.into_raw_parts();
    /// // Reconstruct with Tensor::from_raw_parts_owned(raw) and let Drop free it.
    /// ```
    pub fn into_raw_parts(self) -> OwnedRawParts<A, D> {
        let this = core::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is a valid owned tensor; `as_mut_ptr` returns a
        // pointer to the storage base, ownership of which is transferred to
        // the caller as part of the returned raw parts.
        let ptr = unsafe { (&*this).storage_base_mut_ptr_unchecked() };
        OwnedRawParts {
            ptr,
            len: this.storage.len(),
            cap: this.storage.capacity(),
            align: this.storage.alignment(),
            // D and Strides<D> require Clone; this is enforced by the impl bound.
            shape: this.shape.clone(),
            strides: this.strides.clone(),
            offset: this.offset,
        }
    }

    /// Reconstructs an owned tensor from raw parts obtained via
    /// `into_raw_parts`. Takes ownership of memory allocated by Xenon's
    /// aligned allocator.
    ///
    /// # Safety
    ///
    /// - `raw.ptr` must point to memory allocated by Xenon's aligned allocator
    /// - `raw.len`, `raw.cap`, and `raw.align` must be the original allocator
    ///   metadata
    /// - `raw.shape` and `raw.strides` must describe a valid, non-overlapping
    ///   canonical F-order layout
    /// - `raw.offset` must be 0 for owned raw parts
    /// - The caller transfers ownership; do NOT free `raw.ptr` separately
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidLayout { reason: InvalidLayoutReason::* })`
    /// (see `26-error.md §5.1`) when directly checkable metadata validation
    /// fails. Reason variants used here are the canonical `InvalidLayoutReason`
    /// values for owned raw-parts reconstruction; the unsafe obligation remains
    /// the memory/pointer guarantees that cannot be checked from metadata alone.
    pub unsafe fn from_raw_parts_owned(
        raw: OwnedRawParts<A, D>,
    ) -> Result<Self, XenonError> {
        // 1) offset must be zero for owned raw parts.
        if raw.offset != 0 {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::OwnedRequiresZeroOffset,
            });
        }

        // 2) shape product must be representable AND must equal raw.len.
        let expected_len = raw.shape.checked_size().map_err(|_| XenonError::InvalidLayout {
            operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
            storage_kind: StorageKindTag::Owned,
            shape: raw.shape.slice().to_vec(),
            strides: raw.strides.as_slice().to_vec(),
            offset: raw.offset,
            storage_len: raw.len,
            reason: InvalidLayoutReason::ShapeProductOverflow,
        })?;
        if raw.len != expected_len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::LenShapeMismatch,
            });
        }

        // 3) capacity must cover len.
        if raw.cap < raw.len {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::CapacityBelowLen,
            });
        }

        // 4) align must be a valid power of two and at least align_of::<A>().
        if !raw.align.is_power_of_two() || raw.align < core::mem::align_of::<A>() {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::AlignmentInvalid,
            });
        }

        // 5) strides must equal canonical F-order strides.
        let expected_strides = layout::compute_f_strides(&raw.shape)?;
        if raw.strides != expected_strides {
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed("tensor::from_raw_parts_owned"),
                storage_kind: StorageKindTag::Owned,
                shape: raw.shape.slice().to_vec(),
                strides: raw.strides.as_slice().to_vec(),
                offset: raw.offset,
                storage_len: raw.len,
                reason: InvalidLayoutReason::OwnedRequiresCanonicalFOrder,
            });
        }

        // SAFETY: The caller's # Safety contract guarantees raw.ptr is valid
        // memory allocated by Xenon's aligned allocator with the recorded
        // (len, cap, align) metadata. Ownership transfer is part of the
        // contract; raw.ptr must not be freed externally.
        let storage = unsafe { Owned::from_raw_parts(raw.ptr, raw.len, raw.cap, raw.align) };

        let logical_ptr: *const A = if raw.len == 0 {
            // Empty tensors must not pass a potentially dangling storage pointer
            // to compute_layout_flags; use a well-defined non-dereferenceable sentinel.
            core::ptr::NonNull::<A>::dangling().as_ptr()
        } else {
            // offset == 0 already verified, so raw.ptr IS the logical first element.
            raw.ptr
        };
        let flags = layout::compute_layout_flags::<A, D>(&raw.shape, &raw.strides, logical_ptr);
        // Routed through the single canonical `pub(crate) unsafe fn
        // new_unchecked` (§5.6, generic `impl<S: RawStorage, D>` form — the
        // Owned-specialized parallel form was removed because it would
        // collide with the generic form for `S = Owned<A>`; see §5.6 comment
        // block) to keep private-field access localized to ONE entry point
        // and to satisfy the locked invariant "TensorBase has 6 fields
        // including `derived_from_view_mut`" (R10 B-01).
        // SAFETY: All six constructor-input invariants of `new_unchecked`
        // are satisfied: (1) shape was overflow-checked above; (2) strides
        // were verified canonical F-order above; (3) offset == 0 was
        // verified above; (4) flags were just produced by
        // `compute_layout_flags` for the same shape/strides/logical_ptr;
        // (5) the logical access range `[0, raw.len)` lies within `storage`
        // because `raw.len == shape.checked_size()` and `raw.cap >= raw.len`
        // were both verified; (6) `derived_from_view_mut: false` —
        // `from_raw_parts_owned` is an Owned reconstruction path, not a
        // ViewMut downgrade (the generic `new_unchecked` Safety contract
        // mandates this argument be `false` for any `S = Owned<A>` call).
        Ok(unsafe {
            TensorBase::new_unchecked(storage, raw.shape, raw.strides, raw.offset, flags, false)
        })
    }
}
```

> **`InvalidLayoutReason` 字段引用（v2.0.x）：** 上述错误构造使用 `26-error.md §5.1` 定义的**封闭超集枚举**字段。Owned raw-parts 专属错误使用 `OwnedRequiresZeroOffset` / `LenShapeMismatch` / `CapacityBelowLen` / `AlignmentInvalid` / `OwnedRequiresCanonicalFOrder`；shape 元素数溢出统一使用 `ShapeProductOverflow`（不再使用旧名 `ElementCountOverflow`）；无法证明可写布局不重叠统一使用 `AmbiguousOverlap`（不再使用旧名 `OverlapNotProvable`）。**禁止**在本节新增未列入 `26-error.md §5.1` 的局部变体——`26-error.md §5.1` 是 `InvalidLayoutReason` 的唯一权威源；新增 case 必须先扩展该枚举。`Cow::Borrowed("...")` 与 `StorageKindTag::Owned` 同样按 `26-error.md §5.1` 字段表填写。
>
> **关于 `# Safety` 与 `unsafe block`：** Rust 2024 / `unsafe_op_in_unsafe_fn` 要求即便函数本身已是 `unsafe fn`，函数体内执行 unsafe 操作仍需显式 `unsafe { ... }` 包裹。上面伪码中的 `Owned::from_raw_parts(...)` 调用已用 `unsafe { ... }` 包裹并附 `// SAFETY:` 注释。
>
> **关于 `into_raw_parts` 中读取 storage base pointer：** 由于 `ManuallyDrop` 内仍然对内部 storage 拥有所有权，`storage_base_mut_ptr_unchecked` 是 `Owned<A>` 的 `pub(crate)` 内部 helper，等价于 `Owned::as_mut_ptr` 的读字段实现，但避免对 `&mut` 的形式要求；调用本身仍是 `unsafe`，因为读取裸指针不构造任何外部别名。

**设计约束：** `into_raw_parts` 仅适用于 Owned 存储，且导出的内存布局必须满足 Xenon 的 owned 不变量：F-order contiguous、`offset == 0`、canonical F-order strides。若调用方持有的是 view 或带 offset 的逻辑子视图，必须先显式物化为新的 owned contiguous tensor，再跨越 FFI 边界导出裸指针。如需将 View 转为 Owned 再解构，参见 `21-type.md §5.5`。

**内存回收规则：**

| 规则                 | 说明                                                            |
| -------------------- | --------------------------------------------------------------- |
| ✅ 重建张量后 Drop   | 使用 `Tensor::from_raw_parts_owned(raw)` 重建，让 Drop 处理释放 |
| ❌ 直接调用系统 free | 分配器不匹配，导致 UB 或内存泄漏                                |
| ❌ 忽略返回值        | 内存泄漏                                                        |

**裸指针直接构造 Owned 张量的约束：** 当前版本不提供从任意裸指针直接构造 `Owned` 张量的接口。`from_raw_parts()` / `from_raw_parts_mut()` 仅构造视图（View / ViewMut），`from_raw_parts_owned()` 仅从 `into_raw_parts()` 导出的 `OwnedRawParts` 重建 Owned 张量。原因是 `Owned` 存储需要 Xenon 分配器的元数据（capacity、alignment），这些信息无法从单一裸指针推断。若调用方需要从裸指针创建 Owned 张量，须先将数据复制到 Xenon 分配的张量中（如通过 `Tensor::from_shape_vec()` 等构造方法）。

### 5.8 视图方法

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

### 5.9 Good/Bad 对比

```rust,ignore
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

```rust,ignore
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

**设计决策：** `shape` 与 `strides` 分离建模：`shape` 字段类型为 `D`，`strides` 字段类型为 `Strides<D>`。

**实现方案：**

| 层次                 | 类型         | 说明                                                                                                  |
| -------------------- | ------------ | ----------------------------------------------------------------------------------------------------- |
| `TensorBase.strides` | `Strides<D>` | 与 shape rank 一致：静态维度编译期保证、`IxDyn` 构造期保证（参见 `06-layout.md §5.5`）                |
| `strides()` 返回值   | `&[usize]`   | 直接来自 `Strides<D>`（参见 `06-layout.md §5`）                                                       |
| layout 模块计算      | `usize`      | F-order、转置与零步长布局在 layout 层计算（参见 `06-layout.md §5.3` / `§5.7`）                        |

**权衡：**

- `Strides<D>` 保证 strides 与 shape rank 一致：对静态维度（`Ix0`-`Ix6`）通过类型系统在编译期保证；对动态维度（`IxDyn`）通过 `Strides<IxDyn>` 内部 `Vec<usize>` 与 `IxDyn` 的 `dims: Vec<usize>` 在构造期验证 `len()` 相等（参见 `06-layout.md §5.5`）。
- 静态维度使用栈分配数组（性能）
- 当前版本仅覆盖非负步长与零步长（广播）；负步长布局不在当前版本范围内（参见 `需求说明书 §7`）

### 6.2 offset 字段设计

```
Original array storage: [a, b, c, d, e, f, g, h]
shape: [8], strides: [1], offset: 0

After slicing [2..5]:
storage: [a, b, c, d, e, f, g, h]  // shared, no copy
shape: [3], strides: [1], offset: 2  // metadata adjustment only
Logical view: [c, d, e]
```

- **安全性论证**：安全构造路径必须调用 `validate_access_range(shape, strides, offset, storage_len)` 之类的检查来计算所有逻辑索引可达的最小/最大物理偏移，并验证它们都落在底层 storage 范围内。unsafe raw-parts 路径可复用这些检查拒绝明显错误的元数据，但访问范围前提本身仍由调用方保证。只有这些前提成立后，`as_ptr()` 才能把“logical-first pointer”定义为逻辑首元素地址。
- **重要设计约定：** `TensorBase::offset` 是所有存储模式（Owned、ViewRepr、ViewMutRepr、ArcRepr）共用的唯一偏移字段。`ArcRepr` 不存储独立的 offset — 数据访问的起始位置完全由 `TensorBase::offset` 决定。这避免了双重偏移计算的 bug，并使偏移逻辑集中在一处。
- **logical-first pointer 契约：** `TensorBase::as_ptr()` / `as_mut_ptr()` 返回的是逻辑首元素指针，而不是 storage base pointer。layout 标志计算、连续切片快路径和 FFI raw-parts safety 文档都必须使用这一同一约定；若需要 storage base pointer，只能通过 storage 层 API 或 raw-parts 输入显式提供。
- **raw-parts 设计补充：** `storage_len` 是 raw-parts 视图构造的必填输入。`ViewRepr` / `ViewMutRepr` 需要保存 backing storage 的可访问元素数，`validate_access_range(...)` 也必须基于该长度执行边界校验；仅有 `ptr + shape + strides + offset` 不足以安全重建视图。
- **空张量指针说明：** 当 `len == 0` 时，元数据仍可描述一个合法的空视图，但 `as_ptr()` / `as_mut_ptr()` 不能对 storage base pointer 执行 `add(offset)`。Rust 的指针算术要求结果仍落在同一已分配对象内；对悬垂哨兵或空存储基指针做偏移计算会触发未定义行为，且对 ZST 即使执行 `add(0)` 也不应依赖这种做法。设计上因此统一采用“空张量 `offset` 仅需满足 `offset <= storage_len`，但不做实际偏移”的契约，并让指针 API 直接返回 `NonNull::dangling().as_ptr()` 快路径。

> **错误字段约定（v2.0.x）：** `validate_access_range` 与 `validate_non_overlapping_layout`
> 构造的 `XenonError::InvalidLayout` 字段必须使用 `26-error.md §5.1` 定义的封闭枚举
> （`InvalidLayoutReason` / `StorageKindTag` / `Cow<'static, str>`），不得退化为自由
> 文本，也**不得**在本节发明未列入 `26-error.md §5.1` 的局部 reason——下面伪码中
> 出现的所有 `InvalidLayoutReason::*` 标识符（`ShapeProductOverflow` /
> `EmptyTensorOffsetExceedsStorage` / `StrideExceedsIsizeMax` / `StrideSpanOverflow`
> / `AccessRangeOverflow` / `AccessRangeExceedsStorage` / `AmbiguousOverlap` 等）
> 必须已在 `26-error.md §5.1` 列出。新增 case 必须先扩展 `26-error.md §5.1` 的枚举。
> `storage_kind` 由调用方提供（视图路径填 `StorageKindTag::View` /
> `StorageKindTag::ViewMut`，owned raw-parts 路径填 `StorageKindTag::Owned`）。
>
> **验证前提：** `validate_access_range` 假定调用方已在更早阶段拒绝
> `stride > isize::MAX` 的 stride，或在算法首部追加该检查。这是因为后续
> `<*const A>::add(stride * (shape - 1))` 的指针运算需要 stride 可表示为非负
> `isize`；该前提在 §5.7 的 `# Safety` 列表中已显式列入构造器自验项。

```text
validate_access_range(shape, strides, offset, storage_len, op_name, kind):
    if shape.checked_size() overflows:
        return Err(XenonError::InvalidLayout {
            operation: Cow::Borrowed(op_name),
            storage_kind: kind,
            shape: shape.slice().to_vec(),
            strides: strides.as_slice().to_vec(),
            offset,
            storage_len,
            reason: InvalidLayoutReason::ShapeProductOverflow,
        })

    if shape.checked_size() == Ok(0):
        if offset > storage_len:
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::EmptyTensorOffsetExceedsStorage,
            })
        return Ok(())

    // Reject any stride whose pointer-arithmetic equivalent would overflow isize
    // (this is the # Safety precondition for from_raw_parts*).
    for axis in 0..ndim:
        if strides[axis] > isize::MAX as usize:
            return Err(XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::StrideExceedsIsizeMax,
            })

    max_offset = offset

    for axis in 0..ndim:
        if shape[axis] == 0:
            return Ok(())

        span = (shape[axis] - 1).checked_mul(strides[axis])
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::StrideSpanOverflow,
            })?
        max_offset = max_offset.checked_add(span)
            .ok_or_else(|| XenonError::InvalidLayout {
                operation: Cow::Borrowed(op_name),
                storage_kind: kind,
                shape: shape.slice().to_vec(),
                strides: strides.as_slice().to_vec(),
                offset,
                storage_len,
                reason: InvalidLayoutReason::AccessRangeOverflow,
            })?

    if max_offset >= storage_len:
        return Err(XenonError::InvalidLayout {
            operation: Cow::Borrowed(op_name),
            storage_kind: kind,
            shape: shape.slice().to_vec(),
            strides: strides.as_slice().to_vec(),
            offset,
            storage_len,
            reason: InvalidLayoutReason::AccessRangeExceedsStorage,
        })

    return Ok(())
```

### 6.3 内存布局示意

```
Tensor2<f64> = TensorBase<Owned<f64>, Ix2>

┌───────────────────────────────────────────┐
│ storage: Owned<f64>                       │
│  ┌─────────────────────────────────────┐  │
│  │ data: AlignedBuf<f64> (64B aligned) │  │
│  │ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]      │  │
│  └─────────────────────────────────────┘  │
│ shape: Ix2(2, 3)                          │
│ strides: Strides::from_slice(&[1, 2])     │
│ offset: 0                                 │
│ flags: F_CONTIGUOUS | ALIGNED             │
└───────────────────────────────────────────┘

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
  - 内容: 结构体定义，6 个字段：storage、shape、strides、offset、flags、derived_from_view_mut（最后一个为私有 1-bit ViewMut→View 降级来源标记，仅由 view_mut().view() / 内部切片降级路径设置；详见 §5.1 / §5.3）
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
  - 内容: `shape()`/`strides()`/`ndim()`/`len()`/`is_empty()`/`offset()`/`raw_dim()`/`flags()`/`storage_kind()`/`access_semantics()`/`data_location()`
  - 测试: `test_shape_query`, `test_len_empty`, `test_access_semantics`, `test_data_location`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现布局查询委托方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `layout_state()`/`is_f_contiguous()`/`is_aligned()`/`has_zero_stride()`
  - 测试: `test_layout_flags_delegate`, `test_layout_state_classification`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6**: 实现指针访问与连续切片方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `as_ptr()`/`as_storage_ptr()`/`as_mut_ptr()`/`as_slice()`/`as_mut_slice()`
  - 测试: `test_as_ptr`, `test_as_mut_ptr`, `test_as_storage_ptr`, `test_as_slice`, `test_as_mut_slice`
  - 前置: T4
  - 预计: 10 min

### Wave 3: 构造和视图

- [ ] **T7**: 实现 `from_raw_parts` 系列 (construct.rs)
  - 文件: `src/tensor/construct.rs`
  - 内容: `from_raw_parts`(不可变)/`from_raw_parts_mut`(可变)，显式接收 `storage_len` 并统一走 `validate_access_range`
  - 测试: `test_from_raw_parts_view`, `test_from_raw_parts_mut`, `test_from_raw_parts_invalid_range`
  - 前置: T2
  - 预计: 10 min

- [ ] **T8**: 实现内部 unsafe 构造方法 (construct.rs)
  - 文件: `src/tensor/construct.rs`
  - 内容: `from_raw_vec_unchecked`（pub(crate) unsafe 内部方法）
  - 测试: `test_from_raw_vec_unchecked_valid`, `test_from_raw_vec_unchecked_invalid_shape`
  - 前置: T5, T7
  - 预计: 5 min

> **注意**：公开安全构造方法 `from_shape_vec` 的实现位于 `src/construct/from.rs`（参见
> `18-construction.md §5.3`），本文件 §5.6 仅列其公开签名，不属于本目录任务。对应测试
> `test_from_shape_vec_valid` / `test_from_shape_vec_invalid` 应在构造模块的测试文件中。

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

### 8.2 单元测试清单

| 测试函数                            | 测试内容                                                          | 优先级 |
| ----------------------------------- | ----------------------------------------------------------------- | ------ |
| `test_tensor_shape_2d`              | `Tensor2::from_shape_vec([3,4], data)` 后 shape 查询              | 高     |
| `test_tensor_len`                   | `len()` 返回 shape 乘积                                           | 高     |
| `test_tensor_is_empty`              | 空数组 `is_empty()` 返回 true                                     | 高     |
| `test_tensor_ndim_static`           | `Tensor2` 的 `ndim()` == 2                                        | 高     |
| `test_tensor_ndim_dynamic`          | `TensorD` 的 `ndim()` 运行时                                      | 中     |
| `test_tensor_strides_f_order`       | F-order 步长正确 `[1, shape[0], ...]`                             | 高     |
| `test_tensor_layout_state`          | 按 `crate::layout::LayoutState`（参见 `06-layout.md §5`）完成分类 | 高     |
| `test_tensor_flags_f_contiguous`    | 新构造张量 F-连续                                                 | 高     |
| `test_tensor_flags_aligned`         | 新构造张量对齐                                                    | 高     |
| `test_tensor_as_ptr`                | 指针指向正确位置                                                  | 高     |
| `test_tensor_as_mut_ptr`            | 可变指针指向正确位置                                              | 高     |
| `test_tensor_storage_kind`          | `Owned`/`View`/`ViewMut`/`Shared` 的存储位置查询正确              | 高     |
| `test_tensor_access_semantics`      | 各存储模式返回正确的 `AccessSemantics`                           | 高     |
| `test_tensor_data_location`         | `data_location()` 返回 `DataLocation::Cpu`                       | 中     |
| `test_tensor_as_storage_ptr`        | `as_storage_ptr()` 返回 storage 基指针而非逻辑首元素指针         | 高     |
| `test_tensor_has_zero_stride`       | 广播视图 `has_zero_stride()` 返回 true                           | 中     |
| `test_tensor_as_slice`              | 连续张量 `as_slice()` 返回 `Some`，非连续返回 `None`             | 高     |
| `test_tensor_as_slice_empty`        | 空张量 `as_slice()` 返回 `Some(&[])`                             | 中     |
| `test_tensor_as_mut_slice`          | 可写连续张量 `as_mut_slice()` 返回 `Some`                        | 高     |
| `test_tensor_view`                  | `view()` 创建正确视图                                             | 高     |
| `test_tensor_view_mut`              | `view_mut()` 创建正确可变视图                                     | 高     |
| `test_from_shape_vec_valid`         | 合法构造成功                                                      | 高     |
| `test_from_shape_vec_len_mismatch`  | 长度不匹配返回错误                                                | 高     |
| `test_from_raw_parts_invalid_range` | raw-parts 越界访问范围被拒绝                                      | 高     |
| `test_type_aliases_compile`         | 所有类型别名编译通过                                              | 高     |
| `test_tensor0_scalar`               | 0D 标量张量 `len()==1`                                            | 中     |
| `test_tensor_empty_dim`             | 含 0 维度的张量 `is_empty()`                                      | 中     |

### 8.3 边界测试场景

| 场景                  | 预期行为                                     |
| --------------------- | -------------------------------------------- |
| 空张量 `shape=[0, 3]` | `len()==0`, `is_empty()==true`               |
| 单元素 `shape=[1, 1]` | `len()==1`, F-连续                           |
| 标量 `Tensor0<f64>`   | `ndim()==0`, `len()==1`                      |
| 高维 `Tensor6`        | `ndim()==6`, 步长正确                        |
| 动态维度 `TensorD`    | `ndim()` 运行时值正确                        |
| 大张量 `10_000_000` 元素 | 构造成功，长度与 flags 保持正确              |
| 非连续转置视图        | 可构造 `view()`，但连续切片快路径返回 `None`                          |
| 非零 offset 视图      | `as_storage_ptr() != as_ptr()`，差值等于 `offset`                     |
| 空张量 + 多种 offset  | 只要 `offset <= storage_len` 即合法                                    |
| 非法元素类型编译失败  | compile-fail 测试拒绝不满足元素约束的类型    |

### 8.4 属性测试不变量

| 不变量                                            | 测试方法                                  |
| ------------------------------------------------- | ----------------------------------------- |
| `tensor.len() == tensor.shape().iter().product()` | 随机形状                                  |
| `tensor.view().shape() == tensor.shape()`         | 随机形状和存储模式                        |
| `from_shape_vec` 后 `is_f_contiguous() == true`   | 随机合法形状                              |
| 安全构造路径在访问范围不合法时返回错误            | 随机 shape/stride/offset/storage_len 组合 |

### 8.5 集成测试

| 测试文件               | 测试内容                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `tests/test_tensor.rs` | `from_shape_vec` / `view` / `view_mut` / `as_ptr` 与 `dimension`、`storage`、`layout`、`index` 的端到端协同路径 |

### 8.6 Feature gate / 配置测试

| 配置项         | 覆盖方式              | 说明                                        |
| -------------- | --------------------- | ------------------------------------------- |
| 默认配置       | 常规单元/集成测试路径 | 本模块无独立 feature gate，默认配置即主路径 |
| 非默认 feature | 不适用                | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型       | 覆盖方式                                                           | 说明                                                         |
| -------------- | ------------------------------------------------------------------ | ------------------------------------------------------------ |
| 存储访问边界   | compile-fail 测试只读存储不暴露可写 API                            | 验证 `Storage` / `StorageMut` 约束在 `TensorBase` 上正确投影 |
| 别名边界       | 编译期验证 `Tensor{N}` / `TensorView{N}` / `ArcTensor{N}` 全部展开 | 验证便捷别名与核心类型实例化保持一致                         |
| raw-parts 边界 | 编译期与运行时测试结合验证 `Strides<D>` / `D` / `offset` 契约      | 验证元数据契约不会被类型层或构造层打破                       |

---

## 9. 与其他模块的交互

### 9.1 核心数据流

```text
User calls constructors / `view()` / `view_mut()` / query APIs
    │
    ├── dimension provides shape metadata
    ├── storage provides the backing buffer and ownership model
    ├── tensor combines shape + strides + offset + flags
    ├── layout computes contiguity / alignment / zero-stride flags
    └── index / iter / math / ffi and other upper layers continue consuming `TensorBase` as the unified carrier
```

### 9.2 典型构造数据流图

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

### 9.3 与 storage 模块的接口

| 接口                                             | 方向                     | 契约                                                                                                           |
| ------------------------------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `Storage::as_ptr()` / `StorageMut::as_mut_ptr()` | `tensor` 消费 `storage`  | storage 层返回 storage base pointer；`TensorBase` 负责叠加 `offset` 并形成 logical-first pointer               |
| `Owned::from_vec_aligned(data)`                  | `tensor` 消费 `storage`  | 当前版本默认采用 64 字节对齐分配策略；若存在例外，须显式文档化，且不得改变 `需求说明书 §19` 规定的逻辑元素顺序 |
| `Storage<Elem = A>` / `StorageMut<Elem = A>`     | `tensor` 消费 `storage`  | 元素类型、只读/可写访问能力完全由存储模式 trait 约束决定，`tensor` 不重复维护独立元素类型参数                  |

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

### 9.4 与 dimension 模块的接口

| 接口                        | 方向                      | 契约                                                                                                                     |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `Dimension::slice()`        | `tensor` 消费 `dimension` | `shape()` 返回的轴长切片与底层 `D` 保持一致，不复制逻辑元数据语义                                                        |
| `Dimension::checked_size()` | `tensor` 消费 `dimension` | `tensor` 基于已验证 shape 暴露 `len()`；凡直接消费 `shape: D` 做构造、分配或范围校验时必须先用 `checked_size()` 避免溢出 |
| `Dimension` rank contract   | `tensor` 消费 `dimension` | `shape` 与 `Strides<D>` 必须保持相同 rank，供 `simd` / `parallel` 继续消费同一布局契约                                   |

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
        // from_raw_vec_unchecked also validates shape via checked_size() internally,
        // ensuring this expect never fires in sound code.
        self.shape.checked_size().expect("tensor shape must be validated before len()")
    }
}
```

`len()` 的返回值始终来源于逻辑 `shape`，不允许退化为读取 `storage.len()`；后者仅用于 raw-parts 与底层访问范围校验。

### 9.5 与 layout 模块的接口

| 接口                                                          | 方向                             | 契约                                                                                           |
| ------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `layout::compute_f_strides(&shape)`                           | `tensor` 消费 `layout`           | 安全构造拥有型连续张量时统一按 F-order 生成 stride                                             |
| `layout::compute_layout_flags(&shape, &strides, logical_ptr)` | `tensor` 消费 `layout`           | `flags` 的计算必须基于 logical-first pointer 契约，与 `as_ptr()` / `as_slice()` 的可见语义一致 |
| `LayoutState` / layout flags queries                          | `simd`、`parallel` 消费 `tensor` | 上游加速模块只通过 `TensorBase` 查询连续性、对齐和广播状态，不绕过 `tensor` 直接重建布局判断   |

```text
from_shape_vec construction call chain (logical illustration;
authoritative implementation resides in src/construct/, see 18-construction.md §5.3):

    shape.into_dimension()
         │
         ├─ shape.checked_size()            → element count (or InvalidShape)
         ├─ data.len() != expected          → InvalidShape
         ├─ layout::compute_f_strides(&shape) → F-order strides (or InvalidLayout)
         ├─ Owned::from_vec_aligned(data)   → 64-byte aligned storage
         ├─ compute_layout_flags(&shape, &strides, logical_ptr)
         │                                  → LayoutFlags
         └─ call tensor's `pub(crate)` internal constructor with private-field access
                                             → Ok(TensorBase<Owned<f64>, Ix2>)
```

---

## 10. 错误处理与语义边界

| 项目              | 内容                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | `from_shape_vec()`、`from_raw_parts*()` 等构造校验失败返回 `XenonError`；上下文字段应包含操作名、shape、strides、offset、storage_len 或期望长度等元数据 |
| Panic             | 本模块公开安全构造不以 panic 作为常规错误通道；仅在内部已验证快捷路径或明显违背 `unsafe` 前提的后续使用中可能出现 panic/未定义行为风险                  |
| 路径一致性        | scalar、SIMD 快路径与 parallel 上游消费必须共享同一逻辑首元素与 flags 语义，不允许因路径差异改变结果                                                    |
| 容差边界          | 不适用                                                                                                                                                  |

---

## 11. 设计决策记录

### 决策 1：TensorBase<S, D> 双参数泛型设计

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
| 替代方案 | 裸用 `strides: D`（直接把 `D` 当 stride carrier 用）— 放弃，无法显式区分 shape 与 stride 的语义，且会让 `Strides<D>` 失去 newtype 文档价值。**当前方案使用 `Strides<D>` newtype 内部复用 `D` 表示**（详见 `06-layout.md §5.2`），这是"复用 D 的存储表示但隔离语义"的折中——并非该替代方案，请勿混淆 |

### 决策 3：offset 字段必要性

| 属性     | 值                                                                             |
| -------- | ------------------------------------------------------------------------------ |
| 决策     | 包含 `offset: usize` 字段                                                      |
| 理由     | 切片操作 O(1)（仅修改元数据）；无数据复制；统一机制适用所有存储模式；BLAS 兼容 |
| 替代方案 | 无 offset，切片时调整 storage 指针 — 放弃，Owned 无法调整指针                  |

### 决策 4：不实现 Deref<Target=TensorView>

| 属性     | 值                                                                            |
| -------- | ----------------------------------------------------------------------------- |
| 决策     | 不实现 `Deref<Target = TensorView>`                                           |
| 理由     | 显式优于隐式（`.view()` 清晰表达意图）；避免隐式生命周期传播；与 ndarray 一致 |
| 替代方案 | 实现 Deref — 放弃，隐式转换可能导致意外借用                                   |

### 决策 5：`OwnedRawParts<A, D>` 不承诺 C ABI 稳定

| 属性     | 值                                                                                                                                                              |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 移除原先的 `#[repr(C)]` 注释，明确 `OwnedRawParts<A, D>` 是 Rust 内部 round-trip 类型，不作为 FFI ABI 边界                                                       |
| 理由     | `D` 与 `Strides<D>` 是 Rust 泛型，对 `IxDyn` 包含 `Vec<usize>` 字段，无法被 `#[repr(C)]` 稳定描述；FFI 已有专用 `TensorExport` / `TensorExportMut` 类型           |
| 替代方案 | 强行保留 `#[repr(C)]` — 放弃，会让外部 C 代码误以为可解码该结构，导致跨语言 UB                                                                                  |

### 决策 6：`from_raw_vec_unchecked` 走 `pub(crate)` 真正的 unchecked 路径

| 属性     | 值                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `from_raw_vec_unchecked` 不再隐式重做 `shape.checked_size()` 验证；调用方在 `# Safety` 中证明已验证                              |
| 理由     | "_unchecked" 名称要求语义透明：要么不做任何 fallible 校验，要么提供 `Result` 版本。混合"unchecked + 内部 panic"会让安全契约模糊 |
| 替代方案 | 改名 `from_raw_vec_with_shape_check` 并返回 `Result` — 放弃，会引入冗余 fallible 路径与上层 `from_shape_vec` 重复                |

---

## 12. 性能考量

| 方面       | 设计决策                                                     |
| ---------- | ------------------------------------------------------------ |
| 栈上元数据 | 静态维度（Ix0-Ix6）的 TensorBase 元数据完全在栈上            |
| 零成本抽象 | 不同存储模式编译为不同类型，无虚调用                         |
| 查询复杂度 | `shape()`/`ndim()`/`flags()` 为 O(1)，`len()` 当前为 O(ndim) |
| 视图零拷贝 | `view()`/`view_mut()` 仅复制元数据                           |
| 单态化     | Dimension + Storage trait 在泛型上下文中单态化               |

---

## 13. 平台与工程约束

| 约束       | 说明                                    |
| ---------- | --------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`  |
| MSRV       | Rust 1.85+                              |
| 单 crate   | 保持单 crate 边界                       |
| SemVer     | 张量元数据字段与构造契约变更遵循 SemVer |
| 最小依赖   | 无新增第三方依赖                        |

---

## 附录 A：完整类型关系图

```
TensorBase<S, D>
├── TensorBase<Owned<A>, D>          →  Tensor<A, D>
│   ├── Tensor0<A>                      (Ix0)
│   ├── Tensor1<A>                      (Ix1)
│   ├── ...
│   ├── Tensor6<A>                      (Ix6)
│   └── TensorD<A>                      (IxDyn)
├── TensorBase<ViewRepr<'a, A>, D>   →  TensorView<'a, A, D>
│   ├── TensorView0<'a, A>              (Ix0)
│   ├── TensorView1<'a, A>              (Ix1)
│   ├── ...
│   ├── TensorView6<'a, A>              (Ix6)
│   └── TensorViewD<'a, A>             (IxDyn)
├── TensorBase<ViewMutRepr<'a, A>, D>→  TensorViewMut<'a, A, D>
│   ├── TensorViewMut0<'a, A>           (Ix0)
│   ├── TensorViewMut1<'a, A>           (Ix1)
│   ├── ...
│   ├── TensorViewMut6<'a, A>           (Ix6)
│   └── TensorViewMutD<'a, A>          (IxDyn)
└── TensorBase<ArcRepr<A>, D>        →  ArcTensor<A, D>
    ├── ArcTensor0<A>                    (Ix0)
    ├── ArcTensor1<A>                    (Ix1)
    ├── ...
    ├── ArcTensor6<A>                    (Ix6)
    └── ArcTensorD<A>                    (IxDyn)
```

## 附录 B：命名约定速查

| 模式               | 示例                    | 含义               |
| ------------------ | ----------------------- | ------------------ |
| `Tensor{N}`        | `Tensor2<A>`            | N 维拥有型数组     |
| `TensorD`          | `TensorD<A>`            | 动态维度拥有型数组 |
| `TensorView{N}`    | `TensorView2<'a, A>`    | N 维不可变视图     |
| `TensorViewMut{N}` | `TensorViewMut2<'a, A>` | N 维可变视图       |
| `ArcTensor{N}`     | `ArcTensor2<A>`         | N 维 Arc 共享数组  |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
