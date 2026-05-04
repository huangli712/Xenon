# 索引操作模块设计

> 文档编号: 17
> 模块目录: src/index/
> 任务阶段: Phase 3
> 前置文档: 02-dimension.md, 06-layout.md, 07-tensor.md, 26-error.md
> 需求参考: 需求说明书 §4、§6、§18、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位/概述

### 1.1 职责边界表

| 职责             | 包含                                                                             |
| ---------------- | -------------------------------------------------------------------------------- |
| 多维整数索引     | `usize` 多维索引、`try_at` / `try_at_mut` / `get` / `get_mut` / `get_unchecked*` |
| 范围索引（切片） | 以范围描述符表达的只读切片、`slice` 编程式接口                                   |
| 元数据更新       | 按 F-order 规则更新 offset、shape、stride 与布局标记                             |
| 错误边界         | 安全接口返回可恢复错误，unsafe 接口由调用方保证前提                              |

| 职责             | 不包含                                     |
| ---------------- | ------------------------------------------ |
| 多维整数索引     | 负索引、布尔掩码、整数数组高级索引         |
| 范围索引（切片） | 负步长切片、共享可写切片、隐式复制切片     |
| 元数据更新       | 改变逻辑元素顺序、引入与源张量无关的新存储 |
| 错误边界         | 将越界默认为 panic 的安全主路径            |

### 1.2 设计原则

| 原则               | 体现                                                            |
| ------------------ | --------------------------------------------------------------- |
| `usize` 元数据角色 | `usize` 仅用于索引、轴、shape 与切片边界，不属于张量元素类型    |
| F-order 一致性     | 所有偏移量推导都遵循 `offset = sum(index[i] * strides[i])`      |
| 零拷贝视图         | 切片优先共享底层数据，仅返回只读或共享只读结果                  |
| 安全/不安全分层    | 安全接口显式验证 rank 与边界；unsafe 路径仅跳过检查，不改变语义 |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 需求映射 | 需求说明书 §4、§6、§18、§27、§28                                                                                         |
| 范围内   | `usize` 多维索引、范围索引（切片）、rank 一致性检查、越界 recoverable error、unsafe 未检查变体、切片后 shape/stride 更新 |
| 范围外   | **本模块仅覆盖 NumPy "basic indexing" 的子集，不覆盖 NumPy "advanced/fancy indexing"。** 明确不在当前版本范围内的能力：(1) 负索引（`a[-1]`）；(2) 负步长切片（`a[::-1]`）；(3) 布尔掩码索引（`a[mask]`）；(4) 整数数组高级索引（`a[[0, 2, 4]]`）；(5) `np.newaxis` / `None` 添加新轴；(6) 共享可写视图；(7) 额外索引语法。**也不实现** `std::ops::Index` 与 `std::ops::IndexMut` trait（原因：标准库 trait 强制 panic 语义，不符合 Xenon 的 Result 错误模型）。访问元素请使用 `try_at()` / `get()` / `try_at_mut()` / `get_mut()` 或 `unsafe` 的 `get_unchecked` 系列。未来若引入 fancy indexing，需单独的设计文档与 ABI 兼容评估，**不应**作为隐式扩展加入本模块。 |
| 非目标   | 不新增索引能力，不引入新的存储模式或复制语义                                                                             |

---

## 3. 文件位置

```text
src/
└── index/
    ├── mod.rs           # module root and public re-exports
    ├── ndindex.rs       # NdIndex trait and tuple/slice index implementations
    ├── access.rs        # try_at/get/get_unchecked and mutable variants
    └── slice.rs         # SliceInfo, slice, shape/stride updates
```

按能力拆分为 `ndindex`、`access`、`slice`，可把“索引地址计算”和“切片元数据变换”分开维护；`mod.rs` 负责统一导出，保持对外仍是单一 `src/index/` 模块边界。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/index/
|
├── mod.rs
│   └── re-exports from ndindex, access, slice
|
├── ndindex.rs
│   ├── crate::dimension   # Dimension, Ix0~Ix6, IxDyn, rank / axis metadata
│   ├── crate::layout      # Strides<D>, F-order offset interpretation
│   ├── crate::error       # XenonError::InvalidAxis, IndexOutOfBounds, DimensionMismatch
│   └── crate::private     # Sealed (closes NdIndex from external implementation)
|
├── access.rs
│   ├── crate::tensor      # TensorBase<S, D>, .shape(), .strides(), .ndim(), storage mode query
│   ├── crate::dimension   # Dimension
│   ├── crate::layout      # Strides<D>
│   ├── crate::storage     # Storage, StorageMut, read-only / writable storage capability
│   └── crate::error       # XenonError::IndexOutOfBounds, DimensionMismatch
|
└── slice.rs
    ├── crate::tensor      # TensorBase<S, D>, TensorView<'a, A, I>
    ├── crate::dimension   # Dimension, Ix0~Ix6, IxDyn
    ├── crate::layout      # Strides<D>, layout flags
    ├── crate::storage     # Storage, read-only storage capability
    └── crate::error       # XenonError::InvalidArgument, IndexOutOfBounds
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                    |
| ----------- | --------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `TensorView<'a, A, I>`, `.shape()`, `.strides()`, `.ndim()`, storage mode query |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, rank / axis metadata                                             |
| `layout`    | `Strides<D>`, layout flags, F-order offset interpretation                                           |
| `storage`   | `Storage`, `StorageMut`, read-only / writable storage capability                                    |
| `error`     | `XenonError::InvalidAxis`, `InvalidArgument`, `IndexOutOfBounds`, `DimensionMismatch`               |
| `private`   | `Sealed`，用于封闭 `NdIndex` 的外部实现面                                                           |

### 4.3 依赖合法性

| 项目           | 说明                                                       |
| -------------- | ---------------------------------------------------------- |
| 新增第三方依赖 | 无                                                         |
| 合法性结论     | 合法；当前设计仅复用项目内既有模块与标准库                 |
| 替代方案       | 不适用；索引能力无需额外 crate，也不应因文档重写扩展依赖面 |

### 4.4 依赖方向声明

依赖方向：单向向上。`index` 仅消费 `tensor`、`dimension`、`layout`、`storage`、`error` 的既有能力，不被这些底层模块反向依赖。

---

## 5. 公共 API 设计

### 5.1 核心接口草案

```rust,ignore
use crate::private::Sealed;

pub trait NdIndex<D: Dimension>: Sealed {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Result<usize, XenonError>;

    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `index` length matches the dimension rank
    /// - Each index component is within bounds for the corresponding axis
    /// - The resulting offset does not overflow `usize`
    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize;

    /// Converts this index into a `Vec<usize>` for error reporting.
    ///
    /// Guarantees all `NdIndex` implementors can produce a uniform diagnostic
    /// representation regardless of whether the index originates from a tuple
    /// or a slice.
    fn to_index_vec(&self) -> Vec<usize>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SliceInfoElem {
    Index(usize),
    Range {
        start: usize,
        end: usize,
        // Step slicing is not in scope for the current version and may be designed in a future version.
    },
}

#[derive(Debug, Clone)]
pub enum SliceInfoIndices {
    Inline {
        len: u8,
        elems: [Option<SliceInfoElem>; 6],
    },
    Dynamic(Vec<SliceInfoElem>),
}

pub struct SliceInfo<I, D>
where
    I: Dimension,
    D: Dimension,
{
    indices: SliceInfoIndices,
    in_dim: D,
    out_dim: I,
}

impl<I, D> SliceInfo<I, D>
where
    I: Dimension,
    D: Dimension,
{
    /// Constructs a `SliceInfo` from indices and dimension types.
    ///
    /// Performs **structural** validation only; it does **not** validate
    /// per-axis bounds against any tensor shape. See decision 3 (B9.a).
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidArgument { operation: "SliceInfo::new",
    /// kind: InvalidArgumentKind::OperationSpecific { .. } }` when:
    ///
    /// - `indices.len() != in_dim.ndim()` (rank of slice descriptor must
    ///   match the input dimension type's rank).
    /// - `out_dim.ndim() != count_of(Range)` (one output axis per Range,
    ///   none per Index).
    /// - `Range { start, end }` has `start > end` — surfaced as
    ///   `InvalidArgumentKind::RangeStartAfterEnd { axis, start, end }`.
    ///
    /// Per-axis bounds checking against actual tensor shape (`Range.end <=
    /// shape[axis]`, `Index < shape[axis]`) is **deferred** to
    /// `TensorBase::slice(info)` because `SliceInfo::new` does not know the
    /// concrete tensor shape.
    pub fn new(indices: SliceInfoIndices, in_dim: D, out_dim: I) -> Result<Self, XenonError>;

    pub fn indices(&self) -> &SliceInfoIndices;

    pub fn input_dim(&self) -> &D;

    pub fn output_dim(&self) -> &I;
}
```

设计说明：为支持 `XenonError::IndexOutOfBounds` 与 `26-error.md` 的规范对齐，`NdIndex<D>` 将提供 `fn to_index_vec(&self) -> Vec<usize>`（或等价 helper）用于把任意合法索引表示统一转换为 `Vec<usize>`。这样 tuple-based `Ix0`~`Ix6` 与切片形式索引都能在错误上报路径中生成一致的结构化诊断数据。

`SliceInfo<I, D>` 是切片描述符的公开包装类型：`D` 表示输入维度，`I` 表示切片后的输出维度；其内部字段保持私有，必须通过带校验的公开构造器建立，以避免手工拼出"索引长度、输入维度、输出维度彼此矛盾"的无效状态。

**校验职责分工（决策 3 / 用户已批准 B9.a）**：

| 校验类型 | 由谁负责 | 时机 | 失败错误 |
| --- | --- | --- | --- |
| 结构性校验（rank 一致、output 维度匹配 Range 计数、Range start≤end） | `SliceInfo::new` | 构造时 | `XenonError::InvalidArgument { kind: InvalidArgumentKind::OperationSpecific / RangeStartAfterEnd }` |
| 边界校验（Range.end ≤ shape[axis]、Index < shape[axis]） | `TensorBase::slice(info)` | 切片应用时 | `XenonError::InvalidArgument { kind: InvalidArgumentKind::RangeOutOfBounds }` |

理由：`SliceInfo::new` 只接收 `in_dim: D`（维度类型，不携带 shape 值），无法验证具体 shape 边界。把边界校验下沉到 `TensorBase::slice(info)` 让 SliceInfo 可在不绑定具体 shape 的前提下被构造、传递、复用。

`SliceInfo::new` 在构造时执行的结构性校验：

1. **indices 长度 == in_dim.ndim()**：切片描述符的元素数量必须精确匹配输入维度数。失败 → `InvalidArgumentKind::OperationSpecific { argument: "indices", constraint: "len must equal in_dim.ndim()" }`。
2. **out_dim.ndim() == count_of(Range)**：每个 `Range` 元素保留一个输出轴，每个 `Index(usize)` 折叠一个轴；输出维度数必须等于 `Range` 元素的计数。失败 → `InvalidArgumentKind::OperationSpecific { argument: "out_dim", constraint: "ndim must equal Range count in indices" }`。
3. **Range 内部一致性**：每个 `Range { start, end }` 必须满足 `start <= end`。失败 → `InvalidArgumentKind::RangeStartAfterEnd { axis, start, end }`（参见 26-error v3.0.0 §5.1）。

`TensorBase::slice(info)` 在切片应用时执行的边界校验（决策 3 详见 §11）：

4. **每个 Range 的 `end <= shape[axis]`**：失败 → `InvalidArgumentKind::RangeOutOfBounds { axis, axis_len, start, end }`。
5. **每个 Index 的值 < shape[axis]**：失败 → `XenonError::IndexOutOfBounds { operation: "slice", attempted_index, axis, shape }`。

这为当前版本的 `slice()` 提供了稳定、可验证的编程式入口，同时把"shape-aware"校验留给真正持有 shape 的层。范围语法中的省略边界应在进入 `SliceInfoElem::Range` 前先被规范化为显式 `start` / `end`。

**Inline / Dynamic 选择规则**：`SliceInfoIndices::Inline { len, elems }` 使用固定 6 槽位，覆盖静态维度集合 `Ix0..Ix6` 的所有合法切片描述。当输入维度为 `IxDyn` 且 `indices.len() > 6` 时，`SliceInfo::new` 必须使用 `SliceInfoIndices::Dynamic(Vec<SliceInfoElem>)` 路径以容纳任意 rank。`indices.len() <= 6` 时两种表示都合法，但实现 SHOULD 优先选择 `Inline` 以避免堆分配。

### 5.2 张量访问与切片 API

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn try_at<I>(&self, index: I) -> Result<&A, XenonError>
    where
        I: NdIndex<D>;

    pub fn get(&self, index: &[usize]) -> Result<&A, XenonError>;

    /// # Safety
    ///
    /// Caller must ensure index is valid: len == ndim and each component < shape[i].
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;

    pub fn slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension;    // I = output dimension after slicing; corresponds to D in TensorView<'a, A, D> defined in 07-tensor.md

}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    pub fn try_at_mut<I>(&mut self, index: I) -> Result<&mut A, XenonError>
    where
        I: NdIndex<D>;

    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut A, XenonError>;

    /// # Safety
    ///
    /// Caller must ensure index is valid: len == ndim and each component < shape[i].
    /// Caller must also have exclusive mutable access to the referenced element.
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A;
}

```

- 当前版本把 `try_at()` / `try_at_mut()` 与 `slice()` 作为对外规范的主恢复路径；`get(&[usize])` / `get_mut(&[usize])` 保留为基于 slice index 的 convenience wrapper，不取代规范主入口。
- `get(&[usize])` / `get_mut(&[usize])` 作为 convenience wrapper：先验证 `index.len() == self.ndim()`（不一致时返回 `XenonError::DimensionMismatch { operation, expected: self.ndim(), actual: index.len() }`），再逐轴验证 `index[i] < shape[i]`（越界时返回 `XenonError::IndexOutOfBounds { operation: "get" / "get_mut", attempted_index: index.to_vec(), axis: 首个越界轴, shape: self.shape().to_vec() }`），最后用 `compute_offset` 直接计算偏移返回引用。这条路径**不通过** `try_at<I: NdIndex<D>>` 委托，因为对静态 `D=Ix2`，`IxDyn` 没有实现 `NdIndex<Ix2>`（封闭元素集合的 `NdIndex` 实现按维度类型严格分类），强制把 `&[usize]` 转 `IxDyn` 再走 `try_at` 会触发 trait bound 不满足。两条路径的偏移计算逻辑等价（都使用 §6.2 `compute_offset`），但 trait 分派路径不同，独立实现以避免类型约束混淆。
- `SliceInfo` 稳定构造入口： 调用方可通过 `SliceInfo::new(indices, in_dim, out_dim)` 直接构造切片描述符；该构造器是公开且带**结构性**校验的稳定 API（边界校验由 `TensorBase::slice(info)` 在应用时完成，参见 §5.1 表）。

### 5.3 Good / Bad 对比

```rust,ignore
// Good - checked indexing keeps recoverable errors on the main path.
// `try_at` is the canonical safe entry point; both `(usize, usize)` and
// `&[usize]` indices are accepted via the `NdIndex` impls.
let value  = tensor.try_at((2, 1))?;
let value2 = tensor.try_at(&[2, 1][..])?;

// Good - propagate validation failure instead of hiding it behind panic.
let value = tensor.try_at((2, 1))?;
```


```rust,ignore
// Good - unsafe path is only used when the caller already proved validity.
let value = unsafe { tensor.get_unchecked(&[1, 2, 0]) };

// Bad - using unchecked access as a substitute for normal validation.
let value = unsafe { tensor.get_unchecked(user_index.as_slice()) };
```

---

## 6. 内部实现设计

### 6.1 核心数据结构

`NdIndex<D>` 负责把多维索引转换为偏移量；`SliceInfoElem` / `SliceInfoIndices` 负责表达“定点索引”和“范围索引”的组合。两类描述都只接受 `usize`，从类型层面排除负索引与负步长。`SliceInfoIndices::Inline` 使用固定 6 槽位表示短切片描述，尾部未使用的槽位填充为 `None`，`len` 表示前缀中实际参与校验与计算的元素数量。`SliceInfo<I, D>` 额外负责把这些索引描述与输入/输出维度绑定，但其内部字段不对外公开，只能通过构造器统一校验。

### 6.2 偏移量计算

```rust,ignore
fn compute_offset<D: Dimension>(index: &[usize], strides: &Strides<D>) -> Option<usize> {
    let mut offset = 0usize;
    for (&idx, &stride) in index.iter().zip(strides.iter()) {
        let term = idx.checked_mul(stride)?;
        offset = offset.checked_add(term)?;
    }
    Some(offset)
}
```

内部不变式：

- 索引元组长度必须与张量 rank 一致。
- 每个索引分量必须落在对应轴的有效范围内。
- 所有偏移量计算必须使用已有 stride，不得假设连续布局。
- 偏移量计算使用 checked arithmetic，或依赖已验证的合法 `shape` / `stride` 组合证明其不会溢出；对安全接口，任何溢出都必须转为可恢复错误。
- 任何安全接口在生成引用前都必须先完成上述验证。

### 6.3 切片元数据更新

```text
TensorBase::slice(info):
    // info already passed structural validation in SliceInfo::new.
    // This step performs SHAPE-AWARE bounds validation and metadata update.
    1. Validate info.input_dim() matches self.shape() rank (already guaranteed
       by D type, but assert in debug for safety).
    2. Initialize `slice_delta = 0` (relative element-unit delta accumulator).
       For each SliceInfoElem at axis i:
       a. If Index(idx): check `idx < shape[i]`; on failure return
          IndexOutOfBounds { operation: "slice", attempted_index, axis: i, shape }.
          Fold into `slice_delta` with checked_add(checked_mul(idx, stride[i])).
          Drop axis i from output shape and stride.
       b. If Range { start, end }: check `end <= shape[i]`; on failure return
          InvalidArgument { kind: RangeOutOfBounds { axis: i, axis_len: shape[i],
          start, end } }. (start <= end already guaranteed by SliceInfo::new.)
          Fold start into `slice_delta` with checked_add(checked_mul(start, stride[i])).
          Update output shape[axis] = end - start; keep stride[axis] unchanged.
       After the loop completes, compute the absolute new offset with
       checked arithmetic: `new_offset = self.offset.checked_add(slice_delta)
       .ok_or_else(|| XenonError::InvalidLayout {
           operation: Cow::Borrowed("slice"),
           reason: InvalidLayoutReason::AccessRangeExceedsStorage,
           ..
       })?` (only this absolute `new_offset` is written to the resulting
       `TensorBase.offset`). The `+` operator alone is insufficient: although
       each per-axis `slice_delta` contribution uses `checked_add` /
       `checked_mul`, the final fold back to absolute storage-base offset
       can still overflow at the `usize::MAX` boundary on adversarial inputs
       (R15 B-01 closes this gap).
    3. Recompute layout flags via compute_layout_flags::<A, I>(&new_shape,
       &new_strides, logical_ptr) where `logical_ptr` is computed per the
       v3.0.2 SAFETY rule below: for empty results (`product(new_shape) == 0`)
       use NonNull::<A>::dangling().as_ptr() (do NOT touch the storage
       pointer); for non-empty results use unsafe { self.as_ptr().add(slice_delta) }
       — note `self.as_ptr()` already includes self.offset, so we add the
       relative slice_delta exactly once. Forbidden: `self.as_ptr().add(new_offset)`
       (would double-apply offset). The unsafe pointer add executes only after
       shape-aware bounds checks and checked offset arithmetic have proved
       the element offset valid. See "切片 offset 计算与空切片规则（v3.0.2）"
       below for the full SAFETY contract.
    4. Construct and return TensorView<'_, A, I> with ViewRepr borrowed from
       self.storage.
```

切片后的语义约束如下：

- 结果须保持原有逻辑元素顺序。
- `Index(usize)` 会折叠对应轴并以 checked arithmetic 累加 offset；任一 checked_mul / checked_add 溢出返回 `XenonError::InvalidLayout { reason: AccessRangeExceedsStorage, .. }`（参见 26-error v3.0.0 §5.1 `InvalidLayoutReason`）。
- `Range` 会按起止边界更新 shape；对应轴的 stride 值保持不变。
- 切片结果与源张量共享底层数据时，仅可落在只读或共享只读范围内，不提供共享可写视图。
- **存储表示绝对降级：** 范围索引/切片产出的张量始终承载 `ViewRepr<'a, A>`，与 `15-broadcast.md §6.4` 的广播降级规则、`16-shape.md §5.3` 的转置降级规则保持一致（统一规则见 `05-storage.md v2.0.0 §5.11.1`）。无论源张量是 `Owned<A>`、`ArcRepr<A>`、`ViewRepr<'_, A>` 还是 `ViewMutRepr<'_, A>`，切片产出的视图均为 `ViewRepr<'a, A>`（生命周期绑定源张量），不保留 `ArcRepr` 的引用计数共享所有权语义。
- **`derived_from_view_mut` 传播规则（v3.0.2）：** 切片结果的私有内部字段 `derived_from_view_mut: bool`（参见 `07-tensor.md §5.1` / §5.3）按以下规则设置：
  1. 源 `storage_kind() == StorageKind::ViewMut` → 切片结果设置 `derived_from_view_mut = true`（即使切片本身没有零步长轴）；
  2. 源 `storage_kind() == StorageKind::View` 且源 `derived_from_view_mut == true` → 切片结果继承 `derived_from_view_mut = true`；
  3. 其他所有情形（源为 `Owned` / `Shared` / `View` 且未带降级标记） → 切片结果 `derived_from_view_mut = false`。
  
  `access_semantics()` 的判定**不仅看零步长**，必须按 `07-tensor.md §5.3` 的完整 5-rule 表（结合 `storage_kind()` + `layout_flags().has_zero_stride()` + `derived_from_view_mut`）。如果只看零步长，从可写视图切片得到的普通 contiguous 只读视图会被误报为 `ReadOnly`，违反"由 ViewMut 降级而来的共享只读视图"的语义边界。
- 布局状态只能重新落在 `FContiguous`、`NonContiguous`、`BroadcastView` 三种之一。

**offset 单位：** 本模块中所有 `offset` 字段一律是元素单位（element-count），不是字节单位。指针算术 `self.as_ptr().add(offset)` 对 `*const A` 调用 `add(n: usize)` 时，自动按 `size_of::<A>()` 字节换算，由 Rust 标准库 pointer 类型保证；本模块直接传 element offset 即可。该 `add` 调用必须位于已完成 shape-aware bounds 校验与 checked offset 算术验证之后的 unsafe block 中。

**SliceInfo 校验职责回顾：** `SliceInfo::new` 只做结构性校验（rank 一致、output 维度匹配 Range 计数、Range start≤end）；shape 边界校验（Range.end <= shape[axis]、Index < shape[axis]）由 `TensorBase::slice(info)` 在切片应用时完成，理由详见 §5.1 和决策 3。

切片布局标志规则：切片结果的 layout flags 根据新的 `shape` / `stride` 组合重新计算（调用 06-layout v1.3 `compute_layout_flags::<A, I>`）。若源视图带有 `BroadcastView`，且切片后仍存在任一零步长轴**并且** `product(shape) > 0`，则继续保留 `BroadcastView` flag；若切片导致 `product(shape) == 0`（空数组退化），即使存在 stride == 0 也**不**保留 `BroadcastView`，与 `15-broadcast.md §6.3` / `06-layout.md §5.12` 严格一致；其余情形按普通 F-order / non-contiguous 规则重分类。

**切片 offset 计算与空切片规则（v3.0.2）：** 切片应用时**必须**严格区分两个 offset 概念：

1. `slice_delta`（element 单位）：本次切片在每个轴上累加得到的相对偏移，由 `TensorBase::slice(info)` 在 bounds-check 通过后计算。`Index` 与 `Range { start, end }` **两类元素都贡献** `slice_delta`（见上方步骤 2a / 2b）：
   - `Index(idx)` 折轴：`slice_delta += idx * src_strides[i]`；
   - `Range { start, end }`：`slice_delta += start * src_strides[i]`（`start == 0` 时贡献为 0）；保留 `stride[i]`，更新输出 shape[axis] = `end - start`。
   形式化：`slice_delta = Σᵢ contribution_i * src_strides[i]`，其中 `contribution_i` 对 `Index(idx)` 取 `idx`、对 `Range { start, end }` 取 `start`。所有累加都使用 `checked_add` / `checked_mul` 防溢出。
2. `new_offset = src.offset.checked_add(slice_delta)?`：写回切片结果 `TensorBase::offset` 字段的绝对偏移（仍以 storage base 为基准）。**必须 checked**：尽管 `slice_delta` 的累加已使用 `checked_add` / `checked_mul`，最后从相对 delta 折回绝对 offset 时仍可能在 `usize::MAX` 边界溢出；溢出映射 `XenonError::InvalidLayout { operation: Cow::Borrowed("slice"), reason: InvalidLayoutReason::AccessRangeExceedsStorage, .. }`（R15 B-01 修复）。

`compute_layout_flags::<A, I>` 需要的是**逻辑首元素指针**（non-derefenceable 即可），不是绝对 offset；因此必须按结果 `len` 分支：

```rust,ignore
let logical_ptr: *const A = if result_len == 0 {
    // Empty slice: do NOT call src.as_ptr().add(slice_delta), the storage
    // base might already be NonNull::<A>::dangling() (per 07-tensor §6.2);
    // pass a well-defined non-dereferenceable sentinel instead.
    core::ptr::NonNull::<A>::dangling().as_ptr()
} else {
    // result_len > 0: slice_delta is a pure relative element-unit offset,
    // already verified by TensorBase::slice(info) to land within the
    // source's reachable storage range; src.as_ptr() already includes
    // src.offset (07-tensor §5.4), so adding slice_delta (NOT new_offset)
    // gives the result's logical-first pointer exactly once.
    // SAFETY: TensorBase::slice(info) verified bounds + checked-offset
    // arithmetic; result range [logical_ptr, logical_ptr + result_len)
    // lies within the source's storage.
    unsafe { src.as_ptr().add(slice_delta) }
};
let new_flags = layout::compute_layout_flags::<A, I>(&new_shape, &new_strides, logical_ptr);
```

**禁止**直接写 `src.as_ptr().add(new_offset)` —— `src.as_ptr()` 已经叠加过 `src.offset`，再 add `new_offset` 会双重 offset；且空切片场景下 storage base 可能是 dangling，对其做 add 算术违反 `07-tensor.md §6.2` 空张量规则。

### 6.4 安全性论证

| `unsafe` 点                           | 安全前提                             | 为什么仍然需要                     |
| ------------------------------------- | ------------------------------------ | ---------------------------------- |
| `NdIndex::index_unchecked`            | 调用方已保证 rank 匹配且每个分量有效 | 为内部已验证路径消除重复检查       |
| `get_unchecked` / `get_unchecked_mut` | 调用方已保证索引合法且可写性前提成立 | 为热点访问路径提供零额外分支的能力 |
| `src.as_ptr().add(slice_delta)`（仅当 `result_len > 0`） | `TensorBase::slice(info)` 已完成 shape-aware bounds 校验与 checked offset 算术验证；`src.as_ptr()` 已叠加 `src.offset`，再 add 的是相对 `slice_delta`（element 单位），等价于"叠加一次 offset"；空切片走 `NonNull::<A>::dangling()` 路径，不进入该 unsafe | 为布局 flags 重算提供切片后的逻辑首元素指针；详见 §6.3 切片 offset 计算与空切片规则 |

unsafe 变体只省略检查，不改变偏移量公式、shape/stride 解释或引用别名规则。若输入索引非法，责任由调用方承担；若输入合法，unsafe 与安全路径的结果必须一致。

---

## 7. 实现任务拆分

### Wave 1: 索引地址计算基础

- [ ] **T1**: 定义 `NdIndex<D>` 与 tuple / slice index 的合法性检查
  - 文件: `src/index/ndindex.rs`
  - 内容: rank 匹配、逐轴边界检查、checked / unchecked offset 计算
  - 测试: `test_try_at_2d`, `test_try_at_out_of_bounds`
  - 前置: `dimension`、`tensor` 基础能力已可用
  - 预计: 10 min

### Wave 2: 只读访问与切片描述基础

- [ ] **T2**: 实现 `try_at` / `get` / `get_unchecked`
  - 文件: `src/index/access.rs`
  - 内容: 统一安全与 unsafe 访问路径，保证错误边界一致
  - 测试: `test_get_returns_index_out_of_bounds`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 定义 `SliceInfoElem` 与 `SliceInfoIndices`
  - 文件: `src/index/slice.rs`
  - 内容: inline / dynamic 切片描述表示
  - 测试: `test_slice_basic`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 可写访问与视图构造

- [ ] **T3**: 实现 `try_at_mut` / `get_mut` / `get_unchecked_mut`
  - 文件: `src/index/access.rs`
  - 内容: 仅在 `StorageMut` trait 前提成立时暴露可写访问
  - 测试: `test_try_at_mut_requires_storage_mut`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现 `slice` 的 shape/stride 更新与布局重算
  - 文件: `src/index/slice.rs`
  - 内容: `Index` 折轴、`Range` 更新 shape/stride、只读视图返回
  - 测试: `test_slice_layout_recomputed`、`test_slice_chain`
  - 前置: T4
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置                           | 目的                                                             |
| ----------------------- | ------------------------------ | ---------------------------------------------------------------- |
| 单元测试                | `src/index/` 对应测试模块      | 验证单个访问/切片函数的语义与错误边界                            |
| 集成测试                | 索引与张量交互测试             | 验证 `tensor` + `index` + `error` 的组合行为                     |
| 边界测试                | 与单元/集成测试配套组织        | 覆盖 rank-0、广播视图、非连续切片、越界与 `10^7` 元素偏移边界    |
| 属性测试（按需）        | 索引模块测试目录               | 验证 offset 计算与 shape/stride 更新不变量                       |
| Feature gate / 配置测试 | 不适用                         | 当前模块不涉及 SIMD、并行或可选 feature                          |
| 类型边界 / 编译期测试   | trait 约束测试或编译期失败测试 | 验证 `usize` 仅用于元数据角色，索引 trait 不放宽到负数或元素类型 |

### 8.2 单元测试清单

| 测试函数                               | 测试内容                                       | 优先级 |
| -------------------------------------- | ---------------------------------------------- | ------ |
| `test_try_at_2d`                       | `try_at()` 成功返回二维张量元素引用            | 高     |
| `test_try_at_out_of_bounds`            | 越界返回 `IndexOutOfBounds`                    | 高     |
| `test_try_at_mut_requires_storage_mut` | 只在 `StorageMut` 前提成立时才存在可写访问入口 | 高     |
| `test_get_returns_index_out_of_bounds` | `get()` 失败返回 `IndexOutOfBounds`            | 高     |
| `test_slice_basic`                     | 基本切片结果的 shape 与数据正确                | 高     |
| `test_slice_chain`                     | 视图的视图保持一致的共享数据语义               | 中     |
| `test_slice_layout_recomputed`         | 切片后布局状态被重新计算                       | 高     |
| `test_slice_high_rank_ixdyn`           | `IxDyn` 高 rank 输入的切片元数据正确           | 中     |
| `test_slice_extreme_offset_checked`    | 大步长/大 shape 下偏移计算不溢出或返回错误     | 中     |
| `test_index_large_tensor_offset_boundary` | `10^7` 元素张量末元素索引成功，溢出偏移返回错误 | 高 |

### 8.3 边界测试场景

| 场景                                  | 预期行为                                                   |
| ------------------------------------- | ---------------------------------------------------------- |
| rank-0 张量索引                       | 仅接受零维合法索引形式，偏移为 0                           |
| 广播视图上的只读索引                  | 索引成功但结果仍遵循只读/共享只读语义                      |
| 非连续切片后的访问                    | 偏移量计算继续基于 stride，不假设连续                      |
| 任一轴越界                            | 安全接口返回 recoverable error                            |
| 高 rank（静态上限附近或 `IxDyn`）切片 | rank 校验、输出 shape 与 stride 更新保持正确               |
| `10^7` 元素张量 `[3162,3162]` 的末元素索引与极端 offset 组合 | 合法索引返回正确元素；会溢出的 offset 计算返回错误而非 panic |

### 8.4 属性测试不变量

| 不变量                                               | 测试方法                                            |
| ---------------------------------------------------- | --------------------------------------------------- |
| `checked_offset == unchecked_offset`（在合法输入上） | 随机生成合法 shape / strides / index 并比较两条路径 |
| `slice.len()` 与更新后的 shape 一致                  | 随机合法范围输入，验证逻辑元素数量守恒              |
| 切片保持逻辑顺序                                     | 对合法基础范围切片比较视图遍历序列与参考实现        |

### 8.5 集成测试

| 测试文件                | 测试内容                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_index.rs` | 索引 API 与 `tensor`、`dimension`、`layout`、`storage`、`error` 的端到端集成测试     |

### 8.6 Feature gate / 配置测试

| 配置     | 验证点               |
| -------- | -------------------- |
| 默认配置 | 索引模块语义完整可用 |
| SIMD     | 不适用               |
| 并行     | 不适用               |

### 8.7 类型边界与编译期测试

| 场景                               | 测试方式                        |
| ---------------------------------- | ------------------------------- |
| `usize` 不作为元素类型扩展索引语义 | 文档审查 + 类型约束测试         |
| 非 `usize` 负数索引不被接受        | 编译期失败测试或 trait 约束验证 |
| 非法 `NdIndex` 外部实现            | `Sealed` 约束测试               |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向         | 对方模块    | 接口/类型                              | 约定                                             |
| ------------ | ----------- | -------------------------------------- | ------------------------------------------------ |
| 消费（输入） | `tensor`    | `TensorBase<S, D>`                     | 索引前读取 shape、stride、offset 与存储模式      |
| 消费（输入） | `dimension` | `Dimension`                            | 用于 rank 与轴边界验证                           |
| 消费（输入） | `layout`    | `Strides<D>`, layout flags             | 偏移量解释与切片后布局重算                       |
| 消费（输入） | `storage`   | `Storage`, `StorageMut`                | 区分只读访问与可写访问的 trait 约束边界          |
| 产出（输出） | `tensor`    | `&A`, `&mut A`, `TensorView<'a, A, I>` | 返回值生命周期绑定到源张量；切片结果共享底层数据 |
| 产出（输出） | `error`     | `XenonError`                           | 安全路径对外暴露统一错误类型                     |

### 9.2 数据流描述

```text
User calls tensor.try_at(index)
    │
    ├── index/ validates rank and bounds
    ├── index/ computes offset from shape + strides
    └── tensor/storage returns shared or mutable reference

User calls tensor.slice(info)
    │
    ├── index/ validates each SliceInfoElem
    ├── index/ updates offset and shape (strides per-axis values unchanged; Index removes axis + stride slot, Range keeps stride)
    ├── index/ recomputes layout flags
    └── tensor returns read-only TensorView sharing source data
```

---

## 10. 错误处理与语义边界

| 主题              | 说明                                                                                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | `try_at()` / `get()` / `slice()` 在 rank 不匹配、轴非法、越界时返回 `XenonError`。其中索引长度与张量 `ndim` 不匹配时，错误类型固定为 `XenonError::DimensionMismatch { operation, expected, actual }`；多维索引越界使用 `XenonError::IndexOutOfBounds { operation, attempted_index, axis, shape }`；`slice()` 的 Range 越界使用 `InvalidArgument { kind: InvalidArgumentKind::RangeOutOfBounds { axis, axis_len, start, end } }`；`SliceInfo::new` 的 Range start>end 使用 `InvalidArgument { kind: InvalidArgumentKind::RangeStartAfterEnd { axis, start, end } }`；offset 算术溢出使用 `InvalidLayout { reason: InvalidLayoutReason::AccessRangeExceedsStorage, .. }`。所有字段对齐 26-error v3.0.0 §5.1 封闭枚举。 |
| Trait-bound 边界  | `try_at_mut()` / `get_mut()` / `get_unchecked_mut()` 仅在 `S: StorageMut` 前提成立时存在；不再为“只读存储上的可写索引”设计运行时 `InvalidStorageMode` 分支           |
| Panic             | `std::ops::Index` 与 `std::ops::IndexMut` 不在 Xenon 稳定 API 中实现（见 §3 范围约束）。规范安全主路径是返回 `Result` 的 checked API                                                                                                                  |
| 路径一致性        | 对同一合法输入，checked 与 unchecked 路径必须给出同一偏移和同一逻辑结果；unsafe 只省略检查                                                                           |
| 容差边界          | 不适用；本模块不涉及浮点容差、SIMD 误差或并行归约差异                                                                                                                |

---

## 11. 设计决策记录

### 决策 1: 安全主路径使用 recoverable error

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | `try_at()` / `get()` / `try_at_mut()` / `get_mut()` / `slice()` 作为规范安全接口，失败返回可恢复错误 |
| 理由     | 符合 `需求说明书 §18` 对安全接口的要求，并与 `26-error.md` 的统一诊断模型对齐                        |
| 替代方案 | 全部使用 `Index` / `IndexMut` panic 语法糖 — 放弃，错误恢复与上游组合能力不足                        |
| 替代方案 | 统一返回 `Option` — 放弃，无法承载轴、shape、索引等诊断信息                                          |

### 决策 2: 切片结果保持共享只读语义

| 属性     | 值                                                                |
| -------- | ----------------------------------------------------------------- |
| 决策     | 范围索引返回共享底层数据的只读或共享只读视图，不提供共享可写视图  |
| 理由     | 符合 `需求说明书 §18` 与 `需求说明书 §6`，对共享数据结果收敛到可验证的只读语义 |
| 替代方案 | 允许共享可写切片 — 放弃，超出当前版本范围且引入别名写入风险       |
| 替代方案 | 切片总是复制生成独立张量 — 放弃，会破坏零拷贝视图语义并扩大成本   |

### 决策 3: SliceInfo::new 仅做结构性校验，shape 边界校验下沉到 TensorBase::slice (B9.a)

| 属性     | 值                                                                                                                                                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `SliceInfo::new(indices, in_dim, out_dim)` 只校验结构性约束（rank 一致、output 维度匹配 Range 计数、Range 内 start ≤ end）；shape 边界校验（Range.end ≤ shape[axis]、Index < shape[axis]）由 `TensorBase::slice(info)` 在应用时完成 |
| 理由     | (1) `SliceInfo::new` 只接收维度类型 `D`，不携带具体 shape 值，根本无法验证 shape 边界；(2) 把校验下沉让同一个 `SliceInfo` 实例可在不同 shape 但 rank 相同的张量上复用；(3) 错误来源更清晰：结构错 → SliceInfo 构造期失败；shape 错 → slice 应用期失败 |
| 替代方案 | 在 `SliceInfo::new` 强制要求传入 shape 一并校验                                                                                                                                                                              |
| 拒绝原因 | 会让 `SliceInfo` 与具体张量 shape 强耦合，丢失"切片描述符可在不同张量上复用"的能力，并把构造器签名复杂化（`new(indices, in_dim, out_dim, shape)`）                                                                          |
| 替代方案 | 不在 SliceInfo 做任何校验，全部下沉到 slice                                                                                                                                                                                  |
| 拒绝原因 | 结构性约束（rank 一致、output 维度匹配）在没有 shape 也能校验，下沉会让显然非法的 SliceInfo 在构造期就溜过去，错误诊断时机过晚                                                                                                |

---

## 12. 性能考量

### 12.1 复杂度

| 操作                            | 时间复杂度 | 空间复杂度           |
| ------------------------------- | ---------- | -------------------- |
| `try_at` / `get` / `try_at_mut` | O(rank)    | O(1)                 |
| `get_unchecked*`                | O(rank)    | O(1)                 |
| `slice`                         | O(rank)    | O(1)（仅视图元数据） |

### 12.2 内存与缓存行为

| 场景                  | 行为                                             |
| --------------------- | ------------------------------------------------ |
| 连续 F-order 张量索引 | 偏移量计算后访问目标元素，缓存局部性由原布局保证 |
| 非连续视图索引        | 仍可正确访问，但缓存友好性取决于 stride 跳跃模式 |
| 范围切片              | 仅重建视图元数据并共享源数据，不复制元素         |

### 12.3 性能边界说明

- 偏移量计算成本与 rank 成正比，而非与元素总数成正比。
- 切片为元数据级操作；性能关键点在后续对视图的消费，而非视图创建本身。
- unsafe 变体的价值仅在于消除重复检查，不意味着不同的逻辑结果。

---

## 13. 平台与工程约束

| 约束       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| `std` only | 当前项目基线为 `std` 环境；本文不扩展 `no_std` 路径          |
| MSRV       | Rust 1.85+                                                   |
| 单 crate   | 索引设计保持在现有 crate 内，不引入额外 crate 或拆分子包     |
| SemVer     | 文档仅把旧结构重写为标准模板，不新增索引能力或改变已承诺语义 |
| 最小依赖   | 不新增第三方依赖；继续复用仓库既有模块                       |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-14 |
| 1.0.1 | 2026-04-14 |
| 1.0.2 | 2026-04-15 |
| 1.0.3 | 2026-04-15 |
| 1.0.4 | 2026-04-16 |
| 1.0.5 | 2026-04-16 |
| 1.0.6 | 2026-04-16 |
| 1.0.7 | 2026-04-16 |
| 2.0.0 | 2026-05-02 |
| 2.0.1 | 2026-05-03 |
| 3.0.0 | 2026-05-03 |
| 3.0.1 | 2026-05-04 |
| 3.0.2 | 2026-05-04 |

### v3.0.2 (2026-05-04) — R11/R12 切片 SAFETY 强化（slice_delta + empty dangling）

- §6.3 / §6.4：明确切片 offset 计算规则。`slice_delta`（element 单位的相对偏移）与 `new_offset = src.offset + slice_delta`（写回 `TensorBase.offset` 字段的绝对偏移）必须**严格区分**。`compute_layout_flags` 需要逻辑首元素指针（non-dereferenceable 即可）：空切片（`product(new_shape) == 0`）走 `NonNull::<A>::dangling().as_ptr()`，与 `07-tensor.md §6.2` 空张量规则一致；非空切片只 add `slice_delta`（**不**add `new_offset`，否则 double-apply offset）。§6.3 主算法步骤 3 同步重写。
- §6.4 表行同步更新为 `src.as_ptr().add(slice_delta)`（仅当 `result_len > 0`）。
- 与 `00-coding.md §1.3` / `28-tests.md §1.0` 锁定基线 v3.0.2 对齐（R12 B-03 同步：锁定基线从 v3.0.1 升至 v3.0.2，反映 R11 B-04 实际落地）。

### v3.0.1 (2026-05-04) — R8/R9 协同基线对齐

- 与 `00-coding.md §1.3` / `28-tests.md §1.0` 锁定基线版本号显式对齐；本版无契约变更，仅同步 changelog 行避免 R8 升版后的版本号漂移（R9 评审 B-05 修复）。

### v3.0.0 (2026-05-03) — Indexing 公开入口收敛到 `try_at` / `try_at_mut`

- 将公开安全 indexing 入口规范化为 `try_at(&self, index)` / `try_at_mut(&mut self, index)`，与 `00-coding §1.3` / `26-error §5.1` 锁定的 `IndexOutOfBounds` 错误约定一致。
- `get(&[usize])` / `get_mut(&[usize])` 保留为 `&[usize]` convenience wrapper，文档说明其与 `try_at` 的等价路径。
- 与 `25-safety.md §5.12` 内部 unsafe fn 索引表保持入口名一致（不暴露 unchecked 公开变体）。

### v2.0.1 (2026-05-03) — Medium/Low 文档修复

- §5.2：明确 `try_at()` / `try_at_mut()` 是规范主入口，`get()` / `get_mut()` 仅为 `&[usize]` convenience wrapper。
- §5.3：移除 `expect("index already validated")` 示例，改为继续通过 `?` 传播错误。
- §6.3 / §6.4：将 `self.as_ptr().add(new_offset_in_elements)` 明确纳入已校验后的 unsafe block 与安全性论证。**历史标注（pre-v3.0.2 术语）**：当时使用的内联指针算术 `new_offset_in_elements` 是 absolute offset；自 v3.0.2 起重构为 `slice_delta`（相对元素偏移）+ `new_offset = self.offset.checked_add(slice_delta)?`（绝对偏移）严格区分，并要求空切片走 `NonNull::<A>::dangling().as_ptr()`。详见 v3.0.2 changelog 与现行 §6.3 / §6.4。
- §8.3：删除越界场景中关于非规范 panic 语法糖的残留表述。

### v2.0.0 (2026-05-02) — 校验职责重新分配 + 错误字段对齐

> 本版本是与用户决策 B9.a + 26-error v3.0.0 + 05-storage v2.0.0 + 06-layout v1.3 协同的破坏性内部更新。`SliceInfo::new` 的校验范围**收窄**（结构性校验），shape 边界校验**下沉**到 `TensorBase::slice(info)`；该变更是 `SliceInfo::new` 的语义破坏（部分原本失败的非法构造现在会在 `slice` 时才失败，但同样可恢复）。

**契约更新（B9.a 用户已批准）**：

- §5.1 / §5.2 / §6.3 / §11 决策 3：`SliceInfo::new` 仅做结构性校验（rank 一致、output 维度匹配 Range 计数、Range 内 start ≤ end）；shape 边界校验下沉到 `TensorBase::slice(info)`。
- §5.1 SliceInfo::new doc comment 重写：列出 SliceInfo::new 实际执行的 3 类结构性校验；明确 shape 边界校验 deferred 到 slice。
- §5.1 校验职责分工表：清楚区分 SliceInfo::new 与 TensorBase::slice 的校验范围。
- §11 新增决策 3：完整论证 B9.a 决策的理由与拒绝替代方案的原因。

**协同与一致性更新**：

- §5.1 Inline / Dynamic 选择规则：明确 `SliceInfoIndices::Inline` 覆盖 `Ix0..Ix6`；`IxDyn` 且 `indices.len() > 6` 必须使用 `Dynamic`。
- §5.2 `get(&[usize])` 委托语义澄清：不再"转 IxDyn 后委托 try_at"（此路径会触发 trait bound 不满足），改为独立实现共用 `compute_offset`，但分派路径分离；新增段落说明理由。
- §6.3 切片元数据更新算法重写：明确 SliceInfo::new 已通过结构性校验；列出 shape 边界 + 算术溢出的具体错误变体（`IndexOutOfBounds`、`InvalidArgumentKind::RangeOutOfBounds`、`InvalidLayoutReason::AccessRangeExceedsStorage`）；明确 offset 是元素单位（修复 §6.3 之前 offset 单位不清问题）。
- §6.3 协同引用 `05-storage.md v2.0.0 §5.11.1` 的统一规则（广播 / 转置 / 切片产出 ViewRepr）。
- §6.3 layout flags 重算引用 06-layout v1.3 `compute_layout_flags::<A, I>`。
- §10 错误处理表：列出 5 个错误变体的完整封闭枚举字段，对齐 26-error v3.0.0 §5.1。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
