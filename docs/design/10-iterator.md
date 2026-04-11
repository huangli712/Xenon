# 迭代器模块设计

> 文档编号: 10 | 模块: `src/iter/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`
> 需求参考: 需求说明书 §11

---

## 1. 模块概述

### 1.1 定位

迭代器模块是 Xenon 张量库的数据遍历基础设施，为所有张量操作提供统一、高效、类型安全的遍历机制。支持从简单元素遍历到多张量同步迭代的多种迭代模式。

### 1.2 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 元素遍历 | 按元素遍历（F-order 内存顺序） | 具体运算逻辑（参见 `11-math.md §1`） |
| 轴遍历 | 沿指定轴产生子张量视图 | 子视图的具体操作 |
| 窗口遍历 | 滑动窗口产生子视图 | 卷积运算 |
| 索引遍历 | 带多维索引的元素遍历 | 索引赋值操作 |
| 同步遍历 | 多张量同步遍历（Zip） | 广播规则计算（参见 `15-broadcast.md §3`） |
| 并行迭代 | — | 并行迭代（参见 `09-parallel.md §4.5`） |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (仅依赖 core/alloc，不依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: iter  ← 当前模块
```

### 1.4 设计原则

| 原则 | 体现 |
|------|------|
| 步长感知 | 所有迭代器正确处理非连续内存和负步长 |
| 零拷贝 | 轴迭代器和窗口迭代器产生子视图，不拷贝数据 |
| trait 一致性 | 所有迭代器实现 `Iterator` + `ExactSizeIterator` |
| 空数组安全 | 空数组迭代立即结束，零维张量元素迭代产出恰好 1 个元素 |
| 广播只读 | 禁止对广播结果进行可变迭代 |

---

## 2. 文件位置

```
src/iter/
├── mod.rs         # 模块入口、公开导出、迭代器 trait 声明
├── elements.rs    # Elements / ElementsMut 扁平元素遍历
├── axis.rs        # AxisIter / AxisIterMut 沿轴迭代
├── windows.rs     # Windows 滑动窗口迭代
├── indexed.rs     # IndexedIter / IndexedIterMut 带索引遍历
├── zip.rs         # Zip 多张量同步迭代
```

单文件划分理由：各迭代器类型之间独立，拆分文件降低单文件复杂度，便于并行开发。`lanes.rs` 暂不纳入当前版本。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/iter/
├── crate::tensor        # TensorBase<S, D>, TensorView, TensorViewMut
├── crate::dimension     # Dimension trait, Ix0~Ix6, IxDyn
├── crate::storage       # Storage, StorageMut trait
├── crate::tensor        # 通过 TensorBase 暴露布局/连续性查询
└── crate::broadcast     # broadcast_shape()（仅 Zip 使用）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `TensorView<'a, A, D>`, `TensorViewMut<'a, A, D>`, `.shape()`, `.strides()`, `.as_ptr()`, `.len()`（参见 `07-tensor.md §4`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `RemoveAxis`, `D::Smaller`（参见 `02-dimension.md §4`） |
| `storage` | `Storage<Elem = A>`, `StorageMut<Elem = A>`, `Owned<A>`（参见 `05-storage.md §4`） |
| `tensor` | `.is_f_contiguous()`, 布局标志查询（参见 `07-tensor.md §4`） |
| `broadcast` | `broadcast_shape()`（Zip 构造时验证形状兼容性，参见 `15-broadcast.md §4`） |

### 3.3 依赖方向

> **依赖方向：单向向上。** `iter/` 仅消费 `tensor`、`dimension`、`storage` 等核心模块，不被它们依赖。布局/连续性判断通过 `TensorBase` 暴露的查询接口完成。

---

## 4. 公共 API 设计

> **DoubleEndedIterator 说明：** 当前版本所有迭代器不实现 `DoubleEndedIterator`，因为 F-order 遍历的反向语义在高维情况下不明确，且需求未要求此功能。

### 4.1 Elements 迭代器

```rust
/// Flat element iterator, traverses all elements in F-order memory layout.
pub struct Elements<'a, A, D: Dimension> {
    // Internal fields: view, pointer/index state, remaining count
}

/// Mutable flat element iterator.
pub struct ElementsMut<'a, A, D: Dimension> {
    // Internal fields: mutable view, pointer/index state, remaining count
}

impl<'a, A, D: Dimension> Iterator for Elements<'a, A, D> {
    type Item = &'a A;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for Elements<'a, A, D> {}

impl<'a, A, D: Dimension> Iterator for ElementsMut<'a, A, D> {
    type Item = &'a mut A;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for ElementsMut<'a, A, D> {}
```

### 4.2 AxisIter 沿轴迭代器

```rust
/// Axis iterator, yields a sub-tensor view with reduced dimension each step.
///
/// Input dimension N, output dimension N-1 views.
pub struct AxisIter<'a, A, D: Dimension + RemoveAxis> {
    // Internal fields: view, axis, current position, length
}

/// Mutable axis iterator.
pub struct AxisIterMut<'a, A, D: Dimension + RemoveAxis> {}

impl<'a, A, D: Dimension + RemoveAxis> Iterator for AxisIter<'a, A, D> {
    type Item = TensorView<'a, A, D::Smaller>;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension + RemoveAxis> ExactSizeIterator for AxisIter<'a, A, D> {}

impl<'a, A, D: Dimension + RemoveAxis> Iterator for AxisIterMut<'a, A, D> {
    type Item = TensorViewMut<'a, A, D::Smaller>;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension + RemoveAxis> ExactSizeIterator for AxisIterMut<'a, A, D> {}
```

> **设计决策：** `AxisIter` 的 `Item` 类型为 `TensorView<'a, A, D::Smaller>`。这要求 `D` 满足 `RemoveAxis`，并由 `RemoveAxis` trait 提供 `type Smaller: Dimension` 关联类型。零维张量（Ix0）不支持按轴遍历，因为 `Ix0` 不实现 `RemoveAxis`。

### 4.3 Windows 滑动窗口迭代器

```rust
/// Sliding window iterator, slides a window of specified size across the tensor,
/// yielding a view for each window position.
///
/// Window count = product(shape[i] - window_size[i] + 1).
/// Yields zero windows when the window is larger than the array or the array is empty.
pub struct Windows<'a, A, D: Dimension> {
    // Internal fields: view, window size, current index, remaining count
}

impl<'a, A, D: Dimension> Iterator for Windows<'a, A, D> {
    type Item = TensorView<'a, A, D>;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for Windows<'a, A, D> {}
```

### 4.4 IndexedIter 带索引迭代器

```rust
/// Element iterator with multi-dimensional indices.
///
/// Yields (D::Slice, &'a A) tuples, indices increment in F-order.
pub struct IndexedIter<'a, A, D: Dimension> {
    // Internal fields: Elements iterator, current index, stride state machine
}

pub struct IndexedIterMut<'a, A, D: Dimension> {}

impl<'a, A, D: Dimension> Iterator for IndexedIter<'a, A, D> {
    type Item = (D::Slice, &'a A);
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIter<'a, A, D> {}

impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
    type Item = (D::Slice, &'a mut A);
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIterMut<'a, A, D> {}
```

### 4.5 Zip 多张量同步迭代器

```rust
/// Trait for types that can be iterated in parallel with Zip.
pub trait NdProducer {
    type Dim: Dimension;
    type Item;

    /// Returns the shape of this producer.
    fn shape(&self) -> &Self::Dim;

    /// Returns the strides of this producer.
    fn strides(&self) -> &Strides<Self::Dim>;

    /// Split at the given logical flat index.
    fn split_at(self, index: usize) -> (Self, Self);
}

/// Multi-tensor synchronized iterator, used for zip_with and similar operations.
///
/// `Zip` itself only combines producers that already share the same dimension
/// type `D`. For cross-dimension broadcasting (for example `Ix2` with `Ix3`),
/// upper layers must first normalize both inputs to the common broadcasted
/// dimension and then construct `Zip`. Broadcast results are read-only; mutable
/// iteration is not supported.
pub struct Zip<Parts, D: Dimension> {
    // Internal fields: producer tuple, broadcasted shape, stride info
}

impl<D: Dimension> Zip<(), D> {
    /// Create a Zip from the first tensor.
    pub fn from<P: NdProducer<Dim = D>>(producer: P) -> Zip<(P,), D>;
}

impl<Parts, D: Dimension> Zip<Parts, D> {
    /// Add another tensor to the Zip after shape compatibility has already been
    /// normalized to the same dimension type.
    ///
    /// **注意:** `Zip::and` 要求两个 producer 的维度类型 `D` 必须相同。
    /// 对于不同维度类型的广播场景（如 Ix2 与 Ix3 组合），应使用
    /// `11-math.md` 中的 `zip_with` 函数，后者通过
    /// `BroadcastDim` 约束处理维度不同的情况。
    pub fn and<P: NdProducer<Dim = D>>(self, producer: P)
        -> Result<Zip<(Parts, P), D>, XenonError>;

    /// Execute a closure for each element.
    pub fn for_each<F>(self, f: F)
    where
        F: FnMut(Parts::Item);

    /// Apply a function and collect the results into a new tensor.
    pub fn map_collect<F, A>(self, f: F) -> Tensor<A, D>
    where
        F: FnMut(Parts::Item) -> A,
        A: Element;
}
```

### 4.6 LaneIter 延期到后续版本

`LaneIter` / `LaneIterMut` 不属于需求说明书 §11 的必须项。当前版本先不设计该 API，避免与 `AxisIter`、`Windows`、`IndexedIter` 的职责边界重叠；如后续引入，应在独立文档中重新定义 1D 产出语义、轴方向约定与可变别名规则。

### 4.7 TensorBase 上的迭代器入口方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Element iterator (immutable).
    pub fn iter(&self) -> Elements<'_, A, D>;

    /// Indexed iterator (immutable).
    pub fn indexed_iter(&self) -> IndexedIter<'_, A, D>;

    /// Iterate along an axis (yields sub-tensors orthogonal to the axis).
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, A, D>
    where
        D: RemoveAxis;

    /// Sliding window iterator.
    ///
    /// **注意：** 零维张量（`Ix0`）不支持 `windows()` 操作，因为该方法要求
    /// `D: RemoveAxis`，而 `Ix0` 未实现该 trait。
    pub fn windows(&self, size: impl IntoDimension<Dim = D>) -> Option<Windows<'_, A, D>>;

}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Mutable element iterator.
    pub fn iter_mut(&mut self) -> ElementsMut<'_, A, D>;

    /// Mutable indexed iterator.
    pub fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, A, D>;

    /// Mutable axis iteration.
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, A, D>
    where
        D: RemoveAxis;

}
```

### 4.8 Good / Bad 对比示例

```rust
// Good - safely iterate elements using iter()
let tensor = Tensor::<f64, Ix2>::zeros([3, 4]);
for &elem in tensor.iter() {
    println!("{}", elem);
}
assert_eq!(tensor.iter().count(), 12);

// Good - use ExactSizeIterator to get precise length
let iter = tensor.iter();
assert_eq!(iter.len(), 12);

// Bad - manual index traversal (poor performance, may go out of bounds on non-contiguous data)
for i in 0..tensor.shape()[0] {
    for j in 0..tensor.shape()[1] {
        let _ = tensor[[i, j]];  // not recommended
    }
}

// Bad - calling iter_mut() on a broadcast result
let broadcast_view = tensor.view().broadcast_to([3, 4])?;
// broadcast_view.iter_mut();  // compile error: broadcast view is immutable
```

---

## 5. 内部实现设计

### 5.1 Elements 快速/慢速路径选择

```
Elements::new(view):
    if view.is_f_contiguous():
        // Fast path: 指针递增
        ptr = view.as_ptr()
        end = ptr + view.len()
    else:
        // Slow path: 步长状态机
        stride_state = StrideState::new(&view)
        index = D::zeros(view.ndim())
```

### 5.2 步长状态机（StrideState）

```
increment_index_f(shape, index):
    for i in 0..ndim:
        index[i] += 1
        if index[i] < shape[i]:
            return  // no carry
        index[i] = 0  // carry to next dimension
```

> **负步长偏移量计算：** 对于负步长，元素偏移量计算为有符号加法：`offset = base_offset + Σ(stride[i] * index[i])`，其中 `stride[i]` 为 `isize`（可为负）。当元素地址 = `base_ptr + offset` 时，负步长意味着向低地址方向遍历。
>
> **负步长下的索引递增：** 当 `stride[i] < 0`（负步长，来源于带负步长的切片操作）时，逻辑索引从 `0..n` 递增的方式不变，但物理偏移量的变化方向相反。具体而言：
> ```
> // When stride < 0 (negative stride, from slicing with negative step):
> // To iterate from logical index 0..n, physical offset goes from
> // (dim_len-1)*|stride| down to 0.
> // Therefore, the first logical element maps to the highest physical offset
> // in that dimension, and advancing the logical index decrements the physical offset.
> // increment_index_f should account for sign of stride when computing the
> // pointer to dereference, but the index increment logic itself remains the same.
> ```

### 5.3 广播可变迭代禁止

```rust
// SAFETY: broadcast_to() returns a TensorView with zero-stride dimensions,
// multiple logical indices map to the same physical address; mutable writes
// would create immediate mutable aliasing.
// Therefore broadcast_to() only returns an immutable view.
```

> **编译期防护机制：** 广播结果返回 `TensorView`（不可变视图），而非 `TensorViewMut`。由于 `TensorView` 不提供 `iter_mut()` 方法（`iter_mut()` 要求 `StorageMut` 约束，仅 `TensorViewMut` 和 `Tensor` 满足），对广播结果调用 `iter_mut()` 会在编译期被类型系统拒绝，无需运行时检查。参见 `07-tensor.md §4.7` 中视图方法的约束差异。

### 5.4 填充数组迭代

填充数组的迭代仅遍历逻辑元素。迭代器通过 shape 中的逻辑维度计数，跳过填充区域。

### 5.5 `ElementsProducerRange` — 并行分块内部状态

> **用途：** 供并行后端（`09-parallel.md §4.5`）中的 `ElementsProducer::split_at` 调用。它不再尝试把任意逻辑前缀/后缀表示成两个 `TensorView`，而是保留原始 view，并以扁平逻辑区间 `[start, end)` 描述生产范围。

```rust
pub(crate) struct ElementsProducerRange<'a, A, D> {
    base: TensorView<'a, A, D>,
    start: usize,
    end: usize,
}

impl<'a, A, D> ElementsProducerRange<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    pub(crate) fn from_view(base: TensorView<'a, A, D>) -> Self {
        let len = base.len();
        Self { base, start: 0, end: len }
    }

    pub(crate) fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.end - self.start, "split index out of bounds");
        let mid = self.start + index;
        (
            Self { base: self.base.clone(), start: self.start, end: mid },
            Self { base: self.base, start: mid, end: self.end },
        )
    }
}
```

**设计要点：**

- 不再要求任意逻辑区间可被表达成单个规则 shape/stride 的 `TensorView`。
- 连续与非连续视图统一通过“原 view + 逻辑区间”建模。
- 连续 F-order 视图仍可在 producer 内走更快的指针递增路径；非连续视图则通过逻辑索引到物理偏移的映射按需求址。
- 该内部状态仅供 `09-parallel.md` 中的 Producer 体系使用，不对外暴露。

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/iter/mod.rs` 骨架
  - 文件: `src/iter/mod.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: `07-tensor.md` 完成
  - 预计: 5 min

- [ ] **T2**: 实现 StrideState 步长状态机
  - 文件: `src/iter/elements.rs`（内部辅助结构）
  - 内容: F-order 索引递增逻辑
  - 测试: `test_stride_state_increment`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 核心迭代器

- [ ] **T3**: 实现 `Elements` / `ElementsMut`
  - 文件: `src/iter/elements.rs`
  - 内容: `Iterator` + `ExactSizeIterator` 实现，含快速/慢速路径
  - 测试: `test_elements_contig`, `test_elements_non_contiguous`, `test_elements_empty`, `test_elements_ix0`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `AxisIter` / `AxisIterMut`
  - 文件: `src/iter/axis.rs`
  - 内容: 沿轴迭代，产出子视图
  - 测试: `test_axis_iter_count`, `test_axis_iter_shape`, `test_axis_iter_empty_axis`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现 `Windows`
  - 文件: `src/iter/windows.rs`
  - 内容: 只读滑动窗口迭代，窗口大小验证
  - 测试: `test_windows_count`, `test_windows_too_large`, `test_windows_empty`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 高级迭代器

- [ ] **T6**: 实现 `IndexedIter` / `IndexedIterMut`
  - 文件: `src/iter/indexed.rs`
  - 内容: 基于 Elements 的索引包装
  - 测试: `test_indexed_iter_order`, `test_indexed_iter_ix0`
  - 前置: T3
  - 预计: 10 min

- [ ] **T7**: 实现 `Zip` 同步迭代
  - 文件: `src/iter/zip.rs`
  - 内容: NdProducer trait、Zip 构造与组合、`for_each` / `map_collect`
  - 测试: `test_zip_two_tensors`, `test_zip_broadcast`, `test_zip_for_each`
  - 前置: T3, T4, broadcast 模块
  - 预计: 10 min

### Wave 4: TensorBase 入口集成

- [ ] **T9**: 在 TensorBase 上添加迭代器入口方法
  - 文件: `src/tensor/`（或 `src/iter/mod.rs` 通过 trait extension）
  - 内容: `iter()`, `iter_mut()`, `axis_iter()`, `windows()`, `indexed_iter()` 等
  - 测试: `test_tensor_iter_integration`
  - 前置: T3, T4, T5, T6, T7
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] [T2]
           │     │
Wave 2: [T3] [T4] [T5]
           │     │     │
Wave 3: [T6] ──── [T7]
                    │
Wave 4:         [T9]
```

---

## 7. 测试计划

### 7.1 测试分类总表

| 测试分类 | 说明 | 包含的测试 |
|----------|------|-----------|
| 单元测试 | 验证单个迭代器类型的基本功能 | `test_elements_f_contig`, `test_elements_non_contiguous`, `test_elements_empty`, `test_elements_ix0`, `test_elements_mut_write`, `test_axis_iter_count`, `test_axis_iter_shape`, `test_windows_count`, `test_windows_too_large`, `test_windows_empty`, `test_indexed_iter_order`, `test_indexed_iter_ix0`, `test_zip_two_tensors`, `test_zip_broadcast`, `test_zip_broadcast_readonly` |
| 集成测试 | 验证迭代器与 TensorBase 入口方法的集成 | `test_tensor_iter_integration` |
| 边界测试 | 空数组、零维张量、非连续内存等边界条件 | `test_elements_empty`, `test_elements_ix0`, `test_windows_too_large`, `test_windows_empty`（详见 §7.2） |
| 属性测试 | 通过随机输入验证不变量 | `iter().count() == tensor.len()`, `axis_iter(Axis(i)).count() == shape[i]`, `ExactSizeIterator` 递减不变量（详见 §7.3） |

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_elements_f_contig` | F-order 连续数组：迭代顺序与物理布局一致 | 高 |
| `test_elements_non_contiguous` | 非连续数组（切片后）：步长跳转正确 | 高 |
| `test_elements_empty` | 空数组 `shape=[0, 3]`：`iter()` 立即结束 | 高 |
| `test_elements_ix0` | 零维张量：`iter()` 产出恰好 1 个元素 | 高 |
| `test_elements_mut_write` | `iter_mut()` 写入后数据正确 | 中 |
| `test_axis_iter_count` | `axis_iter(Axis(0)).count() == shape[0]` | 高 |
| `test_axis_iter_shape` | 沿轴迭代产出的子视图形状正确 | 高 |
| `test_windows_count` | 窗口数 = `product(shape - window + 1)` | 高 |
| `test_windows_too_large` | 窗口大于数组返回 `None` | 中 |
| `test_windows_empty` | 空数组返回长度为 0 的窗口迭代器 | 中 |
| `test_indexed_iter_order` | 索引按 F-order 递增 | 高 |
| `test_indexed_iter_ix0` | 零维张量索引为空切片 | 中 |
| `test_zip_two_tensors` | Zip 同步遍历两个同形状张量 | 高 |
| `test_zip_broadcast` | Zip 广播形状兼容 | 高 |
| `test_zip_broadcast_readonly` | 广播结果不可变迭代 | 中 |
| `test_padded_iter` | 填充数组仅遍历逻辑元素 | 低 |

### 7.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `iter()` 立即结束，`count() == 0` |
| 单元素 `shape=[1, 1]` | `iter()` 产出 1 项 |
| 零维张量 Ix0 | `iter()` 产出 1 项，`axis_iter()` 在类型层面不可调用 |
| 非连续切片 `s![.., 0..3]` | `iter()` 正确处理步长跳转 |
| 负步长（反转切片） | `iter()` 正确处理负步长 |
| 广播视图 `shape=[1, 4]` | `iter()` 遍历逻辑元素，`iter_mut()` 编译拒绝 |
| 填充数组 | 仅遍历逻辑元素 |

### 7.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `iter().count() == tensor.len()` | 随机形状 `[0..10, 0..10, 0..10]` |
| `axis_iter(Axis(i)).count() == shape[i]` | 随机形状 |
| `iter().len()` 每次调用后递减 | 迭代过程中检查 `ExactSizeIterator` |

### 7.5 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_iterator.rs` | `tensor.iter()` / `axis_iter()` / `windows()` / `zip()` 与 `tensor`、`broadcast`、`shape` 模块的协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

```rust
// tensor module provides iterator entry methods（参见 07-tensor.md §4.6）
// iter module consumes TensorView / TensorViewMut
```

### 8.2 数据流描述

```text
用户调用 tensor.iter() / axis_iter() / zip()
    │
    ├── tensor 模块提供 TensorView / TensorViewMut 入口
    ├── iter 模块根据 shape + strides 构造迭代器状态
    ├── 若是 zip/broadcast 路径，则先由 broadcast 校验公共形状
    └── 逐步产出元素 / 子视图，供 math / reduction / overload 消费
```

### 8.3 与 storage / dimension 模块

```rust
// Iterators read data via the Storage trait（参见 05-storage.md §4）
// Index state is managed via the Dimension trait（参见 02-dimension.md §4）
```

### 8.4 与 broadcast 模块

```rust
// Upper layers normalize broadcast compatibility before constructing Zip（参见 15-broadcast.md §4）
// Zero strides in broadcast views are handled correctly by the iter module
```

---

## 9. 设计决策记录（ADR）

### 决策 1：F-order 遍历顺序

| 属性 | 值 |
|------|-----|
| 决策 | 所有元素迭代器默认按 F-order（列优先）遍历 |
| 理由 | Xenon 只支持 F-order 布局，F-order 遍历对连续数组产生顺序内存访问，缓存友好性最优 |
| 替代方案 | 提供遍历顺序参数（F/C） |
| 替代方案 | 默认 C-order |
| 拒绝原因 | 项目范围仅 F-order，增加 C-order 选项增加复杂度且无实际收益 |

### 决策 2：ExactSizeIterator 要求

| 属性 | 值 |
|------|-----|
| 决策 | 所有迭代器须实现 `ExactSizeIterator` |
| 理由 | 提供精确的元素数量信息，调用者可预分配缓冲区、验证迭代完整性；`size_hint()` 的上界和下界始终相等 |
| 替代方案 | 仅实现 `Iterator`，不保证精确长度 |
| 拒绝原因 | 归约、zip_with 等操作需要精确长度进行预分配和并行分块 |

### 决策 3：零维张量元素迭代产出 1 个元素

| 属性 | 值 |
|------|-----|
| 决策 | Ix0 张量的 `iter()` 产出恰好 1 个元素 |
| 理由 | 零维张量在数学上表示标量，恰好有 1 个值；与 ndarray 行为一致；`len() == 1` |
| 替代方案 | 迭代产出 0 个元素 |
| 拒绝原因 | 与 `len()` 不一致，违反 `iter().count() == len()` 不变量 |

### 决策 4：广播结果不可变迭代

| 属性 | 值 |
|------|-----|
| 决策 | 广播得到的只读 `TensorView` 不提供 `iter_mut()` |
| 理由 | 广播通过零步长实现，多个逻辑索引映射同一物理地址；可变写入会导致数据竞争和未定义行为 |
| 替代方案 | 允许可变迭代但写入同一地址 |
| 拒绝原因 | 语义不明确，容易引入 bug |

---

## 10. 性能考量

### 10.1 连续 vs 非连续性能对比

| 场景 | 实现路径 | 每元素开销 | 缓存友好性 |
|------|----------|-----------|-----------|
| 连续 F-order | 指针递增 | 1 次指针加法 | 最优（顺序访问） |
| 非连续（切片） | 步长状态机 | 多次乘加运算 | 较差（跳跃访问） |
| 非连续（转置） | 步长状态机 | 多次乘加运算 | 较差（跳跃访问） |
| 广播视图 | 零步长处理 | 条件判断 + 缓存 | 依赖广播维度 |

**性能数据（参考）**:

| 操作 | 连续数组 | 非连续数组（步长=2） | 性能比 |
|------|----------|---------------------|--------|
| 遍历 1M 元素 | ~1ms | ~3ms | 3x |
| 缓存命中率 | ~95% | ~60% | - |

### 10.2 复杂度标注

- `Elements::new()`: O(1)，仅初始化状态
- `Elements::next()`（快速路径）: O(1)，指针递增
- `Elements::next()`（慢速路径）: O(ndim)，索引递增
- `AxisIter::next()`: O(1)，子视图切片
- `Windows::next()`: O(ndim)，计算切片范围

---

## 11. no_std 兼容性

迭代器模块完全兼容 `no_std` 环境。所有迭代器仅依赖 `core` 中的 `Iterator` / `ExactSizeIterator` trait，不进行堆分配。

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `Elements` / `ElementsMut` | ✅ | 指针递增或步长状态机，无堆分配 |
| `AxisIter` / `AxisIterMut` | ✅ | 产生子视图（零拷贝），无堆分配 |
| `Windows` | ✅ | 产生子视图（零拷贝），无堆分配 |
| `IndexedIter` / `IndexedIterMut` | ✅ | 索引状态在栈上维护，无堆分配 |
| `Zip` | ✅ | 组合已有迭代器，调用 `broadcast_shape()` 纯计算 |
| `StrideState` | ✅ | 栈上索引数组，无堆分配 |

迭代器内部不使用 `Vec`、`Box`、`Arc` 等堆类型。唯一的外部依赖为 `broadcast_shape()`，该函数为纯计算函数，不涉及堆分配。

条件编译处理：

```rust
// Iterators depend only on core traits
// StrideState manages indices on the stack
// Zip itself only combines already-normalized producers; broadcast normalization lives above Zip

#[cfg(not(feature = "std"))]
// No conditional compilation needed — iterators work in pure no_std
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
