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
| 元素遍历 | 按元素遍历（F-order 内存顺序） | 具体运算逻辑（参见 `11-elementwise-ops.md §1`） |
| 轴遍历 | 沿指定轴产生子张量视图 | 子视图的具体操作 |
| 窗口遍历 | 滑动窗口产生子视图 | 卷积运算 |
| 索引遍历 | 带多维索引的元素遍历 | 索引赋值操作 |
| 同步遍历 | 多张量同步遍历（Zip） | 广播规则计算（参见 `15-broadcast.md §3`） |
| 并行迭代 | — | 并行迭代（参见 `09-parallel-backend.md §4.5`） |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
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
├── windows.rs     # Windows / WindowsMut 滑动窗口迭代
├── indexed.rs     # IndexedIter / IndexedIterMut 带索引遍历
├── zip.rs         # Zip 多张量同步迭代
└── lanes.rs       # LaneIter / LaneIterMut 沿轴方向的 1D 切片序列迭代
```

单文件划分理由：各迭代器类型之间独立，拆分文件降低单文件复杂度，便于并行开发。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/iter/
├── crate::tensor        # TensorBase<S, D>, TensorView, TensorViewMut
├── crate::dimension     # Dimension trait, Ix0~Ix6, IxDyn
├── crate::storage       # Storage, StorageMut trait
├── crate::layout        # LayoutFlags, Order
└── crate::broadcast     # broadcast_shape()（仅 Zip 使用）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `TensorView<'a, A, D>`, `TensorViewMut<'a, A, D>`, `.shape()`, `.strides()`, `.as_ptr()`, `.len()`（参见 `07-tensor.md §4`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `RemoveAxis`, `D::Smaller`（参见 `02-dimension.md §4`） |
| `storage` | `Storage<Elem = A>`, `StorageMut<Elem = A>`, `Owned<A>`（参见 `05-storage.md §4`） |
| `layout` | `LayoutFlags`, `is_f_contiguous()`（参见 `06-memory-layout.md §4`） |
| `broadcast` | `broadcast_shape()`（Zip 构造时验证形状兼容性，参见 `15-broadcast.md §4`） |

### 3.3 依赖方向

> **依赖方向：单向向上。** `iter/` 仅消费 `tensor`、`dimension`、`storage`、`layout` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

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
pub struct AxisIter<'a, A, D: Dimension> {
    // Internal fields: view, axis, current position, length
}

/// Mutable axis iterator.
pub struct AxisIterMut<'a, A, D: Dimension> {}

impl<'a, A, D: Dimension> Iterator for AxisIter<'a, A, D> {
    type Item = TensorView<'a, A, D::Smaller>;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for AxisIter<'a, A, D> {}
```

> **设计决策：** `AxisIter` 的 `Item` 类型为 `TensorView<'a, A, D::Smaller>`。这要求 `Dimension` trait 提供 `type Smaller: Dimension` 关联类型。零维张量（Ix0）不支持按轴遍历，因为 `Ix0` 无 `Smaller` 类型。

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

pub struct WindowsMut<'a, A, D: Dimension> {}

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
```

### 4.5 Zip 多张量同步迭代器

```rust
/// Multi-tensor synchronized iterator, used for zip_with and similar operations.
///
/// Supports broadcasting: automatically expands when shapes are broadcastable.
/// Broadcast results are read-only; mutable iteration is not supported.
pub struct Zip<Parts, D: Dimension> {
    // Internal fields: producer tuple, broadcasted shape, stride info
}

impl<D: Dimension> Zip<(), D> {
    /// Create a Zip from the first tensor.
    pub fn from<P: NdProducer<Dim = D>>(producer: P) -> Zip<(P,), D>;
}

impl<Parts, D: Dimension> Zip<Parts, D> {
    /// Add another tensor to the Zip, handling broadcast automatically.
    pub fn and<P: NdProducer<Dim = D>>(self, producer: P)
        -> Result<Zip<(Parts, P), D>, BroadcastError>;

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

### 4.6 LaneIter 沿轴方向的一维切片序列迭代器

`LaneIter` 产出沿指定轴方向的所有 **1D 视图**（称为"lane"），每个 lane 是穿越整个轴长度的切片。  
与 `AxisIter` 的区别：`AxisIter` 产出与轴**正交**的子张量（降维），`LaneIter` 产出沿轴**方向**的 1D 切片（不降维，但视图始终是 1D）。

例如对 shape `[rows, cols]` 的矩阵：
- `axis_iter(Axis(0))`：产出每一**行**视图，类型 `TensorView<'a, A, Ix1>`，共 `rows` 个
- `lanes(Axis(0))`：产出每一**列**（沿 axis 0 方向），类型 `TensorView1<'a, A>`，共 `cols` 个
- `lanes(Axis(1))`：产出每一**行**（沿 axis 1 方向），类型 `TensorView1<'a, A>`，共 `rows` 个

```rust
/// Iterator over 1D lanes along a given axis.
///
/// Each lane is a 1D view (`TensorView1`) spanning the full length of the specified axis.
/// The iterator yields one lane for each index combination in all _other_ axes.
///
/// # Relationship to AxisIter
///
/// - `axis_iter(Axis(k))`: yields sub-tensors orthogonal to axis k (D → D::Smaller).
/// - `lanes(Axis(k))`:     yields 1D slices *along* axis k (always Ix1 output).
///
/// # Zero-copy
///
/// Each lane is a view into the original data with no copying.
pub struct LaneIter<'a, A, D: Dimension> {
    // Internal fields: multi-dim index state for the "outer" axes,
    // base view, target axis.
}

/// Mutable variant.
pub struct LaneIterMut<'a, A, D: Dimension> {}

impl<'a, A, D: Dimension> Iterator for LaneIter<'a, A, D> {
    /// A 1D immutable view along the target axis.
    type Item = TensorView1<'a, A>;

    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for LaneIter<'a, A, D> {}

impl<'a, A, D: Dimension> Iterator for LaneIterMut<'a, A, D> {
    /// A 1D mutable view along the target axis.
    type Item = TensorViewMut1<'a, A>;

    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for LaneIterMut<'a, A, D> {}
```

#### LaneIter 产出数量

`LaneIter` 产出的 lane 数 = `tensor.len() / tensor.shape()[axis]`，即除目标轴以外所有轴的元素总数。

| shape | axis | lane 数 | 每 lane 长度 |
|-------|------|---------|-------------|
| `[3, 4]` | `Axis(0)` | 4（每列） | 3 |
| `[3, 4]` | `Axis(1)` | 3（每行） | 4 |
| `[2, 3, 4]` | `Axis(1)` | 8（= 2×4） | 3 |

> **设计决策：** `LaneIter::Item` 固定为 `TensorView1`（1D 视图），而非泛型降维视图。
> 这样无需 `RemoveAxis` 约束，且 API 语义（"给我沿某轴的所有 1D 切片"）更清晰。

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
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, A, D::Smaller>
    where
        D: RemoveAxis;

    /// Sliding window iterator.
    pub fn windows(&self, size: impl IntoDimension<Dim = D>) -> Option<Windows<'_, A, D>>;

    /// Iterate 1D lanes along an axis (yields 1D views along the axis direction).
    ///
    /// # Panics
    ///
    /// Panics if `axis.index() >= self.ndim()`.
    pub fn lanes(&self, axis: Axis) -> LaneIter<'_, A, D>;
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
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, A, D::Smaller>
    where
        D: RemoveAxis;

    /// Mutable 1D lane iterator along an axis.
    ///
    /// # Panics
    ///
    /// Panics if `axis.index() >= self.ndim()`.
    pub fn lanes_mut(&mut self, axis: Axis) -> LaneIterMut<'_, A, D>;
}
```

### 4.7 Good / Bad 对比示例

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
let broadcast_view = tensor.broadcast([3, 4]).unwrap();
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

### 5.3 广播可变迭代禁止

```rust
// SAFETY: broadcast() returns a TensorView with zero-stride dimensions,
// multiple logical indices map to the same physical address; mutable writes
// would cause data races.
// Therefore broadcast() only returns an immutable view.
```

### 5.4 填充数组迭代

填充数组的迭代仅遍历逻辑元素。迭代器通过 shape 中的逻辑维度计数，跳过填充区域。

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
  - 预计: 15 min

- [ ] **T4**: 实现 `AxisIter` / `AxisIterMut`
  - 文件: `src/iter/axis.rs`
  - 内容: 沿轴迭代，产出子视图
  - 测试: `test_axis_iter_count`, `test_axis_iter_shape`, `test_axis_iter_empty_axis`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现 `Windows` / `WindowsMut`
  - 文件: `src/iter/windows.rs`
  - 内容: 滑动窗口迭代，窗口大小验证
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
  - 预计: 15 min

- [ ] **T8**: 实现 `LaneIter` / `LaneIterMut`
  - 文件: `src/iter/lanes.rs`
  - 内容: 沿轴方向的 1D 切片序列迭代；内部维护"外部轴"的多维索引状态机；每次 `next()` 计算偏移量产出 `TensorView1` 视图
  - 测试: `test_lanes_count`, `test_lanes_shape`, `test_lanes_axis0_axis1`, `test_lanes_empty`, `test_lanes_ix1`
  - 前置: T3（步长状态机复用）
  - 预计: 15 min

### Wave 4: TensorBase 入口集成

- [ ] **T9**: 在 TensorBase 上添加迭代器入口方法
  - 文件: `src/tensor/`（或 `src/iter/mod.rs` 通过 trait extension）
  - 内容: `iter()`, `iter_mut()`, `axis_iter()`, `windows()`, `indexed_iter()`, `lanes()`, `lanes_mut()` 等
  - 测试: `test_tensor_iter_integration`
  - 前置: T3, T4, T5, T6, T7, T8
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] [T2]
           │     │
Wave 2: [T3] [T4] [T5]
           │     │     │
Wave 3: [T6] ──── [T7] [T8]
                        │
Wave 4:             [T9]
```

---

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_elements_f_contig` | F-order 连续数组：迭代顺序与物理布局一致 | 高 |
| `test_elements_non_contiguous` | 非连续数组（切片后）：步长跳转正确 | 高 |
| `test_elements_empty` | 空数组 `shape=[0, 3]`：`iter()` 立即结束 | 高 |
| `test_elements_ix0` | 零维张量：`iter()` 产出恰好 1 个元素 | 高 |
| `test_elements_mut_write` | `iter_mut()` 写入后数据正确 | 中 |
| `test_axis_iter_count` | `axis_iter(Axis(0)).count() == shape[0]` | 高 |
| `test_axis_iter_shape` | 沿轴迭代产出的子视图形状正确 | 高 |
| `test_axis_iter_ix0_panic` | 零维张量调用 `axis_iter` 编译失败或 panic | 中 |
| `test_windows_count` | 窗口数 = `product(shape - window + 1)` | 高 |
| `test_windows_too_large` | 窗口大于数组返回 `None` | 中 |
| `test_windows_empty` | 空数组返回 `None` | 中 |
| `test_indexed_iter_order` | 索引按 F-order 递增 | 高 |
| `test_indexed_iter_ix0` | 零维张量索引为空切片 | 中 |
| `test_zip_two_tensors` | Zip 同步遍历两个同形状张量 | 高 |
| `test_zip_broadcast` | Zip 广播形状兼容 | 高 |
| `test_zip_broadcast_readonly` | 广播结果不可变迭代 | 中 |
| `test_padded_iter` | 填充数组仅遍历逻辑元素 | 低 |
| `test_lanes_count_2d_axis0` | `lanes(Axis(0)).count() == shape[1]`（每列一个 lane） | 高 |
| `test_lanes_count_2d_axis1` | `lanes(Axis(1)).count() == shape[0]`（每行一个 lane） | 高 |
| `test_lanes_item_shape` | 每个 lane 的 shape 为 `[shape[axis]]` | 高 |
| `test_lanes_values_col` | `lanes(Axis(0))` 产出的第一个 lane 等于第 0 列数据 | 高 |
| `test_lanes_values_row` | `lanes(Axis(1))` 产出的第一个 lane 等于第 0 行数据 | 高 |
| `test_lanes_3d` | 3D 张量 `lanes(Axis(1))` 产出数量 = shape[0] * shape[2] | 中 |
| `test_lanes_mut_write` | `lanes_mut()` 写入后数据正确 | 高 |
| `test_lanes_ix1` | 1D 张量 `lanes(Axis(0))` 产出 1 个长度 n 的 lane | 中 |
| `test_lanes_empty` | 含 0 轴的张量 `lanes()` 立即结束 | 中 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `iter()` 立即结束，`count() == 0` |
| 单元素 `shape=[1, 1]` | `iter()` 产出 1 项 |
| 零维张量 Ix0 | `iter()` 产出 1 项，`axis_iter()` 不可用 |
| 非连续切片 `s![.., 0..3]` | `iter()` 正确处理步长跳转 |
| 负步长（反转切片） | `iter()` 正确处理负步长 |
| 广播视图 `shape=[1, 4]` | `iter()` 遍历逻辑元素，`iter_mut()` 编译拒绝 |
| 填充数组 | 仅遍历逻辑元素 |
| `lanes(Axis(0))` 空张量 `shape=[0, 4]` | `count() == 4`（外轴非空），每 lane `len() == 0` |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `iter().count() == tensor.len()` | 随机形状 `[0..10, 0..10, 0..10]` |
| `axis_iter(Axis(i)).count() == shape[i]` | 随机形状 |
| `iter().len()` 每次调用后递减 | 迭代过程中检查 `ExactSizeIterator` |

---

## 8. 与其他模块的交互

### 8.1 与 tensor 模块

```rust
// tensor module provides iterator entry methods（参见 07-tensor.md §4.6）
// iter module consumes TensorView / TensorViewMut
```

### 8.2 与 storage / dimension 模块

```rust
// Iterators read data via the Storage trait（参见 05-storage.md §4）
// Index state is managed via the Dimension trait（参见 02-dimension.md §4）
```

### 8.3 与 broadcast 模块

```rust
// Zip calls broadcast_shape() at construction to verify shape compatibility（参见 15-broadcast.md §4）
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
| 决策 | 广播视图（BroadcastView）不提供 `iter_mut()` |
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
| `Windows` / `WindowsMut` | ✅ | 产生子视图（零拷贝），无堆分配 |
| `IndexedIter` / `IndexedIterMut` | ✅ | 索引状态在栈上维护，无堆分配 |
| `Zip` | ✅ | 组合已有迭代器，调用 `broadcast_shape()` 纯计算 |
| `StrideState` | ✅ | 栈上索引数组，无堆分配 |

迭代器内部不使用 `Vec`、`Box`、`Arc` 等堆类型。唯一的外部依赖为 `broadcast_shape()`，该函数为纯计算函数，不涉及堆分配。

条件编译处理：

```rust
// Iterators depend only on core traits
// StrideState manages indices on the stack
// Zip calls broadcast_shape() which is a pure function

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

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
