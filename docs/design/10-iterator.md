# 迭代器模块设计

> 文档编号: 10 | 模块: `src/iter/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`
> 需求参考: 需求说明书 §11
> 范围声明: 范围内

---

## 1. 模块概述

### 1.1 定位

迭代器模块是 Xenon 张量库的数据遍历基础设施，为所有张量操作提供统一、高效、类型安全的遍历机制。当前版本仅覆盖元素遍历、按轴遍历与按索引遍历。

### 1.2 职责边界

| 职责           | 包含                                     | 不包含                                 |
| -------------- | ---------------------------------------- | -------------------------------------- |
| 元素遍历       | 按元素遍历（逻辑 F-order 索引顺序）      | 具体运算逻辑（参见 `11-math.md §1`）   |
| 轴遍历         | 沿指定轴产生子张量视图                   | 子视图的具体操作                       |
| 索引遍历       | 带多维索引的元素遍历                     | 索引赋值操作                           |
| 内部能力提示   | 各操作模块可直接实现自身所需的内部迭代分发；`Windows` 可作为后续版本议题保留 | 不单独设计统一的多输入 lock-step 中间抽象；`Windows` 不纳入当前版本实现范围 |
| 并行迭代       | —                                        | 并行迭代（参见 `09-parallel.md`） |

### 1.3 在架构中的位置

```
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (depends only on core/alloc, not layout)
L4: tensor (depends on storage, dimension)
L5: iter  <- current module
```

### 1.4 设计原则

| 原则         | 体现                                                  |
| ------------ | ----------------------------------------------------- |
| 步长感知     | 所有迭代器正确处理连续、转置后非连续和零步长广播视图  |
| 零拷贝       | 轴迭代器产生子视图，不拷贝数据                        |
| trait 一致性 | 所有迭代器实现 `Iterator` + `ExactSizeIterator`       |
| 空数组安全   | 空数组迭代立即结束，零维张量元素迭代产出恰好 1 个元素 |
| 广播只读     | 禁止对广播结果进行可变迭代                            |

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §11 |
| 范围内   | 元素遍历、按轴遍历、带索引遍历，以及供各运算模块直接实现的内部迭代分发约定。 |
| 范围外   | 独立的多输入 lock-step 迭代抽象、`DoubleEndedIterator`、`Windows` / `LaneIter` 与并行公开迭代接口。 |
| 非目标   | 不扩展当前公开迭代器集合，不新增第三方依赖，也不在本文定义并行 API 契约。 |

---

## 3. 文件位置

```
src/iter/
├── mod.rs         # module entry, public exports, iterator trait definitions
├── elements.rs    # Elements / ElementsMut flat element iteration
├── axis.rs        # AxisIter / AxisIterMut axis-wise iteration
└── indexed.rs     # IndexedIter / IndexedIterMut indexed iteration
```

单文件划分理由：当前版本公开范围仅覆盖元素、按轴、按索引三类迭代器，拆分文件降低单文件复杂度并保持职责清晰。逐元素运算、广播与归约如需 lock-step 遍历，由各操作模块在自身内部直接组织遍历逻辑；`windows.rs`、`lanes.rs` 暂不纳入当前版本。

---

## 4. 依赖关系

### 4.1 Invariants

| 不变量 | 说明 |
| ---- | ---- |
| F-order only | 所有公开迭代器的逻辑产出顺序固定为 F-order，不提供 C-order 或顺序切换选项。 |
| 长度一致性 | `iter().count()`、`indexed_iter().count()` 与 `len()` 必须一致；零维张量恰好产出 1 个元素，空张量产出 0 个元素。 |
| 只读广播 | 广播视图只能参与只读遍历；任何可变迭代入口都不得对 `BroadcastView` 开放。 |
| 轴迭代降维 | `AxisIter` / `AxisIterMut` 每次产出的子视图维度必须为 `D::Smaller`，形状等于原形状移除目标轴后的结果。 |

### 4.2 Error Scenarios

| 场景 | 错误 |
| ---- | ---- |
| `axis_iter()` / `axis_iter_mut()` 的 `axis` 越界 | 返回 `XenonError::InvalidAxis { operation: "axis_iter", axis, ndim, shape }` 或 `XenonError::InvalidAxis { operation: "axis_iter_mut", axis, ndim, shape }`。 |
| rank-0 动态维张量调用按轴迭代 | 返回 `XenonError::InvalidAxis { operation, axis: 0, ndim: 0, shape }`，其中 `shape` 为 `Vec<usize>` 形式的空形状。 |
| 试图把广播结果作为可变迭代输入 | 不提供公开 API；通过类型系统在编译期拒绝，而不是返回运行时错误。 |

### 4.3 依赖图

```
src/iter/
├── crate::tensor        # TensorBase<S, D>, TensorView, TensorViewMut
├── crate::dimension     # Dimension trait, Ix0~Ix6, IxDyn
├── crate::storage       # Storage, StorageMut trait
└── crate::tensor        # Layout and contiguity queries via TensorBase
```

### 4.4 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tensor`    | `TensorBase<S, D>`, `TensorView<'a, A, D>`, `TensorViewMut<'a, A, D>`, `.shape()`, `.strides()`, `.as_ptr()`, `.len()`（参见 `07-tensor.md §5.3` / `§5.4` / `§5.7`） |
| `dimension` | `Dimension`, `Axis`, `Ix0`~`Ix6`, `IxDyn`, `RemoveAxis`, `D::Smaller`（参见 `02-dimension.md §5`）                                               |
| `storage`   | `Storage<Elem = A>`, `StorageMut<Elem = A>`, `Owned<A>`（参见 `05-storage.md §5`）                                                               |
| `tensor`    | `.is_f_contiguous()`, 布局标志查询（参见 `07-tensor.md §5.3`）                                                                                   |

### 4.5 依赖方向

> **依赖方向：单向向上。** `iter/` 仅消费 `tensor`、`dimension`、`storage` 等核心模块，不被它们依赖。布局/连续性判断通过 `TensorBase` 暴露的查询接口完成。

### 4.6 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

> **DoubleEndedIterator 说明：** 当前版本所有迭代器不实现 `DoubleEndedIterator`，因为 F-order 遍历的反向语义在高维情况下不明确，且需求未要求此功能。

### 5.1 Elements 迭代器

```rust
/// Flat element iterator, traverses all elements in logical F-order index order.
pub struct Elements<'a, A, D: Dimension> {
    // Internal fields: view, pointer/index state, remaining count,
    // and PhantomData<&'a A> to tie the yielded references to lifetime 'a.
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

### 5.2 AxisIter 沿轴迭代器

```rust
/// Axis iterator, yields a sub-tensor view with reduced dimension each step.
///
/// Input dimension N, output dimension N-1 views.
pub struct AxisIter<'a, A, D: Dimension + RemoveAxis> {
    // Internal fields: view, axis, current position, length
}

/// Mutable axis iterator.
///
/// # Safety
///
/// Each call to `next()` computes the next subview base offset from the selected
/// axis stride and current logical position, then yields exactly one mutable view
/// for that disjoint slice. Successive positions differ by one axis-step, so the
/// produced `&mut` views cover non-overlapping logical regions along the iterated
/// axis. The iterator advances monotonically and never revisits an earlier offset,
/// which prevents mutable aliasing between yielded items.
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

> **设计决策：** `AxisIter` 的 `Item` 类型为 `TensorView<'a, A, D::Smaller>`。这要求 `D` 满足 `RemoveAxis`，并由 `RemoveAxis` trait 提供 `type Smaller: Dimension` 关联类型。同时，公开构造路径仍须在运行时先检查 `self.ndim() == 0`，并返回可恢复错误，以覆盖动态维度张量在 rank=0 时的按轴遍历请求。

### 5.3 内部迭代分发说明

当前版本不在 `iter` 模块中设计统一的多输入 lock-step 结构体或配套方法。逐元素运算、广播、归约等需要多输入同步遍历的模块，应直接基于 `Elements` / `ElementsMut`、广播视图和各自的内部状态机完成迭代分发。这样可以避免在 `iter` 模块额外引入一个被误解为稳定能力边界的中间抽象。

> **Zip 能力说明：** 当前版本明确不支持 `Zip` 结构体、`zip_with`、`zip_apply` 或任何等价的公开 lock-step 迭代 API；如后续需要，应在独立设计文档中重新定义其错误语义、广播边界和别名约束。

> **说明：** `Windows` 与 `LaneIter` 仍不属于需求说明书 §11 的当前范围；如后续引入，应在独立文档中重新定义窗口语义、1D 产出语义以及别名约束。

### 5.4 IndexedIter 带索引迭代器

```rust
/// Element iterator with multi-dimensional indices.
///
/// Yields (D, &'a A) tuples, indices increment in F-order.
pub struct IndexedIter<'a, A, D: Dimension> {
    // Internal fields: Elements iterator, current index, stride state machine
}

/// Mutable indexed iterator.
///
/// # Safety
///
/// `IndexedIterMut` yields at most one mutable reference for each logical index.
/// The internal state machine visits every logical coordinate once in F-order and
/// computes the element address from the tensor's validated stride metadata. Since
/// no logical index is repeated during a single traversal, each yielded `&mut A`
/// refers to a distinct element slot and does not overlap with previously yielded
/// references.
pub struct IndexedIterMut<'a, A, D: Dimension> {}

impl<'a, A, D: Dimension> Iterator for IndexedIter<'a, A, D> {
    type Item = (D, &'a A);
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIter<'a, A, D> {}

impl<'a, A, D: Dimension> Iterator for IndexedIterMut<'a, A, D> {
    type Item = (D, &'a mut A);
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D: Dimension> ExactSizeIterator for IndexedIterMut<'a, A, D> {}
```

> **索引所有权说明：** 索引值直接使用维度类型 `D` 本身承载（如 `Ix2` 或 `IxDyn`），而不是借用切片。这样与 `Dimension` trait 的权威定义保持一致，也符合按索引迭代“产出同 rank 的拥有型索引值”这一语义。

### 5.5 LaneIter 延期到后续版本

`LaneIter` / `LaneIterMut` 不属于需求说明书 §11 的必须项。当前版本先不设计该 API，避免与 `AxisIter`、`Windows`、`IndexedIter` 的职责边界重叠；如后续引入，应在独立文档中重新定义 1D 产出语义、轴方向约定与可变别名规则。

### 5.6 TensorBase 上的迭代器入口方法

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
    pub fn axis_iter(&self, axis: Axis) -> Result<AxisIter<'_, A, D>, XenonError>
    where
        D: RemoveAxis;

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
    pub fn axis_iter_mut(&mut self, axis: Axis) -> Result<AxisIterMut<'_, A, D>, XenonError>
    where
        D: RemoveAxis;

}
```

### 5.7 Good / Bad 对比示例

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

// Bad - ignoring the recoverable error path for zero-rank axis iteration
let scalar = Tensor::<f64, IxDyn>::from_shape_vec(IxDyn::from_slice(&[]), vec![1.0])?;
// let _ = scalar.axis_iter(Axis(0)).unwrap();
```

---

## 6. 内部实现设计

### 6.1 Elements 快速/慢速路径选择

```
Elements::new(view):
    if view.is_f_contiguous():
        // Fast path: pointer increment
        ptr = view.as_ptr()
        end = ptr + view.len()
    else:
        // Slow path: stride state machine
        stride_state = StrideState::new(&view)
        index = D::zeros(view.ndim())
```

### 6.2 步长状态机（StrideState）

```
increment_index_f(shape, index):
    for i in 0..ndim:
        index[i] += 1
        if index[i] < shape[i]:
            return  // no carry
        index[i] = 0  // carry to next dimension
```

> **偏移量计算边界：** 当前版本仅处理需求允许的合法 stride 布局：连续 F-order、转置产生的非连续视图，以及广播产生的零步长视图。元素偏移量计算仍为 `offset = base_offset + Σ(stride[i] * index[i])`，但不接受负步长输入。

### 6.3 广播可变迭代禁止

```rust
// SAFETY: broadcast_to() returns a TensorView with zero-stride dimensions,
// multiple logical indices map to the same physical address; mutable writes
// would create immediate mutable aliasing.
// Therefore broadcast_to() only returns an immutable view.
```

> **编译期防护机制：** 广播结果返回 `TensorView`（不可变视图），而非 `TensorViewMut`。由于 `TensorView` 不提供 `iter_mut()` 方法（`iter_mut()` 要求 `StorageMut` 约束，仅 `TensorViewMut` 和 `Tensor` 满足），对广播结果调用 `iter_mut()` 会在编译期被类型系统拒绝，无需运行时检查。参见 `07-tensor.md §5.7` 中视图方法的约束差异。

### 6.4 填充数组迭代

填充数组的迭代仅遍历逻辑元素。迭代器通过 shape 中的逻辑维度计数，跳过填充区域。

> **空轴行为：** 空轴（`shape[axis] == 0`）的 `AxisIter` / `AxisIterMut` 产出 0 个元素，与标准库空切片迭代器行为一致。

> **ZST 迭代说明：** ZST（zero-sized type）迭代相关讨论仅用于说明边界情况处理，ZST 不是当前版本的张量元素类型（`require.md` §4）；若内部测试或辅助代码覆盖该路径，也不得把它扩展为公开元素类型承诺。

### 6.5 并行分块说明

当前版本不在 `iter` 模块中设计独立的内部区间分块抽象。若并行后端需要对元素遍历做分块，应由并行执行模块基于自身的任务划分策略直接维护逻辑区间和调度状态；`iter` 文档只约束串行迭代器的外部语义，不再把这类内部多输入遍历或分块结构描述为稳定设计能力。

---

## 7. 实现任务拆分

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

- [ ] **T6**: 实现 `IndexedIter` / `IndexedIterMut`
  - 文件: `src/iter/indexed.rs`
  - 内容: 基于 Elements 的索引包装
  - 测试: `test_indexed_iter_order`, `test_indexed_iter_ix0`
  - 前置: T3
  - 预计: 10 min

### Wave 3: TensorBase 入口集成

- [ ] **T9**: 在 TensorBase 上添加迭代器入口方法
  - 文件: `src/tensor/`（或 `src/iter/mod.rs` 通过 trait extension）
  - 内容: `iter()`, `iter_mut()`, `axis_iter()`, `indexed_iter()` 等
  - 测试: `test_tensor_iter_integration`
  - 前置: T3, T4, T6
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] [T2]
           |     |
Wave 2: [T4] [T3]
                 |
Wave 3:        [T6]
                 |
Wave 4:        [T9]
```

---

## 8. 测试计划

### 8.1 测试分类总表

| 测试分类 | 说明                                   | 包含的测试                                                                                                                                                                                                                                                      |
| -------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 单元测试 | 验证单个迭代器类型的基本功能           | `test_elements_f_contig`, `test_elements_non_contiguous`, `test_elements_empty`, `test_elements_ix0`, `test_elements_mut_write`, `test_axis_iter_count`, `test_axis_iter_shape`, `test_axis_iter_ix0_error`, `test_indexed_iter_order`, `test_indexed_iter_ix0` |
| 集成测试 | 验证迭代器与 TensorBase 入口方法的集成 | `test_tensor_iter_integration`                                                                                                                                                                                                                                  |
| 边界测试 | 空数组、零维张量、非连续内存等边界条件 | `test_elements_empty`, `test_elements_ix0`, `test_axis_iter_ix0_error`（详见 §8.3）                                                                                                                                                                             |
| 属性测试 | 通过随机输入验证不变量                 | `iter().count() == tensor.len()`, `axis_iter(Axis(i)).count() == shape[i]`, `ExactSizeIterator` 递减不变量（详见 §8.4）                                                                                                                                         |

### 8.2 单元测试清单

| 测试函数                       | 测试内容                                      | 优先级 |
| ------------------------------ | --------------------------------------------- | ------ |
| `test_elements_f_contig`       | F-order 连续数组：迭代顺序与物理布局一致      | 高     |
| `test_elements_non_contiguous` | 非连续数组（切片后）：步长跳转正确            | 高     |
| `test_elements_empty`          | 空数组 `shape=[0, 3]`：`iter()` 立即结束      | 高     |
| `test_elements_ix0`            | 零维张量：`iter()` 产出恰好 1 个元素          | 高     |
| `test_elements_mut_write`      | `iter_mut()` 写入后数据正确                   | 中     |
| `test_axis_iter_count`         | `axis_iter(Axis(0)).count() == shape[0]`      | 高     |
| `test_axis_iter_shape`         | 沿轴迭代产出的子视图形状正确                  | 高     |
| `test_indexed_iter_order`      | 索引按 F-order 递增                           | 高     |
| `test_indexed_iter_ix0`        | 零维张量索引为空切片                          | 中     |
| `test_axis_iter_ix0_error`     | 零维动态张量调用 `axis_iter()` 返回可恢复错误 | 高     |
| `test_padded_iter`             | 填充数组仅遍历逻辑元素                        | 低     |

### 8.3 边界测试场景

| 场景                          | 预期行为                                         |
| ----------------------------- | ------------------------------------------------ |
| 空数组 `shape=[0, 3]`         | `iter()` 立即结束，`count() == 0`                |
| 单元素 `shape=[1, 1]`         | `iter()` 产出 1 项                               |
| 零维张量 Ix0 / rank-0 `IxDyn` | `iter()` 产出 1 项，`axis_iter()` 返回可恢复错误 |
| 非连续切片 `s![.., 0..3]`     | `iter()` 正确处理步长跳转                        |
| 广播视图 `shape=[1, 4]`       | `iter()` 遍历逻辑元素，`iter_mut()` 编译拒绝     |
| 填充数组                      | 仅遍历逻辑元素                                   |

### 8.4 属性测试不变量

| 不变量                                   | 测试方法                           |
| ---------------------------------------- | ---------------------------------- |
| `iter().count() == tensor.len()`         | 随机形状 `[0..10, 0..10, 0..10]`   |
| `axis_iter(Axis(i)).count() == shape[i]` | 随机形状                           |
| `iter().len()` 每次调用后递减            | 迭代过程中检查 `ExactSizeIterator` |

### 8.5 集成测试

| 测试文件                 | 测试内容                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `tests/test_iterator.rs` | `tensor.iter()` / `axis_iter()` / `indexed_iter()` 与 `tensor`、`shape` 模块的协同路径 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `iter` / `axis_iter` / `indexed_iter` 在无并行后端时保持既定顺序与长度语义。 |
| 启用并行相关能力 | 并行执行模块自管的内部区间划分必须与串行迭代保持相同的元素覆盖与顺序契约。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| 广播结果不可变，不提供 `iter_mut()` | 编译期测试或 trait 约束验证。 |
| `AxisIter` 仅对满足 `RemoveAxis` 的维度开放 | 编译期测试。 |
| `DoubleEndedIterator` 不属于当前公开接口 | 编译期失败测试或 API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
| ---- | -------- | --------- | ---- |
| `iter → tensor` | `tensor` | `TensorBase<S, D>`、`TensorView<'a, A, D>`、`TensorViewMut<'a, A, D>` | 由 `tensor` 暴露迭代器入口与视图类型，参见 `07-tensor.md` §5。 |
| `iter → dimension` | `dimension` | `Dimension`、`Axis`、`RemoveAxis` | 迭代状态机按维度与轴语义推进，参见 `02-dimension.md` §5。 |
| `iter → storage` | `storage` | `Storage`、`StorageMut` | 元素访问与可变访问分别受只读/可写存储约束保护，参见 `05-storage.md` §5。 |

### 9.2 数据流描述

```text
User calls tensor.iter() / axis_iter() / indexed_iter()
    │
    ├── tensor exposes TensorView / TensorViewMut entry points
    ├── iter builds iterator state from shape + strides
    └── iter yields elements or sub-views for math / reduction / overload
```

### 9.3 与 storage / dimension 模块

```rust
// Iterators read data via the Storage trait (see 05-storage.md §5)
// Index state is managed via the Dimension trait (see 02-dimension.md §5)
```

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `axis_iter()` / `axis_iter_mut()` 在 `axis` 越界或 rank-0 动态维输入时返回 `XenonError::InvalidAxis { operation: Cow<'static, str>, axis: usize, ndim: usize, shape: Vec<usize> }`。 |
| Panic | 公开迭代器 API 不引入新的 panic 语义；仅内部 producer 分块等不变量破坏可使用断言。 |
| 路径一致性 | 连续、非连续、零步长广播视图及并行 producer 的外部迭代顺序与长度语义必须一致。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录（ADR）

### 决策 1：F-order 遍历顺序

| 属性     | 值                                                                                |
| -------- | --------------------------------------------------------------------------------- |
| 决策     | 所有元素迭代器默认按 F-order（列优先）遍历                                        |
| 理由     | Xenon 只支持 F-order 布局，F-order 遍历对连续数组产生顺序内存访问，缓存友好性最优 |
| 替代方案 | 提供遍历顺序参数（F/C）                                                           |
| 替代方案 | 默认 C-order                                                                      |
| 拒绝原因 | 项目范围仅 F-order，增加 C-order 选项增加复杂度且无实际收益                       |

### 决策 2：ExactSizeIterator 要求

| 属性     | 值                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------ |
| 决策     | 所有迭代器须实现 `ExactSizeIterator`                                                             |
| 理由     | 提供精确的元素数量信息，调用者可预分配缓冲区、验证迭代完整性；`size_hint()` 的上界和下界始终相等 |
| 替代方案 | 仅实现 `Iterator`，不保证精确长度                                                                |
| 拒绝原因 | 归约、逐元素运算与并行分块都需要精确长度信息                                                     |

### 决策 3：零维张量元素迭代产出 1 个元素

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | Ix0 张量的 `iter()` 产出恰好 1 个元素                                      |
| 理由     | 零维张量在数学上表示标量，恰好有 1 个值；与 ndarray 行为一致；`len() == 1` |
| 替代方案 | 迭代产出 0 个元素                                                          |
| 拒绝原因 | 与 `len()` 不一致，违反 `iter().count() == len()` 不变量                   |

### 决策 4：广播结果不可变迭代

| 属性     | 值                                                                                   |
| -------- | ------------------------------------------------------------------------------------ |
| 决策     | 广播得到的只读 `TensorView` 不提供 `iter_mut()`                                      |
| 理由     | 广播通过零步长实现，多个逻辑索引映射同一物理地址；可变写入会导致数据竞争和未定义行为 |
| 替代方案 | 允许可变迭代但写入同一地址                                                           |
| 拒绝原因 | 语义不明确，容易引入 bug                                                             |

---

## 12. 性能考量

### 12.1 连续 vs 非连续性能对比

| 场景           | 实现路径   | 每元素开销      | 缓存友好性       |
| -------------- | ---------- | --------------- | ---------------- |
| 连续 F-order   | 指针递增   | 1 次指针加法    | 最优（顺序访问） |
| 非连续（切片） | 步长状态机 | 多次乘加运算    | 较差（跳跃访问） |
| 非连续（转置） | 步长状态机 | 多次乘加运算    | 较差（跳跃访问） |
| 广播视图       | 零步长处理 | 条件判断 + 缓存 | 依赖广播维度     |

**性能数据（参考）**:

| 操作         | 连续数组 | 非连续数组（步长=2） | 性能比 |
| ------------ | -------- | -------------------- | ------ |
| 遍历 1M 元素 | ~1ms     | ~3ms                 | 3x     |
| 缓存命中率   | ~95%     | ~60%                 | -      |

### 12.2 复杂度标注

- `Elements::new()`: O(1)，仅初始化状态
- `Elements::next()`（快速路径）: O(1)，指针递增
- `Elements::next()`（慢速路径）: O(ndim)，索引递增
- `AxisIter::next()`: O(1)，子视图切片
- `IndexedIter::next()`: O(ndim)，维护逻辑索引递增与打包

---

## 13. 平台与工程约束

| 项目       | 约束                                                                                |
| ---------- | ----------------------------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性                          |
| crate 结构 | 保持单 crate 结构，不为迭代器单独拆分子 crate                                       |
| 依赖约束   | 不新增第三方依赖；仅复用项目既有核心模块                                            |
| 范围边界   | 当前版本公开范围仅覆盖元素、按轴、按索引迭代；内部多输入遍历由各操作模块直接实现，`Windows` / `LaneIter` 保留为后续议题 |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-14 |
| 1.2.2 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
