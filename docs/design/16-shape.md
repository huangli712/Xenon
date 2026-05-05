# 形状操作模块设计

> 文档编号: 16
> 模块目录: src/shape/
> 任务阶段: Phase 4
> 前置文档: 07-tensor.md, 06-layout.md, 02-dimension.md
> 需求参考: 需求说明书 §6 - §8、§17、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.0 协同基线

本文档示例与论证以下游已修文档为准——`02-dimension.md` v1.2.7、`05-storage.md` v2.0.2、`06-layout.md` v1.3.2、`07-tensor.md` v2.0.4、`26-error.md` **v3.2.0**。任何 §5 / §6 / §10 引用 `26-error` 的字段或变体名（含 `InvalidShape`、`InvalidLayout` 等）时，均以 v3.2.0 为权威。

### 1.1 职责边界

| 职责           | 包含                                             |
| -------------- | ------------------------------------------------ |
| 转置操作       | `transpose()` 交换步长和形状返回只读视图（O(1)） |
| 连续性标志更新 | 转置后按结果 shape/stride 重新计算连续性标志     |
| 形状操作边界   | 当前版本只规范 `transpose()` 这一公开 API        |

| 职责           | 不包含                                           |
| -------------- | ------------------------------------------------ |
| 转置操作       | 其他形状变换（当前版本不提供）                   |
| 连续性标志更新 | pad / repeat / split（当前版本不提供）           |
| 形状操作边界   | `permute_axes()` / `swap_axes()` / `moveaxis()` （当前版本不提供）|
| 未来形状操作   | 其他形状变换与自动推断维度留待后续版本           |

### 1.2 设计原则

| 原则       | 体现                                                           |
| ---------- | -------------------------------------------------------------- |
| 零拷贝优先 | 转置通过调整 shape 和 stride 返回共享底层数据的只读视图        |
| 语义收敛   | 当前版本规范契约只设计 `transpose()`，不在本文扩展其他形状变换 |
| BLAS 友好  | 正确处理转置产生的非连续布局，确保 shape/stride 元数据一致     |
| 维度安全   | 转置仅做轴反转，不改变逻辑元素值与元素总数                     |
| 视图统一性 | 转置结果统一返回借用视图（`ViewRepr`），与 05-storage v2.0.0 §5.11.1 "广播 / 转置 / 切片产生的只读视图统一使用 `ViewRepr`" 规则一致 |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                             |
| -------- | -------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §6 - §8、§17、§28                                                      |
| 范围内   | `transpose()`、轴反转后的 shape / strides / flags 重算，以及零拷贝只读视图语义。 |
| 范围外   | **以下形状变换全部不在当前版本范围内**：`reshape()` / `as_shape()`（任意形状变换）、`squeeze()` / `unsqueeze()`（去/添单位轴）、`permute_axes()` / `swap_axes()` / `moveaxis()`（任意轴重排）、`flatten()` / `ravel()`（多维降一维）、`expand_dims()`（添加新轴）、`broadcast_to()`（属于 `15-broadcast.md` 范围）。本版本仅提供"全轴反转"的 `transpose()`；未来若有需求，需要单独引入对应的设计文档，并在本表中明确移入"范围内"。 |
| 非目标   | 不在本文讨论连续性重排 API、动态维推断或额外形状 DSL。                           |

---

## 3. 文件位置

```
src/shape/
├── mod.rs             # module entry, re-export public traits and functions
└── transpose.rs       # transpose implementation
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/shape/
├── crate::tensor     # TensorBase<S, D>, TensorView<'_, A, D>, .shape(), .strides(), .offset(), .as_ptr()
├── crate::dimension  # Dimension, Ix0~Ix6, IxDyn, pub sealed Reverse trait
├── crate::layout     # LayoutFlags, LayoutState, Strides<D>, compute_layout_flags
└── crate::storage    # ViewRepr<'a, A> (transpose result borrows source storage)
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                          |
| ----------- | --------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `TensorView<'_, A, D>`, `.shape()`, `.strides()`, `.offset()`, `.as_ptr()`            |
| `dimension` | `Dimension`，以及 02-dimension v1.x §5.11 的 `pub trait Reverse: Dimension`（sealed via `Dimension: Sealed`，外部可命名不可实现；用作 `transpose()` 的 `where` 约束） |
| `layout`    | `LayoutFlags`, `LayoutState`, `Strides<D>`，以及 06-layout v1.3 的 `compute_layout_flags::<A, D>(...)`     |
| `storage`   | `ViewRepr<'a, A>`（转置结果统一持有借用视图，参见 05-storage v2.0.0 §5.11.1）                              |
| `error`     | 无新增可恢复错误；规范性 `transpose()` 不走失败返回路径                                                   |

### 4.3 依赖合法性

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.4 依赖方向声明

依赖方向：单向向上。`shape/` 消费 `tensor`、`dimension`、`layout`、`storage` 的 trait 和类型（包括 `ViewRepr`），不被它们依赖。

---

## 5. 公共 API 设计

### 5.1 转置操作

以下为示意性伪实现，非稳定内部结构约定。

````rust,ignore
impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Transpose the array (reverse axis order).
    ///
    /// Returns a view with reversed axis order, zero-copy operation (O(1)).
    /// Equivalent to matrix transpose for 2D arrays.
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::<f64, _>::zeros([2, 3])?;
    /// let b = a.transpose();
    /// assert_eq!(b.shape(), &[3, 2]);
    /// ```
    pub fn transpose(&self) -> TensorView<'_, A, D>
    where
        D: crate::dimension::Reverse, // pub sealed trait from 02-dimension.md §5.11
    {
        // Uses sealed `Reverse` trait from 02-dimension.md §5.11 to produce
        // reversed dimension/strides. The trait is `pub` (named in this
        // public method's `where` clause) but sealed: external crates can
        // name it but cannot implement it for their own types. Every
        // concrete `D` Xenon supports (Ix0..Ix6, IxDyn) implements
        // `Reverse` returning `Self`, so the bound is satisfied for all
        // supported dimensions; it is documentary and lets the body
        // call `.reverse()`.
        //
        // Reverse signature is `fn reverse(self) -> Self`, so `new_shape`
        // and `new_strides` keep the same dimension type `D`; the static
        // rank is preserved (e.g. `Ix2 -> Ix2`), and only the shape/stride
        // VALUES are reversed.
        //
        // Note: calling `.reverse()` on bare slices would not compile —
        // we must go through the owned dimension/Strides types.
        let new_shape: D = self.raw_dim().clone().reverse();
        let new_strides: Strides<D> = self.strides().clone().reverse();

        // For 0D / 1D inputs, transpose is a metadata no-op; layout flags
        // are guaranteed equivalent because shape/stride are equal (0D)
        // or the only axis is unchanged (1D). compute_layout_flags is
        // still well-defined on the same input and produces equivalent
        // flags, so a short-circuit fast path is an optimization, not a
        // correctness requirement.
        let new_flags = compute_layout_flags::<A, D>(&new_shape, &new_strides, self.as_ptr());

        // Internal construction uses TensorView::new_unchecked() or an
        // equivalent constructor. ViewRepr borrows the source storage
        // (see 05-storage v2.0.0 §5.11.1: broadcast / transpose / slice
        // results all use ViewRepr).
        TensorView {
            // Pseudocode: create ViewRepr by borrowing source storage
            storage: ViewRepr::from(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            flags: new_flags,
            // Propagate the ViewMut-derived alias marker from the source view.
            // See 07-tensor.md §5.1 (TensorBase has 6 fields) and §5.3
            // (access_semantics() rule (3): a transpose of TensorViewMut must
            // preserve the SharedReadOnly classification by carrying
            // `derived_from_view_mut = true` forward).
            derived_from_view_mut: self.derived_from_view_mut,
        }
    }

}
````

### 5.2 转置语义

- 根据 `需求说明书 §17`，当前版本形状操作仅支持转置。
- 其他形状变换与连续性驱动的形状重解释不属于本文档覆盖范围，留待后续版本单独设计。
- 当前版本的 `transpose()` 定义为全轴顺序反转（reverse axis order），即 `shape' = shape[::-1]`，`strides' = strides[::-1]`，等价于矩阵转置。`需求说明书 §17` 所述的“轴置换规则”在本版本中等价于全轴反转。一般化的 `permute_axes()` API 不在当前版本范围内；未来若引入，`transpose()` 仍保持为基于 `Reverse` trait（`02-dimension.md §5.11`）的便捷别名，不改变现有契约。
- 若内部通过 unchecked 视图构造返回转置结果，其安全前提为：（1）转置仅重排轴顺序，不改变逻辑访问范围；（2）反转后的 `shape` / `stride` 组合仍满足原 storage 的可见边界约束（由构造期验证保证）；（3）`offset` 保持不变。因此转置无需新的存储分配，视图构造仍落在原验证范围内。

| 属性     | 行为                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------ |
| 零拷贝   | 始终零拷贝（O(1)），仅调整步长和形状                                                                   |
| 形状变化 | `shape[i]` → `shape[ndim-1-i]`（全反转）                                                               |
| 步长变化 | `strides[i]` → `strides[ndim-1-i]`（全反转）                                                           |
| 连续性   | 按结果布局重分类：转置可能产生 `NonContiguous`；实际布局状态由结果 shape/stride 是否满足 F-order 条件决定。若至少一个被交换轴长度为 1，则仍可保留 `F-contiguous`；带零步长且 `product(shape) > 0` 的转置视图仍为 `BroadcastView`（空数组退化的零步长不触发 `BroadcastView`，与 `15-broadcast.md §6.3` / `06-layout.md §5.11` 一致）；0D/1D 保留原布局状态 |
| 偏移量   | 保持不变                                                                                               |
| 1D 数组  | 转置后形状不变（1D 无轴顺序概念）                                                                      |

### 5.3 存储模式降级表

| 源存储模式            | 转置结果存储模式   |
| --------------------- | ------------------ |
| 持有(Owned)           | 只读引用(ViewRepr) |
| 可写引用(ViewMutRepr) | 只读引用(ViewRepr) |
| 只读引用(ViewRepr)    | 只读引用(ViewRepr) |
| 共享只读(ArcRepr)     | 只读引用(ViewRepr) |

- `transpose()` 的返回类型统一固定为 `TensorView<'_, A, D>`，因此结果始终是基于借用的只读视图，只能持有 `ViewRepr`。即使源张量底层使用 `ArcRepr`（共享只读存储），转置结果也不会保留共享所有权语义，而是降级为普通借用视图。

  **设计论证（与 05-storage v2.0.0 §5.11.1 协同）**：

  1. 05-storage v2.0.0 §5.11.1 明文规定："从广播、转置、切片产生的只读视图**统一使用** `ViewRepr`"。本节遵循该统一规则。
  2. 转置只重写 shape / stride / flags 元数据，不复制底层存储。如果转置后保留 `ArcRepr`，意味着多了一个共享 `Arc` 计数副本，但实际收益有限：源张量已持有共享所有权，调用方若需要"转置后仍是 `ArcRepr`"，可显式链式调用 `tensor.transpose().to_owned().into_shared()` 创建新的 owned 共享副本（参见 21-type §5.5 `to_owned()` + 05-storage §5.11 `Owned::into_shared()`）；该路径会复制数据，不是零拷贝恢复原 `ArcRepr`。
  3. 统一返回 `TensorView<'_, A, D>` 让 `transpose()` 的返回类型在所有源存储模式下都相同；用户代码无需为不同存储类型分支处理转置结果，这与 NumPy / ndarray 的 `transpose()` 一致行为相符。
  4. 如果未来确实需要"`ArcRepr.transpose() → ArcRepr`"特性，仍可通过单独的 `inherent impl on ArcTensor` 添加 `transpose_arc()` 等命名 API；但当前版本不暴露，避免一开始就引入"两个 transpose"分裂语义。
- 若源张量为 `ArcRepr`（共享只读），转置后 `storage_kind()` 返回 `StorageKind::View` 而非 `Shared`。这是有意设计：转置结果的生命周期绑定到调用时借用，而不是源张量的共享引用计数。这是允许的收窄：`需求说明书 §17` 只要求结果落在只读引用或共享只读引用范围内，借用视图满足只读引用约束。对后续广播、格式化、线程共享的影响：转置结果的生命周期绑定到原始张量的借用期；如需跨线程持有转置结果，按 (2) 显式创建新的 owned 共享副本。

### 5.4 Good / Bad 对比

以下为示意性伪实现，非稳定内部结构约定。

```rust,ignore
// Good - use transpose() for zero-copy transpose
let a = Tensor::<f64, _>::zeros([1000, 1000])?;
let b = a.transpose();  // O(1), zero-copy
assert_eq!(b.shape(), &[1000, 1000]);

// Bad - manually copy data for transpose (wastes memory and time).
// Note: `[]` indexing is intentionally not implemented (see 17-indexing
// decisions). Use the safe `try_at` / `try_at_mut` APIs explicitly.
let a = Tensor::<f64, _>::zeros([1000, 1000])?;
let mut b = Tensor::<f64, _>::zeros([1000, 1000])?;
for i in 0..1000 {
    for j in 0..1000 {
        let value = *a.try_at(&[i, j]).expect("loop bounds are valid");
        *b.try_at_mut(&[j, i]).expect("loop bounds are valid") = value;
    }
}
```

---

## 6. 内部实现设计

### 6.1 转置布局变化

转置通过直接修改视图的 shape 和 strides 元数据实现，不拷贝数据。具体：交换对应轴的 shape 和 strides 值（即全反转），并按 `06-layout.md` 的 `LayoutState` 重新分类结果布局。对于 `ndim >= 2` 且不含零步长的普通视图，结果通常归类为 `NonContiguous`；若原视图含零步长**且** `product(shape) > 0`，则结果仍属于 `BroadcastView`；空数组退化的零步长不进入 `BroadcastView`（参见 `15-broadcast.md §6.3` / `06-layout.md §5.11`）；对 0D/1D，转置是 no-op，应保留原有 flags 和布局状态。内部通过创建新的 `TensorView`（共享原始存储的只读引用）实现。

```
Source: shape=[2, 3], strides=[1, 2]  (F-order, F-contiguous)
Transpose: shape=[3, 2], strides=[2, 1]  (strides reversed, not F-contiguous)
```

- Xenon 只支持 F-order 布局，不维护单独的行优先连续性状态。
- 转置后连续性须根据结果的shape 与 stride 重新计算。
- 若结果仍满足 F-order 连续条件（如含长度为 1 的轴的转置），则保留`F-contiguous` 标记。
- 广播视图仍按零步长语义分类为 `LayoutState::BroadcastView`。
- 若需恢复连续内存，使用 `to_contiguous()`。

### 6.2 转置后的连续性标志处理

转置操作不引入新步长值，仅交换现有 `usize` stride 顺序。由于 `需求说明书 §7` 明确当前版本不支持负步长布局，因此这里无需讨论负 stride 或相关标志。连续性标志需要按结果布局重算：转置后连续性须根据结果的 shape 与 stride 重新计算；若结果仍满足 F-order 连续条件（如含长度为 1 的轴的转置），则保留 F-contiguous 标记。零步长等其他已存在标志仍按结果布局分类；若源视图为广播视图，且转置后仍存在任一 `stride == 0` 的轴**并且** `product(shape) > 0`，则继续保留 `BroadcastView` 标记（空数组退化情形不触发，参见 `15-broadcast.md §6.3` / `06-layout.md §5.11`）；对 0D/1D 输入，转置是元数据 no-op，应保留原有连续性标志。

```rust,ignore
// Per 06-layout.md §5.12, transpose delegates flag computation to
// compute_layout_flags(shape, strides, ptr).
// Transpose does not change offset, so the logical-first pointer
// remains unchanged.
let new_flags = compute_layout_flags::<A, D>(
    &new_shape,
    &new_strides,
    self.as_ptr(),
);
```

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/shape/mod.rs` 骨架
  - 文件: `src/shape/mod.rs`, `src/shape/transpose.rs`
  - 内容: 模块声明、转置实现文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: tensor、dimension、layout 模块完成
  - 预计: 5 min

### Wave 2: 转置实现

- [ ] **T2**: 实现 `transpose()`
  - 文件: `src/shape/transpose.rs`
  - 内容: `TensorBase::transpose()`
  - 测试: `test_transpose_2d`, `test_transpose_3d`, `test_transpose_contiguity_swap`, `test_transpose_0d_1d_preserves_contiguity`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 测试

- [ ] **T3**: 编写综合测试
  - 文件: `tests/test_shape.rs`
  - 内容: 转置正确性、0D/1D no-op 语义、大数组 O(1) 行为
  - 测试: 覆盖范围内公共 API
  - 前置: T2
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                                |
| -------- | ------------------------ | ------------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `transpose()` 的核心语义与连续性标志重算                       |
| 集成测试 | `tests/`                 | 验证 `shape` 与 `tensor`、`layout`、`index`、`broadcast` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空数组、单元素、大数组和高维转置等边界                          |
| 属性测试 | `tests/property/`        | 验证转置长度保持与数据一致性不变量                                  |

### 8.2 单元测试清单

| 测试函数                                    | 测试内容                                                   | 优先级 |
| ------------------------------------------- | ---------------------------------------------------------- | ------ |
| `test_transpose_2d`                         | `[2,3]` → `[3,2]`，验证 shape 和数据                       | 高     |
| `test_transpose_3d`                         | `[2,3,4]` → `[4,3,2]`，验证轴反转                          | 高     |
| `test_transpose_not_f_contiguous`           | 典型 2D F-contiguous 转置后 `is_f_contiguous()` 返回 false | 高     |
| `test_transpose_1d_noop`                    | 1D 数组转置后形状不变                                      | 中     |
| `test_transpose_0d_noop`                    | 0D 标量转置后不变                                          | 中     |
| `test_transpose_0d_1d_preserves_contiguity` | 0D/1D 转置保留原连续性标志                                 | 高     |
| `test_transpose_broadcast_view_keeps_flag`  | 广播视图转置后零步长仍保留 `BroadcastView`                 | 中     |
| `test_transpose_owned_returns_view_kind`    | Owned 张量转置后 `storage_kind()` 返回 `StorageKind::View` | 中     |
| `test_transpose_view_mut_returns_view_kind` | ViewMut 张量转置后 `storage_kind()` 返回 `StorageKind::View` | 中   |
| `test_transpose_arc_tensor_returns_view_kind` | ArcRepr 张量转置后 `storage_kind()` 返回 `StorageKind::View` | 高 |

### 8.3 边界测试场景

| 场景                       | 预期行为                            |
| -------------------------- | ----------------------------------- |
| 空数组 `shape=[0, 3]`      | 转置到 `[3, 0]`，逻辑元素值不变     |
| 单元素 `shape=[1, 1]`      | 转置后仍可正确访问唯一元素          |
| 大数组 `[3162, 3162]` 转置 | O(1)，不拷贝                        |
| 高维数组 `[2,3,4,5]` 转置  | 轴顺序完全反转                      |
| 接近静态维度上限的高维转置 | 静态维类型下仍按全轴反转规则工作    |
| `IxDyn` rank 5+ 转置       | 动态高 rank 仍正确反转 shape/stride |

### 8.4 属性测试不变量

| 不变量                              | 测试方法           |
| ----------------------------------- | ------------------ |
| `transpose().len() == tensor.len()` | 随机形状           |
| 转置后数据不变                      | 转置前后逐元素对比 |
| `t.transpose().transpose()` ≡ `t`   | shape、strides 完全一致 |

### 8.5 集成测试

| 测试文件              | 测试内容                                                             |
| --------------------- | -------------------------------------------------------------------- |
| `tests/test_shape.rs` | `transpose()` 与 `tensor`、`layout`、`index`、`broadcast` 的协同路径 |

### 8.6 Feature gate / 配置测试

| 配置              | 验证点                                        |
| ----------------- | --------------------------------------------- |
| 默认配置          | `transpose()` 的零拷贝与 flags 语义保持一致。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。         |

### 8.7 类型边界 / 编译期测试

| 场景                                     | 测试方式                     |
| ---------------------------------------- | ---------------------------- |
| 轴反转辅助逻辑对所有 `D: Dimension` 生效 | 编译期测试与运行时断言。     |
| 0D / 1D 输入保持原维度类型               | 编译期签名检查与运行时断言。 |
| 其他形状变换不属于当前 API               | API 缺失断言。               |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块    | 接口/类型                   | 约定                                            |
| ------------------- | ----------- | --------------------------- | ----------------------------------------------- |
| `shape → tensor`    | `tensor`    | `TensorBase` / `TensorView` | 依赖张量结构与视图创建入口                      |
| `shape → dimension` | `dimension` | `Dimension` trait           | 使用维度 trait 完成形状变换与校验               |
| `shape → layout`    | `layout`    | 连续性与步长查询            | 转置后按结果步长重算连续性标志                  |
| `shape ← broadcast` | `broadcast` | 广播视图语义                | 广播视图因零步长而只读且非连续，转置后仍保持共享底层数据的只读语义 |
| `shape ← index`     | `index`     | 切片结果视图                | 索引/切片结果可继续参与转置；共享底层数据时仍只返回只读视图 |

### 9.2 数据流描述

```text
User calls transpose()
    │
    ├── shape rewrites shape + strides + flags by reversing axes
    ├── the result shares the original storage and stays read-only
    └── the new view can be consumed by index / iter / math paths
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------- |
| Recoverable error | 不适用；当前范围内 `transpose()` 不返回模块级可恢复错误。                                |
| Panic             | 不适用；公开 API 不定义额外 panic 分支。                                                 |
| 路径一致性        | 当前仅有元数据重写路径；无 SIMD / 并行分支，0D/1D 输入与高维输入都必须保持同一转置契约。 |
| 容差边界          | 不适用。                                                                                 |

---

## 11. 设计决策记录

### 决策 1：转置不拷贝数据

| 属性     | 值                                           |
| -------- | -------------------------------------------- |
| 决策     | 转置通过交换步长和形状实现，不拷贝数据       |
| 理由     | O(1) 操作；内存效率高；与 ndarray/NumPy 一致 |
| 替代方案 | 拷贝数据转置 — 放弃，O(n) 开销不必要         |

### 决策 2：当前版本不设计其他形状变换

| 属性     | 值                                                        |
| -------- | --------------------------------------------------------- |
| 决策     | 其他形状变换留待后续版本单独设计，不在当前文档承诺        |
| 理由     | `需求说明书 §17` 明确当前版本形状操作仅支持转置           |
| 替代方案 | 在本阶段继续保留其他形状变换设计 — 放弃，超出当前需求范围 |

### 决策 3：当前版本交付边界仅包含 `transpose()`

| 属性     | 值                                                                                |
| -------- | --------------------------------------------------------------------------------- |
| 决策     | 仅把 `transpose()` 纳入当前版本交付                                               |
| 理由     | `需求说明书 §17` 明确当前版本只要求转置操作本身，文档不应把别名扩写成当前交付承诺 |
| 替代方案 | 在当前版本同时承诺其他形状操作 — 放弃，超出规范性 API 边界                        |

### 决策 4：`transpose()` 不保留 `ArcRepr` 共享所有权语义

| 属性     | 值                                                                                                                                                                                                                                                                                            |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `transpose()` 始终返回 `TensorView<'_, A, D>`；无论源存储是 `Owned`、`ViewMutRepr`、`ViewRepr` 还是 `ArcRepr`，结果统一使用只读借用视图 `ViewRepr`                                                                                                                                            |
| 理由     | (1) 与 05-storage v2.0.0 §5.11.1 的"广播 / 转置 / 切片产生的只读视图统一使用 `ViewRepr`"协同；(2) 转置只重写元数据，不复制存储；显式恢复共享所有权可通过 `tensor.transpose().to_owned().into_shared()` 链式调用（参见 21-type §5.5、05-storage §5.11）；(3) 统一返回类型让用户代码无需对存储模式分支处理 |
| 替代方案 | 让 `ArcRepr` 输入返回保留共享所有权的新视图类型 — 放弃，会破坏 05-storage v2.0.0 §5.11.1 的统一规则，并引入"两个 transpose"的分裂语义                                                                                                                                                          |
| 未来扩展 | 若确实需要"`ArcRepr.transpose() → ArcRepr`"特性，可通过单独的 `inherent impl on ArcTensor` 添加 `transpose_arc()` 等命名 API；当前版本不暴露                                                                                                                                                |

---

## 12. 性能考量

### 12.1 复杂度

| 操作          | 连续输入 | 非连续输入 |
| ------------- | -------- | ---------- |
| `transpose()` | O(1)     | O(1)       |

### 12.2 内存

| 操作          | 内存分配 | 数据拷贝 |
| ------------- | -------- | -------- |
| `transpose()` | 无       | 无       |

### 12.3 缓存行为

| 场景                                    | 缓存友好性 | 说明                                                    |
| --------------------------------------- | ---------- | ------------------------------------------------------- |
| 典型 F-contiguous 转置后遍历            | 较差       | 步长反转，常见情况下会出现内存跳跃访问                  |
| 含长度为 1 轴的 F-contiguous 转置后遍历 | 中         | 结果仍可能保持 `F-contiguous`，缓存行为取决于具体 shape |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                   |
| ---------- | ---------------------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径            |
| MSRV       | Rust 1.85+                                                             |
| 单 crate   | `shape` 设计保持在现有 crate 内，不引入额外 crate                      |
| SemVer     | 当前调整是收敛文档范围到规范性 `transpose()` API，不新增超范围公开 API |
| 最小依赖   | 本模块不新增第三方依赖                                                 |

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
| 1.1.2 | 2026-04-14 |
| 1.1.3 | 2026-04-15 |
| 1.1.4 | 2026-04-15 |
| 2.0.0 | 2026-05-02 |
| 2.0.1 | 2026-05-03 |
| 2.0.2 | 2026-05-04 |

### v2.0.2 (2026-05-04) — patch fix: HAS_ZERO_STRIDE 规则引用从 `§5.12` 更正为 `§5.11`

- §5.2 转置语义表连续性行、§6.1 转置布局变化段、§6.2 连续性标志处理段：三处 HAS_ZERO_STRIDE 规则引用从 `06-layout.md §5.12` 更正为 `§5.11`（与 06-layout v1.3.2 权威拆分一致）。
- §6.2 伪代码注释区 `compute_layout_flags` 引用保持 `§5.12` 不变——该引用指向计算入口，非 HAS_ZERO_STRIDE 规则定义。

### v2.0.1 (2026-05-03) — Low 级文档修复

- §1.1：修复职责边界表格中的格式空格。
- §4.4：依赖方向声明补充 `storage` / `ViewRepr`。
- §5.3：明确 `tensor.transpose().to_owned().into_shared()` 是显式创建新的 owned 共享副本且会复制数据。

### v2.0.0 (2026-05-02) — 协同与一致性更新（公开 API 形态保持兼容）

> 本版本是与 02-dimension v1.x、05-storage v2.0.0、06-layout v1.3、26-error v3.0.0 协同的内部一致性更新。`transpose()` 公开签名不变；仅文档层强化论证 + 依赖表补全 + 伪实现注释更新。

- §1.2 设计原则：新增"视图统一性"行，明确与 05-storage v2.0.0 §5.11.1 的统一规则协同。
- §3 文件位置 / §4.1 依赖图：补 `crate::storage` (`ViewRepr`) 依赖；说明 `Reverse` trait（**v2.0.1 起**为 `pub` sealed trait via `Dimension: Sealed`，外部可命名 / 不可实现；早期 v2.0.0 草案曾设计为 `pub(crate)`，但 `transpose()` 的 `where D: Reverse` 不能引用 crate-private trait，故升级到 sealed pub）。
- §4.2 类型级依赖：明确引用 02-dimension §5.11 `pub trait Reverse: Dimension`（sealed）；明确 06-layout v1.3 `compute_layout_flags`；补 `crate::storage` 行说明 `ViewRepr<'a, A>` 的来源（05-storage v2.0.0 §5.11.1）。
- §5.1 伪实现：注释更新——`Reverse` 在 `transpose()` 的 `where` 子句中可见（`pub` sealed），但实际上对所有受支持的维度类型（`Ix0..Ix6`、`IxDyn`）都已实现，因此对调用方是透明的；`fn reverse(self) -> Self` 同类型反转，不涉及关联类型。新增 0D / 1D 路径的等价性说明（`compute_layout_flags` 在相同输入下产生等价 flags，短路是优化非正确性要求）；引用 05-storage 统一规则。
- §5.3 存储模式降级表后的 ArcRepr 降级说明：完全重写，新增四点设计论证：(1) 与 05-storage v2.0.0 §5.11.1 协同；(2) 显式恢复共享所有权的链式调用路径 `tensor.transpose().to_owned().into_shared()`；(3) 统一返回类型避免用户代码分支；(4) 未来若需要 `ArcRepr.transpose() → ArcRepr` 可通过单独命名 API 添加。
- §5.4 Bad 示例：`a.get(&[i, j])` / `b.get_mut(&[j, i])` 改为安全索引 `try_at` / `try_at_mut`（与 17-indexing 决策 7 "公开安全索引收敛为 try_at / try_at_mut" 一致）；构造路径补 `?`。
- §11 决策 4 完全重写：补充与 05-storage v2.0.0 §5.11.1 协同的论证；新增"未来扩展"行说明若需特殊化 ArcRepr.transpose() 可通过单独命名 API 添加，避免破坏统一规则。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
