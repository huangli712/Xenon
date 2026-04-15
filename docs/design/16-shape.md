# 形状操作模块设计

> 文档编号: 16 | 模块: `src/shape/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `06-layout.md`
> 需求参考: 需求说明书 §17
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                                     | 不包含                                                       |
| -------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| 转置操作       | `transpose()` 交换步长和形状返回只读视图（O(1)）         | 其他形状变换（当前版本不提供）                               |
| 连续性标志更新 | 转置后按结果 shape/stride 重新计算连续性标志             | pad / repeat / split（当前版本不提供）                       |
| 非规范便捷别名 | 若后续提供 `t()` 作为 `transpose()` 简写，也仅能作为非规范便捷别名，不纳入当前版本交付 | `permute_axes()` / `swap_axes()` / `moveaxis()` 留待后续版本 |
| 未来形状操作   | —                                                        | 其他形状变换与自动推断维度留待后续版本                       |

### 1.2 设计原则

| 原则       | 体现                                                            |
| ---------- | --------------------------------------------------------------- |
| 零拷贝优先 | 转置通过调整 shape 和 stride 返回共享底层数据的只读视图         |
| 语义收敛   | 当前版本规范契约只设计 `transpose()`，不在本文扩展其他形状变换 |
| BLAS 友好  | 正确处理转置产生的非连续布局，确保 shape/stride 元数据一致      |
| 维度安全   | 转置仅做轴反转，不改变逻辑元素值与元素总数                      |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent from layout; owned by tensor and consumed via layout results)
L4: tensor (depends on storage, dimension)
L5: broadcast (depends on tensor, dimension)
L6: shape  <- current module
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §17 |
| 范围内   | `transpose()`、轴反转后的 shape / strides / flags 重算，以及零拷贝只读视图语义。 |
| 范围外   | 其他形状变换。 |
| 非目标   | 不在本文讨论连续性重排 API、动态维推断或额外形状 DSL。 |

---

## 3. 文件位置

```
src/shape/
├── mod.rs             # module entry, re-export public traits and functions
└── transpose.rs       # transpose implementation
```

文件划分理由：当前版本仅支持转置，因此保留单一实现文件即可覆盖范围内能力。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │    tensor    │
                    │ TensorBase   │
                    └──────┬───────┘
                           │ uses
              ┌────────────┼────────────────┐
              │   shape                     │
              │   transpose.rs              │
              └──┬───────────┬──────────────┘
                 │ uses      │ uses
          ┌──────▼───┐ ┌─────▼────────────┐
          │ dimension│ │ layout           │
          │ Dimension│ │ LayoutFlags      │
          │ Ix0~IxDyn│ │ LayoutState      │
          └──────────┘ └──────────────────┘
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `TensorView`, `Tensor<A, D>`, `.shape()`, `.strides()`, `.offset()`，参见 `07-tensor.md` §5 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `RemoveAxis`, `IntoDimension`，参见 `02-dimension.md` §3                     |
| `layout`    | `LayoutFlags`, `LayoutState`, `Strides<D>`，参见 `06-layout.md` §3, §4                                          |
| `error`     | 无新增可恢复错误；规范性 `transpose()` 不走失败返回路径                                                        |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `shape/` 消费 `tensor`、`dimension`、`layout` 的 trait 和类型，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 转置操作

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
    /// let a = Tensor::<f64, _>::zeros([2, 3]);
    /// let b = a.transpose();
    /// assert_eq!(b.shape(), &[3, 2]);
    /// ```
    pub fn transpose(&self) -> TensorView<'_, A, D> {
        let new_shape = reverse_axes(self.shape());
        let new_strides = reverse_axes(self.strides());
        let new_flags = update_flags_for_transpose(self.flags(), new_shape.slice(), new_strides.as_slice());

        // actual construction uses TensorView::new_unchecked() or similar
        // internal constructor, see 07-tensor.md
        TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            flags: new_flags,
        }
    }

}
````

> **别名说明：** `t()` 不纳入当前版本正式 API、任务拆分或测试交付；若未来提供，也只能作为非规范语法别名。

#### 5.1.1 转置语义

| 属性     | 行为                                              |
| -------- | ------------------------------------------------- |
| 零拷贝   | 始终零拷贝（O(1)），仅调整步长和形状              |
| 形状变化 | `shape[i]` → `shape[ndim-1-i]`（全反转）          |
| 步长变化 | `strides[i]` → `strides[ndim-1-i]`（全反转）      |
| 连续性   | 按结果布局重分类：转置可能产生 `NonContiguous`；实际布局状态由结果 shape/stride 是否满足 F-order 条件决定。若至少一个被交换轴长度为 1，则仍可保留 `F-contiguous`；带零步长的转置视图仍为 `BroadcastView`；0D/1D 保留原布局状态 |
| 偏移量   | 保持不变                                          |
| 1D 数组  | 转置后形状不变（1D 无轴顺序概念）                 |

#### 5.1.2 Good / Bad 对比

```rust
// Good - use transpose() for zero-copy transpose
let a = Tensor::<f64, _>::zeros([1000, 1000]);
let b = a.transpose();  // O(1), zero-copy
assert_eq!(b.shape(), &[1000, 1000]);

// Bad - manually copy data for transpose (wastes memory and time)
let a = Tensor::<f64, _>::zeros([1000, 1000]);
let mut b = Tensor::<f64, _>::zeros([1000, 1000]);
for i in 0..1000 {
    for j in 0..1000 {
        b[[j, i]] = a[[i, j]];  // O(n^2) copy, forbidden
    }
}
```

---

### 5.2 范围边界说明

> **范围决策：** 根据 `require.md` §17，当前版本形状操作仅支持转置。
> 其他形状变换与连续性驱动的形状重解释不属于本文档覆盖范围，留待后续版本单独设计。

---

## 6. 内部实现设计

### 6.1 转置布局变化

转置通过直接修改视图的 shape 和 strides 元数据实现，不拷贝数据。具体：交换对应轴的 shape 和 strides 值（即全反转），并按 `06-layout.md` 的 `LayoutState` 重新分类结果布局。对于 `ndim >= 2` 且不含零步长的普通视图，结果通常归类为 `NonContiguous`；若原视图含零步长，则结果仍属于 `BroadcastView`；对 0D/1D，转置是 no-op，应保留原有 flags 和布局状态。内部通过创建新的 `TensorView`（共享原始存储的只读引用）实现。

```
Source: shape=[2, 3], strides=[1, 2]  (F-order, F-contiguous)
Transpose: shape=[3, 2], strides=[2, 1]  (strides reversed, not F-contiguous)
```

> **注意**：Xenon 只支持 F-order 布局，不维护单独的行优先连续性状态。转置后连续性须根据结果的
> shape 与 stride 重新计算；若结果仍满足 F-order 连续条件（如含长度为 1 的轴的转置），则保留
> `F-contiguous` 标记；广播视图仍按零步长语义分类为 `LayoutState::BroadcastView`。
> 若需恢复连续内存，使用 `to_contiguous()`。

### 6.2 转置后的连续性标志处理

转置操作不引入新步长值，仅交换现有 `usize` stride 顺序。由于 `require.md` §7 明确当前版本不支持负步长布局，因此这里无需讨论负 stride 或相关标志。连续性标志需要按结果布局重算：转置后连续性须根据结果的 shape 与 stride 重新计算；若结果仍满足 F-order 连续条件（如含长度为 1 的轴的转置），则保留 F-contiguous 标记。零步长等其他已存在标志仍按结果布局分类；对 0D/1D 输入，转置是元数据 no-op，应保留原有连续性标志。

```rust
fn update_flags_for_transpose(
    source_flags: LayoutFlags,
    new_shape: &[usize],
    new_strides: &[usize],
) -> LayoutFlags {
    recompute_layout_flags(source_flags, new_shape, new_strides)
}
```

### 6.3 范围外能力记录

其他形状变换 API 与自动推断维度当前均不在范围内，因此本文不再为它们设计连续性检查逻辑、错误语义或实现任务。

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
  - 内容: `TensorBase::transpose()`, `LayoutFlags::update_for_transpose()`
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

### 并行执行图

```
Wave 1: [T1]
            │
Wave 2: [T2]
            │
Wave 3: [T3]
```

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

| 测试函数                                    | 测试内容                                           | 优先级 |
| ------------------------------------------- | -------------------------------------------------- | ------ |
| `test_transpose_2d`                         | `[2,3]` → `[3,2]`，验证 shape 和数据               | 高     |
| `test_transpose_3d`                         | `[2,3,4]` → `[4,3,2]`，验证轴反转                  | 高     |
| `test_transpose_not_f_contiguous`           | 典型 2D F-contiguous 转置后 `is_f_contiguous()` 返回 false | 高     |
| `test_transpose_1d_noop`                    | 1D 数组转置后形状不变                              | 中     |
| `test_transpose_0d_noop`                    | 0D 标量转置后不变                                  | 中     |
| `test_transpose_0d_1d_preserves_contiguity` | 0D/1D 转置保留原连续性标志                         | 高     |

### 8.3 边界测试场景

| 场景                       | 预期行为                        |
| -------------------------- | ------------------------------- |
| 空数组 `shape=[0, 3]`      | 转置到 `[3, 0]`，逻辑元素值不变 |
| 单元素 `shape=[1, 1]`      | 转置后仍可正确访问唯一元素      |
| 大数组 `[1000, 1000]` 转置 | O(1)，不拷贝                    |
| 高维数组 `[2,3,4,5]` 转置  | 轴顺序完全反转                  |

### 8.4 属性测试不变量

| 不变量                              | 测试方法           |
| ----------------------------------- | ------------------ |
| `transpose().len() == tensor.len()` | 随机形状           |
| 转置后数据不变                      | 转置前后逐元素对比 |

### 8.5 集成测试

| 测试文件              | 测试内容                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `tests/test_shape.rs` | `transpose()` 与 `tensor`、`layout`、`index`、`broadcast` 的协同路径 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `transpose()` 的零拷贝与 flags 语义保持一致。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| 轴反转辅助逻辑对所有 `D: Dimension` 生效 | 编译期测试与运行时断言。 |
| 0D / 1D 输入保持原维度类型 | 编译期签名检查与运行时断言。 |
| 其他形状变换不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块    | 接口/类型                   | 约定                                                                                          |
| ------------------- | ----------- | --------------------------- | --------------------------------------------------------------------------------------------- |
| `shape → tensor`    | `tensor`    | `TensorBase` / `TensorView` | 依赖张量结构与视图创建入口，参见 `07-tensor.md` §5                                            |
| `shape → dimension` | `dimension` | `Dimension` trait           | 使用维度 trait 完成形状变换与校验，参见 `02-dimension.md` §3                                  |
| `shape → layout`    | `layout`    | 连续性与步长查询            | 转置后按结果步长重算连续性标志，参见 `06-layout.md` §3                                        |
| `shape ← broadcast` | `broadcast` | 广播视图语义                | 广播视图因零步长而只读且非连续，转置后仍保持共享底层数据的只读语义，参见 `15-broadcast.md` §5 |
| `shape ← index`     | `index`     | 切片结果视图                | 索引/切片结果可继续参与转置；共享底层数据时仍只返回只读视图                                   |

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

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 不适用；当前范围内 `transpose()` 不返回模块级可恢复错误。 |
| Panic | 不适用；公开 API 不定义额外 panic 分支。 |
| 路径一致性 | 当前仅有元数据重写路径；无 SIMD / 并行分支，0D/1D 输入与高维输入都必须保持同一转置契约。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### 决策 1：转置不拷贝数据

| 属性     | 值                                           |
| -------- | -------------------------------------------- |
| 决策     | 转置通过交换步长和形状实现，不拷贝数据       |
| 理由     | O(1) 操作；内存效率高；与 ndarray/NumPy 一致 |
| 替代方案 | 拷贝数据转置 — 放弃，O(n) 开销不必要         |

### 决策 2：当前版本不设计其他形状变换

| 属性     | 值                                                                  |
| -------- | ------------------------------------------------------------------- |
| 决策     | 其他形状变换留待后续版本单独设计，不在当前文档承诺 |
| 理由     | `require.md` §17 明确当前版本形状操作仅支持转置                     |
| 替代方案 | 在本阶段继续保留其他形状变换设计 — 放弃，超出当前需求范围          |

### 决策 3：当前版本交付边界仅包含 `transpose()`

| 属性     | 值                                                         |
| -------- | ---------------------------------------------------------- |
| 决策     | Phase 4 仅把 `transpose()` 纳入当前版本交付；`t()` 仅可作为后续可选的非规范便捷别名 |
| 理由     | `require.md` §17 明确当前版本只要求转置操作本身，文档不应把别名扩写成当前交付承诺 |
| 替代方案 | 在当前版本同时承诺 `t()` 或其他形状操作 — 放弃，超出规范性 API 边界                |

---

## 12. 性能描述

### 12.1 复杂度

| 操作          | 连续输入 | 非连续输入 |
| ------------- | -------- | ---------- |
| `transpose()` | O(1)     | O(1)       |

### 12.2 内存

| 操作          | 内存分配 | 数据拷贝 |
| ------------- | -------- | -------- |
| `transpose()` | 无       | 无       |

### 12.3 缓存行为

| 场景                                | 缓存友好性 | 说明                                                  |
| ----------------------------------- | ---------- | ----------------------------------------------------- |
| 典型 F-contiguous 转置后遍历        | 较差       | 步长反转，常见情况下会出现内存跳跃访问                |
| 含长度为 1 轴的 F-contiguous 转置后遍历 | 中         | 结果仍可能保持 `F-contiguous`，缓存行为取决于具体 shape |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                 |
| ---------- | -------------------------------------------------------------------- |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径          |
| 单 crate   | `shape` 设计保持在现有 crate 内，不引入额外 crate                    |
| SemVer     | 当前调整是收敛文档范围到规范性 `transpose()` API，不新增超范围公开 API |
| 最小依赖   | 本模块不新增第三方依赖                                               |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
