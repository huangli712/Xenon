# 归约运算模块设计

> 文档编号: 13 | 模块: `src/reduction/` | 阶段: Phase 4
> 前置文档: `02-dimension.md`, `03-element.md`, `07-tensor.md`, `09-parallel.md`, `26-error.md`
> 需求参考: 需求说明书 §9.1, §9.2, §9.3, §14, §27, §28.2, §28.3, §28.4, §28.5
> 范围声明: 范围内

---

## 1. 模块定位/概述

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
| ---- | ---- | ------ |
| sum 归约 | 全局 `sum`、沿轴 `sum_axis`、保留轴版本 `sum_axis_keepdims` | `mean`、`var`、`prod`、`min`、`max`、`argmin`、`argmax` |
| 数值语义 | 整数 checked arithmetic、浮点 `NaN` 传播、空数组返回加法单位元 | 自动类型提升、近似归约、重排求和顺序 |
| 执行路径 | 标量基线路径，以及仅在满足 `require.md` §28.3 数值语义约束时启用的 SIMD / 并行分派 | 为追求吞吐而放宽结果一致性的优化路径 |
| 错误边界 | 轴越界返回 `XenonError::InvalidAxis`；整数溢出 panic | 为 axis 错误使用 `InvalidArgument` |

> **注意**：当前版本归约模块只支持 `sum` 家族，不扩展到其它归约操作。

### 1.2 设计原则

| 原则 | 体现 |
| ---- | ---- |
| 最小范围 | 公开 API 只覆盖 `sum`、`sum_axis`、`sum_axis_keepdims`。 |
| 语义优先 | 空数组返回加法单位元；浮点遵循 IEEE 754；整数溢出按不可恢复算术域错误处理。 |
| 路径一致性 | SIMD 与并行只在满足 `require.md` §28.3 定义的数值语义约束时参与，否则回退标量。 |
| 错误统一 | 所有 axis 越界都统一为 `XenonError::InvalidAxis`，并携带 `operation`、`axis`、`ndim`、`shape`。 |

### 1.3 在架构中的位置

```text
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; tensor owns storage and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: iter, simd, parallel
L6: reduction  <- current module
```

---

## 2. 需求映射与范围约束

| 类型 | 内容 |
| ---- | ---- |
| 需求映射 | 需求说明书 §9.1、§9.2、§9.3、§14、§27、§28.2、§28.3、§28.4、§28.5 |
| 范围内 | 全局 `sum`、沿轴 `sum_axis`、`sum_axis_keepdims`、空数组零语义、整数 checked arithmetic、浮点/复数 IEEE 754 语义、可选 SIMD/并行回退规则。 |
| 范围外 | `mean`、`var`、`prod`、`min`、`max`、`argmin`、`argmax`、自定义 reducer、误差补偿求和。 |
| 非目标 | 不新增第三方数值依赖，不改变 F-order 布局前提，不把 axis 错误扩展成额外局部错误类型。 |

### 2.1 范围约束说明

- 仅以下元素类型参与 `sum` 归约：`i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>`。
- `bool` 不参与 `sum` 归约。
- 轴索引越界时须返回可恢复错误；错误类型统一为 `XenonError::InvalidAxis`。
- 有符号整数归约溢出属于不可恢复错误，必须 panic。
- 当输入为空，或沿轴归约的被归约轴长度为 `0` 时，结果须为对应元素类型的加法单位元。

---

## 3. 文件位置

```text
src/reduction/
├── mod.rs              # module entry and public re-exports
└── sum.rs              # sum / sum_axis / sum_axis_keepdims implementations
```

双文件设计理由：`mod.rs` 仅承担模块边界与导出职责，`sum.rs` 集中承载当前版本唯一的归约族。该模块保持最小语义层，SIMD 与并行优化由 `simd/`、`parallel/` 提供能力边界，但不在此模块内扩展新的归约种类。

---

## 4. 依赖关系

### 4.1 依赖图

```text
src/reduction/
├── mod.rs
│   └── crate::reduction::sum
└── sum.rs
    ├── crate::tensor        # TensorBase<S, D>, Tensor<A, D>, shape/ndim helpers
    ├── crate::dimension     # Axis, Dimension, runtime axis projection helpers
    ├── crate::element       # Numeric, CheckedAdd, ComplexScalar
    ├── crate::error         # XenonError::InvalidAxis
    ├── crate::simd (opt.)   # Optional sum kernels when semantics match scalar path
    └── crate::parallel (opt.) # Optional parallel dispatch when semantics match scalar path
```

### 4.2 类型级依赖

| 来源模块 | 使用的类型/trait |
| -------- | ---------------- |
| `tensor` | `TensorBase<S, D>`、`Tensor<A, D>`、`.shape()`、`.ndim()`、`.iter()`、`.indexed_iter()`、结果张量构造接口 |
| `dimension` | `Axis`、`Dimension`、运行时 axis/shape 投影辅助 |
| `element` | `Numeric`、`CheckedAdd`、`ComplexScalar`、`A::zero()` |
| `error` | `XenonError::InvalidAxis` |
| `simd`（可选） | 仅在可证明与标量累加顺序和结果语义一致时参与 `sum` 实现 |
| `parallel`（可选） | 仅在可证明与标量结果一致时参与分派，并遵守无嵌套并行约束 |

### 4.3 依赖方向

> **依赖方向：单向向上。** `reduction` 仅消费 `tensor`、`dimension`、`element`、`error` 以及项目内可选的 `simd` / `parallel` 能力，不被这些基础模块反向依赖。

### 4.4 依赖合法性与新增依赖说明

| 项目 | 说明 |
| ---- | ---- |
| 新增第三方依赖 | 无；仅可使用需求中已允许的可选依赖 `pulp`、`rayon` 所对应的项目内能力边界。 |
| 合法性结论 | 合法；符合最小依赖、单 crate、`std` 环境约束。 |
| 替代方案 | 不适用；当前范围内无需新增额外归约框架或数值库。 |

---

## 5. 公共 API 设计

### 5.1 核心接口

````rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Returns the sum of all logical elements.
    ///
    /// Empty arrays return the additive identity `A::zero()`.
    /// Rank-0 (scalar) tensors return their single element.
    /// Integer overflow is unrecoverable and must panic.
    pub fn sum(&self) -> A;

    /// Reduces along `axis` and removes that axis from the output shape.
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    /// Rank-0 inputs use the same runtime error contract.
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<A, D::Smaller>, XenonError>;

    /// Reduces along `axis` and keeps the reduced axis with length 1.
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    /// Rank-0 inputs use the same runtime error contract.
    pub fn sum_axis_keepdims(&self, axis: Axis) -> Result<Tensor<A, D>, XenonError>;
}
````

> **0D 轴归约说明：** `sum_axis` 的公开入口必须先做运行时 axis 校验，因此对 `ndim == 0` 的输入（尤其是 `IxDyn([])`）必须返回 `XenonError::InvalidAxis { operation, axis, ndim: 0, shape: vec![] }`，而不是依赖编译期拒绝。合法输入在完成 axis 校验后再进入降维路径，公开返回类型保持 `Tensor<A, D::Smaller>`；`sum_axis_keepdims` 仍保持原 rank。

### 5.2 对外错误契约

沿轴归约的 axis 越界错误必须统一为：

```rust
XenonError::InvalidAxis {
    operation: "sum_axis",
    axis: axis.index(),
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
}
```

```rust
XenonError::InvalidAxis {
    operation: "sum_axis_keepdims",
    axis: axis.index(),
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
}
```

> **设计决策：** 对 `sum_axis` 与 `sum_axis_keepdims`，axis 越界只允许使用 `XenonError::InvalidAxis`。不得再使用 `InvalidArgument` 表达该类错误。

### 5.3 Good / Bad 对比示例

```rust,ignore
// Good - handle recoverable axis errors explicitly
let x = Tensor2::<f64>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let reduced = x.sum_axis(Axis(1))?;
assert_eq!(reduced.shape(), &[2]);

// Good - keepdims preserves rank and sets the reduced axis length to 1
let kept = x.sum_axis_keepdims(Axis(1))?;
assert_eq!(kept.shape(), &[2, 1]);

// Good - empty array sum returns additive identity
let empty = Tensor1::<i32>::zeros([0]);
assert_eq!(empty.sum(), 0);

// Bad - do not document axis errors as InvalidArgument
// Err(XenonError::InvalidArgument { operation: "sum_axis", argument: "axis", .. })

// Bad - do not replace integer overflow panic with a recoverable error
// return Err(XenonError::InvalidAxis { .. });
```

---

## 6. 内部实现设计

### 6.1 核心不变量

| 不变量 | 说明 |
| ------ | ---- |
| 归约族范围 | 当前版本只实现 `sum`，不为其它归约预留公开入口。 |
| 空输入语义 | `sum()` 对空数组返回 `A::zero()`；沿轴归约的被归约轴长度为 `0` 时，对每个输出槽写入 `A::zero()`。 |
| axis 校验顺序 | `sum_axis()` 与 `sum_axis_keepdims()` 必须先校验 `axis < ndim`，再执行归约。 |
| 整数语义 | `i32` / `i64` 累加使用 checked arithmetic，任何溢出立即 panic。 |
| 浮点/复数语义 | `f32` / `f64` / `Complex<_>` 遵循标量加法语义，`NaN` 按 IEEE 754 自动传播。 |
| 执行路径约束 | SIMD / 并行若无法满足 `require.md` §28.3 数值语义约束，则必须回退标量。 |
| 布局前提 | 算法面向 Xenon 当前支持的 F-order 语义和合法 stride 视图，不得引入 C-order 假设。 |

### 6.2 算法描述

```text
sum(tensor):
    acc = A::zero()
    for each logical element x in tensor:
        acc = add_with_type_semantics(acc, x)
    return acc

sum_axis(tensor, axis):
    1. Validate axis against tensor.ndim().
    2. Compute the output shape by removing the target axis.
    3. Allocate the output tensor with zeros.
    4. Iterate all logical input elements.
    5. Map each input index to its output index with the target axis removed.
    6. Accumulate into the corresponding output slot using type-specific add semantics.
    7. Return the reduced tensor after runtime shape projection.

sum_axis_keepdims(tensor, axis):
    1. Validate axis against tensor.ndim().
    2. Clone the input shape.
    3. Set result_shape[axis] = 1.
    4. Allocate the output tensor with zeros.
    5. Iterate all logical input elements.
    6. Map each input index to the keepdims output index by forcing the reduced axis to 0.
    7. Accumulate using the same type-specific add semantics.
    8. Return Tensor<A, D> with the reduced axis length preserved as 1.
```

> **0D 张量语义**：`sum()` 对 rank-0 张量（标量）返回其唯一元素，与 `A::zero()` 语义无关。

### 6.3 类型分派与回退规则

```rust
fn sum_int<I: Numeric + CheckedAdd>(iter: impl Iterator<Item = I>) -> I {
    iter.fold(I::zero(), |acc, x| {
        acc.checked_add(x)
            .expect("integer overflow in reduction (sum)")
    })
}

fn sum_float<F: Numeric + Copy>(iter: impl Iterator<Item = F>) -> F {
    iter.fold(F::zero(), |acc, x| acc + x)
}
```

- 整数路径：`checked_add()` 失败即 panic，不转换为 `XenonError`。
- 浮点路径：保持标量加法顺序；`NaN`、`Inf` 等行为沿用 IEEE 754。
- 复数路径：对实部和虚部分量分别沿用对应实数加法语义，因此含 `NaN` 分量时同样传播。
- SIMD 路径：仅在输入满足 `08-simd.md` 约束（例如元素类型与布局前提受支持）时启用；整数路径必须与标量逐步 checked arithmetic 精确一致，浮点/复数路径允许不同合并顺序，但结果相对标量参考值每个实数分量必须满足 `max(1 ULP, epsilon * |scalar_result|)` 容差，其中 `epsilon` 取对应标量类型（`f32`/`f64`）的 `RealScalar::epsilon()`，否则回退标量。
- 并行路径：仅在满足 `09-parallel.md` 的阈值裁决与禁止嵌套并行约束时启用；整数路径必须保持与串行精确一致，浮点/复数路径允许不同合并顺序，但同样受每个实数分量 `max(1 ULP, epsilon * |scalar_result|)` 的文档化容差约束，其中 `epsilon` 取对应标量类型（`f32`/`f64`）的 `RealScalar::epsilon()`，无法满足时必须回退标量单线程路径。

### 6.3.1 并行阈值配置

归约模块不自定义新的阈值参数，而是直接复用 `parallel` 模块的全局阈值与 guard：

| 项目 | 规则 |
| ---- | ---- |
| 阈值来源 | `sum()` 与 `sum_axis*()` 是否进入并行路径，由 `parallel::should_parallelize(len, is_f_contiguous)` 决定。 |
| 非连续惩罚 | 非连续视图沿用 `parallel` 模块的有效阈值翻倍策略。 |
| 嵌套并行 | 若已在库内部并行区域中，`ParallelGuard::enter()` 失败时必须回退串行，而不是再开第二层并行。 |
| 配置接口 | 阈值读写与重置由 `parallel` 模块统一提供；`reduction` 不额外暴露重复配置。 |

### 6.4 安全性论证

本模块设计不要求新增公开 `unsafe` 接口。若内部实现为性能原因调用张量层已有的低层访问能力，安全前提必须继续建立在以下条件之上：

- 输入 shape / stride / offset 已由 `tensor` 模块的构造约束保证合法。
- 归约仅访问逻辑元素，不访问填充区域或越界内存。
- 输出张量按目标 shape 预先分配，写入索引始终落在结果张量逻辑范围内。
- 对 axis 的运行时校验先于任何基于 axis 的索引投影，因此不会因越界 axis 触发未定义行为。

---

## 7. 实现任务拆分

### Wave 1: 模块骨架与公共边界

- [ ] **T1**: 整理 `src/reduction/mod.rs` 的导出边界
  - 文件: `src/reduction/mod.rs`
  - 内容: 暴露 `sum` 家族公共 API，保持模块入口最小化
  - 测试: 编译通过
  - 前置: `tensor`、`dimension`、`element` 模块完成
  - 预计: 5 min

### Wave 2: 标量归约实现

- [ ] **T2**: 实现全局 `sum()`
  - 文件: `src/reduction/sum.rs`
  - 内容: 完成全量遍历、整数 checked arithmetic、空数组零语义
  - 测试: `test_sum_i32`, `test_sum_empty`, `test_sum_nan`, `test_sum_complex_nan`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 实现 `sum_axis()`
  - 文件: `src/reduction/sum.rs`
  - 内容: 增加 axis 校验、输出 shape 缩减、按轴槽位累加
  - 测试: `test_sum_axis_2d`, `test_sum_axis_invalid_axis`, `test_sum_axis_zero_len_axis`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `sum_axis_keepdims()`
  - 文件: `src/reduction/sum.rs`
  - 内容: 复用按轴累加逻辑，保留被归约轴长度为 `1`
  - 测试: `test_sum_axis_keepdims`, `test_sum_axis_keepdims_invalid_axis`, `test_sum_axis_keepdims_zero_len_axis`
  - 前置: T3
  - 预计: 10 min

### Wave 3: 可选优化路径边界

- [ ] **T5**: 接入 SIMD / 并行分派守卫
  - 文件: `src/reduction/sum.rs`, `src/simd/*`, `src/parallel/*`
  - 内容: 仅在结果与标量路径可证明一致时启用；否则回退
  - 测试: `test_sum_simd_consistency`, `test_sum_parallel_consistency`
  - 前置: T2, T3, T4, simd/parallel 模块
  - 预计: 10 min

### Wave 4: 测试与错误语义收敛

- [ ] **T6**: 收敛可恢复错误和 panic 语义
  - 文件: `src/reduction/sum.rs`, `tests/test_reduction.rs`
  - 内容: 统一 axis 越界为 `InvalidAxis`，确认整数溢出仍 panic
  - 测试: 所有 reduction 测试
  - 前置: T3, T4, T5
  - 预计: 10 min

### 并行执行分组图

```text
Wave 1: [T1]
            │
Wave 2: [T2] -> [T3] -> [T4]
                           │
Wave 3:                  [T5]
                           │
Wave 4:                  [T6]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置 | 说明 |
| -------- | ---- | ---- |
| 单元测试 | `#[cfg(test)] mod tests` 或 `tests/test_reduction.rs` | 验证 `sum` 家族正确性、错误返回与 panic 契约 |
| 集成测试 | `tests/` | 验证 `reduction` 与 `tensor`、`dimension`、`simd`、`parallel`、`error` 的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖空数组、零长度轴、rank-0、单元素、非连续视图 |
| 属性测试（按需） | `tests/property/` 或等效位置 | 验证空输入单位元、不同行布局视图的一致性、keepdims 形状不变量 |
| Feature gate / 配置测试 | 配置矩阵 | 验证默认配置、`simd`、并行启用/关闭时的回退与一致性 |
| 类型边界 / 编译期测试 | 编译期测试框架或 doctest | 验证 `bool` 不参与归约、`sum` 仅对受支持数值类型开放 |

### 8.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
| -------- | -------- | ------ |
| `test_sum_i32` | 整数全局求和正确 | 高 |
| `test_sum_overflow_panic` | 整数溢出触发 panic | 高 |
| `test_sum_nan` | 浮点 `NaN` 传播 | 高 |
| `test_sum_complex_nan` | 复数含 `NaN` 分量时按分量传播 | 高 |
| `test_sum_empty` | 空数组返回加法单位元 | 高 |
| `test_sum_axis_2d` | 二维按轴归约正确 | 高 |
| `test_sum_axis_keepdims` | keepdims 保留 rank 且把目标轴长度置为 `1` | 高 |
| `test_sum_axis_invalid_axis` | `sum_axis()` 越界返回 `InvalidAxis { operation: "sum_axis", ... }` | 高 |
| `test_sum_axis_keepdims_invalid_axis` | `sum_axis_keepdims()` 越界返回 `InvalidAxis { operation: "sum_axis_keepdims", ... }` | 高 |
| `test_sum_axis_zero_len_axis` | 被归约轴长度为 `0` 时输出槽全部为零 | 高 |
| `test_sum_parallel_consistency` | 并行路径与标量结果、错误类别、panic 语义一致 | 高 |
| `test_sum_simd_consistency` | SIMD 路径与标量结果一致，否则正确回退 | 高 |
| `test_sum_large_tensor_parallel_threshold` | 大张量（`10^7` 量级元素）达到阈值后并行路径仍满足文档化语义 | 高 |
| `test_sum_high_rank_ixdyn` | 高 rank 动态维输入上的 `sum_axis*` shape 与 keepdims 语义正确 | 高 |
| `test_sum_inf` | `Inf` / `-Inf` 输入遵循 IEEE 754 语义 | 高 |

### 8.3 边界测试场景

| 场景 | 预期行为 |
| ---- | -------- |
| 空数组 `shape=[0]` | `sum()` 返回加法单位元 |
| rank-0 输入 `shape=[]` | `sum()` 返回该标量元素本身 |
| 被归约轴长度为 `0`，如 `shape=[0, 3]` 沿 `Axis(0)` | 每个输出位置返回零 |
| 单元素数组 | 结果等于该元素本身 |
| rank-0 输入调用 `sum_axis*` | 返回 `InvalidAxis`，因 `axis >= ndim` |
| 非连续视图 | 结果与连续输入一致 |
| 大张量 `len ≈ 10^7` | 可按阈值选择并行路径，结果仍满足文档化数值语义 |
| 高 rank `IxDyn([1,1,1,1,1,1,1,1])` | `sum_axis*` 的输出 shape 与 keepdims 规则正确 |
| `Inf` / `-Inf` 输入 | 浮点结果遵循 IEEE 754；不触发 panic |

### 8.4 属性测试不变量

| 不变量 | 测试方法 |
| ------ | -------- |
| `sum(empty) == A::zero()` | 对所有受支持类型生成空输入验证 |
| `sum_axis_keepdims(axis).shape()[axis] == 1` | 随机合法 shape 与 axis 验证 |
| `sum_axis(axis)` 与 `sum_axis_keepdims(axis)` 在移除长度为 `1` 的目标轴后结果等价 | 随机输入验证 |
| 连续/非连续视图上的 `sum` 结果一致 | 基于切片/转置生成视图后比较 |

### 8.5 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ------ |
| 默认配置 | 仅标量路径也满足全部正确性与错误语义要求 |
| 启用 `simd` | 只在可证明一致时使用 SIMD，否则回退标量 |
| 启用并行 | 受全局阈值配置控制，不得嵌套并行，且结果/错误/panic 语义与标量路径一致 |
| `simd = ["dep:pulp"]` | feature gate 约束保持不变 |

### 8.6 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | -------- |
| `bool` 不参与 `sum` 归约 | 编译期 trait 边界测试 |
| `usize` 不属于归约元素类型 | 编译期 trait 边界测试 |
| `sum` 仅支持 `i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>` | 编译期签名验证 |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
| ---- | -------- | --------- | ---- |
| `reduction → tensor` | `tensor` | `TensorBase<S, D>`、结果张量构造接口 | 输入可为连续或合法非连续视图；归约只观察逻辑元素顺序与 shape/stride 元数据。 |
| `reduction → dimension` | `dimension` | `Axis`、运行时 shape 投影辅助 | 按轴归约必须先验证 `axis < ndim`，再做维度投影。 |
| `reduction → element` | `element` | `Numeric`、`CheckedAdd`、`ComplexScalar` | 依据元素类型分派整数、浮点、复数归约语义。 |
| `reduction → error` | `error` | `XenonError::InvalidAxis` | axis 越界统一返回结构化错误，不再使用 `InvalidArgument`。 |
| `reduction → simd/parallel` | `simd` / `parallel` | 可选加速入口 | 只有在可证明标量等价时才允许接入。 |

### 9.2 数据流描述

```text
User calls sum / sum_axis / sum_axis_keepdims
    │
    ├── reduction validates axis when needed
    ├── reduction selects scalar baseline or an equivalent optional fast path
    ├── reduction accumulates logical elements with type-specific semantics
    ├── tensor constructs the owned output tensor when axis reduction is requested
    └── returns scalar result or Result<Tensor<...>, XenonError>
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `sum_axis()` 与 `sum_axis_keepdims()` 的 axis 越界统一返回 `XenonError::InvalidAxis { operation, axis, ndim, shape }`；其中 `operation` 必须分别为 `"sum_axis"` 和 `"sum_axis_keepdims"`。对运行时 rank-0 输入同样适用该契约。 |
| Panic | `i32` / `i64` 归约中的累加溢出属于不可恢复错误，必须通过 checked arithmetic panic。 |
| Panic 诊断 | panic 文本至少包含 `operation`、元素类型、触发位置（如 `axis`、`output_index` 或 `element_index`）以及适用 `shape`；推荐格式遵循 `26-error.md §4.6`。 |
| 空输入语义 | 空数组 `sum()` 返回加法单位元；沿轴归约时若被归约轴长度为 `0`，结果张量对应槽位也返回加法单位元。 |
| 数值边界 | 整数类型结果须逐元素精确一致。对浮点和复数类型，不同执行路径（标量/SIMD/并行）允许不同合并顺序，但相对标量参考值每个实数分量必须满足 `max(1 ULP, epsilon * |scalar_result|)`，其中 `epsilon` 取对应标量类型（`f32`/`f64`）的 `RealScalar::epsilon()`；`NaN` / `Inf` 仍按 IEEE 754 自动传播。 |
| 路径一致性 | 标量、SIMD、并行路径在启用条件满足时必须返回相同 shape、相同错误类别，以及满足同一数值语义约束的结果；不能证明时必须回退。 |

### 10.1 错误示例

```rust,ignore
Err(XenonError::InvalidAxis {
    operation: "sum_axis",
    axis: axis.index(),
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
})
```

```rust,ignore
Err(XenonError::InvalidAxis {
    operation: "sum_axis_keepdims",
    axis: axis.index(),
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
})
```

```rust,ignore
// Forbidden for axis out-of-bounds in public reduction APIs
Err(XenonError::InvalidArgument {
    operation: "sum_axis_keepdims",
    argument: "axis",
    expected: "axis < ndim",
    actual: axis.index().to_string(),
    axis: Some(axis.index()),
    shape: Some(self.shape().to_vec()),
})
```

---

## 11. 设计决策记录（ADR）

### 决策 1：当前版本只支持 `sum`

| 属性 | 值 |
| ---- | ---- |
| 决策 | 归约模块当前版本只实现 `sum`、`sum_axis`、`sum_axis_keepdims`。 |
| 理由 | 与需求说明书 §14 保持一致，控制范围并优先保证语义闭合。 |
| 替代方案 | 同期加入 `mean`、`prod`、`min/max` 等其它归约。 |
| 拒绝原因 | 超出当前版本范围，会扩大类型约束、错误语义和测试面。 |

### 决策 2：axis 越界统一为 `InvalidAxis`

| 属性 | 值 |
| ---- | ---- |
| 决策 | `sum_axis` 与 `sum_axis_keepdims` 的 axis 越界只返回 `XenonError::InvalidAxis`。 |
| 理由 | `26-error.md` 已为 axis 语义定义专门错误种类，且该错误能统一携带 `axis`、`ndim`、`shape`。 |
| 替代方案 | 让部分入口使用 `InvalidArgument` 表达 axis 参数非法。 |
| 拒绝原因 | 会破坏归约 API 的错误一致性，也弱化 axis 专用诊断字段语义。 |

### 决策 3：整数溢出使用 panic 而非 `Result`

| 属性 | 值 |
| ---- | ---- |
| 决策 | `i32` / `i64` 归约的累加溢出使用 checked arithmetic panic。 |
| 理由 | 需求说明书 §14、§27 将其定义为不可恢复算术域错误。 |
| 替代方案 | 返回 `XenonError`。 |
| 拒绝原因 | 与全局错误规范不一致，并会改变已有 API 的 panic / recoverable 边界。 |

### 决策 4：可选优化必须保持标量等价

| 属性 | 值 |
| ---- | ---- |
| 决策 | SIMD / 并行仅在满足 `require.md` §28.3 数值语义约束时启用，否则回退标量。 |
| 理由 | 归约对累加顺序敏感，必须优先保持统一的对外语义。 |
| 替代方案 | 无条件按数据规模选择 SIMD 或并行。 |
| 拒绝原因 | 可能改变浮点/复数结果或 panic 时机，不满足路径一致性约束。 |

---

## 12. 性能描述

### 12.1 复杂度

| 操作 | 时间复杂度 | 额外空间 |
| ---- | ---------- | -------- |
| `sum()` | O(n) | O(1) |
| `sum_axis()` | O(n) | O(m)，其中 `m` 为输出元素数 |
| `sum_axis_keepdims()` | O(n) | O(m)，其中 `m` 为 keepdims 输出元素数 |

### 12.2 路径说明

| 路径 | 说明 |
| ---- | ---- |
| 标量路径 | 语义基线；始终可用。 |
| SIMD 路径 | 仅在满足等价性前提时启用；否则回退标量。 |
| 并行路径 | 仅在满足等价性和无嵌套并行约束时启用；否则回退标量。 |

### 12.3 缓存与布局说明

- 连续 F-order 输入通常具有更好的缓存局部性。
- 非连续视图仍须返回正确结果，但可能因 stride 跳转而降低吞吐。
- 文档只约束外部语义，不承诺任何会改变结果顺序的重排优化。

---

## 13. 平台与工程约束

| 项目 | 约束 |
| ---- | ---- |
| 标准库环境 | Xenon 当前版本仅支持 `std`。 |
| crate 结构 | 保持单 crate 结构，`reduction` 作为库内模块存在。 |
| 依赖约束 | 不新增第三方依赖；仅可使用需求中已允许的 `rayon` / `pulp` 对应可选能力。 |
| SemVer | `sum` 家族的空输入语义、`InvalidAxis` 错误类别、0D 轴归约运行时诊断与文档化容差规则均属于稳定契约；后续优化不得改变。 |
| 平台语义 | 同平台、同编译配置、同执行路径下结果须确定；跨平台遵循 IEEE 754 语义约束。 |
| API 稳定性 | 不改变当前 `sum` 家族公开接口与错误类别边界。 |

---

## 版本历史

| 版本 | 日期 |
| ---- | ---- |
| 1.0.0 | 2026-04-14 |
| 1.0.1 | 2026-04-15 |
| 1.0.2 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
