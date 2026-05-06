# 归约运算模块设计

> 文档编号: 13
> 模块目录: src/reduction/
> 任务阶段: Phase 4
> 前置文档: 01-architecture.md, 02-dimension.md, 03-element.md, 07-tensor.md, 09-parallel.md, 10-iterator.md, 26-error.md

---

## 1. 模块定位/概述

### 1.1 职责边界

| 职责     | 包含                                                                               |
| -------- | ---------------------------------------------------------------------------------- |
| sum 归约 | 全局 `sum`、沿轴 `sum_axis`、保留轴版本 `sum_axis_keepdims`                        |
| 数值语义 | 整数 checked arithmetic、浮点 `NaN` 传播、空数组返回加法单位元                     |
| 执行路径 | 标量基线路径，以及仅在满足 `需求说明书 §28.3` 数值语义约束时启用的 SIMD / 并行分派 |
| 错误边界 | 轴越界返回 `XenonError::InvalidAxis`；整数溢出 panic                               |

| 职责     | 不包含                                                          |
| -------- | --------------------------------------------------------------- |
| sum 归约 | `mean`、`var`、`prod`、`min`、`max`、`argmin`、`argmax`         |
| 数值语义 | 自动类型提升、额外公开的近似/补偿求和算法（如 Kahan summation） |
| 执行路径 | 为追求吞吐而放宽结果一致性的优化路径                            |
| 错误边界 | 为 axis 错误使用 `InvalidArgument`                              |

### 1.2 设计原则

| 原则       | 体现                                                                        |
| ---------- | --------------------------------------------------------------------------- |
| 最小范围   | 公开 API 只覆盖 `sum`、`sum_axis`、`sum_axis_keepdims`。                    |
| 语义优先   | 空数组返回加法单位元；浮点遵循 IEEE 754；整数溢出按不可恢复算术域错误处理。 |
| 路径一致性 | SIMD 与并行只在 `dispatch` 判定满足 `需求说明书 §28.3` 数值语义约束时参与。 |
| 错误统一   | 所有 axis 越界都统一为 `XenonError::InvalidAxis`。                          |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                   |
| -------- | -------------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §9、§14、§27、§28                                                           |
| 范围内   | 全局 `sum`、沿轴 `sum_axis`、`sum_axis_keepdims`                                       |
| 范围外   | `mean`、`var`、`prod`、`min`、`max`、`argmin`、`argmax`、自定义 reducer、误差补偿求和。|
| 非目标   | 不新增第三方数值依赖，不改变 F-order 布局前提，不把 axis 错误扩展成额外局部错误类型。  |

---

## 3. 文件位置

```text
src/reduction/
├── mod.rs              # module entry and public re-exports
└── sum.rs              # sum / sum_axis / sum_axis_keepdims implementations
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```text
src/reduction/
├── mod.rs
│   └── crate::reduction::sum
└── sum.rs
    ├── crate::tensor        # TensorBase<S, D>, Tensor<A, D>, shape/ndim helpers
    ├── crate::dimension     # Axis, Dimension, runtime axis projection helpers
    ├── crate::element       # Numeric, CheckedAdd, ComplexScalar
    ├── crate::dispatch      # ExecPath, select_exec_path()
    ├── crate::error         # XenonError::InvalidAxis
    ├── crate::simd          # Pure vectorized sum kernel (no scalar fallback)
    └── crate::parallel      # Pure parallel sum execution (no serial fallback)
```

### 4.2 类型级依赖

| 来源模块           | 使用的类型/trait                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `tensor`           | `TensorBase<S, D>`、`Tensor<A, D>`、`.shape()`、`.ndim()`、`.iter()`、结果张量构造接口        |
| `dimension`        | `Axis`、`Dimension`、运行时 axis/shape 投影辅助，以及仅供内部结果维度投影使用的 `RemoveAxis`  |
| `element`          | `Numeric`、`CheckedAdd`、`A::zero()`                                                          |
| `dispatch`（内部） | `select_exec_path()`、`ExecPath`                                                              |
| `error`            | `XenonError::InvalidAxis`                                                                     |
| `simd`（可选）     | 仅在可证明与标量累加顺序和结果语义一致时通过纯向量化 kernel 参与 `sum` 实现                   |
| `parallel`（可选） | 仅在通过 dispatch.rs 路径裁决后提供纯并行执行，不含串行回退，并遵守无嵌套并行约束             |

### 4.3 依赖合法性

| 项目           | 说明                                                                        |
| -------------- | --------------------------------------------------------------------------- |
| 新增第三方依赖 | 无；仅可使用需求中已允许的可选依赖 `pulp`、`rayon` 所对应的项目内能力边界。 |
| 合法性结论     | 合法；符合最小依赖、单 crate、`std` 环境约束。                              |
| 替代方案       | 不适用；当前范围内无需新增额外归约框架或数值库。                            |

### 4.4 依赖方向

依赖方向：单向向上。 `reduction` 仅消费 `tensor`、`dimension`、`element`、`error` 以及项目内可选的 `simd` / `parallel` 能力，不被这些基础模块反向依赖。

---

## 5. 公共 API 设计

### 5.1 核心接口

```rust,ignore
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
    ///
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<A, D::Smaller>, XenonError>
    where
        D: RemoveAxis;

    /// Reduces along `axis` and keeps the reduced axis with length 1.
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.index() >= self.ndim()`.
    /// keepdims does not remove the reduced axis.
    pub fn sum_axis_keepdims(&self, axis: Axis) -> Result<Tensor<A, D>, XenonError>;
}
```

- `sum_axis()` 通过返回 `Tensor<A, D::Smaller>` 描述"移除一条轴后维度降一"的语义，因此公开签名要求 `D: RemoveAxis`。该 trait 是公开 sealed trait（定义见 `02-dimension.md §5.8`），对外可命名但禁止外部实现。对所有实际进入运行时路径的调用，仍必须校验 `axis < ndim` 并返回 `XenonError::InvalidAxis`；其中 0D 张量因 `ndim == 0` 不存在合法轴，统一走此错误路径。
- `sum_axis_keepdims()` 不移除被归约轴，因此不需要 `RemoveAxis` 约束。输出维度类型与输入维度类型相同，被归约轴长度变为 `1`。但对 0D 张量而言不存在任何合法轴，因此 `sum_axis_keepdims()` 仍须返回 `InvalidAxis`，而不能定义为 no-op。

### 5.2 对外错误契约

布尔类型 (`bool`) 不参与 `sum` 归约（`需求说明书 §14`）。该约束由元素层 trait 边界保证。当前版本以 `Numeric` 作为公开 API 的最终边界，不再额外引入更窄的公开 trait 名称。沿轴归约的 axis 越界错误必须统一为：

```rust,ignore
XenonError::InvalidAxis {
    operation: Cow::Borrowed("sum_axis"),
    axis: axis.index(),
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
}
```

```rust,ignore
XenonError::InvalidAxis {
    operation: Cow::Borrowed("sum_axis_keepdims"),
    axis: axis.index(),
    ndim: self.ndim(),
    shape: self.shape().to_vec(),
}
```

对 `sum_axis` 与 `sum_axis_keepdims`，axis 越界只允许使用 `XenonError::InvalidAxis`。

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
// Ok(0) or Err(XenonError::InvalidShape { .. }) — overflow must panic, not return
```

---

## 6. 内部实现设计

### 6.1 核心不变量

| 不变量        | 说明                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------- |
| 归约族范围    | 当前版本只实现 `sum`，不为其它归约预留公开入口。                                                |
| 空输入语义    | `sum()` 对空数组返回 `A::zero()`；沿轴归约的被归约轴长度为 `0` 时，对每个输出槽写入 `A::zero()`。|
| axis 校验顺序 | `sum_axis()` 要求 `D: RemoveAxis`（编译期维度降阶），`sum_axis_keepdims()` 要求 `D: Dimension`（不要求 `RemoveAxis`）。对所有进入 axis 归约路径的调用，都必须先校验 `axis < ndim`；若越界则统一返回 `XenonError::InvalidAxis`。 |
| 整数语义      | `i32` / `i64` 累加使用 checked arithmetic，任何溢出立即 panic。                                 |
| 浮点/复数语义 | `f32` / `f64` / `Complex<_>` 遵循标量加法语义，`NaN` 按 IEEE 754 自动传播。                     |
| 执行路径约束  | SIMD / 并行若无法满足 `需求说明书 §28.3` 数值语义约束，dispatch 必须不选择对应路径。             |
| 布局前提      | 算法面向 Xenon 当前支持的 F-order 语义和合法 stride 视图，不得引入 C-order 假设。               |

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

`sum()` 对 rank-0 张量（标量）返回其唯一元素，与 `A::zero()` 语义无关。

### 6.3 类型分派与回退规则

**调度模型**：由 `dispatch.rs` 通过 `let (path, guard) = dispatch::select_exec_path(...)` 统一裁决三路 `Serial / Simd / Parallel`，返回的 `Option<ParallelGuard>` 仅在选中 `Parallel` 时为 `Some(_)`，并由 `reduction` 按值移交给 `parallel` 后端入口。在 `Parallel` 路径中，单个 worker 拿到 chunk 后可以在 chunk 内部独立做 SIMD admission。串行路径下 SIMD 由 `simd` 后端按其 admission 规则独立判断是否启用；不进入 SIMD 时回退到该路径上的标量循环。

```rust,ignore
fn sum_int<I: Numeric + CheckedAdd>(iter: impl Iterator<Item = I>) -> I {
    iter.fold(I::zero(), |acc, x| {
        acc.checked_add(x)
            .expect("integer overflow in reduction (sum)")
    })
}

fn sum_floating_or_complex<A: Numeric>(iter: impl Iterator<Item = A>) -> A {
    iter.fold(A::zero(), |acc, x| acc + x)
}
```

- 整数路径：`checked_add()` 失败即 panic，不转换为 `XenonError`。
- 浮点路径：保持标量加法顺序；`NaN`、`Inf` 等行为沿用 IEEE 754。
- 复数路径：对实部和虚部分量分别沿用对应实数加法语义，因此含 `NaN` 分量时同样传播。
- 整数 SIMD admission：整数归约默认优先标量/串行路径以保证 checked arithmetic 精确等价。仅当 SIMD 路径能证明与逐步 checked 加法完全一致时才启用优化。
- SIMD 路径：仅在 `dispatch::select_exec_path()` 返回 `ExecPath::Simd` 时委托 `simd/` 纯向量化后端；浮点/复数路径允许不同合并顺序。以 `需求说明书 §28.3` 为权威基线。
- 并行路径：仅在 `dispatch::select_exec_path()` 返回 `(ExecPath::Parallel, Some(guard))` 时委托 `parallel/` 纯并行后端，并按值移交 guard；整数路径必须保持与串行精确一致；若实现无法保证整数 chunk 索引顺序仲裁，**回退责任在调用方（本模块）**——当整数归约的 chunk-order 仲裁无法保证等价时，本模块应直接走 Serial 分支，不调用 `select_exec_path()` 或只在已确认 Parallel 合法时才调用。
- 同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用以 `需求说明书 §28.3` 为权威基线的文档化容差。

**有限值数值容差表：**

该表闭合 `需求说明书 §28.3` 对"文档化误差容差"的要求。它仅适用于浮点 / 复数 `sum` 在不同执行路径之间的有限值结果比较，例如 Serial scalar vs SIMD、Serial scalar vs Parallel、Parallel worker 内 SIMD vs 非 SIMD。它不适用于整数路径；整数 `sum` 必须与逐步 checked arithmetic 精确一致。

| 元素类型       | 比较对象  | 有限值容差                                                                                                                            |
| -------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `f32`          | `sum` 结果 | `abs(actual - expected) <= max(4.0 * f32::EPSILON * (n as f32) * max_abs_input, 4.0 * f32::MIN_POSITIVE)`                            |
| `f64`          | `sum` 结果 | `abs(actual - expected) <= max(4.0 * f64::EPSILON * (n as f64) * max_abs_input, 4.0 * f64::MIN_POSITIVE)`                            |
| `Complex<f32>` | `sum` 结果 | 实部和虚部分别按 `f32` 规则比较；`max_abs_input` 对每个分量独立计算（即实部容差使用所有输入实部的 `max_abs`，虚部使用所有输入虚部的 `max_abs`） |
| `Complex<f64>` | `sum` 结果 | 实部和虚部分别按 `f64` 规则比较；`max_abs_input` 同 `Complex<f32>` 行的规则                                                          |

其中：

- `n` 是本次归约实际累加的元素数（对 `sum_axis` 是被归约轴长度，对全归约 `sum` 是 `tensor.len()`）。
- `max_abs_input` 是参与该输出槽位归约的有限输入值绝对值最大值。
- 若 `n == 0`，空归约返回加法单位元（`A::zero()`），不使用容差比较——空归约结果跨路径必须逐位一致。
- 若所有参与比较的有限输入均为 `0.0`（含 `+0.0` 与 `-0.0`），`max_abs_input == 0.0`，容差退化为表中第二项 `4.0 * MIN_POSITIVE` 的下限。

**非有限值规则：**

- `NaN`：按 IEEE 754 传播语义验证。仅约束 NaN 的存在性，不约束 NaN 的位模式（payload）：
  - **存在性约束（强制）**：若标量基线路径产生 NaN，则 SIMD / 并行 / 并行+SIMD 路径在相同输入下也**必须**产生 NaN（不得返回有限值或 ±Inf）；反之亦然。
  - **位模式不约束**：NaN 的具体 payload 字段（即 `f32::to_bits()` / `f64::to_bits()` 在 NaN 类别内的取值）**不**作为跨路径比较项。IEEE 754 允许实现在 NaN 算术传播中产生不同 payload 的 NaN（例如 `NaN + x` 是否保留输入 NaN 的 payload 由硬件/编译器决定）；不同路径因合并顺序不同可能产生 payload 不同的 NaN，这**不**视为跨路径不一致。
  - **比较方法**：跨路径 NaN 一致性测试使用 `is_nan()` 谓词比较，**不**使用 `to_bits()` 比较；同执行路径同输入的 bit-identical 也仅承诺有限值，对 NaN 不做承诺。
  - **复数**：含 `NaN` 分量的复数结果按实部/虚部分别套用以上规则——只要标量基线对应分量为 NaN，其他路径对应分量也必须为 NaN，分量 payload 不约束。
- `+Inf` / `-Inf`：必须同号同类；有限容差不得把有限值与无穷值视为相等。
- `+0.0` / `-0.0`：符号必须一致；不得用容差抹平零符号差异。
- 容差只约束不同执行路径引入的舍入差异，不允许改变 shape、错误类别、panic 契约或整数溢出语义。

**实现回退条款：** 若某个 SIMD 或并行实现无法证明其结果满足上表（例如使用了 FMA/Kahan/pairwise 但未走完误差分析），调用方（本模块）必须不进入该路径，而不是由 `reduction` 内部在运行后修正结果。这与 §6.3 上一条 bullet 中"回退责任在调用方"的全局规则一致。

**SIMD sum 调用方 type gate：** `SimdKernel::sum` 因返回 `A` 而无 admission 信号通道。本模块在调用 `dispatch::select_exec_path()` 之前须完成元素类型 gate：

- 浮点 / 复数（`f32` / `f64` / `Complex<f32>` / `Complex<f64>`）：可走完整三路 dispatch
- 整数（`i32` / `i64`）且无已验证 widening SIMD 实现：**不**调用 `select_exec_path()`，直接走标量串行实现；亦不进入 parallel 路径除非 chunk-order 仲裁与 checked 等价已证明
- 整数且已验证 widening SIMD 实现（如 ISA 提供 `i32 → i64` widening）：可走完整三路 dispatch；SIMD 实现侧负责保证 checked 等价

### 6.4 并行 axis 归约写回策略

沿轴归约进入并行路径时，写回策略必须按输出槽位分区而不是按输入元素任意抢占：每个并行任务只负责一组互不重叠的输出索引区间，并在其私有局部累加完成后一次性写回对应输出槽位。不得让两个任务同时写入同一个输出元素，也不得通过共享可变引用在任务间累加同一槽位。由此可保证并行 axis-reduction 不发生数据竞争；若当前布局或调度策略无法证明这一点，必须回退串行路径。

### 6.5 并行阈值配置

归约模块不自定义新的阈值参数，而是通过 `dispatch.rs` 统一管理全局阈值与嵌套并行防护：

| 项目       | 规则                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------ |
| 阈值来源   | `sum()` 与 `sum_axis*()` 是否进入并行路径，由 `dispatch::select_exec_path()` 的返回值决定。|
| 非连续惩罚 | 非连续视图沿用 `dispatch.rs` 的有效阈值 saturating 翻倍策略。                              |
| 嵌套并行   | `select_exec_path` 内部检测当前线程是否已处于库内部并行区域；若已嵌套则不会返回 `(Parallel, _)`，从而把并行降级为串行/SIMD 路径，调用方无需再做嵌套防护。|
| 配置接口   | 阈值读写与重置由 `dispatch.rs` 统一提供；`reduction` 不额外暴露重复配置。|

### 6.6 安全性论证

本模块设计不要求新增公开 `unsafe` 接口。若内部实现为性能原因调用张量层已有的低层访问能力，安全前提必须继续建立在以下条件之上：

- 输入 shape / stride / offset 已由 `tensor` 模块的构造约束保证合法。
- 归约仅访问逻辑元素，不访问填充区域或越界内存。
- 输出张量按目标 shape 预先分配，写入索引始终落在结果张量逻辑范围内。
- 对 axis 的运行时校验先于任何基于 axis 的索引投影，因此不会因越界 axis 触发未定义行为。

---

## 7. 实现任务拆分

### Wave 1: 模块骨架与串行归约基线

- [ ] **T1**: 整理 `src/reduction/mod.rs` 的导出边界
  - 文件: `src/reduction/mod.rs`
  - 内容: 暴露 `sum` 家族公共 API，保持模块入口最小化
  - 测试: 编译通过
  - 前置: `tensor`、`dimension`、`element` 模块完成
  - 预计: 5 min

- [ ] **T2**: 实现全局 `sum()`
  - 文件: `src/reduction/sum.rs`
  - 内容: 完成全量遍历、整数 checked arithmetic、空数组零语义
  - 测试: `test_sum_i32`, `test_sum_empty`, `test_sum_nan`, `test_sum_complex_nan`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 按轴归约实现

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
  - 内容: 接入 dispatch 裁决结果；确保 dispatch 不会把不满足语义约束的输入路由到 SIMD / Parallel 路径
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

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类                | 位置                          | 说明                                                                       |
| ----------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| 单元测试                | `#[cfg(test)] mod tests`      | 验证 `sum` 家族正确性、错误返回与 panic 契约                               |
| 集成测试                | `tests/`                      | 验证 `reduction` 与 `tensor`、`dimension`、`simd`、`parallel`等的协同路径  |
| 边界测试                | 同模块测试中标注              | 覆盖空数组、零长度轴、rank-0、单元素、非连续视图                           |
| 属性测试（按需）        | `tests/property/` 或等效位置  | 验证空输入单位元、不同行布局视图的一致性、keepdims 形状不变量              |
| Feature gate / 配置测试 | 配置矩阵                      | 验证默认配置、`simd`、并行启用/关闭时的回退与一致性                        |
| 类型边界 / 编译期测试   | 编译期测试框架或 doctest      | 验证 `bool` 不参与归约、`sum` 仅对受支持数值类型开放                       |

### 8.2 单元测试清单

| 测试函数                                   | 测试内容                                                        | 优先级 |
| ------------------------------------------ | --------------------------------------------------------------- | ------ |
| `test_sum_i32`                             | 整数全局求和正确                                                | 高     |
| `test_sum_overflow_panic`                  | 整数溢出触发 panic                                              | 高     |
| `test_sum_nan`                             | 浮点 `NaN` 传播                                                 | 高     |
| `test_sum_complex_nan`                     | 复数含 `NaN` 分量时按分量传播                                   | 高     |
| `test_sum_empty`                           | 空数组返回加法单位元                                            | 高     |
| `test_sum_axis_2d`                         | 二维按轴归约正确                                                | 高     |
| `test_sum_axis_keepdims`                   | keepdims 保留 rank 且把目标轴长度置为 `1`                       | 高     |
| `test_sum_axis_invalid_axis`               | `sum_axis()` 越界返回 `InvalidAxis`                             | 高     |
| `test_sum_axis_keepdims_invalid_axis`      | `sum_axis_keepdims()` 越界返回 `InvalidAxis`                    | 高     |
| `test_sum_axis_zero_len_axis`              | 被归约轴长度为 `0` 时输出槽全部为零                             | 高     |
| `test_sum_parallel_consistency`            | 并行路径与标量结果、错误类别、panic 语义一致；浮点/复数有限结果满足 §6.3 容差表；非有限值规则按 §6.3：NaN 仅校验存在性（`is_nan()` 谓词，不校验 payload 位模式）、±Inf 同号同类、±0.0 符号一致 | 高     |
| `test_sum_simd_consistency`                | SIMD 路径与标量结果一致；浮点/复数有限结果满足 §6.3 容差表；不满足前提时 dispatch 不选择 SIMD | 高     |
| `test_sum_large_tensor_parallel_threshold` | 大张量（`10^7` 量级元素）达到阈值后并行路径仍满足文档化语义     | 高     |
| `test_sum_high_rank_ixdyn`                 | 高 rank 动态维输入上的 `sum_axis*` shape 与 keepdims 语义正确   | 高     |
| `test_sum_scalar_rank0`                    | rank-0 张量 `sum()` 返回其唯一元素                               | 高     |
| `test_sum_inf`                             | `Inf` / `-Inf` 输入遵循 IEEE 754 语义                           | 高     |

### 8.3 边界测试场景

| 场景                                               | 预期行为                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| 空数组 `shape=[0]`                                 | `sum()` 返回加法单位元                                             |
| rank-0 输入 `shape=[]`                             | `sum()` 返回该标量元素本身                                         |
| 被归约轴长度为 `0`，如 `shape=[0, 3]` 沿 `Axis(0)` | 每个输出位置返回零                                                 |
| 单元素数组                                         | 结果等于该元素本身                                                 |
| 空张量 `shape=[0, 3]`                              | `sum()` 返回加法单位元；`sum_axis*` 输出 shape 与零长度轴语义正确  |
| rank-6 张量 `IxDyn([2,1,3,1,1,4])` 沿 `Axis(5)` 归约 | `sum_axis*` 的 axis 投影、keepdims 与错误诊断保持正确            |
| `10^7` 元素张量归约                                | 默认/SIMD/并行配置下满足 §6.3 有限值容差表与非有限值规则，且 panic 契约一致 |
| 静态 rank-0 输入 `Ix0` 调用 `sum_axis()`           | 编译期可调用（`Ix0: RemoveAxis`，`Smaller = Ix0`）；运行时返回 `InvalidAxis` |
| 静态 rank-0 输入 `Ix0` 调用 `sum_axis_keepdims()`  | 返回 `InvalidAxis`，因为 0D 上不存在合法轴                         |
| 动态 rank-0 输入 `IxDyn([])` 调用 `sum_axis*`      | 返回 `InvalidAxis`，因运行时 `axis >= ndim`                        |
| 非连续视图                                         | 结果与连续输入一致                                                 |
| 大张量 `len ≈ 10^7`                                | 可按阈值选择并行路径，结果仍满足文档化数值语义                     |
| 高 rank `IxDyn([1,1,1,1,1,1,1,1])`                 | `sum_axis*` 的输出 shape 与 keepdims 规则正确                      |
| `Inf` / `-Inf` 输入                                | 浮点结果遵循 IEEE 754；不触发 panic                                |

### 8.4 属性测试不变量

| 不变量                                                                            | 测试方法                       |
| --------------------------------------------------------------------------------- | ------------------------------ |
| `sum(empty) == A::zero()`                                                         | 对所有受支持类型生成空输入验证 |
| `sum_axis_keepdims(axis).shape()[axis] == 1`                                      | 随机合法 shape 与 axis 验证    |
| `sum_axis(axis)` 与 `sum_axis_keepdims(axis)` 在移除长度为 `1` 的目标轴后结果等价 | 随机输入验证                   |
| 连续/非连续视图上的 `sum` 结果一致                                                | 基于切片/转置生成视图后比较    |

### 8.5 集成测试

| 测试文件                  | 测试内容                                                                 |
| ------------------------- | ------------------------------------------------------------------------ |
| `tests/test_reduction.rs` | `sum` / `sum_axis` / `sum_axis_keepdims` 与 tensor 创建、维度变换、元素类型的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置                  | 验证点                                                                 |
| --------------------- | ---------------------------------------------------------------------- |
| 默认配置              | 仅标量路径也满足全部正确性与错误语义要求                               |
| 启用 `simd`           | dispatch 只在可证明一致时选择 SIMD，否则不选择 SIMD 路径               |
| 启用并行              | 受全局阈值配置控制，不得嵌套并行，且结果/错误/panic 语义与标量路径一致 |
| 同时启用 `simd,parallel` | 并行 worker chunk 内可独立做 SIMD admission，整体语义仍满足 §10     |
| `simd = ["dep:pulp"]` | feature gate 约束保持不变                                              |

### 8.7 类型边界 / 编译期测试

| 场景                                                                    | 测试方式              |
| ----------------------------------------------------------------------- | --------------------- |
| `bool` 不参与 `sum` 归约                                                | 编译期 trait 边界测试 |
| `usize` 不属于归约元素类型                                              | 编译期 trait 边界测试 |
| `sum` 仅支持 `i32`、`i64`、`f32`、`f64`、`Complex<f32>`、`Complex<f64>` | 编译期签名验证        |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向                        | 对方模块            | 接口/类型                                | 约定                                                                                                                             |
| --------------------------- | ------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `reduction → tensor`        | `tensor`            | `TensorBase<S, D>`、结果张量构造接口     | 输入可为连续或合法非连续视图；归约只观察逻辑元素顺序与 shape/stride 元数据。归约操作接受只读视图作为输入，按逻辑元素读取并归约。 |
| `reduction → dimension`     | `dimension`         | `Axis`、运行时 shape 投影辅助            | 按轴归约必须先验证 `axis < ndim`，再做维度投影。                                                                                 |
| `reduction → element`       | `element`           | `Numeric`、`CheckedAdd`、`ComplexScalar` | 依据元素类型分派整数、浮点、复数归约语义。                                                                                       |
| `reduction → error`         | `error`             | `XenonError::InvalidAxis`                | axis 越界统一返回结构化错误，不再使用 `InvalidArgument`。                                                                        |
| `reduction → simd/parallel` | `simd` / `parallel` | 可选加速入口                             | 只有在可证明标量等价时才允许接入。                                                                                               |

### 9.2 数据流描述

```text
User calls sum / sum_axis / sum_axis_keepdims
    │
    ├── reduction validates axis when needed (sum_axis* only)
    ├── let (path, guard) = dispatch::select_exec_path(...)
    │       ├── (Serial, None)        → scalar accumulator; SIMD admission may apply per backend
    │       ├── (Simd,   None)        → simd backend reduce kernel
    │       └── (Parallel, Some(g))   → parallel backend; pass guard by value
    │              └── inside each worker chunk: SIMD admission may apply per chunk
    ├── reduction accumulates logical elements with type-specific semantics
    │      (integer: CheckedAdd; float/complex: ordinary +)
    ├── tensor constructs the owned output tensor when axis reduction is requested
    └── returns scalar result or Result<Tensor<...>, XenonError>
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | 对所有需要运行时 axis 校验的 `sum_axis()` / `sum_axis_keepdims()` 调用，axis 越界统一返回 `XenonError::InvalidAxis`。|
| Panic             | `i32` / `i64` 归约中的累加溢出属于不可恢复错误，必须通过 checked arithmetic panic。                                  |
| Panic 诊断        | panic 文本至少包含 `operation`、元素类型、触发位置（如 `axis`、`output_index` 或 `element_index`）以及适用 `shape`。 |
| 空输入语义        | 空数组 `sum()` 返回加法单位元；沿轴归约时若被归约轴长度为 `0`，结果张量对应槽位也返回加法单位元。                    |
| 数值边界          | 整数类型结果须逐元素精确一致。对浮点和复数类型，不同执行路径允许不同合并顺序——**但允许范围严格受 §6.3 跨路径容差表约束**：仅有限值的相对/绝对误差在表内放宽；非有限值不得用容差抹平。 |
| 路径一致性        | 标量、SIMD、并行路径（含 worker 内 SIMD admission）在启用条件满足时必须返回相同 shape、相同错误类别，以及满足同一数值语义约束的结果。worker 内 SIMD 是否启用由 chunk 内独立 admission 决定，不影响整体语义。|

---

## 11. 设计决策记录

### 决策 1：当前版本只支持 `sum`

| 属性     | 值                                                              |
| -------- | --------------------------------------------------------------- |
| 决策     | 归约模块当前版本只实现 `sum`、`sum_axis`、`sum_axis_keepdims`。 |
| 理由     | 与 `需求说明书 §14` 保持一致，控制范围并优先保证语义闭合。      |
| 替代方案 | 同期加入 `mean`、`prod`、`min/max` 等其它归约。                 |
| 拒绝原因 | 超出当前版本范围，会扩大类型约束、错误语义和测试面。            |

### 决策 2：axis 越界统一为 `InvalidAxis`

| 属性     | 值                                                                                         |
| -------- | ------------------------------------------------------------------------------------------ |
| 决策     | `sum_axis` 与 `sum_axis_keepdims` 的 axis 越界只返回 `XenonError::InvalidAxis`。           |
| 理由     | `26-error.md` 已为 axis 语义定义专门错误种类，且该错误能统一携带 `axis`、`ndim`、`shape`。 |
| 替代方案 | 让部分入口使用 `InvalidArgument` 表达 axis 参数非法。                                      |
| 拒绝原因 | 会破坏归约 API 的错误一致性，也弱化 axis 专用诊断字段语义。                                |

### 决策 3：0D axis API 边界

| 属性     | 值                                                                                                                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `sum_axis()` 通过返回 `Tensor<A, D::Smaller>` 在类型层表达"移除一条轴"，公开签名要求 `D: RemoveAxis`。`RemoveAxis` 是公开 sealed trait，对外可命名但禁止外部实现，并为所有维度类型（`Ix0`-`Ix6` 与 `IxDyn`）实现（参见 02-dimension §5.8）。0D 张量（`Ix0` 或 `IxDyn([])`）因 `ndim == 0` 不存在合法轴，`sum_axis()` / `sum_axis_keepdims()` 在 0D 上**编译期可调用**但运行时统一返回 `XenonError::InvalidAxis`。 |
| 理由     | `需求说明书 §14` 与 `02-dimension.md §5.8` 定义 0D 轴 API 须保持 recoverable error 语义。`D::Smaller` 作为关联类型可精确描述"秩降一"的静态语义而无须重塑返回类型。                              |
| 替代方案 | (1) 将 `sum_axis()` 返回类型改为 `Tensor<A, D>` 并舍弃静态秩降信息；(2) 将 `RemoveAxis` 设计为 `Dimension` 的关联类型。                                                                       |
| 拒绝原因 | 前者丧失类型层秩降保证，要求调用方在运行时自行追踪维度变化；后者会扩大 `Dimension` trait 面且违背最小化设计原则。                                                                              |

### 决策 4：整数溢出使用 panic 而非 `Result`

| 属性     | 值                                                                   |
| -------- | -------------------------------------------------------------------- |
| 决策     | `i32` / `i64` 归约的累加溢出使用 checked arithmetic panic。          |
| 理由     | `需求说明书 §14`、`需求说明书 §27` 将其定义为不可恢复算术域错误。    |
| 替代方案 | 返回 `XenonError`。                                                  |
| 拒绝原因 | 与全局错误规范不一致，并会改变已有 API 的 panic / recoverable 边界。 |

### 决策 5：可选优化必须保持标量等价；回退责任在调用方（本模块）

| 属性     | 值                                                                                                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | SIMD / 并行仅在满足 `需求说明书 §28.3` 数值语义约束时启用；若实现版本无法满足该约束，**回退责任在调用方（本模块）**——reduction 在调用 `select_exec_path()` 之前就应自行裁决不进入该路径（例如直接走 Serial），而**不是**由 dispatch 或 reduction 内部根据操作语义回退到串行 |
| 理由     | 与 09-parallel v2.0.0 决策 4（"parallel 不包含串行回退"）和 30-dispatch v2.0.3 决策 7（"select-and-enter 原子绑定"）一致；归约模块本身一旦被 dispatch 选中就忠实执行该路径 |
| 替代方案 | 无条件按数据规模选择 SIMD 或并行                                                                                                                                              |
| 拒绝原因 | 可能改变浮点/复数结果或 panic 时机，不满足路径一致性约束                                                                                                                      |
| 替代方案 | 在 `reduction` 内部回退到串行                                                                                                                                                 |
| 拒绝原因 | 双层回退（reduction 与 dispatch 都做）会让职责漂移，违反单一裁决点原则                                                                                                        |

### 决策 6：worker 内允许 SIMD（与 09-parallel v2.0.0 决策 9 / 08-simd v2.0.0 决策 5 协同）

| 属性     | 值                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 决策     | 进入并行路径后，单个 worker chunk 内可独立做 SIMD admission；chunk 间合并仍由 `parallel` 控制                            |
| 理由     | 撤销 v1.x "并行 worker 不使用 SIMD" 的旧规则，提供 thread × SIMD 双层加速                                                |
| 替代方案 | 保留 v1.x 设计                                                                                                           |
| 拒绝原因 | 与 08-simd / 09-parallel v2.0 决策不协同；归约的 worker 内 chunk 在内积/求和这类紧致循环上有显著吞吐收益                  |

---

## 12. 性能描述

### 12.1 复杂度

| 操作                  | 时间复杂度 | 额外空间                              |
| --------------------- | ---------- | ------------------------------------- |
| `sum()`               | O(n)       | O(1)                                  |
| `sum_axis()`          | O(n)       | O(m)，其中 `m` 为输出元素数           |
| `sum_axis_keepdims()` | O(n)       | O(m)，其中 `m` 为 keepdims 输出元素数 |

### 12.2 路径说明

| 路径      | 说明                                                 |
| --------- | ---------------------------------------------------- |
| 标量路径  | 语义基线；始终可用。                                 |
| SIMD 路径 | 仅当 dispatch 判断满足等价性前提时才会选择该路径；否则 dispatch 不选择 SIMD。 |
| 并行路径  | 仅当 dispatch 判断满足等价性和无嵌套并行约束时才会选择该路径；否则 dispatch 不选择 Parallel。 |

### 12.3 缓存与布局说明

- 连续 F-order 输入通常具有更好的缓存局部性。
- 非连续视图仍须返回正确结果，但可能因 stride 跳转而降低吞吐。
- 文档只约束外部语义，不承诺任何会改变结果顺序的重排优化。

---

## 13. 平台与工程约束

| 项目       | 约束                                                                                                                                                                                                                                                                                   |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`。                                                                                                                                                                                                                                                           |
| crate 结构 | 保持单 crate 结构，`reduction` 作为库内模块存在。                                                                                                                                                                                                                                      |
| 依赖约束   | 不新增第三方依赖；仅可使用需求中已允许的 `rayon` / `pulp` 对应可选能力。                                                                                                                                                                                                               |
| SemVer     | `sum` 家族的空输入语义、`InvalidAxis` 错误类别、`sum_axis()` 的 `D: RemoveAxis` 约束、`sum_axis_keepdims()` 的公开入口要求 `D: Dimension`、0D 张量上的 keepdims axis API 统一返回 `InvalidAxis` 以及以 `需求说明书 §28.3` 为权威基线的容差文档化结论均属于稳定契约；后续优化不得改变。 |
| 平台语义   | 同平台、同编译配置、同执行路径下结果须确定；跨平台遵循 IEEE 754 语义约束。                                                                                                                                                                                                             |
| API 稳定性 | 不改变当前 `sum` 家族公开接口与错误类别边界。                                                                                                                                                                                                                                          |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
