# 并行后端模块设计

> 文档编号: 09 | 模块: `src/parallel/` | 阶段: Phase 5
> 前置文档: `07-tensor.md`, `26-error.md`
> 需求参考: `需求说明书 §1.2`, `需求说明书 §9.2`, `需求说明书 §9.3`, `需求说明书 §27`, `需求说明书 §28.3`, `需求说明书 §28.4`, `需求说明书 §28.5`
> 范围声明: 范围内

---

## 1. 模块定位/概述

并行后端模块是 Xenon 的可选执行后端，通过 `rayon` 为逐元素映射、二元逐元素运算与归约提供纯数据并行能力。该模块默认关闭，仅在启用 `parallel` feature 时参与构建。当前版本覆盖内部 `par_map`、供 `math` 模块消费的 `par_zip_map`、并行 `sum` / `dot`；本模块不负责串行回退、阈值控制或嵌套并行检测，这些执行路径裁决职责已迁移至 `dispatch.rs`。

### 1.1 职责边界表

| 职责           | 包含                                                          | 不包含                           |
| -------------- | ------------------------------------------------------------- | -------------------------------- |
| 并行逐元素执行 | `par_map`、`par_zip_map`、基于视图的并行遍历 | 通用多输入同步并行公开迭代器 API |
| 并行归约       | `par_sum`、`par_dot`、内部 `par_reduce_impl`                  | 矩阵乘法、矩阵分解、GPU 后端     |
| 线程池封装     | 内部 `ParallelPool` 改变执行上下文而不改变 API 语义           | 新的公开调度语义或额外第三方依赖 |

### 1.2 设计原则

| 原则           | 体现                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 语义一致性     | 并行路径不得改变公开 API 的形状、错误类别和数值语义；路径选择由 `dispatch.rs` 统一保证 |
| 最小能力边界   | 当前版本只覆盖 `par_map`、`par_zip_map`、`par_sum`、`par_dot`，不扩展到 GPU 或通用多输入同步公开接口 |
| 可选依赖最小化 | 仅在 `parallel` feature 下引入 `rayon`，默认关闭                              |

### 1.3 架构位置

```text
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout
L3: storage
L4: tensor
L5: iter, dispatch
L6: parallel  <- current module (optional, feature = "parallel")
```

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §1.2`、`需求说明书 §9.2`、`需求说明书 §9.3`、`需求说明书 §27`、`需求说明书 §28.3`、`需求说明书 §28.4`、`需求说明书 §28.5` |
| 范围内   | 可选数据并行、逐元素运算（含二元广播）/归约/内积的并行执行路径 |
| 范围外   | GPU 后端、自动任务图调度、通用多数组 lock-step 并行公开接口、额外第三方依赖                             |
| 非目标   | 不把文档改成 `no_std`，不增加除 `rayon` 之外的外部依赖，不扩展当前并行能力集合                 |

---

## 3. 文件位置

```text
src/parallel/
├── mod.rs         # Module entry, re-exports, ParallelPool
├── par_iter.rs    # ParElements and TensorBase::par_iter()
├── map.rs         # par_map, par_zip_map
├── reduce.rs      # par_reduce_impl, par_sum, par_dot
└── checked.rs     # par_map_checked and error/panic propagation
```

多文件职责划分：
- `mod.rs`：模块声明、re-exports、`ParallelPool` 线程池包装、feature gate 入口。
- `par_iter.rs`：`ParElements` 结构体与 `TensorBase::par_iter()` 单输入元素级并行遍历入口。
- `map.rs`：`par_map`、`par_zip_map` 逐元素映射与二元广播并行入口。
- `reduce.rs`：`par_reduce_impl`、`par_sum`、`par_dot` 并行归约与内积。
- `checked.rs`：`par_map_checked` 及统一的错误/panic 传播逻辑。

---

## 4. 依赖关系

### 4.1 依赖图

```text
src/parallel/
├── rayon (optional)         # ThreadPool, ParallelIterator, current_num_threads
├── crate::tensor            # Tensor, TensorBase, TensorView
├── crate::dimension         # Dimension
├── (module-owned)           # ParElements and par_iter() entry belong to parallel/
└── crate::error             # XenonError
```

### 4.2 类型级依赖表

| 来源模块    | 使用的类型/trait                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| `rayon`     | `rayon::ThreadPool`, `rayon::current_num_threads`, `rayon::iter::ParallelIterator`                              |
| `tensor`    | `Tensor<A, D>`, `TensorBase<S, D>`, `TensorView<'a, A, D>`, `.len()`, `.raw_dim()`, `.is_f_contiguous()` |
| `dimension` | `Dimension` |
| `parallel`  | `ParElements<'a, A, D>`, `TensorBase::par_iter()`, `par_zip_map()`                                             |
| `error`     | `XenonError`, `XenonError::BroadcastError`, `XenonError::DimensionMismatch`, `XenonError::InvalidShape`        |

### 4.3 依赖方向

> **依赖方向：单向向上。** `parallel/` 只提供纯并行执行入口，不包含串行回退。执行路径裁决由 `dispatch.rs` 完成，`parallel/` 不依赖 `dispatch.rs`。`ParElements` 与 `TensorBase::par_iter()` 归属 `parallel` 模块本身，不属于 `iter` 模块。并行路径只建立在上层已完成的张量形状、布局与类型约束之上；广播形状裁决由 `math` 调用侧先完成，再以 `output_dim` 形式传入。

### 4.4 合法性声明

| 项目           | 说明                                                                            |
| -------------- | ------------------------------------------------------------------------------- |
| 新增第三方依赖 | `rayon`（可选）                                                                 |
| 合法性结论     | 合法；符合 `需求说明书 §1.2` 对最小依赖的限制，以及 `需求说明书 §9.2` 对可选并行能力的要求 |
| 替代方案       | 仅用 `std::thread` 不能无损提供当前所需的并行迭代与线程池抽象，因此不采用       |

### 4.5 与迭代器模块的边界

`parallel/` 不定义通用多输入同步并行公开迭代接口。`TensorBase::par_iter()` 只提供单输入元素级并行入口；二元逐元素并行能力以 `pub(crate)` 级 `par_zip_map()` 形式提供，仅供 `math` 模块在完成广播裁决后消费。该边界与 `10-iterator.md §1.2` 中“并行迭代不属于 `iter` 模块公开职责”保持一致。

---

## 5. 公共 API 设计

### 5.1 Feature gate 与运行时状态

```toml
[features]
parallel = ["dep:rayon"]

[dependencies]
rayon = { version = "1.10", optional = true }
```

```rust,ignore
#[cfg(feature = "parallel")]
pub(crate) struct ParallelPool {
    inner: rayon::ThreadPool,
}
```

- `ParallelPool`：内部线程池包装，只改变执行上下文，不改变外部语义；其内部调用仍受 `dispatch.rs` 中的 `ParallelGuard` + `ParallelContext` 保护，自定义 pool 内嵌套调用并行入口时会自动回退串行，不允许嵌套 `ParallelPool` 实例。它属于内部机制，不构成公开 API 契约。

> `ParallelGuard`、阈值状态与嵌套并行防护逻辑已迁移至内部 `dispatch.rs` 模块；本节仅保留与线程池执行上下文直接相关的并行后端状态。

### 5.2 内部执行入口与可见性

> **可见性说明：** `parallel/` 是 `pub(crate)` 内部后端；所有执行后端函数与类型（包括 `par_map`、`par_zip_map`、`par_sum`、`par_dot`、`ParallelPool`、`ParElements`）均保持 `pub(crate)`，仅供 `math` / `reduction` / `matrix` 等语义模块通过 `dispatch.rs` 自动调用。

阈值配置与嵌套并行防护已迁移至内部 `dispatch.rs` 模块，本模块仅提供纯并行执行入口。

> **执行策略说明：** 并行阈值配置由 `dispatch.rs` 统一管理，`parallel/` 模块不提供独立的阈值配置接口。所有并行入口接受 dispatch 层传入的执行策略参数。

### 5.2a 并行阈值配置规范

| 参数 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ---- |
| `parallel_threshold` | `usize` | 65536 | 元素数低于此值时使用串行路径；该值由 `dispatch.rs` 中的编译期常量定义，不属于 `ParallelExecStrategy` 运行时字段 |
| `max_workers` | `Option<usize>` | `None`（使用线程池默认） | 最大并行工作线程数 |
| `chunk_size` | `Option<usize>` | `None`（自动计算） | 每个 chunk 的元素数 |

配置入口由 `dispatch.rs` 提供。`parallel_threshold` 由 `dispatch.rs` 内部常量控制；`parallel/` 模块仅通过 `ParallelExecStrategy` 接收运行时策略字段（`chunk_size`、`max_workers`）。

### 5.2b `ParallelExecStrategy` 参数校验规则

| 字段 | 合法范围 | 默认值 | 非法时行为 |
| ---- | -------- | ------ | ---------- |
| `max_workers` | `Some(1..=pool_size)` 或 `None` | `None` | `0` 或超过线程池大小返回 `InvalidArgument` |
| `chunk_size` | `Some(n)` where `n > 0` 或 `None` | `None` | `0` 返回 `InvalidArgument` |

### 5.3 函数签名

```rust,ignore
pub(crate) struct ParallelExecStrategy {
    pub chunk_size: Option<usize>,
    pub max_workers: Option<usize>,
}

#[cfg(feature = "parallel")]
pub(crate) fn par_map<S, A, B, D, F>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    f: F,
) -> Tensor<B, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_zip_map<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    strategy: &ParallelExecStrategy,
    f: F,
) -> Result<Tensor<C, DO>, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = B>,
    DL: Dimension,
    DR: Dimension,
    DO: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> Result<C, XenonError> + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_reduce_impl<S, A, D, F, ID>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    identity: ID,
    op: F,
) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Send + Sync + Clone,
    F: Fn(A, A) -> A + Sync,
    ID: Fn() -> A + Sync + Clone;

#[cfg(feature = "parallel")]
pub(crate) fn par_sum<S, A, D>(tensor: &TensorBase<S, D>, strategy: &ParallelExecStrategy) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Numeric + Send + Sync + Clone;

#[cfg(feature = "parallel")]
pub(crate) fn par_dot<SL, SR, A, DL, DR>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    strategy: &ParallelExecStrategy,
) -> Result<A, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = A>,
    DL: Dimension,
    DR: Dimension,
    A: Element + Numeric + Send + Sync + Clone;
```

`par_dot()` 在类型层面接受任意 `Dimension` 输入，以便与更通用的上层张量调用路径对接；但其语义契约仍限定为一维向量内积，因此实现必须在运行时检查 `lhs.ndim() == 1`、`rhs.ndim() == 1`，并在进入并行归约前再次确认两侧逻辑长度一致。

整数 `sum` / `dot` 支持并行路径。在并行路径中，每个分片独立执行 checked 算术；若任一分片检测到溢出，并行执行立即终止并传播 panic。并行实现不保证与串行路径拥有相同的“首个溢出位置”，但必须保持“不静默忽略溢出”的语义边界。

复数内积采用共轭线性定义：`result = sum(conj(lhs_i) * rhs_i)`，与 `08-simd.md` 中的复数内积语义完全一致。

### 5.4 并行迭代入口

```rust,ignore
#[cfg(feature = "parallel")]
pub(crate) struct ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
}

#[cfg(feature = "parallel")]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Send + Sync,
{
    pub(crate) fn par_iter(&self) -> ParElements<'_, A, D> {
        ParElements::new(self.view())
    }
}
```

当前版本不提供任何通用并行双输入公开 API；需要二元逐元素调度时，由 `math` 模块先完成广播与输出形状裁决，再通过 `dispatch.rs` 选择并调用 `pub(crate)` 级 `par_zip_map()` 执行并行路径。

### 5.5 Good / Bad 对比示例

```rust,ignore
// Good - shape mismatch stays in Result.
let dot = par_dot(&lhs, &rhs, strategy)?;

// Bad - converting recoverable shape mismatch into unwrap panic.
let dot = par_dot(&lhs, &rhs, strategy).unwrap();
```

### 5.6 文档与示例交付要求

| API                               | 文档要求                                                 | 示例要求                             |
| --------------------------------- | -------------------------------------------------------- | ------------------------------------ |
| `par_map` / `par_sum` / `par_dot` | 明确标注为内部后端入口，只承诺与串行路径语义一致         | 由上层语义模块调用的内部示例         |
| `par_zip_map`                     | 说明其为 `math` 模块经 `dispatch.rs` 选择后消费的内部广播并行入口 | 给出 `add/sub/mul/div` 的内部调度示例 |
| `ParallelPool`                    | 明确标注为内部执行上下文包装，不构成公开 API 契约       | 给出在线程池内执行内部并行入口的示例 |

> `set_parallel_threshold()`、`should_parallelize()` 等接口已迁移至 `dispatch.rs`，不再由 `parallel/` 文档化。

---

## 6. 内部实现设计

### 6.1 路径选择算法

路径选择算法已迁移至 `dispatch.rs` 模块。`parallel/` 仅在被 dispatch 选为执行路径时被调用。

### 6.2 核心执行路径

```text
dispatch-selected parallel entry
    │
    ├── receive validated tensor metadata and closure
    ├── split logical work into fixed chunks
    ├── execute rayon parallel path
    └── propagate panic / Err without swallowing
```

- `parallel/` 假定调用方已经完成阈值、线程环境、SIMD 能力与嵌套并行治理判断。
- 并行函数只负责固定 chunking、执行 `rayon` 并行迭代以及保持结果语义与调用方选择的串行基线一致。
- 调度模型：`dispatch.rs` 只负责决定串行 vs 并行路径；一旦进入 `parallel/`，SIMD vs 标量选择即成为并行后端内部实现细节。

### 6.3 二元逐元素并行路径

```rust,ignore
#[cfg(feature = "parallel")]
pub(crate) fn par_zip_map<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    strategy: &ParallelExecStrategy,
    f: F,
) -> Result<Tensor<C, DO>, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = B>,
    DL: Dimension,
    DR: Dimension,
    DO: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> Result<C, XenonError> + Send + Sync,
{
    let total = output_dim.checked_size().map_err(|_| XenonError::InvalidShape {
        operation: "par_zip_map".into(),
        shape: output_dim.slice().to_vec(),
        expected_elements: 0,
        actual_elements: 0,
        offending_dim: None,
        reason: Some(alloc::borrow::Cow::Borrowed("element count overflow")),
    })?;

    let num_threads = strategy.max_workers.unwrap_or_else(rayon::current_num_threads);
    let chunk_size = strategy
        .chunk_size
        .unwrap_or_else(|| usize::max(1, (total + num_threads - 1) / num_threads));

    // Build broadcast-compatible read-only chunk views for lhs / rhs.
    // Execute f in parallel and collect Result<Vec<C>, XenonError> directly.
    // Panic propagation follows Rayon defaults.
    unimplemented!()
}
```

- `par_zip_map()` 是二元逐元素并行路径的统一设计入口，供 `math` 模块中的 `add` / `sub` / `mul` / `div` 广播运算消费，不直接暴露为公开用户 API。
- `par_zip_map()` 接收的 `lhs`、`rhs` 与 `output_dim` 必须已由调用侧完成兼容性验证；广播裁决（含输出 rank/shape 计算）属于 `math` 模块职责，`parallel/` 不重复做形状推导。
- 广播处理顺序固定为：先由 `math` 模块验证 `lhs` / `rhs` 广播兼容并产出 `output_dim`，再由 `parallel/` 按外轴/块状多维 tile 分块；默认 `chunk_size = max(1, (total_elements + num_threads - 1) / num_threads)` 仍作为 tile 目标工作量上界，其中 `num_threads = rayon::current_num_threads()`，并按固定左折叠顺序合并 chunk 结果。每个 chunk 为两个输入分别构造与该 tile 对应、且仍与 `output_dim` 兼容的只读 sub-view。若某一侧是广播轴（stride 为 `0` 或逻辑重复维），chunk 视图保持该广播语义，不做物理复制。`DL`、`DR`、`DO` 独立建模，以表达输入与输出 rank 可能不同的广播结果。
- 广播 chunk 映射草图：优先按 `output_dim` 的外轴边界生成块状多维 tile，使 chunk 在输出空间内保持可直接切片的矩形子域；若某些退化形状无法形成理想矩形 tile，则实现可退化为“线性索引区间 + 逐元素广播投影”的内部执行形式，而不是要求把任意线性区间整体重建成单个 broadcast sub-view。对输出维中的广播轴，输入侧固定复用同一逻辑坐标；对非广播轴，chunk 保持对应 tile 的区间跨度。实现不得为广播轴做物理展开或额外分配。
- `par_zip_map` 仅包含并行执行逻辑；若调用发生，表示 `dispatch.rs` 已确认当前输入适合走并行路径。
- `par_zip_map()` 作为内部并行入口，假定广播兼容性已由调用方验证，不再额外定义单独的 checked 变体。此为内部前置条件。违反时视为内部 bug，可触发 debug assert，但不得破坏内存安全或对外错误模型。release 模式下行为保持语义定义，不引入未指定行为。并行操作中发生 panic 或返回 `Err` 时，错误不会被静默忽略。语义上，并行操作须至少传播一个错误，不保证传播“第一个”发生的错误。实现上，Rayon 的并行 collect/reduce 可能不会物理中断其他 worker，但错误信息会被收集并在最终结果中报告。

### 6.3a 轴向归约并行方案

- 轴向 `sum_axis(axis)` / `sum_axis_keepdims(axis)` 的并行路径沿未被归约的轴切分为彼此独立的 chunk。
- 每个 chunk 在目标轴上执行串行归约，随后按输出逻辑位置写入局部结果；最终结果按 chunk 索引顺序合并。
- `keepdims` 行为在并行路径下保持不变，仅影响输出 shape，不改变分块策略。
- 空轴归约返回加法单位元。

### 6.4 自动路径派发与所有权

`dispatch.rs` 负责自动路径派发与执行策略裁决。`parallel/` 只接收已经完成路径选择、输入校验与语义前置条件检查的调用。

- `par_map`、`par_zip_map`、`par_sum` 接收的输入都已由上层语义模块和 `dispatch.rs` 验证完毕；`par_dot` 在进入并行归约前仍需自行做运行时校验，要求 `lhs.ndim() == 1`、`rhs.ndim() == 1` 且两侧逻辑长度一致。
- 对归约和内积，若调用方选择并行路径，则 `parallel/` 必须提供固定 chunking 与固定 merge tree，保证同平台、同配置、同路径下结果确定。
- 并行归约采用固定分块策略：`chunk` 大小 = `max(1, (n + num_workers - 1) / num_workers)`，worker 按固定索引范围分配，merge 按 worker 索引顺序合并。
- 若执行对象为整数 `sum` / `dot`，每个 worker 必须在本分片内执行 `checked_add` / `checked_mul` + `checked_add`；任一 worker 发现溢出时必须传播 panic，不得转写为 `XenonError`。
- 调度模型：`dispatch.rs` 只负责决定串行 vs 并行路径；一旦进入 `parallel/`，SIMD vs 标量选择即成为并行后端内部实现细节。

### 6.5 Checked 映射与错误传播

```rust,ignore
pub(crate) fn par_map_checked<A, B, S, D, F>(
    tensor: &TensorBase<S, D>,
    f: F,
) -> Result<Tensor<B, D>, XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> Result<B, XenonError> + Sync + Send,
{
    let output: Result<Vec<B>, XenonError> = tensor.par_iter().map(|x| f(x)).collect();
    Ok(unsafe { Tensor::from_raw_vec_unchecked(output?, tensor.raw_dim()) })
}
```

`par_map_checked()` 不再自行决定是否并行；若被调用，表示 `dispatch.rs` 已选择并行执行路径。

### 6.6 安全性论证

| 主题                             | 论证                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `Tensor::from_raw_vec_unchecked` | 这里只在输出向量长度与 `tensor.raw_dim()` 已由输入张量长度和映射过程保持一致时使用；并行与串行路径都必须保证产出元素数等于输入逻辑元素数。 |
| `par_zip_map` broadcast chunking | 每个并行 chunk 仅借用两个输入的只读 broadcast-compatible sub-view；广播轴保持逻辑重复语义，不进行额外物理展开，因此不会引入越界写或悬垂引用。 |
| panic / `Err` 传播               | 并行操作中发生 panic 或返回 `Err(XenonError)` 时，错误不会被静默忽略；语义上最终结果须至少传播一个错误，不保证传播“第一个”发生的错误。实现上 Rayon 的并行 collect/reduce 可能不会物理中断其他 worker，但错误信息会被收集并在最终结果中报告。 |
| Send/Sync/借用边界               | 并行执行只借用输入张量的只读视图；闭包与元素类型必须满足 `Send` / `Sync` 约束；输出分配与写入归当前 worker 独占，不能向其他 worker 暴露共享可写借用。 |

### 6.7 性能考量

| 方面         | 设计决策                                                                        |
| ------------ | ------------------------------------------------------------------------------- |
| 广播分块     | `par_zip_map()` 按逻辑线性区间分块并复用 broadcast-compatible sub-view，避免复制 |
| 原子访问     | 内部状态只保留并行执行所需的固定成本访问，不额外引入锁                          |
| 路径职责边界 | `parallel/` 只负责分块与执行；`dispatch.rs` 只决定串行 vs 并行，进入 `parallel/` 后的 SIMD/scalar 选择属于并行后端内部实现细节 |

---

## 7. 实现任务拆分

### Wave 1: 基础状态与路径裁决

阈值状态、路径选择与嵌套并行防护已迁移至 `dispatch.rs` 模块，参见 `01-architecture.md`。

### Wave 2: 并行入口与执行内核

- [ ] **T4**: 实现 `ParElements` 与 `TensorBase::par_iter()`
  - 文件: `src/parallel/par_iter.rs`
  - 内容: 单输入元素级并行遍历入口
  - 测试: `test_par_iter_len_matches_tensor_len`
  - 前置: `dispatch.rs` 执行路径裁决已可用，`10-iterator.md` 中只读迭代语义已确定
  - 预计: 10 min

- [ ] **T5**: 实现 `par_map`
  - 文件: `src/parallel/map.rs`
  - 内容: 纯并行逐元素映射入口，执行策略参数由 `dispatch.rs` 统一传入
  - 测试: `test_par_map_parallel_path`
  - 前置: T4
  - 预计: 10 min

- [ ] **T5a**: 实现 `par_zip_map`
  - 文件: `src/parallel/map.rs`
  - 内容: 二元广播逐元素纯并行入口，供 `math` 模块消费
  - 测试: `test_par_zip_map_matches_serial_add`, `test_par_zip_map_broadcast_rhs_scalar`, `test_par_zip_map_shape_mismatch`
  - 前置: T4, `math` 广播语义已确定
  - 预计: 10 min

- [ ] **T6**: 实现 `par_reduce_impl` 与 `par_sum`
  - 文件: `src/parallel/reduce.rs`
  - 内容: 并行归约、identity 合并、语义对齐调用方选定的串行基线
  - 测试: `test_par_sum_matches_serial`, `test_par_sum_empty_matches_identity`
  - 前置: T4, `13-reduction.md` 归约语义已确定
  - 预计: 10 min

- [ ] **T7**: 实现 `par_dot`
  - 文件: `src/parallel/reduce.rs`
  - 内容: 运行时 `ndim() == 1` / 长度一致性检查、并行内积、错误返回与空数组单位元语义
  - 测试: `test_par_dot_matches_serial`, `test_par_dot_shape_mismatch`, `test_par_dot_empty_identity`
  - 前置: T6
  - 预计: 10 min

### Wave 3: 线程池与异常传播

- [ ] **T8**: 实现 `ParallelPool`
  - 文件: `src/parallel/mod.rs`
  - 内容: 自定义 `rayon::ThreadPool` 包装，不改变公开 API 结果语义
  - 测试: `test_parallel_pool_preserves_semantics`
  - 前置: T5, T6, T7
  - 预计: 10 min

- [ ] **T9**: 完成错误与 panic 传播收口
  - 文件: `src/parallel/checked.rs`
  - 内容: `XenonError` 透传、panic 不吞掉
  - 测试: `test_parallel_error_propagation`, `test_parallel_panic_propagation`
  - 前置: T5, T6, T7
  - 预计: 10 min

### Wave 4: 配置与回归验证

- [ ] **T10**: 补齐 feature gate 与配置矩阵测试
  - 文件: `src/parallel/` (全部子文件), `tests/parallel_feature.rs`
  - 内容: 默认关闭、`--features parallel` 构建、单线程/多线程分支验证
  - 测试: `cargo test`, `cargo test --features parallel`
  - 前置: T4-T9
  - 预计: 10 min

并行关系图：

```text
Wave 1: [dispatch.rs external prerequisites]

Wave 2: [T4] -> [T5] -> [T5a] -> [T6] -> [T7]
                 │         │        │       │
                 └─────────┴────────┴──┬────┘
                                        ▼
Wave 3:                           [T8] [T9]
                                        │
                                        ▼
Wave 4:                                [T10]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置                                           | 目的                                                             |
| ----------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| 单元测试                | `src/parallel/` 各子文件内联测试模块                 | 验证并行入口、归约与错误传播                                     |
| 集成测试                | `tests/parallel_feature.rs` 或等效位置         | 验证跨模块语义与 feature gate 行为                               |
| 边界测试                | 与并行测试配套组织                             | 覆盖空张量、单元素、非连续视图、单线程环境                       |
| 属性测试（按需）        | 当前版本不强制                                 | 当前模块以确定性路径与语义对齐为主，暂无必须引入的随机不变量测试 |
| Feature gate / 配置测试 | `cargo test`, `cargo test --features parallel` | 验证默认关闭与启用并行后语义不变                                 |
| 类型边界 / 编译期测试   | trait 约束测试或编译期失败测试                 | 验证 `bool` 不参与 `par_sum` / `par_dot` 等非法组合              |

### 8.2 单元测试清单

| 测试函数                              | 测试内容                                            | 优先级 |
| ------------------------------------- | --------------------------------------------------- | ------ |
| `test_par_map_parallel_path`          | `dispatch.rs` 选中并行路径后结果与串行一致          | 高     |
| `test_par_zip_map_matches_serial_add` | 二元逐元素并行加法结果与串行一致                     | 高     |
| `test_par_zip_map_broadcast_rhs_scalar` | 右侧标量广播时并行路径与串行一致                  | 高     |
| `test_par_sum_matches_serial`         | 并行 `sum` 与串行语义一致                           | 高     |
| `test_par_dot_matches_serial`         | `par_dot` 与串行结果一致                            | 高     |
| `test_parallel_error_propagation`     | 并行 `Err` 及时上传                                 | 高     |
| `test_parallel_panic_propagation`     | 并行 panic 不被吞掉                                 | 高     |

### 8.3 边界测试场景表

| 场景                 | 预期行为                                                                            |
| -------------------- | ----------------------------------------------------------------------------------- |
| 空数组 `len == 0`    | `par_sum()` 返回加法单位元；`par_dot()` 在两个长度为 `0` 的一维输入上返回加法单位元 |
| 单元素张量           | 若 `dispatch.rs` 仍选择并行路径，结果与串行一致                                     |
| 非连续视图           | 若 `dispatch.rs` 选择并行路径，结果仍与串行一致                                     |
| 单线程环境           | 不由 `parallel/` 自行处理；调用方不应选择并行路径                                   |
| 非一维输入           | `par_dot()` 在任一输入 `ndim() != 1` 时返回错误                                     |
| 长度不匹配的一维输入 | `par_dot()` 返回 `XenonError::DimensionMismatch { operation, expected, actual }`    |
| 二元广播逐元素输入   | `par_zip_map()` 在广播兼容时返回与串行 `add/sub/mul/div` 一致的结果                 |

### 8.3a Send/Sync/借用边界测试计划

- [ ] 验证只读输入视图在并行 worker 间共享时不产生可写别名。
- [ ] 验证闭包捕获类型不满足 `Send` / `Sync` 时保持编译期拒绝。
- [ ] 验证输出缓冲区按 worker 独占区间写入，不跨 worker 共享可写借用。

### 8.3b `需求说明书 §28.4` 边界占位

- [ ] 补充空张量、单元素、极端形状与广播退化形状在并行路径下的边界测试。
- [ ] 补充浮点/复数 `NaN`、`±Inf`、`±0.0` 组合下的归约与内积边界测试。
- [ ] 补充整数 `sum` / `dot` 在溢出路径上的 panic 传播边界测试。

### 8.4 属性测试与不变量

| 不变量                                                    | 测试方法                                 |
| --------------------------------------------------------- | ---------------------------------------- |
| `par_map` 与串行 `map` 在相同输入上产出相同形状与逐元素值 | 对整数类型可按多组形状和布局做表驱动校验 |
| `par_zip_map` 与串行广播二元运算在相同输入上产出相同形状与逐元素值 | 对 `add/sub/mul/div` 做表驱动校验 |
| `par_sum` / `par_dot` 在相同执行路径和配置下结果确定      | 对相同输入重复运行并比较结果             |

### 8.5 Feature gate / 配置测试

| 配置                | 验证点                                                     |
| ------------------- | ---------------------------------------------------------- |
| 默认配置            | 可选并行默认关闭，默认构建可编译                           |
| 启用 `parallel`     | `par_map` / `par_sum` / `par_dot` 可用，结果与串行路径一致 |
| 启用 `parallel` + broadcast op | `math` 通过 `par_zip_map` 走并行路径时结果与串行广播一致 |
| 单线程运行          | `dispatch.rs` 不应选择 `parallel/` 路径                    |
| 启用并行 + 嵌套调用 | 嵌套并行防护由 `dispatch.rs` 负责                           |

### 8.6 类型边界与编译期测试

| 场景                                | 测试方式                                 |
| ----------------------------------- | ---------------------------------------- |
| `bool` 不参与 `par_sum` / `par_dot` | 编译期 trait 约束测试                    |
| 非法 feature 组合                   | 配置矩阵测试                             |
| 非法阈值参数                        | 转由 `dispatch.rs` 的运行时测试覆盖      |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向         | 对方模块     | 接口/类型                                         | 约定                                                 |
| ------------ | ------------ | ------------------------------------------------- | ---------------------------------------------------- |
| 消费（输入） | `tensor`     | `Tensor<A, D>`, `TensorBase<S, D>`                | 调用前已满足 shape、layout、类型约束                 |
| 消费（输入） | `parallel`   | `TensorBase::par_iter()`, `ParElements<'a, A, D>` | 二者均为 `pub(crate)` 内部入口，只提供单输入只读并行遍历 |
| 消费（输入） | `parallel`   | `par_zip_map()`                                  | `math` 模块经 `dispatch.rs` 完成路径选择后调用，仅为 crate 内部能力 |
| 消费（输入） | `error`      | `XenonError`                                      | 可恢复错误统一复用项目错误模型                       |
| 被调用（输出） | 上层语义模块 / `dispatch.rs` | `par_map` / `par_sum` / `par_dot` / `par_zip_map` | 仅在 `dispatch.rs` 已选中并行路径后执行               |
| 产出（输出） | 上层语义模块 | `Tensor<B, D>` 或 `Result<A, XenonError>`         | 并行与串行路径保持相同外部语义                       |

### 9.2 数据流

```text
math / reduction / matrix call dispatch entry
    │
    ├── query metadata (.len(), .is_f_contiguous(), alignment, simd support)
    ├── dispatch::select_exec_path(...)
    │       ├── serial / simd path stays outside parallel/
    │       └── parallel path enters parallel/
    ├── parallel path uses par_iter() / par_zip_map() / parallel reduce
    └── return Tensor or Result with unchanged public semantics
```

### 9.3 所有权与生命周期约定

> **约定：** `par_iter()` 借用输入张量的只读视图，所有权不转移。并行执行只读取逻辑元素，不得暴露共享可写访问权；输出张量若需要拥有结果数据，必须在本模块内构造新的拥有型结果。

---

## 10. 错误处理与语义边界

| 主题              | 说明                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Recoverable error | `par_dot()` 的长度不兼容返回 `XenonError::DimensionMismatch { operation, expected, actual }`；`par_zip_map()` 的元素总数溢出返回 `InvalidShape { expected_elements: 0, actual_elements: 0, offending_dim: None, reason: Some(Cow::Borrowed("element count overflow")) }` |
| Panic             | 归约中的整数溢出仍属于不可恢复错误，必须 panic，而不是包装为 `XenonError`                                                                                          |
| 路径一致性        | `dispatch.rs` 负责执行路径选择；一旦进入 `parallel/`，并行路径必须返回与调用方串行基线相同形状、相同错误类别，以及满足同一数值语义约束的结果                    |
| 容差边界          | 浮点与复数若存在执行路径相关的已知舍入差异，只能落在 `需求说明书 §28.3` 与 `需求说明书 §28.5` 允许且已文档化的范围内；容差与比较规则统一遵循 `00-coding.md §7.4` 的定义。 |

```rust,ignore
XenonError::DimensionMismatch {
    operation: "par_dot",
    expected: lhs.len(),
    actual: rhs.len(),
}
```

路径语义边界：

- 并行模块本身不新增专属错误枚举；公开错误必须复用 `26-error.md` 中的统一模型。
- 自定义线程池类参数若存在非法值，应返回 `InvalidArgument { operation, argument, expected, actual, axis, axis_len, start, end, shape }`。
- 若未来在 `parallel/` 内部新增广播错误构造，必须使用 `XenonError::BroadcastError { operation, lhs_shape, rhs_shape, attempted_target_shape, axis }` 这一权威字段集合；当前 `par_zip_map()` 不承担广播兼容性校验。
- panic 与 `Err(XenonError)` 都不得被吞掉；并行执行中发生的错误须至少传播一个，不保证传播“第一个”发生的错误。
- 执行路径裁决由 `dispatch.rs` 负责；`parallel/` 不新增路径选择语义。

### 10.1 浮点/复数并行归约容差

- 浮点与复数并行归约允许与标量路径不同的合并顺序；该差异视为合法实现细节，但必须受 `需求说明书 §28.3` 文档化容差约束。
- 容差与比较规则统一遵循 `00-coding.md §7.4` 的定义。
- 同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。
- `NaN`：按 IEEE 754 语义检查（`NaN !=` 任何值），不使用数值容差。
- `±Inf`：必须同号同类。
- 并行归约/内积结果为有限零值时，符号必须与串行基线一致；做不到则不得启用该并行路径。
- 容差规则仅适用于有限值结果。
- 复数按实部、虚部分别适用同一文档化规则；若某一并行实现无法满足该容差或无法提供固定 chunking + fixed merge tree 的确定性约束，则必须回退串行或调整分块/合并策略。

### 10.2 线程安全

并行后端不改变 `TensorBase<S, D>` 的 `Send` / `Sync` 判定。线程安全性仍由元素类型与存储模式共同决定（参见 `25-safety.md`）。

---

## 11. 设计决策记录

### 决策 1：并行阈值采用全局原子状态

| 属性     | 值                                                           |
| -------- | ------------------------------------------------------------ |
| 决策     | 运行时并行阈值状态迁移至 `dispatch.rs`，继续采用全局原子状态 |
| 理由     | 读取开销固定、无需锁、便于所有执行路径共享同一裁决基线       |
| 替代方案 | 每次调用显式传参 —— 放弃，会让公开 API 过于分裂              |
| 替代方案 | 使用互斥锁配置对象 —— 放弃，对热点路径不必要                 |

### 决策 2：嵌套并行进入失败时必须回退串行

| 属性     | 值                                                                             |
| -------- | ------------------------------------------------------------------------------ |
| 决策     | `ParallelGuard` 迁移至 `dispatch.rs`；进入失败不报错、不 panic，而是选择串行回退 |
| 理由     | `需求说明书 §9.2` 明确禁止库内部二次并行；该场景是执行策略问题而非用户输入错误 |
| 替代方案 | 允许库内部继续二次并行 —— 放弃，违反需求                                       |
| 替代方案 | 将嵌套并行视为 recoverable error —— 放弃，会污染公开 API 语义                  |

补充说明：`ParallelPool` 内部调用同样必须经过 `dispatch.rs` 中的 `ParallelGuard`。若用户在自定义 pool 中再次调用内部并行后端，dispatch helper 会把 `ParallelContext` token 捕获到 Rayon worker 闭包中，并在二次派发时自动回退串行，与全局线程池行为一致；同时不允许嵌套 `ParallelPool` 实例，以避免引入额外调度语义。

### 决策 3：并行模块不新增专属公开错误类型

| 属性     | 值                                                      |
| -------- | ------------------------------------------------------- |
| 决策     | 统一使用 `XenonError` 表达 shape 与参数错误             |
| 理由     | 保持跨模块诊断字段与错误类别一致，满足 `需求说明书 §27` |
| 替代方案 | 定义 `ParallelError` —— 放弃，会破坏统一错误模型        |
| 替代方案 | 以 panic 处理非法阈值 —— 放弃，不符合可恢复错误要求     |

### 决策 4：并行模块不包含串行回退

| 属性     | 值                                                                                         |
| -------- | ------------------------------------------------------------------------------------------ |
| 决策     | `parallel` 只提供纯并行执行入口，不包含串行回退路径                                           |
| 理由     | 执行路径裁决（串行 vs SIMD vs 并行）由 `dispatch.rs` 统一承担，`parallel` 不需要自行判断           |
| 替代方案 | 在 `parallel` 内保留串行回退 —— 放弃，会导致 `dispatch` 判断与 `parallel` 内部判断重复             |

### 决策 5：二元逐元素并行能力以 `par_zip_map()` 形式提供给 `math`

| 属性     | 值                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------- |
| 决策     | `parallel` 提供 `pub(crate)` 级 `par_zip_map()`，由 `math` 在广播裁决完成后调用               |
| 理由     | 满足 `需求说明书 §9.2` 对逐元素二元运算并行路径的要求，同时不把通用多输入并行迭代器暴露为公开 API |
| 替代方案 | 仅保留 `par_map` —— 放弃，无法覆盖 `add/sub/mul/div` 广播逐元素并行需求                        |
| 替代方案 | 将二元广播并行逻辑直接写进 `math` —— 放弃，会复制 `dispatch.rs` 之外的并行执行实现与错误传播策略 |

### 决策 6：执行路径裁决由 `dispatch.rs` 统一收口

| 属性     | 值                                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| 决策     | `math` / `reduction` / `matrix` 调用 `dispatch::select_exec_path()` 决定执行路径 |
| 理由     | 统一串并阈值、SIMD 能力、数据对齐判断，避免多个模块各自实现分支树                   |
| 替代方案 | 每个模块自行判断 serial/parallel/SIMD —— 放弃，易产生阈值漂移和行为不一致           |

---

## 12. 性能描述

### 12.1 复杂度标注

- `par_map()`：时间 `O(n)`，额外结果空间 `O(n)`。
- `par_sum()`：时间 `O(n)`，额外工作空间取决于 `rayon` 分块；逻辑额外空间 `O(1)`。
- `par_dot()`：时间 `O(n)`，逻辑额外空间 `O(1)`。

### 12.2 缓存与执行特征

| 场景         | 实现路径     | 缓存友好性 | 说明                           |
| ------------ | ------------ | ---------- | ------------------------------ |
| 连续 F-order | 顺序分块并行 | 高         | 适合作为并行路径的主要收益来源 |
| 非连续视图   | 步长访问并行 | 中到低     | 缓存命中率下降，但仍保持纯并行语义 |
| 小张量       | 由 `dispatch.rs` 裁决 | 高         | 是否进入 `parallel/` 不在本模块职责内 |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------- |
| `std` only | 本模块依赖 `rayon`，且项目基线仅支持 `std`；不讨论 `no_std`                                  |
| MSRV       | Rust 1.85+                                                                                   |
| 单 crate   | 设计保持在 Xenon 单 crate 内，不引入额外 crate 拆分                                          |
| SemVer     | 并行后端入口属于 `pub(crate)` 内部契约；其语义仍需在 crate 内保持稳定，但不构成面向最终用户的独立公开 API 承诺 |
| 最小依赖   | 仅使用允许的可选依赖 `rayon`，默认关闭                                                       |

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-14 |
| 1.1.0 | 2026-04-14 |
| 1.1.1 | 2026-04-14 |
| 1.1.2 | 2026-04-14 |
| 1.2.0 | 2026-04-15 |
| 1.2.1 | 2026-04-15 |
| 1.3.0 | 2026-04-15 |
| 1.3.1 | 2026-04-15 |
| 1.3.2 | 2026-04-15 |
| 1.3.3 | 2026-04-16 |
| 1.3.4 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
