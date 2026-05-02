# 并行后端模块设计

> 文档编号: 09
> 模块目录: src/parallel/
> 任务阶段: Phase 5
> 前置文档: 01-architecture.md, 03-element.md, 07-tensor.md, 26-error.md
> 需求参考: 需求说明书 §1、§9、§12 - §14、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位/概述

并行后端模块是 Xenon 的可选执行后端，通过 `rayon` 为逐元素映射、二元逐元素运算与归约提供纯数据并行能力。该模块默认关闭，仅在启用 `parallel` feature 时参与构建。当前版本覆盖内部 `par_map`、供 `math` 模块消费的 `par_zip_map`、并行 `sum` / `dot`；本模块不负责串行回退、阈值控制或嵌套并行检测，这些执行路径裁决职责已迁移至 `dispatch.rs`。

### 1.1 职责边界表

| 职责           | 包含                                                | 不包含                           |
| -------------- | --------------------------------------------------- | -------------------------------- |
| 并行逐元素执行 | `par_map`、`par_zip_map`、基于视图的并行遍历        | 通用多输入同步并行公开迭代器 API |
| 并行归约       | `par_sum`、`par_dot`、内部 `par_reduce_impl`        | 矩阵乘法、矩阵分解、GPU 后端     |
| 线程池封装     | 内部 `ParallelPool` 改变执行上下文而不改变 API 语义 | 新的公开调度语义或额外第三方依赖 |

### 1.2 设计原则

| 原则           | 体现                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| 语义一致性     | 并行路径不得改变公开 API 的形状、错误类别和数值语义；路径裁决见 §6.1、决策 4                       |
| 最小能力边界   | 当前版本只覆盖 `par_map`、`par_zip_map`、`par_sum`、`par_dot`，不扩展到 GPU 或通用多输入同步公开接口 |
| 可选依赖最小化 | 仅在 `parallel` feature 下引入 `rayon`，默认关闭                                                     |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                            |
| -------- | ------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §1、§9、§12 - §14、§27、§28                                          |
| 范围内   | 可选数据并行、逐元素运算（含二元广播）/归约/内积的并行执行路径                  |
| 范围外   | GPU 后端、自动任务图调度、通用多数组 lock-step 并行公开接口、额外第三方依赖     |
| 非目标   | 不把文档改成 `no_std`，不增加除 `rayon` 之外的外部依赖，不扩展当前并行能力集合  |

---

## 3. 文件位置

```text
src/parallel/
├── mod.rs         # Module entry, re-exports, ParallelPool
├── iter.rs        # ParElements and TensorBase::par_iter()
├── map.rs         # par_map, par_zip_map
├── reduce.rs      # par_reduce_impl, par_sum, par_dot
└── checked.rs     # par_map_checked and error/panic propagation
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```text
src/parallel/
├── rayon (optional)         # ThreadPool, ParallelIterator, current_num_threads
├── crate::tensor            # Tensor, TensorBase, TensorView
├── crate::element           # Element, Numeric
├── crate::dimension         # Dimension
├── crate::dispatch          # ParallelExecStrategy (defined in dispatch.rs)
├── (module-owned)           # ParElements and par_iter() entry belong to parallel/
└── crate::error             # XenonError
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                         |
| ----------- | -------------------------------------------------------------------------------------------------------- |
| `rayon`     | `rayon::ThreadPool`, `rayon::current_num_threads`, `rayon::iter::ParallelIterator`                       |
| `tensor`    | `Tensor<A, D>`, `TensorBase<S, D>`, `TensorView<'a, A, D>`, `.len()`, `.raw_dim()`, `.is_f_contiguous()` |
| `element`   | `Element`, `Numeric`                                                                                     |
| `dimension` | `Dimension`                                                                                              |
| `dispatch`  | `ParallelExecStrategy`, `ParallelGuard`                                                                  |
| `parallel`  | `ParElements<'a, A, D>`, `TensorBase::par_iter()`, `par_zip_map()`                                       |
| `error`     | `XenonError`, `XenonError::ShapeMismatch`, `XenonError::InvalidShape`, `XenonError::InvalidArgument`, `InvalidShapeKind::ProductOverflow`, `InvalidArgumentKind::OperationSpecific`, `Cow<'static, str>` |

### 4.3 依赖合法性

| 项目           | 说明                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------ |
| 新增第三方依赖 | `rayon`（可选）                                                                            |
| 合法性结论     | 合法；符合 `需求说明书 §1.2` 对最小依赖的限制，以及 `需求说明书 §9.2` 对可选并行能力的要求 |
| 替代方案       | 仅用 `std::thread` 不能无损提供当前所需的并行迭代与线程池抽象，因此不采用                  |

### 4.4 依赖方向

依赖方向：单向向上。`parallel` 只提供纯并行执行入口，不包含串行回退（路径裁决见 §6.1）。`parallel` 通过 `crate::dispatch` 引用 `ParallelExecStrategy` 类型（定义于 `dispatch.rs`），但不依赖 dispatch 的路径裁决实现逻辑。`ParElements` 与 `TensorBase::par_iter()` 归属 `parallel` 模块本身，不属于 `iter` 模块。并行路径只建立在上层已完成的张量形状、布局与类型约束之上；广播形状裁决由 `math` 调用侧先完成，再以 `output_dim` 形式传入。

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
- `ParallelGuard`、阈值状态与嵌套并行防护逻辑已迁移至内部 `dispatch.rs` 模块；本节仅保留与线程池执行上下文直接相关的并行后端状态。

### 5.2 内部执行入口与可见性

**可见性说明：** `parallel` 是 `pub(crate)` 内部后端；所有执行后端函数与类型（包括 `par_map`、`par_zip_map`、`par_sum`、`par_dot`、`ParallelPool`、`ParElements`）均保持 `pub(crate)`，仅供 `math` / `reduction` / `matrix` 等语义模块通过 `dispatch.rs` 自动调用。

**执行策略：** 阈值配置与嵌套并行防护由 `dispatch.rs` 统一管理（见 §6.1、决策 4）；本模块仅通过 `ParallelExecStrategy` 接收 dispatch 已裁决的执行参数，不提供独立的公开阈值配置接口。

### 5.3 内部执行策略参数规范

| 参数                 | 类型            | 默认值                   | 说明                 |
| -------------------- | --------------- | ------------------------ | ---------------------|
| `max_workers`        | `Option<usize>` | `None`（使用线程池默认） | 最大并行工作线程数   |
| `chunk_size`         | `Option<usize>` | `None`（自动计算）       | 每个 chunk 的元素数  |

配置入口不对外暴露。`parallel` 模块仅通过 `ParallelExecStrategy` 接收 dispatch 已裁决完成的执行阶段参数字段；`parallel_threshold` 的权威入口位于 `dispatch.rs`。

### 5.4 `ParallelExecStrategy` 参数校验规则

`ParallelExecStrategy` 的字段合法性由 `dispatch.rs` 在构造时（`ParallelExecStrategy::new(...) -> Result<Self, XenonError>`）一次性校验完成（参见 30-dispatch.md §5.3、决策 8）。`parallel` 模块收到的策略实例已经满足以下范围；若违反则视为 `dispatch.rs` 的内部 bug，可触发 `debug_assert!`，但**不再**由 `parallel` 自己重新返回 `InvalidArgument`。

| 字段          | 合法范围                          | 默认值（dispatch 端） | 非法值的处置                                                |
| ------------- | --------------------------------- | --------------------- | ----------------------------------------------------------- |
| `max_workers` | `Some(1..=pool_size)` 或 `None`   | `None`                | dispatch 端在 `new()` 中拒绝并返回 `InvalidArgument`        |
| `chunk_size`  | `Some(n)` where `n > 0` 或 `None` | `None`                | dispatch 端在 `new()` 中拒绝并返回 `InvalidArgument`        |

**与 §5.5 函数签名的一致性**：因为参数合法性已由 dispatch 保证，`par_map` / `par_reduce_impl` / `par_sum` 等签名不返回 `Result`（它们只可能因为整数溢出 panic 或正常完成，没有可恢复错误通道），不再存在“函数签名无法返回 `InvalidArgument`”的不一致问题。

### 5.5 函数签名

**定义位置**：`ParallelExecStrategy` 定义于 `dispatch.rs`（30-dispatch.md §5.3，字段为 `pub(crate)` 私有，构造仅通过 `ParallelExecStrategy::new(chunk_size, max_workers) -> Result<Self, XenonError>`）。`parallel` 通过 `crate::dispatch` 引用，仅做只读消费。

```rust,ignore
// Authoritative definition lives in dispatch.rs; reproduced here as reference only.
// Fields are pub(crate); external construction is forbidden — use
// `ParallelExecStrategy::new(chunk_size, max_workers)` for validated construction.
pub(crate) struct ParallelExecStrategy {
    pub(crate) chunk_size: Option<usize>,
    pub(crate) max_workers: Option<usize>,
}

#[cfg(feature = "parallel")]
pub(crate) fn par_map<S, A, B, D, F>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
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
    _guard: ParallelGuard,
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
    _guard: ParallelGuard,
    identity: ID,
    op: F,
) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Send + Sync + Clone,
    F: Fn(A, A) -> A + Send + Sync,
    ID: Fn() -> A + Send + Sync + Clone;

#[cfg(feature = "parallel")]
pub(crate) fn par_sum<S, A, D>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_dot<SL, SR, A, DL, DR>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
) -> Result<A, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = A>,
    DL: Dimension,
    DR: Dimension,
    A: Numeric + Send + Sync;
```

**`_guard: ParallelGuard` 设计要点**（与 30-dispatch v1.1.0 决策 7 一致）：
- `_guard` 由 `dispatch::select_exec_path()` 在裁决到 `ExecPath::Parallel` 时返回 `Some(ParallelGuard)`，并由调用侧（`math` / `reduction` / `matrix`）按值移交到 `parallel` 后端入口。
- `parallel` 在函数体内只持有 `_guard` 直至并行执行结束；`ParallelGuard::drop()` 自动清除 thread-local 嵌套防护标记。
- 这样 “选中并行路径” 与 “进入并行临界区” 在调用图上原子绑定：调用方无法忘记 acquire guard，也无法在函数返回后越界使用 guard。
- `ParallelGuard` 类型在 `parallel` feature 关闭时由 `dispatch.rs` 提供为不可构造的占位类型，相关并行入口本身整体被 `#[cfg(feature = "parallel")]` 排除，签名层不会泄露。

- `par_dot()` 在类型层面接受任意 `Dimension` 输入，以便与更通用的上层张量调用路径对接；但其语义契约仍限定为一维向量内积，因此实现必须在运行时检查 `lhs.ndim() == 1`、`rhs.ndim() == 1`，并在进入并行归约前再次确认两侧逻辑长度一致。
- 复数内积采用共轭线性定义：`result = sum(conj(lhs_i) * rhs_i)`，与 `08-simd.md` §6.6 中复数 dot kernel 的共轭线性方向完全一致。
- `Numeric` trait 定义于 `03-element.md` §5.2，提供通用数值运算能力标记（`Element + Add + Sub + Mul + Div + Neg + conjugate`）。
- 整数 `par_sum` / `par_dot` 在并行路径中，每个分片独立执行 checked 算术；若任一分片检测到溢出，panic 将在并行收集完成后传播。诊断仲裁必须按逻辑 chunk 索引确定：始终报告首个失败 chunk（按逻辑索引顺序）。**若实现无法保证这一确定性，`dispatch.rs` 必须在 `select_exec_path()` 的 `ExecPath::Parallel` 分支前提中预先排除该输入**（即由 dispatch 不选择并行，而非 parallel 内部回退；这与决策 4 “parallel 不包含串行回退” 一致）。
- 闭包 bound 统一要求 `Send + Sync`：`F: Fn(...) -> ... + Send + Sync`、`ID: Fn() -> A + Send + Sync + Clone`。`Send` 是 rayon worker 跨线程移动闭包数据的必要前提；仅 `Sync` 不足以覆盖 closure 在某些 by-value 工作单元中的所有权迁移场景。

### 5.6 并行迭代入口

```rust,ignore
#[cfg(feature = "parallel")]
pub(crate) struct ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
    chunk_size: Option<usize>,
    max_workers: Option<usize>,
}
```

`ParElements` 通过同时实现 `rayon::iter::IndexedParallelIterator`（`Item = &'a A`，由其 supertrait `ParallelIterator` 自动得到）以及对应的 `rayon::iter::plumbing::Producer` 桥接来提供并行遍历能力：

- **producer 拆分（修复 Blocker B7）**：`ParElements` 内部实现 `rayon::iter::plumbing::Producer`，由 `with_producer()` 把 view + 当前逻辑区间 `[lo, hi)` 转交给 rayon scheduler；rayon 通过 `producer.split_at(mid)` 将逻辑区间均分为两个互不重叠的子 producer：
  - F-contiguous 子区间：`split_at` 直接对 base pointer 做指针算术 `ptr.add(mid - lo)`，两个子 producer 持有不相交的连续切片。
  - 非连续 / 转置视图：`split_at` 不切分物理切片，而是切分逻辑区间 `[lo, mid)` 与 `[mid, hi)`；每个 producer 内部维护一个轻量的 stride 状态机，按 F-order 对该子区间进行逐元素 stride 访问；rayon 仍可保证两个子 producer 不共享同一逻辑元素。
  - **不变量**：任意 producer 拆分序列覆盖原区间正好一次且互不相交（`Disjoint Coverage`），由 `with_producer` 的契约和 `split_at` 的实现共同保证；该不变量是 `IndexedParallelIterator` 的安全前提，也是 `par_map_checked` 顺序恢复（§6.6）能成立的基础。
- **逻辑顺序契约**：`IndexedParallelIterator` 要求 producer 按 `[0, n)` 的索引顺序覆盖输出；rayon worker 之间执行顺序未定，但每个元素仅被访问一次，且 `collect_into_vec(&mut Vec<_>)` / `collect()` 等 indexed 收集 API 会**按索引位置**写入结果，与 worker 完成顺序无关。
- 对于 F-contiguous 布局，直接按连续内存切片分割以获得最佳缓存局部性；对于非连续布局（如转置视图），退化为逻辑区间 + 逐元素步长访问。
- 分片粒度由 `chunk_size` 和 `max_workers` 字段控制：若 `chunk_size` 为 `Some(n)`，每个分片至多包含 `n` 个元素；若为 `None`，通过 `compute_safe_chunks(total, num_threads)` 自动计算（定义于 `src/parallel/mod.rs`，见 01-architecture.md §5.2a），其中 `num_threads` 取 `max_workers` 或 `rayon::current_num_threads()`。
- `par_iter()` 返回使用默认策略（`chunk_size: None`, `max_workers: None`）的 `ParElements`；`ParElements::with_strategy()` 接受显式策略参数，供 `par_map_checked` 等需要精确控制分块的内部入口使用。

```rust,ignore
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

### 5.7 Good / Bad 对比示例

```rust,ignore
// Good - shape mismatch stays in Result.
let dot = par_dot(&lhs, &rhs, strategy)?;

// Bad - converting recoverable shape mismatch into unwrap panic.
let dot = par_dot(&lhs, &rhs, strategy).unwrap();
```

---

## 6. 内部实现设计

### 6.1 路径选择算法

路径选择算法已迁移至 `dispatch.rs` 模块。`parallel` 仅在被 dispatch 选为执行路径时被调用。

### 6.2 核心执行路径

```text
dispatch-selected parallel entry (receives ParallelGuard by value)
    │
    ├── receive validated tensor metadata and closure
    ├── split logical work into fixed chunks (via Producer::split_at)
    ├── execute rayon parallel path
    │      │
    │      └── inside each worker chunk:
    │             optionally call into simd backend (SIMD admission per chunk)
    │             — see 08-simd.md v2.0.0 决策 5（worker 内 SIMD）
    └── propagate panic / Err without swallowing; drop guard at end
```

- `parallel` 假定调用方已经完成阈值、线程环境、嵌套并行治理判断（由 `dispatch.rs` 的 `select_exec_path()` 完成）。
- 并行函数只负责固定 chunking + 执行 `rayon` 并行迭代（语义一致性要求见 §1.2）。
- **Worker 内 SIMD（v2.0 起）**：单个 worker 拿到 chunk 后，可在 chunk 内部独立调用 `simd` 后端的 `pub(crate)` kernel（如 `dispatch_vector_binary_op`），按 `08-simd.md` §5.4 的 SIMD admission（连续性、对齐、长度阈值、操作覆盖、ISA）独立判断；不进入 SIMD 时回退到该 chunk 内的标量循环。chunk 间合并顺序仍由 `parallel` 模块的固定 chunking + 固定 merge tree 控制，跨 chunk 的语义一致性不被 SIMD 影响。

### 6.3 二元逐元素并行路径

分块策略统一通过 `compute_safe_chunks(total, num_workers)` 计算，该函数定义于 `src/parallel/mod.rs`（参见 01-architecture.md §5.2a）。这避免了多处重复内联公式造成的不一致风险。

```rust,ignore
#[cfg(feature = "parallel")]
pub(crate) fn par_zip_map<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
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
    // checked_size overflow → InvalidShape with the closed-enum kind defined
    // in 26-error.md v3.0.0 §5.1 (InvalidShapeKind::ProductOverflow).
    let total = output_dim.checked_size().map_err(|_| XenonError::InvalidShape {
        operation: Cow::Borrowed("par_zip_map"),
        shape: output_dim.slice().to_vec(),
        kind: InvalidShapeKind::ProductOverflow,
        offending_dim: None,
    })?;

    // num_threads is taken from the validated strategy; falls back to
    // rayon::current_num_threads() only when strategy.max_workers is None.
    // (Single source of truth — see 30-dispatch.md §5.3 for strategy validation.)
    let num_threads = strategy
        .max_workers
        .unwrap_or_else(rayon::current_num_threads);

    // Use compute_safe_chunks from src/parallel/mod.rs (declared in 01-architecture.md §5.2a)
    // to centralize chunk-size policy and apply safety bounds.
    let chunk_size = strategy
        .chunk_size
        .unwrap_or_else(|| crate::parallel::compute_safe_chunks(total, num_threads));

    // Build broadcast-compatible read-only chunk views for lhs / rhs via
    // ParElements-style IndexedParallelIterator + Producer split (see §5.6).
    // Each worker chunk MAY independently call into the simd backend
    // (08-simd.md v2.0.0 决策 5; admission per chunk).
    // Use indexed collect (.collect_into_vec / collect()) to recover F-order
    // result placement regardless of worker completion order.
    // Panic propagation follows Rayon defaults; see §6.7 and §10.
    unimplemented!()
}
```

- `par_zip_map()` 是二元逐元素并行路径的统一设计入口，供 `math` 模块中的 `add` / `sub` / `mul` / `div` 广播运算消费，不直接暴露为公开用户 API。
- `par_zip_map()` 接收的 `lhs`、`rhs` 与 `output_dim` 必须已由调用侧完成兼容性验证；广播裁决（含输出 rank/shape 计算）属于 `math` 模块职责，`parallel/` 不重复做形状推导。
- 广播处理顺序固定为：先由 `math` 模块验证 `lhs` / `rhs` 广播兼容并产出 `output_dim`，再由 `parallel` 按外轴/块状多维 tile 分块；默认 chunk_size 通过 `compute_safe_chunks(total, num_threads)` 确定（定义于 `src/parallel/mod.rs`，见 01-architecture.md §5.2a），作为 tile 目标工作量上界，其中 `num_threads = strategy.max_workers.unwrap_or_else(rayon::current_num_threads)`（与 §5.6 一致：`max_workers` 优先，回退到 rayon 默认）。chunk 间合并按 `IndexedParallelIterator` 的索引顺序写入输出 buffer。每个 chunk 为两个输入分别构造与该 tile 对应、且仍与 `output_dim` 兼容的只读 sub-view。若某一侧是广播轴（stride 为 `0` 或逻辑重复维），chunk 视图保持该广播语义，不做物理复制。`DL`、`DR`、`DO` 独立建模，以表达输入与输出 rank 可能不同的广播结果。
- 广播 chunk 映射草图：优先按 `output_dim` 的外轴边界生成块状多维 tile，使 chunk 在输出空间内保持可直接切片的矩形子域；若某些退化形状无法形成理想矩形 tile，则实现可退化为“逻辑区间 + 逐元素广播投影”的内部执行形式，而不是要求把任意线性区间整体重建成单个 broadcast sub-view。对输出维中的广播轴，输入侧固定复用同一逻辑坐标；对非广播轴，chunk 保持对应 tile 的区间跨度。实现不得为广播轴做物理展开或额外分配。
- `par_zip_map` 仅包含并行执行逻辑；若调用发生，表示 `dispatch.rs` 已确认当前输入适合走并行路径。
- `par_zip_map()` 作为内部并行入口，假定广播兼容性已由调用方验证，不再额外定义单独的 checked 变体，也不依赖 `BroadcastError`。此为内部前置条件。违反时视为内部 bug，可触发 debug assert，但不得破坏内存安全或对外错误模型。release 模式下行为保持语义定义，不引入未指定行为。panic 与 `Err` 传播语义参见 §6.7 与 §10。
- **唯一仍由本函数返回的可恢复错误**是 `output_dim.checked_size()` 的整型溢出，归类为 `XenonError::InvalidArgument { operation: "par_zip_map", kind: InvalidArgumentKind::ShapeProductOverflow { shape } }`（封闭枚举字段，参见 26-error.md v3.0.0 §5.1）。

### 6.4 轴向归约并行方案

以下为实现指导，描述 `reduction` 模块如何利用 `parallel` 提供的原语组合轴向归约的并行路径；`parallel` 本身不暴露 `par_sum_axis` 等轴向专用 API。

- 轴向 `sum_axis(axis)` / `sum_axis_keepdims(axis)` 的并行路径沿未被归约的轴切分为彼此独立的 chunk。
- 每个 chunk 在目标轴上执行串行归约，随后按输出逻辑位置写入局部结果；最终结果按 chunk 索引顺序合并。
- `keepdims` 行为在并行路径下保持不变，仅影响输出 shape，不改变分块策略。
- 空轴归约返回加法单位元。

### 6.5 自动路径派发与所有权

路径选择与执行策略裁决见 §6.1；`parallel` 只接收已经完成路径选择、输入校验与语义前置条件检查的调用。

- `par_map`、`par_zip_map`、`par_sum` 接收的输入都已由上层语义模块和 `dispatch.rs` 验证完毕；`par_dot` 在进入并行归约前仍需自行做运行时校验，要求 `lhs.ndim() == 1`、`rhs.ndim() == 1` 且两侧逻辑长度一致。
- 对归约和内积，若调用方选择并行路径，则 `parallel/` 必须提供固定 chunking 与固定 merge tree，保证同平台、同配置、同路径下结果确定；整数 `sum` / `dot` 的失败诊断还必须按逻辑 chunk 索引顺序仲裁，始终选择首个失败 chunk。
- 并行归约采用固定分块策略：chunk 大小由 `compute_safe_chunks(n, num_workers)` 确定（定义于 `src/parallel/mod.rs`，见 01-architecture.md §5.2a），worker 按固定索引范围分配，merge 按 worker 索引顺序合并。
- 若执行对象为整数 `sum` / `dot`，每个 worker 必须在本分片内执行 `checked_add` / `checked_mul` + `checked_add`；任一 worker 发现溢出时必须传播 panic，不得转写为 `XenonError`。失败诊断固定按逻辑 chunk 索引顺序仲裁。
- **回退归属**：若某实现选择不能保证 “首个失败 chunk 仲裁” 这一不变量，则该实现版本的 `dispatch.rs` 必须在 `select_exec_path()` 阶段就拒绝把整数 `sum` / `dot` 路由到 `Parallel`（例如：将整数归约的并行阈值置为 `usize::MAX`）。这条职责落在 `dispatch.rs` 而**不是** `parallel`，与决策 4（“parallel 不包含串行回退”）保持一致。`parallel` 一旦被调用，就永远不会自行切换到串行路径。

### 6.6 Checked 映射与错误传播

```rust,ignore
pub(crate) fn par_map_checked<A, B, S, D, F>(
    tensor: &TensorBase<S, D>,
    strategy: &ParallelExecStrategy,
    _guard: ParallelGuard,
    f: F,
) -> Result<Tensor<B, D>, XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> Result<B, XenonError> + Sync + Send,
{
    // ParElements implements IndexedParallelIterator (see §5.6), so .collect()
    // and .collect_into_vec() preserve F-order index→position mapping
    // regardless of worker completion order.
    //
    // Two-phase pattern preserves both ordering and short-circuit-on-error:
    //   1) reduce-with: detect first error (in chunk-index order) and bail.
    //   2) on success, indexed collect into a pre-sized Vec<B> guarantees
    //      output[i] corresponds to logical input element i.
    let iter = ParElements::with_strategy(tensor.view(), strategy);
    let total = iter.len(); // IndexedParallelIterator → ExactSize semantics
    let mut out: Vec<B> = Vec::with_capacity(total);
    // Sketch: collect_into_vec writes by index; if any element returns Err,
    // we instead surface the first Err in chunk-index order via a separate
    // try_reduce_with pass that does NOT rely on completion order.
    let result: Result<(), XenonError> = iter
        .clone() // ParElements is cheap to clone (metadata only)
        .try_for_each(|item| { let _ = f(item)?; Ok(()) });
    result?;
    iter.map(|item| f(item).expect("error already surfaced by try_for_each pass"))
        .collect_into_vec(&mut out);

    // Safety: out.len() == total == checked_size(tensor.raw_dim()) (ExactSize),
    // and `out` is laid out in F-order index sequence by collect_into_vec on
    // an IndexedParallelIterator. Element ordering matches the serial baseline.
    Ok(unsafe { Tensor::from_raw_vec_unchecked(out, tensor.raw_dim()) })
}
```

**顺序保证（修复 Blocker B8）**：`ParElements` 实现 `IndexedParallelIterator`（§5.6 producer 拆分契约），其 `collect()` / `collect_into_vec()` 按 producer 索引位置（即 F-order 逻辑顺序）写入输出 buffer，而不是按 worker 完成顺序追加。因此：

- 即便 worker 之间执行顺序不确定，`out[i]` 必然对应 `tensor` 的第 `i` 个 F-order 逻辑元素，`Tensor::from_raw_vec_unchecked(out, raw_dim)` 的"长度一致 + 顺序一致"前提两条都满足。
- 错误优先性：若闭包返回 `Err`，单次 `IndexedParallelIterator::collect()` 在 rayon 当前实现中不保证返回"最早索引"的 `Err`。本设计采用两遍模式：第一遍 `try_for_each` 用作错误探测，遇到 `Err` 立即停止；只有所有 chunk 都成功时，第二遍 `collect_into_vec` 按索引位置写入最终结果。第一遍内部的 `Err` 选择仍由 rayon 决定，但调用方对外语义只承诺"至少传播一个 `Err`"（与 §6.7、§10 一致），不要求"第一个发生的 `Err`"。整数 `sum`/`dot` 的"首个失败 chunk 仲裁"是另一种更强约束，只在 §6.5 中适用，不向 `par_map_checked` 推广。
- panic 中途 drop：`out` 通过 `Vec::with_capacity(total)` 预分配但**未先初始化**；`collect_into_vec` 内部按 `Vec::resize` + indexed writeback 完成填充。若闭包 panic，`Vec` 的 drop 会安全释放已写入元素的存储。`B: Element` 即 `Copy + Sized`，无 drop side-effects，进一步简化论证。

`par_map_checked()` 不再自行决定是否并行（路径选择见 §6.1）；`strategy` 参数控制并行分块策略，与其他并行入口保持一致。

### 6.7 安全性论证

| 主题                             | 论证                                                                                                                                                                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tensor::from_raw_vec_unchecked`（长度） | 这里只在输出向量长度与 `tensor.raw_dim()` 已由输入张量长度和映射过程保持一致时使用；并行与串行路径都必须保证产出元素数等于输入逻辑元素数。`ParElements: IndexedParallelIterator` 提供 `len()` 与 `Producer::split_at` 的不重叠覆盖不变量（§5.6），从而 `collect_into_vec` 后 `out.len() == checked_size(raw_dim)`。 |
| `Tensor::from_raw_vec_unchecked`（顺序） | F-order 顺序由 producer 拆分契约 + `IndexedParallelIterator::collect_into_vec` 的"按索引写入"语义共同保证（修复 Blocker B8）。worker 完成顺序乱序不影响 `out[i]` 对应的逻辑索引；这与串行 `map` 的 F-order 收集严格等价。 |
| `par_zip_map` broadcast chunking | 每个并行 chunk 仅借用两个输入的只读 broadcast-compatible sub-view；广播轴保持逻辑重复语义，不进行额外物理展开，因此不会引入越界写或悬垂引用。Producer split 不切分广播轴的物理切片，仅切分逻辑输出区间。 |
| `ParallelGuard` 转移              | `_guard: ParallelGuard` 由 dispatch 在 `select_exec_path` 内构造并按值移交；`parallel` 函数体在并行执行结束后自然 drop guard，由 RAII 保证 thread-local 嵌套防护标记一定被清除（与 30-dispatch.md 决策 7 一致）。 |
| panic / `Err` 传播               | 并行操作中发生 panic 或返回 `Err(XenonError)` 时，错误不会被静默忽略；语义上最终结果须至少传播一个错误。一般错误不保证传播"第一个"发生的错误（整型 `sum` / `dot` 除外：整型运算的失败诊断固定按逻辑 chunk 索引顺序仲裁，始终选择首个失败 chunk，参见 §5.5、§6.5）。实现上 Rayon 的并行 collect/reduce 可能不会物理中断其他 worker，但错误信息会被收集并在最终结果中报告。 |
| Send/Sync/借用边界               | 并行执行只借用输入张量的只读视图；闭包与元素类型必须满足 `Send` / `Sync` 约束；输出分配与写入归当前 worker 独占，不能向其他 worker 暴露共享可写借用。 |
| Worker 内 SIMD 安全性              | worker 在自己 chunk 的连续内存切片（或逻辑区间）上独立调用 `simd` 后端 kernel，不跨 worker 访问；SIMD admission 由 `08-simd.md` §5.4 在 chunk 内部独立判断，不与跨 worker 的 chunking/合并语义产生交叉依赖（v2.0 起决策 5）。 |

---

## 7. 实现任务拆分

### Wave 1: 基础状态与路径裁决

阈值状态、路径选择与嵌套并行防护已迁移至 `dispatch.rs` 模块，参见 `01-architecture.md`。

### Wave 2: 并行入口与执行内核

- [ ] **T1**: 实现 `ParElements` 与 `TensorBase::par_iter()`
  - 文件: `src/parallel/iter.rs`
  - 内容: 单输入元素级并行遍历入口
  - 测试: `test_par_iter_len_matches_tensor_len`
  - 前置: `dispatch.rs` 执行路径裁决已可用，`10-iterator.md` 中只读迭代语义已确定
  - 预计: 10 min

- [ ] **T2**: 实现 `par_map`
  - 文件: `src/parallel/map.rs`
  - 内容: 纯并行逐元素映射入口，执行策略参数由 `dispatch.rs` 统一传入
  - 测试: `test_par_map_parallel_path`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 实现 `par_zip_map`
  - 文件: `src/parallel/map.rs`
  - 内容: 二元广播逐元素纯并行入口，供 `math` 模块消费
  - 测试: `test_par_zip_map_matches_serial_add`, `test_par_zip_map_broadcast_rhs_scalar`
  - 前置: T1, `math` 广播语义已确定
  - 预计: 10 min

- [ ] **T4**: 实现 `par_reduce_impl` 与 `par_sum`
  - 文件: `src/parallel/reduce.rs`
  - 内容: 并行归约、identity 合并、语义对齐调用方选定的串行基线
  - 测试: `test_par_sum_matches_serial`, `test_par_sum_empty_matches_identity`
  - 前置: T1, `13-reduction.md` 归约语义已确定
  - 预计: 10 min

- [ ] **T5**: 实现 `par_dot`
  - 文件: `src/parallel/reduce.rs`
  - 内容: 运行时 `ndim() == 1` / 长度一致性检查、并行内积、错误返回与空数组单位元语义
  - 测试: `test_par_dot_matches_serial`, `test_par_dot_shape_mismatch`, `test_par_dot_empty_identity`
  - 前置: T4
  - 预计: 10 min

### Wave 3: 线程池与异常传播

- [ ] **T6**: 实现 `ParallelPool`
  - 文件: `src/parallel/mod.rs`
  - 内容: 自定义 `rayon::ThreadPool` 包装，不改变公开 API 结果语义
  - 测试: `test_parallel_pool_preserves_semantics`
  - 前置: T2, T4, T5
  - 预计: 10 min

- [ ] **T7**: 完成错误与 panic 传播收口
  - 文件: `src/parallel/checked.rs`
  - 内容: `XenonError` 透传、panic 不吞掉
  - 测试: `test_parallel_error_propagation`, `test_parallel_panic_propagation`
  - 前置: T2, T4, T5
  - 预计: 10 min

### Wave 4: 配置与回归验证

- [ ] **T8**: 补齐 feature gate 与配置矩阵测试
  - 文件: `src/parallel/` (全部子文件), `tests/test_parallel.rs`
  - 内容: 默认关闭、`--features parallel` 构建、单线程/多线程分支验证
  - 测试: `cargo test`, `cargo test --features parallel`
  - 前置: T1-T7
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置                                           | 目的                                                             |
| ----------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| 单元测试                | `src/parallel/` 各子文件内联测试模块           | 验证并行入口、归约与错误传播                                     |
| 集成测试                | `tests/test_parallel.rs`                       | 验证跨模块语义与 feature gate 行为                               |
| 边界测试                | 与并行测试配套组织                             | 覆盖空张量、单元素、非连续视图、单线程环境                       |
| 属性测试（按需）        | 当前版本不强制                                 | 当前模块以确定性路径与语义对齐为主，暂无必须引入的随机不变量测试 |
| Feature gate / 配置测试 | `cargo test`, `cargo test --features parallel` | 验证默认关闭与启用并行后语义不变                                 |
| 类型边界 / 编译期测试   | trait 约束测试或编译期失败测试                 | 验证 `bool` 不参与 `par_sum` / `par_dot` 等非法组合              |

### 8.2 单元测试清单

| 测试函数                                | 测试内容                                   | 优先级 |
| --------------------------------------- | ------------------------------------------ | ------ |
| `test_par_map_parallel_path`            | `dispatch.rs` 选中并行路径后结果与串行一致 | 高     |
| `test_par_zip_map_matches_serial_add`   | 二元逐元素并行加法结果与串行一致           | 高     |
| `test_par_zip_map_broadcast_rhs_scalar` | 右侧标量广播时并行路径与串行一致           | 高     |
| `test_par_sum_matches_serial`           | 并行 `sum` 与串行语义一致                  | 高     |
| `test_par_dot_matches_serial`           | `par_dot` 与串行结果一致                   | 高     |
| `test_par_map_checked_matches_serial` | `par_map_checked` 在闭包返回 `Ok` 时结果与串行一致 | 高|
| `test_parallel_error_propagation`       | 并行 `Err` 及时上传                        | 高     |
| `test_parallel_panic_propagation`       | 并行 panic 不被吞掉                        | 高     |

### 8.3 边界测试场景

| 场景                 | 预期行为                                                                            |
| -------------------- | ----------------------------------------------------------------------------------- |
| 空数组 `len == 0`    | `par_sum()` 返回加法单位元；`par_dot()` 在两个长度为 `0` 的一维输入上返回加法单位元 |
| 单元素张量           | 若 `dispatch.rs` 仍选择并行路径，结果与串行一致                                     |
| 非连续视图           | 若 `dispatch.rs` 选择并行路径，结果仍与串行一致                                     |
| 单线程环境           | 不由 `parallel/` 自行处理；调用方不应选择并行路径                                   |
| 非一维输入           | `par_dot()` 在任一输入 `ndim() != 1` 时返回错误                                     |
| 长度不匹配的一维输入 | `par_dot()` 返回 `XenonError::ShapeMismatch { operation, left_shape, right_shape }`    |
| 二元广播逐元素输入   | `par_zip_map()` 在广播兼容时返回与串行 `add/sub/mul/div` 一致的结果                 |

### 8.4 属性测试不变量

| 不变量                                                             | 测试方法                                 |
| ------------------------------------------------------------------ | ---------------------------------------- |
| `par_map` 与串行 `map` 在相同输入上产出相同形状与逐元素值          | 对整数类型可按多组形状和布局做表驱动校验 |
| `par_zip_map` 与串行广播二元运算在相同输入上产出相同形状与逐元素值 | 对 `add/sub/mul/div` 做表驱动校验        |
| `par_sum` / `par_dot` 在相同执行路径和配置下结果确定               | 对相同输入重复运行并比较结果             |

### 8.5 集成测试

| 测试文件                       | 测试内容                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `tests/test_parallel.rs`       | 并行 dispatch 与 `math`、`reduction`、`dot` 等语义模块组合路径的端到端验证   |

### 8.6 Feature gate / 配置测试

| 配置                           | 验证点                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| 默认配置                       | 可选并行默认关闭，默认构建可编译                           |
| 启用 `parallel`                | `par_map` / `par_sum` / `par_dot` 可用，结果与串行路径一致 |
| 启用 `parallel` + broadcast op | `math` 通过 `par_zip_map` 走并行路径时结果与串行广播一致   |
| 单线程运行                     | `dispatch.rs` 不应选择 `parallel/` 路径                    |
| 启用并行 + 嵌套调用            | 嵌套并行防护由 `dispatch.rs` 负责                          |

### 8.7 类型边界与编译期测试

| 场景                                | 测试方式                            |
| ----------------------------------- | ----------------------------------- |
| `bool` 不参与 `par_sum` / `par_dot` | 编译期 trait 约束测试               |
| 非法 feature 组合                   | 配置矩阵测试                        |
| 非法阈值参数                        | 转由 `dispatch.rs` 的运行时测试覆盖 |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向           | 对方模块                     | 接口/类型                                         | 约定                                                                |
| -------------- | ---------------------------- | ------------------------------------------------- | ------------------------------------------------------------------- |
| 消费（输入）   | `tensor`                     | `Tensor<A, D>`, `TensorBase<S, D>`                | 调用前已满足 shape、layout、类型约束                                |
| 消费（输入）   | `element`                    | `Element`, `Numeric`                              | 函数签名中 trait 约束所需                                           |
| 消费（输入）   | `error`                      | `XenonError`                                      | 可恢复错误统一复用项目错误模型                                      |
| 模块内部       | `parallel`                   | `TensorBase::par_iter()`, `ParElements<'a, A, D>` | 定义于本模块（参见 §5.6），`pub(crate)` 内部入口，提供单输入只读并行遍历 |
| 被调用（输出） | 上层语义模块 / `dispatch.rs` | `par_map` / `par_sum` / `par_dot` / `par_zip_map` | 仅在 `dispatch.rs` 已选中并行路径后执行                             |
| 产出（输出）   | 上层语义模块                 | `Tensor<B, D>` 或 `Result<A, XenonError>`         | 并行与串行路径保持相同外部语义                                      |

### 9.2 数据流

```text
math / reduction / matrix call dispatch entry
    │
    ├── query metadata (.len(), .is_f_contiguous(), alignment_ok)
    ├── let (path, guard) = dispatch::select_exec_path(...)
    │       ├── (Serial, None)        → serial path stays outside parallel/
    │       ├── (Simd,   None)        → simd path stays outside parallel/
    │       └── (Parallel, Some(g))   → parallel path; g is forwarded by value
    ├── parallel path
    │      │
    │      ├── par_iter() / par_zip_map(.., guard) / par_sum(.., guard) / par_dot(.., guard)
    │      └── inside each worker chunk, SIMD admission may apply per chunk
    │              (08-simd.md v2.0.0 决策 5)
    └── return Tensor or Result with unchanged public semantics; guard auto-drops
```

- `select_exec_path()` 返回类型为 `(ExecPath, Option<ParallelGuard>)`；`Option` 仅在 `ExecPath::Parallel` 分支返回 `Some(_)`，`Serial` / `Simd` 分支返回 `None`（与 30-dispatch.md v1.1.0 决策 7 完全一致）。
- 调用方负责把 `Some(guard)` 按值移交到 `parallel` 后端入口；guard 在并行函数返回时被 drop，自动清除 thread-local 嵌套防护标记。

---

## 10. 错误处理与语义边界

| 主题              | 说明                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------- |
| Recoverable error | `par_dot()` 的长度不兼容返回 `XenonError::ShapeMismatch { operation: "par_dot", left_shape, right_shape }`；`par_dot()` 的非一维输入返回 `XenonError::InvalidArgument { operation: "par_dot", kind: InvalidArgumentKind::OperationSpecific { argument: "ndim", constraint: "rank == 1" } }`；`par_zip_map()` 的元素总数溢出返回 `InvalidShape { operation: "par_zip_map", shape, kind: InvalidShapeKind::ProductOverflow, offending_dim: None }`。所有字段对齐 26-error.md v3.0.0 §5.1 的封闭枚举字段。 |
| Panic             | 归约中的整数溢出仍属于不可恢复错误，必须 panic，而不是包装为 `XenonError`                                       |
| 路径一致性        | 一旦进入 `parallel/`，并行路径必须返回与调用方串行基线相同形状、相同错误类别，以及满足同一数值语义约束的结果（路径选择见 §6.1）  |
| 容差边界          | 浮点与复数若存在执行路径相关的已知舍入差异，只能落在 `需求说明书 §28.3` 与 `需求说明书 §28.5` 允许且已文档化的范围内；以 `需求说明书 §28.3` 为权威基线，`00-coding.md §8` 仅作为实现参考。|

路径语义边界：

- 并行模块本身不新增专属错误枚举；公开错误必须复用 `26-error.md` 中的统一模型。
- 自定义线程池类参数若存在非法值，由 `dispatch::ParallelExecStrategy::new()` 在构造期统一返回 `InvalidArgument`；`parallel` 模块在收到合法策略实例后不再重复返回该错误（参见 §5.4 与 30-dispatch.md §5.3、决策 8）。
- 当前 `par_zip_map()` 不承担广播兼容性校验，也不新增广播专属错误构造。
- panic 与 `Err(XenonError)` 都不得被吞掉；并行执行中发生的错误须至少传播一个。仅对整数 `sum` / `dot`，失败诊断必须额外满足“按逻辑 chunk 索引顺序固定选择首个失败 chunk”；做不到则回退串行路径。
- 路径裁决语义见 §6.1 与决策 4、决策 6。

### 10.1 浮点/复数并行归约容差

- 浮点与复数并行归约允许与标量路径不同的合并顺序；该差异视为合法实现细节，但必须受 `需求说明书 §28.3` 文档化容差约束。
- 以 `需求说明书 §28.3` 为权威基线，`00-coding.md §8` 仅作为实现参考。
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

### 决策 1：并行阈值采用内部编译期常量

| 属性     | 值                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------- |
| 决策     | 并行阈值由 `dispatch.rs` 持有编译期默认值，并允许内部配置覆写                                      |
| 理由     | `需求说明书 §9.2` 要求“须支持并行阈值配置”；保留编译期默认值可维持稳定基线，而配置覆写满足可配置性 |
| 替代方案 | 每次调用显式传参 —— 放弃，会让公开 API 过于分裂                                                    |
| 替代方案 | 仅保留不可配置的固定常量 —— 放弃，违反 `需求说明书 §9.2`                                           |

### 决策 2：嵌套并行进入失败时必须回退串行

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | `ParallelGuard` 迁移至 `dispatch.rs`；进入失败不报错、不 panic，而是选择串行回退 |
| 理由     | `需求说明书 §9.2` 明确禁止库内部二次并行；该场景是执行策略问题而非用户输入错误   |
| 替代方案 | 允许库内部继续二次并行 —— 放弃，违反需求                                         |
| 替代方案 | 将嵌套并行视为 recoverable error —— 放弃，会污染公开 API 语义                    |

`ParallelPool` 内部调用同样必须经过 `dispatch.rs` 中的 `ParallelGuard`。若用户在自定义 pool 中再次调用内部并行后端，dispatch helper 会把 `ParallelContext` token 捕获到 Rayon worker 闭包中，并在二次派发时自动回退串行，与全局线程池行为一致；同时不允许嵌套 `ParallelPool` 实例，以避免引入额外调度语义。

### 决策 3：并行模块不新增专属公开错误类型

| 属性     | 值                                                      |
| -------- | ------------------------------------------------------- |
| 决策     | 统一使用 `XenonError` 表达 shape 与参数错误             |
| 理由     | 保持跨模块诊断字段与错误类别一致，满足 `需求说明书 §27` |
| 替代方案 | 定义 `ParallelError` —— 放弃，会破坏统一错误模型        |
| 替代方案 | 以 panic 处理非法阈值 —— 放弃，不符合可恢复错误要求     |

### 决策 4：并行模块不包含串行回退

| 属性     | 值                                                                                     |
| -------- | -------------------------------------------------------------------------------------- |
| 决策     | `parallel` 只提供纯并行执行入口，不包含串行回退路径                                    |
| 理由     | 执行路径裁决（串行 vs 并行）由 `dispatch.rs` 统一承担，`parallel` 不需要自行判断       |
| 替代方案 | 在 `parallel` 内保留串行回退 —— 放弃，会导致 `dispatch` 判断与 `parallel` 内部判断重复 |

### 决策 5：二元逐元素并行能力以 `par_zip_map()` 形式提供给 `math`

| 属性     | 值                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------- |
| 决策     | `parallel` 提供 `pub(crate)` 级 `par_zip_map()`，由 `math` 在广播裁决完成后调用                   |
| 理由     | 满足 `需求说明书 §9.2` 对逐元素二元运算并行路径的要求，同时不把通用多输入并行迭代器暴露为公开 API |
| 替代方案 | 仅保留 `par_map` —— 放弃，无法覆盖 `add/sub/mul/div` 广播逐元素并行需求                           |
| 替代方案 | 将二元广播并行逻辑直接写进 `math` —— 放弃，会复制 `dispatch.rs` 之外的并行执行实现与错误传播策略  |

### 决策 6：执行路径裁决由 `dispatch.rs` 统一收口

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | `math` / `reduction` / `matrix` 调用 `dispatch::select_exec_path()` 决定执行路径 |
| 理由     | 统一串并阈值与并行前置条件判断，避免多个模块各自实现分支树                       |
| 替代方案 | 每个模块自行判断 serial/parallel —— 放弃，易产生阈值漂移和行为不一致             |

### 决策 7：并行入口接收 `_guard: ParallelGuard` 按值参数

| 属性     | 值                                                                                                                                              |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `par_map` / `par_zip_map` / `par_sum` / `par_dot` / `par_reduce_impl` / `par_map_checked` 都接收 `_guard: ParallelGuard` 按值参数                |
| 理由     | 把"裁决到 Parallel"和"进入并行临界区"在调用图上原子绑定；调用方无法忘记 acquire guard，guard RAII 在函数返回时自动清除 thread-local 嵌套防护标记 |
| 替代方案 | 在 `parallel` 内部自行 `enter()` —— 放弃，会让 `select_exec_path` 与实际并行进入分离，重新引入 30-dispatch v1.0 的 C6 矛盾（哪个函数 consume guard） |
| 替代方案 | 不传 guard，依赖 thread_local 隐式状态 —— 放弃，无法让类型系统强制"只有被 dispatch 选中的调用才能进入并行" |

### 决策 8：`ParElements` 实现 `IndexedParallelIterator` + `Producer`

| 属性     | 值                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `ParElements` 同时实现 `rayon::iter::IndexedParallelIterator` 和 `rayon::iter::plumbing::Producer`，提供 `split_at`-based 不重叠拆分 |
| 理由     | 为 `par_map_checked` 顺序保证、`par_zip_map` 索引收集、`par_sum`/`par_dot` 固定 chunking 提供唯一可证明的 producer 不变量基础（修复 Blocker B7） |
| 替代方案 | 仅实现 `ParallelIterator` —— 放弃，rayon 无法保证 `collect` 的索引顺序；`from_raw_vec_unchecked` 元素错位将导致内存安全前提缺失 |

### 决策 9：worker 内允许 SIMD（决策 4 + 决策 5 协同）

| 属性     | 值                                                                                                                            |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 单个 worker chunk 内可独立调用 `simd` 后端 kernel；chunk 间合并仍由 `parallel` 控制                                            |
| 理由     | 与 08-simd.md v2.0.0 决策 5 对齐：撤销并行/SIMD 互斥，提供 thread × SIMD 双层加速                                              |
| 替代方案 | 保留 v1.x 设计（worker 内禁止 SIMD）—— 放弃，会牺牲大数据吞吐                                                                  |
| 替代方案 | worker 跨 chunk 共享 SIMD 状态 —— 放弃，会让 chunk 不变量与 SIMD admission 互相耦合                                            |

---

## 12. 性能描述

### 12.1 复杂度标注

- `par_map()`：时间 `O(n)`，额外结果空间 `O(n)`。
- `par_sum()`：时间 `O(n)`，额外工作空间取决于 `rayon` 分块；逻辑额外空间 `O(1)`。
- `par_dot()`：时间 `O(n)`，逻辑额外空间 `O(1)`。

### 12.2 缓存与执行特征

| 场景         | 实现路径              | 缓存友好性 | 说明                                  |
| ------------ | --------------------- | ---------- | ------------------------------------- |
| 连续 F-order | 顺序分块并行          | 高         | 适合作为并行路径的主要收益来源        |
| 非连续视图   | 步长访问并行          | 中到低     | 缓存命中率下降，但仍保持纯并行语义    |
| 小张量       | 由 `dispatch.rs` 裁决 | 高         | 是否进入 `parallel/` 不在本模块职责内 |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------------- |
| `std` only | 本模块依赖 `rayon`，且项目基线仅支持 `std`；不讨论 `no_std`                                                    |
| MSRV       | Rust 1.85+                                                                                                     |
| 单 crate   | 设计保持在 Xenon 单 crate 内，不引入额外 crate 拆分                                                            |
| SemVer     | 并行后端入口属于 `pub(crate)` 内部契约；其语义仍需在 crate 内保持稳定，但不构成面向最终用户的独立公开 API 承诺 |
| 最小依赖   | 仅使用允许的可选依赖 `rayon`，默认关闭                                                                         |

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
| 1.4.0 | 2026-04-28 |
| 2.0.0 | 2026-05-02 |

### v2.0.0 (2026-05-02) — SemVer breaking changes

> 本版本是与 30-dispatch v1.1.0、08-simd v2.0.0 协同的破坏性更新；所有 `parallel` 后端入口均为 `pub(crate)` 内部 API，故对外 SemVer 影响实际为零，但内部契约破坏列出如下：

- §5.4 / §5.5：`ParallelExecStrategy` 字段从 `pub` 改为 `pub(crate)`，构造方式收敛到 `ParallelExecStrategy::new()`；`parallel` 模块不再返回 `InvalidArgument` 处置非法策略字段（已由 dispatch 端在构造期拒绝）。
- §5.5：`par_map` / `par_zip_map` / `par_sum` / `par_dot` / `par_reduce_impl` / `par_map_checked` 全部新增 `_guard: ParallelGuard` 按值参数（决策 7）。
- §5.5：`par_reduce_impl` 闭包 bound 从 `F: Fn(A, A) -> A + Sync` 加强为 `F: Fn(A, A) -> A + Send + Sync`，`ID` 同样从 `Fn() -> A + Sync + Clone` 加强为 `Fn() -> A + Send + Sync + Clone`，与其他并行入口保持一致。
- §5.6：`ParElements` 实现 `IndexedParallelIterator` + `Producer`（决策 8），修复 v1.x 缺失的 producer 拆分语义（Blocker B7）。
- §6.1 / §6.2 / §6.3 / §9.2：worker 内允许调用 SIMD 后端 kernel（决策 9，与 08-simd v2.0.0 决策 5 对齐）。
- §6.3：`par_zip_map` 的元素总数溢出错误对齐 26-error v3.0.0 的 `InvalidShape { kind: InvalidShapeKind::ProductOverflow, .. }` 封闭枚举（v2.0.0-rc 误用 `InvalidArgumentKind::ShapeProductOverflow`，已修正）；`num_threads` 来源统一为 `strategy.max_workers.unwrap_or_else(rayon::current_num_threads)`（修复 §5.6 与 §6.3 的不一致）。
- §6.5：整数 `sum` / `dot` 的"首个失败 chunk 仲裁"前提不成立时，**回退责任由 dispatch 承担**（即 `select_exec_path()` 不选择 Parallel）；`parallel` 模块本身永远不串行回退（保持决策 4）。
- §6.6：`par_map_checked` 改用两遍模式（`try_for_each` 错误探测 + `collect_into_vec` 索引收集），从 producer 不变量证明 F-order 顺序与 `from_raw_vec_unchecked` 安全前提（修复 Blocker B8）。
- §10：错误返回字段全部对齐 26-error v3.0.0 的封闭枚举（`InvalidArgumentKind::*`、`ShapeMismatch.operation`）。
- §11：新增决策 7 / 8 / 9。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
