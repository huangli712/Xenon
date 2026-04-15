# 并行后端模块设计

> 文档编号: 09 | 模块: `src/parallel/` | 阶段: Phase 5
> 前置文档: `07-tensor.md`, `10-iterator.md`, `13-reduction.md`, `26-error.md`
> 需求参考: 需求说明书 §9.2, §9.3, §27, §28.3, §28.5
> 范围声明: 范围内

---

## 1. 模块定位/概述

并行后端模块是 Xenon 的可选执行后端，通过 `rayon` 为逐元素映射、二元逐元素运算与归约提供数据并行能力。该模块默认关闭，仅在启用 `parallel` feature 时参与构建。当前版本覆盖内部 `par_map`、供 `math` 模块消费的 `par_zip_map`、并行 `sum` / `dot`、阈值控制、自动串并路径选择与嵌套并行防护；若无法证明并行路径与串行路径的语义一致，必须回退串行。

### 1.1 职责边界表

| 职责           | 包含                                                          | 不包含                           |
| -------------- | ------------------------------------------------------------- | -------------------------------- |
| 并行逐元素执行 | `par_map`、`par_map_with_threshold`、`par_zip_map`、基于视图的并行遍历 | 通用多输入同步并行公开迭代器 API |
| 并行归约       | `par_sum`、`par_dot`、内部 `par_reduce_impl`                  | 矩阵乘法、矩阵分解、GPU 后端     |
| 路径选择       | 阈值控制、连续性判断、单线程环境自动回退                      | 自动任务图调度、跨模块全局调度器 |
| 嵌套并行治理   | `ParallelGuard` 防止库内部二次并行                            | 用户侧任意线程模型抽象           |
| 线程池封装     | 内部 `ParallelPool` 改变执行上下文而不改变 API 语义           | 新的公开调度语义或额外第三方依赖 |

### 1.2 设计原则

| 原则           | 体现                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 语义一致性     | 并行路径不得改变公开 API 的形状、错误类别和数值语义；无法证明一致性时回退串行 |
| 自动回退       | 小张量、非连续视图惩罚、单线程环境或嵌套并行场景都自动回退串行                |
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
L5: iter, internal kernel
L6: parallel  <- current module (optional, feature = "parallel")
```

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §9.2`、`§9.3`、`§27`、`§28.3`、`§28.5` |
| 范围内   | 可选数据并行、并行阈值配置、逐元素运算（含二元广播）/归约/内积的并行执行路径、自动串并选择、禁止库内二次并行 |
| 范围外   | GPU 后端、自动任务图调度、通用多数组 lock-step 并行公开接口、额外第三方依赖                             |
| 非目标   | 不把文档改成 `no_std`，不增加除 `rayon` 之外的外部依赖，不扩展当前并行能力集合                 |

---

## 3. 文件位置

```text
src/parallel/
└── mod.rs    # feature gates, threshold state, internal ParallelGuard/ParallelPool, par_map/par_zip_map/par_sum/par_dot
```

单文件设计：当前版本公开能力集中于阈值配置、路径裁决和少量并行入口，单文件足以容纳核心状态与执行边界；若后续范围发生变化，再评估是否拆分内部实现文件，但不影响本设计的公开能力边界。

---

## 4. 依赖关系

### 4.1 依赖图

```text
src/parallel/
├── rayon (optional)         # ThreadPool, ParallelIterator, current_num_threads
├── crate::tensor            # Tensor, TensorBase, TensorView
├── crate::dimension         # Ix1
├── (module-owned)           # ParElements and par_iter() entry belong to parallel/
├── crate::kernel            # private internal dispatch and serial kernels registered in 01-architecture.md
└── crate::error             # XenonError
```

### 4.2 类型级依赖表

| 来源模块    | 使用的类型/trait                                                                                                |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| `rayon`     | `rayon::ThreadPool`, `rayon::current_num_threads`, `rayon::iter::ParallelIterator`                              |
| `tensor`    | `Tensor<A, D>`, `TensorBase<S, D>`, `TensorView<'a, A, D>`, `.len()`, `.raw_dim()`, `.is_f_contiguous()` |
| `dimension` | `Ix1` |
| `parallel`  | `ParElements<'a, A, D>`, `TensorBase::par_iter()`, `par_zip_map()`                                             |
| `kernel`    | 私有串行 `sum` / `dot` kernel、内部 dispatch helper、identity / combine 约束                                   |
| `error`     | `XenonError`, `XenonError::ShapeMismatch`, `XenonError::InvalidArgument`                                        |

### 4.3 依赖方向

> **依赖方向：单向向上。** `parallel/` 仅消费 `tensor`、私有 `kernel`、`error` 和可选 `rayon`；`ParElements` 与 `TensorBase::par_iter()` 归属 `parallel` 模块本身，不属于 `iter` 模块。并行路径只建立在上层已完成的张量形状、布局与类型约束之上；广播形状裁决由 `math` 调用侧先完成，再以 `output_dim` 形式传入。

> **设计决策：** 串行归约/内积的共享 kernel 应下沉到内部私有模块（如 `kernel`，已在 `01-architecture.md` 注册为 private internal module），避免与公开 `reduction`/`dot` 模块形成循环依赖。`parallel` 只依赖该私有 `kernel` 能力，不直接依赖公开语义模块。

### 4.4 合法性声明

| 项目           | 说明                                                                            |
| -------------- | ------------------------------------------------------------------------------- |
| 新增第三方依赖 | `rayon`（可选）                                                                 |
| 合法性结论     | 合法；符合 `需求说明书 §1.2` 对最小依赖的限制，以及 `§9.2` 对可选并行能力的要求 |
| 替代方案       | 仅用 `std::thread` 不能无损提供当前所需的并行迭代与线程池抽象，因此不采用       |

### 4.5 与迭代器模块的边界

`parallel/` 不定义通用多输入同步并行公开迭代接口。`TensorBase::par_iter()` 只提供单输入元素级并行入口；二元逐元素并行能力以 `pub(crate)` 级 `par_zip_map()` 形式提供，仅供 `math` 模块在完成广播裁决后消费。该边界与 `10-iterator.md §1.2` 中“并行迭代不属于 `iter` 模块公开职责”保持一致。

---

## 5. 公共 API 设计

### 5.1 Feature gate 与运行时状态

```toml
[features]
default = ["std"]
std = []
parallel = ["dep:rayon", "std"]

[dependencies]
rayon = { version = "1.10", optional = true }
```

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_PARALLEL_THRESHOLD: usize = 1024;

static GLOBAL_PARALLEL_THRESHOLD: AtomicUsize =
    AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

pub(crate) struct ParallelGuard;

#[cfg(feature = "parallel")]
pub(crate) struct ParallelPool {
    inner: rayon::ThreadPool,
}
```

- `GLOBAL_PARALLEL_THRESHOLD`：控制自动串并切换阈值。
- `ParallelGuard`：防止库内部嵌套并行；实现上不再只依赖调用线程的 thread-local `Cell<bool>`，而是由私有 dispatch helper 创建一个可捕获的 `ParallelContext` token，并在进入并行区域前同时设置调用线程的本地标志。worker 闭包通过捕获该 token 感知“已处于并行区域”，析构时统一清理；进入失败时必须回退串行而不是 panic，panic 时也通过 RAII 自动清理。
- `ParallelPool`：内部线程池包装，只改变执行上下文，不改变外部语义；其内部调用同样受 `ParallelGuard` + `ParallelContext` 保护，自定义 pool 内嵌套调用并行入口时会自动回退串行，不允许嵌套 `ParallelPool` 实例。它属于内部机制，不构成公开 API 契约。

### 5.2 公开运行时接口与内部执行入口

> **可见性说明：** `parallel/` 对外的稳定契约只包括 feature gate、阈值读写与自动加速行为本身；`par_map`、`par_zip_map`、`par_sum`、`par_dot`、`ParallelPool` 均为内部执行后端，由 `math` / `reduction` / `dot` 等语义模块自动选择调用，不作为公开 API 契约的一部分。
>
> **内部后端声明：** 这些 API 为内部执行后端，由 `math` / `reduction` / `dot` 等语义模块自动选择调用，不作为公开 API 契约的一部分。

### 5.3 函数签名

```rust
#[cfg(feature = "parallel")]
pub fn get_parallel_threshold() -> usize;

#[cfg(feature = "parallel")]
pub fn set_parallel_threshold(threshold: usize) -> Result<(), XenonError>;

#[cfg(feature = "parallel")]
pub fn reset_parallel_threshold();

#[cfg(feature = "parallel")]
pub fn should_parallelize(len: usize, is_f_contiguous: bool) -> bool;

#[cfg(feature = "parallel")]
pub(crate) fn par_map<S, A, B, D, F>(tensor: &TensorBase<S, D>, f: F) -> Tensor<B, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_map_with_threshold<S, A, B, D, F>(
    tensor: &TensorBase<S, D>,
    f: F,
    threshold: usize,
) -> Result<Tensor<B, D>, XenonError>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_zip_map<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    f: F,
) -> Result<Tensor<C, DO>, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = B>,
    DL: Dimension,
    DR: Dimension,
    DO: Dimension,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> Result<C, XenonError> + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_reduce_impl<S, A, D, F, ID>(
    tensor: &TensorBase<S, D>,
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
pub(crate) fn par_sum<S, A, D>(tensor: &TensorBase<S, D>) -> A
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Numeric + Send + Sync + Clone;

#[cfg(feature = "parallel")]
pub(crate) fn par_dot<SL, SR, A>(
    lhs: &TensorBase<SL, Ix1>,
    rhs: &TensorBase<SR, Ix1>,
) -> Result<A, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = A>,
    A: Element + Numeric + Send + Sync + Clone;
```

> **设计决策：** `set_parallel_threshold(threshold)` 返回 `Result<(), XenonError>`，而不是无返回值函数。阈值为 `0` 或其他非法值时，必须返回 `XenonError::InvalidArgument`，以满足 `需求说明书 §27` 的可恢复错误要求。
>
> `par_map_with_threshold(..., threshold)` 与 `set_parallel_threshold()` 保持一致：当 `threshold == 0` 时，必须返回 `Err(XenonError::InvalidArgument)`，而不是静默修正、panic 或退化为默认阈值。

### 5.4 并行迭代入口

```rust
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
    D: Dimension,
    A: Element + Send + Sync,
{
    pub(crate) fn par_iter(&self) -> ParElements<'_, A, D> {
        ParElements::new(self.view())
    }
}
```

当前版本不提供任何通用并行双输入公开 API；需要二元逐元素调度时，由 `math` 模块先完成广播与输出形状裁决，再调用 `pub(crate)` 级 `par_zip_map()` 执行并行路径。

### 5.5 Good / Bad 对比示例

```rust
// Good - invalid threshold returns a recoverable error.
pub fn set_parallel_threshold(threshold: usize) -> Result<(), XenonError> {
    if threshold == 0 {
        return Err(XenonError::InvalidArgument {
            operation: "set_parallel_threshold".into(),
            argument: "threshold".into(),
            expected: "threshold > 0".into(),
            actual: threshold.to_string().into(),
            axis: None,
            shape: None,
        });
    }

    GLOBAL_PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
    Ok(())
}

// Bad - hides argument validation behind panic and breaks API consistency.
pub fn set_parallel_threshold_bad(threshold: usize) {
    assert!(threshold > 0);
    GLOBAL_PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}
```

```rust
// Good - shape mismatch stays in Result.
let dot = par_dot(&lhs, &rhs)?;

// Bad - converting recoverable shape mismatch into unwrap panic.
let dot = par_dot(&lhs, &rhs).unwrap();
```

### 5.6 文档与示例交付要求

| API                               | 文档要求                                           | 示例要求                             |
| --------------------------------- | -------------------------------------------------- | ------------------------------------ |
| `set_parallel_threshold`          | 说明默认阈值、合法值范围、错误返回                 | 给出成功与非法阈值示例               |
| `should_parallelize`              | 说明其只基于长度、连续性和线程数做判断             | 给出小张量回退示例                   |
| `par_map` / `par_sum` / `par_dot` | 明确标注为内部后端入口，只承诺与串行路径语义一致   | 由上层语义模块调用的内部示例         |
| `par_zip_map`                     | 说明其为 `math` 模块消费的内部广播并行入口         | 给出 `add/sub/mul/div` 的内部调度示例 |
| `ParallelPool`                    | 明确标注为内部执行上下文包装，不构成公开 API 契约 | 给出在线程池内执行内部并行入口的示例 |

---

## 6. 内部实现设计

### 6.1 路径选择算法

```rust
#[cfg(feature = "parallel")]
pub fn should_parallelize(len: usize, is_f_contiguous: bool) -> bool {
    if rayon::current_num_threads() <= 1 {
        return false;
    }

    let threshold = get_parallel_threshold();
    let effective_threshold = if is_f_contiguous {
        threshold
    } else {
        threshold.saturating_mul(2)
    };

    len >= effective_threshold
}
```

算法要点：

1. 单线程环境直接回退串行。
2. `should_parallelize()` 只负责阈值、连续性与线程环境判断；嵌套并行检测由外层 dispatch flow 在进入 guard 前完成。
3. 连续 F-order 张量使用全局阈值。
4. 非连续视图将有效阈值翻倍，以覆盖步长访问的缓存损失。
5. 只有长度达到有效阈值时才进入并行路径。

### 6.2 核心执行路径

```rust
fn dispatch_parallel_region<R>(
    len: usize,
    is_f_contiguous: bool,
    serial: impl FnOnce() -> R,
    parallel: impl FnOnce(&ParallelContext) -> R,
) -> R {
    if !should_parallelize(len, is_f_contiguous) {
        return serial();
    }

    let Ok(ctx) = ParallelGuard::enter() else {
        return serial();
    };

    parallel(&ctx)
}
```

```text
dispatch entry
    │
    ├── read len + contiguity metadata
    ├── should_parallelize(...)
    │       ├── false -> serial fallback
    │       └── true  -> continue
    ├── ParallelGuard::enter()
    │       ├── fail -> serial fallback
    │       └── pass -> create ParallelContext token
    ├── execute rayon parallel path with captured context
    └── RAII drop guard/context, propagate panic / Err without swallowing
```

- 判定顺序固定为：**先阈值/线程环境判断，再进入 guard，再执行并行路径**；禁止先 enter guard 再调用 `should_parallelize()`，否则 guard 已激活会把本次合法并行误判为嵌套并行。
- `ParallelContext` 为私有 token，由 dispatch helper 创建并传给所有内部并行入口；worker 闭包通过捕获该 token 感知当前已在并行区域，从而在尝试再次派发时强制回退串行。

### 6.3 二元逐元素并行路径

```rust
#[cfg(feature = "parallel")]
pub(crate) fn par_zip_map<SL, SR, A, B, C, DL, DR, DO, F>(
    lhs: &TensorBase<SL, DL>,
    rhs: &TensorBase<SR, DR>,
    output_dim: &DO,
    f: F,
) -> Result<Tensor<C, DO>, XenonError>
where
    SL: Storage<Elem = A>,
    SR: Storage<Elem = B>,
    DL: Dimension,
    DR: Dimension,
    DO: Dimension,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> Result<C, XenonError> + Send + Sync,
{
    if !should_parallelize(output_dim.size(), lhs.is_f_contiguous() && rhs.is_f_contiguous()) {
        return crate::kernel::zip_map_serial(lhs, rhs, output_dim, f);
    }

    crate::kernel::zip_map_parallel(lhs, rhs, output_dim, f)
}
```

- `par_zip_map()` 是二元逐元素并行路径的统一设计入口，供 `math` 模块中的 `add` / `sub` / `mul` / `div` 广播运算消费，不直接暴露为公开用户 API。
- `par_zip_map()` 接收的 `lhs`、`rhs` 与 `output_dim` 必须已由调用侧完成兼容性验证；广播裁决（含输出 rank/shape 计算）属于 `math` 模块职责，`parallel/` 不重复做形状推导。
- 广播处理顺序固定为：先由 `math` 模块验证 `lhs` / `rhs` 广播兼容并产出 `output_dim`，再由 `parallel/` 按逻辑线性区间分块；每个 chunk 为两个输入分别构造与该线性区间对应、且仍与 `output_dim` 兼容的只读 sub-view。若某一侧是广播轴（stride 为 `0` 或逻辑重复维），chunk 视图保持该广播语义，不做物理复制。`DL`、`DR`、`DO` 独立建模，以表达输入与输出 rank 可能不同的广播结果。
- 错误传播策略与 `par_map_checked()` 一致：广播形状不兼容时立即返回 `XenonError::ShapeMismatch`；worker 内 `Err(XenonError)` 通过 `rayon` 的 `collect::<Result<Vec<_>, _>>()` 立即向外传播；panic 不捕获、不吞掉。
- 串行回退路径与并行路径共用 `kernel/` 中的私有广播索引与 zip kernel，保证二者使用同一语义基线。

### 6.4 自动路径派发与所有权

```text
math / reduction / dot module
    │
    ├── call kernel::dispatch_parallel_policy(...)
    │       ├── check len vs threshold
    │       ├── check parallel runtime availability
    │       └── choose serial or parallel
    │
    ├── serial path
    │       └── kernel serial implementation
    │
    └── parallel path
            ├── split logical work into fixed chunks
            └── merge partial results with a fixed tree
                    ├── SIMD when supported and aligned
                    └── scalar otherwise
```

- 串并裁决的调用方是 `math`、`reduction`、`dot` 等上层语义模块；这些模块不直接内联阈值和 SIMD 判断，而是调用 `kernel/` 私有 dispatch helper。
- `kernel/` 是内部调度促进者，不改变公开 API 所有权边界；它负责汇总长度、阈值、线程环境、SIMD 能力和对齐信息，并返回当前调用应走的执行策略。
- 判定顺序固定：先依据 `len` 与并行阈值、线程数决定是否“有资格并行”，再通过 `ParallelGuard::enter()` 与私有 `ParallelContext` token 检查是否已在并行区域；只有真正进入 parallel 后，才在每个 chunk 内依据 SIMD 可用性与数据对齐决定 **SIMD vs scalar**。禁止先做 SIMD 判定再反推是否并行。
- `ParallelContext` 采用“显式 token + 私有 dispatch helper”机制，而不是仅靠调用线程 TLS：上层模块一旦进入并行区域，token 会被捕获到 Rayon worker 闭包中，所有内部并行 helper 都必须接收或查询该 token，从而在 worker 线程上也能识别嵌套并行并回退串行。
- `par_map`、`par_zip_map`、`par_sum`、`par_dot` 都遵守这一路径归属，从而避免不同模块各自维护一份阈值/SIMD 分支逻辑。
- 对归约和内积，若并行实现无法提供固定 chunking 与固定 merge tree，从而无法满足文档化的确定性要求，则必须回退串行路径。

### 6.5 Checked 映射与错误传播

```rust
pub fn par_map_checked<A, B, D, F>(
    tensor: &Tensor<A, D>,
    f: F,
) -> Result<Tensor<B, D>, XenonError>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> Result<B, XenonError> + Sync,
{
    if !should_parallelize(tensor.len(), tensor.is_f_contiguous()) {
        let mut output = Vec::with_capacity(tensor.len());
        for elem in tensor.iter() {
            output.push(f(elem)?);
        }
        return Ok(unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) });
    }

    let output: Result<Vec<B>, XenonError> = tensor.par_iter().map(|x| f(x)).collect();
    Ok(unsafe { Tensor::from_raw_vec_unchecked(output?, tensor.raw_dim()) })
}
```

### 6.6 安全性论证

| 主题                             | 论证                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `ParallelGuard`                  | 嵌套并行不是公开错误，而是执行策略问题；因此 guard 由私有 dispatch helper 创建，并同时维护调用线程本地标志与可被 worker 捕获的 `ParallelContext` token。进入失败时回退串行，析构时自动清理，避免库内部再开第二层并行，满足 `需求说明书 §9.2`。 |
| `Tensor::from_raw_vec_unchecked` | 这里只在输出向量长度与 `tensor.raw_dim()` 已由输入张量长度和映射过程保持一致时使用；并行与串行路径都必须保证产出元素数等于输入逻辑元素数。 |
| `par_zip_map` broadcast chunking | 每个并行 chunk 仅借用两个输入的只读 broadcast-compatible sub-view；广播轴保持逻辑重复语义，不进行额外物理展开，因此不会引入越界写或悬垂引用。 |
| panic / `Err` 传播               | 并行 worker 上的 panic 与 `Err(XenonError)` 均不得吞掉或延迟到“全部 worker 完成后再统一检查”；必须沿 `rayon` 传播链即时向外反映。          |

### 6.7 性能考量

| 方面         | 设计决策                                                                        |
| ------------ | ------------------------------------------------------------------------------- |
| 阈值默认值   | 默认阈值为 `1024`，避免小张量并行开销反噬                                       |
| 连续性惩罚   | 非连续视图阈值翻倍，减少步长访问下的伪并行收益                                  |
| 广播分块     | `par_zip_map()` 按逻辑线性区间分块并复用 broadcast-compatible sub-view，避免复制 |
| 原子访问     | 全局阈值使用 `AtomicUsize`，读取/更新开销固定且无需锁                           |
| 嵌套回退     | `ParallelGuard` 优先保护语义和资源占用，再谈并行吞吐                            |
| chunk 内决策 | SIMD/scalar 在 chunk 内局部选择，减少对非对齐尾块的过度保守回退                 |

---

## 7. 实现任务拆分

### Wave 1: 基础状态与路径裁决

- [ ] **T1**: 定义 feature gate 与全局阈值状态
  - 文件: `src/parallel/mod.rs`
  - 内容: `parallel` feature、`DEFAULT_PARALLEL_THRESHOLD`、`GLOBAL_PARALLEL_THRESHOLD`、读写/重置阈值接口
  - 测试: `test_default_threshold`, `test_set_get_threshold`, `test_set_threshold_invalid_argument`
  - 前置: `26-error.md` 错误模型已确定
  - 预计: 10 min

- [ ] **T2**: 实现 `should_parallelize()`
  - 文件: `src/parallel/mod.rs`
  - 内容: 基于线程数、长度、连续性做串并裁决
  - 测试: `test_should_parallelize_single_thread_false`, `test_non_contiguous_threshold_penalty`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 实现 `ParallelGuard`
  - 文件: `src/parallel/mod.rs`
  - 内容: 嵌套并行检测与串行回退约定
  - 测试: `test_nested_parallel_guard`
  - 前置: T2
  - 预计: 10 min

### Wave 2: 并行入口与执行内核

- [ ] **T4**: 实现 `ParElements` 与 `TensorBase::par_iter()`
  - 文件: `src/parallel/mod.rs`
  - 内容: 单输入元素级并行遍历入口
  - 测试: `test_par_iter_len_matches_tensor_len`
  - 前置: T3, `10-iterator.md` 中只读迭代语义已确定
  - 预计: 10 min

- [ ] **T5**: 实现 `par_map` / `par_map_with_threshold`
  - 文件: `src/parallel/mod.rs`
  - 内容: 自动路径选择、显式阈值重载、串行回退
  - 测试: `test_par_map_fallback_small`, `test_par_map_with_threshold_override`
  - 前置: T2, T4
  - 预计: 10 min

- [ ] **T5a**: 实现 `par_zip_map`
  - 文件: `src/parallel/mod.rs`
  - 内容: 二元广播逐元素并行入口，供 `math` 模块消费
  - 测试: `test_par_zip_map_matches_serial_add`, `test_par_zip_map_broadcast_rhs_scalar`, `test_par_zip_map_shape_mismatch`
  - 前置: T2, T4, `math` 广播语义已确定
  - 预计: 10 min

- [ ] **T6**: 实现 `par_reduce_impl` 与 `par_sum`
  - 文件: `src/parallel/mod.rs`
  - 内容: 并行归约、identity 合并、语义对齐串行 `sum`
  - 测试: `test_par_sum_matches_serial`, `test_par_sum_empty_matches_identity`
  - 前置: T3, T4, `13-reduction.md` 归约语义已确定
  - 预计: 10 min

- [ ] **T7**: 实现 `par_dot`
  - 文件: `src/parallel/mod.rs`
  - 内容: shape 检查、并行内积、错误返回与空数组单位元语义
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
  - 文件: `src/parallel/mod.rs`
  - 内容: `XenonError` 透传、panic 不吞掉、非法阈值返回 `InvalidArgument`
  - 测试: `test_parallel_error_propagation`, `test_parallel_panic_propagation`
  - 前置: T5, T6, T7
  - 预计: 10 min

### Wave 4: 配置与回归验证

- [ ] **T10**: 补齐 feature gate 与配置矩阵测试
  - 文件: `src/parallel/mod.rs`, `tests/parallel_feature.rs`
  - 内容: 默认关闭、`--features parallel` 构建、单线程/多线程分支验证
  - 测试: `cargo test`, `cargo test --features parallel`
  - 前置: T1-T9
  - 预计: 10 min

并行关系图：

```text
Wave 1: [T1] -> [T2] -> [T3]
                     │
                     ▼
Wave 2: [T4] -> [T5] -> [T5a] -> [T6] -> [T7]
                     │         │        │       │
                     └─────────┴────────┴──┬────┘
                                            ▼
Wave 3:                               [T8] [T9]
                                            │
                                            ▼
Wave 4:                                    [T10]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置                                           | 目的                                                             |
| ----------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| 单元测试                | `src/parallel/mod.rs` 内联测试模块             | 验证阈值、路径选择、guard、并行入口                              |
| 集成测试                | `tests/parallel_feature.rs` 或等效位置         | 验证跨模块语义与 feature gate 行为                               |
| 边界测试                | 与并行测试配套组织                             | 覆盖空张量、单元素、非连续视图、单线程环境                       |
| 属性测试（按需）        | 当前版本不强制                                 | 当前模块以确定性路径与语义对齐为主，暂无必须引入的随机不变量测试 |
| Feature gate / 配置测试 | `cargo test`, `cargo test --features parallel` | 验证默认关闭与启用并行后语义不变                                 |
| 类型边界 / 编译期测试   | trait 约束测试或编译期失败测试                 | 验证 `bool` 不参与 `par_sum` / `par_dot` 等非法组合              |

### 8.2 单元测试清单

| 测试函数                              | 测试内容                                            | 优先级 |
| ------------------------------------- | --------------------------------------------------- | ------ |
| `test_default_threshold`              | 默认阈值为 `1024`                                   | 高     |
| `test_set_get_threshold`              | 阈值设置与读取一致                                  | 高     |
| `test_set_threshold_invalid_argument` | `threshold == 0` 返回 `XenonError::InvalidArgument` | 高     |
| `test_par_map_fallback_small`         | 小张量自动回退串行                                  | 高     |
| `test_par_zip_map_matches_serial_add` | 二元逐元素并行加法结果与串行一致                     | 高     |
| `test_par_zip_map_broadcast_rhs_scalar` | 右侧标量广播时并行路径与串行一致                  | 高     |
| `test_par_sum_matches_serial`         | 并行 `sum` 与串行语义一致                           | 高     |
| `test_par_dot_matches_serial`         | `par_dot` 与串行结果一致                            | 高     |
| `test_nested_parallel_guard`          | 嵌套并行回退串行                                    | 高     |
| `test_parallel_error_propagation`     | 并行 `Err` 及时上传                                 | 高     |
| `test_parallel_panic_propagation`     | 并行 panic 不被吞掉                                 | 高     |

### 8.3 边界测试场景表

| 场景                 | 预期行为                                                                            |
| -------------------- | ----------------------------------------------------------------------------------- |
| 空数组 `len == 0`    | `par_sum()` 返回加法单位元；`par_dot()` 在两个长度为 `0` 的一维输入上返回加法单位元 |
| 单元素张量           | 即使启用 `parallel` 也默认回退串行                                                  |
| 非连续视图           | 有效阈值翻倍，必要时回退串行且结果仍与串行一致                                      |
| 单线程环境           | `should_parallelize()` 返回 `false`                                                 |
| 长度不匹配的一维输入 | `par_dot()` 返回 `XenonError::ShapeMismatch`                                        |
| 二元广播逐元素输入   | `par_zip_map()` 在广播兼容时返回与串行 `add/sub/mul/div` 一致的结果                 |

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
| 单线程运行          | 不出现伪并行，`should_parallelize()` 返回 `false`          |
| 启用并行 + 嵌套调用 | 不出现第二层并行，内层回退串行                             |

### 8.6 类型边界与编译期测试

| 场景                                | 测试方式                                 |
| ----------------------------------- | ---------------------------------------- |
| `bool` 不参与 `par_sum` / `par_dot` | 编译期 trait 约束测试                    |
| 非法 feature 组合                   | 配置矩阵测试                             |
| 非法阈值参数                        | 运行时返回 `XenonError::InvalidArgument` |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向         | 对方模块     | 接口/类型                                         | 约定                                                 |
| ------------ | ------------ | ------------------------------------------------- | ---------------------------------------------------- |
| 消费（输入） | `tensor`     | `Tensor<A, D>`, `TensorBase<S, D>`                | 调用前已满足 shape、layout、类型约束                 |
| 消费（输入） | `parallel`   | `TensorBase::par_iter()`, `ParElements<'a, A, D>` | 二者均为 `pub(crate)` 内部入口，只提供单输入只读并行遍历 |
| 消费（输入） | `parallel`   | `par_zip_map()`                                  | `math` 模块在完成广播裁决后调用，仅为 crate 内部能力 |
| 消费（输入） | `kernel`     | `dispatch_parallel_policy()`                     | 私有 dispatch helper，统一串并与 chunk 内 SIMD 裁决 |
| 消费（输入） | `error`      | `XenonError`                                      | 可恢复错误统一复用项目错误模型                       |
| 产出（输出） | 上层语义模块 | `Tensor<B, D>` 或 `Result<A, XenonError>`         | 并行与串行路径保持相同外部语义                       |

### 9.2 数据流

```text
math / reduction / dot calls dispatch entry
    │
    ├── query metadata (.len(), .is_f_contiguous(), alignment, simd support)
    ├── kernel::dispatch_parallel_policy(...)
    │       ├── choose serial or parallel by threshold/guard/runtime
    │       └── inside each parallel chunk choose SIMD or scalar
    ├── serial path uses existing semantic baseline
    ├── parallel path uses par_iter() / par_zip_map() / parallel reduce
    └── return Tensor or Result with unchanged public semantics
```

### 9.3 所有权与生命周期约定

> **约定：** `par_iter()` 借用输入张量的只读视图，所有权不转移。并行执行只读取逻辑元素，不得暴露共享可写访问权；输出张量若需要拥有结果数据，必须在本模块内构造新的拥有型结果。

---

## 10. 错误处理与语义边界

| 主题              | 说明                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Recoverable error | `par_dot()` 的 shape 不兼容返回 `XenonError::ShapeMismatch`；`set_parallel_threshold()` 与 `par_map_with_threshold()` 的非法阈值返回 `XenonError::InvalidArgument` |
| Panic             | 归约中的整数溢出仍属于不可恢复错误，必须 panic，而不是包装为 `XenonError`                                                                                          |
| 路径一致性        | 串行 / 并行路径必须返回相同形状、相同错误类别，以及满足同一数值语义约束的结果                                                                                      |
| 容差边界          | 浮点与复数若存在执行路径相关的已知舍入差异，只能落在 `需求说明书 §28.3` 与 `§28.5` 允许且已文档化的范围内                                                           |

```rust
XenonError::ShapeMismatch {
    operation: "dot".into(),
    left_shape: vec![lhs.len()],
    right_shape: vec![rhs.len()],
}

XenonError::InvalidArgument {
    operation: "set_parallel_threshold".into(),
    argument: "threshold".into(),
    expected: "threshold > 0".into(),
    actual: threshold.to_string().into(),
    axis: None,
    shape: None,
}
```

路径语义边界：

- 并行模块本身不新增专属错误枚举；公开错误必须复用 `26-error.md` 中的统一模型。
- 自定义线程池或阈值类参数若存在非法值，应返回 `InvalidArgument`。
- `par_zip_map()` 的广播形状不兼容时，必须返回 `XenonError::ShapeMismatch`，不得降级为逐元素截断或隐式复制。
- panic 与 `Err(XenonError)` 都不得被吞掉或延迟为“全部 worker 完成后再统一检查”。
- 若无法证明并行路径与串行路径一致，必须回退串行，而不是定义新语义。

### 10.1 浮点/复数并行归约容差

- 浮点与复数并行归约允许与标量路径不同的合并顺序；该差异视为合法实现细节，但必须受 `需求说明书 §28.3` 文档化容差约束。
- 与 `08-simd.md` 保持一致：`par_sum()`、`par_dot()` 与其他内部并行归约在浮点或复数输入上，其容差待基于最终算法与测试基线定义。
- 复数按实部、虚部分别适用同一文档化规则；若某一并行实现无法满足该容差或无法提供固定 chunking + fixed merge tree 的确定性约束，则必须回退串行或调整分块/合并策略。

---

## 11. 设计决策记录

### 决策 1：并行阈值采用全局原子状态

| 属性     | 值                                                           |
| -------- | ------------------------------------------------------------ |
| 决策     | 使用 `GLOBAL_PARALLEL_THRESHOLD: AtomicUsize` 保存运行时阈值 |
| 理由     | 读取开销固定、无需锁、便于所有并行入口共享同一裁决基线       |
| 替代方案 | 每次调用显式传参 —— 放弃，会让公开 API 过于分裂              |
| 替代方案 | 使用互斥锁配置对象 —— 放弃，对热点路径不必要                 |

### 决策 2：嵌套并行进入失败时必须回退串行

| 属性     | 值                                                                             |
| -------- | ------------------------------------------------------------------------------ |
| 决策     | `ParallelGuard::enter()` 失败不报错、不 panic，而是选择串行回退                |
| 理由     | `需求说明书 §9.2` 明确禁止库内部二次并行；该场景是执行策略问题而非用户输入错误 |
| 替代方案 | 允许库内部继续二次并行 —— 放弃，违反需求                                       |
| 替代方案 | 将嵌套并行视为 recoverable error —— 放弃，会污染公开 API 语义                  |

补充说明：`ParallelPool` 内部调用同样必须经过 `ParallelGuard`。若用户在自定义 pool 中再次调用内部并行后端，dispatch helper 会把 `ParallelContext` token 捕获到 Rayon worker 闭包中，并在二次派发时自动回退串行，与全局线程池行为一致；同时不允许嵌套 `ParallelPool` 实例，以避免引入额外调度语义。

### 决策 3：并行模块不新增专属公开错误类型

| 属性     | 值                                                      |
| -------- | ------------------------------------------------------- |
| 决策     | 统一使用 `XenonError` 表达 shape 与参数错误             |
| 理由     | 保持跨模块诊断字段与错误类别一致，满足 `需求说明书 §27` |
| 替代方案 | 定义 `ParallelError` —— 放弃，会破坏统一错误模型        |
| 替代方案 | 以 panic 处理非法阈值 —— 放弃，不符合可恢复错误要求     |

### 决策 4：并行归约依赖私有 kernel 而非公开 reduction 模块

| 属性     | 值                                                                                         |
| -------- | ------------------------------------------------------------------------------------------ |
| 决策     | `parallel` 的串行基线依赖下沉到内部私有 `kernel` 模块，不直接依赖公开 `reduction`/`dot` |
| 理由     | 避免 `parallel` 与公开归约/矩阵模块互相调用形成循环依赖，同时保留共享串行 kernel 的单一事实来源 |
| 替代方案 | 直接依赖 `reduction` —— 放弃，会形成架构循环                                               |
| 替代方案 | 在 `parallel` 内复制一份串行逻辑 —— 放弃，会造成语义漂移与维护重复                         |

### 决策 5：二元逐元素并行能力以 `par_zip_map()` 形式提供给 `math`

| 属性     | 值                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------- |
| 决策     | `parallel` 提供 `pub(crate)` 级 `par_zip_map()`，由 `math` 在广播裁决完成后调用               |
| 理由     | 满足 `需求说明书 §9.2` 对逐元素二元运算并行路径的要求，同时不把通用多输入并行迭代器暴露为公开 API |
| 替代方案 | 仅保留 `par_map` —— 放弃，无法覆盖 `add/sub/mul/div` 广播逐元素并行需求                        |
| 替代方案 | 将二元广播并行逻辑直接写进 `math` —— 放弃，会复制阈值/guard/错误传播策略                       |

### 决策 6：自动路径派发由私有 `kernel` dispatch helper 统一收口

| 属性     | 值                                                                                 |
| -------- | ---------------------------------------------------------------------------------- |
| 决策     | `math` / `reduction` / `dot` 调用 `kernel::dispatch_parallel_policy()` 决定执行路径 |
| 理由     | 统一串并阈值、SIMD 能力、数据对齐判断，避免多个模块各自实现分支树                   |
| 替代方案 | 每个模块自行判断 serial/parallel/SIMD —— 放弃，易产生阈值漂移和行为不一致           |
| 替代方案 | 先选 SIMD 再决定是否并行 —— 放弃，与 chunk 级策略不符且会放大非对齐尾块成本         |

---

## 12. 性能描述

### 12.1 复杂度标注

- `get_parallel_threshold()`：时间 `O(1)`，空间 `O(1)`。
- `set_parallel_threshold()`：时间 `O(1)`，空间 `O(1)`。
- `should_parallelize()`：时间 `O(1)`，空间 `O(1)`。
- `par_map()`：时间 `O(n)`，额外结果空间 `O(n)`。
- `par_map_with_threshold()`：时间 `O(n)`，额外结果空间 `O(n)`；当 `threshold == 0` 时返回 `Err(XenonError::InvalidArgument)`。
- `par_sum()`：时间 `O(n)`，额外工作空间取决于 `rayon` 分块；逻辑额外空间 `O(1)`。
- `par_dot()`：时间 `O(n)`，逻辑额外空间 `O(1)`。

### 12.2 缓存与执行特征

| 场景         | 实现路径     | 缓存友好性 | 说明                           |
| ------------ | ------------ | ---------- | ------------------------------ |
| 连续 F-order | 顺序分块并行 | 高         | 适合作为并行阈值的主要收益来源 |
| 非连续视图   | 步长访问并行 | 中到低     | 通过阈值翻倍减少无效并行       |
| 小张量       | 串行回退     | 高         | 避免线程调度开销大于计算本身   |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------- |
| `std` only | 本模块依赖 `rayon` 与原子类型，且项目基线仅支持 `std`；不讨论 `no_std`                       |
| 单 crate   | 设计保持在 Xenon 单 crate 内，不引入额外 crate 拆分                                          |
| SemVer     | `set_parallel_threshold()` 的错误返回语义属于公开 API 契约的一部分，后续变更需遵守兼容性约束 |
| 最小依赖   | 仅使用允许的可选依赖 `rayon`，默认关闭                                                       |

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-14 |
| 1.1.0 | 2026-04-14 |
| 1.1.1 | 2026-04-14 |
| 1.1.2 | 2026-04-14 |
| 1.2.0 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
