# 并行后端模块设计

> 文档编号: 09 | 模块: `src/parallel/` | 阶段: Phase 5
> 前置文档: `07-tensor.md`, `10-iterator.md`, `13-reduction.md`, `26-error.md`
> 需求参考: 需求说明书 §9.2, §9.3
> 范围声明: 范围内

---

## 1. 模块定位/概述

并行后端模块是 Xenon 的可选执行后端，通过 `rayon` 为逐元素映射与归约提供数据并行能力。该模块默认关闭，仅在启用 `parallel` feature 时参与构建。当前版本覆盖 `par_map`、并行 `sum` / `dot`、阈值控制、自动串并路径选择与嵌套并行防护；若无法证明并行路径与串行路径的语义一致，必须回退串行。

### 1.1 职责边界表

| 职责 | 包含 | 不包含 |
| ---- | ---- | ------ |
| 并行逐元素执行 | `par_map`、`par_map_with_threshold`、基于单输入视图的并行遍历 | 多输入同步并行公开 API |
| 并行归约 | `par_sum`、`par_dot`、内部 `par_reduce_impl` | 矩阵乘法、矩阵分解、GPU 后端 |
| 路径选择 | 阈值控制、连续性判断、单线程环境自动回退 | 自动任务图调度、跨模块全局调度器 |
| 嵌套并行治理 | `ParallelGuard` 防止库内部二次并行 | 用户侧任意线程模型抽象 |
| 线程池封装 | `ParallelPool` 改变执行上下文而不改变 API 语义 | 新的公开调度语义或额外第三方依赖 |

### 1.2 设计原则

| 原则 | 体现 |
| ---- | ---- |
| 语义一致性 | 并行路径不得改变公开 API 的形状、错误类别和数值语义；无法证明一致性时回退串行 |
| 自动回退 | 小张量、非连续视图惩罚、单线程环境或嵌套并行场景都自动回退串行 |
| 最小能力边界 | 当前版本只覆盖 `par_map`、`par_sum`、`par_dot`，不扩展到 GPU 或多输入同步接口 |
| 可选依赖最小化 | 仅在 `parallel` feature 下引入 `rayon`，默认关闭 |

### 1.3 架构位置

```text
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout
L3: storage
L4: tensor
L5: iter, reduction
L6: parallel  <- current module (optional, feature = "parallel")
```

---

## 2. 需求映射与范围约束

| 类型 | 内容 |
| ---- | ---- |
| 需求映射 | `需求说明书 §9.2`, `§9.3`, `§27`, `§28` |
| 范围内 | 可选数据并行、并行阈值配置、逐元素运算/归约/内积的并行执行路径、自动串并选择、禁止库内二次并行 |
| 范围外 | GPU 后端、自动任务图调度、多数组 lock-step 并行公开接口、额外第三方依赖 |
| 非目标 | 不把文档改成 `no_std`，不增加除 `rayon` 之外的外部依赖，不扩展当前并行能力集合 |

---

## 3. 文件位置

```text
src/parallel/
└── mod.rs    # feature gate、阈值状态、ParallelGuard/ParallelPool、par_map/par_sum/par_dot
```

单文件设计：当前版本公开能力集中于阈值配置、路径裁决和少量并行入口，单文件足以容纳核心状态与执行边界；若后续范围发生变化，再评估是否拆分内部实现文件，但不影响本设计的公开能力边界。

---

## 4. 依赖关系

### 4.1 依赖图

```text
src/parallel/
├── rayon (optional)         # ThreadPool, ParallelIterator, current_num_threads
├── crate::tensor            # Tensor, TensorBase, TensorView, Ix1
├── crate::iter              # par_iter() entry and iterator integration
├── crate::reduction         # serial reduction semantics mirrored by par_sum/par_dot
└── crate::error             # XenonError
```

### 4.2 类型级依赖表

| 来源模块 | 使用的类型/trait |
| -------- | ---------------- |
| `rayon` | `rayon::ThreadPool`, `rayon::current_num_threads`, `rayon::iter::ParallelIterator` |
| `tensor` | `Tensor<A, D>`, `TensorBase<S, D>`, `TensorView<'a, A, D>`, `Ix1`, `.len()`, `.raw_dim()`, `.is_f_contiguous()` |
| `iter` | `ParElements<'a, A, D>`, `TensorBase::par_iter()` |
| `reduction` | 串行 `sum` / `dot` 的语义基线与 identity / combine 约束 |
| `error` | `XenonError`, `XenonError::ShapeMismatch`, `XenonError::InvalidArgument` |

### 4.3 依赖方向

> **依赖方向：单向向上。** `parallel/` 仅消费 `tensor`、`iter`、`reduction`、`error` 和可选 `rayon`，不被这些基础模块反向依赖。并行路径只建立在上层已完成的张量形状、布局与类型约束之上。

### 4.4 合法性声明

| 项目 | 说明 |
| ---- | ---- |
| 新增第三方依赖 | `rayon`（可选） |
| 合法性结论 | 合法；符合 `需求说明书 §1.2` 对最小依赖的限制，以及 `§9.2` 对可选并行能力的要求 |
| 替代方案 | 仅用 `std::thread` 不能无损提供当前所需的并行迭代与线程池抽象，因此不采用 |

### 4.5 与迭代器模块的边界

`parallel/` 不定义多输入同步并行公开迭代接口。`TensorBase::par_iter()` 只提供单输入元素级并行入口，逐元素双输入或更高阶并行调度仍由上层语义模块先完成形状裁决，再决定是否复用本模块的单输入并行内核。该边界与 `10-iterator.md §1.2` 中“并行迭代不属于 `iter` 模块公开职责”保持一致。

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

pub struct ParallelGuard;

#[cfg(feature = "parallel")]
pub struct ParallelPool {
    inner: rayon::ThreadPool,
}
```

- `GLOBAL_PARALLEL_THRESHOLD`：控制自动串并切换阈值。
- `ParallelGuard`：防止嵌套并行；进入失败时必须回退串行而不是 panic。
- `ParallelPool`：可选自定义线程池包装，只改变执行上下文，不改变 API 语义。

### 5.2 公共函数签名

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
pub fn par_map<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync;

#[cfg(feature = "parallel")]
pub fn par_map_with_threshold<A, B, D, F>(
    tensor: &TensorBase<impl Storage<Elem = A>, D>,
    f: F,
    threshold: usize,
) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Send + Sync;

#[cfg(feature = "parallel")]
pub(crate) fn par_reduce_impl<A, D, F, ID>(
    tensor: &Tensor<A, D>,
    identity: ID,
    op: F,
) -> A
where
    D: Dimension,
    A: Element + Send + Sync + Clone,
    F: Fn(A, A) -> A + Sync,
    ID: Fn() -> A + Sync + Clone;

#[cfg(feature = "parallel")]
pub fn par_sum<A, D>(tensor: &Tensor<A, D>) -> A
where
    D: Dimension,
    A: Element + Numeric + Send + Sync + Clone;

#[cfg(feature = "parallel")]
pub fn par_dot<A>(lhs: &Tensor<A, Ix1>, rhs: &Tensor<A, Ix1>) -> Result<A, XenonError>
where
    A: Element + Numeric + Send + Sync + Clone;
```

> **设计决策：** `set_parallel_threshold(threshold)` 返回 `Result<(), XenonError>`，而不是无返回值函数。阈值为 `0` 或其他非法值时，必须返回 `XenonError::InvalidArgument`，以满足 `需求说明书 §27` 的可恢复错误要求。

### 5.3 并行迭代入口

```rust
#[cfg(feature = "parallel")]
pub struct ParElements<'a, A, D>
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
    pub fn par_iter(&self) -> ParElements<'_, A, D> {
        ParElements::new(self.view())
    }
}
```

当前版本不提供任何并行双输入同步 API；需要多输入逐元素调度时，应先由上层语义模块完成形状与执行策略裁决，再决定是否复用单输入并行内核。

### 5.4 Good / Bad 对比示例

```rust
// Good - invalid threshold returns a recoverable error.
pub fn set_parallel_threshold(threshold: usize) -> Result<(), XenonError> {
    if threshold == 0 {
        return Err(XenonError::InvalidArgument {
            operation: "set_parallel_threshold",
            argument: "threshold",
            expected: "threshold > 0",
            actual: threshold.to_string(),
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

### 5.5 文档与示例交付要求

| API | 文档要求 | 示例要求 |
| --- | -------- | -------- |
| `set_parallel_threshold` | 说明默认阈值、合法值范围、错误返回 | 给出成功与非法阈值示例 |
| `should_parallelize` | 说明其只基于长度、连续性和线程数做判断 | 给出小张量回退示例 |
| `par_map` / `par_sum` / `par_dot` | 说明与串行路径的语义一致性 | 给出 feature 开启后的调用示例 |
| `ParallelPool` | 说明仅改变执行上下文，不改变语义 | 给出在线程池内执行并行入口的示例 |

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
2. 连续 F-order 张量使用全局阈值。
3. 非连续视图将有效阈值翻倍，以覆盖步长访问的缓存损失。
4. 只有长度达到有效阈值时才进入并行路径。

### 6.2 核心执行路径

```text
public API call
    │
    ├── read len + contiguity metadata
    ├── ParallelGuard::enter()
    │       ├── fail -> serial fallback
    │       └── pass -> continue
    ├── should_parallelize(...)
    │       ├── false -> serial fallback
    │       └── true  -> rayon parallel path
    └── propagate panic / Err without swallowing
```

### 6.3 Checked 映射与错误传播

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

### 6.4 安全性论证

| 主题 | 论证 |
| ---- | ---- |
| `ParallelGuard` | 嵌套并行不是公开错误，而是执行策略问题；因此进入失败时回退串行，避免库内部再开第二层并行，满足 `需求说明书 §9.2`。 |
| `Tensor::from_raw_vec_unchecked` | 这里只在输出向量长度与 `tensor.raw_dim()` 已由输入张量长度和映射过程保持一致时使用；并行与串行路径都必须保证产出元素数等于输入逻辑元素数。 |
| panic / `Err` 传播 | 并行 worker 上的 panic 与 `Err(XenonError)` 均不得吞掉或延迟到“全部 worker 完成后再统一检查”；必须沿 `rayon` 传播链即时向外反映。 |

### 6.5 性能考量

| 方面 | 设计决策 |
| ---- | -------- |
| 阈值默认值 | 默认阈值为 `1024`，避免小张量并行开销反噬 |
| 连续性惩罚 | 非连续视图阈值翻倍，减少步长访问下的伪并行收益 |
| 原子访问 | 全局阈值使用 `AtomicUsize`，读取/更新开销固定且无需锁 |
| 嵌套回退 | `ParallelGuard` 优先保护语义和资源占用，再谈并行吞吐 |

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
Wave 2: [T4] -> [T5] -> [T6] -> [T7]
                     │       │       │
                     └───────┴──┬────┘
                                 ▼
Wave 3:                    [T8] [T9]
                                 │
                                 ▼
Wave 4:                         [T10]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型 | 位置 | 目的 |
| ---- | ---- | ---- |
| 单元测试 | `src/parallel/mod.rs` 内联测试模块 | 验证阈值、路径选择、guard、并行入口 |
| 集成测试 | `tests/parallel_feature.rs` 或等效位置 | 验证跨模块语义与 feature gate 行为 |
| 边界测试 | 与并行测试配套组织 | 覆盖空张量、单元素、非连续视图、单线程环境 |
| 属性测试（按需） | 当前版本不强制 | 当前模块以确定性路径与语义对齐为主，暂无必须引入的随机不变量测试 |
| Feature gate / 配置测试 | `cargo test`, `cargo test --features parallel` | 验证默认关闭与启用并行后语义不变 |
| 类型边界 / 编译期测试 | trait 约束测试或编译期失败测试 | 验证 `bool` 不参与 `par_sum` / `par_dot` 等非法组合 |

### 8.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
| -------- | -------- | ------ |
| `test_default_threshold` | 默认阈值为 `1024` | 高 |
| `test_set_get_threshold` | 阈值设置与读取一致 | 高 |
| `test_set_threshold_invalid_argument` | `threshold == 0` 返回 `XenonError::InvalidArgument` | 高 |
| `test_par_map_fallback_small` | 小张量自动回退串行 | 高 |
| `test_par_sum_matches_serial` | 并行 `sum` 与串行语义一致 | 高 |
| `test_par_dot_matches_serial` | `par_dot` 与串行结果一致 | 高 |
| `test_nested_parallel_guard` | 嵌套并行回退串行 | 高 |
| `test_parallel_error_propagation` | 并行 `Err` 及时上传 | 高 |
| `test_parallel_panic_propagation` | 并行 panic 不被吞掉 | 高 |

### 8.3 边界测试场景表

| 场景 | 预期行为 |
| ---- | -------- |
| 空数组 `len == 0` | `par_sum()` 返回加法单位元；`par_dot()` 在两个长度为 `0` 的一维输入上返回加法单位元 |
| 单元素张量 | 即使启用 `parallel` 也默认回退串行 |
| 非连续视图 | 有效阈值翻倍，必要时回退串行且结果仍与串行一致 |
| 单线程环境 | `should_parallelize()` 返回 `false` |
| 长度不匹配的一维输入 | `par_dot()` 返回 `XenonError::ShapeMismatch` |

### 8.4 属性测试与不变量

| 不变量 | 测试方法 |
| ------ | -------- |
| `par_map` 与串行 `map` 在相同输入上产出相同形状与逐元素值 | 对整数类型可按多组形状和布局做表驱动校验 |
| `par_sum` / `par_dot` 在相同执行路径和配置下结果确定 | 对相同输入重复运行并比较结果 |

### 8.5 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ------ |
| 默认配置 | 可选并行默认关闭，默认构建可编译 |
| 启用 `parallel` | `par_map` / `par_sum` / `par_dot` 可用，结果与串行路径一致 |
| 单线程运行 | 不出现伪并行，`should_parallelize()` 返回 `false` |
| 启用并行 + 嵌套调用 | 不出现第二层并行，内层回退串行 |

### 8.6 类型边界与编译期测试

| 场景 | 测试方式 |
| ---- | -------- |
| `bool` 不参与 `par_sum` / `par_dot` | 编译期 trait 约束测试 |
| 非法 feature 组合 | 配置矩阵测试 |
| 非法阈值参数 | 运行时返回 `XenonError::InvalidArgument` |

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
| ---- | -------- | --------- | ---- |
| 消费（输入） | `tensor` | `Tensor<A, D>`, `TensorBase<S, D>` | 调用前已满足 shape、layout、类型约束 |
| 消费（输入） | `iter` | `TensorBase::par_iter()`, `ParElements<'a, A, D>` | 只提供单输入只读并行遍历 |
| 消费（输入） | `error` | `XenonError` | 可恢复错误统一复用项目错误模型 |
| 产出（输出） | 上层语义模块 | `Tensor<B, D>` 或 `Result<A, XenonError>` | 并行与串行路径保持相同外部语义 |

### 9.2 数据流

```text
User calls par_sum() / par_dot() / par_map()
    │
    ├── tensor metadata query (.len(), .is_f_contiguous())
    ├── threshold + guard check
    ├── choose serial or rayon path
    │       ├── serial path uses existing semantic baseline
    │       └── rayon path uses par_iter() / parallel reduce
    └── return Tensor or Result with unchanged public semantics
```

### 9.3 所有权与生命周期约定

> **约定：** `par_iter()` 借用输入张量的只读视图，所有权不转移。并行执行只读取逻辑元素，不得暴露共享可写访问权；输出张量若需要拥有结果数据，必须在本模块内构造新的拥有型结果。

---

## 10. 错误处理与语义边界

| 主题 | 说明 |
| ---- | ---- |
| Recoverable error | `par_dot()` 的 shape 不兼容返回 `XenonError::ShapeMismatch`；`set_parallel_threshold()` 的非法值返回 `XenonError::InvalidArgument` |
| Panic | 归约中的整数溢出仍属于不可恢复错误，必须 panic，而不是包装为 `XenonError` |
| 路径一致性 | 串行 / 并行路径必须返回相同形状、相同错误类别，以及满足同一数值语义约束的结果 |
| 容差边界 | 浮点与复数若存在执行路径相关的已知舍入差异，只能落在 `需求说明书 §28.3` 允许且已文档化的范围内 |

```rust
XenonError::ShapeMismatch {
    operation: "dot",
    left_shape: Cow::Borrowed(&[lhs.len()]),
    right_shape: Cow::Borrowed(&[rhs.len()]),
}

XenonError::InvalidArgument {
    operation: "set_parallel_threshold",
    argument: "threshold",
    expected: "threshold > 0",
    actual: threshold.to_string(),
    axis: None,
    shape: None,
}
```

路径语义边界：

- 并行模块本身不新增专属错误枚举；公开错误必须复用 `26-error.md` 中的统一模型。
- 自定义线程池或阈值类参数若存在非法值，应返回 `InvalidArgument`。
- panic 与 `Err(XenonError)` 都不得被吞掉或延迟为“全部 worker 完成后再统一检查”。
- 若无法证明并行路径与串行路径一致，必须回退串行，而不是定义新语义。

---

## 11. 设计决策记录

### 决策 1：并行阈值采用全局原子状态

| 属性 | 值 |
| ---- | -- |
| 决策 | 使用 `GLOBAL_PARALLEL_THRESHOLD: AtomicUsize` 保存运行时阈值 |
| 理由 | 读取开销固定、无需锁、便于所有并行入口共享同一裁决基线 |
| 替代方案 | 每次调用显式传参 —— 放弃，会让公开 API 过于分裂 |
| 替代方案 | 使用互斥锁配置对象 —— 放弃，对热点路径不必要 |

### 决策 2：嵌套并行进入失败时必须回退串行

| 属性 | 值 |
| ---- | -- |
| 决策 | `ParallelGuard::enter()` 失败不报错、不 panic，而是选择串行回退 |
| 理由 | `需求说明书 §9.2` 明确禁止库内部二次并行；该场景是执行策略问题而非用户输入错误 |
| 替代方案 | 允许库内部继续二次并行 —— 放弃，违反需求 |
| 替代方案 | 将嵌套并行视为 recoverable error —— 放弃，会污染公开 API 语义 |

### 决策 3：并行模块不新增专属公开错误类型

| 属性 | 值 |
| ---- | -- |
| 决策 | 统一使用 `XenonError` 表达 shape 与参数错误 |
| 理由 | 保持跨模块诊断字段与错误类别一致，满足 `需求说明书 §27` |
| 替代方案 | 定义 `ParallelError` —— 放弃，会破坏统一错误模型 |
| 替代方案 | 以 panic 处理非法阈值 —— 放弃，不符合可恢复错误要求 |

---

## 12. 性能描述

### 12.1 复杂度标注

- `get_parallel_threshold()`：时间 `O(1)`，空间 `O(1)`。
- `set_parallel_threshold()`：时间 `O(1)`，空间 `O(1)`。
- `should_parallelize()`：时间 `O(1)`，空间 `O(1)`。
- `par_map()`：时间 `O(n)`，额外结果空间 `O(n)`。
- `par_sum()`：时间 `O(n)`，额外工作空间取决于 `rayon` 分块；逻辑额外空间 `O(1)`。
- `par_dot()`：时间 `O(n)`，逻辑额外空间 `O(1)`。

### 12.2 缓存与执行特征

| 场景 | 实现路径 | 缓存友好性 | 说明 |
| ---- | -------- | ---------- | ---- |
| 连续 F-order | 顺序分块并行 | 高 | 适合作为并行阈值的主要收益来源 |
| 非连续视图 | 步长访问并行 | 中到低 | 通过阈值翻倍减少无效并行 |
| 小张量 | 串行回退 | 高 | 避免线程调度开销大于计算本身 |

---

## 13. 平台与工程约束

| 约束 | 说明 |
| ---- | ---- |
| `std` only | 本模块依赖 `rayon` 与原子类型，且项目基线仅支持 `std`；不讨论 `no_std` |
| 单 crate | 设计保持在 Xenon 单 crate 内，不引入额外 crate 拆分 |
| SemVer | `set_parallel_threshold()` 的错误返回语义属于公开 API 契约的一部分，后续变更需遵守兼容性约束 |
| 最小依赖 | 仅使用允许的可选依赖 `rayon`，默认关闭 |

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.1.0 | 2026-04-14 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
