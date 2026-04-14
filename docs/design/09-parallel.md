# 并行后端模块设计

> 文档编号: 09 | 模块: `src/parallel/` | 阶段: Phase 5
> 前置文档: `07-tensor.md`, `10-iterator.md`, `13-reduction.md`, `26-error.md`
> 需求参考: 需求说明书 §9.2, §9.3
> 范围声明: 范围内

---

## §1 Overview（概述）

并行后端模块是 Xenon 的可选执行后端，通过 `rayon` 为逐元素映射与归约提供数据并行能力。该模块默认关闭，仅在启用 `parallel` feature 时参与构建。

当前版本的职责边界如下：

- 范围内：`par_map`、阈值控制、并行 `sum` / `dot` 分派、嵌套并行防护、panic / `Err` 传播一致性
- 范围外：GPU 后端、自动任务图调度、多数组并行同步接口
- 关键约束：并行路径不得改变串行语义；若无法证明一致性，必须回退串行

## §2 Data Structures（数据结构）

### 2.1 Feature gate 与配置

```toml
[features]
default = ["std"]
std = []
parallel = ["dep:rayon", "std"]

[dependencies]
rayon = { version = "1.10", optional = true }
```

并行模块只依赖 `rayon` 与 `std`，不直接依赖 `simd/`。若上层语义模块需要“并行 + SIMD”组合，必须由上层分派策略协调，`parallel/` 只负责并行执行边界。

### 2.2 核心运行时结构

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

- `GLOBAL_PARALLEL_THRESHOLD`：控制自动串并切换阈值
- `ParallelGuard`：防止嵌套并行；进入失败时必须回退串行而不是 panic
- `ParallelPool`：可选自定义线程池包装，仅改变执行上下文，不改变 API 语义

## §3 API（接口）

### 3.1 公共入口

```rust
#[cfg(feature = "parallel")]
pub fn get_parallel_threshold() -> usize;

#[cfg(feature = "parallel")]
pub fn set_parallel_threshold(threshold: usize);

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

### 3.2 并行迭代入口

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

## §4 Algorithm（算法）

### §4.1 Invariants（不变式）

- 仅支持 F-order 语义；并行模块不得引入 C-order 前提或额外布局状态
- `should_parallelize()` 只读取长度与 F-order 连续性，不改变张量元数据
- 进入并行区域前必须先尝试 `ParallelGuard::enter()`；失败时只能串行回退
- 并行路径返回的逻辑结果必须与串行路径一致；若无法证明一致性，必须回退串行
- panic 与 `Err(XenonError)` 都不得被吞掉或延迟为“全部 worker 完成后再统一检查”

### §4.2 Error Scenarios（错误场景）

并行模块本身不新增专属错误枚举；公开错误必须复用 `26-error.md` 中的统一模型。

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

- `par_dot()` 的 shape 不兼容必须返回 `ShapeMismatch`
- 自定义线程池或阈值类参数若存在非法值，应返回 `InvalidArgument`
- 归约中的整数溢出仍属于不可恢复错误，必须 panic，而不是包装为 `XenonError`
- 线程池初始化失败属于实现接线问题时，可映射到统一错误模型；不得发明仅限并行模块的公开字段名

### 4.3 路径选择与传播

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

## §5 Testing（测试）

| 测试函数 | 目的 |
| --- | --- |
| `test_default_threshold` | 默认阈值为 1024 |
| `test_set_get_threshold` | 阈值设置与读取一致 |
| `test_par_map_fallback_small` | 小张量自动回退串行 |
| `test_par_sum_matches_serial` | 并行 `sum` 与串行语义一致 |
| `test_par_dot_matches_serial` | `par_dot` 与串行结果一致 |
| `test_nested_parallel_guard` | 嵌套并行回退串行 |
| `test_parallel_error_propagation` | 并行 `Err` 及时上传 |
| `test_parallel_panic_propagation` | 并行 panic 不被吞掉 |

还必须覆盖以下场景：

- `--features parallel` 与默认构建均可编译
- 单线程环境下 `should_parallelize()` 返回 `false`
- 非连续视图阈值翻倍后仍保持正确回退行为
- 任何并行路径都不依赖多数组同步迭代接口

## §6 References（参考）

- `07-tensor.md`
- `10-iterator.md`
- `13-reduction.md`
- `26-error.md`
- 需求说明书 §9.2, §9.3
