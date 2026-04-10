# 并行后端模块设计

> 文档编号: 09 | 模块: `src/parallel/` | 阶段: Phase 5
> 前置文档: `07-tensor.md`, `10-iterator.md`
> 需求参考: 需求说明书 §9.2, §9.3

---

## 1. 模块定位

并行后端模块是 Xenon 张量库的可选性能加速层，通过 `rayon` crate 提供数据并行能力，为大规模数组操作提供多线程加速。该模块默认关闭，通过 `features = ["parallel"]` 启用。

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| rayon 集成 | 全局线程池、自定义线程池配置 | 自定义线程池调度策略 |
| 逐元素并行 | a + b 等逐元素运算的并行执行 | 嵌套并行 |
| 并行归约 | sum 的并行求和 | GPU 并行 |
| 函数映射并行 | apply / map_inplace 并行执行 | BLAS 并行绑定 |
| 多数组同步 | zip 多数组并行迭代 | 自动 DAG 调度 |
| 自动路径选择 | 根据数据规模自动串行/并行切换 | 编译期静态并行分发 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 透明集成 | 用户代码无需修改，仅通过 feature gate 启用 |
| 自动决策 | 运行时根据数据规模和布局自动选择并行/串行 |
| 可配置性 | 支持全局配置和单次调用覆盖阈值 |
| 安全性 | 禁止嵌套并行，保证无数据竞争；通过 `ParallelGuard` 运行时守卫检测并回退串行 |
| 兼容性 | 可被上层语义模块与 SIMD 组合使用，但 `parallel/` 本身不直接依赖 `simd/` |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (仅依赖 core/alloc，不依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: parallel  ← 当前模块（可选，feature = "parallel"）
```

### 1.4 性能分层中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                      性能分层决策树                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  元素数 ≥ PARALLEL_THRESHOLD (默认 1024) 且 parallel 启用        │
│  └─→ 并行路径                                                   │
│      └─ 每个分块内部使用的具体执行内核（SIMD 或标量）由上层语义模块 │
│         或共享 dispatch 辅助层决定；`parallel/` 本身不直接裁决 SIMD │
│                                                                 │
│  默认情况                                                       │
│  └─→ 串行路径                                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 文件位置

```
src/parallel/
├── mod.rs             # 模块入口、rayon 集成、全局阈值配置、模块导出
├── par_iter.rs        # 并行迭代器（ParElements, ParZip）
└── par_ops.rs         # 并行运算（par_map, par_reduce, par_zip_with）
```

模块划分理由：`mod.rs` 管理配置和公共导出；`par_iter.rs` 封装并行迭代逻辑；`par_ops.rs` 实现具体的并行运算。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/parallel/
├── rayon (可选)               # 外部依赖，feature = "parallel"
├── crate::tensor             # TensorBase<S, D>, 类型别名
├── crate::storage            # Storage, RawStorage trait
├── crate::layout             # LayoutFlags, is_f_contiguous()
├── crate::element            # Element trait
├── crate::broadcast          # broadcast_shape() (zip 场景)
└── （无 direct backend 依赖；SIMD 组合由上层调度层决定）
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `rayon` | `ThreadPool`, `ThreadPoolBuilder`, `ParallelIterator`, `IndexedParallelIterator` |
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.view()`, `.len()`, `Tensor::from_raw_vec_unchecked`（参见 `07-tensor.md §4.5`，pub(crate) 方法）（参见 `07-tensor.md §4`） |
| `storage` | `RawStorage`, `Storage`, `StorageMut`, `.as_slice()`（参见 `05-storage.md §4`） |
| `layout` | `LayoutFlags`, `is_f_contiguous()`（参见 `06-memory.md §4`） |
| `broadcast` | `broadcast_shape()`（参见 `15-broadcast.md §4`） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `parallel/` 仅消费 `tensor`、`storage`、`layout` 等核心模块，不被它们依赖。与 `simd/` 的组合由上层语义模块协调，而不是 `parallel/` 直接依赖 `simd/`；`parallel/` 模块在未启用 feature 时完全不存在。

> **说明**: `parallel` 依赖 `broadcast::broadcast_shape()` 用于 zip 场景。虽然两者同属 L5，但 broadcast 仅消费 L1-L4 的类型，不存在循环依赖。

---

## 4. 公共 API 设计

### 4.1 Xenon 并行约束

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
parallel = ["dep:rayon", "std"]

[dependencies]
rayon = { version = "1.10", optional = true }
```

- 默认关闭，通过 `features = ["parallel"]` 显式启用
- 启用后 rayon 与 `std` 同时引入，提供线程池、原子变量和数据并行能力

> **约束说明：** `parallel` 不是纯 `no_std` feature。并行后端依赖 `std::sync::atomic`、rayon 线程池与线程本地上下文，因此文档、CI 和 feature 矩阵都必须把 `parallel` 视为 `std` 扩展能力。

### 4.2 并行阈值系统

```rust
// src/parallel/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

/// Default parallel threshold: 1024 elements.
///
/// When the number of array elements reaches or exceeds this value,
/// parallel execution is enabled. This value is empirically tuned;
/// on small arrays, parallelization overhead outweighs the benefit.
const DEFAULT_PARALLEL_THRESHOLD: usize = 1024;

/// Global parallel threshold.
static GLOBAL_PARALLEL_THRESHOLD: AtomicUsize =
    AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// Get the current global parallel threshold
#[inline]
pub fn get_parallel_threshold() -> usize {
    GLOBAL_PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

/// Set the global parallel threshold
///
/// # Note
///
/// This setting affects all subsequent parallel operations.
/// It is recommended to set this at program startup to avoid
/// frequent runtime modifications.
pub fn set_parallel_threshold(threshold: usize) {
    GLOBAL_PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// Reset the global parallel threshold to the default value
pub fn reset_parallel_threshold() {
    GLOBAL_PARALLEL_THRESHOLD.store(DEFAULT_PARALLEL_THRESHOLD, Ordering::Relaxed);
}
```

> **设计决策：** 使用 `Ordering::Relaxed`。阈值读取不需要与其他操作同步，稍旧的值也可接受。阈值修改通常是启动时一次性操作。

### 4.3 自动路径选择

```rust
// src/parallel/mod.rs

/// Determine whether parallel execution should be enabled
///
/// # Decision logic
///
/// 1. Check if the `parallel` feature is enabled
/// 2. Check if the element count meets the threshold
/// 3. Consider data layout (non-contiguous arrays need a higher threshold)
///
/// # Arguments
///
/// * `len` - Number of elements
/// * `is_f_contiguous` - Whether the data is F-order contiguous
///
/// # Returns
///
/// `true` if parallel execution should be enabled.
#[cfg(feature = "parallel")]
pub fn should_parallelize(len: usize, is_f_contiguous: bool) -> bool {
    // If only one thread is available, parallelism adds overhead with no benefit
    if rayon::current_num_threads() <= 1 {
        return false;
    }
    let threshold = get_parallel_threshold();
    let meets_threshold = if is_f_contiguous {
        len >= threshold
    } else {
        // Non-contiguous: double the threshold to account for overhead
        len >= threshold * 2
    };
    meets_threshold
}

#[cfg(not(feature = "parallel"))]
pub fn should_parallelize(_len: usize, _is_contiguous: bool) -> bool {
    false
}
```

### 4.4 并行运算 Trait

```rust
// src/parallel/par_ops.rs

use rayon::prelude::*;

/// Parallel map operation
///
/// # Constraints
///
/// * `F: Fn(&A) -> B + Sync` - Function must be shareable across threads
/// * `A: Send + Sync` - Input elements must be thread-safe
/// * `B: Send` - Output elements must be sendable across threads
///
/// # Automatic decision
///
/// If the element count is below the parallel threshold, falls back to serial `map`.
#[cfg(feature = "parallel")]
pub fn par_map<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync,
{
    let _guard = match ParallelGuard::enter() {
        Some(g) => g,
        None => {
            // Already in parallel context, fall back to serial
            return tensor.map(f);
        }
    };
    if !should_parallelize(tensor.len(), tensor.is_f_contiguous()) {
        return tensor.map(f);
    }

    let len = tensor.len();
    let output: Vec<B> = tensor.par_iter().map(|x| f(x)).collect();

    // SAFETY: output length == tensor.len()
    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}

/// Parallel map with a custom parallelization threshold.
/// Only parallelizes if the tensor has more than `threshold` elements.
#[cfg(feature = "parallel")]
pub fn par_map_with_threshold<A, B, D, F>(
    tensor: &TensorBase<impl Storage<Elem = A>, D>,
    f: F,
    threshold: usize,
) -> Tensor<B, D>
where
    A: Element + Send + Sync,
    B: Element + Send,
    D: Dimension,
    F: Fn(&A) -> B + Send + Sync,
{
    let _guard = match ParallelGuard::enter() {
        Some(g) => g,
        None => {
            // Already in parallel context, fall back to serial
            let output: Vec<B> = tensor.iter().map(|x| f(x)).collect();
            // SAFETY: output length == tensor.len()
            return unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) };
        }
    };
    let len = tensor.len();
    if len < threshold {
        // Below threshold: serial execution
        let output: Vec<B> = tensor.iter().map(|x| f(x)).collect();
        // SAFETY: output length == tensor.len()
        return unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) };
    }

    let output: Vec<B> = tensor.par_iter().map(|x| f(x)).collect();

    // SAFETY: output length == tensor.len()
    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}

/// Parallel reduction
///
/// # Constraints
///
/// * The reduction function must be associative
/// * The identity function must return the identity element
#[cfg(feature = "parallel")]
pub fn par_reduce<A, D, F, ID>(tensor: &Tensor<A, D>, identity: ID, op: F) -> A
where
    D: Dimension,
    A: Element + Send + Sync + Clone,
    F: Fn(A, A) -> A + Sync,
    ID: Fn() -> A + Sync + Clone,
{
    let _guard = match ParallelGuard::enter() {
        Some(g) => g,
        None => {
            // Already in parallel context, fall back to serial
            return tensor.iter().fold(identity(), |acc, x| op(acc, *x));
        }
    };
    if tensor.is_empty() {
        return identity();
    }
    if !should_parallelize(tensor.len(), tensor.is_f_contiguous()) {
        return tensor.iter().fold(identity(), |acc, x| op(acc, *x));
    }

    tensor.par_iter().cloned().reduce(identity, op)
}

/// Parallel sum
#[cfg(feature = "parallel")]
pub fn par_sum<A, D>(tensor: &Tensor<A, D>) -> A
where
    D: Dimension,
    A: Element + Numeric + Send + Sync + Clone,
{
    // For integer reductions, the implementation must use checked accumulation
    // and panic on overflow (see 13-reduction.md §5.1 / 26-error.md §5.2).
    // For floating-point reductions, if exact serial equivalence cannot be proven,
    // this entry must transparently fall back to the serial baseline.
par_reduce(tensor, || A::zero(), |a, b| {
    // For integer reductions, dispatch to CheckedAdd in the scalar fallback path
    // or disable parallelization when exact serial equivalence cannot be proven.
    a.checked_add(b).expect("parallel integer reduction overflow")
})
}

/// Parallel zip operation
///
/// # Constraints
///
/// * Shapes must be compatible (identical or broadcastable)
/// * Supports broadcasting
#[cfg(feature = "parallel")]
pub fn par_zip_with<A, B, C, DA, DB, F>(
    a: &Tensor<A, DA>,
    b: &Tensor<B, DB>,
    f: F,
) -> Result<Tensor<C, <DA as BroadcastDim<DB>>::Output>, XenonError>
where
    DA: Dimension + BroadcastDim<DB>,
    DB: Dimension,
    <DA as BroadcastDim<DB>>::Output: Dimension,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> C + Sync,
{
    let _guard = match ParallelGuard::enter() {
        Some(g) => g,
        None => {
            // Already in parallel context, fall back to serial
            return a.zip_with(b, f);
        }
    };
    let broadcast_shape = broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let len: usize = broadcast_shape.iter().product();
    let is_f_contiguous = a.is_f_contiguous() && b.is_f_contiguous();

    if !should_parallelize(len, is_f_contiguous) {
        return Ok(a.zip_with(b, f)?);
    }

    let output: Vec<C> = ParZip::new(a.view(), b.view())?.map(|(x, y)| f(&x, &y)).collect();

    let output_dim = <DA as BroadcastDim<DB>>::Output::from_slice(&broadcast_shape);
    // SAFETY: output length matches dim
    unsafe { Ok(Tensor::from_raw_vec_unchecked(output, output_dim)) }
}
```

### 4.5 并行迭代器

```rust
// src/parallel/par_iter.rs

use rayon::iter::{ParallelIterator, IndexedParallelIterator};

/// Parallel element iterator
///
/// Iterates all logical elements in parallel following the same element order
/// contract as `iter::Elements`.
///
/// # Thread Safety
///
/// `A: Send + Sync` is required because elements are accessed from multiple threads:
/// - `Send`: elements can be transferred across thread boundaries
/// - `Sync`: shared references can be safely accessed from multiple threads
pub struct ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
}

impl<'a, A, D> ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    /// Create a parallel element iterator
    pub fn new(base: TensorView<'a, A, D>) -> Self {
        ParElements { base }
    }

    /// Get total element count
    pub fn len(&self) -> usize { self.base.len() }

    /// Check if empty
    pub fn is_empty(&self) -> bool { self.base.is_empty() }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParallelIterator for ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension + Send,
{
    type Item = &'a A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        // Splits data into chunks and drives parallel execution
        rayon::iter::plumbing::bridge(self, consumer)
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> IndexedParallelIterator for ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension + Send,
{
    fn len(&self) -> usize { self.base.len() }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        // Split the flat index range and map to elements
        callback.callback(ElementsProducer { base: self.base })
    }
}

/// Producer for ordered parallel element iteration.
///
/// Splits the logical flat index range into sub-ranges for parallel processing,
/// preserving the same observable element order as the serial iterator.
pub struct ElementsProducer<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::plumbing::Producer for ElementsProducer<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension + Send,
{
    type Item = &'a A;
    type IntoIter = crate::iter::Elements<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        // Elements::new constructs the iterator from a TensorView
        crate::iter::Elements::new(self.base)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // Split the view at the flat index boundary
// split_elements_at is defined in src/iter/mod.rs (see 10-iterator.md §5.5)
        let (left, right) = crate::iter::split_elements_at(self.base, index);
        (
            ElementsProducer { base: left },
            ElementsProducer { base: right },
        )
    }
}

/// Parallel multi-array synchronized iterator
///
/// Iterates two arrays simultaneously, yielding tuples element wise.
/// Supports broadcasting.
///
/// # Thread Safety
///
/// `A: Send + Sync` and `B: Send + Sync` are required because elements from both
/// arrays are accessed from multiple threads concurrently.
pub struct ParZip<'a, A, B, DA, DB>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    DA: Dimension,
    DB: Dimension,
{
    a: TensorView<'a, A, DA>,
    b: TensorView<'a, B, DB>,
    len: usize,
}

impl<'a, A, B, DA, DB> ParZip<'a, A, B, DA, DB>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    DA: Dimension,
    DB: Dimension,
{
    /// Create a parallel zip iterator
    pub fn new(
        a: TensorView<'a, A, DA>,
        b: TensorView<'a, B, DB>,
    ) -> Result<Self, XenonError> {
        let broadcast_shape = broadcast::broadcast_shape(&a.shape(), &b.shape())?;
        let len = broadcast_shape.iter().product();
        Ok(ParZip { a, b, len })
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

#[cfg(feature = "parallel")]
impl<'a, A, B, DA, DB> ParallelIterator for ParZip<'a, A, B, DA, DB>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    DA: Dimension + Send,
    DB: Dimension + Send,
{
    type Item = (&'a A, &'a B);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        // Splits data into chunks and drives parallel execution
        rayon::iter::plumbing::bridge(self, consumer)
    }
}
```

### 4.6 嵌套并行防护

```rust
// src/parallel/par_iter.rs (or src/tensor/impls.rs)

/// Parallel element iterator method on TensorBase.
///
/// This method is feature-gated and only available when the
/// `parallel` feature is enabled.
#[cfg(feature = "parallel")]
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Send + Sync,
{
    /// Returns a parallel element iterator.
    ///
    /// # Feature Gate
    ///
    /// Only available with `features = ["parallel"]`.
    pub fn par_iter(&self) -> ParElements<'_, A, D> {
        ParElements::new(self.view())
    }
}
```

### 4.7 嵌套并行防护（实现）

```rust
// src/parallel/mod.rs

/// Runtime guard for parallel execution.
///
/// Parallel entry points call `ParallelGuard::enter()` before spawning work.
/// If a guard is already active in the current execution context, nested
/// parallel APIs fall back to serial execution.
pub struct ParallelGuard;

impl ParallelGuard {
    /// Enter a parallel context.
    ///
    /// Returns `None` if the current execution context is already inside
    /// a parallel region.
    pub fn enter() -> Option<Self>;
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        // Releases the explicit parallel token.
    }
}
```

> **设计说明：** 不在公共 API 中显式暴露并行令牌参数。并行入口统一通过 `ParallelGuard::enter()` 建立运行时守卫；若检测到已处于并行区域，则内层操作自动回退串行，避免嵌套并行失效。

### 4.8 Good/Bad 对比示例

```rust
// Good - Use automatic path selection
let result = tensor.par_map(|x| x * 2.0);
// Small arrays automatically serial, large arrays automatically parallel

// Good - Use a custom threshold
let result = tensor.par_map_with_threshold(|x| x * 2.0, 8192);

// Good - Use serial operations inside a parallel context to avoid nesting
tensor.axis_iter(Axis(0)).for_each(|slice| {
    // Inner layer uses serial operations
    let sum: f64 = slice.iter().sum();
});

// Bad - Nested parallelism
tensor.axis_iter(Axis(0)).for_each(|slice| {
    let sum = slice.par_sum(); // Forbidden! Thread pool starvation
});

// Bad - Ignoring parallel errors
let result = par_zip_with(&a, &b, |x, y| x + y).unwrap(); // Forbidden silent ignore
```

---

## 5. 内部实现设计

### 5.1 分块策略

```
分块决策流程

┌─────────────────────────────────────────────────────────────────┐
│  输入: tensor (TensorBase<S, D>)                                 │
│                                                                 │
│  1. 检查连续性                                                    │
│     ├─ is_f_contiguous() ?                                      │
│     │   └─ 使用连续分块策略（参见 `06-memory.md §4.4`）      │
│     │       compute_contiguous_chunks(len, config)              │
│     │                                                           │
│     └─ 非连续                                                    │
│         └─ 沿第一轴分块                                           │
│             compute_strided_chunks(shape, strides, config)      │
│                                                                 │
│  2. 返回分块迭代器                                                │
│     └─ chunks.into_par_iter()                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 线程池管理

```rust
// src/parallel/mod.rs

/// Custom thread pool wrapper
#[cfg(feature = "parallel")]
pub struct ParallelPool {
    inner: rayon::ThreadPool,
}

#[cfg(feature = "parallel")]
impl ParallelPool {
    /// Create a new thread pool
    pub fn new(num_threads: usize) -> Result<Self, PoolInitError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PoolInitError::BuildFailed(e.to_string()))?;
        Ok(ParallelPool { inner: pool })
    }

    /// Execute a closure on this thread pool
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(op)
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}
```

### 5.3 并行错误传播

```rust
// Good - Unrecoverable errors in parallel operations are propagated immediately
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
        // Serial path: propagate errors naturally
        let mut output = Vec::with_capacity(tensor.len());
        for elem in tensor.iter() {
            output.push(f(elem)?);
        }
        // SAFETY: output length == tensor.len() (each element maps to exactly one output)
        return Ok(unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) });
    }

    // Parallel path: collect results, fail fast on first error
    let results: Vec<Result<B, XenonError>> = tensor
        .par_iter()
        .map(|x| f(x))
        .collect();

    // Check for errors
    let mut output = Vec::with_capacity(tensor.len());
    for result in results {
        output.push(result?);
    }
    // SAFETY: output length == tensor.len() (each successful result produces one output)
    Ok(unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) })
}

// Bad - Silently ignoring errors in parallel operations
pub fn par_map_silent<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    F: Fn(&A) -> Option<B> + Sync,
{
    // Forbidden: errors are silently swallowed
    let output: Vec<B> = tensor
        .par_iter()
        .filter_map(|x| f(x))  // Silently discards None
        .collect();
    // output length may not match the expected length!
    // SAFETY: BUG — length invariant is NOT upheld due to filter_map discarding elements
    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}
```

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/parallel/mod.rs` 骨架
  - 文件: `src/parallel/mod.rs`
  - 内容: feature gate 声明、rayon 集成、全局阈值 `AtomicUsize`、`should_parallelize()`、`ParallelGuard`、模块导出
  - 测试: 编译通过、`test_default_threshold`
  - 前置: tensor 模块完成
  - 预计: 10 min

### Wave 2: 迭代器与基础设施扩展

- [ ] **T2**: 创建 `src/parallel/par_iter.rs` 并行迭代器
  - 文件: `src/parallel/par_iter.rs`
  - 内容: `ParElements` 结构体和 `ParallelIterator` 实现、`ParZip` 结构体和 `ParallelIterator` 实现
  - 测试: `test_par_elements_len`、`test_par_zip_len`
  - 前置: T1
  - 预计: 10 min

- [ ] **T6**: 实现嵌套并行防护
  - 文件: `src/parallel/mod.rs`
  - 内容: `ParallelGuard` 显式令牌、串行回退检查、上下文传播辅助函数
  - 测试: `test_nested_parallel_guard`、`test_parallel_token_propagation`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7**: 线程池管理与配置
  - 文件: `src/parallel/mod.rs`
  - 内容: `ParallelPool` 封装、`configure_global_pool()`
  - 测试: `test_custom_pool`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 并行运算

- [ ] **T3**: 实现 `par_map` 和 `par_map_inplace`
  - 文件: `src/parallel/par_ops.rs`
  - 内容: `par_map()`、`par_map_inplace()`、自动阈值检查、串行回退
  - 测试: `test_par_map_result`、`test_par_map_fallback_small`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `par_reduce` 和 `par_sum`
  - 文件: `src/parallel/par_ops.rs`
  - 内容: `par_reduce()`、`par_sum()`、关联操作约束
  - 测试: `test_par_sum_correctness`、`test_par_reduce_empty`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现 `par_zip_with`
  - 文件: `src/parallel/par_ops.rs`
  - 内容: `par_zip_with()`、广播兼容性、并行 zip 实现
  - 测试: `test_par_zip_with_add`、`test_par_zip_broadcast`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 集成与测试

- [ ] **T8**: 集成测试与一致性验证
  - 文件: `tests/parallel_consistency.rs`
  - 内容: 并行与串行结果一致性测试、阈值行为测试、竞态条件检测
  - 测试: `test_par_sum_matches_serial`、`test_threshold_boundary`
  - 前置: T3, T4, T5
  - 预计: 10 min

```
Wave 1:        [T1]
                  │
                  ▼
Wave 2: [T2] [T6] [T7]
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
Wave 3: [T3]  [T4]  [T5]
           │     │     │
           └─────┼─────┘
                 │
                 ▼
Wave 4:        [T8]
```

---

## 7. 测试计划

### 7.1 测试分类表

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个并行操作 |
| 一致性测试 | `tests/parallel_consistency.rs` | 并行结果与串行一致 |
| 边界测试 | 集成测试中标注 | 阈值边界、空数组、单元素 |
| 并发测试 | `tests/concurrent/` | 竞态条件检测 |
| 属性测试 | `tests/property/` | 随机数据验证一致性不变量（参见 §7.4） |

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_default_threshold` | 默认阈值为 1024 | 高 |
| `test_set_get_threshold` | 设置和获取阈值一致 | 高 |
| `test_par_elements_len` | 并行迭代器长度正确 | 高 |
| `test_par_map_result` | 并行映射结果正确 | 高 |
| `test_par_map_fallback_small` | 小数组回退到串行 | 高 |
| `test_par_sum_correctness` | 并行求和结果正确 | 高 |
| `test_par_reduce_empty` | 空数组返回单位元 | 高 |
| `test_par_zip_with_add` | 并行 zip 加法正确 | 高 |
| `test_par_zip_broadcast` | 广播并行正确 | 中 |
| `test_nested_parallel_guard` | 嵌套并行被检测 | 中 |
| `test_parallel_token_propagation` | 并行上下文令牌传播正确 | 中 |
| `test_custom_pool` | 自定义线程池工作正常 | 低 |

### 7.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `len=0` | 并行操作立即返回，不 panic |
| 单元素 `len=1` | 回退到串行 |
| `len = threshold - 1` | 串行执行 |
| `len = threshold` | 并行执行 |
| `len = threshold + 1` | 并行执行 |
| 非连续数组 | 阈值翻倍后决定 |
| 嵌套并行 | 内层自动回退串行 |

### 7.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `par_sum() == iter().sum()` | 随机 `[f64; 0..100000]` |
| `par_map(f) == map(f)` | 随机 `[f32; 0..100000]` |
| `par_zip_with(f) == zip_with(f)` | 随机形状 |
| 并行结果与线程数无关 | 分别用 1, 2, 4, 8 线程验证 |

### 7.5 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/parallel.rs` | `par_map` / `par_sum` / `par_zip_with` 与 `tensor`、`storage`、`simd`、`rayon` guard 的端到端协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

并行路径的每个工作线程可以调用上层提供的分块执行器；若上层在该分块上启用 SIMD，则形成“并行 + SIMD”组合路径（参见 `08-simd.md §8.2`）。

```
并行 + SIMD 组合执行

┌─────────────────────────────────────────────────────────────────┐
│ par_map(tensor, f)                                              │
│   ├─ 分块: [0..N/4), [N/4..N/2), [N/2..3N/4), [3N/4..N)       │
│   │                                                             │
│   ├─ 线程 0: execute_chunk(chunk_0)                             │
│   │   └─ 具体使用 SIMD 还是标量，由上层 dispatch 决定             │
│   │                                                             │
│   ├─ 线程 1: execute_chunk(chunk_1)                             │
│   │   └─ 具体使用 SIMD 还是标量，由上层 dispatch 决定             │
│   │                                                             │
│   └─ ...                                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 数据流描述

```text
上层语义模块调用 par_map / par_sum / par_zip_with
    │
    ├── ParallelGuard::enter()
    │       └── 若已在并行区域 → 自动回退串行
    │
    ├── should_parallelize(len, is_f_contiguous)
    │       ├── 低于阈值 → 串行
    │       └── 达到阈值 → 并行分块
    │
    ├── rayon 对分块并行执行
    │
    └── 每个分块内部的具体执行内核（SIMD / 标量）由上层 dispatch 决定
```

### 8.3 与 storage 模块

并行迭代器通过 `RawStorage` trait 获取原始指针和长度信息（参见 `05-storage.md §4`）。

### 8.4 与 tensor 模块

并行操作消费 `Tensor<A, D>` 的视图，产出新的 `Tensor<A, D>`（参见 `07-tensor.md §4.5`）。

---

## 9. 设计决策记录（ADR）

### 决策 1：选择 rayon 作为并行框架

| 属性 | 值 |
|------|-----|
| 决策 | 使用 rayon crate 作为并行框架 |
| 理由 | 成熟稳定、work-stealing 调度器高效、`ParallelIterator` API 优雅、Rust 生态标准选择、无数据竞争保证 |
| 替代方案 | `crossbeam` — 放弃，更底层，需手动管理任务分发 |
| 替代方案 | `tokio` — 放弃，面向异步 I/O，不适合 CPU 密集型计算 |
| 替代方案 | 手动 `std::thread` — 放弃，代码复杂度高，错误容易引入 |

### 决策 2：并行归约浮点一致性

| 属性 | 值 |
|------|-----|
| 决策 | 所有类型的并行结果都必须与串行实现保持一致；若某条并行路径无法证明该性质，则自动回退串行。当前版本默认仅对可逐块证明一致的整数/逐元素路径启用并行归约，浮点归约保守回退串行。 |
| 理由 | 需求说明书 §28.5 已固定“并行归约结果须与单线程一致”；性能优化不能改变语义结果 |
| 测试约定 | 并行 sum / zip / map 的一致性测试使用与串行结果一致的断言，而非近似比较；参见 `28-tests.md §8.2` |
| 参见 | `13-reduction.md §9 ADR-2`（协调定义） |

### 决策 3：并行阈值默认 1024 元素

| 属性 | 值 |
|------|-----|
| 决策 | 默认并行阈值设为 1024 元素 |
| 理由 | 低于此值时线程调度和同步开销大于并行收益；可配置允许用户调优 |
| 替代方案 | 65536（64K）— 更保守，错过中等规模数组的并行加速 |
| 替代方案 | 256 — 更激进，小数组并行开销可能大于收益 |
| 替代方案 | 0（始终并行）— 放弃，小数组并行会严重降低性能 |

### 决策 4：嵌套并行回退到串行而非 panic

| 属性 | 值 |
|------|-----|
| 决策 | 检测到嵌套并行时自动回退到串行执行 |
| 理由 | 更宽容，不会中断用户程序；通过 `ParallelGuard` 运行时守卫允许调用链在检测到嵌套时回退串行 |
| 替代方案 | panic — 放弃，虽然能暴露问题，但可能破坏生产环境 |
| 替代方案 | 编译期禁止 — 放弃，Rust 类型系统难以在编译期检测嵌套并行 |

---

## 10. 性能考量

### 10.1 并行开销分析

| 开销来源 | 典型值 | 说明 |
|----------|--------|------|
| 线程调度 | ~1-10μs | rayon work-stealing 调度器 |
| 任务分发 | ~0.1-1μs | 分块和分发 |
| 缓存失效 | ~10-100μs | 大数组跨线程访问 |
| 同步屏障 | ~0.1-1μs | reduce 操作的合并步骤 |

### 10.2 最优阈值选取方法

```
性能 vs 元素数 关系图

速度
  │          ┌──────────────── 并行路径
  │         ╱
  │        ╱  交叉点 ≈ 1024
  │───────╱────────────────── 串行路径
  │      ╱
  │     ╱
  │    ╱
  └─────────────────────── 元素数
         ↑
      并行阈值
```

- 实际最优阈值依赖硬件配置，应通过 benchmark 确定
- 默认值 1024 是保守估计，适用于主流硬件
- 用户可通过 `set_parallel_threshold()` 调优

### 10.3 预期加速比

| 操作 | 元素数 | 4 核加速比 | 8 核加速比 |
|------|--------|-----------|-----------|
| par_map (f64) | 1M | ~3.5x | ~6x |
| par_sum (f64) | 1M | ~3x | ~5x |
| par_zip_with (f64) | 1M | ~3x | ~5.5x |

---

## 11. no_std 兼容性

`parallel` 模块**不兼容 `no_std`**。rayon 依赖标准库的线程原语。在 `no_std` 环境下，`parallel` feature 不可用。

```rust
// Conditional compilation: parallel requires std
#[cfg(all(feature = "parallel", not(feature = "std")))]
compile_error!("The 'parallel' feature requires the 'std' feature");
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
