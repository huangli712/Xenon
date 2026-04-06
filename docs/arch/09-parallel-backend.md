# 并行后端模块设计

> 文档编号: 09 | 模块: `src/parallel/` | 阶段: Phase 5
> 前置文档: `07-tensor.md`, `10-iterator.md`
> 需求参考: 需求说明书 §9.2, §9.3

---

## 1. 模块定位

### 1.1 概述

并行后端模块是 Xenon 张量库的可选性能加速层，通过 `rayon` crate 提供数据并行能力，为大规模数组操作提供多线程加速。该模块默认关闭，通过 `features = ["parallel"]` 启用。

### 1.2 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| rayon 集成 | 全局线程池、自定义线程池配置 | 自定义线程池调度策略 |
| 逐元素并行 | a + b 等逐元素运算的并行执行 | 嵌套并行 |
| 并行归约 | sum 的并行求和 | GPU 并行 |
| 函数映射并行 | apply / map_inplace 并行执行 | BLAS 并行绑定 |
| 多数组同步 | zip 多数组并行迭代 | 自动 DAG 调度 |
| 自动路径选择 | 根据数据规模自动串行/并行切换 | 编译期静态并行分发 |

### 1.3 设计原则

| 原则 | 体现 |
|------|------|
| 透明集成 | 用户代码无需修改，仅通过 feature gate 启用 |
| 自动决策 | 运行时根据数据规模和布局自动选择并行/串行 |
| 可配置性 | 支持全局配置和单次调用覆盖阈值 |
| 安全性 | 禁止嵌套并行，保证无数据竞争 |
| 兼容性 | 与 SIMD 模块协同工作，每线程内部使用 SIMD |

### 1.4 性能分层中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                      性能分层决策树                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  元素数 < SIMD_WIDTH (如 16)                                    │
│  └─→ 标量路径 (scalar)                                          │
│                                                                 │
│  元素数 ≥ SIMD_WIDTH 且 连续内存 且 simd 启用                    │
│  └─→ SIMD 路径 (vectorized)                                     │
│                                                                 │
│  元素数 ≥ PARALLEL_THRESHOLD (默认 1024) 且 parallel 启用        │
│  └─→ 并行路径                                                   │
│      ├─ 连续内存 + simd 启用: 分块后各块 SIMD                    │
│      ├─ 连续内存 + simd 禁用: 分块后各块标量                     │
│      └─ 非连续内存: 分块后各块标量                               │
│                                                                 │
│  默认情况                                                       │
│  └─→ 标量路径 (scalar)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
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
├── crate::layout             # LayoutFlags, Order
├── crate::element            # Element trait
├── crate::broadcast          # broadcast_shape() (zip 场景)
└── crate::simd               # 每线程内部 SIMD 加速（可选）
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `rayon` | `ThreadPool`, `ThreadPoolBuilder`, `ParallelIterator`, `IndexedParallelIterator` |
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.view()`, `.len()` |
| `storage` | `RawStorage`, `Storage`, `StorageMut`, `.as_slice()` |
| `layout` | `LayoutFlags`, `is_f_contiguous()` |
| `broadcast` | `broadcast_shape()` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `parallel/` 仅消费 `tensor`、`storage`、`layout` 等核心模块，不被它们依赖。`parallel/` 模块在未启用 feature 时完全不存在。

---

## 4. 公共 API 设计

### 4.1 Xenon 并行约束

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
parallel = ["dep:rayon"]

[dependencies]
rayon = { version = "1.10", optional = true }
```

- 默认关闭，通过 `features = ["parallel"]` 显式启用
- 启用后 rayon 自动引入，提供数据并行能力

### 4.2 并行阈值系统

```rust
// src/parallel/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

/// 默认并行阈值：1024 元素。
///
/// 当数组元素数达到或超过此值时，启用并行执行。
/// 此值经过经验调优，在小数组上并行开销大于收益。
const DEFAULT_PARALLEL_THRESHOLD: usize = 1024;

/// 全局并行阈值。
static GLOBAL_PARALLEL_THRESHOLD: AtomicUsize =
    AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// 获取当前全局并行阈值
#[inline]
pub fn get_parallel_threshold() -> usize {
    GLOBAL_PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

/// 设置全局并行阈值
///
/// # 注意
///
/// 此设置影响所有后续并行操作。
/// 建议在程序启动时设置，避免运行时频繁修改。
pub fn set_parallel_threshold(threshold: usize) {
    GLOBAL_PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// 重置全局并行阈值为默认值
pub fn reset_parallel_threshold() {
    GLOBAL_PARALLEL_THRESHOLD.store(DEFAULT_PARALLEL_THRESHOLD, Ordering::Relaxed);
}
```

> **设计决策：** 使用 `Ordering::Relaxed`。阈值读取不需要与其他操作同步，稍旧的值也可接受。阈值修改通常是启动时一次性操作。

### 4.3 自动路径选择

```rust
// src/parallel/mod.rs

/// 判断是否应该启用并行执行
///
/// # 决策逻辑
///
/// 1. 检查 `parallel` feature 是否启用
/// 2. 检查元素数是否达到阈值
/// 3. 考虑数据布局（非连续数组需更高阈值）
///
/// # 参数
///
/// * `len` - 元素数量
/// * `is_contiguous` - 数据是否连续
///
/// # 返回
///
/// `true` 表示应该启用并行执行。
#[cfg(feature = "parallel")]
pub fn should_parallelize(len: usize, is_contiguous: bool) -> bool {
    let threshold = get_parallel_threshold();
    let meets_threshold = if is_contiguous {
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

/// 并行映射运算
///
/// # 约束
///
/// * `F: Fn(&A) -> B + Sync` - 函数必须可跨线程共享
/// * `A: Send + Sync` - 输入元素必须线程安全
/// * `B: Send` - 输出元素必须可跨线程发送
///
/// # 自动决策
///
/// 如果元素数低于并行阈值，自动回退到串行 `map`。
#[cfg(feature = "parallel")]
pub fn par_map<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync,
{
    if !should_parallelize(tensor.len(), tensor.is_contiguous()) {
        return tensor.map(&f);
    }

    let len = tensor.len();
    let mut output: Vec<B> = Vec::with_capacity(len);
    output.par_extend(tensor.par_iter().map(|x| f(x)));

    // SAFETY: output length == tensor.len()
    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}

/// 并行归约
///
/// # 约束
///
/// * 归约函数必须是关联的
/// * 初始值函数必须返回单位元
#[cfg(feature = "parallel")]
pub fn par_reduce<A, D, F, ID>(tensor: &Tensor<A, D>, identity: ID, op: F) -> A
where
    D: Dimension,
    A: Element + Send + Sync,
    F: Fn(A, A) -> A + Sync,
    ID: Fn() -> A + Sync + Clone,
{
    if tensor.is_empty() {
        return identity();
    }
    if !should_parallelize(tensor.len(), tensor.is_contiguous()) {
        return tensor.iter().fold(identity(), |acc, x| op(acc, *x));
    }

    tensor.par_iter().cloned().reduce(identity, op)
}

/// 并行求和
#[cfg(feature = "parallel")]
pub fn par_sum<A, D>(tensor: &Tensor<A, D>) -> A
where
    D: Dimension,
    A: Element + Numeric + Send + Sync,
{
    par_reduce(tensor, || A::zero(), |a, b| a + b)
}

/// 并行 zip 运算
///
/// # 约束
///
/// * 形状必须兼容（相同或可广播）
/// * 支持广播
#[cfg(feature = "parallel")]
pub fn par_zip_with<A, B, C, DA, DB, F>(
    a: &Tensor<A, DA>,
    b: &Tensor<B, DB>,
    f: F,
) -> Result<Tensor<C, DA>, BroadcastError>
where
    DA: Dimension,
    DB: Dimension,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> C + Sync,
{
    let broadcast_shape = broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let len: usize = broadcast_shape.iter().product();
    let is_contiguous = a.is_contiguous() && b.is_contiguous();

    if !should_parallelize(len, is_contiguous) {
        return Ok(a.zip_with(b, &f)?);
    }

    let mut output: Vec<C> = Vec::with_capacity(len);
    let zip = ParZip::new(a.view(), b.view())?;
    output.par_extend(zip.map(|(x, y)| f(&x, &y)));

    let dim = DA::from_slice(&broadcast_shape)?;
    // SAFETY: output length matches dim
    unsafe { Ok(Tensor::from_raw_vec_unchecked(output, dim)) }
}
```

### 4.5 并行迭代器

```rust
// src/parallel/par_iter.rs

use rayon::iter::{ParallelIterator, IndexedParallelIterator};

/// 并行元素迭代器
///
/// 按内存布局顺序并行迭代所有元素。
pub struct ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
}

impl<'a, A, D> ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    /// 创建并行元素迭代器
    pub fn new(base: TensorView<'a, A, D>) -> Self {
        ParElements { base }
    }

    /// 获取元素总数
    pub fn len(&self) -> usize { self.base.len() }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool { self.base.is_empty() }
}

/// 并行多数组同步迭代器
///
/// 同时迭代两个数组，按元素产出元组。
/// 支持广播。
pub struct ParZip<'a, A, B, DA, DB>
where
    A: Element + Sync,
    B: Element + Sync,
    DA: Dimension,
    DB: Dimension,
{
    a: TensorView<'a, A, DA>,
    b: TensorView<'a, B, DB>,
    len: usize,
}

impl<'a, A, B, DA, DB> ParZip<'a, A, B, DA, DB>
where
    A: Element + Sync,
    B: Element + Sync,
    DA: Dimension,
    DB: Dimension,
{
    /// 创建并行 zip 迭代器
    pub fn new(
        a: TensorView<'a, A, DA>,
        b: TensorView<'a, B, DB>,
    ) -> Result<Self, BroadcastError> {
        let broadcast_shape = broadcast::broadcast_shape(&a.shape(), &b.shape())?;
        let len = broadcast_shape.iter().product();
        Ok(ParZip { a, b, len })
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}
```

### 4.6 嵌套并行防护

```rust
// src/parallel/mod.rs

use std::cell::Cell;

thread_local! {
    /// 并行执行深度计数器
    ///
    /// 0: 未在并行上下文中
    /// 1: 在外层并行上下文中
    /// >1: 嵌套并行（应禁止）
    static PARALLEL_DEPTH: Cell<usize> = Cell::new(0);
}

/// 检查是否在并行上下文中
#[inline]
pub fn is_in_parallel_context() -> bool {
    PARALLEL_DEPTH.with(|cell| cell.get() > 0)
}

/// 并行上下文守卫（RAII）
///
/// 进入时增加深度，离开时自动减少。
/// 检测到嵌套时自动回退到串行。
pub struct ParallelGuard(());

impl ParallelGuard {
    /// 进入并行上下文
    pub fn enter() -> Self {
        PARALLEL_DEPTH.with(|cell| {
            let depth = cell.get();
            cell.set(depth + 1);
        });
        ParallelGuard(())
    }
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        PARALLEL_DEPTH.with(|cell| {
            let depth = cell.get();
            cell.set(depth.saturating_sub(1));
        });
    }
}
```

### 4.7 Good/Bad 对比示例

```rust
// Good - 使用自动路径选择
let result = tensor.par_map(|x| x * 2.0);
// 小数组自动串行，大数组自动并行

// Good - 使用自定义阈值
let result = tensor.par_map_with_threshold(|x| x * 2.0, 8192);

// Good - 在并行上下文中使用串行操作避免嵌套
tensor.par_axis_iter(Axis(0)).for_each(|slice| {
    // 内层使用串行操作
    let sum: f64 = slice.iter().sum();
});

// Bad - 嵌套并行
tensor.par_axis_iter(Axis(0)).for_each(|slice| {
    let sum = slice.par_sum(); // 禁止！线程池饥饿
});

// Bad - 忽略并行错误
let result = par_zip_with(&a, &b, |x, y| x + y).unwrap(); // 禁止静默忽略
```

---

## 5. 内部实现设计

### 5.1 分块策略

```
分块决策流程

┌─────────────────────────────────────────────────────────────────┐
│  输入: tensor (TensorBase<S, D>)                                │
│                                                                 │
│  1. 检查连续性                                                   │
│     ├─ is_f_contiguous() ?                                      │
│     │   └─ 使用连续分块策略                                      │
│     │       compute_contiguous_chunks(len, config)               │
│     │                                                           │
│     └─ 非连续                                                   │
│         └─ 沿第一轴分块                                          │
│             compute_strided_chunks(shape, strides, config)       │
│                                                                 │
│  2. 返回分块迭代器                                               │
│     └─ chunks.into_par_iter()                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 线程池管理

```rust
// src/parallel/mod.rs

/// 自定义线程池包装
pub struct ParallelPool {
    inner: rayon::ThreadPool,
}

impl ParallelPool {
    /// 创建新的线程池
    pub fn new(num_threads: usize) -> Result<Self, PoolInitError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PoolInitError::BuildFailed(e.to_string()))?;
        Ok(ParallelPool { inner: pool })
    }

    /// 在此线程池上执行闭包
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(op)
    }

    /// 获取线程数
    pub fn num_threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}
```

### 5.3 并行错误传播

```rust
// Good - 并行操作中不可恢复错误立即传播
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
    if !should_parallelize(tensor.len(), tensor.is_contiguous()) {
        // Serial path: propagate errors naturally
        let mut output = Vec::with_capacity(tensor.len());
        for elem in tensor.iter() {
            output.push(f(elem)?);
        }
        return Ok(Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()));
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
    Ok(Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()))
}

// Bad - 静默忽略并行操作中的错误
pub fn par_map_silent<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    F: Fn(&A) -> Option<B> + Sync,
{
    // 禁止：错误被静默吞掉
    let output: Vec<B> = tensor
        .par_iter()
        .filter_map(|x| f(x))  // 静默丢弃 None
        .collect();
    // output 长度可能与预期不符！
    Tensor::from_raw_vec_unchecked(output, tensor.raw_dim())
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

- [ ] **T2**: 创建 `src/parallel/par_iter.rs` 并行迭代器
  - 文件: `src/parallel/par_iter.rs`
  - 内容: `ParElements` 结构体和 `ParallelIterator` 实现、`ParZip` 结构体和 `ParallelIterator` 实现
  - 测试: `test_par_elements_len`、`test_par_zip_len`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 并行运算

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

### Wave 3: 集成与测试

- [ ] **T6**: 实现嵌套并行防护
  - 文件: `src/parallel/mod.rs`
  - 内容: `PARALLEL_DEPTH` thread_local、`ParallelGuard` RAII 守卫、`is_in_parallel_context()`
  - 测试: `test_nested_parallel_guard`、`test_parallel_depth_tracking`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7**: 线程池管理与配置
  - 文件: `src/parallel/mod.rs`
  - 内容: `ParallelPool` 封装、`configure_global_pool()`
  - 测试: `test_custom_pool`
  - 前置: T1
  - 预计: 10 min

- [ ] **T8**: 集成测试与一致性验证
  - 文件: `tests/parallel_consistency.rs`
  - 内容: 并行与串行结果一致性测试、阈值行为测试、竞态条件检测
  - 测试: `test_par_sum_matches_serial`、`test_threshold_boundary`
  - 前置: T3, T4, T5
  - 预计: 10 min

```
Wave 1: [T1]
           │
           ▼
Wave 2: [T2] [T6] [T7]
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
   [T3]  [T4]  [T5]
     │     │     │
     └─────┼─────┘
           │
           ▼
Wave 3:  [T8]
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
| `test_parallel_depth_tracking` | 深度计数正确 | 中 |
| `test_custom_pool` | 自定义线程池工作正常 | 低 |

### 7.3 边界测试场景表

| 场景 | 预期行为 |
|------|----------|
| 空数组 `len=0` | 并行操作立即返回，不 panic |
| 单元素 `len=1` | 回退到串行 |
| `len = threshold - 1` | 串行执行 |
| `len = threshold` | 并行执行 |
| `len = threshold + 1` | 并行执行 |
| 非连续数组 | 阈值翻倍后决定 |
| 嵌套并行 | 内层自动回退串行 |

### 7.4 一致性不变量

| 不变量 | 测试方法 |
|--------|----------|
| `par_sum() == iter().sum()` | 随机 `[f64; 0..100000]` |
| `par_map(f) == map(f)` | 随机 `[f32; 0..100000]` |
| `par_zip_with(f) == zip_with(f)` | 随机形状 |
| 并行结果与线程数无关 | 分别用 1, 2, 4, 8 线程验证 |

---

## 8. 与其他模块的交互

### 8.1 与 simd 模块

并行路径的每个工作线程内部可以使用 SIMD。组合使用时：

```
并行 + SIMD 组合执行

┌─────────────────────────────────────────────────────────────────┐
│ par_map(tensor, f)                                              │
│   ├─ 分块: [0..N/4), [N/4..N/2), [N/2..3N/4), [3N/4..N)       │
│   │                                                             │
│   ├─ 线程 0: f(chunk_0)                                         │
│   │   └─ 内部使用 VectorKernel::add (SIMD)                      │
│   │                                                             │
│   ├─ 线程 1: f(chunk_1)                                         │
│   │   └─ 内部使用 VectorKernel::add (SIMD)                      │
│   │                                                             │
│   └─ ...                                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 与 storage 模块

并行迭代器通过 `RawStorage` trait 获取原始指针和长度信息。

### 8.3 与 tensor 模块

并行操作消费 `Tensor<A, D>` 的视图，产出新的 `Tensor<A, D>`。

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

### 决策 2：并行阈值默认 1024 元素

| 属性 | 值 |
|------|-----|
| 决策 | 默认并行阈值设为 1024 元素 |
| 理由 | 低于此值时线程调度和同步开销大于并行收益；可配置允许用户调优 |
| 替代方案 | 65536（64K）— 更保守，错过中等规模数组的并行加速 |
| 替代方案 | 256 — 更激进，小数组并行开销可能大于收益 |
| 替代方案 | 0（始终并行）— 放弃，小数组并行会严重降低性能 |

### 决策 3：嵌套并行回退到串行而非 panic

| 属性 | 值 |
|------|-----|
| 决策 | 检测到嵌套并行时自动回退到串行执行 |
| 理由 | 更宽容，不会中断用户程序；通过 `is_in_parallel_context()` 允许用户自行优化 |
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
// 条件编译：parallel 需要 std
#[cfg(all(feature = "parallel", not(feature = "std")))]
compile_error!("The 'parallel' feature requires the 'std' feature");
```

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
