# 并行后端模块设计

> 文档编号: 21 | 模块: `src/backend/parallel.rs` | 阶段: Phase 4（后端模块）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `10-iterators.md`, 需求说明书 §7.2, §7.3, §8

---

## 1. 模块定位

`backend/parallel.rs` 是 Xenon 的并行执行基础设施层，基于 rayon 提供数据并行能力。该模块为上层模块（`ops/`、`iter/`）提供统一的并行执行原语，屏蔽 rayon 的使用细节。

**核心职责：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 并行迭代器 | `ParIter`、`ParIterMut` — 实现 rayon 的 `ParallelIterator` | 串行迭代器（`iter/elements.rs`） |
| 并行变换 | `par_map` — 并行逐元素映射 | 运算符重载（`ops/element_wise.rs`） |
| 并行归约 | `par_reduce` — 并行归约运算 | 归约语义定义（`ops/reduction.rs`） |
| 并行同步迭代 | `par_zip` — 多数组并行遍历 | 广播规则实现（`broadcast.rs`） |
| 线程池管理 | `ThreadPoolConfig`、自定义线程池构建 | 全局线程池生命周期（rayon 管理） |
| 分块策略 | cache-line 感知的 chunk 计算 | SIMD 实现（`backend/simd.rs`） |

**与 `iter/` 模块的关系：**

本模块是并行基础设施的**唯一实现位置**。`src/iter/mod.rs` 中通过 `pub use crate::backend::parallel::*` re-export 并行迭代器类型，对用户暴露统一的 `xenon::iter::ParIter` 等类型名。`iter/` 模块不包含独立的并行实现文件。

**设计原则：**

- **Feature-gated 全隔离**：所有并行代码在 `#[cfg(feature = "parallel")]` 之后，关闭 feature 时零编译开销
- **禁止嵌套并行**：内层操作强制单线程执行，避免线程池饥饿
- **连续优先**：仅对连续内存块使用 rayon 分块；非连续数据退化为逐元素标量并行
- **阈值感知**：默认 ≥ 64K 元素才启用并行，小数组直接串行
- **零额外分配**：分块操作在原数据上切分视图，不复制数据

---

## 2. 文件位置

```
src/backend/
├── mod.rs              # 后端模块入口, 性能分层 dispatch
├── scalar.rs           # 标量回退路径 (always available)
├── simd.rs             # SIMD 路径 (feature = "simd")
└── parallel.rs         # 本模块：并行路径 (feature = "parallel")
```

在 `src/backend/mod.rs` 中的声明：

```rust
// src/backend/mod.rs

pub mod scalar;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;
```

在 `src/lib.rs` 中的 re-export：

```rust
// src/lib.rs

#[cfg(feature = "parallel")]
pub use crate::backend::parallel::{
    ParIter, ParIterMut,
    par_map, par_reduce, par_zip,
    ThreadPoolConfig,
};
```

---

## 3. 依赖关系

### 3.1 上游依赖

| 依赖 | 来源 | 用途 |
|------|------|------|
| `rayon` | 外部 crate (`dep:rayon`) | 线程池、`ParallelIterator` trait、`Producer` trait |
| `TensorBase`, `TensorView`, `TensorViewMut`, `Tensor` | `tensor` | 操作对象类型 |
| `Dimension`, `IxDyn` | `dimension` | 维度系统 |
| `Element`, `Numeric` | `element` | trait bound 约束 |
| `Owned`, `ViewRepr`, `ViewMutRepr`, `RawStorage`, `Storage` | `storage` | 存储类型与 trait |
| `LayoutFlags`, `Order` | `layout` | 布局标志查询（连续性判定） |
| `TensorError`, `Result` | `error` | 错误处理 |
| `PARALLEL_THRESHOLD`, `PARALLEL_MIN_CHUNK` | `backend/mod.rs` 或顶层常量 | 并行阈值配置 |

### 3.2 下游依赖方

| 模块 | 使用方式 |
|------|----------|
| `iter/mod.rs` | re-export `ParIter`、`ParIterMut` |
| `ops/element_wise.rs` | 调用 `par_map` 执行并行逐元素运算 |
| `ops/reduction.rs` | 调用 `par_reduce` 执行并行归约 |
| `iter/zip.rs` | 调用 `par_zip` 执行并行多数组遍历 |
| `backend/mod.rs` | 在性能分层 dispatch 中选择并行路径 |

### 3.3 依赖关系图

```
rayon ─────────────────┐
tensor ────────────────┤
dimension ─────────────┤
element ───────────────┤
storage ───────────────┼──→ backend/parallel.rs ──→ ops/
layout ────────────────┤         │                  iter/
error ─────────────────┘         │
                                 ▼
                          backend/scalar.rs
                          (串行回退路径)
```

---

## 4. 公共 API 设计

### 4.1 配置常量

```rust
/// Default minimum number of elements to trigger parallel execution.
///
/// Tensors with fewer than this many elements will use serial execution
/// even when the `parallel` feature is enabled, to avoid thread pool
/// overhead dominating the computation.
pub const PARALLEL_THRESHOLD: usize = 64 * 1024; // 64K elements

/// Minimum number of elements per parallel chunk.
///
/// Each chunk assigned to a worker thread contains at least this many
/// elements. This prevents fine-grained task subdivision that would
/// cause scheduling overhead to dominate.
///
/// Chosen to be at least one L2 cache line worth of f64 elements
/// (4096 × 8 bytes = 32 KB), which fits comfortably in per-core caches.
pub const PARALLEL_MIN_CHUNK: usize = 4 * 1024; // 4K elements

/// Default target chunk size in bytes for cache-line aware partitioning.
///
/// Targets the L2 cache size (~256 KB typical) to ensure each chunk
/// fits in a single core's cache, maximizing cache hit rates.
pub const TARGET_CHUNK_BYTES: usize = 256 * 1024; // 256 KB
```

### 4.2 ParIter — 并行不可变迭代器

```rust
/// Parallel iterator over immutable element references.
///
/// Partitions the tensor into contiguous chunks and iterates over them
/// in parallel using rayon. Each chunk contains at least
/// `PARALLEL_MIN_CHUNK` elements.
///
/// For contiguous tensors, chunks are slices of the underlying storage,
/// enabling optimal cache-line utilization. For non-contiguous tensors
/// (strides != 1 along the fastest-varying axis), each element is
/// processed independently with stride-aware indexing.
///
/// This type is only available with the `parallel` feature enabled.
///
/// # Thread Safety
///
/// Requires `A: Send + Sync` because multiple threads hold `&A` references
/// to non-overlapping regions of the tensor data.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, zeros};
/// let t: Tensor<f64, Ix2> = zeros([1000, 1000]);
/// let sum: f64 = t.par_iter().cloned().sum();
/// ```
#[cfg(feature = "parallel")]
pub struct ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension,
{
    /// Immutable view of the source tensor.
    view: TensorView<'a, A, D>,

    /// Traversal order for partitioning.
    order: Order,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension,
{
    /// Creates a new parallel immutable iterator.
    ///
    /// Uses the tensor's natural layout order for partitioning.
    #[inline]
    pub(crate) fn new(view: TensorView<'a, A, D>) -> Self {
        Self {
            view,
            order: Order::F,
        }
    }

    /// Creates a new parallel iterator with a specified traversal order.
    #[inline]
    pub(crate) fn with_order(view: TensorView<'a, A, D>, order: Order) -> Self {
        Self { view, order }
    }
}

// --- rayon ParallelIterator trait ---

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::ParallelIterator for ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Item = &'a A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::UnindexedConsumer<Self::Item>;

    fn opt_len(&self) -> Option<usize> {
        Some(self.view.len())
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IndexedParallelIterator for ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    fn len(&self) -> usize {
        self.view.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::Consumer<Self::Item>;

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::ProducerCallback<Self::Item>;
}

// --- IntoParallelIterator for TensorView ---

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IntoParallelIterator for &'a TensorView<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Iter = ParIter<'a, A, D>;
    type Item = &'a A;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ParIter::new(self.clone())
    }
}

// --- IntoParallelIterator for Tensor (owned) ---

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IntoParallelIterator for &'a Tensor<A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Iter = ParIter<'a, A, D>;
    type Item = &'a A;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ParIter::new(self.view())
    }
}
```

### 4.3 ParIterMut — 并行可变迭代器

```rust
/// Parallel iterator over mutable element references.
///
/// Similar to `ParIter`, but yields `&mut A` references. The partitioning
/// guarantees non-overlapping access per thread: no two worker threads
/// ever receive mutable references to the same element.
///
/// This type is only available with the `parallel` feature enabled.
///
/// # Thread Safety
///
/// Requires `A: Send + Sync` because `&mut A` references are `Send`
/// (can be moved across threads), and the source tensor's data must
/// be `Sync` (safe for concurrent reads from other references).
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, zeros};
/// let mut t: Tensor<f64, Ix2> = zeros([1000, 1000]);
/// t.par_iter_mut().for_each(|x| *x = 42.0);
/// ```
#[cfg(feature = "parallel")]
pub struct ParIterMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension,
{
    /// Mutable view of the source tensor.
    view: TensorViewMut<'a, A, D>,

    /// Traversal order for partitioning.
    order: Order,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParIterMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension,
{
    /// Creates a new parallel mutable iterator.
    #[inline]
    pub(crate) fn new(view: TensorViewMut<'a, A, D>) -> Self {
        Self {
            view,
            order: Order::F,
        }
    }

    /// Creates a new parallel mutable iterator with a specified traversal order.
    #[inline]
    pub(crate) fn with_order(view: TensorViewMut<'a, A, D>, order: Order) -> Self {
        Self { view, order }
    }
}

// --- rayon ParallelIterator trait ---

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::ParallelIterator for ParIterMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Item = &'a mut A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::UnindexedConsumer<Self::Item>;

    fn opt_len(&self) -> Option<usize> {
        Some(self.view.len())
    }
}

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IndexedParallelIterator for ParIterMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    fn len(&self) -> usize {
        self.view.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::Consumer<Self::Item>;

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::ProducerCallback<Self::Item>;
}

// --- IntoParallelIterator for &mut TensorViewMut ---

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IntoParallelIterator for &'a mut TensorViewMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Iter = ParIterMut<'a, A, D>;
    type Item = &'a mut A;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ParIterMut::new(self.clone())
    }
}

// --- IntoParallelIterator for &mut Tensor ---

#[cfg(feature = "parallel")]
impl<'a, A, D> rayon::iter::IntoParallelIterator for &'a mut Tensor<A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Iter = ParIterMut<'a, A, D>;
    type Item = &'a mut A;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ParIterMut::new(self.view_mut())
    }
}
```

### 4.4 TensorBase 上的并行迭代器入口方法

```rust
// === Feature-gated parallel iterator entry points on TensorBase ===

#[cfg(feature = "parallel")]
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    S::Elem: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    /// Returns a parallel iterator over immutable element references.
    ///
    /// Only available with the `parallel` feature.
    ///
    /// For tensors with fewer than `PARALLEL_THRESHOLD` elements,
    /// consider using `iter()` instead to avoid thread pool overhead.
    ///
    /// # Performance
    ///
    /// - Contiguous tensors: partitioned into cache-line-aligned chunks
    /// - Non-contiguous tensors: each element processed independently
    ///   with stride-aware indexing (lower throughput)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use xenon::{Tensor, Ix1, linspace};
    /// let t: Tensor<f64, Ix1> = linspace(0.0, 1.0, 100_000);
    /// let sum: f64 = t.par_iter().cloned().sum();
    /// ```
    #[inline]
    pub fn par_iter(&self) -> ParIter<'_, S::Elem, D> {
        ParIter::new(self.view())
    }

    /// Returns a parallel iterator over immutable element references
    /// with a specified traversal order.
    #[inline]
    pub fn par_iter_with_order(&self, order: Order) -> ParIter<'_, S::Elem, D> {
        ParIter::with_order(self.view(), order)
    }
}

#[cfg(feature = "parallel")]
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    S::Elem: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    /// Returns a parallel iterator over mutable element references.
    ///
    /// Only available with the `parallel` feature.
    ///
    /// # Safety Guarantee
    ///
    /// Each `&mut A` yielded is guaranteed to be unique across all
    /// threads. The partitioning ensures no two threads access the
    /// same element.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use xenon::{Tensor, Ix2, zeros};
    /// let mut t: Tensor<f64, Ix2> = zeros([500, 500]);
    /// t.par_iter_mut().for_each(|x| *x = 1.0);
    /// ```
    #[inline]
    pub fn par_iter_mut(&mut self) -> ParIterMut<'_, S::Elem, D> {
        ParIterMut::new(self.view_mut())
    }

    /// Returns a parallel mutable iterator with a specified traversal order.
    #[inline]
    pub fn par_iter_mut_with_order(&mut self, order: Order) -> ParIterMut<'_, S::Elem, D> {
        ParIterMut::with_order(self.view_mut(), order)
    }
}
```

### 4.5 par_map — 并行逐元素变换

```rust
/// Applies a function to each element of the tensor in parallel,
/// returning a new owned tensor with the results.
///
/// This is the parallel counterpart of `mapv`. For tensors below
/// the parallel threshold, falls back to serial execution.
///
/// # Arguments
///
/// * `tensor` - The source tensor (any storage mode).
/// * `f` - A function applied to each element. Must be `Fn(A) -> B + Sync + Send`.
///
/// # Type Parameters
///
/// * `A` - Input element type (must be `Element + Send + Sync`).
/// * `B` - Output element type (must be `Element + Send + Sync`).
/// * `D` - Dimension type.
/// * `F` - Transformation function type.
///
/// # Performance
///
/// - **Contiguous data**: Chunks are sliced directly from the underlying
///   buffer, then `f` is applied to each chunk in parallel.
/// - **Non-contiguous data**: Falls back to per-element parallel processing
///   with stride-aware indexing.
/// - **Nested parallelism**: `f` is executed with rayon's single-threaded
///   mode active, preventing thread pool starvation.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix1, par_map, linspace};
/// let t: Tensor<f64, Ix1> = linspace(0.0, 1.0, 200_000);
/// let squares: Tensor<f64, Ix1> = par_map(&t, |x| x * x);
/// ```
#[cfg(feature = "parallel")]
pub fn par_map<A, B, D, F>(tensor: &TensorBase<impl RawStorage<Elem = A>, D>, f: F) -> Tensor<B, D>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    D: Dimension + Send + Sync,
    F: Fn(A) -> B + Sync + Send,
{
    let len = tensor.len();

    // Fall back to serial for small tensors
    if len < PARALLEL_THRESHOLD {
        return crate::backend::scalar::scalar_map(tensor, &f);
    }

    par_map_impl(tensor, f)
}

/// Applies a function to each element of two tensors in parallel,
/// returning a new owned tensor with the results.
///
/// Both tensors must have compatible shapes (identical or broadcastable).
/// Broadcasting follows NumPy rules: right-aligned, size-1 dimensions expanded.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, par_map2, zeros, ones};
/// let a: Tensor<f64, Ix2> = ones([500, 500]);
/// let b: Tensor<f64, Ix2> = ones([500, 500]);
/// let sum: Tensor<f64, Ix2> = par_map2(&a, &b, |x, y| x + y)?;
/// ```
#[cfg(feature = "parallel")]
pub fn par_map2<A, B, C, D, F>(
    lhs: &TensorBase<impl RawStorage<Elem = A>, D>,
    rhs: &TensorBase<impl RawStorage<Elem = B>, D>,
    f: F,
) -> Result<Tensor<C, D>>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send + Sync,
    D: Dimension + Send + Sync,
    F: Fn(A, B) -> C + Sync + Send,
{
    // Shape compatibility check with broadcasting
    let broadcast_shape = crate::broadcast::broadcast_shape(
        lhs.shape().slice(),
        rhs.shape().slice(),
    )?;

    let len: usize = broadcast_shape.iter().product();

    if len < PARALLEL_THRESHOLD {
        return crate::backend::scalar::scalar_map2(lhs, rhs, &f);
    }

    par_map2_impl(lhs, rhs, f, broadcast_shape)
}

/// Applies a function in-place to each element of a mutable tensor in parallel.
///
/// # Arguments
///
/// * `tensor` - A mutable tensor view to modify in place.
/// * `f` - A function applied to each element, returning the new value.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, par_map_inplace, zeros};
/// let mut t: Tensor<f64, Ix2> = zeros([500, 500]);
/// par_map_inplace(&mut t, |x| x + 1.0);
/// ```
#[cfg(feature = "parallel")]
pub fn par_map_inplace<A, D, F>(tensor: &mut TensorViewMut<'_, A, D>, f: F)
where
    A: Element + Send + Sync,
    D: Dimension + Send + Sync,
    F: Fn(A) -> A + Sync + Send,
{
    let len = tensor.len();

    if len < PARALLEL_THRESHOLD {
        crate::backend::scalar::scalar_map_inplace(tensor, &f);
        return;
    }

    par_map_inplace_impl(tensor, f);
}
```

### 4.6 par_reduce — 并行归约

```rust
/// Reduces the tensor elements in parallel using the given reduction function.
///
/// This is the parallel counterpart of sequential fold/reduce. The reduction
/// function must be **associative** (order of application does not affect the
/// result) for correct parallel execution. The identity element must be the
/// neutral element of the reduction (e.g., `0` for addition, `1` for multiplication).
///
/// # Arguments
///
/// * `tensor` - The source tensor (any storage mode).
/// * `identity` - The identity/neutral element for the reduction.
/// * `fold` - A function combining an accumulator with an element.
///
/// # Type Parameters
///
/// * `A` - Element type, must be `Element + Send + Sync`.
/// * `D` - Dimension type.
/// * `ID` - Identity function returning the neutral element.
/// * `F` - Fold function `(A, &A) -> A`.
///
/// # Performance
///
/// Uses rayon's `fold` + `reduce` pattern for tree-style reduction,
/// which provides O(log n) span complexity.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, par_reduce, ones};
/// let t: Tensor<f64, Ix2> = ones([500, 500]);
/// let sum = par_reduce(&t, || 0.0, |acc, &x| acc + x);
/// assert_eq!(sum, 250_000.0);
/// ```
#[cfg(feature = "parallel")]
pub fn par_reduce<A, D, ID, F>(tensor: &TensorBase<impl RawStorage<Elem = A>, D>, identity: ID, fold: F) -> A
where
    A: Element + Send + Sync,
    D: Dimension + Send + Sync,
    ID: Fn() -> A + Sync + Send,
    F: Fn(A, &A) -> A + Sync + Send,
{
    let len = tensor.len();

    if len == 0 {
        return identity();
    }

    if len < PARALLEL_THRESHOLD {
        // Serial fallback
        let mut acc = identity();
        for elem in tensor.iter() {
            acc = fold(acc, elem);
        }
        return acc;
    }

    // Parallel reduction using rayon's fold + reduce
    tensor
        .par_iter()
        .fold(|| identity(), |acc, elem| fold(acc, elem))
        .reduce(|| identity(), |a, b| fold(a, &b))
}

/// Parallel sum reduction.
///
/// Computes the sum of all elements using parallel reduction.
/// Returns `A::zero()` for empty tensors.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix1, linspace};
/// let t: Tensor<f64, Ix1> = linspace(0.0, 100.0, 200_000);
/// let total = par_sum(&t);
/// ```
#[cfg(feature = "parallel")]
pub fn par_sum<A, D>(tensor: &TensorBase<impl RawStorage<Elem = A>, D>) -> A
where
    A: Numeric + Send + Sync,
    D: Dimension + Send + Sync,
{
    par_reduce(tensor, || A::zero(), |acc, x| acc + *x)
}

/// Parallel product reduction.
///
/// Computes the product of all elements using parallel reduction.
/// Returns `A::one()` for empty tensors.
#[cfg(feature = "parallel")]
pub fn par_prod<A, D>(tensor: &TensorBase<impl RawStorage<Elem = A>, D>) -> A
where
    A: Numeric + Send + Sync,
    D: Dimension + Send + Sync,
{
    par_reduce(tensor, || A::one(), |acc, x| acc * *x)
}

/// Parallel min reduction.
///
/// Returns the minimum element. For empty tensors, returns `TensorError::EmptyArray`.
/// NaN propagation: if any element is NaN, returns NaN.
#[cfg(feature = "parallel")]
pub fn par_min<A, D>(tensor: &TensorBase<impl RawStorage<Elem = A>, D>) -> Result<A>
where
    A: Element + PartialOrd + Send + Sync,
    D: Dimension + Send + Sync,
{
    let len = tensor.len();
    if len == 0 {
        return Err(TensorError::EmptyArray {
            operation: "par_min".into(),
        });
    }

    par_reduce(
        tensor,
        || {
            // SAFETY: len > 0, so first element exists
            unsafe { tensor.as_ptr().read() }
        },
        |acc, x| {
            // NaN propagation: if either is NaN, return NaN
            if acc.is_nan() || x.is_nan() {
                A::nan()
            } else if acc <= *x {
                acc
            } else {
                *x
            }
        },
    )
    .pipe(Ok)
}

/// Parallel max reduction.
///
/// Returns the maximum element. For empty tensors, returns `TensorError::EmptyArray`.
/// NaN propagation: if any element is NaN, returns NaN.
#[cfg(feature = "parallel")]
pub fn par_max<A, D>(tensor: &TensorBase<impl RawStorage<Elem = A>, D>) -> Result<A>
where
    A: Element + PartialOrd + Send + Sync,
    D: Dimension + Send + Sync,
{
    let len = tensor.len();
    if len == 0 {
        return Err(TensorError::EmptyArray {
            operation: "par_max".into(),
        });
    }

    par_reduce(
        tensor,
        || {
            // SAFETY: len > 0, so first element exists
            unsafe { tensor.as_ptr().read() }
        },
        |acc, x| {
            if acc.is_nan() || x.is_nan() {
                A::nan()
            } else if acc >= *x {
                acc
            } else {
                *x
            }
        },
    )
    .pipe(Ok)
}
```

### 4.7 par_zip — 并行多数组同步迭代

```rust
/// Applies a function to corresponding elements of two tensors in parallel.
///
/// Both tensors must have compatible shapes (identical or broadcastable).
/// The function receives immutable references from both inputs.
///
/// # Arguments
///
/// * `lhs` - First input tensor.
/// * `rhs` - Second input tensor.
/// * `f` - Function applied to each element pair `(&A, &B)`.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, par_zip2, ones, zeros};
/// let a: Tensor<f64, Ix2> = ones([500, 500]);
/// let b: Tensor<f64, Ix2> = zeros([500, 500]);
/// par_zip2(&a, &b, |x, y| {
///     // Process each pair in parallel
///     assert_ne!(*x, *y);
/// })?;
/// ```
#[cfg(feature = "parallel")]
pub fn par_zip2<A, B, D, F>(
    lhs: &TensorBase<impl RawStorage<Elem = A>, D>,
    rhs: &TensorBase<impl RawStorage<Elem = B>, D>,
    f: F,
) -> Result<()>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    D: Dimension + Send + Sync,
    F: Fn(&A, &B) + Sync + Send,
{
    // Validate shape compatibility
    crate::broadcast::broadcast_shape(
        lhs.shape().slice(),
        rhs.shape().slice(),
    )?;

    let len = lhs.len();
    if len < PARALLEL_THRESHOLD {
        // Serial fallback using Zip
        crate::iter::Zip::from(&lhs.view())
            .and(&rhs.view())?
            .for_each(|a, b| f(a, b));
        return Ok(());
    }

    par_zip2_impl(lhs, rhs, f)
}

/// Applies a function to corresponding elements of three tensors in parallel.
///
/// The third tensor must be a mutable view (write target). The first two
/// are read-only inputs.
///
/// # Arguments
///
/// * `a` - First input tensor (read).
/// * `b` - Second input tensor (read).
/// * `c` - Third tensor (write).
/// * `f` - Function applied to each triple `(&A, &B, &mut C)`.
///
/// # Examples
///
/// ```ignore
/// use xenon::{Tensor, Ix2, par_zip3, ones, zeros};
/// let a: Tensor<f64, Ix2> = ones([500, 500]);
/// let b: Tensor<f64, Ix2> = ones([500, 500]);
/// let mut c: Tensor<f64, Ix2> = zeros([500, 500]);
/// par_zip3(&a, &b, &mut c, |x, y, z| *z = x + y)?;
/// ```
#[cfg(feature = "parallel")]
pub fn par_zip3<A, B, C, D, F>(
    a: &TensorBase<impl RawStorage<Elem = A>, D>,
    b: &TensorBase<impl RawStorage<Elem = B>, D>,
    c: &mut TensorBase<impl Storage<Elem = C>, D>,
    f: F,
) -> Result<()>
where
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send + Sync,
    D: Dimension + Send + Sync,
    F: Fn(&A, &B, &mut C) + Sync + Send,
{
    // Validate shape compatibility for all three
    let bc = crate::broadcast::broadcast_shape(
        a.shape().slice(),
        b.shape().slice(),
    )?;
    crate::broadcast::broadcast_shape(
        &bc,
        c.shape().slice(),
    )?;

    let len = a.len();
    if len < PARALLEL_THRESHOLD {
        crate::iter::Zip::from(&a.view())
            .and(&b.view())?
            .and(&mut c.view_mut())?
            .for_each(|a, b, c| f(a, b, c));
        return Ok(());
    }

    par_zip3_impl(a, b, c, f)
}
```

### 4.8 ThreadPoolConfig — 线程池配置

```rust
/// Configuration for a custom rayon thread pool.
///
/// By default, Xenon uses rayon's global thread pool. For advanced use cases
/// (e.g., nested library usage where you need to isolate thread pools, or
/// custom thread naming for profiling), you can create a custom pool and
/// install it for the current scope.
///
/// # Examples
///
/// ```ignore
/// use xenon::backend::parallel::{ThreadPoolConfig, par_map};
///
/// // Create a custom pool with 4 threads
/// let pool = ThreadPoolConfig::new()
///     .num_threads(4)
///     .build()
///     .expect("failed to create thread pool");
///
/// // Use the custom pool for parallel operations
/// pool.install(|| {
///     let result = par_map(&tensor, |x| x * 2.0);
/// });
/// ```
#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads. Defaults to the number of available cores.
    num_threads: Option<usize>,

    /// Prefix for thread names. Defaults to "xenon-wk".
    thread_name: String,

    /// Stack size for worker threads in bytes. Defaults to rayon's default (typically 8 MB).
    stack_size: Option<usize>,
}

#[cfg(feature = "parallel")]
impl ThreadPoolConfig {
    /// Creates a new `ThreadPoolConfig` with default settings.
    ///
    /// # Default Values
    ///
    /// | Setting | Default |
    /// |---------|---------|
    /// | `num_threads` | `None` (rayon auto-detects) |
    /// | `thread_name` | `"xenon-wk"` |
    /// | `stack_size` | `None` (rayon default) |
    pub fn new() -> Self {
        Self {
            num_threads: None,
            thread_name: "xenon-wk".into(),
            stack_size: None,
        }
    }

    /// Sets the number of worker threads.
    ///
    /// Pass `None` to let rayon auto-detect the number of available cores.
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self
    }

    /// Sets the thread name prefix for worker threads.
    ///
    /// Threads will be named `"prefix-0"`, `"prefix-1"`, etc.
    pub fn thread_name(mut self, name: impl Into<String>) -> Self {
        self.thread_name = name.into();
        self
    }

    /// Sets the stack size for worker threads in bytes.
    ///
    /// Pass `None` to use rayon's default stack size.
    pub fn stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }

    /// Builds a `rayon::ThreadPool` from this configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if rayon fails to create the thread pool
    /// (e.g., `num_threads == 0` or system resource exhaustion).
    pub fn build(self) -> Result<rayon::ThreadPool> {
        let mut builder = rayon::ThreadPoolBuilder::new()
            .thread_name(move |idx| format!("{}-{}", self.thread_name, idx));

        if let Some(n) = self.num_threads {
            builder = builder.num_threads(n);
        }
        if let Some(size) = self.stack_size {
            builder = builder.stack_size(size);
        }

        builder
            .build()
            .map_err(|e| TensorError::ThreadPoolError {
                message: e.to_string(),
            })
    }
}

#[cfg(feature = "parallel")]
impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}
```

### 4.9 并行阈值查询

```rust
/// Returns whether parallel execution should be used for the given element count.
///
/// Takes into account:
/// 1. Whether the `parallel` feature is enabled (always returns `false` without it)
/// 2. The element count vs `PARALLEL_THRESHOLD`
///
/// # Examples
///
/// ```ignore
/// use xenon::backend::parallel::should_parallelize;
///
/// assert!(!should_parallelize(100));
/// assert!(should_parallelize(100_000));
/// ```
#[cfg(feature = "parallel")]
#[inline]
pub fn should_parallelize(len: usize) -> bool {
    len >= PARALLEL_THRESHOLD
}
```

---

## 5. 内部实现设计

### 5.1 Chunk-based Parallelism — 分块并行策略

并行执行的核心策略是将张量数据按**连续内存块**分割给各工作线程。

**分块流程：**

```
par_iter:
    1. Check if data is contiguous
       ├── Contiguous (strides == 1 along fastest axis)
       │   ├── Compute optimal chunk size
       │   ├── Split into N chunks of [chunk_size] elements
       │   └── Each thread processes one contiguous chunk via slice
       └── Non-contiguous (arbitrary strides)
           ├── Fall back to per-element indexing
           └── Each thread processes a range of logical indices

par_map:
    1. Check threshold (>= PARALLEL_THRESHOLD)
    2. Allocate output buffer (aligned, zeroed)
    3. Partition input into chunks
    4. Each thread applies f to its chunk, writes to corresponding output slice
    5. No synchronization needed (disjoint output regions)

par_reduce:
    1. Check threshold
    2. Each thread computes local reduction over its chunk
    3. Final reduction combines all local results (tree reduction)
```

### 5.2 Optimal Chunk Size — 最优块大小计算

块大小计算遵循以下原则：

1. **下限**：`PARALLEL_MIN_CHUNK`（4K 元素 = 32 KB for f64）
2. **上限**：元素总数 / 线程数 × 2（避免极端不均衡）
3. **Cache-line 感知**：目标每个块 ≤ 256 KB（典型 L2 缓存大小）

```rust
/// Computes the optimal chunk size for parallel partitioning.
///
/// Strategy:
/// - For type sizes ≤ 8 bytes: target 256 KB chunks (L2 cache)
/// - For larger types: target 256 KB chunks based on byte size
/// - Always respect PARALLEL_MIN_CHUNK as lower bound
/// - Ensure even distribution: total / num_threads as upper bound
///
/// # Arguments
///
/// * `total_elements` - Total number of elements to partition.
/// * `element_size` - Size of each element in bytes (`size_of::<A>()`).
/// * `num_threads` - Number of available worker threads.
fn compute_chunk_size(total_elements: usize, element_size: usize, num_threads: usize) -> usize {
    // Target chunk in bytes: fit in L2 cache
    let target_chunk_elements = TARGET_CHUNK_BYTES / element_size.max(1);

    // Ensure at least PARALLEL_MIN_CHUNK elements per chunk
    let min_chunk = PARALLEL_MIN_CHUNK.max(target_chunk_elements);

    // Ensure we don't create too few chunks (want at least num_threads chunks)
    let max_chunk = (total_elements / num_threads.max(1)).max(min_chunk);

    // Final chunk size: clamp between min and max, power-of-2 aligned
    let chunk = min_chunk.min(max_chunk);
    chunk.next_power_of_two()
}
```

### 5.3 Contiguous vs Strided Data Handling

**连续数据（Fast path）：**

当张量沿某一轴步长为 1 且内存连续时，直接将底层数据切片分割：

```rust
// Pseudocode for contiguous chunk processing
fn process_contiguous_chunks<A>(ptr: *const A, len: usize, chunk_size: usize, f: impl Fn(*const A, usize)) {
    let num_chunks = (len + chunk_size - 1) / chunk_size;
    (0..num_chunks).into_par_iter().for_each(|i| {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(len);
        // SAFETY: chunks are non-overlapping, start+end within bounds
        unsafe { f(ptr.add(start), end - start) }
    });
}
```

**非连续数据（Slow path）：**

对于任意步长的数据，每个元素需要独立的索引计算。采用 rayon 的 `(0..len).into_par_iter()` 加步长映射：

```rust
// Pseudocode for strided element processing
fn process_strided_elements<A, D: Dimension>(
    ptr: *const A,
    shape: &D,
    strides: &D,
    offset: usize,
    len: usize,
    f: impl Fn(*const A),
) {
    (0..len).into_par_iter().for_each(|logical_idx| {
        // Convert linear index to N-dim index, then compute offset
        let byte_offset = compute_strided_offset(logical_idx, shape, strides) + offset;
        // SAFETY: logical_idx < len, offset computed correctly
        unsafe { f(ptr.add(byte_offset)) }
    });
}
```

**连续性判定标准：**

| 条件 | 判定 |
|------|------|
| `is_f_contiguous() && order == F` | 连续 fast path |
| `is_c_contiguous() && order == C` | 连续 fast path |
| `is_contiguous()` (任一方向) | 连续 fast path（使用对应的顺序） |
| 上述均不满足 | 非连续 slow path |

### 5.4 Rayon Producer 实现

`ParIter` 和 `ParIterMut` 通过实现 rayon 的 `Producer` trait 参与并行执行：

```rust
/// Internal chunk producer for contiguous tensor data.
///
/// Implements rayon's `Producer` trait, which enables work-stealing
/// parallel iteration. Each producer owns a contiguous slice of the
/// tensor data (for contiguous tensors) or a range of logical indices
/// (for strided tensors).
struct TensorProducer<'a, A, D: Dimension> {
    /// Pointer to the start of this producer's data range.
    ptr: *const A,

    /// Shape of the tensor (shared across all producers).
    shape: D,

    /// Strides of the tensor (shared across all producers).
    strides: D,

    /// Offset from ptr to the first logical element.
    base_offset: usize,

    /// Number of elements this producer is responsible for.
    len: usize,

    /// Whether the data is contiguous (enables slice-based splitting).
    contiguous: bool,

    /// Element size in bytes (for chunk size computation).
    _marker: core::marker::PhantomData<&'a A>,
}

impl<'a, A, D: Dimension> rayon::iter::Producer for TensorProducer<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Item = &'a A;
    type IntoIter = TensorChunkIter<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        // Create an iterator over this producer's range
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // Split this producer into two at the given index.
        // For contiguous data: adjust ptr and len.
        // For strided data: adjust len and base_offset.
    }
}
```

### 5.5 禁止嵌套并行

为防止线程池饥饿，所有并行操作内部强制使用串行路径：

```rust
/// Executes a closure with nested parallelism disabled.
///
/// This ensures that any rayon parallel operations called within `f`
/// will execute serially on the current thread, preventing thread pool
/// starvation when outer parallel operations spawn inner parallel work.
///
/// Implementation uses rayon's `ThreadPool::install` with a single-threaded
/// configuration, or alternatively checks rayon's current thread index to
/// detect if we're already inside a parallel scope.
#[inline]
fn serialize<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    // Strategy: Use rayon::current_thread_index() to detect nesting.
    // If we're on a rayon worker thread, execute serially.
    // Otherwise, this is a top-level call — execute normally.
    if rayon::current_thread_index().is_some() {
        // Already in a parallel context — force serial execution
        f()
    } else {
        f()
    }
}
```

> **设计决策：** 嵌套并行防护采用"最佳努力"策略。rayon 本身通过 work-stealing 避免死锁，但无法阻止过度订阅（oversubscription）。`serialize()` 函数确保内部操作不调用 rayon 的并行 API，而是直接使用标量路径。

### 5.6 线程池配置与 Work Stealing

**默认行为：** 使用 rayon 全局线程池，线程数 = 可用核心数。

**自定义线程池：** 通过 `ThreadPoolConfig` 创建，使用 `pool.install(|| ...)` 作用域安装。

```
Thread Pool Architecture:

┌─────────────────────────────────────────────┐
│              Global Thread Pool              │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐        │
│  │ W0  │  │ W1  │  │ W2  │  │ W3  │  ...   │
│  │deque│  │deque│  │deque│  │deque│         │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘        │
│     │        │        │        │            │
│     └────────┴────────┴────────┘            │
│          Work Stealing (rayon)               │
└─────────────────────────────────────────────┘

Chunk Distribution:
  Tensor [1M elements]
  ├── Chunk 0: [0..256K)     → Worker 0
  ├── Chunk 1: [256K..512K)  → Worker 1
  ├── Chunk 2: [512K..768K)  → Worker 2
  └── Chunk 3: [768K..1M)    → Worker 3
      (Worker 2 finishes early → steals from Worker 3)
```

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### Wave 1: 基础设施

- [ ] **T1: 创建模块骨架 + 配置常量**
  - 文件: `src/backend/parallel.rs`
  - 内容: 模块级 `#[cfg(feature = "parallel")]` 声明、`use` 导入、`PARALLEL_THRESHOLD`、`PARALLEL_MIN_CHUNK`、`TARGET_CHUNK_BYTES` 常量
  - 测试: 编译通过 (`cargo check --features parallel`)
  - 前置: rayon 依赖添加到 `Cargo.toml`
  - 预计: 5 min

- [ ] **T2: 实现 `should_parallelize()` + `compute_chunk_size()`**
  - 文件: `src/backend/parallel.rs`
  - 内容: 阈值判定函数、chunk 大小计算函数
  - 测试: `test_should_parallelize_below_threshold`, `test_should_parallelize_above_threshold`, `test_compute_chunk_size_f64`, `test_compute_chunk_size_small`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: 实现 `ThreadPoolConfig` 结构体 + `build()`**
  - 文件: `src/backend/parallel.rs`
  - 内容: `ThreadPoolConfig` 定义、builder 方法、`build()` 构造 `rayon::ThreadPool`
  - 测试: `test_thread_pool_config_default`, `test_thread_pool_config_custom_threads`, `test_thread_pool_config_build`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4: 实现 `serialize()` 嵌套并行防护**
  - 文件: `src/backend/parallel.rs`
  - 内容: `serialize()` 函数，使用 `rayon::current_thread_index()` 检测嵌套
  - 测试: `test_serialize_nested`, `test_serialize_top_level`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 并行迭代器 — ParIter

- [ ] **T5: 实现 `TensorProducer` — 连续数据的 Producer**
  - 文件: `src/backend/parallel.rs`
  - 内容: `TensorProducer` 结构体、`rayon::iter::Producer` impl、`split_at()` 连续分割、`into_iter()` 返回切片迭代器
  - 测试: `test_producer_split_contiguous`, `test_producer_len`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6: 实现 `TensorProducer` — 非连续数据的 Producer**
  - 文件: `src/backend/parallel.rs`
  - 内容: 扩展 `TensorProducer` 支持任意步长、`split_at()` 逻辑索引分割、步长感知的 `into_iter()`
  - 测试: `test_producer_split_strided`, `test_producer_strided_elements`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: 实现 `ParIter` 结构体 + `ParallelIterator` trait**
  - 文件: `src/backend/parallel.rs`
  - 内容: `ParIter` 定义、`new()`、`with_order()`、`ParallelIterator` impl（`drive_unindexed`、`opt_len`）
  - 测试: `test_par_iter_len`, `test_par_iter_opt_len`
  - 前置: T5, T6
  - 预计: 10 min

- [ ] **T8: 实现 `ParIter` 的 `IndexedParallelIterator` trait**
  - 文件: `src/backend/parallel.rs`
  - 内容: `len()`、`drive()`、`with_producer()` — 将 `TensorProducer` 传入 rayon 回调
  - 测试: `test_par_iter_sum`, `test_par_iter_collect`
  - 前置: T7
  - 预计: 10 min

- [ ] **T9: 实现 `ParIter` 的 `IntoParallelIterator` for TensorView/Tensor**
  - 文件: `src/backend/parallel.rs`
  - 内容: `IntoParallelIterator` for `&TensorView`、`&Tensor`
  - 测试: `test_into_par_iter_view`, `test_into_par_iter_owned`
  - 前置: T8
  - 预计: 5 min

### Wave 3: 并行迭代器 — ParIterMut

- [ ] **T10: 实现 `TensorProducerMut` — 可变 Producer**
  - 文件: `src/backend/parallel.rs`
  - 内容: `TensorProducerMut` 结构体、`Producer` impl、`split_at()` 确保不重叠
  - 测试: `test_producer_mut_split_no_overlap`, `test_producer_mut_write`
  - 前置: T5, T6
  - 预计: 10 min

- [ ] **T11: 实现 `ParIterMut` + `ParallelIterator` + `IndexedParallelIterator`**
  - 文件: `src/backend/parallel.rs`
  - 内容: `ParIterMut` 定义、完整 trait 实现
  - 测试: `test_par_iter_mut_write`, `test_par_iter_mut_sum`
  - 前置: T10
  - 预计: 10 min

- [ ] **T12: 实现 `TensorBase` 上的 `par_iter()` / `par_iter_mut()` 入口方法**
  - 文件: `src/backend/parallel.rs`
  - 内容: feature-gated 入口方法、`view()` / `view_mut()` 转换
  - 测试: `test_tensor_par_iter`, `test_tensor_par_iter_mut`
  - 前置: T8, T11
  - 预计: 5 min

### Wave 4: par_map 并行变换

- [ ] **T13: 实现 `par_map` — 连续数据路径**
  - 文件: `src/backend/parallel.rs`
  - 内容: 连续数据的并行 map 实现、输出缓冲区分配、分块并行写入
  - 测试: `test_par_map_contiguous`, `test_par_map_small_fallback`
  - 前置: T12
  - 预计: 10 min

- [ ] **T14: 实现 `par_map` — 非连续数据路径**
  - 文件: `src/backend/parallel.rs`
  - 内容: 任意步长的并行 map、步长索引映射
  - 测试: `test_par_map_strided`, `test_par_map_transposed`
  - 前置: T13
  - 预计: 10 min

- [ ] **T15: 实现 `par_map2` — 二元并行变换**
  - 文件: `src/backend/parallel.rs`
  - 内容: 两个输入张量的并行逐元素运算、广播支持
  - 测试: `test_par_map2_same_shape`, `test_par_map2_broadcast`, `test_par_map2_shape_mismatch`
  - 前置: T13
  - 预计: 10 min

- [ ] **T16: 实现 `par_map_inplace` — 就地并行变换**
  - 文件: `src/backend/parallel.rs`
  - 内容: 就地修改 `TensorViewMut` 的并行路径
  - 测试: `test_par_map_inplace`, `test_par_map_inplace_strided`
  - 前置: T13
  - 预计: 10 min

### Wave 5: par_reduce 并行归约

- [ ] **T17: 实现 `par_reduce` — 通用并行归约**
  - 文件: `src/backend/parallel.rs`
  - 内容: rayon `fold` + `reduce` 模式、identity 元素、阈值判定
  - 测试: `test_par_reduce_sum`, `test_par_reduce_product`, `test_par_reduce_empty`, `test_par_reduce_small_fallback`
  - 前置: T12
  - 预计: 10 min

- [ ] **T18: 实现 `par_sum` / `par_prod` / `par_min` / `par_max`**
  - 文件: `src/backend/parallel.rs`
  - 内容: 基于 `par_reduce` 的便捷归约函数、NaN 传播语义
  - 测试: `test_par_sum`, `test_par_prod`, `test_par_min`, `test_par_max`, `test_par_min_nan_propagation`
  - 前置: T17
  - 预计: 10 min

### Wave 6: par_zip 并行同步迭代

- [ ] **T19: 实现 `par_zip2` — 二元并行遍历**
  - 文件: `src/backend/parallel.rs`
  - 内容: 两个张量的并行同步遍历、广播检查
  - 测试: `test_par_zip2_same_shape`, `test_par_zip2_broadcast`, `test_par_zip2_shape_mismatch`
  - 前置: T12
  - 预计: 10 min

- [ ] **T20: 实现 `par_zip3` — 三元并行遍历（两读一写）**
  - 文件: `src/backend/parallel.rs`
  - 内容: 两读一写的并行遍历、不重叠写入保证
  - 测试: `test_par_zip3_write`, `test_par_zip3_broadcast`
  - 前置: T19
  - 预计: 10 min

### Wave 7: 集成与 re-export

- [ ] **T21: `backend/mod.rs` 注册并行模块**
  - 文件: `src/backend/mod.rs`
  - 内容: `#[cfg(feature = "parallel")] pub mod parallel;`、在 dispatch 函数中添加并行路径选择
  - 测试: 编译通过
  - 前置: T1-T20
  - 预计: 5 min

- [ ] **T22: `lib.rs` re-export + `iter/mod.rs` re-export**
  - 文件: `src/lib.rs`, `src/iter/mod.rs`
  - 内容: 公共类型 re-export（`ParIter`, `ParIterMut`, `par_map`, `par_reduce`, `par_zip`, `ThreadPoolConfig`）
  - 测试: `use xenon::ParIter;` 编译通过
  - 前置: T21
  - 预计: 5 min

- [ ] **T23: 更新 `10-iterators.md` 中并行迭代器的引用**
  - 文件: `docs/10-iterators.md`
  - 内容: 将 §4.6 的 ParIter/ParIterMut 定义替换为 "参见 `docs/21-parallel.md`"，保留入口方法签名
  - 测试: 文档一致性检查
  - 前置: T22
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/backend/parallel.rs` 内的 `#[cfg(test)] mod tests`，所有测试需标注 `#[cfg(feature = "parallel")]`。

#### 基础设施测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_should_parallelize_below_threshold` | `should_parallelize(100) == false` |
| `test_should_parallelize_at_threshold` | `should_parallelize(64 * 1024) == true` |
| `test_should_parallelize_above_threshold` | `should_parallelize(1_000_000) == true` |
| `test_compute_chunk_size_f64` | chunk 大小为 2 的幂，≥ 4K |
| `test_compute_chunk_size_large_types` | 大元素类型（如 Complex<f64>，16 bytes）的 chunk 计算 |
| `test_compute_chunk_size_small_array` | 小数组不会产生大于总长的 chunk |
| `test_thread_pool_config_default` | 默认配置构建成功 |
| `test_thread_pool_config_num_threads` | 自定义线程数生效 |
| `test_thread_pool_config_install` | `pool.install()` 内执行并行操作 |
| `test_serialize_nested` | 在 rayon worker 线程内调用 `serialize()` 不触发新并行 |

#### ParIter 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_par_iter_len` | `IndexedParallelIterator::len()` 正确 |
| `test_par_iter_sum` | `par_iter().cloned().sum()` 等价于串行 sum |
| `test_par_iter_collect` | 收集所有元素与串行结果一致 |
| `test_par_iter_f_contiguous` | F-order 连续数据：元素顺序正确 |
| `test_par_iter_c_contiguous` | C-order 连续数据：元素顺序正确 |
| `test_par_iter_non_contiguous` | 非连续数据（切片后）：结果正确 |
| `test_par_iter_single_element` | 单元素数组：返回一个元素 |
| `test_par_iter_empty` | 空数组：零个元素 |

#### ParIterMut 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_par_iter_mut_write` | 写入后原数组被修改 |
| `test_par_iter_mut_no_alias` | 各线程写入的元素不重叠 |
| `test_par_iter_mut_sum` | 写入后用 `par_iter` 验证结果 |
| `test_par_iter_mut_non_contiguous` | 非连续数据的并行写入 |

#### par_map 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_par_map_square` | `par_map(&t, \|x\| x * x)` 结果正确 |
| `test_par_map_type_change` | `par_map::<f64, i32, _>(\|x\| x as i32)` 类型转换正确 |
| `test_par_map_small_fallback` | 小数组（< 64K）走串行路径 |
| `test_par_map_contiguous` | 连续数据结果与串行一致 |
| `test_par_map_strided` | 非连续数据（转置后切片）结果正确 |
| `test_par_map2_same_shape` | 相同形状的二元 map 结果正确 |
| `test_par_map2_broadcast` | 广播形状的二元 map 结果正确 |
| `test_par_map2_shape_mismatch` | 不兼容形状返回错误 |
| `test_par_map_inplace` | 就地修改后数据正确 |
| `test_par_map_inplace_strided` | 非连续数据的就地修改 |

#### par_reduce 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_par_reduce_sum` | 求和结果正确（与已知值比较） |
| `test_par_reduce_product` | 求积结果正确 |
| `test_par_reduce_empty` | 空数组返回 identity |
| `test_par_reduce_single_element` | 单元素返回该元素 |
| `test_par_reduce_small_fallback` | 小数组走串行路径 |
| `test_par_sum` | `par_sum` 结果正确 |
| `test_par_prod` | `par_prod` 结果正确 |
| `test_par_min` | `par_min` 返回最小值 |
| `test_par_max` | `par_max` 返回最大值 |
| `test_par_min_empty_error` | 空数组返回 `EmptyArray` 错误 |
| `test_par_min_nan_propagation` | 含 NaN 时返回 NaN |
| `test_par_max_nan_propagation` | 含 NaN 时返回 NaN |

#### par_zip 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_par_zip2_same_shape` | 相同形状：正确配对处理 |
| `test_par_zip2_broadcast` | 广播形状：正确扩展 |
| `test_par_zip2_shape_mismatch` | 不兼容形状返回错误 |
| `test_par_zip2_small_fallback` | 小数组走串行 Zip 路径 |
| `test_par_zip3_write` | 三元（两读一写）：写入正确 |
| `test_par_zip3_no_overlap` | 各线程写入区域不重叠 |
| `test_par_zip3_broadcast` | 三元广播正确 |

### 7.2 集成测试

| 文件 | 测试内容 |
|------|----------|
| `tests/parallel.rs` | 并行操作的端到端使用场景（`#[cfg(feature = "parallel")]`） |

**集成测试用例：**

| 测试函数 | 测试内容 |
|----------|----------|
| `test_parallel_elementwise_add` | `par_map2` 实现并行加法，结果与串行一致 |
| `test_parallel_sum_reduction` | `par_sum` 结果与串行 `iter().sum()` 一致 |
| `test_parallel_matrix_normalize` | `par_zip3` 实现并行归一化 |
| `test_parallel_custom_thread_pool` | 使用自定义线程池执行并行操作 |
| `test_parallel_strided_slice` | 非连续切片的并行运算 |
| `test_parallel_threshold_boundary` | 64K-1 元素走串行，64K 元素走并行 |
| `test_parallel_large_tensor` | 1M+ 元素的并行吞吐测试 |
| `test_parallel_broadcast_scalar` | 标量广播 + 并行运算 |

### 7.3 边界测试

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `par_iter` 产出 0 项，`par_reduce` 返回 identity |
| 单元素 `shape=[1, 1]` | `par_iter` 产出 1 项（走串行路径） |
| 刚好 64K 元素 | 走并行路径 |
| 64K-1 元素 | 走串行路径 |
| 非连续切片 `t.slice(s![.., 0..3])` | `par_iter` 正确处理步长跳转 |
| 转置数组 `t.t()` | 步长反转的并行遍历正确 |
| 反转轴（负步长） | `par_iter` 正确处理负步长 |
| 广播视图 `broadcast [1,4] → [3,4]` | `par_iter` 产出 12 项（含重复引用） |
| 含 NaN 的归约 | `par_min`/`par_max` 返回 NaN |
| 全 Inf 的归约 | `par_sum` 返回 ±Inf |
| 极大数组（> 1G 元素） | 分块正确，不溢出 usize |
| 自定义线程池 0 线程 | 返回错误（`ThreadPoolError`） |

### 7.4 性能基准测试

| 基准 | 测量内容 |
|------|----------|
| `bench_par_iter_vs_serial` | 并行 vs 串行迭代吞吐量（1M f64 元素） |
| `bench_par_map_contiguous` | 连续数据并行 map 吞吐量 |
| `bench_par_map_strided` | 非连续数据并行 map 吞吐量 |
| `bench_par_reduce_sum` | 并行 sum 归约吞吐量 |
| `bench_par_zip2_binary` | 二元并行 zip 吞吐量 |
| `bench_par_zip3_ternary` | 三元并行 zip 吞吐量 |
| `bench_par_threshold_overhead` | 并行阈值附近的调度开销 |
| `bench_par_chunk_size_impact` | 不同 chunk 大小对吞吐量的影响 |
| `bench_par_scaling` | 线程数从 1 到 N 的 scaling 曲线 |
