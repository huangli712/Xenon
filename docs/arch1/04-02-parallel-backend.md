# 并行后端模块设计文档

> **模块路径**: `src/parallel/`
> **版本**: v18
> **日期**: 2026-03-28
> **前置文档**: 02-project-architecture.md, 03-06-tensor-core.md
> **需求来源**: require-v18.md 第 7.2 节、第 8 节

---

## 1. 模块概述

### 1.1 定位

`parallel` 模块是 Xenon 的可选性能后端，通过 feature gate `parallel` 启用，为大规模数组操作提供数据并行能力。该模块基于 rayon crate 实现，在保持 API 一致性的前提下，自动将适合的运算分发到多线程执行。

### 1.2 在性能分层中的角色

```
┌─────────────────────────────────────────────────────────────────┐
│                        性能分层决策树                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  元素数 < SIMD_WIDTH (如 16)                                    │
│  └─→ 标量路径 (scalar)                                          │
│                                                                 │
│  元素数 ≥ SIMD_WIDTH 且 连续内存 且 simd 启用                    │
│  └─→ SIMD 路径 (vectorized)                                     │
│                                                                 │
│  元素数 ≥ PARALLEL_THRESHOLD (64K) 且 parallel 启用              │
│  └─→ 并行 + SIMD 路径 (每线程内部使用 SIMD)                      │
│      ├─ 连续内存: 分块后各块 SIMD                                │
│      └─ 非连续内存: 分块后各块标量                               │
│                                                                 │
│  simd 未启用 且 元素数 ≥ PARALLEL_THRESHOLD 且 parallel 启用     │
│  └─→ 并行 + 标量路径                                            │
│                                                                 │
│  默认情况                                                       │
│  └─→ 标量路径 (scalar)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 设计目标

| 目标 | 说明 |
|------|------|
| 透明集成 | 用户代码无需修改，仅通过 feature gate 启用 |
| 自动决策 | 运行时根据数据规模和布局自动选择并行/串行 |
| 可配置性 | 支持全局配置和单次调用覆盖阈值 |
| 安全性 | 保证线程安全，禁止嵌套并行，避免数据竞争 |
| 兼容性 | 与 SIMD 模块协同工作，每线程内部使用 SIMD |

### 1.4 范围

**支持并行**:
- 逐元素运算（map、mapv、zip_with）
- 归约操作（sum、prod、min、max、mean 等）
- map/mapv 系列
- zip 多数组同步迭代

**不支持并行**:
- 矩阵乘法（由 BLAS 内部管理线程）
- 单元素操作
- 小数组操作（元素数 < 并行阈值）
- 非连续数组的某些操作（视具体实现）

---

## 2. 文件结构

```
src/parallel/
├── mod.rs             # 模块入口、rayon 集成、全局配置
├── par_iter.rs        # 并行迭代器（ParElements, ParAxisIter, ParZip）
└── par_ops.rs         # 并行运算（par_map, par_reduce, par_zip_with）
```

### 2.1 各文件职责

| 文件 | 职责 | 可见性 |
|------|------|--------|
| `mod.rs` | feature gate 声明、rayon 集成、全局阈值配置、模块导出 | pub（条件编译） |
| `par_iter.rs` | 并行迭代器 trait 和实现 | pub |
| `par_ops.rs` | 并行运算函数和 trait 实现 | pub |

### 2.2 模块依赖

```
┌─────────────────────────────────────────────────────────────────┐
│                         parallel 模块                           │
├─────────────────────────────────────────────────────────────────┤
│  mod.rs                                                         │
│  ├── 依赖 rayon                                                 │
│  ├── 依赖 std::sync::atomic (AtomicUsize)                       │
│  └── 导出 par_iter, par_ops                                     │
│                                                                 │
│  par_iter.rs                                                    │
│  ├── 依赖 crate::tensor (TensorBase)                            │
│  ├── 依赖 crate::layout (LayoutFlags)                           │
│  ├── 依赖 crate::iter (Elements, AxisIter 作为参考)             │
│  └── 依赖 rayon::iter (ParallelIterator)                        │
│                                                                 │
│  par_ops.rs                                                     │
│  ├── 依赖 crate::ops (Elementwise, Reduction trait)             │
│  ├── 依赖 crate::tensor (TensorBase)                            │
│  ├── 依赖 par_iter (ParElements)                                │
│  └── 依赖 rayon::iter (ParallelIterator)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 并行阈值系统

### 3.1 阈值层次

并行阈值系统采用三层配置优先级：

```
┌─────────────────────────────────────────────────────────────────┐
│                      阈值决策优先级                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 单次调用参数覆盖 (最高优先级)                                │
│     └─→ tensor.par_map_with_threshold(f, 128_000)               │
│                                                                 │
│  2. 线程局部配置 (中等优先级)                                    │
│     └─→ ParallelConfig::set_local_threshold(32_768)             │
│                                                                 │
│  3. 全局默认配置 (最低优先级)                                    │
│     └─→ PARALLEL_THRESHOLD = 65_536 (64K)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 全局配置 (AtomicUsize)

```rust
// src/parallel/mod.rs

use std::sync::atomic::{AtomicUsize, Ordering};

/// 默认并行阈值：64K 元素。
///
/// 当数组元素数达到或超过此值时，启用并行执行。
/// 此值经过经验调优，在小数组上并行开销大于收益。
const DEFAULT_PARALLEL_THRESHOLD: usize = 65_536;

/// 全局并行阈值。
///
/// 使用原子变量支持运行时修改，无需可变静态变量。
static GLOBAL_PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// 获取当前全局并行阈值。
///
/// # 示例
///
/// ```
/// let threshold = xenon::parallel::get_parallel_threshold();
/// println!("Current threshold: {}", threshold);
/// ```
pub fn get_parallel_threshold() -> usize {
    GLOBAL_PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

/// 设置全局并行阈值。
///
/// # 注意
///
/// 此设置影响所有后续并行操作。
/// 建议在程序启动时设置，避免运行时频繁修改。
///
/// # 示例
///
/// ```
/// xenon::parallel::set_parallel_threshold(128_000);
/// ```
pub fn set_parallel_threshold(threshold: usize) {
    GLOBAL_PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// 重置全局并行阈值为默认值。
pub fn reset_parallel_threshold() {
    GLOBAL_PARALLEL_THRESHOLD.store(DEFAULT_PARALLEL_THRESHOLD, Ordering::Relaxed);
}
```

**设计决策：使用 `Ordering::Relaxed`**

| 考量 | 说明 |
|------|------|
| 性能 | Relaxed 是最快的原子操作，无内存屏障 |
| 正确性 | 阈值读取不需要与其他操作同步，稍旧的值也可接受 |
| 场景 | 阈值修改通常是启动时一次性操作，不频繁 |

### 3.3 线程局部配置

```rust
// src/parallel/mod.rs

use std::cell::Cell;

thread_local! {
    /// 线程局部并行阈值覆盖。
    ///
    /// 设置为 `None` 时使用全局阈值。
    /// 设置为 `Some(n)` 时覆盖全局阈值。
    static LOCAL_PARALLEL_THRESHOLD: Cell<Option<usize>> = Cell::new(None);
}

/// 获取当前线程的并行阈值覆盖。
pub fn get_local_threshold() -> Option<usize> {
    LOCAL_PARALLEL_THRESHOLD.with(|cell| cell.get())
}

/// 设置当前线程的并行阈值覆盖。
///
/// 此设置仅影响当前线程，适用于需要临时调整阈值的场景。
///
/// # 示例
///
/// ```
/// // 临时提高阈值
/// xenon::parallel::set_local_threshold(Some(1_000_000));
/// // 执行操作
/// let result = tensor.par_map(|x| x * 2);
/// // 恢复默认
/// xenon::parallel::set_local_threshold(None);
/// ```
pub fn set_local_threshold(threshold: Option<usize>) {
    LOCAL_PARALLEL_THRESHOLD.with(|cell| cell.set(threshold));
}

/// 获取有效并行阈值。
///
/// 优先级：线程局部 > 全局 > 默认
pub fn effective_threshold() -> usize {
    get_local_threshold().unwrap_or_else(get_parallel_threshold)
}
```

### 3.4 单次调用参数覆盖

```rust
// src/parallel/par_ops.rs

impl<A, D> Tensor<A, D>
where
    D: Dimension,
    A: Element + Send + Sync,
{
    /// 并行映射，使用指定的阈值。
    ///
    /// # 参数
    ///
    /// * `f` - 映射函数
    /// * `threshold` - 本次操作使用的并行阈值
    ///
    /// # 示例
    ///
    /// ```
    /// let result = tensor.par_map_with_threshold(|x| x * 2, 128_000);
    /// ```
    pub fn par_map_with_threshold<F>(&self, f: F, threshold: usize) -> Self
    where
        F: Fn(&A) -> A + Sync,
    {
        if self.len() >= threshold {
            self.par_map_impl(f)
        } else {
            self.map(f)
        }
    }

    /// 并行映射，使用有效阈值（线程局部或全局）。
    pub fn par_map<F>(&self, f: F) -> Self
    where
        F: Fn(&A) -> A + Sync,
    {
        self.par_map_with_threshold(f, effective_threshold())
    }
}
```

### 3.5 阈值决策逻辑

```rust
// src/parallel/mod.rs

/// 判断是否应该启用并行执行。
///
/// # 决策逻辑
///
/// 1. 检查单次调用覆盖（由调用方传入）
/// 2. 检查线程局部阈值
/// 3. 使用全局阈值
/// 4. 考虑数据布局（非连续数组可能需要更高阈值）
///
/// # 参数
///
/// * `len` - 元素数量
/// * `override_threshold` - 单次调用覆盖阈值（可选）
/// * `is_contiguous` - 数据是否连续
///
/// # 返回
///
/// `true` 表示应该启用并行执行。
pub fn should_parallelize(
    len: usize,
    override_threshold: Option<usize>,
    is_contiguous: bool,
) -> bool {
    // 获取有效阈值
    let threshold = override_threshold
        .or_else(get_local_threshold)
        .unwrap_or_else(get_parallel_threshold);

    // 基本条件：元素数达到阈值
    let meets_threshold = len >= threshold;

    // 非连续数组额外开销更大，需要更大收益才值得并行
    // 对于非连续数组，阈值乘以 2
    let adjusted_meets = if is_contiguous {
        meets_threshold
    } else {
        len >= threshold * 2
    };

    adjusted_meets
}
```

---

## 4. 分块策略

### 4.1 设计原则

| 原则 | 说明 |
|------|------|
| 连续优先 | 优先按连续内存块分割，最大化缓存局部性 |
| 最小块保证 | 每块不小于 4K 元素，避免并行开销过大 |
| 负载均衡 | 块大小尽量均匀，避免某些线程处理过多 |
| 非连续处理 | 非连续数组采用不同的分块策略 |

### 4.2 连续内存块分割算法

```rust
// src/parallel/par_iter.rs

/// 最小块大小：4096 元素。
///
/// 此值确保每个任务有足够工作量，抵消线程调度开销。
const MIN_CHUNK_SIZE: usize = 4096;

/// 分块配置。
#[derive(Clone, Debug)]
pub struct ChunkConfig {
    /// 最小块大小（元素数）。
    pub min_chunk_size: usize,
    /// 目标块数（通常是线程数的倍数）。
    pub target_chunks: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        ChunkConfig {
            min_chunk_size: MIN_CHUNK_SIZE,
            target_chunks: rayon::current_num_threads() * 4, // 每线程 4 块
        }
    }
}

/// 计算连续数组的分块范围。
///
/// # 算法（伪代码）
///
/// ```
/// FUNCTION compute_chunks(total_len, config):
///     IF total_len < config.min_chunk_size THEN
///         RETURN [(0, total_len)]  // 单块，不并行
///     END IF
///
///     // 计算理想块大小
///     ideal_chunk_size = total_len / config.target_chunks
///     
///     // 确保块大小不小于最小值
///     chunk_size = MAX(ideal_chunk_size, config.min_chunk_size)
///     
///     // 计算实际块数
///     num_chunks = CEIL(total_len / chunk_size)
///     
///     // 生成块范围列表
///     chunks = []
///     FOR i FROM 0 TO num_chunks - 1:
///         start = i * chunk_size
///         end = MIN(start + chunk_size, total_len)
///         chunks.APPEND((start, end))
///     END FOR
///     
///     RETURN chunks
/// END FUNCTION
/// ```
///
/// # 参数
///
/// * `total_len` - 总元素数
/// * `config` - 分块配置
///
/// # 返回
///
/// 块范围列表，每个元素为 `(start, end)`，左闭右开区间。
pub fn compute_contiguous_chunks(total_len: usize, config: &ChunkConfig) -> Vec<(usize, usize)> {
    if total_len < config.min_chunk_size {
        return vec![(0, total_len)];
    }

    let ideal_chunk_size = total_len / config.target_chunks;
    let chunk_size = ideal_chunk_size.max(config.min_chunk_size);
    let num_chunks = (total_len + chunk_size - 1) / chunk_size;

    (0..num_chunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(total_len);
            (start, end)
        })
        .collect()
}
```

### 4.3 非连续数组处理

非连续数组（如转置视图、切片视图）的并行分块更复杂，因为元素在内存中不连续。

```rust
// src/parallel/par_iter.rs

/// 非连续数组的分块策略。
///
/// 对于非连续数组，我们按**逻辑索引范围**分块，
/// 而非物理内存范围。每块处理一段连续的逻辑索引。
///
/// # 策略
///
/// 1. 沿第一轴（最外层轴）分块
/// 2. 每块包含若干完整的行/切片
/// 3. 块内遍历仍需处理步长跳跃
///
/// # 伪代码
///
/// ```
/// FUNCTION compute_strided_chunks(shape, strides, config):
///     IF shape.ndim() == 0 THEN
///         RETURN [(0, 1)]  // 标量
///     END IF
///
///     // 沿第一轴分块
///     first_axis_len = shape[0]
///     first_axis_stride = strides[0]
///
///     // 计算每块的行数
///     // 目标：每块至少 min_chunk_size 个元素
///     elements_per_row = PRODUCT(shape[1..])
///     IF elements_per_row == 0 THEN
///         RETURN [(0, first_axis_len)]
///     END IF
///
///     rows_per_chunk = MAX(1, config.min_chunk_size / elements_per_row)
///
///     // 生成分块
///     chunks = []
///     FOR start_row FROM 0 TO first_axis_len STEP rows_per_chunk:
///         end_row = MIN(start_row + rows_per_chunk, first_axis_len)
///         // 记录起始行和结束行
///         chunks.APPEND((start_row, end_row))
///     END FOR
///
///     RETURN chunks
/// END FUNCTION
/// ```
pub fn compute_strided_chunks<D: Dimension>(
    shape: &D,
    strides: &D,
    config: &ChunkConfig,
) -> Vec<(usize, usize)> {
    let ndim = shape.ndim();
    if ndim == 0 {
        return vec![(0, 1)];
    }

    let shape_slice = shape.slice();
    let first_axis_len = shape_slice[0];

    // 计算每行元素数（除第一轴外所有轴的乘积）
    let elements_per_row: usize = shape_slice[1..].iter().product();
    if elements_per_row == 0 {
        return vec![(0, first_axis_len)];
    }

    // 计算每块的行数
    let rows_per_chunk = (config.min_chunk_size / elements_per_row).max(1);

    // 生成分块
    (0..first_axis_len)
        .step_by(rows_per_chunk)
        .map(|start_row| {
            let end_row = (start_row + rows_per_chunk).min(first_axis_len);
            (start_row, end_row)
        })
        .collect()
}
```

### 4.4 最小块大小保证

```rust
// src/parallel/par_iter.rs

/// 确保分块满足最小大小要求。
///
/// 如果计算出的块太小，合并相邻块。
pub fn ensure_min_chunk_size(chunks: &mut Vec<(usize, usize)>, min_size: usize) {
    if chunks.len() <= 1 {
        return;
    }

    let mut merged = Vec::with_capacity(chunks.len());
    let mut current_start = chunks[0].0;
    let mut current_len = chunks[0].1 - chunks[0].0;

    for &(start, end) in &chunks[1..] {
        let len = end - start;

        if current_len + len < min_size {
            // 合并到当前块
            current_len += len;
        } else {
            // 当前块已足够大，保存并开始新块
            merged.push((current_start, current_start + current_len));
            current_start = start;
            current_len = len;
        }
    }

    // 保存最后一块
    if current_len > 0 {
        merged.push((current_start, current_start + current_len));
    }

    *chunks = merged;
}
```

### 4.5 分块决策流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       分块决策流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: tensor (TensorBase<S, D>)                                │
│                                                                 │
│  1. 检查连续性                                                   │
│     ├─ is_f_contiguous() || is_c_contiguous() ?                 │
│     │                                                           │
│     ├─ 是 → 使用连续分块策略                                     │
│     │   └─ compute_contiguous_chunks(len, config)               │
│     │                                                           │
│     └─ 否 → 使用非连续分块策略                                   │
│         └─ compute_strided_chunks(shape, strides, config)       │
│                                                                 │
│  2. 确保最小块大小                                               │
│     └─ ensure_min_chunk_size(chunks, MIN_CHUNK_SIZE)            │
│                                                                 │
│  3. 返回分块迭代器                                               │
│     └─ chunks.into_par_iter()                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 并行迭代器设计

### 5.1 核心设计

并行迭代器需要实现 rayon 的 `ParallelIterator` trait，同时提供与串行迭代器一致的接口。

### 5.2 ParElements — 并行元素迭代器

```rust
// src/parallel/par_iter.rs

use rayon::iter::{ParallelIterator, IndexedParallelIterator};
use rayon::range::Range<usize>;

/// 并行元素迭代器。
///
/// 按内存布局顺序并行迭代所有元素。
/// 元素被分块后分配到不同线程处理。
///
/// # 类型参数
///
/// * `'a` - 视图生命周期
/// * `A` - 元素类型
/// * `D` - 维度类型
///
/// # 线程安全
///
/// * `A` 必须实现 `Send` 和 `Sync`
/// * 各线程访问不重叠的元素区间
pub struct ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    /// 底层数组视图。
    base: TensorView<'a, A, D>,

    /// 分块配置。
    chunk_config: ChunkConfig,
}

impl<'a, A, D> ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    /// 创建并行元素迭代器。
    ///
    /// # 参数
    ///
    /// * `base` - 底层数组视图
    ///
    /// # 返回
    ///
    /// 并行迭代器实例。
    pub fn new(base: TensorView<'a, A, D>) -> Self {
        ParElements {
            base,
            chunk_config: ChunkConfig::default(),
        }
    }

    /// 使用自定义分块配置创建迭代器。
    pub fn with_chunk_config(base: TensorView<'a, A, D>, config: ChunkConfig) -> Self {
        ParElements {
            base,
            chunk_config: config,
        }
    }

    /// 获取元素总数。
    pub fn len(&self) -> usize {
        self.base.len()
    }

    /// 检查是否为空。
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }
}

/// 并行元素迭代器的生产者（Producer）。
///
/// 实现 rayon 的 `Producer` trait，支持分块和并行执行。
struct ParElementsProducer<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
    chunk_config: ChunkConfig,
}

// ParallelIterator 实现
impl<'a, A, D> ParallelIterator for ParElements<'a, A, D>
where
    A: Element + Sync + Send,
    D: Dimension + Send + Sync,
{
    type Item = A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        // 使用 rayon 的 bridge 机制
        rayon::iter::plumbing::bridge(self, consumer)
    }
}

// IndexedParallelIterator 实现
impl<'a, A, D> IndexedParallelIterator for ParElements<'a, A, D>
where
    A: Element + Sync + Send + Clone,
    D: Dimension + Send + Sync,
{
    fn len(&self) -> usize {
        self.base.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        rayon::iter::plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(ParElementsProducer {
            base: self.base,
            chunk_config: self.chunk_config,
        })
    }
}
```

### 5.3 ParAxisIter — 并行轴迭代器

```rust
// src/parallel/par_iter.rs

/// 并行轴迭代器。
///
/// 沿指定轴并行迭代，每次产出降维后的子数组视图。
///
/// # 示例
///
/// ```
/// let tensor = Tensor3::<f64>::zeros([4, 100, 100]);
/// // 沿轴 0 并行迭代，每次处理一个 (100, 100) 的切片
/// tensor.par_axis_iter(Axis(0)).for_each(|slice| {
///     // 处理 slice: TensorView2<f64>
/// });
/// ```
pub struct ParAxisIter<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    /// 底层数组视图。
    base: TensorView<'a, A, D>,

    /// 迭代的轴。
    axis: usize,

    /// 轴长度。
    axis_len: usize,

    /// 分块配置。
    chunk_config: ChunkConfig,
}

impl<'a, A, D> ParAxisIter<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    /// 创建并行轴迭代器。
    ///
    /// # 参数
    ///
    /// * `base` - 底层数组视图
    /// * `axis` - 迭代的轴索引
    ///
    /// # 错误
    ///
    /// 如果 `axis` 超出维度数，返回 `InvalidAxis` 错误。
    pub fn new(base: TensorView<'a, A, D>, axis: usize) -> Result<Self, InvalidAxis> {
        if axis >= base.ndim() {
            return Err(InvalidAxis {
                axis,
                ndim: base.ndim(),
            });
        }

        let axis_len = base.shape()[axis];
        Ok(ParAxisIter {
            base,
            axis,
            axis_len,
            chunk_config: ChunkConfig::default(),
        })
    }

    /// 获取轴长度（迭代次数）。
    pub fn len(&self) -> usize {
        self.axis_len
    }

    /// 检查是否为空。
    pub fn is_empty(&self) -> bool {
        self.axis_len == 0
    }
}

impl<'a, A, D> ParallelIterator for ParAxisIter<'a, A, D>
where
    A: Element + Sync + Send,
    D: Dimension + RemoveAxis + Send + Sync,
    D::Smaller: Send + Sync,
{
    type Item = TensorView<'a, A, D::Smaller>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        // 沿轴迭代，每个索引产出子视图
        (0..self.axis_len)
            .into_par_iter()
            .map(|i| {
                // 安全：i 在 [0, axis_len) 范围内
                self.base.index_axis(self.axis, i)
            })
            .drive_unindexed(consumer)
    }
}
```

### 5.4 ParZip — 并行多数组同步迭代

```rust
// src/parallel/par_iter.rs

/// 并行多数组同步迭代器。
///
/// 同时迭代多个数组，按元素产出元组。
/// 支持广播：形状不一致但可广播时自动扩展。
///
/// # 示例
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);
///
/// // 并行 zip 迭代
/// let result: Vec<_> = ParZip::new(&a, &b).collect();
/// // result: [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]
/// ```
pub struct ParZip<'a, A, B, DA, DB>
where
    A: Element + Sync,
    B: Element + Sync,
    DA: Dimension,
    DB: Dimension,
{
    /// 第一个数组。
    a: TensorView<'a, A, DA>,

    /// 第二个数组。
    b: TensorView<'a, B, DB>,

    /// 广播后的元素数。
    len: usize,

    /// 分块配置。
    chunk_config: ChunkConfig,
}

impl<'a, A, B, DA, DB> ParZip<'a, A, B, DA, DB>
where
    A: Element + Sync,
    B: Element + Sync,
    DA: Dimension,
    DB: Dimension,
{
    /// 创建并行 zip 迭代器。
    ///
    /// # 参数
    ///
    /// * `a` - 第一个数组
    /// * `b` - 第二个数组
    ///
    /// # 错误
    ///
    /// 如果形状不兼容且无法广播，返回 `BroadcastError`。
    pub fn new(a: TensorView<'a, A, DA>, b: TensorView<'a, B, DB>) -> Result<Self, BroadcastError> {
        // 检查广播兼容性
        let broadcast_shape = broadcast::broadcast_shape(&a.shape(), &b.shape())?;
        let len = broadcast_shape.iter().product();

        Ok(ParZip {
            a,
            b,
            len,
            chunk_config: ChunkConfig::default(),
        })
    }

    /// 获取元素总数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 检查是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, A, B, DA, DB> ParallelIterator for ParZip<'a, A, B, DA, DB>
where
    A: Element + Sync + Send,
    B: Element + Sync + Send,
    DA: Dimension + Send + Sync,
    DB: Dimension + Send + Sync,
{
    type Item = (A, B);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        // 使用索引迭代，广播访问
        (0..self.len)
            .into_par_iter()
            .map(|i| {
                // 计算多维索引并访问元素
                // 这里需要处理广播逻辑
                let idx_a = broadcast_index(&self.a, i);
                let idx_b = broadcast_index(&self.b, i);
                (self.a[idx_a], self.b[idx_b])
            })
            .drive_unindexed(consumer)
    }
}

/// 三数组并行 zip。
pub struct ParZip3<'a, A, B, C, DA, DB, DC>
where
    A: Element + Sync,
    B: Element + Sync,
    C: Element + Sync,
    DA: Dimension,
    DB: Dimension,
    DC: Dimension,
{
    a: TensorView<'a, A, DA>,
    b: TensorView<'a, B, DB>,
    c: TensorView<'a, C, DC>,
    len: usize,
    chunk_config: ChunkConfig,
}

// 类似 ParZip 实现...
```

### 5.5 完整 Trait 签名

```rust
// src/parallel/par_iter.rs

/// 并行迭代 trait。
///
/// 为 `TensorBase` 提供并行迭代能力。
pub trait IntoParallelIterator {
    /// 并行迭代器类型。
    type Iter: ParallelIterator;

    /// 转换为并行迭代器。
    fn into_par_iter(self) -> Self::Iter;
}

/// 并行引用迭代 trait。
pub trait IntoParallelRefIterator<'a> {
    /// 元素类型。
    type Item: Send + Sync;

    /// 并行迭代器类型。
    type Iter: ParallelIterator<Item = Self::Item>;

    /// 转换为并行引用迭代器。
    fn par_iter(&'a self) -> Self::Iter;
}

/// 并行可变引用迭代 trait。
pub trait IntoParallelRefMutIterator<'a> {
    /// 元素类型。
    type Item: Send + Sync;

    /// 并行迭代器类型。
    type Iter: ParallelIterator<Item = &'a mut Self::Item>;

    /// 转换为并行可变引用迭代器。
    fn par_iter_mut(&'a mut self) -> Self::Iter;
}

// 为 TensorBase 实现 trait
impl<'a, S, D, A> IntoParallelRefIterator<'a> for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Sync + Send,
{
    type Item = A;
    type Iter = ParElements<'a, A, D>;

    fn par_iter(&'a self) -> Self::Iter {
        ParElements::new(self.view())
    }
}

impl<'a, S, D, A> IntoParallelRefMutIterator<'a> for TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element + Sync + Send,
{
    type Item = A;
    type Iter = ParElementsMut<'a, A, D>;

    fn par_iter_mut(&'a mut self) -> Self::Iter {
        ParElementsMut::new(self.view_mut())
    }
}
```

---

## 6. 并行运算 (par_ops.rs)

### 6.1 par_map — 并行映射

```rust
// src/parallel/par_ops.rs

use rayon::prelude::*;

/// 并行映射运算。
///
/// 对每个元素应用函数，返回新数组。
///
/// # 类型参数
///
/// * `F` - 映射函数类型
///
/// # 约束
///
/// * `F: Fn(&A) -> B + Sync` - 函数必须可跨线程共享
/// * `A: Send + Sync` - 输入元素必须线程安全
/// * `B: Send` - 输出元素必须可跨线程发送
///
/// # 示例
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let b = a.par_map(|x| x * 2.0);
/// assert_eq!(b, Tensor1::from_vec(vec![2.0, 4.0, 6.0, 8.0]));
/// ```
pub fn par_map<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync,
{
    // 检查是否应该并行
    if !should_parallelize(tensor.len(), None, tensor.is_contiguous()) {
        return tensor.map(&f);
    }

    // 分配输出缓冲区
    let len = tensor.len();
    let mut output: Vec<B> = Vec::with_capacity(len);

    // 并行处理
    // 使用 par_iter 会在每块内调用 f
    output.par_extend(tensor.par_iter().map(|x| f(x)));

    // 构造输出张量
    // 安全：output 长度与 tensor.len() 相同
    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}

/// 原地并行映射。
///
/// 直接在原数组上修改，不分配新内存。
///
/// # 约束
///
/// * 输入输出类型必须相同
/// * 需要可变访问
///
/// # 示例
///
/// ```
/// let mut a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// a.par_map_inplace(|x| *x = *x * 2.0);
/// assert_eq!(a, Tensor1::from_vec(vec![2.0, 4.0, 6.0]));
/// ```
pub fn par_map_inplace<A, D, F>(tensor: &mut Tensor<A, D>, f: F)
where
    D: Dimension,
    A: Element + Send + Sync,
    F: Fn(&mut A) + Sync,
{
    if tensor.is_empty() {
        return;
    }

    // 获取可变切片
    // 注意：需要连续内存才能安全地并行修改
    if tensor.is_contiguous() {
        let data = tensor.as_slice_mut().unwrap();
        data.par_iter_mut().for_each(|x| f(x));
    } else {
        // 非连续数组回退到串行
        tensor.mapv_inplace(|x| {
            let mut v = x;
            f(&mut v);
            v
        });
    }
}
```

### 6.2 par_reduce — 并行归约

```rust
// src/parallel/par_ops.rs

/// 并行归约。
///
/// 使用关联操作将所有元素归约为单个值。
///
/// # 类型参数
///
/// * `F` - 归约函数类型
/// * `ID` - 初始值函数类型
///
/// # 约束
///
/// * `F: Fn(A, A) -> A + Sync` - 归约函数
/// * `ID: Fn() -> A + Sync` - 初始值（单位元）
///
/// # 示例
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let sum = a.par_reduce(|| 0.0, |a, b| a + b);
/// assert_eq!(sum, 10.0);
/// ```
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

    // 检查是否应该并行
    if !should_parallelize(tensor.len(), None, tensor.is_contiguous()) {
        return tensor.iter().fold(identity(), |acc, x| op(acc, *x));
    }

    // 并行归约
    tensor
        .par_iter()
        .cloned()
        .reduce(identity, op)
}

/// 并行求和。
///
/// # 示例
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(a.par_sum(), 6.0);
/// ```
pub fn par_sum<A, D>(tensor: &Tensor<A, D>) -> A
where
    D: Dimension,
    A: Element + Numeric + Send + Sync,
{
    par_reduce(tensor, || A::zero(), |a, b| a + b)
}

/// 并行求积。
pub fn par_prod<A, D>(tensor: &Tensor<A, D>) -> A
where
    D: Dimension,
    A: Element + Numeric + Send + Sync,
{
    par_reduce(tensor, || A::one(), |a, b| a * b)
}

/// 并行最大值。
///
/// # 注意
///
/// 如果数组为空，返回 `None`。
/// 如果包含 NaN，结果为 NaN（NaN 传播语义）。
pub fn par_max<A, D>(tensor: &Tensor<A, D>) -> Option<A>
where
    D: Dimension,
    A: Element + PartialOrd + Send + Sync,
{
    if tensor.is_empty() {
        return None;
    }

    tensor
        .par_iter()
        .cloned()
        .reduce_with(|a, b| if a > b { a } else { b })
}

/// 并行最小值。
pub fn par_min<A, D>(tensor: &Tensor<A, D>) -> Option<A>
where
    D: Dimension,
    A: Element + PartialOrd + Send + Sync,
{
    if tensor.is_empty() {
        return None;
    }

    tensor
        .par_iter()
        .cloned()
        .reduce_with(|a, b| if a < b { a } else { b })
}
```

### 6.3 par_zip_with — 并行多数组运算

```rust
// src/parallel/par_ops.rs

/// 并行 zip 运算。
///
/// 对两个数组的对应元素应用函数。
/// 支持广播。
///
/// # 约束
///
/// * `F: Fn(&A, &B) -> C + Sync` - 组合函数
/// * 形状必须兼容（相同或可广播）
///
/// # 示例
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);
/// let c = par_zip_with(&a, &b, |x, y| x + y);
/// assert_eq!(c, Tensor1::from_vec(vec![5.0, 7.0, 9.0]));
/// ```
pub fn par_zip_with<A, B, C, DA, DB, DC, F>(
    a: &Tensor<A, DA>,
    b: &Tensor<B, DB>,
    f: F,
) -> Result<Tensor<C, DC>, BroadcastError>
where
    DA: Dimension,
    DB: Dimension,
    DC: Dimension,
    A: Element + Send + Sync,
    B: Element + Send + Sync,
    C: Element + Send,
    F: Fn(&A, &B) -> C + Sync,
{
    // 计算广播后的形状
    let broadcast_shape = broadcast::broadcast_shape(&a.shape(), &b.shape())?;
    let broadcast_dim = DC::from_slice(&broadcast_shape)?;

    // 检查是否应该并行
    let len: usize = broadcast_shape.iter().product();
    let is_contiguous = a.is_contiguous() && b.is_contiguous();

    if !should_parallelize(len, None, is_contiguous) {
        // 回退到串行
        return Ok(a.zip_with(b, &f)?);
    }

    // 并行处理
    let mut output: Vec<C> = Vec::with_capacity(len);

    // 创建并行 zip 迭代器
    let zip = ParZip::new(a.view(), b.view())?;
    output.par_extend(zip.map(|(x, y)| f(&x, &y)));

    // 构造输出张量
    unsafe { Ok(Tensor::from_raw_vec_unchecked(output, broadcast_dim)) }
}

/// 并行逐元素加法。
pub fn par_add<A, D>(a: &Tensor<A, D>, b: &Tensor<A, D>) -> Result<Tensor<A, D>, BroadcastError>
where
    D: Dimension,
    A: Numeric + Send + Sync,
{
    par_zip_with(a, b, |x, y| *x + *y)
}

/// 并行逐元素减法。
pub fn par_sub<A, D>(a: &Tensor<A, D>, b: &Tensor<A, D>) -> Result<Tensor<A, D>, BroadcastError>
where
    D: Dimension,
    A: Numeric + Send + Sync,
{
    par_zip_with(a, b, |x, y| *x - *y)
}

/// 并行逐元素乘法。
pub fn par_mul<A, D>(a: &Tensor<A, D>, b: &Tensor<A, D>) -> Result<Tensor<A, D>, BroadcastError>
where
    D: Dimension,
    A: Numeric + Send + Sync,
{
    par_zip_with(a, b, |x, y| *x * *y)
}

/// 并行逐元素除法。
pub fn par_div<A, D>(a: &Tensor<A, D>, b: &Tensor<A, D>) -> Result<Tensor<A, D>, BroadcastError>
where
    D: Dimension,
    A: Numeric + Send + Sync,
{
    par_zip_with(a, b, |x, y| *x / *y)
}
```

### 6.4 Trait 集成

```rust
// src/parallel/par_ops.rs

/// 并行运算 trait。
///
/// 为 `TensorBase` 提供并行运算方法。
pub trait ParallelElementwise<A, D>: Sized
where
    D: Dimension,
    A: Element + Send + Sync,
{
    /// 并行映射。
    fn par_map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element + Send,
        F: Fn(&A) -> B + Sync;

    /// 原地并行映射。
    fn par_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&mut A) + Sync;
}

/// 并行归约 trait。
pub trait ParallelReduction<A, D>: Sized
where
    D: Dimension,
    A: Element + Send + Sync,
{
    /// 并行归约。
    fn par_reduce<F, ID>(&self, identity: ID, op: F) -> A
    where
        F: Fn(A, A) -> A + Sync,
        ID: Fn() -> A + Sync + Clone;

    /// 并行求和。
    fn par_sum(&self) -> A
    where
        A: Numeric;

    /// 并行求积。
    fn par_prod(&self) -> A
    where
        A: Numeric;

    /// 并行最大值。
    fn par_max(&self) -> Option<A>
    where
        A: PartialOrd;

    /// 并行最小值。
    fn par_min(&self) -> Option<A>
    where
        A: PartialOrd;
}

// 为 Tensor 实现 trait
impl<A, D> ParallelElementwise<A, D> for Tensor<A, D>
where
    D: Dimension,
    A: Element + Send + Sync,
{
    fn par_map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element + Send,
        F: Fn(&A) -> B + Sync,
    {
        par_map(self, f)
    }

    fn par_map_inplace<F>(&mut self, f: F)
    where
        F: Fn(&mut A) + Sync,
    {
        par_map_inplace(self, f)
    }
}

impl<A, D> ParallelReduction<A, D> for Tensor<A, D>
where
    D: Dimension,
    A: Element + Send + Sync,
{
    fn par_reduce<F, ID>(&self, identity: ID, op: F) -> A
    where
        F: Fn(A, A) -> A + Sync,
        ID: Fn() -> A + Sync + Clone,
    {
        par_reduce(self, identity, op)
    }

    fn par_sum(&self) -> A
    where
        A: Numeric,
    {
        par_sum(self)
    }

    fn par_prod(&self) -> A
    where
        A: Numeric,
    {
        par_prod(self)
    }

    fn par_max(&self) -> Option<A>
    where
        A: PartialOrd,
    {
        par_max(self)
    }

    fn par_min(&self) -> Option<A>
    where
        A: PartialOrd,
    {
        par_min(self)
    }
}
```

---

## 7. 嵌套并行防护

### 7.1 问题背景

嵌套并行（nested parallelism）指在并行迭代内部再次调用并行操作，例如：

```rust
// 嵌套并行示例
tensor.par_axis_iter(Axis(0)).for_each(|slice| {
    // 内层也是并行操作
    let sum = slice.par_sum(); // 危险！
});
```

**嵌套并行的危害**：

| 问题 | 说明 |
|------|------|
| 线程池饥饿 | 外层已占用所有线程，内层无法获取线程 |
| 性能下降 | 线程调度开销增加，实际并行度降低 |
| 死锁风险 | 某些情况下可能导致死锁 |

### 7.2 防护机制设计

Xenon 采用 **线程局部标记** 检测和禁止嵌套并行。

```rust
// src/parallel/mod.rs

use std::cell::Cell;

thread_local! {
    /// 并行执行深度计数器。
    ///
    /// 0: 未在并行上下文中
    /// 1: 在外层并行上下文中
    /// >1: 嵌套并行（应禁止）
    static PARALLEL_DEPTH: Cell<usize> = Cell::new(0);
}

/// 检查是否在并行上下文中。
///
/// # 返回
///
/// `true` 表示当前线程正在执行并行操作。
pub fn is_in_parallel_context() -> bool {
    PARALLEL_DEPTH.with(|cell| cell.get() > 0)
}

/// 获取当前并行深度。
pub fn parallel_depth() -> usize {
    PARALLEL_DEPTH.with(|cell| cell.get())
}

/// 进入并行上下文。
///
/// 增加并行深度计数。
/// 如果检测到嵌套并行，panic。
///
/// # Panic
///
/// 如果当前已在并行上下文中，panic 并提示嵌套并行错误。
pub fn enter_parallel_context() {
    PARALLEL_DEPTH.with(|cell| {
        let depth = cell.get();
        if depth > 0 {
            panic!(
                "Nested parallelism detected! \
                 Cannot start parallel operation inside another parallel operation. \
                 Consider using sequential operations in the inner loop."
            );
        }
        cell.set(depth + 1);
    });
}

/// 退出并行上下文。
///
/// 减少并行深度计数。
pub fn exit_parallel_context() {
    PARALLEL_DEPTH.with(|cell| {
        let depth = cell.get();
        debug_assert!(depth > 0, "exit_parallel_context called without matching enter");
        cell.set(depth.saturating_sub(1));
    });
}

/// 并行上下文守卫。
///
/// RAII 风格的并行上下文管理，自动进入和退出。
pub struct ParallelGuard(());

impl ParallelGuard {
    /// 创建并行守卫，进入并行上下文。
    pub fn enter() -> Self {
        enter_parallel_context();
        ParallelGuard(())
    }
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        exit_parallel_context();
    }
}
```

### 7.3 在并行操作中使用防护

```rust
// src/parallel/par_ops.rs

pub fn par_map<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync,
{
    // 检查是否应该并行
    if !should_parallelize(tensor.len(), None, tensor.is_contiguous()) {
        return tensor.map(&f);
    }

    // 进入并行上下文
    let _guard = ParallelGuard::enter();

    // 执行并行操作
    let len = tensor.len();
    let mut output: Vec<B> = Vec::with_capacity(len);
    output.par_extend(tensor.par_iter().map(|x| f(x)));

    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}

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

    if !should_parallelize(tensor.len(), None, tensor.is_contiguous()) {
        return tensor.iter().fold(identity(), |acc, x| op(acc, *x));
    }

    // 进入并行上下文
    let _guard = ParallelGuard::enter();

    tensor
        .par_iter()
        .cloned()
        .reduce(identity, op)
}
```

### 7.4 内层强制单线程

另一种策略是：检测到嵌套时，内层自动回退到单线程执行，而非 panic。

```rust
// src/parallel/par_ops.rs

pub fn par_map_with_fallback<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync,
{
    // 检查是否在并行上下文中
    if is_in_parallel_context() {
        // 嵌套并行：强制使用串行
        return tensor.map(&f);
    }

    // 非嵌套：正常并行执行
    par_map(tensor, f)
}
```

**两种策略比较**：

| 策略 | 优点 | 缺点 |
|------|------|------|
| Panic（推荐） | 明确暴露问题，强制用户修复 | 可能破坏兼容性 |
| 自动回退 | 不会 panic，更宽容 | 隐藏性能问题，可能导致意外行为 |

**推荐策略**：开发阶段使用 panic，帮助发现问题；生产环境可通过 feature gate 切换到自动回退。

### 7.5 配置嵌套并行行为

```rust
// src/parallel/mod.rs

/// 嵌套并行处理策略。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedParallelismPolicy {
    /// 检测到嵌套时 panic。
    Panic,
    /// 检测到嵌套时自动回退到串行。
    FallbackToSequential,
}

thread_local! {
    static NESTED_PARALLELISM_POLICY: Cell<NestedParallelismPolicy> = 
        Cell::new(NestedParallelismPolicy::Panic);
}

/// 获取当前嵌套并行策略。
pub fn get_nested_parallelism_policy() -> NestedParallelismPolicy {
    NESTED_PARALLELISM_POLICY.with(|cell| cell.get())
}

/// 设置嵌套并行策略。
pub fn set_nested_parallelism_policy(policy: NestedParallelismPolicy) {
    NESTED_PARALLELISM_POLICY.with(|cell| cell.set(policy));
}
```

---

## 8. 线程池管理

### 8.1 默认全局线程池

rayon 默认使用全局线程池，线程数等于 CPU 核心数。

```rust
// src/parallel/mod.rs

/// 获取默认线程数。
///
/// 等同于 `rayon::current_num_threads()`。
pub fn default_num_threads() -> usize {
    rayon::current_num_threads()
}
```

### 8.2 自定义线程池

```rust
// src/parallel/mod.rs

use rayon::ThreadPool;

/// 自定义线程池包装。
///
/// 提供对 rayon 线程池的封装，支持在指定线程池上执行并行操作。
pub struct ParallelPool {
    inner: ThreadPool,
}

impl ParallelPool {
    /// 创建新的线程池。
    ///
    /// # 参数
    ///
    /// * `num_threads` - 线程数
    ///
    /// # 示例
    ///
    /// ```
    /// let pool = ParallelPool::new(4)?;
    /// let result = pool.install(|| tensor.par_sum());
    /// ```
    pub fn new(num_threads: usize) -> Result<Self, PoolInitError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PoolInitError::BuildFailed(e.to_string()))?;

        Ok(ParallelPool { inner: pool })
    }

    /// 在此线程池上执行闭包。
    ///
    /// # 示例
    ///
    /// ```
    /// let pool = ParallelPool::new(4)?;
    /// let sum = pool.install(|| {
    ///     tensor.par_map(|x| x * 2).par_sum()
    /// });
    /// ```
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.install(op)
    }

    /// 获取线程数。
    pub fn num_threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}

/// 线程池初始化错误。
#[derive(Debug, Clone)]
pub enum PoolInitError {
    /// 构建失败。
    BuildFailed(String),
}
```

### 8.3 使用自定义线程池执行并行操作

```rust
// src/parallel/par_ops.rs

impl ParallelPool {
    /// 在此线程池上执行并行映射。
    pub fn par_map<A, B, D, F>(&self, tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
    where
        D: Dimension,
        A: Element + Send + Sync,
        B: Element + Send,
        F: Fn(&A) -> B + Sync,
    {
        self.install(|| par_map(tensor, f))
    }

    /// 在此线程池上执行并行归约。
    pub fn par_reduce<A, D, F, ID>(
        &self,
        tensor: &Tensor<A, D>,
        identity: ID,
        op: F,
    ) -> A
    where
        D: Dimension,
        A: Element + Send + Sync,
        F: Fn(A, A) -> A + Sync,
        ID: Fn() -> A + Sync + Clone,
    {
        self.install(|| par_reduce(tensor, identity, op))
    }
}
```

### 8.4 全局线程池配置

```rust
// src/parallel/mod.rs

/// 配置全局 rayon 线程池。
///
/// 应在程序启动时调用。
///
/// # 参数
///
/// * `num_threads` - 线程数
///
/// # 示例
///
/// ```
/// // 在 main 函数开始时
/// xenon::parallel::configure_global_pool(8)?;
/// ```
pub fn configure_global_pool(num_threads: usize) -> Result<(), PoolInitError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| PoolInitError::BuildFailed(e.to_string()))
}
```

---

## 9. Padding 安全

### 9.1 问题背景

Xenon 支持在主维度添加 padding 以保证 SIMD 对齐。Padding 字节不属于任何逻辑元素，必须确保：

1. **禁止访问**：并行迭代时不能访问 padding 字节
2. **禁止分配**：并行分块时不能将 padding 字节分配给任何线程
3. **禁止暴露**：视图切片不能暴露 padding 字节为可访问元素

### 9.2 Padding 信息追踪

```rust
// src/parallel/mod.rs

/// Padding 信息。
///
/// 记录数组的 padding 区域。
#[derive(Clone, Debug, Default)]
pub struct PaddingInfo {
    /// 是否存在 padding。
    pub has_padding: bool,

    /// 逻辑元素数。
    pub logical_len: usize,

    /// 实际分配元素数（含 padding）。
    pub allocated_len: usize,

    /// Padding 元素数。
    pub padding_len: usize,
}

impl PaddingInfo {
    /// 从张量提取 padding 信息。
    pub fn from_tensor<A, D>(tensor: &Tensor<A, D>) -> Self
    where
        D: Dimension,
    {
        let logical_len = tensor.len();
        let allocated_len = tensor.storage().allocated_len();
        let padding_len = allocated_len.saturating_sub(logical_len);

        PaddingInfo {
            has_padding: padding_len > 0,
            logical_len,
            allocated_len,
            padding_len,
        }
    }

    /// 检查索引是否在 padding 区域。
    pub fn is_padding_index(&self, index: usize) -> bool {
        index >= self.logical_len
    }

    /// 获取安全迭代范围。
    pub fn safe_range(&self) -> std::ops::Range<usize> {
        0..self.logical_len
    }
}
```

### 9.3 并行分块时排除 Padding

```rust
// src/parallel/par_iter.rs

/// 计算安全分块（排除 padding）。
///
/// # 参数
///
/// * `tensor` - 输入张量
/// * `config` - 分块配置
///
/// # 返回
///
/// 分块范围列表，每个范围都在逻辑元素范围内。
pub fn compute_safe_chunks<A, D>(
    tensor: &Tensor<A, D>,
    config: &ChunkConfig,
) -> Vec<(usize, usize)>
where
    D: Dimension,
{
    let padding_info = PaddingInfo::from_tensor(tensor);
    let logical_len = padding_info.logical_len;

    // 使用逻辑长度计算分块
    if logical_len < config.min_chunk_size {
        return vec![(0, logical_len)];
    }

    let ideal_chunk_size = logical_len / config.target_chunks;
    let chunk_size = ideal_chunk_size.max(config.min_chunk_size);
    let num_chunks = (logical_len + chunk_size - 1) / chunk_size;

    (0..num_chunks)
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(logical_len); // 确保不超过逻辑长度
            (start, end)
        })
        .collect()
}
```

### 9.4 并行迭代器中的 Padding 检查

```rust
// src/parallel/par_iter.rs

impl<'a, A, D> ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    /// 创建安全的并行迭代器。
    ///
    /// 确保不访问 padding 字节。
    pub fn safe(base: TensorView<'a, A, D>) -> Self {
        let padding_info = PaddingInfo::from_tensor(&base.to_owned());
        debug_assert!(!padding_info.has_padding || base.is_contiguous(),
            "Padding with non-contiguous view should have been handled during view creation");

        ParElements::new(base)
    }
}

// 在 Producer 实现中验证
impl<'a, A, D> rayon::iter::plumbing::Producer for ParElementsProducer<'a, A, D>
where
    A: Element + Sync + Send + Clone,
    D: Dimension + Send + Sync,
{
    type Item = A;
    type IntoIter = std::vec::IntoIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        // 验证迭代范围不包含 padding
        let padding_info = PaddingInfo::from_tensor(&self.base.to_owned());
        let safe_range = padding_info.safe_range();

        // 安全地收集元素
        self.base
            .iter()
            .take(safe_range.end)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let padding_info = PaddingInfo::from_tensor(&self.base.to_owned());

        // 确保分割点不在 padding 区域
        let safe_index = index.min(padding_info.logical_len);

        // 分割逻辑...
        // 注意：这里需要实际实现分割逻辑
        // 简化示例，实际实现需要处理视图分割
        (self.clone(), self)
    }
}
```

### 9.5 设计保证

| 场景 | 处理方式 |
|------|----------|
| 连续数组含 padding | 分块使用逻辑长度，不超出 |
| 非连续数组 | 视图创建时已排除 padding |
| 切片视图 | 切片操作不暴露 padding |
| 并行归约 | 迭代器自动限制在逻辑范围内 |

---

## 10. Feature Gate 条件编译

### 10.1 parallel feature 强制依赖 std 的原因

```toml
# Cargo.toml
[features]
parallel = ["dep:rayon", "std"]
```

**原因分析**：

| 依赖 | 说明 |
|------|------|
| **rayon crate** | rayon 本身依赖 std，不支持 no_std |
| **线程原语** | 并行需要线程、互斥锁等，std 提供 |
| **AtomicUsize** | 全局阈值配置需要原子操作 |
| **thread_local!** | 嵌套并行检测需要线程局部存储 |

**替代方案（不采用）**：

| 方案 | 问题 |
|------|------|
| 使用 `portable-atomic` | rayon 仍需要 std |
| 使用 `spin` 替代 Mutex | 性能差，不推荐 |
| 自定义线程池 | 工程量大，不现实 |

**结论**：parallel feature 必须依赖 std，这是 rayon 的限制，也是合理的工程决策。

### 10.2 条件编译模式

```rust
// src/lib.rs

#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod parallel;
```

```rust
// src/tensor/mod.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 返回并行迭代器。
    ///
    /// 仅在 `parallel` feature 启用时可用。
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    pub fn par_iter(&self) -> parallel::ParElements<'_, A, D>
    where
        A: Send + Sync,
    {
        parallel::ParElements::new(self.view())
    }

    /// 串行迭代器（始终可用）。
    pub fn iter(&self) -> iter::Elements<'_, A, D> {
        iter::Elements::new(self.view())
    }
}
```

### 10.3 条件方法实现

```rust
// src/ops/elementwise.rs

impl<A, D> Tensor<A, D>
where
    D: Dimension,
    A: Element,
{
    /// 映射操作（串行版本）。
    pub fn map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element,
        F: Fn(&A) -> B,
    {
        // 串行实现
        // ...
    }

    /// 并行映射操作。
    ///
    /// 仅在 `parallel` feature 启用时可用。
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    pub fn par_map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element + Send,
        A: Send + Sync,
        F: Fn(&A) -> B + Sync,
    {
        parallel::par_map(self, f)
    }

    /// 自动选择并行/串行的映射操作。
    ///
    /// 根据数据规模自动选择最优执行路径。
    #[cfg(feature = "parallel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
    pub fn auto_map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element + Send,
        A: Send + Sync,
        F: Fn(&A) -> B + Sync,
    {
        if parallel::should_parallelize(self.len(), None, self.is_contiguous()) {
            self.par_map(f)
        } else {
            self.map(f)
        }
    }
}
```

### 10.4 文档配置

```rust
// src/lib.rs

#![cfg_attr(docsrs, feature(doc_cfg))]

// parallel 模块在文档中会显示 "This is supported on `parallel` only."
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod parallel;
```

---

## 11. 与其他模块的交互

### 11.1 与 iter 模块的交互

```
┌─────────────────────────────────────────────────────────────────┐
│                    iter 与 parallel 关系                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  iter 模块（串行）                 parallel 模块（并行）         │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │ Elements        │    参考      │ ParElements     │          │
│  │ AxisIter        │  ─────────▶  │ ParAxisIter     │          │
│  │ Zip             │              │ ParZip          │          │
│  └─────────────────┘              └─────────────────┘          │
│                                                                 │
│  接口一致性：                                                    │
│  - Elements::len() ≈ ParElements::len()                         │
│  - AxisIter::next() ≈ ParAxisIter 的 for_each                   │
│  - Zip 支持 broadcast ≈ ParZip 支持 broadcast                   │
│                                                                 │
│  转换关系：                                                      │
│  - ParElements 分块后每块内部使用 Elements 语义                  │
│  - 串行回退时直接使用 iter 模块                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 与 ops 模块的交互

```rust
// src/ops/mod.rs

#[cfg(feature = "parallel")]
use crate::parallel;

impl<A, D> Tensor<A, D>
where
    D: Dimension,
    A: Numeric,
{
    /// 逐元素加法（自动选择并行/串行）。
    pub fn add(&self, other: &Self) -> Self {
        #[cfg(feature = "parallel")]
        {
            if parallel::should_parallelize(self.len(), None, self.is_contiguous()) {
                return parallel::par_add(self, other).unwrap();
            }
        }

        // 串行实现
        self.add_sequential(other)
    }

    fn add_sequential(&self, other: &Self) -> Self {
        // 串行加法实现
        // ...
    }
}
```

### 11.3 与 simd 模块的交互

```
┌─────────────────────────────────────────────────────────────────┐
│                 parallel 与 simd 协同                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  执行路径决策：                                                  │
│                                                                 │
│  元素数 ≥ PARALLEL_THRESHOLD                                    │
│  │                                                              │
│  ├─ parallel 启用 且 simd 启用                                  │
│  │   └─→ 并行分块 → 每块内部 SIMD                               │
│  │                                                              │
│  ├─ parallel 启用 且 simd 未启用                                │
│  │   └─→ 并行分块 → 每块内部标量                                │
│  │                                                              │
│  ├─ parallel 未启用 且 simd 启用                                │
│  │   └─→ 单线程 SIMD                                           │
│  │                                                              │
│  └─ parallel 未启用 且 simd 未启用                              │
│      └─→ 单线程标量                                             │
│                                                                 │
│  示例代码：                                                      │
│                                                                 │
│  par_map 内部：                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ chunks.par_iter().for_each(|chunk| {                   │    │
│  │     if simd_enabled && is_contiguous {                 │    │
│  │         simd::vectorized_map(chunk, f);                │    │
│  │     } else {                                           │    │
│  │         scalar_map(chunk, f);                          │    │
│  │     }                                                  │    │
│  │ });                                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.4 与 layout 模块的交互

```rust
// src/parallel/par_iter.rs

impl<'a, A, D> ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    pub fn new(base: TensorView<'a, A, D>) -> Self {
        // 使用 layout 模块的信息决定分块策略
        let flags = base.layout_flags();

        let is_contiguous = flags.is_f_contiguous() || flags.is_c_contiguous();
        let has_zero_stride = flags.has_zero_stride();

        // 广播视图需要特殊处理
        if has_zero_stride {
            // 使用索引迭代而非直接内存访问
            // ...
        }

        ParElements {
            base,
            chunk_config: ChunkConfig::default(),
        }
    }
}
```

### 11.5 模块依赖图

```
┌─────────────────────────────────────────────────────────────────┐
│                     parallel 模块依赖                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                       ┌─────────┐                               │
│                       │ rayon   │  外部依赖                      │
│                       └────┬────┘                               │
│                            │                                    │
│  ┌─────────────────────────┼─────────────────────────┐         │
│  │                         │                         │         │
│  │  ┌──────────────────────┼──────────────────┐      │         │
│  │  │                parallel                │      │         │
│  │  │  ┌─────────────┬──────┴──────┬────────┐│      │         │
│  │  │  │   mod.rs    │  par_iter   │par_ops ││      │         │
│  │  │  └──────┬──────┴──────┬──────┴───┬────┘│      │         │
│  │  └─────────┼─────────────┼──────────┼─────┘      │         │
│  │            │             │          │            │         │
│  └────────────┼─────────────┼──────────┼────────────┘         │
│               │             │          │                      │
│               ▼             ▼          ▼                      │
│  ┌─────────────────────────────────────────────────┐          │
│  │  tensor │ layout │ element │ iter │ ops │ broadcast│       │
│  └─────────────────────────────────────────────────┘          │
│               内部模块依赖                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. 实现任务分解

### 任务 1：模块基础设施 (mod.rs)

**预计时间**: 10 分钟

**任务内容**:
- 创建 `src/parallel/mod.rs` 文件
- 定义全局阈值常量和原子变量
- 实现 `get_parallel_threshold()` 和 `set_parallel_threshold()`
- 实现线程局部阈值覆盖
- 实现 `effective_threshold()` 和 `should_parallelize()`
- 添加必要的 `#[cfg(feature = "parallel")]` 属性

**验收标准**:
- 编译通过（启用 parallel feature）
- 全局阈值可读写
- 阈值决策逻辑正确

---

### 任务 2：嵌套并行防护 (mod.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现线程局部并行深度计数器 `PARALLEL_DEPTH`
- 实现 `is_in_parallel_context()` 和 `parallel_depth()`
- 实现 `enter_parallel_context()` 和 `exit_parallel_context()`
- 实现 `ParallelGuard` RAII 守卫
- 定义 `NestedParallelismPolicy` 枚举
- 实现策略配置函数

**验收标准**:
- 嵌套并行检测正确
- Panic 或回退行为符合配置
- RAII 守卫正确进出

---

### 任务 3：分块策略 (par_iter.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `ChunkConfig` 结构体和默认值
- 实现 `compute_contiguous_chunks()` 连续分块
- 实现 `compute_strided_chunks()` 非连续分块
- 实现 `ensure_min_chunk_size()` 块大小保证
- 编写分块算法的单元测试

**验收标准**:
- 分块算法正确
- 最小块大小得到保证
- 边界情况处理正确

---

### 任务 4：ParElements 迭代器 (par_iter.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `ParElements<'a, A, D>` 结构体
- 实现 `new()` 和 `with_chunk_config()` 构造方法
- 实现 `ParallelIterator` trait
- 实现 `IndexedParallelIterator` trait
- 实现 `ParElementsProducer` 和 `Producer` trait
- 实现 `IntoParallelRefIterator` trait

**验收标准**:
- 迭代器编译通过
- 可在 `par_iter()` 中使用
- 元素产出正确

---

### 任务 5：ParAxisIter 和 ParZip (par_iter.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `ParAxisIter<'a, A, D>` 结构体
- 实现 `new()` 和轴迭代逻辑
- 实现 `ParallelIterator` trait
- 定义 `ParZip<'a, A, B, DA, DB>` 结构体
- 实现广播兼容性检查
- 实现并行 zip 迭代

**验收标准**:
- 轴迭代正确产出子视图
- Zip 支持广播
- 形状不兼容时返回错误

---

### 任务 6：par_map 和 par_reduce (par_ops.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `par_map<A, B, D, F>()` 函数
- 实现 `par_map_inplace<A, D, F>()` 函数
- 实现 `par_reduce<A, D, F, ID>()` 函数
- 实现 `par_sum()`, `par_prod()` 便捷函数
- 实现 `par_max()`, `par_min()` 函数
- 添加嵌套并行防护

**验收标准**:
- 映射和归约结果正确
- 自动回退到串行（小数组）
- 嵌套并行检测生效

---

### 任务 7：par_zip_with (par_ops.rs)

**预计时间**: 10 分钟

**任务内容**:
- 实现 `par_zip_with<A, B, C, DA, DB, DC, F>()` 函数
- 实现 `par_add()`, `par_sub()`, `par_mul()`, `par_div()` 便捷函数
- 处理广播逻辑
- 添加错误处理

**验收标准**:
- 二元运算正确
- 广播语义与串行版本一致
- 错误情况正确处理

---

### 任务 8：Trait 集成 (par_ops.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `ParallelElementwise<A, D>` trait
- 定义 `ParallelReduction<A, D>` trait
- 为 `Tensor<A, D>` 实现 trait
- 在 `src/tensor/mod.rs` 中添加条件方法 `par_iter()`
- 在 `src/ops/mod.rs` 中添加条件方法 `par_map()` 等

**验收标准**:
- trait 方法可调用
- 条件编译正确
- 文档注释完整

---

### 任务 9：线程池管理 (mod.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `ParallelPool` 结构体
- 实现 `new(num_threads)` 构造
- 实现 `install<OP, R>()` 方法
- 实现 `configure_global_pool()` 函数
- 定义 `PoolInitError` 错误类型
- 为 `ParallelPool` 添加 `par_map()`, `par_reduce()` 等方法

**验收标准**:
- 自定义线程池创建成功
- 在指定线程池上执行操作
- 全局线程池可配置

---

### 任务 10：Padding 安全 (mod.rs, par_iter.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `PaddingInfo` 结构体
- 实现 `from_tensor()` 构造方法
- 实现 `is_padding_index()` 检查
- 实现 `compute_safe_chunks()` 安全分块
- 在 `ParElements` 中使用安全分块
- 添加 padding 边界测试

**验收标准**:
- 不访问 padding 字节
- 分块不包含 padding 区域
- 边界测试通过

---

## 13. 设计决策记录

### 13.1 为什么选择 rayon？

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **rayon** | 工作窃取线程池 | 成熟稳定、API 友好、无数据竞争 | 依赖 std |
| crossbeam | 手动线程管理 | 灵活 | 需要更多代码 |
| tokio | 异步运行时 | 生态丰富 | 异步模型不匹配 |
| 自实现 | 手写线程池 | 完全控制 | 工程量大、易出错 |

**选择理由**：
1. **成熟稳定**：rayon 是 Rust 生态中最成熟的并行库
2. **安全保证**：编译期检查数据竞争
3. **API 友好**：与标准迭代器 API 一致
4. **性能优秀**：工作窃取算法负载均衡效果好

### 13.2 为什么默认阈值是 64K？

| 阈值 | 说明 |
|------|------|
| 1K | 太小，并行开销大于收益 |
| 16K | 小数组性能下降 |
| **64K** | 经验最佳值，平衡开销和收益 |
| 256K | 中等数组无法并行，性能损失 |
| 1M | 大部分操作无法并行 |

**选择理由**：
1. **经验调优**：64K 是业界常用值（ndarray 也使用类似阈值）
2. **开销抵消**：64K 元素足够抵消线程调度开销
3. **可配置**：用户可根据场景调整

### 13.3 为什么禁止嵌套并行？

| 策略 | 优点 | 缺点 |
|------|------|------|
| **禁止（panic）** | 明确暴露问题 | 可能破坏兼容性 |
| 允许 | 灵活 | 性能下降、死锁风险 |
| 自动回退 | 不 panic | 隐藏问题 |

**选择理由**：
1. **线程池饥饿**：rayon 使用固定大小线程池，嵌套导致线程不足
2. **性能下降**：嵌套并行通常比单层并行更慢
3. **明确问题**：panic 迫使用户重新设计算法

### 13.4 为什么 parallel 强制依赖 std？

| 原因 | 说明 |
|------|------|
| rayon 依赖 std | rayon 内部使用 std::thread 等 |
| 原子操作 | `AtomicUsize` 需要 std |
| 线程局部存储 | `thread_local!` 需要 std |

**结论**：这是合理的工程决策，并行计算本身就需要操作系统支持。

### 13.5 为什么使用 AtomicUsize 而非 RwLock？

| 方案 | 性能 | 适用场景 |
|------|------|----------|
| **AtomicUsize** | 极快（单指令） | 简单读写 |
| RwLock<usize> | 较慢（需要获取锁） | 复杂数据结构 |
| Mutex<usize> | 最慢（互斥访问） | 需要独占访问 |

**选择理由**：
1. **性能**：AtomicUsize 是最快的同步原语
2. **简单**：阈值读写不需要复杂数据结构
3. **足够**：Relaxed 语义对阈值配置足够

### 13.6 为什么最小块大小是 4K？

| 块大小 | 说明 |
|--------|------|
| 256 | 太小，调度开销大 |
| 1K | 勉强可用，但开销仍明显 |
| **4K** | 经验最佳值，平衡调度和负载 |
| 16K | 负载均衡变差 |
| 64K | 线程数多时效率下降 |

**选择理由**：
1. **缓存友好**：4K 元素通常能放入 L1/L2 缓存
2. **调度开销**：4K 足够抵消任务调度开销
3. **负载均衡**：4K 允许足够的块数实现负载均衡

---

## 附录 A：API 速查

### 全局配置

```rust
// 获取/设置全局阈值
xenon::parallel::get_parallel_threshold() -> usize
xenon::parallel::set_parallel_threshold(usize)
xenon::parallel::reset_parallel_threshold()

// 线程局部配置
xenon::parallel::get_local_threshold() -> Option<usize>
xenon::parallel::set_local_threshold(Option<usize>)
xenon::parallel::effective_threshold() -> usize

// 并行决策
xenon::parallel::should_parallelize(len, override, is_contiguous) -> bool
```

### 嵌套并行

```rust
// 检测
xenon::parallel::is_in_parallel_context() -> bool
xenon::parallel::parallel_depth() -> usize

// 策略
xenon::parallel::NestedParallelismPolicy::Panic
xenon::parallel::NestedParallelismPolicy::FallbackToSequential
xenon::parallel::set_nested_parallelism_policy(policy)
```

### 线程池

```rust
// 自定义线程池
let pool = xenon::parallel::ParallelPool::new(4)?;
pool.install(|| { /* 并行操作 */ });

// 全局配置
xenon::parallel::configure_global_pool(8)?;
```

### 并行迭代

```rust
// 元素迭代
tensor.par_iter()

// 轴迭代
tensor.par_axis_iter(Axis(0))

// Zip
ParZip::new(&a, &b)?
```

### 并行运算

```rust
// 映射
tensor.par_map(|x| x * 2)
tensor.par_map_with_threshold(|x| x * 2, 128_000)

// 归约
tensor.par_sum()
tensor.par_prod()
tensor.par_max()
tensor.par_min()
tensor.par_reduce(|| 0, |a, b| a + b)

// 二元运算
par_zip_with(&a, &b, |x, y| x + y)?
par_add(&a, &b)?
```

---

## 附录 B：性能基准

### 测试环境

- CPU: 8 核 16 线程
- 数据: 1M 元素 f64 数组
- 操作: 逐元素乘法

### 结果（示例）

| 配置 | 时间 (ms) | 加速比 |
|------|-----------|--------|
| 串行 | 12.5 | 1.0x |
| 并行 (4 线程) | 3.4 | 3.7x |
| 并行 (8 线程) | 1.8 | 6.9x |
| 并行 (16 线程) | 1.6 | 7.8x |

### 阈值影响

| 阈值 | 1K 元素 | 64K 元素 | 1M 元素 |
|------|---------|----------|---------|
| 1K | 并行（开销大） | 并行 | 并行 |
| 64K | 串行 | 并行 | 并行 |
| 256K | 串行 | 串行 | 并行 |

---

## 附录 C：常见问题

### Q1: 为什么小数组不并行？

**A**: 线程调度有固定开销（约 1-10 微秒）。对于小数组，并行开销大于计算收益。

### Q2: 如何临时禁用并行？

**A**: 设置极高的阈值：
```rust
set_local_threshold(Some(usize::MAX));
// 执行操作
set_local_threshold(None);
```

### Q3: 并行操作中出现 panic 怎么办？

**A**: rayon 会正确传播 panic，但可能导致部分结果丢失。建议在闭包中避免 panic。

### Q4: 如何调试并行问题？

**A**: 
1. 设置 `RUST_BACKTRACE=1` 查看调用栈
2. 使用 `RAYON_NUM_THREADS=1` 禁用并行进行对比
3. 检查是否有嵌套并行

---

*文档结束*
