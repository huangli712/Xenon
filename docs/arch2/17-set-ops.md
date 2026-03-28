# 集合操作模块设计

> 文档编号: 17 | 模块: `src/ops/set_ops.rs` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `03-element-types.md`, `08-error.md`
> 需求来源: `require-v18.md` §10.3.2, §12.1, §13.3

---

## 1. 模块定位

集合操作模块提供张量的**值域操作**能力——提取唯一值、统计分布、排序、查找非零元素、裁剪值域等。这些操作将张量数据视为一个集合（或序列）进行分析和变换，是科学计算中数据探索和预处理的核心工具。

### 设计目标

| 目标 | 体现 |
|------|------|
| 完整语义 | unique 支持 values/counts/inverse 三种变体；histogram 支持 bin 边界查询 |
| 类型安全 | bincount 仅接受整数类型（通过 trait bound 约束）；histogram 仅接受 `RealScalar` |
| 高效排序 | 默认 pdqsort（O(n log n)），整数小范围退化为计数排序（O(n + k)） |
| 零拷贝视角 | argwhere/nonzero 返回索引的 Owned 张量（不修改源数据） |
| 最小分配 | 输出张量精确分配所需空间，不预分配后截断 |

### 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 唯一值操作 | unique, unique_counts, unique_inverse | 去重（unique 本身即去重） |
| 分布统计 | bincount, histogram, histogram_bin_edges | 分位数计算（归约模块） |
| 排序 | sort, argsort | partial_sort, topk（未来扩展） |
| 索引查找 | argwhere, nonzero | take, compress（索引模块） |
| 值域裁剪 | clip | clamp（clip 的别名，不单独提供） |

---

## 2. 文件位置

```
src/ops/
├── mod.rs              # 模块入口，re-export 公共 API
├── element_wise.rs     # 逐元素运算
├── matrix.rs           # 矩阵运算
├── reduction.rs        # 归约运算
├── accumulate.rs       # 累积归约
└── set_ops.rs          # ★ 本模块：集合操作
```

在 `src/ops/mod.rs` 中的声明：

```rust
pub mod set_ops;

pub use crate::ops::set_ops::{
    unique, unique_counts, unique_inverse,
    bincount, bincount_with_weights,
    histogram, histogram_bin_edges,
    sort, sort_inplace,
    argsort,
    argwhere, nonzero,
    clip,
    SortOrder,
};
```

---

## 3. 依赖关系

### 3.1 本模块的依赖（上游）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `TensorBase`, `Tensor`, `TensorView` | `crate::tensor` | 输入/输出类型 |
| `Element`, `Numeric`, `RealScalar` | `crate::element` | 元素类型约束 |
| `Dimension`, `Ix1`, `IxDyn` | `crate::dimension` | 维度类型 |
| `Owned`, `RawStorage` | `crate::storage` | 存储类型与访问 |
| `LayoutFlags` | `crate::layout` | 布局标志 |
| `TensorError`, `Result` | `crate::error` | 错误处理 |
| `alloc::vec::Vec` | `alloc` | 索引收集与中间缓冲 |

### 3.2 依赖关系图

```
element ──→ tensor ──→ ops/set_ops.rs
error   ──→ tensor     │
layout  ──→ tensor     ├── alloc::vec::Vec
storage ──→ tensor     └── (无外部依赖)
dimension ─→ tensor
```

本模块位于依赖图上层，仅依赖 Phase 2 核心模块，不依赖其他 Phase 3 API 模块。

---

## 4. 公共 API 设计

### 4.1 SortOrder — 排序方向枚举

```rust
/// Sort order for `sort` and `argsort`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order (smallest first).
    Ascending,
    /// Descending order (largest first).
    Descending,
}
```

---

### 4.2 unique — 唯一值提取

```rust
/// Returns the sorted unique values of the flattened tensor.
///
/// The returned 1D tensor contains all distinct values from the input,
/// sorted in ascending order.
///
/// # Arguments
///
/// * `tensor` — The input tensor (any storage type, any dimensionality).
///
/// # Type Constraints
///
/// The element type `A` must implement `Element` + `Ord`.
/// All numeric types (`i8`–`i64`, `u8`–`u64`, `f32`, `f64`) satisfy this.
/// Note: `f32`/`f64` have `Ord` via total ordering (NaN sorts last).
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, unique};
/// let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3]);
/// let u = unique(&a);
/// assert_eq!(u.as_slice(), &[1, 2, 3]);
/// ```
///
/// # Performance
///
/// For integer types with small value ranges, a counting sort strategy is used
/// internally (O(n + k)). For general types, pdqsort is used (O(n log n)).
pub fn unique<S, A, D>(tensor: &TensorBase<S, D>) -> Tensor1<A>
where
    S: RawStorage<Elem = A>,
    A: Element + Ord,
    D: Dimension,
{
    // 1. Collect all elements into a Vec
    // 2. Sort using pdqsort (or counting sort for small-range integers)
    // 3. Deduplicate adjacent equal elements
    // 4. Construct Tensor1 from deduplicated Vec
    ...
}
```

### 4.3 unique_counts — 唯一值 + 出现次数

```rust
/// Returns the sorted unique values and their occurrence counts.
///
/// # Arguments
///
/// * `tensor` — The input tensor.
///
/// # Returns
///
/// A tuple `(values, counts)` where:
/// - `values`: sorted unique elements (`Tensor1<A>`)
/// - `counts`: occurrence count for each unique value (`Tensor1<usize>`)
///
/// # Postcondition
///
/// `values.len() == counts.len()`
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, unique_counts};
/// let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3]);
/// let (values, counts) = unique_counts(&a);
/// assert_eq!(values.as_slice(), &[1, 2, 3]);
/// assert_eq!(counts.as_slice(), &[2, 1, 2]);
/// ```
pub fn unique_counts<S, A, D>(tensor: &TensorBase<S, D>) -> (Tensor1<A>, Tensor1<usize>)
where
    S: RawStorage<Elem = A>,
    A: Element + Ord,
    D: Dimension,
{
    // 1. Sort + deduplicate (same as unique)
    // 2. Count runs of equal elements during deduplication
    // 3. Return (values, counts) as separate Tensor1
    ...
}
```

### 4.4 unique_inverse — 唯一值 + 反向索引

```rust
/// Returns the sorted unique values and the inverse indices.
///
/// The inverse indices map each element of the flattened input to its
/// position in the unique values array:
/// `values[inverse[i]] == input_flat[i]` for all valid `i`.
///
/// # Arguments
///
/// * `tensor` — The input tensor.
///
/// # Returns
///
/// A tuple `(values, inverse)` where:
/// - `values`: sorted unique elements (`Tensor1<A>`)
/// - `inverse`: index into `values` for each input element (`Tensor1<usize>`)
///
/// # Postcondition
///
/// - `inverse.len() == tensor.len()`
/// - `inverse[i] < values.len()` for all `i`
/// - `values[inverse[i]] == input_flat[i]`
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, unique_inverse};
/// let a = Tensor1::from_vec(vec![3, 1, 2, 1, 3]);
/// let (values, inverse) = unique_inverse(&a);
/// assert_eq!(values.as_slice(), &[1, 2, 3]);
/// assert_eq!(inverse.as_slice(), &[2, 0, 1, 0, 2]);
/// // Verify: values[2]=3, values[0]=1, values[1]=2, values[0]=1, values[2]=3 ✓
/// ```
pub fn unique_inverse<S, A, D>(tensor: &TensorBase<S, D>) -> (Tensor1<A>, Tensor1<usize>)
where
    S: RawStorage<Elem = A>,
    A: Element + Ord + Hash,
    D: Dimension,
{
    // 1. Collect (value, original_index) pairs
    // 2. Sort by value
    // 3. Deduplicate, building a HashMap<value, unique_index>
    // 4. Construct inverse array from original order
    // 5. Return (values, inverse)
    ...
}
```

**设计决策**：`unique_inverse` 需要 `Hash` bound 以构建 `HashMap` 进行 O(1) 反向查找。所有标准整数和浮点类型均实现 `Hash`。对于不支持 `Hash` 的自定义类型，用户可组合 `unique` + 手动查找。

---

### 4.5 bincount — 非负整数统计

```rust
/// Counts occurrences of each non-negative integer value.
///
/// For each index `i` in the output, `output[i]` equals the number of
/// times `i` appears in the flattened input tensor.
///
/// # Arguments
///
/// * `tensor` — Input tensor containing non-negative integer values.
/// * `minlength` — Minimum length of the output. If `max(input) + 1 < minlength`,
///   the output is padded with zeros to length `minlength`. Defaults to `0`.
///
/// # Type Constraints
///
/// Only integer types are supported: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`,
/// `u32`, `u64`. This is enforced by the `Integer` trait bound.
///
/// # Panics
///
/// Panics if any input value is negative (runtime check).
/// Panics if any input value exceeds `isize::MAX` (cannot be used as index).
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, bincount};
/// let a = Tensor1::from_vec(vec![0u32, 1, 1, 3, 2, 1]);
/// let counts = bincount(&a, 0);
/// assert_eq!(counts.as_slice(), &[1, 3, 1, 1]);
/// ```
pub fn bincount<S, A, D>(tensor: &TensorBase<S, D>, minlength: usize) -> Tensor1<usize>
where
    S: RawStorage<Elem = A>,
    A: Element + Integer,
    D: Dimension,
{
    // 1. Flatten: iterate all elements
    // 2. Find max value (determines output length)
    // 3. Allocate output of length max(max_val + 1, minlength)
    // 4. For each element: assert value >= 0, then output[value as usize] += 1
    // 5. Return Tensor1<usize>
    ...
}

/// Counts occurrences of non-negative integer values with weighted sums.
///
/// Like `bincount`, but each occurrence contributes its corresponding weight
/// instead of counting 1. The output type matches the weight type.
///
/// # Arguments
///
/// * `tensor` — Input tensor containing non-negative integer values.
/// * `weights` — Weight tensor, must have the same total element count as `tensor`.
/// * `minlength` — Minimum length of the output.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if `weights.len() != tensor.len()`.
///
/// # Panics
///
/// Same as `bincount`: panics on negative input values.
pub fn bincount_with_weights<S, A, W, D, D2>(
    tensor: &TensorBase<S, D>,
    weights: &TensorBase<S, D2>,
    minlength: usize,
) -> Result<Tensor1<W>>
where
    S: RawStorage<Elem = A>,
    A: Element + Integer,
    W: Element + Numeric,
    D: Dimension,
    D2: Dimension,
{
    ...
}
```

**`Integer` trait（marker trait，在 `element.rs` 中定义）**：

```rust
/// Marker trait for integer element types.
///
/// Implemented for: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
/// Used to constrain `bincount` inputs to integer types only.
pub trait Integer: Numeric + Ord + Hash {}
```

---

### 4.6 histogram — 直方图统计

```rust
/// Computes the histogram of a tensor.
///
/// # Arguments
///
/// * `tensor` — Input tensor with `RealScalar` elements.
/// * `bins` — Either a positive integer (number of equal-width bins) or
///   a `Tensor1<A>` of custom bin edges (must be sorted ascending,
///   length >= 2).
/// * `range` — A tuple `(min, max)` specifying the data range. Values outside
///   this range are not counted. Required when `bins` is an integer;
///   ignored when `bins` is a `Tensor1<A>`.
///
/// # Bin edge rules
///
/// All bins are half-open `[left, right)` except the last bin which is
/// closed `[left, right]`. This matches NumPy behavior.
///
/// # Returns
///
/// `Tensor1<usize>` of length `bins`, where each element is the count of
/// input values falling into that bin.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, histogram};
/// let data = Tensor1::from_vec(vec![1.0, 2.0, 2.5, 3.0, 4.0]);
/// let counts = histogram(&data, 3, (1.0, 4.0));
/// assert_eq!(counts.as_slice(), &[2, 1, 2]); // [1,2), [2,3), [3,4]
/// ```
pub fn histogram<S, A, D>(
    tensor: &TensorBase<S, D>,
    bins: HistogramBins<A>,
    range: Option<(A, A)>,
) -> Result<Tensor1<usize>>
where
    S: RawStorage<Elem = A>,
    A: RealScalar,
    D: Dimension,
{
    // 1. Compute bin edges (from bins parameter)
    // 2. Allocate output counts vector of length = number of bins
    // 3. For each element: binary search to find bin, increment count
    // 4. Return Tensor1<usize>
    ...
}

/// Returns the histogram counts and the bin edges.
///
/// Like `histogram`, but also returns the computed bin edges.
///
/// # Returns
///
/// A tuple `(counts, bin_edges)` where:
/// - `counts`: histogram counts (`Tensor1<usize>`)
/// - `bin_edges`: bin boundary values (`Tensor1<A>`), length = bins + 1
pub fn histogram_bin_edges<S, A, D>(
    tensor: &TensorBase<S, D>,
    bins: HistogramBins<A>,
    range: Option<(A, A)>,
) -> Result<(Tensor1<usize>, Tensor1<A>)>
where
    S: RawStorage<Elem = A>,
    A: RealScalar,
    D: Dimension,
{
    ...
}
```

**`HistogramBins` — bins 参数类型**：

```rust
/// Specification of histogram bins.
///
/// Either a fixed number of equal-width bins, or custom bin edges.
#[derive(Clone, Debug)]
pub enum HistogramBins<A> {
    /// Number of equal-width bins. Requires `range` parameter.
    Count(usize),
    /// Custom bin edges. Must be sorted ascending, length >= 2.
    /// The `range` parameter is ignored when custom edges are provided.
    Edges(Tensor1<A>),
}
```

---

### 4.7 sort — 排序

```rust
/// Returns a sorted copy of the flattened tensor.
///
/// The input is treated as a flat 1D sequence regardless of its dimensionality.
/// The returned tensor is always 1D and newly allocated.
///
/// # Arguments
///
/// * `tensor` — The input tensor (any storage type, any dimensionality).
/// * `order` — Sort direction (`Ascending` or `Descending`).
///
/// # Type Constraints
///
/// The element type `A` must implement `Element + Ord`.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, sort, SortOrder};
/// let a = Tensor1::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0]);
/// let sorted = sort(&a, SortOrder::Ascending);
/// assert_eq!(sorted.as_slice(), &[1.0, 1.0, 3.0, 4.0, 5.0]);
/// ```
pub fn sort<S, A, D>(tensor: &TensorBase<S, D>, order: SortOrder) -> Tensor1<A>
where
    S: RawStorage<Elem = A>,
    A: Element + Ord,
    D: Dimension,
{
    // 1. Collect elements into Vec
    // 2. Sort using pdqsort (unstable sort, preserves duplicates)
    // 3. Reverse if Descending
    // 4. Construct Tensor1
    ...
}

/// Sorts an owned 1D tensor in-place.
///
/// Modifies the tensor data directly without allocating a new tensor.
///
/// # Panics
///
/// Panics if the input is not 1D (future: support axis sorting).
pub fn sort_inplace<A>(tensor: &mut Tensor1<A>, order: SortOrder)
where
    A: Element + Ord,
{
    // 1. Get mutable slice of the data
    // 2. Sort in-place using pdqsort
    // 3. Reverse if Descending
    ...
}
```

---

### 4.8 argsort — 排序索引

```rust
/// Returns the indices that would sort the flattened tensor.
///
/// `output[i]` is the index of the `i`-th smallest (or largest) element
/// in the flattened input. Equivalent to NumPy's `np.argsort`.
///
/// # Arguments
///
/// * `tensor` — The input tensor.
/// * `order` — Sort direction.
///
/// # Returns
///
/// A `Tensor1<usize>` of length `tensor.len()`, containing the indices
/// that would sort the flattened input.
///
/// # Stability
///
/// Uses a stable sort so that equal elements preserve their original order.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, argsort, SortOrder};
/// let a = Tensor1::from_vec(vec![30, 10, 20]);
/// let indices = argsort(&a, SortOrder::Ascending);
/// assert_eq!(indices.as_slice(), &[1, 2, 0]); // 10@1, 20@2, 30@0
/// ```
pub fn argsort<S, A, D>(tensor: &TensorBase<S, D>, order: SortOrder) -> Tensor1<usize>
where
    S: RawStorage<Elem = A>,
    A: Element + Ord,
    D: Dimension,
{
    // 1. Create Vec<(value, original_index)>
    // 2. Sort by value using stable sort (preserves order of equals)
    // 3. Extract indices from sorted pairs
    // 4. Reverse if Descending
    // 5. Return Tensor1<usize>
    ...
}
```

---

### 4.9 argwhere — 非零元素索引矩阵

```rust
/// Returns the indices of non-zero elements as a 2D array.
///
/// Each row of the output is a multi-dimensional index into the input tensor.
/// The output has shape `(nonzero_count, ndim)`.
///
/// # Arguments
///
/// * `tensor` — The input tensor with elements that can be compared to zero.
///
/// # Type Constraints
///
/// The element type `A` must implement `Element` and have a concept of "zero".
/// Elements are considered non-zero if they differ from `A::zero()`.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor2, argwhere};
/// let a = Tensor2::from_vec(vec![0, 1, 0, 0, 0, 1], [2, 3]);
/// let indices = argwhere(&a);
/// // indices shape: (2, 2)
/// // indices[0] = [0, 1]  (value 1 at row 0, col 1)
/// // indices[1] = [1, 2]  (value 1 at row 1, col 2)
/// ```
pub fn argwhere<S, A, D>(tensor: &TensorBase<S, D>) -> Tensor2<usize>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    // 1. Iterate all elements with their multi-dimensional indices
    // 2. Collect indices where element != A::zero()
    // 3. Flatten into row-major 2D layout: shape (count, ndim)
    // 4. Return Tensor2<usize>
    ...
}
```

---

### 4.10 nonzero — 各维度非零索引元组

```rust
/// Returns a tuple of index arrays, one per dimension.
///
/// For a tensor of shape `(n0, n1, ..., nk)`, returns a tuple of `k+1`
/// 1D tensors, where the `d`-th tensor contains the `d`-th coordinate of
/// each non-zero element.
///
/// This is the per-dimension decomposition of `argwhere`.
///
/// # Arguments
///
/// * `tensor` — The input tensor.
///
/// # Returns
///
/// A `Vec<Tensor1<usize>>` of length `ndim`. The `d`-th element is a 1D
/// tensor of the `d`-th axis indices for all non-zero elements.
/// All returned tensors have the same length.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor2, nonzero};
/// let a = Tensor2::from_vec(vec![0, 1, 0, 0, 0, 1], [2, 3]);
/// let indices = nonzero(&a);
/// assert_eq!(indices.len(), 2); // 2 dimensions
/// assert_eq!(indices[0].as_slice(), &[0, 1]); // row indices
/// assert_eq!(indices[1].as_slice(), &[1, 2]); // col indices
/// ```
pub fn nonzero<S, A, D>(tensor: &TensorBase<S, D>) -> Vec<Tensor1<usize>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    // 1. Iterate all elements with multi-dim indices
    // 2. For each non-zero element, push its index[d] into buffer[d]
    // 3. Convert each buffer into Tensor1<usize>
    // 4. Return Vec<Tensor1<usize>>
    ...
}
```

---

### 4.11 clip — 值域裁剪

```rust
/// Clamps tensor values to the range `[min_val, max_val]`.
///
/// For each element `x` in the input:
/// - If `x < min_val`, the output is `min_val`
/// - If `x > max_val`, the output is `max_val`
/// - Otherwise, the output is `x`
///
/// # Arguments
///
/// * `tensor` — The input tensor.
/// * `min_val` — The lower bound (inclusive).
/// * `max_val` — The upper bound (inclusive).
///
/// # Panics
///
/// Panics if `min_val > max_val` (degenerate range).
///
/// # Type Constraints
///
/// The element type must implement `Element + PartialOrd`.
/// All numeric types satisfy this.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor1, clip};
/// let a = Tensor1::from_vec(vec![1.0, 5.0, 3.0, -1.0, 10.0]);
/// let b = clip(&a, 0.0, 5.0);
/// assert_eq!(b.as_slice(), &[1.0, 5.0, 3.0, 0.0, 5.0]);
/// ```
pub fn clip<S, A, D>(
    tensor: &TensorBase<S, D>,
    min_val: A,
    max_val: A,
) -> Tensor<A, D>
where
    S: RawStorage<Elem = A>,
    A: Element + PartialOrd,
    D: Dimension,
{
    assert!(min_val <= max_val, "clip: min_val ({min_val:?}) must be <= max_val ({max_val:?})");
    // 1. Allocate output tensor with same shape
    // 2. For each element: if x < min → min, else if x > max → max, else x
    // 3. Return owned tensor
    ...
}
```

---

## 5. 内部实现设计

### 5.1 排序算法选择

排序是 `unique`、`sort`、`argsort` 的核心操作。根据元素类型选择不同策略：

| 策略 | 适用条件 | 复杂度 | 说明 |
|------|----------|--------|------|
| **pdqsort** | 通用（所有 `Ord` 类型） | O(n log n) | Rust 标准库 `slice::sort_unstable` 的底层算法 |
| **计数排序** | 整数类型，值域范围 ≤ 4×n | O(n + k) | u8/i8 始终满足；大范围整数退化为 pdqsort |
| **稳定排序** | argsort 专用 | O(n log n) | 使用 `slice::sort_by` 保持等值元素的原序 |

```rust
// Internal dispatch: choose sort strategy based on type and value range
fn sort_dispatch<A: Element + Ord>(data: &mut [A]) {
    // For u8/i8: always counting sort (range ≤ 256)
    // For other integers: counting sort if range ≤ 4 * data.len()
    // Otherwise: pdqsort via slice::sort_unstable
    ...
}

// Counting sort for small-range integers
fn counting_sort<A: Element + Integer>(data: &mut [A]) {
    // 1. Find min and max
    // 2. Allocate count array of size (max - min + 1)
    // 3. Count occurrences
    // 4. Write back sorted values
    ...
}
```

**设计决策**：

- 使用 `sort_unstable` 而非 `sort`：unique 和 sort 不需要保序性，unstable sort 更快且内存更友好
- argsort 使用 `sort_by`（stable）：保证等值元素保持输入顺序，与 NumPy 行为一致
- 计数排序阈值 4×n：经验值，避免值域过大时内存浪费

### 5.2 Histogram 分箱策略

```rust
// Internal: compute bin edges from bins parameter
fn compute_bin_edges<A: RealScalar>(
    bins: &HistogramBins<A>,
    range: Option<(A, A)>,
    data_min: A,
    data_max: A,
) -> Result<Vec<A>> {
    match bins {
        HistogramBins::Count(n) => {
            let (lo, hi) = range.unwrap_or((data_min, data_max));
            if n == 0 {
                return Err(TensorError::InvalidShape {
                    source_elements: 0,
                    target_shape: vec![0],
                    target_elements: 0,
                });
            }
            let bin_width = (hi - lo) / A::from(n).unwrap();
            Ok((0..=*n).map(|i| lo + A::from(i).unwrap() * bin_width).collect())
        }
        HistogramBins::Edges(edges) => {
            // Validate: sorted ascending, length >= 2
            let slice = edges.as_slice();
            if slice.len() < 2 {
                return Err(TensorError::InvalidShape {
                    source_elements: slice.len(),
                    target_shape: vec![2],
                    target_elements: 2,
                });
            }
            for w in slice.windows(2) {
                if w[0] >= w[1] {
                    return Err(TensorError::InvalidShape {
                        source_elements: 0,
                        target_shape: vec![],
                        target_elements: 0,
                    });
                }
            }
            Ok(slice.to_vec())
        }
    }
}

// Internal: binary search for bin assignment
fn find_bin<A: RealScalar>(value: A, edges: &[A]) -> Option<usize> {
    // Last bin is closed [left, right], others are [left, right)
    if value < edges[0] || value > edges[edges.len() - 1] {
        return None; // Out of range
    }
    // Use partition_point for O(log n) bin lookup
    let idx = edges.partition_point(|&e| e <= value) - 1;
    let last_bin = edges.len() - 2;
    Some(idx.min(last_bin))
}
```

### 5.3 内存分配策略

| 操作 | 输出大小 | 分配策略 |
|------|----------|----------|
| `unique` | ≤ n，精确去重后确定 | 先排序去重到 Vec，再精确分配 |
| `unique_counts` | 同 unique + counts | 两个 Tensor1，长度相同 |
| `unique_inverse` | 同 unique + n 长度 inverse | values 精确分配，inverse 预分配 n |
| `bincount` | max(input)+1 或 minlength | 零初始化 usize 数组 |
| `histogram` | bins 数量 | 零初始化 usize 数组 |
| `sort` | 恒等于 n | 收集所有元素后分配 |
| `argsort` | 恒等于 n | 预分配 n 个 usize |
| `argwhere` | 未知，先收集再分配 | 动态 Vec 收集后转为 Tensor2 |
| `nonzero` | 同 argwhere | 每维度一个动态 Vec |
| `clip` | 恒等于 n | 预分配同形状 Tensor |

### 5.4 unique 实现细节

```rust
fn unique_impl<S, A, D>(tensor: &TensorBase<S, D>) -> Vec<A>
where
    S: RawStorage<Elem = A>,
    A: Element + Ord,
    D: Dimension,
{
    // 1. Collect all elements
    let mut data: Vec<A> = tensor.iter().copied().collect();

    // 2. Sort (dispatches to pdqsort or counting sort)
    data.sort_unstable();

    // 3. Deduplicate (in-place)
    data.dedup();

    data
}
```

`unique_counts` 和 `unique_inverse` 共享排序逻辑但 dedup 策略不同：

```rust
fn unique_counts_impl<S, A, D>(tensor: &TensorBase<S, D>) -> (Vec<A>, Vec<usize>)
where
    S: RawStorage<Elem = A>,
    A: Element + Ord,
    D: Dimension,
{
    let mut data: Vec<A> = tensor.iter().copied().collect();
    data.sort_unstable();

    let mut values = Vec::new();
    let mut counts = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let run_start = i;
        while i < data.len() && data[i] == val {
            i += 1;
        }
        values.push(val);
        counts.push(i - run_start);
    }
    (values, counts)
}

fn unique_inverse_impl<S, A, D>(tensor: &TensorBase<S, D>) -> (Vec<A>, Vec<usize>)
where
    S: RawStorage<Elem = A>,
    A: Element + Ord + Hash,
    D: Dimension,
{
    let n = tensor.len();
    let mut data: Vec<(A, usize)> = tensor.iter().enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Stable sort by value
    data.sort_by(|a, b| a.0.cmp(&b.0));

    // Dedup, building value→index map
    let mut value_to_idx: HashMap<A, usize> = HashMap::new();
    let mut values = Vec::new();
    for (val, _) in &data {
        if !value_to_idx.contains_key(val) {
            let idx = values.len();
            value_to_idx.insert(*val, idx);
            values.push(*val);
        }
    }

    // Build inverse array in original order
    let mut inverse = vec![0usize; n];
    for (val, orig_idx) in &data {
        inverse[*orig_idx] = value_to_idx[val];
    }

    (values, inverse)
}
```

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 基础设施

- [ ] **T1: SortOrder 枚举 + Integer marker trait**
  - 文件: `src/ops/set_ops.rs:1-40`
  - 内容: 定义 `SortOrder` 枚举、`HistogramBins<A>` 枚举
  - 测试: `test_sort_order_equality`, `test_histogram_bins_variants`
  - 前置: element.rs 中已有 `Integer` trait（若没有则在此任务添加）
  - 预计: 10 min

### 6.2 排序操作

- [ ] **T2: sort（只读排序，返回新张量）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `sort<S, A, D>` 实现——收集元素 → pdqsort → 构造 Tensor1
  - 测试: `test_sort_ascending`, `test_sort_descending`, `test_sort_empty`, `test_sort_2d_input`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: sort_inplace（原地排序 1D 张量）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `sort_inplace<A>` 实现——获取可变切片 → 原地排序
  - 测试: `test_sort_inplace_basic`, `test_sort_inplace_single_element`
  - 前置: T2
  - 预计: 5 min

- [ ] **T4: argsort（排序索引）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `argsort<S, A, D>` 实现——创建 (value, index) 对 → 稳定排序 → 提取 index
  - 测试: `test_argsort_basic`, `test_argsort_ties_stable`, `test_argsort_descending`, `test_argsort_empty`
  - 前置: T2
  - 预计: 10 min

### 6.3 唯一值操作

- [ ] **T5: unique（唯一值提取）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `unique<S, A, D>` 实现——收集 → 排序 → dedup → Tensor1
  - 测试: `test_unique_basic`, `test_unique_empty`, `test_unique_all_same`, `test_unique_float`, `test_unique_preserves_type`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6: unique_counts（唯一值 + 计数）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `unique_counts<S, A, D>` 实现——排序 → run-length encoding
  - 测试: `test_unique_counts_basic`, `test_unique_counts_lengths_match`, `test_unique_counts_empty`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: unique_inverse（唯一值 + 反向索引）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `unique_inverse<S, A, D>` 实现——排序 → HashMap 建立 value→index → 构造 inverse
  - 测试: `test_unique_inverse_basic`, `test_unique_inverse_reconstruct`, `test_unique_inverse_all_unique`
  - 前置: T5
  - 预计: 10 min

### 6.4 统计操作

- [ ] **T8: bincount（无权重版本）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `bincount<S, A, D>` 实现——找 max → 分配 → 计数
  - 测试: `test_bincount_basic`, `test_bincount_minlength`, `test_bincount_empty`, `test_bincount_panics_negative`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9: bincount_with_weights（带权重版本）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `bincount_with_weights<S, A, W, D, D2>` 实现——带权重求和
  - 测试: `test_bincount_weights_basic`, `test_bincount_weights_shape_mismatch`
  - 前置: T8
  - 预计: 10 min

- [ ] **T10: histogram + histogram_bin_edges**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `histogram` 和 `histogram_bin_edges` 实现——计算 bin edges → 遍历计数
  - 测试: `test_histogram_equal_bins`, `test_histogram_custom_edges`, `test_histogram_out_of_range`, `test_histogram_empty`
  - 前置: T1
  - 预计: 10 min

### 6.5 索引查找操作

- [ ] **T11: argwhere（非零索引矩阵）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `argwhere<S, A, D>` 实现——遍历 → 收集非零索引 → Tensor2
  - 测试: `test_argwhere_basic`, `test_argwhere_all_zero`, `test_argwhere_all_nonzero`, `test_argwhere_1d`
  - 前置: T1
  - 预计: 10 min

- [ ] **T12: nonzero（各维度索引元组）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `nonzero<S, A, D>` 实现——遍历 → 按维度分离索引 → Vec<Tensor1>
  - 测试: `test_nonzero_basic`, `test_nonzero_all_zero`, `test_nonzero_consistent_with_argwhere`
  - 前置: T11
  - 预计: 10 min

### 6.6 值域操作

- [ ] **T13: clip（值域裁剪）**
  - 文件: `src/ops/set_ops.rs`
  - 内容: `clip<S, A, D>` 实现——分配输出 → 逐元素 min(max(x, lo), hi)
  - 测试: `test_clip_basic`, `test_clip_no_change`, `test_clip_all_below`, `test_clip_all_above`, `test_clip_panics_inverted_range`, `test_clip_float_nan`
  - 前置: T1
  - 预计: 10 min

### 6.7 集成与文档

- [ ] **T14: mod.rs re-export + lib.rs 集成**
  - 文件: `src/ops/mod.rs`, `src/lib.rs`
  - 内容: 模块声明 + re-export 所有公共 API
  - 测试: `use xenon::{unique, sort, histogram, ...}` 编译通过
  - 前置: T1–T13
  - 预计: 5 min

- [ ] **T15: 文档注释审查与 `cargo doc` 验证**
  - 文件: `src/ops/set_ops.rs` 全文件
  - 内容: 所有 pub fn/method/enum/type 补充 doc comment
  - 测试: `cargo doc --no-deps` 无警告
  - 前置: T14
  - 预计: 10 min

---

## 7. 测试计划

### 7.1 单元测试（`src/ops/set_ops.rs` 内 `#[cfg(test)] mod tests`）

#### sort 相关

| 测试函数 | 验证内容 |
|---------|---------|
| `test_sort_ascending` | `[3, 1, 2]` → `[1, 2, 3]` |
| `test_sort_descending` | `[3, 1, 2]` → `[3, 2, 1]` |
| `test_sort_empty` | 空张量 → 空输出 |
| `test_sort_single_element` | `[42]` → `[42]` |
| `test_sort_2d_input` | `[[3,1],[2,4]]` → `[1,2,3,4]`（展平排序） |
| `test_sort_duplicates` | `[3,1,1,2]` → `[1,1,2,3]` |
| `test_sort_float_with_nan` | NaN 排在最后（Ord 语义） |
| `test_argsort_basic` | `[30,10,20]` → `[1,2,0]` |
| `test_argsort_ties_stable` | 等值元素保持输入顺序 |
| `test_argsort_descending` | `[30,10,20]` 降序 → `[0,2,1]` |

#### unique 相关

| 测试函数 | 验证内容 |
|---------|---------|
| `test_unique_basic` | `[3,1,2,1,3]` → `[1,2,3]` |
| `test_unique_empty` | 空张量 → 空 Tensor1 |
| `test_unique_all_same` | `[5,5,5]` → `[5]` |
| `test_unique_preserves_type` | f64 精度不丢失 |
| `test_unique_counts_basic` | `[3,1,2,1,3]` → values `[1,2,3]`, counts `[2,1,2]` |
| `test_unique_counts_lengths_match` | `values.len() == counts.len()` |
| `test_unique_inverse_basic` | `[3,1,2,1,3]` → inverse `[2,0,1,0,2]` |
| `test_unique_inverse_reconstruct` | `values[inverse[i]] == input[i]` 对所有 i 成立 |
| `test_unique_inverse_all_unique` | 每个元素唯一时 inverse 为 permutation |

#### bincount 相关

| 测试函数 | 验证内容 |
|---------|---------|
| `test_bincount_basic` | `[0,1,1,3,2,1]` → `[1,3,1,1]` |
| `test_bincount_minlength` | minlength=5 但 max=3 → 长度 5 |
| `test_bincount_empty` | 空输入 → minlength=0 时长度 0 |
| `test_bincount_panics_negative` | `#[should_panic]` 负值输入 |
| `test_bincount_weights_basic` | 权重累加正确 |
| `test_bincount_weights_shape_mismatch` | weights 长度不匹配返回 Error |

#### histogram 相关

| 测试函数 | 验证内容 |
|---------|---------|
| `test_histogram_equal_bins` | 3 bins 在 [1,4] 范围内正确计数 |
| `test_histogram_custom_edges` | 自定义 bin edges |
| `test_histogram_out_of_range` | 超出范围的值不计入 |
| `test_histogram_empty` | 空张量 → 全零计数 |
| `test_histogram_bin_edges_count` | 等宽 bins 返回正确 edges |
| `test_histogram_bin_edges_custom` | 自定义 edges 原样返回 |
| `test_histogram_last_bin_closed` | 最后一个 bin 包含右端点 |

#### argwhere / nonzero

| 测试函数 | 验证内容 |
|---------|---------|
| `test_argwhere_basic` | 2D 非零索引正确 |
| `test_argwhere_all_zero` | 全零输入 → shape (0, ndim) |
| `test_argwhere_all_nonzero` | 全非零 → shape (n, ndim) |
| `test_argwhere_1d` | 1D 输入 → shape (count, 1) |
| `test_nonzero_basic` | 各维度索引数组正确 |
| `test_nonzero_all_zero` | 全零 → 空 Vec<Tensor1> |
| `test_nonzero_consistent_with_argwhere` | nonzero 结果与 argwhere 行转置一致 |

#### clip

| 测试函数 | 验证内容 |
|---------|---------|
| `test_clip_basic` | 部分元素被裁剪 |
| `test_clip_no_change` | 全在范围内 → 数据不变 |
| `test_clip_all_below` | 全低于 min → 全变 min |
| `test_clip_all_above` | 全高于 max → 全变 max |
| `test_clip_panics_inverted_range` | `#[should_panic]` min > max |
| `test_clip_float_nan` | NaN 比较为 false，保持 NaN |
| `test_clip_2d` | 多维张量裁剪保持形状 |
| `test_clip_integer` | 整数类型裁剪正确 |

### 7.2 集成测试

位于 `tests/set_ops.rs`：

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **跨存储类型** | `test_unique_view_input` | ViewRepr 输入正常工作 |
| | `test_sort_view_mut_input` | ViewMutRepr 输入正常工作 |
| | `test_arc_tensor_input` | ArcRepr 输入正常工作 |
| **大数组** | `test_sort_large_array` | 100K 元素排序正确 |
| | `test_bincount_large_range` | u32 大值域 bincount 正确 |
| **非连续输入** | `test_unique_transposed_input` | 转置视图作为输入正常 |
| | `test_clip_sliced_input` | 切片视图裁剪正确 |
| **组合操作** | `test_unique_then_bincount` | unique 结果作为 bincount 输入 |
| | `test_argsort_then_take` | argsort 结果用于索引 |
| **多维度** | `test_sort_3d_tensor` | 3D 张量展平排序 |
| | `test_argwhere_3d_tensor` | 3D 非零索引正确 |

### 7.3 边界测试

| 测试函数 | 边界条件 |
|---------|---------|
| `test_sort_single_element` | n=1 |
| `test_sort_all_equal` | 所有元素相同 |
| `test_bincount_u8_max` | u8 值 255 |
| `test_histogram_single_bin` | bins=1 |
| `test_histogram_two_values_one_bin` | 两个值落入同一个 bin |
| `test_argwhere_0d_tensor` | 0D 标量的 argwhere |
| `test_nonzero_bool_tensor` | bool 类型张量 |
| `test_clip_min_equals_max` | min==max，所有值变为同一值 |
| `test_unique_single_element` | n=1 → 输出也是 n=1 |
| `test_unique_large_duplicates` | 1000 个相同值 → 输出 1 个 |

---

## 附录 A：公共 API 签名速查

| 函数 | 输入约束 | 返回类型 |
|------|----------|----------|
| `unique` | `A: Element + Ord` | `Tensor1<A>` |
| `unique_counts` | `A: Element + Ord` | `(Tensor1<A>, Tensor1<usize>)` |
| `unique_inverse` | `A: Element + Ord + Hash` | `(Tensor1<A>, Tensor1<usize>)` |
| `bincount` | `A: Element + Integer` | `Tensor1<usize>` |
| `bincount_with_weights` | `A: Integer, W: Numeric` | `Result<Tensor1<W>>` |
| `histogram` | `A: RealScalar` | `Result<Tensor1<usize>>` |
| `histogram_bin_edges` | `A: RealScalar` | `Result<(Tensor1<usize>, Tensor1<A>)>` |
| `sort` | `A: Element + Ord` | `Tensor1<A>` |
| `sort_inplace` | `A: Element + Ord` | `()` |
| `argsort` | `A: Element + Ord` | `Tensor1<usize>` |
| `argwhere` | `A: Element` | `Tensor2<usize>` |
| `nonzero` | `A: Element` | `Vec<Tensor1<usize>>` |
| `clip` | `A: Element + PartialOrd` | `Tensor<A, D>` |

## 附录 B：类型支持矩阵

| 操作 | bool | i8–i64 | u8–u64 | f32 | f64 | Complex |
|------|------|--------|--------|-----|-----|---------|
| `unique` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (无 Ord) |
| `bincount` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `histogram` | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `sort` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (无 Ord) |
| `argsort` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (无 Ord) |
| `argwhere` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `nonzero` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `clip` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ (无 PartialOrd) |

## 附录 C：与 NumPy 行为对照

| 操作 | NumPy | Xenon | 差异 |
|------|-------|-------|------|
| `np.unique` | `return_index`, `return_inverse`, `return_counts` 参数 | 三个独立函数 | 功能等价，API 风格不同 |
| `np.bincount` | `weights`, `minlength` 参数 | 两个函数（有无权重） | 功能等价 |
| `np.histogram` | `bins` 为 int 或 array | `HistogramBins` 枚举 | 功能等价，类型安全 |
| `np.sort` | `axis` 参数 | 仅展平排序（v1） | v1 不支持沿轴排序 |
| `np.argsort` | `axis` 参数 | 仅展平 argsort（v1） | v1 不支持沿轴排序 |
| `np.argwhere` | 返回 (N, ndim) 数组 | 返回 `Tensor2<usize>` | 完全一致 |
| `np.nonzero` | 返回 tuple of arrays | 返回 `Vec<Tensor1<usize>>` | Vec 替代 tuple（动态维度数） |
| `np.clip` | `a_min`, `a_max` 参数 | `min_val`, `max_val` 参数 | 完全一致 |
