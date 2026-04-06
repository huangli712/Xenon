# 形状操作模块设计

> 文档编号: 13 | 模块: `src/shape/` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `02-dimension.md`, `07-tensor-core.md`, `08-error.md`
> 需求来源: `require-v18.md` §11

---

## 1. 模块定位

形状操作模块提供张量的**结构变换**能力——改变形状、排列轴、裁剪/扩展维度、拼接/拆分等。所有操作不改变元素值本身，仅重新解释数据的组织方式。

### 设计目标

| 目标 | 体现 |
|------|------|
| 零拷贝优先 | reshape、transpose、slice、squeeze、flip、swap_axes、move_axis、split/chunk、unstack 均返回视图 |
| 显式拷贝 | stack、concatenate、pad、repeat、tile 返回 Owned 数组，API 命名体现分配语义 |
| 类型保持 | 静态维度操作保持 `D` 不变；跨维度操作显式标注 `D2` 或使用 `IxDyn` |
| 安全边界检查 | 轴索引、形状兼容性通过 `Result` 返回；编程错误（如重复轴号）panic |
| 步长操纵 | transpose/flip/reshape/squeeze 通过步长和偏移实现 O(1) 零拷贝 |

### 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 形状变换 | reshape, transpose, slice, squeeze/expand_dims, flip, swap_axes, move_axis | 广播（由 `broadcast` 模块提供） |
| 拼接/拆分 | concatenate, stack, unstack, split, chunk | 高级索引（由 `indexing` 模块提供） |
| 数据填充/复制 | pad, repeat, tile | 逐元素运算（由 `ops/` 模块提供） |
| 维度转换 | flatten（视情况零拷贝或拷贝） | into_dimension（由 `tensor` 模块提供） |

---

## 2. 文件位置

```
src/shape/
├── mod.rs              # 模块入口，re-export 公共 API
├── reshape.rs          # reshape, flatten, into_shape
├── transpose.rs        # transpose, t(), swap_axes, move_axis
├── slice.rs            # slice (view-based), index_axis
├── squeeze.rs          # squeeze, expand_dims (unsqueeze)
├── split.rs            # split, chunk, unstack
├── pad.rs              # pad (constant, edge, reflect)
└── repeat.rs           # repeat, tile
```

在 `src/lib.rs` 中声明：

```rust
pub mod shape;

// re-export all shape operations
pub use crate::shape::{
    // reshape
    reshape, reshape_into, flatten,
    // transpose
    transpose, swap_axes, move_axis,
    // slice
    slice, index_axis,
    // squeeze
    squeeze, expand_dims,
    // flip
    flip, flipud, fliplr,
    // split
    split, chunk, unstack,
    // stack
    concatenate, stack,
    // pad
    pad, PadMode,
    // repeat
    repeat, tile,
};
```

---

## 3. 依赖关系

```
shape/
├── crate::dimension       # Dimension trait, Ix0~Ix6, IxDyn
├── crate::tensor          # TensorBase, Tensor, TensorView, TensorViewMut, ArcTensor
├── crate::layout          # LayoutFlags, Order, compute_strides, compute_layout_flags
├── crate::storage         # Owned, ViewRepr, ViewMutRepr, ArcRepr, Storage, RawStorage
├── crate::element         # Element trait
├── crate::error           # TensorError, Result
└── alloc::vec::Vec        # split/chunk/unstack 返回值
```

### 模块内部依赖

```
mod.rs
├── reshape.rs   (无内部依赖)
├── transpose.rs (无内部依赖)
├── slice.rs     (无内部依赖)
├── squeeze.rs   (无内部依赖)
├── split.rs     (依赖 slice.rs 的 index_axis)
├── pad.rs       (无内部依赖)
└── repeat.rs    (无内部依赖)
```

各子模块之间基本独立，仅 `split.rs` 通过调用 `index_axis`（来自 `slice.rs`）实现 `unstack`。所有子模块均仅依赖 Phase 2 核心模块。

---

## 4. 公共 API 设计

### 4.1 reshape — 形状重塑

> 需求：§11.1 零拷贝，§2.1 reshape 与维度互转，§6.1

**核心原则**：

- reshape 要求**连续**数据（F-contiguous 或 C-contiguous）
- 不改变维度类型 `D`（静态保持静态）
- 支持 `-1` 推断（仅 `IxDyn`）
- 返回**视图**，共享源数组底层存储

```rust
// ── reshape.rs ─────────────────────────────────────────────

/// Reshapes a tensor to the given shape without copying data.
///
/// The total number of elements must remain the same. The data must be
/// contiguous (F-order or C-order).
///
/// For `IxDyn`, a single dimension may be specified as `0` (interpreted
/// from context) — use `reshape_inferred` for `-1` inference.
///
/// # Arguments
///
/// * `tensor` — The source tensor (any storage type).
/// * `shape` — The target shape. Element count must match.
///
/// # Errors
///
/// Returns `TensorError::LayoutMismatch` if the data is not contiguous.
/// Returns `TensorError::InvalidShape` if the total element count differs.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, reshape};
/// let a = Tensor::<f64, _>::zeros([2, 3]);
/// let b = reshape(&a, Ix2(3, 2))?;
/// assert_eq!(b.shape(), &[3, 2]);
/// ```
pub fn reshape<S, A, D>(tensor: &TensorBase<S, D>, shape: D) -> Result<TensorView<'_, A, D>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    // Validate contiguous
    if !tensor.is_contiguous() {
        return Err(TensorError::LayoutMismatch {
            reason: "reshape requires contiguous data",
        });
    }
    // Validate element count
    if shape.size() != tensor.len() {
        return Err(TensorError::InvalidShape {
            expected: tensor.len(),
            actual: shape.size(),
        });
    }
    // Compute new strides in F-order
    let strides = shape.default_strides();
    let flags = LayoutFlags::compute(&shape, &strides, tensor.as_ptr(), tensor.offset());
    Ok(TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape,
        strides,
        offset: tensor.offset(),
        layout_flags: flags,
    })
}

/// Reshapes to a different dimension type.
///
/// Use this to convert between static and dynamic dimension types.
///
/// # Errors
///
/// Same as `reshape`, plus `TensorError::DimensionMismatch` if the target
/// dimension type requires a different number of axes.
pub fn reshape_into<S, A, D, D2>(
    tensor: &TensorBase<S, D>,
    shape: D2,
) -> Result<TensorView<'_, A, D2>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
    D2: Dimension,
{
    if !tensor.is_contiguous() {
        return Err(TensorError::LayoutMismatch {
            reason: "reshape requires contiguous data",
        });
    }
    if shape.size() != tensor.len() {
        return Err(TensorError::InvalidShape {
            expected: tensor.len(),
            actual: shape.size(),
        });
    }
    let strides = shape.default_strides();
    let flags = LayoutFlags::compute(&shape, &strides, tensor.as_ptr(), tensor.offset());
    Ok(TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape,
        strides,
        offset: tensor.offset(),
        layout_flags: flags,
    })
}

/// Reshapes with `-1` inference for one dimension (dynamic only).
///
/// Exactly one dimension in `shape` may be `usize::MAX` (sentinel for -1).
/// The inferred value is calculated from the total element count.
///
/// # Panics
///
/// Panics if more than one dimension uses the sentinel value.
/// Panics if the inferred dimension is not an integer.
///
/// # Errors
///
/// Same as `reshape`.
pub fn reshape_inferred<S, A>(
    tensor: &TensorBase<S, IxDyn>,
    shape: IxDyn,
) -> Result<TensorView<'_, A, IxDyn>>
where
    S: RawStorage<Elem = A>,
    A: Element,
{
    let inferred = infer_dimension(shape.slice(), tensor.len());
    let concrete = IxDyn::from_indices(&inferred);
    reshape(tensor, concrete)
}

/// Flattens the tensor to 1D (requires contiguous data).
///
/// Returns a 1D view sharing the underlying storage.
///
/// # Errors
///
/// Returns `TensorError::LayoutMismatch` if not contiguous.
pub fn flatten<S, A, D>(tensor: &TensorBase<S, D>) -> Result<TensorView<'_, A, Ix1>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    reshape_into(tensor, Ix1(tensor.len()))
}

/// Flattens dimensions `[start..end]` into a single dimension.
///
/// Dimensions outside the range are preserved. The flattened range must
/// cover contiguous elements in memory.
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if the range is out of bounds.
/// Returns `TensorError::LayoutMismatch` if the flattened dimensions are
/// not contiguous in memory.
pub fn flatten_range<S, A, D>(
    tensor: &TensorBase<S, D>,
    start: usize,
    end: usize,
) -> Result<TensorView<'_, A, IxDyn>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    // Validate range
    if start > end || end > tensor.ndim() {
        return Err(TensorError::InvalidAxis {
            axis: if start > end { start } else { end },
            ndim: tensor.ndim(),
        });
    }
    // Compute new shape: dims[..start] ++ [product(dims[start..end])] ++ dims[end..]
    let old_shape = tensor.shape();
    let flat_size: usize = old_shape[start..end].iter().product();
    let mut new_shape: Vec<usize> = Vec::with_capacity(old_shape.len() - (end - start) + 1);
    new_shape.extend_from_slice(&old_shape[..start]);
    new_shape.push(flat_size);
    new_shape.extend_from_slice(&old_shape[end..]);
    reshape_into(tensor, IxDyn::new(&new_shape))
}

/// Helper: infer a single `-1` dimension from total size.
///
/// Panics if more than one `-1` sentinel found.
fn infer_dimension(shape: &[usize], total: usize) -> Vec<usize> {
    const SENTINEL: usize = usize::MAX;
    let sentinel_count = shape.iter().filter(|&&d| d == SENTINEL).count();
    if sentinel_count > 1 {
        panic!("reshape: at most one dimension can be -1, found {sentinel_count}");
    }
    if sentinel_count == 0 {
        return shape.to_vec();
    }
    let known_product: usize = shape.iter().filter(|&&d| d != SENTINEL).product();
    if known_product == 0 {
        panic!("reshape: cannot infer dimension when known product is 0");
    }
    let inferred = total / known_product;
    if inferred * known_product != total {
        panic!("reshape: inferred dimension is not an integer");
    }
    shape.iter().map(|&d| if d == SENTINEL { inferred } else { d }).collect()
}
```

**方法形式（在 TensorBase 上）**：

除了自由函数外，`reshape` 和 `flatten` 也作为方法直接定义在各存储类型的 `impl` 块上（如 `07-tensor-core.md` §4.5–4.8 中已有的 `reshape` 方法）。本模块的自由函数版本提供统一的跨存储类型调用入口。

---

### 4.2 transpose — 轴排列

> 需求：§11.1 零拷贝

**核心原则**：

- 通过**步长重排**实现零拷贝视图
- `axes` 参数指定新的轴顺序（permutation）
- `t()` 为 2D 张量的快捷转置
- `swap_axes` 交换两个轴
- `move_axis` 将一个轴移动到新位置

```rust
// ── transpose.rs ───────────────────────────────────────────

/// Transposes the tensor by permuting axes.
///
/// Returns a view with permuted shape and strides. No data is copied.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axes` — The permutation. `axes[i] = j` means "new axis i was old axis j".
///   Must be a valid permutation of `0..ndim`.
///
/// # Panics
///
/// Panics if `axes.len() != ndim`.
/// Panics if `axes` contains duplicates or out-of-range values.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix3, transpose};
/// let a = Tensor::<f64, _>::zeros([2, 3, 4]);
/// let b = transpose(&a, [2, 0, 1]);
/// assert_eq!(b.shape(), &[4, 2, 3]);
/// ```
pub fn transpose<S, A, D>(
    tensor: &TensorBase<S, D>,
    axes: impl AsRef<[usize]>,
) -> TensorView<'_, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let axes = axes.as_ref();
    let ndim = tensor.ndim();
    assert_eq!(axes.len(), ndim, "transpose: axes length must match ndim");

    // Validate permutation
    let mut seen = vec![false; ndim];
    for &ax in axes {
        assert!(ax < ndim, "transpose: axis {ax} out of range for ndim={ndim}");
        assert!(!seen[ax], "transpose: duplicate axis {ax}");
        seen[ax] = true;
    }

    let old_shape = tensor.shape();
    let old_strides = tensor.strides();

    // Compute new shape and strides by permutation
    let mut new_shape = D::zeros(ndim);
    let mut new_strides = D::zeros(ndim);
    for (i, &ax) in axes.iter().enumerate() {
        new_shape.slice_mut()[i] = old_shape[ax];
        new_strides.slice_mut()[i] = old_strides[ax];
    }

    // Recompute layout flags
    let flags = LayoutFlags::compute(
        &new_shape,
        &new_strides,
        tensor.as_ptr(),
        tensor.offset(),
    );

    TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape: new_shape,
        strides: new_strides,
        offset: tensor.offset(),
        layout_flags: flags,
    }
}

/// 2D matrix transpose shorthand.
///
/// Equivalent to `transpose(&tensor, [1, 0])`.
///
/// # Panics
///
/// Panics if `tensor.ndim() != 2`.
pub fn t<S, A>(tensor: &TensorBase<S, Ix2>) -> TensorView<'_, A, Ix2>
where
    S: RawStorage<Elem = A>,
    A: Element,
{
    assert_eq!(tensor.ndim(), 2, "t() requires a 2D tensor");
    transpose(tensor, [1, 0])
}

/// Swaps two axes of the tensor.
///
/// Returns a view with axes `a` and `b` swapped. No data is copied.
///
/// # Panics
///
/// Panics if `a >= ndim` or `b >= ndim`.
/// Panics if `a == b` (no-op but indicates likely error).
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix3, swap_axes};
/// let a = Tensor::<f64, _>::zeros([2, 3, 4]);
/// let b = swap_axes(&a, 0, 2);
/// assert_eq!(b.shape(), &[4, 3, 2]);
/// ```
pub fn swap_axes<S, A, D>(
    tensor: &TensorBase<S, D>,
    a: usize,
    b: usize,
) -> TensorView<'_, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    assert!(a < ndim, "swap_axes: axis {a} out of range for ndim={ndim}");
    assert!(b < ndim, "swap_axes: axis {b} out of range for ndim={ndim}");
    assert_ne!(a, b, "swap_axes: axes must be different");

    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.swap(a, b);
    transpose(tensor, perm)
}

/// Moves an axis from one position to another.
///
/// All other axes shift to accommodate. No data is copied.
///
/// # Arguments
///
/// * `source` — The axis to move.
/// * `destination` — The new position for the axis.
///
/// # Panics
///
/// Panics if `source >= ndim` or `destination >= ndim`.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix3, move_axis};
/// // shape [2, 3, 4], move axis 0 to position 2 → [3, 4, 2]
/// let a = Tensor::<f64, _>::zeros([2, 3, 4]);
/// let b = move_axis(&a, 0, 2);
/// assert_eq!(b.shape(), &[3, 4, 2]);
/// ```
pub fn move_axis<S, A, D>(
    tensor: &TensorBase<S, D>,
    source: usize,
    destination: usize,
) -> TensorView<'_, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    assert!(source < ndim, "move_axis: source {source} out of range for ndim={ndim}");
    assert!(destination < ndim, "move_axis: destination {destination} out of range for ndim={ndim}");

    if source == destination {
        // No-op: return a view of the same shape
        return TensorBase {
            storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
            shape: tensor.shape.clone(),
            strides: tensor.strides.clone(),
            offset: tensor.offset(),
            layout_flags: tensor.layout_flags,
        };
    }

    // Build permutation: remove source, insert at destination
    let mut perm: Vec<usize> = (0..ndim).collect();
    let axis_val = perm.remove(source);
    perm.insert(destination, axis_val);
    transpose(tensor, perm)
}
```

---

### 4.3 slice — 视图切片

> 需求：§11.1 零拷贝，§11.2 index_axis 语义

**核心原则**：

- 返回**零拷贝视图**，通过调整 offset/shape/strides 实现
- `SliceInfo` 描述每个轴的切取范围
- 支持负步长（反转）、省略（取全部）
- `index_axis` 沿指定轴取单个切片，返回降维视图

```rust
// ── slice.rs ───────────────────────────────────────────────

/// Describes a range along one axis for slicing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceRange {
    /// Start index (inclusive). `None` means 0.
    pub start: Option<usize>,
    /// End index (exclusive). `None` means axis length.
    pub end: Option<usize>,
    /// Step. `None` means 1. Must be > 0 if present.
    pub step: Option<usize>,
}

impl SliceRange {
    /// Full range: equivalent to `..` (take all elements).
    pub const FULL: SliceRange = SliceRange {
        start: None,
        end: None,
        step: None,
    };

    /// Creates a new slice range.
    pub fn new(start: usize, end: usize) -> Self {
        SliceRange {
            start: Some(start),
            end: Some(end),
            step: None,
        }
    }

    /// Creates a slice range with step.
    pub fn new_with_step(start: usize, end: usize, step: usize) -> Self {
        assert!(step > 0, "slice step must be > 0");
        SliceRange {
            start: Some(start),
            end: Some(end),
            step: Some(step),
        }
    }
}

impl From<core::ops::Range<usize>> for SliceRange {
    fn from(r: core::ops::Range<usize>) -> Self {
        SliceRange::new(r.start, r.end)
    }
}

impl From<core::ops::RangeFrom<usize>> for SliceRange {
    fn from(r: core::ops::RangeFrom<usize>) -> Self {
        SliceRange {
            start: Some(r.start),
            end: None,
            step: None,
        }
    }
}

impl From<core::ops::RangeTo<usize>> for SliceRange {
    fn from(r: core::ops::RangeTo<usize>) -> Self {
        SliceRange {
            start: None,
            end: Some(r.end),
            step: None,
        }
    }
}

impl From<core::ops::RangeFull> for SliceRange {
    fn from(_: core::ops::RangeFull) -> Self {
        SliceRange::FULL
    }
}

/// A collection of slice ranges, one per axis.
#[derive(Clone, Debug)]
pub struct SliceInfo {
    ranges: Vec<SliceRange>,
}

impl SliceInfo {
    /// Creates a SliceInfo from per-axis ranges.
    pub fn new(ranges: Vec<SliceRange>) -> Self {
        SliceInfo { ranges }
    }

    /// Returns the number of axes in this slice descriptor.
    pub fn ndim(&self) -> usize {
        self.ranges.len()
    }

    /// Returns the slice ranges as a slice.
    pub fn ranges(&self) -> &[SliceRange] {
        &self.ranges
    }
}

/// Creates a zero-copy slice view of the tensor.
///
/// Each axis is sliced according to the corresponding `SliceRange`.
/// The returned view shares the underlying storage with the source.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `info` — Per-axis slice ranges. Length must equal `tensor.ndim()`.
///
/// # Errors
///
/// Returns `TensorError::InvalidShape` if `info.ndim() != tensor.ndim()`.
/// Returns `TensorError::IndexOutOfBounds` if any range exceeds axis length.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, shape::slice, shape::SliceRange};
/// let a = Tensor::<f64, _>::zeros([4, 5]);
/// let info = SliceInfo::new(vec![SliceRange::new(1, 3), SliceRange::FULL]);
/// let v = slice(&a, &info)?;
/// assert_eq!(v.shape(), &[2, 5]);
/// ```
pub fn slice<S, A, D>(
    tensor: &TensorBase<S, D>,
    info: &SliceInfo,
) -> Result<TensorView<'_, A, D>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    if info.ndim() != tensor.ndim() {
        return Err(TensorError::InvalidShape {
            expected: tensor.ndim(),
            actual: info.ndim(),
        });
    }

    let old_shape = tensor.shape();
    let old_strides = tensor.strides();

    let mut new_offset = tensor.offset();
    let mut new_shape = D::zeros(tensor.ndim());
    let mut new_strides = D::zeros(tensor.ndim());

    for axis in 0..tensor.ndim() {
        let axis_len = old_shape[axis];
        let range = &info.ranges()[axis];
        let step = range.step.unwrap_or(1);

        let start = range.start.unwrap_or(0);
        let end = range.end.unwrap_or(axis_len);

        // Validate bounds
        if start > end {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index: start,
                size: end,
            });
        }
        if end > axis_len {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index: end,
                size: axis_len,
            });
        }

        let dim_len = (end - start + step - 1) / step; // ceil((end - start) / step)
        new_offset += start * (old_strides[axis] as usize);
        new_shape.slice_mut()[axis] = dim_len;
        new_strides.slice_mut()[axis] = (old_strides[axis] * step as isize);
    }

    let flags = LayoutFlags::compute(
        &new_shape,
        &new_strides,
        tensor.as_ptr(),
        new_offset,
    );

    Ok(TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape: new_shape,
        strides: new_strides,
        offset: new_offset,
        layout_flags: flags,
    })
}

/// Selects a single index along the given axis, reducing dimensionality by 1.
///
/// Returns a view with `ndim - 1` dimensions that shares the source storage.
///
/// # Arguments
///
/// * `tensor` — The source tensor (must have ndim >= 1).
/// * `axis` — The axis to index along.
/// * `index` — The index along the axis.
///
/// # Panics
///
/// Panics if `ndim == 0` (cannot reduce below 0 dimensions).
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if `axis >= ndim`.
/// Returns `TensorError::IndexOutOfBounds` if `index >= axis_length`.
///
/// # BLAS compatibility
///
/// When indexing along the outermost axis of an F-contiguous 3D batch tensor,
/// the resulting 2D view preserves F-contiguity and LDA.
pub fn index_axis<S, A, D>(
    tensor: &TensorBase<S, D>,
    axis: usize,
    index: usize,
) -> Result<TensorView<'_, A, IxDyn>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if ndim == 0 {
        panic!("index_axis: cannot index a 0-dimensional tensor");
    }
    if axis >= ndim {
        return Err(TensorError::InvalidAxis { axis, ndim });
    }
    let axis_len = tensor.shape()[axis];
    if index >= axis_len {
        return Err(TensorError::IndexOutOfBounds {
            axis,
            index,
            size: axis_len,
        });
    }

    // Compute offset for the selected index
    let extra_offset = tensor.strides()[axis] * index as isize;
    let new_offset = tensor.offset() + extra_offset as usize;

    // Build new shape and strides without the indexed axis
    let old_shape = tensor.shape();
    let old_strides = tensor.strides();

    let new_shape: Vec<usize> = old_shape.iter().enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .collect();
    let new_strides: Vec<isize> = old_strides.iter().enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &s)| s)
        .collect();

    let new_dim = IxDyn::new(&new_shape);
    let new_stride_dim = IxDyn::from_stride_vec(
        new_strides.iter().map(|&s| s as usize).collect()
    );

    let flags = LayoutFlags::compute(
        &new_dim,
        &new_stride_dim,
        tensor.as_ptr(),
        new_offset,
    );

    Ok(TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape: new_dim,
        strides: new_stride_dim,
        offset: new_offset,
        layout_flags: flags,
    })
}
```

---

### 4.4 squeeze / expand_dims — 维度增减

> 需求：§11.1 零拷贝（squeeze 通过步长操纵，expand_dims 插入长度1轴）

```rust
// ── squeeze.rs ─────────────────────────────────────────────

/// Removes axes of length 1 from the tensor.
///
/// Returns a view with reduced dimensionality. No data is copied.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axes` — The axes to remove. Each must have length 1.
///   If empty, all size-1 axes are removed.
///
/// # Panics
///
/// Panics if any specified axis has length != 1.
/// Panics if `axes` contains duplicates.
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if any axis is out of bounds.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix3, squeeze};
/// let a = Tensor::<f64, _>::zeros([2, 1, 4]);
/// let b = squeeze(&a, &[1])?;
/// assert_eq!(b.shape(), &[2, 4]);
/// ```
pub fn squeeze<S, A, D>(
    tensor: &TensorBase<S, D>,
    axes: &[usize],
) -> Result<TensorView<'_, A, IxDyn>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    let old_shape = tensor.shape();
    let old_strides = tensor.strides();

    // Determine which axes to remove
    let remove: Vec<usize> = if axes.is_empty() {
        // Remove all size-1 axes
        old_shape.iter().enumerate()
            .filter(|&(_, &s)| s == 1)
            .map(|(i, _)| i)
            .collect()
    } else {
        // Validate specified axes
        for &ax in axes {
            if ax >= ndim {
                return Err(TensorError::InvalidAxis { axis: ax, ndim });
            }
            if old_shape[ax] != 1 {
                panic!("squeeze: axis {ax} has length {} (expected 1)", old_shape[ax]);
            }
        }
        axes.to_vec()
    };

    // Check for duplicates
    let mut seen = vec![false; ndim];
    for &ax in &remove {
        assert!(!seen[ax], "squeeze: duplicate axis {ax}");
        seen[ax] = true;
    }

    // Build new shape/strides without removed axes
    let new_shape: Vec<usize> = old_shape.iter().enumerate()
        .filter(|(i, _)| !remove.contains(i))
        .map(|(_, &s)| s)
        .collect();
    let new_strides: Vec<isize> = old_strides.iter().enumerate()
        .filter(|(i, _)| !remove.contains(i))
        .map(|(_, &s)| s)
        .collect();

    let new_dim = IxDyn::new(&new_shape);
    let new_stride_dim = IxDyn::from_stride_vec(
        new_strides.iter().map(|&s| s as usize).collect()
    );

    let flags = LayoutFlags::compute(
        &new_dim,
        &new_stride_dim,
        tensor.as_ptr(),
        tensor.offset(),
    );

    Ok(TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape: new_dim,
        strides: new_stride_dim,
        offset: tensor.offset(),
        layout_flags: flags,
    })
}

/// Inserts a new axis of length 1 at the specified position.
///
/// Returns a view with increased dimensionality. No data is copied.
/// The new axis has stride 0 (or equal to the product of subsequent dimensions
/// in the default stride sense — the specific value doesn't matter since length is 1).
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axis` — Position where the new axis is inserted (0 <= axis <= ndim).
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if `axis > ndim`.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, expand_dims};
/// let a = Tensor::<f64, _>::zeros([3, 4]);
/// let b = expand_dims(&a, 1)?;
/// assert_eq!(b.shape(), &[3, 1, 4]);
/// ```
pub fn expand_dims<S, A, D>(
    tensor: &TensorBase<S, D>,
    axis: usize,
) -> Result<TensorView<'_, A, IxDyn>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if axis > ndim {
        return Err(TensorError::InvalidAxis { axis, ndim });
    }

    let old_shape = tensor.shape();
    let old_strides = tensor.strides();

    // Insert size-1 axis with a stride that matches memory order
    // For F-order: stride = product of shapes before this axis
    // For C-order: stride = product of shapes after this axis
    // Since length is 1, stride value doesn't affect element access,
    // but we pick a sensible default for LayoutFlags computation.
    let new_axis_stride = if ndim == 0 {
        1
    } else if axis == 0 {
        1 // outermost
    } else if axis == ndim {
        old_strides[ndim - 1] * old_shape[ndim - 1] as isize
    } else {
        old_strides[axis] * old_shape[axis] as isize
    };

    let mut new_shape: Vec<usize> = old_shape.to_vec();
    let mut new_strides_vec: Vec<isize> = old_strides.to_vec();
    new_shape.insert(axis, 1);
    new_strides_vec.insert(axis, new_axis_stride);

    let new_dim = IxDyn::new(&new_shape);
    let new_stride_dim = IxDyn::from_stride_vec(
        new_strides_vec.iter().map(|&s| s as usize).collect()
    );

    let flags = LayoutFlags::compute(
        &new_dim,
        &new_stride_dim,
        tensor.as_ptr(),
        tensor.offset(),
    );

    Ok(TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape: new_dim,
        strides: new_stride_dim,
        offset: tensor.offset(),
        layout_flags: flags,
    })
}
```

---

### 4.5 flip / flipud / fliplr — 翻转

> 需求：§11.1 零拷贝（通过负步长实现）

```rust
// ── transpose.rs (continued) ───────────────────────────────

/// Reverses the tensor along the specified axes.
///
/// Returns a view with negative strides for the flipped axes. No data is copied.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axes` — The axes to flip. Empty means flip all axes.
///
/// # Panics
///
/// Panics if any axis is out of bounds or duplicated.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, flip};
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
/// let b = flip(&a, &[0]);
/// // b reverses rows: [[3,4],[1,2]]
/// ```
pub fn flip<S, A, D>(
    tensor: &TensorBase<S, D>,
    axes: &[usize],
) -> TensorView<'_, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    let axes = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        // Validate
        for &ax in axes {
            assert!(ax < ndim, "flip: axis {ax} out of range for ndim={ndim}");
        }
        axes.to_vec()
    };

    let old_shape = tensor.shape();
    let old_strides = tensor.strides();

    let mut new_shape = D::zeros(ndim);
    let mut new_strides = D::zeros(ndim);
    let mut new_offset = tensor.offset();

    for axis in 0..ndim {
        new_shape.slice_mut()[axis] = old_shape[axis];
        if axes.contains(&axis) {
            // Negate stride and adjust offset to point to last element along this axis
            new_strides.slice_mut()[axis] = -old_strides[axis];
            new_offset += (old_shape[axis] - 1) * old_strides[axis] as usize;
        } else {
            new_strides.slice_mut()[axis] = old_strides[axis];
        }
    }

    let flags = LayoutFlags::compute(
        &new_shape,
        &new_strides,
        tensor.as_ptr(),
        new_offset,
    );

    TensorBase {
        storage: ViewRepr::new(unsafe { &*tensor.as_ptr() }),
        shape: new_shape,
        strides: new_strides,
        offset: new_offset,
        layout_flags: flags,
    }
}

/// Reverses the tensor along axis 0 (vertical flip).
///
/// Equivalent to `flip(&tensor, &[0])`.
pub fn flipud<S, A, D>(tensor: &TensorBase<S, D>) -> TensorView<'_, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    flip(tensor, &[0])
}

/// Reverses the tensor along axis 1 (horizontal flip).
///
/// Equivalent to `flip(&tensor, &[1])`.
///
/// # Panics
///
/// Panics if `ndim < 2`.
pub fn fliplr<S, A, D>(tensor: &TensorBase<S, D>) -> TensorView<'_, A, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    assert!(tensor.ndim() >= 2, "fliplr: requires at least 2 dimensions");
    flip(tensor, &[1])
}
```

---

### 4.6 split / chunk / unstack — 拆分

> 需求：§11.3, §11.4

```rust
// ── split.rs ───────────────────────────────────────────────

/// Splits the tensor along an axis at the given indices.
///
/// Returns N+1 views, where N is the number of split points.
/// Each view shares the source storage (zero-copy).
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axis` — The axis to split along.
/// * `indices` — Split points (exclusive end of each section).
///   Must be sorted in ascending order and within `[0, axis_len]`.
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if `axis >= ndim`.
/// Returns `TensorError::InvalidShape` if indices are not sorted or out of bounds.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, split};
/// let a = Tensor::<f64, _>::zeros([4, 6]);
/// let parts = split(&a, 1, &[2, 4])?;
/// assert_eq!(parts.len(), 3);
/// assert_eq!(parts[0].shape(), &[4, 2]);
/// assert_eq!(parts[1].shape(), &[4, 2]);
/// assert_eq!(parts[2].shape(), &[4, 2]);
/// ```
pub fn split<S, A, D>(
    tensor: &TensorBase<S, D>,
    axis: usize,
    indices: &[usize],
) -> Result<Vec<TensorView<'_, A, IxDyn>>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if axis >= ndim {
        return Err(TensorError::InvalidAxis { axis, ndim });
    }
    let axis_len = tensor.shape()[axis];

    // Validate indices
    for w in indices.windows(2) {
        if w[0] >= w[1] {
            return Err(TensorError::InvalidShape {
                expected: axis_len,
                actual: w[1],
            });
        }
    }
    if let Some(&last) = indices.last() {
        if last > axis_len {
            return Err(TensorError::InvalidShape {
                expected: axis_len,
                actual: last,
            });
        }
    }

    // Build section boundaries
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend_from_slice(indices);
    boundaries.push(axis_len);

    let mut results = Vec::with_capacity(boundaries.len() - 1);
    for w in boundaries.windows(2) {
        let start = w[0];
        let end = w[1];
        let mut info_ranges = vec![SliceRange::FULL; ndim];
        info_ranges[axis] = SliceRange::new(start, end);
        let info = SliceInfo::new(info_ranges);
        results.push(slice(tensor, &info)?);
    }
    Ok(results)
}

/// Splits the tensor along an axis into approximately equal chunks.
///
/// If the axis length is not evenly divisible by `n_chunks`, the first
/// `(axis_len % n_chunks)` chunks each get one extra element.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axis` — The axis to split along.
/// * `n_chunks` — Number of chunks. If 0, returns an empty Vec.
///   If `n_chunks > axis_len`, returns `axis_len` chunks of size 1.
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if `axis >= ndim`.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, chunk};
/// let a = Tensor::<f64, _>::zeros([4, 7]);
/// let chunks = chunk(&a, 1, 3)?;
/// assert_eq!(chunks.len(), 3);
/// assert_eq!(chunks[0].shape(), &[4, 3]); // 7 % 3 == 1, first chunk gets +1
/// assert_eq!(chunks[1].shape(), &[4, 2]);
/// assert_eq!(chunks[2].shape(), &[4, 2]);
/// ```
pub fn chunk<S, A, D>(
    tensor: &TensorBase<S, D>,
    axis: usize,
    n_chunks: usize,
) -> Result<Vec<TensorView<'_, A, IxDyn>>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if axis >= ndim {
        return Err(TensorError::InvalidAxis { axis, ndim });
    }
    if n_chunks == 0 {
        return Ok(Vec::new());
    }

    let axis_len = tensor.shape()[axis];

    if axis_len == 0 {
        // Empty axis: return n_chunks empty views
        let mut results = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            results.push(index_axis(tensor, axis, 0).ok().unwrap_or_else(|| {
                panic!("chunk: cannot index empty axis");
            }));
        }
        return Ok(results);
    }

    let effective_chunks = n_chunks.min(axis_len);
    let chunk_size = axis_len / effective_chunks;
    let remainder = axis_len % effective_chunks;

    // Compute split indices
    let mut indices = Vec::with_capacity(effective_chunks - 1);
    let mut offset = 0;
    for i in 0..effective_chunks {
        offset += chunk_size + if i < remainder { 1 } else { 0 };
        if i < effective_chunks - 1 {
            indices.push(offset);
        }
    }

    split(tensor, axis, &indices)
}

/// Splits the tensor along an axis into N views, each with ndim - 1.
///
/// Returns `axis_length` views. Equivalent to calling `index_axis(axis, i)`
/// for each `i` in `0..axis_length`.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `axis` — The axis to unstack along.
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if `axis >= ndim`.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix3, unstack};
/// let a = Tensor::<f64, _>::zeros([3, 4, 5]);
/// let parts = unstack(&a, 0)?;
/// assert_eq!(parts.len(), 3);
/// assert_eq!(parts[0].shape(), &[4, 5]);
/// ```
pub fn unstack<S, A, D>(
    tensor: &TensorBase<S, D>,
    axis: usize,
) -> Result<Vec<TensorView<'_, A, IxDyn>>>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if axis >= ndim {
        return Err(TensorError::InvalidAxis { axis, ndim });
    }
    let axis_len = tensor.shape()[axis];
    if axis_len == 0 {
        return Ok(Vec::new());
    }
    let mut results = Vec::with_capacity(axis_len);
    for i in 0..axis_len {
        results.push(index_axis(tensor, axis, i)?);
    }
    Ok(results)
}
```

---

### 4.7 concatenate / stack — 拼接

> 需求：§11.1 需拷贝

```rust
// ── split.rs (concatenate/stack are logically part of split.rs) ──

/// Concatenates tensors along an existing axis.
///
/// All tensors must have the same shape except along the concatenation axis.
/// Returns a new owned tensor containing the combined data.
///
/// # Arguments
///
/// * `tensors` — Slice of tensors to concatenate.
/// * `axis` — The axis along which to concatenate.
///
/// # Errors
///
/// Returns `TensorError::InvalidAxis` if `axis >= ndim`.
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible.
///
/// # Panics
///
/// Panics if `tensors` is empty.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, concatenate};
/// let a = Tensor::<f64, _>::zeros([2, 3]);
/// let b = Tensor::<f64, _>::zeros([2, 3]);
/// let c = concatenate(&[&a, &b], 0)?;
/// assert_eq!(c.shape(), &[4, 3]);
/// ```
pub fn concatenate<A, D>(
    tensors: &[&Tensor<A, D>],
    axis: usize,
) -> Result<Tensor<A, D>>
where
    A: Element + Clone,
    D: Dimension,
{
    assert!(!tensors.is_empty(), "concatenate: requires at least one tensor");

    let first = &tensors[0];
    let ndim = first.ndim();
    if axis >= ndim {
        return Err(TensorError::InvalidAxis { axis, ndim });
    }

    // Validate shapes
    let mut total_axis_len = first.shape()[axis];
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.ndim() != ndim {
            return Err(TensorError::ShapeMismatch {
                expected: first.shape().to_vec(),
                actual: t.shape().to_vec(),
            });
        }
        for ax in 0..ndim {
            if ax != axis && t.shape()[ax] != first.shape()[ax] {
                return Err(TensorError::ShapeMismatch {
                    expected: first.shape().to_vec(),
                    actual: t.shape().to_vec(),
                });
            }
        }
        total_axis_len += t.shape()[axis];
    }

    // Compute output shape
    let mut out_shape = D::zeros(ndim);
    for ax in 0..ndim {
        out_shape.slice_mut()[ax] = if ax == axis {
            total_axis_len
        } else {
            first.shape()[ax]
        };
    }

    // Allocate output
    let total_elems = out_shape.size();
    let mut out_storage = Owned::<A>::uninitialized(total_elems);
    let out_strides = out_shape.default_strides();

    // Copy data from each tensor
    let mut axis_offset = 0;
    for t in tensors {
        // For each element in t, compute its position in the output
        for (src_idx, elem) in t.iter().enumerate() {
            // Convert flat index to multi-dim index
            let mut multi_idx: Vec<usize> = Vec::with_capacity(ndim);
            let mut remaining = src_idx;
            for ax in 0..ndim {
                let stride = t.strides()[ax] as usize;
                let dim = t.shape()[ax];
                if ax == axis {
                    multi_idx.push(axis_offset + remaining / (stride.max(1)));
                } else {
                    multi_idx.push(remaining / (stride.max(1)));
                }
                remaining %= stride.max(1);
            }
            // Write to output using out_strides
            let mut dst_offset = 0;
            for ax in 0..ndim {
                dst_offset += multi_idx[ax] * (out_strides.slice()[ax]);
            }
            unsafe {
                out_storage.as_mut_ptr().add(dst_offset).write(elem.clone());
            }
        }
        axis_offset += t.shape()[axis];
    }

    let flags = LayoutFlags::compute(
        &out_shape,
        &out_strides,
        out_storage.as_ptr(),
        0,
    );
    Ok(unsafe { TensorBase::from_storage_unchecked(out_storage, out_shape, out_strides, 0) })
}

/// Stacks tensors along a new axis.
///
/// All tensors must have the same shape. A new axis of length `tensors.len()`
/// is inserted at the specified position.
///
/// # Arguments
///
/// * `tensors` — Slice of tensors to stack.
/// * `axis` — Position of the new axis (0 <= axis <= ndim).
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if shapes differ.
/// Returns `TensorError::InvalidAxis` if `axis > ndim`.
///
/// # Panics
///
/// Panics if `tensors` is empty.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, stack};
/// let a = Tensor::<f64, _>::zeros([3, 4]);
/// let b = Tensor::<f64, _>::zeros([3, 4]);
/// let c = stack(&[&a, &b], 0)?;
/// assert_eq!(c.shape(), &[2, 3, 4]);
/// ```
pub fn stack<A, D>(
    tensors: &[&Tensor<A, D>],
    axis: usize,
) -> Result<Tensor<A, IxDyn>>
where
    A: Element + Clone,
    D: Dimension,
{
    assert!(!tensors.is_empty(), "stack: requires at least one tensor");

    let first = &tensors[0];
    let ndim = first.ndim();
    if axis > ndim {
        return Err(TensorError::InvalidAxis { axis, ndim: ndim + 1 });
    }

    // Validate all shapes match
    for t in tensors.iter().skip(1) {
        if t.shape() != first.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: first.shape().to_vec(),
                actual: t.shape().to_vec(),
            });
        }
    }

    // Compute output shape: insert tensors.len() at axis position
    let n = tensors.len();
    let mut out_shape: Vec<usize> = first.shape().to_vec();
    out_shape.insert(axis, n);

    // Use expand_dims + concatenate approach
    let expanded: Vec<Tensor<A, IxDyn>> = tensors.iter()
        .map(|t| {
            let view = expand_dims(t, axis).unwrap();
            view.to_owned()
        })
        .collect();
    let refs: Vec<&Tensor<A, IxDyn>> = expanded.iter().collect();
    concatenate(&refs, axis)
}
```

---

### 4.8 pad — 填充

> 需求：§11.5

```rust
// ── pad.rs ─────────────────────────────────────────────────

/// Padding mode for the `pad` operation.
#[derive(Clone, Debug, PartialEq)]
pub enum PadMode<A> {
    /// Fill with a constant value.
    Constant(A),
    /// Repeat the edge elements.
    Edge,
    /// Mirror reflect (does not include the edge element itself).
    ///
    /// For a row `[a, b, c, d]` with `before=2`, the padded result starts
    /// as `[c, b, a, b, c, d, ...]`.
    Reflect,
}

/// A pair of padding widths for one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadWidth {
    /// Padding before the data on this axis.
    pub before: usize,
    /// Padding after the data on this axis.
    pub after: usize,
}

impl PadWidth {
    /// Creates a symmetric padding (equal before and after).
    pub fn symmetric(width: usize) -> Self {
        PadWidth { before: width, after: width }
    }

    /// Creates an asymmetric padding.
    pub fn new(before: usize, after: usize) -> Self {
        PadWidth { before, after }
    }

    /// Zero padding (no-op).
    pub fn zero() -> Self {
        PadWidth { before: 0, after: 0 }
    }
}

/// Pads the tensor along each axis with the specified widths and mode.
///
/// Returns a new owned tensor. The original data is copied into the center
/// of the padded output.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `widths` — Per-axis padding widths. Length must equal `ndim`.
///   Each entry specifies `(before, after)` padding for that axis.
/// * `mode` — Padding mode (constant, edge, reflect).
///
/// # Errors
///
/// Returns `TensorError::InvalidShape` if `widths.len() != ndim`.
/// Returns `TensorError::InvalidShape` if reflect padding width exceeds
/// the axis length minus 1 (would need to reflect past available data).
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, pad, PadMode, shape::PadWidth};
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
/// let widths = vec![PadWidth::symmetric(1), PadWidth::symmetric(0)];
/// let b = pad(&a, &widths, PadMode::Constant(0.0))?;
/// assert_eq!(b.shape(), &[4, 2]);
/// ```
pub fn pad<S, A, D>(
    tensor: &TensorBase<S, D>,
    widths: &[PadWidth],
    mode: PadMode<A>,
) -> Result<Tensor<A, D>>
where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if widths.len() != ndim {
        return Err(TensorError::InvalidShape {
            expected: ndim,
            actual: widths.len(),
        });
    }

    let old_shape = tensor.shape();

    // Compute output shape
    let mut new_shape = D::zeros(ndim);
    for ax in 0..ndim {
        new_shape.slice_mut()[ax] = old_shape[ax] + widths[ax].before + widths[ax].after;
    }

    // Validate reflect mode
    if matches!(mode, PadMode::Reflect) {
        for ax in 0..ndim {
            let max_pad = old_shape[ax].saturating_sub(1);
            if widths[ax].before > max_pad || widths[ax].after > max_pad {
                return Err(TensorError::InvalidShape {
                    expected: max_pad,
                    actual: widths[ax].before.max(widths[ax].after),
                });
            }
        }
    }

    // Allocate output and initialize padding regions
    let total = new_shape.size();
    let mut out_storage = Owned::<A>::uninitialized(total);
    let out_strides = new_shape.default_strides();

    // Fill with constant if that mode
    if let PadMode::Constant(ref val) = mode {
        for i in 0..total {
            unsafe {
                out_storage.as_mut_ptr().add(i).write(val.clone());
            }
        }
    } else {
        // For Edge/Reflect: fill everything first, then overwrite center
        // (Edge: fill with edge values; Reflect: computed per-element)
        // Implementation fills center last, so initialize to a default
        for i in 0..total {
            unsafe {
                out_storage.as_mut_ptr().add(i).write(A::zero());
            }
        }
    }

    // Copy original data into the center of the output
    for (src_flat, elem) in tensor.iter().enumerate() {
        // Convert flat index to multi-dim
        let mut src_idx = vec![0usize; ndim];
        let mut rem = src_flat;
        for ax in (0..ndim).rev() {
            let stride = tensor.strides()[ax] as usize;
            if stride > 0 {
                src_idx[ax] = rem / stride;
                rem %= stride;
            }
        }

        // Compute output index (shift by before padding)
        let mut dst_offset = 0;
        for ax in 0..ndim {
            let out_coord = src_idx[ax] + widths[ax].before;
            dst_offset += out_coord * out_strides.slice()[ax];
        }
        unsafe {
            out_storage.as_mut_ptr().add(dst_offset).write(elem.clone());
        }
    }

    // Fill padding regions for Edge and Reflect modes
    match &mode {
        PadMode::Constant(_) => { /* already filled */ }
        PadMode::Edge => {
            fill_edge_padding(&mut out_storage, &new_shape, &out_strides, widths, tensor);
        }
        PadMode::Reflect => {
            fill_reflect_padding(&mut out_storage, &new_shape, &out_strides, widths, tensor);
        }
    }

    let flags = LayoutFlags::compute(&new_shape, &out_strides, out_storage.as_ptr(), 0);
    Ok(unsafe { TensorBase::from_storage_unchecked(out_storage, new_shape, out_strides, 0) })
}

/// Fills edge padding regions by repeating boundary elements.
fn fill_edge_padding<S, A, D>(
    _out: &mut Owned<A>,
    _shape: &D,
    _strides: &D,
    _widths: &[PadWidth],
    _source: &TensorBase<S, D>,
) where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    // Iterate over all output elements that are in padding regions.
    // For each, find the nearest edge element in the source tensor and copy it.
    // This is a straightforward but potentially slow implementation;
    // can be optimized later with per-axis iteration.
    todo!("edge padding fill")
}

/// Fills reflect padding regions by mirroring source elements.
fn fill_reflect_padding<S, A, D>(
    _out: &mut Owned<A>,
    _shape: &D,
    _strides: &D,
    _widths: &[PadWidth],
    _source: &TensorBase<S, D>,
) where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    // For each padding element, compute the reflected source index:
    //   reflect(i, len) = if i < 0 { -i - 1 } else if i >= len { 2*len - 1 - i } else { i }
    // This mirrors without including the edge element itself.
    todo!("reflect padding fill")
}
```

---

### 4.9 repeat / tile — 重复/铺排

> 需求：§11.6

```rust
// ── repeat.rs ──────────────────────────────────────────────

/// Repeats elements along each axis.
///
/// Returns a new owned tensor. Each element is repeated `reps[axis]`
/// times along the corresponding axis.
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `reps` — Repetition counts, one per axis. Length must equal `ndim`.
///   A count of 0 produces an empty result along that axis.
///
/// # Errors
///
/// Returns `TensorError::InvalidShape` if `reps.len() != ndim`.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, repeat};
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
/// let b = repeat(&a, &[2, 3])?;
/// // shape: [4, 6] — each element repeated 2x along axis 0, 3x along axis 1
/// ```
pub fn repeat<S, A, D>(
    tensor: &TensorBase<S, D>,
    reps: &[usize],
) -> Result<Tensor<A, D>>
where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    let ndim = tensor.ndim();
    if reps.len() != ndim {
        return Err(TensorError::InvalidShape {
            expected: ndim,
            actual: reps.len(),
        });
    }

    let old_shape = tensor.shape();

    // Compute output shape
    let mut new_shape = D::zeros(ndim);
    for ax in 0..ndim {
        new_shape.slice_mut()[ax] = old_shape[ax] * reps[ax];
    }

    let total = new_shape.size();
    if total == 0 {
        // reps contains a 0 — return empty tensor
        let strides = new_shape.default_strides();
        let storage = Owned::<A>::uninitialized(0);
        let flags = LayoutFlags::compute(&new_shape, &strides, storage.as_ptr(), 0);
        return Ok(unsafe {
            TensorBase::from_storage_unchecked(storage, new_shape, strides, 0)
        });
    }

    let mut out_storage = Owned::<A>::uninitialized(total);
    let out_strides = new_shape.default_strides();

    // Copy each source element to all its repetitions in the output
    for (src_flat, elem) in tensor.iter().enumerate() {
        // Decompose flat index into multi-dim
        let mut src_idx = vec![0usize; ndim];
        let mut rem = src_flat;
        for ax in (0..ndim).rev() {
            let s = old_shape[ax];
            if s > 0 {
                src_idx[ax] = rem % s;
                rem /= s;
            }
        }

        // Write to all repeated positions
        write_repeated(
            &mut out_storage,
            &out_strides,
            &src_idx,
            reps,
            &new_shape,
            elem.clone(),
        );
    }

    let flags = LayoutFlags::compute(&new_shape, &out_strides, out_storage.as_ptr(), 0);
    Ok(unsafe { TensorBase::from_storage_unchecked(out_storage, new_shape, out_strides, 0) })
}

/// Helper: write a single element to all its repeated positions.
fn write_repeated<A>(
    out: &mut Owned<A>,
    out_strides: &impl Dimension,
    src_idx: &[usize],
    reps: &[usize],
    out_shape: &impl Dimension,
    elem: A,
) {
    let ndim = src_idx.len();
    if ndim == 0 {
        unsafe { out.as_mut_ptr().write(elem); }
        return;
    }

    // Generate all repetition indices via nested loops (recursive)
    fn recurse<A>(
        out: &mut Owned<A>,
        out_strides: &impl Dimension,
        src_idx: &[usize],
        reps: &[usize],
        axis: usize,
        ndim: usize,
        base_offset: usize,
        elem: &A,
    ) where
        A: Clone,
    {
        if axis == ndim {
            unsafe { out.as_mut_ptr().add(base_offset).write(elem.clone()); }
            return;
        }
        let stride = out_strides.slice()[axis];
        let start = src_idx[axis] * reps[axis] * stride;
        for r in 0..reps[axis] {
            let offset = base_offset + start + r * stride;
            recurse(out, out_strides, src_idx, reps, axis + 1, ndim, offset, elem);
        }
    }

    recurse(out, out_strides, src_idx, reps, 0, ndim, 0, &elem);
}

/// Tiles the tensor by repeating it as a whole.
///
/// Unlike `repeat` (which repeats individual elements), `tile` repeats
/// the entire array. The `reps` length may be longer than `ndim` — if so,
/// the tensor is first promoted to higher dimensionality (leading 1s).
///
/// # Arguments
///
/// * `tensor` — The source tensor.
/// * `reps` — Number of times to tile along each axis. If shorter than
///   `ndim`, left-padded with 1s. If longer, the tensor is promoted.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, tile};
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
/// let b = tile(&a, &[2, 3])?;
/// // shape: [4, 6] — the full 2x2 array is tiled 2x3 times
/// ```
pub fn tile<S, A, D>(
    tensor: &TensorBase<S, D>,
    reps: &[usize],
) -> Result<Tensor<A, IxDyn>>
where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    let ndim = tensor.ndim();
    let out_ndim = ndim.max(reps.len());

    // Left-pad both shape and reps to out_ndim
    let pad = out_ndim.saturating_sub(ndim);
    let mut full_shape: Vec<usize> = vec![1; pad];
    full_shape.extend_from_slice(tensor.shape());

    let reps_pad = out_ndim.saturating_sub(reps.len());
    let mut full_reps: Vec<usize> = vec![1; reps_pad];
    full_reps.extend_from_slice(reps);

    // Promote source to out_ndim
    let promoted = if pad > 0 {
        let mut t = tensor.view();
        for _ in 0..pad {
            t = expand_dims(&t, 0)?;
        }
        t
    } else {
        // Safe: we only reshape if already the right ndim
        reshape_into(tensor, IxDyn::new(&full_shape))?
    };

    // Output shape = full_shape[i] * full_reps[i]
    let out_shape: Vec<usize> = full_shape.iter().zip(full_reps.iter())
        .map(|(&s, &r)| s * r)
        .collect();

    let out_dim = IxDyn::new(&out_shape);
    let total = out_dim.size();
    let mut out_storage = Owned::<A>::uninitialized(total);
    let out_strides = out_dim.default_strides();

    // For each output element, compute which source element maps to it
    for out_flat in 0..total {
        // Decompose to multi-dim index
        let mut idx = vec![0usize; out_ndim];
        let mut rem = out_flat;
        for ax in (0..out_ndim).rev() {
            idx[ax] = rem % out_shape[ax];
            rem /= out_shape[ax];
        }

        // Map to source index via modulo
        let mut src_offset = 0;
        for ax in 0..out_ndim {
            let src_ax = idx[ax] % full_shape[ax];
            src_offset += src_ax * promoted.strides()[ax] as usize;
        }
        src_offset += promoted.offset();

        unsafe {
            let src_elem = &*promoted.as_ptr().add(src_offset - promoted.offset());
            out_storage.as_mut_ptr().add(out_flat).write((*src_elem).clone());
        }
    }

    let flags = LayoutFlags::compute(&out_dim, &out_strides, out_storage.as_ptr(), 0);
    Ok(unsafe { TensorBase::from_storage_unchecked(out_storage, out_dim, out_strides, 0) })
}
```

---

## 5. 内部实现设计

### 5.1 View vs Owned 输出决策

| 操作 | 输出类型 | 理由 |
|------|----------|------|
| `reshape` | `TensorView` | 步长重计算，共享存储，零拷贝 |
| `transpose` / `swap_axes` / `move_axis` | `TensorView` | 步长排列，共享存储，零拷贝 |
| `slice` / `index_axis` | `TensorView` | 偏移+形状裁剪，共享存储，零拷贝 |
| `squeeze` / `expand_dims` | `TensorView`（IxDyn） | 维度数变化，但共享存储 |
| `flip` / `flipud` / `fliplr` | `TensorView` | 负步长实现，零拷贝 |
| `split` / `chunk` / `unstack` | `Vec<TensorView>` | 多个零拷贝子视图 |
| `concatenate` / `stack` | `Tensor`（Owned） | 需要新分配，合并多个数据源 |
| `pad` | `Tensor`（Owned） | 需要新分配，填充区域无源数据 |
| `repeat` / `tile` | `Tensor`（Owned） | 需要新分配，元素被多次复制 |
| `flatten` | `TensorView` | 本质是 reshape，共享存储 |

**设计原则**：

- 零拷贝操作返回 `TensorView<'a, A, D>`（或 `IxDyn`），生命周期绑定源张量
- 拷贝操作返回 `Tensor<A, D>`（Owned），独立于源张量
- 维度数变化的操作使用 `IxDyn` 返回类型（因为 `D` 在编译时固定，无法表达 `D - 1` 维度的类型）

### 5.2 步长操纵实现策略

#### Reshape（连续数据）

```
old: shape=[2,3,4], strides=[1,2,6]  (F-order)
new: shape=[6,4],    strides=[1,6]    (F-order)

仅重计算步长，共享同一 data pointer 和 offset。
```

#### Transpose（轴排列）

```
old: shape=[2,3], strides=[1,2]  (F-order)
new: shape=[3,2], strides=[2,1]  (swapped strides)

shape 和 strides 按照排列交换，data pointer 和 offset 不变。
LayoutFlags 重计算：F_CONTIGUOUS ↔ C_CONTIGUOUS。
```

#### Flip（负步长）

```
old: shape=[3], strides=[1], offset=0
     data: [a, b, c]
new: shape=[3], strides=[-1], offset=2
     指向 c, 通过 stride=-1 反向遍历 → [c, b, a]

关键: offset 调整到该轴末尾元素位置。
```

#### Squeeze（移除长度1轴）

```
old: shape=[2,1,4], strides=[4,4,1]
new: shape=[2,4],    strides=[4,1]

移除 axis=1 (length=1)，shape/strides 各去掉一个元素。
data pointer 和 offset 不变。
```

#### ExpandDims（插入长度1轴）

```
old: shape=[3,4], strides=[1,3]  (F-order)
new: shape=[3,1,4], strides=[1,3,3]

插入 axis=1 (length=1)，stride 设为合理的占位值
（因为 length=1，stride 值不影响寻址结果）。
```

### 5.3 维度类型保持规则

| 场景 | 输入 `D` | 输出 `D` | 规则 |
|------|----------|----------|------|
| reshape（同维度数） | `Ix3` | `Ix3` | 保持静态类型 |
| reshape_into（跨类型） | `Ix3` | `IxDyn` | 显式目标类型 |
| transpose | `Ix3` | `Ix3` | 维度数不变，保持类型 |
| slice | `Ix3` | `Ix3` | 维度数不变，保持类型 |
| index_axis | `Ix3` | `IxDyn` | 维度数 -1，必须用 IxDyn |
| squeeze | `Ix3` | `IxDyn` | 维度数减少，必须用 IxDyn |
| expand_dims | `Ix3` | `IxDyn` | 维度数增加，必须用 IxDyn |
| split/chunk | `Ix3` | `Vec<TensorView<IxDyn>>` | 子视图维度数不变但用 IxDyn |
| stack | `Ix2` | `IxDyn` | 维度数 +1 |
| concatenate | `Ix2` | `Ix2` | 维度数不变 |
| pad | `Ix2` | `Ix2` | 维度数不变 |
| repeat | `Ix2` | `Ix2` | 维度数不变 |
| tile | `Ix2` | `IxDyn` | 可能增加维度数 |

**核心限制**：Rust 类型系统无法表达 `D::N - 1` 维度的类型，因此维度数变化的操作统一使用 `IxDyn`。用户可通过 `.into_dimension::<IxN>()` 显式转回静态类型。

### 5.4 LayoutFlags 更新策略

| 操作 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | HAS_ZERO_STRIDE | HAS_NEG_STRIDE |
|------|-------------|-------------|---------|----------------|----------------|
| reshape | 重新计算 | 重新计算 | 可能降级 | 清除 | 清除 |
| transpose | 交换 | 交换 | 继承 | 清除 | 可能设置 |
| slice | 可能降级 | 可能降级 | 可能降级 | 清除 | 可能设置 |
| flip | 清除 | 清除 | 可能降级 | 清除 | 设置 |
| squeeze | 继承 | 继承 | 继承 | 清除 | 继承 |
| expand_dims | 继承 | 继承 | 继承 | 清除 | 继承 |

所有操作均调用 `LayoutFlags::compute()` 进行完整重计算，确保标志一致性。

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 模块骨架

- [ ] **T1: shape/mod.rs 模块骨架**
  - 文件: `src/shape/mod.rs`
  - 内容: 模块声明（`pub mod reshape;` 等）+ 所有公共 API 的 re-export
  - 测试: `use Renon::shape::reshape;` 编译通过
  - 前置: tensor, dimension, layout, error 模块完成
  - 预计: 5 min

### 6.2 Reshape

- [ ] **T2: reshape (同维度类型)**
  - 文件: `src/shape/reshape.rs`
  - 内容: `reshape()` 函数 — 校验连续性、元素数、重算步长、构造视图
  - 测试: 有效 reshape、非连续返回错误、元素数不匹配返回错误
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: reshape_into (跨维度类型)**
  - 文件: `src/shape/reshape.rs`
  - 内容: `reshape_into()` — 同 T2 但接受 `D2` 目标维度
  - 测试: `Ix2 → Ix1` 成功、`Ix2 → Ix3` 失败
  - 前置: T2
  - 预计: 10 min

- [ ] **T4: reshape_inferred (-1 推断) + infer_dimension**
  - 文件: `src/shape/reshape.rs`
  - 内容: `reshape_inferred()` 和 `infer_dimension()` 辅助函数
  - 测试: 单 -1 推断、多 -1 panic、无 -1 直传
  - 前置: T2
  - 预计: 10 min

- [ ] **T5: flatten + flatten_range**
  - 文件: `src/shape/reshape.rs`
  - 内容: `flatten()` (→1D) 和 `flatten_range()` (折叠指定维度范围)
  - 测试: 3D→1D、flatten axis 1..3、非法范围返回错误
  - 前置: T3
  - 预计: 10 min

### 6.3 Transpose

- [ ] **T6: transpose (通用轴排列)**
  - 文件: `src/shape/transpose.rs`
  - 内容: `transpose()` — 校验排列合法性、重排 shape/strides、构造视图
  - 测试: 3D 排列正确、非法排列 panic、重复轴 panic
  - 前置: T1
  - 预计: 10 min

- [ ] **T7: t() (2D 快捷转置)**
  - 文件: `src/shape/transpose.rs`
  - 内容: `t()` — delegate to `transpose(tensor, [1, 0])`
  - 测试: 2D 转置正确、非 2D panic
  - 前置: T6
  - 预计: 5 min

- [ ] **T8: swap_axes**
  - 文件: `src/shape/transpose.rs`
  - 内容: `swap_axes()` — 构建双元素交换排列，delegate to `transpose()`
  - 测试: 交换后 shape 正确、相同轴 panic
  - 前置: T6
  - 预计: 5 min

- [ ] **T9: move_axis**
  - 文件: `src/shape/transpose.rs`
  - 内容: `move_axis()` — 构建移位排列，delegate to `transpose()`
  - 测试: move_axis(0, 2) 正确、no-op case
  - 前置: T6
  - 预计: 10 min

### 6.4 Slice

- [ ] **T10: SliceRange + SliceInfo 类型**
  - 文件: `src/shape/slice.rs`
  - 内容: `SliceRange` struct、`SliceInfo` struct、`From<Range>` 实现
  - 测试: 构造、边界值
  - 前置: T1
  - 预计: 10 min

- [ ] **T11: slice (零拷贝视图切片)**
  - 文件: `src/shape/slice.rs`
  - 内容: `slice()` — 按 SliceInfo 计算 offset/shape/strides，构造视图
  - 测试: 基本切片、带步长切片、越界返回错误
  - 前置: T10
  - 预计: 10 min

- [ ] **T12: index_axis (沿轴索引)**
  - 文件: `src/shape/slice.rs`
  - 内容: `index_axis()` — 移除一个轴，返回降维视图
  - 测试: 3D→2D 正确、F-contiguous 保持、越界返回错误
  - 前置: T11
  - 预计: 10 min

### 6.5 Squeeze / Expand Dims

- [ ] **T13: squeeze**
  - 文件: `src/shape/squeeze.rs`
  - 内容: `squeeze()` — 移除指定长度1轴（或全部长度1轴）
  - 测试: 移除单轴、自动移除所有、非1轴 panic
  - 前置: T1
  - 预计: 10 min

- [ ] **T14: expand_dims**
  - 文件: `src/shape/squeeze.rs`
  - 内容: `expand_dims()` — 在指定位置插入长度1轴
  - 测试: 插入轴0/中间/末尾、越界返回错误
  - 前置: T13
  - 预计: 10 min

### 6.6 Flip

- [ ] **T15: flip + flipud + fliplr**
  - 文件: `src/shape/transpose.rs`
  - 内容: `flip()` 通过负步长实现，`flipud`/`fliplr` 便捷封装
  - 测试: 单轴翻转、全轴翻转、flipud/fliplr 正确性
  - 前置: T6
  - 预计: 10 min

### 6.7 Split / Chunk

- [ ] **T16: split (按索引拆分)**
  - 文件: `src/shape/split.rs`
  - 内容: `split()` — 按边界索引拆分为多个视图
  - 测试: 基本拆分、非法索引返回错误
  - 前置: T11
  - 预计: 10 min

- [ ] **T17: chunk (均匀拆分)**
  - 文件: `src/shape/split.rs`
  - 内容: `chunk()` — 均匀拆分，处理余数
  - 测试: 均匀、不均匀余数、n_chunks=0 返回空 Vec
  - 前置: T16
  - 预计: 10 min

- [ ] **T18: unstack**
  - 文件: `src/shape/split.rs`
  - 内容: `unstack()` — 通过循环调用 `index_axis` 实现
  - 测试: 拆分数量正确、空轴返回空 Vec
  - 前置: T12
  - 预计: 5 min

### 6.8 Concatenate / Stack

- [ ] **T19: concatenate**
  - 文件: `src/shape/split.rs`
  - 内容: `concatenate()` — 校验形状、分配输出、逐张量拷贝
  - 测试: 沿轴0/轴1拼接、形状不匹配返回错误、空输入 panic
  - 前置: T1
  - 预计: 10 min

- [ ] **T20: stack**
  - 文件: `src/shape/split.rs`
  - 内容: `stack()` — expand_dims + concatenate 组合实现
  - 测试: 2D→3D 正确、形状不匹配返回错误
  - 前置: T14, T19
  - 预计: 10 min

### 6.9 Pad

- [ ] **T21: PadMode + PadWidth 类型**
  - 文件: `src/shape/pad.rs`
  - 内容: `PadMode<A>` 枚举、`PadWidth` struct
  - 测试: 构造、默认值
  - 前置: T1
  - 预计: 5 min

- [ ] **T22: pad (Constant 模式)**
  - 文件: `src/shape/pad.rs`
  - 内容: `pad()` 的 Constant 分支 — 分配 + 填充常量 + 拷贝中心数据
  - 测试: 常量填充形状正确、值正确、零宽度等价拷贝
  - 前置: T21
  - 预计: 10 min

- [ ] **T23: pad (Edge 模式)**
  - 文件: `src/shape/pad.rs`
  - 内容: `fill_edge_padding()` — 用边缘元素填充
  - 测试: 边缘值正确传播
  - 前置: T22
  - 预计: 10 min

- [ ] **T24: pad (Reflect 模式)**
  - 文件: `src/shape/pad.rs`
  - 内容: `fill_reflect_padding()` — 镜像反射填充
  - 测试: 反射值正确、超出范围返回错误
  - 前置: T22
  - 预计: 10 min

### 6.10 Repeat / Tile

- [ ] **T25: repeat**
  - 文件: `src/shape/repeat.rs`
  - 内容: `repeat()` — 逐元素重复拷贝到输出
  - 测试: 各轴重复正确、reps 含0返回空、reps 全1等价拷贝
  - 前置: T1
  - 预计: 10 min

- [ ] **T26: tile**
  - 文件: `src/shape/repeat.rs`
  - 内容: `tile()` — 整体铺排，支持 reps 长于 ndim
  - 测试: 基本 tile、reps 短于 ndim 自动补1、reps 长于 ndim 提升维度
  - 前置: T14, T25
  - 预计: 10 min

### 6.11 集成

- [ ] **T27: lib.rs re-export 集成**
  - 文件: `src/lib.rs`
  - 内容: `pub mod shape;` 和 `pub use shape::{...};`
  - 测试: 外部 `use Renon::reshape;` 编译通过
  - 前置: T1–T26
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于各子模块的 `#[cfg(test)] mod tests` 中：

#### reshape.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_reshape_valid_2d` | `[2,3] → [3,2]` | shape 正确，元素总数不变 |
| `test_reshape_non_contiguous_error` | 非连续输入 | 返回 `LayoutMismatch` |
| `test_reshape_wrong_count_error` | 元素数不匹配 | 返回 `InvalidShape` |
| `test_reshape_into_ix2_to_ixdyn` | 静态→动态 | shape 切片值一致 |
| `test_reshape_inferred_single_minus1` | `[2,3,4]` with `-1, 4` | 推断为 `[6, 4]` |
| `test_reshape_inferred_multiple_minus1_panic` | 两个 `-1` | panic |
| `test_flatten_3d_to_1d` | `[2,3,4] → [24]` | shape 正确 |
| `test_flatten_range_valid` | flatten axes 1..3 | shape 正确 |

#### transpose.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_transpose_2d` | `[2,3] → [3,2]` | shape 和 strides 正确交换 |
| `test_transpose_3d_permutation` | `[2,3,4] → perm [2,0,1]` | shape 为 `[4,2,3]` |
| `test_transpose_duplicate_axis_panic` | 重复轴号 | panic |
| `test_t_shorthand` | 2D 转置 | 等价于 `transpose([1,0])` |
| `test_swap_axes_basic` | 交换 axis 0 和 2 | shape 正确 |
| `test_move_axis_forward` | axis 0 → position 2 | shape `[3,4,2]` |
| `test_move_axis_noop` | source == destination | shape 不变 |
| `test_flip_single_axis` | flip axis 0 | 元素顺序反转 |
| `test_flipud` | 等价于 `flip(&[0])` | 正确 |
| `test_fliplr` | 等价于 `flip(&[1])` | 正确 |
| `test_flip_preserves_data` | flip 后 to_owned 验证 | 数据正确反转 |

#### slice.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_slice_basic_range` | `[1..3, ..]` | shape `[2, N]` |
| `test_slice_with_step` | `[..;2]` | 取偶数索引 |
| `test_slice_out_of_bounds` | 超出轴长 | 返回 `IndexOutOfBounds` |
| `test_slice_wrong_ndim` | range 数量不匹配 | 返回 `InvalidShape` |
| `test_index_axis_basic` | 3D 取 axis=1, idx=0 | ndim -1, 数据正确 |
| `test_index_axis_f_contiguous` | F-contiguous 3D batch | 2D 视图保持 F-contiguous |

#### squeeze.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_squeeze_specified_axis` | `[2,1,4]` squeeze axis=1 | shape `[2,4]` |
| `test_squeeze_all_ones` | `[1,3,1]` squeeze all | shape `[3]` |
| `test_squeeze_non_one_panic` | squeeze 长度非1轴 | panic |
| `test_expand_dims_beginning` | `[3,4]` at 0 | shape `[1,3,4]` |
| `test_expand_dims_middle` | `[3,4]` at 1 | shape `[3,1,4]` |
| `test_expand_dims_end` | `[3,4]` at 2 | shape `[3,4,1]` |

#### split.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_split_two_parts` | split at [3] | 2 parts, shapes 正确 |
| `test_split_unsorted_error` | 未排序索引 | 返回 `InvalidShape` |
| `test_chunk_even` | axis_len=6, n=3 | 3 chunks of 2 |
| `test_chunk_uneven` | axis_len=7, n=3 | chunks: [3,2,2] |
| `test_chunk_zero` | n_chunks=0 | 空 Vec |
| `test_unstack_basic` | `[3,4,5]` axis=0 | 3 views of `[4,5]` |
| `test_unstack_empty_axis` | axis_len=0 | 空 Vec |
| `test_concatenate_axis0` | 2×`[2,3]` along 0 | shape `[4,3]` |
| `test_concatenate_shape_mismatch` | 不同宽度 | 返回 `ShapeMismatch` |
| `test_stack_axis0` | 2×`[3,4]` | shape `[2,3,4]` |

#### pad.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_pad_constant_zero` | 全零常量填充 | 填充区域为 0 |
| `test_pad_constant_value` | 自定义常量 | 填充区域为指定值 |
| `test_pad_edge` | Edge 模式 | 边缘值正确传播 |
| `test_pad_reflect` | Reflect 模式 | 镜像值正确 |
| `test_pad_zero_width` | 零宽度 | 等价于 to_owned |
| `test_pad_reflect_exceeds_error` | 反射宽度超过限制 | 返回 `InvalidShape` |

#### repeat.rs 测试

| 测试名 | 场景 | 关键断言 |
|--------|------|----------|
| `test_repeat_basic` | `[2,3]` reps=[2,1] | shape `[4,3]` |
| `test_repeat_with_zero` | reps 含 0 | 空 tensor |
| `test_repeat_all_ones` | reps 全 1 | 等价于 to_owned |
| `test_tile_basic` | `[2,2]` reps=[2,3] | shape `[4,6]` |
| `test_tile_shorter_reps` | reps 短于 ndim | 左侧补 1 |
| `test_tile_longer_reps` | reps 长于 ndim | 维度提升 |

### 7.2 集成测试

位于 `tests/shape_ops.rs`：

| 测试组 | 覆盖内容 |
|--------|----------|
| 链式操作 | `reshape → transpose → slice → squeeze` |
| 跨存储操作 | View → reshape → to_owned → pad |
| 大张量 | 1000×1000 reshape/transpose/slice 性能验证 |
| 非连续布局 | 转置后的切片、切片后的 reshape（应失败） |
| 边界情况 | 0D tensor 的 reshape/expand_dims、1D tensor 的操作 |

### 7.3 属性测试

```rust
// 属性: reshape 后再 reshape 回原形状，数据不变
fn reshape_roundtrip_preserves_data(tensor: &Tensor<f64, IxDyn>) {
    let original_shape = tensor.shape().to_vec();
    let flat = reshape(tensor, Ix1(tensor.len())).unwrap();
    let back = reshape_into(&flat, IxDyn::new(&original_shape)).unwrap();
    assert_eq!(tensor, &back);
}

// 属性: transpose 两次恢复原状
fn double_transpose_identity(tensor: &Tensor<f64, IxDyn>) {
    let ndim = tensor.ndim();
    let perm: Vec<usize> = (0..ndim).collect();
    let t1 = transpose(tensor, &perm);
    let t2 = transpose(&t1, &perm);  // 需要逆排列
    assert_eq!(tensor, &t2.to_owned());
}

// 属性: split 再 concatenate 恢复原张量
fn split_concat_roundtrip(tensor: &Tensor<f64, Ix2>) {
    let parts = split(tensor, 1, &[2]).unwrap();
    let refs: Vec<&Tensor<_, _>> = parts.iter().map(|v| &v.to_owned()).collect();
    let recovered = concatenate(&refs, 1).unwrap();
    assert_eq!(tensor, &recovered);
}
```

### 7.4 测试文件位置

```
src/shape/
├── reshape.rs          // #[cfg(test)] mod tests { ... }
├── transpose.rs        // #[cfg(test)] mod tests { ... }
├── slice.rs            // #[cfg(test)] mod tests { ... }
├── squeeze.rs          // #[cfg(test)] mod tests { ... }
├── split.rs            // #[cfg(test)] mod tests { ... }
├── pad.rs              // #[cfg(test)] mod tests { ... }
└── repeat.rs           // #[cfg(test)] mod tests { ... }

tests/
├── shape_ops.rs        // 集成测试
└── property/
    └── shape_roundtrip.rs  // 属性测试
```

### 7.5 边界覆盖清单

| 边界 | 涉及操作 |
|------|----------|
| 空张量 (len=0) | reshape (size=0)、split/chunk、pad |
| 单元素 (1D len=1) | squeeze (移除唯一轴)、expand_dims |
| 0D 标量 | reshape、expand_dims |
| 高维 (≥4D) | transpose (复杂排列)、flatten_range |
| 非连续布局 | reshape (应返回错误)、slice (应正常) |
| F-order vs C-order | reshape 步长计算、transpose 标志交换 |
| 大步长值 | flatten 大张量、tile 高重复数 |
