# 索引操作模块设计

> 文档编号: 12 | 模块: `src/indexing.rs` + `src/macros.rs` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `02-dimension.md`
> 需求参考: 需求说明书 §12（索引操作）

---

## 1. 模块定位

索引操作模块为 `TensorBase<S, D>` 提供多维数据访问能力，涵盖以下场景：

| 索引类别 | 说明 | 内存影响 |
|----------|------|----------|
| 单元素索引 | `tensor[[i, j, k]]` 访问单个元素 | 零拷贝，返回引用 |
| 范围切片 | `tensor.slice(s![0..3, .., 2])` 取子视图 | 零拷贝，返回视图 |
| 轴索引 | `tensor.index_axis(Axis(1), 2)` 沿轴取切片 | 零拷贝，降维视图 |
| 高级索引（整数数组） | `tensor.take(indices, Axis(1))` 按索引数组取值 | 拷贝，返回 Owned |
| 高级索引（布尔掩码） | `tensor.mask(mask)` 按掩码取值 | 拷贝，返回一维 Owned |
| 条件选择 | `where_(condition, x, y)` | 拷贝，返回 Owned |
| 写入操作 | `tensor.put(indices, values)` 按索引写入值 | 原地修改 |

**核心设计原则：**

- **零拷贝优先**：范围切片和轴索引始终返回视图（`TensorView` / `TensorViewMut`），不分配内存
- **高级索引返回 Owned**：整数数组和布尔掩码索引因结果元素在内存中不连续，必须拷贝到新分配
- **与 ndarray 生态兼容**：`s![]` 宏语法参考 ndarray 的同名宏，降低用户学习成本
- **完备的错误处理**：编程错误（索引越界）panic + 提供 unsafe `_unchecked` 变体；逻辑错误（形状不匹配）返回 `Result`

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| Rust `Index` trait 实现 | `TensorBase[i]` 单维、`TensorBase[[i,j]]` 多维 | — |
| `s![]` 切片宏 | 多轴切片描述生成 | 编译期宏展开逻辑（由 `macros.rs` 提供） |
| `slice()` / `slice_mut()` | 从切片描述创建视图 | — |
| `index_axis()` / `index_axis_mut()` | 沿轴取单切片，返回降维视图 | `select()`（沿轴取多个切片，由 shape 模块提供） |
| 高级索引 | `take`, `take_along_axis`, `mask`, `compress` | — |
| 写入操作 | `put`, `put_mask` | `scatter_to`（由 shape 模块提供） |
| 条件选择 | `where_()` 函数 | — |
| 辅助操作 | `argwhere`, `nonzero` | — |

---

## 2. 文件位置

```
src/
├── indexing.rs          # 索引操作主模块：slice, index_axis, take, mask, put, where_, argwhere, nonzero
├── macros.rs            # s![] 宏定义
├── lib.rs               # pub mod indexing; pub use indexing::*; pub use macros::s;
```

**单文件设计理由：** 索引操作方法虽多，但均围绕同一主题（数据访问），逻辑高度内聚。`s![]` 宏单独放在 `macros.rs` 是因为宏定义需要在 crate 根级别可见。

---

## 3. 依赖关系

```
indexing.rs
├── crate::tensor        # TensorBase, Tensor, TensorView, TensorViewMut, type aliases
├── crate::dimension     # Dimension, Ix0~Ix6, IxDyn, DimensionMismatch
├── crate::storage       # RawStorage, Storage, Owned, ViewRepr, ViewMutRepr
├── crate::layout        # LayoutFlags, compute_strides()
├── crate::element       # Element
├── crate::error         # TensorError, Result
└── crate::construction  # zeros() (for allocating output arrays in advanced indexing)

macros.rs
├── (无外部依赖，纯宏展开)
└── 产出 SliceInfo, SliceInfoItem 类型（在 indexing.rs 中定义）
```

**依赖方向：单向向下。** `indexing` 仅消费核心模块（tensor, dimension, storage, layout, element, error），不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 辅助类型

#### Axis — 轴描述符

```rust
/// A lightweight wrapper around a `usize` identifying a tensor axis.
///
/// Used as a parameter type for operations that target a specific axis,
/// making the API self-documenting (e.g., `index_axis(Axis(1), 0)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(pub usize);

impl Axis {
    /// Creates a new axis descriptor.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::indexing::Axis;
    /// let axis = Axis(1);
    /// assert_eq!(axis.0, 1);
    /// ```
    #[inline]
    pub const fn new(axis: usize) -> Self {
        Axis(axis)
    }

    /// Returns the underlying axis index.
    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }
}
```

#### SliceInfoItem — 单轴切片描述

```rust
/// A single element of a slice descriptor, describing how to slice one axis.
///
/// This is a regular (non-half-open) range with explicit start, end, and step.
/// Unlike `Range<usize>`, it supports explicit step values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceInfoItem {
    /// Start index (inclusive). `None` means 0.
    pub start: Option<usize>,
    /// End index (exclusive). `None` means axis length.
    pub end: Option<usize>,
    /// Step size. Must be >= 1. `None` means 1.
    pub step: Option<usize>,
}

impl SliceInfoItem {
    /// Creates a new slice info item.
    ///
    /// # Panics
    ///
    /// Panics if `step` is `Some(0)`.
    #[inline]
    pub fn new(start: Option<usize>, end: Option<usize>, step: Option<usize>) -> Self {
        if let Some(s) = step {
            assert!(s > 0, "slice step must be non-zero");
        }
        SliceInfoItem { start, end, step }
    }

    /// Slice the full axis (`..`).
    #[inline]
    pub const fn full() -> Self {
        SliceInfoItem { start: None, end: None, step: None }
    }

    /// Slice from `start` to end (`start..`).
    #[inline]
    pub const fn from(start: usize) -> Self {
        SliceInfoItem { start: Some(start), end: None, step: None }
    }

    /// Slice from 0 to `end` (`..end`).
    #[inline]
    pub const fn to(end: usize) -> Self {
        SliceInfoItem { start: None, end: Some(end), step: None }
    }

    /// Slice from `start` to `end` (`start..end`).
    #[inline]
    pub const fn bounded(start: usize, end: usize) -> Self {
        SliceInfoItem { start: Some(start), end: Some(end), step: None }
    }

    /// Slice with an explicit step (`start..end;step`).
    #[inline]
    pub const fn stepped(start: usize, end: usize, step: usize) -> Self {
        SliceInfoItem { start: Some(start), end: Some(end), step: Some(step) }
    }

    /// Resolves this slice against an axis of the given length.
    ///
    /// Returns `(resolved_start, resolved_end, resolved_step, output_length)`.
    ///
    /// # Panics
    ///
    /// Panics if `start` or `end` exceeds `axis_len`.
    #[inline]
    pub fn resolve(&self, axis_len: usize) -> (usize, usize, usize, usize) {
        let step = self.step.unwrap_or(1);
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(axis_len);
        assert!(start <= axis_len, "slice start {} exceeds axis length {}", start, axis_len);
        assert!(end <= axis_len, "slice end {} exceeds axis length {}", end, axis_len);
        let len = if start >= end { 0 } else { (end - start + step - 1) / step };
        (start, end, step, len)
    }
}

/// A newtype index that removes one dimension.
///
/// Used inside `s![]` to indicate a single integer index along an axis,
/// which removes that axis from the output (like NumPy's `a[i]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceIndex(pub usize);

impl From<usize> for SliceIndex {
    #[inline]
    fn from(i: usize) -> Self {
        SliceIndex(i)
    }
}
```

#### SliceInfo — 多轴切片描述集合

```rust
/// A collection of per-axis slice descriptors.
///
/// Produced by the `s![]` macro and consumed by `TensorBase::slice()`.
/// Each element describes how one axis is sliced: either a range (producing
/// a sub-axis) or an index (removing the axis).
///
/// # Type Parameters
///
/// * `D` — The output dimension type after slicing (axes with integer indices
///   are removed, so the output may have fewer dimensions than the input).
#[derive(Clone, Debug)]
pub struct SliceInfo<D: Dimension> {
    /// Per-axis descriptors. Length must match the *input* tensor's ndim.
    items: Vec<SliceItemKind>,

    /// Phantom data carrying the output dimension type.
    out_dim: core::marker::PhantomData<D>,
}

/// A single axis descriptor in a slice specification.
#[derive(Clone, Copy, Debug)]
pub enum SliceItemKind {
    /// Range slice: axis remains in output (possibly shorter).
    Range(SliceInfoItem),
    /// Integer index: axis is removed from output.
    Index(SliceIndex),
    /// New axis marker (inserts a size-1 axis).
    NewAxis,
}

impl<D: Dimension> SliceInfo<D> {
    /// Creates a new `SliceInfo` from a vector of per-axis descriptors.
    ///
    /// # Panics
    ///
    /// Panics if any `SliceInfoItem` has step 0.
    pub fn new(items: Vec<SliceItemKind>) -> Self {
        for item in &items {
            if let SliceItemKind::Range(si) = item {
                if let Some(step) = si.step {
                    assert!(step > 0, "slice step must be non-zero");
                }
            }
        }
        SliceInfo {
            items,
            out_dim: core::marker::PhantomData,
        }
    }

    /// Returns the number of axes in the input tensor this slice applies to.
    #[inline]
    pub fn input_ndim(&self) -> usize {
        self.items.iter().filter(|i| !matches!(i, SliceItemKind::NewAxis)).count()
    }

    /// Returns the number of axes in the output tensor.
    ///
    /// Range items preserve an axis, index items remove one, newaxis adds one.
    #[inline]
    pub fn output_ndim(&self) -> usize {
        self.items.iter().filter(|i| !matches!(i, SliceItemKind::Index(_))).count()
    }

    /// Returns the per-axis descriptors.
    #[inline]
    pub fn items(&self) -> &[SliceItemKind] {
        &self.items
    }
}
```

### 4.2 `s![]` 宏

```rust
/// Multi-axis slice macro. Generates a `SliceInfo` value describing how to
/// slice each axis of a tensor.
///
/// # Syntax
///
/// Each comma-separated element describes one axis:
///
/// | Syntax | Meaning | Axis effect |
/// |--------|---------|-------------|
/// | `..` | Full range | Keep axis |
/// | `a..b` | Range from `a` to `b` (exclusive) | Keep axis |
/// | `a..` | Range from `a` to end | Keep axis |
/// | `..b` | Range from start to `b` | Keep axis |
/// | `a..b;c` | Range with step `c` | Keep axis |
/// | `n` | Single integer index | **Remove** axis |
/// | `NewAxis` | Insert a size-1 axis | **Add** axis |
///
/// # Examples
///
/// ```
/// use xenon::{s, Tensor, Ix3};
///
/// let t: Tensor<f64, Ix3> = Tensor::zeros([4, 5, 6]);
///
/// // Keep first 2 rows, all columns, third column only
/// let view = t.slice(s![0..2, .., 2]);
/// assert_eq!(view.shape(), &[2, 5]);
///
/// // Step by 2 along first axis
/// let stepped = t.slice(s![..;2, .., ..]);
/// assert_eq!(stepped.shape(), &[2, 5, 6]);
/// ```
#[macro_export]
macro_rules! s {
    // Empty: no slicing (0-dim tensor)
    () => { ... };

    // Single element patterns
    (..) => { ... };
    ($start:expr .. $end:expr) => { ... };
    ($start:expr ..) => { ... };
    (.. $end:expr) => { ... };
    ($start:expr .. $end:expr ; $step:expr) => { ... };
    ($index:expr) => { ... };

    // Multi-element: recursive matching
    ($($elem:tt),* $(,)?) => { ... };
}
```

**宏实现策略：**

`s![]` 宏通过逐 token 匹配将各种切片语法转换为 `SliceInfo<IxDyn>` 值。宏展开为 `SliceInfo::new(vec![...])`，每个元素为 `SliceItemKind::Range(...)` 或 `SliceItemKind::Index(...)`。

用户也可手动构造 `SliceInfo` 以获得编译时维度类型检查（指定 `SliceInfo<Ix2>` 等静态维度）。

### 4.3 `Index` trait 实现

#### 单维索引（1D tensor）

```rust
/// Single-dimension element access for 1D tensors.
///
/// # Panics
///
/// Panics if `index >= tensor.len()`.
impl<A, S> core::ops::Index<usize> for TensorBase<S, Ix1>
where
    S: RawStorage<Elem = A>,
{
    type Output = A;

    fn index(&self, index: usize) -> &A {
        assert!(index < self.len(), "index {} out of bounds for axis of length {}", index, self.len());
        // SAFETY: bounds check passed
        unsafe { self.get_unchecked(&[index]) }
    }
}

/// Mutable single-dimension element access for 1D tensors.
///
/// # Panics
///
/// Panics if `index >= tensor.len()`.
impl<A, S> core::ops::IndexMut<usize> for TensorBase<S, Ix1>
where
    S: Storage<Elem = A>,
{
    fn index_mut(&mut self, index: usize) -> &mut A {
        assert!(index < self.len(), "index {} out of bounds for axis of length {}", index, self.len());
        // SAFETY: bounds check passed
        unsafe { self.get_unchecked_mut(&[index]) }
    }
}
```

#### Range 索引（1D tensor）

```rust
/// Range slicing for 1D tensors. Returns a view (zero-copy).
///
/// Supports `Range<usize>`, `RangeFrom<usize>`, `RangeTo<usize>`, `RangeFull`.
impl<'a, A, S> core::ops::Index<core::ops::Range<usize>> for TensorBase<S, Ix1>
where
    S: RawStorage<Elem = A>,
{
    type Output = TensorBase<ViewRepr<&'a A>, Ix1>;

    fn index(&self, range: core::ops::Range<usize>) -> &Self::Output {
        // Transmute lifetime — actual lifetime bounded by &self
        // This follows the same pattern as ndarray
        ...
    }
}

/// `RangeFrom` slicing (`start..`).
impl<'a, A, S> core::ops::Index<core::ops::RangeFrom<usize>> for TensorBase<S, Ix1>
where
    S: RawStorage<Elem = A>,
{
    type Output = TensorBase<ViewRepr<&'a A>, Ix1>;
    ...
}

/// `RangeTo` slicing (`..end`).
impl<'a, A, S> core::ops::Index<core::ops::RangeTo<usize>> for TensorBase<S, Ix1>
where
    S: RawStorage<Elem = A>,
{
    type Output = TensorBase<ViewRepr<&'a A>, Ix1>;
    ...
}

/// `RangeFull` slicing (`..`).
impl<'a, A, S> core::ops::Index<core::ops::RangeFull> for TensorBase<S, Ix1>
where
    S: RawStorage<Elem = A>,
{
    type Output = TensorBase<ViewRepr<&'a A>, Ix1>;
    ...
}
```

> **设计决策：** `Index` trait 的返回类型为引用（`&Output`），因此范围切片返回 `&TensorView`。这要求对 `self` 的借用时长至少与返回视图的生命周期相同，由 Rust 借用检查器自动保证。

### 4.4 `slice()` / `slice_mut()` 方法

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Creates a view into a sub-tensor defined by the given slice descriptor.
    ///
    /// This is a zero-copy operation; the returned view shares the underlying
    /// storage with `self`.
    ///
    /// # Arguments
    ///
    /// * `info` — A `SliceInfo` describing how each axis is sliced.
    ///   The `info` must have exactly as many non-`NewAxis` entries as
    ///   `self.ndim()`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The number of non-`NewAxis` items does not match `self.ndim()`.
    /// - Any index is out of bounds for its axis.
    /// - Any step is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, s, Ix3};
    ///
    /// let t: Tensor<f64, Ix3> = Tensor::zeros([4, 5, 6]);
    /// let view = t.slice(s![0..2, .., 3]);
    /// assert_eq!(view.shape(), &[2, 5]);
    /// ```
    pub fn slice<D2>(&self, info: SliceInfo<D2>) -> TensorView<'_, S::Elem, D2>
    where
        D2: Dimension,
    {
        self.slice_impl(&info)
    }

    /// Creates a mutable view into a sub-tensor defined by the given slice descriptor.
    ///
    /// # Panics
    ///
    /// Same as `slice()`.
    pub fn slice_mut<D2>(&mut self, info: SliceInfo<D2>) -> TensorViewMut<'_, S::Elem, D2>
    where
        S: Storage,
        D2: Dimension,
    {
        self.slice_mut_impl(&info)
    }

    /// Creates a view into a sub-tensor without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - All indices in the slice descriptor are within bounds.
    /// - All steps are non-zero.
    /// - The number of non-`NewAxis` items matches `self.ndim()`.
    pub unsafe fn slice_unchecked<D2>(&self, info: &SliceInfo<D2>) -> TensorView<'_, S::Elem, D2>
    where
        D2: Dimension,
    {
        self.slice_impl_unchecked(info)
    }

    /// Creates a mutable view into a sub-tensor without bounds checking.
    ///
    /// # Safety
    ///
    /// Same as `slice_unchecked`.
    pub unsafe fn slice_unchecked_mut<D2>(
        &mut self,
        info: &SliceInfo<D2>,
    ) -> TensorViewMut<'_, S::Elem, D2>
    where
        S: Storage,
        D2: Dimension,
    {
        self.slice_mut_impl_unchecked(info)
    }
}
```

### 4.5 `index_axis()` / `index_axis_mut()`

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Returns a view of the tensor with the given axis removed, selecting
    /// the element at `index` along that axis.
    ///
    /// This is a zero-copy operation. The returned view has one fewer
    /// dimension than `self`.
    ///
    /// # Arguments
    ///
    /// * `axis` — The axis to index. Must be `< self.ndim()`.
    /// * `index` — The index along `axis`. Must be `< self.len_of(axis.0)`.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of bounds or `index` is out of bounds for
    /// the given axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Axis, Ix3};
    ///
    /// let t: Tensor<f64, Ix3> = Tensor::zeros([4, 5, 6]);
    /// let slice = t.index_axis(Axis(1), 2);
    /// assert_eq!(slice.shape(), &[4, 6]);
    /// ```
    pub fn index_axis(&self, axis: Axis, index: usize) -> TensorView<'_, S::Elem, D::Smaller>
    where
        D: RemoveAxis,
    {
        let axis_idx = axis.0;
        assert!(axis_idx < self.ndim(), "axis {} out of bounds for {}-dim tensor", axis_idx, self.ndim());
        assert!(index < self.len_of(axis_idx), "index {} out of bounds for axis {} of length {}", index, axis_idx, self.len_of(axis_idx));
        // SAFETY: bounds verified above
        unsafe { self.index_axis_unchecked(axis, index) }
    }

    /// Mutable version of `index_axis`.
    ///
    /// # Panics
    ///
    /// Same as `index_axis`.
    pub fn index_axis_mut(
        &mut self,
        axis: Axis,
        index: usize,
    ) -> TensorViewMut<'_, S::Elem, D::Smaller>
    where
        S: Storage,
        D: RemoveAxis,
    {
        let axis_idx = axis.0;
        assert!(axis_idx < self.ndim(), "axis {} out of bounds for {}-dim tensor", axis_idx, self.ndim());
        assert!(index < self.len_of(axis_idx), "index {} out of bounds for axis {} of length {}", index, axis_idx, self.len_of(axis_idx));
        // SAFETY: bounds verified above
        unsafe { self.index_axis_unchecked_mut(axis, index) }
    }

    /// Unchecked version of `index_axis`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `axis.0 < self.ndim()`
    /// - `index < self.len_of(axis.0)`
    pub unsafe fn index_axis_unchecked(
        &self,
        axis: Axis,
        index: usize,
    ) -> TensorView<'_, S::Elem, D::Smaller>
    where
        D: RemoveAxis,
    {
        self.index_axis_impl(axis, index)
    }

    /// Unchecked mutable version of `index_axis`.
    ///
    /// # Safety
    ///
    /// Same as `index_axis_unchecked`.
    pub unsafe fn index_axis_unchecked_mut(
        &mut self,
        axis: Axis,
        index: usize,
    ) -> TensorViewMut<'_, S::Elem, D::Smaller>
    where
        S: Storage,
        D: RemoveAxis,
    {
        self.index_axis_mut_impl(axis, index)
    }
}
```

**`RemoveAxis` trait 说明：** 需要在 `Dimension` trait 层面增加一个关联类型 `Smaller` 和一个约束 `RemoveAxis`，用于表达"减少一个维度"的类型关系：

```rust
/// Trait for dimension types that support removing one axis.
///
/// Static dimensions (Ix1..Ix6) implement this with `Smaller` being the
/// dimension type with one fewer axis. `Ix0` does NOT implement this
/// (cannot remove an axis from a 0-dim tensor).
/// `IxDyn` implements this with `Smaller = IxDyn`.
pub trait RemoveAxis: Dimension {
    /// The dimension type after removing one axis.
    type Smaller: Dimension;

    /// Removes the axis at the given position, returning the smaller dimension.
    fn remove_axis(&self, axis: usize) -> Self::Smaller;
}
```

### 4.6 高级索引 — take 操作

```rust
impl<A, S, D> TensorBase<S, D>
where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    /// Selects elements along an axis by integer indices.
    ///
    /// This is a **copy-based** operation. The returned tensor is newly allocated.
    ///
    /// # Arguments
    ///
    /// * `indices` — A 1D tensor of `usize` indices. Values must be `< self.len_of(axis)`.
    /// * `axis` — The axis along which to select.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if any index value exceeds the
    /// axis length.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Axis, Ix2};
    ///
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
    /// let indices = Tensor::from_vec(vec![0usize, 2], [2]);
    /// let result = t.take(&indices, Axis(1))?;
    /// // result shape: [2, 2], values: [[1, 3], [4, 6]]
    /// ```
    pub fn take<D2>(&self, indices: &TensorBase<impl RawStorage<Elem = usize>, D2>, axis: Axis) -> Result<Tensor<A, IxDyn>>
    where
        D2: Dimension,
    {
        let axis_idx = axis.0;
        if axis_idx >= self.ndim() {
            return Err(TensorError::InvalidAxis {
                axis: axis_idx,
                ndim: self.ndim(),
            });
        }

        let axis_len = self.len_of(axis_idx);
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= axis_len {
                return Err(TensorError::IndexOutOfBounds {
                    axis: axis_idx,
                    index: idx,
                    size: axis_len,
                });
            }
        }

        Ok(self.take_unchecked_impl(indices, axis_idx))
    }

    /// Selects elements along an axis by integer indices without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure all indices are `< self.len_of(axis)`.
    pub unsafe fn take_unchecked<D2>(
        &self,
        indices: &TensorBase<impl RawStorage<Elem = usize>, D2>,
        axis: Axis,
    ) -> Tensor<A, IxDyn>
    where
        D2: Dimension,
    {
        self.take_unchecked_impl(indices, axis.0)
    }
}

/// Selects elements along an axis using index arrays that broadcast together.
///
/// Similar to `np.take_along_axis` in NumPy.
///
/// # Arguments
///
/// * `self` — Source tensor.
/// * `indices` — Index tensor. Must have the same ndim as `self`.
///   All dimensions except `axis` must match or be broadcastable with `self`.
/// * `axis` — The axis along which to index.
///
/// # Errors
///
/// Returns `TensorError::ShapeMismatch` if shapes are incompatible.
/// Returns `TensorError::IndexOutOfBounds` if any index exceeds the axis length.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, Axis, Ix2};
///
/// let t = Tensor::from_vec(vec![10, 20, 30, 40, 50, 60], [2, 3]);
/// let indices = Tensor::from_vec(vec![2usize, 0, 1, 1], [2, 2]);
/// let result = t.take_along_axis(&indices, Axis(1))?;
/// // result: [[30, 10], [50, 50]]
/// ```
pub fn take_along_axis<A, S, D>(
    tensor: &TensorBase<S, D>,
    indices: &TensorBase<impl RawStorage<Elem = usize>, D>,
    axis: Axis,
) -> Result<Tensor<A, D>>
where
    S: RawStorage<Elem = A>,
    A: Element + Clone,
    D: Dimension,
{
    ...
}
```

### 4.7 高级索引 — 布尔掩码操作

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Selects elements where the boolean mask is `true`.
    ///
    /// Returns a 1D tensor containing all elements for which `mask` is `true`,
    /// in row-major (C) order. This is a **copy-based** operation.
    ///
    /// # Arguments
    ///
    /// * `mask` — A boolean tensor with the same shape as `self`.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeMismatch` if `mask.shape() != self.shape()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Ix2};
    ///
    /// let t = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]);
    /// let mask = Tensor::from_vec(vec![true, false, true, false, true, false], [2, 3]);
    /// let result = t.mask(&mask)?;
    /// // result: [1, 3, 5] (1D tensor)
    /// ```
    pub fn mask(&self, mask: &TensorBase<impl RawStorage<Elem = bool>, D>) -> Result<Tensor1<S::Elem>>
    where
        S::Elem: Element + Clone,
    {
        if self.shape() != mask.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: mask.shape().to_vec(),
            });
        }
        Ok(self.mask_impl(mask))
    }

    /// Selects elements along an axis where the boolean mask is `true`.
    ///
    /// Unlike `mask()` which flattens, this preserves all dimensions except
    /// the masked axis.
    ///
    /// # Arguments
    ///
    /// * `mask` — A 1D boolean tensor with length equal to `self.len_of(axis)`.
    /// * `axis` — The axis along which to apply the mask.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeMismatch` if mask length != axis length.
    /// Returns `TensorError::InvalidAxis` if axis is out of bounds.
    pub fn compress(
        &self,
        mask: &TensorBase<impl RawStorage<Elem = bool>, Ix1>,
        axis: Axis,
    ) -> Result<Tensor<S::Elem, IxDyn>>
    where
        S::Elem: Element + Clone,
    {
        ...
    }
}
```

### 4.8 写入操作 — put

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Writes values at the specified indices along an axis.
    ///
    /// This is the inverse operation of `take()`.
    ///
    /// # Arguments
    ///
    /// * `indices` — A 1D tensor of `usize` indices.
    /// * `values` — Values to write. Must have the same shape as what
    ///   `self.take(indices, axis)` would return.
    /// * `axis` — The axis along which to write.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if any index exceeds the axis length.
    /// Returns `TensorError::ShapeMismatch` if `values` shape is incompatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::{Tensor, Axis, Ix2};
    ///
    /// let mut t = Tensor::zeros::<f64, Ix2>([3, 4]);
    /// let indices = Tensor::from_vec(vec![0usize, 2], [2]);
    /// let values = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
    /// t.put(&indices, &values, Axis(1))?;
    /// // t[:, 0] = [1, 3, 5], t[:, 2] = [2, 4, 6]
    /// ```
    pub fn put<D2, D3>(
        &mut self,
        indices: &TensorBase<impl RawStorage<Elem = usize>, D2>,
        values: &TensorBase<impl RawStorage<Elem = S::Elem>, D3>,
        axis: Axis,
    ) -> Result<()>
    where
        S::Elem: Element + Clone,
        D2: Dimension,
        D3: Dimension,
    {
        ...
    }

    /// Writes values at positions where the boolean mask is `true`.
    ///
    /// # Arguments
    ///
    /// * `mask` — Boolean tensor with the same shape as `self`.
    /// * `values` — 1D tensor of values to write. Length must equal the number
    ///   of `true` elements in `mask`.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeMismatch` if shapes are incompatible.
    pub fn put_mask(
        &mut self,
        mask: &TensorBase<impl RawStorage<Elem = bool>, D>,
        values: &TensorBase<impl RawStorage<Elem = S::Elem>, Ix1>,
    ) -> Result<()>
    where
        S::Elem: Element + Clone,
    {
        ...
    }
}
```

### 4.9 条件选择 — `where_()`

```rust
/// Selects elements from two tensors based on a boolean condition.
///
/// For each element, if `condition` is `true`, the result is taken from `x`;
/// otherwise from `y`. This is the equivalent of NumPy's `np.where()`.
///
/// # Arguments
///
/// * `condition` — Boolean tensor (or broadcastable to the output shape).
/// * `x` — Values selected where `condition` is `true`.
/// * `y` — Values selected where `condition` is `false`.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if the three shapes cannot be
/// broadcast together.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, where_, Ix2};
///
/// let a = Tensor::from_vec(vec![1, 2, 3, 4], [2, 2]);
/// let b = Tensor::from_vec(vec![10, 20, 30, 40], [2, 2]);
/// let cond = Tensor::from_vec(vec![true, false, true, false], [2, 2]);
/// let result = where_(&cond, &a, &b)?;
/// // result: [[1, 20], [3, 40]]
/// ```
pub fn where_<A, S1, S2, S3, D>(
    condition: &TensorBase<S1, D>,
    x: &TensorBase<S2, D>,
    y: &TensorBase<S3, D>,
) -> Result<Tensor<A, D>>
where
    A: Element + Clone,
    S1: RawStorage<Elem = bool>,
    S2: RawStorage<Elem = A>,
    S3: RawStorage<Elem = A>,
    D: Dimension,
{
    // 1. Verify shapes are compatible (or broadcast)
    // 2. Allocate output tensor
    // 3. Iterate: output[i] = if condition[i] { x[i] } else { y[i] }
    ...
}
```

### 4.10 辅助索引操作

```rust
/// Returns the indices of non-zero (true) elements in a boolean tensor.
///
/// Returns a `Vec<Tensor1<usize>>` where each tensor contains the indices
/// along one axis. The length of the Vec equals `tensor.ndim()`.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, nonzero, Ix2};
///
/// let t = Tensor::from_vec(vec![true, false, true, false, false, true], [2, 3]);
/// let indices = nonzero(&t);
/// // indices[0] = [0, 0, 1]  (row indices)
/// // indices[1] = [0, 2, 2]  (col indices)
/// ```
pub fn nonzero<S, D>(tensor: &TensorBase<S, D>) -> Vec<Tensor1<usize>>
where
    S: RawStorage<Elem = bool>,
    D: Dimension,
{
    ...
}

/// Returns the indices of true elements as a 2D array of shape `[n_true, ndim]`.
///
/// This is the equivalent of NumPy's `np.argwhere()`.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, argwhere, Ix2};
///
/// let t = Tensor::from_vec(vec![true, false, true, false, false, true], [2, 3]);
/// let indices = argwhere(&t);
/// // indices shape: [3, 2]
/// // indices: [[0, 0], [0, 2], [1, 2]]
/// ```
pub fn argwhere<S, D>(tensor: &TensorBase<S, D>) -> Tensor2<usize>
where
    S: RawStorage<Elem = bool>,
    D: Dimension,
{
    ...
}
```

---

## 5. 内部实现设计

### 5.1 切片描述符到视图的转换

`slice()` 方法的核心逻辑是将 `SliceInfo` 转换为新的 `shape`、`strides` 和 `offset`：

```
输入: TensorBase<S, D> + SliceInfo<D2>
输出: TensorView<'_, A, D2>

对每个轴 i:
  match info.items()[i]:
    Range(item):
      - resolve (start, end, step, output_len) = item.resolve(axis_len)
      - new_offset += start * strides[i]
      - new_shape[i_out] = output_len
      - new_strides[i_out] = strides[i] * step (as isize)
      - i_out += 1

    Index(idx):
      - new_offset += idx * strides[i]
      - 不产出输出轴（降维）
      - i_out 不变

    NewAxis:
      - new_shape[i_out] = 1
      - new_strides[i_out] = 0  (不消耗数据)
      - i_out += 1
```

**关键实现细节：**

| 方面 | 处理方式 |
|------|----------|
| 步长符号 | 原始 strides 为 `isize`，乘以 `step` 后仍为 `isize` |
| 负步长支持 | 继承源张量的负步长；步长乘法不改变符号 |
| 布局标志 | 切片后重新计算 `LayoutFlags::compute()`，可能从 F/C contiguous 降级 |
| 边界检查 | 在 `slice()` 中一次性检查所有轴，通过后调用 `_unchecked` 内部方法 |
| 输出维度 | 使用 `IxDyn` 构造输出形状，因为轴数量可能在运行时变化 |

### 5.2 `index_axis` 实现

`index_axis(axis, index)` 等价于对指定轴进行整数索引，其他轴保持不变：

```
输入: TensorBase<S, D> shape=[a, b, c, d], axis=1, index=2
输出: TensorView<S::Elem, D::Smaller> shape=[a, c, d]

算法:
  new_offset = old_offset + index * strides[axis]
  new_shape  = [shape[0], shape[2], shape[3]]  (skip axis 1)
  new_strides = [strides[0], strides[2], strides[3]]  (skip axis 1)
  new_layout_flags = LayoutFlags::compute(&new_shape, &new_strides, ...)
```

**零拷贝保证：** 仅修改 metadata（shape, strides, offset），不触碰数据。

### 5.3 高级索引（take）实现

`take()` 是拷贝操作，需要逐元素从源数组读取并写入新分配：

```
输入: Tensor shape=[a, b, c], indices=[i0, i1, ..., ik], axis=1
输出: Tensor shape=[a, k, c]

算法:
  output_shape = [a, k, c]  (axis 长度替换为 indices.len())
  对 output 中每个位置 (r, j, s):
    source_idx = indices[j]
    output[r, j, s] = self[r, source_idx, s]
```

**优化路径：**

| 条件 | 优化 |
|------|------|
| 源数组沿目标轴 F-contiguous | 可使用连续内存拷贝 + 偏移 |
| indices 为连续递增 (0,1,2,...) | 退化为普通切片（零拷贝） |
| 源数组整体 contiguous | 按连续块批量拷贝 |

### 5.4 布尔掩码索引实现

`mask()` 需要两次遍历：第一次计数 true 元素确定输出大小，第二次拷贝：

```
算法 (mask):
  1. count = mask 中 true 的数量
  2. 分配 output: Tensor1<A>::uninitialized(count)
  3. 遍历 (elem, &mask_val) in zip(self, mask):
       if mask_val { output.push(elem.clone()) }
  4. 返回 output
```

**复杂度：** O(n) 其中 n = tensor.len()，需要一次完整遍历。

### 5.5 `where_()` 实现

```
算法:
  1. 验证 condition, x, y 形状兼容（广播检查）
  2. 确定输出形状 = broadcast_shape(condition, x, y)
  3. 分配 output: Tensor<A, D>::uninitialized(output_shape)
  4. 遍历广播后的三数组:
       output[i] = if condition[i] { x[i] } else { y[i] }
  5. 返回 output
```

### 5.6 `RemoveAxis` trait 与维度缩减

为支持 `index_axis` 的编译时维度类型推导，需要在 `Dimension` 体系上增加：

```rust
// Ix1 → Ix0
impl RemoveAxis for Ix1 {
    type Smaller = Ix0;
    fn remove_axis(&self, _axis: usize) -> Ix0 { Ix0 }
}

// Ix2 → Ix1
impl RemoveAxis for Ix2 {
    type Smaller = Ix1;
    fn remove_axis(&self, axis: usize) -> Ix1 {
        let s = self.slice();
        match axis {
            0 => Ix1(s[1]),
            1 => Ix1(s[0]),
            _ => panic!("axis {} out of bounds for Ix2", axis),
        }
    }
}

// Ix3 → Ix2
impl RemoveAxis for Ix3 {
    type Smaller = Ix2;
    fn remove_axis(&self, axis: usize) -> Ix2 { ... }
}

// Ix4 → Ix3, Ix5 → Ix4, Ix6 → Ix5 (同模式)

// Ix0: 不实现 RemoveAxis（无法从 0 维再降维）
// IxDyn → IxDyn
impl RemoveAxis for IxDyn {
    type Smaller = IxDyn;
    fn remove_axis(&self, axis: usize) -> IxDyn {
        let mut dims = self.dims.clone();
        dims.remove(axis);
        IxDyn { dims }
    }
}
```

### 5.7 布局标志计算

切片操作后需要重新计算布局标志：

```rust
fn compute_slice_layout_flags<A>(
    new_shape: &[usize],
    new_strides: &[isize],
    base_ptr: *const A,
    new_offset: usize,
) -> LayoutFlags {
    // 复用 LayoutFlags::compute，传入新的 shape 和 strides
    LayoutFlags::compute(
        &IxDyn::new(new_shape),
        &IxDyn::from(new_strides.iter().map(|&s| s as usize).collect::<Vec<_>>()),
        base_ptr as *const u8,
        new_offset,
    )
}
```

**切片后标志变化规则：**

| 操作 | F_CONTIGUOUS | C_CONTIGUOUS | ALIGNED | 说明 |
|------|-------------|-------------|---------|------|
| `..`（全选） | 保持 | 保持 | 保持 | 无变化 |
| `0..n`（前缀） | 保持 | 保持 | 保持 | offset 不变 |
| `k..n`（非零起始） | 保持 | 保持 | 可能丢失 | offset 破坏对齐 |
| `..;s`（步长 > 1） | 丢失 | 丢失 | 保持 | 非连续 |
| 单索引（降维） | 保持 | 保持 | 可能丢失 | 去除一轴，可能偏移 |

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### Phase 3A: 基础类型定义

- [ ] **T1: SliceInfoItem + SliceIndex + SliceItemKind 类型**
  - 文件: `src/indexing.rs:1-120`
  - 内容: `SliceInfoItem` struct 及其 `new/full/from/to/bounded/stepped/resolve` 方法, `SliceIndex` newtype, `SliceItemKind` enum
  - 测试: `test_slice_info_item_resolve`, `test_slice_info_item_full`, `test_slice_info_item_bounds_panic`
  - 前置: 无
  - 预计: 10 min

- [ ] **T2: SliceInfo<D> 泛型切片描述集合**
  - 文件: `src/indexing.rs`
  - 内容: `SliceInfo<D>` struct, `new()`, `input_ndim()`, `output_ndim()`, `items()`
  - 测试: `test_slice_info_basic`, `test_slice_info_mixed_items`
  - 前置: T1, dimension 模块
  - 预计: 10 min

- [ ] **T3: Axis newtype**
  - 文件: `src/indexing.rs`
  - 内容: `Axis(pub usize)` 定义, `new()`, `index()` 方法
  - 测试: `test_axis_basic`
  - 前置: 无
  - 预计: 5 min

- [ ] **T4: RemoveAxis trait + 静态维度实现**
  - 文件: `src/dimension.rs`（扩展）+ `src/indexing.rs`
  - 内容: `RemoveAxis` trait 定义, `Ix1..Ix6, IxDyn` 的实现
  - 测试: `test_remove_axis_ix2`, `test_remove_axis_ixdyn`
  - 前置: dimension 模块完成
  - 预计: 10 min

### Phase 3B: s![] 宏

- [ ] **T5: s![] 宏 — 基础模式**
  - 文件: `src/macros.rs`
  - 内容: `s![]` 宏展开规则，支持 `..`, `a..b`, `a..`, `..b`, `n` 五种模式
  - 测试: `test_s_macro_full`, `test_s_macro_range`, `test_s_macro_from`, `test_s_macro_to`, `test_s_macro_index`
  - 前置: T1, T2
  - 预计: 10 min

- [ ] **T6: s![] 宏 — 步长和多轴**
  - 文件: `src/macros.rs`
  - 内容: `a..b;c` 步长语法, 多轴逗号分隔, `NewAxis` 标记
  - 测试: `test_s_macro_step`, `test_s_macro_multi_axis`, `test_s_macro_newaxis`
  - 前置: T5
  - 预计: 10 min

### Phase 3C: 切片与轴索引

- [ ] **T7: `slice()` 核心实现 — 只读版本**
  - 文件: `src/indexing.rs`
  - 内容: `slice_impl()` 内部方法，`SliceInfo` → 新 shape/strides/offset 转换逻辑
  - 测试: `test_slice_basic_2d`, `test_slice_with_step`, `test_slice_remove_axis`, `test_slice_preserves_data`
  - 前置: T1-T4
  - 预计: 10 min

- [ ] **T8: `slice_mut()` — 可变版本**
  - 文件: `src/indexing.rs`
  - 内容: `slice_mut()` 和 `slice_mut_impl()`，与 `slice()` 共享维度计算逻辑
  - 测试: `test_slice_mut_modify`, `test_slice_mut_preserves_others`
  - 前置: T7
  - 预计: 10 min

- [ ] **T9: `index_axis()` / `index_axis_mut()`**
  - 文件: `src/indexing.rs`
  - 内容: 沿轴单索引降维，checked + unchecked 版本
  - 测试: `test_index_axis_2d`, `test_index_axis_3d`, `test_index_axis_mut`, `test_index_axis_oob_panic`
  - 前置: T4, T7
  - 预计: 10 min

### Phase 3D: Index trait 实现

- [ ] **T10: `Index<usize>` for 1D tensor**
  - 文件: `src/indexing.rs`
  - 内容: `Index<usize>` 和 `IndexMut<usize>` 实现，返回元素引用
  - 测试: `test_index_usize_1d`, `test_index_mut_usize_1d`, `test_index_oob_panic`
  - 前置: tensor 模块
  - 预计: 10 min

- [ ] **T11: `Index<Range>` for 1D tensor**
  - 文件: `src/indexing.rs`
  - 内容: `Index<Range>`, `Index<RangeFrom>`, `Index<RangeTo>`, `Index<RangeFull>` 实现，返回视图
  - 测试: `test_index_range_1d`, `test_index_range_from`, `test_index_range_to`, `test_index_range_full`
  - 前置: T7, T10
  - 预计: 10 min

### Phase 3E: 高级索引

- [ ] **T12: `take()` — 整数数组索引**
  - 文件: `src/indexing.rs`
  - 内容: `take()` 方法实现，含边界检查 + unchecked 版本
  - 测试: `test_take_basic`, `test_take_oob_error`, `test_take_preserves_other_axes`
  - 前置: T7, construction 模块
  - 预计: 10 min

- [ ] **T13: `take_along_axis()` — 沿轴数组索引**
  - 文件: `src/indexing.rs`
  - 内容: `take_along_axis()` 自由函数实现
  - 测试: `test_take_along_axis_basic`, `test_take_along_axis_shape_mismatch`
  - 前置: T12
  - 预计: 10 min

- [ ] **T14: `mask()` — 布尔掩码索引**
  - 文件: `src/indexing.rs`
  - 内容: `mask()` 方法，两次遍历算法
  - 测试: `test_mask_basic`, `test_mask_shape_mismatch`, `test_mask_all_false_empty`, `test_mask_all_true`
  - 前置: T7
  - 预计: 10 min

- [ ] **T15: `compress()` — 沿轴布尔掩码**
  - 文件: `src/indexing.rs`
  - 内容: `compress()` 方法，保留非目标轴维度
  - 测试: `test_compress_basic`, `test_compress_preserves_dims`
  - 前置: T14
  - 预计: 10 min

### Phase 3F: 写入与条件操作

- [ ] **T16: `put()` — 按索引写入**
  - 文件: `src/indexing.rs`
  - 内容: `put()` 方法，`take()` 的逆操作
  - 测试: `test_put_basic`, `test_put_then_take_roundtrip`, `test_put_oob_error`
  - 前置: T12
  - 预计: 10 min

- [ ] **T17: `put_mask()` — 按掩码写入**
  - 文件: `src/indexing.rs`
  - 内容: `put_mask()` 方法
  - 测试: `test_put_mask_basic`, `test_put_mask_count_mismatch`
  - 前置: T14, T16
  - 预计: 10 min

- [ ] **T18: `where_()` — 条件选择**
  - 文件: `src/indexing.rs`
  - 内容: `where_()` 自由函数实现
  - 测试: `test_where_basic`, `test_where_broadcast`, `test_where_shape_mismatch`
  - 前置: T14, construction 模块
  - 预计: 10 min

### Phase 3G: 辅助操作

- [ ] **T19: `nonzero()` + `argwhere()`**
  - 文件: `src/indexing.rs`
  - 内容: 布尔数组非零元素索引查找
  - 测试: `test_nonzero_basic`, `test_nonzero_empty`, `test_argwhere_basic`, `test_argwhere_2d`
  - 前置: T14
  - 预计: 10 min

### Phase 3H: 集成

- [ ] **T20: `lib.rs` 注册与 re-export**
  - 文件: `src/lib.rs`
  - 内容: `pub mod indexing;`, re-export `s`, `Axis`, `SliceInfo`, `SliceInfoItem`, `where_`, `nonzero`, `argwhere`
  - 测试: `use xenon::{s, Axis, Tensor};` 编译通过
  - 前置: T1-T19
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/indexing.rs` 中的 `#[cfg(test)] mod tests`：

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **SliceInfoItem** | `test_slice_info_item_resolve` | 正确解析 start/end/step/length |
| | `test_slice_info_item_resolve_step2` | 步长为 2 时长度正确 |
| | `test_slice_info_item_resolve_empty` | start >= end 时长度为 0 |
| | `test_slice_info_item_oob_panic` | 越界 start/end panic |
| **SliceInfo** | `test_slice_info_basic` | input_ndim/output_ndim 正确 |
| | `test_slice_info_mixed` | 混合 Range/Index/NewAxis 正确计数 |
| **s![] 宏** | `test_s_macro_full` | `s![..]` 产出 Range(full) |
| | `test_s_macro_range` | `s![0..3]` 产出 Range(bounded) |
| | `test_s_macro_from` | `s![2..]` 产出 Range(from) |
| | `test_s_macro_to` | `s![..5]` 产出 Range(to) |
| | `test_s_macro_index` | `s![2]` 产出 Index |
| | `test_s_macro_step` | `s![0..6;2]` 产出 Range(stepped) |
| | `test_s_macro_multi` | `s![0..3, .., 2]` 产出 3 个 item |
| **slice()** | `test_slice_basic_2d` | shape=[2,5] from [4,5] sliced [0..2, ..] |
| | `test_slice_with_step` | shape=[2,5] from [4,5] sliced [..;2, ..] |
| | `test_slice_remove_axis` | shape=[5] from [4,5] sliced [2, ..] |
| | `test_slice_preserves_data` | 切片视图中的值与原始对应位置一致 |
| | `test_slice_offset_correct` | 多次连续切片 offset 累加正确 |
| | `test_slice_non_contiguous` | 步长切片后 is_contiguous() == false |
| | `test_slice_layout_flags` | 切片后布局标志正确更新 |
| **slice_mut()** | `test_slice_mut_modify` | 通过 mut slice 修改后原数组对应位置改变 |
| | `test_slice_mut_preserves_others` | 修改切片不影响切片外元素 |
| **index_axis()** | `test_index_axis_2d` | shape=[4,6] from [4,5,6] axis=1 index=2 |
| | `test_index_axis_3d` | shape=[5,6] from [4,5,6] axis=0 index=1 |
| | `test_index_axis_mut` | 通过 index_axis_mut 修改后原数组改变 |
| | `test_index_axis_oob_panic` | 轴或索引越界 panic |
| | `test_index_axis_f_contig_preserved` | F-contig 沿最外轴索引保持 F-contig |
| | `test_index_axis_inner_non_contig` | 沿内轴索引产生非连续视图 |
| **Index<usize>** | `test_index_usize_1d` | `tensor[2]` 返回正确元素 |
| | `test_index_mut_usize_1d` | `tensor[2] = val` 写入成功 |
| | `test_index_oob_panic` | 越界 panic |
| **Index<Range>** | `test_index_range_1d` | `tensor[1..4]` 返回 shape=[3] 的视图 |
| | `test_index_range_from` | `tensor[2..]` 正确 |
| | `test_index_range_to` | `tensor[..3]` 正确 |
| | `test_index_range_full` | `tensor[..]` 返回全视图 |

### 7.2 集成测试

位于 `tests/indexing.rs`：

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **take** | `test_take_basic` | 取指定索引元素，shape 正确 |
| | `test_take_oob_error` | 索引越界返回错误 |
| | `test_take_preserves_other_axes` | 非目标轴数据完整 |
| | `test_take_single_index` | 取单个索引等价于 index_axis |
| **take_along_axis** | `test_take_along_axis_basic` | 按索引数组沿轴取值 |
| | `test_take_along_axis_shape_mismatch` | 形状不兼容返回错误 |
| **mask** | `test_mask_basic` | 返回正确的一维数组 |
| | `test_mask_shape_mismatch` | 形状不匹配返回错误 |
| | `test_mask_all_false` | 返回空一维数组 |
| | `test_mask_all_true` | 返回展平的完整数组 |
| **compress** | `test_compress_basic` | 沿轴掩码保留其他维度 |
| **put** | `test_put_basic` | 写入后 take 回读一致 |
| | `test_put_oob_error` | 越界返回错误 |
| | `test_put_then_take_roundtrip` | put → take 往返一致 |
| **put_mask** | `test_put_mask_basic` | 掩码位置正确写入 |
| **where_** | `test_where_basic` | 条件选择正确 |
| | `test_where_with_broadcast` | 标量条件广播 |
| | `test_where_shape_mismatch` | 不兼容形状返回错误 |
| **nonzero** | `test_nonzero_basic` | 返回每轴索引数组 |
| | `test_nonzero_all_false` | 返回空 Vec |
| **argwhere** | `test_argwhere_basic` | 返回 [n_true, ndim] 形状 |
| | `test_argwhere_2d` | 2D 数组索引正确 |

### 7.3 边界测试

| 边界场景 | 测试项 |
|----------|--------|
| **空张量** | `test_slice_empty_tensor`, `test_take_empty_indices`, `test_mask_empty_tensor` |
| **0D 标量** | `test_slice_0d_noop`, `test_index_0d` |
| **1D 单元素** | `test_slice_single_element`, `test_mask_single_true` |
| **大步长** | `test_slice_large_step`, `test_take_large_indices` |
| **负步长** | `test_slice_neg_stride_preserved`, `test_index_axis_neg_stride` |
| **非连续** | `test_slice_transposed`, `test_take_non_contiguous`, `test_mask_broadcast_view` |
| **重复索引** | `test_take_duplicate_indices`, `test_put_duplicate_indices` |

### 7.4 测试覆盖率目标

| 模块 | 目标行覆盖率 |
|------|-------------|
| `indexing.rs` | ≥ 85% |
| 重点覆盖：切片计算、边界检查、布局标志更新 | 100% |
| `macros.rs`（s![] 宏） | ≥ 90%（通过集成测试间接覆盖） |

---

## 附录 A: 与 ndarray 对比

| 设计点 | ndarray | Xenon (本设计) |
|--------|---------|----------------|
| 切片宏 | `s![]` 使用 `Si` / `S` 类型 | `s![]` 使用 `SliceInfoItem` + `SliceItemKind` |
| Index trait | `Index<Ix>` for single elem | `Index<usize>` for 1D, `Index<[usize; N]>` for multi-dim |
| 切片方法 | `.slice(s![...])` | `.slice(s![...])`（一致） |
| 轴索引 | `.index_axis(Axis(n), i)` | `.index_axis(Axis(n), i)`（一致） |
| take | `.select(Axis(n), &indices)` | `.take(&indices, Axis(n))`（命名不同） |
| 布尔掩码 | 无内置 | `.mask(&mask)`（新增） |
| where | 无内置 | `where_(cond, x, y)`（新增） |
| put | 无内置 | `.put(indices, values, axis)`（新增） |
| 条件选择 | 无内置 | `where_()` 函数 |
| 布局标志 | 运行时计算 | 缓存 `LayoutFlags`，切片后重算 |

## 附录 B: 方法速查表

### 零拷贝操作（返回视图）

| 方法 | 签名 | 输入维度 | 输出维度 |
|------|------|----------|----------|
| `slice` | `(&self, SliceInfo<D2>) -> TensorView<A, D2>` | D | D2 |
| `slice_mut` | `(&mut self, SliceInfo<D2>) -> TensorViewMut<A, D2>` | D | D2 |
| `index_axis` | `(&self, Axis, usize) -> TensorView<A, D::Smaller>` | D | D::Smaller |
| `index_axis_mut` | `(&mut self, Axis, usize) -> TensorViewMut<A, D::Smaller>` | D | D::Smaller |

### 拷贝操作（返回 Owned）

| 方法 | 签名 | 输出维度 |
|------|------|----------|
| `take` | `(&self, &Tensor<usize, D2>, Axis) -> Result<Tensor<A, IxDyn>>` | IxDyn |
| `take_along_axis` | `(tensor, indices, Axis) -> Result<Tensor<A, D>>` | D |
| `mask` | `(&self, &Tensor<bool, D>) -> Result<Tensor1<A>>` | Ix1 |
| `compress` | `(&self, &Tensor<bool, Ix1>, Axis) -> Result<Tensor<A, IxDyn>>` | IxDyn |
| `where_` | `(cond, x, y) -> Result<Tensor<A, D>>` | D |
| `nonzero` | `(&Tensor<bool, D>) -> Vec<Tensor1<usize>>` | Vec of Ix1 |
| `argwhere` | `(&Tensor<bool, D>) -> Tensor2<usize>` | Ix2 |

### 原地修改操作

| 方法 | 签名 |
|------|------|
| `put` | `(&mut self, &Tensor<usize, D2>, &Tensor<A, D3>, Axis) -> Result<()>` |
| `put_mask` | `(&mut self, &Tensor<bool, D>, &Tensor<A, Ix1>) -> Result<()>` |

## 附录 C: 错误处理汇总

| 方法 | 错误类型 | 触发条件 |
|------|----------|----------|
| `slice` | panic | 轴数不匹配、索引越界、步长为 0 |
| `index_axis` | panic | 轴越界、索引越界 |
| `take` | `Result::Err(IndexOutOfBounds)` | 索引值超过轴长度 |
| `take` | `Result::Err(InvalidAxis)` | 轴索引超过 ndim |
| `mask` | `Result::Err(ShapeMismatch)` | 掩码与张量形状不同 |
| `compress` | `Result::Err(ShapeMismatch)` | 掩码长度不等于轴长度 |
| `compress` | `Result::Err(InvalidAxis)` | 轴越界 |
| `put` | `Result::Err(IndexOutOfBounds)` | 索引越界 |
| `put` | `Result::Err(ShapeMismatch)` | 值形状不匹配 |
| `put_mask` | `Result::Err(ShapeMismatch)` | 掩码或值形状不匹配 |
| `where_` | `Result::Err(BroadcastError)` | 三数组形状不可广播 |
