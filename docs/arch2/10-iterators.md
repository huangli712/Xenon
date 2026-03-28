# 迭代器模块设计

> 文档编号: 10 | 模块: `src/iter/` | 阶段: Phase 3（API 模块，可并行）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `05-storage.md`, `06-layout.md`, 需求说明书 §9

---

## 1. 模块定位

`iter` 模块为 Xenon 张量提供统一的遍历基础设施，是所有逐元素运算（加减乘除、归约、map）的核心驱动层。

**核心职责：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 元素遍历 | 按内存布局顺序或指定 Order 的逐元素迭代 | 具体运算逻辑（由 `ops/` 模块提供） |
| 轴遍历 | 沿指定轴产子视图（降维切片） | 子视图的具体操作 |
| 窗口遍历 | 滑动窗口产子视图 | 卷积运算 |
| 索引迭代 | 带多维索引的元素迭代 | 索引赋值（由 `indexing` 模块提供） |
| 多数组同步 | Zip 多个数组按相同逻辑顺序遍历 | 广播规则实现（由 `broadcast` 模块提供） |
| 并行迭代 | rayon-based 并行元素迭代 | 线程池管理 |

**设计原则：**

- **步长感知**：所有迭代器正确处理非连续内存（任意 strides），连续情况走快速路径
- **零拷贝**：轴迭代器和窗口迭代器产子视图，不复制数据
- **trait 一致性**：统一实现 `Iterator`、`DoubleEndedIterator`、`ExactSizeIterator`
- **并行就绪**：元素迭代器设计兼容 rayon `ParallelIterator`，通过 feature gate 启用

---

## 2. 文件位置

```
src/iter/
├── mod.rs          # 公共 trait 定义、re-export、iter()/iter_mut() 入口方法
├── elements.rs     # Iter, IterMut, IntoIter — 元素迭代器
├── axis.rs         # AxisIter, AxisIterMut — 轴迭代器（产子视图）
├── windows.rs      # Windows — 滑动窗口迭代器
├── indexed.rs      # IndexedIter — 带索引的元素迭代器
└── zip.rs          # Zip — 多数组同步迭代器
```

**文件划分理由：**

- 每种迭代器类别职责独立，单一文件降低认知负担
- `mod.rs` 仅承担入口分发和 re-export，不含实现逻辑
- 预计每个文件 200-400 行

---

## 3. 依赖关系

```
src/iter/
├── crate::tensor        # TensorBase<S, D>, 类型别名, view/view_mut
├── crate::dimension     # Dimension trait, Ix0~Ix6, IxDyn
├── crate::storage       # Storage, RawStorage, StorageMut trait
├── crate::element       # Element trait
├── crate::layout        # LayoutFlags, Order
├── crate::error         # TensorError, Result
└── crate::broadcast     # broadcast 形状推导 (仅 zip.rs 使用)
```

**依赖方向：单向向上。** `iter/` 仅消费 `tensor` 和 `dimension` 等核心模块，不被它们依赖。

### 依赖的具体类型

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `TensorView`, `TensorViewMut`, `.shape()`, `.strides()`, `.as_ptr()`, `.len()` |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` |
| `storage` | `RawStorage`, `Storage`, `Owned<A>`, `ViewRepr`, `ViewMutRepr`, `ArcRepr` |
| `layout` | `LayoutFlags`, `Order` |
| `element` | `Element` |
| `error` | `TensorError` |
| `broadcast` | `broadcast_shape()` (仅 zip) |

---

## 4. 公共 API 设计

### 4.1 元素迭代器

#### 4.1.1 `Iter` — 不可变元素迭代器

```rust
/// Immutable element iterator over a tensor.
///
/// Yields `&A` references in logical order (respecting strides).
/// For contiguous F-order arrays, the iteration order matches the
/// physical memory layout, enabling optimal cache performance.
///
/// # Order
///
/// By default, iterates in the tensor's natural layout order.
/// Use `.with_order(Order::F)` or `.with_order(Order::C)` to force
/// a specific logical traversal order.
pub struct Iter<'a, A, D>
where
    D: Dimension,
{
    /// Reference to the source tensor metadata.
    tensor: TensorView<'a, A, D>,

    /// Current multi-dimensional index in the iteration.
    index: D,

    /// Number of elements remaining.
    remaining: usize,

    /// Traversal order (F or C). Defaults to the tensor's natural order.
    order: Order,
}

// --- Constructor ---

impl<'a, A, D> Iter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new immutable element iterator.
    ///
    /// Uses the tensor's natural layout order for iteration.
    #[inline]
    fn new(tensor: TensorView<'a, A, D>) -> Self;
}

// --- Iterator trait ---

impl<'a, A, D> Iterator for Iter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    type Item = &'a A;

    #[inline]
    fn next(&mut self) -> Option<&'a A>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D> DoubleEndedIterator for Iter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a A>;
}

impl<'a, A, D> ExactSizeIterator for Iter<'a, A, D>
where
    A: Element,
    D: Dimension,
{}

// --- Additional methods ---

impl<'a, A, D> Iter<'a, A, D>
where
    D: Dimension,
{
    /// Returns the traversal order used by this iterator.
    #[inline]
    pub fn order(&self) -> Order;
}
```

#### 4.1.2 `IterMut` — 可变元素迭代器

```rust
/// Mutable element iterator over a tensor.
///
/// Yields `&mut A` references in logical order (respecting strides).
/// This iterator holds a mutable borrow of the tensor, ensuring
/// exclusive access for the duration of iteration.
///
/// # Safety
///
/// Each `&mut A` yielded is guaranteed to be unique — no two yielded
/// references alias the same element. This is ensured by the mutable
/// borrow of the source tensor.
pub struct IterMut<'a, A, D>
where
    D: Dimension,
{
    /// Mutable reference to the source tensor metadata.
    tensor: TensorViewMut<'a, A, D>,

    /// Current multi-dimensional index in the iteration.
    index: D,

    /// Number of elements remaining.
    remaining: usize,

    /// Traversal order.
    order: Order,
}

// --- Constructor ---

impl<'a, A, D> IterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new mutable element iterator.
    #[inline]
    fn new(tensor: TensorViewMut<'a, A, D>) -> Self;
}

// --- Iterator trait ---

impl<'a, A, D> Iterator for IterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    type Item = &'a mut A;

    #[inline]
    fn next(&mut self) -> Option<&'a mut A>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D> DoubleEndedIterator for IterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A>;
}

impl<'a, A, D> ExactSizeIterator for IterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{}
```

#### 4.1.3 `IntoIter` — 消费型元素迭代器

```rust
/// Consuming element iterator over an owned tensor.
///
/// Yields elements by value, consuming the tensor in the process.
/// This is more efficient than `.iter().cloned()` for owned tensors
/// because it avoids intermediate references.
pub struct IntoIter<A, D>
where
    D: Dimension,
{
    /// The owned tensor being consumed.
    tensor: Tensor<A, D>,

    /// Current multi-dimensional index.
    index: D,

    /// Number of elements remaining.
    remaining: usize,

    /// Traversal order.
    order: Order,
}

// --- Constructor ---

impl<A, D> IntoIter<A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new consuming iterator.
    #[inline]
    fn new(tensor: Tensor<A, D>) -> Self;
}

// --- Iterator trait ---

impl<A, D> Iterator for IntoIter<A, D>
where
    A: Element,
    D: Dimension,
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<A, D> DoubleEndedIterator for IntoIter<A, D>
where
    A: Element,
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<A>;
}

impl<A, D> ExactSizeIterator for IntoIter<A, D>
where
    A: Element,
    D: Dimension,
{}

// --- IntoIterator for Tensor (owned) ---

impl<A, D> IntoIterator for Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    type Item = A;
    type IntoIter = IntoIter<A, D>;

    #[inline]
    fn into_iter(self) -> IntoIter<A, D>;
}

// --- IntoIterator for TensorView (borrowed) ---

impl<'a, A, D> IntoIterator for TensorView<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, D>;

    #[inline]
    fn into_iter(self) -> Iter<'a, A, D>;
}

// --- IntoIterator for TensorViewMut (mutably borrowed) ---

impl<'a, A, D> IntoIterator for TensorViewMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    type Item = &'a mut A;
    type IntoIter = IterMut<'a, A, D>;

    #[inline]
    fn into_iter(self) -> IterMut<'a, A, D>;
}
```

#### 4.1.4 `TensorBase` 上的入口方法

```rust
// === 在 TensorBase<S, D> 上添加的迭代器入口方法 ===

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Returns an iterator over immutable element references.
    ///
    /// Iteration order follows the tensor's natural layout (F-order default).
    /// For contiguous data, this maps directly to sequential memory access.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t: Tensor2<f64> = zeros([3, 4]);
    /// for elem in t.iter() {
    ///     println!("{}", elem);
    /// }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, S::Elem, D>;

    /// Returns an iterator over immutable element references with a
    /// specified traversal order.
    ///
    /// # Arguments
    ///
    /// * `order` - The traversal order (F or C).
    #[inline]
    pub fn iter_with_order(&self, order: Order) -> Iter<'_, S::Elem, D>;
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns an iterator over mutable element references.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t: Tensor1<f64> = zeros([5]);
    /// for elem in t.iter_mut() {
    ///     *elem = 1.0;
    /// }
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, S::Elem, D>;

    /// Returns a mutable iterator with a specified traversal order.
    #[inline]
    pub fn iter_mut_with_order(&mut self, order: Order) -> IterMut<'_, S::Elem, D>;
}
```

---

### 4.2 轴迭代器

#### 4.2.1 `AxisIter` — 不可变轴迭代器

```rust
/// Iterator over sub-views along a specified axis.
///
/// For each step along `axis`, yields a `TensorView` of dimension `D::Smaller`
/// (one dimension fewer). The yielded views are zero-copy — they share
/// the underlying data with the source tensor.
///
/// # Examples
///
/// ```ignore
/// // Iterate over rows of a 2D tensor
/// let t: Tensor2<f64> = zeros([3, 4]);
/// for row in t.axis_iter(Axis(0)) {
///     // row: TensorView1<f64>, shape = [4]
///     assert_eq!(row.shape(), &[4]);
/// }
/// ```
pub struct AxisIter<'a, A, D>
where
    D: Dimension,
{
    /// Reference to the source tensor.
    base: TensorView<'a, A, D>,

    /// The axis being iterated over.
    axis: usize,

    /// Current position along the axis.
    current: usize,

    /// Length of the iterated axis.
    axis_len: usize,
}

// --- Constructor ---

impl<'a, A, D> AxisIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new axis iterator.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= base.ndim()`.
    #[inline]
    fn new(base: TensorView<'a, A, D>, axis: usize) -> Self;
}

// --- Iterator trait ---

impl<'a, A, D> Iterator for AxisIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Each yielded item is a sub-view with one fewer dimension.
    type Item = TensorView<'a, A, <D as Dimension>::Smaller>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D> DoubleEndedIterator for AxisIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>;
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{}
```

> **设计决策：** `AxisIter` 的 `Item` 类型为 `TensorView<'a, A, D::Smaller>`。这要求 `Dimension` trait 提供 `type Smaller: Dimension` 关联类型。对于 `IxDyn`（动态维度），`Smaller = IxDyn`（运行时减一轴）。对于 `Ix0`（标量），`Smaller` 不存在，因此 `AxisIter` 无法对 0 维数组构造（构造函数 panic）。

#### 4.2.2 `AxisIterMut` — 可变轴迭代器

```rust
/// Mutable iterator over sub-views along a specified axis.
///
/// Similar to `AxisIter`, but yields `TensorViewMut` instead of `TensorView`.
/// The mutable borrow ensures exclusive access for the duration of iteration.
///
/// # Safety
///
/// Each yielded `TensorViewMut` references a non-overlapping slice of the
/// underlying data. This is guaranteed by the axis-based partitioning —
/// no two yielded views share the same element along the iterated axis.
pub struct AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    /// Mutable reference to the source tensor.
    base: TensorViewMut<'a, A, D>,

    /// The axis being iterated over.
    axis: usize,

    /// Current position along the axis.
    current: usize,

    /// Length of the iterated axis.
    axis_len: usize,
}

// --- Constructor ---

impl<'a, A, D> AxisIterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new mutable axis iterator.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= base.ndim()`.
    #[inline]
    fn new(base: TensorViewMut<'a, A, D>, axis: usize) -> Self;
}

// --- Iterator trait ---

impl<'a, A, D> Iterator for AxisIterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Each yielded item is a mutable sub-view with one fewer dimension.
    type Item = TensorViewMut<'a, A, <D as Dimension>::Smaller>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D> DoubleEndedIterator for AxisIterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>;
}

impl<'a, A, D> ExactSizeIterator for AxisIterMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{}
```

#### 4.2.3 `Axis` 标记类型与入口方法

```rust
/// Axis marker type for axis-based operations.
///
/// Wraps a `usize` axis index to provide type safety and readability.
///
/// # Examples
///
/// ```ignore
/// use xenon::iter::Axis;
/// let t: Tensor2<f64> = zeros([3, 4]);
/// for row in t.axis_iter(Axis(0)) { /* rows */ }
/// for col in t.axis_iter(Axis(1)) { /* columns */ }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis(pub usize);

// === TensorBase 上的轴迭代器入口方法 ===

impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Returns an iterator over immutable sub-views along the given axis.
    ///
    /// Each yielded view has one fewer dimension than the source tensor.
    ///
    /// # Panics
    ///
    /// Panics if `axis.0 >= self.ndim()`.
    #[inline]
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, S::Elem, D>;
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns an iterator over mutable sub-views along the given axis.
    ///
    /// # Panics
    ///
    /// Panics if `axis.0 >= self.ndim()`.
    #[inline]
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, S::Elem, D>;
}
```

---

### 4.3 窗口迭代器

#### 4.3.1 `Windows` — 滑动窗口迭代器

```rust
/// Sliding window iterator over a tensor.
///
/// Produces sub-views of the given window size, stepping one element
/// at a time along each axis. The number of windows along axis `i`
/// is `shape[i] - window_size[i] + 1`.
///
/// Incomplete windows (where the window would exceed the tensor boundary)
/// are **not** produced.
///
/// # Examples
///
/// ```ignore
/// use xenon::iter::Windows;
/// let t: Tensor1<f64> = Tensor1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let mut wins = t.windows([2]);
/// assert_eq!(wins.next().unwrap().as_slice(), &[1.0, 2.0]);
/// assert_eq!(wins.next().unwrap().as_slice(), &[2.0, 3.0]);
/// assert_eq!(wins.next().unwrap().as_slice(), &[3.0, 4.0]);
/// assert!(wins.next().is_none());
/// ```
pub struct Windows<'a, A, D>
where
    D: Dimension,
{
    /// Reference to the source tensor.
    base: TensorView<'a, A, D>,

    /// Window size per axis.
    window: D,

    /// Current position of the window origin.
    origin: D,

    /// Number of remaining windows (total across all axes).
    remaining: usize,
}

// --- Constructor ---

impl<'a, A, D> Windows<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new sliding window iterator.
    ///
    /// # Panics
    ///
    /// Panics if `window.ndim() != self.ndim()`.
    /// Panics if any `window[i] == 0` or `window[i] > shape[i]`.
    #[inline]
    fn new(base: TensorView<'a, A, D>, window: D) -> Self;
}

// --- Iterator trait ---

impl<'a, A, D> Iterator for Windows<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Each yielded item is a read-only sub-view with the same dimensionality
    /// as the source tensor, but with shape equal to `window`.
    type Item = TensorView<'a, A, D>;

    #[inline]
    fn next(&mut self) -> Option<TensorView<'a, A, D>>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D> ExactSizeIterator for Windows<'a, A, D>
where
    A: Element,
    D: Dimension,
{}
```

> **设计决策：** `Windows` 仅支持不可变视图（产 `TensorView`）。可变窗口迭代会导致重叠区域出现别名问题（同一元素出现在多个窗口中），违反 Rust 的独占引用规则。如需可变窗口操作，应使用 `stride_chunks`（未来扩展）。

#### 4.3.2 入口方法

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Returns an iterator over all sliding windows of the given size.
    ///
    /// # Panics
    ///
    /// Panics if `window.ndim() != self.ndim()`.
    /// Panics if any `window[i] == 0` or `window[i] > self.shape()[i]`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t: Tensor2<f64> = zeros([4, 5]);
    /// for win in t.windows([2, 3]) {
    ///     // win: TensorView2<f64>, shape = [2, 3]
    /// }
    /// ```
    #[inline]
    pub fn windows(&self, window: D) -> Windows<'_, S::Elem, D>;
}
```

---

### 4.4 索引迭代器

#### 4.4.1 `IndexedIter` — 带索引的元素迭代器

```rust
/// Element iterator that yields `(index, &element)` pairs.
///
/// The `index` is a `D`-typed multi-dimensional index, and the element
/// is an immutable reference to the tensor element at that index.
///
/// This is useful for operations that need both the value and position
/// of each element, such as sparse construction or position-dependent
/// transformations.
///
/// # Examples
///
/// ```ignore
/// let t: Tensor2<f64> = zeros([2, 3]);
/// for (idx, &val) in t.indexed_iter() {
///     // idx: Ix2, val: f64
///     println!("t[{}, {}] = {}", idx[0], idx[1], val);
/// }
/// ```
pub struct IndexedIter<'a, A, D>
where
    D: Dimension,
{
    /// Inner element iterator (delegates element access).
    inner: Iter<'a, A, D>,

    /// Current multi-dimensional index.
    index: D,

    /// Traversal order (used to increment index correctly).
    order: Order,
}

// --- Constructor ---

impl<'a, A, D> IndexedIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new indexed iterator.
    #[inline]
    fn new(tensor: TensorView<'a, A, D>) -> Self;
}

// --- Iterator trait ---

impl<'a, A, D> Iterator for IndexedIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Each yielded item is a tuple of (multi-dimensional index, element reference).
    type Item = (D, &'a A);

    #[inline]
    fn next(&mut self) -> Option<(D, &'a A)>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<'a, A, D> ExactSizeIterator for IndexedIter<'a, A, D>
where
    A: Element,
    D: Dimension,
{}
```

> **设计决策：** `IndexedIter` 不实现 `DoubleEndedIterator`。双端迭代时从尾部推进索引需要维护两个独立的索引状态，复杂度高且收益有限。如果需要反向索引遍历，用户可以 `.collect()` 后 `.reverse()`。

#### 4.4.2 入口方法

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Returns an iterator yielding `(index, &element)` pairs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t: Tensor1<i32> = Tensor1::from_vec(vec![10, 20, 30]);
    /// for (idx, &val) in t.indexed_iter() {
    ///     println!("t[{}] = {}", idx, val);
    /// }
    /// ```
    #[inline]
    pub fn indexed_iter(&self) -> IndexedIter<'_, S::Elem, D>;
}
```

---

### 4.5 Zip — 多数组同步迭代

#### 4.5.1 `Zip` — 多数组同步迭代器

```rust
/// Synchronized iterator over multiple tensors.
///
/// Iterates over multiple tensors simultaneously, yielding tuples of
/// element references. All tensors must have compatible shapes
/// (identical or broadcastable).
///
/// # Broadcasting
///
/// If shapes are not identical but are broadcast-compatible, Zip will
/// automatically expand dimensions with stride=0 to match the broadcast
/// shape. This means the iteration count equals the product of the
/// broadcast shape.
///
/// # Type Arity
///
/// Zip supports 2 to 6 tensors via explicit tuple impls. This avoids
/// variadic generic complexity while covering the vast majority of
/// real-world use cases.
///
/// # Examples
///
/// ```ignore
/// use xenon::iter::zip;
///
/// let a: Tensor1<f64> = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b: Tensor1<f64> = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);
/// let mut c: Tensor1<f64> = zeros([3]);
///
/// Zip::from(&a).and(&b).and(&mut c).for_each(|(a, b, c)| {
///     *c = a + b;
/// });
/// ```
pub struct Zip<Parts, D>
where
    D: Dimension,
{
    /// The tensor parts being iterated (tuples of views/view_muts).
    parts: Parts,

    /// Broadcast shape for iteration.
    shape: D,

    /// Current index in the broadcast shape.
    index: D,

    /// Remaining element count.
    remaining: usize,

    /// Traversal order.
    order: Order,
}
```

**Zip 构建器模式：**

```rust
impl<'a, A, D> Zip<TensorView<'a, A, D>, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a new Zip starting from a single tensor view.
    #[inline]
    pub fn from(tensor: &TensorView<'a, A, D>) -> Zip<TensorView<'_, A, D>, D>;
}

// === Zip builder: adding more tensors ===

macro_rules! impl_zip_and {
    ($($Idx:tt: $T:ident),*) => {
        impl<'a, $($T,)* D> Zip<($($T,)*), D>
        where
            D: Dimension,
        {
            /// Adds another tensor to the Zip iteration.
            ///
            /// # Errors
            ///
            /// Returns `TensorError::ShapeMismatch` if the new tensor's shape
            /// is incompatible with the broadcast shape.
            #[inline]
            pub fn and<U>(self, tensor: &'a U) -> Result<Zip<($($T,)* U::Part,), D>>
            where
                U: ZipPart<D>,
            {
                // Validate broadcast compatibility
                // ...
            }
        }
    };
}

// Explicit impls for 2 to 6 tensors
impl_zip_and!(0: A);
impl_zip_and!(0: A, 1: B);
impl_zip_and!(0: A, 1: B, 2: C);
impl_zip_and!(0: A, 1: B, 2: C, 3: D);
impl_zip_and!(0: A, 1: B, 2: C, 3: D, 4: E);
```

**`ZipPart` trait — Zip 参与者抽象：**

```rust
/// Trait for types that can participate in Zip iteration.
///
/// Abstracts over immutable views, mutable views, and broadcast views,
/// allowing Zip to handle all combinations uniformly.
pub trait ZipPart<D: Dimension> {
    /// The element type yielded by this part.
    type Item;

    /// Returns the shape of this part.
    fn shape(&self) -> &D;

    /// Returns the strides of this part.
    fn strides(&self) -> &[isize];

    /// Returns a reference/pointer to the element at the given broadcast index.
    ///
    /// # Safety
    ///
    /// The caller must ensure the index is within the broadcast shape bounds
    /// and that the access does not violate aliasing rules.
    unsafe fn get_at(&self, broadcast_index: &[usize]) -> Self::Item;
}

// Implementations for TensorView, TensorViewMut, and broadcast views
impl<'a, A, D: Dimension> ZipPart<D> for TensorView<'a, A, D> {
    type Item = &'a A;
    // ...
}

impl<'a, A, D: Dimension> ZipPart<D> for TensorViewMut<'a, A, D> {
    type Item = &'a mut A;
    // ...
}
```

**Zip 的 `for_each` 和 apply 方法：**

```rust
// === for_each for binary Zip ===

impl<'a, A, B, D> Zip<(TensorView<'a, A, D>, TensorView<'a, B, D>), D>
where
    A: Element,
    B: Element,
    D: Dimension,
{
    /// Applies a function to each pair of elements.
    #[inline]
    pub fn for_each<F>(self, mut f: F)
    where
        F: FnMut(&'a A, &'a B),
    ;
}

// === for_each for ternary Zip (2 read + 1 write) ===

impl<'a, A, B, C, D> Zip<(
    TensorView<'a, A, D>,
    TensorView<'a, B, D>,
    TensorViewMut<'a, C, D>,
), D>
where
    A: Element,
    B: Element,
    C: Element,
    D: Dimension,
{
    /// Applies a function to each triple (two reads + one write).
    #[inline]
    pub fn for_each<F>(self, mut f: F)
    where
        F: FnMut(&'a A, &'a B, &'a mut C),
    ;
}
```

#### 4.5.2 便捷函数

```rust
/// Creates a Zip from a single tensor reference.
///
/// Convenience function equivalent to `Zip::from(tensor)`.
#[inline]
pub fn zip<'a, A, D>(tensor: &'a TensorView<'a, A, D>) -> Zip<TensorView<'a, A, D>, D>
where
    A: Element,
    D: Dimension,
;
```

---

### 4.6 并行迭代器（feature-gated）

> 以下类型仅在 `feature = "parallel"` 时可用。依赖 `rayon` crate。

#### 4.6.1 `ParIter` — 并行不可变迭代器

```rust
/// Parallel iterator over immutable element references.
///
/// Partitions the tensor into contiguous chunks and iterates over them
/// in parallel using rayon. Each chunk contains at least
/// `PARALLEL_MIN_CHUNK` elements (default 4K).
///
/// This is only available with the `parallel` feature enabled.
#[cfg(feature = "parallel")]
pub struct ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension,
{
    /// Reference to the source tensor.
    tensor: TensorView<'a, A, D>,

    /// Traversal order.
    order: Order,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParallelIterator for ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Item = &'a A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>;

    fn opt_len(&self) -> Option<usize>;
}

#[cfg(feature = "parallel")]
impl<'a, A, D> IndexedParallelIterator for ParIter<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    fn len(&self) -> usize;
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>;
    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>;
}
```

#### 4.6.2 `ParIterMut` — 并行可变迭代器

```rust
/// Parallel iterator over mutable element references.
///
/// Similar to `ParIter`, but yields `&mut A` references.
/// The partitioning guarantees non-overlapping access per thread.
#[cfg(feature = "parallel")]
pub struct ParIterMut<'a, A, D>
where
    A: Send + Element,
    D: Dimension,
{
    /// Mutable reference to the source tensor.
    tensor: TensorViewMut<'a, A, D>,

    /// Traversal order.
    order: Order,
}

#[cfg(feature = "parallel")]
impl<'a, A, D> ParallelIterator for ParIterMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    type Item = &'a mut A;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>;

    fn opt_len(&self) -> Option<usize>;
}

#[cfg(feature = "parallel")]
impl<'a, A, D> IndexedParallelIterator for ParIterMut<'a, A, D>
where
    A: Send + Sync + Element,
    D: Dimension + Send + Sync,
{
    fn len(&self) -> usize;
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>;
    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>;
}
```

#### 4.6.3 入口方法

```rust
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
    /// # Performance
    ///
    /// For small tensors (< `PARALLEL_THRESHOLD` elements), falls back
    /// to sequential iteration to avoid thread pool overhead.
    #[inline]
    pub fn par_iter(&self) -> ParIter<'_, S::Elem, D>;
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
    #[inline]
    pub fn par_iter_mut(&mut self) -> ParIterMut<'_, S::Elem, D>;
}
```

---

### 4.7 `mod.rs` 公共导出

```rust
// src/iter/mod.rs

mod elements;
mod axis;
mod windows;
mod indexed;
mod zip;

#[cfg(feature = "parallel")]
mod parallel;

// Public types
pub use self::elements::{Iter, IterMut, IntoIter};
pub use self::axis::{AxisIter, AxisIterMut, Axis};
pub use self::windows::Windows;
pub use self::indexed::IndexedIter;
pub use self::zip::{Zip, ZipPart, zip};

#[cfg(feature = "parallel")]
pub use self::parallel::{ParIter, ParIterMut};
```

---

## 5. 内部实现设计

### 5.1 步长感知元素访问

所有元素迭代器的核心是**多维索引递增**机制。每次 `next()` 调用：

1. 通过当前索引 `index` 和步长 `strides` 计算元素偏移：`offset = Σ(index[i] * strides[i])`
2. 通过 `base_ptr.add(offset)` 获取元素引用
3. 递增索引（按指定的 Order）

**索引递增算法（F-order）：**

```
increment_index_f(shape, index):
    for i in 0..ndim:
        index[i] += 1
        if index[i] < shape[i]:
            return  // no carry
        index[i] = 0  // carry to next dimension
    // if we exit the loop, all dimensions overflowed → iteration done
```

**索引递增算法（C-order）：**

```
increment_index_c(shape, index):
    for i in (0..ndim).rev():
        index[i] += 1
        if index[i] < shape[i]:
            return
        index[i] = 0
```

**快速路径优化：** 当 `is_f_contiguous()` 或 `is_c_contiguous()` 时，跳过多维索引计算，直接递增指针：

```rust
// Fast path for contiguous F-order data
if self.tensor.is_f_contiguous() && self.order == Order::F {
    // SAFETY: contiguous data, sequential access
    let ptr = self.tensor.as_ptr();
    let elem = unsafe { &*ptr.add(self.consumed) };
    self.consumed += 1;
    return Some(elem);
}
```

### 5.2 非连续迭代支持

对于非连续数组（步长 ≠ 1），每次访问需要跳过 `stride - 1` 个元素。关键保证：

| 保证 | 机制 |
|------|------|
| 不越界 | `remaining` 计数确保不超出 `len()` |
| 步长正确 | 使用有符号步长（`isize`），支持负值 |
| 对齐不假设 | 非连续情况下不使用 SIMD 加载指令 |

**非连续迭代的性能考量：**

- 每次 `next()` 需要 O(ndim) 的索引递增计算
- 实际数据访问是随机内存访问模式，cache 命中率低
- 对于 `ops/` 模块的大规模运算，应优先检查连续性并选择 SIMD 路径

### 5.3 轴迭代的子视图切片

`AxisIter` / `AxisIterMut` 的核心操作是将 N 维张量沿指定轴切片为 N-1 维子视图：

```
axis_iter(base, axis=0):
    base shape = [3, 4, 5]
    axis_len = 3

    yield i=0: view shape=[4,5], offset = base.offset + 0 * strides[0]
    yield i=1: view shape=[4,5], offset = base.offset + 1 * strides[0]
    yield i=2: view shape=[4,5], offset = base.offset + 2 * strides[0]
```

**实现要点：**

- 子视图的 `shape` = `base.shape` 去掉 `axis` 维度
- 子视图的 `strides` = `base.strides` 去掉 `axis` 维度
- 子视图的 `offset` = `base.offset + i * base.strides[axis]`
- 使用 `Dimension::remove_axis(axis)` 构造降维后的维度

### 5.4 Windows 的索引推进

窗口迭代器的索引推进类似多维计数器，但每个轴的上限为 `shape[i] - window[i] + 1`：

```
advance_origin(shape, window, origin, order):
    if order == F:
        for i in 0..ndim:
            origin[i] += 1
            if origin[i] < shape[i] - window[i] + 1:
                return
            origin[i] = 0
```

每次产出的子视图：
- `shape` = `window`（窗口大小）
- `strides` = 继承 `base.strides`（步长不变）
- `offset` = `base.offset + Σ(origin[i] * strides[i])`

### 5.5 Zip 的广播与分块

**广播机制：**

1. `Zip::from(&a)` 记录初始形状 `a.shape()`
2. `.and(&b)` 时检查 `b.shape()` 是否与当前广播形状兼容
3. 兼容规则：每轴 `a_dim == b_dim || a_dim == 1 || b_dim == 1`
4. 广播形状 = 每轴取 `max(a_dim, b_dim)`
5. 广播维度的步长设为 0（`ZipPart` 实现中处理）

**分块策略（并行模式）：**

```
par_iter partitioning:
    total = tensor.len()
    chunk_size = max(total / num_threads, PARALLEL_MIN_CHUNK)
    for chunk in tensor.chunks(chunk_size):
        spawn_thread(|| iterate(chunk))
```

> 仅对连续数据使用并行分块。非连续数据退化为标量并行（每个元素独立处理）。

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### Wave 1: 基础设施

- [ ] **T1: 创建 `src/iter/mod.rs` 骨架**
  - 文件: `src/iter/mod.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: tensor 模块完成
  - 预计: 5 min

- [ ] **T2: 实现 `Axis` 标记类型**
  - 文件: `src/iter/axis.rs`
  - 内容: `Axis(usize)` 结构体、`Debug/Clone/Copy/PartialEq/Eq/Hash` 派生
  - 测试: `test_axis_construction`, `test_axis_equality`
  - 前置: T1
  - 预计: 5 min

### Wave 2: 元素迭代器

- [ ] **T3: 实现 `Iter` 结构体 + `new` 构造函数**
  - 文件: `src/iter/elements.rs`
  - 内容: `Iter` 结构体定义、`new()` 方法、索引初始化
  - 测试: 编译通过
  - 前置: T1
  - 预计: 10 min

- [ ] **T4: 实现 `Iter` 的 `Iterator` trait**
  - 文件: `src/iter/elements.rs`
  - 内容: `next()` 使用步长感知索引递增、`size_hint()`、快速路径（连续数据直接指针递增）
  - 测试: `test_iter_f_contig`, `test_iter_c_contig`, `test_iter_non_contiguous`, `test_iter_empty`
  - 前置: T3
  - 预计: 10 min

- [ ] **T5: 实现 `Iter` 的 `DoubleEndedIterator` + `ExactSizeIterator`**
  - 文件: `src/iter/elements.rs`
  - 内容: `next_back()` 逆向迭代、`ExactSizeIterator` 空 impl
  - 测试: `test_iter_rev`, `test_iter_exact_len`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6: 实现 `IterMut` 结构体 + `Iterator` trait**
  - 文件: `src/iter/elements.rs`
  - 内容: `IterMut` 结构体、`new()`、`next()`、`next_back()`、`size_hint()`
  - 测试: `test_iter_mut_write`, `test_iter_mut_double_ended`, `test_iter_mut_non_contiguous`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: 实现 `IntoIter` + `IntoIterator` trait**
  - 文件: `src/iter/elements.rs`
  - 内容: `IntoIter` 结构体、`next()` 使用 `core::ptr::read` 消费元素、`IntoIterator` for `Tensor`/`TensorView`/`TensorViewMut`
  - 测试: `test_into_iter_owned`, `test_into_iter_view`, `test_into_iter_view_mut`
  - 前置: T5
  - 预计: 10 min

- [ ] **T8: 实现 `TensorBase` 上的 `iter()` / `iter_mut()` 入口方法**
  - 文件: `src/iter/elements.rs`（或 `src/tensor.rs` 通过 trait extension）
  - 内容: `iter()` → `Iter::new(self.view())`, `iter_mut()` → `IterMut::new(self.view_mut())`
  - 测试: `test_tensor_iter`, `test_tensor_iter_mut`
  - 前置: T4, T6
  - 预计: 10 min

### Wave 3: 轴迭代器

- [ ] **T9: 实现 `AxisIter` 结构体 + `Iterator` trait**
  - 文件: `src/iter/axis.rs`
  - 内容: `AxisIter` 结构体、`new()`、`next()` 产 `TensorView` 子视图、`size_hint()`
  - 测试: `test_axis_iter_2d_rows`, `test_axis_iter_2d_cols`, `test_axis_iter_3d`
  - 前置: T2, dimension 的 `remove_axis` 方法
  - 预计: 10 min

- [ ] **T10: 实现 `AxisIter` 的 `DoubleEndedIterator` + `ExactSizeIterator`**
  - 文件: `src/iter/axis.rs`
  - 内容: `next_back()` 从轴末端切片、`ExactSizeIterator`
  - 测试: `test_axis_iter_rev`, `test_axis_iter_exact_len`
  - 前置: T9
  - 预计: 10 min

- [ ] **T11: 实现 `AxisIterMut` 结构体 + 完整 trait**
  - 文件: `src/iter/axis.rs`
  - 内容: `AxisIterMut` 结构体、`new()`、`next()`、`next_back()`、`ExactSizeIterator`
  - 测试: `test_axis_iter_mut_write`, `test_axis_iter_mut_isolation`
  - 前置: T9
  - 预计: 10 min

- [ ] **T12: 实现 `TensorBase` 上的 `axis_iter()` / `axis_iter_mut()` 入口方法**
  - 文件: `src/iter/axis.rs`
  - 内容: 入口方法构造 `AxisIter`/`AxisIterMut`
  - 测试: `test_tensor_axis_iter`, `test_tensor_axis_iter_mut`
  - 前置: T9, T11
  - 预计: 5 min

### Wave 4: 窗口迭代器

- [ ] **T13: 实现 `Windows` 结构体 + `new` 构造函数**
  - 文件: `src/iter/windows.rs`
  - 内容: `Windows` 结构体、`new()` 含参数校验、窗口数计算
  - 测试: `test_windows_new_valid`, `test_windows_new_panic_zero`, `test_windows_new_panic_oversize`
  - 前置: T1
  - 预计: 10 min

- [ ] **T14: 实现 `Windows` 的 `Iterator` + `ExactSizeIterator`**
- [ ] **T14: 实现 `Windows` 的 `Iterator` + `ExactSizeIterator`**
  - 文件: `src/iter/windows.rs`
  - 测试: `test_windows_exact_len`
  - 等同数量验证：**
  - 测试: `test_windows_2d` | 2D [3,4] → 2×3 =6 个窗口 |
  - 测试: `test_windows_2d` | 2D [2,2] 窗口大小 = shape → 3 个窗口
  - 测试: `test_windows_no_incomplete``
  - 测试: `test_windows_no_incomplete`
` |

  - 前置: T14
  - 预计: 5 min

### Wave 5: 索引迭代器

- [ ] **T16: 实现 `IndexedIter` 结构体 + `Iterator` trait**
  - 文件: `src/iter/indexed.rs`
  - 内容: `IndexedIter` 结构体、`new()`、`next()` 产 `(D, &A)`、索引递增逻辑
  - 测试: `test_indexed_iter_1d`, `test_indexed_iter_2d`, `test_indexed_iter_empty`
  - 前置: T4
  - 预计: 10 min

- [ ] **T17: 实现 `TensorBase` 上的 `indexed_iter()` 入口方法**
  - 文件: `src/iter/indexed.rs`
  - 内容: `indexed_iter()` 构造 `IndexedIter`
  - 测试: `test_tensor_indexed_iter`
  - 前置: T16
  - 预计: 5 min

### Wave 6: Zip 多数组迭代

- [ ] **T18: 实现 `ZipPart` trait + 基础 `Zip` 结构体**
  - 文件: `src/iter/zip.rs`
  - 内容: `ZipPart<D>` trait 定义、`Zip<Parts, D>` 结构体、`Zip::from()` 构造
  - 测试: 编译通过
  - 前置: T1, broadcast 模块
  - 预计: 10 min

- [ ] **T19: 实现 `Zip::and()` 链式构建 + 广播检查**
  - 文件: `src/iter/zip.rs`
  - 内容: `.and()` 方法、形状兼容性检查、广播形状计算
  - 测试: `test_zip_same_shape`, `test_zip_broadcast`, `test_zip_shape_mismatch`
  - 前置: T18
  - 预计: 10 min

- [ ] **T20: 实现 `Zip::for_each()` 执行方法**
  - 文件: `src/iter/zip.rs`
  - 内容: 二元/三元 `for_each()`、步长感知元素访问、广播索引映射
  - 测试: `test_zip_for_each_binary`, `test_zip_for_each_ternary`, `test_zip_for_each_broadcast`
  - 前置: T19
  - 预计: 10 min

- [ ] **T21: 实现 `zip()` 便捷函数**
  - 文件: `src/iter/zip.rs`
  - 内容: `zip()` 函数、`for_each` 便捷写法
  - 测试: `test_zip_convenience_fn`
  - 前置: T19
  - 预计: 5 min

### Wave 7: 并行迭代器

- [ ] **T22: 实现 `ParIter` — rayon `ParallelIterator`**
  - 文件: `src/iter/parallel.rs`（新文件，feature-gated）
  - 内容: `ParIter` 结构体、`ParallelIterator` trait、`IndexedParallelIterator`、Producer 实现
  - 测试: `test_par_iter_sum`, `test_par_iter_non_contiguous`（仅 `#[cfg(feature = "parallel")]`）
  - 前置: T4, rayon 依赖
  - 预计: 15 min

- [ ] **T23: 实现 `ParIterMut` — 可变并行迭代器**
  - 文件: `src/iter/parallel.rs`
  - 内容: `ParIterMut` 结构体、`ParallelIterator` trait、Producer 实现
  - 测试: `test_par_iter_mut_write`
  - 前置: T22
  - 预计: 10 min

- [ ] **T24: 实现 `TensorBase` 上的 `par_iter()` / `par_iter_mut()` 入口**
  - 文件: `src/iter/parallel.rs`
  - 内容: feature-gated 入口方法
  - 测试: `test_tensor_par_iter`, `test_tensor_par_iter_mut`
  - 前置: T22, T23
  - 预计: 5 min

### Wave 8: 集成与收尾

- [ ] **T25: `lib.rs` re-export 与集成**
  - 文件: `src/lib.rs`
  - 内容: `pub mod iter;`、re-export 公共类型（`Iter`, `IterMut`, `Axis`, `Windows` 等）
  - 测试: `use xenon::iter::*;` 编译通过
  - 前置: T1-T24
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于各文件内的 `#[cfg(test)] mod tests`：

#### elements.rs 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_iter_f_contig` | F-order 连续数组：迭代顺序与物理布局一致 |
| `test_iter_c_contig` | C-order 连续数组：迭代顺序与物理布局一致 |
| `test_iter_non_contiguous` | 非连续数组（切片后）：步长跳转正确 |
| `test_iter_empty` | 空数组：立即返回 None |
| `test_iter_single_element` | 单元素数组：返回一个元素后结束 |
| `test_iter_neg_stride` | 负步长数组：正确处理反转轴 |
| `test_iter_zero_stride` | 零步长数组（广播）：正确处理广播维度 |
| `test_iter_rev` | DoubleEndedIterator：正向和反向交替调用 |
| `test_iter_exact_len` | ExactSizeIterator：`len()` 等于 `tensor.len()` |
| `test_iter_mut_write` | IterMut：写入后原数组被修改 |
| `test_iter_mut_double_ended` | IterMut：双端迭代正确 |
| `test_iter_mut_non_contiguous` | IterMut：非连续写入正确 |
| `test_into_iter_owned` | IntoIter：消费 Tensor，元素正确 |
| `test_into_iter_view` | IntoIterator for TensorView：产 Iter |
| `test_into_iter_view_mut` | IntoIterator for TensorViewMut：产 IterMut |
| `test_tensor_iter` | TensorBase::iter() 入口方法 |
| `test_tensor_iter_mut` | TensorBase::iter_mut() 入口方法 |

#### axis.rs 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_axis_iter_2d_rows` | 2D [3,4] 沿 axis=0 迭代 → 3 个 [4] 子视图 |
| `test_axis_iter_2d_cols` | 2D [3,4] 沿 axis=1 迭代 → 4 个 [3] 子视图 |
| `test_axis_iter_3d` | 3D [2,3,4] 沿 axis=1 迭代 → 3 个 [2,4] 子视图 |
| `test_axis_iter_rev` | 反向迭代：最后一个先产出 |
| `test_axis_iter_exact_len` | 长度等于 axis_len |
| `test_axis_iter_mut_write` | 可变轴迭代：写入通过源数据可见 |
| `test_axis_iter_mut_isolation` | 各子视图地址不重叠 |
| `test_axis_iter_non_contiguous` | 非连续数组的轴迭代 |
| `test_axis_iter_panic_invalid` | 无效轴索引 panic |

#### windows.rs 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_windows_1d` | 1D [1,2,3,4] 窗口 [2] → 3 个窗口 |
| `test_windows_2d` | 2D [3,4] 窗口 [2,2] → 2×3 = 6 个窗口 |
| `test_windows_exact_len` | 窗口数 = Π(shape[i] - window[i] + 1) |
| `test_windows_empty` | 窗口大小 = shape → 1 个窗口 |
| `test_windows_no_incomplete` | 无不完整窗口 |
| `test_windows_panic_zero` | window[i] == 0 时 panic |
| `test_windows_panic_oversize` | window[i] > shape[i] 时 panic |

#### indexed.rs 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_indexed_iter_1d` | 1D 迭代：索引 [0],[1],[2]... |
| `test_indexed_iter_2d` | 2D 迭代：索引为 Ix2 |
| `test_indexed_iter_values` | 产出的值与直接索引一致 |
| `test_indexed_iter_empty` | 空数组 |

#### zip.rs 测试

| 测试函数 | 测试内容 |
|----------|----------|
| `test_zip_same_shape` | 相同形状：正确配对元素 |
| `test_zip_broadcast` | 可广播形状：正确扩展 |
| `test_zip_shape_mismatch` | 不可广播形状：返回错误 |
| `test_zip_for_each_binary` | 二元 for_each：正确计算 |
| `test_zip_for_each_ternary` | 三元 for_each：两读一写 |
| `test_zip_for_each_broadcast` | 广播 + for_each：正确 |
| `test_zip_non_contiguous` | 非连续数组：正确配对 |

### 7.2 集成测试

| 文件 | 测试内容 |
|------|----------|
| `tests/iterator.rs` | 各迭代器的端到端使用场景 |

**集成测试用例：**

| 测试函数 | 测试内容 |
|----------|----------|
| `test_iter_sum` | 用 iter() 实现手动求和，结果与已知值一致 |
| `test_iter_mut_normalize` | 用 iter_mut() 归一化数组 |
| `test_axis_iter_reduce_sum` | 用 axis_iter() 手动沿轴求和 |
| `test_windows_moving_average` | 用 windows() 实现滑动平均 |
| `test_zip_elementwise_add` | 用 Zip 实现逐元素加法 |
| `test_zip_broadcast_add_scalar` | 用 Zip 实现 scalar + tensor |

### 7.3 边界测试

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `iter()` 立即结束，`axis_iter(Axis(0))` 产出 0 项 |
| 单元素 `shape=[1, 1]` | `iter()` 产出 1 项，`axis_iter` 各轴产出 1 项 |
| 标量 `Ix0` | `iter()` 产出 1 项，`axis_iter` panic（无轴可迭代） |
| 窗口 = shape | `windows()` 产出 1 项（整个数组） |
| 窗口 > shape | `windows()` panic |
| 非连续切片 `t.slice(s![.., 0..3])` | `iter()` 正确处理步长跳转 |
| 反转轴 `neg_stride` | `iter()` 正确处理负步长 |
| 广播视图 `broadcast [1,4] → [3,4]` | `iter()` 产出 12 项（含重复引用） |
| 大数组 (> 64K 元素) | `par_iter()` 自动启用并行（feature-gated） |

### 7.4 属性测试

| 不变量 | 测试方法 |
|--------|----------|
| `iter().count() == tensor.len()` | 随机形状 [0..10, 0..10, 0..10] |
| `iter().rev().collect()` | 反转轴 `neg_stride` | `iter()` 正确处理负步长 |
| 广播视图 `broadcast [1,4] → [3,4]` | `iter()` 产出 12 项（含重复引用） |

### 7.4 性能基准测试

| 基准 | 测量内容 |
|------|----------|
| `bench_iter_contiguous` | 连续数组 iter() 吞吐量 |
| `bench_iter_non_contiguous` | 非连续数组 iter() 吞吐量 |
| `bench_iter_mut_contiguous` | 连续可变迭代吞吐量 |
| `bench_axis_iter` | 轴迭代创建子视图开销 |
| `bench_windows` | 窗口迭代吞吐量 |
| `bench_zip_binary` | 二元 Zip for_each 吞吐量 |
| `bench_par_iter` | 并行迭代吞吐量 vs 串行 |

### 7.5 属性测试

| 不变量 | 测试方法 |
|--------|----------|
| `iter().count() == tensor.len()` | 随机形状和布局 |
| `iter().rev()` 产出与 `iter().collect::<Vec<_>>().rev()` 相同 | 随机形状 |
| `axis_iter(Axis(i)).count() == shape[i]` | 随机形状 |
| `windows(w).count() == Π(shape[j] - w[j] + 1)` | 随机形状和窗口大小 |
| `indexed_iter` 的所有索引唯一 | 随机形状 |
| `zip` 的 for_each 等价于嵌套 for 循环 | 随机形状 |
