# 迭代器模块设计文档

> **模块路径**: `src/iter/`
> **版本**: v18
> **日期**: 2026-03-28
> **前置文档**: 02-project-architecture.md, 03-06-tensor-core.md, 03-05-memory-layout.md

---

## 1. 模块概述

### 1.1 定位

迭代器模块是 Senon 张量库的数据遍历基础设施，为所有张量操作提供统一、高效、类型安全的遍历机制。它实现了多种迭代模式，支持从简单的元素遍历到复杂的多张量同步迭代。

### 1.2 核心职责

| 职责 | 说明 |
|------|------|
| 元素遍历 | 按内存布局或指定顺序遍历所有元素 |
| 结构化遍历 | 沿轴、窗口、通道等结构化维度遍历 |
| 同步遍历 | 多张量同步迭代，支持广播 |
| 索引访问 | 遍历时提供多维索引信息 |
| 并行支持 | 与 rayon 集成，支持数据并行 |

### 1.3 设计理念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         迭代器系统架构                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Iterator Trait 层                            │   │
│  │  std::iter::Iterator  +  IntoIterator  +  DoubleEndedIterator  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│  ┌───────────────────────────────┼───────────────────────────────┐     │
│  │                               │                               │     │
│  ▼                               ▼                               ▼     │
│ ┌─────────────┐          ┌─────────────┐          ┌─────────────┐     │
│ │  Elements   │          │  AxisIter   │          │   Windows   │     │
│ │  扁平遍历    │          │  轴迭代     │          │  窗口迭代    │     │
│ └─────────────┘          └─────────────┘          └─────────────┘     │
│                                                                         │
│ ┌─────────────┐          ┌─────────────┐          ┌─────────────┐     │
│ │  Indexed    │          │    Zip      │          │   Lanes     │     │
│ │  索引迭代    │          │  同步迭代    │          │  行列迭代    │     │
│ └─────────────┘          └─────────────┘          └─────────────┘     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Producer Pattern                             │   │
│  │          (用于 Zip 和并行迭代的可分割生产者)                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                      │
│  ┌───────────────────────────────┼───────────────────────────────┐     │
│  │                               │                               │     │
│  ▼                               ▼                               ▼     │
│ ┌──────────────────────────────────────────────────────────────────┐  │
│ │                        基础设施层                                 │  │
│ │   Order (遍历顺序)  │  StrideState (步长状态机)  │  NdProducer   │  │
│ └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 与 ndarray 的对比

| 特性 | ndarray | Senon |
|------|---------|-------|
| 元素迭代 | `iter()` / `iter_mut()` | `iter()` / `iter_mut()` |
| 轴迭代 | `axis_iter()` | `axis_iter()` |
| 窗口迭代 | `windows()` | `windows()` |
| 索引迭代 | `indexed_iter()` | `indexed_iter()` |
| Zip 模式 | `azip!` 宏 + `Zip` | `Zip` struct（无宏） |
| 并行迭代 | `par_iter()` (rayon) | `par_iter()` (rayon) |
| 默认顺序 | 逻辑顺序（C/F） | 物理内存布局优先 |
| Producer | `NdProducer` trait | `NdProducer` trait |

---

## 2. 文件结构

```
src/iter/
├── mod.rs             # 迭代器 trait 定义、公开导出
├── elements.rs        # Elements 迭代器（扁平遍历）
├── axis.rs            # AxisIter 沿轴迭代
├── windows.rs         # Windows 窗口迭代
├── indexed.rs         # IndexedIter 带索引迭代
├── zip.rs             # Zip 多张量同步迭代
├── lanes.rs           # LaneIter 行/列迭代
├── producer.rs        # NdProducer trait 和实现
├── stride_state.rs    # StrideState 步长状态机
└── order.rs           # Order 遍历顺序枚举
```

### 2.1 各文件职责

| 文件 | 职责 | 可见性 |
|------|------|--------|
| `mod.rs` | 模块导出、公开 API 重新导出 | pub |
| `elements.rs` | 元素迭代器实现 | pub（通过 re-export） |
| `axis.rs` | 轴迭代器实现 | pub |
| `windows.rs` | 窗口迭代器实现 | pub |
| `indexed.rs` | 索引迭代器实现 | pub |
| `zip.rs` | 多张量同步迭代实现 | pub |
| `lanes.rs` | 行/列迭代器实现 | pub |
| `producer.rs` | Producer trait 定义和实现 | pub（trait） |
| `stride_state.rs` | 步长遍历状态机 | pub(crate) |
| `order.rs` | 遍历顺序定义 | pub |

---

## 3. Elements 迭代器

### 3.1 概述

`Elements` 是最基本的迭代器，按指定顺序遍历张量中的所有元素。支持连续和非连续内存布局，支持 F-order 和 C-order 遍历。

### 3.2 结构体定义

```rust
/// 元素迭代器，按指定顺序遍历张量的所有元素。
///
/// # 内存布局感知
///
/// - 连续数组：直接指针递增，最高性能
/// - 非连续数组：使用 StrideState 状态机处理复杂步长
///
/// # 遍历顺序
///
/// - `Order::Default`：按物理内存布局（连续数组最优）
/// - `Order::F`：强制 Fortran 顺序（列优先）
/// - `Order::C`：强制 C 顺序（行优先）
///
/// # 示例
///
/// ```
/// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
/// 
/// // 默认顺序（物理内存布局）
/// for &elem in tensor.iter() {
///     println!("{}", elem);
/// }
///
/// // 强制 C 顺序
/// for &elem in tensor.iter_order(Order::C) {
///     println!("{}", elem);
/// }
/// ```
pub struct Elements<'a, A, D>
where
    D: Dimension,
{
    /// 源张量的不可变视图。
    view: TensorView<'a, A, D>,
    
    /// 当前遍历位置的多维索引。
    /// 对于连续数组，此字段可能不使用（使用指针直接遍历）。
    index: D,
    
    /// 剩余元素数量。
    remaining: usize,
    
    /// 遍历顺序。
    order: Order,
    
    /// 步长状态机（非连续数组使用）。
    stride_state: StrideState<D>,
    
    /// 当前元素指针（连续数组使用）。
    ptr: *const A,
    
    /// 结束指针（连续数组使用）。
    end: *const A,
    
    /// 标记：是否使用快速路径（连续数组）。
    fast_path: bool,
}

/// 可变元素迭代器。
pub struct ElementsMut<'a, A, D>
where
    D: Dimension,
{
    view: TensorViewMut<'a, A, D>,
    index: D,
    remaining: usize,
    order: Order,
    stride_state: StrideState<D>,
    ptr: *mut A,
    end: *mut A,
    fast_path: bool,
}
```

### 3.3 Iterator trait 实现

```rust
impl<'a, A, D> Iterator for Elements<'a, A, D>
where
    D: Dimension,
{
    type Item = &'a A;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        
        self.remaining -= 1;
        
        if self.fast_path {
            // Fast path: contiguous array, direct pointer increment
            let ptr = self.ptr;
            // SAFETY: ptr is valid and within bounds, checked by remaining
            self.ptr = unsafe { ptr.add(1) };
            Some(unsafe { &*ptr })
        } else {
            // Slow path: non-contiguous array, use stride state machine
            let elem = self.view.get(self.index.slice()).unwrap();
            self.stride_state.advance(&mut self.index);
            Some(elem)
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
    
    fn count(self) -> usize {
        self.remaining
    }
}

impl<'a, A, D> ExactSizeIterator for Elements<'a, A, D>
where
    D: Dimension,
{}

impl<'a, A, D> DoubleEndedIterator for Elements<'a, A, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        // Only supported for contiguous arrays
        if !self.fast_path || self.remaining == 0 {
            return None;
        }
        
        self.remaining -= 1;
        self.end = unsafe { self.end.sub(1) };
        Some(unsafe { &*self.end })
    }
}

// ElementsMut 的实现类似，返回 &mut A
impl<'a, A, D> Iterator for ElementsMut<'a, A, D>
where
    D: Dimension,
{
    type Item = &'a mut A;
    // ... 实现与 Elements 类似
}
```

### 3.4 构造方法

```rust
impl<'a, A, D> Elements<'a, A, D>
where
    D: Dimension,
{
    /// 创建元素迭代器，使用默认遍历顺序。
    ///
    /// 默认顺序为物理内存布局顺序（对连续数组最优）。
    pub(crate) fn new(view: TensorView<'a, A, D>) -> Self {
        Self::with_order(view, Order::Default)
    }
    
    /// 创建元素迭代器，指定遍历顺序。
    pub(crate) fn with_order(view: TensorView<'a, A, D>, order: Order) -> Self {
        let len = view.len();
        let actual_order = order.resolve(&view);
        
        // Determine if we can use fast path
        let fast_path = view.is_contiguous() 
            && matches!(order, Order::Default | Order::F)
            && view.is_f_contiguous()
            || view.is_c_contiguous() 
            && matches!(order, Order::Default | Order::C);
        
        let (ptr, end, stride_state, index) = if fast_path {
            let ptr = view.as_ptr();
            let end = unsafe { ptr.add(len) };
            (ptr, end, StrideState::default(), D::zeros(view.ndim()))
        } else {
            let stride_state = StrideState::new(&view, actual_order);
            (std::ptr::null(), std::ptr::null(), stride_state, D::zeros(view.ndim()))
        };
        
        Elements {
            view,
            index,
            remaining: len,
            order: actual_order,
            stride_state,
            ptr,
            end,
            fast_path,
        }
    }
}
```

### 3.5 连续 vs 非连续性能差异

| 场景 | 实现路径 | 每元素开销 | 缓存友好性 |
|------|----------|-----------|-----------|
| 连续 F-order | 指针递增 | 1 次指针加法 | 最优（顺序访问） |
| 连续 C-order | 指针递增 | 1 次指针加法 | 最优（顺序访问） |
| 非连续（切片） | 步长状态机 | 多次乘加运算 | 较差（跳跃访问） |
| 非连续（转置） | 步长状态机 | 多次乘加运算 | 较差（跳跃访问） |
| 广播视图 | 零步长处理 | 条件判断 + 缓存 | 依赖广播维度 |

**性能数据（参考）**:

| 操作 | 连续数组 | 非连续数组（步长=2） | 性能比 |
|------|----------|---------------------|--------|
| 遍历 1M 元素 | ~1ms | ~3ms | 3x |
| 缓存命中率 | ~95% | ~60% | - |
| SIMD 友好度 | 高 | 低 | - |

---

## 4. AxisIter 沿轴迭代

### 4.1 概述

`AxisIter` 沿指定轴遍历，每次产出降维后的子视图。例如，对 3D 张量沿轴 0 迭代，产出 2D 子视图。

### 4.2 结构体定义

```rust
/// 沿轴迭代器，每次产出降维后的子视图。
///
/// # 维度变化
///
/// - 输入维度: N
/// - 输出维度: N-1
///
/// # 示例
///
/// ```
/// let tensor = Tensor3::<f64>::zeros([4, 3, 2]);
///
/// // 沿轴 0 迭代，产出 3x2 的 2D 视图
/// for view in tensor.axis_iter(Axis(0)) {
///     assert_eq!(view.shape(), &[3, 2]);
/// }
///
/// // 沿轴 1 迭代，产出 4x2 的 2D 视图
/// for view in tensor.axis_iter(Axis(1)) {
///     assert_eq!(view.shape(), &[4, 2]);
/// }
/// ```
pub struct AxisIter<'a, A, D>
where
    D: Dimension,
{
    /// 源张量的不可变视图。
    view: TensorView<'a, A, D>,
    
    /// 迭代的轴。
    axis: Axis,
    
    /// 轴长度（迭代次数）。
    len: usize,
    
    /// 当前位置（沿轴的索引）。
    current: usize,
}

/// 沿轴可变迭代器。
///
/// 与 `AxisIter` 类似，但产出可变视图。
/// 由于独占借用语义，不能同时存在多个可变视图。
pub struct AxisIterMut<'a, A, D>
where
    D: Dimension,
{
    view: TensorViewMut<'a, A, D>,
    axis: Axis,
    len: usize,
    current: usize,
}
```

### 4.3 Iterator trait 实现

```rust
impl<'a, A, D> Iterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    type Item = TensorView<'a, A, D::Smaller>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.len {
            return None;
        }
        
        let view = self.view.index_axis(self.axis, self.current);
        self.current += 1;
        Some(view)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.current;
        (remaining, Some(remaining))
    }
    
    fn count(self) -> usize {
        self.len - self.current
    }
}

impl<'a, A, D> ExactSizeIterator for AxisIter<'a, A, D>
where
    D: Dimension,
{}

impl<'a, A, D> DoubleEndedIterator for AxisIter<'a, A, D>
where
    D: Dimension,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current >= self.len {
            return None;
        }
        
        self.len -= 1;
        let view = self.view.index_axis(self.axis, self.len);
        Some(view)
    }
}
```

### 4.4 签名设计

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 沿指定轴迭代，产出降维视图。
    ///
    /// # 参数
    ///
    /// * `axis` - 迭代的轴索引
    ///
    /// # 返回
    ///
    /// 产出 `TensorView<'_, A, D::Smaller>`，维度数减少 1。
    ///
    /// # Panics
    ///
    /// 若 `axis` 超出维度数，panic。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor2::from_shape_vec([3, 4], vec![...])?;
    /// 
    /// // 沿行迭代（轴 0），产出列向量
    /// for row in tensor.axis_iter(Axis(0)) {
    ///     // row: TensorView1<'_, f64>
    /// }
    /// ```
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, A, D::Smaller>
    where
        D: RemoveAxis,
    {
        let len = self.shape()[axis.0];
        AxisIter {
            view: self.view(),
            axis,
            len,
            current: 0,
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// 沿指定轴可变迭代。
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, A, D::Smaller>
    where
        D: RemoveAxis,
    {
        let len = self.shape()[axis.0];
        AxisIterMut {
            view: self.view_mut(),
            axis,
            len,
            current: 0,
        }
    }
}
```

---

## 5. Windows 窗口迭代

### 5.1 概述

`Windows` 滑动窗口迭代器，在张量上滑动指定大小的窗口，产出每个窗口的视图。

### 5.2 结构体定义

```rust
/// 滑动窗口迭代器。
///
/// # 窗口产出规则
///
/// - 窗口数 = (shape[i] - window_size[i] + 1) for each dimension
/// - 边界处理：不产出不完整窗口
/// - 空数组或窗口大于数组：产出零个窗口
///
/// # 示例
///
/// ```
/// let tensor = Tensor1::from_vec(vec![1, 2, 3, 4, 5]);
///
/// // 大小为 3 的滑动窗口
/// let windows: Vec<_> = tensor.windows(3).collect();
/// // 产出: [1,2,3], [2,3,4], [3,4,5]
/// assert_eq!(windows.len(), 3);
///
/// // 2D 窗口
/// let tensor2d = Tensor2::zeros([4, 5]);
/// for window in tensor2d.windows([2, 3]) {
///     // window: TensorView2<'_, f64>，形状 [2, 3]
/// }
/// ```
pub struct Windows<'a, A, D>
where
    D: Dimension,
{
    /// 源张量的不可变视图。
    view: TensorView<'a, A, D>,
    
    /// 窗口大小。
    window_size: D,
    
    /// 当前窗口起始位置。
    index: D,
    
    /// 剩余窗口数。
    remaining: usize,
    
    /// 步长状态机。
    stride_state: StrideState<D>,
}

/// 滑动窗口可变迭代器。
pub struct WindowsMut<'a, A, D>
where
    D: Dimension,
{
    view: TensorViewMut<'a, A, D>,
    window_size: D,
    index: D,
    remaining: usize,
    stride_state: StrideState<D>,
}
```

### 5.3 Iterator trait 实现

```rust
impl<'a, A, D> Iterator for Windows<'a, A, D>
where
    D: Dimension,
{
    type Item = TensorView<'a, A, D>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        
        self.remaining -= 1;
        
        // Create slice for current window
        let slices = self.compute_slices();
        let window = self.view.slice(slices);
        
        // Advance to next window position
        self.stride_state.advance(&mut self.index);
        
        Some(window)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, A, D> Windows<'a, A, D>
where
    D: Dimension,
{
    /// 计算当前窗口的切片范围。
    fn compute_slices(&self) -> Vec<SliceInfoElem> {
        self.index
            .slice()
            .iter()
            .zip(self.window_size.slice().iter())
            .map(|(&start, &size)| {
                SliceInfoElem::Range(start..(start + size))
            })
            .collect()
    }
}
```

### 5.4 窗口大小验证

```rust
impl<'a, A, D> Windows<'a, A, D>
where
    D: Dimension,
{
    /// 创建窗口迭代器。
    ///
    /// # 参数
    ///
    /// * `view` - 源张量视图
    /// * `window_size` - 窗口大小（每个维度）
    ///
    /// # 返回
    ///
    /// 若窗口大小有效，返回 `Some(Windows)`。
    /// 若任一维度的窗口大小 > 数组大小，返回 `None`。
    ///
    /// # 示例
    ///
    /// ```
    /// let tensor = Tensor1::from_vec(vec![1, 2, 3]);
    ///
    /// // 有效窗口
    /// assert!(tensor.windows(2).is_some());
    ///
    /// // 无效窗口（窗口大于数组）
    /// assert!(tensor.windows(5).is_none());
    /// ```
    pub(crate) fn new(view: TensorView<'a, A, D>, window_size: D) -> Option<Self> {
        // Validate window size
        for (&s, &ws) in view.shape().iter().zip(window_size.slice().iter()) {
            if ws > s {
                return None;
            }
            if ws == 0 {
                return None;
            }
        }
        
        // Calculate number of windows
        let remaining = view
            .shape()
            .iter()
            .zip(window_size.slice().iter())
            .map(|(&s, &ws)| s - ws + 1)
            .product();
        
        // Calculate strides for window positions
        let stride_state = StrideState::new_for_windows(&view);
        
        Some(Windows {
            view,
            window_size,
            index: D::zeros(view.ndim()),
            remaining,
            stride_state,
        })
    }
}
```

### 5.5 边界处理

| 场景 | 行为 | 产出 |
|------|------|------|
| 正常窗口 | 产出所有完整窗口 | `product(shape - window + 1)` 个 |
| 窗口 = 数组大小 | 产出整个数组 | 1 个窗口 |
| 窗口 > 数组大小 | 不产出 | 0 个窗口（返回 None） |
| 空数组 | 不产出 | 0 个窗口 |
| 零维数组 | 产出空窗口 | 1 个窗口（若 window_size 也为空） |

---

## 6. IndexedIter 带索引迭代

### 6.1 概述

`IndexedIter` 在遍历元素的同时提供多维索引，适用于需要知道元素位置的场合。

### 6.2 结构体定义

```rust
/// 带索引的元素迭代器。
///
/// 产出 `(索引, 元素引用)` 元组。
///
/// # 示例
///
/// ```
/// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
///
/// for (index, &elem) in tensor.indexed_iter() {
///     println!("tensor[{:?}] = {}", index, elem);
/// }
/// // 输出:
/// // tensor[[0, 0]] = 1
/// // tensor[[0, 1]] = 2
/// // tensor[[0, 2]] = 3
/// // tensor[[1, 0]] = 4
/// // ...
/// ```
pub struct IndexedIter<'a, A, D>
where
    D: Dimension,
{
    /// 内部元素迭代器。
    elements: Elements<'a, A, D>,
    
    /// 当前索引。
    index: D,
    
    /// 步长状态机（用于更新索引）。
    stride_state: StrideState<D>,
}

/// 带索引的可变元素迭代器。
pub struct IndexedIterMut<'a, A, D>
where
    D: Dimension,
{
    elements: ElementsMut<'a, A, D>,
    index: D,
    stride_state: StrideState<D>,
}
```

### 6.3 Iterator trait 实现

```rust
impl<'a, A, D> Iterator for IndexedIter<'a, A, D>
where
    D: Dimension,
{
    /// 产出 (索引, 元素引用) 元组。
    ///
    /// 索引类型为 `D::Slice`（通常是 `&[usize]` 或固定大小数组）。
    type Item = (D::Slice, &'a A);
    
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.elements.next()?;
        let idx = self.index.clone();
        
        // Advance index for next iteration
        self.stride_state.advance(&mut self.index);
        
        Some((idx.slice(), elem))
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.elements.size_hint()
    }
}

impl<'a, A, D> ExactSizeIterator for IndexedIter<'a, A, D>
where
    D: Dimension,
{}
```

### 6.4 索引类型

```rust
/// 索引表示，取决于维度类型。
///
/// | 维度类型 | Slice 类型 | 示例 |
/// |----------|-----------|------|
/// | Ix0 | &[usize] (空) | &[] |
/// | Ix1 | &[usize; 1] | &[0] |
/// | Ix2 | &[usize; 2] | &[0, 1] |
/// | IxDyn | &[usize] | &[0, 1, 2] |
pub trait Dimension {
    /// 切片表示类型。
    type Slice: AsRef<[usize]>;
    
    /// 返回维度的切片表示。
    fn slice(&self) -> Self::Slice;
}
```

---

## 7. Zip 多数组同步迭代

### 7.1 概述

`Zip` 是 Senon 迭代器系统中最强大的组件，支持多个张量的同步迭代。它实现了 ndarray 风格的 Producer pattern，支持广播和并行化。

### 7.2 结构体定义

```rust
/// 多张量同步迭代器。
///
/// # 广播支持
///
/// - 形状完全一致：直接同步迭代
/// - 形状可广播：自动扩展后迭代
/// - 形状不可广播：构造时返回 `BroadcastError`
///
/// # 示例
///
/// ```
/// let a = Tensor1::from_vec(vec![1, 2, 3]);
/// let b = Tensor1::from_vec(vec![4, 5, 6]);
/// let mut c = Tensor1::zeros(3);
///
/// // 同步迭代三个数组
/// Zip::from(&mut c)
///     .and(&a)
///     .and(&b)
///     .for_each(|c, &a, &b| {
///         *c = a + b;
///     });
///
/// assert_eq!(c.as_slice(), &[5, 7, 9]);
/// ```
///
/// # 广播示例
///
/// ```
/// let a = Tensor2::from_shape_vec([2, 3], vec![...])?;  // 2x3
/// let b = Tensor1::from_vec(vec![1, 2, 3]);            // 3 (广播为 1x3)
///
/// // b 自动广播以匹配 a 的形状
/// Zip::from(&a)
///     .and(&b)
///     .for_each(|&a, &b| {
///         println!("a = {}, b = {}", a, b);
///     });
/// ```
pub struct Zip<Parts, D>
where
    D: Dimension,
{
    /// 参与迭代的生产者元组。
    parts: Parts,
    
    /// 广播后的形状。
    shape: D,
    
    /// 广播后的步长（每个生产者）。
    strides: Vec<D>,
    
    /// 遍历顺序。
    order: Order,
    
    /// 是否可并行（所有生产者都支持）。
    parallel_capable: bool,
}
```

### 7.3 Producer Pattern

Producer pattern 是 ndarray 的核心设计，Senon 采用相同模式：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Producer Pattern                                │
│                                                                          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │   NdProducer    │     │  Splitable      │     │  Iterable       │   │
│  │   Trait         │────▶│  Producer       │────▶│  (Iterator)     │   │
│  └─────────────────┘     └────────────────-┘     └─────────────────┘   │
│          │                       │                        │             │
│          ▼                       ▼                        ▼             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  实现者:                                                         │   │
│  │  - TensorView/ViewMut (直接实现)                                 │   │
│  │  - Broadcast (广播视图)                                          │   │
│  │  - Elements (元素迭代)                                           │   │
│  │  - LaneProducer (行/列迭代)                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  关键方法:                                                       │   │
│  │                                                                  │   │
│  │  trait NdProducer {                                             │   │
│  │      type Item;                                                 │   │
│  │      type Dim: Dimension;                                       │   │
│  │                                                                 │   │
│  │      // 分割为两个独立的生产者（用于并行）                          │   │
│  │      fn split_at(self, axis: Axis, index: usize)                │   │
│  │          -> (Self, Self);                                       │   │
│  │                                                                 │   │
│  │      // 创建迭代器                                               │   │
│  │      fn into_iter(self) -> Self::Iter;                          │   │
│  │                                                                 │   │
│  │      // 获取形状                                                 │   │
│  │      fn shape(&self) -> &Self::Dim;                             │   │
│  │  }                                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 NdProducer Trait 定义

```rust
/// N 维生产者 trait。
///
/// 生产者是可以被分割和迭代的对象，用于 Zip 和并行迭代。
///
/// # 安全性
///
/// `split_at` 返回的两个生产者必须访问不重叠的元素。
pub trait NdProducer: Sized {
    /// 产出的元素类型。
    type Item;
    
    /// 维度类型。
    type Dim: Dimension;
    
    /// 迭代器类型。
    type Iter: Iterator<Item = Self::Item>;
    
    /// 返回形状。
    fn shape(&self) -> &Self::Dim;
    
    /// 沿指定轴在指定位置分割。
    ///
    /// # 安全性
    ///
    /// 返回的两个生产者必须访问不重叠的元素区间。
    /// 这对于并行迭代的安全性至关重要。
    ///
    /// # 参数
    ///
    /// * `axis` - 分割的轴
    /// * `index` - 分割位置（左侧包含 index，右侧从 index 开始）
    ///
    /// # 返回
    ///
    /// `(left, right)` 两个独立的生产者。
    fn split_at(self, axis: Axis, index: usize) -> (Self, Self);
    
    /// 转换为迭代器。
    fn into_iter(self) -> Self::Iter;
    
    /// 是否支持并行迭代。
    ///
    /// 某些生产者（如广播视图）可能不支持并行。
    fn parallel_capable(&self) -> bool {
        true
    }
}

/// 可变生产者。
pub trait NdProducerMut: NdProducer {
    /// 转换为可变迭代器。
    fn into_iter_mut(self) -> Self::IterMut;
}
```

### 7.5 Zip 构造与组合

```rust
impl<D> Zip<(), D>
where
    D: Dimension,
{
    /// 从第一个张量创建 Zip。
    ///
    /// 第一个张量决定迭代形状。
    pub fn from<P>(producer: P) -> Zip<(P,), D>
    where
        P: NdProducer<Dim = D>,
    {
        Zip {
            parts: (producer,),
            shape: D::zeros(0), // 从 producer 获取
            strides: vec![],
            order: Order::Default,
            parallel_capable: true,
        }
    }
}

impl<Parts, D> Zip<Parts, D>
where
    D: Dimension,
{
    /// 添加另一个张量到 Zip。
    ///
    /// # 广播
    ///
    /// 若形状不一致，尝试广播。广播失败返回 `BroadcastError`。
    ///
    /// # 示例
    ///
    /// ```
    /// Zip::from(&mut result)
    ///     .and(&a)       // a 广播以匹配 result
    ///     .and(&b)       // b 广播以匹配 result
    ///     .for_each(|r, &a, &b| *r = a + b);
    /// ```
    pub fn and<P>(self, producer: P) -> Result<Zip<(Parts, P), D>, BroadcastError>
    where
        P: NdProducer<Dim = D>,
    {
        // Check broadcast compatibility
        let broadcast_shape = broadcast_shapes(&[self.shape.slice(), producer.shape().slice()])?;
        
        Ok(Zip {
            parts: (self.parts, producer),
            shape: broadcast_shape.into_dimension(),
            strides: self.strides,
            order: self.order,
            parallel_capable: self.parallel_capable && producer.parallel_capable(),
        })
    }
    
    /// 设置遍历顺序。
    pub fn order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }
}
```

### 7.6 Zip 迭代协议

```rust
impl<Parts, D> Zip<Parts, D>
where
    D: Dimension,
    Parts: ZipParts<D>,
{
    /// 对每个元素执行闭包。
    ///
    /// # 参数
    ///
    /// 闭包接收每个生产者产出的元素。
    ///
    /// # 示例
    ///
    /// ```
    /// Zip::from(&a)
    ///     .and(&b)
    ///     .for_each(|&a, &b| {
    ///         println!("a={}, b={}", a, b);
    ///     });
    /// ```
    pub fn for_each<F>(self, mut f: F)
    where
        F: FnMut(Parts::Item),
    {
        let mut iter = self.into_iter();
        while let Some(item) = iter.next() {
            f(item);
        }
    }
    
    /// 应用函数并收集结果。
    ///
    /// # 示例
    ///
    /// ```
    /// let result: Tensor1<f64> = Zip::from(&a)
    ///     .and(&b)
    ///     .map_collect(|&a, &b| a + b)?;
    /// ```
    pub fn map_collect<F, A>(self, f: F) -> Tensor<A, D>
    where
        F: FnMut(Parts::Item) -> A,
        A: Element,
    {
        let shape = self.shape.clone();
        let mut result = Tensor::uninitialized(shape);
        
        Zip::from(result.view_mut())
            .and(self)
            .for_each(|r, item| {
                *r = f(item);
            });
        
        result
    }
}

/// Zip 部分的聚合 trait。
pub trait ZipParts<D: Dimension> {
    /// 产出的元素类型（元组）。
    type Item;
    
    /// 转换为迭代器。
    fn into_iter(self) -> ZipIter<Self, D>
    where
        Self: Sized;
}
```

### 7.7 广播支持

```rust
/// 计算广播后的形状。
///
/// # 广播规则
///
/// 1. 从最右维度开始对齐
/// 2. 维度数不足的在左侧补 1
/// 3. 对应维度相等或其中一个为 1 则兼容
/// 4. 结果形状取每个维度的最大值
///
/// # 示例
///
/// ```
/// // (3, 1) + (1, 4) -> (3, 4)
/// let shapes = [&[3, 1][..], &[1, 4][..]];
/// let result = broadcast_shapes(&shapes)?;
/// assert_eq!(result, vec![3, 4]);
///
/// // (2, 3) + (3,) -> (2, 3)
/// let shapes = [&[2, 3][..], &[3][..]];
/// let result = broadcast_shapes(&shapes)?;
/// assert_eq!(result, vec![2, 3]);
///
/// // (2, 3) + (2,) -> Error (不可广播)
/// let shapes = [&[2, 3][..], &[2][..]];
/// assert!(broadcast_shapes(&shapes).is_err());
/// ```
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, BroadcastError> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }
    
    // Find maximum ndim
    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    
    let mut result = vec![1usize; max_ndim];
    
    for shape in shapes {
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let result_idx = offset + i;
            match (result[result_idx], dim) {
                (r, d) if r == d || d == 1 => {} // OK
                (1, d) => result[result_idx] = d, // Expand
                (r, d) => {
                    return Err(BroadcastError {
                        left: r,
                        right: d,
                        axis: result_idx,
                    });
                }
            }
        }
    }
    
    Ok(result)
}
```

---

## 8. LaneIter 行/列迭代

### 8.1 概述

`LaneIter` 遍历矩阵的行或列，产出 1D 子视图。对于高维数组，遍历最内层的 1D 子数组。

### 8.2 结构体定义

```rust
/// 行/列迭代器，产出 1D 子视图。
///
/// # 示例
///
/// ```
/// let matrix = Tensor2::from_shape_vec([3, 4], vec![...])?;
///
/// // 遍历行
/// for row in matrix.lanes(Axis(0)) {
///     // row: TensorView1<'_, f64>，长度 4
/// }
///
/// // 遍历列
/// for col in matrix.lanes(Axis(1)) {
///     // col: TensorView1<'_, f64>，长度 3
/// }
/// ```
pub struct LaneIter<'a, A, D>
where
    D: Dimension,
{
    /// 源张量视图。
    view: TensorView<'a, A, D>,
    
    /// 遍历的轴（产出此轴的 1D 视图）。
    axis: Axis,
    
    /// 轴长度。
    lane_len: usize,
    
    /// 其他维度的迭代状态。
    outer_iter: Elements<'a, A, D>,
}

/// 行/列可变迭代器。
pub struct LaneIterMut<'a, A, D>
where
    D: Dimension,
{
    view: TensorViewMut<'a, A, D>,
    axis: Axis,
    lane_len: usize,
    outer_iter: ElementsMut<'a, A, D>,
}
```

### 8.3 Iterator trait 实现

```rust
impl<'a, A, D> Iterator for LaneIter<'a, A, D>
where
    D: Dimension,
{
    type Item = TensorView1<'a, A>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Iterate over all other dimensions, yield 1D lanes
        // Implementation details...
        todo!()
    }
}
```

---

## 9. 遍历顺序控制

### 9.1 Order 枚举

```rust
/// 遍历顺序。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// 默认顺序：按物理内存布局。
    ///
    /// - F-contiguous 数组：按 F-order 遍历
    /// - C-contiguous 数组：按 C-order 遍历
    /// - 非连续数组：按存储顺序（通常是 F-order）
    Default,
    
    /// Fortran 顺序（列优先）。
    ///
    /// 最内层维度变化最快。
    /// 对于 2D 数组 [M, N]：遍历顺序为 (0,0), (1,0), (2,0), ..., (0,1), (1,1), ...
    F,
    
    /// C 顺序（行优先）。
    ///
    /// 最外层维度变化最快。
    /// 对于 2D 数组 [M, N]：遍历顺序为 (0,0), (0,1), (0,2), ..., (1,0), (1,1), ...
    C,
}

impl Order {
    /// 解析实际顺序。
    ///
    /// 对于 `Order::Default`，根据张量的内存布局决定实际顺序。
    pub fn resolve<A, D>(&self, tensor: &TensorView<'_, A, D>) -> Order
    where
        D: Dimension,
    {
        match self {
            Order::Default => {
                if tensor.is_f_contiguous() {
                    Order::F
                } else if tensor.is_c_contiguous() {
                    Order::C
                } else {
                    // Default to F for BLAS compatibility
                    Order::F
                }
            }
            order => *order,
        }
    }
}
```

### 9.2 物理顺序 vs 逻辑顺序

| 顺序类型 | 定义 | 性能特点 |
|----------|------|----------|
| **物理顺序** | 按内存地址递增遍历 | 最优缓存局部性 |
| **逻辑 F 顺序** | 按列优先逻辑索引遍历 | F-contiguous 时等于物理顺序 |
| **逻辑 C 顺序** | 按行优先逻辑索引遍历 | C-contiguous 时等于物理顺序 |

**性能影响**:

```
连续 F-order 数组 [1000, 1000]:

| 遍历顺序 | 时间 | 缓存命中率 |
|----------|------|-----------|
| 物理顺序 (F) | 1.0x | ~95% |
| 逻辑 C 顺序 | 10x | ~20% |

连续 C-order 数组 [1000, 1000]:

| 遍历顺序 | 时间 | 缓存命中率 |
|----------|------|-----------|
| 物理顺序 (C) | 1.0x | ~95% |
| 逻辑 F 顺序 | 10x | ~20% |
```

---

## 10. 并行迭代集成

### 10.1 rayon 集成点

```rust
#[cfg(feature = "parallel")]
pub mod parallel {
    use rayon::prelude::*;
    
    /// 并行元素迭代器。
    pub struct ParElements<'a, A, D>
    where
        D: Dimension,
    {
        view: TensorView<'a, A, D>,
        order: Order,
    }
    
    /// 并行 Zip 迭代器。
    pub struct ParZip<Parts, D>
    where
        D: Dimension,
    {
        zip: Zip<Parts, D>,
    }
    
    impl<Parts, D> ParZip<Parts, D>
    where
        D: Dimension,
        Parts: ZipParts<D> + Send,
        Parts::Item: Send,
    {
        /// 并行执行闭包。
        ///
        /// # 分块策略
        ///
        /// - 将元素按连续块分割
        /// - 每块不小于并行阈值（默认 4K 元素）
        /// - 使用 Producer 的 `split_at` 方法分割
        pub fn for_each<F>(self, f: F)
        where
            F: Fn(Parts::Item) + Sync + Send,
        {
            let producer = self.zip.into_producer();
            
            // Split into chunks and process in parallel
            producer
                .into_par_iter()
                .for_each(f);
        }
    }
    
    impl<'a, A, D> IntoParallelIterator for TensorView<'a, A, D>
    where
        D: Dimension,
        A: Sync,
    {
        type Item = &'a A;
        type Iter = ParElements<'a, A, D>;
        
        fn into_par_iter(self) -> Self::Iter {
            ParElements {
                view: self,
                order: Order::Default,
            }
        }
    }
}
```

### 10.2 分块策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        并行迭代分块策略                                   │
│                                                                          │
│  原始张量 [1024, 1024] (1M 元素)                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┐                      │   │
│  │  │ 块 0    │ 块 1    │ 块 2    │ 块 3    │  (线程 0-3)          │   │
│  │  │ 64K 元素│ 64K 元素│ 64K 元素│ 64K 元素│                      │   │
│  │  └─────────┴─────────┴─────────┴─────────┘                      │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┐                      │   │
│  │  │ 块 4    │ 块 5    │ 块 6    │ 块 7    │  (线程 4-7)          │   │
│  │  │ 64K 元素│ 64K 元素│ 64K 元素│ 64K 元素│                      │   │
│  │  └─────────┴─────────┴─────────┴─────────┘                      │   │
│  │  ...                                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  分块规则:                                                               │
│  1. 每块 ≥ 并行阈值 (默认 4K 元素)                                       │
│  2. 块数 = min(线程数, 元素数 / 阈值)                                    │
│  3. 连续内存优先分块（保证缓存友好）                                      │
│  4. 非连续数组：按逻辑索引分块（性能较差）                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 并行阈值配置

```rust
/// 并行执行配置。
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// 启用并行的最小元素数。
    pub threshold: usize,
    
    /// 每个线程处理的最小块大小。
    pub min_chunk_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            threshold: 64 * 1024,  // 64K elements
            min_chunk_size: 4 * 1024,  // 4K elements
        }
    }
}
```

---

## 11. 与其他模块的交互

### 11.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           iter 模块依赖                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         iter                                    │   │
│  │  Elements, AxisIter, Windows, Indexed, Zip, Lanes              │   │
│  └───────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                      │
│         ┌────────────────────────┼────────────────────────┐            │
│         │                        │                        │            │
│         ▼                        ▼                        ▼            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐    │
│  │   tensor    │          │   layout    │          │  broadcast  │    │
│  │ TensorBase  │          │ LayoutFlags │          │ broadcast_  │    │
│  │ TensorView  │          │ Order       │          │   shapes    │    │
│  └─────────────┘          └─────────────┘          └─────────────┘    │
│         │                        │                        │            │
│         └────────────────────────┼────────────────────────┘            │
│                                  │                                      │
│                                  ▼                                      │
│                          ┌─────────────┐                                │
│                          │  dimension  │                                │
│                          │  Ix0-Ix6    │                                │
│                          │  IxDyn      │                                │
│                          └─────────────┘                                │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                          可选依赖 (feature = "parallel")                 │
│  ════════════════════════════════════════════════════════════════════   │
│                                  │                                      │
│                                  ▼                                      │
│                          ┌─────────────┐                                │
│                          │   rayon     │                                │
│                          │  Parallel   │                                │
│                          │  Iterator   │                                │
│                          └─────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 与 tensor 模块的接口

```rust
// tensor 模块提供的迭代器入口方法
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 元素迭代器。
    pub fn iter(&self) -> Elements<'_, A, D>;
    
    /// 指定顺序的元素迭代器。
    pub fn iter_order(&self, order: Order) -> Elements<'_, A, D>;
    
    /// 索引迭代器。
    pub fn indexed_iter(&self) -> IndexedIter<'_, A, D>;
    
    /// 沿轴迭代。
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<'_, A, D::Smaller>
    where
        D: RemoveAxis;
    
    /// 滑动窗口迭代。
    pub fn windows(&self, size: impl IntoDimension<D>) -> Option<Windows<'_, A, D>>;
    
    /// 行/列迭代。
    pub fn lanes(&self, axis: Axis) -> LaneIter<'_, A, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// 可变元素迭代器。
    pub fn iter_mut(&mut self) -> ElementsMut<'_, A, D>;
    
    /// 可变索引迭代器。
    pub fn indexed_iter_mut(&mut self) -> IndexedIterMut<'_, A, D>;
    
    /// 可变沿轴迭代。
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, A, D::Smaller>
    where
        D: RemoveAxis;
}
```

### 11.3 与 layout 模块的接口

```rust
// iter 模块使用 layout 模块的信息
impl<'a, A, D> Elements<'a, A, D>
where
    D: Dimension,
{
    fn choose_path(view: &TensorView<'a, A, D>, order: Order) -> Path {
        match order {
            Order::Default if view.is_f_contiguous() => Path::FastF,
            Order::Default if view.is_c_contiguous() => Path::FastC,
            Order::F if view.is_f_contiguous() => Path::FastF,
            Order::C if view.is_c_contiguous() => Path::FastC,
            _ => Path::Slow,
        }
    }
}

enum Path {
    FastF,   // F-contiguous fast path
    FastC,   // C-contiguous fast path
    Slow,    // Non-contiguous path with stride state machine
}
```

### 11.4 与 broadcast 模块的接口

```rust
// Zip 使用 broadcast 模块的广播规则
impl<Parts, D> Zip<Parts, D>
where
    D: Dimension,
{
    fn add_broadcast<P>(&mut self, producer: P) -> Result<(), BroadcastError>
    where
        P: NdProducer,
    {
        // 使用 broadcast::broadcast_shapes 计算广播形状
        let shapes = vec![self.shape.slice(), producer.shape().slice()];
        let broadcast_shape = broadcast::broadcast_shapes(&shapes)?;
        
        // 更新步长以支持广播访问
        // ...
        
        Ok(())
    }
}
```

---

## 12. 实现任务分解

### 任务 1：Order 枚举和基础类型 (order.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `Order` 枚举（Default, F, C）
- 实现 `resolve` 方法
- 添加文档注释

**验收标准**:
- 枚举编译通过
- `resolve` 方法正确处理各种布局

---

### 任务 2：StrideState 步长状态机 (stride_state.rs)

**预计时间**: 15 分钟

**任务内容**:
- 定义 `StrideState<D>` 结构体
- 实现 `advance` 方法（推进索引）
- 处理 F-order 和 C-order 的索引更新

**验收标准**:
- 正确处理任意维度
- F-order 和 C-order 索引更新正确

---

### 任务 3：Elements 迭代器 (elements.rs)

**预计时间**: 15 分钟

**任务内容**:
- 定义 `Elements` 和 `ElementsMut` 结构体
- 实现 `Iterator` trait
- 实现快速路径（连续数组）和慢速路径（非连续数组）
- 实现 `DoubleEndedIterator`（仅连续数组）

**验收标准**:
- 连续数组使用快速路径
- 非连续数组使用慢速路径
- `size_hint` 正确

---

### 任务 4：AxisIter 轴迭代器 (axis.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `AxisIter` 和 `AxisIterMut` 结构体
- 实现 `Iterator` trait
- 实现 `DoubleEndedIterator`
- 处理 `RemoveAxis` 约束

**验收标准**:
- 正确产出降维视图
- 边界处理正确

---

### 任务 5：Windows 窗口迭代器 (windows.rs)

**预计时间**: 15 分钟

**任务内容**:
- 定义 `Windows` 和 `WindowsMut` 结构体
- 实现窗口大小验证
- 实现 `Iterator` trait
- 处理边界情况（空数组、窗口大于数组）

**验收标准**:
- 窗口大小验证正确
- 不产出不完整窗口
- 空数组处理正确

---

### 任务 6：IndexedIter 索引迭代器 (indexed.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `IndexedIter` 和 `IndexedIterMut` 结构体
- 实现 `Iterator` trait，产出 `(索引, 元素)`
- 处理不同维度类型的索引表示

**验收标准**:
- 索引正确递增
- 索引表示类型正确

---

### 任务 7：NdProducer trait (producer.rs)

**预计时间**: 15 分钟

**任务内容**:
- 定义 `NdProducer` trait
- 定义 `NdProducerMut` trait
- 为 `TensorView` 和 `TensorViewMut` 实现 trait
- 实现 `split_at` 方法

**验收标准**:
- `split_at` 返回不重叠的生产者
- 并行安全性保证

---

### 任务 8：Zip 核心实现 (zip.rs) - Part 1

**预计时间**: 15 分钟

**任务内容**:
- 定义 `Zip` 结构体
- 实现 `from` 构造方法
- 实现 `and` 组合方法
- 实现形状广播检查

**验收标准**:
- 广播检查正确
- 错误处理完善

---

### 任务 9：Zip 核心实现 (zip.rs) - Part 2

**预计时间**: 15 分钟

**任务内容**:
- 实现 `for_each` 方法
- 实现 `map_collect` 方法
- 实现 `ZipParts` trait
- 实现 `ZipIter` 迭代器

**验收标准**:
- 多张量同步迭代正确
- 闭包参数顺序正确

---

### 任务 10：LaneIter 行/列迭代器 (lanes.rs)

**预计时间**: 10 分钟

**任务内容**:
- 定义 `LaneIter` 和 `LaneIterMut` 结构体
- 实现 `Iterator` trait
- 处理高维数组的 lane 迭代

**验收标准**:
- 正确产出 1D 视图
- 高维支持正确

---

### 任务 11：并行迭代器集成 (parallel.rs)

**预计时间**: 15 分钟

**任务内容**:
- 定义 `ParElements` 结构体
- 定义 `ParZip` 结构体
- 实现 `IntoParallelIterator`
- 实现分块策略

**验收标准**:
- rayon 集成正确
- 并行阈值生效
- 分块不重叠

---

### 任务 12：模块导出和集成测试 (mod.rs)

**预计时间**: 10 分钟

**任务内容**:
- 整理公开导出
- 为 `TensorBase` 添加迭代器入口方法
- 编写集成测试

**验收标准**:
- 所有迭代器可通过 `tensor.iter()` 访问
- 集成测试覆盖主要场景

---

## 13. 设计决策记录

### 13.1 为什么默认使用物理内存布局顺序？

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **物理顺序** | 按内存地址递增遍历 | 最优缓存性能 | 逻辑顺序不确定 |
| 逻辑 F 顺序 | 总是列优先 | 顺序可预测 | C-order 数组性能差 |
| 逻辑 C 顺序 | 总是行优先 | 与 NumPy 默认一致 | F-order 数组性能差 |

**选择理由**：
1. **性能优先**：科学计算中遍历性能至关重要
2. **BLAS 兼容**：F-order 是 BLAS 的默认布局
3. **灵活性**：用户可显式指定顺序

### 13.2 为什么使用 Producer Pattern？

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Producer Pattern** | NdProducer trait + split_at | 支持并行、可组合 | 复杂度高 |
| 直接迭代器 | Iterator trait only | 简单 | 不支持并行分割 |
| 回调模式 | for_each(Fn) | 灵活 | 无法组合 |

**选择理由**：
1. **并行支持**：`split_at` 是并行迭代的基础
2. **可组合性**：多个 Producer 可组合成 Zip
3. **与 ndarray 兼容**：降低用户学习成本

### 13.3 为什么 Zip 不使用宏？

| 方案 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| **结构体链式** | `Zip::from(&a).and(&b).and(&c)` | 类型安全、可推导 | 稍冗长 |
| 宏 | `azip!((a, b, c) in &a, &b, &c => ...)` | 简洁 | 调试困难 |

**选择理由**：
1. **类型安全**：编译期检查参数数量和类型
2. **IDE 支持**：更好的自动补全和类型提示
3. **可扩展**：更容易添加新方法

### 13.4 为什么空数组立即结束？

| 方案 | 行为 |
|------|------|
| **立即结束** | `next()` 返回 `None`，产出 0 个元素 |
| 返回空迭代器 | 构造时返回 `None` |
| Panic | panic |

**选择理由**：
1. **一致性**：与 Rust 标准库行为一致
2. **安全**：不 panic，允许正常处理
3. **直观**：空数组没有元素，产出 0 个符合直觉

### 13.5 为什么窗口越界不产出不完整窗口？

| 方案 | 行为 |
|------|------|
| **不产出** | 只产出完整窗口 |
| 产出部分 | 边界窗口可能不完整 |
| Panic | panic |

**选择理由**：
1. **语义清晰**：窗口大小固定，部分窗口语义不明
2. **与 NumPy 一致**：`np.lib.stride_tricks.sliding_window_view` 行为相同
3. **安全性**：避免意外访问越界内存

---

## 附录 A：迭代器性能参考

| 迭代器 | 连续数组 | 非连续数组 | 并行支持 |
|--------|----------|-----------|---------|
| Elements | O(n), 最优 | O(n), 较慢 | ✓ |
| AxisIter | O(k) 子视图创建 | O(k) 子视图创建 | ✓ |
| Windows | O(n×w) | O(n×w) | ✓ |
| IndexedIter | O(n) + 索引计算 | O(n) + 索引计算 | ✗ |
| Zip | O(n) | O(n) | ✓ |
| LaneIter | O(n/k) 子视图 | O(n/k) 子视图 | ✓ |

*n = 元素数, k = 轴长度, w = 窗口大小*

---

## 附录 B：空数组和零维数组行为

| 迭代器 | 空数组 (任一维度为 0) | 零维数组 (Ix0) |
|--------|---------------------|----------------|
| Elements | 产出 0 个元素 | 产出 1 个元素 |
| AxisIter | 产出 0 个子视图 | N/A（无法降维） |
| Windows | 产出 0 个窗口 | 产出 1 个空窗口（若窗口大小也为空） |
| IndexedIter | 产出 0 个 (索引, 元素) | 产出 1 个 ([], 元素) |
| Zip | 产出 0 次调用 | 产出 1 次调用 |
| LaneIter | 产出 0 个 lane | N/A |

---

## 附录 C：迭代器 trait 实现速查

| 迭代器 | Iterator | ExactSizeIterator | DoubleEndedIterator | IntoParallelIterator |
|--------|:--------:|:-----------------:|:-------------------:|:--------------------:|
| Elements | ✓ | ✓ | ✓ (连续) | ✓ |
| ElementsMut | ✓ | ✓ | ✓ (连续) | ✓ |
| AxisIter | ✓ | ✓ | ✓ | ✓ |
| AxisIterMut | ✓ | ✓ | ✓ | ✓ |
| Windows | ✓ | ✓ | ✗ | ✓ |
| WindowsMut | ✓ | ✓ | ✗ | ✓ |
| IndexedIter | ✓ | ✓ | ✗ | ✗ |
| IndexedIterMut | ✓ | ✓ | ✗ | ✗ |
| LaneIter | ✓ | ✓ | ✗ | ✓ |
| Zip | ✓ | ✓ | ✗ | ✓ |

---

*文档结束*
