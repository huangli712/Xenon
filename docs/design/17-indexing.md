# 索引操作模块设计

> 文档编号: 17 | 模块: `src/index/` | 阶段: Phase 3
> 前置文档: `07-tensor.md`, `02-dimension.md`, `06-memory.md`, `26-error.md`
> 需求参考: 需求说明书 §18

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 多维整数索引 | `[i, j, k]` 形式的元素访问， | 高级索引（布尔掩码/整数数组/where 条件选择） |
| 范围索引（切片） | `s![.., 0..3, ..;2]` 宏驱动切片 | 高级索引（布尔掩码、整数数组高级索引） |
| Index trait 实现 | `core::ops::Index` for `TensorBase` | 比较运算符（在逐元素运算模块） |
| 偏移量计算 | F-order 公式：`offset = sum(index[i] * strides[i])` | 其他内存序的偏移量计算 |
| get/get_unchecked | `get()` 返回 `Option<&A>` / `get_unchecked()` unsafe 变体 | 可变迭代器（在 iter 模块） |
| 切片宏 | `s![]` 宏的语法解析和展开 | 切片宏设计在 ndarray 之外的库中 |
| 切片返回视图 | 切片返回 `ViewRepr`（零拷贝） | 勒贝操作（在逐元素运算模块） |
| 负步长支持 | 步长可为负（`HAS_NEG_STRIDE` 标志位） | 反向迭代（在 iter 模块） |
| 边界检查 | checked 版本 panic + unsafe `unchecked` 版本 | 越界 panic 消息（在 error 模块） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零拷贝视图 | 切片返回视图（`TensorView`），不复制数据 |
| 编译期类型安全 | 静态维度通过 `Dimension` trait 保证索引正确性 |
| 边界检查可选 | 提供 checked（panic）和 unchecked（unsafe）两种变体 |
| F-order 偏移量 | 偏移量按 F-order 公式计算 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
 L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: index  ← 当前模块（依赖 tensor, dimension, layout）
```

---

## 2. 文件位置

```
src/index/
├── mod.rs             # 模块入口，Index/IndexMut trait 实现、公开导出
├── multi_dim.rs       # 多维整数索引 [i, j, k]
└── slice_index.rs     # 切片索引、SliceInfo 设计、 s![] 宏
```

文件划分理由：整数索引和切片索引逻辑差异大，独立文件便于单独维护和测试。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │   tensor     │
                    │ TensorBase   │
                    └──────┬───────┘
                           │ 使用
              ┌────────────┼──────────────────┐
              │  index                       │
              │  multi_dim.rs │ slice_index.rs │
              └──┬───────────┬───────────────┘
                 │ 使用       │ 使用
          ┌──────▼───┐ ┌──────▼──────────────┐
          │ dimension │ │  memory-layout   │
          │ Dimension │ │  LayoutFlags     │
          │ Ix0~IxDyn │ │  HAS_NEG_STRIDE │
          └───────────┘ └────────────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `TensorView`, `TensorViewMut`, `.shape()`, `.strides()`, `.as_ptr()`, `.as_mut_ptr()`，参见 `07-tensor.md` §4 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `.slice()`, `.ndim()`，参见 `02-dimension.md` §3 |
| `layout` | `LayoutFlags`, `Strides<D>`, `HAS_NEG_STRIDE`, `HAS_ZERO_STRIDE`，参见 `06-memory.md` §3 |

> **注意**：索引越界使用 panic 处理（而非 XenonError 变体），参见 `26-error.md §5.3`。`get()` 方法返回 `Option<&A>` 提供可恢复的越界检查路径。

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `index/` 消费 `tensor`、`dimension`、`layout` 的 trait 和类型，不被它们依赖。

---

## 4. 公共 API 设计

### 4.0 NdIndex trait — 多维索引类型约束

`NdIndex<D>` 是多维索引类型的 sealed trait，用于 `core::ops::Index` 实现中约束合法的索引参数类型。

```rust
use crate::private::Sealed;

/// Trait for types that can be used as multi-dimensional indices into a tensor.
///
/// Sealed: cannot be implemented outside this crate.
///
/// # Implementations
///
/// | Type | Usage |
/// |------|-------|
/// | `[usize; N]` | Static N-dimensional index (e.g. `tensor[[0, 1]]`) |
/// | `&[usize]`   | Dynamic slice index (runtime length, checked at runtime) |
///
/// Note: The constraint `D: Dimension` on the implementation ensures that the
/// number of index components matches the tensor's dimensionality at runtime.
pub trait NdIndex<D: Dimension>: Sealed {
    /// Converts the index to a linear memory offset, given the tensor's strides.
    ///
    /// Returns `None` if any index component is out of bounds for its dimension.
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<isize>;

    /// Converts the index to a linear memory offset without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure every index component `i` satisfies `i < shape[k]`.
    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> isize;
}

// Static array index (e.g. tensor[[0, 1, 2]])
// Blanket Sealed impl for all array sizes — required by the generic NdIndex impl below.
impl<const N: usize> Sealed for [usize; N] {}

impl<D: Dimension, const N: usize> NdIndex<D> for [usize; N] {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<isize> {
        if self.len() != dim.ndim() { return None; }
        let shape = dim.slice();
        for (i, &idx) in self.iter().enumerate() {
            if idx >= shape[i] { return None; }
        }
        // SAFETY: bounds checked above
        Some(unsafe { self.index_unchecked(strides) })
    }

    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> isize {
        // Use signed arithmetic to handle negative strides
        let mut offset: isize = 0;
        for (&idx, &stride) in self.iter().zip(strides.iter()) {
            offset += stride * idx as isize;
        }
        offset
    }
}

// Dynamic slice index (e.g. tensor[&[0, 1, 2][..]])
impl Sealed for [usize] {}
impl<D: Dimension> NdIndex<D> for [usize] {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<isize> {
        if self.len() != dim.ndim() { return None; }
        let shape = dim.slice();
        for (i, &idx) in self.iter().enumerate() {
            if idx >= shape[i] { return None; }
        }
        Some(unsafe { self.index_unchecked(strides) })
    }

    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> isize {
        // Use signed arithmetic to handle negative strides
        let mut offset: isize = 0;
        for (&idx, &stride) in self.iter().zip(strides.iter()) {
            offset += stride * idx as isize;
        }
        offset
    }
}
```

> **设计决策：** `NdIndex` 使用 `Sealed` trait 防止外部实现，同时通过编译期常量泛型 `[usize; N]`
> 支持静态数组索引语法 `tensor[[0, 1]]`（编译期长度已知），以及 `&[usize]` 支持运行时动态索引。

### 4.1 多维整数索引

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Multi-dimensional index access (read-only, panics on out-of-bounds).
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(tensor[[0, 0]], 1);
    /// assert_eq!(tensor[[1, 2]], 6);
    /// ```
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    fn index<I>(&self, index: I) -> &A
    where
        I: NdIndex<D>;

    /// Gets a reference to the element, returns None if out of bounds.
    ///
    /// # Examples
    /// ```
    /// assert_eq!(tensor.get(&[0, 0]), Some(&1));
    /// assert_eq!(tensor.get(&[2, 0]), None);  // out of bounds
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<&A>;

    /// Gets a reference to the element without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `index[i] < shape[i]` for all dimensions.
    ///
    /// # Examples
    /// ```
    /// // SAFETY: [0, 0] is within bounds [2, 3]
    /// unsafe {
    ///     assert_eq!(*tensor.get_unchecked(&[0, 0]), 1);
    /// }
    /// ```
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;

    /// Gets a mutable reference to the element, returns None if out of bounds.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut A>
    where
        S: StorageMut<Elem = A>;

    /// Gets a mutable reference to the element without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `index[i] < shape[i]` for all dimensions.
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A
    where
        S: StorageMut<Elem = A>;
}
```

> **设计决策：** `get()`/`get_mut()` 返回 `Option<&A>` / `Option<&mut A>`，
> 越界时返回 `None` 而非 panic，便于安全地组合多重索引操作。

> `Index` trait 越界时 panic，因为 Rust 惯例要求 `Index::index` 返回引用。

> `get_unchecked()` 提供 unsafe 快速路径。

### 4.1.1 偏移量计算（F-order）

多维索引到线性偏移量的计算公式：

```rust
offset = sum(index[i] * strides[i])  for i in 0..ndim
```

F-order 中 `strides[i] = product(shape[0..i])`，因此：

```rust
offset = sum(index[i] * product(shape[0..i]))
     = index[0] * 1 + index[1] * shape[0] + index[2] * shape[0] * shape[1] + ...
```

示例（F-order `shape=[2, 3, 4]`, `strides=[1, 2, 6]`）：

```rust
index=[1, 2, 1] → offset = 1*1 + 2*2 + 1*6 = 11
```

### 4.1.2 边界检查 API 对称性

| 方法 | 返回类型 | 边界检查 | 越界行为 |
|------|----------|----------|----------|
| `[[i, j]]` | `&A` | panic | panic |
| `get(&[i, j])` | `Option<&A>` | 返回 None | 返回 None |
| `get_unchecked(&[i, j])` | `&A` | 无 (unsafe) | UB |
| `[[i, j]] = v` (mut) | `&mut A` | panic | panic |
| `get_mut(&[i, j])` | `Option<&mut A>` | 返回 None | 返回 None |
| `get_unchecked_mut(&[i, j])` | `&mut A` | 无 (unsafe) | UB |

### 4.2 范围索引（切片）

#### 4.2.1 SliceInfo 设计

```rust
/// Slice element type, describing a single axis slice.
///
/// **Note:** `step = 0` is illegal and will cause a panic with the message
/// "slice step cannot be zero". This is checked at runtime in the `slice()`
/// method (see Panics section).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SliceInfoElem {
    /// Single index. This axis will be removed, reducing the result dimension by 1.
    Index(isize),

    /// Range slice. This axis is preserved with the length of the range.
    Range {
        /// Start index (inclusive), None means 0
        start: Option<isize>,
        /// End index (exclusive), None means axis length
        end: Option<isize>,
        /// Step size, None means 1. Must not be 0 (panics if zero).
        step: Option<isize>,
    },
}

/// Slice information, containing slice descriptions for all axes.
#[derive(Debug, Clone)]
pub struct SliceInfo<I, D>
where
    I: Dimension,
    D: Dimension,
{
    /// Slice descriptions for each axis.
    indices: Vec<SliceInfoElem>,

    _out_dim: PhantomData<I>,
    _in_dim: PhantomData<D>,
}
```

> **设计决策：** `SliceInfoElem` 使用 `isize` 表示步长和起始/结束索引，
> 支持负步长（步长 < 0 表示反向遍历）。`step` 为 `Option<isize>` 而非 `usize`。
> 这是负步长的关键：`step = -1` 产生反转视图。

#### 4.2.2 s![] 切片宏设计

```rust
/// Slice macro for creating type-safe slice descriptions.
///
/// # Syntax
///
/// | Syntax | Meaning | SliceInfoElem variant |
/// |--------|---------|----------------------|
/// | `..` | Full range | `Range { start: None, end: None, step: None }` |
/// | `a..b` | Range [a, b) | `Range { start: Some(a), end: Some(b), step: None }` |
/// | `a..b;c` | Range with step | `Range { start: Some(a), end: Some(b), step: Some(c) }` |
/// | `..;c` | Full range with step | `Range { start: None, end: None, step: Some(c) }` |
///
/// # Examples
/// ```
/// let view = tensor.slice(s![1..3, ..;2, ..]);
/// assert_eq!(view.shape(), &[2, 3, 4]);  // step=2 on axis 1
/// ```
#[macro_export]
macro_rules! s {
    ($($elem:tt),* $(,)?) => {
        // macro expansion logic
    };
}
```

> **设计决策：** `s![]` 宏设计参考 ndarray 的 `s![]` 宏，
> 提供 Rust 像的声明式宏语法，编译期类型安全。
> 相比过程式构造（`SliceInfo::new(vec![...]).unwrap()`），宏在编译期展开为正确的 `SliceInfoElem` 枚举值。

#### 4.2.3 切片方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Creates a sliced view (zero-copy).
    ///
    /// # Arguments
    /// * `info` - Slice information created by the `s![]` macro
    ///
    /// # Returns
    /// The sliced `TensorView`, zero-copy.
    ///
    /// # Panics
    /// Panics if slice dimensions don't match tensor dimensions, or if index is out of bounds.
    /// Also panics if any slice step is zero: "slice step cannot be zero".
    ///
    /// # Examples
    /// ```
    /// let view = tensor.slice(s![1..3, 2..5, ..]);
    /// assert_eq!(view.shape(), &[2, 3, 6]);
    /// ```
    pub fn slice<I>(&self, info: SliceInfo<I, D>) -> TensorView<'_, A, I>
    where
        I: Dimension,
    {
        // ...
    }

    /// Checked sliced view creation.
    pub fn try_slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension,
    {
        // ...
    }

    /// Creates a mutable sliced view.
    ///
    /// # Panics
    ///
    /// Panics if the tensor contains zero strides (`LayoutFlags::HAS_ZERO_STRIDE`),
    /// because zero strides mean multiple logical indices map to the same physical
    /// address, and mutable access would cause data races.
    pub fn slice_mut<I>(&mut self, info: SliceInfo<I, D>) -> TensorViewMut<'_, A, I>
    where
        I: Dimension,
        S: StorageMut<Elem = A>,
    {
        // ...
    }

    /// Checked mutable sliced view creation.
    pub fn try_slice_mut<I>(&mut self, info: SliceInfo<I, D>) -> Result<TensorViewMut<'_, A, I>, XenonError>
    where
        I: Dimension,
        S: StorageMut<Elem = A>,
    {
        // ...
    }
}
```

> **注意**：对广播视图（含零步长）不得调用 `slice_mut`。如果原张量包含零步长（`LayoutFlags::HAS_ZERO_STRIDE`），`slice_mut()` 将 panic，提示信息为 "mutable slicing is not allowed on broadcast views (tensors with zero strides)"。这是因为零步长意味着多个逻辑索引映射到同一物理地址，可变访问会引起数据竞争。此为 panic 而非错误返回，因为这是一个编程错误。

#### 4.2.4 Good / Bad 对比

```rust
// Good - use get() to safely handle potentially out-of-bounds indices
fn safe_index(tensor: &Tensor<f64, Ix2>, idx: &[usize]) -> Option<&f64> {
    tensor.get(idx)
}

let tensor = Tensor2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Out-of-bounds returns None instead of panicking
let result = safe_index(&tensor, &[5, 0]);
assert!(result.is_none());

// Safe access with a default fallback
let val = *safe_index(&tensor, &[0, 0]).unwrap_or(&0.0);
assert_eq!(val, 1.0);
```

```rust
// Bad - direct [] access panics on out-of-bounds
fn unsafe_index(tensor: &Tensor<f64, Ix2>, idx: &[usize]) -> f64 {
    tensor[idx]  // panic if out of bounds!
}
```

#### 4.2.5 切片后元数据更新

切片操作需要更新视图的元数据：

| 元数据 | 更新规则 |
|--------|----------|
| `shape[i]` | 切片范围长度 |
| `strides[i]` | 原步长 × 切片步长 |
| `offset` | 原偏移 + start × 原步长（内部用 `isize` 计算，再与基指针组合） |
| `LayoutFlags` | 有负步长时设置 `HAS_NEG_STRIDE`；步长恰好为 0 时设置 `HAS_ZERO_STRIDE`（广播视图的广播维度） |

---

## 5. 内部实现设计

### 5.1 偏移量计算（F-order）

```rust
function compute_offset_f(shape: [usize; N], strides: [isize; N], index: [usize; N]) -> isize:
    offset = 0
    for i in 0..N:
        offset += index[i] * strides[i]
    return offset
```

### 5.2 切片步长/形状/偏移量计算

```rust
// normalize_index(idx, dim_len): convert possibly-negative point index to valid usize
// Panics if result is out of bounds [0, dim_len)
fn normalize_index(idx: isize, dim_len: usize) -> usize {
    if idx < 0 {
        let normalized = dim_len as isize + idx;
        assert!(normalized >= 0, "index out of bounds");
        normalized as usize
    } else {
        let normalized = idx as usize;
        assert!(normalized < dim_len, "index out of bounds");
        normalized
    }
}

// normalize_bound(idx, dim_len): convert slice bound to [0, dim_len]
// Negative values are interpreted Python-style; positive end bound may equal dim_len.
fn normalize_bound(idx: isize, dim_len: usize) -> usize {
    let normalized = if idx < 0 {
        dim_len as isize + idx
    } else {
        idx
    };
    assert!((0..=dim_len as isize).contains(&normalized), "slice bound out of bounds");
    normalized as usize
}

function compute_slice(shape, strides, offset, slices: [SliceInfoElem; N])
    -> (new_shape, new_strides, new_offset, new_layout):
    new_shape = []
    new_strides = []
    new_offset = offset
    new_layout = layout

    for i in 0..N:
        match slices[i]:
            Index(idx):
                // Single index: reduce dimension
                // Normalize negative index (Python-style: -1 means last element)
                let normalized_idx = normalize_index(idx, shape[i])
                new_offset += normalized_idx * strides[i]
                // shape and strides do not include this dimension
            Range { start, end, step }:
                // Range slice
                st = step.unwrap_or(1)
                assert!(st != 0, "slice step must not be zero")

                // Default start/end depend on step direction:
                // - Positive step: start=0, end=shape[i]
                // - Negative step: start=shape[i]-1, end=-1 (exclusive sentinel before first element)
                s = start.unwrap_or(if st > 0 { 0 } else { shape[i] as isize - 1 })
                e = end.unwrap_or(if st > 0 { shape[i] as isize } else { -1 })

                // Handle negative values. For negative-step full slices, the default
                // end = -1 is an exclusive sentinel and must not be normalized into
                // a concrete in-bounds index.
                s = normalize_index(s, shape[i])
                e = if st > 0 {
                    normalize_bound(e, shape[i])
                } else if end.is_some() {
                    normalize_bound(e, shape[i])
                } else {
                    e
                }

                dim_len = if st > 0:
                    (e - s + st - 1) / st
                else:
                    (s - e + (-st) - 1) / (-st)
                new_shape.push(dim_len)
                new_strides.push(strides[i] * st)

                // Offset is always computed from the logical first element.
                // Negative traversal direction is already encoded in signed strides.
                new_offset += s * strides[i]

                // Update flags
                if st < 0:
                    new_layout.set_has_neg_stride(true)
    return (new_shape, new_strides, new_offset, new_layout)
```

### 5.3 安全性论证

- `get_unchecked`: 调用者保证索引在合法范围内；偏移量计算与 checked 版本相同
- `slice_unchecked`: 调用者保证切片范围在合法范围内
- 负步长: `HAS_NEG_STRIDE` 标志正确标记，迭代器根据标志处理反向遍历

- 非连续切片: `HAS_ZERO_STRIDE` 标志标记步长为0的维度（仅出现在广播场景中）

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/index/mod.rs` 骨架
  - 文件: `src/index/mod.rs`
  - 内容: 模块声明、子模块文件占位、公开导出声明
  - 测试: 编译通过
  - 前置: `07-tensor.md` 完成
  - 预计: 5 min

- [ ] **T2**: 定义 `SliceInfoElem` 和 `SliceInfo` 类型
  - 文件: `src/index/slice_index.rs`
  - 内容: 枚举定义、`SliceInfo` 结构体、基本方法
  - 测试: `test_slice_info_elem_basic`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 多维整数索引

- [ ] **T3**: 实现 `Index` trait for `TensorBase`
  - 文件: `src/index/multi_dim.rs`
  - 内容: `index()`/`get()`/`get_unchecked()` 实现
  - 测试: `test_index_2d`, `test_index_3d`, `test_index_out_of_bounds`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `IndexMut` 和 `get_mut`/`get_unchecked_mut`
  - 文件: `src/index/multi_dim.rs`
  - 内容: 可变索引方法实现
  - 测试: `test_index_mut_2d`, `test_get_mut_out_of_bounds`
  - 前置: T3
  - 预计: 10 min

### Wave 3: 切片索引

- [ ] **T5**: 实现 `slice()` 和 `slice_mut()` 方法
  - 文件: `src/index/slice_index.rs`
  - 内容: 切片视图创建、元数据计算
  - 测试: `test_slice_basic`, `test_slice_with_step`, `test_slice_empty`
  - 前置: T2, T3
  - 预计: 15 min

- [ ] **T6**: 实现 `s![]` 宏
  - 文件: `src/index/slice_index.rs`
  - 内容: 宏定义、语法解析
  - 测试: `test_slice_macro_basic`, `test_slice_macro_step`
  - 前置: T5
  - 预计: 15 min

### Wave 4: 集成与测试

- [ ] **T7**: 编写综合测试
  - 文件: `tests/index.rs`
  - 内容: 链式切片、视图的视图、负步长验证
  - 测试: 覆盖所有公共 API
  - 前置: T1-T6
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
                      │
Wave 3:      [T5] → [T6]
                      │
Wave 4:           [T7]
```

---

## 7. 测试计划

### 7.0 测试分类表

| 测试分类 | 位置 | 说明 |
|----------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证整数索引、切片和 `s![]` 宏的核心语义 |
| 集成测试 | `tests/` | 验证 `index` 与 `tensor`、`layout`、`iter`、`shape` 的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖零维、空数组、负步长和链式切片等边界 |

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_index_2d` | 2D 张量 `[[0, 0]]`, `[[1, 2]]` | 高 |
| `test_index_3d` | 3D 张量 `[[0, 1, 2]]` | 高 |
| `test_index_out_of_bounds` | 越界 panic 验证 | 高 |
| `test_get_returns_none` | `get()` 越界返回 None | 高 |
| `test_get_unchecked` | unsafe 路径正确读取 | 中 |
| `test_index_mut_2d` | 可变索引写入 | 高 |
| `test_get_mut_out_of_bounds` | `get_mut()` 越界返回 None | 高 |
| `test_slice_basic` | `s![1..3, ..]` 基本切片 | 高 |
| `test_slice_with_step` | `s![..;2, ..]` 步长切片 | 高 |
| `test_slice_negative_step` | `s![..;-1]` 负步长切片 | 高 |
| `test_slice_empty_range` | 空范围切片 | 中 |
| `test_slice_chain` | 切片的切片（视图的视图） | 高 |
| `test_slice_macro_basic` | `s![]` 宏基本使用 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 0 维张量索引 | `tensor[[]]` 返回唯一元素 |
| 空数组索引 | panic（无有效索引） |
| 大张量索引 | 正确偏移量计算 |
| 非连续切片后索引 | 正确处理步长跳转 |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| 切片后 `view.len() == expected_len` | 随机切片参数 |
| `view.shape()` 正确 | 每个维度验证 |
| 视图的视图链式后数据一致 | 链式切片对比原数据 |

### 7.4 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/index.rs` | 整数索引 / 切片 / `s![]` 宏 与 `tensor`、`layout`、`iter`、`shape` 的端到端协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
|------|----------|-----------|------|
| `index → tensor` | `tensor` | `as_ptr()` / `as_mut_ptr()` | 通过张量基础指针入口完成元素访问，参见 `07-tensor.md` §4 |
| `index → storage` | `storage` | `Storage` / `StorageMut` | 通过存储抽象读取或写入元素，参见 `05-storage.md` §3 |
| `index → dimension` | `dimension` | `Dimension::slice()` | 使用维度切片能力推导结果形状，参见 `02-dimension.md` §3 |
| `index → layout` | `layout` | `LayoutFlags` | 切片后更新布局标志与步长语义，参见 `06-memory.md` §3 |
| `index ← iter` | `iter` | `get_unchecked()` | 迭代器路径复用快速索引入口，参见 `10-iterator.md` §4 |
| `index ← shape` | `shape` | reshape 后视图语义 | reshape 等上游操作产生的视图继续复用索引路径，参见 `16-shape.md` §4 |

### 8.2 数据流描述

```text
用户调用 tensor[[...]] / get() / slice() / s![]
    │
    ├── indexing 模块先规范化索引或切片边界
    ├── 再结合 tensor 的 shape / strides / offset 计算逻辑位置
    ├── layout 更新切片后 flags，必要时生成新的 view 元数据
    └── 结果继续被 iter / shape / math 等上层路径消费
```

---

## 9. 设计决策记录

### 决策 1: 切片宏 s![] 设计

| 属性 | 值 |
|------|-----|
| 决策 | 使用声明式宏（`macro_rules!`）实现 `s![]` 切片宏 |
| 理由 | 编译期类型安全；语法简洁类似 ndarray；展开为 `SliceInfoElem` 枚举值 |
| 替代方案 | 过程式构造 — 放弃，运行时解析复杂、类型不安全 |
| 替代方案 | `macro` 过程宏 — 放弃，语法不自然 |

### 决策 2: 负步长支持

| 属性 | 值 |
|------|-----|
| 决策 | 切片步长支持负值，`step` 为 `isize` 类型 |
| 理由 | 支持反转视图（如 `flip` 操作）；与 NumPy 兼容；步长为负时产生反向遍历 |
| 替代方案 | 仅支持正步长 — 放弃，无法实现反转操作 |

---

## 10. 性能考量

### 10.1 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 多维索引 `[[i, j, k]]` | O(ndim) | O(1) |
| `get()` | O(ndim) | O(1) |
| `get_unchecked()` | O(ndim) | O(1) |
| 切片 `slice()` | O(ndim) | O(ndim)（新视图元数据） |
| 视图创建 | O(1) | O(ndim) |

### 10.2 索引计算开销

| 场景 | 每元素开销 |
|------|----------|
| F-contiguous 连续数组 | 1 次指针加法（步长=1时退化为指针递增） |
| 非连续数组（切片后） | 多次乘加运算 |
| 非连续 + 广播 | 乘加 + 跳步长处理 |

### 10.3 缓存行为

| 场景 | 缓存友好性 | 说明 |
|------|-----------|------|
| 连续数组顺序索引 | 最优 | 顺序访问 |
| 步长切片 | 较差 | 跳跃访问 |
| 负步长切片 | 最差 | 反向访问 |

### 10.4 性能数据（参考）

| 操作 | 连续数组 | 非连续数组（步长=2） | 性能比 |
|------|----------|---------------------|--------|
| 随机索引 1M 次 (2D) | ~2ms | ~3ms | 1.5x |
| 随机索引 1M 次 (4D) | ~4ms | ~6ms | 1.5x |
| 切片视图创建 | ~10ns | ~10ns | 1x（纯元数据） |
| 切片后遍历 1M 元素 | ~1ms | ~3ms | 3x |
| 缓存命中率（顺序 2D） | ~95% | ~60% | — |
| 缓存命中率（随机 4D） | ~70% | ~30% | — |

---

## 11. no_std 兼容性

索引操作模块在 `no_std` 环境下可用。整数索引和切片视图创建均为纯元数据操作，不涉及堆分配。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| 多维整数索引 `[[i, j, k]]` | ✅ | 纯指针偏移计算，无堆分配 |
| `get()` / `get_unchecked()` | ✅ | 纯指针偏移计算，无堆分配 |
| `get_mut()` / `get_unchecked_mut()` | ✅ | 纯指针偏移计算，无堆分配 |
| `slice()` / `slice_mut()` | ✅ | 创建 `TensorView`（零拷贝），无堆分配，参见 `07-tensor.md` §4 |
| `s![]` 宏 | ✅ | 编译期展开为 `SliceInfoElem` 枚举值，无堆分配 |
| `SliceInfo` | ✅ | 内部 `Vec` 需 `no_std + alloc`（动态维度场景），参见 `02-dimension.md` §3 |
| `SliceInfoElem` | ✅ | 枚举类型，栈分配，无堆依赖 |
| 负步长支持 | ✅ | `HAS_NEG_STRIDE` 标志位操作，无堆依赖 |

条件编译处理：

```rust
// Integer indexing: pure pointer arithmetic — works in pure no_std
// Slice views: zero-copy metadata creation — works in pure no_std
// SliceInfo: uses Vec internally — needs alloc for dynamic dimensions

#[cfg(not(feature = "std"))]
extern crate alloc;
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-07 |
| 1.0.3 | 2026-04-07 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.0.6 | 2026-04-10 |
| 1.1.0 | 2026-04-10 |
| 1.1.1 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
