# 维度系统模块设计

> 本文档定义 Renon 的维度系统，包括静态维度类型 Ix0~Ix6、动态维度类型 IxDyn，以及统一的 Dimension trait。

---

## 1. 模块定位

维度系统是 Renon 类型架构的核心支柱之一。每个张量的维度信息由类型参数 `D: Dimension` 承载，与存储参数 `S` 正交组合形成 `TensorBase<S, D>` 的双参数泛型体系。

设计目标：

| 目标 | 实现方式 |
|------|----------|
| 零开销静态维度 | 固定大小数组 `[usize; N]`，单态化后无堆分配 |
| 灵活动态维度 | `IxDyn` 使用 `Vec<usize>` 支持运行时维度数 |
| 统一接口 | `Dimension` trait 抽象所有维度操作 |
| 安全转换 | 静态→动态 always succeeds，动态→静态 returns Result |
| F-order 默认 | `default_strides()` 返回列优先步长 |

---

## 2. 文件位置

```
src/
├── dimension.rs       # 所有维度类型和 Dimension trait
```

单文件设计：维度类型之间高度相关，拆分反而增加耦合复杂度。

---

## 3. 依赖关系

```
dimension.rs
  ├── (无外部 crate 依赖，纯 std/core)
  ├── 依赖 core::ops::{Index, IndexMut}
  └── 被 tensor.rs, construction.rs, broadcast.rs, indexing.rs, shape/ 等使用
```

维度模块是整个依赖图的叶子节点之一（与 `error.rs` 并列），不依赖任何其他 Renon 模块。

---

## 4. 公共 API 设计

### 4.1 Dimension Trait

```rust
/// Core trait for tensor dimension representations.
///
/// All dimension types implement this trait, providing a uniform interface
/// for shape queries, stride computation, and dimension conversion.
///
/// # Type Parameters
/// (none — trait is object-safe for generic use)
pub trait Dimension: Clone + Eq + Debug + Send + Sync + 'static {
    /// The maximum number of dimensions this type can represent.
    /// For static types (Ix0..Ix6), this is the fixed NDIM.
    /// For IxDyn, this is [`usize::MAX`].
    const MAX_NDIM: usize;

    /// Return the number of dimensions (rank) of this dimension descriptor.
    ///
    /// # Examples
    /// ```ignore
    /// assert_eq!(Ix2(3, 4).ndim(), 2);
    /// assert_eq!(IxDyn(&[2, 3, 4]).ndim(), 3);
    /// ```
    fn ndim(&self) -> usize;

    /// Return the shape as a slice of dimension sizes.
    fn slice(&self) -> &[usize];

    /// Return the shape as a mutable slice of dimension sizes.
    ///
    /// # Safety
    /// Caller must ensure the modified shape remains consistent
    /// with the tensor's data length.
    fn slice_mut(&mut self) -> &mut [usize];

    /// Compute the total number of elements described by this shape.
    ///
    /// Returns the product of all dimension sizes. For Ix0 (0-dim), returns 1.
    ///
    /// # Panics
    /// Panics on overflow in debug builds.
    fn size(&self) -> usize {
        self.slice().iter().product()
    }

    /// Compute the total number of elements, returning `None` on overflow.
    fn size_checked(&self) -> Option<usize> {
        self.slice().iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }

    /// Compute default strides for column-major (F-order) layout.
    ///
    /// For a shape `[m, n, k]`, F-order strides are `[1, m, m*n]`.
    ///
    /// # Panics
    /// Panics if `size_checked()` overflows.
    fn default_strides(&self) -> Self {
        let ndim = self.ndim();
        if ndim == 0 {
            return Self::zeros(0);
        }
        let mut strides = Vec::with_capacity(ndim);
        strides.push(1usize);
        let shape = self.slice();
        for i in 1..ndim {
            strides.push(strides[i - 1].checked_mul(shape[i - 1]).expect("stride overflow"));
        }
        Self::from_stride_vec(strides)
    }

    /// Compute default strides for the specified memory order.
    ///
    /// # Arguments
    /// * `order` — `MemoryOrder::F` for column-major, `MemoryOrder::C` for row-major
    fn default_strides_order(&self, order: MemoryOrder) -> Self {
        match order {
            MemoryOrder::F => self.default_strides(),
            MemoryOrder::C => {
                let ndim = self.ndim();
                if ndim == 0 {
                    return Self::zeros(0);
                }
                let mut strides = vec![0usize; ndim];
                strides[ndim - 1] = 1;
                let shape = self.slice();
                for i in (0..ndim - 1).rev() {
                    strides[i] = strides[i + 1].checked_mul(shape[i + 1]).expect("stride overflow");
                }
                Self::from_stride_vec(strides)
            }
        }
    }

    /// Create a zero-initialized dimension of the given rank.
    ///
    /// For static types, `ndim` must equal the type's fixed dimension count.
    /// For IxDyn, any `ndim` is valid.
    fn zeros(ndim: usize) -> Self;

    /// Construct from a stride vector (internal helper).
    ///
    /// Used by `default_strides` to create a dimension of matching type
    /// from computed stride values.
    fn from_stride_vec(strides: Vec<usize>) -> Self;

    /// Convert this dimension into another dimension type.
    ///
    /// Static→Dynamic always succeeds.
    /// Dynamic→Static succeeds only if the dimension counts match.
    fn into_dimension<D: Dimension>(self) -> Result<D, DimensionMismatch>
    where
        Self: Sized,
    {
        D::try_from_dimension(&self)
    }

    /// Attempt to create this dimension type from another dimension.
    ///
    /// Returns `Err(DimensionMismatch)` if the source dimension count
    /// does not match the target type's expected count.
    fn try_from_dimension<D: Dimension>(d: &D) -> Result<Self, DimensionMismatch>
    where
        Self: Sized;

    /// Create from an index pattern (for broadcast/compute use).
    ///
    /// Panics if `indices.len()` does not match expected ndim for static types.
    fn from_indices(indices: &[usize]) -> Self;

    /// Whether this is a dynamically-sized dimension type.
    const IS_DYNAMIC: bool;
}
```

### 4.2 MemoryOrder Enum

```rust
/// Memory layout order for multi-dimensional arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryOrder {
    /// Column-major order (Fortran-style). Default for Renon.
    F,
    /// Row-major order (C-style).
    C,
}
```

### 4.3 Static Dimension Types

#### Ix0 — 零维（标量容器）

```rust
/// Zero-dimensional index type (scalar container).
///
/// Represents a shape with zero axes. A tensor with `Ix0` dimension
/// contains exactly one element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix0;

impl Ix0 {
    /// Construct a zero-dimensional index.
    #[inline]
    pub const fn new() -> Self {
        Ix0
    }
}

// Index access (empty — no dimensions to index)
impl Index<usize> for Ix0 {
    type Output = usize;
    fn index(&self, _index: usize) -> &usize {
        panic!("Ix0 has no dimensions to index")
    }
}

impl IndexMut<usize> for Ix0 {
    fn index_mut(&mut self, _index: usize) -> &mut usize {
        panic!("Ix0 has no dimensions to index")
    }
}
```

#### Ix1 — 一维（向量）

```rust
/// One-dimensional index type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix1(pub usize);

impl Ix1 {
    #[inline]
    pub const fn new(i0: usize) -> Self {
        Ix1(i0)
    }
}

impl Index<usize> for Ix1 {
    type Output = usize;
    fn index(&self, index: usize) -> &usize {
        assert!(index == 0, "Ix1 index out of bounds: {index}");
        &self.0
    }
}

impl IndexMut<usize> for Ix1 {
    fn index_mut(&mut self, index: usize) -> &mut usize {
        assert!(index == 0, "Ix1 index out of bounds: {index}");
        &mut self.0
    }
}
```

#### Ix2 — 二维（矩阵）

```rust
/// Two-dimensional index type (matrix shape).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix2(pub usize, pub usize);

impl Ix2 {
    #[inline]
    pub const fn new(i0: usize, i1: usize) -> Self {
        Ix2(i0, i1)
    }
}

impl Index<usize> for Ix2 {
    type Output = usize;
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("Ix2 index out of bounds: {index}"),
        }
    }
}

impl IndexMut<usize> for Ix2 {
    fn index_mut(&mut self, index: usize) -> &mut usize {
        match index {
            0 => &mut self.0,
            1 => &mut self.1,
            _ => panic!("Ix2 index out of bounds: {index}"),
        }
    }
}
```

#### Ix3 ~ Ix6 — 三维至六维

采用相同的模式，每个类型包含 N 个 `usize` 字段：

```rust
/// Three-dimensional index type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix3(pub usize, pub usize, pub usize);

/// Four-dimensional index type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix4(pub usize, pub usize, pub usize, pub usize);

/// Five-dimensional index type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix5(pub usize, pub usize, pub usize, pub usize, pub usize);

/// Six-dimensional index type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ix6(pub usize, pub usize, pub usize, pub usize, pub usize, pub usize);
```

每个类型都实现 `Index<usize>` 和 `IndexMut<usize>`，通过 match 分派到对应字段。

### 4.4 Dynamic Dimension Type

```rust
/// Dynamic dimension type with heap-allocated shape.
///
/// Use `IxDyn` when the number of dimensions is not known at compile time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IxDyn {
    /// Heap-allocated dimension sizes.
    dims: Vec<usize>,
}

impl IxDyn {
    /// Create a new dynamic dimension from a slice.
    #[inline]
    pub fn new(dims: &[usize]) -> Self {
        IxDyn { dims: dims.to_vec() }
    }

    /// Create a zero-initialized dynamic dimension of the given rank.
    #[inline]
    pub fn zeros(ndim: usize) -> Self {
        IxDyn { dims: vec![0; ndim] }
    }

    /// Return the underlying Vec.
    #[inline]
    pub fn into_raw(self) -> Vec<usize> {
        self.dims
    }
}

impl Index<usize> for IxDyn {
    type Output = usize;
    fn index(&self, index: usize) -> &usize {
        &self.dims[index]
    }
}

impl IndexMut<usize> for IxDyn {
    fn index_mut(&mut self, index: usize) -> &mut usize {
        &mut self.dims[index]
    }
}

impl From<Vec<usize>> for IxDyn {
    fn from(dims: Vec<usize>) -> Self {
        IxDyn { dims }
    }
}

impl From<&[usize]> for IxDyn {
    fn from(dims: &[usize]) -> Self {
        IxDyn { dims: dims.to_vec() }
    }
}
```

### 4.5 Dimension Trait Implementations

#### Ix0 实现

```rust
impl Dimension for Ix0 {
    const MAX_NDIM: usize = 0;
    const IS_DYNAMIC: bool = false;

    #[inline]
    fn ndim(&self) -> usize { 0 }

    #[inline]
    fn slice(&self) -> &[usize] { &[] }

    fn slice_mut(&mut self) -> &mut [usize] {
        // Ix0 has no mutable dimensions — return empty slice
        // Uses a static empty mut slice via UnsafeCell pattern
        static mut EMPTY: [usize; 0] = [];
        // Safety: empty slice is always safe to borrow
        unsafe { &mut EMPTY }
    }

    #[inline]
    fn size(&self) -> usize { 1 }

    #[inline]
    fn size_checked(&self) -> Option<usize> { Some(1) }

    #[inline]
    fn zeros(_ndim: usize) -> Self {
        assert!(_ndim == 0, "Ix0 requires ndim == 0");
        Ix0
    }

    fn from_stride_vec(strides: Vec<usize>) -> Self {
        assert!(strides.is_empty(), "Ix0 requires empty strides");
        Ix0
    }

    fn try_from_dimension<D: Dimension>(d: &D) -> Result<Self, DimensionMismatch> {
        if d.ndim() == 0 {
            Ok(Ix0)
        } else {
            Err(DimensionMismatch {
                expected: 0,
                actual: d.ndim(),
            })
        }
    }

    fn from_indices(indices: &[usize]) -> Self {
        assert!(indices.is_empty(), "Ix0 requires empty indices");
        Ix0
    }

    fn default_strides(&self) -> Self { Ix0 }
    fn default_strides_order(&self, _order: MemoryOrder) -> Self { Ix0 }
}
```

#### Ix1 实现

```rust
impl Dimension for Ix1 {
    const MAX_NDIM: usize = 1;
    const IS_DYNAMIC: bool = false;

    #[inline]
    fn ndim(&self) -> usize { 1 }

    #[inline]
    fn slice(&self) -> &[usize] {
        // Safety: self is a transparent wrapper around usize
        unsafe { core::slice::from_raw_parts(&self.0 as *const usize, 1) }
    }

    fn slice_mut(&mut self) -> &mut [usize] {
        unsafe { core::slice::from_raw_parts_mut(&mut self.0 as *mut usize, 1) }
    }

    #[inline]
    fn zeros(ndim: usize) -> Self {
        assert!(ndim == 1, "Ix1 requires ndim == 1");
        Ix1(0)
    }

    fn from_stride_vec(strides: Vec<usize>) -> Self {
        assert!(strides.len() == 1, "Ix1 requires exactly 1 stride");
        Ix1(strides[0])
    }

    fn try_from_dimension<D: Dimension>(d: &D) -> Result<Self, DimensionMismatch> {
        if d.ndim() == 1 {
            Ok(Ix1(d.slice()[0]))
        } else {
            Err(DimensionMismatch { expected: 1, actual: d.ndim() })
        }
    }

    fn from_indices(indices: &[usize]) -> Self {
        assert!(indices.len() == 1, "Ix1 requires exactly 1 index");
        Ix1(indices[0])
    }
}
```

#### Ix2 实现

```rust
impl Dimension for Ix2 {
    const MAX_NDIM: usize = 2;
    const IS_DYNAMIC: bool = false;

    #[inline]
    fn ndim(&self) -> usize { 2 }

    fn slice(&self) -> &[usize] {
        // Safety: Ix2 is #[repr(C)] with two usize fields
        unsafe { core::slice::from_raw_parts(&self.0 as *const usize, 2) }
    }

    fn slice_mut(&mut self) -> &mut [usize] {
        unsafe { core::slice::from_raw_parts_mut(&mut self.0 as *mut usize, 2) }
    }

    fn zeros(ndim: usize) -> Self {
        assert!(ndim == 2, "Ix2 requires ndim == 2");
        Ix2(0, 0)
    }

    fn from_stride_vec(strides: Vec<usize>) -> Self {
        assert!(strides.len() == 2, "Ix2 requires exactly 2 strides");
        Ix2(strides[0], strides[1])
    }

    fn try_from_dimension<D: Dimension>(d: &D) -> Result<Self, DimensionMismatch> {
        if d.ndim() == 2 {
            let s = d.slice();
            Ok(Ix2(s[0], s[1]))
        } else {
            Err(DimensionMismatch { expected: 2, actual: d.ndim() })
        }
    }

    fn from_indices(indices: &[usize]) -> Self {
        assert!(indices.len() == 2, "Ix2 requires exactly 2 indices");
        Ix2(indices[0], indices[1])
    }
}
```

Ix3~Ix6 遵循完全相同的模式，`MAX_NDIM` 和断言中的维度数相应增加。

> **注意**: Ix2~Ix6 需要添加 `#[repr(C)]` 以确保字段内存布局与 `usize` 数组一致，从而使 `slice()` 的 unsafe 代码安全。

#### IxDyn 实现

```rust
impl Dimension for IxDyn {
    const MAX_NDIM: usize = usize::MAX;
    const IS_DYNAMIC: bool = true;

    #[inline]
    fn ndim(&self) -> usize { self.dims.len() }

    #[inline]
    fn slice(&self) -> &[usize] { &self.dims }

    #[inline]
    fn slice_mut(&mut self) -> &mut [usize] { &mut self.dims }

    #[inline]
    fn zeros(ndim: usize) -> Self {
        IxDyn { dims: vec![0; ndim] }
    }

    fn from_stride_vec(strides: Vec<usize>) -> Self {
        IxDyn { dims: strides }
    }

    fn try_from_dimension<D: Dimension>(d: &D) -> Result<Self, DimensionMismatch> {
        // Dynamic ← any dimension type always succeeds
        Ok(IxDyn::new(d.slice()))
    }

    fn from_indices(indices: &[usize]) -> Self {
        IxDyn::new(indices)
    }
}
```

### 4.6 Conversion Impls — From/Into

```rust
// Static → Dynamic: always succeeds (From impl)
impl From<Ix0> for IxDyn {
    fn from(_: Ix0) -> Self { IxDyn { dims: vec![] } }
}
impl From<Ix1> for IxDyn {
    fn from(d: Ix1) -> Self { IxDyn { dims: vec![d.0] } }
}
impl From<Ix2> for IxDyn {
    fn from(d: Ix2) -> Self { IxDyn { dims: vec![d.0, d.1] } }
}
impl From<Ix3> for IxDyn {
    fn from(d: Ix3) -> Self { IxDyn { dims: vec![d.0, d.1, d.2] } }
}
// Ix4, Ix5, Ix6 follow the same pattern

// Dynamic → Static: TryFrom (fallible)
impl TryFrom<IxDyn> for Ix0 {
    type Error = DimensionMismatch;
    fn try_from(d: IxDyn) -> Result<Self, Self::Error> {
        if d.ndim() == 0 { Ok(Ix0) } else { Err(DimensionMismatch { expected: 0, actual: d.ndim() }) }
    }
}
impl TryFrom<IxDyn> for Ix1 {
    type Error = DimensionMismatch;
    fn try_from(d: IxDyn) -> Result<Self, Self::Error> {
        if d.ndim() == 1 { Ok(Ix1(d.dims[0])) } else { Err(DimensionMismatch { expected: 1, actual: d.ndim() }) }
    }
}
impl TryFrom<IxDyn> for Ix2 {
    type Error = DimensionMismatch;
    fn try_from(d: IxDyn) -> Result<Self, Self::Error> {
        if d.ndim() == 2 { Ok(Ix2(d.dims[0], d.dims[1])) } else { Err(DimensionMismatch { expected: 2, actual: d.ndim() }) }
    }
}
// Ix3~Ix6 follow the same pattern
```

### 4.7 DimensionMismatch Error

```rust
/// Error returned when a dimension conversion fails due to rank mismatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionMismatch {
    /// The expected number of dimensions for the target type.
    pub expected: usize,
    /// The actual number of dimensions in the source.
    pub actual: usize,
}

impl core::fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dimension mismatch: expected {} dimensions, got {}", self.expected, self.actual)
    }
}

impl std::error::Error for DimensionMismatch {}
```

### 4.8 Type Aliases (in tensor.rs, listed here for reference)

```rust
/// Type alias for dynamically-dimensioned tensors.
pub type TensorDyn<A> = TensorBase<Owned<A>, IxDyn>;

/// Type alias for 0-dim (scalar) tensors.
pub type Tensor0<A> = TensorBase<Owned<A>, Ix0>;

/// Type alias for 1-dim (vector) tensors.
pub type Tensor1<A> = TensorBase<Owned<A>, Ix1>;

/// Type alias for 2-dim (matrix) tensors.
pub type Tensor2<A> = TensorBase<Owned<A>, Ix2>;

/// Type alias for 3-dim tensors.
pub type Tensor3<A> = TensorBase<Owned<A>, Ix3>;

// ... up to Tensor6<A>
```

---

## 5. 内部实现设计

### 5.1 静态维度存储

静态维度类型 `Ix0`~`Ix6` 使用固定数量的 `usize` 字段：

| 类型 | 存储 | 栈大小 |
|------|------|--------|
| Ix0 | 零大小类型 (ZST) | 0 bytes |
| Ix1 | `usize` × 1 | 8 bytes |
| Ix2 | `usize` × 2 | 16 bytes |
| Ix3 | `usize` × 3 | 24 bytes |
| Ix4 | `usize` × 4 | 32 bytes |
| Ix5 | `usize` × 5 | 40 bytes |
| Ix6 | `usize` × 6 | 48 bytes |

**`#[repr(C)]`** 标注在 Ix2~Ix6 上，保证字段连续排列，使 `slice()` 可安全地将字段首地址视为 `&[usize; N]`。

`slice()` 实现使用 `unsafe` 将字段指针转为切片：

```rust
// Pattern used for Ix2 (same for Ix3~Ix6):
fn slice(&self) -> &[usize] {
    unsafe { core::slice::from_raw_parts(&self.0 as *const usize, N) }
}
```

**安全性论证**: `#[repr(C)]` 确保 `usize` 字段连续排列且无 padding（usize 已对齐），因此从首字段地址读取 N 个 usize 是安全的。

### 5.2 动态维度存储

`IxDyn` 使用 `Vec<usize>` 存储维度信息：

```rust
pub struct IxDyn {
    dims: Vec<usize>,
}
```

- 堆分配，支持任意维度数
- 当 `std` 特性关闭时，需要 `alloc` crate 支持（`Vec` 来自 `alloc::vec::Vec`）

### 5.3 步长计算

#### F-order（列优先，默认）

对于 shape `[m, n, k]`：

```
strides[0] = 1
strides[1] = m
strides[2] = m × n
```

公式：`strides[i] = product(shape[0..i])`

#### C-order（行优先）

对于 shape `[m, n, k]`：

```
strides[0] = n × k
strides[1] = k
strides[2] = 1
```

公式：`strides[i] = product(shape[(i+1)..ndim])`

#### 溢出保护

- `size_checked()` 使用 `checked_mul` 链
- `default_strides()` 使用 `checked_mul` 并在溢出时 panic
- Debug 构建中额外检查

### 5.4 维度转换实现策略

**静态 → IxDyn**:

直接构造 `Vec<usize>` 并填充字段值。`From` impl 提供零成本转换。

**IxDyn → 静态**:

1. 检查 `dims.len() == N`
2. 匹配则提取字段值构造目标类型
3. 不匹配返回 `DimensionMismatch`

**静态 → 静态**（通过 `into_dimension`）:

1. 源维度 `slice()` 获取数据
2. 目标类型 `try_from_dimension` 检查 ndim 匹配
3. 匹配则构造目标类型

---

## 6. 实现任务拆分

每个任务约 10-15 分钟。

### Phase 2A: 基础结构

- [ ] **T1**: 创建 `src/dimension.rs`，定义 `MemoryOrder` 枚举和 `DimensionMismatch` 错误类型
  - 文件: `src/dimension.rs`
  - 包含 Display impl, Error impl

- [ ] **T2**: 定义 `Dimension` trait 完整签名
  - 文件: `src/dimension.rs`
  - 所有关联常量和方法签名
  - `size()` 和 `size_checked()` 的默认实现

### Phase 2B: 静态维度类型

- [ ] **T3**: 实现 `Ix0` — ZST 类型，`Dimension` trait impl
  - 文件: `src/dimension.rs`
  - `Ix0` struct, `new()`, `Index`/`IndexMut` (panic), `Dimension` impl
  - 测试: `ndim() == 0`, `size() == 1`, `slice().is_empty()`

- [ ] **T4**: 实现 `Ix1` — 单字段类型
  - 文件: `src/dimension.rs`
  - `Ix1(usize)`, `Index`/`IndexMut`, `Dimension` impl
  - 测试: `ndim() == 1`, `size()`, `slice()` 返回正确值

- [ ] **T5**: 实现 `Ix2`，添加 `#[repr(C)]`
  - 文件: `src/dimension.rs`
  - 验证 `slice()` 的 unsafe 代码正确性
  - 测试: `ndim() == 2`, 两个字段均可通过 `slice()` 访问

- [ ] **T6**: 实现 `Ix3` ~ `Ix6`
  - 文件: `src/dimension.rs`
  - 复用 Ix2 的模式，参数数量递增
  - 测试: 每个 `ndim()` 和 `slice()` 正确

### Phase 2C: 动态维度类型

- [ ] **T7**: 实现 `IxDyn` — `Vec<usize>` 后端
  - 文件: `src/dimension.rs`
  - `IxDyn` struct, `new()`, `zeros()`, `into_raw()`
  - `From<Vec<usize>>`, `From<&[usize]>`
  - `Index`/`IndexMut`, `Dimension` impl
  - 测试: 任意维度数, `ndim()` 正确

### Phase 2D: 转换实现

- [ ] **T8**: 实现 `From<IxN> for IxDyn` (N=0..6)
  - 文件: `src/dimension.rs`
  - 7 个 `From` impl
  - 测试: 每个静态类型转 IxDyn 后 `slice()` 值一致

- [ ] **T9**: 实现 `TryFrom<IxDyn> for IxN` (N=0..6)
  - 文件: `src/dimension.rs`
  - 7 个 `TryFrom` impl
  - 测试: 成功和失败路径

### Phase 2E: 步长计算

- [ ] **T10**: 实现 `default_strides()` 和 `default_strides_order()`
  - 文件: `src/dimension.rs`
  - F-order 和 C-order 步长计算
  - 溢出保护
  - 测试: 已知 shape 的步长值验证

### Phase 2F: 集成

- [ ] **T11**: 在 `lib.rs` 中注册模块和 re-export
  - 文件: `src/lib.rs`
  - `pub mod dimension;`
  - `pub use dimension::{...};`

---

## 7. 测试计划

### 7.1 单元测试

| 测试组 | 覆盖内容 | 数量估计 |
|--------|----------|----------|
| Ix0 基础 | ndim, size, slice, default_strides | 4 |
| Ix1 基础 | ndim, size, slice, index access | 5 |
| Ix2 基础 | ndim, size, slice, index access, repr(C) | 6 |
| Ix3~Ix6 基础 | 同上 | 每个 5 |
| IxDyn 基础 | new, zeros, ndim, slice, index | 8 |
| 静态→动态转换 | From impls for Ix0..Ix6 | 7 |
| 动态→静态转换 | TryFrom 成功 + 失败路径 | 14 |
| 步长计算 | F-order, C-order, 溢出保护 | 8 |
| 边界情况 | 空 IxDyn, 大维度, 零维度大小 | 6 |

**总计约 70 个单元测试**

### 7.2 属性测试

```rust
#[cfg(test)]
mod property_tests {
    use super::*;

    // Property: static → dynamic → static roundtrip
    fn roundtrip_static_dynamic<D: Dimension + Clone>(d: &D) {
        let dyn_d: IxDyn = d.clone().into_dimension().unwrap();
        let roundtrip: D = dyn_d.into_dimension().unwrap();
        assert_eq!(d.slice(), roundtrip.slice());
    }

    // Property: size() == product of slice()
    fn size_matches_product<D: Dimension>(d: &D) {
        assert_eq!(d.size(), d.slice().iter().product::<usize>());
    }

    // Property: F-strides are monotonically non-decreasing for sorted shapes
    fn f_strides_monotonic<D: Dimension>(d: &D) {
        if d.ndim() <= 1 { return; }
        let strides = d.default_strides();
        for w in strides.slice().windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}
```

### 7.3 测试文件位置

```
src/dimension.rs       // #[cfg(test)] mod tests { ... }
```

所有维度测试内联在模块中，不需要单独的集成测试文件。

---

## 8. no_std 兼容性

```rust
// dimension.rs 顶部
#![no_std] // when std feature is disabled

// 需要的 alloc 类型
extern crate alloc;
use alloc::vec::Vec;
```

`Ix0`~`Ix6` 是纯栈类型，不需要 alloc。只有 `IxDyn` 需要 `Vec`，通过 `alloc` crate 支持。

`MemoryOrder` 和 `DimensionMismatch` 是纯数据类型，完全 `no_std` 兼容。

---

## 9. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 栈分配 | Ix0~Ix6 全部栈分配，无堆开销 |
| ZST 优化 | Ix0 是零大小类型，编译器完全消除 |
| 内联 | 所有 ndim(), slice() 标注 `#[inline]` |
| 单态化 | Dimension trait 在泛型上下文中单态化，无虚调用开销 |
| 步长缓存 | default_strides() 返回同类型，无额外分配 |
| repr(C) | 保证字段连续，slice() 是一次指针转换 |

---

## 10. 与其他模块的交互

```
dimension.rs ─────┐
                  │
    ┌─────────────┼─────────────────┐
    │             │                 │
    v             v                 v
 tensor.rs    layout.rs       broadcast.rs
 (D 类型参数) (步长计算)      (广播规则)
    │
    ├── construction.rs  (构造时指定 D)
    ├── indexing.rs      (索引需要 D)
    ├── shape/*          (reshape 返回同 D)
    └── ffi.rs           (导出 D 信息)
```

`Dimension` 是被所有 API 模块消费的核心 trait。它的设计稳定性直接影响整个库的 API 表面。
