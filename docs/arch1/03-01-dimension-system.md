# Senon 维度系统设计文档

> 版本: 0.1.0 | 最后更新: 2026-03-28

---

## 1. 模块概述

### 1.1 职责

维度系统（`dimension` 模块）负责定义和管理多维数组的形状表示。它提供：

- **静态维度类型**：`Ix0` 至 `Ix6`，在编译期确定维度数，零运行时开销
- **动态维度类型**：`IxDyn`，在运行时确定维度数，灵活性高
- **维度转换 trait**：`Dimension` 和 `IntoDimension`，支持类型安全的维度操作
- **轴标记类型**：`Axis`，用于指定沿哪个轴进行操作

### 1.2 在整体架构中的位置

```
依赖层级：
L0: error, private
L1: dimension  ← 当前模块
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5+: iter, ops, shape_ops, index 等
```

维度系统是核心基础设施模块，位于依赖层级的 L1 层。它被 `layout`、`storage`、`tensor` 等模块直接依赖。

### 1.3 依赖关系

```
dimension
├── error (DimensionError, ShapeError)
└── private (Sealed trait)
```

**依赖说明**：
- `error`：提供 `DimensionMismatch` 错误类型
- `private`：提供 `Sealed` trait，防止外部类型实现 `Dimension`

**被依赖**：
- `layout`：使用 `Dimension` 计算步长
- `storage`：使用 `Dimension` 表示形状
- `tensor`：使用 `Dimension` 作为核心泛型参数
- `shape_ops`：使用 `IntoDimension` 进行 reshape

---

## 2. 文件结构

```
src/dimension/
├── mod.rs             # Dimension trait 定义，模块导出
├── static.rs          # Ix0, Ix1, ..., Ix6 静态维度实现
├── dynamic.rs         # IxDyn 动态维度实现
├── into_dimension.rs  # IntoDimension trait 及其实现
└── axes.rs            # Axis 新类型及轴操作
```

### 2.1 mod.rs

**职责**：
- 定义 `Dimension` trait
- 重新导出所有公开类型
- 定义模块级常量（如 `MAX_DIMENSION`）

**公开接口**：
```rust
pub use self::static_dims::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};
pub use self::dynamic::IxDyn;
pub use self::into_dimension::IntoDimension;
pub use self::axes::Axis;

pub trait Dimension: private::Sealed + Clone + PartialEq + Eq + Debug { ... }

/// Maximum supported dimensionality.
pub const MAX_DIMENSION: usize = 6;
```

### 2.2 static.rs

**职责**：
- 定义 `Ix0` 至 `Ix6` 类型
- 为每个静态维度实现 `Dimension` trait
- 实现静态维度之间的转换

**公开接口**：
```rust
pub struct Ix0;
pub struct Ix1(pub usize);
pub struct Ix2(pub usize, pub usize);
pub struct Ix3(pub usize, pub usize, pub usize);
pub struct Ix4(pub usize, pub usize, pub usize, pub usize);
pub struct Ix5(pub usize, pub usize, pub usize, pub usize, pub usize);
pub struct Ix6(pub usize, pub usize, pub usize, pub usize, pub usize, pub usize);
```

### 2.3 dynamic.rs

**职责**：
- 定义 `IxDyn` 类型
- 实现 `Dimension` trait
- 实现与静态维度的互转

**公开接口**：
```rust
pub struct IxDyn {
    dims: Vec<usize>,
}
```

### 2.4 into_dimension.rs

**职责**：
- 定义 `IntoDimension` trait
- 为元组、数组、切片实现转换

**公开接口**：
```rust
pub trait IntoDimension {
    type Dim: Dimension;
    
    fn into_dimension(self) -> Self::Dim;
}
```

### 2.5 axes.rs

**职责**：
- 定义 `Axis` 新类型
- 提供轴操作辅助方法

**公开接口**：
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(pub usize);
```

---

## 3. Dimension trait 设计

### 3.1 完整 trait 定义

```rust
use core::fmt::Debug;
use crate::private::Sealed;

/// Trait for array dimension types.
///
/// This trait is sealed and cannot be implemented outside of this crate.
///
/// # Type Parameters
/// Implementations exist for `Ix0`, `Ix1`, ..., `Ix6` (static dimensions)
/// and `IxDyn` (dynamic dimension).
pub trait Dimension: Sealed + Clone + PartialEq + Eq + Debug + Send + Sync + 'static {
    /// The maximum number of dimensions this type can represent.
    /// For static dimensions, this equals the dimension count.
    /// For dynamic dimensions, this is `MAX_DIMENSION`.
    const NDIM: Option<usize>;
    
    /// Returns the number of dimensions (rank).
    ///
    /// For static dimensions, this is known at compile time.
    /// For dynamic dimensions, this is determined at runtime.
    fn ndim(&self) -> usize;
    
    /// Returns the shape as a slice of axis lengths.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// assert_eq!(dim.slice(), &[2, 3, 4]);
    /// ```
    fn slice(&self) -> &[usize];
    
    /// Returns a mutable reference to the shape slice.
    ///
    /// # Safety
    /// Caller must ensure the modification maintains dimension invariants.
    fn slice_mut(&mut self) -> &mut [usize];
    
    /// Computes strides for Fortran-order (column-major) layout.
    ///
    /// Returns strides in element units (not bytes).
    /// The first axis has stride 1 in F-order.
    ///
    /// # Returns
    /// A dimension type `Self` containing the computed strides.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// let strides = dim.strides_for_f_order();
    /// assert_eq!(strides.slice(), &[1, 2, 6]); // [1, dim[0], dim[0]*dim[1]]
    /// ```
    fn strides_for_f_order(&self) -> Self;
    
    /// Computes strides for C-order (row-major) layout.
    ///
    /// Returns strides in element units (not bytes).
    /// The last axis has stride 1 in C-order.
    ///
    /// # Returns
    /// A dimension type `Self` containing the computed strides.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// let strides = dim.strides_for_c_order();
    /// assert_eq!(strides.slice(), &[12, 4, 1]); // [dim[1]*dim[2], dim[2], 1]
    /// ```
    fn strides_for_c_order(&self) -> Self;
    
    /// Returns the total number of elements.
    ///
    /// This is the product of all axis lengths.
    /// For `Ix0`, this returns 1 (the scalar has one element).
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// assert_eq!(dim.size(), 24);
    /// ```
    fn size(&self) -> usize;
    
    /// Creates a dimension with all axes set to zero.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3::zeros();
    /// assert_eq!(dim.slice(), &[0, 0, 0]);
    /// ```
    fn zeros() -> Self;
    
    /// Creates a dimension with all axes set to one.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3::ones();
    /// assert_eq!(dim.slice(), &[1, 1, 1]);
    /// ```
    fn ones() -> Self;
    
    /// Converts this dimension to the target dimension type.
    ///
    /// # Type Parameters
    /// - `D`: Target dimension type
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if the dimension counts don't match
    /// when converting from dynamic to static dimension.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// let dyn_dim: IxDyn = dim.into_dimension()?;
    /// assert_eq!(dyn_dim.ndim(), 3);
    /// ```
    fn into_dimension<D>(self) -> Result<D, DimensionMismatch>
    where
        D: Dimension,
    {
        // Default implementation provided in trait
        // Each type provides specialized implementations
    }
    
    /// Converts to dynamic dimension.
    ///
    /// This conversion always succeeds.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// let dyn_dim = dim.into_dyn();
    /// assert_eq!(dyn_dim.ndim(), 3);
    /// ```
    fn into_dyn(self) -> IxDyn;
    
    /// Attempts to convert from dynamic dimension.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `dyn_dim.ndim()` doesn't match
    /// this type's dimension count.
    ///
    /// # Examples
    /// ```ignore
    /// let dyn_dim = IxDyn::from_slice(&[2, 3, 4]);
    /// let dim: Ix3 = Ix3::try_from_dyn(dyn_dim)?;
    /// ```
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch>
    where
        Self: Sized;
    
    /// Returns the axis length at the given index.
    ///
    /// # Panics
    /// Panics if `axis >= self.ndim()`.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 3, 4);
    /// assert_eq!(dim.axis(Axis(1)), 3);
    /// ```
    fn axis(&self, axis: Axis) -> usize {
        self.slice()[axis.0]
    }
    
    /// Sets the axis length at the given index.
    ///
    /// # Panics
    /// Panics if `axis >= self.ndim()`.
    fn set_axis(&mut self, axis: Axis, value: usize) {
        self.slice_mut()[axis.0] = value;
    }
    
    /// Returns the last axis length.
    ///
    /// For `Ix0`, returns 1.
    fn last_axis(&self) -> usize {
        self.slice().last().copied().unwrap_or(1)
    }
    
    /// Returns the first axis length.
    ///
    /// For `Ix0`, returns 1.
    fn first_axis(&self) -> usize {
        self.slice().first().copied().unwrap_or(1)
    }
    
    /// Checks if any axis has zero length.
    ///
    /// # Examples
    /// ```ignore
    /// let dim = Ix3(2, 0, 4);
    /// assert!(dim.contains_zero());
    /// ```
    fn contains_zero(&self) -> bool {
        self.slice().iter().any(|&d| d == 0)
    }
    
    /// Returns an iterator over axis lengths.
    fn iter(&self) -> core::slice::Iter<'_, usize> {
        self.slice().iter()
    }
    
    /// Creates dimension from a slice.
    ///
    /// # Panics
    /// Panics if the slice length doesn't match the dimension count.
    fn from_slice(slice: &[usize]) -> Self
    where
        Self: Sized;
}
```

### 3.2 关联类型和常量

| 成员 | 类型 | 说明 |
|------|------|------|
| `NDIM` | `Option<usize>` | 静态维度为 `Some(N)`，动态维度为 `None` |
| `ndim()` | `fn(&self) -> usize` | 返回维度数 |
| `slice()` | `fn(&self) -> &[usize]` | 返回形状切片 |
| `slice_mut()` | `fn(&mut self) -> &mut [usize]` | 返回可变形状切片 |
| `size()` | `fn(&self) -> usize` | 返回元素总数 |
| `zeros()` | `fn() -> Self` | 创建全零维度 |
| `ones()` | `fn() -> Self` | 创建全一维度 |

### 3.3 默认实现

以下方法在 trait 中提供默认实现：

```rust
// Default implementations in trait definition
fn contains_zero(&self) -> bool { ... }
fn iter(&self) -> core::slice::Iter<'_, usize> { ... }
fn axis(&self, axis: Axis) -> usize { ... }
fn set_axis(&mut self, axis: Axis, value: usize) { ... }
fn last_axis(&self) -> usize { ... }
fn first_axis(&self) -> usize { ... }
```

以下方法需要各类型特化实现：

```rust
// Must be implemented by each type
fn ndim(&self) -> usize;
fn slice(&self) -> &[usize];
fn slice_mut(&mut self) -> &mut [usize];
fn size(&self) -> usize;
fn zeros() -> Self;
fn ones() -> Self;
fn strides_for_f_order(&self) -> Self;
fn strides_for_c_order(&self) -> Self;
fn into_dyn(self) -> IxDyn;
fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch>;
fn from_slice(slice: &[usize]) -> Self;
```

---

## 4. 静态维度类型 Ix0-Ix6

### 4.1 内部表示

```rust
// src/dimension/static.rs

/// Zero-dimensional (scalar) dimension.
///
/// Represents a single element with no axes.
/// - `ndim() == 0`
/// - `slice() == &[]`
/// - `size() == 1`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix0;

/// One-dimensional dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix1(pub usize);

/// Two-dimensional dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix2(pub usize, pub usize);

/// Three-dimensional dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix3(pub usize, pub usize, pub usize);

/// Four-dimensional dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix4(pub usize, pub usize, pub usize, pub usize);

/// Five-dimensional dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix5(pub usize, pub usize, pub usize, pub usize, pub usize);

/// Six-dimensional dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix6(pub usize, pub usize, pub usize, pub usize, pub usize, pub usize);
```

### 4.2 Ix0 特殊语义

**Ix0 表示零维标量**，具有以下特殊属性：

| 属性 | 值 | 说明 |
|------|-----|------|
| `ndim()` | 0 | 没有维度 |
| `slice()` | `&[]` | 空切片 |
| `size()` | 1 | 一个元素（标量） |
| `strides_for_f_order()` | `Ix0` | 无步长 |
| `strides_for_c_order()` | `Ix0` | 无步长 |

**实现要点**：

```rust
impl Dimension for Ix0 {
    const NDIM: Option<usize> = Some(0);
    
    fn ndim(&self) -> usize {
        0
    }
    
    fn slice(&self) -> &[usize] {
        &[]
    }
    
    fn slice_mut(&mut self) -> &mut [usize] {
        &mut []
    }
    
    fn size(&self) -> usize {
        // IMPORTANT: Ix0 represents a single scalar element
        1
    }
    
    fn zeros() -> Self {
        Ix0
    }
    
    fn ones() -> Self {
        Ix0
    }
    
    fn strides_for_f_order(&self) -> Self {
        Ix0
    }
    
    fn strides_for_c_order(&self) -> Self {
        Ix0
    }
    
    fn into_dyn(self) -> IxDyn {
        IxDyn::new()
    }
    
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch> {
        if dyn_dim.ndim() == 0 {
            Ok(Ix0)
        } else {
            Err(DimensionMismatch {
                expected: 0,
                actual: dyn_dim.ndim(),
            })
        }
    }
    
    fn from_slice(slice: &[usize]) -> Self {
        assert!(slice.is_empty(), "Ix0 requires empty slice");
        Ix0
    }
}
```

### 4.3 Ix1-Ix6 实现模式

以 `Ix3` 为例：

```rust
impl Dimension for Ix3 {
    const NDIM: Option<usize> = Some(3);
    
    #[inline]
    fn ndim(&self) -> usize {
        3
    }
    
    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: Ix3 is #[repr(C)] with 3 usize fields
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const usize,
                3
            )
        }
    }
    
    #[inline]
    fn slice_mut(&mut self) -> &mut [usize] {
        // SAFETY: Ix3 is #[repr(C)] with 3 usize fields
        unsafe {
            core::slice::from_raw_parts_mut(
                self as *mut Self as *mut usize,
                3
            )
        }
    }
    
    #[inline]
    fn size(&self) -> usize {
        self.0.wrapping_mul(self.1).wrapping_mul(self.2)
    }
    
    #[inline]
    fn zeros() -> Self {
        Ix3(0, 0, 0)
    }
    
    #[inline]
    fn ones() -> Self {
        Ix3(1, 1, 1)
    }
    
    #[inline]
    fn strides_for_f_order(&self) -> Self {
        Ix3(1, self.0, self.0 * self.1)
    }
    
    #[inline]
    fn strides_for_c_order(&self) -> Self {
        Ix3(self.1 * self.2, self.2, 1)
    }
    
    #[inline]
    fn into_dyn(self) -> IxDyn {
        IxDyn::from_slice(&[self.0, self.1, self.2])
    }
    
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch> {
        if dyn_dim.ndim() == 3 {
            let slice = dyn_dim.slice();
            Ok(Ix3(slice[0], slice[1], slice[2]))
        } else {
            Err(DimensionMismatch {
                expected: 3,
                actual: dyn_dim.ndim(),
            })
        }
    }
    
    fn from_slice(slice: &[usize]) -> Self {
        assert_eq!(slice.len(), 3, "Ix3 requires exactly 3 elements");
        Ix3(slice[0], slice[1], slice[2])
    }
}
```

### 4.4 辅助 trait 实现

```rust
// Index access for Ix1-Ix6
impl core::ops::Index<usize> for Ix3 {
    type Output = usize;
    
    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("index out of bounds: {} >= 3", index),
        }
    }
}

impl core::ops::IndexMut<usize> for Ix3 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut usize {
        match index {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            _ => panic!("index out of bounds: {} >= 3", index),
        }
    }
}

// IntoIterator for Ix1-Ix6
impl IntoIterator for Ix3 {
    type Item = usize;
    type IntoIter = core::array::IntoIter<usize, 3>;
    
    fn into_iter(self) -> Self::IntoIter {
        [self.0, self.1, self.2].into_iter()
    }
}

// From tuple
impl From<(usize, usize, usize)> for Ix3 {
    #[inline]
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        Ix3(a, b, c)
    }
}

impl From<Ix3> for (usize, usize, usize) {
    #[inline]
    fn from(dim: Ix3) -> Self {
        (dim.0, dim.1, dim.2)
    }
}
```

---

## 5. 动态维度 IxDyn

### 5.1 内部表示

```rust
// src/dimension/dynamic.rs

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use alloc::vec;
use alloc::borrow::Cow;

/// Dynamic dimension type.
///
/// The number of dimensions is determined at runtime.
/// Supports 0 to `MAX_DIMENSION` (6) dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct IxDyn {
    dims: Vec<usize>,
}
```

### 5.2 SmallVec 优化考虑

**当前设计**：使用 `Vec<usize>`

**SmallVec 优化分析**：

| 方案 | 优点 | 缺点 |
|------|------|------|
| `Vec<usize>` | 简单、稳定 | 堆分配开销 |
| `SmallVec<[usize; 6]>` | 栈分配（≤6维）| 增加依赖、结构体更大 |
| `ArrayVec<usize, 6>` | 纯栈分配 | 增加依赖、无溢出处理 |

**决策**：当前版本使用 `Vec<usize>`，保持简单。

**未来优化路径**：
1. 如果性能分析显示维度分配是瓶颈，可考虑引入 `smallvec` crate
2. 使用 feature gate 控制：`feature = "smallvec_optimization"`
3. 保持 API 兼容，仅内部实现变更

### 5.3 Dimension trait 实现

```rust
impl Dimension for IxDyn {
    const NDIM: Option<usize> = None; // Dynamic, unknown at compile time
    
    #[inline]
    fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    #[inline]
    fn slice(&self) -> &[usize] {
        &self.dims
    }
    
    #[inline]
    fn slice_mut(&mut self) -> &mut [usize] {
        &mut self.dims
    }
    
    #[inline]
    fn size(&self) -> usize {
        self.dims.iter().product()
    }
    
    #[inline]
    fn zeros() -> Self {
        IxDyn { dims: vec![] }
    }
    
    fn ones() -> Self {
        // Cannot create ones() for dynamic dimension without knowing ndim
        // This is a limitation - users should use from_slice or constructor
        panic!("IxDyn::ones() requires explicit dimension count; use from_element(1, ndim)")
    }
    
    fn strides_for_f_order(&self) -> Self {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut stride = 1usize;
        for &dim in &self.dims {
            strides.push(stride);
            stride = stride.wrapping_mul(dim);
        }
        IxDyn { dims: strides }
    }
    
    fn strides_for_c_order(&self) -> Self {
        let n = self.dims.len();
        let mut strides = vec![0usize; n];
        if n > 0 {
            let mut stride = 1usize;
            for i in (0..n).rev() {
                strides[i] = stride;
                stride = stride.wrapping_mul(self.dims[i]);
            }
        }
        IxDyn { dims: strides }
    }
    
    #[inline]
    fn into_dyn(self) -> IxDyn {
        self
    }
    
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch> {
        // IxDyn can always accept any IxDyn
        Ok(dyn_dim)
    }
    
    #[inline]
    fn from_slice(slice: &[usize]) -> Self {
        IxDyn { dims: slice.to_vec() }
    }
}
```

### 5.4 构造方法

```rust
impl IxDyn {
    /// Creates an empty (0-dimensional) dynamic dimension.
    #[inline]
    pub fn new() -> Self {
        IxDyn { dims: vec![] }
    }
    
    /// Creates a dynamic dimension from a slice.
    #[inline]
    pub fn from_slice(slice: &[usize]) -> Self {
        IxDyn { dims: slice.to_vec() }
    }
    
    /// Creates a dynamic dimension from a Vec.
    #[inline]
    pub fn from_vec(dims: Vec<usize>) -> Self {
        IxDyn { dims }
    }
    
    /// Creates a dynamic dimension with all axes set to a given value.
    ///
    /// # Arguments
    /// - `value`: The value for each axis
    /// - `ndim`: Number of dimensions
    ///
    /// # Panics
    /// Panics if `ndim > MAX_DIMENSION`.
    #[inline]
    pub fn from_element(value: usize, ndim: usize) -> Self {
        assert!(ndim <= MAX_DIMENSION, "dimension {} exceeds MAX_DIMENSION {}", ndim, MAX_DIMENSION);
        IxDyn { dims: vec![value; ndim] }
    }
    
    /// Creates a dynamic dimension filled with ones.
    #[inline]
    pub fn ones(ndim: usize) -> Self {
        Self::from_element(1, ndim)
    }
    
    /// Creates a dynamic dimension filled with zeros.
    #[inline]
    pub fn zeros(ndim: usize) -> Self {
        Self::from_element(0, ndim)
    }
    
    /// Consumes and returns the inner Vec.
    #[inline]
    pub fn into_vec(self) -> Vec<usize> {
        self.dims
    }
}
```

### 5.5 no_std 兼容性

```rust
// In dynamic.rs

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// All Vec operations work the same in both modes
// The alloc crate provides Vec for no_std environments
```

**no_std 注意事项**：
- IxDyn 需要 `alloc` crate（提供 `Vec`）
- 在 `Cargo.toml` 中配置：
  ```toml
  [target.'cfg(not(feature = "std"))'.dependencies]
  alloc = { version = "0.2", default-features = false, features = ["unstable"] }
  ```

---

## 6. IntoDimension trait

### 6.1 trait 定义

```rust
// src/dimension/into_dimension.rs

/// Trait for converting types into dimension types.
///
/// This trait allows flexible construction of dimension types from
/// various sources: tuples, arrays, slices, and other dimension types.
pub trait IntoDimension {
    /// The dimension type produced by this conversion.
    type Dim: Dimension;
    
    /// Consumes self and produces a dimension type.
    fn into_dimension(self) -> Self::Dim;
}
```

### 6.2 实现列表

```rust
// Identity conversion
impl<D: Dimension> IntoDimension for D {
    type Dim = D;
    
    #[inline]
    fn into_dimension(self) -> Self::Dim {
        self
    }
}

// From tuples to Ix1-Ix6
impl IntoDimension for (usize,) {
    type Dim = Ix1;
    
    #[inline]
    fn into_dimension(self) -> Ix1 {
        Ix1(self.0)
    }
}

impl IntoDimension for (usize, usize) {
    type Dim = Ix2;
    
    #[inline]
    fn into_dimension(self) -> Ix2 {
        Ix2(self.0, self.1)
    }
}

impl IntoDimension for (usize, usize, usize) {
    type Dim = Ix3;
    
    #[inline]
    fn into_dimension(self) -> Ix3 {
        Ix3(self.0, self.1, self.2)
    }
}

impl IntoDimension for (usize, usize, usize, usize) {
    type Dim = Ix4;
    
    #[inline]
    fn into_dimension(self) -> Ix4 {
        Ix4(self.0, self.1, self.2, self.3)
    }
}

impl IntoDimension for (usize, usize, usize, usize, usize) {
    type Dim = Ix5;
    
    #[inline]
    fn into_dimension(self) -> Ix5 {
        Ix5(self.0, self.1, self.2, self.3, self.4)
    }
}

impl IntoDimension for (usize, usize, usize, usize, usize, usize) {
    type Dim = Ix6;
    
    #[inline]
    fn into_dimension(self) -> Ix6 {
        Ix6(self.0, self.1, self.2, self.3, self.4, self.5)
    }
}

// From arrays to Ix1-Ix6
impl IntoDimension for [usize; 1] {
    type Dim = Ix1;
    
    #[inline]
    fn into_dimension(self) -> Ix1 {
        Ix1(self[0])
    }
}

impl IntoDimension for [usize; 2] {
    type Dim = Ix2;
    
    #[inline]
    fn into_dimension(self) -> Ix2 {
        Ix2(self[0], self[1])
    }
}

impl IntoDimension for [usize; 3] {
    type Dim = Ix3;
    
    #[inline]
    fn into_dimension(self) -> Ix3 {
        Ix3(self[0], self[1], self[2])
    }
}

// ... similar implementations for [usize; 4], [usize; 5], [usize; 6]

// From slices to IxDyn
impl IntoDimension for &[usize] {
    type Dim = IxDyn;
    
    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn::from_slice(self)
    }
}

impl IntoDimension for Vec<usize> {
    type Dim = IxDyn;
    
    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn::from_vec(self)
    }
}

// From arrays to IxDyn (for flexibility)
impl<const N: usize> IntoDimension for [usize; N] {
    type Dim = IxDyn;
    
    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn::from_slice(&self)
    }
}
```

### 6.3 使用示例

```rust
// From tuple
let dim: Ix3 = (2, 3, 4).into_dimension();

// From array
let dim: Ix2 = [3, 4].into_dimension();

// From slice (to dynamic)
let dim: IxDyn = (&[2, 3, 4, 5][..]).into_dimension();

// From Vec (to dynamic)
let dim: IxDyn = vec![2, 3, 4].into_dimension();

// Identity
let dim: Ix3 = Ix3(2, 3, 4).into_dimension();
```

---

## 7. Axis 类型

### 7.1 类型定义

```rust
// src/dimension/axes.rs

/// Axis marker type.
///
/// Represents an axis index in a multi-dimensional array.
/// Provides type safety and clarity over raw `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Axis(pub usize);

impl Axis {
    /// Creates a new axis marker.
    ///
    /// # Examples
    /// ```
    /// let axis = Axis(0);
    /// assert_eq!(axis.index(), 0);
    /// ```
    #[inline]
    pub fn new(axis: usize) -> Self {
        Axis(axis)
    }
    
    /// Returns the axis index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
    
    /// Returns the next axis.
    #[inline]
    pub fn next(self) -> Self {
        Axis(self.0 + 1)
    }
    
    /// Returns the previous axis, or None if this is axis 0.
    #[inline]
    pub fn prev(self) -> Option<Self> {
        self.0.checked_sub(1).map(Axis)
    }
}

// Display implementation
impl core::fmt::Display for Axis {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Axis({})", self.0)
    }
}

// From usize
impl From<usize> for Axis {
    #[inline]
    fn from(axis: usize) -> Self {
        Axis(axis)
    }
}

// From Axis to usize
impl From<Axis> for usize {
    #[inline]
    fn from(axis: Axis) -> Self {
        axis.0
    }
}
```

### 7.2 轴操作辅助方法

```rust
impl Axis {
    /// Checks if this is the first axis (axis 0).
    #[inline]
    pub fn is_first(self) -> bool {
        self.0 == 0
    }
    
    /// Checks if this is the last axis for the given dimension.
    ///
    /// # Arguments
    /// - `ndim`: Total number of dimensions
    #[inline]
    pub fn is_last(self, ndim: usize) -> bool {
        self.0 == ndim.saturating_sub(1)
    }
    
    /// Wraps negative indices to positive.
    ///
    /// Python-style negative indexing: -1 is the last axis.
    ///
    /// # Arguments
    /// - `ndim`: Total number of dimensions
    ///
    /// # Returns
    /// The wrapped axis index.
    ///
    /// # Panics
    /// Panics if the absolute value exceeds `ndim`.
    pub fn wrap(self, ndim: usize) -> Self {
        let index = if self.0 >= ndim {
            // Negative indexing: wrap around
            self.0.wrapping_sub(ndim)
        } else {
            self.0
        };
        assert!(index < ndim, "axis {} out of bounds for dimension {}", index, ndim);
        Axis(index)
    }
}
```

### 7.3 预定义常量

```rust
impl Axis {
    /// First axis (axis 0).
    pub const AXIS_0: Axis = Axis(0);
    
    /// Second axis (axis 1).
    pub const AXIS_1: Axis = Axis(1);
    
    /// Third axis (axis 2).
    pub const AXIS_2: Axis = Axis(2);
    
    /// Fourth axis (axis 3).
    pub const AXIS_3: Axis = Axis(3);
    
    /// Fifth axis (axis 4).
    pub const AXIS_4: Axis = Axis(4);
    
    /// Sixth axis (axis 5).
    pub const AXIS_5: Axis = Axis(5);
}
```

---

## 8. 维度互转规则

### 8.1 转换规则表

**静态 → 动态（总是成功）**

| 源类型 | 目标类型 | 结果 |
|--------|----------|------|
| `Ix0` | `IxDyn` | `IxDyn { dims: vec![] }` |
| `Ix1` | `IxDyn` | `IxDyn { dims: vec![n] }` |
| `Ix2` | `IxDyn` | `IxDyn { dims: vec![m, n] }` |
| `Ix3` | `IxDyn` | `IxDyn { dims: vec![a, b, c] }` |
| `Ix4` | `IxDyn` | `IxDyn { dims: vec![...] }` |
| `Ix5` | `IxDyn` | `IxDyn { dims: vec![...] }` |
| `Ix6` | `IxDyn` | `IxDyn { dims: vec![...] }` |

**动态 → 静态（需维度匹配）**

| 源类型 | 条件 | 成功结果 | 失败结果 |
|--------|------|----------|----------|
| `IxDyn` | `ndim == 0` | `Ix0` | `Err(DimensionMismatch)` |
| `IxDyn` | `ndim == 1` | `Ix1` | `Err(DimensionMismatch)` |
| `IxDyn` | `ndim == 2` | `Ix2` | `Err(DimensionMismatch)` |
| `IxDyn` | `ndim == 3` | `Ix3` | `Err(DimensionMismatch)` |
| `IxDyn` | `ndim == 4` | `Ix4` | `Err(DimensionMismatch)` |
| `IxDyn` | `ndim == 5` | `Ix5` | `Err(DimensionMismatch)` |
| `IxDyn` | `ndim == 6` | `Ix6` | `Err(DimensionMismatch)` |

### 8.2 转换 API

```rust
// Trait 方法
trait Dimension {
    /// Convert to dynamic dimension (always succeeds)
    fn into_dyn(self) -> IxDyn;
    
    /// Try to convert from dynamic dimension
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch>;
    
    /// Generic conversion with type inference
    fn into_dimension<D>(self) -> Result<D, DimensionMismatch>
    where
        D: Dimension;
}

// 便捷方法
impl IxDyn {
    /// Try to convert to a specific static dimension.
    pub fn into_static<const N: usize>(self) -> Result<[usize; N], DimensionMismatch> {
        if self.ndim() == N {
            Ok(self.dims.try_into().unwrap())
        } else {
            Err(DimensionMismatch {
                expected: N,
                actual: self.ndim(),
            })
        }
    }
}
```

### 8.3 错误类型

```rust
// In error.rs or dimension module

/// Error returned when dimension conversion fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionMismatch {
    /// Expected number of dimensions.
    pub expected: usize,
    /// Actual number of dimensions.
    pub actual: usize,
}

impl core::fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "dimension mismatch: expected {} dimensions, got {}",
            self.expected, self.actual
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DimensionMismatch {}
```

### 8.4 使用示例

```rust
// 静态 → 动态
let dim = Ix3(2, 3, 4);
let dyn_dim: IxDyn = dim.into_dyn();
assert_eq!(dyn_dim.slice(), &[2, 3, 4]);

// 动态 → 静态（成功）
let dyn_dim = IxDyn::from_slice(&[2, 3, 4]);
let dim: Ix3 = Ix3::try_from_dyn(dyn_dim)?;
assert_eq!(dim.slice(), &[2, 3, 4]);

// 动态 → 静态（失败）
let dyn_dim = IxDyn::from_slice(&[2, 3, 4, 5]);
let result: Result<Ix3, _> = Ix3::try_from_dyn(dyn_dim);
assert!(matches!(result, Err(DimensionMismatch { expected: 3, actual: 4 })));

// 泛型转换
let dim = Ix3(2, 3, 4);
let converted: IxDyn = dim.into_dimension()?;
```

---

## 9. Sealed trait 策略

### 9.1 设计目标

防止外部 crate 实现 `Dimension` trait，确保：
- 维度类型的行为可控
- 未来可以添加新的 trait 方法而不破坏兼容性
- 类型安全和不变量得到保证

### 9.2 实现方式

```rust
// src/private.rs

/// Sealed trait for preventing external implementations.
///
/// This trait is private and cannot be implemented by external crates.
/// Only types defined in this crate can implement `Dimension`.
pub trait Sealed {}

// Implement Sealed for all dimension types in this crate
impl Sealed for Ix0 {}
impl Sealed for Ix1 {}
impl Sealed for Ix2 {}
impl Sealed for Ix3 {}
impl Sealed for Ix4 {}
impl Sealed for Ix5 {}
impl Sealed for Ix6 {}
impl Sealed for IxDyn {}
```

```rust
// src/dimension/mod.rs

mod private {
    pub use crate::private::Sealed;
}

// Dimension trait requires Sealed
pub trait Dimension: private::Sealed + ... {
    // ...
}
```

### 9.3 为什么使用 Sealed trait

| 原因 | 说明 |
|------|------|
| **API 稳定性** | 可以添加新的 trait 方法而不破坏外部实现 |
| **类型安全** | 确保只有经过验证的类型实现了 `Dimension` |
| **不变量保证** | 维度类型的行为完全可控 |
| **文档清晰** | 用户知道哪些类型可用 |

### 9.4 替代方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| Sealed trait | 简单、Rust 惯用 | 需要额外模块 |
| `#[doc(hidden)]` trait | 无额外模块 | 仍然可能被实现 |
| 宏生成 | DRY | 增加复杂度 |

**选择**：使用 Sealed trait，这是 Rust 生态的标准做法（参考 `std::io::Read` 的 `__Drop` 模式）。

---

## 10. 与其他模块的交互

### 10.1 与 layout 模块的接口

```rust
// layout 模块使用 Dimension 计算步长

// In src/layout/strides.rs
use crate::dimension::Dimension;

/// Computes strides for a given shape and memory order.
pub fn compute_strides<D: Dimension>(shape: &D, order: MemoryOrder) -> D {
    match order {
        MemoryOrder::F => shape.strides_for_f_order(),
        MemoryOrder::C => shape.strides_for_c_order(),
    }
}
```

### 10.2 与 storage 模块的接口

```rust
// storage 模块使用 Dimension 表示形状

// In src/storage/mod.rs
use crate::dimension::Dimension;

pub trait Storage {
    type Elem;
    type Dim: Dimension;
    
    fn shape(&self) -> &Self::Dim;
    fn len(&self) -> usize {
        self.shape().size()
    }
}
```

### 10.3 与 tensor 模块的接口

```rust
// tensor 模块使用 Dimension 作为泛型参数

// In src/tensor/mod.rs
use crate::dimension::Dimension;

pub struct TensorBase<S, D: Dimension> {
    storage: S,
    shape: D,
    strides: D,
    offset: usize,
}

impl<S, D: Dimension> TensorBase<S, D> {
    pub fn shape(&self) -> &D {
        &self.shape
    }
    
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
}
```

### 10.4 与 shape_ops 模块的接口

```rust
// shape_ops 模块使用 IntoDimension 进行 reshape

// In src/shape_ops/reshape.rs
use crate::dimension::{Dimension, IntoDimension};
use crate::tensor::TensorBase;

impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// Reshapes the tensor to the given dimensions.
    ///
    /// The total element count must remain the same.
    pub fn reshape<Sh>(self, shape: Sh) -> Result<TensorBase<S, Sh::Dim>, ShapeError>
    where
        Sh: IntoDimension,
        Sh::Dim: Dimension,
    {
        let new_dim = shape.into_dimension();
        if self.shape.size() != new_dim.size() {
            return Err(ShapeError::IncompatibleSize {
                from: self.shape.size(),
                to: new_dim.size(),
            });
        }
        // ...
    }
}
```

### 10.5 接口约定总结

| 模块 | 使用的 trait/类型 | 用途 |
|------|-------------------|------|
| `layout` | `Dimension` | 计算步长、检查连续性 |
| `storage` | `Dimension` | 关联类型、形状表示 |
| `tensor` | `Dimension` | 泛型参数、形状访问 |
| `shape_ops` | `Dimension`, `IntoDimension` | reshape、transpose |
| `iter` | `Dimension` | 迭代器泛型参数 |
| `ops` | `Dimension` | 运算泛型参数 |
| `index` | `Dimension`, `Axis` | 索引操作 |

---

## 11. 实现任务分解

### 任务 1：创建模块基础结构

**名称**：创建 dimension 模块文件结构

**涉及文件**：
- `src/dimension/mod.rs`
- `src/dimension/static.rs`（空文件）
- `src/dimension/dynamic.rs`（空文件）
- `src/dimension/into_dimension.rs`（空文件）
- `src/dimension/axes.rs`（空文件）

**前置依赖**：无

**验收标准**：
- [ ] 所有文件创建完成
- [ ] `mod.rs` 导出空模块
- [ ] `cargo check` 通过

---

### 任务 2：实现 Ix0 静态维度

**名称**：实现 Ix0 零维标量

**涉及文件**：
- `src/dimension/static.rs`
- `src/dimension/mod.rs`

**前置依赖**：任务 1

**验收标准**：
- [ ] `Ix0` 结构体定义
- [ ] `Dimension` trait 实现
- [ ] `size()` 返回 1
- [ ] `slice()` 返回空切片
- [ ] 单元测试：`test_ix0_size_is_one`
- [ ] 单元测试：`test_ix0_ndim_is_zero`

---

### 任务 3：实现 Ix1 静态维度

**名称**：实现 Ix1 一维向量

**涉及文件**：
- `src/dimension/static.rs`

**前置依赖**：任务 2

**验收标准**：
- [ ] `Ix1` 结构体定义
- [ ] `Dimension` trait 实现
- [ ] `Index<usize>` 实现
- [ ] 单元测试：`test_ix1_strides_f_order`
- [ ] 单元测试：`test_ix1_strides_c_order`

---

### 任务 4：实现 Ix2 静态维度

**名称**：实现 Ix2 二维矩阵

**涉及文件**：
- `src/dimension/static.rs`

**前置依赖**：任务 3

**验收标准**：
- [ ] `Ix2` 结构体定义
- [ ] `Dimension` trait 实现
- [ ] `From<(usize, usize)>` 实现
- [ ] 单元测试：`test_ix2_strides_f_order`
- [ ] 单元测试：`test_ix2_strides_c_order`

---

### 任务 5：实现 Ix3-Ix6 静态维度

**名称**：实现剩余静态维度类型

**涉及文件**：
- `src/dimension/static.rs`

**前置依赖**：任务 4

**验收标准**：
- [ ] `Ix3`, `Ix4`, `Ix5`, `Ix6` 结构体定义
- [ ] 所有 `Dimension` trait 实现
- [ ] 对应的 `From<tuple>` 实现
- [ ] 单元测试：`test_ix3_size_calculation`
- [ ] 单元测试：`test_ix6_max_dimensions`

---

### 任务 6：实现 IxDyn 动态维度

**名称**：实现 IxDyn 动态维度类型

**涉及文件**：
- `src/dimension/dynamic.rs`

**前置依赖**：任务 5

**验收标准**：
- [ ] `IxDyn` 结构体定义（使用 `Vec<usize>`）
- [ ] `Dimension` trait 实现
- [ ] 构造方法：`new()`, `from_slice()`, `from_vec()`
- [ ] no_std 兼容（使用 `alloc::vec::Vec`）
- [ ] 单元测试：`test_ixdyn_from_slice`
- [ ] 单元测试：`test_ixdyn_strides`

---

### 任务 7：实现维度互转

**名称**：实现静态↔动态维度转换

**涉及文件**：
- `src/dimension/static.rs`
- `src/dimension/dynamic.rs`
- `src/error.rs`（添加 `DimensionMismatch`）

**前置依赖**：任务 6

**验收标准**：
- [ ] `Ix0-Ix6::into_dyn()` 实现
- [ ] `Ix0-Ix6::try_from_dyn()` 实现
- [ ] `DimensionMismatch` 错误类型
- [ ] 单元测试：`test_static_to_dyn_conversion`
- [ ] 单元测试：`test_dyn_to_static_success`
- [ ] 单元测试：`test_dyn_to_static_failure`

---

### 任务 8：实现 IntoDimension trait

**名称**：实现维度转换 trait

**涉及文件**：
- `src/dimension/into_dimension.rs`

**前置依赖**：任务 7

**验收标准**：
- [ ] `IntoDimension` trait 定义
- [ ] 为 `Ix0-Ix6` 实现（identity）
- [ ] 为 `tuple` 实现
- [ ] 为 `array` 实现
- [ ] 为 `&[usize]` 和 `Vec<usize>` 实现
- [ ] 单元测试：`test_tuple_to_ix3`
- [ ] 单元测试：`test_slice_to_ixdyn`

---

### 任务 9：实现 Axis 类型

**名称**：实现轴标记类型

**涉及文件**：
- `src/dimension/axes.rs`

**前置依赖**：任务 8

**验收标准**：
- [ ] `Axis` 新类型定义
- [ ] `From<usize>` 和 `From<Axis> for usize` 实现
- [ ] 辅助方法：`next()`, `prev()`, `wrap()`
- [ ] 单元测试：`test_axis_next_prev`
- [ ] 单元测试：`test_axis_wrap_negative`

---

### 任务 10：实现 Sealed trait

**名称**：实现 Sealed trait 防止外部实现

**涉及文件**：
- `src/private.rs`
- `src/dimension/mod.rs`

**前置依赖**：任务 9

**验收标准**：
- [ ] `Sealed` trait 定义
- [ ] 为所有维度类型实现
- [ ] `Dimension` trait 继承 `Sealed`
- [ ] 编译测试：外部类型无法实现 `Dimension`

---

### 任务 11：模块导出和文档

**名称**：完成模块导出和文档注释

**涉及文件**：
- `src/dimension/mod.rs`
- `src/lib.rs`

**前置依赖**：任务 10

**验收标准**：
- [ ] 所有公开类型正确导出
- [ ] 所有 pub 项有文档注释
- [ ] `#![warn(missing_docs)]` 通过
- [ ] `cargo doc` 无警告

---

### 任务 12：集成测试和边界测试

**名称**：编写集成测试和边界用例

**涉及文件**：
- `tests/dimension_tests.rs`

**前置依赖**：任务 11

**验收标准**：
- [ ] 测试空维度（Ix0）
- [ ] 测试单元素（Ix1(1)）
- [ ] 测试大维度（Ix6 大值）
- [ ] 测试维度溢出（size 超过 usize::MAX）
- [ ] 测试 no_std 兼容性

---

## 12. 设计决策记录

### 决策 1：最大维度数为 6

**背景**：需要确定支持的最大维度数。

**选项**：
1. 无限制（仅受 `IxDyn` 内存限制）
2. 固定为 6
3. 可配置（宏或 const generic）

**选择**：固定为 6

**理由**：
- 科学计算中超过 6 维的情况罕见
- 固定数量允许编译期优化
- 与 ndarray 的设计一致
- 减少代码复杂度

---

### 决策 2：IxDyn 使用 Vec 而非 SmallVec

**背景**：动态维度的内部表示选择。

**选项**：
1. `Vec<usize>`
2. `SmallVec<[usize; 6]>`
3. `ArrayVec<usize, 6>`

**选择**：`Vec<usize>`（当前），SmallVec 作为未来优化

**理由**：
- 保持简单，减少依赖
- 大多数场景下维度数 ≤ 6，堆分配开销可接受
- 未来可通过 feature gate 引入 SmallVec 优化
- API 保持不变

---

### 决策 3：Ix0 的 size() 返回 1

**背景**：零维数组的元素数定义。

**选项**：
1. `size() == 1`（标量语义）
2. `size() == 0`（空数组语义）

**选择**：`size() == 1`

**理由**：
- 数学上零维数组是标量，有且仅有一个元素
- 与 ndarray 和 NumPy 一致
- 允许 `Tensor<A, Ix0>` 表示单个值
- 广播时正确处理

---

### 决策 4：Dimension trait 继承 Sealed

**背景**：是否允许外部类型实现 Dimension。

**选项**：
1. 开放（允许外部实现）
2. Sealed（禁止外部实现）

**选择**：Sealed

**理由**：
- 确保 API 稳定性
- 可以添加新方法而不破坏兼容性
- 保证维度类型行为一致
- Rust 生态的标准做法

---

### 决策 5：步长为有符号类型

**背景**：步长的数值类型选择。

**选择**：步长在 `Dimension` 中返回 `Self`（无符号），在 `Layout` 中转换为 `isize`

**理由**：
- `Dimension` 只存储形状，步长计算结果为无符号
- `Layout` 层处理负步长（翻转视图）
- 分离关注点，保持 `Dimension` 简单

---

### 决策 6：静态维度使用元组结构体

**背景**：静态维度的类型表示。

**选项**：
1. 元组结构体 `Ix3(usize, usize, usize)`
2. 数组结构体 `Ix3 { dims: [usize; 3] }`

**选择**：元组结构体

**理由**：
- 允许模式匹配解构：`let Ix3(a, b, c) = dim;`
- 字段访问简洁：`dim.0`, `dim.1`
- 构造简洁：`Ix3(2, 3, 4)`
- 与 ndarray 一致

---

### 决策 7：Axis 为新类型而非别名

**背景**：轴索引的类型表示。

**选项**：
1. 类型别名 `type Axis = usize;`
2. 新类型 `struct Axis(usize);`

**选择**：新类型

**理由**：
- 类型安全：不会与其他 usize 混淆
- 可添加辅助方法
- 文档更清晰
- Display 输出更友好：`Axis(2)` vs `2`

---

## 附录 A：完整 API 参考

### A.1 Dimension trait

```rust
pub trait Dimension: Sealed + Clone + PartialEq + Eq + Debug + Send + Sync + 'static {
    const NDIM: Option<usize>;
    
    fn ndim(&self) -> usize;
    fn slice(&self) -> &[usize];
    fn slice_mut(&mut self) -> &mut [usize];
    fn size(&self) -> usize;
    fn zeros() -> Self;
    fn ones() -> Self;
    fn strides_for_f_order(&self) -> Self;
    fn strides_for_c_order(&self) -> Self;
    fn into_dyn(self) -> IxDyn;
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, DimensionMismatch> where Self: Sized;
    fn from_slice(slice: &[usize]) -> Self where Self: Sized;
    
    fn into_dimension<D>(self) -> Result<D, DimensionMismatch> where D: Dimension;
    fn axis(&self, axis: Axis) -> usize;
    fn set_axis(&mut self, axis: Axis, value: usize);
    fn last_axis(&self) -> usize;
    fn first_axis(&self) -> usize;
    fn contains_zero(&self) -> bool;
    fn iter(&self) -> core::slice::Iter<'_, usize>;
}
```

### A.2 IntoDimension trait

```rust
pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}
```

### A.3 Axis 类型

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axis(pub usize);

impl Axis {
    pub fn new(axis: usize) -> Self;
    pub fn index(self) -> usize;
    pub fn next(self) -> Self;
    pub fn prev(self) -> Option<Self>;
    pub fn is_first(self) -> bool;
    pub fn is_last(self, ndim: usize) -> bool;
    pub fn wrap(self, ndim: usize) -> Self;
}
```

### A.4 错误类型

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionMismatch {
    pub expected: usize,
    pub actual: usize,
}
```

---

## 附录 B：测试用例清单

| 测试名称 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_ix0_size_is_one` | Ix0 的 size() 返回 1 | 高 |
| `test_ix0_ndim_is_zero` | Ix0 的 ndim() 返回 0 | 高 |
| `test_ix3_strides_f_order` | Ix3 的 F-order 步长计算 | 高 |
| `test_ix3_strides_c_order` | Ix3 的 C-order 步长计算 | 高 |
| `test_ixdyn_from_slice` | IxDyn 从切片构造 | 高 |
| `test_static_to_dyn` | 静态→动态转换 | 高 |
| `test_dyn_to_static_success` | 动态→静态成功 | 高 |
| `test_dyn_to_static_failure` | 动态→静态失败 | 高 |
| `test_tuple_into_dimension` | 元组转维度 | 中 |
| `test_axis_wrap_negative` | 负索引包装 | 中 |
| `test_size_overflow` | size 溢出检测 | 低 |

---

*本文档由 Senon 项目维护。如有问题请提交 Issue 或 PR。*
