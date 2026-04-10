# 维度系统模块设计

> 文档编号: 02 | 模块: `src/dimension/` | 阶段: Phase 1
> 前置文档: `00-coding.md`, `01-architecture.md`
> 需求参考: 需求说明书 §3

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 静态维度类型 | `Ix0`-`Ix6` 元组结构体，编译期确定维度数 | 运行时动态维度选择 |
| 动态维度类型 | `IxDyn`（`Vec<usize>`），运行时维度数 | — |
| Dimension trait | 完整的维度操作接口（ndim/slice/size/checked_size/strides_for_f_order/zeros/ones/into_dyn/try_from_dyn） | — |
| IntoDimension trait | 从元组、数组、切片、Vec 构造维度 | 用户自定义维度源 |
| Axis 类型 | 轴标记新类型（index/next/prev/is_first/is_last） | 轴上的切片/迭代操作（由 tensor 方法提供） |
| RemoveAxis trait | 移除指定轴降维（Ix1→Ix0, ..., Ix6→Ix5, IxDyn→IxDyn） | Ix0 不实现（标量无轴可移除） |
| 维度互转 | 静态→动态（总是成功）、动态→静态（需维度匹配） | 隐式维度转换 |
| F-order 步长计算 | `strides_for_f_order()` 返回无符号步长 | C-order 步长（不在范围内） |
| 步长类型 | 维度层仅保存无符号形状；有符号步长由 `layout::Strides<D>` 单独建模 | 负步长语义（由 layout 模块负责） |
| 内存分配 | — | 不负责任何内存分配 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 编译期安全 | 静态维度 `NDIM = Some(N)`，单态化消除虚调用 |
| 栈优先 | `Ix0`-`Ix6` 全部栈分配，`Ix0` 为 ZST |
| 封闭集合 | `Dimension` trait 继承 `Sealed`，禁止外部实现 |
| 最小依赖 | 仅依赖 `error` 模块和 `private::Sealed` |
| no_std 兼容 | `IxDyn` 使用 `alloc::vec::Vec` |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension  ← 当前模块
L2: layout (依赖 dimension)
L3: storage (仅依赖 core/alloc)
L4: tensor (依赖 storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

---

## 2. 文件位置

```
src/dimension/
├── mod.rs             # Dimension trait 定义，模块导出，MAX_DIMENSION 常量
├── static_dims.rs     # Ix0, Ix1, ..., Ix6 静态维度及 Dimension impl
├── dynamic.rs         # IxDyn 动态维度及 Dimension impl
├── into_dimension.rs  # IntoDimension trait 及其实现
└── axes.rs            # Axis 新类型及轴操作辅助方法
```

单目录设计：维度类型之间高度相关（互转、公共 trait），集中管理减少耦合复杂度。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
src/dimension/
├── crate::error       # XenonError::DimensionMismatch 错误变体
└── crate::private     # Sealed trait（防止外部实现 Dimension）
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `error` | `XenonError::DimensionMismatch`（维度转换失败时返回） |
| `private` | `Sealed`（`Dimension` trait 的 supertrait） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `dimension/` 仅消费 `error` 和 `private`，不被它们反向依赖。
> 被下游模块消费：`layout`（参见 `06-memory.md` §3）、`storage`（参见 `05-storage.md` §3）、`tensor`（参见 `07-tensor.md` §4）、`shape`（参见 `16-shape.md` §4）、`iter`（参见 `10-iterator.md` §4）、`math`（参见 `11-math.md` §4）、`index`（参见 `17-indexing.md` §4）。

---

## 4. 公共 API 设计

### 4.1 Dimension trait 完整定义

```rust
use core::fmt::Debug;
use crate::private::Sealed;
use crate::error::XenonError;

/// Trait for array dimension types.
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// Implementations exist for `Ix0`, `Ix1`, ..., `Ix6` (static dimensions)
/// and `IxDyn` (dynamic dimension).
pub trait Dimension: Sealed + Clone + PartialEq + Eq + Debug + Send + Sync + 'static {
    /// Maximum number of dimensions this type can represent.
    /// Static: `Some(N)`. Dynamic: `None`.
    const NDIM: Option<usize>;

    /// Returns the number of dimensions (rank).
    fn ndim(&self) -> usize;

    /// Returns the shape as a slice of axis lengths.
    fn slice(&self) -> &[usize];

    /// Returns a mutable reference to the shape slice.
    fn slice_mut(&mut self) -> &mut [usize];

    /// Computes strides for Fortran-order (column-major) layout.
    ///
    /// Returns strides in element units (not bytes).
    /// First axis has stride 1.
    fn strides_for_f_order(&self) -> Self;

    /// Returns the total number of elements.
    /// For `Ix0`, returns 1 (scalar has one element).
    ///
    /// # Overflow behavior
    ///
    /// Implementations must not silently wrap on overflow.
    /// Construction paths and layout validation must use `checked_size()` before
    /// allocating memory or computing accessible address ranges.
    fn size(&self) -> usize;

    /// Returns the total number of elements, or `None` if the product overflows.
    fn checked_size(&self) -> Option<usize>;

    /// Creates a dimension with all axes set to zero.
    fn zeros() -> Self;

    /// Creates a dimension with all axes set to one.
    fn ones() -> Self;

    /// Converts to dynamic dimension. Always succeeds.
    fn into_dyn(self) -> IxDyn;

    /// Attempts to convert from dynamic dimension.
    /// Returns `XenonError::DimensionMismatch` if ndim doesn't match.
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError>
    where
        Self: Sized;

    /// Creates dimension from a slice.
    /// Panics if slice length doesn't match dimension count.
    fn from_slice(slice: &[usize]) -> Self
    where
        Self: Sized;

    /// Returns the axis length at the given index.
    fn axis(&self, axis: Axis) -> usize {
        self.slice()[axis.0]
    }

    /// Sets the axis length at the given index.
    fn set_axis(&mut self, axis: Axis, value: usize) {
        self.slice_mut()[axis.0] = value;
    }

    /// Returns the last axis length. `Ix0` returns 1.
    fn last_axis(&self) -> usize {
        self.slice().last().copied().unwrap_or(1)
    }

    /// Returns the first axis length. `Ix0` returns 1.
    fn first_axis(&self) -> usize {
        self.slice().first().copied().unwrap_or(1)
    }

    /// Checks if any axis has zero length.
    fn contains_zero(&self) -> bool {
        self.slice().iter().any(|&d| d == 0)
    }

    /// Returns an iterator over axis lengths.
    fn iter(&self) -> core::slice::Iter<'_, usize> {
        self.slice().iter()
    }

    /// Signed strides are modeled separately by `layout::Strides<D>`.
    /// `Dimension` only describes axis lengths and rank, never traversal sign.
}
```

> **设计决策：** Xenon 仅提供 `strides_for_f_order()`，不提供 `strides_for_c_order()`。
> C-order 布局不在当前范围内（参见需求说明书 §7）。

### 4.2 静态维度类型 Ix0-Ix6

```rust
/// Maximum supported dimensionality.
pub const MAX_DIMENSION: usize = 6;

/// Zero-dimensional (scalar) dimension. ZST.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix0;

/// One-dimensional dimension.
///
/// `#[repr(C)]` guarantees that the single `usize` field is laid out at offset 0,
/// enabling `slice()` to safely reinterpret `&Ix1` as `&[usize; 1]` via pointer cast.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix1(pub usize);

/// Two-dimensional dimension.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix2(pub usize, pub usize);

/// Three-dimensional dimension.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix3(pub usize, pub usize, pub usize);

/// Four-dimensional dimension.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix4(pub usize, pub usize, pub usize, pub usize);

/// Five-dimensional dimension.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix5(pub usize, pub usize, pub usize, pub usize, pub usize);

/// Six-dimensional dimension.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix6(pub usize, pub usize, pub usize, pub usize, pub usize, pub usize);
```

#### Ix0 特殊语义

| 属性 | 值 | 说明 |
|------|-----|------|
| `NDIM` | `Some(0)` | 没有维度 |
| `slice()` | `&[]` | 空切片 |
| `size()` | `1` | 一个元素（标量） |
| `strides_for_f_order()` | `Ix0` | 无步长 |
| 内存大小 | `0` bytes | ZST，编译器完全消除 |

#### Ix1-Ix6 实现模式（以 Ix3 为例）

```rust
impl Dimension for Ix3 {
    const NDIM: Option<usize> = Some(3);

    #[inline]
    fn ndim(&self) -> usize { 3 }

    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: Ix3 is #[repr(C)], so its 3 usize fields are laid out consecutively
        // at offsets 0, size_of::<usize>(), 2*size_of::<usize>() with no padding.
        // Reinterpreting &Ix3 as &[usize; 3] via pointer cast is therefore safe.
        unsafe {
            core::slice::from_raw_parts(self as *const Self as *const usize, 3)
        }
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [usize] {
        // SAFETY: Same as slice() — #[repr(C)] guarantees consecutive layout.
        unsafe {
            core::slice::from_raw_parts_mut(self as *mut Self as *mut usize, 3)
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.checked_size().expect("dimension size overflow")
    }

    #[inline]
    fn checked_size(&self) -> Option<usize> {
        self.0.checked_mul(self.1)?.checked_mul(self.2)
    }

    #[inline]
    fn strides_for_f_order(&self) -> Self {
        Ix3(1, self.0, self.0 * self.1)
    }

    #[inline]
    fn into_dyn(self) -> IxDyn {
        IxDyn::from_slice(&[self.0, self.1, self.2])
    }

    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 3 {
            let s = dyn_dim.slice();
            Ok(Ix3(s[0], s[1], s[2]))
        } else {
            Err(XenonError::DimensionMismatch { expected: 3, actual: dyn_dim.ndim() })
        }
    }

    fn from_slice(slice: &[usize]) -> Self {
        assert_eq!(slice.len(), 3, "Ix3 requires exactly 3 elements");
        Ix3(slice[0], slice[1], slice[2])
    }

    // zeros(), ones() omitted — same pattern
}
```

### 4.3 动态维度 IxDyn

```rust
/// Dynamic dimension type. Dimension count determined at runtime.
/// Supports 0 to `MAX_DIMENSION` (6) dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct IxDyn {
    dims: Vec<usize>,
}

// All IxDyn constructors enforce the MAX_DIMENSION limit:
//
// fn assert_max_dim(dims_len: usize) {
//     assert!(
//         dims_len <= MAX_DIMENSION,
//         "dimension count {} exceeds MAX_DIMENSION ({})",
//         dims_len,
//         MAX_DIMENSION,
//     );
// }
//
// Each constructor (from_slice, from_vec, from_element, ones, zeros)
// calls assert_max_dim(dims.len()) before constructing the IxDyn.

impl IxDyn {
    /// Creates an empty (0-dimensional) dynamic dimension.
    pub fn new() -> Self;

    /// Creates from a slice.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() > MAX_DIMENSION`.
    pub fn from_slice(slice: &[usize]) -> Self;

    /// Creates from a Vec.
    ///
    /// # Panics
    ///
    /// Panics if `dims.len() > MAX_DIMENSION`.
    pub fn from_vec(dims: Vec<usize>) -> Self;

    /// Creates with all axes set to a given value.
    /// Panics if `ndim > MAX_DIMENSION`.
    pub fn from_element(value: usize, ndim: usize) -> Self;

    /// Creates filled with ones.
    ///
    /// # Panics
    ///
    /// Panics if `ndim > MAX_DIMENSION`.
    pub fn ones(ndim: usize) -> Self;

    /// Creates filled with zeros.
    ///
    /// # Panics
    ///
    /// Panics if `ndim > MAX_DIMENSION`.
    pub fn zeros(ndim: usize) -> Self;

    /// Consumes and returns the inner Vec.
    pub fn into_vec(self) -> Vec<usize>;
}

impl Dimension for IxDyn {
    const NDIM: Option<usize> = None; // Dynamic, unknown at compile time

    fn ndim(&self) -> usize { self.dims.len() }
    fn slice(&self) -> &[usize] { &self.dims }
    fn slice_mut(&mut self) -> &mut [usize] { &mut self.dims }
    fn size(&self) -> usize { self.checked_size().expect("dimension size overflow") }

    fn checked_size(&self) -> Option<usize> {
        self.dims.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    }

    fn strides_for_f_order(&self) -> Self {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut stride = 1usize;
        for &dim in &self.dims {
            strides.push(stride);
            stride = stride.checked_mul(dim).expect("f-order stride overflow");
        }
        IxDyn { dims: strides }
    }

    fn into_dyn(self) -> IxDyn { self }

    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        Ok(dyn_dim) // IxDyn always accepts IxDyn
    }

    // ...
}
```

#### no_std 兼容性

```rust
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

### 4.4 IntoDimension trait

```rust
/// Trait for converting types into dimension types.
pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}

// Identity: D -> D
impl<D: Dimension> IntoDimension for D {
    type Dim = D;
    #[inline]
    fn into_dimension(self) -> Self::Dim { self }
}

// Tuples -> Ix1-Ix6
impl IntoDimension for (usize,) { type Dim = Ix1; /* ... */ }
impl IntoDimension for (usize, usize) { type Dim = Ix2; /* ... */ }
impl IntoDimension for (usize, usize, usize) { type Dim = Ix3; /* ... */ }
impl IntoDimension for (usize, usize, usize, usize) { type Dim = Ix4; /* ... */ }
impl IntoDimension for (usize, usize, usize, usize, usize) { type Dim = Ix5; /* ... */ }
impl IntoDimension for (usize, usize, usize, usize, usize, usize) { type Dim = Ix6; /* ... */ }

// Slices -> IxDyn
impl IntoDimension for &[usize] { type Dim = IxDyn; /* ... */ }
impl IntoDimension for Vec<usize> { type Dim = IxDyn; /* ... */ }

// Arrays of rank 0..6 preserve static dimensionality.
impl IntoDimension for [usize; 0] { type Dim = Ix0; /* ... */ }
impl IntoDimension for [usize; 1] { type Dim = Ix1; /* ... */ }
impl IntoDimension for [usize; 2] { type Dim = Ix2; /* ... */ }
impl IntoDimension for [usize; 3] { type Dim = Ix3; /* ... */ }
impl IntoDimension for [usize; 4] { type Dim = Ix4; /* ... */ }
impl IntoDimension for [usize; 5] { type Dim = Ix5; /* ... */ }
impl IntoDimension for [usize; 6] { type Dim = Ix6; /* ... */ }

// Dynamic arrays remain explicitly dynamic via slices/Vec/IxDyn.
```

### 4.5 Axis 新类型

```rust
/// Axis marker type. Provides type safety over raw `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Axis(pub usize);

impl Axis {
    #[inline]
    pub fn new(axis: usize) -> Self { Axis(axis) }

    #[inline]
    pub fn index(self) -> usize { self.0 }

    #[inline]
    pub fn next(self) -> Self { Axis(self.0 + 1) }

    #[inline]
    pub fn prev(self) -> Option<Self> { self.0.checked_sub(1).map(Axis) }

    #[inline]
    pub fn is_first(self) -> bool { self.0 == 0 }

    #[inline]
    pub fn is_last(self, ndim: usize) -> bool { self.0 == ndim.saturating_sub(1) }
}
```

### 4.6 RemoveAxis trait

```rust
/// Trait for dimension types that support removing an axis.
///
/// Implemented for `Ix1`-`Ix6` and `IxDyn`.
/// `Ix0` does NOT implement this trait: a scalar has no axes to remove.
///
/// Used by:
/// - `AxisIter` (see `10-iterator.md §4`): `Item = TensorView<'a, A, D::Smaller>`
/// - `sum_axis` (see `13-reduction.md §4`): result dimension is `D::Smaller`
pub trait RemoveAxis: Dimension {
    /// The dimension type with one fewer axis.
    type Smaller: Dimension;

    /// Remove the specified axis, returning a dimension with one fewer axis.
    ///
    /// # Panics
    ///
    /// Panics if `axis.index() >= self.ndim()`.
    fn remove_axis(&self, axis: Axis) -> Self::Smaller;
}

// Static dimension implementations:
//
// Ix1 -> Ix0
// Ix2 -> Ix1
// Ix3 -> Ix2
// Ix4 -> Ix3
// Ix5 -> Ix4
// Ix6 -> Ix5
// IxDyn -> IxDyn

impl RemoveAxis for Ix1 {
    type Smaller = Ix0;

    fn remove_axis(&self, axis: Axis) -> Ix0 {
        assert!(axis.index() < 1, "axis out of bounds");
        Ix0
    }
}

impl RemoveAxis for Ix2 {
    type Smaller = Ix1;

    fn remove_axis(&self, axis: Axis) -> Ix1 {
        assert!(axis.index() < 2, "axis out of bounds");
        match axis.index() {
            0 => Ix1(self.1),
            _ => Ix1(self.0),
        }
    }
}

// Ix3-Ix6: same pattern — remove the axis, shift remaining fields.
// IxDyn: remove the element at `axis.index()` from the Vec.

impl RemoveAxis for IxDyn {
    type Smaller = IxDyn;

    fn remove_axis(&self, axis: Axis) -> IxDyn {
        assert!(axis.index() < self.ndim(), "axis out of bounds");
        let mut v = self.dims.clone();
        v.remove(axis.index());
        IxDyn { dims: v }
    }
}
```

> **设计决策：** `RemoveAxis` 作为独立 trait 而非 Dimension 的关联类型。
> 这样 `Ix0` 天然不满足 `RemoveAxis` 约束，编译器自动拒绝对标量的轴操作，
> 无需运行时检查。

### 4.7 Sealed trait 策略

```rust
// src/private.rs
pub trait Sealed {}

impl Sealed for Ix0 {}
impl Sealed for Ix1 {}
impl Sealed for Ix2 {}
impl Sealed for Ix3 {}
impl Sealed for Ix4 {}
impl Sealed for Ix5 {}
impl Sealed for Ix6 {}
impl Sealed for IxDyn {}
```

### 4.8 Good / Bad 对比示例

```rust
// Good - unified interface via IntoDimension, clear type inference
fn create_tensor<A, Sh>(shape: Sh) -> Tensor<A, Sh::Dim>
where
    Sh: IntoDimension,
{
    let dim = shape.into_dimension();
    // ...
}
let t = create_tensor::<f64, _>((3, 4));     // Ix2
let t = create_tensor::<f64, _>(&[2, 3, 4]); // IxDyn

// Bad - hardcoded dimension types, not reusable
fn create_tensor_2d<A>(rows: usize, cols: usize) -> Tensor<A, Ix2> { /* ... */ }
fn create_tensor_3d<A>(d1: usize, d2: usize, d3: usize) -> Tensor<A, Ix3> { /* ... */ }
// Every dimension count requires a new function
```

```rust
// Good - use Result for dimension conversion
let dim: Ix3 = Ix3::try_from_dyn(dyn_dim)?;

// Bad - using unwrap, may panic
let dim: Ix3 = Ix3::try_from_dyn(dyn_dim).unwrap();
```

---

### 4.9 BroadcastDim trait（广播层消费）

`BroadcastDim<Other>` 用于编译期计算两个维度类型广播后的输出维度类型。  
该 trait 由广播/运算符重载层消费（参见 `15-broadcast.md` 与 `19-overload.md`），
不属于维度系统的核心职责；`dimension` 模块仅在此记录它依赖静态/动态维度类型这一事实。

> **实现建议：** 跨静态维度的 `BroadcastDim` 实现共计约 57 个（含自身广播 7 个 + 跨静态维度 42 个 + 与 IxDyn 混合 7 个（静态维度→IxDyn）+ 1 个（IxDyn→D 泛型 impl））。
> 建议使用声明宏（`macro_rules!`）生成这些实现，避免手工编写导致的遗漏和错误。

```rust
/// Trait for computing the output dimension type when broadcasting two arrays.
///
/// - `IxN BroadcastDim IxN` → `IxN` (same static dimension)
/// - `IxN BroadcastDim IxDyn` → `IxDyn`
/// - `IxDyn BroadcastDim IxN` → `IxDyn`
/// - `IxDyn BroadcastDim IxDyn` → `IxDyn`
pub trait BroadcastDim<Other: Dimension>: Dimension {
    /// The output dimension type after broadcasting.
    type Output: Dimension;
}

// Same static dimension broadcasts to itself
impl BroadcastDim<Ix0> for Ix0 { type Output = Ix0; }
impl BroadcastDim<Ix1> for Ix1 { type Output = Ix1; }
impl BroadcastDim<Ix2> for Ix2 { type Output = Ix2; }
impl BroadcastDim<Ix3> for Ix3 { type Output = Ix3; }
impl BroadcastDim<Ix4> for Ix4 { type Output = Ix4; }
impl BroadcastDim<Ix5> for Ix5 { type Output = Ix5; }
impl BroadcastDim<Ix6> for Ix6 { type Output = Ix6; }

// Cross-static-dimension broadcast: higher-dimensional type wins.
// NumPy rule: prepend 1s to the shorter shape, then broadcast element-wise.
// The output ndim = max(ndim_a, ndim_b), so the larger static type is Output.
// Runtime compatibility is verified by broadcast_shape() at the call site.
impl BroadcastDim<Ix0> for Ix1 { type Output = Ix1; }
impl BroadcastDim<Ix0> for Ix2 { type Output = Ix2; }
impl BroadcastDim<Ix0> for Ix3 { type Output = Ix3; }
impl BroadcastDim<Ix0> for Ix4 { type Output = Ix4; }
impl BroadcastDim<Ix0> for Ix5 { type Output = Ix5; }
impl BroadcastDim<Ix0> for Ix6 { type Output = Ix6; }

impl BroadcastDim<Ix1> for Ix0 { type Output = Ix1; }
impl BroadcastDim<Ix1> for Ix2 { type Output = Ix2; }
impl BroadcastDim<Ix1> for Ix3 { type Output = Ix3; }
impl BroadcastDim<Ix1> for Ix4 { type Output = Ix4; }
impl BroadcastDim<Ix1> for Ix5 { type Output = Ix5; }
impl BroadcastDim<Ix1> for Ix6 { type Output = Ix6; }

impl BroadcastDim<Ix2> for Ix0 { type Output = Ix2; }
impl BroadcastDim<Ix2> for Ix1 { type Output = Ix2; }
impl BroadcastDim<Ix2> for Ix3 { type Output = Ix3; }
impl BroadcastDim<Ix2> for Ix4 { type Output = Ix4; }
impl BroadcastDim<Ix2> for Ix5 { type Output = Ix5; }
impl BroadcastDim<Ix2> for Ix6 { type Output = Ix6; }

impl BroadcastDim<Ix3> for Ix0 { type Output = Ix3; }
impl BroadcastDim<Ix3> for Ix1 { type Output = Ix3; }
impl BroadcastDim<Ix3> for Ix2 { type Output = Ix3; }
impl BroadcastDim<Ix3> for Ix4 { type Output = Ix4; }
impl BroadcastDim<Ix3> for Ix5 { type Output = Ix5; }
impl BroadcastDim<Ix3> for Ix6 { type Output = Ix6; }

impl BroadcastDim<Ix4> for Ix0 { type Output = Ix4; }
impl BroadcastDim<Ix4> for Ix1 { type Output = Ix4; }
impl BroadcastDim<Ix4> for Ix2 { type Output = Ix4; }
impl BroadcastDim<Ix4> for Ix3 { type Output = Ix4; }
impl BroadcastDim<Ix4> for Ix5 { type Output = Ix5; }
impl BroadcastDim<Ix4> for Ix6 { type Output = Ix6; }

impl BroadcastDim<Ix5> for Ix0 { type Output = Ix5; }
impl BroadcastDim<Ix5> for Ix1 { type Output = Ix5; }
impl BroadcastDim<Ix5> for Ix2 { type Output = Ix5; }
impl BroadcastDim<Ix5> for Ix3 { type Output = Ix5; }
impl BroadcastDim<Ix5> for Ix4 { type Output = Ix5; }
impl BroadcastDim<Ix5> for Ix6 { type Output = Ix6; }

impl BroadcastDim<Ix6> for Ix0 { type Output = Ix6; }
impl BroadcastDim<Ix6> for Ix1 { type Output = Ix6; }
impl BroadcastDim<Ix6> for Ix2 { type Output = Ix6; }
impl BroadcastDim<Ix6> for Ix3 { type Output = Ix6; }
impl BroadcastDim<Ix6> for Ix4 { type Output = Ix6; }
impl BroadcastDim<Ix6> for Ix5 { type Output = Ix6; }

// Any static dimension + IxDyn → IxDyn
impl BroadcastDim<IxDyn> for Ix0 { type Output = IxDyn; }
impl BroadcastDim<IxDyn> for Ix1 { type Output = IxDyn; }
impl BroadcastDim<IxDyn> for Ix2 { type Output = IxDyn; }
impl BroadcastDim<IxDyn> for Ix3 { type Output = IxDyn; }
impl BroadcastDim<IxDyn> for Ix4 { type Output = IxDyn; }
impl BroadcastDim<IxDyn> for Ix5 { type Output = IxDyn; }
impl BroadcastDim<IxDyn> for Ix6 { type Output = IxDyn; }

// IxDyn + any → IxDyn
impl<D: Dimension> BroadcastDim<D> for IxDyn { type Output = IxDyn; }
```

> **设计决策：** 跨静态维度广播（如 `Ix2 + Ix1`）时，输出类型为维度数较大的静态类型（`Ix2`）。
> 这遵循 NumPy 广播规则：低维数组在左侧补 1，结果维度数 = max(ndim_a, ndim_b)。
> 运行时兼容性由 `broadcast_shape()` 在调用处验证。  
> 与 `IxDyn` 混合时始终返回 `IxDyn` 以保证类型安全。

### 4.10 Reverse trait

`Reverse` 用于转置操作，对维度序列进行反转：

```rust
/// Trait for reversing the axis order of a dimension.
///
/// Used by `transpose()` in `shape` (see `16-shape.md §4`).
pub trait Reverse: Dimension {
    /// Reverse the axis order of this dimension.
    fn reverse(self) -> Self;
}

impl Reverse for Ix0 {
    #[inline]
    fn reverse(self) -> Self { self }  // 0D: no-op
}

impl Reverse for Ix1 {
    #[inline]
    fn reverse(self) -> Self { self }  // 1D: no-op
}

impl Reverse for Ix2 {
    #[inline]
    fn reverse(self) -> Self { Ix2(self.1, self.0) }
}

impl Reverse for Ix3 {
    #[inline]
    fn reverse(self) -> Self { Ix3(self.2, self.1, self.0) }
}

// Ix4-Ix6: same pattern
impl Reverse for IxDyn {
    fn reverse(self) -> Self {
        let mut dims = self.into_vec();
        dims.reverse();
        IxDyn::from_vec(dims)
    }
}
```

---

## 5. 内部实现设计

### 5.1 维度互转规则

**静态 → 动态（总是成功）**

| 源类型 | 目标类型 | 结果 |
|--------|----------|------|
| `Ix0` | `IxDyn` | `IxDyn { dims: vec![] }` |
| `Ix1(n)` | `IxDyn` | `IxDyn { dims: vec![n] }` |
| `Ix3(a,b,c)` | `IxDyn` | `IxDyn { dims: vec![a, b, c] }` |

**动态 → 静态（需维度匹配）**

| 源类型 | 条件 | 成功 | 失败 |
|--------|------|------|------|
| `IxDyn` | `ndim == N` | `IxN(...)` | `Err(XenonError::DimensionMismatch)` |

### 5.2 负步长支持说明

维度层保存无符号形状（`usize`），步长计算结果也为无符号。负步长由 `layout` 模块处理（参见 `06-memory.md` §4）：

```
Dimension 层：shape = [3, 4], strides_for_f_order() = Ix2(1, 3)
Layout 层：   strides = [1isize, 3isize]（可翻转为负值表示反向视图）
```

> 维度层关注"形状是什么"，layout 层关注"数据如何排列"。

### 5.3 F-order 步长计算算法

```
strides_for_f_order(shape):
    strides = []
    stride = 1
    for dim in shape:
        strides.append(stride)
        stride = checked_mul(stride, dim)  // overflow -> None / panic in validated paths
    return strides
```

**示例**：`shape = Ix3(2, 3, 4)` → `strides = Ix3(1, 2, 6)`

### 5.4 辅助 trait 实现

静态维度额外实现：`Index<usize>`、`IndexMut<usize>`、`IntoIterator`、`From<(usize, ...)>`。

---

## 6. 与其他模块的交互

### 6.1 接口约定

| 模块 | 使用的 trait/类型 | 用途 |
|------|-------------------|------|
| `layout` | `Dimension` | 计算步长、检查连续性 |
| `storage` | `Dimension` | 关联类型、形状表示 |
| `tensor` | `Dimension` | 泛型参数、形状访问 |
| `shape` | `Dimension`, `IntoDimension` | reshape、transpose |
| `iter` | `Dimension` | 迭代器泛型参数 |
| `math` | `Dimension` | 运算泛型参数 |
| `index` | `Dimension`, `Axis` | 索引操作 |

> 各模块的详细接口约定参见对应设计文档（`05-storage.md` §3、`07-tensor.md` §4、`16-shape.md` §4、`17-indexing.md` §4）。

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建模块文件结构
  - 文件: `src/dimension/mod.rs`, `static_dims.rs`, `dynamic.rs`, `into_dimension.rs`, `axes.rs`
  - 内容: 模块声明、子模块占位、公共导出声明
  - 测试: 编译通过
  - 前置: 无
  - 预计: 5 min

- [ ] **T2**: 定义 `Dimension` trait 骨架
  - 文件: `src/dimension/mod.rs`
  - 内容: `Dimension` trait 定义（所有方法签名）、`MAX_DIMENSION` 常量
  - 测试: 编译通过
  - 前置: T1
  - 预计: 10 min

### Wave 2: 静态维度

- [ ] **T3**: 实现 `Ix0` 零维标量
  - 文件: `src/dimension/static_dims.rs`
  - 内容: `Ix0` 结构体 + `Dimension` impl（`size()=1`, `slice()=&[]`）
  - 测试: `test_ix0_size_is_one`, `test_ix0_ndim_is_zero`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `Ix1`-`Ix2`
  - 文件: `src/dimension/static_dims.rs`
  - 内容: `Ix1`, `Ix2` 结构体 + `Dimension` impl + `Index<usize>` impl
  - 测试: `test_ix1_strides_f_order`, `test_ix2_strides_f_order`
  - 前置: T3
  - 预计: 10 min

- [ ] **T5**: 实现 `Ix3`-`Ix6`
  - 文件: `src/dimension/static_dims.rs`
  - 内容: `Ix3`-`Ix6` 结构体 + `Dimension` impl + `From<tuple>` impl
  - 测试: `test_ix3_size_calculation`, `test_ix6_max_dimensions`
  - 前置: T4
  - 预计: 10 min

### Wave 3: 动态维度与互转

- [ ] **T6**: 实现 `IxDyn` 动态维度
  - 文件: `src/dimension/dynamic.rs`
  - 内容: `IxDyn` 结构体 + `Dimension` impl + 构造方法 + no_std 兼容
  - 测试: `test_ixdyn_from_slice`, `test_ixdyn_strides`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现维度互转 + 错误类型
  - 文件: `src/dimension/static_dims.rs`, `dynamic.rs`, `src/error.rs`
  - 内容: `into_dyn()`, `try_from_dyn()` + `XenonError::DimensionMismatch` 错误
  - 测试: `test_static_to_dyn`, `test_dyn_to_static_success`, `test_dyn_to_static_failure`
  - 前置: T5, T6
  - 预计: 10 min

### Wave 4: 辅助 trait

- [ ] **T8**: 实现 `IntoDimension` trait
  - 文件: `src/dimension/into_dimension.rs`
  - 内容: trait 定义 + tuple/array/slice/Vec 实现
  - 测试: `test_tuple_to_ix3`, `test_slice_to_ixdyn`
  - 前置: T7
  - 预计: 10 min

- [ ] **T9**: 实现 `Axis` 类型
  - 文件: `src/dimension/axes.rs`
  - 内容: `Axis` 新类型 + From/Display + 辅助方法
  - 测试: `test_axis_next_prev`, `test_axis_is_first_last`
  - 前置: T1
  - 预计: 10 min

### Wave 5: 集成完善

- [ ] **T10**: 实现 `Sealed` trait + 模块导出
  - 文件: `src/private.rs`, `src/dimension/mod.rs`
  - 内容: `Sealed` trait 定义 + 所有类型实现 + 公共导出完善
  - 测试: 外部类型无法实现 `Dimension`（编译测试）
  - 前置: T7, T8, T9
  - 预计: 10 min

- [ ] **T11**: 文档注释与 `cargo doc` 验证
  - 文件: 所有 `src/dimension/` 文件
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T10
  - 预计: 10 min

- [ ] **T12**: 集成测试与边界测试
  - 文件: `tests/dimension_tests.rs`
  - 内容: 空维度、单元素、大维度、size 溢出、no_std 兼容性
  - 测试: 见测试计划 §8
  - 前置: T11
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
Wave 2: [T3] → [T4] → [T5]     [T9]
                                    │
                  ┌─────────────────┘
                  ▼
Wave 3:         [T6] → [T7]
                         │
                         ▼
Wave 4:               [T8]
                         │
                         ▼
Wave 5:  [T10] → [T11] → [T12]
```

---

## 8. 测试计划

### 8.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_ix0_size_is_one` | `Ix0.size() == 1` | 高 |
| `test_ix0_ndim_is_zero` | `Ix0.ndim() == 0` | 高 |
| `test_ix0_is_zst` | `size_of::<Ix0>() == 0` | 高 |
| `test_ix1_strides_f_order` | `Ix1(5).strides_for_f_order() == Ix1(1)` | 高 |
| `test_ix2_strides_f_order` | `Ix2(3,4).strides_for_f_order() == Ix2(1,3)` | 高 |
| `test_ix3_strides_f_order` | `Ix3(2,3,4).strides_for_f_order() == Ix3(1,2,6)` | 高 |
| `test_ix3_size_calculation` | `Ix3(2,3,4).size() == 24` | 高 |
| `test_ix6_max_dimensions` | `Ix6(1,2,3,4,5,6).size() == 720` | 中 |
| `test_ixdyn_from_slice` | `IxDyn::from_slice(&[2,3])` | 高 |
| `test_ixdyn_strides` | `IxDyn::from_slice(&[2,3,4]).strides_for_f_order()` | 高 |
| `test_static_to_dyn` | `Ix3(2,3,4).into_dyn()` | 高 |
| `test_dyn_to_static_success` | `Ix3::try_from_dyn(IxDyn::from_slice(&[2,3,4]))` | 高 |
| `test_dyn_to_static_failure` | `Ix3::try_from_dyn(IxDyn::from_slice(&[2,3,4,5]))` → Err | 高 |
| `test_tuple_into_dimension` | `(2,3,4).into_dimension()` → `Ix3(2,3,4)` | 中 |
| `test_slice_to_ixdyn` | `(&[2,3,4][..]).into_dimension()` → `IxDyn` | 中 |
| `test_axis_next_prev` | `Axis(2).next() == Axis(3)`, `Axis(0).prev() == None` | 中 |
| `test_axis_is_first_last` | `Axis(0).is_first()`, `Axis(2).is_last(3)` | 中 |
| `test_size_overflow` | 大值维度 `checked_size()` 返回 `None` | 低 |

### 8.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空维度 `Ix0` | `size()=1`, `ndim()=0`, `slice()=&[]` |
| 单元素 `Ix1(1)` | `size()=1` |
| 零长度轴 `Ix2(0, 3)` | `size()=0`, `contains_zero()=true` |
| 大维度 `Ix6(100,100,100,100,100,100)` | `checked_size()` 在溢出时返回 `None` |
| `IxDyn::ones(0)` | 零维动态维度 |

### 8.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `dim.strides_for_f_order().ndim() == dim.ndim()` | 所有维度类型 |
| `dim.into_dyn().try_from_dyn()` 往返一致 | 静态维度 |
| `dim.size() == dim.slice().iter().product()` | 随机形状 |

---

## 9. 设计决策记录

### 决策 1：最大维度数为 6

| 属性 | 值 |
|------|-----|
| 决策 | 固定最大维度数为 6 |
| 理由 | 科学计算中超 6 维罕见；固定数量允许编译期优化；与 ndarray 设计一致 |
| 替代方案 | 无限制（仅受 IxDyn 内存限制）— 放弃，增加复杂度 |
| 替代方案 | 可配置（宏或 const generic）— 放弃，增加维护成本 |

### 决策 2：IxDyn 使用 Vec 而非 SmallVec

| 属性 | 值 |
|------|-----|
| 决策 | 使用 `Vec<usize>`，SmallVec 作为未来优化 |
| 理由 | 保持简单、减少依赖；≤6 维场景堆分配开销可接受；未来可通过 feature gate 引入 |
| 替代方案 | `SmallVec<[usize; 6]>` — 放弃，增加依赖 |
| 替代方案 | `ArrayVec<usize, 6>` — 放弃，无溢出处理 |

### 决策 3：Ix0 的 size() 返回 1

| 属性 | 值 |
|------|-----|
| 决策 | `Ix0.size() == 1`（标量语义） |
| 理由 | 数学上零维数组是标量；与 ndarray/NumPy 一致；允许 `Tensor<A, Ix0>` 表示单值；广播正确处理 |
| 替代方案 | `size() == 0` — 放弃，与标量语义冲突 |

### 决策 4：Dimension trait 继承 Sealed

| 属性 | 值 |
|------|-----|
| 决策 | `Dimension: Sealed + ...`，禁止外部实现 |
| 理由 | API 稳定性（可添加新方法不破坏外部）；类型安全；不变量可控；Rust 生态标准做法 |
| 替代方案 | 开放实现 — 放弃，失去版本控制能力 |

### 决策 5：仅 F-order 步长计算

| 属性 | 值 |
|------|-----|
| 决策 | 仅提供 `strides_for_f_order()`，不提供 `strides_for_c_order()` |
| 理由 | 需求说明书 §7 明确只支持 F-order 布局；减少 API 表面积；C-order 留作未来扩展 |
| 替代方案 | 同时提供两种 — 放弃，超出需求范围 |

### 决策 6：步长在 Dimension 层为无符号

| 属性 | 值 |
|------|-----|
| 决策 | `strides_for_f_order()` 返回 `Self`（无符号），负步长由 layout 层处理 |
| 理由 | 关注点分离：Dimension 关注形状，Layout 关注数据排列 |
| 替代方案 | Dimension 直接返回 `isize` 步长 — 放弃，维度层不应感知负步长 |

### 决策 7：`size()` 与 `checked_size()` 双接口

| 属性 | 值 |
|------|-----|
| 决策 | 保留 `size()` 作为已验证维度的快捷路径，同时提供 `checked_size()` 供构造与布局验证使用 |
| 理由 | 既保留日常查询便利性，又避免在关键安全路径上静默回绕 |
| 风险 | 如果调用方跳过 `checked_size()` 直接在未验证输入上调用 `size()`，仍可能触发 panic |
| 替代方案 | 仅保留 `checked_size()` — 放弃，普通只读查询会变得冗长 |
| 替代方案 | 继续使用静默回绕乘法 — 放弃，安全路径不能接受静默回绕 |

---

## 10. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 栈分配 | `Ix0`-`Ix6` 全部栈分配，无堆开销 |
| ZST 优化 | `Ix0` 是零大小类型，编译器完全消除 |
| 内联 | 所有 `ndim()`, `slice()`, `size()` 标注 `#[inline]` |
| 单态化 | `Dimension` trait 在泛型上下文中单态化，无虚调用开销 |
| checked overflow | 构造与布局验证统一走 `checked_size()`，避免静默回绕 |
| 编译期常量 | `NDIM: Option<usize>` 编译期已知，可优化分支 |

---

## 11. no_std 兼容性

| 组件 | 兼容方案 |
|------|----------|
| `Ix0`-`Ix6` | 纯栈分配，天然 no_std |
| `IxDyn` | 需要 `alloc` crate（`alloc::vec::Vec`） |
| `IntoDimension` | `Vec<usize>` 源需要 alloc；tuple/array 源天然 no_std |
| Feature gate | `#[cfg(feature = "std")]` 使用 `std::vec::Vec`；`#[cfg(not(feature = "std"))]` 使用 `alloc::vec::Vec` |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
