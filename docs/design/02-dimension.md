# 维度系统模块设计

> 文档编号: 02 | 模块: `src/dimension/` | 阶段: Phase 1
> 前置文档: `00-coding.md`, `01-architecture.md`
> 需求参考: 需求说明书 §3、§16、§17
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责                | 包含                                                                                         | 不包含                                             |
| ------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 静态维度类型        | `Ix0`-`Ix6` 元组结构体，编译期确定维度数                                                     | 运行时动态维度选择                                 |
| 动态维度类型        | `IxDyn`（`Vec<usize>`），运行时维度数                                                        | —                                                  |
 | Dimension trait     | 维度形状与 rank 接口（ndim/slice/checked/checked_size/into_dyn/try_from_dyn）                   | stride 计算、logical-first pointer、布局标志计算   |
| IntoDimension trait | 从元组、数组、切片、Vec 构造维度                                                             | 用户自定义维度源                                   |
| Axis 类型           | 轴标记新类型（index/checked_next/next/prev/is_first/is_last）                               | 轴上的切片/迭代操作（由 tensor 方法提供）          |
| RemoveAxis trait    | 移除指定轴降维（Ix1→Ix0, ..., Ix6→Ix5, IxDyn→IxDyn）                                         | 不负责把标量轴错误建模为编译期拒绝；零维场景统一走运行时可恢复错误 |
| 维度互转            | 静态→动态（总是成功）、动态→静态（需维度匹配）                                               | 隐式维度转换                                       |
| 形状元数据          | 维度层仅保存无符号形状与 rank，供 layout/tensor 读取                                         | stride 元数据及其合法性判定                        |
| 内存分配            | 为 `IxDyn` 动态维度与维度转换进行少量元数据分配                                               | 不负责张量数据分配                                 |

### 1.2 设计原则

| 原则       | 体现                                          |
| ---------- | --------------------------------------------- |
| 编译期安全 | 静态维度 `NDIM = Some(N)`，单态化消除虚调用   |
| 栈优先     | `Ix0`-`Ix6` 全部栈分配，`Ix0` 为 ZST          |
| 封闭集合   | `Dimension` trait 继承 `Sealed`，禁止外部实现 |
| 最小依赖   | 仅依赖 `error` 模块和 `private::Sealed`       |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`        |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension  ← current module
L2: layout (depends on dimension)
L3: storage (depends only on core/alloc)
L4: tensor (depends on storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                 |
| -------- | -------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §3、§16、§17；其中 `BroadcastDim` 支持 §16 广播语义，`Reverse` 支持 §17 转置语义 |
| 范围内   | 静态/动态维度类型、`Dimension`/`IntoDimension`/`RemoveAxis`、轴元数据 |
| 范围外   | 内存分配、布局标志计算、张量运算、C-order 支持                       |
| 非目标   | 引入开放维度扩展机制、负步长维度模型或新的存储后端                   |

---

## 3. 文件位置

```
src/dimension/
├── mod.rs             # Dimension trait definition, module exports, MAX_DIMENSION constant
├── static_dims.rs     # Ix0, Ix1, ..., Ix6 static dimensions and Dimension impls
├── dynamic.rs         # IxDyn dynamic dimension and Dimension impl
├── into_dimension.rs  # IntoDimension trait and its impls
└── axes.rs            # Axis newtype and axis helper methods
```

单目录设计：维度类型之间高度相关（互转、公共 trait），集中管理减少耦合复杂度。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/dimension/
├── crate::error       # XenonError::DimensionMismatch error variant
└── crate::private     # Sealed trait (prevents external Dimension impls)
```

### 4.2 依赖精确到类型级

| 来源模块  | 使用的类型/trait                                      |
| --------- | ----------------------------------------------------- |
| `error`   | `XenonError::DimensionMismatch`（维度转换失败时返回） |
| `private` | `Sealed`（`Dimension` trait 的 supertrait）           |

### 4.2a 依赖合法性

| 项目           | 结论                               |
| -------------- | ---------------------------------- |
| 新增第三方依赖 | 无                                 |
| 合法性结论     | 符合需求说明书最小依赖限制         |
| 替代方案       | 不适用                             |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `dimension/` 仅消费 `error` 和 `private`，不被它们反向依赖。
> 被下游模块消费：`layout`（参见 `06-layout.md` §5）、`tensor`（参见 `07-tensor.md` §5）、`shape`（参见 `16-shape.md` §5）、`iter`（参见 `10-iterator.md` §5）、`math`（参见 `11-math.md` §5）、`index`（参见 `17-indexing.md` §5）。`storage` 不消费 `Dimension`；它只持有底层连续缓冲区。

---

## 5. 公共 API 设计

### 5.1 Dimension trait 完整定义

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

    /// Returns the total number of elements.
    /// For `Ix0`, returns 1 (scalar has one element).
    /// Implementations must not silently wrap on overflow.
    fn checked_size(&self) -> Result<usize, XenonError>;

    /// Validates that this dimension is representable for public safe APIs.
    ///
    /// This method is the non-allocating validation entry when the caller only
    /// needs to verify shape metadata, not consume the computed element count.
    /// The default contract is equivalent to `self.checked_size().map(|_| ())`.
    fn checked(&self) -> Result<(), XenonError>;

    /// Converts to dynamic dimension. Always succeeds.
    fn into_dyn(self) -> IxDyn;

    /// Attempts to convert from dynamic dimension.
    /// Returns `XenonError::DimensionMismatch` if ndim doesn't match.
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError>
    where
        Self: Sized;

    /// Tries to create a dimension from a slice.
    /// Returns `XenonError::DimensionMismatch` on rank mismatch.
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError>
    where
        Self: Sized;

    /// Returns the axis length at the given index.
    fn axis(&self, axis: Axis) -> Result<usize, XenonError> {
        self.slice().get(axis.0).copied().ok_or(XenonError::InvalidAxis {
            operation: "Dimension::axis".into(),
            axis: axis.index(),
            ndim: self.ndim(),
            shape: self.slice().into(),
        })
    }

    /// Sets the axis length at the given index.
    fn set_axis(&mut self, axis: Axis, value: usize) -> Result<(), XenonError> {
        let ndim = self.ndim();
        let shape = self.slice().to_vec();
        let slot = self.slice_mut().get_mut(axis.0).ok_or(XenonError::InvalidAxis {
            operation: "Dimension::set_axis".into(),
            axis: axis.index(),
            ndim,
            shape,
        })?;
        *slot = value;
        Ok(())
    }

    /// Returns the last valid axis marker.
    ///
    /// For `Ix0`, returns `None` because a scalar has no axis.
    fn last_axis(&self) -> Option<Axis> {
        self.ndim().checked_sub(1).map(Axis)
    }

    /// Returns the first valid axis marker.
    ///
    /// For `Ix0`, returns `None` because a scalar has no axis.
    fn first_axis(&self) -> Option<Axis> {
        (self.ndim() > 0).then_some(Axis(0))
    }

    /// Checks if any axis has zero length.
    fn contains_zero(&self) -> bool {
        self.slice().iter().any(|&d| d == 0)
    }

    /// Returns an iterator over axis lengths.
    fn iter(&self) -> core::slice::Iter<'_, usize> {
        self.slice().iter()
    }
}
```

### 5.2 静态维度类型 Ix0-Ix6

```rust
/// Sentinel upper bound for dynamic rank.
///
/// `usize::MAX` here means there is no artificial upper bound beyond `usize`
/// representability and available memory; it is not a practical max supported rank.
pub const MAX_DIMENSION: usize = usize::MAX;

/// Zero-dimensional (scalar) dimension. ZST.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix0;

/// One-dimensional dimension.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as `&[usize; 1]`
/// via pointer cast; this is only safe because `repr(C)` guarantees the `usize`
/// fields are laid out contiguously starting at offset 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix1(pub usize);

/// Two-dimensional dimension.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as `&[usize; 2]`
/// via pointer cast; this is only safe because `repr(C)` guarantees the `usize`
/// fields are laid out contiguously starting at offset 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix2(pub usize, pub usize);

/// Three-dimensional dimension.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as `&[usize; 3]`
/// via pointer cast; this is only safe because `repr(C)` guarantees the `usize`
/// fields are laid out contiguously starting at offset 0.
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

> **说明：** `MAX_DIMENSION` 是哨兵值，表示动态维度类型 `IxDyn` 不设人工上限，而非表示系统支持的最大维度数。

#### Ix0 特殊语义

| 属性                    | 值        | 说明                |
| ----------------------- | --------- | ------------------- |
| `NDIM`                  | `Some(0)` | 没有维度            |
| `slice()`               | `&[]`     | 空切片              |
| `checked_size()`        | `Ok(1)`   | 一个元素（标量）    |
| 内存大小                | `0` bytes | ZST，编译器完全消除 |

> **Ix0 轴语义警告：** `first_axis()` / `last_axis()` 在本设计中统一返回 `Option<Axis>`；对 `Ix0` 必须返回 `None`，以避免把“无轴”误判为“存在一个长度为 1 的轴”。

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
    fn checked_size(&self) -> Result<usize, XenonError> {
        self.0
            .checked_mul(self.1)
            .and_then(|v| v.checked_mul(self.2))
            .ok_or(XenonError::InvalidShape {
                operation: "Dimension::checked_size".into(),
                shape: self.slice().into(),
                expected_elements: 0,
                actual_elements: 0,
                offending_dim: Some(2),
            })
    }

    #[inline]
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
    }

    #[inline]
    fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0, self.1, self.2])
    }

    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 3 {
            let s = dyn_dim.slice();
            Ok(Ix3(s[0], s[1], s[2]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: "Dimension::try_from_dyn".into(),
                expected: 3,
                actual: dyn_dim.ndim(),
            })
        }
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 3 {
            Ok(Ix3(slice[0], slice[1], slice[2]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: "Dimension::try_from_slice".into(),
                expected: 3,
                actual: slice.len(),
            })
        }
    }

    // Remaining helper methods follow the same pattern.
}
```

### 5.3 动态维度 IxDyn

```rust
/// Dynamic dimension type. Dimension count determined at runtime.
/// Dynamic rank is bounded only by `usize` representability and available memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct IxDyn {
    dims: Vec<usize>,
}

// IxDyn constructors do not impose an artificial rank cap.
// Validation focuses on `usize` representability, allocation success,
// and later shape/size checks in tensor construction paths.

impl IxDyn {
    /// Creates an empty (0-dimensional) dynamic dimension.
    pub fn new() -> Self;

    /// Creates from a slice.
    ///
    pub fn from_slice(slice: &[usize]) -> Self;

    /// Creates from a Vec.
    ///
    pub fn from_vec(dims: Vec<usize>) -> Self;

    /// Creates with all axes set to a given value.
    pub fn from_element(value: usize, ndim: usize) -> Self;

    /// Creates filled with ones.
    ///
    pub fn ones(ndim: usize) -> Self;

    /// Creates filled with zeros.
    ///
    pub fn zeros(ndim: usize) -> Self;

    /// Consumes and returns the inner Vec.
    pub fn into_vec(self) -> Vec<usize>;
}

impl Dimension for IxDyn {
    const NDIM: Option<usize> = None; // Dynamic, unknown at compile time

    fn ndim(&self) -> usize { self.dims.len() }
    fn slice(&self) -> &[usize] { &self.dims }
    fn slice_mut(&mut self) -> &mut [usize] { &mut self.dims }
    fn checked_size(&self) -> Result<usize, XenonError> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim).ok_or(()))
            .map_err(|_| XenonError::InvalidShape {
                operation: "Dimension::checked_size".into(),
                shape: self.dims.clone(),
                expected_elements: 0,
                actual_elements: 0,
                offending_dim: None,
            })
    }

    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
    }

    fn into_dyn(self) -> IxDyn { self }

    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        Ok(dyn_dim) // IxDyn always accepts IxDyn
    }

    // ...
}
```

### 5.4 IntoDimension trait

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

// Tuples -> Ix0-Ix6
impl IntoDimension for () { type Dim = Ix0; /* ... */ }
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

> **固定数组范围说明：** 固定数组 `[usize; N]` 当 `N > 6` 时须先转为切片或 `Vec`，再通过 `IxDyn` 构造。当前版本仅支持 `[usize; 0]` 到 `[usize; 6]` 的直接转换。

### 5.5 Axis 新类型

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
    pub fn checked_next(self) -> Option<Self> { self.0.checked_add(1).map(Axis) }

    #[inline]
    /// Caller must guarantee `self.0 < usize::MAX`.
    ///
    /// This helper is intentionally checked: wraparound is treated as a bug and
    /// panics in all build modes. Use `checked_next()` if overflow is part of the
    /// normal control flow.
    pub fn next(self) -> Self {
        Axis(self.0.checked_add(1).expect("Axis::next overflow"))
    }

    #[inline]
    pub fn prev(self) -> Option<Self> { self.0.checked_sub(1).map(Axis) }

    #[inline]
    pub fn is_first(self) -> bool { self.0 == 0 }

    #[inline]
    pub fn is_last(self, ndim: usize) -> bool { ndim > 0 && self.0 == ndim - 1 }
}
```

> **说明：** 当 `ndim == 0`（标量、无轴）时，`is_last()` 返回 `false`。这一定义避免了把“无轴”误判为“最后一轴”，调用方若在轴语义上区分标量场景，应先检查 `ndim > 0`。

> **补充说明：** `checked_next()` 通过 `checked_add(1)` 返回 `Option<Axis>`，在 `axis == usize::MAX` 时返回 `None`。`next()` 仍保留为快捷方法，但内部同样使用 checked 加法；若调用方未先保证 `axis < usize::MAX`，则统一 panic，而不是在 release 构建中静默回绕。

### 5.6 RemoveAxis trait

```rust
/// Trait for dimension types that support removing an axis.
///
/// Implemented for `Ix0`-`Ix6` and `IxDyn`.
/// For `Ix0`, the operation is still a runtime-recoverable error because a scalar
/// has no removable axis; the public contract must not rely on compile-time rejection.
///
/// Used by:
/// - `AxisIter` (see `10-iterator.md §5.2`): `Item = TensorView<'a, A, D::Smaller>`
/// - `sum_axis` (see `13-reduction.md §5.2`): result dimension is `D::Smaller`
pub trait RemoveAxis: Dimension {
    /// The dimension type with one fewer axis.
    type Smaller: Dimension;

    /// Remove the specified axis, returning a dimension with one fewer axis.
    ///
    /// Returns `XenonError::InvalidAxis` if `axis.index() >= self.ndim()`.
    fn remove_axis(&self, axis: Axis) -> Result<Self::Smaller, XenonError>;
}

// Static dimension implementations:
//
// Ix0 -> runtime error
// Ix1 -> Ix0
// Ix2 -> Ix1
// Ix3 -> Ix2
// Ix4 -> Ix3
// Ix5 -> Ix4
// Ix6 -> Ix5
// IxDyn -> IxDyn

impl RemoveAxis for Ix0 {
    type Smaller = Ix0;

    fn remove_axis(&self, axis: Axis) -> Result<Ix0, XenonError> {
        Err(XenonError::InvalidAxis {
            operation: "RemoveAxis::remove_axis".into(),
            axis: axis.index(),
            ndim: 0,
            shape: self.slice().into(),
        })
    }
}

impl RemoveAxis for Ix1 {
    type Smaller = Ix0;

    fn remove_axis(&self, axis: Axis) -> Result<Ix0, XenonError> {
        if axis.index() < 1 {
            Ok(Ix0)
        } else {
            Err(XenonError::InvalidAxis {
                operation: "RemoveAxis::remove_axis".into(),
                axis: axis.index(),
                ndim: 1,
                shape: self.slice().into(),
            })
        }
    }
}

impl RemoveAxis for Ix2 {
    type Smaller = Ix1;

    fn remove_axis(&self, axis: Axis) -> Result<Ix1, XenonError> {
        match axis.index() {
            0 => Ok(Ix1(self.1)),
            1 => Ok(Ix1(self.0)),
            _ => Err(XenonError::InvalidAxis {
                operation: "RemoveAxis::remove_axis".into(),
                axis: axis.index(),
                ndim: 2,
                shape: self.slice().into(),
            }),
        }
    }
}

// Ix3-Ix6: same pattern — remove the axis, shift remaining fields.
// IxDyn: remove the element at `axis.index()` from the Vec.

impl RemoveAxis for IxDyn {
    type Smaller = IxDyn;

    fn remove_axis(&self, axis: Axis) -> Result<IxDyn, XenonError> {
        if axis.index() >= self.ndim() {
            return Err(XenonError::InvalidAxis {
                operation: "RemoveAxis::remove_axis".into(),
                axis: axis.index(),
                ndim: self.ndim(),
                shape: self.slice().into(),
            });
        }
        let mut v = self.dims.clone();
        v.remove(axis.index());
        Ok(IxDyn { dims: v })
    }
}
```

> **设计决策：** `RemoveAxis` 作为独立 trait 而非 `Dimension` 的关联类型。
> 它负责描述“可移除一条轴后的结果维度类型”，但零维轴操作不应依赖编译期拒绝；
> 标量场景若进入统一轴操作入口，仍须返回运行时可恢复错误。

### 5.7 Sealed trait 策略

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

### 5.8 Good / Bad 对比示例

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

// Bad - unwrap a fallible conversion and turn a recoverable error into a panic
let dim = Ix3::try_from_dyn(IxDyn::from_vec(vec![2, 3, 4, 5, 6])).unwrap();
```

---

### 5.9 BroadcastDim trait（广播层消费）

`BroadcastDim<Other>` 用于编译期计算两个维度类型广播后的输出维度类型。  
该 trait 由广播/运算符重载层消费（参见 `15-broadcast.md` 与 `19-overload.md`），
不属于维度系统的核心职责；`dimension` 模块仅在此记录它依赖静态/动态维度类型这一事实。

> **实现建议：** 跨静态维度的 `BroadcastDim` 实现共计约 57 个（含自身广播 7 个 + 跨静态维度 42 个 + 与 IxDyn 混合 7 个（静态维度→IxDyn）+ 1 个（IxDyn→D 泛型 impl））。
> 建议使用声明宏（`macro_rules!`）生成这些实现，避免手工编写导致的遗漏和错误。宏生成后须通过 compile-fail 测试验证全覆盖。

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

### 5.10 PermuteAxes / Reverse trait

`PermuteAxes` 为转置操作提供通用轴置换语义；`Reverse` 仅作为 `transpose()` 默认轴反转形式的便捷层：

```rust
/// Trait for permuting the axis order of a dimension.
///
/// Used by `transpose()` in `shape` (see `16-shape.md` §5.1).
pub trait PermuteAxes: Dimension {
    /// Applies an explicit axis permutation.
    fn permuted_axes(&self, permutation: &[Axis]) -> Result<Self, XenonError>
    where
        Self: Sized;
}

/// Convenience trait for the default reverse-axis transpose.
pub trait Reverse: Dimension {
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
impl PermuteAxes for IxDyn {
    fn permuted_axes(&self, permutation: &[Axis]) -> Result<Self, XenonError> {
        if permutation.len() != self.ndim() {
            return Err(XenonError::DimensionMismatch {
                operation: "PermuteAxes::permuted_axes".into(),
                expected: self.ndim(),
                actual: permutation.len(),
            });
        }
        let mut seen = vec![false; self.ndim()];
        let mut dims = Vec::with_capacity(self.ndim());
        for &axis in permutation {
            let axis_index = axis.index();
            let slot = seen.get_mut(axis_index).ok_or(XenonError::InvalidAxis {
                operation: "PermuteAxes::permuted_axes".into(),
                axis: axis_index,
                ndim: self.ndim(),
                shape: self.slice().into(),
            })?;
            if *slot {
                return Err(XenonError::InvalidAxis {
                    operation: "PermuteAxes::permuted_axes".into(),
                    axis: axis_index,
                    ndim: self.ndim(),
                    shape: self.slice().into(),
                });
            }
            *slot = true;
            dims.push(self.axis(axis)?);
        }
        if seen.iter().any(|present| !present) {
            return Err(XenonError::InvalidAxis {
                operation: "PermuteAxes::permuted_axes".into(),
                axis: self.ndim(),
                ndim: self.ndim(),
                shape: self.slice().into(),
            });
        }
        Ok(IxDyn::from_vec(dims))
    }
}

impl Reverse for IxDyn {
    fn reverse(self) -> Self {
        let mut dims = self.into_vec();
        dims.reverse();
        IxDyn::from_vec(dims)
    }
}
```

> **范围说明：** 当前版本的形状操作只包含 transpose，但 transpose 语义本身须支持显式轴置换；默认的轴反转是 `transpose()` 的一种特例。参见 `require.md` §17。

> **静态维度补充说明：** 对静态维度 `Ix0`..`Ix6`，`PermuteAxes` 通过编译期常量泛型或宏生成实现；当前版本仅 `IxDyn` 提供完整的运行时轴置换。静态维度的 `transpose` 由 `16-shape.md` 定义，不依赖通用 `PermuteAxes` trait。

---

## 6. 内部实现设计

### 6.1 维度互转规则

**静态 → 动态（总是成功）**

| 源类型       | 目标类型 | 结果                            |
| ------------ | -------- | ------------------------------- |
| `Ix0`        | `IxDyn`  | `IxDyn { dims: vec![] }`        |
| `Ix1(n)`     | `IxDyn`  | `IxDyn { dims: vec![n] }`       |
| `Ix3(a,b,c)` | `IxDyn`  | `IxDyn { dims: vec![a, b, c] }` |

**动态 → 静态（需维度匹配）**

| 源类型  | 条件        | 成功       | 失败                                 |
| ------- | ----------- | ---------- | ------------------------------------ |
| `IxDyn` | `ndim == N` | `IxN(...)` | `Err(XenonError::DimensionMismatch)` |

### 6.2 形状表达边界

维度层保存无符号形状（`usize`）与 rank，仅回答“形状是什么”。步长、连续性与广播零步长等布局语义统一由 `layout` 模块建模和校验（参见 `06-layout.md` §5）。

### 6.3 辅助 trait 实现

静态维度额外实现：`Index<usize>`、`IndexMut<usize>`、`IntoIterator`、`From<(usize, ...)>`。

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
  - 内容: `Ix0` 结构体 + `Dimension` impl（`checked_size()=Ok(1)`, `slice()=&[]`）
  - 测试: `test_ix0_size_is_one`, `test_ix0_ndim_is_zero`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `Ix1`-`Ix2`
  - 文件: `src/dimension/static_dims.rs`
  - 内容: `Ix1`, `Ix2` 结构体 + `Dimension` impl + `Index<usize>` impl
  - 测试: `test_ix1_slice`, `test_ix2_slice`
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
  - 内容: `IxDyn` 结构体 + `Dimension` impl + 构造方法
  - 测试: `test_ixdyn_from_slice`, `test_ixdyn_size`
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
  - 文件: `tests/test_dimension.rs`
  - 内容: 空维度、单元素、大维度、size 溢出
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

### 8.1 测试分类表

| 测试分类 | 位置                                             | 说明                                                                |
| -------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`                         | 验证各维度类型、形状/rank API 和辅助 trait                          |
| 集成测试 | `tests/test_dimension.rs`                        | 验证 `dimension` 与 `tensor`、`layout`、`shape`、`index` 的协同路径 |
| 边界测试 | 同模块测试中标注                                 | 覆盖 Ix0、零长度轴、大维度与溢出路径                                |
| 属性测试 | `tests/test_dimension.rs` 或 `tests/property.rs` | 验证 size/维度互转不变量                                            |

### 8.2 单元测试清单

| 测试函数                     | 测试内容                                                 | 优先级 |
| ---------------------------- | -------------------------------------------------------- | ------ |
| `test_ix0_size_is_one`       | `Ix0.checked_size() == Ok(1)`                            | 高     |
| `test_ix0_ndim_is_zero`      | `Ix0.ndim() == 0`                                        | 高     |
| `test_ix0_is_zst`            | `size_of::<Ix0>() == 0`                                  | 高     |
| `test_ix1_slice`             | `Ix1(5).slice() == &[5]`                                 | 高     |
| `test_ix2_slice`             | `Ix2(3,4).slice() == &[3,4]`                             | 高     |
| `test_ix3_slice`             | `Ix3(2,3,4).slice() == &[2,3,4]`                         | 高     |
| `test_ix3_size_calculation`  | `Ix3(2,3,4).checked_size() == Ok(24)`                    | 高     |
| `test_ix6_max_dimensions`    | `Ix6(1,2,3,4,5,6).checked_size() == Ok(720)`             | 中     |
| `test_ixdyn_from_slice`      | `IxDyn::from_slice(&[2,3])`                              | 高     |
| `test_ixdyn_size`            | `IxDyn::from_slice(&[2,3,4]).checked_size() == Ok(24)`   | 高     |
| `test_static_to_dyn`         | `Ix3(2,3,4).into_dyn()`                                  | 高     |
| `test_dyn_to_static_success` | `Ix3::try_from_dyn(IxDyn::from_slice(&[2,3,4]))`         | 高     |
| `test_dyn_to_static_failure` | `Ix3::try_from_dyn(IxDyn::from_slice(&[2,3,4,5]))` → Err | 高     |
| `test_tuple_into_dimension`  | `(2,3,4).into_dimension()` → `Ix3(2,3,4)`                | 中     |
| `test_slice_to_ixdyn`        | `(&[2,3,4][..]).into_dimension()` → `IxDyn`              | 中     |
| `test_axis_next_prev`        | `Axis(2).next() == Axis(3)`, `Axis(0).prev() == None`    | 中     |
| `test_axis_checked_next`     | `Axis(usize::MAX).checked_next() == None`                | 中     |
| `test_axis_is_first_last`    | `Axis(0).is_first()`, `Axis(2).is_last(3)`               | 中     |
| `test_size_overflow`         | 大值维度 `checked_size()` 返回含 `offending_dim` 的 `XenonError::InvalidShape` | 低 |
| `test_permuted_axes_valid_permutation` | `PermuteAxes` 仅接受 `0..ndim-1` 的双射排列 | 高 |
| `test_permuted_axes_duplicate_axis_error` | 重复轴返回可恢复错误 | 高 |
| `test_permuted_axes_missing_axis_error` | 缺失轴返回可恢复错误 | 高 |

### 8.3 边界测试场景

| 场景                                  | 预期行为                              |
| ------------------------------------- | ------------------------------------- |
| 空维度 `Ix0`                          | `checked_size()=Ok(1)`, `ndim()=0`, `slice()=&[]` |
| 单元素 `Ix1(1)`                       | `checked_size()=Ok(1)`                |
| 零长度轴 `Ix2(0, 3)`                  | `checked_size()=Ok(0)`, `contains_zero()=true` |
| 大维度 `Ix6(100,100,100,100,100,100)` | `checked_size()` 在溢出时返回带 `offending_dim` 的错误 |
| `IxDyn::ones(0)`                      | 零维动态维度                          |
| `PermuteAxes` 重复/缺失轴              | 返回可恢复错误，不接受非双射排列       |

### 8.4 属性测试不变量

| 不变量                                           | 测试方法     |
| ------------------------------------------------ | ------------ |
| `Ix3::try_from_dyn(dim.clone().into_dyn()) == Ok(dim)`    | 静态维度     |
| `dim.checked_size()? == dim.slice().iter().product()`     | 随机形状     |

### 8.5 集成测试

| 测试文件                  | 测试内容                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------- |
| `tests/test_dimension.rs` | `IntoDimension`、`Axis`、`BroadcastDim` 与 `tensor`、`shape`、`index` 的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置项 | 覆盖方式                              | 说明                                         |
| ------ | ------------------------------------- | -------------------------------------------- |
| 默认配置 | 常规单元/集成测试路径                  | 本模块无独立 feature gate，默认配置即主路径  |
| 非默认 feature | 不适用                              | 本模块未引入 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型 | 覆盖方式                                  | 说明                                                 |
| -------- | ----------------------------------------- | ---------------------------------------------------- |
| sealed 边界 | compile-fail 测试外部类型实现 `Dimension` | 验证封闭 trait 边界保持成立                          |
| 维度边界 | 运行时验证零维轴操作返回 `XenonError::InvalidAxis` | 验证标量轴操作走可恢复错误路径                     |
| 静动态边界 | 编译期验证数组/元组输入保持预期 `Dim` 类型  | 验证 `IntoDimension` 不会意外退化为动态维度          |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 模块      | 使用的 trait/类型            | 用途                                         |
| --------- | ---------------------------- | -------------------------------------------- |
| `layout`  | `Dimension`                  | 消费形状/rank 元数据并建模步长、检查连续性   |
| `storage` | —                            | 不直接消费 `Dimension`；仅管理底层连续缓冲区 |
| `tensor`  | `Dimension`                  | 泛型参数、形状访问                           |
| `shape`   | `Dimension`, `IntoDimension` | transpose                                    |
| `iter`    | `Dimension`                  | 迭代器泛型参数                               |
| `math`    | `Dimension`                  | 运算泛型参数                                 |
| `index`   | `Dimension`, `Axis`          | 索引操作                                     |

> 各模块的详细接口约定参见对应设计文档（`05-storage.md` §5、`07-tensor.md` §5、`16-shape.md` §5、`17-indexing.md` §5）。当前版本 `shape` 模块仅覆盖 transpose；reshape 超出当前版本范围。

### 9.2 数据流描述

```text
User provides shape / axis / dimension input
    │
    ├── dimension normalizes it into `Ix0`~`Ix6` or `IxDyn`
    ├── tensor / layout consume `ndim()`, `slice()`, and `checked_size()`
    ├── shape / iter / index build higher-level operations on `Axis`, `RemoveAxis`, `Reverse`, and related traits
    └── static/dynamic conversion failures propagate upward as `XenonError::DimensionMismatch`
```

---

## 10. 错误处理与语义边界

| 项目           | 内容 |
| -------------- | ---- |
| Recoverable error | 动态转静态维度数不匹配时返回 `XenonError::DimensionMismatch { operation, expected, actual }`；`checked()`、`checked_size()`、`try_from_slice()`、`axis()`、`set_axis()`、`remove_axis()` 在失败时统一返回结构化 `XenonError` |
| Panic | 本模块公开设计不再把维度溢出、轴越界或切片长度不匹配建模为 panic；若内部已验证快捷路径保留 unwrap/expect，也不得穿透公开 API |
| 路径一致性 | scalar 路径与普通标量化实现必须保持一致；SIMD：不适用；parallel：不适用 |
| 容差边界 | 不适用 |

---

## 11. 设计决策记录

### 决策 1：静态维度数上限为 6

| 属性     | 值                                                                                                                         |
| -------- | -------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 静态维度类型固定覆盖 `Ix0` 到 `Ix6`；动态维度 `IxDyn` 不施加额外 rank 上限                                                 |
| 理由     | 0-6 维静态类型已经覆盖常见高频路径并保留编译期优化；更高 rank 统一由 `IxDyn` 承担，边界仅受 `usize` 表示能力与可用内存限制 |
| 替代方案 | 所有维度都做成无限静态族 — 放弃，类型数量和实现复杂度不可控                                                                |
| 替代方案 | 可配置（宏或 const generic）— 放弃，增加维护成本                                                                           |

### 决策 2：IxDyn 使用 Vec 而非 SmallVec

| 属性     | 值                                                                          |
| -------- | --------------------------------------------------------------------------- |
| 决策     | 使用 `Vec<usize>`，SmallVec 作为未来优化                                    |
| 理由     | 保持简单、减少依赖；≤6 维场景堆分配开销可接受；未来可通过 feature gate 引入 |
| 替代方案 | `SmallVec<[usize; 6]>` — 放弃，增加依赖                                     |
| 替代方案 | `ArrayVec<usize, 6>` — 放弃，无溢出处理                                     |

### 决策 3：Ix0 的 checked_size() 返回 Ok(1)

| 属性     | 值                                                                                        |
| -------- | ----------------------------------------------------------------------------------------- |
| 决策     | `Ix0.checked_size() == Ok(1)`（标量语义）                                                 |
| 理由     | 数学上零维数组是标量；与 ndarray/NumPy 一致；允许 `Tensor<A, Ix0>` 表示单值；广播正确处理 |
| 替代方案 | 将标量元素总数定义为 `0` — 放弃，与标量语义冲突                                           |

### 决策 4：Dimension trait 继承 Sealed

| 属性     | 值                                                                            |
| -------- | ----------------------------------------------------------------------------- |
| 决策     | `Dimension: Sealed + ...`，禁止外部实现                                       |
| 理由     | API 稳定性（可添加新方法不破坏外部）；类型安全；不变量可控；Rust 生态标准做法 |
| 替代方案 | 开放实现 — 放弃，失去版本控制能力                                             |

### 决策 5：仅保留 checked 公开接口

| 属性     | 值                                                                                     |
| -------- | -------------------------------------------------------------------------------------- |
| 决策     | 将 `checked()` 与 `checked_size()` 作为公开 checked 接口；前者负责验证维度元数据，后者负责返回元素总数 |
| 理由     | 避免公开 API 在维度乘法溢出时 panic；调用方只需验证时可使用 `checked()`，需要元素总数时再调用 `checked_size()` |
| 风险     | 调用方需要显式处理 `Result`，文档与示例必须保持一致 |
| 替代方案 | 继续把未 checked 的旧接口作为公开主接口 — 放弃，公开安全路径不能以 panic 表达可恢复错误                     |
| 替代方案 | 继续使用静默回绕乘法 — 放弃，安全路径不能接受静默回绕                                                        |

---

## 12. 性能考量

| 方面             | 设计决策                                             |
| ---------------- | ---------------------------------------------------- |
| 栈分配           | `Ix0`-`Ix6` 全部栈分配，无堆开销                     |
| ZST 优化         | `Ix0` 是零大小类型，编译器完全消除                   |
| 内联             | 所有 `ndim()`, `slice()`, `checked()`, `checked_size()` 标注 `#[inline]` |
| 单态化           | `Dimension` trait 在泛型上下文中单态化，无虚调用开销 |
| checked overflow | 构造与布局验证统一走 `checked_size()`，避免静默回绕  |
| 编译期常量       | `NDIM: Option<usize>` 编译期已知，可优化分支         |

---

## 13. 平台与工程约束

| 约束       | 说明                                   |
| ---------- | -------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std` |
| MSRV       | Rust 1.85+                            |
| 单 crate   | 保持单 crate 边界                      |
| SemVer     | 公开 API 和维度类型变更遵循 SemVer     |
| 最小依赖   | 无新增第三方依赖                       |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-14 |
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-15 |
| 1.2.4 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
