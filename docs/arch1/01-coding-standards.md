# Xenon 编码规范

本文档定义了 Xenon 项目的 Rust 编码约定。Xenon 是一个 N 维数组（张量）库，类似于 Python 的 NumPy。

**项目概述**

- **类型**：单 crate Rust 库
- **目标用户**：库开发者、系统开发者
- **平台支持**：`no_std`（带 alloc），`std` 为默认 feature
- **Feature gates**：`std`（默认）、`parallel`（rayon）、`simd`（pulp）
- **MSRV**：Rust 1.85+
- **许可证**：MIT

---

## 1. 命名规范

### 1.1 模块名

模块名使用 `snake_case`。

```rust
// Good
mod tensor_base;
mod layout;
mod memory_pool;

// Bad
mod tensorBase;
mod TensorBase;
mod tensor-base;
```

### 1.2 类型名

类型名（struct、enum、type alias）使用 `CamelCase`。

```rust
// Good
pub struct TensorBase<S, D> { /* ... */ }
pub struct ViewRepr<'a, A> { /* ... */ }
pub struct ViewMutRepr<'a, A> { /* ... */ }
pub struct ArcRepr<A> { /* ... */ }
pub struct IxDyn { /* ... */ }

// Primary type aliases
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<&'a A>, D>;
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<&'a mut A>, D>;
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;

// Dimension convenience aliases (Tensor shown; same pattern for View/ViewMut/Arc)
pub type Tensor0<A> = Tensor<A, Ix0>;
pub type Tensor1<A> = Tensor<A, Ix1>;
pub type Tensor2<A> = Tensor<A, Ix2>;
pub type Tensor3<A> = Tensor<A, Ix3>;
pub type Tensor4<A> = Tensor<A, Ix4>;
pub type Tensor5<A> = Tensor<A, Ix5>;
pub type Tensor6<A> = Tensor<A, Ix6>;
pub type TensorD<A> = Tensor<A, IxDyn>;

// Bad
pub struct tensor_base<S, D> { /* ... */ }
pub struct TENSORBASE<S, D> { /* ... */ }
```

### 1.3 Trait 名

Trait 名使用 `CamelCase`。对于标记 trait，考虑使用描述性形容词。

```rust
// Good
pub trait Element: Copy + PartialEq + Debug + Display + Send + Sync { /* ... */ }
pub trait Numeric: Element + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self> { /* ... */ }
pub trait RealScalar: Numeric + PartialOrd { /* ... */ }
pub trait Dimension { /* ... */ }
pub trait Storage { /* ... */ }

// Bad
pub trait element { /* ... */ }
pub trait ELEMENT { /* ... */ }
pub trait Has_Element { /* ... */ }
```

### 1.4 函数和方法名

函数和方法名使用 `snake_case`。

```rust
// Good
pub fn shape(&self) -> &[Ix];
pub fn reshape(&self, shape: Shape) -> Result<Tensor<A, D>>;
fn compute_strides(shape: &[Ix]) -> Vec<Ix>;

// Bad
pub fn Shape(&self) -> &[Ix];
pub fn getShape(&self) -> &[Ix];
pub fn computeStrides(shape: &[Ix]) -> Vec<Ix>;
```

### 1.5 常量

常量使用 `SCREAMING_SNAKE_CASE`。

```rust
// Good
pub const MAX_DIMENSION: usize = 12;
pub const DEFAULT_ALIGNMENT: usize = 64;
const INTERNAL_BUFFER_SIZE: usize = 1024;

// Bad
pub const maxDimension: usize = 12;
pub const Max_Dimension: usize = 12;
pub const maxdimension: usize = 12;
```

### 1.6 类型参数

类型参数使用单字母大写。选择有语义的字母。

| 参数 | 含义 |
|------|------|
| `S` | Storage（存储类型） |
| `D` | Dimension（维度类型） |
| `A` | Element type（元素类型，来自 Array） |
| `L` | Layout（布局类型） |
| `T` | 通用类型参数 |
| `E` | Error type（错误类型） |

```rust
// Good
pub struct TensorBase<S, D> {
    storage: S,
    shape: D,
    strides: D,   // signed (isize units), supports negative strides
    offset: usize, // data start offset for slice views
}

pub trait FromShape<A> {
    fn from_shape(shape: Shape) -> Self;
}

// Bad
pub struct TensorBase<Storage, Dimension> { /* ... */ }
pub struct TensorBase<storage, dimension> { /* ... */ }
```

### 1.7 生命周期

生命周期使用短名：`'a`、`'b`、`'c` 等。

```rust
// Good
pub struct View<'a, A, D> {
    data: &'a [A],
    dim: D,
}

impl<'a, A, D> View<'a, A, D> {
    pub fn into_owned(&self) -> Tensor<A, D> { /* ... */ }
}

// Bad
pub struct View<'data, A, D> { /* ... */ }
pub struct View<'DATA, A, D> { /* ... */ }
```

### 1.8 方法前缀约定

遵循 Rust 标准库的命名约定：

| 前缀 | 语义 | 复杂度 | 示例 |
|------|------|--------|------|
| `as_` | 借用转换，O(1)，无分配 | 廉价 | `as_slice()`、`as_ptr()` |
| `to_` | 克隆转换，可能分配 | 可能昂贵 | `to_vec()`、`to_owned()` |
| `into_` | 消耗 self，转换所有权 | 变化 | `into_raw_vec()`、`into_shape()` |
| `is_` | 布尔查询，无副作用 | 廉价 | `is_empty()`、`is_contiguous()` |
| `with_` | 构建器模式，返回 Self | 变化 | `with_strides()`、`with_offset()` |

```rust
// Good
impl<A, D> TensorBase<OwnedRepr<A>, D> {
    /// Borrows the data as a slice. O(1), no allocation.
    pub fn as_slice(&self) -> &[A] { /* ... */ }
    
    /// Converts to a vector. Allocates.
    pub fn to_vec(&self) -> Vec<A> { /* ... */ }
    
    /// Consumes the tensor and returns the underlying vector.
    pub fn into_raw_vec(self) -> Vec<A> { /* ... */ }
    
    /// Checks if the tensor is empty.
    pub fn is_empty(&self) -> bool { /* ... */ }
    
    /// Returns true if the tensor has a contiguous memory layout.
    pub fn is_contiguous(&self) -> bool { /* ... */ }
}

// Bad
pub fn get_slice(&self) -> &[A] { /* ... */ }  // 不需要 get_ 前缀
pub fn to_view(&self) -> View<'_, A, D> { /* ... */ }  // 视图转换应该是 as_
```

### 1.9 Getter 不加 `get_` 前缀

属性访问方法直接使用属性名，不加 `get_` 前缀。

```rust
// Good
pub fn shape(&self) -> &[Ix] { &self.dim.slice() }
pub fn strides(&self) -> &[Ix] { &self.strides.slice() }
pub fn ndim(&self) -> usize { self.dim.ndim() }
pub fn len(&self) -> usize { self.dim.size() }

// Bad
pub fn get_shape(&self) -> &[Ix] { /* ... */ }
pub fn get_strides(&self) -> &[Ix] { /* ... */ }
pub fn get_ndim(&self) -> usize { /* ... */ }
```

**例外**：当方法可能失败或需要参数时，可以使用 `get_` 前缀：

```rust
// Good - 需要 index 参数，可能失败
pub fn get(&self, index: &[Ix]) -> Option<&A> { /* ... */ }

// Good - 可能失败
pub fn get_mut(&mut self, index: &[Ix]) -> Option<&mut A> { /* ... */ }
```

---

## 2. 代码格式

### 2.1 缩进

使用 4 空格缩进，不使用 tab。

```rust
// Good
fn example() {
    let x = {
        let y = 1;
        y + 2
    };
}

// Bad
fn example() {
	let x = {  // tab
		let y = 1;
		y + 2
	};
}
```

### 2.2 rustfmt 配置

项目根目录放置 `rustfmt.toml`：

```toml
edition = "2024"
max_width = 100
tab_spaces = 4
use_small_heuristics = "Default"
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true
reorder_modules = true
trailing_comma = "Vertical"
match_block_trailing_comma = true
use_field_init_shorthand = true
use_try_shorthand = true
```

**行宽限制**：100 字符。超过时优先换行而非缩短变量名。

```rust
// Good
pub fn from_shape_vec_unchecked(
    shape: Shape,
    data: Vec<A>,
) -> Self {
    // ...
}

// Bad
pub fn from_shape_vec_unchecked(shape: Shape, data: Vec<A>) -> Self { /* ... */ }
```

### 2.3 导入分组规则

导入按以下顺序分组，每组之间空一行：

1. `std` / `core` / `alloc`
2. 外部 crate
3. 本 crate 内部模块

```rust
// Good
use alloc::vec::Vec;
use alloc::borrow::Cow;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

#[cfg(feature = "std")]
use std::error::Error;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::dimension::Dimension;
use crate::error::XenonError;
use crate::layout::Layout;
use crate::storage::Storage;

// Bad
use crate::dimension::Dimension;
use alloc::vec::Vec;
use rayon::prelude::*;
use core::ptr::NonNull;
use crate::storage::Storage;
```

**条件编译导入**：保持与对应分组的位置关系。

```rust
use core::ops::{Add, Sub, Mul, Div};

#[cfg(feature = "simd")]
use pulp::Simd;

use crate::element::Element;

#[cfg(feature = "parallel")]
use crate::parallel::ParallelIterator;
```

---

## 3. 类型系统规范

### 3.1 限制 `as` 数值类型转换

公开 API 和常规代码中禁止使用 `as` 进行数值类型转换。使用 `From`/`TryFrom`/`Into` trait。

```rust
// Good
let x: i32 = value.try_into().map_err(|_| ConversionError)?;
let y: f64 = value.into();  // From<i32> for f64

// Bad — in public API or general code
let x: i32 = value as i32;  // 危险：可能截断、改变符号
let y: f64 = value as f64;  // 危险：精度丢失
```

**例外**：以下情况允许 `as`：

```rust
// 1. 与 C FFI 交互
let ptr = slice.as_ptr() as *mut c_void;

// 2. 已验证的索引转换（配合注释）
// SAFETY: len is guaranteed to be < isize::MAX by the allocator
let len = slice.len() as isize;

// 3. 原始指针操作
let offset = ptr as usize;

// 4. 内部 cast 实现（须注释说明安全性）
// CAST-SAFETY: saturating semantics per IEEE 754, matches require-v18 §13.5
let truncated = float_val as i32;
```

### 3.2 泛型约束写法

**内联约束**用于：
- 简单约束（1-2 个 trait）
- 约束仅在类型定义中使用

**`where` 子句**用于：
- 复杂约束（3+ trait）
- 约束涉及关联类型
- 约束较长，影响可读性
- impl 块

```rust
// Good - 简单约束，内联
pub fn sum<A>(&self) -> A
where
    A: Numeric + Add<Output = A>,
{
    // ...
}

// Good - 复杂约束，where 子句
pub fn matmul<A, D1, D2>(
    lhs: &Tensor<A, D1>,
    rhs: &Tensor<A, D2>,
) -> Result<Tensor<A, Ix2>>
where
    A: RealScalar + Mul<Output = A> + Add<Output = A>,
    D1: Dimension<Smaller = D2>,
    D2: Dimension,
{
    // ...
}

// Good - impl 块使用 where
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    pub fn shape(&self) -> &[Ix] {
        self.dim.slice()
    }
}

// Bad - 过于复杂的内联约束
pub fn bad<A: Numeric + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Clone>(&self) -> A {
    // ...
}
```

### 3.3 关联类型 vs 泛型参数

**使用关联类型**当：
- 每个实现只有一个合理的类型选择
- 类型由 trait 逻辑决定
- 简化方法签名

**使用泛型参数**当：
- 同一类型可能有多种实现
- 类型由调用者决定
- 需要多态性

```rust
// Good - 关联类型：维度由存储类型决定
pub trait Storage {
    type Elem;
    type Dim: Dimension;
    
    fn len(&self) -> usize;
}

// Good - 泛型参数：元素类型由调用者决定
pub trait FromShape<A> {
    fn from_shape(shape: Shape) -> Self;
}

// Good - 关联类型：输出类型由运算决定
pub trait BitAnd<Rhs = Self> {
    type Output;
    
    fn bitand(self, rhs: Rhs) -> Self::Output;
}

// Good - 泛型参数：允许多种维度类型
impl<A, D> Tensor<A, D>
where
    D: Dimension,
{
    pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>>
    where
        D2: Dimension,
    {
        // ...
    }
}
```

### 3.4 PhantomData 使用规范

使用 `PhantomData` 表示逻辑上存在但运行时不占用的类型关系。

```rust
// Good - 表示对元素类型的所有权
pub struct View<'a, A, D> {
    ptr: NonNull<A>,
    dim: D,
    strides: D,
    _marker: PhantomData<&'a [A]>,
}

// Good - 表示对 A 的协变性
pub struct OwnedRepr<A> {
    _marker: PhantomData<A>,
}

// Good - 表示 Send/Sync 继承
unsafe impl<A: Send> Send for OwnedRepr<A> {}

// Bad - 不必要的使用
pub struct Bad<A> {
    data: Vec<A>,
    _marker: PhantomData<A>,  // 冗余：Vec<A> 已经包含了 A
}
```

**PhantomData 模式选择**：

| 模式 | 含义 | 协变性 |
|------|------|--------|
| `PhantomData<T>` | 拥有 T | 不变 |
| `PhantomData<&'a T>` | 借用 T | 协变（对 T） |
| `PhantomData<&'a mut T>` | 可变借用 T | 不变 |
| `PhantomData<fn(T) -> T>` | 函数参数/返回 | 协变 |
| `PhantomData<fn() -> T>` | 仅返回 T | 协变 |
| `PhantomData<fn(T) -> ()>` | 仅消费 T | 逆变 |

---

## 4. 错误处理规范

### 4.1 可恢复错误 vs 编程错误

**使用 `Result`**（可恢复错误）：
- 资源不可用（文件、网络）
- 用户输入无效
- 运行时约束违反（形状不匹配、索引越界）

**使用 `panic!`**（编程错误）：
- 前置条件违反（不变量被破坏）
- 逻辑错误（不可能的状态）
- 契约违反（调用者未遵守 API 契约）

```rust
// Good - 可恢复错误
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>> {
    if self.len() != shape.size() {
        return Err(XenonError::InvalidShape {
            from: self.len(),
            to: shape.size(),
        });
    }
    // ...
}

// Good - 编程错误（不变量）
fn compute_offset(&self, index: &[Ix]) -> usize {
    assert_eq!(
        index.len(),
        self.ndim(),
        "index dimension mismatch: expected {}, got {}",
        self.ndim(),
        index.len()
    );
    // ...
}

// Bad - 用 panic 处理可恢复错误
pub fn reshape_bad<D2>(self, shape: D2) -> Tensor<A, D2> {
    if self.len() != shape.size() {
        panic!("size mismatch");  // 应该返回 Result
    }
    // ...
}
```

### 4.2 自定义错误类型

自定义错误类型必须实现 `Display` 和 `Error` trait。错误变体须与需求文档 §16.1 一致。

```rust
use core::fmt;
use alloc::borrow::Cow;

#[cfg(feature = "std")]
use std::error::Error;

/// Unified error type for all Xenon operations.
#[derive(Debug, Clone)]
pub enum XenonError {
    /// Binary operation / zip shapes are incompatible and cannot broadcast.
    ShapeMismatch {
        expected: Cow<'static, [usize]>,
        actual: Cow<'static, [usize]>,
    },
    /// Broadcast rule violated (non-size-1 dimensions differ).
    BroadcastError {
        shape_a: Cow<'static, [usize]>,
        shape_b: Cow<'static, [usize]>,
    },
    /// Contiguous layout required but input is non-contiguous.
    LayoutMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    /// Axis index exceeds the number of dimensions.
    InvalidAxis {
        axis: usize,
        ndim: usize,
    },
    /// Reshape target total size differs from source.
    InvalidShape {
        from: usize,
        to: usize,
    },
    /// Static/dynamic dimension conversion mismatch.
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    /// min/max/argmin/argmax on an empty array.
    EmptyArray,
}

/// Convenience type alias.
pub type Result<T> = core::result::Result<T, XenonError>;

impl fmt::Display for XenonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected [{}], got [{}]",
                    fmt_shape(expected), fmt_shape(actual))
            }
            Self::BroadcastError { shape_a, shape_b } => {
                write!(f, "cannot broadcast [{}] with [{}]",
                    fmt_shape(shape_a), fmt_shape(shape_b))
            }
            Self::LayoutMismatch { expected, actual } => {
                write!(f, "layout mismatch: expected {}, got {}", expected, actual)
            }
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "axis {} out of bounds for {}-dimensional array", axis, ndim)
            }
            Self::InvalidShape { from, to } => {
                write!(f, "cannot reshape {} elements into {}", from, to)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::EmptyArray => {
                write!(f, "operation requires a non-empty array")
            }
        }
    }
}

/// Formats a shape slice as "2, 3, 4" for human-readable output.
fn fmt_shape(s: &[usize]) -> alloc::string::String {
    // implementation omitted for brevity
    todo!()
}

#[cfg(feature = "std")]
impl Error for XenonError {}
```

> **注意**：`IndexOutOfBounds` 使用 panic（checked）/ UB（unchecked），不纳入 `XenonError`。

### 4.3 错误信息包含上下文

错误信息必须包含期望值和实际值。

```rust
// Good
pub fn index(&self, index: &[Ix]) -> Result<&A, IndexError> {
    if index.len() != self.ndim() {
        return Err(IndexError::DimensionMismatch {
            expected: self.ndim(),
            actual: index.len(),
        });
    }
    for (i, (&idx, &dim)) in index.iter().zip(self.shape()).enumerate() {
        if idx >= dim {
            return Err(IndexError::OutOfBounds {
                axis: i,
                index: idx,
                size: dim,
            });
        }
    }
    // ...
}

// Bad - 缺少上下文
pub fn index_bad(&self, index: &[Ix]) -> Result<&A, IndexError> {
    if index.len() != self.ndim() {
        return Err(IndexError::DimensionMismatch);  // 缺少具体值
    }
    // ...
}
```

### 4.4 限制 `unwrap()` / `expect()` 在库代码中

库代码中禁止使用 `unwrap()`。`expect()` 仅允许用于断言已证明的不变量或前置条件，且消息必须说明为何此处不会失败。测试代码不受此限制。

```rust
// Good - 使用 ? 和 Result
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>> {
    if self.len() != shape.size() {
        return Err(XenonError::InvalidShape { from: self.len(), to: shape.size() });
    }
    // ...
}

// Good - expect() 断言已证明的不变量
pub fn split_at(&self, axis: usize, index: usize) -> (TensorView<A, D>, TensorView<A, D>) {
    // axis and index already validated above
    let left = self.slice_axis(axis, ..index)
        .expect("split_at: left slice cannot fail after validation");
    // ...
}

// Good - 使用模式匹配
pub fn get(&self, index: &[Ix]) -> Option<&A> {
    self.offset(index).map(|offset| unsafe {
        // SAFETY: offset is bounds-checked by self.offset()
        self.data.get_unchecked(offset)
    })
}

// Bad - 库代码中使用 unwrap
pub fn sum_bad(&self) -> A {
    let first = self.first().unwrap();  // 禁止
    self.iter().fold(*first, |acc, x| acc + x)
}

// Bad - expect() 未说明不变量
let val = map.get("key").expect("should exist");  // 禁止：未证明为何一定存在

// Allowed - 测试代码
#[cfg(test)]
mod tests {
    #[test]
    fn test_reshape() {
        let arr = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let reshaped = arr.reshape([2, 3]).unwrap();  // 测试中允许
        assert_eq!(reshaped.shape(), &[2, 3]);
    }
}
```

### 4.5 提供 checked + unchecked 变体对

对于性能关键的边界检查操作，提供 checked 和 unchecked 两个版本。

```rust
impl<A, D> TensorBase<OwnedRepr<A>, D>
where
    D: Dimension,
{
    /// Returns a reference to the element at the given index.
    ///
    /// # Errors
    /// Returns `Err` if the index is out of bounds.
    pub fn get(&self, index: &[Ix]) -> Result<&A, IndexError> {
        self.check_index(index)?;
        // SAFETY: index is validated by check_index
        Ok(unsafe { self.get_unchecked(index) })
    }
    
    /// Returns a reference to the element at the given index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `index.len() == self.ndim()`
    /// - `index[i] < self.shape()[i]` for all `i`
    pub unsafe fn get_unchecked(&self, index: &[Ix]) -> &A {
        let offset = self.offset_unchecked(index);
        // SAFETY: caller guarantees offset is valid
        self.data.get_unchecked(offset)
    }
    
    fn check_index(&self, index: &[Ix]) -> Result<(), IndexError> {
        if index.len() != self.ndim() {
            return Err(IndexError::DimensionMismatch {
                expected: self.ndim(),
                actual: index.len(),
            });
        }
        for (i, (&idx, &dim)) in index.iter().zip(self.shape()).enumerate() {
            if idx >= dim {
                return Err(IndexError::OutOfBounds {
                    axis: i,
                    index: idx,
                    size: dim,
                });
            }
        }
        Ok(())
    }
}
```

---

## 5. unsafe 规范

### 5.1 最小化 unsafe 块范围

`unsafe` 块应尽可能小，只包含必要的操作。

```rust
// Good - 最小范围
pub fn get_unchecked(&self, index: usize) -> &A {
    // SAFETY: caller guarantees index < self.len()
    unsafe { &*self.ptr.add(index) }
}

// Bad - 过大范围
pub fn get_unchecked_bad(&self, index: usize) -> &A {
    unsafe {
        // This whole block is unsafe, but only the pointer access needs to be
        let offset = index * 2;  // 不需要 unsafe
        &*self.ptr.add(offset)
    }
}

// Good - 分离安全和 unsafe 部分
pub fn get_unchecked(&self, index: usize) -> &A {
    let offset = index * 2;  // Safe calculation
    // SAFETY: caller guarantees offset is valid
    unsafe { &*self.ptr.add(offset) }
}
```

### 5.2 unsafe fn 必须有 # Safety 文档节

每个 `unsafe fn` 必须在文档中包含 `# Safety` 节，列出所有前提条件。

```rust
/// Creates a tensor from raw components.
///
/// # Safety
/// The caller must ensure that:
/// - `ptr` is valid for `len * size_of::<A>()` bytes and properly aligned
/// - `ptr` points to `len` properly initialized values of type `A`
/// - The memory at `ptr` is not accessed by any other pointer for the
///   lifetime `'a` (except through the returned tensor)
/// - `shape.size() == len`
/// - `strides` are valid for accessing all elements of the shape
pub unsafe fn from_raw_parts<'a>(
    ptr: *mut A,
    len: usize,
    shape: D,
    strides: D,
) -> ViewMut<'a, A, D> {
    // ...
}
```

### 5.3 unsafe 块必须有 // SAFETY: 注释

每个 `unsafe` 块之前或之后必须有 `// SAFETY:` 注释，解释为什么该操作是安全的。

```rust
pub fn set(&mut self, index: &[Ix], value: A) -> Result<(), IndexError> {
    let offset = self.compute_offset(index)?;
    
    // SAFETY: compute_offset returns Ok only if index is in bounds,
    // so offset is a valid index into self.data
    unsafe {
        self.data.as_mut_ptr().add(offset).write(value);
    }
    
    Ok(())
}

// 另一种风格：注释在 unsafe 块内
pub fn set_alt(&mut self, index: &[Ix], value: A) -> Result<(), IndexError> {
    let offset = self.compute_offset(index)?;
    
    unsafe {
        // SAFETY: offset is validated by compute_offset above
        self.data.as_mut_ptr().add(offset).write(value);
    }
    
    Ok(())
}
```

### 5.4 unsafe 封装在安全抽象内部

所有 `unsafe` 代码应封装在安全抽象内部，对外暴露安全 API。

```rust
// 私有模块包含 unsafe 实现
mod raw {
    use core::ptr::NonNull;
    
    /// Raw pointer operations. All functions are unsafe.
    /// This module is not exposed publicly.
    pub unsafe fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize) {
        // SAFETY: caller guarantees src and dst are valid and non-overlapping
        core::ptr::copy_nonoverlapping(src, dst, count);
    }
}

// 公开的 safe API
impl<A> Tensor<A, Ix1> {
    /// Copies elements from `src` to `self`.
    ///
    /// # Panics
    /// Panics if `src.len() != self.len()`.
    pub fn copy_from(&mut self, src: &Self) {
        assert_eq!(self.len(), src.len(), "length mismatch");
        
        // SAFETY: lengths are equal, pointers are valid
        unsafe {
            raw::copy_nonoverlapping(
                src.as_ptr(),
                self.as_mut_ptr(),
                self.len(),
            );
        }
    }
}
```

### 5.5 模块级封装

将 `unsafe` 实现集中在私有模块中，公开模块只暴露安全抽象。

```
src/
├── tensor/
│   ├── mod.rs          # 公开 safe API
│   ├── raw.rs          # 私有 unsafe 原语
│   └── iter.rs         # 安全迭代器
├── storage/
│   ├── mod.rs          # 公开 safe API
│   ├── owned.rs        # 包含 unsafe，但暴露 safe API
│   └── view.rs         # 包含 unsafe，但暴露 safe API
└── lib.rs
```

```rust
// src/storage/owned.rs
// This module contains unsafe code but only exposes safe APIs

use alloc::vec::Vec;
use core::ptr::NonNull;

/// Internal raw storage representation.
/// All direct access to this is unsafe.
pub(crate) struct RawStorage<A> {
    ptr: NonNull<A>,
    len: usize,
    cap: usize,
}

impl<A> RawStorage<A> {
    /// Creates a new empty storage.
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }
    
    /// Returns a pointer to the data.
    /// Safe because we don't expose the pointer to external code.
    pub(crate) fn as_ptr(&self) -> *const A {
        self.ptr.as_ptr()
    }
}

// Public API is completely safe
pub struct OwnedRepr<A> {
    inner: RawStorage<A>,
}
```

---

## 6. 文档规范

### 6.1 所有 pub 项必须有 doc comment

所有 `pub` 项（函数、结构体、trait、模块、常量）必须有文档注释。

```rust
/// An N-dimensional array.
///
/// `TensorBase` is the core data structure representing an N-dimensional array
/// with a generic storage backend and dimension type.
///
/// # Type Parameters
/// - `S`: Storage type (e.g., `OwnedRepr<A>`, `ViewRepr<'a>`)
/// - `D`: Dimension type (e.g., `Ix1`, `Ix2`, `IxDyn`)
pub struct TensorBase<S, D> {
    // ...
}

/// Trait for types that can be stored in a tensor.
///
/// Elements must be `Copy` because tensors may have multiple views
/// to the same data, and we want predictable behavior.
pub trait Element: Copy + Clone + 'static {
    // ...
}

/// The number of dimensions (rank) of the tensor.
pub fn ndim(&self) -> usize {
    self.dim.ndim()
}
```

### 6.2 函数文档结构

函数文档应遵循以下结构：

1. **简述**（第一行）：一句话描述功能
2. **详述**：详细说明行为、用途、注意事项
3. **# Arguments**：参数说明
4. **# Returns**：返回值说明
5. **# Errors**：可能的错误（如果返回 Result）
6. **# Panics**：可能 panic 的情况
7. **# Safety**：unsafe 函数的前提条件
8. **# Examples**：示例代码

```rust
/// Reshapes the tensor to the given dimensions.
///
/// This operation preserves the total number of elements. The data
/// is not copied or moved; only the shape metadata is updated.
///
/// For reshaping with memory reordering, see [`reorder`].
///
/// # Arguments
/// - `shape`: The new shape. Must have the same total size as the current shape.
///
/// # Returns
/// A new tensor with the given shape, or an error if the shapes are incompatible.
///
/// # Errors
/// Returns [`XenonError::InvalidShape`] if `shape.size() != self.len()`.
///
/// # Examples
/// ```rust
/// use xenon::{Tensor, Ix2};
///
/// let arr = Tensor::<i32, _>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
/// let reshaped = arr.reshape(Ix2(3, 2))?;
/// assert_eq!(reshaped.shape(), &[3, 2]);
/// # Ok::<(), xenon::XenonError>(())
/// ```
///
/// [`reorder`]: Self::reorder
/// [`XenonError::InvalidShape`]: XenonError::InvalidShape
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>>
where
    D2: Dimension,
{
    // ...
}
```

### 6.3 示例代码使用 `?` 而非 `unwrap()`

文档示例代码应使用 `?` 运算符处理错误，而非 `unwrap()`。

```rust
// Good
/// # Examples
/// ```rust
/// use xenon::Tensor;
///
/// let arr = Tensor::from_vec(vec![1, 2, 3, 4])?;
/// let sum: i32 = arr.iter().sum();
/// assert_eq!(sum, 10);
/// # Ok::<(), xenon::Error>(())
/// ```

// Bad
/// # Examples
/// ```rust
/// use xenon::Tensor;
///
/// let arr = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();  // 不要这样做
/// ```

对于不返回 Result 的示例，使用隐藏的 main 函数：

```rust
/// # Examples
/// ```
/// use xenon::Tensor;
///
/// let arr = Tensor::<f64, _>::zeros([3, 4]);
/// assert_eq!(arr.shape(), &[3, 4]);
/// ```
```

### 6.4 unsafe 函数的 Safety 节

`unsafe` 函数的文档必须包含 `# Safety` 节，列出所有前提条件。

```rust
/// Writes a value to the element at the given offset.
///
/// # Safety
/// The caller must ensure that:
/// - `offset < self.len()`
/// - The tensor is not accessed concurrently by other threads
///   without synchronization (this is a non-atomic write)
///
/// # Examples
/// ```rust
/// use xenon::Tensor;
///
/// let mut arr = Tensor::<i32, _>::zeros([2, 3]);
///
/// unsafe {
///     arr.write_unchecked(0, 42);
/// }
///
/// assert_eq!(arr[[0, 0]], 42);
/// ```
pub unsafe fn write_unchecked(&mut self, offset: usize, value: A) {
    // ...
}
```

### 6.5 `#![warn(missing_docs)]` 在 lib.rs

`lib.rs` 必须包含 `missing_docs` lint 警告。

```rust
// src/lib.rs

//! Xenon: N-dimensional array library for Rust.
//!
//! Xenon provides efficient N-dimensional arrays (tensors) with support
//! for various storage backends, views, and parallel operations.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// ... modules
```

---

## 7. 测试规范

### 7.1 单元测试

单元测试放在同文件的 `#[cfg(test)] mod tests` 块中。

```rust
// src/tensor.rs

pub struct Tensor<A, D> {
    // ...
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shape_returns_correct_dimensions() {
        let arr = Tensor::<i32, _>::zeros([2, 3, 4]);
        assert_eq!(arr.shape(), &[2, 3, 4]);
    }
    
    #[test]
    fn test_sum_of_empty_tensor_returns_zero() {
        let arr: Tensor<i32, Ix1> = Tensor::zeros([0]);
        assert_eq!(arr.sum(), 0);
    }
    
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_out_of_bounds_panics() {
        let arr = Tensor::from_vec(vec![1, 2, 3]);
        let _ = arr[[3]];  // Should panic
    }
}
```

### 7.2 集成测试

集成测试放在 `tests/` 目录下。

```
tests/
├── common/
│   └── mod.rs          # 共享测试工具
├── tensor_ops.rs       # 张量操作测试
├── broadcasting.rs     # 广播机制测试
└── parallel.rs         # 并行功能测试
```

```rust
// tests/tensor_ops.rs

use xenon::{Tensor, Ix2, Ix3};

mod common;

#[test]
fn test_matmul_2d() {
    let a = Tensor::<f64, _>::from_shape_vec([2, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]).unwrap();
    
    let b = Tensor::<f64, _>::from_shape_vec([3, 2], vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0,
    ]).unwrap();
    
    let c = a.matmul(&b).unwrap();
    
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c[[0, 0]], 58.0);
    assert_eq!(c[[1, 1]], 154.0);
}
```

### 7.3 属性测试

使用 `quickcheck` 或 `proptest` 进行属性测试。

```rust
// tests/property_tests.rs

use quickcheck_macros::quickcheck;
use xenon::{Tensor, Ix1, Ix2};

#[quickcheck]
fn test_reshape_preserves_elements(data: Vec<i32>) -> bool {
    if data.is_empty() {
        return true;  // Skip empty
    }
    
    let n = data.len();
    let arr = Tensor::from_vec(data);
    
    // Reshape to 2D and back
    let rows = n / 2.max(1);
    let cols = n / rows;
    
    if rows * cols != n {
        return true;  // Can't reshape evenly, skip
    }
    
    let reshaped = arr.clone().reshape(Ix2(rows, cols));
    if reshaped.is_err() {
        return true;  // Reshape failed, skip
    }
    
    let back = reshaped.unwrap().reshape(Ix1(n));
    back.is_ok() && back.unwrap().iter().eq(arr.iter())
}

#[quickcheck]
fn test_sum_is_commutative(a: Vec<i32>, b: Vec<i32>) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return true;
    }
    
    let arr_a = Tensor::from_vec(a);
    let arr_b = Tensor::from_vec(b);
    
    let sum_ab = (&arr_a + &arr_b).unwrap();
    let sum_ba = (&arr_b + &arr_a).unwrap();
    
    sum_ab.iter().zip(sum_ba.iter()).all(|(x, y)| x == y)
}
```

### 7.4 测试命名规范

测试函数命名格式：`test_<function>_<scenario>_<expected>`

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_reshape_empty_tensor_succeeds() { /* ... */ }
    
    #[test]
    fn test_reshape_incompatible_size_fails() { /* ... */ }
    
    #[test]
    fn test_index_single_element_returns_value() { /* ... */ }
    
    #[test]
    fn test_index_out_of_bounds_panics() { /* ... */ }
    
    #[test]
    fn test_sum_float_tensor_with_nan_returns_nan() { /* ... */ }
    
    #[test]
    fn test_matmul_incompatible_shapes_fails() { /* ... */ }
}
```

### 7.5 边界用例必须覆盖

测试必须覆盖以下边界情况：

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // 1. 空数组
    #[test]
    fn test_empty_tensor_operations() {
        let empty: Tensor<i32, Ix1> = Tensor::zeros([0]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.sum(), 0);
    }
    
    // 2. 单元素
    #[test]
    fn test_single_element_tensor() {
        let single = Tensor::from_vec(vec![42]);
        assert_eq!(single.len(), 1);
        assert_eq!(single[[0]], 42);
        assert_eq!(single.sum(), 42);
    }
    
    // 3. NaN/Inf
    #[test]
    fn test_nan_handling() {
        let arr = Tensor::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(arr.sum().is_nan());
        assert!(arr.mean().is_nan());
    }
    
    #[test]
    fn test_inf_handling() {
        let arr = Tensor::from_vec(vec![1.0, f64::INFINITY, 3.0]);
        assert!(arr.sum().is_infinite());
    }
    
    // 4. 非连续布局
    #[test]
    fn test_non_contiguous_layout() {
        let arr = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let slice = arr.slice(s![..;2]);  // [1, 3, 5]
        
        assert!(!slice.is_contiguous());
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.to_vec(), vec![1, 3, 5]);
    }
    
    #[test]
    fn test_transpose_non_contiguous() {
        let arr = Tensor::<i32, _>::from_shape_vec([2, 3], vec![
            1, 2, 3,
            4, 5, 6,
        ]).unwrap();
        
        let transposed = arr.t();
        assert!(!transposed.is_contiguous());
        assert_eq!(transposed.shape(), &[3, 2]);
    }
    
    // 5. 高维张量
    #[test]
    fn test_high_dimensional_tensor() {
        let arr = Tensor::<f32, _>::zeros([2, 2, 2, 2, 2, 2]);  // 6-D
        assert_eq!(arr.ndim(), 6);
        assert_eq!(arr.len(), 64);
    }
    
    // 6. 大张量（验证不会栈溢出）
    #[test]
    fn test_large_tensor_allocation() {
        let arr = Tensor::<f64, _>::zeros([1000, 1000]);
        assert_eq!(arr.len(), 1_000_000);
    }
    
    // 7. Subnormal 浮点数
    #[test]
    fn test_subnormal_float_handling() {
        let tiny = f64::MIN_POSITIVE * f64::EPSILON;  // subnormal
        let arr = Tensor::from_vec(vec![tiny, tiny]);
        assert!(arr.sum() > 0.0);  // subnormal arithmetic must not flush to zero
    }
}
```

### 7.6 测试覆盖率与数值精度

**覆盖率要求**：
- 行覆盖率 ≥ 80%（CI 通过 `cargo llvm-cov --fail-under-lines 80` 强制执行）
- unsafe 代码块必须有对应测试
- 每个公开 API 至少一个正向 + 一个负向测试

**数值精度要求**：
- 浮点比较使用相对误差容限，禁止直接 `==` 比较浮点结果

```rust
// Good - 相对误差容限
fn assert_close(a: f64, b: f64, rtol: f64) {
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs()).max(1e-15);
    assert!(diff / max_abs < rtol, "expected {a} ≈ {b}, rtol={rtol}");
}

#[test]
fn test_matmul_precision() {
    // ...
    assert_close(result[[0, 0]], 58.0, 1e-12);
}

// Bad - 直接比较浮点
assert_eq!(result[[0, 0]], 58.0);  // 可能因舍入误差失败
```

### 7.7 归约操作溢出行为

归约操作（sum, prod 等）的溢出行为遵循元素类型语义：
- **整数类型**：debug 模式 panic，release 模式 wrapping（与 Rust 默认一致）
- **浮点类型**：返回 `±Infinity`（IEEE 754 语义）
- **空数组**：返回加法单位元 `zero()`（sum）或乘法单位元 `one()`（prod）

```rust
#[test]
fn test_integer_sum_overflow_debug() {
    let arr = Tensor::from_vec(vec![i32::MAX, 1]);
    // In debug mode: panics on overflow
    // In release mode: wraps around
}

#[test]
fn test_float_sum_overflow() {
    let arr = Tensor::from_vec(vec![f64::MAX, f64::MAX]);
    assert!(arr.sum().is_infinite());
}

#[test]
fn test_empty_sum_returns_zero() {
    let arr: Tensor<f64, Ix1> = Tensor::zeros([0]);
    assert_eq!(arr.sum(), 0.0);
}
```

---

## 8. 性能规范

### 8.1 `#[inline]` 使用

**使用 `#[inline]`**：
- 小函数（1-3 行）
- 泛型函数（必须在调用处实例化）
- 频繁调用的简单方法

**不使用 `#[inline]`**：
- 大函数
- 递归函数
- 很少调用的函数

```rust
// Good - 小函数
#[inline]
pub fn len(&self) -> usize {
    self.dim.size()
}

#[inline]
pub fn is_empty(&self) -> bool {
    self.len() == 0
}

// Good - 泛型函数
#[inline]
pub fn map<B, F>(self, f: F) -> Tensor<B, D>
where
    F: FnMut(A) -> B,
    B: Element,
{
    // ...
}

// Good - 不使用 inline（大函数）
pub fn matmul(&self, other: &Self) -> Result<Tensor<A, Ix2>> {
    // Complex implementation, let compiler decide
    // ...
}
```

### 8.2 `#[inline(always)]` 仅用于关键热路径

只在性能分析确认的热路径上使用 `#[inline(always)]`。

```rust
// Good - 关键热路径，经过性能分析确认
impl<A> Tensor<A, Ix1> {
    /// Gets the element at the given index.
    /// Inlined because this is called in tight loops.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &A {
        // SAFETY: caller guarantees index is valid
        &*self.data.as_ptr().add(index)
    }
}

// Bad - 随意使用
#[inline(always)]  // 不必要
pub fn ndim(&self) -> usize {
    self.dim.ndim()
}
```

### 8.3 避免不必要的分配

优先使用视图和借用，避免不必要的内存分配。

```rust
// Good - 使用视图
pub fn row(&self, index: usize) -> View<'_, A, Ix1> {
    // Returns a view, no allocation
    self.index_axis(Axis(0), index)
}

// Good - 借用而非克隆
pub fn sum(&self) -> A
where
    A: Numeric,
{
    self.iter().fold(A::zero(), |acc, &x| acc + x)
}

// Bad - 不必要的克隆
pub fn row_bad(&self, index: usize) -> Tensor<A, Ix1> {
    // Allocates a new tensor when a view would suffice
    self.index_axis(Axis(0), index).to_owned()
}

// Good - 明确的文档说明分配行为
/// Converts this view into an owned tensor.
///
/// This method allocates new memory and copies the data.
/// Use [`as_slice`] if you only need a reference.
pub fn to_owned(&self) -> Tensor<A, D> {
    // ...
}
```

### 8.4 迭代器优先于索引循环

使用迭代器而非手动索引循环，让编译器优化。

```rust
// Good - 迭代器
pub fn sum(&self) -> A {
    self.iter().fold(A::zero(), |acc, &x| acc + x)
}

// Good - 迭代器链
pub fn sum_positive(&self) -> A
where
    A: Numeric + PartialOrd,
{
    self.iter()
        .filter(|&&x| x > A::zero())
        .fold(A::zero(), |acc, &x| acc + x)
}

// Bad - 手动索引
pub fn sum_bad(&self) -> A {
    let mut total = A::zero();
    for i in 0..self.len() {
        total = total + self[i];  // 每次索引都有边界检查
    }
    total
}

// Good - 必须索引时使用 unchecked（unsafe 块内）
pub fn sum_fast(&self) -> A {
    let mut total = A::zero();
    let ptr = self.as_ptr();
    for i in 0..self.len() {
        // SAFETY: i < self.len()
        total = total + unsafe { *ptr.add(i) };
    }
    total
}
```

### 8.5 Benchmark 使用 criterion

使用 `criterion` 进行性能基准测试。

```rust
// benches/tensor_ops.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xenon::{Tensor, Ix2};

fn bench_matmul(c: &mut Criterion) {
    let a = Tensor::<f64, _>::zeros([128, 256]);
    let b = Tensor::<f64, _>::zeros([256, 64]);
    
    c.bench_function("matmul_128x256_256x64", |bencher| {
        bencher.iter(|| {
            black_box(&a).matmul(black_box(&b))
        });
    });
}

fn bench_sum(c: &mut Criterion) {
    let arr = Tensor::<f64, _>::zeros([1024, 1024]);
    
    c.bench_function("sum_1m_elements", |bencher| {
        bencher.iter(|| {
            black_box(&arr).sum()
        });
    });
}

criterion_group!(benches, bench_matmul, bench_sum);
criterion_main!(benches);
```

---

## 9. Feature gate 规范

### 9.1 Feature 必须是 additive

启用 feature 不能移除功能。所有 feature 组合必须能正确编译。

```toml
# Cargo.toml

[features]
default = ["std"]
std = []                      # Additive: enables std library
parallel = ["dep:rayon"]      # Additive: enables parallel iterators
simd = ["dep:pulp"]           # Additive: enables SIMD

# Bad - non-additive
# no_std = []                 # This removes functionality!
# fast = ["parallel", "simd"] # This is okay, it's additive
```

```rust
// Good - additive features: std adds I/O, Display impls, etc.
#[cfg(feature = "std")]
pub fn write_npy<W: std::io::Write>(&self, writer: W) -> std::io::Result<()> {
    // Only available with std — uses std::io
    // ...
}

// Bad - non-additive
#[cfg(not(feature = "no_std"))]  // Feature should add, not remove
pub fn to_vec(&self) -> Vec<A> {
    // ...
}
```

### 9.2 使用 `dep:` 语法声明可选依赖

使用 `dep:` 语法明确声明可选依赖。

```toml
# Cargo.toml

[dependencies]
# Optional dependencies with dep: syntax
rayon = { version = "1.10", optional = true }
pulp = { version = "0.18", optional = true }

[features]
default = ["std"]
std = []
parallel = ["dep:rayon"]
simd = ["dep:pulp"]
```

### 9.3 `cfg_attr(docsrs, doc(cfg(...)))` 标注条件编译 API

使用 `docsrs` 标注条件编译的 API，使其在文档中正确显示。

```rust
/// Parallel iterator over tensor elements.
///
/// This trait is only available when the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait IntoParallelIterator {
    type Iter: rayon::iter::ParallelIterator<Item = A>;
    
    fn into_par_iter(self) -> Self::Iter;
}

/// SIMD-accelerated operations.
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd {
    use pulp::Simd;
    
    /// Performs SIMD-accelerated element-wise addition.
    pub fn add_simd<S: Simd>(a: &[f32], b: &[f32], out: &mut [f32]) {
        // ...
    }
}
```

在 `Cargo.toml` 中配置 docsrs：

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### 9.4 no_std 兼容

使用 `core` 和 `alloc` 替代 `std`，确保 `no_std` 兼容。

```rust
// src/lib.rs

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::string::String;

// Conditional std imports
#[cfg(feature = "std")]
use std::error::Error;

#[cfg(not(feature = "std"))]
// Use core's Display instead of std::error::Error
use core::fmt::Display;
```

```rust
// src/error.rs

use core::fmt;
use alloc::borrow::Cow;

#[derive(Debug, Clone)]
pub enum XenonError {
    ShapeMismatch {
        expected: Cow<'static, [usize]>,
        actual: Cow<'static, [usize]>,
    },
    // ... other variants (see §4.2)
}

impl fmt::Display for XenonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, actual)
            }
            // ... other variants
        }
    }
}

// Only implement std::error::Error when std is available
#[cfg(feature = "std")]
impl std::error::Error for XenonError {}
```

---

## 10. 版本与兼容性

### 10.1 遵循 SemVer

遵循语义化版本控制（SemVer 2.0）：

- **MAJOR**：不兼容的 API 变更
- **MINOR**：向后兼容的功能新增
- **PATCH**：向后兼容的问题修复

```toml
# Cargo.toml
[package]
version = "0.1.0"  # Initial development version
```

**版本 0.x.x 的特殊规则**：
- `0.0.x`：初始开发，API 可能随时变更
- `0.x.y`（x > 0）：MINOR 版本可能包含破坏性变更

### 10.2 公开 API 变更须更新版本号

任何公开 API 的变更都必须更新版本号。

| 变更类型 | 版本变更 | 示例 |
|----------|----------|------|
| 新增 pub fn | PATCH（0.x）或 MINOR（1.x+） | 添加 `new_method()` |
| 移除 pub fn | MINOR（0.x）或 MAJOR（1.x+） | 移除 `old_method()` |
| 修改 fn 签名 | MINOR（0.x）或 MAJOR（1.x+） | 改变参数类型 |
| 新增 pub type | PATCH（0.x）或 MINOR（1.x+） | 添加 `type Alias = ...` |
| 新增 pub struct 字段 | MINOR（0.x）或 MAJOR（1.x+） | 添加字段到 pub struct |
| 修改 pub struct 字段 | MINOR（0.x）或 MAJOR（1.x+） | 重命名字段 |
| 新增 enum 变体 | PATCH（0.x）或 MINOR（1.x+） | 添加 `Error::NewVariant` |
| 修改 enum 变体 | MINOR（0.x）或 MAJOR（1.x+） | 修改变体字段 |
| 内部实现变更 | PATCH | 优化算法 |

### 10.3 MSRV 变更视为 minor 版本变更

提高 MSRV（最小支持的 Rust 版本）应视为 MINOR 版本变更。

```toml
# Cargo.toml
[package]
rust-version = "1.85"  # MSRV
```

**MSRV 变更策略**：
1. 在 CHANGELOG 中明确说明
2. 提前至少一个 MINOR 版本通知
3. 更新文档中的 MSRV 说明

```markdown
# CHANGELOG.md

## [0.2.0] - 2025-01-15

### Changed
- **MSRV bumped to 1.85** (from 1.75)
  - Required for const generics improvements
  - Users on older Rust versions should use 0.1.x
```

### 10.4 `#[deprecated]` 标注废弃 API

使用 `#[deprecated]` 属性标注废弃的 API，提供迁移指导。

```rust
/// Creates a new tensor filled with zeros.
///
/// # Examples
/// ```rust
/// use xenon::Tensor;
///
/// let arr = Tensor::<f32, _>::zeros([3, 4]);
/// ```
#[deprecated(
    since = "0.2.0",
    note = "Use `Tensor::zeros` directly. This method will be removed in 0.3.0."
)]
pub fn zero<Sh>(shape: Sh) -> Self
where
    Sh: IntoShape,
{
    Self::zeros(shape)
}

/// Creates a new tensor filled with zeros.
///
/// # Migration from `zero()`
/// ```rust
/// // Old (deprecated)
/// # use xenon::Tensor;
/// # #[allow(deprecated)]
/// let arr = Tensor::<f32, _>::zero([3, 4]);
///
/// // New
/// let arr = Tensor::<f32, _>::zeros([3, 4]);
/// ```
pub fn zeros<Sh>(shape: Sh) -> Self
where
    Sh: IntoShape,
{
    // ...
}
```

**废弃流程**：

1. **宣布废弃**：添加 `#[deprecated]` 属性，在文档中提供迁移指导
2. **保留一个版本**：废弃的 API 至少保留一个 MINOR 版本
3. **移除**：在下一个 MAJOR 版本（或 0.x 的 MINOR 版本）中移除

```rust
// v0.1.0
pub fn old_method(&self) -> i32 { /* ... */ }

// v0.2.0
#[deprecated(since = "0.2.0", note = "Use new_method instead")]
pub fn old_method(&self) -> i32 { self.new_method() }
pub fn new_method(&self) -> i32 { /* ... */ }

// v0.3.0
// old_method removed
pub fn new_method(&self) -> i32 { /* ... */ }
```

---

## 附录

### A. rustfmt.toml 完整配置

```toml
edition = "2024"
max_width = 100
comment_width = 100
tab_spaces = 4
use_small_heuristics = "Default"
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true
reorder_modules = true
trailing_comma = "Vertical"
match_block_trailing_comma = true
use_field_init_shorthand = true
use_try_shorthand = true
wrap_comments = true
format_code_in_doc_comments = true
normalize_comments = true
```

### B. clippy.toml 配置

```toml
msrv = "1.85"
```

### C. 推荐的 .cargo/config.toml

```toml
[alias]
xtask = "run --package xtask --"

[build]
rustflags = ["-D", "warnings"]

[term]
verbose = false
```

### D. CI 检查清单

```yaml
# .github/workflows/ci.yml
jobs:
  check:
    - cargo fmt -- --check
    - cargo clippy --all-features -- -D warnings
    - cargo test --all-features
    - cargo test --no-default-features  # no_std test
    - cargo doc --all-features
    - cargo llvm-cov --all-features --fail-under-lines 80  # coverage ≥80%
    - cargo +nightly miri test  # detect UB in unsafe code
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.1.0 | 2026-03-28 |
| 1.0.0 | 2026-03-28 |
