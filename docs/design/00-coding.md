# 通用编码规范

> 文档编号: 00 | 模块: 全局规范 | 阶段: Phase 0
> 前置文档: 无
> 需求参考: 需求说明书 §1, §4, §7, §10, §27, §28

---

## 1. 命名规范

### 1.1 模块名

模块名使用 `snake_case`。

```rust
// Good
mod tensor_base;
mod layout;
mod shape;

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
pub struct ViewRepr<A> { /* ... */ }
pub struct ViewMutRepr<A> { /* ... */ }
pub struct ArcRepr<A> { /* ... */ }
pub struct IxDyn { /* ... */ }

// Primary type aliases
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<'a, A>, D>;
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<'a, A>, D>;
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

Trait 名使用 `CamelCase`。标记 trait 使用描述性形容词。

```rust
// Good
pub trait Element: Copy + PartialEq + Debug + Display + Send + Sync { /* ... */ }
pub trait Numeric: Element + Add<Output=Self> + Sub<Output=Self>
    + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self> { /* ... */ }
pub trait RealScalar: Numeric + PartialOrd { /* ... */ }
pub trait Dimension { /* ... */ }
pub trait Storage { /* ... */ }

// Bad
pub trait element { /* ... */ }
pub trait ELEMENT { /* ... */ }
pub trait Has_Element { /* ... */ }
```

> **索引类型别名**：`Ix` 为 `usize` 的别名，用于维度值和索引：`pub type Ix = usize;`

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
pub const MAX_DIMENSION: usize = 100;
pub const DEFAULT_ALIGNMENT: usize = 64;
const INTERNAL_BUFFER_SIZE: usize = 1024;

// Bad
pub const maxDimension: usize = 100;
pub const Max_Dimension: usize = 6;
```

### 1.6 类型参数

类型参数使用单字母大写，选择有语义的字母。

| 参数 | 含义 |
|------|------|
| `S` | Storage（存储类型） |
| `D` | Dimension（维度类型） |
| `A` | Element type（元素类型，来自 Array） |
| `L` | Layout（布局类型） |
| `T` | 通用类型参数 |
| `E` | Error type（错误类型） |

### 1.7 生命周期

生命周期使用短名：`'a`、`'b`、`'c` 等。

```rust
// Good
pub struct View<'a, A, D> {
    ptr: NonNull<A>,
    shape: D,
    strides: D,
    _marker: PhantomData<&'a [A]>,
}

// Bad
pub struct View<'data, A, D> { /* ... */ }
pub struct View<'DATA, A, D> { /* ... */ }
```

### 1.8 方法前缀约定

| 前缀 | 语义 | 复杂度 | 示例 |
|------|------|--------|------|
| `as_` | 借用转换，O(1)，无分配 | 廉价 | `as_slice()`、`as_ptr()` |
| `to_` | 克隆转换，可能分配 | 可能昂贵 | `to_vec()`、`to_owned()` |
| `into_` | 消耗 self，转换所有权 | 变化 | `into_raw_vec()`、`into_shape()` |
| `is_` | 布尔查询，无副作用 | 廉价 | `is_empty()`、`is_f_contiguous()` |
| `with_` | 构建器模式，返回 Self | 变化 | `with_strides()`、`with_offset()` |

```rust
// Good
impl<A, D> TensorBase<Owned<A>, D> {
    pub fn as_slice(&self) -> &[A] { /* ... */ }
    pub fn to_vec(&self) -> Vec<A> { /* ... */ }
    pub fn into_raw_vec(self) -> Vec<A> { /* ... */ }
    pub fn is_empty(&self) -> bool { /* ... */ }
    pub fn is_f_contiguous(&self) -> bool { /* ... */ }
}

// Bad
pub fn get_slice(&self) -> &[A] { /* ... */ }  // no need for get_ prefix
pub fn to_view(&self) -> View<'_, A, D> { /* ... */ }  // view conversion should use as_
```

### 1.9 Getter 不加 `get_` 前缀

```rust
// Good
pub fn shape(&self) -> &[Ix] { &self.dim.slice() }
pub fn strides(&self) -> &[Ix] { &self.strides.slice() }
pub fn ndim(&self) -> usize { self.dim.ndim() }
pub fn len(&self) -> usize { self.dim.size() }

// Bad
pub fn get_shape(&self) -> &[Ix] { /* ... */ }
pub fn get_strides(&self) -> &[Ix] { /* ... */ }
```

**例外**：当方法可能失败或需要参数时，可使用 `get_` 前缀：

```rust
// Good - requires index parameter, may fail
pub fn get(&self, index: &[Ix]) -> Option<&A> { /* ... */ }

// Good - may fail
pub fn get_mut(&mut self, index: &[Ix]) -> Option<&mut A> { /* ... */ }
```

---

## 2. 代码格式

### 2.1 缩进

使用 4 空格缩进，不使用 tab。

### 2.2 rustfmt 配置

项目根目录放置 `rustfmt.toml`：

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

---

## 3. 类型系统规范

### 3.1 限制 `as` 数值类型转换

公开 API 和常规代码中禁止使用 `as` 进行数值类型转换。使用 `From`/`TryFrom`/`Into` trait。

```rust
// Good
let x: i32 = value.try_into().map_err(|_| ConversionError)?;
let y: f64 = value.into();  // From<i32> for f64

// Bad — in public API or general code
let x: i32 = value as i32;  // dangerous: may truncate or change sign
let y: f64 = value as f64;  // dangerous: precision loss
```

**例外**：以下情况允许 `as`：

```rust
// 1. Interacting with C FFI
let ptr = slice.as_ptr() as *mut c_void;

// 2. Validated index conversion (with comment)
// SAFETY: len is guaranteed to be < isize::MAX by the allocator
let len = slice.len() as isize;

// 3. Raw pointer operations
let offset = ptr as usize;

// 4. Internal cast implementation (must document safety)
// CAST-SAFETY: saturating semantics per IEEE 754
let truncated = float_val as i32;
```

### 3.2 泛型约束写法

**内联约束**用于：简单约束（1-2 个 trait），约束仅在类型定义中使用。

**`where` 子句**用于：复杂约束（3+ trait），约束涉及关联类型，约束较长影响可读性，impl 块。

```rust
// Good - complex constraint, where clause
pub fn dot<A, D1, D2>(
    lhs: &Tensor<A, D1>,
    rhs: &Tensor<A, D2>,
) -> Result<A>
where
    A: Numeric + Mul<Output = A> + Add<Output = A>,
    D1: Dimension,
    D2: Dimension,
{
    // ...
}

// Good - impl block uses where
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    pub fn shape(&self) -> &[Ix] {
        self.dim.slice()
    }
}

// Bad - overly complex inline constraint
pub fn bad<A: Numeric + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Clone>(&self) -> A {
    // ...
}
```

### 3.3 PhantomData 使用规范

使用 `PhantomData` 表示逻辑上存在但运行时不占用的类型关系。

**PhantomData 模式选择表**：

| 模式 | 含义 | 协变性 |
|------|------|--------|
| `PhantomData<T>` | 拥有 T | 协变（covariant） |
| `PhantomData<&'a T>` | 借用 T | 协变（对 T） |
| `PhantomData<&'a mut T>` | 可变借用 T | 不变（invariant） |
| `PhantomData<fn(T) -> T>` | 函数参数/返回 | 不变 |
| `PhantomData<fn() -> T>` | 仅返回 T | 协变（covariant） |
| `PhantomData<fn(T) -> ()>` | 仅消费 T | 逆变（contravariant） |
| `PhantomData<Cell<T>>` | 如需不变 | 不变（invariant） |

```rust
// Good - covariant borrow of element type
pub struct ViewRepr<A> {
    _marker: PhantomData<A>,
}
// Note: ViewRepr carries `'a` as a dedicated lifetime parameter; the storage type itself is ViewRepr<'a, A>
// with lifetime carried via the generic parameter.

// Good - ownership of A
pub struct Owned<A> {
    _marker: PhantomData<A>,
}

// Bad - redundant: Vec<A> already owns A
pub struct Bad<A> {
    data: Vec<A>,
    _marker: PhantomData<A>,
}
```

### 3.4 Send/Sync 实现规范

按存储模式声明 `unsafe impl Send/Sync`，须严格遵循以下规则（参见 `05-storage.md` §5.7）：

| 存储模式 | Send | Sync | 条件 |
|----------|------|------|------|
| `Owned<A>` | 是 | 是 | Send: `A: Send`，Sync: `A: Sync`（与 `Vec<A>` 一致） |
| `ViewRepr<'a, A>` | 是 | 是 | `A: Sync` |
| `ViewMutRepr<'a, A>` | 是 | **否** | `A: Send`（独占借用不可共享） |
| `ArcRepr<A>` | 是 | 是 | `A: Send + Sync` |

> **关键约束**：`ViewMutRepr` 永远不实现 `Sync`——独占借用语义要求同一时刻只有一个线程可访问。

```rust
// Owned: Send+Sync when A: Send+Sync
unsafe impl<A: Send> Send for Owned<A> {}
unsafe impl<A: Sync> Sync for Owned<A> {}

// ViewRepr: Send+Sync when A: Sync
unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}
unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}

// ViewMutRepr: Send only, NEVER Sync
unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A> {}
// No Sync impl — intentionally omitted

// ArcRepr: Send+Sync when A: Send+Sync
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

---

## 4. 错误处理规范

### 4.1 Result vs panic

**使用 `Result`**（可恢复错误）：
- 运行时约束违反（形状不匹配、广播失败）
- 用户输入无效
- 方法型 API 中可恢复的边界检查失败（例如 `reshape()` / `broadcast_with()` / `try_slice()`）

**使用 `panic!`**（编程错误）：
- 前置条件违反（不变量被破坏）
- 逻辑错误（不可能的状态）
- 契约违反
- 索引语法 `[]` 的越界访问（与标准库 `slice` 行为一致）

```rust
// Good - recoverable error
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>> {
    let target = shape.checked_size().ok_or(XenonError::InvalidShape {
        from: self.len(),
        to: usize::MAX,
    })?;
    if self.len() != target {
        return Err(XenonError::InvalidShape {
            from: self.len(),
            to: target,
        });
    }
    // ...
}

// Good - programming error (invariant)
fn compute_offset(&self, index: &[Ix]) -> usize {
    assert_eq!(
        index.len(), self.ndim(),
        "index dimension mismatch: expected {}, got {}",
        self.ndim(), index.len()
    );
    // ...
}

// Bad - using panic for recoverable error
pub fn reshape_bad<D2>(self, shape: D2) -> Tensor<A, D2> {
    let target = shape.checked_size().expect("reshape_bad: target element count overflow");
    if self.len() != target {
        panic!("size mismatch");  // should return Result
    }
    // ...
}
```

### 4.2 XenonError 设计

统一错误类型 `XenonError`，覆盖所有可恢复错误场景（参见 `26-error.md` §4.2）：

> **注意**：索引越界（IndexOutOfBounds）使用 `panic` 而非 `XenonError`，与标准库 `slice` 行为一致。
> 步长相关错误由 `LayoutMismatch` 变体覆盖。参见 `26-error.md` §1.2。

```rust
#[derive(Debug, Clone)]
pub enum XenonError {
    ShapeMismatch { expected: Cow<'static, [usize]>, actual: Cow<'static, [usize]> },
    BroadcastError { shape_a: Cow<'static, [usize]>, shape_b: Cow<'static, [usize]> },
    LayoutMismatch { expected: &'static str, actual: &'static str },
    InvalidAxis { axis: usize, ndim: usize },
    InvalidShape { from: usize, to: usize },
    DimensionMismatch { expected: usize, actual: usize },
    EmptyArray { operation: &'static str },
}

pub type Result<T> = core::result::Result<T, XenonError>;
```

> **关于 `DimensionMismatch` 的说明**：`XenonError::DimensionMismatch` 是统一错误枚举中的变体，包含 `expected` 和 `actual` 两个 `usize` 字段。全项目统一使用 `XenonError::DimensionMismatch` 作为维度不匹配错误，不使用独立 `DimensionMismatch` 结构体。维度转换（`try_from_dyn`）返回 `Result<Self, XenonError>` 而非独立的 `DimensionMismatch` 类型。

### 4.3 unwrap 限制

库代码中禁止使用 `unwrap()`。`expect()` 仅允许用于断言已证明的不变量或前置条件，且消息必须说明为何此处不会失败。测试代码不受此限制。

```rust
// Good - use ? and Result
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>> {
    let target = shape.checked_size().ok_or(XenonError::InvalidShape {
        from: self.len(),
        to: usize::MAX,
    })?;
    if self.len() != target {
        return Err(XenonError::InvalidShape { from: self.len(), to: target });
    }
    // ...
}

// Good - expect() asserts proven invariant
let left = self.slice_axis(axis, ..index)
    .expect("split_at: left slice cannot fail after validation");

// Bad - using unwrap in library code
let first = self.first().unwrap();  // forbidden

// Allowed - test code
#[cfg(test)]
mod tests {
    #[test]
    fn test_reshape() {
        let arr = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6]);
        let reshaped = arr.reshape([2, 3]).unwrap();  // allowed in tests
    }
}
```

### 4.4 checked + unchecked 变体

对于性能关键的边界检查操作，提供 checked 和 unchecked 两个版本：

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// Checked indexing — returns `None` on out of bounds.
    pub fn get(&self, index: &[Ix]) -> Option<&A> {
        if !self.is_index_valid(index) {
            return None;
        }
        // SAFETY: index is validated above
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Indexing via `[]` operator — panics on out of bounds.
    /// Use `get()` for fallible access returning `Option<&A>`.
    // Note: Index trait implementation delegates to get().expect("index out of bounds")

    /// Unchecked indexing — UB on out of bounds.
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
}
```

---

## 5. unsafe 规范

### 5.1 最小化 unsafe 块范围

```rust
// Good - minimal scope
pub fn get_unchecked(&self, index: usize) -> &A {
    // SAFETY: caller guarantees index < self.len()
    unsafe { &*self.ptr.add(index) }
}

// Bad - scope too large
pub fn get_unchecked_bad(&self, index: usize) -> &A {
    unsafe {
        let offset = index * 2;  // no unsafe needed
        &*self.ptr.add(offset)
    }
}
```

### 5.2 unsafe fn 必须有 # Safety 文档节

每个 `unsafe fn` 必须在文档中包含 `# Safety` 节，列出所有前提条件。

```rust
/// Creates a tensor from raw components.
///
/// # Safety
/// The caller must ensure that:
/// - `ptr` is valid for reads/writes over the region defined by shape and strides
/// - `ptr` is properly aligned to `align_of::<A>()`
/// - The memory at `ptr` is not accessed by any other pointer for the
///   lifetime `'a` (except through the returned tensor)
pub unsafe fn from_raw_parts<'a>(
    ptr: *mut A,
    shape: D,
    strides: D,
    offset: usize,
) -> TensorViewMut<'a, A, D> {
    // ...
}
```

### 5.3 unsafe 块必须有 // SAFETY: 注释

```rust
pub fn set(&mut self, index: &[Ix], value: A) -> Result<()> {
    let offset = self.compute_offset(index)?;

    // SAFETY: compute_offset returns Ok only if index is in bounds
    unsafe {
        self.data.as_mut_ptr().add(offset).write(value);
    }

    Ok(())
}
```

### 5.4 unsafe 封装在安全抽象内部

所有 `unsafe` 代码应封装在安全抽象内部，对外暴露安全 API：

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

---

## 6. 文档规范

### 6.1 #![warn(missing_docs)]

`lib.rs` 必须包含 `missing_docs` lint 警告：

> **CI 强制执行**：开发期间使用 `warn` 级别，CI 通过 `RUSTDOCFLAGS="-D warnings"` 将所有 lint 警告提升为错误，确保文档和代码质量。

```rust
// src/lib.rs

//! Xenon: N-dimensional array library for Rust.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
```

### 6.2 所有 pub 项必须有 doc comment

所有 `pub` 项（函数、结构体、trait、模块、常量）必须有文档注释。

### 6.3 函数文档结构

1. **简述**（第一行）：一句话描述功能
2. **详述**：详细说明行为
3. **# Arguments**：参数说明
4. **# Returns**：返回值说明
5. **# Errors**：可能的错误
6. **# Panics**：可能 panic 的情况
7. **# Safety**：unsafe 函数的前提条件
8. **# Examples**：示例代码

```rust
/// Reshapes the tensor to the given dimensions.
///
/// This operation preserves the total number of elements.
///
/// # Errors
/// Returns [`XenonError::InvalidShape`] if `shape.checked_size()` overflows or
/// if the target element count differs from `self.len()`.
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
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>>
where
    D2: Dimension,
{
    // ...
}
```

### 6.4 示例代码使用 `?` 而非 `unwrap()`

文档示例代码应使用 `?` 运算符处理错误：

```rust
// Good
/// # Examples
/// ```rust
/// use xenon::Tensor;
///
/// let arr = Tensor::from_vec(vec![1, 2, 3, 4])?;
/// # Ok::<(), xenon::XenonError>(())
/// ```

// Bad
/// # Examples
/// ```rust
/// let arr = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();  // do not do this
/// ```
```

---

## 7. 测试规范

### 7.1 测试命名规范

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
    fn test_dot_incompatible_shapes_fails() { /* ... */ }
}
```

### 7.2 边界覆盖必须覆盖

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `iter()` 立即结束 |
| 单元素 `shape=[1, 1]` | `iter()` 产出 1 项 |
| 非连续切片 | `iter()` 正确处理步长跳转 |
| NaN/Inf | 遵循 IEEE 754 语义 |
| 高维 `shape=[2,2,2,2,2,2]` | 正确计算偏移 |
| 大张量 `[1000, 1000]` | 不栈溢出 |
| Subnormal 浮点数 | 不 flush to zero |

### 7.3 测试分类

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法 |
| 集成测试 | `tests/` | 验证跨模块交互 |
| 边界测试 | 集成测试中标注 | 空数组、单元素、NaN/Inf、非连续 |
| 属性测试 | `tests/property/` | 随机生成验证不变量 |

### 7.4 测试覆盖率与数值精度

**覆盖率要求**：
- 行覆盖率 ≥ 80%
- unsafe 代码块必须有对应测试
- 每个公开 API 至少一个正向 + 一个负向测试

**浮点比较**：使用相对误差容限（rtol），禁止直接 `==` 比较浮点结果。

```rust
// Good - relative error tolerance
fn assert_close(a: f64, b: f64, rtol: f64) {
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs()).max(1e-15);
    assert!(diff / max_abs < rtol, "expected {a} ≈ {b}, rtol={rtol}");
}

// Bad - direct float comparison
assert_eq!(result[[0, 0]], 58.0);  // may fail due to rounding errors
```

### 7.5 归约操作溢出行为

| 类型 | 行为 |
|------|------|
| 整数类型 | debug 和 release 均使用 checked 算术，溢出时 panic |
| 浮点类型 | 返回 `±Infinity`（IEEE 754 语义） |
| 空数组 sum | 返回加法单位元 `zero()` |

---

## 8. #[inline] 使用规范

**使用 `#[inline]`**：
- 小函数（1-3 行）
- 泛型函数（必须在调用处实例化）
- 频繁调用的简单方法

**不使用 `#[inline]`**：
- 大函数
- 递归函数
- 很少调用的函数

**`#[inline(always)]`** 仅用于经性能分析确认的关键热路径。

```rust
// Good - small function
#[inline]
pub fn len(&self) -> usize {
    self.dim.size()
}

#[inline]
pub fn is_empty(&self) -> bool {
    self.len() == 0
}

// Good - generic function
#[inline]
pub fn map<B, F>(self, f: F) -> Tensor<B, D>
where
    F: FnMut(A) -> B,
    B: Element,
{
    // ...
}
```

---

## 9. Feature Gate 规范

### 9.1 Feature 必须是 additive

启用 feature 不能移除功能。所有 feature 组合必须能正确编译。

```toml
[features]
default = ["std"]
std = []                        # Additive: enables std library
parallel = ["dep:rayon", "std"] # Additive: enables parallel iterators
simd = ["dep:pulp"]             # Additive: enables SIMD
# Note: libm is NOT a dependency (see 01-architecture.md §1.4).
# RealScalar math functions (sin/cos/exp/ln) are only available with "std" feature.
```

### 9.2 使用 `dep:` 语法声明可选依赖

```toml
[dependencies]
rayon = { version = "1.10", optional = true }
pulp = { version = "0.18", optional = true }
```

### 9.3 `cfg_attr(docsrs, doc(cfg(...)))` 标注条件编译 API

```rust
/// Parallel iterator over tensor elements.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub trait IntoParallelIterator {
    type Iter: rayon::iter::ParallelIterator<Item = A>;
    fn into_par_iter(self) -> Self::Iter;
}
```

在 `Cargo.toml` 中配置 docsrs：

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

### 9.4 no_std 兼容

使用 `core` 和 `alloc` 替代 `std`，确保 `no_std` 兼容（参见 `01-architecture.md` §6）：

```rust
// src/lib.rs
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(not(feature = "std"))]
extern crate alloc;

// src/error.rs
use core::fmt;
use alloc::borrow::Cow;

// Only implement std::error::Error when std is available
#[cfg(feature = "std")]
impl std::error::Error for XenonError {}
```

---

## 10. 实现任务拆分

### Wave 1: 规范基础设施

- [ ] **T1**: 创建 `rustfmt.toml` 配置文件
  - 文件: `rustfmt.toml`
  - 内容: §2.2 中的完整配置
  - 测试: `cargo fmt --check` 通过
  - 前置: 无
  - 预计: 5 min

- [ ] **T2**: 创建 `src/lib.rs` 骨架含 lint 声明
  - 文件: `src/lib.rs`
  - 内容: missing_docs、unsafe_op_in_unsafe_fn、no_std cfg
  - 测试: 编译通过
  - 前置: 无
  - 预计: 5 min

- [ ] **T3**: 创建 `.clippy.toml` 或 `clippy` 配置
  - 文件: `.clippy.toml`
  - 内容: disallow `as` casts、unwrap 限制
  - 测试: `cargo clippy` 通过
  - 前置: 无
  - 预计: 5 min

### Wave 2: CI 集成

- [ ] **T4**: CI 配置：fmt + clippy + test 矩阵
  - 文件: `.github/workflows/ci.yml`
  - 内容: std/no_std/parallel/simd feature 组合矩阵
  - 测试: CI 绿灯
  - 前置: T2
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] [T2] [T3]
            │   │   │
            └───┴───┘
                │
                ▼
Wave 2: [T4]
```

---

## 11. 测试计划

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_rustfmt_check` | `cargo fmt --check` 通过 | 高 |
| `test_clippy_check` | `cargo clippy` 无警告 | 高 |
| `ci_no_std_check` | `cargo check --no-default-features` 编译通过 | 高 |
| `test_compile_all_features` | `--all-features` 编译通过 | 高 |

---

## 12. 设计决策记录

### 决策 1：F-order 单一布局

| 属性 | 值 |
|------|-----|
| 决策 | 仅支持列优先（F-order）布局，不支持 C-order |
| 理由 | 简化 API 和实现；BLAS/LAPACK 兼容；减少布局组合爆炸 |
| 替代方案 | 同时支持 F-order 和 C-order — 放弃，增加复杂度且与项目范围不符 |
| 替代方案 | 默认 C-order — 放弃，不利于 BLAS 集成 |

### 决策 2：封闭元素类型集合

| 属性 | 值 |
|------|-----|
| 决策 | 元素类型为封闭集合（i32, i64, f32, f64, Complex\<f32\>, Complex\<f64\>, bool, usize），不支持下游扩展 |
| 理由 | 允许穷举匹配优化；SIMD 路径可针对每种类型特化；避免泛型膨胀 |
| 替代方案 | 开放 Element trait 允许用户实现 — 放弃，无法保证 SIMD 行为一致性 |

### 决策 3：单一错误枚举

| 属性 | 值 |
|------|-----|
| 决策 | 使用单一 `XenonError` 枚举覆盖所有可恢复错误 |
| 理由 | API 简单、模式匹配完整、无错误类型爆炸、no_std 友好 |
| 替代方案 | 多个错误类型 — 放弃，增加 API 复杂度 |
| 替代方案 | 使用 thiserror — 放弃，引入外部依赖 |

### 决策 4：仅 rayon + pulp 依赖

| 属性 | 值 |
|------|-----|
| 决策 | 外部依赖仅 rayon（并行）和 pulp（SIMD），均为可选 |
| 理由 | 最小依赖原则；核心功能零依赖；并行和 SIMD 为渐进增强 |
| 替代方案 | 引入 smallvec — 放弃，增加依赖 |
| 替代方案 | 引入 num-traits — 放弃，Xenon 封闭类型集合可自行定义 trait |

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
