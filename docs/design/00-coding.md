# 通用编码规范

> 文档编号: 00
> 适用范围: 全局编码与工程约束
> 任务阶段: Phase 0
> 前置文档: 无
> 需求参考: 需求说明书 §1, §4, §7, §10, §18, §23, §25, §27, §28
> 范围声明: 范围内

---

## 1. 主题定位与适用范围

本文档是 Xenon 的横切编码规范，约束命名、格式、类型系统、unsafe、文档、测试与 feature gate 的统一写法。

### 1.1 影响范围

本文档适用于 Xenon 项目的所有源码文件、测试文件、基准测试和 CI 配置。

受影响的模块包括 `src/` 下所有子模块以及 `tests/`、`benches/` 目录。

### 1.2 需求映射与范围约束

| 类型     | 内容                                                                 |
| -------- | -------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §1, §4, §7, §10, §18, §23, §25, §27, §28                  |
| 范围内   | 命名、格式、类型系统、unsafe、文档、测试与 feature gate 的统一编码约束 |
| 范围外   | 单个业务模块的算法细节、独立功能设计、额外平台适配策略               |
| 非目标   | 通过本规范引入超出需求范围的新能力、第三方依赖或额外 crate 拆分      |

---

## 2. 命名规范

### 2.1 模块名

模块名使用 `snake_case`。

```rust,ignore
// Good
mod tensor_base;
mod layout;
mod shape;

// Bad
mod tensorBase;
mod TensorBase;
mod tensor-base;
```

### 2.2 类型名

类型名（struct、enum、type alias）使用 `CamelCase`。

```rust,ignore
// Good
pub struct TensorBase<S, D> { /* ... */ }
pub struct ViewRepr<'a, A> { /* ... */ }
pub struct ViewMutRepr<'a, A> { /* ... */ }
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

### 2.3 Trait 名

Trait 名使用 `CamelCase`。标记 trait 使用描述性形容词。

```rust,ignore
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

### 2.4 索引类型别名

`Ix` 为 `usize` 的别名，用于维度值和索引：

```rust
pub type Ix = usize;
```

### 2.5 函数和方法名

函数和方法名使用 `snake_case`。

```rust,ignore
// Good
pub fn shape(&self) -> &[Ix];
pub fn transpose(&self) -> TensorView<'_, A, D>;
fn compute_f_strides(shape: &[Ix]) -> Result<Vec<Ix>, XenonError>;

// Bad
pub fn Shape(&self) -> &[Ix];
pub fn getShape(&self) -> &[Ix];
pub fn computeStrides(shape: &[Ix]) -> Vec<Ix>;
```

### 2.6 常量名

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

### 2.7 类型参数

类型参数使用单字母大写，选择有语义的字母。

| 参数 | 含义                                 |
| ---- | ------------------------------------ |
| `S`  | Storage（存储类型）                  |
| `D`  | Dimension（维度类型）                |
| `A`  | Element type（元素类型，来自 Array） |
| `L`  | Local helper type（仅模块内部局部泛型示例） |
| `T`  | 通用类型参数                         |
| `E`  | Error type（错误类型）               |

### 2.8 生命周期

生命周期使用短名：`'a`、`'b`、`'c` 等。

```rust,ignore
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

### 2.9 方法前缀约定

| 前缀    | 语义                   | 复杂度   | 示例                              |
| ------- | ---------------------- | -------- | --------------------------------- |
| `as_`   | 借用转换，O(1)，无分配 | 廉价     | `as_slice()`、`as_ptr()`          |
| `to_`   | 克隆转换，可能分配     | 可能昂贵 | `to_vec()`、`to_owned()`          |
| `into_` | 消耗 self，转换所有权  | 变化     | `into_raw_vec()`、`into_owned()`  |
| `is_`   | 布尔查询，无副作用     | 廉价     | `is_empty()`、`is_f_contiguous()` |
| `with_` | 构建器模式，返回 Self  | 变化     | `with_shape()`、`with_capacity()` |

```rust,ignore
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

### 2.10 Getter类方法约定

不加 `get_` 前缀。

```rust,ignore
// Good
pub fn shape(&self) -> &[Ix] { self.dim.slice() }
pub fn strides(&self) -> &[Ix] { &self.strides.slice() }
pub fn ndim(&self) -> usize { self.dim.ndim() }
pub fn len(&self) -> usize {
    self.dim
        .checked_size()
        .expect("len: dimension metadata must already be validated by constructors")
}

// Bad
pub fn get_shape(&self) -> &[Ix] { /* ... */ }
pub fn get_strides(&self) -> &[Ix] { /* ... */ }
```

**例外**：当方法可能失败或需要参数时，使用 `try_` 前缀表达 fallible API：

```rust,ignore
// Good - requires index parameter, may fail, returns XenonError::IndexOutOfBounds
pub fn try_at(&self, index: &[Ix]) -> Result<&A> { /* ... */ }

// Good - may fail, returns XenonError::IndexOutOfBounds
pub fn try_at_mut(&mut self, index: &[Ix]) -> Result<&mut A, XenonError> { /* ... */ }

// `[]` remains a separate restricted panic sugar for already-validated paths.
// The corresponding docs must include a `# Panics` section, and internal
// validated builders such as `eye()` may use unchecked indexing only with a
// localized `SAFETY (§8.2): ...` argument.
```

---

## 3. 代码格式

### 3.1 缩进

使用 4 空格缩进，不使用 tab。

### 3.2 rustfmt 配置

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

### 3.3 行宽限制

每行最大宽度为100 字符。超过时优先换行而非缩短变量名。

```rust,ignore
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

### 3.4 导入分组规则

导入按以下顺序分组，每组之间空一行：

1. `std` / `core` / `alloc`
2. 外部 crate
3. 本 crate 内部模块

```rust,ignore
// Good
use alloc::vec::Vec;
use alloc::borrow::Cow;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use std::error::Error;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::dimension::Dimension;
use crate::error::XenonError;
use crate::layout::{compute_f_strides, is_f_contiguous};
use crate::storage::Storage;

// Bad
use crate::dimension::Dimension;
use alloc::vec::Vec;
use rayon::prelude::*;
use core::ptr::NonNull;
use crate::storage::Storage;
```

---

## 4. 类型系统规范

### 4.1 限制 `as` 数值类型转换

公开 API 和常规代码中，对数值 `as` 采用“启用针对数值 `as` 的 lint + 对例外场景做代码评审约束”策略。默认使用 `From`/`TryFrom`/`Into` trait。

```rust,ignore
// Good
let x: i32 = value.try_into().map_err(|_| ConversionError)?;

// Bad — in public API or general code
let x: i32 = value as i32;  // dangerous: may truncate or change sign
let y: f64 = value as f64;  // dangerous: precision loss
```

**例外（须在代码评审中显式确认）**：以下情况允许 `as`：

```rust,ignore
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

### 4.2 泛型约束写法

**内联约束**：用于简单约束（1-2 个 trait），约束仅在类型定义中使用。

**`where` 子句**：用于复杂约束（3+ trait），约束涉及关联类型，约束较长影响可读性，impl 块。

```rust,ignore
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

### 4.3 PhantomData 使用规范

使用 `PhantomData` 表示逻辑上存在但运行时不占用的类型关系。

**PhantomData 模式选择表**：

| 模式                       | 含义          | 协变性                |
| -------------------------- | ------------- | --------------------- |
| `PhantomData<T>`           | 拥有 T        | 表达逻辑拥有关系，行为按拥有 `T` 理解 |
| `PhantomData<&'a T>`       | 借用 T        | 协变（对 T）          |
| `PhantomData<&'a mut T>`   | 可变借用 T    | 不变（invariant）     |
| `PhantomData<fn(T) -> T>`  | 函数参数/返回 | 不变                  |
| `PhantomData<fn() -> T>`   | 仅返回 T      | 协变（covariant）     |
| `PhantomData<fn(T) -> ()>` | 仅消费 T      | 逆变（contravariant） |
| `PhantomData<Cell<T>>`     | 如需不变      | 不变（invariant）     |

```rust,ignore
// Good - covariant borrow of element type
pub struct ViewRepr<'a, A> {
    _marker: PhantomData<&'a A>,
}

// Good - invariant mutable borrow of element type
pub struct ViewMutRepr<'a, A> {
    _marker: PhantomData<&'a mut A>,
}

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

**约定**：

- 只读视图使用 `PhantomData<&'a A>` 表达借用语义
- 可变视图使用 `PhantomData<&'a mut A>` 表达独占借用与不变性
- 只有真实拥有元素所有权的类型才使用`PhantomData<A>`
- 不要在视图类型上用 `PhantomData<A>` 冒充借用关系。

### 4.4 Send/Sync 实现规范

按存储模式声明 `unsafe impl Send/Sync`，须严格遵循以下规则（权威定义参见 `25-safety.md §5.1`，以及 `需求说明书 §25`）：

| 存储模式             | Send | Sync   | 条件                                                 |
| -------------------- | ---- | ------ | ---------------------------------------------------- |
| `Owned<A>`           | 是   | 是     | Send: `A: Send`，Sync: `A: Sync`（与 `Vec<A>` 一致） |
| `ViewRepr<'a, A>`    | 是   | 是     | `A: Sync`                                            |
| `ViewMutRepr<'a, A>` | 是   | **否** | `A: Send`（独占借用不可共享）                        |
| `ArcRepr<A>`         | 是   | 是     | `A: Send + Sync`                                     |

**关键约束**：`ViewMutRepr` 永远不实现 `Sync`——独占借用语义要求同一时刻只有一个线程可访问。

```rust,ignore
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

## 5. 错误处理规范

### 5.1 Result vs panic

**使用 `Result`**（可恢复错误）：

- 运行时约束违反（形状不匹配、广播失败）
- 用户输入无效
- 方法型 API 中可恢复的边界检查失败（例如 `broadcast_to()` / `slice()` / `try_offset_of()`）
- 公开安全索引 API 的越界与维度不匹配必须返回可恢复错误

**使用 `panic!`**（不可恢复错误）：

- 前置条件违反（不变量被破坏）
- 逻辑错误（不可能的状态）
- 契约违反
- 已证明前提下的内部快捷路径可使用 unchecked 或索引语法糖

```rust,ignore
// Good - recoverable error
pub fn broadcast_to<D2>(&self, shape: D2) -> Result<TensorView<'_, A, D2>>
where
    D2: Dimension,
{
    if !is_broadcast_compatible(self.shape(), shape.as_ref()) {
        return Err(XenonError::BroadcastError {
            operation: "broadcast_to",
            lhs_shape: self.shape().into(),
            rhs_shape: shape.as_ref().into(),
            attempted_target_shape: Some(shape.as_ref().into()),
            axis: None,
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
pub fn broadcast_to_bad<D2>(&self, shape: D2) -> TensorView<'_, A, D2>
where
    D2: Dimension,
{
    assert!(is_broadcast_compatible(self.shape(), shape.as_ref()));
    // ...
}
```

### 5.2 XenonError 设计

单一 `XenonError` 错误模型的架构决策已在 `01-architecture.md §13` 中定义。本节仅说明该决策在编码层面的公开 API、错误映射与文档写法约束。统一错误类型 `XenonError`，覆盖所有可恢复错误场景，以下为字段形态示意，非权威定义。错误模型以 `26-error.md` 为准。

```rust,ignore
#[derive(Debug, Clone)]
pub enum XenonError {
    BroadcastError {
        operation: &'static str,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        attempted_target_shape: Option<Vec<usize>>,
        axis: Option<usize>,
    },
    InvalidArgument {
        operation: Cow<'static, str>,
        argument: Cow<'static, str>,
        expected: Cow<'static, str>,
        actual: Cow<'static, str>,
        axis: Option<usize>,
        axis_len: Option<usize>,
        start: Option<usize>,
        end: Option<usize>,
        shape: Option<Vec<usize>>,
    },
    InvalidShape {
        operation: Cow<'static, str>,
        shape: Vec<usize>,
        expected_elements: usize,
        actual_elements: usize,
        offending_dim: Option<usize>,
        reason: Option<Cow<'static, str>>,
    },
    IndexOutOfBounds {
        operation: Cow<'static, str>,
        attempted_index: Vec<usize>,
        axis: usize,
        shape: Vec<usize>,
    },
    InvalidAxis {
        operation: Cow<'static, str>,
        axis: usize,
        ndim: usize,
        shape: Vec<usize>,
    },
}

pub type Result<T> = std::result::Result<T, XenonError>;
```

### 5.3 unwrap 限制

库代码中禁止使用 `unwrap()`。`expect()` 仅允许用于断言已证明的不变量或前置条件，且消息必须说明为何此处不会失败。测试代码不受此限制。

```rust,ignore
// Good - use ? and Result
pub fn broadcast_to<D2>(&self, shape: D2) -> Result<TensorView<'_, A, D2>>
where
    D2: Dimension,
{
    if !is_broadcast_compatible(self.shape(), shape.as_ref()) {
        return Err(XenonError::BroadcastError {
            operation: "broadcast_to",
            lhs_shape: self.shape().into(),
            rhs_shape: shape.as_ref().into(),
            attempted_target_shape: Some(shape.as_ref().into()),
            axis: None,
        });
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
    fn test_broadcast() {
        let arr = Tensor::from_vec(vec![1, 2, 3]); // non-normative convenience layer
        let view = arr.broadcast_to([3, 3]).unwrap();  // allowed in tests
    }
}
```

### 5.4 checked + unchecked 变体

对于性能关键的边界检查操作，提供 checked 和 unchecked 两个版本：

```rust,ignore
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// Checked indexing — returns `Err(XenonError::IndexOutOfBounds{...})` on out of bounds.
    pub fn try_at(&self, index: &[Ix]) -> Result<&A> {
        if !self.is_index_valid(index) {
            return Err(XenonError::IndexOutOfBounds {
                operation: "try_at".into(),
                attempted_index: index.into(),
                axis: 0,
                shape: self.shape().into(),
            });
        }
        // SAFETY: index is validated above
        Ok(unsafe { self.get_unchecked(index) })
    }

    /// `[]` indexing sugar for already-validated internal paths.
    /// Public safe APIs should use `try_at()` / `try_at_mut()` and return recoverable errors.
    // Note: `[]` remains restricted panic sugar outside Xenon's stable safe API contract.

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

## 6. unsafe 规范

### 6.1 最小化 unsafe 块范围

```rust,ignore
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

### 6.2 unsafe fn 必须有 # Safety 文档节

每个 `unsafe fn` 必须在文档中包含 `# Safety` 节，列出所有前提条件。

```rust,ignore
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

### 6.3 unsafe 块必须有 // SAFETY: 注释

```rust,ignore
pub fn set(&mut self, index: &[Ix], value: A) -> Result<()> {
    let offset = self.compute_offset(index)?;

    // SAFETY: compute_offset returns Ok only if index is in bounds
    unsafe {
        self.data.as_mut_ptr().add(offset).write(value);
    }

    Ok(())
}
```

### 6.4 ZST 与空数组安全

涉及零大小类型（ZST）和空数组的 unsafe 操作，须确保不引发未定义行为，包括但不限于空切片指针运算和零长度偏移计算。实现中必须先证明指针来源、对齐、可达范围与偏移语义在 `len == 0` 或 `size_of::<T>() == 0` 时仍然成立，不得把“不会解引用”当作可跳过前提校验的理由。

```rust,ignore
pub unsafe fn ptr_at_unchecked<T>(ptr: *const T, len: usize, index: usize) -> *const T {
    debug_assert!(index <= len);
    // SAFETY: caller guarantees provenance, bounds, and ZST/empty-slice behavior.
    ptr.add(index)
}
```

### 6.5 unsafe 封装在安全抽象内部

所有 `unsafe` 代码应封装在安全抽象内部，对外暴露安全 API：

```
src/
├── tensor/
│   ├── mod.rs          # public safe API
│   ├── raw.rs          # private unsafe primitives
│   └── iter.rs         # safe iterators
├── storage/
│   ├── mod.rs          # public safe API
│   ├── owned.rs        # contains unsafe, exposes safe API
│   └── view.rs         # contains unsafe, exposes safe API
└── lib.rs
```

---

## 7. 文档规范

### 7.1 `lib.rs` 项目级 lint 基线

`lib.rs` 必须声明项目级统一 lint 基线；其中 `#![warn(clippy::unwrap_used)]` 是整个项目库代码的统一要求，不得在其他模块或文档片段中弱化或省略。本节的 `lib.rs` lint 列表是权威来源。`01-architecture.md §8` 及其他文档中的 `lib.rs` 片段须与本节保持一致。
>
> **CI 强制执行**：开发期间使用 `warn` 级别；CI 中应分别执行 `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`（rustdoc 警告）、`cargo clippy -- -D warnings`（Clippy 警告），并在需要把常规编译警告也提升为错误时额外使用 `RUSTFLAGS="-D warnings" cargo check`。

```rust,ignore
// src/lib.rs

//! Xenon: N-dimensional array library for Rust.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(rust_2024_compatibility)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]
```

### 7.2 所有 pub 项必须有 doc comment

所有 `pub` 项（函数、结构体、trait、模块、常量）必须有文档注释。

### 7.3 函数文档结构

1. **简述**（第一行）：一句话描述功能
2. **详述**：详细说明行为
3. **# Arguments**：参数说明
4. **# Returns**：返回值说明
5. **# Errors**：可能的错误
6. **# Panics**：可能 panic 的情况
7. **# Safety**：unsafe 函数的前提条件
8. **# Examples**：示例代码

````rust,ignore
/// Returns a broadcasted read-only view for the requested shape.
///
/// This operation reuses the original storage and only updates metadata.
///
/// # Errors
/// Returns [`XenonError::BroadcastError`] if `shape` is not broadcast-compatible
/// with `self.shape()`.
///
/// # Examples
/// ```rust
/// use xenon::{Tensor1, Ix2};
///
/// let arr = Tensor1::from_vec(vec![1, 2, 3])?; // non-normative convenience layer
/// let view = arr.broadcast_to(Ix2(3, 3))?;
/// assert_eq!(view.shape(), &[3, 3]);
/// # Ok::<(), xenon::XenonError>(())
/// ```
pub fn broadcast_to<D2>(&self, shape: D2) -> Result<TensorView<'_, A, D2>>
where
    D2: Dimension,
{
    // ...
}
````

### 7.4 示例代码使用 `?` 而非 `unwrap()`

文档示例代码应使用 `?` 运算符处理错误：

````rust,ignore
// Good
/// # Examples
/// ```rust
/// use xenon::Tensor;
///
/// let arr = Tensor::from_vec(vec![1, 2, 3, 4])?; // non-normative convenience layer
/// # Ok::<(), xenon::XenonError>(())
/// ```

// Bad
/// # Examples
/// ```rust
/// let arr = Tensor::from_vec(vec![1, 2, 3, 4]).unwrap();  // do not do this
/// ```
````

---

## 8. 测试规范

### 8.1 测试命名规范

测试函数命名格式：`test_<function>_<scenario>_<expected>`。其中 `Index` / `IndexMut` 的 `[]` 语法仅作为已验证路径的受限人体工学 panic sugar；公开安全 API 必须优先通过 `try_at()` / `try_at_mut()` 暴露可恢复错误语义。

```rust,ignore
#[cfg(test)]
mod tests {
    #[test]
    fn test_broadcast_scalar_to_matrix_succeeds() { /* ... */ }

    #[test]
    fn test_broadcast_incompatible_shape_fails() { /* ... */ }

    #[test]
    fn test_index_single_element_returns_value() { /* ... */ }

    #[test]
    fn test_index_out_of_bounds_panics() {
        // This test intentionally exercises `Index` panic sugar (`[]`),
        // not Xenon's recoverable safe indexing API.
    }

    #[test]
    fn test_safe_index_returns_error() {
        // Safe public indexing must use `try_at()` / `try_at_mut()` and
        // return `XenonError::IndexOutOfBounds` instead of panicking.
    }

    #[test]
    fn test_sum_float_tensor_with_nan_returns_nan() { /* ... */ }

    #[test]
    fn test_dot_incompatible_shapes_fails() { /* ... */ }
}
```

### 8.2 边界覆盖必须覆盖

| 场景                       | 预期行为                  |
| -------------------------- | ------------------------- |
| 空数组 `shape=[0, 3]`      | `iter()` 立即结束         |
| 单元素 `shape=[1, 1]`      | `iter()` 产出 1 项        |
| 非连续切片                 | `iter()` 正确处理步长跳转 |
| NaN/Inf                    | 遵循 IEEE 754 语义        |
| 高维 `shape=[2,2,2,2,2,2]` | 正确计算偏移              |
| 大张量 `shape=[10_000_000]` 或 GiB 级输入 | 不栈溢出，且边界检查保持可恢复错误语义 |
| Subnormal 浮点数           | 不 flush to zero          |
| 需求说明书 §28.4 占位：large-tensor   | 后续补充超大张量边界回归用例 |
| 需求说明书 §28.4 占位：high-dim       | 后续补充高维 shape / stride / index 回归用例 |
| 需求说明书 §28.4 占位：extreme-value  | 后续补充极值/特殊值数值回归用例 |

### 8.3 测试分类

| 类型     | 位置                     | 目的                            |
| -------- | ------------------------ | ------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法               |
| 集成测试 | `tests/`                 | 验证跨模块交互                  |
| 边界测试 | 集成测试中标注           | 空数组、单元素、NaN/Inf、非连续 |
| 属性测试 | `tests/property/`        | 随机生成验证不变量              |

### 8.4 测试覆盖率与数值精度

**覆盖率与质量门槛**：

- unsafe 代码块必须有对应测试
- 每个公开 API 至少一个正向 + 一个负向测试
- 如仓库后续引入覆盖率门槛，应作为可选 CI 质量信号维护，不得写成超出 `需求说明书 §28` 的硬性发布准入条件

**浮点比较**：对存在舍入误差容差的数值路径，使用近似比较；对比较 API 自身、布尔结果、文档明确要求精确一致的场景，允许/要求精确断言。参见 `需求说明书 §12`（NaN 遵循 IEEE 754）和 `需求说明书 §28.3`。

```rust,ignore
// Good - tolerance uses max(1 ULP, epsilon * |scalar_result|)
fn assert_close(a: f64, b: f64, epsilon: f64) {
    let diff = (a - b).abs();
    let scalar_result = a.abs().max(b.abs());
    let tol = f64::EPSILON.max(epsilon * scalar_result.abs());
    assert!(diff <= tol, "expected {a} ≈ {b}, tol={tol}");
}

// Bad - direct float comparison
assert_eq!(result[[0, 0]], 58.0);  // may fail due to rounding errors
```

### 8.5 归约操作溢出行为

| 类型       | 行为                                               |
| ---------- | -------------------------------------------------- |
| 整数类型   | debug 和 release 均使用 checked 算术，溢出时 panic |
| 浮点类型   | 返回 `±Infinity`（IEEE 754 语义）                  |
| 空数组 sum | 返回加法单位元 `zero()`                            |

---

## 9. #[inline] 使用规范

**使用 `#[inline]`**：

- 小函数（1-3 行）
- 泛型函数（必须在调用处实例化）
- 频繁调用的简单方法

**不使用 `#[inline]`**：

- 大函数
- 递归函数
- 很少调用的函数

**`#[inline(always)]`** 仅用于经性能分析确认的关键热路径。

```rust,ignore
// Good - small function
#[inline]
pub fn len(&self) -> usize {
    self.dim
        .checked_size()
        .expect("len: dimension metadata must already be validated by constructors")
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

## 10. Feature Gate 规范

### 10.1 Feature 必须是 additive

启用 feature 不能移除功能。所有 feature 组合必须能正确编译。

| Feature    | 依赖        | 说明                   |
| ---------- | ----------- | ---------------------- |
| `simd`     | `dep:pulp`  | SIMD 加速，默认关闭    |
| `parallel` | `dep:rayon` | 并行计算，默认关闭     |

其中 `simd` 与 `parallel` 都建立在 Xenon 的 `std` 前提之上；`std` 是无条件工程基线，不单独建模为 feature。

```toml
[features]
parallel = ["dep:rayon"]      # Additive: enables internal parallel execution backend
simd = ["dep:pulp"]           # Additive: enables SIMD with std-backed intrinsics
# Note: libm is NOT a dependency (see 01-architecture.md §1.4).
# RealScalar math functions (sin/exp/ln) rely on Xenon's unconditional std baseline.
```

### 10.2 使用 `dep:` 语法声明可选依赖

```toml
[dependencies]
rayon = { version = "1.10", optional = true }
pulp = { version = "0.18", optional = true }
```

### 10.3 `cfg_attr(docsrs, doc(cfg(...)))` 标注条件编译 API

```rust
/// Internal parallel execution backend marker.
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub(crate) trait ParallelBackend {
    fn for_each<F>(&self, f: F)
    where
        F: Fn(usize) + Send + Sync;
}
```

在 `Cargo.toml` 中配置 docsrs：

```toml
[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

## 平台与工程约束

| 约束项     | 要求                                                             |
| ---------- | ---------------------------------------------------------------- |
| `std` only | 所有代码依赖 `std`，不讨论 `no_std`                              |
| MSRV       | Rust 1.85+                                                       |
| 单 crate   | 保持单 crate 边界，不引入额外 crate                              |
| SemVer     | 遵循 SemVer，公开 API 变更须同步版本号                           |
| 最小依赖   | 仅允许 `rayon`（并行）和 `pulp`（SIMD）作为可选外部依赖，默认关闭 |

---

## 11. 实现任务拆分

### Wave 1: 规范基础设施

- [ ] **T1**: 创建 `rustfmt.toml` 配置文件
  - 文件: `rustfmt.toml`
  - 内容: §3.2 中的完整配置
  - 测试: `cargo fmt --check` 通过
  - 前置: 无
  - 预计: 5 min

- [ ] **T2**: 创建 `src/lib.rs` 骨架含 lint 声明
  - 文件: `src/lib.rs`
  - 内容: missing_docs、unsafe_op_in_unsafe_fn、clippy::unwrap_used、std-only lint 配置
  - 测试: 编译通过
  - 前置: 无
  - 预计: 5 min

- [ ] **T3**: 创建 `.clippy.toml` 或 `clippy` 配置
  - 文件: `.clippy.toml`
  - 内容: 启用针对数值 `as` 的 lint + 对例外场景做代码评审约束（例外仅限 C FFI 指针转换、已验证的索引/长度到指针偏移转换、原始指针地址操作、以及已文档化语义的内部专用 cast 实现）以及 unwrap 限制
  - 测试: `cargo clippy` 通过
  - 前置: 无
  - 预计: 5 min

### Wave 2: CI 集成

- [ ] **T4**: CI 配置：fmt + clippy + test 矩阵
  - 文件: `.github/workflows/ci.yml`
  - 内容: std-only、parallel、simd feature 组合矩阵
  - 测试: CI 绿灯
  - 前置: T2
  - 预计: 10 min

> **CI 策略补充说明：** `cargo fmt --check`、`cargo test`、关键 compile-fail/文档检查哪些属于阻塞发布的 hard gate，哪些仅作为 advisory 信号，仍需项目级统一裁决；本规范只定义建议纳入的检查项，不越权替代仓库治理决策。
>
> **工程治理说明：** 以下为工程治理建议，不构成 `需求说明书 §28` 当前版本的规范性基线。

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

## 验证与落地方式

| 测试函数                    | 测试内容                  | 优先级 |
| --------------------------- | ------------------------- | ------ |
| `test_rustfmt_check`        | `cargo fmt --check` 通过  | 高     |
| `test_clippy_check`         | `cargo clippy` 无警告     | 高     |
| `test_std_default_check`    | 默认 `std` 配置编译通过   | 高     |
| `test_compile_all_features` | `--all-features` 编译通过 | 高     |

> **覆盖补充：** 边界类（空张量/单元素/大张量/极值/高维/非法元素类型）、并行/SIMD 路径一致性、compile-fail 约束测试由对应模块文档（`27-benchmark.md §8`、`28-tests.md §8`）具体定义。

## 12. 验证补充

### 12.1 Feature gate / 配置测试

| 配置        | 验证点                                           |
| ----------- | ------------------------------------------------ |
| 默认配置    | 编码规范对应 lint、格式与文档约束默认生效        |
| 启用 `simd` | 条件编译代码仍遵循相同命名、文档与 unsafe 规范    |
| 启用并行    | 条件编译代码仍遵循相同错误语义、测试与注释要求    |
| 全 feature  | 所有 feature 组合下格式、lint 与文档检查口径一致 |

### 12.2 类型边界 / 编译期测试

| 场景                         | 测试方式                                 |
| ---------------------------- | ---------------------------------------- |
| `unsafe fn` 文档节完整性     | `cargo doc` + rustdoc/clippy 文档 lint   |
| `unwrap()` / `as` 使用边界   | `cargo clippy` 与仓库 lint 配置校验      |
| feature gate 可见性与导出边界 | `cargo check` / `cargo test` 配置矩阵验证 |

> **compile-fail 测试机制建议：** 对 sealed trait、feature gate 可见性、错误用法示例等编译期失败场景，建议使用 `trybuild` 或等价 compile-fail harness 统一维护；若项目后续选择其他机制，需保证失败快照与错误意图可审计。

> **工程治理说明：** 以下为工程治理建议，不构成 `需求说明书 §28` 当前版本的规范性基线。

---

## 错误处理与语义边界

本文档不直接定义错误类型，但要求所有受影响模块遵循 `26-error.md` 的错误语义边界；编码规范只约束 `Result`、panic、`unsafe` 注释与文档节的写法，不单独裁决公开 API 的错误分类。

---

## 13. 设计决策记录

> **架构交叉引用**：F-order 单一布局、封闭元素类型集合与单一 `XenonError` 错误枚举的正式架构决策已记录于 `01-architecture.md §13`；本节仅保留这些决策对编码规范与实现约束的直接影响，便于在编码语境中引用。

### 决策 1：F-order 单一布局

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | 仅支持列优先（F-order）布局，不支持 C-order                    |
| 理由     | 简化 API 和实现；BLAS/LAPACK 兼容；减少布局组合爆炸            |
| 替代方案 | 同时支持 F-order 和 C-order — 放弃，增加复杂度且与项目范围不符 |
| 替代方案 | 默认 C-order — 放弃，不利于 BLAS 集成                          |

### 决策 2：封闭元素类型集合

| 属性     | 值                                                                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 元素类型为封闭集合（i32, i64, f32, f64, Complex\<f32\>, Complex\<f64\>, bool），不支持下游扩展；`usize` 仅作为索引与形状元数据类型使用 |
| 理由     | 允许穷举匹配优化；SIMD 路径可针对每种类型特化；避免泛型膨胀                                                                            |
| 替代方案 | 开放 Element trait 允许用户实现 — 放弃，无法保证 SIMD 行为一致性                                                                       |

### 决策 3：单一错误枚举

| 属性     | 值                                                                              |
| -------- | ------------------------------------------------------------------------------- |
| 决策     | 使用单一 `XenonError` 枚举覆盖所有公开可恢复错误                                |
| 理由     | API 简单、模式匹配完整，并能集中承载索引、广播、类型转换与 FFI 的结构化诊断信息 |
| 替代方案 | 多个错误类型 — 放弃，增加 API 复杂度                                            |
| 替代方案 | 使用 thiserror — 放弃，引入外部依赖                                             |

### 决策 4：仅 rayon + pulp 依赖

| 属性     | 值                                                         |
| -------- | ---------------------------------------------------------- |
| 决策     | 外部依赖仅 rayon（并行）和 pulp（SIMD），均为可选          |
| 理由     | 最小依赖原则；核心功能零依赖；并行和 SIMD 为渐进增强       |
| 替代方案 | 引入 smallvec — 放弃，增加依赖                             |
| 替代方案 | 引入 num-traits — 放弃，Xenon 封闭类型集合可自行定义 trait |

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
