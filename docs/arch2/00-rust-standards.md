# Xenon Rust 编程规范

> 本文档定义 Xenon 项目通用的 Rust 编程规范，所有模块设计与实现均须遵守。

---

## 1. 通用约定

### 1.1 MSRV

- 最低支持 Rust 版本：1.85+
- 使用 edition 2024
- 在 `Cargo.toml` 中声明 `rust-version = "1.85"`

### 1.2 Crate 结构

- 单 crate（`xenon`），遵循 [SemVer](https://semver.org)
- 公共 API 仅通过 `lib.rs` 和各模块的公开项暴露
- 内部实现细节放在 `private` 模块或标记 `#[doc(hidden)]`

### 1.3 依赖策略

- 最小依赖原则：仅允许 `rayon`（可选）和 `pulp`（可选）作为外部依赖
- `thiserror` 用于错误类型派生（如果 MSRV 允许）
- 禁止引入重型依赖（如 `num-traits`、`serde`）

---

## 2. 命名规范

### 2.1 类型命名

| 类别 | 风格 | 示例 |
|------|------|------|
| 结构体 | UpperCamelCase | `TensorBase`, `Owned`, `IxDyn` |
| 枚举 | UpperCamelCase | `PadMode`, `Order` |
| trait | UpperCamelCase | `Element`, `Dimension`, `Storage` |
| 类型别名 | UpperCamelCase | `Tensor`, `TensorView`, `Tensor1` |
| 常量 | SCREAMING_SNAKE_CASE | `F_CONTIGUOUS`, `DEFAULT_ALIGNMENT` |
| 关联常量 | SCREAMING_SNAKE_CASE | `Element::ZERO`, `RealScalar::EPSILON` |

### 2.2 函数与方法命名

| 类别 | 风格 | 示例 |
|------|------|------|
| 公共方法 | snake_case | `is_f_contiguous()`, `to_owned()` |
| 转换方法 | `to_`/`into_`/`as_` 前缀 | `to_f_contiguous()`, `into_owned()`, `as_ptr()` |
| 布尔查询 | `is_`/`has_`/`can_` 前缀 | `is_contiguous()`, `has_zero_stride()` |
| 构造函数 | `from_`/`new` | `from_vec()`, `zeros()`, `from_raw_parts()` |
| 迭代器 | `iter`/`iter_mut`/`into_iter` | `iter()`, `axis_iter()`, `windows()` |
| 检查变体 | `_checked`/`_unchecked` 后缀 | `get_unchecked()`, `index_checked()` |

### 2.3 模块命名

- 全小写 snake_case：`dimension`, `element`, `storage`, `layout`, `tensor`
- 公共模块：`src/dimension.rs` 或 `src/dimension/mod.rs`
- 内部辅助模块：`src/private/` 下

### 2.4 泛型参数命名

| 参数 | 含义 | 约定 |
|------|------|------|
| `A` | 元素类型 (Atom) | 用于张量数据元素 |
| `D` | 维度类型 (Dimension) | 用于维度系统 |
| `S` | 存储类型 (Storage) | 用于存储系统 |
| `T` | 通用泛型参数 | 用于 Complex<T> 等 |

---

## 3. 代码格式

### 3.1 基本格式

- 缩进：**4 个空格**（不用 Tab）
- 行宽：100 字符（与 rustfmt 默认一致）
- 使用 `rustfmt` 自动格式化，配置 `.rustfmt.toml`

```toml
# .rustfmt.toml
max_width = 100
tab_spaces = 4
edition = "2024"
```

### 3.2 import 风格

```rust
// 标准库 / core
use core::ops::{Add, Sub, Mul, Div};

// 外部 crate（如启用）
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// crate 内部
use crate::dimension::Dimension;
use crate::error::ShapeMismatch;
```

- 按 `std/core` → 外部 crate → crate 内部分组
- 每个 group 之间空一行
- 避免通配符 import（`use crate::*`），测试代码除外

---

## 4. 错误处理

### 4.1 错误类型定义

- 使用 `thiserror` 派生 `Error` trait（如果可用）
- 所有错误类型实现 `Display` 和 `Error`
- 错误信息包含上下文（期望值 vs 实际值）

```rust
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    // ...
}
```

### 4.2 处理策略

| 场景 | 策略 |
|------|------|
| 可恢复错误（形状、布局、轴） | 返回 `Result<T, E>` |
| 编程错误（索引越界） | `panic`，同时提供 `unsafe _unchecked` 变体 |
| 整数溢出（sum/prod） | checked 算术，溢出时 panic |
| 空数组 min/max | 返回 `Result` |

### 4.3 Result 别名

```rust
pub type Result<T> = core::result::Result<T, TensorError>;
```

### 4.4 Panic 安全

- 所有 panic 场景须在文档中注明
- unsafe 函数的 `_unchecked` 变体不检查边界，须在 `Safety` 文档节说明前提条件

---

## 5. 测试规范

### 5.1 测试组织

```
src/
  dimension.rs       // #[cfg(test)] mod tests { ... }  单元测试
  tensor.rs          // #[cfg(test)] mod tests { ... }
tests/
  integration/       // 集成测试
    construction.rs
    broadcasting.rs
    ...
benches/             // 基准测试
  reduction.rs
  element_ops.rs
```

### 5.2 测试命名

```rust
#[test]
fn test_<功能>_<场景>_<预期>() {
    // 例如：
    // test_zeros_2d_shape_correct
    // test_add_shape_mismatch_returns_error
    // test_index_out_of_bounds_panics
}
```

### 5.3 测试分类

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法 |
| 集成测试 | `tests/` | 验证跨模块交互 |
| 边界测试 | 集成测试中标注 | 空数组、单元素、NaN/Inf、非连续 |
| 属性测试 | `tests/property/` | 随机生成验证不变量 |

### 5.4 断言约定

```rust
// 相等比较
assert_eq!(result, expected);

// 浮点近似比较
use crate::testing::assert_close;
assert_close!(result, expected, 1e-10);

// panic 测试
#[should_panic(expected = "index out of bounds")]
```

---

## 6. 文档规范

### 6.1 doc comment 要求

- 所有 pub fn/method/struct/enum/trait 必须有 doc comment
- doc comment 以第三人称动词开头（`Returns...`, `Creates...`, `Panics...`）
- 必须包含以下节（如适用）：

```rust
/// Creates a new tensor filled with zeros.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
/// * `order` - Memory layout order (F-order by default).
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{Tensor, zeros, Ix2};
/// let a: Tensor<f64, Ix2> = zeros([3, 4]);
/// assert_eq!(a.shape(), &[3, 4]);
/// ```
pub fn zeros<A, D>(shape: D) -> Tensor<A, D> { ... }
```

### 6.2 Safety 文档节

所有 unsafe 函数和 unsafe impl 必须包含 `# Safety` 节：

```rust
/// Returns a mutable raw pointer to the tensor data.
///
/// # Safety
///
/// The caller must ensure that:
/// - The pointer is not used after the tensor is dropped.
/// - No other references to the data exist when writing through this pointer.
pub unsafe fn as_mut_ptr(&mut self) -> *mut A { ... }
```

---

## 7. Feature Gates

### 7.1 Feature 定义

```toml
[features]
default = ["std"]
std = []                  # 标准库支持
parallel = ["dep:rayon"]  # 并行支持
simd = ["dep:pulp"]       # SIMD 支持
```

### 7.2 条件编译约定

```rust
// std 相关功能
#[cfg(feature = "std")]
use std::sync::Arc;

// no_std 兼容的 std 功能
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

// 并行功能
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// SIMD 功能
#[cfg(feature = "simd")]
use pulp::Arch;
```

### 7.3 no_std 兼容性

- 默认 feature 为 `std`
- `no_std` 模式需 `alloc` crate
- 所有堆分配通过 `alloc::` 路径
- 避免直接使用 `std::` 中的类型，优先使用 `core::`

---

## 8. 性能规范

### 8.1 零开销抽象

- 泛型单态化：所有泛型函数在编译时单态化，无虚函数调用
- 内联：性能关键路径标注 `#[inline]` 或 `#[inline(always)]`
- 布局优化：`#[repr(C)]` 用于 FFI 兼容类型，`#[repr(transparent)]` 用于 newtype

### 8.2 内存分配

- 默认 64 字节对齐
- 避免隐式克隆（`clone()` 须显式调用）
- 视图创建 O(1)，不分配内存
- 提供 `O(1)` 大小查询方法

### 8.3 性能标注

```rust
#[inline]           // 小函数建议内联
pub fn ndim(&self) -> usize { ... }

#[inline(always)]   // 热路径强制内联（仅限经过 benchmark 验证的关键路径）
unsafe fn get_unchecked(&self, index: &[usize]) -> &A { ... }
```

---

## 9. 安全规范

### 9.1 unsafe 准则

- 最小化 unsafe 块范围
- 每个 unsafe 块须有 `// SAFETY:` 注释说明为何安全
- 安全封装：所有 unsafe 操作须提供安全封装

```rust
pub fn get(&self, index: &[usize]) -> Result<&A> {
    self.check_index(index)?;
    // SAFETY: check_index guarantees bounds validity
    Ok(unsafe { self.get_unchecked(index) })
}
```

### 9.2 未定义行为防护

- 禁止通过指针访问未初始化内存（除 `MaybeUninit` 内部使用）
- 禁止在 safe API 中暴露可能产生 UB 的操作
- 所有 `_unchecked` 变体须为 `unsafe fn`

---

## 10. 模块组织

### 10.1 文件结构约定

```
src/
  lib.rs             // crate 入口，re-export 公共 API
  private/           // 内部辅助模块，#[doc(hidden)]
    mod.rs
  dimension.rs       // 维度系统（Ix0~Ix6, IxDyn, Dimension trait）
  element.rs         // 元素类型 trait（Element, Numeric, RealScalar, ComplexScalar）
  complex.rs         // Complex<T> 类型定义
  storage/           // 存储系统
    mod.rs
    owned.rs
    view.rs
    view_mut.rs
    arc.rs
  layout.rs          // 内存布局（步长、标志、对齐）
  tensor.rs          // TensorBase 核心抽象
  error.rs           // 错误类型
  construction.rs    // 构造函数
  iter/              // 迭代器
    mod.rs
  broadcast.rs       // 广播机制
  indexing.rs        // 索引操作
  shape/             // 形状操作
    mod.rs
  ops/               // 运算操作
    mod.rs
    element_wise.rs
    matrix.rs
    reduction.rs
    set_ops.rs
  ffi.rs             // FFI 集成
  workspace.rs       // 工作空间
  backend/           // 计算后端
    mod.rs
    simd.rs
    parallel.rs
```

### 10.2 模块可见性

- 公共模块：`pub mod`
- 公共类型：`pub struct`/`pub enum`/`pub trait`
- 内部类型：`pub(crate)` 或 `pub(super)`
- 实现细节：`#[doc(hidden)]`

---

## 11. 实现任务拆分原则

每个实现任务须满足以下约束：

| 约束 | 要求 |
|------|------|
| 完成时间 | 约 10 分钟 |
| 范围 | 单个函数或单个 trait impl |
| 可验证 | 有对应的测试（先写测试） |
| 可提交 | 单次 git commit 可描述 |

### 任务模板

```
Task: 实现 <函数名>
文件: src/<module>.rs:<行范围>
测试: tests/<test_file>.rs::<test_fn>
前置: <依赖的任务>
预计: <分钟>
```
