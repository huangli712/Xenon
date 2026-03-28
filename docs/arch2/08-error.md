# 错误处理模块设计

> 模块: `src/error.rs`
> 版本: v1 | 日期: 2026-03-28
> 状态: 设计阶段

---

## 1. 模块定位

错误处理模块是 Xenon 的全局基础设施层，定义统一的错误类型 `TensorError` 和结果别名 `Result<T>`。该模块在架构图中位于最顶层（参见 [01-architecture-overview.md](01-architecture-overview.md) §3 依赖图），被所有核心模块和 API 模块直接依赖。

核心设计目标：

- **分类清晰**：可恢复错误返回 `Result`，编程错误（索引越界）panic 并提供 `unsafe` unchecked 变体
- **上下文丰富**：每个错误变体携带期望值与实际值，便于调试
- **零依赖开销**：使用 `thiserror` 派生 `Display`/`Error`，编译时消除运行时反射
- **no_std 兼容**：所有堆分配走 `alloc::` 路径，错误类型实现 `Clone`

---

## 2. 文件位置

```
src/
  error.rs          # 本模块：TensorError enum + Result 别名
```

在 `src/lib.rs` 中的声明：

```rust
pub mod error;

// re-export
pub use crate::error::{TensorError, Result};
```

---

## 3. 依赖关系

### 3.1 本模块的依赖（上游）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `thiserror` | 外部 crate（Cargo.toml） | 派生 `Error` + `Display` trait |
| `core::fmt` | `core` | `Display` trait 基础 |
| `core::error::Error` | `core`（Rust 1.81+） | `Error` trait |
| `alloc::vec::Vec` | `alloc` | 错误字段中的形状/索引存储 |

### 3.2 依赖本模块的下游

| 模块 | 使用方式 |
|------|----------|
| `tensor.rs` | 构造/索引方法返回 `Result<T>` |
| `broadcast.rs` | 广播失败返回 `TensorError::BroadcastError` |
| `shape/reshape.rs` | reshape 校验返回 `TensorError::InvalidShape` |
| `shape/slice.rs` | 布局校验返回 `TensorError::LayoutMismatch` |
| `ops/element_wise.rs` | 二元运算形状校验 |
| `ops/reduction.rs` | 空数组归约返回 `TensorError::EmptyArray` |
| `indexing.rs` | 索引越界 panic（使用 `TensorError::IndexOutOfBounds` 格式化消息） |
| `dimension.rs` | 维度转换返回 `TensorError::DimensionMismatch` |
| `iter/zip.rs` | zip 形状校验返回 `TensorError::ShapeMismatch` |
| 所有 API 模块 | 通过 `use crate::error::{TensorError, Result}` 引入 |

### 3.3 Cargo.toml 依赖声明

```toml
[dependencies]
thiserror = "2"
```

> **说明**：`thiserror` v2 支持 no_std，与 Xenon 的 MSRV 1.85+ 和 no_std 目标兼容。`thiserror` 是唯一的外部依赖（rayon 和 pulp 为可选依赖），符合最小依赖原则。

---

## 4. 公共 API 设计

### 4.1 Result 类型别名

```rust
/// Result alias for all Xenon operations that can fail.
///
/// All fallible Xenon APIs return `Result<T>` rather than
/// `std::result::Result<T, TensorError>`.
pub type Result<T> = core::result::Result<T, TensorError>;
```

### 4.2 TensorError 枚举

```rust
use alloc::vec::Vec;

/// Unified error type for all Xenon tensor operations.
///
/// # Error handling strategy
///
/// | Category | Strategy |
/// |----------|----------|
/// | Recoverable (shape, layout, axis, empty) | Return `Result<T, TensorError>` |
/// | Programming error (index out of bounds) | `panic!` with descriptive message |
/// | Unchecked variant | `unsafe` function, no bounds check (UB on invalid input) |
///
/// All variants carry context information (expected vs actual values)
/// to aid debugging.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorError {
    // ── Shape errors ──────────────────────────────────────────────

    /// Binary operation or zip with incompatible shapes that cannot be broadcast.
    ///
    /// Returned when two arrays have shapes that differ and broadcasting
    /// cannot reconcile them.
    #[error(
        "shape mismatch: incompatible shapes for binary operation \
         (lhs={lhs_shape:?}, rhs={rhs_shape:?})"
    )]
    ShapeMismatch {
        /// Shape of the left-hand side operand.
        lhs_shape: Vec<usize>,
        /// Shape of the right-hand side operand.
        rhs_shape: Vec<usize>,
    },

    /// Broadcast rules not satisfied.
    ///
    /// Returned when attempting to broadcast two shapes but at least one
    /// pair of corresponding dimensions is incompatible (neither equal nor 1).
    #[error(
        "broadcast error: cannot broadcast shapes together \
         (lhs={lhs_shape:?}, rhs={rhs_shape:?}): \
         dimension {axis} has sizes {lhs_dim} and {rhs_dim}, \
         neither of which is 1"
    )]
    BroadcastError {
        /// Shape of the left-hand side operand.
        lhs_shape: Vec<usize>,
        /// Shape of the right-hand side operand.
        rhs_shape: Vec<usize>,
        /// Axis index (0-based, in the broadcast-aligned dimension space)
        /// where incompatibility was detected.
        axis: usize,
        /// Size of the lhs dimension at the conflicting axis.
        lhs_dim: usize,
        /// Size of the rhs dimension at the conflicting axis.
        rhs_dim: usize,
    },

    // ── Layout errors ─────────────────────────────────────────────

    /// Operation requires a contiguous layout but the input is non-contiguous.
    ///
    /// Returned by operations like `reshape` which require contiguous memory
    /// but the input view has a non-contiguous layout (e.g., a transposed view).
    #[error(
        "layout mismatch: operation requires {required} layout, \
         but the input array is not contiguous \
         (flags={flags:?})"
    )]
    LayoutMismatch {
        /// Description of the required layout (e.g., "C-contiguous", "F-contiguous").
        required: &'static str,
        /// Current layout flags of the input array, for diagnostics.
        flags: LayoutContext,
    },

    // ── Axis errors ───────────────────────────────────────────────

    /// Axis index is out of range for the array's dimensionality.
    ///
    /// Returned by operations that accept an `axis` parameter
    /// (e.g., `sum_axis`, `axis_iter`).
    #[error(
        "invalid axis: axis {axis} is out of range \
         for a {ndim}-dimensional array \
         (valid range: 0..{ndim})"
    )]
    InvalidAxis {
        /// The axis index that was requested.
        axis: usize,
        /// Number of dimensions in the array.
        ndim: usize,
    },

    // ── Shape/reshape errors ──────────────────────────────────────

    /// Reshape target element count does not match the source.
    ///
    /// Returned by `reshape` when the product of the target shape
    /// does not equal the number of elements in the source array.
    #[error(
        "invalid shape: cannot reshape array of \
         {source_elements} elements into shape {target_shape:?} \
         ({target_elements} elements)"
    )]
    InvalidShape {
        /// Number of elements in the source array.
        source_elements: usize,
        /// The requested target shape.
        target_shape: Vec<usize>,
        /// Product of the target shape dimensions.
        target_elements: usize,
    },

    // ── Dimension errors ──────────────────────────────────────────

    /// Static/dynamic dimension conversion count mismatch.
    ///
    /// Returned when converting between static dimension types
    /// (e.g., `Ix3`) and dynamic (`IxDyn`) if the number of
    /// dimensions does not match.
    #[error(
        "dimension mismatch: expected {expected} dimensions, \
         got {actual}"
    )]
    DimensionMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        actual: usize,
    },

    // ── Index errors ──────────────────────────────────────────────

    /// Multi-dimensional index is out of bounds.
    ///
    /// **Handling strategy**: The checked indexing operator (`Index` trait)
    /// panics with this message. The `get` method returns
    /// `Result<T, TensorError::IndexOutOfBounds>`. The unsafe
    /// `get_unchecked` method skips bounds checking entirely (UB on
    /// invalid input).
    #[error(
        "index out of bounds: index {index:?} is out of range \
         for array with shape {shape:?}"
    )]
    IndexOutOfBounds {
        /// The multi-dimensional index that was requested.
        index: Vec<usize>,
        /// The shape of the array.
        shape: Vec<usize>,
    },

    // ── Empty array errors ────────────────────────────────────────

    /// Operation cannot be performed on an empty array.
    ///
    /// Returned by reduction operations (`min`, `max`, `argmin`,
    /// `argmax`) when the input array has zero elements.
    #[error(
        "empty array: cannot compute {operation} \
         on an array with zero elements"
    )]
    EmptyArray {
        /// Name of the operation that was attempted
        /// (e.g., "min", "max", "argmin", "argmax").
        operation: &'static str,
    },
}
```

### 4.3 LayoutContext 辅助类型

`LayoutMismatch` 中的 `flags` 字段使用 `LayoutContext` 类型而非原始 `u8` 标志位，提供人类可读的诊断信息：

```rust
/// Diagnostic context for layout-related errors.
///
/// Carries a human-readable description of the current layout state
/// to aid in debugging layout mismatch errors.
#[derive(Debug, Clone)]
pub struct LayoutContext {
    /// Human-readable description of the current layout
    /// (e.g., "non-contiguous (transposed)", "C-contiguous", "F-contiguous").
    pub description: String,
}

impl core::fmt::Display for LayoutContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.description)
    }
}
```

> **设计决策**：使用 `String` 而非 `&'static str` 是因为布局描述需要动态拼接（如 "non-contiguous, has_zero_stride, f-contiguous=false, c-contiguous=true"）。`LayoutContext` 封装而非直接暴露 `String`，为未来扩展（如添加布局标志位原始值）留出空间。

### 4.4 辅助构造方法

每个错误变体提供 `new` 关联函数，封装字段构造逻辑，使调用点更简洁：

```rust
impl TensorError {
    // ── ShapeMismatch ─────────────────────────────────────────────

    /// Creates a `ShapeMismatch` error for incompatible binary operation shapes.
    pub fn shape_mismatch(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self::ShapeMismatch { lhs_shape, rhs_shape }
    }

    // ── BroadcastError ────────────────────────────────────────────

    /// Creates a `BroadcastError` for incompatible broadcast shapes.
    pub fn broadcast_error(
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        axis: usize,
        lhs_dim: usize,
        rhs_dim: usize,
    ) -> Self {
        Self::BroadcastError {
            lhs_shape,
            rhs_shape,
            axis,
            lhs_dim,
            rhs_dim,
        }
    }

    // ── LayoutMismatch ────────────────────────────────────────────

    /// Creates a `LayoutMismatch` error for a contiguous-layout requirement.
    pub fn layout_mismatch(required: &'static str, description: String) -> Self {
        Self::LayoutMismatch {
            required,
            flags: LayoutContext { description },
        }
    }

    /// Creates a `LayoutMismatch` error specifically for reshape on non-contiguous data.
    pub fn layout_mismatch_for_reshape(current_description: String) -> Self {
        Self::layout_mismatch("contiguous", current_description)
    }

    // ── InvalidAxis ───────────────────────────────────────────────

    /// Creates an `InvalidAxis` error.
    pub fn invalid_axis(axis: usize, ndim: usize) -> Self {
        Self::InvalidAxis { axis, ndim }
    }

    // ── InvalidShape ──────────────────────────────────────────────

    /// Creates an `InvalidShape` error for a reshape operation.
    ///
    /// Automatically computes `target_elements` as the product of `target_shape`.
    pub fn invalid_shape(source_elements: usize, target_shape: Vec<usize>) -> Self {
        let target_elements: usize = target_shape.iter().product();
        Self::InvalidShape {
            source_elements,
            target_shape,
            target_elements,
        }
    }

    // ── DimensionMismatch ─────────────────────────────────────────

    /// Creates a `DimensionMismatch` error.
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    // ── IndexOutOfBounds ──────────────────────────────────────────

    /// Creates an `IndexOutOfBounds` error.
    pub fn index_out_of_bounds(index: Vec<usize>, shape: Vec<usize>) -> Self {
        Self::IndexOutOfBounds { index, shape }
    }

    /// Panics with an `IndexOutOfBounds` message.
    ///
    /// Used by the `Index` trait implementation and checked indexing
    /// operators. This is the **panic boundary**: programming errors
    /// (out-of-bounds index) result in panic rather than `Result`.
    pub fn panic_index_out_of_bounds(index: Vec<usize>, shape: Vec<usize>) -> ! {
        panic!(
            "index out of bounds: index {:?} is out of range \
             for array with shape {:?}",
            index, shape
        )
    }

    // ── EmptyArray ────────────────────────────────────────────────

    /// Creates an `EmptyArray` error.
    pub fn empty_array(operation: &'static str) -> Self {
        Self::EmptyArray { operation }
    }
}
```

### 4.5 完整文件概览

`src/error.rs` 文件的完整公共导出清单：

```rust
// src/error.rs — Public exports

pub use self::internal::LayoutContext;
pub type Result<T> = core::result::Result<T, TensorError>;
pub enum TensorError { /* 8 variants */ }
// TensorError impl block with 9 constructor methods
```

---

## 5. 内部实现设计

### 5.1 Error 传播策略

```
调用方                     返回类型               错误处理
─────────────────────────────────────────────────────────────
a + b (二元运算)           Result<Tensor>         ShapeMismatch / BroadcastError
a.reshape([3,4])          Result<Tensor>         InvalidShape / LayoutMismatch
a.sum_axis(0)             Result<Tensor>         InvalidAxis
a.min()                   Result<A>              EmptyArray
a[[i, j]]                 &A (panic)             panic! (IndexOutOfBounds)
a.get(&[i, j])            Result<&A>             IndexOutOfBounds (via Result)
a.get_unchecked(&[i, j])  &A (unsafe)            UB on invalid (no check)
Ix3::try_from(dyn)        Result<Ix3>            DimensionMismatch
```

**传播规则**：

1. **可恢复错误**通过 `?` 运算符向上传播，最终由调用方决定如何处理
2. **编程错误**（索引越界）直接 panic——这是 Rust 的标准约定（`Index` trait 无法返回 `Result`）
3. 所有返回 `Result` 的 API 在文档中标注 `# Errors` 节说明可能返回的错误变体

### 5.2 上下文信息设计

每个错误变体携带 **期望值 vs 实际值** 的对比信息：

| 变体 | 期望（expected） | 实际（actual） |
|------|-------------------|----------------|
| `ShapeMismatch` | `lhs_shape` | `rhs_shape` |
| `BroadcastError` | `lhs_dim` at axis | `rhs_dim` at axis |
| `LayoutMismatch` | `required` layout | `flags` (current) |
| `InvalidAxis` | `0..ndim` (valid range) | `axis` (requested) |
| `InvalidShape` | `source_elements` | `target_elements` |
| `DimensionMismatch` | `expected` ndim | `actual` ndim |
| `IndexOutOfBounds` | `shape` bounds | `index` requested |
| `EmptyArray` | ≥1 elements | 0 elements (implicit) |

### 5.3 Panic vs Result 边界

```
┌──────────────────────────────────────────────────────┐
│                    Xenon API Surface                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────┐  ┌───────────────────────┐ │
│  │   Result 边界       │  │   Panic 边界          │ │
│  │                     │  │                       │ │
│  │  • reshape()        │  │  • index operator []  │ │
│  │  • sum_axis()       │  │  • bincount (neg val) │ │
│  │  • broadcast()      │  │  • integer overflow   │ │
│  │  • min/max/arg*()   │  │                       │ │
│  │  • get()            │  │                       │ │
│  │  • into_dyn()       │  │                       │ │
│  └─────────────────────┘  └───────────────────────┘ │
│                                                      │
│  ┌─────────────────────┐                             │
│  │   unsafe 边界       │                             │
│  │                     │                             │
│  │  • get_unchecked()  │  ← 无检查，UB on invalid   │
│  │  • uget()           │                             │
│  └─────────────────────┘                             │
└──────────────────────────────────────────────────────┘
```

**关键原则**：

- **Panic 场景**在文档中标注 `# Panics` 节
- **unsafe 函数**在文档中标注 `# Safety` 节，说明前置条件
- 所有 panic 场景都有对应的 `Result`-returning 替代方法（如 `get()` 替代 `[]`）
- 所有 `Result`-returning 方法都有对应的 `unsafe` unchecked 变体供性能关键路径使用

### 5.4 Panic 消息格式

索引越界的 panic 消息直接使用内联格式化，不经过 `TensorError::IndexOutOfBounds` 的 `Display` impl（避免 panic 路径上的分配开销）：

```rust
/// Panics with an index-out-of-bounds message.
/// Used by the `Index` trait implementation.
fn assert_index_valid(index: &[usize], shape: &[usize]) {
    for (axis, (&idx, &dim)) in index.iter().zip(shape.iter()).enumerate() {
        assert!(
            idx < dim,
            "index out of bounds: index {:?} has value {} at axis {}, \
             but axis {} has size {} (shape={:?})",
            index, idx, axis, axis, dim, shape
        );
    }
    if index.len() != shape.len() {
        panic!(
            "index out of bounds: index {:?} has {} dimensions, \
             but array has {} dimensions (shape={:?})",
            index, index.len(), shape.len(), shape
        );
    }
}
```

### 5.5 no_std 兼容

```rust
// src/error.rs 顶部导入
use alloc::string::String;
use alloc::vec::Vec;
```

`thiserror` v2 天然支持 no_std。`TensorError` 不依赖任何 `std` 专用类型。

### 5.6 Error trait 额外实现

`thiserror` 自动派生 `Error` trait。额外手动实现以下 trait 以确保完备性：

```rust
// thiserror 自动处理，无需手动实现：
// - impl std::fmt::Display for TensorError
// - impl std::error::Error for TensorError

// 确保 Clone 可用（thiserror 不自动派生 Clone，需手动标注）
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorError { ... }
```

`TensorError` 实现 `Clone`，因为内部字段（`Vec<usize>`, `String`, `&'static str`）均满足 `Clone`。这使得错误可以在多个上下文中传播而不受借用限制。

---

## 6. 实现任务拆分

每个任务约 10 分钟，可独立验证和提交。

### Task 1: 创建 error.rs 骨架 + 模块导入

```
Task: 创建 error.rs 文件骨架
文件: src/error.rs:1-30
测试: 无（编译检查）
前置: 无
预计: 5 min
```

内容：文件级 doc comment、`use alloc::{string::String, vec::Vec}`、`use core::fmt`、空 `enum TensorError` 骨架。

### Task 2: 定义 ShapeMismatch + BroadcastError 变体

```
Task: 实现 ShapeMismatch 和 BroadcastError 变体
文件: src/error.rs
测试: src/error.rs #[cfg(test)] mod tests
前置: Task 1
预计: 10 min
```

内容：两个变体定义、`#[error(...)]` 属性、对应构造方法、单元测试验证 `to_string()` 输出。

### Task 3: 定义 LayoutMismatch 变体 + LayoutContext

```
Task: 实现 LayoutMismatch 变体和 LayoutContext 类型
文件: src/error.rs
测试: src/error.rs #[cfg(test)] mod tests
前置: Task 1
预计: 10 min
```

内容：`LayoutContext` struct + `Display` impl、`LayoutMismatch` 变体、`layout_mismatch()` / `layout_mismatch_for_reshape()` 构造方法、单元测试。

### Task 4: 定义 InvalidAxis + InvalidShape + DimensionMismatch 变体

```
Task: 实现维度相关错误变体
文件: src/error.rs
测试: src/error.rs #[cfg(test)] mod tests
前置: Task 1
预计: 10 min
```

内容：三个变体定义、构造方法、单元测试验证 Display 输出包含期望值/实际值。

### Task 5: 定义 IndexOutOfBounds + EmptyArray 变体

```
Task: 实现索引和空数组错误变体
文件: src/error.rs
测试: src/error.rs #[cfg(test)] mod tests
前置: Task 1
预计: 10 min
```

内容：两个变体定义、`panic_index_out_of_bounds()` 方法、`index_out_of_bounds()` / `empty_array()` 构造方法、单元测试。

### Task 6: 定义 Result<T> 类型别名

```
Task: 定义 Result 类型别名
文件: src/error.rs
测试: src/error.rs #[cfg(test)] mod tests
前置: Task 2-5
预计: 5 min
```

内容：`pub type Result<T> = core::result::Result<T, TensorError>;`、doc comment、验证 `Result` 类型在函数签名中正确使用。

### Task 7: 补全所有变体 doc comments

```
Task: 为所有公共类型添加完整文档注释
文件: src/error.rs
测试: cargo doc --no-deps（验证无 doc warning）
前置: Task 2-6
预计: 10 min
```

内容：模块级 doc comment（含 `//!`）、每个变体的 doc comment（触发场景说明）、每个构造方法的 doc comment（`# Arguments` 节）、`LayoutContext` 的 doc comment。

### Task 8: 注册模块 + re-export

```
Task: 在 lib.rs 中声明 error 模块并 re-export
文件: src/lib.rs
测试: cargo check
前置: Task 6
预计: 5 min
```

内容：`pub mod error;`、`pub use crate::error::{TensorError, Result};`、验证从外部 crate 可访问。

### Task 9: 单元测试 — Display 输出验证

```
Task: 编写 Display 输出验证测试
文件: src/error.rs #[cfg(test)] mod tests
测试: cargo test --lib error
前置: Task 2-5
预计: 10 min
```

内容：为 8 个变体各编写 `to_string()` 测试，验证输出包含关键上下文信息（如 `assert!(msg.contains("expected"))`)。

### Task 10: 单元测试 — Error trait + Clone + Send + Sync

```
Task: 验证 Error/Clone/Send/Sync trait 实现
文件: src/error.rs #[cfg(test)] mod tests
测试: cargo test --lib error
前置: Task 2-6
预计: 5 min
```

内容：编译时断言（`fn _assert_error_impls()` 模式）验证 `TensorError: Error + Clone + Send + Sync`、`Result<T>: Debug` 等。

### Task 11: 单元测试 — 构造方法便利性

```
Task: 测试所有构造方法
文件: src/error.rs #[cfg(test)] mod tests
测试: cargo test --lib error
前置: Task 2-6
预计: 10 min
```

内容：验证每个 `TensorError::xxx()` 构造方法正确设置所有字段、`invalid_shape()` 自动计算 `target_elements`。

### Task 12: 集成测试 — 错误传播端到端

```
Task: 编写错误传播集成测试骨架
文件: tests/error_propagation.rs
测试: cargo test --test error_propagation
前置: Task 8 + tensor 模块就绪后
预计: 10 min
```

内容：验证 `reshape` 返回 `InvalidShape`、`sum_axis` 返回 `InvalidAxis`、错误通过 `?` 正确传播。**注**：此任务依赖 tensor 模块，在 Phase 2 W4 后执行。

---

## 7. 测试计划

### 7.1 单元测试（`src/error.rs` 内 `#[cfg(test)] mod tests`）

| 测试函数 | 覆盖目标 | 断言 |
|----------|----------|------|
| `test_shape_mismatch_display` | `ShapeMismatch` Display | 消息包含 `lhs_shape` 和 `rhs_shape` |
| `test_broadcast_error_display` | `BroadcastError` Display | 消息包含 `axis`, `lhs_dim`, `rhs_dim` |
| `test_layout_mismatch_display` | `LayoutMismatch` Display | 消息包含 `required` 和 flags 描述 |
| `test_invalid_axis_display` | `InvalidAxis` Display | 消息包含 `axis` 和 `ndim` |
| `test_invalid_shape_display` | `InvalidShape` Display | 消息包含元素数对比 |
| `test_invalid_shape_auto_product` | `invalid_shape()` 构造 | 自动计算 `target_elements` |
| `test_dimension_mismatch_display` | `DimensionMismatch` Display | 消息包含 `expected` 和 `actual` |
| `test_index_out_of_bounds_display` | `IndexOutOfBounds` Display | 消息包含 `index` 和 `shape` |
| `test_empty_array_display` | `EmptyArray` Display | 消息包含 `operation` 名称 |
| `test_error_impls_send_sync` | trait 约束 | 编译通过 |
| `test_error_impls_clone` | Clone | `error.clone()` 与原错误相等 |
| `test_layout_context_display` | `LayoutContext` Display | 输出与 description 一致 |

### 7.2 集成测试（`tests/error_propagation.rs`）

> **注**：需在 `tensor` 模块实现后编写。

| 测试函数 | 场景 | 预期 |
|----------|------|------|
| `test_reshape_invalid_shape` | reshape 到不兼容形状 | `Err(InvalidShape)` |
| `test_reshape_non_contiguous` | reshape 非连续视图 | `Err(LayoutMismatch)` |
| `test_sum_axis_invalid` | axis 超出范围 | `Err(InvalidAxis)` |
| `test_min_empty_array` | 对空数组取 min | `Err(EmptyArray)` |
| `test_argmax_empty_array` | 对空数组取 argmax | `Err(EmptyArray)` |
| `test_add_shape_mismatch` | 形状不可广播的加法 | `Err(ShapeMismatch)` 或 `Err(BroadcastError)` |
| `test_index_oob_panics` | `[]` 越界索引 | `#[should_panic]` |
| `test_get_oob_returns_error` | `get()` 越界索引 | `Err(IndexOutOfBounds)` |

### 7.3 测试命名约定

遵循 [00-rust-standards.md](00-rust-standards.md) §5.2：

```rust
#[test]
fn test_shape_mismatch_display_contains_both_shapes() { ... }

#[test]
#[should_panic(expected = "index out of bounds")]
fn test_index_operator_oob_panics() { ... }

#[test]
fn test_reshape_invalid_shape_returns_error() { ... }
```

### 7.4 覆盖率目标

| 指标 | 目标 |
|------|------|
| 枚举变体覆盖 | 8/8 变体均有 Display 测试 |
| 构造方法覆盖 | 9/9 公共方法均有测试 |
| trait impl 覆盖 | `Error`, `Display`, `Clone`, `Debug`, `Send`, `Sync` |
| panic 路径覆盖 | `panic_index_out_of_bounds` 在集成测试中验证 |
