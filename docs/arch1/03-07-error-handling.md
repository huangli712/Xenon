# 03-07 错误处理模块设计

> **模块路径**: `src/error.rs`  
> **版本**: v1.0  
> **日期**: 2026-03-28  
> **依赖**: `core`, `alloc`, `std` (可选 feature)

---

## 1. 模块概述

### 1.1 职责定义

`error.rs` 是 Senon 张量库的错误处理基础设施，负责：

| 职责 | 说明 |
|------|------|
| 错误类型定义 | 定义统一的 `SenonError` 枚举，覆盖所有可恢复错误场景 |
| 错误上下文 | 为每个错误提供期望值与实际值的对比信息 |
| 类型别名 | 提供 `Result<T>` 类型别名，简化 API 签名 |
| no_std 兼容 | 在无 `std` 环境下仅实现 `Display`，在 `std` 环境下实现 `std::error::Error` |

### 1.2 设计目标

| 目标 | 实现方式 |
|------|----------|
| **API 简洁** | 单一错误枚举而非多个错误类型，避免错误类型爆炸 |
| **信息丰富** | 每个错误变体携带上下文（期望 vs 实际），便于调试 |
| **零运行时开销** | 错误类型为枚举，无堆分配（shape 信息使用 `SmallVec` 栈存储） |
| **Rust 惯例一致** | 索引越界采用 panic，与标准库 slice 行为一致 |

### 1.3 错误分类总览

```
错误处理策略
├── 可恢复错误 (Result<SenonError>)
│   ├── ShapeMismatch      — 二元运算形状不兼容
│   ├── BroadcastError     — 广播规则不满足
│   ├── LayoutMismatch     — 布局要求不满足
│   ├── InvalidAxis        — 轴索引无效
│   ├── InvalidShape       — reshape 目标形状无效
│   ├── DimensionMismatch  — 维度类型转换失败
│   └── EmptyArray         — 空数组上的非法操作
│
└── 编程错误 (panic / unchecked)
    └── IndexOutOfBounds   — 索引越界
```

---

## 2. 文件结构

### 2.1 单文件设计

```
src/
├── lib.rs
├── error.rs          ← 错误模块（本设计文档目标）
├── tensor.rs
├── shape.rs
├── stride.rs
└── ...
```

### 2.2 为什么不需要子模块

| 考量 | 结论 |
|------|------|
| **代码量** | 预估约 200-300 行，单文件足够管理 |
| **内聚性** | 所有错误类型紧密相关，共同构成一个概念单元 |
| **依赖关系** | 错误模块仅依赖 `core`/`alloc`，无复杂依赖图 |
| **API 边界** | 对外暴露单一 `SenonError` 类型，无需细分子模块 |
| **维护成本** | 单文件更易于查找和修改，避免过度工程化 |

**设计决策**: 保持 `error.rs` 为单文件模块。若未来错误类型超过 15 个或代码超过 500 行，可考虑按错误类别拆分（如 `error/shape.rs`, `error/layout.rs`）。

---

## 3. 错误枚举设计

### 3.1 SenonError 定义

```rust
use core::fmt;
use alloc::vec::Vec;
use alloc::borrow::Cow;

/// Senon 张量库的统一错误类型。
/// 
/// 所有可恢复错误均通过此枚举表示，提供丰富的上下文信息
/// 以便于调试和错误处理。
#[derive(Debug, Clone, PartialEq)]
pub enum SenonError {
    /// 二元运算或 zip 操作时，两个数组的形状不兼容且无法广播。
    ShapeMismatch {
        /// 期望的形状（通常是左侧操作数或目标形状）
        expected: Cow<'static, [usize]>,
        /// 实际遇到的形状（右侧操作数）
        found: Cow<'static, [usize]>,
    },

    /// 广播操作失败：形状无法按广播规则对齐。
    BroadcastError {
        /// 第一个输入数组的形状
        shape_a: Cow<'static, [usize]>,
        /// 第二个输入数组的形状
        shape_b: Cow<'static, [usize]>,
        /// 失败原因描述（哪个维度无法对齐）
        reason: Cow<'static, str>,
    },

    /// 布局要求不满足：操作要求连续布局但输入为非连续。
    LayoutMismatch {
        /// 期望的布局类型描述
        expected: &'static str,
        /// 实际的布局类型描述
        found: &'static str,
        /// 触发此错误的操作名称
        operation: &'static str,
    },

    /// 轴索引超出数组的维度范围。
    InvalidAxis {
        /// 请求的轴索引
        axis: isize,
        /// 数组的实际维度数
        ndim: usize,
    },

    /// reshape 目标形状的元素总数与源数组不一致。
    InvalidShape {
        /// 期望的元素总数
        expected_len: usize,
        /// 实际的元素总数
        found_len: usize,
        /// 尝试 reshape 到的目标形状
        target_shape: Cow<'static, [usize]>,
    },

    /// 静态维度与动态维度转换时维度数不匹配。
    DimensionMismatch {
        /// 期望的维度数
        expected_ndim: usize,
        /// 实际的维度数
        found_ndim: usize,
    },

    /// 在空数组上执行不支持的操作（如 min/max/argmin/argmax）。
    EmptyArray {
        /// 触发此错误的操作名称
        operation: &'static str,
    },
}
```

### 3.2 各变体详细说明

#### 3.2.1 ShapeMismatch

| 属性 | 说明 |
|------|------|
| **触发场景** | 二元运算（加、减、乘等）或 `zip` 操作时，两个数组形状不兼容且无法广播 |
| **典型示例** | `[3, 4] + [2, 5]` — 形状完全不同且无法广播 |
| **expected** | 左侧操作数的形状，或文档约定的"期望形状" |
| **found** | 右侧操作数的实际形状 |

```rust
// 触发示例
let a = Tensor::<f64, Ix2>::zeros([3, 4]);
let b = Tensor::<f64, Ix2>::zeros([2, 5]);
a.try_add(&b)?;  // Err(ShapeMismatch { expected: [3, 4], found: [2, 5] })
```

#### 3.2.2 BroadcastError

| 属性 | 说明 |
|------|------|
| **触发场景** | 尝试广播两个形状时，存在非 size-1 维度不相等 |
| **典型示例** | `[3, 4]` 与 `[3, 5]` 广播 — 第二维度 4 ≠ 5 且均非 1 |
| **shape_a** | 第一个输入形状 |
| **shape_b** | 第二个输入形状 |
| **reason** | 描述哪个维度无法对齐，如 "dimension 1: 4 != 5 and neither is 1" |

```rust
// 触发示例
let a = Tensor::<f64, Ix2>::zeros([3, 4]);
let b = Tensor::<f64, Ix2>::zeros([3, 5]);
broadcast(&a, &b)?;  // Err(BroadcastError { 
                     //     shape_a: [3, 4], 
                     //     shape_b: [3, 5],
                     //     reason: "dimension 1: 4 != 5"
                     // })
```

#### 3.2.3 LayoutMismatch

| 属性 | 说明 |
|------|------|
| **触发场景** | 操作要求连续布局（如 `reshape`、`as_slice`），但输入为非连续 |
| **典型示例** | 对转置后的数组调用 `reshape` |
| **expected** | 期望的布局类型，如 `"F-contiguous"` 或 `"C-contiguous"` 或 `"contiguous"` |
| **found** | 实际布局，如 `"non-contiguous"` |
| **operation** | 触发此错误的操作，如 `"reshape"`、`"as_slice"` |

```rust
// 触发示例
let a = Tensor::<f64, Ix2>::zeros([3, 4]);
let t = a.t();  // 转置，非连续
t.reshape([12])?;  // Err(LayoutMismatch {
                   //     expected: "contiguous",
                   //     found: "non-contiguous",
                   //     operation: "reshape"
                   // })
```

#### 3.2.4 InvalidAxis

| 属性 | 说明 |
|------|------|
| **触发场景** | 指定的轴索引超出 `[0, ndim)` 范围或负索引超出 `[-ndim, -1]` |
| **典型示例** | 对 2D 数组访问 `axis=3` |
| **axis** | 用户请求的轴索引（支持负索引，保留原始值便于调试） |
| **ndim** | 数组的实际维度数 |

```rust
// 触发示例
let a = Tensor::<f64, Ix2>::zeros([3, 4]);
a.sum_axis(5)?;  // Err(InvalidAxis { axis: 5, ndim: 2 })
a.sum_axis(-3)?; // Err(InvalidAxis { axis: -3, ndim: 2 })
```

#### 3.2.5 InvalidShape

| 属性 | 说明 |
|------|------|
| **触发场景** | `reshape` 目标形状的元素总数与源数组不一致 |
| **典型示例** | 将 12 元素数组 reshape 为 `[3, 5]`（15 元素） |
| **expected_len** | 源数组的元素总数 |
| **found_len** | 目标形状计算出的元素总数 |
| **target_shape** | 用户请求的目标形状 |

```rust
// 触发示例
let a = Tensor::<f64, Ix1>::zeros([12]);
a.reshape([3, 5])?;  // Err(InvalidShape { 
                     //     expected_len: 12,
                     //     found_len: 15,
                     //     target_shape: [3, 5]
                     // })
```

#### 3.2.6 DimensionMismatch

| 属性 | 说明 |
|------|------|
| **触发场景** | 静态维度类型（IxN）与动态维度（IxDyn）互转时维度数不匹配 |
| **典型示例** | 将 `IxDyn` (ndim=3) 转换为 `Ix2` |
| **expected_ndim** | 目标维度类型要求的维度数 |
| **found_ndim** | 源数组的实际维度数 |

```rust
// 触发示例
let a: Tensor<f64, IxDyn> = Tensor::zeros(IxDyn(&[2, 3, 4]));
let b: Tensor<f64, Ix2> = a.into_dimension()?;  
// Err(DimensionMismatch { expected_ndim: 2, found_ndim: 3 })
```

#### 3.2.7 EmptyArray

| 属性 | 说明 |
|------|------|
| **触发场景** | 在空数组上执行 min/max/argmin/argmax 等需要至少一个元素的操作 |
| **典型示例** | 对形状 `[0, 3]` 的数组调用 `argmax()` |
| **operation** | 触发此错误的操作名称 |

```rust
// 触发示例
let a = Tensor::<f64, Ix1>::zeros([0]);
a.argmax()?;  // Err(EmptyArray { operation: "argmax" })
```

### 3.3 为什么 IndexOutOfBounds 不在枚举中

| 原因 | 说明 |
|------|------|
| **Rust 惯例** | 标准库 `slice` 索引越界触发 panic，Senon 保持一致 |
| **性能** | 索引操作频繁，每次返回 `Result` 会增加匹配开销 |
| **语义** | 索引越界属于编程错误（bug），而非可恢复的业务错误 |
| **unchecked 变体** | 对性能敏感的场景提供 `get_unchecked` unsafe 变体 |

```rust
// 索引操作的设计（非错误枚举的一部分）
impl<A, D> Tensor<A, D> {
    /// Checked indexing — 越界时 panic
    pub fn index(&self, index: &[usize]) -> &A { ... }
    
    /// Unchecked indexing — 越界时 UB，用于性能关键路径
    /// 
    /// # Safety
    /// 调用者须保证 `index` 的每个分量均在对应维度范围内。
    pub unsafe fn index_unchecked(&self, index: &[usize]) -> &A { ... }
    
    /// Checked indexing returning Option — 不 panic
    pub fn get(&self, index: &[usize]) -> Option<&A> { ... }
}
```

---

## 4. Display 实现

### 4.1 错误消息格式设计原则

| 原则 | 说明 |
|------|------|
| **期望优先** | 先显示期望值，再显示实际值 |
| **人类可读** | 使用自然语言描述，避免裸数据 |
| **上下文完整** | 包含足够信息让开发者快速定位问题 |
| **一致性** | 所有错误消息遵循统一格式 |

### 4.2 Display 实现

```rust
impl fmt::Display for SenonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, found } => {
                write!(
                    f,
                    "shape mismatch: expected shape {:?}, found {:?}",
                    expected.as_ref(),
                    found.as_ref()
                )
            }

            Self::BroadcastError { shape_a, shape_b, reason } => {
                write!(
                    f,
                    "broadcast failed: shapes {:?} and {:?} cannot be broadcast together: {}",
                    shape_a.as_ref(),
                    shape_b.as_ref(),
                    reason
                )
            }

            Self::LayoutMismatch { expected, found, operation } => {
                write!(
                    f,
                    "layout mismatch during {}: expected {} layout, found {}",
                    operation, expected, found
                )
            }

            Self::InvalidAxis { axis, ndim } => {
                write!(
                    f,
                    "invalid axis: axis {} is out of bounds for array with {} dimension(s)",
                    axis, ndim
                )
            }

            Self::InvalidShape { expected_len, found_len, target_shape } => {
                write!(
                    f,
                    "invalid shape: reshape target {:?} has {} elements, but source has {} elements",
                    target_shape.as_ref(), found_len, expected_len
                )
            }

            Self::DimensionMismatch { expected_ndim, found_ndim } => {
                write!(
                    f,
                    "dimension mismatch: expected {} dimension(s), found {} dimension(s)",
                    expected_ndim, found_ndim
                )
            }

            Self::EmptyArray { operation } => {
                write!(
                    f,
                    "empty array: cannot perform '{}' on an empty array",
                    operation
                )
            }
        }
    }
}
```

### 4.3 错误消息示例

| 错误类型 | 输出示例 |
|----------|----------|
| ShapeMismatch | `shape mismatch: expected shape [3, 4], found [2, 5]` |
| BroadcastError | `broadcast failed: shapes [3, 4] and [3, 5] cannot be broadcast together: dimension 1: 4 != 5 and neither is 1` |
| LayoutMismatch | `layout mismatch during reshape: expected contiguous layout, found non-contiguous` |
| InvalidAxis | `invalid axis: axis 5 is out of bounds for array with 2 dimension(s)` |
| InvalidShape | `invalid shape: reshape target [3, 5] has 15 elements, but source has 12 elements` |
| DimensionMismatch | `dimension mismatch: expected 2 dimension(s), found 3 dimension(s)` |
| EmptyArray | `empty array: cannot perform 'argmax' on an empty array` |

---

## 5. Result 类型别名

### 5.1 定义

```rust
/// Senon 库的 Result 类型别名。
/// 
/// 使用 SenonError 作为统一错误类型，简化函数签名。
pub type Result<T> = core::result::Result<T, SenonError>;
```

### 5.2 使用示例

```rust
// 函数签名简化
pub fn reshape<D2: Dimension>(&self, shape: D2) -> Result<Tensor<A, D2>>;

// 用户代码
match tensor.reshape([3, 4]) {
    Ok(new_tensor) => { /* ... */ },
    Err(SenonError::InvalidShape { expected_len, found_len, .. }) => {
        eprintln!("Cannot reshape: {} elements != {}", expected_len, found_len);
    }
    Err(e) => return Err(e),
}
```

### 5.3 为什么使用类型别名而非新类型

| 考量 | 结论 |
|------|------|
| **与 `?` 运算符兼容** | 类型别名可直接使用 `?`，无需 `Into` 转换 |
| **与 `std::result::Result` 方法兼容** | 保留所有内置方法（`map`, `and_then`, `unwrap_or` 等） |
| **文档简洁** | `Result<T>` 比 `SenonResult<T>` 更符合 Rust 惯例 |
| **无额外抽象** | 类型别名零运行时开销 |

---

## 6. panic vs Result 边界

### 6.1 决策矩阵

| 错误类型 | 处理方式 | 理由 |
|----------|----------|------|
| ShapeMismatch | Result | 运行时数据决定，可恢复 |
| BroadcastError | Result | 运行时数据决定，可恢复 |
| LayoutMismatch | Result | 运行时状态决定，可恢复 |
| InvalidAxis | Result | 可能来自用户输入，可恢复 |
| InvalidShape | Result | 可能来自用户输入，可恢复 |
| DimensionMismatch | Result | 类型转换失败，可恢复 |
| EmptyArray | Result | 运行时状态决定，可恢复 |
| **IndexOutOfBounds** | **panic** | **编程错误，与 Rust slice 一致** |

### 6.2 IndexOutOfBounds 为何使用 panic

| 理由 | 说明 |
|------|------|
| **Rust 标准库惯例** | `[i]` 越界 panic，`get(i)` 返回 `Option` |
| **性能考虑** | 索引操作频繁，`Result` 匹配有开销 |
| **语义正确性** | 索引越界是 bug，不应"处理"而应"修复" |
| **API 一致性** | `tensor[[i, j]]` 与 `array[i][j]` 行为一致 |

### 6.3 Checked/Unchecked 变体规则

```rust
impl<A, D> Tensor<A, D> {
    // 1. 索引运算符 — panic on out of bounds
    // tensor[[i, j]]  -> panic if out of bounds
    
    // 2. Checked 方法 — 返回 Option
    pub fn get(&self, index: &[usize]) -> Option<&A>;
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut A>;
    
    // 3. Unchecked 方法 — unsafe, UB on out of bounds
    /// # Safety
    /// index 必须在合法范围内
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A;
}
```

### 6.4 API 命名约定

| 后缀 | 语义 | 示例 |
|------|------|------|
| 无后缀 | panic on error | `tensor[[i, j]]`, `tensor.index(&[i, j])` |
| `_checked` 或 `get_` | 返回 Option/Result | `tensor.get(&[i, j])` |
| `_unchecked` | unsafe, UB on error | `unsafe { tensor.get_unchecked(&[i, j]) }` |
| `try_` | 返回 Result | `tensor.try_reshape(shape)` |

---

## 7. no_std 兼容性

### 7.1 条件编译策略

```rust
use core::fmt;
use alloc::vec::Vec;
use alloc::borrow::Cow;

// SenonError 定义 (使用 core::fmt 和 alloc)
// ...

impl fmt::Display for SenonError {
    // 使用 core::fmt，无需 std
}

// std feature 下额外实现 std::error::Error
#[cfg(feature = "std")]
impl std::error::Error for SenonError {}
```

### 7.2 为什么 no_std 下不实现 Error trait

| 原因 | 说明 |
|------|------|
| **std::error::Error 在 core 中不可用** | `Error` trait 定义在 `std::error` 模块 |
| **no_std 生态的 Display 约定** | `anyhow`、`eyre` 等 no_std 兼容库仅要求 `Display` |
| **避免 no_std 依赖膨胀** | 不引入 `error-in-core` 等第三方 crate |

### 7.3 Cargo.toml 配置

```toml
[features]
default = ["std"]
std = []
```

### 7.4 用户使用场景

```rust
// std 环境（默认）
use Senon::{Tensor, SenonError, Result};

fn main() -> Result<()> {
    let a = Tensor::zeros([3, 4]);
    let b = a.reshape([2, 6])?;  // SenonError impl std::error::Error
    Ok(())
}

// no_std 环境
#![no_std]

extern crate alloc;

use Senon::{Tensor, SenonError};

fn process() -> Result<Tensor<f64, Ix2>, SenonError> {
    let a = Tensor::zeros([3, 4]);
    a.reshape([2, 6])  // SenonError 仅 impl Display
}
```

---

## 8. 与其他模块的交互

### 8.1 错误产生模块映射

| 模块 | 产生的错误类型 | 触发场景 |
|------|----------------|----------|
| `shape.rs` | DimensionMismatch | IxN ↔ IxDyn 转换 |
| `tensor.rs` | InvalidShape | reshape 操作 |
| `tensor.rs` | LayoutMismatch | reshape 非连续数组 |
| `tensor.rs` | EmptyArray | min/max/argmin/argmax |
| `ops/binary.rs` | ShapeMismatch | 二元运算形状不兼容 |
| `ops/binary.rs` | BroadcastError | 广播失败 |
| `ops/reduce.rs` | InvalidAxis | 归约操作轴索引错误 |
| `ops/reduce.rs` | EmptyArray | 空数组归约 |
| `linalg/*.rs` | LayoutMismatch | BLAS 操作要求连续布局 |
| `index.rs` | (panic) IndexOutOfBounds | 索引越界 |

### 8.2 错误流向图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户代码                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Senon 公共 API                               │
│  tensor.reshape() / tensor.sum_axis() / a + b / ...             │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ shape.rs │   │ ops/*.rs │   │tensor.rs │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             │ DimensionMismatch           │ InvalidShape
             │              │ LayoutMismatch
             │              │ EmptyArray
             │ BroadcastError
             │ ShapeMismatch
             │ InvalidAxis
             │
             └──────────────┼──────────────┘
                            ▼
                   ┌─────────────────┐
                   │   SenonError    │
                   │   (error.rs)    │
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Result<T>     │
                   └─────────────────┘
```

### 8.3 模块依赖关系

```rust
// lib.rs
mod error;       // 无内部依赖，最先加载

mod shape;       // 依赖 error::DimensionMismatch
mod stride;      // 无错误类型
mod layout;      // 无错误类型
mod tensor;      // 依赖 error::*
mod ops;         // 依赖 error::ShapeMismatch, BroadcastError, InvalidAxis
mod linalg;      // 依赖 error::LayoutMismatch
```

---

## 9. 实现任务分解

### 9.1 任务列表

每个任务预估约 10-15 分钟。

| 任务 | 内容 | 预估时间 | 依赖 |
|------|------|----------|------|
| **Task 1** | 定义 `SenonError` 枚举及所有变体 | 15 min | 无 |
| **Task 2** | 实现 `Display` trait | 10 min | Task 1 |
| **Task 3** | 实现 `std::error::Error` (cfg-gated) | 5 min | Task 2 |
| **Task 4** | 定义 `Result<T>` 类型别名 | 2 min | Task 1 |
| **Task 5** | 实现 `From` traits (如 `From<Infallible>`) | 5 min | Task 1 |
| **Task 6** | 单元测试：Display 输出格式 | 15 min | Task 2 |
| **Task 7** | 单元测试：Error trait (std feature) | 10 min | Task 3 |
| **Task 8** | 文档注释 | 10 min | Task 1-4 |
| **Task 9** | 集成测试：与其他模块交互 | 15 min | Task 1-8 |
| **Task 10** | CI 配置：no_std 编译检查 | 5 min | Task 1-9 |

### 9.2 任务详情

#### Task 1: 定义 SenonError 枚举

```rust
// 预期产出
pub enum SenonError {
    ShapeMismatch { expected: Cow<'static, [usize]>, found: Cow<'static, [usize]> },
    BroadcastError { shape_a: Cow<'static, [usize]>, shape_b: Cow<'static, [usize]>, reason: Cow<'static, str> },
    LayoutMismatch { expected: &'static str, found: &'static str, operation: &'static str },
    InvalidAxis { axis: isize, ndim: usize },
    InvalidShape { expected_len: usize, found_len: usize, target_shape: Cow<'static, [usize]> },
    DimensionMismatch { expected_ndim: usize, found_ndim: usize },
    EmptyArray { operation: &'static str },
}
```

**验证点**:
- [ ] 枚举编译通过
- [ ] 所有变体字段类型正确
- [ ] `#[derive(Debug, Clone, PartialEq)]` 正确

#### Task 2: 实现 Display trait

**验证点**:
- [ ] 每个变体有清晰的错误消息
- [ ] 消息包含期望值和实际值
- [ ] 格式一致性

#### Task 3: 实现 std::error::Error

```rust
#[cfg(feature = "std")]
impl std::error::Error for SenonError {}
```

**验证点**:
- [ ] `#[cfg(feature = "std")]` 正确
- [ ] std feature 下可编译
- [ ] no_std 下不引入 std

#### Task 4: Result 类型别名

```rust
pub type Result<T> = core::result::Result<T, SenonError>;
```

#### Task 5: From traits

```rust
impl From<core::convert::Infallible> for SenonError {
    fn from(e: core::convert::Infallible) -> Self {
        match e {}
    }
}
```

#### Task 6: Display 测试

```rust
#[test]
fn test_shape_mismatch_display() {
    let err = SenonError::ShapeMismatch {
        expected: Cow::Borrowed(&[3, 4]),
        found: Cow::Borrowed(&[2, 5]),
    };
    assert_eq!(
        err.to_string(),
        "shape mismatch: expected shape [3, 4], found [2, 5]"
    );
}

// 类似测试覆盖所有变体...
```

#### Task 7: Error trait 测试

```rust
#[cfg(feature = "std")]
#[test]
fn test_error_trait() {
    let err = SenonError::EmptyArray { operation: "argmax" };
    let _: &dyn std::error::Error = &err;  // 确保实现 Error trait
}
```

#### Task 8: 文档注释

- [ ] 枚举级别文档
- [ ] 每个变体的文档
- [ ] 字段含义说明
- [ ] 使用示例

#### Task 9: 集成测试

```rust
#[test]
fn test_reshape_invalid_shape() {
    let a = Tensor::<f64, _>::zeros([3, 4]);
    let result = a.reshape([2, 5]);
    assert!(matches!(result, Err(SenonError::InvalidShape { .. })));
}
```

#### Task 10: CI 配置

```yaml
# .github/workflows/ci.yml
- name: Check no_std
  run: cargo build --no-default-features --target thumbv6m-none-eabi
```

---

## 10. 设计决策记录

### 10.1 为什么使用单一枚举

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **单一枚举 SenonError** | API 简单、模式匹配完整、无错误类型爆炸 | 枚举可能变大 | **采用** |
| 多个错误类型 (ShapeError, LayoutError...) | 语义分离更清晰 | 错误类型爆炸、`?` 需转换、API 复杂 | 不采用 |
| 使用 `thiserror` 宏 | 减少样板代码 | 引入依赖、no_std 需额外处理 | 可选优化 |

**理由**: Senon 错误类型数量可控（7 个），单一枚举足以管理。用户可通过模式匹配精确处理特定错误，无需处理多个错误类型间的转换。

### 10.2 为什么 shape 信息使用 Cow<'static, [usize]>

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **Cow<'static, [usize]>** | 静态形状零分配、动态形状可持有 | 略复杂 | **采用** |
| Vec<usize> | 简单 | 总是堆分配 | 不采用 |
| SmallVec<[usize; 6]> | 小形状栈分配 | 引入 smallvec 依赖 | 可选优化 |
| 固定数组 [usize; N] | 无分配 | 不支持动态维度 | 不采用 |

**理由**: `Cow` 允许在编译期已知形状时使用借用（零分配），运行时动态形状时使用拥有。平衡了性能与灵活性。

### 10.3 为什么 IndexOutOfBounds 使用 panic

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **panic** | 与 Rust 标准库一致、性能好 | 不可恢复 | **采用** |
| Result | 可恢复 | 与 slice 行为不一致、性能开销 | 不采用 |
| panic + unchecked | 平衡安全与性能 | 需要 unsafe | **配套提供** |

**理由**: 索引越界是编程错误而非业务错误，应通过代码修复而非运行时处理。提供 `get()` 返回 `Option` 供需要安全索引的场景使用。

---

## 11. 参考资料

- [Rust Error Handling Survey](https://blog.burntsushi.net/rust-error-handling/)
- [ndarray Error Types](https://docs.rs/ndarray/latest/ndarray/error/index.html)
- [std::error::Error Trait](https://doc.rust-lang.org/std/error/trait.Error.html)
- [Cow Documentation](https://doc.rust-lang.org/std/borrow/enum.Cow.html)

---

*Senon 错误处理模块设计 — v1.0*
