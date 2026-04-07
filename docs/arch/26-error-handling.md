# 错误处理模块设计

> 文档编号: 26 | 模块: `src/error.rs` | 阶段: Phase 1
> 前置文档: `00-coding-standards.md`
> 需求参考: 需求说明书 §27

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 统一错误枚举 | `XenonError` 枚举，覆盖所有可恢复错误场景 | 具体业务逻辑（运算、索引等） |
| 错误上下文 | 每个错误变体携带期望值与实际值 | 日志系统、错误追踪 |
| Result 类型别名 | `type Result<T> = core::result::Result<T, XenonError>` | 自定义 Result 扩展方法 |
| Display 实现 | 所有变体的人类可读错误消息 | 结构化错误序列化 |
| std::error::Error 实现 | `#[cfg(feature = "std")]` 条件编译 | no_std 下的 Error trait（core 中不可用） |
| 错误分类规则 | 可恢复→Result、不可恢复→panic 的决策矩阵 | 运行时错误恢复策略 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| API 简洁 | 单一错误枚举，避免错误类型爆炸 |
| 信息丰富 | 每个变体携带上下文（期望 vs 实际） |
| 零堆分配 | shape 使用 `Cow<'static, [usize]>`，静态形状零分配 |
| no_std 友好 | 仅依赖 `core` + `alloc` |
| Rust 惯例一致 | 索引越界使用 panic，与标准库 slice 行为一致 |

---

## 2. 文件位置

```
src/
├── lib.rs
├── error.rs          ← 错误模块（本设计文档目标）
├── dimension/
├── element/
├── ...
```

单文件设计：预估约 200-300 行，所有错误类型紧密相关共同构成一个概念单元，仅依赖 `core`/`alloc`，无复杂依赖图。若未来错误变体超过 15 个或代码超过 500 行，可考虑按错误类别拆分。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
src/error.rs
├── core::fmt          # Display trait
├── core::result       # Result<T, E>
├── alloc::borrow      # Cow<'static, [usize]>, Cow<'static, str>
├── alloc::vec         # Vec<usize> (Cow::Owned 变体)
└── alloc::string      # String (Cow::Owned 变体)
```

### 3.2 依赖精确到类型级

| 来源 | 使用的类型/trait |
|------|-----------------|
| `core::fmt` | `fmt::Display`, `fmt::Formatter`, `fmt::Result` |
| `core::result` | `Result<T, E>`（用于类型别名） |
| `alloc::borrow` | `Cow<'static, [usize]>` |
| `alloc::vec` | `Vec<usize>`（`Cow::Owned` 内部使用） |
| `alloc::string` | `String`（`Cow::Owned` 内部使用） |
| `std::error` | `Error` trait（仅 `#[cfg(feature = "std")]`） |

### 3.3 依赖方向声明

> **依赖方向：无内部依赖。** `error.rs` 是整个项目的 L0 层，不依赖 crate 内任何其他模块。
> 被所有下游模块消费：`dimension`、`element`、`layout`、`storage`、`tensor`、`ops`、`shape_ops`、`index` 等。

---

## 4. 公共 API 设计

### 4.1 XenonError 枚举完整定义

```rust
use alloc::borrow::Cow;
use core::fmt;

/// Unified error type for all Xenon operations.
///
/// All recoverable errors are represented through this enum,
/// providing rich context information for debugging.
#[derive(Debug, Clone, PartialEq)]
pub enum XenonError {
    /// Binary operation or zip shapes are incompatible and cannot broadcast.
    ShapeMismatch {
        /// Expected shape (typically left operand or target shape).
        expected: Cow<'static, [usize]>,
        /// Actual shape encountered (right operand).
        actual: Cow<'static, [usize]>,
    },

    /// Broadcast rule violated: non-size-1 dimensions differ.
    BroadcastError {
        /// Shape of the first input array.
        shape_a: Cow<'static, [usize]>,
        /// Shape of the second input array.
        shape_b: Cow<'static, [usize]>,
    },

    /// Contiguous layout required but input is non-contiguous.
    LayoutMismatch {
        /// Expected layout description (e.g., "F-contiguous", "contiguous").
        expected: &'static str,
        /// Actual layout description (e.g., "non-contiguous").
        actual: &'static str,
    },

    /// Axis index exceeds the number of dimensions.
    InvalidAxis {
        /// Requested axis index.
        axis: usize,
        /// Actual number of dimensions in the array.
        ndim: usize,
    },

    /// Reshape target total size differs from source.
    InvalidShape {
        /// Source element count.
        from: usize,
        /// Target element count.
        to: usize,
    },

    /// Static/dynamic dimension conversion mismatch.
    DimensionMismatch {
        /// Expected number of dimensions.
        expected: usize,
        /// Actual number of dimensions.
        actual: usize,
    },

    /// Operation requires a non-empty array (e.g., dot on empty).
    EmptyArray,

    /// Integer reduction overflow.
    OverflowError,
}
```

### 4.2 Result 类型别名

```rust
/// Convenience type alias for Xenon operations.
pub type Result<T> = core::result::Result<T, XenonError>;
```

### 4.3 Display 实现

```rust
impl fmt::Display for XenonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "shape mismatch: expected [{}], got [{}]",
                    fmt_shape(expected),
                    fmt_shape(actual)
                )
            }
            Self::BroadcastError { shape_a, shape_b } => {
                write!(
                    f,
                    "cannot broadcast [{}] with [{}]",
                    fmt_shape(shape_a),
                    fmt_shape(shape_b)
                )
            }
            Self::LayoutMismatch { expected, actual } => {
                write!(
                    f,
                    "layout mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::InvalidAxis { axis, ndim } => {
                write!(
                    f,
                    "axis {} out of bounds for {}-dimensional array",
                    axis, ndim
                )
            }
            Self::InvalidShape { from, to } => {
                write!(
                    f,
                    "cannot reshape {} elements into {}",
                    from, to
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::EmptyArray => {
                write!(f, "operation requires a non-empty array")
            }
            Self::OverflowError => {
                write!(f, "integer overflow in reduction")
            }
        }
    }
}

/// Formats a shape slice as "2, 3, 4" for human-readable output.
fn fmt_shape(s: &[usize]) -> alloc::string::String {
    use alloc::string::ToString;
    s.iter()
        .map(|d| d.to_string())
        .collect::<alloc::vec::Vec<_>>()
        .join(", ")
}
```

### 4.4 std::error::Error 实现（条件编译）

```rust
#[cfg(feature = "std")]
impl std::error::Error for XenonError {}
```

### 4.5 Display 输出示例

| 错误类型 | 输出示例 |
|----------|----------|
| `ShapeMismatch` | `shape mismatch: expected [3, 4], got [2, 5]` |
| `BroadcastError` | `cannot broadcast [3, 4] with [3, 5]` |
| `LayoutMismatch` | `layout mismatch: expected F-contiguous, got non-contiguous` |
| `InvalidAxis` | `axis 5 out of bounds for 2-dimensional array` |
| `InvalidShape` | `cannot reshape 12 elements into 15` |
| `DimensionMismatch` | `dimension mismatch: expected 2, got 3` |
| `EmptyArray` | `operation requires a non-empty array` |
| `OverflowError` | `integer overflow in reduction` |

### 4.6 Good / Bad 对比示例

```rust
// Good - 使用 ? 和 XenonError
pub fn reshape<D2>(self, shape: D2) -> Result<Tensor<A, D2>> {
    if self.len() != shape.size() {
        return Err(XenonError::InvalidShape {
            from: self.len(),
            to: shape.size(),
        });
    }
    // ...
}

// Bad - 库代码中使用 unwrap
pub fn sum_bad(&self) -> A {
    let first = self.first().unwrap();  // 禁止
    self.iter().fold(*first, |acc, x| acc + x)
}
```

```rust
// Good - 整数溢出使用 checked 算术
pub fn sum_checked(&self) -> Result<A> {
    let mut total = A::zero();
    for &x in self.iter() {
        total = total.checked_add(&x).ok_or(XenonError::OverflowError)?;
    }
    Ok(total)
}

// Bad - 整数溢出静默 wrapping
pub fn sum_bad(&self) -> A {
    self.iter().fold(A::zero(), |acc, &x| acc + x)  // release 下静默 wrapping
}
```

---

## 5. 内部实现设计

### 5.1 错误分类规则

```
错误处理策略
├── 可恢复错误 (Result<XenonError>)
│   ├── ShapeMismatch      — 二元运算形状不兼容
│   ├── BroadcastError     — 广播规则不满足
│   ├── LayoutMismatch     — 布局要求不满足
│   ├── InvalidAxis        — 轴索引无效
│   ├── InvalidShape       — reshape 目标形状无效
│   ├── DimensionMismatch  — 维度类型转换失败
│   ├── EmptyArray         — 空数组上的非法操作
│   └── OverflowError      — 整数归约溢出
│
└── 编程错误 (panic / unchecked)
    └── IndexOutOfBounds   — 索引越界（与 std slice 行为一致）
```

### 5.2 panic vs Result 决策矩阵

| 错误类型 | 处理方式 | 理由 |
|----------|----------|------|
| ShapeMismatch | Result | 运行时数据决定，可恢复 |
| BroadcastError | Result | 运行时数据决定，可恢复 |
| LayoutMismatch | Result | 运行时状态决定，可恢复 |
| InvalidAxis | Result | 可能来自用户输入，可恢复 |
| InvalidShape | Result | 可能来自用户输入，可恢复 |
| DimensionMismatch | Result | 类型转换失败，可恢复 |
| EmptyArray | Result | 运行时状态决定，可恢复 |
| OverflowError | Result | 整数归约溢出，可恢复 |
| **IndexOutOfBounds** | **panic** | **编程错误，与 Rust slice 一致** |

### 5.3 IndexOutOfBounds 为何使用 panic

| 理由 | 说明 |
|------|------|
| Rust 标准库惯例 | `[i]` 越界 panic，`get(i)` 返回 `Option` |
| 性能考虑 | 索引操作频繁，`Result` 匹配有开销 |
| 语义正确性 | 索引越界是 bug，不应"处理"而应"修复" |
| API 一致性 | `tensor[[i, j]]` 与 `array[i][j]` 行为一致 |

配套提供 `get()` → `Option` 和 `get_unchecked()` → unsafe 变体。

### 5.4 并行操作中错误立即传播

并行操作中发生不可恢复错误时须立即传播，不得静默忽略：

```rust
// Good - 并行归约中 panic 立即传播
#[cfg(feature = "parallel")]
pub fn par_sum(&self) -> A
where
    A: Numeric + Send,
{
    self.par_iter().fold(|| A::zero(), |acc, &x| acc + x)
        .reduce(|| A::zero(), |a, b| a + b)
}
```

### 5.5 资源释放不得 panic（Drop 安全）

所有 `Drop` 实现不得 panic，确保即使在其他 panic 过程中也能安全清理：

```rust
// Good - Drop 不 panic
impl<A> Drop for OwnedRepr<A> {
    fn drop(&mut self) {
        // SAFETY: ptr and len are valid by construction
        unsafe {
            let slice = core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len);
            self.ptr.drop_in_place(slice);
        }
        // deallocation may panic in OOM, but we catch_unwind in Drop context
        if self.cap > 0 {
            alloc::alloc::dealloc(
                self.ptr.as_ptr() as *mut u8,
                alloc::alloc::Layout::array::<A>(self.cap).unwrap_or_else(|_| {
                    // Layout overflow — memory was never allocated
                    alloc::alloc::Layout::new::<A>()
                }),
            );
        }
    }
}
```

### 5.6 no_std 兼容性

```rust
// src/error.rs
use core::fmt;
use alloc::borrow::Cow;

// XenonError 定义（使用 core::fmt 和 alloc）
// ...

impl fmt::Display for XenonError {
    // 使用 core::fmt，无需 std
}

// 仅在 std feature 下实现 std::error::Error
#[cfg(feature = "std")]
impl std::error::Error for XenonError {}
```

---

## 6. 与其他模块的交互

### 6.1 错误产生模块映射

| 模块 | 产生的错误类型 | 触发场景 |
|------|----------------|----------|
| `dimension/` | `DimensionMismatch` | IxN ↔ IxDyn 转换 |
| `tensor/` | `InvalidShape` | reshape 操作 |
| `tensor/` | `LayoutMismatch` | reshape 非连续数组 |
| `tensor/` | `EmptyArray` | dot 等操作对空数组 |
| `ops/` | `ShapeMismatch` | 二元运算形状不兼容 |
| `ops/` | `BroadcastError` | 广播失败 |
| `ops/` | `InvalidAxis` | sum_axis 轴索引错误 |
| `ops/` | `OverflowError` | 整数 sum 归约溢出 |
| `layout/` | `LayoutMismatch` | 连续性检查失败 |
| `index/` | (panic) IndexOutOfBounds | 索引越界 |

### 6.2 错误流向图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户代码                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Xenon 公共 API                               │
│  tensor.reshape() / tensor.sum_axis() / a + b / dot()          │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │dimension │   │  ops/    │   │ tensor/  │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             │ DimensionMismatch           │ InvalidShape
             │              │ LayoutMismatch
             │              │ EmptyArray
             │ BroadcastError
             │ ShapeMismatch
             │ InvalidAxis
             │ OverflowError
             │
             └──────────────┼──────────────┘
                            ▼
                   ┌─────────────────┐
                   │   XenonError    │
                   │   (error.rs)    │
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Result<T>     │
                   └─────────────────┘
```

---

## 7. 实现任务拆分

### Wave 1: XenonError 核心

- [ ] **T1**: 定义 `XenonError` 枚举及所有变体
  - 文件: `src/error.rs`
  - 内容: 8 个变体（ShapeMismatch, BroadcastError, LayoutMismatch, InvalidAxis, InvalidShape, DimensionMismatch, EmptyArray, OverflowError），derive Debug/Clone/PartialEq
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 定义 `Result<T>` 类型别名
  - 文件: `src/error.rs`
  - 内容: `pub type Result<T> = core::result::Result<T, XenonError>;`
  - 测试: 编译通过
  - 前置: T1
  - 预计: 2 min

- [ ] **T3**: 实现 `Display` trait
  - 文件: `src/error.rs`
  - 内容: 所有 8 个变体的 Display 实现 + `fmt_shape` 辅助函数
  - 测试: 各变体 to_string() 输出正确
  - 前置: T1
  - 预计: 10 min

### Wave 2: std 集成与文档

- [ ] **T4**: 实现 `#[cfg(feature = "std")] std::error::Error`
  - 文件: `src/error.rs`
  - 内容: 条件编译的 Error trait 实现
  - 测试: `dyn std::error::Error` 向上转型成功
  - 前置: T3
  - 预计: 3 min

- [ ] **T5**: 完整文档注释
  - 文件: `src/error.rs`
  - 内容: 枚举级文档 + 每个变体文档 + 每个字段文档
  - 测试: `cargo doc` 无 missing_docs 警告
  - 前置: T1-T4
  - 预计: 10 min

### Wave 3: 测试

- [ ] **T6**: 单元测试 — Display 输出格式
  - 文件: `src/error.rs` (`#[cfg(test)] mod tests`)
  - 内容: 每个变体的 `to_string()` 断言
  - 测试: 8 个测试函数
  - 前置: T3
  - 预计: 10 min

- [ ] **T7**: 单元测试 — Error trait（std feature）
  - 文件: `src/error.rs` (`#[cfg(test)] mod tests`)
  - 内容: `#[cfg(feature = "std")]` 下验证 Error trait 实现
  - 测试: `test_error_trait_std`
  - 前置: T4
  - 预计: 5 min

- [ ] **T8**: 单元测试 — no_std 编译检查
  - 文件: `src/error.rs`
  - 内容: `--no-default-features` 编译通过
  - 测试: `cargo build --no-default-features`
  - 前置: T1-T5
  - 预计: 5 min

### 并行执行分组图

```
Wave 1: [T1]
           │
           ├──▶ [T2]
           │
           └──▶ [T3]
                   │
Wave 2: [T4] ◄─────┘
           │
           └──▶ [T5]
                   │
Wave 3: ┌──[T6]────┤
        │          │
        ├──[T7]────┤
        │          │
        └──[T8] ◄──┘
```

---

## 8. 测试计划

### 8.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_shape_mismatch_display` | ShapeMismatch 格式化输出 | 高 |
| `test_broadcast_error_display` | BroadcastError 格式化输出 | 高 |
| `test_layout_mismatch_display` | LayoutMismatch 格式化输出 | 高 |
| `test_invalid_axis_display` | InvalidAxis 格式化输出 | 高 |
| `test_invalid_shape_display` | InvalidShape 格式化输出 | 高 |
| `test_dimension_mismatch_display` | DimensionMismatch 格式化输出 | 高 |
| `test_empty_array_display` | EmptyArray 格式化输出 | 高 |
| `test_overflow_error_display` | OverflowError 格式化输出 | 高 |
| `test_error_trait_std` | std feature 下 Error trait 实现 | 中 |
| `test_result_type_alias` | Result<T> 类型别名可用性 | 中 |
| `test_partial_eq` | PartialEq 可正确比较相同/不同变体 | 中 |
| `test_clone` | Clone 产生相等副本 | 低 |

### 8.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空形状 `Cow::Borrowed(&[])` | Display 输出 `[]` |
| 大形状 `Cow::Owned(vec![1000000, 1000000])` | Display 正确格式化 |
| 多次 Display 格式化 | 结果一致（无内部状态变化） |

### 8.3 集成测试场景

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_reshape_invalid_shape_returns_error` | reshape 元素数不匹配返回 InvalidShape | 高 |
| `test_sum_axis_invalid_axis_returns_error` | sum_axis 轴越界返回 InvalidAxis | 高 |
| `test_dot_empty_array_returns_error` | dot 空数组返回 EmptyArray | 高 |
| `test_broadcast_incompatible_returns_error` | 不兼容广播返回 BroadcastError | 高 |

---

## 9. 设计决策记录

### 决策 1：单一枚举 vs 多错误类型

| 属性 | 值 |
|------|-----|
| 决策 | 使用单一 `XenonError` 枚举 |
| 理由 | API 简单、模式匹配完整、无错误类型爆炸、`?` 无需转换 |
| 替代方案 | 多个错误类型（ShapeError, LayoutError, ...） — 放弃，增加 API 复杂度 |
| 替代方案 | 使用 thiserror 宏 — 放弃，引入外部依赖，与最小依赖原则冲突 |

### 决策 2：shape 信息使用 Cow\<'static, [usize]\>

| 属性 | 值 |
|------|-----|
| 决策 | 使用 `Cow<'static, [usize]>` 存储 shape 信息 |
| 理由 | 静态形状零分配（`Cow::Borrowed`）；动态形状可持有（`Cow::Owned`） |
| 替代方案 | `Vec<usize>` — 放弃，总是堆分配 |
| 替代方案 | `SmallVec<[usize; 6]>` — 放弃，引入 smallvec 依赖 |
| 替代方案 | 固定数组 `[usize; N]` — 放弃，不支持动态维度 |

### 决策 3：IndexOutOfBounds 使用 panic

| 属性 | 值 |
|------|-----|
| 决策 | 索引越界使用 panic，不纳入 XenonError |
| 理由 | 与 Rust 标准库 slice 行为一致；索引操作频繁，Result 有性能开销；语义上属于编程错误 |
| 替代方案 | Result — 放弃，与 slice 行为不一致 |
| 配套措施 | 提供 `get()` → `Option` 和 `get_unchecked()` → unsafe 变体 |

### 决策 4：整数溢出使用 Result

| 属性 | 值 |
|------|-----|
| 决策 | 整数归约溢出返回 `XenonError::OverflowError`，而非 panic 或静默 wrapping |
| 理由 | 需求说明书 §14 规定整数归约溢出视为不可恢复错误，但使用 Result 更安全；debug 和 release 行为一致 |
| 替代方案 | panic — 放弃，可在 drop 过程中引发 double panic |
| 替代方案 | wrapping — 放弃，静默错误违背正确性优先原则 |

### 决策 5：LayoutMismatch 字段使用 &'static str

| 属性 | 值 |
|------|-----|
| 决策 | `LayoutMismatch` 的 `expected` 和 `actual` 字段使用 `&'static str` |
| 理由 | 布局描述为固定字符串（"F-contiguous"、"non-contiguous" 等），无需动态分配 |
| 替代方案 | `Cow<'static, str>` — 放弃，布局描述无需动态生成 |
| 替代方案 | 自定义 Layout 枚举 — 放弃，过度工程化 |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
