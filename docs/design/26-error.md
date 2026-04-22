# 错误处理模块设计

> 文档编号: 26
> 模块目录: src/error.rs
> 任务阶段: Phase 1
> 前置文档: 01-architecture.md, 07-tensor.md, 21-type.md
> 需求参考: 需求说明书 §8、§12 - §28
> 范围声明: 范围内

---

## 1. 模块定位

本文档定义 Xenon 错误处理模块的设计方案。该模块负责提供全部公开 API 的统一错误模型，包括：

- 可恢复错误如何通过 `Result` 暴露
- 不可恢复错误如何通过 panic 报告
- 错误上下文字段的最小集合
- 类型转换、索引、形状、FFI 等场景的统一诊断规则

### 1.1 职责边界

| 职责           | 包含                                                   | 不包含                       |
| -------------- | ------------------------------------------------------ | ---------------------------- |
| 可恢复错误模型 | `XenonError` 枚举、`Result<T>` 别名、结构化诊断字段    | 日志系统、遥测               |
| panic 规范     | panic 分类规则、panic 消息模板、允许 panic 的边界定义  | 运行时 panic 拦截与恢复      |
| 错误上下文     | 每个变体的最小结构化字段定义、Display 格式化要求       | 错误上报平台、序列化错误模型 |
| 错误传播       | 并行路径中的 `Err` 与 panic 传播规则                   | 跨进程序列化                 |

### 1.2 设计原则

| 原则       | 体现                                                           |
| ---------- | -------------------------------------------------------------- |
| 统一性     | 所有公开 API 使用同一 `XenonError` 枚举，不按模块拆分错误类型  |
| 结构化诊断 | 每个错误变体携带结构化字段，不依赖纯字符串消息                 |
| 可恢复优先 | 安全公开 API 对非法输入返回 `Result`，仅特定边界允许 panic     |
| 最小分配   | 诊断字段的堆分配成本为可接受的工程开销，换取跨 API 一致性      |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                     |
| -------- | ------------------------------------------------------------------------ |
| 需求映射 | 需求说明书 §8, §12 - §28                                                 |
| 范围内   | 可恢复错误返回值、panic 分类、诊断字段、类型转换失败、索引失败、FFI 失败 |
| 范围外   | 自定义日志、第三方错误包装器、跨进程序列化                               |
| 非目标   | 通过 `panic` 替代本应可恢复的用户输入错误                                |

---

## 3. 文件位置

```
src/
└── error.rs       # XenonError, Result<T>, Display impl, helper enums
```

单文件设计：错误类型高度相关且全局共享，拆分反而增加跨模块耦合复杂度。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/error.rs
├── core::any::TypeId          # TypeConversion variant type identity
├── core::fmt                   # Display implementation
├── alloc::borrow::Cow          # Flexible string representation for error fields
└── alloc::vec::Vec             # Heap allocation for shape, index fields
```

### 4.2 类型级依赖

| 来源              | 使用的类型/trait                                                 |
| ----------------- | ---------------------------------------------------------------- |
| `core::any`       | `TypeId`（`TypeConversion` 变体的 `source_type` / `target_type`）|
| `core::fmt`       | `Display`, `Formatter`, `fmt::Result`（`Display` 实现）          |
| `alloc::borrow`   | `Cow<'static, str>`（多个变体的 `operation` 等字段）             |
| `alloc::vec`      | `Vec<usize>`（`shape`、`attempted_index` 等字段）                |

### 4.3 依赖合法性

| 项目           | 说明                                              |
| -------------- | ------------------------------------------------- |
| 新增第三方依赖 | 无新增依赖                                        |
| 合法性结论     | 符合最小依赖限制                                  |
| 替代方案       | 不适用；错误模型统一由 crate 内部类型与标准库承载 |

### 4.4 依赖方向声明

依赖方向：单向向上。`error.rs` 不依赖 crate 内任何其他模块，仅依赖标准库/核心库类型；被所有其他模块消费。

---

## 5. 公共 API 设计

### 5.1 公共接口草案与关键签名

```rust,ignore
use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::any::TypeId;
use core::fmt;

/// Unified recoverable error type for all public Xenon APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum XenonError {
    ShapeMismatch {
        operation: Cow<'static, str>,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },

    BroadcastError {
        operation: &'static str,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        attempted_target_shape: Option<Vec<usize>>,
        axis: Option<usize>,
    },

    LayoutMismatch {
        operation: Cow<'static, str>,
        required_layout: Cow<'static, str>,
        actual_layout: Cow<'static, str>,
        shape: Vec<usize>,
    },

    InvalidLayout {
        operation: Cow<'static, str>,
        storage_kind: Cow<'static, str>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        storage_len: usize,
        reason: Cow<'static, str>,
    },

    InvalidAxis {
        operation: Cow<'static, str>,
        axis: usize,
        ndim: usize,
        shape: Vec<usize>,
    },

    InvalidShape {
        operation: Cow<'static, str>,
        shape: Vec<usize>,
        expected_elements: usize,
        actual_elements: usize,
        offending_dim: Option<usize>,
        reason: Option<Cow<'static, str>>,
    },

    DimensionMismatch {
        operation: Cow<'static, str>,
        expected: usize,
        actual: usize,
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

    InvalidStorageMode {
        operation: Cow<'static, str>,
        expected: Cow<'static, str>,
        actual: Cow<'static, str>,
        shape: Option<Vec<usize>>,
        source_storage_mode: Option<Cow<'static, str>>,
        target_storage_mode: Option<Cow<'static, str>>,
        conversion_type: Option<Cow<'static, str>>,
    },

    Ffi {
        operation: &'static str,
        category: FfiErrorCategory,
        backend: &'static str,
        precondition: &'static str,
        actual: Cow<'static, str>,
    },

    Workspace {
        operation: Cow<'static, str>,
        category: WorkspaceErrorCategory,
        size: Option<usize>,
        align: Option<usize>,
        split: Option<usize>,
        len: Option<usize>,
        reason: Option<Cow<'static, str>>,
    },

    IndexOutOfBounds {
        operation: Cow<'static, str>,
        attempted_index: Vec<usize>,
        axis: usize,
        shape: Vec<usize>,
    },

    TypeConversion {
        source_type: TypeId,
        target_type: TypeId,
        reason: ConversionFailureReason,
        element_index: Option<usize>,
    },
}

/// FFI error category for `XenonError::Ffi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiErrorCategory {
    InvalidRank,
    BlasIncompatibleLayout,
    IntegerOverflow,
}

/// Workspace error category for `XenonError::Workspace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceErrorCategory {
    AllocFailed,
    InvalidLayout,
    AlreadyBorrowed,
    SplitOutOfBounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionFailureReason {
    LossyIntegerNarrowing,
    LossyFloatNarrowing,
    FloatToInteger,
    IntegerToFloatPrecisionLoss,
    NonZeroImaginaryPart,
    UnsupportedByRequirement,
}

pub type Result<T> = core::result::Result<T, XenonError>;
```

- 当前版本不定义 `EmptyArray` 公开错误变体。
- `XenonError` 须实现 `std::error::Error` trait，提供 `source()` 方法用于链式错误追踪。
- 对于所有 `XenonError` 变体，`source()` 返回 `None`，除非内部保留了链式错误源（当前版本不保留）。
- 公开 API 统一使用 prelude 导出的 `crate::error::Result`（即 `Result<T, XenonError>` 别名）作为返回类型。
- 所有可恢复错误直接以 `XenonError` 结构化变体返回，不使用模块内部错误类型。
- 每个变体携带适用的结构化字段，满足 `需求说明书 §27` 对公开诊断信息的要求。

### 5.2 可恢复错误与 panic 的边界

| 场景                                       | 处理方式                                     | 说明                                  |
| ------------------------------------------ | -------------------------------------------- | ------------------------------------- |
| 形状不兼容 / 广播失败                      | `Result::Err(XenonError)`                    | 运行时输入决定，可恢复                |
| 轴越界 / 参数非法 / FFI 前提失败           | `Result::Err(XenonError)`                    | 调用方可修正输入并重试                |
| `cast()` 有损或前提不满足                  | `Result::Err(XenonError::TypeConversion(_))` | `需求说明书 §23` 强制要求             |
| 方法型索引失败                             | `Result::Err(XenonError::IndexOutOfBounds)`  | 需返回结构化索引上下文                |
| 语言级 `Index` 语法 `tensor[i]` 越界       | panic                                        | 属于 Rust 语法糖边界，非 `Result` API |
| 有符号整数算术溢出 / 除以零 / 结果不可表示 | panic                                        | 仅适用于 `i32` / `i64`，见需求说明书  |
| `sqrt(negative)`、`ln(negative)`、`ln(0)`  | IEEE 754 返回 `NaN` / `-Inf`，不得 panic     | `f32` / `f64` 数学域边界              |

### 5.3 安全 API 的 panic 边界

> **总原则：** 所有安全公开 API 对非法输入须返回可恢复错误（`Result`）；仅 `unsafe` 函数的前提违反和内部 helper 可使用 panic。

| 类别                          | 允许 panic 的边界                                               | 约束                                                              |
| ----------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| 语言级语法边界                | `tensor[i]` / `tensor[i] = value`                               | 仅指 `Index`/`IndexMut` 语法糖；越界时可 panic                    |
| 需求明确定义的算术域边界      | `i32` / `i64` 的逐元素算术、归约、内积                          | 溢出、除以零、结果不可表示时 panic                                |
| internal / unsafe helper 边界 | private helper、`unsafe fn` 前提检查、未对外公开的 typed helper | 仅限实现内部或不安全前提；不得作为安全公开 API 的用户输入错误出口 |

除上表外，其余安全公开 API 遇到错误条件时都必须返回 `Result<_, XenonError>`，不得以 panic 代替可恢复错误；即使是 FFI convenience helper，只要属于安全公开 API，也必须遵循这一规则。

### 5.4 公开 API 边界规则

| 边界位置               | 规则                          |
| ---------------------- | ----------------------------- |
| Public API return type | `Result<_, XenonError>`       |
| 错误构造方式           | 直接构造 `XenonError` 结构化变体，不经过中间错误类型映射 |

该表为公开错误边界的唯一基线。

### 5.5 类型转换错误规范

`cast()` 的错误模型须与 `21-type.md` 保持一致：

- `cast<B>(&self)` 返回 `Result<Tensor<B, D>, XenonError>`
- 任何被 `需求说明书 §23` 判定为有损的默认转换组合，都须返回 `XenonError::TypeConversion { source_type, target_type, reason, element_index }`
- 仅当需求显式给出附加成功前提时，满足前提后才可成功
- `Complex -> Real` 不是编译期拒绝；当 `im == 0` 时允许继续转换，否则返回 `XenonError::TypeConversion { ... }`
- `bool` 不参与逐元素类型转换，因此不得用 `TypeConversion` 为 `bool` 扩大支持范围

类型转换失败统一通过 `XenonError::TypeConversion { source_type, target_type, reason, element_index }` 返回，其中字段为公开字段，用户可直接通过模式匹配访问。

### 5.6 结构化上下文字段要求

所有错误变体都须带"错误类别 + 适用上下文"的结构化字段；仅字符串消息不足以满足要求。

| 变体                                  | 最小结构化字段                                                                                                                                                                         |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ShapeMismatch`                       | `operation`, `left_shape`, `right_shape`                                                                                                                                               |
| `BroadcastError`                      | `operation`, `lhs_shape`, `rhs_shape`, `attempted_target_shape?`, `axis?`                                                                                                              |
| `LayoutMismatch`                      | `operation`, `required_layout`, `actual_layout`, `shape`                                                                                                                               |
| `InvalidLayout`                       | `operation`, `storage_kind`, `shape`, `strides`, `offset`, `storage_len`, `reason`                                                                                                     |
| `InvalidAxis`                         | `operation`, `axis`, `ndim`, `shape`                                                                                                                                                   |
| `InvalidShape`                        | `operation`, `shape`, `expected_elements`, `actual_elements`, `offending_dim?`, `reason?`                                                                                              |
| `DimensionMismatch`                   | `operation`, `expected`, `actual`                                                                                                                                                      |
| `InvalidArgument`                     | `operation`, `argument`, `expected`, `actual`, `axis?`, `axis_len?`, `start?`, `end?`, `shape?`；范围切片越界时必须额外携带 `axis`、`axis_len`、`start`、`end`，不得仅以字符串拼接描述 |
| `InvalidStorageMode`                  | `operation`, `expected`, `actual`, `shape?`, `source_storage_mode?`, `target_storage_mode?`, `conversion_type?`                                                                        |
| `Ffi`                                 | `operation`, `category`, `backend`, `precondition`, `actual`                                                                                                                           |
| `Workspace`                           | `operation`, `category`, `size?`, `align?`, `split?`, `len?`, `reason?`                                                                                                                |
| `IndexOutOfBounds`                    | `operation`, `attempted_index`, `axis`, `shape`；`attempted_index` 表示完整多维索引 tuple，`axis` 指出首个越界维度                                                                     |
| `TypeConversion`                      | `source_type`, `target_type`, `reason`, `element_index?`                                                                                                                               |

> **分配成本说明：** `attempted_index: Vec<usize>`、`shape: Vec<usize>` 以及 `InvalidArgument` / `InvalidStorageMode` 中的可选 `Vec<usize>` 字段会带来少量堆分配成本；这是当前版本可接受的诊断开销，用于换取跨公开 API 的一致结构化上下文。

### 5.7 Display 与 panic 信息要求

Display 输出和 panic 文本都必须能让调用方定位问题来源；最少应包含操作名、错误类别以及适用上下文。

**正式规则：** panic 信息必须包含 `operation` + error kind + 至少一个关键上下文字段（如 `axis`、`shape`、`index`、类型等）。

```rust,ignore
impl fmt::Display for XenonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                operation,
                left_shape,
                right_shape,
            } => write!(
                f,
                "shape mismatch in {}: left [{}], right [{}]",
                operation,
                fmt_shape(left_shape),
                fmt_shape(right_shape),
            ),
            Self::BroadcastError {
                operation,
                lhs_shape,
                rhs_shape,
                attempted_target_shape,
                axis,
            } => write!(
                f,
                "broadcast error in {}: lhs [{}], rhs [{}], attempted_target {}, axis {}",
                operation,
                fmt_shape(lhs_shape),
                fmt_shape(rhs_shape),
                attempted_target_shape
                    .as_ref()
                    .map(|value| fmt_shape(value))
                    .unwrap_or_else(|| "<any>".to_string()),
                axis.map(|value| value.to_string()).unwrap_or_else(|| "<any>".to_string()),
            ),
            Self::LayoutMismatch {
                operation,
                required_layout,
                actual_layout,
                shape,
            } => write!(
                f,
                "layout mismatch in {}: required {}, actual {}, shape [{}]",
                operation,
                required_layout,
                actual_layout,
                fmt_shape(shape),
            ),
            Self::InvalidLayout {
                operation,
                storage_kind,
                shape,
                strides,
                offset,
                storage_len,
                reason,
            } => write!(
                f,
                "invalid layout in {}: storage_kind={}, shape [{}], strides [{}], offset {}, storage_len {}, reason: {}",
                operation,
                storage_kind,
                fmt_shape(shape),
                fmt_strides(strides),
                offset,
                storage_len,
                reason,
            ),
            Self::InvalidAxis {
                operation,
                axis,
                ndim,
                shape,
            } => write!(
                f,
                "invalid axis in {}: axis {}, ndim {}, shape [{}]",
                operation,
                axis,
                ndim,
                fmt_shape(shape),
            ),
            Self::InvalidShape {
                operation,
                shape,
                expected_elements,
                actual_elements,
                offending_dim,
                reason,
            } => write!(
                f,
                "invalid shape in {}: shape [{}], expected_elements {}, actual_elements {}, offending_dim {}, reason {}",
                operation,
                fmt_shape(shape),
                expected_elements,
                actual_elements,
                offending_dim.map(|value| value.to_string()).unwrap_or_else(|| "<any>".to_string()),
                reason.as_deref().unwrap_or("<any>"),
            ),
            Self::DimensionMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in {}: expected {}, actual {}",
                operation,
                expected,
                actual,
            ),
            Self::InvalidArgument {
                operation,
                argument,
                expected,
                actual,
                axis,
                axis_len,
                start,
                end,
                shape,
            } => write!(
                f,
                "invalid argument in {}: {} expected {}, actual {}, axis {}, axis_len {}, start {}, end {}, shape {}",
                operation,
                argument,
                expected,
                actual,
                axis.map(|value| value.to_string()).unwrap_or_else(|| "<any>".to_string()),
                axis_len.map(|value| value.to_string()).unwrap_or_else(|| "<any>".to_string()),
                start.map(|value| value.to_string()).unwrap_or_else(|| "<any>".to_string()),
                end.map(|value| value.to_string()).unwrap_or_else(|| "<any>".to_string()),
                shape.as_ref().map(|value| fmt_shape(value)).unwrap_or_else(|| "<any>".to_string()),
            ),
            Self::InvalidStorageMode {
                operation,
                expected,
                actual,
                shape,
                source_storage_mode,
                target_storage_mode,
                conversion_type,
            } => write!(
                f,
                "invalid storage mode in {}: expected {}, actual {}, shape {}, source {}, target {}, conversion {}",
                operation,
                expected,
                actual,
                shape.as_ref().map(|value| fmt_shape(value)).unwrap_or_else(|| "<any>".to_string()),
                source_storage_mode.as_deref().unwrap_or("<any>"),
                target_storage_mode.as_deref().unwrap_or("<any>"),
                conversion_type.as_deref().unwrap_or("<any>"),
            ),
            Self::Ffi {
                operation,
                category,
                backend,
                precondition,
                actual,
            } => write!(
                f,
                "ffi error in {}: {:?} (backend={}, precondition={}, actual={})",
                operation,
                category,
                backend,
                precondition,
                actual,
            ),
            Self::Workspace {
                operation,
                category,
                size,
                align,
                split,
                len,
                reason,
            } => write!(
                f,
                "workspace error in {}: {:?}, size={}, align={}, split={}, len={}, reason={}",
                operation,
                category,
                size.map(|v| v.to_string()).unwrap_or_else(|| "<any>".to_string()),
                align.map(|v| v.to_string()).unwrap_or_else(|| "<any>".to_string()),
                split.map(|v| v.to_string()).unwrap_or_else(|| "<any>".to_string()),
                len.map(|v| v.to_string()).unwrap_or_else(|| "<any>".to_string()),
                reason.as_deref().unwrap_or("<any>"),
            ),
            Self::TypeConversion {
                source_type,
                target_type,
                reason,
                element_index,
            } => write!(
                f,
                "type conversion failed at element {}: {:?} -> {:?} ({:?})",
                element_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "<any>".to_string()),
                source_type,
                target_type,
                reason,
            ),
            Self::IndexOutOfBounds {
                operation,
                attempted_index,
                axis,
                shape,
            } => write!(
                f,
                "index out of bounds in {}: index [{}], axis {}, shape [{}]",
                operation,
                fmt_shape(attempted_index),
                axis,
                fmt_shape(shape),
            ),
        }
    }
}
```

> **Display 约束：** 对 `Option<Vec<usize>>` 等可选结构化字段，`Display` 实现必须做人性化格式化；`None` 统一显示为 `<any>`，不得直接打印 `Some(...)` / `None` 调试文本。

### 5.8 统一 panic 类别

除文档已提到的归约溢出外，以下不可恢复情形都须纳入统一 panic 规范：

- 逐元素整数算术溢出
- 整数除以零
- 结果不可表示（例如 `abs(i32::MIN)`、`i32::MIN / -1`）
- 整数内积的乘积或累加溢出

推荐 panic message 模板：`"Xenon: {operation} overflow for {type} at {context}"`

| panic 类别                   | 推荐消息示例                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------ |
| 逐元素加法溢出               | `"Xenon: add overflow for i32 at element_index=7"`                             |
| 归约溢出                     | `"Xenon: sum overflow for i64 at axis=1, output_index=3"`                      |
| 内积溢出                     | `"Xenon: dot overflow for i32 at lane=12"`                                     |
| 语言级索引 panic             | `"Xenon: index out of bounds for tensor[i] at axis=0, index=9, len=4"`         |
| internal/unsafe helper panic | `"Xenon: ptr_at precondition violation in internal helper at axis=1, index=8"` |

### 5.9 Good / Bad 对比式代码示例

#### 类型转换

```rust,ignore
// Good - cast is fallible and reports the failing element.
pub fn cast<B: Element>(&self) -> Result<Tensor<B, D>, XenonError>
where
    A: CastTo<B>,
{
    let mut out = Vec::with_capacity(self.len());
    for (index, value) in self.iter().enumerate() {
        let converted = value.cast_to().map_err(|err| {
            // Preserve the conversion error, enriching with element index.
            match err {
                XenonError::TypeConversion { source_type, target_type, reason, .. } => {
                    XenonError::TypeConversion {
                        source_type,
                        target_type,
                        reason,
                        element_index: Some(index),
                    }
                }
                other => other,
            }
        })?;
        out.push(converted);
    }
    // Internal helper, not a public API.
    Ok(Tensor::from_shape_vec_aligned(self.shape().clone(), out))
}

// Bad - silently saturating or truncating.
pub fn cast_bad<B: Element>(&self) -> Tensor<B, D>
where
    A: CastTo<B>,
{
    let out = self.iter().map(|value| value.cast_to_lossy()).collect();
    // Internal helper, not a public API.
    Tensor::from_shape_vec_aligned(self.shape().clone(), out)
}
```

#### 整数算术溢出

```rust,ignore
// Good - checked arithmetic with explicit panic message.
let value = lhs
    .checked_mul(rhs)
    .expect("Xenon: dot overflow for i32 at lhs_index=3, rhs_index=3");

// Bad - silent wrapping in release mode.
let value = lhs * rhs;
```

---

## 6. 内部实现设计

### 6.1 算法描述

```
//
// the following pseudo-codes describe the core error construction flow.
//
construct_xenon_error(operation, variant, context_fields):
    1. match variant to select the appropriate XenonError enum variant
    2. populate all mandatory structured fields from context_fields
    3. set optional fields to None when not applicable
    4. return XenonError::{variant} { ... }

fmt_display(error, formatter):
    1. match error variant
    2. for each variant, format operation + kind + key context
    3. for Option fields, render as human-readable:
       - Some(value) -> display value
       - None -> display "<any>"
    4. write formatted string to formatter
```

### 6.2 安全性论证

本模块不涉及 `unsafe` 代码。错误类型的构造、`Display` 实现与 `Clone`/`PartialEq` 派生均为安全操作。

### 6.3 性能考量表

| 方面         | 设计决策                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------ |
| 分配开销     | `Vec<usize>` 字段（`shape`、`attempted_index`）在错误构造时产生少量堆分配                  |
| 零分配路径   | 错误路径本身非热路径；少量分配换取结构化诊断上下文是可接受的工程权衡                       |
| Clone 成本   | `XenonError` 的 `Clone` 会复制 `Vec` 和 `Cow` 字段；仅在测试或显式需要时调用              |
| PartialEq    | 用于测试断言；`TypeId` 的 `PartialEq` 比较为整数级比较，`Vec` 为逐元素比较                 |

---

## 7. 实现任务拆分

### Wave 1: 基础类型

- [ ] **T1**: 创建 `src/error.rs` 骨架
  - 文件: `src/error.rs`
  - 内容: 模块声明、`XenonError` 枚举定义（所有变体及字段）
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 定义辅助枚举类型
  - 文件: `src/error.rs`
  - 内容: `FfiErrorCategory`、`WorkspaceErrorCategory`、`ConversionFailureReason` 枚举，`Result<T>` 类型别名
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: Display 实现

- [ ] **T3**: 实现 `fmt::Display` for `XenonError`
  - 文件: `src/error.rs`
  - 内容: 各变体的 Display 格式化实现，包含辅助函数 `fmt_shape`、`fmt_strides`
  - 测试: `test_display_*` 系列
  - 前置: T2
  - 预计: 15 min

### Wave 3: Error trait 与导出

- [ ] **T4**: 实现 `std::error::Error` for `XenonError`
  - 文件: `src/error.rs`
  - 内容: `Error` trait 实现，`source()` 返回 `None`
  - 测试: `test_error_trait_source_none`
  - 前置: T2
  - 预计: 5 min

- [ ] **T5**: 添加 prelude 导出
  - 文件: `src/error.rs`, `src/lib.rs`（或对应 prelude 文件）
  - 内容: 公开导出 `XenonError`、`Result`、辅助枚举
  - 测试: 编译通过，外部 crate 可通过 prelude 使用
  - 前置: T4
  - 预计: 5 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
               │
Wave 2:       [T3]
               │
Wave 3:       [T4] → [T5]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置               | 目的                                                       |
| ----------------------- | ------------------ | ---------------------------------------------------------- |
| 单元测试                | `src/error.rs` 内  | 验证 `XenonError` 各变体的 Display、Clone、PartialEq       |
| 集成测试                | 集成测试目录       | 验证跨模块 API 的错误映射正确性                            |
| 边界测试                | 与集成测试配套     | 空形状、非法轴、越界索引、复数虚部非零、整数极值、NaN/Inf  |
| panic 测试              | 集成测试目录       | 验证逐元素整数溢出、除以零、`abs(MIN)`、dot overflow        |
| 并行测试                | 集成测试目录       | 验证 `Err` 与 panic 在并行路径中的传播一致性               |
| Feature gate / 配置测试 | 配置矩阵          | 验证可选 SIMD/并行路径与标量路径的错误类别一致             |
| 类型边界 / 编译期测试   | 编译期测试框架     | 验证 `TypeId` 字段在 `const` 上下文中的可用性              |

### 8.2 单元测试清单

| 测试函数                                       | 测试内容                                        | 优先级 |
| ---------------------------------------------- | ----------------------------------------------- | ------ |
| `test_cast_lossy_returns_type_conversion`      | 有损转换返回 `TypeConversion`                   | 高     |
| `test_cast_reports_element_index`              | 转换失败包含 `element_index`                    | 高     |
| `test_complex_to_real_requires_zero_imag`      | 复数转实数的附加成功前提                        | 高     |
| `test_invalid_argument_has_structured_context` | `InvalidArgument` 不再只有 message              | 高     |
| `test_invalid_shape_reports_dimension_context` | `InvalidShape` 包含维度/元素数上下文            | 高     |
| `test_index_error_reports_axis_and_shape`      | 索引错误包含 `attempted_index`、`axis`、`shape` | 高     |
| `test_integer_division_by_zero_panics`         | 除以零走统一 panic                              | 高     |
| `test_dot_overflow_panics`                     | 内积溢出走统一 panic                            | 高     |
| `test_display_shape_mismatch`                  | `ShapeMismatch` 的 Display 输出格式            | 中     |
| `test_display_option_fields_render_any`        | `None` 字段显示为 `<any>`                      | 中     |
| `test_error_trait_source_none`                 | `std::error::Error` 的 `source()` 返回 `None`  | 中     |
| `test_clone_eq_roundtrip`                      | `Clone` + `PartialEq` 往返一致                 | 中     |

### 8.3 边界测试场景表

| 场景                              | 预期行为                                         |
| --------------------------------- | ------------------------------------------------ |
| 空形状 `shape=[0, 3]`            | 返回 `InvalidShape` 或加法单位元（视 API 而定）  |
| 非法轴 `axis=5, ndim=2`          | 返回 `InvalidAxis` 结构化错误                    |
| 越界索引 `index=[9], shape=[4]`   | 返回 `IndexOutOfBounds` 结构化错误               |
| 复数虚部非零 `Complex(1, 2)`     | 转换为实数类型返回 `TypeConversion { NonZeroImaginaryPart }` |
| 整数极值 `i32::MIN`              | `abs(i32::MIN)` 走 panic                        |
| NaN/Inf 转换                      | `f64::NaN` → `i32` 返回 `TypeConversion` 错误   |

### 8.4 Feature gate / 配置测试

| 配置      | 验证点                                         |
| --------- | ---------------------------------------------- |
| 默认配置  | SIMD/并行关闭时错误类别与结构化字段一致         |
| 启用 SIMD | SIMD 路径错误类别与标量路径相同                 |
| 启用并行  | 并行路径错误传播与串行路径一致，不静默吞掉错误 |

### 8.5 评审要求

- 任何新增公开 API 都必须明确写出"返回 `Result` 还是 panic"的裁决
- 任何新增错误变体都必须说明结构化字段，不得只增加 `message: &'static str`
- 任何新增类型转换组合都必须同时更新 `21-type.md` 与本规范中的错误路径说明

---

## 9. 模块交互设计

### 9.1 接口约定

| 方向           | 对方模块              | 接口/类型                        | 约定                                            |
| -------------- | --------------------- | -------------------------------- | ----------------------------------------------- |
| 被消费（输出） | `tensor` / `shape`   | `XenonError::ShapeMismatch`     | 形状校验失败时构造并返回                        |
| 被消费（输出） | `index`              | `XenonError::IndexOutOfBounds`  | 方法型索引越界时构造并返回                      |
| 被消费（输出） | `broadcast` / `math` | `XenonError::BroadcastError`    | 广播不兼容时构造并返回                          |
| 被消费（输出） | `reduction`          | `XenonError::InvalidAxis`       | 轴越界时构造并返回；溢出走 panic                |
| 被消费（输出） | `convert`            | `XenonError::TypeConversion`    | 有损转换失败时构造并返回                        |
| 被消费（输出） | `ffi`                | `XenonError::Ffi`               | FFI 前提不满足时构造并返回                      |
| 被消费（输出） | 所有模块             | `Result<T>`                     | 公开 API 返回类型统一使用此别名                 |

### 9.2 数据流描述

```
Caller invokes public API (e.g., tensor.broadcast_to(shape))
    │
    ├── API validates input (shape, axis, type, layout, ...)
    │       │
    │       ├── Valid → proceed with computation → return Ok(result)
    │       │
    │       └── Invalid → construct XenonError::{Variant} with structured context
    │               │
    │               └── return Err(XenonError)
    │
    └── For panic-bound operations (integer overflow, Index syntax sugar, ...)
            │
            ├── Triggered → panic with formatted message
            │       "Xenon: {operation} overflow for {type} at {context}"
            │
            └── In parallel path (rayon)
                    │
                    ├── Err → propagate to join point as soon as detectable
                    └── panic → propagate to join point, never silently swallow
```

### 9.3 生命周期与所有权约定

> **约定**: `XenonError` 为 `Clone` 类型，错误构造时所有 `Cow<'static, str>` 字段使用 `'static` 生命周期，不借用调用方的临时数据。`Vec<usize>` 字段在构造时独立拥有，错误可安全地跨线程传递和存储。

---

## 10. 错误处理与语义边界

| 主题              | 需要说明的内容                                                  |
| ----------------- | --------------------------------------------------------------- |
| Recoverable error | 本模块定义可恢复错误的统一类型，错误构造本身不会失败（infallible） |
| Panic             | 本模块不直接触发 panic；各消费模块按 panic 边界表（§5.2）触发    |
| 路径一致性        | 标量 / SIMD / 并行路径须保持相同错误类别和结构化字段             |
| 容差边界          | 不适用；错误类型不涉及浮点数值计算                              |

### 10.1 数学函数定义域边界

| 场景             | `f32` / `f64` 行为 | 约束来源           |
| ---------------- | ------------------ | ------------------ |
| `sqrt(negative)` | 返回 `NaN`         | `需求说明书 §28.3` |
| `ln(negative)`   | 返回 `NaN`         | `需求说明书 §28.3` |
| `ln(0)`          | 返回 `-Inf`        | `需求说明书 §28.3` |

这些情形遵循 IEEE 754 语义，属于数值结果边界，不属于 panic 边界。

### 10.2 并行路径与资源释放

- 并行路径中的 `Err(XenonError)` 须尽快向上传播，不得延后为"全部 worker 完成后再统一检查"
- 并行路径中的 panic 不得被吞掉或伪装为成功结果
- 所有资源释放逻辑不得再触发 panic；在 `panic = abort` 环境下允许进程级终止带来资源不回收

对 `rayon` 上下文中的"立即传播"，本文采用工程化解释：

- 任一 worker 首次观察到 panic 或 `Err` 后，须终止该 worker 的当前执行路径并向 join 点报告失败
- 其他 worker 可能在 join 检测到失败前完成自己已经领取的当前 chunk；这是 `rayon` work-stealing 调度的实际限制
- 因此 `需求说明书 §27` 中的"立即"含义是"as soon as practically detectable"，而不是"所有线程瞬时同步中止"

### 10.3 受影响模块

| 模块/能力            | 影响内容                                     |
| -------------------- | -------------------------------------------- |
| `tensor` / `shape`   | 形状校验、布局前提、元素总数校验             |
| `index`              | 越界索引、按轴索引、切片边界诊断             |
| `broadcast` / `math` | 广播失败、形状不兼容、参数非法               |
| `reduction`          | 非法轴、空输入单位元语义、整数溢出 panic     |
| `convert`            | 类型转换失败的元素索引定位                   |
| `ffi`                | FFI 前提失败与后端约束诊断                   |
| `parallel`           | panic / `Err` 的尽快传播，不得静默吞掉       |

---

## 11. 设计决策记录

### 决策 1：统一 `XenonError` 枚举而非模块独立错误类型

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | 所有公开 API 使用单一 `XenonError` 枚举，不按模块拆分为独立错误类型             |
| 理由     | 避免调用方为不同模块维护不同的错误处理逻辑；结构化字段可按变体精确匹配          |
| 替代方案 | 每个模块定义自己的 `XxxError` 枚举 — 放弃，增加调用方负担且难以跨模块统一诊断  |
| 替代方案 | 使用 `anyhow` / `eyre` 等动态错误类型 — 放弃，违反最小依赖约束且损失结构化匹配  |

### 决策 2：panic 边界的严格划分

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | 安全公开 API 仅在语言级语法边界和需求明确定义的算术域边界允许 panic             |
| 理由     | 需求说明书 §27 要求可恢复错误以返回值形式报告；宽泛的 panic 会破坏调用方的错误恢复能力 |
| 替代方案 | 对所有非法输入统一 panic — 放弃，违反可恢复优先原则                             |
| 替代方案 | 对 `tensor[i]` 越界返回 Result — 放弃，受限于 Rust `Index` trait 签名           |

### 决策 3：结构化字段而非纯字符串消息

| 属性     | 值                                                                               |
| -------- | -------------------------------------------------------------------------------- |
| 决策     | 每个错误变体携带结构化字段（`operation`、`shape`、`axis` 等），不用 `message: String` |
| 理由     | 结构化字段允许调用方按程序逻辑匹配和处理错误，满足 `需求说明书 §27` 对诊断信息的要求 |
| 替代方案 | 仅提供 `String` 消息 — 放弃，无法程序化匹配错误类别                            |
| 替代方案 | 使用 `thiserror` 派生 — 放弃，违反最小依赖约束                                 |

---

## 12. 性能描述

### 12.1 复杂度标注

- 错误构造：O(k)，其中 k 为 `Vec<usize>` 字段中的元素总数（通常为 ndim，即 ≤ 7）
- `Clone`：与构造相同的分配开销
- `Display`：O(n) 格式化，n 为输出字符串长度

### 12.2 性能考量

| 场景           | 行为                                                         |
| -------------- | ------------------------------------------------------------ |
| 正常路径       | 错误类型不被构造，零开销                                     |
| 错误路径       | 少量堆分配（`Vec<usize>` + `Cow`），为可接受的诊断开销      |
| 热路径影响     | 无；错误构造仅在异常路径触发                                 |
| 测试路径开销   | `Clone` + `PartialEq` 用于断言；非生产路径                   |

---

## 13. 平台与工程约束

| 约束       | 需要说明的内容                                                            |
| ---------- | ------------------------------------------------------------------------- |
| `std` only | 方案依赖 `std`；需求说明书 §1.3 明确仅支持 `std` 环境，不讨论 `no_std`   |
| 单 crate   | 保持单 crate 边界；错误类型定义在 crate 内部，不引入额外 crate            |
| SemVer     | 新增错误变体或修改诊断字段属于公开 API 兼容性变更，须遵循 SemVer          |
| 最小依赖   | 不引入额外第三方依赖；错误模型由标准库与 crate 内部类型承载               |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.0.6 | 2026-04-10 |
| 1.1.0 | 2026-04-14 |
| 1.1.1 | 2026-04-14 |
| 1.1.2 | 2026-04-14 |
| 1.1.3 | 2026-04-15 |
| 1.1.4 | 2026-04-15 |
| 1.1.5 | 2026-04-16 |
| 2.0.0 | 2026-04-22 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
