# 错误处理规范

> 文档编号: 26 | 适用范围: 所有公开 API | 阶段: Phase 1
> 前置文档: `01-architecture.md`, `07-tensor.md`, `21-type.md`
> 需求参考: 需求说明书 §8, §12-§26, §27, §28.2, §28.3, §28.5
> 范围声明: 范围内

---

## 1. 主题定位与适用范围

### 1.1 主题定位

本文档定义 Xenon 全部公开 API 的统一错误模型，包括：

- 可恢复错误如何通过 `Result` 暴露
- 不可恢复错误如何通过 panic 报告
- 错误上下文字段的最小集合
- 类型转换、索引、形状、FFI 等场景的统一诊断规则

### 1.2 适用范围

| 项目     | 内容                                                          |
| -------- | ------------------------------------------------------------- |
| 覆盖对象 | 所有公开 API、公开 trait 入口、公开运算符语义、并行执行路径   |
| 范围内   | `XenonError`、`Result<T>`、panic 分类、诊断字段、并行传播规则 |
| 范围外   | 日志系统、遥测、错误上报平台、序列化错误模型                  |
| 非目标   | 让每个模块在公开 API 层各自暴露一套彼此独立的错误类型         |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                             |
| -------- | ---------------------------------------------------------------------------------------------------------------- |
| 需求映射 | `require.md §8`, `§12`, `§13`, `§14`, `§16`, `§18`, `§21`, `§23`, `§25`, `§26`, `§27`, `§28.2`, `§28.3`, `§28.5`, `§6.2` |
| 范围内   | 可恢复错误返回值、panic 分类、诊断字段、类型转换失败、索引失败、FFI 失败                                         |
| 范围外   | 自定义日志、第三方错误包装器、跨进程序列化                                                                       |
| 非目标   | 通过 `panic` 替代本应可恢复的用户输入错误                                                                        |

### 2.1 关键约束

- 可恢复错误须通过返回值形式报告
- 不可恢复错误须统一通过 panic 暴露
- 所有可恢复错误以及对应 panic 信息都须携带适用的结构化上下文
- `cast()` 失败属于可恢复错误，必须进入 `XenonError`

---

## 3. 影响范围

### 3.1 受影响模块

| 模块/能力            | 影响内容                                     |
| -------------------- | -------------------------------------------- |
| `tensor` / `shape`   | 形状校验、布局前提、元素总数校验             |
| `index`              | 越界索引、按轴索引、切片边界诊断             |
| `broadcast` / `math` | 广播失败、形状不兼容、参数非法               |
| `reduction`          | 非法轴、空输入单位元语义、整数溢出 panic     |
| `convert`            | `TypeConversionError` 内部映射与元素索引定位 |
| `ffi`                | FFI 前提失败与后端约束诊断                   |
| `parallel`           | panic / `Err` 的尽快传播，不得静默吞掉       |

### 3.2 统一依赖方向

> **依赖方向：单向向上。** 错误规范由 `error.rs` 提供基础类型，但本文约束的是所有公开 API 的外部行为，而非某一个源码目录的局部实现。

### 3.3 依赖合法性与新增依赖说明

| 项目           | 说明                                              |
| -------------- | ------------------------------------------------- |
| 新增第三方依赖 | 无新增依赖                                        |
| 合法性结论     | 符合最小依赖限制                                  |
| 替代方案       | 不适用；错误模型统一由 crate 内部类型与标准库承载 |

---

## 4. 规范内容

### 4.1 可恢复错误与 panic 的边界

| 场景                                       | 处理方式                                     | 说明                                    |
| ------------------------------------------ | -------------------------------------------- | --------------------------------------- |
| 形状不兼容 / 广播失败                      | `Result::Err(XenonError)`                    | 运行时输入决定，可恢复                  |
| 轴越界 / 参数非法 / FFI 前提失败           | `Result::Err(XenonError)`                    | 调用方可修正输入并重试                  |
| `cast()` 有损或前提不满足                  | `Result::Err(XenonError::TypeConversion(_))` | `require.md §23` 强制要求               |
| 方法型索引失败                             | `Result::Err(XenonError::IndexOutOfBounds)`  | 需返回结构化索引上下文                  |
| 语言级 `Index` 语法 `tensor[i]` 越界       | panic                                        | 属于 Rust 语法糖边界，非 `Result` API   |
| 有符号整数算术溢出 / 除以零 / 结果不可表示 | panic                                        | 仅适用于 `i32` / `i64`，见 `require.md` |
| `sqrt(negative)`、`ln(negative)`、`ln(0)`  | IEEE 754 返回 `NaN` / `-Inf`，不得 panic     | `f32` / `f64` 数学域边界                |

### 4.1.1 安全 API 的 panic 边界

> **总原则：** 所有安全公开 API 对非法输入须返回可恢复错误（`Result`）；仅 `unsafe` 函数的前提违反和内部 helper 可使用 panic。

| 类别                          | 允许 panic 的边界                                               | 约束                                                              |
| ----------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| 语言级语法边界                | `tensor[i]` / `tensor[i] = value`                               | 仅指 `Index`/`IndexMut` 语法糖；越界时可 panic                    |
| 需求明确定义的算术域边界      | `i32` / `i64` 的逐元素算术、归约、内积                          | 溢出、除以零、结果不可表示时 panic                                |
| internal / unsafe helper 边界 | private helper、`unsafe fn` 前提检查、未对外公开的 typed helper | 仅限实现内部或不安全前提；不得作为安全公开 API 的用户输入错误出口 |

除上表外，其余安全公开 API 遇到错误条件时都必须返回 `Result<_, XenonError>`，不得以 panic 代替可恢复错误；即使是 FFI convenience helper，只要属于安全公开 API，也必须遵循这一规则。

### 4.2 统一错误类型

```rust
use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

/// Unified recoverable error type for all public Xenon APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum XenonError {
    ShapeMismatch {
        operation: Cow<'static, str>,
        left_shape: Vec<usize>,
        right_shape: Vec<usize>,
    },

    // 本定义为 BroadcastError 的唯一权威源；其他文档引用时须与此一致。
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

    // 本定义为 InvalidShape 的唯一权威源；其他文档不得增删字段，除非先同步更新此处。
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

    Ffi(FfiError),

    Workspace(WorkspaceError),

    IndexOutOfBounds {
        operation: Cow<'static, str>,
        attempted_index: Vec<usize>,
        axis: usize,
        shape: Vec<usize>,
    },

    TypeConversion(TypeConversionError),
}

/// Module-internal payload for XenonError::TypeConversion.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeConversionError {
    source_type: Cow<'static, str>,
    target_type: Cow<'static, str>,
    reason: TypeConversionReason,
    element_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeConversionReason {
    LossyIntegerNarrowing,
    LossyFloatNarrowing,
    FloatToInteger,
    IntegerToFloatPrecisionLoss,
    NonZeroImaginaryPart,
    UnsupportedByRequirement,
}

pub type Result<T> = core::result::Result<T, XenonError>;
```

当前版本不定义 `EmptyArray` 公开错误变体。按 `require.md §13-§14`，空输入 `dot` 与 `sum` 返回加法单位元；若未来新增确需“至少一个元素”的 API，应在对应版本中单独裁定是否引入专门错误变体。

`XenonError` 须实现 `std::error::Error` trait，提供 `source()` 方法用于链式错误追踪。

公开 API 统一使用 prelude 导出的 `crate::error::Result`（即 `Result<T, XenonError>` 别名）作为返回类型。

模块可以为内部实现保留局部错误分类（例如 `FfiError`、`WorkspaceError`、`TypeConversionError`），以避免在模块内部丢失语义；但凡进入 Xenon 的公开 API 边界，必须统一包装为 `XenonError`（如 `XenonError::Ffi(...)`、`XenonError::Workspace(...)`、`XenonError::TypeConversion(...)`），不得直接向外暴露模块私有错误类型。

`WorkspaceError` 与 `FfiError` 都必须实现 `std::error::Error`，以便 `XenonError::source()` 暴露完整的内层错误链。

> **FfiError 结构化说明：** `FfiError` 已采用结构化字段设计，具体字段定义参见 `23-ffi.md`。

> **存储模式转换说明：** 存储模式转换失败统一使用 `XenonError::InvalidStorageMode`，并携带 `source_storage_mode`、`target_storage_mode`、`conversion_type` 字段；当某个调用点并不涉及显式存储模式转换时，这些字段可为 `None`。

### 4.2.1 公开 API 边界映射

| 边界位置               | 规则                                                     |
| ---------------------- | -------------------------------------------------------- |
| Public API return type | `Result<_, XenonError>`                                  |
| Internal mapping       | `WorkspaceError -> XenonError::Workspace(...)`           |
| Internal mapping       | `FfiError -> XenonError::Ffi(...)`                       |
| Internal mapping       | `TypeConversionError -> XenonError::TypeConversion(...)` |

该表为公开错误边界的唯一基线；其他设计文档若在公开 API 层直接暴露 `WorkspaceError` 或独立的 `TypeConversionError`，均视为与本文冲突，必须以本文为准修正。

### 4.3 类型转换错误规范

`cast()` 的错误模型须与 `21-type.md` 保持一致：

- `cast<B>(&self)` 返回 `Result<Tensor<B, D>, XenonError>`
- 任何被 `require.md §23` 判定为有损的默认转换组合，都须返回 `XenonError::TypeConversion(TypeConversionError)`
- 仅当需求显式给出附加成功前提时，满足前提后才可成功
- `Complex -> Real` 不是编译期拒绝；当 `im == 0` 时允许继续转换，否则返回 `XenonError::TypeConversion(TypeConversionError)`
- `bool` 不参与逐元素类型转换，因此不得用 `TypeConversion` 为 `bool` 扩大支持范围

规范名称固定为 `XenonError::TypeConversion(TypeConversionError)`：

- `TypeConversionError` 是模块内部载荷结构，不是公开错误边界上的独立错误类型
- `03-element.md`、`04-complex.md` 及其他文档引用类型转换失败时，只能引用 `XenonError::TypeConversion(TypeConversionError)`
- 不得再把独立 `TypeConversionError` 写成公开 API 可直接返回的错误类型

```rust
// Good - cast is fallible and reports the failing element.
pub fn cast<B: Element>(&self) -> Result<Tensor<B, D>, XenonError>
where
    A: CastTo<B>,
{
    let mut out = Vec::with_capacity(self.len());
    for (index, value) in self.iter().enumerate() {
        out.push(value.cast_to(index)?);
    }
    Ok(Tensor::from_shape_vec_aligned(self.shape().clone(), out))
}

// Bad - silently saturating or truncating.
pub fn cast_bad<B: Element>(&self) -> Tensor<B, D>
where
    A: CastTo<B>,
{
    let out = self.iter().map(|value| value.cast_to_lossy()).collect();
    Tensor::from_shape_vec_aligned(self.shape().clone(), out)
}
```

### 4.4 结构化上下文字段要求

所有错误变体都须带“错误类别 + 适用上下文”的结构化字段；仅字符串消息不足以满足要求。

| 变体                                  | 最小结构化字段                                                                     |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `ShapeMismatch`                       | `operation`, `left_shape`, `right_shape`                                           |
| `BroadcastError`                      | `operation`, `lhs_shape`, `rhs_shape`, `attempted_target_shape?`, `axis?`          |
| `LayoutMismatch`                      | `operation`, `required_layout`, `actual_layout`, `shape`                           |
| `InvalidLayout`                       | `operation`, `storage_kind`, `shape`, `strides`, `offset`, `storage_len`, `reason` |
| `InvalidAxis`                         | `operation`, `axis`, `ndim`, `shape`                                               |
| `InvalidShape`                        | `operation`, `shape`, `expected_elements`, `actual_elements`, `offending_dim?`, `reason?` |
| `DimensionMismatch`                   | `operation`, `expected`, `actual`                                                  |
| `InvalidArgument`                     | `operation`, `argument`, `expected`, `actual`, `axis?`, `axis_len?`, `start?`, `end?`, `shape?`；范围切片越界时必须额外携带 `axis`、`axis_len`、`start`、`end`，不得仅以字符串拼接描述 |
| `InvalidStorageMode`                  | `operation`, `expected`, `actual`, `shape?`, `source_storage_mode?`, `target_storage_mode?`, `conversion_type?` |
| `Ffi(FfiError)`                       | 由 `FfiError` 提供 `operation`, `backend`, `precondition`, `actual`                |
| `Workspace(...)`                      | 由 `WorkspaceError` 提供 `size`, `align`, `split`, `len` 等适用结构化字段          |
| `IndexOutOfBounds`                    | `operation`, `attempted_index`, `axis`, `shape`；`attempted_index` 表示完整多维索引 tuple，`axis` 指出首个越界维度 |
| `TypeConversion(TypeConversionError)` | `source_type`, `target_type`, `reason`, `element_index`                            |

> **分配成本说明：** `attempted_index: Vec<usize>`、`shape: Vec<usize>` 以及 `InvalidArgument` / `InvalidStorageMode` 中的可选 `Vec<usize>` 字段会带来少量堆分配成本；这是当前版本可接受的诊断开销，用于换取跨公开 API 的一致结构化上下文。

### 4.5 Display 与 panic 信息要求

Display 输出和 panic 文本都必须能让调用方定位问题来源；最少应包含操作名、错误类别以及适用上下文。

**正式规则：** panic 信息必须包含 `operation` + error kind + 至少一个关键上下文字段（如 `axis`、`shape`、`index`、类型等）。

```rust
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
                "broadcast error in {}: lhs [{}], rhs [{}], attempted_target {}, axis {:?}",
                operation,
                fmt_shape(lhs_shape),
                fmt_shape(rhs_shape),
                attempted_target_shape
                    .as_ref()
                    .map(|value| fmt_shape(value))
                    .unwrap_or_else(|| "<none>".to_string()),
                axis,
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
                "invalid shape in {}: shape [{}], expected_elements {}, actual_elements {}, offending_dim {:?}, reason {}",
                operation,
                fmt_shape(shape),
                expected_elements,
                actual_elements,
                offending_dim,
                reason.as_deref().unwrap_or("<none>"),
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
            Self::Ffi(err) => write!(f, "ffi error: {}", err),
            Self::Workspace(err) => write!(f, "workspace error: {}", err),
            Self::TypeConversion(err) => write!(
                f,
                "type conversion failed at element {}: {} -> {} ({:?})",
                err.element_index,
                err.source_type,
                err.target_type,
                err.reason,
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

### 4.6 必须统一覆盖的 panic 类别

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

```rust
// Good - checked arithmetic with explicit panic message.
let value = lhs
    .checked_mul(rhs)
    .expect("Xenon: dot overflow for i32 at lhs_index=3, rhs_index=3");

// Bad - silent wrapping in release mode.
let value = lhs * rhs;
```

### 4.7 数学函数定义域边界

| 场景             | `f32` / `f64` 行为 | 约束来源           |
| ---------------- | ------------------ | ------------------ |
| `sqrt(negative)` | 返回 `NaN`         | `require.md §28.3` |
| `ln(negative)`   | 返回 `NaN`         | `require.md §28.3` |
| `ln(0)`          | 返回 `-Inf`        | `require.md §28.3` |

这些情形遵循 IEEE 754 语义，属于数值结果边界，不属于 panic 边界。

### 4.8 并行路径与资源释放

- 并行路径中的 `Err(XenonError)` 须尽快向上传播，不得延后为“全部 worker 完成后再统一检查”
- 并行路径中的 panic 不得被吞掉或伪装为成功结果
- 所有资源释放逻辑不得再触发 panic；在 `panic = abort` 环境下允许进程级终止带来资源不回收

对 `rayon` 上下文中的“立即传播”，本文采用工程化解释：

- 任一 worker 首次观察到 panic 或 `Err` 后，须终止该 worker 的当前执行路径并向 join 点报告失败
- 其他 worker 可能在 join 检测到失败前完成自己已经领取的当前 chunk；这是 `rayon` work-stealing 调度的实际限制
- 因此 `require.md §27` 中的“立即”含义是“as soon as practically detectable”，而不是“所有线程瞬时同步中止”

---

## 5. 验证与落地方式

### 5.1 验证清单

| 验证项     | 要求                                                                                  |
| ---------- | ------------------------------------------------------------------------------------- |
| 单元测试   | 覆盖 `XenonError` 各变体的 Display、Clone、PartialEq 与结构化字段                     |
| 集成测试   | 覆盖 `transpose`、`broadcast`、`sum_axis`、`cast`、`ffi`、方法型索引等 API 的错误映射 |
| 边界测试   | 覆盖空形状、非法轴、越界索引、复数虚部非零、整数极值、NaN/Inf 转换                    |
| panic 测试 | 覆盖逐元素整数溢出、除以零、`abs(MIN)`、dot overflow                                  |
| 并行测试   | 覆盖 `Err` 与 panic 在并行路径中的传播一致性                                          |

### 5.2 重点测试用例

| 测试函数                                       | 测试内容                                        |
| ---------------------------------------------- | ----------------------------------------------- |
| `test_cast_lossy_returns_type_conversion`      | 有损转换返回 `TypeConversion`                   |
| `test_cast_reports_element_index`              | 转换失败包含 `element_index`                    |
| `test_complex_to_real_requires_zero_imag`      | 复数转实数的附加成功前提                        |
| `test_invalid_argument_has_structured_context` | `InvalidArgument` 不再只有 message              |
| `test_invalid_shape_reports_dimension_context` | `InvalidShape` 包含维度/元素数上下文            |
| `test_index_error_reports_axis_and_shape`      | 索引错误包含 `attempted_index`、`axis`、`shape` |
| `test_integer_division_by_zero_panics`         | 除以零走统一 panic                              |
| `test_dot_overflow_panics`                     | 内积溢出走统一 panic                            |

### 5.3 评审要求

- 任何新增公开 API 都必须明确写出“返回 `Result` 还是 panic”的裁决
- 任何新增错误变体都必须说明结构化字段，不得只增加 `message: &'static str`
- 任何新增类型转换组合都必须同时更新 `21-type.md` 与本规范中的错误路径说明

---

## 6. 平台与工程约束

本文档本身即为错误处理规范。

错误处理规范须遵循项目统一工程约束：

- 仅支持 `std` 环境（参见 `require.md §1.3`）
- 保持单 crate 结构
- 遵循 SemVer
- 不引入额外第三方依赖来包装错误模型

| 项目       | 约束                                         |
| ---------- | -------------------------------------------- |
| 平台       | 仅 `std`                                     |
| crate 结构 | 单 crate                                     |
| 依赖       | 不新增第三方依赖                             |
| 一致性     | 不同执行路径下错误类别与诊断字段必须保持一致 |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
