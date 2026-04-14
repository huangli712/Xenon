# 错误处理规范

> 文档编号: 26 | 适用范围: 所有公开 API | 阶段: Phase 1
> 前置文档: `01-architecture.md`, `07-tensor.md`, `21-type.md`
> 需求参考: 需求说明书 §27
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
| 非目标   | 为每个模块单独定义一套错误类型                                |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                     |
| -------- | ------------------------------------------------------------------------ |
| 需求映射 | `require.md §27`                                                         |
| 范围内   | 可恢复错误返回值、panic 分类、诊断字段、类型转换失败、索引失败、FFI 失败 |
| 范围外   | 自定义日志、第三方错误包装器、跨进程序列化                               |
| 非目标   | 通过 `panic` 替代本应可恢复的用户输入错误                                |

### 2.1 关键约束

- 可恢复错误须通过返回值形式报告
- 不可恢复错误须统一通过 panic 暴露
- 所有可恢复错误以及对应 panic 信息都须携带适用的结构化上下文
- `cast()` 失败属于可恢复错误，必须进入 `XenonError`

---

## 3. 影响范围

### 3.1 受影响模块

| 模块/能力            | 影响内容                               |
| -------------------- | -------------------------------------- |
| `tensor` / `shape`   | 形状校验、布局前提、元素总数校验       |
| `index`              | 越界索引、按轴索引、切片边界诊断       |
| `broadcast` / `math` | 广播失败、形状不兼容、参数非法         |
| `reduction`          | 非法轴、空输入、整数溢出 panic         |
| `convert`            | `TypeConversion` 错误及元素索引定位    |
| `ffi`                | FFI 前提失败与后端约束诊断             |
| `parallel`           | panic / `Err` 的立即传播，不得静默吞掉 |

### 3.2 统一依赖方向

> **依赖方向：单向向上。** 错误规范由 `error.rs` 提供基础类型，但本文约束的是所有公开 API 的外部行为，而非某一个源码目录的局部实现。

---

## 4. 规范内容

### 4.1 可恢复错误与 panic 的边界

| 场景                                 | 处理方式                                  | 说明                      |
| ------------------------------------ | ----------------------------------------- | ------------------------- |
| 形状不兼容 / 广播失败                | `Result::Err(XenonError)`                 | 运行时输入决定，可恢复    |
| 轴越界 / 参数非法 / FFI 前提失败     | `Result::Err(XenonError)`                 | 调用方可修正输入并重试    |
| `cast()` 有损或前提不满足            | `Result::Err(XenonError::TypeConversion)` | `require.md §23` 强制要求 |
| 方法型索引失败                       | `Result::Err(XenonError::IndexError)`     | 需返回结构化索引上下文    |
| 语言级 `Index` 语法越界              | panic                                     | 与 Rust 索引惯例保持一致  |
| 整数溢出 / 整数除以零 / 结果不可表示 | panic                                     | 属于不可恢复算术域错误    |

### 4.2 统一错误类型

```rust
use alloc::borrow::Cow;
use core::fmt;

/// Unified recoverable error type for all public Xenon APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum XenonError {
    ShapeMismatch {
        operation: &'static str,
        left_shape: Cow<'static, [usize]>,
        right_shape: Cow<'static, [usize]>,
    },

    BroadcastError {
        input_shape: Cow<'static, [usize]>,
        target_shape: Cow<'static, [usize]>,
        axis: Option<usize>,
    },

    LayoutMismatch {
        operation: &'static str,
        required_layout: &'static str,
        actual_layout: &'static str,
        shape: Cow<'static, [usize]>,
    },

    InvalidAxis {
        axis: usize,
        ndim: usize,
        shape: Cow<'static, [usize]>,
    },

    InvalidShape {
        operation: &'static str,
        shape: Cow<'static, [usize]>,
        expected_elements: Option<usize>,
        actual_elements: Option<usize>,
        offending_dim: Option<usize>,
    },

    DimensionMismatch {
        expected: usize,
        actual: usize,
    },

    EmptyArray {
        operation: &'static str,
        shape: Cow<'static, [usize]>,
    },

    InvalidArgument {
        operation: &'static str,
        argument: &'static str,
        expected: &'static str,
        actual: &'static str,
        axis: Option<usize>,
        shape: Option<Cow<'static, [usize]>>,
    },

    FfiError {
        operation: &'static str,
        backend: &'static str,
        precondition: &'static str,
        actual: &'static str,
    },

    IndexError {
        attempted_index: usize,
        axis: usize,
        shape: Cow<'static, [usize]>,
    },

    TypeConversion {
        source_type: &'static str,
        target_type: &'static str,
        reason: TypeConversionReason,
        element_index: usize,
    },
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

### 4.3 类型转换错误规范

`cast()` 的错误模型须与 `21-type.md` 保持一致：

- `cast<B>(&self)` 返回 `Result<Tensor<B, D>, XenonError>`
- 任何被 `require.md §23` 判定为有损的默认转换组合，都须返回 `XenonError::TypeConversion`
- 仅当需求显式给出附加成功前提时，满足前提后才可成功
- `Complex -> Real` 不是编译期拒绝；当 `im == 0` 时允许继续转换，否则返回 `TypeConversion`
- `bool` 不参与逐元素类型转换，因此不得用 `TypeConversion` 为 `bool` 扩大支持范围

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

| 变体              | 最小结构化字段                                                                   |
| ----------------- | -------------------------------------------------------------------------------- |
| `InvalidArgument` | `operation`, `argument`, `expected`, `actual`, `axis?`, `shape?`                 |
| `FfiError`        | `operation`, `backend`, `precondition`, `actual`                                 |
| `InvalidShape`    | `operation`, `shape`, `expected_elements?`, `actual_elements?`, `offending_dim?` |
| `IndexError`      | `attempted_index`, `axis`, `shape`                                               |
| `TypeConversion`  | `source_type`, `target_type`, `reason`, `element_index`                          |

### 4.5 Display 与 panic 信息要求

Display 输出和 panic 文本都必须能让调用方定位问题来源；最少应包含操作名、错误类别以及适用上下文。

```rust
impl fmt::Display for XenonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeConversion {
                source_type,
                target_type,
                reason,
                element_index,
            } => write!(
                f,
                "type conversion failed at element {}: {} -> {} ({:?})",
                element_index,
                source_type,
                target_type,
                reason,
            ),
            Self::IndexError {
                attempted_index,
                axis,
                shape,
            } => write!(
                f,
                "index {} out of bounds for axis {} of shape [{}]",
                attempted_index,
                axis,
                fmt_shape(shape),
            ),
            _ => todo!("other variants follow the same structured-display rule"),
        }
    }
}
```

### 4.6 必须统一覆盖的 panic 类别

除文档已提到的归约溢出外，以下不可恢复情形都须纳入统一 panic 规范：

- 逐元素整数算术溢出
- 整数除以零
- 结果不可表示（例如 `abs(i32::MIN)`、`i32::MIN / -1`）
- 整数内积的乘积或累加溢出

```rust
// Good - checked arithmetic with explicit panic message.
let value = lhs
    .checked_mul(rhs)
    .expect("integer overflow in dot product: lhs * rhs is not representable");

// Bad - silent wrapping in release mode.
let value = lhs * rhs;
```

### 4.7 并行路径与资源释放

- 并行路径中的 `Err(XenonError)` 须尽快向上传播，不得延后为“全部 worker 完成后再统一检查”
- 并行路径中的 panic 不得被吞掉或伪装为成功结果
- 所有资源释放逻辑不得再触发 panic；在 `panic = abort` 环境下允许进程级终止带来资源不回收

---

## 5. 验证与落地方式

### 5.1 验证清单

| 验证项     | 要求                                                                                |
| ---------- | ----------------------------------------------------------------------------------- |
| 单元测试   | 覆盖 `XenonError` 各变体的 Display、Clone、PartialEq 与结构化字段                   |
| 集成测试   | 覆盖 `reshape`、`broadcast`、`sum_axis`、`cast`、`ffi`、方法型索引等 API 的错误映射 |
| 边界测试   | 覆盖空形状、非法轴、越界索引、复数虚部非零、整数极值、NaN/Inf 转换                  |
| panic 测试 | 覆盖逐元素整数溢出、除以零、`abs(MIN)`、dot overflow                                |
| 并行测试   | 覆盖 `Err` 与 panic 在并行路径中的传播一致性                                        |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
