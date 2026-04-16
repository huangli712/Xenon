# 元素类型体系模块设计

> 文档编号: 03 | 模块: `src/element/` | 阶段: Phase 1
> 前置文档: `00-coding.md`, `01-architecture.md`
> 需求参考: 需求说明书 §4、§5、§12、§13、§14、§15、§23
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责                | 包含                                                                                            | 不包含                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Element trait       | 基础约束（Copy+Clone+PartialEq+Debug+Display+Send+Sync+Sealed）+ zero()/one()                   | —                                                               |
| Numeric trait       | Element + Add+Sub+Mul+Div+Neg + conjugate（通用数值运算能力标记）                               | 运算实现本身（委托给 core::ops）                                |
| Signum trait        | 为 `i32`/`i64`/`f32`/`f64` 提供统一 `signum()` 能力                                             | 复数 signum 或开放类型扩展                                      |
| RealScalar trait    | Numeric + PartialOrd + abs/sqrt/sin/exp/ln/floor/ceil + NaN 检测                                | 复数运算                                                        |
| ComplexScalar trait | Numeric + re/im/norm（复数专用只读能力）                                                        | 复数类型定义（在 `src/complex/` 模块，参见 `04-complex.md` §5） |
| 基础类型实现        | 为 i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool 实现上述 trait；`usize` 仅用于索引/形状元数据 | 类型转换逻辑（在 `src/convert/` 模块）                          |
| Sealed trait        | 封闭集合，禁止外部 crate 实现                                                                   | 开放扩展                                                        |

### 1.2 设计原则

| 原则          | 体现                                                                           |
| ------------- | ------------------------------------------------------------------------------ |
| 能力最小化    | 每层 trait 仅声明必要约束，避免过度限制泛型                                    |
| 正交性        | 数值运算（Numeric）、实数函数（RealScalar）、复数运算（ComplexScalar）职责分离 |
| 零运行时开销  | 所有约束为编译期静态分派                                                       |
| 封闭集合      | Sealed trait 阻止下游 crate 扩展类型集                                         |
| IEEE 754 兼容 | 浮点特殊值（NaN、Inf）处理遵循标准语义                                         |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: element  ← current module
L1: complex (element depends on complex type definitions; complex does not depend on element)
L2: layout (depends on dimension)
L3: storage (depends only on core/alloc)
L4: tensor (depends on storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

> **说明**：`element` 模块位于 L1 层级，但内部依赖同级的 `complex` 模块（`complex` 不依赖 `element`）。这是 L1 内部的单向依赖，`element` 使用 `Complex<T>` 类型作为 trait 实现目标，`complex` 仅提供类型定义和基础运算，不涉及 `Element`/`Numeric` 等 trait。这种单向依赖是合理的。

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                                      |
| -------- | ------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §4、§5、§12、§13、§14、§15、§20、§23、§24                      |
| 范围内   | `Element`/`Numeric`/`RealScalar`/`ComplexScalar` trait 与封闭元素类型集合 |
| 范围外   | 张量存储、自动类型提升、开放外部元素扩展、具体类型转换执行逻辑            |
| 非目标   | 引入新的基础数值类型集合、运行时类型擦除或动态分派元素系统                |

---

## 3. 文件位置

```
src/element/
├── mod.rs             # Element trait definitions and module re-exports
├── numeric.rs         # Numeric trait definitions (arithmetic bounds)
├── real.rs            # RealScalar trait definitions (real math functions)
├── complex.rs         # ComplexScalar trait definitions (complex operations)
└── primitives.rs      # Trait implementations for primitive element types
```

模块内聚设计：四层 trait 按文件分离，基础类型实现集中在 `primitives.rs`。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/element/
├── crate::error      # XenonError for recoverable conversion diagnostics
├── crate::complex    # Complex<T> type definition
├── crate::private    # Sealed trait infrastructure
├── core::ops         # Add/Sub/Mul/Div/Neg operator traits
├── core::fmt         # Debug/Display formatting
└── core::cmp         # PartialEq/PartialOrd comparisons
```

### 4.2 依赖精确到类型级

| 来源模块         | 使用的类型/trait                                           |
| ---------------- | ---------------------------------------------------------- |
| `crate::error`   | `XenonError`（显式类型转换失败时返回）                     |
| `crate::complex` | `Complex<f32>`, `Complex<f64>`（元素类型实现目标）         |
| `crate::private` | `Sealed`（封闭 trait 实现边界）                            |
| `core::ops`      | `Add`, `Sub`, `Mul`, `Div`, `Neg`（Numeric supertrait）    |
| `core::fmt`      | `Debug`, `Display`（Element supertrait）                   |
| `core::cmp`      | `PartialEq`, `PartialOrd`（Element/RealScalar supertrait） |

### 4.2a 依赖合法性与新增依赖说明

| 项目           | 结论                       |
| -------------- | -------------------------- |
| 新增第三方依赖 | 无                         |
| 合法性结论     | 符合需求说明书最小依赖限制 |
| 替代方案       | 不适用                     |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `element/` 消费 `complex` 的类型定义（即 `element` 依赖 `complex`），`complex` 不反向依赖 `element`。
> 被下游消费：`math`（参见 `11-math.md` §4）、`reduction`（参见 `13-reduction.md` §4）、`tensor`（参见 `07-tensor.md` §5）等模块使用 Element/Numeric/RealScalar/ComplexScalar 作为泛型约束。

---

## 5. 公共 API 设计

### 5.1 Element trait

```rust
/// Base trait for all tensor element types.
///
/// Sealed: cannot be implemented outside this crate.
/// All tensor elements must be Copy, thread-safe, and have zero/one identities.
pub trait Element:
    Copy
    + Clone
    + PartialEq
    + core::fmt::Debug
    + core::fmt::Display
    + Send
    + Sync
    + Sealed
{
    /// Additive identity (zero).
    fn zero() -> Self;

    /// Multiplicative identity (one).
    fn one() -> Self;
}
```

| Supertrait  | 作用                                   |
| ----------- | -------------------------------------- |
| `Copy`      | 值语义，可按位复制，避免所有权转移开销 |
| `Clone`     | 显式克隆能力                           |
| `PartialEq` | 相等比较，用于断言和测试               |
| `Debug`     | 调试格式输出 `{:?}`                    |
| `Display`   | 用户友好格式输出 `{}`                  |
| `Send`      | 可跨线程移动（并行迭代必需）           |
| `Sync`      | 可跨线程共享引用（并行只读访问必需）   |
| `Sealed`    | 防止外部类型实现                       |

### 5.2 Numeric trait

```rust
/// Numeric element trait.
///
/// Adds arithmetic operations on top of Element.
/// Xenon's generic numeric core currently covers signed integers, real scalars,
/// and complex scalars: `i32`, `i64`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
/// `bool` is explicitly excluded. `usize` is reserved for index/shape metadata
/// and is not part of the tensor element set.
///
/// The native operator supertraits describe syntax availability only.
/// Overflow-sensitive integer paths must additionally follow Xenon's checked
/// arithmetic contracts in operation modules so that recoverable vs panic
/// behavior remains consistent with `require.md`.
///
/// Note: `Sealed` is not listed as a separate supertrait here because
/// `Element` already inherits `Sealed`.
pub trait Numeric:
    Element
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    /// Returns the canonical conjugate of the value.
    ///
    /// Real-valued types return `self` unchanged (identity).
    /// Complex-valued types return the mathematical conjugate.
    fn conjugate(self) -> Self;
}
```

| 必需项                | 语义说明                                                    |
| --------------------- | ----------------------------------------------------------- |
| `Add/Sub/Mul/Div/Neg` | 提供通用数值四则运算与取负能力                              |
| `conjugate(self)`     | 统一提供共轭语义：实数类型返回 `self`，复数类型返回数学共轭 |

> **设计决策：** `Numeric` 在保留通用算术分层的同时，通过 `Numeric::conjugate()` 统一提供共轭语义：实数类型为恒等操作，复数类型执行数学共轭。`ComplexScalar` 保留复数专用能力（`re`/`im`/`norm`），不再单独承担 `conjugate` 的唯一 trait 入口角色。这与 `01-architecture.md`、`11-math.md`、`12-matrix.md` 的泛型约定保持一致。
>
> **整数算术契约**：`Add/Sub/Mul/Div/Neg` 只表达运算符可用性，不单独定义 Xenon 的溢出语义。凡需求文档要求“溢出/除零/结果不可表示即 panic”的整数运算路径，具体模块必须通过 checked 标量原语或等价显式检查落实，不得仅凭原生运算符 trait 假定语义成立。

### 5.2a Signum trait

> **公开性说明：** 以下 trait 为内部实现辅助，不纳入稳定公开 API 面。具体可见性由实现决定。

```rust
/// Unified signum capability for the sealed signed scalar set.
///
/// Implemented only for `i32`, `i64`, `f32`, and `f64`.
/// Integer implementations return `-1`, `0`, or `1`.
/// Floating-point implementations follow IEEE 754 sign-function semantics.
pub(crate) trait Signum: Numeric + Sealed {
    fn signum(self) -> Self;
}

impl Signum for i32 {
    #[inline]
    fn signum(self) -> Self { i32::signum(self) }
}

impl Signum for i64 {
    #[inline]
    fn signum(self) -> Self { i64::signum(self) }
}

impl Signum for f32 {
    #[inline]
    fn signum(self) -> Self { f32::signum(self) }
}

impl Signum for f64 {
    #[inline]
    fn signum(self) -> Self { f64::signum(self) }
}
```

> **适用范围说明：** 张量级 `signum()` 公开能力需覆盖 `i32`、`i64`、`f32`、`f64`（对应 `require.md §12`）。因此本设计将 `signum` 收敛到独立的 sealed `Signum` 子 trait：整数通过 `-1/0/1` 语义实现，浮点继续服从 `RealScalar::signum()` 的 IEEE 754 契约；`Complex<T>` 与 `bool` 不实现该能力。

### 5.3 RealScalar trait

```rust
/// Real-valued scalar trait.
///
/// Provides math functions, constants, and NaN detection.
/// Only f32 and f64 implement this trait.
pub trait RealScalar: Numeric + PartialOrd + Sealed {
    // Sealed is already inherited via Element (which Numeric extends),
    // but listed here for defensive clarity — makes the sealed intent explicit
    // at each trait level.
    // ========== Math functions ==========
    fn abs(self) -> Self;
    /// Returns the IEEE 754 sign of the value.
    ///
    /// For finite non-zero inputs, the result is `1.0` or `-1.0` with the
    /// same sign bit as the input. `signum(+0.0) == +0.0`,
    /// `signum(-0.0) == -0.0`, `signum(+∞) == 1.0`, `signum(-∞) == -1.0`,
    /// and `signum(NaN) == NaN`.
    fn signum(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;

    // ========== Constants ==========
    // Internal convenience accessors only; keep the minimum set actually needed.
    fn epsilon() -> Self;
    fn min_positive() -> Self;
    fn max_value() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn nan() -> Self;

    // ========== Special value detection ==========
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;

    // ========== NaN-propagating min/max ==========
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}
```

> **设计决策：** `min`/`max` 采用 NaN 传播语义（任一参数为 NaN → 返回 NaN）。Xenon 通过 `RealScalar` 自身的契约固定这一行为，不直接复用标准库 `f32::min` / `f64::min` 的全部语义细节。
>
> **跨模块约束：** `math`、`reduction`、`format` 等任何需要依赖该语义的模块，必须调用 `RealScalar::min()` / `RealScalar::max()` 或与其完全等价的封装，不得直接退回标准库固有方法，否则会破坏 Xenon 在 NaN 传播上的统一契约。
>
> **`signum` 语义补充：** `RealScalar::signum()` 明确采用 IEEE 754 sign-function 语义：保留带符号零，`NaN` 传播为 `NaN`，无穷值返回对应符号的单位值。`11-math.md` 中张量级 `signum()` 的浮点语义以此 trait 契约为权威基线；整数 `signum` 仍按比较结果返回 `-1/0/1`。
>
> **常量访问器说明：** `epsilon`、`min_positive`、`max_value`、`infinity`、`neg_infinity`、`nan` 为内部辅助方法，保留当前版本实际需要的最小集。以下常量访问器为实现便利，不构成稳定 API 承诺。
>
> **NaN 语义显式说明：** 此方法采用 NaN 传播语义（任一参数为 NaN → 返回 NaN），与标准库 `f32::min` / `f64::min` 的 NaN 处理不同。命名为 `min` / `max` 而非 `minimum` / `maximum` 是为了 API 简洁性。

### 5.4 ComplexScalar trait

```rust
/// Complex scalar trait.
///
/// Provides the minimal complex-specific operations required by the current
/// tensor API surface. Only Complex<f32> and Complex<f64> implement this.
pub trait ComplexScalar: Numeric + Sealed {
    // Sealed is already inherited via Element (which Numeric extends),
    // but listed here for defensive clarity — makes the sealed intent explicit
    // at each trait level.
    /// Real part type (must be RealScalar).
    type Real: RealScalar;

    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    fn norm(self) -> Self::Real;
}
```

> **设计说明：** `Numeric::conjugate()` 是全体数值元素唯一的统一共轭入口：实数路径返回恒等，复数路径返回数学共轭。`ComplexScalar` 只保留 `re` / `im` / `norm` 这类复数专用能力，不再重复声明 `conjugate()`，从而避免两个公开 trait 暴露同名契约。
>
> **范围说明：** `ComplexScalar` 公开面仅保留当前范围内真正需要的复数能力。`arg`/`exp`/`ln`/`sqrt`/`from_polar`/`i` 等超出当前张量 API 范围的方法若实现需要，降为 `complex` 模块内部 helper，不放入本公开 trait。

### 5.4a OrderedCompareElement trait

> **公开性说明：** 以下 trait 为内部实现辅助，不纳入稳定公开 API 面。具体可见性由实现决定。

```rust
/// Ordered comparison element trait.
///
/// Restricts public ordered comparisons (`lt` / `gt`) to element types with the
/// current version's required ordering semantics.
pub(crate) trait OrderedCompareElement: Element + PartialOrd + Sealed {}

impl OrderedCompareElement for i32 {}
impl OrderedCompareElement for i64 {}
impl OrderedCompareElement for f32 {}
impl OrderedCompareElement for f64 {}
```

> **设计决策：** `OrderedCompareElement` 用于把有序比较能力显式收敛到 `i32`、`i64`、`f32`、`f64`。这与需求说明书 §12 的 `lt` / `gt` 适用类型保持一致，并避免 `bool` 或 `Complex<T>` 因泛化的 `PartialOrd` 约束误入公开比较 API。

### 5.5 支持的类型与 trait 矩阵

| 类型           | Element | Numeric | RealScalar | ComplexScalar |
| -------------- | :-----: | :-----: | :--------: | :-----------: |
| `i32`          |    ✓    |    ✓    |     ✗      |       ✗       |
| `i64`          |    ✓    |    ✓    |     ✗      |       ✗       |
| `f32`          |    ✓    |    ✓    |     ✓      |       ✗       |
| `f64`          |    ✓    |    ✓    |     ✓      |       ✗       |
| `Complex<f32>` |    ✓    |    ✓    |     ✗      |       ✓       |
| `Complex<f64>` |    ✓    |    ✓    |     ✗      |       ✓       |
| `bool`         |    ✓    |    ✗    |     ✗      |       ✗       |

> **OrderedCompareElement 适用类型：** `i32`、`i64`、`f32`、`f64`。

> **Xenon 特定约束：** 仅支持上表列出的 7 种元素类型。不支持 `usize`、u8/u16/u32/i8/i16 等其他整数类型；`usize` 仅作为索引和形状元数据使用。

### 5.6 BoolElement trait

> **公开性说明：** 以下 trait 为内部实现辅助，不纳入稳定公开 API 面。具体可见性由实现决定。

```rust
/// Bool-specific element capability.
///
/// Provides logical NOT for `bool` tensors without making `bool` part of `Numeric`.
pub(crate) trait BoolElement: Element + core::ops::Not<Output = Self> {
    fn logical_not(self) -> Self;
}

impl BoolElement for bool {
    #[inline]
    fn logical_not(self) -> Self {
        !self
    }
}
```

### 5.7 Sealed trait 策略

```rust
// src/element/mod.rs
// Uses the shared Sealed trait from crate::private
// (see src/private.rs, referenced in 01-architecture.md §3)
use crate::private::Sealed;

// Sealed implementations in primitives.rs
impl Sealed for i32 {}
impl Sealed for i64 {}
impl Sealed for f32 {}
impl Sealed for f64 {}
impl Sealed for Complex<f32> {}
impl Sealed for Complex<f64> {}
impl Sealed for bool {}
```

外部 crate 尝试实现时编译错误：

```rust
// External crate attempt:
use xenon::element::Element;
struct MyType;
impl Element for MyType { /* error[E0277]: Sealed not satisfied */ }
```

### 5.8 Good / Bad 对比示例

```rust
// Good - Numeric constraint automatically excludes bool and non-Numeric types
fn sum<'a, A, D>(tensor: &TensorView<'a, A, D>) -> A
where
    A: Numeric,
    D: Dimension,
{
    tensor.iter().fold(A::zero(), |acc, &x| acc + x)
}
// sum(&bool_tensor);   // Compile error: bool does not satisfy Numeric

// Bad - Element constraint cannot exclude non-arithmetic element types
fn sum_bad<'a, A, D>(tensor: &TensorView<'a, A, D>) -> A
where
    A: Element,
    D: Dimension,
{
    // Cannot use + operator, Element has no Add bound
    todo!()
}
```

```rust
// Good - explicit type conversion, no automatic promotion
let a: Tensor<f64, Ix2> = Tensor::zeros((3, 4));
let b: Tensor<i32, Ix2> = Tensor::zeros((3, 4));
let b64 = b.cast::<f64>()?;
let c = &a + &b64;

// Bad - expecting automatic type promotion (not supported in Xenon)
// let c = &a + &b;  // Compile error: no matching impl for f64 + i32
```

### 5.9 CastTo\<T\> trait（类型转换）

`CastTo<T>` 的 trait 定义位于 `src/element/mod.rs`，具体 impl 与错误构造示例统一放在 `src/convert/cast.rs`；`convert/` 负责消费该 trait 并承载受支持转换矩阵的实现（参见 `21-type.md §5.1`），不在其他模块重复定义或分散实现。

```rust
// src/element/mod.rs

/// Element-wise type conversion trait.
///
/// Defines explicit conversion from `Self` to `T`.
/// Lossless conversions return `Ok(T)`.
/// Lossy conversions default to recoverable `XenonError` unless a documented
/// success precondition is satisfied (see `21-type.md §5.3`).
///
/// This trait is implemented only inside Xenon for the supported source/target pairs.
/// External crates cannot extend the conversion matrix.
pub trait CastTo<T>: Element {
    /// Performs the type conversion.
    fn cast_to(self) -> Result<T, XenonError>;
}

// Implemented for all supported source → target type pairs.
// See 21-type.md §5.3 for full implementation table.
// Examples:
//   impl CastTo<f64> for f32  -- lossless upcast
//   impl CastTo<f32> for f64  -- Result, may fail on precision loss
//   impl CastTo<i32> for f64  -- Result, may fail on fractional or out-of-range input
//   impl CastTo<i32> for i64  -- Result, may fail on narrowing overflow
//   impl CastTo<Complex<f64>> for f64  -- lossless
//   impl CastTo<f64> for Complex<f64>  -- Result, requires zero imaginary part
```

> **错误映射说明：** `XenonError` 是唯一公开错误类型。类型转换错误对外统一表示为 `XenonError::TypeConversion(TypeConversionError)`；若实现内部保留 `TypeConversionReason` 等细节类型，也只能作为 `TypeConversionError` 的内部载荷，不能成为公开 API 的错误返回类型。
>
> **Bool 边界说明：** `bool` 不为任何目标类型实现 `CastTo<T>`；`bool` 张量调用 `.cast::<f32>()` 等转换必须在编译期失败。
>
> **无损/有损区分说明：** 同类型拷贝和无损转换虽然通过 `Result` 返回，但按契约语义上不可失败。调用方仍应按项目错误处理规范选择 `?` 或显式处理；实现层不应依赖 `unwrap` 作为常规路径。

`Complex<T> → Real` 转换已纳入 `CastTo<T>` 的支持矩阵，但采用条件成功语义：仅当虚部为 `0` 时才继续执行对应的实数目标转换；若虚部非零，则返回可恢复错误。具体规则如下：

| 转换示例                                | 默认语义                                                             |
| --------------------------------------- | -------------------------------------------------------------------- |
| `impl CastTo<f32> for Complex<f32>`     | 虚部必须为 `0`；满足时按 `f32 -> f32` 恒等转换，否则返回可恢复错误   |
| `impl CastTo<f64> for Complex<f64>`     | 虚部必须为 `0`；满足时按 `f64 -> f64` 恒等转换，否则返回可恢复错误   |
| `impl CastTo<f32> for Complex<f64>`     | 虚部必须为 `0`；满足时按 `f64 -> f32` 有损转换，否则返回可恢复错误   |
| `impl CastTo<f64> for Complex<f32>`     | 虚部必须为 `0`；满足时按 `f32 -> f64` 无损转换，否则返回可恢复错误   |
| `impl CastTo<i32> for Complex<f32/f64>` | 虚部必须为 `0`；满足时按 `float -> i32` 规则转换，否则返回可恢复错误 |
| `impl CastTo<i64> for Complex<f32/f64>` | 虚部必须为 `0`；满足时按 `float -> i64` 规则转换，否则返回可恢复错误 |

```rust
// src/convert/cast.rs

use crate::error::{TypeConversionError, TypeConversionReason, XenonError};

impl CastTo<f32> for Complex<f32> {
    fn cast_to(self) -> Result<f32, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f32>".into(),
                target_type: "f32".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        Ok(self.re)
    }
}

impl CastTo<f64> for Complex<f64> {
    fn cast_to(self) -> Result<f64, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f64>".into(),
                target_type: "f64".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        Ok(self.re)
    }
}

impl CastTo<f32> for Complex<f64> {
    fn cast_to(self) -> Result<f32, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f64>".into(),
                target_type: "f32".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        CastTo::<f32>::cast_to(self.re)
    }
}

impl CastTo<f64> for Complex<f32> {
    fn cast_to(self) -> Result<f64, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f32>".into(),
                target_type: "f64".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        CastTo::<f64>::cast_to(self.re)
    }
}

impl CastTo<i32> for Complex<f32> {
    fn cast_to(self) -> Result<i32, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f32>".into(),
                target_type: "i32".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        CastTo::<i32>::cast_to(self.re)
    }
}

impl CastTo<i64> for Complex<f64> {
    fn cast_to(self) -> Result<i64, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f64>".into(),
                target_type: "i64".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        CastTo::<i64>::cast_to(self.re)
    }
}

impl CastTo<i32> for Complex<f64> {
    fn cast_to(self) -> Result<i32, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f64>".into(),
                target_type: "i32".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        CastTo::<i32>::cast_to(self.re)
    }
}

impl CastTo<i64> for Complex<f32> {
    fn cast_to(self) -> Result<i64, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f32>".into(),
                target_type: "i64".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }));
        }
        CastTo::<i64>::cast_to(self.re)
    }
}
```

> **实现约束：** `Complex<T> -> Real` 的所有 `CastTo` 实现都必须先检查虚部是否为 `0`。检查通过后，再复用对应实部的标量转换规则；检查失败时统一返回 `XenonError::TypeConversion(TypeConversionError)`。`TypeConversionError::element_index` 是必填字段：标量级示例使用 `0` 作为当前位置，占位表示“当前唯一元素”；张量级批量转换则必须填写实际失败元素索引。

> **类型级能力说明：** 理论上可进一步把 `CastTo<T>` 拆为 lossless / lossy 两组 trait，以在类型层编码“绝不失败”与“可能失败”的差异；但当前版本仍统一返回 `Result<T, XenonError>`，以保持转换入口单一、错误载荷一致，并避免为有限封闭类型集合引入第二套 trait 维度与额外泛型分支。

| 转换示例                   | 默认语义                                        |
| -------------------------- | ----------------------------------------------- |
| `impl CastTo<i64> for i32` | 无损，`Result<i64, XenonError>`（实际不会失败） |
| `impl CastTo<f64> for f32` | 无损，`Result<f64, XenonError>`（实际不会失败） |
| `impl CastTo<f32> for f64` | 有损，默认返回可恢复错误                        |
| `impl CastTo<i32> for f64` | 有损，默认返回可恢复错误                        |
| `impl CastTo<i32> for i64` | 有损，默认返回可恢复错误                        |

### 5.10 Checked arithmetic traits（整数溢出检测）

`CheckedAdd` 为整数类型提供 checked 加法，供 `sum` 归约操作在整数溢出时 panic（参见 `13-reduction.md §5.1`）。

> **公开性说明：** 以下 trait 为内部实现辅助，不纳入稳定公开 API 面。具体可见性由实现决定。

```rust
// src/element/mod.rs

/// Checked addition for types that support it.
///
/// Returns `None` on overflow instead of wrapping.
/// Only implemented for integer types (`i32`, `i64`).
/// Float types use ordinary `+` (NaN propagation handles the semantics).
///
/// Used by integer `sum()` reduction to guarantee overflow is detected
/// in both debug and release builds (per requirement §14).
pub(crate) trait CheckedAdd: Numeric {
    /// Returns `Some(self + rhs)` if no overflow, `None` otherwise.
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

impl CheckedAdd for i32 {
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        i32::checked_add(self, rhs)
    }
}

impl CheckedAdd for i64 {
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        i64::checked_add(self, rhs)
    }
}
// f32, f64, Complex: NOT implemented — overflow is handled by IEEE 754 semantics.
// bool: NOT implemented — not Numeric.

/// Checked subtraction for integer-only overflow-sensitive paths.
pub(crate) trait CheckedSub: Numeric + Sealed {
    fn checked_sub(self, rhs: Self) -> Option<Self>;
}

impl CheckedSub for i32 {
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> { i32::checked_sub(self, rhs) }
}

impl CheckedSub for i64 {
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> { i64::checked_sub(self, rhs) }
}

/// Checked multiplication for integer-only overflow-sensitive paths.
pub(crate) trait CheckedMul: Numeric + Sealed {
    fn checked_mul(self, rhs: Self) -> Option<Self>;
}

impl CheckedMul for i32 {
    #[inline]
    fn checked_mul(self, rhs: Self) -> Option<Self> { i32::checked_mul(self, rhs) }
}

impl CheckedMul for i64 {
    #[inline]
    fn checked_mul(self, rhs: Self) -> Option<Self> { i64::checked_mul(self, rhs) }
}

/// Checked negation for integer-only overflow-sensitive paths.
pub(crate) trait CheckedNeg: Numeric + Sealed {
    fn checked_neg(self) -> Option<Self>;
}

impl CheckedNeg for i32 {
    #[inline]
    fn checked_neg(self) -> Option<Self> { i32::checked_neg(self) }
}

impl CheckedNeg for i64 {
    #[inline]
    fn checked_neg(self) -> Option<Self> { i64::checked_neg(self) }
}
```

> **设计决策：** `CheckedAdd` 仅覆盖整数加法（`i32`/`i64`），用于归约等必须精确检测溢出的路径。逐元素整数乘法不通过 `CheckedAdd` 约束暴露，具体溢出策略由对应运算模块单独规定。
>
> **补充说明：** 当前元素层统一提供 `CheckedAdd` / `CheckedSub` / `CheckedMul` / `CheckedNeg` 四类整数 checked 原语，供 `math`、`matrix`、`reduction` 等模块复用；除法、余数与更高阶组合检查仍由具体运算模块在实现层完成（参见 `11-math.md`、`12-matrix.md`）。

> **架构同步说明：** 本文新增的 `BoolElement`、`OrderedCompareElement`、`Checked*`、`CastTo` 等 trait 须与 `01-architecture.md` 的核心类型速查表保持同步。

---

## 6. 内部实现设计

### 6.1 bool 排除策略

`bool` 仅实现 `Element`，不实现 `Numeric`：

```rust
// primitives.rs
impl Element for bool {
    fn zero() -> Self { false }
    fn one() -> Self { true }
}
// No `impl Numeric for bool` — bool arithmetic has no mathematical meaning
impl BoolElement for bool {
    fn logical_not(self) -> Self { !self }
}
```

编译时阻止无效泛型实例化：`fn sum<A: Numeric>` 无法接受 `bool` 张量；需要布尔专用逐元素逻辑非时，使用 `BoolElement::logical_not()` 或 `!`。

此外，`bool` 不实现任何 `CastTo<T>`；`bool_tensor.cast::<f32>()` 必须在编译期失败，而不是返回运行时类型转换错误。

### 6.2 usize 语义边界

`usize` 不属于 Xenon 的张量元素集合，仅作为索引、轴和形状元数据类型使用。所有元素 trait（`Element`/`Numeric`/`RealScalar`/`ComplexScalar`）都不为 `usize` 提供实现，也不再为其预留算术扩展路径。

### 6.3 类型提升规则

**Xenon 不支持自动类型提升。** 所有跨类型运算须显式转换：

```rust
// Implicit conversion not supported
// let a: f64 = 1.0;
// let b: i32 = 2;
// let c = a + b;  // Compile error

// Must convert explicitly through Xenon's cast contract
let c = a + b.cast_to()?;
```

### 6.4 NaN/Inf 处理语义

| 方法                  | NaN 输入 | Inf 输入 |
| --------------------- | -------- | -------- |
| `abs(NaN)`            | NaN      | Inf      |
| `sqrt(-1.0)`          | NaN      | —        |
| `ln(0.0)`             | —        | -Inf     |
| `exp(Inf)`            | —        | Inf      |
| `min(a, b)` 含 NaN    | NaN      | 正常比较 |
| `partial_cmp(NaN, _)` | None     | 正常比较 |

### 6.5 RealScalar 实现（以 f64 为例）

```rust
impl Numeric for i32 {
    #[inline]
    fn conjugate(self) -> Self { self }
}

impl Numeric for i64 {
    #[inline]
    fn conjugate(self) -> Self { self }
}

impl Numeric for f32 {
    #[inline]
    fn conjugate(self) -> Self { self }
}

impl Numeric for f64 {
    #[inline]
    fn conjugate(self) -> Self { self }
}

// RealScalar math functions are implemented in the `std` environment required by Xenon.
impl RealScalar for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn floor(self) -> Self { self.floor() }
    fn ceil(self) -> Self { self.ceil() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else if self <= other {
            self
        } else {
            other
        }
    }

    fn max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else if self >= other {
            self
        } else {
            other
        }
    }
    // ...
}
// Same pattern applies to f32.

impl Numeric for Complex<f64> {
    #[inline]
    fn conjugate(self) -> Self { Self::conj(self) }
}

impl ComplexScalar for Complex<f64> {
    type Real = f64;

    fn re(self) -> Self::Real { self.re }
    fn im(self) -> Self::Real { self.im }
    fn norm(self) -> Self::Real { self.norm() }
}
// Same pattern applies to Complex<f32>.
```

> **补充说明：** `i32`/`i64`/`f32`/`f64` 作为实数路径上的 `Numeric` 实现，`conjugate()` 一律为恒等操作；`Complex<f32>`/`Complex<f64>` 的数学共轭也统一通过 `Numeric::conjugate()` 暴露，`ComplexScalar` 实现只补充复数特有的 `re`/`im`/`norm`。

---

## 7. 实现任务拆分

### Wave 1: 基础 trait 定义

- [ ] **T1**: 创建 `mod.rs`，导入共享 Sealed trait 并定义 Element trait
  - 文件: `src/element/mod.rs`
  - 内容: 从 `crate::private` 导入 `Sealed`，定义 `Element` trait、模块 re-export
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 创建 `numeric.rs`，定义 Numeric trait 及其核心方法契约
  - 文件: `src/element/numeric.rs`
  - 内容: `Numeric` trait 定义（四则运算 supertrait + 统一 `conjugate()` 语义）
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: 扩展 trait 定义

- [ ] **T3**: 创建 `real.rs`，定义 RealScalar trait
  - 文件: `src/element/real.rs`
  - 内容: `RealScalar` trait 定义（数学函数 + 常量 + NaN 检测，不含范围外接口）
  - 测试: 编译通过
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 创建 `complex.rs`，定义 ComplexScalar trait
  - 文件: `src/element/complex.rs`
  - 内容: `ComplexScalar` trait 定义（关联类型 Real + 复数方法）
  - 测试: 编译通过
  - 前置: T2
  - 预计: 10 min

### Wave 3: 基础类型实现

- [ ] **T5**: 为 i32/i64 实现 Element + Numeric
  - 文件: `src/element/primitives.rs`
  - 内容: `Sealed` impl、`Element` impl、`Numeric` impl
  - 测试: `test_i32_zero_one`, `test_i64_arithmetic`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 为 f32/f64 实现 Element + Numeric + RealScalar
  - 文件: `src/element/primitives.rs`
  - 内容: 三层 trait 实现 + 数学函数委托
  - 测试: `test_f64_sqrt`, `test_f32_nan_detection`
  - 前置: T3, T5
  - 预计: 10 min

- [ ] **T7**: 为 bool 实现 Element（仅此）
  - 文件: `src/element/primitives.rs`
  - 内容: `Element` impl（`zero()=false`, `one()=true`）+ `BoolElement::logical_not()`，不实现 Numeric
  - 测试: `test_bool_element_only`
  - 前置: T1
  - 预计: 5 min

- [ ] **T8**: 补充索引/形状侧对 `usize` 的边界说明
  - 文件: `src/element/mod.rs`
  - 内容: 文档中明确 `usize` 仅作为索引和形状元数据使用，不属于元素 trait 实现集合
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 4: 复数类型实现

- [ ] **T9**: 为 Complex<f32>/Complex<f64> 实现 Element + Numeric + ComplexScalar
  - 文件: `src/element/primitives.rs`
  - 内容: 三层 trait 实现，关联类型 `Real = f32`/`f64`
  - 测试: `test_complex_f64_conj`, `test_complex_f32_norm`
  - 前置: T4, T5
  - 预计: 10 min

### Wave 5: 集成完善

- [ ] **T10**: 校准数学能力与转换错误语义文档
  - 文件: `src/element/real.rs`, `src/element/mod.rs`
  - 内容: 保持 `std` 环境下的数学接口边界，并将有损 CastTo 默认语义标注为可恢复错误
  - 测试: 编译通过
  - 前置: T6
  - 预计: 10 min

- [ ] **T11**: 文档注释与 cargo doc 验证
  - 文件: 所有 `src/element/` 文件
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T9
  - 预计: 10 min

- [ ] **T12**: 集成测试（跨模块交互验证）
  - 文件: `tests/test_element.rs`
  - 内容: 各类型各层 trait 的完整性验证
  - 测试: 见测试计划 §8
  - 前置: T10, T11
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
Wave 2: [T3]      [T4]       [T5]
        │           │           │
        │           │           ▼
Wave 3: [T6]      [T9] ← ────┘
         │          │
        [T10]      [T11] → [T12]

(T7, T8 are independent of Wave 2-3 and can run in parallel with T3/T4/T5)
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                           | 说明                                                                  |
| -------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`                       | 验证各 trait 和基础类型实现                                           |
| 集成测试 | `tests/test_element.rs`                        | 验证 `element` 与 `tensor`、`math`、`reduction`、`convert` 的协同路径 |
| 边界测试 | 同模块测试中标注                               | 覆盖 NaN/Inf、bool 限制与 sealed 行为                                 |
| 属性测试 | `tests/test_element.rs` 或 `tests/property.rs` | 验证零元、单位元与数学函数不变量                                      |

### 8.2 单元测试清单

| 测试函数                        | 测试内容                                                   | 优先级 |
| ------------------------------- | ---------------------------------------------------------- | ------ |
| `test_i32_zero_one`             | `i32::zero()==0`, `i32::one()==1`                          | 高     |
| `test_i64_zero_one`             | `i64::zero()==0`, `i64::one()==1`                          | 高     |
| `test_i32_arithmetic`           | `i32` 的 Add/Sub/Mul/Div/Neg                               | 高     |
| `test_f32_zero_one`             | `f32::zero()==0.0`, `f32::one()==1.0`                      | 高     |
| `test_f64_zero_one`             | `f64::zero()==0.0`, `f64::one()==1.0`                      | 高     |
| `test_f64_sqrt`                 | `f64::sqrt(4.0)==2.0`                                      | 高     |
| `test_f64_sin`                  | `sin(0)==0`                                                | 高     |
| `test_f64_exp_ln_inverse`       | 对 `x > 0` 且有限输入使用容差断言验证 `exp(ln(x)) ≈ x`     | 高     |
| `test_f32_nan_detection`        | `NaN.is_nan()`, `Inf.is_infinite()`                        | 高     |
| `test_f64_nan_propagating_min`  | `min(NaN, 1.0).is_nan()`                                   | 高     |
| `test_bool_element_only`        | `bool::zero()==false`, `bool::one()==true`                 | 高     |
| `test_bool_not_numeric`         | bool 不满足 Numeric（编译测试）                            | 高     |
| `test_bool_cast_to_f32_fails`   | `bool` 张量 `.cast::<f32>()` 不可编译（compile-fail）      | 高     |
| `test_usize_not_element`        | `usize` 不属于 Element（编译测试）                         | 中     |
| `test_complex_f64_zero_one`     | `Complex<f64>::zero()`, `Complex<f64>::one()`              | 高     |
| `test_complex_f64_conj`         | `Complex::new(3.0, 4.0).conj() == Complex::new(3.0, -4.0)` | 高     |
| `test_complex_f32_norm`         | `Complex::new(3.0f32, 4.0f32).norm() == 5.0`               | 高     |
| `test_sealed_prevents_external` | 外部类型无法实现 Element（编译测试）                       | 中     |

### 8.3 边界测试场景

| 场景                                          | 预期行为                                                 |
| --------------------------------------------- | -------------------------------------------------------- |
| `f64::nan().is_nan()`                         | 返回 `true`                                              |
| `f64::infinity().is_finite()`                 | 返回 `false`                                             |
| `f64::sqrt(-1.0).is_nan()`                    | 返回 `true`                                              |
| `f64::ln(0.0)`                                | 返回 `-Inf`                                              |
| `Complex::new(f64::NAN, 0.0).norm().is_nan()` | 返回 `true`                                              |
| `bool` 张量调用 `sum()`                       | 编译错误（Numeric 约束不满足）                           |
| `bool` 张量调用 `.cast::<f32>()`              | 编译错误（未实现 `CastTo<f32>`）                         |
| §28.4 占位：large-tensor                      | 后续补充大批量元素类型转换/归约边界回归                  |
| §28.4 占位：high-dim                          | 后续补充高维张量在元素 trait 分发上的回归                |
| §28.4 占位：extreme-value                     | 后续补充 `NaN` / `Inf` / `MIN` / `MAX` / `-0.0` 组合回归 |

### 8.4 属性测试不变量

| 不变量                                    | 测试方法                  |
| ----------------------------------------- | ------------------------- |
| `A::zero() + a == a`                      | 所有 Numeric 类型，随机 a |
| `A::one() * a == a`                       | 所有 Numeric 类型，随机 a |
| `(!b) == BoolElement::logical_not(b)`     | `bool`                    |
| `let y = a.sqrt(); (y * y) ≈ a`（容差内） | f32/f64，随机非负数 a     |
| `a.exp().ln() ≈ a`                        | f32/f64，随机有限 a       |
| `x.ln().exp() ≈ x`                        | f32/f64，随机正且有限 x   |

### 8.5 集成测试

| 测试文件                | 测试内容                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------- |
| `tests/test_element.rs` | 各元素类型在 `tensor`、`math`、`reduction`、`convert` 中的 trait 约束与端到端行为验证 |

### 8.6 Feature gate / 配置测试

| 配置项         | 覆盖方式              | 说明                                        |
| -------------- | --------------------- | ------------------------------------------- |
| 默认配置       | 常规单元/集成测试路径 | 本模块无独立 feature gate，默认配置即主路径 |
| 非默认 feature | 不适用                | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型       | 覆盖方式                                                                  | 说明                                           |
| -------------- | ------------------------------------------------------------------------- | ---------------------------------------------- |
| sealed 边界    | compile-fail 测试外部类型实现 `Element`                                   | 验证封闭元素集合不会被外部 crate 扩展          |
| 元素能力边界   | compile-fail 测试 `bool: Numeric`、`bool.cast::<f32>()`、`usize: Element` | 验证布尔与索引元数据类型不会越界进入算术元素层 |
| trait 分层边界 | 编译期验证 `RealScalar`/`ComplexScalar` 仅覆盖规定类型                    | 验证 trait 能力分层不被误扩展                  |

---

## 9. 模块交互设计

### 9.1 接口约定

| 模块           | 使用的 trait                | 用途                   |
| -------------- | --------------------------- | ---------------------- |
| `overload`     | `Numeric`                   | 逐元素运算泛型约束     |
| `reduction`    | `Numeric`（sum）            | 归约运算泛型约束       |
| `tensor`       | `Element`                   | Tensor<A, D> 的 A 约束 |
| `matrix`       | `Numeric` / `ComplexScalar` | 内积运算               |
| `cast/convert` | `Element`                   | 类型转换               |

> 各模块的详细接口约定参见对应设计文档（`11-math.md` §4、`13-reduction.md` §4、`21-type.md` §4）。

### 9.2 接口边界

```
┌───────────────────────────────────────────────────────────────┐
│  math / reduction / matrix (consume Element/Numeric bounds)  │
└──────────────────────┬────────────────────────────────────────┘
                       │ generic bounds
┌──────────────────────▼────────────────────────────────────────┐
│  element (defines traits)                                    │
└──────────────────────┬────────────────────────────────────────┘
                       │ type dependency
┌──────────────────────▼────────────────────────────────────────┐
│  complex (defines Complex<T>)                                │
└───────────────────────────────────────────────────────────────┘
```

### 9.3 数据流描述

```text
Upstream modules declare element bounds
    │
    ├── tensor accepts the sealed element set via `Element` (excluding `usize`)
    ├── math / matrix / reduction select capabilities via `Numeric`, `RealScalar`, and `ComplexScalar`
    ├── convert / set / format continue consuming type-level capabilities or formatting semantics
    └── unsupported element types are rejected by compile-time trait bounds
```

---

## 10. 错误处理与语义边界

| 项目              | 内容                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | 有损 `CastTo` 默认返回可恢复错误；`Complex<T> -> Real` 在虚部非零时也返回可恢复错误；对外统一使用 `XenonError::TypeConversion(TypeConversionError)` |
| Panic             | 本模块 trait 方法本身不以 panic 作为常规错误语义；若底层标准库数学实现遇到其自身前置条件，遵循标准库行为                                            |
| 路径一致性        | scalar 路径必须与普通标量实现一致；SIMD：不适用；parallel：不适用                                                                                   |
| 容差边界          | 浮点相关比较遵循 IEEE 754 与各测试中显式容差；整数与布尔类型不适用                                                                                  |

---

## 11. 设计决策记录

### 决策 1：封闭集合，不支持下游扩展

| 属性     | 值                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| 决策     | 所有 trait 继承 Sealed，仅允许 crate 内类型实现                              |
| 理由     | API 稳定性（可添加新方法不破坏外部）；所有实现类型行为经过验证；版本控制能力 |
| 替代方案 | 开放实现 — 放弃，失去版本控制能力，可能导致不一致行为                        |

### 决策 2：仅支持 7 种元素类型

| 属性     | 值                                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------------- |
| 决策     | 仅支持 i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool 作为张量元素类型；`usize` 仅作为索引/形状元数据 |
| 理由     | 科学计算元素类型需要稳定且平台无关的数值语义；`usize` 作为平台相关的无符号宽度，不适合作为数值元素    |
| 替代方案 | 支持全部整数类型（u8/u16/u32/i8/i16）— 放弃，增加矩阵复杂度                                           |

### 决策 3：bool 排除 Numeric

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | `bool` 仅实现 `Element`，不实现 `Numeric`                                  |
| 理由     | 布尔四则运算无数学意义；防止 `sum([true, false])` 等无意义操作；编译时阻止 |
| 替代方案 | bool 实现 Numeric（true=1, false=0）— 放弃，语义不清晰                     |

### 决策 4：usize 不属于元素 trait 集合

| 属性     | 值                                                                                                                          |
| -------- | --------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `usize` 不实现 `Element`/`Numeric`/`RealScalar`/`ComplexScalar`，仅用于索引和形状                                           |
| 理由     | `usize` 在 Xenon 中承担索引和形状元数据语义，而不是数值计算语义；其平台相关位宽会引入跨平台差异，不适合作为科学计算元素类型 |
| 替代方案 | 让 `usize` 作为元素类型存在但排除在 `Numeric` 之外 — 放弃，仍会混淆元素集合与索引语义                                       |

### 决策 5：不支持自动类型提升

| 属性     | 值                                                                     |
| -------- | ---------------------------------------------------------------------- |
| 决策     | 类型转换须显式，不支持隐式提升                                         |
| 理由     | 显式优于隐式，避免精度损失；性能可预测；与 Rust 哲学一致               |
| 替代方案 | 类似 C++ 的类型提升规则 — 放弃，增加复杂度，可能导致难以调试的精度问题 |

### 决策 6：RealScalar 和 ComplexScalar 平行继承 Numeric

| 属性     | 值                                                                                  |
| -------- | ----------------------------------------------------------------------------------- |
| 决策     | 两者都继承 Numeric，无交叉继承                                                      |
| 理由     | 提供正交的数学函数集；复数无自然全序（不应实现 PartialOrd）；未来可扩展其他标量类型 |
| 替代方案 | ComplexScalar 继承 RealScalar — 放弃，语义不正确                                    |

---

## 12. 性能考量

| 方面         | 设计决策                                           |
| ------------ | -------------------------------------------------- |
| 零运行时开销 | 所有 trait 约束为编译期静态分派，无虚调用          |
| 内联         | RealScalar 数学方法标注 `#[inline]`                |
| 单态化       | `Tensor<A, D>` 中 A 的 trait 约束在编译期单态化    |
| Sealed       | 封闭集合允许编译器做更激进的优化（已知完整类型集） |

---

## 13. 平台与工程约束

| 约束       | 说明                                          |
| ---------- | --------------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`        |
| MSRV       | Rust 1.85+                                    |
| 单 crate   | 保持单 crate 边界                             |
| SemVer     | 公开 trait、类型约束与转换语义变更遵循 SemVer |
| 最小依赖   | 无新增第三方依赖                              |

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
| 1.2.1 | 2026-04-10 |
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-14 |
| 1.2.4 | 2026-04-14 |
| 1.2.5 | 2026-04-15 |
| 1.2.6 | 2026-04-15 |
| 1.2.7 | 2026-04-15 |
| 1.2.8 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
