# 元素类型体系模块设计

> 模块: `src/element.rs`
> 版本: v1 | 日期: 2026-03-28
> 状态: 设计阶段

---

## 1. 模块定位

元素类型体系（Element Type System）是 Xenon 的类型基础设施层，定义多维数组中可存储的元素类型及其能力边界。该模块通过四层 trait 继承体系，在编译时区分不同元素类型支持的操作集合，为上层运算模块（逐元素运算、归约、矩阵运算等）提供泛型约束。

核心设计目标：

- **编译时区分能力**：bool 不支持算术，整数不支持 `sqrt`，在 trait 约束层面而非运行时区分
- **零开销抽象**：所有 trait 方法内联，泛型单态化后等价于直接调用原始类型方法
- **no_std 兼容**：使用 `core::ops`、`core::fmt` 等，不依赖 `std`
- **无外部 trait 依赖**：不引入 `num-traits`，所有 trait 本模块自建

---

## 2. 文件位置

```
src/
  element.rs          # 本模块：四层 trait 定义 + 基础类型 impl
```

在 `src/lib.rs` 中的声明：

```rust
pub mod element;

// re-export
pub use crate::element::{Element, Numeric, RealScalar, ComplexScalar};
```

---

## 3. 依赖关系

### 3.1 本模块的依赖（上游）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `core::ops::{Add, Sub, Mul, Div, Neg}` | `core` | Numeric trait 算术运算约束 |
| `core::fmt::{Debug, Display}` | `core` | Element trait 格式化约束 |
| `core::cmp::PartialEq, PartialOrd` | `core` | Element / RealScalar 比较约束 |
| `core::marker::{Copy, Send, Sync}` | `core` | Element trait 基础约束 |
| `core::clone::Clone` | `core` | Element trait 克隆约束 |

### 3.2 依赖本模块的下游模块

| 模块 | 使用的 trait | 用途 |
|------|-------------|------|
| `complex.rs` | `Numeric` | `Complex<T>` 的 `T` 约束为 `RealScalar`，`Complex<T>` 自身实现 `Numeric` |
| `storage/` | `Element` | 存储的元素类型须满足 `Element` 约束 |
| `tensor.rs` | `Element` | `TensorBase<S, D>` 的泛型参数 `A: Element` |
| `construction.rs` | `Element::zero()`, `Element::one()` | `zeros()`, `ones()` 构造函数 |
| `ops/element_wise.rs` | `Numeric` | 逐元素运算需要算术 trait |
| `ops/reduction.rs` | `Numeric`, `RealScalar` | 归约运算（sum 需要 `zero()` + `Add`，min/max 需要 `PartialOrd`） |
| `ops/matrix.rs` | `Numeric`, `RealScalar` | 矩阵运算（点积、矩阵-向量乘法） |
| `conversion.rs` | `Element`, `RealScalar` | 类型转换 cast |
| `backend/` | `RealScalar`, `ComplexScalar` | SIMD/并行路径分派 |

### 3.3 依赖关系图

```
core::ops, core::fmt, core::cmp, core::marker
        │
        ▼
  ┌──────────────┐
  │  element.rs  │  ← 本模块（Phase 2, W1）
  └──────┬───────┘
         │
    ┌────┴─────┐
    ▼          ▼
 complex   storage
 (W2)      (W3)
```

本模块属于 Phase 2 第一波次（W1），与 `dimension`、`error`、`layout` 无依赖关系，可并行实现。

---

## 4. 公共 API 设计

### 4.1 Element trait（基础层）

所有可存入多维数组的类型必须实现此 trait。包括整数、浮点、复数和 bool。

```rust
/// Base element trait for all types stored in tensors.
///
/// Provides the minimal interface required for tensor storage:
/// value semantics (Copy), equality comparison, formatting,
/// and thread safety.
///
/// # Super-traits
///
/// - `Copy`: bitwise copy is safe (no ownership semantics)
/// - `Clone`: explicit cloning (auto-derived from Copy)
/// - `PartialEq`: equality comparison for assertions and tests
/// - `Debug + Display`: formatting for debugging and user output
/// - `Send + Sync`: safe to share across threads (parallel iteration)
pub trait Element: Copy + Clone + PartialEq + Debug + Display + Send + Sync {
    /// Returns the additive identity (zero).
    ///
    /// For numeric types this is `0` / `0.0`.
    /// For `bool` this is `false`.
    fn zero() -> Self;

    /// Returns the multiplicative identity (one).
    ///
    /// For numeric types this is `1` / `1.0`.
    /// For `bool` this is `true`.
    fn one() -> Self;
}
```

**设计决策**：

- `zero()` 和 `one()` 为**必须实现的方法**（required method），不提供默认实现。原因：`num_traits::Zero`/`One` 为外部依赖，本模块不引入；Rust 标准库无统一的 zero/one trait；使用 required method 让编译器强制每个 impl 提供正确值
- `Send + Sync` 作为 super-trait：张量并行迭代要求元素可安全跨线程传递
- 不包含 `Default`：`Default` 语义不一定是零值（如 `f64::default()` 是 `0.0` 但缺乏明确语义），使用显式 `zero()` 更清晰

### 4.2 Numeric trait（数值层）

继承 Element，额外约束四则运算。仅数值类型（整数、浮点、复数）实现，bool **不实现**。

```rust
/// Numeric element trait for types supporting arithmetic operations.
///
/// Inherits `Element` and adds the five basic arithmetic operators,
/// all with `Output = Self` to ensure closed operations.
///
/// # Exclusions
///
/// `bool` does NOT implement `Numeric` — it only implements `Element`.
/// This prevents boolean tensors from participating in arithmetic operations
/// at compile time.
pub trait Numeric: Element + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {}
```

**设计决策**：

- 所有算术 operator trait 作为 super-trait 而非关联方法：利用 `core::ops` 的运算符重载语法 `a + b`，用户代码更自然
- `Output = Self` 约束：保证运算不改变类型，避免隐式类型提升
- 空 trait body（marker trait pattern）：所有能力通过 super-trait 表达，无需额外方法
- bool 排除机制：bool 实现了 `Add<Output=bool>` 但没有实现 `Mul`/`Div`/`Neg`，因此无法满足 `Numeric` 的全部 super-trait 约束，自然被排除。但为安全起见，我们**不为 bool 实现 `Numeric`**，即使未来 Rust 为 bool 添加了缺失的算术 trait

### 4.3 RealScalar trait（实数层）

继承 Numeric，为 `f32` 和 `f64` 提供数学函数、常量和特殊值检测。

```rust
/// Real-valued scalar trait for floating-point types (f32, f64).
///
/// Extends `Numeric` with mathematical functions, constants,
/// special value detection, and NaN-propagating min/max.
///
/// # Implementors
///
/// Only `f32` and `f64` implement this trait.
pub trait RealScalar: Numeric + PartialOrd {
    // ── Mathematical functions ──────────────────────────────

    /// Returns the absolute value.
    fn abs(self) -> Self;

    /// Returns the square root.
    ///
    /// Returns NaN for negative inputs.
    fn sqrt(self) -> Self;

    /// Returns the cube root.
    fn cbrt(self) -> Self;

    /// Returns the natural logarithm.
    ///
    /// Returns NaN for negative inputs, `-inf` for zero.
    fn ln(self) -> Self;

    /// Returns the base-2 logarithm.
    fn log2(self) -> Self;

    /// Returns the base-10 logarithm.
    fn log10(self) -> Self;

    /// Returns `e^(self)`, the exponential function.
    fn exp(self) -> Self;

    /// Returns `2^(self)`.
    fn exp2(self) -> Self;

    /// Returns the sine of the angle (in radians).
    fn sin(self) -> Self;

    /// Returns the cosine of the angle (in radians).
    fn cos(self) -> Self;

    /// Returns the tangent of the angle (in radians).
    fn tan(self) -> Self;

    /// Returns the arcsine.
    ///
    /// Returns NaN for inputs outside `[-1, 1]`.
    fn asin(self) -> Self;

    /// Returns the arccosine.
    ///
    /// Returns NaN for inputs outside `[-1, 1]`.
    fn acos(self) -> Self;

    /// Returns the arctangent.
    fn atan(self) -> Self;

    /// Returns the four-quadrant arctangent of `self / other`.
    fn atan2(self, other: Self) -> Self;

    /// Returns the hyperbolic sine.
    fn sinh(self) -> Self;

    /// Returns the hyperbolic cosine.
    fn cosh(self) -> Self;

    /// Returns the hyperbolic tangent.
    fn tanh(self) -> Self;

    /// Returns the largest integer less than or equal to self.
    fn floor(self) -> Self;

    /// Returns the smallest integer greater than or equal to self.
    fn ceil(self) -> Self;

    /// Returns the nearest integer.
    fn round(self) -> Self;

    /// Raises self to an integer power.
    fn powi(self, n: i32) -> Self;

    /// Raises self to a floating-point power.
    fn powf(self, n: Self) -> Self;

    // ── Constants ───────────────────────────────────────────

    /// Machine epsilon (difference between 1.0 and the next representable value).
    fn epsilon() -> Self;

    /// Smallest positive normal (non-subnormal) value.
    fn min_positive_value() -> Self;

    /// Largest finite representable value.
    fn max_value() -> Self;

    /// Positive infinity.
    fn infinity() -> Self;

    /// Negative infinity.
    fn neg_infinity() -> Self;

    /// Not-a-Number (NaN) constant.
    fn nan() -> Self;

    /// Archimedes' constant (π).
    fn PI() -> Self;

    /// Euler's number (e).
    fn E() -> Self;

    /// Ratio of a circle's circumference to its diameter (alias for PI).
    fn FRAC_PI_2() -> Self;

    /// π/4.
    fn FRAC_PI_4() -> Self;

    /// 1/π.
    fn FRAC_1_PI() -> Self;

    /// 2/π.
    fn FRAC_2_PI() -> Self;

    /// sqrt(2).
    fn SQRT_2() -> Self;

    /// 1/sqrt(2).
    fn FRAC_1_SQRT_2() -> Self;

    /// ln(2).
    fn LN_2() -> Self;

    /// ln(10).
    fn LN_10() -> Self;

    /// log2(e).
    fn LOG2_E() -> Self;

    /// log10(e).
    fn LOG10_E() -> Self;

    // ── Special value detection ─────────────────────────────

    /// Returns `true` if this value is NaN.
    #[must_use]
    fn is_nan(self) -> bool;

    /// Returns `true` if this value is positive or negative infinity.
    #[must_use]
    fn is_infinite(self) -> bool;

    /// Returns `true` if this value is neither infinite nor NaN.
    #[must_use]
    fn is_finite(self) -> bool;

    // ── NaN-propagating min/max ─────────────────────────────

    /// Returns the minimum of two values with NaN propagation.
    ///
    /// If either argument is NaN, returns NaN.
    /// This differs from `f32::min` / `f64::min` which return the non-NaN value.
    fn nan_min(self, other: Self) -> Self;

    /// Returns the maximum of two values with NaN propagation.
    ///
    /// If either argument is NaN, returns NaN.
    /// This differs from `f32::max` / `f64::max` which return the non-NaN value.
    fn nan_max(self, other: Self) -> Self;
}
```

**设计决策**：

- `PartialOrd` 作为 super-trait：浮点数的 NaN 语义使 `Ord` 不可用（`NaN != NaN`），使用 `PartialOrd` 正确反映浮点偏序性
- 常量使用方法（`PI()`）而非关联常量（`const PI`）：`f32::PI` / `f64::PI` 已在标准库中定义为关联常量，方法形式可提供统一 trait 接口，且在泛型上下文中更灵活
- `nan_min` / `nan_max` 命名加 `nan_` 前缀：与标准库 `f32::min`（非传播语义）区分，语义明确
- 所有数学函数为 required method：`f32` 和 `f64` 各有对应的标准库方法，impl 直接委托

### 4.4 ComplexScalar trait（复数层）

继承 Numeric，为 `Complex<f32>` 和 `Complex<f64>` 提供复数专属运算。

```rust
/// Complex-valued scalar trait for complex number types.
///
/// Extends `Numeric` with complex-specific operations:
/// real/imaginary part access, conjugation, norm, argument,
/// and complex transcendental functions.
///
/// # Implementors
///
/// `Complex<f32>` and `Complex<f64>` from the `complex` module.
///
/// # Note
///
/// This trait references `Complex<T>` defined in `src/complex.rs`.
/// The impl blocks are in `src/complex.rs`, not in `src/element.rs`.
pub trait ComplexScalar: Numeric {
    /// The real floating-point type underlying this complex number.
    type Real: RealScalar;

    /// Returns the real part.
    fn re(self) -> Self::Real;

    /// Returns the imaginary part.
    fn im(self) -> Self::Real;

    /// Returns the complex conjugate.
    fn conj(self) -> Self;

    /// Returns the modulus (magnitude), computed using `hypot` to avoid overflow.
    fn norm(self) -> Self::Real;

    /// Returns the argument (phase angle) in the range `(-π, π]`.
    fn arg(self) -> Self::Real;

    /// Returns the complex exponential `e^(self)`.
    fn complex_exp(self) -> Self;

    /// Returns the complex natural logarithm (principal value).
    fn complex_ln(self) -> Self;

    /// Returns the complex square root (principal value).
    fn complex_sqrt(self) -> Self;

    /// Constructs a complex number from polar coordinates `(r, theta)`.
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

    /// Returns the imaginary unit `i` (0 + 1i).
    fn i() -> Self;
}
```

**设计决策**：

- `type Real: RealScalar` 关联类型：复数运算经常需要提取实部、构造实数常量，关联类型明确复数与实数的对应关系
- 复数专属函数加 `complex_` 前缀（`complex_exp` vs `exp`）：避免与 `Numeric` 级别的运算混淆。注意：`ComplexScalar` 继承自 `Numeric`，`Numeric` 层的 `Add`/`Mul` 等已经可以处理复数加减乘除
- `norm()` 返回 `Self::Real` 而非 `Self`：模长是实数，不是复数
- `from_polar` 为关联函数（非方法）：从极坐标构造新的复数，不基于已有实例
- `ComplexScalar` 的 impl 放在 `src/complex.rs` 中：减少模块间耦合，`element.rs` 只定义 trait，`complex.rs` 负责实现

### 4.5 基础类型 impl

#### Element impl：整数类型

```rust
impl Element for i8 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for i16 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for i32 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for i64 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for u8 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for u16 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for u32 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}

impl Element for u64 {
    #[inline]
    fn zero() -> Self { 0 }
    #[inline]
    fn one() -> Self { 1 }
}
```

#### Element impl：浮点类型

```rust
impl Element for f32 {
    #[inline]
    fn zero() -> Self { 0.0 }
    #[inline]
    fn one() -> Self { 1.0 }
}

impl Element for f64 {
    #[inline]
    fn zero() -> Self { 0.0 }
    #[inline]
    fn one() -> Self { 1.0 }
}
```

#### Element impl：bool

```rust
impl Element for bool {
    #[inline]
    fn zero() -> Self { false }
    #[inline]
    fn one() -> Self { true }
}
```

#### Numeric impl：整数类型

```rust
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
```

注意：这些空 impl 之所以成立，是因为 Rust 标准库已为所有整数类型实现了 `Add<Output=Self>`, `Sub<Output=Self>`, `Mul<Output=Self>`, `Div<Output=Self>`, `Neg<Output=Self>`，以及 `Copy`, `Clone`, `PartialEq`, `Debug`, `Display`, `Send`, `Sync`。

#### Numeric impl：浮点类型

```rust
impl Numeric for f32 {}
impl Numeric for f64 {}
```

#### RealScalar impl：f32

```rust
impl RealScalar for f32 {
    #[inline] fn abs(self) -> Self { self.abs() }
    #[inline] fn sqrt(self) -> Self { self.sqrt() }
    #[inline] fn cbrt(self) -> Self { self.cbrt() }
    #[inline] fn ln(self) -> Self { self.ln() }
    #[inline] fn log2(self) -> Self { self.log2() }
    #[inline] fn log10(self) -> Self { self.log10() }
    #[inline] fn exp(self) -> Self { self.exp() }
    #[inline] fn exp2(self) -> Self { self.exp2() }
    #[inline] fn sin(self) -> Self { self.sin() }
    #[inline] fn cos(self) -> Self { self.cos() }
    #[inline] fn tan(self) -> Self { self.tan() }
    #[inline] fn asin(self) -> Self { self.asin() }
    #[inline] fn acos(self) -> Self { self.acos() }
    #[inline] fn atan(self) -> Self { self.atan() }
    #[inline] fn atan2(self, other: Self) -> Self { self.atan2(other) }
    #[inline] fn sinh(self) -> Self { self.sinh() }
    #[inline] fn cosh(self) -> Self { self.cosh() }
    #[inline] fn tanh(self) -> Self { self.tanh() }
    #[inline] fn floor(self) -> Self { self.floor() }
    #[inline] fn ceil(self) -> Self { self.ceil() }
    #[inline] fn round(self) -> Self { self.round() }
    #[inline] fn powi(self, n: i32) -> Self { self.powi(n) }
    #[inline] fn powf(self, n: Self) -> Self { self.powf(n) }

    // Constants
    #[inline] fn epsilon() -> Self { f32::EPSILON }
    #[inline] fn min_positive_value() -> Self { f32::MIN_POSITIVE }
    #[inline] fn max_value() -> Self { f32::MAX }
    #[inline] fn infinity() -> Self { f32::INFINITY }
    #[inline] fn neg_infinity() -> Self { f32::NEG_INFINITY }
    #[inline] fn nan() -> Self { f32::NAN }
    #[inline] fn PI() -> Self { core::f32::consts::PI }
    #[inline] fn E() -> Self { core::f32::consts::E }
    #[inline] fn FRAC_PI_2() -> Self { core::f32::consts::FRAC_PI_2 }
    #[inline] fn FRAC_PI_4() -> Self { core::f32::consts::FRAC_PI_4 }
    #[inline] fn FRAC_1_PI() -> Self { core::f32::consts::FRAC_1_PI }
    #[inline] fn FRAC_2_PI() -> Self { core::f32::consts::FRAC_2_PI }
    #[inline] fn SQRT_2() -> Self { core::f32::consts::SQRT_2 }
    #[inline] fn FRAC_1_SQRT_2() -> Self { core::f32::consts::FRAC_1_SQRT_2 }
    #[inline] fn LN_2() -> Self { core::f32::consts::LN_2 }
    #[inline] fn LN_10() -> Self { core::f32::consts::LN_10 }
    #[inline] fn LOG2_E() -> Self { core::f32::consts::LOG2_E }
    #[inline] fn LOG10_E() -> Self { core::f32::consts::LOG10_E }

    // Special value detection
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
    #[inline] fn is_infinite(self) -> bool { self.is_infinite() }
    #[inline] fn is_finite(self) -> bool { self.is_finite() }

    // NaN-propagating min/max
    #[inline]
    fn nan_min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            self.min(other)
        }
    }

    #[inline]
    fn nan_max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            self.max(other)
        }
    }
}
```

#### RealScalar impl：f64

```rust
impl RealScalar for f64 {
    #[inline] fn abs(self) -> Self { self.abs() }
    #[inline] fn sqrt(self) -> Self { self.sqrt() }
    #[inline] fn cbrt(self) -> Self { self.cbrt() }
    #[inline] fn ln(self) -> Self { self.ln() }
    #[inline] fn log2(self) -> Self { self.log2() }
    #[inline] fn log10(self) -> Self { self.log10() }
    #[inline] fn exp(self) -> Self { self.exp() }
    #[inline] fn exp2(self) -> Self { self.exp2() }
    #[inline] fn sin(self) -> Self { self.sin() }
    #[inline] fn cos(self) -> Self { self.cos() }
    #[inline] fn tan(self) -> Self { self.tan() }
    #[inline] fn asin(self) -> Self { self.asin() }
    #[inline] fn acos(self) -> Self { self.acos() }
    #[inline] fn atan(self) -> Self { self.atan() }
    #[inline] fn atan2(self, other: Self) -> Self { self.atan2(other) }
    #[inline] fn sinh(self) -> Self { self.sinh() }
    #[inline] fn cosh(self) -> Self { self.cosh() }
    #[inline] fn tanh(self) -> Self { self.tanh() }
    #[inline] fn floor(self) -> Self { self.floor() }
    #[inline] fn ceil(self) -> Self { self.ceil() }
    #[inline] fn round(self) -> Self { self.round() }
    #[inline] fn powi(self, n: i32) -> Self { self.powi(n) }
    #[inline] fn powf(self, n: Self) -> Self { self.powf(n) }

    // Constants
    #[inline] fn epsilon() -> Self { f64::EPSILON }
    #[inline] fn min_positive_value() -> Self { f64::MIN_POSITIVE }
    #[inline] fn max_value() -> Self { f64::MAX }
    #[inline] fn infinity() -> Self { f64::INFINITY }
    #[inline] fn neg_infinity() -> Self { f64::NEG_INFINITY }
    #[inline] fn nan() -> Self { f64::NAN }
    #[inline] fn PI() -> Self { core::f64::consts::PI }
    #[inline] fn E() -> Self { core::f64::consts::E }
    #[inline] fn FRAC_PI_2() -> Self { core::f64::consts::FRAC_PI_2 }
    #[inline] fn FRAC_PI_4() -> Self { core::f64::consts::FRAC_PI_4 }
    #[inline] fn FRAC_1_PI() -> Self { core::f64::consts::FRAC_1_PI }
    #[inline] fn FRAC_2_PI() -> Self { core::f64::consts::FRAC_2_PI }
    #[inline] fn SQRT_2() -> Self { core::f64::consts::SQRT_2 }
    #[inline] fn FRAC_1_SQRT_2() -> Self { core::f64::consts::FRAC_1_SQRT_2 }
    #[inline] fn LN_2() -> Self { core::f64::consts::LN_2 }
    #[inline] fn LN_10() -> Self { core::f64::consts::LN_10 }
    #[inline] fn LOG2_E() -> Self { core::f64::consts::LOG2_E }
    #[inline] fn LOG10_E() -> Self { core::f64::consts::LOG10_E }

    // Special value detection
    #[inline] fn is_nan(self) -> bool { self.is_nan() }
    #[inline] fn is_infinite(self) -> bool { self.is_infinite() }
    #[inline] fn is_finite(self) -> bool { self.is_finite() }

    // NaN-propagating min/max
    #[inline]
    fn nan_min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            self.min(other)
        }
    }

    #[inline]
    fn nan_max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::nan()
        } else {
            self.max(other)
        }
    }
}
```

#### ComplexScalar impl（位于 complex.rs）

```rust
// In src/complex.rs (NOT in src/element.rs)

impl ComplexScalar for Complex<f32> {
    type Real = f32;

    #[inline] fn re(self) -> f32 { self.re }
    #[inline] fn im(self) -> f32 { self.im }
    #[inline] fn conj(self) -> Self { Self { re: self.re, im: -self.im } }
    #[inline] fn norm(self) -> f32 { self.re.hypot(self.im) }
    #[inline] fn arg(self) -> f32 { self.im.atan2(self.re) }

    #[inline]
    fn complex_exp(self) -> Self {
        let r = self.re.exp();
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }

    #[inline]
    fn complex_ln(self) -> Self {
        Self { re: self.norm().ln(), im: self.arg() }
    }

    #[inline]
    fn complex_sqrt(self) -> Self {
        let r = self.norm();
        let re = ((r + self.re) / 2.0).sqrt();
        let im = ((r - self.re) / 2.0).sqrt();
        // Adjust sign of im to match arg(self) in (-π, π]
        let im_signed = if self.im < 0.0 { -im } else { im };
        Self { re, im: im_signed }
    }

    #[inline]
    fn from_polar(r: f32, theta: f32) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    #[inline]
    fn i() -> Self { Self { re: 0.0, im: 1.0 } }
}

impl ComplexScalar for Complex<f64> {
    type Real = f64;

    #[inline] fn re(self) -> f64 { self.re }
    #[inline] fn im(self) -> f64 { self.im }
    #[inline] fn conj(self) -> Self { Self { re: self.re, im: -self.im } }
    #[inline] fn norm(self) -> f64 { self.re.hypot(self.im) }
    #[inline] fn arg(self) -> f64 { self.im.atan2(self.re) }

    #[inline]
    fn complex_exp(self) -> Self {
        let r = self.re.exp();
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }

    #[inline]
    fn complex_ln(self) -> Self {
        Self { re: self.norm().ln(), im: self.arg() }
    }

    #[inline]
    fn complex_sqrt(self) -> Self {
        let r = self.norm();
        let re = ((r + self.re) / 2.0).sqrt();
        let im = ((r - self.re) / 2.0).sqrt();
        let im_signed = if self.im < 0.0 { -im } else { im };
        Self { re, im: im_signed }
    }

    #[inline]
    fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    #[inline]
    fn i() -> Self { Self { re: 0.0, im: 1.0 } }
}
```

---

## 5. 内部实现设计

### 5.1 Trait 层次结构

```
                    ┌──────────┐
                    │  Element │  Copy + Clone + PartialEq + Debug + Display + Send + Sync + zero() + one()
                    └────┬─────┘
                         │
               ┌─────────┴──────────┐
               │                    │
        ┌──────┴──────┐      (bool — 仅 Element，到此为止)
        │   Numeric   │      Element + Add + Sub + Mul + Div + Neg
        └──────┬──────┘
               │
       ┌───────┴────────┐
       │                │
┌──────┴───────┐  ┌─────┴──────────┐
│  RealScalar  │  │ ComplexScalar  │
│ Numeric +    │  │ Numeric +      │
│ PartialOrd + │  │ type Real +    │
│ math funcs   │  │ complex ops    │
│ constants    │  │                │
│ detection    │  │                │
└──────────────┘  └────────────────┘
   f32, f64        Complex<f32>, Complex<f64>
```

### 5.2 bool 排除机制

bool 被排除在 Numeric 之外的机制是 **super-trait 约束**：

1. `Numeric` 要求 `Neg<Output = Self>`
2. Rust 标准库未为 `bool` 实现 `Neg`
3. 因此 `bool` 无法满足 `Numeric` 的 super-trait 链
4. 编译器会在尝试将 `bool` 用于需要 `Numeric` 的泛型时报错

此外，即使 Rust 未来为 bool 添加了 `Neg` 等实现，我们也**显式选择不为 bool 写 `impl Numeric for bool {}`**，保持语义正确性。

### 5.3 Required vs Default 方法

| Trait | 方法 | 策略 | 原因 |
|-------|------|------|------|
| `Element` | `zero()`, `one()` | Required | 每个类型的值不同，无通用默认 |
| `Numeric` | （空 body） | N/A | Marker trait，能力由 super-trait 表达 |
| `RealScalar` | 全部方法 | Required | f32/f64 的标准库方法直接委托，无通用默认 |
| `ComplexScalar` | 全部方法 | Required | 复数运算逻辑不可复用于 f32/f64 |

不使用默认方法的原因：本模块只有 `f32`/`f64` 两个 impl（RealScalar），每个方法体只有一行委托，默认方法不减少代码量且增加间接性。

### 5.4 no_std 考量

本模块完全兼容 `no_std`，具体措施：

| 标准库路径 | no_std 替代 | 说明 |
|-----------|-------------|------|
| `std::ops::Add` | `core::ops::Add` | 算术运算 trait |
| `std::fmt::Debug` | `core::fmt::Debug` | 格式化 trait |
| `std::marker::Copy` | `core::marker::Copy` | 标记 trait |
| `std::f32::consts::PI` | `core::f32::consts::PI` | 浮点常量 |
| `std::f64::EPSILON` | `core::f64::EPSILON` | 浮点属性 |

本模块不使用堆分配，不需要 `alloc` crate。

Import 结构：

```rust
use core::fmt::{Debug, Display};
use core::ops::{Add, Sub, Mul, Div, Neg};
use core::cmp::{PartialEq, PartialOrd};
use core::marker::{Copy, Send, Sync};
use core::clone::Clone;
```

### 5.5 模块文件结构

```rust
// src/element.rs — 完整内容组织

use core::fmt::{Debug, Display};
use core::ops::{Add, Sub, Mul, Div, Neg};
use core::cmp::{PartialEq, PartialOrd};
use core::marker::{Copy, Send, Sync};
use core::clone::Clone;

// ── Trait definitions ──────────────────────────────────────

pub trait Element: Copy + Clone + PartialEq + Debug + Display + Send + Sync { ... }

pub trait Numeric: Element + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {}

pub trait RealScalar: Numeric + PartialOrd { ... }

pub trait ComplexScalar: Numeric { ... }

// ── Element implementations ────────────────────────────────

impl Element for i8 { ... }
impl Element for i16 { ... }
impl Element for i32 { ... }
impl Element for i64 { ... }
impl Element for u8 { ... }
impl Element for u16 { ... }
impl Element for u32 { ... }
impl Element for u64 { ... }
impl Element for f32 { ... }
impl Element for f64 { ... }
impl Element for bool { ... }

// ── Numeric implementations ────────────────────────────────

impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for f32 {}
impl Numeric for f64 {}

// ── RealScalar implementations ─────────────────────────────

impl RealScalar for f32 { ... }
impl RealScalar for f64 { ... }

// ComplexScalar implementations are in src/complex.rs

// ── Unit tests ─────────────────────────────────────────────

#[cfg(test)]
mod tests { ... }
```

### 5.6 NaN 处理策略

NaN 处理策略在本模块中的体现：

| 场景 | 本模块提供的支持 | 实际逻辑位置 |
|------|-----------------|-------------|
| NaN 传播 min/max | `RealScalar::nan_min()`, `nan_max()` | `element.rs` |
| NaN 检测 | `RealScalar::is_nan()` | `element.rs`（委托 `f32::is_nan`） |
| NaN 常量 | `RealScalar::nan()` | `element.rs`（委托 `f32::NAN`） |
| 归约中的 NaN 传播 | 使用 `Numeric` 的 `Add` 约束，IEEE 754 自动传播 | `ops/reduction.rs` |
| 排序中的 NaN | `PartialOrd` 返回 `None` | 消费者使用 `partial_cmp` |

关键点：`nan_min`/`nan_max` 语义**不同于**标准库的 `f32::min`/`f32::max`。标准库版本在遇到 NaN 时返回非 NaN 值（IEEE 754-2008 totalOrder），而我们采用 **NaN 传播语义**（任一参数为 NaN 则结果为 NaN），这与 NumPy 的 `np.nanmin`/`np.nanmax` 不同——我们的语义更接近 `np.minimum`/`np.maximum` 在 NaN 处理上的行为。

---

## 6. 实现任务拆分

每个任务约 10 分钟，单一职责，可独立验证。

### T1: Element trait 定义与整数 impl
- [ ] **文件**: `src/element.rs:1-60`
- **内容**: 定义 `Element` trait，impl `Element` for `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- **测试**: `test_element_i32_zero_one`, `test_element_u64_zero_one`
- **前置**: 无
- **预计**: 10 min

### T2: Element impl for f32/f64/bool
- [ ] **文件**: `src/element.rs:61-90`
- **内容**: impl `Element` for `f32`, `f64`, `bool`
- **测试**: `test_element_f64_zero_one`, `test_element_bool_zero_one`
- **前置**: T1
- **预计**: 5 min

### T3: Numeric trait 定义与所有 impl
- [ ] **文件**: `src/element.rs:91-120`
- **内容**: 定义 `Numeric` trait（空 body + super-trait 链），impl `Numeric` for 所有整数 + f32 + f64
- **测试**: `test_numeric_add_i32`, `test_numeric_mul_f64`, `test_bool_not_numeric`（编译失败测试）
- **前置**: T1, T2
- **预计**: 10 min

### T4: RealScalar trait 定义
- [ ] **文件**: `src/element.rs:121-220`
- **内容**: 定义 `RealScalar` trait 的完整签名（数学函数 + 常量 + 检测方法 + nan_min/nan_max）
- **测试**: 编译检查（trait 定义本身无运行时测试）
- **前置**: T3
- **预计**: 10 min

### T5: RealScalar impl for f32
- [ ] **文件**: `src/element.rs:221-310`
- **内容**: impl `RealScalar for f32`，所有方法委托到 `f32` 标准库方法
- **测试**: `test_real_f32_math_functions`, `test_real_f32_constants`, `test_real_f32_nan_detection`, `test_real_f32_nan_min_max`
- **前置**: T4
- **预计**: 10 min

### T6: RealScalar impl for f64
- [ ] **文件**: `src/element.rs:311-400`
- **内容**: impl `RealScalar for f64`，结构同 f32 impl
- **测试**: `test_real_f64_math_functions`, `test_real_f64_constants`, `test_real_f64_nan_min_max`
- **前置**: T4
- **预计**: 10 min

### T7: ComplexScalar trait 定义
- [ ] **文件**: `src/element.rs:401-440`
- **内容**: 定义 `ComplexScalar` trait 的完整签名（关联类型 `Real`，所有复数方法）
- **测试**: 编译检查
- **前置**: T3
- **预计**: 5 min

### T8: lib.rs 集成与 re-export
- [ ] **文件**: `src/lib.rs`
- **内容**: 添加 `pub mod element;`，添加 `pub use crate::element::{Element, Numeric, RealScalar, ComplexScalar};`
- **测试**: `test_public_api_import`
- **前置**: T1-T7
- **预计**: 5 min

### T9: 单元测试完善
- [ ] **文件**: `src/element.rs` (`#[cfg(test)] mod tests`)
- **内容**: 补充边界测试（NaN 传播、Inf 算术、零值、负零）、属性测试（零元律、单位元律）
- **测试**: 全部测试文件
- **前置**: T1-T7
- **预计**: 15 min

### T10: 文档注释与内联标注
- [ ] **文件**: `src/element.rs` 全文件
- **内容**: 为所有 pub trait、pub fn 添加 doc comment（`///`），添加 `#[inline]` 标注
- **测试**: `cargo doc --no-deps` 无警告
- **前置**: T1-T7
- **预计**: 10 min

---

## 7. 测试计划

### 7.1 单元测试（`src/element.rs` 内 `#[cfg(test)] mod tests`）

#### Element trait 测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_element_i32_zero_one` | `i32::zero() == 0`, `i32::one() == 1` |
| `test_element_f64_zero_one` | `f64::zero() == 0.0`, `f64::one() == 1.0` |
| `test_element_bool_zero_one` | `bool::zero() == false`, `bool::one() == true` |
| `test_element_copy_semantics` | 赋值后修改不影响原值 |
| `test_element_all_integer_types` | 遍历 i8/u8/i16/u16/i32/u32/i64/u64 的 zero/one |

#### Numeric trait 测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_numeric_add_i32` | `1i32 + 2i32 == 3i32`（泛型函数验证） |
| `test_numeric_mul_f64` | `2.0f64 * 3.0f64 == 6.0f64`（泛型函数验证） |
| `test_numeric_neg_i64` | `-(5i64) == -5i64` |
| `test_numeric_div_u32` | `10u32 / 3u32 == 3u32` |
| `test_bool_not_numeric` | 编译失败测试：bool 不能用于 `Numeric` 约束的函数 |

编译失败测试使用 `#[cfg(doctest)]` 或 `compile_error` 宏模式：

```rust
/// Test that bool does not satisfy Numeric constraint.
/// This is a compile-time guarantee verified by the type system.
/// Uncommenting the following would fail to compile:
///
/// ```compile_fail
/// fn requires_numeric<T: xenon::Numeric>() {}
/// requires_numeric::<bool>();
/// ```
```

#### RealScalar trait 测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_real_f32_abs` | `(-3.0f32).abs() == 3.0` |
| `test_real_f64_sqrt` | `4.0f64.sqrt() == 2.0` |
| `test_real_f32_trig` | `f32::PI().sin()` 和 `f32::PI().cos()` 近似值 |
| `test_real_f64_exp_ln_inverse` | `x.exp().ln() ≈ x` |
| `test_real_f32_constants` | `f32::epsilon() > 0.0`, `f32::PI() > 3.14` |
| `test_real_f64_nan_detection` | `f64::nan().is_nan() == true`, `1.0_f64.is_nan() == false` |
| `test_real_f32_infinite_detection` | `f32::infinity().is_infinite()`, `f32::neg_infinity().is_infinite()` |
| `test_real_f32_finite_detection` | `1.0_f32.is_finite() == true`, `f32::nan().is_finite() == false` |
| `test_real_f64_nan_min_propagation` | `1.0_f64.nan_min(f64::nan()).is_nan()` |
| `test_real_f64_nan_max_propagation` | `1.0_f64.nan_max(f64::nan()).is_nan()` |
| `test_real_f64_nan_min_both_nan` | `f64::nan().nan_min(f64::nan()).is_nan()` |
| `test_real_f32_nan_min_normal` | `1.0_f32.nan_min(2.0) == 1.0` |
| `test_real_f32_powi` | `2.0_f32.powi(10) == 1024.0` |
| `test_real_f64_powf` | `(4.0_f64).powf(0.5) == 2.0` |
| `test_real_f32_floor_ceil_round` | `1.7.floor() == 1.0`, `1.2.ceil() == 2.0`, `1.5.round() == 2.0` |
| `test_real_f64_atan2` | `0.0_f64.atan2(1.0) == 0.0` |

#### 边界测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_real_f64_inf_arithmetic` | `f64::infinity() + 1.0 == f64::infinity()`, `1.0 / 0.0 == f64::infinity()` |
| `test_real_f32_zero_div_zero` | `0.0f32 / 0.0` 结果为 NaN |
| `test_real_f64_one_div_zero` | `1.0f64 / 0.0 == f64::infinity()` |
| `test_real_f32_neg_zero` | `-0.0f32` 的行为：`(-0.0).is_finite() == true` |

#### 属性测试（代数不变量）

| 测试函数 | 验证不变量 |
|---------|-----------|
| `test_identity_add_zero` | `x + T::zero() == x`（对所有 Numeric 类型） |
| `test_identity_mul_one` | `x * T::one() == x`（对所有 Numeric 类型） |
| `test_real_exp_ln_roundtrip` | `x.ln().exp() ≈ x`（对 `x > 0`） |
| `test_real_sqrt_square` | `x.sqrt() * x.sqrt() ≈ x`（对 `x >= 0`） |

### 7.2 集成测试

本模块的集成测试主要在 `tests/` 目录下由下游模块（construction、ops 等）间接覆盖，不单独设集成测试文件。

### 7.3 测试辅助宏

```rust
/// Assert that two floating-point values are approximately equal.
macro_rules! assert_close {
    ($left:expr, $right:expr, $tol:expr) => {
        let l = $left;
        let r = $right;
        let tol = $tol;
        assert!(
            (l - r).abs() <= tol,
            "assertion failed: `|left - right| <= tol`\n  left: `{}`\n right: `{}`\n   tol: `{}`",
            l, r, tol
        );
    };
}
```

---

## 附录 A：类型矩阵

| 类型 | Element | Numeric | RealScalar | ComplexScalar |
|------|---------|---------|------------|---------------|
| `bool` | ✅ | ❌ | ❌ | ❌ |
| `i8` | ✅ | ✅ | ❌ | ❌ |
| `i16` | ✅ | ✅ | ❌ | ❌ |
| `i32` | ✅ | ✅ | ❌ | ❌ |
| `i64` | ✅ | ✅ | ❌ | ❌ |
| `u8` | ✅ | ✅ | ❌ | ❌ |
| `u16` | ✅ | ✅ | ❌ | ❌ |
| `u32` | ✅ | ✅ | ❌ | ❌ |
| `u64` | ✅ | ✅ | ❌ | ❌ |
| `f32` | ✅ | ✅ | ✅ | ❌ |
| `f64` | ✅ | ✅ | ✅ | ❌ |
| `Complex<f32>` | ✅ | ✅ | ❌ | ✅ |
| `Complex<f64>` | ✅ | ✅ | ❌ | ✅ |

## 附录 B：NaN/Inf 行为汇总

| 操作 | 行为 | IEEE 754 一致性 |
|------|------|----------------|
| `0.0 / 0.0` | NaN | ✅ |
| `1.0 / 0.0` | +Inf | ✅ |
| `-1.0 / 0.0` | -Inf | ✅ |
| `NaN + x` | NaN | ✅（传播） |
| `NaN * x` | NaN | ✅（传播） |
| `RealScalar::nan_min(NaN, x)` | NaN | NaN 传播语义 |
| `RealScalar::nan_max(NaN, x)` | NaN | NaN 传播语义 |
| `NaN.partial_cmp(&x)` | `None` | ✅（PartialOrd） |
| 归约 sum 含 NaN | NaN | ✅（IEEE 754 传播） |

## 附录 C：与 Complex 模块的交互

```
src/element.rs                    src/complex.rs
┌─────────────────────┐          ┌─────────────────────┐
│ pub trait Element   │          │ pub struct Complex<T>│
│ pub trait Numeric   │◄─────────│ impl Element for C<T>│
│ pub trait RealScalar│          │ impl Numeric for C<T>│
│ pub trait Complex-  │◄─────────│ impl ComplexScalar   │
│         Scalar      │          │   for Complex<f32>   │
│                     │          │ impl ComplexScalar   │
│                     │          │   for Complex<f64>   │
└─────────────────────┘          └─────────────────────┘
         ▲                                │
         │        Complex<T> 需要          │
         │   T: RealScalar (约束)          │
         └────────────────────────────────┘
```

`complex.rs` 依赖 `element.rs` 中的 trait 定义；
`element.rs` 不依赖 `complex.rs`（ComplexScalar impl 在 `complex.rs` 中）。
依赖方向为单向：`complex` → `element`。
