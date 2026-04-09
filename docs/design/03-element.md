# 元素类型体系模块设计

> 文档编号: 03 | 模块: `src/element/` | 阶段: Phase 1
> 前置文档: `00-coding.md`, `01-architecture.md`
> 需求参考: 需求说明书 §4

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| Element trait | 基础约束（Copy+Clone+PartialEq+Debug+Display+Send+Sync+Sealed）+ zero()/one() | — |
| Numeric trait | Element + Add+Sub+Mul+Div+Neg（四则运算能力标记） | 运算实现本身（委托给 core::ops） |
| RealScalar trait | Numeric + PartialOrd + abs/sqrt/sin/cos/exp/ln/floor/ceil + NaN 检测 | 复数运算 |
| ComplexScalar trait | Numeric + norm/conj/from_polar/arg/exp/ln/sqrt（复数运算接口） | 复数类型定义（在 `src/complex/` 模块，参见 `04-complex.md` §4） |
| 基础类型实现 | 为 i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool/usize 实现上述 trait | 类型转换逻辑（在 `src/convert/` 模块） |
| Sealed trait | 封闭集合，禁止外部 crate 实现 | 开放扩展 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 能力最小化 | 每层 trait 仅声明必要约束，避免过度限制泛型 |
| 正交性 | 数值运算（Numeric）、实数函数（RealScalar）、复数运算（ComplexScalar）职责分离 |
| 零运行时开销 | 所有约束为编译期静态分派 |
| 封闭集合 | Sealed trait 阻止下游 crate 扩展类型集 |
| IEEE 754 兼容 | 浮点特殊值（NaN、Inf）处理遵循标准语义 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: element  ← 当前模块
L1: complex（element 依赖 complex 的类型定义，complex 不反向依赖 element）
L2: layout (依赖 dimension)
L3: storage (仅依赖 core/alloc)
L4: tensor (依赖 storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

> **说明**：`element` 模块位于 L1 层级，但内部依赖同级的 `complex` 模块（`complex` 不依赖 `element`）。这是 L1 内部的单向依赖，`element` 使用 `Complex<T>` 类型作为 trait 实现目标，`complex` 仅提供类型定义和基础运算，不涉及 `Element`/`Numeric` 等 trait。这种单向依赖是合理的。

---

## 2. 文件位置

```
src/element/
├── mod.rs             # Element trait 定义、Sealed trait、模块 re-export
├── numeric.rs         # Numeric trait 定义（四则运算约束）
├── real.rs            # RealScalar trait 定义（实数数学函数）
├── complex.rs         # ComplexScalar trait 定义（复数运算接口）
└── primitives.rs      # 为基础类型实现各层 trait
```

模块内聚设计：四层 trait 按文件分离，基础类型实现集中在 `primitives.rs`。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
src/element/
├── crate::complex    # Complex<T> 类型定义
├── core::ops         # Add/Sub/Mul/Div/Neg 运算符 trait
├── core::fmt         # Debug/Display 格式化
└── core::cmp         # PartialEq/PartialOrd 比较
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `crate::complex` | `Complex<f32>`, `Complex<f64>`（元素类型实现目标） |
| `core::ops` | `Add`, `Sub`, `Mul`, `Div`, `Neg`（Numeric supertrait） |
| `core::fmt` | `Debug`, `Display`（Element supertrait） |
| `core::cmp` | `PartialEq`, `PartialOrd`（Element/RealScalar supertrait） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `element/` 消费 `complex` 的类型定义（即 `element` 依赖 `complex`），`complex` 不反向依赖 `element`。
> 被下游消费：`math`（参见 `11-math.md` §4）、`reduction`（参见 `13-reduction.md` §4）、`tensor`（参见 `07-tensor.md` §4）等模块使用 Element/Numeric/RealScalar/ComplexScalar 作为泛型约束。

---

## 4. 公共 API 设计

### 4.1 Element trait

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

| Supertrait | 作用 |
|------------|------|
| `Copy` | 值语义，可按位复制，避免所有权转移开销 |
| `Clone` | 显式克隆能力 |
| `PartialEq` | 相等比较，用于断言和测试 |
| `Debug` | 调试格式输出 `{:?}` |
| `Display` | 用户友好格式输出 `{}` |
| `Send` | 可跨线程移动（并行迭代必需） |
| `Sync` | 可跨线程共享引用（并行只读访问必需） |
| `Sealed` | 防止外部类型实现 |

### 4.2 Numeric trait

```rust
/// Numeric element trait.
///
/// Adds arithmetic operations on top of Element.
/// Only numeric types (integers, floats, complex) implement this.
/// `bool` does NOT implement Numeric.
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
    /// Returns the conjugate of this value.
    ///
    /// For real numeric types (i32, i64, f32, f64), this returns `self` unchanged.
    /// For `ComplexScalar` types, this returns the complex conjugate (re - im*i).
    ///
    /// This method is needed by `12-matrix` for unified dot product implementation,
    /// allowing a single generic algorithm to handle both real and complex inner products.
    fn conjugate(self) -> Self;

    /// Performs addition with overflow checking for integers.
    /// - Integer types: uses checked_add, panics on overflow
    /// - Float types: standard IEEE 754 addition
    /// - Complex types: component-wise addition
    fn safe_add(self, rhs: Self) -> Self;
}

// Real type implementations return self (identity):
//
// impl Numeric for i32 {
//     fn conjugate(self) -> Self { self }
//     fn safe_add(self, rhs: Self) -> Self { self.checked_add(rhs).expect("integer overflow in reduction") }
// }
// impl Numeric for i64 {
//     fn conjugate(self) -> Self { self }
//     fn safe_add(self, rhs: Self) -> Self { self.checked_add(rhs).expect("integer overflow in reduction") }
// }
// impl Numeric for f32 {
//     fn conjugate(self) -> Self { self }
//     fn safe_add(self, rhs: Self) -> Self { self + rhs }
// }
// impl Numeric for f64 {
//     fn conjugate(self) -> Self { self }
//     fn safe_add(self, rhs: Self) -> Self { self + rhs }
// }
// Complex type implementations return the complex conjugate:
// impl Numeric for Complex<f32> {
//     fn conjugate(self) -> Self { Complex::new(self.re, -self.im) }
//     fn safe_add(self, rhs: Self) -> Self { Complex::new(self.re + rhs.re, self.im + rhs.im) }
// }
// impl Numeric for Complex<f64> {
//     fn conjugate(self) -> Self { Complex::new(self.re, -self.im) }
//     fn safe_add(self, rhs: Self) -> Self { Complex::new(self.re + rhs.re, self.im + rhs.im) }
// }
```

> **设计决策：** `Numeric` 定义 `conjugate()` 方法，为实数类型返回 `self`，为复数类型返回共轭。这使得统一的内积（dot product）实现可以泛化处理实数和复数情况（实数内积 `∑ aᵢ*bᵢ`，复数内积 `∑ aᵢ·conjugate(bᵢ)`）。其余四则运算由 `Add/Sub/Mul/Div/Neg` trait 提供。
>
> **`Numeric::conjugate()` 与 `ComplexScalar::conj()` 的关系和使用说明：**
>
> - `Numeric::conjugate()` 对实数类型（i32, i64, f32, f64）是恒等操作（返回自身），对复数类型（Complex<f32>, Complex<f64>）委托给复数共轭实现
> - 对于泛型代码中仅约束 `Numeric` 时，调用 `x.conjugate()` 会解析到 `Numeric::conjugate()`
> - `ComplexScalar::conj()` 返回实数类型结果（模运算），与 `Numeric::conjugate()` 返回同类型共轭的语义不同。方法名已区分以避免混淆
> - **设计决策注释：** 两个方法语义不同（`Numeric::conjugate` 返回 `Self` 的共轭，`ComplexScalar::conj` 返回实数模），`Numeric::conjugate` 的存在是为了让纯 `Numeric` 约束的泛型代码也能调用共轭，无需额外约束 `ComplexScalar`

### 4.3 RealScalar trait

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
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;

    // ========== Constants ==========
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

> **设计决策：** `min`/`max` 采用 NaN 传播语义（任一参数为 NaN → 返回 NaN），与 `f32::min`/`f64::min` 一致。

### 4.4 ComplexScalar trait

```rust
/// Complex scalar trait.
///
/// Provides complex-specific operations on top of Numeric.
/// Only Complex<f32> and Complex<f64> implement this.
pub trait ComplexScalar: Numeric + Sealed {
    // Sealed is already inherited via Element (which Numeric extends),
    // but listed here for defensive clarity — makes the sealed intent explicit
    // at each trait level.
    /// Real part type (must be RealScalar).
    type Real: RealScalar;

    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    /// Returns the complex conjugate (re - im*i).
    ///
    /// Note: `Numeric` also defines a `conjugate()` method with identical semantics.
    /// For types that implement both `Numeric` and `ComplexScalar` (e.g., `Complex<f64>`),
    /// use fully-qualified syntax to disambiguate:
    /// - `ComplexScalar::conj(x)` — via ComplexScalar trait
    /// - `Numeric::conjugate(x)` — via Numeric trait
    fn conj(self) -> Self;
    fn norm(self) -> Self::Real;
    fn arg(self) -> Self::Real;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;
    fn i() -> Self;
}
```

### 4.5 支持的类型与 trait 矩阵

| 类型 | Element | Numeric | RealScalar | ComplexScalar |
|------|:-------:|:-------:|:----------:|:-------------:|
| `i32` | ✓ | ✓ | ✗ | ✗ |
| `i64` | ✓ | ✓ | ✗ | ✗ |
| `f32` | ✓ | ✓ | ✓ | ✗ |
| `f64` | ✓ | ✓ | ✓ | ✗ |
| `Complex<f32>` | ✓ | ✓ | ✗ | ✓ |
| `Complex<f64>` | ✓ | ✓ | ✗ | ✓ |
| `bool` | ✓ | ✗ | ✗ | ✗ |
| `usize` | ✓ | ✗ | ✗ | ✗ |

> **Xenon 特定约束：** 仅支持上表列出的 8 种类型。不支持 u8/u16/u32/i8/i16 等其他整数类型。

### 4.6 Sealed trait 策略

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
impl Sealed for usize {}
```

外部 crate 尝试实现时编译错误：
```rust
// External crate attempt:
use xenon::element::Element;
struct MyType;
impl Element for MyType { /* error[E0277]: Sealed not satisfied */ }
```

### 4.7 Good / Bad 对比示例

```rust
// Good - Numeric constraint automatically excludes bool
fn sum<A: Numeric>(tensor: &TensorView<A>) -> A {
    tensor.iter().fold(A::zero(), |acc, &x| acc + x)
}
// sum(&bool_tensor);  // Compile error: bool does not satisfy Numeric ✓

// Bad - Element constraint cannot exclude bool
fn sum_bad<A: Element>(tensor: &TensorView<A>) -> A {
    // Cannot use + operator, Element has no Add bound
    todo!()
}
```

```rust
// Good - explicit type conversion, no automatic promotion
let a: Tensor<f64, Ix2> = Tensor::zeros((3, 4));
let b: Tensor<i32, Ix2> = Tensor::zeros((3, 4));
let c = &a + &b.cast::<f64>();  // explicit conversion

// Bad - expecting automatic type promotion (not supported in Xenon)
// let c = &a + &b;  // Compile error: no matching impl for f64 + i32
```

### 4.8 CastTo\<T\> trait（类型转换）

`CastTo<T>` 定义逐元素类型转换规则，由 `convert/cast.rs` 模块使用（参见 `21-type.md §4`）。

```rust
// src/element/mod.rs (or element/cast.rs)

/// Element-wise type conversion trait.
///
/// Defines explicit conversion from `Self` to `T`.
/// Overflow behavior is explicitly defined per implementation (see `21-type.md §4.3`).
///
/// This trait is NOT sealed — it is implemented for all supported type pairs.
/// External crates cannot add new element types (due to `Sealed`), but they do not
/// need to implement `CastTo` either, as all supported conversions are provided internally.
pub trait CastTo<T>: Element {
    /// Performs the type conversion.
    fn cast_to(self) -> T;
}

// Implemented for all supported source → target type pairs.
// See 21-type.md §5.1 for full implementation table.
// Examples:
//   impl CastTo<f64> for f32  -- lossless upcast
//   impl CastTo<f32> for f64  -- round-to-nearest-even
//   impl CastTo<i32> for f64  -- truncate + saturating (NaN→0)
//   impl CastTo<i32> for i64  -- saturating narrowing
//   impl CastTo<Complex<f64>> for f64  -- real part, im = 0.0
// Note: CastTo<f64> for Complex<f64> is intentionally NOT implemented.
```

### 4.9 CheckedAdd trait（整数溢出检测）

`CheckedAdd` 为整数类型提供 checked 加法，供 `sum` 归约操作在整数溢出时 panic（参见 `13-reduction.md §5.1`）。

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
pub trait CheckedAdd: Numeric {
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
// bool, usize: NOT implemented — not Numeric.
```

> **设计决策：** `CheckedAdd` 仅覆盖整数加法（`i32`/`i64`）。整数**乘法**溢出不在 `CheckedAdd` 范围内——逐元素乘法（elementwise multiply）对整数使用 **wrapping 语义**（`wrapping_mul`）。此设计决策的理由：
> - 加法溢出检测主要用于 `sum()` 归约操作（参见 `13-reduction.md §5.1`），归约结果应精确
> - 乘法溢出在逐元素运算中按 wrapping 处理，与 Rust release 模式默认行为一致，避免性能退化
> - 如需 checked 乘法，用户应在调用前手动检查参数范围

---

## 5. 内部实现设计

### 5.1 bool 排除策略

`bool` 仅实现 `Element`，不实现 `Numeric`：

```rust
// primitives.rs
impl Element for bool {
    fn zero() -> Self { false }
    fn one() -> Self { true }
}
// No `impl Numeric for bool` — bool arithmetic has no mathematical meaning
```

编译时阻止无效泛型实例化：`fn sum<A: Numeric>` 无法接受 `bool` 张量。

### 5.2 usize 包含策略

`usize` 仅实现 `Element`，用于索引操作场景。不实现 `Numeric`（`usize` 主要作为形状/索引类型使用，而非计算类型）。

### 5.3 类型提升规则

**Xenon 不支持自动类型提升。** 所有跨类型运算须显式转换：

```rust
// Implicit conversion not supported
// let a: f64 = 1.0;
// let b: i32 = 2;
// let c = a + b;  // Compile error

// Must convert explicitly (no `as` — use From/TryFrom)
let c = a + f64::from(b);
```

### 5.4 NaN/Inf 处理语义

| 方法 | NaN 输入 | Inf 输入 |
|------|----------|----------|
| `abs(NaN)` | NaN | Inf |
| `sqrt(-1.0)` | NaN | — |
| `ln(0.0)` | — | -Inf |
| `exp(Inf)` | — | Inf |
| `min(a, b)` 含 NaN | NaN | 正常比较 |
| `partial_cmp(NaN, _)` | None | 正常比较 |

### 5.5 RealScalar 实现（以 f64 为例）

```rust
// RealScalar is only implemented when std is available.
// In no_std environments, RealScalar math functions (sin/cos/exp/ln/etc.) are NOT available.
// Callers that need these functions in no_std should gate their code with #[cfg(feature = "std")].
#[cfg(feature = "std")]
impl RealScalar for f64 {
    fn abs(self) -> Self { self.abs() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn floor(self) -> Self { self.floor() }
    fn ceil(self) -> Self { self.ceil() }
    fn is_nan(self) -> bool { self.is_nan() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn min(self, other: Self) -> Self { self.min(other) }
    fn max(self, other: Self) -> Self { self.max(other) }
    // ...
}
// Same pattern applies to f32.
```

---

## 6. 与其他模块的交互

### 6.1 接口约定

| 模块 | 使用的 trait | 用途 |
|------|-------------|------|
| `overload` | `Numeric` | 逐元素运算泛型约束 |
| `reduction` | `Numeric`（sum）、`RealScalar`（min/max） | 归约运算泛型约束 |
| `tensor` | `Element` | Tensor<A, D> 的 A 约束 |
| `linalg` | `Numeric` | 内积运算 |
| `cast/convert` | `Element` | 类型转换 |

> 各模块的详细接口约定参见对应设计文档（`11-math.md` §4、`13-reduction.md` §4、`21-type.md` §4）。

### 6.2 接口边界

```
┌───────────────────────────────────────────────────────────────┐
│  math / reduction / linalg (使用 Element/Numeric 约束)         │
└──────────────────────┬────────────────────────────────────────┘
                       │ 泛型约束
┌──────────────────────▼────────────────────────────────────────┐
│  element (定义 trait)                                          │
└──────────────────────┬────────────────────────────────────────┘
                       │ 类型依赖
┌──────────────────────▼────────────────────────────────────────┐
│  complex (定义 Complex<T>)                                     │
└───────────────────────────────────────────────────────────────┘
```

---

## 7. 实现任务拆分

### Wave 1: 基础 trait 定义

- [ ] **T1**: 创建 `mod.rs`，定义 Sealed trait 和 Element trait
  - 文件: `src/element/mod.rs`
  - 内容: `Sealed` trait（private module）、`Element` trait 定义、模块 re-export
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 创建 `numeric.rs`，定义 Numeric trait
  - 文件: `src/element/numeric.rs`
  - 内容: `Numeric` trait 定义（marker trait，所有约束通过 supertrait）
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: 扩展 trait 定义

- [ ] **T3**: 创建 `real.rs`，定义 RealScalar trait
  - 文件: `src/element/real.rs`
  - 内容: `RealScalar` trait 定义（数学函数 + 常量 + NaN 检测）
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
  - 内容: `Element` impl（`zero()=false`, `one()=true`），不实现 Numeric
  - 测试: `test_bool_element_only`
  - 前置: T1
  - 预计: 5 min

- [ ] **T8**: 为 usize 实现 Element（仅此）
  - 文件: `src/element/primitives.rs`
  - 内容: `Element` impl（`zero()=0`, `one()=1`），不实现 Numeric
  - 测试: `test_usize_element_only`
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

- [ ] **T10**: 添加 no_std 兼容性（条件编译 std）
  - 文件: `src/element/real.rs`
  - 内容: `#[cfg(feature = "std")]` 下实现 `RealScalar`；`#[cfg(not(feature = "std"))]` 下不实现（数学方法不可用）
  - 测试: 两种 feature 组合编译通过
  - 前置: T6
  - 预计: 10 min

- [ ] **T11**: 文档注释与 cargo doc 验证
  - 文件: 所有 `src/element/` 文件
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T9
  - 预计: 10 min

- [ ] **T12**: 集成测试（跨模块交互验证）
  - 文件: `tests/element_tests.rs`
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

(T7, T8 独立于 Wave 2-3，可与 T3/T4/T5 并行)
```

---

## 8. 测试计划

### 8.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_i32_zero_one` | `i32::zero()==0`, `i32::one()==1` | 高 |
| `test_i64_zero_one` | `i64::zero()==0`, `i64::one()==1` | 高 |
| `test_i32_arithmetic` | `i32` 的 Add/Sub/Mul/Div/Neg | 高 |
| `test_f32_zero_one` | `f32::zero()==0.0`, `f32::one()==1.0` | 高 |
| `test_f64_zero_one` | `f64::zero()==0.0`, `f64::one()==1.0` | 高 |
| `test_f64_sqrt` | `f64::sqrt(4.0)==2.0` | 高 |
| `test_f64_sin_cos` | `sin(0)==0`, `cos(0)==1` | 高 |
| `test_f64_exp_ln_inverse` | `exp(ln(x))==x` | 高 |
| `test_f32_nan_detection` | `NaN.is_nan()`, `Inf.is_infinite()` | 高 |
| `test_f64_nan_propagating_min` | `min(NaN, 1.0).is_nan()` | 高 |
| `test_bool_element_only` | `bool::zero()==false`, `bool::one()==true` | 高 |
| `test_bool_not_numeric` | bool 不满足 Numeric（编译测试） | 高 |
| `test_usize_element_only` | `usize::zero()==0`, `usize::one()==1` | 中 |
| `test_usize_not_numeric` | usize 不满足 Numeric（编译测试） | 中 |
| `test_complex_f64_zero_one` | `Complex<f64>::zero()`, `Complex<f64>::one()` | 高 |
| `test_complex_f64_conj` | `Complex::new(3.0, 4.0).conj() == Complex::new(3.0, -4.0)` | 高 |
| `test_complex_f32_norm` | `Complex::new(3.0f32, 4.0f32).norm() == 5.0` | 高 |
| `test_complex_f64_from_polar` | `from_polar(1.0, PI/2) ≈ i` | 中 |
| `test_sealed_prevents_external` | 外部类型无法实现 Element（编译测试） | 中 |

### 8.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| `f64::nan().is_nan()` | 返回 `true` |
| `f64::infinity().is_finite()` | 返回 `false` |
| `f64::sqrt(-1.0).is_nan()` | 返回 `true` |
| `f64::ln(0.0)` | 返回 `-Inf` |
| `Complex::new(f64::NAN, 0.0).norm().is_nan()` | 返回 `true` |
| `bool` 张量调用 `sum()` | 编译错误（Numeric 约束不满足） |

### 8.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `A::zero() + a == a` | 所有 Numeric 类型，随机 a |
| `A::one() * a == a` | 所有 Numeric 类型，随机 a |
| `a.sqrt().sqrt() == a.powf(0.25)` | f32/f64，随机正数 a |
| `a.exp().ln() ≈ a` | f32/f64，随机有限 a |

---

## 9. 设计决策记录

### 决策 1：封闭集合，不支持下游扩展

| 属性 | 值 |
|------|-----|
| 决策 | 所有 trait 继承 Sealed，仅允许 crate 内类型实现 |
| 理由 | API 稳定性（可添加新方法不破坏外部）；所有实现类型行为经过验证；版本控制能力 |
| 替代方案 | 开放实现 — 放弃，失去版本控制能力，可能导致不一致行为 |

### 决策 2：仅支持 8 种类型

| 属性 | 值 |
|------|-----|
| 决策 | 仅支持 i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool/usize |
| 理由 | 需求说明书 §4 明确列出；减少维护负担；覆盖科学计算核心场景 |
| 替代方案 | 支持全部整数类型（u8/u16/u32/i8/i16）— 放弃，增加矩阵复杂度 |

### 决策 3：bool 排除 Numeric

| 属性 | 值 |
|------|-----|
| 决策 | `bool` 仅实现 `Element`，不实现 `Numeric` |
| 理由 | 布尔四则运算无数学意义；防止 `sum([true, false])` 等无意义操作；编译时阻止 |
| 替代方案 | bool 实现 Numeric（true=1, false=0）— 放弃，语义不清晰 |

### 决策 4：usize 仅实现 Element

| 属性 | 值 |
|------|-----|
| 决策 | `usize` 仅实现 `Element`，不实现 `Numeric`，不参与张量算术运算 |
| 理由 | usize 在 Xenon 中作为索引和形状类型使用，不作为计算类型。需求 §4 中的"整数"指 i32/i64（有符号、定宽），usize 是平台相关的无符号类型，跨平台行为不一致（32/64 位），不适合科学计算中的数值运算。允许 usize 参与算术会导致与索引语义混淆 |
| 替代方案 | usize 实现 Numeric — 放弃，语义不合适，跨平台宽度不一致 |

### 决策 5：不支持自动类型提升

| 属性 | 值 |
|------|-----|
| 决策 | 类型转换须显式，不支持隐式提升 |
| 理由 | 显式优于隐式，避免精度损失；性能可预测；与 Rust 哲学一致 |
| 替代方案 | 类似 C++ 的类型提升规则 — 放弃，增加复杂度，可能导致难以调试的精度问题 |

### 决策 6：RealScalar 和 ComplexScalar 平行继承 Numeric

| 属性 | 值 |
|------|-----|
| 决策 | 两者都继承 Numeric，无交叉继承 |
| 理由 | 提供正交的数学函数集；复数无自然全序（不应实现 PartialOrd）；未来可扩展其他标量类型 |
| 替代方案 | ComplexScalar 继承 RealScalar — 放弃，语义不正确 |

---

## 10. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 零运行时开销 | 所有 trait 约束为编译期静态分派，无虚调用 |
| 内联 | RealScalar 数学方法标注 `#[inline]` |
| 单态化 | `Tensor<A, D>` 中 A 的 trait 约束在编译期单态化 |
| Sealed | 封闭集合允许编译器做更激进的优化（已知完整类型集） |

---

## 11. no_std 兼容性

> **⚠️ 警告**：在 `no_std` 环境下，`RealScalar` trait **没有任何实现者**。`f32`/`f64` 的 `RealScalar` impl 需要 `std` feature，因为 `sin()`、`cos()`、`sqrt()`、`exp()`、`ln()`、`hypot()`、`atan2()` 等浮点数学方法在 Rust 1.85 中仍通过 `std` 提供（位于 `std` 而非 `core`），而非因为 Xenon 引入了 `libm` 依赖。因此 `no_std` 环境中这些数学函数**需要 std feature**（浮点数学函数在 Rust 1.85 中仍通过 std 提供）。需要这些函数的代码必须以 `#[cfg(feature = "std")]` 门控。仅使用 `Element`/`Numeric` trait 的代码在 `no_std` 下正常工作。

| 组件 | 兼容方案 |
|------|----------|
| `Element` / `Numeric` / `ComplexScalar` | 纯 trait，天然 no_std |
| `RealScalar` 数学函数 | 仅在 `#[cfg(feature = "std")]` 下实现；**需要 std feature**（浮点数学函数 `sin`/`cos`/`sqrt`/`exp`/`ln`/`floor`/`ceil`/`abs` 在 Rust 1.85 中仍通过 `std` 提供）。no_std 环境中不可用，需要数学函数的代码须以 `#[cfg(feature = "std")]` 门控 |
| `RealScalar` 常量与 NaN 检测 | `epsilon()`/`min_positive()`/`max_value()`/`infinity()`/`neg_infinity()`/`nan()`/`is_nan()`/`is_infinite()`/`is_finite()`/`min()`/`max()` 不依赖 `std` 数学函数，理论上可在 no_std 下实现；但当前 `RealScalar` 整体以 `#[cfg(feature = "std")]` 门控，保持 impl 一致性 |
| Feature gate | `Cargo.toml`: `default = ["std"]`；`RealScalar` 数学函数需要 std feature（浮点数学函数在 Rust 1.85 中仍通过 `std` 提供，Xenon 不引入额外 libm 依赖） |

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
