# 元素类型体系设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `src/element/`  
> **需求来源**: require-v18.md §2.2-2.7

---

## 1. 模块概述

### 1.1 设计哲学

元素类型体系（Element Type System）是 Xenon 数值计算库的基础设施，采用**分层 trait 继承**设计模式。该体系通过四层 trait 逐级叠加能力约束，在编译时确保类型安全，同时为泛型算法提供精确的能力边界。

**设计原则**:
- **能力最小化**: 每层 trait 仅声明必要的约束，避免过度限制
- **正交性**: 数值运算（Numeric）、实数函数（RealScalar）、复数运算（ComplexScalar）职责分离
- **零运行时开销**: 所有约束均为编译期静态分派
- **IEEE 754 兼容**: 浮点特殊值（NaN、Inf）处理遵循标准语义

### 1.2 在架构中的位置

```
┌─────────────────────────────────────────────────────────┐
│                    用户层 API                            │
│         Tensor<A, D>, TensorView, TensorViewMut         │
└─────────────────────┬───────────────────────────────────┘
                      │ 泛型约束 A: Element / Numeric / RealScalar
┌─────────────────────▼───────────────────────────────────┐
│                 元素类型体系 (本模块)                      │
│  Element ← Numeric ← RealScalar                         │
│          ← Numeric ← ComplexScalar                      │
└─────────────────────┬───────────────────────────────────┘
                      │ 为基础类型实现 trait
┌─────────────────────▼───────────────────────────────────┐
│              基础类型 (primitives.rs)                    │
│    u8, u16, u32, u64, i8, i16, i32, i64                 │
│    f32, f64, bool, Complex<f32>, Complex<f64>           │
└─────────────────────────────────────────────────────────┘
```

### 1.3 依赖关系

| 依赖模块 | 用途 |
|---------|------|
| `crate::complex` | 提供 `Complex<T>` 类型定义 |
| `core::ops` | 算术运算符 trait（Add, Sub, Mul, Div, Neg） |
| `core::fmt` | Debug/Display 格式化 |
| `core::cmp` | PartialEq/PartialOrd 比较 |

---

## 2. 文件结构

```
src/element/
├── mod.rs             # 模块入口，定义 Element trait 及 Sealed 机制
├── numeric.rs         # Numeric trait 定义（四则运算约束）
├── real.rs            # RealScalar trait 定义（实数数学函数）
├── complex.rs         # ComplexScalar trait 定义（复数运算）
└── primitives.rs      # 为基础类型实现各层 trait
```

### 2.1 各文件职责

| 文件 | 职责 | 核心内容 |
|------|------|----------|
| `mod.rs` | 模块组织与 Element trait | Element trait 定义、Sealed trait、模块 re-export |
| `numeric.rs` | 数值运算约束 | Numeric: Element + Add + Sub + Mul + Div + Neg |
| `real.rs` | 实数函数接口 | RealScalar: Numeric + 数学函数 + 常量 + NaN 检测 |
| `complex.rs` | 复数运算接口 | ComplexScalar: Numeric + 复数方法 |
| `primitives.rs` | 基础类型实现 | 为 u8/i32/f32/f64/bool/Complex 实现所有适用 trait |

---

## 3. Element trait 设计

### 3.1 Trait 定义

```rust
/// 元素类型基础约束
///
/// 所有可作为张量元素的类型必须实现此 trait。
/// 提供零元、单位元、值语义、相等比较、格式化及线程安全保证。
pub trait Element: 
    Copy + 
    Clone + 
    PartialEq + 
    core::fmt::Debug + 
    core::fmt::Display + 
    Send + 
    Sync +
    Sealed
{
    /// 加法单位元（零）
    ///
    /// # Examples
    /// ```
    /// assert_eq!(0i32, i32::zero());
    /// assert_eq!(0.0f64, f64::zero());
    /// ```
    fn zero() -> Self;

    /// 乘法单位元（一）
    ///
    /// # Examples
    /// ```
    /// assert_eq!(1i32, i32::one());
    /// assert_eq!(1.0f64, f64::one());
    /// ```
    fn one() -> Self;
}
```

### 3.2 Supertrait 说明

| Supertrait | 作用 |
|------------|------|
| `Copy` | 值语义，可按位复制，避免所有权转移开销 |
| `Clone` | 显式克隆能力（Copy 的超集） |
| `PartialEq` | 相等比较，用于断言和测试 |
| `Debug` | 调试格式输出 `{:?}` |
| `Display` | 用户友好格式输出 `{}` |
| `Send` | 可跨线程移动（并行迭代必需） |
| `Sync` | 可跨线程共享引用（并行只读访问必需） |
| `Sealed` | 防止外部类型实现（见 §9） |

### 3.3 实现类型列表

| 类型 | 实现 Element | 备注 |
|------|-------------|------|
| `u8` | ✓ | 无符号 8 位整数 |
| `u16` | ✓ | 无符号 16 位整数 |
| `u32` | ✓ | 无符号 32 位整数 |
| `u64` | ✓ | 无符号 64 位整数 |
| `i8` | ✓ | 有符号 8 位整数 |
| `i16` | ✓ | 有符号 16 位整数 |
| `i32` | ✓ | 有符号 32 位整数 |
| `i64` | ✓ | 有符号 64 位整数 |
| `f32` | ✓ | 32 位浮点 |
| `f64` | ✓ | 64 位浮点 |
| `bool` | ✓ | 布尔值（仅基础层） |
| `Complex<f32>` | ✓ | 32 位复数 |
| `Complex<f64>` | ✓ | 64 位复数 |

---

## 4. Numeric trait 设计

### 4.1 Trait 定义

```rust
/// 数值类型约束
///
/// 在 Element 基础上增加四则运算能力。
/// 仅数值类型（整数、浮点、复数）可实现此 trait，bool 不实现。
pub trait Numeric: 
    Element + 
    Add<Output = Self> + 
    Sub<Output = Self> + 
    Mul<Output = Self> + 
    Div<Output = Self> + 
    Neg<Output = Self> +
    Sealed
{
    // Marker trait, no additional methods
    // 所有约束通过 supertrait 表达
}
```

### 4.2 与 Element 的关系

```
Element (基础约束)
    ↑
    │ 继承
    │
Numeric (增加四则运算)
```

- `Numeric` 继承 `Element` 的所有约束
- 额外要求 `Add/Sub/Mul/Div/Neg` 运算符，且返回类型为 `Self`
- `Numeric` 自身不定义新方法，仅作为**标记 trait** 表达能力组合

### 4.3 bool 排除策略

**机制**: `bool` 类型**仅实现 `Element`**，不实现 `Numeric`。

**原因**:
1. 布尔值四则运算无数学意义
2. 防止 `bool` 参与数值归约（sum、prod）
3. 编译时阻止无效泛型实例化

**实现方式**:
```rust
// primitives.rs

// bool 实现 Element
impl Element for bool {
    fn zero() -> Self { false }
    fn one() -> Self { true }
}

// bool 不实现 Numeric（无 impl Numeric for bool）
// 编译器会在泛型约束时报错
```

**编译时检查示例**:
```rust
fn sum<A: Numeric>(arr: &[A]) -> A {
    arr.iter().fold(A::zero(), |acc, &x| acc + x)
}

// sum(&[true, false, true]);  // 编译错误：bool 不满足 Numeric
sum(&[1i32, 2, 3]);             // 编译通过
```

### 4.4 实现类型列表

| 类型 | 实现 Numeric | 备注 |
|------|-------------|------|
| `u8` | ✓ | 整数除法向零截断 |
| `u16` | ✓ | |
| `u32` | ✓ | |
| `u64` | ✓ | |
| `i8` | ✓ | |
| `i16` | ✓ | |
| `i32` | ✓ | |
| `i64` | ✓ | |
| `f32` | ✓ | IEEE 754 浮点 |
| `f64` | ✓ | IEEE 754 浮点 |
| `bool` | ✗ | **排除** |
| `Complex<f32>` | ✓ | 复数四则运算 |
| `Complex<f64>` | ✓ | 复数四则运算 |

---

## 5. RealScalar trait 设计

### 5.1 Trait 定义

```rust
/// 实数标量约束
///
/// 在 Numeric 基础上提供数学函数、常量及特殊值检测。
/// 仅 f32 和 f64 实现此 trait。
pub trait RealScalar: Numeric + PartialOrd + Sealed {
    // ========== 数学函数 ==========
    
    /// 绝对值
    fn abs(self) -> Self;
    
    /// 平方根
    fn sqrt(self) -> Self;
    
    /// 立方根
    fn cbrt(self) -> Self;
    
    /// 自然对数（底为 e）
    fn ln(self) -> Self;
    
    /// 以 2 为底的对数
    fn log2(self) -> Self;
    
    /// 以 10 为底的对数
    fn log10(self) -> Self;
    
    /// 自然指数（e^x）
    fn exp(self) -> Self;
    
    /// 以 2 为底的指数（2^x）
    fn exp2(self) -> Self;
    
    /// 正弦
    fn sin(self) -> Self;
    
    /// 余弦
    fn cos(self) -> Self;
    
    /// 正切
    fn tan(self) -> Self;
    
    /// 反余弦
    fn asin(self) -> Self;
    
    /// 反余弦
    fn acos(self) -> Self;
    
    /// 反正切
    fn atan(self) -> Self;
    
    /// 双参数反正切（atan2(y, x)）
    fn atan2(self, other: Self) -> Self;
    
    /// 双曲正弦
    fn sinh(self) -> Self;
    
    /// 双曲余弦
    fn cosh(self) -> Self;
    
    /// 双曲正切
    fn tanh(self) -> Self;
    
    /// 向下取整
    fn floor(self) -> Self;
    
    /// 向上取整
    fn ceil(self) -> Self;
    
    /// 四舍五入
    fn round(self) -> Self;
    
    /// 整数幂（x^n）
    fn powi(self, n: i32) -> Self;
    
    /// 实数幂（x^y）
    fn powf(self, n: Self) -> Self;

    // ========== 常量 ==========
    
    /// 机器精度（1.0 与下一个可表示值的差）
    fn epsilon() -> Self;
    
    /// 最小正规正数
    fn min_positive() -> Self;
    
    /// 最大有限值
    fn max_value() -> Self;
    
    /// 正无穷
    fn infinity() -> Self;
    
    /// 负无穷
    fn neg_infinity() -> Self;
    
    /// NaN（非数值）
    fn nan() -> Self;

    // ========== 特殊值检测 ==========
    
    /// 是否为 NaN
    fn is_nan(self) -> bool;
    
    /// 是否为无穷大（正或负）
    fn is_infinite(self) -> bool;
    
    /// 是否为有限值
    fn is_finite(self) -> bool;

    // ========== NaN 传播的 min/max ==========
    
    /// 最小值（NaN 传播）
    ///
    /// 任一参数为 NaN 时返回 NaN。
    fn min(self, other: Self) -> Self;
    
    /// 最大值（NaN 传播）
    ///
    /// 任一参数为 NaN 时返回 NaN。
    fn max(self, other: Self) -> Self;
}
```

### 5.2 与 Numeric 的关系

```
Element
    ↑
Numeric + PartialOrd
    ↑
    │ 继承
    │
RealScalar (数学函数 + 常量 + NaN 检测)
```

- `RealScalar` 要求 `Numeric + PartialOrd`（实数可偏序比较）
- 额外提供约 30 个数学方法
- 仅 `f32` 和 `f64` 实现

### 5.3 NaN/Inf 处理语义

| 方法 | NaN 输入行为 | Inf 输入行为 |
|------|-------------|-------------|
| `abs()` | 返回 NaN | 返回 Inf |
| `sqrt(-1.0)` | 返回 NaN | — |
| `ln(0.0)` | — | 返回 -Inf |
| `ln(-1.0)` | 返回 NaN | — |
| `exp(Inf)` | — | 返回 Inf |
| `min(a, b)` | 任一为 NaN → NaN | 正常比较 |
| `max(a, b)` | 任一为 NaN → NaN | 正常比较 |
| `a.partial_cmp(&b)` | 任一为 NaN → None | 正常比较 |

**关键约定**:
- `min`/`max` 采用 **NaN 传播语义**（与 `f32::min`/`f64::min` 一致）
- `PartialOrd::partial_cmp` 对 NaN 返回 `None`（符合 IEEE 754）
- 所有算术运算遵循 IEEE 754 规则

### 5.4 实现类型列表

| 类型 | 实现 RealScalar |
|------|----------------|
| `f32` | ✓ |
| `f64` | ✓ |
| 所有整数 | ✗ |
| `Complex<T>` | ✗ |

---

## 6. ComplexScalar trait 设计

### 6.1 Trait 定义

```rust
/// 复数标量约束
///
/// 在 Numeric 基础上提供复数特有操作。
/// 仅 Complex<f32> 和 Complex<f64> 实现此 trait。
pub trait ComplexScalar: Numeric + Sealed {
    /// 实部类型（必须是 RealScalar）
    type Real: RealScalar;

    /// 获取实部
    fn re(self) -> Self::Real;

    /// 获取虚部
    fn im(self) -> Self::Real;

    /// 共轭复数
    fn conj(self) -> Self;

    /// 模（使用 hypot 算法避免溢出）
    fn norm(self) -> Self::Real;

    /// 辐角（返回 (-π, π]）
    fn arg(self) -> Self::Real;

    /// 复数指数（e^z）
    fn exp(self) -> Self;

    /// 复数对数（主值）
    fn ln(self) -> Self;

    /// 复数平方根（主值）
    fn sqrt(self) -> Self;

    /// 从极坐标构造
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

    /// 虚数单位（0 + 1i）
    fn i() -> Self;
}
```

### 6.2 与 RealScalar 的关系

```
Element
    ↑
Numeric
    ↑
    ├──────────┬──────────┐
    │          │          │
RealScalar  ComplexScalar  (未来可扩展其他标量类型)
    │          │
  f32/f64   Complex<f32/f64>
```

- `RealScalar` 和 `ComplexScalar` **平行继承** `Numeric`，无交叉继承
- `ComplexScalar` 通过关联类型 `Real` 引用实部类型
- 两者提供**正交**的数学函数集

### 6.3 实现类型列表

| 类型 | 实现 ComplexScalar | Real 关联类型 |
|------|-------------------|--------------|
| `Complex<f32>` | ✓ | `f32` |
| `Complex<f64>` | ✓ | `f64` |

### 6.4 关键方法说明

| 方法 | 算法/约定 |
|------|----------|
| `norm()` | 使用 `hypot(re, im)` 避免中间溢出 |
| `arg()` | 使用 `atan2(im, re)`，返回 `(-π, π]` |
| `sqrt()` | 返回主值（实部非负） |
| `ln()` | 返回主值（虚部在 `(-π, π]`） |
| `from_polar()` | `r * (cos(theta) + i*sin(theta))` |

---

## 7. primitives.rs 实现矩阵

### 7.1 完整实现矩阵

| 类型 | Element | Numeric | RealScalar | ComplexScalar |
|------|:-------:|:-------:|:----------:|:-------------:|
| `u8` | ✓ | ✓ | ✗ | ✗ |
| `u16` | ✓ | ✓ | ✗ | ✗ |
| `u32` | ✓ | ✓ | ✗ | ✗ |
| `u64` | ✓ | ✓ | ✗ | ✗ |
| `i8` | ✓ | ✓ | ✗ | ✗ |
| `i16` | ✓ | ✓ | ✗ | ✗ |
| `i32` | ✓ | ✓ | ✗ | ✗ |
| `i64` | ✓ | ✓ | ✗ | ✗ |
| `f32` | ✓ | ✓ | ✓ | ✗ |
| `f64` | ✓ | ✓ | ✓ | ✗ |
| `bool` | ✓ | ✗ | ✗ | ✗ |
| `Complex<f32>` | ✓ | ✓ | ✗ | ✓ |
| `Complex<f64>` | ✓ | ✓ | ✗ | ✓ |

### 7.2 实现要点

**整数类型**（u8/u16/u32/u64/i8/i16/i32/i64）:
- 实现 `Element` + `Numeric`
- `zero()` = `0`
- `one()` = `1`
- 四则运算使用 `core::ops` 默认实现

**浮点类型**（f32/f64）:
- 实现 `Element` + `Numeric` + `RealScalar`
- 数学函数委托给 `libm`（no_std）或 `std::math`（std）
- `min`/`max` 使用 `f32::min`/`f64::min`（NaN 传播）

**布尔类型**（bool）:
- 仅实现 `Element`
- `zero()` = `false`
- `one()` = `true`

**复数类型**（Complex<f32>/Complex<f64>）:
- 实现 `Element` + `Numeric` + `ComplexScalar`
- `zero()` = `Complex { re: 0.0, im: 0.0 }`
- `one()` = `Complex { re: 1.0, im: 0.0 }`

---

## 8. NaN/Inf 处理约定

### 8.1 IEEE 754 兼容性

Xenon 严格遵循 IEEE 754 浮点标准，确保与 Rust 标准库行为一致。

### 8.2 场景行为规范

| 场景 | 行为 | 示例 |
|------|------|------|
| 归约含 NaN | 结果为 NaN | `sum([1.0, NaN, 2.0])` → `NaN` |
| `min(a, b)` 含 NaN | 任一为 NaN → NaN | `min(1.0, NaN)` → `NaN` |
| `max(a, b)` 含 NaN | 任一为 NaN → NaN | `max(NaN, 2.0)` → `NaN` |
| `partial_cmp` 含 NaN | 返回 `None` | `NaN.partial_cmp(&1.0)` → `None` |
| `0.0 / 0.0` | 返回 NaN | `0.0 / 0.0` → `NaN` |
| `1.0 / 0.0` | 返回 Inf | `1.0 / 0.0` → `Inf` |
| `-1.0 / 0.0` | 返回 -Inf | `-1.0 / 0.0` → `-Inf` |
| `Inf + (-Inf)` | 返回 NaN | `Inf + (-Inf)` → `NaN` |
| `Inf * 0.0` | 返回 NaN | `Inf * 0.0` → `NaN` |

### 8.3 整数除法

整数类型无 NaN/Inf，除零行为：
- **调试模式**: panic（Rust 默认行为）
- **发布模式**: 未定义行为（由编译器决定）

**建议**: 上层 API 应在除法前检查除数，或使用 `checked_div`。

### 8.4 复数 NaN/Inf

`Complex<T>` 的 NaN/Inf 由实部和虚部独立决定：
- `Complex { re: NaN, im: 0.0 }` 的 `norm()` 为 NaN
- `Complex { re: Inf, im: 0.0 }` 的 `exp()` 可能为 Inf 或 NaN（取决于算法）

---

## 9. Sealed trait 策略

### 9.1 设计目标

防止外部 crate 为自定义类型实现 `Element`/`Numeric`/`RealScalar`/`ComplexScalar`，确保：
1. **API 稳定性**: 内部实现可重构，不影响外部
2. **一致性保证**: 所有实现类型的行为经过验证
3. **版本控制**: 未来添加新方法不会破坏外部实现

### 9.2 实现机制

```rust
// mod.rs

/// 私有标记 trait，防止外部实现
mod private {
    pub trait Sealed {}
}

// 公开别名（仅在当前 crate 内可见）
pub(crate) use private::Sealed;

// 或使用 pub use 配合隐藏构造函数
pub trait Sealed: private::Sealed {}
```

### 9.3 应用示例

```rust
// primitives.rs

use crate::element::Sealed;

// 为 f64 实现 Sealed（仅限本 crate）
impl Sealed for f64 {}

// 现在可以实现 Element
impl Element for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}
```

### 9.4 外部尝试实现的编译错误

```rust
// 外部 crate 尝试
use xenon::element::Element;

struct MyType;

impl Element for MyType {  // 编译错误
    // error[E0277]: the trait bound `MyType: Sealed` is not satisfied
}
```

---

## 10. 与其他模块的交互

### 10.1 与 `complex` 模块的交互

| 交互点 | 说明 |
|--------|------|
| 类型定义 | `Complex<T>` 定义在 `crate::complex` |
| Trait 实现 | `ComplexScalar` 在 `element` 模块定义，在 `primitives.rs` 为 `Complex<T>` 实现 |
| 依赖方向 | `element` 依赖 `complex`（类型定义） |

```rust
// complex.rs
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

// element/primitives.rs
use crate::complex::Complex;

impl Element for Complex<f64> { /* ... */ }
impl ComplexScalar for Complex<f64> { /* ... */ }
```

### 10.2 与 `ops` 模块的交互

| 交互点 | 说明 |
|--------|------|
| 泛型约束 | `ops::add()` 要求 `A: Numeric` |
| 运算符重载 | `Tensor` 实现 `Add/Sub/Mul/Div` 基于 `A: Numeric` |

```rust
// ops.rs
use crate::element::Numeric;

pub fn add<A: Numeric>(a: &Tensor<A>, b: &Tensor<A>) -> Tensor<A> {
    // 逐元素相加，依赖 A: Add<Output=A>
}
```

### 10.3 与 `reduction` 模块的交互

| 交互点 | 说明 |
|--------|------|
| `sum`/`prod` | 要求 `A: Numeric`（需要 `zero()`/`one()` 和加法/乘法） |
| `min`/`max` | 要求 `A: RealScalar`（需要 NaN 传播语义） |
| `any`/`all` | 要求 `A: Element`（仅逻辑归约） |

```rust
// reduction.rs
use crate::element::{Numeric, RealScalar};

pub fn sum<A: Numeric>(tensor: &TensorView<A>) -> A {
    tensor.iter().fold(A::zero(), |acc, &x| acc + x)
}

pub fn min<A: RealScalar>(tensor: &TensorView<A>) -> A {
    tensor.iter().fold(A::infinity(), |acc, &x| acc.min(x))
}
```

### 10.4 接口边界

```
┌─────────────────────────────────────────────────────────┐
│  ops / reduction / linalg (使用 Element/Numeric 约束)   │
└────────────────────┬────────────────────────────────────┘
                     │ 泛型约束
┌────────────────────▼────────────────────────────────────┐
│  element (定义 trait)                                   │
└────────────────────┬────────────────────────────────────┘
                     │ 类型依赖
┌────────────────────▼────────────────────────────────────┐
│  complex (定义 Complex<T>)                              │
└─────────────────────────────────────────────────────────┘
```

---

## 11. 实现任务分解

### 任务清单

| # | 任务 | 文件 | 预估时间 | 依赖 |
|---|------|------|----------|------|
| 1 | 创建 `mod.rs`，定义 `Sealed` trait 和 `Element` trait | `mod.rs` | 10 min | 无 |
| 2 | 创建 `numeric.rs`，定义 `Numeric` trait | `numeric.rs` | 5 min | #1 |
| 3 | 创建 `real.rs`，定义 `RealScalar` trait | `real.rs` | 15 min | #2 |
| 4 | 创建 `complex.rs`，定义 `ComplexScalar` trait | `complex.rs` | 10 min | #2 |
| 5 | 在 `primitives.rs` 为整数类型实现 `Element` + `Numeric` | `primitives.rs` | 10 min | #1, #2 |
| 6 | 在 `primitives.rs` 为 `f32`/`f64` 实现 `Element` + `Numeric` + `RealScalar` | `primitives.rs` | 15 min | #1, #2, #3 |
| 7 | 在 `primitives.rs` 为 `bool` 实现 `Element`（仅此） | `primitives.rs` | 5 min | #1 |
| 8 | 在 `primitives.rs` 为 `Complex<f32>`/`Complex<f64>` 实现 `Element` + `Numeric` + `ComplexScalar` | `primitives.rs` | 15 min | #1, #2, #4 |
| 9 | 添加 `no_std` 兼容性（条件编译 `libm` vs `std`） | `real.rs` | 10 min | #3 |
| 10 | 编写单元测试（各类型的 `zero()`/`one()` 及数学函数） | `tests/element_test.rs` | 15 min | #5-#8 |
| 11 | 编写文档测试（trait 文档中的示例） | 各 trait 文件 | 10 min | #1-#4 |
| 12 | 集成测试（与 `ops`/`reduction` 模块的交互） | `tests/integration_test.rs` | 10 min | #10 |

**总预估时间**: 约 130 分钟（2 小时 10 分钟）

### 任务依赖图

```
#1 (Element) ─┬─→ #2 (Numeric) ─┬─→ #3 (RealScalar) ─→ #6 (f32/f64 impl)
              │                 │
              │                 ├─→ #4 (ComplexScalar) ─→ #8 (Complex impl)
              │                 │
              │                 └─→ #5 (整数 impl)
              │
              └─→ #7 (bool impl)

#3 ─→ #9 (no_std)
#5, #6, #7, #8 ─→ #10 (单元测试)
#1-#4 ─→ #11 (文档测试)
#10 ─→ #12 (集成测试)
```

---

## 12. 设计决策记录

### 12.1 为什么采用四层 trait 体系？

**决策**: 使用 `Element` → `Numeric` → `RealScalar`/`ComplexScalar` 分层。

**理由**:
1. **能力最小化**: 每层仅声明必要约束，避免过度限制泛型
2. **正交性**: 实数函数和复数函数分离，避免 `Complex` 实现无意义的 `sin`/`cos`
3. **编译时检查**: `bool` 无法用于 `sum`（不实现 `Numeric`），整数无法调用 `sqrt`（不实现 `RealScalar`）

**替代方案**: 单一 `Scalar` trait 包含所有方法（使用默认实现或 `Option` 返回值）。
**拒绝原因**: 运行时检查增加开销，API 不清晰。

### 12.2 为什么 `Numeric` 不定义任何方法？

**决策**: `Numeric` 仅通过 supertrait 组合约束，不定义新方法。

**理由**:
1. 四则运算已由 `Add/Sub/Mul/Div/Neg` trait 提供
2. 避免重复定义（与标准库冲突）
3. 作为"标记 trait"表达能力组合更符合 Rust 惯例

### 12.3 为什么 `bool` 不实现 `Numeric`？

**决策**: `bool` 仅实现 `Element`。

**理由**:
1. 布尔四则运算无数学意义
2. 防止 `sum([true, false])` 等无意义操作
3. `bool` 可用于逻辑归约（`any`/`all`），但不应参与数值计算

### 12.4 为什么 `RealScalar::min`/`max` 采用 NaN 传播语义？

**决策**: 任一参数为 NaN 时返回 NaN。

**理由**:
1. 与 `f32::min`/`f64::min` 标准库行为一致
2. NaN 传播符合 IEEE 754 哲学（异常值可见）
3. 避免 NaN 被意外忽略（如 `min(NaN, 1.0)` 若返回 `1.0` 会隐藏错误）

**替代方案**: 忽略 NaN（类似 `NaN-aware min`）。
**拒绝原因**: 与标准库不一致，可能导致难以调试的 bug。

### 12.5 为什么 `ComplexScalar` 和 `RealScalar` 平行继承 `Numeric`？

**决策**: 两者都继承 `Numeric`，无交叉继承。

**理由**:
1. 复数和实数提供**正交**的数学函数集
2. 复数无自然全序，不应实现 `PartialOrd`（`RealScalar` 需要）
3. 未来可扩展其他标量类型（如定点数）而不影响现有层次

### 12.6 为什么使用 Sealed trait？

**决策**: 所有 trait 继承私有 `Sealed`。

**理由**:
1. **API 稳定**: 添加新方法不会破坏外部实现
2. **一致性**: 所有实现类型由 Xenon 团队验证
3. **未来扩展**: 可添加关联类型或常量而不影响下游

**替代方案**: 开放实现（外部 crate 可为自定义类型实现）。
**拒绝原因**: 失去版本控制能力，可能导致不一致行为。

### 12.7 为什么不支持自动类型提升？

**决策**: 类型转换须显式（如 `i32` + `f64` 须先 `as f64`）。

**理由**:
1. **显式优于隐式**: 避免精度损失或意外行为
2. **性能可预测**: 无隐式转换开销
3. **与 Rust 哲学一致**: Rust 不支持运算符隐式转换

**替代方案**: 类似 C++ 的类型提升规则。
**拒绝原因**: 增加复杂度，可能导致难以调试的精度问题。

### 12.8 为什么 `Complex<T>` 的 `norm()` 使用 hypot？

**决策**: 使用 `hypot(re, im)` 而非 `sqrt(re*re + im*im)`。

**理由**:
1. **避免溢出**: `re*re` 可能溢出，而 `hypot` 使用缩放算法
2. **数值稳定性**: `hypot` 在极端值下更准确

**示例**:
```rust
let c = Complex { re: 1e200_f64, im: 1e200_f64 };
// sqrt(re*re + im*im) → 溢出 → Inf
// hypot(re, im)       → 正确 → 1.414e200
```

---

## 附录 A: no_std 兼容性

### A.1 Feature Gate 配置

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
```

### A.2 条件编译策略

```rust
// real.rs
#[cfg(feature = "std")]
use std::math::{sqrt, ln, exp, sin, cos, /* ... */ };

#[cfg(not(feature = "std"))]
use libm::{sqrt, ln, exp, sin, cos, /* ... */ };

impl RealScalar for f64 {
    fn sqrt(self) -> Self {
        sqrt(self)  // 根据 feature 选择 std 或 libm
    }
}
```

### A.3 依赖

| Feature | 依赖 |
|---------|------|
| `std` | 无（使用 `std::f64::consts` 和 `std::math`） |
| `no_std` | `libm` crate（提供浮点数学函数） |

---

## 附录 B: 常见问题

### B.1 如何为自定义类型实现 Element？

**答**: 不支持。使用 newtype 模式包装基础类型：

```rust
struct MyFloat(f64);

impl Element for MyFloat {
    fn zero() -> Self { MyFloat(0.0) }
    fn one() -> Self { MyFloat(1.0) }
}
// 需手动实现所有 supertrait
```

### B.2 为什么整数不支持 `sqrt()`？

**答**: `sqrt()` 定义在 `RealScalar`，仅 `f32`/`f64` 实现。整数需先转换：

```rust
let x: i32 = 4;
let y = (x as f64).sqrt() as i32;  // 显式转换
```

### B.3 复数如何比较大小？

**答**: 复数无自然全序，不实现 `Ord`/`PartialOrd`。使用 `norm()` 比较：

```rust
let a = Complex { re: 3.0, im: 4.0 };
let b = Complex { re: 5.0, im: 0.0 };
if a.norm() < b.norm() { /* ... */ }  // 比较模
```

---

**文档结束**
