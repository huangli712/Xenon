# 复数类型模块设计

> 文档编号: 04
> 模块目录: src/complex/
> 任务阶段: Phase 1
> 前置文档: 00-coding.md, 01-architecture.md
> 需求参考: 需求说明书 §4、§5、§12 - §15、§23 - §25、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责         | 包含                                                                             |
| ------------ | -------------------------------------------------------------------------------- |
| 类型定义     | `Complex<T>` 结构体（`#[repr(C)]`，re/im 字段）                                  |
| 构造方法     | `new(re, im)`                                                                    |
| 基础方法     | `re()`, `im()`, `conj()`, `is_real()`, `is_imaginary()`                          |
| 数学方法     | `norm()`, `norm_sqr()`                                                           |
| 算术运算     | Complex±Complex, Complex×Complex, Complex÷Complex, 一元负号                      |
| 实数构造     | `From<T> for Complex<T>` 的显式标量构造                                          |
| 混合运算边界 | `Complex<T> op Complex<T>` 前须先完成显式构造                                    |
| 格式化输出   | Display（`"a+bj"` / `"a-bj"`）, Debug                                            |
| 双字段 C 布局基础 | `#[repr(C)]` + 编译期静态断言                                               |
| 类型转换语义| 定义 `Complex<f32>↔Complex<f64>`, `f32/f64→Complex`, `i32/i64→Complex` 的语义边界 |

| 职责              | 不包含                                                               |
| ----------------- | -------------------------------------------------------------------- |
| 构造方法          | 额外公开构造器（如 `from_polar`）                                    |
| 数学方法          | `to_polar()`、`arg` / `exp` / `ln` / `sqrt`、复数 FFT、高阶复数运算  |
| 算术运算          | 跨精度混合运算                                                       |
| 混合运算边界      | `Complex<T> op T` / `T op Complex<T>` 便捷运算符；跨精度混合运算     |
| 双字段 C 布局基础 | 跨精度混合运算                                                       |
| 类型转换语义      | 转换实现入口与张量级转换 owner（由 `convert/` 负责）                 |

### 1.2 设计原则

| 原则         | 体现                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| FFI 友好     | `#[repr(C)]` 保证字段顺序稳定，并为与两字段 C 结构体互操作提供布局基础 |
| 零依赖       | 不引入任何外部 crate（不使用 num-complex）                             |
| 同元素类型运算 | 运算阶段只允许同元素类型；跨精度/实复类型变化仅允许通过显式转换完成  |
| 数值稳定     | `norm()` 使用 hypot 算法避免中间溢出                                   |
| NaN 语义正确 | `NaN != NaN`，不实现 Eq/Ord                                            |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                      |
| -------- | --------------------------------------------------------- |
| 需求映射 | 需求说明书 §4、§5、§12 - §15、§23 - §25、§28              |
| 范围内   | `Complex<T>` 类型定义、同精度复数算术                     |
| 范围内   | `conj()` / `norm()` / `norm_sqr()`                        |
| 范围内   | 格式化、类型转换、FFI 布局边界                            |
| 范围外   | 跨精度混合运算、FFT 与高阶复数算法、额外公开复数数学函数  |
| 非目标   | 引入第三方复数库依赖、开放任意 `T` 的公开实例化           |

---

## 3. 文件位置

```
src/complex/
├── mod.rs     # Complex<T> definition, basic methods, math methods, PartialEq, Display, layout checks
└── ops.rs     # Arithmetic operator impls (Add/Sub/Mul/Div/Neg for Complex<T>)

src/convert/
└── cast.rs    # Consumes element::CastTo and hosts Complex-related conversion implementations
```

文件职责按模块 owner 划分：`complex/` 负责类型定义与运算；`CastTo` trait 定义位于 `element` 模块；复数相关的转换实现归入 `convert/` 模块（参见 `21-type.md`）。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/complex/
├── crate::private   # Sealed trait for ComplexFloat
├── core::ops        # Add/Sub/Mul/Div/Neg operator traits
├── core::fmt        # Debug/Display formatting
└── core::cmp        # PartialEq

src/convert/
└── cast.rs          # Conversion owner for Complex-related CastTo/From implementations
```

- `Complex<T>` 不引入任何第三方 crate
- `complex/` 自身仅依赖 `crate::private` 与 `core`
- 可恢复类型转换错误由 `convert/` owner 在其实现入口中处理
- 涉及浮点数学的复数方法运行在 Xenon 的 `std` 环境前提下。

### 4.2 依赖精确到类型级

| 来源模块     | 使用的类型/trait                                         |
| ------------ | -------------------------------------------------------- |
| `private`    | `Sealed`（封闭 `ComplexFloat` 的公开实现范围）           |
| `core::ops`  | `Add`, `Sub`, `Mul`, `Div`, `Neg`（算术运算）            |
| `core::fmt`  | `Debug`, `Display`（格式化输出）                         |
| `core::cmp`  | `PartialEq`（相等比较）                                  |
| `convert/`   | 作为 `CastTo`/`XenonError` 的 owner 承载复数相关转换实现 |

### 4.3 依赖合法性

| 项目           | 结论                           |
| -------------- | ------------------------------ |
| 新增第三方依赖 | 无                             |
| 合法性结论     | 符合需求说明书最小依赖限制     |
| 替代方案       | 不适用                         |

### 4.4 依赖方向声明

**依赖方向**：核心内聚、单向向下。`complex/` 仅依赖项目内基础模块 `private` 与 `core`，不依赖其他上层业务模块；类型转换实现入口由 `convert/` owner 承载。

---

## 5. 公共 API 设计

### 5.1 Complex<T> 完整定义

```rust,ignore
/// Complex number: a + bj.
///
/// # Memory layout
///
/// - `#[repr(C)]` guarantees fields are laid out in declaration order
/// - Size: `2 * size_of::<T>()`
/// - Alignment: `align_of::<T>()`
/// - Layout-compatible with a two-field C struct `{ T re; T im; }`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex<T: ComplexFloat> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}
```

- `Complex<T>` 保持公开泛型形式，但其公开 bound 使用 `pub trait ComplexFloat`。
- 该 trait 通过 `private::Sealed` supertrait 封闭实现范围。
- 下游只能使用 `Complex<f32>` 与`Complex<f64>`，不能为其他类型自行实现 `ComplexFloat`。

### 5.2 泛型约束

`Complex<T>` 的方法分为两类：

1. **基础方法**（无需浮点数学）：在 `impl<T: ComplexFloat> Complex<T>` 中实现，公开可用。包括 `re()`、`im()`、`conj()`、`from_imag()`、`is_real()`、`is_imaginary()`；纯实数构造统一走 `From<T> for Complex<T>`。

2. **数学方法**（需要浮点数学）：对外稳定承诺仅包括 `norm()`、`norm_sqr()`。

Xenon 公开使用 sealed 的 `ComplexFloat` 约束把 `Complex<T>` 封闭到 `f32` / `f64`：

```rust,ignore
/// Public bound for `Complex<T>`.
///
/// This trait is a pure sealing mechanism — it carries no methods.
/// It is sealed via `private::Sealed`, so downstream crates must not
/// implement it. Xenon only supports `f32` and `f64` as complex components.
pub trait ComplexFloat: private::Sealed + Copy + Default + Debug + PartialEq + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{}

impl ComplexFloat for f32 {}
impl ComplexFloat for f64 {}
```

`ComplexFloat` 是用于表达 `Complex<T>` 公开 bound 的 `pub` trait，但其实现范围由 `private::Sealed` 封闭到 `f32`/`f64`；它不是面向下游的公开扩展点。此 trait 实际上是 `Complex<f32>` / `Complex<f64>` 的内部抽象。`ComplexFloat` 不携带任何方法，仅作类型封闭之用。

### 5.3 构造方法

```rust,ignore
impl<T: ComplexFloat> Complex<T> {
    /// Creates a new complex number.
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

// Methods that don't need Float math — available for all T satisfying
// basic arithmetic constraints.
impl<T: ComplexFloat> Complex<T> {
    /// Creates a purely imaginary number (re = 0).
    #[inline]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::default(), im)
    }
}
```

### 5.4 基础方法

```rust,ignore
// Methods that don't need Float math — publicly available without pub(crate) dependency.
impl<T: ComplexFloat> Complex<T> {
    /// Returns the real part.
    #[inline]
    pub fn re(self) -> T { self.re }

    /// Returns the imaginary part.
    #[inline]
    pub fn im(self) -> T { self.im }

    /// Returns the complex conjugate: conj(a + bj) = a - bj.
    ///
    /// **Design note:** `Complex::conj()` is an inherent method returning `Self`
    /// (the same complex type with negated imaginary part). Generic code should
    /// use `Numeric::conjugate()` as the unified trait-level API; `ComplexScalar`
    /// only carries complex-specific read capabilities such as `re`, `im`, and `norm`.
    /// See `03-element.md` §5.2 / §5.4 for the trait layering details.
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Returns true if imaginary part is zero.
    #[inline]
    pub fn is_real(self) -> bool {
        self.im == T::default()
    }

    /// Returns true if real part is zero.
    #[inline]
    pub fn is_imaginary(self) -> bool {
        self.re == T::default()
    }
}

// NaN/finite checks require Float — provided via concrete impls.
impl Complex<f32> {
    /// Returns true if either part is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Returns true if both parts are finite (not NaN and not infinite).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Complex<f64> {
    /// Returns true if either part is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Returns true if both parts are finite (not NaN and not infinite).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}
```

### 5.5 数学方法

数学方法通过具体的 `impl Complex<f32>` 和 `impl Complex<f64>` 块提供，而非泛型 `impl<T: Float>`。

```rust,ignore
// Concrete impl for Complex<f32> — public stable methods.
impl Complex<f32> {
    /// Modulus |z| = sqrt(re² + im²), using hypot to avoid overflow.
    #[inline]
    pub fn norm(self) -> f32 {
        self.re.hypot(self.im)
    }

    /// Squared modulus |z|² = re² + im² (avoids sqrt).
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

// Concrete impl for Complex<f64> — public stable methods.
impl Complex<f64> {
    /// Modulus |z| = sqrt(re² + im²), using hypot to avoid overflow.
    ///
    /// # Example
    /// ```
    /// let z = Complex::new(3.0_f64, 4.0);
    /// assert!((z.norm() - 5.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared modulus |z|² = re² + im² (avoids sqrt).
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}
```

### 5.6 算术运算实现

```rust,ignore
// Complex + Complex: (a+bj) + (c+dj) = (a+c) + (b+d)j
impl<T: ComplexFloat> core::ops::Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

// Complex - Complex: (a+bj) - (c+dj) = (a-c) + (b-d)j
impl<T: ComplexFloat> core::ops::Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

// Complex * Complex: (a+bj) * (c+dj) = (ac-bd) + (ad+bc)j
impl<T: ComplexFloat> core::ops::Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

// Complex / Complex: Smith's algorithm for better numerical stability
// Concrete impls are required because the branch selection uses T::abs(),
// which is not part of ComplexFloat (the trait is a pure sealing mechanism).
impl core::ops::Div for Complex<f32> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self::new(
                (self.re + self.im * r) / denom,
                (self.im - self.re * r) / denom,
            )
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.re * r + rhs.im;
            Self::new(
                (self.re * r + self.im) / denom,
                (self.im * r - self.re) / denom,
            )
        }
    }
}

impl core::ops::Div for Complex<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self::new(
                (self.re + self.im * r) / denom,
                (self.im - self.re * r) / denom,
            )
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.re * r + rhs.im;
            Self::new(
                (self.re * r + self.im) / denom,
                (self.im * r - self.re) / denom,
            )
        }
    }
}

// -Complex: -(a+bj) = -a - bj
impl<T: ComplexFloat> core::ops::Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
```

`Complex<T>` 除法遵循 Smith 算法 + 底层 IEEE 754 标量运算的组合语义。实现层使用 Smith 算法避免直接形成 `c² + d²`；当输入含 `0`、`NaN`、`Inf` 或中间结果出现上溢/下溢时，最终结果继续由底层 `f32` / `f64` 标量运算的 IEEE 754 传播规则决定，不额外返回可恢复错误，也不引入额外 panic。该约束适用于 `Complex<T> / Complex<T>` 公开运算符语义，并与 `需求说明书 §13` / `需求说明书 §28.3` 保持一致。

规范性示例：
- `Complex::new(1.0, 2.0) / Complex::new(0.0, 0.0)`：结果按 Smith 分支中的底层 IEEE 754 标量除法/乘加传播计算，允许产生 `Inf` / `NaN` 组合；文档不额外定义独立错误通道。
- `Complex::new(f64::NAN, 1.0) / Complex::new(3.0, 4.0)`：结果分量遵循 `NaN` 传播，输出可含 `NaN` 分量。
- `Complex::new(f64::INFINITY, 1.0) / Complex::new(3.0, 4.0)`：结果分量遵循 `Inf` 参与的底层 IEEE 754 标量运算语义，可产生 `Inf`、有限值或 `NaN` 组合。

### 5.7 显式实数构造与混合运算边界

```rust,ignore
let z = Complex::new(1.0_f64, 2.0);
let rhs = Complex::from(3.0_f64);
let sum = z + rhs;
```

- 当前版本不提供 `Complex<T> op T` 或 `T op Complex<T>` 便捷运算符。调用方若需要与实数混合运算，必须先通过显式 `From<T> for Complex<T>` 构造把实数提升为同元素类型的复数值，再参与 `Complex<T> op Complex<T>` 运算。
- `From<T> for Complex<T>` 是当前版本**唯一**允许的显式标量构造路径，不存在通过运算符触发的隐式实数到复数转换。张量级场景中，双方元素类型必须预先一致（见 `需求说明书 §5`）；用户须先把实数显式构造成 `Complex<T>`，再参与张量运算。

### 5.8 PartialEq 实现

```rust,ignore
impl<T: ComplexFloat> PartialEq for Complex<T> {
    /// Component-wise equality. NaN != NaN (IEEE 754).
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}
// NOT implementing Eq: NaN violates reflexivity.
// NOT implementing PartialOrd/Ord: complex numbers have no natural total order.
```

### 5.9 格式化输出

```rust,ignore
impl<T: ComplexFloat + core::fmt::Display + PositiveZero> core::fmt::Display for Complex<T> {
    /// Formats as "a+bj", "a-bj", "a", "bj", or "0".
    ///
    /// NaN handling: when the imaginary part is NaN, always display
    /// as "re+NaNj". This deliberately normalizes the textual form instead
    /// of inferring a sign from NaN payload/sign-bit details.
    /// `-0.0` must preserve the underlying scalar Display output, `Inf` / `-Inf`
    /// follow the component scalar formatting, and NaN payload bits are not
    /// rendered textually distinct.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.im != self.im {
            return write!(f, "{}+NaNj", self.re);
        }
        if self.im == T::default() {
            if scalar_is_positive_zero(self.im) {
                write!(f, "{}", self.re)
            } else {
                write!(f, "{}{}j", self.re, self.im)
            }
        } else if self.re == T::default() {
            write!(f, "{}j", self.im)
        } else if self.im > T::default() {
            write!(f, "{}+{}j", self.re, self.im)
        } else {
            write!(f, "{}{}j", self.re, self.im) // negative sign included in im
        }
    }
}

// Concrete f32/f64 helpers distinguish +0.0 from -0.0 via IEEE-754 bit patterns.
trait PositiveZero {
    fn is_positive_zero(self) -> bool;
}

impl PositiveZero for f32 {
    #[inline]
    fn is_positive_zero(self) -> bool {
        self.to_bits() == 0.0f32.to_bits()
    }
}

impl PositiveZero for f64 {
    #[inline]
    fn is_positive_zero(self) -> bool {
        self.to_bits() == 0.0f64.to_bits()
    }
}

#[inline]
fn scalar_is_positive_zero<T: PositiveZero>(im: T) -> bool {
    im.is_positive_zero()
}
```

| 输入                      | Display 输出 |
| ------------------------- | ------------ |
| `Complex::new(3.0, 4.0)`  | `"3+4j"`     |
| `Complex::new(3.0, -4.0)` | `"3-4j"`     |
| `Complex::new(3.0, 0.0)`  | `"3"`        |
| `Complex::new(3.0, -0.0)` | `"3-0j"` |
| `Complex::new(0.0, 4.0)`  | `"4j"`       |
| `Complex::new(0.0, 0.0)`  | `"0"`        |

**稳定性规则**： 

- `-0.0` 必须保留其符号信息。
- 当虚部为 `-0.0` 时，输出须包含虚部表示，不能落入“仅输出实部”的 `im == 0` 折叠路径。
- 实现上须显式区分 `+0.0` 与 `-0.0`。
- `NaN` 的 payload / sign bit 不作为稳定输出契约的一部分。
- `Inf` / `-Inf` 直接沿用分量标量的 `Display` 结果。

### 5.10 类型转换

本节只保留语义矩阵。具体 `From` / `CastTo` 实现由 `convert/cast.rs` 统一承载。

整数到复数的受支持路径按 `需求说明书 §23.1` 与 `需求说明书 §23.2` 的规则补充如下：

| 源类型 | 目标类型 | 语义 | 默认行为 |
|--------|----------|------|---------|
| `i32` | `Complex<f64>` | 实部 `i32→f64` 无损，虚部为 `0` | 成功 |
| `i32` | `Complex<f32>` | 实部 `i32→f32` 有损，虚部为 `0` | 返回可恢复错误 |
| `i64` | `Complex<f64>` | 实部 `i64→f64` 有损，虚部为 `0` | 返回可恢复错误 |
| `i64` | `Complex<f32>` | 实部 `i64→f32` 有损，虚部为 `0` | 返回可恢复错误 |

其中语义遵循 `需求说明书 §23.2` 的闭合规则：先按对应实数类型到目标复数实部分量类型的规则转换实部，再引入值为 `0` 的虚部。当前版本不额外扩展 `需求说明书 §23.1` 之外的整数→复数组合。

- `Complex` 类型的逐元素类型转换统一由 `03-element.md` 定义的 `CastTo<T>` trait 管理，trait 定义位于 `element` 模块，具体实现归入 `convert/` 模块；本节不再单独定义张量级转换入口。`From` 仅用于**不可能失败且不丢失精度**的标量级构造或 widening：`From<T> for Complex<T>`（实数到同精度复数，虚部补 `0`）与 `From<Complex<f32>> for Complex<f64>`（分量无损 widening）。其中 `From<T> for Complex<T>` 是当前版本**唯一**允许的显式实数到复数标量构造路径。
- `CastTo<T>` 直接返回 `XenonError::TypeConversion`。
- 除上述 infallible 构造外，其余显式类型转换统一通过 `CastTo<T>` trait 实现（参见 `03-element.md` §5.9 和 `21-type.md`），包括 `Complex<f64> -> Complex<f32>`、`Complex<T> -> T` 以及其他跨精度/跨类型组合；其中有损窄化路径默认返回可恢复错误。
- 禁止为任何可能失败或可能丢失精度的转换实现 `From`。这类转换必须走 `21-type.md` 定义的 `CastTo<T>`。

复杂到实数的受支持路径同样受 `需求说明书 §23.1` 与 `需求说明书 §23.2` 约束，且统一由 `03-element.md` §5.9 定义的 `CastTo<T>` trait 作为唯一 owner；`complex/` 模块文档仅声明其语义，不重复定义独立转换入口。

| 源类型 | 目标类型 | 语义 | 默认行为 |
|--------|----------|------|---------|
| `Complex<f32>` | `f32` | 仅当虚部为 `0` 时返回实部 | 默认返回 `XenonError::TypeConversion` |
| `Complex<f64>` | `f64` | 仅当虚部为 `0` 时返回实部 | 默认返回 `XenonError::TypeConversion` |
| `Complex<f32>` | `f64` | 仅当虚部为 `0` 时，按 `f32→f64` 规则转换实部 | 默认返回 `XenonError::TypeConversion` |
| `Complex<f64>` | `f32` | 仅当虚部为 `0` 时，按 `f64→f32` 规则转换实部 | 默认返回 `XenonError::TypeConversion` |

- `Complex -> Real` 的具体 `CastTo<T>` 实现同样位于 `convert/cast.rs`。`complex/` 仅保留“虚部必须为 `0`；失败返回 `XenonError::TypeConversion`”这一语义约束，字段模型以 `26-error.md §4.2` / `§4.4` 为准。
- `-0.0` 补充说明： 复数到实数转换对“虚部是否为零”的判断遵循 IEEE 754 比较语义；因此 `-0.0` 视为零，`Complex::new(3.0, -0.0)` 允许按虚部为零的路径继续转换，不应被误判为非零虚部。

### 5.11 内存布局静态断言

```rust,ignore
// Compile-time layout verification
const _: () = {
    assert!(core::mem::size_of::<Complex<f32>>() == 8);
    assert!(core::mem::align_of::<Complex<f32>>() == 4);
    assert!(core::mem::size_of::<Complex<f64>>() == 16);
    assert!(core::mem::align_of::<Complex<f64>>() == 8);
};
```

- `#[repr(C)]` 仅用于固定字段顺序与 C 兼容结构体表示；安全前提建立在按 `re` / `im` 两个已知字段访问，而非依赖“无 padding”假设。
-  除 size/align 静态断言外，测试计划还应补充 `re` 位于偏移 0、`im` 位于 `size_of::<T>()` 偏移处的验证意图，用于防止未来重构破坏两字段 C struct 的约定布局。

### 5.12 FFI 布局兼容性说明

| C 表示                             | 内存布局           | Rust 等价      |
| ---------------------------------- | ------------------ | -------------- |
| `struct { float re; float im; }`   | `[float, float]`   | `Complex<f32>` |
| `struct { double re; double im; }` | `[double, double]` | `Complex<f64>` |

- 公开 FFI 契约只保证 `#[repr(C)] struct { re: T, im: T }` 等价布局。
- `Complex<T>` 在 Rust 侧始终保持双字段 `#[repr(C)]` 布局，作为最低层内存表示；

FFI 示例：

```rust,ignore
// Rust side
#[no_mangle]
pub unsafe extern "C" fn process_complex(
    z: *const Complex<f64>,
    out: *mut Complex<f64>,
) -> i32 {
    if z.is_null() || out.is_null() {
        return -1;
    }

    // SAFETY: the C caller must pass non-null, properly aligned pointers to
    // valid `Complex<f64>` objects that are live for the duration of this call.
    unsafe {
        *out = *z * *z;
    }
    0
}

// C side
// typedef struct { double re; double im; } complex64_t;
// int32_t process_complex(const complex64_t* z, complex64_t* out);
```

### 5.13 Good / Bad 对比示例

```rust,ignore
// Good - same precision arithmetic, type safe
let z = Complex::new(1.0_f64, 2.0);
let w = Complex::new(3.0, 4.0);
let sum = z + w;          // Complex<f64>
let scaled = z * Complex::from(2.0_f64); // Complex<f64>, explicit conversion first

// Bad - cross-precision mixed arithmetic (compile error)
let z = Complex::new(1.0_f64, 2.0);
// let bad = z + 3.0_f32;  // Compile error: type mismatch
// let also_bad = z + 3.0_f64; // Compile error: no Complex<f64> + f64 convenience impl

// Good - explicit cross-precision conversion
let z32 = Complex::new(1.0_f32, 2.0);
let z64: Complex<f64> = z32.into(); // Explicit upcast
let result = z64 + Complex::from(3.0_f64); // Explicit real-to-complex construction
```

```rust,ignore
// Good - use norm() for modulus (hypot prevents overflow)
let big = Complex::new(1e200_f64, 1e200);
let n = big.norm(); // 1.414...e200, safe

// Bad - manual calculation may overflow
let overflow = (big.re * big.re + big.im * big.im).sqrt(); // Inf!
```

---

## 6. 内部实现设计

### 6.1 算法描述

**Complex × Complex 公式**:

```
(a+bj)(c+dj) = ac + adj + bcj + bdj²
             = ac + adj + bcj - bd    [j² = -1]
             = (ac-bd) + (ad+bc)j
```

**Complex ÷ Complex 公式**（Smith's algorithm，避免直接形成 `c² + d²`）:

```
if |c| >= |d|:
    r = d / c
    denom = c + d * r
    result_re = (a + b * r) / denom
    result_im = (b - a * r) / denom
else:
    r = c / d
    denom = c * r + d
    result_re = (a * r + b) / denom
    result_im = (b * r - a) / denom
```

优势：避免在极大或极小值输入下直接计算 `c² + d²`，降低中间溢出、下溢与灾难性消去风险。

**norm() hypot 算法**:

```
hypot(a, b):
    a = |a|, b = |b|
    if a < b: swap(a, b)
    if a == 0: return b
    ratio = b / a
    return a * sqrt(1 + ratio²)
```

优势：中间结果不超过 `2 * max(|a|, |b|)`，避免 `a²` 溢出。

### 6.2 不支持的运算清单

| 运算                             | 原因                                |
| -------------------------------- | ----------------------------------- |
| `Complex<f64> + f64`             | 公开 API 不提供 `Complex<T> op T`，须先显式构造 `Complex::from(…)` |
| `Complex<f64> + f32`             | 跨精度，且公开 API 不提供 `Complex<T> op T` 便捷运算符             |
| `Complex<f64> + i32`             | 整数与复数，须先显式转换并构造匹配的 `Complex<T>`                  |
| `impl Eq for Complex<T>`         | NaN 违反自反性                      |
| `impl Ord for Complex<T>`        | 复数无自然全序                      |
| `impl PartialOrd for Complex<T>` | 字典序无数学意义                    |
| Serde 序列化                     | 不在当前范围（参见 `需求说明书 §2.2`） |

---

## 7. 任务拆分

### Wave 1: 核心定义

- [ ] **T1**: 定义 `Complex<T>` 结构体和 `new()` 构造方法
  - 文件: `src/complex/mod.rs`
  - 内容: 结构体定义 + `new()` + `ComplexFloat` trait 定义 + 对 `f32`/`f64` 的实现
  - 测试: `test_complex_new`, 编译通过
  - 前置: private 模块已经就绪
  - 预计: 10 min

- [ ] **T2**: 实现内存布局静态断言
  - 文件: `src/complex/mod.rs`
  - 内容: `const _` 断言 `size_of`/`align_of`
  - 测试: `test_complex_layout_f32`, `test_complex_layout_f64`
  - 前置: T1
  - 预计: 5 min

### Wave 2: 基础方法

- [ ] **T3**: 实现基础访问方法和构造辅助
  - 文件: `src/complex/mod.rs`
  - 内容: `re()`, `im()`, `from_imag()`, `conj()`, `is_real()`, `is_imaginary()`；纯实数构造统一使用 `From<T> for Complex<T>`
  - 测试: `test_conj`, `test_is_real`, `test_from_imag`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `PartialEq` + `Display`
  - 文件: `src/complex/mod.rs`
  - 内容: `PartialEq` impl（NaN!=NaN）、`Display` impl（a+bj 格式）
  - 测试: `test_eq`, `test_nan_neq`, `test_display_format`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 算术运算

- [ ] **T5**: 实现 Complex ± Complex 运算符
  - 文件: `src/complex/ops.rs`
  - 内容: `Add<Complex<T>> for Complex<T>`, `Sub<Complex<T>> for Complex<T>`
  - 测试: `test_add_complex`, `test_sub_complex`
  - 前置: T1
  - 预计: 10 min

- [ ] **T6**: 实现 Complex × Complex, Complex ÷ Complex 运算符
  - 文件: `src/complex/ops.rs`
  - 内容: `Mul<Complex<T>> for Complex<T>`, `Div<Complex<T>> for Complex<T>`, `Neg`
  - 测试: `test_mul_complex`, `test_div_complex`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7**: 收紧实数与复数混合运算边界
  - 文件: `src/complex/ops.rs`
  - 内容: 仅保留 `Complex<T> op Complex<T>`，并补充 `Complex::from(real)` 的显式转换用法示例
  - 测试: `test_real_to_complex_add`
  - 前置: T5, T6
  - 预计: 10 min

### Wave 4: 数学方法

- [ ] **T8**: 实现稳定数学方法 `norm`, `norm_sqr`
  - 文件: `src/complex/mod.rs`
  - 内容: `norm()`（hypot）, `norm_sqr()`
  - 测试: `test_norm_3_4_5`, `test_norm_no_overflow`
  - 前置: T1
  - 预计: 10 min

### Wave 5: 类型转换与集成

- [ ] **T9**: 实现类型转换
  - 文件: `src/convert/cast.rs`
  - 内容: `element` 模块定义 `CastTo<T>` trait；`convert/cast.rs` 承载复数相关转换实现，包括 `From<Complex<f32>> for Complex<f64>`、`From<f32> for Complex<f32>`、`From<f64> for Complex<f64>` 与统一的窄化转换
  - 测试: `test_f32_to_f64_lossless`, `test_f64_to_f32_precision_loss`, `test_real_to_complex`
  - 前置: T1
  - 预计: 10 min

- [ ] **T10**: 文档注释与 `cargo doc` 验证
  - 文件: 所有 `src/complex/` 文件 + `src/convert/cast.rs`
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T8, T9
  - 预计: 10 min

- [ ] **T11**: 集成测试与边界测试
  - 文件: `tests/test_complex.rs`
  - 内容: 完整测试计划（见 §8）
  - 测试: 见测试计划
  - 前置: T10
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                      | 说明                                     |
| -------- | ------------------------- | ---------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`  | 验证复数结构、运算、格式化与布局         |
| 集成测试 | `tests/test_complex.rs`   | 验证 `complex` 与 `element`、`math`、`matrix`、`ffi` 的协同路径 |
| 边界测试 | 同模块测试中标注          | 覆盖 NaN/Inf、极大/极小值与 FFI 布局前提 |
| 属性测试 | `tests/property_tests.rs` | 验证共轭、模长不变量                         |

### 8.2 单元测试清单

| 测试函数                         | 测试内容                                                    | 优先级 |
| -------------------------------- | ----------------------------------------------------------- | ------ |
| `test_complex_new`               | `Complex::new(3.0, 4.0).re == 3.0`                          | 高     |
| `test_complex_layout_f32`        | `size_of::<Complex<f32>>() == 8`                            | 高     |
| `test_complex_layout_f64`        | `size_of::<Complex<f64>>() == 16`                           | 高     |
| `test_from_imag`                 | `from_imag(3.0).re == 0.0`                                  | 高     |
| `test_conj`                      | `Complex::new(3.0, 4.0).conj() == Complex::new(3.0, -4.0)`  | 高     |
| `test_is_real_imaginary`         | `Complex::from(1.0).is_real()`, `from_imag(1.0).is_imaginary()` | 中 |
| `test_add_complex`               | `(1+2j) + (3+4j) == (4+6j)`                                 | 高     |
| `test_sub_complex`               | `(5+7j) - (2+3j) == (3+4j)`                                 | 高     |
| `test_mul_complex`               | `(1+2j) * (3+4j) == (-5+10j)`                               | 高     |
| `test_div_complex`               | 使用容差断言验证 `(6+8j) / (3+4j)` 约等于 `(2+0j)`          | 高     |
| `test_div_zero_propagates_ieee754` | `(1+2j) / (0+0j)` 遵循 Smith 算法 + IEEE 754 标量传播语义，且不返回错误/额外 panic | 高     |
| `test_neg_complex`               | `-(1+2j) == (-1-2j)`                                        | 高     |
| `test_real_to_complex_add`       | `Complex::from(3.0) + (1+2j) == (4+2j)`                     | 高     |
| `test_norm_3_4_5`                | `Complex::new(3.0, 4.0).norm() == 5.0`                      | 高     |
| `test_norm_no_overflow`          | `Complex::new(1e200, 1e200).norm()` 不溢出                  | 高     |
| `test_norm_sqr`                  | `norm_sqr() == re² + im²`                                   | 中     |
| `test_eq_nan`                    | `Complex::new(NaN, 0.0) != self`                            | 高     |
| `test_display_format`            | `"3+4j"`, `"3-4j"`, `"3-0j"`, `"3"`, `"4j"`, `"0"`          | 中     |
| `test_f32_to_f64_lossless`       | `Complex<f32>→Complex<f64>` 无损                            | 高     |
| `test_f64_to_f32_precision_loss` | `Complex<f64>→Complex<f32>` 精度降低                        | 中     |
| `test_real_to_complex`           | `f64→Complex<f64>` 虚部为 0                                 | 高     |

推荐将 `test_div_complex` 写成显式容差断言，而不是直接对浮点结果做 `==` 比较：

```rust,ignore
let result = Complex::new(6.0_f64, 8.0) / Complex::new(3.0_f64, 4.0);
assert!((result.re - 2.0).abs() < 1e-10 && result.im.abs() < 1e-10);
```

### 8.3 边界测试场景

| 场景                          | 预期行为                                                 |
| ----------------------------- | -------------------------------------------------------- |
| 零 `Complex::new(0.0, 0.0)`   | `norm()==0`                                                  |
| NaN 参与                      | `Complex::new(NaN, 0.0).norm().is_nan()`                 |
| Inf 参与                      | `Complex::new(Inf, 1.0).norm().is_finite()==false`          |
| 极大值 norm                   | `Complex::new(1e200, 1e200).norm()` 不溢出（≈1.414e200） |
| 极小值 norm                   | `Complex::new(1e-200, 1e-200).norm()` 正确               |
| `Complex::new(1.0, 2.0) / Complex::new(0.0, 0.0)` | 遵循 Smith 算法 + IEEE 754 标量传播语义，不返回可恢复错误，也不额外 panic |
| 连续字段布局                  | `Complex<f64>` 的 `re/im` 字段顺序稳定，可逐元素读取     |

### 8.4 属性测试不变量

| 不变量                                                           | 测试方法                                |
| ---------------------------------------------------------------- | --------------------------------------- |
| `(z * z.conj()).re == z.norm_sqr()` 且 `(z * z.conj()).im == 0`  | 随机 z                                  |
| `(z / w) * w ≈ z`                                                | 随机 z, w（w ≠ 0）                      |

### 8.5 集成测试

| 测试文件                | 测试内容                                              |
| ----------------------- | ----------------------------------------------------- |
| `tests/test_complex.rs` | 复数类型与 `element` trait 体系、`math` 逐元素运算、`matrix` 共轭内积以及 `ffi` 布局约束的端到端验证 |

### 8.6 Feature gate / 配置测试

| 配置项 | 覆盖方式                 | 说明                                         |
| ------ | ------------------------ | -------------------------------------------- |
| 默认配置 | 常规单元/集成测试路径  | 本模块无独立 feature gate，默认配置即主路径  |
| 非默认 feature | 不适用           | 本模块未定义 feature gate，故无额外配置矩阵  |

### 8.7 类型边界 / 编译期测试

| 测试类型 | 覆盖方式                                         | 说明                                    |
| -------- | ------------------------------------------------ | --------------------------------------- |
| sealed 边界 | 编译期验证 `ComplexFloat` 仅允许 `f32` / `f64`    | 防止公开 API 暴露任意 `T` 的实例化  |
| 语义边界 | compile-fail 测试 `Eq` / `Ord` / `PartialOrd` 未实现 | 验证 NaN 与复数序语义边界不被破坏   |
| 精度边界 | compile-fail 测试 `Complex<T> op T` 与跨精度混合运算 | 验证公开 API 仅接受预先同类型化后的 `Complex<T> op Complex<T>` |

---

## 9. 与其他模块的交互

### 9.1 与 element 模块的集成

| 交互点     | 说明                                                                                               |
| ---------- | -------------------------------------------------------------------------------------------------- |
| 类型定义   | `Complex<T>` 定义在 `crate::complex`                                                               |
| Trait 实现 | `Element`/`Numeric`/`ComplexScalar` 在 `element` 模块定义，在 `primitives.rs` 为 `Complex<T>` 实现 |
| 依赖方向   | `element` 依赖 `complex`（类型定义）；`complex` 不依赖 `element`                                   |

### 9.2 数据流描述

```
User constructs `Complex<f64>::new(re, im)`
    │
    ├── complex/ provides core methods, arithmetic, and scalar-level conversion semantics
    │
    ├── element/ implements Element/Numeric/ComplexScalar for `Complex<f64>`
    │
    ├── convert/ consumes `CastTo` and hosts tensor-level conversion implementations
    │
    ├── math / matrix / format and other upper layers consume these traits and inherent methods
    │
    └── FFI paths interoperate via a two-field C struct layout; `_Complex` ABI compatibility is not promised
```

---

## 10. 错误处理与语义边界

| 项目              | 内容                                                                                |
| ----------------- | ----------------------------------------------------------------------------------- |
| Recoverable error | `CastTo<T>` trait 级的有损窄化路径返回 `XenonError::TypeConversion`                 |
| Panic             | 常规复数运算与方法不以 panic 作为错误通道；若调用底层标准库浮点 API，遵循其既有语义 |
| 路径一致性        | scalar 路径与普通标量实现必须一致；SIMD：不适用；parallel：不适用                   |
| 容差边界          | 复数数值测试采用显式容差；布局、格式化与类型边界测试不适用                          |

---

## 11. 设计决策记录

### 决策 1：自定义 vs num-complex

| 属性     | 值                                                                                  |
| -------- | ----------------------------------------------------------------------------------- |
| 决策     | 自定义 `Complex<T>`，不依赖 `num-complex`                                           |
| 理由     | 零额外依赖；可精确控制 trait 实现；严格同精度互操作；与 Element 体系无缝集成        |
| 替代方案 | 使用 `num-complex` — 放弃，引入 num-traits 传递依赖，且 trait surface 需要额外适配  |
| 后果     | 需自行实现 `norm`/`norm_sqr` 等基础数学方法；高阶数学函数（如 `exp`/`ln`/`sqrt`）留待后续模块按需引入；获得 API 完全控制权                           |

### 决策 2：不实现 Eq/Ord

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | 不实现 `Eq`、`PartialOrd`、`Ord`                                                                     |
| 理由     | `Eq` 要求自反性（x==x），但 NaN!=NaN 违反此性质；复数无自然全序；实现 Eq 会导致 HashSet 等未定义行为 |
| 替代方案 | 实现 Eq — 放弃，NaN 导致语义错误                                                                     |
| 替代方案 | 实现 PartialOrd（字典序）— 放弃，无数学意义                                                          |

### 决策 3：norm() 使用 hypot 而非 sqrt(re²+im²)

| 属性     | 值                                                                 |
| -------- | ------------------------------------------------------------------ |
| 决策     | 使用 `hypot(re, im)` 计算模长                                      |
| 理由     | 数值稳定：当 re/im 很大时 `re*re` 可能溢出，hypot 使用缩放算法避免 |
| 替代方案 | `sqrt(re*re + im*im)` — 放弃，大数溢出                             |

### 决策 4：不实现复合赋值运算符

| 属性     | 值                                                                                      |
| -------- | --------------------------------------------------------------------------------------- |
| 决策     | 当前版本不实现 `AddAssign`/`SubAssign`/`MulAssign`/`DivAssign`                          |
| 理由     | `需求说明书 §20` 指出"所有组合均产生新的独立张量"；运算符重载仅产生新张量，不需要原地修改 |
| 替代方案 | 实现全部复合赋值 — 放弃，超出当前需求                                                   |

---

## 12. 性能考量

| 方面         | 设计决策                                                               |
| ------------ | ---------------------------------------------------------------------- |
| `#[repr(C)]` | 保证字段顺序稳定，便于与双字段 C 结构体做布局验证                      |
| 内联         | 所有运算方法标注 `#[inline]`，编译器可内联                             |
| 零堆分配     | `Complex<T>` 为 `Copy` 类型，全部栈分配                                |
| 单态化       | `Complex<f32>` 和 `Complex<f64>` 各自单态化，无虚调用                  |
| hypot 开销   | `norm()` 的 hypot 比直接 sqrt 多几次比较和分支，但避免溢出，开销可接受 |

---

## 13. 平台与工程约束

| 约束       | 说明                                    |
| ---------- | --------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std`  |
| MSRV       | 最低 Rust 版本要求为 1.85+              |
| 单 crate   | 保持单 crate 边界                       |
| SemVer     | 公开 Complex 类型及算术 API 遵循 SemVer |
| 最小依赖   | 无新增第三方依赖                        |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-14 |
| 1.1.2 | 2026-04-15 |
| 1.1.3 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
