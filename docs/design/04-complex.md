# 复数类型模块设计

> 文档编号: 04 | 模块: `src/complex/` | 阶段: Phase 1
> 前置文档: `00-coding.md`, `01-architecture.md`
> 需求参考: 需求说明书 §4, §5, §23, §24, §25, §28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责              | 包含                                                                                              | 不包含                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 类型定义          | `Complex<T>` 结构体（`#[repr(C)]`，re/im 字段）                                                   | —                                                                                           |
| 构造方法          | `new(re, im)`                                                                                     | 额外公开构造器（如 `from_polar`）                                                           |
| 基础方法          | `re()`, `im()`, `conj()`, `is_real()`, `is_imaginary()`                                           | —                                                                                           |
| 数学方法          | `norm()`（hypot）, `norm_sqr()`, `to_polar()`                                                     | 将 `arg` / `exp` / `ln` / `sqrt` 暴露为公开稳定 API、复数 FFT、高阶复数运算                |
| 算术运算          | Complex±Complex, Complex×Complex, Complex÷Complex, 一元负号                                       | 跨精度混合运算                                                                              |
| 实数混合运算      | 同精度标量便捷：`Complex<f32> op f32`、`Complex<f64> op f64`；张量级运算前须先显式转为 `Complex<T>` | 跨精度：`f32+Complex<f64>`（须显式转换）；将该类标量便捷 impl 直接外推为张量级重载            |
| 格式化输出        | Display（`"a+bj"` / `"a-bj"`）, Debug                                                             | —                                                                                           |
| 双字段 C 布局基础 | `#[repr(C)]` + 编译期静态断言                                                                     | 跨精度混合运算                                                                              |
| 类型转换          | `Complex<f32>↔Complex<f64>`, `f32/f64→Complex`, `i32/i64→Complex`                                 | 未在 `require.md` §23.1 列出的额外整数→复数组合                                               |

### 1.2 设计原则

| 原则         | 体现                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| FFI 友好     | `#[repr(C)]` 保证字段顺序稳定，并为与两字段 C 结构体互操作提供布局基础 |
| 零依赖       | 不引入任何外部 crate（不使用 num-complex）                             |
| 同精度互操作 | 仅支持 `f32↔Complex<f32>` 和 `f64↔Complex<f64>`                        |
| 数值稳定     | `norm()` 使用 hypot 算法避免中间溢出                                   |
| NaN 语义正确 | `NaN != NaN`，不实现 Eq/Ord                                            |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: complex  ← current module
L2: layout (depends on dimension)
L3: storage (depends only on core/alloc)
L4: tensor (depends on storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

### 1.4 自定义实现 vs num-complex

| 考量         | 自定义实现                                       | num-complex                |
| ------------ | ------------------------------------------------ | -------------------------- |
| FFI 布局基础 | `#[repr(C)]` 为双字段 C 结构体互操作提供布局基础 | 同样支持，但需验证         |
| 依赖控制     | 零额外依赖，符合最小依赖原则                     | 引入 num-traits 等传递依赖 |
| API 精确控制 | 可精确控制 trait 实现（禁止 Eq/Ord）             | trait surface 更宽，需额外裁剪/适配 |
| 精度约束     | 严格限制同精度互操作                             | 支持更宽松的混合精度       |
| Element 集成 | 无缝集成到 Xenon 元素类型体系                    | 需要额外适配层             |

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                            |
| -------- | --------------------------------------------------------------- |
| 需求映射 | 需求说明书 §4、§5、§23、§24、§25、§28                           |
| 范围内   | `Complex<T>` 类型定义、同精度复数算术、`conj()` / `norm()` / `norm_sqr()` / `to_polar()`、格式化、类型转换、FFI 布局边界 |
| 范围外   | `num-complex` 兼容层、跨精度混合运算、FFT 与高阶复数算法、额外公开复数数学函数 |
| 非目标   | 引入第三方复数库依赖、开放任意 `T` 的公开实例化、把内部数学 helper 稳定化，或 `_Complex` ABI 保证 |

> **当前版本范围声明**：本版本保留 `Complex<T> op T`（右侧实数）的**标量级便捷实现**与 `Complex<T> op Complex<T>` 运算。前者仅用于单个复数值与同精度实数值的便捷组合，**不直接适用于张量逐元素运算或 `19-overload.md` 的运算符重载设计**。根据 `require.md` §5，涉及实数与复数的张量逐元素运算、运算符重载及相关 API 组合时，参与运算双方的元素类型须预先一致；用户须先将实数显式转换为 `Complex<T>`，再进入张量级运算。左侧实数运算（`T op Complex<T>`）仍不在当前版本范围内，若后续版本需要支持，须单独设计并继续满足上述同类型前提。

> **公开 API 收紧**：`arg` / `exp` / `ln` / `sqrt` / `from_polar` / `i` 可作为 `complex/` 内部实现辅助能力存在，用于支撑 `to_polar()`、测试或后续上层模块实现，但**不作为当前版本对下游承诺的公开稳定 API**。本文档中的对应算法仅描述内部实现方案。

---

## 3. 文件位置

```
src/complex/
├── mod.rs     # Complex<T> definition, basic methods, math methods, PartialEq, Display, layout checks
├── ops.rs     # Arithmetic operator impls (Add/Sub/Mul/Div/Neg + real/complex mixed ops)

src/convert/
└── cast.rs    # CastTo-based conversion implementations, including Complex-related paths
```

文件职责按模块 owner 划分：`complex/` 负责类型定义与运算；`CastTo` trait 定义位于 `element` 模块；复数相关的转换实现归入 `convert/` 模块（参见 `21-type.md`）。

> **转换归属说明**：`CastTo` trait 定义位于 `element` 模块；复数相关的转换实现参见 `21-type.md`（`convert` 模块）。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/complex/
├── crate::error     # XenonError for recoverable conversion failures
├── crate::private   # Sealed trait for ComplexFloat
├── core::ops        # Add/Sub/Mul/Div/Neg operator traits
├── core::fmt        # Debug/Display formatting
└── core::cmp        # PartialEq
```

> **零外部依赖。** `Complex<T>` 不引入任何第三方 crate；项目内依赖仅包括 `crate::error`（用于 `CastTo<T>` 窄化等可恢复错误）和 `crate::private`（用于 sealed 的 `ComplexFloat` 约束），其余基础结构、方法和运算依赖 `core`。涉及浮点数学的复数方法运行在 Xenon 的 `std` 环境前提下。

### 4.2 依赖精确到类型级

| 来源        | 使用的类型/trait                              |
| ----------- | --------------------------------------------- |
| `error`     | `XenonError`（`CastTo<T>` 窄化等可恢复错误） |
| `private`   | `Sealed`（封闭 `ComplexFloat` 的公开实现范围） |
| `core::ops` | `Add`, `Sub`, `Mul`, `Div`, `Neg`（算术运算） |
| `core::fmt` | `Debug`, `Display`（格式化输出）              |
| `core::cmp` | `PartialEq`（相等比较）                       |

### 4.2a 依赖合法性

| 项目           | 结论                           |
| -------------- | ------------------------------ |
| 新增第三方依赖 | 无                             |
| 合法性结论     | 符合需求说明书最小依赖限制     |
| 替代方案       | 不适用                         |

### 4.3 依赖方向声明

> **依赖方向：核心内聚、单向向下。** `complex/` 仅依赖项目内基础模块 `error` 与 `private`，不依赖其他上层业务模块。
> 被下游消费：`element/` 模块为 `Complex<f32>`/`Complex<f64>` 实现 Element/Numeric/ComplexScalar trait（参见 `03-element.md` §5.1 / §5.2 / §5.4）。

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

> **公开边界约束：** `Complex<T>` 保持公开泛型形式，但其公开 bound 使用 `pub trait ComplexFloat`。
> 该 trait 通过 `private::Sealed` 超 trait 封闭实现范围，因此下游只能使用 `Complex<f32>` 与
> `Complex<f64>`，不能为其他类型自行实现 `ComplexFloat`。

### 5.2 泛型约束

`Complex<T>` 的方法分为两类：

1. **基础方法**（无需浮点数学）：在 `impl<T: ComplexFloat> Complex<T>` 中实现，公开可用。包括 `re()`、`im()`、`conj()`、`from_real()`、`from_imag()`、`is_real()`、`is_imaginary()`。

2. **数学方法**（需要浮点数学）：对外稳定承诺仅包括 `norm()`、`norm_sqr()`、`to_polar()`；`arg_impl()`、`exp_impl()`、`ln_impl()`、`sqrt_impl()`、`from_polar_impl()` 等仅作为 `complex/` 内部 helper，由具体的 `impl Complex<f32>` / `impl Complex<f64>` 或私有模块承载。

Xenon 公开使用 sealed 的 `ComplexFloat` 约束把 `Complex<T>` 封闭到 `f32` / `f64`；内部仍可定义 `Float` trait 绑定必要数学方法，作为实现细节：

```rust,ignore
/// Public bound for `Complex<T>`.
///
/// This trait is sealed via `private::Sealed`, so downstream crates must not
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

> **API 边界说明**：`ComplexFloat` 是用于表达 `Complex<T>` 公开 bound 的 `pub` trait，但其实现范围由 `private::Sealed` 封闭到 `f32`/`f64`；它不是面向下游的公开扩展点。`Float` 则保持内部实现细节。公开能力仍主要通过 `Element`/`Numeric`/`ComplexScalar` 等更高层 trait 暴露。

内部实现细节：

```rust,ignore
/// Internal trait for floating-point types used in Complex<T>.
/// Only f32 and f64 implement this.
/// NOT part of the public API — math methods are exposed via concrete impl blocks.
pub(crate) trait Float:
    Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
}

impl Float for f32 { /* delegates to inherent methods */ }
impl Float for f64 { /* delegates to inherent methods */ }
```

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
    /// Creates a purely real number (im = 0).
    #[inline]
    pub fn from_real(re: T) -> Self {
        Self::new(re, T::default())
    }

    /// Creates a purely imaginary number (re = 0).
    #[inline]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::default(), im)
    }
}

// Internal helper implementations for f32.
impl Complex<f32> {
    /// Internal constructor from polar coordinates: r * (cos(theta) + j*sin(theta)).
    #[inline]
    pub(crate) fn from_polar_impl(r: f32, theta: f32) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Internal imaginary-unit helper (f32 specialization).
    #[inline]
    pub(crate) fn i_impl() -> Self {
        Self::new(0.0, 1.0)
    }
}

// Internal helper implementations for f64.
impl Complex<f64> {
    /// Internal constructor from polar coordinates: r * (cos(theta) + j*sin(theta)).
    #[inline]
    pub(crate) fn from_polar_impl(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Internal imaginary-unit helper (f64 specialization).
    #[inline]
    pub(crate) fn i_impl() -> Self {
        Self::new(0.0, 1.0)
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
    /// (the same complex type with negated imaginary part). It differs from
    /// `Numeric::conjugate()` which is a trait method on the `Numeric` trait.
    /// Both produce the same mathematical result for complex types, but
    /// `Numeric::conjugate()` is the intended API for generic code constrained
    /// only on `Numeric`. See `03-element.md` §5.2 for details.
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

> **注意**：数学方法通过具体的 `impl Complex<f32>` 和 `impl Complex<f64>` 块提供，而非泛型 `impl<T: Float>`。这避免了 `Float` trait（`pub(crate)`）暴露到公共 API。

````rust,ignore
// Concrete impl for Complex<f32> — public stable methods plus internal helpers.
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

    /// Returns polar coordinates (r, theta).
    #[inline]
    pub fn to_polar(self) -> (f32, f32) {
        (self.norm(), self.arg_impl())
    }
}

impl Complex<f32> {
    /// Internal argument helper: atan2(im, re). Range: (-π, π].
    #[inline]
    pub(crate) fn arg_impl(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Internal complex exponential: e^z = e^re * (cos(im) + j*sin(im)).
    #[inline]
    pub(crate) fn exp_impl(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }

    /// Internal complex natural logarithm (principal value): ln|z| + j*arg(z).
    #[inline]
    pub(crate) fn ln_impl(self) -> Self {
        Self::new(self.norm().ln(), self.arg_impl())
    }

    /// Internal complex square root (principal value, real part >= 0).
    #[inline]
    pub(crate) fn sqrt_impl(self) -> Self {
        let r = self.norm();
        if r == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let half = 0.5;
        let re_part = ((r + self.re) * half).sqrt();
        let im_part = ((r - self.re) * half).sqrt();
        let im_sign = if self.im >= 0.0 { im_part } else { -im_part };
        Self::new(re_part, im_sign)
    }
}

// Concrete impl for Complex<f64> — public stable methods plus internal helpers.
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

    /// Returns polar coordinates (r, theta).
    #[inline]
    pub fn to_polar(self) -> (f64, f64) {
        (self.norm(), self.arg_impl())
    }
}

impl Complex<f64> {
    /// Internal argument helper: atan2(im, re). Range: (-π, π].
    #[inline]
    pub(crate) fn arg_impl(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Internal complex exponential: e^z = e^re * (cos(im) + j*sin(im)).
    #[inline]
    pub(crate) fn exp_impl(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }

    /// Internal complex natural logarithm (principal value): ln|z| + j*arg(z).
    #[inline]
    pub(crate) fn ln_impl(self) -> Self {
        Self::new(self.norm().ln(), self.arg_impl())
    }

    /// Internal complex square root (principal value, real part >= 0).
    #[inline]
    pub(crate) fn sqrt_impl(self) -> Self {
        let r = self.norm();
        if r == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let half = 0.5;
        let re_part = ((r + self.re) * half).sqrt();
        let im_part = ((r - self.re) * half).sqrt();
        let im_sign = if self.im >= 0.0 { im_part } else { -im_part };
        Self::new(re_part, im_sign)
    }
}
````

> **精度说明：** 对于实部为负的复数，标准 sqrt 算法可能在分支割线附近产生灾难性消去（catastrophic cancellation）。这是已知的精度限制。对精度要求极高的场景，可考虑使用更稳定的数值算法。

### 5.6 算术运算实现

```rust,ignore
// Complex + Complex: (a+bj) + (c+dj) = (a+c) + (b+d)j
impl<T: Float> core::ops::Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

// Complex - Complex: (a+bj) - (c+dj) = (a-c) + (b-d)j
impl<T: Float> core::ops::Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

// Complex * Complex: (a+bj) * (c+dj) = (ac-bd) + (ad+bc)j
impl<T: Float> core::ops::Mul for Complex<T> {
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
impl<T: Float> core::ops::Div for Complex<T> {
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
impl<T: Float> core::ops::Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
```

### 5.7 混合运算（同精度实数与复数）

```rust,ignore
// Complex + T: (a+bj) + r = (a+r) + bj
impl<T: Float> core::ops::Add<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self { Self::new(self.re + rhs, self.im) }
}

// Complex * T: (a+bj) * r = ar + brj
impl<T: Float> core::ops::Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self { Self::new(self.re * rhs, self.im * rhs) }
}

// Complex / T: (a+bj) / r = (a/r) + (b/r)j
impl<T: Float> core::ops::Div<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self { Self::new(self.re / rhs, self.im / rhs) }
}

// Complex - T
impl<T: Float> core::ops::Sub<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self { Self::new(self.re - rhs, self.im) }
}
```

> **设计决策：** 当前版本仅支持同精度的 `Complex<T> op T`。左侧实数运算 `T op Complex<T>` 暂不纳入稳定 API，调用方若需要该语义，应先显式执行 `Complex::from(real)` 再参与复数运算。`Complex<f64> + f32` 这类跨精度混合运算仍然编译错误，须显式转换。

> **边界说明：** 这些 `Complex<T> op T` impl 仅为标量级便捷能力，**不属于张量级逐元素运算或 `19-overload.md` 运算符重载设计的稳定承诺**。张量级场景中，双方元素类型必须预先一致（见 `require.md` §5）；用户须先把实数显式转换为 `Complex<T>`，再参与张量运算。

### 5.8 PartialEq 实现

```rust,ignore
impl<T: Float> PartialEq for Complex<T> {
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
impl<T: Float + core::fmt::Display> core::fmt::Display for Complex<T> {
    /// Formats as "a+bj", "a-bj", "a", "bj", or "0".
    ///
    /// NaN handling: when the imaginary part is NaN, always display
    /// as "re+NaNj". This deliberately normalizes the textual form instead
    /// of inferring a sign from NaN payload/sign-bit details.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.im.is_nan() {
            return write!(f, "{}+NaNj", self.re);
        }
        if self.im == T::zero() {
            write!(f, "{}", self.re)
        } else if self.re == T::zero() {
            write!(f, "{}j", self.im)
        } else if self.im > T::zero() {
            write!(f, "{}+{}j", self.re, self.im)
        } else {
            write!(f, "{}{}j", self.re, self.im) // negative sign included in im
        }
    }
}
```

| 输入                      | Display 输出 |
| ------------------------- | ------------ |
| `Complex::new(3.0, 4.0)`  | `"3+4j"`     |
| `Complex::new(3.0, -4.0)` | `"3-4j"`     |
| `Complex::new(3.0, 0.0)`  | `"3"`        |
| `Complex::new(0.0, 4.0)`  | `"4j"`       |
| `Complex::new(0.0, 0.0)`  | `"0"`        |

### 5.10 类型转换

```rust,ignore
// Precision promotion: f32 -> f64 (lossless)
// This is the lossless direction — follows std's philosophy that From
// should only be implemented for lossless conversions.
impl From<Complex<f32>> for Complex<f64> {
    #[inline]
    fn from(z: Complex<f32>) -> Self {
        // Lossless: f32 → f64 preserves all bits
        Self::new(f64::from(z.re), f64::from(z.im))
    }
}

// Precision reduction: f64 -> f32 (lossy)
// NOT implemented as `From` — lossy conversion contradicts std's philosophy.
// Complex-to-complex narrowing is unified under `CastTo<T>` in `03-element.md`
// and implemented under the convert module umbrella (see `21-type.md`).
impl CastTo<Complex<f32>> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>, XenonError> {
        Ok(Complex::<f32>::new(
            CastTo::<f32>::cast_to(self.re)?,
            CastTo::<f32>::cast_to(self.im)?,
        ))
    }
}

// Real -> Complex (same precision)
impl From<f32> for Complex<f32> {
    #[inline]
    fn from(re: f32) -> Self { Self::new(re, 0.0) }
}

impl From<f64> for Complex<f64> {
    #[inline]
    fn from(re: f64) -> Self { Self::new(re, 0.0) }
}
```

整数到复数的受支持路径按 `require.md` §23.1 与 §23.2 的规则补充如下：

| 源类型 | 目标类型 | 语义 | 默认行为 |
|--------|----------|------|---------|
| `i32` | `Complex<f64>` | 实部 `i32→f64` 无损，虚部为 `0` | 成功 |
| `i32` | `Complex<f32>` | 实部 `i32→f32` 有损，虚部为 `0` | 返回可恢复错误 |
| `i64` | `Complex<f64>` | 实部 `i64→f64` 有损，虚部为 `0` | 返回可恢复错误 |
| `i64` | `Complex<f32>` | 实部 `i64→f32` 有损，虚部为 `0` | 返回可恢复错误 |

其中语义遵循 `require.md` §23.2 的闭合规则：先按对应实数类型到目标复数实部分量类型的规则转换实部，再引入值为 `0` 的虚部。当前版本不额外扩展 `require.md` §23.1 之外的整数→复数组合。

> **统一转换入口说明：** `Complex` 类型的逐元素类型转换统一由 `03-element.md` 定义的 `CastTo<T>` trait 管理，trait 定义位于 `element` 模块，具体实现归入 `convert/` 模块；本节不再单独定义张量级转换入口；本模块仅保留复数类型自身固有、且符合无损语义的 `From`/`Into` 标量级转换（如 `f32 -> Complex<f32>`、`Complex<f32> -> Complex<f64>`）。

复杂到实数的受支持路径同样受 `require.md` §23.1 与 §23.2 约束，且统一由 `03-element.md` §5.9 定义的 `CastTo<T>` trait 作为唯一 owner；`complex/` 模块文档仅声明其语义，不重复定义独立转换入口。

| 源类型 | 目标类型 | 语义 | 默认行为 |
|--------|----------|------|---------|
| `Complex<f32>` | `f32` | 仅当虚部为 `0` 时返回实部 | 虚部非零时返回 `XenonError::TypeConversion { ... }` |
| `Complex<f64>` | `f64` | 仅当虚部为 `0` 时返回实部 | 虚部非零时返回 `XenonError::TypeConversion { ... }` |
| `Complex<f32>` | `f64` | 仅当虚部为 `0` 时，按 `f32→f64` 规则转换实部 | 虚部非零时返回 `XenonError::TypeConversion { ... }` |
| `Complex<f64>` | `f32` | 仅当虚部为 `0` 时，按 `f64→f32` 规则转换实部 | 虚部非零时返回 `XenonError::TypeConversion { ... }` |

```rust,ignore
// Complex -> Real conversions are owned by CastTo<T> in src/element/.
// complex/ documents the rule here to keep the conversion matrix complete.

impl CastTo<f64> for Complex<f64> {
    fn cast_to(self) -> Result<f64, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion {
                source_type: "Complex<f64>".into(),
                target_type: "f64".into(),
                reason: "non-zero imaginary part".into(),
                element_index: None,
            });
        }
        Ok(self.re)
    }
}

impl CastTo<f32> for Complex<f64> {
    fn cast_to(self) -> Result<f32, XenonError> {
        if self.im != 0.0 {
            return Err(XenonError::TypeConversion {
                source_type: "Complex<f64>".into(),
                target_type: "f32".into(),
                reason: "non-zero imaginary part".into(),
                element_index: None,
            });
        }
        CastTo::<f32>::cast_to(self.re)
    }
}
```

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

**安全性论证**: `#[repr(C)]` 仅用于固定字段顺序与 C 兼容结构体表示；安全前提建立在按 `re` / `im` 两个已知字段访问，而非依赖“无 padding”假设。

### 5.12 FFI 布局兼容性说明

| C 表示                             | 内存布局           | Rust 等价      |
| ---------------------------------- | ------------------ | -------------- |
| `struct { float re; float im; }`   | `[float, float]`   | `Complex<f32>` |
| `struct { double re; double im; }` | `[double, double]` | `Complex<f64>` |

Xenon 对需求说明书 §5 的裁决是：**公开 FFI 契约只保证 `#[repr(C)] struct { re: T, im: T }` 等价布局**。

1. `Complex<T>` 在 Rust 侧始终保持双字段 `#[repr(C)]` 布局，作为最低层内存表示；
2. 对外文档与测试只承诺其可按两字段 C 结构体解释；
3. **不保证** 与 C99 `_Complex` 的 ABI 或调用约定兼容，相关互操作若有需要，应由上游按目标平台单独验证并适配。

FFI 示例：

```rust,ignore
// Rust side
#[no_mangle]
pub extern "C" fn process_complex(z: *const Complex<f64>) -> Complex<f64> {
    // SAFETY: the C caller must pass a non-null, properly aligned pointer to a
    // valid `Complex<f64>` object that is live for the duration of this call.
    unsafe { *z * *z }
}

// C side
// typedef struct { double re; double im; } complex64_t;
// complex64_t process_complex(const complex64_t* z);
```

### 5.13 Good / Bad 对比示例

```rust,ignore
// Good - same precision arithmetic, type safe
let z = Complex::new(1.0_f64, 2.0);
let w = Complex::new(3.0, 4.0);
let sum = z + w;          // Complex<f64>
let scaled = z * 2.0_f64; // Complex<f64>, same precision

// Bad - cross-precision mixed arithmetic (compile error)
let z = Complex::new(1.0_f64, 2.0);
// let bad = z + 3.0_f32;  // Compile error: type mismatch

// Good - explicit cross-precision conversion
let z32 = Complex::new(1.0_f32, 2.0);
let z64: Complex<f64> = z32.into(); // Explicit upcast
let result = z64 + 3.0_f64;         // Now same precision
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
| `Complex<f64> + f32`             | 跨精度，须显式转换                  |
| `Complex<f64> + i32`             | 整数与复数，须先转浮点              |
| `impl Eq for Complex<T>`         | NaN 违反自反性                      |
| `impl Ord for Complex<T>`        | 复数无自然全序                      |
| `impl PartialOrd for Complex<T>` | 字典序无数学意义                    |
| Serde 序列化                     | 不在当前范围（参见需求说明书 §2.2） |

---

## 7. 任务拆分

### Wave 1: 核心定义

- [ ] **T1**: 定义 `Complex<T>` 结构体和 `new()` 构造方法
  - 文件: `src/complex/mod.rs`
  - 内容: 结构体定义 + `new()` + `Float` trait 定义 + `impl Float for f32/f64`
  - 测试: `test_complex_new`, 编译通过
  - 前置: 无
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
  - 内容: `re()`, `im()`, `from_real()`, `from_imag()`, `conj()`, `is_real()`, `is_imaginary()`；`i` 仅保留为内部 helper
  - 测试: `test_conj`, `test_is_real`, `test_from_real`
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

- [ ] **T7**: 实现 Complex 与同精度实数混合运算
  - 文件: `src/complex/ops.rs`
  - 内容: `Complex±T`, `Complex*T`, `Complex/T`，并补充 `Complex::from(real)` 的显式转换用法示例
  - 测试: `test_add_real`, `test_mul_real`, `test_div_real`
  - 前置: T5, T6
  - 预计: 10 min

### Wave 4: 数学方法

- [ ] **T8**: 实现稳定数学方法 `norm`, `norm_sqr`, `to_polar`
  - 文件: `src/complex/mod.rs`
  - 内容: `norm()`（hypot）, `norm_sqr()`, `to_polar()`；内部复用 `arg_impl()`
  - 测试: `test_norm_3_4_5`, `test_norm_no_overflow`, `test_arg_impl_range`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9**: 实现内部复数数学 helper
  - 文件: `src/complex/mod.rs`
  - 内容: `arg_impl()`, `exp_impl()`, `ln_impl()`, `sqrt_impl()`, `from_polar_impl()`, `i_impl()`
  - 测试: `test_exp_impl_ln_impl_inverse`, `test_sqrt_impl_neg_one`, `test_from_polar_impl_i`（作为内部回归测试）
  - 前置: T8
  - 预计: 10 min

### Wave 5: 类型转换与集成

- [ ] **T10**: 实现类型转换
  - 文件: `src/convert/cast.rs`
  - 内容: `element` 模块定义 `CastTo<T>` trait；`convert/cast.rs` 承载复数相关转换实现，包括 `From<Complex<f32>> for Complex<f64>`、`From<f32> for Complex<f32>`、`From<f64> for Complex<f64>` 与统一的窄化转换
  - 测试: `test_f32_to_f64_lossless`, `test_f64_to_f32_precision_loss`, `test_real_to_complex`
  - 前置: T1
  - 预计: 10 min

- [ ] **T11**: 文档注释与 `cargo doc` 验证
  - 文件: 所有 `src/complex/` 文件 + `src/convert/cast.rs`
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T9, T10
  - 预计: 10 min

- [ ] **T12**: 集成测试与边界测试
  - 文件: `tests/test_complex.rs`
  - 内容: 完整测试计划（见 §8）
  - 测试: 见测试计划
  - 前置: T11
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
          │
          ├──────────┬──────────┐
          ▼          ▼          ▼
Wave 2: [T3]      [T4]      [T5, T6]
                                │
                                ▼
                             [T7]
          │
          ▼
Wave 3: [T8] → [T9]

Wave 4: [T10]
          │
          ▼
Wave 5: [T11] → [T12]
```


---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                           | 说明                                                            |
| -------- | ---------------------------------------------- | --------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`                       | 验证复数结构、运算、格式化与布局                                |
| 集成测试 | `tests/test_complex.rs`                        | 验证 `complex` 与 `element`、`math`、`matrix`、`ffi` 的协同路径 |
| 边界测试 | 同模块测试中标注                               | 覆盖 NaN/Inf、极大/极小值与 FFI 布局前提                        |
| 属性测试 | `tests/test_complex.rs` 或 `tests/property.rs` | 验证共轭、模长、指数对数与极坐标不变量                          |

### 8.2 单元测试清单

| 测试函数                         | 测试内容                                                    | 优先级 |
| -------------------------------- | ----------------------------------------------------------- | ------ |
| `test_complex_new`               | `Complex::new(3.0, 4.0).re == 3.0`                          | 高     |
| `test_complex_layout_f32`        | `size_of::<Complex<f32>>() == 8`                            | 高     |
| `test_complex_layout_f64`        | `size_of::<Complex<f64>>() == 16`                           | 高     |
| `test_from_real_imag`            | `from_real(5.0).im == 0.0`, `from_imag(3.0).re == 0.0`      | 高     |
| `test_conj`                      | `Complex::new(3.0, 4.0).conj() == Complex::new(3.0, -4.0)`  | 高     |
| `test_is_real_imaginary`         | `from_real(1.0).is_real()`, `from_imag(1.0).is_imaginary()` | 中     |
| `test_add_complex`               | `(1+2j) + (3+4j) == (4+6j)`                                 | 高     |
| `test_sub_complex`               | `(5+7j) - (2+3j) == (3+4j)`                                 | 高     |
| `test_mul_complex`               | `(1+2j) * (3+4j) == (-5+10j)`                               | 高     |
| `test_div_complex`               | 使用容差断言验证 `(6+8j) / (3+4j)` 约等于 `(2+0j)`         | 高     |
| `test_neg_complex`               | `-(1+2j) == (-1-2j)`                                        | 高     |
| `test_add_real`                  | `(1+2j) + 3.0 == (4+2j)`                                    | 高     |
| `test_real_to_complex_add`       | `Complex::from(3.0) + (1+2j) == (4+2j)`                     | 高     |
| `test_mul_real`                  | `(1+2j) * 3.0 == (3+6j)`                                    | 高     |
| `test_div_by_real`               | `(6+4j) / 2.0 == (3+2j)`                                    | 高     |
| `test_real_to_complex_div`       | `Complex::from(5.0) / (3+4j)` 正确                          | 中     |
| `test_norm_3_4_5`                | `Complex::new(3.0, 4.0).norm() == 5.0`                      | 高     |
| `test_norm_no_overflow`          | `Complex::new(1e200, 1e200).norm()` 不溢出                  | 高     |
| `test_norm_sqr`                  | `norm_sqr() == re² + im²`                                   | 中     |
| `test_arg_impl_range`            | `arg_impl()` 在 `(-π, π]` 范围内                            | 高     |
| `test_exp_impl_ln_impl_inverse`  | `ln_impl(z).exp_impl() ≈ z`（约束：`z ≠ 0` 且避开主值分支割线附近） | 高 |
| `test_sqrt_impl_neg_one`         | `Complex::new(-1.0, 0.0).sqrt_impl() ≈ j`                   | 高     |
| `test_from_polar_impl_i`         | `from_polar_impl(1.0, π/2) ≈ j`                             | 中     |
| `test_eq_nan`                    | `Complex::new(NaN, 0.0) != self`                            | 高     |
| `test_display_format`            | `"3+4j"`, `"3-4j"`, `"3"`, `"4j"`, `"0"`                    | 中     |
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
| 零 `Complex::new(0.0, 0.0)`   | `norm()==0`, `arg_impl()==0`, `sqrt_impl()==0`（内部回归测试） |
| `Complex::new(0.0, 0.0).ln_impl()` | 返回 `-∞ + 0j`（内部回归测试）                      |
| NaN 参与                      | `Complex::new(NaN, 0.0).norm().is_nan()`                 |
| Inf 参与                      | `Complex::new(Inf, 0.0).exp_impl()` 正确处理（内部回归测试） |
| 极大值 norm                   | `Complex::new(1e200, 1e200).norm()` 不溢出（≈1.414e200） |
| 极小值 norm                   | `Complex::new(1e-200, 1e-200).norm()` 正确               |
| 连续字段布局                  | `Complex<f64>` 的 `re/im` 字段顺序稳定，可逐元素读取     |

### 8.4 属性测试不变量

| 不变量                                                             | 测试方法                                |
| ------------------------------------------------------------------ | --------------------------------------- |
| `(z * z.conj()).re == z.norm_sqr()` 且 `(z * z.conj()).im == 0`    | 随机 z                                  |
| `ln_impl(z).exp_impl() ≈ z`                                        | 随机有限 z（`z ≠ 0`，且避开分支割线附近） |
| `z.sqrt_impl() * z.sqrt_impl() ≈ z`                                | 随机 z                                  |
| `(z / w) * w ≈ z`                                                  | 随机 z, w（w ≠ 0）                      |
| `Complex::from_polar_impl(z.norm(), z.arg_impl()) ≈ z`             | 随机 z（作为内部 helper 回归测试）      |

### 8.5 集成测试

| 测试文件                | 测试内容                                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `tests/test_complex.rs` | 复数类型与 `element` trait 体系、`math` 逐元素运算、`matrix` 共轭内积以及 `ffi` 布局约束的端到端验证 |

### 8.6 Feature gate / 配置测试

| 配置项 | 覆盖方式                             | 说明                                         |
| ------ | ------------------------------------ | -------------------------------------------- |
| 默认配置 | 常规单元/集成测试路径                 | 本模块无独立 feature gate，默认配置即主路径  |
| 非默认 feature | 不适用                             | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.7 类型边界 / 编译期测试

| 测试类型 | 覆盖方式                                         | 说明                                                    |
| -------- | ------------------------------------------------ | ------------------------------------------------------- |
| sealed 边界 | 编译期验证 `ComplexFloat` 仅允许 `f32` / `f64`     | 防止公开 API 暴露任意 `T` 的实例化                      |
| 语义边界 | compile-fail 测试 `Eq` / `Ord` / `PartialOrd` 未实现 | 验证 NaN 与复数序语义边界不被破坏                       |
| 精度边界 | compile-fail 测试跨精度混合运算                     | 验证仅同精度混合运算进入稳定 API                         |

---

## 9. 模块交互

### 9.1 与 element 模块的集成

| 交互点     | 说明                                                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 类型定义   | `Complex<T>` 定义在 `crate::complex`                                                                                            |
| Trait 实现 | `Element`/`Numeric`/`ComplexScalar` 在 `element` 模块定义（参见 `03-element.md` §5.1 / §5.2 / §5.4），在 `primitives.rs` 为 `Complex<T>` 实现 |
| 依赖方向   | `element` 依赖 `complex`（类型定义）；`complex` 不依赖 `element`                                                                |

### 9.2 接口边界

```
┌───────────────────────────────────────────────────────────────┐
│  element (Element/Numeric/ComplexScalar trait impls)          │
└───────────────────────┬───────────────────────────────────────┘
                        │ type dependency (Complex<T> definition)
┌───────────────────────▼───────────────────────────────────────┐
│  complex (Complex<T> definition, arithmetic, scalar conversions) │
└───────────────────────────────────────────────────────────────┘
```

### 9.3 数据流描述

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

| 项目           | 内容 |
| -------------- | ---- |
| Recoverable error | `CastTo<T>` 承载的有损窄化路径返回可恢复错误；公开错误类型统一为 `XenonError::TypeConversion { source_type, target_type, reason, element_index }` |
| Panic | 本模块常规复数运算与方法不以 panic 作为错误通道；若调用底层标准库浮点 API，遵循其既有语义 |
| 路径一致性 | scalar 路径与普通标量实现必须一致；SIMD：不适用；parallel：不适用 |
| 容差边界 | 复数数值测试采用显式容差；布局、格式化与类型边界测试不适用 |

---

## 11. 设计决策记录

### 决策 1：自定义 vs num-complex

| 属性     | 值                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------- |
| 决策     | 自定义 `Complex<T>`，不依赖 `num-complex`                                                                   |
| 理由     | 零额外依赖；可精确控制 trait 实现（禁止 Eq/Ord）；严格同精度互操作；FFI 布局可验证；与 Element 体系无缝集成 |
| 替代方案 | 使用 `num-complex` — 放弃，引入 num-traits 传递依赖，且 trait surface 需要额外适配                          |
| 后果     | 需自行实现所有数学方法；增加维护成本；获得 API 完全控制权                                                   |

### 决策 2：不实现 Eq/Ord

| 属性     | 值                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------- |
| 决策     | 不实现 `Eq`、`PartialOrd`、`Ord`                                                                     |
| 理由     | `Eq` 要求自反性（x==x），但 NaN!=NaN 违反此性质；复数无自然全序；实现 Eq 会导致 HashSet 等未定义行为 |
| 替代方案 | 实现 Eq — 放弃，NaN 导致语义错误                                                                  |
| 替代方案 | 实现 PartialOrd（字典序）— 放弃，无数学意义                                                          |

### 决策 3：norm() 使用 hypot 而非 sqrt(re²+im²)

| 属性     | 值                                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------- |
| 决策     | 使用 `hypot(re, im)` 计算模长                                                                                       |
| 理由     | 数值稳定：当 re/im 很大时 `re*re` 可能溢出，hypot 使用缩放算法避免；标准库 `f32::hypot`/`f64::hypot` 已实现稳定算法 |
| 替代方案 | `sqrt(re*re + im*im)` — 放弃，大数溢出                                                                              |

### 决策 4：不实现复合赋值运算符

| 属性     | 值                                                                                      |
| -------- | --------------------------------------------------------------------------------------- |
| 决策     | 当前版本不实现 `AddAssign`/`SubAssign`/`MulAssign`/`DivAssign`                          |
| 理由     | 需求说明书 §20 指出"所有组合均产生新的独立张量"；运算符重载仅产生新张量，不需要原地修改 |
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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
