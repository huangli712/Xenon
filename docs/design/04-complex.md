# 复数类型模块设计

> 文档编号: 04 | 模块: `src/complex/` | 阶段: Phase 1
> 前置文档: `00-coding.md`, `01-architecture.md`
> 需求参考: 需求说明书 §5

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 类型定义 | `Complex<T>` 结构体（`#[repr(C)]`，re/im 字段） | — |
| 构造方法 | `new(re, im)`, `from_polar(r, theta)` | — |
| 基础方法 | `re()`, `im()`, `conj()`, `is_real()`, `is_imaginary()` | — |
| 数学方法 | `norm()`（hypot）, `arg()`, `exp()`, `ln()`, `sqrt()` | 复数 FFT、高阶复数运算 |
| 算术运算 | Complex±Complex, Complex×Complex, Complex÷Complex, 一元负号 | 跨精度混合运算 |
| 实数混合运算 | 同精度：`f32±Complex<f32>`, `f64±Complex<f64>` | 跨精度：`f32+Complex<f64>`（须显式转换） |
| 格式化输出 | Display（`"a+bi"` / `"a-bi"`）, Debug | — |
| C99 兼容布局 | `#[repr(C)]` + 编译期静态断言 | 跨精度混合运算 |
| 类型转换 | `Complex<f32>↔Complex<f64>`, `f32/f64→Complex` | 整数→复数（须显式先转浮点） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| FFI 友好 | `#[repr(C)]` 保证与 C99 `_Complex` 布局兼容 |
| 零依赖 | 不引入任何外部 crate（不使用 num-complex） |
| 同精度互操作 | 仅支持 `f32↔Complex<f32>` 和 `f64↔Complex<f64>` |
| 数值稳定 | `norm()` 使用 hypot 算法避免中间溢出 |
| NaN 语义正确 | `NaN != NaN`，不实现 Eq/Ord |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: complex  ← 当前模块
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

### 1.4 自定义实现 vs num-complex

| 考量 | 自定义实现 | num-complex |
|------|-----------|-------------|
| FFI 兼容性 | `#[repr(C)]` 保证 C99 `_Complex` 兼容 | 同样支持，但需验证 |
| 依赖控制 | 零额外依赖，符合最小依赖原则 | 引入 num-traits 等传递依赖 |
| API 精确控制 | 可精确控制 trait 实现（禁止 Eq/Ord） | 实现了 Eq，与需求不符 |
| 精度约束 | 严格限制同精度互操作 | 支持更宽松的混合精度 |
| Element 集成 | 无缝集成到 Xenon 元素类型体系 | 需要额外适配层 |

---

## 2. 文件位置

```
src/complex/
├── mod.rs     # Complex<T> 定义、基础方法、数学方法、PartialEq、Display、布局断言
├── ops.rs     # 算术运算符实现（Add/Sub/Mul/Div/Neg + 实数混合运算）
└── cast.rs    # 类型转换（Complex<f32>↔Complex<f64>、实数→复数）
```

三文件设计：核心定义、运算、类型转换职责分离。运算和转换高度独立，可并行开发。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
src/complex/
├── core::ops        # Add/Sub/Mul/Div/Neg 运算符 trait
├── core::fmt        # Debug/Display 格式化
└── core::cmp        # PartialEq
```

> **零外部依赖。** `Complex<T>` 的全部实现仅依赖 `core`，天然 no_std 兼容。

### 3.2 依赖精确到类型级

| 来源 | 使用的类型/trait |
|------|-----------------|
| `core::overload` | `Add`, `Sub`, `Mul`, `Div`, `Neg`（算术运算） |
| `core::fmt` | `Debug`, `Display`（格式化输出） |
| `core::cmp` | `PartialEq`（相等比较） |

### 3.3 依赖方向声明

> **依赖方向：单向向下。** `complex/` 不依赖项目中任何其他模块。
> 被下游消费：`element/` 模块为 `Complex<f32>`/`Complex<f64>` 实现 Element/Numeric/ComplexScalar trait（参见 `03-element.md` §5.3）。

---

## 4. 公共 API 设计

### 4.1 Complex<T> 完整定义

```rust
/// Complex number: a + bi.
///
/// # Memory layout
///
/// - `#[repr(C)]` guarantees fields are laid out in declaration order
/// - Size: `2 * size_of::<T>()`
/// - Alignment: `align_of::<T>()`
/// - Compatible with C99 `_Complex T`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}
```

### 4.2 泛型约束

`Complex<T>` 的方法分为两类：

1. **基础方法**（无需浮点数学）：在 `impl<T: Copy + PartialEq + Default + core::overload::Neg<Output=T>> Complex<T>` 中实现，公开可用。包括 `re()`、`im()`、`conj()`、`from_real()`、`from_imag()`、`is_real()`、`is_imaginary()`。

2. **数学方法**（需要浮点数学）：在具体的 `impl Complex<f32>` 和 `impl Complex<f64>` 块中实现，公开可用。包括 `norm()`、`norm_sqr()`、`arg()`、`exp()`、`ln()`、`sqrt()`、`to_polar()`、`from_polar()`。

Xenon 内部定义 `Float` trait 绑定必要数学方法，仅作为内部实现细节：

```rust
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
    + core::overload::Add<Output = Self>
    + core::overload::Sub<Output = Self>
    + core::overload::Mul<Output = Self>
    + core::overload::Div<Output = Self>
    + core::overload::Neg<Output = Self>
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

### 4.3 构造方法

```rust
impl<T> Complex<T> {
    /// Creates a new complex number.
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

// Methods that don't need Float math — available for all T satisfying
// basic arithmetic constraints.
impl<T: Copy + PartialEq + Default + core::overload::Neg<Output = T>> Complex<T> {
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

// Concrete implementations for f32 — public API, no pub(crate) dependency
impl Complex<f32> {
    /// Creates from polar coordinates: r * (cos(theta) + i*sin(theta)).
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Imaginary unit i (f32 specialization).
    #[inline]
    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }
}

// Concrete implementations for f64 — public API, no pub(crate) dependency
impl Complex<f64> {
    /// Creates from polar coordinates: r * (cos(theta) + i*sin(theta)).
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// Imaginary unit i (f64 specialization).
    #[inline]
    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }
}
```

### 4.4 基础方法

```rust
// Methods that don't need Float math — publicly available without pub(crate) dependency.
impl<T: Copy + PartialEq + Default + core::overload::Neg<Output = T>> Complex<T> {
    /// Returns the real part.
    #[inline]
    pub fn re(self) -> T { self.re }

    /// Returns the imaginary part.
    #[inline]
    pub fn im(self) -> T { self.im }

    /// Returns the complex conjugate: conj(a + bi) = a - bi.
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

### 4.5 数学方法

> **注意**：数学方法通过具体的 `impl Complex<f32>` 和 `impl Complex<f64>` 块提供，而非泛型 `impl<T: Float>`。这避免了 `Float` trait（`pub(crate)`）暴露到公共 API。

```rust
// Concrete impl for Complex<f32> — all methods are public.
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

    /// Argument (phase angle): atan2(im, re). Range: (-π, π].
    #[inline]
    pub fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Complex exponential: e^z = e^re * (cos(im) + i*sin(im)).
    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }

    /// Complex natural logarithm (principal value): ln|z| + i*arg(z).
    #[inline]
    pub fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.arg())
    }

    /// Complex square root (principal value, real part >= 0).
    #[inline]
    pub fn sqrt(self) -> Self {
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

    /// Converts to polar coordinates (r, theta).
    #[inline]
    pub fn to_polar(self) -> (f32, f32) {
        (self.norm(), self.arg())
    }
}

// Concrete impl for Complex<f64> — all methods are public.
// Same logic as f32, with f64 types.
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

    /// Argument (phase angle): atan2(im, re). Range: (-π, π].
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex exponential: e^z = e^re * (cos(im) + i*sin(im)).
    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }

    /// Complex natural logarithm (principal value): ln|z| + i*arg(z).
    #[inline]
    pub fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.arg())
    }

    /// Complex square root (principal value, real part >= 0).
    #[inline]
    pub fn sqrt(self) -> Self {
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

    /// Converts to polar coordinates (r, theta).
    #[inline]
    pub fn to_polar(self) -> (f64, f64) {
        (self.norm(), self.arg())
    }
}
```

> **精度说明：** 对于实部为负的复数，标准 sqrt 算法可能在分支割线附近产生灾难性消去（catastrophic cancellation）。这是已知的精度限制。对精度要求极高的场景，可考虑使用更稳定的数值算法。

### 4.6 算术运算实现

```rust
// Complex + Complex: (a+bi) + (c+di) = (a+c) + (b+d)i
impl<T: Float> core::overload::Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

// Complex - Complex: (a+bi) - (c+di) = (a-c) + (b-d)i
impl<T: Float> core::overload::Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

// Complex * Complex: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
impl<T: Float> core::overload::Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

// Complex / Complex: multiply by conjugate of denominator
// (a+bi) / (c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)
impl<T: Float> core::overload::Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

// -Complex: -(a+bi) = -a - bi
impl<T: Float> core::overload::Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
```

### 4.7 混合运算（同精度实数与复数）

```rust
// Complex + T: (a+bi) + r = (a+r) + bi
impl<T: Float> core::overload::Add<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self { Self::new(self.re + rhs, self.im) }
}

// T + Complex: r + (a+bi) = (r+a) + bi
impl<T: Float> core::overload::Add<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output { Complex::new(self + rhs.re, rhs.im) }
}

// Complex * T: (a+bi) * r = ar + bri
impl<T: Float> core::overload::Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self { Self::new(self.re * rhs, self.im * rhs) }
}

// T * Complex: r * (a+bi) = ar + bri
impl<T: Float> core::overload::Mul<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output { Complex::new(self * rhs.re, self * rhs.im) }
}

// Complex / T: (a+bi) / r = (a/r) + (b/r)i
impl<T: Float> core::overload::Div<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self { Self::new(self.re / rhs, self.im / rhs) }
}

// T / Complex: r / (a+bi) = r(a-bi) / (a²+b²)
impl<T: Float> core::overload::Div<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Complex::new(self * rhs.re / denom, -self * rhs.im / denom)
    }
}

// Complex - T and T - Complex (similar pattern)
impl<T: Float> core::overload::Sub<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self { Self::new(self.re - rhs, self.im) }
}

impl<T: Float> core::overload::Sub<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output { Complex::new(self - rhs.re, -rhs.im) }
}
```

> **设计决策：** 仅支持同精度混合运算。`Complex<f64> + f32` 编译错误。跨精度须显式转换。

### 4.8 PartialEq 实现

```rust
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

### 4.9 格式化输出

```rust
impl<T: Float + core::fmt::Display> core::fmt::Display for Complex<T> {
    /// Formats as "a+bi", "a-bi", "a", "bi", or "0".
    ///
    /// NaN handling: when the imaginary part is NaN, explicitly display
    /// as "re+NaNi" (or "re-NaNi") to avoid the ambiguity of NaN comparison
    /// in the `im > T::zero()` branch.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.im.is_nan() {
            return write!(f, "{}+NaNi", self.re);
        }
        if self.im == T::zero() {
            write!(f, "{}", self.re)
        } else if self.re == T::zero() {
            write!(f, "{}i", self.im)
        } else if self.im > T::zero() {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im) // negative sign included in im
        }
    }
}
```

| 输入 | Display 输出 |
|------|-------------|
| `Complex::new(3.0, 4.0)` | `"3+4i"` |
| `Complex::new(3.0, -4.0)` | `"3-4i"` |
| `Complex::new(3.0, 0.0)` | `"3"` |
| `Complex::new(0.0, 4.0)` | `"4i"` |
| `Complex::new(0.0, 0.0)` | `"0"` |

### 4.10 类型转换

```rust
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
// Use the named method `to_f32()` instead.
impl Complex<f64> {
    /// Converts to `Complex<f32>` with possible precision loss.
    ///
    /// Uses IEEE 754 round-to-nearest-even for each component.
    /// This is the only permitted lossy numeric conversion.
    #[inline]
    pub fn to_f32(self) -> Complex<f32> {
        Complex::new(self.re as f32, self.im as f32)
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

### 4.11 内存布局静态断言

```rust
// Compile-time layout verification
const _: () = {
    assert!(core::mem::size_of::<Complex<f32>>() == 8);
    assert!(core::mem::align_of::<Complex<f32>>() == 4);
    assert!(core::mem::size_of::<Complex<f64>>() == 16);
    assert!(core::mem::align_of::<Complex<f64>>() == 8);
};
```

**安全性论证**: `#[repr(C)]` 确保 `re` 和 `im` 字段连续排列且无 padding。从首字段地址读取 2 个 T 值是安全的。

### 4.12 C99 兼容性验证

| C 类型 | 内存布局 | Rust 等价 |
|--------|----------|-----------|
| `_Complex float` | `[float, float]` | `Complex<f32>` |
| `_Complex double` | `[double, double]` | `Complex<f64>` |

FFI 示例：

```rust
// Rust side
#[no_mangle]
pub extern "C" fn process_complex(z: *const Complex<f64>) -> Complex<f64> {
    unsafe { *z * *z }
}

// C side
// #include <complex.h>
// double _Complex process_complex(const double _Complex* z);
```

### 4.13 Good / Bad 对比示例

```rust
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

```rust
// Good - use norm() for modulus (hypot prevents overflow)
let big = Complex::new(1e200_f64, 1e200);
let n = big.norm(); // 1.414...e200, safe

// Bad - manual calculation may overflow
let overflow = (big.re * big.re + big.im * big.im).sqrt(); // Inf!
```

---

## 5. 内部实现设计

### 5.1 算法描述

**Complex × Complex 公式**:

```
(a+bi)(c+di) = ac + adi + bci + bdi²
             = ac + adi + bci - bd    [i² = -1]
             = (ac-bd) + (ad+bc)i
```

**Complex ÷ Complex 公式**（乘以分母共轭）:

```
(a+bi)     (a+bi)(c-di)     (ac+bd) + (bc-ad)i
--------  =  -------------  =  ----------------------
(c+di)     (c+di)(c-di)           c² + d²
```

**norm() hypot 算法**:

```
hypot(a, b):
    a = |a|, b = |b|
    if a > b: swap(a, b)
    if a == 0: return b
    ratio = b / a
    return a * sqrt(1 + ratio²)
```

优势：中间结果不超过 `2 * max(|a|, |b|)`，避免 `a²` 溢出。

### 5.2 不支持的运算清单

| 运算 | 原因 |
|------|------|
| `Complex<f64> + f32` | 跨精度，须显式转换 |
| `Complex<f64> + i32` | 整数与复数，须先转浮点 |
| `impl Eq for Complex<T>` | NaN 违反自反性 |
| `impl Ord for Complex<T>` | 复数无自然全序 |
| `impl PartialOrd for Complex<T>` | 字典序无数学意义 |
| Serde 序列化 | 不在当前范围（参见需求说明书 §2.2） |

---

## 6. 与其他模块的交互

### 6.1 与 element 模块的集成

| 交互点 | 说明 |
|--------|------|
| 类型定义 | `Complex<T>` 定义在 `crate::complex` |
| Trait 实现 | `Element`/`Numeric`/`ComplexScalar` 在 `element` 模块定义（参见 `03-element.md` §4.4），在 `primitives.rs` 为 `Complex<T>` 实现 |
| 依赖方向 | `element` 依赖 `complex`（类型定义）；`complex` 不依赖 `element` |

### 6.2 接口边界

```
┌───────────────────────────────────────────────────────────────┐
│  element (Element/Numeric/ComplexScalar trait implementations)│
└───────────────────────┬───────────────────────────────────────┘
                        │ 类型依赖（Complex<T> 定义）
┌───────────────────────▼───────────────────────────────────────┐
│  complex (Complex<T> 定义，算术运算，类型转换)                   │
└───────────────────────────────────────────────────────────────┘
```

---

## 7. 实现任务拆分

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
  - 内容: `re()`, `im()`, `from_real()`, `from_imag()`, `i()`, `conj()`, `is_real()`, `is_imaginary()`
  - 测试: `test_conj`, `test_is_real`, `test_from_real`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `PartialEq` + `Display`
  - 文件: `src/complex/mod.rs`
  - 内容: `PartialEq` impl（NaN!=NaN）、`Display` impl（a+bi 格式）
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

- [ ] **T7**: 实现 Complex 与实数混合运算（8 种组合）
  - 文件: `src/complex/ops.rs`
  - 内容: `Complex±T`, `T±Complex`, `Complex*T`, `T*Complex`, `Complex/T`, `T/Complex`, `Complex-T`, `T-Complex`
  - 测试: `test_add_real`, `test_mul_real`, `test_div_real`
  - 前置: T5, T6
  - 预计: 10 min

### Wave 4: 数学方法

- [ ] **T8**: 实现数学方法 `conj`, `norm`, `norm_sqr`, `arg`
  - 文件: `src/complex/mod.rs`
  - 内容: `norm()`（hypot）, `norm_sqr()`, `arg()`（atan2）, `to_polar()`
  - 测试: `test_norm_3_4_5`, `test_norm_no_overflow`, `test_arg_range`
  - 前置: T1
  - 预计: 10 min

- [ ] **T9**: 实现数学方法 `exp`, `ln`, `sqrt`, `from_polar`
  - 文件: `src/complex/mod.rs`
  - 内容: `exp()`, `ln()`, `sqrt()`, `from_polar()`
  - 测试: `test_exp_ln_inverse`, `test_sqrt_neg_one`, `test_from_polar_i`
  - 前置: T8
  - 预计: 10 min

### Wave 5: 类型转换与集成

- [ ] **T10**: 实现类型转换
  - 文件: `src/complex/cast.rs`
  - 内容: `From<Complex<f32>> for Complex<f64>`, `From<f32> for Complex<f32>`, `From<f64> for Complex<f64>`, `Complex<f64>::to_f32()` 方法
  - 测试: `test_f32_to_f64_lossless`, `test_f64_to_f32_precision_loss`, `test_real_to_complex`
  - 前置: T1
  - 预计: 10 min

- [ ] **T11**: 文档注释与 `cargo doc` 验证
  - 文件: 所有 `src/complex/` 文件
  - 内容: 所有 pub 项添加文档注释
  - 测试: `cargo doc` 无警告
  - 前置: T9, T10
  - 预计: 10 min

- [ ] **T12**: 集成测试与边界测试
  - 文件: `tests/complex_tests.rs`
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

### 8.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_complex_new` | `Complex::new(3.0, 4.0).re == 3.0` | 高 |
| `test_complex_layout_f32` | `size_of::<Complex<f32>>() == 8` | 高 |
| `test_complex_layout_f64` | `size_of::<Complex<f64>>() == 16` | 高 |
| `test_from_real_imag` | `from_real(5.0).im == 0.0`, `from_imag(3.0).re == 0.0` | 高 |
| `test_conj` | `Complex::new(3.0, 4.0).conj() == Complex::new(3.0, -4.0)` | 高 |
| `test_is_real_imaginary` | `from_real(1.0).is_real()`, `from_imag(1.0).is_imaginary()` | 中 |
| `test_add_complex` | `(1+2i) + (3+4i) == (4+6i)` | 高 |
| `test_sub_complex` | `(5+7i) - (2+3i) == (3+4i)` | 高 |
| `test_mul_complex` | `(1+2i) * (3+4i) == (-5+10i)` | 高 |
| `test_div_complex` | `(6+8i) / (3+4i) == (2+0i)` | 高 |
| `test_neg_complex` | `-(1+2i) == (-1-2i)` | 高 |
| `test_add_real` | `(1+2i) + 3.0 == (4+2i)` | 高 |
| `test_real_add_complex` | `3.0 + (1+2i) == (4+2i)` | 高 |
| `test_mul_real` | `(1+2i) * 3.0 == (3+6i)` | 高 |
| `test_div_by_real` | `(6+4i) / 2.0 == (3+2i)` | 高 |
| `test_real_div_complex` | `5.0 / (3+4i)` 正确 | 中 |
| `test_norm_3_4_5` | `Complex::new(3.0, 4.0).norm() == 5.0` | 高 |
| `test_norm_no_overflow` | `Complex::new(1e200, 1e200).norm()` 不溢出 | 高 |
| `test_norm_sqr` | `norm_sqr() == re² + im²` | 中 |
| `test_arg_range` | `arg()` 在 `(-π, π]` 范围内 | 高 |
| `test_exp_ln_inverse` | `z.exp().ln() ≈ z` | 高 |
| `test_sqrt_neg_one` | `Complex::new(-1.0, 0.0).sqrt() ≈ i` | 高 |
| `test_from_polar_i` | `from_polar(1.0, π/2) ≈ i` | 中 |
| `test_eq_nan` | `Complex::new(NaN, 0.0) != self` | 高 |
| `test_display_format` | `"3+4i"`, `"3-4i"`, `"3"`, `"4i"`, `"0"` | 中 |
| `test_f32_to_f64_lossless` | `Complex<f32>→Complex<f64>` 无损 | 高 |
| `test_f64_to_f32_precision_loss` | `Complex<f64>→Complex<f32>` 精度降低 | 中 |
| `test_real_to_complex` | `f64→Complex<f64>` 虚部为 0 | 高 |

### 8.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 零 `Complex::new(0.0, 0.0)` | `norm()==0`, `arg()==0`, `sqrt()==0` |
| `Complex::new(0.0, 0.0).ln()` | 返回 `-∞ + 0i` |
| NaN 参与 | `Complex::new(NaN, 0.0).norm().is_nan()` |
| Inf 参与 | `Complex::new(Inf, 0.0).exp()` 正确处理 |
| 极大值 norm | `Complex::new(1e200, 1e200).norm()` 不溢出（≈1.414e200） |
| 极小值 norm | `Complex::new(1e-200, 1e-200).norm()` 正确 |
| 数组 transmute | `&[Complex<f64>; N]` 可安全转为 `&[f64; 2N]` |

### 8.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `z * z.conj() == z.norm_sqr()` | 随机 z |
| `z.exp().ln() ≈ z` | 随机有限 z |
| `z.sqrt() * z.sqrt() ≈ z` | 随机 z |
| `(z / w) * w ≈ z` | 随机 z, w（w ≠ 0） |
| `z.from_polar(z.norm(), z.arg()) ≈ z` | 随机 z |

---

## 9. 设计决策记录

### 决策 1：自定义 vs num-complex

| 属性 | 值 |
|------|-----|
| 决策 | 自定义 `Complex<T>`，不依赖 `num-complex` |
| 理由 | 零额外依赖；可精确控制 trait 实现（禁止 Eq/Ord）；严格同精度互操作；FFI 布局可验证；与 Element 体系无缝集成 |
| 替代方案 | 使用 `num-complex` — 放弃，引入 num-traits 传递依赖；实现了 Eq 与 NaN 语义冲突 |
| 后果 | 需自行实现所有数学方法；增加维护成本；获得 API 完全控制权 |

### 决策 2：不实现 Eq/Ord

| 属性 | 值 |
|------|-----|
| 决策 | 不实现 `Eq`、`PartialOrd`、`Ord` |
| 理由 | `Eq` 要求自反性（x==x），但 NaN!=NaN 违反此性质；复数无自然全序；实现 Eq 会导致 HashSet 等未定义行为 |
| 替代方案 | 实现 Eq（同 num-complex）— 放弃，NaN 导致语义错误 |
| 替代方案 | 实现 PartialOrd（字典序）— 放弃，无数学意义 |

### 决策 3：norm() 使用 hypot 而非 sqrt(re²+im²)

| 属性 | 值 |
|------|-----|
| 决策 | 使用 `hypot(re, im)` 计算模长 |
| 理由 | 数值稳定：当 re/im 很大时 `re*re` 可能溢出，hypot 使用缩放算法避免；标准库 `f32::hypot`/`f64::hypot` 已实现稳定算法 |
| 替代方案 | `sqrt(re*re + im*im)` — 放弃，大数溢出 |

### 决策 4：不实现复合赋值运算符

| 属性 | 值 |
|------|-----|
| 决策 | 当前版本不实现 `AddAssign`/`SubAssign`/`MulAssign`/`DivAssign` |
| 理由 | 需求说明书 §20 指出"所有组合均产生新的独立张量"；运算符重载仅产生新张量，不需要原地修改 |
| 替代方案 | 实现全部复合赋值 — 放弃，超出当前需求 |

---

## 10. 性能考量

| 方面 | 设计决策 |
|------|----------|
| `#[repr(C)]` | 保证 FFI 兼容且无 padding，内存紧凑 |
| 内联 | 所有运算方法标注 `#[inline]`，编译器可内联 |
| 零堆分配 | `Complex<T>` 为 `Copy` 类型，全部栈分配 |
| 单态化 | `Complex<f32>` 和 `Complex<f64>` 各自单态化，无虚调用 |
| hypot 开销 | `norm()` 的 hypot 比直接 sqrt 多几次比较和分支，但避免溢出，开销可接受 |

---

## 11. no_std 兼容性

| 组件 | 兼容方案 |
|------|----------|
| `Complex<T>` 结构体 | 纯 `#[repr(C)]`，天然 no_std |
| 基础方法（`re()`, `im()`, `conj()`, `from_real()`, `from_imag()`, `is_real()`, `is_imaginary()`） | 不依赖 `Float` trait，**no_std 可用** |
| 算术运算（`+`, `-`, `*`, `/`, 一元负号） | 仅依赖 `core::ops`，天然 no_std |
| 数学方法（`norm()`, `arg()`, `exp()`, `ln()`, `sqrt()`, `to_polar()`, `from_polar()`） | 具体类型 `impl Complex<f32>` / `impl Complex<f64>` 内部调用 `f32::hypot`、`f32::atan2`、`f32::exp`、`f32::sin` 等方法。**这些浮点数学函数在 Rust 1.85 中仍在 `std` 中，不在 `core`**。因此 Complex 的数学方法在 no_std 下不可用，**需要启用 `std` feature** |
| 数学方法（`norm_sqr()`） | 仅使用 `+` 和 `*`，不依赖浮点函数，**no_std 可用** |
| `is_nan()`, `is_finite()` | 具体类型实现，`f32::is_nan()`/`f32::is_finite()` 在 `core` 中提供，**no_std 可用** |
| 类型转换 | `From` trait 实现和 `to_f32()` 方法，天然 no_std |

> **与 `00-coding.md` §9.1 保持一致**：libm **不是** Xenon 的依赖。`Complex<f32>`/`Complex<f64>` 的数学方法（`norm`, `exp`, `ln`, `sqrt`, `arg`, `to_polar`, `from_polar`）使用 `f32`/`f64` 的 inherent 方法（`hypot`, `atan2`, `exp`, `sin`, `cos`, `ln`, `sqrt` 等）。**这些浮点数学函数在 Rust 1.85 中仍在 `std` 中，不在 `core`**。因此在 `no_std` 环境下，这些数学方法不可用，需要启用 `std` feature。`norm_sqr()`、基础方法、算术运算、类型转换等不依赖浮点数学函数，在 `no_std` 下均可使用。

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
