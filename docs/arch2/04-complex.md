# 复数类型模块设计

> 模块：`src/complex.rs` | 阶段：Phase 2 W2 | 依赖：`element.rs`

---

## 1. 模块定位

`complex` 模块为 Xenon 提供自研复数类型 `Complex<T>`，作为多维数组的合法元素类型之一。本模块不依赖任何外部 crate（如 `num-complex`），完全自主实现，满足以下核心目标：

| 目标 | 说明 |
|------|------|
| FFI 兼容 | `#[repr(C)]` 布局兼容 C99 `_Complex`，可安全 `transmute` 指针 |
| 零开销 | 泛型单态化，无虚调用，所有方法 `#[inline]` |
| 同精度互操作 | `Complex<f64>` 与 `f64` 可直接混合运算；跨精度须显式转换 |
| 无序性 | 仅实现 `PartialEq`，不实现 `Eq`/`PartialOrd`/`Ord` |
| 数值稳定 | `norm()` 使用 hypot 算法避免中间溢出 |

### 在元素类型层次中的位置

```
Element (基础层)
└── Numeric (数值层)
    ├── RealScalar  → f32, f64
    └── ComplexScalar → Complex<f32>, Complex<f64>
```

`Complex<T>` 本身是数据结构，`ComplexScalar` trait 定义于 `element.rs` 中描述复数的能力接口。本模块负责：
1. 定义 `Complex<T>` 结构体及其全部算术/转换实现
2. 为 `Complex<T>` 实现 `Element`、`Numeric` 等 trait（使其可作为张量元素）
3. `ComplexScalar` trait 在 `element.rs` 中声明，本模块仅为 `Complex<f32>`、`Complex<f64>` 提供实现

---

## 2. 文件位置

```
src/
  complex.rs          ← 本模块，Complex<T> 定义 + 所有算术 impl
  element.rs          ← ComplexScalar trait 声明，Complex<T> impl Element/Numeric
  lib.rs              ← pub mod complex; pub use complex::Complex;
```

`lib.rs` 中的导出：

```rust
pub mod complex;
pub use complex::Complex;
```

---

## 3. 依赖关系

```
complex.rs
  ├── 依赖 element.rs（通过 crate::element 引用 Element, Numeric, RealScalar）
  ├── 依赖 core::ops::{Add, Sub, Mul, Div, Neg}
  └── 依赖 core::ops::{AddAssign, SubAssign, MulAssign, DivAssign}
```

**不依赖**任何外部 crate。所有数学运算使用 `core` 或通过 `T` 自身的方法完成。

---

## 4. 公共 API 设计

### 4.1 结构体定义

```rust
/// A complex number in Cartesian form.
///
/// # Memory layout
///
/// `#[repr(C)]` guarantees the fields are laid out as `[re, im]` in memory,
/// compatible with C99 `_Complex T`. This enables safe pointer transmutation
/// between `*const Complex<f64>` and `*const f64` (with doubled element count).
///
/// # Invariants
///
/// - `size_of::<Complex<T>>() == 2 * size_of::<T>()`
/// - `align_of::<Complex<T>>() == align_of::<T>()`
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}
```

**字段可见性**：`pub`，允许用户直接读写 `re`/`im`，与 `num-complex` 风格一致。

### 4.2 构造方法

```rust
impl<T> Complex<T> {
    /// Creates a new complex number from real and imaginary parts.
    #[inline]
    pub const fn new(re: T, im: T) -> Self;

    /// Creates a complex number with zero imaginary part.
    #[inline]
    pub const fn from_real(re: T) -> Self;
}

impl<T: Copy> Complex<T> {
    /// Returns the imaginary unit (0 + 1i).
    /// Only meaningful when T has a `one()` concept.
    #[inline]
    pub fn i() -> Self
    where
        T: core::ops::Neg<Output = T>,
    {
        // Used via ComplexScalar::i() — this is a convenience.
        unimplemented!("see ComplexScalar impl")
    }
}
```

> **设计决策**：`i()` 更适合作为 `ComplexScalar` 的关联常量/方法返回，此处不独立实现。实际虚数单位通过 `ComplexScalar::i()` 获取。

### 4.3 复数专属方法

以下方法在 `Complex<T>` 上直接定义（`T: RealScalar` 约束），同时 `ComplexScalar` trait 会暴露相同的接口。

```rust
impl<T: RealScalar> Complex<T> {
    /// Returns the complex conjugate (re - i*im).
    #[inline]
    pub fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    /// Returns the magnitude (modulus) using the hypot algorithm.
    ///
    /// This avoids overflow for large components by delegating to
    /// `T::hypot(self.re, self.im)` which uses the scaled algorithm.
    #[inline]
    pub fn norm(self) -> T {
        T::hypot(self.re, self.im)
    }

    /// Returns the argument (phase angle) in the range (-π, π].
    #[inline]
    pub fn arg(self) -> T {
        T::atan2(self.im, self.re)
    }

    /// Creates a complex number from polar coordinates (r, theta).
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self {
            re: r * T::cos(theta),
            im: r * T::sin(theta),
        }
    }

    /// Returns the complex square root (principal value).
    ///
    /// For z = a + bi, the principal square root is:
    ///   sqrt(z) = sqrt(|z| + a)/sqrt(2) + sign(b) * sqrt(|z| - a)/sqrt(2) * i
    pub fn sqrt(self) -> Self {
        // See Section 5.2 for algorithm details
    }

    /// Returns e^(self) using Euler's formula.
    ///
    /// e^(a+bi) = e^a * (cos(b) + i*sin(b))
    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = T::exp(self.re);
        Self {
            re: exp_re * T::cos(self.im),
            im: exp_re * T::sin(self.im),
        }
    }

    /// Returns the principal natural logarithm.
    ///
    /// ln(z) = ln|z| + i*arg(z)
    pub fn ln(self) -> Self {
        Self {
            re: T::ln(self.norm()),
            im: self.arg(),
        }
    }

    /// Checks approximate equality component-wise.
    ///
    /// Returns `true` if `|self.re - other.re| <= epsilon` AND
    /// `|self.im - other.im| <= epsilon`.
    #[inline]
    pub fn approx_eq(self, other: Self, epsilon: T) -> bool {
        (self.re - other.re).abs() <= epsilon && (self.im - other.im).abs() <= epsilon
    }
}
```

### 4.4 PartialEq 实现

```rust
/// Component-wise equality. NaN != NaN per IEEE 754 semantics.
impl<T: PartialEq> PartialEq for Complex<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}
```

**不实现** `Eq`、`PartialOrd`、`Ord`——复数无自然全序。

### 4.5 算术运算：Complex × Complex

```rust
impl<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl<T: Clone + Sub<Output = T> + Mul<Output = T>> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        Self {
            re: self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone(),
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>> Div for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
        let denom = rhs.re.clone() * rhs.re.clone() + rhs.im.clone() * rhs.im.clone();
        Self {
            re: (self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone()) / denom.clone(),
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self { re: -self.re, im: -self.im }
    }
}
```

**复合赋值运算符**：

```rust
impl<T: Clone + AddAssign> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T: Clone + SubAssign> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + MulAssign> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + DivAssign> DivAssign for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}
```

### 4.6 算术运算：Complex × Real（同精度互操作）

核心设计：为 `Complex<T>` 实现 `Add<T>`、`Sub<T>` 等 trait，其中 `T: RealScalar`。同时实现反向运算 `T + Complex<T>` 等。

```rust
// Complex<T> + T
impl<T: Clone + Add<Output = T>> Add<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self { re: self.re + rhs, im: self.im }
    }
}

// T + Complex<T>
impl<T: Clone + Add<Output = T>> Add<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        Complex { re: self + rhs.re, im: rhs.im }
    }
}

// Complex<T> - T
impl<T: Clone + Sub<Output = T>> Sub<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self { re: self.re - rhs, im: self.im }
    }
}

// T - Complex<T>
impl<T: Clone + Sub<Output = T> + Neg<Output = T>> Sub<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output {
        Complex { re: self - rhs.re, im: -rhs.im }
    }
}

// Complex<T> * T (scalar multiplication)
impl<T: Clone + Mul<Output = T>> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self { re: self.re * rhs.clone(), im: self.im * rhs }
    }
}

// T * Complex<T> (scalar multiplication, commutative)
impl<T: Clone + Mul<Output = T>> Mul<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        Complex { re: self.clone() * rhs.re, im: self * rhs.im }
    }
}

// Complex<T> / T (scalar division)
impl<T: Clone + Div<Output = T>> Div<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self { re: self.re / rhs.clone(), im: self.im / rhs }
    }
}

// T / Complex<T> (real divided by complex)
// Uses conjugate: r/(a+bi) = r*(a-bi)/(a²+b²)
impl<T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>> Div<Complex<T>> for T {
    type Output = Complex<T>;
    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        let denom = rhs.re.clone() * rhs.re.clone() + rhs.im.clone() * rhs.im.clone();
        Complex {
            re: self.clone() * rhs.re / denom.clone(),
            im: self * (-rhs.im) / denom,
        }
    }
}
```

**复合赋值：Complex × Real**

```rust
impl<T: Clone + AddAssign> AddAssign<T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

impl<T: Clone + SubAssign> SubAssign<T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

impl<T: Clone + MulAssign> MulAssign<T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.re *= rhs.clone();
        self.im *= rhs;
    }
}

impl<T: Clone + DivAssign> DivAssign<T> for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.re /= rhs.clone();
        self.im /= rhs;
    }
}
```

> **精度约束**：以上所有 `Complex<T> ↔ T` 运算中，`T` 同时出现于 `Complex<T>` 和实参位置。由于 Rust 类型系统要求两侧类型完全匹配，跨精度运算（如 `Complex<f64> + f32`）会在编译期被拒绝。这完全符合需求规格 §3.2。

### 4.7 From/Into 转换

```rust
impl<T> From<T> for Complex<T> {
    /// Converts a real number to a complex number with zero imaginary part.
    #[inline]
    fn from(re: T) -> Self {
        Self {
            re,
            im: T::default(), // requires Default — see below
        }
    }
}

impl<T> From<(T, T)> for Complex<T> {
    /// Creates from a (re, im) tuple.
    #[inline]
    fn from((re, im): (T, T)) -> Self {
        Self { re, im }
    }
}

impl<T> From<Complex<T>> for (T, T) {
    /// Destructures into a (re, im) tuple.
    #[inline]
    fn from(c: Complex<T>) -> Self {
        (c.re, c.im)
    }
}
```

> **注意**：`From<T>` 需要 `T: Default` 以获得零值。对于浮点类型，`f32::default() == 0.0`，`f64::default() == 0.0`，满足需求。或者可在 `T: RealScalar` 约束下使用 `T::zero()`。实际实现中选择后者以避免 `Default` 约束泄漏。

修正方案：

```rust
impl<T: crate::element::Element> From<T> for Complex<T> {
    #[inline]
    fn from(re: T) -> Self {
        Self { re, im: T::zero() }
    }
}
```

### 4.8 Display 实现

```rust
impl<T: core::fmt::Display + core::fmt::Debug + PartialOrd + Default> core::fmt::Display for Complex<T> {
    /// Formats as "re+imi" or "re-imi" depending on sign of imaginary part.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Display logic: handle sign of im, special-case zero parts
        // ...
    }
}
```

### 4.9 Element / Numeric trait 实现

```rust
// Complex<T> satisfies Element when T: Element
impl<T: crate::element::Element> crate::element::Element for Complex<T> {
    #[inline]
    fn zero() -> Self {
        Self { re: T::zero(), im: T::zero() }
    }

    #[inline]
    fn one() -> Self {
        Self { re: T::one(), im: T::zero() }
    }
}

// Complex<T> satisfies Numeric when T: RealScalar
// (because Complex arithmetic ops need T to support +,-,*,/)
impl<T: crate::element::RealScalar> crate::element::Numeric for Complex<T> {}
```

---

## 5. 内部实现设计

### 5.1 Hypot 算法用于 norm()

直接计算 `sqrt(re² + im²)` 在 `|re|` 或 `|im|` 很大时会溢出为 `inf`，即使最终结果在浮点表示范围内。Hypot 算法通过缩放避免此问题：

```rust
impl<T: RealScalar> Complex<T> {
    #[inline]
    pub fn norm(self) -> T {
        T::hypot(self.re, self.im)
    }
}
```

`T::hypot` 的标准实现（对于 `f32`/`f64`，直接使用 `libm` 的 `hypot`）内部逻辑为：

```
function hypot(a, b):
    |a|, |b| ← abs(a), abs(b)
    if |a| < |b|: swap(a, b)    // 确保 |a| >= |b|
    if |a| == 0: return |b|      // 处理 (0, 0) 情况
    r ← b / a                    // |r| <= 1，不会溢出
    return |a| * sqrt(1 + r²)    // sqrt(1+x) 对 x<=1 数值稳定
```

对于 `f64` 和 `f32`，Rust 标准库已提供 `f64::hypot()` 和 `f32::hypot()`，它们调用底层 `libm` 实现，已处理好溢出/下溢/NaN 等边界情况。

### 5.2 复数平方根算法

```rust
impl<T: RealScalar> Complex<T> {
    pub fn sqrt(self) -> Self {
        // Handle special cases first
        if self.im == T::zero() {
            if self.re >= T::zero() {
                // sqrt of non-negative real → real result
                return Self::from_real(T::sqrt(self.re));
            } else {
                // sqrt of negative real → pure imaginary
                return Self { re: T::zero(), im: T::sqrt(-self.re) };
            }
        }

        // General case: principal square root of a + bi
        // |z| = hypot(a, b)
        // sqrt(z) = sqrt((|z| + a) / 2) + sign(b) * sqrt((|z| - a) / 2) * i
        let abs_z = self.norm();
        let re_part = T::sqrt((abs_z + self.re) / (T::one() + T::one()));
        let im_part = T::sqrt((abs_z - self.re) / (T::one() + T::one()));
        let sign_im = if self.im >= T::zero() { im_part } else { -im_part };

        Self { re: re_part, im: sign_im }
    }
}
```

> 使用 `T::one() + T::one()` 代替字面量 `2.0` 以保持泛型兼容性。

### 5.3 实数互操作的实现机制

**同精度保证**完全通过 Rust 类型系统实现——不需要运行时检查：

```rust
// 编译通过：T = f64 在两侧匹配
let c: Complex<f64> = Complex::new(1.0, 2.0);
let result = c + 3.0f64;  // OK: Complex<f64> + f64

// 编译失败：T = f64 在左侧，但右侧是 f32
let c: Complex<f64> = Complex::new(1.0, 2.0);
let result = c + 3.0f32;  // ERROR: no impl for Complex<f64> + f32
```

这是因为 `impl<T> Add<T> for Complex<T>` 中，`T` 由 `Complex<T>` 的类型参数和 `Add<T>` 的 `Rhs` 类型参数同时决定，两侧必须是同一类型。

### 5.4 内存布局保证

```rust
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}
```

`#[repr(C)]` 保证：

| 属性 | 保证 | 验证方式 |
|------|------|----------|
| 字段顺序 | `re` 在低地址，`im` 紧随其后 | `#[repr(C)]` 定义 |
| 大小 | `size_of::<Complex<T>>() == 2 * size_of::<T>()` | 无 padding（两字段同类型） |
| 对齐 | `align_of::<Complex<T>>() == align_of::<T>()` | `#[repr(C)]` 无额外对齐 |
| C99 兼容 | `*const Complex<f64>` 可 transmute 为 `*const C99 _Complex double` | `#[repr(C)]` 与 C ABI 兼容 |
| 数组等价 | `[Complex<f64>; N]` 的内存布局与 `[f64; 2*N]` 完全等价 | 连续排列无 padding |

**静态断言**（编译期验证）：

```rust
// In complex.rs, module level or in a test
const _: () = {
    assert!(core::mem::size_of::<Complex<f64>>() == 2 * core::mem::size_of::<f64>());
    assert!(core::mem::align_of::<Complex<f64>>() == core::mem::align_of::<f64>());
    assert!(core::mem::size_of::<Complex<f32>>() == 2 * core::mem::size_of::<f32>());
    assert!(core::mem::align_of::<Complex<f32>>() == core::mem::align_of::<f32>());
};
```

---

## 6. 实现任务拆分

每个任务约 10 分钟，可独立验证和提交。

### Task 1: 结构体定义 + 构造方法
```
Task: Define Complex<T> struct and constructors
File: src/complex.rs:1-40
Test: test_complex_new, test_complex_from_real, test_complex_from_tuple
Deps: None
Est: 10 min
```
- [ ] 定义 `#[repr(C)] pub struct Complex<T> { pub re: T, pub im: T }`
- [ ] 实现 `new()`, `from_real()`
- [ ] 实现 `From<(T,T)>`, `From<T>` (where T: Element)
- [ ] 添加编译期内存布局断言
- [ ] 添加 `#[derive(Copy, Clone, Debug, Default)]`

### Task 2: PartialEq + Display
```
Task: Implement PartialEq and Display for Complex<T>
File: src/complex.rs:41-80
Test: test_complex_eq, test_complex_ne_nan, test_complex_display
Deps: Task 1
Est: 10 min
```
- [ ] 实现 `PartialEq<Complex<T>> for Complex<T>`（逐分量）
- [ ] 实现 `Display for Complex<T>`
- [ ] 验证 NaN != NaN 语义

### Task 3: Complex × Complex 算术
```
Task: Implement Add/Sub/Mul/Div/Neg for Complex<T>
File: src/complex.rs:81-160
Test: test_complex_add, test_complex_sub, test_complex_mul, test_complex_div, test_complex_neg
Deps: Task 1
Est: 10 min
```
- [ ] `impl Add for Complex<T>`
- [ ] `impl Sub for Complex<T>`
- [ ] `impl Mul for Complex<T>`
- [ ] `impl Div for Complex<T>`
- [ ] `impl Neg for Complex<T>`

### Task 4: 复合赋值运算符
```
Task: Implement AddAssign/SubAssign/MulAssign/DivAssign for Complex<T>
File: src/complex.rs:161-200
Test: test_complex_add_assign, test_complex_mul_assign
Deps: Task 3
Est: 8 min
```
- [ ] `impl AddAssign for Complex<T>`
- [ ] `impl SubAssign for Complex<T>`
- [ ] `impl MulAssign for Complex<T>`
- [ ] `impl DivAssign for Complex<T>`

### Task 5: Complex × Real 互操作
```
Task: Implement arithmetic between Complex<T> and T
File: src/complex.rs:201-300
Test: test_complex_add_real, test_real_add_complex, test_complex_mul_real, test_real_div_complex
Deps: Task 1, element.rs (RealScalar or at least the trait bounds)
Est: 10 min
```
- [ ] `impl Add<T> for Complex<T>`
- [ ] `impl Add<Complex<T>> for T`
- [ ] `impl Sub<T> for Complex<T>`
- [ ] `impl Sub<Complex<T>> for T`
- [ ] `impl Mul<T> for Complex<T>`
- [ ] `impl Mul<Complex<T>> for T`
- [ ] `impl Div<T> for Complex<T>`
- [ ] `impl Div<Complex<T>> for T`

### Task 6: Real 复合赋值
```
Task: Implement AddAssign/SubAssign/MulAssign/DivAssign<T> for Complex<T>
File: src/complex.rs:301-330
Test: test_complex_add_assign_real, test_complex_div_assign_real
Deps: Task 5
Est: 5 min
```
- [ ] `impl AddAssign<T> for Complex<T>`
- [ ] `impl SubAssign<T> for Complex<T>`
- [ ] `impl MulAssign<T> for Complex<T>`
- [ ] `impl DivAssign<T> for Complex<T>`

### Task 7: 复数方法（conj, norm, arg, from_polar, approx_eq）
```
Task: Implement core complex methods
File: src/complex.rs:331-400
Test: test_complex_conj, test_complex_norm, test_complex_arg, test_complex_from_polar, test_approx_eq
Deps: Task 1, element.rs (RealScalar)
Est: 10 min
```
- [ ] `conj(self) -> Self`
- [ ] `norm(self) -> T`（使用 `T::hypot`）
- [ ] `arg(self) -> T`（使用 `T::atan2`）
- [ ] `from_polar(r, theta) -> Self`
- [ ] `approx_eq(self, other, epsilon) -> bool`

### Task 8: 复数超越函数（exp, ln, sqrt）
```
Task: Implement exp, ln, sqrt for Complex<T>
File: src/complex.rs:401-460
Test: test_complex_exp, test_complex_ln, test_complex_sqrt, test_complex_exp_ln_inverse
Deps: Task 7
Est: 10 min
```
- [ ] `exp(self) -> Self`（Euler 公式）
- [ ] `ln(self) -> Self`（对数主值）
- [ ] `sqrt(self) -> Self`（主平方根，含特殊值处理）

### Task 9: Element + Numeric trait 实现
```
Task: Implement Element and Numeric traits for Complex<T>
File: src/element.rs (add impls), src/complex.rs (add trait impls if needed)
Test: test_complex_element_zero, test_complex_element_one
Deps: Task 3, Task 7, element.rs
Est: 8 min
```
- [ ] `impl Element for Complex<T> where T: Element`
- [ ] `impl Numeric for Complex<T> where T: RealScalar`
- [ ] 确认 `Complex<T>` 满足所有 `Numeric` 约束

### Task 10: ComplexScalar trait 实现
```
Task: Implement ComplexScalar for Complex<f32> and Complex<f64>
File: src/element.rs (ComplexScalar impl), src/complex.rs (helper methods)
Test: test_complex_scalar_f64, test_complex_scalar_f32
Deps: Task 8, Task 9, element.rs
Est: 10 min
```
- [ ] 为 `Complex<f64>` 实现 `ComplexScalar`
- [ ] 为 `Complex<f32>` 实现 `ComplexScalar`
- [ ] 实现 `i()` 关联方法
- [ ] 验证所有 `ComplexScalar` 方法可用

### Task 11: lib.rs 集成 + 文档
```
Task: Wire up module in lib.rs and add doc comments
File: src/lib.rs, src/complex.rs
Test: test_complex_public_api_visible
Deps: Task 10
Est: 8 min
```
- [ ] `pub mod complex;` in `lib.rs`
- [ ] `pub use complex::Complex;` in `lib.rs`
- [ ] 审查所有 doc comments
- [ ] 运行 `cargo doc` 无 warning

---

## 7. 测试计划

### 7.1 单元测试（`src/complex.rs` 内 `#[cfg(test)] mod tests`）

| 测试函数 | 验证内容 |
|----------|----------|
| `test_complex_new` | `new(3.0, 4.0)` 产生 `re=3, im=4` |
| `test_complex_from_real` | `from_real(5.0)` 产生 `re=5, im=0` |
| `test_complex_from_tuple` | `Complex::from((1.0, 2.0))` |
| `test_complex_eq` | 相等/不等情况 |
| `test_complex_ne_nan` | `Complex::new(f64::NAN, 0.0) != Complex::new(f64::NAN, 0.0)` |
| `test_complex_add` | `(1+2i) + (3+4i) = (4+6i)` |
| `test_complex_sub` | `(5+3i) - (2+1i) = (3+2i)` |
| `test_complex_mul` | `(1+2i)*(3+4i) = (-5+10i)` |
| `test_complex_div` | `(-5+10i)/(3+4i) ≈ (1+2i)` |
| `test_complex_neg` | `-(1+2i) = (-1-2i)` |
| `test_complex_add_assign` | `a += b` 正确性 |
| `test_complex_mul_assign` | `a *= b` 正确性 |
| `test_complex_add_real` | `(1+2i) + 3 = (4+2i)` |
| `test_real_add_complex` | `3 + (1+2i) = (4+2i)` |
| `test_complex_mul_real` | `(1+2i) * 3 = (3+6i)` |
| `test_real_div_complex` | `5 / (3+4i) ≈ (0.6-0.8i)` |
| `test_complex_conj` | `(3+4i).conj() = (3-4i)` |
| `test_complex_norm` | `(3+4i).norm() == 5.0` |
| `test_norm_no_overflow` | `(1e300+1e300i).norm()` 不溢出 |
| `test_complex_arg` | `(1+1i).arg() ≈ π/4` |
| `test_complex_from_polar` | `from_polar(1, π/4) ≈ (0.707, 0.707)` |
| `test_approx_eq` | 近似相等正确判断 |
| `test_approx_eq_reject` | 超过 epsilon 时返回 false |
| `test_complex_exp` | `exp(iπ) ≈ -1` |
| `test_complex_ln` | `ln(exp(z)) ≈ z` 往返测试 |
| `test_complex_sqrt` | `sqrt(-1) = i` |
| `test_complex_sqrt_real_neg` | `sqrt(Complex::from_real(-4)) = 2i` |
| `test_memory_layout_f64` | `size_of == 16, align_of == 8` |
| `test_memory_layout_f32` | `size_of == 8, align_of == 4` |
| `test_array_memory_equiv` | `[Complex<f64>; 3]` transmute 为 `[f64; 6]` 正确 |

### 7.2 集成测试

| 文件 | 测试函数 | 验证内容 |
|------|----------|----------|
| `tests/arithmetic.rs` | `test_complex_tensor_element_wise` | Complex 张量逐元素运算 |
| `tests/ffi.rs` | `test_complex_pointer_transmute` | C99 兼容指针转换 |
| `tests/edge_cases.rs` | `test_complex_nan_propagation` | NaN 在复数运算中正确传播 |
| `tests/edge_cases.rs` | `test_complex_inf_arithmetic` | Inf 参与复数运算遵循 IEEE 754 |
| `tests/edge_cases.rs` | `test_complex_zero_div` | 复数除以零的行为 |

### 7.3 属性测试

| 文件 | 测试属性 | 描述 |
|------|----------|------|
| `tests/property/` | `complex_add_commutative` | `a + b == b + a` |
| `tests/property/` | `complex_mul_conj_norm` | `z * z.conj() == norm(z)²` |
| `tests/property/` | `complex_exp_ln_roundtrip` | `exp(ln(z)) ≈ z` (z ≠ 0) |
| `tests/property/` | `complex_sqrt_sq_roundtrip` | `sqrt(z)² ≈ z` |

---

## 附录 A：完整公共 API 签名汇总

```rust
// === Struct ===
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

// === Constructors ===
impl<T> Complex<T> {
    pub const fn new(re: T, im: T) -> Self;
}

// === Real-dependent methods ===
impl<T: RealScalar> Complex<T> {
    pub fn from_real(re: T) -> Self;           // Actually needs T: Element for zero()
    pub fn conj(self) -> Self;
    pub fn norm(self) -> T;
    pub fn arg(self) -> T;
    pub fn from_polar(r: T, theta: T) -> Self;
    pub fn sqrt(self) -> Self;
    pub fn exp(self) -> Self;
    pub fn ln(self) -> Self;
    pub fn approx_eq(self, other: Self, epsilon: T) -> bool;
}

// === PartialEq ===
impl<T: PartialEq> PartialEq for Complex<T> { ... }

// === Complex × Complex arithmetic ===
impl<T: Clone + Add<Output=T>> Add for Complex<T> { type Output = Self; }
impl<T: Clone + Sub<Output=T>> Sub for Complex<T> { type Output = Self; }
impl<T: Clone + Sub<Output=T> + Mul<Output=T>> Mul for Complex<T> { type Output = Self; }
impl<T: Clone + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>> Div for Complex<T> { type Output = Self; }
impl<T: Neg<Output=T>> Neg for Complex<T> { type Output = Self; }

// === Complex × Complex compound assignment ===
impl<T: Clone + AddAssign> AddAssign for Complex<T> { ... }
impl<T: Clone + SubAssign> SubAssign for Complex<T> { ... }
impl<T: Clone + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + MulAssign> MulAssign for Complex<T> { ... }
impl<T: Clone + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + DivAssign> DivAssign for Complex<T> { ... }

// === Complex × Real arithmetic ===
impl<T: Clone + Add<Output=T>> Add<T> for Complex<T> { type Output = Self; }
impl<T: Clone + Add<Output=T>> Add<Complex<T>> for T { type Output = Complex<T>; }
impl<T: Clone + Sub<Output=T>> Sub<T> for Complex<T> { type Output = Self; }
impl<T: Clone + Sub<Output=T> + Neg<Output=T>> Sub<Complex<T>> for T { type Output = Complex<T>; }
impl<T: Clone + Mul<Output=T>> Mul<T> for Complex<T> { type Output = Self; }
impl<T: Clone + Mul<Output=T>> Mul<Complex<T>> for T { type Output = Complex<T>; }
impl<T: Clone + Div<Output=T>> Div<T> for Complex<T> { type Output = Self; }
impl<T: Clone + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>> Div<Complex<T>> for T { type Output = Complex<T>; }

// === Complex × Real compound assignment ===
impl<T: Clone + AddAssign> AddAssign<T> for Complex<T> { ... }
impl<T: Clone + SubAssign> SubAssign<T> for Complex<T> { ... }
impl<T: Clone + MulAssign> MulAssign<T> for Complex<T> { ... }
impl<T: Clone + DivAssign> DivAssign<T> for Complex<T> { ... }

// === Conversions ===
impl<T: Element> From<T> for Complex<T> { ... }
impl<T> From<(T, T)> for Complex<T> { ... }
impl<T> From<Complex<T>> for (T, T) { ... }

// === Display ===
impl<T: Display + Debug + PartialOrd + Default> Display for Complex<T> { ... }

// === Element / Numeric trait impls ===
impl<T: Element> Element for Complex<T> { ... }
impl<T: RealScalar> Numeric for Complex<T> { ... }
```

## 附录 B：禁止事项清单

| 项目 | 原因 |
|------|------|
| ❌ 引入 `num-complex` | 需求明确要求自研，无外部依赖 |
| ❌ 实现 `Eq` | NaN 导致自反性不成立 |
| ❌ 实现 `PartialOrd` / `Ord` | 复数无自然全序 |
| ❌ 跨精度隐式转换 | `Complex<f64> + f32` 须编译失败 |
| ❌ 整数隐式提升 | `Complex<f64> + i32` 须编译失败 |
| ❌ 直接 `sqrt(re² + im²)` 用于 norm | 大值时溢出 |
