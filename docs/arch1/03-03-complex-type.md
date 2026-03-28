# 复数类型模块设计文档

> 版本: 0.1.0 | 最后更新: 2026-03-28 | 状态: 设计中

---

## 1. 模块概述

### 1.1 为什么自定义 Complex 类型

Xenon 选择实现自定义 `Complex<T>` 类型，而非使用 `num-complex` crate，原因如下：

| 考量 | 自定义实现 | num-complex |
|------|-----------|-------------|
| **FFI 兼容性** | `#[repr(C)]` 保证与 C99 `_Complex` 布局兼容 | 同样支持，但需验证 |
| **依赖控制** | 零额外依赖，符合最小依赖原则 | 引入 num-traits 等传递依赖 |
| **API 精确控制** | 可精确控制 trait 实现（如禁止 Eq/Ord） | 实现了 Eq，与需求不符 |
| **精度约束** | 严格限制同精度互操作 | 支持更宽松的混合精度 |
| **与 Element trait 集成** | 无缝集成到 Xenon 元素类型体系 | 需要额外适配层 |

### 1.2 设计目标

1. **FFI 友好**: 内存布局兼容 C99 `_Complex`，可直接 transmute 指针
2. **零依赖**: 不引入任何外部 crate
3. **类型安全**: 编译期保证同精度互操作，拒绝跨精度隐式转换
4. **数值稳定**: `norm()` 使用 hypot 算法避免中间溢出
5. **语义正确**: NaN != NaN，不实现 Eq/Ord

### 1.3 模块边界

```
src/complex/
├── mod.rs     # Complex<T> 定义、基础方法、trait 实现
├── ops.rs     # 算术运算符实现（Add/Sub/Mul/Div）
└── cast.rs    # 类型转换（Complex<f32> ↔ Complex<f64>、实数→复数）
```

---

## 2. 文件结构

### 2.1 mod.rs — 核心定义

**职责**:
- `Complex<T>` 结构体定义
- 构造方法 `new(re, im)`
- 基础方法 `re()`, `im()`, `conj()`
- 数学方法 `norm()`, `arg()`, `exp()`, `ln()`, `sqrt()`, `from_polar()`
- `PartialEq` 实现
- `approx_eq()` 方法
- 内存布局静态断言

**导出**:
```rust
pub use Complex;
```

### 2.2 ops.rs — 算术运算

**职责**:
- `Complex<T> + Complex<T>` 实现
- `Complex<T> - Complex<T>` 实现
- `Complex<T> * Complex<T>` 实现
- `Complex<T> / Complex<T>` 实现
- `Complex<T> + T` / `T + Complex<T>` 互操作
- `Complex<T> * T` / `T * Complex<T>` 互操作
- `Complex<T> / T` / `T / Complex<T>` 互操作
- `Neg` 一元运算符

**约束**: 仅支持 `T: Float`（f32 或 f64）

### 2.3 cast.rs — 类型转换

**职责**:
- `Complex<f32>` → `Complex<f64>` (精度提升)
- `Complex<f64>` → `Complex<f32>` (精度降低)
- `f32`/`f64` → `Complex<f32>`/`Complex<f64>` (实数提升)

**不支持的转换**:
- 整数 → 复数（需显式先转浮点）
- 跨精度互操作（如 `f32` + `Complex<f64>`）

---

## 3. Complex<T> 结构体设计

### 3.1 完整定义

```rust
/// 复数类型，表示 a + bi。
///
/// # 内存布局
///
/// - `#[repr(C)]` 保证字段按声明顺序连续排列
/// - 大小: `2 * size_of::<T>()`
/// - 对齐: `align_of::<T>()`
/// - 兼容 C99 `_Complex T` 类型
///
/// # 示例
///
/// ```
/// use xenon::complex::Complex;
///
/// let z = Complex::new(3.0, 4.0);  // 3 + 4i
/// assert_eq!(z.re(), 3.0);
/// assert_eq!(z.im(), 4.0);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex<T> {
    /// 实部
    pub re: T,
    /// 虚部
    pub im: T,
}
```

### 3.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `re` | `T` | 实部 (real part) |
| `im` | `T` | 虚部 (imaginary part) |

### 3.3 泛型约束

`Complex<T>` 的核心实现仅支持 `T: Float`，即 `f32` 或 `f64`。

```rust
use num_traits::Float;  // 实际实现中使用 std 库 trait

// Xenon 内部定义的 Float trait（简化版）
pub trait Float:
    Copy
    + Clone
    + Default
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    // 必要的数学方法
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
    fn epsilon() -> Self;
}

impl Float for f32 { /* ... */ }
impl Float for f64 { /* ... */ }
```

### 3.4 构造方法

```rust
impl<T> Complex<T> {
    /// 创建新的复数。
    ///
    /// # 参数
    ///
    /// - `re`: 实部
    /// - `im`: 虚部
    ///
    /// # 示例
    ///
    /// ```
    /// let z = Complex::new(1.0, 2.0);  // 1 + 2i
    /// ```
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Float> Complex<T> {
    /// 返回实部。
    #[inline]
    pub fn re(self) -> T {
        self.re
    }

    /// 返回虚部。
    #[inline]
    pub fn im(self) -> T {
        self.im
    }

    /// 创建纯实数（虚部为 0）。
    #[inline]
    pub fn from_real(re: T) -> Self {
        Self::new(re, T::zero())
    }

    /// 创建纯虚数（实部为 0）。
    #[inline]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::zero(), im)
    }

    /// 虚数单位 i。
    #[inline]
    pub fn i() -> Self {
        Self::new(T::zero(), T::one())
    }
}
```

---

## 4. 内存布局保证

### 4.1 repr(C) 语义

`#[repr(C)]` 确保以下布局属性：

1. **字段顺序**: `re` 在前，`im` 在后，与声明顺序一致
2. **无填充**: 两个相同类型的字段紧密排列，中间无 padding
3. **对齐**: 结构体对齐等于字段类型的对齐

### 4.2 大小与对齐断言

```rust
// mod.rs 末尾的静态断言

/// 编译期验证 Complex<f32> 的内存布局。
const _: () = {
    assert!(size_of::<Complex<f32>>() == 8);   // 2 * 4 bytes
    assert!(align_of::<Complex<f32>>() == 4);  // align_of::<f32>()

    assert!(size_of::<Complex<f64>>() == 16);  // 2 * 8 bytes
    assert!(align_of::<Complex<f64>>() == 8);  // align_of::<f64>()
};
```

### 4.3 与 C99 _Complex 的兼容性

C99 的 `_Complex float` 和 `_Complex double` 布局：

| C 类型 | 内存布局 | Rust 等价 |
|--------|----------|-----------|
| `_Complex float` | `[float, float]` | `Complex<f32>` |
| `_Complex double` | `[double, double]` | `Complex<f64>` |

**FFI 示例**:

```rust
// Rust 侧
#[repr(C)]
pub struct Complex<T> {
    re: T,
    im: T,
}

// 可以安全地与 C 代码交换指针
#[no_mangle]
pub extern "C" fn process_complex(z: *const Complex<f64>) -> Complex<f64> {
    unsafe {
        let z = *z;
        z * z  // 返回 z²
    }
}
```

```c
// C 侧
#include <complex.h>

typedef double _Complex cdouble_t;  // 或 _Complex double

cdouble_t process_complex(const cdouble_t* z);

int main() {
    double _Complex z = 3.0 + 4.0 * I;
    cdouble_t result = process_complex(&z);
    // result = (3+4i)² = -7 + 24i
}
```

### 4.4 数组布局等价性

`Complex<T>` 数组与交错实虚 `T` 数组内存等价：

```rust
// 这两种布局在内存中完全相同
let complex_arr: [Complex<f64>; 4] = [
    Complex::new(1.0, 2.0),
    Complex::new(3.0, 4.0),
    Complex::new(5.0, 6.0),
    Complex::new(7.0, 8.0),
];

// 内存: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
//       re   im   re   im   re   im   re   im

// 可安全 transmute
let flat: &[f64; 8] = unsafe {
    &*(complex_arr.as_ptr() as *const [f64; 8])
};
assert_eq!(flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
```

### 4.5 static_assert 验证方法

```rust
// 使用 const fn 和 const 泛型进行编译期验证

/// 验证内存布局的辅助 trait。
trait ValidateLayout {
    const SIZE_OK: bool;
    const ALIGN_OK: bool;
}

impl<T> ValidateLayout for Complex<T> {
    const SIZE_OK: bool = size_of::<Self>() == 2 * size_of::<T>();
    const ALIGN_OK: bool = align_of::<Self>() == align_of::<T>();
}

// 编译期断言宏
macro_rules! static_assert {
    ($cond:expr) => {
        const _: () = assert!($cond);
    };
}

// 使用
static_assert!(size_of::<Complex<f64>>() == 16);
static_assert!(align_of::<Complex<f64>>() == 8);
static_assert!(<Complex<f64> as ValidateLayout>::SIZE_OK);
static_assert!(<Complex<f64> as ValidateLayout>::ALIGN_OK);
```

---

## 5. 算术运算设计

### 5.1 Complex + Complex

```rust
impl<T: Float> Add for Complex<T> {
    type Output = Self;

    /// (a + bi) + (c + di) = (a+c) + (b+d)i
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}
```

### 5.2 Complex - Complex

```rust
impl<T: Float> Sub for Complex<T> {
    type Output = Self;

    /// (a + bi) - (c + di) = (a-c) + (b-d)i
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}
```

### 5.3 Complex * Complex

```rust
impl<T: Float> Mul for Complex<T> {
    type Output = Self;

    /// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}
```

**公式推导**:

```
(a + bi)(c + di) = ac + adi + bci + bdi²
                 = ac + adi + bci - bd    [因为 i² = -1]
                 = (ac - bd) + (ad + bc)i
```

### 5.4 Complex / Complex

```rust
impl<T: Float> Div for Complex<T> {
    type Output = Self;

    /// (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
    ///
    /// 通过乘以分母的共轭实现。
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}
```

**公式推导**:

```
(a + bi)     (a + bi)(c - di)     (ac + bd) + (bc - ad)i
--------  =  -----------------  =  ----------------------
(c + di)     (c + di)(c - di)           c² + d²
```

### 5.5 一元负号

```rust
impl<T: Float + Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    /// -(a + bi) = -a - bi
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}
```

---

## 6. 与实数的互操作

### 6.1 设计原则

| 规则 | 说明 |
|------|------|
| **同精度** | `Complex<f64>` 只能与 `f64` 互操作，`Complex<f32>` 只能与 `f32` 互操作 |
| **无整数** | 不支持 `Complex<f64> + i32`，需显式 `Complex<f64> + 32.0` |
| **无跨精度** | 不支持 `Complex<f64> + f32`，需显式 cast |

### 6.2 Complex + T（实数提升为复数）

```rust
impl<T: Float> Add<T> for Complex<T> {
    type Output = Self;

    /// (a + bi) + r = (a + r) + bi
    ///
    /// 实数 r 被隐式提升为 Complex(r, 0)。
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}
```

### 6.3 T + Complex（交换律）

```rust
impl<T: Float> Add<Complex<T>> for T {
    type Output = Complex<T>;

    /// r + (a + bi) = (r + a) + bi
    ///
    /// 交换律成立。
    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        Complex::new(self + rhs.re, rhs.im)
    }
}
```

### 6.4 Complex * T（标量乘法）

```rust
impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;

    /// (a + bi) * r = ar + bri
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}
```

### 6.5 T * Complex（交换律）

```rust
impl<T: Float> Mul<Complex<T>> for T {
    type Output = Complex<T>;

    /// r * (a + bi) = ar + bri
    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        Complex::new(self * rhs.re, self * rhs.im)
    }
}
```

### 6.6 Complex / T（标量除法）

```rust
impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;

    /// (a + bi) / r = (a/r) + (b/r)i
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
```

### 6.7 T / Complex（共轭乘法）

```rust
impl<T: Float> Div<Complex<T>> for T {
    type Output = Complex<T>;

    /// r / (a + bi) = r(a - bi) / (a² + b²)
    ///
    /// 分子乘以分母的共轭。
    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Complex::new(
            self * rhs.re / denom,
            -self * rhs.im / denom,
        )
    }
}
```

**公式推导**:

```
r         r(a - bi)       ra        -rbi
------  =  ----------  =  ------  +  ------
(a+bi)     (a+bi)(a-bi)    a²+b²      a²+b²
```

### 6.8 复合赋值运算符

```rust
impl<T: Float> AddAssign<T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re = self.re + rhs;
    }
}

impl<T: Float> SubAssign<T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re = self.re - rhs;
    }
}

impl<T: Float> MulAssign<T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.re = self.re * rhs;
        self.im = self.im * rhs;
    }
}

impl<T: Float> DivAssign<T> for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.re = self.re / rhs;
        self.im = self.im / rhs;
    }
}

// Complex + Complex 的复合赋值
impl<T: Float> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re = self.re + rhs.re;
        self.im = self.im + rhs.im;
    }
}

// ... SubAssign, MulAssign, DivAssign 同理
```

### 6.9 不支持的互操作

以下互操作**不实现**：

```rust
// ❌ 不支持：整数与复数
let z = Complex::new(1.0, 2.0);
let _ = z + 1i32;  // 编译错误

// ❌ 不支持：跨精度
let z64 = Complex::new(1.0f64, 2.0f64);
let _ = z64 + 3.0f32;  // 编译错误

// ✅ 正确用法：显式转换
let z = Complex::new(1.0, 2.0);
let _ = z + 1.0;       // 同精度浮点

let z64 = Complex::new(1.0f64, 2.0f64);
let _ = z64 + 3.0f64;  // 同精度
let _ = z64 + Complex::from(3.0f32);  // 显式转换后操作
```

---

## 7. 数学方法

### 7.1 conj() — 共轭

```rust
impl<T: Float> Complex<T> {
    /// 返回复共轭。
    ///
    /// conj(a + bi) = a - bi
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}
```

### 7.2 norm() — 模长（hypot 算法）

```rust
impl<T: Float> Complex<T> {
    /// 计算模长 |z| = sqrt(re² + im²)。
    ///
    /// 使用 hypot 算法避免中间结果溢出。
    ///
    /// # 算法
    ///
    /// hypot(a, b) 的实现：
    /// 1. 取绝对值：|a|, |b|
    /// 2. 找最大值：max(|a|, |b|)
    /// 3. 若 max == 0，返回 0（避免除零）
    /// 4. 缩放：a' = a/max, b' = b/max
    /// 5. 计算：max * sqrt(a'² + b'²)
    ///
    /// 这样 a'² + b'² ≤ 2，不会溢出。
    ///
    /// # 示例
    ///
    /// ```
    /// let z = Complex::new(3.0, 4.0);
    /// assert!((z.norm() - 5.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn norm(self) -> T {
        self.re.hypot(self.im)
    }

    /// 模长的平方（避免 sqrt 开销）。
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }
}
```

**hypot 算法详解**:

标准库 `f32::hypot` 和 `f64::hypot` 已经实现了稳定的算法：

```rust
// std 库内部实现（简化版）
fn hypot(mut a: f64, mut b: f64) -> f64 {
    a = a.abs();
    b = b.abs();
    
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    
    if a == 0.0 {
        return b;
    }
    
    let ratio = b / a;
    return a * (1.0 + ratio * ratio).sqrt();
}
```

**溢出避免示例**:

```rust
// 直接计算会溢出
let big = 1e200f64;
let overflow_result = (big * big + big * big).sqrt();  // inf

// hypot 不会溢出
let safe_result = big.hypot(big);  // 1.414...e200
```

### 7.3 arg() — 辐角

```rust
impl<T: Float> Complex<T> {
    /// 计算辐角（相位角）。
    ///
    /// arg(a + bi) = atan2(b, a)
    ///
    /// 返回值范围：(-π, π]
    ///
    /// # 特殊情况
    ///
    /// | re | im | 结果 |
    /// |----|----|----|
    /// | 0  | 0  | 0  |
    /// | >0 | 0  | 0  |
    /// | <0 | 0  | π  |
    /// | 0  | >0 | π/2 |
    /// | 0  | <0 | -π/2 |
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }
}
```

### 7.4 exp() — 指数函数

```rust
impl<T: Float> Complex<T> {
    /// 计算复指数 e^z。
    ///
    /// exp(a + bi) = e^a * (cos(b) + i*sin(b))
    ///
    /// # 公式
    ///
    /// - re = e^a * cos(b)
    /// - im = e^a * sin(b)
    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(
            exp_re * self.im.cos(),
            exp_re * self.im.sin(),
        )
    }
}
```

### 7.5 ln() — 自然对数

```rust
impl<T: Float> Complex<T> {
    /// 计算复自然对数（主值）。
    ///
    /// ln(z) = ln|z| + i*arg(z)
    ///
    /// # 注意
    ///
    /// - z = 0 时返回 -∞ + i*0
    /// - z 为负实数时，虚部为 π
    #[inline]
    pub fn ln(self) -> Self {
        Self::new(
            self.norm().ln(),
            self.arg(),
        )
    }
}
```

### 7.6 sqrt() — 平方根

```cpp
impl<T: Float> Complex<T> {
    /// 计算复平方根（主值）。
    ///
    /// # 算法
    ///
    /// sqrt(a + bi) 的计算：
    ///
    /// 1. r = |z| = sqrt(a² + b²)
    /// 2. 实部 = sqrt((r + a) / 2)
    /// 3. 虚部 = sign(b) * sqrt((r - a) / 2)
    ///
    /// 其中 sign(b) = 1 if b >= 0 else -1
    ///
    /// # 示例
    ///
    /// ```
    /// let z = Complex::new(-1.0, 0.0);
    /// let sqrt_z = z.sqrt();
    /// assert!((sqrt_z.re - 0.0).abs() < 1e-10);
    /// assert!((sqrt_z.im - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn sqrt(self) -> Self {
        let r = self.norm();
        
        if r == T::zero() {
            return Self::new(T::zero(), T::zero());
        }
        
        let half = T::one() / (T::one() + T::one());
        let sqrt_r_plus_a = ((r + self.re) * half).sqrt();
        let sqrt_r_minus_a = ((r - self.re) * half).sqrt();
        
        // sign(im) * sqrt((r - a) / 2)
        let im_sqrt = if self.im >= T::zero() {
            sqrt_r_minus_a
        } else {
            -sqrt_r_minus_a
        };
        
        Self::new(sqrt_r_plus_a, im_sqrt)
    }
}
```

### 7.7 from_polar() — 极坐标构造

```rust
impl<T: Float> Complex<T> {
    /// 从极坐标构造复数。
    ///
    /// # 参数
    ///
    /// - `r`: 模长（半径）
    /// - `theta`: 辐角（弧度）
    ///
    /// # 公式
    ///
    /// z = r * (cos(θ) + i*sin(θ))
    ///   = r*cos(θ) + i*r*sin(θ)
    ///
    /// # 示例
    ///
    /// ```
    /// let z = Complex::from_polar(1.0, std::f64::consts::FRAC_PI_2);
    /// assert!((z.re - 0.0).abs() < 1e-10);
    /// assert!((z.im - 1.0).abs() < 1e-10);  // i
    /// ```
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// 转换为极坐标 (r, theta)。
    #[inline]
    pub fn to_polar(self) -> (T, T) {
        (self.norm(), self.arg())
    }
}
```

---

## 8. 相等与近似比较

### 8.1 PartialEq 实现

```rust
impl<T: Float> PartialEq for Complex<T> {
    /// 逐分量比较相等性。
    ///
    /// # NaN 语义
    ///
    /// - NaN != NaN（遵循 IEEE 754）
    /// - 任何包含 NaN 的复数与任何复数（包括自身）都不相等
    ///
    /// # 示例
    ///
    /// ```
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(1.0, 2.0);
    /// assert!(a == b);
    ///
    /// let nan = Complex::new(f64::NAN, 0.0);
    /// assert!(nan != nan);  // NaN != NaN
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}
```

### 8.2 为什么不实现 Eq

```rust
// ❌ 不实现
impl<T: Float> Eq for Complex<T> {}
```

**原因**:
- `Eq` trait 要求自反性（`x == x` 对所有 `x` 成立）
- `Complex<f64>` 可能包含 NaN
- `NaN != NaN` 违反自反性
- 实现 `Eq` 会导致 unsafe 代码和 HashMap/HashSet 的未定义行为

### 8.3 为什么不实现 PartialOrd/Ord

```rust
// ❌ 不实现
impl<T: Float> PartialOrd for Complex<T> {}
impl<T: Float> Ord for Complex<T> {}
```

**原因**:
- 复数没有自然的全序关系
- 虽然可以定义字典序（先比较 re，再比较 im），但这在数学上没有意义
- 部分排序（如按模长）会丢失信息，且 `|z1| == |z2|` 不意味着 `z1 == z2`

### 8.4 approx_eq() — 近似相等

```rust
impl<T: Float> Complex<T> {
    /// 判断两个复数是否近似相等。
    ///
    /// # 参数
    ///
    /// - `other`: 另一个复数
    /// - `epsilon`: 容差阈值
    ///
    /// # 判断条件
    ///
    /// |re₁ - re₂| < ε && |im₁ - im₂| < ε
    ///
    /// # 注意
    ///
    /// - 此方法不适用于包含 NaN 或 Inf 的值
    /// - 对于相对误差判断，请使用 `approx_eq_rel`
    ///
    /// # 示例
    ///
    /// ```
    /// let a = Complex::new(1.0, 2.0);
    /// let b = Complex::new(1.0000001, 2.0000001);
    /// assert!(a.approx_eq(b, 1e-5));
    /// assert!(!a.approx_eq(b, 1e-8));
    /// ```
    #[inline]
    pub fn approx_eq(self, other: Self, epsilon: T) -> bool {
        (self.re - other.re).abs() < epsilon && (self.im - other.im).abs() < epsilon
    }

    /// 使用相对误差判断近似相等。
    ///
    /// # 公式
    ///
    /// |a - b| < max(|a|, |b|, 1.0) * ε
    #[inline]
    pub fn approx_eq_rel(self, other: Self, epsilon: T) -> bool {
        let abs_diff_re = (self.re - other.re).abs();
        let abs_diff_im = (self.im - other.im).abs();
        
        let max_re = self.re.abs().max(other.re.abs()).max(T::one());
        let max_im = self.im.abs().max(other.im.abs()).max(T::one());
        
        abs_diff_re < max_re * epsilon && abs_diff_im < max_im * epsilon
    }
}
```

---

## 9. 类型转换 (cast.rs)

### 9.1 Complex<f32> → Complex<f64>

```rust
impl From<Complex<f32>> for Complex<f64> {
    /// 精度提升：f32 → f64 无精度损失。
    #[inline]
    fn from(z: Complex<f32>) -> Self {
        Self::new(z.re as f64, z.im as f64)
    }
}
```

### 9.2 Complex<f64> → Complex<f32>

```rust
impl From<Complex<f64>> for Complex<f32> {
    /// 精度降低：f64 → f32 可能损失精度（IEEE 754 round-to-nearest-even）。
    #[inline]
    fn from(z: Complex<f64>) -> Self {
        Self::new(z.re as f32, z.im as f32)
    }
}
```

### 9.3 实数 → 复数

```rust
impl From<f32> for Complex<f32> {
    #[inline]
    fn from(re: f32) -> Self {
        Self::new(re, 0.0)
    }
}

impl From<f64> for Complex<f64> {
    #[inline]
    fn from(re: f64) -> Self {
        Self::new(re, 0.0)
    }
}
```

### 9.4 不支持的转换

```rust
// ❌ 整数 → 复数（不实现）
// 用户需要显式转换：
let z = Complex::from(42.0);  // 先转浮点

// ❌ 跨精度转换（不实现 From）
// 用户需要显式转换：
let z32 = Complex::new(1.0f32, 2.0f32);
let z64: Complex<f64> = z32.into();  // 显式
```

### 9.5 TryFrom for 损失性转换

对于可能丢失信息的转换，可考虑使用 `TryFrom`（可选实现）：

```rust
// 可选：提供更明确的错误处理
impl TryFrom<Complex<f64>> for Complex<f32> {
    type Error = PrecisionLossError;
    
    fn try_from(z: Complex<f64>) -> Result<Self, Self::Error> {
        // 检查是否超出 f32 范围
        if z.re.abs() > f32::MAX as f64 || z.im.abs() > f32::MAX as f64 {
            return Err(PrecisionLossError::Overflow);
        }
        Ok(Self::new(z.re as f32, z.im as f32))
    }
}
```

---

## 10. 与 element 模块的集成

### 10.1 Element trait 实现

```rust
// 在 src/element/mod.rs 中定义
pub trait Element:
    Copy
    + Clone
    + Default
    + Debug
    + Display
    + PartialEq
    + Send
    + Sync
{
    /// 加法单位元。
    fn zero() -> Self;
    
    /// 乘法单位元。
    fn one() -> Self;
}

// 在 src/complex/mod.rs 中实现
impl<T: Float> Element for Complex<T> {
    #[inline]
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    
    #[inline]
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }
}
```

### 10.2 Numeric trait 实现

```rust
// 在 src/element/numeric.rs 中定义
pub trait Numeric: Element
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
}

// Complex<T> 自动满足 Numeric（通过 blanket impl 或显式 impl）
impl<T: Float> Numeric for Complex<T> {}
```

### 10.3 ComplexScalar trait 实现

```rust
// 在 src/element/complex.rs 中定义
pub trait ComplexScalar: Numeric {
    /// 元素类型（f32 或 f64）。
    type Real: RealScalar;
    
    /// 返回实部。
    fn re(self) -> Self::Real;
    
    /// 返回虚部。
    fn im(self) -> Self::Real;
    
    /// 返回共轭。
    fn conj(self) -> Self;
    
    /// 返回模长（使用 hypot）。
    fn norm(self) -> Self::Real;
    
    /// 返回辐角。
    fn arg(self) -> Self::Real;
    
    /// 复指数。
    fn exp(self) -> Self;
    
    /// 复自然对数。
    fn ln(self) -> Self;
    
    /// 复平方根。
    fn sqrt(self) -> Self;
    
    /// 从极坐标构造。
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;
    
    /// 虚数单位 i。
    fn i() -> Self;
}

// 实现
impl ComplexScalar for Complex<f32> {
    type Real = f32;
    
    #[inline]
    fn re(self) -> Self::Real { self.re }
    
    #[inline]
    fn im(self) -> Self::Real { self.im }
    
    #[inline]
    fn conj(self) -> Self { Complex::new(self.re, -self.im) }
    
    #[inline]
    fn norm(self) -> Self::Real { self.re.hypot(self.im) }
    
    #[inline]
    fn arg(self) -> Self::Real { self.im.atan2(self.re) }
    
    #[inline]
    fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Complex::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }
    
    #[inline]
    fn ln(self) -> Self {
        Complex::new(self.norm().ln(), self.arg())
    }
    
    #[inline]
    fn sqrt(self) -> Self {
        // ... 实现（见第 7 节）
    }
    
    #[inline]
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self {
        Complex::new(r * theta.cos(), r * theta.sin())
    }
    
    #[inline]
    fn i() -> Self { Complex::new(0.0, 1.0) }
}

impl ComplexScalar for Complex<f64> {
    type Real = f64;
    
    // ... 同上，类型替换为 f64
}
```

### 10.4 与 RealScalar 的关系

```
          Element
             │
          Numeric
           /   \
    RealScalar  ComplexScalar
         │          │
       f32/f64  Complex<f32/f64>
```

- `RealScalar` 和 `ComplexScalar` 都继承自 `Numeric`
- 两者互不继承，各自独立
- 复数操作可以返回实数（如 `norm()` 返回 `f32`/`f64`）

---

## 11. 实现任务分解

### 任务列表

| ID | 任务 | 文件 | 预估时间 | 依赖 |
|----|------|------|----------|------|
| C1 | 定义 `Complex<T>` 结构体和 `new()` 构造方法 | `mod.rs` | 10 分钟 | 无 |
| C2 | 实现基础访问方法 `re()`, `im()`, `from_real()`, `from_imag()`, `i()` | `mod.rs` | 10 分钟 | C1 |
| C3 | 实现内存布局静态断言 | `mod.rs` | 10 分钟 | C1 |
| C4 | 实现 `PartialEq` trait | `mod.rs` | 10 分钟 | C1 |
| C5 | 实现 `approx_eq()` 和 `approx_eq_rel()` 方法 | `mod.rs` | 10 分钟 | C4 |
| C6 | 实现 Complex ± Complex 运算符 (`Add`, `Sub`) | `ops.rs` | 10 分钟 | C1 |
| C7 | 实现 Complex × Complex, Complex ÷ Complex 运算符 (`Mul`, `Div`) | `ops.rs` | 10 分钟 | C1 |
| C8 | 实现 Complex 与实数的互操作（6 种组合） | `ops.rs` | 10 分钟 | C6, C7 |
| C9 | 实现数学方法 `conj()`, `norm()`, `norm_sqr()`, `arg()` | `mod.rs` | 10 分钟 | C1 |
| C10 | 实现数学方法 `exp()`, `ln()`, `sqrt()`, `from_polar()` | `mod.rs` | 10 分钟 | C9 |
| C11 | 实现类型转换 `From<Complex<f32>>`, `From<Complex<f64>>`, `From<f32/f64>` | `cast.rs` | 10 分钟 | C1 |

### 依赖关系图

```
C1 ──┬── C2
     │
     ├── C3
     │
     ├── C4 ── C5
     │
     ├── C6 ──┬── C8
     │        │
     └── C7 ──┘
     
C1 ── C9 ── C10

C1 ── C11
```

### 并行执行策略

```
Phase 1: [C1]
            │
            ├──────────────┬──────────────┬──────────────┐
            ▼              ▼              ▼              ▼
Phase 2: [C2, C3]      [C4]          [C6, C7]        [C9, C11]
                          │              │
                          ▼              ▼
Phase 3:              [C5]          [C8]
                                         │
                                         ▼
Phase 4:                              [C10]
```

---

## 12. 设计决策记录

### ADR-001: 为什么不使用 num-complex

**状态**: 已采纳

**背景**:
Xenon 需要一个复数类型来支持科学计算。`num-complex` 是 Rust 生态中最成熟的复数库。

**决策**:
自定义 `Complex<T>` 类型，不依赖 `num-complex`。

**理由**:
1. **依赖最小化**: `num-complex` 依赖 `num-traits`，引入额外传递依赖
2. **trait 实现控制**: `num-complex` 实现了 `Eq`，但复数包含 NaN 时不满足 Eq 语义
3. **精度约束**: 需求要求严格限制同精度互操作，`num-complex` 更宽松
4. **FFI 兼容性验证**: 自定义类型可以精确控制并验证与 C99 `_Complex` 的兼容性
5. **与 Element 体系集成**: 自定义类型可以无缝集成到 Xenon 的 trait 层次

**后果**:
- 需要自己实现所有数学方法
- 增加维护成本
- 获得 API 的完全控制权

---

### ADR-002: 为什么不支持跨精度互操作

**状态**: 已采纳

**背景**:
`Complex<f64> + f32` 是否应该支持？

**决策**:
不支持。必须显式转换到同精度。

**理由**:
1. **类型安全**: 隐式精度转换容易引入难以调试的精度损失 bug
2. **性能可预测**: 隐式转换可能引入运行时开销
3. **与 Rust 哲学一致**: Rust 倾向于显式而非隐式
4. **NumPy 行为**: NumPy 也不支持不同 dtype 数组的隐式运算

**示例**:
```rust
// ❌ 不支持
let z: Complex<f64> = Complex::new(1.0, 2.0);
let _ = z + 3.0f32;  // 编译错误

// ✅ 显式转换
let _ = z + Complex::from(3.0f64);
let _ = z + f64::from(3.0f32);
```

---

### ADR-003: 为什么不实现 Eq/PartialOrd

**状态**: 已采纳

**背景**:
`num-complex` 实现了 `Eq`，是否应该遵循？

**决策**:
不实现 `Eq`、`PartialOrd`、`Ord`。

**理由**:
1. **Eq 的语义**: `Eq` 要求自反性（`x == x`），但 `NaN != NaN`
2. **数学正确性**: 复数没有自然全序
3. **集合语义**: 实现 `Eq` 会允许 `HashSet<Complex<f64>>`，但 NaN 会导致未定义行为
4. **PartialOrd 的问题**: 虽然可以实现（如字典序），但没有数学意义

**替代方案**:
- 提供 `approx_eq()` 用于近似比较
- 提供 `lexicographic_cmp()` 用于需要排序的场景（用户自行决定语义）

---

### ADR-004: norm() 使用 hypot 而非 sqrt(re² + im²)

**状态**: 已采纳

**背景**:
模长可以计算为 `sqrt(re*re + im*im)` 或使用 `hypot(re, im)`。

**决策**:
使用 `hypot(re, im)`。

**理由**:
1. **数值稳定性**: 当 `re` 或 `im` 很大时，`re*re` 可能溢出
2. **标准库保证**: `f32::hypot` 和 `f64::hypot` 已经实现了稳定的算法
3. **性能可接受**: hypot 的开销相对于安全性可以接受

**示例**:
```rust
// 直接计算：溢出
let big = 1e200f64;
let overflow = (big * big + big * big).sqrt();  // inf

// hypot：安全
let safe = big.hypot(big);  // 1.414...e200
```

---

### ADR-005: 是否提供 fmt::Display 实现

**状态**: 已采纳

**决策**:
实现 `Display` trait，格式为 `a + bi` 或 `a - bi`。

**格式规则**:
- `3 + 4i` → `"3+4i"`
- `3 - 4i` → `"3-4i"`
- `0 + 4i` → `"4i"`
- `3 + 0i` → `"3"`
- `0 + 0i` → `"0"`

**实现**:
```rust
impl<T: Float + Display> Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im == T::zero() {
            write!(f, "{}", self.re)
        } else if self.re == T::zero() {
            write!(f, "{}i", self.im)
        } else if self.im > T::zero() {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im)  // 负号自动包含
        }
    }
}
```

---

### ADR-006: 是否实现 serde 序列化

**状态**: 推迟

**背景**:
是否应该为 `Complex<T>` 实现 `Serialize`/`Deserialize`？

**决策**:
当前版本不实现。作为可选 feature 在未来版本添加。

**理由**:
1. **范围控制**: 需求文档明确范围外包含 serde 序列化
2. **依赖控制**: serde 是重量级依赖，违反最小依赖原则
3. **用户选择**: 用户可以自行实现或使用 `#[serde(remote)]` 派生

---

## 附录 A: 完整 API 速查

```rust
// 构造
Complex::new(re, im)           // 创建复数
Complex::from_real(re)         // 从实数创建
Complex::from_imag(im)         // 从虚数创建
Complex::from_polar(r, theta)  // 从极坐标创建
Complex::i()                   // 虚数单位

// 访问
z.re()                         // 实部
z.im()                         // 虚部
z.to_polar()                   // 转极坐标 (r, theta)

// 数学方法
z.conj()                       // 共轭
z.norm()                       // 模长（hypot）
z.norm_sqr()                   // 模长平方
z.arg()                        // 辐角
z.exp()                        // 指数
z.ln()                         // 对数
z.sqrt()                       // 平方根

// 比较
z == w                         // 逐分量相等（PartialEq）
z.approx_eq(w, eps)            // 近似相等（绝对误差）
z.approx_eq_rel(w, eps)        // 近似相等（相对误差）

// 算术运算符
z + w, z - w, z * w, z / w    // Complex ± Complex
z + r, z - r, z * r, z / r    // Complex ± 实数（同精度）
r + z, r - z, r * z, r / z    // 实数 ± Complex（同精度）
-z                            // 负号
z += w, z -= w, z *= w, z /= w // 复合赋值

// 类型转换
Complex::<f64>::from(z32)      // f32 → f64
Complex::<f32>::from(z64)      // f64 → f32
Complex::from(3.0)             // 实数 → 复数

// 常量（通过 trait 或关联常量）
Complex::zero()                // 0 + 0i
Complex::one()                 // 1 + 0i
```

---

## 附录 B: 测试用例清单

### 基础构造测试
- [ ] `new()` 创建正确
- [ ] `from_real()` 虚部为 0
- [ ] `from_imag()` 实部为 0
- [ ] `from_polar()` 正确转换
- [ ] `i()` 返回 `0 + 1i`

### 内存布局测试
- [ ] `size_of::<Complex<f32>>() == 8`
- [ ] `align_of::<Complex<f32>>() == 4`
- [ ] `size_of::<Complex<f64>>() == 16`
- [ ] `align_of::<Complex<f64>>() == 8`
- [ ] 数组 transmute 正确

### 算术运算测试
- [ ] Complex + Complex
- [ ] Complex - Complex
- [ ] Complex * Complex
- [ ] Complex / Complex
- [ ] Complex + T / T + Complex
- [ ] Complex * T / T * Complex
- [ ] Complex / T / T / Complex
- [ ] -Complex
- [ ] 复合赋值运算符

### 数学方法测试
- [ ] `conj()` 正确
- [ ] `norm()` 与手动计算一致
- [ ] `norm()` 大数不溢出
- [ ] `arg()` 范围正确 (-π, π]
- [ ] `exp()` 与手动计算一致
- [ ] `ln()` 与 `exp()` 互逆
- [ ] `sqrt()` 平方后接近原值
- [ ] `sqrt(-1) = i`

### 相等与比较测试
- [ ] 相等复数 `==` 返回 true
- [ ] 不等复数 `==` 返回 false
- [ ] `NaN == NaN` 返回 false
- [ ] `approx_eq()` 在阈值内返回 true
- [ ] `approx_eq()` 超阈值返回 false
- [ ] 不实现 `Eq`（编译期检查）
- [ ] 不实现 `PartialOrd`（编译期检查）

### 类型转换测试
- [ ] `Complex<f32>` → `Complex<f64>` 无损
- [ ] `Complex<f64>` → `Complex<f32>` 精度降低
- [ ] `f32` → `Complex<f32>` 正确
- [ ] `f64` → `Complex<f64>` 正确
- [ ] 整数不隐式转换（编译期检查）

### 边界情况测试
- [ ] 零的 `norm()` 为 0
- [ ] 零的 `arg()` 为 0
- [ ] 零的 `ln()` 为 -∞
- [ ] 零的 `sqrt()` 为 0
- [ ] Inf 参与运算
- [ ] NaN 参与运算
- [ ] 极大值 `norm()` 不溢出
- [ ] 极小值 `norm()` 正确

### FFI 兼容性测试
- [ ] 与 C `_Complex float` 布局兼容
- [ ] 与 C `_Complex double` 布局兼容
- [ ] 指针 transmute 正确

---

## 附录 C: 与 num-complex 的对比

| 特性 | Xenon Complex | num-complex |
|------|---------------|-------------|
| `#[repr(C)]` | ✅ | ✅ |
| `Eq` 实现 | ❌（NaN 安全） | ✅（有隐患） |
| 跨精度互操作 | ❌（显式转换） | ✅（部分支持） |
| 整数互操作 | ❌ | ✅ |
| 外部依赖 | 无 | num-traits |
| hypot norm | ✅ | ✅ |
| serde 支持 | ❌（未来） | ✅（feature） |
| bytemuck 支持 | ❌ | ✅（feature） |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
