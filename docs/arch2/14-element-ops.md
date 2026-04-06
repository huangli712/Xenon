# 逐元素运算模块设计

> 文档编号: 14 | 模块: `src/ops/element_wise.rs` | 阶段: Phase 3（API 模块）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `03-element-types.md`, `11-broadcast.md`

---

## 1. 模块定位

逐元素运算模块（Element-wise Operations）是 Renon 运算体系的基础层，提供对所有数值类型张量的逐元素数学运算。该模块通过 trait 约束在编译时区分不同元素类型支持的操作集合：算术运算适用于 `Numeric`，三角/指数/对数/取整运算适用于 `RealScalar`，复数运算适用于 `ComplexScalar`。

### 核心设计目标

| 目标 | 体现 |
|------|------|
| 编译时类型安全 | 通过 trait bound 在编译时阻止非法运算（如 bool 张量做算术） |
| 运算符重载 | `a + b`、`a * 2.0` 等自然语法，同时提供显式函数调用形式 |
| 性能分层分派 | 标量路径（通用）→ SIMD 路径（连续内存 + feature gate）→ 并行路径（大数组 + feature gate） |
| 广播集成 | 二元运算自动处理形状兼容性与广播 |
| 就地变体 | 提供 `_inplace` 方法避免不必要的内存分配 |
| 统一返回语义 | 非就地运算始终返回 `Tensor<A, D>`（Owned 存储） |

### 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 逐元素数学运算 | add, sub, mul, div, neg, abs, sin, cos, exp, ln, sqrt 等 | 矩阵运算（`ops/matrix.rs`） |
| 运算符重载 | `Add`, `Sub`, `Mul`, `Div`, `Neg` for TensorBase | 归约运算（`ops/reduction.rs`） |
| 标量广播 | tensor-scalar、scalar-tensor 运算 | 广播机制实现（`broadcast.rs`） |
| 就地运算 | `_inplace` 变体，操作 `TensorViewMut` | 集合运算（`ops/set_ops.rs`） |
| SIMD 分派入口 | 调用 `backend/scalar.rs` 或 `backend/simd.rs` | SIMD 具体实现（`backend/`） |

---

## 2. 文件位置

```
src/ops/
├── mod.rs              # pub mod element_wise; + 公共 re-export
├── element_wise.rs     # 本模块：逐元素运算全部实现
```

在 `src/lib.rs` 中的声明：

```rust
pub mod ops;

// 通过 ops/mod.rs re-export
pub use crate::ops::element_wise::*;
```

单文件设计理由：逐元素运算本质是"对每个元素做同一操作"的模式重复，按操作类别用 `// ── Section ──` 注释分组即可。文件预计 ~1200 行，处于合理范围。

---

## 3. 依赖关系

### 3.1 本模块的依赖（上游）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `TensorBase`, `Tensor`, `TensorView`, `TensorViewMut` | `tensor` | 操作对象类型 |
| `Dimension`, `IxDyn`, `Ix0` | `dimension` | 维度系统、广播结果维度 |
| `Element`, `Numeric`, `RealScalar`, `ComplexScalar` | `element` | trait bound 约束 |
| `Owned`, `ViewRepr`, `ViewMutRepr` | `storage` | 存储类型区分 |
| `LayoutFlags` | `layout` | 布局标志查询（连续性判定） |
| `TensorError`, `Result` | `error` | 错误处理 |
| `broadcast_with`, `broadcast_shape` | `broadcast` | 二元运算广播 |
| `core::ops::{Add, Sub, Mul, Div, Neg}` | `core` | 运算符重载 |
| `backend::scalar::*` | `backend/scalar` | 标量回退路径 |
| `backend::simd::*` (feature gated) | `backend/simd` | SIMD 加速路径 |

### 3.2 依赖本模块的下游模块

| 模块 | 使用方式 |
|------|----------|
| `ops/matrix.rs` | batch_add/batch_scale 复用逐元素运算 |
| `ops/reduction.rs` | 某些归约实现可复用 map 基础设施 |
| `conversion.rs` | `map`/`mapv`/`mapv_inplace` 操作 |
| 集成测试 | `tests/arithmetic.rs` |

### 3.3 依赖关系图

```
broadcast ──┐
element ────┤
tensor ─────┤──→ ops/element_wise.rs ──→ (用户代码)
layout ─────┤         │
error ──────┘         │
                      ▼
              backend/
              ├── scalar.rs  (always)
              ├── simd.rs    (feature = "simd")
              └── parallel.rs (feature = "parallel")
```

---

## 4. 公共 API 设计

### 4.1 辅助 trait：MapElem

所有逐元素运算共享一个内部 dispatch 机制。为避免代码重复，定义一个内部 trait 统一抽象"遍历张量的每个元素并应用函数"：

```rust
/// Internal trait for element-wise mapping over tensors.
///
/// This is the single point of dispatch for all element-wise operations.
/// Each operation (sin, exp, add, etc.) provides a closure to one of
/// the `map_*` methods, and this trait handles iteration strategy
/// (contiguous vs strided) and backend selection (scalar vs SIMD).
#[doc(hidden)]
pub(crate) trait MapElem<A, D>
where
    A: Element,
    D: Dimension,
{
    /// Returns the shape of the tensor.
    fn shape(&self) -> &D;

    /// Returns the strides of the tensor.
    fn strides(&self) -> &D;

    /// Returns the data start offset.
    fn offset(&self) -> usize;

    /// Returns the layout flags.
    fn layout_flags(&self) -> LayoutFlags;

    /// Returns a raw const pointer to the data start.
    fn as_ptr(&self) -> *const A;

    /// Returns the total number of elements.
    fn len(&self) -> usize {
        self.shape().size()
    }
}
```

所有 `TensorBase<S, D>` （where `S: RawStorage<Elem = A>`）自动实现 `MapElem`。

### 4.2 内部 dispatch 函数

```rust
/// Applies a unary function element-wise, returning a new owned tensor.
///
/// Dispatch strategy:
/// 1. If SIMD is available and data is contiguous → SIMD path
/// 2. Otherwise → scalar path
/// 3. If parallel is available and len >= PARALLEL_THRESHOLD → parallel path
fn apply_unary<A, D, F>(
    source: &impl MapElem<A, D>,
    f: F,
) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
    F: Fn(A) -> A;

/// Applies a unary function element-wise in-place.
///
/// # Safety
///
/// The caller must ensure the mutable pointer is valid and exclusive.
fn apply_unary_inplace<A, D, F>(
    target: &mut TensorViewMut<'_, A, D>,
    f: F,
)
where
    A: Element,
    D: Dimension,
    F: Fn(A) -> A;

/// Applies a binary function element-wise, returning a new owned tensor.
///
/// Handles shape compatibility check and broadcasting.
fn apply_binary<A, D, F>(
    lhs: &impl MapElem<A, D>,
    rhs: &impl MapElem<A, D>,
    f: F,
) -> Result<Tensor<A, D>>
where
    A: Element,
    D: Dimension,
    F: Fn(A, A) -> A;
```

### 4.3 算术运算（Binary）— add, sub, mul, div

#### 显式函数形式

```rust
// ── src/ops/element_wise.rs: Binary Arithmetic ──────────────────

/// Computes the element-wise addition of two tensors.
///
/// The two tensors must have compatible shapes (identical or broadcastable).
/// Returns a new owned tensor with the broadcast result shape.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor.
/// * `rhs` - Right-hand side tensor.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
///
/// # Examples
///
/// ```
/// use Renon::{Tensor, Ix2, add};
/// let a = Tensor::<f64, Ix2>::zeros([3, 4]);
/// let b = Tensor::<f64, Ix2>::ones([3, 4]);
/// let c = add(&a, &b)?;
/// assert_eq!(c[[0, 0]], 1.0);
/// ```
pub fn add<A, D>(lhs: &Tensor<A, D>, rhs: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary(lhs, rhs, |a, b| a + b)
}

/// Computes the element-wise subtraction of two tensors.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor (minuend).
/// * `rhs` - Right-hand side tensor (subtrahend).
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn sub<A, D>(lhs: &Tensor<A, D>, rhs: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary(lhs, rhs, |a, b| a - b)
}

/// Computes the element-wise multiplication of two tensors.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn mul<A, D>(lhs: &Tensor<A, D>, rhs: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary(lhs, rhs, |a, b| a * b)
}

/// Computes the element-wise division of two tensors.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn div<A, D>(lhs: &Tensor<A, D>, rhs: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary(lhs, rhs, |a, b| a / b)
}
```

#### 运算符重载（Tensor-Tensor）

```rust
// ── Operator overloads for TensorBase<Owned<A>, D> ──────────────

impl<A, D> core::ops::Add for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: Self) -> Self::Output {
        add(&self, &rhs).expect("shape mismatch in add")
    }
}

impl<A, D> core::ops::Sub for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        sub(&self, &rhs).expect("shape mismatch in sub")
    }
}

impl<A, D> core::ops::Mul for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        mul(&self, &rhs).expect("shape mismatch in mul")
    }
}

impl<A, D> core::ops::Div for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn div(self, rhs: Self) -> Self::Output {
        div(&self, &rhs).expect("shape mismatch in div")
    }
}
```

#### 引用形式运算符（&Tensor op &Tensor）

```rust
impl<'a, A, D> core::ops::Add<&'a Tensor<A, D>> for &Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: &'a Tensor<A, D>) -> Self::Output {
        add(self, rhs).expect("shape mismatch in add")
    }
}

impl<'a, A, D> core::ops::Sub<&'a Tensor<A, D>> for &Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn sub(self, rhs: &'a Tensor<A, D>) -> Self::Output {
        sub(self, rhs).expect("shape mismatch in sub")
    }
}

impl<'a, A, D> core::ops::Mul<&'a Tensor<A, D>> for &Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn mul(self, rhs: &'a Tensor<A, D>) -> Self::Output {
        mul(self, rhs).expect("shape mismatch in mul")
    }
}

impl<'a, A, D> core::ops::Div<&'a Tensor<A, D>> for &Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn div(self, rhs: &'a Tensor<A, D>) -> Self::Output {
        div(self, rhs).expect("shape mismatch in div")
    }
}
```

#### TensorView 参与运算

```rust
impl<'a, A, D> core::ops::Add<TensorView<'a, A, D>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: TensorView<'a, A, D>) -> Self::Output {
        apply_binary(&self, &rhs, |a, b| a + b).expect("shape mismatch in add")
    }
}

impl<'a, A, D> core::ops::Add<Tensor<A, D>> for TensorView<'a, A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: Tensor<A, D>) -> Self::Output {
        apply_binary(&self, &rhs, |a, b| a + b).expect("shape mismatch in add")
    }
}
// Sub, Mul, Div 同理（省略，模式完全相同）
```

#### 标量广播运算符

```rust
/// Computes element-wise addition of a tensor and a scalar.
///
/// The scalar is broadcast to match the tensor's shape.
pub fn add_scalar<A, D>(tensor: &Tensor<A, D>, scalar: A) -> Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    let scalar_view = scalar_to_view(scalar);
    apply_binary(tensor, &scalar_view, |a, b| a + b)
        .expect("scalar broadcast should never fail")
}

/// Computes element-wise subtraction: tensor - scalar.
pub fn sub_scalar<A, D>(tensor: &Tensor<A, D>, scalar: A) -> Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    let scalar_view = scalar_to_view(scalar);
    apply_binary(tensor, &scalar_view, |a, b| a - b)
        .expect("scalar broadcast should never fail")
}

/// Computes element-wise multiplication of a tensor and a scalar.
pub fn mul_scalar<A, D>(tensor: &Tensor<A, D>, scalar: A) -> Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    let scalar_view = scalar_to_view(scalar);
    apply_binary(tensor, &scalar_view, |a, b| a * b)
        .expect("scalar broadcast should never fail")
}

/// Computes element-wise division: tensor / scalar.
pub fn div_scalar<A, D>(tensor: &Tensor<A, D>, scalar: A) -> Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    let scalar_view = scalar_to_view(scalar);
    apply_binary(tensor, &scalar_view, |a, b| a / b)
        .expect("scalar broadcast should never fail")
}

// ── Scalar operator overloads ──────────────────────────────────

impl<A, D> core::ops::Add<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, scalar: A) -> Self::Output {
        add_scalar(&self, scalar)
    }
}

impl<A, D> core::ops::Sub<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn sub(self, scalar: A) -> Self::Output {
        sub_scalar(&self, scalar)
    }
}

impl<A, D> core::ops::Mul<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn mul(self, scalar: A) -> Self::Output {
        mul_scalar(&self, scalar)
    }
}

impl<A, D> core::ops::Div<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn div(self, scalar: A) -> Self::Output {
        div_scalar(&self, scalar)
    }
}

// ── Reverse scalar operators (scalar op Tensor) ────────────────

impl<A, D> core::ops::Mul<Tensor<A, D>> for A
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn mul(self, tensor: Tensor<A, D>) -> Self::Output {
        mul_scalar(&tensor, self)
    }
}

impl<A, D> core::ops::Add<Tensor<A, D>> for A
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, tensor: Tensor<A, D>) -> Self::Output {
        add_scalar(&tensor, self)
    }
}
```

#### 复合赋值运算符

```rust
impl<A, D> core::ops::AddAssign for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add_assign(&mut self, rhs: Self) {
        let mut view = self.view_mut();
        add_inplace(&mut view, &rhs).expect("shape mismatch in add_assign");
    }
}

impl<A, D> core::ops::SubAssign for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn sub_assign(&mut self, rhs: Self) {
        let mut view = self.view_mut();
        sub_inplace(&mut view, &rhs).expect("shape mismatch in sub_assign");
    }
}

impl<A, D> core::ops::MulAssign for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn mul_assign(&mut self, rhs: Self) {
        let mut view = self.view_mut();
        mul_inplace(&mut view, &rhs).expect("shape mismatch in mul_assign");
    }
}

impl<A, D> core::ops::DivAssign for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn div_assign(&mut self, rhs: Self) {
        let mut view = self.view_mut();
        div_inplace(&mut view, &rhs).expect("shape mismatch in div_assign");
    }
}
```

### 4.4 一元运算（Unary）— neg, abs, signum

```rust
// ── src/ops/element_wise.rs: Unary Arithmetic ──────────────────

/// Computes the element-wise negation of a tensor.
///
/// # Constraints
///
/// Element type must implement `Numeric` (i.e., supports `Neg`).
pub fn neg<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    apply_unary(tensor, |x| -x)
}

/// Computes the element-wise absolute value of a tensor.
///
/// # Constraints
///
/// Element type must implement `Numeric`. For signed integers,
/// this uses wrapping_abs. For floating-point types, prefer
/// `abs_real` which uses `RealScalar::abs`.
pub fn abs<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: Numeric + core::ops::Neg<Output = A>,
    D: Dimension,
{
    // For integer types: use wrapping_abs or conditional negation
    apply_unary(tensor, |x| {
        // This is a simplified version; actual impl will use
        // per-type specialization or a dedicated trait method.
        if x < A::zero() { -x } else { x }
    })
}

/// Computes the element-wise absolute value of a real-valued tensor.
///
/// Uses `RealScalar::abs`, which correctly handles floating-point
/// special values (NaN, Infinity, negative zero).
pub fn abs_real<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.abs())
}

/// Computes the element-wise signum function.
///
/// Returns -1.0 for negative, 0.0 for zero, 1.0 for positive inputs.
/// For NaN input, returns NaN.
pub fn signum<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| {
        if x.is_nan() {
            A::nan()
        } else if x < A::zero() {
            -A::one()
        } else if x > A::zero() {
            A::one()
        } else {
            A::zero()
        }
    })
}

// ── Unary operator overload ──────────────────────────────────

impl<A, D> core::ops::Neg for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn neg(self) -> Self::Output {
        neg(&self)
    }
}

impl<A, D> core::ops::Neg for &Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn neg(self) -> Self::Output {
        neg(self)
    }
}
```

### 4.5 三角函数 — sin, cos, tan, asin, acos, atan, sinh, cosh, tanh

```rust
// ── src/ops/element_wise.rs: Trigonometric Functions ───────────

/// Computes the element-wise sine (input in radians).
pub fn sin<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.sin())
}

/// Computes the element-wise cosine (input in radians).
pub fn cos<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.cos())
}

/// Computes the element-wise tangent (input in radians).
///
/// Returns +/-infinity for values where cos(x) = 0.
pub fn tan<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.tan())
}

/// Computes the element-wise arcsine.
///
/// Returns NaN for inputs outside `[-1, 1]`.
pub fn asin<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.asin())
}

/// Computes the element-wise arccosine.
///
/// Returns NaN for inputs outside `[-1, 1]`.
pub fn acos<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.acos())
}

/// Computes the element-wise arctangent.
pub fn atan<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.atan())
}

/// Computes the element-wise hyperbolic sine.
pub fn sinh<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.sinh())
}

/// Computes the element-wise hyperbolic cosine.
pub fn cosh<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.cosh())
}

/// Computes the element-wise hyperbolic tangent.
pub fn tanh<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.tanh())
}
```

### 4.6 指数/对数 — exp, exp2, ln, log, log2, log10

```rust
// ── src/ops/element_wise.rs: Exponential & Logarithm ───────────

/// Computes the element-wise exponential `e^x`.
pub fn exp<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.exp())
}

/// Computes the element-wise base-2 exponential `2^x`.
pub fn exp2<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.exp2())
}

/// Computes the element-wise natural logarithm `ln(x)`.
///
/// Returns NaN for negative inputs, -inf for zero.
pub fn ln<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.ln())
}

/// Computes the element-wise logarithm with a specified base.
///
/// Uses the change-of-base formula: `log_b(x) = ln(x) / ln(base)`.
///
/// # Panics
///
/// Panics if `base <= 0` or `base == 1`.
pub fn log<A, D>(tensor: &Tensor<A, D>, base: A) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    assert!(base > A::zero() && base != A::one(), "log base must be > 0 and != 1");
    let inv_base_ln = A::one() / base.ln();
    apply_unary(tensor, move |x| x.ln() * inv_base_ln)
}

/// Computes the element-wise base-2 logarithm `log2(x)`.
pub fn log2<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.log2())
}

/// Computes the element-wise base-10 logarithm `log10(x)`.
pub fn log10<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.log10())
}
```

### 4.7 幂运算 — sqrt, cbrt, pow, powi, powf

```rust
// ── src/ops/element_wise.rs: Power Functions ───────────────────

/// Computes the element-wise square root.
///
/// Returns NaN for negative inputs.
pub fn sqrt<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.sqrt())
}

/// Computes the element-wise cube root.
pub fn cbrt<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.cbrt())
}

/// Computes element-wise `x^y` where both x and y are tensors.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn pow<A, D>(base: &Tensor<A, D>, exp: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: RealScalar,
    D: Dimension,
{
    apply_binary(base, exp, |b, e| b.powf(e))
}

/// Computes element-wise `x^n` where n is an integer exponent (scalar).
pub fn powi<A, D>(tensor: &Tensor<A, D>, n: i32) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.powi(n))
}

/// Computes element-wise `x^n` where n is a floating-point exponent (scalar).
pub fn powf<A, D>(tensor: &Tensor<A, D>, n: A) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.powf(n))
}
```

### 4.8 取整函数 — floor, ceil, round, trunc

```rust
// ── src/ops/element_wise.rs: Rounding Functions ────────────────

/// Computes the element-wise floor (largest integer ≤ x).
pub fn floor<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.floor())
}

/// Computes the element-wise ceiling (smallest integer ≥ x).
pub fn ceil<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.ceil())
}

/// Computes the element-wise rounding to nearest integer.
///
/// Uses IEEE 754 round-to-nearest-even (banker's rounding).
pub fn round<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| x.round())
}

/// Computes the element-wise truncation (round toward zero).
pub fn trunc<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| {
        // trunc is not in RealScalar trait; implement via floor/ceil
        if x < A::zero() { x.ceil() } else { x.floor() }
    })
}
```

**设计说明**：`trunc` 不在 `RealScalar` trait 中定义（因为 `f32::trunc`/`f64::trunc` 可用）。实际实现时，考虑在 `RealScalar` trait 中添加 `trunc` 方法（作为后期扩展），或直接在 `apply_unary` 闭包中使用 `libm`/标准库。本设计文档中暂用 floor/ceil 组合实现，最终实现时直接委托 `f32::trunc`/`f64::trunc`。

### 4.9 比较与裁剪 — max, min, clamp

```rust
// ── src/ops/element_wise.rs: Comparison & Clamping ─────────────

/// Computes element-wise maximum of two tensors.
///
/// Uses NaN-propagating semantics: if either element is NaN, result is NaN.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn max<A, D>(lhs: &Tensor<A, D>, rhs: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: RealScalar,
    D: Dimension,
{
    apply_binary(lhs, rhs, |a, b| a.nan_max(b))
}

/// Computes element-wise minimum of two tensors.
///
/// Uses NaN-propagating semantics: if either element is NaN, result is NaN.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn min<A, D>(lhs: &Tensor<A, D>, rhs: &Tensor<A, D>) -> Result<Tensor<A, D>>
where
    A: RealScalar,
    D: Dimension,
{
    apply_binary(lhs, rhs, |a, b| a.nan_min(b))
}

/// Computes element-wise clamping: each element is clamped to `[lo, hi]`.
///
/// Returns NaN if any input element is NaN.
///
/// # Panics
///
/// Panics if `lo > hi`.
///
/// # Arguments
///
/// * `tensor` - Input tensor.
/// * `lo` - Lower bound (scalar).
/// * `hi` - Upper bound (scalar).
pub fn clamp<A, D>(tensor: &Tensor<A, D>, lo: A, hi: A) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    assert!(lo <= hi, "clamp: lo must be <= hi");
    apply_unary(tensor, |x| {
        if x.is_nan() {
            A::nan()
        } else if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    })
}
```

### 4.10 其他运算 — recip, recip_sqrt

```rust
// ── src/ops/element_wise.rs: Miscellaneous ─────────────────────

/// Computes the element-wise reciprocal `1/x`.
///
/// For `x = 0`, returns +/-infinity following IEEE 754 rules.
pub fn recip<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| A::one() / x)
}

/// Computes the element-wise reciprocal square root `1/sqrt(x)`.
///
/// Returns NaN for negative inputs, +infinity for zero.
pub fn recip_sqrt<A, D>(tensor: &Tensor<A, D>) -> Tensor<A, D>
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary(tensor, |x| A::one() / x.sqrt())
}
```

### 4.11 就地运算变体（In-place Variants）

就地运算直接修改 `TensorViewMut` 中的数据，不分配新内存。适用于已持有可变视图且无需保留原数据的场景。

```rust
// ── src/ops/element_wise.rs: In-place Operations ───────────────

/// Computes element-wise addition in-place: `target += rhs`.
///
/// The shapes must be identical (no broadcasting for in-place operations
/// on the target side; `rhs` may be broadcast).
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
pub fn add_inplace<A, D>(
    target: &mut TensorViewMut<'_, A, D>,
    rhs: &Tensor<A, D>,
) -> Result<()>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary_inplace(target, rhs, |a, b| a + b)
}

/// Computes element-wise subtraction in-place: `target -= rhs`.
pub fn sub_inplace<A, D>(
    target: &mut TensorViewMut<'_, A, D>,
    rhs: &Tensor<A, D>,
) -> Result<()>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary_inplace(target, rhs, |a, b| a - b)
}

/// Computes element-wise multiplication in-place: `target *= rhs`.
pub fn mul_inplace<A, D>(
    target: &mut TensorViewMut<'_, A, D>,
    rhs: &Tensor<A, D>,
) -> Result<()>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary_inplace(target, rhs, |a, b| a * b)
}

/// Computes element-wise division in-place: `target /= rhs`.
pub fn div_inplace<A, D>(
    target: &mut TensorViewMut<'_, A, D>,
    rhs: &Tensor<A, D>,
) -> Result<()>
where
    A: Numeric,
    D: Dimension,
{
    apply_binary_inplace(target, rhs, |a, b| a / b)
}

/// Applies element-wise negation in-place: `target = -target`.
pub fn neg_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: Numeric,
    D: Dimension,
{
    apply_unary_inplace(target, |x| -x)
}

/// Applies element-wise sine in-place.
pub fn sin_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.sin())
}

/// Applies element-wise cosine in-place.
pub fn cos_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.cos())
}

/// Applies element-wise exp in-place.
pub fn exp_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.exp())
}

/// Applies element-wise ln in-place.
pub fn ln_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.ln())
}

/// Applies element-wise sqrt in-place.
pub fn sqrt_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.sqrt())
}

/// Applies element-wise abs in-place (real-valued).
pub fn abs_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.abs())
}

/// Applies element-wise floor in-place.
pub fn floor_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.floor())
}

/// Applies element-wise ceil in-place.
pub fn ceil_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.ceil())
}

/// Applies element-wise round in-place.
pub fn round_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| x.round())
}

/// Applies element-wise clamp in-place.
pub fn clamp_inplace<A, D>(
    target: &mut TensorViewMut<'_, A, D>,
    lo: A,
    hi: A,
) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    assert!(lo <= hi, "clamp: lo must be <= hi");
    apply_unary_inplace(target, |x| {
        if x.is_nan() {
            A::nan()
        } else if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    })
}

/// Applies element-wise reciprocal in-place.
pub fn recip_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| A::one() / x)
}

/// Applies element-wise reciprocal sqrt in-place.
pub fn recip_sqrt_inplace<A, D>(target: &mut TensorViewMut<'_, A, D>) -> ()
where
    A: RealScalar,
    D: Dimension,
{
    apply_unary_inplace(target, |x| A::one() / x.sqrt())
}
```

### 4.12 辅助函数（Internal）

```rust
// ── Internal helpers ────────────────────────────────────────────

/// Creates a 0-dimensional view wrapping a scalar value.
///
/// This is used for scalar broadcast in tensor-scalar operations.
/// The resulting view has shape Ix0, which can be broadcast to any
/// dimension via the broadcast module.
fn scalar_to_view<A>(value: A) -> TensorView<'static, A, Ix0>
where
    A: Element,
{
    // SAFETY: We create a dangling-but-aligned pointer for a single element.
    // The actual data comes from the stack. Since Ix0 tensors have len=1
    // and we use a static lifetime, we need a different approach.
    //
    // Actual implementation: use a thread-local or static storage,
    // or create a Tensor<A, Ix0> and convert to view.
    //
    // Simpler approach: treat scalar as a Tensor<A, Ix0> and call apply_binary.
    todo!("Implemented using Owned allocation for a single element")
}

/// Applies a binary function in-place, mutating the target tensor.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes are incompatible.
fn apply_binary_inplace<A, D, F>(
    target: &mut TensorViewMut<'_, A, D>,
    rhs: &Tensor<A, D>,
    f: F,
) -> Result<()>
where
    A: Element,
    D: Dimension,
    F: Fn(A, A) -> A,
{
    // 1. Check shape compatibility (rhs may broadcast to target shape)
    // 2. Iterate target elements, applying f(target_elem, rhs_elem)
    // 3. For contiguous data: linear scan
    // 4. For strided data: nested index iteration
    todo!()
}
```

---

## 5. 内部实现设计

### 5.1 迭代策略选择

逐元素运算的迭代策略取决于内存布局：

```rust
/// Internal dispatch for unary element-wise operations.
fn apply_unary<A, D, F>(source: &impl MapElem<A, D>, f: F) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
    F: Fn(A) -> A,
{
    let len = source.len();
    let shape = source.shape().clone();

    // Allocate output with default F-order strides
    let mut output = Owned::<A>::uninitialized(len);
    let out_strides = crate::layout::compute_strides(&shape, Order::F);

    let src_flags = source.layout_flags();

    // Decision tree for iteration strategy
    if src_flags.is_contiguous() && len > 0 {
        // ── Contiguous path ──
        if can_dispatch_simd::<A>(len, src_flags) {
            #[cfg(feature = "simd")]
            {
                dispatch_simd_unary(source.as_ptr(), output.as_mut_ptr(), len, &f);
            }
            #[cfg(not(feature = "simd"))]
            {
                dispatch_scalar_contiguous(source.as_ptr(), output.as_mut_ptr(), len, &f);
            }
        } else {
            dispatch_scalar_contiguous(source.as_ptr(), output.as_mut_ptr(), len, &f);
        }

        // Parallel overlay (if feature enabled and array is large enough)
        #[cfg(feature = "parallel")]
        if len >= PARALLEL_THRESHOLD {
            dispatch_parallel_unary(source.as_ptr(), output.as_mut_ptr(), len, &f);
        }
    } else {
        // ── Strided path ──
        dispatch_scalar_strided(source, output.as_mut_ptr(), &f);
    }

    unsafe {
        TensorBase::from_storage_unchecked(output, shape, out_strides, 0)
    }
}
```

### 5.2 连续内存迭代（Contiguous Path）

```rust
/// Scalar contiguous iteration: linear scan over raw pointers.
///
/// This is the simplest and most common path. For contiguous data,
/// elements are laid out sequentially in memory, so we can iterate
/// with pointer arithmetic.
#[inline]
fn dispatch_scalar_contiguous<A, F>(
    src: *const A,
    dst: *mut A,
    len: usize,
    f: &F,
)
where
    A: Element,
    F: Fn(A) -> A,
{
    for i in 0..len {
        // SAFETY: caller guarantees [src, src + len) and [dst, dst + len) are valid
        unsafe {
            let val = src.add(i).read();
            dst.add(i).write(f(val));
        }
    }
}
```

### 5.3 跨步迭代（Strided Path）

```rust
/// Scalar strided iteration: iterate by index, computing offset via strides.
///
/// Used when data is not contiguous (e.g., after transpose or slicing).
/// This path is always scalar — SIMD requires contiguous data.
fn dispatch_scalar_strided<A, D, F>(
    source: &impl MapElem<A, D>,
    dst: *mut A,
    f: &F,
)
where
    A: Element,
    D: Dimension,
    F: Fn(A) -> A,
{
    let shape = source.shape();
    let strides = source.strides();
    let base = source.as_ptr();
    let ndim = shape.ndim();

    // Generic N-dimensional strided iteration
    // Uses a stack-allocated index counter
    let mut index = vec![0usize; ndim];

    let mut out_idx = 0usize;
    loop {
        // Compute source offset: base + Σ(index[i] * strides[i])
        let mut offset: isize = 0;
        for i in 0..ndim {
            offset += index[i] as isize * strides.slice_signed()[i];
        }

        // SAFETY: offset is within bounds by construction
        unsafe {
            let val = base.offset(offset).read();
            dst.add(out_idx).write(f(val));
        }
        out_idx += 1;

        // Increment multi-dimensional index (rightmost axis fastest for F-order output)
        let mut carry = true;
        for i in (0..ndim).rev() {
            if carry {
                index[i] += 1;
                if index[i] >= shape.slice()[i] {
                    index[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }
}
```

### 5.4 SIMD 分派

```rust
/// Checks whether SIMD dispatch is viable for the given operation.
#[cfg(feature = "simd")]
#[inline]
fn can_dispatch_simd<A>(len: usize, flags: LayoutFlags) -> bool {
    // SIMD requires:
    // 1. Data is contiguous (strides == 1)
    // 2. Array is large enough to fill at least one SIMD register
    // 3. Element type is SIMD-compatible (f32 or f64)
    flags.is_contiguous()
        && flags.is_aligned()
        && len >= core::mem::size_of::<A>() / core::mem::size_of::<u8>()
        && (core::mem::size_of::<A>() == 4 || core::mem::size_of::<A>() == 8)
}

#[cfg(feature = "simd")]
fn dispatch_simd_unary<A, F>(
    src: *const A,
    dst: *mut A,
    len: usize,
    f: &F,
)
where
    A: Element,
    F: Fn(A) -> A,
{
    use pulp::Arch;
    let arch = Arch::new();

    // The actual SIMD dispatch uses pulp's runtime feature detection.
    // Each operation (sin, exp, etc.) has a dedicated SIMD implementation
    // in backend/simd.rs, called via arch.dispatch(...).
    //
    // For simplicity, this skeleton shows the pattern:
    // 1. Process SIMD-width chunks
    // 2. Process remainder with scalar
    //
    // The concrete implementations live in backend/simd.rs and are
    // called per-operation (not generically), because SIMD operations
    // are instruction-specific.

    // Fallback: if no SIMD kernel exists for this operation,
    // use scalar contiguous path
    dispatch_scalar_contiguous(src, dst, len, f);
}
```

**SIMD 实现策略**：每个操作（sin, cos, exp, add 等）在 `backend/simd.rs` 中有专门的 SIMD kernel，通过 `pulp::Arch` 的 `dispatch` 方法分派到最佳指令集。逐元素运算模块不直接编写 SIMD intrinsics，而是调用 backend 提供的 kernel。

### 5.5 并行分派

```rust
#[cfg(feature = "parallel")]
fn dispatch_parallel_unary<A, F>(
    src: *const A,
    dst: *mut A,
    len: usize,
    f: &F,
)
where
    A: Element + Send + Sync,
    F: Fn(A) -> A + Sync,
{
    use rayon::prelude::*;

    // Split into chunks of at least PARALLEL_MIN_CHUNK elements
    let chunk_size = (len / rayon::current_num_threads())
        .max(PARALLEL_MIN_CHUNK);

    // SAFETY: caller guarantees [src, src+len) and [dst, dst+len) are valid
    // and non-overlapping. Chunks are non-overlapping by construction.
    unsafe {
        rayon::slice::from_raw_parts(src, len)
            .par_chunks(chunk_size)
            .zip(rayon::slice::from_raw_parts_mut(dst, len).par_chunks_mut(chunk_size))
            .for_each(|(src_chunk, dst_chunk)| {
                for (s, d) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                    *d = f(*s);
                }
            });
    }
}
```

### 5.6 二元运算与广播集成

```rust
fn apply_binary<A, D, F>(
    lhs: &impl MapElem<A, D>,
    rhs: &impl MapElem<A, D>,
    f: F,
) -> Result<Tensor<A, D>>
where
    A: Element,
    D: Dimension,
    F: Fn(A, A) -> A,
{
    let lhs_shape = lhs.shape().slice();
    let rhs_shape = rhs.shape().slice();

    // Step 1: Check if shapes are identical (fast path)
    if lhs_shape == rhs_shape {
        return Ok(apply_binary_same_shape(lhs, rhs, &f));
    }

    // Step 2: Attempt broadcast
    let result_shape = broadcast_shape(lhs_shape, rhs_shape)?;

    // Step 3: Broadcast both inputs to result shape
    //   - If an input's shape already matches result_shape, use it directly
    //   - Otherwise, create a broadcast view (stride=0 for broadcast dims)
    let lhs_broadcast = broadcast_to(lhs, &result_shape)?;
    let rhs_broadcast = broadcast_to(rhs, &result_shape)?;

    // Step 4: Apply operation on broadcast views
    Ok(apply_binary_same_shape(&lhs_broadcast, &rhs_broadcast, &f))
}

fn apply_binary_same_shape<A, D, F>(
    lhs: &impl MapElem<A, D>,
    rhs: &impl MapElem<A, D>,
    f: &F,
) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
    F: Fn(A, A) -> A,
{
    let len = lhs.len();
    let shape = lhs.shape().clone();
    let mut output = Owned::<A>::uninitialized(len);
    let out_strides = crate::layout::compute_strides(&shape, Order::F);

    // Both inputs must have the same shape at this point.
    // Dispatch similar to unary: contiguous vs strided.
    if lhs.layout_flags().is_contiguous() && rhs.layout_flags().is_contiguous() {
        dispatch_scalar_binary_contiguous(
            lhs.as_ptr(), rhs.as_ptr(), output.as_mut_ptr(), len, f,
        );
    } else {
        dispatch_scalar_binary_strided(lhs, rhs, output.as_mut_ptr(), f);
    }

    unsafe {
        TensorBase::from_storage_unchecked(output, shape, out_strides, 0)
    }
}
```

### 5.7 Feature Gate 总结

```rust
// Conditional compilation for backend dispatch
//
// Default (no features): scalar-only, contiguous + strided
// +simd: SIMD path for contiguous f32/f64 data
// +parallel: rayon parallel for large arrays
// +simd + parallel: SIMD within each parallel chunk

#[cfg(feature = "simd")]
use crate::backend::simd;

#[cfg(feature = "parallel")]
use crate::backend::parallel;

const PARALLEL_THRESHOLD: usize = 64 * 1024;  // 64K elements
const PARALLEL_MIN_CHUNK: usize = 4 * 1024;   // 4K elements per thread
```

### 5.8 操作与 Trait Bound 矩阵

| 操作 | Element Bound | 说明 |
|------|---------------|------|
| `add`, `sub`, `mul`, `div` | `Numeric` | 整数、浮点、复数均可用 |
| `neg` | `Numeric` | 需 `Neg<Output=Self>` |
| `abs` | `Numeric` (整数) / `RealScalar` (浮点) | 两个版本：`abs`（通用）、`abs_real`（浮点专用） |
| `signum` | `RealScalar` | 仅浮点有意义的语义 |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan` | `RealScalar` | 仅浮点 |
| `sinh`, `cosh`, `tanh` | `RealScalar` | 仅浮点 |
| `exp`, `exp2`, `ln`, `log2`, `log10`, `log` | `RealScalar` | 仅浮点 |
| `sqrt`, `cbrt` | `RealScalar` | 仅浮点 |
| `powi` | `RealScalar` | 整数幂 |
| `powf`, `pow` | `RealScalar` | 浮点幂 |
| `floor`, `ceil`, `round`, `trunc` | `RealScalar` | 仅浮点 |
| `max`, `min` | `RealScalar` | NaN 传播语义 |
| `clamp` | `RealScalar` | NaN 传播语义 |
| `recip`, `recip_sqrt` | `RealScalar` | 仅浮点 |

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 基础设施

- [ ] **T1: 模块骨架 + MapElem trait + apply_unary 函数框架**
  - 文件: `src/ops/element_wise.rs:1-80`
  - 内容: 模块导入、`MapElem` trait 定义、`MapElem` for `TensorBase<S, D>` impl、`apply_unary` 骨架（仅标量连续路径）
  - 测试: `test_map_elem_basic`（构造小 tensor，apply_unary 闭包验证）
  - 前置: tensor, element, layout, error, broadcast 模块完成
  - 预计: 10 min

- [ ] **T2: apply_binary 函数 + 广播集成**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `apply_binary` 框架、`apply_binary_same_shape`、形状兼容性检查与广播调用
  - 测试: `test_apply_binary_same_shape`, `test_apply_binary_broadcast`
  - 前置: T1, broadcast 模块
  - 预计: 10 min

- [ ] **T3: strided 迭代路径**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `dispatch_scalar_strided` 一元、`dispatch_scalar_binary_strided` 二元
  - 测试: `test_strided_unary`, `test_strided_binary`（使用转置后的张量验证）
  - 前置: T2
  - 预计: 10 min

- [ ] **T4: in-place 运算框架**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `apply_unary_inplace`、`apply_binary_inplace` 框架
  - 测试: `test_inplace_unary`, `test_inplace_binary`
  - 前置: T1, T2
  - 预计: 10 min

### 6.2 算术运算

- [ ] **T5: add/sub/mul/div 显式函数 + Tensor-Tensor 运算符重载**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `add()`, `sub()`, `mul()`, `div()` 函数、`impl Add/Sub/Mul/Div for Tensor<A, D>`
  - 测试: `test_add_tensors`, `test_sub_tensors`, `test_mul_tensors`, `test_div_tensors`, `test_shape_mismatch_error`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6: 引用运算符重载（&Tensor op &Tensor, TensorView 参与）**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `impl Add for &Tensor`、`impl Add<TensorView> for Tensor`、及其它组合
  - 测试: `test_ref_add`, `test_view_tensor_add`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: 标量广播运算符**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar`、`impl Add<A> for Tensor<A, D>`、反向运算符 `impl Mul<Tensor> for A`、`scalar_to_view` 辅助函数
  - 测试: `test_add_scalar`, `test_mul_scalar`, `test_scalar_mul_tensor`, `test_scalar_broadcast_shape`
  - 前置: T5
  - 预计: 10 min

- [ ] **T8: 复合赋值运算符（AddAssign 等）+ in-place 算术函数**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `impl AddAssign/SubAssign/MulAssign/DivAssign`、`add_inplace`, `sub_inplace`, `mul_inplace`, `div_inplace`
  - 测试: `test_add_assign`, `test_inplace_add`
  - 前置: T4, T5
  - 预计: 10 min

### 6.3 一元运算

- [ ] **T9: neg + 运算符重载**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `neg()` 函数、`impl Neg for Tensor`、`impl Neg for &Tensor`、`neg_inplace`
  - 测试: `test_neg`, `test_neg_operator`, `test_neg_inplace`
  - 前置: T1, T4
  - 预计: 10 min

- [ ] **T10: abs + signum**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `abs()`（Numeric 版）、`abs_real()`（RealScalar 版）、`signum()`、`abs_inplace`
  - 测试: `test_abs_integer`, `test_abs_real_float`, `test_signum`, `test_signum_nan`
  - 前置: T1, T4
  - 预计: 10 min

### 6.4 数学函数

- [ ] **T11: 三角函数（sin, cos, tan, asin, acos, atan）**
  - 文件: `src/ops/element_wise.rs`
  - 内容: 6 个三角函数、对应 in-place 变体
  - 测试: `test_sin`, `test_cos`, `test_tan`, `test_asin`, `test_acos`, `test_atan`, `test_sin_cos_inverse`
  - 前置: T1
  - 预计: 10 min

- [ ] **T12: 双曲函数（sinh, cosh, tanh）**
  - 文件: `src/ops/element_wise.rs`
  - 内容: 3 个双曲函数、对应 in-place 变体
  - 测试: `test_sinh`, `test_cosh`, `test_tanh`
  - 前置: T1
  - 预计: 5 min

- [ ] **T13: 指数/对数函数（exp, exp2, ln, log2, log10, log）**
  - 文件: `src/ops/element_wise.rs`
  - 内容: 6 个函数、in-place 变体
  - 测试: `test_exp`, `test_exp2`, `test_ln`, `test_log2`, `test_log10`, `test_log_base`, `test_exp_ln_roundtrip`
  - 前置: T1
  - 预计: 10 min

- [ ] **T14: 幂运算（sqrt, cbrt, pow, powi, powf）**
  - 文件: `src/ops/element_wise.rs`
  - 内容: 5 个函数、in-place 变体
  - 测试: `test_sqrt`, `test_cbrt`, `test_pow_tensor`, `test_powi`, `test_powf`, `test_sqrt_negative_nan`
  - 前置: T1, T2
  - 预计: 10 min

- [ ] **T15: 取整函数（floor, ceil, round, trunc）**
  - 文件: `src/ops/element_wise.rs`
  - 内容: 4 个函数、in-place 变体
  - 测试: `test_floor`, `test_ceil`, `test_round`, `test_trunc`, `test_floor_ceil_symmetry`
  - 前置: T1
  - 预计: 5 min

### 6.5 比较与杂项

- [ ] **T16: max, min, clamp**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `max()`, `min()`, `clamp()`、NaN 传播语义、in-place 变体
  - 测试: `test_max`, `test_min`, `test_clamp`, `test_max_nan_propagation`, `test_clamp_nan`
  - 前置: T2
  - 预计: 10 min

- [ ] **T17: recip, recip_sqrt**
  - 文件: `src/ops/element_wise.rs`
  - 内容: `recip()`, `recip_sqrt()`、in-place 变体
  - 测试: `test_recip`, `test_recip_sqrt`, `test_recip_zero_inf`
  - 前置: T1
  - 预计: 5 min

### 6.6 后端集成

- [ ] **T18: SIMD 分派集成**
  - 文件: `src/ops/element_wise.rs`, `src/backend/simd.rs`
  - 内容: `can_dispatch_simd`、SIMD 路径条件编译、调用 `backend::simd` kernel（初版仅 add/mul 的 f32/f64 SIMD）
  - 测试: `test_simd_add_f32`, `test_simd_add_f64`（对比标量路径结果一致）
  - 前置: T5, backend/simd 模块
  - 预计: 10 min

- [ ] **T19: 并行分派集成**
  - 文件: `src/ops/element_wise.rs`, `src/backend/parallel.rs`
  - 内容: `dispatch_parallel_unary`、并行阈值逻辑、条件编译
  - 测试: `test_parallel_add_large`（元素数 > 64K 时结果与串行一致）
  - 前置: T5, backend/parallel 模块
  - 预计: 10 min

### 6.7 集成与导出

- [ ] **T20: ops/mod.rs + lib.rs re-export**
  - 文件: `src/ops/mod.rs`, `src/lib.rs`
  - 内容: `pub mod element_wise;`、re-export 所有公共函数名
  - 测试: `test_public_api_import`（`use Renon::{add, sin, exp, ...}` 编译通过）
  - 前置: T1-T19
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试（`src/ops/element_wise.rs` 内 `#[cfg(test)] mod tests`）

#### 算术运算测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_add_tensors` | `[1,2,3] + [4,5,6] == [5,7,9]` |
| `test_sub_tensors` | `[5,7,9] - [1,2,3] == [4,5,6]` |
| `test_mul_tensors` | `[1,2,3] * [2,3,4] == [2,6,12]` |
| `test_div_tensors` | `[6.0,8.0,10.0] / [2.0,4.0,5.0] == [3.0,2.0,2.0]` |
| `test_add_shape_mismatch_error` | `[3,4] + [2,3]` 返回 BroadcastError |
| `test_add_broadcast` | `[3,4] + [1,4]` 成功广播 |
| `test_add_broadcast_scalar` | `[3,4] + 2.0` 全部元素加 2.0 |
| `test_sub_scalar_operator` | `tensor - 1.0` 等价于每个元素减 1.0 |
| `test_scalar_mul_tensor` | `2.0 * tensor` 等价于每个元素乘 2.0 |
| `test_add_assign` | `a += b` 修改 a 的数据 |
| `test_neg_operator` | `-tensor` 等价于 neg() |
| `test_neg_i32` | `[-1, 0, 1]` 取反后 `[1, 0, -1]` |

#### 三角函数测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_sin_zero` | `sin(0) == 0` |
| `test_cos_zero` | `cos(0) == 1` |
| `test_sin_pi` | `sin(π) ≈ 0`（浮点近似） |
| `test_cos_pi` | `cos(π) ≈ -1` |
| `test_tan_pi_quarter` | `tan(π/4) ≈ 1` |
| `test_asin_range` | `asin(0.5)` 在正确范围内 |
| `test_asin_out_of_range_nan` | `asin(2.0)` 为 NaN |
| `test_acos_out_of_range_nan` | `acos(-2.0)` 为 NaN |
| `test_sin_cos_inverse` | `asin(sin(x)) ≈ x` 对 x ∈ [-π/2, π/2] |
| `test_sinh_cosh_identity` | `cosh²(x) - sinh²(x) ≈ 1` |
| `test_tanh_range` | `tanh(x) ∈ [-1, 1]` 对所有有限 x |

#### 指数/对数测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_exp_zero` | `exp(0) == 1` |
| `test_exp_one` | `exp(1) ≈ e` |
| `test_ln_one` | `ln(1) == 0` |
| `test_exp_ln_roundtrip` | `exp(ln(x)) ≈ x` 对 x > 0 |
| `test_ln_negative_nan` | `ln(-1)` 为 NaN |
| `test_ln_zero_neg_inf` | `ln(0)` 为 -inf |
| `test_log2_powers` | `log2(8) == 3` |
| `test_log10_powers` | `log10(100) == 2` |
| `test_log_base` | `log(27, 3) ≈ 3` |
| `test_exp2` | `exp2(10) ≈ 1024` |

#### 幂运算测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_sqrt_four` | `sqrt(4) == 2` |
| `test_sqrt_negative_nan` | `sqrt(-1)` 为 NaN |
| `test_cbrt_negative` | `cbrt(-8) == -2`（实数立方根） |
| `test_powi` | `powi([2.0], 3) == [8.0]` |
| `test_powf` | `powf([4.0], 0.5) == [2.0]` |
| `test_pow_tensor_tensor` | `pow([2,3], [3,2]) == [8,9]` |
| `test_sqrt_square` | `x.sqrt() * x.sqrt() ≈ x` 对 x ≥ 0 |

#### 取整测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_floor` | `floor([1.7, -1.7]) == [1.0, -2.0]` |
| `test_ceil` | `ceil([1.2, -1.2]) == [2.0, -1.0]` |
| `test_round_half` | `round(1.5)` 为 2.0（IEEE 754 round-to-nearest-even） |
| `test_trunc` | `trunc([1.7, -1.7]) == [1.0, -1.0]` |
| `test_floor_ceil_symmetry` | `floor(x) <= x <= ceil(x)` 对所有有限 x |

#### 比较与杂项测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_max` | `max([1,3,2], [2,1,3]) == [2,3,3]` |
| `test_min` | `min([1,3,2], [2,1,3]) == [1,1,2]` |
| `test_max_nan_propagation` | `max(NaN, 1.0)` 为 NaN |
| `test_clamp` | `clamp([-1, 0.5, 2], 0, 1) == [0, 0.5, 1]` |
| `test_clamp_nan` | `clamp(NaN, 0, 1)` 为 NaN |
| `test_clamp_lo_gt_hi_panic` | `clamp(x, 1, 0)` panic |
| `test_recip` | `recip([2.0, 4.0]) == [0.5, 0.25]` |
| `test_recip_zero` | `recip(0.0)` 为 +inf |
| `test_recip_sqrt` | `recip_sqrt([4.0]) == [0.5]` |

#### In-place 运算测试

| 测试函数 | 验证内容 |
|---------|---------|
| `test_add_inplace` | `target += rhs` 正确修改 target |
| `test_sin_inplace` | `sin_inplace(target)` 正确修改 target |
| `test_inplace_preserves_shape` | inplace 操作后 shape 不变 |
| `test_inplace_strided` | 对非连续 view_mut 执行 inplace 正确 |

### 7.2 集成测试

位于 `tests/arithmetic.rs`：

| 测试分类 | 测试项 |
|---------|--------|
| **跨类型运算** | `test_add_i32_tensors`, `test_sin_f32_tensor`, `test_sin_f64_tensor` |
| **广播运算** | `test_binary_broadcast_1d_2d`, `test_scalar_broadcast_high_dim` |
| **View 参与** | `test_view_add_owned`, `test_slice_add` |
| **链式运算** | `test_chained_ops`（如 `(a + b) * c - d`） |
| **运算符语法** | `test_operator_syntax_add`, `test_operator_syntax_mul_scalar` |

### 7.3 边界测试

位于 `tests/edge_cases.rs`（逐元素部分）：

| 测试函数 | 验证内容 |
|---------|---------|
| `test_empty_tensor_add` | 空张量加法返回空张量，不 panic |
| `test_single_element_ops` | 单元素张量的所有运算正确 |
| `test_nan_propagation_unary` | `sin(NaN)` 为 NaN |
| `test_nan_propagation_binary` | `NaN + 1.0` 为 NaN |
| `test_inf_arithmetic` | `inf + inf == inf`, `inf - inf == NaN` |
| `test_large_tensor_ops` | 1M 元素张量运算不 OOM、结果正确 |
| `test_non_contiguous_ops` | 转置后张量运算结果与连续版本一致 |
| `test_broadcast_zero_stride_ops` | 广播张量参与运算正确 |

### 7.4 属性测试

位于 `tests/property/`（如适用）：

| 不变量 | 验证方式 |
|--------|---------|
| 加法交换律 | `add(a, b) == add(b, a)` |
| 加法结合律 | `add(add(a, b), c) == add(a, add(b, c))` |
| 乘法单位元 | `mul(a, ones) == a` |
| 加法零元 | `add(a, zeros) == a` |
| exp∘ln 恒等 | `exp(ln(x)) ≈ x` 对 x > 0 |
| sin∘asin 恒等 | `sin(asin(x)) ≈ x` 对 x ∈ [-1, 1] |
| sqrt 自反 | `sqrt(x) * sqrt(x) ≈ x` 对 x ≥ 0 |
| floor≤ceil | `floor(x) <= ceil(x)` 对所有有限 x |

### 7.5 基准测试

位于 `benches/element_ops.rs`：

| 基准 | 变量 |
|------|------|
| `bench_add_contiguous` | f32, f64 × [1K, 64K, 1M] × 标量/SIMD/并行 |
| `bench_mul_contiguous` | 同上 |
| `bench_sin_contiguous` | f32, f64 × [1K, 64K, 1M] |
| `bench_exp_contiguous` | f32, f64 × [1K, 64K, 1M] |
| `bench_add_strided` | f64 × 转置后 [1000×1000] |
| `bench_scalar_broadcast` | tensor+scalar × [64K, 1M] |
| `bench_inplace_vs_allocating` | add_inplace vs add × [1K, 64K] |

---

## 附录 A：完整公共函数清单

| 函数 | Trait Bound | 类型 | 就地变体 |
|------|-------------|------|----------|
| `add` | `Numeric` | Binary | `add_inplace` |
| `sub` | `Numeric` | Binary | `sub_inplace` |
| `mul` | `Numeric` | Binary | `mul_inplace` |
| `div` | `Numeric` | Binary | `div_inplace` |
| `neg` | `Numeric` | Unary | `neg_inplace` |
| `abs` | `Numeric` | Unary | — |
| `abs_real` | `RealScalar` | Unary | `abs_inplace` |
| `signum` | `RealScalar` | Unary | — |
| `sin` | `RealScalar` | Unary | `sin_inplace` |
| `cos` | `RealScalar` | Unary | `cos_inplace` |
| `tan` | `RealScalar` | Unary | — |
| `asin` | `RealScalar` | Unary | — |
| `acos` | `RealScalar` | Unary | — |
| `atan` | `RealScalar` | Unary | — |
| `sinh` | `RealScalar` | Unary | — |
| `cosh` | `RealScalar` | Unary | — |
| `tanh` | `RealScalar` | Unary | — |
| `exp` | `RealScalar` | Unary | `exp_inplace` |
| `exp2` | `RealScalar` | Unary | — |
| `ln` | `RealScalar` | Unary | `ln_inplace` |
| `log` | `RealScalar` | Unary (param) | — |
| `log2` | `RealScalar` | Unary | — |
| `log10` | `RealScalar` | Unary | — |
| `sqrt` | `RealScalar` | Unary | `sqrt_inplace` |
| `cbrt` | `RealScalar` | Unary | — |
| `pow` | `RealScalar` | Binary | — |
| `powi` | `RealScalar` | Unary (param) | — |
| `powf` | `RealScalar` | Unary (param) | — |
| `floor` | `RealScalar` | Unary | `floor_inplace` |
| `ceil` | `RealScalar` | Unary | `ceil_inplace` |
| `round` | `RealScalar` | Unary | `round_inplace` |
| `trunc` | `RealScalar` | Unary | — |
| `max` | `RealScalar` | Binary | — |
| `min` | `RealScalar` | Binary | — |
| `clamp` | `RealScalar` | Unary (2 param) | `clamp_inplace` |
| `recip` | `RealScalar` | Unary | `recip_inplace` |
| `recip_sqrt` | `RealScalar` | Unary | `recip_sqrt_inplace` |

**统计**：37 个公共函数 + 15 个就地变体 + 运算符重载（~20 个 impl）

## 附录 B：运算符重载清单

| 运算符 | LHS 类型 | RHS 类型 | Output | Bound |
|--------|---------|---------|--------|-------|
| `+` | `Tensor` | `Tensor` | `Tensor` | `Numeric` |
| `+` | `&Tensor` | `&Tensor` | `Tensor` | `Numeric` |
| `+` | `Tensor` | `TensorView` | `Tensor` | `Numeric` |
| `+` | `TensorView` | `Tensor` | `Tensor` | `Numeric` |
| `+` | `Tensor` | `A` (scalar) | `Tensor` | `Numeric` |
| `+` | `A` (scalar) | `Tensor` | `Tensor` | `Numeric` |
| `-` | 同 `+` 的所有组合 | | | `Numeric` |
| `*` | 同 `+` 的所有组合 | | | `Numeric` |
| `/` | 同 `+` 的所有组合 | | | `Numeric` |
| `-` (unary) | `Tensor` | — | `Tensor` | `Numeric` |
| `-` (unary) | `&Tensor` | — | `Tensor` | `Numeric` |
| `+=` | `Tensor` | `Tensor` | `()` | `Numeric` |
| `-=` | `Tensor` | `Tensor` | `()` | `Numeric` |
| `*=` | `Tensor` | `Tensor` | `()` | `Numeric` |
| `/=` | `Tensor` | `Tensor` | `()` | `Numeric` |

## 附录 C：性能分派决策树

```
逐元素运算请求
    │
    ├── 元素数 == 0 → 立即返回空 Tensor
    │
    ├── 数据是否连续？(layout_flags.is_contiguous())
    │   ├── 否 → 标量跨步路径（不论规模）
    │   │
    │   └── 是 → 检查 SIMD 可用性
    │       │
    │       ├── #[cfg(feature = "simd")]
    │       │   ├── 类型为 f32/f64 且数据对齐 → SIMD 路径
    │       │   └── 否 → 标量连续路径
    │       │
    │       └── #[cfg(not(feature = "simd"))]
    │           └── 标量连续路径
    │
    └── 并行叠加（在 SIMD 或标量路径之上）
        │
        ├── #[cfg(feature = "parallel")]
        │   ├── 元素数 ≥ PARALLEL_THRESHOLD (64K)
        │   │   └── 按连续块分给 rayon 线程池
        │   │       └── 每线程内部使用 SIMD 或标量路径
        │   └── 否 → 单线程执行
        │
        └── #[cfg(not(feature = "parallel"))]
            └── 单线程执行
```
