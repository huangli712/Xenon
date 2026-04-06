# 逐元素运算模块设计文档

> **文档版本**: v1.0  
> **最后更新**: 2026-03-28  
> **模块路径**: `src/ops/`  
> **需求来源**: require-v18.md §10.1 + §13.2-13.3

---

## 1. 模块概述

### 1.1 设计哲学

逐元素运算（Elementwise Operations）是 Senon 数值计算的核心能力之一，设计遵循以下原则：

- **类型安全边界**: 算术运算仅支持数值类型（`Numeric`），`bool` 被显式排除，编译时阻止无效操作
- **广播透明集成**: 所有二元运算符自动支持广播，用户无需显式调用广播函数
- **存储模式无关**: 运算对 `Tensor`、`TensorView`、`TensorViewMut`、`ArcTensor` 统一工作
- **性能分层执行**: 根据数据规模和布局自动选择 SIMD/并行/标量路径
- **原地操作支持**: 提供 `_inplace` 变体避免不必要的内存分配

### 1.2 在架构中的位置

```
┌─────────────────────────────────────────────────────────┐
│                    用户层 API                            │
│    tensor + tensor, tensor * scalar, tensor.sin()       │
└─────────────────────┬───────────────────────────────────┘
                      │ 运算符 trait + 方法调用
┌─────────────────────▼───────────────────────────────────┐
│              逐元素运算模块 (本模块)                      │
│  elementwise.rs: map, mapv, mapv_inplace, zip_with      │
│  arithmetic.rs: Add, Sub, Mul, Div, Rem, Neg, Not       │
│  comparison.rs: is_close, allclose, clip, clamp         │
└─────────────────────┬───────────────────────────────────┘
                      │ 依赖
┌─────────────────────▼───────────────────────────────────┐
│            底层模块                                      │
│  iter (迭代器) | broadcast (广播) | simd (向量化)        │
│  element (类型约束) | tensor (核心结构)                  │
└─────────────────────────────────────────────────────────┘
```

### 1.3 运算分类

| 类别 | 操作 | Trait 约束 | 文件 |
|------|------|-----------|------|
| 映射操作 | map, mapv, mapv_inplace | `Element` | `elementwise.rs` |
| 多元操作 | zip_with, apply | `Element` | `elementwise.rs` |
| 算术运算 | add, sub, mul, div | `Numeric` | `arithmetic.rs` |
| 运算符重载 | `+`, `-`, `*`, `/`, `+=`, etc. | `Numeric` | `arithmetic.rs` |
| 一元运算 | neg (`-`), not (`!`) | `Numeric` / `Element` | `arithmetic.rs` |
| 三角函数 | sin, cos, tan, asin, acos, atan | `RealScalar` | `elementwise.rs` |
| 指数/对数 | exp, ln, log2, log10 | `RealScalar` | `elementwise.rs` |
| 数值函数 | abs, sign, floor, ceil, round, square, reciprocal, pow | `RealScalar` / `Numeric` | `elementwise.rs` |
| 比较运算 | is_close, allclose, clip, clamp | `RealScalar` | `comparison.rs` |

### 1.4 bool 类型处理

**关键约束**: `bool` 类型**仅支持** `Element` trait，**不实现** `Numeric`。

| 运算类别 | bool 支持 | 原因 |
|----------|----------|------|
| 四则运算 (add/sub/mul/div) | ❌ | 布尔值四则运算无数学意义 |
| 三角函数 | ❌ | 需要 `RealScalar` |
| 指数/对数 | ❌ | 需要 `RealScalar` |
| 一元 neg (`-`) | ❌ | 需要 `Numeric` |
| 一元 not (`!`) | ✅ | 逻辑取反，仅需 `Element` |
| map/mapv | ✅ | 通用映射 |
| clip/clamp | ❌ | 需要比较语义 |
| is_close/allclose | ❌ | 需要数值比较 |

---

## 2. 文件结构

```
src/ops/
├── mod.rs             # 模块入口，re-export 公开 trait 和函数
├── elementwise.rs     # 逐元素运算核心（map, zip_with, 数学函数）
├── arithmetic.rs      # 运算符重载（Add, Sub, Mul, Div, Rem, Neg, Not）
└── comparison.rs      # 比较运算（is_close, allclose, clip, clamp）
```

### 2.1 各文件职责

| 文件 | 职责 | 核心内容 |
|------|------|----------|
| `mod.rs` | 模块组织与导出 | re-export `Elementwise`, `ZipWith`, 数学函数 |
| `elementwise.rs` | 映射与数学函数 | `map`, `mapv`, `mapv_inplace`, `zip_with`, `apply`, 数学方法 |
| `arithmetic.rs` | 运算符重载 | `impl Add/Sub/Mul/Div/Rem for TensorBase`, 复合赋值, Neg/Not |
| `comparison.rs` | 比较与裁剪 | `is_close`, `allclose`, `clip`, `clamp` |

### 2.2 模块依赖

| 依赖模块 | 用途 |
|---------|------|
| `crate::tensor` | `TensorBase<S, D>` 核心类型 |
| `crate::element` | `Element`, `Numeric`, `RealScalar` trait 约束 |
| `crate::iter` | `Elements`, `Zip` 迭代器 |
| `crate::broadcast` | 广播规则实现 |
| `crate::simd` (可选) | SIMD 向量化路径 |
| `crate::parallel` (可选) | 并行执行路径 |

---

## 3. map/mapv/mapv_inplace 设计

### 3.1 方法签名

```rust
// elementwise.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 逐元素映射（通过引用）
    ///
    /// 对每个元素调用闭包，返回新分配的 Tensor。
    /// 闭包接收元素引用，返回新值。
    ///
    /// # Examples
    /// ```
    /// let a = tensor!([1.0, 2.0, 3.0]);
    /// let b = a.map(|x| x * 2.0);
    /// assert_eq!(b, tensor!([2.0, 4.0, 6.0]));
    /// ```
    pub fn map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element,
        F: FnMut(&A) -> B,
    {
        // 实现...
    }

    /// 逐元素映射（通过值）
    ///
    /// 对每个元素调用闭包，返回新分配的 Tensor。
    /// 闭包接收元素值（Copy），返回新值。
    ///
    /// # Examples
    /// ```
    /// let a = tensor!([1.0, 2.0, 3.0]);
    /// let b = a.mapv(|x| x.sqrt());
    /// assert_eq!(b, tensor!([1.0, 1.414, 1.732]));
    /// ```
    pub fn mapv<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element,
        F: FnMut(A) -> B,
    {
        // 实现...
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// 原地逐元素映射
    ///
    /// 对每个元素调用闭包，原地修改。
    /// 闭包接收元素值，返回同类型新值。
    ///
    /// # Examples
    /// ```
    /// let mut a = tensor!([1.0, 2.0, 3.0]);
    /// a.mapv_inplace(|x| x * x);
    /// assert_eq!(a, tensor!([1.0, 4.0, 9.0]));
    /// ```
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: FnMut(A) -> A,
    {
        // 实现...
    }
}
```

### 3.2 语义对比

| 方法 | 闭包参数 | 返回类型 | 内存分配 | 适用场景 |
|------|---------|---------|---------|---------|
| `map` | `&A` | `Tensor<B, D>` | 新分配 | 需要引用（如调用方法）、类型转换 |
| `mapv` | `A`（值） | `Tensor<B, D>` | 新分配 | 简单计算、类型转换 |
| `mapv_inplace` | `A`（值） | `()`（原地修改） | 无 | 避免分配、修改现有数组 |

### 3.3 与迭代器的关系

```rust
// map 等价于
fn map<B, F>(&self, f: F) -> Tensor<B, D>
where
    B: Element,
    F: FnMut(&A) -> B,
{
    let mut result = Tensor::zeros(self.shape());
    for (src, dst) in self.iter().zip(result.iter_mut()) {
        *dst = f(src);
    }
    result
}

// mapv 等价于
fn mapv<B, F>(&self, f: F) -> Tensor<B, D>
where
    B: Element,
    F: FnMut(A) -> B,
{
    let mut result = Tensor::zeros(self.shape());
    for (src, dst) in self.iter().zip(result.iter_mut()) {
        *dst = f(*src);
    }
    result
}

// mapv_inplace 等价于
fn mapv_inplace<F>(&mut self, f: F)
where
    F: FnMut(A) -> A,
{
    for elem in self.iter_mut() {
        *elem = f(*elem);
    }
}
```

### 3.4 ndarray 语义参考

Senon 的 `map`/`mapv`/`mapv_inplace` 语义与 ndarray crate 保持一致：

| ndarray 方法 | Senon 对应 | 说明 |
|-------------|-----------|------|
| `map(|x| f(*x))` | `map(|x| f(x))` | 引用版本 |
| `mapv(|x| f(x))` | `mapv(|x| f(x))` | 值版本 |
| `mapv_inplace(|x| f(x))` | `mapv_inplace(|x| f(x))` | 原地版本 |
| `map_into()` | N/A | ndarray 特有，Senon 用所有权转移替代 |

---

## 4. zip_with/apply 设计

### 4.1 方法签名

```rust
// elementwise.rs

/// 二元逐元素操作
///
/// 对两个数组的对应元素调用二元闭包，返回新 Tensor。
/// 支持广播：若形状不匹配但可广播，自动扩展后操作。
pub fn zip_with<A, B, C, D, E, F>(
    a: &TensorBase<A, D>,
    b: &TensorBase<B, E>,
    f: F,
) -> Result<Tensor<C, <D as BroadcastDim<E>>::Output>, BroadcastError>
where
    A: Storage,
    B: Storage,
    C: Element,
    D: Dimension,
    E: Dimension,
    F: FnMut(A::Elem, B::Elem) -> C,
{
    // 广播后逐元素应用
}

/// N 元逐元素操作
///
/// 对 N 个数组的对应元素调用 N 元闭包。
/// 使用 `Zip` 迭代器实现。
pub fn apply<'a, A, D, F>(tensors: &[&'a TensorBase<A, D>], f: F) -> Result<Tensor<A::Elem, D>, ShapeError>
where
    A: Storage,
    D: Dimension,
    F: FnMut(&[A::Elem]) -> A::Elem,
{
    // 验证所有形状一致，然后逐元素应用
}
```

### 4.2 zip_with 广播语义

```rust
// 示例：广播加法
let a = Tensor::from_shape_vec([3, 1], vec![1.0, 2.0, 3.0]).unwrap();  // shape: [3, 1]
let b = Tensor::from_shape_vec([1, 4], vec![0.1, 0.2, 0.3, 0.4]).unwrap();  // shape: [1, 4]

let c = zip_with(&a, &b, |x, y| x + y).unwrap();
// 结果 shape: [3, 4]
// c[[i, j]] = a[[i, 0]] + b[[0, j]]
```

### 4.3 与 Zip 迭代器的关系

```rust
// zip_with 内部使用 iter::Zip
pub fn zip_with<A, B, C, D, E, F>(...) -> Result<Tensor<C, ...>, ...>
{
    // 1. 计算广播后的形状
    let output_shape = a.shape().broadcast_shape(b.shape())?;
    
    // 2. 广播两个输入
    let a_broadcast = a.broadcast(output_shape)?;
    let b_broadcast = b.broadcast(output_shape)?;
    
    // 3. 使用 Zip 迭代器逐元素应用
    let mut result = Tensor::zeros(output_shape);
    Zip::from(&mut result)
        .and(&a_broadcast)
        .and(&b_broadcast)
        .for_each(|r, &a_val, &b_val| {
            *r = f(a_val, b_val);
        });
    
    Ok(result)
}
```

### 4.4 apply 多元语义

```rust
// 示例：三个数组的加权和
let a = tensor![[1.0, 2.0], [3.0, 4.0]];
let b = tensor![[0.5, 0.5], [0.5, 0.5]];
let c = tensor![[0.1, 0.1], [0.1, 0.1]];

let weighted_sum = apply(&[&a, &b, &c], |vals| {
    vals[0] * 0.5 + vals[1] * 0.3 + vals[2] * 0.2
}).unwrap();
```

---

## 5. 运算符重载 (arithmetic.rs)

### 5.1 完整 impl 列表

#### 5.1.1 二元运算符

| 运算符 | Trait | LHS | RHS | Output | 广播 |
|--------|-------|-----|-----|--------|------|
| `+` | `Add` | `Tensor<A, D>` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ |
| `+` | `Add` | `Tensor<A, D>` | `TensorView<'_, A, E>` | `Tensor<A, ...>` | ✓ |
| `+` | `Add` | `Tensor<A, D>` | `A` (标量) | `Tensor<A, D>` | 标量广播 |
| `+` | `Add` | `A` (标量) | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 |
| `-` | `Sub` | `Tensor<A, D>` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ |
| `-` | `Sub` | `Tensor<A, D>` | `TensorView<'_, A, E>` | `Tensor<A, ...>` | ✓ |
| `-` | `Sub` | `Tensor<A, D>` | `A` (标量) | `Tensor<A, D>` | 标量广播 |
| `-` | `Sub` | `A` (标量) | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 |
| `*` | `Mul` | `Tensor<A, D>` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ |
| `*` | `Mul` | `Tensor<A, D>` | `TensorView<'_, A, E>` | `Tensor<A, ...>` | ✓ |
| `*` | `Mul` | `Tensor<A, D>` | `A` (标量) | `Tensor<A, D>` | 标量广播 |
| `*` | `Mul` | `A` (标量) | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 |
| `/` | `Div` | `Tensor<A, D>` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ |
| `/` | `Div` | `Tensor<A, D>` | `TensorView<'_, A, E>` | `Tensor<A, ...>` | ✓ |
| `/` | `Div` | `Tensor<A, D>` | `A` (标量) | `Tensor<A, D>` | 标量广播 |
| `/` | `Div` | `A` (标量) | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 |
| `%` | `Rem` | `Tensor<A, D>` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ |
| `%` | `Rem` | `Tensor<A, D>` | `A` (标量) | `Tensor<A, D>` | 标量广播 |
| `%` | `Rem` | `A` (标量) | `Tensor<A, D>` | `Tensor<A, D>` | 标量广播 |

**约束**: 所有二元运算要求 `A: Numeric`（排除 `bool`）。

#### 5.1.2 复合赋值运算符

| 运算符 | Trait | LHS | RHS | 广播 |
|--------|-------|-----|-----|------|
| `+=` | `AddAssign` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ (仅 RHS 广播) |
| `+=` | `AddAssign` | `Tensor<A, D>` | `TensorView<'_, A, E>` | ✓ (仅 RHS 广播) |
| `+=` | `AddAssign` | `Tensor<A, D>` | `A` (标量) | 标量广播 |
| `-=` | `SubAssign` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ (仅 RHS 广播) |
| `-=` | `SubAssign` | `Tensor<A, D>` | `TensorView<'_, A, E>` | ✓ (仅 RHS 广播) |
| `-=` | `SubAssign` | `Tensor<A, D>` | `A` (标量) | 标量广播 |
| `*=` | `MulAssign` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ (仅 RHS 广播) |
| `*=` | `MulAssign` | `Tensor<A, D>` | `TensorView<'_, A, E>` | ✓ (仅 RHS 广播) |
| `*=` | `MulAssign` | `Tensor<A, D>` | `A` (标量) | 标量广播 |
| `/=` | `DivAssign` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ (仅 RHS 广播) |
| `/=` | `DivAssign` | `Tensor<A, D>` | `TensorView<'_, A, E>` | ✓ (仅 RHS 广播) |
| `/=` | `DivAssign` | `Tensor<A, D>` | `A` (标量) | 标量广播 |
| `%=` | `RemAssign` | `Tensor<A, D>` | `Tensor<A, D>` | ✓ (仅 RHS 广播) |
| `%=` | `RemAssign` | `Tensor<A, D>` | `A` (标量) | 标量广播 |

**约束**: 复合赋值要求 `A: Numeric`，且 LHS 必须可写（`Tensor` 或 `TensorViewMut`）。

#### 5.1.3 存储模式组合矩阵

| LHS 存储 | RHS 存储 | 返回存储 | 说明 |
|----------|----------|----------|------|
| `Owned<A>` | `Owned<A>` | `Owned<A>` | 两个拥有型相加 |
| `Owned<A>` | `ViewRepr<&A>` | `Owned<A>` | 拥有型 + 视图 |
| `Owned<A>` | `ViewMutRepr<&mut A>` | `Owned<A>` | 拥有型 + 可变视图 |
| `ViewRepr<&A>` | `Owned<A>` | `Owned<A>` | 视图 + 拥有型 |
| `ViewRepr<&A>` | `ViewRepr<&A>` | `Owned<A>` | 两个视图 |
| `ArcRepr<A>` | `Owned<A>` | `Owned<A>` | Arc + 拥有型 |
| `ArcRepr<A>` | `ViewRepr<&A>` | `Owned<A>` | Arc + 视图 |

**原则**: 二元运算始终返回 `Owned` 类型（新分配），即使输入包含视图。

### 5.2 广播语义详解

#### 5.2.1 NumPy 广播规则

Senon 遵循 NumPy 广播规则：

1. **维度对齐**: 从最右维度开始对齐，维度数不足的在左侧补 1
2. **兼容条件**: 对应维度相等，或其中一个为 1
3. **结果形状**: 每个维度取两者的最大值

```
Shape A:     [3, 1, 4]
Shape B:     [    4, 1]
                    ↓ 广播
Result:      [3, 4, 4]
```

#### 5.2.2 运算符广播实现

```rust
// arithmetic.rs

impl<A, D> Add for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: Self) -> Self::Output {
        // 形状相同时直接逐元素相加
        if self.shape() == rhs.shape() {
            return zip_with_same_shape(&self, &rhs, |a, b| a + b);
        }
        
        // 形状不同时尝试广播
        match self.broadcast_with(&rhs) {
            Ok((a_broadcast, b_broadcast)) => {
                zip_with_broadcast(&a_broadcast, &b_broadcast, |a, b| a + b)
            }
            Err(e) => panic!("Broadcast error: {}", e),
        }
    }
}

// 标量广播特化
impl<A, D> Add<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: A) -> Self::Output {
        self.mapv(|x| x + rhs)
    }
}
```

#### 5.2.3 复合赋值广播

复合赋值的广播规则：**仅 RHS 可广播**，LHS 必须拥有完整存储。

```rust
// 合法：RHS 广播到 LHS 形状
let mut a = Tensor::zeros([3, 4]);
let b = Tensor::zeros([4]);  // shape [4] 可广播到 [3, 4]
a += &b;  // OK

// 非法：LHS 不能是广播视图
let a_view = large_tensor.slice(s![.., 0]);  // shape [100]，但有零步长
let b = Tensor::zeros([100]);
// a_view += &b;  // 编译错误：视图不可写，或运行时错误
```

### 5.3 实现代码结构

```rust
// arithmetic.rs

use core::ops::{Add, Sub, Mul, Div, Rem, Neg, Not};
use core::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};

use crate::element::{Element, Numeric};
use crate::tensor::{TensorBase, Tensor};
use crate::dimension::Dimension;

// ========== Tensor op Tensor ==========

impl<A, D> Add<Tensor<A, D>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn add(self, rhs: Tensor<A, D>) -> Self::Output {
        // 广播 + 逐元素
    }
}

impl<A, D> Sub<Tensor<A, D>> for Tensor<A, D> { /* ... */ }
impl<A, D> Mul<Tensor<A, D>> for Tensor<A, D> { /* ... */ }
impl<A, D> Div<Tensor<A, D>> for Tensor<A, D> { /* ... */ }
impl<A, D> Rem<Tensor<A, D>> for Tensor<A, D> { /* ... */ }

// ========== Tensor op Scalar ==========

impl<A, D> Add<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn add(self, rhs: A) -> Self::Output {
        self.mapv(|x| x + rhs)
    }
}

// ... Sub<A>, Mul<A>, Div<A>, Rem<A>

// ========== Scalar op Tensor ==========

impl<A, D> Add<Tensor<A, D>> for A
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn add(self, rhs: Tensor<A, D>) -> Self::Output {
        rhs.mapv(|x| self + x)
    }
}

// ... Sub<Tensor>, Mul<Tensor>, Div<Tensor>, Rem<Tensor>

// ========== Compound Assignment ==========

impl<A, D> AddAssign<Tensor<A, D>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add_assign(&mut self, rhs: Tensor<A, D>) {
        // 广播 RHS，原地加
    }
}

impl<A, D> AddAssign<A> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add_assign(&mut self, rhs: A) {
        self.mapv_inplace(|x| x + rhs);
    }
}

// ... SubAssign, MulAssign, DivAssign, RemAssign
```

---

## 6. 一元运算

### 6.1 Neg 运算符

`Neg`（负号）要求 `Numeric`，`bool` 不支持。

```rust
// arithmetic.rs

impl<A, D> Neg for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn neg(self) -> Self::Output {
        self.mapv(|x| -x)
    }
}

impl<A, D> Neg for &Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn neg(self) -> Self::Output {
        self.mapv(|x| -x)
    }
}
```

**类型支持**:

| 类型 | 支持 Neg | 说明 |
|------|----------|------|
| 整数 (i8/i16/i32/i64) | ✓ | 有符号整数 |
| 无符号整数 (u8/u16/u32/u64) | ✓ | 对无符号数取负为溢出（wrap around） |
| 浮点 (f32/f64) | ✓ | IEEE 754 负数 |
| 复数 (Complex<f32/f64>) | ✓ | 共轭取负 |
| bool | ✗ | 不实现 `Numeric` |

### 6.2 Not 运算符

`Not`（逻辑/位取反）仅需 `Element`，`bool` 支持。

```rust
// arithmetic.rs

impl<A, D> Not for Tensor<A, D>
where
    A: Element + Not<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn not(self) -> Self::Output {
        self.mapv(|x| !x)
    }
}

impl<A, D> Not for &Tensor<A, D>
where
    A: Element + Not<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;
    
    fn not(self) -> Self::Output {
        self.mapv(|x| !x)
    }
}
```

**类型支持**:

| 类型 | 支持 Not | 说明 |
|------|----------|------|
| bool | ✓ | 逻辑取反 `!true = false` |
| 整数 (i8/i16/i32/i64/u8/u16/u32/u64) | ✓ | 位取反 `!0xFF = 0x00` |
| 浮点 (f32/f64) | ✗ | 浮点无 `Not` trait |
| 复数 (Complex<T>) | ✗ | 复数无 `Not` trait |

### 6.3 使用示例

```rust
// Neg 示例
let a = tensor![[1.0, -2.0], [-3.0, 4.0]];
let b = -a;  // [[-1.0, 2.0], [3.0, -4.0]]

// Not 示例（bool）
let mask = tensor![[true, false], [false, true]];
let inverted = !mask;  // [[false, true], [true, false]]

// Not 示例（整数位取反）
let bits = tensor![[0u8, 255], [128, 64]];
let flipped = !bits;  // [[255, 0], [127, 191]]
```

---

## 7. 数学函数

### 7.1 方法签名

数学函数作为 `TensorBase` 的方法实现，要求 `A: RealScalar`（仅 f32/f64）。

```rust
// elementwise.rs

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = f32>,
    D: Dimension,
{
    // 三角函数
    pub fn sin(&self) -> Tensor<f32, D> { self.mapv(|x| x.sin()) }
    pub fn cos(&self) -> Tensor<f32, D> { self.mapv(|x| x.cos()) }
    pub fn tan(&self) -> Tensor<f32, D> { self.mapv(|x| x.tan()) }
    pub fn asin(&self) -> Tensor<f32, D> { self.mapv(|x| x.asin()) }
    pub fn acos(&self) -> Tensor<f32, D> { self.mapv(|x| x.acos()) }
    pub fn atan(&self) -> Tensor<f32, D> { self.mapv(|x| x.atan()) }
    
    // 指数/对数
    pub fn exp(&self) -> Tensor<f32, D> { self.mapv(|x| x.exp()) }
    pub fn ln(&self) -> Tensor<f32, D> { self.mapv(|x| x.ln()) }
    pub fn log2(&self) -> Tensor<f32, D> { self.mapv(|x| x.log2()) }
    pub fn log10(&self) -> Tensor<f32, D> { self.mapv(|x| x.log10()) }
    
    // 双曲函数
    pub fn sinh(&self) -> Tensor<f32, D> { self.mapv(|x| x.sinh()) }
    pub fn cosh(&self) -> Tensor<f32, D> { self.mapv(|x| x.cosh()) }
    pub fn tanh(&self) -> Tensor<f32, D> { self.mapv(|x| x.tanh()) }
    
    // 幂运算
    pub fn powi(&self, n: i32) -> Tensor<f32, D> { self.mapv(|x| x.powi(n)) }
    pub fn powf(&self, n: f32) -> Tensor<f32, D> { self.mapv(|x| x.powf(n)) }
    pub fn sqrt(&self) -> Tensor<f32, D> { self.mapv(|x| x.sqrt()) }
    pub fn cbrt(&self) -> Tensor<f32, D> { self.mapv(|x| x.cbrt()) }
}

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = f64>,
    D: Dimension,
{
    // f64 版本（同上，类型替换为 f64）
    pub fn sin(&self) -> Tensor<f64, D> { self.mapv(|x| x.sin()) }
    // ...
}

// 泛型版本（使用 RealScalar trait）
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    pub fn sin(&self) -> Tensor<A, D> { self.mapv(|x| x.sin()) }
    pub fn cos(&self) -> Tensor<A, D> { self.mapv(|x| x.cos()) }
    // ... 所有数学函数
}
```

### 7.2 函数分类与约束

| 函数 | Trait 约束 | 适用类型 | NaN 行为 |
|------|-----------|---------|---------|
| `sin`, `cos`, `tan` | `RealScalar` | f32, f64 | NaN 输入 → NaN 输出 |
| `asin`, `acos`, `atan` | `RealScalar` | f32, f64 | 超出定义域 → NaN |
| `sinh`, `cosh`, `tanh` | `RealScalar` | f32, f64 | NaN 输入 → NaN 输出 |
| `exp`, `exp2` | `RealScalar` | f32, f64 | Inf 输入 → Inf 输出 |
| `ln`, `log2`, `log10` | `RealScalar` | f32, f64 | 负数输入 → NaN |
| `sqrt`, `cbrt` | `RealScalar` | f32, f64 | 负数 sqrt → NaN |
| `powi`, `powf` | `RealScalar` | f32, f64 | 0^0 → 1.0 |

### 7.3 复数数学函数

复数 (`Complex<T>`) 实现的数学函数通过 `ComplexScalar` trait：

```rust
// 复数支持的数学函数（在 complex.rs 实现）
impl<T: RealScalar> ComplexScalar for Complex<T> {
    fn exp(self) -> Self { /* 复数指数 */ }
    fn ln(self) -> Self { /* 复数对数 */ }
    fn sqrt(self) -> Self { /* 复数平方根 */ }
}

// Tensor 方法（仅支持复数特有的数学函数）
impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: RealScalar,
{
    pub fn exp(&self) -> Tensor<Complex<T>, D> { self.mapv(|x| x.exp()) }
    pub fn ln(&self) -> Tensor<Complex<T>, D> { self.mapv(|x| x.ln()) }
    pub fn sqrt(&self) -> Tensor<Complex<T>, D> { self.mapv(|x| x.sqrt()) }
    pub fn conj(&self) -> Tensor<Complex<T>, D> { self.mapv(|x| x.conj()) }
    pub fn norm(&self) -> Tensor<T, D> { self.mapv(|x| x.norm()) }
    pub fn arg(&self) -> Tensor<T, D> { self.mapv(|x| x.arg()) }
}
```

**注意**: 复数不支持 `sin`/`cos`/`tan`（Senon 当前版本），未来可扩展。

---

## 8. 数值函数

### 8.1 方法签名

```rust
// elementwise.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 绝对值（整数和浮点）
    pub fn abs(&self) -> Tensor<A, D>
    where
        A: Numeric + ops::Neg<Output = A>,
    {
        self.mapv(|x| if x < A::zero() { -x } else { x })
    }
    
    /// 符号函数
    /// 返回：正数 → 1，零 → 0，负数 → -1
    pub fn sign(&self) -> Tensor<A, D>
    where
        A: Numeric + PartialOrd,
    {
        self.mapv(|x| {
            if x > A::zero() { A::one() }
            else if x < A::zero() { -A::one() }
            else { A::zero() }
        })
    }
    
    /// 平方（x * x）
    pub fn square(&self) -> Tensor<A, D> {
        self.mapv(|x| x * x)
    }
    
    /// 倒数（1 / x）
    pub fn reciprocal(&self) -> Tensor<A, D> {
        self.mapv(|x| A::one() / x)
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 向下取整
    pub fn floor(&self) -> Tensor<A, D> { self.mapv(|x| x.floor()) }
    
    /// 向上取整
    pub fn ceil(&self) -> Tensor<A, D> { self.mapv(|x| x.ceil()) }
    
    /// 四舍五入
    pub fn round(&self) -> Tensor<A, D> { self.mapv(|x| x.round()) }
    
    /// 幂运算（x^y）
    pub fn pow(&self, exp: A) -> Tensor<A, D> { self.mapv(|x| x.powf(exp)) }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// 整数幂运算（x^n）
    pub fn powi(&self, n: i32) -> Tensor<A, D>
    where
        A: RealScalar,
    {
        self.mapv(|x| x.powi(n))
    }
}
```

### 8.2 函数分类与约束

| 函数 | Trait 约束 | 适用类型 | 说明 |
|------|-----------|---------|------|
| `abs` | `Numeric` | 整数、浮点、复数 | 绝对值 |
| `sign` | `Numeric + PartialOrd` | 整数、浮点 | 符号函数 |
| `square` | `Numeric` | 整数、浮点、复数 | x * x |
| `reciprocal` | `Numeric` | 整数、浮点、复数 | 1 / x（注意整数除法） |
| `floor` | `RealScalar` | f32, f64 | 向下取整 |
| `ceil` | `RealScalar` | f32, f64 | 向上取整 |
| `round` | `RealScalar` | f32, f64 | 四舍五入 |
| `pow` | `RealScalar` | f32, f64 | x^y（浮点幂） |
| `powi` | `RealScalar` | f32, f64 | x^n（整数幂） |

### 8.3 特殊情况

**整数 `reciprocal`**:
```rust
let a = tensor![[2, 4, 5]];
let b = a.reciprocal();  // [[0, 0, 0]]（整数除法截断）
// 建议：先转换为浮点
let c = a.cast::<f64>().reciprocal();  // [[0.5, 0.25, 0.2]]
```

**复数 `abs`**:
```rust
let a = tensor![Complex { re: 3.0, im: 4.0 }];
let b = a.abs();  // [5.0]（返回实数）
```

---

## 9. 比较运算 (comparison.rs)

### 9.1 is_close 方法

```rust
// comparison.rs

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 逐元素近似相等比较
    ///
    /// 检查 `|self - other| <= atol + rtol * |other|`
    ///
    /// # Parameters
    /// - `other`: 比较目标
    /// - `rtol`: 相对容差（默认 1e-5）
    /// - `atol`: 绝对容差（默认 1e-8）
    ///
    /// # Returns
    /// 形状与 self 相同的 `Tensor<bool, D>`
    ///
    /// # Examples
    /// ```
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.00001, 2.0], [3.0, 4.0001]];
    /// let close = a.is_close(&b, 1e-5, 1e-8);
    /// // [[true, true], [true, false]]
    /// ```
    pub fn is_close<B, E>(
        &self,
        other: &TensorBase<B, E>,
        rtol: A,
        atol: A,
    ) -> Tensor<bool, <D as BroadcastDim<E>>::Output>
    where
        B: Storage<Elem = A>,
        E: Dimension,
    {
        // 广播后逐元素比较
        let shape = self.shape().broadcast_shape(other.shape()).unwrap();
        let a_broadcast = self.broadcast(shape).unwrap();
        let b_broadcast = other.broadcast(shape).unwrap();
        
        let mut result = Tensor::zeros(shape);
        Zip::from(&mut result)
            .and(&a_broadcast)
            .and(&b_broadcast)
            .for_each(|r, &a_val, &b_val| {
                let diff = (a_val - b_val).abs();
                let tol = atol + rtol * b_val.abs();
                *r = diff <= tol;
            });
        result
    }
}
```

### 9.2 allclose 方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 全元素近似相等检查
    ///
    /// 检查所有元素是否近似相等。
    /// 等价于 `is_close(...).all()`
    ///
    /// # Examples
    /// ```
    /// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    /// let b = tensor![[1.00001, 2.0], [3.0, 4.0]];
    /// assert!(a.allclose(&b, 1e-4, 1e-8));
    /// ```
    pub fn allclose<B, E>(
        &self,
        other: &TensorBase<B, E>,
        rtol: A,
        atol: A,
    ) -> bool
    where
        B: Storage<Elem = A>,
        E: Dimension,
    {
        self.is_close(other, rtol, atol).iter().all(|&x| x)
    }
}
```

### 9.3 clip/clamp 方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    /// 裁剪到指定范围
    ///
    /// 将所有元素限制在 [min, max] 范围内。
    ///
    /// # Parameters
    /// - `min`: 最小值
    /// - `max`: 最大值
    ///
    /// # Returns
    /// 新 Tensor，元素为 `max(min_val, min(max_val, x))`
    ///
    /// # Examples
    /// ```
    /// let a = tensor![-1.0, 0.5, 1.0, 2.0, 3.0];
    /// let b = a.clip(0.0, 2.0);
    /// assert_eq!(b, tensor![0.0, 0.5, 1.0, 2.0, 2.0]);
    /// ```
    pub fn clip(&self, min: A, max: A) -> Tensor<A, D> {
        self.mapv(|x| x.max(min).min(max))
    }
    
    /// clip 的别名（与 std::clamp 一致）
    pub fn clamp(&self, min: A, max: A) -> Tensor<A, D> {
        self.clip(min, max)
    }
    
    /// 原地裁剪
    pub fn clip_inplace(&mut self, min: A, max: A) {
        self.mapv_inplace(|x| x.max(min).min(max));
    }
    
    /// 原地 clamp
    pub fn clamp_inplace(&mut self, min: A, max: A) {
        self.clip_inplace(min, max);
    }
}
```

### 9.4 语义总结

| 方法 | 返回类型 | 语义 |
|------|----------|------|
| `is_close` | `Tensor<bool, D>` | 逐元素近似比较 |
| `allclose` | `bool` | 全元素近似相等 |
| `clip` / `clamp` | `Tensor<A, D>` | 裁剪到范围 |
| `clip_inplace` / `clamp_inplace` | `()` | 原地裁剪 |

---

## 10. SIMD 加速集成

### 10.1 SIMD 适用操作

| 操作 | SIMD 支持 | 条件 |
|------|----------|------|
| `add`, `sub`, `mul`, `div` | ✓ | 连续内存 + `simd` feature |
| `neg` | ✓ | 连续内存 + `simd` feature |
| `abs` | ✓ | 连续内存 + `simd` feature |
| `sin`, `cos`, `tan` | 部分 | pulp 提供有限支持 |
| `exp`, `ln` | 部分 | pulp 提供有限支持 |
| `sqrt` | ✓ | 连续内存 + `simd` feature |
| `floor`, `ceil`, `round` | ✓ | 连续内存 + `simd` feature |
| `map` / `mapv` | 视闭包 | 闭包需 SIMD 友好 |
| `zip_with` | 视闭包 | 闭包需 SIMD 友好 |

### 10.2 执行路径选择

```
                    ┌─────────────────┐
                    │  操作请求        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ simd feature 启用?│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ No                          │ Yes
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │ 标量路径        │           │ 连续 + 对齐?    │
     └────────────────┘           └────────┬───────┘
                                           │
                            ┌──────────────┴──────────────┐
                            │ No                          │ Yes
                            ▼                             ▼
                   ┌────────────────┐           ┌────────────────┐
                   │ 非对齐 SIMD     │           │ 对齐 SIMD       │
                   └────────────────┘           └────────────────┘
```

### 10.3 SIMD 实现示例

```rust
// elementwise.rs (simd feature)

#[cfg(feature = "simd")]
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = f32>,
    D: Dimension,
{
    pub fn add_simd(&self, other: &Self) -> Tensor<f32, D> {
        use pulp::Arch;
        
        let arch = Arch::new();
        
        // 检查连续性和对齐
        if self.is_contiguous() && other.is_contiguous() 
           && self.is_aligned() && other.is_aligned() 
           && self.shape() == other.shape() 
        {
            // SIMD 路径
            let mut result = Tensor::zeros(self.shape());
            
            arch.dispatch(|| {
                let n = self.len();
                let chunks = n / 8;  // AVX2: 8 x f32
                
                for i in 0..chunks {
                    let a = self.load_f32x8(i * 8);
                    let b = other.load_f32x8(i * 8);
                    let c = a + b;
                    result.store_f32x8(i * 8, c);
                }
                
                // 尾部标量处理
                for i in (chunks * 8)..n {
                    result[i] = self[i] + other[i];
                }
            });
            
            result
        } else {
            // 回退到标量
            self.mapv(|x| x) + other.mapv(|x| x)
        }
    }
}
```

### 10.4 性能阈值

| 条件 | 路径 |
|------|------|
| 元素数 < SIMD 宽度 (8/16) | 标量 |
| 元素数 ≥ SIMD 宽度 + 连续 | SIMD |
| 元素数 ≥ 并行阈值 (64K) + `parallel` feature | SIMD + 并行 |

---

## 11. 与其他模块的交互

### 11.1 与 `iter` 模块的交互

| 交互点 | 说明 |
|--------|------|
| `map`/`mapv` | 内部使用 `Elements` 迭代器 |
| `zip_with` | 内部使用 `Zip` 迭代器 |
| `mapv_inplace` | 内部使用 `ElementsMut` 迭代器 |
| 并行迭代 | `parallel` feature 启用时使用 `par_iter` |

```rust
// map 的迭代器实现
pub fn map<B, F>(&self, f: F) -> Tensor<B, D>
where
    B: Element,
    F: FnMut(&A) -> B,
{
    let mut result = Tensor::zeros(self.shape());
    result.iter_mut()
        .zip(self.iter())
        .for_each(|(dst, &src)| *dst = f(&src));
    result
}
```

### 11.2 与 `broadcast` 模块的交互

| 交互点 | 说明 |
|--------|------|
| 二元运算符 | 自动调用 `broadcast_with()` |
| `zip_with` | 广播后迭代 |
| `is_close` | 广播后比较 |

```rust
// Add impl 中的广播
impl<A, D> Add for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add(self, rhs: Self) -> Self::Output {
        let (a, b) = self.broadcast_with(&rhs)
            .expect("Incompatible shapes for broadcast");
        zip_with(&a, &b, |x, y| x + y)
    }
}
```

### 11.3 与 `simd` 模块的交互

| 交互点 | 说明 |
|--------|------|
| SIMD 分发 | 通过 `Arch::new().dispatch()` |
| 对齐检测 | 使用 `Layout::is_aligned()` |
| 连续性检测 | 使用 `Layout::is_contiguous()` |

```rust
// SIMD 路径选择
#[cfg(feature = "simd")]
fn add_impl(a: &Tensor<f32, D>, b: &Tensor<f32, D>) -> Tensor<f32, D> {
    if a.is_contiguous() && b.is_contiguous() && a.is_aligned() && b.is_aligned() {
        add_simd(a, b)
    } else {
        add_scalar(a, b)
    }
}

#[cfg(not(feature = "simd"))]
fn add_impl(a: &Tensor<f32, D>, b: &Tensor<f32, D>) -> Tensor<f32, D> {
    add_scalar(a, b)
}
```

### 11.4 与 `element` 模块的交互

| 交互点 | 说明 |
|--------|------|
| 泛型约束 | `A: Numeric` 用于算术，`A: RealScalar` 用于数学函数 |
| bool 排除 | 编译时阻止 bool 参与算术运算 |
| 类型转换 | `cast::<B>()` 用于类型提升 |

```rust
// 泛型约束示例
pub fn sum<A: Numeric>(&self) -> A {
    self.iter().fold(A::zero(), |acc, &x| acc + x)
}

// 编译时错误示例
// let bool_tensor: Tensor<bool, _> = ...;
// let _ = bool_tensor + bool_tensor;  // 编译错误：bool 不满足 Numeric
```

---

## 12. 实现任务分解

### 任务清单

| # | 任务 | 文件 | 预估时间 | 依赖 |
|---|------|------|----------|------|
| 1 | 创建 `elementwise.rs`，实现 `map` 方法 | `elementwise.rs` | 10 min | tensor, iter |
| 2 | 实现 `mapv` 和 `mapv_inplace` 方法 | `elementwise.rs` | 10 min | #1 |
| 3 | 实现 `zip_with` 函数（含广播支持） | `elementwise.rs` | 15 min | #1, broadcast |
| 4 | 实现 `apply` 函数（多元操作） | `elementwise.rs` | 10 min | #3 |
| 5 | 创建 `arithmetic.rs`，实现 `Tensor op Tensor` 运算符 | `arithmetic.rs` | 15 min | #3 |
| 6 | 实现 `Tensor op Scalar` 和 `Scalar op Tensor` | `arithmetic.rs` | 10 min | #5 |
| 7 | 实现复合赋值运算符 (`+=`, `-=`, etc.) | `arithmetic.rs` | 10 min | #5 |
| 8 | 实现 `Neg` 和 `Not` 一元运算符 | `arithmetic.rs` | 10 min | #5 |
| 9 | 实现数学函数方法 (`sin`, `cos`, `exp`, `ln`, etc.) | `elementwise.rs` | 15 min | #2 |
| 10 | 实现数值函数方法 (`abs`, `sign`, `floor`, etc.) | `elementwise.rs` | 10 min | #2 |
| 11 | 创建 `comparison.rs`，实现 `is_close`, `allclose`, `clip` | `comparison.rs` | 15 min | #3 |
| 12 | 编写单元测试和文档测试 | `tests/` | 15 min | #1-#11 |

**总预估时间**: 约 145 分钟（2.5 小时）

### 任务依赖图

```
#1 (map) ──→ #2 (mapv/mapv_inplace) ──→ #9 (数学函数)
         │                              │
         │                              └──→ #10 (数值函数)
         │
         └──→ #3 (zip_with) ──→ #4 (apply)
                  │
                  ├──→ #5 (Tensor op Tensor) ──→ #6 (Tensor op Scalar)
                  │                              │
                  │                              └──→ #7 (复合赋值)
                  │                              │
                  │                              └──→ #8 (Neg/Not)
                  │
                  └──→ #11 (comparison)

#1-#11 ──→ #12 (测试)
```

---

## 13. 设计决策记录

### 13.1 为什么 `bool` 不支持算术运算？

**决策**: `bool` 仅实现 `Element`，不实现 `Numeric`。

**理由**:
1. 布尔值四则运算无数学意义（`true + true` 应该是什么？）
2. 防止误用：`sum([true, false, true])` 无意义
3. 与 Rust 标准库一致：`bool` 不实现 `Add/Sub/Mul/Div`
4. 编译时类型安全：泛型 `A: Numeric` 自动排除 `bool`

**替代方案**: 允许 `bool` 隐式转换为 `0/1`。
**拒绝原因**: 隐式转换与 Rust 哲学冲突，可能导致难以发现的 bug。

### 13.2 为什么运算符重载返回 `Owned` 而非视图？

**决策**: 所有二元运算符返回 `Tensor<A, D>`（新分配），而非视图。

**理由**:
1. **生命周期简化**: 视图返回值需要复杂的生命周期标注
2. **避免悬垂引用**: 运算结果的生命周期与输入解耦
3. **语义清晰**: `a + b` 产生新数据，符合直觉
4. **优化空间**: 编译器可通过内联消除小数组分配

**替代方案**: 返回 `TensorView<'lifetime, A, D>` 延迟求值。
**拒绝原因**: 增加复杂度，生命周期传播困难，与 ndarray 行为不一致。

### 13.3 为什么 `map` 和 `mapv` 都保留？

**决策**: 同时提供 `map(|x| ...)`（引用）和 `mapv(|x| ...)`（值）。

**理由**:
1. **灵活性**: 某些操作需要引用（如调用方法 `x.method()`）
2. **性能**: 值传递对 `Copy` 类型更高效（避免间接寻址）
3. **类型转换**: `map` 可转换类型，`mapv` 也可，但语义不同
4. **与 ndarray 一致**: 用户熟悉的 API

**示例**:
```rust
// map：需要引用
let b = a.map(|x| x.to_string());  // A -> String

// mapv：简单计算
let b = a.mapv(|x| x * 2.0);  // f32 -> f32
```

### 13.4 为什么复合赋值 (`+=`) 仅支持 RHS 广播？

**决策**: `a += b` 中 `b` 可广播到 `a`，但 `a` 不能是广播视图。

**理由**:
1. **写入语义**: 广播视图是只读的（步长为 0），无法写入
2. **内存安全**: 广播视图的"重复"元素对应同一内存位置，写入会冲突
3. **与 NumPy 一致**: `np.ndarray.__iadd__` 行为相同

**示例**:
```rust
let mut a = Tensor::zeros([3, 4]);
let b = Tensor::zeros([4]);  // 可广播
a += &b;  // OK

let c = Tensor::zeros([3, 1]);  // 广播视图
let d = Tensor::zeros([3, 4]);
// c += &d;  // 编译错误或运行时错误：c 是广播视图
```

### 13.5 为什么 `clip` 和 `clamp` 是别名？

**决策**: `clip(min, max)` 和 `clamp(min, max)` 功能相同。

**理由**:
1. **API 兼容**: `clip` 是 NumPy/PyTorch 术语，`clamp` 是 Rust 标准库术语
2. **用户习惯**: 不同背景用户可能习惯不同名称
3. **零成本**: 别名无运行时开销

### 13.6 为什么 `is_close` 返回 `Tensor<bool>` 而非 `bool`？

**决策**: `is_close` 逐元素比较，返回布尔张量。

**理由**:
1. **细粒度信息**: 用户可知道哪些元素近似相等
2. **可组合**: 结果可用于掩码索引、条件选择
3. **与 `allclose` 区分**: `allclose` 返回 `bool`（全元素聚合）

**示例**:
```rust
let a = tensor![1.0, 2.0, 3.0];
let b = tensor![1.00001, 2.1, 3.0];

let close = a.is_close(&b, 1e-4, 1e-8);  // [true, false, true]
let all = a.allclose(&b, 1e-4, 1e-8);    // false
```

### 13.7 为什么 SIMD 支持有限？

**决策**: 当前版本仅部分操作支持 SIMD（算术、abs、sqrt、floor/ceil/round）。

**理由**:
1. **pulp 限制**: `pulp` crate 主要提供算术和基础数学的 SIMD
2. **超越函数复杂**: `sin`/`cos`/`exp`/`ln` 的 SIMD 实现需要专门库
3. **精度权衡**: 超越函数 SIMD 可能损失精度
4. **渐进增强**: 核心操作优先，未来可扩展

**替代方案**: 使用 `libm` 或 `sleef` 提供完整 SIMD 数学。
**延迟原因**: 增加依赖，当前版本优先保证核心功能正确性。

### 13.8 为什么 `pow` 和 `powi` 分开？

**决策**: `pow(exp: A)` 接受浮点指数，`powi(n: i32)` 接受整数指数。

**理由**:
1. **性能**: 整数幂可通过乘法实现，比浮点幂快
2. **精度**: 整数幂无精度损失
3. **与 Rust 标准库一致**: `f32::powi` vs `f32::powf`
4. **语义清晰**: 调用者明确知道使用哪种算法

---

## 附录 A: API 速查表

### A.1 映射操作

| 方法 | 签名 | 说明 |
|------|------|------|
| `map` | `(&self, F: FnMut(&A) -> B) -> Tensor<B, D>` | 引用映射 |
| `mapv` | `(&self, F: FnMut(A) -> B) -> Tensor<B, D>` | 值映射 |
| `mapv_inplace` | `(&mut self, F: FnMut(A) -> A)` | 原地映射 |

### A.2 二元操作

| 函数 | 签名 | 说明 |
|------|------|------|
| `zip_with` | `(&TensorBase<A, D>, &TensorBase<B, E>, F) -> Tensor<C, ...>` | 二元映射（支持广播） |
| `apply` | `(&[&TensorBase<A, D>], F) -> Tensor<A::Elem, D>` | N 元映射 |

### A.3 运算符

| 运算符 | Trait | 支持类型 |
|--------|-------|---------|
| `+`, `-`, `*`, `/`, `%` | `Add/Sub/Mul/Div/Rem` | `Numeric` |
| `+=`, `-=`, `*=`, `/=`, `%=` | `*Assign` | `Numeric` |
| `-` (一元) | `Neg` | `Numeric` |
| `!` | `Not` | `Element + Not` |

### A.4 数学函数

| 方法 | Trait 约束 | 说明 |
|------|-----------|------|
| `sin`, `cos`, `tan` | `RealScalar` | 三角函数 |
| `asin`, `acos`, `atan` | `RealScalar` | 反三角 |
| `sinh`, `cosh`, `tanh` | `RealScalar` | 双曲函数 |
| `exp`, `ln`, `log2`, `log10` | `RealScalar` | 指数/对数 |
| `sqrt`, `cbrt` | `RealScalar` | 根号 |
| `powi`, `powf` | `RealScalar` | 幂运算 |

### A.5 数值函数

| 方法 | Trait 约束 | 说明 |
|------|-----------|------|
| `abs` | `Numeric` | 绝对值 |
| `sign` | `Numeric + PartialOrd` | 符号 |
| `square` | `Numeric` | 平方 |
| `reciprocal` | `Numeric` | 倒数 |
| `floor`, `ceil`, `round` | `RealScalar` | 取整 |

### A.6 比较操作

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `is_close` | `Tensor<bool, D>` | 逐元素近似比较 |
| `allclose` | `bool` | 全元素近似相等 |
| `clip` / `clamp` | `Tensor<A, D>` | 裁剪到范围 |

---

## 附录 B: 错误类型

| 错误 | 触发场景 | 处理方式 |
|------|----------|----------|
| `BroadcastError` | 形状不兼容且无法广播 | `Result` |
| `ShapeError` | `apply` 中形状不一致 | `Result` |
| panic | `clip` 中 min > max | panic（参数校验） |

---

**文档结束**
