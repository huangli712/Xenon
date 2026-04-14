# 逐元素运算模块设计

> 文档编号: 11 | 模块: `src/math/` | 阶段: Phase 4
> 前置文档: `10-iterator.md`, `15-broadcast.md`
> 需求参考: 需求说明书 §12
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责     | 包含                                                                  | 不包含                                                  |
| -------- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| 算术运算 | add/sub/mul/div，数值类型：i32/i64/f32/f64/Complex                    | 归约运算（sum/prod/min/max，参见 `13-reduction.md §1`） |
| 一元运算 | abs/neg/square/signum（Numeric + PartialOrd），数学函数（RealScalar） | 篮选/排序                                               |
| 数学函数 | sin/sqrt/exp/ln/floor/ceil，仅 f32/f64                                | 运算符重载（参见 `19-overload.md §1`）                  |
| 复数运算 | norm（返回实数类型）/conj，仅 Complex                                 | 比较运算（eq/ne/lt/gt）                                 |
| 逻辑非   | `!`，仅 bool                                                          | 位运算                                                  |
| 比较运算 | eq/ne 对所有 Element 可用；lt/gt 仅限 RealScalar，返回 bool 张量，NaN 遵循 IEEE 754 | 搜索/排序                                               |
| 标量运算 | 标量与张量的逐元素运算                                                | 矩阵运算（dot/matmul）                                  |
| 广播支持 | 所有二元运算和比较运算支持广播                                        | 批量运算                                                |

### 1.2 设计原则

| 原则         | 体现                                          |
| ------------ | --------------------------------------------- |
| 类型安全边界 | 算术运算仅支持 `Numeric`，bool 编译时排除     |
| 广播透明集成 | 所有二元运算自动支持广播                      |
| 存储模式无关 | 对 Tensor、TensorView、TensorViewMut 统一工作 |
| NaN 语义明确 | IEEE 754 NaN 传播规则                         |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: broadcast (依赖 tensor, dimension)
L6: math（逐元素运算） ← 当前模块（依赖 broadcast, iter, element）
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §12 |
| 范围内   | 逐元素算术、一元运算、数学函数、复数 `norm` / `conjugate`、逻辑非、比较运算与广播语义。 |
| 范围外   | SIMD 数学函数专用 kernel（当前版本数学函数统一回退标量；SIMD 仅覆盖逐元素算术/一元运算和整数归约）、混合类型逐元素运算以及 `map` 系列公开 API。 |
| 非目标   | 不新增新的数学库依赖，不在本文扩展 mixed-type API 或更通用的逐元素映射原语。 |

---

## 3. 文件位置

```
src/math/
├── mod.rs              # 模块入口，re-export 公开 API
├── zip.rs              # 二元逐元素（zip_with，含广播）
├── unary.rs            # 一元运算（abs, neg, signum, square, sin, sqrt, exp, ln, floor, ceil, norm, conj, not）
├── binary.rs           # 二元算术方法（add, sub, mul, div, add_scalar, sub_scalar, mul_scalar, div_scalar）
└── comparison.rs       # 比较运算（eq, ne, lt, gt）
```

多文件设计理由：按操作元数分组（一元 vs 二元）可保持当前最小范围；通用映射基础设施（`map` / `mapv` / `mapv_inplace`）不属于需求说明书 §12 的本期最小交付，暂不纳入当前版本。运算符重载（Add/Sub/Mul/Div trait 实现）保留在 `src/overload/arithmetic.rs`。SIMD 加速由独立 backend 模块 `src/simd/` 承载，`math/` 仅负责语义 API 与分发入口。

---

## 4. 依赖关系

### 4.1 依赖图

```
src/math/
├── crate::tensor        # TensorBase<S, D>, TensorView
├── crate::iter          # Elements, ElementsMut, Zip (pub(crate))
├── crate::element       # Element, Numeric, RealScalar, ComplexScalar
├── crate::broadcast     # broadcast_shape() for binary ops
└── crate::simd (opt.)   # pulp::Arch backend dispatch
```

### 4.2 类型级依赖

| 来源模块       | 使用的类型/trait                                                                       |
| -------------- | -------------------------------------------------------------------------------------- |
| `tensor`       | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.shape()`（参见 `07-tensor.md §4`） |
| `iter`         | `Elements`, `ElementsMut`, `Zip`（`pub(crate)` 内部工具，参见 `10-iterator.md §4.3`） |
| `element`      | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`（参见 `03-element.md §4`）         |
| `broadcast`    | `broadcast_shape()`, `broadcast_to()` 返回的 `TensorView`（参见 `15-broadcast.md §4`） |
| `dimension`    | `BroadcastDim<E>` trait（编译期维度推导，参见 `02-dimension.md §4.9`）                 |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd.md §4`）                                                   |
| `error`        | `XenonError`（含 `BroadcastError` 变体，参见 `26-error.md §4`）                        |

### 4.3 依赖方向

> **依赖方向：单向向上。** `math` 模块消费 `iter`、`tensor`、`element`、`broadcast` 模块，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 范围边界说明

`map` / `mapv` / `mapv_inplace` 属于更通用的逐元素映射基础设施，但不在需求说明书 §12 的当前最小范围内。当前版本文档不将其作为公开 API 承诺；如后续需要，应以独立议题重新评估与类型转换、就地修改、错误语义的边界关系。

### 5.2 二元 zip 操作

> **维度推导说明：** 此函数使用 `BroadcastDim<DB>` 进行编译期维度推导，该 trait 定义于 `02-dimension.md §4.9`，详见该文档。

```rust
/// Binary element-wise operation with broadcast support.
pub fn zip_with<A, B, C, DA, DB, F>(
    a: &TensorBase<impl Storage<Elem = A>, DA>,
    b: &TensorBase<impl Storage<Elem = B>, DB>,
    f: F,
) -> Result<Tensor<C, <DA as BroadcastDim<DB>>::Output>, XenonError>
where
    DA: BroadcastDim<DB>,
    DB: Dimension,
    F: FnMut(A, B) -> C,
    A: Element + Copy,
    B: Element + Copy,
    C: Element;
```

### 5.3 算术运算（Numeric 约束）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Element-wise addition (with broadcast support).
    pub fn add<E>(&self, other: &TensorBase<impl Storage<Elem = A>, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Add<Output = A>;

    /// Element-wise subtraction.
    pub fn sub<E>(&self, other: &TensorBase<impl Storage<Elem = A>, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Sub<Output = A>;

    /// Element-wise multiplication.
    pub fn mul<E>(&self, other: &TensorBase<impl Storage<Elem = A>, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Mul<Output = A>;

    /// Element-wise division.
    pub fn div<E>(&self, other: &TensorBase<impl Storage<Elem = A>, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Div<Output = A>;
}
```

支持的类型：i32, i64, f32, f64, Complex\<f32\>, Complex\<f64\>。

> **整数算术补充约束：** 对 `i32` / `i64` 的 `add` / `sub` / `mul` / `div`，实现必须使用 checked arithmetic；凡发生溢出、除以零或结果不可表示，均按需求说明书 §12 与 §27 走 panic 语义，不得回落为 wrapping 行为。

### 5.4 一元运算（Numeric 约束）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + PartialOrd,
{
    pub fn abs(&self) -> Tensor<A, D>;
    pub fn neg(&self) -> Tensor<A, D>;
    pub fn square(&self) -> Tensor<A, D>;

    /// Element-wise sign function: returns -1, 0, or 1 based on the sign of each element.
    ///
    /// Available for all ordered numeric types: i32, i64, f32, f64.
    /// For complex types (no natural ordering), use `ComplexScalar::arg()` instead.
    ///
    /// # NaN behavior (floats)
    ///
    /// `signum(NaN)` returns `NaN` (IEEE 754 semantics, via PartialOrd).
pub fn signum(&self) -> Tensor<A, D>;
}
```

> **整数一元运算补充约束：** `abs` / `square` / `signum` 在整数路径上同样必须使用 checked arithmetic。特别是最小负值取绝对值、平方溢出等情形，均须视为不可恢复错误并触发 panic。

### 5.5 数学函数（RealScalar 约束：仅 f32/f64）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    pub fn sin(&self) -> Tensor<A, D>;
    pub fn sqrt(&self) -> Tensor<A, D>;
    pub fn exp(&self) -> Tensor<A, D>;
    pub fn ln(&self) -> Tensor<A, D>;
    pub fn floor(&self) -> Tensor<A, D>;
    pub fn ceil(&self) -> Tensor<A, D>;
}
```

### 5.6 复数运算（ComplexScalar 约束）

```rust
impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: RealScalar,
{
    /// Norm operation, returns a real-typed tensor.
    pub fn norm(&self) -> Tensor<T, D>;
}

impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: Element + Copy,
{
    /// Conjugate operation.
    pub fn conjugate(&self) -> Tensor<Complex<T>, D>;
}
```

> **类型一致性约束：** 参与逐元素运算或比较的双方元素类型须预先一致。因此，`Complex<T>` 与实数标量的混合张量 API（如 `add_real_scalar` / `mul_real_scalar`）不属于当前公开范围；若内部实现需要复用相应标量逻辑，也只能作为不对外承诺的内部辅助路径存在。

### 5.7 逻辑非（仅 bool）

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = bool>,
    D: Dimension,
{
    /// Logical NOT.
    pub fn not(&self) -> Tensor<bool, D>;
}
```

### 5.8 比较运算

`eq` / `ne` 对所有元素类型可用（包括 `bool` 与 `Complex`）；`lt` / `gt` 仅对 `RealScalar` 可用，因此只覆盖 `f32` / `f64`。`bool` 与 `Complex` 类型不支持 `lt` / `gt`。

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    /// Element-wise equality comparison, returns a bool tensor. NaN comparison follows IEEE 754.
    pub fn eq<DB>(&self, other: &TensorBase<impl Storage<Elem = A>, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        D: BroadcastDim<DB>,
        DB: Dimension;

    pub fn ne<DB>(&self, other: &TensorBase<impl Storage<Elem = A>, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        D: BroadcastDim<DB>,
        DB: Dimension;
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: RealScalar,
{
    pub fn lt<DB>(&self, other: &TensorBase<impl Storage<Elem = A>, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        D: BroadcastDim<DB>,
        DB: Dimension;

    pub fn gt<DB>(&self, other: &TensorBase<impl Storage<Elem = A>, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        D: BroadcastDim<DB>,
        DB: Dimension;
}
```

> **NaN 语义：** `eq(NaN, NaN)` 返回 `false`，`ne(NaN, NaN)` 返回 `true`，遵循 IEEE 754。

### 5.9 标量与张量运算

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Element-wise tensor-scalar addition.
    pub fn add_scalar(&self, scalar: A) -> Tensor<A, D>;

    /// Element-wise tensor-scalar subtraction.
    pub fn sub_scalar(&self, scalar: A) -> Tensor<A, D>;

    /// Element-wise tensor-scalar multiplication.
    pub fn mul_scalar(&self, scalar: A) -> Tensor<A, D>;

    /// Element-wise tensor-scalar division.
    pub fn div_scalar(&self, scalar: A) -> Tensor<A, D>;
}
```

### 5.10 Good / Bad 对比示例

```rust
// Good - use zip_with for broadcast addition
let a = Tensor::<f64, Ix2>::zeros([3, 1]);
let b = Tensor::<f64, Ix2>::zeros([1, 4]);
let c = zip_with(&a, &b, |x, y| x + y)?;  // shape [3, 4]

// Bad - manual loop iteration (poor performance, no broadcast support)
let mut result = Tensor::<f64, Ix2>::zeros([3, 4]);
for i in 0..3 {
    for j in 0..4 {
        result[[i, j]] = a[[i, 0]] + b[[0, j]];  // not recommended
    }
}

// Bad - using arithmetic operations on bool
// let b: Tensor<bool, _> = ...;
// b.add(&other);  // compile error: bool does not satisfy Numeric
```

---

## 6. 内部实现设计

### 6.1 二元与一元运算的共享执行骨架

```
apply_unary(view, f):
    result = Tensor::zeros(view.shape())
    for (src, dst) in view.iter().zip(result.iter_mut()):
        *dst = f(*src)
    return result
```

### 6.2 zip_with 实现（含广播）

```
zip_with(a, b, f):
    broadcast_shape = broadcast_shape(a.shape(), b.shape())?
a_broadcast = a.broadcast_to(broadcast_shape)
b_broadcast = b.broadcast_to(broadcast_shape)
    result = Tensor::zeros(broadcast_shape)
    Zip::from(result.view_mut())
        .and(a_broadcast)
        .and(b_broadcast)
        .for_each(|r, a_val, b_val| *r = f(a_val, b_val))
    return result
```

### 6.3 SIMD 加速路径

```rust
// Use #[cfg] attribute for conditional compilation instead of runtime cfg!()
// This ensures the SIMD path is fully eliminated at compile time when the
// feature is not enabled.

#[cfg(feature = "simd")]
fn add_impl_simd<A>(a: &TensorView<A, D>, b: &TensorView<A, D>) -> Tensor<A, D>
where
    D: Dimension,
    A: Numeric + Copy,
{
    if a.is_f_contiguous() && b.is_f_contiguous() {
        return simd::dispatch_binary_op(simd::BinaryOp::Add, a, b);
    }
    zip_with_scalar(a, b, |x, y| x + y)
}

#[cfg(not(feature = "simd"))]
fn add_impl_simd<A>(a: &TensorView<A, D>, b: &TensorView<A, D>) -> Tensor<A, D>
where
    D: Dimension,
    A: Numeric + Copy,
{
    zip_with_scalar(a, b, |x, y| x + y)
}
```

参见 `08-simd.md §4.5` 了解 SIMD 后端详情。当前版本 SIMD 仅覆盖逐元素算术/一元运算和整数归约；`sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` 等数学函数统一回退标量实现。

---

## 7. 实现任务拆分

### Wave 1: 二元操作与一元运算

- [ ] **T2**: 实现 `zip_with`（含广播支持）
  - 文件: `src/math/zip.rs`
  - 内容: 基于逐元素遍历骨架的二元操作
  - 测试: `test_zip_with_same_shape`, `test_zip_with_broadcast`
  - 前置: 10-iterator.md, broadcast 模块
  - 预计: 10 min

- [ ] **T3**: 实现一元运算（abs/neg/signum/square）
  - 文件: `src/math/unary.rs`
  - 内容: 基于统一逐元素遍历骨架实现一元运算，并为整数路径补齐 checked arithmetic
  - 测试: `test_abs`, `test_neg`, `test_signum`, `test_square`
  - 前置: 10-iterator.md
  - 预计: 10 min

- [ ] **T4**: 实现数学函数（sin/sqrt/exp/ln/floor/ceil）
  - 文件: `src/math/unary.rs`
  - 内容: RealScalar 约束的数学方法
  - 测试: `test_sin`, `test_sqrt`, `test_exp`, `test_floor_ceil`
  - 前置: 10-iterator.md
  - 预计: 10 min

- [ ] **T5**: 实现复数操作（`conjugate`）与复数数学函数（`norm`）
  - 文件: `src/math/unary.rs`
  - 内容: `conjugate` 与 `norm` 的范围内实现
  - 测试: `test_norm`, `test_conj`
  - 前置: 10-iterator.md
  - 预计: 10 min

### Wave 3: 算术与比较运算

- [ ] **T6**: 实现算术运算（add/sub/mul/div）
  - 文件: `src/math/binary.rs`
  - 内容: 基于 `zip_with` 的算术运算，标量版本
  - 测试: `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_mul_scalar`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现逻辑非（not）和比较运算（eq/ne/lt/gt）
  - 文件: `src/math/unary.rs`（not）, `src/math/comparison.rs`（eq/ne/lt/gt）
  - 内容: bool 取反、比较运算返回 bool 张量
  - 测试: `test_not_bool`, `test_eq_f64`, `test_lt_i32`, `test_nan_comparison`
  - 前置: T2
  - 预计: 10 min

### Wave 4: SIMD 集成

- [ ] **T8**: 添加 SIMD 加速路径
  - 文件: `src/math/binary.rs`, `src/simd/vector.rs`
  - 内容: 在 `math` 中接入独立 `simd/` backend，为连续数组提供 SIMD 路径并保留标量回退
  - 测试: `test_add_simd_vs_scalar`, `test_mul_simd_vs_scalar`
  - 前置: T3, 08-simd.md
  - 预计: 10 min

### 并行执行分组图

```
            ┌───────┬────────┬───────┐
            |       |        |       |
            v       v        v       v
Wave 1:    [T2]    [T3]     [T4]    [T5]
            |
         ┌──┴─────┐
         |        |
         v        v
Wave 2: [T6]    [T7]
         |
         v
Wave 3: [T8]
```

---

## 8. 测试计划

### 8.1 测试分类总表

| 测试分类 | 说明                                      | 包含的测试                                                                                                                                                                                                                                                                                                                                                                              |
| -------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 单元测试 | 验证单个运算函数的基本正确性              | `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_add_broadcast`, `test_mul_scalar`, `test_abs`, `test_neg`, `test_signum`, `test_square_checked_overflow`, `test_sin`, `test_sqrt`, `test_exp_ln_roundtrip`, `test_floor_ceil`, `test_norm`, `test_conj`, `test_not_bool`, `test_eq_f64`, `test_lt_i32`, `test_nan_comparison`, `test_empty_tensor`, `test_add_simd_vs_scalar` |
| 集成测试 | 验证运算模块与迭代器/广播模块的端到端集成 | `test_zip_with_same_shape`, `test_zip_with_broadcast`（参见 §6 T2）                                                                                                                                                                                                                                                                                                                     |
| 边界测试 | 空张量、NaN、Inf、非连续输入等边界条件    | `test_empty_tensor`, `test_nan_comparison`, `test_add_simd_vs_scalar`（详见 §7.2）                                                                                                                                                                                                                                                                                                      |
| 属性测试 | 通过随机输入验证数学不变量                | 详见下方属性测试不变量表                                                                                                                                                                                                                                                                                                                                                                |

**属性测试不变量**

| 不变量                                                                  | 测试方法                                                  |
| ----------------------------------------------------------------------- | --------------------------------------------------------- |
| 加法交换律（实数类型）                                                  | 对随机 f32/f64/i32/i64 张量：`a.add(&b) == b.add(&a)`     |
| NaN 传播：所有运算遇到 NaN 输入时输出包含 NaN                           | 构造含 NaN 的张量，验证 sin/sqrt/add/mul 等运算结果含 NaN |
| 标量运算逆元：`a.add_scalar(k).sub_scalar(k) == a`                      | 随机张量和标量值                                          |
| zip_with 结合性（实数加法）：`(a.add(&b)).add(&c) == a.add(&b.add(&c))` | 随机同形状 f64 张量，容差比较                             |

### 8.2 单元测试清单

| 测试函数                       | 测试内容                                 | 优先级 |
| ------------------------------ | ---------------------------------------- | ------ |
| `test_add_i32`                 | i32 加法正确                             | 高     |
| `test_add_f64`                 | f64 加法正确                             | 高     |
| `test_add_complex`             | Complex\<f64\> 加法正确                  | 高     |
| `test_add_broadcast`           | 广播加法 shape [3,1]+[1,4]=[3,4]         | 高     |
| `test_mul_scalar`              | 标量乘法正确                             | 中     |
| `test_abs`                     | abs(-3) = 3, abs(f64) 正确               | 高     |
| `test_neg`                     | neg 正确，含复数                         | 中     |
| `test_signum`                  | signum 正/零/负                          | 中     |
| `test_square_checked_overflow` | 整数平方溢出触发 panic                   | 高     |
| `test_sin`                     | sin(0) = 0, sin(pi/2) ≈ 1                | 高     |
| `test_sqrt`                    | sqrt(4) = 2, sqrt(-1) = NaN              | 高     |
| `test_exp_ln_roundtrip`        | exp(ln(x)) ≈ x                           | 中     |
| `test_floor_ceil`              | floor(1.7)=1, ceil(1.3)=2                | 中     |
| `test_norm`                    | Complex{3,4}.norm() = 5.0                | 高     |
| `test_conjugate`               | Complex{1,2}.conjugate() = Complex{1,-2} | 中     |
| `test_not_bool`                | !true = false, !false = true             | 中     |
| `test_eq_f64`                  | 逐元素相等比较                           | 高     |
| `test_lt_i32`                  | 逐元素小于比较                           | 高     |
| `test_nan_comparison`          | NaN 比较遵循 IEEE 754                    | 高     |
| `test_empty_tensor`            | 空张量运算返回空张量                     | 中     |
| `test_add_simd_vs_scalar`      | SIMD 路径结果与标量一致                  | 中     |

### 8.3 边界测试场景

| 场景                  | 预期行为                                   |
| --------------------- | ------------------------------------------ |
| 空张量 `shape=[0, 3]` | add 返回空张量                             |
| 单元素张量            | 所有运算正确                               |
| NaN 输入（f32/f64）   | NaN 传播（sin(NaN)=NaN, 0\*NaN=NaN）       |
| Inf 输入              | exp(Inf)=Inf, ln(0)=-Inf                   |
| 广播形状不兼容        | zip_with 返回 `XenonError::BroadcastError` |
| 非连续输入（切片后）  | 运算结果与连续输入一致                     |

### 8.4 集成测试

| 测试文件             | 测试内容                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_math.rs` | `zip_with` / 标量路径与 `iter`、`broadcast`、`tensor`、`simd` backend 的端到端集成 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 所有逐元素运算走标量 / fallback 路径且语义满足文档约束。 |
| 启用 `simd` | 连续输入上的 SIMD 分发结果与默认配置保持一致，非连续输入仍正确回退。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `bool` 不参与算术运算 | 编译期测试或 trait 约束验证。 |
| `lt` / `gt` 仅对 `RealScalar` 开放 | 编译期失败测试。 |
| mixed-type 逐元素运算不属于当前公开范围 | API 缺失断言或编译期失败测试。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向               | 对方模块    | 接口/类型                                  | 约定                                                                                                                                          |
| ------------------ | ----------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `math → iter`      | `iter`      | `Elements`, `Zip`                          | 逐元素运算复用 `Elements` 及相关遍历入口，二元与归约路径可复用 `iter::Zip` 这个 `pub(crate)` 内部工具（参见 `10-iterator.md` §4.3）         |
| `math → broadcast` | `broadcast` | `broadcast_shape()`                        | 二元运算先调用广播模块推导兼容视图（参见 `15-broadcast.md` §4）                                                                               |
| `math → element`   | `element`   | `Numeric` / `RealScalar` / `ComplexScalar` | 通过元素约束区分数值与复数运算语义（参见 `03-element.md` §4）                                                                                 |
| `math → simd`      | `simd`      | SIMD backend dispatch facade               | 连续数组且 feature 开启时通过稳定的 backend facade 分发到 SIMD 或标量路径，`math` 不直接依赖具体 vector kernel 名称（参见 `08-simd.md` §4.5） |

### 9.2 数据流描述

```text
User calls add / unary op / zip_with
    │
    ├── math selects unary, binary, or scalar execution
    ├── binary ops validate broadcast compatibility first
    ├── iter / Zip produce element streams from shape + strides
    └── SIMD dispatch is used only when enabled and semantically equivalent
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 广播不兼容时返回 `XenonError::BroadcastError`，携带输入 shape 上下文；显式 helper 保持 `Result` 路径。 |
| Panic | 整数 `add/sub/mul/div`、`abs/square/signum` 的溢出、除零或结果不可表示均按需求触发 panic。 |
| 路径一致性 | 标量与 SIMD 路径必须保持相同 shape、错误类别、NaN/复数语义；不满足前提时统一回退标量实现。 |
| 容差边界 | 当前不引入额外数值容差；仅在可证明语义等价时启用 SIMD，否则回退标量路径。 |

---

## 11. 设计决策记录（ADR）

### 决策 1：不在当前版本公开 map 系列

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | 当前版本不把 `map` / `mapv` / `mapv_inplace` 纳入公开 API 承诺 |
| 理由     | 需求说明书 §12 仅要求明确列出的逐元素运算，不要求通用映射原语  |
| 替代方案 | 直接在本期暴露完整 map 系列                                    |
| 拒绝原因 | 会扩大 API 面且引入额外语义边界，不符合当前最小范围            |

### 决策 2：NaN 比较遵循 IEEE 754

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 比较运算（eq/ne/lt/gt）遵循 IEEE 754 语义：NaN != NaN                 |
| 理由     | 与 Rust 标准库 `f64::partial_cmp` 行为一致；与 NumPy/ndarray 行为一致 |
| 替代方案 | 提供总排序比较（total_cmp）                                           |
| 拒绝原因 | 当前版本不需要总排序，可未来扩展                                      |

### 决策 3：SIMD 优化路径

| 属性     | 值                                                        |
| -------- | --------------------------------------------------------- |
| 决策     | 连续 + 对齐内存时，仅对已覆盖的逐元素算术/一元运算自动使用 SIMD 路径；数学函数仍回退到标量 |
| 理由     | SIMD 路径只在连续内存上有意义；非连续时标量路径更简单正确 |
| 替代方案 | 所有路径都用标量                                          |
| 拒绝原因 | 性能差距显著（2-4x），科学计算用户期望高性能              |

> **补充**：SIMD 实现位于独立 backend 模块 `src/simd/`，`math/` 仅按连续性和 feature gate 决定是否委托该 backend；当前版本数学函数不接入专门 SIMD kernel。

---

## 12. 性能考量

### 12.1 SIMD 加速预期

| 操作         | 标量路径 | SIMD 路径（AVX2）  | 加速比 |
| ------------ | -------- | ------------------ | ------ |
| add f32 (1M) | ~2ms     | ~0.5ms             | 4x     |
| mul f64 (1M) | ~3ms     | ~1ms               | 3x     |
| sin f64 (1M) | ~20ms    | 标量回退（≈20ms）  | ≈1.0x  |

### 12.2 复杂度标注

- `zip_with`: O(n) 时间，O(n) 空间
- 广播操作: O(n) 时间，O(n) 空间（结果），广播本身零拷贝

---

## 13. 平台与工程约束

| 项目       | 约束                                                                                           |
| ---------- | ---------------------------------------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性                                     |
| crate 结构 | 保持单 crate 结构，不拆分独立 math crate                                                       |
| 依赖约束   | 仅允许项目基线中的可选 SIMD / 并行依赖，不新增额外第三方数学库                                 |
| 范围边界   | 当前版本仅覆盖需求说明书 §12 明确列出的逐元素运算；`map` 系列与实复混合公开 API 不在本期范围内 |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-10 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
