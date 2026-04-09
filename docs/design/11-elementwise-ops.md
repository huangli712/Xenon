# 逐元素运算模块设计

> 文档编号: 11 | 模块: `src/ops/elementwise.rs` | 阶段: Phase 4
> 前置文档: `10-iterator.md`, `15-broadcast.md`
> 需求参考: 需求说明书 §12

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 算术运算 | add/sub/mul/div，数值类型：i32/i64/f32/f64/Complex | 归约运算（sum/prod/min/max，参见 `13-reduction.md §1`） |
| 一元运算 | abs/neg/square/signum（Numeric + PartialOrd），数学函数（RealScalar） | 篮选/排序 |
| 数学函数 | sin/sqrt/exp/ln/floor/ceil，仅 f32/f64 | 运算符重载（参见 `19-operator-overload.md §1`） |
| 复数运算 | norm（返回实数类型）/conj，仅 Complex | 比较运算（eq/ne/lt/gt） |
| 逻辑非 | `!`，仅 bool | 位运算 |
| 比较运算 | eq/ne/lt/gt，返回 bool 张量，NaN 遵循 IEEE 754 | 搜索/排序 |
| 标量运算 | 标量与张量的逐元素运算 | 矩阵运算（dot/matmul） |
| 广播支持 | 所有二元运算和比较运算支持广播 | 批量运算 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 类型安全边界 | 算术运算仅支持 `Numeric`，bool 编译时排除 |
| 广播透明集成 | 所有二元运算自动支持广播 |
| 存储模式无关 | 对 Tensor、TensorView、TensorViewMut 统一工作 |
| NaN 语义明确 | IEEE 754 NaN 传播规则 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: broadcast (依赖 tensor, dimension)
L6: ops/elementwise  ← 当前模块（依赖 broadcast, iter, element）
```

---

## 2. 文件位置

```
src/ops/
├── mod.rs              # 模块入口，re-export 公开 API
└── elementwise.rs     # 逐元素运算核心（map, zip_with, 数学函数）
```

单文件设计理由：逐元素运算聚焦于 map/zip_with/apply 及数学函数，运算符重载在独立模块实现。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/ops/elementwise.rs
├── crate::tensor        # TensorBase<S, D>, TensorView
├── crate::iter          # Elements, ElementsMut, Zip
├── crate::element       # Element, Numeric, RealScalar, ComplexScalar
├── crate::broadcast     # broadcast_shape()（二元运算广播）
└── crate::simd (可选)   # pulp::Arch（SIMD 加速路径）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.shape()`（参见 `07-tensor.md §4`） |
| `iter` | `Elements`, `ElementsMut`, `Zip`（参见 `10-iterator.md §4`） |
| `element` | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`（参见 `03-element-types.md §4`） |
| `broadcast` | `broadcast_shape()`, `BroadcastView`（参见 `15-broadcast.md §4`） |
| `dimension` | `BroadcastDim<E>` trait（编译期维度推导，参见 `02-dimension.md §4.9`） |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd-backend.md §4`） |
| `error` | `XenonError`, `BroadcastError`（参见 `26-error-handling.md §4`） |

### 3.3 依赖方向

> **依赖方向：单向向上。** `ops/elementwise` 消费 `iter`、`tensor`、`element`、`broadcast` 模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 核心映射操作

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Element-wise mapping (by reference), returns a newly allocated Tensor.
    pub fn map<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element,
        F: FnMut(&A) -> B;

    /// Element-wise mapping (by value), returns a newly allocated Tensor.
    ///
    /// All Xenon element types are `Copy`, so value semantics is safe and zero-cost.
    pub fn mapv<B, F>(&self, f: F) -> Tensor<B, D>
    where
        B: Element,
        A: Copy,
        F: FnMut(A) -> B;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// In-place element-wise mapping.
    ///
    /// All Xenon element types are `Copy`, so value semantics is safe and zero-cost.
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        A: Copy,
        F: FnMut(A) -> A;
}
```

### 4.2 二元 zip 操作

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

### 4.3 算术运算（Numeric 约束）

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

### 4.4 一元运算（Numeric 约束）

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

### 4.5 数学函数（RealScalar 约束：仅 f32/f64）

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

### 4.6 复数运算（ComplexScalar 约束）

```rust
impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: RealScalar,
{
    /// Norm operation, returns a real-typed tensor.
    pub fn norm(&self) -> Tensor<T, D>;

    /// Conjugate operation.
    pub fn conj(&self) -> Tensor<Complex<T>, D>;
}
```

### 4.7 逻辑非（仅 bool）

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

### 4.8 比较运算

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

### 4.9 标量与张量运算

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

### 4.10 Good / Bad 对比示例

```rust
// Good - use map for type conversion
let a: Tensor<i32, Ix1> = Tensor::from_slice(&[1, 2, 3]);
let b: Tensor<f64, Ix1> = a.map(|&x| x as f64);

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

## 5. 内部实现设计

### 5.1 map 实现

```
map(view, f):
    result = Tensor::zeros(view.shape())
    for (src, dst) in view.iter().zip(result.iter_mut()):
        *dst = f(src)
    return result
```

### 5.2 zip_with 实现（含广播）

```
zip_with(a, b, f):
    broadcast_shape = broadcast_shape(a.shape(), b.shape())?
    a_broadcast = a.broadcast(broadcast_shape)
    b_broadcast = b.broadcast(broadcast_shape)
    result = Tensor::zeros(broadcast_shape)
    Zip::from(result.view_mut())
        .and(a_broadcast)
        .and(b_broadcast)
        .for_each(|r, a_val, b_val| *r = f(a_val, b_val))
    return result
```

### 5.3 SIMD 加速路径

```rust
// Use #[cfg] attribute for conditional compilation instead of runtime cfg!()
// This ensures the SIMD path is fully eliminated at compile time when the
// feature is not enabled.

#[cfg(feature = "simd")]
fn add_impl_simd<A>(a: &TensorView<A, D>, b: &TensorView<A, D>) -> Tensor<A, D>
where
    A: Numeric + Copy,
{
    if a.is_contiguous() && b.is_contiguous() {
        return simd::add_vectorized(a, b);
    }
    zip_with_scalar(a, b, |x, y| x + y)
}

#[cfg(not(feature = "simd"))]
fn add_impl_simd<A>(a: &TensorView<A, D>, b: &TensorView<A, D>) -> Tensor<A, D>
where
    A: Numeric + Copy,
{
    zip_with_scalar(a, b, |x, y| x + y)
}
```

参见 `08-simd-backend.md §4.5` 了解 SIMD 后端详情。

---

## 6. 实现任务拆分

### Wave 1: 核心映射

- [ ] **T1**: 实现 `map` / `mapv` / `mapv_inplace`
  - 文件: `src/ops/elementwise.rs`
  - 内容: 基于 `Elements` 迭代器的映射操作
  - 测试: `test_map`, `test_mapv`, `test_mapv_inplace`
  - 前置: 10-iterator.md 完成
  - 预计: 10 min

### Wave 2: 二元操作与一元运算

- [ ] **T2**: 实现 `zip_with`（含广播支持）
  - 文件: `src/ops/elementwise.rs`
  - 内容: 基于 `Zip` 迭代器的二元操作
  - 测试: `test_zip_with_same_shape`, `test_zip_with_broadcast`
  - 前置: T1, broadcast 模块
  - 预计: 10 min

- [ ] **T3**: 实现一元运算（abs/neg/signum/square）
  - 文件: `src/ops/elementwise.rs`
  - 内容: 基于 `mapv` 的一元运算
  - 测试: `test_abs`, `test_neg`, `test_signum`, `test_square`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现数学函数（sin/sqrt/exp/ln/floor/ceil）
  - 文件: `src/ops/elementwise.rs`
  - 内容: RealScalar 约束的数学方法
  - 测试: `test_sin`, `test_sqrt`, `test_exp`, `test_floor_ceil`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现复数运算（norm/conj）
  - 文件: `src/ops/elementwise.rs`
  - 内容: ComplexScalar 约束的复数方法
  - 测试: `test_norm`, `test_conj`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 算术与比较运算

- [ ] **T6**: 实现算术运算（add/sub/mul/div）
  - 文件: `src/ops/elementwise.rs`
  - 内容: 基于 `zip_with` 的算术运算，标量版本
  - 测试: `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_mul_scalar`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现逻辑非（not）和比较运算（eq/ne/lt/gt）
  - 文件: `src/ops/elementwise.rs`
  - 内容: bool 取反、比较运算返回 bool 张量
  - 测试: `test_not_bool`, `test_eq_f64`, `test_lt_i32`, `test_nan_comparison`
  - 前置: T2
  - 预计: 10 min

### Wave 4: SIMD 集成

- [ ] **T8**: 添加 SIMD 加速路径
  - 文件: `src/ops/elementwise.rs`（#[cfg(feature = "simd")] 块）
  - 内容: 算术运算的 SIMD 路径，连续数组检测
  - 测试: `test_add_simd_vs_scalar`, `test_mul_simd_vs_scalar`
  - 前置: T3, 08-simd-backend.md
  - 预计: 15 min

### 并行执行分组图

```
Wave 1:                    [T1]
                            |
            ┌───────┬───────┴───────┬───────┐
            |       |               |       |
            v       v               v       v
Wave 2:    [T2]    [T3]            [T4]    [T5]
            |
         ┌──┴─────┐
         |        |
         v        v
Wave 3: [T6]    [T7]
         |
         v
Wave 4: [T8]
```

---

## 7. 测试计划

### 7.0 测试分类总表

| 测试分类 | 说明 | 包含的测试 |
|----------|------|-----------|
| 单元测试 | 验证单个运算函数的基本正确性 | `test_map_type_conversion`, `test_mapv_square`, `test_mapv_inplace`, `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_add_broadcast`, `test_mul_scalar`, `test_abs`, `test_neg`, `test_signum`, `test_sin`, `test_sqrt`, `test_exp_ln_roundtrip`, `test_floor_ceil`, `test_norm`, `test_conj`, `test_not_bool`, `test_eq_f64`, `test_lt_i32`, `test_nan_comparison`, `test_empty_tensor`, `test_add_simd_vs_scalar` |
| 集成测试 | 验证运算模块与迭代器/广播模块的端到端集成 | `test_zip_with_same_shape`, `test_zip_with_broadcast`（参见 §6 T2） |
| 边界测试 | 空张量、NaN、Inf、非连续输入等边界条件 | `test_empty_tensor`, `test_nan_comparison`, `test_add_simd_vs_scalar`（详见 §7.2） |
| 属性测试 | 通过随机输入验证数学不变量 | 详见下方属性测试不变量表 |

#### 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| 加法交换律（实数类型） | 对随机 f32/f64/i32/i64 张量：`a.add(&b) == b.add(&a)` |
| NaN 传播：所有运算遇到 NaN 输入时输出包含 NaN | 构造含 NaN 的张量，验证 sin/sqrt/add/mul 等运算结果含 NaN |
| map 恒等：`a.map(\|&x\| x) == a` | 随机形状和类型的张量 |
| 标量运算逆元：`a.add_scalar(k).sub_scalar(k) == a` | 随机张量和标量值 |
| zip_with 结合性（实数加法）：`(a.add(&b)).add(&c) == a.add(&b.add(&c))` | 随机同形状 f64 张量，容差比较 |

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_map_type_conversion` | `map` 从 i32 到 f64 转换正确 | 高 |
| `test_mapv_square` | `mapv(\|x\| x * x)` 正确 | 高 |
| `test_mapv_inplace` | 原地修改后数据正确 | 高 |
| `test_add_i32` | i32 加法正确 | 高 |
| `test_add_f64` | f64 加法正确 | 高 |
| `test_add_complex` | Complex\<f64\> 加法正确 | 高 |
| `test_add_broadcast` | 广播加法 shape [3,1]+[1,4]=[3,4] | 高 |
| `test_mul_scalar` | 标量乘法正确 | 中 |
| `test_abs` | abs(-3) = 3, abs(f64) 正确 | 高 |
| `test_neg` | neg 正确，含复数 | 中 |
| `test_signum` | signum 正/零/负 | 中 |
| `test_sin` | sin(0) = 0, sin(pi/2) ≈ 1 | 高 |
| `test_sqrt` | sqrt(4) = 2, sqrt(-1) = NaN | 高 |
| `test_exp_ln_roundtrip` | exp(ln(x)) ≈ x | 中 |
| `test_floor_ceil` | floor(1.7)=1, ceil(1.3)=2 | 中 |
| `test_norm` | Complex{3,4}.norm() = 5.0 | 高 |
| `test_conj` | Complex{1,2}.conj() = Complex{1,-2} | 中 |
| `test_not_bool` | !true = false, !false = true | 中 |
| `test_eq_f64` | 逐元素相等比较 | 高 |
| `test_lt_i32` | 逐元素小于比较 | 高 |
| `test_nan_comparison` | NaN 比较遵循 IEEE 754 | 高 |
| `test_empty_tensor` | 空张量运算返回空张量 | 中 |
| `test_add_simd_vs_scalar` | SIMD 路径结果与标量一致 | 中 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 `shape=[0, 3]` | add 返回空张量 |
| 单元素张量 | 所有运算正确 |
| NaN 输入（f32/f64） | NaN 传播（sin(NaN)=NaN, 0*NaN=NaN） |
| Inf 输入 | exp(Inf)=Inf, ln(0)=-Inf |
| 广播形状不兼容 | zip_with 返回 BroadcastError |
| 非连续输入（切片后） | 运算结果与连续输入一致 |

---

## 8. 与其他模块的交互

| 交互模块 | 接口约定 |
|----------|----------|
| `iter` | map 内部使用 `Elements`，zip_with 内部使用 `Zip`（参见 `10-iterator.md §4`） |
| `broadcast` | 二元运算调用 `broadcast_shape()`（参见 `15-broadcast.md §4`） |
| `element` | 泛型约束 `Numeric`/`RealScalar`/`ComplexScalar`（参见 `03-element-types.md §4`） |
| `simd`（可选） | 连续数组时自动走 SIMD 路径（参见 `08-simd-backend.md §4.5`） |

---

## 9. 设计决策记录（ADR）

### 决策 1：map 返回新张量 vs 原地修改

| 属性 | 值 |
|------|-----|
| 决策 | `map`/`mapv` 返回新分配的 `Tensor<B, D>`，`mapv_inplace` 原地修改 |
| 理由 | 不可变视图无法写入；返回新张量的生命周期与输入解耦，避免悬垂引用；原地修改作为显式 opt-in |
| 替代方案 | 全部原地修改 |
| 拒绝原因 | 类型转换（i32→f64）无法原地修改，破坏 API 一致性 |

### 决策 2：NaN 比较遵循 IEEE 754

| 属性 | 值 |
|------|-----|
| 决策 | 比较运算（eq/ne/lt/gt）遵循 IEEE 754 语义：NaN != NaN |
| 理由 | 与 Rust 标准库 `f64::partial_cmp` 行为一致；与 NumPy/ndarray 行为一致 |
| 替代方案 | 提供总排序比较（total_cmp） |
| 拒绝原因 | 当前版本不需要总排序，可未来扩展 |

### 决策 3：SIMD 优化路径

| 属性 | 值 |
|------|-----|
| 决策 | 连续 + 对齐内存时自动使用 SIMD 路径，非连续时回退到标量 |
| 理由 | SIMD 路径只在连续内存上有意义；非连续时标量路径更简单正确 |
| 替代方案 | 所有路径都用标量 |
| 拒绝原因 | 性能差距显著（2-4x），科学计算用户期望高性能 |

---

## 10. 性能考量

### 10.1 SIMD 加速预期

| 操作 | 标量路径 | SIMD 路径（AVX2） | 加速比 |
|------|----------|-------------------|--------|
| add f32 (1M) | ~2ms | ~0.5ms | 4x |
| mul f64 (1M) | ~3ms | ~1ms | 3x |
| sin f64 (1M) | ~20ms | ~12ms（部分 SIMD） | 1.7x |

### 10.2 复杂度标注

- `map`/`mapv`: O(n) 时间，O(n) 空间
- `zip_with`: O(n) 时间，O(n) 空间
- `mapv_inplace`: O(n) 时间，O(1) 额外空间
- 广播操作: O(n) 时间，O(n) 空间（结果），广播本身零拷贝

---

## 11. no_std 兼容性

逐元素运算模块在 `no_std` 环境下可用，但需注意结果分配和数学函数依赖。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `map` / `mapv` | ✅ | 返回新 `Tensor`，需 `no_std + alloc` |
| `mapv_inplace` | ✅ | 原地修改，无额外分配 |
| `zip_with` | ✅ | 返回新 `Tensor`，需 `no_std + alloc` |
| 算术运算 (add/sub/mul/div) | ✅ | 基于 `zip_with`，需 `no_std + alloc` |
| 数学函数 (sin/sqrt/exp/ln/...) | ✅ | 数学函数（sin/exp/ln/sqrt 等）需要 `std` feature，no_std 环境下不可用（RealScalar 数学方法在 no_std 下无实现者） |
| 比较运算 (eq/ne/lt/gt) | ✅ | 无特殊依赖 |
| 复数运算 (norm/conj) | ✅ | 基于 `map`，需 `no_std + alloc` |
| 逻辑非 (not) | ✅ | 基于 `map`，需 `no_std + alloc` |
| SIMD 加速路径 | ✅ | pulp crate 支持 `no_std`（参见 `08-simd-backend.md §11`） |

条件编译处理：

```rust
// map/zip_with return new Tensor — needs alloc::vec::Vec
// Math functions (sin/exp/ln/sqrt etc.) require `std` feature;
// in no_std environments, RealScalar math methods have no implementors.

#[cfg(not(feature = "std"))]
extern crate alloc;
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
