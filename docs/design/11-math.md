# 逐元素运算模块设计

> 文档编号: 11 | 模块: `src/math/` | 阶段: Phase 4
> 前置文档: `03-element.md`, `08-simd.md`, `09-parallel.md`, `10-iterator.md`, `15-broadcast.md`, `26-error.md`
> 需求参考: 需求说明书 §4, §9.1, §9.2, §9.3, §12, §20, §27, §28.2, §28.3, §28.4, §28.5
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责     | 包含                                                                  | 不包含                                                  |
| -------- | --------------------------------------------------------------------- | ------------------------------------------------------- |
| 算术运算 | add/sub/mul/div，数值类型：i32/i64/f32/f64/Complex                    | 归约运算（sum/prod/min/max，参见 `13-reduction.md §1`） |
| 一元运算 | abs（有序数值）；signum（浮点按符号位、整数按比较）；neg/square（Numeric）；数学函数（RealScalar） | 篮选/排序                                               |
| 数学函数 | sin/sqrt/exp/ln/floor/ceil，仅 f32/f64                                | 运算符重载（参见 `19-overload.md §1`）                  |
| 复数运算 | modulus/模（返回实数类型）/conjugate（公开 API；内部 Complex 方法名可记为 conj），仅 Complex | 比较运算（eq/ne/lt/gt）                                 |
| 逻辑非   | `!`，仅 bool                                                          | 位运算                                                  |
| 比较运算 | eq/ne 对所有 Element 可用；lt/gt 对 i32/i64/f32/f64 可用，返回 bool 张量，NaN 遵循 IEEE 754 | 搜索/排序                                               |
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
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; tensor owns storage and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: broadcast (depends on tensor, dimension)
L6: math (element-wise operations) <- current module (depends on broadcast, iter, element)
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §4, §9.1, §9.2, §9.3, §12, §20, §27, §28.2, §28.3, §28.4, §28.5 |
| 范围内   | 逐元素算术、一元运算、数学函数、复数 `modulus` / `conjugate`、逻辑非、比较运算、标量-张量逐元素语义与广播语义。 |
| 范围外   | 混合类型逐元素运算以及 `map` 系列公开 API。SIMD 与并行覆盖范围仅限本模块负责的逐元素运算；若当前类型/ISA/语义约束不满足，则自动回退标量。 |
| 非目标   | 不新增新的数学库依赖，不在本文扩展 mixed-type API 或更通用的逐元素映射原语。 |

---

## 3. 文件位置

```
src/math/
├── mod.rs              # module entry, re-export public APIs
├── binary.rs           # binary arithmetic methods and shared binary execution skeleton
├── unary.rs            # unary operations (abs, neg, signum, square, sin, sqrt, exp, ln, floor, ceil, modulus, conjugate, not)
└── comparison.rs       # comparison operations (eq, ne, lt, gt)

Optional dependency touchpoints:
src/simd/               # optional SIMD backend consumed by math dispatch
src/parallel/           # optional parallel backend consumed by math dispatch
```

多文件设计理由：按操作元数分组（一元 vs 二元）可保持当前最小范围；更通用的逐元素映射基础设施不属于需求说明书 §12 的本期最小交付，暂不纳入当前版本。运算符重载（Add/Sub/Mul/Div trait 实现）保留在 `src/overload/arithmetic.rs`。SIMD 加速由独立 backend 模块 `src/simd/` 承载，`math/` 仅负责语义 API 与分发入口。

---

## 4. 依赖关系

### 4.1 Invariants

| 不变量 | 说明 |
| ---- | ---- |
| 广播先决 | 所有二元逐元素运算与比较运算必须先验证广播兼容性，再遍历广播后的只读视图。 |
| 输出形状稳定 | 二元运算返回张量的 shape 必须等于广播结果 shape；一元运算与标量运算保持输入 shape 不变。 |
| 比较类型边界 | `lt` / `gt` 只对 `i32`、`i64`、`f32`、`f64` 开放；`bool` 与 `Complex` 不得通过公开 API 进入该路径。 |
| SIMD 语义等价 | SIMD 覆盖范围见 `08-simd.md §5.4a`；在本模块内仅讨论需求说明书 §12 定义的逐元素运算。任一路径的 shape、NaN 语义和错误边界都必须与公开契约一致；不满足 SIMD 前提时统一回退标量。 |

### 4.2 Error Scenarios

| 场景 | 错误 |
| ---- | ---- |
| 二元运算广播失败 | 返回 `XenonError::BroadcastError`；在二元逐元素语境下，诊断信息必须分别区分 `input_shape(lhs)` 与 `other_shape(rhs)`，不得再让 `target_shape` 承载右操作数 shape 的含义。 |
| 构造结果张量时元素总数与 shape 不一致 | 返回 `XenonError::InvalidShape { operation: Cow<'static, str>, shape: Vec<usize>, expected_elements: usize, actual_elements: usize, offending_dim: Option<usize> }`。 |
| 公开 API 收到不满足前提的参数 | 返回 `XenonError::InvalidArgument { operation: Cow<'static, str>, argument: Cow<'static, str>, expected: Cow<'static, str>, actual: Cow<'static, str>, axis: Option<usize>, shape: Option<Vec<usize>> }`。 |
| 整数算术溢出、除零或结果不可表示 | 属于 panic 语义，不进入 `XenonError`。 |

### 4.3 依赖图

```
src/math/
├── crate::tensor        # TensorBase<S, D>, TensorView
├── crate::iter          # Elements, ElementsMut
├── crate::element       # Element, Numeric, RealScalar, ComplexScalar
├── crate::broadcast     # broadcast_shape() for binary ops
├── crate::dispatch      # ExecPath, select_exec_path() for execution path decision
├── crate::simd (opt.)   # Pure vectorized backend
├── crate::parallel (opt.) # Pure parallel backend
└── crate::error         # XenonError
```

### 4.4 类型级依赖

| 来源模块       | 使用的类型/trait                                                                       |
| -------------- | -------------------------------------------------------------------------------------- |
| `tensor`       | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.shape()`（参见 `07-tensor.md §5`） |
| `iter`         | `Elements`, `ElementsMut`（参见 `10-iterator.md §5`）                                  |
| `element`      | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`, `OrderedCompareElement`（定义见 `03-element.md §5.4a`）         |
| `complex`      | `Complex<f32>`, `Complex<f64>`（参见 `04-complex.md §5`）                              |
| `broadcast`    | `broadcast_shape()`, `broadcast_to()` 返回的 `TensorView`（参见 `15-broadcast.md §5`） |
| `dimension`    | `BroadcastDim<E>` trait（编译期维度推导，参见 `02-dimension.md §5.9`）                 |
| `storage`      | `Storage<Elem = A>`, `StorageMut<Elem = A>`                                            |
| `dispatch`（内部） | `select_exec_path()`、`ExecPath`、`should_parallelize()`、`can_use_simd()` |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd.md §5`）                                                   |
| `parallel`（可选） | `par_zip_map()`（纯并行执行入口，不含串行回退，参见 `09-parallel.md §5` / `§6`） |
| `error`        | `XenonError`（含 `BroadcastError` 变体，参见 `26-error.md §4`）                        |

### 4.5 依赖方向

> **依赖方向：单向向上。** `math` 模块消费 `iter`、`tensor`、`element`、`broadcast` 模块，不被它们依赖。

### 4.6 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 范围边界说明

更通用的逐元素映射基础设施不在需求说明书 §12 的当前最小范围内。当前版本文档不将其作为公开 API 承诺；如后续需要，应以独立议题重新评估与类型转换、就地修改、错误语义的边界关系。

### 5.2 二元逐元素执行约定

> **维度推导说明：** 二元逐元素方法统一使用 `BroadcastDim<DB>` 进行编译期维度推导，该 trait 定义于 `02-dimension.md §5.9`，详见该文档。

当前版本不承诺独立的通用二元逐元素 helper 公开函数。二元算术、比较与内部辅助路径统一采用“先广播，再直接遍历广播后视图并写入结果张量”的执行模型；启用 `parallel` feature 时，同一语义也可委托并行路径执行。

### 5.3 算术运算（Numeric 约束）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Element-wise addition (with broadcast support).
    pub fn add<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Add<Output = A>;

    /// Element-wise subtraction.
    pub fn sub<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Sub<Output = A>;

    /// Element-wise multiplication.
    pub fn mul<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Mul<Output = A>;

    /// Element-wise division.
    pub fn div<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension,
        A: Numeric + Copy + Div<Output = A>;
}
```

支持的类型：i32, i64, f32, f64, Complex\<f32\>, Complex\<f64\>。

> **整数算术补充约束：** 对 `i32` / `i64` 的 `add` / `sub` / `mul` / `div`，实现必须使用 checked arithmetic；凡发生溢出、除以零或结果不可表示，均按需求说明书 §12 与 §27 走 panic 语义，不得回落为 wrapping 行为。

### 5.4 一元运算（分离 trait bounds）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + PartialOrd,
{
    pub fn abs(&self) -> Tensor<A, D>;

    /// Element-wise sign function: returns -1, 0, or 1 based on the sign of each element.
    ///
    /// Available for all ordered numeric types: i32, i64, f32, f64.
    ///
    /// # NaN behavior (floats)
    ///
    /// Floating-point `signum` follows IEEE 754 sign-function semantics,
    /// including NaN propagation and signed-zero behavior.
    /// Integer `signum` follows `PartialOrd`.
    pub fn signum(&self) -> Tensor<A, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    pub fn neg(&self) -> Tensor<A, D>;
    pub fn square(&self) -> Tensor<A, D>;
}
```

`abs` / `signum` 仅对具备自然顺序的数值类型开放：i32, i64, f32, f64。`neg` / `square` 对所有 `Numeric` 类型开放：i32, i64, f32, f64, Complex<f32>, Complex<f64>。

> **整数一元运算补充约束：** `abs` / `square` 在整数路径上必须使用 checked arithmetic。特别是最小负值取绝对值、平方溢出等情形，均须视为不可恢复错误并触发 panic。`signum` 仅做符号分类，不额外要求 checked arithmetic。

> **`neg()` 整数边界：** 对有符号整数，`neg(i32::MIN)` / `neg(i64::MIN)` 等不可表示情形视为不可恢复错误，遵循 panic 语义。

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

> **数学函数精度约束：** `sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` 使用 Rust 提供的数学能力，不引入外部数学 crate；实现必须在文档与测试中声明其采用的 ULP 误差边界。当前版本至少要求把这些方法视为“遵循 IEEE 754，误差受 Rust / 平台实现文档化 ULP 上界约束”的接口承诺，而非无误差精确计算。

### 5.6 复数运算（ComplexScalar 约束）

```rust
impl<S, D, T> TensorBase<S, D>
where
    S: Storage<Elem = Complex<T>>,
    D: Dimension,
    T: RealScalar,
{
    /// Modulus operation, returns a real-typed tensor.
    pub fn modulus(&self) -> Tensor<T, D>;
}

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = Complex<f32>>,
    D: Dimension,
{
    /// Conjugate operation.
    pub fn conjugate(&self) -> Tensor<Complex<f32>, D>;
}

impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = Complex<f64>>,
    D: Dimension,
{
    /// Conjugate operation.
    pub fn conjugate(&self) -> Tensor<Complex<f64>, D>;
}
```

> **命名说明：** 公开张量 API 统一使用 `conjugate()`（与 `Numeric::conjugate()` 保持一致）；`conj` 仅允许作为内部 `Complex` 方法名或实现细节出现，不构成公开 API 命名承诺。

> **术语说明：** `modulus()` 对应需求说明书 §12 中的“模”运算。`Complex<f32> → f32`，`Complex<f64> → f64`。

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

`eq` / `ne` 对所有元素类型可用（包括 `bool` 与 `Complex`）；`lt` / `gt` 的需求级支持范围固定为 `i32`、`i64`、`f32`、`f64`，返回 `Tensor<bool, _>`。`bool` 与 `Complex` 类型不支持 `lt` / `gt`。

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    /// Element-wise equality comparison, returns a bool tensor. NaN comparison follows IEEE 754.
    pub fn eq<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;

    pub fn ne<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: OrderedCompareElement,
{
    /// Element-wise less-than comparison.
    ///
    /// Supported ordered element types are i32, i64, f32, and f64.
    pub fn lt<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;

    /// Element-wise greater-than comparison.
    ///
    /// Supported ordered element types are i32, i64, f32, and f64.
    pub fn gt<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;
}
```

> **类型边界说明：** `lt` / `gt` 不再复用 `RealScalar` 或更宽泛的 `Numeric + PartialOrd` 约束；公开 API 以 `OrderedCompareElement` 明确收敛到 `i32`、`i64`、`f32`、`f64` 四类元素类型。该 trait 定义见 `03-element.md §5.4a`。

> **NaN 语义：** `eq(NaN, NaN)` 返回 `false`，`ne(NaN, NaN)` 返回 `true`，遵循 IEEE 754。

> **标量比较入口：** 与 `require.md §12` 一致，比较运算也提供标量-张量入口；标量按可广播到目标全形状的零维输入处理，因此成功路径的形状与对应张量输入版本一致。

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    pub fn eq_scalar(&self, scalar: A) -> Tensor<bool, D>;
    pub fn ne_scalar(&self, scalar: A) -> Tensor<bool, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: OrderedCompareElement,
{
    pub fn lt_scalar(&self, scalar: A) -> Tensor<bool, D>;
    pub fn gt_scalar(&self, scalar: A) -> Tensor<bool, D>;
}
```

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

> **标量算术语义：** 标量版算术方法与张量-张量运算遵循相同的 checked arithmetic 语义：有符号整数溢出、除以零、结果不可表示均遵循 panic 语义。

### 5.10 Good / Bad 对比示例

```rust,ignore
// Good - use method API for broadcast addition
let a = Tensor::<f64, Ix2>::zeros([3, 1]);
let b = Tensor::<f64, Ix2>::zeros([1, 4]);
let c = a.add(&b)?;  // shape [3, 4]

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
    src_iter = view.iter()
    dst_iter = result.iter_mut()
    while let (Some(src), Some(dst)) = (src_iter.next(), dst_iter.next()):
        *dst = f(*src)
    return result
```

### 6.2 二元逐元素实现（含广播）

```
apply_binary(a, b, f):
    broadcast_shape = broadcast_shape(a.shape(), b.shape())?
    a_broadcast = a.broadcast_to(broadcast_shape)
    b_broadcast = b.broadcast_to(broadcast_shape)
    result = Tensor::zeros(broadcast_shape)
    dst_iter = result.iter_mut()
    a_iter = a_broadcast.iter()
    b_iter = b_broadcast.iter()
    while let (Some(dst), Some(a_val), Some(b_val)) = (
        dst_iter.next(),
        a_iter.next(),
        b_iter.next(),
    ):
        *dst = f(*a_val, *b_val)
    return result
```

### 6.3 SIMD 加速路径

SIMD 分发由 `math` 模块在满足连续性、对齐和 feature gate 前提时通过 `dispatch::select_exec_path()` 选择执行路径，再委托 `simd/` 纯向量化后端执行。参见 `08-simd.md §5.5` 了解 SIMD 后端详情。数学模块定义需求说明书 §12 中列出的逐元素运算，但当前版本只对 `08-simd.md §5.4a` 覆盖矩阵列出的子集尝试 SIMD；未列出的运算、类型、ISA 或不满足语义约束的路径统一回退标量实现。

> **并行路径：** 当 `parallel` feature 启用时，二元逐元素运算通过 `dispatch::select_exec_path()` 判断后委托 `parallel::par_zip_map` 执行并行遍历。并行模块不含串行回退，串行路径由本模块串行实现承担。

---

## 7. 实现任务拆分

### Wave 1: 二元操作与一元运算

- [ ] **T1**: 创建 `src/math/mod.rs` 骨架与公开导出
  - 文件: `src/math/mod.rs`
  - 内容: 模块声明、re-export 公开 API、为后续二元/一元/比较文件预留入口
  - 测试: 编译通过
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 实现共享二元逐元素执行骨架（含广播支持）
  - 文件: `src/math/binary.rs`
  - 内容: 基于直接遍历广播视图的二元操作内部辅助路径
  - 测试: `test_binary_same_shape`, `test_binary_broadcast`
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
  - 测试: `test_norm`, `test_conjugate`
  - 前置: 10-iterator.md
  - 预计: 10 min

### Wave 2: 算术与比较运算

- [ ] **T6**: 实现算术运算（add/sub/mul/div）
  - 文件: `src/math/binary.rs`
  - 内容: 基于共享二元逐元素执行骨架的算术运算，标量版本
  - 测试: `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_mul_scalar`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现逻辑非（not）和比较运算（eq/ne/lt/gt）
  - 文件: `src/math/unary.rs`（not）, `src/math/comparison.rs`（eq/ne/lt/gt）
  - 内容: bool 取反、比较运算返回 bool 张量
  - 测试: `test_not_bool`, `test_eq_f64`, `test_lt_i32`, `test_nan_comparison`
  - 前置: T2
  - 预计: 10 min

### Wave 3: SIMD 集成

- [ ] **T8**: 添加 SIMD 加速路径
  - 文件: `src/math/binary.rs`, `src/simd/vector.rs`
  - 内容: 在 `math` 中接入独立 `simd/` backend，为连续数组上的逐元素算术提供 SIMD 路径并保留标量回退
  - 测试: `test_add_simd_vs_scalar`, `test_mul_simd_vs_scalar`
  - 前置: T3, 08-simd.md
  - 预计: 10 min

### 并行执行分组图

```
            +-------+--------+-------+
            |       |        |       |
            v       v        v       v
Wave 1:    [T2]    [T3]     [T4]    [T5]
            |
         +--+-----+
         |        |
         v        v
Wave 2: [T6]    [T7]
            |
            v
Wave 3:    [T8]
```

---

## 8. 测试计划

### 8.1 测试分类总表

| 测试分类 | 说明                                      | 包含的测试                                                                                                                                                                                                                                                                                                                                                                              |
| -------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 单元测试 | 验证单个运算函数的基本正确性              | `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_add_broadcast`, `test_mul_scalar`, `test_abs`, `test_neg`, `test_signum`, `test_square_checked_overflow`, `test_sin`, `test_sqrt`, `test_exp_ln_roundtrip`, `test_floor_ceil`, `test_norm`, `test_conjugate`, `test_not_bool`, `test_eq_f64`, `test_lt_i32`, `test_nan_comparison`, `test_empty_tensor`, `test_add_simd_vs_scalar` |
| 集成测试 | 验证运算模块与迭代器/广播模块的端到端集成 | `test_binary_same_shape`, `test_binary_broadcast`（参见 §6 T2）                                                                                                                                                                                                                                                                                                                         |
| 边界测试 | 空张量、大张量、高维、NaN/Inf、非连续输入以及整数 panic 场景 | `test_empty_tensor`, `test_large_tensor_add_parallel`, `test_high_rank_broadcast`, `test_nan_comparison`, `test_inf_math_functions`, `test_add_simd_vs_scalar`, `test_div_i32_by_zero_panics`, `test_abs_i32_min_panics`（详见 §8.3） |
| 属性测试 | 通过随机输入验证数学不变量                | 详见下方属性测试不变量表                                                                                                                                                                                                                                                                                                                                                                |

**属性测试不变量**

| 不变量                                                                  | 测试方法                                                  |
| ----------------------------------------------------------------------- | --------------------------------------------------------- |
| 加法交换律（整数与无 NaN 实数输入）                                     | 对随机 i32/i64 与有限 f32/f64 张量验证 `a.add(&b) == b.add(&a)` |
| NaN 传播：数值型逐元素运算遇到 NaN 输入时输出按 IEEE 754 传播 NaN         | 构造含 NaN 的张量，验证 sin/sqrt/add/mul 等数值型逐元素运算结果含 NaN |
| 标量运算逆元：`a.add_scalar(k).sub_scalar(k) == a`                      | 对整数与有限浮点随机张量和标量值验证                      |
| 取反对合：`a.neg().neg() == a`                                          | 对所有 `Numeric` 支持类型验证                             |

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
| `test_large_tensor_add_parallel` | 大张量（`10^7` 量级元素）在串行/并行配置下结果与 shape 一致 | 高     |
| `test_high_rank_broadcast`     | 高 rank 动态维张量广播逐元素运算保持正确 shape 与元素对应 | 高     |
| `test_inf_math_functions`      | `Inf` / `-Inf` 输入遵循 IEEE 754 语义   | 高     |
| `test_div_i32_by_zero_panics`  | 整数除零触发带诊断的 panic              | 高     |
| `test_abs_i32_min_panics`      | `abs(i32::MIN)` 触发带诊断的 panic      | 高     |

### 8.3 边界测试场景

| 场景                  | 预期行为                                   |
| --------------------- | ------------------------------------------ |
| 空张量 `shape=[0, 3]` | add 返回空张量                             |
| 单元素张量            | 所有运算正确                               |
| 大张量 `len ≈ 10^7`   | `add` / `mul` 在默认与 `parallel` 配置下均保持 shape、错误类别与数值语义一致 |
| 高 rank `IxDyn` 输入  | 广播与逐元素结果 shape 正确，遍历不越界     |
| NaN 输入（f32/f64）   | NaN 传播（sin(NaN)=NaN, 0\*NaN=NaN）       |
| Inf 输入              | exp(Inf)=Inf, ln(0)=-Inf                   |
| 广播形状不兼容        | 返回 `XenonError::BroadcastError`；诊断上分别区分 `input_shape(lhs)` 与 `other_shape(rhs)`，不得把 `target_shape` 复用于右操作数 shape |
| 非连续输入（切片后）  | 运算结果与连续输入一致                     |
| 整数除零 / 最小值绝对值 | panic 信息至少包含 `operation`、`type`、`trigger`、`element_index` 与适用 `shape` |

### 8.4 集成测试

| 测试文件             | 测试内容                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_math.rs` | 二元逐元素辅助路径 / 标量路径与 `iter`、`broadcast`、`tensor`、`simd` backend 的端到端集成 |

### 8.5 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 所有逐元素运算走标量 / fallback 路径且语义满足文档约束。 |
| 启用 `simd`（`simd = ["dep:pulp"]`） | 连续输入上的 SIMD 分发结果与默认配置保持一致，非连续输入仍正确回退。 |
| 启用 `parallel`（`parallel = ["dep:rayon"]`） | 大输入上的并行逐元素路径与默认配置保持相同 shape、错误类别与数值语义，并遵守阈值与无嵌套并行约束。 |
| 同时启用 `simd,parallel` | 并行 chunk 内可局部选择 SIMD 或标量，但对外语义仍与默认配置一致。 |

### 8.6 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `bool` 不参与算术运算 | 编译期测试或 trait 约束验证。 |
| `lt` / `gt` 对 `i32` / `i64` / `f32` / `f64` 开放，但对 `bool` / `Complex` 关闭 | 编译期测试。 |
| mixed-type 逐元素运算不属于当前公开范围 | API 缺失断言或编译期失败测试。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向               | 对方模块    | 接口/类型                                  | 约定                                                                                                                                          |
| ------------------ | ----------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `math → iter`      | `iter`      | `Elements`, `ElementsMut`                  | 逐元素运算复用 `iter()` / `iter_mut()` 及相关遍历入口；二元路径直接遍历广播后的视图（参见 `10-iterator.md` §4）                               |
| `math → broadcast` | `broadcast` | `broadcast_shape()`                        | 二元运算先调用广播模块推导兼容视图（参见 `15-broadcast.md` §4）                                                                               |
| `math → element`   | `element`   | `Numeric` / `RealScalar` / `ComplexScalar` | 通过元素约束区分数值与复数运算语义（参见 `03-element.md` §4）                                                                                 |
| `math → simd`      | `simd`      | SIMD backend dispatch facade               | 连续数组且 feature 开启时通过稳定的 backend facade 分发到 SIMD 或标量路径，`math` 不直接依赖具体 vector kernel 名称（参见 `08-simd.md` §4.5） |
| `math → parallel`  | `parallel`  | `par_zip_map()` / threshold / guard        | 大输入时二元逐元素运算可委托并行后端；若阈值未达到或 guard 失败则回退标量（参见 `09-parallel.md` §5 / `§6`） |

### 9.2 数据流描述

```text
User calls add / unary op / comparison method
    │
    ├── math selects unary, binary, or scalar execution
    ├── binary ops validate broadcast compatibility first
    ├── iter produces element streams from shape + strides
    ├── parallel dispatch is used only when threshold/guard checks pass
    └── SIMD dispatch is used only when enabled and semantically equivalent
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 广播不兼容时返回 `XenonError::BroadcastError`；诊断上分别区分 `input_shape(lhs)` 与 `other_shape(rhs)`，不得把 `target_shape` 复用于右操作数 shape。参数不满足公开前提时返回 `XenonError::InvalidArgument { operation: Cow<'static, str>, argument: Cow<'static, str>, expected: Cow<'static, str>, actual: Cow<'static, str>, axis: Option<usize>, shape: Option<Vec<usize>> }`。 |
| Panic | 整数 `add/sub/mul/div`、标量版 `add_scalar/sub_scalar/mul_scalar/div_scalar`、`abs/neg/square` 的溢出、除零或结果不可表示均按需求触发 panic；`signum` 不新增 panic 约束。panic 信息至少包含 `operation`、`type`、`trigger`、`element_index`，并在适用时附带 `shape`。推荐格式：`Xenon: {operation} overflow for {type} at element_index={i}, shape={shape}, trigger={trigger}`。 |
| 路径一致性 | 标量、SIMD 与并行路径必须保持相同 shape、错误类别、NaN/复数语义；不满足前提或 guard 失败时统一回退标量实现。 |
| 容差边界 | `floor` / `ceil` 的 SIMD/并行路径结果必须与标量路径逐元素完全一致，不允许容差；`sqrt` / `exp` / `ln` / `sin` 以及其他 SIMD 容差约束参见 `08-simd.md`。复数结果按实部、虚部分量分别应用对应实数容差规则；仅在可证明语义等价或满足文档化容差时启用 SIMD，否则回退标量路径。 |

---

## 11. 设计决策记录（ADR）

### 决策 1：不在当前版本公开通用映射 helper

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | 当前版本不把更通用的逐元素映射基础设施纳入公开 API 承诺 |
| 理由     | 需求说明书 §12 仅要求明确列出的逐元素运算，不要求额外的通用映射原语 |
| 替代方案 | 直接在本期暴露完整映射 helper 集合 |
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
| 决策     | 连续 + 对齐内存时，`math` 仅对 `08-simd.md §5.4a` 覆盖矩阵列出的子集委托 SIMD backend；未列出的逐元素运算保持标量路径 |
| 理由     | SIMD 路径只在连续内存上有意义；非连续时标量路径更简单正确 |
| 替代方案 | 所有路径都用标量                                          |
| 拒绝原因 | 性能差距显著（2-4x），科学计算用户期望高性能              |

> **补充**：SIMD 实现位于独立 backend 模块 `src/simd/`，`math/` 仅按连续性和 feature gate 决定是否委托该 backend；逐元素运算的 SIMD 设计细节见 `08-simd.md`。若某个操作在当前类型或 ISA 上尚无满足语义约束的 SIMD kernel，则自动回退标量实现。

---

## 12. 性能考量

### 12.1 SIMD 加速预期

| 操作         | 标量路径 | SIMD 路径（AVX2）  | 加速比 |
| ------------ | -------- | ------------------ | ------ |
| add f32 (1M) | ~2ms     | ~0.5ms             | 4x     |
| mul f64 (1M) | ~3ms     | ~1ms               | 3x     |
| sin f64 (1M) | ~20ms    | 标量回退（≈20ms）  | ≈1.0x  |

### 12.2 复杂度标注

- 二元逐元素执行骨架：O(n) 时间，O(n) 空间
- 广播操作: O(n) 时间，O(n) 空间（结果），广播本身零拷贝

---

## 13. 平台与工程约束

| 项目       | 约束                                                                                           |
| ---------- | ---------------------------------------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性                                     |
| crate 结构 | 保持单 crate 结构，不拆分独立 math crate                                                       |
| SemVer     | 逐元素方法签名、支持类型集合、广播错误类别以及整数 panic 诊断字段均属于稳定契约；后续新增优化路径不得改变这些公开语义 |
| 依赖约束   | 仅允许项目基线中的可选 SIMD / 并行依赖，不新增额外第三方数学库                                 |
| 线程安全   | 所有逐元素运算接受 `&self`（一元/比较）或 `&self` + `&TensorBase`（二元）；这些调用能否在线程间安全共享或传递，取决于元素类型与底层存储模式是否满足相应 `Send` / `Sync` 前提。`get_unchecked` 等 unsafe 方法仍要求调用方保证独占访问。 |
| 范围边界   | 当前版本仅覆盖需求说明书 §12 明确列出的逐元素运算；通用映射 helper 与实复混合公开 API 不在本期范围内 |

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
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-15 |
| 1.2.4 | 2026-04-15 |
| 1.2.5 | 2026-04-15 |
| 1.3.0 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
