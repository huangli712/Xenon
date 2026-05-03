# 逐元素运算模块设计

> 文档编号: 11
> 模块目录: src/math/
> 任务阶段: Phase 4
> 前置文档: 03-element.md, 08-simd.md, 09-parallel.md, 10-iterator.md, 15-broadcast.md, 26-error.md
> 需求参考: 需求说明书 §4、§9、§12、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责     | 包含                                                                  |
| -------- | --------------------------------------------------------------------- |
| 算术运算 | add/sub/mul/div，数值类型：i32/i64/f32/f64/Complex                    |
| 一元运算 | abs（有序数值）；signum（浮点按符号位、整数按比较）；neg/square（Numeric）；数学函数（RealScalar） |
| 数学函数 | sin/sqrt/exp/ln/floor/ceil，仅 f32/f64                                |
| 复数运算 | modulus/模（返回实数类型）/conjugate（公开 API；内部 Complex 方法名可记为 conj），仅 Complex |
| 逻辑非   | `!`，仅 bool                                                          |
| 比较运算 | `equal`/`not_equal` 对所有 Element 可用；`less`/`greater` 对 i32/i64/f32/f64 可用，返回 bool 张量，NaN 遵循 IEEE 754 |
| 标量运算 | 标量与张量的逐元素运算                                                |
| 广播支持 | 所有二元运算和比较运算支持广播                                        |

| 职责     | 不包含                                                  |
| -------- | ------------------------------------------------------- |
| 算术运算 | 归约运算（参见 `13-reduction.md §1`） |
| 一元运算 | 筛选/排序                                               |
| 数学函数 | 运算符重载（参见 `19-overload.md §1`）                  |
| 复数运算 | 比较运算（`less`/`greater`；`equal`/`not_equal` 对复数仍可用）|
| 逻辑非   | 位运算                                                  |
| 比较运算 | 搜索/排序                                               |
| 标量运算 | 矩阵运算（dot/matmul）                                  |
| 广播支持 | 批量运算                                                |

### 1.2 设计原则

| 原则         | 体现                                          |
| ------------ | --------------------------------------------- |
| 类型安全边界 | 算术运算仅支持 `Numeric`，bool 编译时排除     |
| 广播透明集成 | 所有二元运算自动支持广播                      |
| 存储模式无关 | 对 Tensor、TensorView、TensorViewMut 统一工作 |
| NaN 语义明确 | IEEE 754 NaN 传播规则                         |

---

## 2. 需求映射与范围约束

| 项目     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §4、§9、§12、§27、§28 |
| 范围内   | 逐元素算术、一元运算、数学函数、复数 `modulus` / `conjugate`、逻辑非、比较运算、标量-张量逐元素语义与广播语义。当前版本的数学函数集合**仅包含** `sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil`。 |
| 范围外   | 混合类型逐元素运算以及 `map` 系列公开 API。**当前版本的数学函数集合不包含**：`cos` / `tan` / `asin` / `acos` / `atan` / `atan2` / `sinh` / `cosh` / `tanh` / `asinh` / `acosh` / `atanh` 等其他三角与双曲函数；`log2` / `log10` / `log1p` / `expm1` / `exp2`；`pow` / `powi` / `powf` / `cbrt` / `hypot` 等幂/根函数；`round` / `trunc` / `fract` 等取整变体；任何 special functions（`erf` / `gamma` / `lgamma` / `bessel*` 等）。这些函数若有需求需单独引入议题评估（包括 NaN / 边界 / SIMD admission / `f32` vs `f64` 精度策略）；不在当前版本作为隐式扩展加入。SIMD 与并行覆盖范围仅限本模块负责的逐元素运算；若当前类型/ISA/语义约束不满足，则自动回退标量。 |
| 非目标   | 不新增新的数学库依赖，不在本文扩展 mixed-type API 或更通用的逐元素映射原语。 |

---

## 3. 文件位置

```
src/math/
├── mod.rs              # module entry, re-export public APIs
├── binary.rs           # binary arithmetic methods and shared binary execution skeleton
├── unary.rs            # unary operations (abs, neg, signum, square, sin, sqrt, exp, ln, floor, ceil, modulus, conjugate, not)
└── comparison.rs       # comparison operations (equal, not_equal, less, greater)

Optional dependency touchpoints:
src/simd/               # optional SIMD backend consumed by math dispatch
src/parallel/           # optional parallel backend consumed by math dispatch
```

多文件设计理由：按操作元数分组（一元 vs 二元）可保持当前最小范围；更通用的逐元素映射基础设施不属于 `需求说明书 §12` 的本期最小交付，暂不纳入当前版本。运算符重载（Add/Sub/Mul/Div trait 实现）保留在 `src/overload/arithmetic.rs`。SIMD 加速由独立 backend 模块 `src/simd/` 承载，`math/` 仅负责语义 API 与分发入口。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

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

### 4.2 类型级依赖

| 来源模块       | 使用的类型/trait                                                                       |
| -------------- | -------------------------------------------------------------------------------------- |
| `tensor`       | `TensorBase<S, D>`, `Tensor<A, D>`, `TensorView`, `.shape()`（参见 `07-tensor.md §5`） |
| `iter`         | `Elements`, `ElementsMut`（参见 `10-iterator.md §5`）                                  |
| `element`      | `Element`, `Numeric`, `RealScalar`, `ComplexScalar`, `OrderedCompareElement`（定义见 `03-element.md §5.5`）|
| `complex`      | `Complex<f32>`, `Complex<f64>`（参见 `04-complex.md §5`）                              |
| `broadcast`    | `broadcast_shape()`, `broadcast_to()` 返回的 `TensorView`（参见 `15-broadcast.md §5`） |
| `dimension`    | `BroadcastDim<E>` public sealed trait（对外可命名的公开 sealed trait，用于编译期维度推导，参见 `02-dimension.md §5.10`）|
| `storage`      | `Storage<Elem = A>`, `StorageMut<Elem = A>`                                            |
| `error`        | `XenonError`（含 `BroadcastError` 变体，参见 `26-error.md §5`）                        |
| `dispatch`（内部） | `select_exec_path()`、`ExecPath`、`ParallelGuard`（`select_exec_path` 返回 `(ExecPath, Option<ParallelGuard>)`，与 30-dispatch v1.1.0 select-and-enter 原子裁决一致；旧 `should_parallelize()` 已废弃） |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd.md §5`）                                                   |
| `parallel`（可选） | `par_zip_map()`（纯并行执行入口，不含串行回退，参见 `09-parallel.md §5` / `§6`）   |

### 4.3 依赖合法性

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无   |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

### 4.4 依赖方向声明

依赖方向：单向向上。`math` 模块消费 `iter`、`tensor`、`element`、`broadcast` 模块，不被它们依赖。

---

## 5. 公共 API 设计

### 5.1 范围边界说明

更通用的逐元素映射基础设施不在 `需求说明书 §12` 的当前最小范围内。当前版本文档不将其作为公开 API 承诺；如后续需要，应以独立议题重新评估与类型转换、就地修改、错误语义的边界关系。

### 5.2 二元逐元素执行约定

二元逐元素方法统一使用 `BroadcastDim<DB>` 进行编译期维度推导；`BroadcastDim` 是 public sealed trait，因此在公开 API 中可被外部稳定命名。该 trait 定义于 `02-dimension.md §5.10`，详见该文档。

当前版本不承诺独立的通用二元逐元素 helper 公开函数。二元算术、比较与内部辅助路径统一采用"先广播，再直接遍历广播后视图并写入结果张量"的执行模型。调度模型：由 `dispatch.rs` 通过 `let (path, guard) = dispatch::select_exec_path(...)` 统一决定串行 / SIMD / 并行路径（参见 30-dispatch.md v1.1.0 决策 7）；进入并行路径后，单个 worker chunk 内**可以**独立调用 SIMD 后端 kernel（v2.0 起，参见 08-simd.md v2.0.0 决策 5），即"thread × SIMD 双层加速"模型。串行路径下 SIMD 由 `simd` 后端按其 admission 规则独立判断是否启用；不进入 SIMD 时回退到该路径内的标量循环。

### 5.3 算术运算（Numeric 约束）

```rust,ignore
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
        E: Dimension;

    /// Element-wise subtraction.
    pub fn sub<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension;

    /// Element-wise multiplication.
    pub fn mul<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension;

    /// Element-wise division.
    pub fn div<S2, E>(&self, other: &TensorBase<S2, E>)
        -> Result<Tensor<A, <D as BroadcastDim<E>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<E>,
        E: Dimension;
}
```

**Trait bound 简化说明**：旧版方法签名重复写 `A: Numeric + Copy + Add<Output = A>` 等，但 `Numeric: Add + Sub + Mul + Div + Neg`（参见 `03-element.md §5.2` super-trait 定义），且 `Numeric: Element: Copy`，因此 `Add` / `Sub` / `Mul` / `Div` / `Copy` 全部由 `A: Numeric` 间接保证；新版直接以 `A: Numeric` 表达，避免冗余 bound 与"签名 trait 与实现 trait 不闭合"的歧义。整数路径的 checked 语义不通过 `Add<Output = A>` 等原生运算符落实，而是通过 `crate::element::Checked*` 原语显式包装（§5.3 末尾示例）。

所有整数逐元素运算在实现层使用此 trait，确保 debug 和 release 均在溢出/除零时 panic。浮点和复数使用标准算术运算符。

- 支持的类型：i32, i64, f32, f64, Complex<f32>, Complex<f64>。
- 对 `i32` / `i64` 的 `add` / `sub` / `mul` / `div`，实现必须使用 checked arithmetic；凡发生溢出、除以零或结果不可表示，均按 `需求说明书 §12` 与 `需求说明书 §27` 走 panic 语义，不得回落为 wrapping 行为。
- 整数 checked arithmetic 直接复用 element 层原语，**不在 math 模块内部定义同语义 trait**：使用 `crate::element::{CheckedAdd, CheckedSub, CheckedMul, CheckedNeg, CheckedDiv}`（权威定义见 `03-element.md §5.9`）。这些原语返回 `Option<Self>`；math 模块的整数路径将 `None` 翻译为 panic，由此实现"在 debug 与 release 均在溢出/除零时 panic"的语义。

```rust,ignore
// Inside math implementation (illustrative; not a new trait):
use crate::element::{CheckedAdd, CheckedSub, CheckedMul, CheckedDiv};

#[inline]
fn add_or_panic<A: CheckedAdd>(a: A, b: A) -> A {
    match a.checked_add(b) {
        Some(value) => value,
        None => panic!(
            "integer overflow in element-wise add: operation=add, trigger=overflow",
        ),
    }
}
// Analogous wrappers for sub / mul / div reuse element-layer primitives.
```

### 5.4 一元运算（分离 trait bounds）

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + OrderedCompareElement,
{
    pub fn abs(&self) -> Tensor<A, D>;
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

impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Numeric + OrderedCompareElement,
    D: Dimension,
{
    /// Element-wise signum.
    ///
    /// **Integer types (`i32`, `i64`)**: returns `-1` for negative,
    /// `0` for zero, `1` for positive (purely sign-based).
    ///
    /// **Floating-point types (`f32`, `f64`)**: follows IEEE 754 /
    /// `f32::signum` / `f64::signum` exactly:
    ///
    ///   - `signum(NaN)`  = `NaN`
    ///   - `signum(+0.0)` = `+1.0`
    ///   - `signum(-0.0)` = `-1.0`
    ///   - positive normal/subnormal/Inf → `+1.0`
    ///   - negative normal/subnormal/Inf → `-1.0`
    ///
    /// The integer "zero → 0" and float "+0.0 → +1.0 / -0.0 → -1.0"
    /// rules are deliberately different; they are not contradictions.
    /// See 03-element.md §5.3 for the per-element-type contract.
    pub fn signum(&self) -> Tensor<A, D>;
}
```

- `abs` / `signum` 仅对具备自然顺序的数值类型开放：i32, i64, f32, f64。
- `neg` / `square` 对所有 `Numeric` 类型开放：i32, i64, f32, f64, Complex<f32>, Complex<f64>。
- `abs()` 约束说明：`OrderedCompareElement` 限定到 i32/i64/f32/f64 四种类型，与 abs 的实际支持范围严格匹配。
- `signum()` trait bound 修正：旧版仅要求 `A: OrderedCompareElement`，无法表达"返回 -1 / 0 / 1"所需的常量构造能力。新版 bound 为 `A: Numeric + OrderedCompareElement`，由 `Numeric: Element` 提供 `A::zero()`、`A::one()`，由 `Numeric` 的 `Neg` 提供 `-A::one()`；浮点路径直接调用 `RealScalar::signum`（不使用 `-A::one()`）。`OrderedCompareElement` 限定到 `i32/i64/f32/f64`，`Numeric` 不引入复数路径（`Complex<T>` 不实现 `OrderedCompareElement`，编译期已被排除）。
- 对有符号整数，`neg(i32::MIN)` / `neg(i64::MIN)` 等不可表示情形视为不可恢复错误，遵循 panic 语义；实现层使用 `crate::element::CheckedNeg::checked_neg`（参见 `03-element.md §5.10`），`None` 翻译为 panic。
- `abs` 在整数路径上的 checked 推导：对 `A: i32 / i64`，`abs(x) := if x >= A::zero() { x } else { x.checked_neg().expect("integer overflow in abs") }`，等价于"在最小负值处溢出 → panic"。无需新增 `CheckedAbs` trait。
- `square` 在整数路径上必须使用 `CheckedMul`；溢出 → panic。
- `signum` 仅做符号分类，不额外要求 checked arithmetic。

### 5.5 数学函数（RealScalar 约束：仅 f32/f64）

```rust,ignore
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

- `sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil` 使用 Rust 提供的数学能力，不引入外部数学 crate。
- 精确类（`floor` / `ceil`）：结果须与标量路径逐元素一致。
- 近似类（`sin` / `sqrt` / `exp` / `ln`）：以 `需求说明书 §28.3` 为权威基线；实现细节参见 `00-coding.md §8.4`。
- 同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。

### 5.6 复数运算（ComplexScalar 约束）

```rust,ignore
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

- 公开张量 API 统一使用 `conjugate()`（与 `Numeric::conjugate()` 保持一致）；`conj` 仅允许作为内部 `Complex` 方法名或实现细节出现，不构成公开 API 命名承诺。

**关于实数张量的 conjugate()**：实数（`i32`/`i64`/`f32`/`f64`）类型的共轭等于自身。Xenon 不为实数张量提供 `conjugate()` 入口，要求显式调用避免冗余 API。如需统一处理，使用 `Numeric::conjugate()`（标量级）或在泛型代码中通过 trait bound 调用。
- `modulus()` 对应 `需求说明书 §12` 中的“模”运算。`Complex<f32> → f32`，`Complex<f64> → f64`。
- 参与逐元素运算或比较的双方元素类型须预先一致。因此，`Complex<T>` 与实数标量的混合张量 API（如 `add_real_scalar` / `mul_real_scalar`）不属于当前公开范围；若内部实现需要复用相应标量逻辑，也只能作为不对外承诺的内部辅助路径存在。

### 5.7 逻辑非（仅 bool）

```rust,ignore
impl<S, D> TensorBase<S, D>
where
    S: Storage<Elem = bool>,
    D: Dimension,
{
    /// Logical NOT.
    pub fn not(&self) -> Tensor<bool, D>;
}
```

### 5.8 比较运算（NumPy 风格命名）

- `equal` / `not_equal` 对所有元素类型可用（包括 `bool` 与 `Complex`）。
- `less` / `greater` 的需求级支持范围固定为 `i32`、`i64`、`f32`、`f64`，返回 `Tensor<bool, _>`。
- `bool` 与 `Complex` 类型不支持 `less` / `greater`。

**命名规则（设计决策 4，对齐 NumPy）**：
公开 API 不使用 `eq` / `ne` / `lt` / `gt` 这组缩写命名，避免与 Rust 标准库 `PartialEq::eq`、`PartialOrd::lt` 等同名 trait 方法在调用语法、文档自动链接、IDE 跳转上产生命名冲突。Rust 标准库这些方法返回 `bool`（标量布尔），而 Xenon 张量比较方法返回 `Tensor<bool, _>`（逐元素布尔张量），语义不同；用 NumPy 风格的全词命名（`equal` / `not_equal` / `less` / `greater`）让张量逐元素比较与标量布尔比较在调用点上明确可区分。

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    /// Element-wise equality comparison, returns a bool tensor.
    /// NaN comparison follows IEEE 754: `equal(NaN, NaN)` is element-wise `false`.
    pub fn equal<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;

    /// Element-wise inequality comparison; `not_equal(NaN, NaN)` is element-wise `true`.
    pub fn not_equal<S2, DB>(&self, other: &TensorBase<S2, DB>)
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
    pub fn less<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;

    /// Element-wise greater-than comparison.
    ///
    /// Supported ordered element types are i32, i64, f32, and f64.
    pub fn greater<S2, DB>(&self, other: &TensorBase<S2, DB>)
        -> Result<Tensor<bool, <D as BroadcastDim<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDim<DB>,
        DB: Dimension;
}
```

- `less` / `greater` 不再复用 `RealScalar` 或更宽泛的 `Numeric + PartialOrd` 约束；公开 API 以 `OrderedCompareElement` 明确收敛到 `i32`、`i64`、`f32`、`f64` 四类元素类型。该 trait 定义见 `03-element.md §5.5`。
- `equal(NaN, NaN)` 在浮点类型上每个 lane 返回 `false`，`not_equal(NaN, NaN)` 返回 `true`，遵循 IEEE 754；这与 Rust 标准库 `f64::partial_cmp(NaN, NaN) == None` 一致。
- 标量比较入口与 `需求说明书 §12` 一致，比较运算也提供标量-张量入口；标量按可广播到目标全形状的零维输入处理，因此成功路径的形状与对应张量输入版本一致。

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    pub fn equal_scalar(&self, scalar: A) -> Tensor<bool, D>;
    pub fn not_equal_scalar(&self, scalar: A) -> Tensor<bool, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: OrderedCompareElement,
{
    pub fn less_scalar(&self, scalar: A) -> Tensor<bool, D>;
    pub fn greater_scalar(&self, scalar: A) -> Tensor<bool, D>;
}
```

### 5.9 标量与张量运算

```rust,ignore
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

- 标量版算术方法与张量-张量运算遵循相同的 checked arithmetic 语义：有符号整数溢出、除以零、结果不可表示均遵循 panic 语义。
- 标量与张量之间的逐元素运算，标量按可广播到目标张量全形状的零维输入语义处理，统一经由广播路径实现，不另起独立语义。

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

`modulus()` 的内部执行骨架与标准一元运算不同：输入元素类型为 `Complex<T>`，输出为 `T`。因此它不能直接复用 `apply_unary(view, f)` 这类“输入/输出同类型”的骨架，而需要独立的执行骨架处理类型变化。

### 6.2 二元逐元素实现（含广播）

```
apply_binary(a, b, f):
    broadcast_shape = broadcast_shape(a.shape(), b.shape())?
    a_broadcast = a.broadcast_to(broadcast_shape)?
    b_broadcast = b.broadcast_to(broadcast_shape)?
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

### 6.3 SIMD 与并行加速路径

本文描述的逐元素运算功能范围以 `需求说明书 §12` 为准。SIMD 和并行加速路径的当前正式支持子集以 `08-simd.md` 和 `09-parallel.md` 定义的能力边界为准，不在本文档中另行扩张覆盖承诺。

调度模型（v2.0 起，与 30-dispatch v1.1.0、08-simd v2.0.0、09-parallel v2.0.0 协同）：

1. 由 `dispatch::select_exec_path(...)` 返回 `(ExecPath, Option<ParallelGuard>)`，三路 `Serial / Simd / Parallel` 互斥裁决。
2. 若选中 `Serial` 或 `Simd`：在串行执行上下文中由 `simd` 后端按 `08-simd.md §5.4` admission 规则独立决定是否进入 SIMD kernel；不进入时走标量循环。
3. 若选中 `Parallel`：调用方将 `Some(guard)` 按值移交到 `parallel` 后端入口；每个 worker 拿到 chunk 后**可以**独立调用 SIMD 后端 kernel（即 worker 内 SIMD admission，与上一条路径上的 SIMD admission 同源），这是 v2.0 起新增的"thread × SIMD 双层加速"能力（08-simd v2.0.0 决策 5、09-parallel v2.0.0 决策 9）。chunk 间合并顺序仍由 `parallel` 模块的固定 chunking + 固定 merge tree 控制。
4. 未列出的运算、类型、ISA 或不满足语义约束的路径统一回退标量实现。

| 操作类别 | SIMD 状态 | 并行状态 |
| -------- | --------- | -------- |
| 算术（`+ - * /`） | 覆盖：仅在 `08-simd.md` 定义的正式支持子集内尝试 SIMD；其余情况回退标量 | 覆盖：仅在 `09-parallel.md` 定义的正式支持子集内尝试并行；其余情况回退串行 |
| 一元（`neg` / `abs` / `signum` / `square`） | 覆盖：仅在 `08-simd.md` 定义的正式支持子集内尝试 SIMD；其余情况回退标量 | 覆盖：仅在 `09-parallel.md` 定义的正式支持子集内尝试并行；其余情况回退串行 |
| 比较（`equal` / `not_equal` / `less` / `greater`） | 覆盖：仅在 `08-simd.md` 定义的正式支持子集内尝试 SIMD；其余情况回退标量 | 覆盖：仅在 `09-parallel.md` 定义的正式支持子集内尝试并行；其余情况回退串行 |
| 数学（`sin` / `sqrt` / `exp` / `ln` / `floor` / `ceil`） | 覆盖：仅在 `08-simd.md` 定义的正式支持子集内尝试 SIMD，否则回退标量 | 覆盖：仅在 `09-parallel.md` 定义的正式支持子集内尝试并行；worker 内 SIMD 是否启用由 chunk 内独立 admission 决定 |
| 复数（`modulus` / `conjugate`） | 覆盖：仅在 `08-simd.md` 定义的正式支持子集内尝试 SIMD；其余情况回退标量 | 覆盖：仅在 `09-parallel.md` 定义的正式支持子集内尝试并行；其余情况回退串行 |
| 逻辑（`not`） | 覆盖：仅在 `08-simd.md` 定义的正式支持子集内尝试 SIMD；其余情况回退标量 | 覆盖：仅在 `09-parallel.md` 定义的正式支持子集内尝试并行；其余情况回退串行 |

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
  - 前置: T1, 10-iterator.md, broadcast 模块
  - 预计: 10 min

- [ ] **T3**: 实现一元运算（abs/neg/signum/square）
  - 文件: `src/math/unary.rs`
  - 内容: 基于统一逐元素遍历骨架实现一元运算，并为整数路径补齐 checked arithmetic
  - 测试: `test_abs`, `test_neg`, `test_signum`, `test_square`
  - 前置: T1, 10-iterator.md
  - 预计: 10 min

- [ ] **T4**: 实现数学函数（sin/sqrt/exp/ln/floor/ceil）
  - 文件: `src/math/unary.rs`
  - 内容: RealScalar 约束的数学方法
  - 测试: `test_sin`, `test_sqrt`, `test_exp`, `test_floor_ceil`
  - 前置: T1, 10-iterator.md
  - 预计: 10 min

- [ ] **T5**: 实现复数操作（`conjugate`）与复数数学函数（`modulus`）
  - 文件: `src/math/unary.rs`
  - 内容: `conjugate` 与 `modulus` 的范围内实现
  - 测试: `test_modulus`, `test_conjugate`
  - 前置: T1, 10-iterator.md
  - 预计: 10 min

### Wave 2: 算术与比较运算

- [ ] **T6**: 实现算术运算（add/sub/mul/div）
  - 文件: `src/math/binary.rs`
  - 内容: 基于共享二元逐元素执行骨架的算术运算，标量版本
  - 测试: `test_add_i32`, `test_add_f64`, `test_add_complex`, `test_mul_scalar`
  - 前置: T2
  - 预计: 10 min

- [ ] **T7**: 实现逻辑非（not）和比较运算（`equal` / `not_equal` / `less` / `greater`）
  - 文件: `src/math/unary.rs`（not）, `src/math/comparison.rs`（`equal`/`not_equal`/`less`/`greater`）
  - 内容: bool 取反、比较运算返回 bool 张量
  - 测试: `test_not_bool`, `test_equal_f64`, `test_less_i32`, `test_nan_comparison`
  - 前置: T2
  - 预计: 10 min

### Wave 3: SIMD 集成

- [ ] **T8**: 接入 SIMD backend 统一分发
  - 文件: `src/math/binary.rs`, `src/math/unary.rs`, `src/math/comparison.rs`, `src/simd/vector.rs`
  - 内容: 在 `math` 中接入独立 `simd/` backend 的统一分发点；具体操作、类型和 ISA 覆盖以 `08-simd.md` 的正式覆盖矩阵与 admission 规则为准
  - 测试: `test_add_simd_vs_scalar`, `test_mul_simd_vs_scalar`
  - 前置: T3, 08-simd.md
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                      | 说明                                     |
| -------- | ------------------------- | ---------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests`  | 验证逐元素算术、一元运算、数学函数、比较运算与复数运算 |
| 集成测试 | `tests/test_math.rs`      | 验证 `math` 与 `iter`、`broadcast`、`tensor`、`simd` backend 的端到端集成 |
| 边界测试 | 同模块测试中标注          | 覆盖空张量、大张量、高维广播、NaN/Inf、非连续输入及整数 panic 场景 |
| 属性测试 | `tests/property_tests.rs` | 验证加法交换律、NaN 传播、标量逆元与取反对合不变量 |

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
| `test_modulus`                 | Complex{3,4}.modulus() = 5.0             | 高     |
| `test_conjugate`               | Complex{1,2}.conjugate() = Complex{1,-2} | 中     |
| `test_not_bool`                | !true = false, !false = true             | 中     |
| `test_equal_f64`               | 逐元素相等比较（NumPy 风格命名）         | 高     |
| `test_less_i32`                | 逐元素小于比较（NumPy 风格命名）         | 高     |
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
| 空张量 `shape=[0, 3]` 的一元/二元/比较运算 | `sin` / `add` / `equal` 均返回空张量，shape 保持为 `[0, 3]` |
| rank-6 广播输入 `IxDyn([1,1,1,1,1,4])` 与 `IxDyn([2,1,3,1,1,4])` | 广播结果 shape 为 `IxDyn([2,1,3,1,1,4])`，逐元素对应关系正确 |
| `10^7` 元素张量 `add` / `mul` | 默认与 `parallel` 配置下结果 shape、错误类别与数值语义一致 |
| 大张量 `len ≈ 10^7`   | `add` / `mul` 在默认与 `parallel` 配置下均保持 shape、错误类别与数值语义一致 |
| 高 rank `IxDyn` 输入  | 广播与逐元素结果 shape 正确，遍历不越界    |
| NaN 输入（f32/f64）   | NaN 传播（sin(NaN)=NaN, 0\*NaN=NaN）       |
| Inf 输入              | exp(Inf)=Inf, ln(0)=-Inf                   |
| 广播形状不兼容        | 返回 `XenonError::BroadcastError`          |
| 非连续输入（切片后）  | 运算结果与连续输入一致                     |
| 整数除零 / 最小值绝对值 | panic 信息至少包含 `operation`、`type`、`trigger`、`element_index` 与适用 `shape` |

### 8.4 属性测试不变量

| 不变量                                                  | 测试方法                                                  |
| ------------------------------------------------------- | --------------------------------------------------------- |
| 加法交换律（整数与无 NaN 实数输入）                     | 对随机 i32/i64 与有限 f32/f64 张量验证 `a.add(&b) == b.add(&a)` |
| 数值型逐元素运算遇到 NaN 输入时输出按 IEEE 754 传播 NaN | 构造含 NaN 的张量，验证 sin/sqrt/add/mul 等数值型逐元素运算结果含 NaN |
| 标量运算逆元：`a.add_scalar(k).sub_scalar(k) == a`      | 对整数与有限浮点随机张量和标量值验证                      |
| 取反对合：`a.neg().neg() == a`                          | 对所有 `Numeric` 支持类型验证                             |

### 8.5 集成测试

| 测试文件             | 测试内容                                                                           |
| -------------------- | ---------------------------------------------------------------------------------- |
| `tests/test_math.rs` | 二元逐元素辅助路径 / 标量路径与 `iter`、`broadcast`、`tensor`、`simd` backend 的端到端集成 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 所有逐元素运算走标量 / fallback 路径且语义满足文档约束。 |
| 启用 `simd`（`simd = ["dep:pulp"]`） | 连续输入上的 SIMD 分发结果与默认配置保持一致，非连续输入仍正确回退。 |
| 启用 `parallel`（`parallel = ["dep:rayon"]`） | 大输入上的并行逐元素路径与默认配置保持相同 shape、错误类别与数值语义，并遵守阈值与无嵌套并行约束。 |
| 同时启用 `simd,parallel` | 串行路径上 SIMD admission 可生效；并行路径中每个 worker chunk 可独立做 SIMD admission，不满足条件时该 chunk 回退标量；对外语义仍与默认配置一致。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `bool` 不参与算术运算 | 编译期测试或 trait 约束验证。 |
| `less` / `greater` 对 `i32` / `i64` / `f32` / `f64` 开放，但对 `bool` / `Complex` 关闭 | 编译期测试。 |
| mixed-type 逐元素运算不属于当前公开范围 | API 缺失断言或编译期失败测试。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向               | 对方模块    | 接口/类型                                  | 约定                                   |
| ------------------ | ----------- | ------------------------------------------ | -------------------------------------- |
| `math → iter`      | `iter`      | `Elements`, `ElementsMut`                  | 逐元素运算复用 `iter()` / `iter_mut()` 及相关遍历入口；二元路径直接遍历广播后的视图（参见 `10-iterator.md` §5）|
| `math → broadcast` | `broadcast` | `broadcast_shape()`                        | 二元运算先调用广播模块推导兼容视图（参见 `15-broadcast.md` §5）|
| `math → element`   | `element`   | `Numeric` / `RealScalar` / `ComplexScalar` | 通过元素约束区分数值与复数运算语义（参见 `03-element.md` §5）|
| `math → simd`      | `simd`      | SIMD backend dispatch facade               | 连续数组且 feature 开启时通过稳定的 backend facade 分发到 SIMD 或标量路径，`math` 不直接依赖具体 vector kernel 名称（参见 `08-simd.md` §5） |
| `math → parallel`  | `parallel`  | `par_zip_map(.., guard, ..)` / `ParallelGuard` | `dispatch::select_exec_path()` 返回 `(ExecPath, Option<ParallelGuard>)`；选中 `Parallel` 时 `math` 把 `Some(guard)` 按值移交给 `parallel` 后端入口。worker 内允许独立调用 SIMD 后端 kernel（参见 `09-parallel.md` v2.0.0 §6.2 / §11 决策 9） |

### 9.2 数据流描述

```text
User calls add / unary op / comparison method
    │
    ├── math selects unary, binary, or scalar execution
    ├── binary ops validate broadcast compatibility first
    ├── let (path, guard) = dispatch::select_exec_path(...)
    │       ├── (Serial, None)        → scalar loop, may enter SIMD per backend admission
    │       ├── (Simd,   None)        → SIMD kernel by simd backend
    │       └── (Parallel, Some(g))   → parallel path; pass guard by value
    ├── parallel path: workers split logical work into chunks
    │       └── each chunk MAY call SIMD backend independently (v2.0 decision)
    └── iter produces element streams from shape + strides on each path
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 广播不兼容时返回 `XenonError::BroadcastError { operation, lhs_shape, rhs_shape, attempted_target_shape, axis }`（字段对齐 26-error v3.0.0 §5.1）。参数不满足公开前提时返回 `XenonError::InvalidArgument { operation, kind: InvalidArgumentKind::* }`，按操作族选择对应封闭枚举变体。 |
| Panic | 整数 `add/sub/mul/div`、标量版 `add_scalar/sub_scalar/mul_scalar/div_scalar`、`abs/neg/square` 的溢出、除零或结果不可表示均按需求触发 panic；`signum` 不新增 panic 约束。panic 信息至少包含 `operation`、`type`、`trigger`、`element_index`，并在适用时附带 `shape`。 |
| 路径一致性 | 标量、SIMD 与并行（含 worker 内 SIMD）路径必须保持相同 shape、错误类别、NaN/复数语义；不满足前提或 SIMD admission 失败时各路径内部回退到该路径上的标量实现，不跨路径切换。 |
| 容差边界 | 精确类（`floor` / `ceil`）结果须与标量路径逐元素一致。近似类（`sin` / `sqrt` / `exp` / `ln`）以 `需求说明书 §28.3` 为权威基线；实现细节参见 `00-coding.md §8.4`。复数结果按实部、虚部分量分别应用对应实数规则；同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。 |

---

## 11. 设计决策记录

### 决策 1：不在当前版本公开通用映射 helper

| 属性     | 值                                                             |
| -------- | -------------------------------------------------------------- |
| 决策     | 当前版本不把更通用的逐元素映射基础设施纳入公开 API 承诺        |
| 理由     | `需求说明书 §12` 仅要求明确列出的逐元素运算，不要求额外的通用映射原语 |
| 替代方案 | 直接在本期暴露完整映射 helper 集合 |
| 拒绝原因 | 会扩大 API 面且引入额外语义边界，不符合当前最小范围            |

### 决策 2：NaN 比较遵循 IEEE 754

| 属性     | 值                                                                    |
| -------- | --------------------------------------------------------------------- |
| 决策     | 比较运算（`equal` / `not_equal` / `less` / `greater`）遵循 IEEE 754 语义：NaN != NaN |
| 理由     | 与 Rust 标准库 `f64::partial_cmp` 行为一致；与 NumPy/ndarray 行为一致 |
| 替代方案 | 提供总排序比较（total_cmp）                                           |
| 拒绝原因 | 当前版本不需要总排序，可未来扩展                                      |

### 决策 3：SIMD 优化路径

| 属性     | 值                                                        |
| -------- | --------------------------------------------------------- |
| 决策     | 连续 + 对齐内存时，`math` 仅对 `08-simd.md §5.6` 覆盖矩阵列出的子集委托 SIMD backend；未列出的逐元素运算保持标量路径 |
| 理由     | SIMD 路径只在连续内存上有意义；非连续时标量路径更简单正确 |
| 替代方案 | 所有路径都用标量                                          |
| 拒绝原因 | 性能差距显著（2-4x），科学计算用户期望高性能              |

SIMD 实现位于独立 backend 模块 `src/simd/`，`math/` 仅按连续性和 feature gate 决定是否委托该 backend；逐元素运算的 SIMD 设计细节见 `08-simd.md`。若某个操作在当前类型或 ISA 上尚无满足语义约束的 SIMD kernel，则自动回退标量实现。

### 决策 4：比较运算采用 NumPy 风格命名（`equal/not_equal/less/greater`）

| 属性     | 值                                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 公开 API 使用 `equal` / `not_equal` / `less` / `greater`（及对应 `_scalar` 版本），不使用 `eq` / `ne` / `lt` / `gt` 缩写命名      |
| 理由     | Rust 标准库 `PartialEq::eq`、`PartialOrd::lt` 等同名 trait 方法返回标量 `bool`；张量逐元素比较返回 `Tensor<bool, _>`，语义不同。同名会让方法解析、文档自动链接、IDE 跳转产生歧义。NumPy 风格全词命名让张量逐元素比较与标量布尔比较在调用点上明确可区分。 |
| 替代方案 | 保留 `eq` / `ne` / `lt` / `gt` 命名 |
| 拒绝原因 | 与 Rust 习惯冲突；用户在泛型代码中无法靠类型签名区分张量比较与标量比较 |

### 决策 5：worker 内允许 SIMD（与 09-parallel v2.0.0 决策 9 / 08-simd v2.0.0 决策 5 协同）

| 属性     | 值                                                                                                                                      |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 进入并行路径后，单个 worker chunk 内可独立做 SIMD admission；chunk 间合并仍由 `parallel` 控制                                            |
| 理由     | 撤销 v1.x 的"并行 vs SIMD 互斥"，提供 thread × SIMD 双层加速，对大数组吞吐显著提升                                                     |
| 替代方案 | 保留 v1.x 设计（worker 内禁止 SIMD）                                                                                                    |
| 拒绝原因 | 与并行路径的可消费性能上限脱节；用户感知到的并行路径与串行 SIMD 路径在大数据量下相互妨碍                                                |

### 决策 6：标量算术 API trait bound 简化

| 属性     | 值                                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------- |
| 决策     | `add` / `sub` / `mul` / `div`（及标量版本）方法签名只声明 `A: Numeric`，不重复声明 `Add<Output = A>` / `Copy` 等  |
| 理由     | `Numeric: Element + Add + Sub + Mul + Div + Neg`（参见 03-element §5.2），`Numeric: Element: Copy`；重复 bound 制造"签名 trait 与实现 trait 不闭合"的歧义 |
| 替代方案 | 保留显式 `+ Copy + Add<Output = A>` 等                                                                            |
| 拒绝原因 | 冗余且容易误读为"非 Numeric 但 Add 的类型也允许"，不符合封闭元素集合 |

---

## 12. 性能考量

### 12.1 SIMD 加速预期（参考性，不作为契约）

下表为典型 AVX2 平台上的指示性测量结果，仅供性能基线参考；具体加速比因 ISA / 元素类型 / 数据量 / `feature` 配置而异。基准测试的权威覆盖与回归阈值见 `27-benchmark.md`。

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
| `std` only | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性                                     |
| MSRV       | Rust 1.85+                                                                                     |
| 单 crate   | 保持单 crate 结构，不拆分独立 math crate                                                       |
| SemVer     | 逐元素方法签名、支持类型集合、广播错误类别以及整数 panic 诊断字段均属于稳定契约；后续新增优化路径不得改变这些公开语义 |
| 最小依赖   | 仅允许项目基线中的可选 SIMD / 并行依赖，不新增额外第三方数学库                                 |

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
| 1.3.1 | 2026-04-16 |
| 2.0.0 | 2026-05-02 |
| 2.0.1 | 2026-05-03 |

### v2.0.1 (2026-05-03) — Medium/Low review fixes

- §5.3：将 checked arithmetic 示例从 `expect` 改为 `match` + 诊断化 panic 示意。
- §7 / §8.2：同步比较测试命名为 `equal` / `less`，并将 T8 改为接入 SIMD backend 统一分发，覆盖以 08-simd 为准。
- §8.6：同步 `simd,parallel` 组合配置下 worker chunk 独立 SIMD admission 的 v2.0 规则。

### v2.0.0 (2026-05-02) — SemVer breaking changes

> 本版本是与 26-error v3.0.0、30-dispatch v1.1.0、08-simd v2.0.0、09-parallel v2.0.0 协同的破坏性更新。

- §1.1、§5.8 / §5.8 标量版：比较方法重命名为 NumPy 风格 `equal` / `not_equal` / `less` / `greater`（及 `_scalar` 后缀变体），撤销 `eq` / `ne` / `lt` / `gt`（决策 4）。这是公开 API 的破坏性重命名。
- §5.2 / §6.3 / §9.1 / §9.2：调度模型对齐 30-dispatch v1.1.0 决策 7（`select_exec_path()` 返回 `(ExecPath, Option<ParallelGuard>)`）和 09-parallel v2.0.0 决策 9（worker 内允许 SIMD）；`math` 把 `Some(guard)` 按值移交给 `parallel` 后端入口（决策 5）。
- §5.3：算术方法 trait bound 简化为只写 `A: Numeric`，移除冗余的 `Copy` / `Add<Output = A>` 等（决策 6）；功能不变。
- §5.4：`signum` trait bound 从 `A: OrderedCompareElement` 加强为 `A: Numeric + OrderedCompareElement`，使 `-1 / 0 / 1` 的常量构造在 trait 层有承载（修复 H-R "signum bound 不足"）；浮点 / 整数 signum 语义在同一 doc 注释中并列展示，不再前后矛盾。
- §5.4：`abs` 整数 panic 推导改为基于 `crate::element::CheckedNeg`，无需新增 `CheckedAbs` trait；`square` 显式标注使用 `CheckedMul`。
- §10：错误字段引用对齐 26-error v3.0.0（`BroadcastError`、`InvalidArgument { kind: InvalidArgumentKind::* }`）。
- §11：新增决策 4 / 5 / 6。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
