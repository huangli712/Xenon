# 矩阵运算模块设计

> 文档编号: 12
> 模块目录: src/matrix/
> 任务阶段: Phase 4
> 前置文档: 00-coding.md, 03-element.md, 07-tensor.md, 08-simd.md, 09-parallel.md, 10-iterator.md, 11-math.md, 13-reduction.md, 26-error.md
> 需求参考: 需求说明书 §4、§9、§10、§13、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责      | 包含                                                                                         |
| --------- | -------------------------------------------------------------------------------------------- |
| 向量内积  | dot product（实数内积：sum(a[i] \* b[i])）                                                   |
| 复数内积  | 共轭线性定义（sum(conjugate(a[i]) \* b[i])）                                                 |
| SIMD 状态 | dot 可选接入 `simd` / `parallel` 能力                                                        |
| 错误处理  | 非 1D 输入返回 `XenonError::InvalidArgument`；长度不匹配返回 `XenonError::ShapeMismatch` |

| 职责      | 不包含         |
|---------- | -------------- |
| 向量内积  | 矩阵乘法、外积 |
| 复数内积  | 批量矩阵乘法   |
| SIMD 状态 | BLAS 绑定      |
| 错误处理  |  —             |

**关于模块命名**：当前版本 `matrix/` 只实现一维向量内积 `dot`，命名上比内容范围宽。这是有意保留：模块名先于完整线性代数能力固定下来，是为后续在不破坏 API 路径的前提下扩展矩阵乘法、批量矩阵运算等线性代数能力（这些扩展明确不在当前版本范围内，参见 §2 范围外条目）。当前版本 `dot()` 是模块的唯一公开 API。

### 1.2 设计原则

| 原则                               | 体现                                                               |
| ---------------------------------- | ------------------------------------------------------------------ |
| 最小范围                           | 当前仅实现向量内积，复杂线性代数由上游库通过 FFI 实现              |
| 错误恢复                           | 维度不匹配返回可恢复错误（`XenonError`）；整数溢出为不可恢复 panic |
| 语义优先                           | dot 先保证语义与错误契约一致                                       |
| 与上游 BLAS 集成预期的语义兼容前提 | 内存布局与内积语义保持可对接上游 BLAS 集成的预期前提               |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                     |
| -------- | ---------------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §4、§9、§10、§13、§27、§28                                                    |
| 范围内   | 向量内积 `dot`、复数共轭线性语义、形状检查、空向量单位元，以及可选 SIMD / 并行执行路径。 |
| 范围外   | 矩阵-矩阵乘法、外积、批量矩阵乘法、矩阵分解以及 BLAS/LAPACK 绑定。                       |
| 非目标   | 不把 `matrix` 扩展为通用线性代数层，不新增第三方线性代数依赖。                           |

---

## 3. 文件位置

```
src/matrix/
├── mod.rs              # module entry, re-exports, dot() public API
└── dot.rs              # vector dot-product implementation (scalar / SIMD / parallel dispatch)
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/matrix/
├── mod.rs
│   ├── crate::tensor        # TensorView<A, D>
│   ├── crate::element       # Numeric, ComplexScalar
│   ├── crate::iter          # Elements
│   ├── crate::dispatch      # ExecPath, select_exec_path() for execution path decision
│   └── crate::error         # XenonError
├── dot.rs
│   ├── crate::tensor        # TensorView<A, D>
│   ├── crate::element       # Numeric
│   ├── crate::iter          # Elements
│   ├── crate::dispatch      # select_exec_path(), ExecPath, ParallelGuard
│   ├── crate::error         # XenonError
│   ├── crate::simd (opt.)   # Pure vectorized dot kernel
│   └── crate::parallel (opt.) # Pure parallel dot execution
```

### 4.2 类型级依赖

| 来源模块           | 使用的类型/trait                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `tensor`           | `TensorView<'a, A, D>`, `.ndim()`, `.shape()`, `.len()`, `.as_ptr()`, `.is_f_contiguous()` |
| `element`          | `Numeric`, `ComplexScalar`                                                                 |
| `iter`             | `Elements`, `.iter()`                                                                      |
| `dispatch`（内部） | `select_exec_path()`、`ExecPath`、`ParallelGuard`                                          |
| `error`            | `XenonError::InvalidArgument`, `XenonError::ShapeMismatch`                             |
| `simd`（可选）     | 为满足条件的输入提供 dot 的 SIMD kernel（参见 `08-simd.md`）                               |
| `parallel`（可选） | 为 dot 提供并行执行能力                                                                    |

### 4.3 依赖合法性

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.4 依赖方向

依赖方向：单向向上。`matrix` 模块仅消费 `tensor`、`element`、`iter`、`error`、`simd`、`parallel` 模块。

---

## 5. 公共 API 设计

### 5.1 向量内积

```rust,ignore
/// Vector dot product: result = sum(a[i] * b[i])
///
/// For complex numbers, the conjugate-linear definition is used (§1.1).
/// For real types, `A::conjugate()` is a no-op identity (returns `self`),
/// so this naturally handles both real and complex dot products.
///
/// Supported types: i32, i64, f32, f64, Complex<f32>, Complex<f64>.
/// Not supported: bool, usize (they do not implement Numeric).
///
/// # Arguments
///
/// * `a` - tensor whose logical rank must be 1
/// * `b` - tensor whose logical rank must be 1
///
/// # Returns
///
/// `Result<A, XenonError>` - the dot product value or a shape mismatch error
///
/// # Errors
///
/// Returns a recoverable error when either input is not logically 1D.
/// Returns `XenonError::ShapeMismatch { operation, left_shape, right_shape }`
/// when lengths do not match.
/// Empty vectors are valid inputs and return the additive identity `A::zero()`.
/// When available, dot may delegate to the `simd` module for SIMD acceleration
/// or to the `parallel` module for parallel execution while preserving the same
/// observable semantics.
/// Integer overflow during accumulation is unrecoverable and must panic via
/// checked arithmetic, matching `13-reduction.md`.
///
/// # Examples
///
/// ```
/// let a = Tensor1::from_shape_vec(Ix1(3), vec![1.0, 2.0, 3.0])?;
/// let b = Tensor1::from_shape_vec(Ix1(3), vec![4.0, 5.0, 6.0])?;
/// let result = dot(&a.view(), &b.view())?;
/// assert_eq!(result, 32.0);  // 1*4 + 2*5 + 3*6
/// ```
pub fn dot<S1, S2, A, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Result<A, XenonError>
where
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    A: Numeric,
    D1: Dimension,
    D2: Dimension;
// The free function and TensorBase method accept the same generic parameters and can be
// called directly with any tensor form (owned/view/viewmut/arc).

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Stable method-style API; semantically equivalent to `matrix::dot()`.
    pub fn dot<S2, D2>(&self, other: &TensorBase<S2, D2>) -> Result<A, XenonError>
    where
        S2: Storage<Elem = A>,
        D2: Dimension;
}
```

整数内积使用 checked arithmetic 进行中间乘积和累加。泛型约束 `A: Numeric`（Numeric 已蕴含 Copy）在实现层直接复用 element 层 sealed traits `CheckedMul` 与 `CheckedAdd`（权威定义见 `03-element.md §5.9`）确保 `i32` / `i64` 路径使用 checked `mul` / `add`，并在 `None` 时按整数溢出策略 panic。

### 5.2 复数内积语义

```rust,ignore
// Complex dot product worked example (definition in §1.1):
// dot(Complex{re: 1, im: 2}, Complex{re: 3, im: 4})
// = conjugate(Complex{1,2}) * Complex{3,4}
// = Complex{1,-2} * Complex{3,4}
// = Complex{1*3-(-2)*4, 1*4+(-2)*3}
// = Complex{3+8, 4-6}
// = Complex{11, -2}
```

- 复数内积的 SIMD 加速参见 `08-simd.md` 覆盖矩阵。目标是对 `Complex<f32>` / `Complex<f64>` 内积提供 SIMD 路径。
- `TensorBase::dot<S2, D2>(&self, other: &TensorBase<S2, D2>) -> Result<A, XenonError>` 是稳定的 method-style API；它与自由函数 `dot(&TensorBase<S1, D1>, &TensorBase<S2, D2>)`（签名见 §5.1）共享相同的泛型参数与契约：允许两侧使用不同的维度类型 `D1 / D2`、不同的存储类型 `S1 / S2`（涵盖 `Owned` / `View` / `ViewMut` / `Arc` 全部模式），只在运行时检查双方是否都为逻辑 1D 且长度一致。两者必须共享相同的错误类别、复数共轭线性定义，以及以 `需求说明书 §28.3` 为权威基线的容差规则。

### 5.3 Good / Bad 对比示例

```rust,ignore
// Good - use dot() and handle errors
let a = Tensor1::<f64>::from_shape_vec(Ix1(3), vec![1.0, 2.0, 3.0])?;
let b = Tensor1::<f64>::from_shape_vec(Ix1(3), vec![4.0, 5.0, 6.0])?;
let result = dot(&a.view(), &b.view())?;
assert_eq!(result, 32.0);

// Good - complex dot product
let ca = Tensor1::<Complex<f64>>::from_shape_vec(Ix1(1), vec![Complex{re: 1.0, im: 2.0}])?;
let cb = Tensor1::<Complex<f64>>::from_shape_vec(Ix1(1), vec![Complex{re: 3.0, im: 4.0}])?;
let cresult = dot(&ca.view(), &cb.view())?;
// conjugate(1+2i) * (3+4i) = (1-2i)(3+4i) = 3+4i-6i-8i^2 = 3+4i-6i+8 = 11-2i

// Bad - unhandled error on dimension mismatch
let a = Tensor1::<f64>::from_shape_vec(Ix1(2), vec![1.0, 2.0])?;
let b = Tensor1::<f64>::from_shape_vec(Ix1(3), vec![1.0, 2.0, 3.0])?;
let _ = dot(&a.view(), &b.view()).unwrap();
```

---

## 6. 内部实现设计

### 6.1 执行路径选择

```
dot_impl(a, b):
    if a.ndim() != 1:
        return Err(XenonError::InvalidArgument { ... })

    if b.ndim() != 1:
        return Err(XenonError::InvalidArgument { ... })

    if a.len() != b.len():
        return Err(XenonError::ShapeMismatch { ... })

    let (path, guard) = dispatch::select_exec_path(
        a.len(),
        a.is_f_contiguous() && b.is_f_contiguous(),
        alignment_ok,
    );
    match path {
        ExecPath::Parallel => {
            let Some(guard) = guard else {
                unreachable!("dispatch returned (Parallel, None) — invariant violated");
            };
            parallel::par_dot(&a.view(), &b.view(), strategy, guard)
        },
        ExecPath::Simd    => simd::dot::<A>(a, b),
        ExecPath::Serial  => scalar::dot_impl(a, b),
    }
```

- `dot` 必须先完成逻辑 1D 与长度一致性检查。
- 调度模型：由 `dispatch.rs` 通过 `let (path, guard) = dispatch::select_exec_path(...)` 统一决定串行 / SIMD / 并行路径（参见 30-dispatch.md v1.1.0 决策 7）；返回的 `Option<ParallelGuard>` 仅在 `ExecPath::Parallel` 分支为 `Some(_)`，并由 `matrix::dot` 按值移交给 `parallel::par_dot`。
- **Worker 内 SIMD（v2.0 起）**：进入并行路径后，单个 worker 拿到 chunk 后**可以**在 chunk 内部独立做 SIMD admission（参见 08-simd.md v2.0.0 决策 5、09-parallel.md v2.0.0 决策 9）。这取代了 v1.x "并行 worker 不使用 SIMD" 的旧规则，提供 thread × SIMD 双层加速。串行路径下 SIMD 由 `simd` 后端按其 admission 规则独立判断是否启用；不进入 SIMD 时回退到该路径上的标量循环。
- SIMD 路径要求 `a` 和 `b` **均为** F-contiguous 且满足对齐前提；若任一输入不满足条件，dispatch 不会选择 `ExecPath::Simd`，最终落在 `Serial` 或 `Parallel` 的标量内核上（worker 内 SIMD admission 仍是 chunk 内独立判断）。
- `par_dot()` 的公开签名（见 09-parallel v2.0.0 §5.5）保持泛型 `DL: Dimension, DR: Dimension`，并接收 `_guard: ParallelGuard` 按值参数；在实现内部执行运行时 `ndim == 1` / 长度一致性校验。`matrix::dot()` 调用 `par_dot()` 时直接传 `&a.view()` / `&b.view()`，**不再通过 `as_ix1_view` 私有桥接收窄到 `Ix1`**（旧版的 `as_ix1_view` 设计与公开 `par_dot` 签名不闭合，已删除；旧文本见 §6.4 修订说明）。所有路径都必须保持一致的结果、错误模型与整数溢出 panic 语义。

### 6.2 并行阈值与禁止嵌套并行

`dot` 的并行路径必须直接复用 `dispatch.rs` 中的运行时裁决，而不是在 `matrix/` 内部复制一套独立阈值逻辑：

| 约束         | 要求                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------- |
| 阈值来源     | 是否进入并行路径由 `dispatch::select_exec_path(len, is_f_contiguous, alignment_ok)` 的返回值决定（30-dispatch v1.1.0 §5.5）。 |
| 非连续惩罚   | 非连续视图沿用 `dispatch.rs` 的有效阈值 saturating 翻倍策略（30-dispatch v1.1.0 决策 5），仅当收益明确时才进入并行。 |
| 禁止嵌套并行 | `select_exec_path` 内部检测当前线程是否已处于库内部并行区域；若已嵌套则不会返回 `(Parallel, _)`，从而把并行降级为串行/SIMD 路径，调用方无需再做嵌套防护（参见 30-dispatch v1.1.0 决策 7：select-and-enter 原子绑定）。 |
| 路径顺序     | 同 §6.1 执行路径选择。                                                                              |

这满足 `需求说明书 §9.2` / `需求说明书 §9.3` 对"支持阈值配置"和"库内部不得开启第二层并行"的要求。

### 6.3 标量实现

公开 API 允许 `D1 != D2`（例如 `Tensor<f64, IxDyn>` 与 `Tensor<f64, Ix1>`）。运行时 1D 校验通过后，两个视图都是逻辑 1D；标量 helper 用 `Elements` 元素迭代器消费，**不要求两个泛型维度类型相同**。这与 `as_ix1_view` 私有收窄方案的明显区别在于：标量实现一开始就只依赖 `Elements`（参见 10-iterator §5.1），不依赖具体维度类型 `D`。

```rust,ignore
fn scalar_dot_int<I, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> I
where
    I: Numeric + crate::element::CheckedMul + crate::element::CheckedAdd,
    S1: Storage<Elem = I>,
    S2: Storage<Elem = I>,
    D1: Dimension,
    D2: Dimension,
{
    // a.ndim() / b.ndim() == 1 and a.len() == b.len() are guaranteed by the
    // caller (see §6.1). Iter element-by-element regardless of D1/D2.
    a.iter()
        .zip(b.iter())
        .fold(I::zero(), |acc, (&x, &y)| {
            let product = x.checked_mul(y)
                .expect("dot overflow during multiplication");
            acc.checked_add(product)
                .expect("dot overflow during accumulation")
        })
}

fn scalar_dot_float_or_complex<A, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> A
where
    A: Numeric, // includes Complex<f32> / Complex<f64> via Numeric::conjugate
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    a.iter()
        .zip(b.iter())
        .fold(A::zero(), |acc, (&x, &y)| acc + x.conjugate() * y)
}
```

### 6.4 统一内积实现（实数与复数分派）

`dot()` 内部统一使用 `x.conjugate() * y` 的乘积生成规则（共轭线性定义见 §1.1，`Numeric::conjugate()` 详见 `03-element.md §5.2`），再按元素类型分派累加策略：整数路径需要同时对**乘法**和**累加**做 checked arithmetic，浮点/复数路径使用普通加法。整数路径不得因 identity conjugate 而绕过溢出检查。

**v2.0 修订说明（删除 `as_ix1_view` 私有桥接）**：v1.x 设计在并行路径前用 `as_ix1_view(view: &TensorView<'_, A, D>) -> Result<TensorView<'_, A, Ix1>, XenonError>` 把任意维度视图收窄到 `Ix1` 再调用 `parallel::par_dot()`。该桥接同时存在三类问题：

1. **类型链不闭合**：公开 API 接收 `&TensorBase<S, D>`（任意 storage），helper 接收 `&TensorView<'_, A, D>`，`Owned` / `Arc` 路径未说明如何获取生命周期合法的 view。
2. **指针 / offset 约定不清**：原代码把 `view.as_ptr()`（已应用 offset 的逻辑首元素指针）传给 `from_raw_parts(ptr, ..., offset=0)`，但 07-tensor §5.7 明确规定 `from_raw_parts` 的 `ptr` 必须是 storage base pointer、`offset` 是 base 到逻辑首元素的位移；旧设计实际丢失了 offset 信息。
3. **release 失去保护**：`debug_assert_eq!(view.ndim(), 1)` 在 release 下被消除，违反输入前提的调用会进入未定义行为。

修订后的方案：`matrix::dot()` 不做维度收窄，直接传 `&a.view()` / `&b.view()` 给 `parallel::par_dot()`；后者本身就接受泛型 `DL: Dimension, DR: Dimension`（参见 09-parallel v2.0.0 §5.5），并在内部做运行时 `ndim == 1` / 长度一致性校验。这彻底删除了 `as_ix1_view` helper 与 v1.x 的 `unsafe from_raw_parts` 调用路径。

```rust,ignore
// conjugate method in the Numeric trait (defined in 03-element.md §5.2)
// Real types: fn conjugate(self) -> Self { self }
// Complex types: fn conjugate(self) -> Self { Complex::conj(self) }

/// Unified dot dispatch for both real and complex types.
/// Uses `x.conjugate() * y` to generate products. Integer accumulation is routed
/// through checked integer arithmetic; floating-point and complex accumulation use ordinary `+`.
fn dot_impl<A, S1, S2, D1, D2>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
) -> Result<A, XenonError>
where
    A: Numeric,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension,
    D2: Dimension,
{
    // 1. Validate rank-1 precondition at runtime; either side non-1D returns
    //    InvalidArgument with closed-enum kind (see 26-error v3.0.0 §5.1).
    if a.ndim() != 1 {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("a"),
                constraint: Cow::Borrowed("rank == 1"),
            },
        });
    }
    if b.ndim() != 1 {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("dot"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("b"),
                constraint: Cow::Borrowed("rank == 1"),
            },
        });
    }
    // 2. Validate length match.
    if a.len() != b.len() {
        return Err(XenonError::ShapeMismatch {
            operation: Cow::Borrowed("dot"),
            left_shape: a.shape().to_vec(),
            right_shape: b.shape().to_vec(),
        });
    }
    // 3. Choose execution path; pass &a.view() / &b.view() directly to parallel
    //    backend without any Ix1 narrowing.
    let (path, guard) = dispatch::select_exec_path(
        a.len(),
        a.is_f_contiguous() && b.is_f_contiguous(),
        alignment_ok(a, b),
    );
    match path {
        ExecPath::Parallel => {
            let Some(guard) = guard else {
                unreachable!("dispatch returned (Parallel, None)");
            };
            parallel::par_dot::<_, _, A, _, _>(&a.view(), &b.view(), &strategy, guard)
        }
        ExecPath::Simd    => Ok(simd::dot::<A>(a, b)),
        ExecPath::Serial  => Ok(scalar::dot_dispatch::<A, _, _, _, _>(a, b)),
    }
}
```

- 统一使用 `Numeric::conjugate()` 实现 `x.conjugate() * y` 乘积生成规则（定义见 §1.1），避免为复数类型单独实现 `complex_dot` 函数。实数类型的 `conjugate()` 为零开销（内联后等价于直接使用 `x * y`），不引入额外运行时成本。
- 对整数 dot，乘法和累加都属于需求层面的不可恢复溢出路径；文档不得只对累加做 checked 处理而把乘法留给 release wrapping 语义。panic 信息至少包含 `operation=dot`、元素类型、触发阶段（`multiply` / `accumulate`）、逻辑位置（如 `lane` 或 `element_index`）以及适用 `shape`。
- 错误字段全部对齐 26-error v3.0.0 §5.1 的封闭枚举：`InvalidArgument { operation, kind: InvalidArgumentKind::OperationSpecific { argument, constraint } }`，`ShapeMismatch { operation, left_shape, right_shape }`。不再使用旧版自由文本 `expected` / `actual` / `argument: "input"` 等字段。

---

## 7. 实现任务拆分

### Wave 1: 基础

- [ ] **T1**: 创建 `src/matrix/` 模块骨架
  - 文件: `src/matrix/mod.rs`, `src/matrix/dot.rs`
  - 内容: 模块声明、dot 函数签名
  - 测试: 编译通过
  - 前置: tensor 模块完成
  - 预计: 5 min

### Wave 2: 前置校验与标量执行

- [ ] **T2**: 实现 dot 基础执行路径
  - 文件: `src/matrix/dot.rs`
  - 内容: rank/shape 校验、标量内积实现，以及接入 `dispatch.rs` 的串行 vs 并行分派骨架，实数和复数
  - 测试: `test_dot_basic`, `test_dot_complex`
  - 前置: T1
  - 预计: 10 min

### Wave 3: 标量路径收口

- [ ] **T3**: 接入并校验标量路径收口
  - 文件: `src/matrix/dot.rs`
  - 内容: 固化 rank/shape 校验后的标量路径收口，并把 `dispatch.rs` 的串行 vs 并行决策接到 dot
  - 测试: `test_dot_basic`, `test_dot_complex`
  - 前置: T2
  - 预计: 5 min

### Wave 4: SIMD / 并行路径校验

- [ ] **T4**: 接入并校验 SIMD 路径集成
  - 文件: `src/matrix/dot.rs`, `src/simd/mod.rs`
  - 内容: 在不进入并行路径时接入 SIMD kernel，并确保不满足条件时回退标量
  - 测试: `test_dot_simd_path_with_feature`
  - 前置: T3, simd 模块
  - 预计: 5 min
- [ ] **T5**: 接入并校验并行路径集成
  - 文件: `src/matrix/dot.rs`, `src/parallel/mod.rs`
  - 内容: 复用 `dispatch.rs` 决定并行路径，确保禁止嵌套并行，且 worker 内局部路径选择不触发第二层并行
  - 测试: `test_dot_parallel_path`
  - 前置: T3, parallel 模块
  - 预计: 5 min

### Wave 5: 测试

- [ ] **T6**: 编写测试
  - 文件: `tests/test_matrix.rs`
  - 内容: 正确性/维度不匹配/复数/feature-gate 回退测试
  - 测试: 所有矩阵测试
  - 前置: T2, T3, T4, T5
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                                        | 说明                                                         |
| -------- | ------------------------------------------- | ------------------------------------------------------------ |
| 单元测试 | `#[cfg(test)] mod tests`                    | 验证 `dot()` 的核心正确性与错误分支                          |
| 集成测试 | `tests/`                                    | 验证 `dot()` 与 `tensor`、`iter`、`simd`、`error` 的协同路径 |
| 边界测试 | 同模块测试中标注                            | 覆盖空向量、单元素、非连续输入等边界                         |
| 属性测试 | `tests/property/` 或 `tests/test_matrix.rs` | 验证空向量单位元、复数共轭线性与标量/非连续路径一致性不变量  |

### 8.2 单元测试清单

| 测试函数                                      | 测试内容                                                        | 优先级 |
| --------------------------------------------- | --------------------------------------------------------------- | ------ |
| `test_dot_basic`                              | 两个长度为 3 的向量内积正确                                     | 高     |
| `test_dot_complex`                            | 复数内积满足共轭线性                                            | 高     |
| `test_dot_shape_mismatch`                     | 长度不匹配返回 ShapeMismatch 错误                               | 高     |
| `test_dot_int_overflow_mul`                   | 整数乘法溢出触发 panic                                          | 高     |
| `test_dot_int_overflow_add`                   | 整数累加溢出触发 panic                                          | 高     |
| `test_dot_empty`                              | 两个空向量内积返回加法单位元                                    | 中     |
| `test_dot_single_element`                     | 单元素向量内积                                                  | 中     |
| `test_dot_simd_path_with_feature`             | 启用 `simd` 后 dot 可走 SIMD 路径且结果语义一致                 | 高     |
| `test_dot_parallel_path`                      | 启用并行路径后结果与标量语义一致                                | 高     |
| `test_dot_large_vector_parallel_threshold`    | 大向量达到阈值后可走并行路径，结果与标量一致                    | 高     |
| `test_dot_nested_parallel_falls_back`         | 已处于库内并行区域时不得开启第二层并行                          | 高     |
| `test_dot_simd_parallel_combined_consistency` | SIMD+并行组合路径与标量串行结果一致                             | 高     |
| `test_dot_parallel_threshold_boundary`        | 并行阈值边界两侧都保持正确路径选择与结果语义                    | 高     |
| `test_dot_high_rank_invalid_argument`         | 高 rank 输入（如 6D/动态高维）调用 `dot` 返回 `InvalidArgument` | 高     |
| `test_dot_nan_input`                          | 实数 `dot` 任一输入含 `NaN` 时结果为 `NaN`                      | 高     |
| `test_dot_float_tolerance_across_paths`       | 浮点路径在标量/SIMD/并行之间满足以 `需求说明书 §28.3` 为权威基线的文档化容差 | 高     |

### 8.3 边界测试场景

| 场景                                       | 预期行为                                                      |
| ------------------------------------------ | ------------------------------------------------------------- |
| 空向量 `shape=[0]`（标量/SIMD/并行配置下） | 均返回加法单位元，不引入额外错误语义                          |
| 单元素向量                                 | 返回 a[0] \* b[0]                                             |
| 高维输入 `shape=[1,1,1,1,1,1]` 调用 `dot`  | 返回 `InvalidArgument`，诊断字段完整                          |
| `10^7` 量级元素向量 `dot`                  | 阈值切换、文档化容差与 panic 契约在标量/SIMD/并行路径上一致   |
| 阈值边界输入                               | 覆盖低于/等于/高于并行阈值时的路径裁决与结果一致性            |
| 非连续向量（切片后）                       | dispatch 不会选择 `ExecPath::Simd`；最终走 Serial 标量或 Parallel 路径；结果与标量基线一致（worker 内 SIMD admission 在 chunk 内独立判断，是否可 SIMD 由 chunk 连续性与对齐条件决定，不满足则走标量） |
| `NaN` 输入                                 | 实数 `dot([NaN], [1.0])` 或 `dot([1.0], [NaN])` 返回 `NaN`    |
| `Inf` / `-Inf` 输入                        | 遵循 IEEE 754；例如实数 `dot([Inf], [2.0]) == Inf`            |

### 8.4 属性测试不变量

| 不变量                                                              | 测试方法                   |
| ------------------------------------------------------------------- | -------------------------- |
| `dot([], []) == A::zero()`                                          | 空向量对所有受支持类型成立 |
| `dot(a, b)` 与标量实现一致（整数严格一致，浮点/复数满足文档化容差） | 随机 1D 连续/非连续输入    |
| 复数 `dot(a, b)` 满足共轭线性定义（§1.1）                           | 随机复数向量               |

### 8.5 集成测试

| 测试文件               | 测试内容                                                                     |
| ---------------------- | ---------------------------------------------------------------------------- |
| `tests/test_matrix.rs` | `dot()` 与 `tensor`、`iter`、`element`、`simd`、`error` 路径的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置                     | 验证点                                                                |
| ------------------------ | --------------------------------------------------------------------- |
| 默认配置                 | `dot()` 通过标量路径满足实数/复数与错误语义契约。                     |
| 启用 `simd`              | dot 可选择 SIMD 路径；结果与默认语义一致。                            |
| 启用并行                 | dot 可选择并行归约路径；结果、错误类别与 panic 语义仍与标量路径一致。 |
| 同时启用 `simd,parallel` | 路径选择同 §6.1；整体结果须与标量串行基线一致。                       |

### 8.7 类型边界 / 编译期测试

| 场景                                                   | 测试方式                             |
| ------------------------------------------------------ | ------------------------------------ |
| `bool` / `usize` 不参与 `dot()`                        | 编译期测试。                         |
| `dot()` 仅接受逻辑 1D 输入                             | 运行时错误测试与编译期签名检查结合。 |
| matrix-matrix multiply 与 decomposition 不属于当前 API | API 缺失断言。                       |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块   | 接口/类型      | 约定                                                     |
| ------------------- | ---------- | -------------- | -------------------------------------------------------- |
| `matrix → tensor`   | `tensor`   | `TensorView<'_, A, D>` | 消费任意维度张量视图，但在运行时检查其逻辑 rank 是否为 1，参见 `07-tensor.md` §5 |
| `matrix → iter`     | `iter`     | `Elements` | 使用元素迭代器遍历输入，参见 `10-iterator.md` §5.1                       |
| `matrix → element`  | `element`  | `Numeric` / `ComplexScalar` | 通过泛型约束区分实数与复数路径，参见 `03-element.md` §5 |
| `matrix → simd`     | `simd`     | dot kernel | 满足条件时委托给 `simd` 模块做内积加速，且保持统一语义             |
| `matrix → parallel` | `parallel` | parallel reduction | 大输入时可委托给 `parallel` 模块做并行归约，且保持统一语义 |
| `matrix → error`    | `error`    | `XenonError::InvalidArgument`, `XenonError::ShapeMismatch` | 非 1D 输入或长度不匹配时返回可恢复错误，字段使用规范形式  |

### 9.2 数据流描述

```text
User calls dot(a, b)
    │
    ├── matrix validates rank-1 and equal length preconditions
    ├── complex inputs apply conjugate-linear product generation
    ├── let (path, guard) = dispatch::select_exec_path(...)
    │       ├── (Serial, None)        → scalar loop, may enter SIMD per backend admission
    │       ├── (Simd,   None)        → SIMD kernel by simd backend
    │       └── (Parallel, Some(g))   → parallel::par_dot(.., g, ..)
    │              └── inside each worker chunk: SIMD admission may apply per chunk
    │                  (08-simd v2.0.0 决策 5; 09-parallel v2.0.0 决策 9)
    └── the module returns a scalar result or a recoverable error
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                      |
| ----------------- | ------------------------------------------------------------------------- |
| Recoverable error | 左/右输入非 1D 时返回 `XenonError::InvalidArgument { operation: "dot", kind: InvalidArgumentKind::OperationSpecific { argument: "a"或"b", constraint: "rank == 1" } }`；长度不匹配时返回 `XenonError::ShapeMismatch { operation: "dot", left_shape, right_shape }`。字段对齐 26-error v3.0.0 §5.1 封闭枚举。 |
| Panic             | 整数 dot 的乘法溢出与累加溢出均为不可恢复错误，按 checked arithmetic 触发 panic。|
| 路径一致性        | 执行路径选择参见 §6.1；任何可选路径都不得改变结果、错误类别或 panic 语义。worker 内 SIMD（v2.0 起）的 chunk 内独立 admission 不影响整体结果与错误模型。|
| 容差边界          | 以 `需求说明书 §28.3` 为权威基线；实现细节参见 `00-coding.md §8.4`。同执行路径基础算术/比较默认精确一致；仅跨路径比较和数学函数比较允许使用文档化容差。|

---

## 11. 设计决策记录

### 决策 1：共轭线性定义选择

| 属性     | 值                                                                   |
| -------- | -------------------------------------------------------------------- |
| 决策     | 复数内积采用共轭线性定义（§1.1）                                     |
| 理由     | 这是数学和物理学中的标准定义；与 NumPy（np.vdot）、BLAS（zdotc）一致 |
| 替代方案 | 简单内积：sum(a[i] \* b[i])（不共轭）                                |
| 拒绝原因 | 不符合共轭线性空间的数学定义，与主流库行为不一致                     |

### 决策 2：错误恢复 vs panic

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | 长度不匹配返回 `Result::Err(XenonError::ShapeMismatch)`                |
| 理由     | 运行时形状检查失败属于可恢复错误；用户可能动态构造向量长度，应允许优雅处理 |
| 替代方案 | panic                                                                      |
| 拒绝原因 | 与 `需求说明书 §13` “维度或形状不匹配时须提供可恢复的错误处理路径” 不一致  |

### 决策 3：SIMD 优化策略

| 属性     | 值                                                                                           |
| -------- | -------------------------------------------------------------------------------------------- |
| 决策     | dot 接入 SIMD / 并行可选路径，执行路径选择参见 §6.1。                                        |
| 理由     | inner product 需要覆盖 SIMD / 并行能力，同时保持与标量路径一致的语义、错误模型和整数溢出契约 |
| 替代方案 | 始终只使用标量实现                                                                           |
| 拒绝原因 | 与需求说明书对 inner product 的 SIMD / 并行覆盖要求不一致                                    |

### 决策 4：删除 `as_ix1_view` 私有桥接，直接传 view 给 `par_dot`

| 属性     | 值                                                                                                                                     |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `matrix::dot()` 的并行路径直接传 `&a.view()` / `&b.view()` 给 `parallel::par_dot()`，不做维度收窄                                       |
| 理由     | 1) `par_dot` 公开签名本身已支持 `DL: Dimension, DR: Dimension`（09-parallel v2.0.0 §5.5）；2) 避免 v1.x `as_ix1_view` 的指针/offset 约定不清问题；3) 避免 release 下 `debug_assert` 失保护 |
| 替代方案 | 保留 `as_ix1_view` 私有 helper                                                                                                         |
| 拒绝原因 | helper 用 `view.as_ptr() + offset=0` 调用 `from_raw_parts`，与 07-tensor §5.7 中 `ptr` 必须是 storage base、`offset` 必须显式给出的契约相违；ArcRepr / ViewMutRepr 路径未说明；类型链不闭合 |

### 决策 5：worker 内允许 SIMD（与 09-parallel v2.0.0 决策 9 / 08-simd v2.0.0 决策 5 协同）

| 属性     | 值                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 决策     | 进入并行路径后，单个 worker chunk 内可独立做 SIMD admission；chunk 间合并仍由 `parallel` 控制                            |
| 理由     | 撤销 v1.x "并行 worker 不使用 SIMD" 的限制（§6.1/§9.2 中的旧表述），提供 thread × SIMD 双层加速                          |
| 替代方案 | 保留 v1.x 设计                                                                                                           |
| 拒绝原因 | 大数组 dot 的吞吐被人为限制；v2.0 起统一与 08-simd / 09-parallel 协同                                                    |

---

## 12. 性能考量

### 12.1 当前版本性能预期

| 操作                                 | 当前路径                | 说明                                                                |
| ------------------------------------ | ----------------------- | ------------------------------------------------------------------- |
| dot f32 (`len < threshold`)          | 串行路径（SIMD 或标量） | 小输入避免并行调度开销；串行路径可按局部条件选择 SIMD 或标量        |
| dot f32 (`len >= threshold`)         | 由 `dispatch.rs` 决定   | 路径选择参见 §6.1                                                   |
| dot f64 (`len >= threshold`)         | 由 `dispatch.rs` 决定   | 与 f32 相同，但仍受 ISA、对齐与并行阈值条件约束                     |
| dot complex f64 (`len >= threshold`) | 由 `dispatch.rs` 决定   | 复数内积必须保持共轭线性语义，容差以 `需求说明书 §28.3` 为权威基线  |

### 12.2 复杂度标注

- 标量 dot: O(n) 时间，O(1) 额外空间
- dot（任一路径）: O(n) 时间；并行路径的额外调度开销取决于执行器实现

---

## 13. 平台与工程约束

| 项目       | 约束                                                                             |
| ---------- | -------------------------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性                       |
| MSRV       | Rust 1.85+                                                                       |
| 单 crate   | 保持单 crate 结构，`matrix` 作为库内模块存在                                     |
| SemVer     | `dot()` 的输入维度前提、错误类别、复数共轭线性定义以及文档化容差结论属于稳定契约 |
| 最小依赖   | 不引入额外线性代数第三方依赖；BLAS 绑定仍属范围外                                |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-10 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-14 |
| 1.1.4 | 2026-04-15 |
| 1.1.5 | 2026-04-15 |
| 1.1.6 | 2026-04-15 |
| 1.2.0 | 2026-04-15 |
| 1.2.1 | 2026-04-15 |
| 1.2.2 | 2026-04-15 |
| 1.2.3 | 2026-04-16 |
| 1.2.4 | 2026-04-16 |
| 2.0.0 | 2026-05-02 |
| 2.0.1 | 2026-05-03 |

### v2.0.1 (2026-05-03) — Medium/Low review fixes

- §5.1：统一代码块 fence 格式。
- §6.1 / §6.2：将并行 guard 示例改为 let-else 内部断言，并删除旧版 `should_parallelize` / `ParallelGuard::enter()` 阈值表述。
- §8.3：明确非连续输入下 worker chunk SIMD admission 由 chunk 连续性与对齐条件决定。

### v2.0.0 (2026-05-02) — SemVer breaking changes

> 本版本是与 26-error v3.0.0、30-dispatch v1.1.0、08-simd v2.0.0、09-parallel v2.0.0、07-tensor v2.0.0 协同的破坏性更新。

- §1.1 末尾新增"关于模块命名"段落：解释 `matrix/` 名宽内容窄是有意保留以支持后续扩展。
- §5.2 自由函数签名描述对齐 §5.1：`dot(&TensorBase<S1, D1>, &TensorBase<S2, D2>)`，明确允许不同 storage 与不同维度。
- §6.1 / §6.2 / §9.2：调度模型对齐 30-dispatch v1.1.0 决策 7（`select_exec_path()` 返回 `(ExecPath, Option<ParallelGuard>)`）；`matrix::dot` 把 `Some(guard)` 按值移交 `parallel::par_dot`；嵌套并行防护由 dispatch 在裁决阶段处理（决策 7：select-and-enter 原子绑定）。
- §6.1：worker 内允许 SIMD（决策 5，对齐 08-simd v2.0.0 决策 5、09-parallel v2.0.0 决策 9）；删除"并行 worker 不使用 SIMD"的旧表述。
- §6.3：标量 helper 签名从 `(&TensorView<'_, I, D>, &TensorView<'_, I, D>)` 改为接受不同维度类型 `D1 / D2` 与不同 storage `S1 / S2`，消除 v1.x 与公开签名不闭合的问题。
- §6.4：**删除 `as_ix1_view` 私有桥接 helper**（决策 4）；`dot_impl` 直接调用 `parallel::par_dot(&a.view(), &b.view(), strategy, guard)`，不做维度收窄。
- §6.4：错误字段全部对齐 26-error v3.0.0 §5.1 封闭枚举：`InvalidArgument { kind: InvalidArgumentKind::OperationSpecific { argument, constraint } }`、`ShapeMismatch { operation, left_shape, right_shape }`；移除 v1.x 自由文本字段。
- §10：错误字段引用同步更新；路径一致性表述纳入 worker 内 SIMD。
- §11：新增决策 4 / 5。
- §8.3：非连续向量场景描述更新为 dispatch 不选 SIMD，最终落 Serial / Parallel 标量。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
