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
| 错误处理  | 非 1D 输入返回 `XenonError::InvalidArgument`；长度不匹配返回 `XenonError::DimensionMismatch` |

| 职责      | 不包含         |
|---------- | -------------- |
| 向量内积  | 矩阵乘法、外积 |
| 复数内积  | 批量矩阵乘法   |
| SIMD 状态 | BLAS 绑定      |
| 错误处理  |  —             |

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
│   ├── crate::dispatch      # select_exec_path(), should_parallelize()
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
| `dispatch`（内部） | `select_exec_path()`、`ExecPath`、`should_parallelize()`                                   |
| `error`            | `XenonError::InvalidArgument`, `XenonError::DimensionMismatch`                             |
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

````rust,ignore
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
/// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
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
pub fn dot<A, D1, D2>(
    a: &TensorView<'_, A, D1>,
    b: &TensorView<'_, A, D2>,
) -> Result<A, XenonError>
where
    A: Numeric,
    D1: Dimension,
    D2: Dimension;
// Note: Numeric (defined in 03-element.md) already implies
// Mul<Output=Self> + Add<Output=Self>, so the public constraint
// `Numeric + Copy` is sufficient. The internal implementation
// (dot_impl) repeats these bounds explicitly for clarity.

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + Copy,
{
    /// Stable method-style API; semantically equivalent to `matrix::dot()`.
    pub fn dot<S2, D2>(&self, other: &TensorBase<S2, D2>) -> Result<A, XenonError>
    where
        S2: Storage<Elem = A>,
        D2: Dimension;
}
````

整数内积使用 checked arithmetic 进行中间乘积和累加。泛型约束 `A: Numeric + Copy` 在实现层通过 sealed trait `CheckedArith`（定义见 `11-math.md`）确保 `i32` / `i64` 路径使用 checked `mul` / `add`。

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
- `TensorBase::dot<S2, D2>(&self, other: &TensorBase<S2, D2>) -> Result<A, XenonError>` 是稳定的 method-style API；它与自由函数 `dot(&TensorView<'_, A, D1>, &TensorView<'_, A, D2>)` 一样，允许两侧使用不同的维度类型，只在运行时检查双方是否都为逻辑 1D。两者必须共享相同的错误类别、复数共轭线性定义，以及以 `需求说明书 §28.3` 为权威基线的容差规则。

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
        return Err(XenonError::DimensionMismatch { ... })

    match dispatch::select_exec_path(a.len(), a.is_f_contiguous() && b.is_f_contiguous(), alignment_ok):
        ExecPath::Parallel => parallel::par_dot(as_ix1_view(a)?, as_ix1_view(b)?)
        ExecPath::Simd    => simd::SimdKernel::dot(a, b)
        ExecPath::Serial  => scalar::dot_impl(a, b)
```

- `dot` 必须先完成逻辑 1D 与长度一致性检查。
- 调度模型：由 `dispatch.rs` 统一决定串行 vs 并行路径。
- 若进入并行路径，每个 worker 在不触发第二层并行前提下，可局部选择 SIMD 或标量路径。
- SIMD 路径要求 `a` 和 `b` **均为** F-contiguous 且满足对齐前提；若任一输入不满足条件，必须回退到标量或并行中的标量 chunk 路径。
- `par_dot()` 自身的 API 契约仍与 `09-parallel.md` 一致，保持对泛型 `D: Dimension` 输入开放，并在实现内部执行运行时 1D 校验。这里“只接受 `Ix1`”描述的是 `matrix::dot()` 进入并行后端前的私有桥接约束，而不是 `par_dot()` 的公开函数签名。桥接实现详见 §6.4。所有路径都必须保持一致的结果、错误模型与整数溢出 panic 语义。

### 6.2 并行阈值与禁止嵌套并行

`dot` 的并行路径必须直接复用 `09-parallel.md` 中的运行时裁决，而不是在 `matrix/` 内部复制一套独立阈值逻辑：

| 约束         | 要求                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------- |
| 阈值来源     | 是否进入并行路径由 `dispatch::should_parallelize(len, is_f_contiguous)` 与全局阈值配置决定。        |
| 非连续惩罚   | 非连续视图沿用 `dispatch.rs` 的有效阈值翻倍策略；仅当收益明确时才进入并行。                         |
| 禁止嵌套并行 | 若当前线程已处于库内部并行区域，则 `dispatch::ParallelGuard::enter()` 失败并强制回退标量/串行路径，不得再开启第二层并行。 |
| 路径顺序     | 同 §6.1 执行路径选择。                                                                              |

这满足 `需求说明书 §9.2` / `需求说明书 §9.3` 对“支持阈值配置”和“库内部不得开启第二层并行”的要求。

### 6.3 标量实现

```rust,ignore
fn scalar_dot_int<I, D>(
    a: &TensorView<'_, I, D>,
    b: &TensorView<'_, I, D>,
) -> I {
    a.iter()
        .zip(b.iter())
        .fold(I::zero(), |acc, (&x, &y)| {
            let product = x.checked_mul(y)
                .expect("dot overflow during multiplication");
            acc.checked_add(product).expect("dot overflow during accumulation")
        })
}

fn scalar_dot_float_or_complex<A, D>(
    a: &TensorView<'_, A, D>,
    b: &TensorView<'_, A, D>,
) -> A {
    a.iter()
        .zip(b.iter())
        .fold(A::zero(), |acc, (&x, &y)| acc + x.conjugate() * y)
}
```

### 6.4 统一内积实现（实数与复数分派）

`dot()` 内部统一使用 `x.conjugate() * y` 的乘积生成规则（共轭线性定义见 §1.1，`Numeric::conjugate()` 详见 `03-element.md §5.2`），再按元素类型分派累加策略：整数路径需要同时对**乘法**和**累加**做 checked arithmetic，浮点/复数路径使用普通加法。整数路径不得因 identity conjugate 而绕过溢出检查。

```rust,ignore
// conjugate method in the Numeric trait (defined in 03-element.md §5.2)
// Real types: fn conjugate(self) -> Self { self }
// Complex types: fn conjugate(self) -> Self { Complex::conj(self) }

fn as_ix1_view<'a, A, D>(view: &TensorView<'a, A, D>) -> Result<TensorView<'a, A, Ix1>, XenonError>
where
    D: Dimension,
{
    debug_assert_eq!(view.ndim(), 1);
    // Reconstruct a 1D view from the same raw parts, narrowing D -> Ix1.
    // Uses the from_raw_parts constructor defined in 07-tensor.md §5.6.
    unsafe {
        TensorView::from_raw_parts(
            view.as_ptr(),
            view.len(),
            Ix1(view.shape()[0]),
            Strides::<Ix1>::from_slice(&[view.strides()[0]])?,
            0,
        )
    }
    .map_err(|_| XenonError::InvalidArgument {
        operation: "dot".into(),
        argument: "input".into(),
        expected: "logical 1D tensor".into(),
        actual: format!("ndim={}", view.ndim()).into(),
        axis: None,
        axis_len: None,
        start: None,
        end: None,
        shape: Some(view.shape().to_vec()),
    })
}

/// Unified dot dispatch for both real and complex types.
/// Uses `x.conjugate() * y` to generate products. Integer accumulation is routed
/// through checked integer arithmetic; floating-point and complex accumulation use ordinary `+`.
fn dot_impl<A, D1, D2>(
    a: &TensorView<'_, A, D1>,
    b: &TensorView<'_, A, D2>,
) -> Result<A, XenonError> {
    // 1. validate rank-1 precondition at runtime
    // 2. choose simd / private Ix1 bridge + parallel::par_dot / scalar execution path
    // 3. dispatch to integer checked path or float/complex path inside the selected backend
    unimplemented!("dispatches to simd, parallel, or scalar dot backends")
}
```

- `as_ix1_view()` 只在已验证 `ndim == 1` 后做维度收窄，不重排元素，也不强制把视图转为连续布局。若输入本身是合法的非连续 1D 视图，则返回的 `TensorView<'_, A, Ix1>` 保留原始 stride；后续是否可进入 SIMD 路径，仍由连续性与对齐检查单独决定。
- 推荐桥接形式是通过上方 `as_ix1_view()` 私有 helper，把 `TensorView<'_, A, D>` 收窄为 `TensorView<'_, A, Ix1>` 后再调用 `parallel::par_dot()`。该 helper 内部用 `TensorView::from_raw_parts`（定义见 07-tensor.md §5.6）从同一份原始指针/shape/stride 重建 1D 视图，不重新分配也不复制元素；若未来为性能保留 `unsafe` 快路径，也只能放在这个私有 helper 内，并以先前的 `ndim == 1` 运行时断言为前提，而不能暴露成公开 API 契约。若 rank 校验失败，`dot()` 必须在桥接前直接返回 `XenonError::InvalidArgument`。
- 统一使用 `Numeric::conjugate()` 实现 `x.conjugate() * y` 乘积生成规则（定义见 §1.1），避免为复数类型单独实现 `complex_dot` 函数。实数类型的 `conjugate()` 为零开销（内联后等价于直接使用 `x * y`），不引入额外运行时成本。
- 对整数 dot，乘法和累加都属于需求层面的不可恢复溢出路径；文档不得只对累加做 checked 处理而把乘法留给 release wrapping 语义。panic 信息至少包含 `operation=dot`、元素类型、触发阶段（`multiply` / `accumulate`）、逻辑位置（如 `lane` 或 `element_index`）以及适用 `shape`。

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
| `test_dot_dimension_mismatch`                 | 长度不匹配返回 DimensionMismatch 错误                           | 高     |
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
| 非连续向量（切片后）                       | 回退到标量路径，结果正确                                      |
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
| `matrix → error`    | `error`    | `XenonError::InvalidArgument`, `XenonError::DimensionMismatch` | 非 1D 输入或长度不匹配时返回可恢复错误，字段使用规范形式  |

### 9.2 数据流描述

```text
User calls dot(a, b)
    │
    ├── matrix validates rank-1 and equal length preconditions
    ├── complex inputs apply conjugate-linear product generation
    ├── dispatch.rs chooses serial vs parallel; parallel workers choose simd or scalar locally
    └── the module returns a scalar result or a recoverable error
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                      |
| ----------------- | ------------------------------------------------------------------------- |
| Recoverable error | 左/右输入非 1D 时分别返回 `XenonError::InvalidArgument`；长度不匹配时返回 `XenonError::DimensionMismatch`。 |
| Panic             | 整数 dot 的乘法溢出与累加溢出均为不可恢复错误，按 checked arithmetic 触发 panic。|
| 路径一致性        | 执行路径选择参见 §6.1；任何可选路径都不得改变结果、错误类别或 panic 语义。|
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
| 决策     | 长度不匹配返回 `Result::Err(XenonError::DimensionMismatch)`                |
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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
