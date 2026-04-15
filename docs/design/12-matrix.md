# 矩阵运算模块设计

> 文档编号: 12 | 模块: `src/matrix/` | 阶段: Phase 4
> 前置文档: `03-element.md`, `07-tensor.md`, `08-simd.md`, `09-parallel.md`, `10-iterator.md`, `13-reduction.md`, `26-error.md`
> 需求参考: 需求说明书 §9.2, §9.3, §13, §27, §28.2, §28.3, §28.4, §28.5
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责      | 包含                                         | 不包含         |
| --------- | -------------------------------------------- | -------------- |
| 向量内积  | dot product（实数内积：sum(a[i] \* b[i])）   | 矩阵乘法、外积 |
| 复数内积  | 共轭线性定义（sum(conjugate(a[i]) \* b[i])） | 批量矩阵乘法   |
| SIMD 状态 | dot 可选接入 `simd` 模块 kernel 做 SIMD 加速，并可选接入 `parallel` 模块执行并行归约 | BLAS 绑定      |
| 错误处理  | 非 1D 输入返回 `XenonError::InvalidArgument`；形状不匹配返回 `XenonError::ShapeMismatch { operation, left_shape, right_shape }` | —              |

> **注意**：当前版本仅支持向量内积（dot）。不包含：矩阵乘法、外积、批量矩阵乘法、BLAS 绑定。

### 1.2 设计原则

| 原则      | 体现                                                  |
| --------- | ----------------------------------------------------- |
| 最小范围  | 当前仅实现向量内积，复杂线性代数由上游库通过 FFI 实现 |
| 错误恢复  | 维度不匹配返回可恢复错误（`XenonError`）；整数溢出为不可恢复 panic |
| 语义优先  | dot 先保证语义与错误契约一致，再按能力选择标量 / SIMD / 并行路径 |
| 与上游 BLAS 集成预期的语义兼容前提 | 内存布局与内积语义保持可对接上游 BLAS 集成的预期前提 |

### 1.3 在架构中的位置

```
Dependency levels:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; tensor owns storage and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: matrix  <- current module
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §9.2, §9.3, §13, §27, §28.2, §28.3, §28.4, §28.5 |
| 范围内   | 向量内积 `dot`、复数共轭线性语义、形状检查、空向量单位元，以及可选 SIMD / 并行执行路径。 |
| 范围外   | 矩阵-矩阵乘法、外积、批量矩阵乘法、矩阵分解以及 BLAS/LAPACK 绑定。 |
| 非目标   | 不把 `matrix` 扩展为通用线性代数层，不新增第三方线性代数依赖。 |

---

## 3. 文件位置

```
src/matrix/
├── mod.rs              # module entry, re-exports, dot() public API
└── dot.rs              # vector dot-product implementation (scalar / SIMD / parallel dispatch)
```

多文件设计理由：`matrix/` 保持最小语义层，只暴露 dot API 与执行路径分派；`src/simd/` 提供可选的 SIMD kernel，`parallel` 模块提供可选的并行执行能力。默认路径仍可回退到纯标量实现，以保持统一语义与错误契约。

---

## 4. 依赖关系与实现约束

### 4.1 Invariants

| 不变量 | 说明 |
| ------ | ---- |
| 逻辑 rank | `dot(a, b)` 的两个输入都必须是逻辑 1D；若 `ndim != 1`，必须走可恢复错误路径，而不是静默降级或 panic。 |
| 长度一致 | 两个输入在逻辑上一维时，`len(a) == len(b)` 才允许继续计算；否则返回 `ShapeMismatch`。 |
| 复数语义 | 复数内积固定使用 `sum(conjugate(a[i]) * b[i])`；实数类型的 `conjugate()` 为恒等操作。 |
| 执行路径 | `dot` 在语义检查后可按能力选择标量、`simd` 模块 kernel 或 `parallel` 模块并行归约；各路径必须保持一致的结果、错误类别与 panic 契约。 |
| 溢出契约 | 整数 dot 的乘法溢出与累加溢出均为不可恢复错误，必须通过 checked arithmetic panic。 |

### 4.2 Error Scenarios

| 场景 | 对外语义 |
| ---- | -------- |
| 输入不是逻辑 1D | 返回 `XenonError::InvalidArgument { operation: "dot".into(), argument: "input".into(), expected: "logical 1D tensor".into(), actual: format!("ndim={}", input.ndim()).into(), axis: None, shape: Some(input.shape().to_vec()) }`。 |
| 两个 1D 输入长度不一致 | 返回 `XenonError::ShapeMismatch { operation: "dot".into(), left_shape: a.shape().to_vec(), right_shape: b.shape().to_vec() }`。 |
| 整数乘法或累加溢出 | 触发 panic；这属于不可恢复算术域错误。 |
| 空向量输入 | 合法，返回加法单位元 `A::zero()`。 |

### 4.3 依赖图

```
src/matrix/
├── mod.rs
│   ├── crate::tensor        # TensorView<A, D>
│   ├── crate::element       # Numeric, ComplexScalar
│   ├── crate::iter          # Elements
│   └── crate::error         # XenonError
├── dot.rs
│   ├── crate::tensor        # TensorView<A, D>
│   ├── crate::element       # Numeric
│   ├── crate::iter          # Elements
│   ├── crate::error         # XenonError
│   ├── crate::simd (opt.)   # Optional SIMD dot kernel
│   └── crate::parallel (opt.) # Optional parallel reduction path
├── crate::iter              # Element iteration helpers
├── crate::simd (opt.)       # Optional SIMD dot kernel
└── crate::parallel (opt.)   # Optional parallel reduction path
```

### 4.4 类型级依赖

| 来源模块       | 使用的类型/trait                                                                           |
| -------------- | ------------------------------------------------------------------------------------------ |
| `tensor`       | `TensorView<'a, A, D>`, `.ndim()`, `.shape()`, `.len()`, `.as_ptr()`, `.is_f_contiguous()` |
| `element`      | `Numeric`, `ComplexScalar`                                                                 |
| `iter`         | `Elements`, `.iter()`                                                                      |
| `error`        | `XenonError::InvalidArgument`, `XenonError::ShapeMismatch`                                 |
| `simd`（可选） | 为满足条件的输入提供 dot 的 SIMD kernel（参见 `08-simd.md`）                               |
| `parallel`（可选） | 为大输入提供 dot 的并行归约执行路径，并复用阈值/guard 运行时状态                        |

### 4.5 依赖方向

> **依赖方向：单向向上。** `matrix` 模块仅消费 `tensor`、`element`、`iter`、`error`、`simd`、`parallel` 模块。

### 4.6 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 向量内积

````rust,ignore
/// Vector dot product: result = sum(a[i] * b[i])
///
/// For complex numbers, the conjugate-linear definition is used:
/// result = sum(conjugate(a[i]) * b[i])
///
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
pub fn dot<A, D1, D2>(
    a: &TensorView<'_, A, D1>,
    b: &TensorView<'_, A, D2>,
) -> Result<A, XenonError>
where
    A: Numeric + Copy,
    D1: Dimension,
    D2: Dimension;
// Note: Numeric (defined in 03-element.md) already implies
// Mul<Output=Self> + Add<Output=Self>, so the public constraint
// `Numeric + Copy` is sufficient. The internal implementation
// (dot_impl) repeats these bounds explicitly for clarity.
````

### 5.2 复数内积语义

```rust
// Complex dot product implements conjugate-linearity
// dot(Complex{re: 1, im: 2}, Complex{re: 3, im: 4})
// = conjugate(Complex{1,2}) * Complex{3,4}
// = Complex{1,-2} * Complex{3,4}
// = Complex{1*3-(-2)*4, 1*4+(-2)*3}
// = Complex{3+8, 4-6}
// = Complex{11, -2}
```

> **SIMD 覆盖说明：** 复数内积的 SIMD 加速参见 `08-simd.md` 覆盖矩阵。目标是对 `Complex<f32>` / `Complex<f64>` 内积提供 SIMD 路径。

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
        return Err(XenonError::InvalidArgument {
            operation: "dot".into(),
            argument: "input".into(),
            expected: "logical 1D tensor".into(),
            actual: format!("ndim={}", a.ndim()).into(),
            axis: None,
            shape: Some(a.shape().to_vec()),
        })

    if b.ndim() != 1:
        return Err(XenonError::InvalidArgument {
            operation: "dot".into(),
            argument: "input".into(),
            expected: "logical 1D tensor".into(),
            actual: format!("ndim={}", b.ndim()).into(),
            axis: None,
            shape: Some(b.shape().to_vec()),
        })

    if a.len() != b.len():
        return Err(XenonError::ShapeMismatch {
            operation: "dot".into(),
            left_shape: a.shape().to_vec(),
            right_shape: b.shape().to_vec(),
        })

    if simd path is available for (a, b):
        return simd::dot_impl(a, b)

    if parallel path is enabled for (a, b):
        return parallel::par_dot(a, b)

    return scalar::dot_impl(a, b)
```

> **执行路径约束：** `dot` 必须先完成逻辑 1D 与长度一致性检查；随后可按条件选择 `simd` 模块 kernel、`parallel` 模块的公开 `par_dot()` 并行归约入口或标量回退路径。所有路径都必须保持一致的结果、错误模型与整数溢出 panic 语义。

### 6.1.1 并行阈值与禁止嵌套并行

`dot` 的并行路径必须直接复用 `09-parallel.md` 中的运行时裁决，而不是在 `matrix/` 内部复制一套独立阈值逻辑：

| 约束 | 要求 |
| ---- | ---- |
| 阈值来源 | 是否进入并行路径由 `parallel::should_parallelize(len, is_f_contiguous)` 与全局阈值配置决定。 |
| 非连续惩罚 | 非连续视图沿用 `parallel` 模块的有效阈值翻倍策略；仅当收益明确时才进入并行。 |
| 禁止嵌套并行 | 若当前线程已处于库内部并行区域，则 `ParallelGuard::enter()` 失败并强制回退标量/串行路径，不得再开启第二层并行。 |
| 路径顺序 | 先做 rank/shape 校验，再做阈值与 guard 判定，最后在并行 chunk 内局部选择 SIMD 或标量。 |

这满足 `require.md §9.2` / `§9.3` 对“支持阈值配置”和“库内部不得开启第二层并行”的要求。

### 6.2 标量实现

```rust
fn scalar_dot_int<I, D>(
    a: &TensorView<I, D>,
    b: &TensorView<I, D>,
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
    a: &TensorView<A, D>,
    b: &TensorView<A, D>,
) -> A {
    a.iter()
        .zip(b.iter())
        .fold(A::zero(), |acc, (&x, &y)| acc + x.conjugate() * y)
}
```

### 6.3 统一内积实现（实数与复数分派）

`dot()` 内部统一使用 `x.conjugate() * y` 的乘积生成规则，再按元素类型分派累加策略：整数路径需要同时对**乘法**和**累加**做 checked arithmetic，浮点/复数路径使用普通加法。这通过 `Numeric` trait 中的 `fn conjugate(self) -> Self` 方法实现：

- 实数类型（`f32`、`f64`、`i32`、`i64`）：`conjugate(x) == x`（恒等实现，直接返回 `self`）
- 复数类型（`Complex<f32>`、`Complex<f64>`）：`conjugate(x)` 返回共轭复数

```rust
// conjugate method in the Numeric trait (defined in 03-element.md §5.2)
// Real types: fn conjugate(self) -> Self { self }
// Complex types: fn conjugate(self) -> Self { Complex::conjugate(self) }

/// Unified dot dispatch for both real and complex types.
/// Uses `x.conjugate() * y` to generate products. Integer accumulation is routed
/// through checked integer arithmetic; floating-point and complex accumulation use ordinary `+`.
fn dot_impl<A, D1, D2>(
    a: &TensorView<'_, A, D1>,
    b: &TensorView<'_, A, D2>,
) -> Result<A, XenonError> {
    // 1. validate rank-1 precondition at runtime
    // 2. choose simd / parallel::par_dot / scalar execution path
    // 3. dispatch to integer checked path or float/complex path inside the selected backend
    unimplemented!("dispatches to simd, parallel, or scalar dot backends")
}
```

> **设计决策：** 通过 `Numeric::conjugate()` 方法实现实数与复数的统一分派，避免为复数类型单独实现 `complex_dot` 函数。
> 实数类型的 `conjugate()` 为零开销（内联后等价于直接使用 `x * y`），不引入额外运行时成本。

> **整数溢出补充：** 对整数 dot，乘法和累加都属于需求层面的不可恢复溢出路径；文档不得只对累加做 checked 处理而把乘法留给 release wrapping 语义。panic 信息至少包含 `operation=dot`、元素类型、触发阶段（`multiply` / `accumulate`）、逻辑位置（如 `lane` 或 `element_index`）以及适用 `shape`。

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
  - 内容: rank/shape 校验、标量内积实现，以及 SIMD / parallel 分派骨架，实数和复数
  - 测试: `test_dot_basic`, `test_dot_complex`
  - 前置: T1
  - 预计: 10 min

### Wave 3: SIMD / 并行路径校验

- [ ] **T3**: 接入并校验可选 SIMD / 并行路径
  - 文件: `src/matrix/dot.rs`, `src/simd/mod.rs`, `src/parallel/mod.rs`
  - 内容: 满足条件时接入 SIMD kernel 或并行归约；并行路径复用全局阈值配置并禁止嵌套并行，不满足条件时回退标量
  - 测试: `test_dot_simd_path_with_feature`, `test_dot_parallel_path`
  - 前置: T2, simd / parallel 模块
  - 预计: 10 min

### Wave 4: 测试

- [ ] **T4**: 编写测试
  - 文件: `tests/test_matrix.rs`
  - 内容: 正确性/维度不匹配/复数/feature-gate 回退测试
  - 测试: 所有矩阵测试
  - 前置: T2, T3
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1]
            │
Wave 2: [T2]
            │
Wave 3: [T3]
            │
Wave 4: [T4]
```

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

| 测试函数                    | 测试内容                          | 优先级 |
| --------------------------- | --------------------------------- | ------ |
| `test_dot_basic`            | 两个长度为 3 的向量内积正确       | 高     |
| `test_dot_complex`          | 复数内积满足共轭线性              | 高     |
| `test_dot_shape_mismatch`   | 长度不匹配返回 ShapeMismatch 错误 | 高     |
| `test_dot_int_overflow_mul` | 整数乘法溢出触发 panic            | 高     |
| `test_dot_int_overflow_add` | 整数累加溢出触发 panic            | 高     |
| `test_dot_empty`            | 两个空向量内积返回加法单位元      | 中     |
| `test_dot_single_element`   | 单元素向量内积                    | 中     |
| `test_dot_simd_path_with_feature` | 启用 `simd` 后 dot 可走 SIMD 路径且结果语义一致 | 高     |
| `test_dot_parallel_path`          | 启用并行路径后结果与标量语义一致 | 高     |
| `test_dot_large_vector_parallel_threshold` | 大向量达到阈值后可走并行路径，结果与标量一致 | 高     |
| `test_dot_high_rank_invalid_argument` | 高 rank 输入（如 6D/动态高维）调用 `dot` 返回 `InvalidArgument` | 高     |
| `test_dot_float_tolerance_across_paths` | 浮点路径在标量/SIMD/并行之间满足文档化容差 | 高     |

### 8.3 边界测试场景

| 场景                 | 预期行为                 |
| -------------------- | ------------------------ |
| 空向量 `shape=[0]`   | 返回加法单位元（零）     |
| 单元素向量           | 返回 a[0] \* b[0]        |
| 大向量（`10^7` 量级元素） | 可按阈值选择并行 dot 路径，结果正确 |
| 高维输入 `shape=[1,1,1,1,1,1]` | 返回 `InvalidArgument`，诊断中包含 `operation`、`argument`、`expected`、`actual`、`shape` |
| 非连续向量（切片后） | 回退到标量路径，结果正确 |
| `Inf` / `-Inf` 输入  | 遵循 IEEE 754；例如实数 `dot([Inf], [2.0]) == Inf` |

### 8.4 属性测试不变量

| 不变量                                          | 测试方法                   |
| ----------------------------------------------- | -------------------------- |
| `dot([], []) == A::zero()`                      | 空向量对所有受支持类型成立 |
| `dot(a, b)` 与标量实现一致                      | 随机 1D 连续/非连续输入    |
| 复数 `dot(a, b) == sum(conjugate(a[i]) * b[i])` | 随机复数向量               |

### 8.5 集成测试

| 测试文件               | 测试内容                                                                     |
| ---------------------- | ---------------------------------------------------------------------------- |
| `tests/test_matrix.rs` | `dot()` 与 `tensor`、`iter`、`element`、`simd`、`error` 路径的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `dot()` 通过标量路径满足实数/复数与错误语义契约。 |
| 启用 `simd` | dot 可选择 SIMD 路径；结果与默认语义一致。 |
| 启用并行 | dot 可选择并行归约路径；必须复用全局阈值配置并禁止嵌套并行，结果、错误类别与 panic 语义仍与标量路径一致。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `bool` / `usize` 不参与 `dot()` | 编译期测试。 |
| `dot()` 仅接受逻辑 1D 输入 | 运行时错误测试与编译期签名检查结合。 |
| matrix-matrix multiply 与 decomposition 不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向               | 对方模块  | 接口/类型                   | 约定                                                                             |
| ------------------ | --------- | --------------------------- | -------------------------------------------------------------------------------- |
| `matrix → tensor`  | `tensor`  | `TensorView<A, D>`          | 消费任意维度张量视图，但在运行时检查其逻辑 rank 是否为 1，参见 `07-tensor.md` §5 |
| `matrix → iter`    | `iter`    | `Elements`                  | 使用元素迭代器遍历输入，参见 `10-iterator.md` §5.1                               |
| `matrix → element` | `element` | `Numeric` / `ComplexScalar` | 通过泛型约束区分实数与复数路径，参见 `03-element.md` §5                          |
| `matrix → simd`    | `simd`    | dot kernel                  | 满足条件时委托给 `simd` 模块做内积加速，且保持统一语义                           |
| `matrix → parallel`| `parallel`| parallel reduction          | 大输入时可委托给 `parallel` 模块做并行归约，且保持统一语义                       |
| `matrix → error`   | `error`   | `XenonError::InvalidArgument`, `XenonError::ShapeMismatch` | 非 1D 输入或形状不匹配时返回可恢复错误，字段使用规范形式 |

### 9.2 数据流描述

```text
User calls dot(a, b)
    │
    ├── matrix validates rank-1 and equal length preconditions
    ├── complex inputs apply conjugate-linear product generation
    ├── matrix selects simd / parallel / scalar execution backend
    └── the module returns a scalar result or a recoverable error
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 输入非 1D 时返回 `XenonError::InvalidArgument { operation: Cow<'static, str>, argument: Cow<'static, str>, expected: Cow<'static, str>, actual: Cow<'static, str>, axis: Option<usize>, shape: Option<Vec<usize>> }`，典型取值为 `axis: None`、`shape: Some(input.shape().to_vec())`；长度不匹配时返回 `XenonError::ShapeMismatch { operation: Cow<'static, str>, left_shape: Vec<usize>, right_shape: Vec<usize> }`。 |
| Panic | 整数 dot 的乘法溢出与累加溢出均为不可恢复错误，按 checked arithmetic 触发 panic。panic 文本至少包含 `operation=dot`、元素类型、`trigger`（`multiply` / `accumulate`）、逻辑位置（如 `lane`）以及输入 shape。 |
| 路径一致性 | `dot` 可选择标量、SIMD 或并行路径；任何可选路径都不得改变结果、错误类别或 panic 语义。 |
| 容差边界 | 整数类型结果须逐元素精确一致。对浮点和复数类型，不同执行路径（标量/SIMD/并行）的结果允许因合并顺序不同而存在差异，但必须遵守与 `08-simd.md` / `09-parallel.md` 一致的规则：相对标量参考值每个实数分量最多 `2 ULP`。 |

---

## 11. 设计决策记录（ADR）

### 决策 1：共轭线性定义选择

| 属性     | 值                                                                   |
| -------- | -------------------------------------------------------------------- |
| 决策     | 复数内积采用共轭线性定义：sum(conjugate(a[i]) \* b[i])               |
| 理由     | 这是数学和物理学中的标准定义；与 NumPy（np.vdot）、BLAS（zdotc）一致 |
| 替代方案 | 简单内积：sum(a[i] \* b[i])（不共轭）                                |
| 拒绝原因 | 不符合共轭线性空间的数学定义，与主流库行为不一致                     |

### 决策 2：错误恢复 vs panic

| 属性     | 值                                                                         |
| -------- | -------------------------------------------------------------------------- |
| 决策     | 维度不匹配返回 `Result::Err(XenonError::ShapeMismatch)`                    |
| 理由     | 运行时形状检查失败属于可恢复错误；用户可能动态构造向量长度，应允许优雅处理 |
| 替代方案 | panic                                                                      |
| 拒绝原因 | 与需求说明书 §13 "维度或形状不匹配时须提供可恢复的错误处理路径" 不一致     |

### 决策 3：SIMD 优化策略

| 属性     | 值                                                    |
| -------- | ----------------------------------------------------- |
| 决策     | `dot` 在满足条件时可接入 SIMD kernel 与并行归约，否则回退标量实现 |
| 理由     | inner product 需要覆盖 SIMD / 并行能力，同时保持与标量路径一致的语义、错误模型和整数溢出契约 |
| 替代方案 | 始终只使用标量实现                     |
| 拒绝原因 | 与需求说明书对 inner product 的 SIMD / 并行覆盖要求不一致 |

---

## 12. 性能考量

### 12.1 当前版本性能预期

以下性能表述是**非规范性目标**，用于说明路径选择意图；真正的公开契约仍是前文的语义、错误与容差边界。是否进入并行路径以 `parallel` 模块的全局阈值为准。

| 操作                 | 当前路径 | 说明 |
| -------------------- | -------- | ---- |
| dot f32 (`len < threshold`) | 标量或 SIMD | 小输入避免并行调度开销；连续输入可优先尝试 SIMD |
| dot f32 (`len >= threshold`) | SIMD / 并行优先，失败时回退标量 | 大输入优先尝试并行；chunk 内可局部选择 SIMD |
| dot f64 (`len >= threshold`) | SIMD / 并行优先，失败时回退标量 | 与 f32 相同，但受 ISA 与对齐条件约束 |
| dot complex f64 (`len >= threshold`) | 视 kernel 支持情况选择并行或标量 | 复数内积同样必须保持共轭线性语义与 2 ULP 分量容差 |

### 12.2 复杂度标注

- 标量 dot: O(n) 时间，O(1) 额外空间
- dot（任一路径）: O(n) 时间；并行路径的额外调度开销取决于执行器实现

---

## 13. 平台与工程约束

| 项目       | 约束                                                           |
| ---------- | -------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性     |
| crate 结构 | 保持单 crate 结构，`matrix` 作为库内模块存在                   |
| SemVer     | `dot()` 的输入维度前提、错误类别、复数共轭线性定义以及浮点/复数路径容差规则属于稳定契约；后续优化不得改变这些公开语义 |
| 依赖约束   | 不引入额外线性代数第三方依赖；BLAS 绑定仍属范围外              |
| API 边界   | `dot()` 仅对逻辑上一维输入开放，并通过运行时检查返回可恢复错误 |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
