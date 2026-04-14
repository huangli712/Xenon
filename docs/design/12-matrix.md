# 矩阵运算模块设计

> 文档编号: 12 | 模块: `src/matrix/` | 阶段: Phase 4
> 前置文档: `03-element.md`, `07-tensor.md`, `10-iterator.md`, `26-error.md`
> 需求参考: 需求说明书 §13
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责      | 包含                                         | 不包含         |
| --------- | -------------------------------------------- | -------------- |
| 向量内积  | dot product（实数内积：sum(a[i] \* b[i])）   | 矩阵乘法、外积 |
| 复数内积  | 共轭线性定义（sum(conjugate(a[i]) \* b[i])） | 批量矩阵乘法   |
| SIMD 加速 | 连续内存的 SIMD 路径                         | BLAS 绑定      |
| 错误处理  | 形状不匹配返回 XenonError::ShapeMismatch     | —              |

> **注意**：当前版本仅支持向量内积（dot）。不包含：矩阵乘法、外积、批量矩阵乘法、BLAS 绑定。

### 1.2 设计原则

| 原则      | 体现                                                  |
| --------- | ----------------------------------------------------- |
| 最小范围  | 当前仅实现向量内积，复杂线性代数由上游库通过 FFI 实现 |
| 错误恢复  | 维度不匹配返回可恢复错误，不 panic                    |
| SIMD 友好 | 连续内存自动走 SIMD 路径                              |
| BLAS 兼容 | 内存布局支持 BLAS 调用约定                            |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: matrix  ← 当前模块
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §13 |
| 范围内   | 向量内积 `dot`、复数共轭线性语义、形状检查、空向量单位元与可选 SIMD 加速。 |
| 范围外   | 矩阵-矩阵乘法、外积、批量矩阵乘法、矩阵分解以及 BLAS/LAPACK 绑定。 |
| 非目标   | 不把 `matrix` 扩展为通用线性代数层，不新增第三方线性代数依赖。 |

---

## 3. 文件位置

```
src/matrix/
├── mod.rs              # 模块入口，re-exports，dot() 公共 API
└── dot.rs              # 向量内积实现，必要时委托 `src/simd/`
```

多文件设计理由：`matrix/` 保持最小语义层，只暴露 dot API 与标量逻辑；SIMD 加速由独立 backend 模块 `src/simd/` 提供，便于与 `math/`、`reduction/` 共享实现和测试策略。

---

## 4. 依赖关系

### 4.1 依赖图

```
src/matrix/
├── mod.rs
│   ├── crate::tensor        # TensorView<A, D>
│   ├── crate::element       # Numeric, ComplexScalar
│   └── crate::error         # XenonError
├── dot.rs
│   ├── crate::tensor        # TensorView<A, D>
│   └── crate::element       # Numeric
└── crate::simd (opt.)       # Shared SIMD backend for dot
```

### 4.2 类型级依赖

| 来源模块       | 使用的类型/trait                                                                           |
| -------------- | ------------------------------------------------------------------------------------------ |
| `tensor`       | `TensorView<'a, A, D>`, `.ndim()`, `.shape()`, `.len()`, `.as_ptr()`, `.is_f_contiguous()` |
| `element`      | `Numeric`, `ComplexScalar`                                                                 |
| `error`        | `XenonError::ShapeMismatch`                                                                |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd.md` §3）                                                       |

### 4.3 依赖方向

> **依赖方向：单向向上。** `matrix` 模块仅消费 `tensor`、`element`、`error`、`simd` 模块。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 向量内积

````rust
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
/// Returns `XenonError::ShapeMismatch` when lengths do not match.
/// Empty vectors are valid inputs and return the additive identity `A::zero()`.
/// Integer overflow during accumulation is unrecoverable and must panic via
/// checked arithmetic, matching `13-reduction.md`.
///
/// # Examples
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);
/// let result = dot(&a.view(), &b.view())?;
/// assert_eq!(result, 32.0);  // 1*4 + 2*5 + 3*6
/// ```
pub fn dot<A, D>(
    a: &TensorView<'_, A, D>,
    b: &TensorView<'_, A, D>,
) -> Result<A, XenonError>
where
    A: Numeric + Copy,
    D: Dimension;
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

### 5.3 Good / Bad 对比示例

```rust
// Good - use dot() and handle errors
let a = Tensor1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
let b = Tensor1::<f64>::from_vec(vec![4.0, 5.0, 6.0]);
let result = dot(&a.view(), &b.view())?;
assert_eq!(result, 32.0);

// Good - complex dot product
let ca = Tensor1::<Complex<f64>>::from_vec(vec![Complex{re: 1.0, im: 2.0}]);
let cb = Tensor1::<Complex<f64>>::from_vec(vec![Complex{re: 3.0, im: 4.0}]);
let cresult = dot(&ca.view(), &cb.view())?;
// conjugate(1+2i) * (3+4i) = (1-2i)(3+4i) = 3+4i-6i-8i^2 = 3+4i-6i+8 = 11-2i

// Bad - unhandled error on dimension mismatch
let a = Tensor1::<f64>::from_vec(vec![1.0, 2.0]);
let b = Tensor1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
// let _ = dot(&a.view(), &b.view())?;  // do not discard the recoverable error path
```

---

## 6. 内部实现设计

### 6.1 执行路径选择

```
dot_impl(a, b):
    if a.len() != b.len():
        return Err(ShapeMismatch)

    #[cfg(feature = "simd")]
    if a.is_f_contiguous() && b.is_f_contiguous():
        return simd::dot_impl(a, b)

    return scalar::dot_impl(a, b)
```

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
// Numeric trait 中的 conjugate 方法（定义于 03-element.md §4.3）
// Real types: fn conjugate(self) -> Self { self }
// Complex types: fn conjugate(self) -> Self { Complex::conjugate(self) }

/// Unified dot dispatch for both real and complex types.
/// Uses `x.conjugate() * y` to generate products. Integer accumulation is routed
/// through checked integer arithmetic; floating-point and complex accumulation use ordinary `+`.
fn dot_impl<A, D>(
    a: &TensorView<'_, A, D>,
    b: &TensorView<'_, A, D>,
) -> Result<A, XenonError> {
    // 1. validate rank-1 precondition at runtime
    // 2. dispatch to integer checked path or float/complex path
    // 3. optionally delegate to simd backend only when the selected kernel
    //    preserves Xenon's exact-result contract
    unimplemented!("dispatches to scalar_dot_int or scalar_dot_float_or_complex")
}
```

> **设计决策：** 通过 `Numeric::conjugate()` 方法实现实数与复数的统一分派，避免为复数类型单独实现 `complex_dot` 函数。
> 实数类型的 `conjugate()` 为零开销（内联后等价于直接使用 `x * y`），不引入额外运行时成本。

> **整数溢出补充：** 对整数 dot，乘法和累加都属于需求层面的不可恢复溢出路径；文档不得只对累加做 checked 处理而把乘法留给 release wrapping 语义。

---

## 7. 实现任务拆分

### Wave 1: 基础

- [ ] **T1**: 创建 `src/matrix/` 模块骨架
  - 文件: `src/matrix/mod.rs`, `src/matrix/dot.rs`
  - 内容: 模块声明、dot 函数签名
  - 测试: 编译通过
  - 前置: tensor 模块完成
  - 预计: 5 min

### Wave 2: 标量实现

- [ ] **T2**: 实现标量 dot
  - 文件: `src/matrix/dot.rs`
  - 内容: 标量内积实现，实数和复数
  - 测试: `test_dot_basic`, `test_dot_complex`
  - 前置: T1
  - 预计: 10 min

### Wave 3: SIMD 加速

- [ ] **T3**: 审查 SIMD dot 路径
  - 文件: `src/matrix/dot.rs`, `src/simd/vector.rs`
  - 内容: `matrix::dot` 接入独立 `simd/` backend 的内积实现
  - 测试: `test_dot_simd_consistency`
  - 前置: T2, simd 模块
  - 预计: 10 min

### Wave 4: 测试

- [ ] **T4**: 编写测试
  - 文件: `tests/test_matrix.rs`
  - 内容: 正确性/维度不匹配/复数/SIMD 一致性测试
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
| `test_dot_simd_consistency` | SIMD 路径结果与标量一致           | 高     |

### 8.3 边界测试场景

| 场景                 | 预期行为                 |
| -------------------- | ------------------------ |
| 空向量 `shape=[0]`   | 返回加法单位元（零）     |
| 单元素向量           | 返回 a[0] \* b[0]        |
| 大向量（1M 元素）    | SIMD 路径启用，结果正确  |
| 非连续向量（切片后） | 回退到标量路径，结果正确 |

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
| 启用 `simd` | 连续实数向量的 SIMD dot 与标量结果一致，非连续输入回退标量。 |

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
| `matrix → tensor`  | `tensor`  | `TensorView<A, D>`          | 消费任意维度张量视图，但在运行时检查其逻辑 rank 是否为 1，参见 `07-tensor.md` §4 |
| `matrix → iter`    | `iter`    | `Elements`                  | 使用元素迭代器遍历输入，参见 `10-iterator.md` §3                                 |
| `matrix → element` | `element` | `Numeric` / `ComplexScalar` | 通过泛型约束区分实数与复数路径，参见 `03-element.md` §3                          |
| `matrix → simd`    | `simd`    | SIMD backend                | 连续内存时可自动走 SIMD 路径，参见 `08-simd.md` §3                               |
| `matrix → error`   | `error`   | `XenonError::ShapeMismatch` | 形状不匹配时返回可恢复错误                                                       |

### 9.2 数据流描述

```text
User calls dot(a, b)
    │
    ├── matrix validates rank-1 and equal length preconditions
    ├── complex inputs apply conjugate-linear product generation
    ├── contiguous inputs may use SIMD; other inputs use scalar iteration
    └── the module returns a scalar result or a recoverable error
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | 输入非 1D 或长度不匹配时返回 `XenonError`（如 `ShapeMismatch` / rank 检查错误），携带期望与实际维度信息。 |
| Panic | 整数 dot 的乘法溢出与累加溢出均为不可恢复错误，按 checked arithmetic 触发 panic。 |
| 路径一致性 | 标量与 SIMD 路径必须返回同一 dot 结果；不满足连续前提时统一回退标量实现。 |
| 容差边界 | 当前不放宽任何数值容差；SIMD 仅在可证明与标量结果一致时启用。 |

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
| 决策     | 连续 + 同类型内存自动走 SIMD；非连续回退标量          |
| 理由     | SIMD 仅在连续内存上有意义；非连续时标量路径更简单正确 |
| 替代方案 | 全部使用标量                                          |
| 拒绝原因 | 性能差距显著（3-4x），科学计算用户期望高性能          |

---

## 12. 性能考量

### 12.1 SIMD 加速预期

| 操作                 | 标量路径 | SIMD 路径（AVX2） | 加速比 |
| -------------------- | -------- | ----------------- | ------ |
| dot f32 (1M)         | ~1.5ms   | ~0.4ms            | 3.7x   |
| dot f64 (1M)         | ~2ms     | ~0.7ms            | 2.9x   |
| dot complex f64 (1M) | ~6ms     | 标量回退          | 1.0x   |

### 12.2 复杂度标注

- 标量 dot: O(n) 时间，O(1) 额外空间
- SIMD dot: O(n) 时间，O(1) 额外空间（更低常数因子）

---

## 13. 平台与工程约束

| 项目       | 约束                                                           |
| ---------- | -------------------------------------------------------------- |
| 标准库环境 | Xenon 当前版本仅支持 `std`，本文档不再承诺 `no_std` 兼容性     |
| crate 结构 | 保持单 crate 结构，`matrix` 作为库内模块存在                   |
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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
