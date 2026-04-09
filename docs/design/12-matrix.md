# 矩阵运算模块设计

> 文档编号: 12 | 模块: `src/matrix/` | 阶段: Phase 4
> 前置文档: `03-element-types.md`, `07-tensor.md`, `10-iterator.md`, `26-error-handling.md`
> 需求参考: 需求说明书 §13

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 向量内积 | dot product（实数内积：sum(a[i] * b[i])） | 矩阵乘法、外积 |
| 复数内积 | 共轭线性定义（sum(conj(a[i]) * b[i])） | 批量矩阵乘法 |
| SIMD 加速 | 连续内存的 SIMD 路径 | BLAS 绑定 |
| 错误处理 | 形状不匹配返回 XenonError::ShapeMismatch | — |

> **注意**：当前版本仅支持向量内积（dot）。不包含：矩阵乘法、外积、批量矩阵乘法、BLAS 绑定。

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 最小范围 | 当前仅实现向量内积，复杂线性代数由上游库通过 FFI 实现 |
| 错误恢复 | 维度不匹配返回可恢复错误，不 panic |
| SIMD 友好 | 连续内存自动走 SIMD 路径 |
| BLAS 兼容 | 内存布局支持 BLAS 调用约定 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: matrix  ← 当前模块
```

---

## 2. 文件位置

```
src/matrix/
├── mod.rs              # 模块入口，re-exports，dot() 公共 API
├── dot.rs              # 标量向量内积实现
└── simd.rs             # SIMD 加速内积路径（cfg feature = "simd"）
```

多文件设计理由：将标量路径与 SIMD 路径分离至独立文件，保持各实现路径的职责清晰，便于后续扩展（如并行路径）。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/matrix/
├── mod.rs
│   ├── crate::tensor        # TensorView<A, Ix1>
│   ├── crate::element       # Numeric, ComplexScalar（参见 03-element-types.md）
│   └── crate::error         # XenonError（参见 26-error-handling.md）
├── dot.rs
│   ├── crate::tensor        # TensorView<A, Ix1>
│   └── crate::element       # Numeric
└── simd.rs（可选）
    ├── crate::tensor        # TensorView<A, Ix1>
    ├── crate::element       # Numeric
    └── pulp::Arch
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorView<'a, A, Ix1>`, `.shape()`, `.len()`, `.as_ptr()`, `.is_contiguous()` |
| `element` | `Numeric`, `ComplexScalar` |
| `error` | `XenonError::ShapeMismatch` |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd.md` §3） |

### 3.3 依赖方向

> **依赖方向：单向向上。** `matrix` 模块仅消费 `tensor`、`element`、`error`、`simd` 模块。

---

## 4. 公共 API 设计

### 4.1 向量内积

```rust
/// Vector dot product: result = sum(a[i] * b[i])
///
/// For complex numbers, the conjugate-linear definition is used:
/// result = sum(conj(a[i]) * b[i])
///
/// For real types, `A::conj()` is a no-op identity (returns `self`),
/// so this naturally handles both real and complex dot products.
///
/// Supported types: i32, i64, f32, f64, Complex<f32>, Complex<f64>.
/// Not supported: bool, usize (they do not implement Numeric).
///
/// # Arguments
///
/// * `a` - vector of shape (N,)
/// * `b` - vector of shape (N,)
///
/// # Returns
///
/// `Result<A, XenonError>` - the dot product value or a shape mismatch error
///
/// # Errors
///
/// Returns `XenonError::ShapeMismatch` when dimensions do not match
///
/// # Examples
///
/// ```
/// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let b = Tensor1::from_vec(vec![4.0, 5.0, 6.0]);
/// let result = dot(&a.view(), &b.view())?;
/// assert_eq!(result, 32.0);  // 1*4 + 2*5 + 3*6
/// ```
pub fn dot<A>(
    a: &TensorView<'_, A, Ix1>,
    b: &TensorView<'_, A, Ix1>,
) -> Result<A, XenonError>
where
    A: Numeric + Copy;
// Note: Numeric (defined in 03-element-types.md) already implies
// Mul<Output=Self> + Add<Output=Self>, so the public constraint
// `Numeric + Copy` is sufficient. The internal implementation
// (dot_impl) repeats these bounds explicitly for clarity.
```

### 4.2 复数内积语义

```rust
// Complex dot product implements conjugate-linearity
// dot(Complex{re: 1, im: 2}, Complex{re: 3, im: 4})
// = conj(Complex{1,2}) * Complex{3,4}
// = Complex{1,-2} * Complex{3,4}
// = Complex{1*3-(-2)*4, 1*4+(-2)*3}
// = Complex{3+8, 4-6}
// = Complex{11, -2}
```

### 4.3 Good / Bad 对比示例

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
// conj(1+2i) * (3+4i) = (1-2i)(3+4i) = 3+4i-6i-8i^2 = 3+4i-6i+8 = 11-2i

// Bad - unhandled error on dimension mismatch
let a = Tensor1::<f64>::from_vec(vec![1.0, 2.0]);
let b = Tensor1::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
// dot(&a.view(), &b.view()).unwrap();  // panic! should use ? instead
```

---

## 5. 内部实现设计

### 5.1 执行路径选择

```
dot_impl(a, b):
    if a.len() != b.len():
        return Err(ShapeMismatch)
    
    #[cfg(feature = "simd")]
    if a.is_contiguous() && b.is_contiguous():
        return simd::dot_impl(a, b)
    
    return scalar::dot_impl(a, b)
```

### 5.2 标量实现

```rust
fn scalar_dot<A: Numeric + Copy>(a: &TensorView<A, Ix1>, b: &TensorView<A, Ix1>) -> A {
    a.iter().zip(b.iter()).fold(A::zero(), |acc, (&x, &y)| acc + x.conj() * y)
}
```

### 5.3 统一内积实现（实数与复数分派）

`dot()` 内部统一使用 `acc + x.conj() * y` 模式。这通过 `Numeric` trait 中的 `fn conj(self) -> Self` 方法实现：
- 实数类型（`f32`、`f64`、`i32`、`i64`）：`conj(x) == x`（恒等实现，直接返回 `self`）
- 复数类型（`Complex<f32>`、`Complex<f64>`）：`conj(x)` 返回共轭复数

```rust
// Numeric trait 中的 conj 方法（定义于 03-element-types.md §4.x）
// Real types: fn conj(self) -> Self { self }
// Complex types: fn conj(self) -> Self { Complex::conj(self) }

/// Unified dot implementation for both real and complex types.
/// Uses `x.conj() * y` — for real types conj() is identity, for complex
/// types it returns the conjugate.
fn dot_impl<A: Numeric + Copy + Mul<Output=A> + Add<Output=A>>(
    a: &TensorView<'_, A, Ix1>,
    b: &TensorView<'_, A, Ix1>,
) -> A {
    a.iter().zip(b.iter())
     .map(|(&x, &y)| x.conj() * y)  // conj() is now on Numeric
     .fold(A::zero(), |acc, v| acc + v)
}
```

> **设计决策：** 通过 `Numeric::conj()` 方法实现实数与复数的统一分派，避免为复数类型单独实现 `complex_dot` 函数。
> 实数类型的 `conj()` 为零开销（内联后等价于直接使用 `x * y`），不引入额外运行时成本。

---

## 6. 实现任务拆分

### Wave 1: 基础

- [ ] **T1**: 创建 `src/matrix/` 模块骨架
  - 文件: `src/matrix/mod.rs`, `src/matrix/dot.rs`, `src/matrix/simd.rs`
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
  - 文件: `src/matrix/simd.rs`
  - 内容: SIMD 加速的内积实现
  - 测试: `test_dot_simd_consistency`
  - 前置: T2, simd 模块
  - 预计: 10 min

### Wave 4: 测试

- [ ] **T4**: 编写测试
  - 文件: `tests/matrix.rs`
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

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_dot_basic` | 两个长度为 3 的向量内积正确 | 高 |
| `test_dot_complex` | 复数内积满足共轭线性 | 高 |
| `test_dot_shape_mismatch` | 长度不匹配返回 ShapeMismatch 错误 | 高 |
| `test_dot_empty` | 两个空向量内积返回 0（加法单位元） | 中 |
| `test_dot_single_element` | 单元素向量内积 | 中 |
| `test_dot_simd_consistency` | SIMD 路径结果与标量一致 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空向量 `shape=[0]` | 返回 Ok(0)（加法单位元） |
| 单元素向量 | 返回 a[0] * b[0] |
| 大向量（1M 元素） | SIMD 路径启用，结果正确 |
| 非连续向量（切片后） | 回退到标量路径，结果正确 |

---

## 8. 与其他模块的交互

| 交互模块 | 接口约定 |
|----------|----------|
| `tensor` | 消费 `TensorView<A, Ix1>`，参见 `07-tensor.md` §4 |
| `iter` | 使用 `Elements` 迭代器遍历元素，参见 `10-iterator.md` §3 |
| `element` | 泛型约束 `Numeric` / `ComplexScalar`，参见 `03-element-types.md` §3 |
| `simd`（可选） | 连续内存时自动走 SIMD 路径，参见 `08-simd.md` §3 |
| `error` | 形状不匹配返回 `XenonError::ShapeMismatch` |

---

## 9. 设计决策记录（ADR）

### 决策 1：共轭线性定义选择

| 属性 | 值 |
|------|-----|
| 决策 | 复数内积采用共轭线性定义：sum(conj(a[i]) * b[i]) |
| 理由 | 这是数学和物理学中的标准定义；与 NumPy（np.vdot）、BLAS（zdotc）一致 |
| 替代方案 | 简单内积：sum(a[i] * b[i])（不共轭） |
| 拒绝原因 | 不符合共轭线性空间的数学定义，与主流库行为不一致 |

### 决策 2：错误恢复 vs panic

| 属性 | 值 |
|------|-----|
| 决策 | 维度不匹配返回 `Result::Err(XenonError::ShapeMismatch)` |
| 理由 | 运行时形状检查失败属于可恢复错误；用户可能动态构造向量长度，应允许优雅处理 |
| 替代方案 | panic |
| 拒绝原因 | 与需求说明书 §13 "维度或形状不匹配时须提供可恢复的错误处理路径" 不一致 |

### 决策 3：SIMD 优化策略

| 属性 | 值 |
|------|-----|
| 决策 | 连续 + 同类型内存自动走 SIMD；非连续回退标量 |
| 理由 | SIMD 仅在连续内存上有意义；非连续时标量路径更简单正确 |
| 替代方案 | 全部使用标量 |
| 拒绝原因 | 性能差距显著（3-4x），科学计算用户期望高性能 |

---

## 10. 性能考量

### 10.1 SIMD 加速预期

| 操作 | 标量路径 | SIMD 路径（AVX2） | 加速比 |
|------|----------|-------------------|--------|
| dot f32 (1M) | ~1.5ms | ~0.4ms | 3.7x |
| dot f64 (1M) | ~2ms | ~0.7ms | 2.9x |
| dot complex f64 (1M) | ~6ms | ~2.5ms | 2.4x |

### 10.2 复杂度标注

- 标量 dot: O(n) 时间，O(1) 额外空间
- SIMD dot: O(n) 时间，O(1) 额外空间（更低常数因子）

---

## 11. no_std 兼容性

矩阵运算模块在 `no_std` 环境下可用。标量路径仅依赖 `core` trait，SIMD 路径依赖 pulp crate（支持 `no_std`）。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `dot()`（标量路径） | ✅ | 使用 `Iterator::fold`，无堆分配 |
| `dot()`（复数路径） | ✅ | 使用 `conj()` + `fold`，无堆分配 |
| `dot()`（SIMD 路径） | ✅ | pulp crate 支持 `no_std`，参见 `08-simd.md` §11 |
| `XenonError::ShapeMismatch` | ✅ | 使用 `core::fmt::Display`，无堆依赖 |

条件编译处理：

```rust
// Scalar dot: pure Iterator::fold — works in pure no_std
// SIMD dot: pulp crate supports no_std via core::arch intrinsics

#[cfg(not(feature = "std"))]
extern crate alloc;

// No Vec/Box/Arc needed — dot() returns a scalar value
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
| 1.0.5 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
