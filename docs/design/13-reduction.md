# 归约运算模块设计

> 文档编号: 13 | 模块: `src/reduction/` | 阶段: Phase 4
> 前置文档: `02-dimension.md`, `03-element.md`, `07-tensor.md`, `10-iterator.md`
> 需求参考: 需求说明书 §14

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| sum 归约 | 全局归约 / 沿轴归约 / keepdims 选项 | mean/var/prod/min/max/argmin/argmax |
| 整数溢出处理 | checked_add，overflow 视为不可恢复错误 | wrapping/saturating 算术 |
| 浮点语义 | IEEE 754 NaN 传播 | 整数除法归约 |
| 空数组处理 | sum 返回加法单位元（零） | 篮选/排序操作 |
| SIMD 加速 | 连续内存的 SIMD 归约路径 | 并行归约（见 09-parallel.md） |
| 并行加速 | 并行归约结果须与单线程一致 | 篮选/排序操作 |

> **注意**：当前版本仅支持 sum 归约！不包含 mean/var/prod/min/max/argmin/argmax 等。

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 整数溢出安全 | 使用 checked_add，overflow 时 panic |
| NaN 传播 | 浮点 sum 中任一元素为 NaN 则返回 NaN |
| 空数组安全 | 空数组 sum 返回加法单位元（零） |
| SIMD 友好 | 连续数组自动走 SIMD 归约路径 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: reduction  ← 当前模块
```

---

## 2. 文件位置

```
src/reduction/
├── mod.rs              # 模块根，re-exports，公共 API 入口
├── sum.rs              # 全局 sum 和沿轴 sum_axis 实现
├── simd.rs             # SIMD 加速归约路径（cfg feature = "simd"）
└── parallel.rs         # 并行归约路径（cfg feature = "parallel"）
```

多文件设计理由：虽然当前仅实现 sum 归约，但归约模块涉及多条实现路径（标量、SIMD、并行），拆分为独立目录有助于：
- 各实现路径独立维护和条件编译
- 未来扩展其他归约操作时结构清晰
- 与项目中 `set/`、`shape/` 等模块保持一致的多文件组织风格

---

## 3. 依赖关系

### 3.1 依赖图

```
src/reduction/
├── mod.rs          # crate::tensor        # TensorBase<S, D>, TensorView
├── sum.rs          # crate::iter          # Elements, AxisIter
├── simd.rs         # crate::element       # Numeric, RealScalar
└── parallel.rs     # crate::error         # XenonError
                    # crate::simd (可选)   # pulp::Arch（SIMD 归约路径）
                    # crate::parallel (可选) # 并行归约路径
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.len()`, `.iter()` |
| `iter` | `Elements`, `AxisIter`, `ExactSizeIterator` |
| `element` | `Numeric`, `RealScalar` |
| `dimension` | `Dimension`, `RemoveAxis`, `D::Smaller` |
| `error` | `XenonError` |
| `simd`（可选） | `pulp::Arch`（参见 `08-simd.md` §3） |
| `parallel`（可选） | 并行归约路径（参见 `09-parallel.md` §3） |

### 3.3 依赖方向

> **依赖方向：单向向上。** `reduction` 消费 `iter`、`tensor`、`element` 模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 全局 sum 归约

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Global sum.
    ///
    /// # Empty array behavior
    ///
    /// Returns `A::zero()` (additive identity).
    ///
    /// # Integer overflow
    ///
    /// Panics on overflow (uses checked_add).
    ///
    /// # NaN behavior
    ///
    /// Returns NaN if any element is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.sum(), 6.0);
    ///
    /// // Empty array
    /// let empty: Tensor1<f64> = Tensor1::zeros([0]);
    /// assert_eq!(empty.sum(), 0.0);
    /// ```
    pub fn sum(&self) -> A;
}
```

> **分派策略说明：** `sum()` 内部通过在 `Numeric` trait 上提供 `safe_add(self, rhs: Self) -> Self` 方法实现类型分派：整数类型使用 `checked_add` 溢出时 panic，浮点/复数类型使用 IEEE 754 加法。这避免了在公共 API 中暴露 `CheckedAdd` 约束，同时保持了正确的溢出行为。

### 4.2 沿轴 sum 归约

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric,
{
    /// Sum along an axis, removing the reduced axis from the output shape.
    ///
    /// # Arguments
    ///
    /// - `axis`: reduction axis
    ///
    /// # Returns
    ///
    /// `Result<Tensor<A, D::Smaller>, XenonError>` — result has one fewer dimension.
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidAxis { axis: axis.index(), ndim: self.ndim() })`
    /// if `axis` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor2::from_shape_vec([2, 3], vec![1,2,3,4,5,6]);
    /// let row_sum = a.sum_axis(Axis(0))?;  // shape: [3], values: [5, 7, 9]
    /// let col_sum = a.sum_axis(Axis(1))?;  // shape: [2], values: [6, 15]
    /// ```
    pub fn sum_axis(&self, axis: Axis) -> Result<Tensor<A, D::Smaller>, XenonError>
    where
        // Note: CheckedAdd is not exposed in the public constraint.
        // Integer overflow detection is handled internally via
        // Numeric::safe_add (see §5.1 dispatch strategy).
        A: Numeric + Zero,
        D: RemoveAxis;

    /// Sum along an axis, keeping the reduced axis with length 1.
    ///
    /// # Arguments
    ///
    /// - `axis`: reduction axis
    ///
    /// # Returns
    ///
    /// `Result<Tensor<A, D>, XenonError>` — result has the same number of dimensions; reduced axis has length 1.
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidAxis { axis: axis.index(), ndim: self.ndim() })`
    /// if `axis` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor2::from_shape_vec([2, 3], vec![1,2,3,4,5,6]);
    /// let row_sum = a.sum_axis_keepdims(Axis(0))?;  // shape: [1, 3]
    /// let col_sum = a.sum_axis_keepdims(Axis(1))?;  // shape: [2, 1]
    /// ```
    pub fn sum_axis_keepdims(&self, axis: Axis) -> Result<Tensor<A, D>, XenonError>
    where
        // Note: CheckedAdd is not exposed in the public constraint.
        // Integer overflow detection is handled internally via
        // Numeric::safe_add (see §5.1 dispatch strategy).
        A: Numeric + Zero,
        D: RemoveAxis;
}
```

### 4.3 Good / Bad 对比示例

```rust
// Good - safely sum using sum()
let a = tensor!([1, 2, 3, 4]);
assert_eq!(a.sum(), 10);

// Good - empty array safe
let empty: Tensor1<i32> = Tensor1::zeros([0]);
assert_eq!(empty.sum(), 0);

// Good - axis reduction
let m = Tensor2::from_shape_vec([2, 3], vec![1,2,3,4,5,6]);
let row_sum = m.sum_axis(Axis(0));

// Bad - manual sum implementation (may miss overflow checks)
let mut total = 0i32;
for &x in tensor.iter() {
    total += x;  // Not recommended: silently wraps around in release mode (Rust integer overflow behavior)
}

// Bad - using unwrap() ignoring errors
let first = tensor.iter().next().unwrap();  // panics on empty array
```

---

## 5. 内部实现设计

### 5.1 整数溢出处理

整数 `sum` 使用 `CheckedAdd` trait（定义于 `03-element.md §4.9`）进行安全累加：

```rust
// Integer sum implementation — uses CheckedAdd trait (see 03-element.md §4.9)
fn sum_int<I: Numeric + CheckedAdd>(iter: impl Iterator<Item = &I>) -> I {
    iter.fold(I::zero(), |acc, &x| {
        acc.checked_add(x).expect("integer overflow in reduction (sum)")
    })
}
```

`sum` 的泛型约束中，整数类型（`i32`/`i64`）满足 `Numeric + CheckedAdd`，浮点和复数类型则走 §5.2 的 IEEE 754 路径（不使用 `CheckedAdd`）。

### 5.2 浮点 NaN 传播

```rust
// Float sum implementation
fn sum_float<F: RealScalar>(iter: impl Iterator<Item = &F>) -> F {
    iter.fold(F::zero(), |acc, &x| acc + *x)
    // NaN + anything = NaN, auto-propagation
}
```

### 5.3 空数组处理

```rust
// Empty array: fold initial value is zero(), empty iteration returns zero() directly
// sum of [] = 0 (additive identity)
```

### 5.4 sum_axis_keepdims 实现步骤

`sum_axis_keepdims` 与 `sum_axis` 的区别仅在于输出形状构造：

```
sum_axis_keepdims(tensor, axis):
    1. 计算 result_shape = tensor.shape().to_owned()
    2. result_shape[axis] = 1       // 将被归约的轴长度设为 1
    3. 分配 result = Tensor::zeros(result_shape)
    4. 对每个沿 axis 的切片 s in tensor.axis_iter(axis):
           result.slice_mut(对应外层坐标) += s.sum()
    5. 返回 result

输出类型：Tensor<A, D>（维度数与输入相同，被归约轴长度为 1）
```

对静态维度（如 `Ix2`），通过 `Dimension::set_axis(axis, 1)` 将对应轴设为 1，得到新的 shape 实例，再构造输出张量。这要求 `D: RemoveAxis`（用于 axis 边界检查），但输出仍为 `Tensor<A, D>` 而非降维类型。

---

## 6. 实现任务拆分

### Wave 1: 全局归约

- [ ] **T1**: 创建 `src/reduction/` 模块骨架
  - 文件: `src/reduction/mod.rs`, `src/reduction/sum.rs`
  - 内容: 模块声明、全局 `sum` 函数签名
  - 测试: 编译通过
  - 前置: tensor, element, iter 模块完成
  - 预计: 5 min

- [ ] **T2**: 实现整数类型全局 `sum`
  - 文件: `src/reduction/sum.rs`
  - 内容: checked_add 累加，overflow 时 panic
  - 测试: `test_sum_i32`, `test_sum_i64`, `test_sum_overflow_panic`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 实现浮点类型全局 `sum`
  - 文件: `src/reduction/sum.rs`
  - 内容: 浮点累加，NaN 传播
  - 测试: `test_sum_f32`, `test_sum_f64`, `test_sum_nan`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现复数类型全局 `sum`
  - 文件: `src/reduction/sum.rs`
  - 内容: 复数累加
  - 测试: `test_sum_complex`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 沿轴归约

- [ ] **T5**: 实现沿轴 `sum_axis`
  - 文件: `src/reduction/sum.rs`
  - 内容: 使用 AxisIter 遍历，保持 keepdims 选项
  - 测试: `test_sum_axis_2d`, `test_sum_axis_3d`, `test_sum_axis_keepdims`
  - 前置: T2, T3, T4
  - 预计: 15 min

### Wave 3: SIMD 加速

- [ ] **T6**: 实现 SIMD 归约路径
  - 文件: `src/reduction/simd.rs`（cfg feature = "simd"）
  - 内容: SIMD 水平求和，尾部标量处理
  - 测试: `test_sum_simd_consistency`
  - 前置: T2, T3, simd 模块
  - 预计: 15 min

### Wave 4: 并行加速

- [ ] **T7**: 实现并行归约路径
  - 文件: `src/reduction/parallel.rs`（cfg feature = "parallel"）
  - 内容: 分块并行归约，合并结果须与单线程一致
  - 测试: `test_sum_parallel_consistency`
  - 前置: T5, parallel 模块
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1]
           │
Wave 2: [T2] [T3] [T4]
           │     │     │
           └─────┴─────┘
                  │
Wave 3:         [T5]
                  │
Wave 4:         [T6]
                  │
Wave 5:         [T7]
```

---

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_sum_i32` | i32 向量求和正确 | 高 |
| `test_sum_i64` | i64 向量求和正确 | 高 |
| `test_sum_f32` | f32 向量求和正确 | 高 |
| `test_sum_f64` | f64 向量求和正确 | 高 |
| `test_sum_complex_f32` | Complex<f32> 求和正确 | 中 |
| `test_sum_complex_f64` | Complex<f64> 求和正确 | 中 |
| `test_sum_overflow_panic` | 整数溢出 panic | 高 |
| `test_sum_nan` | NaN 传播：含 NaN 的数组 sum 返回 NaN | 高 |
| `test_sum_empty` | 空数组 sum 返回零 | 高 |
| `test_sum_single_element` | 单元素 sum 正确 | 中 |
| `test_sum_axis_2d` | 2D 沿轴 0/1 sum 正确 | 高 |
| `test_sum_axis_3d` | 3D 沿各轴 sum 正确 | 中 |
| `test_sum_axis_keepdims` | `sum_axis_keepdims` 保留轴长度 1 | 高 |
| `test_sum_axis_empty` | 沿轴长度为 0 时结果正确 | 中 |
| `test_sum_simd_consistency` | SIMD 路径结果与标量一致 | 高 |
| `test_sum_parallel_consistency` | 并行路径结果与单线程一致 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | 全局 sum 返回 0，`sum_axis(Axis(0))` 返回 `[0, 0, 0]` |
| 单元素 `[1]` | sum 返回元素值本身 |
| 大数组（1M 元素） | sum 结果正确，无溢出 |
| i32 最大值附近 | checked_add 正确检测溢出并 panic |
| f64 含 NaN | sum 返回 NaN |
| f64 含 +Inf/-Inf | sum 返回 +Inf/-Inf |
| 非连续数组（切片后） | sum 结果与连续数组一致 |

---

## 8. 与其他模块的交互

| 交互模块 | 接口约定 |
|----------|----------|
| `iter` | 使用 `Elements` 迭代器遍历元素，`AxisIter` 遍历轴，参见 `10-iterator.md` §4 |
| `tensor` | 消费 `TensorBase<S, D>`，返回 `Tensor<A, D>`，参见 `07-tensor.md` §4 |
| `element` | 泛型约束 `Numeric`（全局 sum），`RealScalar`（浮点特化），参见 `03-element.md` §3 |
| `simd`（可选） | 连续数组自动走 SIMD 归约路径，参见 `08-simd.md` §3 |
| `parallel`（可选） | 大数组自动走并行归约路径，参见 `09-parallel.md` §4 |

---

## 9. 设计决策记录（ADR）

### 决策 1：overflow 处理策略

| 属性 | 值 |
|------|-----|
| 决策 | 整数 sum 使用 checked_add，overflow 时 panic |
| 理由 | 整数溢出通常是编程错误（选择了过小的类型），静默 wrap-around 可能导致严重 bug；与需求说明书 §14 "整数归约溢出视为不可恢复错误" 一致 |
| 替代方案 | 使用 wrapping_add（静默溢出） |
| 替代方案 | 使用 saturating_add（饱和溢出） |
| 拒绝原因 | 需求明确要求 overflow 为不可恢复错误；wrap/saturate 隐藏问题 |

### 决策 2：并行归约一致性保证

| 属性 | 值 |
|------|-----|
| 决策 | **整数归约**并行结果与串行完全一致（checked 加法保证）；**浮点归约**并行结果与串行允许 ≤ 2 ULP 差异 |
| 理由 | 浮点加法不满足结合律（IEEE 754），分块并行时累加顺序不同必然引入舍入差异，无法在不使用 Kahan 补偿的情况下保证浮点精确一致性。整数因不存在舍入，仍可保证精确一致。 |
| 实现约定 | 浮点并行 sum 的测试使用相对容差 (`rtol < 1e-14` for f64, `rtol < 1e-6` for f32) 而非精确相等比较 |
| 替代方案 | 要求所有类型精确一致 — 放弃，浮点在不使用 Kahan 的情况下无法实现 |
| 参见 | `09-parallel.md §9 ADR-2`（协调一致） |
| **一致性解释** | 对于浮点类型，"一致"解释为：逐元素运算逐位一致，归约运算允许 ≤2 ULP 差异（因浮点加法不满足结合律）。此解释与 NumPy 行为一致，并在文档中明确记录。 |

### 决策 3：Kahan 补偿求和

| 属性 | 值 |
|------|-----|
| 决策 | 当前版本使用普通求和，不实现 Kahan 补偿 |
| 理由 | SIMD 实现更简单；大多数场景精度足够；可未来扩展 |
| 替代方案 | 默认使用 Kahan 补偿求和 |
| 拒绝原因 | 增加实现复杂度，降低 SIMD 效率；精度问题仅在极端场景出现 |

---

## 10. 性能考量

### 10.1 Kahan 补偿求和 vs 普通求和

| 方法 | 精度 | SIMD 友好 | 实现复杂度 |
|------|------|-----------|-----------|
| 普通求和 | 足够（大多数场景） | 高（向量化累加） | 低 |
| Kahan 补偿 | 高（补偿精度损失） | 低（4x 操作/元素） | 高 |

### 10.2 SIMD 归约策略

| 步骤 | 操作 |
|------|------|
| 1 | 加载 SIMD 向量（8x f32 / 4x f64） |
| 2 | 乘法/累加（向量化） |
| 3 | 水平求和（reduce_add） |
| 4 | 尾部标量处理 |

### 10.3 性能数据（参考）

| 操作 | 标量路径 | SIMD 路径（AVX2） | 加速比 |
|------|----------|-------------------|--------|
| sum f32 (1M) | ~1.5ms | ~0.4ms | 3.7x |
| sum f64 (1M) | ~2ms | ~0.7ms | 2.9x |
| sum i32 (1M) | ~1ms | ~0.3ms | 3.3x |

### 10.4 复杂度标注

- 全局 sum: O(n) 时间，O(1) 额外空间
- 沿轴 sum: O(n) 时间，O(n/axis_len) 额外空间（结果数组）

---

## 11. no_std 兼容性

归约运算模块在 `no_std` 环境下可用。全局归约返回标量值无堆分配；沿轴归约需 `alloc` 分配结果张量。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| 全局 `sum()` | ✅ | 使用 `Iterator::fold`，返回标量值，无堆分配 |
| 沿轴 `sum_axis()` | ✅ | 需 `no_std + alloc`，分配结果 `Tensor` |
| 整数 `checked_add` | ✅ | `core` 内建，无额外依赖 |
| NaN 传播 | ✅ | IEEE 754 浮点语义，`core` 内建 |
| SIMD 归约路径 | ✅ | pulp crate 支持 `no_std`，参见 `08-simd.md` §11 |
| 并行归约路径 | ❌ | rayon 依赖 `std` 线程原语，参见 `09-parallel.md` §11 |

条件编译处理：

```rust
// Global sum: pure Iterator::fold — works in pure no_std
// sum_axis: allocates result Tensor → needs alloc::vec::Vec
// SIMD path: pulp supports no_std
// Parallel path: requires std (rayon dependency)

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Parallel reduction disabled under no_std automatically
// (rayon is gated behind "parallel" feature which requires "std")
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-07 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
