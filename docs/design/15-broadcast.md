# 广播模块设计

> 文档编号: 15 | 模块: `src/broadcast.rs` | 阶段: Phase 3
> 前置文档: `02-dimension.md`, `07-tensor.md`, `06-memory.md`, `26-error.md`
> 需求参考: 需求说明书 §16

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 广播规则实现 | NumPy 广播规则（从右向左对齐、维度兼容检查） | 广播运算本身的调度（由 `math` 负责） |
| 形状推导 | `broadcast_shape()` 计算两个形状广播后的结果形状 | 数据复制（广播不拷贝数据，通过零步长实现） |
| 步长推导 | `broadcast_strides()` 计算广播后步长（含零步长） | 可变迭代（广播视图禁止可变迭代） |
| 兼容性检查 | `can_broadcast()` 检查两个形状是否兼容 | 逐元素运算（由 `math` 负责） |
| 广播视图创建 | `broadcast_to()` 返回零拷贝广播视图 | 多操作数调度（由调用方自行逐对广播） |
| 显式广播方法 | `broadcast_with()` 同时广播两个张量 | 算术运算符重载（由 `overload` 负责） |
| 错误报告 | 维度不兼容返回 `XenonError::BroadcastError` | |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零拷贝广播 | 通过零步长（stride=0）实现逻辑扩展，不复制数据 |
| 只读约束 | 广播视图只允许只读访问，禁止可变迭代 |
| NumPy 兼容 | 严格遵循 NumPy 广播规则，降低迁移成本 |
| O(1) 创建 | 广播视图创建仅需调整元数据，O(ndim) 时间 |
| 零步长安全 | 零步长访问永远不会越界，安全性由不变式保证 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: broadcast  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── broadcast.rs       # 广播规则实现（单文件）
```

单文件设计：广播逻辑高度内聚，拆分反而增加耦合复杂度。包含形状兼容检查、形状推导、步长计算、错误类型定义。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │   error      │
                    │ XenonError   │
                    └──────┬───────┘
                           │ 使用
                    ┌──────▼───────┐
                    │  broadcast   │
                    │ broadcast.rs │
                    └──┬───────┬───┘
                   使用 │       │ 使用
              ┌────────▼──┐ ┌──▼──────────┐
              │ dimension │ │   tensor    │
              │ Dimension │ │ TensorBase  │
              │ Ix0~IxDyn │ │ TensorView  │
              └───────────┘ └─────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `error` | `XenonError::BroadcastError`，参见 `26-error.md` §4 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `BroadcastDim<E>` trait, `.slice()`, `.size()`，参见 `02-dimension.md` §3, §4.9 |
| `tensor` | `TensorBase<S, D>`, `TensorView`, `.shape()`, `.strides()`，参见 `07-tensor.md` §4 |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `broadcast` 仅消费 `dimension` 和 `tensor` 的 trait 和类型，不被它们依赖。`math/`、`iter/` 等上层模块消费 `broadcast`。

---

## 4. 公共 API 设计

### 4.1 广播形状计算

```rust
/// Compute the broadcasted result shape of two shapes.
///
/// Follows NumPy broadcasting rules: align dimensions from right to left,
/// dimensions of size 1 can be expanded, returns an error if incompatible.
///
/// # Arguments
/// * `shape_a` - The first shape
/// * `shape_b` - The second shape
///
/// # Returns
/// * `Ok(IxDyn)` - The broadcasted shape as a dynamic dimension
/// * `Err(XenonError::BroadcastError)` - Shapes are incompatible
///
/// # no_std Compatibility
/// `broadcast_shape()` returns `IxDyn` (internally containing `Vec<usize>`),
/// which requires heap allocation. The project uses `extern crate alloc`
/// which is always available (no separate feature). See §11 for details.
///
/// # Examples
/// ```
/// let result = broadcast_shape(&[3, 1, 4], &[4, 1])?;
/// assert_eq!(result.slice(), &[3, 4, 4]);
/// ```
pub fn broadcast_shape(
    shape_a: &[usize],
    shape_b: &[usize],
) -> Result<IxDyn, XenonError>;
```

### 4.2 兼容性检查

```rust
/// Check whether two shapes are broadcast-compatible.
///
/// # Examples
/// ```
/// assert!(can_broadcast(&[3, 1, 4], &[4, 1]));   // OK
/// assert!(!can_broadcast(&[3, 2], &[3, 4]));     // incompatible
/// ```
pub fn can_broadcast(shape_a: &[usize], shape_b: &[usize]) -> bool;
```

### 4.3 广播步长计算

```rust
/// Compute broadcast strides. Zero strides are inserted for broadcasted dimensions.
/// Returns strides as isize (negative strides from source are preserved).
///
/// Broadcast dimensions have stride 0; non-broadcast dimensions preserve the
/// original stride (including sign).
///
/// # Arguments
/// * `orig_shape` - The original shape
/// * `orig_strides` - The original strides
/// * `target_shape` - The target broadcast shape
///
/// # Returns
/// `Result<Vec<isize>, XenonError>` — strides for the broadcasted view, with 0 for broadcast dims.
///
/// # Note
/// This function returns `Vec<isize>` which requires heap allocation.
/// The project uses `extern crate alloc` which is always available (no separate feature).
pub fn broadcast_strides(
    orig_shape: &[usize],
    orig_strides: &[isize],
    target_shape: &[usize],
) -> Result<Vec<isize>, XenonError>;
```

### 4.4 显式广播方法

```rust
impl<'a, A, D> TensorView<'a, A, D>
where
    D: Dimension,
{
    /// Broadcast the view to the target shape.
    ///
    /// Returns a new read-only view with logical expansion via zero strides.
    /// Does not copy data; O(ndim) operation.
    ///
    /// # Arguments
    /// * `shape` - The target shape
    ///
    /// # Returns
    /// * `Ok(TensorView)` - The broadcasted view
    /// * `Err(XenonError::BroadcastError)` - Shapes are incompatible
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::from_shape_vec([1, 4], vec![1, 2, 3, 4])?;
    /// let broadcasted = a.view().broadcast_to([3, 4])?;
    /// assert_eq!(broadcasted.shape(), &[3, 4]);
    /// assert_eq!(broadcasted.strides(), &[0, 1]);
    /// ```
    pub fn broadcast_to<E>(&self, shape: E) -> Result<TensorView<'a, A, E>, XenonError>
    where
        E: Dimension,
    {
        // ...
    }
}

/// Simultaneously broadcast two tensor views to a common shape.
///
/// Convenience method that internally calls `broadcast_shape` + two `broadcast_to`.
///
/// The return type uses the `BroadcastDim<E>` trait (defined in `02-dimension.md §4.9`)
/// to determine the broadcasted dimension type. For example:
/// - `Ix2::broadcast_dim(Ix1)` returns `Ix2`
/// - `IxDyn::broadcast_dim(any)` returns `IxDyn`
pub fn broadcast_with<'a, A, D, E>(
    a: &TensorView<'a, A, D>,
    b: &TensorView<'a, A, E>,
) -> Result<
    (TensorView<'a, A, <D as BroadcastDim<E>>::Output>,
     TensorView<'a, A, <E as BroadcastDim<D>>::Output>),
    XenonError,
>
where
    D: Dimension,
    E: Dimension,
{
    // ...
}
```

### 4.5 Good / Bad 对比

```rust
// Good - Use ? and Result for broadcast error handling
fn process(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Result<Tensor<f64, Ix2>, XenonError> {
    let (a_bc, b_bc) = broadcast_with(&a.view(), &b.view())?;
    zip_with(&a_bc, &b_bc, |x, y| x + y)
}

// Bad - Directly unwrap broadcast result
fn process_bad(a: &Tensor<f64, Ix2>, b: &Tensor<f64, Ix1>) -> Tensor<f64, Ix2> {
    let (a_bc, b_bc) = broadcast_with(&a.view(), &b.view()).unwrap(); // Forbidden: may panic
    zip_with(&a_bc, &b_bc, |x, y| x + y).unwrap()
}
```

---

## 5. 内部实现设计

### 5.1 NumPy 广播规则算法

广播遵循三个核心规则：

1. **维度对齐**: 从最右维度开始对齐，维度数不足的数组在左侧补 1
2. **兼容条件**: 对应维度相等，或其中一个为 1（或不存在）
3. **结果形状**: 每个维度按 NumPy 规则推导；若一侧为 0 且另一侧为 1，则结果为 0；若两侧相等则保持该值；否则由非 1 一侧决定结果维度

### 5.2 形状兼容性检查算法（伪代码）

```
function can_broadcast(shape_a: [usize; N], shape_b: [usize; M]) -> bool:
    // 从右向左对齐比较
    i = N - 1  // shape_a 的最右索引
    j = M - 1  // shape_b 的最右索引

    // pseudo-code over signed indices; Rust implementation should use checked decrements
    while i >= 0 or j >= 0:
        dim_a = if i >= 0 then shape_a[i] else 1
        dim_b = if j >= 0 then shape_b[j] else 1

        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return false  // 不兼容

        i = i - 1
        j = j - 1

    return true  // 兼容
```

### 5.3 广播形状推导算法（伪代码）

```
function broadcast_shape(shape_a: [usize; N], shape_b: [usize; M])
-> Result<[usize; max(N,M)], XenonError>:
    result_ndim = max(N, M)
    result_shape = array of size result_ndim

    i = N - 1
    j = M - 1
    k = result_ndim - 1

    while k >= 0:
        dim_a = if i >= 0 then shape_a[i] else 1
        dim_b = if j >= 0 then shape_b[j] else 1

        if dim_a != dim_b and dim_a != 1 and dim_b != 1:
            return Error(XenonError::BroadcastError {
                shape_a,
                shape_b,
            })

        if dim_a == 0 and dim_b == 1:
            result_shape[k] = 0
        else if dim_b == 0 and dim_a == 1:
            result_shape[k] = 0
        else:
            result_shape[k] = max(dim_a, dim_b)
        i = i - 1; j = j - 1; k = k - 1

    return Ok(result_shape)
```

### 5.4 零步长语义

零步长是广播的核心实现技术。当某个维度从 1 扩展到 N 时，该维度的步长设为 0：

- 访问索引 `[i, j, k]` 和 `[i, j, k+1]` 在该维度上访问同一内存位置
- 逻辑上"重复"该元素 N 次，物理上只有 1 个副本

```
原始: shape=[1, 4], strides=[4, 1], data=[a, b, c, d]
广播到 [3, 4]: strides=[0, 1]
  [0,j] → data[j]
  [1,j] → data[j]  (重复! 步长=0 导致偏移不变)
  [2,j] → data[j]  (重复!)
```

### 5.5 零步长安全论证

**安全性论证**: 对于 `shape[i] = 1` 的维度，任何合法索引 `idx` 满足 `0 <= idx < target_shape[i]`，访问的偏移量为 `stride[i] * idx = 0 * idx = 0`。因此：

1. 零步长访问永远不会越界
2. 同一广播维度的多个索引访问同一内存位置，但广播视图是只读的（`TensorView`），无数据竞争
3. 多次读取同一位置是安全的

### 5.6 HAS_ZERO_STRIDE 标志位

`LayoutFlags` 中的 `HAS_ZERO_STRIDE` 标志位标记该张量视图存在零步长维度：

```rust
// Layout flags update for broadcast
fn update_flags_for_broadcast(source_flags: LayoutFlags, new_strides: &[isize]) -> LayoutFlags {
    let mut flags = source_flags;
    flags.set_has_zero_stride(new_strides.iter().any(|&s| s == 0));

    if flags.has_zero_stride() {
        // Broadcast view is not F-contiguous (zero-stride dimensions break contiguity)
        flags.set_f_contiguous(false);
    }
    flags
}
```

### 5.7 广播结果只读约束

广播视图必须只读。原因：

1. **别名冲突**: 广播视图中多个索引指向同一物理位置，写入不确定性
2. **语义模糊**: `broadcasted[[0, 0]] = x` 和 `broadcasted[[1, 0]] = y` 写入同一位置，后者覆盖前者
3. **内存安全**: 并发写入同一位置可能造成数据竞争

**实现方式**: `broadcast_to()` 仅对 `TensorView`（只读）可用，不提供 `TensorViewMut` 的广播方法。

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/broadcast.rs` 骨架与错误类型
  - 文件: `src/broadcast.rs`
  - 内容: 模块声明，以及对 `XenonError::BroadcastError` 的使用与文档接线
  - 测试: 编译通过
  - 前置: error 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `can_broadcast()` 兼容性检查
  - 文件: `src/broadcast.rs`
  - 内容: 从右向左对齐比较算法
  - 测试: `test_can_broadcast_compatible`, `test_can_broadcast_incompatible`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 形状与步长推导

- [ ] **T3**: 实现 `broadcast_shape()` 形状推导
  - 文件: `src/broadcast.rs`
  - 内容: 广播形状推导算法，不兼容返回 `BroadcastError`
  - 测试: `test_broadcast_shape_basic`, `test_broadcast_shape_error`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 实现 `broadcast_strides()` 步长计算
  - 文件: `src/broadcast.rs`
  - 内容: 零步长计算算法
  - 测试: `test_broadcast_strides_zero_stride`, `test_broadcast_strides_preserved`
  - 前置: T3
  - 预计: 10 min

### Wave 3: 视图创建与集成

- [ ] **T5**: 实现 `broadcast_to()` 方法
  - 文件: `src/broadcast.rs`, `src/tensor/mod.rs`
  - 内容: 广播视图创建，步长/形状/offset/LayoutFlags 更新
  - 测试: `test_broadcast_to_basic`, `test_broadcast_to_error`
  - 前置: T4
  - 预计: 15 min

- [ ] **T6**: 实现 `broadcast_with()` 便捷方法
  - 文件: `src/broadcast.rs`
  - 内容: 同时广播两个视图到公共形状
  - 测试: `test_broadcast_with_same_shape`, `test_broadcast_with_different_ndim`
  - 前置: T5
  - 预计: 10 min

### Wave 4: 测试与文档

- [ ] **T7**: 编写综合测试
  - 文件: `tests/broadcast.rs`
  - 内容: 各种广播组合、不兼容报错、禁止可变迭代验证
  - 测试: 覆盖所有公共 API
  - 前置: T1-T6
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
                      │
Wave 3:      [T5] → [T6]
                      │
Wave 4:           [T7]
```

---

## 7. 测试计划

### 7.0 测试分类表

| 测试分类 | 位置 | 说明 |
|----------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证广播规则、zero-stride 元数据与只读约束 |
| 集成测试 | `tests/` | 验证 `broadcast` 与 `math`、`overload`、`iter`、`layout` 的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖标量广播、空数组和高维广播等边界 |

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_can_broadcast_compatible` | 兼容形状：`[3,1,4]` vs `[4,1]` | 高 |
| `test_can_broadcast_incompatible` | 不兼容：`[3,2]` vs `[3,4]` | 高 |
| `test_broadcast_shape_basic` | `[3,1,4]` ⊕ `[4,1]` = `[3,4,4]` | 高 |
| `test_broadcast_shape_scalar` | `[5,4]` ⊕ `[1]` = `[5,4]` | 高 |
| `test_broadcast_shape_high_dim` | `[2,3,4,5]` ⊕ `[4,1]` = `[2,3,4,5]` | 中 |
| `test_broadcast_shape_error_msg` | 不兼容时错误信息包含维度信息 | 高 |
| `test_broadcast_strides_zero` | 广播维度步长为 0 | 高 |
| `test_broadcast_strides_preserved` | 非广播维度步长保留原值 | 高 |
| `test_broadcast_to_basic` | `[1,4]` → `[3,4]`，验证 shape 和 strides | 高 |
| `test_broadcast_to_error` | 不兼容形状返回 Err | 高 |
| `test_broadcast_with` | 两个不同形状张量同时广播 | 高 |
| `test_broadcast_read_only` | 广播视图不允许可变访问（编译期检查） | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 0 维张量广播到 `[M,N]` | 全零步长，所有元素相同 |
| 空数组广播 | 不触发，返回空视图 |
| `[1,1,1]` 广播到 `[5,5,5]` | 三轴全零步长 |
| 标量广播（`[1]` → `[1000,1000]`） | 内存仅 1 元素 |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `broadcast_to(X, S)` 后 `view.len() == S.size()` | 随机形状对 |
| 广播后原始数据不变 | 广播前后对比 |
| `can_broadcast(a, b) == true` ⟹ `broadcast_shape(a, b)` 成功 | 随机形状 |

### 7.4 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/broadcast.rs` | `broadcast_with()` 与 `math`、`overload`、`iter::Zip`、`layout` 标志位更新的协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
|------|----------|-----------|------|
| `broadcast ← overload` | `overload` | `broadcast_shape()` / `broadcast_with()` | 运算符重载路径调用广播模块完成公共形状推导 |
| `broadcast ← iter/zip` | `iter/zip` | `Zip::and()` | `Zip` 组合路径支持广播视图并检查兼容性，参见 `10-iterator.md` §5 |
| `broadcast ← shape` | `shape` | `broadcast_to()` | `shape` 模块通过该入口创建广播视图，参见 `16-shape.md` §4 |
| `broadcast → layout` | `layout` | `LayoutFlags` | 广播后设置 `HAS_ZERO_STRIDE` 并更新连续性，参见 `06-memory.md` §3 |
| `broadcast ← math` | `math` | 二元运算前置广播 | 二元运算在逐元素计算前先广播两个操作数，参见 `11-math.md` §4 |

### 8.2 数据流描述

```text
用户发起二元运算 / 显式 broadcast_to()
    │
    ├── broadcast 先比较两个输入 shape，生成公共输出 shape
    ├── 对被扩展的轴写入 zero stride 元数据
    ├── layout 更新 HAS_ZERO_STRIDE / contiguity 标志
    └── 返回只读 TensorView，供 iter::Zip / math / overload 继续消费
```

---

## 9. 设计决策记录

### 决策 1：零拷贝广播 vs 数据复制

| 属性 | 值 |
|------|-----|
| 决策 | 使用零步长实现广播，不复制数据 |
| 理由 | 内存效率（`[1,1000]` 广播到 `[1000,1000]` 仅 1KB vs 1MB）；O(1) 创建；可组合 |
| 替代方案 | 数据复制 — 放弃，内存开销大，广播链导致多次复制 |

### 决策 2：广播视图只读约束

| 属性 | 值 |
|------|-----|
| 决策 | 广播视图只允许只读访问，不提供可变广播视图 |
| 理由 | 别名冲突（多索引指向同一位置写入不确定性）；语义模糊；与 NumPy 一致 |
| 替代方案 | 允许写入并自动传播 — 放弃，语义混乱，性能开销大 |

### 决策 3：零步长实现方式

| 属性 | 值 |
|------|-----|
| 决策 | 步长直接设为 0，配合 `HAS_ZERO_STRIDE` 标志位 |
| 理由 | 简单直接；零步长乘法无运行时开销；缓存友好（重复读取同一地址） |
| 替代方案 | 特殊迭代器跳过广播维度 — 放弃，增加迭代器复杂度，通用性差 |

### 决策 4：错误处理方式

| 属性 | 值 |
|------|-----|
| 决策 | 维度不兼容返回 `XenonError::BroadcastError`，错误载荷保持与 `26-error.md` 中央定义一致 |
| 理由 | 广播模块只负责报告 shape 级不兼容；统一错误结构由 `26-error.md` 集中裁决，避免下游依赖未冻结的维度级字段 |
| 替代方案 | 简单 panic — 放弃，不可恢复，诊断信息不足 |

---

## 10. 性能考量

### 10.1 复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| `can_broadcast()` | O(min(N, M)) | O(1) |
| `broadcast_shape()` | O(max(N, M)) | O(max(N, M)) |
| `broadcast_strides()` | O(max(N, M)) | O(max(N, M)) |
| `broadcast_to()` | O(ndim) | O(1) 额外空间 |
| 数据复制广播 | O(elements) | O(elements) |

### 10.2 零步长访问开销

| 场景 | 标量访问 | 零步长访问 |
|------|---------|-----------|
| 索引计算 | `base + sum(stride[i] * idx[i])` | 相同（乘 0 无开销） |
| 缓存行为 | 连续访问，缓存友好 | 重复访问同一地址，缓存极佳 |
| SIMD | 可向量化 | 可向量化（加载一次，广播） |

### 10.3 性能数据（参考）

| 操作 | 连续数组 | 广播视图（零步长） | 性能比 |
|------|----------|---------------------|--------|
| 遍历 1M 元素（广播维度=1） | ~1ms | ~1.1ms | 1.1x |
| 标量广播 mapv | ~1ms | ~0.9ms | 0.9x（更优） |

---

## 11. no_std 兼容性

广播模块在 `no_std` 环境下可用，但需注意动态维度和错误类型的堆分配依赖。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `can_broadcast()` | ✅ | 纯计算函数，无堆分配 |
| `broadcast_shape()` | ✅ | 返回 `IxDyn`（内部 `Vec<usize>`），`extern crate alloc` 始终可用 |
| `broadcast_strides()` | ✅ | 返回 `Vec<isize>`，`extern crate alloc` 始终可用 |
| `broadcast_to()` | ✅ | 创建 `TensorView`（零拷贝），无堆分配 |
| `broadcast_with()` | ✅ | 创建两个 `TensorView`，需 `no_std + alloc`（IxDyn），参见 `02-dimension.md` §3 |
| `XenonError::BroadcastError` | ✅ | 使用 `core::fmt::Display`，无堆依赖，参见 `26-error.md` §4 |

条件编译处理：

```rust
// IxDyn uses alloc::vec::Vec internally (see 02-dimension.md §4.3)
// No external dependency needed — reuses existing dimension type.

#[cfg(not(feature = "std"))]
extern crate alloc;

// XenonError::BroadcastError uses core::fmt::Display, no std::error::Error needed
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
