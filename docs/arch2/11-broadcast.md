# 广播机制模块设计

> 文档编号: 11 | 模块: `src/broadcast.rs` | 阶段: Phase 3（API 模块）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `07-tensor-core.md`, `06-layout.md`, `02-dimension.md`, `08-error.md`

---

## 1. 模块定位

广播模块实现了 NumPy 风格的多维数组广播机制，使形状不同但兼容的数组能进行逐元素运算。这是 Renon 逐元素运算（`ops/element_wise.rs`）、矩阵批量运算（`ops/matrix.rs`）和 zip 迭代器（`iter/zip.rs`）的核心基础设施。

### 核心设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| 广播实现方式 | 零拷贝视图（stride=0） | NumPy/ndarray 标准做法，无数据复制 |
| 返回类型 | `TensorView<'a, A, IxDyn>` | 广播可能改变维度数（左侧补 1），须用动态维度 |
| 广播视图可写性 | 只读 | stride=0 意味着多索引指向同一元素，写入会产生歧义 |
| 形状兼容判定 | NumPy 两步规则 | 生态兼容性，用户心智模型一致 |
| 广播时机 | 惰性（按需创建视图） | 运算符隐式调用 `broadcast_with`，用户也可显式调用 `broadcast_to` |

### 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 形状兼容性判定 | `broadcast_shape()`、`can_broadcast()` | 具体运算的形状校验（由 ops 模块负责） |
| 广播视图创建 | `broadcast()`、`broadcast_to()` | 运算符重载（由 ops 模块负责） |
| 双数组联合广播 | `broadcast_with()` | zip 迭代器实现（由 iter 模块负责） |
| 虚拟步长计算 | stride=0 替换 | 常规步长计算（由 layout 模块负责） |

---

## 2. 文件位置

```
src/broadcast.rs              # 本模块：所有广播逻辑
src/lib.rs                    # pub mod broadcast; + re-export
tests/broadcasting.rs          # 集成测试
tests/property/broadcast_rules.rs  # 属性测试
```

单文件设计：广播是自包含的算法模块，约 400-500 行，拆分无必要。

---

## 3. 依赖关系

```
broadcast.rs
├── crate::dimension       # Dimension trait, IxDyn
├── crate::storage         # ViewRepr (广播结果为只读视图)
├── crate::tensor          # TensorBase, TensorView, Tensor type aliases
├── crate::layout          # LayoutFlags, compute_flags
└── crate::error           # TensorError::BroadcastError, Result
```

### 依赖的具体类型

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `dimension` | `Dimension`, `IxDyn`, `Ix0` |
| `storage` | `ViewRepr<'a, A>` |
| `tensor` | `TensorBase<S, D>`, `TensorView<'a, A, D>`, `Tensor<A, D>` |
| `layout` | `LayoutFlags`, `compute_flags` |
| `error` | `TensorError::BroadcastError`, `Result<T>` |

### 下游消费者

| 模块 | 使用方式 |
|------|----------|
| `ops/element_wise.rs` | 二元运算前调用 `broadcast_with` |
| `ops/matrix.rs` | batch 维度广播 |
| `iter/zip.rs` | 形状不一致时自动广播 |
| `ops/reduction.rs` | 部分归约场景 |

---

## 4. 公共 API 设计

### 4.1 `broadcast_shape` — 计算广播结果形状

```rust
/// Computes the broadcast result shape from two input shapes.
///
/// Follows NumPy broadcasting rules:
/// 1. Shapes are right-aligned (the shorter shape is left-padded with 1s).
/// 2. For each aligned dimension pair, both must be equal or one of them must be 1.
/// 3. The result dimension is `max(lhs_dim, rhs_dim)`.
///
/// # Arguments
///
/// * `lhs` - Shape of the left-hand side array.
/// * `rhs` - Shape of the right-hand side array.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if any aligned dimension pair is
/// incompatible (neither equal nor 1).
///
/// # Examples
///
/// ```ignore
/// use Renon::broadcast::broadcast_shape;
///
/// let result = broadcast_shape(&[3, 1], &[1, 4])?;
/// assert_eq!(result, vec![3, 4]);
///
/// let result = broadcast_shape(&[5, 3, 1], &[3, 4])?;
/// assert_eq!(result, vec![5, 3, 4]);
/// ```
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>>;
```

### 4.2 `can_broadcast` — 检查形状兼容性

```rust
/// Checks whether two shapes are compatible for broadcasting.
///
/// Equivalent to `broadcast_shape(lhs, rhs).is_ok()` but avoids allocating
/// the result shape when only a boolean answer is needed.
///
/// # Arguments
///
/// * `lhs` - Shape of the left-hand side array.
/// * `rhs` - Shape of the right-hand side array.
///
/// # Returns
///
/// `true` if the shapes can be broadcast together, `false` otherwise.
///
/// # Examples
///
/// ```ignore
/// use Renon::broadcast::can_broadcast;
///
/// assert!(can_broadcast(&[3, 1], &[1, 4]));
/// assert!(can_broadcast(&[5, 3, 1], &[3, 4]));
/// assert!(!can_broadcast(&[3, 2], &[2, 3]));
/// ```
#[inline]
pub fn can_broadcast(lhs: &[usize], rhs: &[usize]) -> bool;
```

### 4.3 `broadcast` — 将张量广播到目标形状（自由函数）

```rust
/// Broadcasts a tensor to the target shape, returning a read-only view.
///
/// The returned view shares the underlying data with the input tensor.
/// Broadcast dimensions (where the source has size 1 but the target has a
/// larger size) are represented with stride 0, so the single element is
/// logically repeated without any memory copy.
///
/// # Type Parameters
///
/// * `S` - Storage type of the input tensor (any `RawStorage`).
/// * `D` - Dimension type of the input tensor.
///
/// # Arguments
///
/// * `tensor` - The tensor to broadcast.
/// * `target_shape` - The desired output shape.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if the tensor's shape is not
/// compatible with `target_shape` under NumPy broadcasting rules.
///
/// # Examples
///
/// ```ignore
/// use Renon::{Tensor, Ix2, broadcast};
///
/// let a: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
/// let b = broadcast(&a, &[3, 4])?;  // shape [1, 4] -> [3, 4]
/// assert_eq!(b.shape(), &[3, 4]);
/// // b is a view: stride[0] == 0, stride[1] == 1
/// ```
pub fn broadcast<S, D>(
    tensor: &TensorBase<S, D>,
    target_shape: &[usize],
) -> Result<TensorView<'_, S::Elem, IxDyn>>
where
    S: RawStorage,
    D: Dimension;
```

### 4.4 `broadcast_to` 方法（TensorBase 上的方法）

此方法通过在 `TensorBase` 上添加扩展方法实现。由于 `broadcast.rs` 不能修改 `tensor.rs`，采用以下策略：

- 在 `broadcast.rs` 中定义一个 `BroadcastExt` trait，为所有 `TensorBase<S, D>` 提供 `broadcast_to` 方法
- 在 `lib.rs` 中 re-export 该 trait，用户只需 `use Renon::BroadcastExt`

```rust
/// Extension trait providing broadcast methods on `TensorBase`.
///
/// Import this trait to enable `broadcast_to()` and `broadcast_with()` as
/// methods on any tensor type:
///
/// ```ignore
/// use Renon::{Tensor, Ix2, BroadcastExt};
///
/// let a: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
/// let b = a.broadcast_to(&[3, 4])?;
/// ```
pub trait BroadcastExt<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Broadcasts this tensor to the given target shape.
    ///
    /// Convenience wrapper around the free function [`broadcast`].
    /// See [`broadcast`] for full documentation.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::BroadcastError` if the shapes are incompatible.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use Renon::{Tensor, Ix2, BroadcastExt};
    ///
    /// let a: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
    /// let b = a.broadcast_to(&[3, 4])?;
    /// assert_eq!(b.shape(), &[3, 4]);
    /// ```
    fn broadcast_to(
        &self,
        target_shape: &[usize],
    ) -> Result<TensorView<'_, S::Elem, IxDyn>>;

    /// Broadcasts this tensor together with another tensor, returning
    /// a pair of read-only views with a common shape.
    ///
    /// This is the primary entry point for binary operations (add, sub, etc.)
    /// that need to align two tensors via broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to broadcast with.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::BroadcastError` if the shapes are incompatible.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use Renon::{Tensor, Ix2, BroadcastExt};
    ///
    /// let a: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [3, 1]);
    /// let b: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
    /// let (va, vb) = a.broadcast_with(&b)?;
    /// assert_eq!(va.shape(), &[3, 4]);
    /// assert_eq!(vb.shape(), &[3, 4]);
    /// ```
    fn broadcast_with<S2, D2>(
        &self,
        other: &TensorBase<S2, D2>,
    ) -> Result<(
        TensorView<'_, S::Elem, IxDyn>,
        TensorView<'_, S2::Elem, IxDyn>,
    )>
    where
        S2: RawStorage,
        D2: Dimension;
}

// Blanket impl for all TensorBase<S, D>
impl<S, D> BroadcastExt<S, D> for TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    fn broadcast_to(
        &self,
        target_shape: &[usize],
    ) -> Result<TensorView<'_, S::Elem, IxDyn>> {
        broadcast(self, target_shape)
    }

    fn broadcast_with<S2, D2>(
        &self,
        other: &TensorBase<S2, D2>,
    ) -> Result<(
        TensorView<'_, S::Elem, IxDyn>,
        TensorView<'_, S2::Elem, IxDyn>,
    )>
    where
        S2: RawStorage,
        D2: Dimension,
    {
        broadcast_with(self, other)
    }
}
```

### 4.5 `broadcast_with` — 双数组联合广播（自由函数）

```rust
/// Broadcasts two tensors to a common compatible shape.
///
/// Computes the broadcast result shape from both input shapes, then
/// broadcasts each tensor to that shape. Returns a pair of read-only
/// views sharing data with the original tensors.
///
/// # Type Parameters
///
/// * `S1` - Storage type of the first tensor.
/// * `D1` - Dimension type of the first tensor.
/// * `S2` - Storage type of the second tensor.
/// * `D2` - Dimension type of the second tensor.
///
/// # Arguments
///
/// * `lhs` - The left-hand side tensor.
/// * `rhs` - The right-hand side tensor.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if the shapes cannot be reconciled.
///
/// # Examples
///
/// ```ignore
/// use Renon::{Tensor, Ix2, broadcast_with};
///
/// let a: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3, 1]);
/// let b: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
/// let (va, vb) = broadcast_with(&a, &b)?;
/// assert_eq!(va.shape(), &[3, 4]);
/// assert_eq!(vb.shape(), &[3, 4]);
/// ```
pub fn broadcast_with<S1, D1, S2, D2>(
    lhs: &TensorBase<S1, D1>,
    rhs: &TensorBase<S2, D2>,
) -> Result<(
    TensorView<'_, S1::Elem, IxDyn>,
    TensorView<'_, S2::Elem, IxDyn>,
)>
where
    S1: RawStorage,
    D1: Dimension,
    S2: RawStorage,
    D2: Dimension;
```

### 4.6 `broadcast_strides` — 内部虚拟步长计算

```rust
/// Computes the broadcast strides for a source shape broadcasted to a target shape.
///
/// For each dimension:
/// - If the source dimension equals the target dimension, the original stride is preserved.
/// - If the source dimension is 1 and the target is larger, the stride is set to 0
///   (virtual repetition).
/// - If the source has fewer dimensions, the "missing" dimensions get stride 0.
///
/// # Arguments
///
/// * `src_shape` - The original tensor shape.
/// * `src_strides` - The original tensor strides (signed, element units).
/// * `target_shape` - The broadcast target shape.
///
/// # Returns
///
/// A `Vec<isize>` of strides for the broadcast view.
///
/// # Panics
///
/// Panics if `src_shape` and `src_strides` have different lengths.
/// Panics if `src_shape` has more dimensions than `target_shape`.
/// (These are programming errors; callers must ensure compatibility.)
fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    target_shape: &[usize],
) -> Vec<isize>;
```

---

## 5. 内部实现设计

### 5.1 NumPy 广播规则详解

广播遵循 NumPy 的两步规则：

**步骤 1 — 维度对齐**：将两个形状右对齐，较短形状的左侧补 1。

```
形状 A:        [5, 3, 1]
形状 B:           [3, 4]
                   ↓
对齐后 A:     [5, 3, 1]
对齐后 B:     [1, 3, 4]
```

**步骤 2 — 逐维比较**：从右到左（或从左到右，顺序无关），检查每对维度：
- 两者相等 → 结果维度 = 该值
- 其中一个为 1 → 结果维度 = 较大值，size-1 的维度步长设为 0
- 两者不相等且均不为 1 → **广播失败**，返回 `BroadcastError`

### 5.2 形状兼容性检查算法

```rust
fn check_broadcast_compat(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = lhs.len().max(rhs.len());
    let mut result = Vec::with_capacity(max_ndim);

    // Iterate from right to left (highest axis index to lowest)
    for i in 0..max_ndim {
        let li = if i < lhs.len() { lhs[lhs.len() - 1 - i] } else { 1 };
        let ri = if i < rhs.len() { rhs[rhs.len() - 1 - i] } else { 1 };

        match (li, ri) {
            (a, b) if a == b => result.push(a),
            (1, b) => result.push(b),
            (a, 1) => result.push(a),
            (a, b) => {
                return Err(TensorError::BroadcastError {
                    lhs_shape: lhs.to_vec(),
                    rhs_shape: rhs.to_vec(),
                    axis: max_ndim - 1 - i,  // Convert to left-to-right axis index
                    lhs_dim: a,
                    rhs_dim: b,
                });
            }
        }
    }

    // Reverse to get left-to-right order
    result.reverse();
    Ok(result)
}
```

### 5.3 虚拟步长计算

广播的核心是 **零拷贝**：通过将 size-1 维度的步长设为 0，实现元素的"虚拟重复"。当索引沿该维度前进时，步长 0 使指针不移动，始终指向同一元素。

```
源张量:  shape=[1, 4], strides=[1, 1]  (F-order)
目标形状: [3, 4]
广播后:   shape=[3, 4], strides=[0, 1]

访问 broadcast_view[[2, 3]]:
  offset = 2 * strides[0] + 3 * strides[1]
         = 2 * 0 + 3 * 1
         = 3
  → 与访问 broadcast_view[[0, 3]] 相同（行 2 是行 0 的虚拟重复）
```

步长计算算法：

```rust
fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    target_shape: &[usize],
) -> Vec<isize> {
    assert_eq!(src_shape.len(), src_strides.len());
    assert!(src_shape.len() <= target_shape.len());

    let ndim = target_shape.len();
    let mut result = Vec::with_capacity(ndim);

    for i in 0..ndim {
        // Map from target axis to source axis (right-aligned)
        let src_i = i as isize - (ndim - src_shape.len()) as isize;

        if src_i < 0 {
            // Left-padded dimension: source doesn't have this axis → stride 0
            result.push(0);
        } else {
            let si = src_i as usize;
            if src_shape[si] == target_shape[i] {
                // Same size: preserve original stride
                result.push(src_strides[si]);
            } else {
                // src_shape[si] == 1, target_shape[i] > 1: broadcast via stride 0
                result.push(0);
            }
        }
    }

    result
}
```

### 5.4 零拷贝广播视图构建

`broadcast()` 函数创建一个新的 `TensorView`，复用源张量的存储指针和偏移量，替换形状和步长：

```rust
pub fn broadcast<S, D>(
    tensor: &TensorBase<S, D>,
    target_shape: &[usize],
) -> Result<TensorView<'_, S::Elem, IxDyn>>
where
    S: RawStorage,
    D: Dimension,
{
    let src_shape = tensor.shape();
    let src_strides = tensor.strides();

    // Step 1: Validate compatibility
    if !is_broadcast_compatible(src_shape, target_shape) {
        return Err(make_broadcast_error(src_shape, target_shape));
    }

    // Step 2: Compute broadcast strides
    let new_strides = broadcast_strides(src_shape, src_strides, target_shape);

    // Step 3: Build broadcast view
    let new_shape = IxDyn::new(target_shape);
    let new_strides_dim = IxDyn::from(new_strides);

    // Step 4: Compute layout flags (sets HAS_ZERO_STRIDE)
    let ptr_addr = tensor.as_ptr() as usize;
    let elem_size = core::mem::size_of::<S::Elem>();
    let layout_flags = crate::layout::compute_flags(
        target_shape,
        &new_strides,
        ptr_addr,
        elem_size,
        crate::layout::DEFAULT_ALIGNMENT,
    );

    // Step 5: Create view (shares data, zero-copy)
    Ok(unsafe {
        TensorView::from_raw_parts(
            tensor.as_ptr(),
            new_shape,
            new_strides_dim,
            tensor.offset(),
        )
    })
}
```

**关键细节**：

1. **存储指针不变**：广播视图的 `as_ptr()` 与源张量相同（同一底层缓冲区）
2. **偏移量不变**：`offset` 字段保持源张量的值
3. **新形状和步长**：`IxDyn` 容纳任意维度数的广播结果
4. **LayoutFlags 重算**：必须调用 `compute_flags`，因为步长变化会影响连续性和零步长标志
5. **只读保证**：返回 `TensorView`（非 `TensorViewMut`），类型系统保证不可写

### 5.5 标量广播

标量（0 维张量 `Tensor<A, Ix0>`）可以广播到任意形状。此时源形状为空 `[]`，所有目标维度步长为 0：

```
源张量:  shape=[], strides=[] (标量, 包含 1 个元素)
目标形状: [3, 4]
广播后:   shape=[3, 4], strides=[0, 0]

所有索引都指向同一个标量元素。
```

`broadcast_strides` 天然处理这种情况：当 `src_shape.len() == 0` 时，所有目标维度的 `src_i < 0`，步长全部设为 0。

### 5.6 惰性广播策略

广播是惰性的——视图创建时只修改元数据（shape, strides, flags），不复制数据。实际数据访问在索引或迭代时发生。

| 操作 | 广播开销 |
|------|----------|
| `broadcast()` / `broadcast_to()` | O(ndim)，仅计算步长和标志 |
| `broadcast_with()` | O(ndim₁ + ndim₂)，计算结果形状 + 两组步长 |
| 迭代广播视图 | O(n)，每次索引访问涉及 ndim 次乘法（含 stride=0 优化） |
| `to_owned()`（物化） | O(n)，复制所有逻辑元素到新缓冲区 |

### 5.7 与 LayoutFlags 的交互

广播后 `LayoutFlags` 的变化规则：

| 属性 | 广播前 | 广播后 |
|------|--------|--------|
| F_CONTIGUOUS | 可能 | **通常丢失**（stride=0 不满足连续性判定） |
| C_CONTIGUOUS | 可能 | **通常丢失**（同上） |
| ALIGNED | 是 | **保持**（数据指针不变） |
| HAS_ZERO_STRIDE | 否 | **新增**（广播维度的步长为 0） |
| HAS_NEG_STRIDE | 可能 | 继承源（广播不改变非广播步长的符号） |

**特例**：当广播不引入任何新维度（例如 `[3, 1]` → `[3, 1]`，no-op），所有标志保持不变。当 `[3, 1]` → `[3, 4]` 时，stride[1] 变为 0，HAS_ZERO_STRIDE 被设置。

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### Wave 1: 核心算法

- [ ] **T1: 创建 `src/broadcast.rs` 骨架 + 模块导入**
  - 文件: `src/broadcast.rs:1-30`
  - 内容: 文件级 doc comment、`use` 导入（Dimension, IxDyn, TensorBase, TensorView, LayoutFlags, TensorError, Result）
  - 测试: 编译通过
  - 前置: tensor, dimension, layout, error 模块完成
  - 预计: 5 min

- [ ] **T2: 实现 `broadcast_shape()` — 形状兼容性检查 + 结果形状计算**
  - 文件: `src/broadcast.rs`
  - 内容: 右对齐逻辑、逐维比较、错误构造、结果形状返回
  - 测试: `test_broadcast_shape_basic`, `test_broadcast_shape_left_pad`, `test_broadcast_shape_incompatible`, `test_broadcast_shape_scalar`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: 实现 `can_broadcast()` — 布尔兼容性检查**
  - 文件: `src/broadcast.rs`
  - 内容: 复用 `broadcast_shape` 内部逻辑但不分配结果，提前返回
  - 测试: `test_can_broadcast_true_cases`, `test_can_broadcast_false_cases`
  - 前置: T2
  - 预计: 8 min

### Wave 2: 步长计算与视图构建

- [ ] **T4: 实现 `broadcast_strides()` — 虚拟步长计算**
  - 文件: `src/broadcast.rs`
  - 内容: 右对齐映射、stride=0 替换逻辑、断言校验
  - 测试: `test_broadcast_strides_preserve`, `test_broadcast_strides_zero`, `test_broadcast_strides_left_pad`, `test_broadcast_strides_scalar`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5: 实现 `broadcast()` 自由函数 — 单张量广播**
  - 文件: `src/broadcast.rs`
  - 内容: 兼容性检查 + 步长计算 + `TensorView` 构建 + `LayoutFlags` 计算
  - 测试: `test_broadcast_basic`, `test_broadcast_shape_expanded`, `test_broadcast_incompatible_returns_error`, `test_broadcast_preserves_data`
  - 前置: T2, T4
  - 预计: 10 min

- [ ] **T6: 实现 `broadcast_with()` 自由函数 — 双张量联合广播**
  - 文件: `src/broadcast.rs`
  - 内容: 调用 `broadcast_shape` 获取公共形状 + 分别调用 `broadcast` 创建两个视图
  - 测试: `test_broadcast_with_basic`, `test_broadcast_with_different_ndim`, `test_broadcast_with_incompatible_returns_error`
  - 前置: T5
  - 预计: 10 min

### Wave 3: 扩展 trait 与集成

- [ ] **T7: 实现 `BroadcastExt` trait — 方法化接口**
  - 文件: `src/broadcast.rs`
  - 内容: `BroadcastExt<S, D>` trait 定义 + `TensorBase<S, D>` 的 blanket impl
  - 测试: `test_broadcast_ext_to`, `test_broadcast_ext_with`
  - 前置: T5, T6
  - 预计: 10 min

- [ ] **T8: lib.rs 注册模块 + re-export**
  - 文件: `src/lib.rs`
  - 内容: `pub mod broadcast;`、`pub use broadcast::{broadcast, broadcast_shape, broadcast_with, can_broadcast, BroadcastExt};`
  - 测试: 外部 `use Renon::BroadcastExt;` 编译通过
  - 前置: T7
  - 预计: 5 min

- [ ] **T9: 单元测试 — 边界情况**
  - 文件: `src/broadcast.rs #[cfg(test)] mod tests`
  - 内容: 空张量广播、单元素广播、高维（≥4 维）广播、相同形状 no-op、标量广播、`IxDyn` 与静态维度混合
  - 测试: `test_broadcast_empty_tensor`, `test_broadcast_single_element`, `test_broadcast_high_dim`, `test_broadcast_same_shape_noop`, `test_broadcast_scalar_to_any`
  - 前置: T5
  - 预计: 10 min

- [ ] **T10: 集成测试**
  - 文件: `tests/broadcasting.rs`
  - 内容: 跨模块测试——构造张量 → 广播 → 验证形状/步长/数据正确性、与运算符联动测试
  - 测试: `test_broadcast_then_index`, `test_broadcast_with_arithmetic`, `test_broadcast_layout_flags`
  - 前置: T8
  - 预计: 10 min

---

## 7. 测试计划

### 7.1 单元测试（`src/broadcast.rs` 内 `#[cfg(test)] mod tests`）

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **形状计算** | `test_broadcast_shape_basic` | `[3, 1]` × `[1, 4]` → `[3, 4]` |
| | `test_broadcast_shape_left_pad` | `[5, 3, 1]` × `[3, 4]` → `[5, 3, 4]` |
| | `test_broadcast_shape_scalar` | `[]` × `[2, 3]` → `[2, 3]` |
| | `test_broadcast_shape_same` | `[3, 4]` × `[3, 4]` → `[3, 4]` |
| | `test_broadcast_shape_incompatible` | `[3, 2]` × `[2, 3]` → `Err(BroadcastError)` |
| | `test_broadcast_shape_zero_dim` | `[0, 3]` × `[1, 3]` → `[0, 3]` |
| **兼容性检查** | `test_can_broadcast_true_cases` | 多种兼容形状返回 `true` |
| | `test_can_broadcast_false_cases` | 不兼容形状返回 `false` |
| **步长计算** | `test_broadcast_strides_preserve` | 相同维度步长不变 |
| | `test_broadcast_strides_zero` | size-1 维度步长变为 0 |
| | `test_broadcast_strides_left_pad` | 左侧补维度步长为 0 |
| | `test_broadcast_strides_scalar` | 标量广播所有步长为 0 |
| **广播视图** | `test_broadcast_basic` | 视图形状正确、共享数据指针 |
| | `test_broadcast_preserves_data` | 索引访问返回正确元素值 |
| | `test_broadcast_has_zero_stride_flag` | `layout_flags.has_zero_stride()` 为 `true` |
| | `test_broadcast_alignment_preserved` | `is_aligned()` 与源相同 |
| | `test_broadcast_contiguity_lost` | 广播后通常 `!is_contiguous()` |
| | `test_broadcast_incompatible_returns_error` | 不兼容形状返回 `BroadcastError` |
| **联合广播** | `test_broadcast_with_basic` | 两个视图形状相同 |
| | `test_broadcast_with_different_ndim` | 不同维度数正确处理 |
| **边界情况** | `test_broadcast_empty_tensor` | 含零维度张量的广播 |
| | `test_broadcast_single_element` | 单元素张量广播到多维 |
| | `test_broadcast_high_dim` | 4-6 维张量广播 |
| | `test_broadcast_same_shape_noop` | 相同形状广播为 identity |
| | `test_broadcast_scalar_to_any` | 标量广播到各种形状 |
| | `test_broadcast_1d_to_2d` | `[4]` 广播到 `[3, 4]` |
| | `test_broadcast_negative_stride` | 源有负步长时广播保持负步长 |

### 7.2 集成测试（`tests/broadcasting.rs`）

| 测试函数 | 场景 | 预期 |
|----------|------|------|
| `test_broadcast_then_index` | 广播后通过索引访问 | 所有位置返回正确的广播元素 |
| `test_broadcast_to_owned` | 广播视图 `to_owned()` | 物化后数据与预期一致 |
| `test_broadcast_with_arithmetic` | 广播 + 逐元素加法 | 结果形状 = 广播后形状，数据正确 |
| `test_broadcast_layout_flags` | 广播后 LayoutFlags 检查 | HAS_ZERO_STRIDE=true, ALIGNED 继承 |
| `test_broadcast_read_only` | 广播视图不可写 | 编译时保证（TensorView 而非 TensorViewMut） |
| `test_broadcast_ixdyn_tensor` | IxDyn 张量广播 | 正确处理动态维度 |
| `test_broadcast_ix2_tensor` | 静态维度张量广播 | 返回 IxDyn 视图，数据正确 |
| `test_broadcast_error_message` | 不兼容广播的错误信息 | 包含 lhs/rhs 形状和冲突维度 |

### 7.3 属性测试（`tests/property/broadcast_rules.rs`）

| 不变量 | 测试方法 |
|--------|----------|
| `can_broadcast(a, b) == true` ⟹ `broadcast(a, target).is_ok()` | 随机兼容形状对 |
| 广播视图的 `to_owned()` 长度 = 目标形状元素总数 | 随机形状 |
| 广播后步长中 0 的个数 = 源 size-1 维度数 + 左侧补的维度数 | 随机形状 |
| `broadcast_with(a, b)` 结果形状 = `broadcast_shape(a.shape, b.shape)` | 随机兼容对 |
| `can_broadcast(a, b) == can_broadcast(b, a)` | 随机形状对 |
| 广播 no-op: `broadcast(x, x.shape())` 的视图与原数据相同 | 随机张量 |

### 7.4 典型测试用例

#### 基本 2D 广播

```rust
#[test]
fn test_broadcast_basic() {
    // shape [1, 4] broadcast to [3, 4]
    let a: Tensor<f64, Ix2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [1, 4]);
    let b = broadcast(&a, &[3, 4]).unwrap();

    assert_eq!(b.shape(), &[3, 4]);
    assert_eq!(b.strides(), &[0, 1]);  // dim 0 stride = 0
    assert!(b.has_zero_stride());

    // All rows should be identical
    for row in 0..3 {
        for col in 0..4 {
            assert_eq!(b.get(&[row, col]).unwrap(), &a.get(&[0, col]).unwrap());
        }
    }
}
```

#### 不同维度数广播

```rust
#[test]
fn test_broadcast_with_different_ndim() {
    // [5, 3, 1] × [3, 4] → [5, 3, 4]
    let a: Tensor<f64, Ix3> = Tensor::zeros([5, 3, 1]);
    let b: Tensor<f64, Ix2> = Tensor::zeros([3, 4]);

    let (va, vb) = broadcast_with(&a, &b).unwrap();

    assert_eq!(va.shape(), &[5, 3, 4]);
    assert_eq!(vb.shape(), &[5, 3, 4]);
    assert_eq!(va.strides()[2], 0);  // a's dim 2 was 1
    assert_eq!(vb.strides()[0], 0);  // b's dim 0 was left-padded
}
```

#### 标量广播

```rust
#[test]
fn test_broadcast_scalar_to_any() {
    let scalar: Tensor<f64, Ix0> = Tensor::from_elem(42.0);
    let view = broadcast(&scalar, &[2, 3, 4]).unwrap();

    assert_eq!(view.shape(), &[2, 3, 4]);
    assert_eq!(view.strides(), &[0, 0, 0]);
    // Every element is 42.0
    for idx in 0..24 {
        let i = idx / 12;
        let j = (idx / 4) % 3;
        let k = idx % 4;
        assert_eq!(*view.get(&[i, j, k]).unwrap(), 42.0);
    }
}
```

#### 广播失败

```rust
#[test]
fn test_broadcast_incompatible_returns_error() {
    let a: Tensor<f64, Ix2> = Tensor::zeros([3, 2]);
    let b: Tensor<f64, Ix2> = Tensor::zeros([2, 3]);

    let result = broadcast_with(&a, &b);
    assert!(result.is_err());

    match result.unwrap_err() {
        TensorError::BroadcastError { lhs_shape, rhs_shape, axis, .. } => {
            assert_eq!(lhs_shape, vec![3, 2]);
            assert_eq!(rhs_shape, vec![2, 3]);
            // Incompatible at axis 0 (3 vs 2) or axis 1 (2 vs 3)
            assert!(axis == 0 || axis == 1);
        }
        _ => panic!("expected BroadcastError"),
    }
}
```

---

## 附录 A: 广播语义速查表

| 源形状 | 目标形状 | 广播后步长 | 说明 |
|--------|----------|-----------|------|
| `[]` | `[3, 4]` | `[0, 0]` | 标量广播 |
| `[4]` | `[3, 4]` | `[0, 1]` | 1D → 2D，左侧补 1 |
| `[1, 4]` | `[3, 4]` | `[0, 1]` | 行广播（行重复） |
| `[3, 1]` | `[3, 4]` | `[1, 0]` | 列广播（列重复） |
| `[3, 4]` | `[3, 4]` | `[1, 4]` | No-op（形状相同） |
| `[1, 1]` | `[3, 4]` | `[0, 0]` | 全广播（单元素展开） |
| `[5, 1, 4]` | `[5, 3, 4]` | `[4, 0, 1]` | 中间轴广播 |
| `[1, 3, 1]` | `[2, 3, 4]` | `[0, 1, 0]` | 多轴同时广播 |
| `[2, 3]` | `[3, 4]` | — | **失败**（2≠3 且 3≠4） |

## 附录 B: 与运算模块的集成模式

```
用户调用: a + b (逐元素加法)
    │
    ├── ops/element_wise.rs 检查形状
    │   ├── 形状完全一致 → 直接运算
    │   └── 形状不一致 → 调用 broadcast_with(&a, &b)
    │       ├── 广播成功 → 对两个广播视图执行运算
    │       └── 广播失败 → 返回 BroadcastError
    │
    └── 运算执行路径
        ├── 分配结果 Tensor<A, IxDyn>（广播后形状）
        ├── 遍历广播视图的所有逻辑元素
        │   （stride=0 使访问自动回到正确位置）
        └── 返回结果张量
```

## 附录 C: lib.rs re-export 清单

```rust
// src/lib.rs additions for broadcast module
pub mod broadcast;

pub use crate::broadcast::{
    broadcast,
    broadcast_shape,
    broadcast_with,
    can_broadcast,
    BroadcastExt,
};
```
