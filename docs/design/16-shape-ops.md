# 形状操作模块设计

> 文档编号: 16 | 模块: `src/shape_ops/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `06-memory-layout.md`
> 需求参考: 需求说明书 §17

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 转置操作 | `transpose()` / `t()` 交换步长和形状返回视图（O(1)） | squeeze / expand_dims（当前版本不提供） |
| N 维转置 | `permute_axes()` 任意轴排列 | stack / concatenate（当前版本不提供） |
| reshape 操作 | `reshape()` / `into_shape()` 改变形状 | pad / repeat / split（当前版本不提供） |
| 连续性检查 | reshape 壍检查数据连续性决定零拷贝或需拷贝路径 | flatten（由 reshape 实现） |
| 转置便捷方法 | `swap_axes()` / `moveaxis()` | arange / linspace / logspace（当前版本不提供） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零拷贝优先 | 凡能通过调整步长和偏移量实现的操作，均返回视图而非复制数据 |
| 路径选择显式 | reshape 连续时零拷贝，非连续时须显式拷贝，语义清晰无隐式陷阱） |
| BLAS 友好 | 保持 F-order 布局的连续性，确保与 BLAS 互操作 |
| 维度安全 | reshape 须保持总元素数不变，编译期/运行时检查 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: shape_ops  ← 当前模块
```

---

## 2. 文件位置

```
src/shape_ops/
├── mod.rs             # 模块入口，re-export 公开 trait 和函数
├── transpose.rs       # transpose, t, permute_axes, swap_axes, moveaxis
└── reshape.rs         # reshape, into_shape
```

文件划分理由：转置和重塑逻辑各自独立，拆分后职责清晰，方便单独测试和维护。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │   tensor     │
                    │ TensorBase   │
                    └──────┬───────┘
                           │ 使用
              ┌────────────┼────────────────┐
              │  shape_ops                  │
              │  transpose.rs │ reshape.rs  │
              └──┬───────────┬──────────────┘
                 │ 使用       │ 使用
          ┌──────▼───┐ ┌────▼────────────┐
          │ dimension │ │ memory-layout  │
          │ Dimension │ │ LayoutFlags    │
          │ Ix0~IxDyn │ │ Order          │
          └───────────┘ └────────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `TensorView`, `Tensor<A, D>`, `.shape()`, `.strides()`, `.offset()`，参见 `07-tensor.md` §4 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `RemoveAxis`, `IntoDimension`，参见 `02-dimension.md` §3 |
| `memory_layout` | `LayoutFlags`, `Order`, `is_f_contiguous()`, `is_c_contiguous()`，参见 `06-memory-layout.md` §3 |
| `error` | `XenonError::InvalidShape`, `XenonError::LayoutMismatch`，参见 `26-error-handling.md` §4 |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `shape_ops/` 消费 `tensor`、`dimension`、`memory_layout` 的 trait 和类型，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 转置操作

```rust
impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Transpose the array (reverse axis order).
    ///
    /// Returns a view with reversed axis order, zero-copy operation (O(1)).
    /// Equivalent to matrix transpose for 2D arrays.
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::<f64, _>::zeros([2, 3]);
    /// let b = a.transpose();
    /// assert_eq!(b.shape(), &[3, 2]);
    /// ```
    pub fn transpose(&self) -> TensorView<'_, A, D>
    where
        D: Reverse,
    {
        let new_shape = self.shape().reverse();
        let new_strides = self.strides().reverse();
        
        TensorView {
            storage: ViewRepr::from(&self.storage),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout: self.layout.update_for_transpose(),
        }
    }

    /// Shorthand for 2D matrix transpose.
    ///
    /// Equivalent to `.transpose()`, provides concise syntax similar to ndarray.
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::<f64, _>::zeros([3, 4]);
    /// let b = a.t();
    /// assert_eq!(b.shape(), &[4, 3]);
    /// ```
    pub fn t(&self) -> TensorView<'_, A, D>
    where
        D: Dimension + Reverse,
    {
        self.transpose()
    }
}
```

#### 4.1.1 转置语义

| 属性 | 行为 |
|------|------|
| 零拷贝 | 始终零拷贝（O(1)），仅调整步长和形状 |
| 形状变化 | `shape[i]` → `shape[ndim-1-i]`（全反转） |
| 步长变化 | `strides[i]` → `strides[ndim-1-i]`（全反转） |
| 连续性 | F-contiguous ↔ C-contiguous 互换 |
| 偏移量 | 保持不变 |
| 1D 数组 | 转置后形状不变（1D 无轴顺序概念） |

### 4.1.2 转置布局变化

F-contiguous 数组转置后变为 C-contiguous，反之亦然：

```
原始: shape=[2, 3], strides=[1, 2]  (F-order, F-contiguous)
转置: shape=[3, 2], strides=[2, 1]  (C-order, C-contiguous)
```

### 4.1.3 HAS_NEG_STRIDE 栄志处理

转置操作不引入负步长（仅交换步长值），因此无需设置 `HAS_NEG_STRIDE` 标志。但如果原始视图已有负步长，转置后保留。

```rust
fn update_flags_for_transpose(source_flags: LayoutFlags) -> LayoutFlags {
    let mut flags = source_flags;
    // Transpose swaps F-contiguous and C-contiguous
    let was_f = flags.is_f_contiguous();
    let was_c = flags.is_c_contiguous();
    flags.set_f_contiguous(was_c);
    flags.set_c_contiguous(was_f);
    flags
}
```

### 4.1.4 Good / Bad 对比

```rust
// Good - use t() for zero-copy transpose
let a = Tensor::<f64, _>::zeros([1000, 1000]);
let b = a.t();  // O(1), zero-copy
assert_eq!(b.shape(), &[1000, 1000]);

// Bad - manually copy data for transpose (wastes memory and time)
let a = Tensor::<f64, _>::zeros([1000, 1000]);
let mut b = Tensor::<f64, _>::zeros([1000, 1000]);
for i in 0..1000 {
    for j in 0..1000 {
        b[[j, i]] = a[[i, j]];  // O(n^2) copy, forbidden
    }
}
```

---

### 4.2 Reshape 操作

```rust
impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Reshape the array (zero-copy version).
    ///
    /// Only performs zero-copy when data is contiguous, otherwise returns an error.
    /// Total element count must remain unchanged.
    ///
    /// # Arguments
    /// * `shape` - Target shape
    ///
    /// # Returns
    /// * `Ok(TensorView)` - Reshaped view (zero-copy)
    /// * `Err(InvalidShape)` - Total element count mismatch
    /// * `Err(LayoutMismatch)` - Source array is non-contiguous
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::<f64, _>::zeros([2, 3, 4]);  // 24 elements
    /// let b = a.reshape([6, 4]).unwrap();          // shape: [6, 4]
    /// ```
    pub fn reshape<E>(&self, shape: E) -> Result<TensorView<'_, A, E>, XenonError>
    where
        E: Dimension,
    {
        // 1. Check element count
        let new_len: usize = shape.slice().iter().product();
        if new_len != self.len() {
            return Err(XenonError::InvalidShape {
                expected: self.len(),
                actual: new_len,
            });
        }
        
        // 2. Check F-contiguity (only F-contiguous arrays can be reshaped zero-copy)
        //    C-contiguous arrays (e.g. transposed views) are NOT eligible for
        //    zero-copy reshape because Xenon always outputs F-order.
        if !self.is_f_contiguous() {
            return Err(XenonError::LayoutMismatch {
                reason: "source array is not F-contiguous, use into_shape() instead",
            });
        }
        
        // 3. Compute new strides (always F-order)
        let new_strides = compute_strides(&shape, Order::F);
        
        // 4. Create view
        Ok(TensorView {
            storage: ViewRepr::from(&self.storage),
            shape,
            strides: new_strides,
            offset: self.offset,
            layout: LayoutFlags::from_order(Order::F),
        })
    }

    /// Consume the array and reshape it.
    ///
    /// Zero-copy if data is contiguous; automatically copies then reshapes if non-contiguous.
    ///
    /// # Arguments
    /// * `shape` - Target shape
    ///
    /// # Returns
    /// * `Ok(Tensor<A, E>)` - Reshaped owning tensor
    /// * `Err(InvalidShape)` - Total element count mismatch
    ///
    /// # Examples
    /// ```
    /// let a = Tensor::<f64, _>::zeros([4, 6]);
    /// let b = a.into_shape([2, 12]).unwrap();  // shape: [2, 12]
    /// ```
    pub fn into_shape<E>(self, shape: E) -> Result<Tensor<A, E>, XenonError>
    where
        E: Dimension,
        S: IntoOwned<A>,
    {
        let new_len: usize = shape.slice().iter().product();
        if new_len != self.len() {
            return Err(XenonError::InvalidShape {
                expected: self.len(),
                actual: new_len,
            });
        }
        
        // F-contiguous: reshape in-place (zero-copy)
        if self.is_f_contiguous() {
            let new_strides = compute_strides(&shape, Order::F);
            return Ok(Tensor {
                storage: self.storage.into_owned(),
                shape,
                strides: new_strides,
                offset: 0,
                layout: LayoutFlags::from_order(Order::F),
            });
        }
        
        // Non-contiguous: copy to contiguous then reshape
        let owned = self.to_contiguous();
        let new_strides = compute_strides(&shape, Order::F);
        Ok(Tensor {
            storage: owned.storage,
            shape,
            strides: new_strides,
            offset: 0,
            layout: LayoutFlags::from_order(Order::F),
        })
    }
}
```

### 4.2.1 Reshape 语义

| 属性 | 行为 |
|------|------|
| 元素总数 | 必须保持不变，否则返回 `InvalidShape` |
| 连续数组 | `reshape()` O(1) 零拷贝，仅更新元数据 |
| 非连续数组 | `reshape()` 返回 `LayoutMismatch`；`into_shape()` O(n) 自动拷贝 |
| 布局保持 | F-contiguous 输入默认输出 F-contiguous |
| 空数组 | 允许 reshape 到任意元素总数为 0 的形状 |

### 4.2.2 连续性检查逻辑

```
is_contiguous() = is_f_contiguous() || is_c_contiguous()

F-contiguous: strides[i] = product(shape[0..i])  (column-major)
  e.g. shape=[2,3,4], strides=[1,2,6]

C-contiguous: strides[i] = product(shape[i+1..ndim])  (row-major)
  e.g. shape=[2,3,4], strides=[12,4,1]

Non-contiguous example (reshape fails):
  original: shape=[4,6], strides=[6,1]
  after slice: shape=[2,6], strides=[12,1]  (non-contiguous)
  → reshape([12]) fails because strides[0]=12 ≠ 6
```

### 4.2.3 Good / Bad 对比

```rust
// Good - reshape contiguous array directly (O(1))
let a = Tensor::<f64, _>::zeros([2, 3, 4]);
let b = a.reshape([6, 4])?;  // O(1) zero-copy

// Good - use into_shape for non-contiguous array (auto-handled)
let slice = tensor.slice(s![1..3, ..]);
let b = slice.into_shape([12])?;  // O(n) copy if needed

// Bad - calling reshape on non-contiguous array (will fail)
let slice = tensor.slice(s![1..3, ..]);
slice.reshape([12])?;  // Returns Err(LayoutMismatch), should use into_shape()
```

---

## 5. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/shape_ops/mod.rs` 骨架
  - 文件: `src/shape_ops/mod.rs`, `src/shape_ops/transpose.rs`, `src/shape_ops/reshape.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: tensor、dimension、layout 模块完成
  - 预计: 5 min

### Wave 2: 转置实现

- [ ] **T2**: 实现 `transpose()` / `t()` 方法
  - 文件: `src/shape_ops/transpose.rs`
  - 内容: `TensorBase::transpose()`, `TensorBase::t()`, `LayoutFlags::update_for_transpose()`
  - 测试: `test_transpose_2d`, `test_transpose_3d`, `test_transpose_contiguity_swap`
  - 前置: T1
  - 预计: 10 min

### Wave 3: Reshape 实现

- [ ] **T3**: 实现 `reshape()` 方法
  - 文件: `src/shape_ops/reshape.rs`
  - 内容: `TensorBase::reshape()` — 连续性检查 + 零拷贝路径
  - 测试: `test_reshape_success`, `test_reshape_invalid_shape`, `test_reshape_non_contiguous`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `into_shape()` 方法
  - 文件: `src/shape_ops/reshape.rs`
  - 内容: `TensorBase::into_shape()` — 连续零拷贝 + 非连续拷贝路径
  - 测试: `test_into_shape_contiguous`, `test_into_shape_non_contiguous`
  - 前置: T3
  - 预计: 10 min

### Wave 4: 测试

- [ ] **T5**: 编写综合测试
  - 文件: `tests/shape_ops.rs`
  - 内容: 转置正确性、reshape 成功/失败、非连续 reshape、大数组性能、reshape 各种路径、大数组测试
  - 测试: 覆盖所有公共 API
  - 前置: T2, T3, T4
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1]
            │
Wave 2: [T2] ─── [T3] ─── [T4]
                          │
Wave 3:              [T5]
```

---

## 6. 测试计划

### 6.1 单元测试

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_transpose_2d` | `[2,3]` → `[3,2]`，验证 shape 和数据 | 高 |
| `test_transpose_3d` | `[2,3,4]` → `[4,3,2]`，验证轴反转 | 高 |
| `test_transpose_contiguity_swap` | F-contiguous 转置后 C-contiguous | 高 |
| `test_transpose_1d_noop` | 1D 数组转置后形状不变 | 中 |
| `test_transpose_0d_noop` | 0D 标量转置后不变 | 中 |
| `test_reshape_success` | `[2,3,4]` → `[6,4]`，O(1) 零拷贝 | 高 |
| `test_reshape_invalid_shape` | `[2,3]` → `[5,4]`，返回 `InvalidShape` | 高 |
| `test_reshape_non_contiguous` | 非连续数组 reshape 返回 `LayoutMismatch` | 高 |
| `test_into_shape_contiguous` | 连续数组 `into_shape` 零拷贝 | 高 |
| `test_into_shape_non_contiguous` | 非连续数组 `into_shape` 自动拷贝 | 高 |
| `test_reshape_empty` | 空数组 reshape | 中 |

### 6.2 边界测试

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | reshape 到 `[0, 3]` 或 `[3, 0]` 成功 |
| 单元素 `shape=[1, 1]` | reshape 到 `[1]` 成功 |
| 大数组 `[1000, 1000]` 转置 | O(1)，不拷贝 |
| 高维数组 `[2,3,4,5]` reshape | 到 `[6, 20]` 成功 |

### 6.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `transpose().len() == tensor.len()` | 随机形状 |
| `reshape(s).len() == tensor.len()` | 随机形状 |
| 转置后数据不变 | 转置前后逐元素对比 |
| 连续 reshape 后数据不变 | reshape 前后逐元素对比 |

---

## 7. 与其他模块的交互

| 交互模块 | 方向 | 说明 |
|----------|------|------|
| `tensor` | shape_ops → tensor | 使用 `TensorBase` 结构和 `TensorView` 创建方法，参见 `07-tensor.md` §4 |
| `dimension` | shape_ops → dimension | 使用 `Dimension` trait 的形状操作方法，参见 `02-dimension.md` §3 |
| `memory_layout` | shape_ops → memory_layout | 检查连续性、计算步长，参见 `06-memory-layout.md` §3 |
| `broadcast` | shape_ops ← broadcast | 广播视图不可 reshape（非连续），参见 `15-broadcast.md` §5 |
| `index` | index → shape_ops | 切片后可 reshape（如连续），参见 `17-indexing.md` §4 |

---

## 8. 设计决策记录

### 决策 1：转置不拷贝数据

| 属性 | 值 |
|------|-----|
| 决策 | 转置通过交换步长和形状实现，不拷贝数据 |
| 理由 | O(1) 操作；内存效率高；与 ndarray/NumPy 一致 |
| 替代方案 | 拷贝数据转置 — 放弃，O(n) 开销不必要 |

### 决策 2：reshape 语义（零拷贝 vs 拷贝）

| 属性 | 值 |
|------|-----|
| 决策 | 连续数据零拷贝 reshape（O(1)），非连续数据显式报错或自动拷贝（O(n)） |
| 理由 | 避免隐式拷贝的性能陷阱；用户可通过函数选择行为 |
| 替代方案 | 总是拷贝 — 放弃，不必要的性能损失；总是零拷贝 — 放弃，非连续数据语义错误 |

### 决策 3：当前版本仅支持 transpose 和 reshape

| 属性 | 值 |
|------|-----|
| 决策 | Phase 4 仅实现 transpose 和 reshape，其他形状操作留待后续版本 |
| 理由 | 需求说明书 §17 明确当前版本只提供转置和 reshape 操作 |
| 替代方案 | 一次性实现所有操作 — 放弃，范围过大，违反增量开发原则 |

---

## 9. 性能考量

### 9.1 复杂度

| 操作 | 连续输入 | 非连续输入 |
|------|----------|-----------|
| `transpose()` | O(1) | O(1) |
| `t()` | O(1) | O(1) |
| `reshape()` | O(1) | 返回 Err |
| `into_shape()` | O(1) | O(n)（拷贝） |

### 9.2 内存

| 操作 | 内存分配 | 数据拷贝 |
|------|----------|----------|
| `transpose()` | 无 | 无 |
| `t()` | 无 | 无 |
| `reshape()` (连续) | 无 | 无 |
| `into_shape()` (非连续) | O(n) | O(n) |

### 9.3 缓存行为

| 场景 | 缓存友好性 | 说明 |
|------|-----------|------|
| F-contiguous 转置后遍历 | 较差 | 变为 C-contiguous，跳跃访问 |
| 连续 reshape | 最优 | 步长与内存布局匹配 |
| 非连续 reshape（拷贝后） | 最优 | 拷贝后变为 F-contiguous |

---

## 10. no_std 兼容性

形状操作模块在 `no_std` 环境下可用，但需注意非连续 reshape 路径的堆分配依赖。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `transpose()` / `t()` | ✅ | 纯元数据操作（交换步长和形状），无堆分配 |
| `permute_axes()` | ✅ | 纯元数据操作，无堆分配 |
| `swap_axes()` / `moveaxis()` | ✅ | 纯元数据操作，无堆分配 |
| `reshape()`（连续路径） | ✅ | 仅更新元数据，无堆分配 |
| `into_shape()`（非连续路径） | ✅ | 需 `no_std + alloc`，调用 `to_contiguous()` 拷贝数据，参见 `05-storage.md` §5 |
| `LayoutFlags` 更新 | ✅ | 位标志操作，无依赖，参见 `06-memory-layout.md` §3 |

条件编译处理：

```rust
// transpose: zero-copy, pure metadata swap — works in pure no_std
// reshape (contiguous): zero-copy, pure metadata — works in pure no_std
// into_shape (non-contiguous): calls to_contiguous() → alloc::vec::Vec

#[cfg(not(feature = "std"))]
extern crate alloc;
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

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
