# 实用操作模块设计

> 文档编号: 20 | 模块: `src/ops/utility.rs` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `10-iterator.md`
> 需求参考: 需求说明书 §21, §22

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 范围裁剪 | `clip`（将元素限制在 [min, max] 范围内） | 其他 numpy 风格变换（flip/roll/shift） |
| 填充操作 | `fill`（原地填充所有逻辑元素） | 构造方法（zeros/ones/full，由 construct.rs 提供） |
| 连续性保证 | `to_contiguous`（确保内存连续存储） | 布局计算逻辑（由 layout 模块提供） |
| 非连续布局支持 | 通过迭代器正确处理非连续内存 | 布局优化策略 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 步长感知 | `fill`/`clip` 通过迭代器正确处理非连续内存布局 |
| 原地优先 | `fill` 为原地操作（`&mut self`），避免额外分配 |
| 类型安全 | `clip` 限制为数值类型（`RealScalar`/`PartialOrd`），编译期拒绝 `bool` |
| 语义清晰 | `to_contiguous` 返回 `Tensor<A, D>`，调用方可预测生命周期 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: ops/utility  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── ops/
    └── utility.rs    # clip / fill / to_contiguous
```

单文件设计：实用操作之间无强依赖，代码量适中（~200 行），无需拆分。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/ops/utility.rs
├── crate::tensor        # TensorBase<S, D>, Tensor, 类型别名
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut trait
├── crate::element       # Element, RealScalar trait
├── crate::layout        # is_f_contiguous / is_c_contiguous 查询
└── crate::iter          # Elements 迭代器（fill/clip 内部使用）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()` |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` |
| `storage` | `Storage<Elem=A>`, `StorageMut<Elem=A>` |
| `element` | `Element`, `RealScalar`（clip 约束） |
| `layout` | `is_f_contiguous()`, `is_c_contiguous()` |
| `iter` | `iter()`, `iter_mut()` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `ops/utility` 仅消费 `tensor`、`iter` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 clip 操作

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialOrd,
{
    /// Clamp each element to the [min, max] range.
    ///
    /// Returns a new tensor; the original tensor is unchanged.
    ///
    /// # Arguments
    ///
    /// * `min` - lower bound
    /// * `max` - upper bound
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `min > max`.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = Tensor1::from_vec(vec![-1.0, 0.5, 1.0, 2.0, 3.0]);
    /// let clipped = t.clip(0.0, 2.0);
    /// assert_eq!(clipped.to_vec(), vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    /// ```
    pub fn clip(&self, min: A, max: A) -> Tensor<A, D>
    where
        A: Clone,
    {
        debug_assert!(min <= max, "clip: min must be <= max");
        self.mapv(|x| if x < min { min.clone() } else if x > max { max.clone() } else { x })
    }

    /// Clip in place.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut t = Tensor1::from_vec(vec![-1.0, 0.5, 3.0]);
    /// t.clip_inplace(0.0, 2.0);
    /// assert_eq!(t.to_vec(), vec![0.0, 0.5, 2.0]);
    /// ```
    pub fn clip_inplace(&mut self, min: A, max: A)
    where
        S: StorageMut<Elem = A>,
        A: Clone,
    {
        debug_assert!(min <= max, "clip_inplace: min must be <= max");
        for elem in self.iter_mut() {
            if *elem < min {
                *elem = min.clone();
            } else if *elem > max {
                *elem = max.clone();
            }
        }
    }
}
```

### 4.2 fill 操作

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Fill all logical elements with the specified value (in-place).
    ///
    /// Correctly handles non-contiguous layouts: iterates over all logical
    /// elements via the iterator. Modifies storage directly without copying.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut t = Tensor1::<f64>::zeros([5]);
    /// t.fill(3.14);
    /// assert!(t.iter().all(|&x| x == 3.14));
    /// ```
    pub fn fill(&mut self, value: A) {
        for elem in self.iter_mut() {
            *elem = value.clone();
        }
    }
}
```

### 4.3 连续性保证（to_contiguous）

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Ensure data is stored contiguously in memory.
    ///
    /// - If already F-contiguous, returns `to_owned()` (copy)
    /// - If already C-contiguous, returns a C-contiguous version
    /// - If non-contiguous, copies into F-contiguous layout
    ///
    /// # Returns
    ///
    /// Always returns an owned `Tensor<A, D>`.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = Tensor2::<f64>::zeros([3, 4]);
    /// let contig = t.to_contiguous();
    /// assert!(contig.is_f_contiguous());
    /// ```
    pub fn to_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            self.to_owned()
        } else if self.is_c_contiguous() {
            self.to_c_contiguous()
        } else {
            self.to_f_contiguous()
        }
    }
}
```

### 4.4 Good / Bad 对比

```rust
// Good - use fill for in-place filling, zero extra allocation
let mut t = Tensor1::<f64>::zeros([1000]);
t.fill(42.0);

// Bad - create a temporary Vec then construct a new tensor, double allocation
let data = vec![42.0; 1000];
let t = Tensor1::from_vec(data);
```

```rust
// Good - check contiguity first before deciding whether to convert
if tensor.is_f_contiguous() {
    process(&tensor);
} else {
    let contiguous = tensor.to_contiguous();
    process(&contiguous);
}

// Bad - unconditionally call to_contiguous, wastes a copy when already contiguous
let contiguous = tensor.to_contiguous();  // potentially unnecessary O(n) copy
process(&contiguous);
```

---

## 5. 内部实现设计

### 5.1 clip 算法

```
clip(tensor, min, max):
    allocate result tensor with same shape
    for each element x in tensor (via iterator):
        result[i] = clamp(x, min, max)
    return result
```

### 5.2 fill 算法（非连续布局支持）

```
fill(tensor, value):
    for each mutable reference elem in tensor (via iter_mut):
        *elem = value.clone()
```

关键点：`iter_mut()` 已经正确处理非连续布局的步长跳转，因此 `fill` 天然支持非连续内存。

### 5.3 to_contiguous 路径选择

```
to_contiguous(tensor):
    if is_f_contiguous(tensor):
        return to_owned(tensor)        // O(n) copy, layout unchanged
    else if is_c_contiguous(tensor):
        return to_c_contiguous(tensor) // O(n) copy, preserve C-order
    else:
        return to_f_contiguous(tensor) // O(n) copy, convert to F-order
```

### 5.4 NaN 处理语义

| clip 场景 | 输入 | min | max | 输出 | 说明 |
|-----------|------|-----|-----|------|------|
| 正常范围 | `0.5` | `0.0` | `1.0` | `0.5` | 在范围内，不变 |
| 低于下界 | `-1.0` | `0.0` | `1.0` | `0.0` | 钳位到 min |
| 高于上界 | `2.0` | `0.0` | `1.0` | `1.0` | 钳位到 max |
| NaN 输入 | `NaN` | `0.0` | `1.0` | `NaN` | NaN 不满足 `< min` 也不满足 `> max`，保持 NaN |
| NaN 下界 | `0.5` | `NaN` | `1.0` | `0.5` | `NaN < 0.5` 为 false，不触发 |
| NaN 上界 | `0.5` | `0.0` | `NaN` | `0.5` | `0.5 > NaN` 为 false，不触发 |
| NaN 双界 | `0.5` | `NaN` | `NaN` | `0.5` | 均不触发 |

> **设计决策：** NaN 的 clip 行为遵循 IEEE 754 比较语义：`NaN < x` 和 `NaN > x` 均为 false，
> 因此 NaN 值在 clip 中保持不变。这与 NumPy 的 `np.clip` 行为一致。

---

## 6. 实现任务拆分

### Wave 1: 基础操作

- [ ] **T1**: 实现 `fill` 方法
  - 文件: `src/ops/utility.rs`
  - 内容: `fill(&mut self, value: A)` 方法，通过 `iter_mut()` 原地填充
  - 测试: `test_fill_basic`, `test_fill_non_contiguous`
  - 前置: tensor 模块、iter 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `clip` 方法
  - 文件: `src/ops/utility.rs`
  - 内容: `clip(&self, min: A, max: A) -> Tensor<A, D>` 和 `clip_inplace` 方法
  - 测试: `test_clip_basic`, `test_clip_nan`, `test_clip_integers`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 连续性保证

- [ ] **T3**: 实现 `to_contiguous` 方法
  - 文件: `src/ops/utility.rs`
  - 内容: 基于 `is_f_contiguous()`/`is_c_contiguous()` 检查，调用对应的连续性转换方法
  - 测试: `test_to_contiguous_f_order`, `test_to_contiguous_c_order`, `test_to_contiguous_non_contiguous`
  - 前置: T2, layout 模块的 `is_f_contiguous`/`is_c_contiguous` 完成
  - 预计: 10 min

- [ ] **T4**: 编写综合测试
  - 文件: `tests/utility.rs`
  - 内容: 边界测试（空数组、单元素、大数组、非连续布局）
  - 测试: `test_clip_empty`, `test_clip_single_element`, `test_fill_zero_dim`
  - 前置: T1, T2, T3
  - 预计: 15 min

### 并行执行分组图

```
Wave 1: [T1] [T2]
           │
           ▼
Wave 2: [T3] [T4]
```

---

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_clip_basic` | 基本裁剪：元素限制在 [0, 2] 范围 | 高 |
| `test_clip_no_change` | 所有元素在范围内，无变化 | 高 |
| `test_clip_nan` | NaN 输入保持 NaN | 高 |
| `test_clip_inplace` | 原地裁剪正确性 | 高 |
| `test_clip_integers` | i32/i64 整数裁剪 | 中 |
| `test_fill_basic` | 基本填充所有元素为指定值 | 高 |
| `test_fill_non_contiguous` | 非连续布局正确填充所有逻辑元素 | 高 |
| `test_fill_empty` | 空数组 fill 不 panic | 中 |
| `test_to_contiguous_f_order` | F-order 连续输入返回 owned 拷贝 | 高 |
| `test_to_contiguous_c_order` | C-order 连续输入返回 C-order owned | 高 |
| `test_to_contiguous_non_contiguous` | 非连续输入返回 F-order owned | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `clip`/`fill`/`to_contiguous` 均正常处理，无 panic |
| 单元素 `shape=[1]` | `clip` 正确裁剪单个元素 |
| 零维张量 | `clip` 返回标量裁剪结果 |
| 非连续切片 | `fill` 通过迭代器正确填充所有逻辑元素 |
| NaN 边界 | `clip(NaN, 0.0, 1.0)` 保持 NaN |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `clip(min, max)` 结果的每个元素 ∈ [min, max] | 随机张量 + 随机 min/max |
| `fill(v)` 后 `iter().all(|x| *x == v)` | 随机形状 + 随机值 |
| `to_contiguous()` 返回的张量 `is_f_contiguous() ∨ is_c_contiguous()` | 随机非连续布局 |

---

## 8. 与其他模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| `fill` → `iter` | 依赖 | 通过 `iter_mut()` 遍历元素 |
| `clip` → `iter` | 依赖 | 通过 `iter()` 读取、写入新张量 |
| `to_contiguous` → `layout` | 依赖 | 查询连续性状态 |
| `to_contiguous` → `convert` | 依赖 | 调用 `to_owned()`/`to_f_contiguous()`/`to_c_contiguous()` |

---

## 9. 设计决策记录（ADR）

### 决策 1：NaN 的 clip 行为

| 属性 | 值 |
|------|-----|
| 决策 | NaN 在 clip 中保持不变（不钳位） |
| 理由 | 遵循 IEEE 754 比较语义（`NaN < x` = false, `NaN > x` = false），与 NumPy `np.clip` 行为一致 |
| 替代方案 | NaN 裁剪到 min — 放弃，与 IEEE 754 和 NumPy 不一致 |
| 替代方案 | NaN 裁剪到 max — 放弃，同上 |

### 决策 2：to_contiguous 返回类型

| 属性 | 值 |
|------|-----|
| 决策 | 返回 `Tensor<A, D>`（Owned），不使用 `Cow` |
| 理由 | API 简洁（无生命周期参数）、调用方可预测行为、与 ndarray 设计一致；调用方可通过 `is_f_contiguous()` 先检查避免不必要拷贝 |
| 替代方案 | 返回 `Cow<TensorBase<S, D>>` — 放弃，引入生命周期复杂度，调用方难以处理 |
| 替代方案 | 已连续时返回视图（借引用） — 放弃，返回类型不确定，违反直觉 |

---

## 10. 性能考量

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| `clip` | O(n) | O(n) | 新分配一个张量 |
| `clip_inplace` | O(n) | O(1) | 原地修改，零额外分配 |
| `fill` | O(n) | O(1) | 原地修改，`Clone` 开销取决于类型 |
| `to_contiguous`（已连续） | O(n) | O(n) | 拷贝到新 owned |
| `to_contiguous`（非连续） | O(n) | O(n) | 拷贝 + 重新排列 |

**优化提示**：

- 连续布局的 `fill` 可用 `ptr::write_bytes` 优化（仅限 `Copy` 类型）
- `clip` 的热点路径可考虑 SIMD 加速（参见 `08-simd-backend.md`）

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
