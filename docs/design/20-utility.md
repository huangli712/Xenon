# 实用操作模块设计

> 文档编号: 20 | 模块: `src/util/` | 阶段: Phase 4
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
| 类型安全 | `clip` 限制为有序标量类型（`i32`、`i64`、`f32`、`f64`），编译期拒绝 `bool`、`Complex` 和 `usize` |
| 语义清晰 | `to_contiguous` 返回 `Tensor<A, D>`，调用方可预测生命周期 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: broadcast, iter, ffi
L6: util  ← 当前模块（依赖 tensor, dimension, storage, layout, iter）
```

---

## 2. 文件位置

```
src/
└── util/
    ├── mod.rs           # 模块根，re-exports
    ├── clip.rs          # clip / clip_inplace（范围裁剪）
    ├── fill.rs          # fill（原地填充）
    └── contiguous.rs    # to_contiguous（连续性保证）
```

多文件设计：三个操作（clip、fill、to_contiguous）按职责分离，通过 `mod.rs` 统一 re-export。

> **注意**：`to_contiguous()` 的公共 API 与语义边界都属于 `util` 模块。若实现上复用内部连续化路径，也只把它视为 `util` 的内部实现细节，不再把连续性保证语义归到 `convert`。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/util/
├── crate::tensor        # TensorBase<S, D>, Tensor, 类型别名
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut trait
├── crate::element       # Element, RealScalar trait
├── crate::layout        # is_f_contiguous 查询
└── crate::iter          # Elements 迭代器（fill/clip 内部使用）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()`（参见 `07-tensor.md` §4） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md` §4） |
| `storage` | `Storage<Elem=A>`, `StorageMut<Elem=A>`（参见 `05-storage.md` §4） |
| `element` | `Element`，以及 utility 层定义的 operation-specific `ClipElement` 约束 |
| `layout` | `is_f_contiguous()`（参见 `06-memory.md` §4） |
| `iter` | `iter()`, `iter_mut()`（参见 `10-iterator.md` §4） |
| `tensor` | `Tensor<A, D>` 的结果构造路径 | `clip` 分配新的 owned 结果张量并通过 `iter()` / `iter_mut()` 写入 |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `util` 仅消费 `tensor`、`iter` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 clip 操作

```rust
pub trait ClipElement: Element + PartialOrd {}

impl ClipElement for i32 {}
impl ClipElement for i64 {}
impl ClipElement for f32 {}
impl ClipElement for f64 {}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: ClipElement,
{
    /// Clamp each element to the [min, max] range.
    ///
    /// Returns a new tensor; the original tensor is unchanged.
    ///
    /// # Supported Types
    ///
    /// Available for types implementing `ClipElement`: i32, i64, f32, f64.
    /// **Not available for `Complex<f32>`/`Complex<f64>`** because complex numbers
    /// have no natural total ordering (`Complex` does not implement `PartialOrd`,
    /// see `04-complex.md §4`).
/// **Not available for `bool` / `Complex<_>`** because clip requires an ordered scalar domain.
    /// (see `03-element.md §3`).
    ///
    /// # Arguments
    ///
    /// * `min` - lower bound
    /// * `max` - upper bound
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidArgument)` when `min > max`.
    /// For floating-point tensors, `min`/`max` must not be `NaN`.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])?;
    /// let clipped = t.clip(0.0, 2.0)?;
    /// assert_eq!(clipped.to_vec(), vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    /// ```
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError>
    where
        A: Clone,
    {
        if min > max {
            return Err(XenonError::InvalidArgument { message: "clip requires min <= max" });
        }
        let mut out = Tensor::zeros(self.raw_dim());
        for (src, dst) in self.iter().zip(out.iter_mut()) {
            *dst = if *src < min {
                min.clone()
            } else if *src > max {
                max.clone()
            } else {
                src.clone()
            };
        }
        Ok(out)
    }

    /// Clip in place.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut t = Tensor1::from_shape_vec([3], vec![-1.0, 0.5, 3.0])?;
    /// t.clip_inplace(0.0, 2.0)?;
    /// assert_eq!(t.to_vec(), vec![0.0, 0.5, 2.0]);
    /// ```
    pub fn clip_inplace(&mut self, min: A, max: A) -> Result<(), XenonError>
    where
        S: StorageMut<Elem = A>,
        A: Clone,
    {
        if min > max {
            return Err(XenonError::InvalidArgument { message: "clip requires min <= max" });
        }
        for elem in self.iter_mut() {
            if *elem < min {
                *elem = min.clone();
            } else if *elem > max {
                *elem = max.clone();
            }
        }
        Ok(())
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
    /// **Note**: broadcast views are read-only in Xenon. A tensor with zero strides
    /// produced by broadcasting cannot satisfy `StorageMut`, so `fill()` is not
    /// available on broadcast results.
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

> `to_contiguous()` 是本模块（20-utility.md §4）定义的公共 API。内部可复用连续化实现，但这不构成 `convert` 模块的独立公共能力。
>
> **依赖说明**: `to_contiguous()` 由 utility 模块暴露；若非连续路径需要额外实现步骤，也仅属于 utility 的内部细节。类型转换语义仍归 convert，连续性保证语义仍归 utility。

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Ensure data is stored contiguously in memory (always F-order).
    ///
    /// - If already F-contiguous, returns `to_owned()` (copy, layout preserved)
    /// - Otherwise (non-contiguous, e.g. transposed views), copies into F-contiguous layout
    ///
    /// Xenon only supports F-order (see requirement §7).
    /// `to_contiguous()` always produces F-order output.
    ///
    /// # Returns
    ///
    /// Always returns an owned `Tensor<A, D>` with F-contiguous layout.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = Tensor2::<f64>::zeros([3, 4]);
    /// let contig = t.to_contiguous();
    /// assert!(contig.is_f_contiguous());
    ///
    /// // Even transposed views become F-contiguous
    /// let transposed = t.t();
    /// let contig2 = transposed.to_contiguous();
    /// assert!(contig2.is_f_contiguous());
    /// ```
    pub fn to_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            self.to_owned()
        } else {
            util_internal_to_f_contiguous(self)
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
let t = Tensor1::from_shape_vec([1000], data)?;
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
    for each (src, dst) pair via iter()/iter_mut():
        *dst = clamp(*src, min, max)
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
        return to_owned(tensor)        // O(n) copy, layout unchanged (already F-order)
    else:
        return util_internal_to_f_contiguous(tensor)  // O(n) copy, always convert to F-order
        // Non-contiguous inputs (e.g. transposed or sliced views) are
        // converted to F-order. Xenon only supports F-order.
```

### 5.4 NaN 处理语义

| clip 场景 | 输入 | min | max | 输出 | 说明 |
|-----------|------|-----|-----|------|------|
| 正常范围 | `0.5` | `0.0` | `1.0` | `0.5` | 在范围内，不变 |
| 低于下界 | `-1.0` | `0.0` | `1.0` | `0.0` | 钳位到 min |
| 高于上界 | `2.0` | `0.0` | `1.0` | `1.0` | 钳位到 max |
| NaN 输入 | `NaN` | `0.0` | `1.0` | `NaN` | NaN 不满足 `< min` 也不满足 `> max`，保持 NaN |

> 对浮点数，NaN 的 clip 行为遵循 IEEE 754 比较语义：`NaN < x` 和 `NaN > x` 均为 false，
> 因此 NaN 值在 clip 中保持不变。这与 NumPy 的 `np.clip` 行为一致。
> 另一方面，`min`/`max` 作为边界参数必须是已定义的可比较标量值；若任一边界为 `NaN`，则返回 `InvalidArgument`，避免把无效边界静默当成合法区间。

---

## 6. 实现任务拆分

### Wave 1: 基础操作

- [ ] **T1**: 实现 `fill` 方法
  - 文件: `src/util/fill.rs`
  - 内容: `fill(&mut self, value: A)` 方法，通过 `iter_mut()` 原地填充
  - 测试: `test_fill_basic`, `test_fill_non_contiguous`
  - 前置: tensor 模块、iter 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `clip` 方法
  - 文件: `src/util/clip.rs`
  - 内容: `clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError>` 和 `clip_inplace` 方法
  - 测试: `test_clip_basic`, `test_clip_nan`, `test_clip_nan_bound`, `test_clip_integers`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 连续性保证

- [ ] **T3**: 实现 `to_contiguous` 方法
  - 文件: `src/util/contiguous.rs`
  - 内容: 基于 `is_f_contiguous()` 检查，非 F-contiguous 输入始终转为 F-order
  - 测试: `test_to_contiguous_f_order`, `test_to_contiguous_transposed_becomes_f`, `test_to_contiguous_non_contiguous`
  - 前置: T2, layout 模块的 `is_f_contiguous` 完成
  - 预计: 10 min

- [ ] **T4**: 编写综合测试
  - 文件: `tests/test_utility.rs`
  - 内容: 边界测试（空数组、单元素、大数组、非连续布局）
  - 测试: `test_clip_empty`, `test_clip_single_element`, `test_clip_non_contiguous`, `test_clip_inplace_non_contiguous`, `test_fill_zero_dim`
  - 前置: T1, T2, T3
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
```

---

## 7. 测试计划

### 7.1 测试分类表

| 测试分类 | 位置 | 说明 |
|----------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `clip`、`fill` 和 `to_contiguous` 的核心语义 |
| 集成测试 | `tests/` | 验证 `utility` 与 `tensor`、`iter`、`layout`、`convert` 的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖空数组、零维张量、NaN 和非连续布局等边界 |

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_clip_basic` | 基本裁剪：元素限制在 [0, 2] 范围 | 高 |
| `test_clip_no_change` | 所有元素在范围内，无变化 | 高 |
| `test_clip_nan` | NaN 输入保持 NaN | 高 |
| `test_clip_nan_bound` | NaN 作为 min/max 返回 `InvalidArgument` | 高 |
| `test_clip_inplace` | 原地裁剪正确性 | 高 |
| `test_clip_integers` | i32/i64 整数裁剪 | 中 |
| `test_clip_non_contiguous` | 非连续布局返回正确裁剪结果 | 高 |
| `test_clip_inplace_non_contiguous` | 非连续布局原地裁剪所有逻辑元素 | 高 |
| `test_fill_basic` | 基本填充所有元素为指定值 | 高 |
| `test_fill_non_contiguous` | 非连续布局正确填充所有逻辑元素 | 高 |
| `test_fill_empty` | 空数组 fill 不 panic | 中 |
| `test_to_contiguous_f_order` | F-order 连续输入返回 owned 拷贝 | 高 |
| `test_to_contiguous_transposed_becomes_f` | 转置视图转为 F-order owned | 高 |
| `test_to_contiguous_non_contiguous` | 非连续输入返回 F-order owned | 高 |

### 7.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `shape=[0, 3]` | `clip`/`fill`/`to_contiguous` 均正常处理，无 panic |
| 单元素 `shape=[1]` | `clip` 正确裁剪单个元素 |
| 零维张量 | `clip` 返回标量裁剪结果 |
| 非连续切片 | `fill`/`clip` 通过迭代器正确处理所有逻辑元素 |
| NaN 边界 | `clip(x, NaN, 1.0)` 或 `clip(x, 0.0, NaN)` 返回 `InvalidArgument` |

### 7.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `clip(min, max)` 结果的每个元素 ∈ [min, max] | 随机张量 + 随机 min/max |
| `fill(v)` 后 `iter().all(|x| *x == v)` | 随机形状 + 随机值 |
| `to_contiguous()` 返回的张量 `is_f_contiguous() == true` | 随机非连续布局 |

### 7.5 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_utility.rs` | `clip`/`fill`/`to_contiguous` 与 `tensor`、`iter`、`layout`、`convert` 的协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
|------|----------|-----------|------|
| `utility → iter` | `iter` | `iter_mut()` | `fill` 通过可变迭代器遍历逻辑元素，参见 `10-iterator.md` §4.1 |
| `utility → iter` | `iter` | `iter()` | `clip` 通过只读迭代器读取并写入新张量，参见 `10-iterator.md` §4.1 |
| `utility → layout` | `layout` | 连续性查询 | `to_contiguous` 先查询当前布局是否已经连续，参见 `06-memory.md` §4 |
| `utility → tensor` | `tensor` | `to_owned()` / owned 构造路径 | `to_contiguous` 复用张量 owned 化与连续化路径；`clip` 通过 owned 结果张量构造返回新值 |

### 8.2 数据流描述

```text
用户调用 fill() / clip() / to_contiguous()
    │
    ├── utility 模块先判断是原地修改、生成新 tensor，还是仅做连续化
    ├── fill / clip 通过 iter / iter_mut 访问逻辑元素
    ├── to_contiguous 先查询 layout 连续性，再按需走 utility 内部连续化路径
    └── 最终返回修改后的原张量或新的 owned F-order 张量
```

---

## 9. 设计决策记录

### ADR-1：NaN 的 clip 行为

| 属性 | 值 |
|------|-----|
| 决策 | NaN 在 clip 中保持不变（不钳位） |
| 理由 | 遵循 IEEE 754 比较语义（`NaN < x` = false, `NaN > x` = false），与 NumPy `np.clip` 行为一致 |
| 替代方案 | NaN 裁剪到 min — 放弃，与 IEEE 754 和 NumPy 不一致 |
| 替代方案 | NaN 裁剪到 max — 放弃，同上 |

### ADR-2：to_contiguous 返回类型

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

- 连续布局的 `fill` 仅在填充值是全零 bit-pattern 时才可使用 `ptr::write_bytes(0)` 优化；一般情况仍应逐元素写入，避免把任意 `Copy` 值错误地按字节复制
- `clip` 的热点路径可考虑 SIMD 加速（参见 `08-simd.md` §4）

---

## 11. no_std 兼容性

实用操作模块在 `no_std` 环境下可用。原地操作（`fill`、`clip_inplace`）无堆分配；`clip` 和 `to_contiguous` 返回新张量需 `alloc`。迭代器的 `no_std` 兼容性参见 `10-iterator.md` §11。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `fill` | ✅ | 原地操作，通过 `iter_mut()` 遍历，无堆分配 |
| `clip` | ✅ | 返回新 `Tensor`，需 `no_std + alloc` |
| `clip_inplace` | ✅ | 原地修改，无额外分配 |
| `to_contiguous` | ✅ | 返回新 `Tensor`，需 `no_std + alloc` |

条件编译处理：

```rust
// fill / clip_inplace: in-place via iter_mut — pure no_std
// clip: returns new Tensor → needs alloc::vec::Vec
// to_contiguous: returns owned Tensor → needs alloc::vec::Vec

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
