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
| 类型安全 | `clip` 限制为有序标量类型（`i32`、`i64`、`usize`、`f32`、`f64`），编译期拒绝 `bool` 和 `Complex` |
| 语义清晰 | `to_contiguous` 返回 `Tensor<A, D>`，调用方可预测生命周期 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
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

> **注意**：`to_contiguous()` 的公共 API 定义在 `src/util/contiguous.rs`（本模块），内部委托给 `src/convert/contiguous.rs` 中的 `to_f_contiguous()` 辅助函数（参见 `21-type.md §4.7`）。

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
| `element` | `Element`, `RealScalar`，以及 utility 层定义的 operation-specific `ClipElement` 约束 |
| `layout` | `is_f_contiguous()`（参见 `06-memory.md` §4） |
| `iter` | `iter()`, `iter_mut()`（参见 `10-iterator.md` §4） |
| `math` | `mapv()`（clip 内部调用，参见 `11-math.md` §4） |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `util` 仅消费 `tensor`、`iter` 等核心模块，不被它们依赖。

---

## 4. 公共 API 设计

### 4.1 clip 操作

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Numeric + PartialOrd,
{
    /// Clamp each element to the [min, max] range.
    ///
    /// Returns a new tensor; the original tensor is unchanged.
    ///
    /// # Supported Types
    ///
    /// Available for types implementing `Numeric + PartialOrd`: i32, i64, f32, f64.
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

> `to_contiguous()` 是本模块（20-utility.md §4）定义的公共 API。内部实现委托给 `src/convert/contiguous.rs` 中的 `to_f_contiguous()` 辅助函数（见 21-type.md §2.1 文件结构）。
>
> **依赖说明**: `to_contiguous()` 是 utility 模块暴露的公共 API；其非连续路径内部调用 convert 模块中的 `to_f_contiguous()` helper。类型转换语义仍归 convert，连续性保证语义归 utility。

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
        return to_owned(tensor)        // O(n) copy, layout unchanged (already F-order)
    else:
        return to_f_contiguous(tensor) // O(n) copy, always convert to F-order
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
| NaN 下界 | `0.5` | `NaN` | `1.0` | `0.5` | `NaN < 0.5` 为 false，不触发 |
| NaN 上界 | `0.5` | `0.0` | `NaN` | `0.5` | `0.5 > NaN` 为 false，不触发 |
| NaN 双界 | `0.5` | `NaN` | `NaN` | `0.5` | 均不触发 |

> **设计决策：** NaN 的 clip 行为遵循 IEEE 754 比较语义：`NaN < x` 和 `NaN > x` 均为 false，
> 因此 NaN 值在 clip 中保持不变。这与 NumPy 的 `np.clip` 行为一致。

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
  - 内容: `clip(&self, min: A, max: A) -> Tensor<A, D>` 和 `clip_inplace` 方法
  - 测试: `test_clip_basic`, `test_clip_nan`, `test_clip_integers`
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
  - 文件: `tests/utility.rs`
  - 内容: 边界测试（空数组、单元素、大数组、非连续布局）
  - 测试: `test_clip_empty`, `test_clip_single_element`, `test_fill_zero_dim`
  - 前置: T1, T2, T3
  - 预计: 15 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
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
| `test_to_contiguous_transposed_becomes_f` | 转置视图转为 F-order owned | 高 |
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
| `to_contiguous()` 返回的张量 `is_f_contiguous() == true` | 随机非连续布局 |

### 7.4 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/utility.rs` | `clip`/`fill`/`to_contiguous` 与 `tensor`、`iter`、`layout`、`convert` 的协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 交互点 | 方向 | 说明 |
|--------|------|------|
| `fill` → `iter` | 依赖 | 通过 `iter_mut()` 遍历元素（参见 `10-iterator.md` §4.1） |
| `clip` → `iter` | 依赖 | 通过 `iter()` 读取、写入新张量（参见 `10-iterator.md` §4.1） |
| `to_contiguous` → `layout` | 依赖 | 查询连续性状态（参见 `06-memory.md` §4） |
| `to_contiguous` → `convert` | 依赖 | 调用 `to_owned()`/`to_f_contiguous()`（参见 `21-type.md` §4.5 和 §4.7），始终输出 F-order；`to_f_contiguous()` 在 21 中定义，负责将非连续内存重排为 F-order 连续布局 |

### 8.2 数据流描述

```text
用户调用 fill() / clip() / to_contiguous()
    │
    ├── utility 模块先判断是原地修改、生成新 tensor，还是仅做连续化
    ├── fill / clip 通过 iter / iter_mut 访问逻辑元素
    ├── to_contiguous 先查询 layout 连续性，再按需委托 convert::to_f_contiguous()
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

- 连续布局的 `fill` 可用 `ptr::write_bytes` 优化（仅限 `Copy` 类型）
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
