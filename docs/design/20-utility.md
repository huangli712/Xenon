# 实用操作模块设计

> 文档编号: 20 | 模块: `src/util/` | 阶段: Phase 4
> 前置文档: `05-storage.md`, `06-layout.md`, `07-tensor.md`, `10-iterator.md`
> 需求参考: 需求说明书 §21, §22
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                     | 不包含                                            |
| -------------- | ---------------------------------------- | ------------------------------------------------- |
| 范围裁剪       | `clip`（将元素限制在 [min, max] 范围内） | 其他 numpy 风格变换（flip/roll/shift），以及不再作为稳定公共 API 的 `clip_inplace` |
| 填充操作       | `fill`（原地填充所有逻辑元素）           | 构造方法（zeros/ones/full，由 construct.rs 提供） |
| 连续性保证     | `to_contiguous`（确保内存连续存储）      | 布局计算逻辑（由 layout 模块提供）                |
| 非连续布局支持 | 通过迭代器正确处理非连续内存             | 布局优化策略                                      |

### 1.2 设计原则

| 原则     | 体现                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------ |
| 步长感知 | `fill`/`clip` 通过迭代器正确处理非连续内存布局                                                   |
| 原地优先 | `fill` 为原地操作（`&mut self`），避免额外分配                                                   |
| 类型安全 | `clip` 限制为有序标量类型（`i32`、`i64`、`f32`、`f64`），编译期拒绝 `bool`、`Complex` 和 `usize` |
| 语义清晰 | `to_contiguous` 返回 `Tensor<A, D>`，调用方可预测生命周期                                        |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent from layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: broadcast, iter, ffi
L6: util  <- current module (depends on tensor, dimension, storage, layout, iter)
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §21, §22 |
| 范围内   | `clip`、`fill`、`to_contiguous` / `into_contiguous`。 |
| 范围外   | sort、argsort、searchsorted，以及除 clip / fill / contiguous 之外的其他 utility 操作。 |
| 非目标   | 不把 `util` 扩展为通用算法杂项集合，不新增第三方依赖，也不重定义 convert / layout 的职责。 |

---

## 3. 文件位置

```
src/
└── util/
    ├── mod.rs           # Module root, re-exports
    ├── clip.rs          # clip (range clamping) and internal clamp helpers
    ├── fill.rs          # fill (in-place fill)
    └── contiguous.rs    # to_contiguous (contiguity guarantee)
```

多文件设计：三个操作（clip、fill、to_contiguous）按职责分离，通过 `mod.rs` 统一 re-export。

> **注意**：`to_contiguous()` 的公共 API 与语义边界都属于 `util` 模块。若实现上复用内部连续化路径，也只把它视为 `util` 的内部实现细节，不再把连续性保证语义归到 `convert`。

---

## 4. 依赖关系

### 4.1 依赖图

```
src/util/
├── crate::tensor        # TensorBase<S, D>, Tensor, type aliases
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut trait
├── crate::element       # Element, RealScalar trait
├── crate::layout        # is_f_contiguous query
└── crate::iter          # Elements iterator for fill / clip internals
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                      |
| ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()`, `.storage_kind()`（参见 `07-tensor.md` §5）           |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md` §5）                                                      |
| `storage`   | `Storage<Elem=A>`, `StorageMut<Elem=A>`, `StorageIntoOwned<Elem=A>`（参见 `05-storage.md` §5）                      |
| `element`   | `Element`，以及 utility 层定义的 operation-specific `ClipElement` 约束                                              |
| `layout`    | `is_f_contiguous()`（参见 `06-layout.md` §5）                                                                       |
| `iter`      | `iter()`, `iter_mut()`（参见 `10-iterator.md` §5）                                                                  |
| `tensor`    | `Tensor<A, D>` 的结果构造路径；`clip` 分配新的 owned 结果张量并通过 `iter()` / `iter_mut()` 写入逻辑元素           |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `util` 仅消费 `tensor`、`iter` 等核心模块，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 clip 操作

````rust,ignore
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
/// see `04-complex.md §5`).
/// **Not available for `bool` / `Complex<_>`** because clip requires an ordered scalar domain.
    /// (see `03-element.md §5.3`).
    ///
    /// # Arguments
    ///
    /// * `min` - lower bound
    /// * `max` - upper bound
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidArgument)` when `min > max` or either bound is `NaN`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])?;
    /// let clipped = t.clip(0.0, 2.0)?;
    /// assert_eq!(clipped.to_vec(), vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    /// ```ignore
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError>
    where
        A: Clone,
    {
        if min.partial_cmp(&max).is_none() || min > max {
            return Err(XenonError::InvalidArgument {
                operation: "clip",
                argument: "min/max",
                expected: "min <= max; NaN bounds are invalid for floating-point inputs",
                actual: "min > max or NaN bound",
                axis: None,
                shape: Some(self.shape().to_vec()),
            });
        }
        let mut out = Tensor::uninit_like(self.raw_dim())?;
        for (src, dst) in self.iter().zip(out.iter_uninit_mut()) {
            dst.write(if *src < min {
                min.clone()
            } else if *src > max {
                max.clone()
            } else {
                src.clone()
            });
        }
        let out = unsafe { out.assume_init() };
        Ok(out)
    }
}
````

> 浮点参数非法时：`min > max` 或任一边界为 `NaN` 时返回可恢复错误。
>
> `clip` 总是返回新的 owned 张量，但本文不再把“先 `zeros()` 再逐元素覆写”写成稳定实现承诺；实现可使用 `MaybeUninit` 或等价的内部未初始化 owned 缓冲区，一次写入最终值，避免无意义的零填充后再覆写。

> **边界收缩：** `clip_inplace` 不属于 `require.md` §21.1 的强制公共接口。若实现上需要原地 clamp helper，可仅作为 `src/util/clip.rs` 的内部辅助，不纳入稳定 API 承诺与测试矩阵。

### 5.2 fill 操作

````rust,ignore
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
    /// This is the writable-layer API. Read-only and shared-readonly tensors do
    /// not satisfy `StorageMut`; higher-level convenience entry points, if
    /// exposed elsewhere in the tensor module, must reject those cases with the
    /// documented recoverable error rather than reaching this implementation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor1::<f64>::zeros([5])?;
    /// t.fill(3.14)?;
    /// assert!(t.iter().all(|&x| x == 3.14));
    /// ```
    pub fn fill(&mut self, value: A) {
        fill_storage_mut(&mut self.storage, &self.layout(), value)
    }
}
````

> **分层说明：** `fill` 的核心实现收敛在 `S: StorageMut` 的可写层，不再在单一 `impl<S: Storage>` 中混合“可写执行 + 只读报错”两种类型状态。若 `tensor` 层另外提供面向所有存储模式的统一入口，则只读或共享只读结果（例如广播结果）须在到达此 helper 前返回 `XenonError::ReadOnlyStorage`；该错误属于上层入口的公开语义，而不是 `fill_storage_mut()` 的内部分支。

#### 5.2.1 fill 的显式写入语义

- `fill` 必须按**逻辑索引**迭代，并且只写入逻辑元素。
- 对带 padding 的底层存储：不得写入任何 padding bytes。
- 对非连续但可写的视图：必须严格按 layout strides 导航到每个逻辑元素。
- 对存在零步长的布局：按照 `require.md` §16，它们来自广播只读结果；因此公开 `fill` 不应在可写层遇到这类输入，相关拒绝应作为上层只读入口的不变量维护，而不是 `fill` 自身的额外错误分支。

```
fill_logical_only(storage, layout, value):
    for logical_index in 0..layout.logical_len():
        offset = layout.offset_for_logical_index(logical_index)
        write storage[offset] = clone(value)
```

> 上述伪代码强调的是契约，而不是公开 API：实现可以使用递归多维索引、stride-aware iterator 或其他等价内部辅助函数，但结果必须等价于“按逻辑索引逐元素写入，且不触碰 padding / 非逻辑区域”。`ReadOnlyStorage` 拒绝分支与广播零步长输入的屏蔽，属于到达该 helper 之前的上层职责。

### 5.3 连续性保证（to_contiguous）

> `to_contiguous()` 是本模块（20-utility.md §4）定义的公共 API。内部可复用连续化实现，但这不构成 `convert` 模块的独立公共能力。
>
> **依赖说明**: `to_contiguous()` 由 utility 模块暴露；若非连续路径需要额外实现步骤，也仅属于 utility 的内部细节。类型转换语义仍归 convert，连续性保证语义仍归 utility。

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Ensure data is stored contiguously in memory (always F-order).
    ///
    /// - `to_contiguous(&self)` always returns a fresh owned tensor
    /// - `into_contiguous(self)` may reuse the existing owned allocation when already F-contiguous
    /// - Non-contiguous inputs are re-packed into F-contiguous layout
    ///
    /// Xenon only supports F-order (see requirement §7).
    /// `to_contiguous()` always produces F-order output.
    ///
    /// # Returns
    ///
    /// Always returns an independent owned `Tensor<A, D>` with F-contiguous layout.
    ///
    /// # Examples
    ///
    /// ```
    /// let t = Tensor2::<f64>::zeros([3, 4])?;
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

    /// Consume the tensor and ensure F-contiguous owned storage.
    ///
    /// Reuses the existing owned data when the input is already F-contiguous;
    /// otherwise materializes a new contiguous tensor.
    pub fn into_contiguous(self) -> Tensor<A, D>
    where
        S: StorageIntoOwned<Elem = A>,
    {
        if self.is_f_contiguous() {
            Tensor {
                storage: self.storage.into_owned(),
                shape: self.shape,
                strides: self.strides,
                offset: self.offset,
                flags: self.flags,
            }
        } else {
            util_internal_to_f_contiguous(&self)
        }
    }
}
````

> **设计说明：** `to_contiguous(&self)` 是稳定的“总是返回独立 owned 结果”入口；当输入已是连续 F-order 时，它不得改变逻辑值，且可以复用现有数据作为读取来源，但因为返回值必须与借用源解除别名，所以仍会物化为新的 owned 张量。
> `into_contiguous(self)` 是满足 `require.md` §22 的消费式入口：当输入已经是连续 F-order 且具备 owned 化前提时，可复用现有数据，否则退化为重新打包。

### 5.4 Good / Bad 对比

```rust,ignore
// Good - use fill for in-place filling, zero extra allocation
let mut t = Tensor1::<f64>::zeros([1000])?;
t.fill(42.0)?;

// Bad - create a temporary Vec then construct a new tensor, double allocation
let data = vec![42.0; 1000];
let t = Tensor1::from_shape_vec([1000], data)?;
```

```rust,ignore
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

## 6. 内部实现设计

### 6.1 clip 算法

```
clip(tensor, min, max):
    allocate uninitialized owned result with same shape
    for each (src, dst) pair via iter()/iter_uninit_mut():
        dst.write(clamp(*src, min, max))
    mark result initialized
    return result
```

### 6.2 fill 算法（非连续布局支持）

```
fill(tensor, value):
    for each logical index in layout order:
        offset = offset_for_logical_index(layout, logical index)
        write storage[offset] = clone(value)
```

关键点：utility 层的核心 `fill` helper 只接收可写存储，因此不再引入 `InvalidLayout` 这种对广播零步长布局的公开错误分支；广播结果与其他只读结果的拒绝由上层入口负责。可写存储仍必须遵守“只写逻辑元素”的契约：连续布局可走快路径，带 padding / 非连续布局必须按逻辑索引与 strides 写入，且不得触碰 padding bytes。

### 6.3 to_contiguous 路径选择

```
to_contiguous(tensor):
    if is_f_contiguous(tensor):
        return to_owned(tensor)        // O(n) copy, layout unchanged (already F-order)
    else:
        return util_internal_to_f_contiguous(tensor)  // O(n) copy, always convert to F-order
        // Non-contiguous inputs (e.g. transposed or sliced views) are
        // converted to F-order. Xenon only supports F-order.

into_contiguous(tensor):
    if is_f_contiguous(tensor):
        return reuse_owned_storage(tensor)  // O(1) when storage is already owned/F-order
    else:
        return util_internal_to_f_contiguous(&tensor)
```

### 6.4 NaN 处理语义

| clip 场景 | 输入   | min   | max   | 输出  | 说明                                          |
| --------- | ------ | ----- | ----- | ----- | --------------------------------------------- |
| 正常范围  | `0.5`  | `0.0` | `1.0` | `0.5` | 在范围内，不变                                |
| 低于下界  | `-1.0` | `0.0` | `1.0` | `0.0` | 钳位到 min                                    |
| 高于上界  | `2.0`  | `0.0` | `1.0` | `1.0` | 钳位到 max                                    |
| NaN 输入  | `NaN`  | `0.0` | `1.0` | `NaN` | NaN 不满足 `< min` 也不满足 `> max`，保持 NaN |

> 对浮点数，NaN 的 clip 行为遵循 IEEE 754 比较语义：`NaN < x` 和 `NaN > x` 均为 false，
> 因此 NaN 值在 clip 中保持不变。这与 NumPy 的 `np.clip` 行为一致。
> 另一方面，`min`/`max` 作为边界参数必须是已定义的可比较标量值；若任一边界为 `NaN`，则返回 `InvalidArgument`，避免把无效边界静默当成合法区间。

---

## 7. 实现任务拆分

### Wave 1: 基础操作

- [ ] **T1**: 实现 `fill` 方法
  - 文件: `src/util/fill.rs`
  - 内容: 在 `S: StorageMut` 层实现 `fill(&mut self, value: A)` 核心 helper；若上层需要统一入口，则由上层负责把只读与共享只读场景映射为 `XenonError::ReadOnlyStorage`
  - 测试: `test_fill_basic`, `test_fill_non_contiguous`, `test_fill_readonly_storage_error`, `test_fill_broadcast_view_error`, `test_fill_transposed_view_error`, `test_fill_padded_writes_logical_only`
  - 前置: tensor 模块、iter 模块完成
  - 预计: 10 min

- [ ] **T2**: 实现 `clip` 方法
  - 文件: `src/util/clip.rs`
  - 内容: `clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError>`；内部可复用非公开 clamp helper
  - 测试: `test_clip_basic`, `test_clip_nan`, `test_clip_nan_bound`, `test_clip_integers`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 连续性保证

- [ ] **T3**: 实现 `to_contiguous` 方法
  - 文件: `src/util/contiguous.rs`
  - 内容: 实现 `to_contiguous(&self)` 与 `into_contiguous(self)`；非 F-contiguous 输入始终转为 F-order，连续 owned 输入允许复用数据
  - 测试: `test_to_contiguous_f_order`, `test_into_contiguous_reuses_owned_data`, `test_to_contiguous_transposed_becomes_f`, `test_to_contiguous_non_contiguous`
  - 前置: T2, layout 模块的 `is_f_contiguous` 完成
  - 预计: 10 min

- [ ] **T4**: 编写综合测试
  - 文件: `tests/test_utility.rs`
  - 内容: 边界测试（空数组、单元素、大数组、非连续布局）
  - 测试: `test_clip_empty`, `test_clip_single_element`, `test_clip_non_contiguous`, `test_fill_zero_dim`
  - 前置: T1, T2, T3
  - 预计: 10 min

### 并行执行分组图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                               |
| -------- | ------------------------ | ------------------------------------------------------------------ |
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `clip`、`fill` 和 `to_contiguous` 的核心语义                  |
| 集成测试 | `tests/`                 | 验证 `utility` 与 `tensor`、`iter`、`layout` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空数组、零维张量、NaN 和非连续布局等边界                       |

### 8.2 单元测试清单

| 测试函数                                  | 测试内容                                | 优先级 |
| ----------------------------------------- | --------------------------------------- | ------ |
| `test_clip_basic`                         | 基本裁剪：元素限制在 [0, 2] 范围        | 高     |
| `test_clip_no_change`                     | 所有元素在范围内，无变化                | 高     |
| `test_clip_nan`                           | NaN 输入保持 NaN                        | 高     |
| `test_clip_nan_bound`                     | NaN 作为 min/max 返回 `InvalidArgument` | 高     |
| `test_clip_integers`                      | i32/i64 整数裁剪                        | 中     |
| `test_clip_non_contiguous`                | 非连续布局返回正确裁剪结果              | 高     |
| `test_fill_basic`                         | 基本填充所有元素为指定值                | 高     |
| `test_fill_non_contiguous`                | 非连续布局正确填充所有逻辑元素          | 高     |
| `test_fill_readonly_storage_error`        | 只读存储返回 `ReadOnlyStorage`          | 高     |
| `test_fill_broadcast_view_error`          | 对 broadcast 结果执行 fill 返回可恢复错误 | 高   |
| `test_fill_transposed_view_error`         | 对只读转置视图执行 fill 返回可恢复错误  | 高     |
| `test_fill_padded_writes_logical_only`    | 带 padding 的可写张量仅覆写逻辑元素     | 高     |
| `test_fill_empty`                         | 空数组 fill 不 panic                    | 中     |
| `test_to_contiguous_f_order`              | F-order 连续输入返回 owned 拷贝         | 高     |
| `test_into_contiguous_reuses_owned_data`  | F-order owned 输入消费后复用原数据      | 高     |
| `test_to_contiguous_transposed_becomes_f` | 转置视图转为 F-order owned              | 高     |
| `test_to_contiguous_non_contiguous`       | 非连续输入返回 F-order owned            | 高     |

### 8.3 边界测试场景

| 场景                  | 预期行为                                                          |
| --------------------- | ----------------------------------------------------------------- |
| 空数组 `shape=[0, 3]` | `clip`/`fill`/`to_contiguous` 均正常处理，无 panic                |
| 单元素 `shape=[1]`    | `clip` 正确裁剪单个元素                                           |
| 零维张量              | `clip` 返回标量裁剪结果                                           |
| 非连续切片            | `fill`/`clip` 通过迭代器正确处理所有逻辑元素                      |
| broadcast 只读视图    | `fill` 返回可恢复错误，不允许对零步长别名布局写入                 |
| 只读转置视图          | `fill` 返回 `ReadOnlyStorage`，不修改底层数据                     |
| 带 padding 的可写布局 | `fill` 只修改逻辑元素，对 padding bytes 保持不变                  |
| NaN 边界              | `clip(x, NaN, 1.0)` 或 `clip(x, 0.0, NaN)` 返回 `InvalidArgument` |

### 8.4 属性测试不变量

| 不变量                                                                         | 测试方法                |
| ------------------------------------------------------------------------------ | ----------------------- |
| `clip(min, max)` 结果的每个元素 ∈ [min, max]                                   | 随机张量 + 随机 min/max |
| `fill(v)` 后 `iter().all(\|x\| *x == v)`                                      | 随机形状 + 随机值       |
| `to_contiguous()` / `into_contiguous()` 返回的张量 `is_f_contiguous() == true` | 随机非连续布局          |

### 8.5 集成测试

| 测试文件                | 测试内容                                                                 |
| ----------------------- | ------------------------------------------------------------------------ |
| `tests/test_utility.rs` | `clip`/`fill`/`to_contiguous` 与 `tensor`、`iter`、`layout` 的协同路径 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `clip` / `fill` / `to_contiguous` 在默认构建下保持错误分层与 F-order 输出语义。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `clip` 仅对 `ClipElement` 开放，拒绝 `bool` / `Complex` | 编译期测试。 |
| `into_contiguous(self)` 仅对支持 owned 转换的存储模式开放 | 编译期测试。 |
| sort / argsort / searchsorted 不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向               | 对方模块 | 接口/类型                                      | 约定                                                                                                       |
| ------------------ | -------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `utility → iter`   | `iter`   | `iter_mut()`                                   | `fill` 通过可变迭代器遍历逻辑元素，参见 `10-iterator.md` §5.6                                              |
| `utility → iter`   | `iter`   | `iter()`                                       | `clip` 通过只读迭代器读取并写入新张量，参见 `10-iterator.md` §5.6                                          |
| `utility → layout` | `layout` | 连续性查询                                     | `to_contiguous` 先查询当前布局是否已经连续，参见 `06-layout.md` §5.4                                       |
| `utility → tensor` | `tensor` | `to_owned()` / `into_owned()` / owned 构造路径 | `to_contiguous` 与 `into_contiguous` 复用张量 owned 化与连续化路径；`clip` 通过 owned 结果张量构造返回新值；跨文档连续化归属统一在 utility |

### 9.2 数据流描述

```text
User calls fill() / clip() / to_contiguous() / into_contiguous()
    │
    ├── utility decides between in-place update, new tensor creation, or contiguity repair
    ├── fill / clip traverse logical elements through iter / iter_mut
    ├── to_contiguous checks layout flags before materializing F-order storage
    └── the module returns either the updated tensor or a new owned F-order tensor
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `clip` 在 `min > max` 或边界为 `NaN` 时返回 `XenonError::InvalidArgument { operation, argument, expected, actual, axis, shape }`；若上层统一入口允许对所有存储模式请求填充，则对只读存储返回 `XenonError::ReadOnlyStorage`。`XenonError` 是本模块唯一公开错误类型。 |
| Panic | 公开 utility API 不定义额外 panic 语义；连续化与裁剪失败统一走显式错误或正常返回。 |
| 路径一致性 | 连续与非连续布局都必须通过同一逻辑元素语义工作；当前无独立 SIMD / 并行分支。 |
| 容差边界 | `clip` 对浮点数遵循 IEEE 754 比较语义；不额外引入近似容差。 |

---

## 11. 设计决策记录

### ADR-1：NaN 的 clip 行为

| 属性     | 值                                                                                          |
| -------- | ------------------------------------------------------------------------------------------- |
| 决策     | NaN 在 clip 中保持不变（不钳位）                                                            |
| 理由     | 遵循 IEEE 754 比较语义（`NaN < x` = false, `NaN > x` = false），与 NumPy `np.clip` 行为一致 |
| 替代方案 | NaN 裁剪到 min — 放弃，与 IEEE 754 和 NumPy 不一致                                          |
| 替代方案 | NaN 裁剪到 max — 放弃，同上                                                                 |

### ADR-2：to_contiguous 返回类型

| 属性     | 值                                                                                                                                        |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 返回 `Tensor<A, D>`（Owned），不使用 `Cow`                                                                                                |
| 理由     | API 简洁（无生命周期参数）、调用方可预测行为、与 ndarray 设计一致；同时补充消费式 `into_contiguous(self)` 以在已连续 owned 输入上复用数据 |
| 替代方案 | 返回 `Cow<TensorBase<S, D>>` — 放弃，引入生命周期复杂度，调用方难以处理                                                                   |
| 替代方案 | 已连续时返回视图（借引用） — 放弃，返回类型不确定，违反直觉                                                                               |

---

## 12. 性能考量

| 操作                              | 时间复杂度 | 空间复杂度 | 说明                             |
| --------------------------------- | ---------- | ---------- | -------------------------------- |
| `clip`                            | O(n)       | O(n)       | 新分配一个张量                   |
| `fill`                            | O(n)       | O(1)       | 原地修改；utility 核心 helper 仅在可写层执行，`Clone` 开销取决于类型 |
| `to_contiguous`（已连续）         | O(n)       | O(n)       | 借用入口拷贝到新 owned           |
| `into_contiguous`（已连续 owned） | O(1)       | O(1)       | 直接复用现有 F-order owned 数据  |
| `to_contiguous`（非连续）         | O(n)       | O(n)       | 拷贝 + 重新排列                  |

**优化提示**：

- 连续布局的 `fill` 仅在填充值是全零 bit-pattern 时才可使用 `ptr::write_bytes(0)` 优化；一般情况仍应逐元素写入，避免把任意 `Copy` 值错误地按字节复制
- `clip` 的热点路径可考虑 SIMD 加速（参见 `08-simd.md` §5）

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                 |
| ---------- | ------------------------------------------------------------------------------------ |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径                          |
| 单 crate   | `util` 设计保持在现有 crate 内，不引入额外 crate                                     |
| SemVer     | 当前文档补充了 `into_contiguous(self)` 的复用语义，并明确 `clip` 的 NaN 边界错误语义 |
| 最小依赖   | 本模块不新增第三方依赖                                                               |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-14 |
| 1.1.4 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
