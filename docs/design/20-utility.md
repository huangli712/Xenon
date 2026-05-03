# 实用操作模块设计

> 文档编号: 20
> 模块目录: src/util/
> 任务阶段: Phase 4
> 前置文档: 05-storage.md, 06-layout.md, 07-tensor.md, 10-iterator.md
> 需求参考: 需求说明书 §21、§22、§27、§28
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                      |
| -------------- | ----------------------------------------- |
| 范围裁剪       | `clip`（将元素限制在 [min, max] 范围内）  |
| 填充操作       | `fill` / `try_fill`                       |
| 连续性保证     | `to_contiguous`（确保内存连续存储）       |
| 非连续布局支持 | 通过迭代器正确处理非连续内存              |

| 职责           | 不包含                                                     |
| -------------- | ---------------------------------------------------------- |
| 范围裁剪       | 其他 numpy 风格变换（flip/roll/shift），以及`clip_inplace` |
| 填充操作       | 构造方法（zeros/ones/full，由 construct.rs 提供）          |
| 连续性保证     | 布局计算逻辑（由 layout 模块提供）                         |
| 非连续布局支持 | 布局优化策略                                               |

### 1.2 设计原则

| 原则     | 体现                                                                        |
| -------- | --------------------------------------------------------------------------- |
| 步长感知 | `fill`/`clip` 通过迭代器正确处理非连续内存布局                              |
| 原地优先 | `fill` 为原地操作（`&mut self`），避免额外分配                              |
| 类型安全 | `clip` 限制为有序标量类型（`i32`、`i64`、`f32`、`f64`），编译期拒绝其它类型 |
| 语义清晰 | `to_contiguous` 返回 `Tensor<A, D>`，调用方可预测生命周期                   |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                       |
| -------- | ------------------------------------------------------------------------------------------ |
| 需求映射 | 需求说明书 §21、§22、§27、§28                                                              |
| 范围内   | `clip`、`try_fill` / `fill`、`to_contiguous` / `into_contiguous`。                         |
| 范围外   | sort、argsort、searchsorted，以及除 clip / fill / contiguous 之外的其他 utility 操作。     |
| 非目标   | 不把 `util` 扩展为通用算法杂项集合，不新增第三方依赖，也不重定义 convert / layout 的职责。 |

---

## 3. 文件位置

```
src/util/
├── mod.rs           # Module root, re-exports
├── clip.rs          # clip (range clamping) and internal clamp helpers
├── fill.rs          # fill (in-place fill)
└── contiguous.rs    # to_contiguous (contiguity guarantee)
```

多文件设计：三个操作（clip、fill、to_contiguous）按职责分离，通过 `mod.rs` 统一 re-export。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/util/
├── crate::tensor        # TensorBase<S, D>, Tensor, type aliases
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut trait
├── crate::element       # Element, OrderedCompareElement trait
├── crate::layout        # is_f_contiguous query
├── crate::iter          # Elements iterator for fill / clip internals
└── crate::error         # XenonError (clip / try_fill recoverable errors)
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                 |
| ----------- | -------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()`；`clip` 通过 `iter()` 读取源数据并构造新的 owned 结果张量（参见 `07-tensor.md` §5） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md` §5）                   |
| `storage`   | `Storage<Elem=A>`, `StorageMut<Elem=A>`, `StorageIntoOwned<Elem=A>`（参见 `05-storage.md` §5）|
| `element`   | `Element`，`OrderedCompareElement`（clip 复用，参见 `03-element.md` §5.5）       |
| `layout`    | `is_f_contiguous()`（张量层方法参见 `07-tensor.md` §5.3，算法定义参见 `06-layout.md` §5.7） |
| `iter`      | `iter()`, `iter_mut()`（参见 `10-iterator.md` §5）                               |
| `error`     | `XenonError`、`InvalidArgumentKind::OperationSpecific`、`StorageKindTag`（参见 `26-error.md v3.0.0 §5.1`）|

### 4.3 依赖合法性

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.4 依赖方向声明

依赖方向：单向向上。`util` 仅消费 `tensor`、`iter` 等核心模块，不被它们依赖。

---

## 5. 公共 API 设计

### 5.1 clip 操作

````rust,ignore
// clip reuses OrderedCompareElement (see 03-element.md §5.5) as element bound,
// no separate ClipElement trait needed.

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: OrderedCompareElement,
{
    /// Clamp each element to the [min, max] range.
    ///
    /// Returns a new tensor; the original tensor is unchanged.
    ///
    /// # Supported Types
    ///
    /// Available for types implementing `OrderedCompareElement`: i32, i64, f32, f64.
    /// **Not available for `Complex<f32>`/`Complex<f64>`** because complex numbers
    /// have no natural total ordering (`Complex` does not implement `PartialOrd`,
    /// see `04-complex.md §5`).
    ///
    /// # Arguments
    ///
    /// * `min` - lower bound
    /// * `max` - upper bound
    ///
    /// # Errors
    ///
    /// Returns `Err(XenonError::InvalidArgument {
    ///     operation: Cow::Borrowed("clip"),
    ///     kind: InvalidArgumentKind::OperationSpecific {
    ///         argument: Cow::Borrowed("min/max"),
    ///         constraint: Cow::Borrowed("min <= max; NaN bounds are invalid"),
    ///     },
    /// })` when `min > max` or either bound is `NaN`. The `kind` variant
    /// follows `26-error.md v3.0.0 §5.1 InvalidArgumentKind` (closed enum).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])?;
    /// let clipped = t.clip(0.0, 2.0)?;
    /// assert_eq!(clipped.to_vec(), vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    /// ```
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError>
    where
        A: Clone,
    {
        if min.partial_cmp(&max).is_none() || min > max {
            return Err(XenonError::InvalidArgument {
                operation: Cow::Borrowed("clip"),
                kind: InvalidArgumentKind::OperationSpecific {
                    argument: Cow::Borrowed("min/max"),
                    constraint: Cow::Borrowed(
                        "min <= max; NaN bounds are invalid for floating-point inputs",
                    ),
                },
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

- 浮点参数非法时：`min > max` 或任一边界为 `NaN` 时返回可恢复错误。
- `clip` 总是返回新的 owned 张量，但本文不再把"先 `zeros()` 再逐元素覆写"写成稳定实现承诺；实现可使用 `MaybeUninit` 或等价的内部未初始化 owned 缓冲区，一次写入最终值，避免无意义的零填充后再覆写。
- `clip()` 的实现可能依赖内部未初始化构造能力（如 `uninit_like`、`iter_uninit_mut`、`assume_init` 或等价 helper）；这些内部 helper 不属于稳定公共 API。
- `clip_inplace` 不属于 `需求说明书 §21.1` 的强制公共接口。若实现上需要原地 clamp helper，可仅作为 `src/util/clip.rs` 的内部辅助，不纳入稳定 API 承诺与测试矩阵。
- `InvalidArgument` 的字段必须严格对齐 `26-error.md v3.0.0 §5.1` 的封闭枚举：`operation: Cow<'static, str>` + `kind: InvalidArgumentKind`。`clip` 的边界违规属于 `OperationSpecific { argument, constraint }` 子变体；不再使用旧版 `expected/actual/axis/start/end` 自由文本字段（这些字段在 v3.0.0 已被移除）。`shape` 不再作为 `InvalidArgument` 的字段携带，`InvalidArgumentKind` 变体内部按需嵌入诊断数据。

### 5.2 fill 操作

- `fill()` 是 `S: StorageMut` 层的主公开入口；`try_fill()` 是 `S: Storage` 层的次级便捷入口，运行时判定可写性。两层入口的分派规则见 §5.3。

```rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Fill all logical elements with the specified value.
    ///
    /// Secondary convenience entry point.
    ///
    /// Use this when writability is only known at runtime. Dispatch consults
    /// the optional mutable-handle capability exposed by the storage layer.
    /// Writable storage performs the fill via `fill_mut()`; read-only/shared
    /// read-only storage, or storage without that optional capability,
    /// returns `XenonError::InvalidStorageMode`.
    pub fn try_fill(&mut self, value: A) -> Result<(), XenonError>
    {
        fill_try_dispatch(self, value)
    }
}
```

```rust,ignore
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
    /// Primary public entry point for writable storage.
    /// Use `try_fill()` only when the caller needs a fallible convenience
    /// wrapper over potentially non-writable storage.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor1::<f64>::zeros([5])?;
    /// t.fill(3.14);
    /// assert!(t.iter().all(|&x| x == 3.14));
    /// ```
    pub fn fill(&mut self, value: A) {
        fill_storage_mut(
            &mut self.storage,
            self.shape(),
            self.strides(),
            self.offset(),
            self.flags(),
            value,
        )
    }
}
```

- 需求来源：`需求说明书 §21.2` 要求只读引用和共享只读引用须拒绝填充请求。

### 5.3 `fill_try_dispatch()` 分派准则

`fill_try_dispatch()` 的内部判定标准固定为：

- 先通过 storage 层提供的 `pub(crate)` 内部可写能力 helper（参见 `05-storage.md` 的存储模式与可写能力约束；helper 名称不作为本文稳定契约）判定当前存储是否支持可写路径；
- `Owned` / `ViewMut` / 其他满足 `StorageMut` 的存储：进入 `fill_storage_mut()` 直接写入路径；
- `View` / `SharedReadOnly` / 其他只读或共享只读存储：返回 `XenonError::InvalidStorageMode`；
- 连续布局可走快路径，非连续或带 padding 布局必须退回“仅写逻辑元素”的 stride-aware 路径。

### 5.4 fill 的显式写入语义

- `fill` 必须按**逻辑索引**迭代，并且只写入逻辑元素。
- 对带 padding 的底层存储：不得写入任何 padding bytes。
- 对非连续但可写的视图：必须严格按 `shape` / `strides` / `offset` / `flags` 或等价 layout helper 导航到每个逻辑元素。
- 对存在零步长的布局：按照 `需求说明书 §16`，它们来自广播只读结果；这类只读/共享只读张量的 `try_fill()` 必须返回 `InvalidStorageMode`，而 `fill()` 因 `StorageMut` 约束在编译期不可用。

```
fill_logical_only(storage, shape, strides, offset, flags, value):
    logical_len = product(shape)
    for logical_index in 0..logical_len:
        offset = offset_for_logical_index(shape, strides, offset, logical_index)
        write storage[offset] = clone(value)
```

上述伪代码强调的是契约，而不是公开 API：实现可以使用递归多维索引、stride-aware iterator 或其他等价内部辅助函数，但结果必须等价于“按逻辑索引逐元素写入，且不触碰 padding / 非逻辑区域”。

### 5.5 连续性保证（to_contiguous）

`to_contiguous()` 是本模块定义的公共 API。内部可复用连续化实现，但这不构成 `convert` 模块的独立公共能力。
- `to_contiguous()` 由 utility 模块暴露。
- 若非连续路径需要额外实现步骤，也仅属于 utility 的内部细节。
- 类型转换语义仍归 convert，连续性保证语义仍归 utility。

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
    /// - `into_contiguous(self)` may reuse data only for canonical F-contiguous `Owned` input
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
    /// let transposed = t.transpose();
    /// let contig2 = transposed.to_contiguous();
    /// assert!(contig2.is_f_contiguous());
    /// ```
    ///
    /// # Trait bounds for `to_owned()`
    ///
    /// `to_owned()` is defined on `S: Storage<Elem = A>` with `A: Element`
    /// (the current impl also carries `A: Element + Clone`; see
    /// `21-type.md §5.5`). All 4 storage modes
    /// (Owned/ViewRepr/ViewMutRepr/ArcRepr) satisfy this bound,
    /// so `to_contiguous` is available on all storage types.
    pub fn to_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            self.to_owned()
        } else {
            util_internal_to_f_contiguous(self)
        }
    }

    /// Consume the tensor and ensure F-contiguous owned storage.
    ///
    /// Reuses the existing owned data **only when the input is already a
    /// canonical F-contiguous `Owned` tensor**; otherwise materializes a new
    /// contiguous tensor. In particular, an input that is merely
    /// "logically F-contiguous" (i.e. `is_f_contiguous()` returns `true`)
    /// but whose storage contains tail padding, has a non-zero offset, or
    /// is not an `Owned` representation, does NOT qualify for reuse and must
    /// go through `util_internal_to_f_contiguous` to produce a canonical
    /// owned result with no padding.
    pub fn into_contiguous(self) -> Tensor<A, D>
    where
        S: StorageIntoOwned<Elem = A>,
    {
        // Use `is_canonical_f_contiguous_owned()` — strictly stronger than
        // `is_f_contiguous()` — to gate the O(1) reuse path. See §6.3.
        if self.is_canonical_f_contiguous_owned() {
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

- `to_contiguous(&self)` 是稳定的"总是返回独立 owned 结果"入口；当输入已是连续 F-order 时，它不得改变逻辑值，且可以复用现有数据作为读取来源，但因为返回值必须与借用源解除别名，所以仍会物化为新的 owned 张量。
- `into_contiguous(self)` 是满足 `需求说明书 §22` 的消费式入口：`F-contiguous`（即 `is_f_contiguous()`）只表示逻辑上按 F-order 连续；`canonical F-contiguous owned`（即 `is_canonical_f_contiguous_owned()`，crate-internal predicate）进一步要求 storage 表示为 `Owned`、`offset == 0`、底层 buffer 无 tail padding，从而可作为 canonical owned 表示直接复用。`to_contiguous()` 对已连续输入始终返回新的 canonical F-order owned 拷贝。`into_contiguous()` **仅当 `is_canonical_f_contiguous_owned()` 为真时**才可 O(1) 复用现有数据；其他所有情况（包括仅 `is_f_contiguous()` 为真但带 tail padding、或非 `Owned` storage、或 `offset != 0` 的输入）都必须重新物化为 canonical F-order owned 结果。详细 predicate 定义与分派表见 §6.3。
- `to_contiguous()` / `into_contiguous()` 专注于连续性保证：仅在输入已是 canonical F-contiguous `Owned` 时，`into_contiguous()` 才可 O(1) 复用。`to_owned()` / `into_owned()`（见 `21-type.md`）专注于独立拷贝：无论原始布局如何，始终产出独立的拥有型存储。二者可能产生相同结果（非连续输入 → 连续 owned），但语义主语不同。
- `util_internal_to_f_contiguous()` 只接受“逻辑索引语义已验证、shape / strides / offset 自洽”的输入张量；调用方须先完成这些张量不变量检查。该 helper 的职责仅限于把当前逻辑元素重排并物化为 canonical F-order owned 结果，不再重复承担布局合法性验证。

### 5.6 Good / Bad 对比

```rust,ignore
// Good - use fill for in-place filling, zero extra allocation
let mut t = Tensor1::<f64>::zeros([1000])?;
t.fill(42.0);

// Bad - create a temporary Vec then construct a new tensor, double allocation
let data = vec![42.0; 1000];
let t = Tensor1::from_shape_vec([1000], data)?;
```

```rust,ignore
// Good - when the downstream function accepts a borrow and does not require owned storage,
// check contiguity before deciding whether to materialize an owned copy.
if tensor.is_f_contiguous() {
    process(&tensor);
} else {
    let contiguous = tensor.to_contiguous();
    process(&contiguous);
}

// Bad in borrow-only downstream paths - unconditionally calling to_contiguous
// wastes a copy when the input is already contiguous and ownership is not needed.
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

### 6.2 fill 算法

写入契约与伪代码见 §5.4；分派规则见 §5.3。

- 连续布局可走快路径（直接 memset 或逐元素写入）；带 padding / 非连续布局必须按逻辑索引与 strides 写入，且不得触碰 padding bytes。


### 6.3 to_contiguous 路径选择

`to_contiguous` / `into_contiguous` 共用同一 canonical 输出契约（无 inter-axis / tail padding，offset == 0，layout 由 `06-layout` 计算），但分派逻辑使用两套 predicate：

| Predicate                                     | 含义                                                                                                                                                                                                                          | 用途                                                            |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `is_f_contiguous()`                           | **逻辑** F-连续：`strides` 满足 `s_0 = 1`、`s_{i+1} = s_i * shape_i`，但不约束底层 storage 形态、offset 是否为 0、是否带 tail padding。算法定义见 `06-layout.md §5.7`。              | 仅作为 `to_contiguous` 的优化提示：可委托 `to_owned()` 走线性 memcpy 快路径。 |
| `is_canonical_f_contiguous_owned()` *(crate-internal)* | **canonical owned**：(1) `is_f_contiguous() == true`，(2) `S` 的运行时表示是 `Owned` 单一所有者，(3) `offset == 0`，(4) 底层 buffer 容量等于逻辑元素数 × `size_of::<A>()`（即无 tail padding）。 | 控制 `into_contiguous` 的 O(1) 复用路径。 |

```
to_contiguous(tensor):
    // `to_contiguous` ALWAYS yields a fresh canonical F-order owned tensor.
    // The `is_f_contiguous()` branch is purely an optimization hint: when
    // the input is logically F-contiguous, the implementation may delegate
    // to `to_owned()`, which (per `21-type.md §5.5`) is required to allocate
    // a brand-new canonical F-order buffer with NO tail padding (NOT to
    // alias / wrap / reuse the source storage). Non-F-contiguous inputs
    // (e.g. transposed views, sliced views, broadcast views) are re-packed
    // by `util_internal_to_f_contiguous`. Both branches end at the same
    // canonical contract.
    if is_f_contiguous(tensor):
        return to_owned(tensor)        // O(n) fresh canonical F-order copy
    else:
        return util_internal_to_f_contiguous(tensor)  // O(n) re-pack into F-order

into_contiguous(tensor):
    // O(1) reuse is permitted ONLY when the input is already a canonical
    // F-contiguous `Owned` tensor (see the predicate table above). All other
    // inputs — including ones that pass `is_f_contiguous()` but carry tail
    // padding, a non-zero offset, or a non-`Owned` storage representation —
    // MUST go through the re-pack path to honour the no-padding contract.
    if is_canonical_f_contiguous_owned(tensor):
        return reuse_owned_storage(tensor)  // O(1) move of the canonical buffer
    else:
        return util_internal_to_f_contiguous(&tensor)  // O(n) re-pack
```

设计契约：

- `to_contiguous()` 与 `into_contiguous()` 都必须返回 canonical F-order owned 张量：无 inter-axis padding、无 tail padding、`offset == 0`、layout flags 重新由 `06-layout` 计算。
- `is_f_contiguous()` 不是 `into_contiguous()` 复用的充分条件，因为它不能区分 owned/Arc/View、不能保证 offset==0、不能拒绝 tail padding。把它误用为 `into_contiguous()` 分派条件会让带 tail padding 的输入直接跨过 re-pack，破坏 SIMD/FFI 等下游对 canonical contiguous 的隐式假设。
- `is_canonical_f_contiguous_owned()` 是 crate-internal predicate，外部调用方不能名指；它的具体判定由 storage / layout 协同提供（参见 `05-storage.md §5.5` 的 ArcRepr→Owned 转换不变量与 `06-layout.md §5.7` 的连续性公式）。

### 6.4 NaN 处理语义

| clip 场景 | 输入   | min   | max   | 输出  | 说明                                          |
| --------- | ------ | ----- | ----- | ----- | --------------------------------------------- |
| 正常范围  | `0.5`  | `0.0` | `1.0` | `0.5` | 在范围内，不变                                |
| 低于下界  | `-1.0` | `0.0` | `1.0` | `0.0` | 钳位到 min                                    |
| 高于上界  | `2.0`  | `0.0` | `1.0` | `1.0` | 钳位到 max                                    |
| NaN 输入  | `NaN`  | `0.0` | `1.0` | `NaN` | NaN 不满足 `< min` 也不满足 `> max`，保持 NaN |

- 对浮点数，NaN 的 clip 行为遵循 IEEE 754 比较语义：`NaN < x` 和 `NaN > x` 均为 false，> 因此 NaN 值在 clip 中保持不变。这与 Numpy 的 `np.clip` 行为一致。
- 另一方面，`min`/`max` 作为边界参数必须是已定义的可比较标量值；若任一边界为 `NaN`，则返回 `InvalidArgument`，避免把无效边界静默当成合法区间。

---

## 7. 实现任务拆分

### Wave 1: 基础操作

- [ ] **T1**: 实现 `fill` 方法
  - 文件: `src/util/fill.rs`
  - 内容: 在 `S: StorageMut` 层实现 `fill(&mut self, value: A)` 核心 helper，并补上对所有张量开放的 `try_fill(&mut self, value: A) -> Result<(), XenonError>` 分发路径；该分发依赖 storage 层提供的可选可变句柄接口（或等价能力），且只读张量或缺失该能力的存储返回 `InvalidStorageMode`
  - 测试: `test_fill_basic`, `test_fill_non_contiguous`, `test_fill_padded_writes_logical_only`, `test_try_fill_read_only_returns_read_only_storage`
  - 前置: tensor 模块、iter 模块完成
  - 预计: 10 min

### Wave 2: 裁剪操作

- [ ] **T2**: 实现 `clip` 方法
  - 文件: `src/util/clip.rs`
  - 内容: `clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError>`；内部可复用非公开 clamp helper
  - 测试: `test_clip_basic`, `test_clip_nan`, `test_clip_nan_bound`, `test_clip_integers`
  - 前置: 无
  - 预计: 10 min

### Wave 3: 连续性保证

- [ ] **T3**: 实现 `to_contiguous` 方法
  - 文件: `src/util/contiguous.rs`
  - 内容: 实现 `to_contiguous(&self)` 与 `into_contiguous(self)`；非 F-contiguous 输入始终转为 F-order，连续 owned 输入允许复用数据
  - 测试: `test_to_contiguous_f_order`, `test_into_contiguous_reuses_owned_data`, `test_to_contiguous_transposed_becomes_f`, `test_to_contiguous_non_contiguous`
  - 前置: layout 模块的 `is_f_contiguous` 完成
  - 预计: 10 min

### Wave 4: 综合测试

- [ ] **T4**: 编写综合测试
  - 文件: `tests/test_utility.rs`
  - 内容: 边界测试（空数组、单元素、大数组、非连续布局）
  - 测试: `test_clip_empty`, `test_clip_single_element`, `test_clip_non_contiguous`, `test_fill_zero_dim`
  - 前置: T1, T2, T3
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                    |
| -------- | ------------------------ | ------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证 `clip`、`fill` 和 `to_contiguous` 的核心语义       |
| 集成测试 | `tests/`                 | 验证 `utility` 与 `tensor`、`iter`、`layout` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空数组、零维张量、NaN 和非连续布局等边界            |

### 8.2 单元测试清单

| 测试函数                                  | 测试内容                                       | 优先级 |
| ----------------------------------------- | ---------------------------------------------- | ------ |
| `test_clip_basic`                         | 基本裁剪：元素限制在 [0, 2] 范围               | 高     |
| `test_clip_no_change`                     | 所有元素在范围内，无变化                       | 高     |
| `test_clip_nan`                           | NaN 输入保持 NaN                               | 高     |
| `test_clip_nan_bound`                     | NaN 作为 min/max 返回 `InvalidArgument`        | 高     |
| `test_clip_integers`                      | i32/i64 整数裁剪                               | 中     |
| `test_clip_non_contiguous`                | 非连续布局返回正确裁剪结果                     | 高     |
| `test_fill_basic`                         | 基本填充所有元素为指定值                       | 高     |
| `test_fill_non_contiguous`                | 非连续布局正确填充所有逻辑元素                 | 高     |
| `test_fill_padded_writes_logical_only`    | 带 padding 的可写张量仅覆写逻辑元素            | 高     |
| `test_try_fill_writable_matches_fill`     | `try_fill()` 在可写张量上与 `fill()` 语义一致  | 高     |
| `test_try_fill_read_only_returns_error`   | `try_fill()` 在只读 / 共享只读 / 广播只读张量上返回 `InvalidStorageMode` | 高     |
| `test_fill_empty`                         | 空数组 fill 不 panic                           | 中     |
| `test_to_contiguous_f_order`              | F-order 连续输入返回 owned 拷贝                | 高     |
| `test_into_contiguous_reuses_owned_data`  | F-order owned 输入消费后复用原数据             | 高     |
| `test_to_contiguous_transposed_becomes_f` | 转置视图转为 F-order owned                     | 高     |
| `test_to_contiguous_non_contiguous`       | 非连续输入返回 F-order owned                   | 高     |

### 8.3 边界测试场景

| 场景                  | 预期行为                                                          |
| --------------------- | ----------------------------------------------------------------- |
| 空数组 `shape=[0, 3]` | `clip`/`fill`/`to_contiguous` 均正常处理，无 panic                |
| 单元素 `shape=[1]`    | `clip` 正确裁剪单个元素                                           |
| 零维张量              | `clip` 返回标量裁剪结果                                           |
| 非连续切片            | `fill`/`clip` 通过迭代器正确处理所有逻辑元素                      |
| 带 padding 的可写布局 | `fill` 只修改逻辑元素，对 padding bytes 保持不变                  |
| NaN 边界              | `clip(x, NaN, 1.0)` 或 `clip(x, 0.0, NaN)` 返回 `InvalidArgument` |
| 高维非连续布局        | rank-6 切片 / 转置 / 广播混合输入调用 `to_contiguous()` 后返回 F-order owned，元素顺序正确 |
| 超大张量连续化        | `10^7` 量级张量调用 `to_contiguous()` / `into_contiguous()` 后保持 shape 正确且不越界 |
| 可写 / 只读分派边界   | `fill()` 仅对可写存储成功；`try_fill()` 在只读 / 广播只读输入上返回 `InvalidStorageMode` |

### 8.4 属性测试不变量

| 不变量                                                                         | 测试方法                |
| ------------------------------------------------------------------------------ | ----------------------- |
| `clip(min, max)` 对非 NaN 输入产出 `[min, max]` 内元素；NaN 输入保持 NaN；NaN 边界返回错误 | 随机张量 + 随机 min/max |
| `fill(v)` 后 `iter().all(\|x\| *x == v)`                                       | 随机形状 + 随机值       |
| `to_contiguous()` / `into_contiguous()` 返回的张量 `is_f_contiguous() == true` | 随机非连续布局          |

### 8.5 集成测试

| 测试文件                | 测试内容                                                               |
| ----------------------- | ---------------------------------------------------------------------- |
| `tests/test_utility.rs` | `clip`/`fill`/`to_contiguous` 与 `tensor`、`iter`、`layout` 的协同路径 |

### 8.6 Feature gate / 配置测试

| 配置              | 验证点                                                                          |
| ----------------- | ------------------------------------------------------------------------------- |
| 默认配置          | `clip` / `fill` / `to_contiguous` 在默认构建下保持错误分层与 F-order 输出语义。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。                                           |

### 8.7 类型边界 / 编译期测试

| 场景                                                          | 测试方式                                                |
| ------------------------------------------------------------- | ------------------------------------------------------- |
| `clip` 仅对 `OrderedCompareElement` 开放，拒绝 `bool` / `Complex` | 编译期测试。                                        |
| `try_fill()` 对只读 / 共享只读 / 广播只读结果返回公开错误契约 | 运行时测试，断言返回 `XenonError::InvalidStorageMode`。 |
| `into_contiguous(self)` 仅对支持 owned 转换的存储模式开放     | 编译期测试。                                            |
| sort / argsort / searchsorted 不属于当前 API                  | API 缺失断言。                                          |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向               | 对方模块 | 接口/类型      | 约定                   |
| ------------------ | -------- | -------------- | ---------------------- |
| `utility → iter`   | `iter`   | `iter_mut()`   | `fill` 通过 storage 层 helper 直接写入逻辑元素（参见 §5.4），参见 `10-iterator.md` §5.6 |
| `utility → iter`   | `iter`   | `iter()`       | `clip` 通过只读迭代器读取并写入新张量，参见 `10-iterator.md` §5.6 |
| `utility → layout` | `layout` | 连续性查询     | `to_contiguous` 先查询当前布局是否已经连续，张量层方法参见 `07-tensor.md` §5.3，算法定义参见 `06-layout.md` §5.7 |
| `utility → tensor` | `tensor` | `to_owned()` / `into_owned()` | `to_contiguous` 与 `into_contiguous` 复用张量 owned 化路径（定义参见 `21-type.md` §5.5）；跨文档连续化归属统一在 utility |
| `utility → tensor` | `tensor` | owned 结果张量构造 | `clip` 分配新的 owned 结果张量，通过 `iter()` 读取源数据并写入 |

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

| 主题              | 内容                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------ |
| Recoverable error | `clip` 在 `min > max` 或任一边界为 `NaN` 时返回 `XenonError::InvalidArgument { operation: Cow::Borrowed("clip"), kind: InvalidArgumentKind::OperationSpecific { argument, constraint } }`（字段定义见 `26-error.md v3.0.0 §5.1`）。`try_fill()` 在只读 / 共享只读 / 缺失 `StorageMut` 能力的存储上返回 `XenonError::InvalidStorageMode { operation, expected: StorageKindTag::ViewMut（或 Owned）, actual: StorageKindTag::View（或 Arc）, shape: Some(self.shape().to_vec()), conversion: None }`，其中 `expected` 表示分派期望具备的可写能力对应的模式标签。`fill()` 因 `StorageMut` 编译期约束不会进入这条错误路径。`XenonError` 是本模块唯一公开错误类型。 |
| Panic             | 公开 utility API 不定义额外 panic 语义；连续化与裁剪失败统一走显式错误或正常返回。   |
| 路径一致性        | 连续与非连续布局都必须通过同一逻辑元素语义工作；当前无独立 SIMD / 并行分支。         |
| 容差边界          | `clip` 对浮点数遵循 IEEE 754 比较语义；不额外引入近似容差。                          |

---

## 11. 设计决策记录

### 决策 1：NaN 的 clip 行为

| 属性     | 值                                                                                          |
| -------- | ------------------------------------------------------------------------------------------- |
| 决策     | NaN 在 clip 中保持不变（不钳位）                                                            |
| 理由     | 遵循 IEEE 754 比较语义（`NaN < x` = false, `NaN > x` = false），与 NumPy `np.clip` 行为一致 |
| 替代方案 | NaN 裁剪到 min — 放弃，与 IEEE 754 和 NumPy 不一致                                          |
| 替代方案 | NaN 裁剪到 max — 放弃，同上                                                                 |

### 决策 2：to_contiguous 返回类型

| 属性     | 值                                                                                          |
| -------- | ------------------------------------------------------------------------------------------- |
| 决策     | 返回 `Tensor<A, D>`（Owned），不使用 `Cow`                                                  |
| 理由     | API 简洁（无生命周期参数）、调用方可预测行为、与 ndarray 设计一致；同时补充消费式 `into_contiguous(self)` 以在已连续 owned 输入上复用数据 |
| 替代方案 | 返回 `Cow<TensorBase<S, D>>` — 放弃，引入生命周期复杂度，调用方难以处理                     |
| 替代方案 | 已连续时返回视图（借引用） — 放弃，返回类型不确定，违反直觉                                 |

---

## 12. 性能考量

| 操作                              | 时间复杂度 | 空间复杂度 | 说明                                                                 |
| --------------------------------- | ---------- | ---------- | -------------------------------------------------------------------- |
| `clip`                            | O(n)       | O(n)       | 新分配一个张量                                                       |
| `fill`                            | O(n)       | O(1)       | 原地修改；utility 核心 helper 仅在可写层执行，`Clone` 开销取决于类型 |
| `to_contiguous`（已连续）         | O(n)       | O(n)       | 借用入口拷贝到新 owned                                               |
| `to_contiguous`（非连续）         | O(n)       | O(n)       | 拷贝 + 重新排列                                                      |
| `into_contiguous`（已连续 owned） | O(1)       | O(1)       | 直接复用现有 F-order owned 数据                                      |

- 连续布局的 `fill` 仅在填充值是全零 bit-pattern 时才可使用 `ptr::write_bytes(0)` 优化；一般情况仍应逐元素写入，避免把任意 `Copy` 值错误地按字节复制
- `clip` 的热点路径可考虑 SIMD 加速（参见 `08-simd.md` §5）

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                 |
| ---------- | ------------------------------------------------------------------------------------ |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 路径                          |
| MSRV       | Rust 1.85+                                                                           |
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
| 1.1.5 | 2026-04-15 |
| 1.1.6 | 2026-04-16 |
| 1.1.7 | 2026-04-16 |
| 2.0.0 | 2026-05-02 |

### v2.0.0 (2026-05-02) — 错误字段对齐 26-error v3.0.0

> 本版本与 `26-error.md v3.0.0` 协同更新；为非破坏性的内部错误结构调整（公开 API 形态保持不变，调用方仍通过 `Result<T, XenonError>` 处理错误）。

**契约更新**：

- §5.1 `clip` 的 `XenonError::InvalidArgument` 字段重写：从旧版 `argument/expected/actual/axis/axis_len/start/end/shape` 自由文本字段，改为 `kind: InvalidArgumentKind::OperationSpecific { argument: Cow<'static, str>, constraint: Cow<'static, str> }` 封闭枚举（对齐 `26-error.md v3.0.0 §5.1`）。
- §5.1 `operation` 字段统一使用 `Cow::Borrowed("clip")`（`Cow<'static, str>` 字段类型要求）。
- §5.1 `clip` doc comment `# Errors` 段重写：展开 `kind` 子变体，明示 `OperationSpecific` 的 argument / constraint 内容。
- §5.1 设计要点段：明确"`shape` 不再作为 `InvalidArgument` 的字段携带"，并指出旧版自由文本字段在 v3.0.0 已移除。

**协同与一致性更新**：

- §4.2 类型级依赖表新增 `error` 行：列出 `XenonError`、`InvalidArgumentKind::OperationSpecific`、`StorageKindTag`。
- §5.5 文档注释引用从 `21-type.md §5.x` 修正为具体的 `21-type.md §5.5`。
- §10 错误处理表 `Recoverable error` 一行重写：完整列出 `clip` 的 `InvalidArgument.kind` 与 `try_fill` 的 `InvalidStorageMode` 字段（`operation` / `expected: StorageKindTag::ViewMut（或 Owned）` / `actual: StorageKindTag::View（或 Arc）` / `shape: Some(...)` / `conversion: None`）。
- §5.3 将 `try_fill()` 分派前提收敛为 storage 层 `pub(crate)` 内部 helper；§5.5 / §6.3 统一 F-contiguous 与 canonical F-contiguous 语义；§5.6 限定 `to_contiguous()` Bad 示例适用场景；§8.4 修正 NaN 相关 clip 属性测试不变量。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
