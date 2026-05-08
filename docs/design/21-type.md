# 类型转换模块设计

> 文档编号: 21
> 模块目录: src/convert/
> 任务阶段: Phase 4
> 前置文档: 07-tensor.md, 03-element.md

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                                 | 不包含                                       |
| -------------- | ---------------------------------------------------- | -------------------------------------------- |
| 逐元素类型转换 | `cast<B: CastElement>(&self) -> Result<Tensor<B, D>, XenonError>` | 隐式类型提升（需显式调用）      |
| 同类型拷贝     | `to_owned`、`into_owned`                             | 标准库 `From`/`Into` 实现（归构造模块）      |
| 范围边界       | 逐元素类型转换与同类型拷贝 | 存储模式互转（归 `storage` / `tensor`）、连续化 helper（归 `utility`） |

### 1.2 设计原则

| 原则       | 体现                                                                       |
| ---------- | -------------------------------------------------------------------------- |
| 显式转换   | 所有类型转换须显式调用 `cast()`，无隐式提升                                |
| 失败可诊断 | 有损转换默认返回可恢复错误，错误上下文由 `XenonError::TypeConversion` 承载 |
| 存储约束   | `cast` 面向所有可读存储开放，但结果统一物化为 owned 张量                   |
| 需求闭合   | 仅支持 `需求说明书 §23.1` 与 `需求说明书 §23.2` 定义的类型对及其成功前提   |

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §23、§27、§28 |
| 范围内   | `cast()` / `CastTo` 为核心公开转换面；`to_owned()` / `into_owned()` 作为同模块便利 API 保留。 |
| 范围外   | 存储模式互转（归 `storage` / `tensor`）、标准库 `From` / `TryFrom` 实现（归构造模块）、连续化 helper（归 `utility`）。|
| 非目标   | 不默认放宽有损转换规则，不新增第三方转换库，也不把 `convert/` 扩展为独立的非 cast 存储转换层。|

---

## 3. 文件位置

```
src/
└── convert/                 # Type conversion module
    ├── mod.rs               # Module root, re-exports
    └── cast.rs              # cast() core plus colocated convenience APIs backed by the same matrix
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/convert/
├── mod.rs          # Re-exports: CastTo, cast, to_owned, into_owned
├── cast.rs         # Depends on element (CastTo), tensor (TensorBase), and shared convenience-path helpers

External dependencies:
├── crate::tensor        # TensorBase<S, D>, Tensor, TensorView
├── crate::dimension     # Dimension trait
├── crate::storage       # Storage, StorageMut, StorageOwned trait
├── crate::element       # Element, CastTo trait
├── crate::layout        # is_f_contiguous query
└── crate::error         # XenonError, Result<T>
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                     |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase`, `Tensor`, `.shape()`, `.strides()`, `.is_f_contiguous()`（参见 `07-tensor.md` §5）     |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md` §5）                                       |
| `storage`   | `Storage`, `StorageMut`, `Owned<A>`, `ViewRepr`, `ViewMutRepr`, `ArcRepr`（参见 `05-storage.md` §5） |
| `element`   | `Element`, `CastTo<B>`（参见 `03-element.md` §5.9）, `CastElement`（参见 `03-element.md §5.9.1`）    |
| `layout`    | `is_f_contiguous()`（参见 `06-layout.md` §5）                                                        |
| `error`     | `XenonError`、`Result<T>`、`ConversionFailureReason`                                                 |
| `iter`      | `iter()` 用于 `cast()` / `to_owned()` 的逐元素遍历（参见 `10-iterator.md` §5）                       |

### 4.3 依赖合法性

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

### 4.4 依赖方向声明

依赖方向：单向向上。`convert` 仅消费 `tensor`、`storage` 等核心模块，不被它们依赖。

---

## 5. 公共 API 设计

### 5.1 CastTo trait

- `CastTo<T>` trait 的唯一 owner 是 `03-element.md §5.9`。`convert` 模块只消费该 trait，并在受支持的源/目标类型矩阵上提供 `cast()` 路径，不重新定义 trait。
- `CastElement` 是公开 sealed marker trait，**唯一 owner 是 `03-element.md §5.9.1`**（v2.1.0 协同新增）。`convert` 模块只消费、不重新定义。`bool` 不属于 `CastElement`，因此 `cast::<bool>()` 在编译期被拒绝。封闭实现集合见 `03-element.md §5.9.1`：i32 / i64 / f32 / f64 / Complex<f32> / Complex<f64>。

```rust,ignore
// In `21-type.md`, only consumed via `use crate::element::CastElement;`.
// See `03-element.md §5.9.1` for the trait definition and impl set.
```

### 5.2 cast 方法

- 公开 API 统一使用 `Result<T, XenonError>`，`crate::error::Result<_>` 为等价类型别名。
- `element_index` 为按逻辑元素遍历顺序的 0-based 线性索引，非多维索引。
- `cast()` 面向所有可读存储开放；无论输入是 `Owned`、`ViewRepr`、`ViewMutRepr` 还是 `ArcRepr`，结果统一物化为新的 owned 张量，以保持返回类型与所有权语义一致。源类型与目标类型都进一步收缩为 `CastElement`，从签名层面排除 `bool`。
- `cast<B>()` 仅在 `A: CastElement + ConvertTo<B>` 时可用。`bool` 不实现 `CastElement`，因此 `Tensor<bool, _>` 上 `cast()` 在编译期不可调用，而不是落到运行时 `TypeConversion`。
- **`ConvertTo<B>` 是 `pub(crate) sealed` 内部分流 trait**，仅在 `convert/cast.rs` 定义，作为三层架构（§6.1 / §6.1.bis / §11 决策 4）的静态调度入口：Tier-1 lossless type pair 的 impl 直接走 `B::from(value)` 委托给 std `From`，Tier-2 / Tier-3 的 impl 委托给 `<A as CastTo<B>>::cast_to(value)`。这样公开的 `cast<B>()` 不再要求源/目标对必须实现 `CastTo`，避免 Tier-1 因为没有 `CastTo` impl 而无法通过 trait bound（项目不变量：Tier-1 不实现 `CastTo`）。

````rust,ignore
impl<S, A, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: CastElement,
{
    /// Element-wise type conversion.
    ///
    /// Available for any readable storage mode.
    /// The conversion always materializes an owned result tensor.
    ///
    /// # Type Parameters
    ///
    /// * `B` - Target element type
    ///
    /// # Errors
    ///
    /// Returns `XenonError::TypeConversion {
    ///     operation: Cow::Borrowed("cast"),
    ///     source_type: &'static str,
    ///     target_type: &'static str,
    ///     reason: ConversionFailureReason,
    ///     element_index: Some(usize),
    /// }` when any element cannot be converted under the rules defined in
    /// `需求说明书 §23`. `source_type` / `target_type` are `&'static str`
    /// (see `26-error.md v3.2.0 §5.1`); the canonical name strings come
    /// from `<A as Element>::ELEMENT_TYPE_NAME`
    /// (see `03-element.md §5.1.1`). They **must not** use
    /// `core::any::TypeId` and **must not** be free-form text. `operation`
    /// must be supplied (the v3.0.0 contract removed the implicit-empty
    /// default).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3])?;
    /// let b: Tensor1<f64> = a.cast()?;
    ///
    /// let c = Tensor1::from_shape_vec(Ix1(1), vec![Complex::new(1.0_f64, 0.0)])?;
    /// let d: Tensor1<f64> = c.cast()?;
    /// ```
    pub fn cast<B: CastElement>(&self) -> Result<Tensor<B, D>, XenonError>
    where
        A: ConvertTo<B>,
    {
        // iter() traverses elements in logical (F-order) linear order;
        // the enumerated `index` is thus the 0-based logical element index
        // used in `element_index` of XenonError::TypeConversion (see §10).
        //
        // Three-tier dispatch (per §6.1 / §6.1.bis / §11 Decision 4):
        //   ConvertTo::<B>::convert((*x))
        //     ├── Tier-1 (lossless static): impl returns `Ok(B::from(value))`,
        //     │   delegating to Rust std `From`. Never errors.
        //     ├── Tier-2 (lossy static):    impl delegates to
        //     │   `<A as CastTo<B>>::cast_to(value)` (which itself returns
        //     │   `Result<B, XenonError>`); for static lossy pairs cast_to
        //     │   always returns `Err(TypeConversion { reason: ... })`.
        //     │   The forwarding impl is `<A as CastTo<B>>::cast_to(self)`,
        //     │   NOT `Err(<A as CastTo<B>>::cast_to(self))` (that would be
        //     │   `Err(Result<...>)` nesting).
        //     └── Tier-3 (dynamic):         impl delegates to
        //         `<A as CastTo<B>>::cast_to(value)` — runtime check returns
        //         `Ok` or `Err` per element.
        //
        // The error rewrap below covers Tier-2/Tier-3; Tier-1 returns Ok
        // and never enters the `map_err` branch.
        let mut data: Vec<B> = Vec::with_capacity(self.len());
        for (index, x) in self.iter().enumerate() {
            let value = ConvertTo::<B>::convert(*x).map_err(|err| match err {
                // Inject the element index and ensure operation/source_type/
                // target_type are populated. CastTo::cast_to() (called from
                // the Tier-2/3 branches of ConvertTo::convert) emits the
                // structural fields (source_type/target_type/reason); cast()
                // attaches operation = "cast" and the resolved element_index.
                XenonError::TypeConversion {
                    source_type,
                    target_type,
                    reason,
                    ..
                } => XenonError::TypeConversion {
                    operation: Cow::Borrowed("cast"),
                    source_type,
                    target_type,
                    reason,
                    element_index: Some(index),
                },
                other => other,
            })?;
            data.push(value);
        }
        // Return through a pub(crate) internal helper that constructs an owned
        // tensor from already-validated shape/data length. The helper is not a
        // public API surface of convert; its exact name is intentionally kept
        // outside this document's stable contract.
        // SAFETY: `data.len() == self.len() == product(self.raw_dim())` because
        // the loop pushes exactly one element per F-order iteration over
        // `self`; `self.raw_dim()` was validated when `self` was constructed
        // (no shape-product overflow); F-order strides/flags are computed
        // internally by the helper. The helper is `pub(crate) unsafe fn`
        // (see §5.6); the `unsafe` block satisfies its safety contract.
        Ok(unsafe { Tensor::from_shape_vec_aligned_unchecked(self.raw_dim(), data) })
    }
}
````

### 5.3 类型转换路径表

| 源类型         | 目标类型       | 默认语义 | 说明                                        |
| -------------- | -------------- | -------- | ------------------------------------------- |
| `i32`          | `i64`          | 成功     | 无损                                        |
| `f32`          | `f64`          | 成功     | 无损                                        |
| `i32`          | `f64`          | 成功     | 无损                                        |
| `i32`          | `Complex<f64>` | 成功     | 实部无损转为 `f64`，虚部补 `0`              |
| `f32`          | `Complex<f64>` | 成功     | 实部无损转为 `f64`，虚部补 `0`              |
| `f32`          | `Complex<f32>` | 成功     | 虚部补 `0`                                  |
| `f64`          | `Complex<f64>` | 成功     | 虚部补 `0`                                  |
| `Complex<f32>` | `Complex<f64>` | 成功     | 分量宽化                                    |
| `i64`          | `i32`          | 错误     | 有损，默认失败                              |
| `f64`          | `f32`          | 错误     | 有损，默认失败                              |
| `f32`          | `i32`          | 错误     | 有损，默认失败                              |
| `f32`          | `i64`          | 错误     | 有损，默认失败                              |
| `f64`          | `i32`          | 错误     | 有损，默认失败                              |
| `f64`          | `i64`          | 错误     | 有损，默认失败                              |
| `i32`          | `f32`          | 错误     | 有损，默认失败                              |
| `i64`          | `f32`          | 错误     | 有损，默认失败                              |
| `i64`          | `f64`          | 错误     | 精度敏感：`±2^53` 内精确，超出按 IEEE 754 舍入。当前按 B10.a 选定为有损默认失败。 |
| `i32`          | `Complex<f32>` | 错误     | 由 `i32 -> f32` 有损导致默认失败            |
| `i64`          | `Complex<f64>` | 错误     | 由 `i64 -> f64` 精度敏感导致默认失败（同 `i64 → f64` 条目） |
| `i64`          | `Complex<f32>` | 错误     | 有损，默认失败                              |
| `f64`          | `Complex<f32>` | 错误     | 有损，默认失败                              |
| `Complex<f32>` | `f64`          | 条件成功 | 条件成功（虚部非 0 时返回 `NonZeroImaginaryPart`；虚部为 0 时再按 `f32 -> f64` 规则处理） |
| `Complex<f32>` | `f32`          | 条件成功 | 条件成功（虚部非 0 时返回 `NonZeroImaginaryPart`；虚部为 0 时返回实部） |
| `Complex<f32>` | `i32`          | 条件成功 | 条件成功（虚部为 0 时，按内层实数转换规则处理；内层有损则仍返回错误） |
| `Complex<f32>` | `i64`          | 条件成功 | 条件成功（虚部为 0 时，按内层实数转换规则处理；内层有损则仍返回错误） |
| `Complex<f64>` | `f64`          | 条件成功 | 条件成功（虚部非 0 时返回 `NonZeroImaginaryPart`；虚部为 0 时返回实部） |
| `Complex<f64>` | `f32`          | 条件成功 | 条件成功（虚部为 0 时，按内层实数转换规则处理；内层有损则仍返回错误） |
| `Complex<f64>` | `i32`          | 条件成功 | 条件成功（虚部为 0 时，按内层实数转换规则处理；内层有损则仍返回错误） |
| `Complex<f64>` | `i64`          | 条件成功 | 条件成功（虚部为 0 时，按内层实数转换规则处理；内层有损则仍返回错误） |
| `Complex<f64>` | `Complex<f32>` | 错误     | 分量精度丢失，默认失败                      |

- `bool` 不参与 `cast()`；任何 `bool` 相关逐元素类型转换都不在本模块范围内。
- `CastTo` 的完整实现矩阵通过显式 impl 列表（每对 `(A, B)` 一份手写 `impl CastTo<B> for A`）或 crate-internal `macro_rules!` 生成 impl，并以 compile-fail 矩阵测试 / 完整性测试守住覆盖。**不**使用任何形式的运行时 enum dispatch 来选择转换路径——所有 cast 行为在编译期通过单态化决议，避免引入额外间接调用与代码路径。
- §5.3 和 §5.4 的表项加闭合规则覆盖所有受支持组合，编译期测试验证无遗漏。

### 5.4 闭合规则映射

- 凡 `需求说明书 §23.1` 已逐项列出的组合，其默认语义与附加成功前提以 `需求说明书 §23.1` 表格为准；闭合规则仅用于补足未逐项列出的受支持组合，不得覆盖或重新解释已列组合的语义。
- 未在上表逐项展开、但属于受支持源/目标集合的组合，按 `需求说明书 §23.2` 闭合：
  - 实数 → 整数：一律默认为 `FloatToInteger` 错误（浮点值域与整数表示不兼容，包括 NaN/Inf 场景）
  - 整数 → 浮点（窄精度）：一律默认为有损失败（如 `i64 → f32`、`i64 → f64`[^precision_note]、`i32 → f32`），返回 `IntegerToFloatPrecisionLoss`

[^precision_note]: `i64 → f64` 在数学上 `±2^53` 范围内可精确表示，但超出此范围的 `i64` 值在转为 `f64` 时按 IEEE 754 round-to-nearest-even 可能丢失低阶位信息。当前按 B10.a 选定为有损默认失败。
  - 实数 → 复数：先按实数到目标复数实部分量类型的规则转换实部，再补 `0` 虚部
  - 复数 → 实数：仅当虚部为 `0` 时才可继续；但这只是必要条件而非充分条件。若实部到目标实数类型的内层转换按 `需求说明书 §23.1` 属于默认有损失败，则整体转换仍为默认错误并必须返回 `Err`
  - 复数 → 复数：实部和虚部分别按对应实数转换规则处理
  - 任一步为有损时，默认整体返回 `XenonError::TypeConversion`

### 5.5 to_owned / into_owned

- `to_owned()` / `into_owned()` 的公开语义只在本文维护；它们返回的 owned 结果固定为 Xenon 的 canonical F-order。`20-utility.md` 可引用它们作为 `to_contiguous()` 的实现依赖，但不再重复定义其契约。
- 同类型拷贝（`to_owned()`/`into_owned()`）不通过 fallible `cast()` 建模，而是始终成功的基础操作。`cast::<A>()` 不适用于同类型拷贝场景。
- `ArcRepr → Owned` 始终分配并复制（O(n)），与引用计数无关。

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Clones data into a new owned tensor.
    ///
    /// Always allocates new memory and copies data, even if input is already Owned.
    /// The returned owned tensor uses Xenon's canonical F-order layout.
    ///
    /// # Examples
    ///
    /// ```
    /// let view: TensorView<f64, Ix1> = tensor.view();
    /// let owned: Tensor<f64, Ix1> = view.to_owned();
    /// ```
    pub fn to_owned(&self) -> Tensor<A, D> {
        // Always produce F-order (Xenon only supports F-order, see requirement §7).
        // Iterate in F-order and collect into a new aligned allocation.
        let mut data: Vec<A> = Vec::with_capacity(self.len());
        for elem in self.iter().cloned() {
            data.push(elem);
        }
        // Internal pub(crate) helper that performs unchecked construction
        // when shape/data-length consistency is guaranteed by the caller.
        // Here, `data.len() == self.len() == product(self.shape())` and
        // `self.shape()` was validated when `self` was constructed, so
        // shape-product overflow / element-count mismatch cannot occur.
        // The fallible `from_shape_vec_aligned` is therefore not used in
        // this path; instead we invoke the unchecked helper to keep
        // `to_owned()` infallible. See 05-storage.md §6.1 for the
        // unchecked-construction contract.
        // SAFETY: `data.len() == self.len() == product(self.raw_dim())` (loop
        // pushes one element per F-order iteration); `self.raw_dim()` was
        // validated at `self` construction time; F-order strides/flags are
        // computed internally by the helper. The helper is `pub(crate)
        // unsafe fn` (see §5.6); the `unsafe` block satisfies its safety
        // contract.
        unsafe { Tensor::from_shape_vec_aligned_unchecked(self.raw_dim(), data) }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A>,
    D: Dimension,
    A: Element,
{
    /// Consumes the tensor, converting to owned.
    ///
    /// - `Tensor`: returned directly, O(1)
    /// - `TensorView`/`TensorViewMut`: allocates and copies into canonical F-order, O(n)
    /// - `ArcTensor`: always allocates and copies into canonical F-order, O(n), regardless of ref count
    ///
    /// # Examples
    ///
    /// ```
    /// let tensor: Tensor<f64, Ix1> = Tensor::from_shape_vec(Ix1(2), vec![1.0, 2.0])?;
    /// let owned: Tensor<f64, Ix1> = tensor.into_owned(); // O(1), same data
    ///
    /// let view: TensorView<f64, Ix1> = tensor.view();
    /// let owned_from_view: Tensor<f64, Ix1> = view.into_owned(); // O(n), new allocation
    /// ```
    pub fn into_owned(self) -> Tensor<A, D> {
        // Use storage-level StorageIntoOwned::into_owned_storage() to
        // convert the storage to Owned (see 05-storage.md §5.9).
        // The tensor-level method then ensures canonical F-order:
        //   Owned    → O(1) when source is already F-contiguous with offset=0
        //   View/ViewMut/Arc → O(n) allocate + copy into canonical F-order
        let owned_storage = self.storage.into_owned_storage();
        self.into_owned_from_owned_storage(owned_storage)
    }
}
````

### 5.6 内部构造辅助边界

`from_shape_vec_aligned_unchecked` 是 `TensorBase::new_unchecked`（07-tensor.md §5.6）的**薄封装**（thin wrapper），存在目的是为 `cast()` / `to_owned()` / `into_owned()` 提供从已验证 `(shape, Vec<A>)` 对零开销构造 owned 张量的单一路径。它不是独立的 unsafe 入口——其安全性由 `new_unchecked` 的安全契约兜底。

- 内部实现调用链：
  1. 通过 `Owned::from_vec_aligned` 从 `data: Vec<A>` 构造对齐 storage
  2. 通过 `layout::compute_f_strides` 从 `shape` 推导规范 F-order strides
  3. 通过 `layout::compute_layout_flags` 从 `(shape, strides, storage.as_ptr())` 计算布局标志
  4. 调用 `TensorBase::new_unchecked(storage, shape, strides, /*offset=*/ 0, flags, /*derived_from_view_mut=*/ false)` 完成构造

- **本函数的安全契约**（v2.1.2 大幅缩减，其余 forwarded 到 `new_unchecked`）：
  - 调用方必须证明 `shape.checked_size().is_ok()` 且 `data.len() == shape.checked_size().unwrap()`。
  - 所有其他张量不变式（offset 计算、别名安全、布局标志正确性、F-order 元数据合法性）均由 `TensorBase::new_unchecked` 的契约兜底——详见 07-tensor.md §5.6。
  - 本函数**永远**不设置 `derived_from_view_mut = true`（它始终构造 Owned，无从携带 ViewMut 降级标记）。

  规范签名（被 `25-safety.md §5.12.1` 索引引用）：

  ```rust,ignore
  impl<A, D> Tensor<A, D>
  where
      A: Element,
      D: Dimension,
  {
      /// Thin wrapper around `TensorBase::new_unchecked` (&sect;07-tensor.md §5.6)
      /// for zero-overhead Owned construction from a validated `(shape, Vec<A>)` pair.
      ///
      /// Internally builds aligned storage via `Owned::from_vec_aligned`, computes
      /// canonical F-order strides via `layout::compute_f_strides`, computes layout
      /// flags via `layout::compute_layout_flags`, then calls
      /// `TensorBase::new_unchecked(...)` with `offset=0` and
      /// `derived_from_view_mut=false`.
      ///
      /// # Safety
      ///
      /// Caller must prove:
      /// - `shape.checked_size()` is `Ok` (no overflow)
      /// - `data.len() == shape.checked_size().unwrap()`
      ///
      /// All other tensor invariants (offset arithmetic, alias safety, layout
      /// flag correctness) are forwarded to `TensorBase::new_unchecked`'s contract
      /// — see 07-tensor.md §5.6.
      ///
      /// Used by `cast()` / `to_owned()` / `into_owned()` after they have
      /// already proved length / shape consistency at the call site.
      pub(crate) unsafe fn from_shape_vec_aligned_unchecked(
          shape: D,
          data: Vec<A>,
      ) -> Tensor<A, D>;
  }
  ```
- helper 名称中的 `unchecked` 严格表示"跳过可由调用方安全封装重复检查的 metadata 校验"，**不**表示"内部实现可放任任意输入"；违反本函数安全契约（shape/data 长度不匹配）的后果是 UB，由 `new_unchecked` 的存储/元数据不一致语义传递过去。
- 之所以让 helper 保留 `unsafe` 而不是把它做成 safe 函数，是为了让 `cast()` / `to_owned()` 的 infallible 签名真正零额外检查开销；safe wrapper 路径已由 `Tensor::from_shape_vec` 提供（fallible，返回 `Result`）。

### 5.7 Good / Bad 对比

```rust,ignore
// Good - explicit and fallible cast
let a: Tensor<i32, Ix1> = Tensor::from_shape_vec(Ix1(2), vec![1, 2])?;
let b: Tensor<f64, Ix1> = a.cast()?;

// Bad - implicit type promotion (Xenon does not support this)
let floats: Tensor<f64, Ix1> = Tensor::from_shape_vec(Ix1(2), vec![1.0, 2.0])?;
let ints: Tensor<i32, Ix1> = floats + 1.0;  // Compile error: type mismatch

// Good - complex to real is allowed only when imag == 0
let complex_t: Tensor<Complex<f64>, Ix1> = Tensor::from_shape_vec(Ix1(1), vec![Complex::new(3.0, 0.0)])?;
let re_parts: Tensor<f64, Ix1> = complex_t.cast()?;

// Bad - assuming lossy conversion succeeds by default
let floats: Tensor<f64, Ix1> = Tensor::from_shape_vec(Ix1(2), vec![1.5, 2.7])?;
let ints: Tensor<i32, Ix1> = floats.cast().unwrap();  // forbidden: returns TypeConversion error
```

---

## 6. 内部实现设计

### 6.1 CastTo 实现

> **三层架构（v2.1.1，与 §11 决策 4 一致）**：
> - **Tier-1 — 静态无损（`From`）**：通过 Rust 标准 `From`/`Into` 表达，例如 `f64: From<f32>`、`i64: From<i32>`、`f64: From<i32>`。这一层**不**经过 `CastTo`，调用点用 `T::from(value)` 或 `value.into()` 即零开销转换。`cast()` 主循环对静态无损 type pair 不实例化 `CastTo`，而是内部直接走 `From` 委托。
> - **Tier-2 — 静态有损（`CastTo` 静态 `Err`）**：例如 `f64 → f32`、`f64 → i32`。`impl CastTo<T> for U` 直接 `Err(TypeConversion { reason: LossyFloatNarrowing | LossyFloatToInt | ... })`，无运行时值域检查。
> - **Tier-3 — 动态条件（`CastTo` 运行时判定）**：仅 `Complex<T> → T` 这一类需要按 `im == 0.0` 等运行时条件决定 `Ok / Err`。
>
> 下方仅展示 Tier-2 / Tier-3 的 `CastTo` 实现；Tier-1 通过 `From` 表达，本节为完整记录附录展示在 §6.1.bis。

`CastTo` 的规范签名统一为（与 `03-element.md` §5.9 的权威定义对齐）：

```rust,ignore
pub trait CastTo<T: Element>: Element {
    fn cast_to(self) -> Result<T, XenonError>;
}
```

`T: Element` bound 与 `Self: Element` super-trait 一起约束源/目标类型只能在封闭元素集合（`i32 / i64 / f32 / f64 / Complex<f32> / Complex<f64> / bool`）内取值，外部 crate 无法扩展转换矩阵。`bool` 默认排除在源/目标之外（参见 `03-element.md §5.9`）。

调用形态为 `value.cast_to()`；`element_index` 由调用方在逐元素遍历时单独跟踪，而不是作为 `CastTo` 的参数传入。`element_index` 为按逻辑元素遍历顺序的 0-based 线性索引，非多维索引。

```rust,ignore
// Element index is tracked by the caller, not passed to CastTo
let converted: Result<T, XenonError> = value.cast_to();
```

```rust,ignore
use std::borrow::Cow;

use crate::element::Element; // for `<A as Element>::ELEMENT_TYPE_NAME`
use crate::error::{ConversionFailureReason, XenonError};

// All TypeConversion errors below set `operation = Cow::Borrowed("cast_to")`
// (a stable non-empty operation name; satisfies `26-error.md §8.2`'s
// non-empty contract for direct `CastTo::cast_to()` callers) and
// `element_index = None`; the tensor-level caller (cast() in §5.2)
// rewraps to `operation = Cow::Borrowed("cast")` and injects the
// resolved element index. See §5.2 cast() for the rewrap pattern.
// Empty `Cow::Borrowed("")` is FORBIDDEN at this layer.
//
// Note (26-error v3.2.0): source_type / target_type are `&'static str`
// (NOT `core::any::TypeId`, NOT `ElementType` enum). Values come from
// `<A as Element>::ELEMENT_TYPE_NAME`, ensuring `error` (L0) holds no
// `element` dependency while still producing readable Display output.

// === Lossy-by-default conversion ===
impl CastTo<f32> for f64 {
    #[inline]
    fn cast_to(self) -> Result<f32, XenonError> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME, // == "f64"
            target_type: <f32 as Element>::ELEMENT_TYPE_NAME, // == "f32"
            reason: ConversionFailureReason::LossyFloatNarrowing,
            element_index: None,
        })
    }
}

// === Lossless widening — Tier-1 (`From`, NOT `CastTo`) ===
// f32 → f64, i32 → i64, i32 → f64 are expressed via Rust standard `From`
// (e.g. `f64::from(x)`, `<f64 as From<i32>>::from(x)`). They do NOT have
// a `CastTo` impl; the `cast()` main loop dispatches statically-known
// lossless pairs to `T::from(value)` directly. See §6.1.bis for the
// lossless impl listing and §11 Decision 4 for the three-tier rationale.

// === Float → integer (lossy-by-default) ===
impl CastTo<i32> for f64 {
    #[inline]
    fn cast_to(self) -> Result<i32, XenonError> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME, // "f64"
            target_type: <i32 as Element>::ELEMENT_TYPE_NAME, // "i32"
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        })
    }
}

// === Real → complex (lossless, zero imaginary) ===
// NOT a `CastTo` impl. These cells are Tier-1 lossless and routed through
// `ConvertTo<B>` directly with `Ok(Complex::new(value, 0.0))` shims (see
// §6.1.ter "Tier-1 lossless real→complex and complex widening" block); `cast()` dispatches them
// without ever instantiating `CastTo`. Listed here only as a reminder that
// real→complex zero-imaginary widenings are NOT `CastTo` cases.

// === Complex → complex (lossless widening) ===
// NOT a `CastTo` impl. `Complex<f32> → Complex<f64>` is Tier-1 lossless and
// expressed in §6.1.ter via a direct `ConvertTo` shim (componentwise
// `f64::from(...)`). It is NOT routed through `CastTo`. Listed here only
// as a reminder that complex widening is NOT a `CastTo` case.

// === Conditionally successful conversions ===
impl CastTo<f64> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<f64, XenonError> {
        if self.im == 0.0 {
            Ok(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed("cast_to"),
                source_type: <Complex<f64> as Element>::ELEMENT_TYPE_NAME, // "Complex<f64>"
                target_type: <f64 as Element>::ELEMENT_TYPE_NAME, // "f64"
                reason: ConversionFailureReason::NonZeroImaginaryPart,
                element_index: None,
            })
        }
    }
}

// === Lossy-by-default conversion ===
impl CastTo<i32> for i64 {
    #[inline]
    fn cast_to(self) -> Result<i32, XenonError> {
        Err(XenonError::TypeConversion {
            operation: Cow::Borrowed("cast_to"),
            source_type: <i64 as Element>::ELEMENT_TYPE_NAME, // "i64"
            target_type: <i32 as Element>::ELEMENT_TYPE_NAME, // "i32"
            reason: ConversionFailureReason::LossyIntegerNarrowing,
            element_index: None,
        })
    }
}

// `CastTo<f64> for i64` is not explicitly listed here because it falls
// into the lossy-by-default macro-generated pattern per §5.4.
// See §5.3 i64→f64 entry: B10.a selects lossy-by-default failure.
```

### 6.1.bis Tier-1 静态无损（`From` impls）

下列 `From` impls **不**通过 `CastTo`，由 Rust 标准库直接提供。锁定的 6×6 `CastElement`-only 矩阵 (bool 排除) 中包含的 Tier-1 std `From` 共 3 对：`f64: From<f32>`、`i64: From<i32>`、`f64: From<i32>`（`u32` 等其它整数类型不在封闭元素集，不参与 ConvertTo 矩阵）。`cast()` 主循环对 Tier-1 type pair 的实现委托：

```rust,ignore
// In `cast()` main loop (§5.2 pseudo-code), the static dispatch picks
// `From` for lossless pairs and `CastTo` for lossy / dynamic pairs:
//
//   if STATICALLY_LOSSLESS::<A, T>() {
//       data.push(T::from(value));            // Tier-1: From, infallible
//   } else {
//       let converted = value.cast_to()       // Tier-2 / Tier-3
//           .map_err(|e| /* inject operation + element_index */)?;
//       data.push(converted);
//   }

// Tier-1 lossless pairs (provided by std, NOT impl'd in this crate):
//
//   impl From<f32> for f64        // f32 → f64
//   impl From<i32> for i64        // i32 → i64
//   impl From<i32> for f64        // i32 → f64
//
// (Same-type "conversions" — e.g. f64 → f64 — short-circuit even earlier
// to `value` directly without traversing the From/CastTo branch.)
```

设计要点：

- **Tier-1 不允许 `Err`**：以编译期 `From` 表达就消除了运行时分支与错误路径，`cast()` 在 Tier-1 路径不会构造 `XenonError::TypeConversion`。
- **Tier-2/Tier-3 由 `CastTo` 承担**：见 §6.1 上方代码块。Tier-2 静态返回 `Err`，Tier-3 按运行时条件返回 `Ok / Err`。
- **静态分流的实现细节**：`STATICALLY_LOSSLESS::<A, T>()` 是 §5.2 内部 helper（不公开），按 `<A as Element>::ELEMENT_TYPE` × `<T as Element>::ELEMENT_TYPE` 静态判定。具体实现可用 trait 关联常量、宏、或 specialization-free 的若干 `where` 子句穷举三对，详见 §5.2 伪代码注释。

### 6.1.ter `ConvertTo<B>` 内部 sealed 分流 trait

`ConvertTo<B>` 是 `convert/cast.rs` 内部的 `pub(crate) sealed` trait，作为 `cast<B>()` 公开 API 的静态分流入口。它统一三层架构的实现（Tier-0 / Tier-1 / Tier-2 / Tier-3），让 `cast<B>()` 的 trait bound 不必直接要求 `CastTo<B>`（避免 Tier-1 type pair 因不实现 `CastTo` 而无法通过 bound）：

> **覆盖范围（`CastElement`-only 6×6 矩阵）**：`ConvertTo<B>` 的源/目标类型集合严格为 `CastElement` 封闭实现集（`i32 / i64 / f32 / f64 / Complex<f32> / Complex<f64>`，**不**包含 `bool`），因此完整矩阵是 6×6 = 36 个 type pair，**不**是基于 7 元素全集的 49 对。`bool` 既不在 `CastElement` 封闭集合内（`03-element.md §5.9.1`），也不在 `cast()` 类型矩阵内（`§5.1`）；`Tensor<bool, _>` 不能调用 `cast()`、`Tensor<_, _>` 也不能 `cast::<bool>()`，编译期通过 `B: CastElement` bound 拒绝。`bool` 张量的同型拷贝由 `to_owned()` 承担（`§5.5`），不走 `cast::<bool>()` 路径。下方"same-type identity short-circuit (Tier-0)"块仅覆盖 `CastElement` 6 类同型对，`bool -> bool` 不属于该 Tier-0。

```rust,ignore
// crate-internal, NOT exported. Sealed via crate::private::Sealed.
pub(crate) trait ConvertTo<B>: crate::element::CastElement
where
    B: crate::element::CastElement,
{
    /// Returns `Ok(B::from(self))` for Tier-1 lossless pairs;
    /// delegates to `<Self as CastTo<B>>::cast_to(self)` for Tier-2/3 pairs.
    fn convert(self) -> Result<B, XenonError>;
}

// === Tier-1: Lossless. Use `From`, no `CastTo` impl. ===
impl ConvertTo<f64> for f32 {
    #[inline] fn convert(self) -> Result<f64, XenonError> { Ok(f64::from(self)) }
}
impl ConvertTo<i64> for i32 {
    #[inline] fn convert(self) -> Result<i64, XenonError> { Ok(i64::from(self)) }
}
impl ConvertTo<f64> for i32 {
    #[inline] fn convert(self) -> Result<f64, XenonError> { Ok(f64::from(self)) }
}

// === Same-type identity short-circuit (Tier-0 — no conversion at all). ===
// Covers the 6 `CastElement` types only (not `bool`); `bool` self-copy is
// handled by `to_owned()` per §5.5, not by `cast::<bool>()`.
impl ConvertTo<f32>          for f32          { #[inline] fn convert(self) -> Result<f32, XenonError>          { Ok(self) } }
impl ConvertTo<f64>          for f64          { #[inline] fn convert(self) -> Result<f64, XenonError>          { Ok(self) } }
impl ConvertTo<i32>          for i32          { #[inline] fn convert(self) -> Result<i32, XenonError>          { Ok(self) } }
impl ConvertTo<i64>          for i64          { #[inline] fn convert(self) -> Result<i64, XenonError>          { Ok(self) } }
impl ConvertTo<Complex<f32>> for Complex<f32> { #[inline] fn convert(self) -> Result<Complex<f32>, XenonError> { Ok(self) } }
impl ConvertTo<Complex<f64>> for Complex<f64> { #[inline] fn convert(self) -> Result<Complex<f64>, XenonError> { Ok(self) } }

// === Tier-1 lossless real→complex and complex widening (zero-imaginary widening for real→complex; componentwise f64::from for complex widening) ===
// `f → Complex<f>` and `i → Complex<f>` lossless cells are expressed by
// `From` (where std provides them) or by direct `Ok(Complex::new(...))`
// shims here when std does not. They are NOT routed through `CastTo`:
// `CastTo` is reserved for Tier-2 / Tier-3 (lossy or dynamic) per §6.1.
impl ConvertTo<Complex<f32>> for f32 {
    #[inline] fn convert(self) -> Result<Complex<f32>, XenonError> { Ok(Complex::new(self, 0.0)) }
}
impl ConvertTo<Complex<f64>> for f64 {
    #[inline] fn convert(self) -> Result<Complex<f64>, XenonError> { Ok(Complex::new(self, 0.0)) }
}
impl ConvertTo<Complex<f64>> for f32 {
    #[inline] fn convert(self) -> Result<Complex<f64>, XenonError> { Ok(Complex::new(f64::from(self), 0.0)) }
}
impl ConvertTo<Complex<f64>> for i32 {
    #[inline] fn convert(self) -> Result<Complex<f64>, XenonError> { Ok(Complex::new(f64::from(self), 0.0)) }
}
impl ConvertTo<Complex<f64>> for Complex<f32> {
    #[inline] fn convert(self) -> Result<Complex<f64>, XenonError> {
        Ok(Complex::new(f64::from(self.re), f64::from(self.im)))
    }
}

// === Tier-2 / Tier-3: lossy / dynamic. Delegate to CastTo. ===
// `impl<A, B> ConvertTo<B> for A where A: CastTo<B>` won't work due to
// coherence with the lossless / identity impls above; instead each
// remaining type pair gets its own forwarding impl. Each forwarding impl
// is a `#[inline]` thin shim around `<A as CastTo<B>>::cast_to(self)`.
//
// Complete 6×6 matrix below (36 cells total accounted for):
//   - 6 Tier-0 identity impls (listed above)
//   - 3 Tier-1 lossless arithmetic impls via std `From` (listed above)
//   - 5 Tier-1 lossless real→complex and complex widening impls (listed above; 4 real→complex + 1 complex→complex widening)
//   - 22 Tier-2/Tier-3 forwarding impls (listed below)
//
// Macro generation is acceptable for the 22 forwarding impls; they are
// mechanical and exhaustively determined by §5.3's type-pair classification.

// --- Tier-2 lossy (static Err): 14 cells ---
// f64 → {f32, i32, i64}
impl ConvertTo<f32> for f64 { #[inline] fn convert(self) -> Result<f32, XenonError> { <f64 as CastTo<f32>>::cast_to(self) } }
impl ConvertTo<i32> for f64 { #[inline] fn convert(self) -> Result<i32, XenonError> { <f64 as CastTo<i32>>::cast_to(self) } }
impl ConvertTo<i64> for f64 { #[inline] fn convert(self) -> Result<i64, XenonError> { <f64 as CastTo<i64>>::cast_to(self) } }
// f32 → {i32, i64}
impl ConvertTo<i32> for f32 { #[inline] fn convert(self) -> Result<i32, XenonError> { <f32 as CastTo<i32>>::cast_to(self) } }
impl ConvertTo<i64> for f32 { #[inline] fn convert(self) -> Result<i64, XenonError> { <f32 as CastTo<i64>>::cast_to(self) } }
// i64 → {i32, f32, f64}  (i64→f64 is Tier-2 per B10.a, see §5.3 footnote)
impl ConvertTo<i32> for i64 { #[inline] fn convert(self) -> Result<i32, XenonError> { <i64 as CastTo<i32>>::cast_to(self) } }
impl ConvertTo<f32> for i64 { #[inline] fn convert(self) -> Result<f32, XenonError> { <i64 as CastTo<f32>>::cast_to(self) } }
impl ConvertTo<f64> for i64 { #[inline] fn convert(self) -> Result<f64, XenonError> { <i64 as CastTo<f64>>::cast_to(self) } }
// i32 → f32  (lossy per closed rule §5.4)
impl ConvertTo<f32> for i32 { #[inline] fn convert(self) -> Result<f32, XenonError> { <i32 as CastTo<f32>>::cast_to(self) } }
// real → complex (lossy through inner real conversion)
impl ConvertTo<Complex<f32>> for i32          { #[inline] fn convert(self) -> Result<Complex<f32>, XenonError> { <i32 as CastTo<Complex<f32>>>::cast_to(self) } }
impl ConvertTo<Complex<f32>> for i64          { #[inline] fn convert(self) -> Result<Complex<f32>, XenonError> { <i64 as CastTo<Complex<f32>>>::cast_to(self) } }
impl ConvertTo<Complex<f64>> for i64          { #[inline] fn convert(self) -> Result<Complex<f64>, XenonError> { <i64 as CastTo<Complex<f64>>>::cast_to(self) } }
impl ConvertTo<Complex<f32>> for f64          { #[inline] fn convert(self) -> Result<Complex<f32>, XenonError> { <f64 as CastTo<Complex<f32>>>::cast_to(self) } }
// complex → complex narrowing
impl ConvertTo<Complex<f32>> for Complex<f64> { #[inline] fn convert(self) -> Result<Complex<f32>, XenonError> { <Complex<f64> as CastTo<Complex<f32>>>::cast_to(self) } }

// --- Tier-3 dynamic (Complex → real, conditional on im==0): 8 cells ---
impl ConvertTo<f32> for Complex<f32> { #[inline] fn convert(self) -> Result<f32, XenonError> { <Complex<f32> as CastTo<f32>>::cast_to(self) } }
impl ConvertTo<f64> for Complex<f32> { #[inline] fn convert(self) -> Result<f64, XenonError> { <Complex<f32> as CastTo<f64>>::cast_to(self) } }
impl ConvertTo<i32> for Complex<f32> { #[inline] fn convert(self) -> Result<i32, XenonError> { <Complex<f32> as CastTo<i32>>::cast_to(self) } }
impl ConvertTo<i64> for Complex<f32> { #[inline] fn convert(self) -> Result<i64, XenonError> { <Complex<f32> as CastTo<i64>>::cast_to(self) } }
impl ConvertTo<f32> for Complex<f64> { #[inline] fn convert(self) -> Result<f32, XenonError> { <Complex<f64> as CastTo<f32>>::cast_to(self) } }
impl ConvertTo<f64> for Complex<f64> { #[inline] fn convert(self) -> Result<f64, XenonError> { <Complex<f64> as CastTo<f64>>::cast_to(self) } }
impl ConvertTo<i32> for Complex<f64> { #[inline] fn convert(self) -> Result<i32, XenonError> { <Complex<f64> as CastTo<i32>>::cast_to(self) } }
impl ConvertTo<i64> for Complex<f64> { #[inline] fn convert(self) -> Result<i64, XenonError> { <Complex<f64> as CastTo<i64>>::cast_to(self) } }

// === 36-cell coverage check ===
//   Tier-0 identity:            6 cells (f32/f64/i32/i64/Complex<f32>/Complex<f64> self)
//   Tier-1 std From:            3 cells (f32→f64, i32→i64, i32→f64)
//   Tier-1 real→complex:        5 cells (f32→C<f32>, f64→C<f64>, f32→C<f64>, i32→C<f64>, C<f32>→C<f64>)
//   Tier-2 lossy:              14 cells (f64→{f32,i32,i64}, f32→{i32,i64}, i64→{i32,f32,f64},
//                                       i32→f32, i32→C<f32>, i64→C<f32>, i64→C<f64>, f64→C<f32>, C<f64>→C<f32>)
//   Tier-3 dynamic:             8 cells (C<f32>→{f32,f64,i32,i64}, C<f64>→{f32,f64,i32,i64})
//   Total:                     36 cells = 6 × 6.
```

> **6×6 矩阵分类总览（行=源 A，列=目标 B；编号即上文 impl 的 Tier 标签）**：
>
> | A \ B          | `i32` | `i64` | `f32` | `f64` | `Complex<f32>` | `Complex<f64>` |
> | -------------- | ----- | ----- | ----- | ----- | -------------- | -------------- |
> | `i32`          | T0    | T1    | T2    | T1    | T2             | T1             |
> | `i64`          | T2    | T0    | T2    | T2    | T2             | T2             |
> | `f32`          | T2    | T2    | T0    | T1    | T1             | T1             |
> | `f64`          | T2    | T2    | T2    | T0    | T2             | T1             |
> | `Complex<f32>` | T3    | T3    | T3    | T3    | T0             | T1             |
> | `Complex<f64>` | T3    | T3    | T3    | T3    | T2             | T0             |
>
> Tier-0 = identity (6); Tier-1 = lossless via `From` 或 zero-imaginary widening (8); Tier-2 = lossy 静态失败 (14); Tier-3 = 动态条件 (8)。合计 36，与上方 impl 列表一一对应。

> **同型 `cast::<A>()` 语义澄清（与 §5.5 协同）**：`§5.5` 说"`cast::<A>()` 不适用于同类型拷贝场景"——这是**使用建议**，不是编译期禁止；技术上 Tier-0 identity impls (`ConvertTo<f32> for f32` 等) 让 `Tensor<f32>.cast::<f32>()` 编译通过并返回 `Ok(self)`。同型 cast 在功能上等价于 `to_owned()`，但走 `cast()` 路径会引入 fallible signature 与 enumerate 开销，因此推荐使用 `to_owned()`。Tier-0 identity impls 存在的真正目的是让 `ConvertTo<B>` 的 trait bound 在泛型上下文中也能覆盖 `A == B` 边界，避免上层调用方为同型/异型分支两条路径。

设计要点：

- **零运行时开销**：每个 `impl ConvertTo<B> for A` 都是 `#[inline]` 的 thin shim，编译器单态化后等同于直接 `From` / `CastTo` 调用。
- **bound 与 Tier-1 兼容**：`cast<B>()` 只要求 `A: ConvertTo<B>`，Tier-1 type pair（`f32→f64` 等）有 `ConvertTo` impl 但**没有** `CastTo` impl，编译通过。
- **错误形态统一**：`ConvertTo::convert` 永远返回 `Result<B, XenonError>`；Tier-1 必为 `Ok`，Tier-2/Tier-3 与 `CastTo::cast_to` 一致。
- **sealed**：通过 `crate::private::Sealed` 阻止外部 crate 实现 `ConvertTo`，与 `CastTo` 的 sealed 方式一致。

### 6.2 溢出行为汇总

`cast()` 是 fallible API。凡被 `需求说明书 §23` 判定为有损的转换，默认返回 `XenonError::TypeConversion`；仅在该节明确给出额外成功前提时，满足前提后方可成功。

| 输入值/组合                    | 目标类型 | 结果                  | 说明                   |
| ------------------------------ | -------- | --------------------- | ---------------------- |
| `f64::NAN`                     | `i32`    | `Err(TypeConversion)` | 浮点到整数属于有损转换 |
| `f64::INFINITY`                | `i32`    | `Err(TypeConversion)` | 不提供饱和语义         |
| `i64::MAX`                     | `i32`    | `Err(TypeConversion)` | 不提供截断或饱和       |
| `Complex { re: 1.0, im: 0.0 }` | `f64`    | `Ok(1.0)`             | 满足附加成功前提       |
| `Complex { re: 1.0, im: 2.0 }` | `f64`    | `Err(TypeConversion)` | 虚部非零               |

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 实现 `CastTo` trait 的 Tier-2 / Tier-3 转换路径（Tier-1 不经过 `CastTo`）
  - 文件: `src/convert/cast.rs`
  - 内容: 实现 `element` 模块中的 fallible `CastTo<T>` trait 的 Tier-2 lossy（静态错误，14 cells）+ Tier-3 dynamic（条件性，8 cells）两层；Tier-1 lossless（11 cells: 6 identity + 3 std `From` + 5 real→complex zero-imaginary widening + 1 complex widening）由 `ConvertTo` 直接走 std `From` / direct shims，**不**实例化 `CastTo`，详见 §6.1 / §6.1.ter
  - 测试: `test_cast_f32_to_f64`（Tier-1，验证 `ConvertTo` 直通不实例化 `CastTo`）、`test_cast_f64_to_f32`（Tier-2 lossy）、`test_cast_complex_f64_to_f64_when_imag_zero`（Tier-3 dynamic）
  - 前置: element 模块完成
  - 预计: 10 min

- [ ] **T2**: 创建 `convert/` 模块骨架
  - 文件: `src/convert/mod.rs`, `src/lib.rs`
  - 内容: 子模块声明、`pub use` 导出
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: 核心方法

- [ ] **T3**: 实现 `to_owned` / `into_owned`
  - 文件: `src/convert/cast.rs`（或与 `cast()` 共置的同模块实现文件）
  - 内容: `to_owned()` 克隆方法与 `into_owned()` 消费方法；不在本模块扩展 view/view_mut/into_shared 等额外存储模式互转入口，也不把同类型 owned 化表述为独立“存储模式互转”任务
  - 测试: `test_to_owned_from_view`, `test_into_owned_from_tensor`, `test_into_owned_from_arc`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

- [ ] **T4**: 实现 `cast` 方法
  - 文件: `src/convert/cast.rs`
  - 内容: `cast<B>(&self) -> Result<Tensor<B, D>, XenonError>` 方法实现，覆盖所有可读存储输入并统一产出 owned 结果
  - 测试: `test_cast_f64_to_f32_returns_error`, `test_cast_i32_to_f64`, `test_cast_reports_element_index`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

- [ ] **T5**: 扩展 CastTo 实现（整数↔整数、实数↔复数、复数↔复数）
  - 文件: `src/convert/cast.rs`
  - 内容: 补齐 `需求说明书 §23.1` 与 `需求说明书 §23.2` 定义的全部组合；`bool` 不参与
  - 测试: `test_cast_real_to_complex`, `test_cast_complex_to_real_requires_zero_imag`, `test_cast_complex_f64_to_complex_f32_returns_error`
  - 前置: T1
  - 预计: 10 min

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                                             |
| -------- | ------------------------ | -------------------------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证类型转换、错误路径、owned 化与借用转换语义                                   |
| 集成测试 | `tests/`                 | 验证 `convert` 与 `tensor`、`element`、`storage`、`layout`、`complex` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖空张量、NaN/Inf、有损转换报错、复数虚部约束和非连续视图等边界                |
| 属性测试 | `tests/property/`        | 验证 cast/to_owned 保持 shape 与转换规则不变量                                   |

### 8.2 单元测试清单

| 测试函数                                                        | 测试内容                           | 优先级 |
| --------------------------------------------------------------- | ---------------------------------- | ------ |
| `test_cast_f64_to_f32_returns_error`                            | f64→f32 默认返回 `TypeConversion`  | 高     |
| `test_cast_f32_to_f64`                                          | f32→f64 无损转换                   | 高     |
| `test_cast_float_to_int_returns_error`                          | 浮点→整数默认返回 `TypeConversion` | 高     |
| `test_cast_nan_to_int_returns_error`                            | NaN → 整数返回错误                 | 高     |
| `test_cast_inf_to_int_returns_error`                            | ±Inf → 整数返回错误                | 高     |
| `test_cast_int_narrowing_returns_error`                         | 整数窄化默认返回错误               | 高     |
| `test_cast_real_to_complex`                                     | 实数→复数虚部为 0                  | 中     |
| `test_cast_complex_to_real_requires_zero_imag`                  | 仅在虚部为 0 时成功                | 高     |
| `test_cast_complex_to_int_requires_zero_imag_and_inner_success` | 复数到整数复合前提                 | 高     |
| `test_cast_reports_element_index`                               | 错误包含失败元素索引               | 高     |
| `test_to_owned_from_view`                                       | View → Owned 数据一致              | 高     |
| `test_to_owned_from_arc`                                        | Arc → Owned 正确复制               | 高     |
| `test_into_owned_tensor`                                        | Owned → Owned 零拷贝               | 高     |

### 8.3 边界测试场景

| 场景                                 | 预期行为                                        |
| ------------------------------------ | ----------------------------------------------- |
| 空张量 `cast`                        | 返回空张量，形状不变                            |
| 单元素无损 `cast`                    | 成功并保持形状                                  |
| 非连续 View `cast`                   | 直接 `cast()`，结果正确且保持形状               |
| `i64::MAX → i32`                     | 返回 `TypeConversion`                           |
| `f64::NAN → i32`                     | 返回 `TypeConversion`                           |
| `Complex { re: 1.0, im: 0.0 } → f64` | 成功                                            |
| `Complex { re: 1.0, im: 2.0 } → f64` | 返回 `TypeConversion`                           |

### 8.4 属性测试不变量

| 不变量                                | 测试方法         |
| ------------------------------------- | ---------------- |
| `cast()?.shape() == original.shape()` | 随机形状         |
| 所有无损组合逐元素值保持不变          | 随机数据         |
| 所有有损组合默认失败                  | 按类型对枚举验证 |
| `to_owned().shape() == view.shape()`  | 随机形状         |

### 8.5 集成测试

| 测试文件                   | 测试内容                                                                                                   |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `tests/test_conversion.rs` | `cast` / `to_owned` / `into_owned` 与 `tensor`、`element`、`storage`、`layout`、`complex` 的端到端协同路径 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | `cast` / `to_owned` / `into_owned` 在默认构建下保持显式转换与错误诊断契约。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `bool` 不参与 `cast()`，且不属于 `CastElement` | 编译期测试。 |
| `Tensor<bool, _>.cast::<T>()` 作为源类型被拒绝 | compile-fail：验证 `bool`/`BoolElement` 不能作为 `cast()` 的源元素类型。 |
| `Tensor<usize, _>.cast::<T>()` 与 `Tensor<T, _>.cast::<usize>()` 都被拒绝 | compile-fail：验证 `usize` 不属于 `CastElement`，不会被闭合规则意外放入矩阵。 |
| `cast()` 对所有可读存储提供，但统一返回 owned 结果 | 编译期测试。 |
| saturation / truncation casts 与额外 `From/Into` 非张量转换不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块  | 接口/类型                               | 约定                                                                                              |
| ------------------- | --------- | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `convert → tensor`  | `tensor`  | `TensorBase<S, D>` / `StorageIntoOwned` | `cast()`、`to_owned()`、`into_owned()` 都定义在张量抽象之上；其中 `to_owned()` / `into_owned()` 负责产出 canonical F-order owned 结果 |
| `convert → element` | `element` | `ConvertTo` (internal) / `CastTo`       | 逐元素类型转换通过内部 `ConvertTo<B>` 分流：Tier-1 lossless 走 std `From` / direct shims（**不**实例化 `CastTo`）；Tier-2 lossy 与 Tier-3 dynamic 委托 `<A as CastTo<B>>::cast_to(value)`。`CastTo` trait 定义见 `03-element.md §5.9`，三层架构详见 §6.1 / §6.1.ter |
| `convert → math`    | `math`    | 逐元素转换语义                          | `cast()` 采用迭代收集路径，不复用 `mapv()` 的同类型返回语义                                       |
| `convert → storage` | `storage` | `Owned` / readable storage traits       | convert 只消费可读存储与 owned 化能力，不在本文扩展额外存储模式互转矩阵                           |
| `convert → utility` | `utility`  | `to_contiguous`, `into_contiguous`   | 外部调用方若需要显式连续化入口，由 `util::to_contiguous()` 负责（参见 `20-utility.md §5.5`） |
| `convert → complex` | `complex` | `Complex<T>`                            | 复数目标类型转换依赖 `Complex` 定义；Complex → 实数默认为错误（虚部非 0 时返回 `NonZeroImaginaryPart`），仅在 `im == 0` 且内层实数转换本身无损时可成功，参见 `04-complex.md` §5 |

### 9.2 数据流描述

```text
User calls cast() / to_owned() / into_owned()
    │
    ├── convert reads tensor shape / strides / storage mode metadata
    ├── cast collects elements and re-encodes them via ConvertTo dispatch
    │     (Tier-1 lossless via std `From` / direct shims; Tier-2/Tier-3 via CastTo)
    ├── owned-conversion paths choose explicit O(1) transfer or O(n) copy by source storage mode
    ├── ArcRepr → Owned always allocates and copies (O(n))
    └── the module returns a new owned tensor
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `cast()` 在有损转换、虚部非零或其他规则不满足时返回 `XenonError::TypeConversion { operation: Cow::Borrowed("cast"), source_type: &'static str, target_type: &'static str, reason: ConversionFailureReason, element_index: Some(usize) }`（字段定义见 `26-error.md v3.2.0 §5.1`）。源/目标类型字段为 `&'static str`，值由 `<A as Element>::ELEMENT_TYPE_NAME` 提供（`03-element.md §5.1.1`），**不**使用 `core::any::TypeId`，**也不**直接持有 `ElementType` 枚举值。`element_index` 为按逻辑元素遍历顺序的 0-based 线性索引，非多维索引；`CastTo::cast_to()` 自身不知道线性索引，因此其返回的错误中 `operation = Cow::Borrowed("cast_to")`（**稳定非空操作名**，与 `26-error.md §8.2` `test_type_conversion_carries_operation` 的非空契约一致；空字符串被禁止，参见 §6.1 / `04-complex.md §10`），`element_index` 为 `None`，由 `cast()` 在 `map_err` 中将 `operation` 覆盖为 `Cow::Borrowed("cast")` 并注入实际索引（见 §5.2 实现）。 |
| Panic | 公开转换 API 不定义额外 panic 语义；有损场景统一返回可恢复错误。 |
| 路径一致性 | `cast`、`to_owned`、`into_owned` 必须保持相同 shape 与逻辑元素顺序；其中 `to_owned` / `into_owned` 的 owned 结果固定为 canonical F-order。无 SIMD / 并行分支。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### 决策 1：默认失败而非饱和/截断

| 属性     | 值                                                                        |
| -------- | ------------------------------------------------------------------------- |
| 决策     | 所有有损转换默认返回 `XenonError::TypeConversion`                         |
| 理由     | 这是 `需求说明书 §23` 的强制要求；文档不得私自引入饱和、截断或 NaN→0 语义 |
| 替代方案 | saturating / truncating — 放弃，与需求冲突                                |
| 替代方案 | panic on overflow — 放弃，需求要求可恢复错误                              |

### 决策 2：cast() 对所有可读存储开放

| 属性     | 值                                                                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | `cast()` 对所有可读存储开放，并统一返回 owned tensor                                                                                              |
| 理由     | 这与 `需求说明书 §23` 的逐元素转换要求一致，同时避免把“能否读取输入”与“结果是否拥有数据”混为一谈；输入可借用，结果仍统一 owned，API 语义保持单一。|
| 替代方案 | 仅在 `Owned` 上实现 — 放弃，会无依据地缩小 `需求说明书 §23` 的适用范围                                                                            |
| 替代方案 | 按输入存储模式返回不同结果类型 — 放弃，会引入生命周期与所有权分歧，破坏公开 API 一致性                                                            |

### 决策 3：收缩 convert 模块边界到当前需求集合

| 属性     | 值                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| 决策     | `convert/` 的核心覆盖面收敛到 `cast()` / `CastTo`；`to_owned()` / `into_owned()` 仅作为同模块便利 API 保留，其余存储模式互转仅作跨文档引用，不在本文展开 |
| 理由     | 当前 `需求说明书 §23` 只要求逐元素类型转换与同类型拷贝；以 cast 作为模块核心可保持边界清晰，同时允许便利 API 复用同一基础设施而不把文档扩展到完整存储模式互转 |
| 替代方案 | 在本文继续完整展开所有存储模式互转 — 放弃，会把 convert 文档扩展到非本节需求范围 |

### 决策 4：默认无损成功、有损 TypeConversion 错误、逐元素检查（B10.a）

| 属性     | 值                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | **三层架构（v2.1.1）**：(Tier-1) 静态无损通过 Rust 标准 `From` / `Into` 表达（`f64: From<f32>`、`i64: From<i32>`、`f64: From<i32>`），**不**经过 `CastTo`，`cast()` 主循环对 Tier-1 直接 `T::from(value)` 零开销；(Tier-2) 静态有损在 `impl CastTo<T> for U` 中直接 `Err(TypeConversion)`，不尝试运行时值域检查；(Tier-3) 动态条件性（如 `Complex<T> → T` 仅当 `im == 0.0`）在 `CastTo::cast_to()` 内做逐元素运行时判定 |
| 理由     | (1) `需求说明书 §23` 要求"有损转换默认失败"，把判定下推到类型对级别（即 `impl CastTo<i32> for f64` 一律 `Err`）即可满足；(2) 不需要为每个 i64 值检查 `±2^53` 边界（这超出了当前需求；§5.3 中已标注 i64→f64 待需求方确认）；(3) 唯一需要逐元素检查的场景是 `Complex → Real`：`im == 0.0` 是动态条件，必须运行时判定 — 这部分实现已在 §6.1 `CastTo<f64> for Complex<f64>` 中正确给出 |
| 替代方案 | 默认对所有有损转换尝试动态饱和/截断 — 放弃，与 `需求说明书 §23` 冲突                                                                |
| 替代方案 | 对 `i64 → f64` 默认成功（数学上窄化但 IEEE 754 round-to-nearest）— 暂保守归类有损，§5.3 已标注待需求方决定                         |
| 替代方案 | 在 `cast()` 主循环中对每个元素跑值域检查 — 放弃，多数有损 type pair（如 `f64 → f32`）静态可判失败，无需逐元素扫描                   |

---

## 12. 性能考量

| 操作                  | 时间复杂度 | 空间复杂度 | 说明                             |
| --------------------- | ---------- | ---------- | -------------------------------- |
| `cast`                | O(n)       | O(n)       | 任意可读输入均物化为新 owned 张量 |
| `to_owned`            | O(n)       | O(n)       | 总是拷贝                         |
| `into_owned`（Owned） | O(1)       | O(1)       | 直接返回                         |
| `into_owned`（View）  | O(n)       | O(n)       | 拷贝                             |
| `into_owned`（Arc）   | O(n)       | O(n)       | 总是分配并复制                   |

---

## 13. 平台与工程约束

| 项目       | 约束                                                      |
| ---------- | --------------------------------------------------------- |
| 平台       | 仅 `std`                                                  |
| MSRV       | Rust 1.85+                                                |
| crate 结构 | 单 crate                                                  |
| 最小依赖   | 不新增第三方依赖                                          |
| 错误语义   | 所有执行路径都须保持同一 `Result` / `TypeConversion` 契约 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
