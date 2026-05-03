# 类型转换模块设计

> 文档编号: 21
> 模块目录: src/convert/
> 任务阶段: Phase 4
> 前置文档: 07-tensor.md, 03-element.md
> 需求参考: 需求说明书 §23、§27、§28
> 范围声明: 范围内

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
| 失败可诊断 | 有损转换默认返回可恢复错误，错误上下文由 `XenonError::TypeConversion` 承载（字段对齐 `26-error.md v3.2.0 §5.1`，`source_type` / `target_type` 字段类型为 `&'static str`，值来自 `<A as Element>::ELEMENT_TYPE_NAME`；不使用 `TypeId`，也不直接持有 `ElementType` 枚举值） |
| 存储约束   | `cast` 面向所有可读存储开放，但结果统一物化为 owned 张量                   |
| 需求闭合   | 仅支持 `需求说明书 §23.1` 与 `需求说明书 §23.2` 定义的类型对及其成功前提   |
| 静态分流   | 决策 4（B10.a）：无损/默认有损在类型对级别静态判定，不做逐元素扫描；仅 `Complex → Real` 等条件性成功才在 `cast_to()` 内逐元素判定（如 `im == 0.0`）|

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

| 来源模块    | 使用的类型/trait                                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| `tensor`    | `TensorBase<S, D>`, `Tensor<A, D>`, `.shape()`, `.strides()`, `.is_f_contiguous()`（参见 `07-tensor.md` §5） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`（参见 `02-dimension.md` §5）                                               |
| `storage`   | `Storage<Elem=A>`, `StorageMut`, `Owned<A>`, `ViewRepr`, `ViewMutRepr`, `ArcRepr`（参见 `05-storage.md` §5） |
| `element`   | `Element`, `CastTo<B>`（参见 `03-element.md` §5.9）, `CastElement`（sealed marker，参见 `03-element.md §5.9.1`）。convert 只消费这些，不重新定义 |
| `layout`    | `is_f_contiguous()`（参见 `06-layout.md` §5）                                                                |
| `error`     | `XenonError`（含 `TypeConversion::source_type` / `target_type: &'static str`，v3.2.0 起；详见 `26-error.md §5.4`）、`Result<T>`、`ConversionFailureReason`。本模块**不**通过 error 间接消费 `ElementType`——`ElementType` 类型权威定义在 `crate::element`（`03-element.md §5.1.1`），如需本类型本模块直接 `use crate::element::ElementType;`；如需类型名字符串，使用 `<A as Element>::ELEMENT_TYPE_NAME` 或 `crate::element::element_type_name_of::<A>()` |
| `iter`      | `iter()` 用于 `cast()` / `to_owned()` 的逐元素遍历（参见 `10-iterator.md` §5）                             |

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
        //     ├── Tier-2 (lossy static):    impl returns
        //     │   `Err(<A as CastTo<B>>::cast_to(value))` — cast_to() always
        //     │   produces `Err(TypeConversion { reason: ... })` for static
        //     │   lossy pairs.
        //     └── Tier-3 (dynamic):         impl returns
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

- `cast()` / `to_owned()` 在实现上可以复用张量或存储层的内部构造 helper，但这些 helper 的命名、文件布局、是否存在 unchecked 变体以及具体对齐策略，都不属于 convert 模块的稳定文档面。
- `cast()` / `to_owned()` 通过 `pub(crate)` 内部 helper 从已验证的 shape/data 长度构造 owned 结果。该 helper 形态如 `from_shape_vec_aligned_unchecked`，**是 `pub(crate) unsafe fn`**（不是 safe 函数）；调用点必须用 `unsafe { ... }` 块包裹，且每个调用点必须挂 `// SAFETY: ...` 注释说明 `shape` 已验证元素总数等于 `data.len()`、由 `shape` 推导出的 F-order 元数据在当前版本范围内合法（无 stride 溢出、无 offset 越界、无非法零步长来源）。
- helper 名称中的 `unchecked` 严格表示"跳过可由调用方安全封装重复检查的 metadata 校验"，**不**表示"内部实现可放任任意输入"——任何错误的 `(shape, data.len())` 配对仍会构成 UB。底层使用哪一种分配器或对齐值，不应写入该 safety 契约（这部分由 storage 层的 `Owned::from_vec_aligned` 自行决定）。
- 之所以让 helper 保留 `unsafe` 而不是把它做成 safe 函数（再在内部 panic 检查），是为了让 `cast()` / `to_owned()` 的 infallible 签名真正零额外检查开销；safe wrapper 路径已由 `Tensor::from_shape_vec` 提供（fallible，返回 `Result`）。

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

// All TypeConversion errors below leave `operation` empty (Cow::Borrowed(""))
// and `element_index = None`; the caller (cast() in §5.2) is responsible for
// injecting `operation = Cow::Borrowed("cast")` and the resolved element
// index. See §5.2 cast() for the rewrap pattern.
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
            operation: Cow::Borrowed(""),
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
            operation: Cow::Borrowed(""),
            source_type: <f64 as Element>::ELEMENT_TYPE_NAME, // "f64"
            target_type: <i32 as Element>::ELEMENT_TYPE_NAME, // "i32"
            reason: ConversionFailureReason::FloatToInteger,
            element_index: None,
        })
    }
}

// === Real → complex (lossless, zero imaginary) ===
impl CastTo<Complex<f32>> for f32 {
    #[inline]
    fn cast_to(self) -> Result<Complex<f32>, XenonError> {
        Ok(Complex::new(self, 0.0))
    }
}

// === Complex → complex (lossless widening) ===
impl CastTo<Complex<f64>> for Complex<f32> {
    #[inline]
    fn cast_to(self) -> Result<Complex<f64>, XenonError> {
        Ok(Complex::new(self.re as f64, self.im as f64))
    }
}

// === Conditionally successful conversions ===
impl CastTo<f64> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<f64, XenonError> {
        if self.im == 0.0 {
            Ok(self.re)
        } else {
            Err(XenonError::TypeConversion {
                operation: Cow::Borrowed(""),
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
            operation: Cow::Borrowed(""),
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

下列 `From` impls **不**通过 `CastTo`，由 Rust 标准库（`f64: From<f32>`、`i64: From<i32>`、`f64: From<i32>`、`f64: From<u32>` 等）直接提供。`cast()` 主循环对 Tier-1 type pair 的实现委托：

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

// === Tier-2 / Tier-3: lossy / dynamic. Delegate to CastTo. ===
// Macro-generated: `impl<A, B> ConvertTo<B> for A where A: CastTo<B>`
// won't work due to coherence with the lossless impls above; instead, each
// remaining type pair gets its own forwarding impl:
impl ConvertTo<f32> for f64 {
    #[inline] fn convert(self) -> Result<f32, XenonError> { <f64 as CastTo<f32>>::cast_to(self) }
}
impl ConvertTo<i32> for f64 {
    #[inline] fn convert(self) -> Result<i32, XenonError> { <f64 as CastTo<i32>>::cast_to(self) }
}
// ... similarly for every Tier-2/Tier-3 (A, B) pair documented in §5.3.
```

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

- [ ] **T1**: 实现 `CastTo` trait 的核心转换路径
  - 文件: `src/convert/cast.rs`
  - 内容: 复用 `element` 模块中的 fallible `CastTo<T>` trait，实现无损与默认错误路径
  - 测试: `test_cast_f32_to_f64`, `test_cast_i32_to_i64`, `test_cast_complex_f64_to_f64_when_imag_zero`
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
| `convert → element` | `element` | `CastTo`                                | 逐元素类型转换通过 `CastTo` trait 驱动，参见 `03-element.md` §5.9                                 |
| `convert → math`    | `math`    | 逐元素转换语义                          | `cast()` 采用迭代收集路径，不复用 `mapv()` 的同类型返回语义                                       |
| `convert → storage` | `storage` | `Owned` / readable storage traits       | convert 只消费可读存储与 owned 化能力，不在本文扩展额外存储模式互转矩阵                           |
| `convert → utility` | `utility`  | `to_contiguous`, `into_contiguous`   | 外部调用方若需要显式连续化入口，由 `util::to_contiguous()` 负责（参见 `20-utility.md §5.5`） |
| `convert → complex` | `complex` | `Complex<T>`                            | 复数目标类型转换依赖 `Complex` 定义；Complex → 实数默认为错误（虚部非 0 时返回 `NonZeroImaginaryPart`），仅在 `im == 0` 且内层实数转换本身无损时可成功，参见 `04-complex.md` §5 |

### 9.2 数据流描述

```text
User calls cast() / to_owned() / into_owned()
    │
    ├── convert reads tensor shape / strides / storage mode metadata
    ├── cast collects elements and re-encodes them via CastTo rules
    ├── owned-conversion paths choose explicit O(1) transfer or O(n) copy by source storage mode
    ├── ArcRepr → Owned always allocates and copies (O(n))
    └── the module returns a new owned tensor
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `cast()` 在有损转换、虚部非零或其他规则不满足时返回 `XenonError::TypeConversion { operation: Cow::Borrowed("cast"), source_type: &'static str, target_type: &'static str, reason: ConversionFailureReason, element_index: Some(usize) }`（字段定义见 `26-error.md v3.2.0 §5.1`）。源/目标类型字段为 `&'static str`，值由 `<A as Element>::ELEMENT_TYPE_NAME` 提供（`03-element.md §5.1.1`），**不**使用 `core::any::TypeId`，**也不**直接持有 `ElementType` 枚举值。`element_index` 为按逻辑元素遍历顺序的 0-based 线性索引，非多维索引；`CastTo::cast_to()` 自身不知道线性索引，因此其返回的错误中 `operation` 留空、`element_index` 为 `None`，由 `cast()` 在 `map_err` 中注入正确值（见 §5.2 实现）。 |
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

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-08 |
| 1.2.2 | 2026-04-10 |
| 1.2.3 | 2026-04-14 |
| 1.2.4 | 2026-04-15 |
| 1.2.5 | 2026-04-15 |
| 1.2.6 | 2026-04-15 |
| 2.0.0 | 2026-05-02 |
| 2.1.0 | 2026-05-03 |
| 2.1.1 | 2026-05-03 |

### v2.1.1 (2026-05-03) — TypeConversion 字段类型反转为 &'static str

> 本版本与 `26-error.md v3.2.0`、`03-element.md v1.4.0` 协同。**ABI 兼容**：用户通过 `Result<T, XenonError>` 接收错误的形态不变；变化是错误字段值类型。

- §4.2：`error` 依赖行修正为不再通过 error re-export 间接消费 `ElementType`；本模块如需类型枚举本身用 `use crate::element::ElementType;`，如需类型名字符串用 `<A as Element>::ELEMENT_TYPE_NAME` 或 `crate::element::element_type_name_of::<A>()`。
- §5.2 `cast()` doc comment `# Errors` 段更新：`source_type` / `target_type` 字段类型从 `ElementType` 改为 `&'static str`，值由 `<A as Element>::ELEMENT_TYPE_NAME` 提供。
- §6.1 全部 13 处 `CastTo` 实现示例：错误构造 `source_type: ElementType::F64` → `source_type: <f64 as Element>::ELEMENT_TYPE_NAME`，所有数值类型对均同步。
- §10 错误处理表：`source_type` / `target_type` 类型说明改为 `&'static str`。
- 设计权衡说明：error 模块不直接依赖 element（不持有 `ElementType` 枚举），通过 `&'static str` 解耦；`ElementType` 类型枚举权威定义同时回归 `crate::element`（`03-element.md §5.1.1`）以让职责归属对齐。

### v2.1.0 (2026-05-03) — CastElement owner 协同 + ElementType 重新定位

> v2.1.0 阶段曾让 `ElementType` 权威定义下沉到 `crate::error`（用以让 error 自带枚举 Display）；该决策已在 v2.1.1 / `26-error.md v3.2.0` 反转——`ElementType` 重新由 `crate::element` 拥有，error 模块改用 `&'static str` 字段。本节其他内容保留，仅 `ElementType` 路径相关说明已被 v2.1.1 取代。

- §5.1：明确 `CastElement` 的唯一 owner 是 `03-element.md §5.9.1`；本模块只通过 `use crate::element::CastElement;` 消费，不再在本文档中重复展开 trait 定义与 impl 列表（解决 P0 C1 修复任务中的"CastElement owner 缺失"项）。
- ~~§6.1 / §10：`ElementType` 引用提示更新——v3.1.0 起权威定义在 `26-error.md §5.1`，`crate::element::ElementType` 是通过 `pub use` re-export 的稳定上层路径~~（**v2.1.1 反转**：`ElementType` 权威定义在 `crate::element`，详见 `03-element.md §5.1.1`；error 模块字段类型改为 `&'static str`）。

### v2.0.0 (2026-05-02) — 错误字段对齐 26-error v3.0.0 + B10.a 决策落地

> 本版本与 `26-error.md v3.0.0` 协同更新；属于内部错误结构的破坏性调整（公开 API 形态保持兼容，调用方仍通过 `Result<T, XenonError>` 处理错误，但 `XenonError::TypeConversion` 的字段构造方式已改变）。

**契约更新**：

- §5.2 `cast()` doc comment `# Errors` 段重写：完整列出 `TypeConversion` 字段（`operation: Cow<'static, str>`、`source_type: ElementType`、`target_type: ElementType`、`reason`、`element_index: Some(usize)`），明示 `source_type/target_type` 是 `ElementType` 封闭枚举，**禁止**使用 `core::any::TypeId`。
- §5.2 `cast()` 函数体重写：`map_err` 闭包注入 `operation: Cow::Borrowed("cast")`；尾部通过 `pub(crate)` 内部 helper 从已验证的 shape/data 长度构造 owned 结果，helper 不作为 convert 稳定文档面。
- §5.5 `to_owned()` 函数体重写：从 fallible 的 `Tensor::from_shape_vec_aligned(self.raw_dim(), data)` 改为 `pub(crate)` 内部 helper `Tensor::from_shape_vec_aligned_unchecked(self.raw_dim(), data)`，配合 doc comment 中"shape/data 长度一致性已构造期保证"的论证，让 `to_owned()` 保持 infallible 签名。
- §6.1 `CastTo` 实现示例完全重写：把 `core::any::TypeId::of::<T>()` 替换为 `ElementType::F64`、`ElementType::I32`、`ElementType::Complex64` 等封闭枚举值；为每个 `Err(TypeConversion {..})` 添加 `operation: Cow::Borrowed("")` 占位字段（`cast()` 的 `map_err` 会注入 "cast"）；移除 `use core::any::TypeId;`，改为 `use crate::element::ElementType;` + `use crate::error::ConversionFailureReason;`。

**协同与一致性更新**：

- §1.2 设计原则表新增"静态分流"一行：明确标注决策 4（B10.a）落地——无损/默认有损在类型对级别静态判定，仅条件性成功（`Complex → Real` 的 `im == 0.0`）才逐元素动态判定。
- §1.2 "失败可诊断"一行补充"字段对齐 26-error v3.0.0 §5.1，使用 `ElementType` 而非 `TypeId`"。
- §4.2 类型级依赖表更新：`element` 行从 §5.8（Sealed trait 策略）修正为 §5.9（CastTo<T>），并补充 `ElementType` 标签依赖；`error` 行展开为 `XenonError`、`Result<T>`、`ConversionFailureReason`、`ElementType`；新增 `iter` 行（`cast()` / `to_owned()` 都通过 `self.iter()` 遍历）。
- §10 错误处理表 `Recoverable error` 一行重写：完整列出 `TypeConversion` 五字段；明示 `CastTo::cast_to()` 自身 emits `operation` 留空 + `element_index = None`，由 `cast()` 在 `map_err` 中注入。
- §11 新增决策 4：完整论证 B10.a 决策——三层结构（静态无损 / 静态有损 / 动态条件性）+ 拒绝替代方案的理由（拒绝默认饱和、拒绝 `i64 → f64` 默认成功、拒绝 `cast()` 主循环逐元素扫描）。
- §5.3 / §5.4 / §6.1 移除 `i64 → f64` 待确认文案，明确按 B10.a 选定为有损默认失败；Complex→Real 表项改为“条件成功”；§5.2 / §5.6 统一 `cast()` 的 `pub(crate)` 内部 helper 边界；修正 T5 任务缩进。

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
