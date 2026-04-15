# 类型转换模块设计

> 文档编号: 21 | 模块: `src/convert/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `03-element.md`
> 需求参考: 需求说明书 §23
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                                 | 不包含                                                 |
| -------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| 逐元素类型转换 | `cast<B: CastElement>(&self) -> Result<Tensor<B, D>, XenonError>` | 隐式类型提升（需显式调用）                             |
| 同类型拷贝     | `to_owned`、`into_owned`                             | 标准库 `From`/`Into` 实现（归构造模块）                |
| 范围边界       | §23 要求的逐元素类型转换与同类型拷贝                 | 存储模式互转（归 `storage` / `tensor`）、连续化 helper（归 `utility`） |

### 1.2 设计原则

| 原则       | 体现                                                                       |
| ---------- | -------------------------------------------------------------------------- |
| 显式转换   | 所有类型转换须显式调用 `cast()`，无隐式提升                                |
| 失败可诊断 | 有损转换默认返回可恢复错误，错误上下文由 `XenonError::TypeConversion` 承载 |
| 存储约束   | `cast` 面向所有可读存储开放，但结果统一物化为 owned 张量                                           |
| 需求闭合   | 仅支持 `require.md §23.1` 与 `§23.2` 定义的类型对及其成功前提              |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned and consumed by tensor)
L4: tensor (depends on storage, dimension)
L5: broadcast, iter, ffi
L6: math, matrix, reduction, shape, index, util
L7: convert  ← current module
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §23 |
| 范围内   | `cast()` 逐元素类型转换，以及 `to_owned()` / `into_owned()` 同类型拷贝。 |
| 范围外   | 存储模式互转（归 `storage` / `tensor`）、标准库 `From` / `TryFrom` 实现（归构造模块）、连续化 helper（归 `utility`），以及超出需求矩阵的隐式转换。 |
| 非目标   | 不默认放宽有损转换规则，不新增第三方转换库，也不在本模块新增独立的连续化 API。 |

---

## 3. 文件位置

```
src/
└── convert/                 # Type conversion module
    ├── mod.rs               # Module root, re-exports
    ├── cast.rs              # cast() methods and conversion paths consuming element::CastTo
    └── owned.rs             # to_owned, into_owned
```

多文件设计：按转换职责拆分，便于后续扩展（如新增转换路径、存储模式等）。

### 3.1 文件职责

| 文件            | 职责                                                                      | 预估行数 |
| --------------- | ------------------------------------------------------------------------- | -------- |
| `mod.rs`        | 模块根，re-exports 所有公共类型                                           | ~20      |
| `cast.rs`       | 消费 `CastTo<T>` trait，提供所有类型转换 impl 与 `cast()` 方法            | ~200     |
| `owned.rs`      | `to_owned()`、`into_owned()`，并记录同类型拷贝的 owned 结果语义           | ~100     |

---

## 4. 依赖关系

### 4.1 依赖图

```
src/convert/
├── mod.rs          # Re-exports: CastTo, cast, to_owned, into_owned
├── cast.rs         # Depends on element (CastTo) and tensor (TensorBase)
├── owned.rs        # Depends on tensor (TensorBase), storage, layout

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
| `element`   | `Element`, `CastTo<B>`（参见 `03-element.md` §5.9；convert 只消费该 trait，不重新定义）                      |
| `layout`    | `is_f_contiguous()`（参见 `06-layout.md` §5）                                                                |
| `error`     | `XenonError`, `Result<T>`（参见 `26-error.md` §4）                                                           |

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `convert` 仅消费 `tensor`、`storage` 等核心模块，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.1 CastTo trait

> `CastTo<T>` trait 的唯一 owner 是 `03-element.md §5.9`。`convert` 模块只消费该 trait，并在受支持的源/目标类型矩阵上提供 `cast()` 路径，不重新定义 trait。

````rust,ignore
pub trait CastElement: Element + private::Sealed {}

impl CastElement for i32 {}
impl CastElement for i64 {}
impl CastElement for f32 {}
impl CastElement for f64 {}
impl CastElement for Complex<f32> {}
impl CastElement for Complex<f64> {}
````

> `CastElement` 为封闭 trait，下游不得扩展。`bool` 不属于 `CastElement`，因此 `cast::<bool>()` 在编译期被拒绝。

### 5.2 cast 方法

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
    /// Returns `XenonError::TypeConversion(TypeConversionError)` when any element cannot be converted
    /// under the rules defined in `require.md §23`.
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
        A: CastTo<B>,
    {
        let mut data: Vec<B> = Vec::with_capacity(self.len());
        for (index, x) in self.iter().enumerate() {
            let value = (*x).cast_to().map_err(|err| match err {
                XenonError::TypeConversion(err) => {
                    XenonError::TypeConversion(TypeConversionError {
                        source_type: err.source_type,
                        target_type: err.target_type,
                        reason: err.reason,
                        element_index: index,
                    })
                }
                other => other,
            })?;
            data.push(value);
        }
        Ok(Tensor::from_shape_vec_aligned(self.raw_dim(), data))
    }
}
````

> **设计决策（修订）：** `require.md §23` 要求的是逐元素转换语义，而不是“仅限 Owned 输入”。因此 `cast()` 面向所有可读存储开放；无论输入是 `Owned`、`ViewRepr`、`ViewMutRepr` 还是 `ArcRepr`，结果统一物化为新的 owned 张量，以保持返回类型与所有权语义一致。源类型与目标类型都进一步收缩为 `CastElement`，从签名层面排除 `bool`。

> **bool 源类型边界：** `cast<B>()` 仅在 `A: CastElement + CastTo<B>` 时可用。`bool` 不实现 `CastElement`，因此 `Tensor<bool, _>` 上 `cast()` 在编译期不可调用，而不是落到运行时 `TypeConversion`。

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
| `i64`          | `f64`          | 错误     | 有损，默认失败                              |
| `i32`          | `Complex<f32>` | 错误     | 由 `i32 -> f32` 有损导致默认失败            |
| `i64`          | `Complex<f64>` | 错误     | 由 `i64 -> f64` 有损导致默认失败            |
| `i64`          | `Complex<f32>` | 错误     | 有损，默认失败                              |
| `f64`          | `Complex<f32>` | 错误     | 有损，默认失败                              |
| `Complex<f32>` | `f64`          | 条件成功 | 仅当 `im == 0`，再按 `f32 -> f64` 规则处理  |
| `Complex<f32>` | `f32`          | 条件成功 | 仅当 `im == 0`；否则错误                    |
| `Complex<f32>` | `i32`          | 条件成功 | 仅当 `im == 0`，再按 `f32 -> i32` 规则处理  |
| `Complex<f32>` | `i64`          | 条件成功 | 仅当 `im == 0`，再按 `f32 -> i64` 规则处理  |
| `Complex<f64>` | `f64`          | 条件成功 | 仅当 `im == 0`；否则错误                    |
| `Complex<f64>` | `f32`          | 条件成功 | 仅当 `im == 0`，再按 `f64 -> f32` 规则处理  |
| `Complex<f64>` | `i32`          | 条件成功 | 仅当 `im == 0`，再按 `f64 -> i32` 规则处理  |
| `Complex<f64>` | `i64`          | 条件成功 | 仅当 `im == 0`，再按 `f64 -> i64` 规则处理  |
| `Complex<f64>` | `Complex<f32>` | 错误     | 分量精度丢失，默认失败                      |

> `bool` 不参与 `cast()`；任何 `bool` 相关逐元素类型转换都不在本模块范围内。

### 5.4 闭合规则映射

凡 `§23.1` 已逐项列出的组合，其默认语义与附加成功前提以 `§23.1` 表格为准；闭合规则仅用于补足未逐项列出的受支持组合，不得覆盖或重新解释已列组合的语义。

未在上表逐项展开、但属于受支持源/目标集合的组合，按 `require.md §23.2` 闭合：

- 实数 → 复数：先按实数到目标复数实部分量类型的规则转换实部，再补 `0` 虚部
- 复数 → 实数：仅当虚部为 `0` 时继续；随后按实部到目标实数类型的规则处理
- 复数 → 复数：实部和虚部分别按对应实数转换规则处理
- 任一步为有损时，默认整体返回 `XenonError::TypeConversion(TypeConversionError)`

### 5.5 Good / Bad 对比

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

### 5.6 to_owned / into_owned

> **跨模块协作说明：** 本节保留 `to_owned()` / `into_owned()` 的实现细节，是因为 `require.md §23` 明确将同类型拷贝纳入本节约束；其中与具体存储表示相关的路径选择仍以 `tensor` / `storage` 模块为主责。

> **归属声明：** `to_owned()` / `into_owned()` 的公开语义只在本文维护；它们返回的 owned 结果固定为 Xenon 的 canonical F-order。`20-utility.md` 可引用它们作为 `to_contiguous()` 的实现依赖，但不再重复定义其契约。

````rust,ignore
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
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
        // from_shape_vec is the normative construction path; this aligned variant stays an internal helper for Xenon's allocation path (see 05-storage.md §5.1)
        Tensor::from_shape_vec_aligned(self.raw_dim(), data)
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
    pub fn into_owned(self) -> Tensor<A, D>;
}
````

> **说明：** 同类型拷贝（`to_owned()`/`into_owned()`）不通过 fallible `cast()` 建模，而是始终成功的基础操作。`cast::<A>()` 不适用于同类型拷贝场景。
>
> **统一规则（Wave 1）：** `ArcRepr → Owned` 始终分配并复制（O(n)），与引用计数无关。

### 5.7 内部构造辅助边界

> `cast()` / `to_owned()` 在实现上可以复用张量或存储层的内部构造 helper，但这些 helper 的命名、文件布局、是否存在 unchecked 变体以及具体对齐策略，都不属于 convert 模块的稳定文档面。

> 若内部保留类似 `from_shape_vec_aligned_unchecked` 的便捷路径，其 `# Safety` 只能要求调用方保证：`shape` 的已验证元素总数与 `data.len()` 一致，且由 `shape` 推导出的 F-order 元数据在当前版本范围内合法。底层使用哪一种分配器或对齐值，不应写入该 safety 契约。

---

## 6. 内部实现设计

### 6.1 CastTo 实现（核心转换路径）

```rust,ignore
// === Lossy-by-default conversion ===
impl CastTo<f32> for f64 {
    #[inline]
    fn cast_to(self) -> Result<f32, XenonError> {
        Err(XenonError::TypeConversion(TypeConversionError {
            source_type: "f64".into(),
            target_type: "f32".into(),
            reason: TypeConversionReason::LossyFloatNarrowing,
            element_index: 0,
        }))
    }
}

impl CastTo<f64> for f32 {
    #[inline]
    fn cast_to(self) -> Result<f64, XenonError> { Ok(self as f64) }
}

impl CastTo<i64> for i32 {
    #[inline]
    fn cast_to(self) -> Result<i64, XenonError> { Ok(self as i64) }
}

// === Conditionally successful conversions ===
impl CastTo<f64> for Complex<f64> {
    #[inline]
    fn cast_to(self) -> Result<f64, XenonError> {
        if self.im == 0.0 {
            Ok(self.re)
        } else {
            Err(XenonError::TypeConversion(TypeConversionError {
                source_type: "Complex<f64>".into(),
                target_type: "f64".into(),
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            }))
        }
    }
}

// === Lossy-by-default conversion ===
impl CastTo<i32> for i64 {
    #[inline]
    fn cast_to(self) -> Result<i32, XenonError> {
        Err(XenonError::TypeConversion(TypeConversionError {
            source_type: "i64".into(),
            target_type: "i32".into(),
            reason: TypeConversionReason::LossyIntegerNarrowing,
            element_index: 0,
        }))
    }
}
```

### 6.2 溢出行为汇总

> **错误语义约定：** `cast()` 是 fallible API。凡被 `require.md §23` 判定为有损的转换，默认返回 `XenonError::TypeConversion(TypeConversionError)`；仅在该节明确给出额外成功前提时，满足前提后方可成功。

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
  - 内容: 复用 `element` 模块中的 fallible `CastTo<T>` trait，实现无损与条件成功路径
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

- [ ] **T3**: 实现 `to_owned` / `into_owned` 及存储模式互转
  - 文件: `src/convert/owned.rs`
  - 内容: `to_owned()` 克隆方法与 `into_owned()` 消费方法；不在本模块扩展 view/view_mut/into_shared 等额外存储模式互转入口
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
  - 内容: 补齐 `require.md §23.1` 与 `§23.2` 定义的全部组合；`bool` 不参与
  - 测试: `test_cast_real_to_complex`, `test_cast_complex_to_real_requires_zero_imag`, `test_cast_complex_f64_to_complex_f32_returns_error`
  - 前置: T1
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] ──▶ [T2]
                    │
Wave 2: [T3] [T4] [T5]  (parallel)
```

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
| `cast()` 对所有可读存储提供，但统一返回 owned 结果 | 编译期测试。 |
| saturation / truncation casts 与额外 `From/Into` 非张量转换不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块  | 接口/类型                               | 约定                                                                                              |
| ------------------- | --------- | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `convert → tensor`  | `tensor`  | `TensorBase<S, D>` / `StorageIntoOwned` | `cast()`、`to_owned()`、`into_owned()` 都定义在张量抽象之上；其中 `to_owned()` / `into_owned()` 负责产出 canonical F-order owned 结果 |
| `convert → element` | `element` | `CastTo`                                | 逐元素类型转换通过 `CastTo` trait 驱动，参见 `03-element.md` §4                                   |
| `convert → math`    | `math`    | 逐元素转换语义                          | `cast()` 采用迭代收集路径，不复用 `mapv()` 的同类型返回语义                                       |
| `convert → storage` | `storage` | `Owned` / readable storage traits       | convert 只消费可读存储与 owned 化能力，不在本文扩展额外存储模式互转矩阵                           |
| `convert → layout`  | `layout`  | F-order metadata                        | `cast()`、`to_owned()`、`into_owned()` 保持张量 shape 与逻辑元素顺序，并为 owned 结果建立 canonical F-order 元数据；若调用方需要显式连续化入口，则由 `util::to_contiguous()` 负责 |
| `convert → complex` | `complex` | `Complex<T>`                            | 复数目标类型转换依赖 `Complex` 定义；Complex → 实数可在 `im == 0` 时成功，参见 `04-complex.md` §4 |

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
| Recoverable error | `cast()` 在有损转换、虚部非零或其他规则不满足时返回 `XenonError::TypeConversion(TypeConversionError)`，携带源类型、目标类型、失败原因与元素索引。 |
| Panic | 公开转换 API 不定义额外 panic 语义；有损场景统一返回可恢复错误。 |
| 路径一致性 | `cast`、`to_owned`、`into_owned` 必须保持相同 shape 与逻辑元素顺序；其中 `to_owned` / `into_owned` 的 owned 结果固定为 canonical F-order。无 SIMD / 并行分支。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### 决策 1：默认失败而非饱和/截断

| 属性     | 值                                                                        |
| -------- | ------------------------------------------------------------------------- |
| 决策     | 所有有损转换默认返回 `XenonError::TypeConversion(TypeConversionError)`    |
| 理由     | 这是 `require.md §23` 的强制要求；文档不得私自引入饱和、截断或 NaN→0 语义 |
| 替代方案 | saturating / truncating — 放弃，与需求冲突                                |
| 替代方案 | panic on overflow — 放弃，需求要求可恢复错误                              |

### 决策 2：cast() 对所有可读存储开放

| 属性     | 值                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 决策     | `cast()` 对所有可读存储开放，并统一返回 owned tensor                                                                                                          |
| 理由     | 这与 `require.md §23` 的逐元素转换要求一致，同时避免把“能否读取输入”与“结果是否拥有数据”混为一谈；输入可借用，结果仍统一 owned，API 语义保持单一。             |
| 替代方案 | 仅在 `Owned` 上实现 — 放弃，会无依据地缩小 `require.md §23` 的适用范围                                                                                         |
| 替代方案 | 按输入存储模式返回不同结果类型 — 放弃，会引入生命周期与所有权分歧，破坏公开 API 一致性                                                                        |

### 决策 3：收缩 convert 模块边界到当前需求集合

| 属性     | 值                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| 决策     | convert 模块仅覆盖 `cast()`、`to_owned()`、`into_owned()`；其余存储模式互转仅作跨文档引用，不在本文展开 |
| 理由     | 当前 `require.md §23` 只要求逐元素类型转换与同类型拷贝；继续讨论 `view` / `view_mut` / `into_shared` 会超出收缩后的边界 |
| 替代方案 | 在本文继续完整展开所有存储模式互转 — 放弃，会把 convert 文档扩展到非本节需求范围 |

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

本模块须遵循项目统一工程约束，不单独定义 `no_std` 目标：

- 仅支持 `std` 环境（参见 `require.md §1.3`）
- 保持单 crate 结构
- 遵循 SemVer
- 不新增超出项目基线的第三方依赖

| 项目       | 约束                                                      |
| ---------- | --------------------------------------------------------- |
| 平台       | 仅 `std`                                                  |
| MSRV       | Rust 1.85+                                                |
| crate 结构 | 单 crate                                                  |
| 依赖       | 不新增第三方依赖                                          |
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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
