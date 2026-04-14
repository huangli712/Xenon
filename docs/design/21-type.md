# 类型转换模块设计

> 文档编号: 21 | 模块: `src/convert/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `03-element.md`
> 需求参考: 需求说明书 §23
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责           | 包含                                                 | 不包含                       |
| -------------- | ---------------------------------------------------- | ---------------------------- |
| 逐元素类型转换 | `cast<B>(&self) -> Result<Tensor<B, D>, XenonError>` | 隐式类型提升（需显式调用）   |
| 同类型拷贝     | `to_owned`、`into_owned`                             | 跨模块转换逻辑（如 reshape） |
| 存储模式互转   | Owned ↔ ViewRepr ↔ ArcRepr                           | 隐式 Deref 转换              |
| 标准库接口     | `From`/`Into`（标准库类型转换）                      | 非标准转换（如序列化）       |

### 1.2 设计原则

| 原则       | 体现                                                                       |
| ---------- | -------------------------------------------------------------------------- |
| 显式转换   | 所有类型转换须显式调用 `cast()`，无隐式提升                                |
| 失败可诊断 | 有损转换默认返回可恢复错误，错误上下文由 `XenonError::TypeConversion` 承载 |
| 存储约束   | `cast` 当前仅在 `Owned` 上提供；其它存储模式须先 `to_owned`                |
| 需求闭合   | 仅支持 `require.md §23.1` 与 `§23.2` 定义的类型对及其成功前提              |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: broadcast, iter, ffi
L6: math, matrix, reduction, shape, index, util
L7: convert  ← 当前模块
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §23 |
| 范围内   | `cast()`、`to_owned()`、`into_owned()`、存储模式互转以及标准库 `From` / `TryFrom` 入口。 |
| 范围外   | 饱和 / 截断 cast 语义、非张量类型的额外 `From` / `Into` 族以及超出需求矩阵的隐式转换。 |
| 非目标   | 不默认放宽有损转换规则，不新增第三方转换库，也不把连续性保证升级为 convert 的公开职责。 |

---

## 3. 文件位置

```
src/
└── convert/                 # 类型转换（目录模块）
    ├── mod.rs               # 模块根，re-exports
    ├── cast.rs              # cast() 方法、类型转换路径（消费 element 中定义的 CastTo）
    ├── owned.rs             # to_owned、into_owned、存储模式互转
    ├── from_impl.rs         # From/TryFrom trait 实现
    └── contiguous.rs        # 连续化内部 helper（若保留，仅服务 util::to_contiguous 的实现）
```

多文件设计：按转换职责拆分，便于后续扩展（如新增转换路径、存储模式等）。

### 3.1 文件职责

| 文件            | 职责                                                                      | 预估行数 |
| --------------- | ------------------------------------------------------------------------- | -------- |
| `mod.rs`        | 模块根，re-exports 所有公共类型                                           | ~20      |
| `cast.rs`       | 消费 `CastTo<T>` trait，提供所有类型转换 impl 与 `cast()` 方法            | ~200     |
| `owned.rs`      | `to_owned()`、`into_owned()`、存储模式互转（view/view_mut/into_shared）   | ~100     |
| `from_impl.rs`  | `From<Vec<A>>`、`From<&[A]>`、`From<[A; N]>`、`From<&Tensor> for View` 等 | ~80      |
| `contiguous.rs` | 连续化内部 helper（公共语义由 util::to_contiguous 持有）                  | ~50      |

---

## 4. 依赖关系

### 4.1 依赖图

```
src/convert/
├── mod.rs          # Re-exports: CastTo, cast, to_owned, into_owned, From impls
├── cast.rs         # Depends on element (CastTo) and tensor (TensorBase)
├── owned.rs        # Depends on tensor (TensorBase), storage, layout
├── from_impl.rs    # Depends on tensor (Tensor / TensorView), construct (from_shape_vec / from_vec)
└── contiguous.rs   # Depends on tensor (TensorBase) and layout

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

### 5.2 cast 方法

````rust
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
    A: Element,
{
    /// Element-wise type conversion.
    ///
    /// Only applicable to the `Owned` storage mode.
    /// For other storage modes, call `to_owned()` first: `view.to_owned().cast::<B>()?`.
    ///
    /// # Type Parameters
    ///
    /// * `B` - Target element type
    ///
    /// # Errors
    ///
    /// Returns `XenonError::TypeConversion` when any element cannot be converted
    /// under the rules defined in `require.md §23`.
    ///
    /// # Examples
    ///
    /// ```
    /// let a = Tensor1::from_vec(vec![1_i32, 2, 3]);
    /// let b: Tensor1<f64> = a.cast()?;
    ///
    /// let c = Tensor1::from_vec(vec![Complex::new(1.0_f64, 0.0)]);
    /// let d: Tensor1<f64> = c.cast()?;
    /// ```
    pub fn cast<B: Element>(&self) -> Result<Tensor<B, D>, XenonError>
    where
        A: CastTo<B>,
    {
        let mut data: Vec<B> = Vec::with_capacity(self.len());
        for (index, x) in self.iter().enumerate() {
            let value = (*x).cast_to().map_err(|err| match err {
                XenonError::TypeConversion {
                    source_type,
                    target_type,
                    reason,
                    ..
                } => XenonError::TypeConversion {
                    source_type,
                    target_type,
                    reason,
                    element_index: index,
                },
                other => other,
            })?;
            data.push(value);
        }
        Ok(Tensor::from_shape_vec_aligned(self.shape().clone(), data))
    }
}
````

> **设计决策（修订）**：根据需求说明书 §23，`cast()` 仅适用于持有数据的存储模式。当前设计将这一范围收窄为 `Owned<A>`；`ViewRepr` / `ViewMutRepr` / `ArcRepr` 均须先调用 `to_owned()`，再进行显式类型转换。

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
| `f32/f64`      | `i32/i64`      | 错误     | 有损，默认失败                              |
| `i32`          | `f32`          | 错误     | 有损，默认失败                              |
| `i64`          | `f32/f64`      | 错误     | 有损，默认失败                              |
| `i32`          | `Complex<f32>` | 错误     | 由 `i32 -> f32` 有损导致默认失败            |
| `i64`          | `Complex<f64>` | 错误     | 由 `i64 -> f64` 有损导致默认失败            |
| `i64`          | `Complex<f32>` | 错误     | 有损，默认失败                              |
| `f64`          | `Complex<f32>` | 错误     | 有损，默认失败                              |
| `Complex<f32>` | `f64`          | 条件成功 | 仅当 `im == 0`，再按 `f32 -> f64` 规则处理  |
| `Complex<f32>` | `f32`          | 条件成功 | 仅当 `im == 0`；否则错误                    |
| `Complex<f32>` | `i32/i64`      | 条件成功 | 仅当 `im == 0`，再按 `f32 -> 整数` 规则处理 |
| `Complex<f64>` | `f64`          | 条件成功 | 仅当 `im == 0`；否则错误                    |
| `Complex<f64>` | `f32`          | 条件成功 | 仅当 `im == 0`，再按 `f64 -> f32` 规则处理  |
| `Complex<f64>` | `i32/i64`      | 条件成功 | 仅当 `im == 0`，再按 `f64 -> 整数` 规则处理 |
| `Complex<f64>` | `Complex<f32>` | 错误     | 分量精度丢失，默认失败                      |

> `bool` 不参与 `cast()`；任何 `bool` 相关逐元素类型转换都不在本模块范围内。

### 5.4 闭合规则映射

未在上表逐项展开、但属于受支持源/目标集合的组合，按 `require.md §23.2` 闭合：

- 实数 → 复数：先按实数到目标复数实部分量类型的规则转换实部，再补 `0` 虚部
- 复数 → 实数：仅当虚部为 `0` 时继续；随后按实部到目标实数类型的规则处理
- 复数 → 复数：实部和虚部分别按对应实数转换规则处理
- 任一步为有损时，默认整体返回 `XenonError::TypeConversion`

### 5.5 Good / Bad 对比

```rust
// Good - explicit and fallible cast
let a: Tensor<i32, Ix1> = Tensor::from_vec(vec![1, 2]);
let b: Tensor<f64, Ix1> = a.cast()?;

// Bad - implicit type promotion (Xenon does not support this)
let floats: Tensor<f64, Ix1> = Tensor::from_vec(vec![1.0, 2.0]);
let ints: Tensor<i32, Ix1> = floats + 1.0;  // Compile error: type mismatch

// Good - complex to real is allowed only when imag == 0
let complex_t: Tensor<Complex<f64>, Ix1> = Tensor::from_vec(vec![Complex::new(3.0, 0.0)]);
let re_parts: Tensor<f64, Ix1> = complex_t.cast()?;

// Bad - assuming lossy conversion succeeds by default
let floats: Tensor<f64, Ix1> = Tensor::from_vec(vec![1.5, 2.7]);
let ints: Tensor<i32, Ix1> = floats.cast().unwrap();  // forbidden: returns TypeConversion error
```

### 5.6 to_owned / into_owned

````rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Clones data into a new owned tensor.
    ///
    /// Always allocates new memory and copies data, even if input is already Owned.
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
        // from_vec_aligned: copies into a 64-byte aligned allocation (see 05-storage.md §5.1)
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
    /// - `TensorView`/`TensorViewMut`: copies data, O(n)
    /// - `ArcTensor`: returned directly if ref count is 1, otherwise copied
    pub fn into_owned(self) -> Tensor<A, D>;
}
````

> **说明：** 同类型拷贝（`to_owned()`/`into_owned()`）不通过 fallible `cast()` 建模，而是始终成功的基础操作。`cast::<A>()` 不适用于同类型拷贝场景。

### 5.7 from_vec_aligned — 对齐构造辅助

`from_vec_aligned` 是 `Owned<A>` 的内部辅助方法，将 `Vec<A>` 的数据**拷贝**到 64 字节对齐的新分配中（O(n) 操作，参见 `05-storage.md §5.1`）。这条快速路径仅适用于 Xenon 当前封闭元素集合中的 `Copy` 元素；用户原始的 `Vec` 分配被丢弃，不会复用。

```rust
// Defined in src/storage/owned.rs (see 05-storage.md §5.1)
impl<A> Owned<A> {
    /// Creates Owned by copying data into a 64-byte aligned allocation.
    ///
    /// The user's original Vec allocation is discarded; a fresh aligned allocation
    /// is made and data is copied over.
    pub fn from_vec_aligned(data: Vec<A>) -> Self where A: Copy {
        let len = data.len();
        // Allocate aligned memory, copy elements, return
        // ... (implementation detail: uses AlignedAlloc + Vec::from_raw_parts)
    }
}
```

`from_shape_vec_aligned` 是 `Tensor` 上的对应内部便捷方法（无需验证长度，调用方已保证）：

```rust
# use crate::layout::Strides;
impl<A, D> TensorBase<Owned<A>, D> where A: Element, D: Dimension {
    /// Constructs a Tensor from pre-validated data with aligned allocation.
    /// Called internally after size validation is complete.
    ///
    /// Skips length validation because the caller guarantees `data.len()` matches
    /// the validated element count for `shape` (for example after `iter().collect()`,
    /// `to_owned()`, or the public 1D convenience constructor `from_vec()`).
    pub(crate) fn from_shape_vec_aligned(shape: D, data: Vec<A>) -> Self {
        let strides: Strides<D> = shape.strides_for_f_order();
        let storage = Owned::from_vec_aligned(data);
        TensorBase { storage, shape, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) }
    }

    /// Constructs a Tensor from pre-validated data without length checking.
    ///
    /// This is the `unsafe` counterpart of `from_shape_vec_aligned`, used
    /// when the caller has already validated the length (e.g., via `iter().collect()`).
    ///
    /// # Safety
    ///
    /// 调用者必须保证：
    /// - `data.len()` 等于 `shape` 对应的已验证元素总数（元素总数匹配）
    /// - `shape` 和 `strides` 描述合法的内存布局
    /// - `data` 中所有元素已正确初始化
    /// - 内存已通过 Xenon 的 64 字节对齐分配器分配（使用 `from_vec_aligned` 保证）
    pub(crate) unsafe fn from_shape_vec_aligned_unchecked(shape: D, data: Vec<A>) -> Self {
        let strides: Strides<D> = shape.strides_for_f_order();
        let storage = Owned::from_vec_aligned(data);
        TensorBase { storage, shape, strides, offset: 0, flags: LayoutFlags::from_order(Order::F) }
    }
}
```

### 5.8 连续化内部实现

本节描述的连续化实现仅作为 `20-utility.md §4.3` 中 `to_contiguous()` 的内部实现细节。`convert` 模块可以持有 `contiguous.rs` 这一内部 helper 文件，但连续性保证的**公共语义、命名和用户入口**始终归 `util` 模块，而不是 `convert` 模块。

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Copies data into a new F-contiguous owned tensor.
    ///
    /// Iterates elements in F-order and re-packs them into a fresh aligned allocation.
    /// Always returns an F-contiguous `Tensor<A, D>`.
    ///
    /// Internal path used by `util::to_contiguous()` when the source is non-contiguous.
    ///
    /// **Note:** iter() traverses logical elements using Xenon's iterator semantics.
    /// For F-contiguous arrays this matches memory order; for non-contiguous arrays
    /// it follows the iterator contract defined in `10-iterator.md` while re-packing
    /// the result into a fresh F-contiguous allocation.
pub(crate) fn util_internal_to_f_contiguous(&self) -> Tensor<A, D> {
        let mut data = Vec::with_capacity(self.len());
        // iter() follows the iterator contract from 10-iterator.md and yields
        // logical elements in the order expected by to_contiguous().
        for elem in self.iter().cloned() {
            data.push(elem);
        }
        Tensor::from_shape_vec_aligned(self.raw_dim(), data)
    }
}
```

### 5.9 存储模式互转

| 源 → 目标              | 操作                              | 复杂度       |
| ---------------------- | --------------------------------- | ------------ |
| Owned → ViewRepr       | 借用（`view()`）                  | O(1)         |
| Owned → ViewMutRepr    | 可变借用（`view_mut()`）          | O(1)         |
| Owned → ArcRepr        | Arc 包装（`into_shared()`）       | O(1)         |
| ViewRepr → Owned       | `to_owned()`（拷贝）              | O(n)         |
| ViewMutRepr → ViewRepr | 只读重借用（`view()` / `into()`） | O(1)         |
| ViewMutRepr → Owned    | `to_owned()`（拷贝）              | O(n)         |
| ArcRepr → ViewRepr     | 共享只读借用（`view()`）          | O(1)         |
| ArcRepr → Owned        | `make_mut()` 或 clone             | O(1) 或 O(n) |

> **完整性说明：** Xenon 当前不提供 `ViewRepr/ViewMutRepr/ArcRepr → ViewMutRepr` 这类可能引入别名写入的新公开路径；若调用方需要可写 owned 结果，统一经 `to_owned()` / `into_owned()` 收敛。

### 5.10 标准库类型转换接口

```rust
// Vec<A> → Tensor<A, Ix1>
impl<A: Element> From<Vec<A>> for Tensor<A, Ix1> {
    fn from(vec: Vec<A>) -> Self { Self::from_shape_vec([vec.len()], vec).expect("Vec -> Tensor1 shape is always valid") }
}

// &[A] → Tensor<A, Ix1>
impl<A: Element + Clone> From<&[A]> for Tensor<A, Ix1> {
    fn from(slice: &[A]) -> Self { Self::from_shape_slice([slice.len()], slice).expect("slice -> Tensor1 shape is always valid") }
}

// [A; N] → Tensor<A, Ix1>
impl<A: Element, const N: usize> From<[A; N]> for Tensor<A, Ix1> {
    fn from(arr: [A; N]) -> Self { Self::from_shape_vec([N], arr.into_iter().collect()).expect("array -> Tensor1 shape is always valid") }
}

// &Tensor → TensorView
impl<'a, A: Element, D: Dimension> From<&'a Tensor<A, D>> for TensorView<'a, A, D> {
    fn from(tensor: &'a Tensor<A, D>) -> Self { tensor.view() }
}

// &mut Tensor → TensorViewMut
impl<'a, A: Element, D: Dimension> From<&'a mut Tensor<A, D>> for TensorViewMut<'a, A, D> {
    fn from(tensor: &'a mut Tensor<A, D>) -> Self { tensor.view_mut() }
}
```

---

## 6. 内部实现设计

### 6.1 CastTo 实现（核心转换路径）

```rust
// === Lossy-by-default conversion ===
impl CastTo<f32> for f64 {
    #[inline]
    fn cast_to(self) -> Result<f32, XenonError> {
        Err(XenonError::TypeConversion {
            source_type: "f64",
            target_type: "f32",
            reason: TypeConversionReason::LossyFloatNarrowing,
            element_index: 0,
        })
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
            Err(XenonError::TypeConversion {
                source_type: "Complex<f64>",
                target_type: "f64",
                reason: TypeConversionReason::NonZeroImaginaryPart,
                element_index: 0,
            })
        }
    }
}

// === Lossy-by-default conversion ===
impl CastTo<i32> for i64 {
    #[inline]
    fn cast_to(self) -> Result<i32, XenonError> {
        Err(XenonError::TypeConversion {
            source_type: "i64",
            target_type: "i32",
            reason: TypeConversionReason::LossyIntegerNarrowing,
            element_index: 0,
        })
    }
}
```

### 6.2 溢出行为汇总

> **错误语义约定：** `cast()` 是 fallible API。凡被 `require.md §23` 判定为有损的转换，默认返回 `XenonError::TypeConversion`；仅在该节明确给出额外成功前提时，满足前提后方可成功。

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
  - 内容: `to_owned()` 克隆方法、`into_owned()` 消费方法、view/view_mut/into_shared
  - 测试: `test_to_owned_from_view`, `test_into_owned_from_tensor`, `test_into_owned_from_arc`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

- [ ] **T4**: 实现 `cast` 方法
  - 文件: `src/convert/cast.rs`
  - 内容: `cast<B>(&self) -> Result<Tensor<B, D>, XenonError>` 方法实现
  - 测试: `test_cast_f64_to_f32_returns_error`, `test_cast_i32_to_f64`, `test_cast_reports_element_index`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

- [ ] **T5**: 实现连续化内部 helper（供 util::to_contiguous 复用）
  - 文件: `src/convert/contiguous.rs`
  - 内容: 连续化内部 helper `util_internal_to_f_contiguous()`，实现非连续输入到 F-order owned 的重排，供 util::to_contiguous 复用
  - 测试: `test_to_contiguous_from_view`, `test_to_contiguous_already_contiguous`
  - 前置: T2, tensor 模块完成
  - 预计: 10 min

### Wave 3: From trait 实现

- [ ] **T6**: 实现标准库 `From` trait
  - 文件: `src/convert/from_impl.rs`
  - 内容: `From<Vec<A>>`, `From<&[A]>`, `From<[A; N]>`, `From<&Tensor> for View` 等
  - 测试: `test_from_vec`, `test_from_slice`, `test_from_array`, `test_from_tensor_view`
  - 前置: T3
  - 预计: 10 min

- [ ] **T7**: 扩展 CastTo 实现（整数↔整数、实数↔复数、复数↔复数）
  - 文件: `src/convert/cast.rs`
  - 内容: 补齐 `require.md §23.1` 与 `§23.2` 定义的全部组合；`bool` 不参与
  - 测试: `test_cast_real_to_complex`, `test_cast_complex_to_real_requires_zero_imag`, `test_cast_complex_f64_to_complex_f32_returns_error`
  - 前置: T1
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] ──▶ [T2]
                    │
Wave 2: [T3] [T4] [T5]  (并行)
             │
Wave 3: [T6] [T7]  (并行)
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
| `test_from_vec`                                                 | Vec → Tensor1                      | 高     |
| `test_from_slice`                                               | &[A] → Tensor1                     | 中     |
| `test_from_tensor_view`                                         | &Tensor → TensorView               | 高     |

### 8.3 边界测试场景

| 场景                                 | 预期行为                                        |
| ------------------------------------ | ----------------------------------------------- |
| 空张量 `cast`                        | 返回空张量，形状不变                            |
| 单元素无损 `cast`                    | 成功并保持形状                                  |
| 非连续 View `cast`                   | 先 `to_owned()` 再 `cast()`，结果正确且保持形状 |
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
| `bool` 不参与 `cast()` | 编译期测试。 |
| `cast()` 仅在 `Owned` 上提供 | 编译期测试。 |
| saturation / truncation casts 与额外 `From/Into` 非张量转换不属于当前 API | API 缺失断言。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块  | 接口/类型                               | 约定                                                                                              |
| ------------------- | --------- | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `convert → tensor`  | `tensor`  | `TensorBase<S, D>` / `StorageIntoOwned` | `cast()`、`to_owned()`、`into_owned()` 都定义在张量抽象之上，参见 `07-tensor.md` §4               |
| `convert → element` | `element` | `CastTo`                                | 逐元素类型转换通过 `CastTo` trait 驱动，参见 `03-element.md` §4                                   |
| `convert → math`    | `math`    | 逐元素转换语义                          | `cast()` 采用迭代收集路径，不复用 `mapv()` 的同类型返回语义                                       |
| `convert → storage` | `storage` | `Owned` / `ViewRepr` / `ArcRepr`        | 存储模式互转依赖 owned 化与借用语义，参见 `05-storage.md` §4                                      |
| `convert → layout`  | `layout`  | `is_f_contiguous()`                     | `to_owned()` 始终复制到 owned；若调用方需要连续性保证，则由 `util::to_contiguous()` 显式触发重排  |
| `convert → complex` | `complex` | `Complex<T>`                            | 复数目标类型转换依赖 `Complex` 定义；Complex → 实数可在 `im == 0` 时成功，参见 `04-complex.md` §4 |

### 9.2 数据流描述

```text
User calls cast() / to_owned() / into_owned()
    │
    ├── convert reads tensor shape / strides / storage mode metadata
    ├── cast collects elements and re-encodes them via CastTo rules
    ├── owned-conversion paths choose between O(1) transfer and O(n) copy
    ├── util::to_contiguous() triggers the internal F-order repacking helper when needed
    └── the module returns a new owned tensor or an explicit view conversion result
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `cast()` 在有损转换、虚部非零或其他规则不满足时返回 `XenonError::TypeConversion`，携带源类型、目标类型、失败原因与元素索引。 |
| Panic | 公开转换 API 不定义额外 panic 语义；有损场景统一返回可恢复错误。 |
| 路径一致性 | `cast`、`to_owned`、`into_owned` 与内部连续化 helper 必须保持相同 shape 和 F-order owned 结果约束；无 SIMD / 并行分支。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### 决策 1：默认失败而非饱和/截断

| 属性     | 值                                                                        |
| -------- | ------------------------------------------------------------------------- |
| 决策     | 所有有损转换默认返回 `XenonError::TypeConversion`                         |
| 理由     | 这是 `require.md §23` 的强制要求；文档不得私自引入饱和、截断或 NaN→0 语义 |
| 替代方案 | saturating / truncating — 放弃，与需求冲突                                |
| 替代方案 | panic on overflow — 放弃，需求要求可恢复错误                              |

### 决策 2：cast() 仅在持有数据的存储模式上实现

| 属性     | 值                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 决策     | `cast()` 仅在 `Owned<A>` 上实现                                                                                                                              |
| 理由     | 需求 §23 明确要求"类型转换仅适用于持有数据的存储模式"。为避免 `ArcRepr` 是否算作“持有数据”产生歧义，统一要求 View / ViewMut / Arc 都先 `to_owned()` 再转换。 |
| 替代方案 | 在 `Storage` 约束上实现（允许 View 直接 cast） — 放弃，与需求 §23 冲突                                                                                       |
| 替代方案 | 在 `Owned` 和 `ArcRepr` 上同时实现 — 放弃，会把共享存储重新定义成持有型存储，增加语义歧义                                                                    |

### 决策 3：存储模式转换策略

| 属性     | 值                                                                           |
| -------- | ---------------------------------------------------------------------------- |
| 决策     | 提供显式方法（`view()`, `view_mut()`, `into_shared()`），不使用 `Into` trait |
| 理由     | 显式方法命名更清晰，避免隐式行为；`From` 仅用于标准库接口                    |
| 替代方案 | 为所有模式对实现 `From` — 放弃，组合爆炸（N×N 对）                           |

---

## 12. 性能考量

| 操作                  | 时间复杂度 | 空间复杂度 | 说明           |
| --------------------- | ---------- | ---------- | -------------- |
| `cast`                | O(n)       | O(n)       | 新分配一个张量 |
| `to_owned`            | O(n)       | O(n)       | 总是拷贝       |
| `into_owned`（Owned） | O(1)       | O(1)       | 直接返回       |
| `into_owned`（View）  | O(n)       | O(n)       | 拷贝           |
| `view()`              | O(1)       | O(1)       | 仅元数据       |
| `into_shared()`       | O(1)       | O(1)       | Arc 包装       |

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

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
