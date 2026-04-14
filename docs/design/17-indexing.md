# 索引操作模块设计

> 文档编号: 17 | 模块: `src/index/` | 阶段: Phase 3
> 前置文档: `07-tensor.md`, `02-dimension.md`, `06-memory.md`, `26-error.md`
> 需求参考: 需求说明书 §18
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责              | 包含                                                                                          | 不包含                                       |
| ----------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 多维整数索引      | `[i, j, k]` 形式的元素访问，                                                                  | 高级索引（布尔掩码/整数数组/where 条件选择） |
| 范围索引（切片）  | `s![.., 0..3, ..;2]` 宏驱动切片                                                               | 高级索引（布尔掩码、整数数组高级索引）       |
| Index trait 实现  | `core::ops::Index` for `TensorBase`                                                           | 比较运算符（在逐元素运算模块）               |
| 偏移量计算        | F-order 公式：`offset = sum(index[i] * strides[i])`                                           | 其他内存序的偏移量计算                       |
| at/get/get_unchecked | `at()` 返回 `Result<&A, XenonError>`、`get()` 返回 `Option<&A>`、`get_unchecked()` unsafe 变体 | 可变迭代器（在 iter 模块）                   |
| 切片宏            | `s![]` 宏的语法解析和展开                                                                     | 切片宏设计在 ndarray 之外的库中              |
| 切片返回视图      | 切片返回只读 `ViewRepr`（零拷贝）                                                             | 勒贝操作（在逐元素运算模块）                 |
| 步长切片          | 仅支持正 `usize` 步长的范围切片                                                               | 负步长、反向视图（当前版本不提供）           |
| 边界检查          | `Index` panic、`at*` 返回 `Result`、`get*` 返回 `Option`、`slice`/`try_slice` 返回 `Result`、`unchecked` 为 unsafe | 越界 panic 消息模板（在 error 模块）         |

### 1.2 设计原则

| 原则           | 体现                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------- |
| 零拷贝视图     | 切片返回视图（`TensorView`），不复制数据                                                      |
| 编译期类型安全 | 静态维度通过 `Dimension` trait 保证索引正确性                                                 |
| 边界检查分层   | `Index` panic、`at*` 返回 `Result`、`get*` 返回 `Option`、`slice`/`try_slice` 返回 `Result`、`unchecked` 为 unsafe |
| F-order 偏移量 | 偏移量按 F-order 公式计算                                                                     |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (独立于 layout，由 tensor 持有并消费 layout 结果)
L4: tensor (依赖 storage, dimension)
L5: index  ← 当前模块（依赖 tensor, dimension, layout）
```

---

## 2. 需求映射与范围约束

| 类型     | 内容 |
| -------- | ---- |
| 需求映射 | 需求说明书 §18 |
| 范围内   | 多维整数索引、`at` / `get` / `get_unchecked` 分层、只读切片、`s![]` 宏与正步长切片。 |
| 范围外   | 高级索引（布尔掩码、fancy indexing）、负步长 / 负索引与共享可写切片。 |
| 非目标   | 不扩展新的索引语法族，不引入负步长布局，也不新增第三方切片宏依赖。 |

---

## 3. 文件位置

```
src/index/
├── mod.rs             # 模块入口，Index/IndexMut trait 实现、公开导出
├── multi_dim.rs       # 多维整数索引 [i, j, k]
└── slice_index.rs     # 切片索引、SliceInfo 设计、 s![] 宏
```

文件划分理由：整数索引和切片索引逻辑差异大，独立文件便于单独维护和测试。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
                    ┌──────────────┐
                    │    tensor    │
                    │ TensorBase   │
                    └──────┬───────┘
                           │ uses
              ┌────────────┼──────────────────┐
              │   index                      │
              │   multi_dim.rs │ slice_index.rs │
              └──┬───────────┬───────────────┘
                 │ uses      │ uses
          ┌──────▼───┐ ┌──────▼──────────────┐
          │ dimension│ │ memory-layout       │
          │ Dimension│ │ LayoutFlags         │
          │ Ix0~IxDyn│ │ HAS_ZERO_STRIDE     │
          └──────────┘ └─────────────────────┘
```

### 4.2 类型级依赖

| 来源模块    | 使用的类型/trait                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `tensor`    | `TensorBase<S, D>`, `TensorView`, `.shape()`, `.strides()`, `.as_ptr()`, `.as_mut_ptr()`，参见 `07-tensor.md` §4 |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `.slice()`, `.ndim()`，参见 `02-dimension.md` §3                              |
| `layout`    | `LayoutFlags`, `Strides<D>`, `HAS_ZERO_STRIDE`，参见 `06-memory.md` §3                                           |

> **注意**：`tensor[[...]]` / `tensor[[...]] = ...` 这类 `Index` / `IndexMut` 语法仅作为便捷 sugar，越界时使用 panic（参见 `26-error.md §5.3`），这是 Rust `Index` trait 必须返回引用的契约所决定的。
> 整数索引的规范安全路径是 `at()` / `at_mut()`：任一前提不满足时返回带诊断上下文的 `XenonError`。`get()` / `get_mut()` 保留为更轻量的 `Option` 便捷接口，但不是主错误返回模型。

### 4.3 依赖方向声明

> **依赖方向：单向向上。** `index/` 消费 `tensor`、`dimension`、`layout` 的 trait 和类型，不被它们依赖。

### 4.4 依赖合法性与替代方案

| 项目           | 说明 |
| -------------- | ---- |
| 新增第三方依赖 | 无 |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。 |

---

## 5. 公共 API 设计

### 5.0 NdIndex trait — 多维索引类型约束

`NdIndex<D>` 是多维索引类型的 sealed trait，用于 `core::ops::Index` 实现中约束合法的索引参数类型。

```rust
use crate::private::Sealed;

/// Trait for types that can be used as multi-dimensional indices into a tensor.
///
/// Sealed: cannot be implemented outside this crate.
///
/// # Implementations
///
/// | Type | Usage |
/// |------|-------|
/// | `[usize; N]` | Static N-dimensional index (e.g. `tensor[[0, 1]]`) |
/// | `&[usize]`   | Dynamic slice index (runtime length, checked at runtime) |
///
/// Note: The constraint `D: Dimension` on the implementation ensures that the
/// number of index components matches the tensor's dimensionality at runtime.
pub trait NdIndex<D: Dimension>: Sealed {
    /// Converts the index to a linear memory offset, given the tensor's strides.
    ///
    /// Returns `None` if any index component is out of bounds for its dimension.
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<usize>;

    /// Converts the index to a linear memory offset without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure every index component `i` satisfies `i < shape[k]`.
    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize;
}

// Static array index (e.g. tensor[[0, 1, 2]])
// Blanket Sealed impl for all array sizes — required by the generic NdIndex impl below.
impl<const N: usize> Sealed for [usize; N] {}

impl<D: Dimension, const N: usize> NdIndex<D> for [usize; N] {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<usize> {
        if self.len() != dim.ndim() { return None; }
        let shape = dim.slice();
        for (i, &idx) in self.iter().enumerate() {
            if idx >= shape[i] { return None; }
        }
        // SAFETY: bounds checked above
        Some(unsafe { self.index_unchecked(strides) })
    }

    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize {
        let mut offset: usize = 0;
        for (&idx, &stride) in self.iter().zip(strides.iter()) {
            offset += stride * idx;
        }
        offset
    }
}

// Dynamic slice index (e.g. tensor[&[0, 1, 2][..]])
impl Sealed for &[usize] {}
impl<D: Dimension> NdIndex<D> for &[usize] {
    fn index_checked(&self, dim: &D, strides: &Strides<D>) -> Option<usize> {
        if self.len() != dim.ndim() { return None; }
        let shape = dim.slice();
        for (i, &idx) in self.iter().enumerate() {
            if idx >= shape[i] { return None; }
        }
        Some(unsafe { self.index_unchecked(strides) })
    }

    unsafe fn index_unchecked(&self, strides: &Strides<D>) -> usize {
        let mut offset: usize = 0;
        for (&idx, &stride) in self.iter().zip(strides.iter()) {
            offset += stride * idx;
        }
        offset
    }
}
```

> **设计决策：** `NdIndex` 使用 `Sealed` trait 防止外部实现，同时通过编译期常量泛型 `[usize; N]`
> 支持静态数组索引语法 `tensor[[0, 1]]`（编译期长度已知），以及 `&[usize]` 支持运行时动态索引。

### 5.1 多维整数索引

````rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Primary checked indexing API.
    ///
    /// This is the canonical safe path for integer indexing. It returns a
    /// recoverable `XenonError` with diagnostic context when rank or bounds
    /// validation fails.
    ///
    /// # Errors
    /// Returns `Err(XenonError)` when the attempted index rank does not match
    /// the tensor rank, or when any component is out of bounds. The error must
    /// carry diagnostic context including the tensor shape and the attempted
    /// index value.
    pub fn at<I>(&self, index: I) -> Result<&A, XenonError>
    where
        I: NdIndex<D>;

    /// Multi-dimensional index access convenience sugar.
    ///
    /// This method is a thin wrapper over `at()` for `Index` syntax. It is not
    /// the canonical safe API.
    ///
    /// # Examples
    /// ```
    /// let tensor = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(tensor[[0, 0]], 1);
    /// assert_eq!(tensor[[1, 2]], 6);
    /// ```
    ///
    /// # Panics
    /// Panics if rank or bounds validation fails. This panic is acceptable only
    /// because Rust's `Index` trait contract requires returning a reference;
    /// callers needing recoverable errors must use `at()`.
}

impl<S, D, A, I> core::ops::Index<I> for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    I: NdIndex<D>,
{
    type Output = A;

    fn index(&self, index: I) -> &Self::Output;
}

impl<S, D, A, I> core::ops::IndexMut<I> for TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    I: NdIndex<D>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output;
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Primary checked mutable indexing API.
    ///
    /// This is the canonical safe mutable path and is only available on
    /// writable storage modes.
    ///
    /// # Errors
    /// Returns `Err(XenonError)` when rank or bounds validation fails. Safe
    /// mutable indexing against read-only or shared-read-only storage must be
    /// rejected with `XenonError::InvalidStorageMode`.
    pub fn at_mut<I>(&mut self, index: I) -> Result<&mut A, XenonError>
    where
        S: StorageMut<Elem = A>,
        I: NdIndex<D>;

    /// Secondary ergonomic checked accessor.
    ///
    /// This method intentionally returns `Option` for light-weight probing.
    /// Callers that need structured diagnostics should use `at()` first.
    ///
    /// # Examples
    /// ```
    /// assert_eq!(tensor.get(&[0, 0]), Some(&1));
    /// assert_eq!(tensor.get(&[2, 0]), None);  // out of bounds
    /// ```
    pub fn get(&self, index: &[usize]) -> Option<&A>;

    /// Gets a reference to the element without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `index[i] < shape[i]` for all dimensions.
    ///
    /// # Examples
    /// ```
    /// // SAFETY: [0, 0] is within bounds [2, 3]
    /// unsafe {
    ///     assert_eq!(*tensor.get_unchecked(&[0, 0]), 1);
    /// }
    /// ```
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A;

    /// Secondary ergonomic checked mutable accessor.
    ///
    /// This method is only available on writable storage modes. Callers that
    /// need structured diagnostics should use `at_mut()` first.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut A>
    where
        S: StorageMut<Elem = A>;

    /// Gets a mutable reference to the element without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `index[i] < shape[i]` for all dimensions.
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut A
    where
        S: StorageMut<Elem = A>;
}
````

> **设计决策：** `at()` / `at_mut()` 是整数索引的规范安全接口，遵循 `require.md §18`：当索引长度与 rank 不匹配、任一分量越界、或可变访问不满足存储模式前提时，返回可恢复错误而不是 panic。

> `Index` / `IndexMut` 仍保留为便捷语法，并在失败时 panic；这是 Rust `Index` trait 需要返回引用的契约使然，因此 panic 仅限 trait sugar，规范错误返回路径是 `at()` / `at_mut()`。

> `get()` / `get_mut()` 保留 `Option` 语义，作为不携带诊断上下文的次级便捷接口；`get_unchecked*()` 提供 unsafe 快速路径。

> `at_mut()` 仅对可写存储模式开放（如 `Owned`、`ViewMutRepr`）。对只读或共享只读存储，任何安全的可变索引入口都必须返回 `XenonError::InvalidStorageMode`，而不能暴露未定义的可变引用。

### 5.1.1 偏移量计算（F-order）

多维索引到线性偏移量的计算公式：

```rust
offset = sum(index[i] * strides[i])  for i in 0..ndim
```

F-order 中 `strides[i] = product(shape[0..i])`，因此：

```rust
offset = sum(index[i] * product(shape[0..i]))
     = index[0] * 1 + index[1] * shape[0] + index[2] * shape[0] * shape[1] + ...
```

示例（F-order `shape=[2, 3, 4]`, `strides=[1, 2, 6]`）：

```rust
index=[1, 2, 1] → offset = 1*1 + 2*2 + 1*6 = 11
```

### 5.1.2 边界检查 API 对称性

| 方法                         | 返回类型                 | 边界检查             | 失败行为                                                                 |
| ---------------------------- | ------------------------ | -------------------- | ------------------------------------------------------------------------ |
| `at([i, j])`                 | `Result<&A, XenonError>` | 返回 `Err`           | 返回带 shape / attempted index 诊断上下文的可恢复错误                   |
| `[[i, j]]`                   | `&A`                     | panic                | panic（`Index` trait sugar；规范安全路径是 `at()`）                      |
| `get(&[i, j])`               | `Option<&A>`             | 返回 `None`          | 返回 `None`（轻量探测路径，不携带诊断上下文；次级于 `at()`）            |
| `get_unchecked(&[i, j])`     | `&A`                     | 无 (unsafe)          | UB                                                                       |
| `at_mut([i, j])`             | `Result<&mut A, XenonError>` | 返回 `Err`       | 越界/维度不匹配返回诊断错误；只读或共享只读存储返回 `InvalidStorageMode` |
| `[[i, j]] = v` (mut)         | `&mut A`                 | panic                | panic（`IndexMut` trait sugar；规范安全路径是 `at_mut()`）               |
| `get_mut(&[i, j])`           | `Option<&mut A>`         | 返回 `None`          | 返回 `None`（轻量探测路径，不携带诊断上下文；次级于 `at_mut()`）         |
| `get_unchecked_mut(&[i, j])` | `&mut A`                 | 无 (unsafe)          | UB                                                                       |

### 5.2 范围索引（切片）

#### 5.2.1 SliceInfo 设计

```rust
/// Slice element type, describing a single axis slice.
///
/// **Note:** `step = 0` is illegal and returns a recoverable error from
/// `slice()` / `try_slice()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SliceInfoElem {
    /// Single index. This axis will be removed, reducing the result dimension by 1.
    Index(usize),

    /// Range slice. This axis is preserved with the length of the range.
    Range {
        /// Start index (inclusive), None means 0
        start: Option<usize>,
        /// End index (exclusive), None means axis length
        end: Option<usize>,
        /// Step size, None means 1. Must not be 0.
        step: Option<usize>,
    },
}

/// Storage strategy for slice descriptors.
///
/// Xenon supports up to 6 static dimensions. Common `s![]` call sites therefore
/// use the inline representation and avoid heap allocation. Only dynamically
/// constructed slice descriptions fall back to `Vec`.
#[derive(Debug, Clone)]
pub enum SliceInfoIndices {
    /// Inline storage for the common static-dimension path.
    Inline {
        len: u8,
        elems: [SliceInfoElem; 6],
    },

    /// Dynamic storage for runtime-constructed slice descriptions.
    Dynamic(Vec<SliceInfoElem>),
}

/// Slice information, containing slice descriptions for all axes.
#[derive(Debug, Clone)]
pub struct SliceInfo<I, D>
where
    I: Dimension,
    D: Dimension,
{
    /// Slice descriptions for each axis.
    indices: SliceInfoIndices,

    _out_dim: PhantomData<I>,
    _in_dim: PhantomData<D>,
}
```

> **设计决策：** `SliceInfoElem` 使用 `usize` 表示点索引、边界和步长。
> 这与 `require.md` §18 中“当前版本仅支持以 `usize` 表示的多维索引和范围索引”保持一致，
> 同时直接排除负索引与负步长语义。

> **存储策略：** `SliceInfo` 采用双路径表示。`s![]` 宏和绝大多数静态维度调用点使用
> `SliceInfoIndices::Inline`，在栈上保存最多 6 个 `SliceInfoElem`；只有运行时动态拼装的
> 切片描述才使用 `SliceInfoIndices::Dynamic(Vec<...>)`，因此“切片操作零拷贝”与
> “动态 SliceInfo 需要 alloc” 可以同时成立而不再自相矛盾。

#### 5.2.2 s![] 切片宏设计

````rust
/// Slice macro for creating type-safe slice descriptions.
///
/// # Syntax
///
/// | Syntax | Meaning | SliceInfoElem variant |
/// |--------|---------|----------------------|
/// | `..` | Full range | `Range { start: None, end: None, step: None }` |
/// | `a..b` | Range [a, b) | `Range { start: Some(a), end: Some(b), step: None }` |
/// | `a..b;c` | Range with step | `Range { start: Some(a), end: Some(b), step: Some(c) }` |
/// | `..;c` | Full range with step | `Range { start: None, end: None, step: Some(c) }` |
///
/// # Examples
/// ```
/// let view = tensor.slice(s![1..3, ..;2, ..]);
/// assert_eq!(view.shape(), &[2, 3, 4]);  // step=2 on axis 1
/// ```
#[macro_export]
macro_rules! s {
    ($($elem:tt),* $(,)?) => {
        // macro expansion logic
    };
}
````

> **设计决策：** `s![]` 宏设计参考 ndarray 的 `s![]` 宏，
> 提供 Rust 像的声明式宏语法，编译期类型安全。
> 相比过程式构造（`SliceInfo::from_vec(vec![...])?`），宏优先展开为
> `SliceInfoIndices::Inline` 表示，从而在静态维度场景下不引入堆分配；仅运行时动态切片构造
> 才进入 `Vec` 路径。

#### 5.2.3 切片方法

````rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Creates a sliced view (zero-copy).
    ///
    /// # Arguments
    /// * `info` - Slice information created by the `s![]` macro
    ///
    /// # Returns
    /// The sliced `TensorView`, zero-copy.
    ///
    /// # Errors
    /// Returns `Err` if slice dimensions don't match tensor dimensions, if any
    /// index is out of bounds, or if any slice step is zero.
    ///
    /// # Examples
    /// ```
    /// let view = tensor.slice(s![1..3, 2..5, ..]);
    /// assert_eq!(view.shape(), &[2, 3, 6]);
    /// ```
    pub fn slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension,
    {
        // ...
    }

    /// Checked sliced view creation.
    pub fn try_slice<I>(&self, info: SliceInfo<I, D>) -> Result<TensorView<'_, A, I>, XenonError>
    where
        I: Dimension,
    {
        // ...
    }
}
````

> **注意**：根据 `require.md` §18，当切片与源张量共享底层数据时，结果只允许落在只读引用或共享只读引用范围内，
> 因此当前版本不提供 `slice_mut()` / `try_slice_mut()` 这类共享可写切片 API。

> **错误语义分层：**
>
> - `at()` / `at_mut()`：整数索引的规范安全接口；任一条件不满足时返回 `XenonError`。
> - `tensor[[...]]`：语言级 `Index` / `IndexMut` 语法，失败时 panic；这是 trait 契约要求，规范错误返回路径仍是 `at()` / `at_mut()`。
> - `get()` / `get_mut()`：返回 `Option` 的次级便捷接口，不携带诊断上下文。
> - `slice()` / `try_slice()`：维度不匹配、越界、`step == 0` 等情况统一返回 `XenonError`。
> - `get_unchecked*()`：完全由调用者保证前置条件，违反即 UB。

#### 5.2.4 Good / Bad 对比

```rust
// Good - use at() as the canonical safe indexing path
fn safe_index(tensor: &Tensor<f64, Ix2>, idx: [usize; 2]) -> Result<&f64, XenonError> {
    tensor.at(idx)
}

let tensor = Tensor2::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Out-of-bounds returns a recoverable error with context instead of panicking
let result = safe_index(&tensor, [5, 0]);
assert!(matches!(result, Err(XenonError::IndexError { .. })));

// Callers can still opt into a fallback after checking the Result
let val = *safe_index(&tensor, [0, 0]).unwrap_or(&0.0);
assert_eq!(val, 1.0);
```

```rust
// Bad - direct [] access panics on out-of-bounds
fn unsafe_index(tensor: &Tensor<f64, Ix2>, idx: &[usize]) -> f64 {
    tensor[idx]  // panic if out of bounds!
}
```

#### 5.2.5 切片后元数据更新

切片操作需要更新视图的元数据：

| 元数据        | 更新规则                                                                        |
| ------------- | ------------------------------------------------------------------------------- |
| `shape[i]`    | 切片范围长度                                                                    |
| `strides[i]`  | 原步长 × 切片步长                                                               |
| `offset`      | 原偏移 + start × 原步长                                                         |
| `LayoutFlags` | 基于结果 shape/stride 重算；只保留需求允许的 F-order 连续/非连续/零步长视图状态 |

#### 5.2.6 切片后的连续性规则

| 切片模式                                                         | `is_f_contiguous()` 结果 | 说明                                               |
| ---------------------------------------------------------------- | ------------------------ | -------------------------------------------------- |
| 每个保留轴都是完整范围 `..` 且步长为 `1`                         | 保持原 contiguity        | 纯视图收缩，不改变剩余轴的物理顺序                 |
| 仅移除长度为 1 或被单点索引掉的轴，剩余轴仍保持规范 F-order 步长 | 仍可保持 F-contiguous    | 结果继续满足范围内布局语义                         |
| 任一轴使用非 `1` 步长                                            | 变为非连续               | 包括 `..;2` 等正步长跳采样                         |
| 任一轴产生零步长                                                 | 变为非连续               | 仅在输入本身已来自广播视图时保留 `HAS_ZERO_STRIDE` |

> **实现约束：** 切片后不得仅凭旧 flags 盲目继承连续性，必须基于新 `shape` / `strides` 重新计算 flags。

---

## 6. 内部实现设计

### 6.1 偏移量计算（F-order）

```rust
function compute_offset_f(shape: [usize; N], strides: [isize; N], index: [usize; N]) -> isize:
    offset = 0
    for i in 0..N:
        offset += index[i] * strides[i]
    return offset
```

### 6.2 切片步长/形状/偏移量计算

```rust
function compute_slice(shape, strides, offset, slices: [SliceInfoElem; N])
    -> (new_shape, new_strides, new_offset, new_layout):
    new_shape = []
    new_strides = []
    new_offset = offset
    new_layout = layout

    for i in 0..N:
        match slices[i]:
            Index(idx):
                // Single index: reduce dimension
                if idx >= shape[i]:
                    return Err(IndexOutOfBounds)
                new_offset += idx * strides[i]
                // shape and strides do not include this dimension
            Range { start, end, step }:
                // Range slice
                st = step.unwrap_or(1)
                if st == 0:
                    return Err(InvalidSliceStep)

                s = start.unwrap_or(0)
                e = end.unwrap_or(shape[i])
                if s > shape[i] or e > shape[i] or s > e:
                    return Err(SliceOutOfBounds)

                dim_len = (e - s + st - 1) / st
                new_shape.push(dim_len)
                new_strides.push(strides[i] * st)
                new_offset += s * strides[i]
    return (new_shape, new_strides, new_offset, new_layout)
```

### 6.3 安全性论证

- `get_unchecked`: 调用者保证索引在合法范围内；偏移量计算与 checked 版本相同
- `slice_unchecked`: 调用者保证切片范围在合法范围内
- 非连续切片: `HAS_ZERO_STRIDE` 标志标记步长为0的维度（仅出现在广播场景中）

---

## 7. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/index/mod.rs` 骨架
  - 文件: `src/index/mod.rs`
  - 内容: 模块声明、子模块文件占位、公开导出声明
  - 测试: 编译通过
  - 前置: `07-tensor.md` 完成
  - 预计: 5 min

- [ ] **T2**: 定义 `SliceInfoElem` 和 `SliceInfo` 类型
  - 文件: `src/index/slice_index.rs`
  - 内容: 枚举定义、`SliceInfoIndices` 双路径存储、`SliceInfo` 结构体、`from_inline` / `from_vec` 基本方法
  - 测试: `test_slice_info_elem_basic`, `test_slice_info_inline`, `test_slice_info_dynamic`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 多维整数索引

- [ ] **T3**: 实现 `at()` 与 `Index` trait for `TensorBase`
  - 文件: `src/index/multi_dim.rs`
  - 内容: `at()`/`index()`/`get()`/`get_unchecked()` 实现
  - 测试: `test_at_2d`, `test_index_2d`, `test_index_3d`, `test_at_out_of_bounds`, `test_index_out_of_bounds`
  - 前置: T1
  - 预计: 10 min

- [ ] **T4**: 实现 `at_mut()`、`IndexMut` 和 `get_mut`/`get_unchecked_mut`
  - 文件: `src/index/multi_dim.rs`
  - 内容: 可变索引方法实现，以及只读存储的 `InvalidStorageMode` 错误路径
  - 测试: `test_at_mut_2d`, `test_at_mut_invalid_storage_mode`, `test_index_mut_2d`, `test_get_mut_out_of_bounds`
  - 前置: T3
  - 预计: 10 min

### Wave 3: 切片索引

- [ ] **T5**: 实现 `slice()` 和 `try_slice()` 方法
  - 文件: `src/index/slice_index.rs`
  - 内容: 只读切片视图创建、元数据计算、`step == 0` 错误返回
  - 测试: `test_slice_basic`, `test_slice_with_step`, `test_slice_empty`
  - 前置: T2, T3
  - 预计: 10 min

- [ ] **T6**: 实现 `s![]` 宏
  - 文件: `src/index/slice_index.rs`
  - 内容: 宏定义、语法解析
  - 测试: `test_slice_macro_basic`, `test_slice_macro_step`
  - 前置: T5
  - 预计: 10 min

### Wave 4: 集成与测试

- [ ] **T7**: 编写综合测试
  - 文件: `tests/test_index.rs`
  - 内容: 链式切片、视图的视图、正步长切片验证
  - 测试: 覆盖所有公共 API
  - 前置: T1-T6
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] → [T2]
                  │
Wave 2:      [T3] → [T4]
                      │
Wave 3:      [T5] → [T6]
                      │
Wave 4:           [T7]
```

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                           |
| -------- | ------------------------ | -------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证整数索引、切片和 `s![]` 宏的核心语义                       |
| 集成测试 | `tests/`                 | 验证 `index` 与 `tensor`、`layout`、`iter`、`shape` 的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖零维、空数组、正步长和链式切片等边界                       |
| 属性测试 | `tests/property/`        | 验证切片长度、结果 shape、连续性与链式切片不变量               |

### 8.2 单元测试清单

| 测试函数                     | 测试内容                              | 优先级 |
| ---------------------------- | ------------------------------------- | ------ |
| `test_at_2d`                 | `at()` 成功返回元素引用               | 高     |
| `test_index_2d`              | 2D 张量 `[[0, 0]]`, `[[1, 2]]`        | 高     |
| `test_index_3d`              | 3D 张量 `[[0, 1, 2]]`                 | 高     |
| `test_at_out_of_bounds`      | `at()` 越界返回 `XenonError`          | 高     |
| `test_index_out_of_bounds`   | `tensor[[...]]` 越界 panic 验证       | 高     |
| `test_get_returns_none`      | `get()` 越界返回 None                 | 高     |
| `test_get_unchecked`         | unsafe 路径正确读取                   | 中     |
| `test_at_mut_2d`             | `at_mut()` 成功返回可变引用           | 高     |
| `test_at_mut_invalid_storage_mode` | `at_mut()` 在只读存储上返回 `InvalidStorageMode` | 高     |
| `test_index_mut_2d`          | 可变索引写入                          | 高     |
| `test_get_mut_out_of_bounds` | `get_mut()` 越界返回 None             | 高     |
| `test_slice_basic`           | `s![1..3, ..]` 基本切片               | 高     |
| `test_slice_with_step`       | `s![..;2, ..]` 步长切片               | 高     |
| `test_slice_empty_range`     | 空范围切片                            | 中     |
| `test_slice_chain`           | 切片的切片（视图的视图）              | 高     |
| `test_try_slice_step_zero`   | `try_slice()` 对 `step == 0` 返回错误 | 高     |
| `test_slice_info_inline`     | `s![]` 走 inline slice info 表示      | 高     |
| `test_slice_info_dynamic`    | 运行时构造走动态 slice info 表示      | 中     |
| `test_slice_macro_basic`     | `s![]` 宏基本使用                     | 高     |

### 8.3 边界测试场景

| 场景             | 预期行为                             |
| ---------------- | ------------------------------------ |
| 0 维张量索引     | `tensor[[]]` 返回唯一元素            |
| 空数组索引       | panic（无有效索引）                  |
| 大张量索引       | 正确偏移量计算                       |
| 非连续切片后索引 | 正确处理步长跳转                     |
| `step == 0`      | `slice()` / `try_slice()` 均返回错误 |

### 8.4 属性测试不变量

| 不变量                                                  | 测试方法                       |
| ------------------------------------------------------- | ------------------------------ |
| 切片后 `view.len() == expected_len`                     | 随机切片参数                   |
| `view.shape()` 正确                                     | 每个维度验证                   |
| 视图的视图链式后数据一致                                | 链式切片对比原数据             |
| 连续切片仅在规范 F-order 条件下保持 `is_f_contiguous()` | 随机完整范围/步长/单点索引组合 |

### 8.5 集成测试

| 测试文件              | 测试内容                                                                            |
| --------------------- | ----------------------------------------------------------------------------------- |
| `tests/test_index.rs` | 整数索引 / 切片 / `s![]` 宏 与 `tensor`、`layout`、`iter`、`shape` 的端到端协同路径 |

### 8.6 Feature gate / 配置测试

| 配置 | 验证点 |
| ---- | ---- |
| 默认配置 | 整数索引、切片与 `s![]` 宏在默认构建下保持统一错误分层。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。 |

### 8.7 类型边界 / 编译期测试

| 场景 | 测试方式 |
| ---- | ---- |
| `NdIndex` 仅允许受支持索引类型 | 编译期测试。 |
| 负步长 / 负索引 / advanced indexing 不属于当前 API | API 缺失断言或编译期失败测试。 |
| 只读存储上的 `at_mut()` / `get_mut()` 需被拒绝 | 运行时错误测试与 trait 约束验证。 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                | 对方模块    | 接口/类型                   | 约定                                                                |
| ------------------- | ----------- | --------------------------- | ------------------------------------------------------------------- |
| `index → tensor`    | `tensor`    | `as_ptr()` / `as_mut_ptr()` | 通过张量基础指针入口完成元素访问，参见 `07-tensor.md` §4            |
| `index → storage`   | `storage`   | `Storage` / `StorageMut`    | 通过存储抽象读取或写入元素，参见 `05-storage.md` §3                 |
| `index → dimension` | `dimension` | `Dimension::slice()`        | 使用维度切片能力推导结果形状，参见 `02-dimension.md` §3             |
| `index → layout`    | `layout`    | `LayoutFlags`               | 切片后更新布局标志与步长语义，参见 `06-memory.md` §3                |
| `index ← iter`      | `iter`      | `get_unchecked()`           | 迭代器路径复用快速索引入口，参见 `10-iterator.md` §4                |
| `index ← shape`     | `shape`     | transpose 后视图语义        | 转置等上游操作产生的只读视图继续复用索引路径，参见 `16-shape.md` §4 |

### 9.2 数据流描述

```text
User calls tensor[[...]] / get() / slice() / s![]
    │
    ├── indexing normalizes index or slice boundaries
    ├── tensor shape / strides / offset determine the logical location
    ├── layout updates slice-derived flags when a new view is created
    └── iter / shape / math can consume the resulting view or reference
```

---

## 10. 错误处理与语义边界

| 主题 | 内容 |
| ---- | ---- |
| Recoverable error | `at()` / `at_mut()` / `slice()` / `try_slice()` 在 rank 不匹配、越界、`step == 0` 或存储模式不合法时返回 `XenonError`，携带 shape、index、axis 或 storage-mode 上下文。 |
| Panic | `Index` / `IndexMut` 语法糖在验证失败时 panic；`get_unchecked*()` 前置条件违反属于 unsafe UB。 |
| 路径一致性 | 整数索引、切片、视图的视图路径必须保持同一 F-order 偏移和边界语义；无 SIMD / 并行分支。 |
| 容差边界 | 不适用。 |

---

## 11. 设计决策记录

### 决策 1: 切片宏 s![] 设计

| 属性     | 值                                                                  |
| -------- | ------------------------------------------------------------------- |
| 决策     | 使用声明式宏（`macro_rules!`）实现 `s![]` 切片宏                    |
| 理由     | 编译期类型安全；语法简洁类似 ndarray；展开为 `SliceInfoElem` 枚举值 |
| 替代方案 | 过程式构造 — 放弃，运行时解析复杂、类型不安全                       |
| 替代方案 | `macro` 过程宏 — 放弃，语法不自然                                   |

### 决策 2: 仅支持 `usize` 索引与正步长切片

| 属性     | 值                                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------------- |
| 决策     | 点索引、范围边界和步长统一采用 `usize`，不提供负索引和负步长                                                            |
| 理由     | `require.md` §18 要求当前版本仅支持以 `usize` 表示的多维索引和范围索引，且 `require.md` §7 明确当前版本不支持负步长布局 |
| 替代方案 | 继续支持 `isize` 与负步长 — 放弃，超出当前需求范围                                                                      |

---

## 12. 性能考量

### 12.1 复杂度

| 操作                   | 时间复杂度 | 空间复杂度              |
| ---------------------- | ---------- | ----------------------- |
| 多维索引 `[[i, j, k]]` | O(ndim)    | O(1)                    |
| `get()`                | O(ndim)    | O(1)                    |
| `get_unchecked()`      | O(ndim)    | O(1)                    |
| 切片 `slice()`         | O(ndim)    | O(ndim)（新视图元数据） |
| 视图创建               | O(1)       | O(ndim)                 |

### 12.2 索引计算开销

| 场景                  | 每元素开销                             |
| --------------------- | -------------------------------------- |
| F-contiguous 连续数组 | 1 次指针加法（步长=1时退化为指针递增） |
| 非连续数组（切片后）  | 多次乘加运算                           |
| 非连续 + 广播         | 乘加 + 跳步长处理                      |

### 12.3 缓存行为

| 场景             | 缓存友好性 | 说明     |
| ---------------- | ---------- | -------- |
| 连续数组顺序索引 | 最优       | 顺序访问 |
| 步长切片         | 较差       | 跳跃访问 |

### 12.4 性能数据（参考）

| 操作                  | 连续数组 | 非连续数组（步长=2） | 性能比         |
| --------------------- | -------- | -------------------- | -------------- |
| 随机索引 1M 次 (2D)   | ~2ms     | ~3ms                 | 1.5x           |
| 随机索引 1M 次 (4D)   | ~4ms     | ~6ms                 | 1.5x           |
| 切片视图创建          | ~10ns    | ~10ns                | 1x（纯元数据） |
| 切片后遍历 1M 元素    | ~1ms     | ~3ms                 | 3x             |
| 缓存命中率（顺序 2D） | ~95%     | ~60%                 | —              |
| 缓存命中率（随机 4D） | ~70%     | ~30%                 | —              |

---

## 13. 平台与工程约束

| 约束       | 说明                                                               |
| ---------- | ------------------------------------------------------------------ |
| `std` only | Xenon 当前版本仅支持 `std` 环境，本文不再讨论 `no_std` 兼容性      |
| 单 crate   | `index` 设计保持在现有 crate 内，不引入额外 crate                  |
| SemVer     | 本次调整是文档与需求基线对齐：移除负步长、负索引和共享可写切片承诺 |
| 最小依赖   | 本模块不新增第三方依赖                                             |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-07 |
| 1.0.3 | 2026-04-07 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-08 |
| 1.0.6 | 2026-04-10 |
| 1.1.0 | 2026-04-10 |
| 1.1.1 | 2026-04-10 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
