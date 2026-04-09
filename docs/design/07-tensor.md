# 张量类型模块设计

> 文档编号: 07 | 模块: `src/tensor/` | 阶段: Phase 3
> 前置文档: `02-dimension.md`, `05-storage.md`, `06-memory-layout.md`
> 需求参考: 需求说明书 §8

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 核心结构体 | `TensorBase<S, D>` 双参数泛型结构体定义 | 运算逻辑（参见 `11-math.md §1`） |
| 类型别名 | `Tensor`/`TensorView`/`TensorViewMut`/`ArcTensor` 及维度便捷别名 | 广播规则（参见 `15-broadcast.md §3`） |
| 基础查询 | shape/ndim/len/strides/is_empty/is_f_contiguous/is_aligned 等方法 | 形状操作（reshape/transpose，参见 `16-shape.md §1`） |
| 安全构造 | 从形状和数据构造，验证合法性 | 索引操作（参见 `17-indexing.md §1`） |
| unsafe 构造 | `from_raw_parts`，用于 FFI | 切片操作（参见 `17-indexing.md §5`） |
| 视图方法 | view/view_mut/into_view | 集合操作（参见 `11-math.md §1`） |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零开销抽象 | 不同存储模式在运行时无额外开销 |
| 类型安全 | 通过泛型约束在编译期保证访问权限 |
| 统一接口 | 所有张量类型共享相同的核心 API |
| 最小核心 | 核心结构仅包含必要字段，功能通过扩展方法提供 |
| 栈上元数据 | 静态维度的 TensorBase 元数据完全在栈上 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)  ← 当前模块
L5: ops/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

---

## 2. 文件位置

```
src/tensor/
├── mod.rs             # TensorBase<S, D> 结构体定义 + 公开导出
├── impls.rs           # 核心查询方法实现
├── aliases.rs         # 类型别名定义
└── construct.rs       # 内部构造方法（unsafe 底层构造）
```

文件划分理由：结构体定义、方法实现、类型别名、构造方法各自独立且职责清晰。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

```
┌─────────────────────────────────────────────────────────────┐
│                     TensorBase<S, D>                        │
│                   (src/tensor/mod.rs)                       │
└────────────────────────┬────────────────────────────────────┘
                         │ 使用
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   storage     │ │   dimension   │ │    layout     │
│ - Owned<A>    │ │ - Ix0-Ix6     │ │ - LayoutFlags │
│ - ViewRepr    │ │ - IxDyn       │ │ - is_f_contig │
│ - ViewMutRepr │ │ - Dimension   │ │ - strides     │
│ - ArcRepr     │ │   trait       │ │   compute     │
│ - Storage     │ │               │ │               │
│   trait       │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `storage` | `Owned<A>`, `ViewRepr<&'a A>`, `ViewMutRepr<&'a mut A>`, `ArcRepr<A>`, `Storage`, `StorageMut`, `StorageOwned`, `StorageShared`（参见 `05-storage.md §4`） |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn`, `.slice()`, `.size()`, `.ndim()`（参见 `02-dimension.md §4`） |
| `layout` | `LayoutFlags`, `compute_f_strides()`, `is_f_contiguous()`, `is_aligned()`（参见 `06-memory-layout.md §4`） |

> 注意：`Layout` 结构体定义于 `06-memory-layout.md`，目前为"供未来扩展预留"。`TensorBase` 当前直接内联 `LayoutFlags` 字段而非嵌套 `Layout`。

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `tensor/` 消费 `storage`、`dimension`、`layout` 的 trait 和类型，不被它们依赖。`ops/`、`iter/` 等上层模块消费 `tensor`。

---

## 4. 公共 API 设计

### 4.1 TensorBase\<S, D\> 核心结构体

```rust
/// Core abstraction for multi-dimensional arrays.
///
/// # Type Parameters
///
/// * `S` - Storage mode, determining ownership and access rights
/// * `D` - Dimension type, determining rank and shape representation
///
/// # Memory Layout
///
/// Struct size depends on the concrete instantiation of S and D. For static dimensions (Ix0-Ix6),
/// D is a stack-allocated fixed-size array; for dynamic dimensions (IxDyn), D contains a heap-allocated Vec.
#[repr(C)]
pub struct TensorBase<S, D> {
    /// Underlying data storage.
    storage: S,

    /// Length of each axis.
    shape: D,

    /// Stride of each axis (in element units).
    ///
    /// > **Design Decision:** Strides are stored as `usize` values in the D type
    /// > (same type as shape, guaranteeing matching dimension count at compile time).
    /// > Negative strides are represented as large `usize` values (two's complement
    /// > representation), and can be zero-cost reinterpret-cast to `isize` via
    /// > `strides_isize()` on the Dimension trait. `LayoutFlags::HAS_NEG_STRIDE` is
    /// > a cached flag for fast query optimization only—it does not affect the stored
    /// > stride values themselves.
    /// > See §9 ADR-2 for details.
    strides: D,

    /// Data start offset (in element units).
    ///
    /// Supports zero-copy implementation of slice views.
    offset: usize,

    /// Layout flags (u8 bitflags).
    ///
    /// Caches contiguity, alignment, zero/negative stride info for O(1) queries.
    flags: LayoutFlags,
}
```

> **设计说明：** TensorBase 直接嵌入 `offset` 和 `flags` 字段，而非使用 doc 06 的 `Layout` 结构体。
> 这是因为 `offset` 与存储指针配合进行偏移计算，属于张量实例的固有属性，将二者分离避免了
> 额外的间接层。`Layout` 结构体仅作为纯标志位容器，其 `flags` 字段在需要时可通过
> `TensorBase::flags()` 获取。

```

### 4.2 类型别名（完整列表）

```rust
// === Primary type aliases ===

/// Owning multi-dimensional array.
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;

/// Immutable view.
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<&'a A>, D>;

/// Mutable view.
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<&'a mut A>, D>;

/// Atomically reference-counted shared array.
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;

// === Owned dimension convenience aliases ===

pub type Tensor0<A> = Tensor<A, Ix0>;
pub type Tensor1<A> = Tensor<A, Ix1>;
pub type Tensor2<A> = Tensor<A, Ix2>;
pub type Tensor3<A> = Tensor<A, Ix3>;
pub type Tensor4<A> = Tensor<A, Ix4>;
pub type Tensor5<A> = Tensor<A, Ix5>;
pub type Tensor6<A> = Tensor<A, Ix6>;
pub type TensorD<A> = Tensor<A, IxDyn>;

// === View dimension convenience aliases ===

pub type TensorView0<'a, A> = TensorView<'a, A, Ix0>;
pub type TensorView1<'a, A> = TensorView<'a, A, Ix1>;
pub type TensorView2<'a, A> = TensorView<'a, A, Ix2>;
pub type TensorView3<'a, A> = TensorView<'a, A, Ix3>;
pub type TensorView4<'a, A> = TensorView<'a, A, Ix4>;
pub type TensorView5<'a, A> = TensorView<'a, A, Ix5>;
pub type TensorView6<'a, A> = TensorView<'a, A, Ix6>;
pub type TensorViewD<'a, A> = TensorView<'a, A, IxDyn>;

// === ViewMut dimension convenience aliases ===

pub type TensorViewMut0<'a, A> = TensorViewMut<'a, A, Ix0>;
pub type TensorViewMut1<'a, A> = TensorViewMut<'a, A, Ix1>;
pub type TensorViewMut2<'a, A> = TensorViewMut<'a, A, Ix2>;
pub type TensorViewMut3<'a, A> = TensorViewMut<'a, A, Ix3>;
pub type TensorViewMut4<'a, A> = TensorViewMut<'a, A, Ix4>;
pub type TensorViewMut5<'a, A> = TensorViewMut<'a, A, Ix5>;
pub type TensorViewMut6<'a, A> = TensorViewMut<'a, A, Ix6>;
pub type TensorViewMutD<'a, A> = TensorViewMut<'a, A, IxDyn>;

// === Arc dimension convenience aliases ===

pub type ArcTensor0<A> = ArcTensor<A, Ix0>;
pub type ArcTensor1<A> = ArcTensor<A, Ix1>;
pub type ArcTensor2<A> = ArcTensor<A, Ix2>;
pub type ArcTensor3<A> = ArcTensor<A, Ix3>;
pub type ArcTensor4<A> = ArcTensor<A, Ix4>;
pub type ArcTensor5<A> = ArcTensor<A, Ix5>;
pub type ArcTensor6<A> = ArcTensor<A, Ix6>;
pub type ArcTensorD<A> = ArcTensor<A, IxDyn>;
```

### 4.3 基础信息查询方法

```rust
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// Returns a slice of axis lengths.
    pub fn shape(&self) -> &[usize];

    /// Returns a slice of strides (isize, in element units).
    ///
    /// Strides may be negative (reversed dimensions) or zero (broadcast dimensions).
    pub fn strides(&self) -> &[isize];

    /// Returns the number of dimensions.
    ///
    /// For static dimensions (Ix0-Ix6), this is a compile-time constant.
    /// For dynamic dimensions (IxDyn), this is a runtime value.
    pub fn ndim(&self) -> usize;

    /// Returns the total number of elements (product of all dimension lengths).
    pub fn len(&self) -> usize;

    /// Returns whether the array is empty (any dimension length is 0).
    pub fn is_empty(&self) -> bool;

    /// Returns the data start offset (in element units).
    pub fn offset(&self) -> usize;

    /// Returns a clone of the dimension type.
    pub fn raw_dim(&self) -> D;

    /// Returns the complete layout flags.
    pub fn flags(&self) -> LayoutFlags;

    /// Whether the data is F-order contiguous.
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.flags.is_f_contiguous()
    }

    /// Whether the data is 64-byte aligned.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.flags.is_aligned()
    }

    /// Whether there is a zero stride (broadcast dimension).
    #[inline]
    pub fn has_zero_stride(&self) -> bool {
        self.flags.has_zero_stride()
    }

    /// Whether there is a negative stride (reversed dimension).
    #[inline]
    pub fn has_neg_stride(&self) -> bool {
        self.flags.has_neg_stride()
    }

    /// Returns true if the tensor is contiguous in memory.
    ///
    /// For Xenon (F-order only library), this is equivalent to `is_f_contiguous()`.
    pub fn is_contiguous(&self) -> bool {
        self.is_f_contiguous()
    }
}
```

### 4.4 指针访问方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a raw pointer to the data start.
    pub fn as_ptr(&self) -> *const A;

    /// Returns the raw base pointer WITHOUT adding the offset.
    ///
    /// Unlike `as_ptr()` which returns `storage.as_ptr().add(offset)`,
    /// this method returns `storage.as_ptr()` directly — the raw base
    /// pointer of the storage buffer. The caller is responsible for
    /// manually accounting for `self.offset` when computing element
    /// addresses.
    ///
    /// # Safety
    ///
    /// Caller must ensure offset is valid and data is initialized.
    /// The returned pointer does NOT point to the first logical element;
    /// use `as_ptr()` for that. Any pointer arithmetic based on this
    /// pointer must include `self.offset` to access the correct data.
    pub unsafe fn as_ptr_unchecked(&self) -> *const A;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Returns a mutable raw pointer to the data start.
    pub fn as_mut_ptr(&mut self) -> *mut A;
}
```

### 4.5 安全构造方法

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// Constructs an owning tensor from shape and data, validating correctness.
    ///
    /// # Arguments
    ///
    /// * `shape` - Length of each axis
    /// * `data` - Element data (Vec)
    ///
    /// # Errors
    ///
    /// Returns `Err` when:
    /// - `data.len() != shape.size()`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = Tensor2::<f64>::from_shape_vec([3, 4], vec![1.0; 12])?;
    /// ```
    pub fn from_shape_vec(shape: D, data: Vec<A>) -> Result<Self, XenonError>;
    // NOTE: from_shape_vec internally copies data into new 64-byte aligned memory
    // via `Owned::from_vec_aligned(data)`. Time complexity is O(n).
    // This ensures the tensor always satisfies SIMD alignment requirements.

    /// Construct a tensor from a Vec without validating shape/stride consistency.
    ///
    /// # Safety
    /// - `data.len()` must equal the product of all shape dimensions
    /// - `shape` and `strides` must be consistent with the data layout
    pub(crate) unsafe fn from_raw_vec_unchecked(data: Vec<A>, shape: D) -> Self {
        // computes F-order strides internally
        // ...
    }
}
```

### 4.6 unsafe 构造方法

```rust
impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    D: Dimension,
{
    /// Constructs an immutable view from raw parts.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`
    /// - The memory range starting at `ptr` covers all accessible elements
    /// - Memory remains valid for the lifetime `'a` of the returned view
    /// - All accessible elements are properly initialized
    /// - `shape` and `strides` have consistent lengths
    /// - Offsets computed from any valid index are within bounds
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self;
}

impl<'a, A, D> TensorBase<ViewMutRepr<&'a mut A>, D>
where
    D: Dimension,
{
    /// Constructs a mutable view from raw parts.
    ///
    /// # Safety
    ///
    /// Same as `from_raw_parts`, with the additional requirement of exclusive access.
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self;
}
```

### 4.7 视图方法

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Creates an immutable view (zero-copy).
    pub fn view(&self) -> TensorView<'_, A, D>;
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Creates a mutable view (zero-copy, exclusive access).
    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D>;
}

impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    /// Consumes the array, converting to an immutable view.
    ///
    /// Suitable for converting Owned/ArcRepr to View without lifetime binding.
    ///
    /// > **注意：** 此方法的泛型约束 `S: Into<S2>` 对涉及生命周期的存储类型
    /// > 适用性有限。对于常见场景（Owned → View、ArcRepr → View），
    /// > 推荐使用具体的 `view()` 和 `view_mut()` 方法，它们直接返回
    /// > 正确的视图类型，避免了泛型转换的限制。
    pub fn into_view<S2>(self) -> TensorBase<S2, D>
    where
        S: Into<S2>;
}
```

### 4.8 Good/Bad 对比

```rust
// Good - Use generic constraints to accept any readable tensor
fn process<S, D, A>(tensor: &TensorBase<S, D>)
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    let ptr = tensor.as_ptr();
    // ...
}

// Bad - Hardcoded Owned type
fn process_bad<A, D>(tensor: &Tensor<A, D>)
where
    D: Dimension,
{
    let ptr = tensor.as_ptr();
    // ...
}
```

```rust
// Good - Use from_shape_vec to validate correctness
let t = Tensor2::<f64>::from_shape_vec([3, 4], vec![1.0; 12])?;

// Bad - Use unsafe from_raw_parts to skip validation
let t = unsafe {
    TensorView2::from_raw_parts(data.as_ptr(), Ix2([3, 4]), Ix2([1, 3]), 0)
};
```

---

## 5. 内部实现设计

### 5.1 步长存储策略

> **设计决策：** `strides` 字段类型为 `D`（与 `shape` 同类型），但步长值需要支持负数。
>
> **实现方案：**
>
> | 层次 | 类型 | 说明 |
> |------|------|------|
> | `TensorBase.strides` | `D`（存储 `usize`） | 与 shape 同类型，编译期保证维度数一致 |
> | `strides()` 返回值 | `&[isize]` | 通过 Dimension trait 的 `strides_isize()` 方法转换（参见 `02-dimension.md §6`） |
> | layout 模块计算 | `isize` | 负步长和零步长在 layout 层计算（参见 `06-memory-layout.md §4.2`） |
>
> **权衡：**
> - D 类型保证 strides 与 shape 维度数相同（编译期）
> - 静态维度使用栈分配数组（性能）
> - 负步长符号通过 `LayoutFlags::HAS_NEG_STRIDE` 标记辅助处理

### 5.2 offset 字段设计

```
原始数组 storage: [a, b, c, d, e, f, g, h]
shape: [8], strides: [1], offset: 0

切片 [2..5] 后：
storage: [a, b, c, d, e, f, g, h]  // 共享，不复制
shape: [3], strides: [1], offset: 2  // 仅调整元数据
逻辑视图: [c, d, e]
```

**安全性论证**：`offset` 始终 ≤ `storage.len()`，由构造方法验证。`as_ptr()` 返回 `storage.as_ptr().add(offset)`，由 `Storage` trait 保证指针有效。

> **重要设计约定：** `TensorBase::offset` 是所有存储模式（Owned、ViewRepr、ViewMutRepr、ArcRepr）共用的唯一偏移字段。`ArcRepr` 不存储独立的 offset — 数据访问的起始位置完全由 `TensorBase::offset` 决定。这避免了双重偏移计算的 bug，并使偏移逻辑集中在一处。

### 5.3 内存布局示意

```
Tensor2<f64> = TensorBase<Owned<f64>, Ix2>

┌─────────────────────────────────────────┐
│ storage: Owned<f64>                     │
│   ┌───────────────────────────────────┐ │
│   │ data: Vec<f64> (64B 对齐)          │ │
│   │ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]    │ │
│   └───────────────────────────────────┘ │
│ shape: Ix2([2, 3])                      │
│ strides: Ix2([1, 2])  // F-order        │
│ offset: 0                               │
│ flags: F_CONTIGUOUS | ALIGNED           │
└─────────────────────────────────────────┘

逻辑视图：
  [[1.0, 3.0, 5.0],
   [2.0, 4.0, 6.0]]
```

---

## 6. 实现任务拆分

### Wave 1: 结构体定义和基础

- [ ] **T1**: 创建 `src/tensor/mod.rs` 骨架
  - 文件: `src/tensor/mod.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: storage、dimension、layout 模块完成
  - 预计: 5 min

- [ ] **T2**: 定义 `TensorBase<S, D>` 结构体
  - 文件: `src/tensor/mod.rs`
  - 内容: `#[repr(C)]` 结构体定义，5 个字段：storage、shape、strides、offset、flags
  - 测试: 结构体编译通过
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 定义类型别名 (aliases.rs)
  - 文件: `src/tensor/aliases.rs`
  - 内容: 4 个主类型别名 + 4×8 = 32 个维度便捷别名
  - 测试: 所有别名编译通过
  - 前置: T2
  - 预计: 10 min

### Wave 2: 核心查询方法

- [ ] **T4**: 实现形状与步长查询方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `shape()`/`strides()`/`ndim()`/`len()`/`is_empty()`/`offset()`/`raw_dim()`/`flags()`
  - 测试: `test_shape_query`, `test_len_empty`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5**: 实现布局查询委托方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `is_f_contiguous()`/`is_aligned()`/`has_zero_stride()`/`has_neg_stride()`
  - 测试: `test_layout_flags_delegate`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6**: 实现指针访问方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `as_ptr()`/`as_ptr_unchecked()`/`as_mut_ptr()`
  - 测试: `test_as_ptr`, `test_as_mut_ptr`
  - 前置: T4
  - 预计: 10 min

### Wave 3: 构造和视图

- [ ] **T7**: 实现 `from_raw_parts` 系列 (construct.rs)
  - 文件: `src/tensor/construct.rs`
  - 内容: `from_raw_parts`(不可变)/`from_raw_parts_mut`(可变)，完整 Safety 文档
  - 测试: `test_from_raw_parts_view`, `test_from_raw_parts_mut`
  - 前置: T2
  - 预计: 10 min

- [ ] **T8**: 实现安全构造方法 (construct.rs)
  - 文件: `src/tensor/construct.rs`
  - 内容: `from_shape_vec`/`new_unchecked`(内部方法)
  - 测试: `test_from_shape_vec_valid`, `test_from_shape_vec_invalid`
  - 前置: T5, T7
  - 预计: 10 min

- [ ] **T9**: 实现视图创建方法
  - 文件: `src/tensor/impls.rs`
  - 内容: `view()`/`view_mut()`/`into_view()`
  - 测试: `test_view_create`, `test_view_mut_create`
  - 前置: T6
  - 预计: 10 min

### Wave 4: 测试和收尾

- [ ] **T10**: 集成测试和文档
  - 文件: `tests/tensor.rs`
  - 内容: 跨模块交互测试、边界测试、类型别名编译验证
  - 测试: 完整集成测试套件
  - 前置: T3, T9
  - 预计: 15 min

### 并行执行图

```
Wave 1: [T1] → [T2] → [T3]
                ↓
Wave 2:        [T4] → [T5]
                ↓      ↓
               [T6]   [T7]
                ↓      ↓
Wave 3:       [T8] → [T9]
                ↓
Wave 4:       [T10]
```

---

## 7. 测试计划

### 7.1 测试分类

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个方法 |
| 集成测试 | `tests/` | 验证跨模块交互 |
| 边界测试 | 集成测试中标注 | 空数组、单元素、高维 |
| 编译测试 | `tests compile_fail` | 验证类型约束 |

### 7.1b 集成测试函数列表

以下集成测试函数验证 TensorBase 跨模块边界的正确性：

| 测试函数 | 测试内容 |
|----------|----------|
| `test_tensor_cross_dim_interop` | TensorBase 与 Dimension 模块交互：验证 Ix0~Ix6 和 IxDyn 的 shape/strides 查询 |
| `test_tensor_storage_layout_integration` | TensorBase 与 Storage/Layout 模块交互：验证 from_shape_vec 后的标志位计算和指针正确性 |
| `test_tensor_view_roundtrip` | 验证 view() → view_mut() → 原始数据的零拷贝往返一致性 |

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_tensor_shape_2d` | `Tensor2::from_shape_vec([3,4], data)` 后 shape 查询 | 高 |
| `test_tensor_len` | `len()` 返回 shape 乘积 | 高 |
| `test_tensor_is_empty` | 空数组 `is_empty()` 返回 true | 高 |
| `test_tensor_ndim_static` | `Tensor2` 的 `ndim()` == 2 | 高 |
| `test_tensor_ndim_dynamic` | `TensorD` 的 `ndim()` 运行时 | 中 |
| `test_tensor_strides_f_order` | F-order 步长正确 `[1, shape[0], ...]` | 高 |
| `test_tensor_flags_f_contiguous` | 新构造张量 F-连续 | 高 |
| `test_tensor_flags_aligned` | 新构造张量对齐 | 高 |
| `test_tensor_as_ptr` | 指针指向正确位置 | 高 |
| `test_tensor_as_mut_ptr` | 可变指针指向正确位置 | 高 |
| `test_tensor_view` | `view()` 创建正确视图 | 高 |
| `test_tensor_view_mut` | `view_mut()` 创建正确可变视图 | 高 |
| `test_from_shape_vec_valid` | 合法构造成功 | 高 |
| `test_from_shape_vec_len_mismatch` | 长度不匹配返回错误 | 高 |
| `test_type_aliases_compile` | 所有类型别名编译通过 | 高 |
| `test_tensor0_scalar` | 0D 标量张量 `len()==1` | 中 |
| `test_tensor_empty_dim` | 含 0 维度的张量 `is_empty()` | 中 |

### 7.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 `shape=[0, 3]` | `len()==0`, `is_empty()==true` |
| 单元素 `shape=[1, 1]` | `len()==1`, F-连续 |
| 标量 `Tensor0<f64>` | `ndim()==0`, `len()==1` |
| 高维 `Tensor6` | `ndim()==6`, 步长正确 |
| 动态维度 `TensorD` | `ndim()` 运行时值正确 |

### 7.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `tensor.len() == tensor.shape().iter().product()` | 随机形状 |
| `tensor.view().shape() == tensor.shape()` | 随机形状和存储模式 |
| `from_shape_vec` 后 `is_f_contiguous() == true` | 随机合法形状 |

---

## 8. 设计决策记录

### 决策 1：TensorBase\<S, D\> 双参数泛型设计

| 属性 | 值 |
|------|-----|
| 决策 | 使用 `TensorBase<S, D>` 双参数泛型，S 为存储模式，D 为维度类型 |
| 理由 | 零开销（编译期单态化）；类型安全（编译期禁止只读视图写入）；统一接口（所有存储模式共享 API） |
| 替代方案 | `TensorBase<A, S, D>` 三参数 — 放弃，A 可从 S 推导，冗余 |
| 替代方案 | 分离类型（Tensor/TensorView 独立结构体） — 放弃，代码重复 |
| 替代方案 | 单一 `Tensor<A, D>` + 运行时标志 — 放弃，运行时开销 |

### 决策 2：步长存储策略

| 属性 | 值 |
|------|-----|
| 决策 | `strides` 字段使用 `D` 类型存储，layout 层维护 `isize` 步长 |
| 理由 | 编译期保证 strides 与 shape 维度数相同；静态维度使用栈分配（性能）；类型一致性 |
| 替代方案 | `strides: Vec<isize>` — 放弃，静态维度也要堆分配 |
| 替代方案 | `strides: [isize; N]` — 放弃，不支持动态维度 |
| 替代方案 | `Strides<D>` 独立类型 — 放弃，增加类型复杂度 |

### 决策 3：offset 字段必要性

| 属性 | 值 |
|------|-----|
| 决策 | 包含 `offset: usize` 字段 |
| 理由 | 切片操作 O(1)（仅修改元数据）；无数据复制；统一机制适用所有存储模式；BLAS 兼容 |
| 替代方案 | 无 offset，切片时调整 storage 指针 — 放弃，Owned 无法调整指针 |

### 决策 4：不实现 Deref\<Target=TensorView\>

| 属性 | 值 |
|------|-----|
| 决策 | 不实现 `Deref<Target = TensorView>` |
| 理由 | 显式优于隐式（`.view()` 清晰表达意图）；避免隐式生命周期传播；与 ndarray 一致 |
| 替代方案 | 实现 Deref — 放弃，隐式转换可能导致意外借用 |

---

## 9. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 栈上元数据 | 静态维度（Ix0-Ix6）的 TensorBase 元数据完全在栈上 |
| 零成本抽象 | 不同存储模式编译为不同类型，无虚调用 |
| O(1) 查询 | shape/ndim/len/flags 查询均为 O(1) |
| 视图零拷贝 | `view()`/`view_mut()` 仅复制元数据 |
| 单态化 | Dimension + Storage trait 在泛型上下文中单态化 |

**TensorBase 大小分析（参考）**：

| 实例化 | 大小（估算） | 说明 |
|--------|-------------|------|
| `Tensor2<f64>` | ~72 bytes | Owned(24) + Ix2(16) + Ix2(16) + usize(8) + u8(1) + padding(7) = 72 bytes |
| `TensorView2<f64>` | ~56 bytes | ViewRepr<&'a f64>(≈ 8 bytes: 裸指针+PhantomData) + Ix2(16) + Ix2(16) + usize(8) + u8(1) + padding ≈ 56 bytes |
| `TensorD<f64>` | ~96 bytes | Owned(24) + IxDyn(24×2) + usize(8) + u8(1) + padding |

**性能数据（参考）**：

| 操作 | 开销 | 说明 |
|------|------|------|
| `shape()` | ~1ns | 切片返回 |
| `len()` | ~2ns | 乘积计算 |
| `view()` | ~5ns | 元数据复制 |
| `from_shape_vec()` | ~1μs + alloc | 包含验证和步长计算 |

---

## 10. no_std 兼容性

张量模块在 `no_std` 环境下的支持情况取决于存储模式和维度类型。

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `Tensor<A, D>`（Owned） | ✅ | 使用 `alloc::vec::Vec`，需 `no_std + alloc` |
| `TensorView<'a, A, D>` | ✅ | 仅使用 `core`（裸指针 + 生命周期） |
| `TensorViewMut<'a, A, D>` | ✅ | 仅使用 `core`（裸指针 + 生命周期） |
| `ArcTensor<A, D>` | ✅ | 使用 `alloc::sync::Arc`，需 `no_std + alloc` |
| 静态维度 `Ix0`~`Ix6` | ✅ | 栈分配，无堆依赖 |
| 动态维度 `IxDyn` | ✅ | 使用 `alloc::vec::Vec`，需 `no_std + alloc` |
| `from_raw_parts` / `from_raw_parts_mut` | ✅ | 仅使用 `core`，裸指针操作 |
| `LayoutFlags` | ✅ | 裸 `u8` 位标志，无依赖（参见 `06-memory-layout.md §11`） |

条件编译处理：

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

// Owned::from_vec uses alloc::vec::Vec
// ArcRepr uses alloc::sync::Arc
// IxDyn uses alloc::vec::Vec for dynamic dimensions
// ViewRepr/ViewMutRepr are pure core — no alloc needed
```

---

## 11. 与其他模块的交互

### 11.1 与 storage 模块的接口

```rust
// TensorBase obtains element type via Storage trait's associated type
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn as_ptr(&self) -> *const A {
        unsafe { self.storage.as_ptr().add(self.offset) }
    }
}
```

### 11.2 与 dimension 模块的接口

```rust
// Dimension trait provides shape and stride operations
impl<S, D> TensorBase<S, D>
where
    D: Dimension,
{
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }

    pub fn len(&self) -> usize {
        self.shape.size()
    }
}
```

### 11.3 与 layout 模块的接口

```rust
// Layout module provides stride computation and contiguity checks
// TensorBase computes LayoutFlags during construction
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    pub fn from_shape_vec(shape: D, data: Vec<A>) -> Result<Self, XenonError> {
        if data.len() != shape.size() {
            return Err(ShapeError);
        }
        let strides = layout::compute_f_strides(&shape);
        let ptr = data.as_ptr();
        let flags = layout::compute_flags(&shape, &strides, ptr);
        Ok(Self {
            storage: Owned::from_vec_aligned(data),  // Copies data into 64-byte aligned memory (O(n))
            shape,
            strides,  // isize → D conversion
            offset: 0,
            flags,
        })
    }
}
```

---

## 附录 A：完整类型关系图

```
                        TensorBase<S, D>
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    TensorBase<        TensorBase<          TensorBase<
      Owned<A>,         ViewRepr<           ViewMutRepr<
         D>            &'a A>, D>          &'a mut A>, D>
          │                   │                   │
          ▼                   ▼                   ▼
      Tensor<A,D>      TensorView<'a,A,D>  TensorViewMut<'a,A,D>
          │                   │                   │
    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
    │           │       │           │       │           │
 Tensor1<A> TensorD<A> TensorView1 TensorViewD TensorViewMut1 TensorViewMutD
    │           │       │           │       │           │
   ...         ...     ...         ...     ...         ...
```

## 附录 B：命名约定速查

| 模式 | 示例 | 含义 |
|------|------|------|
| `Tensor{N}` | `Tensor2<A>` | N 维拥有型数组 |
| `TensorD` | `TensorD<A>` | 动态维度拥有型数组 |
| `TensorView{N}` | `TensorView2<'a, A>` | N 维不可变视图 |
| `TensorViewMut{N}` | `TensorViewMut2<'a, A>` | N 维可变视图 |
| `ArcTensor{N}` | `ArcTensor2<A>` | N 维 Arc 共享数组 |

## 附录 C：数据流图

```
用户调用 zeros::<f64, Ix2>([3, 4])
    │
    ├── Dimension::ndim()         → 2
    ├── Dimension::slice()        → [3, 4]
    ├── 计算总元素数               → 12
    ├── 计算步长 (F-order)         → [1, 3]
    ├── 对齐分配 12 * 8 = 96 字节  → 64 字节对齐
    ├── 计算 LayoutFlags           → F_CONTIGUOUS | ALIGNED
    └── 返回 TensorBase<Owned<f64>, Ix2>
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
