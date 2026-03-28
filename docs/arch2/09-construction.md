# 构造与转换模块设计

> 文档编号: 09 | 模块: `src/construction.rs` + `src/conversion.rs` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `05-storage.md`, `06-layout.md`, `07-tensor-core.md`, `08-error.md`, 需求说明书 §13

---

## 1. 模块定位

构造与转换模块是 Xenon 的 API 层核心入口之一，负责：

1. **数组构造** — 通过自由函数（`zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `diag`, `arange`, `linspace`, `logspace`, `from_vec`, `from_slice`, `from_raw_parts`）创建张量
2. **类型转换** — 通过 `TensorBase` 上的方法（`cast`, `to_owned`, `to_arc`, `view`, `view_mut`, `into_owned`）实现存储模式和元素类型的转换
3. **运算符重载** — 通过 `core::ops` trait（`Add`, `Sub`, `Mul`, `Div`, `Neg`, `Not`, `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr` 及其 `Assign` 变体）提供张量级算术运算

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 张量构造函数 | `zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `diag`, `arange`, `linspace`, `logspace` | 迭代器构造（由 `iter/` 模块提供） |
| 数据导入 | `from_vec`, `from_slice`, `from_raw_parts` | 文件 I/O |
| 存储类型转换 | `to_owned`, `to_arc`, `view`, `view_mut`, `into_owned` | 布局转换（由 `to_f_contiguous` 等，见 `shape/` 模块） |
| 元素类型转换 | `cast`（`f64` → `f32` 等精度变换） | 自动类型提升（Xenon 不支持） |
| 逐元素运算符 | `Add/Sub/Mul/Div/Neg/Not/BitAnd/BitOr/BitXor/Shl/Shr` + Assign 变体 | 矩阵运算（由 `ops/matrix.rs` 提供）、归约（由 `ops/reduction.rs` 提供） |

---

## 2. 文件位置

```
src/
  construction.rs    # 构造函数：zeros, ones, full, empty, eye, identity, diag, arange, linspace, logspace, from_vec, from_slice, from_raw_parts
  conversion.rs      # 类型转换：cast, From impls, 运算符重载 (Add, Sub, Mul, Div, Neg, Not, BitAnd, BitOr, BitXor, Shl, Shr + Assign 变体)
```

在 `src/lib.rs` 中的声明与 re-export：

```rust
pub mod construction;
pub mod conversion;

// 构造函数 re-export
pub use crate::construction::{
    zeros, ones, full, empty, eye, identity, diag,
    arange, linspace, logspace,
    from_vec, from_slice, from_raw_parts,
};
```

**文件拆分理由：** `construction.rs` 约为 800 行，专注于数据创建的 13 个自由函数；`conversion.rs` 约为 1200 行，包含 `cast` 方法、`From` impl 和大量运算符 trait impl。两者职责分明、无交叉依赖，拆分后各自处于合理行数范围。

---

## 3. 依赖关系

### 3.1 construction.rs 依赖

```
construction.rs
  ├── crate::tensor       # TensorBase<S, D>, Tensor<A, D>, TensorView, type aliases
  ├── crate::element      # Element, Numeric, RealScalar (用于约束元素类型)
  ├── crate::storage      # Owned<A>, ViewRepr, RawStorage, Storage
  ├── crate::layout       # Order, LayoutFlags, compute_strides, compute_layout_flags, shape_to_elem_count
  ├── crate::dimension    # Dimension, Ix1, Ix2, IxDyn
  ├── crate::error        # TensorError, Result
  └── alloc::vec::Vec     # no_std 兼容的 Vec
```

### 3.2 conversion.rs 依赖

```
conversion.rs
  ├── crate::tensor       # TensorBase<S, D>, Tensor<A, D>, TensorView, TensorViewMut, ArcTensor
  ├── crate::element      # Element, Numeric, RealScalar, ComplexScalar (cast 目标类型约束)
  ├── crate::storage      # Owned<A>, ViewRepr, ViewMutRepr, ArcRepr, Storage, RawStorage
  ├── crate::layout       # Order, LayoutFlags, compute_strides, compute_layout_flags
  ├── crate::dimension    # Dimension
  ├── crate::error        # TensorError, Result
  ├── crate::complex      # Complex<T> (实数 ↔ 复数 cast)
  ├── core::ops           # Add, Sub, Mul, Div, Neg, Not, BitAnd, BitOr, BitXor, Shl, Shr, AddAssign, ...
  └── alloc::sync::Arc    # no_std 兼容的 Arc
```

### 3.3 被依赖关系

```
construction.rs / conversion.rs
  └── 被以下模块使用：
        ├── tests/construction.rs         # 集成测试
        ├── tests/arithmetic.rs           # 运算符集成测试
        ├── tests/edge_cases.rs           # 边界测试
        ├── benches/construction.rs       # 基准测试
        ├── benches/element_ops.rs        # 逐元素运算基准
        └── examples/basics.rs            # 示例
```

---

## 4. 公共 API 设计

### 4.1 构造函数（construction.rs 中的自由函数）

#### 4.1.1 `zeros`

```rust
/// Creates a tensor filled with zeros.
///
/// The tensor is allocated with 64-byte alignment and uses F-order strides
/// by default.
///
/// # Type Parameters
///
/// * `A` - Element type (must implement `Element` to provide `zero()`)
/// * `D` - Dimension type (inferred from the `shape` argument)
///
/// # Arguments
///
/// * `shape` - The shape of the tensor. Implements `Into<D>` so accepts
///   `[usize; N]`, `Vec<usize>`, tuples, etc.
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{zeros, Tensor, Ix2};
/// let a: Tensor<f64, Ix2> = zeros([3, 4]);
/// assert_eq!(a.shape(), &[3, 4]);
/// assert_eq!(a[[0, 0]], 0.0);
/// ```
pub fn zeros<A, D>(shape: D) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    let len = shape.size();
    let storage = Owned::<A>::zeros(len);
    let strides = crate::layout::compute_strides(shape.slice(), Order::default());
    let strides = D::from_slice_strides(&strides);
    unsafe { TensorBase::from_storage_unchecked(storage, shape, strides, 0) }
}
```

#### 4.1.2 `ones`

```rust
/// Creates a tensor filled with ones.
///
/// # Type Parameters
///
/// * `A` - Element type (must implement `Element` to provide `one()`)
/// * `D` - Dimension type
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{ones, Tensor, Ix2};
/// let a: Tensor<f64, Ix2> = ones([2, 3]);
/// assert_eq!(a[[1, 2]], 1.0);
/// ```
pub fn ones<A, D>(shape: D) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    full(shape, A::one())
}
```

#### 4.1.3 `full`

```rust
/// Creates a tensor filled with a constant value.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
/// * `value` - The fill value for every element.
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{full, Tensor, Ix1};
/// let a: Tensor<f32, Ix1> = full(5, 3.14);
/// assert_eq!(a[[2]], 3.14);
/// ```
pub fn full<A, D>(shape: D, value: A) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    let len = shape.size();
    let storage = Owned::<A>::from_elem(len, value);
    let strides = crate::layout::compute_strides(shape.slice(), Order::default());
    let strides = D::from_slice_strides(&strides);
    unsafe { TensorBase::from_storage_unchecked(storage, shape, strides, 0) }
}
```

#### 4.1.4 `empty`

```rust
/// Creates an uninitialized tensor.
///
/// The tensor is allocated with 64-byte alignment but elements are **not**
/// initialized. Reading from an empty tensor is undefined behavior unless
/// the elements are written first.
///
/// Use this when you will immediately overwrite all elements (e.g., as an
/// output buffer for a computation).
///
/// # Safety
///
/// The caller must ensure all elements are initialized before reading.
/// Prefer `zeros()` or `full()` for safe construction.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Panics
///
/// Panics if the total number of elements overflows `usize`.
///
/// # Examples
///
/// ```ignore
/// use xenon::{empty, Tensor, Ix2};
/// let mut a: Tensor<f64, Ix2> = unsafe { empty([3, 4]) };
/// // MUST write to all elements before reading!
/// for idx in 0..3 {
///     for jdx in 0..4 {
///         a[[idx, jdx]] = (idx * 4 + jdx) as f64;
///     }
/// }
/// ```
pub unsafe fn empty<A, D>(shape: D) -> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    let len = shape.size();
    let storage = Owned::<A>::uninitialized(len);
    let strides = crate::layout::compute_strides(shape.slice(), Order::default());
    let strides = D::from_slice_strides(&strides);
    TensorBase::from_storage_unchecked(storage, shape, strides, 0)
}
```

#### 4.1.5 `eye`

```rust
/// Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.
///
/// For non-square matrices, the diagonal runs from `[0, 0]` to
/// `[min(n, m) - 1, min(n, m) - 1]`.
///
/// # Arguments
///
/// * `n` - Number of rows.
/// * `m` - Number of columns.
///
/// # Panics
///
/// Panics if `n * m` overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{eye, Tensor, Ix2};
/// let a: Tensor<f64, Ix2> = eye(3, 3);
/// assert_eq!(a[[0, 0]], 1.0);
/// assert_eq!(a[[0, 1]], 0.0);
/// assert_eq!(a[[1, 1]], 1.0);
/// ```
pub fn eye<A>(n: usize, m: usize) -> Tensor<A, Ix2>
where
    A: Element,
{
    let mut result = zeros(Ix2(n, m));
    let diag_len = n.min(m);
    for i in 0..diag_len {
        result[[i, i]] = A::one();
    }
    result
}
```

#### 4.1.6 `identity`

```rust
/// Creates a square identity matrix.
///
/// Equivalent to `eye(n, n)`.
///
/// # Arguments
///
/// * `n` - Number of rows and columns.
///
/// # Panics
///
/// Panics if `n * n` overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::{identity, Tensor, Ix2};
/// let a: Tensor<f32, Ix2> = identity(3);
/// assert_eq!(a[[2, 2]], 1.0);
/// assert_eq!(a[[0, 1]], 0.0);
/// ```
pub fn identity<A>(n: usize) -> Tensor<A, Ix2>
where
    A: Element,
{
    eye(n, n)
}
```

#### 4.1.7 `diag`

```rust
/// Creates a 2-D square tensor with the given 1-D tensor on its diagonal.
///
/// # Arguments
///
/// * `diagonal` - A 1-D tensor whose values will be placed on the diagonal.
///
/// # Panics
///
/// Panics if `n * n` overflows `usize`, where `n = diagonal.len()`.
///
/// # Examples
///
/// ```
/// use xenon::{diag, Tensor1, Tensor2, Ix1, Ix2};
/// let d: Tensor1<f64, Ix1> = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
/// let a: Tensor2<f64, Ix2> = diag(&d);
/// assert_eq!(a[[0, 0]], 1.0);
/// assert_eq!(a[[1, 1]], 2.0);
/// assert_eq!(a[[0, 1]], 0.0);
/// ```
pub fn diag<A>(diagonal: &Tensor<A, Ix1>) -> Tensor<A, Ix2>
where
    A: Element,
{
    let n = diagonal.len();
    let mut result = zeros(Ix2(n, n));
    for i in 0..n {
        result[[i, i]] = diagonal[[i]];
    }
    result
}
```

#### 4.1.8 `arange`

```rust
/// Returns evenly spaced values within a given interval.
///
/// Values are generated from `start` (inclusive) to `stop` (exclusive),
/// separated by `step`. This is the tensor equivalent of Python's
/// `range(start, stop, step)`.
///
/// # Arguments
///
/// * `start` - Start value (inclusive).
/// * `stop` - Stop value (exclusive).
/// * `step` - Step size. Must be non-zero.
///
/// # Panics
///
/// Panics if `step == 0`.
/// Panics if the computed length overflows `usize`.
///
/// # Examples
///
/// ```
/// use xenon::arange;
/// let a = arange(0.0, 1.0, 0.25);
/// // a ≈ [0.0, 0.25, 0.5, 0.75]
/// assert_eq!(a.len(), 4);
/// ```
pub fn arange<A>(start: A, stop: A, step: A) -> Tensor<A, Ix1>
where
    A: RealScalar,
{
    assert!(step != A::zero(), "arange: step must not be zero");
    let len = if step > A::zero() {
        ((stop - start) / step).ceil().to_usize().unwrap_or(0)
    } else {
        ((start - stop) / (-step)).ceil().to_usize().unwrap_or(0)
    };
    let mut storage = Owned::<A>::uninitialized(len);
    let mut val = start;
    for i in 0..len {
        unsafe {
            storage.as_mut_ptr().add(i).write(val);
        }
        val = val + step;
    }
    let strides = Ix1(1);
    unsafe { TensorBase::from_storage_unchecked(storage, Ix1(len), strides, 0) }
}
```

#### 4.1.9 `linspace`

```rust
/// Returns evenly spaced numbers over a specified interval.
///
/// Generates `num` evenly spaced samples between `start` (inclusive)
/// and `stop` (inclusive). Equivalent to NumPy's `linspace`.
///
/// # Arguments
///
/// * `start` - Start value (inclusive).
/// * `stop` - Stop value (inclusive).
/// * `num` - Number of samples to generate. Must be >= 1.
///
/// # Panics
///
/// Panics if `num == 0`.
///
/// # Examples
///
/// ```
/// use xenon::linspace;
/// let a = linspace(0.0, 1.0, 5);
/// // a = [0.0, 0.25, 0.5, 0.75, 1.0]
/// assert_eq!(a.len(), 5);
/// ```
pub fn linspace<A>(start: A, stop: A, num: usize) -> Tensor<A, Ix1>
where
    A: RealScalar,
{
    assert!(num > 0, "linspace: num must be >= 1");
    let mut storage = Owned::<A>::uninitialized(num);
    if num == 1 {
        unsafe { storage.as_mut_ptr().write(stop); }
    } else {
        let step = (stop - start) / A::from_usize(num - 1).unwrap();
        for i in 0..num {
            unsafe {
                storage.as_mut_ptr().add(i).write(start + step * A::from_usize(i).unwrap());
            }
        }
    }
    let strides = Ix1(1);
    unsafe { TensorBase::from_storage_unchecked(storage, Ix1(num), strides, 0) }
}
```

#### 4.1.10 `logspace`

```ruby
/// Returns numbers spaced evenly on a log scale.
///
/// Generates `num` samples between `base^start` and `base^stop`,
/// evenly spaced in the log domain.
///
/// # Arguments
///
/// * `start` - Start exponent (inclusive).
/// * `stop` - Stop exponent (inclusive).
/// * `num` - Number of samples. Must be >= 1.
/// * `base` - The base of the log space.
///
/// # Panics
///
/// Panics if `num == 0`.
///
/// # Examples
///
/// ```
/// use xenon::logspace;
/// let a = logspace(0.0, 3.0, 4, 10.0);
/// // a = [1.0, 10.0, 100.0, 1000.0]
/// assert_eq!(a.len(), 4);
/// ```
pub fn logspace<A>(start: A, stop: A, num: usize, base: A) -> Tensor<A, Ix1>
where
    A: RealScalar,
{
    assert!(num > 0, "logspace: num must be >= 1");
    // Use linspace to generate exponents, then raise base to those powers
    let exponents = linspace(start, stop, num);
    let mut storage = Owned::<A>::uninitialized(num);
    for i in 0..num {
        unsafe {
            storage.as_mut_ptr().add(i).write(base.powf(exponents[[i]]));
        }
    }
    let strides = Ix1(1);
    unsafe { TensorBase::from_storage_unchecked(storage, Ix1(num), strides, 0) }
}
```

#### 4.1.11 `from_vec`

```rust
/// Creates a 1-D tensor from a `Vec<A>`.
///
/// The data is moved into the tensor (zero-copy when the Vec's allocation
/// is already 64-byte aligned). Otherwise, a new aligned buffer is allocated
/// and the data is copied.
///
/// # Arguments
///
/// * `vec` - The data vector. Length determines the tensor shape.
///
/// # Panics
///
/// Panics if `vec.len()` * `size_of::<A>()` overflows `isize`.
///
/// # Examples
///
/// ```
/// use xenon::from_vec;
/// let a = from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(a.shape(), &[4]);
/// assert_eq!(a[[2]], 3.0);
/// ```
pub fn from_vec<A>(vec: Vec<A>) -> Tensor<A, Ix1>
where
    A: Element,
{
    let len = vec.len();
    let storage = Owned::<A>::from_vec(vec);
    let strides = Ix1(1);
    unsafe { TensorBase::from_storage_unchecked(storage, Ix1(len), strides, 0) }
}
```

#### 4.1.12 `from_slice`

```rust
/// Creates a 1-D tensor by copying data from a slice.
///
/// # Arguments
///
/// * `data` - The source slice. Length determines the tensor shape.
///
/// # Panics
///
/// Panics if `data.len()` * `size_of::<A>()` overflows `isize`.
///
/// # Examples
///
/// ```
/// use xenon::from_slice;
/// let a = from_slice(&[1, 2, 3, 4, 5]);
/// assert_eq!(a.shape(), &[5]);
/// ```
pub fn from_slice<A>(data: &[A]) -> Tensor<A, Ix1>
where
    A: Element,
{
    let len = data.len();
    let storage = Owned::<A>::from_slice(data);
    let strides = Ix1(1);
    unsafe { TensorBase::from_storage_unchecked(storage, Ix1(len), strides, 0) }
}
```

#### 4.1.13 `from_raw_parts`

```rust
/// Creates a tensor view from raw pointer components.
///
/// This is the unsafe escape hatch for integrating with external memory
/// (FFI, BLAS, custom allocators).
///
/// # Safety
///
/// The caller must ensure that:
/// - `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`.
/// - The memory range `[ptr, ptr + offset + max_indexable_offset)` is valid
///   and all accessible elements are properly initialized.
/// - The memory remains valid for lifetime `'a`.
/// - No mutable references to the same memory exist during `'a`.
/// - `shape` and `strides` have the same length.
/// - All indexable offsets fall within valid memory.
///
/// # Arguments
///
/// * `ptr` - Pointer to the start of the data buffer.
/// * `shape` - The tensor dimensions.
/// * `strides` - Element strides (must match `shape` length).
/// * `offset` - Data start offset in element units.
///
/// # Examples
///
/// ```ignore
/// use xenon::from_raw_parts;
/// let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let view = unsafe {
///     from_raw_parts(data.as_ptr(), [2, 3], [1, 2], 0)
/// };
/// assert_eq!(view.shape(), &[2, 3]);
/// ```
pub unsafe fn from_raw_parts<'a, A, D>(
    ptr: *const A,
    shape: D,
    strides: D,
    offset: usize,
) -> TensorView<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    TensorView::from_raw_parts(ptr, shape, strides, offset)
}
```

### 4.2 类型转换方法（conversion.rs 中的 TensorBase 方法）

#### 4.2.1 `cast` — 元素类型转换

```rust
/// Element type conversion for tensors.
///
/// Performs an explicit cast of every element from type `A` to type `B`.
/// No automatic type promotion is performed — the conversion must be
/// explicitly requested.
///
/// # Type Parameters
///
/// * `A` - Source element type.
/// * `B` - Target element type.
/// * `S` - Source storage type.
/// * `D` - Dimension type (preserved).
///
/// # Cast Precision Rules
///
/// | Direction | Behavior |
/// |-----------|----------|
/// | Float → Float (high→low) | IEEE 754 round-to-nearest-even |
/// | Float → Float (low→high) | Exact, no precision loss |
/// | Float → Integer | Truncate toward zero, saturating on overflow |
/// | Integer → Float | Round-to-nearest-even |
/// | Integer → Integer (narrowing) | Saturating cast |
/// | NaN → Integer | Result is 0 |
/// | Inf → Integer | Saturate to MAX/MIN |
/// | bool → Numeric | true = 1, false = 0 |
/// | Numeric → bool | Non-zero = true, zero = false |
/// | Real → Complex | Imaginary part is 0 |
/// | Complex → Real | Not allowed (use `.re()` explicitly) |
///
/// # Examples
///
/// ```
/// use xenon::{from_vec, Tensor};
/// let a: Tensor<f64, _> = from_vec(vec![1.5, 2.7, 3.9]);
/// let b: Tensor<i32, _> = a.cast::<i32>();
/// assert_eq!(b[[0]], 1);  // truncated toward zero
/// ```
impl<S, A, D> TensorBase<S, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    pub fn cast<B>(self) -> Tensor<B, D>
    where
        B: Element + From<A>,
    {
        let len = self.len();
        let mut new_storage = Owned::<B>::uninitialized(len);
        for (i, elem) in self.iter().enumerate() {
            unsafe {
                new_storage.as_mut_ptr().add(i).write(B::from(*elem));
            }
        }
        let new_strides = D::from_slice_strides(
            &crate::layout::compute_strides(self.shape(), Order::default())
        );
        unsafe {
            TensorBase::from_storage_unchecked(new_storage, self.shape.clone(), new_strides, 0)
        }
    }
}
```

**`Cast` trait 设计：** 为避免要求所有类型对实现 `From`/`Into`，定义内部 `Cast` trait 处理 Xenon 特有的转换规则（NaN→0, Inf→饱和等）：

```rust
/// Internal trait for safe element-wise type casting.
///
/// Unlike `From`/`Into`, this trait handles edge cases like
/// NaN→0, Inf→saturate, and bool↔numeric conversions.
pub trait Cast<T>: Element {
    /// Performs the element-wise cast.
    fn cast(self) -> T;
}

// Implementations for standard type pairs:
impl Cast<f32> for f64 { ... }
impl Cast<f64> for f32 { ... }
impl Cast<i32> for f64 { ... }
impl Cast<f64> for i32 { ... }
impl Cast<i64> for f64 { ... }
impl Cast<f64> for i64 { ... }
impl Cast<i32> for i64 { ... }
impl Cast<i64> for i32 { ... }
impl Cast<bool> for f64 { ... }
impl Cast<f64> for bool { ... }
// ... all numeric type pairs
```

`cast` 方法使用 `Cast` trait 而非标准 `From`：

```rust
pub fn cast<B>(self) -> Tensor<B, D>
where
    B: Element,
    A: Cast<B>,
{ ... }
```

#### 4.2.2 `to_owned` — 深拷贝为拥有存储

已在 `07-tensor-core.md` §4.5–4.8 中为 `Owned`、`View`、`ViewMut`、`ArcRepr` 各定义。此处不重复。

#### 4.2.3 `to_arc` — 转换为 Arc 共享存储

```rust
/// Converts the tensor to an `ArcTensor` (shared ownership).
///
/// For `Owned`: wraps the storage in `Arc` (O(1), no copy).
/// For `View`/`ViewMut`: deep-copies the data into an `ArcRepr`.
/// For `ArcRepr`: shallow clone (O(1), refcount + 1).
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    pub fn to_arc(self) -> ArcTensor<A, D> {
        TensorBase {
            storage: ArcRepr::from_owned_storage(self.storage),
            shape: self.shape,
            strides: self.strides,
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }
}

impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    A: Element,
    D: Dimension,
{
    pub fn to_arc(&self) -> ArcTensor<A, D>
    where
        A: Clone,
    {
        let owned = self.to_owned();
        owned.to_arc()
    }
}

impl<'a, A, D> TensorBase<ViewMutRepr<&'a mut A>, D>
where
    A: Element,
    D: Dimension,
{
    pub fn to_arc(self) -> ArcTensor<A, D>
    where
        A: Clone,
    {
        let owned = self.to_owned();
        owned.to_arc()
    }
}

impl<A, D> TensorBase<ArcRepr<A>, D>
where
    A: Element,
    D: Dimension,
{
    pub fn to_arc(self) -> ArcTensor<A, D> {
        self  // Already ArcRepr
    }
}
```

#### 4.2.4 `view` / `view_mut` / `into_owned`

已在 `07-tensor-core.md` §4.5–4.8 中定义。此处仅补充 `to_arc` 作为新增方法。

### 4.3 `From` trait 实现

```rust
// === Vec<A> → Tensor<A, Ix1> ===

impl<A: Element> From<Vec<A>> for Tensor<A, Ix1> {
    fn from(vec: Vec<A>) -> Self {
        from_vec(vec)
    }
}

// === &[A] → Tensor<A, Ix1> ===

impl<A: Element> From<&[A]> for Tensor<A, Ix1> {
    fn from(slice: &[A]) -> Self {
        from_slice(slice)
    }
}

// === [A; N] → Tensor<A, Ix1> ===

impl<A: Element, const N: usize> From<[A; N]> for Tensor<A, Ix1> {
    fn from(arr: [A; N]) -> Self {
        from_vec(Vec::from(arr))
    }
}

// === Owned → ArcRepr (zero-copy) ===

impl<A: Element, D: Dimension> From<Tensor<A, D>> for ArcTensor<A, D> {
    fn from(tensor: Tensor<A, D>) -> Self {
        tensor.to_arc()
    }
}

// === ArcRepr → Owned (potentially zero-copy via try_unwrap) ===

impl<A: Element + Clone, D: Dimension> From<ArcTensor<A, D>> for Tensor<A, D> {
    fn from(tensor: ArcTensor<A, D>) -> Self {
        tensor.into_owned()
    }
}
```

### 4.4 运算符重载（conversion.rs 中的 trait impl）

#### 4.4.1 设计原则

- **所有二元运算符隐式支持广播**：形状不同时尝试广播，广播失败返回 `TensorError`
- **运算结果总是 `Tensor<A, D>`（Owned 存储）**：运算产生新分配，不修改输入
- **标量广播**：标量可参与张量运算（`tensor + 1.0`），通过 `Scalar` wrapper 实现
- **Assign 变体**：`AddAssign` 等就地修改，要求左侧为 `Tensor<A, D>` 或 `TensorViewMut<A, D>`
- **运算分派路径**：连续 → SIMD（若启用），非连续 → 标量

#### 4.4.2 Add / Sub / Mul / Div（二元运算，Tensor × Tensor）

以 `Add` 为例，其余结构相同：

```rust
/// Element-wise addition of two tensors with broadcasting.
///
/// If the shapes are identical, performs direct element-wise addition.
/// If shapes differ, attempts broadcasting before falling back to error.
///
/// # Errors
///
/// Returns `TensorError::BroadcastError` if shapes cannot be broadcast together.
///
/// # Examples
///
/// ```
/// use xenon::{zeros, from_vec};
/// let a = from_vec(vec![1.0, 2.0, 3.0]);
/// let b = from_vec(vec![4.0, 5.0, 6.0]);
/// let c = a + b;
/// assert_eq!(c[[0]], 5.0);
/// ```
impl<A, D> core::ops::Add for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: Self) -> Self::Output {
        // Shape check + broadcast attempt
        binary_op(self, rhs, |a, b| a + b)
    }
}

/// View + View → Tensor
impl<'a, 'b, A, D> core::ops::Add<TensorView<'b, A, D>> for TensorView<'a, A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: TensorView<'b, A, D>) -> Self::Output {
        binary_op(self, rhs, |a, b| a + b)
    }
}

/// Tensor + View → Tensor
impl<'a, A, D> core::ops::Add<TensorView<'a, A, D>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: TensorView<'a, A, D>) -> Self::Output {
        binary_op(self, rhs, |a, b| a + b)
    }
}

/// View + Tensor → Tensor
impl<'a, A, D> core::ops::Add<Tensor<A, D>> for TensorView<'a, A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, rhs: Tensor<A, D>) -> Self::Output {
        binary_op(self, rhs, |a, b| a + b)
    }
}
```

**注：** `Sub`, `Mul`, `Div` 遵循相同的 impl 模式，替换运算闭包为 `|a, b| a - b` / `|a, b| a * b` / `|a, b| a / b`。

#### 4.4.3 标量广播运算

```rust
/// Wrapper type for scalar-tensor operations.
///
/// Enables `tensor + scalar` and `scalar + tensor` syntax.
pub struct Scalar<A>(pub A);

// Tensor + Scalar
impl<A, D> core::ops::Add<Scalar<A>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, scalar: Scalar<A>) -> Self::Output {
        self.map(|x| x + scalar.0)
    }
}

// Scalar + Tensor
impl<A, D> core::ops::Add<Tensor<A, D>> for Scalar<A>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn add(self, tensor: Tensor<A, D>) -> Self::Output {
        tensor.map(|x| self.0 + x)
    }
}
```

#### 4.4.4 Neg / Not（一元运算）

```rust
/// Unary negation.
impl<A, D> core::ops::Neg for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

/// Unary bitwise NOT (for integer types).
impl<A, D> core::ops::Not for Tensor<A, D>
where
    A: Element + core::ops::Not<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn not(self) -> Self::Output {
        self.map(|x| !x)
    }
}
```

#### 4.4.5 BitAnd / BitOr / BitXor（位运算）

```rust
/// Element-wise bitwise AND.
impl<A, D> core::ops::BitAnd for Tensor<A, D>
where
    A: Element + core::ops::BitAnd<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn bitand(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a & b)
    }
}

/// Element-wise bitwise OR.
impl<A, D> core::ops::BitOr for Tensor<A, D>
where
    A: Element + core::ops::BitOr<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn bitor(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a | b)
    }
}

/// Element-wise bitwise XOR.
impl<A, D> core::ops::BitXor for Tensor<A, D>
where
    A: Element + core::ops::BitXor<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn bitxor(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a ^ b)
    }
}
```

#### 4.4.6 Shl / Shr（位移运算）

```rust
/// Element-wise left shift.
impl<A, D> core::ops::Shl for Tensor<A, D>
where
    A: Element + core::ops::Shl<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn shl(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a << b)
    }
}

/// Element-wise right shift.
impl<A, D> core::ops::Shr for Tensor<A, D>
where
    A: Element + core::ops::Shr<Output = A>,
    D: Dimension,
{
    type Output = Tensor<A, D>;

    fn shr(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a >> b)
    }
}
```

#### 4.4.7 Assign 变体

```rust
/// In-place addition (AddAssign).
///
/// Modifies `self` element-wise. Requires mutable (owned or view-mut) storage.
///
/// # Panics
///
/// Panics if shapes are incompatible and broadcasting fails.
impl<A, D> core::ops::AddAssign for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add_assign(&mut self, rhs: Self) {
        assign_op(self, rhs, |a, b| *a = *a + b);
    }
}

/// In-place addition with a view.
impl<'a, A, D> core::ops::AddAssign<TensorView<'a, A, D>> for Tensor<A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add_assign(&mut self, rhs: TensorView<'a, A, D>) {
        assign_op(self, rhs, |a, b| *a = *a + b);
    }
}

/// In-place addition for mutable view.
impl<'a, A, D> core::ops::AddAssign<Tensor<A, D>> for TensorViewMut<'a, A, D>
where
    A: Numeric,
    D: Dimension,
{
    fn add_assign(&mut self, rhs: Tensor<A, D>) {
        assign_op(self, rhs, |a, b| *a = *a + b);
    }
}

// SubAssign, MulAssign, DivAssign follow the same pattern:
// impl core::ops::SubAssign for Tensor<A, D> { ... }
// impl core::ops::MulAssign for Tensor<A, D> { ... }
// impl core::ops::DivAssign for Tensor<A, D> { ... }

// BitAndAssign, BitOrAssign, BitXorAssign for integer types:
// impl core::ops::BitAndAssign for Tensor<A, D> { ... }
// impl core::ops::BitOrAssign for Tensor<A, D> { ... }
// impl core::ops::BitXorAssign for Tensor<A, D> { ... }

// ShlAssign, ShrAssign for integer types:
// impl core::ops::ShlAssign for Tensor<A, D> { ... }
// impl core::ops::ShrAssign for Tensor<A, D> { ... }
```

#### 4.4.8 `bool` 张量的逻辑运算

```rust
/// Logical AND for bool tensors (reuses BitAnd).
/// `true & true` = `true`, all other combinations = `false`.
impl<D> core::ops::BitAnd for Tensor<bool, D>
where
    D: Dimension,
{
    type Output = Tensor<bool, D>;

    fn bitand(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a & b)
    }
}

/// Logical OR for bool tensors (reuses BitOr).
impl<D> core::ops::BitOr for Tensor<bool, D>
where
    D: Dimension,
{
    type Output = Tensor<bool, D>;

    fn bitor(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, |a, b| a | b)
    }
}
```

---

## 5. 内部实现设计

### 5.1 内存分配策略

所有构造函数（除 `from_raw_parts` 和 `empty`）通过 `Owned<A>` 的构造方法分配内存：

| 构造函数 | 分配方法 | 初始化 |
|----------|----------|--------|
| `zeros` | `Owned::<A>::zeros(len)` | 零初始化（`ptr::write_bytes(0)`） |
| `ones` | `Owned::<A>::from_elem(len, A::one())` | 逐元素写入 `one()` |
| `full` | `Owned::<A>::from_elem(len, value)` | 逐元素写入 `value` |
| `empty` | `Owned::<A>::uninitialized(len)` | 不初始化（MaybeUninit） |
| `eye` / `identity` / `diag` | 先 `zeros` 再逐元素覆写 | 混合 |
| `arange` / `linspace` / `logspace` | `Owned::<A>::uninitialized(len)` + 逐元素写入 | 逐元素 |
| `from_vec` | `Owned::<A>::from_vec(vec)` | 移动或复制 |
| `from_slice` | `Owned::<A>::from_slice(data)` | 复制 |

**对齐保证：** 所有 `Owned<A>` 分配使用 64 字节对齐（AVX-512 缓存行）。小数组优化：当 `len * size_of::<A>() <= 64` 时，允许降级到元素自然对齐（由 `Owned` 内部处理）。

### 5.2 F-order 默认步长计算

所有构造函数（`from_raw_parts` 除外）使用 F-order 步长：

```rust
// For shape [M, N, P]:
// F-order strides = [1, M, M*N]
// C-order strides = [N*P, P, 1]

fn compute_default_strides<D: Dimension>(shape: &D) -> D {
    let strides = crate::layout::compute_strides(shape.slice(), Order::default());
    D::from_slice_strides(&strides)
}
```

**F-order 为默认的理由：**
1. BLAS/LAPACK 使用列优先布局
2. Fortran 科学计算传统
3. 与 NumPy 的 `order='F'` 对应

**注意：** `from_raw_parts` 不计算步长——步长由调用方显式提供。

### 5.3 运算分派机制

#### 5.3.1 二元运算分派

```rust
/// Internal dispatch for binary element-wise operations.
///
/// Handles shape checking, broadcasting, and execution path selection.
fn binary_op<S1, S2, A, D, F>(
    lhs: TensorBase<S1, D>,
    rhs: TensorBase<S2, D>,
    op: F,
) -> Tensor<A, D>
where
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
    A: Numeric,
    D: Dimension,
    F: Fn(A, A) -> A,
{
    // 1. Shape check: identical or broadcastable
    let (shape, lhs_view, rhs_view) = if lhs.shape() == rhs.shape() {
        // Fast path: identical shapes, no broadcast needed
        (lhs.shape.clone(), lhs, rhs)
    } else {
        // Attempt broadcast
        broadcast_binary_shapes(&lhs, &rhs)?
    };

    // 2. Allocate output
    let len = shape.size();
    let mut out = Owned::<A>::uninitialized(len);

    // 3. Execute operation
    //    - If both inputs contiguous + simd feature → SIMD path
    //    - If len >= PARALLEL_THRESHOLD + parallel feature → parallel path
    //    - Otherwise → scalar path
    #[cfg(feature = "simd")]
    if lhs_view.is_f_contiguous() && rhs_view.is_f_contiguous() && lhs_view.is_aligned() {
        // SIMD fast path
        simd_binary_op(&lhs_view, &rhs_view, &mut out, &op);
    } else {
        scalar_binary_op(&lhs_view, &rhs_view, &mut out, &op);
    }

    #[cfg(not(feature = "simd"))]
    scalar_binary_op(&lhs_view, &rhs_view, &mut out, &op);

    // 4. Construct output tensor
    let strides = compute_default_strides(&shape);
    unsafe { TensorBase::from_storage_unchecked(out, shape, strides, 0) }
}
```

#### 5.3.2 Assign 运算分派

```rust
/// Internal dispatch for in-place binary operations.
///
/// The left-hand side must be writable (Owned or ViewMut).
fn assign_op<S, SR, A, D, F>(
    lhs: &mut TensorBase<S, D>,
    rhs: TensorBase<SR, D>,
    op: F,
)
where
    S: StorageMut<Elem = A>,
    SR: RawStorage<Elem = A>,
    A: Numeric,
    D: Dimension,
    F: Fn(&mut A, A),
{
    assert_shapes_compatible(lhs.shape(), rhs.shape());
    // Iterate and apply in-place
    for (dst, src) in lhs.iter_mut().zip(rhs.iter()) {
        op(dst, *src);
    }
}
```

#### 5.3.3 标量路径

```rust
/// Scalar (non-SIMD) binary operation.
fn scalar_binary_op<S1, S2, A, D, F>(
    lhs: &TensorBase<S1, D>,
    rhs: &TensorBase<S2, D>,
    out: &mut Owned<A>,
    op: &F,
)
where
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
    A: Numeric,
    D: Dimension,
    F: Fn(A, A) -> A,
{
    for (i, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        unsafe {
            out.as_mut_ptr().add(i).write(op(*a, *b));
        }
    }
}
```

### 5.4 运算符覆盖矩阵

完整的运算符 trait 覆盖：

| 运算符 | trait | 输出类型 | 元素约束 | 存储组合 |
|--------|-------|----------|----------|----------|
| `+` | `Add` | `Tensor<A, D>` | `Numeric` | T×T, T×V, V×T, V×V, T×Scalar, Scalar×T |
| `-` | `Sub` | `Tensor<A, D>` | `Numeric` | 同 Add |
| `*` | `Mul` | `Tensor<A, D>` | `Numeric` | 同 Add |
| `/` | `Div` | `Tensor<A, D>` | `Numeric` | 同 Add |
| `-a` | `Neg` | `Tensor<A, D>` | `Numeric` | T, V |
| `!a` | `Not` | `Tensor<A, D>` | `Element + Not` | T, V |
| `&` | `BitAnd` | `Tensor<A, D>` | `Element + BitAnd` | T×T |
| `\|` | `BitOr` | `Tensor<A, D>` | `Element + BitOr` | T×T |
| `^` | `BitXor` | `Tensor<A, D>` | `Element + BitXor` | T×T |
| `<<` | `Shl` | `Tensor<A, D>` | `Element + Shl` | T×T |
| `>>` | `Shr` | `Tensor<A, D>` | `Element + Shr` | T×T |
| `+=` | `AddAssign` | `()` (in-place) | `Numeric` | T+=T, T+=V, VM+=T, VM+=V |
| `-=` | `SubAssign` | `()` | `Numeric` | 同 AddAssign |
| `*=` | `MulAssign` | `()` | `Numeric` | 同 AddAssign |
| `/=` | `DivAssign` | `()` | `Numeric` | 同 AddAssign |
| `&=` | `BitAndAssign` | `()` | `Element + BitAndAssign` | T+=T |
| `\|=` | `BitOrAssign` | `()` | `Element + BitOrAssign` | T+=T |
| `^=` | `BitXorAssign` | `()` | `Element + BitXorAssign` | T+=T |
| `<<=` | `ShlAssign` | `()` | `Element + ShlAssign` | T+=T |
| `>>=` | `ShrAssign` | `()` | `Element + ShrAssign` | T+=T |

> **T** = `Tensor<A, D>`, **V** = `TensorView<'a, A, D>`, **VM** = `TensorViewMut<'a, A, D>`

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 构造函数（construction.rs）

- [ ] **T1: 创建 construction.rs 文件骨架 + 模块注册**
  - 文件: `src/construction.rs:1-30`
  - 内容: 文件级 doc comment、`use` 导入、模块声明
  - 测试: `cargo check` 编译通过
  - 前置: Phase 2 全部完成（tensor, storage, layout, element, dimension, error）
  - 预计: 10 min

- [ ] **T2: 实现 zeros + ones + full**
  - 文件: `src/construction.rs`
  - 内容: `zeros()`, `ones()`, `full()` 三个函数及完整 doc comments
  - 测试: `test_zeros_shape_correct`, `test_ones_fill_value`, `test_full_custom_value`, `test_zeros_empty_shape`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: 实现 empty**
  - 文件: `src/construction.rs`
  - 内容: `unsafe fn empty()` + Safety 文档节
  - 测试: `test_empty_allocation_succeeds`（仅验证分配，不读取）
  - 前置: T2
  - 预计: 10 min

- [ ] **T4: 实现 eye + identity + diag**
  - 文件: `src/construction.rs`
  - 内容: `eye()`, `identity()`, `diag()` 三个函数
  - 测试: `test_eye_diagonal_ones`, `test_eye_non_square`, `test_identity_square`, `test_diag_from_1d`
  - 前置: T2
  - 预计: 10 min

- [ ] **T5: 实现 arange**
  - 文件: `src/construction.rs`
  - 内容: `arange()` 函数，含正/负步长、空范围边界
  - 测试: `test_arange_positive_step`, `test_arange_negative_step`, `test_arange_empty_range`, `#[should_panic] test_arange_zero_step`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6: 实现 linspace + logspace**
  - 文件: `src/construction.rs`
  - 内容: `linspace()`, `logspace()` 函数
  - 测试: `test_linspace_endpoints`, `test_linspace_single_element`, `test_logspace_powers_of_ten`, `#[should_panic] test_linspace_zero_num`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: 实现 from_vec + from_slice + from_raw_parts**
  - 文件: `src/construction.rs`
  - 内容: `from_vec()`, `from_slice()`, `from_raw_parts()` 函数
  - 测试: `test_from_vec_shape`, `test_from_slice_copies`, `test_from_raw_parts_view`（unsafe 测试）
  - 前置: T2
  - 预计: 10 min

- [ ] **T8: lib.rs 注册 + re-export**
  - 文件: `src/lib.rs`
  - 内容: `pub mod construction;` + 所有构造函数 re-export
  - 测试: `cargo check`，验证 `use xenon::zeros` 编译通过
  - 前置: T1-T7
  - 预计: 5 min

### 6.2 类型转换（conversion.rs）

- [ ] **T9: 创建 conversion.rs 文件骨架**
  - 文件: `src/conversion.rs:1-30`
  - 内容: 文件级 doc comment、`use` 导入、内部 `Cast` trait 定义
  - 测试: `cargo check` 编译通过
  - 前置: T1 (construction.rs 骨架)
  - 预计: 10 min

- [ ] **T10: 实现 Cast trait（所有数值类型对）**
  - 文件: `src/conversion.rs`
  - 内容: `Cast` trait 定义 + 所有 `impl Cast<T> for S` 类型对（f64→f32, f64→i32, i32→f64, bool→f64 等）
  - 测试: `test_cast_f64_to_f32`, `test_cast_f64_to_i32_truncate`, `test_cast_nan_to_i32_zero`, `test_cast_inf_to_i32_saturate`, `test_cast_bool_to_numeric`
  - 前置: T9
  - 预计: 10 min

- [ ] **T11: 实现 cast 方法 + From trait impls**
  - 文件: `src/conversion.rs`
  - 内容: `TensorBase::cast()` 方法、`From<Vec<A>>`、`From<&[A]>`、`From<[A; N]>`、`From<Tensor> for ArcTensor`、`From<ArcTensor> for Tensor`
  - 测试: `test_cast_preserves_shape`, `test_from_vec`, `test_from_slice`, `test_from_array`, `test_owned_to_arc_zero_copy`
  - 前置: T10
  - 预计: 10 min

- [ ] **T12: 实现 to_arc 方法**
  - 文件: `src/conversion.rs`
  - 内容: 四种存储类型各自的 `to_arc()` 方法
  - 测试: `test_owned_to_arc`, `test_view_to_arc_deep_copy`, `test_viewmut_to_arc`, `test_arc_to_arc_identity`
  - 前置: T11
  - 预计: 10 min

- [ ] **T13: lib.rs 注册 conversion 模块**
  - 文件: `src/lib.rs`
  - 内容: `pub mod conversion;`
  - 测试: `cargo check`
  - 前置: T9-T12
  - 预计: 5 min

### 6.3 运算符重载（conversion.rs 续）

- [ ] **T14: 实现二元运算分派框架（binary_op / assign_op）**
  - 文件: `src/conversion.rs`
  - 内容: `binary_op()` 内部函数、`assign_op()` 内部函数、`scalar_binary_op()` 标量路径、形状校验逻辑
  - 测试: `test_binary_op_identical_shapes`, `test_binary_op_shape_mismatch_error`
  - 前置: T13
  - 预计: 10 min

- [ ] **T15: 实现 Add / Sub / Mul / Div trait（Tensor × Tensor）**
  - 文件: `src/conversion.rs`
  - 内容: 四种运算符的 `impl Op for Tensor<A, D>`、`impl Op for TensorView` 交叉组合
  - 测试: `test_add_tensors`, `test_sub_tensors`, `test_mul_tensors`, `test_div_tensors`, `test_add_view_and_tensor`
  - 前置: T14
  - 预计: 10 min

- [ ] **T16: 实现标量广播（Tensor + Scalar, Scalar + Tensor）**
  - 文件: `src/conversion.rs`
  - 内容: `Scalar<A>` wrapper 类型、标量参与的 Add/Sub/Mul/Div impl
  - 测试: `test_tensor_add_scalar`, `test_scalar_mul_tensor`, `test_scalar_sub_tensor`
  - 前置: T15
  - 预计: 10 min

- [ ] **T17: 实现 Neg / Not / BitAnd / BitOr / BitXor / Shl / Shr**
  - 文件: `src/conversion.rs`
  - 内容: 一元运算和位运算 trait impl
  - 测试: `test_neg_tensor`, `test_not_int_tensor`, `test_bitand_int_tensors`, `test_shl_int_tensors`
  - 前置: T15
  - 预计: 10 min

- [ ] **T18: 实现 Assign 变体（AddAssign / SubAssign / MulAssign / DivAssign）**
  - 文件: `src/conversion.rs`
  - 内容: `AddAssign` 等 trait impl，包括 Tensor+=Tensor、Tensor+=View、ViewMut+=Tensor 组合
  - 测试: `test_add_assign_tensor`, `test_mul_assign_view_mut`, `test_sub_assign_in_place`
  - 前置: T15
  - 预计: 10 min

- [ ] **T19: 实现 BitAssign / ShiftAssign 变体**
  - 文件: `src/conversion.rs`
  - 内容: `BitAndAssign`, `BitOrAssign`, `BitXorAssign`, `ShlAssign`, `ShrAssign` impl
  - 测试: `test_bitand_assign`, `test_shl_assign`
  - 前置: T18
  - 预计: 10 min

### 6.4 集成与文档

- [ ] **T20: 补全所有 doc comments + doc tests**
  - 文件: `src/construction.rs`, `src/conversion.rs`
  - 内容: 所有 pub 函数的完整 `///` 文档（含 `# Arguments`, `# Panics`, `# Examples`, `# Safety` 节）
  - 测试: `cargo doc --no-deps` 无 warning
  - 前置: T1-T19
  - 预计: 10 min

---

## 7. 测试计划

### 7.1 单元测试（`#[cfg(test)] mod tests`）

#### construction.rs 单元测试

| 测试函数 | 覆盖目标 | 关键断言 |
|----------|----------|----------|
| `test_zeros_shape_correct` | `zeros` 基本功能 | shape 正确，所有元素 == 0 |
| `test_zeros_2d_f_contiguous` | `zeros` 布局 | F-contiguous, aligned |
| `test_zeros_empty` | `zeros` 空张量 | shape=[0,3], len=0, 不 panic |
| `test_zeros_scalar` | `zeros` 0-D | shape=[], len=1 |
| `test_ones_fill_value` | `ones` 基本功能 | 所有元素 == 1 |
| `test_full_custom_value` | `full` 基本功能 | 所有元素 == fill value |
| `test_eye_diagonal_ones` | `eye` 对角线 | diag == 1, off-diag == 0 |
| `test_eye_non_square` | `eye` 非方阵 | shape=[2,3], diag len=2 |
| `test_identity_square` | `identity` | 等价于 eye(n, n) |
| `test_diag_from_1d` | `diag` 基本功能 | diag 值来自输入，其余为 0 |
| `test_arange_positive_step` | `arange` 正步长 | [0.0, 0.5, 1.0, 1.5] |
| `test_arange_negative_step` | `arange` 负步长 | [3.0, 2.0, 1.0] |
| `test_arange_empty_range` | `arange` 空范围 | len == 0 |
| `test_arange_zero_step_panic` | `arange` step=0 | `#[should_panic]` |
| `test_linspace_endpoints` | `linspace` 端点包含 | 首尾 == start/stop |
| `test_linspace_single_element` | `linspace` num=1 | 单个元素 == stop |
| `test_linspace_zero_num_panic` | `linspace` num=0 | `#[should_panic]` |
| `test_logspace_powers_of_ten` | `logspace` 基本功能 | [1, 10, 100, 1000] |
| `test_from_vec_shape` | `from_vec` 形状 | shape == [len] |
| `test_from_vec_data_correct` | `from_vec` 数据 | 逐元素比较 |
| `test_from_slice_copies` | `from_slice` 复制 | 修改源不影响 tensor |
| `test_from_raw_parts_view` | `from_raw_parts` | unsafe 测试，验证 shape/strides |
| `test_empty_allocation` | `empty` 分配 | 仅验证 shape/len，不读数据 |

#### conversion.rs 单元测试

| 测试函数 | 覆盖目标 | 关键断言 |
|----------|----------|----------|
| `test_cast_f64_to_f32` | 精度降级 | 逐元素比较（epsilon 容差） |
| `test_cast_f32_to_f64` | 精度提升 | 精确匹配 |
| `test_cast_f64_to_i32_truncate` | 浮点→整数 | 向零截断 |
| `test_cast_f64_to_i32_overflow` | 浮点→整数溢出 | 饱和到 i32::MAX |
| `test_cast_nan_to_i32` | NaN→整数 | 结果 == 0 |
| `test_cast_inf_to_i32` | Inf→整数 | 饱和到 i32::MAX/MIN |
| `test_cast_bool_to_f64` | bool→数值 | true=1.0, false=0.0 |
| `test_cast_i32_to_bool` | 数值→bool | 0=false, 非零=true |
| `test_cast_preserves_shape` | cast 形状 | shape 不变 |
| `test_from_vec_via_from_trait` | `From<Vec>` | 等价于 `from_vec()` |
| `test_from_array` | `From<[A; N]>` | 正确长度和数据 |
| `test_owned_to_arc` | `to_arc` | Arc refcount == 1 |
| `test_view_to_arc_deep_copy` | View→Arc | 深拷贝，指针不同 |
| `test_arc_to_owned_no_copy` | Arc→Owned (refcount==1) | try_unwrap 成功 |
| `test_arc_to_owned_copies` | Arc→Owned (refcount>1) | 深拷贝 |
| `test_add_tensors` | `Add` 基本功能 | 逐元素正确 |
| `test_sub_tensors` | `Sub` 基本功能 | 逐元素正确 |
| `test_mul_tensors` | `Mul` 基本功能 | 逐元素正确 |
| `test_div_tensors` | `Div` 基本功能 | 逐元素正确 |
| `test_add_tensors_shape_mismatch` | `Add` 形状不匹配 | panic（或返回错误，取决于策略） |
| `test_add_view_and_tensor` | `Add` 跨存储 | View + Tensor == 正确结果 |
| `test_tensor_add_scalar` | 标量广播 | tensor + 1.0 正确 |
| `test_scalar_mul_tensor` | 标量广播 | 2.0 * tensor 正确 |
| `test_neg_tensor` | `Neg` | 逐元素取反 |
| `test_not_int_tensor` | `Not` (整数) | 逐元素取反 |
| `test_bitand_int_tensors` | `BitAnd` | 逐元素正确 |
| `test_shl_int_tensors` | `Shl` | 逐元素正确 |
| `test_add_assign_in_place` | `AddAssign` | 原地修改，不重新分配 |
| `test_mul_assign_view_mut` | `MulAssign` on ViewMut | 就地修改 |
| `test_add_assign_shape_mismatch` | `AddAssign` 形状不匹配 | panic |

### 7.2 集成测试

#### `tests/construction.rs`

| 测试函数 | 场景 | 预期 |
|----------|------|------|
| `test_zeros_2d_integration` | 跨模块：zeros + shape + view | 创建→查看→验证 |
| `test_from_vec_cast_integration` | 构造→转换链 | `from_vec → cast → to_arc → view → to_owned` |
| `test_eye_arange_operations` | 构造→运算 | `eye + arange` 等运算链 |
| `test_linspace_cast_integration` | linspace→cast | `linspace<f64> → cast<f32>` |
| `test_logspace_base2` | logspace base=2 | [1, 2, 4, 8] |

#### `tests/arithmetic.rs`

| 测试函数 | 场景 | 预期 |
|----------|------|------|
| `test_add_broadcast_1d` | [3] + [1] | 广播到 [3] |
| `test_add_broadcast_2d` | [3,1] + [1,4] | 广播到 [3,4] |
| `test_arithmetic_chain` | `(a + b) * c - d` | 正确链式运算 |
| `test_assign_arithmetic` | `a += b; a *= c;` | 就地修改正确 |
| `test_scalar_arithmetic_mixed` | `a + 1.0 + 2.0 * b` | 标量混合运算 |

### 7.3 边界测试（`tests/edge_cases.rs` 子集）

| 测试函数 | 边界条件 | 预期 |
|----------|----------|------|
| `test_zeros_single_element` | shape=[1] | len=1, 值=0 |
| `test_zeros_high_dimensional` | shape=[2,2,2,2,2] | 正确创建，5D |
| `test_empty_tensor_construction` | shape=[0,3] | 不 panic，len=0 |
| `test_full_bool_tensor` | `full([3], true)` | bool 张量填充 |
| `test_arange_large_step` | arange(0, 1000000, 1000000) | len=1 |
| `test_linspace_close_values` | linspace(0.0, 1e-15, 3) | 极小间隔 |
| `test_cast_extreme_values` | cast(f64::MAX → i32) | 饱和到 i32::MAX |
| `test_cast_subnormal` | cast(subnormal f64 → f32) | 正确处理 |
| `test_add_empty_tensors` | [] + [] | 结果为空 |
| `test_mul_by_zero_tensor` | ones * zeros | 全零 |

### 7.4 基准测试（`benches/construction.rs`）

| 基准函数 | 场景 | 测量内容 |
|----------|------|----------|
| `bench_zeros_100x100` | `zeros([100, 100])` | 分配 + 初始化时间 |
| `bench_eye_1000x1000` | `eye(1000, 1000)` | 分配 + 对角线填充 |
| `bench_arange_1m` | `arange(0.0, 1e6, 1.0)` | 100 万元素序列生成 |
| `bench_linspace_1m` | `linspace(0.0, 1.0, 1_000_000)` | 100 万元素等间隔生成 |
| `bench_from_vec_1m` | `from_vec(vec![0.0; 1_000_000])` | Vec→Tensor 转换 |
| `bench_cast_f64_to_f32_1m` | `tensor.cast::<f32>()` | 100 万元素类型转换 |
| `bench_add_100x100` | `a + b` (100×100) | 逐元素加法 |
| `bench_mul_100x100` | `a * b` (100×100) | 逐元素乘法 |

### 7.5 覆盖率目标

| 指标 | 目标 |
|------|------|
| 构造函数覆盖 | 13/13 函数均有单元测试 |
| 转换方法覆盖 | `cast`, `to_arc`, `From` impls 全覆盖 |
| 运算符覆盖 | 所有 11 个二元运算符 + 2 个一元运算符 + 11 个 Assign 变体 |
| Cast 类型对覆盖 | f64↔f32, f64↔i32, f64↔i64, i32↔i64, bool↔f64, bool↔i32 |
| 边界条件 | 空张量、单元素、NaN/Inf、subnormal、高维（≥4D） |
