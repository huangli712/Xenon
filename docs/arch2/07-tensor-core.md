# TensorBase 核心抽象模块设计

> 文档编号: 07 | 模块: `src/tensor.rs` | 阶段: Phase 2 W4
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, 需求说明书 §6

---

## 1. 模块定位

`tensor` 模块是 Xenon 的**核心抽象层**，定义了整个库的统一数据结构 `TensorBase<S, D>` 及其类型别名体系。所有上层模块（构造、运算、索引、形状操作等）均以 `TensorBase` 为操作对象。

**核心设计理念：**

- **双参数泛型正交设计**：`S`（存储模式）与 `D`（维度类型）完全独立，通过类型别名组合出具体类型
- **零开销抽象**：泛型单态化消除运行时分派，视图创建零分配
- **类型系统驱动的安全保证**：只读/可写/拥有通过类型系统在编译时区分，无需运行时检查
- **统一的操作接口**：所有存储模式共享同一套方法签名，差异通过 trait bound 表达

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 数据结构定义 | `TensorBase<S, D>` struct | 具体运算实现 |
| 类型别名 | `Tensor`, `TensorView`, `TensorViewMut`, `ArcTensor` 及维度便捷别名 | 构造函数（`zeros` 等） |
| 基础访问方法 | `shape()`, `strides()`, `as_ptr()` 等 | 逐元素迭代（由 `iter/` 模块提供） |
| 布局查询 | `is_f_contiguous()` 等 O(1) 查询 | 布局计算逻辑（由 `layout` 模块提供） |
| 类型转换 | `view()`, `into_owned()`, `into_dimension()` | 广播（由 `broadcast` 模块提供） |
| 格式化输出 | `Debug` / `Display` impl | — |

---

## 2. 文件位置

```
src/tensor.rs              # 主文件，所有 TensorBase 定义
src/lib.rs                 # re-export 公共类型
```

单文件设计理由：`TensorBase` 是高度内聚的类型定义，方法虽多但均为同一类型的访问器/转换器，拆分到多文件反而增加认知负担。文件预计 ~1500 行，处于合理范围。

---

## 3. 依赖关系

```
tensor.rs
├── crate::dimension       # Dimension trait, Ix0~Ix6, IxDyn
├── crate::storage         # Storage trait, Owned, ViewRepr, ViewMutRepr, ArcRepr
├── crate::layout          # LayoutFlags, Order, 步长计算函数
├── crate::element         # Element trait
└── crate::error           # TensorError, Result
```

**依赖方向：单向向下**。`tensor` 不被 `dimension`/`storage`/`layout`/`element`/`error` 依赖，处于依赖图的上层。

### 依赖的具体类型

| 来源模块 | 使用的类型/_trait_ |
|----------|-------------------|
| `dimension` | `Dimension`, `Ix0`, `Ix1`, `Ix2`, `Ix3`, `Ix4`, `Ix5`, `Ix6`, `IxDyn` |
| `storage` | `Storage`, `RawStorage`, `Owned<A>`, `ViewRepr<&'a A>`, `ViewMutRepr<&'a mut A>`, `ArcRepr<A>` |
| `layout` | `LayoutFlags`, `Order`, `compute_strides()`, `compute_layout_flags()` |
| `element` | `Element` |
| `error` | `TensorError`, `Result` |

---

## 4. 公共 API 设计

### 4.1 TensorBase 结构体定义

```rust
/// The core N-dimensional array type.
///
/// `TensorBase` is parameterized by two orthogonal type parameters:
/// - `S`: Storage mode, determining ownership and access rights
/// - `D`: Dimension type, determining static or dynamic dimensionality
///
/// This type is never spelled out directly in user code; use the type aliases
/// `Tensor`, `TensorView`, `TensorViewMut`, or `ArcTensor` instead.
#[repr(C)]
pub struct TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Underlying data storage (owned, borrowed, or shared).
    storage: S,

    /// Shape of the tensor (length of each axis).
    shape: D,

    /// Strides in element units (signed, supports negative strides for reversed axes).
    strides: D,

    /// Data start offset in element units (enables zero-copy slice views).
    offset: usize,

    /// Cached layout properties for O(1) queries.
    layout_flags: LayoutFlags,
}
```

**`#[repr(C)]` 理由**：保证字段内存布局可预测，便于 FFI 调试和 `into_raw_parts()` / `from_raw_parts()` 的实现。

### 4.2 类型别名体系

#### 4.2.1 主类型别名

```rust
/// Owning N-dimensional array. Deep-clones on `.clone()`.
pub type Tensor<A, D> = TensorBase<Owned<A>, D>;

/// Immutable view into an N-dimensional array. O(1) clone (copies metadata only).
pub type TensorView<'a, A, D> = TensorBase<ViewRepr<&'a A>, D>;

/// Mutable view into an N-dimensional array. Not `Clone` (exclusive borrow).
pub type TensorViewMut<'a, A, D> = TensorBase<ViewMutRepr<&'a mut A>, D>;

/// Atomically reference-counted N-dimensional array with copy-on-write semantics.
pub type ArcTensor<A, D> = TensorBase<ArcRepr<A>, D>;
```

#### 4.2.2 维度便捷别名 — Owned

```rust
pub type Tensor0<A> = Tensor<A, Ix0>;      // 0-dimensional (scalar)
pub type Tensor1<A> = Tensor<A, Ix1>;      // 1-dimensional (vector)
pub type Tensor2<A> = Tensor<A, Ix2>;      // 2-dimensional (matrix)
pub type Tensor3<A> = Tensor<A, Ix3>;      // 3-dimensional
pub type Tensor4<A> = Tensor<A, Ix4>;      // 4-dimensional
pub type Tensor5<A> = Tensor<A, Ix5>;      // 5-dimensional
pub type Tensor6<A> = Tensor<A, Ix6>;      // 6-dimensional
pub type TensorD<A> = Tensor<A, IxDyn>;    // dynamic dimensionality
```

#### 4.2.3 维度便捷别名 — View

```rust
pub type TensorView0<'a, A> = TensorView<'a, A, Ix0>;
pub type TensorView1<'a, A> = TensorView<'a, A, Ix1>;
pub type TensorView2<'a, A> = TensorView<'a, A, Ix2>;
pub type TensorView3<'a, A> = TensorView<'a, A, Ix3>;
pub type TensorView4<'a, A> = TensorView<'a, A, Ix4>;
pub type TensorView5<'a, A> = TensorView<'a, A, Ix5>;
pub type TensorView6<'a, A> = TensorView<'a, A, Ix6>;
pub type TensorViewD<'a, A> = TensorView<'a, A, IxDyn>;
```

#### 4.2.4 维度便捷别名 — ViewMut

```rust
pub type TensorViewMut0<'a, A> = TensorViewMut<'a, A, Ix0>;
pub type TensorViewMut1<'a, A> = TensorViewMut<'a, A, Ix1>;
pub type TensorViewMut2<'a, A> = TensorViewMut<'a, A, Ix2>;
pub type TensorViewMut3<'a, A> = TensorViewMut<'a, A, Ix3>;
pub type TensorViewMut4<'a, A> = TensorViewMut<'a, A, Ix4>;
pub type TensorViewMut5<'a, A> = TensorViewMut<'a, A, Ix5>;
pub type TensorViewMut6<'a, A> = TensorViewMut<'a, A, Ix6>;
pub type TensorViewMutD<'a, A> = TensorViewMut<'a, A, IxDyn>;
```

#### 4.2.5 维度便捷别名 — Arc

```rust
pub type ArcTensor0<A> = ArcTensor<A, Ix0>;
pub type ArcTensor1<A> = ArcTensor<A, Ix1>;
pub type ArcTensor2<A> = ArcTensor<A, Ix2>;
pub type ArcTensor3<A> = ArcTensor<A, Ix3>;
pub type ArcTensor4<A> = ArcTensor<A, Ix4>;
pub type ArcTensor5<A> = ArcTensor<A, Ix5>;
pub type ArcTensor6<A> = ArcTensor<A, Ix6>;
pub type ArcTensorD<A> = ArcTensor<A, IxDyn>;
```

### 4.3 通用 `impl` 块 — 所有存储类型共享

以下方法定义在 `impl<S, D> TensorBase<S, D>` 上，其中 `S: RawStorage`（只读访问）和 `D: Dimension`：

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    // ── 形状与维度信息 ─────────────────────────────────────────

    /// Returns the shape as a slice of axis lengths.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.slice()
    }

    /// Returns the strides as a slice of signed element counts.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.strides.slice_signed()
    }

    /// Returns the strides as a slice of signed byte counts.
    #[inline]
    pub fn strides_bytes(&self) -> Cow<'_, [isize]> {
        // stride_bytes[i] = strides[i] * size_of::<A>()
        // Implementation note: uses Cow to avoid allocation when possible
        ...
    }

    /// Returns the data start offset in element units.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements.
    ///
    /// Computed as the product of all axis lengths. For a 0-dimensional tensor,
    /// returns 1.
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.size()
    }

    /// Returns `true` if the tensor contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of the given axis.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= self.ndim()`.
    #[inline]
    pub fn len_of(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    // ── 布局查询 ───────────────────────────────────────────────

    /// Returns `true` if the data is contiguous in F-order (column-major).
    #[inline]
    pub fn is_f_contiguous(&self) -> bool {
        self.layout_flags.is_f_contiguous()
    }

    /// Returns `true` if the data is contiguous in C-order (row-major).
    #[inline]
    pub fn is_c_contiguous(&self) -> bool {
        self.layout_flags.is_c_contiguous()
    }

    /// Returns `true` if the data is contiguous in either F-order or C-order.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.layout_flags.is_contiguous()
    }

    /// Returns `true` if the data pointer is aligned to 64 bytes (AVX-512).
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.layout_flags.is_aligned()
    }

    /// Returns `true` if any axis has stride 0 (broadcast dimension).
    #[inline]
    pub fn has_zero_stride(&self) -> bool {
        self.layout_flags.has_zero_stride()
    }

    /// Returns `true` if any axis has a negative stride (reversed dimension).
    #[inline]
    pub fn has_neg_stride(&self) -> bool {
        self.layout_flags.has_neg_stride()
    }

    /// Returns the cached layout flags.
    #[inline]
    pub fn layout_flags(&self) -> LayoutFlags {
        self.layout_flags
    }

    // ── 指针访问 ───────────────────────────────────────────────

    /// Returns a raw const pointer to the data start.
    ///
    /// The pointer points to `self.storage.as_ptr() + self.offset`.
    #[inline]
    pub fn as_ptr(&self) -> *const S::Elem {
        unsafe {
            self.storage.as_ptr().add(self.offset)
        }
    }

    /// Returns the element type size in bytes.
    #[inline]
    pub fn elem_size(&self) -> usize {
        core::mem::size_of::<S::Elem>()
    }
}
```

### 4.4 可变存储 `impl` 块

适用于 `S: Storage`（可读写）：

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns a raw mutable pointer to the data start.
    ///
    /// The pointer points to `self.storage.as_mut_ptr() + self.offset`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S::Elem {
        unsafe {
            self.storage.as_mut_ptr().add(self.offset)
        }
    }
}
```

### 4.5 Owned 专用 `impl` 块

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a `TensorBase` from an owned storage, shape, strides, and offset.
    ///
    /// Layout flags are computed from the provided shape and strides.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - All elements in the storage buffer are properly initialized.
    /// - The shape, strides, and offset describe a valid memory region
    ///   within the storage buffer.
    pub(crate) unsafe fn from_storage_unchecked(
        storage: Owned<A>,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self {
        let layout_flags = LayoutFlags::compute(&shape, &strides, storage.as_ptr(), offset);
        TensorBase {
            storage,
            shape,
            strides,
            offset,
            layout_flags,
        }
    }

    /// Consumes the tensor and returns an owned version.
    ///
    /// Since this is already an owned tensor, this is a no-op (identity).
    #[inline]
    pub fn into_owned(self) -> Tensor<A, D> {
        self
    }

    /// Returns an owned (deep) copy of the tensor.
    ///
    /// Always allocates a new buffer with default 64-byte alignment,
    /// regardless of the original alignment.
    pub fn to_owned(&self) -> Tensor<A, D>
    where
        A: Clone,
    {
        let mut new_storage = Owned::<A>::uninitialized(self.len());
        // Copy elements in logical order, producing F-contiguous output
        for (dst_idx, src_elem) in self.iter().enumerate() {
            unsafe {
                new_storage.as_mut_ptr().add(dst_idx).write(src_elem.clone());
            }
        }
        let new_strides = crate::layout::compute_strides(&self.shape, Order::F);
        unsafe {
            TensorBase::from_storage_unchecked(new_storage, self.shape.clone(), new_strides, 0)
        }
    }

    /// Returns an immutable view of the tensor.
    #[inline]
    pub fn view(&self) -> TensorView<'_, A, D> {
        TensorBase {
            storage: ViewRepr::new(unsafe { &*self.storage.as_ptr() }),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }

    /// Returns a mutable view of the tensor.
    #[inline]
    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D> {
        TensorBase {
            storage: ViewMutRepr::new(unsafe { &mut *self.storage.as_mut_ptr() }),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }

    /// Converts to a different dimension type.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::DimensionMismatch` if the target dimension type
    /// requires a different number of axes than the current shape.
    pub fn into_dimension<D2>(self) -> Result<Tensor<A, D2>>
    where
        D2: Dimension,
    {
        let new_shape = D2::from_dimension(self.shape)?;
        let new_strides = D2::from_dimension(self.strides)?;
        Ok(TensorBase {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout_flags: self.layout_flags,
        })
    }

    /// Reshapes the tensor to the given shape.
    ///
    /// The total number of elements must remain the same.
    /// The dimension type `D` is preserved.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidShape` if the total element count differs.
    /// Returns `TensorError::LayoutMismatch` if the data is not contiguous.
    pub fn reshape(&self, shape: D) -> Result<Tensor<A, D>> {
        if !self.is_contiguous() {
            return Err(TensorError::LayoutMismatch {
                reason: "reshape requires contiguous data",
            });
        }
        if shape.size() != self.shape.size() {
            return Err(TensorError::InvalidShape {
                expected: self.shape.size(),
                actual: shape.size(),
            });
        }
        let strides = crate::layout::compute_strides(&shape, Order::F);
        Ok(TensorBase {
            storage: self.storage.clone(),
            shape,
            strides,
            offset: self.offset,
            layout_flags: LayoutFlags::compute(
                &self.shape,
                &strides,
                self.storage.as_ptr(),
                self.offset,
            ),
        })
    }

    /// Reshapes the tensor with a different dimension type.
    ///
    /// # Errors
    ///
    /// Same as `reshape`, plus `DimensionMismatch` if the dimension type
    /// is incompatible.
    pub fn reshape_into<D2>(&self, shape: D2) -> Result<Tensor<A, D2>>
    where
        D2: Dimension,
    {
        if !self.is_contiguous() {
            return Err(TensorError::LayoutMismatch {
                reason: "reshape requires contiguous data",
            });
        }
        if shape.size() != self.shape.size() {
            return Err(TensorError::InvalidShape {
                expected: self.shape.size(),
                actual: shape.size(),
            });
        }
        let strides = crate::layout::compute_strides(&shape, Order::F);
        Ok(TensorBase {
            storage: self.storage.clone(),
            shape,
            strides,
            offset: self.offset,
            layout_flags: LayoutFlags::compute(
                &shape,
                &strides,
                self.storage.as_ptr(),
                self.offset,
            ),
        })
    }
}
```

### 4.6 View 专用 `impl` 块

```rust
impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a view from raw pointer components.
    ///
    /// # Safety
    ///
    /// See `from_raw_parts` in the FFI section (§14.5 of the spec).
    /// The caller must ensure:
    /// - `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`.
    /// - The memory range covers all accessible elements given `shape`, `strides`, `offset`.
    /// - The memory remains valid for lifetime `'a`.
    /// - No mutable references to the same memory exist during `'a`.
    /// - `shape` and `strides` have the same length.
    /// - All indexable offsets fall within valid memory.
    /// - All accessible elements are properly initialized.
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self {
        let storage = ViewRepr::new(&*ptr);
        let layout_flags = LayoutFlags::compute(&shape, &strides, ptr, offset);
        TensorBase {
            storage,
            shape,
            strides,
            offset,
            layout_flags,
        }
    }

    /// Returns an owned (deep) copy of the viewed data.
    pub fn to_owned(&self) -> Tensor<A, D>
    where
        A: Clone,
    {
        let mut new_storage = Owned::<A>::uninitialized(self.len());
        for (dst_idx, src_elem) in self.iter().enumerate() {
            unsafe {
                new_storage.as_mut_ptr().add(dst_idx).write(src_elem.clone());
            }
        }
        let new_strides = crate::layout::compute_strides(&self.shape, Order::F);
        unsafe {
            TensorBase::from_storage_unchecked(new_storage, self.shape.clone(), new_strides, 0)
        }
    }

    /// Consumes the view and returns an owned copy.
    ///
    /// Equivalent to `self.to_owned()` but moves `self`.
    pub fn into_owned(self) -> Tensor<A, D>
    where
        A: Clone,
    {
        self.to_owned()
    }

    /// Returns a new immutable view (re-borrows).
    #[inline]
    pub fn view(&self) -> TensorView<'a, A, D> {
        TensorBase {
            storage: self.storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }

    /// Converts to a different dimension type.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::DimensionMismatch` if dimension counts don't match.
    pub fn into_dimension<D2>(self) -> Result<TensorView<'a, A, D2>>
    where
        D2: Dimension,
    {
        let new_shape = D2::from_dimension(self.shape)?;
        let new_strides = D2::from_dimension(self.strides)?;
        Ok(TensorBase {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout_flags: self.layout_flags,
        })
    }

    /// Reshapes the view to the given shape (requires contiguous data).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::LayoutMismatch` if not contiguous.
    /// Returns `TensorError::InvalidShape` if element count differs.
    pub fn reshape(&self, shape: D) -> Result<TensorView<'a, A, D>> {
        if !self.is_contiguous() {
            return Err(TensorError::LayoutMismatch {
                reason: "reshape requires contiguous data",
            });
        }
        if shape.size() != self.shape.size() {
            return Err(TensorError::InvalidShape {
                expected: self.shape.size(),
                actual: shape.size(),
            });
        }
        let strides = crate::layout::compute_strides(&shape, Order::F);
        Ok(TensorBase {
            storage: self.storage,
            shape,
            strides,
            offset: self.offset,
            layout_flags: LayoutFlags::compute(
                &shape,
                &strides,
                self.as_ptr(),
                self.offset,
            ),
        })
    }
}
```

### 4.7 ViewMut 专用 `impl` 块

```rust
impl<'a, A, D> TensorBase<ViewMutRepr<&'a mut A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a mutable view from raw pointer components.
    ///
    /// # Safety
    ///
    /// Same requirements as `ViewRepr::from_raw_parts`, plus:
    /// - No other references (mutable or immutable) to the same memory
    ///   exist during `'a` (exclusive access).
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self {
        let storage = ViewMutRepr::new(&mut *ptr);
        let layout_flags = LayoutFlags::compute(&shape, &strides, ptr, offset);
        TensorBase {
            storage,
            shape,
            strides,
            offset,
            layout_flags,
        }
    }

    /// Returns an owned (deep) copy of the viewed data.
    pub fn to_owned(&self) -> Tensor<A, D>
    where
        A: Clone,
    {
        let mut new_storage = Owned::<A>::uninitialized(self.len());
        for (dst_idx, src_elem) in self.iter().enumerate() {
            unsafe {
                new_storage.as_mut_ptr().add(dst_idx).write(src_elem.clone());
            }
        }
        let new_strides = crate::layout::compute_strides(&self.shape, Order::F);
        unsafe {
            TensorBase::from_storage_unchecked(new_storage, self.shape.clone(), new_strides, 0)
        }
    }

    /// Consumes the mutable view and returns an owned copy.
    pub fn into_owned(self) -> Tensor<A, D>
    where
        A: Clone,
    {
        self.to_owned()
    }

    /// Returns an immutable view sharing the same data.
    #[inline]
    pub fn view(&self) -> TensorView<'_, A, D> {
        TensorBase {
            storage: ViewRepr::new(unsafe { &*self.storage.as_ptr() }),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }

    /// Re-borrows as a mutable view.
    #[inline]
    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D> {
        TensorBase {
            storage: ViewMutRepr::new(unsafe { &mut *self.storage.as_mut_ptr() }),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }

    /// Converts to a different dimension type.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::DimensionMismatch` if dimension counts don't match.
    pub fn into_dimension<D2>(self) -> Result<TensorViewMut<'a, A, D2>>
    where
        D2: Dimension,
    {
        let new_shape = D2::from_dimension(self.shape)?;
        let new_strides = D2::from_dimension(self.strides)?;
        Ok(TensorBase {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout_flags: self.layout_flags,
        })
    }

    /// Reshapes the mutable view (requires contiguous data).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::LayoutMismatch` if not contiguous.
    /// Returns `TensorError::InvalidShape` if element count differs.
    pub fn reshape(&mut self, shape: D) -> Result<TensorViewMut<'_, A, D>> {
        if !self.is_contiguous() {
            return Err(TensorError::LayoutMismatch {
                reason: "reshape requires contiguous data",
            });
        }
        if shape.size() != self.shape.size() {
            return Err(TensorError::InvalidShape {
                expected: self.shape.size(),
                actual: shape.size(),
            });
        }
        let strides = crate::layout::compute_strides(&shape, Order::F);
        Ok(TensorBase {
            storage: ViewMutRepr::new(unsafe { &mut *self.storage.as_mut_ptr() }),
            shape,
            strides,
            offset: self.offset,
            layout_flags: LayoutFlags::compute(
                &shape,
                &strides,
                self.as_ptr(),
                self.offset,
            ),
        })
    }
}
```

### 4.8 ArcRepr 专用 `impl` 块

```rust
impl<A, D> TensorBase<ArcRepr<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Returns an owned (deep) copy of the tensor data.
    ///
    /// If the `Arc` has a single owner (refcount == 1), this is equivalent
    /// to `Arc::try_unwrap` and avoids copying. Otherwise, performs a deep copy.
    pub fn into_owned(self) -> Tensor<A, D>
    where
        A: Clone,
    {
        match self.storage.try_unwrap() {
            Ok(owned) => TensorBase {
                storage: owned,
                shape: self.shape,
                strides: self.strides,
                offset: self.offset,
                layout_flags: self.layout_flags,
            },
            Err(arc_storage) => {
                // Fallback: deep copy
                let view: TensorView<'_, A, D> = TensorBase {
                    storage: ViewRepr::new(unsafe { &*arc_storage.as_ptr() }),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    offset: self.offset,
                    layout_flags: self.layout_flags,
                };
                view.to_owned()
            }
        }
    }

    /// Returns an owned (deep) copy of the tensor data.
    pub fn to_owned(&self) -> Tensor<A, D>
    where
        A: Clone,
    {
        let mut new_storage = Owned::<A>::uninitialized(self.len());
        for (dst_idx, src_elem) in self.iter().enumerate() {
            unsafe {
                new_storage.as_mut_ptr().add(dst_idx).write(src_elem.clone());
            }
        }
        let new_strides = crate::layout::compute_strides(&self.shape, Order::F);
        unsafe {
            TensorBase::from_storage_unchecked(new_storage, self.shape.clone(), new_strides, 0)
        }
    }

    /// Returns an immutable view of the tensor.
    #[inline]
    pub fn view(&self) -> TensorView<'_, A, D> {
        TensorBase {
            storage: ViewRepr::new(unsafe { &*self.storage.as_ptr() }),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }

    /// Obtains a mutable reference to the underlying data, performing
    /// copy-on-write if necessary.
    ///
    /// If the reference count is > 1, this deep-copies the data into a new
    /// allocation (64-byte aligned) and decrements the original Arc's refcount.
    /// Returns a mutable slice to the (possibly newly allocated) data.
    pub fn make_mut(&mut self) -> &mut [A]
    where
        A: Clone,
    {
        self.storage.make_mut()
    }

    /// Converts to a different dimension type.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::DimensionMismatch` if dimension counts don't match.
    pub fn into_dimension<D2>(self) -> Result<ArcTensor<A, D2>>
    where
        D2: Dimension,
    {
        let new_shape = D2::from_dimension(self.shape)?;
        let new_strides = D2::from_dimension(self.strides)?;
        Ok(TensorBase {
            storage: self.storage,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout_flags: self.layout_flags,
        })
    }

    /// Reshapes the tensor (requires contiguous data).
    ///
    /// # Errors
    ///
    /// Returns `TensorError::LayoutMismatch` if not contiguous.
    /// Returns `TensorError::InvalidShape` if element count differs.
    pub fn reshape(&self, shape: D) -> Result<ArcTensor<A, D>> {
        if !self.is_contiguous() {
            return Err(TensorError::LayoutMismatch {
                reason: "reshape requires contiguous data",
            });
        }
        if shape.size() != self.shape.size() {
            return Err(TensorError::InvalidShape {
                expected: self.shape.size(),
                actual: shape.size(),
            });
        }
        let strides = crate::layout::compute_strides(&shape, Order::F);
        Ok(TensorBase {
            storage: self.storage.clone(), // Arc clone (refcount + 1)
            shape,
            strides,
            offset: self.offset,
            layout_flags: LayoutFlags::compute(
                &shape,
                &strides,
                self.storage.as_ptr(),
                self.offset,
            ),
        })
    }
}
```

### 4.9 Clone 实现

```rust
// Owned: deep clone
impl<A, D> Clone for TensorBase<Owned<A>, D>
where
    A: Element + Clone,
    D: Dimension,
{
    fn clone(&self) -> Self {
        self.to_owned()
    }
}

// View: shallow clone (copies metadata only, re-borrows)
impl<'a, A, D> Clone for TensorBase<ViewRepr<&'a A>, D>
where
    A: Element,
    D: Dimension,
{
    fn clone(&self) -> Self {
        TensorBase {
            storage: self.storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }
}

// ViewMut: NOT Clone (exclusive borrow semantics)
// No Clone impl for TensorBase<ViewMutRepr<&'a mut A>, D>

// ArcTensor: shallow clone (refcount + 1)
impl<A, D> Clone for TensorBase<ArcRepr<A>, D>
where
    A: Element,
    D: Dimension,
{
    fn clone(&self) -> Self {
        TensorBase {
            storage: self.storage.clone(), // Arc::clone
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            layout_flags: self.layout_flags,
        }
    }
}
```

### 4.10 Copy 实现

```rust
// View is Copy (metadata-only, re-borrows)
impl<'a, A, D> Copy for TensorBase<ViewRepr<&'a A>, D>
where
    A: Element,
    D: Dimension + Copy,
{
}
```

**注意**：仅 `TensorView` 实现 `Copy`（元数据量小且固定）。`Tensor`（含堆分配）、`TensorViewMut`（独占语义）、`ArcTensor`（含 Arc 引用计数）均不实现 `Copy`。

### 4.11 Debug / Display 实现

```rust
/// Debug formatting: shows internal structure (storage type, shape, strides, offset, flags).
impl<S, D> core::fmt::Debug for TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorBase")
            .field("shape", &self.shape.slice())
            .field("strides", &self.strides.slice_signed())
            .field("offset", &self.offset)
            .field("layout_flags", &self.layout_flags)
            .finish_non_exhaustive()
    }
}

/// Display formatting: NumPy-style array output.
///
/// For large arrays (>1000 elements), shows summary with `...` ellipsis.
impl<S, D> core::fmt::Display for TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
    S::Elem: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const DISPLAY_MAX_ELEMENTS: usize = 1000;

        if self.len() == 0 {
            write!(f, "[]")?;
            return Ok(());
        }

        // For small arrays, show all elements in NumPy format
        // For large arrays, show first and last few elements with "..."
        if self.len() <= DISPLAY_MAX_ELEMENTS {
            // Full display: format as NumPy array
            format_numpy_full(self, f)?;
        } else {
            // Summary display: shape, dtype, and preview
            format_numpy_summary(self, f)?;
        }
        Ok(())
    }
}
```

**NumPy 格式化输出规则：**

| 场景 | 输出示例 |
|------|----------|
| 1D `[1, 2, 3]` | `[1, 2, 3]` |
| 2D `[[1, 2], [3, 4]]` | `[[1, 2],\n [3, 4]]` |
| 大数组 | `[[1, 2, ..., 99],\n [100, 101, ..., 199],\n ...,\n [9900, 9901, ..., 9999]]` |
| 空数组 | `[]` |
| 0D 标量 | `42` |

### 4.12 FFI 方法

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Converts a multi-dimensional index to a raw pointer.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn index_to_ptr(&self, index: &[usize]) -> *const S::Elem {
        self.check_index(index);
        unsafe { self.index_to_ptr_unchecked(index) }
    }

    /// Converts a multi-dimensional index to a raw pointer (unchecked).
    ///
    /// # Safety
    ///
    /// The caller must ensure the index is within bounds.
    #[inline]
    pub unsafe fn index_to_ptr_unchecked(&self, index: &[usize]) -> *const S::Elem {
        let offset = self.index_to_offset_unchecked(index);
        self.storage.as_ptr().add(offset)
    }

    /// Converts a multi-dimensional index to an element offset.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn index_to_offset(&self, index: &[usize]) -> usize {
        self.check_index(index);
        unsafe { self.index_to_offset_unchecked(index) }
    }

    /// Converts a multi-dimensional index to an element offset (unchecked).
    ///
    /// # Safety
    ///
    /// The caller must ensure the index is within bounds.
    #[inline]
    pub unsafe fn index_to_offset_unchecked(&self, index: &[usize]) -> usize {
        let mut offset = self.offset;
        for i in 0..self.ndim() {
            offset = offset.wrapping_add(
                (*index.get_unchecked(i) as isize * *self.strides.slice_signed().get_unchecked(i))
                    as usize,
            );
        }
        offset
    }

    /// Returns a pointer to element at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure the index is within bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: &[usize]) -> &S::Elem {
        &*self.index_to_ptr_unchecked(index)
    }
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns a mutable pointer to element at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure the index is within bounds.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: &[usize]) -> &mut S::Elem {
        let ptr = self.as_mut_ptr();
        let offset = self.index_to_offset_unchecked(index);
        &mut *ptr.add(offset - self.offset)
    }

    /// Returns a pointer to element at the given index, with bounds checking.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if any axis index is out of bounds.
    #[inline]
    pub fn get(&self, index: &[usize]) -> Result<&S::Elem> {
        self.check_index(index)?;
        // SAFETY: check_index verified bounds
        Ok(unsafe { self.get_unchecked(index) })
    }

    /// Returns a mutable reference to element at the given index, with bounds checking.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if any axis index is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: &[usize]) -> Result<&mut S::Elem> {
        self.check_index(index)?;
        // SAFETY: check_index verified bounds
        Ok(unsafe { self.get_unchecked_mut(index) })
    }
}
```

### 4.13 跨存储类型转换 `impl`

```rust
// Owned -> View (implicit borrowing via Deref)
impl<A, D> core::ops::Deref for TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    type Target = TensorBase<ViewRepr<'static, A>, D>;
    // Note: actual lifetime is tied to &self, not 'static
    // This is a common pattern; the lifetime is shortened by the borrow
    ...
}

// ArcTensor -> View (implicit borrowing via Deref)
impl<A, D> core::ops::Deref for TensorBase<ArcRepr<A>, D>
where
    A: Element,
    D: Dimension,
{
    type Target = TensorBase<ViewRepr<'static, A>, D>;
    ...
}
```

**设计决策：** `Owned` 和 `ArcRepr` 通过 `Deref` 到 `TensorView`，使得所有接受 `&TensorView` 的方法也能直接用于 `&Tensor` 和 `&ArcTensor`，无需显式调用 `.view()`。但 `DerefMut` **不实现**于 `ArcRepr`（写操作须通过 `make_mut()` 显式触发 Cow）。

### 4.14 PartialEq 实现

```rust
/// Element-wise equality comparison.
///
/// Two tensors are equal if they have the same shape and all corresponding
/// elements are equal. Uses `PartialEq` on the element type, so NaN != NaN.
impl<S1, S2, A, D> PartialEq<TensorBase<S2, D>> for TensorBase<S1, D>
where
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
    A: Element + PartialEq,
    D: Dimension,
{
    fn eq(&self, other: &TensorBase<S2, D>) -> bool {
        if self.shape != other.shape {
            return false;
        }
        // Element-wise comparison
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
```

### 4.15 Send / Sync 实现

Send/Sync 由 Rust 编译器通过自动推导实现，无需手动 `unsafe impl`。推导规则如下：

| 存储类型 | Send 条件 | Sync 条件 |
|----------|-----------|-----------|
| `Owned<A>` | `A: Send` | `A: Sync` |
| `ViewRepr<&'a A>` | `A: Sync` | `A: Sync` |
| `ViewMutRepr<&'a mut A>` | `A: Send` | 永远不是 Sync |
| `ArcRepr<A>` | `A: Send + Sync` | `A: Send + Sync` |

这些推导通过 `Owned<A>: Send iff A: Send`、`&'a A: Send iff A: Sync` 等 Rust 标准规则自动满足，无需显式代码。

---

## 5. 内部实现设计

### 5.1 数据访问机制

`TensorBase` 通过 `RawStorage` 和 `Storage` trait 访问底层数据，而非直接持有缓冲区：

```rust
// Read-only storage access (all TensorBase types)
pub trait RawStorage {
    type Elem: Element;

    /// Returns a raw const pointer to the start of the buffer (before offset).
    fn as_ptr(&self) -> *const Self::Elem;
}

// Read-write storage access (Owned, ViewMut, Arc only)
pub trait Storage: RawStorage {
    /// Returns a raw mutable pointer to the start of the buffer.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;
}
```

**关键设计：**

- `TensorBase.as_ptr()` 返回 `storage.as_ptr() + offset`，即实际数据起始位置
- 元素寻址公式：`base_ptr + offset + Σ(index[i] * strides[i])`
- 步长为 `isize`，支持负值（反转轴）和零值（广播轴）
- 所有指针运算在 `unsafe` 块中完成，安全封装在 checked 方法之后

### 5.2 LayoutFlags 管理策略

```rust
// LayoutFlags is a bitfield stored as u8
bitflags::bitflags! {
    pub struct LayoutFlags: u8 {
        const F_CONTIGUOUS   = 0b0000_0001;
        const C_CONTIGUOUS   = 0b0000_0010;
        const ALIGNED        = 0b0000_0100;
        const HAS_ZERO_STRIDE = 0b0000_1000;
        const HAS_NEG_STRIDE = 0b0001_0000;
    }
}
```

**标志计算时机：**

| 操作 | 标志处理 |
|------|----------|
| `TensorBase::from_storage_unchecked()` | 调用 `LayoutFlags::compute()` 计算全部标志 |
| `view()` / `view_mut()` | 继承源数组标志（视图共享同一内存布局） |
| `slice()` | 继承后按需降级（偏移可能破坏对齐） |
| `reshape()` | 重新计算全部标志（shape 和 strides 改变） |
| `transpose()` | 交换 F/C 标志，保留其他 |
| `broadcast()` | 设置 HAS_ZERO_STRIDE，重新计算连续性和对齐 |

**`LayoutFlags::compute()` 伪代码：**

```rust
impl LayoutFlags {
    pub fn compute<D: Dimension>(
        shape: &D,
        strides: &D,
        base_ptr: *const u8,  // raw pointer for alignment check
        offset: usize,
    ) -> Self {
        let mut flags = LayoutFlags::empty();
        let ndim = shape.ndim();
        let shape_slice = shape.slice();
        let stride_slice = strides.slice_signed();

        // Check F-contiguous: strides[0] == 1, strides[i] == strides[i-1] * shape[i-1]
        let is_f = check_f_contiguous(shape_slice, stride_slice);
        let is_c = check_c_contiguous(shape_slice, stride_slice);

        if is_f { flags |= LayoutFlags::F_CONTIGUOUS; }
        if is_c { flags |= LayoutFlags::C_CONTIGUOUS; }

        // Check alignment: (base_ptr + offset * elem_size) % 64 == 0
        let data_ptr = unsafe { base_ptr.add(offset) }; // conceptual
        if is_aligned_to(data_ptr, 64) {
            flags |= LayoutFlags::ALIGNED;
        }

        // Check zero/negative strides
        if stride_slice.iter().any(|&s| s == 0) {
            flags |= LayoutFlags::HAS_ZERO_STRIDE;
        }
        if stride_slice.iter().any(|&s| s < 0) {
            flags |= LayoutFlags::HAS_NEG_STRIDE;
        }

        flags
    }
}
```

### 5.3 内存安全保证

**借用检查器集成：**

TensorBase 的四种存储模式精确映射到 Rust 的所有权模型：

| 存储模式 | Rust 等价物 | 借用检查 |
|----------|-------------|----------|
| `Owned<A>` | `Vec<A>` | 拥有，可任意读写 |
| `ViewRepr<&'a A>` | `&'a [A]` | 共享借用，只读 |
| `ViewMutRepr<&'a mut A>` | `&'a mut [A]` | 独占借用，可读写 |
| `ArcRepr<A>` | `Arc<Vec<A>>` | 共享所有权，Cow 写 |

**安全封装模式：**

```rust
// Public safe API
pub fn get(&self, index: &[usize]) -> Result<&A> {
    self.check_index(index)?;          // Runtime bounds check
    // SAFETY: check_index guarantees the index is valid
    Ok(unsafe { self.get_unchecked(index) })
}

// Public unsafe API (for hot paths)
/// # Safety
///
/// The caller must ensure all index values are within bounds
/// for their respective axes.
pub unsafe fn get_unchecked(&self, index: &[usize]) -> &A {
    let offset = self.index_to_offset_unchecked(index);
    &*self.storage.as_ptr().add(offset)
}
```

**视图生命周期管理：**

```rust
// View's lifetime is tied to the source tensor's borrow
impl<A, D> Tensor<A, D> {
    pub fn view(&self) -> TensorView<'_, A, D> {
        // 'a is bound to &self's lifetime
        // Borrow checker ensures the source outlives the view
        TensorBase {
            storage: ViewRepr::new(unsafe { &*self.storage.as_ptr() }),
            ...
        }
    }

    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D> {
        // Exclusive borrow: no other references can exist
        TensorBase {
            storage: ViewMutRepr::new(unsafe { &mut *self.storage.as_mut_ptr() }),
            ...
        }
    }
}
```

### 5.4 视图创建的零开销保证

视图创建仅复制 5 个字段（`storage` 元数据、`shape`、`strides`、`offset`、`layout_flags`），不分配任何堆内存：

- `TensorView` 创建：复制 `*const A` 指针 + shape/strides/offset/flags → O(1)
- `TensorViewMut` 创建：复制 `*mut A` 指针 + shape/strides/offset/flags → O(1)
- `TensorView::clone()`：复制上述元数据 → O(1)
- `TensorView` 支持 `Copy`（当 `D: Copy` 时）

### 5.5 索引校验

```rust
impl<S, D> TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// Checks that the given index is within bounds for each axis.
    #[inline]
    fn check_index(&self, index: &[usize]) -> Result<()> {
        if index.len() != self.ndim() {
            return Err(TensorError::IndexOutOfBounds {
                axis: usize::MAX, // sentinel: wrong number of indices
                index: index.len(),
                size: self.ndim(),
            });
        }
        for (axis, (&idx, &size)) in index.iter().zip(self.shape.slice()).enumerate() {
            if idx >= size {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index: idx,
                    size,
                });
            }
        }
        Ok(())
    }
}
```

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 基础结构

- [ ] **T1: TensorBase struct 定义 + 类型别名**
  - 文件: `src/tensor.rs:1-60`
  - 内容: struct 定义、所有 type alias、mod 导入
  - 测试: 编译通过 + `size_of::<Tensor2<f64>>()` 预期值
  - 前置: dimension, storage, layout, error 模块完成
  - 预计: 10 min

- [ ] **T2: 通用访问器方法（shape/strides/offset/ndim/len/is_empty）**
  - 文件: `src/tensor.rs`
  - 内容: `shape()`, `strides()`, `strides_bytes()`, `offset()`, `ndim()`, `len()`, `is_empty()`, `len_of()`
  - 测试: 构造 tensor 后验证各访问器返回正确值
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: 布局查询方法**
  - 文件: `src/tensor.rs`
  - 内容: `is_f_contiguous()`, `is_c_contiguous()`, `is_contiguous()`, `is_aligned()`, `has_zero_stride()`, `has_neg_stride()`, `layout_flags()`
  - 测试: 验证不同布局下标志正确性（F-contig, C-contig, non-contig）
  - 前置: T2
  - 预计: 10 min

### 6.2 指针与索引

- [ ] **T4: 指针访问方法（as_ptr / as_mut_ptr）**
  - 文件: `src/tensor.rs`
  - 内容: `as_ptr()` (RawStorage bound), `as_mut_ptr()` (Storage bound), `elem_size()`
  - 测试: 验证指针指向正确地址，offset 正确应用
  - 前置: T2
  - 预计: 10 min

- [ ] **T5: 索引校验与元素访问（check_index, get, get_unchecked）**
  - 文件: `src/tensor.rs`
  - 内容: `check_index()`, `get()`, `get_mut()`, `get_unchecked()`, `get_unchecked_mut()`
  - 测试: 有效索引返回正确元素，越界索引返回错误
  - 前置: T4
  - 预计: 10 min

- [ ] **T6: FFI 辅助方法（index_to_ptr, index_to_offset, from_raw_parts）**
  - 文件: `src/tensor.rs`
  - 内容: `index_to_ptr()`, `index_to_ptr_unchecked()`, `index_to_offset()`, `index_to_offset_unchecked()`, `from_raw_parts()`, `from_raw_parts_mut()`
  - 测试: 从已知内存构造视图并验证索引计算
  - 前置: T5
  - 预计: 10 min

### 6.3 存储类型专有方法

- [ ] **T7: Owned 专有方法（into_owned, to_owned, view, view_mut）**
  - 文件: `src/tensor.rs`
  - 内容: `Owned` 的 `into_owned()`, `to_owned()`, `view()`, `view_mut()`, `from_storage_unchecked()`
  - 测试: 验证 Owned → View 转换，clone 深拷贝语义
  - 前置: T4
  - 预计: 10 min

- [ ] **T8: View 专有方法（to_owned, into_owned, into_dimension）**
  - 文件: `src/tensor.rs`
  - 内容: `ViewRepr` 的 `to_owned()`, `into_owned()`, `view()`, `into_dimension()`
  - 测试: 验证 View → Owned 深拷贝，维度转换成功/失败
  - 前置: T7
  - 预计: 10 min

- [ ] **T9: ViewMut 专有方法（to_owned, view, view_mut, into_dimension）**
  - 文件: `src/tensor.rs`
  - 内容: `ViewMutRepr` 的 `to_owned()`, `into_owned()`, `view()`, `view_mut()`, `into_dimension()`
  - 测试: 验证独占语义，ViewMut → View 降级，ViewMut → Owned 拷贝
  - 前置: T7
  - 预计: 10 min

- [ ] **T10: ArcRepr 专有方法（into_owned, to_owned, make_mut, view）**
  - 文件: `src/tensor.rs`
  - 内容: `ArcRepr` 的 `into_owned()`, `to_owned()`, `make_mut()`, `view()`, `into_dimension()`
  - 测试: 验证 Arc refcount 行为、Cow 语义（make_mut 触发深拷贝）
  - 前置: T7
  - 预计: 10 min

### 6.4 类型转换

- [ ] **T11: 维度转换与 reshape**
  - 文件: `src/tensor.rs`
  - 内容: 所有存储类型的 `into_dimension::<D2>()` 和 `reshape()`
  - 测试: Ix2 → IxDyn 成功，IxDyn(ndim=3) → Ix2 失败，reshape 元素数校验
  - 前置: T8, T9, T10
  - 预计: 10 min

- [ ] **T12: Clone / Copy 实现**
  - 文件: `src/tensor.rs`
  - 内容: `Clone` for Owned/View/ArcRepr, `Copy` for View
  - 测试: 验证深拷贝 vs 浅拷贝语义，Copy 编译通过
  - 前置: T7
  - 预计: 10 min

### 6.5 格式化与 trait 实现

- [ ] **T13: Debug / Display 实现**
  - 文件: `src/tensor.rs`
  - 内容: `Debug`（内部结构）、`Display`（NumPy 格式化）
  - 测试: 验证格式化输出符合预期（1D/2D/大数组/空数组/0D 标量）
  - 前置: T5
  - 预计: 10 min

- [ ] **T14: PartialEq 实现**
  - 文件: `src/tensor.rs`
  - 内容: 跨存储类型的 `PartialEq` 实现
  - 测试: 相同形状相等、不同形状不等、NaN 不等于自身
  - 前置: T5
  - 预计: 10 min

- [ ] **T15: Deref 实现（Owned → View, Arc → View）**
  - 文件: `src/tensor.rs`
  - 内容: `Deref` for Owned/ArcRepr 到 TensorView
  - 测试: 通过 `&tensor` 自动获取 TensorView 方法
  - 前置: T7, T10
  - 预计: 10 min

### 6.6 lib.rs 集成

- [ ] **T16: lib.rs re-export 与模块声明**
  - 文件: `src/lib.rs`
  - 内容: `pub mod tensor;` 和 re-export 所有公共类型别名
  - 测试: 外部 `use xenon::Tensor;` 编译通过
  - 前置: T1-T15
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/tensor.rs` 中的 `#[cfg(test)] mod tests`：

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **构造** | `test_from_storage_unchecked` | shape/strides/offset/flags 正确 |
| **访问器** | `test_shape_strides_offset` | 返回值与构造参数一致 |
| **布局** | `test_f_contiguous_flags` | F-order 创建的 Tensor 为 F-contiguous |
| | `test_c_contiguous_flags` | C-order 创建的 Tensor 为 C-contiguous |
| | `test_both_contiguous_1d` | 1D Tensor 同时为 F 和 C contiguous |
| | `test_non_contiguous_after_slice` | 非连续切片后标志正确 |
| **索引** | `test_get_valid_index` | 返回正确元素值 |
| | `test_get_out_of_bounds` | 返回 `IndexOutOfBounds` 错误 |
| | `test_get_wrong_ndim` | 索引维度数不匹配时返回错误 |
| **指针** | `test_as_ptr_with_offset` | 指针 = base + offset * elem_size |
| | `test_index_to_offset` | offset = base_offset + Σ(idx * stride) |
| **转换** | `test_view_from_owned` | 共享同一数据（指针相同） |
| | `test_view_mut_from_owned` | 独占可变访问 |
| | `test_to_owned_copies_data` | 新指针 != 原指针，数据相同 |
| | `test_into_dimension_ix2_to_ixdyn` | 静态 → 动态成功 |
| | `test_into_dimension_ixdyn_to_ix2_fail` | 维度数不匹配时返回错误 |
| **reshape** | `test_reshape_valid` | 形状改变，元素数不变 |
| | `test_reshape_wrong_count` | 元素数不同时返回错误 |
| | `test_reshape_non_contiguous` | 非连续数据返回 LayoutMismatch |
| **Clone** | `test_owned_clone_deep` | clone 后修改不影响原数据 |
| | `test_view_clone_shallow` | clone 后指针相同 |
| | `test_view_copy` | Copy trait 编译通过 |
| | `test_arc_clone_refcount` | clone 后 Arc refcount == 2 |
| **Arc** | `test_make_mut_cow` | refcount > 1 时 make_mut 深拷贝 |
| | `test_make_mut_no_copy` | refcount == 1 时 make_mut 不分配 |
| | `test_arc_into_owned_no_copy` | refcount == 1 时 into_owned 零拷贝 |
| **比较** | `test_partial_eq_same` | 相同数据相等 |
| | `test_partial_eq_different_shape` | 不同形状不相等 |
| | `test_partial_eq_nan` | NaN != NaN |
| **格式化** | `test_debug_format` | 输出含 shape/strides/offset |
| | `test_display_1d` | `[1, 2, 3]` |
| | `test_display_2d` | `[[1, 2], [3, 4]]` |
| | `test_display_empty` | `[]` |
| | `test_display_scalar` | `42` |

### 7.2 边界测试

位于 `tests/edge_cases.rs` 或 `src/tensor.rs` 测试模块中：

| 边界场景 | 测试项 |
|----------|--------|
| **空张量** | `test_empty_tensor_len_zero`, `test_empty_tensor_iter_yields_nothing`, `test_empty_reshape` |
| **单元素** | `test_scalar_tensor_len_one`, `test_scalar_ndim_zero`, `test_scalar_both_contiguous` |
| **0D (Ix0)** | `test_ix0_shape_empty_slice`, `test_ix0_len_is_one`, `test_ix0_to_ixdyn_roundtrip` |
| **高维 (4D+)** | `test_4d_shape_strides`, `test_6d_index_access` |
| **负步长** | `test_neg_stride_flags`, `test_neg_stride_index_correct` |
| **零步长** | `test_zero_stride_flags`, `test_zero_stride_broadcast_view` |
| **非连续** | `test_transpose_not_contiguous`, `test_slice_not_contiguous` |
| **大 offset** | `test_large_offset_view_access` |

### 7.3 属性测试

位于 `tests/property/`：

| 不变量 | 测试 |
|--------|------|
| `view().to_owned() == original` | 对任意 Tensor，视图 → 拷贝 回到等价数据 |
| `reshape.identity` | `t.reshape(t.shape()) == t`（连续时） |
| `ndim consistency` | `tensor.ndim() == tensor.shape().len()` |
| `len consistency` | `tensor.len() == product(tensor.shape())` |
| `into_dimension roundtrip` | `IxN → IxDyn → IxN` 保持形状 |

### 7.4 测试覆盖率目标

| 模块 | 目标行覆盖率 |
|------|-------------|
| `tensor.rs` | ≥ 90%（核心模块，高标准） |
| 重点覆盖：索引校验、指针运算、布局标志、类型转换 | 100% |

---

## 附录 A: 方法速查表

### 所有存储类型共享（RawStorage bound）

| 方法 | 签名 | 复杂度 |
|------|------|--------|
| `shape` | `(&self) -> &[usize]` | O(1) |
| `strides` | `(&self) -> &[isize]` | O(1) |
| `strides_bytes` | `(&self) -> Cow<'_, [isize]>` | O(ndim) |
| `offset` | `(&self) -> usize` | O(1) |
| `ndim` | `(&self) -> usize` | O(1) |
| `len` | `(&self) -> usize` | O(1) |
| `is_empty` | `(&self) -> bool` | O(1) |
| `len_of` | `(&self, axis: usize) -> usize` | O(1) |
| `is_f_contiguous` | `(&self) -> bool` | O(1) |
| `is_c_contiguous` | `(&self) -> bool` | O(1) |
| `is_contiguous` | `(&self) -> bool` | O(1) |
| `is_aligned` | `(&self) -> bool` | O(1) |
| `has_zero_stride` | `(&self) -> bool` | O(1) |
| `has_neg_stride` | `(&self) -> bool` | O(1) |
| `layout_flags` | `(&self) -> LayoutFlags` | O(1) |
| `as_ptr` | `(&self) -> *const A` | O(1) |
| `elem_size` | `(&self) -> usize` | O(1) |
| `get` | `(&self, index: &[usize]) -> Result<&A>` | O(ndim) |
| `get_unchecked` | `unsafe (&self, index: &[usize]) -> &A` | O(ndim) |
| `index_to_ptr` | `(&self, index: &[usize]) -> *const A` | O(ndim) |
| `index_to_ptr_unchecked` | `unsafe (&self, index: &[usize]) -> *const A` | O(ndim) |
| `index_to_offset` | `(&self, index: &[usize]) -> usize` | O(ndim) |
| `index_to_offset_unchecked` | `unsafe (&self, index: &[usize]) -> usize` | O(ndim) |

### 可写存储（Storage bound）

| 方法 | 签名 | 复杂度 |
|------|------|--------|
| `as_mut_ptr` | `(&mut self) -> *mut A` | O(1) |
| `get_mut` | `(&mut self, index: &[usize]) -> Result<&mut A>` | O(ndim) |
| `get_unchecked_mut` | `unsafe (&mut self, index: &[usize]) -> &mut A` | O(ndim) |

### 各存储类型专有

| 方法 | Owned | View | ViewMut | Arc |
|------|-------|------|---------|-----|
| `into_owned` | ✅ (identity) | ✅ (clone) | ✅ (clone) | ✅ (try_unwrap or clone) |
| `to_owned` | ✅ | ✅ | ✅ | ✅ |
| `view` | ✅ | ✅ | ✅ | ✅ |
| `view_mut` | ✅ | — | ✅ | — |
| `into_dimension` | ✅ | ✅ | ✅ | ✅ |
| `reshape` | ✅ | ✅ | ✅ | ✅ |
| `make_mut` | — | — | — | ✅ |
| `from_raw_parts` | — | ✅ | — | — |
| `from_raw_parts_mut` | — | — | ✅ | — |

---

## 附录 B: 与 ndarray 对比

| 设计点 | ndarray | Xenon (本设计) |
|--------|---------|----------------|
| 核心类型 | `ArrayBase<S, D>` | `TensorBase<S, D>` |
| 默认布局 | C-order | F-order（科学计算导向） |
| 步长类型 | `isize` | `isize`（一致） |
| 布局标志 | 无（运行时计算） | `LayoutFlags` bitfield（O(1) 缓存查询） |
| Arc 支持 | 无 | `ArcRepr<A>` 内置 Cow |
| 对齐 | 无特殊保证 | 默认 64 字节对齐 + 标志缓存 |
| View Copy | `Copy` when `D: Copy` | `Copy` when `D: Copy`（一致） |
| Display | Debug only | NumPy 风格 Display |
