# FFI 集成模块设计

> 文档编号: 18 | 模块: `src/ffi.rs` | 阶段: Phase 3（API 模块，可并行）
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `05-storage.md`, `06-layout.md`, `07-tensor-core.md`, 需求说明书 §14

---

## 1. 模块定位

FFI 集成模块为 Renon 张量库提供与外部 C/BLAS 库互操作的能力。它位于依赖图的上层，依赖核心模块（tensor、storage、layout）但不被核心模块依赖。

**核心设计理念：**

- **零成本桥接**：所有 FFI 方法为内联访问器或纯计算函数，不引入额外分配或运行时开销
- **BLAS 优先**：列优先（F-order）为默认布局，提供完整的 BLAS 兼容性查询与参数计算
- **安全封装**：`unsafe` 操作集中在明确的函数边界内，每个 `unsafe` 函数携带完整的 `# Safety` 文档
- **所有权显式化**：`into_raw_parts` / `from_raw_parts` 明确标记所有权转移时机

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 指针访问 | `as_ptr()`, `as_mut_ptr()` 的 FFI 友好封装 | 底层存储指针管理（由 storage 模块提供） |
| 原始部件构造/解构 | `from_raw_parts`, `into_raw_parts` | 张量构造函数（由 construction 模块提供） |
| BLAS 兼容性查询 | `lda()`, `is_blas_compatible()`, `blas_layout()`, `blas_trans()` | 实际 BLAS 调用（由上层绑定库负责） |
| C 回调接口 | 类型别名与 trait bound | 具体 C 库绑定 |
| 索引转换 | `index_to_ptr()`, `index_to_offset()` | 高级索引操作（由 indexing 模块提供） |

**与 tensor 核心模块的关系：**

tensor 核心模块（`07-tensor-core.md`）已定义了 `as_ptr()`、`as_mut_ptr()`、`from_raw_parts()` 等基础方法。本模块**扩展**而非重复这些方法：

- 核心模块提供：基础指针访问（`as_ptr` 返回 `storage.as_ptr() + offset`）和原始部件构造
- FFI 模块提供：BLAS 语义层（`lda`、`blas_layout`、`blas_trans`）、所有权转移（`into_raw_parts`）、C 回调类型定义

---

## 2. 文件位置

```
src/ffi.rs              # FFI 集成模块主文件
src/lib.rs              # pub mod ffi; 声明 + re-export 公共类型
tests/ffi.rs            # 集成测试
examples/ffi_integration.rs  # FFI 使用示例
```

单文件设计理由：FFI 模块以扩展方法（extension trait）和独立函数为主，逻辑集中，预计 ~600 行。

---

## 3. 依赖关系

```
ffi.rs
├── crate::tensor       # TensorBase<S, D>, Tensor, TensorView, TensorViewMut, type aliases
├── crate::storage       # Storage, RawStorage, StorageMut, Owned, ViewRepr, ViewMutRepr
├── crate::layout        # LayoutFlags, Order, compute_strides, is_f_contiguous_impl
├── crate::dimension     # Dimension, Ix0~Ix6, IxDyn
├── crate::element       # Element
└── crate::error         # TensorError, Result
```

**依赖方向：单向向下。** FFI 模块仅消费核心模块的 API，不反向依赖。

### 依赖的具体类型

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `tensor` | `TensorBase`, `Tensor`, `TensorView`, `TensorViewMut`, `ArcTensor` |
| `storage` | `RawStorage`, `Storage`, `Owned`, `ViewRepr`, `ViewMutRepr`, `ArcRepr` |
| `layout` | `LayoutFlags`, `Order` |
| `dimension` | `Dimension` |
| `element` | `Element` |
| `error` | `TensorError` |

---

## 4. 公共 API 设计

### 4.1 `RawParts` 结构体

`into_raw_parts` 的返回类型，封装解构后的张量原始部件：

```rust
/// Decomposed raw components of a tensor for FFI transfer.
///
/// Returned by [`Tensor::into_raw_parts`]. The caller assumes ownership
/// of the underlying data buffer and is responsible for ensuring its
/// correct disposal (e.g., by reconstructing a `Tensor` via
/// [`Tensor::from_raw_parts_owned`]).
///
/// # Memory Safety
///
/// After calling `into_raw_parts`, the original tensor is consumed.
/// The caller must ensure:
/// - The `ptr` remains valid for as long as the data is accessed.
/// - No other code reads from or writes to the memory through Rust
///   references while the raw pointer is in use by foreign code.
/// - When done, the memory is either reconstructed into a `Tensor`
///   (via `from_raw_parts_owned`) or properly deallocated.
#[repr(C)]
pub struct RawParts<A> {
    /// Pointer to the data buffer start (before offset).
    /// This is the base allocation pointer, NOT the data start pointer.
    /// Data begins at `ptr.add(offset)`.
    pub ptr: *mut A,

    /// Number of elements in the underlying buffer.
    pub len: usize,

    /// Shape of the tensor (axis lengths).
    pub shape: Vec<usize>,

    /// Strides in element units (signed, supports negative strides).
    pub strides: Vec<isize>,

    /// Data start offset in element units from `ptr`.
    pub offset: usize,

    /// Alignment of the original allocation in bytes.
    pub alignment: usize,
}
```

> **设计决策**：`RawParts` 使用 `#[repr(C)]` 保证 C 兼容的内存布局，方便传递给 C 代码。`shape` 和 `strides` 使用 `Vec` 而非固定数组，以兼容 `IxDyn` 动态维度。`ptr` 是基础分配指针（而非数据起始指针），因为释放时需要原始分配地址。

### 4.2 `BlasLayout` 枚举

描述张量相对于 BLAS 期望的列优先布局的兼容性：

```rust
/// BLAS memory layout compatibility.
///
/// Describes how a tensor's memory layout maps to BLAS expectations.
/// BLAS routines expect column-major (F-order) data by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlasLayout {
    /// Data is F-contiguous (column-major) — pass directly to BLAS.
    /// No transposition needed.
    ColMajor = 0,

    /// Data is C-contiguous (row-major) — can be passed to BLAS with
    /// transposition flag (`Trans::T` or `Trans::C`).
    RowMajor = 1,

    /// Data is not contiguous in either order — cannot be passed to
    /// BLAS directly. Must copy to a contiguous buffer first.
    None = 2,
}
```

### 4.3 `BlasTrans` 枚举

BLAS `TRANS` 参数的 Rust 类型安全封装：

```rust
/// BLAS transposition parameter.
///
/// Maps to the `TRANS` character parameter in BLAS routines
/// ('N' = no transpose, 'T' = transpose, 'C' = conjugate transpose).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlasTrans {
    /// No transposition (`'N'` in BLAS).
    NoTrans = 0,

    /// Transpose (`'T'` in BLAS).
    Trans = 1,

    /// Conjugate transpose (`'C'` in BLAS).
    ConjTrans = 2,
}

impl BlasTrans {
    /// Returns the BLAS character code for this transposition flag.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(BlasTrans::NoTrans.as_char(), b'N');
    /// assert_eq!(BlasTrans::Trans.as_char(), b'T');
    /// assert_eq!(BlasTrans::ConjTrans.as_char(), b'C');
    /// ```
    #[inline]
    pub const fn as_char(self) -> u8 {
        match self {
            BlasTrans::NoTrans => b'N',
            BlasTrans::Trans => b'T',
            BlasTrans::ConjTrans => b'C',
        }
    }
}
```

### 4.4 C 兼容回调类型

为自定义操作提供 C 兼容的函数指针接口：

```rust
/// C-compatible callback for element-wise operations (read-only).
///
/// Receives a pointer to a contiguous element buffer and its length.
/// The callback must not modify the data.
///
/// # Safety
///
/// The callback must:
/// - Not write to the memory pointed to by `data`.
/// - Not retain the pointer after returning.
/// - Be safe to call from any thread if used with parallel operations.
pub type CCallbackRead = unsafe extern "C" fn(data: *const u8, len: usize, user_data: *mut core::ffi::c_void);

/// C-compatible callback for element-wise operations (read-write).
///
/// Receives a pointer to a contiguous element buffer and its length.
/// The callback may modify the data in place.
///
/// # Safety
///
/// The callback must:
/// - Not access memory beyond `data + len * elem_size`.
/// - Not retain the pointer after returning.
/// - Be safe to call from any thread if used with parallel operations.
pub type CCallbackWrite = unsafe extern "C" fn(data: *mut u8, len: usize, user_data: *mut core::ffi::c_void);

/// C-compatible callback for binary element-wise operations.
///
/// Receives pointers to two input buffers and one output buffer,
/// all with the same length in bytes.
///
/// # Safety
///
/// The callback must:
/// - Not access memory beyond the provided lengths.
/// - Not retain any pointer after returning.
/// - `out` must not alias `a` or `b`.
pub type CCallbackBinary = unsafe extern "C" fn(
    a: *const u8,
    b: *const u8,
    out: *mut u8,
    len: usize,
    user_data: *mut core::ffi::c_void,
);
```

### 4.5 FFI Extension Trait — 所有存储类型共享

为 `TensorBase<S, D>` 提供统一的 FFI 方法。核心模块已实现的 `as_ptr()` / `as_mut_ptr()` 不在此处重复；本 trait 提供 BLAS 语义层和所有权转移。

```rust
/// FFI extension methods for tensor types.
///
/// Provides BLAS compatibility queries, raw pointer decomposition,
/// and C-compatible callback invocation.
///
/// This trait is implemented for all `TensorBase<S, D>` types where
/// `S: RawStorage` and `D: Dimension`.
pub trait FfiExt<A, D>
where
    A: Element,
    D: Dimension,
{
    /// Returns the BLAS leading dimension for a 2D tensor.
    ///
    /// The leading dimension (lda/ldb/ldc) is the stride of the second
    /// dimension in F-order layout. For a column-major matrix with shape
    /// `[m, n]`, `lda = m` (i.e., `strides[1]`).
    ///
    /// # Panics
    ///
    /// Panics if `self.ndim() != 2`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 3x4 F-contiguous matrix: strides = [1, 3]
    /// let a: Tensor2<f64> = zeros([3, 4]);
    /// assert_eq!(a.lda(), 3);
    /// ```
    fn lda(&self) -> usize;

    /// Returns `true` if the tensor's memory layout is directly compatible
    /// with BLAS routines.
    ///
    /// A tensor is BLAS-compatible if:
    /// - It is F-contiguous (column-major) or C-contiguous (row-major).
    /// - All strides are positive (no reversed axes).
    /// - No axis has stride 0 (no broadcast dimensions).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a: Tensor2<f64> = zeros([3, 4]);  // F-contiguous
    /// assert!(a.is_blas_compatible());
    /// ```
    fn is_blas_compatible(&self) -> bool;

    /// Returns the BLAS layout classification of this tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a: Tensor2<f64> = zeros([3, 4]);  // F-contiguous
    /// assert_eq!(a.blas_layout(), BlasLayout::ColMajor);
    /// ```
    fn blas_layout(&self) -> BlasLayout;

    /// Returns the BLAS transposition flag for this tensor.
    ///
    /// When passing a C-contiguous (row-major) tensor to a BLAS routine
    /// that expects F-order data, use `Trans` to indicate the layout
    /// mismatch. The BLAS routine will interpret the data correctly.
    ///
    /// Returns `None` if the tensor is not BLAS-compatible.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // F-contiguous: no transposition needed
    /// let a: Tensor2<f64> = zeros([3, 4]);  // F-order
    /// assert_eq!(a.blas_trans(), Some(BlasTrans::NoTrans));
    ///
    /// // C-contiguous: pass with Trans flag
    /// let b = a.to_c_contiguous();           // C-order
    /// assert_eq!(b.blas_trans(), Some(BlasTrans::Trans));
    /// ```
    fn blas_trans(&self) -> Option<BlasTrans>;

    /// Returns `true` if the data is contiguous in Fortran (column-major) order.
    ///
    /// Equivalent to `self.layout_flags().is_f_contiguous()`.
    /// Provided as a convenience alias with Fortran naming for BLAS users.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let a: Tensor2<f64> = zeros([3, 4]);  // default F-order
    /// assert!(a.is_fortran_contiguous());
    /// ```
    fn is_fortran_contiguous(&self) -> bool;

    /// Returns the maximum byte offset from the data start pointer
    /// that can be accessed through this tensor.
    ///
    /// Useful for validating that an external buffer is large enough
    /// before constructing a view via `from_raw_parts`.
    ///
    /// # Computation
    ///
    /// Scans all axes and computes `max(index[i]) * abs(strides[i])`
    /// for each axis, then sums them. The result is in element units.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // shape [3, 4], strides [1, 3], offset 0
    /// // max offset = (3-1)*1 + (4-1)*3 = 2 + 9 = 11 (elements)
    /// let a: Tensor2<f64> = zeros([3, 4]);
    /// assert_eq!(a.max_offset(), 11);
    /// ```
    fn max_offset(&self) -> usize;

    /// Invokes a C-compatible read callback over contiguous data.
    ///
    /// If the tensor is contiguous, passes the raw pointer directly.
    /// If not contiguous, copies data to a temporary contiguous buffer
    /// first, then invokes the callback on the copy.
    ///
    /// # Arguments
    ///
    /// * `callback` - C function pointer to invoke.
    /// * `user_data` - Opaque pointer passed through to the callback.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `callback` is a valid function pointer.
    /// - `user_data` is a valid pointer (or null) for the callback's use.
    /// - The callback does not write to the data buffer.
    /// - The callback does not retain the pointer after returning.
    unsafe fn with_contiguous_read(
        &self,
        callback: CCallbackRead,
        user_data: *mut core::ffi::c_void,
    );

    /// Returns the data pointer as a `*const u8` byte pointer.
    ///
    /// Equivalent to `self.as_ptr() as *const u8`.
    /// Useful for passing to C APIs that operate on byte buffers.
    #[inline]
    fn as_byte_ptr(&self) -> *const u8;

    /// Returns the byte size of the stored element type.
    #[inline]
    fn element_size(&self) -> usize;
}

/// FFI extension methods for mutable tensor types.
///
/// Provides mutable variants of FFI operations for tensors with
/// write access (`S: Storage`).
pub trait FfiExtMut<A, D>: FfiExt<A, D>
where
    A: Element,
    D: Dimension,
{
    /// Returns the data pointer as a `*mut u8` byte pointer.
    ///
    /// Equivalent to `self.as_mut_ptr() as *mut u8`.
    #[inline]
    fn as_mut_byte_ptr(&mut self) -> *mut u8;

    /// Invokes a C-compatible write callback over contiguous data.
    ///
    /// If the tensor is contiguous, passes the raw pointer directly.
    /// If not contiguous, the data is copied to a temporary buffer,
    /// the callback writes to the buffer, and results are copied back.
    ///
    /// # Arguments
    ///
    /// * `callback` - C function pointer to invoke.
    /// * `user_data` - Opaque pointer passed through to the callback.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `callback` is a valid function pointer.
    /// - `user_data` is a valid pointer (or null) for the callback's use.
    /// - The callback does not access memory beyond `len * element_size` bytes.
    /// - The callback does not retain the pointer after returning.
    unsafe fn with_contiguous_write(
        &mut self,
        callback: CCallbackWrite,
        user_data: *mut core::ffi::c_void,
    );
}
```

### 4.6 Owned 专用 FFI 方法

以下方法仅适用于 `Tensor<A, D>`（拥有所有权），涉及所有权转移：

```rust
impl<A, D> Tensor<A, D>
where
    A: Element,
    D: Dimension,
{
    /// Consumes the tensor and returns its raw components for FFI transfer.
    ///
    /// This transfers ownership of the underlying buffer to the caller.
    /// The returned `RawParts` contains the base allocation pointer,
    /// buffer length, shape, strides, offset, and alignment.
    ///
    /// After calling this method, the caller is responsible for the
    /// memory's lifetime. To reclaim ownership, use [`from_raw_parts_owned`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t: Tensor2<f64> = zeros([3, 4]);
    /// let raw = t.into_raw_parts();
    /// // raw.ptr is now owned by the caller
    /// // ... pass to C code ...
    /// // Reclaim:
    /// let t2 = unsafe { Tensor2::from_raw_parts_owned(raw) };
    /// ```
    pub fn into_raw_parts(self) -> RawParts<A> {
        let shape: Vec<usize> = self.shape().to_vec();
        let strides: Vec<isize> = self.strides().to_vec();
        let ptr = self.storage.as_mut_ptr(); // base allocation ptr
        let len = self.storage.len();
        let offset = self.offset();
        let alignment = self.storage.alignment();
        core::mem::forget(self); // Prevent Drop from deallocating
        RawParts {
            ptr,
            len,
            shape,
            strides,
            offset,
            alignment,
        }
    }

    /// Reconstructs an owning tensor from raw components previously
    /// obtained via [`into_raw_parts`].
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `parts.ptr` is non-null, non-dangling, and aligned to `parts.alignment` bytes.
    /// - The memory range `[ptr, ptr + len)` is valid and was allocated with
    ///   `parts.alignment` byte alignment.
    /// - `parts.shape` and `parts.strides` have the same length.
    /// - For every valid index, the computed offset `offset + Σ(index[i] * strides[i])`
    ///   falls within `[0, len)`.
    /// - All accessible elements are properly initialized.
    /// - The memory is not simultaneously accessed through any other reference.
    /// - The caller has exclusive ownership of the memory (no other code
    ///   will attempt to free or modify it).
    ///
    /// # Panics
    ///
    /// Panics if `parts.ptr` is null.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t: Tensor2<f64> = zeros([3, 4]);
    /// let raw = t.into_raw_parts();
    /// // ... pass to C code and back ...
    /// let t2 = unsafe { Tensor2::from_raw_parts_owned(raw) };
    /// assert_eq!(t2.shape(), &[3, 4]);
    /// ```
    pub unsafe fn from_raw_parts_owned(parts: RawParts<A>) -> Self {
        assert!(!parts.ptr.is_null(), "null pointer in RawParts");
        let storage = Owned::from_raw_parts(parts.ptr, parts.len, parts.alignment);
        let shape = D::from_slice(&parts.shape);
        let strides = D::from_slice_signed(&parts.strides);
        let layout_flags = LayoutFlags::compute(
            &shape,
            &strides,
            parts.ptr as usize + parts.offset * core::mem::size_of::<A>(),
            core::mem::size_of::<A>(),
            parts.alignment,
        );
        TensorBase {
            storage,
            shape,
            strides,
            offset: parts.offset,
            layout_flags,
        }
    }
}
```

### 4.7 View 专用 FFI 方法

```rust
impl<'a, A, D> TensorView<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a read-only view from raw pointer components.
    ///
    /// This method is inherited from the tensor core module and documented
    /// here for FFI context. See [`TensorBase::from_raw_parts`] for the
    /// canonical definition.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` is non-null, non-dangling, and aligned to `align_of::<A>()`.
    /// - The memory range starting at `ptr` covers all accessible elements
    ///   given `shape`, `strides`, and `offset`.
    /// - The memory remains valid for lifetime `'a`.
    /// - No mutable references to the same memory exist during `'a`.
    /// - `shape` and `strides` have the same length.
    /// - For every valid index, the computed offset falls within valid memory.
    /// - All accessible elements are properly initialized.
    /// - If the view is used across threads, the underlying memory satisfies
    ///   `Send` / `Sync` requirements.
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self;

    /// Returns the raw pointer to the data start as a `*const T`.
    ///
    /// The returned pointer is `storage.as_ptr() + offset`.
    /// This is an alias for [`TensorBase::as_ptr`] provided for
    /// explicit FFI usage.
    #[inline]
    pub fn as_ptr(&self) -> *const A;
}

impl<'a, A, D> TensorViewMut<'a, A, D>
where
    A: Element,
    D: Dimension,
{
    /// Creates a mutable view from raw pointer components.
    ///
    /// # Safety
    ///
    /// Same requirements as `TensorView::from_raw_parts`, plus:
    /// - No other references (mutable or immutable) to the same memory
    ///   exist during lifetime `'a` (exclusive access guarantee).
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        shape: D,
        strides: D,
        offset: usize,
    ) -> Self;

    /// Returns the raw mutable pointer to the data start.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut A;
}
```

### 4.8 独立函数 — BLAS 参数计算

不绑定到特定张量实例的工具函数，方便在没有张量对象时进行布局计算：

```rust
/// Computes the BLAS leading dimension for a 2D column-major layout.
///
/// For an F-contiguous matrix with shape `[m, n]`, the leading
/// dimension is `m` (the first dimension's size). This equals
/// `strides[1]` in F-order layout.
///
/// # Arguments
///
/// * `shape` - The 2D tensor shape `[rows, cols]`.
/// * `strides` - The strides in element units.
///
/// # Panics
///
/// Panics if `shape.len() != 2` or `strides.len() != 2`.
///
/// # Examples
///
/// ```ignore
/// // shape [3, 4], F-order strides [1, 3]
/// assert_eq!(compute_lda(&[3, 4], &[1, 3]), 3);
///
/// // shape [3, 4], C-order strides [4, 1]
/// assert_eq!(compute_lda(&[3, 4], &[4, 1]), 4);
/// ```
#[inline]
pub fn compute_lda(shape: &[usize], strides: &[isize]) -> usize {
    assert!(shape.len() == 2 && strides.len() == 2, "lda requires 2D tensor");
    strides[1].max(strides[0]) as usize
}

/// Checks whether the given shape and strides are BLAS-compatible.
///
/// BLAS compatibility requires:
/// - F-contiguous or C-contiguous layout.
/// - All strides are strictly positive.
/// - No zero strides (no broadcast dimensions).
///
/// # Arguments
///
/// * `shape` - Tensor dimensions.
/// * `strides` - Strides in element units.
///
/// # Examples
///
/// ```ignore
/// assert!(is_blas_compatible(&[3, 4], &[1, 3]));   // F-contiguous
/// assert!(is_blas_compatible(&[3, 4], &[4, 1]));   // C-contiguous
/// assert!(!is_blas_compatible(&[3, 4], &[0, 1]));  // broadcast, not compatible
/// ```
pub fn is_blas_compatible(shape: &[usize], strides: &[isize]) -> bool {
    // All strides must be positive
    if strides.iter().any(|&s| s <= 0) {
        return false;
    }
    // Must be F-contiguous or C-contiguous
    crate::layout::is_f_contiguous_impl(shape, strides)
        || crate::layout::is_c_contiguous_impl(shape, strides)
}

/// Classifies the BLAS layout from shape and strides.
///
/// # Arguments
///
/// * `shape` - Tensor dimensions.
/// * `strides` - Strides in element units.
///
/// # Returns
///
/// - `BlasLayout::ColMajor` if F-contiguous.
/// - `BlasLayout::RowMajor` if C-contiguous (but not F-contiguous).
/// - `BlasLayout::None` if neither or if any stride is non-positive.
pub fn classify_blas_layout(shape: &[usize], strides: &[isize]) -> BlasLayout {
    if strides.iter().any(|&s| s <= 0) {
        return BlasLayout::None;
    }
    let is_f = crate::layout::is_f_contiguous_impl(shape, strides);
    let is_c = crate::layout::is_c_contiguous_impl(shape, strides);
    match (is_f, is_c) {
        (true, _) => BlasLayout::ColMajor,
        (false, true) => BlasLayout::RowMajor,
        (false, false) => BlasLayout::None,
    }
}

/// Computes the maximum element offset from the base pointer.
///
/// For each axis, the maximum index is `shape[i] - 1`. The maximum
/// offset is `offset + Σ((shape[i] - 1) * abs(strides[i]))`.
///
/// This does NOT include `offset` — it returns the maximum additional
/// offset relative to the data start pointer.
///
/// # Arguments
///
/// * `shape` - Tensor dimensions.
/// * `strides` - Strides in element units.
///
/// # Examples
///
/// ```ignore
/// // shape [3, 4], strides [1, 3]
/// // max offset = 2*1 + 3*3 = 11
/// assert_eq!(compute_max_offset(&[3, 4], &[1, 3]), 11);
/// ```
pub fn compute_max_offset(shape: &[usize], strides: &[isize]) -> usize {
    shape.iter().zip(strides.iter()).map(|(&dim, &stride)| {
        if dim == 0 {
            0
        } else {
            (dim - 1) * (stride.unsigned_abs())
        }
    }).sum()
}
```

---

## 5. 内部实现设计

### 5.1 `FfiExt` trait 实现

```rust
impl<S, A, D> FfiExt<A, D> for TensorBase<S, D>
where
    S: RawStorage<Elem = A>,
    A: Element,
    D: Dimension,
{
    fn lda(&self) -> usize {
        assert_eq!(self.ndim(), 2, "lda requires a 2D tensor");
        compute_lda(self.shape(), self.strides())
    }

    fn is_blas_compatible(&self) -> bool {
        is_blas_compatible(self.shape(), self.strides())
    }

    fn blas_layout(&self) -> BlasLayout {
        classify_blas_layout(self.shape(), self.strides())
    }

    fn blas_trans(&self) -> Option<BlasTrans> {
        if strides.iter().any(|&s| s <= 0) {
            return None;
        }
        let is_f = crate::layout::is_f_contiguous_impl(self.shape(), self.strides());
        let is_c = crate::layout::is_c_contiguous_impl(self.shape(), self.strides());
        match (is_f, is_c) {
            (true, _) => Some(BlasTrans::NoTrans),
            (false, true) => Some(BlasTrans::Trans),
            (false, false) => None,
        }
    }

    fn is_fortran_contiguous(&self) -> bool {
        self.layout_flags().is_f_contiguous()
    }

    fn max_offset(&self) -> usize {
        compute_max_offset(self.shape(), self.strides())
    }

    unsafe fn with_contiguous_read(
        &self,
        callback: CCallbackRead,
        user_data: *mut core::ffi::c_void,
    ) {
        let elem_size = core::mem::size_of::<A>();
        if self.is_contiguous() {
            // Fast path: pass raw pointer directly
            callback(self.as_ptr() as *const u8, self.len(), user_data);
        } else {
            // Slow path: copy to contiguous buffer, invoke, drop
            let mut buf: Vec<A> = Vec::with_capacity(self.len());
            for elem in self.iter() {
                buf.push(elem.clone());
            }
            callback(buf.as_ptr() as *const u8, buf.len(), user_data);
        }
    }

    #[inline]
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }

    #[inline]
    fn element_size(&self) -> usize {
        core::mem::size_of::<A>()
    }
}
```

### 5.2 `FfiExtMut` trait 实现

```rust
impl<S, A, D> FfiExtMut<A, D> for TensorBase<S, D>
where
    S: Storage<Elem = A>,
    A: Element,
    D: Dimension,
{
    #[inline]
    fn as_mut_byte_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr() as *mut u8
    }

    unsafe fn with_contiguous_write(
        &mut self,
        callback: CCallbackWrite,
        user_data: *mut core::ffi::c_void,
    ) {
        if self.is_contiguous() {
            // Fast path: pass mutable pointer directly
            callback(self.as_mut_ptr() as *mut u8, self.len(), user_data);
        } else {
            // Slow path: copy to contiguous buffer, invoke, copy back
            let mut buf: Vec<A> = self.iter().cloned().collect();
            callback(buf.as_mut_ptr() as *mut u8, buf.len(), user_data);
            // Copy back from contiguous buffer to strided layout
            for (i, elem) in buf.into_iter().enumerate() {
                // Write back element-by-element using strided addressing
                *self.get_unchecked_mut(&self.linear_to_index(i)) = elem;
            }
        }
    }
}
```

### 5.3 `into_raw_parts` 所有权转移语义

```
into_raw_parts() 调用流程:
│
├── 1. 提取 shape/strides 为 Vec（从 Dimension trait 的 slice() 复制）
├── 2. 获取 base_ptr = storage.as_mut_ptr()（基础分配指针，非 data start）
├── 3. 获取 len = storage.len()（缓冲区总元素数）
├── 4. 获取 offset = self.offset()
├── 5. 获取 alignment = storage.alignment()
├── 6. core::mem::forget(self)  ← 阻止 Drop 释放内存
└── 7. 返回 RawParts { ptr, len, shape, strides, offset, alignment }
```

**关键保证：**

| 保证 | 说明 |
|------|------|
| 无 double-free | `mem::forget` 阻止 Tensor Drop 执行 |
| ptr 是基础指针 | 指向分配起始位置（非 data start），释放时需要此地址 |
| shape/strides 独立 | 返回的 `Vec` 是深拷贝，不与原 Tensor 共享内存 |
| 生命周期解除 | 返回后 Rust 不再追踪这块内存，调用方全权负责 |

### 5.4 `from_raw_parts` Safety 不变量

```
from_raw_parts / from_raw_parts_mut 调用前必须满足:

┌─────────────────────────────────────────────────────────────┐
│ 1. 指针有效性                                                │
│    ptr ≠ null, ptr ≠ dangling                                │
│    ptr 对齐到 align_of::<A>()                                 │
│                                                              │
│ 2. 内存范围                                                  │
│    ptr 起始的缓冲区 ≥ max_offset + 1 个元素                   │
│    max_offset = Σ((shape[i]-1) * abs(strides[i]))            │
│                                                              │
│ 3. 生命周期                                                  │
│    from_raw_parts:      内存在 'a 期间有效                    │
│    from_raw_parts_owned: 调用方独占所有权                      │
│                                                              │
│ 4. 别名规则                                                  │
│    from_raw_parts:      可共享读取，不可写入                   │
│    from_raw_parts_mut:  独占访问，无其他引用                   │
│    from_raw_parts_owned: 独占所有权，无其他引用                 │
│                                                              │
│ 5. 布局一致性                                                │
│    shape.len() == strides.len()                               │
│    所有 strides 为合法的元素步长                               │
│                                                              │
│ 6. 边界安全                                                  │
│    对所有合法索引 i ∈ [0, shape[j]):
│      offset + Σ(i[j] * strides[j]) ∈ [0, buf_len)           │
│                                                              │
│ 7. 元素初始化                                                │
│    所有可访问元素已正确初始化                                   │
│                                                              │
│ 8. 线程安全                                                  │
│    跨线程使用时，底层内存满足 Send/Sync                         │
└─────────────────────────────────────────────────────────────┘
```

### 5.5 BLAS Leading Dimension 计算

BLAS 的 leading dimension（lda/ldb/ldc）是列优先布局中**相邻列首元素之间的距离**。

```
F-order 矩阵 shape=[m, n], strides=[1, m]:

内存布局: [col0_row0, col0_row1, ..., col0_rowM, col1_row0, col1_row1, ...]
          |<--- lda = m --->|       |<--- lda = m --->|

lda = strides[1] = m

BLAS 调用示例 (dgemm):
  dgemm('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                                              ^^^     ^^^     ^^^
```

**关键规则：**

| 布局 | shape `[m, n]` | strides | lda | trans |
|------|----------------|---------|-----|-------|
| F-contiguous | `[m, n]` | `[1, m]` | `m` | `'N'` |
| C-contiguous | `[m, n]` | `[n, 1]` | `n` | `'T'` |
| 非连续 | `[m, n]` | 其他 | N/A | N/A |

> **注意**：对于 C-contiguous 矩阵传递给 BLAS 时，实际上是将矩阵视为其转置来传递。BLAS 看到的是 shape `[n, m]`、lda=`n`，配合 `Trans='T'` 标志实现正确的运算语义。

### 5.6 列优先连续性检查

列优先（Fortran）连续性的判定算法（与 `layout` 模块的 `is_f_contiguous_impl` 一致）：

```
is_fortran_contiguous(shape, strides):
    expected = 1
    for i in 0..ndim:
        if shape[i] == 1: continue      // size-1 dim: stride irrelevant
        if shape[i] == 0: return true   // zero-size: trivially contiguous
        if strides[i] != expected: return false
        expected *= shape[i]
    return true
```

对于 2D 矩阵 `[m, n]`：
- F-contiguous: `strides = [1, m]` → `stride[0] == 1` ✓, `stride[1] == m` ✓
- C-contiguous: `strides = [n, 1]` → `stride[0] == 1`? 仅当 `n == 1`

### 5.7 C 回调调用的安全封装模式

```
with_contiguous_read(callback, user_data):

┌─ Fast Path (is_contiguous == true) ─────────────────┐
│ 1. as_ptr() → raw pointer                            │
│ 2. callback(ptr, len, user_data)                     │
│ 3. 完成，零拷贝                                       │
└──────────────────────────────────────────────────────┘

┌─ Slow Path (is_contiguous == false) ────────────────┐
│ 1. 分配临时 Vec<A> (len 个元素)                       │
│ 2. 按逻辑顺序拷贝元素到 Vec                            │
│ 3. callback(vec.as_ptr(), len, user_data)            │
│ 4. Vec Drop（自动释放临时缓冲区）                      │
│ 总开销: O(n) 分配 + O(n) 拷贝                         │
└──────────────────────────────────────────────────────┘
```

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### Wave 1: 类型定义

- [ ] **T1: BlasLayout + BlasTrans 枚举定义**
  - 文件: `src/ffi.rs`
  - 内容: `BlasLayout` 枚举（`ColMajor`, `RowMajor`, `None`）、`BlasTrans` 枚举（`NoTrans`, `Trans`, `ConjTrans`）及其 `as_char()` 方法
  - 测试: `test_blas_layout_variants`, `test_blas_trans_as_char`
  - 前置: 无
  - 预计: 8 min

- [ ] **T2: RawParts 结构体定义**
  - 文件: `src/ffi.rs`
  - 内容: `RawParts<A>` 结构体（`#[repr(C)]`，含 ptr/len/shape/strides/offset/alignment 字段）及完整 doc comments
  - 测试: `test_raw_parts_size_and_alignment`
  - 前置: 无
  - 预计: 8 min

- [ ] **T3: C 回调类型定义**
  - 文件: `src/ffi.rs`
  - 内容: `CCallbackRead`, `CCallbackWrite`, `CCallbackBinary` 类型别名
  - 测试: 编译通过（类型别名无运行时测试）
  - 前置: 无
  - 预计: 5 min

### Wave 2: 独立函数

- [ ] **T4: BLAS 参数计算函数**
  - 文件: `src/ffi.rs`
  - 内容: `compute_lda()`, `is_blas_compatible()`, `classify_blas_layout()`, `compute_max_offset()`
  - 测试: `test_compute_lda_f_order`, `test_compute_lda_c_order`, `test_is_blas_compatible`, `test_classify_blas_layout`, `test_compute_max_offset`
  - 前置: layout 模块（`is_f_contiguous_impl`, `is_c_contiguous_impl`）
  - 预计: 10 min

### Wave 3: Extension Trait 定义与实现

- [ ] **T5: FfiExt trait 定义**
  - 文件: `src/ffi.rs`
  - 内容: `FfiExt<A, D>` trait 定义（`lda`, `is_blas_compatible`, `blas_layout`, `blas_trans`, `is_fortran_contiguous`, `max_offset`, `with_contiguous_read`, `as_byte_ptr`, `element_size`）
  - 测试: 编译通过
  - 前置: T1, T4
  - 预计: 10 min

- [ ] **T6: FfiExt trait 实现（只读方法）**
  - 文件: `src/ffi.rs`
  - 内容: `impl<S, A, D> FfiExt<A, D> for TensorBase<S, D>` 的 `lda`, `is_blas_compatible`, `blas_layout`, `blas_trans`, `is_fortran_contiguous`, `max_offset`, `as_byte_ptr`, `element_size`
  - 测试: `test_ffi_lda`, `test_ffi_blas_layout_f_order`, `test_ffi_blas_layout_c_order`, `test_ffi_blas_trans`, `test_ffi_is_fortran_contiguous`, `test_ffi_max_offset`
  - 前置: T5
  - 预计: 10 min

- [ ] **T7: FfiExtMut trait 定义与实现**
  - 文件: `src/ffi.rs`
  - 内容: `FfiExtMut<A, D>` trait（`as_mut_byte_ptr`, `with_contiguous_write`）及其实现
  - 测试: `test_ffi_as_mut_byte_ptr`, `test_ffi_with_contiguous_write_fast_path`, `test_ffi_with_contiguous_write_slow_path`
  - 前置: T6
  - 预计: 10 min

- [ ] **T8: with_contiguous_read / with_contiguous_write 实现**
  - 文件: `src/ffi.rs`
  - 内容: 回调调用的 fast path（连续数据直接传递）和 slow path（非连续数据拷贝后传递）
  - 测试: `test_with_contiguous_read_direct`, `test_with_contiguous_read_copy`, `test_with_contiguous_write_direct`, `test_with_contiguous_write_copy`
  - 前置: T6, T7
  - 预计: 10 min

### Wave 4: 所有权转移

- [ ] **T9: into_raw_parts 实现**
  - 文件: `src/ffi.rs`
  - 内容: `Tensor<A, D>::into_raw_parts(self) -> RawParts<A>`，包含 `mem::forget` 语义
  - 测试: `test_into_raw_parts_shape`, `test_into_raw_parts_no_double_free`, `test_into_raw_parts_ptr_valid`
  - 前置: T2
  - 预计: 10 min

- [ ] **T10: from_raw_parts_owned 实现**
  - 文件: `src/ffi.rs`
  - 内容: `Tensor<A, D>::from_raw_parts_owned(RawParts<A>) -> Self`（unsafe），含完整 Safety 文档
  - 测试: `test_from_raw_parts_owned_roundtrip`, `test_from_raw_parts_owned_null_panics`
  - 前置: T9
  - 预计: 10 min

### Wave 5: 文档与集成

- [ ] **T11: Display 实现 + 文档注释完善**
  - 文件: `src/ffi.rs`
  - 内容: `Display for BlasLayout`, `Display for BlasTrans`, 补全所有 pub 项 doc comments
  - 测试: `test_display_blas_layout`, `test_display_blas_trans`
  - 前置: T1
  - 预计: 8 min

- [ ] **T12: lib.rs 集成 + re-export**
  - 文件: `src/lib.rs`, `src/ffi.rs`
  - 内容: `pub mod ffi;` 声明、re-export `BlasLayout`, `BlasTrans`, `RawParts`, `FfiExt`, `FfiExtMut` 及独立函数
  - 测试: 外部 `use Renon::ffi::*` 编译通过
  - 前置: T1-T11
  - 预计: 5 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/ffi.rs` 中的 `#[cfg(test)] mod tests`：

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **枚举** | `test_blas_layout_variants` | `ColMajor ≠ RowMajor ≠ None` |
| | `test_blas_trans_as_char` | `NoTrans → b'N'`, `Trans → b'T'`, `ConjTrans → b'C'` |
| | `test_display_blas_layout` | `"ColMajor"`, `"RowMajor"`, `"None"` |
| | `test_display_blas_trans` | `"NoTrans"`, `"Trans"`, `"ConjTrans"` |
| **RawParts** | `test_raw_parts_size_and_alignment` | `size_of::<RawParts<f64>>() > 0`, 字段偏移量合理 |
| **LDA 计算** | `test_compute_lda_f_order` | shape `[3,4]` strides `[1,3]` → `lda=3` |
| | `test_compute_lda_c_order` | shape `[3,4]` strides `[4,1]` → `lda=4` |
| | `test_compute_lda_panics_1d` | shape `[5]` → panic |
| | `test_compute_lda_panics_3d` | shape `[2,3,4]` → panic |
| **BLAS 兼容性** | `test_is_blas_compatible_f_order` | F-contiguous → `true` |
| | `test_is_blas_compatible_c_order` | C-contiguous → `true` |
| | `test_is_blas_compatible_broadcast` | strides 含 0 → `false` |
| | `test_is_blas_compatible_neg_stride` | strides 含负值 → `false` |
| | `test_is_blas_compatible_non_contig` | strides `[2, 6]` → `false` |
| **BLAS 布局** | `test_classify_blas_layout_f` | F-contiguous → `ColMajor` |
| | `test_classify_blas_layout_c` | C-contiguous → `RowMajor` |
| | `test_classify_blas_layout_1d` | 1D 同时为 F 和 C → `ColMajor`（F 优先） |
| | `test_classify_blas_layout_none` | 非连续 → `None` |
| **Max Offset** | `test_compute_max_offset_2d` | shape `[3,4]` strides `[1,3]` → `11` |
| | `test_compute_max_offset_1d` | shape `[5]` strides `[1]` → `4` |
| | `test_compute_max_offset_zero_dim` | shape `[0,3]` → `0` |
| **FfiExt** | `test_ffi_lda` | 通过 trait 方法调用，返回正确 lda |
| | `test_ffi_blas_trans_f_order` | `Some(NoTrans)` |
| | `test_ffi_blas_trans_c_order` | `Some(Trans)` |
| | `test_ffi_blas_trans_non_contig` | `None` |
| | `test_ffi_is_fortran_contiguous` | F-order tensor → `true` |
| | `test_ffi_as_byte_ptr` | `as_byte_ptr()` 指向正确地址 |
| | `test_ffi_element_size` | `f64` → `8`, `f32` → `4` |

### 7.2 集成测试

位于 `tests/ffi.rs`：

| 测试项 | 测试内容 |
|--------|----------|
| `test_into_raw_parts_roundtrip` | `Tensor → into_raw_parts → from_raw_parts_owned → Tensor`，数据不变 |
| `test_into_raw_parts_no_double_free` | `into_raw_parts` 后原 Tensor 不执行 Drop（miri 验证） |
| `test_into_raw_parts_shape_strides` | 返回的 shape/strides 与原 Tensor 一致 |
| `test_from_raw_parts_null_panics` | null ptr → panic |
| `test_from_raw_parts_view_lifetime` | View 的生命周期正确绑定到源数据 |
| `test_from_raw_parts_mut_exclusive` | ViewMut 的独占语义正确（编译时验证） |
| `test_with_contiguous_read_fast_path` | F-contiguous tensor 直接传递指针（指针地址相同） |
| `test_with_contiguous_read_slow_path` | 非连续 tensor 拷贝后传递（数据一致） |
| `test_with_contiguous_write_fast_path` | F-contiguous tensor 直接修改（回调写入后数据可见） |
| `test_with_contiguous_write_slow_path` | 非连续 tensor 拷贝 → 修改 → 写回（数据一致） |
| `test_blas_workflow_gemm_params` | 模拟 dgemm 参数计算：两个 F-order 矩阵的 lda/trans 正确 |
| `test_c_order_blas_transpose` | C-order 矩阵通过 `Trans` 标志正确映射到 BLAS |
| `test_blas_layout_after_slice` | 切片后 blas_layout 正确降级 |
| `test_blas_layout_after_transpose` | 转置后 blas_layout F↔C 翻转 |

### 7.3 边界测试

| 场景 | 预期行为 |
|------|----------|
| 空张量（shape 含 0） | `is_blas_compatible() == true`，`lda()` 正常，`max_offset() == 0` |
| 单元素张量 | `blas_layout() == ColMajor`，`blas_trans() == Some(NoTrans)` |
| 标量（Ix0） | `lda()` panic（非 2D），`is_blas_compatible() == true` |
| 1D 张量 | `lda()` panic（非 2D），`blas_layout() == ColMajor` |
| 大矩阵（10000x10000） | `lda()` 返回正确值，无溢出 |
| 广播张量（stride=0） | `is_blas_compatible() == false`，`blas_layout() == None` |
| 反转轴（stride<0） | `is_blas_compatible() == false`，`blas_layout() == None` |
| 非对齐数据 | `is_fortran_contiguous()` 不受对齐影响 |

### 7.4 Safety 测试（miri）

| 测试项 | 验证内容 |
|--------|----------|
| `test_from_raw_parts_valid_memory` | 从有效内存构造 View，读取不触发 UB |
| `test_into_raw_parts_no_leak` | `into_raw_parts` + `from_raw_parts_owned` 循环不泄漏（miri 验证） |
| `test_with_contiguous_callback_no_ub` | 回调在连续和非连续路径上均无 UB |

### 7.5 编译时测试（ui tests）

```
tests/ui/
├── ffi_ext_not_on_non_tensor.rs    // 验证 FfiExt 仅对 TensorBase 实现
└── from_raw_parts_owned_not_on_view.rs  // 验证 from_raw_parts_owned 仅对 Tensor<A,D> 可用
```
