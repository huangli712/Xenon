# FFI 接口模块设计

> 文档编号: 23 | 模块: `src/ffi/` | 阶段: Phase 4
> 前置文档: `07-tensor.md`, `06-memory.md`
> 需求参考: 需求说明书 §25

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 原始指针 API | `as_ptr()`/`as_mut_ptr()` | BLAS 绑定实现（由上游库通过 `blas-sys` crate 提供） |
| 裸指针构造张量 | `from_raw_parts`/`from_raw_parts_mut` | GPU 内存操作 |
| 裸指针解构张量 | `into_raw_parts` | 跨进程共享内存 |
| BLAS 兼容性 API | `blas_layout()`/`is_blas_compatible()` | 自动调用 BLAS（由上游库负责） |
| 多维索引转换 | `offset_of()`/`ptr_at()` | 序列化/反序列化 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零拷贝 | 指针 API 无数据拷贝，O(1) 开销 |
| 安全边界清晰 | 所有 unsafe 函数有详尽 Safety 文档 |
| BLAS 友好 | 提供完整的 BLAS 兼容性检查和布局查询 |
| 最小约束 | FFI 方法避免重复安全检查（调用方已 unsafe） |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: ffi  ← 当前模块
```

---

## 2. 文件位置

```
src/
└── ffi/
    ├── mod.rs         # 模块根，re-exports
    ├── types.rs       # BlasLayout, BlasTrans, BlasInfo 类型定义
    ├── ptr.rs         # 原始指针 API（as_ptr, as_mut_ptr, from_raw_parts, from_raw_parts_mut, into_raw_parts）
    ├── blas.rs        # BLAS 兼容性检查（is_blas_compatible, blas_info, lda）
    └── offset.rs      # 多维索引到指针偏移（offset_of, ptr_at）
```

多文件设计：将 FFI 按职责拆分为多个文件，便于后期拓展和维护。

| 文件 | 职责 |
|------|------|
| `mod.rs` | 模块入口，导出公共 API |
| `types.rs` | `BlasLayout`/`BlasTrans` 枚举、`BlasInfo` 结构体 |
| `ptr.rs` | 原始指针访问（`as_ptr`/`as_mut_ptr`）和裸指针构造/解构（`from_raw_parts`/`into_raw_parts`） |
| `blas.rs` | BLAS 兼容性检查和参数查询（`is_blas_compatible`/`blas_info`/`lda`） |
| `offset.rs` | 多维索引到偏移量和指针转换（`offset_of`/`ptr_at`） |

---

## 3. 依赖关系

### 3.1 依赖图

```
src/ffi/
├── mod.rs
│   └── re-exports from types, ptr, blas, offset
├── types.rs
│   └── (无外部依赖，仅 core)
├── ptr.rs
│   ├── crate::tensor        # TensorBase<S, D>, offset
│   ├── crate::dimension     # Dimension trait
│   ├── crate::storage       # Storage, StorageMut, StorageIntoRaw
│   └── crate::layout        # is_f_contiguous
├── blas.rs
│   ├── crate::tensor        # TensorBase<S, D>
│   ├── crate::storage       # Storage
│   ├── crate::layout        # is_contiguous, has_zero_stride, has_neg_stride
│   └── super::types         # BlasInfo, BlasLayout
│   └── super::ptr           # as_ptr
└── offset.rs
    ├── crate::tensor        # TensorBase<S, D>
    ├── crate::dimension     # Dimension trait
    └── crate::storage       # Storage<Elem=A>
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait | 参考 | 使用者 |
|----------|-----------------|------|--------|
| `tensor` | `TensorBase<S, D>`, `.shape()`, `.strides()`, `.as_ptr()`, `.as_mut_ptr()`, `.offset()` | `07-tensor.md` §4 | `ptr.rs`, `blas.rs`, `offset.rs` |
| `dimension` | `Dimension`, `Ix0`~`Ix6`, `IxDyn` | `02-dimension.md` §4 | `ptr.rs`, `offset.rs` |
| `storage` | `Storage<Elem=A>`, `StorageMut<Elem=A>`, `StorageIntoRaw` | `05-storage.md` §4 | `ptr.rs`, `blas.rs`, `offset.rs` |
| `layout` | `is_f_contiguous()`, `has_zero_stride()`, `has_neg_stride()` | `06-memory.md` §4 | `ptr.rs`, `blas.rs` |

### 3.3 依赖方向声明

> **依赖方向：单向向上。** `ffi` 仅消费 `tensor`、`storage` 等核心模块，为上游库提供接口。

---

## 4. 公共 API 设计

### 4.1 辅助类型

```rust
/// BLAS matrix layout identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasLayout {
    /// Column-major (Fortran order).
    /// Corresponds to BLAS `CblasColMajor` (102).
    ColumnMajor,
}

/// BLAS transpose identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlasTrans {
    /// No transpose.
    NoTrans,
    /// Transpose.
    Trans,
    /// Conjugate transpose (complex only).
    ConjTrans,
}
```

### 4.2 原始指针 API

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Returns a read-only raw pointer to the data start.
    ///
    /// The pointer points to the first logical element (considering offset).
    /// The returned pointer is invalid after `self` is modified or dropped.
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_ptr();
    /// // Can be passed to read-only C functions
    /// ```
    pub fn as_ptr(&self) -> *const A {
        // SAFETY: self.storage.as_ptr() returns a valid non-null pointer.
        // self.offset is guaranteed to be within bounds by TensorBase construction invariants.
        unsafe {
            self.storage.as_ptr().add(self.offset)
        }
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
{
    /// Returns a mutable raw pointer to the data start.
    ///
    /// Only available for writable storage (Owned, ViewMut).
    ///
    /// # Example
    ///
    /// ```
    /// let mut tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let ptr = tensor.as_mut_ptr();
    /// // Can be passed to C functions requiring a mutable pointer
    /// ```
    pub fn as_mut_ptr(&mut self) -> *mut A {
        // SAFETY: self.storage.as_mut_ptr() returns a valid non-null mutable pointer.
        // self.offset is guaranteed to be within bounds by TensorBase construction invariants.
        // The &mut self reference ensures exclusive access.
        unsafe {
            self.storage.as_mut_ptr().add(self.offset)
        }
    }
}
```

### 4.3 从裸指针构造张量

```rust
impl<'a, A, D> TensorBase<ViewRepr<&'a A>, D>
where
    D: Dimension,
{
    /// Constructs an immutable view from raw pointer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Data start pointer (immutable)
    /// * `shape` - Length of each axis
    /// * `strides` - Strides per axis (element units, signed)
    /// * `offset` - Data start offset (element units)
    ///
    /// # Returns
    ///
    /// A new `TensorView<'a, A, D>` instance.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following:
    ///
    /// | Prerequisite | Description |
    /// |----------|------|
    /// | Pointer validity | `ptr` must be non-null, non-dangling, and aligned to `align_of::<A>()` |
    /// | Memory range | Memory starting from `ptr` must cover all accessible elements (considering offset, shape, strides) |
    /// | Lifetime | Memory must remain valid for lifetime `'a` |
    /// | Aliasing rules | Memory can be read-shared but must not be written to |
    /// | Layout consistency | `shape` and `strides` lengths must match |
    /// | Element initialization | All accessible elements must be properly initialized |
    ///
    /// # Example
    ///
    /// ```
    /// let data: [f64; 12] = [0.0; 12];
    /// let view = unsafe {
    ///     TensorView2::from_raw_parts(
    ///         data.as_ptr(),
    ///         [3, 4],
    ///         Strides::from_slice(&[1, 3]),
    ///         0,
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw_parts(
        ptr: *const A,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Self {
        // SAFETY: Caller guarantees ptr is valid, aligned, and points to
        // properly initialized data for the lifetime 'a. Shape and strides
        // are consistent and describe a valid memory region.
        TensorBase {
            storage: ViewRepr::new(ptr),
            shape,
            strides,
            offset,
            flags: layout::compute_flags(&shape, &strides, ptr),
        }
    }

    /// Create from raw parts with isize strides (for FFI compatibility).
    ///
    /// FFI users (e.g., BLAS, LAPACK) commonly work with `isize` strides
    /// where negative strides indicate reversed axes. This method converts
    /// `isize` strides to the internal `Strides<D>` representation.
    ///
    /// # Safety
    ///
    /// Same safety requirements as `from_raw_parts`, with additional constraint:
    /// strides must be representable in the internal Dimension type.
    pub unsafe fn from_raw_parts_isize_strides(
        ptr: *const A,
        shape: D,
        strides: &[isize],  // isize strides from external code
        offset: usize,
    ) -> Self
    where
        D: Dimension,
    {
        // Convert isize strides to internal Dimension representation
        let dim_strides = Strides::<D>::from_slice(strides);
        TensorBase {
            storage: ViewRepr::new(ptr),
            shape,
            strides: dim_strides,
            offset,
            flags: layout::compute_flags(&shape, &dim_strides, ptr),
        }
    }
}

impl<'a, A, D> TensorBase<ViewMutRepr<&'a mut A>, D>
where
    D: Dimension,
{
    /// Constructs a mutable view from raw pointer.
    ///
    /// Same as `from_raw_parts`, but requires exclusive access (no other references).
    ///
    /// # Safety
    ///
    /// Same as `from_raw_parts`, with additional requirement: no other references to the memory,
    /// and the logical element set described by `(shape, strides, offset)` must not alias itself.
    ///
    /// # Example
    ///
    /// ```
    /// let mut data: [f64; 12] = [0.0; 12];
    /// let view = unsafe {
    ///     TensorViewMut2::from_raw_parts_mut(
    ///         data.as_mut_ptr(),
    ///         [3, 4],
    ///         Strides::from_slice(&[1, 3]),
    ///         0,
    ///     )
    /// };
    /// ```
    pub unsafe fn from_raw_parts_mut(
        ptr: *mut A,
        shape: D,
        strides: Strides<D>,
        offset: usize,
    ) -> Self {
        // SAFETY: Caller guarantees exclusive mutable access to the memory
        // for lifetime 'a. Same validity requirements as from_raw_parts.
        TensorBase {
            storage: ViewMutRepr::new(ptr),
            shape,
            strides,
            offset,
            flags: layout::compute_flags(&shape, &strides, ptr),
        }
    }
}
```

### 4.4 将张量解构为裸指针

```rust
impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// Consumes the tensor, returning raw parts.
    ///
    /// The caller is responsible for freeing the returned memory.
    ///
    /// # Returns
    ///
    /// A tuple `(ptr, shape, strides, offset)` where `strides` uses `Strides<D>`.
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// let (ptr, shape, strides, offset) = tensor.into_raw_parts();
    /// // Caller now owns ptr, responsible for freeing
    /// ```
    pub fn into_raw_parts(self) -> (*mut A, D, Strides<D>, usize) {
        // Use ManuallyDrop to prevent Drop from running while we extract fields.
        // This is safer than the "extract fields first, then mem::forget" pattern
        // because it eliminates the risk of Double Drop on panic during field access.
        let this = core::mem::ManuallyDrop::new(self);
        let ptr = unsafe { this.storage.as_mut_ptr() };
        let shape = this.shape.clone();
        let strides = this.strides.clone();
        let offset = this.offset;

        (ptr, shape, strides, offset)
    }
}
```

> **设计决策：** `into_raw_parts` 仅适用于 Owned 存储，且导出的内存布局必须满足 Xenon 的 owned 不变量：F-order contiguous、`offset == 0`。View/ViewMut 的数据仍由原借用绑定，调用方应谨慎处理。如需将 View 转为 Owned 再解构，参见 `21-type.md` §4.5。

#### 内存管理

`into_raw_parts()` 返回的指针由 Xenon 的 64 字节对齐分配器分配。正确回收内存的方式如下：

| 规则 | 说明 |
|------|------|
| ✅ 重建张量后 Drop | 使用 `Tensor::from_raw_parts_owned()` 重建，让 Drop 处理释放 |
| ❌ 直接调用系统 free | 分配器不匹配，导致 UB 或内存泄漏 |
| ❌ 忽略返回值 | 内存泄漏 |

```rust
/// Reconstructs an owned tensor from raw parts obtained via `into_raw_parts`.
/// Takes ownership of memory allocated by Xenon's aligned allocator.
///
/// # Safety
///
/// - `ptr` must point to memory allocated by Xenon's `AlignedAlloc` (64-byte aligned)
/// - `shape` and `strides` must describe a valid, non-overlapping F-order layout
/// - The caller transfers ownership; do NOT free `ptr` separately
/// - Total elements accessible via shape/strides must not exceed allocated size
/// - `offset` must be 0 for owned round-trips; non-zero offset requires re-materializing first
pub unsafe fn from_raw_parts_owned(
    ptr: *mut A,
    shape: D,
    strides: Strides<D>,
    offset: usize,
) -> TensorBase<Owned<A>, D> {
    let len = shape.size();
    let storage = Owned::from_raw(ptr, len);
    let flags = layout::compute_flags(&shape, &strides, ptr);
    TensorBase { storage, shape, strides, offset, flags }
}
```

```rust
// Correct round-trip: into_raw_parts → use pointer → from_raw_parts_owned → drop
let tensor = Tensor2::<f64>::zeros([3, 4]);
let (ptr, shape, strides, offset) = tensor.into_raw_parts();

// ... use ptr in FFI code ...

// Reconstruct and let Drop handle deallocation
unsafe {
    let reconstructed = Tensor::<f64, _>::from_raw_parts_owned(ptr, shape, strides, offset);
    drop(reconstructed);  // Correctly deallocates with Xenon's aligned allocator
}
```

### 4.5 BLAS 兼容性 API

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Checks whether the memory layout can be directly passed to BLAS.
    ///
    /// # BLAS Compatibility Conditions
    ///
    /// | Condition | Description |
    /// |------|------|
    /// | Contiguity | F-contiguous (Xenon only supports F-order) |
    /// | Positive strides | All strides > 0 (no reversed dimensions) |
    /// | No zero strides | No broadcast dimensions |
    ///
    /// # Returns
    ///
    /// `true` if directly passable to BLAS; `false` if a copy is needed first.
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert!(a.is_blas_compatible());
    ///
    /// let b = a.slice(s![.., 1..3]);
    /// assert!(!b.is_blas_compatible());
    /// ```
    pub fn is_blas_compatible(&self) -> bool {
        self.is_f_contiguous()      // method name: see 07-tensor.md §4.3
            && !self.has_zero_stride()
            && !self.has_neg_stride()
    }
}
```

### 4.6 blas_info 和 BlasInfo 结构体

```rust
/// BLAS matrix information.
///
/// Contains all parameters needed for BLAS function calls.
pub struct BlasInfo {
    /// Data pointer (generic byte pointer).
    ///
    /// Note: `data_ptr` is typed as `*const u8` for generality.
    /// When calling BLAS functions, cast to the concrete type:
    /// `blas_info.data_ptr as *const f64`.
    pub data_ptr: *const u8,
    /// Leading dimension (element units).
    pub leading_dim: i32,
    /// Number of rows.
    pub rows: i32,
    /// Number of columns.
    pub cols: i32,
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns BLAS layout identifier and parameter information.
    ///
    /// # Returns
    ///
    /// - `Some(BlasInfo)`: compatibility conditions met
    /// - `None`: not BLAS compatible
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// let info = a.blas_info().unwrap();
    /// assert_eq!(info.rows, 3);
    /// assert_eq!(info.cols, 4);
    /// ```
    pub fn blas_info(&self) -> Option<BlasInfo> {
        if !self.is_blas_compatible() || self.ndim() != 2 {
            return None;
        }

        let data_ptr = self.as_ptr() as *const u8;
        let lda = self.lda()? as i32;
        let rows = self.shape()[0] as i32;
        let cols = self.shape()[1] as i32;

        Some(BlasInfo {
            data_ptr,
            leading_dim: lda,
            rows,
            cols,
        })
    }
}
```

### 4.7 LDA 查询

```rust
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns the leading dimension (only meaningful for 2D arrays).
    ///
    /// For F-order matrix `A[M, N]`, `LDA = stride[1]`.
    ///
    /// **Note:** `lda()` 仅对 BLAS-compatible 的 2D 张量有效。对非连续张量（如切片后的视图），
    /// 返回的步长无法直接用于 BLAS 调用。建议在调用前先检查 `is_f_contiguous()`。
    ///
    /// # Returns
    ///
    /// - `Some(isize)`: LDA of a BLAS-compatible 2D array
    /// - `None`: not a BLAS-compatible 2D array
    ///
    /// # Example
    ///
    /// ```
    /// let a = Tensor2::<f64>::zeros([3, 4]);
    /// assert_eq!(a.lda(), Some(3));  // F-order, LDA = M = 3
    /// ```
    pub fn lda(&self) -> Option<isize> {
        if self.ndim() != 2 || !self.is_blas_compatible() {
            return None;
        }
        let strides = self.strides();  // returns &[isize] (see 07-tensor.md §4.3)
        Some(strides[1])               // already isize, no cast needed
    }
}
```

### 4.8 多维索引到指针偏移

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Converts a multi-dimensional index to an element offset relative to the
    /// logical first element pointer.
    ///
    /// Offset = Σ(stride[i] * index[i]) for all i in [0, ndim)
    ///
    /// Returns an `isize` offset relative to the logical first element pointer.
    /// Negative strides are already encoded in the signed stride metadata; the
    /// resulting offset for any in-bounds logical index is therefore measured
    /// from `as_ptr()` and does not require a second base adjustment.
    ///
    /// # Panics
    ///
    /// - Index length does not match number of dimensions
    /// - Index out of bounds (any index[i] >= shape[i])
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor2::<f64>::zeros([3, 4]);
    /// // shape=[3,4], strides=[1,3], F-order
    /// // index [1, 2] → offset = 1*1 + 2*3 = 7
    /// assert_eq!(tensor.offset_of(&[1, 2]), 7);
    /// ```
    pub fn offset_of(&self, index: &[isize]) -> isize {
        assert_eq!(index.len(), self.ndim(), "index length must match ndim");
        let strides = self.strides();  // returns &[isize]
        let shape = self.shape();
        let mut offset: isize = 0;
        for i in 0..self.ndim() {
            assert!(index[i] >= 0, "index must be non-negative");
            assert!(index[i] as usize < shape[i], "index out of bounds");
            offset += strides[i] * index[i];
        }
        offset
    }

    /// Converts a multi-dimensional index to a raw pointer to the corresponding element.
    ///
    /// # Panics
    ///
    /// - Index length does not match number of dimensions
    /// - Index out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// let tensor = Tensor1::<i32>::from_vec(vec![10, 20, 30, 40]);
    /// let ptr = tensor.ptr_at(&[2]);
    /// assert_eq!(unsafe { *ptr }, 30);
    /// ```
    pub fn ptr_at(&self, index: &[isize]) -> *const A {
        let offset = self.offset_of(index);
        // SAFETY: offset is within storage bounds as validated by dimension checks
        unsafe { self.as_ptr().offset(offset) }
    }
}
```

### 4.9 Good/Bad 对比

```rust
// blas_trans() is only queried after obtaining a BLAS-compatible representation.
// For Xenon's direct BLAS path, this is always a standard F-order matrix and
// therefore returns BlasTrans::NoTrans.
impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// Returns the BLAS transpose identifier for this tensor.
    ///
    /// Returns `BlasTrans::NoTrans` for BLAS-compatible F-order matrices.
    /// Transposed or otherwise non-BLAS-compatible views must first be converted
    /// into an owned F-order tensor before calling BLAS.
    pub fn blas_trans(&self) -> BlasTrans {
        BlasTrans::NoTrans
    }
}

// Good - Check BLAS compatibility before passing
if tensor.is_blas_compatible() {
    let info = tensor.blas_info().unwrap();
    unsafe {
        call_blas_dgemm(info, tensor.blas_trans(), ...);
    }
} else {
    let contiguous = tensor.to_contiguous();
    let info = contiguous.blas_info().unwrap();
    unsafe {
        call_blas_dgemm(info, contiguous.blas_trans(), ...);
    }
}

// Bad - Pass directly without checking BLAS compatibility
unsafe {
    call_blas_dgemm(CblasColMajor, CblasNoTrans, ...,
        tensor.as_ptr(), tensor.lda().unwrap(),
        ...,
    );  // UB if tensor is non-contiguous!
}
```

---

## 5. 内部实现设计

### 5.1 指针有效性论证

`as_ptr()` 和 `as_mut_ptr()` 的返回值有效性由 Rust 借用检查器保证（`NonNull` 指针在 `Owned` 支持下）。

对于 View 类型，`offset` 在 storage 范围内则结果合法。数据来自原始 Tensor 的 storage，生命周期由原始引用保证。

`from_raw_parts` 的 Safety 由调用方保证：所有前提条件文档为运行时"契约"，违反任何条件都将导致未定义行为。

### 5.2 BLAS 兼容性检查流程

```
is_blas_compatible():
    │
├── is_f_contiguous()? ─── No ──→ false
    │
    ├── has_zero_stride()? ── Yes ──→ false
    │
    └── has_neg_stride()? ── Yes ──→ false
    │
    └── All passed ────────────────→ true
```

---

## 6. 实现任务拆分

### Wave 1: 基础设施

- [ ] **T1**: 创建 `src/ffi/` 模块骨架和辅助类型
  - 文件: `src/ffi/mod.rs`, `src/ffi/types.rs`
  - 内容: 模块声明、re-exports、`BlasLayout`/`BlasTrans` 枚举、`BlasInfo` 结构体
  - 测试: `test_blas_layout_column_major`, `test_blas_trans_variants`
  - 前置: 无
  - 预计: 10 min

### Wave 2: 指针 API

- [ ] **T2**: 实现原始指针访问和裸指针构造/解构
  - 文件: `src/ffi/ptr.rs`
  - 内容: `as_ptr()`, `as_mut_ptr()`, `from_raw_parts`, `from_raw_parts_mut`, `into_raw_parts` 及 Safety 文档
  - 测试: `test_as_ptr_basic`, `test_as_mut_ptr_basic`, `test_from_raw_parts_roundtrip`, `test_into_raw_parts`
  - 前置: T1
  - 预计: 25 min

### Wave 3: BLAS 和索引（可并行）

- [ ] **T3**: 实现 BLAS 兼容性 API
  - 文件: `src/ffi/blas.rs`
  - 内容: `is_blas_compatible()`, `blas_info()`, `lda()`
  - 测试: `test_is_blas_compatible_f_order`, `test_is_blas_compatible_non_contiguous`, `test_lda_f_order`
  - 前置: T1
  - 预计: 15 min

- [ ] **T4**: 实现多维索引到指针偏移
  - 文件: `src/ffi/offset.rs`
  - 内容: `offset_of()`, `ptr_at()`
  - 测试: `test_offset_of_various`, `test_ptr_at_various`
  - 前置: T1
  - 预计: 10 min

### 并行执行图

```
Wave 1:    [T1]
             │
Wave 2:    [T2]
             │
Wave 3: ┌────┴────┐
        │         │
       [T3]      [T4]   (可并行)
```

---

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_as_ptr_basic` | `as_ptr()` 返回有效指针 | 高 |
| `test_as_mut_ptr_basic` | `as_mut_ptr()` 返回有效可写指针 | 高 |
| `test_as_ptr_offset` | 指针考虑 offset 后指向正确元素 | 高 |
| `test_is_blas_compatible_f_order` | F-order 连续数组兼容 | 高 |
| `test_is_blas_compatible_non_contiguous` | 非连续切片不兼容 | 高 |
| `test_is_blas_compatible_broadcast` | 广播维度（零步长）不兼容 | 高 |
| `test_is_blas_compatible_flipped` | 负步长（翻转）不兼容 | 高 |
| `test_blas_info_f_order` | F-order 返回正确 BlasInfo | 高 |
| `test_lda_f_order` | F-order [3,4] 返回 3 | 高 |
| `test_lda_non_contiguous` | 非连续（切片）数组 lda() 返回 None | 中 |
| `test_from_raw_parts_roundtrip` | 构造 → 读取一致性 | 高 |
| `test_from_raw_parts_mut_roundtrip` | 可变构造 → 修改 → 读取 | 高 |
| `test_into_raw_parts` | Owned 张量解构后指针有效 | 高 |
| `test_into_raw_parts_memory_leak` | 解构后正确释放 | 中 |
| `test_offset_of_various` | 各种索引的偏移量正确性 | 高 |
| `test_ptr_at_various` | 各种索引的指针正确性 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空张量 | `as_ptr()` 返回非空但不应解引用 |
| 单元素张量 | `as_ptr()` 指向唯一元素 |
| 非连续切片 | `is_blas_compatible()` 返回 `false` |
| 广播维度 | `is_blas_compatible()` 返回 `false` |
| 1D 张量 | `lda()` 返回 `None` |
| 零维张量 | `offset_of(&[])` 返回 0 |
| 未对齐指针 | `from_raw_parts` 的 Safety 文档需说明对齐要求 |

### 7.3 内存安全测试

| 场景 | 验证方式 |
|------|----------|
| `from_raw_parts` + Drop | 无内存泄漏（借用语义） |
| `into_raw_parts` + 手动释放 | 正确释放（通过 allocator API） |
| `from_raw_parts` 野指针 | AddressSanitizer 检测 |

### 7.4 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/ffi.rs` | 指针 API / BLAS 兼容检查 / raw-parts roundtrip 与 `tensor`、`layout`、`storage` 的端到端协同路径 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 指针访问 | ffi → tensor | 通过 `TensorBase` 的 storage 获取指针（参见 `07-tensor.md` §4） |
| BLAS 检查 | ffi ← layout | 使用 `is_f_contiguous()`、`has_zero_stride()`、`has_neg_stride()`（参见 `06-memory.md` §4） |
| 解构 | ffi → storage | `into_raw_parts` 使用 `StorageIntoRaw` trait（参见 `05-storage.md` §4） |
| BLAS 参数 | 上游库 ← ffi | 上游 BLAS 库调用 `blas_info()`、`lda()` 等获取参数 |

### 8.2 数据流描述

```text
上游库调用 as_ptr() / blas_info() / into_raw_parts()
    │
    ├── ffi 模块从 tensor/storage 读取原始指针、shape、strides、offset
    ├── layout 模块负责判断 BLAS 兼容性与 leading-dimension 前提
    ├── raw-parts 路径在 owned roundtrip 时把所有权移交给调用方或重建回 tensor
    └── 最终向外部 C / BLAS 边界暴露零拷贝参数
```

---

## 9. 设计决策记录

### 决策 1: BLAS 兼容 API 设计

| 属性 | 值 |
|------|-----|
| 决策 | 提供结构化的 `BlasInfo` 查询方法，而非仅返回布尔值 |
| 理由 | 上游库需要完整的 BLAS 参数（data ptr、lda、rows、cols），结构体返回比单独方法调用更便捷 |
| 替代方案 | 仅返回 `bool is_blas_compatible()` — 放弃，上游库需要重复获取多个参数 |
| 替代方案 | 返回 raw C 常量 — 放弃，不符合 Rust 惯例 |

> **补充**：Xenon 的直接 BLAS 路径只接受 BLAS-compatible 的 F-order 2D 张量。转置或非连续视图必须先显式 materialize 为 `to_contiguous()` 结果，再以 `BlasTrans::NoTrans` 传给上游 BLAS。

### 决策 2: Safety 独立边界

| 属性 | 值 |
|------|-----|
| 决策 | `from_raw_parts` 和 `from_raw_parts_mut` 使用最小 Safety 模约束集 |
| 理由 | 将安全责任尽可能交给调用方，库本身不做额外假设；与 `std::slice::from_raw_parts` 设计一致 |
| 替代方案 | 库内部验证所有 Safety 条件 — 放弃，运行时开销过大（O(n) 检查） |

### 决策 3: 性能 — 零拷贝优先

| 属性 | 值 |
|------|-----|
| 决策 | FFI 方法避免重复安全检查 |
| 理由 | FFI 的核心价值是零开销；重复检查会增加不必要的运行时开销；调用方已在 unsafe 块中，可自行决定安全级别 |
| 替代方案 | 每次调用都检查连续性 — 放弃，与零开销目标冲突 |

---

## 10. 性能考量

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| `as_ptr()` | O(1) | 仅指针加法 |
| `as_mut_ptr()` | O(1) | 仅指针加法 |
| `is_blas_compatible()` | O(1) | 检查布局标志 |
| `blas_info()` | O(1) | 包含 `is_blas_compatible()` + 构造 |
| `lda()` | O(1) | 步长查询 |
| `offset_of()` | O(ndim) | 逐轴计算 |
| `ptr_at()` | O(ndim) | `offset_of()` + 指针加法 |
| `from_raw_parts()` | O(1) | 仅构造视图 |
| `into_raw_parts()` | O(1) | 提取字段 + `ManuallyDrop` |

**性能提示**:

- `as_ptr()` 和 `as_mut_ptr()` 应标注 `#[inline]`
- `offset_of()` 在热路径中可能需要内联
- `is_blas_compatible()` 检查现有 `LayoutFlags`，无需重新计算

---

## 11. no_std 兼容性

FFI 模块完全兼容 `no_std` 环境。所有操作均为指针运算和结构体构造，无堆分配。存储层的 `no_std` 兼容性参见 `05-storage.md` §11，布局层的 `no_std` 兼容性参见 `06-memory.md` §11。

```rust
// No extern crate alloc needed — FFI module uses no heap allocation
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `BlasLayout` / `BlasTrans` / `BlasInfo` | ✅ | 纯枚举/结构体，无分配 |
| `as_ptr()` / `as_mut_ptr()` | ✅ | 指针加法，O(1) |
| `from_raw_parts` / `from_raw_parts_mut` | ✅ | 构造视图，O(1)，无分配 |
| `into_raw_parts` | ✅ | 字段提取 + `core::mem::ManuallyDrop`，无分配 |
| `is_blas_compatible()` | ✅ | 布局标志检查，无分配 |
| `blas_info()` / `lda()` | ✅ | 布局查询，无分配 |
| `offset_of()` / `ptr_at()` | ✅ | 算术运算，O(ndim)，无分配 |

条件编译处理：

```rust
// All FFI methods use only core::mem, core::ptr, core::ops
// No alloc::vec::Vec or std::ffi required
// into_raw_parts uses core::mem::ManuallyDrop (available in no_std)
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.1.2 | 2026-04-10 |
| 1.1.3 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
