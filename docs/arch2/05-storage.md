# 存储系统模块设计

> Renon 存储系统：定义张量数据的所有权模型、访问控制与内存管理策略。

---

## 1. 模块定位

存储系统是 Renon 张量库的核心基础设施层，位于模块依赖链的中间位置：

```
dimension ──┐
element   ──┼──→ storage ──→ tensor ──→ (API 模块)
layout    ──┘
```

存储系统解决的核心问题：

| 问题 | 解决方案 |
|------|----------|
| 谁拥有数据？ | 四种存储模式通过类型系统区分所有权 |
| 谁可以读写？ | 三层 trait 体系（Storage / RawStorage / StorageMut）在编译时保证 |
| 数据如何分配？ | 64 字节对齐堆分配（Owned）或零分配视图（View） |
| 如何跨线程共享？ | ArcRepr 提供原子引用计数 + 写时复制 |
| 如何扩展到 GPU？ | Device 关联类型预留设备扩展点 |

---

## 2. 文件位置

```
src/storage/
├── mod.rs          # Storage/RawStorage/StorageMut trait, Device trait, Cpu type, re-exports
├── owned.rs        # Owned<A> struct and implementations
├── view.rs         # ViewRepr<&'a A> struct and implementations
├── view_mut.rs     # ViewMutRepr<&'a mut A> struct and implementations
└── arc.rs          # ArcRepr<A> struct and implementations (COW via make_mut)
```

---

## 3. 依赖关系

### 3.1 上游依赖（本模块需要）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `Element` trait | `crate::element` | 约束存储元素的类型要求（Copy, Clone, Send, Sync 等） |
| `Dimension` trait | `crate::dimension` | 关联类型 `D` 用于 shape/stride（Storage trait 不直接使用，但 tensor 层通过 Storage 获取原始数据） |
| `DEFAULT_ALIGNMENT` | `crate::layout` 或模块常量 | 64 字节对齐常量 |

### 3.2 下游消费者

| 消费者 | 用途 |
|--------|------|
| `tensor.rs` (TensorBase<S, D>) | S 参数即存储类型，通过 Storage trait 访问数据 |
| `construction.rs` | 构造函数创建 Owned<A> 存储 |
| `shape/slice.rs` | 切片操作创建 ViewRepr / ViewMutRepr |
| `ops/*` | 运算通过 RawStorage / StorageMut 读写数据 |
| `ffi.rs` | 通过 `as_ptr()` 获取原始指针 |

### 3.3 标准库 / alloc 依赖

```rust
// Conditional imports for no_std compatibility
#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::ptr::NonNull;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
```

---

## 4. 公共 API 设计

### 4.1 常量

```rust
/// Default memory alignment for owned storage allocations (AVX-512 cache line).
pub const DEFAULT_ALIGNMENT: usize = 64;
```

### 4.2 Device trait 与 Cpu 类型

```rust
/// Marker trait for compute devices that manage storage allocation.
///
/// Currently only `Cpu` is implemented. Future versions will add
/// `Cuda`, `Vulkan`, etc.
pub trait Device: Default + Clone + core::fmt::Debug + Send + Sync {
    /// The raw pointer type produced by this device.
    /// For Cpu this is `*mut A`.
    type Ptr<A>;

    /// Allocates `len` elements of type `A` with `alignment`-byte alignment.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len * size_of::<A>()` does not overflow `isize`.
    /// The returned pointer points to uninitialized memory.
    unsafe fn allocate<A>(len: usize, alignment: usize) -> Self::Ptr<A>;

    /// Deallocates memory previously allocated by `allocate`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer was allocated by this device
    /// and that no references to the memory exist.
    unsafe fn deallocate<A>(ptr: Self::Ptr<A>, len: usize, alignment: usize);
}

/// CPU device marker type.
///
/// Uses standard heap allocation with configurable alignment.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cpu;
```

> **设计决策**：Device trait 采用标记 trait 模式。当前版本仅提供 `Cpu`，其 `allocate` 使用 `std::alloc::alloc` / `alloc::alloc` 实现对齐分配。trait 定义中预留了未来 GPU 设备的扩展能力，但不提前引入异步等复杂抽象。

### 4.3 Storage trait（基础 trait）

所有存储模式的公共接口，提供元素类型、设备和原始指针访问。

```rust
/// Base storage trait defining the element type, device, and raw pointer access.
///
/// All four storage modes (Owned, ViewRepr, ViewMutRepr, ArcRepr) implement
/// this trait. It provides the minimal interface needed to inspect storage
/// metadata and obtain a raw const pointer.
///
/// # Type Parameters
///
/// - `A`: Element type (must implement `Element`)
///
/// # Associated Types
///
/// - `Element`: The stored element type
/// - `Device`: The compute device managing this storage
pub trait Storage {
    /// The element type stored in this storage.
    type Element;

    /// The compute device that manages this storage's memory.
    type Device: Device;

    /// Returns a raw const pointer to the start of the stored data.
    ///
    /// The pointer may be offset from the actual allocation base when used
    /// inside a tensor view (offset applied by TensorBase).
    #[inline]
    fn as_ptr(&self) -> *const Self::Element;

    /// Returns the number of elements in this storage.
    ///
    /// This is the total capacity of the underlying buffer, not the number
    /// of logically valid elements (which may be fewer for sliced views).
    #[inline]
    fn len(&self) -> usize;

    /// Returns `true` if this storage contains no elements.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
```

### 4.4 RawStorage trait（只读访问）

继承 `Storage`，提供不可变切片访问。所有四种存储模式均可实现此 trait。

```rust
/// Read-only storage access trait.
///
/// Extends `Storage` with slice-level read access. All four storage modes
/// implement this trait since all support reading.
pub trait RawStorage: Storage {
    /// Returns a shared slice of the stored elements.
    ///
    /// The slice covers the entire storage buffer. For tensor-level access
    /// with offset/stride, use `TensorBase` methods.
    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        if self.is_empty() {
            &[]
        } else {
            // SAFETY: Storage guarantees the pointer is valid for `len` elements.
            unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
        }
    }
}
```

### 4.5 StorageMut trait（读写访问）

继承 `RawStorage`，提供可变指针和可变切片。仅 `Owned<A>` 和 `ViewMutRepr<&'a mut A>` 实现。

```rust
/// Read-write storage access trait.
///
/// Extends `RawStorage` with mutable pointer and slice access.
/// Only `Owned<A>` and `ViewMutRepr<&'a mut A>` implement this trait,
/// enforced at the type level.
pub trait StorageMut: RawStorage {
    /// Returns a raw mutable pointer to the start of the stored data.
    ///
    /// # Safety
    ///
    /// The caller must ensure no other references (shared or mutable)
    /// to the data exist when writing through this pointer.
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut Self::Element;

    /// Returns a mutable slice of the stored elements.
    ///
    /// The slice covers the entire storage buffer.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Element] {
        if self.is_empty() {
            &mut []
        } else {
            // SAFETY: StorageMut guarantees exclusive access; pointer is valid for `len` elements.
            unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
        }
    }
}
```

### 4.6 Owned<A> 结构体

拥有数据的堆分配存储。64 字节对齐，深拷贝克隆。

```rust
/// Owned heap-allocated storage with 64-byte alignment.
///
/// This is the primary storage mode for creating and owning tensor data.
/// The data is allocated on the heap with AVX-512-friendly alignment.
///
/// # Memory Layout
///
/// - Allocation alignment: 64 bytes by default
/// - Elements are stored contiguously
/// - Drop deallocates via the global allocator
///
/// # Clone Semantics
///
/// Cloning performs a deep copy of all elements (O(n)).
///
/// # Examples
///
/// ```ignore
/// use Renon::storage::Owned;
///
/// let s = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(s.len(), 3);
/// ```
pub struct Owned<A> {
    /// Pointer to the heap-allocated, 64-byte-aligned buffer.
    ptr: NonNull<A>,

    /// Number of elements in the buffer.
    len: usize,

    /// Alignment used for this allocation (always a power of 2, >= align_of::<A>()).
    alignment: usize,
}

// === Owned: constructors ===

impl<A: Element> Owned<A> {
    /// Creates a new `Owned` storage by copying elements from a slice.
    ///
    /// Allocates a 64-byte-aligned buffer and copies all elements from `data`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` * `size_of::<A>()` overflows `isize`.
    pub fn from_slice(data: &[A]) -> Self;

    /// Creates a new `Owned` storage by taking ownership of a `Vec<A>`.
    ///
    /// If the Vec's internal allocation is already 64-byte aligned, reuses it.
    /// Otherwise, allocates a new aligned buffer and copies the data.
    ///
    /// # Panics
    ///
    /// Panics if `vec.len()` * `size_of::<A>()` overflows `isize`.
    pub fn from_vec(vec: Vec<A>) -> Self;

    /// Creates a new `Owned` storage with `len` elements, all initialized to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `len` * `size_of::<A>()` overflows `isize`.
    pub fn from_elem(len: usize, value: A) -> Self;

    /// Creates a new `Owned` storage with `len` zero-initialized elements.
    ///
    /// # Panics
    ///
    /// Panics if `len` * `size_of::<A>()` overflows `isize`.
    pub fn zeros(len: usize) -> Self;

    /// Creates a new `Owned` storage with uninitialized memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory is initialized before reading.
    pub unsafe fn uninitialized(len: usize) -> Self;

    /// Returns the alignment of this allocation in bytes.
    #[inline]
    pub fn alignment(&self) -> usize;

    /// Converts this storage into a `Vec<A>`.
    ///
    /// If the allocation is compatible with Vec's layout, this is O(1).
    /// Otherwise, copies the data into a new Vec.
    pub fn into_vec(self) -> Vec<A>;
}

// === Owned: Storage / RawStorage / StorageMut impls ===

impl<A> Storage for Owned<A> {
    type Element = A;
    type Device = Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A;

    #[inline]
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool;
}

impl<A> RawStorage for Owned<A> {}

impl<A> StorageMut for Owned<A> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut A;
}

// === Owned: Clone (deep copy) ===

impl<A: Element> Clone for Owned<A> {
    /// Performs a deep copy of the storage buffer.
    ///
    /// Allocates a new 64-byte-aligned buffer and copies all elements.
    /// Complexity: O(n).
    fn clone(&self) -> Self;
}

// === Owned: Drop ===

impl<A> Drop for Owned<A> {
    /// Deallocates the aligned heap buffer.
    ///
    /// Runs element destructors (if any) before freeing memory.
    fn drop(&mut self);
}

// === Owned: Send / Sync ===

// SAFETY: Owned<A> owns its data exclusively. If A: Send, the buffer
// can be safely sent between threads. If A: Sync, the buffer can be
// safely shared between threads (though Owned provides exclusive access).
unsafe impl<A: Send> Send for Owned<A> {}
unsafe impl<A: Sync> Sync for Owned<A> {}
```

### 4.7 ViewRepr<&'a A> 结构体

不可变借用存储。零分配，O(1) 克隆。

```rust
/// Immutable borrow storage (read-only view).
///
/// Holds a borrowed reference to data owned by another storage.
/// Creating a view is O(1) — no memory allocation occurs.
///
/// # Clone Semantics
///
/// Cloning copies only the pointer metadata (O(1)), not the underlying data.
///
/// # Lifetime
///
/// The `'a` lifetime parameter ensures the view cannot outlive the source data.
pub struct ViewRepr<'a, A> {
    ptr: *const A,
    len: usize,
    _marker: PhantomData<&'a A>,
}

// === ViewRepr: constructors ===

impl<'a, A> ViewRepr<'a, A> {
    /// Creates a new view from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` is valid for `len` elements for the lifetime `'a`.
    /// - No mutable references to the same memory exist for the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *const A, len: usize) -> Self;

    /// Creates a new view from a slice.
    #[inline]
    pub fn from_slice(slice: &'a [A]) -> Self;
}

// === ViewRepr: Storage / RawStorage impls ===

impl<'a, A> Storage for ViewRepr<'a, A> {
    type Element = A;
    type Device = Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A;

    #[inline]
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool;
}

impl<'a, A> RawStorage for ViewRepr<'a, A> {}

// Note: ViewRepr does NOT implement StorageMut — it is read-only.

// === ViewRepr: Clone (O(1) metadata copy) ===

impl<'a, A> Clone for ViewRepr<'a, A> {
    /// Clones the view metadata (pointer + length), not the underlying data.
    /// Complexity: O(1).
    #[inline]
    fn clone(&self) -> Self;
}

// === ViewRepr: Copy ===

impl<'a, A> Copy for ViewRepr<'a, A> {}

// === ViewRepr: Send / Sync ===

// SAFETY: ViewRepr<'a, A> is essentially &'a [A]. It is Send/Sync
// if A is Send/Sync, following the same rules as shared references.
unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}
unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}
```

### 4.8 ViewMutRepr<&'a mut A> 结构体

独占可变借用存储。零分配，**不可克隆**。

```rust
/// Exclusive mutable borrow storage (read-write view).
///
/// Holds a mutable borrowed reference to data owned by another storage.
/// Creating a mutable view is O(1) — no memory allocation occurs.
///
/// # Clone Semantics
///
/// This type is **NOT cloneable**. The exclusive borrow (`&mut`) guarantees
/// that no other code can read or write the data simultaneously, which is
/// critical for safe mutation.
///
/// # Lifetime
///
/// The `'a` lifetime parameter ensures the mutable view cannot outlive
/// the source data. Rust's borrow checker ensures no other references
/// exist for the lifetime `'a`.
pub struct ViewMutRepr<'a, A> {
    ptr: *mut A,
    len: usize,
    _marker: PhantomData<&'a mut A>,
}

// === ViewMutRepr: constructors ===

impl<'a, A> ViewMutRepr<'a, A> {
    /// Creates a new mutable view from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` is valid for `len` elements for the lifetime `'a`.
    /// - No other references (shared or mutable) to the same memory
    ///   exist for the lifetime `'a`.
    pub unsafe fn from_raw_parts(ptr: *mut A, len: usize) -> Self;

    /// Creates a new mutable view from a mutable slice.
    #[inline]
    pub fn from_slice(slice: &'a mut [A]) -> Self;

    /// Reborrows this mutable view as an immutable view.
    ///
    /// This is useful for passing a `ViewMutRepr` to functions that
    /// only need read access, without consuming the mutable borrow.
    #[inline]
    pub fn reborrow(&self) -> ViewRepr<'a, A>;
}

// === ViewMutRepr: Storage / RawStorage / StorageMut impls ===

impl<'a, A> Storage for ViewMutRepr<'a, A> {
    type Element = A;
    type Device = Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A;

    #[inline]
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool;
}

impl<'a, A> RawStorage for ViewMutRepr<'a, A> {}

impl<'a, A> StorageMut for ViewMutRepr<'a, A> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut A;
}

// Note: ViewMutRepr does NOT implement Clone — it is exclusively borrowed.
// Note: ViewMutRepr does NOT implement Copy.

// === ViewMutRepr: Send / Sync ===

// SAFETY: ViewMutRepr<'a, A> is essentially &'a mut [A].
// It is Send if A: Send (mutable access can transfer between threads).
// It is Sync if A: Sync (though typically &mut is not shared, but the
// Sync bound is consistent with standard library behavior).
unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A> {}
unsafe impl<'a, A: Sync> Sync for ViewMutRepr<'a, A> {}
```

### 4.9 ArcRepr<A> 结构体

共享所有权存储，通过 `Arc` 引用计数实现写时复制（COW）。

```rust
/// Shared-ownership storage backed by `Arc` with copy-on-write semantics.
///
/// Multiple tensors can share the same underlying data without copying.
/// When mutation is needed, `make_mut()` performs copy-on-write:
/// if the reference count is > 1, it deep-copies the data before
/// returning a mutable reference.
///
/// # Clone Semantics
///
/// Cloning increments the `Arc` reference count (O(1), atomic).
///
/// # Thread Safety
///
/// `ArcRepr` is `Send + Sync` when `A: Send + Sync`. The reference count
/// uses atomic operations, making it safe to share across threads.
///
/// # Examples
///
/// ```ignore
/// use Renon::storage::ArcRepr;
///
/// let s = ArcRepr::from_vec(vec![1.0, 2.0, 3.0]);
/// let s2 = s.clone(); // O(1), refcount = 2
/// let data = s2.make_mut(); // COW: deep copy, refcount back to 1
/// data[0] = 10.0;
/// ```
pub struct ArcRepr<A> {
    inner: Arc<ArcInner<A>>,
}

/// Internal storage for ArcRepr, holding the data buffer and metadata.
struct ArcInner<A> {
    ptr: NonNull<A>,
    len: usize,
    alignment: usize,
}

// === ArcRepr: constructors ===

impl<A: Element> ArcRepr<A> {
    /// Creates a new `ArcRepr` by copying elements from a slice.
    ///
    /// Allocates a 64-byte-aligned buffer and copies all elements.
    /// Initial reference count is 1.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` * `size_of::<A>()` overflows `isize`.
    pub fn from_slice(data: &[A]) -> Self;

    /// Creates a new `ArcRepr` from a `Vec<A>`.
    ///
    /// Reuses the Vec's allocation if already 64-byte aligned,
    /// otherwise allocates a new aligned buffer and copies.
    pub fn from_vec(vec: Vec<A>) -> Self;

    /// Creates a new `ArcRepr` with `len` elements initialized to `value`.
    pub fn from_elem(len: usize, value: A) -> Self;

    /// Creates a new `ArcRepr` with zero-initialized elements.
    pub fn zeros(len: usize) -> Self;
}

// === ArcRepr: make_mut (COW) ===

impl<A: Element> ArcRepr<A> {
    /// Returns a mutable slice, performing copy-on-write if needed.
    ///
    /// If the internal `Arc` reference count is 1, this is O(1) — no copy occurs.
    /// If the count is > 1, this deep-copies the data into a new 64-byte-aligned
    /// buffer, decrements the old Arc's count, and returns a mutable reference
    /// to the new buffer.
    ///
    /// # Atomicity
    ///
    /// The reference count check and the decision to copy are performed atomically.
    /// Multiple threads calling `make_mut()` concurrently on the same `ArcRepr`
    /// will not cause data races; at most one thread will get the original buffer
    /// (if it decrements from 2 to 1), and others will deep-copy.
    ///
    /// # Performance
    ///
    /// - Single owner (refcount == 1): O(1), no allocation.
    /// - Shared (refcount > 1): O(n), allocates + copies.
    ///
    /// # Returns
    ///
    /// A `&mut [A]` slice with exclusive access to the data.
    pub fn make_mut(&mut self) -> &mut [A];

    /// Returns the current `Arc` reference count.
    ///
    /// This is intended for debugging and diagnostics only.
    /// The count may change at any time in a multi-threaded context.
    pub fn refcount(&self) -> usize;

    /// Returns `true` if this is the sole owner of the data (refcount == 1).
    #[inline]
    pub fn is_unique(&self) -> bool;

    /// Attempts to unwrap the inner data, returning `Owned<A>` if this
    /// is the sole owner. Otherwise returns `Err(self)`.
    pub fn try_into_owned(self) -> core::result::Result<Owned<A>, Self>;

    /// Returns the alignment of the underlying allocation.
    #[inline]
    pub fn alignment(&self) -> usize;
}

// === ArcRepr: Storage / RawStorage impls ===

impl<A> Storage for ArcRepr<A> {
    type Element = A;
    type Device = Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A;

    #[inline]
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool;
}

impl<A> RawStorage for ArcRepr<A> {}

// Note: ArcRepr does NOT implement StorageMut directly.
// Mutation goes through `make_mut()` which handles COW.

// === ArcRepr: Clone (shallow, O(1)) ===

impl<A> Clone for ArcRepr<A> {
    /// Increments the Arc reference count. O(1).
    fn clone(&self) -> Self;
}

// === ArcRepr: Send / Sync ===

// SAFETY: ArcRepr<A> wraps Arc<ArcInner<A>>. Arc is Send + Sync when
// its contents are Send + Sync. ArcInner owns a heap buffer that is
// safe to share/send if A is.
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

### 4.10 转换 trait 实现

各存储类型之间的转换关系：

```rust
// Owned<A> → ViewRepr<&'a A>  (通过 TensorBase.view())
// Owned<A> → ViewMutRepr<&'a mut A>  (通过 TensorBase.view_mut())
// Owned<A> → ArcRepr<A>  (显式转换)
impl<A: Element> Owned<A> {
    /// Converts this owned storage into a shared `ArcRepr`.
    ///
    /// Reuses the existing allocation (wraps in Arc without copying).
    pub fn into_arc(self) -> ArcRepr<A>;
}

// ArcRepr<A> → Owned<A>  (try_into_owned 或 make_mut + 重建)
// ViewRepr / ViewMutRepr → Owned<A>  (to_owned, 深拷贝)
impl<'a, A: Element> ViewRepr<'a, A> {
    /// Creates an owned deep copy of the viewed data.
    pub fn to_owned(&self) -> Owned<A>;
}

impl<'a, A: Element> ViewMutRepr<'a, A> {
    /// Creates an owned deep copy of the viewed data.
    pub fn to_owned(&self) -> Owned<A>;
}

// Owned<A> → ViewRepr (Deref-style)
impl<A> AsRef<ViewRepr<'_, A>> for Owned<A> {
    fn as_ref(&self) -> &ViewRepr<'_, A>;
}
```

### 4.11 mod.rs 公共导出

```rust
// src/storage/mod.rs

mod owned;
mod view;
mod view_mut;
mod arc;

pub use self::owned::Owned;
pub use self::view::ViewRepr;
pub use self::view_mut::ViewMutRepr;
pub use self::arc::ArcRepr;

pub use self::device::{Device, Cpu};
pub use self::traits::{Storage, RawStorage, StorageMut};
```

---

## 5. 内部实现设计

### 5.1 64 字节对齐分配策略

Owned 和 ArcRepr 的堆分配使用 Rust 全局分配器进行自定义对齐分配。

```rust
// Internal: Aligned buffer allocation (in private module or inline)

use core::alloc::Layout;

/// Allocates a 64-byte-aligned buffer for `len` elements of type `A`.
///
/// Returns a NonNull pointer to uninitialized memory.
///
/// # Panics
///
/// Panics if `len * size_of::<A>()` overflows `isize`.
/// Panics if the allocation fails (out of memory).
unsafe fn allocate_aligned<A>(len: usize, alignment: usize) -> NonNull<A> {
    let byte_len = len.checked_mul(core::mem::size_of::<A>())
        .expect("allocation size overflow");
    let layout = Layout::from_size_align(byte_len.max(1), alignment)
        .expect("invalid layout");

    let ptr = if byte_len == 0 {
        // Zero-size allocation: use a dangling pointer with correct alignment.
        NonNull::dangling()
    } else {
        // SAFETY: layout is valid (size > 0, alignment is power of 2).
        NonNull::new(std::alloc::alloc(layout))
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
            .cast()
    };
    ptr
}

/// Deallocates a previously allocated aligned buffer.
///
/// # Safety
///
/// - `ptr` must have been returned by `allocate_aligned`.
/// - `len` and `alignment` must match the original allocation.
unsafe fn deallocate_aligned<A>(ptr: NonNull<A>, len: usize, alignment: usize) {
    let byte_len = len * core::mem::size_of::<A>();
    if byte_len > 0 {
        let layout = Layout::from_size_align_unchecked(byte_len, alignment);
        std::alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
    }
}
```

**关键设计决策**：

| 决策 | 理由 |
|------|------|
| 不使用 `Vec<A>` 作为内部存储 | Vec 不保证 64 字节对齐 |
| 使用 `NonNull<A>` + 手动分配 | 完全控制对齐和布局 |
| `from_vec` 检测现有对齐 | 当 Vec 恰好满足对齐要求时避免不必要的拷贝 |
| 零长度分配返回 dangling ptr | 避免实际分配零字节，与标准库 Layout 实践一致 |

### 5.2 Arc COW 实现细节

`ArcRepr::make_mut()` 的实现策略：

```
make_mut() 调用流程:
│
├── Arc::get_mut(&mut self.inner)
│   ├── Some(inner)  →  refcount == 1
│   │   └── 直接返回 &mut slice，无需拷贝
│   │
│   └── None  →  refcount > 1 或跨线程共享
│       └── 深拷贝:
│           1. 分配新的 64 字节对齐缓冲区
│           2. 从旧缓冲区拷贝所有元素
│           3. 替换 self.inner = Arc::new(new_inner)
│           4. 旧 Arc 引用计数自动递减
│           5. 返回 &mut slice
```

```rust
impl<A: Element> ArcRepr<A> {
    pub fn make_mut(&mut self) -> &mut [A] {
        // Fast path: if we're the sole owner, Arc::get_mut succeeds.
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            // SAFETY: we have exclusive access; ptr is valid for len elements.
            return unsafe {
                core::slice::from_raw_parts_mut(inner.ptr.as_ptr(), inner.len)
            };
        }

        // Slow path: deep copy needed (COW).
        let old = &*self.inner;
        let new_owned = unsafe {
            // SAFETY: old.ptr is valid for old.len elements.
            let src = core::slice::from_raw_parts(old.ptr.as_ptr(), old.len);
            Owned::from_slice(src)
        };

        // Replace the Arc with a new one (old Arc refcount decrements on drop).
        self.inner = Arc::new(ArcInner {
            ptr: new_owned.ptr,   // Takes ownership of the allocation.
            len: new_owned.len,
            alignment: new_owned.alignment,
        });

        // Prevent double-free: forget the Owned wrapper since ArcInner now owns the buffer.
        core::mem::forget(new_owned);

        // SAFETY: we just created a unique Arc; ptr is valid for len elements.
        unsafe {
            core::slice::from_raw_parts_mut(self.inner.ptr.as_ptr(), self.inner.len)
        }
    }
}
```

> **原子性说明**：`Arc::get_mut` 内部使用 `Arc::strong_count` 检查引用计数是否为 1。如果多线程并发调用 `make_mut()`，只有一个线程能成功获得 `get_mut`（因为引用计数 > 1 后都不成功），其他线程走深拷贝路径。这保证了不会出现数据竞争。

### 5.3 no_std 注意事项

```rust
// Conditional compilation strategy:
//
// std feature (default):
//   - use std::sync::Arc
//   - use std::alloc::alloc / dealloc
//   - use std::vec::Vec
//
// no_std mode:
//   - use alloc::sync::Arc
//   - use alloc::alloc::alloc / dealloc
//   - use alloc::vec::Vec
//
// All core logic uses core:: primitives:
//   - core::ptr::NonNull
//   - core::marker::PhantomData
//   - core::mem
//   - core::slice
```

Owned 内部不使用 `Vec<A>` 作为存储，因此不依赖 `Vec` 的对齐行为。仅 `from_vec` 和 `into_vec` 转换方法需要 `Vec`。

### 5.4 View 生命周期管理

视图类型的生命周期遵循 Rust 标准借用规则：

| 场景 | 生命周期关系 |
|------|-------------|
| `Owned<A>` → `ViewRepr<'a, A>` | `'a` 绑定到 Owned 的借用持续时间 |
| `Owned<A>` → `ViewMutRepr<'a, A>` | `'a` 绑定到 Owned 的可变借用持续时间 |
| `ViewRepr<'a, A>` → `ViewRepr<'b, A>` | `'b: 'a`（视图的视图不能超过原视图） |
| `ArcRepr<A>` → `ViewRepr<'a, A>` | `'a` 绑定到 ArcRepr 的借用持续时间 |

**关键约束**：
- `ViewMutRepr` 不可克隆，确保同一时间只有一个可变引用
- `ViewRepr` 实现 `Copy`，允许自由复制（仅复制元数据）
- 视图不管理内存生命周期，只持有指针 + 长度 + 生命周期标记

### 5.5 Owned Drop 实现

```rust
impl<A> Drop for Owned<A> {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr is valid for len elements, allocated with self.alignment.
            unsafe {
                // Drop elements first (if A has a non-trivial destructor).
                let slice = core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len);
                core::ptr::drop_in_place(slice);
                // Then deallocate the buffer.
                deallocate_aligned::<A>(self.ptr, self.len, self.alignment);
            }
        }
    }
}
```

对于 `Element: Copy` 类型，`drop_in_place` 是空操作，编译器会优化掉。但保留此调用确保未来如果支持非 Copy 元素类型时仍然正确。

---

## 6. 实现任务拆分

每个任务约 10 分钟，可独立提交。

### Wave 1: 基础设施

- [ ] **Task 1**: 创建 `src/storage/mod.rs` 骨架
  - 文件: `src/storage/mod.rs`
  - 内容: 模块声明、公共导出、占位 `use` 语句
  - 测试: 编译通过
  - 前置: 无
  - 预计: 5 min

- [ ] **Task 2**: 实现 Device trait 和 Cpu 类型
  - 文件: `src/storage/mod.rs`（或 `src/storage/device.rs`）
  - 内容: `Device` trait 定义、`Cpu` 结构体、`allocate` / `deallocate` 方法
  - 测试: `test_cpu_allocate_and_deallocate`, `test_cpu_zero_len_allocation`
  - 前置: Task 1
  - 预计: 10 min

### Wave 2: 核心 trait

- [ ] **Task 3**: 实现 Storage / RawStorage / StorageMut trait 定义
  - 文件: `src/storage/mod.rs`（或 `src/storage/traits.rs`）
  - 内容: 三个 trait 的完整定义，包含 doc comments
  - 测试: 编译通过（trait 定义无单元测试）
  - 前置: Task 2
  - 预计: 10 min

### Wave 3: Owned<A>

- [ ] **Task 4**: 实现 `Owned<A>` 结构体 + 基本构造函数
  - 文件: `src/storage/owned.rs`
  - 内容: struct 定义、`from_slice`、`zeros`、`uninitialized`
  - 测试: `test_owned_from_slice`, `test_owned_zeros`, `test_owned_len_is_empty`
  - 前置: Task 3
  - 预计: 10 min

- [ ] **Task 5**: 实现 `Owned<A>` 的 `from_vec` 和 `into_vec`
  - 文件: `src/storage/owned.rs`
  - 内容: Vec 对齐检测、重新分配或复用逻辑
  - 测试: `test_owned_from_vec_aligned`, `test_owned_from_vec_unaligned`, `test_owned_into_vec_roundtrip`
  - 前置: Task 4
  - 预计: 10 min

- [ ] **Task 6**: 实现 `Owned<A>` 的 Storage / RawStorage / StorageMut trait
  - 文件: `src/storage/owned.rs`
  - 内容: as_ptr, len, is_empty, as_mut_ptr, as_slice, as_mut_slice
  - 测试: `test_owned_storage_trait_methods`, `test_owned_storage_mut_write`
  - 前置: Task 4
  - 预计: 10 min

- [ ] **Task 7**: 实现 `Owned<A>` 的 Clone / Drop / Send / Sync
  - 文件: `src/storage/owned.rs`
  - 内容: 深拷贝 Clone、带 drop_in_place 的 Drop、unsafe Send/Sync
  - 测试: `test_owned_clone_is_deep_copy`, `test_owned_drop_no_leak`, `test_owned_send_sync`
  - 前置: Task 6
  - 预计: 10 min

### Wave 4: ViewRepr

- [ ] **Task 8**: 实现 `ViewRepr<&'a A>` 完整实现
  - 文件: `src/storage/view.rs`
  - 内容: struct、from_raw_parts、from_slice、Storage/RawStorage impl、Clone/Copy
  - 测试: `test_view_from_slice`, `test_view_clone_is_shallow`, `test_view_read_only`
  - 前置: Task 3
  - 预计: 10 min

### Wave 5: ViewMutRepr

- [ ] **Task 9**: 实现 `ViewMutRepr<&'a mut A>` 完整实现
  - 文件: `src/storage/view_mut.rs`
  - 内容: struct、from_raw_parts、from_slice、Storage/RawStorage/StorageMut impl、reborrow
  - 测试: `test_view_mut_write`, `test_view_mut_not_cloneable`（编译失败测试）, `test_view_mut_reborrow`
  - 前置: Task 3
  - 预计: 10 min

### Wave 6: ArcRepr

- [ ] **Task 10**: 实现 `ArcRepr<A>` 结构体 + 基本构造函数 + Clone
  - 文件: `src/storage/arc.rs`
  - 内容: struct、ArcInner、from_slice、from_vec、zeros、Clone
  - 测试: `test_arc_from_slice`, `test_arc_clone_shallow`, `test_arc_refcount`
  - 前置: Task 4（复用对齐分配逻辑）
  - 预计: 10 min

- [ ] **Task 11**: 实现 `ArcRepr::make_mut()` COW 机制
  - 文件: `src/storage/arc.rs`
  - 内容: make_mut 方法、引用计数判断、深拷贝逻辑
  - 测试: `test_arc_make_mut_unique_no_copy`, `test_arc_make_mut_shared_copies`, `test_arc_make_mut_isolation`
  - 前置: Task 10
  - 预计: 10 min

- [ ] **Task 12**: 实现 `ArcRepr` Storage/RawStorage trait + 转换方法
  - 文件: `src/storage/arc.rs`
  - 内容: Storage impl、try_into_owned、into_arc（在 Owned 上）、is_unique
  - 测试: `test_arc_storage_trait`, `test_arc_try_into_owned`, `test_owned_into_arc`
  - 前置: Task 10
  - 预计: 10 min

### Wave 7: 集成

- [ ] **Task 13**: 整合 mod.rs 公共导出 + 对齐分配器提取到 private 模块
  - 文件: `src/storage/mod.rs`, `src/private/alloc.rs`
  - 内容: 统一导出、将 `allocate_aligned` / `deallocate_aligned` 移至 private::alloc
  - 测试: `test_public_imports_complete`, `test_cross_storage_conversions`
  - 前置: Task 7, 8, 9, 12
  - 预计: 10 min

---

## 7. 测试计划

### 7.1 单元测试

| 测试类别 | 测试项 | 位置 |
|----------|--------|------|
| **Owned 构造** | `from_slice` 正确拷贝 | `owned.rs::tests` |
| | `from_vec` 复用/重新分配 | `owned.rs::tests` |
| | `zeros` 全部为零 | `owned.rs::tests` |
| | `uninitialized` 长度正确 | `owned.rs::tests` |
| | `from_elem` 填充正确 | `owned.rs::tests` |
| **Owned 内存** | 64 字节对齐验证 | `owned.rs::tests` |
| | 零长度分配不 panic | `owned.rs::tests` |
| | Drop 后无泄漏（集成 valgrind/miri） | `owned.rs::tests` |
| **Owned Clone** | 深拷贝隔离性 | `owned.rs::tests` |
| | 修改克隆不影响原件 | `owned.rs::tests` |
| **ViewRepr** | 指针指向源数据 | `view.rs::tests` |
| | Clone 是浅拷贝 | `view.rs::tests` |
| | Copy trait 可用 | `view.rs::tests` |
| | 生命周期正确（编译时验证） | `view.rs::tests` |
| **ViewMutRepr** | 可写入数据 | `view_mut.rs::tests` |
| | 不可 Clone（编译失败测试） | `tests/ui/` |
| | reborrow 返回只读视图 | `view_mut.rs::tests` |
| | 写入通过源数据可见 | `view_mut.rs::tests` |
| **ArcRepr 基本** | from_vec 构造 | `arc.rs::tests` |
| | Clone 增加引用计数 | `arc.rs::tests` |
| | is_unique 判断正确 | `arc.rs::tests` |
| **ArcRepr COW** | make_mut 唯一不拷贝 | `arc.rs::tests` |
| | make_mut 共享时深拷贝 | `arc.rs::tests` |
| | 拷贝后数据隔离 | `arc.rs::tests` |
| | 64 字节对齐保持 | `arc.rs::tests` |
| **转换** | Owned → ArcRepr | 集成测试 |
| | ArcRepr → Owned（try_into_owned） | 集成测试 |
| | View → Owned（to_owned） | 集成测试 |
| **Send/Sync** | Owned\<f64\>: Send + Sync | 编译时验证 |
| | ViewRepr: Send + Sync | 编译时验证 |
| | ArcRepr: Send + Sync | 编译时验证 |

### 7.2 集成测试

| 文件 | 测试内容 |
|------|----------|
| `tests/storage_roundtrip.rs` | 所有存储模式之间的转换完整性 |
| `tests/storage_alignment.rs` | 各存储类型的实际对齐验证 |
| `tests/storage_thread_safety.rs` | ArcRepr 多线程 make_mut 安全性 |

### 7.3 边界测试

| 场景 | 预期行为 |
|------|----------|
| 零长度存储 | `len() == 0`, `is_empty() == true`, `as_slice()` 返回空切片 |
| 单元素存储 | 正常工作，对齐仍为 64 字节 |
| 大数组 (> 1GB) | 不 panic，正确分配 |
| 非常大 len（接近 `isize::MAX`） | panic 并报告 overflow |
| make_mut 在空 ArcRepr 上 | 返回空 `&mut []`，不分配 |

### 7.4 属性测试

| 不变量 | 测试方法 |
|--------|----------|
| `Owned::from_slice(data).to_vec() == data` | 随机 `Vec<f64>` |
| `ArcRepr::clone().refcount() == 原refcount + 1` | 多次 clone |
| `make_mut` 后数据与克隆前一致 | 随机数据 + 随机 clone 次数 |
| 对齐始终是 64 的倍数 | 所有构造方式 |

### 7.5 编译时测试（ui tests）

```
tests/ui/
├── view_mut_not_cloneable.rs      // 验证 ViewMutRepr 不能 clone
├── view_not_storage_mut.rs         // 验证 ViewRepr 不实现 StorageMut
├── arc_not_storage_mut.rs          // 验证 ArcRepr 不直接实现 StorageMut
└── view_mut_not_copy.rs            // 验证 ViewMutRepr 不能 Copy
```

---

## 附录 A: Storage trait 层次图

```
                    ┌──────────┐
                    │ Storage  │  (as_ptr, len, is_empty)
                    │          │  type Element, type Device
                    └────┬─────┘
                         │
                ┌────────┴────────┐
                │                 │
         ┌──────┴──────┐   ┌─────┴──────┐
         │ RawStorage  │   │            │
         │ (as_slice)  │   │            │
         └──────┬──────┘   │            │
                │          │            │
         ┌──────┴──────┐   │            │
         │ StorageMut  │   │            │
         │(as_mut_ptr, │   │            │
         │ as_mut_slice│   │            │
         └─────────────┘   │            │
                           │            │
    实现矩阵:
    ┌─────────────┬──────────┬──────────┬─────────────┐
    │             │ Storage  │RawStorag │ StorageMut  │
    ├─────────────┼──────────┼──────────┼─────────────┤
    │ Owned<A>    │    ✅    │    ✅    │     ✅      │
    │ ViewRepr    │    ✅    │    ✅    │     ❌      │
    │ ViewMutRepr │    ✅    │    ✅    │     ✅      │
    │ ArcRepr<A>  │    ✅    │    ✅    │     ❌*     │
    └─────────────┴──────────┴──────────┴─────────────┘

    * ArcRepr mutation via make_mut(), not through StorageMut trait.
```

## 附录 B: 克隆语义对比

```
    Owned<A>::clone()
    ├── 分配新的 64 字节对齐缓冲区
    ├── 拷贝全部元素
    └── 返回独立副本
    复杂度: O(n)

    ViewRepr::clone() / Copy
    ├── 拷贝指针 + 长度
    └── 返回新的视图元数据
    复杂度: O(1)

    ViewMutRepr
    └── 不可克隆（独占语义）

    ArcRepr<A>::clone()
    ├── Arc 引用计数 +1（原子操作）
    └── 返回共享同一数据的句柄
    复杂度: O(1)
```

## 附录 C: 内存布局示意

```
    Owned<A> (len=4):
    ┌───────────────────────────────────────────────────────────────────┐
    │  64-byte aligned heap                                            │
    │  ┌──────┬──────┬──────┬──────┬ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ padding ─ ─│
    │  │ A[0] │ A[1] │ A[2] │ A[3] │                                 │
    │  └──────┴──────┴──────┴──────┴                                  │
    │  ^                                                              │
    │  ptr (NonNull<A>)                                               │
    └───────────────────────────────────────────────────────────────────┘

    ViewRepr<'a, A> (len=4):
    ┌──────────────┐       ┌──────────────────────────────┐
    │ ViewRepr     │       │  Source data (borrowed)       │
    │ ptr ─────────┼──────→│  [A[0], A[1], A[2], A[3]]   │
    │ len=4        │       │  (owned by Owned/ArcRepr)     │
    │ PhantomData  │       └──────────────────────────────┘
    └──────────────┘

    ArcRepr<A> (len=4, refcount=2):
    ┌──────────────┐       ┌──────────────┐
    │ ArcRepr #1   │       │ ArcRepr #2   │
    │ inner ──┐    │       │ inner ──┐    │
    └─────────┼────┘       └─────────┼────┘
              │                      │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────┐
              │ Arc<ArcInner<A>> │
              │ strong=2         │
              │  ┌─────────────┐│
              │  │ ptr, len=4, ││
              │  │ alignment   ││
              │  └──────┬──────┘│
              └─────────┼────────┘
                        ▼
              ┌─────────────────────────────────┐
              │  64-byte aligned buffer          │
              │  [A[0], A[1], A[2], A[3]]        │
              └─────────────────────────────────┘
```
