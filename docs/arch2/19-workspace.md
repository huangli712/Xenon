# 工作空间模块设计

> 文档编号: 19 | 模块: `src/workspace.rs` | 阶段: Phase 3
> 前置文档: `00-rust-standards.md`, `01-architecture-overview.md`, `05-storage.md`, `07-tensor-core.md`
> 需求依据: 需求说明书 §15（临时工作空间）

---

## 1. 模块定位

`workspace` 模块是 Xenon 的**临时缓冲区管理基础设施**，为需要中间存储的运算（矩阵乘法转置缓冲区、排序临时空间、einsum 中间张量等）提供统一的内存分配与复用机制。

**核心问题：** 许多数值运算在执行过程中需要临时缓冲区（scratch space），例如：
- `tensordot` / `batch_matmul` 需要转置缓冲区
- `einsum` 多步缩并需要中间张量存储
- 未来排序操作需要临时工作数组
- 上游 BLAS 库可能需要预分配工作空间

若每次运算都临时分配/释放堆内存，将产生大量 `malloc` / `free` 系统调用，严重影响性能。Workspace 模块通过**缓冲区复用**策略解决此问题。

**核心设计目标：**

| 目标 | 体现 |
|------|------|
| 缓冲区复用 | 同一 Workspace 可连续借出多次，归还后复用底层缓冲区 |
| 对齐保证 | 默认 64 字节对齐，支持自定义更大对齐 |
| 零初始化 | 借出的缓冲区不保证零初始化（性能优先），调用方须自行初始化 |
| 分割支持 | 支持 `split_at` 将工作空间递归二分为不重叠子空间 |
| Scratch 查询 | 提供 `ScratchNeed` trait 让操作声明其临时内存需求 |
| 零运行时开销 | Scratch 查询为纯计算，不分配内存 |
| 独立模块 | 无核心模块依赖，可独立实现和测试 |

**本模块职责边界：**

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 缓冲区管理 | Workspace 分配、借出、归还、扩容 | 运算逻辑 |
| Scratch 查询 | `ScratchNeed` trait、`ScratchReq` 需求描述类型 | 具体操作的 scratch 计算 |
| 分割 | `split_at` 递归二分 | 线程池管理 |
| 对齐 | 64 字节默认对齐 + 自定义对齐 | SIMD 调度 |

---

## 2. 文件位置

```
src/
    workspace.rs            # 本模块：Workspace, ScratchReq, ScratchNeed trait
    lib.rs                  # pub mod workspace; + re-export
```

单文件设计理由：Workspace 是高度内聚的工具模块，API 表面积小（~400 行），拆分无收益。

`src/lib.rs` 中的集成：

```rust
pub mod workspace;

pub use crate::workspace::{Workspace, ScratchReq, ScratchNeed};
```

---

## 3. 依赖关系

### 3.1 上游依赖（本模块需要）

| 依赖 | 来源 | 用途 |
|------|------|------|
| `core::ptr::NonNull` | `core` | 对齐缓冲区指针 |
| `core::alloc::Layout` | `core` | 自定义对齐分配 |
| `core::mem::MaybeUninit` | `core` | 零初始化缓冲区表示 |
| `core::marker::PhantomData` | `core` | 生命周期标记 |

### 3.2 下游消费者

| 消费者 | 使用方式 |
|--------|----------|
| `ops/matrix.rs` | `tensordot`、`batch_matmul`、`einsum` 通过 Workspace 获取转置缓冲区 |
| `ops/set_ops.rs` | 排序、唯一化等操作的临时工作数组 |
| 上游 BLAS 库 | 通过 FFI 指针 API 从 Workspace 获取对齐工作缓冲区 |
| 用户代码 | 预分配 Workspace 后传入运算函数，避免重复分配 |

### 3.3 不依赖任何 Xenon 核心模块

```
workspace.rs 不依赖:
    ❌ crate::tensor
    ❌ crate::storage
    ❌ crate::dimension
    ❌ crate::element
    ❌ crate::layout
    ❌ crate::error

仅依赖:
    ✅ core::ptr, core::alloc, core::mem, core::marker
    ✅ alloc::alloc (feature = "std" 时使用 std::alloc)
```

这保证了 Workspace 可以在 Phase 2 核心模块完成后立即实现，也使其可供上游库独立使用。

---

## 4. 公共 API 设计

### 4.1 ScratchReq — 临时内存需求描述

```rust
/// Describes the scratch memory requirement of an operation.
///
/// `ScratchReq` is a structured description of how much temporary memory
/// an operation needs, including size and alignment constraints. It supports
/// composition: multiple requirements can be merged (max for sequential,
/// sum for parallel).
///
/// # Composition Rules
///
/// - **Sequential operations**: take the maximum (`a.max(b)`)
/// - **Parallel operations**: take the sum (`a + b`)
///
/// # Examples
///
/// ```
/// use xenon::workspace::ScratchReq;
///
/// let a = ScratchReq::new(1024, 64);
/// let b = ScratchReq::new(2048, 64);
///
/// // Sequential: reuse the same buffer
/// let seq = a.max(b);
/// assert_eq!(seq.size(), 2048);
///
/// // Parallel: need both simultaneously
/// let par = a + b;
/// assert_eq!(par.size(), 3072);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchReq {
    /// Required size in bytes.
    size: usize,

    /// Required alignment in bytes (must be a power of 2).
    alignment: usize,
}
```

**ScratchReq 方法：**

```rust
impl ScratchReq {
    /// Creates a new scratch requirement with the given size and alignment.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is not a power of 2 or if `alignment` is 0.
    #[inline]
    pub const fn new(size: usize, alignment: usize) -> Self;

    /// Creates a new scratch requirement for `count` elements of type `A`
    /// with default 64-byte alignment.
    ///
    /// Equivalent to `ScratchReq::new(count * size_of::<A>(), 64)`.
    ///
    /// # Panics
    ///
    /// Panics if `count * size_of::<A>()` overflows.
    #[inline]
    pub fn for_elements<A>(count: usize) -> Self;

    /// Creates a scratch requirement for `count` elements of type `A`
    /// with a custom alignment.
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is not a power of 2 or if `count * size_of::<A>()` overflows.
    #[inline]
    pub fn for_elements_aligned<A>(count: usize, alignment: usize) -> Self;

    /// Returns the required size in bytes.
    #[inline]
    pub const fn size(&self) -> usize;

    /// Returns the required alignment in bytes.
    #[inline]
    pub const fn alignment(&self) -> usize;

    /// Returns a zero-size requirement (no scratch needed).
    #[inline]
    pub const fn none() -> Self;

    /// Returns `true` if no scratch memory is needed (size == 0).
    #[inline]
    pub const fn is_empty(&self) -> bool;

    /// Merges two requirements for **sequential** operations.
    ///
    /// Returns a requirement with `max(a.size, b.size)` and `max(a.alignment, b.alignment)`.
    /// Use this when operations run one after another and can reuse the same buffer.
    #[inline]
    pub const fn max(self, other: Self) -> Self;

    /// Merges two requirements for **parallel** operations.
    ///
    /// Returns a requirement with `a.size + b.size` (with padding for alignment)
    /// and `max(a.alignment, b.alignment)`. Use this when both buffers are needed
    /// simultaneously.
    ///
    /// The combined size accounts for alignment padding between the two allocations.
    #[inline]
    pub fn parallel_sum(self, other: Self) -> Self;
}

/// Addition is an alias for `parallel_sum` (parallel composition).
impl core::ops::Add for ScratchReq {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        self.parallel_sum(other)
    }
}

/// Add-assign for parallel composition.
impl core::ops::AddAssign for ScratchReq {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = self.parallel_sum(other);
    }
}
```

### 4.2 ScratchNeed trait — 操作声明临时内存需求

```rust
/// Trait for operations that can declare their scratch memory requirements.
///
/// Upstream libraries (e.g., linear algebra libraries) implement this trait
/// for their operation types. Xenon provides the infrastructure; the upstream
/// library defines the specific queries.
///
/// # Design Rationale
///
/// Scratch size queries are pure computations — no memory allocation occurs.
/// This enables callers to:
/// 1. Query the scratch requirement before allocating
/// 2. Pre-allocate or reuse a single `Workspace` across multiple operations
/// 3. Compose multiple operations' requirements (sequential or parallel)
///
/// # Examples
///
/// ```
/// use xenon::workspace::{ScratchNeed, ScratchReq};
///
/// struct MatMulOp { m: usize, n: usize, k: usize }
///
/// impl ScratchNeed for MatMulOp {
///     fn scratch_req(&self) -> ScratchReq {
///         // Needs a transpose buffer for the larger input
///         let max_elems = (self.m * self.k).max(self.k * self.n);
///         ScratchReq::for_elements::<f64>(max_elems)
///     }
/// }
/// ```
pub trait ScratchNeed {
    /// Returns the scratch memory requirement for this operation.
    ///
    /// This is a pure computation with zero runtime allocation overhead.
    fn scratch_req(&self) -> ScratchReq;
}
```

### 4.3 Workspace — 临时缓冲区容器

```rust
/// A reusable temporary buffer container for scratch space allocation.
///
/// `Workspace` manages a single underlying byte buffer that can be borrowed
/// out as typed slices. After a borrow is dropped, the buffer is available
/// for reuse — avoiding repeated heap allocations.
///
/// # Memory Properties
///
/// - **Default alignment**: 64 bytes (AVX-512 cache line friendly)
/// - **Custom alignment**: configurable via `with_alignment()`
/// - **No zero-initialization**: borrowed slices contain uninitialized data
///   for performance; callers must initialize before reading
/// - **Growth-only**: capacity increases on demand, never shrinks
///
/// # Borrow Semantics
///
/// The workspace uses a cursor-based lending model:
/// - Each `borrow` / `borrow_mut` call advances an internal cursor
/// - Multiple non-overlapping borrows can coexist (nested borrowing)
/// - The cursor resets when all borrows are dropped
///
/// # Thread Safety
///
/// `Workspace` is `Send` but not `Sync`. Thread safety is the caller's
/// responsibility via Rust's borrow checker. A `&mut Workspace` can be
/// passed across threads, but shared references cannot be used to borrow
/// mutable slices.
///
/// # Examples
///
/// ```
/// use xenon::workspace::{Workspace, ScratchReq};
///
/// // Create a workspace with 1 KB capacity
/// let mut ws = Workspace::new(1024);
///
/// // Borrow a typed slice for temporary computation
/// let buf: &mut [f64] = ws.borrow_mut(128);
/// // ... use buf for computation ...
/// // buf is dropped, workspace can reuse the space
///
/// // Reserve enough for a known scratch requirement
/// let req = ScratchReq::for_elements::<f64>(256);
/// ws.reserve(req);
/// ```
pub struct Workspace {
    /// Underlying byte buffer (64-byte aligned).
    buffer: NonNull<u8>,

    /// Total capacity of the buffer in bytes.
    capacity: usize,

    /// Current borrow cursor position in bytes.
    /// When 0, no borrows are active.
    cursor: usize,

    /// Alignment of this workspace's buffer in bytes.
    alignment: usize,
}
```

#### 4.3.1 构造与销毁

```rust
impl Workspace {
    /// Creates a new workspace with the given initial capacity in bytes.
    ///
    /// The buffer is allocated with 64-byte alignment. If `capacity` is 0,
    /// no allocation occurs; the first borrow will trigger allocation.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` overflows `isize` when rounded up to alignment.
    pub fn new(capacity: usize) -> Self;

    /// Creates a new workspace with a custom buffer alignment.
    ///
    /// Use this when operations require alignment greater than 64 bytes
    /// (e.g., 128-byte for certain AVX-512 extended instructions).
    ///
    /// # Panics
    ///
    /// Panics if `alignment` is not a power of 2 or if `alignment` is 0.
    pub fn with_alignment(capacity: usize, alignment: usize) -> Self;

    /// Creates a workspace from a pre-allocated buffer.
    ///
    /// Takes ownership of the provided buffer. The caller must ensure the
    /// buffer is properly aligned.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` is properly aligned for the given `alignment`
    /// - `ptr` is valid for `capacity` bytes
    /// - The memory was allocated with a compatible allocator and can be
    ///   freed with `dealloc(ptr, Layout::from_size_align(capacity, alignment))`
    pub unsafe fn from_raw_parts(ptr: NonNull<u8>, capacity: usize, alignment: usize) -> Self;

    /// Creates an empty workspace with no initial allocation.
    ///
    /// The first borrow will allocate the required capacity.
    #[inline]
    pub fn empty() -> Self {
        Self::new(0)
    }
}

impl Drop for Workspace {
    /// Deallocates the internal buffer.
    ///
    /// # Panics
    ///
    /// Panics on drop if there are outstanding borrows (this should not
    /// happen under correct usage since borrows hold `&mut self`).
    fn drop(&mut self);
}
```

#### 4.3.2 容量查询与预留

```rust
impl Workspace {
    /// Returns the total buffer capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize;

    /// Returns the number of currently used bytes (cursor position).
    #[inline]
    pub fn used(&self) -> usize;

    /// Returns the number of available bytes for new borrows.
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity - self.cursor
    }

    /// Returns the alignment of this workspace's buffer in bytes.
    #[inline]
    pub fn alignment(&self) -> usize;

    /// Ensures the workspace has at least `additional` bytes of free capacity.
    ///
    /// If the current capacity is insufficient, reallocates the buffer to
    /// `max(current * 2, cursor + additional)` bytes, preserving the alignment.
    /// Existing data in the buffer is **not** preserved (since all data is
    /// temporary by definition).
    ///
    /// # Panics
    ///
    /// Panics if the reallocation size overflows `isize`.
    pub fn reserve(&mut self, additional: usize);

    /// Ensures the workspace can satisfy the given scratch requirement.
    ///
    /// Equivalent to calling `reserve(req.size())` after verifying the
    /// alignment requirement is met. If the workspace's alignment is
    /// lower than required, panics (a new workspace with correct alignment
    /// must be created instead).
    ///
    /// # Panics
    ///
    /// Panics if `req.alignment() > self.alignment()`.
    pub fn reserve_for(&mut self, req: ScratchReq);
}
```

#### 4.3.3 借出（类型化切片访问）

```rust
impl Workspace {
    /// Borrows a mutable typed slice of `count` elements from the workspace.
    ///
    /// The returned slice is aligned to `max(self.alignment, align_of::<A>())`
    /// bytes. The memory is **not zero-initialized** — the caller must
    /// initialize all elements before reading.
    ///
    /// This advances the internal cursor by `count * size_of::<A>()` bytes
    /// (rounded up to alignment). The cursor resets when the returned
    /// `WorkspaceBorrow` is dropped.
    ///
    /// If the workspace does not have enough capacity, it will automatically
    /// grow (reallocate) to accommodate the request.
    ///
    /// # Type Parameters
    ///
    /// - `A`: Element type for the borrowed slice.
    ///
    /// # Arguments
    ///
    /// - `count`: Number of elements to borrow.
    ///
    /// # Panics
    ///
    /// Panics if `count * size_of::<A>()` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::workspace::Workspace;
    ///
    /// let mut ws = Workspace::new(1024);
    /// let buf: &mut [f64] = ws.borrow_mut(100);
    /// // buf is 100 uninitialized f64s
    /// for elem in buf.iter_mut() {
    ///     *elem = 0.0; // initialize before use
    /// }
    /// ```
    pub fn borrow_mut<A>(&mut self, count: usize) -> &mut [MaybeUninit<A>] {
        // Implementation:
        // 1. Compute byte_size = count * size_of::<A>()
        // 2. Align cursor forward to satisfy align_of::<A>()
        // 3. If cursor + byte_size > capacity, grow
        // 4. Return slice from cursor position
        // 5. Advance cursor
        ...
    }

    /// Borrows a mutable typed slice, zero-initializing it.
    ///
    /// Same as `borrow_mut`, but writes zeroes to all bytes before returning.
    /// Use this when zero-initialization is required (e.g., accumulation buffers).
    ///
    /// # Performance
    ///
    /// This is slower than `borrow_mut` due to the zeroing pass. Only use
    /// when the overhead is justified.
    pub fn borrow_mut_zeroed<A>(&mut self, count: usize) -> &mut [A]
    where
        A: Default + Copy,
    {
        // Same as borrow_mut, then zero-fill via ptr::write_bytes
        ...
    }

    /// Resets the borrow cursor to zero, allowing reuse of the entire buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no previously borrowed slices are still
    /// in use. In practice, this is guaranteed by Rust's borrow checker
    /// since all borrow methods take `&mut self`.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
```

> **设计决策 — `MaybeUninit<A>` vs `A`：** `borrow_mut` 返回 `&mut [MaybeUninit<A>]` 而非 `&mut [A]`，因为工作空间缓冲区不保证初始化。调用方必须显式初始化后才能通过 `.assume_init_mut()` 转换为 `&mut [A]`。这是一个安全的 API，防止读取未初始化内存。对于已知需要零初始化的场景，提供 `borrow_mut_zeroed` 便捷方法。

#### 4.3.4 分割（递归二分）

```rust
impl Workspace {
    /// Splits the remaining available space at the given byte offset.
    ///
    /// Returns two independent sub-workspaces that can be borrowed from
    /// concurrently without overlapping. This is O(1) — no memory allocation.
    ///
    /// Use this for parallel operations where two computations need
    /// independent scratch buffers from the same allocation.
    ///
    /// # Arguments
    ///
    /// - `mid`: Byte offset at which to split the remaining space.
    ///   `mid` must be ≤ `self.available()`.
    ///
    /// # Returns
    ///
    /// Two `WorkspaceSplit` values representing the left and right halves.
    ///
    /// # Panics
    ///
    /// Panics if `mid > self.available()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::workspace::Workspace;
    ///
    /// let mut ws = Workspace::new(4096);
    /// let (mut left, mut right) = ws.split_at(2048);
    ///
    /// let buf_a: &mut [f64] = left.borrow_mut(128);
    /// let buf_b: &mut [f64] = right.borrow_mut(128);
    /// // buf_a and buf_b do not overlap
    /// ```
    pub fn split_at(&mut self, mid: usize) -> (WorkspaceSplit<'_>, WorkspaceSplit<'_>) {
        // Implementation:
        // 1. Validate mid <= available()
        // 2. Create two WorkspaceSplit views into non-overlapping regions
        // 3. Each sub-workspace tracks its own cursor independently
        ...
    }

    /// Splits the remaining space to satisfy two scratch requirements.
    ///
    /// Convenience method that computes the split point from the first
    /// requirement's size (aligned up to its alignment).
    ///
    /// # Panics
    ///
    /// Panics if the combined requirements exceed available capacity.
    pub fn split_for(
        &mut self,
        left: ScratchReq,
        right: ScratchReq,
    ) -> (WorkspaceSplit<'_>, WorkspaceSplit<'_>);
}
```

### 4.4 WorkspaceSplit — 分割后的子工作空间

```rust
/// A non-overlapping view into a portion of a parent `Workspace`.
///
/// Created by `Workspace::split_at`. Supports the same `borrow_mut` and
/// further `split_at` operations as the parent, but operates on a
/// bounded sub-region of the original buffer.
///
/// The lifetime `'a` is tied to the parent workspace's mutable borrow,
/// ensuring the parent cannot be used while sub-workspaces exist.
pub struct WorkspaceSplit<'a> {
    /// Pointer to the start of this sub-region.
    ptr: NonNull<u8>,

    /// Size of this sub-region in bytes.
    size: usize,

    /// Current cursor within this sub-region.
    cursor: usize,

    /// Alignment of the parent buffer.
    alignment: usize,

    /// Lifetime tied to parent workspace.
    _marker: PhantomData<&'a mut [u8]>,
}

impl<'a> WorkspaceSplit<'a> {
    /// Borrows a mutable typed slice from this sub-workspace.
    ///
    /// Same semantics as `Workspace::borrow_mut`, but bounded to this
    /// sub-region's address range.
    pub fn borrow_mut<A>(&mut self, count: usize) -> &mut [MaybeUninit<A>];

    /// Borrows a mutable typed slice, zero-initializing it.
    pub fn borrow_mut_zeroed<A>(&mut self, count: usize) -> &mut [A]
    where
        A: Default + Copy;

    /// Returns the remaining available bytes in this sub-region.
    #[inline]
    pub fn available(&self) -> usize {
        self.size - self.cursor
    }

    /// Recursively splits this sub-region into two non-overlapping parts.
    pub fn split_at(&mut self, mid: usize) -> (WorkspaceSplit<'_>, WorkspaceSplit<'_>);

    /// Resets the sub-region's cursor to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
```

### 4.5 运算集成模式

#### 4.5.1 可选 Workspace 参数模式

运算函数通过 `Option<&mut Workspace>` 参数接受可选的工作空间：

```rust
// In ops/matrix.rs or similar:

/// Computes the matrix multiplication C = A * B.
///
/// If `workspace` is provided, uses it for internal temporary buffers
/// (e.g., transposition). If `None`, allocates temporary buffers on the heap.
pub fn matmul<A, S1, S2>(
    a: &TensorBase<S1, Ix2>,
    b: &TensorBase<S2, Ix2>,
    workspace: Option<&mut Workspace>,
) -> Result<Tensor2<A>>
where
    A: RealScalar,
    S1: RawStorage<Elem = A>,
    S2: RawStorage<Elem = A>,
{
    // Internal helper: get workspace buffer or fall back to heap
    let needs_transpose = !a.is_f_contiguous() || !b.is_f_contiguous();
    if needs_transpose {
        let buf = match workspace {
            Some(ws) => ws.borrow_mut::<A>(a.len().max(b.len())),
            None => {
                // Fall back: allocate on heap
                // ...
            }
        };
        // ... use buf for transposition ...
    }
    // ...
}
```

#### 4.5.2 ScratchNeed 查询模式

```rust
// Upstream library usage example:

/// Scratch requirement for matmul with the given dimensions.
pub struct MatmulScratch {
    m: usize,
    k: usize,
    n: usize,
}

impl ScratchNeed for MatmulScratch {
    fn scratch_req(&self) -> ScratchReq {
        // Needs transpose buffer for the larger input
        let max_elems = (self.m * self.k).max(self.k * self.n);
        ScratchReq::for_elements::<f64>(max_elems)
    }
}

// Usage:
let scratch = MatmulScratch { m: 128, k: 64, n: 256 };
let req = scratch.scratch_req();
let mut ws = Workspace::new(0);
ws.reserve_for(req);

// Now ws has enough capacity for matmul
matmul(&a, &b, Some(&mut ws))?;

// Reuse the same workspace for another operation
let scratch2 = MatmulScratch { m: 256, k: 128, n: 512 };
ws.reserve_for(scratch2.scratch_req());
matmul(&c, &d, Some(&mut ws))?;
```

---

## 5. 内部实现设计

### 5.1 缓冲区分配策略

Workspace 使用与 Storage 模块相同的对齐分配技术：

```rust
use core::alloc::Layout;

// Conditional allocator import
#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, handle_alloc_error};

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, handle_alloc_error};

/// Allocates an aligned byte buffer.
///
/// Returns a NonNull pointer to `capacity` bytes aligned to `alignment`.
fn allocate_buffer(capacity: usize, alignment: usize) -> NonNull<u8> {
    if capacity == 0 {
        return NonNull::dangling();
    }
    let layout = Layout::from_size_align(capacity, alignment)
        .expect("invalid workspace layout: alignment must be power of 2");
    // SAFETY: layout is valid (non-zero size, power-of-2 alignment)
    NonNull::new(unsafe { alloc(layout) })
        .unwrap_or_else(|| handle_alloc_error(layout))
}

/// Deallocates a previously allocated buffer.
///
/// # Safety
///
/// - `ptr` must have been returned by `allocate_buffer`
/// - `capacity` and `alignment` must match the original allocation
unsafe fn deallocate_buffer(ptr: NonNull<u8>, capacity: usize, alignment: usize) {
    if capacity > 0 {
        let layout = Layout::from_size_align_unchecked(capacity, alignment);
        dealloc(ptr.as_ptr(), layout);
    }
}
```

### 5.2 增长策略

```
Workspace::reserve(additional)
    │
    ├── cursor + additional <= capacity
    │   └── No growth needed. Return.
    │
    └── cursor + additional > capacity
        │
        ├── Compute new_capacity = max(capacity * 2, cursor + additional)
        │   Round up to alignment boundary.
        │
        ├── allocate_buffer(new_capacity, alignment)
        │
        ├── Copy existing used bytes (cursor bytes) to new buffer
        │   (Preserves active borrows if any — though borrow_mut takes &mut self,
        │    so no active borrows can exist during reserve.)
        │
        ├── Deallocate old buffer
        │
        └── Update self.buffer, self.capacity
```

**增长策略选择理由：**

| 策略 | 理由 |
|------|------|
| 2x 指数增长 | 均摊 O(1) 扩容，与 `Vec` 一致 |
| 不缩容 | 工作空间通常在多次运算间复用，缩容反而增加分配 |
| 不保留旧数据 | `reserve` 发生在 `borrow_mut` 内部，此时 cursor 已归零，无需拷贝 |
| 对齐向上取整 | 确保 `cursor` 始终对齐，简化后续分配 |

### 5.3 借出与对齐计算

```rust
// Internal borrow logic (simplified):
fn borrow_mut<A>(&mut self, count: usize) -> &mut [MaybeUninit<A>] {
    let elem_size = core::mem::size_of::<A>();
    let elem_align = core::mem::align_of::<A>();

    // Compute required alignment for this allocation
    let required_align = elem_align.max(self.alignment);

    // Align the cursor forward
    let aligned_cursor = align_up(self.cursor, required_align);

    // Compute total byte size
    let byte_size = count.checked_mul(elem_size)
        .expect("workspace borrow size overflow");

    let end = aligned_cursor.checked_add(byte_size)
        .expect("workspace borrow offset overflow");

    // Grow if needed
    if end > self.capacity {
        self.reserve(end - self.cursor);
    }

    // SAFETY:
    // - self.buffer is valid for self.capacity bytes
    // - aligned_cursor is within [0, capacity) and properly aligned
    // - end <= capacity after potential growth
    // - MaybeUninit<A> has the same layout as A
    let ptr = unsafe {
        self.buffer.as_ptr().add(aligned_cursor) as *mut MaybeUninit<A>
    };

    self.cursor = end;

    // SAFETY: ptr is valid for count elements, properly aligned
    unsafe { core::slice::from_raw_parts_mut(ptr, count) }
}

/// Aligns `offset` up to the next multiple of `alignment`.
#[inline]
const fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}
```

### 5.4 分割实现

```rust
fn split_at(&mut self, mid: usize) -> (WorkspaceSplit<'_>, WorkspaceSplit<'_>) {
    assert!(mid <= self.available(), "split point exceeds available space");

    let base = unsafe { self.buffer.as_ptr().add(self.cursor) };

    // SAFETY: mid <= available, so both ranges are within buffer
    let left_ptr = NonNull::new(base).expect("null pointer in split");
    let right_ptr = NonNull::new(unsafe { base.add(mid) })
        .expect("null pointer in split");

    let left = WorkspaceSplit {
        ptr: left_ptr,
        size: mid,
        cursor: 0,
        alignment: self.alignment,
        _marker: PhantomData,
    };

    let right = WorkspaceSplit {
        ptr: right_ptr,
        size: self.available() - mid,
        cursor: 0,
        alignment: self.alignment,
        _marker: PhantomData,
    };

    (left, right)
}
```

**分割安全性保证：**

- `WorkspaceSplit` 的生命周期 `'a` 绑定到 `&mut Workspace`，保证父工作空间在子空间存活期间不可用
- 两个 `WorkspaceSplit` 的地址范围不重叠
- `WorkspaceSplit` 支持递归 `split_at`，进一步细分

### 5.5 与运算模块的集成点

```
运算函数调用模式:
    │
    ├── fn matmul(..., ws: Option<&mut Workspace>)
    │   │
    │   ├── match ws {
    │   │   ├── Some(ws) => ws.borrow_mut::<A>(needed)
    │   │   └── None => Vec::with_capacity(needed)  // fallback
    │   │
    │   └── 使用缓冲区后，drop 借用，cursor 重置
    │
    ├── fn tensordot(..., ws: Option<&mut Workspace>)
    │   │
    │   ├── 可能需要两次借出：转置 a 和转置 b
    │   │   ├── ws.borrow_mut::<A>(a.len())  // 第一次借出
    │   │   ├── drop 借用
    │   │   └── ws.borrow_mut::<A>(b.len())  // 第二次借出（复用同一空间）
    │   │
    │   └── 或并行需求：split_at 后两个子空间同时使用
    │
    └── ScratchNeed 查询（纯计算，无分配）
        ├── let req = op.scratch_req();
        ├── ws.reserve_for(req);
        └── 执行操作
```

### 5.6 无 MaybeUninit 的便捷封装（内部使用）

```rust
/// Internal helper: borrows a mutable slice assuming the caller will
/// immediately initialize it. Returns `&mut [A]` instead of `&mut [MaybeUninit<A>]`.
///
/// # Safety
///
/// The caller must initialize all elements before the slice is read.
/// This is intended for internal use within operations that write to
/// the entire buffer before reading.
unsafe fn borrow_mut_assume_init<A>(&mut self, count: usize) -> &mut [A] {
    let uninit = self.borrow_mut::<A>(count);
    // SAFETY: caller guarantees initialization before read
    &mut *(uninit as *mut [MaybeUninit<A>] as *mut [A])
}
```

此方法仅在内部运算模块中使用，不暴露为公共 API。公共 API 通过 `borrow_mut_zeroed` 或 `borrow_mut` + 显式 `MaybeUninit` 操作保证安全。

---

## 6. 实现任务拆分

> 每个任务约 10 分钟，可独立验证和提交。

### 6.1 基础类型

- [ ] **T1: ScratchReq 结构体 + 核心方法**
  - 文件: `src/workspace.rs:1-100`
  - 内容: `ScratchReq` struct 定义、`new()`, `for_elements()`, `for_elements_aligned()`, `size()`, `alignment()`, `none()`, `is_empty()`, `max()`
  - 测试: `test_scratch_req_new`, `test_scratch_req_for_elements`, `test_scratch_req_max`, `test_scratch_req_none`
  - 前置: 无
  - 预计: 10 min

- [ ] **T2: ScratchReq 组合运算（parallel_sum, Add, AddAssign）**
  - 文件: `src/workspace.rs`
  - 内容: `parallel_sum()` 方法、`Add` trait impl、`AddAssign` trait impl
  - 测试: `test_scratch_req_parallel_sum`, `test_scratch_req_add`, `test_scratch_req_add_assign`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3: ScratchNeed trait 定义**
  - 文件: `src/workspace.rs`
  - 内容: `ScratchNeed` trait 定义 + doc comments
  - 测试: 编译通过（trait 定义无单元测试，通过具体实现验证）
  - 前置: T1
  - 预计: 5 min

### 6.2 Workspace 核心

- [ ] **T4: Workspace 结构体定义 + 构造/销毁**
  - 文件: `src/workspace.rs`
  - 内容: `Workspace` struct 定义、`new()`, `with_alignment()`, `from_raw_parts()`, `empty()`, `Drop` impl、内部 `allocate_buffer` / `deallocate_buffer` 辅助函数
  - 测试: `test_workspace_new`, `test_workspace_empty`, `test_workspace_drop_no_leak`, `test_workspace_with_alignment`
  - 前置: 无
  - 预计: 10 min

- [ ] **T5: Workspace 容量查询 + reserve**
  - 文件: `src/workspace.rs`
  - 内容: `capacity()`, `used()`, `available()`, `alignment()`, `reserve()`, `reserve_for()`
  - 测试: `test_workspace_capacity`, `test_workspace_reserve_grows`, `test_workspace_reserve_for`, `test_workspace_reserve_for_alignment_mismatch_panics`
  - 前置: T4
  - 预计: 10 min

- [ ] **T6: Workspace borrow_mut + borrow_mut_zeroed**
  - 文件: `src/workspace.rs`
  - 内容: `borrow_mut::<A>()` 返回 `&mut [MaybeUninit<A>]`、`borrow_mut_zeroed::<A>()`、`reset()`、内部 `align_up` 辅助函数
  - 测试: `test_borrow_mut_basic`, `test_borrow_mut_alignment`, `test_borrow_mut_auto_grow`, `test_borrow_mut_zeroed`, `test_borrow_mut_sequential_reuse`, `test_borrow_mut_overflow_panics`
  - 前置: T5
  - 预计: 10 min

### 6.3 分割

- [ ] **T7: WorkspaceSplit 结构体 + borrow_mut**
  - 文件: `src/workspace.rs`
  - 内容: `WorkspaceSplit` struct 定义、`borrow_mut()`, `borrow_mut_zeroed()`, `available()`, `reset()`
  - 测试: `test_split_borrow_mut`, `test_split_non_overlapping`
  - 前置: T6
  - 预计: 10 min

- [ ] **T8: Workspace::split_at + split_for + 递归分割**
  - 文件: `src/workspace.rs`
  - 内容: `Workspace::split_at()`, `Workspace::split_for()`, `WorkspaceSplit::split_at()`
  - 测试: `test_split_at_basic`, `test_split_for`, `test_split_recursive`, `test_split_at_exceeds_available_panics`
  - 前置: T7
  - 预计: 10 min

### 6.4 集成与导出

- [ ] **T9: lib.rs 模块声明 + re-export**
  - 文件: `src/lib.rs`
  - 内容: `pub mod workspace;` 和 `pub use crate::workspace::{Workspace, ScratchReq, ScratchNeed};`
  - 测试: 外部 `use xenon::workspace::Workspace;` 编译通过
  - 前置: T1-T8
  - 预计: 5 min

- [ ] **T10: 集成测试 — Workspace 在运算中的使用模拟**
  - 文件: `tests/workspace.rs`
  - 内容: 模拟 matmul 场景的 Workspace 使用、scratch 查询 + 预分配 + 复用、分割并行使用
  - 测试: `test_workspace_simulated_matmul`, `test_workspace_scratch_query_then_use`, `test_workspace_parallel_split_usage`
  - 前置: T9
  - 预计: 10 min

---

## 7. 测试计划

### 7.1 单元测试

位于 `src/workspace.rs` 中的 `#[cfg(test)] mod tests`：

| 测试分类 | 测试项 | 关键断言 |
|----------|--------|----------|
| **ScratchReq 构造** | `test_scratch_req_new` | size 和 alignment 正确存储 |
| | `test_scratch_req_for_elements` | `for_elements::<f64>(10)` → `size=80, alignment=64` |
| | `test_scratch_req_none` | `none()` → `size=0, alignment=1` |
| | `test_scratch_req_alignment_panic` | `new(10, 3)` panic（非 2 的幂） |
| **ScratchReq 组合** | `test_scratch_req_max` | `max(a, b).size == max(a.size, b.size)` |
| | `test_scratch_req_parallel_sum` | `a + b` 的 size >= `a.size + b.size` |
| | `test_scratch_req_add_assign` | `a += b` 等价于 `a = a + b` |
| **Workspace 构造** | `test_workspace_new` | capacity 正确，alignment == 64 |
| | `test_workspace_empty` | capacity == 0，首次 borrow 触发分配 |
| | `test_workspace_with_alignment` | 自定义 alignment 正确 |
| | `test_workspace_drop_no_leak` | 反复创建/销毁不 OOM（循环 1000 次） |
| **Workspace 容量** | `test_workspace_capacity` | capacity() 返回正确值 |
| | `test_workspace_reserve_grows` | reserve 后 capacity >= 旧 capacity + additional |
| | `test_workspace_reserve_noop` | 容量充足时 reserve 不重分配 |
| **Workspace 借出** | `test_borrow_mut_basic` | 返回正确长度的切片 |
| | `test_borrow_mut_alignment` | 切片起始地址满足对齐要求 |
| | `test_borrow_mut_auto_grow` | 容量不足时自动扩容 |
| | `test_borrow_mut_zeroed` | 所有字节为零 |
| | `test_borrow_mut_sequential_reuse` | drop 后再次 borrow 返回相同地址区域 |
| | `test_borrow_mut_overflow_panics` | `usize::MAX / size_of::<A>() + 1` 个元素 panic |
| **Workspace 分割** | `test_split_at_basic` | 两个子空间 available 之和 == 父空间 available |
| | `test_split_non_overlapping` | 两子空间地址范围不重叠 |
| | `test_split_recursive` | 子空间可继续 split |
| | `test_split_for` | 按 ScratchReq 分割满足两个需求 |
| | `test_split_at_zero` | `split_at(0)` → 左为空，右为全部 |
| | `test_split_at_exceeds_available_panics` | mid > available 时 panic |

### 7.2 集成测试

| 文件 | 测试内容 |
|------|----------|
| `tests/workspace.rs` | 完整的使用模式：构造 → scratch 查询 → reserve → borrow → 计算 → reset → 复用 |
| | 模拟 matmul 场景：transpose buffer 借出 → 写入 → 读取验证 |
| | 模拟并行场景：split → 两个子空间独立 borrow → 验证不重叠 |

### 7.3 边界测试

| 场景 | 预期行为 |
|------|----------|
| 零容量 Workspace | `Workspace::empty()`，首次 `borrow_mut` 触发分配 |
| 零元素借出 | `borrow_mut::<f64>(0)` 返回空切片，不分配 |
| 单字节借出 | `borrow_mut::<u8>(1)` 正常工作 |
| 大对齐需求 | `Workspace::with_alignment(0, 128)` 后借出满足 128 字节对齐 |
| 反复扩容 | 多次超过当前容量的 borrow 后 workspace 仍工作正确 |
| 嵌套分割 3 层 | split → split → split 后各层独立 borrow 不重叠 |

### 7.4 属性测试

| 不变量 | 测试方法 |
|--------|----------|
| `ScratchReq::max(a, b).size >= max(a.size, b.size)` | 随机 ScratchReq 对 |
| `(a + b).size >= a.size + b.size` | 随机 ScratchReq 对（含对齐填充） |
| borrow 返回的切片起始地址对齐 | 随机 count 和类型参数 |
| split 后两子空间地址不重叠 | 随机 split 点 |
| 反复 borrow/drop 不泄漏 | 循环 10000 次，检查 RSS 不增长 |

### 7.5 性能验证

| 测试 | 方法 |
|------|------|
| borrow 开销 | benchmark：`borrow_mut::<f64>(1024)` vs `Vec::with_capacity(1024)` |
| 分割开销 | benchmark：`split_at` vs 手动指针分割 |
| 复用效果 | benchmark：1000 次 matmul 复用 Workspace vs 每次新建 Vec |

---

## 附录 A: Workspace 内部状态机

```
Workspace 状态转换:

    ┌──────────┐
    │  Empty   │ ← Workspace::empty() / Workspace::new(0)
    │  (no buf)│
    └────┬─────┘
         │ borrow_mut() triggers allocation
         ▼
    ┌──────────┐
    │  Ready   │ ← Workspace::new(N) / reset()
    │  cursor=0│
    └────┬─────┘
         │ borrow_mut()
         ▼
    ┌──────────┐
    │ Borrowed │ ← cursor > 0
    │ N bytes  │
    └────┬─────┘
         │ drop borrow + reset()
         ▼
    ┌──────────┐
    │  Ready   │ ← buffer reused, no reallocation
    │  cursor=0│
    └──────────┘
```

## 附录 B: 与 ndarray / NumPy 的设计对比

| 特性 | NumPy | ndarray (Rust) | Xenon |
|------|-------|----------------|-------|
| 临时缓冲区 | 内部分配，无用户控制 | 无专门 workspace 模块 | 显式 Workspace 参数 |
| Scratch 查询 | 无公共 API | 无 | ScratchNeed trait |
| 缓冲区复用 | 隐式（内部分配器） | 无 | 显式 Workspace 复用 |
| 分割 | 无 | 无 | split_at 递归分割 |
| 对齐 | 编译时常量 | 无保证 | 可配置（默认 64 字节） |
| 零初始化 | 默认零初始化 | 无保证 | 显式选择（borrow_mut vs borrow_mut_zeroed） |

**设计决策理由：** Xenon 选择显式 Workspace 参数而非隐式内部分配，因为：
1. 库开发者可跨多次运算复用同一缓冲区，避免重复 `malloc`
2. ScratchNeed 允许上游库在执行前查询需求，预分配一次
3. 与 BLAS 集成场景中，上游库需要控制工作内存的分配策略

## 附录 C: Workspace 内存布局示意

```
Workspace (capacity=4096, alignment=64):

    ┌─────────────────────────────────────────────────────────────────┐
    │  64-byte aligned buffer (4096 bytes)                           │
    │                                                                │
    │  ┌───────────────────────┬───────────────────────────────────┐│
    │  │  Used (cursor)        │  Available                        ││
    │  │  [borrow #1: 800B]    │  [free: 3296B]                    ││
    │  │  64B-aligned start    │                                    ││
    │  └───────────────────────┴───────────────────────────────────┘│
    │  ^                                                            │
    │  buffer (NonNull<u8>)                                        │
    └─────────────────────────────────────────────────────────────────┘

After split_at(1600):

    ┌─────────────────────────────────────────────────────────────────┐
    │  Left sub-workspace (1600B)    │  Right sub-workspace (1696B)  │
    │  ┌──────────┬────────┐       │  ┌──────────────────────────┐│
    │  │borrow 512│ free   │       │  │        free              ││
    │  └──────────┴────────┘       │  └──────────────────────────┘│
    └─────────────────────────────────────────────────────────────────┘
    Non-overlapping: left end ≤ right start
```
