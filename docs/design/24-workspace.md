# 临时工作空间模块设计

> 文档编号: 24 | 模块: `src/workspace/` | 阶段: Phase 4
> 前置文档: `05-storage.md`
> 需求参考: 需求说明书 §26

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 对齐分配 | 使用 `alloc::alloc` 进行指定对齐的内存分配 | arena 分配器（更复杂的分配策略） |
| no_std 支持 | 仅依赖 `core` 和 `alloc`，不依赖 `std` | 池分配（pooled allocation） |
| 分割 | `split_at` 将工作空间 O(1) 分割为两个子空间 | 栈分配（stack allocation，由调用方自行管理） |
| 动态扩容 | `ensure_capacity` 支持单向增长（不缩容） | 自动缩容策略 |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 借用语义 | 借出期间不可再次借出；只读借用当前版本同一时刻最多一个活跃 guard，归还后可复用 |
| 单向增长 | 只扩容不缩容，避免内存抖动 |
| 未初始化感知 | 底层字节视为 `MaybeUninit<u8>`；只有调用方显式声明初始化完成后，才能获取已初始化视图 |
| O(1) 分割 | 仅指针算术，无内存分配 |
| 显式生命周期 | 不可跨线程传递（`!Send + !Sync`），仅限创建它的线程使用 | 调用方负责线程安全 |

### 1.3 在架构中的位置

```
依赖层级：

L1: dimension, element, complex
L2: workspace  ← 当前模块（独立于 tensor）
          
上游库:
  blas-wrapper ──→ workspace
  fft-lib ───────→ workspace
  tensor (可选) ──→ workspace
```

Workspace 模块位于 L2，独立于核心张量类型系统，可被上游库直接使用而无需引入 tensor 依赖（参见 `01-architecture.md §5`）。

---

## 2. 文件位置

```
src/
└── workspace/               # 临时工作空间（目录模块）
    ├── mod.rs               # 模块根，re-exports
    ├── error.rs             # WorkspaceError 枚举
    ├── workspace.rs         # Workspace 结构体、常量、构造、析构
    ├── borrow.rs            # WorkspaceBorrow、WorkspaceBorrowMut 借用守卫
    ├── split.rs             # SplitBorrowMut 分割守卫
    └── expand.rs            # ensure_capacity、reallocate 扩容
```

多文件设计：按职责拆分，便于后续扩展（如新增借用策略、分配策略等）。

### 2.1 文件职责

| 文件 | 职责 | 预估行数 |
|------|------|
| `mod.rs` | 模块根，re-exports 所有公共类型 | ~20 |
| `error.rs` | `WorkspaceError` 枚举及 Display/Error impl | ~40 |
| `workspace.rs` | `Workspace` 结构体、常量、`new()`、`with_default_capacity()`、`Drop` | ~100 |
| `borrow.rs` | `WorkspaceBorrow`、`WorkspaceBorrowMut` 及其方法和 Drop | ~120 |
| `split.rs` | `SplitBorrowMut` 及其方法（`split_at`、`split_at_mut`、Drop） | ~100 |
| `expand.rs` | `ensure_capacity()`、`reallocate()` 扩容逻辑 | ~60 |

---

## 3. 依赖关系

### 3.1 依赖图

```
src/workspace/
├── mod.rs          # re-exports: Workspace, WorkspaceBorrow, WorkspaceBorrowMut, SplitBorrowMut, WorkspaceError
├── error.rs        # 独立，无内部依赖
├── workspace.rs    # 依赖 error
├── borrow.rs       # 依赖 workspace（通过引用）
├── split.rs        # 依赖 workspace（通过引用）、error
└── expand.rs       # 依赖 workspace（通过 &mut self）、error

外部依赖:
├── core            # ptr::NonNull, marker, sync::atomic, fmt
├── alloc           # alloc::alloc, alloc::dealloc, alloc::Layout
└── 无 crate 内部错误模块依赖；`WorkspaceError` 定义在 `workspace/error.rs`
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `core` | `NonNull<u8>`, `PhantomData`, `AtomicU8`, `fmt::Debug`, `fmt::Display` |
| `alloc` | `alloc()`, `dealloc()`, `Layout` |
| `workspace/error.rs` | `WorkspaceError`（模块内独立错误类型，参见 `26-error.md §4.7`） |

### 3.3 依赖方向声明

> **依赖方向：单向。** `workspace` 仅依赖 `core` 和 `alloc`，不依赖 `tensor`（参见 `07-tensor.md §3`）。上游库和 `tensor` 可消费 `workspace`。

### 3.4 WorkspaceError 的独立性

`WorkspaceError` 是 workspace 模块的独立错误类型，不属于 `XenonError` 枚举。调用方如需将其转换为其他错误类型，可自行实现 `From<WorkspaceError>` 用于其自定义错误类型。

> **与 XenonError 的关系**: `WorkspaceError` 是独立于 `XenonError` 的错误类型。Workspace 的错误场景（分配失败、空间不足）与张量操作的错误场景不同，使用独立类型避免 XenonError 膨胀。参见 `26-error.md`。

> **与线程安全需求的边界**: workspace 不是 `require.md §10` 中张量 storage mode 的一部分，而是独立的上游缓冲区工具。其 `!Send + !Sync` 设计不与张量存储模式的线程安全承诺冲突。

> **初始化语义约定**: Workspace 的底层缓冲区始终按“可能未初始化”建模。公共 API 默认只暴露 `MaybeUninit` 视图；只有调用方能够证明某一前缀或某一 typed region 已被完整写入时，才允许通过 `assume_init_*` 系列 unsafe API 将其解释为已初始化视图。

```rust
/// Error type for workspace operations.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkspaceError {
    /// Allocation failed (out of memory or invalid layout).
    AllocFailed { size: usize, align: usize },
    /// The requested alignment is not a power of 2.
    InvalidLayout { size: usize, align: usize },
    /// Cannot borrow: workspace is already mutably borrowed.
    AlreadyBorrowed,
    /// Split point is out of bounds.
    SplitOutOfBounds { split: usize, len: usize },
}
```

---

## 4. 公共 API 设计

### 4.1 Workspace 结构体

```rust
use core::ptr::NonNull;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU8, Ordering};
use alloc::alloc::{alloc, dealloc, Layout};

/// Temporary workspace.
///
/// Used for storing temporary buffers in numerical computation,
/// supporting aligned allocation and reuse.
///
/// # Lifetime Rules
///
/// - Cannot be re-borrowed while borrowed (enforced by borrow guards)
/// - Can be reused after returning
/// - Not transferable across threads (`!Send + !Sync`); only usable on the
///   creating thread. This ensures memory safety — Workspace holds raw
///   pointers, and cross-thread transfer could cause data races.
///
/// # Initialization Model
///
/// The backing allocation is modeled as uninitialized scratch memory.
/// Borrow APIs therefore expose `MaybeUninit<u8>` / `MaybeUninit<T>` views by default.
/// Converting to initialized `&[u8]` or `&mut [T]` is an explicit unsafe step that
/// requires the caller to prove the corresponding region has been fully initialized.
///
/// # Example
///
/// ```
/// let mut ws = Workspace::new(1024, 64)?;
///
/// // Mutable borrow
/// let mut buf = ws.borrow_mut()?;
/// let bytes = buf.as_maybe_uninit_slice();
/// // Initialize or reinterpret the scratch region...
///
/// // Return (RAII automatic)
/// drop(buf);
///
/// // Can borrow again
/// let buf2 = ws.borrow_mut()?;
/// ```
pub struct Workspace {
    /// Data pointer (non-null, aligned).
    ptr: NonNull<u8>,

    /// Current capacity in bytes.
    capacity: usize,

    /// Alignment in bytes at allocation time.
    alignment: usize,

    /// Borrow state (atomic).
    ///
    /// - 0: not borrowed
    /// - 1: shared borrow
    /// - 2: exclusive borrow
    borrow_state: AtomicU8,

    /// Active split count. Incremented by split_at(), decremented by SplitBorrowMut::drop().
    /// Only when this reaches 0 is borrow_state reset to BORROW_NONE.
    split_count: core::sync::atomic::AtomicUsize,

    /// Marker that intentionally prevents auto-deriving `Send`/`Sync`.
    _not_send_sync: PhantomData<*mut ()>,
}
```

> **设计决策：** 使用 `AtomicU8` 管理借用状态而非 `Mutex`，原因：无锁、状态简单（仅需 3 个值）。但这要求目标平台提供原子能力，因此本文档中的 `no_std` 兼容性语义应解读为 **`no_std + alloc + atomics`**（参见 `25-safety.md §4.1`）。

### 4.2 常量

```rust
impl Workspace {
    /// Default alignment: 64 bytes (AVX-512 cache line).
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// Minimum alignment.
    pub const MIN_ALIGNMENT: usize = 8;

    /// Default initial capacity: 4 KB.
    pub const DEFAULT_CAPACITY: usize = 4096;

    /// Borrow state constants.
    const BORROW_NONE: u8 = 0;
    const BORROW_SHARED: u8 = 1;
    const BORROW_EXCLUSIVE: u8 = 2;

    /// Growth factor numerator (1.5x).
    const GROWTH_FACTOR_NUMERATOR: usize = 3;
    const GROWTH_FACTOR_DENOMINATOR: usize = 2;

    /// Returns the current capacity in bytes.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Returns the alignment in bytes at allocation time.
    pub fn alignment(&self) -> usize { self.alignment }
}
```

### 4.3 构造方法

```rust
impl Workspace {
    /// Create a new workspace.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity in bytes
    /// * `alignment` - Alignment in bytes, must be a power of 2 and >= 8
    ///
    /// # Errors
    ///
    /// - `WorkspaceError::AllocFailed`: Memory allocation failed
    /// - `WorkspaceError::InvalidLayout`: Invalid layout parameters
    ///
    /// # Example
    ///
    /// ```
    /// let ws = Workspace::new(1024, 64)?;
    /// assert!(ws.capacity() >= 1024);
    /// ```
    pub fn new(capacity: usize, alignment: usize) -> Result<Self, WorkspaceError> {
        if !alignment.is_power_of_two() || alignment < Self::MIN_ALIGNMENT {
            return Err(WorkspaceError::InvalidLayout {
                size: capacity,
                align: alignment,
            });
        }

        let size = capacity.max(1);
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| WorkspaceError::InvalidLayout {
                size,
                align: alignment,
            })?;

        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(WorkspaceError::AllocFailed {
            size,
            align: alignment,
        })?;

        Ok(Self {
            ptr,
            capacity: size,
            alignment,
            borrow_state: AtomicU8::new(Self::BORROW_NONE),
            split_count: core::sync::atomic::AtomicUsize::new(0),
            _not_send_sync: PhantomData,
        })
    }

    /// Create a workspace with default parameters.
    pub fn with_default_capacity() -> Result<Self, WorkspaceError> {
        Self::new(Self::DEFAULT_CAPACITY, Self::DEFAULT_ALIGNMENT)
    }
}
```

### 4.4 析构方法

```rust
impl Drop for Workspace {
    fn drop(&mut self) {
        // SAFETY: layout was valid at allocation time, and ptr is the same.
        unsafe {
            let layout = Layout::from_size_align_unchecked(
                self.capacity,
                self.alignment,
            );
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

// Clone forbidden (semantically unique)
```

### 4.5 借用 API

```rust
/// Immutable borrow guard.
///
/// RAII guard that automatically returns the workspace on drop.
pub struct WorkspaceBorrow<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

/// Mutable borrow guard.
///
/// RAII guard that automatically returns the workspace on drop.
pub struct WorkspaceBorrowMut<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

impl Workspace {
    /// Acquire the workspace for shared inspection of the scratch region.
    ///
    /// # Errors
    ///
    /// `WorkspaceError::AlreadyBorrowed`: Workspace is already borrowed.
    ///
    /// Note: the current design allows at most one active shared guard at a time.
    /// This keeps the runtime state machine simple and matches the workspace's
    /// temporary-scratch-buffer positioning. The returned guard still models the
    /// bytes as potentially uninitialized; use `assume_init_slice` only when the
    /// caller can prove the inspected prefix has been written.
    pub fn borrow(&self) -> Result<WorkspaceBorrow<'_>, WorkspaceError> {
        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_SHARED,
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if prev.is_err() {
            return Err(WorkspaceError::AlreadyBorrowed);
        }

        Ok(WorkspaceBorrow {
            ptr: self.ptr,
            len: self.capacity,
            workspace: self,
        })
    }

    /// Mutably borrow the workspace.
    ///
    /// # Errors
    ///
    /// `WorkspaceError::AlreadyBorrowed`: Workspace is already borrowed.
    pub fn borrow_mut(&self) -> Result<WorkspaceBorrowMut<'_>, WorkspaceError> {
        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_EXCLUSIVE,
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if prev.is_err() {
            return Err(WorkspaceError::AlreadyBorrowed);
        }

        Ok(WorkspaceBorrowMut {
            ptr: self.ptr,
            len: self.capacity,
            workspace: self,
        })
    }
}
```

### 4.6 借用守卫方法

```rust
impl<'a> WorkspaceBorrow<'a> {
    /// Returns the data pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns the scratch region as possibly-uninitialized bytes.
    pub fn as_maybe_uninit_slice(&self) -> &[core::mem::MaybeUninit<u8>] {
        // SAFETY: `MaybeUninit<u8>` may represent uninitialized bytes.
        unsafe {
            core::slice::from_raw_parts(
                self.ptr.as_ptr() as *const core::mem::MaybeUninit<u8>,
                self.len,
            )
        }
    }

    /// Interprets an initialized prefix as `&[u8]`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the first `initialized_len` bytes have been
    /// fully initialized before calling this method.
    pub unsafe fn assume_init_slice(&self, initialized_len: usize) -> &[u8] {
        assert!(initialized_len <= self.len);
        core::slice::from_raw_parts(self.ptr.as_ptr(), initialized_len)
    }

    /// Returns the borrow length.
    pub fn len(&self) -> usize { self.len }

    /// Returns whether the borrow is empty.
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl<'a> WorkspaceBorrowMut<'a> {
    /// Returns the mutable data pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the mutable scratch region as possibly-uninitialized bytes.
    pub fn as_maybe_uninit_slice(&mut self) -> &mut [core::mem::MaybeUninit<u8>] {
        // SAFETY: `MaybeUninit<u8>` may represent uninitialized bytes.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut core::mem::MaybeUninit<u8>,
                self.len,
            )
        }
    }

    /// Typed access to possibly-uninitialized scratch memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - Alignment satisfies the type's requirements
    /// - Capacity is sufficient to hold `count` elements
pub unsafe fn as_maybe_uninit_typed_slice<T>(
        &mut self,
        count: usize,
    ) -> &mut [core::mem::MaybeUninit<T>] {
        assert!(core::mem::size_of::<T>() != 0, "ZST typed borrows are not supported");
        let byte_len = count
            .checked_mul(core::mem::size_of::<T>())
            .expect("typed workspace borrow byte length overflow");
        assert!(byte_len <= self.len);
        assert!(self.ptr.as_ptr() as usize % core::mem::align_of::<T>() == 0);
        core::slice::from_raw_parts_mut(
            self.ptr.as_ptr() as *mut core::mem::MaybeUninit<T>,
            count,
        )
    }

    /// Interprets the first `count` elements as initialized `T` values.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the first `count` typed elements have been
    /// fully initialized before calling this method.
    pub unsafe fn assume_init_typed_slice<T>(&mut self, count: usize) -> &mut [T] {
        assert!(core::mem::size_of::<T>() != 0, "ZST typed borrows are not supported");
        let byte_len = count
            .checked_mul(core::mem::size_of::<T>())
            .expect("typed workspace borrow byte length overflow");
        assert!(byte_len <= self.len);
        assert!(self.ptr.as_ptr() as usize % core::mem::align_of::<T>() == 0);
        core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
    }
}

// RAII return
impl<'a> Drop for WorkspaceBorrow<'a> {
    fn drop(&mut self) {
        self.workspace.borrow_state.store(
            Workspace::BORROW_NONE,
            Ordering::Release,
        );
    }
}

impl<'a> Drop for WorkspaceBorrowMut<'a> {
    fn drop(&mut self) {
        self.workspace.borrow_state.store(
            Workspace::BORROW_NONE,
            Ordering::Release,
        );
    }
}
```

### 4.7 分割 API

```rust
/// Borrow guard for a split sub-space.
///
/// Similar to `WorkspaceBorrowMut`, but allows multiple split guards
/// to coexist (pointing to non-overlapping memory regions).
pub struct SplitBorrowMut<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
    /// Reference to the split count. The split() operation atomically
    /// increments the SPLIT_COUNT counter. Each SplitBorrowMut holds a
    /// reference to this counter. Drop decrements the counter; only when
    /// the counter reaches 0 is borrow_state reset to BORROW_NONE.
    split_count: &'a core::sync::atomic::AtomicUsize,
}

impl Workspace {
    /// Split the workspace at the specified position into two sub-spaces.
    ///
    /// # Arguments
    ///
    /// * `mid` - Split point (byte offset)
    ///
    /// # Returns
    ///
    /// Two mutable borrow guards for the sub-spaces `(left, right)`.
    ///
    /// # Complexity
    ///
    /// O(1) — pointer arithmetic only, no memory allocation.
    ///
    /// # RAII Behavior
    ///
    /// Dropping **the last** `SplitBorrowMut` releases the workspace for re-use.
    /// Reference counting ensures the workspace is not re-borrowable until ALL
    /// sub-spaces (including those from recursive `split_at_mut` calls) are dropped.
    /// Hold all sub-spaces until done to maintain safety.
    ///
    /// # Errors
    ///
    /// - `WorkspaceError::SplitOutOfBounds`: `mid > capacity`
    /// - `WorkspaceError::AlreadyBorrowed`: Already borrowed
    ///
    /// # Example
    ///
    /// ```
    /// let mut ws = Workspace::new(1024, 64)?;
    /// let (left, right) = ws.split_at(512)?;
    /// // left: [0, 512), right: [512, 1024)
    /// ```
    pub fn split_at(
        &self,
        mid: usize,
    ) -> Result<(SplitBorrowMut<'_>, SplitBorrowMut<'_>), WorkspaceError> {
        if mid > self.capacity {
            return Err(WorkspaceError::SplitOutOfBounds { split: mid, len: self.capacity });
        }

        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_EXCLUSIVE,
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if prev.is_err() {
            return Err(WorkspaceError::AlreadyBorrowed);
        }

        // Set split_count to 2 (two sub-spaces will be created)
        self.split_count.store(2, Ordering::Release);

        let left_ptr = self.ptr;
        // SAFETY: mid <= capacity, so ptr + mid is within allocation
        let right_ptr = unsafe {
            NonNull::new_unchecked(self.ptr.as_ptr().add(mid))
        };

        Ok((
            SplitBorrowMut { ptr: left_ptr, len: mid, workspace: self, split_count: &self.split_count },
            SplitBorrowMut { ptr: right_ptr, len: self.capacity - mid, workspace: self, split_count: &self.split_count },
        ))
    }
}

impl<'a> SplitBorrowMut<'a> {
    /// Returns the split sub-space as possibly-uninitialized bytes.
    pub fn as_maybe_uninit_slice(&mut self) -> &mut [core::mem::MaybeUninit<u8>] {
        // SAFETY: split guards still expose scratch memory as possibly uninitialized.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut core::mem::MaybeUninit<u8>,
                self.len,
            )
        }
    }

    /// Continue splitting (recursive binary split).
    ///
    /// O(1) — pointer arithmetic only.
    ///
    /// **Safety design:** `split_at_mut` consumes `self` rather than borrowing.
    /// This ensures each `SplitBorrowMut` instance has an independent lifetime,
    /// preventing counter inconsistency. The original `SplitBorrowMut` is consumed
    /// and no longer valid; the two sub-splits independently manage their Drop
    /// behavior.
    ///
    /// **Reference count invariant:** This method atomically increments
    /// `split_count` by 1 before creating the two sub-splits. Rationale:
    /// consuming `self` prevents its Drop from running (net −1 decrement avoided),
    /// but two new guards are created (net +2 decrements expected). The net
    /// change in active guards is +1, so `split_count` must increase by 1
    /// to remain consistent. Without this increment, the last sub-split's
    /// Drop would observe `prev == 1` prematurely and reset `borrow_state`
    /// while other sub-splits are still active — a safety hazard.
    pub fn split_at_mut(
        self,
        mid: usize,
    ) -> Result<(SplitBorrowMut<'a>, SplitBorrowMut<'a>), WorkspaceError> {
        if mid > self.len {
            return Err(WorkspaceError::SplitOutOfBounds { split: mid, len: self.len });
        }

        let this = core::mem::ManuallyDrop::new(self);

        // SAFETY: Increment split_count to account for the additional
        // sub-space created by this recursive split. `self` is consumed
        // (not dropped), so its implicit "slot" in the count is released.
        // But we create 2 new guards, so the net active guard count
        // increases by 1. This ensures Drop correctly waits for ALL
        // active sub-splits before resetting borrow_state.
        this.split_count.fetch_add(1, core::sync::atomic::Ordering::Release);

        let left_ptr = this.ptr;
        let right_ptr = unsafe {
            NonNull::new_unchecked(this.ptr.as_ptr().add(mid))
        };

        Ok((
            SplitBorrowMut { ptr: left_ptr, len: mid, workspace: this.workspace, split_count: this.split_count },
            SplitBorrowMut { ptr: right_ptr, len: this.len - mid, workspace: this.workspace, split_count: this.split_count },
        ))
    }

    /// Returns the sub-space length.
    pub fn len(&self) -> usize { self.len }
}

/// Drop releases the exclusive borrow on the workspace.
///
/// Uses reference counting: each split_at() sets split_count to the number
/// of sub-spaces (2 for binary split); each split_at_mut() atomically
/// increments split_count by 1 (net +1 active guard). Each
/// SplitBorrowMut::drop() atomically decrements split_count. Only when
/// split_count reaches 0 is borrow_state reset to BORROW_NONE. This
/// prevents the safety hazard where dropping one sub-space prematurely
/// resets borrow_state while other sub-spaces are still in use (including
/// sub-spaces from recursive split_at_mut calls).
///
/// # Safety Invariant
///
/// After `drop`, the caller must not use any existing references into the workspace
/// memory. The Rust borrow checker enforces this via the `'a` lifetime.
impl<'a> Drop for SplitBorrowMut<'a> {
    fn drop(&mut self) {
        // Atomically decrement the split count.
        let prev = self.split_count.fetch_sub(1, core::sync::atomic::Ordering::AcqRel);
        // Only reset borrow_state when this is the last active split.
        if prev == 1 {
            self.workspace.borrow_state.store(
                Workspace::BORROW_NONE,
                core::sync::atomic::Ordering::Release,
            );
        }
    }
}
```

### 4.8 扩容 API

```rust
impl Workspace {
    /// Ensure capacity is at least `min_capacity`.
    ///
    /// If current capacity is insufficient, a larger memory region will be allocated.
    /// New capacity = max(requested capacity, current capacity × 1.5).
    ///
    /// # Errors
    ///
    /// - `WorkspaceError::AlreadyBorrowed`: Workspace is already borrowed
    /// - `WorkspaceError::AllocFailed`: Memory allocation failed
    ///
    /// # Example
    ///
    /// ```
    /// let mut ws = Workspace::new(1024, 64)?;
    /// ws.ensure_capacity(2048)?;  // Grow to at least 2048
    /// ```
    pub fn ensure_capacity(
        &mut self,
        min_capacity: usize,
    ) -> Result<(), WorkspaceError> {
        if min_capacity <= self.capacity {
            return Ok(());
        }

        // Check borrow state
        let state = self.borrow_state.load(Ordering::Acquire);
        if state != Self::BORROW_NONE {
            return Err(WorkspaceError::AlreadyBorrowed);
        }

        // 1.5x growth
        let grown = self.capacity * Self::GROWTH_FACTOR_NUMERATOR
            / Self::GROWTH_FACTOR_DENOMINATOR;
        let new_capacity = grown.max(min_capacity);

        self.reallocate(new_capacity)
    }

    /// Internal reallocation.
    fn reallocate(&mut self, new_capacity: usize) -> Result<(), WorkspaceError> {
        let new_layout = Layout::from_size_align(new_capacity, self.alignment)
            .map_err(|_| WorkspaceError::InvalidLayout {
                size: new_capacity,
                align: self.alignment,
            })?;

        let new_ptr = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_ptr)
            .ok_or(WorkspaceError::AllocFailed {
                size: new_capacity,
                align: self.alignment,
            })?;

        // Copy old data
        // SAFETY: src and dst do not overlap, copy min(old, new) bytes
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                new_ptr.as_ptr(),
                self.capacity.min(new_capacity),
            );
        }

        // Free old memory
        // SAFETY: old layout was valid at allocation time
        unsafe {
            let old_layout = Layout::from_size_align_unchecked(
                self.capacity,
                self.alignment,
            );
            dealloc(self.ptr.as_ptr(), old_layout);
        }

        self.ptr = new_ptr;
        self.capacity = new_capacity;

        Ok(())
    }
}
```

### 4.9 Good/Bad 对比

```rust
// Good - Split workspace using split_at
let mut ws = Workspace::new(1024, 64)?;
let (mut left, mut right) = ws.split_at(512)?;
// left and right point to non-overlapping memory regions
// Safe for independent sub-space processing on the same owning thread

// Good - Expose scratch memory as MaybeUninit first
let mut ws = Workspace::new(1024, 64)?;
let mut buf = ws.borrow_mut()?;
let scratch = buf.as_maybe_uninit_slice();
// Initialize scratch before reinterpretation

// Bad - Treating scratch memory as initialized bytes without proof
let mut ws = Workspace::new(1024, 64)?;
let mut buf = ws.borrow_mut()?;
let bytes: &mut [u8] = unsafe { buf.assume_init_typed_slice::<u8>(1024) };
// UB risk if the region has not been fully initialized first

// Bad - Directly manipulating raw pointers to bypass borrow checking
let mut ws = Workspace::new(1024, 64)?;
let ptr = ws.ptr.as_ptr();  // ptr field is private!
// Bypasses borrow checking, may cause data races

// Good - Ensure capacity before use
let mut ws = Workspace::new(256, 64)?;
ws.ensure_capacity(1024)?;  // Grow first
let mut buf = ws.borrow_mut()?;
// Safe to use the larger buffer

// Bad - Resize during borrow
let mut ws = Workspace::new(256, 64)?;
let buf = ws.borrow_mut()?;
ws.ensure_capacity(1024);  // Compile error! borrow_mut requires &mut self
```

---

## 5. 内部实现设计

### 5.1 对齐分配实现

```
Workspace 内存布局（64 字节对齐）

地址:     0x00           0x40           0x80           0xC0
          ├──────────────┼──────────────┼──────────────┼──────────────┤
数据:     |  scratch  |  scratch  |  scratch  |  scratch  |
          |  buffer  |  buffer  |  buffer  |  buffer  |
          └────────────┴────────────┴────────────┴────────────┘
          │<───────────────── capacity ─────────────────>│

          ↑
          ptr (NonNull<u8>)

对齐检查: (addr % 64) == 0 ✓
```

### 5.2 split_at O(1) 原理

```
传统方案（需分配）:
┌─────────────────────────────────────┐
│ Workspace [1024 bytes]              │
└─────────────────────────────────────┘
         │
         ▼ 分配新 Workspace (512 bytes)
┌──────────────────┐  ┌──────────────────┐
│ Left [512 bytes] │  │ Right [512 bytes]│
│ (独立分配)       │  │ (独立分配)       │
└──────────────────┘  └──────────────────┘
         O(n) 内存拷贝 ❌

本设计（零分配）:
┌─────────────────────────────────────┐
│ Workspace [1024 bytes]              │
│ ptr = 0x1000                        │
└─────────────────────────────────────┘
         │
         ▼ 仅指针算术
┌──────────────────┐  ┌──────────────────┐
│ SplitBorrowMut   │  │ SplitBorrowMut   │
│ ptr = 0x1000     │  │ ptr = 0x1200     │
│ len = 512        │  │ len = 512        │
│ (视图，无分配)   │  │ (视图，无分配)   │
└──────────────────┘  └──────────────────┘
         O(1) ✓
```

### 5.3 split_at_mut 安全设计

> **安全设计决策：** `split_at_mut` 采用消费式设计（consuming self）而非借用：调用 `split_at_mut(self, mid)` 消耗原 `SplitBorrowMut`，返回两个新的子 `SplitBorrowMut`。这确保每个 `SplitBorrowMut` 实例有独立的生命周期，避免计数器不一致问题。原始 `SplitBorrowMut` 被消耗后不再有效，两个子分割各自独立管理其 Drop 行为。
>
> **引用计数安全性：** `split_at_mut` 在创建两个子分割之前，通过 `fetch_add(1)` 原子递增 `split_count`。这是因为：
>
> - 消耗 `self` 意味着原 `SplitBorrowMut` 的 `Drop` 不会执行（避免了隐含的 −1 递减）
> - 但两个新的 `SplitBorrowMut` 被创建（产生 +2 递减）
> - 净变化为 +1 个活跃守卫，因此 `split_count` 需要增加 1 以保持一致
> - 如果不加 1，最后一个子分割的 `Drop` 会过早观察到 `prev == 1`，在其他子分割仍然活跃时重置 `borrow_state`，造成安全隐患
>
> **示例（3 个活跃子空间）**：
> 1. `split_at(512)` → `split_count = 2`，创建 `left` 和 `right`
> 2. `right.split_at_mut(128)` → `fetch_add(1)`，`split_count = 3`，创建 `right_a` 和 `right_b`
> 3. `left` drop → `split_count: 3→2`，`prev=3 ≠ 1`，不重置 ✅
> 4. `right_a` drop → `split_count: 2→1`，`prev=2 ≠ 1`，不重置 ✅
> 5. `right_b` drop → `split_count: 1→0`，`prev=1`，重置 `borrow_state` ✅

### 5.4 扩容安全性论证

**扩容期间保证不违反已有引用安全性**（参见 `05-storage.md §5`）：

1. `ensure_capacity` 需要 `&mut self`，编译器保证无其他引用
2. 方法内部显式检查 `borrow_state` 是否为 NONE
3. 扩容后旧指针失效（`dealloc`），新指针更新
4. 由于 `&mut self` 保证独占，无悬挂引用
5. 任何 `MaybeUninit` / `assume_init_*` 视图都不得跨越 `ensure_capacity` 保留

```
扩容流程：
ensure_capacity(&mut self, 2048)
    │
    ├── 1. 检查 borrow_state == NONE  ✓
    ├── 2. 分配新内存 (2048 bytes)
    ├── 3. copy_nonoverlapping 旧 → 新
    ├── 4. dealloc 旧内存
    └── 5. 更新 ptr 和 capacity
```

---

## 6. 实现任务拆分

### Wave 1: 基础结构

- [ ] **T1**: 定义 `WorkspaceError` 枚举
  - 文件: `src/workspace/error.rs`
  - 内容: `WorkspaceError` 枚举及 Display/Error impl
  - 测试: `test_workspace_error_display`
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 定义 `Workspace` 结构体和构造方法
  - 文件: `src/workspace/workspace.rs`
  - 内容: `Workspace` 结构体、常量定义、`new()`、`with_default_capacity()`、`Drop`
  - 测试: `test_workspace_new`, `test_workspace_new_default`, `test_workspace_constants`, `test_workspace_drop_no_leak`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 编写模块根 `mod.rs`
  - 文件: `src/workspace/mod.rs`
  - 内容: 子模块声明、re-exports
  - 测试: 编译通过
  - 前置: T1
  - 预计: 5 min

### Wave 2: 借用机制

- [ ] **T4**: 实现借用守卫类型和 MaybeUninit 访问方法
  - 文件: `src/workspace/borrow.rs`
  - 内容: `WorkspaceBorrow`、`WorkspaceBorrowMut` 结构体、`borrow()`、`borrow_mut()`、`as_maybe_uninit_slice()`、`assume_init_slice()`、`as_maybe_uninit_typed_slice()`、`assume_init_typed_slice()`、`Drop` 实现
  - 测试: `test_borrow_basic`, `test_borrow_mut_basic`, `test_borrow_double_fails`, `test_borrow_after_drop`, `test_assume_init_requires_initialized_prefix`
  - 前置: T2
  - 预计: 10 min

### Wave 3: 分割和扩容

- [ ] **T5**: 实现 `split_at` 和 `SplitBorrowMut`
  - 文件: `src/workspace/split.rs`
  - 内容: `SplitBorrowMut` 结构体、`split_at()`、`split_at_mut()` 递归、`Drop` 实现
  - 测试: `test_split_at_basic`, `test_split_at_recursive`, `test_split_at_oob`
  - 前置: T2
  - 预计: 10 min

- [ ] **T6**: 实现扩容策略 `ensure_capacity`/`reallocate`
  - 文件: `src/workspace/expand.rs`
  - 内容: `ensure_capacity()`、`reallocate()`
  - 测试: `test_ensure_capacity_no_grow`, `test_ensure_capacity_grow`, `test_ensure_capacity_while_borrowed_fails`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 集成和文档

- [ ] **T7**: 完善模块导出和文档注释
  - 文件: `src/workspace/mod.rs` 及各子模块
  - 内容: 完善公共导出、完整文档注释、使用示例
  - 测试: `cargo doc` 通过
  - 前置: T4, T5, T6
  - 预计: 10 min

### 并行执行图

```
Wave 1:          [T1]
                ╱    ╲
Wave 2:      [T3]    [T2]             ← T2、T3 并行（均依赖 T1）
                     ╱  |  ╲
Wave 3:           [T4] [T5] [T6]      ← T4、T5、T6 并行（均依赖 T2）
                     ╲  |  ╱
Wave 4:               [T7]            ← 依赖 T4、T5、T6 全部完成
```

> **关键路径**: T1 → T2 → T4/T5/T6（最长） → T7。T3 不在关键路径上，可在任何时间完成。

---

## 7. 测试计划

### 7.1 测试分类表

| 测试分类 | 位置 | 说明 |
|----------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证工作空间分配、借用、分割与扩容语义 |
| 集成测试 | `tests/` | 验证 `workspace` 与 `ffi`、上游 scratch-buffer 场景的协同路径 |
| 边界测试 | 同模块测试中标注 | 覆盖零容量、最小对齐、大容量和递归分割等边界 |
| 属性测试 | `tests/property/` | 验证借用状态机、typed borrow 字节长度与分割不变量 |

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_workspace_new_basic` | 指定容量和对齐创建工作空间 | 高 |
| `test_workspace_new_default` | 默认参数创建 | 高 |
| `test_workspace_new_invalid_alignment` | 非法对齐值返回 `WorkspaceError::InvalidLayout` | 高 |
| `test_workspace_drop_no_leak` | Drop 后内存正确释放 | 中 |
| `test_borrow_basic` | 不可变借用和 `MaybeUninit` 切片访问 | 高 |
| `test_borrow_mut_basic` | 可变借用和 `MaybeUninit` 类型化访问 | 高 |
| `test_borrow_double_fails` | 重复借用返回错误 | 高 |
| `test_borrow_after_drop` | 归还后可再次借用 | 高 |
| `test_assume_init_requires_initialized_prefix` | 已初始化视图只覆盖调用方证明已初始化的前缀 | 高 |
| `test_split_at_basic` | 固定位置分割 | 高 |
| `test_split_at_recursive` | 递归分割（多级） | 中 |
| `test_split_at_oob` | 越界分割返回错误 | 高 |
| `test_ensure_capacity_no_grow` | 容量足够时不扩容 | 高 |
| `test_ensure_capacity_grow` | 容量不足时扩容到 1.5 倍 | 高 |
| `test_ensure_capacity_while_borrowed_fails` | 借用期间扩容失败 | 高 |
| `test_alignment_verification` | 对齐值验证 | 中 |
| `test_typed_slice_alignment` | 类型化切片对齐检查 | 高 |

### 7.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 零容量工作空间 | `new(0, 64)` 返回最小容量 1 的工作空间 |
| 最小对齐（8 字节） | 正常创建和使用 |
| 大容量（1MB+） | 正常分配和释放 |
| 递归分割到空子空间 | `split_at(0)` 返回空左半 |
| `ensure_capacity(0)` | 无操作（容量已足够） |
| 未初始化区域只读访问 | 仅允许 `MaybeUninit` 视图，不暴露 `&[u8]` |

### 7.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `capacity() >= new()` 请求的容量 | 随机容量 |
| `split_at` 后 `left.len + right.len == capacity` | 随机分割点 |
| 借用后 `borrow_state != NONE` | 借用状态检查 |
| 扩容后对齐不变 | `alignment()` 一致 |
| 未初始化区域只能通过 `MaybeUninit` 视图暴露 | 随机借用 + API 形状检查 |

### 7.5 集成测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/workspace.rs` | `new` / `borrow` / `split_at_mut` / `ensure_capacity` 与 `ffi`、上游 scratch-buffer 场景的端到端协同验证，并验证 `MaybeUninit` 视图与 `assume_init_*` 前缀约束 |

---

## 8. 与其他模块的交互

### 8.1 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
|------|----------|-----------|------|
| `workspace ← 上游库` | `上游库` | 临时缓冲区请求 | BLAS 等上游库可把 workspace 作为临时缓冲区使用；默认只获得 `MaybeUninit` 视图 |
| `workspace ← 上游库` | `上游库` | `split_at()` | FFT 等场景可通过分割接口拆分工作空间；子空间同样只暴露 `MaybeUninit` 视图 |
| `workspace ← 上游库` | `上游库` | `assume_init_*` | 调用方必须先完成写入并证明对应前缀/typed region 已初始化，才能重解释为已初始化视图 |
| `workspace ← tensor` | `tensor` | 临时 scratch 空间 | 张量运算在需要时可借用 workspace 提供临时空间；任何借出的 scratch 视图都不得跨越 `ensure_capacity` 或持久化到 tensor 元数据中 |

### 8.2 数据流描述

```
上层模块请求临时工作空间
    │
    ├── Workspace::new(capacity, alignment)
    │       └── 分配 64-byte 对齐原始缓冲区
    │
    ├── borrow() / borrow_mut() / split_at()
    │       └── 生成线程内有效的 `MaybeUninit` 借用守卫或子空间视图
    │
    ├── 调用方在守卫生命周期内写入临时数据
    │       └── 如需读取已初始化前缀，显式调用 `assume_init_*`
    │
    └── Drop 守卫后恢复 borrow_state，可供后续操作复用
```

---

## 9. 设计决策记录(ADR)

### 决策 1：设计选择 - workspace vs arena vs pool

| 属性 | 值 |
|------|-----|
| 决策 | 使用简单的 workspace 类型而非 arena 或 pool 分配器 |
| 理由 | 实现简单（~400 行）、语义清晰（借用/归还）、满足需求（对齐/分割/扩容）；arena 分配器更复杂，pool 附加管理生命周期困难 |
| 替代方案 | arena 分配器（bump 分配） — 放弃，需求不复杂，无需 zone/reset |
| 替代方案 | pool 分配（对象池） — 放弃，工作空间操作原始字节，无需对象语义 |

### 决策 2:借用期间不可再次借出

| 属性 | 值 |
|------|-----|
| 决策 | 借出期间禁止再次借出（由 AtomicU8 CAS 保证） |
| 理由 | 安全性：避免同一缓冲区被多次借出导致数据竞争；简单性: 单一借用模型更易理解；split_at() 生成的子空间全部释放后，父工作空间才恢复可借用状态。 |
| 替代方案 | 允许共享借用（多个 reader） — 未来可扩展，当前版本简化 |

### 决策 3:扩容安全性保证

| 属性 | 值 |
|------|-----|
| 决策 | 扩容前显式检查 borrow_state == NONE，且需要 `&mut self` |
| 理由 | `&mut self` 由编译器保证独占访问；显式检查原子状态作为双重保障；防止扩容导致已有引用失效 |
| 替代方案 | 不检查直接扩容 — 放弃，UB 风险太高 |

### 决策 4:不保证零初始化

| 属性 | 值 |
|------|-----|
| 决策 | 工作空间不保证零初始化 |
| 理由 | 性能: 零初始化是 O(n) 操作;不必要: 大多数场景下调用方会覆盖全部数据;与 C 一致: 与 malloc 行为一致 |
| 替代方案 | 构造时零初始化 — 放弃，性能损失 |

### 决策 5：Workspace 不实现 Send/Sync

| 属性 | 值 |
|------|-----|
| 决策 | Workspace 不实现 Send 或 Sync |
| 理由 | Workspace 设计为线程内临时缓冲区，文档约束为 `!Send + !Sync`。即使存在运行时借用状态检查，也不将其建模为可跨线程传递或共享的基础类型；若调用方需要多线程临时缓冲区，应在线程边界外自行分配和管理独立工作空间。 |
| 替代方案 | 使用 AtomicU8 实现 Send + Sync — 放弃，仅有运行时检查不够，多线程场景下需要完整同步 |

---

## 10. 性能考量

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| `new()` | O(1) | O(capacity) | 一次分配 |
| `borrow()` | O(1) | O(1) | 原子 CAS |
| `borrow_mut()` | O(1) | O(1) | 原子 CAS |
| `split_at()` | O(1) | O(1) | 仅指针算术 |
| `split_at_mut()` | O(1) | O(1) | 仅指针算术 |
| `ensure_capacity()` | O(n) | O(new_capacity) | 分配 + 拷贝 + 释放 |
| `as_maybe_uninit_typed_slice()` | O(1) | O(1) | 仅指针转换 |
| `assume_init_typed_slice()` | O(1) | O(1) | 仅在调用方已证明初始化后重解释 |

**性能提示**:

- 减少 `ensure_capacity` 调用次数，尽量在初始化时分配足够容量
- 使用 `split_at` 递归分割避免多次分配
- 缓存复用：同一个 Workspace 可在多个操作间复用

---

## 11. no_std 兼容性

| 依赖 | 来源 | no_std 兼容 |
|------|------|:-----------:|
| `core::ptr::NonNull` | core | ✅ |
| `core::sync::atomic::AtomicU8` | core | ✅ |
| `core::fmt` | core | ✅ |
| `alloc::alloc::alloc` | alloc | ✅ |
| `alloc::alloc::dealloc` | alloc | ✅ |
| `alloc::alloc::Layout` | alloc | ✅ |

所有依赖均在 `core` 或 `alloc` 中；若目标平台提供原子操作，则兼容 `no_std + alloc + atomics`（参见 `01-architecture.md §6`）。若无原子支持，需改用更窄的借用模型或目标专用实现。

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
```

```rust
// Conditional export
#[cfg(feature = "std")]
impl std::error::Error for WorkspaceError {}
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.1.1 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-08 |
| 1.2.2 | 2026-04-10 |

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
