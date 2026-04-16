# 临时工作空间模块设计

> 文档编号: 24 | 模块: `src/workspace/` | 阶段: Phase 4
> 前置文档: `05-storage.md`
> 需求参考: `需求说明书 §26`, `需求说明书 §27`, `需求说明书 §28.1`, `需求说明书 §28.2`
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责     | 包含                                            | 不包含                                       |
| -------- | ----------------------------------------------- | -------------------------------------------- |
| 对齐分配 | 使用 `alloc::alloc` 进行指定对齐的内存分配      | arena 分配器（更复杂的分配策略）             |
| 平台约束 | 当前版本依赖 `std` 环境下的分配与原子能力       | 池分配（pooled allocation）                  |
| 分割     | `split_at_mut` 将工作空间 O(1) 分割为两个子空间 | 栈分配（stack allocation，由调用方自行管理） |
| 动态扩容 | `ensure_capacity` 支持单向增长（不缩容）        | 自动缩容策略                                 |

### 1.2 设计原则

| 原则         | 体现                                                                                                                    |
| ------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 借用语义     | 借出期间不可再次借出；只读借用当前版本同一时刻最多一个活跃 guard，归还后可复用                                          |
| 单向增长     | 只扩容不缩容，避免内存抖动                                                                                              |
| 未初始化感知 | 底层字节视为 `MaybeUninit<u8>`；只有调用方显式声明初始化完成后，才能获取已初始化视图                                    |
| O(1) 分割    | 仅指针算术，无内存分配                                                                                                  |
| 显式生命周期 | 当前实现默认不可跨线程传递（`!Send + !Sync`）；这是为简化借用安全性论证采取的实现选择，而非 `需求说明书 §26` 的强制要求 |

### 1.3 在架构中的位置

```
Dependency layers:

L1: dimension, element, complex
L2: workspace  <- current module (independent of tensor)

Upstream libraries:
  upstream numeric libraries ──→ workspace
  tensor (optional) ───────────→ workspace
```

Workspace 模块位于 L2，独立于核心张量类型系统，可被上游库直接使用而无需引入 tensor 依赖（参见 `01-architecture.md §5`）。

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                                                       |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| 需求映射 | `需求说明书 §26`, `需求说明书 §27`, `需求说明书 §28.1`, `需求说明书 §28.2`                                 |
| 范围内   | 对齐分配、借用守卫、`split_at_mut` 递归分割、`ensure_capacity` 单向增长，以及 `MaybeUninit` scratch 语义。 |
| 范围外   | 持久化分配、custom allocators、arena / pool 策略与跨线程共享工作空间。                                     |
| 非目标   | 不把 workspace 设计成通用分配器框架，不新增第三方 allocator / sync 依赖。                                  |

| 需求条款         | 本文承接方式                                                                                                                             |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 需求说明书 §26 临时工作空间 | 定义对齐分配、借用守卫、`split_at_mut()` 与 `ensure_capacity()` 的行为边界。                                                             |
| 需求说明书 §27 错误处理     | 公开入口统一使用 `crate::error::Result<_>`；其错误值为 `XenonError::Workspace { reason: String }`，覆盖长度、对齐、ZST、越界和扩容溢出。 |
| 需求说明书 §28.1 文档       | 关键 API 提供文档和示例；非完整可编译示例统一标记 `ignore`。                                                                             |
| 需求说明书 §28.2 测试       | 测试计划覆盖单元、集成、边界、类型边界与借用状态机路径。                                                                                 |

---

## 3. 文件位置

```
src/
└── workspace/               # Temporary workspace (directory module)
    ├── mod.rs               # Module root, re-exports
    ├── error.rs             # WorkspaceError enum
    ├── workspace.rs         # Workspace struct, constants, constructors, destructor
    ├── borrow.rs            # WorkspaceBorrow and WorkspaceBorrowMut borrow guards
    ├── split.rs             # SplitBorrowMut split guard
    └── expand.rs            # ensure_capacity and reallocate growth
```

多文件设计：按职责拆分，便于后续扩展（如新增借用策略、分配策略等）。

### 3.1 文件职责

| 文件           | 职责                                                                                        | 预估行数 |
| -------------- | ------------------------------------------------------------------------------------------- | -------- |
| `mod.rs`       | 模块根，re-exports 所有公共类型                                                             | ~20      |
| `error.rs`     | `WorkspaceError` 枚举及 Display/Error impl                                                  | ~40      |
| `workspace.rs` | `Workspace` 结构体、常量、`new()`、`with_default_capacity()`、`Drop`                        | ~100     |
| `borrow.rs`    | `WorkspaceBorrow`、`WorkspaceBorrowMut` 及其方法和 Drop                                     | ~120     |
| `split.rs`     | `SplitBorrowMut` 及其方法（`as_maybe_uninit_slice`、`len`、顶层/递归 `split_at_mut`、Drop） | ~100     |
| `expand.rs`    | `ensure_capacity()`、`reallocate()` 扩容逻辑                                                | ~60      |

---

## 4. 依赖关系

### 4.1 依赖图

```
src/workspace/
├── mod.rs          # Re-exports: Workspace, WorkspaceBorrow, WorkspaceBorrowMut, SplitBorrowMut
├── error.rs        # Defines internal WorkspaceError; consumed by workspace.rs/split.rs/expand.rs and mapped at the public boundary through crate::error
├── workspace.rs    # Depends on error
├── borrow.rs       # Depends on workspace (by reference)
├── split.rs        # Depends on workspace (by reference) and error
└── expand.rs       # Depends on workspace (&mut self) and error

External dependencies:
├── core            # ptr::NonNull, marker, sync::atomic, fmt
├── alloc           # alloc::alloc, alloc::dealloc, alloc::Layout
└── crate::error      # XenonError public boundary mapping
```

### 4.2 类型级依赖

| 来源模块             | 使用的类型/trait                                                       |
| -------------------- | ---------------------------------------------------------------------- |
| `core`               | `NonNull<u8>`, `PhantomData`, `AtomicU8`, `fmt::Debug`, `fmt::Display` |
| `alloc`              | `alloc()`, `dealloc()`, `Layout`                                       |
| `workspace/error.rs` | `WorkspaceError`（模块内错误分类）                                     |
| `crate::error`       | `XenonError`（公开 Xenon API 边界包装）                                |

### 4.3 依赖方向声明

> **依赖方向：单向。** `workspace` 仅依赖 `core`、`alloc`、原子能力、模块内 `WorkspaceError`，以及 `crate::error` 的公开错误边界封装，不依赖 `tensor`（参见 `07-tensor.md §3`）。上游库和 `tensor` 可消费 `workspace`。

### 4.4 依赖合法性与替代方案

| 项目           | 说明                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                            |
| 合法性结论     | 合法；当前设计仅复用 Xenon 既有模块、标准库以及文档中已声明的项目内可选能力。 |
| 替代方案       | 不适用；当前范围内无需额外第三方依赖。                                        |

### 4.5 WorkspaceError 的独立性

`WorkspaceError` 保留为 workspace 模块内部使用的错误分类，用于表达分配失败、借用冲突、分割越界等局部语义；`mod.rs` 不对外 re-export 该类型。当错误穿过 Xenon 的公开 API 边界时，须包装为 `XenonError::Workspace { reason }` 这样的 opaque 公开变体。因此本文档中的 `new()`、`borrow()`、`borrow_mut()`、`split_at_mut()`、`ensure_capacity()` 等公开入口在最终对外签名上统一使用 `crate::error::Result<_>`（即 `26-error.md` 定义的统一结果别名），而不是直接暴露 `WorkspaceError`。

> **与 XenonError 的关系**: `WorkspaceError` 仍作为内部分类存在，以避免 workspace 模块内部丢失领域语义；但公开 Xenon API 不直接暴露它，`mod.rs` 的公共接口也不包含它，而是统一返回包装后的 `XenonError::Workspace { reason }`。参见 `26-error.md`。
>
> **诊断一致性说明**: `需求说明书 §27` 倾向公开恢复性错误携带结构化上下文；当前 `XenonError::Workspace { reason }` 仍是 opaque 包装，而更细的字段保留在模块内部 `WorkspaceError`。是否把这些字段提升到公开错误模型属于统一错误枚举的设计变更，需与 `26-error.md` 一并评审，本次仅记录该边界。

> **与线程安全需求的边界**: workspace 不是 `需求说明书 §10` 中张量 storage mode 的一部分，而是独立的上游缓冲区工具。`!Send + !Sync` 为当前实现选择，非 `需求说明书 §26` 的强制要求。采用此限制是为了简化借用安全性论证；未来版本可考虑放宽为 `Send`（需配合安全的跨线程借用检查）。

> **初始化语义约定**: Workspace 的底层缓冲区始终按“可能未初始化”建模。公共 API 默认只暴露 `MaybeUninit` 视图；只有调用方能够证明某一前缀或某一 typed region 已被完整写入时，才允许通过 `assume_init_*` 系列 unsafe API 将其解释为已初始化视图。

```rust,ignore
use alloc::borrow::Cow;

/// Internal error type for workspace operations.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WorkspaceError {
    /// Allocation failed (out of memory or invalid layout).
    AllocFailed { size: usize, align: usize },
    /// The requested alignment is not a power of 2.
    InvalidLayout { size: usize, align: usize, reason: Cow<'static, str> },
    /// Cannot borrow: workspace is already mutably borrowed.
    AlreadyBorrowed,
    /// Split point is out of bounds.
    SplitOutOfBounds { split: usize, len: usize },
}
```

---

## 5. 公共 API 设计

### 5.1 Workspace 结构体

````rust,ignore
use core::ptr::NonNull;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU8, Ordering};
use alloc::alloc::{alloc, dealloc, Layout};
use crate::error::{Result, XenonError};

/// Temporary workspace.
///
/// Used for storing temporary buffers in numerical computation,
/// supporting aligned allocation and reuse.
///
/// # Lifetime Rules
///
/// - Cannot be re-borrowed while borrowed (enforced by borrow guards)
/// - Can be reused after returning
/// - The current implementation is not transferable across threads
///   (`!Send + !Sync`), which simplifies the borrow-safety argument around raw
///   pointers. This is an implementation choice rather than a `需求说明书 §26`
///   mandate; future versions may relax it with safe cross-thread borrow checks.
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
    /// - 1: read guard
    /// - 2: exclusive borrow
    borrow_state: AtomicU8,

    /// Active split count. Incremented by split_at_mut(), decremented by SplitBorrowMut::drop().
    /// Only when this reaches 0 is borrow_state reset to BORROW_NONE.
    split_count: core::sync::atomic::AtomicUsize,

    /// Marker that intentionally prevents auto-deriving `Send`/`Sync`.
    _not_send_sync: PhantomData<*mut ()>,
}
````

> **设计决策：** 使用 `AtomicU8` 管理借用状态而非 `Mutex`，原因：无锁、状态简单（仅需 3 个值）。当前版本默认运行于 `std` 环境，因此直接依赖标准平台提供的原子与分配能力。

> **设计备注：** 当前实现使用 `AtomicU8`/`AtomicUsize` 管理借用状态。由于 `Workspace` 当前实现选择为 `!Send + !Sync`，理论上可以使用 `Cell`/`RefCell` 替代以简化实现。选择原子操作是为了在未来版本可能支持跨线程借用检查时减少迁移成本。

### 5.2 常量

```rust,ignore
impl Workspace {
    /// Default alignment: 64 bytes (AVX-512 cache line).
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// Minimum alignment.
    pub const MIN_ALIGNMENT: usize = 8;

    /// Default initial capacity: 4 KB.
    pub const DEFAULT_CAPACITY: usize = 4096;

    /// Borrow state constants.
    const BORROW_NONE: u8 = 0;
    const BORROW_READ: u8 = 1;
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

### 5.3 构造方法

> **错误公开边界**：`WorkspaceError` 为模块内部类型，不对外暴露。公开 API 统一通过 `XenonError::Workspace { reason: String }` 报告错误，详见 `26-error.md`。
>
> **结果类型说明：** 公开 API 统一使用 `Result<T, XenonError>`，`crate::error::Result<_>` 为等价类型别名。

````rust,ignore
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
    /// - `XenonError::Workspace { reason }`: Memory allocation failed or layout parameters invalid
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ws = Workspace::new(1024, 64)?;
    /// assert!(ws.capacity() >= 1024);
    /// ```
    pub fn new(capacity: usize, alignment: usize) -> Result<Self> {
        if !alignment.is_power_of_two() || alignment < Self::MIN_ALIGNMENT {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: alignment must be a power of two and >= MIN_ALIGNMENT".into(),
            });
        }

        let size = capacity.max(1);
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| XenonError::Workspace {
                reason: "invalid workspace layout: Layout::from_size_align rejected the requested size/alignment".into(),
            })?;

        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(XenonError::Workspace {
            reason: "workspace allocation failed".into(),
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
    pub fn with_default_capacity() -> Result<Self> {
        Self::new(Self::DEFAULT_CAPACITY, Self::DEFAULT_ALIGNMENT)
    }
}
````

### 5.4 析构方法

```rust,ignore
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

### 5.5 借用 API

```rust,ignore
/// Immutable borrow guard.
///
/// RAII guard that automatically returns the workspace on drop.
pub struct WorkspaceBorrow<'a> {
    /// Start pointer of the borrowed scratch region.
    ptr: NonNull<u8>,
    /// Length of the borrowed region in bytes.
    len: usize,
    /// Parent workspace whose borrow state is released on drop.
    workspace: &'a Workspace,
}

/// Mutable borrow guard.
///
/// RAII guard that automatically returns the workspace on drop.
pub struct WorkspaceBorrowMut<'a> {
    /// Start pointer of the borrowed scratch region.
    ptr: NonNull<u8>,
    /// Length of the borrowed region in bytes.
    len: usize,
    /// Parent workspace whose borrow state is released on drop.
    workspace: &'a Workspace,
}

impl Workspace {
    /// Acquire the workspace for read-only inspection of the scratch region.
    ///
    /// # Errors
    ///
    /// `XenonError::Workspace { reason }`: Workspace is already borrowed.
    ///
    /// Note: the current design allows at most one active read guard at a time.
    /// This keeps the runtime state machine simple and matches the workspace's
    /// temporary-scratch-buffer positioning. The returned guard still models the
    /// bytes as potentially uninitialized; use `assume_init_slice` only when the
    /// caller can prove the inspected prefix has been written.
    ///
    /// `borrow()`/`borrow_mut()` take `&self` rather than `&mut self` because
    /// exclusivity is enforced at runtime by the internal `AtomicU8` state
    /// machine. Concurrent or overlapping borrow attempts are rejected by
    /// returning a `XenonError::Workspace { reason }` value.
    pub fn borrow(&self) -> Result<WorkspaceBorrow<'_>> {
        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_READ,
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if prev.is_err() {
            return Err(XenonError::Workspace {
                reason: "workspace already borrowed".into(),
            });
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
    /// `XenonError::Workspace { reason }`: Workspace is already borrowed.
    pub fn borrow_mut(&self) -> Result<WorkspaceBorrowMut<'_>> {
        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_EXCLUSIVE,
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if prev.is_err() {
            return Err(XenonError::Workspace {
                reason: "workspace already borrowed".into(),
            });
        }

        Ok(WorkspaceBorrowMut {
            ptr: self.ptr,
            len: self.capacity,
            workspace: self,
        })
    }
}
```

### 5.6 借用守卫方法

```rust,ignore
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
    pub unsafe fn assume_init_slice(
        &self,
        initialized_len: usize,
    ) -> Result<&[u8]> {
        if initialized_len > self.len {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: initialized_len exceeds borrow length".into(),
            });
        }
        Ok(core::slice::from_raw_parts(self.ptr.as_ptr(), initialized_len))
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
    /// The caller must still uphold the initialization model for `T`; directly
    /// checkable size/alignment/count violations are reported as `Result` errors.
    pub unsafe fn as_maybe_uninit_typed_slice<T>(
        &mut self,
        count: usize,
    ) -> Result<&mut [core::mem::MaybeUninit<T>]> {
        if core::mem::size_of::<T>() == 0 {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: zero-sized types are not supported for typed workspace borrows".into(),
            });
        }
        let byte_len = count
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(XenonError::Workspace {
                reason: "invalid workspace layout: count * size_of::<T>() overflowed".into(),
            })?;
        if byte_len > self.len {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: typed slice byte length exceeds borrow length".into(),
            });
        }
        if self.ptr.as_ptr() as usize % core::mem::align_of::<T>() != 0 {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: workspace pointer does not satisfy T alignment".into(),
            });
        }
        Ok(core::slice::from_raw_parts_mut(
            self.ptr.as_ptr() as *mut core::mem::MaybeUninit<T>,
            count,
        ))
    }

    /// Interprets the first `count` elements as initialized `T` values.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that:
    /// - the first `count` typed elements have been fully initialized,
    /// - those values are valid instances of `T`,
    /// - the requested region fits in this borrow (`count * size_of::<T>() <= self.len()`),
    /// - and the scratch region satisfies `T` alignment requirements.
    pub unsafe fn assume_init_typed_slice<T>(
        &mut self,
        count: usize,
    ) -> Result<&mut [T]> {
        if core::mem::size_of::<T>() == 0 {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: zero-sized types are not supported for typed workspace borrows".into(),
            });
        }
        let byte_len = count
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(XenonError::Workspace {
                reason: "invalid workspace layout: count * size_of::<T>() overflowed".into(),
            })?;
        if byte_len > self.len {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: typed slice byte length exceeds borrow length".into(),
            });
        }
        if self.ptr.as_ptr() as usize % core::mem::align_of::<T>() != 0 {
            return Err(XenonError::Workspace {
                reason: "invalid workspace layout: workspace pointer does not satisfy T alignment".into(),
            });
        }
        Ok(core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count))
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

### 5.7 分割 API

````rust,ignore
/// Borrow guard for a split sub-space.
///
/// Similar to `WorkspaceBorrowMut`, but allows multiple split guards
/// to coexist (pointing to non-overlapping memory regions).
/// Public surface intentionally matches the minimal split use case:
/// `as_maybe_uninit_slice()`, `len()`, `split_at_mut()`, and `Drop`. This type deliberately does not
/// expose `as_mut_ptr()`, `is_empty()`, or typed `assume_init_*` helpers until
/// their aliasing and initialization contracts are separately frozen.
pub struct SplitBorrowMut<'a> {
    /// Start pointer of this split sub-space.
    ptr: NonNull<u8>,
    /// Length of this split sub-space in bytes.
    len: usize,
    /// Parent workspace whose borrow state is restored when all splits drop.
    workspace: &'a Workspace,
    /// Reference to the split count. The split operation atomically
    /// increments the SPLIT_COUNT counter. Each SplitBorrowMut holds a
    /// reference to this counter. Drop decrements the counter; only when
    /// the counter reaches 0 is borrow_state reset to BORROW_NONE.
    split_count: &'a core::sync::atomic::AtomicUsize,
}

impl Workspace {
    /// Split the workspace mutably at the specified position into two sub-spaces.
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
    /// - `XenonError::Workspace { reason }`: `mid > capacity` or already borrowed
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ws = Workspace::new(1024, 64)?;
    /// let (left, right) = ws.split_at_mut(512)?;
    /// // left: [0, 512), right: [512, 1024)
    /// ```
    pub fn split_at_mut(
        &self,
        mid: usize,
    ) -> Result<(SplitBorrowMut<'_>, SplitBorrowMut<'_>)> {
        if mid > self.capacity {
            return Err(XenonError::Workspace {
                reason: format!("split out of bounds: split={}, len={}", mid, self.capacity),
            });
        }

        let prev = self.borrow_state.compare_exchange(
            Self::BORROW_NONE,
            Self::BORROW_EXCLUSIVE,
            Ordering::Acquire,
            Ordering::Relaxed,
        );

        if prev.is_err() {
            return Err(XenonError::Workspace {
                reason: "workspace already borrowed".into(),
            });
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
    ) -> Result<(SplitBorrowMut<'a>, SplitBorrowMut<'a>)> {
        if mid > self.len {
            return Err(XenonError::Workspace {
                reason: format!("split out of bounds: split={}, len={}", mid, self.len),
            });
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
/// Uses reference counting: each top-level `split_at_mut()` sets split_count to the number
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
/// Any pair of active split guards — including recursively produced descendants —
/// must always cover disjoint, non-overlapping byte ranges.
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
````

### 5.8 扩容 API

````rust,ignore
impl Workspace {
    /// Ensure capacity is at least `min_capacity`.
    ///
    /// If current capacity is insufficient, a larger memory region will be allocated.
    /// New capacity = max(requested capacity, current capacity × 1.5).
    ///
    /// Growth may preserve existing bytes in the current implementation, but
    /// callers must not rely on that behavior.
    /// After growth, all previous views and borrows are invalidated. Treat the
    /// entire scratch region as unspecified until it is re-initialized.
    /// Growth only guarantees that capacity satisfies the new request and that
    /// alignment remains unchanged.
    ///
    /// # Errors
    ///
    /// - `XenonError::Workspace { reason }`: Workspace is already borrowed or memory allocation failed
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ws = Workspace::new(1024, 64)?;
    /// ws.ensure_capacity(2048)?;  // Grow to at least 2048
    /// ```
    pub fn ensure_capacity(
        &mut self,
        min_capacity: usize,
    ) -> Result<()> {
        if min_capacity <= self.capacity {
            return Ok(());
        }

        // Check borrow state
        let state = self.borrow_state.load(Ordering::Acquire);
        if state != Self::BORROW_NONE {
            return Err(XenonError::Workspace {
                reason: "workspace already borrowed".into(),
            });
        }

        // 1.5x growth
        // Growth-factor arithmetic must use checked_mul to avoid usize overflow.
        // On overflow, return XenonError::Workspace { reason }
        // at the public boundary rather than
        // panicking or silently wrapping.
        let grown = self.capacity
            .checked_mul(Self::GROWTH_FACTOR_NUMERATOR)
            .ok_or(XenonError::Workspace {
                reason: "workspace allocation failed: capacity growth overflow".into(),
            })?
            / Self::GROWTH_FACTOR_DENOMINATOR;
        let new_capacity = grown.max(min_capacity);

        self.reallocate(new_capacity)
    }

    /// Internal reallocation.
    fn reallocate(&mut self, new_capacity: usize) -> Result<()> {
        let new_layout = Layout::from_size_align(new_capacity, self.alignment)
            .map_err(|_| XenonError::Workspace {
                reason: "invalid workspace layout: Layout::from_size_align rejected the requested size/alignment during reallocate".into(),
            })?;

        let new_ptr = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_ptr)
            .ok_or(XenonError::Workspace {
                reason: "workspace allocation failed during reallocate".into(),
            })?;

        // The implementation may copy old bytes during growth, but callers must
        // not rely on content preservation. All previous views and borrows are
        // invalid after reallocation, and the scratch region must be treated as
        // unspecified until re-initialized.
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
````

> **错误映射说明：** `reallocate()` 使用统一的 `Result<()>` 别名作为模块内实现签名，内部 `WorkspaceError` 通过 `XenonError::Workspace { reason }` 做统一包装；不得让 `WorkspaceError` 直接穿透公开 API。

> **扩容语义说明：** `ensure_capacity()` / `reallocate()` 的公开契约仅保证扩容后容量不小于请求值且对齐保持不变。当前实现可能复制旧缓冲区中的字节，但调用方不得依赖内容被保留；扩容后所有旧视图与借用均失效，必须把整个 scratch 区域重新视为 unspecified 状态，并在重新初始化后再通过 `assume_init_*` 系列 API 解释为已初始化数据。

### 5.9 Good/Bad 对比

```rust,ignore
// Good - Split workspace using split_at_mut
let mut ws = Workspace::new(1024, 64)?;
let (mut left, mut right) = ws.split_at_mut(512)?;
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
let bytes: &mut [u8] = unsafe { buf.assume_init_typed_slice::<u8>(1024)? };
// Still requires the unsafe initialization proof; direct input validation errors use Result.

// Bad - Directly manipulating raw pointers to bypass borrow checking
let mut ws = Workspace::new(1024, 64)?;
let ptr = ws.ptr.as_ptr();  // ptr field is private!
// Bypasses borrow checking, may cause data races

// Good - Ensure capacity before use
let mut ws = Workspace::new(256, 64)?;
ws.ensure_capacity(1024)?;  // Grow first
let mut buf = ws.borrow_mut()?;
// Safe to use the larger buffer

// Bad - Re-enter a borrow-only API while a guard is still alive
let ws = Workspace::new(256, 64)?;
let _buf = ws.borrow_mut()?;
let _again = ws.split_at_mut(128)?;  // Returns `XenonError::Workspace { reason }` at runtime
```

---

## 6. 内部实现设计

### 6.1 对齐分配实现

```
Workspace memory layout (64-byte aligned)

Address:  0x00           0x40           0x80           0xC0
          ├──────────────┼──────────────┼──────────────┼──────────────┤
Data:     |  scratch  |  scratch  |  scratch  |  scratch  |
           |  buffer  |  buffer  |  buffer  |  buffer  |
           └────────────┴────────────┴────────────┴────────────┘
           │<───────────────── capacity ─────────────────>│

          ↑
          ptr (NonNull<u8>)

Alignment check: (addr % 64) == 0 ✓
```

### 6.2 split_at_mut O(1) 原理

```
Traditional approach (requires allocation):
┌─────────────────────────────────────┐
│ Workspace [1024 bytes]              │
└─────────────────────────────────────┘
         │
         ▼ Allocate a new Workspace (512 bytes)
┌──────────────────┐  ┌──────────────────┐
│ Left [512 bytes] │  │ Right [512 bytes]│
│ (separate alloc)  │  │ (separate alloc)  │
└──────────────────┘  └──────────────────┘
         O(n) memory copy ❌

This design (zero allocation):
┌─────────────────────────────────────┐
│ Workspace [1024 bytes]              │
│ ptr = 0x1000                        │
└─────────────────────────────────────┘
         │
         ▼ Pointer arithmetic only
┌──────────────────┐  ┌──────────────────┐
│ SplitBorrowMut   │  │ SplitBorrowMut   │
│ ptr = 0x1000     │  │ ptr = 0x1200     │
│ len = 512        │  │ len = 512        │
│ (view, no alloc)  │  │ (view, no alloc)  │
└──────────────────┘  └──────────────────┘
         O(1) ✓
```

### 6.3 split_at_mut 安全设计

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
>
> 1. `split_at_mut(512)` → `split_count = 2`，创建 `left` 和 `right`
> 2. `right.split_at_mut(128)` → `fetch_add(1)`，`split_count = 3`，创建 `right_a` 和 `right_b`

> **公开/内部边界说明：** 本文中的公开安全 API（如 `new`、`borrow`、`borrow_mut`、`split_at_mut`、`ensure_capacity`）统一返回可恢复错误，而不是把输入校验失败暴露为 panic；只有无法直接验证的 `unsafe` 初始化前提继续由调用方承担。3. `left` drop → `split_count: 3→2`，`prev=3 ≠ 1`，不重置 ✅ 4. `right_a` drop → `split_count: 2→1`，`prev=2 ≠ 1`，不重置 ✅ 5. `right_b` drop → `split_count: 1→0`，`prev=1`，重置 `borrow_state` ✅

### 6.4 扩容安全性论证

**扩容期间保证不违反已有引用安全性**（参见 `05-storage.md §5`）：

1. `ensure_capacity` 需要 `&mut self`，编译器保证无其他引用
2. 方法内部显式检查 `borrow_state` 是否为 NONE
3. 扩容后旧指针失效（`dealloc`），新指针更新
4. 由于 `&mut self` 保证独占，无悬挂引用
5. 任何 `MaybeUninit` / `assume_init_*` 视图都不得跨越 `ensure_capacity` 保留

```
Growth flow:
ensure_capacity(&mut self, 2048)
    │
    ├── 1. Check borrow_state == NONE  ✓
    ├── 2. Allocate new memory (2048 bytes)
    ├── 3. copy_nonoverlapping old → new
    ├── 4. dealloc old memory
    └── 5. Update ptr and capacity
```

---

## 7. 实现任务拆分

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

- [ ] **T5**: 实现 `split_at_mut` 和 `SplitBorrowMut`
  - 文件: `src/workspace/split.rs`
  - 内容: `SplitBorrowMut` 结构体、顶层 `split_at_mut()`、递归 `split_at_mut()`、`Drop` 实现
  - 测试: `test_split_at_mut_basic`, `test_split_at_mut_recursive`, `test_split_at_mut_oob`
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
Wave 2:      [T3]    [T2]             <- T2 and T3 run in parallel (both depend on T1)
                     ╱  |  ╲
Wave 3:           [T4] [T5] [T6]      <- T4, T5, and T6 run in parallel (all depend on T2)
                     ╲  |  ╱
Wave 4:               [T7]            <- depends on T4, T5, and T6 all being complete
```

> **关键路径**: T1 → T2 → T4/T5/T6（最长） → T7。T3 不在关键路径上，可在任何时间完成。

---

## 8. 测试计划

### 8.1 测试分类表

| 测试分类 | 位置                     | 说明                                                          |
| -------- | ------------------------ | ------------------------------------------------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证工作空间分配、借用、分割与扩容语义                        |
| 集成测试 | `tests/`                 | 验证 `workspace` 与 `ffi`、上游 scratch-buffer 场景的协同路径 |
| 边界测试 | 同模块测试中标注         | 覆盖零容量、最小对齐、大容量和递归分割等边界                  |
| 属性测试 | `tests/property/`        | 验证借用状态机、typed borrow 字节长度与分割不变量             |

### 8.2 单元测试清单

| 测试函数                                       | 测试内容                                                                      | 优先级 |
| ---------------------------------------------- | ----------------------------------------------------------------------------- | ------ |
| `test_workspace_new_basic`                     | 指定容量和对齐创建工作空间                                                    | 高     |
| `test_workspace_new_default`                   | 默认参数创建                                                                  | 高     |
| `test_workspace_new_invalid_alignment`         | 非法对齐值返回 `XenonError::Workspace { reason }`，且 `reason` 提供可诊断文本 | 高     |
| `test_workspace_drop_no_leak`                  | Drop 后内存正确释放                                                           | 中     |
| `test_borrow_basic`                            | 不可变借用和 `MaybeUninit` 切片访问                                           | 高     |
| `test_borrow_mut_basic`                        | 可变借用和 `MaybeUninit` 类型化访问                                           | 高     |
| `test_borrow_double_fails`                     | 重复借用返回错误                                                              | 高     |
| `test_borrow_after_drop`                       | 归还后可再次借用                                                              | 高     |
| `test_assume_init_requires_initialized_prefix` | 已初始化视图只覆盖调用方证明已初始化的前缀                                    | 高     |
| `test_split_at_mut_basic`                      | 固定位置分割                                                                  | 高     |
| `test_split_at_mut_recursive`                  | 递归分割（多级）                                                              | 中     |
| `test_split_at_mut_oob`                        | 越界分割返回错误                                                              | 高     |
| `test_ensure_capacity_no_grow`                 | 容量足够时不扩容                                                              | 高     |
| `test_ensure_capacity_grow`                    | 容量不足时扩容到 1.5 倍                                                       | 高     |
| `test_ensure_capacity_while_borrowed_fails`    | 借用期间扩容失败                                                              | 高     |
| `test_alignment_verification`                  | 对齐值验证                                                                    | 中     |
| `test_typed_slice_alignment`                   | 类型化切片对齐检查                                                            | 高     |
| `test_workspace_not_send_not_sync`             | `Workspace` 的 `!Send + !Sync` 编译期负向验证                                 | 高     |
| `test_split_borrow_mut_not_send_not_sync`      | `SplitBorrowMut` 的 `!Send + !Sync` 编译期负向验证                            | 高     |
| `test_recursive_split_drop_order_independent`  | 递归 split 以任意 drop 顺序归还时都能正确复用工作空间                         | 中     |

### 8.3 边界测试场景

| 场景                 | 预期行为                                  |
| -------------------- | ----------------------------------------- |
| 零容量工作空间       | `new(0, 64)` 返回最小容量 1 的工作空间    |
| 最小对齐（8 字节）   | 正常创建和使用                            |
| 大容量（1MB+）       | 正常分配和释放                            |
| 递归分割到空子空间   | `split_at_mut(0)` 返回空左半              |
| `ensure_capacity(0)` | 无操作（容量已足够）                      |
| 未初始化区域只读访问 | 仅允许 `MaybeUninit` 视图，不暴露 `&[u8]` |

### 8.4 属性测试不变量

| 不变量                                               | 测试方法                |
| ---------------------------------------------------- | ----------------------- |
| `capacity() >= new()` 请求的容量                     | 随机容量                |
| `split_at_mut` 后 `left.len + right.len == capacity` | 随机分割点              |
| 借用后 `borrow_state != NONE`                        | 借用状态检查            |
| 扩容后对齐不变                                       | `alignment()` 一致      |
| 未初始化区域只能通过 `MaybeUninit` 视图暴露          | 随机借用 + API 形状检查 |

### 8.5 集成测试

| 测试文件                  | 测试内容                                                                                                                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_workspace.rs` | `new` / `borrow` / `split_at_mut` / `ensure_capacity` 与 `ffi`、上游 scratch-buffer 场景的端到端协同验证，并验证 `MaybeUninit` 视图与 `assume_init_*` 的 `Result` + unsafe 前缀约束 |

### 8.6 Feature gate / 配置测试

| 配置              | 验证点                                                                                                                                        |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 默认配置          | workspace 在默认构建下验证当前实现采用 `!Send + !Sync`，并保持 MaybeUninit 暴露与借用状态机语义；该约束仅验证现状，不构成未来版本的稳定保证。 |
| 其他 feature 组合 | 不适用；当前模块无额外 feature gate。                                                                                                         |

### 8.7 类型边界 / 编译期测试

| 场景                                                     | 测试方式                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------ |
| `Workspace` 当前不实现 `Send` / `Sync`                   | 编译期测试，验证当前实现选择；若未来放宽，该测试基线可随版本调整。 |
| `SplitBorrowMut` 当前不实现 `Send` / `Sync`              | 编译期测试，验证当前实现选择；若未来放宽，该测试基线可随版本调整。 |
| typed borrow 拒绝 ZST 与不对齐区域                       | 编译期 / 运行时错误测试。                                          |
| custom allocator 与 persistent allocation 不属于当前 API | API 缺失断言。                                                     |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向                             | 对方模块             | 接口/类型         | 约定                                                                                                                          |
| -------------------------------- | -------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `workspace ← upstream libraries` | `upstream libraries` | 临时缓冲区请求    | 上游数值库可把 workspace 作为临时缓冲区使用；默认只获得 `MaybeUninit` 视图                                                    |
| `workspace ← upstream libraries` | `upstream libraries` | `split_at_mut()`  | 上游数值库场景可通过分割接口拆分工作空间；子空间同样只暴露 `MaybeUninit` 视图                                                 |
| `workspace ← upstream libraries` | `upstream libraries` | `assume_init_*`   | 调用方必须先完成写入并证明对应前缀/typed region 已初始化，才能重解释为已初始化视图                                            |
| `workspace ← tensor`             | `tensor`             | 临时 scratch 空间 | 张量运算在需要时可借用 workspace 提供临时空间；任何借出的 scratch 视图都不得跨越 `ensure_capacity` 或持久化到 tensor 元数据中 |

### 9.2 数据流描述

```text
Upper-layer code requests temporary scratch space
    │
    ├── Workspace::new(capacity, alignment)
    │       └── allocates a 64-byte-aligned raw buffer
    │
    ├── borrow() / borrow_mut() / split_at_mut()
    │       └── creates thread-local MaybeUninit guards or split sub-spaces
    │
    ├── the caller writes temporary data during the guard lifetime
    │       └── assume_init_* is used only after initialization is proven
    │
    └── dropping the guard restores borrow_state for later reuse
```

---

## 10. 错误处理与语义边界

| 主题              | 内容                                                                                                                                                                                                                                                                             |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recoverable error | `new()` / `ensure_capacity()` / `borrow*()` / `split_at_mut()` 失败时统一返回 `XenonError::Workspace { reason }`；`reason` 由内部 `WorkspaceError` 归一化生成，并保留容量、对齐、分割点或借用状态等诊断文本；typed helper 的 ZST、长度和对齐输入错误也通过同一公开错误边界报告。 |
| Panic             | 不为公开 API 输入校验引入 panic；`unsafe` 初始化前提若被违反，仍属于调用方责任范围内的 UB。                                                                                                                                                                                      |
| 路径一致性        | 当前仅有单一借用状态机与扩容路径；无 SIMD / 并行分支，所有 guard 释放规则必须保持一致。                                                                                                                                                                                          |
| 容差边界          | 不适用。                                                                                                                                                                                                                                                                         |

---

## 11. 设计决策记录(ADR)

### 决策 1：设计选择 - workspace vs arena vs pool

| 属性     | 值                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| 决策     | 使用简单的 workspace 类型而非 arena 或 pool 分配器                                                                    |
| 理由     | 实现简单（~400 行）、语义清晰（借用/归还）、满足需求（对齐/分割/扩容）；arena 分配器更复杂，pool 附加管理生命周期困难 |
| 替代方案 | arena 分配器（bump 分配） — 放弃，需求不复杂，无需 zone/reset                                                         |
| 替代方案 | pool 分配（对象池） — 放弃，工作空间操作原始字节，无需对象语义                                                        |

### 决策 2：借用期间不可再次借出

| 属性     | 值                                                                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 决策     | 借出期间禁止再次借出（由 AtomicU8 CAS 保证）                                                                                                      |
| 理由     | 安全性：避免同一缓冲区被多次借出导致数据竞争；简单性: 单一借用模型更易理解；`split_at_mut()` 生成的子空间全部释放后，父工作空间才恢复可借用状态。 |
| 替代方案 | 允许多个并发 read guard（多个 reader） — 未来可扩展，当前版本简化                                                                                 |

### 决策 3：扩容安全性保证

| 属性     | 值                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 决策     | 扩容前显式检查 borrow_state == NONE，且需要 `&mut self`                                  |
| 理由     | `&mut self` 由编译器保证独占访问；显式检查原子状态作为双重保障；防止扩容导致已有引用失效 |
| 替代方案 | 不检查直接扩容 — 放弃，UB 风险太高                                                       |

### 决策 4：不保证零初始化

| 属性     | 值                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------- |
| 决策     | 工作空间不保证零初始化                                                                            |
| 理由     | 性能: 零初始化是 O(n) 操作;不必要: 大多数场景下调用方会覆盖全部数据;与 C 一致: 与 malloc 行为一致 |
| 替代方案 | 构造时零初始化 — 放弃，性能损失                                                                   |

### 决策 5：Workspace 不实现 Send/Sync

| 属性     | 值                                                                                                                                                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 决策     | Workspace 当前实现不实现 Send 或 Sync；该结论记录当前实现选择，而非冻结为长期稳定保证                                                                                                                                                                                                      |
| 理由     | `!Send + !Sync` 为当前实现选择，目的是简化借用安全性论证。即使存在运行时借用状态检查，也暂不将其建模为可跨线程传递或共享的基础类型；若调用方需要多线程临时缓冲区，应在线程边界外自行分配和管理独立工作空间。相关测试仅验证当前行为，未来版本可在补充安全的跨线程借用检查后重新评估并放宽。 |
| 替代方案 | 放宽为 Send（并配套跨线程借用检查） — 未来可评估；仅依赖当前 AtomicU8 状态机不足以直接支持完整多线程语义                                                                                                                                                                                   |

---

## 12. 性能考量

| 操作                            | 时间复杂度 | 空间复杂度      | 说明                           |
| ------------------------------- | ---------- | --------------- | ------------------------------ |
| `new()`                         | O(1)       | O(capacity)     | 一次分配                       |
| `borrow()`                      | O(1)       | O(1)            | 原子 CAS                       |
| `borrow_mut()`                  | O(1)       | O(1)            | 原子 CAS                       |
| `split_at_mut()`                | O(1)       | O(1)            | 仅指针算术                     |
| `ensure_capacity()`             | O(n)       | O(new_capacity) | 分配 + 拷贝 + 释放             |
| `as_maybe_uninit_typed_slice()` | O(1)       | O(1)            | 仅指针转换                     |
| `assume_init_typed_slice()`     | O(1)       | O(1)            | 仅在调用方已证明初始化后重解释 |

**性能提示**:

- 减少 `ensure_capacity` 调用次数，尽量在初始化时分配足够容量
- 使用 `split_at_mut` 递归分割避免多次分配
- 缓存复用：同一个 Workspace 可在多个操作间复用

---

## 13. 平台与工程约束

| 约束       | 说明                                                  |
| ---------- | ----------------------------------------------------- |
| `std` only | 当前版本 workspace 设计以 `std` 环境为前提            |
| MSRV       | Rust 1.85+                                            |
| 单 crate   | workspace 保持在主 crate 内，不拆出独立 scratch crate |
| SemVer     | 借用状态机、分割语义和错误语义属于公开契约            |
| 最小依赖   | 不新增第三方分配器或同步依赖                          |

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
| 1.2.4 | 2026-04-14 |
| 1.2.5 | 2026-04-15 |
| 1.2.6 | 2026-04-16 |
| 1.2.7 | 2026-04-16 |
| 1.2.8 | 2026-04-16 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
