# 临时工作空间模块设计

> 文档编号: 24 | 模块: `src/workspace.rs` | 阶段: Phase 4
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
| 借用语义 | 借出期间不可再次借出，归还后可复用 |
| 单向增长 | 只扩容不缩容，避免内存抖动 |
| 不保证初始化 | 性能优先，调用方自行初始化使用的数据 |
| O(1) 分割 | 仅指针算术，无内存分配 |
| 显式生命周期 | 不绑定线程，调用方负责线程安全 |

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

工作空间模块是独立的，不依赖张量类型，可被上游库直接使用。

---

## 2. 文件位置

```
src/
└── workspace.rs    # 临时工作空间（独立模块）
```

单文件设计：工作空间功能内聚性高，代码量 ~400 行，无需拆分。

---

## 3. 依赖关系

### 3.1 依赖图

```
src/workspace.rs
├── core                # ptr::NonNull, marker, sync::atomic, fmt
├── alloc               # alloc::alloc, alloc::dealloc, alloc::Layout
└── crate::error        # WorkspaceError（可选，也可在本模块定义）
```

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `core` | `NonNull<u8>`, `PhantomData`, `AtomicU8`, `fmt::Debug`, `fmt::Display` |
| `alloc` | `alloc()`, `dealloc()`, `Layout` |
| `error` | `WorkspaceError`（或本模块内定义） |

### 3.3 依赖方向声明

> **依赖方向：单向。** `workspace` 仅依赖 `core` 和 `alloc`，不依赖 `tensor`。上游库和 `tensor` 可消费 `workspace`。

---

## 4. 公共 API 设计

### 4.1 Workspace 结构体

```rust
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU8, Ordering};
use alloc::alloc::{alloc, dealloc, Layout};

/// 临时工作空间。
///
/// 用于存储数值计算中的临时缓冲区，支持对齐分配和复用。
///
/// # 生命周期规则
///
/// - 借用期间不可再次借出（由借用守卫保证）
/// - 归还后可复用
/// - 不绑定线程，线程安全由调用方保证
///
/// # 初始化保证
///
/// 不保证零初始化。调用方须自行初始化使用的数据。
///
/// # Example
///
/// ```
/// let mut ws = Workspace::new(1024, 64)?;
///
/// // 可变借用
/// let mut buf = ws.borrow_mut()?;
/// // 使用缓冲区...
///
/// // 归还（RAII 自动）
/// drop(buf);
///
/// // 可再次借用
/// let buf2 = ws.borrow_mut()?;
/// ```
pub struct Workspace {
    /// 数据指针（非空，已对齐）。
    ptr: NonNull<u8>,

    /// 当前容量（字节）。
    capacity: usize,

    /// 分配时的对齐值（字节）。
    alignment: usize,

    /// 借用状态（原子）。
    ///
    /// - 0: 未借用
    /// - 1: 共享借用
    /// - 2: 独占借用
    borrow_state: AtomicU8,
}
```

> **设计决策：** 使用 `AtomicU8` 管理借用状态而非 `Mutex`，原因：无锁（`no_std` 兼容）、状态简单（仅需 3 个值）。

### 4.2 常量

```rust
impl Workspace {
    /// 默认对齐值：64 字节（AVX-512 缓存行）。
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// 最小对齐值。
    pub const MIN_ALIGNMENT: usize = 8;

    /// 默认初始容量：4 KB。
    pub const DEFAULT_CAPACITY: usize = 4096;

    /// 借用状态常量。
    const BORROW_NONE: u8 = 0;
    const BORROW_SHARED: u8 = 1;
    const BORROW_EXCLUSIVE: u8 = 2;

    /// 增长因子分子（1.5 倍）。
    const GROWTH_FACTOR_NUMERATOR: usize = 3;
    const GROWTH_FACTOR_DENOMINATOR: usize = 2;
}
```

### 4.3 构造方法

```rust
impl Workspace {
    /// 创建新的工作空间。
    ///
    /// # Arguments
    ///
    /// * `capacity` - 初始容量（字节）
    /// * `alignment` - 对齐值（字节），须为 2 的幂且 ≥ 8
    ///
    /// # Errors
    ///
    /// - `WorkspaceError::AllocFailed`: 内存分配失败
    /// - `WorkspaceError::InvalidLayout`: 布局参数无效
    ///
    /// # Panics
    ///
    /// `alignment` 不是 2 的幂或小于 `MIN_ALIGNMENT` 时 panic。
    ///
    /// # Example
    ///
    /// ```
    /// let ws = Workspace::new(1024, 64)?;
    /// assert!(ws.capacity() >= 1024);
    /// ```
    pub fn new(capacity: usize, alignment: usize) -> Result<Self, WorkspaceError> {
        assert!(alignment.is_power_of_two(), "alignment must be power of 2");
        assert!(alignment >= Self::MIN_ALIGNMENT, "alignment too small");

        let size = capacity.max(1);
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| WorkspaceError::InvalidLayout)?;

        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(WorkspaceError::AllocFailed)?;

        Ok(Self {
            ptr,
            capacity: size,
            alignment,
            borrow_state: AtomicU8::new(Self::BORROW_NONE),
        })
    }

    /// 使用默认参数创建工作空间。
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

// 禁止 Clone（语义上唯一）
```

### 4.5 借用 API

```rust
/// 不可变借用守卫。
///
/// RAII 守卫， drop 时自动归还工作空间。
pub struct WorkspaceBorrow<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

/// 可变借用守卫。
///
/// RAII 守卫, drop 时自动归还工作空间。
pub struct WorkspaceBorrowMut<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

impl Workspace {
    /// 不可变借用工作空间。
    ///
    /// # Errors
    ///
    /// `WorkspaceError::AlreadyBorrowed`: 工作空间已被借用。
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

    /// 可变借用工作空间。
    ///
    /// # Errors
    ///
    /// `WorkspaceError::AlreadyBorrowed`: 工作空间已被借用。
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
    /// 返回数据指针。
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// 返回数据切片。
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr valid for 'a, len matches capacity
 unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// 返回借用长度。
    pub fn len(&self) -> usize { self.len }

    /// 检查是否为空。
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl<'a> WorkspaceBorrowMut<'a> {
    /// 返回可变数据指针。
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// 返回可变数据切片。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr valid for 'a, len matches capacity, unique access
 unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// 类型化访问。
    ///
    /// # Safety
    ///
    /// 调用方须保证：
    /// - 对齐满足类型要求
    /// - 容量足够容纳 `count` 个元素
    pub unsafe fn as_typed_slice<T>(&mut self, count: usize) -> &mut [T] {
        assert!(count * core::mem::size_of::<T>() <= self.len);
        assert!(self.ptr.as_ptr() as usize % core::mem::align_of::<T>() == 0);
        core::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
    }
}

// RAII 归还
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
/// 分割后的子空间借用守卫。
///
/// 与 `WorkspaceBorrowMut` 类似，但允许多个分割守卫同时存在
/// （指向不重叠的内存区域）。
pub struct SplitBorrowMut<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

impl Workspace {
    /// 在指定位置分割工作空间为两个子空间。
    ///
    /// # Arguments
    ///
    /// * `mid` - 分割点（字节偏移）
    ///
    /// # Returns
    ///
    /// 两个子空间的可变借用守卫 `(left, right)`。
    ///
    /// # Complexity
    ///
    /// O(1) — 仅指针算术，无内存分配。
    ///
    /// # Errors
    ///
    /// - `WorkspaceError::SplitOutOfBounds`: `mid > capacity`
    /// - `WorkspaceError::AlreadyBorrowed`: 已被借用
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
            return Err(WorkspaceError::SplitOutOfBounds);
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

        let left_ptr = self.ptr;
        // SAFETY: mid <= capacity, so ptr + mid is within allocation
        let right_ptr = unsafe {
            NonNull::new_unchecked(self.ptr.as_ptr().add(mid))
        };

        Ok((
            SplitBorrowMut { ptr: left_ptr, len: mid, workspace: self },
            SplitBorrowMut { ptr: right_ptr, len: self.capacity - mid, workspace: self },
        ))
    }
}

impl<'a> SplitBorrowMut<'a> {
    /// 返回可变切片。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr valid, len matches allocation
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// 继续分割（递归二分）。
    ///
    /// O(1) — 仅指针算术。
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> Result<(SplitBorrowMut<'_>, SplitBorrowMut<'_>), WorkspaceError> {
        if mid > self.len {
            return Err(WorkspaceError::SplitOutOfBounds);
        }

        let left_ptr = self.ptr;
        let right_ptr = unsafe {
            NonNull::new_unchecked(self.ptr.as_ptr().add(mid))
        };

        Ok((
            SplitBorrowMut { ptr: left_ptr, len: mid, workspace: self.workspace },
            SplitBorrowMut { ptr: right_ptr, len: self.len - mid, workspace: self.workspace },
        ))
    }

    /// 返回子空间长度。
    pub fn len(&self) -> usize { self.len }
}

// Drop 不修改 borrow_state（由最外层 WorkspaceBorrowMut 负责）
impl<'a> Drop for SplitBorrowMut<'a> {
    fn drop(&mut self) {
        // 无操作：借用状态由原始 split_at 调用产生的守卫管理
    }
}
```

### 4.8 扩容 API

```rust
impl Workspace {
    /// 确保容量至少为 `min_capacity`。
    ///
    /// 如果当前容量不足，将重新分配更大的内存。
    /// 新容量 = max(请求容量, 当前容量 × 1.5)。
    ///
    /// # Errors
    ///
    /// - `WorkspaceError::AlreadyBorrowed`: 工作空间已被借用
    /// - `WorkspaceError::AllocFailed`: 内存分配失败
    ///
    /// # Example
    ///
    /// ```
    /// let mut ws = Workspace::new(1024, 64)?;
    /// ws.ensure_capacity(2048)?;  // 扩容到至少 2048
    /// ```
    pub fn ensure_capacity(
        &mut self,
        min_capacity: usize,
    ) -> Result<(), WorkspaceError> {
        if min_capacity <= self.capacity {
            return Ok(());
        }

        // 检查借用状态
        let state = self.borrow_state.load(Ordering::Acquire);
        if state != Self::BORROW_NONE {
            return Err(WorkspaceError::AlreadyBorrowed);
        }

        // 1.5 倍增长
        let grown = self.capacity * Self::GROWTH_FACTOR_NUMERATOR
            / Self::GROWTH_FACTOR_DENOMINATOR;
        let new_capacity = grown.max(min_capacity);

        self.reallocate(new_capacity)
    }

    /// 内部重新分配。
    fn reallocate(&mut self, new_capacity: usize) -> Result<(), WorkspaceError> {
        let new_layout = Layout::from_size_align(new_capacity, self.alignment)
            .map_err(|_| WorkspaceError::InvalidLayout)?;

        let new_ptr = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_ptr)
            .ok_or(WorkspaceError::AllocFailed)?;

        // 复制旧数据
        // SAFETY: src and dst do not overlap, copy min(old, new) bytes
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                new_ptr.as_ptr(),
                self.capacity.min(new_capacity),
            );
        }

        // 释放旧内存
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
// Good - 使用 split_at 分割工作空间
let mut ws = Workspace::new(1024, 64)?;
let (mut left, mut right) = ws.split_at(512)?;
// left 和 right 指向不重叠的内存区域
// 可安全并行使用

// Bad - 直接操作裸指针绕过借用检查
let mut ws = Workspace::new(1024, 64)?;
let ptr = ws.ptr.as_ptr();  // ptr 字段是私有的!
// 绕过了借用检查，可能导致数据竞争

// Good - 确保容量后使用
let mut ws = Workspace::new(256, 64)?;
ws.ensure_capacity(1024)?;  // 先扩容
let mut buf = ws.borrow_mut()?;
// 安全使用更大的缓冲区

// Bad - 借用期间扩容
let mut ws = Workspace::new(256, 64)?;
let buf = ws.borrow_mut()?;
ws.ensure_capacity(1024);  // 编译错误！ borrow_mut 需要 &mut self
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

### 5.3 扩容安全性论证

**扩容期间保证不违反已有引用安全性**：

1. `ensure_capacity` 需要 `&mut self`，编译器保证无其他引用
2. 方法内部显式检查 `borrow_state` 是否为 NONE
3. 扩容后旧指针失效（`dealloc`），新指针更新
4. 由于 `&mut self` 保证独占，无悬挂引用

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

- [ ] **T1**: 定义 `WorkspaceError` 枚举和 `Workspace` 结构体
  - 文件: `src/workspace.rs`
  - 内容: `WorkspaceError` 枚举、`Workspace` 结构体、常量定义
  - 测试: `test_workspace_error_display`, `test_workspace_constants`
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: 实现构造方法 `new`、`with_default_capacity`
  - 文件: `src/workspace.rs`
  - 内容: `new()`、`with_default_capacity()`、`allocate()` 内部方法
  - 测试: `test_workspace_new`, `test_workspace_new_default`
  - 前置: T1
  - 预计: 10 min

### Wave 2: 借用机制

- [ ] **T3**: 实现 `Drop` 和借用守卫类型
  - 文件: `src/workspace.rs`
  - 内容: `WorkspaceBorrow`、`WorkspaceBorrowMut` 结构体和 `Drop` 实现
  - 测试: `test_workspace_drop_no_leak`, `test_workspace_borrow_drop_return`
  - 前置: T2
  - 预计: 15 min

- [ ] **T4**: 实现 `borrow`/`borrow_mut` 方法和守卫方法
  - 文件: `src/workspace.rs`
  - 内容: `borrow()`、`borrow_mut()`、`as_slice()`、`as_mut_slice()`、`as_typed_slice()`
  - 测试: `test_borrow_basic`, `test_borrow_mut_basic`, `test_borrow_double_fails`
  - 前置: T3
  - 预计: 10 min

### Wave 3: 分割和扩容

- [ ] **T5**: 实现 `split_at` 和 `SplitBorrowMut`
  - 文件: `src/workspace.rs`
  - 内容: `split_at()`、`SplitBorrowMut` 结构体及 `split_at_mut()` 递归
  - 测试: `test_split_at_basic`, `test_split_at_recursive`, `test_split_at_oob`
  - 前置: T4
  - 预计: 15 min

- [ ] **T6**: 实现扩容策略 `ensure_capacity`/`reallocate`
  - 文件: `src/workspace.rs`
  - 内容: `ensure_capacity()`、`reallocate()`
  - 测试: `test_ensure_capacity_no_grow`, `test_ensure_capacity_grow`, `test_ensure_capacity_while_borrowed_fails`
  - 前置: T2
  - 预计: 10 min

### Wave 4: 集成和文档

- [ ] **T7**: 编写模块导出和文档注释
  - 文件: `src/workspace.rs`
  - 内容: 公共导出、完整文档注释、使用示例
  - 测试: `cargo doc` 通过
  - 前置: T5, T6
  - 预计: 10 min

### 并行执行图

```
Wave 1: [T1] [T2]
            │
Wave 2: [T3] [T4]
            │
Wave 3: [T5] [T6]
            │
Wave 4: [T7]
```

---

## 7. 测试计划

### 7.1 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_workspace_new_basic` | 匇定容量和对齐创建工作空间 | 高 |
| `test_workspace_new_default` | 默认参数创建 | 高 |
| `test_workspace_new_invalid_alignment` | 非法对齐值 panic | 高 |
| `test_workspace_drop_no_leak` | Drop 后内存正确释放 | 中 |
| `test_borrow_basic` | 不可变借用和切片访问 | 高 |
| `test_borrow_mut_basic` | 可变借用和类型化访问 | 高 |
| `test_borrow_double_fails` | 重复借用返回错误 | 高 |
| `test_borrow_after_drop` | 归还后可再次借用 | 高 |
| `test_split_at_basic` | 匇定位置分割 | 高 |
| `test_split_at_recursive` | 递归分割（多级） | 中 |
| `test_split_at_oob` | 越界分割返回错误 | 高 |
| `test_ensure_capacity_no_grow` | 宎量足够时不扩容 | 高 |
| `test_ensure_capacity_grow` | 容量不足时扩容到 1.5 倍 | 高 |
| `test_ensure_capacity_while_borrowed_fails` | 借用期间扩容失败 | 高 |
| `test_alignment_verification` | 对齐值验证 | 中 |
| `test_typed_slice_alignment` | 类型化切片对齐检查 | 高 |

### 7.2 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 零容量工作空间 | `new(0, 64)` 返回最小容量 1 的工作空间 |
| 最小对齐（8 字节） | 正常创建和使用 |
| 大容量（1MB+） | 正常分配和释放 |
| 递归分割到空子空间 | `split_at(0)` 返回空左半 |
| `ensure_capacity(0)` | 无操作（容量已足够） |

### 7.3 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `capacity() >= new()` 请求的容量 | 随机容量 |
| `split_at` 后 `left.len + right.len == capacity` | 随机分割点 |
| 借用后 `borrow_state != NONE` | 篡借检查 |
| 扩容后对齐不变 | `alignment()` 一致 |

---

## 8. 与其他模块的交互

| 交互点 | 方向 | 说明 |
|--------|------|------|
| 上游库 → workspace | 上游 BLAS 库通过 `borrow_mut()` 获取缓冲区 |
| 上游库 → workspace | 上游 FFT 库通过 `split_at()` 分割工作空间 |
| tensor -> workspace | Tensor 操作通过 workspace 分配临时空间（可选） |

---

## 9. 设计决策记录(ADR)

### 决策 1：设计选择 - workspace vs arena vs pool

| 属性 | 值 |
|------|-----|
| 决策 | 使用简单的 workspace 类型而非 arena 或 pool 分配器 |
| 理由 | 实现简单（~400 行）、语义清晰（借用/归还）、满足需求（对齐/分割/扩容）；arena 分配器更复杂，pool 颸附加管理生命周期困难 |
| 替代方案 | arena 分配器（bump 分配） — 放弃，需求不复杂，无需 zone/reset |
| 替代方案 | pool 分配（对象池） — 放弃，工作空间操作原始字节，无需对象语义 |

### 决策 2:借用期间不可再次借出

| 属性 | 值 |
|------|-----|
| 决策 | 借出期间禁止再次借出（由 AtomicU8 CAS 保证） |
| 理由 | 安全性：避免同一缓冲区被多次借出导致数据竞争；简单性: 单一借用模型更易理解 |
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
| `as_typed_slice()` | O(1) | O(1) | 仅指针转换 |

**性能提示**:

- 减少 `ensure_capacity` 调用次数，尽量在初始化时分配足够容量
- 使用 `split_at` 递归分割避免多次分配
- 缓存复用：同一个 Workspace 可在多个操作间复用

---

## 11. no_std 其容性

| 依赖 | 来源 | no_std 兼容 |
|------|------|:-----------:|
| `core::ptr::NonNull` | core | ✅ |
| `core::sync::atomic::AtomicU8` | core | ✅ |
| `core::fmt` | core | ✅ |
| `alloc::alloc::alloc` | alloc | ✅ |
| `alloc::alloc::dealloc` | alloc | ✅ |
| `alloc::alloc::Layout` | alloc | ✅ |

所有依赖均在 `core` 或 `alloc` 中，完全兼容 `no_std`。

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
```

```rust
// 条件导出
#[cfg(feature = "std")]
impl std::error::Error for WorkspaceError {}
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
