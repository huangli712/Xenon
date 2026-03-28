# 05-10 临时工作空间模块设计

> **模块路径**: `src/workspace.rs`  
> **版本**: v1.0  
> **日期**: 2026-03-28  
> **依赖**: `core`, `alloc`

---

## 1. 模块概述

### 1.1 工作空间在高性能计算中的角色

临时工作空间（Scratch Workspace）是高性能数值计算中的关键基础设施。在矩阵运算、FFT、卷积等操作中，经常需要临时缓冲区来存储中间结果。工作空间模块的设计目标是：

| 目标 | 说明 |
|------|------|
| **零分配热路径** | 预分配后可复用，避免频繁堆分配 |
| **SIMD 友好** | 默认 64 字节对齐，满足 AVX-512 要求 |
| **可预测性能** | 增长策略明确，无隐式缩容 |
| **组合式设计** | Scratch 需求可声明式组合，一次性分配 |
| **FFI 友好** | 提供原始指针，供上游库零开销集成 |

### 1.2 设计哲学

```
工作空间设计原则
├── 借用语义 — 借出期间不可再次借出，归还后可复用
├── 显式生命周期 — 不绑定线程，调用方负责线程安全
├── 不保证初始化 — 性能优先，调用方自行初始化
├── 单向增长 — 只扩容不缩容，避免内存抖动
└── O(1) 分割 — 支持递归二分，无额外分配
```

### 1.3 典型使用场景

| 场景 | 说明 |
|------|------|
| **BLAS 包装器** | matmul 需要 O(N²) 临时空间存储分块 |
| **FFT 库** | 蝶形运算需要 bit-reversal 缓冲区 |
| **卷积算法** | im2col 需要 O(K²×C×H×W) 临时空间 |
| **并行计算** | 每线程独立工作空间，避免竞争 |

### 1.4 在架构中的位置

```
依赖层级：

L1: error
          ↓
L2: workspace ←── error
          ↓
L4: tensor ←── workspace (可选依赖，供高级操作使用)
          
上游库:
  blas-wrapper ──→ workspace
  fft-lib ───────→ workspace
```

工作空间模块是独立的，不依赖张量类型，可被上游库直接使用。

---

## 2. 文件结构

```
src/workspace.rs        # 临时工作空间
├── Workspace           # 工作空间结构体
├── WorkspaceBorrow     # 借用守卫（RAII）
├── WorkspaceBorrowMut  # 可变借用守卫（RAII）
├── ScratchRequirement  # 内存需求描述
├── ScratchBuilder      # 需求组合构建器
└── 辅助 trait 和函数
```

---

## 3. Workspace 结构体设计

### 3.1 完整结构定义

```rust
use core::ptr::NonNull;
use core::marker::PhantomData;
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
/// ```ignore
/// // 创建工作空间
/// let mut ws = Workspace::new(1024, 64)?;
///
/// // 借用
/// let mut buf = ws.borrow_mut()?;
/// 
/// // 使用缓冲区
/// buf.as_mut_slice()[0] = 1.0;
///
/// // 归还（自动，RAII）
/// drop(buf);
///
/// // 可再次借用
/// let buf2 = ws.borrow_mut()?;
/// ```
pub struct Workspace {
    /// 数据指针（非空，已对齐）
    ptr: NonNull<u8>,
    
    /// 当前容量（字节）
    capacity: usize,
    
    /// 分配时的对齐值（字节）
    alignment: usize,
    
    /// 借用状态
    /// 
    /// - 0: 未借用
    /// - 1: 共享借用（borrow）
    /// - 2: 独占借用（borrow_mut）
    borrow_state: core::sync::atomic::AtomicU8,
}

// 常量
impl Workspace {
    /// 默认对齐值：64 字节（AVX-512 缓存行）
    pub const DEFAULT_ALIGNMENT: usize = 64;
    
    /// 最小对齐值
    pub const MIN_ALIGNMENT: usize = 8;
    
    /// 默认初始容量：4 KB
    pub const DEFAULT_CAPACITY: usize = 4096;
    
    /// 借用状态：未借用
    const BORROW_NONE: u8 = 0;
    
    /// 借用状态：共享借用
    const BORROW_SHARED: u8 = 1;
    
    /// 借用状态：独占借用
    const BORROW_EXCLUSIVE: u8 = 2;
}
```

### 3.2 内部表示示意

```
Workspace 内存布局（64 字节对齐）

地址:     0x00           0x40           0x80           0xC0
          ├──────────────┼──────────────┼──────────────┼──────────────┤
数据:     |  scratch  |  scratch  |  scratch  |  scratch  |
          |  buffer    |  buffer    |  buffer    |  buffer    |
          └────────────┴────────────┴────────────┴────────────┘
          │<───────────────── capacity ─────────────────>│
          
          ↑
          ptr (NonNull<u8>)

对齐检查: (addr % 64) == 0 ✓

元数据:
┌────────────────────────────────────────────────────┐
│ Workspace                                          │
├────────────────────────────────────────────────────┤
│ ptr: NonNull<u8>        ──→ 数据起始地址           │
│ capacity: usize         ──→ 已分配字节数           │
│ alignment: usize        ──→ 分配时的对齐值         │
│ borrow_state: AtomicU8  ──→ 借用状态（原子）       │
└────────────────────────────────────────────────────┘
```

### 3.3 构造方法

```rust
impl Workspace {
    /// 创建新的工作空间。
    ///
    /// # Arguments
    ///
    /// * `capacity` - 初始容量（字节）
    /// * `alignment` - 对齐值（字节），须为 2 的幂
    ///
    /// # Errors
    ///
    /// 如果内存分配失败，返回 `WorkspaceError::AllocFailed`。
    ///
    /// # Panics
    ///
    /// 如果 `alignment` 不是 2 的幂或小于 8，将 panic。
    pub fn new(capacity: usize, alignment: usize) -> Result<Self, WorkspaceError> {
        Self::allocate(capacity, alignment)
    }
    
    /// 使用默认参数创建工作空间。
    pub fn with_default_capacity() -> Result<Self, WorkspaceError> {
        Self::new(Self::DEFAULT_CAPACITY, Self::DEFAULT_ALIGNMENT)
    }
    
    /// 从 ScratchRequirement 创建工作空间。
    ///
    /// 自动计算满足需求的最小容量。
    pub fn from_requirement(req: &ScratchRequirement) -> Result<Self, WorkspaceError> {
        let capacity = req.size;
        let alignment = req.alignment;
        Self::new(capacity, alignment)
    }
    
    /// 分配内存的内部实现。
    fn allocate(capacity: usize, alignment: usize) -> Result<Self, WorkspaceError> {
        assert!(alignment.is_power_of_two(), "alignment must be power of 2");
        assert!(alignment >= Self::MIN_ALIGNMENT, "alignment too small");
        
        let size = capacity.max(1);
        
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| WorkspaceError::InvalidLayout)?;
        
        let ptr = unsafe { alloc(layout) };
        
        let ptr = NonNull::new(ptr)
            .ok_or(WorkspaceError::AllocFailed)?;
        
        Ok(Self {
            ptr,
            capacity: size,
            alignment,
            borrow_state: core::sync::atomic::AtomicU8::new(Self::BORROW_NONE),
        })
    }
}
```

### 3.4 析构方法

```rust
impl Drop for Workspace {
    fn drop(&mut self) {
        // 释放内存
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
// impl Clone for Workspace { ... } // 不实现
```

### 3.5 对齐分配实现细节

```rust
/// 对齐分配的关键实现
impl Workspace {
    /// 检查当前对齐是否满足要求。
    pub fn is_aligned_to(&self, align: usize) -> bool {
        let addr = self.ptr.as_ptr() as usize;
        addr % align == 0
    }
    
    /// 返回当前对齐值。
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    
    /// 返回当前容量。
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
```

---

## 4. 借用 API

### 4.1 借用守卫类型

```rust
/// 不可变借用守卫。
///
/// RAII 守卫，drop 时自动归还工作空间。
pub struct WorkspaceBorrow<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}

/// 可变借用守卫。
///
/// RAII 守卫，drop 时自动归还工作空间。
pub struct WorkspaceBorrowMut<'a> {
    ptr: NonNull<u8>,
    len: usize,
    workspace: &'a Workspace,
}
```

### 4.2 borrow/borrow_mut 签名

```rust
impl Workspace {
    /// 不可变借用工作空间。
    ///
    /// # Errors
    ///
    /// 如果工作空间已被借用（共享或独占），返回 `WorkspaceError::AlreadyBorrowed`。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ws = Workspace::new(1024, 64)?;
    /// let borrow = ws.borrow()?;
    /// // 使用 borrow.as_slice()...
    /// ```
    pub fn borrow(&self) -> Result<WorkspaceBorrow<'_>, WorkspaceError> {
        // 原子 CAS：期望 NONE，设置为 SHARED
        use core::sync::atomic::Ordering;
        
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
    /// 如果工作空间已被借用（共享或独占），返回 `WorkspaceError::AlreadyBorrowed`。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ws = Workspace::new(1024, 64)?;
    /// let mut borrow = ws.borrow_mut()?;
    /// borrow.as_mut_slice()[0] = 42;
    /// ```
    pub fn borrow_mut(&self) -> Result<WorkspaceBorrowMut<'_>, WorkspaceError> {
        use core::sync::atomic::Ordering;
        
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

### 4.3 借用守卫方法

```rust
impl<'a> WorkspaceBorrow<'a> {
    /// 返回数据指针。
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    
    /// 返回数据切片。
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
    
    /// 返回借用长度。
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// 检查是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a> WorkspaceBorrowMut<'a> {
    /// 返回可变数据指针。
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    
    /// 返回可变数据切片。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
    
    /// 类型化访问。
    ///
    /// # Safety
    ///
    /// 调用方须保证：
    /// - 对齐满足类型要求
    /// - 容量足够容纳 count 个元素
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
            core::sync::atomic::Ordering::Release,
        );
    }
}

impl<'a> Drop for WorkspaceBorrowMut<'a> {
    fn drop(&mut self) {
        self.workspace.borrow_state.store(
            Workspace::BORROW_NONE,
            core::sync::atomic::Ordering::Release,
        );
    }
}
```

### 4.4 生命周期图解

```
借用生命周期示意

时间线:
t0: let ws = Workspace::new(1024, 64)?;
    ┌─────────────────────────────────┐
    │ Workspace                       │
    │ borrow_state = NONE             │
    │ ptr ────────────────────────────┼──► [buffer]
    └─────────────────────────────────┘

t1: let borrow = ws.borrow()?;
    ┌─────────────────────────────────┐
    │ Workspace                       │
    │ borrow_state = SHARED ◄─────────┼── 借出中
    └─────────────────────────────────┘
    
    ┌─────────────────────────────────┐
    │ WorkspaceBorrow                 │
    │ ptr ────────────────────────────┼──► [buffer] (只读)
    │ workspace ──────────────────────┼──► ws
    └─────────────────────────────────┘

t2: let borrow2 = ws.borrow();  // Err(AlreadyBorrowed)
    // 借出期间不可再次借出

t3: drop(borrow);  // 自动归还
    ┌─────────────────────────────────┐
    │ Workspace                       │
    │ borrow_state = NONE ◄───────────┼── 已归还
    └─────────────────────────────────┘

t4: let borrow3 = ws.borrow_mut()?;  // OK
    // 归还后可复用
```

---

## 5. 分割 API

### 5.1 split_at 签名

```rust
impl Workspace {
    /// 在指定位置分割工作空间为两个子空间。
    ///
    /// # Arguments
    ///
    /// * `mid` - 分割点（字节偏移）
    ///
    /// # Returns
    ///
    /// 返回两个子空间的可变借用守卫 `(left, right)`。
    ///
    /// # Complexity
    ///
    /// O(1) — 仅指针算术，无内存分配。
    ///
    /// # Errors
    ///
    /// - `mid > capacity`: 返回 `WorkspaceError::SplitOutOfBounds`
    /// - 已被借用: 返回 `WorkspaceError::AlreadyBorrowed`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ws = Workspace::new(1024, 64)?;
    /// let (left, right) = ws.split_at(512)?;
    ///
    /// // left 和 right 指向不重叠的内存区域
    /// // left: [0, 512)
    /// // right: [512, 1024)
    /// ```
    pub fn split_at(
        &self,
        mid: usize,
    ) -> Result<(SplitBorrowMut<'_>, SplitBorrowMut<'_>), WorkspaceError> {
        if mid > self.capacity {
            return Err(WorkspaceError::SplitOutOfBounds);
        }
        
        use core::sync::atomic::Ordering;
        
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
        let right_ptr = unsafe {
            NonNull::new_unchecked(self.ptr.as_ptr().add(mid))
        };
        
        Ok((
            SplitBorrowMut {
                ptr: left_ptr,
                len: mid,
                workspace: self,
            },
            SplitBorrowMut {
                ptr: right_ptr,
                len: self.capacity - mid,
                workspace: self,
            },
        ))
    }
}
```

### 5.2 SplitBorrowMut 类型

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

impl<'a> SplitBorrowMut<'a> {
    /// 返回可变指针。
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    
    /// 返回可变切片。
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
    
    /// 继续分割（递归二分）。
    ///
    /// # Complexity
    ///
    /// O(1) — 仅指针算术。
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> Result<(SplitBorrowMut<'a>, SplitBorrowMut<'a>), WorkspaceError> {
        if mid > self.len {
            return Err(WorkspaceError::SplitOutOfBounds);
        }
        
        let left_ptr = self.ptr;
        let right_ptr = unsafe {
            NonNull::new_unchecked(self.ptr.as_ptr().add(mid))
        };
        
        // 注意：这里不需要修改 borrow_state
        // 因为父级已经持有独占借用
        
        Ok((
            SplitBorrowMut {
                ptr: left_ptr,
                len: mid,
                workspace: self.workspace,
            },
            SplitBorrowMut {
                ptr: right_ptr,
                len: self.len - mid,
                workspace: self.workspace,
            },
        ))
    }
    
    /// 返回子空间长度。
    pub fn len(&self) -> usize {
        self.len
    }
}

// Drop 不修改 borrow_state（由最外层 WorkspaceBorrowMut 负责）
impl<'a> Drop for SplitBorrowMut<'a> {
    fn drop(&mut self) {
        // 无操作
        // 借用状态由原始 split_at 调用的守卫管理
    }
}

// 特殊 trait：标记分割守卫不管理借用状态
trait Sealed {}
impl<'a> Sealed for SplitBorrowMut<'a> {}
```

### 5.3 O(1) 实现原理

```
split_at O(1) 原理

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

关键点:
1. 不复制数据，只计算新指针
2. 子空间是原空间的视图
3. 生命周期绑定到原 Workspace
4. 借用规则保证无别名冲突
```

### 5.4 递归分割示例

```rust
// 递归二分示例
fn example_recursive_split() -> Result<(), WorkspaceError> {
    let mut ws = Workspace::new(1024, 64)?;
    
    // 第一层分割
    let (mut left, mut right) = ws.split_at(512)?;
    
    // 第二层分割（左半部分）
    let (mut ll, mut lr) = left.split_at_mut(256)?;
    
    // 第三层分割（继续）
    let (mut lll, mut llr) = ll.split_at_mut(128)?;
    
    // 现在有 4 个不重叠的子空间:
    // lll: [0, 128)
    // llr: [128, 256)
    // lr:  [256, 512)
    // right: [512, 1024)
    
    // 可并行使用，无数据竞争
    lll.as_mut_slice()[0] = 1;
    llr.as_mut_slice()[0] = 2;
    lr.as_mut_slice()[0] = 3;
    right.as_mut_slice()[0] = 4;
    
    Ok(())
}
```

---

## 6. 增长策略

### 6.1 扩容算法

```rust
impl Workspace {
    /// 增长因子（约 1.5 倍）
    const GROWTH_FACTOR_NUMERATOR: usize = 3;
    const GROWTH_FACTOR_DENOMINATOR: usize = 2;
    
    /// 确保容量至少为 `min_capacity`。
    ///
    /// 如果当前容量不足，将重新分配更大的内存。
    /// 扩容后保持原有对齐。
    ///
    /// # Errors
    ///
    /// - `AlreadyBorrowed`: 工作空间已被借用
    /// - `AllocFailed`: 内存分配失败
    ///
    /// # Growth Strategy
    ///
    /// 新容量 = max(请求容量, 当前容量 × 1.5)
    pub fn ensure_capacity(
        &mut self,
        min_capacity: usize,
    ) -> Result<(), WorkspaceError> {
        if min_capacity <= self.capacity {
            return Ok(());
        }
        
        // 检查借用状态
        use core::sync::atomic::Ordering;
        let state = self.borrow_state.load(Ordering::Acquire);
        if state != Self::BORROW_NONE {
            return Err(WorkspaceError::AlreadyBorrowed);
        }
        
        // 计算新容量（1.5 倍增长）
        let new_capacity = {
            let grown = self.capacity * Self::GROWTH_FACTOR_NUMERATOR 
                       / Self::GROWTH_FACTOR_DENOMINATOR;
            grown.max(min_capacity)
        };
        
        // 重新分配
        self.reallocate(new_capacity)
    }
    
    /// 内部重新分配实现。
    fn reallocate(&mut self, new_capacity: usize) -> Result<(), WorkspaceError> {
        // 分配新内存
        let new_layout = Layout::from_size_align(new_capacity, self.alignment)
            .map_err(|_| WorkspaceError::InvalidLayout)?;
        
        let new_ptr = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_ptr)
            .ok_or(WorkspaceError::AllocFailed)?;
        
        // 复制旧数据（如果有）
        // 注意：不保证数据有效性，仅复制内存
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                new_ptr.as_ptr(),
                self.capacity.min(new_capacity),
            );
        }
        
        // 释放旧内存
        unsafe {
            let old_layout = Layout::from_size_align_unchecked(
                self.capacity,
                self.alignment,
            );
            dealloc(self.ptr.as_ptr(), old_layout);
        }
        
        // 更新元数据
        self.ptr = new_ptr;
        self.capacity = new_capacity;
        
        Ok(())
    }
}
```

### 6.2 不缩容的理由

| 理由 | 说明 |
|------|------|
| **避免抖动** | 频繁扩缩容导致内存碎片和性能不稳定 |
| **复用优势** | 保留大缓冲区供后续操作复用 |
| **简单性** | 单向增长策略更简单，减少边界情况 |
| **可预测性** | 用户可预测内存使用模式 |
| **显式控制** | 如需释放内存，可显式 drop 并重建 |

### 6.3 增长策略图解

```
容量变化示意

请求序列: 1024 → 2048 → 1536 → 4096

Step 1: 初始分配
capacity = 1024
┌────────────────────────────────────┐
│ [1024 bytes]                       │
└────────────────────────────────────┘

Step 2: 请求 2048 (需要扩容)
new_capacity = max(2048, 1024 * 1.5) = 2048
┌────────────────────────────────────┬────────────────────────────────────┐
│ [2048 bytes]                       │ (额外 1024 未使用)                 │
└────────────────────────────────────┴────────────────────────────────────┘

Step 3: 请求 1536 (无需扩容)
capacity = 2048 (不变)
┌────────────────────────────────────┬────────────────────────────────────┐
│ [1536 used]                        │ [512 预留]                         │
└────────────────────────────────────┴────────────────────────────────────┘

Step 4: 请求 4096 (需要扩容)
new_capacity = max(4096, 2048 * 1.5) = 4096
┌────────────────────────────────────┬────────────────────────────────────┐
│ [4096 bytes]                       │                                    │
└────────────────────────────────────┴────────────────────────────────────┘
```

---

## 7. ScratchRequirement 设计

### 7.1 完整结构定义

```rust
/// 结构化的内存需求描述。
///
/// 用于在执行操作前查询所需临时内存，支持声明式组合。
///
/// # 组合规则
///
/// - `sequential(a, b)`: 取 max，因为内存可复用
/// - `parallel(a, b)`: 取 sum，因为需要同时存在
///
/// # Example
///
/// ```ignore
/// // 单个操作的需求
/// let matmul_req = ScratchRequirement::new(1024, 64);
///
/// // 组合多个操作
/// let combined = ScratchRequirement::parallel([
///     matmul_req,
///     ScratchRequirement::new(512, 64),
/// ]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchRequirement {
    /// 所需字节数
    pub size: usize,
    
    /// 对齐要求（字节）
    pub alignment: usize,
    
    /// 可选的描述信息（用于调试）
    pub description: Option<&'static str>,
}

impl ScratchRequirement {
    /// 创建新的内存需求。
    ///
    /// # Arguments
    ///
    /// * `size` - 所需字节数
    /// * `alignment` - 对齐要求，须为 2 的幂
    pub const fn new(size: usize, alignment: usize) -> Self {
        Self {
            size,
            alignment,
            description: None,
        }
    }
    
    /// 创建带描述的内存需求。
    pub const fn with_description(
        size: usize,
        alignment: usize,
        description: &'static str,
    ) -> Self {
        Self {
            size,
            alignment,
            description: Some(description),
        }
    }
    
    /// 零需求（不需要额外内存）。
    pub const ZERO: Self = Self::new(0, 1);
    
    /// 创建指定数量元素的内存需求。
    pub fn for_elements<T>(count: usize) -> Self {
        Self::new(
            count * core::mem::size_of::<T>(),
            core::mem::align_of::<T>(),
        )
    }
}
```

### 7.2 对齐合并算法

```rust
impl ScratchRequirement {
    /// 合并对齐要求。
    ///
    /// 返回两个对齐值中较大的那个（满足两者）。
    #[inline]
    pub const fn merge_alignment(a: usize, b: usize) -> usize {
        if a >= b { a } else { b }
    }
    
    /// 合并两个需求（通用）。
    ///
    /// 使用自定义的 size 合并函数。
    #[inline]
    pub const fn merge_with<F>(a: Self, b: Self, size_merge: F) -> Self
    where
        F: Fn(usize, usize) -> usize,
    {
        Self {
            size: size_merge(a.size, b.size),
            alignment: Self::merge_alignment(a.alignment, b.alignment),
            description: None,
        }
    }
}
```

---

## 8. Scratch 组合逻辑

### 8.1 sequential（顺序执行，取 max）

```rust
impl ScratchRequirement {
    /// 顺序执行的内存需求。
    ///
    /// 多个操作顺序执行时，内存可复用，取最大需求。
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 操作 A 需要 1024 字节
    /// // 操作 B 需要 512 字节
    /// // 顺序执行只需要 1024 字节（B 可复用 A 的空间）
    /// let seq = ScratchRequirement::sequential([req_a, req_b]);
    /// assert_eq!(seq.size, 1024);
    /// ```
    pub fn sequential<I>(requirements: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        requirements.into_iter().fold(
            Self::ZERO,
            |acc, req| Self::merge_with(acc, req, usize::max),
        )
    }
    
    /// 顺序合并两个需求。
    #[inline]
    pub const fn then(self, other: Self) -> Self {
        Self {
            size: if self.size >= other.size { self.size } else { other.size },
            alignment: Self::merge_alignment(self.alignment, other.alignment),
            description: None,
        }
    }
}
```

### 8.2 parallel（并行执行，取 sum）

```rust
impl ScratchRequirement {
    /// 并行执行的内存需求。
    ///
    /// 多个操作并行执行时，需要同时占用内存，取总和。
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 操作 A 需要 1024 字节
    /// // 操作 B 需要 512 字节
    /// // 并行执行需要 1536 字节（两者同时存在）
    /// let par = ScratchRequirement::parallel([req_a, req_b]);
    /// assert_eq!(par.size, 1536);
    /// ```
    pub fn parallel<I>(requirements: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        requirements.into_iter().fold(
            Self::ZERO,
            |acc, req| Self::merge_with(acc, req, |a, b| {
                // 饱和加法，防止溢出
                a.saturating_add(b)
            }),
        )
    }
    
    /// 并行合并两个需求。
    #[inline]
    pub const fn and(self, other: Self) -> Self {
        Self {
            size: self.size.saturating_add(other.size),
            alignment: Self::merge_alignment(self.alignment, other.alignment),
            description: None,
        }
    }
}
```

### 8.3 组合逻辑算法图解

```
组合逻辑示例

场景: 矩阵运算流水线

操作 A: matmul(M=100, N=100, K=100)
  scratch_size = M * K * 8 + K * N * 8 = 80000 + 80000 = 160000 bytes

操作 B: transpose(M=100, N=100)
  scratch_size = M * N * 8 = 80000 bytes

操作 C: reduction(M=100, N=100)
  scratch_size = 0 bytes

Sequential 组合 (A → B → C):
┌─────────────────────────────────────────────────────────┐
│ Time ──────────────────────────────────────────────────►│
│                                                         │
│ ┌─────────┐   ┌─────────┐   ┌─────┐                    │
│ │    A    │ → │    B    │ → │  C  │                    │
│ │ 160000  │   │  80000  │   │  0  │                    │
│ └─────────┘   └─────────┘   └─────┘                    │
│                                                         │
│ max(160000, 80000, 0) = 160000 bytes                   │
└─────────────────────────────────────────────────────────┘

Parallel 组合 (A || B):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│ ┌─────────────────────┐                                │
│ │         A           │                                │
│ │      160000         │                                │
│ └─────────────────────┘                                │
│ ┌─────────────────────┐   (同时存在)                   │
│ │         B           │                                │
│ │       80000         │                                │
│ └─────────────────────┘                                │
│                                                         │
│ 160000 + 80000 = 240000 bytes                          │
└─────────────────────────────────────────────────────────┘

混合组合 ((A → B) || C):
┌─────────────────────────────────────────────────────────┐
│ sequential(A, B) = max(160000, 80000) = 160000         │
│ parallel(sequential(A, B), C) = 160000 + 0 = 160000    │
└─────────────────────────────────────────────────────────┘
```

### 8.4 ScratchBuilder 构建器

```rust
/// Scratch 需求构建器。
///
/// 提供链式 API 构建复杂的内存需求。
pub struct ScratchBuilder {
    requirements: alloc::vec::Vec<ScratchRequirement>,
    mode: CombineMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CombineMode {
    Sequential,
    Parallel,
}

impl ScratchBuilder {
    /// 创建新的构建器。
    pub fn new() -> Self {
        Self {
            requirements: alloc::vec::Vec::new(),
            mode: CombineMode::Sequential,
        }
    }
    
    /// 设置为顺序组合模式。
    pub fn sequential(mut self) -> Self {
        self.mode = CombineMode::Sequential;
        self
    }
    
    /// 设置为并行组合模式。
    pub fn parallel(mut self) -> Self {
        self.mode = CombineMode::Parallel;
        self
    }
    
    /// 添加一个需求。
    pub fn add(mut self, req: ScratchRequirement) -> Self {
        self.requirements.push(req);
        self
    }
    
    /// 添加指定大小的需求。
    pub fn add_size(self, size: usize, alignment: usize) -> Self {
        self.add(ScratchRequirement::new(size, alignment))
    }
    
    /// 构建最终需求。
    pub fn build(self) -> ScratchRequirement {
        match self.mode {
            CombineMode::Sequential => {
                ScratchRequirement::sequential(self.requirements)
            }
            CombineMode::Parallel => {
                ScratchRequirement::parallel(self.requirements)
            }
        }
    }
}

impl Default for ScratchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// 使用示例
fn example_builder() {
    let req = ScratchBuilder::new()
        .sequential()
        .add_size(1024, 64)
        .add_size(512, 64)
        .parallel()
        .add_size(256, 64)
        .build();
}
```

---

## 9. 与工作空间集成

### 9.1 从 ScratchRequirement 到 Workspace 分配

```rust
impl Workspace {
    /// 从 ScratchRequirement 创建工作空间。
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 定义操作需求
    /// let req = ScratchRequirement::parallel([
    ///     ScratchRequirement::new(1024, 64),
    ///     ScratchRequirement::new(512, 64),
    /// ]);
    ///
    /// // 一次性分配
    /// let ws = Workspace::from_requirement(&req)?;
    /// assert!(ws.capacity() >= 1536);
    /// assert!(ws.alignment() >= 64);
    /// ```
    pub fn from_requirement(req: &ScratchRequirement) -> Result<Self, WorkspaceError> {
        Self::new(req.size, req.alignment)
    }
    
    /// 确保工作空间满足需求。
    ///
    /// 如果当前容量或对齐不足，将重新分配。
    pub fn ensure_meets(&mut self, req: &ScratchRequirement) -> Result<(), WorkspaceError> {
        // 检查对齐
        if req.alignment > self.alignment {
            // 需要重新分配以满足更大对齐
            return self.reallocate_with_alignment(self.capacity, req.alignment);
        }
        
        // 检查容量
        self.ensure_capacity(req.size)
    }
    
    /// 带对齐的重新分配。
    fn reallocate_with_alignment(
        &mut self,
        new_capacity: usize,
        new_alignment: usize,
    ) -> Result<(), WorkspaceError> {
        // ... 类似 ensure_capacity 的实现
        todo!()
    }
}
```

### 9.2 Scratch trait 抽象

```rust
/// 定义操作的 scratch 内存需求。
///
/// 上游库可实现此 trait 来声明其内存需求。
pub trait ScratchSize {
    /// 返回操作的 scratch 内存需求。
    fn scratch_requirement() -> ScratchRequirement;
}

// 示例：矩阵乘法
struct MatmulOp {
    m: usize,
    n: usize,
    k: usize,
}

impl ScratchSize for MatmulOp {
    fn scratch_requirement() -> ScratchRequirement {
        // 静态需求（如果维度固定）
        ScratchRequirement::ZERO
    }
}

impl MatmulOp {
    /// 动态计算 scratch 需求。
    pub fn scratch_requirement_dynamic(&self) -> ScratchRequirement {
        let size = self.m * self.k * 8 + self.k * self.n * 8;
        ScratchRequirement::new(size, 64)
    }
}
```

### 9.3 零运行时开销保证

```rust
/// 静态 scratch 需求示例。
///
/// 所有计算在编译期完成，运行时零开销。
pub struct StaticScratch<const SIZE: usize, const ALIGN: usize>;

impl<const SIZE: usize, const ALIGN: usize> ScratchSize for StaticScratch<SIZE, ALIGN> {
    fn scratch_requirement() -> ScratchRequirement {
        ScratchRequirement::new(SIZE, ALIGN)
    }
}

// 使用示例
type MyOp = StaticScratch<1024, 64>;

fn use_static_scratch() {
    // 编译期已知
    const REQ: ScratchRequirement = ScratchRequirement::new(1024, 64);
    
    // 无运行时计算
    let ws = Workspace::new(REQ.size, REQ.alignment).unwrap();
}
```

---

## 10. no_std 兼容性

### 10.1 alloc 下的实现

```rust
// 条件编译
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

// Workspace 不依赖 std
// 使用 alloc::alloc 进行内存分配

impl Workspace {
    /// 在 no_std 环境下创建工作空间。
    #[cfg(not(feature = "std"))]
    pub fn new_no_std(capacity: usize, alignment: usize) -> Result<Self, WorkspaceError> {
        Self::allocate(capacity, alignment)
    }
}
```

### 10.2 依赖分析

| 依赖 | 来源 | no_std 兼容 |
|------|------|:-----------:|
| `alloc::alloc` | alloc crate | ✅ |
| `alloc::vec::Vec` | alloc crate | ✅ |
| `core::ptr::NonNull` | core | ✅ |
| `core::sync::atomic` | core | ✅ |
| `core::marker` | core | ✅ |

### 10.3 特性门控

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
```

```rust
// 条件导出
#[cfg(feature = "std")]
pub use std::error::Error;

#[cfg(not(feature = "std"))]
pub trait Error: core::fmt::Debug + core::fmt::Display {}
```

---

## 11. 与其他模块的交互

### 11.1 与上游库的集成示例

```rust
// 假设的上游 BLAS 包装库
mod blas_wrapper {
    use xenon::workspace::{Workspace, ScratchRequirement};
    
    /// SGEMM 操作。
    pub struct Sgemm {
        m: usize,
        n: usize,
        k: usize,
    }
    
    impl Sgemm {
        /// 计算 scratch 需求。
        pub fn scratch_requirement(&self) -> ScratchRequirement {
            // 需要 M*K + K*N 个 f32
            ScratchRequirement::for_elements::<f32>(self.m * self.k + self.k * self.n)
                .with_description(
                    self.m * self.k + self.k * self.n,
                    64,
                    "SGEMM scratch",
                )
        }
        
        /// 执行矩阵乘法。
        pub fn execute(&self, ws: &mut Workspace) -> Result<(), BlasError> {
            let mut borrow = ws.borrow_mut()?;
            let scratch = unsafe {
                borrow.as_typed_slice::<f32>(self.m * self.k + self.k * self.n)
            };
            
            // 使用 scratch 执行计算
            // ...
            
            Ok(())
        }
    }
}
```

### 11.2 与 Tensor 模块的交互

```rust
// 可选：与 Tensor 集成的高级 API
#[cfg(feature = "tensor")]
impl Workspace {
    /// 为 Tensor 操作分配 scratch 空间。
    pub fn for_tensor_op<A, D: Dimension>(
        &mut self,
        shape: &D,
    ) -> Result<TensorScratch<A>, WorkspaceError> {
        let elem_count = shape.size();
        let bytes = elem_count * core::mem::size_of::<A>();
        
        self.ensure_capacity(bytes)?;
        
        let mut borrow = self.borrow_mut()?;
        // ...
        todo!()
    }
}
```

### 11.3 与并行模块的交互

```rust
// 多线程工作空间管理
#[cfg(feature = "parallel")]
pub struct ThreadLocalWorkspace {
    inner: std::thread::LocalKey<Workspace>,
}

#[cfg(feature = "parallel")]
impl ThreadLocalWorkspace {
    /// 获取当前线程的工作空间。
    pub fn get_or_create(&self) -> Result<&Workspace, WorkspaceError> {
        // 使用 thread_local! 实现
        todo!()
    }
}
```

---

## 12. 实现任务分解

### 任务清单

| # | 任务 | 预估时间 | 依赖 | 产出 |
|---|------|----------|------|------|
| 1 | 定义 `Workspace` 结构体和常量 | 10 min | 无 | `workspace.rs` |
| 2 | 实现构造方法（`new`, `allocate`） | 10 min | T1 | `workspace.rs` |
| 3 | 实现 `Drop` 和内存释放 | 5 min | T2 | `workspace.rs` |
| 4 | 实现借用守卫（`WorkspaceBorrow`, `WorkspaceBorrowMut`） | 15 min | T3 | `workspace.rs` |
| 5 | 实现 `borrow`/`borrow_mut` 方法 | 10 min | T4 | `workspace.rs` |
| 6 | 实现 `split_at` 和 `SplitBorrowMut` | 15 min | T5 | `workspace.rs` |
| 7 | 实现增长策略（`ensure_capacity`, `reallocate`） | 15 min | T3 | `workspace.rs` |
| 8 | 实现 `ScratchRequirement` 和组合逻辑 | 15 min | 无 | `workspace.rs` |
| 9 | 实现 `ScratchBuilder` 构建器 | 10 min | T8 | `workspace.rs` |
| 10 | 实现与 `Workspace` 的集成方法 | 10 min | T7, T8 | `workspace.rs` |
| 11 | 编写单元测试 | 20 min | T1-T10 | `tests/workspace.rs` |
| 12 | 编写文档和示例 | 10 min | T1-T11 | `workspace.rs` |

### 任务依赖图

```
T1 ──→ T2 ──→ T3 ──┬──→ T4 ──→ T5 ──→ T6
                   │
                   └──→ T7 ─────────────┐
                                        │
T8 ──→ T9 ──────────────────────────────┼──→ T10 ──→ T11 ──→ T12
                                        │
                                        └───────────────────────┘
```

### 并行执行建议

- **Wave 1**: T1, T8（可独立开始）
- **Wave 2**: T2（依赖 T1）, T9（依赖 T8）
- **Wave 3**: T3, T4（依赖 T2）
- **Wave 4**: T5, T7（依赖 T3, T4）
- **Wave 5**: T6（依赖 T5）
- **Wave 6**: T10（依赖 T7, T9）
- **Wave 7**: T11（依赖所有实现任务）
- **Wave 8**: T12（依赖 T11）

---

## 13. 设计决策记录

### D1: 为什么借用期间不可再次借出？

**决策**: 工作空间借出期间禁止再次借出。

**理由**:
1. **安全性**: 避免同一缓冲区被多次借出导致数据竞争
2. **简单性**: 单一借用模型更易理解和正确使用
3. **替代方案**: 使用 `split_at` 分割为多个不重叠区域

### D2: 为什么不保证零初始化？

**决策**: 工作空间不保证零初始化。

**理由**:
1. **性能**: 零初始化是 O(n) 操作，对大缓冲区开销显著
2. **不必要**: 大多数场景下调用方会覆盖全部数据
3. **显式控制**: 需要零初始化时可显式调用 `fill(0)`
4. **与 C 一致**: 与 malloc 行为一致，FFI 友好

### D3: 为什么只扩容不缩容？

**决策**: 工作空间只扩容，不缩容。

**理由**:
1. **避免抖动**: 频繁扩缩容导致性能不稳定
2. **复用优势**: 保留大缓冲区供后续操作复用
3. **简单性**: 单向增长策略实现更简单
4. **显式控制**: 需要释放内存时可显式 drop 并重建

### D4: 为什么使用 1.5 倍增长因子？

**决策**: 增长因子为 1.5（3/2）。

**理由**:
1. **平衡**: 介于 2 倍（内存浪费）和 1.25 倍（频繁重分配）之间
2. **经典选择**: 与 `std::Vec` 的策略类似
3. **数学最优**: 某些分析表明 1.5 左右是最优增长因子

### D5: 为什么 split_at 是 O(1)？

**决策**: `split_at` 实现为 O(1) 复杂度。

**理由**:
1. **零分配**: 仅计算新指针，不分配内存
2. **视图模式**: 子空间是原空间的视图，非独立分配
3. **生命周期绑定**: 子空间生命周期绑定到原 Workspace

### D6: 为什么使用 AtomicU8 而非 Mutex？

**决策**: 使用 `AtomicU8` 管理借用状态。

**理由**:
1. **无锁**: 原子操作比互斥锁更轻量
2. **no_std 兼容**: `AtomicU8` 在 `core` 中可用
3. **简单状态**: 借用状态仅需 3 个值（NONE/SHARED/EXCLUSIVE）

### D7: 为什么 ScratchRequirement 是纯数据？

**决策**: `ScratchRequirement` 是简单的数据结构，无运行时行为。

**理由**:
1. **零运行时开销**: 所有计算在组合时完成
2. **可序列化**: 简单结构便于序列化和传递
3. **可组合**: 支持声明式组合多个需求

---

## 附录 A: API 快速参考

```rust
// === Workspace ===
impl Workspace {
    pub const DEFAULT_ALIGNMENT: usize = 64;
    pub const DEFAULT_CAPACITY: usize = 4096;
    
    pub fn new(capacity: usize, alignment: usize) -> Result<Self, WorkspaceError>;
    pub fn with_default_capacity() -> Result<Self, WorkspaceError>;
    pub fn from_requirement(req: &ScratchRequirement) -> Result<Self, WorkspaceError>;
    
    pub fn capacity(&self) -> usize;
    pub fn alignment(&self) -> usize;
    pub fn is_aligned_to(&self, align: usize) -> bool;
    
    pub fn borrow(&self) -> Result<WorkspaceBorrow<'_>, WorkspaceError>;
    pub fn borrow_mut(&self) -> Result<WorkspaceBorrowMut<'_>, WorkspaceError>;
    pub fn split_at(&self, mid: usize) 
        -> Result<(SplitBorrowMut<'_>, SplitBorrowMut<'_>), WorkspaceError>;
    
    pub fn ensure_capacity(&mut self, min_capacity: usize) -> Result<(), WorkspaceError>;
    pub fn ensure_meets(&mut self, req: &ScratchRequirement) -> Result<(), WorkspaceError>;
}

// === ScratchRequirement ===
impl ScratchRequirement {
    pub const ZERO: Self;
    
    pub const fn new(size: usize, alignment: usize) -> Self;
    pub const fn with_description(size: usize, alignment: usize, desc: &'static str) -> Self;
    pub fn for_elements<T>(count: usize) -> Self;
    
    pub fn sequential<I>(requirements: I) -> Self;
    pub fn parallel<I>(requirements: I) -> Self;
    pub const fn then(self, other: Self) -> Self;
    pub const fn and(self, other: Self) -> Self;
}

// === ScratchBuilder ===
impl ScratchBuilder {
    pub fn new() -> Self;
    pub fn sequential(self) -> Self;
    pub fn parallel(self) -> Self;
    pub fn add(self, req: ScratchRequirement) -> Self;
    pub fn add_size(self, size: usize, alignment: usize) -> Self;
    pub fn build(self) -> ScratchRequirement;
}
```

## 附录 B: 错误类型

```rust
/// 工作空间错误类型。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceError {
    /// 内存分配失败
    AllocFailed,
    
    /// 无效的内存布局
    InvalidLayout,
    
    /// 工作空间已被借用
    AlreadyBorrowed,
    
    /// 分割点越界
    SplitOutOfBounds,
    
    /// 对齐不足
    InsufficientAlignment,
}

impl core::fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AllocFailed => write!(f, "memory allocation failed"),
            Self::InvalidLayout => write!(f, "invalid memory layout"),
            Self::AlreadyBorrowed => write!(f, "workspace already borrowed"),
            Self::SplitOutOfBounds => write!(f, "split point out of bounds"),
            Self::InsufficientAlignment => write!(f, "insufficient alignment"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WorkspaceError {}
```

---

*本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。*
