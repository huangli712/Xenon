# 03-04 存储系统模块设计

> **模块路径**: `src/storage/`  
> **版本**: v1.0  
> **日期**: 2026-03-28  
> **依赖**: `core`, `alloc`, `std` (可选 feature)

---

## 1. 模块概述

### 1.1 职责定义

存储系统是 Xenon 张量库的基础设施层，负责管理张量数据的底层存储方式。核心职责包括：

| 职责 | 说明 |
|------|------|
| **存储抽象** | 定义统一的 `Storage` trait 层次，支持多种存储模式 |
| **内存管理** | 提供拥有、借用、共享三种所有权语义 |
| **对齐分配** | 实现 64 字节对齐的内存分配器，优化 SIMD 性能 |
| **类型安全** | 通过 trait 约束在编译期保证访问权限正确性 |
| **扩展性** | 预留 `Device` 关联类型，为 GPU 后端做准备 |

### 1.2 四种存储模式设计哲学

Xenon 采用 ndarray 风格的存储抽象，提供四种互补的存储模式：

```
存储模式分类
├── 拥有型 (Owned)
│   └── Owned<A> — 拥有数据，可读可写，深拷贝克隆
│
├── 借用型 (Borrowed)
│   ├── ViewRepr<&'a A> — 共享借用，只读，O(1) 克隆
│   └── ViewMutRepr<&'a mut A> — 独占借用，可读可写，不可克隆
│
└── 共享型 (Shared)
    └── ArcRepr<A> — Arc 共享，可读，写时复制，浅拷贝克隆
```

#### 1.2.1 访问级别与所有权矩阵

| 存储模式 | 拥有数据 | 可读 | 可写 | 克隆语义 | 分配方式 |
|----------|:--------:|:----:|:----:|----------|----------|
| `Owned<A>` | ✅ | ✅ | ✅ | 深拷贝 | 64 字节对齐堆分配 |
| `ViewRepr<&'a A>` | ❌ (借用) | ✅ | ❌ | O(1) 元数据拷贝 | 无分配 |
| `ViewMutRepr<&'a mut A>` | ❌ (独占借用) | ✅ | ✅ | 不可克隆 | 无分配 |
| `ArcRepr<A>` | ✅ (共享) | ✅ | make_mut() | 浅拷贝 (引用计数+1) | 写时按需分配 |

#### 1.2.2 设计权衡

| 考量 | Owned | View | ViewMut | Arc |
|------|-------|------|---------|-----|
| **创建开销** | 高 (分配) | 低 (借用) | 低 (借用) | 中 (Arc 包装) |
| **克隆开销** | O(n) | O(1) | 不可克隆 | O(1) |
| **写入开销** | 无 | 不可写 | 无 | 可能 O(n) (CoW) |
| **线程安全** | Send+Sync | Send+Sync | Send only | Send+Sync |
| **典型用途** | 创建、运算结果 | 切片、子数组 | 原地修改 | 跨线程共享、延迟复制 |

### 1.3 与 ndarray 的设计对比

| 方面 | ndarray | Xenon | 理由 |
|------|---------|-------|------|
| Trait 层次 | RawData → Data → DataMut → DataOwned | RawStorage → Storage → StorageMut → StorageOwned | 命名更清晰 |
| Arc 支持 | `ArcArray` 独立类型 | `ArcRepr<A>` 统一 trait 体系 | 更好的正交性 |
| 对齐分配 | 默认对齐 | 64 字节强制对齐 | AVX-512 优化 |
| Device 扩展 | 无 | `type Device` 关联类型 | GPU 预留 |

---

## 2. 文件结构

```
src/storage/
├── mod.rs             # Storage trait 和 RawStorage trait 定义
├── owned.rs           # Owned<A> 拥有型存储
├── view.rs            # ViewRepr<T> 不可变视图
├── view_mut.rs        # ViewMutRepr<T> 可变视图
├── arc.rs             # ArcRepr<A> 原子引用计数存储
├── alloc.rs           # 64 字节对齐分配器
└── traits.rs          # IsOwned, IsView 等 marker traits
```

### 2.1 各文件职责

| 文件 | 职责 | 核心类型 |
|------|------|----------|
| `mod.rs` | 定义 Storage trait 层次，模块导出 | `RawStorage`, `Storage`, `StorageMut`, `StorageOwned` |
| `owned.rs` | 实现拥有型存储 | `Owned<A>` |
| `view.rs` | 实现不可变视图存储 | `ViewRepr<&'a A>` |
| `view_mut.rs` | 实现可变视图存储 | `ViewMutRepr<&'a mut A>` |
| `arc.rs` | 实现 Arc 共享存储及 CoW | `ArcRepr<A>` |
| `alloc.rs` | 64 字节对齐的全局分配器 | `AlignedAlloc` |
| `traits.rs` | Marker traits 用于类型约束 | `IsOwned`, `IsView`, `IsViewMut`, `IsArc` |

### 2.2 模块依赖图

```
                    ┌─────────────┐
                    │  traits.rs  │
                    │ (marker)    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ owned.rs │    │ view.rs  │    │ view_mut │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         │    ┌──────────┴──────────┐    │
         │    │                     │    │
         ▼    ▼                     ▼    ▼
    ┌────────────────┐        ┌──────────┐
    │   alloc.rs     │        │  arc.rs  │
    └───────┬────────┘        └────┬─────┘
            │                      │
            └──────────┬───────────┘
                       │
                       ▼
                ┌─────────────┐
                │   mod.rs    │
                │ (trait def) │
                └─────────────┘
```

---

## 3. Storage Trait 层次设计

参考 ndarray 的 `RawData → Data → DataMut → DataOwned` 层次，Xenon 采用类似的分层设计：

### 3.1 Trait 层次图

```
RawStorage                    (最底层，提供原始指针访问)
    │
    ├── Storage : RawStorage  (安全读取)
    │       │
    │       ├── StorageMut : Storage  (安全写入)
    │       │       │
    │       │       └── StorageOwned : StorageMut  (拥有数据)
    │       │
    │       └── StorageShared : Storage  (共享数据，如 Arc)
    │
    └── RawStorageMut : RawStorage  (原始可变指针)
```

### 3.2 RawStorage Trait

最基础的 trait，仅提供原始指针访问，不保证安全性。

```rust
use core::ptr::NonNull;

/// 底层存储的原始指针访问。
///
/// 这是最基础的存储 trait，不提供任何安全保证。
/// 所有存储类型都必须实现此 trait。
///
/// # Safety
///
/// 实现者必须保证：
/// - `as_ptr()` 返回的指针在存储生命周期内有效
/// - 对于同一存储实例，多次调用 `as_ptr()` 返回相同地址
pub unsafe trait RawStorage {
    /// 存储的元素类型
    type Elem;

    /// 存储的设备类型（当前仅支持 Cpu）
    type Device;

    /// 返回数据起始位置的原始指针。
    ///
    /// 返回的指针可能未对齐或指向未初始化内存，
    /// 调用者须自行确保安全性。
    fn as_ptr(&self) -> *const Self::Elem;

    /// 返回存储的元素数量。
    fn len(&self) -> usize;

    /// 检查存储是否为空。
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 检查指针是否满足指定对齐要求。
    ///
    /// # Arguments
    /// * `align` - 对齐字节数，须为 2 的幂
    #[inline]
    fn is_aligned_to(&self, align: usize) -> bool {
        (self.as_ptr() as usize) % align == 0
    }

    /// 检查是否为 64 字节对齐（AVX-512 优化要求）。
    #[inline]
    fn is_aligned(&self) -> bool {
        self.is_aligned_to(64)
    }
}
```

### 3.3 RawStorageMut Trait

扩展 `RawStorage`，提供可变原始指针访问。

```rust
/// 可变存储的原始指针访问。
///
/// # Safety
///
/// 实现者必须保证：
/// - `as_mut_ptr()` 返回的指针在存储生命周期内有效
/// - 不存在其他指向同一数据的可变引用（别名规则）
pub unsafe trait RawStorageMut: RawStorage {
    /// 返回数据起始位置的可变原始指针。
    ///
    /// # Safety
    ///
    /// 调用者须保证：
    /// - 指针仅用于写入已初始化的元素
    /// - 不违反 Rust 别名规则
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;

    /// 将存储转换为 NonNull 指针（用于 FFI）。
    ///
    /// # Safety
    ///
    /// 存储必须非空。
    #[inline]
    unsafe fn as_non_null(&mut self) -> NonNull<Self::Elem> {
        NonNull::new_unchecked(self.as_mut_ptr())
    }
}
```

### 3.4 Storage Trait

在 `RawStorage` 之上提供安全的读取访问。

```rust
/// 安全的存储读取访问。
///
/// 实现 `Storage` 的类型可以安全地读取其包含的元素。
/// 所有元素保证已正确初始化。
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(storage.get(0), Some(&1.0));
/// ```
pub unsafe trait Storage: RawStorage {
    /// 获取指定索引处元素的不可变引用。
    ///
    /// # Arguments
    /// * `index` - 元素索引
    ///
    /// # Returns
    /// * `Some(&Elem)` - 索引有效时
    /// * `None` - 索引越界时
    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Elem> {
        if index < self.len() {
            // SAFETY: index 已检查边界
            Some(unsafe { &*self.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// 获取指定索引处元素的不可变引用（不检查边界）。
    ///
    /// # Safety
    ///
    /// 调用者必须保证 `index < self.len()`。
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Elem {
        &*self.as_ptr().add(index)
    }

    /// 返回整个存储的切片视图。
    ///
    /// 仅当存储连续时可用。
    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        // SAFETY: Storage 保证所有元素已初始化
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// 返回存储的迭代器。
    fn iter(&self) -> StorageIter<'_, Self>
    where
        Self: Sized,
    {
        StorageIter::new(self)
    }
}

/// Storage 的迭代器
pub struct StorageIter<'a, S: Storage + ?Sized> {
    storage: &'a S,
    index: usize,
}

impl<'a, S: Storage + ?Sized> StorageIter<'a, S> {
    fn new(storage: &'a S) -> Self {
        Self { storage, index: 0 }
    }
}

impl<'a, S: Storage + ?Sized> Iterator for StorageIter<'a, S> {
    type Item = &'a S::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.storage.len() {
            let item = unsafe { self.storage.get_unchecked(self.index) };
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.storage.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, S: Storage + ?Sized> ExactSizeIterator for StorageIter<'a, S> {}
```

### 3.5 StorageMut Trait

扩展 `Storage`，提供安全的写入访问。

```rust
/// 安全的存储读写访问。
///
/// 实现 `StorageMut` 的类型可以安全地读取和修改其包含的元素。
///
/// # Example
///
/// ```ignore
/// let mut storage: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// *storage.get_mut(0).unwrap() = 10.0;
/// assert_eq!(storage.get(0), Some(&10.0));
/// ```
pub unsafe trait StorageMut: Storage + RawStorageMut {
    /// 获取指定索引处元素的可变引用。
    ///
    /// # Arguments
    /// * `index` - 元素索引
    ///
    /// # Returns
    /// * `Some(&mut Elem)` - 索引有效时
    /// * `None` - 索引越界时
    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Elem> {
        if index < self.len() {
            // SAFETY: index 已检查边界
            Some(unsafe { &mut *self.as_mut_ptr().add(index) })
        } else {
            None
        }
    }

    /// 获取指定索引处元素的可变引用（不检查边界）。
    ///
    /// # Safety
    ///
    /// 调用者必须保证 `index < self.len()`。
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Elem {
        &mut *self.as_mut_ptr().add(index)
    }

    /// 返回整个存储的可变切片视图。
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        // SAFETY: StorageMut 保证所有元素已初始化且独占访问
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// 用指定值填充整个存储。
    #[inline]
    fn fill(&mut self, value: Self::Elem)
    where
        Self::Elem: Clone,
    {
        for i in 0..self.len() {
            unsafe {
                *self.get_unchecked_mut(i) = value.clone();
            }
        }
    }

    /// 用闭包生成的值填充整个存储。
    #[inline]
    fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> Self::Elem,
    {
        for i in 0..self.len() {
            unsafe {
                *self.get_unchecked_mut(i) = f();
            }
        }
    }

    /// 复制另一个存储的内容到当前存储。
    ///
    /// # Panics
    ///
    /// 如果两个存储长度不同，将 panic。
    #[inline]
    fn copy_from<Src: Storage<Elem = Self::Elem>>(&mut self, src: &Src)
    where
        Self::Elem: Clone,
    {
        assert_eq!(self.len(), src.len(), "storage length mismatch");
        for i in 0..self.len() {
            unsafe {
                *self.get_unchecked_mut(i) = src.get_unchecked(i).clone();
            }
        }
    }
}
```

### 3.6 StorageOwned Trait

扩展 `StorageMut`，表示拥有数据所有权的存储。

```rust
/// 拥有数据所有权的存储。
///
/// 实现 `StorageOwned` 的类型拥有其数据的完全所有权，
/// 可以进行分配、释放和深拷贝。
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::zeros(100);
/// let cloned = storage.to_owned();
/// ```
pub unsafe trait StorageOwned: StorageMut + Clone {
    /// 元素类型（与 RawStorage::Elem 相同，但重新声明以添加约束）
    type Elem: Clone;

    /// 分配指定大小的存储，用零填充。
    ///
    /// # Arguments
    /// * `len` - 元素数量
    fn zeros(len: usize) -> Self
    where
        Self::Elem: Default;

    /// 分配指定大小的存储，用指定值填充。
    ///
    /// # Arguments
    /// * `len` - 元素数量
    /// * `value` - 填充值
    fn from_elem(len: usize, value: Self::Elem) -> Self;

    /// 从 Vec 构造存储。
    ///
    /// # Arguments
    /// * `vec` - 源 Vec
    fn from_vec(vec: Vec<Self::Elem>) -> Self;

    /// 从迭代器构造存储。
    ///
    /// # Arguments
    /// * `iter` - 源迭代器
    fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self;

    /// 将存储转换为 Vec。
    fn into_vec(self) -> Vec<Self::Elem>;

    /// 创建存储的深拷贝。
    fn to_owned(&self) -> Self;

    /// 返回存储的容量（已分配但可能未使用的空间）。
    fn capacity(&self) -> usize;

    /// 尝试调整存储容量。
    ///
    /// # Arguments
    /// * `new_capacity` - 新容量
    ///
    /// # Returns
    /// * `Ok(())` - 成功调整
    /// * `Err(())` - 调整失败（通常因为存储不支持或容量不足）
    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), ()>;
}
```

### 3.7 StorageShared Trait

为共享存储（如 `ArcRepr`）提供特殊操作。

```rust
/// 共享存储的特殊操作。
///
/// 实现 `StorageShared` 的类型允许多个所有者共享同一数据，
/// 通常通过引用计数实现。
pub unsafe trait StorageShared: Storage + Clone {
    /// 元素类型
    type Elem: Clone;

    /// 检查是否为唯一所有者。
    ///
    /// 返回 `true` 表示可以安全地进行原地修改，
    /// 无需复制数据。
    fn is_unique(&self) -> bool;

    /// 获取数据的独占可变访问。
    ///
    /// 如果当前不是唯一所有者，将执行写时复制：
    /// 1. 分配新的内存
    /// 2. 复制数据
    /// 3. 释放原引用
    ///
    /// # Returns
    ///
    /// 返回可变切片，允许直接修改数据。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut arc_storage: ArcRepr<f64> = ArcRepr::from_vec(vec![1.0, 2.0, 3.0]);
    /// let shared = arc_storage.clone();  // 引用计数 = 2
    ///
    /// // make_mut 触发复制
    /// let data = arc_storage.make_mut();
    /// data[0] = 10.0;  // 仅修改 arc_storage，不影响 shared
    /// ```
    fn make_mut(&mut self) -> &mut [Self::Elem];

    /// 尝试获取独占所有权，不复制数据。
    ///
    /// 如果是唯一所有者，返回包含数据的 `Owned` 存储；
    /// 否则返回 `Err(self)`。
    fn try_into_owned(self) -> Result<Owned<Self::Elem>, Self>
    where
        Self: Sized;

    /// 获取当前引用计数。
    ///
    /// 注意：此方法主要用于调试，返回值可能立即过时。
    fn ref_count(&self) -> usize;
}
```

### 3.8 Trait 实现矩阵

| 存储类型 | RawStorage | Storage | RawStorageMut | StorageMut | StorageOwned | StorageShared |
|----------|:----------:|:-------:|:-------------:|:----------:|:------------:|:-------------:|
| `Owned<A>` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `ViewRepr<&A>` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `ViewMutRepr<&mut A>` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `ArcRepr<A>` | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |

---

## 4. Owned<A> 设计

### 4.1 结构定义

```rust
use alloc::vec::Vec;

/// 拥有型存储。
///
/// `Owned<A>` 拥有数据的完全所有权，数据存储在堆上，
/// 使用 64 字节对齐分配以优化 SIMD 操作。
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::zeros(100);
/// assert_eq!(storage.len(), 100);
/// assert!(storage.is_aligned());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Owned<A> {
    /// 内部数据存储
    /// 
    /// 使用自定义分配器确保 64 字节对齐。
    /// 小数组（≤ 64 字节）可能使用栈分配优化。
    data: Vec<A>,

    /// 分配时指定的对齐值
    alignment: usize,
}

impl<A> Owned<A> {
    /// 默认对齐值：64 字节（AVX-512 缓存行）
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// 最小对齐值：元素的自然对齐
    pub const MIN_ALIGNMENT: usize = core::mem::align_of::<A>();
}
```

### 4.2 构造方法

```rust
impl<A> Owned<A> {
    /// 创建空的拥有型存储。
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            alignment: Self::DEFAULT_ALIGNMENT,
        }
    }

    /// 创建指定容量的空存储。
    ///
    /// # Arguments
    /// * `capacity` - 预分配的容量
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            alignment: Self::DEFAULT_ALIGNMENT,
        }
    }

    /// 从 Vec 构造，使用默认对齐。
    ///
    /// 注意：传入的 Vec 可能不满足 64 字节对齐，
    /// 如需保证对齐，使用 `from_vec_aligned`。
    #[inline]
    pub fn from_vec(vec: Vec<A>) -> Self {
        Self {
            data: vec,
            alignment: Self::DEFAULT_ALIGNMENT,
        }
    }

    /// 从 Vec 构造，使用指定对齐。
    ///
    /// # Arguments
    /// * `vec` - 源 Vec
    /// * `alignment` - 对齐字节数，须为 2 的幂且 ≥ 元素自然对齐
    ///
    /// # Panics
    ///
    /// 如果对齐值无效，将 panic。
    #[inline]
    pub fn from_vec_aligned(vec: Vec<A>, alignment: usize) -> Self {
        assert!(alignment.is_power_of_two(), "alignment must be power of 2");
        assert!(alignment >= core::mem::align_of::<A>(), "alignment too small");
        
        // 检查当前对齐
        let ptr = vec.as_ptr();
        if (ptr as usize) % alignment != 0 {
            // 需要重新分配
            let len = vec.len();
            let mut aligned = Self::allocate_aligned(len, alignment);
            unsafe {
                core::ptr::copy_nonoverlapping(vec.as_ptr(), aligned.as_mut_ptr(), len);
            }
            core::mem::forget(vec);
            return aligned;
        }

        Self { data: vec, alignment }
    }

    /// 分配指定大小的对齐存储，不初始化。
    ///
    /// # Safety
    ///
    /// 返回的存储包含未初始化内存。
    #[inline]
    pub unsafe fn allocate_aligned(len: usize, alignment: usize) -> Self {
        use alloc::alloc::{alloc, dealloc, Layout};
        
        let size = len * core::mem::size_of::<A>();
        let layout = Layout::from_size_align(size.max(1), alignment)
            .expect("invalid layout");
        
        let ptr = alloc(layout);
        if ptr.is_null() {
            alloc::alloc::handle_alloc_error(layout);
        }
        
        // 包装为 Vec
        let data = Vec::from_raw_parts(ptr as *mut A, 0, len);
        Self { data, alignment }
    }
}

impl<A: Clone> Owned<A> {
    /// 创建指定大小的存储，用指定值填充。
    #[inline]
    pub fn from_elem(len: usize, value: A) -> Self {
        let mut storage = Self::with_capacity(len);
        storage.data.resize(len, value);
        storage
    }

    /// 创建指定大小的存储，用默认值填充。
    #[inline]
    pub fn zeros(len: usize) -> Self 
    where 
        A: Default 
    {
        Self::from_elem(len, A::default())
    }
}
```

### 4.3 Trait 实现

```rust
// === RawStorage ===

unsafe impl<A> RawStorage for Owned<A> {
    type Elem = A;
    type Device = crate::device::Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A {
        self.data.as_ptr()
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
}

// === RawStorageMut ===

unsafe impl<A> RawStorageMut for Owned<A> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut A {
        self.data.as_mut_ptr()
    }
}

// === Storage ===

unsafe impl<A> Storage for Owned<A> {}

// === StorageMut ===

unsafe impl<A> StorageMut for Owned<A> {}

// === StorageOwned ===

unsafe impl<A: Clone> StorageOwned for Owned<A> {
    type Elem = A;

    #[inline]
    fn zeros(len: usize) -> Self 
    where 
        Self::Elem: Default 
    {
        Self::zeros(len)
    }

    #[inline]
    fn from_elem(len: usize, value: Self::Elem) -> Self {
        Self::from_elem(len, value)
    }

    #[inline]
    fn from_vec(vec: Vec<Self::Elem>) -> Self {
        Self::from_vec(vec)
    }

    #[inline]
    fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }

    #[inline]
    fn into_vec(self) -> Vec<Self::Elem> {
        self.data
    }

    #[inline]
    fn to_owned(&self) -> Self {
        self.clone()
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), ()> {
        self.data.reserve(new_capacity.saturating_sub(self.data.capacity()));
        Ok(())
    }
}

// === Send / Sync ===

// Owned<A> 是 Send + Sync，当且仅当 A 是 Send + Sync
unsafe impl<A: Send> Send for Owned<A> {}
unsafe impl<A: Sync> Sync for Owned<A> {}

// === Default ===

impl<A> Default for Owned<A> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// === From traits ===

impl<A> From<Vec<A>> for Owned<A> {
    #[inline]
    fn from(vec: Vec<A>) -> Self {
        Self::from_vec(vec)
    }
}

impl<A, const N: usize> From<[A; N]> for Owned<A> 
where 
    A: Clone 
{
    #[inline]
    fn from(arr: [A; N]) -> Self {
        Self::from_vec(arr.to_vec())
    }
}
```

### 4.4 内存布局示意

```
Owned<f64> 内存布局（假设 8 个元素，64 字节对齐）

地址:     0x00  0x08  0x10  0x18  0x20  0x28  0x30  0x38
          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
数据:     | f64 | f64 | f64 | f64 | f64 | f64 | f64 | f64 |
          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
          │<───────────────── 64 字节 ─────────────────>│
          
对齐检查: (地址 % 64) == 0 ✓

Owned<Complex<f64>> 内存布局（假设 4 个元素，64 字节对齐）

地址:     0x00        0x10        0x20        0x30
          ├───────────┼───────────┼───────────┼───────────┤
数据:     | Cplx<f64> | Cplx<f64> | Cplx<f64> | Cplx<f64> |
          | (re, im)  | (re, im)  | (re, im)  | (re, im)  |
          └───────────┴───────────┴───────────┴───────────┘
          │<─────────────────── 64 字节 ──────────────────│
```

---

## 5. ViewRepr<&'a A> 设计

### 5.1 结构定义

```rust
use core::marker::PhantomData;

/// 不可变视图存储。
///
/// `ViewRepr<&'a A>` 是对数据的共享借用，提供只读访问。
/// 克隆操作仅复制元数据（指针 + 长度），不复制数据。
///
/// # 生命周期
///
/// `'a` 表示借用数据的生命周期。视图不能比其引用的数据存活更久。
///
/// # Example
///
/// ```ignore
/// let owned: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// let view: ViewRepr<&f64> = ViewRepr::new(owned.as_ptr(), 3);
/// assert_eq!(view.get(0), Some(&1.0));
/// ```
#[derive(Debug)]
pub struct ViewRepr<A> {
    /// 数据指针
    ptr: *const A,
    
    /// 元素数量
    len: usize,
    
    /// 生命周期标记
    _marker: PhantomData<A>,
}

/// 类型别名：不可变视图
pub type View<'a, A> = ViewRepr<&'a A>;
```

### 5.2 构造方法

```rust
impl<'a, A> ViewRepr<&'a A> {
    /// 从原始指针和长度创建视图。
    ///
    /// # Arguments
    /// * `ptr` - 数据起始指针
    /// * `len` - 元素数量
    ///
    /// # Safety
    ///
    /// 调用者必须保证：
    /// - `ptr` 非空且对齐到 `align_of::<A>()`
    /// - `ptr` 起始的 `len` 个元素已正确初始化
    /// - 数据在 `'a` 生命周期内保持有效
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *const A, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// 从切片创建视图。
    #[inline]
    pub fn from_slice(slice: &'a [A]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    /// 从 Owned 存储创建视图。
    #[inline]
    pub fn from_owned(owned: &'a Owned<A>) -> Self {
        Self {
            ptr: owned.as_ptr(),
            len: owned.len(),
            _marker: PhantomData,
        }
    }

    /// 创建子视图（切片）。
    ///
    /// # Arguments
    /// * `start` - 起始索引
    /// * `end` - 结束索引（不含）
    ///
    /// # Panics
    ///
    /// 如果索引越界，将 panic。
    #[inline]
    pub fn slice(&self, start: usize, end: usize) -> ViewRepr<&'a A> {
        assert!(start <= end, "start must be <= end");
        assert!(end <= self.len, "end must be <= len");
        
        Self {
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            _marker: PhantomData,
        }
    }
}
```

### 5.3 Clone 实现（O(1) 元数据复制）

```rust
// Clone 仅复制指针和长度，不复制数据
impl<'a, A> Clone for ViewRepr<&'a A> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

// Copy 等价于 Clone
impl<'a, A> Copy for ViewRepr<&'a A> {}
```

### 5.4 Trait 实现

```rust
// === RawStorage ===

unsafe impl<'a, A> RawStorage for ViewRepr<&'a A> {
    type Elem = A;
    type Device = crate::device::Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A {
        self.ptr
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

// === Storage ===

unsafe impl<'a, A> Storage for ViewRepr<&'a A> {}

// 注意：ViewRepr 不实现 RawStorageMut 或 StorageMut（只读）

// === Send / Sync ===

// ViewRepr<&'a A> 是 Send + Sync，当且仅当 A 是 Sync
// 这是因为共享引用要求跨线程共享时类型必须是 Sync
unsafe impl<'a, A: Sync> Send for ViewRepr<&'a A> {}
unsafe impl<'a, A: Sync> Sync for ViewRepr<&'a A> {}

// === PartialEq ===

impl<'a, A: PartialEq> PartialEq for ViewRepr<&'a A> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
```

### 5.5 生命周期约束

```rust
// 视图的生命周期规则：
//
// 1. 视图不能比源数据存活更久
// 2. 可以从同一源数据创建多个视图
// 3. 视图之间不互斥

// 示例：生命周期约束
//
// let owned: Owned<f64> = Owned::zeros(10);
// let view1: View<'_, f64> = View::from_owned(&owned);
// let view2: View<'_, f64> = view1.clone();  // OK: 多个共享视图
//
// drop(owned);  // 编译错误：view1 和 view2 仍在使用 owned

// 示例：视图切片
//
// let view: View<'_, f64> = ...;
// let sub_view = view.slice(2, 5);  // 生命周期与 view 相同
```

---

## 6. ViewMutRepr<&'a mut A> 设计

### 6.1 结构定义

```rust
/// 可变视图存储。
///
/// `ViewMutRepr<&'a mut A>` 是对数据的独占借用，提供读写访问。
/// 由于独占语义，此类型**不可克隆**。
///
/// # 独占语义
///
/// 在任意时刻，对同一数据只能存在一个 `ViewMutRepr`。
/// 这由 Rust 借用检查器在编译期保证。
///
/// # Example
///
/// ```ignore
/// let mut owned: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// let view_mut: ViewMutRepr<&mut f64> = ViewMutRepr::from_owned(&mut owned);
/// view_mut.fill(0.0);
/// ```
#[derive(Debug)]
pub struct ViewMutRepr<A> {
    /// 数据指针
    ptr: *mut A,
    
    /// 元素数量
    len: usize,
    
    /// 生命周期标记
    _marker: PhantomData<A>,
}

/// 类型别名：可变视图
pub type ViewMut<'a, A> = ViewMutRepr<&'a mut A>;
```

### 6.2 构造方法

```rust
impl<'a, A> ViewMutRepr<&'a mut A> {
    /// 从原始可变指针和长度创建视图。
    ///
    /// # Safety
    ///
    /// 调用者必须保证：
    /// - `ptr` 非空且对齐到 `align_of::<A>()`
    /// - `ptr` 起始的 `len` 个元素已正确初始化
    /// - 数据在 `'a` 生命周期内保持有效
    /// - 调用期间不存在其他指向该数据的引用（独占访问）
    #[inline]
    pub unsafe fn from_raw_parts_mut(ptr: *mut A, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// 从可变切片创建视图。
    #[inline]
    pub fn from_mut_slice(slice: &'a mut [A]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    /// 从 Owned 存储创建可变视图。
    #[inline]
    pub fn from_owned(owned: &'a mut Owned<A>) -> Self {
        Self {
            ptr: owned.as_mut_ptr(),
            len: owned.len(),
            _marker: PhantomData,
        }
    }

    /// 创建可变子视图。
    ///
    /// # Panics
    ///
    /// 如果索引越界，将 panic。
    #[inline]
    pub fn slice_mut(&mut self, start: usize, end: usize) -> ViewMutRepr<&'a mut A> {
        assert!(start <= end, "start must be <= end");
        assert!(end <= self.len, "end must be <= len");
        
        ViewMutRepr {
            ptr: unsafe { self.ptr.add(start) },
            len: end - start,
            _marker: PhantomData,
        }
    }

    /// 降级为不可变视图。
    #[inline]
    pub fn as_view(&self) -> ViewRepr<&'a A> {
        ViewRepr {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}
```

### 6.3 不可 Clone 的设计

```rust
// ViewMutRepr 刻意不实现 Clone
// 原因：独占语义意味着同一时刻只能存在一个可变引用

// 以下代码会编译失败：
//
// let mut owned: Owned<f64> = ...;
// let view: ViewMut<'_, f64> = ViewMut::from_owned(&mut owned);
// let view2 = view.clone();  // 编译错误：Clone 未实现

// 替代方案：分割视图
//
// let mut view: ViewMut<'_, f64> = ...;
// let (left, right) = view.split_at_mut(5);  // 分割为两个不重叠的可变视图
```

### 6.4 Trait 实现

```rust
// === RawStorage ===

unsafe impl<'a, A> RawStorage for ViewMutRepr<&'a mut A> {
    type Elem = A;
    type Device = crate::device::Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A {
        self.ptr
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

// === RawStorageMut ===

unsafe impl<'a, A> RawStorageMut for ViewMutRepr<&'a mut A> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut A {
        self.ptr
    }
}

// === Storage ===

unsafe impl<'a, A> Storage for ViewMutRepr<&'a mut A> {}

// === StorageMut ===

unsafe impl<'a, A> StorageMut for ViewMutRepr<&'a mut A> {}

// 注意：ViewMutRepr 不实现 StorageOwned（不拥有数据）

// === Send / Sync ===

// ViewMutRepr<&'a mut A> 是 Send，当且仅当 A 是 Send
// 原因：可变引用可以跨线程转移所有权
unsafe impl<'a, A: Send> Send for ViewMutRepr<&'a mut A> {}

// ViewMutRepr<&'a mut A> 永远不是 Sync
// 原因：可变引用不能共享，Rust 借用规则禁止
// impl<'a, A> !Sync for ViewMutRepr<&'a mut A> {}
```

### 6.5 独占语义示意

```
独占语义示例

时间线：
t0: let mut owned = Owned::zeros(10);
    ┌─────────────────────────────────┐
    │         Owned<f64>              │
    │  [0.0, 0.0, 0.0, ..., 0.0]      │
    └─────────────────────────────────┘

t1: let view_mut = ViewMut::from_owned(&mut owned);
    ┌─────────────────────────────────┐
    │         Owned<f64>              │ ◄── 被独占借用
    │  [0.0, 0.0, 0.0, ..., 0.0]      │
    └──────────────┬──────────────────┘
                   │ 独占访问
                   ▼
    ┌─────────────────────────────────┐
    │       ViewMut<&mut f64>         │
    │  ptr ───────────────────────────┼──► 数据
    │  len = 10                       │
    └─────────────────────────────────┘

t2: view_mut.fill(1.0);  // OK: 独占访问
    ┌─────────────────────────────────┐
    │         Owned<f64>              │
    │  [1.0, 1.0, 1.0, ..., 1.0]      │ ◄── 被修改
    └─────────────────────────────────┘

t3: let view2 = ViewMut::from_owned(&mut owned);
    // 编译错误：owned 已被 view_mut 借用
```

---

## 7. ArcRepr<A> 设计

### 7.1 结构定义

```rust
use alloc::sync::Arc;
use alloc::vec::Vec;

/// 原子引用计数存储。
///
/// `ArcRepr<A>` 使用 `Arc` 共享数据所有权，允许多个所有者
/// 安全地共享同一数据。通过 `make_mut()` 实现写时复制 (CoW)。
///
/// # 写时复制 (Copy-on-Write)
///
/// 当尝试修改数据时，如果不是唯一所有者，将自动复制数据，
/// 避免影响其他共享者。
///
/// # Example
///
/// ```ignore
/// let arc: ArcRepr<f64> = ArcRepr::from_vec(vec![1.0, 2.0, 3.0]);
/// let shared = arc.clone();  // 引用计数 = 2
///
/// // make_mut 触发复制
/// let data = arc.make_mut();
/// data[0] = 10.0;  // 仅修改 arc，不影响 shared
/// ```
#[derive(Debug)]
pub struct ArcRepr<A> {
    /// Arc 包装的数据
    /// 
    /// 使用 Arc<[A]> 或 Arc<Vec<A>> 取决于实现策略
    inner: Arc<Vec<A>>,
    
    /// 当前存储的长度（可能与 inner.len() 不同，支持切片）
    len: usize,
    
    /// 起始偏移量
    offset: usize,
}

impl<A> ArcRepr<A> {
    /// 从 Vec 构造。
    #[inline]
    pub fn from_vec(vec: Vec<A>) -> Self {
        let len = vec.len();
        Self {
            inner: Arc::new(vec),
            len,
            offset: 0,
        }
    }

    /// 从 Owned 存储构造。
    #[inline]
    pub fn from_owned(owned: Owned<A>) -> Self {
        Self::from_vec(owned.into_vec())
    }

    /// 获取当前引用计数。
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// 检查是否为唯一所有者。
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }
}
```

### 7.2 make_mut() 完整语义

```rust
impl<A: Clone> ArcRepr<A> {
    /// 获取数据的独占可变访问。
    ///
    /// # 写时复制 (CoW) 行为
    ///
    /// 1. **引用计数 = 1**：直接返回可变引用，无复制
    /// 2. **引用计数 > 1**：
    ///    a. 分配新的 64 字节对齐内存
    ///    b. 复制所有数据到新内存
    ///    c. 原引用计数减 1
    ///    d. 返回新内存的可变引用
    ///
    /// # 原子性保证
    ///
    /// - 引用计数检查与递减为原子操作
    /// - 多线程并发 `make_mut()` 不会导致数据竞争或重复拷贝
    /// - 使用 `Arc::get_mut()` 或 `Arc::make_mut()` 实现
    ///
    /// # Example
    ///
    /// ```ignore
    /// let arc: ArcRepr<f64> = ArcRepr::from_vec(vec![1.0, 2.0, 3.0]);
    /// let shared = arc.clone();
    ///
    /// // arc.ref_count() == 2
    /// let data = arc.make_mut();  // 触发复制
    /// // arc.ref_count() == 1, shared.ref_count() == 1
    /// data[0] = 10.0;
    /// ```
    #[inline]
    pub fn make_mut(&mut self) -> &mut [A] {
        // 使用 Arc::make_mut 实现原子性 CoW
        // 这会自动处理引用计数检查和复制
        let data = Arc::make_mut(&mut self.inner);
        
        // 返回偏移后的切片
        &mut data[self.offset..self.offset + self.len]
    }

    /// 尝试获取独占所有权，不复制数据。
    ///
    /// # Returns
    ///
    /// - `Ok(Owned<A>)`：如果是唯一所有者
    /// - `Err(self)`：如果存在其他所有者
    #[inline]
    pub fn try_into_owned(self) -> Result<Owned<A>, Self> {
        match Arc::try_unwrap(self.inner) {
            Ok(vec) => Ok(Owned::from_vec_aligned(vec, Owned::<A>::DEFAULT_ALIGNMENT)),
            Err(arc) => Err(Self {
                inner: arc,
                len: self.len,
                offset: self.offset,
            }),
        }
    }

    /// 强制转换为 Owned，必要时复制数据。
    #[inline]
    pub fn into_owned(self) -> Owned<A> {
        match self.try_into_owned() {
            Ok(owned) => owned,
            Err(this) => {
                // 复制数据
                let vec = this.inner[this.offset..this.offset + this.len].to_vec();
                Owned::from_vec_aligned(vec, Owned::<A>::DEFAULT_ALIGNMENT)
            }
        }
    }
}
```

### 7.3 写时复制流程图

```
make_mut() 执行流程

┌─────────────────────────────────────────────────────────────┐
│                      make_mut() 调用                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ 引用计数 == 1? │
                 └───────┬────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼ YES                       ▼ NO
    ┌──────────────┐          ┌──────────────────┐
    │ 直接返回     │          │ 分配新内存       │
    │ &mut [A]     │          │ (64 字节对齐)    │
    │ (零开销)     │          └────────┬─────────┘
    └──────────────┘                   │
                                       ▼
                              ┌──────────────────┐
                              │ 复制数据到新内存 │
                              │ (深拷贝 O(n))    │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ 原引用计数减 1   │
                              │ (原子操作)       │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ 返回新内存的     │
                              │ &mut [A]         │
                              └──────────────────┘
```

### 7.4 原子性语义详解

```rust
// 原子性保证的实现细节：
//
// Arc::make_mut 的行为：
// 1. 原子地检查引用计数
// 2. 如果 > 1，原子地递减引用计数
// 3. 分配新内存并复制数据
//
// 多线程场景：
//
// 线程 A                          线程 B
// ─────────                       ─────────
// arc.make_mut()
//   ├─ 检查 ref_count = 2        
//   ├─ 开始复制...               arc.make_mut()
//   │                               ├─ 检查 ref_count = 2 (原子读取)
//   │                               ├─ 开始复制...
//   ├─ 完成复制                   
//   ├─ 替换 Arc (原子操作)        ├─ 完成复制
//   └─ 返回 &mut [A]              ├─ 替换 Arc (原子操作)
//                                  └─ 返回 &mut [A]
//
// 结果：两个线程各自拥有独立的副本，无数据竞争
```

### 7.5 Trait 实现

```rust
// === RawStorage ===

unsafe impl<A> RawStorage for ArcRepr<A> {
    type Elem = A;
    type Device = crate::device::Cpu;

    #[inline]
    fn as_ptr(&self) -> *const A {
        unsafe { self.inner.as_ptr().add(self.offset) }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

// === Storage ===

unsafe impl<A> Storage for ArcRepr<A> {}

// 注意：ArcRepr 不实现 RawStorageMut 或 StorageMut
// 可变访问通过 StorageShared::make_mut() 获取

// === StorageShared ===

unsafe impl<A: Clone> StorageShared for ArcRepr<A> {
    type Elem = A;

    #[inline]
    fn is_unique(&self) -> bool {
        self.ref_count() == 1
    }

    #[inline]
    fn make_mut(&mut self) -> &mut [A] {
        self.make_mut()
    }

    #[inline]
    fn try_into_owned(self) -> Result<Owned<A>, Self> {
        self.try_into_owned()
    }

    #[inline]
    fn ref_count(&self) -> usize {
        self.ref_count()
    }
}

// === Clone (浅拷贝) ===

impl<A> Clone for ArcRepr<A> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),  // 仅增加引用计数
            len: self.len,
            offset: self.offset,
        }
    }
}

// === Send / Sync ===

// ArcRepr<A> 是 Send + Sync，当且仅当 A 是 Send + Sync
// Arc 内部使用原子操作，天然线程安全
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

---

## 8. 对齐分配器 (alloc.rs)

### 8.1 设计目标

| 目标 | 说明 |
|------|------|
| **默认对齐** | 64 字节，AVX-512 缓存行对齐 |
| **可配置** | 支持更大对齐（如 128 字节） |
| **no_std 兼容** | 使用 `alloc::alloc` 模块 |
| **零初始化** | 可选择是否零初始化 |
| **小数组优化** | ≤ 64 字节时降级到自然对齐 |

### 8.2 核心实现

```rust
use core::ptr::NonNull;
use alloc::alloc::{alloc, dealloc, Layout};

/// 64 字节对齐的内存分配器。
///
/// 用于张量数据的堆分配，优化 SIMD 访问性能。
pub struct AlignedAlloc;

impl AlignedAlloc {
    /// 默认对齐值：64 字节
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// 分配指定大小和对齐的内存块，不初始化。
    ///
    /// # Arguments
    ///
    /// * `size` - 字节数
    /// * `align` - 对齐字节数，须为 2 的幂
    ///
    /// # Returns
    ///
    /// 返回非空指针，或 panic（内存不足时）。
    ///
    /// # Panics
    ///
    /// - 如果 `align` 不是 2 的幂
    /// - 如果 `size` 为 0
    /// - 如果内存分配失败
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ptr = AlignedAlloc::alloc(1024, 64);
    /// // 使用 ptr...
    /// AlignedAlloc::dealloc(ptr, 1024, 64);
    /// ```
    #[inline]
    pub fn alloc(size: usize, align: usize) -> NonNull<u8> {
        assert!(align.is_power_of_two(), "alignment must be power of 2");
        assert!(size > 0, "size must be > 0");

        let layout = Layout::from_size_align(size, align)
            .expect("invalid layout");

        let ptr = unsafe { alloc(layout) };
        
        NonNull::new(ptr)
            .unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout))
    }

    /// 分配并零初始化。
    ///
    /// # Arguments
    ///
    /// * `size` - 字节数
    /// * `align` - 对齐字节数
    #[inline]
    pub fn alloc_zeroed(size: usize, align: usize) -> NonNull<u8> {
        use alloc::alloc::alloc_zeroed;

        assert!(align.is_power_of_two(), "alignment must be power of 2");
        assert!(size > 0, "size must be > 0");

        let layout = Layout::from_size_align(size, align)
            .expect("invalid layout");

        let ptr = unsafe { alloc_zeroed(layout) };
        
        NonNull::new(ptr)
            .unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout))
    }

    /// 释放内存。
    ///
    /// # Safety
    ///
    /// - `ptr` 必须是由 `alloc` 或 `alloc_zeroed` 返回的
    /// - `size` 和 `align` 必须与分配时相同
    #[inline]
    pub unsafe fn dealloc(ptr: NonNull<u8>, size: usize, align: usize) {
        let layout = Layout::from_size_align(size, align)
            .expect("invalid layout");
        dealloc(ptr.as_ptr(), layout);
    }

    /// 计算分配大小，考虑对齐填充。
    ///
    /// # Arguments
    ///
    /// * `elem_size` - 单个元素大小
    /// * `count` - 元素数量
    /// * `align` - 对齐要求
    #[inline]
    pub fn calculate_size(elem_size: usize, count: usize, align: usize) -> usize {
        // 确保总大小是对齐的倍数
        let total = elem_size * count;
        let padded = (total + align - 1) / align * align;
        padded
    }

    /// 检查是否应该使用对齐分配。
    ///
    /// 小数组（≤ 64 字节）使用自然对齐即可。
    #[inline]
    pub fn should_use_aligned_alloc(elem_size: usize, count: usize, align: usize) -> bool {
        let total = elem_size * count;
        total > align
    }
}
```

### 8.3 类型化分配接口

```rust
/// 类型化的对齐分配器。
pub struct TypedAlloc<A> {
    _marker: core::marker::PhantomData<A>,
}

impl<A> TypedAlloc<A> {
    /// 元素大小
    pub const ELEM_SIZE: usize = core::mem::size_of::<A>();
    
    /// 元素自然对齐
    pub const NATURAL_ALIGN: usize = core::mem::align_of::<A>();

    /// 分配指定数量元素的内存，不初始化。
    ///
    /// # Arguments
    ///
    /// * `count` - 元素数量
    /// * `align` - 对齐字节数
    ///
    /// # Safety
    ///
    /// 返回的内存未初始化。
    #[inline]
    pub unsafe fn alloc(count: usize, align: usize) -> NonNull<A> {
        let size = AlignedAlloc::calculate_size(Self::ELEM_SIZE, count, align);
        let ptr = AlignedAlloc::alloc(size, align);
        NonNull::new_unchecked(ptr.as_ptr() as *mut A)
    }

    /// 分配并零初始化。
    ///
    /// # Safety
    ///
    /// 仅适用于零值是有效初始化的类型。
    #[inline]
    pub unsafe fn alloc_zeroed(count: usize, align: usize) -> NonNull<A> {
        let size = AlignedAlloc::calculate_size(Self::ELEM_SIZE, count, align);
        let ptr = AlignedAlloc::alloc_zeroed(size, align);
        NonNull::new_unchecked(ptr.as_ptr() as *mut A)
    }

    /// 释放内存。
    ///
    /// # Safety
    ///
    /// - `ptr` 必须是由 `alloc` 或 `alloc_zeroed` 返回的
    /// - `count` 和 `align` 必须与分配时相同
    #[inline]
    pub unsafe fn dealloc(ptr: NonNull<A>, count: usize, align: usize) {
        let size = AlignedAlloc::calculate_size(Self::ELEM_SIZE, count, align);
        AlignedAlloc::dealloc(
            NonNull::new_unchecked(ptr.as_ptr() as *mut u8),
            size,
            align,
        );
    }

    /// 分配并填充默认值。
    #[inline]
    pub fn alloc_default(count: usize, align: usize) -> NonNull<A>
    where
        A: Default + Clone,
    {
        unsafe {
            let ptr = Self::alloc(count, align);
            let slice = core::slice::from_raw_parts_mut(ptr.as_ptr(), count);
            for elem in slice {
                elem.write(A::default());
            }
            ptr
        }
    }

    /// 分配并填充指定值。
    #[inline]
    pub fn alloc_with(count: usize, align: usize, value: A) -> NonNull<A>
    where
        A: Clone,
    {
        unsafe {
            let ptr = Self::alloc(count, align);
            let slice = core::slice::from_raw_parts_mut(ptr.as_ptr(), count);
            for elem in slice {
                elem.write(value.clone());
            }
            ptr
        }
    }
}
```

### 8.4 no_std 下的实现策略

```rust
// 在 no_std 环境下，使用 alloc crate 的接口

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, alloc_zeroed, Layout};

#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, alloc_zeroed, Layout};

// 关键点：
// 1. alloc crate 提供 no_std 下的堆分配支持
// 2. Layout 用于描述内存布局（大小 + 对齐）
// 3. alloc/dealloc 是原始的底层接口
// 4. 必须手动管理内存生命周期

// 示例：no_std 下分配数组
//
// #![no_std]
// extern crate alloc;
// 
// use alloc::alloc::{alloc, dealloc, Layout};
// 
// fn allocate_array<T>(count: usize, align: usize) -> *mut T {
//     let layout = Layout::from_size_align(
//         count * core::mem::size_of::<T>(),
//         align
//     ).unwrap();
//     
//     unsafe { alloc(layout) as *mut T }
// }
```

### 8.5 Layout 计算详解

```rust
/// Layout 计算工具
pub mod layout_utils {
    use core::mem::{size_of, align_of};

    /// 计算数组的 Layout。
    ///
    /// # Arguments
    ///
    /// * `elem_size` - 元素大小
    /// * `elem_align` - 元素对齐
    /// * `count` - 元素数量
    /// * `requested_align` - 请求的对齐（≥ elem_align）
    ///
    /// # Returns
    ///
    /// 成功返回 `Layout`，失败返回 `None`。
    #[inline]
    pub fn array_layout(
        elem_size: usize,
        elem_align: usize,
        count: usize,
        requested_align: usize,
    ) -> Option<Layout> {
        // 对齐必须至少是元素对齐
        let align = requested_align.max(elem_align);
        
        // 对齐必须是 2 的幂
        if !align.is_power_of_two() {
            return None;
        }

        // 计算总大小，检查溢出
        let size = elem_size.checked_mul(count)?;
        
        // 大小必须非零
        if size == 0 {
            return None;
        }

        // 大小必须是对齐的倍数
        let padded_size = size.checked_add(align - 1)? / align * align;

        Layout::from_size_align(padded_size, align).ok()
    }

    /// 检查指针是否满足对齐要求。
    #[inline]
    pub fn is_aligned(ptr: *const u8, align: usize) -> bool {
        (ptr as usize) % align == 0
    }

    /// 向上对齐大小。
    #[inline]
    pub fn align_up(size: usize, align: usize) -> usize {
        (size + align - 1) / align * align
    }

    /// 向下对齐大小。
    #[inline]
    pub fn align_down(size: usize, align: usize) -> usize {
        size / align * align
    }
}
```

---

## 9. Marker Traits (traits.rs)

### 9.1 设计目的

Marker traits 用于在类型系统层面区分不同的存储特性，使编译器能够：

1. **约束泛型参数**：确保函数只接受特定类型的存储
2. **静态分发**：为不同存储类型生成特化代码
3. **文档化意图**：明确表达存储的语义特性

### 9.2 Trait 定义

```rust
/// 标记存储类型拥有数据。
///
/// 实现 `IsOwned` 的类型拥有其数据的完全所有权，
/// 可以自由分配和释放。
///
/// # 实现者
///
/// - `Owned<A>`
/// - `ArcRepr<A>` (共享拥有)
pub unsafe trait IsOwned: RawStorage {}

/// 标记存储类型是视图（借用）。
///
/// 实现 `IsView` 的类型是对其他数据的借用，
/// 不拥有数据所有权。
///
/// # 实现者
///
/// - `ViewRepr<&A>`
pub unsafe trait IsView: RawStorage {}

/// 标记存储类型是可变视图。
///
/// 实现 `IsViewMut` 的类型是对数据的独占借用，
/// 可以修改数据但不拥有所有权。
///
/// # 实现者
///
/// - `ViewMutRepr<&mut A>`
pub unsafe trait IsViewMut: RawStorage {}

/// 标记存储类型使用 Arc 共享。
///
/// 实现 `IsArc` 的类型使用引用计数共享数据，
/// 支持写时复制。
///
/// # 实现者
///
/// - `ArcRepr<A>`
pub unsafe trait IsArc: RawStorage {}

/// 标记存储类型支持克隆。
///
/// 注意：`ViewMutRepr` 刻意不实现此 trait。
pub unsafe trait IsClonable: Clone {}

/// 标记存储类型支持深拷贝。
///
/// 实现 `IsDeepClone` 的类型在克隆时会复制所有数据。
pub unsafe trait IsDeepClone: IsClonable {}

/// 标记存储类型支持浅拷贝。
///
/// 实现 `IsShallowClone` 的类型在克隆时仅复制元数据。
pub unsafe trait IsShallowClone: IsClonable {}
```

### 9.3 Trait 实现

```rust
// === Owned ===

unsafe impl<A> IsOwned for Owned<A> {}
unsafe impl<A: Clone> IsClonable for Owned<A> {}
unsafe impl<A: Clone> IsDeepClone for Owned<A> {}

// === ViewRepr ===

unsafe impl<'a, A> IsView for ViewRepr<&'a A> {}
unsafe impl<'a, A> IsClonable for ViewRepr<&'a A> {}
unsafe impl<'a, A> IsShallowClone for ViewRepr<&'a A> {}

// === ViewMutRepr ===

unsafe impl<'a, A> IsViewMut for ViewMutRepr<&'a mut A> {}
// 注意：ViewMutRepr 不实现 IsClonable

// === ArcRepr ===

unsafe impl<A> IsOwned for ArcRepr<A> {}
unsafe impl<A> IsArc for ArcRepr<A> {}
unsafe impl<A> IsClonable for ArcRepr<A> {}
unsafe impl<A> IsShallowClone for ArcRepr<A> {}
```

### 9.4 使用示例

```rust
// 约束函数只接受拥有型存储
fn require_owned<S: IsOwned>(storage: &S) {
    // 只能传入 Owned 或 ArcRepr
}

// 约束函数只接受可写存储
fn require_mutable<S: StorageMut>(storage: &mut S) {
    // 只能传入 Owned 或 ViewMutRepr
}

// 约束函数只接受支持克隆的存储
fn require_clonable<S: IsClonable>(storage: &S) -> S {
    storage.clone()
}

// 深拷贝 vs 浅拷贝的区分
fn clone_storage<S>(storage: &S) -> S
where
    S: IsDeepClone + IsShallowClone,
{
    // 编译错误：不能同时实现两种克隆语义
    // 这确保了类型安全
    storage.clone()
}
```

### 9.5 Sealed Trait 模式

```rust
// 防止外部类型实现 marker traits
// 使用 sealed trait 模式

mod private {
    pub trait Sealed {}
}

// 在公开 trait 中使用 Sealed
pub unsafe trait IsOwned: private::Sealed + RawStorage {}

// 仅为我们控制的类型实现 Sealed
impl<A> private::Sealed for Owned<A> {}
impl<'a, A> private::Sealed for ViewRepr<&'a A> {}
impl<'a, A> private::Sealed for ViewMutRepr<&'a mut A> {}
impl<A> private::Sealed for ArcRepr<A> {}

// 这样外部用户无法为自己的类型实现 IsOwned
```

---

## 10. Device 扩展性

### 10.1 设计目标

| 目标 | 说明 |
|------|------|
| **当前支持** | 仅 Cpu |
| **未来扩展** | Cuda, Metal, Vulkan, WebGPU |
| **接口稳定** | 不修改 Storage trait 签名 |
| **零运行时开销** | Device 类型作为泛型参数 |

### 10.2 Device Trait 定义

```rust
/// 设备抽象。
///
/// 定义存储后端的执行设备。
/// 当前仅支持 Cpu，未来将支持 GPU 设备。
pub trait Device: Clone + Send + Sync + 'static {
    /// 设备标识符
    type Id: Clone + PartialEq + core::fmt::Debug + Send + Sync;

    /// 返回设备标识符。
    fn id(&self) -> Self::Id;

    /// 检查是否为默认设备。
    fn is_default(&self) -> bool;
}

/// CPU 设备。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpu;

impl Device for Cpu {
    type Id = ();

    #[inline]
    fn id(&self) -> Self::Id {}

    #[inline]
    fn is_default(&self) -> bool {
        true
    }
}

// 未来扩展示例（不在当前版本实现）：
//
// #[cfg(feature = "cuda")]
// pub struct Cuda {
//     device_id: usize,
// }
//
// impl Device for Cuda {
//     type Id = usize;
//     
//     fn id(&self) -> Self::Id {
//         self.device_id
//     }
//     
//     fn is_default(&self) -> bool {
//         self.device_id == 0
//     }
// }
```

### 10.3 Storage 中的 Device 关联类型

```rust
// RawStorage 中的 Device 关联类型

pub unsafe trait RawStorage {
    type Elem;
    
    /// 存储所在的设备类型。
    ///
    /// 当前仅支持 Cpu。未来可能扩展到 Cuda、Metal 等。
    type Device: crate::device::Device;
    
    // ...
}

// 各存储类型的 Device 实现

unsafe impl<A> RawStorage for Owned<A> {
    type Elem = A;
    type Device = Cpu;  // 当前仅支持 Cpu
}

unsafe impl<'a, A> RawStorage for ViewRepr<&'a A> {
    type Elem = A;
    type Device = Cpu;
}

// 未来扩展示例：
//
// unsafe impl<A> RawStorage for CudaOwned<A> {
//     type Elem = A;
//     type Device = Cuda;
// }
```

### 10.4 扩展策略

```
设备扩展路线图

Phase 1 (当前版本):
├── Device trait 定义
├── Cpu 实现
└── Storage::Device = Cpu

Phase 2 (未来):
├── Cuda 设备支持
│   ├── CudaOwned<A>
│   ├── CudaView<A>
│   └── 设备间数据传输
│
├── Metal 设备支持 (macOS)
│   └── ...
│
└── 统一的设备无关 API
    └── TensorBase<S, D> 自动适配设备
```

### 10.5 设备相关约束

```rust
/// 约束存储在 CPU 上。
pub trait OnCpu: RawStorage<Device = Cpu> {}

impl<S: RawStorage<Device = Cpu>> OnCpu for S {}

/// 约束存储在 GPU 上（未来）。
#[cfg(feature = "cuda")]
pub trait OnGpu: RawStorage<Device = Cuda> {}

// 使用示例：
//
// fn cpu_only<S: OnCpu>(storage: &S) {
//     // 仅接受 CPU 存储
// }
```

---

## 11. 与其他模块的交互

### 11.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                     TensorBase<S, D>                         │
│                   (src/tensor/mod.rs)                        │
└────────────────────────┬────────────────────────────────────┘
                         │ 使用
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Traits                            │
│   RawStorage, Storage, StorageMut, StorageOwned              │
│                   (src/storage/mod.rs)                       │
└──────────┬────────────────┬────────────────┬────────────────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Owned   │     │ViewRepr  │     │ArcRepr   │
    │          │     │ViewMutRepr│    │          │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                │                │
         ▼                │                │
    ┌──────────┐          │                │
    │ alloc.rs │          │                │
    │(对齐分配)│          │                │
    └──────────┘          │                │
                          │                │
         ┌────────────────┴────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Layout Module                             │
│        LayoutFlags, Strides, Contiguity                      │
│                   (src/layout/mod.rs)                        │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 与 Tensor 模块的接口

```rust
// src/tensor/mod.rs

use crate::storage::{RawStorage, Storage, StorageMut, StorageOwned, StorageShared};

/// 张量核心结构
pub struct TensorBase<S, D>
where
    S: RawStorage,
    D: Dimension,
{
    /// 底层存储
    storage: S,
    
    /// 形状
    shape: D,
    
    /// 步长（有符号，单位为元素）
    strides: D,
    
    /// 数据起始偏移量
    offset: usize,
}

impl<S, D> TensorBase<S, D>
where
    S: Storage,
    D: Dimension,
{
    /// 返回数据起始指针。
    #[inline]
    pub fn as_ptr(&self) -> *const S::Elem {
        unsafe { self.storage.as_ptr().add(self.offset) }
    }

    /// 返回元素数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.size()
    }

    /// 检查是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S, D> TensorBase<S, D>
where
    S: StorageMut,
    D: Dimension,
{
    /// 返回可变数据指针。
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut S::Elem {
        unsafe { self.storage.as_mut_ptr().add(self.offset) }
    }

    /// 填充张量。
    #[inline]
    pub fn fill(&mut self, value: S::Elem)
    where
        S::Elem: Clone,
    {
        // 使用迭代器遍历并填充
        for idx in self.index_iter() {
            unsafe {
                *self.as_mut_ptr().add(self.index_to_offset(&idx)) = value.clone();
            }
        }
    }
}

impl<A, D> TensorBase<Owned<A>, D>
where
    D: Dimension,
{
    /// 创建全零张量。
    #[inline]
    pub fn zeros(shape: D) -> Self
    where
        A: Default + Clone,
    {
        let len = shape.size();
        Self {
            storage: Owned::zeros(len),
            shape,
            strides: D::default_strides(&shape),
            offset: 0,
        }
    }
}
```

### 11.3 与 Layout 模块的接口

```rust
// Storage 为 Layout 提供的信息：

// 1. 对齐状态
impl<S: RawStorage> TensorBase<S, D> {
    pub fn is_aligned(&self) -> bool {
        self.storage.is_aligned()
    }

    pub fn alignment(&self) -> usize {
        // 对于 Owned，返回 64
        // 对于 View，检查指针的实际对齐
        let ptr = self.as_ptr();
        let mut align = 1;
        while (ptr as usize) % (align * 2) == 0 {
            align *= 2;
        }
        align
    }
}

// 2. 连续性检查
impl<S: Storage, D: Dimension> TensorBase<S, D> {
    pub fn is_f_contiguous(&self) -> bool {
        // 检查步长是否符合 F-order 连续布局
        // ...
    }

    pub fn is_c_contiguous(&self) -> bool {
        // 检查步长是否符合 C-order 连续布局
        // ...
    }
}
```

### 11.4 与 Parallel 模块的接口

```rust
// 并行迭代需要存储满足 Send/Sync

#[cfg(feature = "parallel")]
impl<S, D> TensorBase<S, D>
where
    S: Storage + Sync,
    S::Elem: Send,
    D: Dimension,
{
    /// 并行迭代器。
    pub fn par_iter(&self) -> ParIter<'_, S, D> {
        ParIter::new(self)
    }
}

#[cfg(feature = "parallel")]
impl<S, D> TensorBase<S, D>
where
    S: StorageMut + Send,
    S::Elem: Send,
    D: Dimension,
{
    /// 并行可变迭代器。
    pub fn par_iter_mut(&mut self) -> ParIterMut<'_, S, D> {
        ParIterMut::new(self)
    }
}
```

### 11.5 类型转换流程

```
存储类型转换

┌─────────┐    into_owned()    ┌─────────┐
│  View   │ ─────────────────► │  Owned  │
└─────────┘                    └─────────┘
     │                              ▲
     │ to_owned()                   │
     ▼                              │
┌─────────┐    into_owned()    ┌─────────┐
│ViewMut  │ ─────────────────► │  Owned  │
└─────────┘                    └─────────┘
                                    ▲
     into_owned()                   │
     ┌──────────────────────────────┘
     │
┌─────────┐
│ArcRepr  │
└─────────┘
     │
     │ try_into_owned()
     ▼
┌─────────┐
│  Owned  │  (如果 Arc 是唯一的)
└─────────┘
     │
     │ clone() + ArcRepr::from_owned()
     ▼
┌─────────┐
│ArcRepr  │
└─────────┘
```

---

## 12. 实现任务分解

### 12.1 任务列表

每个任务预估约 10-15 分钟。

| 任务 | 内容 | 预估时间 | 依赖 |
|------|------|----------|------|
| **Task 1** | 定义 `RawStorage` trait | 15 min | 无 |
| **Task 2** | 定义 `Storage` trait | 10 min | Task 1 |
| **Task 3** | 定义 `RawStorageMut` 和 `StorageMut` traits | 10 min | Task 2 |
| **Task 4** | 定义 `StorageOwned` trait | 10 min | Task 3 |
| **Task 5** | 定义 `StorageShared` trait | 10 min | Task 2 |
| **Task 6** | 实现 `alloc.rs` 对齐分配器 | 20 min | 无 |
| **Task 7** | 实现 `Owned<A>` 结构和所有 traits | 20 min | Task 1-6 |
| **Task 8** | 实现 `ViewRepr<&A>` 结构和 traits | 15 min | Task 1-5 |
| **Task 9** | 实现 `ViewMutRepr<&mut A>` 结构和 traits | 15 min | Task 1-5 |
| **Task 10** | 实现 `ArcRepr<A>` 及 `make_mut()` | 25 min | Task 1-5, 7 |
| **Task 11** | 实现 `traits.rs` marker traits | 10 min | Task 1-10 |
| **Task 12** | 单元测试和文档 | 30 min | Task 1-11 |

### 12.2 任务详情

#### Task 1: 定义 RawStorage trait

```rust
// 预期产出
pub unsafe trait RawStorage {
    type Elem;
    type Device: Device;
    fn as_ptr(&self) -> *const Self::Elem;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { ... }
    fn is_aligned_to(&self, align: usize) -> bool { ... }
    fn is_aligned(&self) -> bool { ... }
}
```

**验证点**:
- [ ] trait 编译通过
- [ ] 关联类型正确定义
- [ ] 默认方法实现正确

#### Task 2: 定义 Storage trait

```rust
// 预期产出
pub unsafe trait Storage: RawStorage {
    fn get(&self, index: usize) -> Option<&Self::Elem>;
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Elem;
    fn as_slice(&self) -> &[Self::Elem];
    fn iter(&self) -> StorageIter<'_, Self>;
}
```

**验证点**:
- [ ] 继承 RawStorage
- [ ] 安全访问方法实现

#### Task 3: 定义 RawStorageMut 和 StorageMut

```rust
// 预期产出
pub unsafe trait RawStorageMut: RawStorage {
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;
}

pub unsafe trait StorageMut: Storage + RawStorageMut {
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Elem>;
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Elem;
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
    fn fill(&mut self, value: Self::Elem);
    fn copy_from<Src: Storage<Elem = Self::Elem>>(&mut self, src: &Src);
}
```

#### Task 4: 定义 StorageOwned trait

```rust
// 预期产出
pub unsafe trait StorageOwned: StorageMut + Clone {
    type Elem: Clone;
    fn zeros(len: usize) -> Self where Self::Elem: Default;
    fn from_elem(len: usize, value: Self::Elem) -> Self;
    fn from_vec(vec: Vec<Self::Elem>) -> Self;
    fn into_vec(self) -> Vec<Self::Elem>;
    fn to_owned(&self) -> Self;
    fn capacity(&self) -> usize;
    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), ()>;
}
```

#### Task 5: 定义 StorageShared trait

```rust
// 预期产出
pub unsafe trait StorageShared: Storage + Clone {
    type Elem: Clone;
    fn is_unique(&self) -> bool;
    fn make_mut(&mut self) -> &mut [Self::Elem];
    fn try_into_owned(self) -> Result<Owned<Self::Elem>, Self>;
    fn ref_count(&self) -> usize;
}
```

#### Task 6: 实现 alloc.rs

**关键实现**:
- `AlignedAlloc::alloc(size, align)`
- `AlignedAlloc::alloc_zeroed(size, align)`
- `AlignedAlloc::dealloc(ptr, size, align)`
- `TypedAlloc<A>` 包装器
- `layout_utils` 计算工具

**验证点**:
- [ ] 64 字节对齐正确
- [ ] no_std 编译通过
- [ ] 内存泄漏测试

#### Task 7: 实现 Owned<A>

**关键实现**:
- 结构定义
- 构造方法（`new`, `zeros`, `from_vec`, `from_elem`）
- 所有 trait 实现（`RawStorage` → `StorageOwned`）
- `Send`/`Sync` 实现
- `From` traits

**验证点**:
- [ ] 分配对齐正确
- [ ] 所有 trait 方法工作
- [ ] 克隆是深拷贝

#### Task 8: 实现 ViewRepr<&A>

**关键实现**:
- 结构定义（含 PhantomData）
- 构造方法（`from_raw_parts`, `from_slice`, `slice`）
- `Clone`/`Copy` 实现（O(1)）
- `RawStorage` 和 `Storage` 实现
- 不实现 `RawStorageMut`/`StorageMut`

**验证点**:
- [ ] 生命周期约束正确
- [ ] 克隆是 O(1)
- [ ] Send/Sync 条件正确

#### Task 9: 实现 ViewMutRepr<&mut A>

**关键实现**:
- 结构定义
- 构造方法
- 所有 trait 实现
- **刻意不实现 Clone**

**验证点**:
- [ ] 独占语义正确
- [ ] Clone 未实现
- [ ] Send 实现但 Sync 不实现

#### Task 10: 实现 ArcRepr<A>

**关键实现**:
- 结构定义（使用 `Arc<Vec<A>>`）
- `make_mut()` 原子性 CoW
- `try_into_owned()`
- `StorageShared` 实现
- `Clone` 实现（浅拷贝）

**验证点**:
- [ ] 引用计数正确
- [ ] CoW 语义正确
- [ ] 多线程安全

#### Task 11: 实现 marker traits

**关键实现**:
- `IsOwned`, `IsView`, `IsViewMut`, `IsArc`
- `IsClonable`, `IsDeepClone`, `IsShallowClone`
- Sealed trait 模式
- 各存储类型的实现

#### Task 12: 测试和文档

**测试清单**:
- [ ] `Owned` 分配/释放/克隆测试
- [ ] `ViewRepr` 生命周期/克隆测试
- [ ] `ViewMutRepr` 独占语义测试
- [ ] `ArcRepr` CoW/引用计数测试
- [ ] 对齐分配测试
- [ ] Send/Sync 测试
- [ ] no_std 编译测试

**文档清单**:
- [ ] 所有 trait 的文档注释
- [ ] 所有结构体的文档注释
- [ ] 使用示例
- [ ] Safety 文档

---

## 13. 设计决策记录

### 13.1 为什么使用 trait 层次而非单一 trait

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **Trait 层次** | 细粒度约束、清晰语义、复用性好 | 类型复杂 | **采用** |
| 单一 Storage trait | 简单 | 无法区分只读/可写、泛型约束不精确 | 不采用 |
| 多个独立 traits | 灵活 | 缺乏层次关系、语义不清 | 不采用 |

**理由**: Trait 层次允许在不同抽象级别上约束泛型参数。例如，`fn foo<S: Storage>(s: &S)` 接受任何可读存储，而 `fn bar<S: StorageMut>(s: &mut S)` 仅接受可写存储。

### 13.2 为什么 ViewMutRepr 不实现 Clone

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **不实现 Clone** | 强制独占语义、避免别名 | 使用不便（需重新借用） | **采用** |
| 实现 Clone | 使用方便 | 破坏独占语义、违反 Rust 借用规则 | 不采用 |

**理由**: Rust 的借用规则要求可变引用必须独占。如果 `ViewMutRepr` 可克隆，将允许多个可变引用指向同一数据，违反 Rust 安全保证。

### 13.3 为什么使用 64 字节默认对齐

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **64 字节** | AVX-512 优化、缓存行对齐 | 小数组内存浪费 | **采用** |
| 16 字节 | SSE/NEON 对齐 | AVX-512 未对齐 | 不采用 |
| 32 字节 | AVX 对齐 | AVX-512 未对齐 | 不采用 |
| 动态对齐 | 灵活 | 复杂、运行时开销 | 可选优化 |

**理由**: 64 字节是 AVX-512 的缓存行大小，也是现代 CPU 的常见缓存行大小。对齐访问可显著提升 SIMD 性能。小数组优化可自动降级到自然对齐。

### 13.4 为什么 ArcRepr 不实现 StorageMut

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **通过 make_mut() 访问** | 明确 CoW 语义、避免意外复制 | API 不一致 | **采用** |
| 实现 StorageMut | API 一致 | 隐藏 CoW 成本、意外复制 | 不采用 |

**理由**: `ArcRepr` 的可变访问涉及潜在的 O(n) 复制。通过 `make_mut()` 显式调用，用户明确知晓可能的性能影响。

### 13.5 为什么使用 Sealed Trait 模式

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **Sealed Trait** | 防止外部实现、API 稳定 | 略复杂 | **采用** |
| 开放 trait | 简单 | 外部可实现、API 不稳定 | 不采用 |

**理由**: Storage traits 是 Xenon 内部契约，不允许外部类型实现。Sealed trait 模式确保只有 Xenon 定义的存储类型满足这些约束，维护 API 稳定性。

### 13.6 为什么 Device 使用关联类型而非泛型

| 备选方案 | 优点 | 缺点 | 决策 |
|----------|------|------|------|
| **关联类型** | 类型推断简单、签名清晰 | 每种存储类型固定一个设备 | **采用** |
| 泛型参数 | 同一存储类型可多设备 | 类型推断复杂、签名冗长 | 不采用 |

**理由**: 在当前设计下，`Owned<A>` 只能在 CPU 上，`CudaOwned<A>` 只能在 GPU 上。每种存储类型与设备一一对应，关联类型更合适。

---

## 14. 参考资料

- [ndarray Storage Traits](https://docs.rs/ndarray/latest/ndarray/trait.RawData.html)
- [Rust Alloc Module](https://doc.rust-lang.org/alloc/alloc/index.html)
- [Arc and Reference Counting](https://doc.rust-lang.org/std/sync/struct.Arc.html)
- [Memory Layout in Rust](https://doc.rust-lang.org/nomicon/data.html)
- [AVX-512 Programming](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

---

*Xenon 存储系统模块设计 — v1.0*
