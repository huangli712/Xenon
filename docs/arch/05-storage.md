# 存储系统模块设计

> 文档编号: 05 | 模块: `src/storage/` | 阶段: Phase 2
> 前置文档: `02-dimension.md`, `03-element-types.md`, `04-complex-type.md`
> 需求参考: 需求说明书 §6

---

## 1. 模块定位

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| 存储抽象 | 定义统一的 `Storage` trait 层次，支持多种存储模式 | 具体运算逻辑（由 `ops/` 提供） |
| 内存管理 | 拥有、借用、共享三种所有权语义 | 并行调度（由 `parallel` 模块提供） |
| 对齐分配 | 64 字节对齐的内存分配器，优化 SIMD 性能 | 高级线性代数（矩阵分解等） |
| 类型安全 | 通过 trait 约束在编译期保证访问权限正确性 | GPU 存储后端（当前仅 CPU） |
| 多级访问 | 只读、可写、拥有三种访问控制级别 | 迭代器实现（由 `iter/` 提供） |
| ZST 安全 | 零大小类型和空数组操作不引发未定义行为 | — |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 零开销抽象 | 不同存储模式编译为不同类型，无运行时判断 |
| 类型安全 | 不可变视图编译期禁止写入，可变视图禁止克隆 |
| 统一 trait 层次 | `RawStorage → Storage → StorageMut → StorageOwned`，逐级增强能力 |
| 最小依赖 | 仅依赖 `core`/`alloc`，`std` 通过 feature gate 可选 |

---

## 2. 文件位置

```
src/storage/
├── mod.rs             # Storage trait 层次定义，模块导出
├── owned.rs           # Owned<A> 拥有型存储
├── view.rs            # ViewRepr<T> 不可变视图
├── view_mut.rs        # ViewMutRepr<T> 可变视图
├── arc.rs             # ArcRepr<A> 原子引用计数存储
├── alloc.rs           # 64 字节对齐分配器
└── traits.rs          # IsOwned, IsView 等 marker traits
```

单文件设计理由：各文件职责清晰，存储类型之间高度相关但不适合合并，拆分保持可维护性。

---

## 3. 依赖关系

### 3.1 依赖图（ASCII）

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
    │ owned.rs │    │ view.rs  │    │view_mut  │
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

### 3.2 类型级依赖

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `core` | `*const T`, `*mut T`, `NonNull<T>`, `PhantomData<T>` |
| `alloc` | `Vec<A>`, `Arc<A>`, `alloc`/`dealloc` |
| `element` | 元素类型约束（通过 trait bound 间接使用） |

### 3.3 依赖方向声明

> **依赖方向：单向向下。** `storage/` 仅依赖 `core`/`alloc` 和 `element` 的类型约束，不被 `dimension`/`layout` 等模块依赖。`tensor/` 和 `iter/` 模块消费 storage 的 trait 和类型。

---

## 4. 公共 API 设计

### 4.1 四种存储模式设计哲学

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

#### 访问级别与所有权矩阵

| 存储模式 | 拥有数据 | 可读 | 可写 | 克隆语义 | 分配方式 |
|----------|:--------:|:----:|:----:|----------|----------|
| `Owned<A>` | ✅ | ✅ | ✅ | 深拷贝 | 64 字节对齐堆分配 |
| `ViewRepr<&'a A>` | ❌ (借用) | ✅ | ❌ | O(1) 元数据拷贝 | 无分配 |
| `ViewMutRepr<&'a mut A>` | ❌ (独占借用) | ✅ | ✅ | 不可克隆 | 无分配 |
| `ArcRepr<A>` | ✅ (共享) | ✅ | make_mut() | 浅拷贝 (引用计数+1) | 写时按需分配 |

#### 设计权衡对比

| 考量 | Owned | View | ViewMut | Arc |
|------|-------|------|---------|-----|
| **创建开销** | 高 (分配) | 低 (借用) | 低 (借用) | 中 (Arc 包装) |
| **克隆开销** | O(n) | O(1) | 不可克隆 | O(1) |
| **写入开销** | 无 | 不可写 | 无 | 可能 O(n) (CoW) |
| **线程安全** | Send+Sync | Send+Sync | Send only | Send+Sync |
| **典型用途** | 创建、运算结果 | 切片、子数组 | 原地修改 | 跨线程共享、延迟复制 |

### 4.2 Storage Trait 层次

```
RawStorage                    (最底层，提供原始指针访问)
    │
    ├── Storage : RawStorage  (安全读取)
    │       │
    │       ├── StorageMut : Storage + RawStorageMut  (安全写入)
    │       │       │
    │       │       └── StorageOwned : StorageMut  (拥有数据)
    │       │
    │       └── StorageShared : Storage  (共享数据，如 Arc)
    │
    └── RawStorageMut : RawStorage  (原始可变指针)
```

### 4.3 RawStorage Trait

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

    /// 返回数据起始位置的原始指针。
    fn as_ptr(&self) -> *const Self::Elem;

    /// 返回存储的元素数量。
    fn len(&self) -> usize;

    /// 检查存储是否为空。
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 检查指针是否满足指定对齐要求。
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

### 4.4 RawStorageMut Trait

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

### 4.5 Storage Trait

```rust
/// 安全的存储读取访问。
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(storage.get(0), Some(&1.0));
/// ```
pub unsafe trait Storage: RawStorage {
    /// 获取指定索引处元素的不可变引用。
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
    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        // SAFETY: Storage 保证所有元素已初始化
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}
```

### 4.6 StorageMut Trait

```rust
/// 安全的存储读写访问。
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
}
```

### 4.7 StorageOwned Trait

```rust
/// 拥有数据所有权的存储。
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::zeros(100);
/// let cloned = storage.to_owned();
/// ```
pub unsafe trait StorageOwned: StorageMut + Clone {
    /// 元素类型
    type Elem: Clone;

    /// 分配指定大小的存储，用零填充。
    fn zeros(len: usize) -> Self
    where
        Self::Elem: Default;

    /// 分配指定大小的存储，用指定值填充。
    fn from_elem(len: usize, value: Self::Elem) -> Self;

    /// 从 Vec 构造存储。
    fn from_vec(vec: Vec<Self::Elem>) -> Self;

    /// 从迭代器构造存储。
    fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self;

    /// 将存储转换为 Vec。
    fn into_vec(self) -> Vec<Self::Elem>;

    /// 创建存储的深拷贝。
    fn to_owned(&self) -> Self;

    /// 返回存储的容量。
    fn capacity(&self) -> usize;

    /// 尝试调整存储容量。
    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), ()>;
}
```

### 4.8 StorageShared Trait

```rust
/// 共享存储的特殊操作。
///
/// 实现 `StorageShared` 的类型允许多个所有者共享同一数据，
/// 通常通过引用计数实现。
pub unsafe trait StorageShared: Storage + Clone {
    /// 元素类型
    type Elem: Clone;

    /// 检查是否为唯一所有者。
    fn is_unique(&self) -> bool;

    /// 获取数据的独占可变访问（写时复制）。
    fn make_mut(&mut self) -> &mut [Self::Elem];

    /// 尝试获取独占所有权，不复制数据。
    fn try_into_owned(self) -> Result<Owned<Self::Elem>, Self>
    where
        Self: Sized;

    /// 获取当前引用计数。
    fn ref_count(&self) -> usize;
}
```

### 4.9 Trait 实现矩阵

| 存储类型 | RawStorage | Storage | RawStorageMut | StorageMut | StorageOwned | StorageShared |
|----------|:----------:|:-------:|:-------------:|:----------:|:------------:|:-------------:|
| `Owned<A>` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `ViewRepr<&A>` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `ViewMutRepr<&mut A>` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `ArcRepr<A>` | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |

### 4.10 Good/Bad 对比

```rust
// Good - 使用 Storage trait 约束接受任何可读存储
fn process<S: Storage<Elem = f64>>(storage: &S) {
    let slice = storage.as_slice();
    // ...
}

// Bad - 硬编码 Owned 类型，拒绝视图
fn process_bad(storage: &Owned<f64>) {
    let slice = storage.as_slice();
    // ...
}
```

```rust
// Good - ArcRepr 通过 make_mut 显式获取写访问
fn modify_arc(arc: &mut ArcRepr<f64>) {
    let data = arc.make_mut();  // 显式 CoW
    data[0] = 10.0;
}

// Bad - 直接尝试通过 Arc 写入（编译错误）
// arc.as_mut_ptr();  // ArcRepr 不实现 StorageMut
```

---

## 5. 内部实现设计

### 5.1 Owned\<A\> 结构体

```rust
/// 拥有型存储。
///
/// `Owned<A>` 拥有数据的完全所有权，数据存储在堆上，
/// 使用 64 字节对齐分配以优化 SIMD 操作。
#[derive(Debug, Clone, PartialEq)]
pub struct Owned<A> {
    /// 内部数据存储（64 字节对齐堆分配）
    data: Vec<A>,
}

impl<A> Owned<A> {
    /// 默认对齐值：64 字节（AVX-512 缓存行）
    pub const DEFAULT_ALIGNMENT: usize = 64;
}
```

### 5.2 64 字节对齐分配器

```rust
/// 64 字节对齐的内存分配器。
pub struct AlignedAlloc;

impl AlignedAlloc {
    /// 默认对齐值：64 字节
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// 分配指定大小和对齐的内存块，不初始化。
    ///
    /// # Panics
    ///
    /// - `align` 不是 2 的幂
    /// - `size` 为 0
    /// - 内存分配失败
    pub fn alloc(size: usize, align: usize) -> NonNull<u8>;

    /// 分配并零初始化。
    pub fn alloc_zeroed(size: usize, align: usize) -> NonNull<u8>;

    /// 释放内存。
    ///
    /// # Safety
    ///
    /// - `ptr` 必须是由 `alloc` 或 `alloc_zeroed` 返回的
    /// - `size` 和 `align` 必须与分配时相同
    pub unsafe fn dealloc(ptr: NonNull<u8>, size: usize, align: usize);

    /// 检查是否应该使用对齐分配（小数组降级）。
    #[inline]
    pub fn should_use_aligned_alloc(elem_size: usize, count: usize, align: usize) -> bool {
        let total = elem_size * count;
        total > align
    }
}
```

**安全性论证**：`AlignedAlloc` 使用 `alloc::alloc::Layout` 确保对齐值是 2 的幂且总大小合法。分配失败时调用 `handle_alloc_error` 而非返回空指针，避免 UB。

### 5.3 ViewRepr\<&'a A\> 结构体

```rust
/// 不可变视图存储。
#[derive(Debug)]
pub struct ViewRepr<A> {
    ptr: *const A,
    len: usize,
    _marker: PhantomData<A>,
}

/// 类型别名
pub type View<'a, A> = ViewRepr<&'a A>;

// Clone 仅复制指针和长度，不复制数据（O(1)）
impl<'a, A> Clone for ViewRepr<&'a A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, len: self.len, _marker: PhantomData }
    }
}

impl<'a, A> Copy for ViewRepr<&'a A> {}
```

### 5.4 ViewMutRepr\<&'a mut A\> 结构体

```rust
/// 可变视图存储。
///
/// **不可克隆**：独占语义意味着同一时刻只能存在一个可变引用。
#[derive(Debug)]
pub struct ViewMutRepr<A> {
    ptr: *mut A,
    len: usize,
    _marker: PhantomData<A>,
}

/// 类型别名
pub type ViewMut<'a, A> = ViewMutRepr<&'a mut A>;

// 刻意不实现 Clone — Rust 借用规则要求可变引用独占
```

### 5.5 ArcRepr\<A\> 结构体

```rust
/// 原子引用计数存储。
///
/// 使用 `Arc` 共享数据所有权，通过 `make_mut()` 实现写时复制 (CoW)。
#[derive(Debug)]
pub struct ArcRepr<A> {
    inner: Arc<Vec<A>>,
    len: usize,
    offset: usize,
}
```

**写时复制流程**：

```
make_mut() 执行流程
┌──────────────────────┐
│      make_mut()      │
└──────────┬───────────┘
           │
           ▼
  ┌────────────────┐
  │ 引用计数 == 1? │
  └───────┬────────┘
          │
    ┌─────┴─────┐
    │ YES       │ NO
    ▼           ▼
 直接返回    分配新内存 → 复制数据 → 原引用计数减 1
 &mut [A]    (O(n) 深拷贝)
```

### 5.6 Marker Traits

```rust
/// 标记存储类型拥有数据。
pub unsafe trait IsOwned: RawStorage {}

/// 标记存储类型是视图（借用）。
pub unsafe trait IsView: RawStorage {}

/// 标记存储类型是可变视图。
pub unsafe trait IsViewMut: RawStorage {}

/// 标记存储类型使用 Arc 共享。
pub unsafe trait IsArc: RawStorage {}
```

使用 Sealed trait 模式防止外部类型实现。

### 5.7 Send/Sync 实现规则

| 存储类型 | Send | Sync | 原因 |
|----------|:----:|:----:|------|
| `Owned<A>` | A: Send + Sync | A: Send + Sync | 拥有数据，等价于 `Vec<A>` |
| `ViewRepr<&'a A>` | A: Sync | A: Sync | 共享借用需要 Sync 才能跨线程共享 |
| `ViewMutRepr<&'a mut A>` | A: Send | ❌ 永远不 | 独占借用可转移但不可共享 |
| `ArcRepr<A>` | A: Send + Sync | A: Send + Sync | Arc 内部原子操作保证线程安全 |

---

## 6. ZST 和空数组处理

| 场景 | 预期行为 | 安全性论证 |
|------|----------|-----------|
| ZST 元素类型 | `Owned<()>` 分配 0 字节内存，len 正常计算 | `alloc::alloc::Layout` 对 size=0 有特殊处理 |
| 空数组 `len=0` | `as_ptr()` 返回非空悬垂指针，`as_slice()` 返回空切片 | `Vec` 保证空时 `as_ptr()` 非空 |
| ZST + 空数组 | 不引发分配，不引发 UB | ZST 不需要实际内存 |

---

## 7. no_std 兼容性

```rust
#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, alloc_zeroed, Layout};
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `Owned<A>` | ✅ | 使用 `alloc::vec::Vec` |
| `ViewRepr` | ✅ | 仅使用 `core` |
| `ViewMutRepr` | ✅ | 仅使用 `core` |
| `ArcRepr<A>` | ✅ | 使用 `alloc::sync::Arc` |
| `AlignedAlloc` | ✅ | 使用 `alloc::alloc` |

---

## 8. 实现任务拆分

### Wave 1: 基础 trait 定义

- [ ] **T1**: 创建 `src/storage/mod.rs` 骨架
  - 文件: `src/storage/mod.rs`
  - 内容: 模块声明、子模块文件占位、公共导出声明
  - 测试: 编译通过
  - 前置: dimension 模块完成
  - 预计: 5 min

- [ ] **T2**: 定义 `RawStorage` trait
  - 文件: `src/storage/mod.rs`
  - 内容: `unsafe trait RawStorage` 完整定义，包含 `as_ptr`/`len`/`is_empty`/`is_aligned_to`/`is_aligned`
  - 测试: `test_raw_storage_compile`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 定义 `Storage` trait
  - 文件: `src/storage/mod.rs`
  - 内容: `unsafe trait Storage: RawStorage`，包含 `get`/`get_unchecked`/`as_slice`
  - 测试: `test_storage_compile`
  - 前置: T2
  - 预计: 10 min

- [ ] **T4**: 定义 `RawStorageMut` 和 `StorageMut` traits
  - 文件: `src/storage/mod.rs`
  - 内容: `RawStorageMut` 和 `StorageMut` 完整定义
  - 测试: `test_storage_mut_compile`
  - 前置: T3
  - 预计: 10 min

- [ ] **T5**: 定义 `StorageOwned` 和 `StorageShared` traits
  - 文件: `src/storage/mod.rs`
  - 内容: `StorageOwned` 和 `StorageShared` 完整定义
  - 测试: `test_storage_traits_compile`
  - 前置: T4
  - 预计: 10 min

### Wave 2: 对齐分配器和 Owned 实现

- [ ] **T6**: 实现 `alloc.rs` 对齐分配器
  - 文件: `src/storage/alloc.rs`
  - 内容: `AlignedAlloc` 结构体，`alloc`/`alloc_zeroed`/`dealloc`/`should_use_aligned_alloc`
  - 测试: `test_aligned_alloc_64`, `test_aligned_alloc_zeroed`
  - 前置: T1
  - 预计: 15 min

- [ ] **T7**: 实现 `Owned<A>` 结构体和构造方法
  - 文件: `src/storage/owned.rs`
  - 内容: `Owned<A>` 定义，`new`/`with_capacity`/`from_vec`/`from_vec_aligned`/`zeros`/`from_elem`
  - 测试: `test_owned_new`, `test_owned_from_vec`
  - 前置: T6
  - 预计: 15 min

- [ ] **T8**: 实现 `Owned<A>` 所有 trait 实现
  - 文件: `src/storage/owned.rs`
  - 内容: `RawStorage`/`Storage`/`StorageMut`/`StorageOwned` 实现，`Send`/`Sync`/`From`/`Default`
  - 测试: `test_owned_storage_trait`, `test_owned_send_sync`
  - 前置: T5, T7
  - 预计: 15 min

### Wave 3: 视图和 Arc 实现

- [ ] **T9**: 实现 `ViewRepr<&'a A>`
  - 文件: `src/storage/view.rs`
  - 内容: 结构定义、`from_raw_parts`/`from_slice`/`from_owned`/`slice`、Clone/Copy、`RawStorage`/`Storage` 实现
  - 测试: `test_view_clone_o1`, `test_view_lifetime`
  - 前置: T5
  - 预计: 15 min

- [ ] **T10**: 实现 `ViewMutRepr<&'a mut A>`
  - 文件: `src/storage/view_mut.rs`
  - 内容: 结构定义、`from_raw_parts_mut`/`from_mut_slice`/`from_owned`/`as_view`、不实现 Clone、`RawStorage`/`RawStorageMut`/`Storage`/`StorageMut` 实现
  - 测试: `test_view_mut_no_clone`, `test_view_mut_exclusive`
  - 前置: T5
  - 预计: 15 min

- [ ] **T11**: 实现 `ArcRepr<A>` 及 `make_mut()`
  - 文件: `src/storage/arc.rs`
  - 内容: 结构定义、`from_vec`/`from_owned`/`ref_count`/`is_unique`/`make_mut`/`try_into_owned`/`into_owned`、`StorageShared` 实现
  - 测试: `test_arc_cow`, `test_arc_ref_count`
  - 前置: T5, T7
  - 预计: 20 min

### Wave 4: Marker traits 和收尾

- [ ] **T12**: 实现 `traits.rs` marker traits
  - 文件: `src/storage/traits.rs`
  - 内容: `IsOwned`/`IsView`/`IsViewMut`/`IsArc`，Sealed trait 模式，各存储类型的实现
  - 测试: `test_marker_traits`
  - 前置: T8, T9, T10, T11
  - 预计: 10 min

- [ ] **T13**: 集成测试和文档
  - 文件: `tests/storage.rs`
  - 内容: 跨存储类型交互测试、ZST 测试、空数组测试、no_std 编译验证
  - 测试: 完整集成测试套件
  - 前置: T12
  - 预计: 20 min

### 并行执行图

```
Wave 1: [T1] → [T2] → [T3] → [T4] → [T5]
                                  ↓
Wave 2:                    [T6] → [T7] → [T8]
                           ↓                ↓
Wave 3:              [T9] [T10]    →   [T11]
                           ↓
Wave 4:              [T12] → [T13]
```

---

## 9. 测试计划

### 9.1 测试分类

| 类型 | 位置 | 目的 |
|------|------|------|
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法 |
| 集成测试 | `tests/` | 验证跨存储类型交互 |
| 边界测试 | 集成测试中标注 | ZST、空数组、大数组 |
| 属性测试 | `tests/property/` | 随机生成验证不变量 |

### 9.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_owned_new_empty` | `Owned::new()` 创建空存储 | 高 |
| `test_owned_zeros` | `Owned::zeros(100)` 全零 | 高 |
| `test_owned_from_vec` | 从 Vec 构造并验证内容 | 高 |
| `test_owned_alignment` | 验证 64 字节对齐 | 高 |
| `test_owned_clone_deep` | 克隆后修改不影响原数据 | 高 |
| `test_view_from_slice` | 从切片创建视图 | 高 |
| `test_view_clone_o1` | 克隆仅复制元数据 | 中 |
| `test_view_lifetime` | 视图不能比源数据存活更久 | 高 |
| `test_view_mut_exclusive` | 可变视图独占语义 | 高 |
| `test_view_mut_no_clone` | 编译期验证不可克隆 | 高 |
| `test_arc_ref_count` | 引用计数正确 | 高 |
| `test_arc_cow` | 写时复制语义正确 | 高 |
| `test_arc_make_mut_unique` | 唯一时 make_mut 无复制 | 中 |
| `test_aligned_alloc_64` | 分配器 64 字节对齐 | 高 |
| `test_zst_no_ub` | ZST 元素类型不引发 UB | 高 |
| `test_empty_array` | 空数组操作安全 | 高 |

### 9.3 边界测试场景

| 场景 | 预期行为 |
|------|----------|
| 空数组 `Owned::new()` | `len() == 0`, `is_empty() == true` |
| 单元素 `Owned::from_vec(vec![1.0])` | `len() == 1`, `get(0) == Some(&1.0)` |
| ZST 元素 `Owned::zeros(1000::<()>` | 不分配内存，不引发 UB |
| 大数组 16M 元素 | 正常分配和访问 |

### 9.4 属性测试不变量

| 不变量 | 测试方法 |
|--------|----------|
| `owned.to_owned().as_slice() == owned.as_slice()` | 随机元素类型和大小 |
| `view.clone().as_ptr() == view.as_ptr()` | 随机切片范围 |
| `arc.make_mut()` 后引用计数为 1 | 随机共享数量 |

---

## 10. 与其他模块的交互

### 10.1 与 Tensor 模块

`TensorBase<S, D>` 的 `S` 参数约束为 `Storage` 或 `StorageMut`，通过关联类型 `Elem` 获取元素类型。

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn as_ptr(&self) -> *const A {
        unsafe { self.storage.as_ptr().add(self.offset) }
    }
}
```

### 10.2 与 Layout 模块

Storage 提供对齐信息（`is_aligned()`），Layout 模块查询对齐状态更新 `LayoutFlags::ALIGNED`。

### 10.3 与 Parallel 模块

并行迭代要求 `S: Storage + Sync`（读）或 `S: StorageMut + Send`（写），由 storage 的 Send/Sync 实现保证。

---

## 11. 设计决策记录

### 决策 1：64 字节默认对齐

| 属性 | 值 |
|------|-----|
| 决策 | 使用 64 字节作为默认对齐 |
| 理由 | AVX-512 缓存行大小；现代 CPU 缓存行通常 64 字节；满足 SSE/AVX/AVX2/AVX-512 所有 SIMD 指令 |
| 替代方案 | 16 字节 — 放弃，AVX-512 未对齐 |
| 替代方案 | 32 字节 — 放弃，AVX-512 未对齐 |

### 决策 2：ArcRepr 不实现 StorageMut（CoW 策略）

| 属性 | 值 |
|------|-----|
| 决策 | `ArcRepr` 通过 `StorageShared::make_mut()` 提供写访问，不实现 `StorageMut` |
| 理由 | 可变访问涉及潜在 O(n) 复制，显式调用让用户知晓性能影响 |
| 替代方案 | 实现 `StorageMut` — 放弃，隐藏 CoW 成本 |

### 决策 3：ArcRepr 作为统一 trait 体系的一部分

| 属性 | 值 |
|------|-----|
| 决策 | `ArcRepr<A>` 纳入统一 trait 体系（通过 `StorageShared`），而非独立类型 |
| 理由 | 更好的正交性，泛型代码可通过 `Storage` trait 统一处理所有存储模式 |
| 替代方案 | ndarray 风格 `ArcArray` 独立类型 — 放弃，增加类型复杂度 |

---

## 12. 性能考量

| 方面 | 设计决策 |
|------|----------|
| 对齐分配 | 64 字节对齐，小数组自动降级到自然对齐 |
| 视图克隆 | O(1)，仅复制指针和长度 |
| Arc 克隆 | O(1)，仅增加引用计数 |
| Arc make_mut | 唯一时 O(1)，非唯一时 O(n) 深拷贝 |
| Owned 克隆 | O(n) 深拷贝 |
| 内联 | 所有 `as_ptr`/`len`/`get` 标注 `#[inline]` |
| 单态化 | Storage trait 在泛型上下文中单态化，无虚调用开销 |

**性能数据（参考）**:

| 操作 | 开销 | 说明 |
|------|------|------|
| `Owned::zeros(1M)` | ~1ms | 包含分配和零初始化 |
| `View::clone()` | ~2ns | 仅复制 3 个字段 |
| `Arc::clone()` | ~5ns | 原子引用计数增加 |
| `Arc::make_mut()`（唯一） | ~2ns | 直接返回可变引用 |
| `Arc::make_mut()`（非唯一，1M 元素） | ~1ms | 深拷贝 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
