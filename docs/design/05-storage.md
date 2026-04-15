# 存储系统模块设计

> 文档编号: 05 | 模块: `src/storage/` | 阶段: Phase 2
> 前置文档: `02-dimension.md`, `03-element.md`, `04-complex.md`
> 需求参考: 需求说明书 §6
> 范围声明: 范围内

---

## 1. 模块定位

### 1.1 职责边界

| 职责     | 包含                                              | 不包含                             |
| -------- | ------------------------------------------------- | ---------------------------------- |
| 存储抽象 | 定义统一的 `Storage` trait 层次，支持多种存储模式 | 具体运算逻辑（由 `overload` 提供） |
| 内存管理 | 拥有、借用（只读/可写）与共享只读四种存储模式    | 并行调度（由 `parallel` 模块提供） |
| 对齐分配 | 64 字节对齐的内存分配器，优化 SIMD 性能           | 高级线性代数（矩阵分解等）         |
| 类型安全 | 通过 trait 约束在编译期保证访问权限正确性         | GPU 存储后端（当前仅 CPU）         |
| 多级访问 | 只读、可写、共享只读、拥有四种访问/持有模式       | 迭代器实现（由 `iter`模块 提供）   |
| ZST 安全 | 零大小类型和空数组操作不引发未定义行为            | —                                  |

### 1.2 设计原则

| 原则            | 体现                                                             |
| --------------- | ---------------------------------------------------------------- |
| 零开销抽象      | 不同存储模式编译为不同类型，无运行时判断                         |
| 类型安全        | 不可变视图编译期禁止写入，可变视图禁止克隆                       |
| 统一 trait 层次 | `RawStorage → Storage → StorageMut → StorageOwned`，逐级增强能力 |
| 最小依赖        | 无新增第三方依赖；实现以标准库与既有 `core`/`alloc` 能力为基础   |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (does not depend on layout; exposed under a `std` environment)  ← current module
L4: tensor (depends on storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/
```

---

## 2. 需求映射与范围约束

| 项目     | 内容                                                              |
| -------- | ----------------------------------------------------------------- |
| 需求映射 | 需求说明书 §6                                                     |
| 范围内   | `Storage` trait 层次、拥有/借用/共享存储模式、对齐分配与 ZST 处理 |
| 范围外   | GPU 后端、并行调度、张量运算逻辑、负步长布局解释                  |
| 非目标   | 引入第三方分配器依赖、共享可写存储模式或运行时类型擦除存储层      |

---

## 3. 文件位置

```
src/storage/
├── mod.rs             # Storage trait hierarchy and module exports
├── owned.rs           # Owned<A> owning storage
├── view.rs            # ViewRepr<'a, A> immutable view
├── view_mut.rs        # ViewMutRepr<'a, A> mutable view
├── arc.rs             # ArcRepr<A> shared read-only storage
├── alloc.rs           # 64-byte aligned allocator
└── traits.rs          # marker traits such as IsOwned and IsView
```

单文件设计理由：各文件职责清晰，存储类型之间高度相关但不适合合并，拆分保持可维护性。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

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

### 4.2 类型级依赖

| 来源模块         | 使用的类型/trait                                     |
| ---------------- | ---------------------------------------------------- |
| `core`           | `*const T`, `*mut T`, `NonNull<T>`, `PhantomData<T>` |
| `alloc`          | `Vec<A>`, `alloc`/`dealloc`                          |
| `std::sync`      | `Arc<A>`                                             |
| `crate::error`   | `XenonError`（用于 `try_reserve` 等可恢复错误）      |
| `crate::private` | `Sealed`（用于 marker trait 封闭实现）               |

### 4.2a 依赖合法性

| 项目           | 结论                           |
| -------------- | ------------------------------ |
| 新增第三方依赖 | 无                             |
| 合法性结论     | 符合需求说明书最小依赖限制     |
| 替代方案       | 不适用                         |

### 4.3 依赖方向声明

> **依赖方向：单向向下。** storage 模块不依赖 layout；实现层主要使用标准库及既有 `core` / `alloc` / `std::sync` 能力，并依赖 `crate::error` 提供 `XenonError`、依赖 `crate::private` 提供 `Sealed`。`storage/` 不直接依赖 `element` 模块；元素类型约束通过 `TensorBase` 的泛型参数间接体现（`Storage::Elem` 关联类型）。`tensor/`（参见 `07-tensor.md` §5）和 `iter/`（参见 `10-iterator.md` §4）模块消费 storage 的 trait 和类型。
>
> **前向引用：** `Strides` 的辅助接口（如 `Strides::from_slice()`、`.iter()`、`.as_slice()`）由 `06-layout.md` 定义并由上层 `tensor` / `iter` / `ffi` 路径消费。storage 文档仅引用这些能力，不在本模块重复定义 layout API。

---

## 5. 公共 API 设计

### 5.1 四种存储模式设计哲学

Xenon 采用 ndarray 风格的存储抽象，提供四种互补的存储模式：

```
Storage mode taxonomy
├── Owned
│   └── Owned<A> — owns data, readable/writable, deep-clone
│
├── Borrowed
│   ├── ViewRepr<'a, A> — shared borrow, read-only, O(1) clone
│   └── ViewMutRepr<'a, A> — exclusive borrow, readable/writable, non-cloneable
│
└── Shared read-only
    └── ArcRepr<A> — Arc-backed shared read-only handle, shallow-clone
```

#### 访问级别与所有权矩阵

| 存储模式             |   拥有数据    | 可读 |              可写              | 克隆语义            | 分配方式                        |
| -------------------- | :-----------: | :--: | :----------------------------: | ------------------- | ------------------------------- |
| `Owned<A>`           |      ✅       |  ✅  |               ✅               | 深拷贝              | 当前默认实现为 64 字节对齐堆分配（可配置；ZST 不分配） |
| `ViewRepr<'a, A>`    |   ❌ (借用)   |  ✅  |               ❌               | O(1) 元数据拷贝     | 无分配                          |
| `ViewMutRepr<'a, A>` | ❌ (独占借用) |  ✅  |               ✅               | 不可克隆            | 无分配                          |
| `ArcRepr<A>`         |   ✅ (共享)   |  ✅  | 仅通过 CoW 生成独占 owned 数据 | 浅拷贝 (引用计数+1) | 共享对齐缓冲，写时按需深拷贝    |

> **术语澄清：** `require.md` §6.2 的“共享只读引用”描述的是抽象访问语义，而不是唯一具体表示。对拥有型来源，Xenon 用 `ArcRepr<A>` 表达可共享底层缓冲区的只读结果；对 `ViewMutRepr<'a, A>` 的零拷贝降级，Xenon 用 `ViewRepr<'a, A>` 表达放弃写权限后的只读重借用。两者都满足“共享只读、不提供写访问”的语义，只是前者共享所有权，后者共享借用生命周期。`make_mut()` 只是 `ArcRepr<A>` 从共享只读状态显式转入独占 owned 缓冲的实现机制；它不意味着 `ArcRepr<A>` 本身提供共享可写访问，也不意味着它实现 `StorageMut`。

#### 抽象模式 ↔ 具体表示 对照表

| 抽象模式 | 具体表示 | 适用语境 / 说明 |
| -------- | -------- | --------------- |
| `Owned` | `Owned<A>` | 拥有底层分配，可零拷贝借出只读 / 可写视图，也可零拷贝降级为共享只读 |
| `WritableRef` | `ViewMutRepr<'a, A>` | 基于独占借用的可写引用 |
| `ReadOnlyRef` | `ViewRepr<'a, A>` | 普通只读借用视图 |
| `SharedReadOnlyRef` | `ArcRepr<A>` | 来自拥有型来源时的共享只读表示，提供共享所有权 |
| `SharedReadOnlyRef` | `ViewRepr<'a, A>` | 来自 `ViewMutRepr<'a, A>` 零拷贝降级时的共享只读表示；共享的是借用生命周期而非所有权 |

#### 设计权衡对比

| 考量         | Owned          | View         | ViewMut   | Arc                  |
| ------------ | -------------- | ------------ | --------- | -------------------- |
| **创建开销** | 高 (分配)      | 低 (借用)    | 低 (借用) | 中 (Arc 包装)        |
| **克隆开销** | O(n)           | O(1)         | 不可克隆  | O(1)                 |
| **写入开销** | 无             | 不可写       | 无        | 可能 O(n) (CoW)      |
| **线程安全** | Send+Sync      | Send+Sync    | Send only | Send+Sync            |
| **典型用途** | 创建、运算结果 | 切片、子数组 | 原地修改  | 跨线程共享、延迟复制 |

### 5.2 Storage Trait 层次

```
RawStorage                    (lowest layer, raw pointer access)
    │
    ├── Storage : RawStorage  (safe read access)
    │       │
    │       ├── StorageMut : Storage + RawStorageMut  (safe write access)
    │       │       │
    │       │       └── StorageOwned : StorageMut  (owns data)
    │       │
    │       └── StorageShared : Storage  (shared read-only data, e.g. ArcRepr)
    │
    └── RawStorageMut : RawStorage  (raw mutable pointer access)
```

### 5.3 RawStorage Trait

```rust
use core::ptr::NonNull;

/// Raw pointer access to underlying storage.
///
/// This is the most fundamental storage trait, providing no safety guarantees.
/// All storage types must implement this trait.
///
/// # Safety
///
/// Implementors must ensure:
/// - The pointer returned by `as_ptr()` remains valid for the storage's lifetime
/// - Multiple calls to `as_ptr()` on the same storage instance return the same address
pub unsafe trait RawStorage {
    /// The element type of the storage.
    type Elem;

    /// Returns a raw pointer to the start of the data.
    fn as_ptr(&self) -> *const Self::Elem;

    /// Returns the number of elements in storage.
    fn len(&self) -> usize;

    /// Checks if the storage is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks if the pointer satisfies the specified alignment requirement.
    #[inline]
    fn is_aligned_to(&self, align: usize) -> bool {
        (self.as_ptr() as usize) % align == 0
    }

    /// Checks if the storage satisfies the current default alignment (64 bytes).
    ///
    /// Note: 64 bytes is the current default implementation choice for SIMD-
    /// friendly owned buffers, not a hard requirement of the storage model.
    /// Callers needing a different alignment should use `is_aligned_to()`.
    #[inline]
    fn is_aligned(&self) -> bool {
        self.is_aligned_to(64)
    }
}
```

### 5.4 RawStorageMut Trait

```rust
/// Raw pointer access for mutable storage.
///
/// # Safety
///
/// Implementors must ensure:
/// - The pointer returned by `as_mut_ptr()` remains valid for the storage's lifetime
/// - No other mutable references to the same data exist (aliasing rules)
pub unsafe trait RawStorageMut: RawStorage {
    /// Returns a raw mutable pointer to the start of the data.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;

    /// Converts the storage to a NonNull pointer (for FFI).
    ///
    /// # Safety
    ///
    /// The storage must be non-empty.
    #[inline]
    unsafe fn as_non_null(&mut self) -> NonNull<Self::Elem> {
        NonNull::new_unchecked(self.as_mut_ptr())
    }
}
```

### 5.5 Storage Trait

````rust
/// Safe read access to the entire backing storage.
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(storage.get(0), Some(&1.0));
/// ```
pub unsafe trait Storage: RawStorage {
    /// Returns an immutable reference to the element at the given index.
    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Elem> {
        if index < self.len() {
            // SAFETY: index is bounds-checked
            Some(unsafe { &*self.as_ptr().add(index) })
        } else {
            None
        }
    }

    /// Returns an immutable reference to the element at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < self.len()`.
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Elem {
        &*self.as_ptr().add(index)
    }

    /// Returns a slice view of the storage-visible backing range.
    ///
    /// This is a storage-level API. Implementors must expose the visible range
    /// of the storage object itself: for subviews, `as_ptr()` must already be
    /// offset-adjusted and `len()` must cover only the storage-visible element
    /// range. Hidden allocator capacity or alignment padding must not be
    /// exposed through this API.
    ///
    /// This API still does **not** account for tensor-level `shape` /
    /// `strides` metadata, so callers must not treat it as an arbitrary logical
    /// tensor slice. The tensor-level zero-copy fast path lives in
    /// `TensorBase::as_slice()` (see `07-tensor.md §5.4a`).
    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        // SAFETY: Storage guarantees all elements are initialized
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}
````

### 5.6 StorageMut Trait

````rust
/// Safe read-write access to storage.
///
/// # Example
///
/// ```ignore
/// let mut storage: Owned<f64> = Owned::from_vec(vec![1.0, 2.0, 3.0]);
/// if let Some(slot) = storage.get_mut(0) {
///     *slot = 10.0;
/// }
/// assert_eq!(storage.get(0), Some(&10.0));
/// ```
pub unsafe trait StorageMut: Storage + RawStorageMut {
    /// Returns a mutable reference to the element at the given index.
    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Elem> {
        if index < self.len() {
            // SAFETY: index is bounds-checked
            Some(unsafe { &mut *self.as_mut_ptr().add(index) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure `index < self.len()`.
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Elem {
        &mut *self.as_mut_ptr().add(index)
    }

    /// Returns a mutable slice view of the storage-visible backing range.
    ///
    /// Like `Storage::as_slice()`, this is a storage-level API. The pointer
    /// must already be offset-adjusted to the storage-visible range, and hidden
    /// allocator padding must not be exposed here.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        // SAFETY: StorageMut guarantees all elements are initialized and exclusive access
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Fills the entire storage with the given value.
    #[inline]
    fn fill(&mut self, value: Self::Elem)
    where
        Self::Elem: Copy,
    {
        self.as_mut_slice().fill(value);
    }
}
````

> **警告：** ⚠️ `as_slice()` / `as_mut_slice()` 返回的是当前存储对象可见的 backing range，而不是原始分配器缓冲区；对带偏移的视图，该范围应已完成 offset 调整，且不得暴露容量尾部或对齐填充。对于非连续张量，调用方仍须结合 `shape` / `strides()` 解释逻辑元素顺序。

### 5.7 StorageOwned Trait

````rust
/// Storage that owns data.
///
/// # Example
///
/// ```ignore
/// let storage: Owned<f64> = Owned::zeros(100);
/// let cloned = storage.to_owned();
/// ```
pub unsafe trait StorageOwned: StorageMut + Clone {
    /// Allocates storage of the given size, zero-filled.
    fn zeros(len: usize) -> Self
    where
        Self::Elem: Default;

    /// Allocates storage of the given size, filled with the given value.
    fn from_elem(len: usize, value: Self::Elem) -> Self;

    /// Constructs storage from a Vec.
    fn from_vec(vec: Vec<Self::Elem>) -> Self
    where
        Self::Elem: Copy;

    /// Constructs storage from an iterator.
    fn from_iter<I: IntoIterator<Item = Self::Elem>>(iter: I) -> Self;

    /// Converts storage into a Vec.
    fn into_vec(self) -> Vec<Self::Elem>;

    /// Creates a deep copy of the storage.
    fn to_owned(&self) -> Self;

    /// Returns the capacity of the storage.
    fn capacity(&self) -> usize;

    /// Attempts to resize the storage capacity.
    fn try_reserve(&mut self, new_capacity: usize) -> Result<(), XenonError>;
}
````

### 5.8 StorageShared Trait

```rust
/// Special operations for shared storage.
///
/// Types implementing `StorageShared` allow multiple owners to share the same
/// read-only data, typically through reference counting.
pub unsafe trait StorageShared: Storage + Clone {
    /// Checks if this is the sole owner.
    fn is_unique(&self) -> bool;

    /// Triggers copy-on-write and yields exclusive mutable access to the
    /// underlying buffer.
    ///
    /// For `ArcRepr<A>`, this API is an explicit copy-on-write hook for
    /// obtaining temporary exclusive access to the buffer managed inside the
    /// shared storage object. It does **not** make `ArcRepr<A>` a
    /// shared-writable storage mode, does **not** imply `StorageMut` is
    /// implemented for `ArcRepr<A>`, and must not be interpreted as a public
    /// zero-copy `ArcRepr<A> -> Owned<A>` conversion.
    ///
    /// NOTE: This is NOT a shared-writable mode. It triggers copy-on-write,
    /// producing exclusive `&mut` access.
    ///
    /// The returned mutable slice is only an internal CoW access path.
    /// Public `into_owned_storage()` for `ArcRepr<A>` still allocates and
    /// copies regardless of reference count, per `require.md` §6.2.
    ///
    /// Note: sub-view offset and length are managed by `TensorBase`, not by
    /// `ArcRepr` (see design decision in §6.5).
    fn make_mut(&mut self) -> &mut [Self::Elem];

    /// Returns the current reference count.
    fn ref_count(&self) -> usize;
}
```

### 5.9 StorageIntoOwned Trait

消耗式转为 Owned 存储，用于 `into_owned()` 等需要物化独占 owned 结果的路径。

```rust
/// Storage types that can be converted into an owned tensor by consuming self.
///
/// - `Owned<A>` → O(1), returns self directly
/// - `ViewRepr`/`ViewMutRepr` → O(n), copies data
/// - `ArcRepr` → O(n), always allocates and copies into a fresh owned buffer
pub unsafe trait StorageIntoOwned: Storage {
    /// Consume this storage, returning an `Owned<A>` storage.
    fn into_owned_storage(self) -> Owned<Self::Elem>
    where
        Self::Elem: Clone;
}
```

### 5.10 StorageIntoRaw Trait

消耗式解构为裸指针，用于 `into_raw_parts()`（参见 `23-ffi.md §5.4`）。

```rust
/// Storage types that can be destructured into raw parts.
///
/// Only `Owned<A>` implements this trait (other storage modes cannot transfer ownership of raw memory).
pub unsafe trait StorageIntoRaw: StorageOwned {
    /// Consume the storage, returning a raw pointer.
    ///
    /// # Safety
    ///
/// The caller must preserve the allocator metadata required to reconstruct
/// the owned buffer. In Xenon's tensor-level FFI API, this metadata is carried
/// by `OwnedRawParts` (see `23-ffi.md §5.4`).
    unsafe fn into_raw(self) -> *mut Self::Elem;
}
```

### 5.11 Trait 实现矩阵

| 存储类型             | RawStorage | Storage | RawStorageMut | StorageMut | StorageOwned | StorageShared | StorageIntoOwned | StorageIntoRaw |
| -------------------- | :--------: | :-----: | :-----------: | :--------: | :----------: | :-----------: | :--------------: | :------------: |
| `Owned<A>`           |     ✅     |   ✅    |      ✅       |     ✅     |      ✅      |      ❌       |        ✅        |       ✅       |
| `ViewRepr<'a, A>`    |     ✅     |   ✅    |      ❌       |     ❌     |      ❌      |      ❌       |        ✅        |       ❌       |
| `ViewMutRepr<'a, A>` |     ✅     |   ✅    |      ✅       |     ✅     |      ❌      |      ❌       |        ✅        |       ❌       |
| `ArcRepr<A>`         |     ✅     |   ✅    |      ❌       |     ❌     |      ❌      |      ✅       |        ✅        |       ❌       |

### 5.11.1 存储模式转换矩阵

| From                 | To                   | 复杂度      | 说明                                     |
| -------------------- | -------------------- | ----------- | ---------------------------------------- |
| `Owned<A>`           | `Owned<A>`           | O(1) / O(n) | move 为 O(1)，`to_owned()` 深拷贝为 O(n) |
| `Owned<A>`           | `ViewRepr<'_, A>`    | O(1)        | 借用视图，不转移所有权                   |
| `Owned<A>`           | `ViewMutRepr<'_, A>` | O(1)        | 独占借用视图                             |
| `Owned<A>`           | `ArcRepr<A>`         | O(1)        | 零拷贝降级为共享只读引用                 |
| `ViewRepr<'_, A>`    | `Owned<A>`           | O(n)        | 复制逻辑元素到新的对齐缓冲               |
| `ViewMutRepr<'_, A>` | `ViewRepr<'_, A>`    | O(1)        | 零拷贝只读重借用，放弃写权限             |
| `ViewMutRepr<'_, A>` | `Owned<A>`           | O(n)        | 复制逻辑元素到新的对齐缓冲               |
| `ArcRepr<A>`         | `ViewRepr<'_, A>`    | O(1)        | 共享只读借用                             |
| `ArcRepr<A>`         | `Owned<A>`           | O(n)        | 总是分配新的独占缓冲并复制数据；不提供条件零拷贝 |

| 抽象源（具体表示） | `ReadOnlyRef` | `SharedReadOnlyRef` | `WritableRef` | `Owned` |
| ------------------ | ------------- | ------------------- | ------------- | ------- |
| `Owned` (`Owned<A>`) | `Owned<A> -> ViewRepr<'_, A>`：零拷贝 | `Owned<A> -> ArcRepr<A>`：零拷贝 | `Owned<A> -> ViewMutRepr<'_, A>`：零拷贝 | — |
| `WritableRef` (`ViewMutRepr<'_, A>`) | `ViewMutRepr<'_, A> -> ViewRepr<'_, A>`：零拷贝 | `ViewMutRepr<'_, A> -> ViewRepr<'_, A>`：零拷贝；此处 `ViewRepr` 充当基于借用的 shared-readonly 表示，满足 `require.md` §6.2 | — | `ViewMutRepr<'_, A> -> Owned<A>`：须分配 |
| `ReadOnlyRef` (`ViewRepr<'_, A>`) | — | `ViewRepr<'_, A> -> ArcRepr<A>`：不可转换 | `ViewRepr<'_, A> -> ViewMutRepr<'_, A>`：不可转换 | `ViewRepr<'_, A> -> Owned<A>`：须分配 |
| `SharedReadOnlyRef` (`ArcRepr<A>` / 降级后的 `ViewRepr<'_, A>`) | `ArcRepr<A> -> ViewRepr<'_, A>`：零拷贝；若来源已是降级后的 `ViewRepr`，则为同一借用语义下的只读重借用 | — | `ArcRepr<A> -> ViewMutRepr<'_, A>`：不可转换；降级后的 `ViewRepr` 同样不可恢复写权限 | `ArcRepr<A> -> Owned<A>`：须分配；降级后的 `ViewRepr<'_, A>` 亦须分配 |

> **补充说明：** 上表逐格对应 `require.md` §6.2 的抽象转换矩阵，并把每个抽象格子落实为具体表示路径：零拷贝表示仅重借用、共享只读降级或降级访问权限；须分配表示需要分配新的 owned 缓冲并复制数据；不可转换表示 Rust 类型系统下无法在不违反所有权/独占借用约束的前提下完成该转换，例如 `ViewRepr<'_, A> -> ArcRepr<A>` 与 `ArcRepr<A> -> ViewMutRepr<'_, A>`。

> **类型安全论证**：`ViewMutRepr<'a, A>` 的零拷贝降级路径是只读重借用 `ViewRepr<'a, A>`，其语义与 `require.md` §6.2 对“可写引用 → 共享只读引用 = 零拷贝”的要求一致：结果显式放弃写权限，且在该只读结果存续期间不得再并发写入同一底层数据。相对地，`ViewMutRepr<'a, A>` 不持有底层分配所有权，因此仍不能在零拷贝前提下构造 `ArcRepr<A>` 所需的共享所有权句柄。

> **约束：** Xenon 当前元素类型集合是封闭且按值语义处理的集合；`Owned::from_vec` 保持 `Elem: Copy` 约束，并统一复制到内部 64B 对齐缓冲（参见 `06-layout.md §5.6`）。其它从迭代器或构造器进入 `Owned` 的路径由上层构造模块统一收敛。

> **错误语义补充：** 上表中的复杂度只描述成功路径。凡转换违反 `require.md` §6.2 的可变性或独占性前提（例如试图把共享只读结果继续当作可写存储传播），已公开的运行时转换 API 须返回可恢复错误，而不是隐式降级为别的存储模式。

### 5.12 Good/Bad 对比

```rust
// Good - Use Storage trait bound to accept any readable storage
fn process<S: Storage<Elem = f64>>(storage: &S) {
    let slice = storage.as_slice();
    // ...
}

// Bad - Hardcoded Owned type, rejects views
fn process_bad(storage: &Owned<f64>) {
    let slice = storage.as_slice();
    // ...
}
```

```rust
// Good - ArcRepr explicitly triggers CoW before mutable access
fn modify_arc(arc: &mut ArcRepr<f64>) {
    let data = arc.make_mut();  // Explicit CoW
    data[0] = 10.0;
}

// Bad - Attempting to treat ArcRepr as shared-writable storage (compile error)
// arc.as_mut_ptr();  // ArcRepr does not implement StorageMut
```

---

## 6. 内部实现设计

### 6.1 Owned\<A\> 结构体

```rust
/// Owned storage.
///
/// `Owned<A>` has full ownership of the data, stored on the heap.
/// Internally wraps an alignment-aware buffer rather than a plain `Vec<A>`.
///
/// # Alignment Strategy
///
/// Xenon construction paths (`zeros`, `ones`, etc.) use
/// an internal `AlignedBuf<A>` wrapper that stores `(ptr, len, cap, align)` and
/// deallocates with the same layout it allocated with. This avoids the undefined
/// behavior risk of handing a 64-byte aligned allocation to a plain `Vec<A>`.
///
/// `from_vec` is a storage-layer constructor. Public tensor construction keeps
/// using `from_shape_vec` / `from_shape_slice` at the construct layer.
/// `from_vec` accepts a user-provided `Vec<A>` and copies its contents
/// into Xenon's internal aligned buffer representation.
///
/// Use `is_aligned()` at runtime to query whether the backing allocation
/// satisfies the 64-byte alignment requirement.
#[derive(Debug, Clone, PartialEq)]
pub struct Owned<A> {
    /// Internal data storage.
    ///
    /// When constructed by Xenon (zeros/ones), the underlying allocation
    /// is managed by `AlignedBuf<A>` and deallocated with the exact original layout.
    /// When constructed via `from_vec`, data is copied into a fresh aligned buffer.
    data: AlignedBuf<A>,
}

impl<A> Owned<A> {
    /// Default alignment: 64 bytes (AVX-512 cache line).
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// Creates Owned from a user-provided Vec.
    ///
    /// Consumes the input Vec and copies its contents into Xenon's internal
    /// alignment-aware buffer representation. This keeps `Owned<A>` consistent
    /// with the rest of the storage model, where owned construction paths are
    /// free to require fresh aligned storage.
    pub fn from_vec(data: Vec<A>) -> Self
    where
        A: Copy,
    {
        Self::from_vec_aligned(data)
    }

    /// Creates Owned by copying data into a fresh 64-byte aligned allocation.
    ///
    /// Backing implementation used by `from_vec` and Xenon construction paths
    /// (`from_shape_vec`, etc.) to guarantee SIMD-compatible alignment.
    pub fn from_vec_aligned(data: Vec<A>) -> Self
    where
        A: Copy,
    {
        let len = data.len();
        let elem_size = core::mem::size_of::<A>();
        if len == 0 || elem_size == 0 {
            return Self { data: AlignedBuf::empty() };
        }
        let size = len
            .checked_mul(elem_size)
            .expect("storage allocation size overflow");
        // Allocate aligned memory and copy elements
        // SAFETY: AlignedAlloc returns a valid, aligned allocation of the requested size.
        let ptr = AlignedAlloc::alloc(size, Self::DEFAULT_ALIGNMENT);
        let typed_ptr = ptr.as_ptr() as *mut A;
        // For Copy types (all Xenon element types), use bulk copy for efficiency.
        // SAFETY: typed_ptr is valid for `len` elements; data.as_ptr() is valid for `len` elements;
        // the two ranges are non-overlapping (typed_ptr is freshly allocated).
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), typed_ptr, len); }
        // SAFETY: ptr was allocated by AlignedAlloc with len elements.
        Self { data: unsafe { AlignedBuf::from_raw_parts(typed_ptr, len, len, Self::DEFAULT_ALIGNMENT) } }
    }

    /// Creates Owned with 64-byte aligned allocation.
    ///
    /// Used internally by Xenon construction methods (zeros, ones).
    ///
    /// # Safety
    ///
    /// `ptr` must have been allocated by `AlignedAlloc::alloc` with the
    /// specified capacity and alignment.
    pub(crate) unsafe fn from_aligned_raw_parts(
        ptr: *mut A,
        len: usize,
        capacity: usize,
    ) -> Self {
        // SAFETY: caller guarantees ptr was allocated by AlignedAlloc
        // with the given len and capacity.
        Self { data: AlignedBuf::from_raw_parts(ptr, len, capacity, Self::DEFAULT_ALIGNMENT) }
    }
}
```

### 6.2 当前默认 64 字节对齐分配器

```rust
/// Aligned memory allocator.
pub struct AlignedAlloc;

impl AlignedAlloc {
    /// Current default alignment: 64 bytes.
    ///
    /// This default is configurable and is not a hard requirement of the
    /// storage abstraction itself.
    pub const DEFAULT_ALIGNMENT: usize = 64;

    /// Allocates a memory block of the given size and alignment, without initialization.
    ///
    /// # Panics
    ///
    /// - `align` is not a power of two
    /// - `size` is 0
    /// - Memory allocation fails
    ///
    /// For zero-sized types (ZST, `size_of::<A>() == 0`), this allocator must not be called; use `NonNull::dangling()` directly and return a dangling pointer instead. Callers (such as `Owned::new`) are responsible for skipping allocation when size == 0.
    pub fn alloc(size: usize, align: usize) -> NonNull<u8>;

    /// Allocates and zero-initializes.
    pub fn alloc_zeroed(size: usize, align: usize) -> NonNull<u8>;

    /// Deallocates memory.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by `alloc` or `alloc_zeroed`
    /// - `size` and `align` must be the same as during allocation
    pub unsafe fn dealloc(ptr: NonNull<u8>, size: usize, align: usize);

}
```

**分配策略说明**：当前默认实现选择 64 字节对齐，以匹配 SIMD 友好的 owned 缓冲策略；这是一项实现选择，而不是 `require.md` 所要求的唯一对齐值。对齐值可配置。为保持文档与当前设计一致，`AlignedAlloc` 不提供“小数组回退到普通分配”的分支。除 ZST 与 `len == 0` 这两类显式跳过分配的情形外，当前默认实现的真实堆分配统一使用该默认对齐值。

**安全性论证**：`AlignedAlloc` 使用 `alloc::alloc::Layout` 确保对齐值是 2 的幂且总大小合法。分配失败时调用 `handle_alloc_error` 而非返回空指针，避免 UB。

### 6.3 ViewRepr\<'a, A\> 结构体

```rust
/// Immutable view storage.
#[derive(Debug)]
pub struct ViewRepr<'a, A> {
    ptr: *const A,
    len: usize,
    _marker: PhantomData<&'a A>,
}

/// Type alias.
pub type View<'a, A> = ViewRepr<'a, A>;

// Clone only copies the pointer and length, not the data (O(1)).
impl<'a, A> Clone for ViewRepr<'a, A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, len: self.len, _marker: PhantomData }
    }
}

impl<'a, A> Copy for ViewRepr<'a, A> {}
```

### 6.4 ViewMutRepr\<'a, A\> 结构体

```rust
/// Mutable view storage.
///
/// **Not cloneable**: exclusive semantics mean only one mutable reference can exist at a time.
#[derive(Debug)]
pub struct ViewMutRepr<'a, A> {
    ptr: *mut A,
    len: usize,
    _marker: PhantomData<&'a mut A>,
}

/// Type alias.
pub type ViewMut<'a, A> = ViewMutRepr<'a, A>;

// Intentionally no Clone impl — Rust borrowing rules require mutable references to be exclusive
```

### 6.5 ArcRepr\<A\> 结构体

```rust
/// Shared read-only storage backed by `Arc`.
///
/// Uses `Arc<AlignedBuf<A>>` to share aligned owned storage while preserving the
/// same backing-allocation invariants as `Owned<A>`. The public storage mode is
/// shared read-only; copy-on-write (CoW) via `make_mut()` is only an explicit
/// transition path to an exclusive owned buffer.
#[derive(Debug)]
pub struct ArcRepr<A> {
    inner: Arc<AlignedBuf<A>>,
}
```

> **设计决策：** `ArcRepr<A>` 不存储独立的 `offset` 字段。偏移量与切片范围由外层 `TensorBase` 元数据管理（见 `07-tensor.md §5.1`），而 `ArcRepr` 只表示共享只读引用语义下的底层连续缓冲区。`make_mut()` 仅针对整个底层缓冲区执行写时复制，用于在共享存储对象内部获得独占可写缓冲访问，不负责解释张量视图的 `offset` / `shape` / `strides`，也不把 `ArcRepr` 提升为共享可写存储模式，更不构成公开的 `ArcRepr<A> -> Owned<A>` 零拷贝转换。

**写时复制流程**：

```
make_mut() flow
┌──────────────────────┐
│      make_mut()      │
└──────────┬───────────┘
           │
           ▼
  ┌────────────────┐
  │ ref_count == 1 │
  └───────┬────────┘
          │
    ┌─────┴─────┐
    │ YES       │ NO
    ▼           ▼
return directly  allocate new buffer → copy data → drop one shared handle
   &mut [A]           (O(n) deep copy)
```

### 6.6 Marker Traits

```rust
/// Marker trait for storage types that own data.
pub unsafe trait IsOwned: RawStorage {}

/// Marker trait for storage types that are views (borrowed).
pub unsafe trait IsView: RawStorage {}

/// Marker trait for storage types that are mutable views.
pub unsafe trait IsViewMut: RawStorage {}

/// Marker trait for storage types that use Arc sharing.
pub unsafe trait IsArc: RawStorage {}
```

使用 Sealed trait 模式防止外部类型实现。

### 6.7 Send/Sync 实现规则

> **线程安全权威源**：存储类型的 `Send`/`Sync` 实现规则由 `25-safety.md` §4 定义。本文档仅做概要说明；若与 `25-safety.md` 存在冲突，以 `25-safety.md` 为准。

| 存储类型             |      Send      |      Sync      | 原因                                                                  |
| -------------------- | :------------: | :------------: | --------------------------------------------------------------------- |
| `Owned<A>`           |    A: Send     |    A: Sync     | 拥有数据，Send/Sync 条件与 `Vec<A>` 一致（分别要求 A:Send 和 A:Sync） |
| `ViewRepr<'a, A>`    |    A: Sync     |    A: Sync     | 共享借用需要 Sync 才能跨线程共享                                      |
| `ViewMutRepr<'a, A>` |    A: Send     |   ❌ 永远不    | 独占借用可转移但不可共享                                              |
| `ArcRepr<A>`         | A: Send + Sync | A: Send + Sync | Arc 内部原子操作保证线程安全                                          |

```rust
// SAFETY: ViewRepr<'a, A> only allows shared (read-only) access.
// It is safe to send between threads if A: Sync (shared refs are Send when T: Sync).
unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}
// SAFETY: Multiple threads can hold &ViewRepr simultaneously if A: Sync.
unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}

// SAFETY: ViewMutRepr<'a, A> allows exclusive access via &mut self.
// It is safe to send between threads if A: Send.
// Sync is NOT implemented: &mut T is not Sync (exclusive access must not be shared).
unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A> {}
```

---

## 7. 实现任务拆分

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
  - 内容: `AlignedAlloc` 结构体，`alloc`/`alloc_zeroed`/`dealloc`
  - 测试: `test_aligned_alloc_64`, `test_aligned_alloc_zeroed`
  - 前置: T1
  - 预计: 10 min

- [ ] **T7**: 实现 `Owned<A>` 结构体和构造方法
  - 文件: `src/storage/owned.rs`
  - 内容: `Owned<A>` 定义，`new`/`with_capacity`/`from_vec`/`from_vec_aligned`/`zeros`/`from_elem`
  - 测试: `test_owned_new`, `test_owned_from_vec`
  - 前置: T6
  - 预计: 10 min

- [ ] **T8**: 实现 `Owned<A>` 所有 trait 实现
  - 文件: `src/storage/owned.rs`
  - 内容: `RawStorage`/`Storage`/`StorageMut`/`StorageOwned` 实现，`Send`/`Sync`/`From`/`Default`
  - 测试: `test_owned_storage_trait`, `test_owned_send_sync`
  - 前置: T5, T7
  - 预计: 10 min

### Wave 3: 视图和 Arc 实现

- [ ] **T9**: 实现 `ViewRepr<'a, A>`
  - 文件: `src/storage/view.rs`
  - 内容: 结构定义、`from_raw_parts`/`from_slice`/`from_owned`/`slice`、Clone/Copy、`RawStorage`/`Storage` 实现
  - 测试: `test_view_clone_o1`, `test_view_lifetime`
  - 前置: T5
  - 预计: 10 min

- [ ] **T10**: 实现 `ViewMutRepr<'a, A>`
  - 文件: `src/storage/view_mut.rs`
  - 内容: 结构定义、`from_raw_parts_mut`/`from_mut_slice`/`from_owned`/`as_view`、不实现 Clone、`RawStorage`/`RawStorageMut`/`Storage`/`StorageMut` 实现
  - 测试: `test_view_mut_no_clone`, `test_view_mut_exclusive`
  - 前置: T5
  - 预计: 10 min

- [ ] **T11**: 实现 `ArcRepr<A>` 及 `make_mut()`
  - 文件: `src/storage/arc.rs`
  - 内容: 结构定义、`from_vec`/`from_owned`/`ref_count`/`is_unique`/`make_mut`/`into_owned`、`StorageShared` 实现
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
  - 文件: `tests/test_storage.rs`
  - 内容: 跨存储类型交互测试、ZST 测试、空数组测试
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

## 8. 测试计划

### 8.1 测试分类

| 类型     | 位置                     | 目的                |
| -------- | ------------------------ | ------------------- |
| 单元测试 | `#[cfg(test)] mod tests` | 验证单个函数/方法   |
| 集成测试 | `tests/`                 | 验证跨存储类型交互  |
| 边界测试 | 集成测试中标注           | ZST、空数组、大数组 |
| 属性测试 | `tests/property/`        | 随机生成验证不变量  |

### 8.2 单元测试清单

| 测试函数                          | 测试内容                                        | 优先级 |
| --------------------------------- | ----------------------------------------------- | ------ |
| `test_owned_new_empty`            | `Owned::new()` 创建空存储                       | 高     |
| `test_owned_zeros`                | `Owned::zeros(100)` 全零                        | 高     |
| `test_owned_from_vec`             | 从 Vec 构造并验证内容                           | 高     |
| `test_owned_alignment_from_zeros` | Xenon 构造路径（zeros）的 64 字节对齐验证       | 高     |
| `test_owned_alignment_from_vec`   | from_vec 路径复制到 64 字节对齐缓冲             | 高     |
| `test_owned_clone_deep`           | 克隆后修改不影响原数据                          | 高     |
| `test_view_from_slice`            | 从切片创建视图                                  | 高     |
| `test_view_clone_o1`              | 克隆仅复制元数据                                | 中     |
| `test_view_lifetime`              | 视图不能比源数据存活更久                        | 高     |
| `test_view_mut_exclusive`         | 可变视图独占语义                                | 高     |
| `test_view_mut_no_clone`          | 编译期验证不可克隆                              | 高     |
| `test_arc_ref_count`              | 引用计数正确                                    | 高     |
| `test_arc_cow`                    | 写时复制语义正确                                | 高     |
| `test_arc_make_mut_unique`        | 唯一时 make_mut 无复制                          | 中     |
| `test_aligned_alloc_64`           | 分配器 64 字节对齐                              | 高     |
| `test_zst_no_ub`                  | ZST 元素类型不引发 UB，且不会调用对齐分配器     | 高     |
| `test_empty_array`                | 空数组操作安全                                  | 高     |
| `test_arc_alignment_preserved`    | `ArcRepr` 保持与 `Owned` 相同的对齐语义         | 高     |
| `test_storage_into_owned_matrix`  | 各存储模式到 `Owned` 的转换复杂度与语义符合矩阵 | 中     |

### 8.3 边界测试场景

| 场景                                | 预期行为                              |
| ----------------------------------- | ------------------------------------- |
| 空数组 `Owned::new()`               | `len() == 0`, `is_empty() == true`    |
| 单元素 `Owned::from_vec(vec![1.0])` | `len() == 1`, `get(0) == Some(&1.0)`  |
| ZST 元素 `Owned::<()>::zeros(1000)` | 不分配内存，不引发 UB                 |
| 大数组 16M 元素                     | 正常分配和访问                        |
| 容量乘法接近 `usize::MAX`           | 显式 panic 或返回错误，不发生整数回绕 |

### 8.4 属性测试不变量

| 不变量                                            | 测试方法           |
| ------------------------------------------------- | ------------------ |
| `owned.to_owned().as_slice() == owned.as_slice()` | 随机元素类型和大小 |
| `view.clone().as_ptr() == view.as_ptr()`          | 随机切片范围       |
| `arc.make_mut()` 后引用计数为 1                   | 随机共享数量       |
| `Owned::from_vec_aligned` 在 ZST 上不调用分配器   | 随机 ZST 长度      |

### 8.5 Feature gate / 配置测试

| 配置项 | 覆盖方式                             | 说明                                         |
| ------ | ------------------------------------ | -------------------------------------------- |
| 默认配置 | 常规单元/集成测试路径                 | 本模块无独立 feature gate，默认配置即主路径  |
| 非默认 feature | 不适用                             | 本模块未定义 feature gate，故无额外配置矩阵 |

### 8.6 类型边界 / 编译期测试

| 测试类型 | 覆盖方式                                         | 说明                                                    |
| -------- | ------------------------------------------------ | ------------------------------------------------------- |
| 存储可变性边界 | compile-fail 测试 `ViewRepr` 不可写、`ViewMutRepr` 不可克隆 | 验证借用与独占语义在类型系统层成立                      |
| trait 层次边界 | 编译期验证 `ArcRepr` 不实现 `StorageMut`           | 验证共享存储不会被误用为共享可写模式                    |
| ZST 边界 | 编译期与单元测试结合验证 ZST 路径不要求真实分配      | 验证零大小类型不会打破 trait 契约                       |

---

## 9. 模块交互设计

### 9.0 接口约定

| 方向 | 对方模块 | 接口/类型 | 约定 |
| ---- | -------- | --------- | ---- |
| 消费（输入） | `layout` | `Strides` 及其辅助接口 | 仅引用 `06-layout.md` 已定义的 `from_slice()` / `iter()` / `as_slice()` 等能力；storage 不重复定义 layout API |
| 产出（输出） | `tensor` | `Storage` / `StorageMut` / `StorageOwned` / `StorageShared`，以及 `Owned<A>` / `ViewRepr<'a, A>` / `ViewMutRepr<'a, A>` / `ArcRepr<A>` | `TensorBase<S, D>` 通过 `S: Storage` 或 `S: StorageMut` 消费底层存储；`TensorBase` 负责解释 `offset` / `shape` / `strides` |
| 产出（输出） | `parallel` | `S: Storage + Sync` 或 `S: StorageMut + Send` | 并行路径只能在满足 Send/Sync 前提时消费 storage；不得突破共享只读 / 独占可写边界 |
| 产出（输出） | `iter` / `ffi` | `as_ptr()` / `as_slice()` / `into_raw()` | 上层仅消费 storage-visible backing range；不得把 storage API 误解为逻辑张量切片 |

### 9.1 与 Tensor 模块

`TensorBase<S, D>` 的 `S` 参数约束为 `Storage` 或 `StorageMut`，通过关联类型 `Elem` 获取元素类型（参见 `07-tensor.md` §5）：

```rust
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    pub fn as_ptr(&self) -> *const A {
        // storage.as_ptr() is the storage base pointer. TensorBase is responsible
        // for applying offset and exposing the logical-first pointer to upper layers.
        unsafe { self.storage.as_ptr().add(self.offset) }
    }
}
```

### 9.2 与 Layout 模块

Storage 提供对齐信息（`is_aligned()`），Layout 模块查询对齐状态更新 `LayoutFlags::ALIGNED`（参见 `06-layout.md` §5）。

### 9.3 与 Parallel 模块

并行迭代要求 `S: Storage + Sync`（读）或 `S: StorageMut + Send`（写），由 storage 的 Send/Sync 实现保证（参见 `09-parallel.md` §4）。

### 9.4 数据流描述

```
User calls `TensorBase::as_ptr()`
    │
    ├── tensor/ reads `storage: S` and `offset`
    │
    ├── `storage.as_ptr()` returns the storage base pointer
    │
    ├── tensor/ applies `.add(offset)` to obtain the logical-first pointer
    │
    └── upper layers (`iter` / `ffi` / `parallel`) continue consuming that pointer or slice view
```

---

## 10. 错误处理与语义边界

| 项目           | 内容 |
| -------------- | ---- |
| Recoverable error | `try_reserve()`、显式运行时转换与 raw-parts 元数据校验失败返回 `XenonError`；上下文字段应包含操作名、存储模式、长度/容量或布局信息 |
| Panic | 对齐分配器在 `align` 非 2 的幂、`size == 0` 或分配失败时 panic；已验证快捷路径中的整数溢出也可 panic |
| 路径一致性 | scalar：不适用；SIMD 对齐与非对齐路径必须返回一致元素值；parallel 存储访问须与串行路径保持一致 |
| 容差边界 | 不适用 |

### 10.1 ZST 和空数组处理

| 场景           | 预期行为                                             | 安全性论证                                                            |
| -------------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| ZST 元素类型   | 使用非空悬挂哨兵指针，不调用分配器，len 正常计算     | ZST 不需要真实 backing storage，且禁止把 `size=0` 传给 `AlignedAlloc` |
| 空数组 `len=0` | `as_ptr()` 返回非空悬垂指针，`as_slice()` 返回空切片 | `Vec` 保证空时 `as_ptr()` 非空                                        |
| ZST + 空数组   | 不引发分配，不引发 UB                                | ZST 不需要实际内存                                                    |

---

## 11. 设计决策记录

### 决策 1：64 字节默认对齐

| 属性     | 值                                                                                        |
| -------- | ----------------------------------------------------------------------------------------- |
| 决策     | 当前默认实现使用 64 字节作为默认对齐；该默认值可配置，且不构成需求层硬约束                 |
| 理由     | AVX-512 缓存行大小；现代 CPU 缓存行通常 64 字节；满足 SSE/AVX/AVX2/AVX-512 所有 SIMD 指令 |
| 替代方案 | 16 字节 — 放弃，AVX-512 未对齐                                                            |
| 替代方案 | 32 字节 — 放弃，AVX-512 未对齐                                                            |

### 决策 2：ArcRepr 不实现 StorageMut（CoW 策略）

| 属性     | 值                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------- |
| 决策     | `ArcRepr` 通过 `StorageShared::make_mut()` 触发 CoW 并转入独占 owned 缓冲，不实现 `StorageMut`；`make_mut()` 仅是内部 CoW 机制，不构成共享可写模式 |
| 理由     | 可变访问涉及潜在 O(n) 复制，显式调用让用户知晓性能影响；返回的 `&mut` 是复制后或唯一所有权下的独占访问，并保持“当前版本不提供共享可写存储模式”的需求边界 |
| 替代方案 | 实现 `StorageMut` — 放弃，隐藏 CoW 成本                                                                  |

### 决策 3：ArcRepr 作为统一 trait 体系的一部分

| 属性     | 值                                                                     |
| -------- | ---------------------------------------------------------------------- |
| 决策     | `ArcRepr<A>` 纳入统一 trait 体系（通过 `StorageShared`），而非独立类型 |
| 理由     | 更好的正交性，泛型代码可通过 `Storage` trait 统一处理所有存储模式      |
| 替代方案 | ndarray 风格 `ArcArray` 独立类型 — 放弃，增加类型复杂度                |

---

## 12. 性能描述

| 方面         | 设计决策                                                            |
| ------------ | ------------------------------------------------------------------- |
| 对齐分配     | `Owned` 与 `ArcRepr` 统一保持 64 字节对齐语义；ZST 和空数组跳过分配 |
| 视图克隆     | O(1)，仅复制指针和长度                                              |
| Arc 克隆     | O(1)，仅增加引用计数                                                |
| Arc make_mut | 唯一时 O(1)，非唯一时 O(n) 深拷贝                                   |
| Owned 克隆   | O(n) 深拷贝                                                         |
| 内联         | 所有 `as_ptr`/`len`/`get` 标注 `#[inline]`                          |
| 单态化       | Storage trait 在泛型上下文中单态化，无虚调用开销                    |

**性能数据（参考）**:

| 操作                                 | 开销 | 说明               |
| ------------------------------------ | ---- | ------------------ |
| `Owned::zeros(1M)`                   | ~1ms | 包含分配和零初始化 |
| `View::clone()`                      | ~2ns | 仅复制 3 个字段    |
| `Arc::clone()`                       | ~5ns | 原子引用计数增加   |
| `Arc::make_mut()`（唯一）            | ~2ns | 直接返回可变引用   |
| `Arc::make_mut()`（非唯一，1M 元素） | ~1ms | 深拷贝             |

---

## 13. 平台与工程约束

| 约束       | 说明                                   |
| ---------- | -------------------------------------- |
| `std` only | 本模块依赖 `std` 环境，不讨论 `no_std` |
| 单 crate   | 保持单 crate 边界                      |
| SemVer     | 存储类型和 trait 变更遵循 SemVer       |
| 最小依赖   | 无新增第三方依赖                       |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-07 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.1.0 | 2026-04-08 |
| 1.2.0 | 2026-04-08 |
| 1.2.1 | 2026-04-10 |
| 1.2.2 | 2026-04-14 |
| 1.2.3 | 2026-04-14 |
| 1.2.4 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
