# 05-11 线程安全模块设计

> **模块路径**: `src/thread_safety/` (内部模块，散布于各存储实现中)
> **版本**: v1.0
> **日期**: 2026-03-28
> **前置文档**: 03-04-storage-system.md, 04-02-parallel-backend.md
> **需求来源**: require-v18.md 第 8 节

---

## 1. 模块概述

### 1.1 线程安全在科学计算库中的重要性

Senon 作为科学计算多维数组库，需要在多线程环境下安全运行。线程安全设计直接影响以下关键场景：

| 场景 | 线程安全需求 |
|------|--------------|
| **并行迭代** | 多线程同时访问不同元素区间 |
| **跨线程共享** | 通过 ArcRepr 在线程间传递只读数据 |
| **写时复制** | ArcRepr::make_mut() 的原子性保证 |
| **数据竞争预防** | 确保 ViewMutRepr 独占访问不被共享 |
| **与 rayon 集成** | ParallelIterator 要求 Send 约束 |

### 1.2 Rust 线程安全机制回顾

Rust 通过 `Send` 和 `Sync` trait 在类型系统层面保证线程安全：

| Trait | 含义 | 编译器自动实现条件 |
|-------|------|-------------------|
| `Send` | 类型可以安全地在线程间移动所有权 | 所有字段都实现 `Send` |
| `Sync` | 类型可以安全地在线程间共享不可变引用 | `&T` 实现 `Send` |

**关键规则**：
- `&T` 是 `Send` ⟺ `T: Sync`
- `&mut T` 是 `Send` ⟺ `T: Send`
- 裸指针 `*const T` 和 `*mut T` 既不是 `Send` 也不是 `Sync`

### 1.3 Senon 的线程安全挑战

Senon 的四种存储模式带来不同的线程安全考量：

```
存储模式           挑战
────────────────────────────────────────────────────────────
Owned<A>          • 内部包含 Vec<A>，Vec 本身是 Send+Sync
                  • 需确保元素类型 A 的约束正确传播

ViewRepr<&'a A>   • 包含裸指针 *const A
                  • 需通过 unsafe impl 正确表达线程安全

ViewMutRepr<&'a mut A>
                  • 包含裸指针 *mut A
                  • 独占语义决定其不是 Sync
                  • 但可以转移所有权（Send）

ArcRepr<A>        • 内部使用 Arc<Vec<A>>
                  • Arc 的原子计数保证线程安全
                  • make_mut() 需要原子性 CoW
```

---

## 2. Send/Sync 实现矩阵

### 2.1 完整条件矩阵

| 存储模式 | Send | Sync | Send 条件 | Sync 条件 | 理由 |
|----------|:----:|:----:|-----------|-----------|------|
| `Owned<A>` | ✅ | ✅ | `A: Send` | `A: Sync` | Vec 内部线程安全，元素约束传播 |
| `ViewRepr<&'a A>` | ✅ | ✅ | `A: Sync` | `A: Sync` | 共享引用跨线程共享要求 T: Sync |
| `ViewMutRepr<&'a mut A>` | ✅ | ❌ | `A: Send` | N/A | 独占引用不可共享，但可转移 |
| `ArcRepr<A>` | ✅ | ✅ | `A: Send + Sync` | `A: Send + Sync` | Arc 原子计数，要求元素线程安全 |

### 2.2 条件推导逻辑

#### 2.2.1 Owned<A> 的 Send/Sync

```rust
// Owned<A> 内部结构
pub struct Owned<A> {
    data: Vec<A>,
    alignment: usize,
}

// 推导：
// 1. Vec<A> 是 Send + Sync，当 A: Send + Sync
// 2. usize 是 Send + Sync
// 3. 因此 Owned<A> 应该是 Send + Sync，当 A: Send + Sync
```

**自动实现 vs 手动实现**：
- Rust 编译器会自动为全 `Send`/`Sync` 字段的结构体实现 trait
- 但 `Owned<A>` 可能有其他 `unsafe` 代码依赖，建议显式实现以文档化意图

#### 2.2.2 ViewRepr<&'a A> 的 Send/Sync

```rust
// ViewRepr 内部结构
pub struct ViewRepr<A> {
    ptr: *const A,      // 裸指针：不是 Send，不是 Sync
    len: usize,
    _marker: PhantomData<A>,
}
```

**关键洞察**：
- `&'a A` 跨线程共享 ⟺ `A: Sync`
- `&'a A` 跨线程移动 ⟺ `A: Sync`（因为移动的是引用，不是数据）
- `ViewRepr<&'a A>` 语义上等同于 `&'a [A]`
- `&'a [A]` 是 `Send + Sync` 当 `A: Sync`

#### 2.2.3 ViewMutRepr<&'a mut A> 的 Send/Sync

```rust
// ViewMutRepr 内部结构
pub struct ViewMutRepr<A> {
    ptr: *mut A,        // 裸指针：不是 Send，不是 Sync
    len: usize,
    _marker: PhantomData<A>,
}
```

**关键洞察**：
- `&'a mut A` 可以跨线程移动（转移独占访问权）⟺ `A: Send`
- `&'a mut A` 不能跨线程共享（Rust 借用规则禁止）
- 因此 `ViewMutRepr` 是 `Send` 但不是 `Sync`

#### 2.2.4 ArcRepr<A> 的 Send/Sync

```rust
// ArcRepr 内部结构
pub struct ArcRepr<A> {
    inner: Arc<Vec<A>>, // Arc 的 Send/Sync 取决于 T
    len: usize,
    offset: usize,
}
```

**Arc 的线程安全**：
- `Arc<T>: Send` ⟺ `T: Send + Sync`
- `Arc<T>: Sync` ⟺ `T: Send + Sync`
- 原因：Arc 允许多个所有者共享 `&T`，因此 `T` 必须线程安全

### 2.3 与 ndarray 的对比

| 方面 | ndarray | Senon | 差异原因 |
|------|---------|-------|----------|
| Owned | 自动推导 | 显式 unsafe impl | 文档化意图 |
| View | `ArrayView: Send+Sync` 当 `A: Sync` | 相同 | 语义一致 |
| ViewMut | `ArrayViewMut: Send` 当 `A: Send` | 相同 | 语义一致 |
| ArcArray | `ArcArray: Send+Sync` 当 `A: Send+Sync` | 相同 | 语义一致 |

---

## 3. unsafe impl Send/Sync 设计

### 3.1 何时需要手动实现

| 情况 | 是否需要 unsafe impl | 示例 |
|------|---------------------|------|
| 结构体包含裸指针 | **需要** | `ViewRepr`, `ViewMutRepr` |
| 结构体全是 Send+Sync 字段 | 不需要 | `Owned<A>` (Vec + usize) |
| 包含 `PhantomData` | 可能需要 | 如果 `PhantomData` 包含非 Send 类型 |
| 包装标准库类型 | 不需要 | `ArcRepr` 包装 `Arc` |

### 3.2 ViewRepr 的 unsafe impl

```rust
// src/storage/view.rs

/// # Safety
///
/// `ViewRepr<&'a A>` 实现了 `Send`，因为：
///
/// 1. **所有权转移语义**：移动 `ViewRepr` 不会移动底层数据，
///    只是将视图的元数据（指针 + 长度）转移到新线程。
///
/// 2. **共享引用约束**：`&'a A` 可以跨线程移动当且仅当 `A: Sync`。
///    这确保了即使多个线程持有 `&A`，它们也只能进行只读访问。
///
/// 3. **生命周期保证**：`'a` 生命周期确保视图不会比源数据存活更久，
///    防止悬垂指针。
///
/// 4. **无内部可变性**：视图本身不拥有数据，也没有内部可变性，
///    跨线程移动不会导致数据竞争。
///
/// **反例：如果 `A` 不是 `Sync`**
///
/// 假设 `A = Cell<i32>`（不是 `Sync`），两个线程可以同时持有
/// `&Cell<i32>` 并通过 `Cell::set` 修改数据，导致数据竞争。
/// 因此要求 `A: Sync` 是必要的。
unsafe impl<'a, A: Sync> Send for ViewRepr<&'a A> {}

/// # Safety
///
/// `ViewRepr<&'a A>` 实现了 `Sync`，因为：
///
/// 1. **共享访问安全**：多个线程可以同时持有 `&ViewRepr<&'a A>`，
///    这等价于多个线程持有 `&&A`（指向共享引用的共享引用）。
///
/// 2. **嵌套引用约束**：`&&T` 是 `Send`（因此 `&T` 是 `Sync`）
///    当且仅当 `T: Sync`。由于 `&A` 本身是 `Send` 当 `A: Sync`，
///    因此 `&ViewRepr<&'a A>` 是 `Send` 当 `A: Sync`。
///
/// 3. **只读访问**：通过共享引用访问视图只能进行只读操作，
///    不会修改视图本身或底层数据。
///
/// 4. **无状态**：视图的 `ptr` 和 `len` 在创建后不可变，
///    多线程读取这些字段是安全的。
unsafe impl<'a, A: Sync> Sync for ViewRepr<&'a A> {}
```

### 3.3 ViewMutRepr 的 unsafe impl

```rust
// src/storage/view_mut.rs

/// # Safety
///
/// `ViewMutRepr<&'a mut A>` 实现了 `Send`，因为：
///
/// 1. **独占所有权转移**：移动 `ViewMutRepr` 会转移独占访问权到新线程。
///    Rust 借用检查器保证原线程不再持有任何引用。
///
/// 2. **无别名保证**：由于 `&mut A` 是独占引用，在任意时刻只有一个
///    `ViewMutRepr` 可以访问数据。跨线程移动后，新线程成为唯一访问者。
///
/// 3. **元素类型约束**：`&mut T` 是 `Send` 当且仅当 `T: Send`。
///    这确保了元素可以安全地在新线程中被读写。
///
/// 4. **生命周期不变性**：`'a` 生命周期确保视图不会比源数据存活更久。
///
/// **反例：如果 `A` 不是 `Send`**
///
/// 假设 `A = Rc<i32>`（不是 `Send`），将 `ViewMutRepr<&mut Rc<i32>>`
/// 移动到另一个线程会导致两个线程可能同时访问同一个 `Rc`，
/// 而 `Rc` 的引用计数不是原子的，会导致数据竞争。
///
/// **注意：ViewMutRepr 不实现 Clone**
///
/// `ViewMutRepr` 刻意不实现 `Clone`，因为复制会产生别名，违反独占语义。
/// 这是 `ViewMutRepr` 线程安全的关键保障。
unsafe impl<'a, A: Send> Send for ViewMutRepr<&'a mut A> {}

// ViewMutRepr 不实现 Sync
// 原因：&mut T 不能共享，Rust 借用规则禁止
//
// 如果 ViewMutRepr 是 Sync，那么 &ViewMutRepr 可以跨线程共享，
// 这意味着多个线程可以同时调用 view_mut.as_mut_slice() 获取
// &mut [A]，导致别名和潜在的数据竞争。
//
// Rust 的负面 trait impl（!Sync）在 stable Rust 中不能显式写，
// 但由于 ViewMutRepr 包含 *mut A（不是 Sync），编译器不会
// 自动实现 Sync，这正是我们想要的行为。
```

### 3.4 Owned 的显式实现（可选但推荐）

```rust
// src/storage/owned.rs

/// # Safety
///
/// `Owned<A>` 实现 `Send`，因为：
///
/// 1. **独占所有权**：`Owned` 拥有数据的完全所有权，
///    转移 `Owned` 会转移所有数据到新线程。
///
/// 2. **元素类型约束**：`A: Send` 确保元素可以安全地跨线程移动。
///
/// 3. **Vec 安全性**：`Vec<A>` 本身是 `Send` 当 `A: Send`，
///    我们只是将此属性显式声明。
///
/// 4. **无共享状态**：转移后，原线程不再持有任何引用。
unsafe impl<A: Send> Send for Owned<A> {}

/// # Safety
///
/// `Owned<A>` 实现 `Sync`，因为：
///
/// 1. **共享只读访问**：多个线程可以同时持有 `&Owned<A>`，
///    通过它只能进行只读操作（如 `get()`）。
///
/// 2. **元素类型约束**：`A: Sync` 确保元素可以安全地跨线程共享引用。
///
/// 3. **内部不可变性**：通过 `&Owned` 不能修改内部数据
///    （需要 `&mut Owned` 才能调用 `get_mut()`）。
///
/// 4. **Vec 安全性**：`Vec<A>` 本身是 `Sync` 当 `A: Sync`。
unsafe impl<A: Sync> Sync for Owned<A> {}
```

### 3.5 ArcRepr 的实现

```rust
// src/storage/arc.rs

/// # Safety
///
/// `ArcRepr<A>` 实现 `Send`，因为：
///
/// 1. **Arc 原子性**：`Arc<Vec<A>>` 使用原子引用计数，
///    多线程增加/减少引用计数是安全的。
///
/// 2. **元素约束**：`A: Send + Sync` 确保：
///    - `Send`：数据可以在线程间移动（当 Arc 唯一时）
///    - `Sync`：多个线程可以同时持有 `&A`
///
/// 3. **只读共享**：多个 `ArcRepr` 共享同一数据时，只能读取。
///    写入需要通过 `make_mut()` 获取独占访问。
///
/// 4. **make_mut 原子性**：`Arc::make_mut` 保证：
///    - 原子地检查引用计数
///    - 如果 >1，复制数据并原子递减原引用计数
///    - 无数据竞争
///
/// **反例：如果 `A` 不是 `Send + Sync`**
///
/// 假设 `A = Cell<i32>`（不是 `Sync`），多个线程可以同时通过
/// 不同的 `ArcRepr` 访问同一个 `Cell`，导致数据竞争。
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}

/// # Safety
///
/// `ArcRepr<A>` 实现 `Sync`，因为：
///
/// 1. **共享引用安全**：多个线程可以同时持有 `&ArcRepr<A>`，
///    这允许它们读取数据（通过 `get()`）。
///
/// 2. **Arc 同步保证**：`Arc<Vec<A>>` 是 `Sync` 当 `A: Send + Sync`，
///    因为内部数据可以被多个线程安全地共享引用。
///
/// 3. **无内部可变性**：通过 `&ArcRepr` 不能修改数据。
///    `make_mut(&mut self)` 需要 `&mut ArcRepr`，借用检查器
///    保证同一时刻只有一个可变引用。
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

### 3.6 Safety 证明模板

对于每个 `unsafe impl Send/Sync`，必须包含以下证明要素：

```
1. **语义正确性**：解释为什么该类型的语义允许 Send/Sync
2. **约束必要性**：解释每个 trait bound 为什么是必需的
3. **反例分析**：展示如果约束不满足会发生什么问题
4. **内部状态**：分析内部字段的可变性
5. **生命周期**：如有生命周期参数，解释其作用
```

---

## 4. 并行迭代安全

### 4.1 访问隔离保证

并行迭代的核心安全原则是：**各线程访问不重叠的元素区间**。

```
┌─────────────────────────────────────────────────────────────────┐
│                    并行迭代访问隔离示意                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  数组: [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]       │
│        └────线程0───┘└────线程1───┘└────线程2───┘└──线程3──┘     │
│                                                                 │
│  关键保证：                                                      │
│  • 每个元素最多被一个线程访问                                    │
│  • 线程间无共享写入                                              │
│  • 分块边界清晰，无重叠                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 连续数组的分块

```rust
// src/parallel/par_iter.rs

/// 安全分块保证
///
/// # Safety Invariant
///
/// 对于连续数组（F-contiguous 或 C-contiguous），分块满足：
///
/// 1. **完整性**：所有块的并集覆盖 [0, len)
/// 2. **不重叠**：任意两个块的交集为空
/// 3. **有序性**：块的顺序与内存布局一致
/// 4. **边界对齐**：块边界是元素边界，不会跨越
///
/// # 伪代码
///
/// ```
/// fn safe_chunks(len: usize, num_threads: usize) -> Vec<Range<usize>> {
///     let chunk_size = (len + num_threads - 1) / num_threads;
///     (0..num_threads)
///         .map(|i| {
///             let start = i * chunk_size;
///             let end = min(start + chunk_size, len);
///             start..end
///         })
///         .filter(|r| !r.is_empty())
///         .collect()
/// }
/// ```
///
/// # 证明
///
/// 1. 完整性：
///    - 最后一块的 end = len
///    - 所有块的并集 = [0, chunk_size) ∪ [chunk_size, 2*chunk_size) ∪ ... = [0, len)
///
/// 2. 不重叠：
///    - 块 i 范围是 [i*chunk_size, min((i+1)*chunk_size, len))
///    - 块 i+1 范围是 [(i+1)*chunk_size, ...)
///    - i*chunk_size <= (i+1)*chunk_size，无重叠
///
/// 3. 边界对齐：
///    - 块大小是元素数的整数倍
///    - 连续数组中，索引 i 对应元素 i
///    - 无跨元素边界问题
```

### 4.3 非连续数组的步长处理

非连续数组（如转置视图、切片视图）的并行分块更复杂：

```rust
// src/parallel/par_iter.rs

/// 非连续数组的并行安全
///
/// # 挑战
///
/// 非连续数组的元素在内存中不连续存储，访问需要按步长跳跃：
///
/// ```
/// 数组: shape=[3,4], strides=[4,1] (C-order, 连续)
/// 转置: shape=[4,3], strides=[1,4] (F-order, 非连续)
///
/// 内存布局:
/// 原始: [e00, e01, e02, e03, e10, e11, e12, e13, e20, e21, e22, e23]
/// 转置访问顺序: e00, e10, e20, e01, e11, e21, e02, e12, e22, e03, e13, e23
/// ```
///
/// # 安全分块策略
///
/// 1. **沿最外层轴分块**：每个块包含若干完整的"行"（沿第一轴的切片）
/// 2. **块内步长处理**：块内遍历仍需正确计算偏移
/// 3. **最小块大小**：确保每块有足够工作量，抵消步长计算开销
///
/// # Safety Invariant
///
/// 对于非连续数组，分块满足：
///
/// 1. **逻辑完整性**：所有块的逻辑索引并集覆盖所有元素
/// 2. **物理隔离**：不同线程访问的物理内存区间不重叠
///    （注意：由于步长跳跃，物理区间可能不连续）
/// 3. **边界安全**：任何合法索引计算的偏移不越界
///
/// # 伪代码
///
/// ```
/// fn safe_strided_chunks(
///     shape: &[isize],
///     strides: &[isize],
///     min_chunk_elements: usize,
/// ) -> Vec<Range<usize>> {
///     // 沿第一轴分块
///     let first_axis_len = shape[0];
///     let elements_per_slice: usize = shape[1..].iter().product();
///
///     if elements_per_slice == 0 {
///         return vec![0..first_axis_len];
///     }
///
///     let slices_per_chunk = max(1, min_chunk_elements / elements_per_slice);
///
///     (0..first_axis_len)
///         .step_by(slices_per_chunk)
///         .map(|start| {
///             let end = min(start + slices_per_chunk, first_axis_len);
///             start..end
///         })
///         .collect()
/// }
/// ```
///
/// # 证明要点
///
/// 1. **物理隔离**：
///    - 每个线程处理不同的第一轴索引范围 [start, end)
///    - 索引 (i, j, k, ...) 其中 i ∈ [start, end)
///    - 不同块的 i 值不重叠，因此物理偏移不重叠
///
/// 2. **边界安全**：
///    - 所有索引 i ∈ [0, shape[0])
///    - 偏移计算: offset = i*strides[0] + j*strides[1] + ...
///    - 需确保: 0 <= offset < total_elements（由数组构造保证）
pub fn compute_strided_chunks(/* ... */) { /* ... */ }
```

### 4.4 可变并行迭代的独占访问

```rust
// src/parallel/par_iter.rs

/// 可变并行迭代的安全保证
///
/// # 核心原则
///
/// 可变并行迭代要求 **编译期 + 运行期双重保证**：
///
/// 1. **编译期**：`par_iter_mut` 需要 `&mut self`，借用检查器保证无别名
/// 2. **运行期**：分块算法保证各线程访问不重叠区间
///
/// # 类型签名
///
/// ```ignore
/// fn par_iter_mut(&mut self) -> ParElementsMut<'_, A, D>
/// where
///     A: Element + Send + Sync,  // 元素必须线程安全
/// ```
///
/// # Safety Invariant
///
/// `ParElementsMut` 保证：
///
/// 1. **独占入口**：创建需要 `&mut self`，确保同一时刻只有一个可变迭代器
/// 2. **分块隔离**：内部使用与只读迭代相同的分块算法
/// 3. **元素级独占**：每个元素被恰好一个线程的 `&mut A` 引用
///
/// # 伪代码证明
///
/// ```
/// // 假设 len = 12, num_threads = 4
/// // 分块: [0..3], [3..6], [6..9], [9..12]
///
/// // 线程 0 持有 &mut data[0..3]
/// // 线程 1 持有 &mut data[3..6]
/// // 线程 2 持有 &mut data[6..9]
/// // 线程 3 持有 &mut data[9..12]
///
/// // Rust 借用规则：
/// // - 每个切片是 &mut [A]，独占访问其范围
/// // - 不同切片范围不重叠，因此不违反借用规则
/// ```
///
/// # 与 rayon 的集成
///
/// rayon 的 `par_iter_mut` 使用相同的分块策略：
///
/// ```ignore
/// data.par_chunks_mut(chunk_size)
///     .for_each(|chunk| {
///         // chunk: &mut [A]
///         // 每个线程独占访问其 chunk
///     });
/// ```
pub struct ParElementsMut<'a, A, D> {
    // 内部实现...
}
```

### 4.5 非连续可变迭代的限制

```rust
// src/parallel/par_iter.rs

/// 非连续可变迭代的安全限制
///
/// # 问题
///
/// 对于非连续数组，`par_iter_mut` 面临额外挑战：
///
/// 1. **不能直接切片**：非连续数组不能转为 `&mut [A]`
/// 2. **需要逐元素访问**：每个元素需要单独计算偏移
/// 3. **性能考量**：非连续访问的开销可能抵消并行收益
///
/// # 策略：回退到串行或限制并行
///
/// ```ignore
/// fn par_iter_mut(&mut self) -> ParElementsMut<'_, A, D> {
///     if self.is_contiguous() {
///         // 安全：可以转为切片，直接分块
///         ParElementsMut::Contiguous(/* ... */)
///     } else {
///         // 非连续：选项 1 - 回退串行
///         ParElementsMut::Sequential(self.iter_mut())
///
///         // 非连续：选项 2 - 限制并行（沿轴分块）
///         ParElementsMut::Strided(/* ... */)
///     }
/// }
/// ```
///
/// # Senon 的选择
///
/// 对于非连续可变迭代，Senon 采用 **沿轴分块** 策略：
///
/// - 如果可以沿某轴分割为不重叠的子数组，使用并行
/// - 否则回退到串行
///
/// 这确保了安全性，同时在常见场景（如沿 batch 轴并行）保持性能。
```

---

## 5. Padding 字节规则

### 5.1 Padding 的来源和目的

Senon 支持在主维度添加 padding 以保证 SIMD 对齐：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Padding 内存布局示意                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  逻辑数组: shape = [3, 4], 12 个元素                             │
│  SIMD 对齐要求: 每行 8 的倍数                                    │
│  分配: shape = [3, 8], 24 个元素                                 │
│                                                                 │
│  内存布局:                                                       │
│  [e00] [e01] [e02] [e03] [pad] [pad] [pad] [pad]                │
│  [e10] [e11] [e12] [e13] [pad] [pad] [pad] [pad]                │
│  [e20] [e21] [e22] [e23] [pad] [pad] [pad] [pad]                │
│                                                                 │
│  逻辑元素: 0-11                                                  │
│  Padding 元素: 12-23                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Padding 并发规则

| 规则 | 要求 | 实现方式 |
|------|------|----------|
| **禁止访问** | 任何线程不得读写 padding 字节（初始化除外） | 迭代器只遍历逻辑元素 |
| **禁止分配** | 并行分块不得将 padding 字节分配给任何线程 | 分块基于逻辑长度 |
| **禁止暴露** | 视图切片不得暴露 padding 字节为可访问元素 | 切片边界检查 |

### 5.3 具体实施方案

```rust
// src/storage/owned.rs

impl<A> Owned<A> {
    /// 逻辑元素数量
    pub fn len(&self) -> usize {
        self.data.len()  // Vec 的 len 是逻辑长度
    }

    /// 分配的元素数量（含 padding）
    pub fn allocated_len(&self) -> usize {
        self.data.capacity()  // capacity 是实际分配大小
    }

    /// Padding 元素数量
    pub fn padding_len(&self) -> usize {
        self.allocated_len().saturating_sub(self.len())
    }
}
```

```rust
// src/parallel/par_iter.rs

/// 安全分块（排除 padding）
///
/// # 参数
///
/// * `logical_len` - 逻辑元素数量
/// * `config` - 分块配置
///
/// # 返回
///
/// 分块范围列表，每个范围都在 [0, logical_len) 内。
///
/// # Safety Guarantee
///
/// 返回的分块保证：
/// 1. 所有范围是 [0, logical_len) 的子集
/// 2. 不会产生任何 [logical_len, allocated_len) 范围的访问
pub fn compute_safe_chunks(
    logical_len: usize,
    config: &ChunkConfig,
) -> Vec<Range<usize>> {
    // 分块基于逻辑长度，不涉及 padding
    if logical_len < config.min_chunk_size {
        return vec![0..logical_len];
    }

    let chunk_size = (logical_len / config.target_chunks)
        .max(config.min_chunk_size);

    (0..logical_len)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(logical_len);
            start..end
        })
        .collect()
}
```

### 5.4 切片操作的 Padding 保护

```rust
// src/tensor/ops.rs

impl<A, S, D> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// 创建切片视图
    ///
    /// # Safety Guarantee
    ///
    /// 切片边界检查确保：
    /// 1. 切片范围是 [0, logical_len) 的子集
    /// 2. 不会暴露 padding 元素
    ///
    /// # Panics
    ///
    /// 如果切片范围越界（包括访问 padding 区域），将 panic。
    pub fn slice(&self, ranges: &[Range<usize>]) -> TensorView<'_, A, D> {
        // 验证每个维度的范围
        for (i, range) in ranges.iter().enumerate() {
            assert!(
                range.end <= self.shape()[i],
                "slice range {} exceeds dimension {} (logical shape)",
                range.end,
                self.shape()[i]
            );
        }

        // 创建视图...
    }
}
```

### 5.5 Padding 零初始化

```rust
// src/storage/owned.rs

impl<A> Owned<A>
where
    A: Default + Clone,
{
    /// 创建带 padding 的对齐存储
    ///
    /// # Padding 初始化
    ///
    /// Padding 区域初始化为 `A::default()`，确保：
    /// 1. 无未初始化内存
    /// 2. 如果意外访问，得到合法值而非 UB
    ///
    /// # 性能考量
    ///
    /// 零初始化 padding 有轻微开销，但：
    /// - 安全性收益大于性能损失
    /// - 可通过 `unsafe` API 提供未初始化版本（需用户保证不访问 padding）
    pub fn zeros_with_padding(shape: &[usize], align: usize) -> Self {
        let logical_len: usize = shape.iter().product();
        let padded_len = Self::compute_padded_len(logical_len, align);

        let mut data = Vec::with_capacity(padded_len);
        data.resize(padded_len, A::default());

        Self {
            data,
            alignment: align,
        }
    }
}
```

---

## 6. ArcRepr 并发语义

### 6.1 Arc 的线程安全基础

```rust
// Arc 的线程安全由以下保证：

// 1. 原子引用计数
//    - Arc::clone() 原子增加 strong_count
//    - Arc::drop() 原子减少 strong_count
//    - 当 count 降为 0 时，释放数据

// 2. 内存顺序
//    - clone 使用 Acquire：确保看到之前所有修改
//    - drop 使用 Release：确保所有修改对其他线程可见

// 3. 数据访问
//    - 通过 &Arc<T> 只能获得 &T（只读）
//    - 修改需要 Arc::make_mut（CoW）或 try_unwrap
```

### 6.2 make_mut 的原子性

```rust
// src/storage/arc.rs

impl<A: Clone> ArcRepr<A> {
    /// 获取独占可变访问（写时复制）
    ///
    /// # 原子性保证
    ///
    /// `Arc::make_mut` 提供以下原子性保证：
    ///
    /// 1. **原子检查**：检查 `Arc::strong_count` 是否为 1
    /// 2. **原子递减**（如果需要复制）：原子减少原 Arc 的引用计数
    /// 3. **无竞争复制**：复制发生在当前线程，无并发写入
    ///
    /// # 多线程 make_mut 场景
    ///
    /// ```
    /// 初始状态：arc1 和 arc2 共享数据，strong_count = 2
    ///
    /// 线程 A                          线程 B
    /// ─────────────────────────       ─────────────────────────
    /// arc1.make_mut()
    ///   ├─ 检查 count = 2
    ///   ├─ 开始复制数据...
    ///   │                             arc2.make_mut()
    ///   │                               ├─ 检查 count = 2 (原子读取)
    ///   │                               ├─ 开始复制数据...
    ///   ├─ 替换 arc1.inner (原子操作)
    ///   │  count: 2 → 1                │
    ///   │                               ├─ 替换 arc2.inner (原子操作)
    ///   │                               │  count: 2 → 1
    ///   └─ 返回 &mut [A]               └─ 返回 &mut [A]
    ///
    /// 结果：
    /// - arc1 指向新副本 A'
    /// - arc2 指向新副本 A''
    /// - 原数据引用计数降为 0，被释放
    /// - 两个线程各自拥有独立的副本，无数据竞争
    /// ```
    ///
    /// # 证明要点
    ///
    /// 1. **无数据竞争**：
    ///    - 复制发生在各自线程的栈上
    ///    - 替换 `inner` 是原子操作
    ///    - 原数据只有读取（用于复制），无写入
    ///
    /// 2. **无重复释放**：
    ///    - 每次成功复制后，原引用计数原子递减
    ///    - 只有最后一个持有者会触发释放
    ///
    /// 3. **无悬垂指针**：
    ///    - 复制完成前，原数据保持有效
    ///    - `Arc` 保证数据在所有引用消失前不被释放
    ///
    /// # 性能提示
    ///
    /// 多线程并发 `make_mut` 会导致多次复制。
    /// 如果知道只有一个线程需要修改，建议：
    /// 1. 先 `Arc::try_unwrap` 获取所有权
    /// 2. 或使用单写多读模式
    #[inline]
    pub fn make_mut(&mut self) -> &mut [A] {
        // 使用 Arc::make_mut 的原子性
        let data = Arc::make_mut(&mut self.inner);

        // 返回偏移后的切片
        &mut data[self.offset..self.offset + self.len]
    }
}
```

### 6.3 多线程 clone 安全

```rust
// src/storage/arc.rs

impl<A> Clone for ArcRepr<A> {
    /// 克隆 ArcRepr（浅拷贝）
    ///
    /// # 线程安全
    ///
    /// 1. **原子计数**：`Arc::clone` 原子增加 `strong_count`
    /// 2. **无数据复制**：仅复制指针，不复制数据
    /// 3. **生命周期**：`Arc` 保证数据在所有引用消失前有效
    ///
    /// # 多线程场景
    ///
    /// ```
    /// let arc = ArcRepr::from_vec(vec![1, 2, 3]);
    ///
    /// // 跨线程传递
    /// let arc_clone = arc.clone();  // strong_count = 2
    /// thread::spawn(move || {
    ///     // arc_clone 在此线程有效
    ///     println!("{:?}", arc_clone.get(0));
    /// });
    ///
    /// // arc 在主线程仍有效
    /// println!("{:?}", arc.get(1));
    /// ```
    ///
    /// # Safety
    ///
    /// 由于 `Arc::clone` 是线程安全的，`ArcRepr::clone` 也是线程安全的，
    /// 前提是 `A: Send + Sync`。
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),  // 原子增加引用计数
            len: self.len,
            offset: self.offset,
        }
    }
}
```

### 6.4 try_into_owned 的原子性

```rust
// src/storage/arc.rs

impl<A> ArcRepr<A> {
    /// 尝试获取独占所有权
    ///
    /// # 原子性
    ///
    /// `Arc::try_unwrap` 原子地检查引用计数：
    /// - 如果 `strong_count == 1`：返回 `Ok(inner)`
    /// - 如果 `strong_count > 1`：返回 `Err(self)`
    ///
    /// # 多线程场景
    ///
    /// ```
    /// 线程 A                          线程 B
    /// ─────────────────────────       ─────────────────────────
    /// arc.try_into_owned()
    ///   ├─ 检查 count = 2
    ///   └─ 返回 Err(arc)              arc.try_into_owned()
    ///                                   ├─ 检查 count = 2
    ///                                   └─ 返回 Err(arc)
    ///
    /// // 两个线程都失败，arc 仍然共享
    /// ```
    ///
    /// # 用途
    ///
    /// 当知道只有一个持有者时，避免 `make_mut` 的复制：
    ///
    /// ```
    /// let arc = ArcRepr::from_vec(vec![1, 2, 3]);
    /// let owned: Owned<i32> = arc.try_into_owned().unwrap();
    /// // owned 拥有数据，无复制开销
    /// ```
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
}
```

---

## 7. ViewMutRepr 不是 Sync 的证明

### 7.1 为什么独占借用不能共享

```
┌─────────────────────────────────────────────────────────────────┐
│              ViewMutRepr 不是 Sync 的证明                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  定义：                                                         │
│  • T: Sync ⟺ &T: Send                                          │
│  • 即：T 的共享引用可以跨线程移动                                │
│                                                                 │
│  假设 ViewMutRepr<&mut A> 是 Sync：                             │
│  • 则 &ViewMutRepr<&mut A> 是 Send                              │
│  • 即可以将 &ViewMutRepr 移动到另一个线程                        │
│                                                                 │
│  这会导致什么问题？                                              │
│                                                                 │
│  线程 1                          线程 2                         │
│  ─────────────────────────       ─────────────────────────      │
│  let mut view: ViewMut<...> = ...                               │
│  let view_ref = &view                                           │
│  send(view_ref) ─────────────►  receive(view_ref)               │
│  let slice1 = view_ref.data()   let slice2 = view_ref.data()    │
│  // slice1: &mut [A]            // slice2: &mut [A]             │
│                                                                 │
│  问题：两个线程同时持有同一数据的 &mut [A]！                      │
│  • 违反 Rust 借用规则：同一时刻只能有一个可变引用                 │
│  • 可能导致数据竞争：两个线程同时写入同一元素                     │
│                                                                 │
│  结论：ViewMutRepr 不能是 Sync                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 形式化证明

```
定理：ViewMutRepr<&'a mut A> 不实现 Sync

证明（反证法）：

1. 假设 ViewMutRepr<&'a mut A>: Sync

2. 由 Sync 定义：&ViewMutRepr<&'a mut A>: Send

3. 假设存在两个线程 T1 和 T2，以及一个 ViewMutRepr 实例 v

4. T1 可以创建 &v 并发送到 T2（由步骤 2）

5. T1 可以通过 v.as_mut_slice() 获取 &mut [A]
   T2 可以通过 &v 创建另一个 &ViewMutRepr，然后通过某种方式获取 &mut [A]

6. 现在两个线程同时持有同一数据的可变引用
   这违反了 Rust 的别名规则

7. 与 Rust 内存安全保证矛盾

8. 因此假设不成立，ViewMutRepr 不是 Sync ∎
```

### 7.3 为什么 ViewMutRepr 是 Send

```
定理：ViewMutRepr<&'a mut A>: Send，当 A: Send

证明：

1. Send 的含义：类型可以安全地跨线程移动所有权

2. ViewMutRepr<&'a mut A> 的移动语义：
   • 移动后，原线程不再持有任何引用
   • 新线程成为独占访问者
   • 没有两个线程同时访问

3. 元素类型约束 A: Send：
   • 确保元素可以安全地在新线程中被读写
   • 如果 A 不是 Send（如 Rc），移动后在新线程访问是不安全的

4. 生命周期 'a：
   • 确保视图不会比源数据存活更久
   • 源数据必须在线程间共享或移动时保持有效

5. 无内部可变性：
   • ViewMutRepr 的字段（ptr, len）在创建后不变
   • 移动不会修改这些字段

6. 因此 ViewMutRepr<&'a mut A>: Send 是安全的 ∎
```

### 7.4 代码示例

```rust
// ViewMutRepr 是 Send 的正确用法

fn send_view_mut() {
    let mut owned = Owned::from_vec(vec![1, 2, 3]);
    let view_mut = ViewMut::from_owned(&mut owned);

    // 正确：移动 view_mut 到新线程
    thread::spawn(move || {
        // view_mut 在此线程独占访问
        let data = view_mut.as_mut_slice();
        data[0] = 10;
    });
}

// ViewMutRepr 不是 Sync 的体现

fn cannot_share_view_mut() {
    let mut owned = Owned::from_vec(vec![1, 2, 3]);
    let view_mut = ViewMut::from_owned(&mut owned);
    let view_ref = &view_mut;

    // 编译错误：&ViewMut 不是 Send
    // 因为 ViewMut 不是 Sync
    // thread::spawn(move || {
    //     println!("{:?}", view_ref);
    // });
}
```

---

## 8. 与 rayon 的集成

### 8.1 rayon 的 Send 约束

rayon 的 `ParallelIterator` 要求所有参与并行的数据实现 `Send`：

```rust
// rayon 的 trait 定义（简化）

pub trait ParallelIterator {
    type Item: Send;  // 注意：Item 必须是 Send

    // ...
}

pub trait IndexedParallelIterator: ParallelIterator {
    fn len(&self) -> usize;
    // ...
}
```

### 8.2 Tensor 的并行迭代约束

```rust
// src/parallel/par_iter.rs

use rayon::prelude::*;

/// 并行元素迭代
///
/// # 约束
///
/// * `A: Send + Sync` — 元素必须线程安全
/// * `D: Send + Sync` — 维度类型必须线程安全
///
/// # 为什么需要 Sync？
///
/// 并行迭代通过 `&TensorView` 访问元素，这要求 `A: Sync`。
/// 迭代器本身（包含视图引用）需要 `Send`，这要求 `D: Sync`。
impl<'a, A, D> ParallelIterator for ParElements<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension + Send + Sync,
{
    type Item = A;  // 克隆的元素，需要 Send

    // ...
}

/// 可变并行元素迭代
///
/// # 约束
///
/// * `A: Send + Sync` — 元素必须线程安全
///
/// # 为什么可变迭代也需要 Sync？
///
/// 虽然每个线程获得 `&mut A`，但分块逻辑需要读取形状和步长，
/// 这些操作通过共享引用访问，因此仍需要 `Sync`。
impl<'a, A, D> ParallelIterator for ParElementsMut<'a, A, D>
where
    A: Element + Send + Sync,
    D: Dimension + Send + Sync,
{
    type Item = &'a mut A;  // 可变引用，需要 A: Send

    // ...
}
```

### 8.3 ParElements 的实现

```rust
// src/parallel/par_iter.rs

/// 并行元素迭代器
///
/// # 设计
///
/// 使用 rayon 的 `IndexedParallelIterator` trait：
/// 1. 将元素范围分割为块
/// 2. 每个块发送到不同线程
/// 3. 每个线程独立遍历其块内的元素
///
/// # Safety
///
/// * 分块保证不重叠访问
/// * `A: Sync` 保证共享读取安全
/// * 迭代器消费后，视图仍然有效（生命周期约束）
pub struct ParElements<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
    chunk_config: ChunkConfig,
}

// 实现 rayon 的 Producer trait
struct ParElementsProducer<'a, A, D>
where
    A: Element + Sync,
    D: Dimension,
{
    base: TensorView<'a, A, D>,
    range: Range<usize>,
}

impl<'a, A, D> rayon::iter::plumbing::Producer for ParElementsProducer<'a, A, D>
where
    A: Element + Sync + Send + Clone,
    D: Dimension + Send + Sync,
{
    type Item = A;
    type IntoIter = Elements<'a, A, D>;

    fn into_iter(self) -> Self::IntoIter {
        // 创建子范围迭代器
        self.base.iter_range(self.range)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // 分割范围
        let mid = self.range.start + index;
        (
            ParElementsProducer {
                base: self.base.clone(),
                range: self.range.start..mid,
            },
            ParElementsProducer {
                base: self.base,
                range: mid..self.range.end,
            },
        )
    }
}
```

### 8.4 并行运算的 Send 约束

```rust
// src/parallel/par_ops.rs

/// 并行映射
///
/// # 约束
///
/// * `A: Send + Sync` — 输入元素线程安全
/// * `B: Send` — 输出元素可跨线程发送
/// * `F: Sync` — 函数可跨线程共享
///
/// # 为什么 F 需要 Sync？
///
/// 并行操作会在多个线程中调用同一个闭包。
/// 闭包通过共享引用访问，因此需要 `Sync`。
pub fn par_map<A, B, D, F>(tensor: &Tensor<A, D>, f: F) -> Tensor<B, D>
where
    D: Dimension,
    A: Element + Send + Sync,
    B: Element + Send,
    F: Fn(&A) -> B + Sync,
{
    // 使用 rayon 的 par_iter
    let mut output: Vec<B> = Vec::with_capacity(tensor.len());
    output.par_extend(tensor.par_iter().map(|x| f(x)));

    // 构造输出张量
    unsafe { Tensor::from_raw_vec_unchecked(output, tensor.raw_dim()) }
}

/// 并行归约
///
/// # 约束
///
/// * `A: Send + Sync` — 元素线程安全
/// * `F: Sync` — 归约函数可跨线程共享
/// * `ID: Sync + Clone` — 初始值函数可跨线程共享且可克隆
pub fn par_reduce<A, D, F, ID>(tensor: &Tensor<A, D>, identity: ID, op: F) -> A
where
    D: Dimension,
    A: Element + Send + Sync,
    F: Fn(A, A) -> A + Sync,
    ID: Fn() -> A + Sync + Clone,
{
    tensor
        .par_iter()
        .cloned()
        .reduce(identity, op)
}
```

### 8.5 与存储模式的兼容性

| 存储模式 | par_iter() | par_iter_mut() | 约束 |
|----------|:----------:|:--------------:|------|
| `Tensor<A, D>` (Owned) | ✅ | ✅ | `A: Send + Sync` |
| `TensorView<'a, A, D>` | ✅ | ❌ | `A: Sync` |
| `TensorViewMut<'a, A, D>` | ✅ | ✅ | `A: Send + Sync` |
| `ArcTensor<A, D>` | ✅ | ❌ (需 make_mut) | `A: Send + Sync` |

```rust
// ArcTensor 的并行迭代

impl<A, D> ArcTensor<A, D>
where
    A: Element + Send + Sync,
    D: Dimension,
{
    /// 只读并行迭代
    pub fn par_iter(&self) -> ParElements<'_, A, D> {
        self.view().par_iter()
    }

    /// 可变并行迭代（需要写时复制）
    ///
    /// # 注意
    ///
    /// 此方法会调用 `make_mut`，如果引用计数 > 1 会触发复制。
    pub fn par_iter_mut(&mut self) -> ParElementsMut<'_, A, D>
    where
        A: Clone,
    {
        // 先触发 CoW
        self.make_mut();
        // 然后创建可变视图
        // ...
    }
}
```

---

## 9. 与其他模块的交互

### 9.1 与 storage 模块的接口

```
┌─────────────────────────────────────────────────────────────────┐
│                    storage 模块的线程安全接口                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Storage trait 层次：                                            │
│                                                                 │
│  RawStorage                                                     │
│      │                                                          │
│      ├── Storage : RawStorage                                   │
│      │       │                                                  │
│      │       ├── StorageMut : Storage                           │
│      │       │       │                                          │
│      │       │       └── StorageOwned : StorageMut              │
│      │       │                                                  │
│      │       └── StorageShared : Storage  ← ArcRepr 实现        │
│      │                                                          │
│      └── RawStorageMut : RawStorage                             │
│                                                                 │
│  线程安全相关的关联类型和约束：                                   │
│                                                                 │
│  trait Storage {                                                │
│      type Elem: Send + Sync;  // 元素必须线程安全                │
│      // ...                                                     │
│  }                                                              │
│                                                                 │
│  trait StorageMut: Storage {                                    │
│      // 可变访问需要元素可移动                                   │
│      type Elem: Send;                                           │
│      // ...                                                     │
│  }                                                              │
│                                                                 │
│  trait StorageShared: Storage {                                 │
│      // 共享存储的线程安全由 Arc 保证                            │
│      fn make_mut(&mut self) -> &mut [Self::Elem];               │
│      // ...                                                     │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 与 parallel 模块的接口

```
┌─────────────────────────────────────────────────────────────────┐
│                    parallel 模块的线程安全要求                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  并行迭代器 trait：                                              │
│                                                                 │
│  pub trait IntoParallelRefIterator<'a> {                        │
│      type Item: Send + Sync;  // 并行迭代元素必须线程安全        │
│      type Iter: ParallelIterator<Item = Self::Item>;            │
│      fn par_iter(&'a self) -> Self::Iter;                       │
│  }                                                              │
│                                                                 │
│  pub trait IntoParallelRefMutIterator<'a> {                     │
│      type Item: Send;  // 可变迭代元素必须可移动                 │
│      type Iter: ParallelIterator<Item = &'a mut Self::Item>;    │
│      fn par_iter_mut(&'a mut self) -> Self::Iter;               │
│  }                                                              │
│                                                                 │
│  实现条件：                                                      │
│                                                                 │
│  impl<'a, S, D, A> IntoParallelRefIterator<'a> for TensorBase<S, D>
///  where                                                         │
│      S: Storage<Elem = A>,                                      │
│      D: Dimension,                                              │
│      A: Element + Send + Sync,  // 关键约束                     │
│  {                                                              │
│      // ...                                                     │
│  }                                                              │
│                                                                 │
│  分块安全保证：                                                  │
│                                                                 │
│  • compute_safe_chunks() 保证不重叠                             │
│  • Padding 区域被排除                                           │
│  • 生命周期确保视图有效                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 与 iter 模块的接口

```
┌─────────────────────────────────────────────────────────────────┐
│                    iter 模块的线程安全考量                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  串行迭代器（不涉及线程安全）：                                   │
│                                                                 │
│  • Elements<'a, A, D> — 元素迭代器                              │
│  • AxisIter<'a, A, D> — 轴迭代器                                │
│  • Windows<'a, A, D> — 窗口迭代器                               │
│  • Indexed<'a, A, D> — 带索引迭代器                             │
│                                                                 │
│  这些迭代器本身不需要实现 Send/Sync，因为：                       │
│  • 它们包含 & 或 &mut 引用，生命周期约束了使用范围                │
│  • 跨线程传递由 parallel 模块处理                               │
│                                                                 │
│  但迭代器产出的 Item 类型需要考虑线程安全：                       │
│                                                                 │
│  • Elements 产出 &A — 需要 A: Sync 才能跨线程                    │
│  • ElementsMut 产出 &mut A — 需要 A: Send 才能跨线程             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 模块依赖关系图

```
                    ┌─────────────────┐
                    │   tensor_core   │
                    │ (TensorBase)    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    storage      │ │      iter       │ │    parallel     │
│ (Owned, View,   │ │ (Elements,      │ │ (ParElements,   │
│  ViewMut, Arc)  │ │  AxisIter)      │ │  par_map, etc)  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │    Send/Sync      │    Send/Sync      │    Send/Sync
         │    实现           │    约束           │    约束
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  thread_safety  │
                    │ (本文档)         │
                    │ 设计原则和证明   │
                    └─────────────────┘
```

---

## 10. 实现任务分解

### 任务概览

| 任务 | 预估时间 | 优先级 | 依赖 |
|------|----------|--------|------|
| T1: Owned Send/Sync 实现 | 10 分钟 | 高 | 无 |
| T2: ViewRepr Send/Sync 实现 | 15 分钟 | 高 | 无 |
| T3: ViewMutRepr Send/Sync 实现 | 15 分钟 | 高 | 无 |
| T4: ArcRepr Send/Sync 实现 | 10 分钟 | 高 | 无 |
| T5: 并行迭代分块安全验证 | 15 分钟 | 中 | T1-T4 |
| T6: Padding 安全测试 | 10 分钟 | 中 | T5 |
| T7: 文档和 Safety 注释 | 15 分钟 | 中 | T1-T4 |
| T8: 线程安全集成测试 | 15 分钟 | 低 | T1-T6 |

### T1: Owned Send/Sync 实现 (10 分钟)

**目标**：为 `Owned<A>` 实现 `Send` 和 `Sync`

**步骤**：
1. 在 `src/storage/owned.rs` 中添加 `unsafe impl Send`
2. 添加 `unsafe impl Sync`
3. 编写详细的 `# Safety` 文档注释
4. 添加单元测试验证约束传播

**验收标准**：
- `Owned<i32>: Send + Sync` 编译通过
- `Owned<Rc<i32>>: Send` 编译失败（符合预期）

**代码框架**：
```rust
// src/storage/owned.rs

/// # Safety
/// 
/// [详细 Safety 证明]
unsafe impl<A: Send> Send for Owned<A> {}

/// # Safety
/// 
/// [详细 Safety 证明]
unsafe impl<A: Sync> Sync for Owned<A> {}

#[cfg(test)]
mod tests {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn owned_is_send_sync() {
        assert_send::<Owned<i32>>();
        assert_sync::<Owned<i32>>();
    }
}
```

### T2: ViewRepr Send/Sync 实现 (15 分钟)

**目标**：为 `ViewRepr<&'a A>` 实现 `Send` 和 `Sync`

**步骤**：
1. 在 `src/storage/view.rs` 中添加 `unsafe impl Send`
2. 添加 `unsafe impl Sync`
3. 解释为什么需要 `A: Sync` 而非 `A: Send`
4. 添加生命周期相关的测试

**验收标准**：
- `ViewRepr<&i32>: Send + Sync` 当 `i32: Sync`
- 测试跨线程传递视图

**代码框架**：
```rust
// src/storage/view.rs

/// # Safety
/// 
/// [详细 Safety 证明，包括为什么需要 A: Sync]
unsafe impl<'a, A: Sync> Send for ViewRepr<&'a A> {}

/// # Safety
/// 
/// [详细 Safety 证明]
unsafe impl<'a, A: Sync> Sync for ViewRepr<&'a A> {}

#[cfg(test)]
mod tests {
    use std::thread;

    #[test]
    fn view_cross_thread() {
        let owned = Owned::from_vec(vec![1, 2, 3]);
        let view = View::from_owned(&owned);

        thread::spawn(move || {
            // view 在新线程中可用
            assert_eq!(*view.get(0).unwrap(), 1);
        }).join().unwrap();
    }
}
```

### T3: ViewMutRepr Send/Sync 实现 (15 分钟)

**目标**：为 `ViewMutRepr<&'a mut A>` 实现 `Send`（不实现 `Sync`）

**步骤**：
1. 在 `src/storage/view_mut.rs` 中添加 `unsafe impl Send`
2. 解释为什么 `ViewMutRepr` 不是 `Sync`
3. 添加负面测试（验证不是 `Sync`）
4. 添加跨线程移动测试

**验收标准**：
- `ViewMutRepr<&mut i32>: Send` 当 `i32: Send`
- `ViewMutRepr<&mut i32>` 不是 `Sync`（编译器自动推断）

**代码框架**：
```rust
// src/storage/view_mut.rs

/// # Safety
/// 
/// [详细 Safety 证明，包括为什么可移动但不可共享]
unsafe impl<'a, A: Send> Send for ViewMutRepr<&'a mut A> {}

// ViewMutRepr 不实现 Sync（包含 *mut A，编译器不会自动实现）

#[cfg(test)]
mod tests {
    use std::thread;

    fn assert_send<T: Send>() {}
    fn assert_not_sync<T>() where T: Send {}  // 仅 Send，不 Sync

    #[test]
    fn view_mut_is_send() {
        assert_send::<ViewMutRepr<&mut i32>>();
    }

    #[test]
    fn view_mut_cross_thread() {
        let mut owned = Owned::from_vec(vec![1, 2, 3]);
        let view_mut = ViewMut::from_owned(&mut owned);

        thread::spawn(move || {
            // view_mut 在新线程中独占访问
            view_mut.as_mut_slice()[0] = 10;
        }).join().unwrap();

        // 注意：owned 已被移动，这里无法访问
    }
}
```

### T4: ArcRepr Send/Sync 实现 (10 分钟)

**目标**：为 `ArcRepr<A>` 实现 `Send` 和 `Sync`

**步骤**：
1. 在 `src/storage/arc.rs` 中添加 `unsafe impl Send`
2. 添加 `unsafe impl Sync`
3. 解释为什么需要 `A: Send + Sync`
4. 添加 `make_mut` 的线程安全测试

**验收标准**：
- `ArcRepr<i32>: Send + Sync` 当 `i32: Send + Sync`
- 多线程 clone 和 make_mut 安全

**代码框架**：
```rust
// src/storage/arc.rs

/// # Safety
/// 
/// [详细 Safety 证明，包括 Arc 的原子性]
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}

/// # Safety
/// 
/// [详细 Safety 证明]
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}

#[cfg(test)]
mod tests {
    use std::thread;

    #[test]
    fn arc_cross_thread_clone() {
        let arc = ArcRepr::from_vec(vec![1, 2, 3]);
        let arc2 = arc.clone();

        thread::spawn(move || {
            // arc 在新线程中
            assert_eq!(*arc.get(0).unwrap(), 1);
        }).join().unwrap();

        // arc2 在主线程中
        assert_eq!(*arc2.get(1).unwrap(), 2);
    }

    #[test]
    fn arc_make_mut_thread_safe() {
        let arc = ArcRepr::from_vec(vec![1, 2, 3]);
        let arc2 = arc.clone();

        let mut arc_clone = arc;
        thread::spawn(move || {
            let data = arc_clone.make_mut();
            data[0] = 10;
        }).join().unwrap();

        // arc2 不受影响（CoW）
        assert_eq!(*arc2.get(0).unwrap(), 1);
    }
}
```

### T5: 并行迭代分块安全验证 (15 分钟)

**目标**：验证并行迭代的分块算法是安全的

**步骤**：
1. 审查 `compute_safe_chunks` 的实现
2. 添加边界条件测试
3. 添加 Padding 排除测试
4. 添加并发访问不重叠验证

**验收标准**：
- 分块覆盖所有元素
- 分块不重叠
- 不访问 Padding 区域

**代码框架**：
```rust
// src/parallel/par_iter.rs

#[cfg(test)]
mod chunk_safety_tests {
    use super::*;

    #[test]
    fn chunks_cover_all_elements() {
        let len = 100;
        let config = ChunkConfig::default();
        let chunks = compute_safe_chunks(len, &config);

        let covered: Vec<bool> = vec![false; len];
        for (start, end) in &chunks {
            for i in *start..*end {
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|&x| x));
    }

    #[test]
    fn chunks_do_not_overlap() {
        let len = 100;
        let config = ChunkConfig::default();
        let chunks = compute_safe_chunks(len, &config);

        for i in 0..chunks.len() {
            for j in (i+1)..chunks.len() {
                let (s1, e1) = chunks[i];
                let (s2, e2) = chunks[j];
                // 检查不重叠
                assert!(e1 <= s2 || e2 <= s1);
            }
        }
    }

    #[test]
    fn chunks_exclude_padding() {
        let logical_len = 12;
        let allocated_len = 16;  // 4 个 padding 元素
        let config = ChunkConfig::default();
        let chunks = compute_safe_chunks(logical_len, &config);

        for (start, end) in &chunks {
            assert!(*end <= logical_len);
        }
    }
}
```

### T6: Padding 安全测试 (10 分钟)

**目标**：验证 Padding 区域不会被访问

**步骤**：
1. 创建带 Padding 的数组
2. 验证迭代器不访问 Padding
3. 验证切片不暴露 Padding
4. 验证并行迭代不分配 Padding

**验收标准**：
- 迭代器产出正确数量的元素
- 切片边界检查正确
- 并行分块不包含 Padding

**代码框架**：
```rust
// src/storage/owned.rs

#[cfg(test)]
mod padding_safety_tests {
    use super::*;

    #[test]
    fn iterator_excludes_padding() {
        let owned = Owned::zeros_with_padding(&[12], 64);
        let logical_len = owned.len();
        let padding_len = owned.padding_len();

        // 迭代器只产出逻辑元素
        let count = owned.iter().count();
        assert_eq!(count, logical_len);
        assert!(padding_len > 0);  // 确认存在 padding
    }

    #[test]
    #[should_panic]
    fn slice_cannot_access_padding() {
        let owned = Owned::zeros_with_padding(&[12], 64);
        // 尝试切片到 padding 区域
        owned.slice(0..owned.allocated_len());
    }
}
```

### T7: 文档和 Safety 注释 (15 分钟)

**目标**：为所有线程安全相关代码添加文档

**步骤**：
1. 审查所有 `unsafe impl` 的 `# Safety` 注释
2. 确保包含反例分析
3. 添加模块级文档说明线程安全策略
4. 更新 CHANGELOG

**验收标准**：
- 每个 `unsafe impl` 有完整的 `# Safety` 注释
- 模块文档解释线程安全策略
- 文档通过 `cargo doc` 检查

**代码框架**：
```rust
// src/storage/mod.rs

//! # 线程安全
//!
//! 本模块的所有存储类型都实现了适当的 `Send` 和 `Sync` trait：
//!
//! | 存储类型 | Send | Sync | 条件 |
//! |----------|------|------|------|
//! | `Owned<A>` | ✅ | ✅ | `A: Send + Sync` |
//! | `ViewRepr<&'a A>` | ✅ | ✅ | `A: Sync` |
//! | `ViewMutRepr<&'a mut A>` | ✅ | ❌ | `A: Send` |
//! | `ArcRepr<A>` | ✅ | ✅ | `A: Send + Sync` |
//!
//! ## 设计原则
//!
//! [解释设计原则]
//!
//! ## 与并行迭代的关系
//!
//! [解释与 rayon 的集成]
```

### T8: 线程安全集成测试 (15 分钟)

**目标**：添加跨模块的线程安全集成测试

**步骤**：
1. 创建 `tests/thread_safety.rs`
2. 测试所有存储模式的跨线程使用
3. 测试并行迭代的正确性
4. 测试 ArcRepr 的并发行为

**验收标准**：
- 所有测试通过
- 测试覆盖所有存储模式
- 测试覆盖常见并发场景

**代码框架**：
```rust
// tests/thread_safety.rs

use Senon::{Tensor, TensorView, TensorViewMut, ArcTensor};
use std::thread;

#[test]
fn owned_cross_thread() {
    let tensor = Tensor1::from_vec(vec![1, 2, 3]);
    thread::spawn(move || {
        assert_eq!(tensor.len(), 3);
    }).join().unwrap();
}

#[test]
fn view_cross_thread() {
    let tensor = Tensor1::from_vec(vec![1, 2, 3]);
    let view = tensor.view();
    thread::spawn(move || {
        assert_eq!(*view.get(0).unwrap(), 1);
    }).join().unwrap();
}

#[test]
fn arc_concurrent_access() {
    let arc = ArcTensor1::from_vec(vec![1, 2, 3]);
    let handles: Vec<_> = (0..4).map(|_| {
        let arc = arc.clone();
        thread::spawn(move || {
            // 并发读取
            arc.get(0).unwrap()
        })
    }).collect();

    for h in handles {
        assert_eq!(*h.join().unwrap(), 1);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_iteration_correctness() {
    let tensor = Tensor1::from_vec((0..100).collect());
    let sum: i32 = tensor.par_iter().sum();
    assert_eq!(sum, (0..100).sum());
}
```

---

## 11. 设计决策记录

### D1: 为什么显式实现 unsafe impl 而非依赖自动推导

**背景**：`Owned<A>` 的字段全是 `Send + Sync`，编译器会自动推导 `Send + Sync`。

**决策**：显式使用 `unsafe impl` 并添加详细 `# Safety` 注释。

**理由**：
1. **文档化意图**：显式声明表明我们考虑过线程安全
2. **未来兼容**：如果将来添加非 Send/Sync 字段，编译器会报错
3. **教育价值**：为其他开发者提供参考

**替代方案**：依赖自动推导，不写任何代码。

**取舍**：显式实现增加少量代码，但提高可维护性。

### D2: ViewMutRepr 为什么不实现负面 trait

**背景**：Rust stable 不支持显式 `impl !Sync`。

**决策**：不实现 `Sync`，依赖编译器自动推断。

**理由**：
1. `ViewMutRepr` 包含 `*mut A`，编译器不会自动实现 `Sync`
2. 这正是我们想要的行为
3. 通过文档说明 `ViewMutRepr` 不是 `Sync`

**替代方案**：使用 nightly 特性显式声明 `!Sync`。

**取舍**：保持 stable 兼容，通过文档表达意图。

### D3: 并行迭代的最小块大小选择

**背景**：分块太小会增加并行开销，太大会降低并行度。

**决策**：默认最小块大小 4096 元素。

**理由**：
1. 4096 ≈ 4KB，是一个内存页的 1/4
2. 经验值：太小（< 1K）时线程调度开销明显
3. 与 rayon 默认行为接近

**替代方案**：
- 8192：更保守，减少开销但降低小数组并行度
- 1024：更激进，增加小数组并行度但可能增加开销

**取舍**：4096 是平衡点，可通过配置调整。

### D4: 非连续可变迭代的策略

**背景**：非连续数组的可变迭代难以直接分块。

**决策**：沿第一轴分块，无法分块时回退串行。

**理由**：
1. 大多数科学计算场景沿 batch 轴并行
2. 沿第一轴分块保证子数组不重叠
3. 回退串行保证正确性

**替代方案**：
- 完全禁止非连续可变迭代：太严格
- 强制转为连续：增加拷贝开销

**取舍**：提供并行机会，同时保证正确性。

### D5: Padding 零初始化 vs 未初始化

**背景**：Padding 区域可以零初始化或保持未初始化。

**决策**：默认零初始化，提供 unsafe API 用于性能优化。

**理由**：
1. 安全性：防止意外访问导致 UB
2. 调试：零值更容易识别
3. 性能：初始化开销相对较小

**替代方案**：
- 完全不初始化：性能最优，但风险高
- 使用 `MaybeUninit`：复杂，需要 unsafe 访问

**取舍**：默认安全，为极端性能需求提供 unsafe 选项。

### D6: ArcRepr make_mut 的重复复制问题

**背景**：多线程同时 `make_mut` 可能导致多次复制。

**决策**：接受可能的重复复制，不添加额外同步。

**理由**：
1. `Arc::make_mut` 已经是原子的，无数据竞争
2. 重复复制是性能问题，不是正确性问题
3. 添加额外同步会增加常见路径的开销

**替代方案**：
- 使用全局锁：保证单次复制，但降低并发性
- 使用条件变量：复杂，收益不明确

**取舍**：保持简单，文档中说明性能提示。

---

## 附录 A: Send/Sync 快速参考

```rust
// 快速判断类型的 Send/Sync

// 1. 基本类型：全是 Send + Sync
fn _basic_types() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<i32>();
    assert_send_sync::<f64>();
    assert_send_sync::<bool>();
}

// 2. 引用类型
fn _reference_types() {
    // &T 是 Send ⟺ T: Sync
    // &T 是 Sync ⟺ T: Sync
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<&i32>();    // i32: Sync ✓
    assert_sync::<&i32>();    // i32: Sync ✓

    // &mut T 是 Send ⟺ T: Send
    // &mut T 不是 Sync
    assert_send::<&mut i32>(); // i32: Send ✓
    // assert_sync::<&mut i32>(); // 编译失败
}

// 3. 裸指针
fn _raw_pointers() {
    // *const T 和 *mut T 既不是 Send 也不是 Sync
    fn assert_not_send<T>() where T: ?Sized {}
    // fn _assert_send<T: Send>() {}

    // 以下会编译失败：
    // _assert_send::<*const i32>();
    // _assert_send::<*mut i32>();
}

// 4. 智能指针
fn _smart_pointers() {
    // Box<T>: Send + Sync ⟺ T: Send + Sync
    // Rc<T>: 不是 Send，不是 Sync
    // Arc<T>: Send + Sync ⟺ T: Send + Sync

    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<Box<i32>>();
    // assert_send_sync::<Rc<i32>>(); // 编译失败
    assert_send_sync::<Arc<i32>>();
}
```

---

## 附录 B: 线程安全检查清单

实现新的存储类型时，请检查以下项目：

- [ ] **确定 Send/Sync 状态**
  - [ ] 是否需要跨线程移动？（Send）
  - [ ] 是否需要跨线程共享？（Sync）
  - [ ] 内部字段是否支持？

- [ ] **编写 unsafe impl**
  - [ ] 添加 `unsafe impl Send`（如果需要）
  - [ ] 添加 `unsafe impl Sync`（如果需要）
  - [ ] 编写详细的 `# Safety` 注释

- [ ] **Safety 注释检查**
  - [ ] 解释语义正确性
  - [ ] 解释约束必要性
  - [ ] 提供反例分析
  - [ ] 分析内部状态可变性

- [ ] **添加测试**
  - [ ] `assert_send` / `assert_sync` 静态检查
  - [ ] 跨线程使用测试
  - [ ] 并发访问测试（如果适用）

- [ ] **更新文档**
  - [ ] 模块级文档更新 Send/Sync 矩阵
  - [ ] 类型文档说明线程安全特性
  - [ ] 更新 CHANGELOG

---

## 变更历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-03-28 | 初始版本 |
