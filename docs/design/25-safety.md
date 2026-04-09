# 线程安全设计

> 文档编号: 25 | 模块: 横切关注点 | 阶段: Phase 2
> 前置文档: `05-storage.md`, `07-tensor.md`
> 需求参考: 需求说明书 §10

---

## 1. 模块定位

线程安全是 Xenon 的横切关注点，贯穿所有存储模式和计算后端。本文档定义各存储模式（参见 `05-storage.md §4`）的 `Send`/`Sync` 实现规则，确保 Xenon 张量（参见 `07-tensor.md §4`）可在多线程环境下安全使用。

### 1.1 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| Send/Sync 实现 | 各存储模式的 Send/Sync trait 实现 | 锁机制 (Mutex/RwLock) |
| 正确性保证 | unsafe impl 的安全性论证和证明 | 通道 (mpsc/crossbeam) |
| 并行安全约束 | SIMD 与并行组合的安全约束 | 异步运行时 (tokio/async-std) |
| 编译期保证 | 通过 Rust 类型系统在编译期排除数据竞争 | 运行时锁或同步原语 |
| 广播安全 | 广播结果不可变迭代的约束 | — |

### 1.2 设计原则

| 原则 | 体现 |
|------|------|
| 编译期保证 | 无运行时锁，所有线程安全由 Rust 类型系统在编译期保证 |
| 最小约束 | 每种存储模式仅实现其语义允许的最小 Send/Sync 约束 |
| unsafe 安全论证 | 每个 unsafe impl 附带完整 SAFETY 注释 |
| 所有权协同 | 充分利用 Rust 所有权系统与线程安全的天然协同 |

### 1.3 在架构中的位置

```
依赖层级：
L0: error, private
L1: dimension, element, complex
L2: layout (依赖 dimension)
L3: storage (依赖 layout)
L4: tensor (依赖 storage, dimension)
L5: ops/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/

横切关注点：
┌─────────────────────────────────────────────────────────────────┐
│  线程安全 (Send/Sync)  ← 当前文档（横跨 L3-L5 各模块）               │
│  ─ 横贯 storage (L3)、tensor (L4)、parallel/simd (L5)            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 线程安全在 Xenon 中的重要性

| 场景 | 线程安全需求 |
|------|--------------|
| 并行迭代 | 多线程同时访问不同元素区间 |
| 跨线程共享 | 通过 ArcRepr 在线程间传递只读数据 |
| 写时复制 | ArcRepr::make_mut() 的原子性保证 |
| 数据竞争预防 | 确保 ViewMutRepr 独占访问不被共享 |
| rayon 集成 | ParallelIterator 要求 Send 约束 |

---

## 2. 文件位置

线程安全实现散布于各存储模块中，不单独创建文件：

```
src/
├── storage/
│   ├── mod.rs          # 模块级线程安全文档
│   ├── owned.rs        # Owned<A> 的 Send/Sync impl
│   ├── view.rs         # ViewRepr<&'a A> 的 Send/Sync impl
│   ├── view_mut.rs     # ViewMutRepr<&'a mut A> 的 Send/Sync impl
│   └── arc.rs          # ArcRepr<A> 的 Send/Sync impl
├── parallel/
│   ├── mod.rs          # ParallelGuard, 嵌套并行防护
│   └── par_iter.rs     # ParElements 线程安全约束
└── simd/
    └── mod.rs          # Arch 缓存的线程安全初始化
```

---

## 3. 依赖关系

### 3.1 依赖图

```
src/storage/
├── core::marker         # PhantomData, Send, Sync
├── alloc::sync::Arc     # ArcRepr 使用的原子引用计数（no_std 兼容；std feature 下 alloc::sync::Arc 等价于 std::sync::Arc）
└── std::cell::Cell      # thread_local (parallel 模块)

src/parallel/
├── std::sync::atomic    # AtomicUsize (阈值)
├── std::cell::Cell      # thread_local (嵌套并行检测)
└── rayon                # ParallelIterator (Send 约束)
```

### 3.2 依赖精确到类型级

| 来源模块 | 使用的类型/trait |
|----------|-----------------|
| `core::marker` | `PhantomData<A>`, `Send`, `Sync` |
| `std::sync` | `Arc<Vec<A>>`, `AtomicUsize` |
| `std::cell` | `Cell<usize>`, `Cell<Option<usize>>` |
| `rayon::iter` | `ParallelIterator` (要求 `Item: Send`，参见 `09-parallel.md §4`) |

### 3.3 依赖方向声明

> **依赖方向：线程安全是横切关注点。** 各存储模块自行声明 Send/Sync（参见 `05-storage.md §5`），parallel 模块消费这些约束（参见 `09-parallel.md §4`）。无循环依赖。

---

## 4. 公共 API 设计

> **注意：** `05-storage.md §5.7` 包含一个较早的、不正确的 Send/Sync 条件表，已被更正。本文档（25-safety.md）是 Send/Sync 条件的权威参考。

### 4.1 Send/Sync 实现规则表

完整矩阵：

| 存储模式 | Send | Sync | 条件 | 理由 |
|----------|:----:|:----:|------|------|
| `Owned<A>` | ✅ | ✅ | `A: Send + Sync` | Vec 内部线程安全，元素约束传播 |
| `ViewRepr<&'a A>` | ✅ | ✅ | `A: Sync` | 共享引用跨线程共享要求 `T: Sync` |
| `ViewMutRepr<&'a mut A>` | ✅ | ✗ | `A: Send` | 独占借用不可共享，但可转移 |
| `ArcRepr<A>` | ✅ | ✅ | `A: Send + Sync` | Arc 原子计数，要求元素线程安全 |

各存储模式的完整 API 定义参见 `05-storage.md §4`。

### 4.2 Owned<A> 的 Send/Sync

```rust
// src/storage/owned.rs

/// # Safety
///
/// `Owned<A>` implements `Send` because:
///
/// 1. **Exclusive ownership**: `Owned` has full ownership of the data.
///    Transferring `Owned` transfers all data to the new thread.
///
/// 2. **Element type constraint**: `A: Send` ensures elements can be safely moved across threads.
///
/// 3. **Vec safety**: `Vec<A>` is `Send` when `A: Send`,
///    we are just making this property explicit.
///
/// 4. **No shared state**: After transfer, the original thread holds no references.
///
/// **Counter-example: if `A` is not `Send`**
///
/// Suppose `A = Rc<i32>` (not `Send`), moving `Owned<Rc<i32>>`
/// to another thread could cause two threads to access the same `Rc`
/// simultaneously, and `Rc`'s reference count is not atomic, leading to data races.
unsafe impl<A: Send> Send for Owned<A> {}

/// # Safety
///
/// `Owned<A>` implements `Sync` because:
///
/// 1. **Shared read-only access**: Multiple threads can hold `&Owned<A>` simultaneously,
///    through which only read-only operations (e.g. `get()`) are possible.
///
/// 2. **Element type constraint**: `A: Sync` ensures elements can be safely shared by reference across threads.
///
/// 3. **Interior immutability**: Through `&Owned` the internal data cannot be modified
///    (`&mut Owned` is required to call `get_mut()`).
///
/// 4. **Vec safety**: `Vec<A>` is `Sync` when `A: Sync`.
unsafe impl<A: Sync> Sync for Owned<A> {}
```

### 4.3 ViewRepr<&'a A> 的 Send/Sync

```rust
// src/storage/view.rs

/// # Safety
///
/// `ViewRepr<&'a A>` implements `Send` because:
///
/// 1. **Ownership transfer semantics**: Moving `ViewRepr` does not move the underlying data,
///    it only transfers the view's metadata (pointer + length) to the new thread.
///
/// 2. **Shared reference constraint**: `&'a A` can be moved across threads if and only if `A: Sync`.
///    This ensures that even when multiple threads hold `&A`, they can only perform read-only access.
///
/// 3. **Lifetime guarantee**: The `'a` lifetime ensures the view does not outlive the source data,
///    preventing dangling pointers.
///
/// 4. **No interior mutability**: The view itself does not own data and has no interior mutability,
///    so moving it across threads does not cause data races.
///
/// **Counter-example: if `A` is not `Sync`**
///
/// Suppose `A = Cell<i32>` (not `Sync`), two threads could both hold
/// `&Cell<i32>` and modify data via `Cell::set`, leading to data races.
unsafe impl<'a, A: Sync> Send for ViewRepr<&'a A> {}

/// # Safety
///
/// `ViewRepr<&'a A>` implements `Sync` because:
///
/// 1. **Shared access safety**: Multiple threads can hold `&ViewRepr<&'a A>` simultaneously,
///    which is equivalent to multiple threads holding `&&A` (a shared reference to a shared reference).
///
/// 2. **Read-only access**: Accessing the view through a shared reference only permits read-only operations,
///    without modifying the view itself or the underlying data.
///
/// 3. **Stateless**: The view's `ptr` and `len` are immutable after creation,
///    so reading these fields from multiple threads is safe.
unsafe impl<'a, A: Sync> Sync for ViewRepr<&'a A> {}
```

### 4.4 ViewMutRepr<&'a mut A> 的 Send（不实现 Sync）

```rust
// src/storage/view_mut.rs

/// # Safety
///
/// `ViewMutRepr<&'a mut A>` implements `Send` because:
///
/// 1. **Exclusive ownership transfer**: Moving `ViewMutRepr` transfers exclusive access to the new thread.
///    Rust's borrow checker guarantees the original thread holds no more references.
///
/// 2. **No aliasing guarantee**: Since `&mut A` is an exclusive reference, at any given moment only one
///    `ViewMutRepr` can access the data. After cross-thread movement, the new thread becomes the sole accessor.
///
/// 3. **Element type constraint**: `&mut T` is `Send` if and only if `T: Send`.
///    This ensures elements can be safely read and written in the new thread.
///
/// 4. **Lifetime invariance**: The `'a` lifetime ensures the view does not outlive the source data.
///
/// **Counter-example: if `A` is not `Send`**
///
/// Suppose `A = Rc<i32>` (not `Send`), moving `ViewMutRepr<&mut Rc<i32>>`
/// to another thread could cause two threads to access the same `Rc`
/// simultaneously, and `Rc`'s reference count is not atomic, leading to data races.
///
/// **Note: ViewMutRepr does not implement Clone**
///
/// `ViewMutRepr` deliberately does not implement `Clone`, because copying would create aliases,
/// violating exclusive semantics. This is a key guarantee for `ViewMutRepr` thread safety.
unsafe impl<'a, A: Send> Send for ViewMutRepr<&'a mut A> {}

// ViewMutRepr does not implement Sync
//
// Reason: &mut T cannot be shared; Rust's borrowing rules forbid it.
//
// If ViewMutRepr were Sync, then &ViewMutRepr could be shared across threads,
// meaning multiple threads could simultaneously obtain &mut [A], causing aliasing
// and potential data races.
//
// Rust's negative trait impls (!Sync) cannot be explicitly written in stable Rust,
// but since ViewMutRepr contains *mut A (which is not Sync), the compiler will not
// auto-derive Sync, which is exactly the behavior we want.
```

### 4.5 ArcRepr<A> 的 Send/Sync

```rust
// src/storage/arc.rs

/// # Safety
///
/// `ArcRepr<A>` implements `Send` because:
///
/// 1. **Arc atomicity**: `Arc<Vec<A>>` uses atomic reference counting,
///    incrementing/decrementing the count across threads is safe.
///
/// 2. **Element constraint**: `A: Send + Sync` ensures:
///    - `Send`: data can be moved between threads (when the Arc is unique)
///    - `Sync`: multiple threads can hold `&A` simultaneously
///
/// 3. **Read-only sharing**: When multiple `ArcRepr`s share the same data, they can only read.
///    Writing requires obtaining exclusive access via `make_mut()`.
///
/// 4. **make_mut atomicity**: `Arc::make_mut` guarantees:
///    - Atomically check the reference count
///    - If >1, copy the data and atomically decrement the original reference count
///    - No data races
///
/// **Counter-example: if `A` is not `Send + Sync`**
///
/// Suppose `A = Cell<i32>` (not `Sync`), multiple threads could simultaneously
/// access the same `Cell` through different `ArcRepr`s, leading to data races.
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}

/// # Safety
///
/// `ArcRepr<A>` implements `Sync` because:
///
/// 1. **Shared reference safety**: Multiple threads can hold `&ArcRepr<A>` simultaneously,
///    allowing them to read data (via `get()`).
///
/// 2. **Arc synchronization guarantee**: `Arc<Vec<A>>` is `Sync` when `A: Send + Sync`,
///    because the internal data can be safely shared by reference across multiple threads.
///
/// 3. **No interior mutability**: Data cannot be modified through `&ArcRepr`.
///    `make_mut(&mut self)` requires `&mut ArcRepr`, and the borrow checker
///    guarantees only one mutable reference exists at any given time.
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

### 4.6 ViewMutRepr 不实现 Sync 的证明

```
┌─────────────────────────────────────────────────────────────────┐
│              ViewMutRepr 不是 Sync 的证明                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  定义：T: Sync ⟺ &T: Send                                      │
│                                                                 │
│  假设 ViewMutRepr<&mut A> 是 Sync：                              │
│  • 则 &ViewMutRepr 可以跨线程移动                                  │
│                                                                 │
│  线程 1                          线程 2                          │
│  ─────────────────────────       ─────────────────────────      │
│  let mut view: ViewMut<...> = ...                               │
│  let view_ref = &view                                           │
│  send(view_ref) ─────────────►  receive(view_ref)               │
│  let slice1 = view_ref.data()   let slice2 = view_ref.data()    │
│  // slice1: &mut [A]            // slice2: &mut [A]             │
│                                                                 │
│  问题：两个线程同时持有同一数据的 &mut [A]！                         │
│  • 违反 Rust 借用规则                                            │
│  • 可能导致数据竞争                                              │
│                                                                 │
│  结论：ViewMutRepr 不能是 Sync                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 广播结果不可变迭代的原因

```rust
// Broadcast results use ViewRepr (read-only view), no mutable iterator provided

// Good - broadcast results are read-only
let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
let b = a.broadcast(&[3, 3]);  // broadcast result: ViewRepr
let sum: f64 = b.iter().sum();  // OK: read-only iteration

// Bad - compilation error: broadcast results cannot be mutably iterated
// let mut b_mut = a.broadcast(&[3, 3]);
// b_mut.iter_mut()  // Compile error! ViewRepr does not implement StorageMut
```

> **设计决策：** 广播结果使用 `ViewRepr`（只读视图），因为广播不拷贝数据，语义上仅为只读（参见 `15-broadcast.md §4`）。如果允许可变迭代，修改广播结果会意外修改原数据的多个位置，这既不符合广播语义，也容易引入 bug。

### 4.8 Good/Bad 对比示例

```rust
// Good - ViewMutRepr cross-thread movement (transfers exclusive access)
fn send_view_mut() {
    let mut owned = Owned::from_vec(vec![1.0, 2.0, 3.0]);
    let view_mut = owned.view_mut();

    // Correct: move view_mut to a new thread
    std::thread::spawn(move || {
        // view_mut has exclusive access in this thread
        let data = view_mut.as_mut_slice();
        data[0] = 10.0;
    });
}

// Bad - attempting to share ViewMutRepr (compilation error)
fn cannot_share_view_mut() {
    let mut owned = Owned::from_vec(vec![1.0, 2.0, 3.0]);
    let view_mut = owned.view_mut();
    let view_ref = &view_mut;

    // Compilation error: &ViewMutRepr is not Send
    // because ViewMutRepr is not Sync
    // std::thread::spawn(move || {
    //     println!("{:?}", view_ref);
    // });
}

// Good - ArcRepr cross-thread sharing
fn share_arc_tensor() {
    let arc = ArcTensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let arc_clone = arc.clone();  // strong_count = 2

    std::thread::spawn(move || {
        // arc_clone is safely read in this thread
        assert_eq!(arc_clone[0], 1.0);
    });

    // arc is still safe in the main thread
    assert_eq!(arc[1], 2.0);
}

// Good - parallel iteration element constraint
fn parallel_iteration(tensor: &Tensor2<f64>) {
    // f64: Send + Sync, satisfies rayon constraint
    let sum = tensor.par_iter().sum::<f64>();
}
```

---

## 5. 内部实现设计

### 5.1 Rust 所有权系统与线程安全的协同

```
┌─────────────────────────────────────────────────────────────────┐
│              Rust 所有权 → 线程安全 映射                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Rust 所有权规则              线程安全保证                       │
│  ──────────────────────      ──────────────────────              │
│                                                                 │
│  move 语义                   Owned/ArcRepr 可跨线程移动 (Send)   │
│  &T 共享引用                 ViewRepr 可跨线程共享 (Sync)        │
│  &mut T 独占引用             ViewMutRepr 只可移动 (Send only)    │
│  Arc 原子计数                ArcRepr 可安全共享 (Send + Sync)    │
│  生命周期 'a                 视图不会比源数据存活更久              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 并行操作安全约束

并行迭代的安全保证基于分块访问隔离（参见 `09-parallel.md §5`）：

```
┌─────────────────────────────────────────────────────────────────┐
│                并行迭代访问隔离示意                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  数组: [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]      │
│        └────线程0───┘└────线程1───┘└────线程2───┘└──线程3──┘    │
│                                                                 │
│  关键保证：                                                      │
│  • 每个元素最多被一个线程访问                                    │
│  • 线程间无共享写入                                              │
│  • 分块边界清晰，无重叠                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 SIMD 与并行组合安全

SIMD 操作与并行操作的组合安全性保证（参见 `08-simd.md §5`）：

```
┌─────────────────────────────────────────────────────────────────┐
│             SIMD + 并行 组合安全保证                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 并行分块保证不重叠访问                                       │
│  2. 每个线程内部的 SIMD 操作在独占数据区间上执行                  │
│  3. SIMD kernel 无内部可变性（无共享状态）                       │
│  4. pulp Arch 实例是 Copy 类型，可零成本跨线程复制               │
│  5. 无需额外同步机制                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 实现任务拆分

### Wave 1: Send/Sync 实现

- [ ] **T1**: Owned<A> 的 Send/Sync 实现
  - 文件: `src/storage/owned.rs`
  - 内容: `unsafe impl<A: Send> Send for Owned<A> {}`、`unsafe impl<A: Sync> Sync for Owned<A> {}`、完整 SAFETY 注释
  - 测试: `test_owned_send_sync`、`test_owned_negative_rc`
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: ViewRepr<&'a A> 的 Send/Sync 实现
  - 文件: `src/storage/view.rs`
  - 内容: `unsafe impl<'a, A: Sync> Send for ViewRepr<&'a A> {}`、`unsafe impl<'a, A: Sync> Sync for ViewRepr<&'a A> {}`、完整 SAFETY 注释
  - 测试: `test_view_send_sync`、`test_view_cross_thread`
  - 前置: 无
  - 预计: 10 min

- [ ] **T3**: ViewMutRepr<&'a mut A> 的 Send 实现
  - 文件: `src/storage/view_mut.rs`
  - 内容: `unsafe impl<'a, A: Send> Send for ViewMutRepr<&'a mut A> {}`、不实现 Sync 的注释、完整 SAFETY 注释
  - 测试: `test_view_mut_send`、`test_view_mut_not_sync`
  - 前置: 无
  - 预计: 10 min

- [ ] **T4**: ArcRepr<A> 的 Send/Sync 实现
  - 文件: `src/storage/arc.rs`
  - 内容: `unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}`、`unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}`、完整 SAFETY 注释
  - 测试: `test_arc_send_sync`、`test_arc_concurrent_read`
  - 前置: 无
  - 预计: 10 min

### Wave 2: 并行安全验证

- [ ] **T5**: 并行迭代分块安全验证
  - 文件: `src/parallel/par_iter.rs`
  - 内容: 分块完整性/不重叠/边界安全的测试
  - 测试: `test_chunks_cover_all`、`test_chunks_no_overlap`
  - 前置: T1-T4
  - 预计: 10 min

- [ ] **T6**: 线程安全集成测试
  - 文件: `tests/thread_safety.rs`
  - 内容: 多线程传递测试、并发访问测试
  - 测试: `test_owned_cross_thread`、`test_arc_concurrent_access`
  - 前置: T1-T5
  - 预计: 10 min

- [ ] **T7**: 文档和 Safety 注释审查
  - 文件: `src/storage/mod.rs`
  - 内容: 模块级线程安全文档、Send/Sync 矩阵、更新 CHANGELOG
  - 测试: `cargo doc` 通过
  - 前置: T1-T4
  - 预计: 10 min

```
Wave 1: [T1] [T2] [T3] [T4]
           │     │     │     │
           └─────┴─────┴─────┘
                     │
                     ▼
Wave 2:            [T5]
                     │
              ┌──────┴──────┐
              ▼             ▼
            [T6]          [T7]
```

---

## 7. 测试计划

### 7.1 测试分类表

| 类型 | 位置 | 目的 |
|------|------|------|
| 编译期检查 | `static_assertions` 或手动函数 | 验证 Send/Sync 约束传播 |
| 跨线程测试 | `#[test]` with `std::thread` | 验证跨线程使用安全性 |
| 并发访问测试 | `tests/thread_safety.rs` | 多线程并发场景验证 |
| 模型检测 | `loom` 或 `miri` | 数据竞争和内存安全检测 |

> **注意：** `static_assertions` crate 用于编译期断言 Send/Sync 实现。需在 `Cargo.toml` 的 `[dev-dependencies]` 中添加：
> ```toml
> static_assertions = "1.1"
> ```

> **dev-dependency 说明**：编译期负向测试（如 `assert_not_impl_any!`）需要 `static_assertions` crate。需在 `Cargo.toml` 的 `[dev-dependencies]` 中添加 `static_assertions = "1.1"`。

### 7.2 单元测试清单

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_owned_send_sync` | `Owned<f64>: Send + Sync` 编译通过 | 高 |
| `test_owned_negative_rc` | `Owned<Rc<i32>>` 不满足 Send | 高 |
| `test_view_send_sync` | `ViewRepr<&f64>: Send + Sync` 编译通过 | 高 |
| `test_view_cross_thread` | 视图跨线程传递正确 | 高 |
| `test_view_mut_send` | `ViewMutRepr<&mut f64>: Send` 编译通过 | 高 |
| `test_view_mut_not_sync` | `ViewMutRepr` 不是 Sync（编译失败检查） | 高 |
| `test_arc_send_sync` | `ArcRepr<f64>: Send + Sync` 编译通过 | 高 |
| `test_arc_concurrent_read` | 多线程并发读取 ArcRepr | 中 |
| `test_chunks_cover_all` | 分块覆盖所有元素 | 中 |
| `test_chunks_no_overlap` | 分块不重叠 | 中 |
| `test_owned_cross_thread` | Owned 跨线程移动 | 中 |
| `test_arc_make_mut_threading` | make_mut 原子性验证 | 低 |

### 7.3 编译期静态检查模板

使用 `static_assertions` crate 的 `assert_impl_all!` 和 `assert_not_impl_any!` 宏进行编译期验证：

```rust
use static_assertions::{assert_impl_all, assert_not_impl_any};

// Positive: verify traits are implemented
assert_impl_all!(Owned<f64>: Send, Sync);
assert_impl_all!(ViewRepr<&f64>: Send, Sync);
assert_impl_all!(ViewMutRepr<&mut f64>: Send);
assert_impl_all!(ArcRepr<f64>: Send, Sync);

// Negative: verify traits are NOT implemented
assert_not_impl_any!(ViewMutRepr<&mut f64>: Sync);
assert_not_impl_any!(Owned<Rc<i32>>: Send);

#[test]
fn owned_send_sync() {
    // Compile-time check via static_assertions above
}

#[test]
fn view_send_sync() {
    assert_impl_all!(ViewRepr<&f64>: Send, Sync);
}

#[test]
fn view_mut_send_only() {
    assert_impl_all!(ViewMutRepr<&mut f64>: Send);
    assert_not_impl_any!(ViewMutRepr<&mut f64>: Sync);
}

#[test]
fn arc_send_sync() {
    assert_impl_all!(ArcRepr<f64>: Send, Sync);
}
```

### 7.4 多线程传递测试

```rust
#[test]
fn test_owned_cross_thread() {
    let tensor = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let handle = std::thread::spawn(move || {
        // tensor is available in the new thread
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor.len(), 3);
    });
    handle.join().unwrap();
}

#[test]
fn test_arc_concurrent_access() {
    let arc = ArcTensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let mut handles = vec![];

    for _ in 0..4 {
        let arc_clone = arc.clone();
        handles.push(std::thread::spawn(move || {
            // All threads read concurrently
            let sum: f64 = arc_clone.iter().sum();
            sum
        }));
    }

    let sums: Vec<f64> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    // All threads should get the same result
    assert!(sums.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10));
}
```

### 7.5 loom/miri 测试策略

| 工具 | 用途 | 运行方式 |
|------|------|
| `loom` | 并发模型检测，验证 ArcRepr make_mut 原子性 | `cargo test --features loom` |
| `miri` | 内存安全检测，验证 unsafe impl 的正确性 | `cargo miri test` |
| `ThreadSanitizer` | 数据竞争检测 | `RUSTFLAGS="-Z sanitizer=thread" cargo test` |

---

## 8. 与其他模块的交互

### 8.1 与 storage 模块

```
┌─────────────────────────────────────────────────────────────────┐
│                    storage 模块的线程安全接口                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  trait RawStorage {                                             │
│      type Elem;  // 无 Send/Sync 约束                            │
│  }                                                              │
│                                                                 │
│  trait Storage: RawStorage {                                    │
│      type Elem;                                                 │
│      fn as_ptr(&self) -> *const Self::Elem;                     │
│  }                                                              │
│                                                                 │
│  trait StorageMut: Storage {                                    │
│      fn as_mut_ptr(&mut self) -> *mut Self::Elem;               │
│  }                                                              │
│                                                                 │
│  各实现的 Send/Sync 由具体类型（Owned/ViewRepr/...）决定            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 与 parallel 模块

```
┌─────────────────────────────────────────────────────────────────┐
│                    parallel 模块的线程安全要求                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  impl<'a, S, D, A> IntoParallelRefIterator for TensorBase<S, D> │
│  where                                                          │
│      S: Storage<Elem = A>,                                      │
│      D: Dimension,                                              │
│      A: Element + Send + Sync,  // 关键约束                      │
│  {                                                              │
│      type Iter = ParElements<'a, A, D>;                         │
│  }                                                              │
│                                                                 │
│  分块安全保证：                                                   │
│  • compute_safe_chunks() 保证不重叠                              │
│  • 生命周期确保视图有效                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 与 rayon 的集成

rayon 的 `ParallelIterator` 要求 `Item: Send`（参见 `09-parallel.md §8`）：

| 存储模式 | `par_iter()` | `par_iter_mut()` | 约束 |
|----------|:----------:|:--------------:|------|
| `Tensor<A, D>` (Owned) | ✅ | ✅ | `A: Send + Sync` |
| `TensorView<'a, A, D>` | ✅ | ❌ | `A: Sync` |
| `TensorViewMut<'a, A, D>` | ✅ | ✅ | `A: Send + Sync` |
| `ArcTensor<A, D>` | ✅ | ❌ (需 make_mut) | `A: Send + Sync` |

#### TensorViewMut 的 par_iter_mut() 安全性论证

`TensorViewMut` 支持 `par_iter_mut()` 的安全性基于以下保证：

1. **独占语义**：`TensorViewMut` 持有 `&mut [A]`，Rust 借用规则保证在任意时刻只有唯一的可变访问者
2. **分块不重叠**：`par_iter_mut()` 将数据按块分割，每个线程获得独占的 `&mut [A]` 切片，块之间无重叠
3. **Send 约束**：`A: Send` 确保元素类型可跨线程传递
4. **Sync 约束**：`A: Sync` 确保在 `par_iter()` 只读路径中，多线程可安全共享 `&A`
5. **生命周期**：`TensorViewMut` 的 `'a` 生命周期确保视图不会比源数据存活更久

```rust
// Safety argument for par_iter_mut() on TensorViewMut:
//
// TensorViewMut<'a, A, D> holds &mut [A] exclusively.
// par_iter_mut() splits the slice into non-overlapping chunks,
// each chunk is sent to a different thread as &mut [A].
// Since chunks don't overlap, no two threads can access the same element,
// satisfying the aliasing rule. A: Send + Sync ensures element-level safety.
```

---

## 9. 设计决策记录（ADR）

### 决策 1：显式 unsafe impl 而非依赖自动推导

| 属性 | 值 |
|------|-----|
| 决策 | 使用显式 `unsafe impl Send/Sync`，而非依赖编译器自动推导 |
| 理由 | 文档化意图，每个 impl 附带完整 SAFETY 注释（参见 `00-coding-standards.md §5`），便于审查和维护 |
| 替代方案 | 依赖自动推导 — 放弃，缺少安全性论证文档，修改内部字段时可能意外改变线程安全语义 |

### 决策 2：ViewMutRepr 不实现 Sync

| 属性 | 值 |
|------|-----|
| 决策 | `ViewMutRepr<&'a mut A>` 仅实现 `Send`，不实现 `Sync` |
| 理由 | 独占借用语义（`&mut T`）不可共享。如果 Sync，则 `&ViewMutRepr` 可跨线程移动，导致多线程同时持有 `&mut [A]`，违反 Rust 别名规则 |
| 替代方案 | 通过 Mutex 包装实现 Sync — 放弃，引入运行时锁，违反"编译期保证"原则 |

### 决策 3：ArcRepr 要求 A: Send + Sync

| 属性 | 值 |
|------|-----|
| 决策 | `ArcRepr<A>` 的 Send/Sync 要求 `A: Send + Sync` |
| 理由 | `Arc<T>` 的线程安全要求 `T: Send + Sync`，因为多个线程可同时持有 `&T`。ArcRepr 内部使用 `Arc<Vec<A>>`，因此继承此约束 |
| 替代方案 | 仅要求 `A: Send` — 放弃，允许多线程同时通过 `&ArcRepr` 读取 `A`，如果 `A` 不是 `Sync`，存在数据竞争风险 |

### 决策 4：编译期保证优于运行时锁

| 属性 | 值 |
|------|-----|
| 决策 | 所有线程安全通过 Rust 类型系统在编译期保证，不使用运行时锁 |
| 理由 | 零运行时开销、编译期排除数据竞争、与 Rust 安全理念一致 |
| 替代方案 | 运行时 Mutex/RwLock — 放弃，引入锁开销和死锁风险 |

---

## 10. no_std 兼容性

线程安全的 Send/Sync 标记 trait 位于 `core::marker`，可在 `no_std` 下使用。`ArcRepr` 使用 `alloc::sync::Arc`，在 `no_std + alloc` 下可用。并行功能依赖 `std`。

```rust
// Send/Sync traits are in core::marker — no_std available
use core::marker::{Send, Sync};

// ArcRepr uses alloc::sync::Arc — available in no_std + alloc
// Parallel/rayon uses std::thread — requires std
```

| 组件 | no_std 支持 | 说明 |
|------|:----------:|------|
| `Owned<A>` Send/Sync | ✅ | 仅 `unsafe impl`，无运行时依赖 |
| `ViewRepr<&'a A>` Send/Sync | ✅ | 仅 `unsafe impl`，无运行时依赖 |
| `ViewMutRepr<&'a mut A>` Send | ✅ | 仅 `unsafe impl`，无运行时依赖 |
| `ArcRepr<A>` Send/Sync | ✅ | 使用 `alloc::sync::Arc`，`no_std + alloc` 可用 |
| 并行迭代（rayon） | ❌ | rayon 依赖 `std` 线程原语，参见 `09-parallel.md §11` |
| 嵌套并行防护（`thread_local!`） | ❌ | `std::cell::Cell` + `thread_local!` 需要 `std` |
| SIMD Arch 缓存线程安全初始化 | ❌ | 依赖 `std::sync::OnceLock` 或等效机制 |

**no_std 下 SIMD Arch 缓存的替代方案：**

在 `no_std` 环境中，`std::sync::OnceLock` 不可用于缓存 SIMD 架构检测结果。可选方案：

- 对于支持 `critical_section` 的 `no_std` 目标：使用 `critical_section::Mutex<Option<Arch>>` 实现线程安全缓存。
- 对于禁用中断的裸机目标：在禁用中断期间检测一次并缓存结果。
- 对于无 OS 的裸机目标：考虑每次直接调用 `Arch::new()`（对于偶尔使用，开销可接受）。

条件编译处理：

```rust
// Owned/ViewRepr/ViewMutRepr: unsafe impl Send/Sync
// Uses only core::marker::{Send, Sync} — pure no_std

// ArcRepr: available in no_std + alloc (uses alloc::sync::Arc)
// No feature gate needed — alloc::sync::Arc is always available when alloc is.
unsafe impl<A: Send + Sync> Send for ArcRepr<A> {}

// Parallel/rayon: gated behind "parallel" feature (implies "std")
#[cfg(feature = "parallel")]
mod parallel { ... }
```

---

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
