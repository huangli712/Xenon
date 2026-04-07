# 线程安全设计

> 文档编号: 25 | 模块: 横切关注点 | 阶段: Phase 2
> 前置文档: `05-storage.md`, `07-tensor.md`
> 需求参考: 需求说明书 §10

---

## 1. 模块定位

### 1.1 概述

线程安全是 Xenon 的横切关注点，贯穿所有存储模式和计算后端。本文档定义各存储模式的 `Send`/`Sync` 实现规则，确保 Xenon 张量可在多线程环境下安全使用。

### 1.2 职责边界

| 职责 | 包含 | 不包含 |
|------|------|--------|
| Send/Sync 实现 | 各存储模式的 Send/Sync trait 实现 | 锁机制 (Mutex/RwLock) |
| 正确性保证 | unsafe impl 的安全性论证和证明 | 通道 (mpsc/crossbeam) |
| 并行安全约束 | SIMD 与并行组合的安全约束 | 异步运行时 (tokio/async-std) |
| 编译期保证 | 通过 Rust 类型系统在编译期排除数据竞争 | 运行时锁或同步原语 |
| 广播安全 | 广播结果不可可变迭代的约束 | — |

### 1.3 设计原则

| 原则 | 体现 |
|------|------|
| 编译期保证 | 无运行时锁，所有线程安全由 Rust 类型系统在编译期保证 |
| 最小约束 | 每种存储模式仅实现其语义允许的最小 Send/Sync 约束 |
| unsafe 安全论证 | 每个 unsafe impl 附带完整 SAFETY 注释 |
| 所有权协同 | 充分利用 Rust 所有权系统与线程安全的天然协同 |

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
├── std::sync::Arc       # ArcRepr 使用的原子引用计数
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
| `rayon::iter` | `ParallelIterator` (要求 `Item: Send`) |

### 3.3 依赖方向声明

> **依赖方向：线程安全是横切关注点。** 各存储模块自行声明 Send/Sync，parallel 模块消费这些约束。无循环依赖。

---

## 4. 公共 API 设计

### 4.1 Send/Sync 实现规则表

完整矩阵：

| 存储模式 | Send | Sync | 条件 | 理由 |
|----------|:----:|:----:|------|------|
| `Owned<A>` | ✅ | ✅ | `A: Send + Sync` | Vec 内部线程安全，元素约束传播 |
| `ViewRepr<&'a A>` | ✅ | ✅ | `A: Sync` | 共享引用跨线程共享要求 `T: Sync` |
| `ViewMutRepr<&'a mut A>` | ✅ | ✗ | `A: Send` | 独占借用不可共享，但可转移 |
| `ArcRepr<A>` | ✅ | ✅ | `A: Send + Sync` | Arc 原子计数，要求元素线程安全 |

### 4.2 Owned<A> 的 Send/Sync

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
///
/// **反例：如果 `A` 不是 `Send`**
///
/// 假设 `A = Rc<i32>`（不是 `Send`），将 `Owned<Rc<i32>>`
/// 移动到另一个线程会导致两个线程可能同时访问同一个 `Rc`，
/// 而 `Rc` 的引用计数不是原子的，会导致数据竞争。
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

### 4.3 ViewRepr<&'a A> 的 Send/Sync

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
unsafe impl<'a, A: Sync> Send for ViewRepr<&'a A> {}

/// # Safety
///
/// `ViewRepr<&'a A>` 实现了 `Sync`，因为：
///
/// 1. **共享访问安全**：多个线程可以同时持有 `&ViewRepr<&'a A>`，
///    这等价于多个线程持有 `&&A`（指向共享引用的共享引用）。
///
/// 2. **只读访问**：通过共享引用访问视图只能进行只读操作，
///    不会修改视图本身或底层数据。
///
/// 3. **无状态**：视图的 `ptr` 和 `len` 在创建后不可变，
///    多线程读取这些字段是安全的。
unsafe impl<'a, A: Sync> Sync for ViewRepr<&'a A> {}
```

### 4.4 ViewMutRepr<&'a mut A> 的 Send（不实现 Sync）

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
//
// 原因：&mut T 不能共享，Rust 借用规则禁止。
//
// 如果 ViewMutRepr 是 Sync，那么 &ViewMutRepr 可以跨线程共享，
// 这意味着多个线程可以同时获取 &mut [A]，导致别名和潜在的数据竞争。
//
// Rust 的负面 trait impl（!Sync）在 stable Rust 中不能显式写，
// 但由于 ViewMutRepr 包含 *mut A（不是 Sync），编译器不会
// 自动实现 Sync，这正是我们想要的行为。
```

### 4.5 ArcRepr<A> 的 Send/Sync

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

### 4.6 ViewMutRepr 不实现 Sync 的证明

```
┌─────────────────────────────────────────────────────────────────┐
│              ViewMutRepr 不是 Sync 的证明                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  定义：T: Sync ⟺ &T: Send                                      │
│                                                                 │
│  假设 ViewMutRepr<&mut A> 是 Sync：                             │
│  • 则 &ViewMutRepr 可以跨线程移动                                │
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
│  • 违反 Rust 借用规则                                           │
│  • 可能导致数据竞争                                              │
│                                                                 │
│  结论：ViewMutRepr 不能是 Sync                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 广播结果不可可变迭代的原因

```rust
// 广播结果使用 ViewRepr（只读视图），不提供可变迭代器

// Good - 广播结果只能只读访问
let a = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
let b = a.broadcast(&[3, 3]);  // broadcast result: ViewRepr
let sum: f64 = b.iter().sum();  // OK: read-only iteration

// Bad - 编译错误：广播结果不能可变迭代
// let mut b_mut = a.broadcast(&[3, 3]);
// b_mut.iter_mut()  // 编译失败！ViewRepr 不实现 StorageMut
```

> **设计决策：** 广播结果使用 `ViewRepr`（只读视图），因为广播不拷贝数据，语义上仅为只读。如果允许可变迭代，修改广播结果会意外修改原数据的多个位置，这既不符合广播语义，也容易引入 bug。

### 4.8 Good/Bad 对比示例

```rust
// Good - ViewMutRepr 跨线程移动（转移独占访问权）
fn send_view_mut() {
    let mut owned = Owned::from_vec(vec![1.0, 2.0, 3.0]);
    let view_mut = owned.view_mut();

    // 正确：移动 view_mut 到新线程
    std::thread::spawn(move || {
        // view_mut 在此线程独占访问
        let data = view_mut.as_mut_slice();
        data[0] = 10.0;
    });
}

// Bad - 尝试共享 ViewMutRepr（编译失败）
fn cannot_share_view_mut() {
    let mut owned = Owned::from_vec(vec![1.0, 2.0, 3.0]);
    let view_mut = owned.view_mut();
    let view_ref = &view_mut;

    // 编译错误：&ViewMutRepr 不是 Send
    // 因为 ViewMutRepr 不是 Sync
    // std::thread::spawn(move || {
    //     println!("{:?}", view_ref);
    // });
}

// Good - ArcRepr 跨线程共享
fn share_arc_tensor() {
    let arc = ArcTensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let arc_clone = arc.clone();  // strong_count = 2

    std::thread::spawn(move || {
        // arc_clone 在此线程安全读取
        assert_eq!(arc_clone[0], 1.0);
    });

    // arc 在主线程仍安全
    assert_eq!(arc[1], 2.0);
}

// Good - 并行迭代元素约束
fn parallel_iteration(tensor: &Tensor2<f64>) {
    // f64: Send + Sync，满足 rayon 约束
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

```rust
// 编译期 Send/Sync 验证工具函数
fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_not_sync<T>() where T: ?Sized {}

#[test]
fn owned_send_sync() {
    assert_send::<Owned<f64>>();
    assert_sync::<Owned<f64>>();
    assert_send::<Owned<i32>>();
    assert_sync::<Owned<i32>>();
}

#[test]
fn view_send_sync() {
    assert_send::<ViewRepr<&f64>>();
    assert_sync::<ViewRepr<&f64>>();
}

#[test]
fn view_mut_send_only() {
    assert_send::<ViewMutRepr<&mut f64>>();
    // assert_sync::<ViewMutRepr<&mut f64>>(); // 编译失败，符合预期
}

#[test]
fn arc_send_sync() {
    assert_send::<ArcRepr<f64>>();
    assert_sync::<ArcRepr<f64>>();
}
```

### 7.4 多线程传递测试

```rust
#[test]
fn test_owned_cross_thread() {
    let tensor = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let handle = std::thread::spawn(move || {
        // tensor 在新线程中可用
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
|------|------|----------|
| `loom` | 并发模型检测，验证 ArcRepr make_mut 原子性 | `cargo test --features loom` |
| `miri` | 内存安全检测，验证 unsafe impl 的正确性 | `cargo miri test` |
| `ThreadSanitizer` | 数据竞争检测 | `RUSTFLAGS="-Z sanitizer=thread" cargo test` |

---

## 8. 与其他模块的交互

### 8.1 与 storage 模块

```
┌─────────────────────────────────────────────────────────────────┐
│                    storage 模块的线程安全接口                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  trait RawStorage {                                             │
│      type Elem;  // 无 Send/Sync 约束                           │
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
│  各实现的 Send/Sync 由具体类型（Owned/ViewRepr/...）决定         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 与 parallel 模块

```
┌─────────────────────────────────────────────────────────────────┐
│                    parallel 模块的线程安全要求                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  impl<'a, S, D, A> IntoParallelRefIterator for TensorBase<S, D> │
│  where                                                          │
│      S: Storage<Elem = A>,                                      │
│      D: Dimension,                                              │
│      A: Element + Send + Sync,  // 关键约束                     │
│  {                                                              │
│      type Iter = ParElements<'a, A, D>;                         │
│  }                                                              │
│                                                                 │
│  分块安全保证：                                                  │
│  • compute_safe_chunks() 保证不重叠                             │
│  • 生命周期确保视图有效                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 与 rayon 的集成

rayon 的 `ParallelIterator` 要求 `Item: Send`：

| 存储模式 | `par_iter()` | `par_iter_mut()` | 约束 |
|----------|:----------:|:--------------:|------|
| `Tensor<A, D>` (Owned) | ✅ | ✅ | `A: Send + Sync` |
| `TensorView<'a, A, D>` | ✅ | ❌ | `A: Sync` |
| `TensorViewMut<'a, A, D>` | ✅ | ✅ | `A: Send + Sync` |
| `ArcTensor<A, D>` | ✅ | ❌ (需 make_mut) | `A: Send + Sync` |

---

## 9. 设计决策记录（ADR）

### 决策 1：显式 unsafe impl 而非依赖自动推导

| 属性 | 值 |
|------|-----|
| 决策 | 使用显式 `unsafe impl Send/Sync`，而非依赖编译器自动推导 |
| 理由 | 文档化意图，每个 impl 附带完整 SAFETY 注释，便于审查和维护 |
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

## 版本历史

| 版本 | 日期 |
|------|------|
| 1.0.0 | 2026-04-07 |

---

*本文档由 Xenon 维护。如有问题请提交 Issue 或 PR。*
