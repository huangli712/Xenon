# 线程安全规范

> 文档编号: 25 | 适用范围: `src/storage/`, `src/parallel/`, `src/simd/`, `src/ffi/` | 阶段: Phase 2
> 前置文档: `05-storage.md`, `07-tensor.md`
> 需求参考: 需求说明书 §10
> 范围声明: 范围内

---

## 1. 主题定位与适用范围

> **文档定位说明：** 本文档为横切规范文档，关注线程安全性约束的跨模块适用。部分模块化章节（如文件位置、公共 API）仅用于补充说明，主线为约束定义与验证。

线程安全是 Xenon 的横切关注点，贯穿所有存储模式和计算后端。本文档定义各存储模式（参见 `05-storage.md §5`）的 `Send`/`Sync` 实现规则，确保 Xenon 张量（参见 `07-tensor.md §5`）可在多线程环境下安全使用。

> **范围注记：** workspace 的线程安全属性参见 `24-workspace.md`；本文不将 workspace 纳入 `require.md §10` 的存储模式线程安全矩阵。

### 1.1 职责边界

| 职责           | 包含                                   | 不包含                       |
| -------------- | -------------------------------------- | ---------------------------- |
| Send/Sync 实现 | 各存储模式的 Send/Sync trait 实现      | 锁机制 (Mutex/RwLock)        |
| 正确性保证     | unsafe impl 的安全性论证和证明         | 通道 (mpsc/crossbeam)        |
| 并行安全约束   | SIMD 与并行组合的安全约束              | 异步运行时 (tokio/async-std) |
| 编译期保证     | 通过 Rust 类型系统在编译期排除数据竞争 | 运行时锁或同步原语           |
| 广播安全       | 广播结果不可变迭代的约束               | —                            |

### 1.2 设计原则

| 原则            | 体现                                                 |
| --------------- | ---------------------------------------------------- |
| 编译期保证      | 无运行时锁，所有线程安全由 Rust 类型系统在编译期保证 |
| 最小约束        | 每种存储模式仅实现其语义允许的最小 Send/Sync 约束    |
| unsafe 安全论证 | 每个 unsafe impl 附带完整 SAFETY 注释                |
| 所有权协同      | 充分利用 Rust 所有权系统与线程安全的天然协同         |

### 1.3 在架构中的位置

```
Dependency layers:
L0: error, private
L1: dimension, element, complex
L2: layout (depends on dimension)
L3: storage (independent of layout; owned by tensor and consumes layout results)
L4: tensor (depends on storage, dimension)
L5: math/, iter/, index/, shape/, broadcast/, construct/, ffi/, convert/, format/

Cross-cutting concern:
┌─────────────────────────────────────────────────────────────────┐
│  Thread safety (Send/Sync)  <- current document (spans modules across L3-L5) │
│  - Spans storage (L3), tensor (L4), and parallel/simd (L5)         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 线程安全在 Xenon 中的重要性

| 场景         | 线程安全需求                       |
| ------------ | ---------------------------------- |
| 并行迭代     | 多线程同时访问不同元素区间         |
| 跨线程共享   | 通过 ArcRepr 在线程间传递只读数据  |
| 写时复制     | ArcRepr 内部唯一化 / 必要时复制后恢复可写性的实现约束 |
| 数据竞争预防 | 确保 ViewMutRepr 独占访问不被共享  |
| rayon 集成   | ParallelIterator 要求 Send 约束    |

## 2. 需求映射与范围约束

| 类型     | 内容                                                             |
| -------- | ---------------------------------------------------------------- |
| 需求映射 | `require.md §10`, `§16`, `§17`, `§21.2`, `§26`, `§28.2`, `§28.5` |
| 范围内   | `Send`/`Sync` 规则、并行安全边界、广播只读约束、跨线程访问证明   |
| 范围外   | 异步运行时、锁封装策略、通用并发原语抽象                         |
| 非目标   | 通过线程安全规范引入新的同步原语、运行时锁或额外第三方并发依赖   |

---

## 3. 文件位置

线程安全实现散布于各存储模块中，不单独创建文件：

> **命名说明：** 以下 API 名称为当前候选方案，最终命名以实现为准。

```
src/
├── storage/
│   ├── mod.rs          # Module-level thread-safety docs
│   ├── owned.rs        # Send/Sync impls for Owned<A>
│   ├── view.rs         # Send/Sync impls for ViewRepr<'a, A>
│   ├── view_mut.rs     # Send/Sync impls for ViewMutRepr<'a, A>
│   └── arc.rs          # Send/Sync impls for ArcRepr<A>
├── parallel/
│   ├── mod.rs          # Nested parallelism guard candidates
│   └── par_iter.rs     # Thread-safety constraints for ParElements
└── simd/
    └── mod.rs          # Thread-safe initialization of the Arch cache
```

---

## 4. 依赖关系

### 4.1 依赖图

```
src/storage/
├── core::marker         # PhantomData, Send, Sync
└── alloc::sync::Arc     # Atomic reference counting used by ArcRepr

src/parallel/
├── std::sync::atomic    # AtomicUsize (thresholds)
├── std::cell::Cell      # thread_local (nested parallel detection)
└── rayon                # ParallelIterator (Send constraint)
```

### 4.2 依赖精确到类型级

| 来源模块       | 使用的类型/trait                                                   |
| -------------- | ------------------------------------------------------------------ |
| `core::marker` | `PhantomData<A>`, `Send`, `Sync`                                   |
| `std::sync`    | `Arc<Vec<A>>`, `AtomicUsize`                                       |
| `std::cell`    | `Cell<usize>`, `Cell<Option<usize>>`（仅 `parallel` 模块）         |
| `rayon::iter`  | `ParallelIterator` (要求 `Item: Send`，参见 `09-parallel.md §5.3`) |

### 4.3 依赖方向声明

> **依赖方向：线程安全是横切关注点。** 各存储模块自行声明 Send/Sync（参见 `05-storage.md §5`），parallel 模块消费这些约束（参见 `09-parallel.md §5.2`）。无循环依赖。

### 4.4 依赖合法性与新增依赖说明

| 项目           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| 新增第三方依赖 | 无新增依赖；仅复用既有 `rayon` 可选依赖与标准库同步原语      |
| 合法性结论     | 符合最小依赖限制                                             |
| 替代方案       | 不适用；线程安全约束优先通过 Rust 类型系统与既有模块实现表达 |

---

## 5. 公共 API 设计

> **权威来源对齐：** 本文档的 Send/Sync 定义以 `05-storage.md §5.3`、`00-coding.md §3.4` 与 `require.md` 为基准；若与其他设计文档冲突，以 `require.md` 为规范基线解决，并同步修正相关文档。

> **注意：** 本文档与 `05-storage.md §5.7` 采用同一组 Send/Sync 规则；若后续出现不一致，应按 `require.md` 统一校正并同步回写相关设计文档。

### 5.1 Send/Sync 实现规则表

完整矩阵：

| 存储模式             | Send | Sync | 条件                             | 理由                                               |
| -------------------- | :--: | :--: | -------------------------------- | -------------------------------------------------- |
| `Owned<A>`           |  ✅  |  ✅  | Send: `A: Send`; Sync: `A: Sync` | 独占拥有型存储分别按移动安全和共享安全传播元素约束 |
| `ViewRepr<'a, A>`    |  ✅  |  ✅  | `A: Sync`                        | 共享视图跨线程共享要求元素可安全共享               |
| `ViewMutRepr<'a, A>` |  ✅  |  ✗   | `A: Send`                        | 独占可写视图可转移但不可共享                       |
| `ArcRepr<A>`         |  ✅  |  ✅  | `A: Send + Sync`                 | Arc 原子计数，读共享安全；写路径仅能在内部唯一化 / 必要时复制后恢复可写性 |

> **补充说明：** `ViewRepr` 仅持有共享引用（`&A`），跨线程传递共享引用只要求 `A: Sync`（允许多线程共享读取），不要求 `A: Send`（所有权转移）。这是 Rust 标准库 `&T: Send + Sync where T: Sync` 的直接推论。

各存储模式的完整 API 定义参见 `05-storage.md §5`。

### 5.2 TensorBase<S, D> 自动推导规则

`TensorBase<S, D>` 的 `Send`/`Sync` 由 Rust 自动推导，规则如下：

| 存储模式 S                             | TensorBase\<S, D\> 的 Send | TensorBase\<S, D\> 的 Sync |
| -------------------------------------- | -------------------------- | -------------------------- |
| `Owned<A>` where `A: Send` / `A: Sync` | ✅ Send                    | ✅ Sync                    |
| `ViewRepr<'a, A>` where A: Sync        | ✅ Send                    | ✅ Sync                    |
| `ViewMutRepr<'a, A>` where A: Send     | ✅ Send                    | ❌ (exclusive borrow)      |
| `ArcRepr<A>` where A: Send + Sync      | ✅ Send                    | ✅ Sync                    |

> **说明**: `D: Dimension` 要求 `Dimension: Send + Sync`；所有 Dimension 类型（`Ix0`-`Ix6`, `IxDyn`）内部仅包含 Copy 类型的值数组或 `Vec<usize>`，因此自动满足 `Send + Sync`。

### 5.2.1 安全违规分类表

| 安全违规类型           | 检测层级                     | 处理方式            |
| ---------------------- | ---------------------------- | ------------------- |
| 存储模式不支持可写操作 | 类型层（编译期）             | 通过 trait 约束拒绝 |
| 广播结果尝试可变访问   | 类型层（编译期）             | 通过返回类型拒绝    |
| 并行中二次并行         | 运行时（嵌套并行防护机制）   | 自动回退串行路径    |
| 整数溢出               | 运行时（checked arithmetic） | panic（不可恢复）   |

### 5.3 Owned<A> 的 Send/Sync

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

### 5.4 ViewRepr<'a, A> 的 Send/Sync

```rust
// src/storage/view.rs

/// # Safety
///
/// `ViewRepr<'a, A>` implements `Send` because:
///
/// 1. **Ownership transfer semantics**: Moving `ViewRepr` does not move the underlying data,
///    it only transfers the view's metadata (pointer + length) to the new thread.
///
/// 2. **Shared reference constraint**: the view semantically exposes shared access, which is
///    safe across threads if and only if `A: Sync`.
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
unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}

/// # Safety
///
/// `ViewRepr<'a, A>` implements `Sync` because:
///
/// 1. **Shared access safety**: Multiple threads can hold `&ViewRepr<'a, A>` simultaneously,
///    and only shared access to the underlying elements is exposed.
///
/// 2. **Read-only access**: Accessing the view through a shared reference only permits read-only operations,
///    without modifying the view itself or the underlying data.
///
/// 3. **Stateless**: The view's `ptr` and `len` are immutable after creation,
///    so reading these fields from multiple threads is safe.
unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}
```

### 5.5 ViewMutRepr<'a, A> 的 Send（不实现 Sync）

```rust
// src/storage/view_mut.rs

/// # Safety
///
/// `ViewMutRepr<'a, A>` implements `Send` because:
///
/// 1. **Exclusive ownership transfer**: Moving `ViewMutRepr` transfers exclusive access to the new thread.
///    Rust's borrow checker guarantees the original thread holds no more references.
///
/// 2. **No aliasing guarantee**: mutable view semantics are exclusive; at any given moment only one
///    `ViewMutRepr` can access the data. After cross-thread movement, the new thread becomes the sole accessor.
///
/// 3. **Element type constraint**: mutable element access can cross threads only if `A: Send`.
///
/// 4. **Lifetime invariance**: The `'a` lifetime ensures the view does not outlive the source data.
///
/// **Counter-example: if `A` is not `Send`**
///
/// Suppose `A = Rc<i32>` (not `Send`), moving `ViewMutRepr<'_, Rc<i32>>`
/// to another thread could cause two threads to access the same `Rc`
/// simultaneously, and `Rc`'s reference count is not atomic, leading to data races.
///
/// **Note: ViewMutRepr does not implement Clone**
///
/// `ViewMutRepr` deliberately does not implement `Clone`, because copying would create aliases,
/// violating exclusive semantics. This is a key guarantee for `ViewMutRepr` thread safety.
unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A> {}

// ViewMutRepr does not implement Sync
//
// Reason: &mut T cannot be shared; Rust's borrowing rules forbid it.
//
// ViewMutRepr deliberately models exclusive access via `&'a mut A` semantics.
// Its representation carries mutable provenance (`*mut A` plus mutable borrow marker),
// so shared references to ViewMutRepr must not become a back door to aliasing mutable access.
// Rust's negative trait impls (!Sync) cannot be written on stable, therefore the design
// relies on the underlying mutable-pointer / PhantomData shape to prevent Sync auto-derivation.
```

### 5.6 ArcRepr<A> 的 Send/Sync

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
///    Any later write path must first regain unique ownership (cloning when needed)
///    before mutable access is re-enabled.
///
/// 4. **COW exclusivity**: Safety relies on the exclusive access guaranteed by
///    Rust's `&mut` borrowing. Xenon's internal write path must first prove or
///    establish unique ownership of the backing allocation; if uniqueness cannot
///    be reused directly, it materializes a private copy before exposing mutable access.
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
///    Any internal transition back to writable state requires exclusive `&mut`
///    access and is not exposed as a separate public API guarantee.
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

### 5.7 ViewMutRepr 不实现 Sync 的证明

`ViewMutRepr<'_, A>` 包含 `&mut [A]` 字段。由于 `&mut T` 不实现 `Sync`（其别名与可变访问语义不能被跨线程共享；底层可变指针 `*mut T` 也不实现 `Sync`），`ViewMutRepr` 也不会自动实现 `Sync`。因此这里依赖 Rust 的 auto-trait 推导结果，而不是显式编写 `unsafe impl !Sync` 之类的额外声明。

### 5.8 广播结果不可变迭代的原因

```rust
// Broadcast results use ViewRepr (read-only view), no mutable iterator provided

// Good - broadcast results are read-only
let a = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
let b = a.view().broadcast_to([3, 3])?;  // broadcast result: TensorView
let sum: f64 = b.iter().sum();  // OK: read-only iteration

// Bad - compilation error: broadcast results cannot be mutably iterated
// let mut b_mut = a.view().broadcast_to([3, 3]).unwrap();
// b_mut.iter_mut()  // Compile error! ViewRepr does not implement StorageMut
```

> **设计决策：** 广播结果使用 `ViewRepr`（只读视图），因为广播不拷贝数据，语义上仅为只读（参见 `15-broadcast.md §5`）。如果允许可变迭代，修改广播结果会意外修改原数据的多个位置，这既不符合广播语义，也容易引入 bug。

### 5.9 Good/Bad 对比示例

```rust
// Good - ViewMutRepr cross-thread movement inside thread::scope
fn send_view_mut() {
    let mut owned = Owned::from_vec(vec![1.0, 2.0, 3.0]);
    let view_mut = owned.view_mut();

    std::thread::scope(|scope| {
        scope.spawn(move || {
            // view_mut has exclusive access in this thread
            for x in view_mut.iter_mut() {
                *x *= 10.0;
            }
        });
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

    std::thread::scope(|scope| {
        scope.spawn(move || {
            // arc_clone is safely read in this thread
            assert_eq!(arc_clone[0], 1.0);
        });

        // arc is still safe in the parent thread during the scope
        assert_eq!(arc[1], 2.0);
    });
}

// Good - parallel iteration element constraint
fn parallel_iteration(tensor: &Tensor2<f64>) {
    // f64: Send + Sync, satisfies rayon constraint
    let sum = tensor.par_iter().sum::<f64>();
}
```

---

## 6. 内部实现设计

### 6.1 Rust 所有权系统与线程安全的协同

```
┌─────────────────────────────────────────────────────────────────┐
│           Rust ownership -> thread safety mapping                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Rust ownership rule         Thread-safety guarantee             │
│  ──────────────────────      ──────────────────────              │
│                                                                 │
│  move semantics              Owned/ArcRepr can move across threads (Send) │
│  shared &T reference         ViewRepr can be shared across threads (Sync)  │
│  exclusive &mut T reference  ViewMutRepr can only move (Send only)         │
│  Arc atomic refcount         ArcRepr can be shared safely (Send + Sync)    │
│  lifetime 'a                 Views cannot outlive source data               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 并行操作安全约束

并行迭代的安全保证基于分块访问隔离（参见 `09-parallel.md §6.2`）：

```
┌─────────────────────────────────────────────────────────────────┐
│             Parallel iteration access isolation                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Array: [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]    │
│         └──thread0──┘└──thread1──┘└──thread2──┘└─thread3──┘    │
│                                                                 │
│  Key guarantees:                                                │
│  • Each element is accessed by at most one thread               │
│  • No shared writes across threads                              │
│  • Chunk boundaries are explicit and non-overlapping            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 SIMD 与并行组合安全

SIMD 操作与并行操作的组合安全性保证（参见 `08-simd.md §5`）：

```
┌─────────────────────────────────────────────────────────────────┐
│             SIMD + parallel safety guarantees                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Parallel chunking guarantees non-overlapping access         │
│  2. SIMD work in each thread runs on an exclusive data region   │
│  3. SIMD kernels have no interior mutability (no shared state)  │
│  4. `pulp::Arch` values are Copy and can cross threads cheaply  │
│  5. No extra synchronization is required                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 实现任务拆分

### Wave 1: Send/Sync 实现

- [ ] **T1**: Owned<A> 的 Send/Sync 实现
  - 文件: `src/storage/owned.rs`
  - 内容: `unsafe impl<A: Send> Send for Owned<A> {}`、`unsafe impl<A: Sync> Sync for Owned<A> {}`、完整 SAFETY 注释
  - 测试: `test_owned_send_sync`、`test_owned_negative_rc`
  - 前置: 无
  - 预计: 10 min

- [ ] **T2**: ViewRepr<'a, A> 的 Send/Sync 实现
  - 文件: `src/storage/view.rs`
  - 内容: `unsafe impl<'a, A: Sync> Send for ViewRepr<'a, A> {}`、`unsafe impl<'a, A: Sync> Sync for ViewRepr<'a, A> {}`、完整 SAFETY 注释
  - 测试: `test_view_send_sync`、`test_view_cross_thread`
  - 前置: 无
  - 预计: 10 min

- [ ] **T3**: ViewMutRepr<'a, A> 的 Send 实现
  - 文件: `src/storage/view_mut.rs`
  - 内容: `unsafe impl<'a, A: Send> Send for ViewMutRepr<'a, A> {}`、不实现 Sync 的注释、完整 SAFETY 注释
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
  - 文件: `tests/test_thread_safety.rs`
  - 内容: 多线程传递测试、并发访问测试
  - 测试: `test_owned_cross_thread`、`test_arc_concurrent_access`
  - 前置: T1-T5
  - 预计: 10 min

- [ ] **T7**: 文档和 Safety 注释审查（可选工程整理）
  - 文件: `src/storage/mod.rs`
  - 内容: 模块级线程安全文档、Send/Sync 矩阵；如仓库后续自行维护 `CHANGELOG.md`，该更新仅作为工程辅助整理，不属于本规范必需交付物
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

## 8. 测试计划

### 8.1 测试分类表

| 类型         | 位置                                     | 目的                                                      |
| ------------ | ---------------------------------------- | --------------------------------------------------------- |
| 编译期检查   | 仓库既有编译期测试机制或手写断言辅助函数 | 验证 Send/Sync 约束传播                                   |
| 跨线程测试   | `#[test]` with `std::thread`             | 验证跨线程使用安全性                                      |
| 并发访问测试 | `tests/test_thread_safety.rs`            | 多线程并发场景验证                                        |
| 边界测试     | 同模块测试中标注                         | 验证广播只读、非 `Send` / `Sync` 元素、嵌套并行回退等边界 |
| 属性测试     | 编译期断言 + 参数化并发用例              | 验证 trait 约束传播与分块不重叠不变量                     |

### 8.2 单元测试清单

| 测试函数                      | 测试内容                                  | 优先级 |
| ----------------------------- | ----------------------------------------- | ------ |
| `test_owned_send_sync`        | `Owned<f64>: Send + Sync` 编译通过        | 高     |
| `test_owned_negative_rc`      | `Owned<Rc<i32>>` 不满足 Send              | 高     |
| `test_view_send_sync`         | `ViewRepr<'_, f64>: Send + Sync` 编译通过 | 高     |
| `test_view_cross_thread`      | 视图跨线程传递正确                        | 高     |
| `test_view_mut_send`          | `ViewMutRepr<'_, f64>: Send` 编译通过     | 高     |
| `test_view_mut_not_sync`      | `ViewMutRepr` 不是 Sync（编译失败检查）   | 高     |
| `test_arc_send_sync`          | `ArcRepr<f64>: Send + Sync` 编译通过      | 高     |
| `test_arc_concurrent_read`    | 多线程并发读取 ArcRepr                    | 中     |
| `test_chunks_cover_all`       | 分块覆盖所有元素                          | 中     |
| `test_chunks_no_overlap`      | 分块不重叠                                | 中     |
| `test_owned_cross_thread`     | Owned 跨线程移动                          | 中     |
| `test_arc_internal_restore_writable_after_uniquify` | 内部唯一化 / 必要时复制后的写路径独占性验证 | 低     |

### 8.3 编译期静态检查模板

可使用仓库自带的编译期测试框架或等价断言辅助函数进行验证：

```rust
fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn owned_send_sync() {
    assert_send::<Owned<f64>>();
    assert_sync::<Owned<f64>>();
}

#[test]
fn view_send_sync() {
    assert_send::<ViewRepr<'_, f64>>();
    assert_sync::<ViewRepr<'_, f64>>();
}

#[test]
fn view_mut_send_only() {
    assert_send::<ViewMutRepr<'_, f64>>();
}

#[test]
fn arc_send_sync() {
    assert_send::<ArcRepr<f64>>();
    assert_sync::<ArcRepr<f64>>();
}
```

### 8.4 边界测试场景

| 场景                               | 预期行为                         |
| ---------------------------------- | -------------------------------- |
| `Owned<Rc<i32>>`                   | 编译期不满足 `Send`              |
| `ViewMutRepr` 被共享引用跨线程共享 | 编译期不满足 `Sync`              |
| 广播结果调用 `iter_mut()`          | 在类型层面不可调用               |
| 嵌套并行进入 `par_iter()`          | 检测后回退串行，不得共享可变状态 |

### 8.5 属性测试不变量

| 不变量                                            | 测试方法                                |
| ------------------------------------------------- | --------------------------------------- |
| `Owned<A>: Send + Sync` 当且仅当 `A: Send + Sync` | 用仓库既有编译期断言机制对正/反样例验证 |
| `ViewMutRepr` 永不实现 `Sync`                     | 编译期负向断言                          |
| 并行分块覆盖全部元素且互不重叠                    | 参数化 shape / chunk 大小验证           |

### 8.6 集成测试

| 测试文件                      | 测试内容                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| `tests/test_thread_safety.rs` | `Owned` / `View` / `ViewMut` / 共享 Arc 存储张量与 `parallel`、跨线程传递场景的端到端协同验证 |

### 8.7 多线程传递测试

```rust
#[test]
fn test_owned_cross_thread() {
    let tensor = Tensor1::from_vec(vec![1.0, 2.0, 3.0]);
    let handle = std::thread::spawn(move || {
        // tensor is available in the new thread
        assert_eq!(tensor[0], 1.0);
        assert_eq!(tensor.len(), 3);
    });
    handle.join().expect("worker thread should complete without panic");
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
        .map(|h| h.join().expect("worker thread should complete without panic"))
        .collect();

    // All threads should get the same result
    assert!(sums.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10));
}
```

### 8.8 Feature gate / 配置测试

| 配置       | 验证点                                                      |
| ---------- | ----------------------------------------------------------- |
| 默认配置   | `Send`/`Sync` 规则在无并行后端时仍成立                      |
| 启用并行   | `par_iter` / 并行逐元素执行路径仅接受满足线程安全边界的类型 |
| 启用 SIMD  | SIMD 与线程安全规则正交，不引入共享可变状态                 |
| 全 feature | 组合启用时线程安全约束与回退策略保持一致                    |

### 8.9 类型边界 / 编译期测试

| 场景                             | 测试方式                            |
| -------------------------------- | ----------------------------------- |
| `Owned<A>` 的 `Send`/`Sync` 传播 | 编译期断言辅助函数                  |
| `ViewMutRepr` 永不实现 `Sync`    | 编译期负向断言或等价约束验证        |
| 非 `Send` / `Sync` 元素被拒绝    | 编译期失败测试或手写 trait 边界检查 |

---

## 错误处理与语义边界

本文档不直接定义错误类型，但要求所有受影响模块在暴露线程安全相关失败或回退行为时遵循 `26-error.md` 的错误语义边界；线程安全规范只定义 trait 边界、panic 禁区与并行路径的一致性要求。

---

## 9. 与其他模块的交互

### 9.1 接口约定

```
┌─────────────────────────────────────────────────────────────────┐
│                 Thread-safety interface of the storage module       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  trait RawStorage {                                             │
│      type Elem;  // no Send/Sync constraint                         │
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
│  Send/Sync for each implementation is decided by the concrete type │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 数据流描述

```text
After a storage type is created or borrowed
    │
    ├── safety documentation first defines its minimum Send / Sync constraints
    ├── the parallel module uses that to decide which tensors may enter parallel paths such as par_iter / parallel element-wise execution
    ├── workspace-related thread-safety rules remain in the dedicated workspace design document
    └── cross-thread safety is ultimately guaranteed by the type system plus a small number of runtime constraints
```

### 9.3 与 parallel 模块

```
┌─────────────────────────────────────────────────────────────────┐
│                 Thread-safety requirements of the parallel module  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  impl<'a, S, D, A> IntoParallelRefIterator for TensorBase<S, D> │
│  where                                                          │
│      S: Storage<Elem = A>,                                      │
│      D: Dimension,                                              │
│      A: Element + Send + Sync,  // key constraint                  │
│  {                                                              │
│      type Iter = ParElements<'a, A, D>;                         │
│  }                                                              │
│                                                                 │
│  Chunking safety guarantees:                                       │
│  • compute_safe_chunks() guarantees non-overlap                    │
│  • lifetimes ensure the view remains valid                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 与 rayon 的集成

rayon 的 `ParallelIterator` 要求 `Item: Send`（参见 `09-parallel.md §5.3`）：

| 存储模式                  | `par_iter()` | 并行逐元素写路径 | 约束                                                                      |
| ------------------------- | :----------: | :--------------: | ------------------------------------------------------------------------- |
| `Tensor<A, D>` (Owned)    |      ✅      |        ✅        | `A: Send + Sync`                                                          |
| `TensorView<'a, A, D>`    |      ✅      |        ❌        | `A: Sync`                                                                 |
| `TensorViewMut<'a, A, D>` | ⚠️ 条件支持  |        ✅        | `par_iter()` 需先显式降级为只读视图；并行写路径要求独占借用且块划分不重叠 |
| `ArcTensor<A, D>`         |      ✅      | ❌（若实现内部写路径，则必须先内部唯一化 / 必要时复制后恢复可写） | `A: Send + Sync`                                                     |

### 9.5 与 workspace 模块的边界

workspace 的线程安全规则、借用状态机与分割守卫生命周期不属于本文档范围，统一参见 `24-workspace.md §5.7` 与 `24-workspace.md §6.3`。本文仅要求并行与张量存储相关设计在引用 workspace 时，不得与该文档定义的线程安全边界冲突。

#### TensorViewMut 的并行逐元素写路径安全性论证

`TensorViewMut` 仅在**显式独占借用**场景下支持并行逐元素写路径；若需要并行只读访问，应先通过只读重借用转换为 `TensorView` 后再调用 `par_iter()`。其安全性基于以下保证：

1. **独占语义**：`TensorViewMut` 持有 `&mut [A]`，Rust 借用规则保证在任意时刻只有唯一的可变访问者
2. **分块不重叠**：并行逐元素写路径会将数据按块分割，每个线程获得独占的 `&mut [A]` 切片，块之间无重叠
3. **Send 约束**：`A: Send` 确保元素类型可跨线程传递
4. **只读路径分离**：`par_iter()` 不直接依赖 `TensorViewMut: Sync`；调用方先显式获得只读视图，再走 `TensorView` 的并行只读实现
5. **生命周期**：`TensorViewMut` 的 `'a` 生命周期确保视图不会比源数据存活更久

```rust
// Safety argument for the parallel element-wise write path on TensorViewMut:
//
// TensorViewMut<'a, A, D> holds &mut [A] exclusively.
// The internal parallel write path splits the slice into non-overlapping chunks,
// each chunk is sent to a different thread as &mut [A].
// Since chunks don't overlap, no two threads can access the same element,
// satisfying the aliasing rule. A: Send ensures element ownership may cross threads;
// shared read requirements remain on the borrowed source type, not on mutable chunks.
```

---

## 10. 设计决策记录（ADR）

### 决策 1：显式 unsafe impl 而非依赖自动推导

| 属性     | 值                                                                                   |
| -------- | ------------------------------------------------------------------------------------ |
| 决策     | 使用显式 `unsafe impl Send/Sync`，而非依赖编译器自动推导                             |
| 理由     | 文档化意图，每个 impl 附带完整 SAFETY 注释（参见 `00-coding.md §5`），便于审查和维护 |
| 替代方案 | 依赖自动推导 — 放弃，缺少安全性论证文档，修改内部字段时可能意外改变线程安全语义      |

### 决策 2：ViewMutRepr 不实现 Sync

| 属性     | 值                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 决策     | `ViewMutRepr<'a, A>` 仅实现 `Send`，不实现 `Sync`                                                                              |
| 理由     | 独占借用语义（`&mut T`）不可共享。如果 Sync，则 `&ViewMutRepr` 可跨线程移动，导致多线程同时持有 `&mut [A]`，违反 Rust 别名规则 |
| 替代方案 | 通过 Mutex 包装实现 Sync — 放弃，引入运行时锁，违反"编译期保证"原则                                                            |

### 决策 3：ArcRepr 要求 A: Send + Sync

| 属性     | 值                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| 决策     | `ArcRepr<A>` 的 Send/Sync 要求 `A: Send + Sync`                                                                       |
| 理由     | `Arc<T>` 的线程安全要求 `T: Send + Sync`，因为多个线程可同时持有 `&T`。ArcRepr 内部使用 `Arc<Vec<A>>`，因此继承此约束 |
| 替代方案 | 仅要求 `A: Send` — 放弃，允许多线程同时通过 `&ArcRepr` 读取 `A`，如果 `A` 不是 `Sync`，存在数据竞争风险               |

### 决策 4：编译期保证优于运行时锁

| 属性     | 值                                                         |
| -------- | ---------------------------------------------------------- |
| 决策     | 所有线程安全通过 Rust 类型系统在编译期保证，不使用运行时锁 |
| 理由     | 零运行时开销、编译期排除数据竞争、与 Rust 安全理念一致     |
| 替代方案 | 运行时 Mutex/RwLock — 放弃，引入锁开销和死锁风险           |

---

## 11. 平台与工程约束

| 约束       | 说明                                                                                                            |
| ---------- | --------------------------------------------------------------------------------------------------------------- |
| `std` only | 当前版本线程安全规范以 `std` 环境为前提                                                                         |
| 单 crate   | 线程安全约束分布在既有模块中，不拆分独立并发 crate                                                              |
| 最小依赖   | 不预设 `static_assertions`、`loom`、`critical_section` 等额外依赖；如需引入，仅可作为仓库内部 dev-only 评估工具 |
| SemVer     | `Send` / `Sync` 承诺属于公开类型语义的一部分，变更需审慎评估兼容性                                              |

---

## 版本历史

| 版本  | 日期       |
| ----- | ---------- |
| 1.0.0 | 2026-04-07 |
| 1.0.1 | 2026-04-08 |
| 1.0.2 | 2026-04-08 |
| 1.0.3 | 2026-04-08 |
| 1.0.4 | 2026-04-08 |
| 1.0.5 | 2026-04-10 |
| 1.0.6 | 2026-04-14 |
| 1.0.7 | 2026-04-14 |
| 1.0.8 | 2026-04-15 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
