# 线程安全规范

> 文档编号: 25
> 适用目录: src/storage/, src/tensor/, src/iter/, src/simd/, src/ffi/，以及启用 `parallel` feature 后受影响的公开 API
> 任务阶段: Phase 2
> 前置文档: 05-storage.md, 07-tensor.md
> 需求参考: 需求说明书 §10、§16、§17、§21、§26、§28
> 范围声明: 范围内

---

## 1. 主题定位与适用范围

本文档为横切规范文档，关注线程安全性约束的跨模块适用。部分模块化章节（如文件位置、公共 API）仅用于补充说明，主线为约束定义与验证。线程安全是 Xenon 的横切关注点，贯穿所有存储模式和计算后端。本文档定义各存储模式（参见 `05-storage.md §5`）的 `Send`/`Sync` 实现规则，确保 Xenon 张量（参见 `07-tensor.md §5`）可在多线程环境下安全使用。

**范围注记：** workspace 的线程安全属性参见 `24-workspace.md`；本文不将 workspace 纳入 `需求说明书 §10` 的存储模式线程安全矩阵。

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
| 编译期保证为主  | 线程安全保证以 Rust 类型系统与 auto-trait 推导为主，特定场景辅以运行时检查（如嵌套并行检测） |
| 最小约束        | 每种存储模式仅实现其语义允许的最小 Send/Sync 约束    |
| unsafe 安全论证 | 每个 unsafe impl 附带完整 SAFETY 注释                |
| 所有权协同      | 充分利用 Rust 所有权系统与线程安全的天然协同         |

### 1.3 影响范围

- `storage`: 各存储模式的 `Send`/`Sync` 约束与内部唯一化写路径
- `tensor`: `TensorBase<S, D>` 的 auto-trait 传播与公开语义边界
- `iterator`: 只读/可写迭代器的线程可用性与别名约束
- `simd`: SIMD 内核在多线程场景下的无共享状态约束
- `dispatch`: 嵌套并行检测（`ParallelGuard`）与自动回退串行路径
- `parallel`: 启用 `parallel` feature 后公开 API 的内部并行执行路径
- `ffi`: 跨边界指针与导出描述符在多线程中的可共享/可写前提

## 2. 需求映射与范围约束

| 类型     | 内容                                                             |
| -------- | ---------------------------------------------------------------- |
| 需求映射 | 需求说明书 §10、§16、§17、§21、§26、§28                          |
| 范围内   | `Send`/`Sync` 规则、并行安全边界、广播只读约束、跨线程访问证明   |
| 范围外   | 异步运行时、锁封装策略、通用并发原语抽象                         |
| 非目标   | 通过线程安全规范引入新的同步原语、运行时锁或额外第三方并发依赖   |

---

## 3. 文件位置

线程安全实现散布于各存储模块中，不单独创建文件：

```
src/
├── storage/
│   ├── mod.rs          # Module-level thread-safety docs
│   ├── owned.rs        # Send/Sync impls for Owned<A>
│   ├── view.rs         # Send/Sync impls for ViewRepr<'a, A>
│   ├── viewmut.rs      # Send/Sync impls for ViewMutRepr<'a, A>
│   └── arc.rs          # Send/Sync impls for ArcRepr<A>
└── simd/
    └── mod.rs          # Thread-safe initialization of the Arch cache
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/storage/
├── core::marker         # PhantomData, Send, Sync
└── alloc::sync::Arc     # Atomic reference counting used by ArcRepr

parallel feature implementation paths/
├── std::sync::atomic    # AtomicUsize (thresholds)
├── std::cell::Cell      # thread_local (nested parallel detection)
└── rayon                # ParallelIterator (Send constraint)
```

### 4.2 类型级依赖

| 来源模块       | 使用的类型/trait                                                   |
| -------------- | ------------------------------------------------------------------ |
| `core::marker` | `PhantomData<A>`, `Send`, `Sync`                                   |
| `std::sync`    | `Arc<_>`, `AtomicUsize`                                            |
| `std::cell`    | `Cell<usize>`, `Cell<Option<usize>>`（仅启用 `parallel` feature 的内部执行路径） |
| `rayon::iter`  | `ParallelIterator` (要求 `Item: Send`，参见 `09-parallel.md §5.5`) |

### 4.3 依赖合法性

| 项目           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| 新增第三方依赖 | 无新增依赖；仅复用既有 `rayon` 可选依赖与标准库同步原语      |
| 合法性结论     | 符合最小依赖限制                                             |
| 替代方案       | 不适用；线程安全约束优先通过 Rust 类型系统与既有模块实现表达 |

### 4.4 依赖方向声明

依赖方向：线程安全是横切关注点。各存储模块自行声明 Send/Sync（参见 `05-storage.md §5`）；启用 `parallel` feature 后，相关公开 API 的内部并行执行路径消费这些约束（参见 `09-parallel.md §5.2`）。无循环依赖。

---

## 5. 公共 API 设计

**权威来源声明：** 本文档（25-safety.md §5.1）是 Xenon 库 Send/Sync 与线程安全规则的**唯一权威定义**。00-coding.md §4.4、05-storage.md §6.8、07-tensor.md §5.1 等其他文档若与本节冲突，应以本节为准统一校正。需求说明书作为更高层规范，若与本节冲突时以需求说明书为最终基线。

### 5.1 Send/Sync 实现规则表

自动推导结果（以 `23-ffi.md` 中 `TensorExport`/`TensorExportMut` 的 `Send`/`Sync` 由 Rust auto-trait 自动推导的模型为前提，参见 §5.4）：

| 存储模式             | Send | Sync | 条件                             | 理由                                               |
| -------------------- | :--: | :--: | -------------------------------- | -------------------------------------------------- |
| `Owned<A>`           |  ✅  |  ✅  | Send: `A: Send`; Sync: `A: Sync` | 独占拥有型存储分别按移动安全和共享安全传播元素约束 |
| `ViewRepr<'a, A>`    |  ✅  |  ✅  | `A: Sync`                        | 共享视图跨线程共享要求元素可安全共享               |
| `ViewMutRepr<'a, A>` |  ✅  |  ✗   | `A: Send`                        | 独占可写视图可转移但不可共享                       |
| `ArcRepr<A>`         |  ✅  |  ✅  | `A: Send + Sync`                 | Arc 原子计数，读共享安全；写路径仅能在内部唯一化 / 必要时复制后恢复可写性 |

**补充说明：** `ViewRepr` 仅持有共享引用（`&A`），跨线程传递共享引用只要求 `A: Sync`（允许多线程共享读取），不要求 `A: Send`（所有权转移）。这是 Rust 标准库 `&T: Send + Sync where T: Sync` 的直接推论。各存储模式的完整 API 定义参见 `05-storage.md §5`；对应的语义访问分类（`ReadOnly`/`SharedReadOnly`/`Writable`/`Owned`）参见 `07-tensor.md §5.3` 中 `AccessSemantics` 枚举定义（亦见 `05-storage.md §5.1` 的语义分类表）。

### 5.2 TensorBase<S, D> 自动推导规则

`TensorBase<S, D>` 的 `Send`/`Sync` 由 Rust 根据存储模式 `S` 的约束自动推导，结果与 §5.1 规则表一致。`D: Dimension` 要求 `Dimension: Send + Sync`；所有 Dimension 类型（`Ix0`-`Ix6`, `IxDyn`）内部仅包含 Copy 类型的值数组或 `Vec<usize>`，因此自动满足 `Send + Sync`。

### 5.3 安全违规分类表

| 安全违规类型           | 检测层级                     | 处理方式            |
| ---------------------- | ---------------------------- | ------------------- |
| 存储模式不支持可写操作 | 类型层（编译期）             | 通过 trait 约束拒绝 |
| 广播结果尝试可变访问   | 类型层（编译期）             | 通过返回类型拒绝    |
| 并行中二次并行         | 运行时（嵌套并行防护机制）   | 自动回退串行路径    |
| 逐元素算术 / 归约 / 内积中的整数溢出 | 运行时（checked arithmetic） | panic（不可恢复）   |
| 元数据 / 索引偏移 / FFI 校验类 checked arithmetic 失败 | 运行时（checked arithmetic） | 按 `26-error.md` 返回 `Result` |

**别名分类规范入口：** 凡是需要区分别名类别的模块（如 unsafe 指针算术、并行分块安全、FFI 导出决策），**必须** 使用 `TensorBase::alias_class()`（定义于 `07-tensor.md §5.3`）作为**规范入口**。`alias_class()` 返回 `AliasClass` 枚举，将 `AccessSemantics::SharedReadOnly` 的三合一语义摘要拆分为 `ArcShared` / `BroadcastAlias` / `ViewMutDerived` / `Unique` 四种精确类别。在该入口外部直接组合 `storage_kind()`、`has_zero_stride()`、`derived_from_view_mut()` 三个标志由本安全契约禁止——这些标志组合是 `alias_class()` 的实现细节，外部直接组合会引入遗漏边缘情形（如空张量广播条件 `product(shape) > 0`）的风险。`AliasClass` 枚举与 `alias_class()` 方法的权威定义见 `07-tensor.md §5.3`；`HAS_ZERO_STRIDE` 边界条件（`any(stride == 0) && product(shape) > 0`）以 `06-layout.md §5.11` 为准。

### 5.4 当前受支持元素类型的线程安全传播

当前所有受支持元素类型（`i32/i64/f32/f64`、`Complex<f32/f64>`、`bool`）均满足 `Send + Sync`，其线程安全属性随 §5.1 规则自动传播至各存储模式。

### 5.5 Owned<A> 的 Send/Sync

```rust,ignore
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

### 5.6 ViewRepr<'a, A> 的 Send/Sync

```rust,ignore
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

### 5.7 ViewMutRepr<'a, A> 的 Send（不实现 Sync）

```rust,ignore
// src/storage/viewmut.rs

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

// ViewMutRepr does not implement Sync.
//
// Mechanism (authoritative; aligned with `05-storage.md §6.10`):
//   1. The `ptr: *mut A` field opts ViewMutRepr OUT of the `Send` and `Sync`
//      auto-traits by default. Raw pointers (`*const T` / `*mut T`) carry no
//      auto-trait implementations regardless of T.
//   2. We then EXPLICITLY restore `Send` for `A: Send` via the
//      `unsafe impl Send` above (motivated by exclusive-ownership transfer,
//      see the doc comment).
//   3. We deliberately do NOT provide a corresponding `unsafe impl Sync`,
//      so `Sync` remains opted out. Sharing `&ViewMutRepr` across threads
//      could allow concurrent mutation through the raw `*mut A` pointer,
//      which would violate Rust's aliasing rules.
//
// `_marker: PhantomData<&'a mut A>` is purely a variance / dropck marker:
// it makes `ViewMutRepr` invariant in `A` and tells the borrow checker that
// `'a` is an exclusive-borrow lifetime. It is NOT what makes ViewMutRepr
// `!Sync` — that role belongs to (1) and (3) above. (Earlier drafts
// attributed `!Sync` to `PhantomData<&mut A>`; that explanation was
// imprecise: `&mut T` is in fact `Sync` whenever `T: Sync`, so the
// PhantomData alone would not exclude `Sync`.)
//
// This explanation is consistent with the struct definition in
// `05-storage.md §5.4 / §6.10`. Keep this document and `05-storage.md`
// aligned if the design changes.
```

### 5.8 ArcRepr<A> 的 Send/Sync

**共享可写边界说明：** ArcRepr 相关的唯一化（uniquify）后恢复独占写能力仅是内部实现机制。这不构成共享可写存储模式。当前版本不提供共享可写存储模式（参见 `需求说明书 §6.1`）。

```rust,ignore
// src/storage/arc.rs

/// # Safety
///
/// `ArcRepr<A>` implements `Send` because:
///
/// 1. **Arc atomicity**: Arc-backed shared storage uses atomic reference counting,
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
/// 2. **Arc synchronization guarantee**: the shared storage handle is `Sync` when
///    `A: Send + Sync`, because the backing allocation can then be safely shared by
///    reference across multiple threads.
///
/// 3. **No interior mutability**: Data cannot be modified through `&ArcRepr`.
///    Any internal transition back to writable state requires exclusive `&mut`
///    access and is not exposed as a separate public API guarantee.
unsafe impl<A: Send + Sync> Sync for ArcRepr<A> {}
```

### 5.9 广播结果不可变迭代的原因

```rust,ignore
// Broadcast results use ViewRepr (read-only view), no mutable iterator provided

// Good - broadcast results are read-only
let a = Tensor1::from_shape_vec([3], vec![1.0, 2.0, 3.0])?;
let b = a.view().broadcast_to([3, 3])?;  // broadcast result: TensorView
let sum: f64 = b.iter().sum();  // OK: read-only iteration

// Bad - compilation error: broadcast results cannot be mutably iterated
// let mut b_mut = a.view().broadcast_to([3, 3]).unwrap();
// b_mut.iter_mut()  // Compile error! ViewRepr does not implement StorageMut
```

**设计决策：** 广播结果使用 `ViewRepr`（只读视图），因为广播不拷贝数据，语义上仅为只读（参见 `15-broadcast.md §5`）。如果允许可变迭代，修改广播结果会意外修改原数据的多个位置，这既不符合广播语义，也容易引入 bug。

### 5.10 Good/Bad 对比示例

```rust,ignore
// Good - ViewMutRepr cross-thread movement inside thread::scope
fn send_view_mut() {
    let mut owned = Tensor1::from_shape_vec(Ix1(3), vec![1.0, 2.0, 3.0])
        .expect("shape and data length must match");
    let mut view_mut = owned.view_mut();

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
    let mut owned = Tensor1::from_shape_vec(Ix1(3), vec![1.0, 2.0, 3.0])
        .expect("shape and data length must match");
    let view_mut = owned.view_mut();
    let view_ref = &view_mut;

    // Compilation error: &ViewMutRepr is not Send
    // because ViewMutRepr is not Sync
    // std::thread::spawn(move || {
    //     println!("{:?}", view_ref);
    // });
}

// Good - ArcRepr cross-thread sharing
fn share_arc_tensor() -> Result<(), XenonError> {
    let arc = ArcTensor1::from_shape_vec(Ix1(3), vec![1.0, 2.0, 3.0])?;
    let arc_clone = arc.clone();  // strong_count = 2

    // Read fixed offsets up-front. Xenon does not implement `std::ops::Index`;
    // use `try_at` (17-indexing §5.2) for fallible structured indexing.
    let parent_v1 = *arc.try_at(Ix1(1))?;
    assert_eq!(parent_v1, 2.0);

    std::thread::scope(|scope| {
        scope.spawn(move || {
            // The closure captures `arc_clone` by move; failures from
            // try_at are unwrapped here only because indices are constants.
            let v0 = *arc_clone.try_at(Ix1(0))
                .expect("constant index 0 is in bounds for shape [3]");
            assert_eq!(v0, 1.0);
        });
    });

    Ok(())
}

// Good - parallel iteration element constraint
fn parallel_iteration(tensor: &Tensor2<f64>) {
    // With the `parallel` feature enabled, the implementation may choose
    // an internal parallel execution path for public APIs on eligible tensors.
    let sum = tensor.sum();
}
```

### 5.11 FFI unsafe 与线程安全边界

**适用范围：** `src/ffi/`（详见 `23-ffi.md`）。本节是横切线程安全规范在 FFI 边界的延伸，与 `23-ffi.md` 的 unsafe 接口契约协同。

**核心约束：**

| 约束类别 | 规则 |
|:--|:--|
| 描述符 Send/Sync | `TensorExport<'a, A>` / `TensorExportMut<'a, A>`（`23-ffi.md §5.4`）按 Rust auto-trait 推导：含 `*const A` / `*mut A` 字段使其默认 `!Send + !Sync`；不为这两个类型提供任何 `unsafe impl Send/Sync`。跨线程移动/共享导出的 raw 描述符**必须**由调用方在 `unsafe` 块中重新构造一个匹配 lifetime/borrow 的描述符，且仅在能证明无别名 + 无并发写时进行 |
| C 端并发写 | 同一 `TensorExportMut` 在 C 侧**禁止**并发写（即便 C 端没有 Rust 借用规则的强制）；这是 `from_raw_parts_mut` round-trip 的隐含前提，违反时 Rust 侧重新构造的 `&mut` 别名会导致 UB |
| Raw pointer 输入责任 | `TensorBase::from_raw_parts(_mut)`（`07-tensor.md §5.6 / §5.7`）的调用方必须保证：(1) provenance — `data_ptr` 必须从可被 `'a` 长度合法借用的对象派生；(2) lifetime — `'a` 不超过该对象的有效借用期；(3) alignment — `data_ptr % align_of::<A>() == 0`；(4) initialization — 范围内所有元素已是有效的 `A` 实例；(5) aliasing — `from_raw_parts_mut` 时无其他活跃别名（无论 Rust 侧还是 C 侧） |
| Allocator 归属 | Owned 路径不得跨 allocator 释放：通过 `from_raw_parts` 接收的 raw 描述符**禁止**被 Xenon 用 `dealloc(System)` 释放（除非显式声明的 round-trip 构造由 Xenon 自己分配）。详见 `23-ffi.md §10` 的 `ForeignAllocatorMismatch` 错误 |
| Panic 跨边界 | Rust 侧 panic 不得跨 `extern "C"` 函数边界。所有 `extern "C"` 导出函数必须用 `catch_unwind` 包裹并把 panic 转换为错误码（`23-ffi.md §6.4`），否则会触发 UB |

**与本文档其他章节的关系：**
- §5.7 的 `ViewMutRepr !Sync` 论证适用于 Xenon 内部的 ViewMut；FFI 中的 `TensorExportMut` 是**单独的、独立的描述符类型**，不实现 ViewMut 的相关 trait，因此 FFI 描述符不沿用 ViewMut 的 Send/Sync 结论，而是按 raw pointer 默认推导。
- §5.4 的 "当前受支持元素类型线程安全传播" 同样适用于通过 FFI 传出的元素类型，但前提是元素以原生 Rust 类型（i32/i64/f32/f64/Complex<f32>/Complex<f64>/bool）形式存在；C 侧若把字节范围重新解释为另一种类型，进入 `23-ffi.md §10` 的 `AbiMismatch` 错误路径。

更详细的 FFI 错误模型与 unsafe 边界：见 `23-ffi.md §10 / §11`。本节只规范线程安全维度。

### 5.12 unsafe 入口索引

下列 `unsafe fn` 在 Xenon crate 内部 / 公开 API 面使用，每个调用点必须用 `unsafe { ... }` 块包裹并附 `// SAFETY:` 注释证明其契约满足。本节只列入口，详细 `# Safety` 契约由各 owner 文档维护。

#### 5.12.1 `pub(crate)` 内部 unsafe fn 清单（4 项）

仅 crate 内部可见，不进入公开 API。调用点全部位于本 crate 源代码内：

| `pub(crate) unsafe fn` | Owner 文档 | 契约要点 |
|:--|:--|:--|
| `TensorBase::<S, D>::new_unchecked(storage, shape, strides, offset, flags, derived_from_view_mut) where S: RawStorage` | `07-tensor.md §5.6` | **唯一 canonical unsafe 构造器**——所有其他内部 unchecked 构造器必须 forward 到此处；shape/strides/flags/offset 互一致；flags 由 `compute_layout_flags` 产出；逻辑访问范围在 storage 内；shape product 已 overflow-check；`derived_from_view_mut` 对 Owned 路径（`S = Owned<A>`）必须 `false`，仅在 `ViewMutRepr` 降级 / 切片自带降级标记的源场景为 `true` |
| `TensorBase::<Owned<A>, D>::from_raw_vec_unchecked(data: Vec<A>, shape: D)` | `07-tensor.md §5.6` | `data.as_ptr()` 满足 `A` 对齐；`shape.checked_size()` 已验证；`data.len()` 等于该值；F-order 元数据合法 |
| `Tensor::from_shape_vec_aligned_unchecked(shape: D, data: Vec<A>)` | `21-type.md §5.6` (cast/to_owned helper) | `TensorBase::new_unchecked` 的**薄封装**（本条指数录存在供完整性索引；实质性安全契约已 forward 到 07-tensor.md §5.6）；`data.len() == product(shape)`；shape 已验证；无独立 unsafe 不变式 |

#### 5.12.2 `pub` 公开 unsafe API 清单（5 项）

公开 unsafe API；下游用户可直接调用，必须遵守同样的 `unsafe { } + // SAFETY:` 调用规范：

| `pub unsafe fn` | Owner 文档 | 契约要点 |
|:--|:--|:--|
| `TensorBase::<ViewRepr<'a, A>, D>::from_raw_parts(...)` | `07-tensor.md §5.7` (FFI 入口) | provenance / lifetime / alignment / initialization / aliasing 五点（与本文 §5.11 一致） |
| `TensorBase::<ViewMutRepr<'a, A>, D>::from_raw_parts_mut(...)` | `07-tensor.md §5.7` (FFI 入口) | 同 `from_raw_parts` 五点 + 可写布局非重叠校验（参见 `07-tensor.md §5.7`） |
| `TensorBase::<Owned<A>, D>::from_raw_parts_owned(raw: OwnedRawParts<A, D>)` | `07-tensor.md §5.7` (Owned 重建入口) | `raw` 必须由配对的 `into_raw_parts` 产生且未被释放；元数据互一致；详见 `07-tensor.md §5.7` |
| `WorkspaceBorrowMut::as_maybe_uninit_typed_slice<T>(&mut self, count)` | `24-workspace.md §5.6` | `T: crate::element::Element`；count 不致 byte 长度溢出（否则 `TypedViewRejection::TypedByteLengthOverflow`）；返回 `&mut [MaybeUninit<T>]` 调用方负责完整初始化才能 `assume_init`（R13 E-01：从 §5.12.1 移到此公开表，与 `24-workspace.md §5.6` `pub unsafe fn` 实际定义可见性一致） |
| `WorkspaceBorrowMut::assume_init_typed_slice<T>(&mut self, count)` | `24-workspace.md §5.6` | `T: Element`；调用方已保证范围内 `count` 个 `T` 已被有效初始化（R13 E-01：可见性同上） |

调用点要求：每个 `unsafe { ... }` 块必须紧邻 `// SAFETY:` 注释，注释引用 owner 文档章节并列出本调用点已建立的不变式如何满足契约。25-safety §5.12 仅作为索引；具体契约文本以 owner 文档为准，禁止在两处分别维护。

---

## 6. 内部实现设计

### 6.1 Rust 所有权系统与线程安全的协同

```
┌────────────────────────────────────────────────────────────────────────────┐
│           Rust ownership -> thread safety mapping                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Rust ownership rule         Thread-safety guarantee                       │
│  ──────────────────────      ──────────────────────                        │
│                                                                            │
│  move semantics              Owned/ArcRepr can move across threads (Send)  │
│  shared &T reference         ViewRepr can be shared across threads (Sync)  │
│  exclusive &mut T reference  ViewMutRepr can only move (Send only)         │
│  Arc atomic refcount         ArcRepr can be shared safely (Send + Sync)    │
│  lifetime 'a                 Views cannot outlive source data              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 并行操作安全约束

启用 `parallel` feature 后，公开 API 的内部并行执行路径安全保证基于分块访问隔离（参见 `09-parallel.md §6.2`）：

```
┌─────────────────────────────────────────────────────────────────┐
│             Parallel iteration access isolation                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Array: [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]      │
│         └──thread0──┘└──thread1──┘└──thread2──┘└─thread3──┘     │
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
  - 文件: `src/storage/viewmut.rs`
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

- [ ] **T5**: 并行执行路径分块安全验证
  - 文件: `src/parallel/iter.rs`
  - 内容: 分块完整性/不重叠/边界安全的测试
  - 测试: `test_chunks_cover_all`、`test_chunks_no_overlap`
  - 前置: T1-T4
  - 预计: 10 min

- [ ] **T6**: 线程安全集成测试
  - 文件: `tests/test_parallel.rs`、`tests/test_error.rs`
  - 内容: 多线程传递测试、并发访问测试归入现有并行/错误测试文件，不单独设立 `tests/test_thread_safety.rs`
  - 测试: `test_owned_cross_thread`、`test_arc_concurrent_access`
  - 前置: T1-T5
  - 预计: 10 min

- [ ] **T7**: 文档和 Safety 注释审查（可选工程整理）
  - 文件: `src/storage/mod.rs`
  - 内容: 模块级线程安全文档、Send/Sync 矩阵；如仓库后续自行维护 `CHANGELOG.md`，该更新仅作为工程辅助整理，不属于本规范必需交付物
  - 测试: `cargo doc` 通过
  - 前置: T1-T4
  - 预计: 10 min

## 8. 测试计划

### 8.1 测试分类表

| 类型         | 位置                                     | 目的                                                      |
| ------------ | ---------------------------------------- | --------------------------------------------------------- |
| 编译期检查   | 仓库既有编译期测试机制或手写断言辅助函数 | 验证 Send/Sync 约束传播                                   |
| 跨线程测试   | `#[test]` with `std::thread`             | 验证跨线程使用安全性                                      |
| 并发访问测试 | `tests/test_parallel.rs` / `tests/test_error.rs` | 多线程并发场景验证                                |
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

### 8.3 边界测试场景

| 场景                               | 预期行为                         |
| ---------------------------------- | -------------------------------- |
| `Owned<Rc<i32>>`                   | 编译期不满足 `Send`              |
| `ViewMutRepr` 被共享引用跨线程共享 | 编译期不满足 `Sync`              |
| `IterMut` / `AxisIterMut` 被共享引用跨线程共享 | 编译期不满足 `Sync` |
| 广播结果调用 `iter_mut()`          | 在类型层面不可调用               |
| 嵌套并行进入公开 API 的内部并行路径 | 检测后回退串行，不得共享可变状态 |

### 8.4 属性测试不变量

| 不变量                                            | 测试方法                                |
| ------------------------------------------------- | --------------------------------------- |
| `Owned<A>: Send + Sync` 当且仅当 `A: Send + Sync` | 用仓库既有编译期断言机制对正/反样例验证 |
| `ViewMutRepr` 永不实现 `Sync`                     | 编译期负向断言                          |
| 可写迭代器永不实现 `Sync`                         | 编译期负向断言                          |
| 并行分块覆盖全部元素且互不重叠                    | 参数化 shape / chunk 大小验证           |

### 8.5 集成测试

| 测试文件                      | 测试内容                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| `tests/test_parallel.rs` / `tests/test_error.rs` | 线程安全测试归入现有并行/错误测试文件，覆盖 `Owned` / `View` / `ViewMut` / 共享 Arc 存储张量与 `parallel`、跨线程传递场景的端到端协同验证 |

### 8.6 Feature gate / 配置测试

| 配置       | 验证点                                                      |
| ---------- | ----------------------------------------------------------- |
| 默认配置   | `Send`/`Sync` 规则在无并行后端时仍成立                      |
| 启用并行   | 公开 API 的可选内部并行执行路径仅接受满足线程安全边界的类型 |
| 启用 SIMD  | SIMD 与线程安全规则正交，不引入共享可变状态                 |
| 全 feature | 组合启用时线程安全约束与回退策略保持一致                    |

### 8.7 类型边界 / 编译期测试

| 场景                             | 测试方式                            |
| -------------------------------- | ----------------------------------- |
| `Owned<A>` 的 `Send`/`Sync` 传播 | 编译期断言辅助函数                  |
| `ViewMutRepr` 永不实现 `Sync`    | 编译期负向断言或等价约束验证        |
| 非 `Send` / `Sync` 元素被拒绝    | 编译期失败测试或手写 trait 边界检查 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

```
┌────────────────────────────────────────────────────────────────────┐
│                 Thread-safety interface of the storage module      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  trait RawStorage {                                                │
│      type Elem;  // no Send/Sync constraint                        │
│  }                                                                 │
│                                                                    │
│  trait Storage: RawStorage {                                       │
│      type Elem;                                                    │
│      fn as_ptr(&self) -> *const Self::Elem;                        │
│  }                                                                 │
│                                                                    │
│  trait StorageMut: Storage {                                       │
│      fn as_mut_ptr(&mut self) -> *mut Self::Elem;                  │
│  }                                                                 │
│                                                                    │
│  Send/Sync for each implementation is decided by the concrete type │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 9.2 数据流描述

```text
After a storage type is created or borrowed
    │
    ├── safety documentation first defines its minimum Send / Sync constraints
    ├── the parallel module uses that to decide which tensors may enter optional internal parallel execution paths of public APIs
    ├── workspace-related thread-safety rules remain in the dedicated workspace design document
    └── cross-thread safety is ultimately guaranteed by the type system plus a small number of runtime constraints
```

### 9.3 与 `parallel` feature 影响的公开 API

```
┌─────────────────────────────────────────────────────────────────┐
│      Thread-safety requirements of public APIs with optional    │
│              internal execution paths behind `parallel`         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Public APIs keep their stable signatures.                      │
│  When the `parallel` feature is enabled, eligible execution     │
│  paths may internally split work into non-overlapping chunks.   │
│                                                                 │
│  Internal safety guarantees:                                    │
│  • compute_safe_chunks(total, num_workers) divides the logical  │
│    element range [0, total) into non-overlapping [start, end)   │
│    intervals. Each interval is assigned to exactly one worker   │
│    and returned as &[(usize, usize)]. The intervals satisfy     │
│    start[0] = 0, end[i] = start[i+1], end[last] = total.        │
│    Implementation is in src/parallel/mod.rs as a pub(crate) fn. │
│    Panics if num_workers = 0 (violates internal precondition).  │
│  • lifetimes ensure the view remains valid                      │
│  • element/thread-safety bounds remain checked before dispatch  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 与 rayon 的集成

启用 `parallel` feature 时，内部执行路径可借助 rayon 等并行后端；后端仍要求被分发到线程的工作单元满足 `Send`/`Sync` 边界（参见 `09-parallel.md §5.5`）：

| 存储模式                  | 公开 API 可选内部并行只读路径 | 内部并行逐元素写路径 | 约束                                                           |
| ------------------------- | :--------------------------: | :------------------: | --------------------------------------------------------------- |
| `Tensor<A, D>` (Owned)    |              ✅              |          ✅          | `A: Send + Sync`                                                |
| `TensorView<'a, A, D>`    |              ✅              |          ❌          | `A: Sync`                                                       |
| `TensorViewMut<'a, A, D>` |       ⚠️ 需先降级为只读       |          ✅          | 只读路径经显式只读重借用进入；写路径要求独占借用且块划分不重叠  |
| `ArcTensor<A, D>`         |              ✅              | ❌（若实现内部写路径，则必须先内部唯一化 / 必要时复制后恢复可写） | `A: Send + Sync`   |

**ViewMutRepr 并行写路径机制说明：** `ViewMutRepr` 虽为 `!Sync`，但并行写路径通过独占 `&mut` 借用接管整个视图后将其分块为互不重叠的子视图，每个子视图仅由一个线程独占持有，因此不违反 `Sync` 约束。该机制与 §5.7 中 `ViewMutRepr: Send where A: Send` 的论证一致——独占所有权可跨线程转移，但不允许共享。

并行写路径的非重叠保证由 `07-tensor.md §5.7` 的 `validate_non_overlapping_layout` 算法提供（保守拒绝难以证明非重叠的可写布局）。

### 9.5 与 workspace 模块的边界

workspace 的线程安全规则（`!Send + !Sync` 实现选择及理由，参见 `24-workspace.md §5.1` 与 `24-workspace.md` 决策 5）、借用状态机与分割守卫生命周期（参见 `24-workspace.md §6.3`）不属于本文档范围。本文仅要求并行与张量存储相关设计在引用 workspace 时，不得与该文档定义的线程安全边界冲突。

---

## 10. 错误处理与语义边界

- 本文档不直接定义错误类型，但要求所有受影响模块在暴露线程安全相关失败或回退行为时遵循 `26-error.md` 的错误语义边界。
- 线程安全规范只定义 trait 边界、panic 禁区与并行路径的一致性要求。

---

## 11. 设计决策记录

### 决策 1：显式 unsafe impl 而非依赖自动推导

| 属性     | 值                                                                                   |
| -------- | ------------------------------------------------------------------------------------ |
| 决策     | 使用显式 `unsafe impl Send/Sync`，而非依赖编译器自动推导                             |
| 理由     | 文档化意图，每个 impl 附带完整 SAFETY 注释（参见 `00-coding.md §6.2–§6.3`），便于审查和维护 |
| 替代方案 | 依赖自动推导 — 放弃，缺少安全性论证文档，修改内部字段时可能意外改变线程安全语义      |

### 决策 2：ViewMutRepr 不实现 Sync

| 属性     | 值                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 决策     | `ViewMutRepr<'a, A>` 仅实现 `Send`，不实现 `Sync`                                                                              |
| 理由     | 独占借用语义（`&mut T`）不可共享，详细论证参见 §5.7 unsafe impl 注释与 §5.3 安全违规分类表                                     |
| 替代方案 | 通过 Mutex 包装实现 Sync — 放弃，引入运行时锁，违反“以编译期类型系统为主，特定场景辅以运行时检查”的原则                        |

### 决策 3：ArcRepr 要求 A: Send + Sync

| 属性     | 值                                                                                                                    |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| 决策     | `ArcRepr<A>` 的 Send/Sync 要求 `A: Send + Sync`                                                                       |
| 理由     | `Arc<T>` 的线程安全要求 `T: Send + Sync`，因为多个线程可同时持有 `&T`。ArcRepr 通过抽象共享存储句柄承载只读共享，因此继承此约束 |
| 替代方案 | 仅要求 `A: Send` — 放弃，允许多线程同时通过 `&ArcRepr` 读取 `A`，如果 `A` 不是 `Sync`，存在数据竞争风险               |

### 决策 4：编译期保证优于运行时锁

| 属性     | 值                                                         |
| -------- | ---------------------------------------------------------- |
| 决策     | 线程安全保证以 Rust 类型系统在编译期为主；仅在特定执行路径（如嵌套并行检测）辅以运行时检查，不使用运行时锁 |
| 理由     | 保持零锁开销与编译期数据竞争排除，同时为少数无法仅靠类型系统表达的执行路径约束保留最小运行时防护           |
| 替代方案 | 运行时 Mutex/RwLock — 放弃，引入锁开销和死锁风险           |

---

## 12. 性能考量

| 开销点 | 影响范围 | 可接受理由 |
| ------ | -------- | ---------- |
| `ArcRepr` COW 深拷贝 | 非唯一持有时写路径触发 `clone` | 仅在首次写入共享数据时发生，后续写操作走唯一持有快速路径 |
| `ParallelGuard` thread_local 读写 | 启用 `parallel` feature 后每次并行操作 | 单次 `Cell::get/set`，纳秒级开销，远低于线程调度成本 |
| `ArcRepr` 原子引用计数 | `clone`/`drop` 涉及 `AtomicUsize` 操作 | 标准库 `Arc` 开销，与 `std::sync::Arc` 一致 |

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                                            |
| ---------- | --------------------------------------------------------------------------------------------------------------- |
| `std` only | 当前版本线程安全规范以 `std` 环境为前提                                                                         |
| MSRV       | Rust 1.85+                                                                                                      |
| 单 crate   | 线程安全约束分布在既有模块中，不拆分独立并发 crate                                                              |
| SemVer     | `Send` / `Sync` 承诺属于公开类型语义的一部分，变更需审慎评估兼容性                                              |
| 最小依赖   | 不预设 `static_assertions`、`loom`、`critical_section` 等额外依赖；如需引入，仅可作为仓库内部 dev-only 评估工具 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
