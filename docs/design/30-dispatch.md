# 执行路径派发模块设计

> 文档编号: 30
> 模块文件: src/dispatch.rs
> 任务阶段: Phase 5
> 前置文档: 06-layout.md, 07-tensor.md, 09-parallel.md

---

## 1. 模块定位

### 1.1 职责边界

| 职责       | 包含                                                                        |
| ---------- | --------------------------------------------------------------------------- |
| 路径裁决   | `ExecPath` 三路仲裁（Serial / Simd / Parallel）                             |
| 阈值管理   | 并行阈值与 SIMD 通用阈值均提供编译期默认值+ 对称的内部运行时覆写与重置接口  |
| 嵌套防护   | `ParallelGuard` / `ParallelContext` 的 thread-local RAII 保护，防止二次并行 |
| 策略参数   | `ParallelExecStrategy` 定义，供 parallel/ 后端消费 i                        |
| 快捷查询   | `should_parallelize()` 布尔查询，供仅关注串行/并行二选的调用方使用          |

| 职责       | 不包含                                                      |
| ---------- | ----------------------------------------------------------- |
| 路径裁决   | ISA 检测与选择（归 `pulp::Arch`，在 `simd/` 内部完成）      |
| 阈值管理   | SIMD 最终准入判定（lane 宽度、元素类型支持等——归 `simd/`）  |
| 嵌套防护   | 串行回退实现或并行/ SIMD 执行逻辑本身（归各语义模块与后端） |
| 策略参数   | 广播形状仲裁（归 `math/broadcast`）                         |
| 快捷查询   | 标量实现代码（归 `math/matrix/reduction`）                  |

### 1.2 设计原则

| 原则         | 体现                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------- |
| 单一裁决点   | 所有 `math` / `matrix` / `reduction` 模块通过 `dispatch::select_exec_path()` 进行三路裁决 |
| 二级裁决模型 | dispatch 选择三条互斥路径之一；SIMD / Parallel 后端在被选中后做内部细化                   |
| 零成本抽象   | `feature = "parallel/simd"` 关闭时 dispatch 仍然存在                                      |
| 嵌套并行防护 | thread-local guard 防止库内部二次并行，失败时静默回退 `Serial` 路径                       |
| 透明回退     | 进入失败或条件不满足时静默回退到 `ExecPath::Serial`，绝不 panic 或返回错误                |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                            |
| -------- | ------------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §9、§13、§14、§28                                                    |
| 范围内   | `ExecPath` 三路裁决、并行阈值配置、嵌套并行防护、`ParallelExecStrategy` 定义、SIMD 路径推荐 |
| 范围外   | SIMD 准入判定（ISA 检测、lane 宽度选择）、scalar implementations、广播形状仲裁  |
| 标量回退 | dispatch 自身不包含标量实现代码；回退到 `Serial` 后由各语义模块自行执行标量路径 |
| 非目标   | 不在 dispatch 模块内引入第三方依赖（包括 pulp）；不扩展 dispatch 为子目录模块   |

---

## 3. 文件位置

```
src/dispatch.rs                    # Single-file module
```

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/dispatch.rs
├── std::sync::atomic         # AtomicUsize (threshold storage)
├── std::cell::Cell           # Cell<bool> (thread-local guard)
├── core::marker::PhantomData # Guard's _private field
├── crate::error              # XenonError, InvalidArgument (only via ParallelExecStrategy::new)
├── crate::tensor             # .len(), .is_f_contiguous() (layout queries via tensor)
└── crate::layout             # Alignment helpers (via tensor)
```

### 4.2 类型级依赖

| 来源模块  | 使用的类型/trait                              |
| --------- | --------------------------------------------- |
| `tensor`  | `TensorBase`, `.len()`, `.is_f_contiguous()`  |
| `layout`  | `is_aligned()`                                |
| `error`   | `XenonError`, `XenonError::InvalidArgument`   |
| `std`     | `AtomicUsize`, `Cell<bool>`, `thread_local!`  |
| `core`    | `core::marker::PhantomData`                   |

### 4.3 依赖合法性

| 项目           | 说明                                                               |
| -------------- | ------------------------------------------------------------------ |
| 新增第三方依赖 | 无                                                                 |
| 合法性结论     | 合法；dispatch 仅使用 std 与 crate 内部既有模块，符合最小依赖原则  |
| 替代方案       | 不适用；当前设计无需额外依赖                                       |

### 4.4 依赖方向声明

依赖方向：单向向上。dispatch 仅消费 `tensor`/`layout` 等核心模块；被 `parallel`、`math`、`matrix`、`reduction` 等模块消费。

---

## 5. 公共 API 设计

### 5.1 Crate 级约束

- 所有 dispatch API 均为 `pub(crate)`；不对外公开任何 dispatch 类型或函数。
- `dispatch.rs` 始终编译，模块本体不依赖任何 feature gate。
- `ParallelExecStrategy` 相关代码路径只在 `feature = "parallel"` 下编译并使用 `rayon`。
- dispatch 内涉及读取 rayon 池大小的代码必须用 `#[cfg(feature = "parallel")]` 包裹。
- 非 parallel feature 下，`ParallelExecStrategy::new()` 不可达，即 dispatch 永远不返回 `ExecPath::Parallel`。
- 运行时行为随 feature flag 变化：
  - `feature = "parallel"` 关闭时，`select_exec_path()` 永不返回 `ExecPath::Parallel`
  - `feature = "simd"` 关闭时，`select_exec_path()` 永不返回 `ExecPath::Simd`
  - 两者均关闭时，`select_exec_path()` 总是返回 `ExecPath::Serial`

### 5.2 ExecPath 枚举

```rust,ignore
/// ExecPath represents the three mutually exclusive execution paths
/// that dispatch.rs may recommend to callers.
///
/// The caller uses a single match on this enum to delegate to the
/// appropriate backend. This is the primary API of the dispatch module.
///
/// Note: `select_exec_path()` returns `(ExecPath, Option<ParallelGuard>)`.
/// When `ExecPath::Parallel` is selected, the accompanying guard is
/// `Some(_)` and the caller must keep it alive for the duration of the
/// parallel region. See §5.5 for the atomic selection-and-entry contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecPath {
    /// Serial scalar execution.
    ///
    /// The caller uses its own scalar implementation. This is the
    /// default fallback when neither SIMD nor parallel preconditions
    /// are met.
    Serial,

    /// Serial path with SIMD acceleration.
    ///
    /// Caller delegates to the `simd/` backend for vectorized execution.
    /// However, `simd/` may further internally fall back to scalar if
    /// its own preconditions fail (e.g., alignment, ISA availability,
    /// element type support). That fallback is invisible to dispatch.
    ///
    /// dispatch only signals "SIMD path is preferred for this input shape";
    /// `simd/` retains final admission authority.
    Simd,

    /// Parallel execution.
    ///
    /// Caller delegates to the `parallel/` backend. Each worker chunk
    /// MAY independently invoke a SIMD kernel from `simd/` per chunk-local
    /// admission (thread × SIMD double-layer acceleration; see
    /// `08-simd.md` §9.3 and `09-parallel.md`). This path is only returned
    /// when `feature = "parallel"` is enabled AND the input meets the
    /// parallel threshold AND the thread is not already inside a library-internal
    /// parallel region.
    ///
    /// When this variant is returned, `select_exec_path()` also yields
    /// the corresponding `ParallelGuard` (held by the caller) — selection
    /// and entry are bound into a single atomic step. See §5.5.
    Parallel,
}
```

**语义约定**：`ExecPath::Simd` 表示 dispatch 推荐 SIMD 路径；`simd/` 模块仍保有最终准入权（检测 ISA、lane 宽度、对齐细节等），并可在内部回退标量。dispatch 不参与该回退决策，也不感知其发生。

**`ExecPath::Parallel` 与 guard 的原子绑定**： 一旦 `select_exec_path()` 返回 `ExecPath::Parallel`，调用方必然拿到 `Some(ParallelGuard)`。两者不可分离。任何让调用方先收到 `ExecPath::Parallel` 再尝试单独 `enter()` 的设计都会引入 TOCTOU 窗口（中间被另一并行 API 抢占进入），与"嵌套并行检测"语义冲突。

**feature gate 影响：**

| feature 组合        | `ExecPath::Serial` | `ExecPath::Simd` | `ExecPath::Parallel` |
| ------------------- | :----------------: | :--------------: | :------------------: |
| 默认（无 feature）  | ✅ 始终可返回      | ❌ 永不返回      | ❌ 永不返回          |
| 仅 `simd`           | ✅                 | ✅ 条件返回      | ❌ 永不返回          |
| 仅 `parallel`       | ✅                 | ❌ 永不返回      | ✅ 条件返回          |
| `simd` + `parallel` | ✅                 | ✅ 条件返回      | ✅ 条件返回          |

### 5.3 ParallelExecStrategy 结构体

```rust,ignore
/// Execution strategy parameters consumed by the parallel backend.
///
/// Defined in dispatch.rs; consumed by `parallel/` module functions
/// such as `par_map`, `par_zip_map`, `par_sum`, and `par_dot`.
///
/// Fields are kept private to enforce construction via the
/// validating constructor `new()`. This guarantees that any
/// `ParallelExecStrategy` value observed by `parallel/` is already
/// well-formed and the backend never needs a defensive validation
/// step. See `09-parallel.md` §5.4 for the consumed semantics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParallelExecStrategy {
    /// Suggested chunk size for parallel chunking.
    ///
    /// `None` means the parallel module decides autonomously
    /// (typically via `compute_safe_chunks`).
    chunk_size: Option<usize>,

    /// Maximum worker count.
    ///
    /// `None` means use rayon's default thread pool size.
    max_workers: Option<usize>,
}

impl ParallelExecStrategy {
    /// Construct a validated strategy.
    ///
    /// All `max_workers` and `chunk_size` validation happens here at
    /// construction time:
    ///
    /// - `chunk_size == Some(0)` → `InvalidArgument`
    /// - `max_workers == Some(0)` → `InvalidArgument`
    /// - `max_workers == Some(n)` with `n > rayon::current_num_threads()`
    ///   → `InvalidArgument`
    ///
    /// The pool-size upper bound is read once at construction via
    /// `rayon::current_num_threads()`. Since rayon's global pool size
    /// is fixed for the lifetime of the process by default, this value
    /// is stable for typical use; if the caller intentionally swaps
    /// thread pools mid-execution, the strategy must be reconstructed.
    /// `parallel/` consumes a pre-validated strategy and never returns
    /// `InvalidArgument` for `max_workers` itself (single source of
    /// validation).
    pub(crate) fn new(
        chunk_size: Option<usize>,
        max_workers: Option<usize>,
    ) -> Result<Self, XenonError>;

    /// Default strategy: let the parallel backend decide everything.
    pub(crate) fn auto() -> Self {
        Self { chunk_size: None, max_workers: None }
    }

    pub(crate) fn chunk_size(&self) -> Option<usize> { self.chunk_size }
    pub(crate) fn max_workers(&self) -> Option<usize> { self.max_workers }
}
```

**参数校验规则：**

| 字段          | 合法范围                          | 默认值 | 非法时行为                                                                  |
| ------------- | --------------------------------- | ------ | --------------------------------------------------------------------------- |
| `max_workers` | `Some(1..=rayon::current_num_threads())` 或 `None` | `None` | `Some(0)` 与 `Some(n) where n > pool_size` 都在 `new()` 内返回 `InvalidArgument` |
| `chunk_size`  | `Some(n)` where `n > 0` 或 `None`                  | `None` | `Some(0)` 在 `new()` 内返回 `InvalidArgument`              |

**字段不变量归属：** 所有 `max_workers` / `chunk_size` 校验统一在 dispatch 内的 `ParallelExecStrategy::new()` 构造器中完成，包括对 rayon 线程池上限的检查（通过 `rayon::current_num_threads()` 一次性读取）。`parallel/` 模块只消费已校验的策略，不再返回 `max_workers` 相关的 `InvalidArgument`。

### 5.4 ParallelGuard / ParallelContext

```rust,ignore
/// RAII guard that indicates the current thread is inside a
/// library-internal parallel region.
///
/// A `ParallelGuard` value is **only ever obtained as the second tuple
/// element of `select_exec_path()`** when that function selects
/// `ExecPath::Parallel`. There is no public `enter()` constructor: this
/// removes the TOCTOU window between "decide to parallelize" and "enter
/// the parallel region", and ensures the nested-parallel check is
/// observed as a single atomic step inside `select_exec_path()`.
///
/// While the guard is alive, the thread-local flag is set. Any nested
/// call to `select_exec_path()` or `should_parallelize()` that would
/// otherwise return `ExecPath::Parallel` will instead return
/// `ExecPath::Serial` (with `None` guard).
///
/// Dropping the guard clears the thread-local flag, allowing future
/// parallel execution on this thread.
pub(crate) struct ParallelGuard {
    // No public fields: external construction is impossible.
    // Field omitted; instances are produced only by select_exec_path().
    _private: core::marker::PhantomData<*const ()>,
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        // Clear the thread-local IN_PARALLEL flag.
    }
}
```

**关键 API 边界变化：** `ParallelGuard::enter()` 不再作为公开 API 暴露。取而代之，进入并行区域的唯一入口是 `select_exec_path()` 返回 `(ExecPath::Parallel, Some(guard))`。`parallel/` 后端在收到该 guard 后将其持有到并行区域结束即可，无需也不能再次调用 `enter()`。

**线程亲和性（thread affinity）契约：** `ParallelGuard` 有意 `!Send + !Sync`（通过 `_private: PhantomData<*const ()>` 推导）。其 `Drop` 实现清除调用线程的 thread-local `IN_PARALLEL` flag——若 guard 被 move 到 Rayon worker 线程并在 worker 上 drop，会清错线程的 TLS，破坏嵌套并行检测的正确性。因此：
- `parallel/` 后端必须保持 outer guard 在调用线程（dispatching thread）的入口函数栈帧上，直到整个 Rayon 并行区域结束。
- Rayon worker 闭包不得捕获 outer guard。
- 每个 worker 闭包内 chunk 的执行必须包裹在 `dispatch::with_parallel_worker_context` 中，使该 worker 自身的 TLS 在 chunk 执行期间观测到 `IN_PARALLEL == true`，从而让 worker 内部嵌套调用 dispatch 时正确回退串行路径。

**实现提示**：基于 thread-local `Cell<bool>` 实现，仅 dispatch 模块内部可见：

```rust,ignore
// Internal implementation sketch (not a public API commitment)
std::thread_local! {
    static IN_PARALLEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Internal-only: produces a guard, intended exclusively for use by
// select_exec_path(). Not exposed outside this module.
fn try_acquire_guard() -> Option<ParallelGuard> {
    IN_PARALLEL.with(|flag| {
        if flag.get() {
            None
        } else {
            flag.set(true);
            Some(ParallelGuard { _private: core::marker::PhantomData })
        }
    })
}
```

`ParallelContext` 是 thread-local 状态 token，由 `ParallelGuard` 内部管理，不对外暴露为独立类型。

**Worker 上下文 helper**：为支持上述线程亲和性契约，dispatch 模块同时提供以下内部 helper，供 `parallel/` 在每个 Rayon worker 闭包中使用：

```rust,ignore
/// Runs `f` while marking the current worker thread as being inside a
/// Xenon-internal parallel region.
///
/// Used by `parallel/` inside Rayon worker closures: outer `ParallelGuard`
/// stays on the dispatching thread; each worker closure wraps its chunk
/// execution in this helper so nested `select_exec_path()` calls inside
/// the worker thread correctly observe `IN_PARALLEL == true` and fall
/// back to `ExecPath::Serial`.
///
/// The helper does NOT construct or consume `ParallelGuard`. It saves
/// the previous TLS value, sets it to `true`, runs `f`, and restores
/// the previous value on drop (panic-safe via the inner `Reset` RAII).
pub(crate) fn with_parallel_worker_context<R>(f: impl FnOnce() -> R) -> R {
    IN_PARALLEL.with(|flag| {
        let previous = flag.replace(true);
        struct Reset<'a>(&'a core::cell::Cell<bool>, bool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _reset = Reset(flag, previous);
        f()
    })
}
```

**嵌套并行行为矩阵：**

| 场景                              | `select_exec_path()` 中的 guard 检查 | 返回值                                 |
| --------------------------------- | :----------------------------------: | -------------------------------------- |
| 首次进入并行区域，且阈值满足      | 获取成功                             | `(ExecPath::Parallel, Some(guard))`    |
| 已在并行区域内，再次调用          | 获取失败                             | `(ExecPath::Serial, None)` 或 `Simd/Serial` 路径（按其他条件）  |
| guard drop 后再次调用             | 获取成功                             | `(ExecPath::Parallel, Some(guard))`    |
| 库内部嵌套并行 API 调用           | 获取失败                             | `(ExecPath::Serial, None)`             |
| `feature = "parallel"` 关闭       | 不进入 guard 路径                    | `(ExecPath::Serial, None)` 或 `Simd`  |
| `len < effective_parallel_threshold` | 不进入 guard 路径                 | `(ExecPath::Simd, None)` 或 `Serial`  |

### 5.5 核心 API 函数

```rust,ignore
/// Selects the optimal execution path for an operation, atomically
/// binding "select Parallel" with "enter the parallel region".
///
/// This is the central dispatch function. All `math`, `matrix`, and
/// `reduction` modules call this to decide their execution strategy.
///
/// # Arguments
///
/// * `len` - Logical element count of the input(s).
/// * `is_contiguous` - Whether all inputs are F-order contiguous.
/// * `alignment_ok` - Caller asserts that input data pointers satisfy
///   the alignment expected by the SIMD kernel that would be selected
///   at the given length (`layout::is_aligned()`). If `false`, the SIMD
///   path may select an unaligned variant or fall back internally. This
///   is a HINT, not a hard precondition; the SIMD backend
///   (`08-simd.md §5.7`) makes the final per-kernel admission decision.
///
/// # Returns
///
/// A pair `(ExecPath, Option<ParallelGuard>)` where:
///
/// * `(ExecPath::Parallel, Some(guard))` — the parallel path was
///   selected **and** the nested-parallel check has been performed
///   atomically inside this function. The caller must keep `guard`
///   alive for the entire parallel region.
///
/// * `(ExecPath::Simd, None)` — the SIMD path was recommended.
///   No guard is associated with SIMD execution.
///
/// * `(ExecPath::Serial, None)` — the serial path was selected
///   (default fallback). No guard.
///
/// The `Option<ParallelGuard>` is `Some` **iff** the first element
/// is `ExecPath::Parallel`. This invariant is enforced by the
/// implementation and may be relied upon by callers.
///
/// # Selection conditions
///
/// * `ExecPath::Parallel` — `feature = "parallel"` is enabled, **and**
///   the current thread is not already inside a parallel region, **and**
///   `len >= effective_parallel_threshold` (see §5.6 for non-contiguous
///   penalty), **and** the parallel threshold is not zero (see §5.6,
///   "threshold = 0" semantics).
///
/// * `ExecPath::Simd` — `feature = "simd"` is enabled, **and**
///   `is_contiguous`, **and** `len >= SIMD_THRESHOLD`,
///   **and** `ExecPath::Parallel` was not chosen.
///
///   **Note:** `alignment_ok` is **NOT** a
///   hard precondition for `ExecPath::Simd`. It is propagated to the
///   `simd` backend as a kernel-capability hint; the SIMD kernel
///   itself decides whether to dispatch to an aligned or unaligned
///   variant based on its `08-simd.md §5.7` admission rules. Most
///   element-wise kernels accept unaligned input. Forcing alignment
///   as a hard gate here would incorrectly close legal unaligned
///   SIMD paths.
///
/// * `ExecPath::Serial` — otherwise.
///
/// # Behavior under feature gates
///
/// - Without `feature = "parallel"`: first element is never `ExecPath::Parallel`
/// - Without `feature = "simd"`: first element is never `ExecPath::Simd`
/// - With both disabled: always returns `(ExecPath::Serial, None)`
///
/// # Determinism
///
/// Given the same inputs, the same global threshold state, and the
/// same thread-local parallel-region state, this function always
/// returns the same `ExecPath`. It is not a pure function in the
/// strict sense (it reads thread-local state and may transition the
/// "in-parallel" flag), but it is deterministic with respect to its
/// observable inputs and produces no other side effects.
///
/// # Atomicity
///
/// The decision "select Parallel" and the action "enter parallel
/// region" are performed as a single critical section against the
/// thread-local flag. There is no observable intermediate state
/// where another caller on the same thread could see "Parallel
/// selected but not yet entered".
pub(crate) fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool,
) -> (ExecPath, Option<ParallelGuard>);

// ---
// Op-agnostic boundary: select_exec_path is purely a hardware/
// path arbitration function. It does NOT consider operation semantics
// (element type, integer overflow, ordering equivalence, NaN propagation).
// Callers responsible for op-level legality MUST gate before calling this
// function — for example, integer reductions that require chunk-order
// arbitration must call this only when the calling module has decided
// parallel routing is acceptable. See 09-parallel.md §6.5 and
// 13-reduction.md §6.3 for examples of caller-side gating.
// ---

/// Quick boolean query for "should I use parallel?"
///
/// Used only by callers that need to **observe** whether parallel
/// execution would be selected, without committing to it. This
/// function does **not** acquire a `ParallelGuard`.
///
/// Returns `true` iff `select_exec_path(len, is_contiguous, _).0`
/// would return `ExecPath::Parallel`. The `alignment_ok` parameter
/// is irrelevant for this query: alignment only affects SIMD
/// eligibility, not parallel.
///
/// **Important:** if a caller observes `should_parallelize() == true`
/// and then calls `select_exec_path()`, the second call may still
/// return `ExecPath::Serial` if another caller on the same thread
/// entered a parallel region between the two queries. This function
/// is intended for diagnostics and should-not-be-used-for-control-flow
/// scenarios; for actual execution dispatch, use `select_exec_path()`
/// directly.
pub(crate) fn should_parallelize(len: usize, is_contiguous: bool) -> bool;
```

### 5.6 阈值配置

**编译期默认值：**

| 阈值参数              | 默认值  | 适用场景                               |
| --------------------- | :-----: | -------------------------------------- |
| `PARALLEL_THRESHOLD`  |  65536  | f32 / f64 逐元素及归约的并行入口下限  |
| `SIMD_THRESHOLD`      |    64   | f32 / f64 逐元素运算的 SIMD 入口下限   |

以上默认值与 `08-simd.md` §5.7 所述一致。归约（`sum`/`dot`）的 SIMD 阈值更高（1024 / 512），但该阈值由 `simd/` 模块内部管理，不属于 dispatch 裁决范围（dispatch 仅依据调用方传入的 `len` 和 `alignment_ok` 做统一推荐）。

**内部覆写接口（pub(crate)）：**

```rust,ignore
/// Override the parallel threshold at runtime.
///
/// Intended for internal testing and benchmarking.
///
/// # Special value: 0
///
/// Setting `threshold = 0` is interpreted as **"disable the parallel
/// path entirely"**, NOT as "always parallelize". This is the only
/// sentinel meaning of zero. After this call, `select_exec_path()`
/// will never return `ExecPath::Parallel` regardless of `len`, until
/// `set_parallel_threshold(non_zero)` or `reset_parallel_threshold()`
/// is called.
pub(crate) fn set_parallel_threshold(threshold: usize);

/// Reset the parallel threshold to its compile-time default.
pub(crate) fn reset_parallel_threshold();

/// Override the SIMD threshold at runtime.
///
/// Intended for internal testing and benchmarking. Provides symmetric
/// override capability with `set_parallel_threshold` since `SIMD_THRESHOLD`
/// is already stored in an `AtomicUsize`.
///
/// # Special values
///
/// SIMD admission uses the comparison `len >= SIMD_THRESHOLD`. Therefore:
///
/// - `threshold = 0`: comparison `len >= 0` is always true. Combined with
///   the other gates (`is_contiguous`, feature flag), this **lowers the
///   length floor to its minimum** but does NOT disable SIMD. Use this
///   value to force SIMD admission for benchmarking small inputs.
///
/// - `threshold = usize::MAX`: comparison `len >= usize::MAX` is reachable
///   only when `len == usize::MAX`, which is unreachable for any real
///   tensor (since `Vec::len()` is bounded by `isize::MAX` on standard
///   platforms). This is the canonical **"disable SIMD path"** sentinel.
///
/// Note: this asymmetry with `set_parallel_threshold(0)` (which disables
/// parallel) is intentional — parallel admission uses `len >= threshold &&
/// threshold != 0` per §6.3, while SIMD admission uses plain `len >= threshold`.
/// Each path's "disable" sentinel reflects its own admission predicate.
pub(crate) fn set_simd_threshold(threshold: usize);

/// Reset the SIMD threshold to its compile-time default.
pub(crate) fn reset_simd_threshold();
```

**非连续策略（统一规则）：** SIMD 路径与并行路径采取不同规则——

| 输入条件          | 是否考虑 `Simd`           | 是否考虑 `Parallel`                                  |
| ----------------- | ------------------------- | ---------------------------------------------------- |
| 连续 + 对齐       | 是（`len >= SIMD_THRESHOLD`），`alignment_ok` 作为能力提示传入 simd 后端 | 是（`len >= PARALLEL_THRESHOLD`）                |
| 连续 + 非对齐     | **是**（连续 + `len >= SIMD_THRESHOLD` 即可进入；`alignment_ok = false` 转为传给 simd 的 unaligned-kernel 提示，由 `08-simd.md §5.7` 决定具体 kernel 选择） | 是（`len >= PARALLEL_THRESHOLD`）            |
| 非连续            | **否**（连续性仍是 SIMD 准入硬性条件） | 是，但 `len >= 2 * PARALLEL_THRESHOLD`（饱和乘法防溢出） |

**三条规则的差异有意为之：**

- **SIMD 连续性**：`is_contiguous == false` 时**直接拒绝**进入 `Simd` 路径——SIMD 后端要求连续输入，无法通过单纯放宽阈值满足。
- **SIMD 对齐**：`alignment_ok == false` 时**不拒绝**进入 `Simd` 路径——`alignment_ok` 仅作为能力提示位透传到 simd 后端，由 `08-simd.md §5.7` 的 admission 规则决定走 aligned 或 unaligned kernel；多数逐元素 kernel 默认接受 unaligned 输入（与 `08-simd.md` 协同）。
- **Parallel 非连续**：`is_contiguous == false` 时**不拒绝**，但通过**有效阈值翻倍**抑制进入——非连续的并行 worker 仍可执行（只是缓存局部性差），收益曲线由翻倍阈值近似补偿。

**溢出保护：** `effective_parallel_threshold = base.saturating_mul(2)`。当 `set_parallel_threshold` 被设为接近 `usize::MAX` 的值时，饱和乘法把翻倍结果钉在 `usize::MAX`，从而 `len >= effective` 永不成立——同样落到 `Serial`。这避免任何 wrap-around 导致的非确定性路径选择。

### 5.7 Guard API

`ParallelGuard` 没有公开（pub/pub(crate)）的构造函数。
进入并行区域的唯一入口是 `select_exec_path()` 返回 `(ExecPath::Parallel, Some(guard))`。
该设计让调用方拿到的 guard 与"select 到 Parallel 的决策"严格对应，
消除了"先 select 到 Parallel，再 enter() 失败回退"的无效中间状态。

`Drop` 实现释放 thread-local flag，是 `ParallelGuard` 唯一对外可观察的行为。

### 5.8 路径选择阈值（分操作类型参考）

dispatch 持有的阈值适用于**所有操作类型**的统一入口裁决。各操作的具体 SIMD 阈值差异由 `simd/` 内部处理。以下表格仅供参考说明各模块的总体阈值策略，**dispatch 自身只持有两个统一阈值**（`PARALLEL_THRESHOLD = 65536`、`SIMD_THRESHOLD = 64`），不感知操作类型；**具体 per-op 阈值由 `simd/` 后端在 `ExecPath::Simd` 被选中后执行最终 admission 时裁决**（与下方"调用方-dispatch-后端的阈值分工"以及 `08-simd.md §5.6` "条件实现，默认标量回退" 一致；调用方**不**做 per-op **长度阈值** gating，但**必须**做 op-语义 gating——参见 `08-simd.md §5.6.1` / `09-parallel.md §6.5` 整数 checked 等价性等场景）：

| 操作类型       | 元素类型                        | 并行最小长度 | SIMD 最小长度 | 说明                                   |
| -------------- | ------------------------------- | :----------: | :-----------: | -------------------------------------- |
| 逐元素算术     | `f32` / `f64`                   |    65536     |      64       | 连续 + 对齐时优先 SIMD                 |
| 逐元素算术     | `Complex<f32>` / `Complex<f64>` |    65536     |     128       | AoS 输入的 SIMD 阈值高于实数路径       |
| 逐元素算术     | `i32` / `i64`                   |    65536     |      64       | 与浮点逐元素一致                       |
| 归约 `sum`     | `f32` / `f64`                   |    65536     |    1024       | SIMD 归约阈值更高（由 simd/ 内部裁决） |
| 归约 `sum`     | `i32` / `i64`                   |    65536     |     512       | 整数 widening accumulator              |
| 内积 `dot`     | `f32` / `f64`                   |    65536     |     512       | 同归约                                 |
| 内积 `dot`     | `i32` / `i64`                   |    65536     |     256       | 同归约                                 |

**调用方-dispatch-后端的阈值分工：**

- **dispatch 持有**：通用最小阈值（`PARALLEL_THRESHOLD`、`SIMD_THRESHOLD`），用于"是否值得进入非标量路径"的粗粒度裁决。调用方**不得**基于通用长度阈值（`len`）在调用 `select_exec_path()` 之前自行 gate（即不得绕过 dispatch 自行做 `Serial` 长度短路）。
- **调用方持有两类职责**：
  1. **通用元数据传入**：`len` / `is_contiguous` / `alignment_ok` 直接传入 `select_exec_path()`，不持有 op-/element-type-specific **长度阈值**。
  2. **op-语义合法性 gating（必须）**：调用方**必须**基于操作语义合法性在调用 `select_exec_path()` 之前 gate，包括但不限于：
     - 整数路径 SIMD/Parallel checked-arithmetic 等价性缺失（如 `sum<i32>` / `dot<i32>` 在某后端无 checked SIMD widening kernel）→ 调用方直接走 Serial，不调用 `select_exec_path()`；详见 `08-simd.md §5.6.1` / `09-parallel.md §6.5` / `13-reduction.md §6.3` / `12-matrix.md §6.1`。
     - 顺序敏感约束（如归约要求确定性 chunk-order）→ 调用方裁定是否进入 Parallel；详见 `09-parallel.md §6.5`。
     - 元素类型在该 op 下被显式排除（如 `bool` 不进入 `sum`）→ 调用方在类型层或调用前直接拒绝。
  
  这两类职责的边界是：**通用长度阈值**由 dispatch 持有，调用方不得绕过；**op-语义合法性**由调用方持有，dispatch 不感知。
- **`simd/` 后端持有**：op-/element-type-specific 阈值（如归约 `sum_f64` 的 SIMD 准入阈值 1024）、lane 宽度、ISA 可用性、操作覆盖矩阵等最终准入条件。`simd/` 在 `ExecPath::Simd` 被选中后执行**最终二次 admission**——通过则走 SIMD kernel，不通过则内部回退标量。这与 `08-simd.md §5.6` "条件实现，默认标量回退" 与本文 §9.4 "simd 持有最终 admission" 严格一致。

非连续与对齐处理见 §5.6。

---

## 6. 内部实现设计

### 6.1 select_exec_path 算法

```
Algorithm: select_exec_path(len, is_contiguous, alignment_ok)

Input:
    len            : usize   — total logical element count
    is_contiguous  : bool    — all inputs are F-order contiguous
    alignment_ok   : bool    — all inputs satisfy SIMD alignment fast-path

Output:
    (ExecPath, Option<ParallelGuard>)
    // Invariant: second element is Some(_) iff first element == ExecPath::Parallel

Steps:
    1. base_parallel_threshold = get_parallel_threshold()

       // "threshold == 0" sentinel: explicit parallel disable
       if base_parallel_threshold == 0:
           parallel_eligible_by_threshold = false
       else:
           // Non-contiguous penalty via saturating multiplication
           // (avoids wrap-around on extreme threshold values)
           if is_contiguous:
               effective_parallel = base_parallel_threshold
           else:
               effective_parallel = base_parallel_threshold.saturating_mul(2)
           parallel_eligible_by_threshold = (len >= effective_parallel)

    2. // Try parallel first: select-and-enter atomically
       if cfg!(feature = "parallel")
          AND parallel_eligible_by_threshold:
              if let Some(guard) = try_acquire_guard():
                  // Acquired the thread-local in-parallel flag.
                  // Selection and entry happen as a single critical section;
                  // no other thread-local observer can see "Parallel selected
                  // but not yet entered".
                  return (ExecPath::Parallel, Some(guard))
              // try_acquire_guard() returned None: thread is already
              // inside a parallel region. Fall through to SIMD/Serial.

    3. // Check SIMD eligibility (no guard involved)
       //
       // alignment_ok is NOT a hard gate here. It is propagated to the
       // simd backend as a kernel-capability hint; the simd kernel itself
       // (per 08-simd.md §5.7 admission rules) decides whether to use
       // aligned or unaligned variants. Most element-wise kernels accept
       // unaligned input, so forcing alignment here would wrongly close
       // legal SIMD paths.
       if cfg!(feature = "simd")
          AND is_contiguous
          AND len >= get_simd_threshold():
              return (ExecPath::Simd, None)

    4. // Default fallback
       return (ExecPath::Serial, None)
```

**优先级说明：** 并行检查在 SIMD 之前。这是因为在同时启用两个 feature 且输入足够大的场景下，并行路径的吞吐收益通常高于 SIMD 串行加速。若并行不可用或输入未达并行阈值，再考虑 SIMD。

**Guard 与路径绑定的关键不变量：**

- Step 2 中 `try_acquire_guard()` **要么**返回 `Some(guard)` 并设置 thread-local flag、并立即返回 `(Parallel, Some(guard))`；**要么**返回 `None` 并落到 Step 3/4。中间没有其他状态。
- 一旦 `Some(guard)` 被产生，guard 必然作为返回值的一部分被调用方持有；guard 永不在 dispatch 内部被丢弃。

**确定性保证：** 该算法在以下输入集合相同时确定性：`(len, is_contiguous, alignment_ok, threshold_state, feature_flags, thread_local_in_parallel_state)`。thread-local 的 `IN_PARALLEL` 状态显式被纳入"输入"，因为嵌套并行场景下相同的 `(len, contig, align)` 在不同 thread-local 状态下会有不同结果。这是正确语义。

### 6.2 ParallelGuard 实现

```rust,ignore
use std::cell::Cell;

std::thread_local! {
    /// Thread-local flag indicating whether the current thread is
    /// executing inside a library-internal parallel region.
    static IN_PARALLEL: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct ParallelGuard {
    _private: core::marker::PhantomData<*const ()>,
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        IN_PARALLEL.with(|flag| {
            flag.set(false);
        });
    }
}

/// Module-private: try to acquire the in-parallel flag and, on success,
/// produce a guard whose Drop will release it.
///
/// This is the **only** way a `ParallelGuard` value comes into existence.
/// It is intentionally not exposed as a `pub(crate)` API; it is wired into
/// `select_exec_path()` as the single atomic site for selection-and-entry.
fn try_acquire_guard() -> Option<ParallelGuard> {
    IN_PARALLEL.with(|flag| {
        if flag.get() {
            None
        } else {
            flag.set(true);
            Some(ParallelGuard { _private: core::marker::PhantomData })
        }
    })
}

/// Query-only: check if currently in parallel region without setting the flag.
/// Used by `should_parallelize()` for diagnostic queries that do not commit.
fn is_in_parallel() -> bool {
    IN_PARALLEL.with(|flag| flag.get())
}
```

**安全性论证：**

| 主题          | 论证                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------- |
| thread-local      | `Cell<bool>` 是 `!Sync`，但通过 `thread_local!` 访问确保每个线程拥有独立副本，无数据竞争。        |
| RAII 保证         | `ParallelGuard` 的 `Drop` 实现在任何退出路径（包括 panic unwind）下都会重置 flag，不会泄漏状态。  |
| 原子选择          | `try_acquire_guard()` 内部先 `flag.get()` 再 `flag.set(true)`；这两步在单线程内是顺序执行，无 TOCTOU。 |
| 公共入口收敛      | `try_acquire_guard()` 是模块私有函数，仅由 `select_exec_path()` 调用。`ParallelGuard` 没有 `pub(crate)` 构造函数，杜绝调用方绕过原子契约。 |
| 线程亲和性        | `ParallelGuard` 是 `!Send + !Sync`，因为 `Drop` 清除当前线程的 TLS——若 guard 被 drop 在另一线程会清错线程的 flag。`parallel/` 后端必须在调用线程持有 guard，禁止在 Rayon 闭包中捕获。 |
| Worker 上下文     | Rayon worker 闭包必须使用 `with_parallel_worker_context` 在 chunk 执行期间设置 worker 自身 TLS；不得捕获或 move outer guard。 |
| 零分配            | 不在堆上分配，不涉及原子操作（`Cell` 是非原子内部可变性），开销为单次 thread-local 访问。         |
```

### 6.3 阈值存储

```rust,ignore
use std::sync::atomic::{AtomicUsize, Ordering};

/// Compile-time default for parallel threshold.
const DEFAULT_PARALLEL_THRESHOLD: usize = 65536;

/// Runtime-overridable parallel threshold.
///
/// Uses `AtomicUsize` for lock-free reads. Written only during
/// initialization or explicit override (testing/benchmarking).
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// Compile-time default for SIMD threshold.
const DEFAULT_SIMD_THRESHOLD: usize = 64;

/// Runtime-overridable SIMD threshold.
///
/// Uses `AtomicUsize` for lock-free reads, symmetric with
/// `PARALLEL_THRESHOLD`. Written only during initialization or
/// explicit override (testing/benchmarking).
static SIMD_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_SIMD_THRESHOLD);

#[inline]
fn get_parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
fn get_simd_threshold() -> usize {
    SIMD_THRESHOLD.load(Ordering::Relaxed)
}

pub(crate) fn set_parallel_threshold(threshold: usize) {
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

pub(crate) fn reset_parallel_threshold() {
    PARALLEL_THRESHOLD.store(DEFAULT_PARALLEL_THRESHOLD, Ordering::Relaxed);
}

pub(crate) fn set_simd_threshold(threshold: usize) {
    SIMD_THRESHOLD.store(threshold, Ordering::Relaxed);
}

pub(crate) fn reset_simd_threshold() {
    SIMD_THRESHOLD.store(DEFAULT_SIMD_THRESHOLD, Ordering::Relaxed);
}
```

**Ordering 选择理由：** `Relaxed` 足够——阈值只在初始化/测试覆写时变更一次；读侧（`select_exec_path` 热路径）不需要与任何其他原子变量同步，仅需读到某个合法值即可。不存在需要 `Acquire/Release` 语义的 paired 操作。

### 6.4 Feature Gate 处理

```rust,ignore
// Compile-time feature detection for variant elimination.
//
// When `feature = "parallel"` is absent, the compiler eliminates
// the entire parallel-eligibility branch as dead code; the
// `try_acquire_guard()` call site is removed.
//
// When `feature = "simd"` is absent, the SIMD branch is eliminated.

pub(crate) fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool,
) -> (ExecPath, Option<ParallelGuard>) {
    // Threshold == 0 sentinel: explicit parallel disable
    let base = get_parallel_threshold();
    let parallel_eligible_by_threshold = if base == 0 {
        false
    } else {
        let effective = if is_contiguous {
            base
        } else {
            base.saturating_mul(2) // overflow-safe doubling
        };
        len >= effective
    };

    // Parallel check: select-and-enter atomically
    #[cfg(feature = "parallel")]
    {
        if parallel_eligible_by_threshold {
            if let Some(guard) = try_acquire_guard() {
                return (ExecPath::Parallel, Some(guard));
            }
            // Acquisition failed (already in parallel region): fall through.
        }
    }

    // SIMD check: compiled away when feature is absent
    //
    // alignment_ok is consumed by the simd backend as a capability hint,
    // not as a dispatch-side hard gate. See §5.5 / §5.6.
    #[cfg(feature = "simd")]
    {
        if is_contiguous && len >= get_simd_threshold() {
            // alignment_ok is forwarded to the simd backend as a
            // kernel-capability hint (see §5.5 for the full contract
            // of this parameter). simd internally decides aligned vs
            // unaligned dispatch per 08-simd.md §5.7.
            let _simd_alignment_hint = alignment_ok;
            // (Forwarded to simd backend — the implementor wires
            // this hint into the kernel selector. Not ignored.)
            return (ExecPath::Simd, None);
        }
    }

    (ExecPath::Serial, None)
}

#[cfg(feature = "parallel")]
pub(crate) fn should_parallelize(len: usize, is_contiguous: bool) -> bool {
    let base = get_parallel_threshold();
    if base == 0 || is_in_parallel() {
        return false;
    }
    let effective = if is_contiguous {
        base
    } else {
        base.saturating_mul(2)
    };
    len >= effective
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn should_parallelize(_len: usize, _is_contiguous: bool) -> bool {
    false
}
```

**关于 `alignment_ok` 的实际传递路径**：本伪代码中 `_simd_alignment_hint` 仅在 dispatch 选路阶段使用，作为"是否进入 SIMD 路径"的早期短路提示，并不会通过参数或数据结构显式 forward 给 SIMD 后端。SIMD 后端在 chunk 内部独立通过 `layout::is_aligned()` 重新检查实际指针对齐情况，并在 per-kernel admission 阶段（见 08-simd.md §5.7）选择 aligned 或 unaligned 变体。因此 `alignment_ok = false` 不会硬性禁止 SIMD 路径，仅作为 dispatch 阶段的优化启发；最终对齐准入由 SIMD 后端独立裁决。

**`ParallelGuard` 的 cfg 处理：** `ParallelGuard` 类型与 `try_acquire_guard()` / `is_in_parallel()` / `IN_PARALLEL` thread-local 仅在 `feature = "parallel"` 启用时存在对应实现；`feature = "parallel"` 关闭时仅保留一个零大小占位结构体以保持 `select_exec_path()` 的返回类型签名稳定，但**永不构造**且**无 Drop 行为**。

```rust,ignore
#[cfg(not(feature = "parallel"))]
pub(crate) struct ParallelGuard {
    // Zero-size, never constructed in this build configuration.
    //
    // Unlike the real guard under `feature = "parallel"`, this placeholder
    // has no Drop implementation and no thread-local state to release, so
    // it must be Send + Sync by construction. Otherwise
    // `(ExecPath, Option<ParallelGuard>)` would unnecessarily become non-Send
    // in default/no-parallel builds, where the option is provably always `None`.
    _private: core::marker::PhantomData<()>,
}
// No `try_acquire_guard()`, no `Drop` impl needed — the type is unconstructible.
```

**真 guard !Send vs placeholder Send 的有意不对称：** 在 `feature = "parallel"` 启用下，真 `ParallelGuard` 必须是 `!Send + !Sync`，因为它持有清除当前线程 TLS flag 的释放语义。在 `feature = "parallel"` 关闭下，placeholder `ParallelGuard` **不**持有任何线程亲和的释放语义（无构造、无 Drop），因此**必须**是 `Send + Sync`，以避免 `Option<ParallelGuard>`（始终 `None`）在默认构建下被无端打上 `!Send` 标签。这种不对称是有意为之的安全契约差异，并非疏漏。

### 6.5 决策流 ASCII 图

```
Caller (math / matrix / reduction)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  dispatch::select_exec_path(len, is_contiguous, align_ok)  │
│                                                             │
│  Returns: (ExecPath, Option<ParallelGuard>)                 │
│  Invariant: guard is Some(_) IFF ExecPath::Parallel         │
└────────────────────────┬───────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌────────────────────┐ ┌──────────────┐ ┌──────────────┐
│ feature=parallel + │ │ feature=simd │ │ default      │
│ threshold != 0 +   │ │ + contig +   │ │ fallback     │
│ len >= effective + │ │ len >= SIMD  │ │              │
│ guard acquired     │ │ threshold;   │ │              │
│ (atomic)           │ │ alignment_ok │ │              │
│                    │ │ is HINT only │ │              │
│                    │ │ (forwarded   │ │              │
│                    │ │  to simd/)   │ │              │
└──────────┬─────────┘ └──────┬───────┘ └──────┬───────┘
           │                  │                │
           ▼                  ▼                ▼
(Parallel, Some(guard)) (Simd, None)    (Serial, None)
           │                  │                │
           ▼                  ▼                ▼
┌────────────────────┐ ┌──────────────┐ ┌──────────────┐
│ parallel/ backend  │ │ simd/        │ │ caller's     │
│ keeps the guard    │ │ backend      │ │ own scalar   │
│ on the dispatching │ │ (may         │ │ impl         │
│ thread (entry-fn   │ │  internally  │ │              │
│ frame); each Rayon │ │  fall back   │ │              │
│ worker closure     │ │  to scalar)  │ │              │
│ wraps its chunk in │ │              │ │              │
│ with_parallel_     │ │              │ │              │
│ worker_context;    │ │              │ │              │
│ chunk MAY invoke   │ │              │ │              │
│ SIMD per chunk-    │ │              │ │              │
│ local admission.   │ │              │ │              │
│ Outer              │ │              │ │              │
│ guard Drop on the  │ │              │ │              │
│ dispatching thread │ │              │ │              │
│ releases TLS flag. │ │              │ │              │
└────────────────────┘ └──────────────┘ └──────────────┘
```

**关键语义：** dispatch 做出三路裁决后，调用方执行单次 `match` 分发。`ExecPath::Simd` 分支由 `simd/` 后端接管——它可能在内部因 ISA、lane 宽度等原因回退标量，这一回退对 dispatch 完全透明。`ExecPath::Parallel` 分支总是带着已获取的 guard 进入 `parallel/` 后端；后端不再自己 `enter()`，guard 始终留在调用线程的 entry function frame 中（`!Send` 不允许 move 到 worker），每个 Rayon worker 闭包内 chunk 执行被包裹在 `dispatch::with_parallel_worker_context` 中以正确传递 worker 自身 TLS 状态。

---

## 7. 实现任务拆分

### Wave 1: 骨架

- [ ] **T1**: 创建 `src/dispatch.rs` 骨架
  - 文件: `src/dispatch.rs`
  - 内容: `ExecPath` 枚举定义、`ParallelExecStrategy` 结构体、模块级文档注释
  - 测试: 编译通过
  - 前置: `tensor` 模块完成
  - 预计: 5 min

### Wave 2: 路径裁决

- [ ] **T2**: 实现 `select_exec_path()` 与 `should_parallelize()`
  - 文件: `src/dispatch.rs`
  - 内容: 三路裁决逻辑、阈值读取（含 0 sentinel 与 saturating_mul）、feature gate 分支、非连续策略、返回 `(ExecPath, Option<ParallelGuard>)`
  - 测试: `test_exec_path_serial_below_threshold`, `test_exec_path_parallel_above_threshold`, `test_exec_path_simd_when_aligned`, `test_select_returns_guard_iff_parallel`
  - 前置: T1, T4
  - 预计: 10 min

- [ ] **T3**: 实现阈值配置接口
  - 文件: `src/dispatch.rs`
  - 内容: `AtomicUsize` 阈值存储、`set_parallel_threshold()`、`reset_parallel_threshold()`、`Relaxed` ordering 注释、`threshold == 0` sentinel 文档化
  - 测试: `test_threshold_override_respected`, `test_threshold_zero_disables_parallel`, `test_threshold_saturating_mul_no_overflow`
  - 前置: T2
  - 预计: 5 min

### Wave 3: 嵌套防护

- [ ] **T4**: 实现 `ParallelGuard` 与模块私有 `try_acquire_guard()`
  - 文件: `src/dispatch.rs`
  - 内容: `thread_local! { IN_PARALLEL: Cell<bool> }`、`ParallelGuard` 类型（无公开/`pub(crate)` 构造函数）、模块私有 `try_acquire_guard()`、`is_in_parallel()` 诊断 helper、`Drop` 实现
  - 测试: `test_nested_select_falls_back_to_serial`, `test_guard_drop_releases_flag`
  - 前置: T1
  - 预计: 10 min

- [ ] **T5**: 实现 `ParallelExecStrategy::new()` 校验构造器
  - 文件: `src/dispatch.rs`
  - 内容: 字段私有化、`new()` 拒绝 `Some(0)`、`auto()` infallible 默认、字段访问器
  - 测试: `test_parallel_strategy_new_rejects_zero`, `test_parallel_strategy_new_accepts_none`
  - 前置: T1
  - 预计: 5 min

### Wave 4: 测试与验证

- [ ] **T6**: 编写 dispatch 全套单元测试
  - 文件: `src/dispatch.rs` (#[cfg(test)])
  - 内容: 各路径返回验证、阈值边界、feature gate 组合、嵌套防护、非连续惩罚
  - 测试: 见 §8.2 完整清单
  - 前置: T2, T3, T4, T5
  - 预计: 10 min

**总预计时间：** ~45 min。所有任务均在同一文件 `src/dispatch.rs` 内完成。

---

## 8. 测试计划

### 8.1 测试分类表

| 类型                    | 位置                                | 目的                                               |
| ----------------------- | ----------------------------------- | -------------------------------------------------- |
| 单元测试                | `src/dispatch.rs` (#[cfg(test)])    | 验证路径裁决、阈值、嵌套防护                       |
| 集成测试                | `tests/test_parallel.rs` 等         | 验证 dispatch 与 parallel/simd/math/reduction 协同 |
| 边界测试                | 同模块测试中标注                    | 覆盖 len=0, len=1, 恰在阈值边缘                    |
| 属性测试                | 与单元测试配套                      | 验证 idempotent 和 monotonic 不变性                |
| Feature gate / 配置测试 | `cargo test` 多 feature 组合        | 验证 feature 关闭时分支消除                        |
| 类型/编译期测试         | trait 约束 + 编译期断言             | 验证 `ExecPath` 无公开导出、`ParallelGuard` 不可外部构造 |

### 8.2 单元测试清单

| 测试函数                                          | 测试内容                                                       | 优先级 |
| ------------------------------------------------- | -------------------------------------------------------------- | ------ |
| `test_exec_path_serial_below_threshold`           | `len < threshold` 时返回 `(Serial, None)`                      | 高     |
| `test_exec_path_parallel_above_threshold`         | 连续大输入启用 parallel feature 时返回 `(Parallel, Some(_))`   | 高     |
| `test_exec_path_simd_when_aligned`                | 中等输入 + 连续 + 对齐且启用 simd feature 时返回 `(Simd, None)`| 高     |
| `test_exec_path_serial_when_noncontiguous_below_doubled_threshold` | 非连续且 len < 2*threshold 时返回 `(Serial, None)` | 高 |
| `test_select_returns_guard_iff_parallel`          | 返回值不变量：`Some(guard)` 当且仅当 `ExecPath::Parallel`     | 高     |
| `test_nested_select_falls_back_to_serial`         | 持有未 drop 的 guard 时再次 `select_exec_path()` 返回 Serial（非 Parallel）| 高 |
| `test_guard_drop_releases_flag`                   | guard drop 后再次调用 `select_exec_path()` 可重新获得 Parallel + 新 guard | 高 |
| `test_threshold_override_respected`               | `set_parallel_threshold()` 后 `select_exec_path()` 使用新阈值  | 高     |
| `test_threshold_zero_disables_parallel`           | `set_parallel_threshold(0)` 后任意 len 都不返回 Parallel       | 高     |
| `test_threshold_saturating_mul_no_overflow`       | `set_parallel_threshold(usize::MAX / 2 + 1)` 后非连续路径 saturating，永不返回 Parallel | 高 |
| `test_reset_threshold_restores_default`           | `reset_parallel_threshold()` 恢复编译期默认值                  | 中     |
| `test_no_parallel_feature_never_returns_parallel` | 未启用 `parallel` feature 时永不为 `Parallel`                  | 高     |
| `test_no_simd_feature_never_returns_simd`         | 未启用 `simd` feature 时永不为 `Simd`                         | 高     |
| `test_should_parallelize_diagnostic_does_not_acquire_guard` | `should_parallelize() == true` 后 thread-local 仍为 false；可立即 `select_exec_path()` 拿到 guard | 高 |
| `test_parallel_preferred_over_simd_for_large_input` | 同时满足并行和 SIMD 条件时返回 `(Parallel, Some(_))`         | 中     |
| `test_deterministic_same_input_same_output`       | 固定 thread-local 状态时相同输入多次调用返回结构等价的结果（含 ExecPath，guard 不参与等价比较） | 中 |
| `test_simd_rejected_when_noncontiguous`           | 非连续输入即便 `len >= SIMD_THRESHOLD` 也不返回 Simd          | 高     |
| `test_simd_allows_misaligned_hint_when_contiguous` | 连续但非对齐（`alignment_ok = false`）时，只要 `len >= SIMD_THRESHOLD` 仍应返回 `ExecPath::Simd`；`alignment_ok` 仅作为能力提示位透传给 simd 后端，由 `08-simd.md §5.7` 决定具体 kernel；dispatch 层不再以非对齐为由拒绝 SIMD 路径。任何 aligned-only kernel 的最终拒绝由 simd 后端测试覆盖，不在本测试范围内。 | 高     |
| `test_parallel_strategy_new_rejects_zero`         | `ParallelExecStrategy::new(Some(0), _)` 与 `new(_, Some(0))` 返回 `InvalidArgument` | 高 |
| `test_parallel_strategy_new_accepts_none`         | `ParallelExecStrategy::new(None, None)` 等价 `auto()`         | 中     |

### 8.3 边界测试场景

| 场景                                        | 预期行为                                                   |
| ------------------------------------------- | ---------------------------------------------------------- |
| `len = 0`                                   | 返回 `(Serial, None)`                                      |
| `len = 1`                                   | 返回 `(Serial, None)`（远低于任何阈值）                    |
| `len = PARALLEL_THRESHOLD - 1`              | 返回 `(Serial, None)` 或 `(Simd, None)`（取决于 alignment）|
| `len = PARALLEL_THRESHOLD`（恰在阈值）      | 返回 `(Parallel, Some(_))`（若连续且 feature 启用）        |
| 非连续 `len = 2 * PARALLEL_THRESHOLD - 1`   | 不满足翻倍阈值，返回 `(Serial, None)`                       |
| 非连续 `len = 2 * PARALLEL_THRESHOLD`       | 满足翻倍阈值，可能返回 `(Parallel, Some(_))`                |
| `ParallelGuard` 在 panic unwind 中          | `Drop` 仍执行，flag 被正确清除；后续 select 可重新获得 guard |
| 阈值被设为 `0`                              | `select_exec_path()` 永不返回 `Parallel`（显式禁用 sentinel） |
| 阈值被设为 `usize::MAX`                     | 连续路径下需 `len == usize::MAX` 才进 Parallel；非连续路径 `saturating_mul(2) = usize::MAX`，永不进 Parallel |
| 阈值被设为 `usize::MAX / 2 + 1`             | 连续路径下需要 `len >= usize::MAX/2+1`；非连续路径饱和到 `usize::MAX`，永不进 Parallel（验证 saturating_mul） |

### 8.4 属性测试不变量

| 不变量                                                              | 测试方法                               |
| ------------------------------------------------------------------- | -------------------------------------- |
| Idempotent（无 guard 持有时）：相同参数多次调用返回相同 ExecPath（guard 由每次调用重新获取并立即 drop） | 循环调用 + 立即 drop 验证 |
| Monotonic in `len`：若 `len1 <= len2`，则 `select_exec_path(len1, ...).0` 的"加速等级"不超过 `select_exec_path(len2, ...).0`（Serial < Simd < Parallel） | 对递增 len 序列验证 |
| Feature gate 不变：禁用 feature 时对应路径永不返回                 | 编译期 `#[cfg(not(feature = "..."))]` 测试 + 运行时断言 |
| Guard 不变：`ParallelGuard` 存活期间任何 `select_exec_path()` 调用都不返回 `(Parallel, Some(_))` | 嵌套 select 测试 |
| 返回值不变量：`select_exec_path()` 第二元素为 `Some(_)` 当且仅当第一元素为 `ExecPath::Parallel` | 全场景断言 |

### 8.5 集成测试

| 测试文件                   | 测试内容                                                                 |
| -------------------------- | ------------------------------------------------------------------------ |
| `tests/test_parallel.rs`   | dispatch 与 `parallel/` 组合的端到端路径裁决、嵌套防护验证               |
| `tests/test_matrix.rs`     | `matrix::dot()` 通过 dispatch 的三路分发与结果一致性验证（§12-matrix.md）|
| `tests/test_reduction.rs`  | `reduction::sum()` 通过 dispatch 的路径裁决与串行基线一致性              |
| `tests/test_math.rs`       | `math` 二元逐元素通过 dispatch 的并行/SIMD 路径与标量结果一致性          |

### 8.6 Feature gate / 配置测试

| 配置                     | 验证点                                                                |
| ------------------------ | --------------------------------------------------------------------- |
| 默认配置                 | 仅返回 `Serial`，dispatch 编译通过且所有单元测试 pass                 |
| 仅启用 `parallel`        | `Simd` 永不返回；`Parallel` 条件返回                                  |
| 仅启用 `simd`            | `Parallel` 永不返回；`Simd` 条件返回                                  |
| 同时启用 `simd,parallel` | 三路裁决按优先级正确；阈值与 guard 行为一致                           |
| 阈值覆写                 | 运行时 `set_parallel_threshold()` 生效且不影响其他不变量               |

### 8.7 类型边界 / 编译期测试

| 场景                                           | 测试方式                     |
| ---------------------------------------------- | ---------------------------- |
| `ExecPath` 无 `pub` 导出                       | 编译期可见性检查             |
| `ParallelGuard` 不可外部构造（无 `pub`/`pub(crate)` 构造器；唯一来源为 `select_exec_path()` 返回值） | 编译期可见性检查 + 反例编译失败测试 |
| `ParallelExecStrategy` 无 `pub` 导出，字段私有  | 编译期可见性检查             |
| `feature = "parallel"` 启用下，真 `ParallelGuard` 是 `!Send + !Sync` | compile-fail（尝试把 guard `move` 到另一线程闭包应失败）+ static 类型断言 |
| `feature = "parallel"` 关闭下，placeholder `ParallelGuard` 是 `Send + Sync`，因此 `Option<ParallelGuard>` 不应被错误打上 `!Send` 标签 | static 类型断言（`fn assert_send<T: Send>()`、`fn assert_sync<T: Sync>()`） |
| dispatch 模块在默认 feature 集（不启用 `parallel` / `simd`）下不依赖 `pulp` 或 `rayon` | `cargo tree --no-dev-deps --no-default-features` 验证。在 `--features parallel` 下 dispatch 通过 cfg 共享 rayon 依赖（用于 `current_num_threads()` 池大小校验），不引入新顶层依赖 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向           | 对方模块         | 接口/类型                                  | 约定                                                                                            |
| -------------- | ---------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 被调用（输出） | `math`           | `select_exec_path()`, `should_parallelize()` | `math` 在广播裁决后调用 dispatch 决定串行/并行路径；并行 worker 内 chunk 可独立做 SIMD admission（参见 `08-simd.md` §9.3、`09-parallel.md`） |
| 被调用（输出） | `matrix`         | `select_exec_path()`, `should_parallelize()` | `matrix::dot()` 完成 rank/shape 校验后调用 dispatch 三路分发（参见 `12-matrix.md` §6.1）         |
| 被调用（输出） | `reduction`      | `select_exec_path()`                        | `reduction::sum()` 调用 dispatch 决定执行路径（参见 `13-reduction.md` §6.3）                    |
| 被调用（输出） | `simd`（间接）   | `ExecPath::Simd`                            | dispatch 仅推荐 SIMD 路径；`simd/` 内部做最终 admission（ISA/ lane 宽度检测）                   |
| 被消费（输入） | `parallel`       | `ParallelExecStrategy`                      | dispatch 定义该类型，`parallel/` 通过 `crate::dispatch` 引用并消费其字段                        |
| 被消费（输入） | `parallel`       | `ParallelGuard`                             | `parallel/` 后端接收并持有 `select_exec_path()` 返回的 `ParallelGuard`（select-and-enter 原子绑定，不再单独调用 `enter()`；参见 §5.4 / §5.7 / §9.3 与 `09-parallel.md` §5.1） |
| 消费（输入）   | `tensor`         | `.len()`, `.is_f_contiguous()`              | 调用方在调用 dispatch 前自行查询这些元数据并传入                                               |
| 消费（输入）   | `layout`（间接） | `is_aligned()`                              | 调用方在调用 dispatch 前自行查询对齐状态并传入 `alignment_ok`                                  |

### 9.2 数据流

```
math / matrix / reduction 调用方
    │
    ├── 查询张量元数据: .len(), .is_f_contiguous(), alignment
    │
    ├── 调用 dispatch::select_exec_path(len, is_contiguous, alignment_ok)
    │       │
    │       ├── ExecPath::Serial  → 调用方自己的标量实现
    │       ├── ExecPath::Simd    → simd/ 后端（可能内部回退标量）
    │       └── ExecPath::Parallel → parallel/ 后端（各 worker chunk 可独立做 SIMD admission）
    │
    └── 返回结果（Tensor / scalar / Result），公开语义不变
```

### 9.3 与 parallel 模块的交互

| 交互项           | dispatch 职责                                   | parallel 职责                                  |
| ---------------- | ----------------------------------------------- | ---------------------------------------------- |
| 路径裁决         | dispatch 决定是否进入并行路径并获取 guard         | parallel 被选中后纯粹执行，不自行判断路径，不再调用 `enter()` |
| 阈值             | dispatch 持有权威阈值并管理覆写                 | parallel 通过 `ParallelExecStrategy` 接收参数   |
| 嵌套防护         | dispatch 在 `select_exec_path()` 内原子获取 guard | parallel 接收 guard 并持有到并行区结束         |
| `ParallelExecStrategy` | dispatch 定义该结构体并提供构造器校验   | parallel 消费 chunk_size / max_workers 字段    |
| 串行回退         | dispatch 选择 `ExecPath::Serial` 时不进入 parallel | parallel 自身不包含串行回退代码               |

**guard 传递契约：** parallel 后端函数（如 `par_map`、`par_sum`）的内部入口不再调用 `ParallelGuard::enter()`——该函数已在 dispatch 内部消失，由 `try_acquire_guard()` 模块私有 helper 取代。parallel 后端的入口签名（概念上）应当接受 guard 作为参数：

```rust,ignore
// Conceptual signature inside parallel/ backend
pub(crate) fn par_map_internal<...>(
    /* ...inputs..., */
    strategy: ParallelExecStrategy,
    _guard: ParallelGuard, // takes ownership; dropped at function exit
) -> Result<...>;
```

调用方持有 `(ExecPath::Parallel, Some(guard))` 时通过 `match` 把 guard 转交给 parallel 后端入口；parallel 后端拥有 guard 直到工作完成，guard 在函数退出（含 panic unwind）时 `Drop` 释放 thread-local flag。

**线程亲和性约束：** 由于 `ParallelGuard` 是 `!Send + !Sync`（见 §5.4 / §6.2），parallel 后端**必须**把 guard 保留在调用线程的 entry function frame 中（即 `par_map_internal` 的栈帧），**不得**把 guard 捕获进 Rayon worker 闭包。每个 worker 闭包内 chunk 执行必须包裹在 `dispatch::with_parallel_worker_context(|| { ... })` 中——该 helper 不构造、不消费 guard，仅在 worker 自身 TLS 上设置/还原 `IN_PARALLEL == true`，使 worker 内部嵌套调用 `select_exec_path()` 正确回退串行路径。

参见 `09-parallel.md` §6.1（路径选择）。

### 9.4 与 simd 模块的交互

| 交互项       | dispatch 职责                                               | simd 职责                                                   |
| ------------ | ----------------------------------------------------------- | ----------------------------------------------------------- |
| ISA 检测     | dispatch **不参与** ISA 检测                                | `pulp::Arch` 做 ISA 检测与缓存（参见 `08-simd.md` §5.4）    |
| 路径推荐     | dispatch 通过 `ExecPath::Simd` 推荐 SIMD 路径               | simd 接收推荐后可自行拒绝（内部回退标量）                    |
| 准入条件     | dispatch 仅检查 len 与 F-连续性；`alignment_ok` 仅作为 hint 透传给 simd，不作为硬门槛 | simd 内部检查元素类型、ISA lane 宽度、操作支持矩阵；自行决定 aligned vs unaligned kernel 分发 |
| 长度阈值     | dispatch 持有 SIMD 通用阈值（64）                           | simd 持有操作特定阈值（如 sum 的 1024，参见 `08-simd.md` §5.7）|
| 调用方式     | dispatch 不直接调用 simd 代码                               | 语义模块在 `match ExecPath::Simd` 分支中调用 simd 后端       |

dispatch 与 simd 之间是**推荐-接受**关系，而非命令-执行关系。dispatch 说"SIMD 路径可能合适"，simd 说"我能做"或"我回退"。这种分层避免 dispatch 理解 ISA 细节。

---

## 10. 错误处理与语义边界

| 主题              | 说明                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| Recoverable error | dispatch 大部分 API **不产生**可恢复错误。所有路径裁决条件不满足时静默回退 `Serial`。`ParallelExecStrategy::new()` 是唯一例外：构造非法策略时返回 `XenonError::InvalidArgument`（参数不符合 §5.3 字段不变量）。 |
| Panic             | dispatch 在正常操作中永不 panic。唯一可能的 panic 场景是内部不变量违反（如 `AtomicUsize` 存储逻辑错误），属于实现 bug。`saturating_mul(2)` 显式避免了非连续阈值翻倍的 wrap-around 风险。 |
| 路径一致性        | `select_exec_path()` 在固定的输入集合（含 thread-local 状态）下确定性。详见 §6.1 与 §12.3。                |
| 语义透明          | 调用方从 dispatch 获得的只是路径推荐；无论走哪条路径，公开 API 的 shape、错误类别和数值语义必须一致。       |
| Guard 语义        | `ParallelGuard` 由 `select_exec_path()` 在选中并行路径时原子产生；调用方收到 `(Parallel, None)` 是不可能的，"回退 Serial"通过 `(Serial, None)` 显式表达。 |

---

## 11. 设计决策记录

### 决策 1：三路 ExecPath（Serial / Simd / Parallel）

| 属性     | 值                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| 决策     | `ExecPath` 枚举包含三个变体：`Serial`、`Simd`、`Parallel`。调用方通过单次 `match` 分发。                    |
| 理由     | 调用方（`math`/`matrix`/`reduction`）需要区分三种互斥执行策略；三路枚举比两层 `Option` 或独立布尔值更清晰，且 match 强制穷尽检查，编译器可做分支优化。 |
| 替代方案 | 两层 `Option<ParallelPath>` + `Option<SimdPath>` —— 放弃，语义模糊，易出现不完整分支覆盖。                  |
| 替代方案 | 四路（含 `SimdParallel`） —— 放弃，三路枚举已覆盖路径选择。注：架构允许并行 worker 内做 SIMD admission（08-simd.md、09-parallel.md），但仍由 `simd/` 在 worker chunk 内自治判定，不需要在 `ExecPath` 顶层新增 `SimdParallel` 变体；dispatch 只返回 `Parallel`，SIMD 进入由 worker 自决。 |
| 来源     | **用户决策 B**：`ExecPath` 有三个变体。                                                                     |

### 决策 2：dispatch 是 ISA-agnostic

| 属性     | 值                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 决策     | dispatch.rs 不参与 ISA 检测或 SIMD 能力判定。ISA 检测、lane 宽度选择、对齐细节均保留在 `simd/` 模块内部。                 |
| 理由     | ISA 检测属于 SIMD 后端实现细节；将其隔离在 `simd/` 内部保持 dispatch 简单、可测试、无平台依赖。dispatch 只需要知道"输入是否连续+对齐"。 |
| 替代方案 | dispatch 调用 `pulp::Arch::new()` 做 ISA 检测 —— 放弃，会引入 `pulp` 依赖到 dispatch（违反零依赖原则）、模糊职责边界。  |
| 来源     | 用户决策：dispatch 是 ISA-agnostic；ISA 检测是 `pulp::Arch` 的职责。                                               |

### 决策 3：嵌套并行检测通过 thread-local + RAII guard

| 属性     | 值                                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------------- |
| 决策     | 使用 `thread_local! { Cell<bool> }` 配合 `ParallelGuard` RAII 实现嵌套并行检测。                                 |
| 理由     | `需求说明书 §9.2` 明确禁止库内部二次并行。thread-local 检测零分配、零同步开销；RAII 保证异常安全（panic unwind 时正确释放）。 |
| 替代方案 | 全局 `AtomicBool` —— 放弃，无法区分不同线程；且 `ParallelGuard` 必须是 per-thread 的。                            |
| 替代方案 | 将嵌套并行视为可恢复错误 —— 放弃，会污染公开 API 语义，且违背"静默回退"设计原则。                                |

### 决策 4：编译期默认阈值 + 可选内部运行时覆写

| 属性     | 值                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------- |
| 决策     | 并行阈值持有编译期默认值（65536），并通过 `AtomicUsize` 允许内部运行时覆写（`pub(crate)`）。        |
| 理由     | 满足 `需求说明书 §9.2` 对"须支持并行阈值配置"的要求；编译期默认值维持稳定基线；运行时覆写满足测试/基准需求。 |
| 替代方案 | 仅保留不可配置的固定常量 —— 放弃，违反需求。                                                        |
| 替代方案 | 每次调用显式传参 —— 放弃，会让 `select_exec_path()` 签名膨胀且与现有调用方（`math`/`matrix`/`reduction`）不一致。 |

### 决策 5：非连续惩罚——有效阈值翻倍

| 属性     | 值                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 决策     | 非连续输入时，Parallel 有效阈值翻倍；SIMD 直接拒绝非连续输入。                           |
| 理由     | 非连续输入缓存局部性差；在输入不够大时进入并行路径的收益可能为负（调度开销 > 加速）。Parallel 通过翻倍阈值保守抑制，SIMD 则因连续性是硬性准入条件而直接拒绝。 |
| 替代方案 | 完全不考虑连续性 —— 放弃，会在小规模非连续输入上引起性能退化。                           |
| 替代方案 | 更复杂的性能模型（如 stride 模式分析）—— 放弃，超出当前版本范围且增加维护成本。           |

### 决策 6：单文件模块（非子目录）

| 属性     | 值                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------ |
| 决策     | dispatch 保持为单一文件 `src/dispatch.rs`，不拆分为 `src/dispatch/` 子目录。                      |
| 理由     | dispatch 表面积极小且内聚性极高（≤200 行实现代码）。单文件减少模块边界样板代码、简化依赖声明。    |
| 替代方案 | 拆分为 `dispatch/` 子目录（如 `path.rs` + `guard.rs` + `threshold.rs`）—— 放弃，对当前规模过度工程化。 |
| 触发条件 | 若未来 dispatch 膨胀到 >500 行或引入新的独立关注点（如更细粒度的 per-op 阈值配置），可重新评估。  |

### 决策 7：select-and-enter 原子绑定

| 属性     | 值                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 决策     | `select_exec_path()` 返回 `(ExecPath, Option<ParallelGuard>)`；当且仅当首元素为 `ExecPath::Parallel` 时第二元素为 `Some(_)`。`ParallelGuard` 没有任何公开/`pub(crate)` 构造函数；唯一产生途径是 dispatch 模块内部的 `try_acquire_guard()`。 |
| 理由     | 旧设计中 `select_exec_path()` 只返回 `ExecPath`，调用方再单独 `enter()`。这造成两个问题：(a) C6 矛盾——三处文档（§5.4 / §6.1 / §6.4）对"select 是否 consume guard"表述不一致；(b) TOCTOU 窗口——select 返回 Parallel 与 backend 调用 enter() 之间，另一并行 API 可能抢先进入 parallel region，使 backend 实际执行嵌套并行。把 select 与 acquire 绑定为单次 thread-local 临界区根本性消除这两个问题。 |
| 替代方案 | 保留独立 `enter()` 并通过文档约束调用方"必须立刻 enter"——放弃，约束无法在类型系统层面强制，且不消除 TOCTOU。 |
| 替代方案 | `select_exec_path()` 返回 `Result<ParallelGuard, ExecPath>` 风格——放弃，过度工程化且让 Serial/SIMD 路径退化为"错误"，语义反直觉。 |
| 来源     | **用户决策 B8.a**（"select_exec_path() 返回 (ExecPath, Option<ParallelGuard>)"）。                                       |

### 决策 8：`set_parallel_threshold(0)` 为显式禁用 sentinel

| 属性     | 值                                                                                                              |
| -------- | --------------------------------------------------------------------------------------------------------------- |
| 决策     | `set_parallel_threshold(0)` 唯一含义为"禁用并行路径"，永不解释为"对所有 len 都启用"。                              |
| 理由     | C7 指出`len >= effective_parallel_threshold` 在 threshold == 0 时会让所有 len（含 0）满足并行条件，与 §5.6 文字"effectively disables the parallel path"直接矛盾。把 0 作为显式 sentinel 一次性闭合该语义，且对测试/基准最有用——禁用而非穷尽启用。 |
| 替代方案 | 用 `Option<usize>` 表达启用/禁用——放弃，与 `AtomicUsize` 存储不兼容；引入 `usize::MAX` 当作 disable 又无法与"巨大但非禁用"区分。 |

---

## 12. 性能考量

### 12.1 开销分析

| 开销点                    | 每调用成本              | 说明                                                                     |
| ------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `select_exec_path()`      | ~3 comparisons + 2 loads | 两次 `AtomicUsize::load(Relaxed)` + 一次 `Cell::get()`（仅并行分支）      |
| 阈值读取                 | 单次 Relaxed atomic load | 无锁，无竞争；Relaxed 在 x86 上等同于普通 load。                          |
| guard 获取（select 内部） | 一次 thread-local 访问   | `select_exec_path()` 内部 `Cell::get()` + `Cell::set()`，纳秒级；不再暴露独立的 `ParallelGuard::enter()` API |
| ParallelGuard::drop()    | 一次 thread-local 访问   | `Cell::set(false)`，纳秒级                                                |

**总体评估：** 每次 `select_exec_path()` 调用的总开销 < 10 ns（在典型 x86_64 硬件上）。相比之下，被派发的操作本身（如 1M 元素的 `add`）耗时在微秒至毫秒级，dispatch 开销可忽略不计。

### 12.2 零成本抽象

| 场景                              | 编译结果                                                                    |
| --------------------------------- | --------------------------------------------------------------------------- |
| `feature = "parallel"` 关闭       | 并行分支被 `#[cfg]` 消除；`ParallelGuard` 相关代码不生成                    |
| `feature = "simd"` 关闭           | SIMD 分支被 `#[cfg]` 消除；`ExecPath::Simd` 变体可能被编译器标记为不可达    |
| 两者均关闭                        | `select_exec_path()` 被优化为立即返回 `ExecPath::Serial`                    |
| 阈值常量传播                      | 编译期常量在 release build 中被内联为立即数                                 |

### 12.3 缓存效应与确定性表述

- 阈值存储在 `static AtomicUsize` 中，不与其他频繁写入的变量共享 cache line，无 false sharing 风险。
- `IN_PARALLEL` thread-local 变量每个线程一份，无跨线程竞争。
- `select_exec_path()` **不是**严格意义上的纯函数。它读取两类隐式状态（全局阈值 `AtomicUsize` 与 thread-local `IN_PARALLEL` flag），并且在选中并行路径时**修改** thread-local flag。它在以下意义上确定：固定 `(len, is_contiguous, alignment_ok, threshold_state, in_parallel_state)` 时返回值确定。其结果**不可**跨调用缓存——`IN_PARALLEL` 状态会在 guard 生命周期内变化，且即便单线程内不同时刻调用 dispatch，全局阈值也可能被覆写。每操作调用一次 dispatch 的开销已经极低，无需缓存。

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `std` only | dispatch 仅使用 `std`（`AtomicUsize`、`Cell`、`thread_local!`），不依赖任何外部 crate                            |
| MSRV       | Rust 1.85+                                                                                                       |
| 单 crate   | dispatch 是 Xenon 单 crate 的内部模块，不独立发布                                                               |
| SemVer     | dispatch 全为 `pub(crate)`——无对外 SemVer 承诺。内部 API 变更不影响下游用户，但需保持与 `parallel`/`math`/`matrix`/`reduction` 的接口一致 |
| 最小依赖   | 不引入**额外**第三方 crate；`pulp` / `rayon` 仅作为 `feature = "simd"` / `feature = "parallel"` 下的可选依赖，dispatch 共享 `parallel/` 已声明的 rayon 依赖项以读取池大小（仅在 `cfg(feature = "parallel")` 块内使用 `rayon::current_num_threads()`）。非 `parallel` 编译下 dispatch 完全不引入 rayon 符号 |
| 确定性     | 同平台、同编译配置、同输入下 `select_exec_path()` 结果必须确定，不依赖随机数或时间                               |
| 不变量     | `ExecPath` 枚举为 `#[derive(Copy, Clone, Debug, PartialEq, Eq)]`，零成本传递                                     |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
