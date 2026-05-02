# 执行路径派发模块设计

> 文档编号: 30
> 模块文件: src/dispatch.rs
> 任务阶段: Phase 5
> 前置文档: 06-layout.md, 07-tensor.md, 09-parallel.md
> 需求参考: 需求说明书 §9、§13、§14、§28
> 范围声明: 范围内

---

## 1. 模块定位

dispatch 模块是 Xenon 张量库内部执行路径的统一裁决层。它负责根据输入张量的形状、连续性与对齐特征，在三条互斥执行路径——串行（`Serial`）、SIMD 加速（`Simd`）、并行（`Parallel`）——中选择最优的一条，并集中管理并行阈值配置、嵌套并行防护与执行策略参数。dispatch 本身不包含任何实际运算实现，也不参与 ISA 检测或 SIMD 能力判定。

### 1.1 职责边界

| 职责       | 包含                                                                 |
| ---------- | -------------------------------------------------------------------- |
| 路径裁决   | `ExecPath` 三路仲裁（Serial / Simd / Parallel），通过 `select_exec_path()` 统一入口 |
| 阈值管理   | 并行阈值与 SIMD 阈值的编译期默认值、内部运行时覆写与重置接口           |
| 嵌套防护   | `ParallelGuard` / `ParallelContext` 的 thread-local RAII 保护，防止库内部二次并行 |
| 策略参数   | `ParallelExecStrategy` 定义（chunk_size、max_workers），供 parallel/ 后端消费 |
| 快捷查询   | `should_parallelize()` 布尔查询，供仅关注串行/并行二选的调用方使用    |

| 职责       | 不包含                                                      |
| ---------- | ----------------------------------------------------------- |
| 路径裁决   | ISA 检测与选择（那归 `pulp::Arch`，在 `simd/` 内部完成）    |
| 阈值管理   | SIMD 最终准入判定（lane 宽度、元素类型支持等——那归 `simd/`） |
| 嵌套防护   | 串行回退实现或并行/ SIMD 执行逻辑本身（那归各语义模块与后端） |
| 策略参数   | 广播形状仲裁（那归 `math/broadcast`）                        |
| 快捷查询   | 标量实现代码（那归 `math/matrix/reduction`）                 |

### 1.2 设计原则

| 原则         | 体现                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------- |
| 单一裁决点   | 所有 `math` / `matrix` / `reduction` 模块通过 `dispatch::select_exec_path()` 进行三路裁决 |
| 二级裁决模型 | dispatch 选择三条互斥路径之一；SIMD / Parallel 后端在被选中后做内部细化                   |
| 零成本抽象   | `feature = "parallel"` 关闭时 dispatch 仍然存在，但 `ExecPath::Parallel` 永不返回；`ExecPath::Simd` 仅当 `feature = "simd"` 启用时返回 |
| 嵌套并行防护 | thread-local guard 防止库内部二次并行，失败时静默回退 `Serial` 路径                        |
| 透明回退     | 进入失败或条件不满足时静默回退到 `ExecPath::Serial`，绝不 panic 或返回错误                |

---

## 2. 需求映射与范围约束

| 类型     | 内容                                                                         |
| -------- | ---------------------------------------------------------------------------- |
| 需求映射 | 需求说明书 §9（并行需求）、§13（内积需求）、§14（归约需求）、§28（性能需求） |
| 范围内   | `ExecPath` 三路裁决、并行阈值配置、嵌套并行防护、`ParallelExecStrategy` 定义、SIMD 路径推荐 |
| 范围外   | SIMD 准入判定（ISA 检测、lane 宽度选择）、scalar implementations、广播形状仲裁 |
| 标量回退 | dispatch 自身不包含标量实现代码；回退到 `Serial` 后由各语义模块自行执行标量路径 |
| 非目标   | 不在 dispatch 模块内引入第三方依赖（包括 pulp）；不扩展 dispatch 为子目录模块 |

---

## 3. 文件位置

```
src/dispatch.rs                    # Single-file module
```

单文件设计理由：dispatch 模块表面积极小、内聚性极高——仅包含 `ExecPath` 枚举、`ParallelExecStrategy` 结构体、`ParallelGuard` / `ParallelContext`、`select_exec_path()` / `should_parallelize()` 两个核心函数以及阈值配置访问器。拆分为子目录会增加模块边界管理成本而无实际收益。

---

## 4. 依赖关系

### 4.1 依赖图（ASCII）

```
src/dispatch.rs
├── std::sync::atomic        # AtomicUsize (threshold storage)
├── std::cell::Cell          # Cell<bool> (thread-local guard)
├── crate::tensor            # .len(), .is_f_contiguous() (layout queries via tensor)
└── crate::layout            # Alignment helpers (via tensor)
```

### 4.2 类型级依赖

| 来源模块  | 使用的类型/trait                                         |
| --------- | -------------------------------------------------------- |
| `tensor`  | `TensorBase<S, D>`, `.len()`, `.is_f_contiguous()`       |
| `layout`  | `is_aligned()`（通过 tensor 暴露的查询接口间接使用）     |
| `std`     | `AtomicUsize`, `Cell<bool>`                              |

### 4.3 依赖合法性

| 项目           | 说明                                                                 |
| -------------- | -------------------------------------------------------------------- |
| 新增第三方依赖 | 无                                                                    |
| 合法性结论     | 合法；dispatch 仅使用 std 与 crate 内部既有模块，符合最小依赖原则     |
| 替代方案       | 不适用；当前设计无需额外依赖                                          |

### 4.4 依赖方向声明

依赖方向：单向向上。dispatch 处于 L4 层级（参见 `01-architecture.md` §5.2），仅消费 `tensor`/`layout` 等核心模块；被 `parallel`、`math`、`matrix`、`reduction` 等 L5 模块消费。dispatch 绝不被其下游消费方反向依赖。

---

## 5. 公共 API 设计

### 5.1 Crate 级约束

- 所有 dispatch API 均为 `pub(crate)`；不对外公开任何 dispatch 类型或函数。
- `Cargo.toml` feature 交互：dispatch.rs **始终编译**（不依赖任何 feature gate），但其运行时行为随 feature flag 变化：
  - `feature = "parallel"` 关闭时，`select_exec_path()` 永不返回 `ExecPath::Parallel`
  - `feature = "simd"` 关闭时，`select_exec_path()` 永不返回 `ExecPath::Simd`
  - 两者均关闭时，`select_exec_path()` 总是返回 `ExecPath::Serial`

### 5.2 ExecPath 枚举（核心类型）

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
    /// `simd/` retains final admission authority per Decision 2 (§11).
    Simd,

    /// Parallel execution.
    ///
    /// Caller delegates to the `parallel/` backend. Workers execute
    /// scalar code (no SIMD inside parallel workers per architectural
    /// decision). This path is only returned when `feature = "parallel"`
    /// is enabled AND the input meets the parallel threshold AND the
    /// thread is not already inside a library-internal parallel region.
    ///
    /// When this variant is returned, `select_exec_path()` also yields
    /// the corresponding `ParallelGuard` (held by the caller) — selection
    /// and entry are bound into a single atomic step. See §5.5 and
    /// Decision 7 (§11).
    Parallel,
}
```

**语义约定：** `ExecPath::Simd` 表示 dispatch **推荐** SIMD 路径；`simd/` 模块仍保有最终准入权（检测 ISA、lane 宽度、对齐细节等），并可在内部回退标量。dispatch 不参与该回退决策，也不感知其发生。

**`ExecPath::Parallel` 与 guard 的原子绑定：** 一旦 `select_exec_path()` 返回 `ExecPath::Parallel`，调用方必然拿到 `Some(ParallelGuard)`。两者**不可分离**——这是 Decision 7 的核心契约（参见 §11）。任何让调用方先收到 `ExecPath::Parallel` 再尝试单独 `enter()` 的设计都会引入 TOCTOU 窗口（中间被另一并行 API 抢占进入），与"嵌套并行检测"语义冲突。

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
    /// Returns `InvalidArgument` if `chunk_size == Some(0)` or
    /// `max_workers == Some(0)`. Upper bound on `max_workers` (≤ thread
    /// pool size) is enforced by the parallel backend at consumption
    /// time, since the pool size is a runtime value.
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
| `max_workers` | `Some(1..=pool_size)` 或 `None`   | `None` | `Some(0)` 在 `new()` 内返回 `InvalidArgument`；超过线程池大小由 `parallel/` 在运行时返回 `InvalidArgument` |
| `chunk_size`  | `Some(n)` where `n > 0` 或 `None` | `None` | `Some(0)` 在 `new()` 内返回 `InvalidArgument`                              |

**字段不变量归属：** 字段层不变量（非零）由构造器在 dispatch 内强制；运行时不变量（worker 上限依赖 rayon 线程池大小）由 `parallel/` 模块在消费时检查。这避免了 dispatch 编译期需要感知 rayon 状态。

### 5.4 ParallelGuard / ParallelContext

```rust,ignore
/// RAII guard that indicates the current thread is inside a
/// library-internal parallel region.
///
/// When `ParallelGuard::enter()` returns `Ok(guard)`, the thread-local
/// flag is set. While the guard is alive, any nested call to
/// `select_exec_path()` or `should_parallelize()` that would return
/// `ExecPath::Parallel` will instead fall back to `Serial`.
///
/// Dropping the guard clears the thread-local flag, allowing
/// future parallel execution on this thread.
pub(crate) struct ParallelGuard {
    // private: marker to prevent external construction
    _private: (),
}

impl ParallelGuard {
    /// Try to enter a parallel execution region.
    ///
    /// Returns `Ok(guard)` if the current thread is not already inside
    /// a parallel region. Returns `Err(())` if nested parallel execution
    /// is detected—the caller must fall back to `ExecPath::Serial`.
    ///
    /// This function is infallible in the `Result` sense; it never panics
    /// and never returns a recoverable error type. The `Err(())` is a
    /// sentinel for "cannot parallelize now"—the caller must respond by
    /// taking the serial path.
    pub(crate) fn enter() -> Result<Self, ()> {
        // Implementation uses thread_local! + Cell<bool>
        // See §6.2 for internal design.
        unimplemented!()
    }
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        // Clear the thread-local flag.
    }
}
```

**实现提示：** 基于 thread-local `Cell<bool>` 实现：

```rust,ignore
// Internal implementation sketch (not a public API commitment)
std::thread_local! {
    static IN_PARALLEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
```

`ParallelContext` 是 thread-local 状态 token，由 `ParallelGuard` 内部管理，不对外暴露为独立类型。各个 `parallel/` 后端函数在入口处调用 `ParallelGuard::enter()`；若返回 `Err(())` 则表明当前已处于并行区域内，必须回退串行路径。

**嵌套并行行为矩阵：**

| 场景                              | `ParallelGuard::enter()` 结果 | dispatch 返回        |
| --------------------------------- | :---------------------------: | -------------------- |
| 首次进入并行区域                  | `Ok(guard)`                   | `ExecPath::Parallel` |
| 已在并行区域内，再次尝试进入      | `Err(())`                     | `ExecPath::Serial`   |
| guard drop 后再次进入             | `Ok(guard)`                   | `ExecPath::Parallel` |
| `ParallelPool` 内部调用并行后端   | `Err(())`（同嵌套规则）       | `ExecPath::Serial`   |

### 5.5 核心 API 函数

```rust,ignore
/// Selects the optimal execution path for an operation.
///
/// This is the central dispatch function. All `math`, `matrix`, and
/// `reduction` modules call this to decide their execution strategy.
///
/// # Arguments
///
/// * `len` - Logical element count of the input(s).
/// * `is_contiguous` - Whether all inputs are F-order contiguous.
/// * `alignment_ok` - Whether all inputs satisfy the SIMD alignment
///   fast-path precondition (as determined by `layout::is_aligned()`).
///
/// # Returns
///
/// * `ExecPath::Parallel` — if `len >= parallel_threshold` **and**
///   contiguity allows, **and** `feature = "parallel"` is enabled,
///   **and** the current thread is not already inside a parallel region.
///
/// * `ExecPath::Simd` — if `len >= simd_threshold` **and** contiguous
///   **and** aligned, **and** `feature = "simd"` is enabled, **and**
///   parallel is either disabled or `len < parallel_threshold`.
///
/// * `ExecPath::Serial` — otherwise (the default fallback).
///
/// # Behavior under feature gates
///
/// - Without `feature = "parallel"`: never returns `ExecPath::Parallel`
/// - Without `feature = "simd"`: never returns `ExecPath::Simd`
/// - With both disabled: always returns `ExecPath::Serial`
///
/// # Determinism
///
/// This function is deterministic: given the same inputs and global
/// threshold state, it always returns the same `ExecPath`.
pub(crate) fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool,
) -> ExecPath;

/// Quick boolean query for "should I use parallel?"
///
/// Used by callers that already know they have only Serial/Parallel
/// options (no SIMD relevance). Equivalent to checking whether
/// `select_exec_path(...)` would return `ExecPath::Parallel`.
///
/// This function respects the same threshold and feature-gate logic
/// as `select_exec_path()`, but it does not consider SIMD at all.
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
/// Setting `threshold = 0` effectively disables the parallel path
/// for all future `select_exec_path()` calls.
pub(crate) fn set_parallel_threshold(threshold: usize);

/// Reset the parallel threshold to its compile-time default.
pub(crate) fn reset_parallel_threshold();
```

**非连续惩罚策略：** 当输入非 F-order 连续时，有效阈值翻倍：

| 条件                           | 有效并行阈值                | 有效 SIMD 阈值               |
| ------------------------------ | --------------------------- | ---------------------------- |
| 连续 + 对齐                    | `PARALLEL_THRESHOLD`        | `SIMD_THRESHOLD`             |
| 连续 + 非对齐                  | `PARALLEL_THRESHOLD`        | 不返回 `Simd`（对齐失败）    |
| 非连续                         | `2 * PARALLEL_THRESHOLD`    | 不返回 `Simd`（连续性失败）  |
| 非连续 + `len < 2 * threshold` | 回退 `Serial`               | 回退 `Serial`                |

这一策略确保仅在收益明确（输入足够大以摊平非连续访问的缓存惩罚）时才选择非标量路径。

### 5.7 Guard API（补充）

```rust,ignore
impl ParallelGuard {
    /// Try to enter parallel execution.
    ///
    /// Returns `Ok(guard)` if not already in a parallel region;
    /// `Err(())` if a nested call is detected (caller falls back to Serial).
    pub(crate) fn enter() -> Result<Self, ()>;
}

// Drop implementation releases the thread-local flag.
```

### 5.8 路径选择阈值（分操作类型参考）

dispatch 持有的阈值适用于**所有操作类型**的统一入口裁决。各操作的具体 SIMD 阈值差异由 `simd/` 内部处理。以下表格列出 dispatch 向各操作类别推荐的通用策略，与 `08-simd.md` §5.7 和 `09-parallel.md` 保持一致：

| 操作类型       | 元素类型                        | 并行最小长度 | SIMD 最小长度 | 说明                                   |
| -------------- | ------------------------------- | :----------: | :-----------: | -------------------------------------- |
| 逐元素算术     | `f32` / `f64`                   |    65536     |      64       | 连续 + 对齐时优先 SIMD                 |
| 逐元素算术     | `Complex<f32>` / `Complex<f64>` |    65536     |     128       | AoS 输入的 SIMD 阈值高于实数路径       |
| 逐元素算术     | `i32` / `i64`                   |    65536     |      64       | 与浮点逐元素一致                       |
| 归约 `sum`     | `f32` / `f64`                   |    65536     |    1024       | SIMD 归约阈值更高（由 simd/ 内部裁决） |
| 归约 `sum`     | `i32` / `i64`                   |    65536     |     512       | 整数 widening accumulator              |
| 内积 `dot`     | `f32` / `f64`                   |    65536     |     512       | 同归约                                 |
| 内积 `dot`     | `i32` / `i64`                   |    65536     |     256       | 同归约                                 |

- dispatch 的 `select_exec_path()` 接收调用方传入的具体 `len`、`is_contiguous` 和 `alignment_ok`；调用方应根据自身操作类型决定是否传入更严格的限制（例如，归约调用方在长度足够大时才走 dispatch，此时 dispatch 依据统一阈值裁决并行；SIMD 路径的最终长度准入由 `simd/` 做二次检查）。
- 非连续惩罚：有效并行阈值在上表基础上翻倍。

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
    ExecPath::{Serial | Simd | Parallel}

Steps:
    1. effective_parallel_threshold = PARALLEL_THRESHOLD
       if not is_contiguous:
           effective_parallel_threshold *= 2

    2. effective_simd_threshold = SIMD_THRESHOLD
       // Simd path requires both contiguity and alignment;
       // if either fails, Simd is not eligible.

    3. // Check parallel eligibility
       if cfg!(feature = "parallel")
          AND len >= effective_parallel_threshold
          AND ParallelGuard::enter().is_ok():
              return ExecPath::Parallel
          // Note: ParallelGuard is consumed here;
          // the caller may re-enter the guard in the parallel backend.

    4. // Check SIMD eligibility
       if cfg!(feature = "simd")
          AND is_contiguous
          AND alignment_ok
          AND len >= effective_simd_threshold:
              return ExecPath::Simd

    5. // Default fallback
       return ExecPath::Serial
```

**优先级说明：** 并行检查在 SIMD 之前。这是因为在同时启用两个 feature 且输入足够大的场景下，并行路径的吞吐收益通常高于 SIMD 串行加速。若并行不可用或输入未达并行阈值，再考虑 SIMD。

**确定性保证：** 该算法完全确定性——相同的 `(len, is_contiguous, alignment_ok, threshold_state, feature_flags)` 输入始终产生相同的 `ExecPath`。

### 6.2 ParallelGuard 实现

```rust,ignore
use std::cell::Cell;

std::thread_local! {
    /// Thread-local flag indicating whether the current thread is
    /// executing inside a library-internal parallel region.
    static IN_PARALLEL: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct ParallelGuard {
    _private: (),
}

impl ParallelGuard {
    pub(crate) fn enter() -> Result<Self, ()> {
        IN_PARALLEL.with(|flag| {
            if flag.get() {
                // Already inside a parallel region — reject nesting
                Err(())
            } else {
                flag.set(true);
                Ok(ParallelGuard { _private: () })
            }
        })
    }
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        IN_PARALLEL.with(|flag| {
            flag.set(false);
        });
    }
}

/// Query-only: check if currently in parallel region without setting the flag.
/// Used by `select_exec_path()` for the pre-check before attempting `enter()`.
fn is_in_parallel() -> bool {
    IN_PARALLEL.with(|flag| flag.get())
}
```

**安全性论证：**

| 主题          | 论证                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------- |
| thread-local  | `Cell<bool>` 是 `!Sync`，但通过 `thread_local!` 访问确保每个线程拥有独立副本，无数据竞争。        |
| RAII 保证     | `ParallelGuard` 的 `Drop` 实现在任何退出路径（包括 panic unwind）下都会重置 flag，不会泄漏状态。  |
| 重入安全      | `enter()` 内部先检查 flag，再设置 flag，无 TOCTOU 问题（单线程内顺序执行）。                       |
| 零分配        | 不在堆上分配，不涉及原子操作（`Cell` 是非原子内部可变性），开销为单次 thread-local 访问。         |

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

/// Runtime SIMD threshold (currently not overridable; reserved for
/// future testing needs).
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
```

**Ordering 选择理由：** `Relaxed` 足够——阈值只在初始化/测试覆写时变更一次；读侧（`select_exec_path` 热路径）不需要与任何其他原子变量同步，仅需读到某个合法值即可。不存在需要 `Acquire/Release` 语义的 paired 操作。

### 6.4 Feature Gate 处理

```rust,ignore
// Compile-time feature detection for variant elimination.
//
// When `feature = "parallel"` is absent, the compiler eliminates
// the entire parallel-eligibility branch as dead code, making
// `ExecPath::Parallel` truly zero-cost.
//
// When `feature = "simd"` is absent, the SIMD branch is eliminated.

pub(crate) fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool,
) -> ExecPath {
    let effective_parallel = if is_contiguous {
        get_parallel_threshold()
    } else {
        get_parallel_threshold() * 2
    };

    // Parallel check: compiled away when feature is absent
    #[cfg(feature = "parallel")]
    {
        if len >= effective_parallel && !is_in_parallel() {
            // Note: we do NOT consume the guard here;
            // the caller will enter() in the parallel backend.
            return ExecPath::Parallel;
        }
    }

    // SIMD check: compiled away when feature is absent
    #[cfg(feature = "simd")]
    {
        if is_contiguous && alignment_ok && len >= get_simd_threshold() {
            return ExecPath::Simd;
        }
    }

    ExecPath::Serial
}
```

### 6.5 决策流 ASCII 图

```
Caller (math / matrix / reduction)
    │
    ▼
┌────────────────────────────────────────────────────────┐
│         dispatch::select_exec_path(len, contiguous,     │
│                                    alignment_ok)        │
└────────────────────────┬───────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│ feature =     │ │ feature =    │ │ default      │
│ "parallel" +  │ │ "simd" +     │ │ fallback     │
│ len >= eff.   │ │ contiguous + │ │              │
│ threshold +   │ │ aligned +    │ │              │
│ not nested    │ │ len >= SIMD  │ │              │
│               │ │ threshold +  │ │              │
│               │ │ parallel     │ │              │
│               │ │ disabled or  │ │              │
│               │ │ len < PAR.   │ │              │
└───────┬───────┘ └──────┬───────┘ └──────┬───────┘
        │                │                │
        ▼                ▼                ▼
ExecPath::Parallel  ExecPath::Simd  ExecPath::Serial
        │                │                │
        ▼                ▼                ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│ parallel/     │ │ simd/        │ │ caller's     │
│ backend       │ │ backend      │ │ own scalar   │
│ (workers run  │ │ (may         │ │ impl         │
│  scalar code, │ │  internally  │ │              │
│  no SIMD)     │ │  fall back   │ │              │
│               │ │  to scalar)  │ │              │
└───────────────┘ └──────────────┘ └──────────────┘
```

**关键语义：** dispatch 做出三路裁决后，调用方执行单次 `match` 分发。`ExecPath::Simd` 分支由 `simd/` 后端接管——它可能在内部因 ISA、lane 宽度等原因回退标量，这一回退对 dispatch 完全透明。

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
  - 内容: 三路裁决逻辑、阈值读取、feature gate 分支、非连续惩罚
  - 测试: `test_exec_path_serial_below_threshold`, `test_exec_path_parallel_above_threshold`, `test_exec_path_simd_when_aligned`
  - 前置: T1
  - 预计: 10 min

- [ ] **T3**: 实现阈值配置接口
  - 文件: `src/dispatch.rs`
  - 内容: `AtomicUsize` 阈值存储、`set_parallel_threshold()`、`reset_parallel_threshold()`、`Relaxed` ordering 注释
  - 测试: `test_threshold_override_respected`
  - 前置: T2
  - 预计: 5 min

### Wave 3: 嵌套防护

- [ ] **T4**: 实现 `ParallelGuard` / `ParallelContext`
  - 文件: `src/dispatch.rs`
  - 内容: `thread_local! { IN_PARALLEL: Cell<bool> }`、`ParallelGuard::enter()`、`Drop` 实现
  - 测试: `test_parallel_guard_blocks_nesting`, `test_parallel_guard_releases_on_drop`
  - 前置: T1
  - 预计: 10 min

### Wave 4: 测试与验证

- [ ] **T5**: 编写 dispatch 全套单元测试
  - 文件: `src/dispatch.rs` (#[cfg(test)])
  - 内容: 各路径返回验证、阈值边界、feature gate 组合、嵌套防护、非连续惩罚
  - 测试: 见 §8.2 完整清单
  - 前置: T2, T3, T4
  - 预计: 10 min

**总预计时间：** ~40 min。所有任务均在同一文件 `src/dispatch.rs` 内完成。

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
| `test_exec_path_serial_below_threshold`           | `len < threshold` 时返回 `Serial`                              | 高     |
| `test_exec_path_parallel_above_threshold`         | 连续大输入启用 parallel feature 时返回 `Parallel`             | 高     |
| `test_exec_path_simd_when_aligned`                | 中等输入 + 连续 + 对齐且启用 simd feature 时返回 `Simd`       | 高     |
| `test_exec_path_serial_when_noncontiguous_below_doubled_threshold` | 非连续且 len < 2*threshold 时返回 `Serial`         | 高     |
| `test_parallel_guard_blocks_nesting`              | 已处于并行区域时 `ParallelGuard::enter()` 返回 `Err(())`      | 高     |
| `test_parallel_guard_releases_on_drop`            | Drop 后再次 `enter()` 成功                                     | 高     |
| `test_threshold_override_respected`               | `set_parallel_threshold()` 后 `select_exec_path()` 使用新阈值  | 高     |
| `test_reset_threshold_restores_default`           | `reset_parallel_threshold()` 恢复编译期默认值                  | 中     |
| `test_no_parallel_feature_never_returns_parallel` | 未启用 `parallel` feature 时永不为 `Parallel`                  | 高     |
| `test_no_simd_feature_never_returns_simd`         | 未启用 `simd` feature 时永不为 `Simd`                         | 高     |
| `test_should_parallelize_matches_select`          | `should_parallelize()` 与 `select_exec_path()` 并行判断一致    | 中     |
| `test_parallel_preferred_over_simd_for_large_input` | 同时满足并行和 SIMD 条件时返回 `Parallel`                  | 中     |
| `test_deterministic_same_input_same_output`       | 相同输入多次调用返回相同结果                                   | 中     |

### 8.3 边界测试场景

| 场景                                        | 预期行为                                               |
| ------------------------------------------- | ------------------------------------------------------ |
| `len = 0`                                   | 返回 `ExecPath::Serial`                                |
| `len = 1`                                   | 返回 `ExecPath::Serial`（远低于任何阈值）              |
| `len = PARALLEL_THRESHOLD - 1`              | 返回 `Serial` 或 `Simd`（取决于 alignment）            |
| `len = PARALLEL_THRESHOLD`（恰在阈值）      | 返回 `Parallel`（若连续且 feature 启用）               |
| 非连续 `len = 2 * PARALLEL_THRESHOLD - 1`   | 不满足翻倍阈值，返回 `Serial`                          |
| 非连续 `len = 2 * PARALLEL_THRESHOLD`       | 满足翻倍阈值，可能返回 `Parallel`                      |
| `ParallelGuard` 在 panic unwind 中          | `Drop` 仍执行，flag 被正确清除                         |
| 阈值被设为 `0`                              | `select_exec_path()` 永不为 `Parallel`                 |
| 阈值被设为 `usize::MAX`                     | `select_exec_path()` 永不为 `Parallel`                 |

### 8.4 属性测试不变量

| 不变量                                                              | 测试方法                               |
| ------------------------------------------------------------------- | -------------------------------------- |
| Idempotent：相同参数多次调用返回相同结果                            | 循环调用验证                           |
| Monotonic in `len`：若 `len1 <= len2`，则 `select_exec_path(len1, ...)` 的"加速等级"不超过 `select_exec_path(len2, ...)`（Serial < Simd < Parallel） | 对递增 len 序列验证                    |
| Feature gate 不变：禁用 feature 时对应路径永不返回                 | 编译期 `#[cfg(not(feature = "..."))]` 测试 + 运行时断言 |
| Guard 不变：`ParallelGuard` 存活期间 `ParallelGuard::enter()` 始终失败 | 嵌套 `enter()` 测试                   |

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
| `ParallelGuard` 不可外部构造（无 `pub` 构造器） | 编译期可见性检查             |
| `ParallelExecStrategy` 无 `pub` 导出            | 编译期可见性检查             |
| dispatch 模块不依赖 `pulp` 或 `rayon`           | `cargo tree --no-dev-deps` 验证 |

---

## 9. 与其他模块的交互

### 9.1 接口约定

| 方向           | 对方模块         | 接口/类型                                  | 约定                                                                                            |
| -------------- | ---------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 被调用（输出） | `math`           | `select_exec_path()`, `should_parallelize()` | `math` 在广播裁决后调用 dispatch 决定串行/并行路径；并行 worker 内执行标量代码（不使用 SIMD）    |
| 被调用（输出） | `matrix`         | `select_exec_path()`, `should_parallelize()` | `matrix::dot()` 完成 rank/shape 校验后调用 dispatch 三路分发（参见 `12-matrix.md` §6.1）         |
| 被调用（输出） | `reduction`      | `select_exec_path()`                        | `reduction::sum()` 调用 dispatch 决定执行路径（参见 `13-reduction.md` §6.3）                    |
| 被调用（输出） | `simd`（间接）   | `ExecPath::Simd`                            | dispatch 仅推荐 SIMD 路径；`simd/` 内部做最终 admission（ISA/ lane 宽度检测）                   |
| 被消费（输入） | `parallel`       | `ParallelExecStrategy`                      | dispatch 定义该类型，`parallel/` 通过 `crate::dispatch` 引用并消费其字段                        |
| 被消费（输入） | `parallel`       | `ParallelGuard`                             | `parallel/` 后端在入口调用 `ParallelGuard::enter()`；失败时回退（参见 `09-parallel.md` §5.1）   |
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
    │       └── ExecPath::Parallel → parallel/ 后端（各 worker 执行标量代码，无 SIMD）
    │
    └── 返回结果（Tensor / scalar / Result），公开语义不变
```

### 9.3 与 parallel 模块的交互

| 交互项           | dispatch 职责                                   | parallel 职责                                  |
| ---------------- | ----------------------------------------------- | ---------------------------------------------- |
| 路径裁决         | dispatch 决定是否进入并行路径                   | parallel 被选中后纯粹执行，不自行判断路径       |
| 阈值             | dispatch 持有权威阈值并管理覆写                 | parallel 通过 `ParallelExecStrategy` 接收参数   |
| 嵌套防护         | dispatch 提供 `ParallelGuard` 机制              | parallel 在入口处调用 `ParallelGuard::enter()`  |
| `ParallelExecStrategy` | dispatch 定义该结构体                    | parallel 消费 chunk_size / max_workers 字段    |
| 串行回退         | dispatch 选择 `ExecPath::Serial` 时不进入 parallel | parallel 自身不包含串行回退代码               |

参见 `09-parallel.md` §6.1（路径选择）、决策 4（parallel 不包含串行回退）、决策 6（执行路径裁决由 dispatch 统一收口）。

### 9.4 与 simd 模块的交互

| 交互项       | dispatch 职责                                               | simd 职责                                                   |
| ------------ | ----------------------------------------------------------- | ----------------------------------------------------------- |
| ISA 检测     | dispatch **不参与** ISA 检测                                | `pulp::Arch` 做 ISA 检测与缓存（参见 `08-simd.md` §5.4）    |
| 路径推荐     | dispatch 通过 `ExecPath::Simd` 推荐 SIMD 路径               | simd 接收推荐后可自行拒绝（内部回退标量）                    |
| 准入条件     | dispatch 仅检查 len、连续性、对齐                           | simd 内部检查元素类型、ISA lane 宽度、操作支持矩阵          |
| 长度阈值     | dispatch 持有 SIMD 通用阈值（64）                           | simd 持有操作特定阈值（如 sum 的 1024，参见 `08-simd.md` §5.7）|
| 调用方式     | dispatch 不直接调用 simd 代码                               | 语义模块在 `match ExecPath::Simd` 分支中调用 simd 后端       |

dispatch 与 simd 之间是**推荐-接受**关系，而非命令-执行关系。dispatch 说"SIMD 路径可能合适"，simd 说"我能做"或"我回退"。这种分层避免 dispatch 理解 ISA 细节（per Decision X，§11）。

---

## 10. 错误处理与语义边界

| 主题              | 说明                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| Recoverable error | dispatch 自身**不产生**可恢复错误。所有条件不满足时静默回退 `Serial`。                                      |
| Panic             | dispatch 在正常操作中永不 panic。唯一可能的 panic 场景是内部不变量违反（如 `AtomicUsize` 存储逻辑错误），属于实现 bug。 |
| 路径一致性        | `select_exec_path()` 是确定性函数：相同输入 + 相同全局状态 → 相同 `ExecPath`。                              |
| 语义透明          | 调用方从 dispatch 获得的只是路径推荐；无论走哪条路径，公开 API 的 shape、错误类别和数值语义必须一致。       |
| Guard 语义        | `ParallelGuard::enter()` 返回 `Err(())` 不是可恢复错误——它是路径选择的信号，调用方必须回退 `Serial`。       |

---

## 11. 设计决策记录

### 决策 1：三路 ExecPath（Serial / Simd / Parallel）

| 属性     | 值                                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| 决策     | `ExecPath` 枚举包含三个变体：`Serial`、`Simd`、`Parallel`。调用方通过单次 `match` 分发。                    |
| 理由     | 调用方（`math`/`matrix`/`reduction`）需要区分三种互斥执行策略；三路枚举比两层 `Option` 或独立布尔值更清晰，且 match 强制穷尽检查，编译器可做分支优化。 |
| 替代方案 | 两层 `Option<ParallelPath>` + `Option<SimdPath>` —— 放弃，语义模糊，易出现不完整分支覆盖。                  |
| 替代方案 | 四路（含 `SimdParallel`） —— 放弃，当前架构规定并行 worker 内不使用 SIMD，该变体无存在必要。               |
| 来源     | **用户决策 B**：`ExecPath` 有三个变体。                                                                     |

### 决策 2：dispatch 是 ISA-agnostic

| 属性     | 值                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
| 决策     | dispatch.rs 不参与 ISA 检测或 SIMD 能力判定。ISA 检测、lane 宽度选择、对齐细节均保留在 `simd/` 模块内部。                 |
| 理由     | ISA 检测属于 SIMD 后端实现细节；将其隔离在 `simd/` 内部保持 dispatch 简单、可测试、无平台依赖。dispatch 只需要知道"输入是否连续+对齐"。 |
| 替代方案 | dispatch 调用 `pulp::Arch::new()` 做 ISA 检测 —— 放弃，会引入 `pulp` 依赖到 dispatch（违反零依赖原则）、模糊职责边界。  |
| 来源     | **用户决策 X**：dispatch 是 ISA-agnostic；ISA 检测是 `pulp::Arch` 的职责。                                               |

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
| 决策     | 非连续输入时，并行/SIMD 有效阈值翻倍。                                                   |
| 理由     | 非连续输入缓存局部性差；在输入不够大时进入并行/SIMD 路径的收益可能为负（调度开销 > 加速）。翻倍阈值是一个保守但简单的启发式。 |
| 替代方案 | 完全不考虑连续性 —— 放弃，会在小规模非连续输入上引起性能退化。                           |
| 替代方案 | 更复杂的性能模型（如 stride 模式分析）—— 放弃，超出当前版本范围且增加维护成本。           |

### 决策 6：单文件模块（非子目录）

| 属性     | 值                                                                                               |
| -------- | ------------------------------------------------------------------------------------------------ |
| 决策     | dispatch 保持为单一文件 `src/dispatch.rs`，不拆分为 `src/dispatch/` 子目录。                      |
| 理由     | dispatch 表面积极小且内聚性极高（≤200 行实现代码）。单文件减少模块边界样板代码、简化依赖声明。    |
| 替代方案 | 拆分为 `dispatch/` 子目录（如 `path.rs` + `guard.rs` + `threshold.rs`）—— 放弃，对当前规模过度工程化。 |
| 触发条件 | 若未来 dispatch 膨胀到 >500 行或引入新的独立关注点（如更细粒度的 per-op 阈值配置），可重新评估。  |

---

## 12. 性能考量

### 12.1 开销分析

| 开销点                    | 每调用成本              | 说明                                                                     |
| ------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `select_exec_path()`      | ~3 comparisons + 2 loads | 两次 `AtomicUsize::load(Relaxed)` + 一次 `Cell::get()`（仅并行分支）      |
| 阈值读取                 | 单次 Relaxed atomic load | 无锁，无竞争；Relaxed 在 x86 上等同于普通 load。                          |
| ParallelGuard::enter()   | 一次 thread-local 访问   | `Cell::get()` + `Cell::set()`，纳秒级                                     |
| ParallelGuard::drop()    | 一次 thread-local 访问   | `Cell::set(false)`，纳秒级                                                |

**总体评估：** 每次 `select_exec_path()` 调用的总开销 < 10 ns（在典型 x86_64 硬件上）。相比之下，被派发的操作本身（如 1M 元素的 `add`）耗时在微秒至毫秒级，dispatch 开销可忽略不计。

### 12.2 零成本抽象

| 场景                              | 编译结果                                                                    |
| --------------------------------- | --------------------------------------------------------------------------- |
| `feature = "parallel"` 关闭       | 并行分支被 `#[cfg]` 消除；`ParallelGuard` 相关代码不生成                    |
| `feature = "simd"` 关闭           | SIMD 分支被 `#[cfg]` 消除；`ExecPath::Simd` 变体可能被编译器标记为不可达    |
| 两者均关闭                        | `select_exec_path()` 被优化为立即返回 `ExecPath::Serial`                    |
| 阈值常量传播                      | 编译期常量在 release build 中被内联为立即数                                 |

### 12.3 缓存效应

- 阈值存储在 `static AtomicUsize` 中，不与其他频繁写入的变量共享 cache line，无 false sharing 风险。
- `IN_PARALLEL` thread-local 变量每个线程一份，无跨线程竞争。
- `select_exec_path()` 是纯函数（除读取全局阈值），无副作用；其结果可被调用方安全缓存，但通常不需要——每操作调用一次的开销已经极低。

---

## 13. 平台与工程约束

| 约束       | 说明                                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `std` only | dispatch 仅使用 `std`（`AtomicUsize`、`Cell`、`thread_local!`），不依赖任何外部 crate                            |
| MSRV       | Rust 1.85+                                                                                                       |
| 单 crate   | dispatch 是 Xenon 单 crate 的内部模块，不独立发布                                                               |
| SemVer     | dispatch 全为 `pub(crate)`——无对外 SemVer 承诺。内部 API 变更不影响下游用户，但需保持与 `parallel`/`math`/`matrix`/`reduction` 的接口一致 |
| 最小依赖   | 无新增依赖；不依赖 `pulp`、`rayon` 或任何其他第三方 crate                                                        |
| 确定性     | 同平台、同编译配置、同输入下 `select_exec_path()` 结果必须确定，不依赖随机数或时间                               |
| 不变量     | `ExecPath` 枚举为 `#[derive(Copy, Clone, Debug, PartialEq, Eq)]`，零成本传递                                     |

---

## 版本历史

| 版本  | 日期       | 说明                                                                                   |
| ----- | ---------- | -------------------------------------------------------------------------------------- |
| 1.0.0 | 2026-05-02 | 初始版本：依据用户决策 B（三路 ExecPath）+ 决策 X（ISA-agnostic），定义 dispatch 的完整设计 |

---

_本文档由 Xenon 项目维护。如有问题请提交 Issue 或 PR。_
