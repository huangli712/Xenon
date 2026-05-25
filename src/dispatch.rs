//! Internal execution-path dispatch.
//!
//! See `docs/design/30-dispatch.md` for the full design (path
//! arbitration, nested-parallel guard, threshold storage, feature
//! gates). All items in this module are `pub(crate)` by default; a
//! minimal subset is re-exported under `#[doc(hidden)]` at the crate
//! root solely so integration tests under `tests/` can observe
//! dispatch decisions. These re-exports are NOT a stable public API.

/// Three mutually exclusive execution paths recommended by dispatch.
///
/// See `30-dispatch.md §5.2` for full per-variant semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecPath {
    /// Serial scalar execution. Default fallback when neither SIMD
    /// nor parallel preconditions are met.
    Serial,
    /// Serial path with SIMD acceleration. dispatch only signals
    /// "SIMD path is preferred"; the `simd/` backend retains final
    /// admission (ISA, lane width, alignment per `08-simd.md §5.7`).
    Simd,
    /// Parallel execution. Returned only when `feature = "parallel"`
    /// is enabled, the input meets the parallel threshold, and the
    /// current thread is not already inside a library-internal
    /// parallel region. Always accompanied by `Some(ParallelGuard)`
    /// per `30-dispatch.md §5.5`.
    Parallel,
}

/// Execution strategy parameters consumed by the parallel backend.
///
/// Defined here; consumed by `parallel/` module functions such as
/// `par_map`, `par_zip_map`, `par_sum`, `par_dot`. Fields are private
/// to enforce construction via `ParallelExecStrategy::new()` (see
/// `30-dispatch.md §5.3`).
///
/// Only compiled under `feature = "parallel"` per §5.1: outside that
/// feature `select_exec_path()` can never return `ExecPath::Parallel`,
/// so the strategy type itself is unreachable.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone, Copy)]
pub struct ParallelExecStrategy {
    /// Suggested chunk size for parallel chunking. `None` means the
    /// parallel backend decides (typically via `compute_safe_chunks`).
    chunk_size: Option<usize>,
    /// Maximum worker count. `None` means use rayon's default thread
    /// pool size.
    max_workers: Option<usize>,
}

#[cfg(feature = "parallel")]
impl ParallelExecStrategy {
    /// Construct a validated strategy. Performs ALL field-level
    /// validation per 30-dispatch.md §5.3 so that the parallel/
    /// backend can consume the value without re-validation.
    ///
    /// # Errors
    ///
    /// Returns an `XenonError` from `dispatch_invalid_argument` when:
    /// - `chunk_size == Some(0)` — chunk size must be non-zero
    /// - `max_workers == Some(0)` — worker count must be non-zero
    /// - `max_workers == Some(n)` with `n > rayon::current_num_threads()` —
    ///   worker count must not exceed the rayon pool size
    pub fn new(
        chunk_size: Option<usize>,
        max_workers: Option<usize>,
    ) -> crate::error::Result<Self> {
        if matches!(chunk_size, Some(0)) {
            return Err(crate::error::XenonError::dispatch_invalid_argument(
                "chunk_size",
                "must be non-zero",
                "0",
            ));
        }
        if matches!(max_workers, Some(0)) {
            return Err(crate::error::XenonError::dispatch_invalid_argument(
                "max_workers",
                "must be non-zero",
                "0",
            ));
        }
        if let Some(n) = max_workers {
            // Read the pool size once at construction time per §5.3 line 215-219.
            let pool = rayon::current_num_threads();
            if n > pool {
                return Err(crate::error::XenonError::dispatch_invalid_argument(
                    "max_workers",
                    format!("must not exceed rayon pool size ({pool})"),
                    n.to_string(),
                ));
            }
        }
        Ok(Self {
            chunk_size,
            max_workers,
        })
    }

    /// Default strategy: let the parallel backend decide everything.
    pub fn auto() -> Self {
        Self {
            chunk_size: None,
            max_workers: None,
        }
    }

    pub(crate) fn chunk_size(&self) -> Option<usize> {
        self.chunk_size
    }

    pub(crate) fn max_workers(&self) -> Option<usize> {
        self.max_workers
    }
}

// ---------------------------------------------------------------------------
// Threshold storage — constants, atomics, getters
// ---------------------------------------------------------------------------

/// Compile-time default for parallel threshold.
const DEFAULT_PARALLEL_THRESHOLD: usize = 65_536;

/// Runtime-overridable parallel threshold.
///
/// Uses `AtomicUsize` for lock-free reads. Written only during
/// initialization or explicit override (testing/benchmarking).
static PARALLEL_THRESHOLD: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// Compile-time default for SIMD threshold.
const DEFAULT_SIMD_THRESHOLD: usize = 64;

/// Runtime-overridable SIMD threshold.
static SIMD_THRESHOLD: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_SIMD_THRESHOLD);

fn get_parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed)
}

fn get_simd_threshold() -> usize {
    SIMD_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Threshold runtime override API (testing/benchmarking only)
// ---------------------------------------------------------------------------

/// Override the parallel threshold at runtime.
///
/// Setting `threshold = 0` disables the parallel path entirely
/// (sentinel per 30-dispatch.md §5.6 line 494-499).
pub fn set_parallel_threshold(threshold: usize) {
    PARALLEL_THRESHOLD.store(threshold, std::sync::atomic::Ordering::Relaxed);
}

/// Reset the parallel threshold to its compile-time default.
pub fn reset_parallel_threshold() {
    set_parallel_threshold(DEFAULT_PARALLEL_THRESHOLD);
}

/// Override the SIMD threshold at runtime.
///
/// Use `usize::MAX` to disable the SIMD path (sentinel per
/// 30-dispatch.md §5.6 line 520-523).
pub fn set_simd_threshold(threshold: usize) {
    SIMD_THRESHOLD.store(threshold, std::sync::atomic::Ordering::Relaxed);
}

/// Reset the SIMD threshold to its compile-time default.
pub fn reset_simd_threshold() {
    set_simd_threshold(DEFAULT_SIMD_THRESHOLD);
}

// ---------------------------------------------------------------------------
// Core dispatch functions
// ---------------------------------------------------------------------------

/// Selects the optimal execution path for an operation, atomically
/// binding "select Parallel" with "enter the parallel region".
///
/// See `30-dispatch.md §5.5` for the full contract.
pub fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool,
) -> (ExecPath, Option<ParallelGuard>) {
    #[cfg(feature = "parallel")]
    if is_in_parallel() {
        return (ExecPath::Serial, None);
    }

    // §6.4: zero sentinel disables parallel; non-contiguous gets saturating doubled threshold.
    let base = get_parallel_threshold();
    let parallel_eligible_by_threshold = if base == 0 {
        false
    } else {
        let effective = if is_contiguous {
            base
        } else {
            base.saturating_mul(2)
        };
        len >= effective
    };

    #[cfg(feature = "parallel")]
    {
        if parallel_eligible_by_threshold && let Some(guard) = try_acquire_guard() {
            return (ExecPath::Parallel, Some(guard));
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = parallel_eligible_by_threshold;
    }

    #[cfg(feature = "simd")]
    {
        if is_contiguous && len >= get_simd_threshold() {
            // alignment_ok is a hint to the simd backend (§5.5 / §5.6 / §6.4);
            // dispatch does not gate SIMD on alignment.
            let _simd_alignment_hint = alignment_ok;
            return (ExecPath::Simd, None);
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        let _ = alignment_ok;
    }

    (ExecPath::Serial, None)
}

/// Quick boolean query for "should I use parallel?"
///
/// Does **not** acquire a `ParallelGuard`. Includes `is_in_parallel()`
/// check (§6.4) so the result matches what `select_exec_path()` would do.
#[cfg(feature = "parallel")]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "dispatch is staged before downstream integration")
)]
pub(crate) fn should_parallelize(len: usize, is_contiguous: bool) -> bool {
    let base = get_parallel_threshold();
    // §6.4: zero sentinel disables; in-parallel TLS suppresses nested parallel.
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
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "dispatch is staged before downstream integration")
)]
pub(crate) fn should_parallelize(_len: usize, _is_contiguous: bool) -> bool {
    false
}

// ---------------------------------------------------------------------------
// ParallelGuard — nested-parallel guard (feature-gated)
// ---------------------------------------------------------------------------

/// RAII guard that indicates the current thread is inside a
/// library-internal parallel region.
///
/// A `ParallelGuard` value is **only ever obtained as the second tuple
/// element of `select_exec_path()`** when that function selects
/// `ExecPath::Parallel`. There is no public `enter()` constructor.
///
/// While the guard is alive, the thread-local flag is set. Any nested
/// call to `select_exec_path()` or `should_parallelize()` will observe
/// `is_in_parallel() == true` and fall back to `Serial`.
///
/// Dropping the guard clears the thread-local flag.
///
/// Under `feature = "parallel"`, the guard is `!Send + !Sync` because
/// its `Drop` clears the **current** thread's TLS flag.
#[cfg(feature = "parallel")]
pub struct ParallelGuard {
    _private: core::marker::PhantomData<*const ()>,
}

#[cfg(feature = "parallel")]
impl Drop for ParallelGuard {
    fn drop(&mut self) {
        IN_PARALLEL.with(|flag| flag.set(false));
    }
}

/// Placeholder `ParallelGuard` when `feature = "parallel"` is disabled.
///
/// Zero-size, never constructed; no Drop, intentionally `Send + Sync`.
/// This keeps `(ExecPath, Option<ParallelGuard>)` `Send + Sync` in
/// default builds where the option is always `None`.
#[cfg(not(feature = "parallel"))]
pub struct ParallelGuard {
    _private: core::marker::PhantomData<()>,
}

// ---------------------------------------------------------------------------
// Thread-local IN_PARALLEL flag and guard acquisition helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
std::thread_local! {
    static IN_PARALLEL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Module-private: produces a guard only via `select_exec_path()`.
#[cfg(feature = "parallel")]
fn try_acquire_guard() -> Option<ParallelGuard> {
    IN_PARALLEL.with(|flag| {
        (!flag.replace(true)).then_some(ParallelGuard {
            _private: core::marker::PhantomData,
        })
    })
}

/// Query-only: check if currently in parallel region without setting the flag.
#[cfg(feature = "parallel")]
fn is_in_parallel() -> bool {
    IN_PARALLEL.with(|flag| flag.get())
}

// ---------------------------------------------------------------------------
// with_parallel_worker_context — worker TLS helper
// ---------------------------------------------------------------------------

/// Runs `f` while marking the current worker thread as being inside a
/// Xenon-internal parallel region.
///
/// Used by `parallel/` inside Rayon worker closures: outer `ParallelGuard`
/// stays on the dispatching thread; each worker closure wraps its chunk
/// execution in this helper so nested `select_exec_path()` calls inside
/// the worker thread correctly observe `IN_PARALLEL == true`.
///
/// Does NOT construct or consume `ParallelGuard`. Saves/restores the
/// previous TLS value (panic-safe via inner `Reset` RAII).
#[cfg(feature = "parallel")]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "dispatch is staged before downstream integration")
)]
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

/// No-op passthrough when parallel feature is disabled.
#[cfg(not(feature = "parallel"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "dispatch is staged before downstream integration")
)]
pub(crate) fn with_parallel_worker_context<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[cfg(test)]
mod tests {
    use super::ExecPath;

    struct ThresholdTestLock;

    static THRESHOLD_TEST_LOCK: std::sync::Mutex<ThresholdTestLock> =
        std::sync::Mutex::new(ThresholdTestLock);

    struct ThresholdTestGuard<'lock> {
        _lock: std::sync::MutexGuard<'lock, ThresholdTestLock>,
        parallel_threshold: usize,
        simd_threshold: usize,
    }

    impl ThresholdTestGuard<'_> {
        fn new() -> Self {
            let lock = match THRESHOLD_TEST_LOCK.lock() {
                Ok(lock) => lock,
                Err(poisoned) => poisoned.into_inner(),
            };
            Self {
                _lock: lock,
                parallel_threshold: super::get_parallel_threshold(),
                simd_threshold: super::get_simd_threshold(),
            }
        }
    }

    impl Drop for ThresholdTestGuard<'_> {
        fn drop(&mut self) {
            super::set_parallel_threshold(self.parallel_threshold);
            super::set_simd_threshold(self.simd_threshold);
        }
    }

    // === W10T1: Skeleton compile-smoke ===

    #[test]
    fn test_exec_path_enum_defined() {
        // Skeleton compile-smoke: verifies the enum type is usable.
        // Real threshold-vs-path behaviour is tested in W10T4/W10T6.
        assert_eq!(ExecPath::Serial, ExecPath::Serial);
    }

    // === W10T2: ParallelGuard and worker context ===

    #[cfg(feature = "parallel")]
    #[test]
    fn test_guard_drop_releases_flag() {
        let _threshold_guard = ThresholdTestGuard::new();

        let (first_path, first_guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(first_path, ExecPath::Parallel);

        let first_guard = first_guard.expect("parallel path must return a guard");
        drop(first_guard);

        let (second_path, second_guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(second_path, ExecPath::Parallel);
        assert!(second_guard.is_some());
        drop(second_guard);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_worker_context_restores_previous_flag() {
        // 30-dispatch §5.4: with_parallel_worker_context must save the
        // previous TLS value, set it to true, run f, and restore on drop.
        assert!(!super::is_in_parallel());
        let observed = super::with_parallel_worker_context(super::is_in_parallel);
        assert!(observed, "TLS must be true inside the worker context");
        assert!(
            !super::is_in_parallel(),
            "TLS must be restored after the context exits"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_guard_drop_releases_flag_after_panic_unwind() {
        let _threshold_guard = ThresholdTestGuard::new();

        let panic_result = std::panic::catch_unwind(|| {
            let (path, guard) = super::select_exec_path(usize::MAX, true, true);
            assert_eq!(path, ExecPath::Parallel);

            let _guard = guard.expect("parallel path must return a guard");
            panic!("intentional panic to test ParallelGuard drop during unwind");
        });
        assert!(panic_result.is_err());

        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Parallel);
        assert!(guard.is_some());
        drop(guard);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_worker_context_restores_previous_flag_after_panic_unwind() {
        assert!(!super::is_in_parallel());

        let panic_result = std::panic::catch_unwind(|| {
            super::with_parallel_worker_context(|| {
                assert!(super::is_in_parallel());
                panic!("intentional panic to test worker-context unwind reset");
            });
        });
        assert!(panic_result.is_err());
        assert!(!super::is_in_parallel());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_worker_context_nests_correctly() {
        // Nested worker contexts: inner runs with true, outer restores
        // its own previous value (also true), then outermost restores false.
        let outer_inner = super::with_parallel_worker_context(|| {
            assert!(super::is_in_parallel());
            super::with_parallel_worker_context(super::is_in_parallel)
        });
        assert!(outer_inner);
        assert!(!super::is_in_parallel());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_guard_is_not_send_or_sync() {
        static_assertions::assert_not_impl_any!(super::ParallelGuard: Send, Sync);
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_placeholder_parallel_guard_is_send_sync() {
        static_assertions::assert_impl_all!(super::ParallelGuard: Send, Sync);
    }

    // === W10T3: ParallelExecStrategy construction ===

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_rejects_zero() {
        let chunk_size_error = super::ParallelExecStrategy::new(Some(0), None)
            .expect_err("chunk_size=0 must be rejected");
        assert!(matches!(
            chunk_size_error,
            crate::error::XenonError::InvalidArgument {
                kind: crate::error::InvalidArgumentKind::InvalidConfig {
                    argument,
                    constraint,
                    actual,
                },
                ..
            } if argument == "chunk_size"
                && constraint == "must be non-zero"
                && actual == "0"
        ));

        let max_workers_error = super::ParallelExecStrategy::new(None, Some(0))
            .expect_err("max_workers=0 must be rejected");
        assert!(matches!(
            max_workers_error,
            crate::error::XenonError::InvalidArgument {
                kind: crate::error::InvalidArgumentKind::InvalidConfig {
                    argument,
                    constraint,
                    actual,
                },
                ..
            } if argument == "max_workers"
                && constraint == "must be non-zero"
                && actual == "0"
        ));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_accepts_none() {
        // 30-dispatch §8.2 line 1008: new(None, None) ≡ auto().
        let lhs =
            super::ParallelExecStrategy::new(None, None).expect("new(None, None) should succeed");
        let rhs = super::ParallelExecStrategy::auto();
        assert_eq!(lhs.chunk_size(), rhs.chunk_size());
        assert_eq!(lhs.max_workers(), rhs.max_workers());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_rejects_oversized_max_workers() {
        // 30-dispatch §5.3 line 211-213: max_workers > rayon pool size → InvalidArgument.
        let pool = rayon::current_num_threads();
        let actual = pool.saturating_add(1);
        let error = super::ParallelExecStrategy::new(None, Some(actual))
            .expect_err("oversized max_workers must be rejected");

        assert!(matches!(
            error,
            crate::error::XenonError::InvalidArgument {
                kind: crate::error::InvalidArgumentKind::InvalidConfig {
                    argument,
                    constraint,
                    actual: actual_value,
                },
                ..
            } if argument == "max_workers"
                && constraint == format!("must not exceed rayon pool size ({pool})")
                && actual_value == actual.to_string()
        ));
    }

    // === W10T4: select_exec_path / should_parallelize ===

    #[test]
    fn test_select_returns_guard_iff_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        // Default thresholds: parallel=65536. usize::MAX should be eligible
        // under feature=parallel; otherwise Serial/Simd with None guard.
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path == ExecPath::Parallel, guard.is_some());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_should_parallelize_diagnostic_does_not_acquire_guard() {
        let _threshold_guard = ThresholdTestGuard::new();
        // should_parallelize is a pure query: it must not consume the
        // nested-parallel slot that select_exec_path() needs to return
        // (Parallel, Some(_)).
        assert!(super::should_parallelize(usize::MAX, true));

        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Parallel);
        assert!(guard.is_some());
        drop(guard);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_should_parallelize_returns_false_when_in_parallel_region() {
        let _threshold_guard = ThresholdTestGuard::new();
        // 30-dispatch §6.4 line 826: should_parallelize returns false when
        // is_in_parallel() == true, even if length threshold is satisfied.
        let _outer = super::try_acquire_guard().expect("outer guard acquisition");
        assert!(!super::should_parallelize(usize::MAX, true));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_nested_select_falls_back_to_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        // Hold an outer guard, then verify that a nested select_exec_path()
        // correctly falls back to Serial instead of re-entering Parallel.
        let _outer = super::try_acquire_guard().expect("outer guard acquisition");
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    #[cfg(all(feature = "parallel", feature = "simd"))]
    #[test]
    fn test_nested_simd_eligible_select_falls_back_to_serial() {
        let _threshold_guard = ThresholdTestGuard::new();

        let (path, guard) = super::with_parallel_worker_context(|| {
            super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, true, true)
        });

        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // === W10T5: Threshold configuration ===

    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_override_respected() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(42);

        let (below_path, below_guard) = super::select_exec_path(41, true, true);
        assert_ne!(below_path, ExecPath::Parallel);
        assert!(below_guard.is_none());

        let (at_path, at_guard) = super::select_exec_path(42, true, true);
        assert_eq!(at_path, ExecPath::Parallel);
        assert!(at_guard.is_some());
        drop(at_guard);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_zero_disables_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        // 30-dispatch §5.6 line 494-499: threshold=0 is the parallel-disable sentinel.
        super::set_parallel_threshold(0);
        assert!(!super::should_parallelize(usize::MAX, true));

        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_ne!(path, ExecPath::Parallel);
        assert!(guard.is_none());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_saturating_mul_no_overflow() {
        let _threshold_guard = ThresholdTestGuard::new();
        // 30-dispatch §8.2 / §8.3: threshold near usize::MAX must saturate, never wrap.
        super::set_parallel_threshold(usize::MAX / 2 + 1);
        assert!(!super::should_parallelize(usize::MAX - 1, false));
    }

    #[test]
    fn test_reset_threshold_restores_default() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(1);
        super::reset_parallel_threshold();
        assert_eq!(
            super::get_parallel_threshold(),
            super::DEFAULT_PARALLEL_THRESHOLD
        );
    }

    #[test]
    fn test_reset_simd_threshold_restores_default() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_simd_threshold(1);
        super::reset_simd_threshold();
        assert_eq!(super::get_simd_threshold(), super::DEFAULT_SIMD_THRESHOLD);
    }

    // === W10T6: Full dispatch unit tests per 30-dispatch.md §8.2 / §8.3 / §8.4 ===

    // --- Path selection — Serial ---

    #[test]
    fn test_exec_path_serial_below_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        // Default thresholds: parallel=65536, simd=64. len=1 is below both.
        let (path, guard) = super::select_exec_path(1, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Path selection — Parallel ---

    #[cfg(feature = "parallel")]
    #[test]
    fn test_exec_path_parallel_above_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Parallel);
        assert!(guard.is_some());
        drop(guard);
    }

    // --- Path selection — SIMD ---

    #[cfg(feature = "simd")]
    #[test]
    fn test_exec_path_simd_when_aligned() {
        let _threshold_guard = ThresholdTestGuard::new();
        // 30-dispatch §5.6 line 494-499: threshold=0 is the parallel-disable sentinel.
        super::set_parallel_threshold(0);
        let (path, guard) = super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, true, true);
        assert_eq!(path, ExecPath::Simd);
        assert!(guard.is_none());
    }

    // --- Non-contiguous penalty ---

    #[test]
    fn test_exec_path_serial_when_noncontiguous_below_doubled_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        // 30-dispatch §5.6 line 541: non-contiguous needs len >= 2*threshold.
        super::set_parallel_threshold(100);
        let (path, guard) = super::select_exec_path(199, false, true);
        assert_ne!(path, ExecPath::Parallel);
        assert!(guard.is_none());
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_rejected_when_noncontiguous() {
        let _threshold_guard = ThresholdTestGuard::new();
        // SIMD requires contiguous; non-contiguous must not select Simd.
        super::set_parallel_threshold(0); // disable parallel via sentinel
        let (path, _guard) = super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, false, true);
        assert_ne!(path, ExecPath::Simd);
        assert_eq!(path, ExecPath::Serial);
    }

    // --- Alignment hint pass-through ---

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_allows_misaligned_hint_when_contiguous() {
        let _threshold_guard = ThresholdTestGuard::new();
        // 30-dispatch §5.5 line 403-410: alignment_ok=false must NOT close SIMD.
        super::set_parallel_threshold(0); // disable parallel via sentinel
        let (path, _guard) = super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, true, false);
        assert_eq!(path, ExecPath::Simd);
    }

    // --- Priority: Parallel > SIMD ---

    #[cfg(all(feature = "parallel", feature = "simd"))]
    #[test]
    fn test_parallel_preferred_over_simd_for_large_input() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Parallel);
        assert!(guard.is_some());
        drop(guard);
    }

    // --- Feature gate combos ---

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_no_parallel_feature_never_returns_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, _guard) = super::select_exec_path(usize::MAX, true, true);
        assert_ne!(path, ExecPath::Parallel);
    }

    #[cfg(not(feature = "simd"))]
    #[test]
    fn test_no_simd_feature_never_returns_simd() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(0); // disable parallel via sentinel
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Determinism ---

    #[test]
    fn test_deterministic_same_input_same_output() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::reset_parallel_threshold();
        let first = super::select_exec_path(1, true, true).0;
        let second = super::select_exec_path(1, true, true).0;
        assert_eq!(first, second);
    }

    // --- Boundary: length extremes (§8.3) ---

    #[test]
    fn test_len_zero_returns_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(0, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    #[test]
    fn test_len_one_returns_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(1, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Boundary: threshold edges (§8.3) ---

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_threshold_boundary() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(100);
        // Exactly at threshold → eligible for parallel.
        let (path, _guard) = super::select_exec_path(100, true, true);
        assert_eq!(path, ExecPath::Parallel);
        // One below threshold → should NOT be parallel.
        let (path2, _guard2) = super::select_exec_path(99, true, true);
        assert_ne!(path2, ExecPath::Parallel);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_noncontiguous_doubled_threshold_boundary() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(100);
        // Non-contiguous: threshold × 2 = 200. 199 → not eligible; 200 → eligible.
        let (path, _guard) = super::select_exec_path(199, false, true);
        assert_ne!(path, ExecPath::Parallel);
        let (path2, _guard2) = super::select_exec_path(200, false, true);
        assert_eq!(path2, ExecPath::Parallel);
    }

    // --- Property: Monotonic path grade (§8.4) ---

    fn path_grade(path: ExecPath) -> u8 {
        match path {
            ExecPath::Serial => 0,
            ExecPath::Simd => 1,
            ExecPath::Parallel => 2,
        }
    }

    #[test]
    fn test_monotonic_path_grade() {
        let _threshold_guard = ThresholdTestGuard::new();
        // For increasing len with same other params, path grade never decreases.
        // 30-dispatch §8.4: Serial < Simd < Parallel
        let get_grade = |len: usize| -> u8 {
            let (path, guard) = super::select_exec_path(len, true, true);
            drop(guard);
            path_grade(path)
        };
        let sizes = [0usize, 1, 63, 64, 65_535, 65_536, 65_537];
        let mut max_so_far: u8 = 0;
        for &len in &sizes {
            let grade = get_grade(len);
            assert!(
                grade >= max_so_far,
                "monotonic violation: len={len}, grade={grade}, max_so_far={max_so_far}"
            );
            max_so_far = grade;
        }
    }
}
