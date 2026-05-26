//! Internal execution-path dispatch.
//!
//! Arbitrates between Serial, SIMD, and Parallel execution paths based on
//! input length, contiguity, alignment hints, runtime thresholds, and
//! feature gates. Holds the nested-parallel guard (TLS flag) so library
//! code never starts a parallel region inside another parallel region.
//!
//! All items here are `pub(crate)`. A minimal subset is re-exported
//! through `crate::prelude` so integration tests under `tests/` (which
//! are external crates) can observe dispatch decisions, tweak
//! thresholds, and exercise parallel kernels directly. Those
//! re-exports are NOT a stable public API.

// ---------------------------------------------------------------------------
// Threshold storage — constants
// ---------------------------------------------------------------------------

/// Compile-time default for parallel threshold.
const DEFAULT_PARALLEL_THRESHOLD: usize = 65_536;

/// Compile-time default for SIMD threshold.
const DEFAULT_SIMD_THRESHOLD: usize = 64;

// ---------------------------------------------------------------------------
// Threshold storage — atomics
// ---------------------------------------------------------------------------

/// Runtime-overridable parallel threshold.
///
/// Uses `AtomicUsize` for lock-free reads. Written only during
/// initialization or explicit override (testing/benchmarking).
static PARALLEL_THRESHOLD: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// Runtime-overridable SIMD threshold.
static SIMD_THRESHOLD: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_SIMD_THRESHOLD);

// ---------------------------------------------------------------------------
// Threshold storage — getters
// ---------------------------------------------------------------------------

fn get_parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg_attr(not(feature = "simd"), allow(dead_code))]
fn get_simd_threshold() -> usize {
    SIMD_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Threshold storage — setters
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Threshold runtime override API (testing/benchmarking only)
// ---------------------------------------------------------------------------

/// Override the parallel threshold at runtime.
///
/// Setting `threshold = 0` disables the parallel path entirely (sentinel).
#[cfg(any(test, feature = "parallel"))]
pub fn set_parallel_threshold(threshold: usize) {
    PARALLEL_THRESHOLD.store(threshold, std::sync::atomic::Ordering::Relaxed);
}

/// Reset the parallel threshold to its compile-time default.
#[cfg(any(test, feature = "parallel"))]
pub fn reset_parallel_threshold() {
    set_parallel_threshold(DEFAULT_PARALLEL_THRESHOLD);
}

/// Override the SIMD threshold at runtime.
///
/// Use `usize::MAX` to disable the SIMD path (sentinel).
#[cfg(any(test, feature = "simd"))]
pub fn set_simd_threshold(threshold: usize) {
    SIMD_THRESHOLD.store(threshold, std::sync::atomic::Ordering::Relaxed);
}

/// Reset the SIMD threshold to its compile-time default.
#[cfg(any(test, feature = "simd"))]
pub fn reset_simd_threshold() {
    set_simd_threshold(DEFAULT_SIMD_THRESHOLD);
}





/// Three mutually exclusive execution paths recommended by dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecPath {
    /// Serial scalar execution. Default fallback when neither SIMD
    /// nor parallel preconditions are met.
    Serial,
    /// Serial path with SIMD acceleration. dispatch only signals
    /// "SIMD path is preferred"; the `simd/` backend retains final
    /// admission (ISA, lane width, alignment).
    Simd,
    /// Parallel execution. Returned only when `feature = "parallel"`
    /// is enabled, the input meets the parallel threshold, and the
    /// current thread is not already inside a library-internal
    /// parallel region. Always accompanied by `Some(ParallelGuard)`.
    Parallel,
}

/// Execution strategy parameters consumed by the parallel backend.
///
/// Defined here; consumed by `parallel/` module functions such as
/// `par_map`, `par_zip_map`, `par_sum`, `par_dot`. Fields are private
/// to enforce construction via `ParallelExecStrategy::new()`.
///
/// Only compiled under `feature = "parallel"`: outside that feature
/// `select_exec_path()` can never return `ExecPath::Parallel`, so the
/// strategy type itself is unreachable.
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
    /// validation so that the parallel/ backend can consume the value
    /// without re-validation.
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
            // Read the pool size once at construction time.
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
#[derive(Debug)]
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
#[derive(Debug)]
pub struct ParallelGuard {
    _private: core::marker::PhantomData<()>,
}


// ---------------------------------------------------------------------------
// Core dispatch functions
// ---------------------------------------------------------------------------

/// Selects the optimal execution path for an operation, atomically
/// binding "select Parallel" with "enter the parallel region".
pub fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool,
) -> (ExecPath, Option<ParallelGuard>) {
    #[cfg(feature = "parallel")]
    if is_in_parallel() {
        return (ExecPath::Serial, None);
    }

    // Zero sentinel disables parallel; non-contiguous gets saturating
    // doubled threshold.
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
            // alignment_ok is a hint to the simd backend; dispatch does
            // not gate SIMD on alignment.
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
/// check so the result matches what `select_exec_path()` would do.
#[cfg(feature = "parallel")]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "dispatch is staged before downstream integration")
)]
pub(crate) fn should_parallelize(len: usize, is_contiguous: bool) -> bool {
    let base = get_parallel_threshold();
    // Zero sentinel disables; in-parallel TLS suppresses nested parallel.
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

    // === ExecPath enum smoke ===

    /// Verify `ExecPath` is constructable and equality-comparable.
    #[test]
    fn test_exec_path_enum_defined() {
        assert_eq!(ExecPath::Serial, ExecPath::Serial);
    }

    // === ParallelGuard and worker context ===

    /// Verify dropping a `ParallelGuard` releases the TLS flag so a
    /// subsequent `select_exec_path()` can re-enter the parallel region.
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

    /// Verify `with_parallel_worker_context` saves the previous TLS
    /// value, sets it to true, runs `f`, and restores on drop.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_worker_context_restores_previous_flag() {
        assert!(!super::is_in_parallel());
        let observed = super::with_parallel_worker_context(super::is_in_parallel);
        assert!(observed, "TLS must be true inside the worker context");
        assert!(
            !super::is_in_parallel(),
            "TLS must be restored after the context exits"
        );
    }

    /// Verify `ParallelGuard::drop` clears the TLS flag even during
    /// panic unwind.
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

    /// Verify the worker-context RAII restores the TLS flag during
    /// panic unwind.
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

    /// Verify nested worker contexts: inner runs with true, outer
    /// restores its own previous value (also true), then outermost
    /// restores false.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_worker_context_nests_correctly() {
        let outer_inner = super::with_parallel_worker_context(|| {
            assert!(super::is_in_parallel());
            super::with_parallel_worker_context(super::is_in_parallel)
        });
        assert!(outer_inner);
        assert!(!super::is_in_parallel());
    }

    /// Verify `ParallelGuard` is `!Send + !Sync` so the TLS flag stays
    /// bound to the acquiring thread.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_guard_is_not_send_or_sync() {
        static_assertions::assert_not_impl_any!(super::ParallelGuard: Send, Sync);
    }

    /// Verify the placeholder `ParallelGuard` (without `parallel`
    /// feature) is `Send + Sync` so `(ExecPath, Option<_>)` stays so too.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_placeholder_parallel_guard_is_send_sync() {
        static_assertions::assert_impl_all!(super::ParallelGuard: Send, Sync);
    }

    // === ParallelExecStrategy construction ===

    /// Verify `new()` rejects `chunk_size=0` and `max_workers=0`.
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

    /// Verify `new(None, None)` produces a strategy equivalent to `auto()`.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_accepts_none() {
        let lhs =
            super::ParallelExecStrategy::new(None, None).expect("new(None, None) should succeed");
        let rhs = super::ParallelExecStrategy::auto();
        assert_eq!(lhs.chunk_size(), rhs.chunk_size());
        assert_eq!(lhs.max_workers(), rhs.max_workers());
    }

    /// Verify `new()` rejects `max_workers` exceeding the rayon pool size.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_rejects_oversized_max_workers() {
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

    // === select_exec_path / should_parallelize ===

    /// Verify `select_exec_path` returns `Some(guard)` iff path is
    /// `Parallel`.
    #[test]
    fn test_select_returns_guard_iff_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path == ExecPath::Parallel, guard.is_some());
    }

    /// Verify `should_parallelize` is a pure query: it must not consume
    /// the nested-parallel slot that `select_exec_path()` needs to
    /// return `(Parallel, Some(_))`.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_should_parallelize_diagnostic_does_not_acquire_guard() {
        let _threshold_guard = ThresholdTestGuard::new();
        assert!(super::should_parallelize(usize::MAX, true));

        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Parallel);
        assert!(guard.is_some());
        drop(guard);
    }

    /// Verify `should_parallelize` returns false when already inside a
    /// parallel region, even if length threshold is satisfied.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_should_parallelize_returns_false_when_in_parallel_region() {
        let _threshold_guard = ThresholdTestGuard::new();
        let _outer = super::try_acquire_guard().expect("outer guard acquisition");
        assert!(!super::should_parallelize(usize::MAX, true));
    }

    /// Verify a nested `select_exec_path()` falls back to `Serial`
    /// when an outer parallel guard is alive.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_nested_select_falls_back_to_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let _outer = super::try_acquire_guard().expect("outer guard acquisition");
        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    /// Verify a nested SIMD-eligible call inside a worker context
    /// still falls back to `Serial`.
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

    // === Threshold configuration ===

    /// Verify runtime threshold override is honored by `select_exec_path()`.
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

    /// Verify `threshold = 0` disables the parallel path entirely.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_zero_disables_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(0);
        assert!(!super::should_parallelize(usize::MAX, true));

        let (path, guard) = super::select_exec_path(usize::MAX, true, true);
        assert_ne!(path, ExecPath::Parallel);
        assert!(guard.is_none());
    }

    /// Verify threshold near `usize::MAX` saturates without wrapping
    /// when doubled for non-contiguous inputs.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_saturating_mul_no_overflow() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(usize::MAX / 2 + 1);
        assert!(!super::should_parallelize(usize::MAX - 1, false));
    }

    /// Verify `reset_parallel_threshold` restores the compile-time default.
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

    /// Verify `reset_simd_threshold` restores the compile-time default.
    #[test]
    fn test_reset_simd_threshold_restores_default() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_simd_threshold(1);
        super::reset_simd_threshold();
        assert_eq!(super::get_simd_threshold(), super::DEFAULT_SIMD_THRESHOLD);
    }

    // === Full dispatch unit tests ===

    // --- Path selection — Serial ---

    /// Verify Serial path is chosen when `len` is below all thresholds.
    #[test]
    fn test_exec_path_serial_below_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(1, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Path selection — Parallel ---

    /// Verify Parallel path is chosen for large inputs.
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

    /// Verify SIMD path is chosen for aligned contiguous inputs when
    /// the parallel path is disabled.
    #[cfg(feature = "simd")]
    #[test]
    fn test_exec_path_simd_when_aligned() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(0);
        let (path, guard) = super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, true, true);
        assert_eq!(path, ExecPath::Simd);
        assert!(guard.is_none());
    }

    // --- Non-contiguous penalty ---

    /// Verify non-contiguous inputs need `len >= 2 * threshold` to
    /// qualify for the parallel path.
    #[test]
    fn test_exec_path_serial_when_noncontiguous_below_doubled_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(100);
        let (path, guard) = super::select_exec_path(199, false, true);
        assert_ne!(path, ExecPath::Parallel);
        assert!(guard.is_none());
    }

    /// Verify SIMD path is rejected for non-contiguous inputs.
    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_rejected_when_noncontiguous() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(0); // disable parallel via sentinel
        let (path, _guard) = super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, false, true);
        assert_ne!(path, ExecPath::Simd);
        assert_eq!(path, ExecPath::Serial);
    }

    // --- Alignment hint pass-through ---

    /// Verify `alignment_ok=false` is a hint only and does NOT close
    /// the SIMD path.
    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_allows_misaligned_hint_when_contiguous() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::set_parallel_threshold(0); // disable parallel via sentinel
        let (path, _guard) = super::select_exec_path(super::DEFAULT_SIMD_THRESHOLD, true, false);
        assert_eq!(path, ExecPath::Simd);
    }

    // --- Priority: Parallel > SIMD ---

    /// Verify Parallel takes priority over SIMD for large inputs.
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

    /// Verify Parallel path is never returned without the `parallel`
    /// feature.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_no_parallel_feature_never_returns_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, _guard) = super::select_exec_path(usize::MAX, true, true);
        assert_ne!(path, ExecPath::Parallel);
    }

    /// Verify SIMD path is never returned without the `simd` feature.
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

    /// Verify identical inputs produce identical path choices
    /// (no randomization in dispatch).
    #[test]
    fn test_deterministic_same_input_same_output() {
        let _threshold_guard = ThresholdTestGuard::new();
        super::reset_parallel_threshold();
        let first = super::select_exec_path(1, true, true).0;
        let second = super::select_exec_path(1, true, true).0;
        assert_eq!(first, second);
    }

    // --- Boundary: length extremes ---

    /// Verify `len == 0` returns Serial.
    #[test]
    fn test_len_zero_returns_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(0, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    /// Verify `len == 1` returns Serial.
    #[test]
    fn test_len_one_returns_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = super::select_exec_path(1, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Boundary: threshold edges ---

    /// Verify boundary: `len == threshold` is eligible, `len == threshold - 1`
    /// is not.
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

    /// Verify non-contiguous boundary: `2 * threshold` is eligible,
    /// `2 * threshold - 1` is not.
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

    // --- Property: Monotonic path grade ---

    fn path_grade(path: ExecPath) -> u8 {
        match path {
            ExecPath::Serial => 0,
            ExecPath::Simd => 1,
            ExecPath::Parallel => 2,
        }
    }

    /// Verify path grade is non-decreasing as `len` grows with other
    /// parameters fixed (Serial < Simd < Parallel).
    #[test]
    fn test_monotonic_path_grade() {
        let _threshold_guard = ThresholdTestGuard::new();
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
