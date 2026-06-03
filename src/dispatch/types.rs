//! Execution-path types consumed by dispatch.
//!
//! Defines `ExecPath` (Serial / Simd / Parallel), the parallel
//! `ParallelExecStrategy`, and the `ParallelGuard` RAII sentinel
//! (with its placeholder variant for non-parallel builds).

use core::marker::PhantomData;

#[cfg(feature = "parallel")]
use crate::error::Result;

#[cfg(feature = "parallel")]
use super::exec::{dispatch_invalid_argument, IN_PARALLEL};

// ----------------------------------------------------------------------------
// Execution path types
// ----------------------------------------------------------------------------

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
    pub fn new(chunk_size: Option<usize>, max_workers: Option<usize>) -> Result<Self> {
        if matches!(chunk_size, Some(0)) {
            return Err(
                dispatch_invalid_argument("chunk_size", "must be non-zero", "0")
            );
        }
        if matches!(max_workers, Some(0)) {
            return Err(
                dispatch_invalid_argument("max_workers", "must be non-zero", "0")
            );
        }
        if let Some(n) = max_workers {
            // Read the pool size once at construction time.
            let pool = rayon::current_num_threads();
            if n > pool {
                return Err(
                    dispatch_invalid_argument(
                        "max_workers",
                        format!("must not exceed rayon pool size ({pool})"),
                        n.to_string()
                    )
                );
            }
        }
        Ok(Self { chunk_size, max_workers })
    }

    /// Default strategy: let the parallel backend decide everything.
    pub fn auto() -> Self {
        Self { chunk_size: None, max_workers: None }
    }

    /// Suggested chunk size for parallel processing, or `None` to let the
    /// parallel backend decide.
    pub(crate) fn chunk_size(&self) -> Option<usize> {
        self.chunk_size
    }

    /// Maximum worker count for the parallel backend, or `None` for the
    /// rayon default pool size.
    pub(crate) fn max_workers(&self) -> Option<usize> {
        self.max_workers
    }
}

// ----------------------------------------------------------------------------
// ParallelGuard — nested-parallel guard (feature-gated)
// ----------------------------------------------------------------------------

/// RAII guard that indicates the current thread is inside a
/// library-internal parallel region.
///
/// A `ParallelGuard` value is **only ever obtained as the second tuple
/// element of `select_exec_path()`** when that function selects
/// `ExecPath::Parallel`. There is no public `enter()` constructor.
///
/// While the guard is alive, the thread-local flag is set. Any nested
/// call to `select_exec_path()` will observe `is_in_parallel() == true`
/// and fall back to `Serial`.
///
/// Dropping the guard clears the thread-local flag.
///
/// Under `feature = "parallel"`, the guard is `!Send + !Sync` because
/// its `Drop` clears the **current** thread's TLS flag.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub struct ParallelGuard {
    pub(crate) _private: PhantomData<*const ()>,
}

/// Placeholder `ParallelGuard` when `feature = "parallel"` is disabled.
///
/// Zero-size, never constructed; no Drop, intentionally `Send + Sync`.
/// This keeps `(ExecPath, Option<ParallelGuard>)` `Send + Sync` in
/// default builds where the option is always `None`.
#[cfg(not(feature = "parallel"))]
#[derive(Debug)]
pub struct ParallelGuard {
    pub(crate) _private: PhantomData<()>,
}

#[cfg(feature = "parallel")]
impl Drop for ParallelGuard {
    /// Clears the thread-local `IN_PARALLEL` flag, allowing nested
    /// `select_exec_path` calls to re-enter the parallel region.
    fn drop(&mut self) {
        IN_PARALLEL.with(|flag| flag.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "parallel")]
    use crate::error::{InvalidArgumentKind, XenonError};

    // === ExecPath enum smoke ===

    /// Verify `ExecPath` is constructable and equality-comparable.
    #[test]
    fn test_exec_path_enum_defined() {
        assert_eq!(ExecPath::Serial, ExecPath::Serial);
    }

    // === ParallelGuard Send/Sync ===

    /// Verify `ParallelGuard` is `!Send + !Sync` so the TLS flag stays
    /// bound to the acquiring thread.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_guard_is_not_send_or_sync() {
        static_assertions::assert_not_impl_any!(ParallelGuard: Send, Sync);
    }

    /// Verify the placeholder `ParallelGuard` (without `parallel`
    /// feature) is `Send + Sync` so `(ExecPath, Option<_>)` stays so too.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_placeholder_parallel_guard_is_send_sync() {
        static_assertions::assert_impl_all!(ParallelGuard: Send, Sync);
    }

    // === ParallelExecStrategy construction ===

    /// Verify `new()` rejects `chunk_size=0` and `max_workers=0`.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_rejects_zero() {
        let chunk_size_error = ParallelExecStrategy::new(Some(0), None)
            .expect_err("chunk_size=0 must be rejected");
        assert!(matches!(
            chunk_size_error,
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::InvalidConfig {
                    argument,
                    constraint,
                    actual,
                },
                ..
            } if argument == "chunk_size"
                && constraint == "must be non-zero"
                && actual == "0"
        ));

        let max_workers_error = ParallelExecStrategy::new(None, Some(0))
            .expect_err("max_workers=0 must be rejected");
        assert!(matches!(
            max_workers_error,
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::InvalidConfig {
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
            ParallelExecStrategy::new(None, None).expect("new(None, None) should succeed");
        let rhs = ParallelExecStrategy::auto();
        assert_eq!(lhs.chunk_size(), rhs.chunk_size());
        assert_eq!(lhs.max_workers(), rhs.max_workers());
    }

    /// Verify `new()` rejects `max_workers` exceeding the rayon pool size.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_strategy_new_rejects_oversized_max_workers() {
        let pool = rayon::current_num_threads();
        let actual = pool.saturating_add(1);
        let error = ParallelExecStrategy::new(None, Some(actual))
            .expect_err("oversized max_workers must be rejected");

        assert!(matches!(
            error,
            XenonError::InvalidArgument {
                kind: InvalidArgumentKind::InvalidConfig {
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
}
