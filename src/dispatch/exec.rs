//! Execution-path selection engine.
//!
//! Provides `select_exec_path()`, the thread-local `IN_PARALLEL` flag,
//! guard-acquisition helpers, the `with_parallel_worker_context` TLS
//! wrapper, and the private `dispatch_invalid_argument` error constructor.

#[cfg(feature = "parallel")]
use core::marker::PhantomData;

#[cfg(feature = "parallel")]
use core::cell::Cell;

#[cfg(feature = "parallel")]
use std::borrow::Cow;

#[cfg(feature = "parallel")]
use crate::error::{InvalidArgumentKind, XenonError};

#[cfg(feature = "parallel")]
use super::get_parallel_threshold;

#[cfg(feature = "simd")]
use super::get_simd_threshold;

use super::{ExecPath, ParallelGuard};

// ----------------------------------------------------------------------------
// Core dispatch functions
// ----------------------------------------------------------------------------

/// Selects the optimal execution path for an operation, atomically
/// binding "select Parallel" with "enter the parallel region".
pub fn select_exec_path(
    len: usize,
    is_contiguous: bool,
    alignment_ok: bool
) -> (ExecPath, Option<ParallelGuard>) {
    #[cfg(feature = "parallel")]
    if is_in_parallel() {
        return (ExecPath::Serial, None);
    }

    #[cfg(feature = "parallel")]
    {
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
        if parallel_eligible_by_threshold && let Some(guard) = try_acquire_guard() {
            return (ExecPath::Parallel, Some(guard));
        }
    }

    #[cfg(feature = "simd")]
    {
        if is_contiguous && len >= get_simd_threshold() {
            // alignment_ok is a hint to the simd backend; dispatch does
            // not gate SIMD on alignment.
            let _ = alignment_ok;
            return (ExecPath::Simd, None);
        }
    }

    // Discard unused params under feature-disabled builds.
    #[cfg(not(feature = "parallel"))]
    let _ = (len, is_contiguous);

    #[cfg(not(feature = "simd"))]
    let _ = alignment_ok;

    (ExecPath::Serial, None)
}

// ----------------------------------------------------------------------------
// Thread-local IN_PARALLEL flag and guard acquisition helpers
// ----------------------------------------------------------------------------

// Per-thread flag indicating the current thread is inside a library-internal
// parallel region. Set by `try_acquire_guard()` when a `ParallelGuard` is
// issued; cleared by the guard's `Drop`.
//
// Read by `select_exec_path()` to force nested calls to fall back to
// `Serial`, preventing rayon worker threads from re-entering rayon.
#[cfg(feature = "parallel")]
std::thread_local! {
    pub(crate) static IN_PARALLEL: Cell<bool> = const { Cell::new(false) };
}

/// Module-private: produces a guard only via `select_exec_path()`.
#[cfg(feature = "parallel")]
fn try_acquire_guard() -> Option<ParallelGuard> {
    IN_PARALLEL.with(|flag| {
        (!flag.replace(true)).then_some(ParallelGuard {
            _private: PhantomData,
        })
    })
}

/// Query-only: check if currently in parallel region without setting the flag.
#[cfg(feature = "parallel")]
fn is_in_parallel() -> bool {
    IN_PARALLEL.with(|flag| flag.get())
}

// ----------------------------------------------------------------------------
// with_parallel_worker_context — worker TLS helper
// ----------------------------------------------------------------------------

/// Runs `f` while marking the current worker thread as being inside a
/// Xenon-internal parallel region.
///
/// Used by `parallel` inside Rayon worker closures: outer `ParallelGuard`
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
        struct Reset<'a>(&'a Cell<bool>, bool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _reset = Reset(flag, previous);
        f()
    })
}

// ----------------------------------------------------------------------------
// Error construction helper
// ----------------------------------------------------------------------------

/// Construct a dispatch-specific `InvalidArgument` error with an
/// `InvalidConfig` detail. Private to `dispatch` so the error module
/// does not carry module-specific constructors.
#[cfg(feature = "parallel")]
pub(crate) fn dispatch_invalid_argument(
    argument: impl Into<Cow<'static, str>>,
    constraint: impl Into<Cow<'static, str>>,
    actual: impl Into<Cow<'static, str>>,
) -> XenonError {
    XenonError::InvalidArgument {
        operation: Cow::Borrowed("dispatch"),
        kind: InvalidArgumentKind::InvalidConfig {
            argument: argument.into(),
            constraint: constraint.into(),
            actual: actual.into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;
    use super::super::threshold::{DEFAULT_PARALLEL_THRESHOLD, DEFAULT_SIMD_THRESHOLD};

    // === select_exec_path ===

    /// Verify `select_exec_path` returns `Some(guard)` iff path is
    /// `Parallel`.
    #[test]
    fn test_select_returns_guard_iff_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = select_exec_path(usize::MAX, true, true);
        assert_eq!(path == ExecPath::Parallel, guard.is_some());
    }

    /// Verify a nested `select_exec_path()` falls back to `Serial`
    /// when an outer parallel guard is alive.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_nested_select_falls_back_to_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let _outer = try_acquire_guard().expect("outer guard acquisition");
        let (path, guard) = select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    /// Verify a nested SIMD-eligible call inside a worker context
    /// still falls back to `Serial`.
    #[cfg(all(feature = "parallel", feature = "simd"))]
    #[test]
    fn test_nested_simd_eligible_select_falls_back_to_serial() {
        let _threshold_guard = ThresholdTestGuard::new();

        let (path, guard) = with_parallel_worker_context(|| {
            select_exec_path(DEFAULT_SIMD_THRESHOLD, true, true)
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
        set_parallel_threshold(42);

        let (below_path, below_guard) = select_exec_path(41, true, true);
        assert_ne!(below_path, ExecPath::Parallel);
        assert!(below_guard.is_none());

        let (at_path, at_guard) = select_exec_path(42, true, true);
        assert_eq!(at_path, ExecPath::Parallel);
        assert!(at_guard.is_some());
        drop(at_guard);
    }

    /// Verify `threshold = 0` disables the parallel path entirely.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_zero_disables_parallel() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(0);

        let (path, guard) = select_exec_path(usize::MAX, true, true);
        assert_ne!(path, ExecPath::Parallel);
        assert!(guard.is_none());
    }

    /// Verify threshold near `usize::MAX` saturates without wrapping
    /// when doubled for non-contiguous inputs.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_threshold_saturating_mul_no_overflow() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(usize::MAX / 2 + 1);
        let (path, _guard) = select_exec_path(usize::MAX - 1, false, true);
        assert_ne!(path, ExecPath::Parallel);
    }

    /// Verify `reset_parallel_threshold` restores the compile-time default.
    #[test]
    fn test_reset_threshold_restores_default() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(1);
        reset_parallel_threshold();
        assert_eq!(
            get_parallel_threshold(),
            DEFAULT_PARALLEL_THRESHOLD
        );
    }

    /// Verify `reset_simd_threshold` restores the compile-time default.
    #[test]
    fn test_reset_simd_threshold_restores_default() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_simd_threshold(1);
        reset_simd_threshold();
        assert_eq!(get_simd_threshold(), DEFAULT_SIMD_THRESHOLD);
    }

    // === Full dispatch unit tests ===

    // --- Path selection — Serial ---

    /// Verify Serial path is chosen when `len` is below all thresholds.
    #[test]
    fn test_exec_path_serial_below_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = select_exec_path(1, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Path selection — Parallel ---

    /// Verify Parallel path is chosen for large inputs.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_exec_path_parallel_above_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = select_exec_path(usize::MAX, true, true);
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
        set_parallel_threshold(0);
        let (path, guard) = select_exec_path(DEFAULT_SIMD_THRESHOLD, true, true);
        assert_eq!(path, ExecPath::Simd);
        assert!(guard.is_none());
    }

    // --- Non-contiguous penalty ---

    /// Verify non-contiguous inputs need `len >= 2 * threshold` to
    /// qualify for the parallel path.
    #[test]
    fn test_exec_path_serial_when_noncontiguous_below_doubled_threshold() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(100);
        let (path, guard) = select_exec_path(199, false, true);
        assert_ne!(path, ExecPath::Parallel);
        assert!(guard.is_none());
    }

    /// Verify SIMD path is rejected for non-contiguous inputs.
    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_rejected_when_noncontiguous() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(0); // disable parallel via sentinel
        let (path, _guard) = select_exec_path(DEFAULT_SIMD_THRESHOLD, false, true);
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
        set_parallel_threshold(0); // disable parallel via sentinel
        let (path, _guard) = select_exec_path(DEFAULT_SIMD_THRESHOLD, true, false);
        assert_eq!(path, ExecPath::Simd);
    }

    // --- Priority: Parallel > SIMD ---

    /// Verify Parallel takes priority over SIMD for large inputs.
    #[cfg(all(feature = "parallel", feature = "simd"))]
    #[test]
    fn test_parallel_preferred_over_simd_for_large_input() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = select_exec_path(usize::MAX, true, true);
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
        let (path, _guard) = select_exec_path(usize::MAX, true, true);
        assert_ne!(path, ExecPath::Parallel);
    }

    /// Verify SIMD path is never returned without the `simd` feature.
    #[cfg(not(feature = "simd"))]
    #[test]
    fn test_no_simd_feature_never_returns_simd() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(0); // disable parallel via sentinel
        let (path, guard) = select_exec_path(usize::MAX, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    // --- Determinism ---

    /// Verify identical inputs produce identical path choices
    /// (no randomization in dispatch).
    #[test]
    fn test_deterministic_same_input_same_output() {
        let _threshold_guard = ThresholdTestGuard::new();
        reset_parallel_threshold();
        let first = select_exec_path(1, true, true).0;
        let second = select_exec_path(1, true, true).0;
        assert_eq!(first, second);
    }

    // --- Boundary: length extremes ---

    /// Verify `len == 0` returns Serial.
    #[test]
    fn test_len_zero_returns_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = select_exec_path(0, true, true);
        assert_eq!(path, ExecPath::Serial);
        assert!(guard.is_none());
    }

    /// Verify `len == 1` returns Serial.
    #[test]
    fn test_len_one_returns_serial() {
        let _threshold_guard = ThresholdTestGuard::new();
        let (path, guard) = select_exec_path(1, true, true);
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
        set_parallel_threshold(100);
        // Exactly at threshold → eligible for parallel.
        let (path, _guard) = select_exec_path(100, true, true);
        assert_eq!(path, ExecPath::Parallel);
        // One below threshold → should NOT be parallel.
        let (path2, _guard2) = select_exec_path(99, true, true);
        assert_ne!(path2, ExecPath::Parallel);
    }

    /// Verify non-contiguous boundary: `2 * threshold` is eligible,
    /// `2 * threshold - 1` is not.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_noncontiguous_doubled_threshold_boundary() {
        let _threshold_guard = ThresholdTestGuard::new();
        set_parallel_threshold(100);
        // Non-contiguous: threshold × 2 = 200. 199 → not eligible; 200 → eligible.
        let (path, _guard) = select_exec_path(199, false, true);
        assert_ne!(path, ExecPath::Parallel);
        let (path2, _guard2) = select_exec_path(200, false, true);
        assert_eq!(path2, ExecPath::Parallel);
    }

    // --- Property: Monotonic path grade ---

    /// Map an `ExecPath` to a numeric grade so monotonicity can be
    /// asserted: Serial (0) < Simd (1) < Parallel (2).
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
            let (path, guard) = select_exec_path(len, true, true);
            #[cfg(feature = "parallel")]
            drop(guard);
            #[cfg(not(feature = "parallel"))]
            let _ = guard;
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
