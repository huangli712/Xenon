//! Runtime-overridable dispatch thresholds.
//!
//! Compile-time defaults, lock-free atomic storage, and the
//! public `set_parallel_threshold` / `set_simd_threshold` /
//! `reset_parallel_threshold` / `reset_simd_threshold` API used
//! by benchmarks and integration tests.

#[cfg(any(test, feature = "parallel", feature = "simd"))]
use std::sync::atomic::{AtomicUsize, Ordering};

// ----------------------------------------------------------------------------
// Threshold storage — constants
// ----------------------------------------------------------------------------

/// Compile-time default for parallel threshold.
#[cfg(any(test, feature = "parallel"))]
pub(crate) const DEFAULT_PARALLEL_THRESHOLD: usize = 65_536;

/// Compile-time default for SIMD threshold.
#[cfg(any(test, feature = "simd"))]
pub(crate) const DEFAULT_SIMD_THRESHOLD: usize = 64;

// ----------------------------------------------------------------------------
// Threshold storage — atomics
// ----------------------------------------------------------------------------

/// Runtime-overridable parallel threshold.
///
/// Uses `AtomicUsize` for lock-free reads. Written only during
/// initialization or explicit override (testing/benchmarking).
#[cfg(any(test, feature = "parallel"))]
static PARALLEL_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_PARALLEL_THRESHOLD);

/// Runtime-overridable SIMD threshold.
#[cfg(any(test, feature = "simd"))]
static SIMD_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_SIMD_THRESHOLD);

// ----------------------------------------------------------------------------
// Threshold storage — getters
// ----------------------------------------------------------------------------

#[cfg(any(test, feature = "parallel"))]
pub(crate) fn get_parallel_threshold() -> usize {
    PARALLEL_THRESHOLD.load(Ordering::Relaxed)
}

#[cfg(any(test, feature = "simd"))]
pub(crate) fn get_simd_threshold() -> usize {
    SIMD_THRESHOLD.load(Ordering::Relaxed)
}

// ----------------------------------------------------------------------------
// Threshold runtime override API (testing/benchmarking only)
// ----------------------------------------------------------------------------

/// Override the parallel threshold at runtime.
///
/// Setting `threshold = 0` disables the parallel path entirely (sentinel).
#[cfg(any(test, feature = "parallel"))]
pub fn set_parallel_threshold(threshold: usize) {
    PARALLEL_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// Override the SIMD threshold at runtime.
///
/// Use `usize::MAX` to disable the SIMD path (sentinel).
#[cfg(any(test, feature = "simd"))]
pub fn set_simd_threshold(threshold: usize) {
    SIMD_THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// Reset the parallel threshold to its compile-time default.
#[cfg(any(test, feature = "parallel"))]
pub fn reset_parallel_threshold() {
    set_parallel_threshold(DEFAULT_PARALLEL_THRESHOLD);
}

/// Reset the SIMD threshold to its compile-time default.
#[cfg(any(test, feature = "simd"))]
pub fn reset_simd_threshold() {
    set_simd_threshold(DEFAULT_SIMD_THRESHOLD);
}
