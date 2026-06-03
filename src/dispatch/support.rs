//! Test-support utilities for threshold mutation.
//!
//! Holds `ThresholdTestLock` (the global mutex marker) and
//! `ThresholdTestGuard` (the RAII snapshot/restore guard) so that
//! tests can safely mutate the global threshold atomics without
//! leaking state to subsequent tests.

#[cfg(any(test, feature = "parallel", feature = "simd"))]
use std::sync::{Mutex, MutexGuard};

#[cfg(any(test, feature = "parallel"))]
use super::{get_parallel_threshold, set_parallel_threshold};

#[cfg(any(test, feature = "simd"))]
use super::{get_simd_threshold, set_simd_threshold};

/// Marker type held inside `THRESHOLD_TEST_LOCK` to give the mutex a distinct
/// type without carrying meaningful data.
#[cfg(any(test, feature = "parallel", feature = "simd"))]
#[derive(Debug)]
struct ThresholdTestLock;

/// Serializes tests that mutate the global threshold atomics. Without this
/// lock, concurrent test execution would race on threshold state and produce
/// flaky results.
#[cfg(any(test, feature = "parallel", feature = "simd"))]
static THRESHOLD_TEST_LOCK: Mutex<ThresholdTestLock> = Mutex::new(ThresholdTestLock);

/// RAII guard that captures the current thresholds on construction and
/// restores them on drop, so threshold-mutating tests do not leak state to
/// subsequent tests. Holds `THRESHOLD_TEST_LOCK` for its lifetime to serialize
/// such tests against one another.
///
/// Gated like the threshold setters (`any(test, feature = ...)`) and
/// re-exported through `crate::prelude` so integration tests under `tests/`
/// (external crates) acquire the SAME process-global lock instead of racing
/// through a competing mechanism. NOT a stable public API.
///
/// Each threshold field is gated independently because the parallel and SIMD
/// setters are themselves feature-gated; this keeps the guard compilable under
/// any individual feature combination.
#[cfg(any(test, feature = "parallel", feature = "simd"))]
#[derive(Debug)]
pub struct ThresholdTestGuard<'lock> {
    _lock: MutexGuard<'lock, ThresholdTestLock>,
    #[cfg(any(test, feature = "parallel"))]
    parallel_threshold: usize,
    #[cfg(any(test, feature = "simd"))]
    simd_threshold: usize,
}

#[cfg(any(test, feature = "parallel", feature = "simd"))]
impl ThresholdTestGuard<'_> {
    /// Acquire the global threshold lock (recovering from poisoning) and
    /// snapshot the current parallel and SIMD thresholds.
    pub fn new() -> Self {
        let lock = match THRESHOLD_TEST_LOCK.lock() {
            Ok(lock) => lock,
            Err(poisoned) => poisoned.into_inner(),
        };
        Self {
            _lock: lock,
            #[cfg(any(test, feature = "parallel"))]
            parallel_threshold: get_parallel_threshold(),
            #[cfg(any(test, feature = "simd"))]
            simd_threshold: get_simd_threshold(),
        }
    }
}

#[cfg(any(test, feature = "parallel", feature = "simd"))]
impl Default for ThresholdTestGuard<'_> {
    /// Delegates to [`ThresholdTestGuard::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "parallel", feature = "simd"))]
impl Drop for ThresholdTestGuard<'_> {
    /// Restores the captured parallel and SIMD thresholds.
    fn drop(&mut self) {
        #[cfg(any(test, feature = "parallel"))]
        set_parallel_threshold(self.parallel_threshold);
        #[cfg(any(test, feature = "simd"))]
        set_simd_threshold(self.simd_threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::threshold::{DEFAULT_PARALLEL_THRESHOLD, DEFAULT_SIMD_THRESHOLD};

    /// Verify guard can be constructed and dropped (proves mutex
    /// acquisition does not deadlock).
    #[test]
    fn test_construction_and_drop() {
        let guard = ThresholdTestGuard::new();
        drop(guard);
    }

    /// Verify `Default` produces a valid guard.
    #[test]
    fn test_default() {
        let guard = ThresholdTestGuard::default();
        drop(guard);
    }

    /// Verify drop restores the parallel threshold after mutation.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_restores_parallel_threshold() {
        let original = get_parallel_threshold();
        {
            let _guard = ThresholdTestGuard::new();
            set_parallel_threshold(0);
            assert_eq!(get_parallel_threshold(), 0);
        }
        assert_eq!(get_parallel_threshold(), original);
    }

    /// Verify drop restores the SIMD threshold after mutation.
    #[cfg(feature = "simd")]
    #[test]
    fn test_restores_simd_threshold() {
        let original = get_simd_threshold();
        {
            let _guard = ThresholdTestGuard::new();
            set_simd_threshold(usize::MAX);
            assert_eq!(get_simd_threshold(), usize::MAX);
        }
        assert_eq!(get_simd_threshold(), original);
    }

    /// Verify drop restores both thresholds independently.
    #[cfg(all(feature = "parallel", feature = "simd"))]
    #[test]
    fn test_restores_both_thresholds() {
        let par_orig = get_parallel_threshold();
        let simd_orig = get_simd_threshold();
        {
            let _guard = ThresholdTestGuard::new();
            set_parallel_threshold(0);
            set_simd_threshold(usize::MAX);
        }
        assert_eq!(get_parallel_threshold(), par_orig);
        assert_eq!(get_simd_threshold(), simd_orig);
    }

    /// Verify the compile-time defaults match the documented values.
    #[test]
    fn test_default_threshold_values() {
        assert_eq!(DEFAULT_PARALLEL_THRESHOLD, 65_536);
        assert_eq!(DEFAULT_SIMD_THRESHOLD, 64);
    }
}
