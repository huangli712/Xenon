#[cfg(any(test, feature = "parallel", feature = "simd"))]
use std::sync::{Mutex, MutexGuard};

#[cfg(any(test, feature = "parallel", feature = "simd"))]
use super::{get_parallel_threshold, get_simd_threshold};
#[cfg(any(test, feature = "parallel"))]
use super::set_parallel_threshold;
#[cfg(any(test, feature = "simd"))]
use super::set_simd_threshold;

// ---------------------------------------------------------------------------
// Threshold test guard — serialization + state restoration for tests
// ---------------------------------------------------------------------------

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
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "parallel", feature = "simd"))]
impl Drop for ThresholdTestGuard<'_> {
    fn drop(&mut self) {
        #[cfg(any(test, feature = "parallel"))]
        set_parallel_threshold(self.parallel_threshold);
        #[cfg(any(test, feature = "simd"))]
        set_simd_threshold(self.simd_threshold);
    }
}
