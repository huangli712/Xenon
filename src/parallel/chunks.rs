//! Chunk-size computation for parallel splitting.
//!
//! W15: compute_safe_chunks — per-chunk element count for rayon splitting.

/// Compute a per-chunk element count for parallel splitting.
///
/// Algorithm (09-parallel §6.3):
/// - `MIN_CHUNK = 1024`           : lower bound to amortize rayon scheduling.
/// - `TARGET_CHUNKS_PER_WORKER=4` : give work-stealing slack for tail balance.
/// - `total == 0`        → returns 1 (dummy; no work will be scheduled).
/// - `total <= workers`  → returns 1 (one element per worker; rest idle).
/// - otherwise           → max(ceil_div(total, workers*4), MIN_CHUNK).
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) fn compute_safe_chunks(total: usize, num_workers: usize) -> usize {
    const MIN_CHUNK: usize = 1024;
    const TARGET_CHUNKS_PER_WORKER: usize = 4;

    if total == 0 {
        return 1;
    }
    if total <= num_workers {
        return 1;
    }
    let target_chunks = num_workers.saturating_mul(TARGET_CHUNKS_PER_WORKER);
    let raw = total.div_ceil(target_chunks);
    raw.max(MIN_CHUNK)
}

#[cfg(test)]
mod chunk_tests {
    use super::compute_safe_chunks;

    #[test]
    fn test_compute_safe_chunks_empty_returns_one() {
        // 09-parallel §6.3 line 350: total == 0 → 1 (dummy)
        assert_eq!(compute_safe_chunks(0, 8), 1);
    }

    #[test]
    fn test_compute_safe_chunks_few_elements_returns_one() {
        // 09-parallel §6.3 line 353: total <= workers → 1
        assert_eq!(compute_safe_chunks(5, 8), 1);
    }

    #[test]
    fn test_compute_safe_chunks_respects_min_chunk() {
        // total=10_000, workers=8 → ceil(10_000 / 32) = 313 < MIN_CHUNK(1024)
        // → returns 1024
        assert_eq!(compute_safe_chunks(10_000, 8), 1024);
    }

    #[test]
    fn test_compute_safe_chunks_large_total_overrides_min() {
        // total=1_000_000, workers=8 → ceil(1_000_000 / 32) = 31_250 > 1024
        // → returns 31_250
        assert_eq!(compute_safe_chunks(1_000_000, 8), 31_250);
    }
}
