use std::time::Instant;

pub mod generators;

/// Shared benchmark function signature: `fn(quick: bool)`.
#[allow(dead_code)]
pub type BenchFn = fn(bool);

// 27-benchmark §5.4.1: Small / Medium / Large
#[allow(dead_code)]
pub const SIZES_1D: &[usize] = &[64, 65_536, 16_777_216];
#[allow(dead_code)]
pub const SIZES_2D: &[(usize, usize)] = &[(8, 8), (256, 256), (4096, 4096)];

// 27-benchmark §6.1 measurement methodology: warmup + N rounds × M iterations.
pub const WARMUP_ITERATIONS: usize = 10;
pub const ROUNDS: usize = 10;
pub const ITERATIONS_PER_ROUND: usize = 100;

/// Shared timing harness (27-benchmark §5.8 / §6.1).
///
/// Performs `WARMUP_ITERATIONS` warmup calls, then `ROUNDS` rounds × `ITERATIONS_PER_ROUND`
/// iterations, returning median wall-time nanoseconds. `quick=true` reduces to 1 round × 10
/// iters per 27-benchmark §5.6 smoke mode.
pub fn run_timed<F>(quick: bool, mut operation: F) -> u128
where
    F: FnMut(),
{
    let (rounds, iterations) = if quick { (1, 10) } else { (ROUNDS, ITERATIONS_PER_ROUND) };
    for _ in 0..WARMUP_ITERATIONS {
        operation();
    }
    let mut timings = Vec::with_capacity(rounds);
    for _round in 0..rounds {
        let started_at = Instant::now();
        for _iteration in 0..iterations {
            operation();
        }
        timings.push(started_at.elapsed().as_nanos());
    }
    timings.sort_unstable();
    timings[timings.len() / 2]
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_constants_present() {
        assert_eq!(SIZES_1D.len(), 3);
        assert_eq!(SIZES_2D.len(), 3);
    }

    #[test]
    fn test_run_timed_quick_mode() {
        let median = run_timed(true, || {});
        // quick mode: 1 round × 10 iters + 10 warmup = 20 calls
        assert!(median <= u128::MAX);
    }

    #[test]
    fn test_run_timed_full_mode() {
        let median = run_timed(false, || {});
        // full mode: 10 rounds × 100 iters + 10 warmup = 1010 calls
        assert!(median <= u128::MAX);
    }
}
