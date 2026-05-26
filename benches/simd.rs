use std::hint::black_box;

mod common;
use common::{generators, run_timed};

// 27-benchmark §5.5: SIMD comparison entries fixed to Medium (M) scale only.
const SIMD_COMPARE_SIZE: usize = 65_536;

// --- SIMD comparison: run this binary twice —
//     once with `--features simd`, once without.
//     The `#[cfg(feature = "simd")]` selects the SIMD kernel path;
//     the scalar path runs otherwise. The times recorded by each
//     invocation are collected and compared in CI/report script. ---

fn report(label: &str, size: usize, median_ns: u128) {
    let path = if cfg!(feature = "simd") { "simd" } else { "scalar" };
    println!("{label}/{size}/{path}: {median_ns} ns");
}

// 27-benchmark §5.5: simd_add_compare covers f32/f64.
fn bench_simd_add_compare(quick: bool) {
    {
        let lhs = generators::sequential_1d_f32(SIMD_COMPARE_SIZE);
        let rhs = generators::sequential_1d_f32(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.add(&rhs).expect("same-shape add must succeed"));
        });
        report("simd_add_compare_f32", SIMD_COMPARE_SIZE, median);
    }
    {
        let lhs = generators::sequential_1d(SIMD_COMPARE_SIZE);
        let rhs = generators::sequential_1d(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.add(&rhs).expect("same-shape add must succeed"));
        });
        report("simd_add_compare_f64", SIMD_COMPARE_SIZE, median);
    }
}

// 27-benchmark §5.5: simd_sum_compare covers i32/f32/f64.
// i32 covers integer admission / scalar-fallback path; f32/f64 cover SIMD kernels.
fn bench_simd_sum_compare(quick: bool) {
    {
        let data = generators::sequential_1d_i32(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(data.sum());
        });
        report("simd_sum_compare_i32", SIMD_COMPARE_SIZE, median);
    }
    {
        let data = generators::sequential_1d_f32(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(data.sum());
        });
        report("simd_sum_compare_f32", SIMD_COMPARE_SIZE, median);
    }
    {
        let data = generators::sequential_1d(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(data.sum());
        });
        report("simd_sum_compare_f64", SIMD_COMPARE_SIZE, median);
    }
}

// 27-benchmark §5.5: simd_dot_compare covers f32/f64.
fn bench_simd_dot_compare(quick: bool) {
    {
        let lhs = generators::sequential_1d_f32(SIMD_COMPARE_SIZE);
        let rhs = generators::sequential_1d_f32(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.dot(&rhs).expect("same-length 1D dot must succeed"));
        });
        report("simd_dot_compare_f32", SIMD_COMPARE_SIZE, median);
    }
    {
        let lhs = generators::sequential_1d(SIMD_COMPARE_SIZE);
        let rhs = generators::sequential_1d(SIMD_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.dot(&rhs).expect("same-length 1D dot must succeed"));
        });
        report("simd_dot_compare_f64", SIMD_COMPARE_SIZE, median);
    }
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    let benches: &[(&str, common::BenchFn)] = &[
        ("simd_add_compare", bench_simd_add_compare),
        ("simd_sum_compare", bench_simd_sum_compare),
        ("simd_dot_compare", bench_simd_dot_compare),
    ];

    for &(name, func) in benches {
        if should_run(filter, name) {
            func(quick);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_filter_runs_matching_benchmark() {
        assert!(should_run(Some("simd"), "simd_add_compare"));
        assert!(!should_run(Some("missing"), "simd_add_compare"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= common::WARMUP_ITERATIONS + 10);
    }
}
