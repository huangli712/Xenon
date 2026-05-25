use std::hint::black_box;

mod utils;
use utils::{generators, run_timed};

// 27-benchmark §5.5: parallel comparison entries fixed to Large (L) scale only.
const PARALLEL_COMPARE_SIZE: usize = 16_777_216;

// --- parallel comparison: run this binary twice —
//     once with `--features parallel`, once without.
//     The `#[cfg(feature = "parallel")]` selects the parallel path;
//     the serial path runs otherwise. Times are collected by CI/report. ---

fn report(label: &str, size: usize, median_ns: u128) {
    let path = if cfg!(feature = "parallel") { "parallel" } else { "serial" };
    println!("{label}/{size}/{path}: {median_ns} ns");
}

// 27-benchmark §5.5: par_add_compare uses f64 only.
fn bench_par_add_compare(quick: bool) {
    let lhs = generators::sequential_1d(PARALLEL_COMPARE_SIZE);
    let rhs = generators::sequential_1d(PARALLEL_COMPARE_SIZE);
    let median = run_timed(quick, || {
        let _result = black_box(lhs.add(&rhs).expect("same-shape add must succeed"));
    });
    report("par_add_compare_f64", PARALLEL_COMPARE_SIZE, median);
}

// 27-benchmark §5.5: par_sum_compare is fixed to i64 (parallel reduction kernel).
fn bench_par_sum_compare(quick: bool) {
    let data = generators::sequential_1d_i64(PARALLEL_COMPARE_SIZE);
    let median = run_timed(quick, || {
        let _result = black_box(data.sum());
    });
    report("par_sum_compare_i64", PARALLEL_COMPARE_SIZE, median);
}

// 27-benchmark §5.5: par_dot_compare covers f64 and Complex<f64>.
fn bench_par_dot_compare(quick: bool) {
    {
        let lhs = generators::sequential_1d(PARALLEL_COMPARE_SIZE);
        let rhs = generators::sequential_1d(PARALLEL_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.dot(&rhs).expect("same-length 1D dot must succeed"));
        });
        report("par_dot_compare_f64", PARALLEL_COMPARE_SIZE, median);
    }
    {
        let lhs = generators::complex_1d(PARALLEL_COMPARE_SIZE);
        let rhs = generators::complex_1d(PARALLEL_COMPARE_SIZE);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.dot(&rhs).expect("same-length 1D dot must succeed"));
        });
        report("par_dot_compare_complex", PARALLEL_COMPARE_SIZE, median);
    }
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    let benches: &[(&str, utils::BenchFn)] = &[
        ("par_add_compare", bench_par_add_compare),
        ("par_sum_compare", bench_par_sum_compare),
        ("par_dot_compare", bench_par_dot_compare),
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
        assert!(should_run(Some("par"), "par_sum_compare"));
        assert!(!should_run(Some("missing"), "par_sum_compare"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= utils::WARMUP_ITERATIONS + 10);
    }
}
