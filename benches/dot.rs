use std::hint::black_box;

mod utils;
use utils::{generators, run_timed, SIZES_1D};

fn bench_dot_1d_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let rhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.dot(&rhs).expect("same-length 1D dot must succeed"));
        });
        println!("dot_1d_f64/{size}: {median} ns");
    }
}

fn bench_dot_1d_complex(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::complex_1d(size);
        let rhs = generators::complex_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.dot(&rhs).expect("same-length 1D dot must succeed"));
        });
        println!("dot_1d_complex/{size}: {median} ns");
    }
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    if should_run(filter, "dot_1d_f64") { bench_dot_1d_f64(quick); }
    if should_run(filter, "dot_1d_complex") { bench_dot_1d_complex(quick); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_filter_runs_matching_benchmark() {
        assert!(should_run(Some("dot"), "dot_1d_f64"));
        assert!(!should_run(Some("missing"), "dot_1d_f64"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= utils::WARMUP_ITERATIONS + 10);
    }
}
