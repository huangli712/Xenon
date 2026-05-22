use std::hint::black_box;

mod utils;
use utils::{generators, run_timed, SIZES_2D};

fn bench_transpose_2d(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let data = generators::sequential_2d(rows, cols);
        let median = run_timed(quick, || {
            let _result = black_box(data.transpose());
        });
        println!("transpose_2d/{rows}x{cols}: {median} ns");
    }
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);
    if should_run(filter, "transpose_2d") {
        bench_transpose_2d(quick);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_filter_runs_matching_benchmark() {
        assert!(should_run(Some("transpose"), "transpose_2d"));
        assert!(!should_run(Some("missing"), "transpose_2d"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= utils::WARMUP_ITERATIONS + 10);
    }
}
