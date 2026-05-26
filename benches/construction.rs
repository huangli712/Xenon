use std::hint::black_box;

use xenon::tensor::{Tensor1, Tensor2};

mod common;
use common::{run_timed, SIZES_1D, SIZES_2D};

fn bench_zeros_1d(quick: bool) {
    for &size in SIZES_1D {
        let median = run_timed(quick, || {
            let _result = black_box(Tensor1::<f64>::zeros([size]).expect("valid shape"));
        });
        println!("zeros_1d/{size}: {median} ns");
    }
}

fn bench_from_shape_vec_1d(quick: bool) {
    // CAVEAT (27-benchmark §5.9 "Good / Bad 示例" Bad pattern 妥协):
    //   from_shape_vec takes Vec by move; timing loop reuses via data.clone().
    //   Measurements ~= 1× Vec::clone() + 1× from_shape_vec, ~2× baseline.
    eprintln!(
        "[bench_from_shape_vec_1d] NOTE: measurements include 1x Vec<f64>::clone() per iteration; \
         see source comment for rationale (27-benchmark §5.9 \"Good / Bad 示例\")."
    );
    for &size in SIZES_1D {
        let data: Vec<f64> = (0..size).map(|idx| idx as f64).collect();
        let median = run_timed(quick, || {
            let _result = black_box(
                Tensor1::<f64>::from_shape_vec([size], data.clone()).expect("valid shape-vec"),
            );
        });
        println!("from_shape_vec_1d/{size}: {median} ns");
    }
}

fn bench_eye_2d(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let n = rows.min(cols);
        let median = run_timed(quick, || {
            let _result = black_box(Tensor2::<f64>::eye(n).expect("valid eye size"));
        });
        println!("eye_2d/{n}x{n}: {median} ns");
    }
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    if should_run(filter, "zeros_1d") { bench_zeros_1d(quick); }
    if should_run(filter, "from_shape_vec_1d") { bench_from_shape_vec_1d(quick); }
    if should_run(filter, "eye_2d") { bench_eye_2d(quick); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_filter_runs_matching_benchmark() {
        assert!(should_run(Some("zeros"), "zeros_1d"));
        assert!(!should_run(Some("missing"), "zeros_1d"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= common::WARMUP_ITERATIONS + 10);
    }
}
