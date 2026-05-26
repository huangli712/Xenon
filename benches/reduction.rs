use std::hint::black_box;

use xenon::dimension::Axis;

mod common;
use common::{generators, run_timed, SIZES_1D, SIZES_2D};

fn bench_sum_1d_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.sum());
        });
        println!("sum_1d_f64/{size}: {median} ns");
    }
}

fn bench_sum_2d_axis0(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let data = generators::sequential_2d(rows, cols);
        let median = run_timed(quick, || {
            let _result = black_box(data.sum_axis(Axis(0)).expect("axis 0 within rank"));
        });
        println!("sum_2d_axis0/{rows}x{cols}: {median} ns");
    }
}

fn bench_sum_2d_axis1(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let data = generators::sequential_2d(rows, cols);
        let median = run_timed(quick, || {
            let _result = black_box(data.sum_axis(Axis(1)).expect("axis 1 within rank"));
        });
        println!("sum_2d_axis1/{rows}x{cols}: {median} ns");
    }
}

fn bench_sum_2d_keepdims(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let data = generators::sequential_2d(rows, cols);
        let median = run_timed(quick, || {
            let _result = black_box(
                data.sum_axis_keepdims(Axis(1)).expect("axis 1 within rank"),
            );
        });
        println!("sum_2d_keepdims/{rows}x{cols}: {median} ns");
    }
}

fn bench_sum_sliced(quick: bool) {
    // 27-benchmark §5.5: sum_sliced runs at Medium scale only (single row).
    const SLICED_SIZE: usize = 65_536;
    let fixture = generators::strided_view_1d(SLICED_SIZE);
    let median = run_timed(quick, || {
        let sliced = fixture.view();
        let _result = black_box(sliced.sum());
    });
    println!("sum_sliced/{SLICED_SIZE}: {median} ns");
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    let benches: &[(&str, common::BenchFn)] = &[
        ("sum_1d_f64", bench_sum_1d_f64),
        ("sum_2d_axis0", bench_sum_2d_axis0),
        ("sum_2d_axis1", bench_sum_2d_axis1),
        ("sum_2d_keepdims", bench_sum_2d_keepdims),
        ("sum_sliced", bench_sum_sliced),
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
        assert!(should_run(Some("sum"), "sum_1d_f64"));
        assert!(!should_run(Some("missing"), "sum_1d_f64"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= common::WARMUP_ITERATIONS + 10);
    }
}
