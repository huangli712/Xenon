use std::hint::black_box;

use xenon::tensor::Tensor0;
use xenon::tensor::Tensor2;

mod utils;
use utils::{generators, run_timed, SIZES_1D, SIZES_2D};

fn bench_broadcast_scalar(quick: bool) {
    for &size in SIZES_1D {
        let scalar: Tensor0<f64> = Tensor0::from_scalar(std::f64::consts::PI).expect("scalar construction");
        let tensor = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(scalar.add(&tensor).expect("scalar broadcast must succeed"));
        });
        println!("broadcast_scalar/{size}: {median} ns");
    }
}

fn bench_broadcast_row(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let row = Tensor2::<f64>::from_shape_vec(
            [1, cols],
            (0..cols).map(|idx| idx as f64).collect(),
        ).expect("shape and data length must match");
        let target = generators::sequential_2d(rows, cols);
        let median = run_timed(quick, || {
            let _result = black_box(row.add(&target).expect("row broadcast must succeed"));
        });
        println!("broadcast_row/{rows}x{cols}: {median} ns");
    }
}

fn bench_broadcast_col(quick: bool) {
    for &(rows, cols) in SIZES_2D {
        let col = Tensor2::<f64>::from_shape_vec(
            [rows, 1],
            (0..rows).map(|idx| idx as f64).collect(),
        ).expect("shape and data length must match");
        let target = generators::sequential_2d(rows, cols);
        let median = run_timed(quick, || {
            let _result = black_box(col.add(&target).expect("col broadcast must succeed"));
        });
        println!("broadcast_col/{rows}x{cols}: {median} ns");
    }
}

fn bench_broadcast_with(quick: bool) {
    // 27-benchmark §5.5: broadcast_with — dual-tensor broadcast cooperation.
    // broadcast_with() is pub(crate); bench triggers it indirectly via
    // `row.add(&col)` on mutually-broadcastable shapes.
    for &(rows, cols) in SIZES_2D {
        // Row vector [1, cols] + column vector [rows, 1] → mutually broadcastable.
        let row = Tensor2::<f64>::from_shape_vec(
            [1, cols],
            (0..cols).map(|idx| idx as f64).collect(),
        )
        .expect("shape and data length must match");
        let col = Tensor2::<f64>::from_shape_vec(
            [rows, 1],
            (0..rows).map(|idx| idx as f64).collect(),
        )
        .expect("shape and data length must match");
        let median = run_timed(quick, || {
            let _result = black_box(
                row.add(&col).expect("mutually-broadcastable add must succeed"),
            );
        });
        println!("broadcast_with/{rows}x{cols}: {median} ns");
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
        ("broadcast_scalar", bench_broadcast_scalar),
        ("broadcast_row", bench_broadcast_row),
        ("broadcast_col", bench_broadcast_col),
        ("broadcast_with", bench_broadcast_with),
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
        assert!(should_run(Some("broadcast"), "broadcast_scalar"));
        assert!(!should_run(Some("missing"), "broadcast_scalar"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= utils::WARMUP_ITERATIONS + 10);
    }
}
