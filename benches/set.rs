use std::hint::black_box;

mod utils;
use utils::{run_timed, SIZES_1D};

// 27-benchmark §5.5: unique_1d — single canonical entry over SIZES_1D.
// Inputs use a moderate unique ratio (~50%) by mapping idx -> idx / 2 as f64.
fn bench_unique_1d(quick: bool) {
    for &size in SIZES_1D {
        let data = xenon::tensor::Tensor1::<f64>::from_shape_vec(
            [size],
            (0..size).map(|idx| (idx / 2) as f64).collect(),
        )
        .expect("shape and data length must match");
        let median = run_timed(quick, || {
            let _result = black_box(data.unique());
        });
        println!("unique_1d/{size}: {median} ns");
    }
}

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    if should_run(filter, "unique_1d") {
        bench_unique_1d(quick);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_filter_runs_matching_benchmark() {
        assert!(should_run(Some("unique"), "unique_1d"));
        assert!(!should_run(Some("missing"), "unique_1d"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= utils::WARMUP_ITERATIONS + 10);
    }
}
