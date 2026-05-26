use std::hint::black_box;

mod common;
use common::{generators, run_timed, SIZES_1D};

// --- f64 contiguous benches ---

fn bench_elem_add_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let rhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.add(&rhs).expect("same-shape add must succeed"));
        });
        println!("elem_add_f64/{size}: {median} ns");
    }
}

fn bench_elem_sub_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let rhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.sub(&rhs).expect("same-shape sub must succeed"));
        });
        println!("elem_sub_f64/{size}: {median} ns");
    }
}

fn bench_elem_mul_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let rhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.mul(&rhs).expect("same-shape mul must succeed"));
        });
        println!("elem_mul_f64/{size}: {median} ns");
    }
}

fn bench_elem_div_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let rhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.div(&rhs).expect("same-shape div must succeed"));
        });
        println!("elem_div_f64/{size}: {median} ns");
    }
}

fn bench_elem_abs_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.abs());
        });
        println!("elem_abs_f64/{size}: {median} ns");
    }
}

fn bench_elem_sin_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.sin());
        });
        println!("elem_sin_f64/{size}: {median} ns");
    }
}

fn bench_elem_exp_f64(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.exp());
        });
        println!("elem_exp_f64/{size}: {median} ns");
    }
}

// --- f32 benches ---

fn bench_elem_add_f32(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d_f32(size);
        let rhs = generators::sequential_1d_f32(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.add(&rhs).expect("same-shape add must succeed"));
        });
        println!("elem_add_f32/{size}: {median} ns");
    }
}

fn bench_elem_sin_f32(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::sequential_1d_f32(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.sin());
        });
        println!("elem_sin_f32/{size}: {median} ns");
    }
}

// --- Complex<f64> bench ---

fn bench_elem_add_complex(quick: bool) {
    for &size in SIZES_1D {
        let lhs = generators::complex_1d(size);
        let rhs = generators::complex_1d(size);
        let median = run_timed(quick, || {
            let _result = black_box(lhs.add(&rhs).expect("same-shape add must succeed"));
        });
        println!("elem_add_complex/{size}: {median} ns");
    }
}

// --- non-contiguous (sliced) bench ---

fn bench_elem_add_sliced(quick: bool) {
    // §5.5 schedules this entry at Medium scale (65,536).
    const SLICED_SIZE: usize = 65_536;
    let fixture = generators::strided_view_1d(SLICED_SIZE);
    let rhs = generators::sequential_1d(SLICED_SIZE);
    let median = run_timed(quick, || {
        let lhs_sliced = fixture.view();
        let _result = black_box(lhs_sliced.add(&rhs).expect("same-shape add must succeed"));
    });
    println!("elem_add_sliced/{SLICED_SIZE}: {median} ns");
}

// --- main dispatch ---

fn should_run(filter: Option<&str>, name: &str) -> bool {
    filter.is_none_or(|needle| name.contains(needle))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let quick = args.iter().any(|arg| arg == "--quick");
    let filter = args.iter().find(|arg| !arg.starts_with("--")).map(String::as_str);

    let benches: &[(&str, common::BenchFn)] = &[
        ("elem_add_f64", bench_elem_add_f64),
        ("elem_sub_f64", bench_elem_sub_f64),
        ("elem_mul_f64", bench_elem_mul_f64),
        ("elem_div_f64", bench_elem_div_f64),
        ("elem_abs_f64", bench_elem_abs_f64),
        ("elem_sin_f64", bench_elem_sin_f64),
        ("elem_exp_f64", bench_elem_exp_f64),
        ("elem_add_f32", bench_elem_add_f32),
        ("elem_sin_f32", bench_elem_sin_f32),
        ("elem_add_complex", bench_elem_add_complex),
        ("elem_add_sliced", bench_elem_add_sliced),
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
        assert!(should_run(Some("elem"), "elem_add_f64"));
        assert!(!should_run(Some("missing"), "elem_add_f64"));
    }

    #[test]
    fn test_quick_mode_uses_single_round() {
        let mut calls = 0usize;
        let median = run_timed(true, || calls += 1);
        assert!(median <= u128::MAX);
        assert!(calls >= common::WARMUP_ITERATIONS + 10);
    }
}
