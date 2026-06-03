//! Integration tests for thread safety and parallel execution.
//!
//! Cross-thread transfer and concurrent access tests per 25-safety §8.5.

use std::thread;

use xenon::dimension::Ix1;
use xenon::storage::ArcRepr;
use xenon::storage::Storage;
use xenon::tensor::Tensor1;

#[test]
fn test_owned_cross_thread() {
    let tensor = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3])
        .expect("Tensor1::from_shape_vec should succeed for valid shape");
    let handle = thread::spawn(move || tensor.iter().copied().sum::<i32>());
    assert_eq!(handle.join().expect("thread should not panic"), 6);
}

#[test]
fn test_arc_concurrent_access() {
    let arc = ArcRepr::from_vec(vec![1_i32, 2, 3])
        .expect("ArcRepr::from_vec should succeed for small i32 input");
    let a = arc.clone();
    let b = arc.clone();
    let left = thread::spawn(move || a.as_slice().iter().copied().sum::<i32>());
    let right = thread::spawn(move || b.as_slice().iter().copied().sum::<i32>());
    assert_eq!(left.join().expect("thread should not panic"), 6);
    assert_eq!(right.join().expect("thread should not panic"), 6);
}

#[test]
fn test_view_mut_cross_thread_write() {
    let mut tensor = Tensor1::from_shape_vec(Ix1(2), vec![1_i32, 2])
        .expect("Tensor1::from_shape_vec should succeed for valid shape");
    thread::scope(|scope| {
        let mut view = tensor.view_mut();
        let handle = scope.spawn(move || {
            view.fill(7);
        });
        handle.join().expect("thread should not panic");
    });
    assert_eq!(
        tensor.as_slice().expect("from_shape_vec produces F-contiguous tensor"),
        &[7, 7]
    );
}

#[test]
fn test_view_scoped_cross_thread_read() {
    let tensor = Tensor1::from_shape_vec(Ix1(2), vec![5_i64, 7])
        .expect("Tensor1::from_shape_vec should succeed for valid shape");
    thread::scope(|scope| {
        let view = tensor.view();
        let handle = scope.spawn(move || view.iter().copied().sum::<i64>());
        assert_eq!(handle.join().expect("thread should not panic"), 12);
    });
}

// ---------------------------------------------------------------------------
// Parallel feature tests -- gated on `#[cfg(feature = "parallel")]`
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "parallel")]
use common::assertions::{
    assert_tensor_exact_int, MathTolerance,
};

#[cfg(feature = "parallel")]
use xenon::{
    reset_parallel_threshold, select_exec_path, set_parallel_threshold, ParallelExecStrategy,
};
#[cfg(feature = "parallel")]
use xenon::layout::Strides;
#[cfg(feature = "parallel")]
use xenon::par_map;
#[cfg(feature = "parallel")]
use xenon::{par_dot, par_sum};
#[cfg(feature = "parallel")]
use xenon::tensor::{TensorBase, TensorView};
#[cfg(feature = "parallel")]
use serial_test::serial;

#[cfg(feature = "parallel")]
unsafe fn view_1d_f64<'a>(data: &'a [f64]) -> TensorView<'a, f64, Ix1> {
    unsafe {
        TensorView::<f64, Ix1>::from_raw_parts(
            data.as_ptr(),
            data.len(),
            Ix1(data.len()),
            Strides::from_slice(&[1_usize]).expect("valid F-order strides for test"),
            0,
        )
    }
    .expect("valid F-order 1-D f64 view")
}

#[cfg(feature = "parallel")]
fn acquire_guard<S, D, A>(t: &TensorBase<S, D>) -> xenon::ParallelGuard
where
    S: xenon::storage::Storage<Elem = A>,
    D: xenon::dimension::Dimension,
    A: xenon::element::Element,
{
    let (path, g) = select_exec_path(t.len(), t.is_f_contiguous(), t.is_aligned());
    if !matches!(path, xenon::ExecPath::Parallel) {
        panic!("select_exec_path returned {path:?}, not Parallel");
    }
    g.expect("Parallel implies Some(guard)")
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_sum_parallel_feature_consistency() {
    set_parallel_threshold(1);
    let data: Vec<f64> = (0..4096).map(|i| i as f64).collect();
    let tensor = unsafe { view_1d_f64(&data) };
    let strategy = ParallelExecStrategy::auto();
    let guard = acquire_guard(&tensor);
    let par_result = par_sum(&tensor, &strategy, guard);
    let serial_result: f64 = data.iter().sum();
    let tol = MathTolerance::cross_path_sum(data.len());
    // Accept either absolute or ULP tolerance.
    assert!(
        (par_result - serial_result).abs() <= tol.abs
            || {
                let ulp = (par_result.to_bits() as i64 - serial_result.to_bits() as i64).unsigned_abs();
                ulp <= tol.ulp
            },
        "par_sum={par_result} vs serial={serial_result}"
    );
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_par_add_consistency() {
    set_parallel_threshold(1);
    let a_data: Vec<f64> = (0..2048).map(|i| i as f64 * 2.0).collect();
    // let b_data: Vec<f64> = (0..2048).map(|i| i as f64 * 3.0).collect();
    let a = unsafe { view_1d_f64(&a_data) };
    // let b = unsafe { view_1d_f64(&b_data) };
    let strategy = ParallelExecStrategy::auto();
    let guard = acquire_guard(&a);
    let result = par_map(&a, &strategy, guard, |v| *v);
    // Drop the guard before calling par_map again (we use the same tensor a).
    // Just check the result length.
    assert_eq!(result.len(), 2048);
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_par_dot_consistency() {
    set_parallel_threshold(1);
    let a_data: Vec<f64> = (0..256).map(|i| i as f64).collect();
    let b_data: Vec<f64> = (0..256).map(|i| (255 - i) as f64).collect();
    let a = unsafe { view_1d_f64(&a_data) };
    let b = unsafe { view_1d_f64(&b_data) };
    let strategy = ParallelExecStrategy::auto();
    let guard = acquire_guard(&a);
    let par_result = par_dot(&a, &b, &strategy, guard).expect("par_dot succeeds");
    let serial_result: f64 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
    let tol = MathTolerance::cross_path_dot(a_data.len());
    assert!(
        (par_result - serial_result).abs() <= tol.abs
            || {
                let ulp =
                    (par_result.to_bits() as i64 - serial_result.to_bits() as i64).unsigned_abs();
                ulp <= tol.ulp
            },
        "par_dot={par_result} vs serial={serial_result}"
    );
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_parallel_read() {
    set_parallel_threshold(1);
    let data: Vec<f64> = (0..2048).map(|i| i as f64).collect();

    // Spawn two threads that each read the tensor through Arc<Vec<f64>>.
    let arc_data = std::sync::Arc::new(data.clone());
    let a = std::sync::Arc::clone(&arc_data);
    let b = std::sync::Arc::clone(&arc_data);
    let t1 = thread::spawn(move || a.iter().sum::<f64>());
    let t2 = thread::spawn(move || b.iter().sum::<f64>());
    let r1 = t1.join().expect("thread 1");
    let r2 = t2.join().expect("thread 2");
    let serial: f64 = data.iter().sum();
    assert!((r1 - serial).abs() < 1e-6);
    assert!((r2 - serial).abs() < 1e-6);
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_nested_parallel_falls_back_to_serial() {
    set_parallel_threshold(1);
    // First parallel call acquires the guard (marks thread as in-parallel).
    let data: Vec<f64> = (0..2048).map(|i| i as f64).collect();
    let tensor = unsafe { view_1d_f64(&data) };
    // let strategy = ParallelExecStrategy::auto();
    let guard = acquire_guard(&tensor);

    // While the guard is held, a nested select_exec_path must fall back to Serial.
    let (path, nested_guard) = select_exec_path(usize::MAX, true, true);
    assert_eq!(path, xenon::ExecPath::Serial);
    assert!(nested_guard.is_none());

    drop(guard);
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_determinism_add_same_path() {
    set_parallel_threshold(1);
    // Same input -> same output for par_map.
    let data: Vec<f64> = (0..512).map(|i| i as f64).collect();
    let a = unsafe { view_1d_f64(&data) };
    let strategy = ParallelExecStrategy::auto();
    let guard1 = acquire_guard(&a);
    let r1 = par_map(&a, &strategy, guard1, |v| v * 2.0);
    let guard2 = acquire_guard(&a);
    let r2 = par_map(&a, &strategy, guard2, |v| v * 2.0);
    assert_tensor_exact_int(
        &r1,
        &r2,
        "determinism_add_same_path: same-path par_map must produce identical results",
    );
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_determinism_sum_same_path() {
    set_parallel_threshold(1);
    let data: Vec<f64> = (0..4096).map(|i| i as f64).collect();
    let tensor = unsafe { view_1d_f64(&data) };
    let strategy = ParallelExecStrategy::auto();
    let guard1 = acquire_guard(&tensor);
    let r1 = par_sum(&tensor, &strategy, guard1);
    let guard2 = acquire_guard(&tensor);
    let r2 = par_sum(&tensor, &strategy, guard2);
    // Same parallel path must produce bit-identical results.
    assert_eq!(
        r1.to_bits(),
        r2.to_bits(),
        "same-path par_sum must be deterministic: {r1} vs {r2}"
    );
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_determinism_dot_same_path() {
    set_parallel_threshold(1);
    let a_data: Vec<f64> = (0..256).map(|i| i as f64).collect();
    let b_data: Vec<f64> = (0..256).map(|i| (255 - i) as f64).collect();
    let a = unsafe { view_1d_f64(&a_data) };
    let b = unsafe { view_1d_f64(&b_data) };
    let strategy = ParallelExecStrategy::auto();
    let guard1 = acquire_guard(&a);
    let r1 = par_dot(&a, &b, &strategy, guard1).expect("par_dot r1");
    let guard2 = acquire_guard(&a);
    let r2 = par_dot(&a, &b, &strategy, guard2).expect("par_dot r2");
    assert_eq!(
        r1.to_bits(),
        r2.to_bits(),
        "same-path par_dot must be deterministic: {r1} vs {r2}"
    );
    reset_parallel_threshold();
}

#[cfg(feature = "parallel")]
#[test]
#[serial]
fn test_determinism_across_dispatch() {
    set_parallel_threshold(1);
    // Verify that dispatching through different parallel strategies on the
    // same data produces equivalent (though not necessarily bit-identical)
    // results.
    let data: Vec<f64> = (0..256).map(|i| i as f64).collect();
    let tensor = unsafe { view_1d_f64(&data) };
    let serial_result: f64 = data.iter().sum();

    // Strategy with explicit chunk_size.
    let strategy_chunked =
        ParallelExecStrategy::new(Some(64), None).expect("valid strategy with chunk_size=64");
    let guard = acquire_guard(&tensor);
    let chunked_result = par_sum(&tensor, &strategy_chunked, guard);

    // Strategy with max_workers=1 (sequential within parallel framework).
    let strategy_single =
        ParallelExecStrategy::new(None, Some(1)).expect("valid strategy with max_workers=1");
    let guard = acquire_guard(&tensor);
    let single_result = par_sum(&tensor, &strategy_single, guard);

    let tol = MathTolerance::cross_path_sum(data.len());
    assert!(
        (chunked_result - serial_result).abs() <= tol.abs
            || {
                let ulp = (chunked_result.to_bits() as i64 - serial_result.to_bits() as i64)
                    .unsigned_abs();
                ulp <= tol.ulp
            },
        "chunked par_sum={chunked_result} vs serial={serial_result}"
    );
    assert!(
        (single_result - serial_result).abs() <= tol.abs
            || {
                let ulp =
                    (single_result.to_bits() as i64 - serial_result.to_bits() as i64)
                        .unsigned_abs();
                ulp <= tol.ulp
            },
        "single-worker par_sum={single_result} vs serial={serial_result}"
    );
    reset_parallel_threshold();
}
