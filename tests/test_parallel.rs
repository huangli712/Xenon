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