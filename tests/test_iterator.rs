// tests/test_iterator.rs
//
// Integration coverage for tensor iterators per 10-iterator.md.
// Tests exercise the public entry points: iter(), iter_mut(), axis_iter(),
// indexed_iter(), and edge cases (empty tensors, rank-0 runtime errors).

use xenon::dimension::{Axis, Ix1, IxDyn};
use xenon::error::XenonError;
use xenon::tensor::{Tensor, Tensor1, Tensor2};

#[test]
fn test_iter_elements() {
    let t = Tensor1::from_shape_vec(Ix1(4), vec![10_i32, 20, 30, 40])
        .expect("valid construction");
    let values: Vec<i32> = t.iter().copied().collect();
    assert_eq!(values, vec![10, 20, 30, 40]);
    assert_eq!(t.iter().len(), 4);
}

#[test]
fn test_iter_mut_elements() {
    let mut t = Tensor1::from_shape_vec(Ix1(4), vec![1_i32, 2, 3, 4])
        .expect("valid construction");
    for v in t.iter_mut() {
        *v *= 10;
    }
    let values: Vec<i32> = t.iter().copied().collect();
    assert_eq!(values, vec![10, 20, 30, 40]);
}

#[test]
fn test_axis_iter() {
    // 2×3 F-order tensor: logical matrix
    //   col 0 = [1, 2], col 1 = [3, 4], col 2 = [5, 6]
    //   [[1, 3, 5],
    //    [2, 4, 6]]
    let t = Tensor2::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
        .expect("valid construction");

    // Iterate over axis 0 (rows): yields 2 sub-views of shape [3].
    let mut axis0 = t.axis_iter(Axis(0)).expect("Axis(0) valid for 2-D");
    assert_eq!(axis0.len(), 2);

    let row0: Vec<i32> = axis0.next().expect("first row").iter().copied().collect();
    assert_eq!(row0, vec![1, 3, 5]);

    let row1: Vec<i32> = axis0.next().expect("second row").iter().copied().collect();
    assert_eq!(row1, vec![2, 4, 6]);
    assert!(axis0.next().is_none());

    // Iterate over axis 1 (columns): yields 3 sub-views of shape [2].
    let mut axis1 = t.axis_iter(Axis(1)).expect("Axis(1) valid for 2-D");
    assert_eq!(axis1.len(), 3);

    let col0: Vec<i32> = axis1.next().expect("first col").iter().copied().collect();
    assert_eq!(col0, vec![1, 2]);
}

#[test]
fn test_axis_iter_ix0_runtime_error() {
    // Use IxDyn with rank 0 for the axis iterator error test,
    // because Ix0 does not implement RemoveAxis (required by axis_iter).
    let scalar = Tensor::<f64, IxDyn>::from_shape_vec(
        IxDyn::from_slice(&[]),
        vec![1.23],
    )
    .expect("valid construction");
    // axis_iter is not supported on rank-0; use match to avoid Debug bound on AxisIter.
    match scalar.axis_iter(Axis(0)) {
        Err(e) => assert!(matches!(e, XenonError::InvalidAxis { axis: 0, ndim: 0, .. })),
        Ok(_) => panic!("expected InvalidAxis error for rank-0 axis_iter"),
    }
}

#[test]
fn test_indexed_iter() {
    // 2×2 F-order: [1, 2, 3, 4] → logical [[1, 3], [2, 4]]
    let t = Tensor2::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("valid construction");
    let pairs: Vec<(xenon::dimension::Ix2, i32)> =
        t.indexed_iter().map(|(idx, v)| (idx, *v)).collect();
    assert_eq!(pairs.len(), 4);
    // F-order: index order is (0,0), (1,0), (0,1), (1,1)
    assert_eq!(pairs[0], (xenon::dimension::Ix2(0, 0), 1));
    assert_eq!(pairs[1], (xenon::dimension::Ix2(1, 0), 2));
    assert_eq!(pairs[2], (xenon::dimension::Ix2(0, 1), 3));
    assert_eq!(pairs[3], (xenon::dimension::Ix2(1, 1), 4));
}

#[test]
fn test_empty_tensor_iter_count() {
    let t = Tensor1::<i32>::from_shape_vec(Ix1(0), vec![])
        .expect("valid construction");
    assert_eq!(t.iter().len(), 0);
    assert_eq!(t.iter().count(), 0);
    assert!(t.iter().next().is_none());
}
