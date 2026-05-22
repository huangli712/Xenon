use xenon::error::XenonError;
use xenon::tensor::{Tensor0, Tensor1, Tensor2};

// ── §7 T4 / §8.3 — test_clip_empty ──

#[test]
fn test_clip_empty() {
    let tensor =
        Tensor1::<f64>::from_shape_vec([0], Vec::new()).expect("from_shape_vec matching shape");
    let clipped = tensor.clip(0.0, 1.0).expect("valid clip bounds");
    assert_eq!(clipped.len(), 0);
}

// ── §7 T4 / §8.3 — test_fill_zero_dim ──

#[test]
fn test_fill_zero_dim() {
    let mut tensor = Tensor0::from_scalar(1_i32).expect("from_scalar valid");
    tensor.fill(9);
    assert_eq!(*tensor.get(&[]).expect("valid index"), 9);
}

// ── §7 T4 / §8.3 — test_clip_non_contiguous ──
//
// F-order construction of `[2, 3]` from `[1..=6]` yields:
//   col 0 = [1, 2]; col 1 = [3, 4]; col 2 = [5, 6]
//   logical matrix
//     [1 3 5]
//     [2 4 6]
// After `transpose()` → 3×2:
//     [1 2]
//     [3 4]
//     [5 5]
// clip(2, 5):
//     [2 2]
//     [3 4]
//     [5 5]

#[test]
fn test_clip_non_contiguous() {
    let tensor = Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
        .expect("from_shape_vec matching shape");
    let clipped = tensor.transpose().clip(2, 5).expect("valid clip bounds");
    assert_eq!(clipped.shape(), &[3, 2]);
    assert_eq!(*clipped.get(&[0, 0]).expect("valid index"), 2);
    assert_eq!(*clipped.get(&[0, 1]).expect("valid index"), 2);
    assert_eq!(*clipped.get(&[1, 0]).expect("valid index"), 3);
    assert_eq!(*clipped.get(&[1, 1]).expect("valid index"), 4);
    assert_eq!(*clipped.get(&[2, 0]).expect("valid index"), 5);
    assert_eq!(*clipped.get(&[2, 1]).expect("valid index"), 5);
}

// ── §8.5 — test_to_contiguous_integration (utility ↔ tensor ↔ iter ↔ layout) ──
//
// 2×2 F-order construction is canonical itself; we transpose to exercise
// the non-contiguous repack path. After `transpose()` the 2×2 logical matrix is:
//   [1 2]      (col 0 of original = [1, 2] → row 0 transposed)
//   [3 4]      (col 1 of original = [3, 4] → row 1 transposed)
// `to_contiguous()` produces a canonical F-order owned with the SAME
// logical values at the SAME logical positions.

#[test]
fn test_to_contiguous_integration() {
    let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("from_shape_vec matching shape");
    let contiguous = tensor.transpose().to_contiguous();
    assert!(contiguous.is_f_contiguous());
    assert_eq!(contiguous.shape(), &[2, 2]);
    // Element-wise (order-independent):
    assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
    assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 2);
    assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 3);
    assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
}

// ── §8.5 — test_into_contiguous_integration (ownership + canonical F-order) ──

#[test]
fn test_into_contiguous_integration() {
    let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("from_shape_vec matching shape");
    let contiguous = tensor.into_contiguous();
    assert!(contiguous.is_f_contiguous());
    // Element-wise (order-independent):
    assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
    assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
    assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
    assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
}

// ── §8.3 — test_try_fill_read_only_integration (dispatch boundary) ──

#[test]
fn test_try_fill_read_only_integration() {
    let tensor =
        Tensor1::from_shape_vec([3], vec![1_i32, 2, 3]).expect("from_shape_vec matching shape");
    let mut view = tensor.view();
    let err = view.try_fill(9).expect_err("view is read-only");
    assert!(matches!(err, XenonError::InvalidStorageMode { .. }));
}

// ── §7 T4 — test_clip_single_element ──

#[test]
fn test_clip_single_element() {
    let tensor =
        Tensor1::<i64>::from_shape_vec([1], vec![10]).expect("from_shape_vec matching shape");
    let clipped = tensor.clip(0, 5).expect("valid clip bounds");
    assert_eq!(clipped.shape(), &[1]);
    assert_eq!(*clipped.get(&[0]).expect("valid index"), 5);
}

// ── §8.3 — large-array boundary (no panic, all elements in [min, max]) ──

#[test]
fn test_clip_large_array() {
    let data: Vec<i64> = (0..10_000).map(|i| i % 100).collect();
    let tensor =
        Tensor1::<i64>::from_shape_vec([10_000], data).expect("from_shape_vec matching shape");
    let clipped = tensor.clip(20, 80).expect("valid clip bounds");
    assert_eq!(clipped.shape(), &[10_000]);
    assert!(clipped.iter().all(|&x| (20..=80).contains(&x)));
}

// ── Additional integration tests for clip / fill / to_contiguous / into_contiguous / try_fill ──

#[test]
fn test_try_fill_writable_success() {
    let mut tensor = Tensor1::<i32>::zeros([4]).expect("zeros valid shape");
    tensor.try_fill(7).expect("try_fill on owned should succeed");
    let values: Vec<i32> = tensor.iter().copied().collect();
    assert_eq!(values, vec![7, 7, 7, 7]);
}

#[test]
fn test_clip() {
    let tensor = Tensor1::from_shape_vec([5], vec![-5_i32, 0, 3, 8, 12])
        .expect("valid construction");
    let clipped = tensor.clip(0, 10).expect("valid clip bounds");
    let values: Vec<i32> = clipped.iter().copied().collect();
    assert_eq!(values, vec![0, 0, 3, 8, 10]);
}

#[test]
fn test_to_contiguous() {
    let tensor = xenon::tensor::Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("valid construction");
    // Transposed view is non-contiguous.
    let view = tensor.transpose();
    assert!(!view.is_f_contiguous());
    let contiguous = view.to_contiguous();
    assert!(contiguous.is_f_contiguous());
    assert_eq!(contiguous.shape(), &[2, 2]);
    // Values preserved (logical F-order iter → canonical F-order owned).
    assert_eq!(*contiguous.try_at((0, 0)).expect("valid index"), 1);
    assert_eq!(*contiguous.try_at((0, 1)).expect("valid index"), 2);
    assert_eq!(*contiguous.try_at((1, 0)).expect("valid index"), 3);
    assert_eq!(*contiguous.try_at((1, 1)).expect("valid index"), 4);
}

#[test]
fn test_fill_inplace() {
    let mut tensor = Tensor1::<f64>::zeros([5]).expect("zeros valid shape");
    tensor.fill(1.23);
    let values: Vec<f64> = tensor.iter().copied().collect();
    assert_eq!(values, vec![1.23; 5]);
}

#[test]
fn test_try_fill_rejects_readonly_or_broadcast() {
    let tensor = Tensor1::<i32>::from_shape_vec([3], vec![1, 2, 3])
        .expect("valid construction");
    let mut view = tensor.view();
    let err = view.try_fill(0).expect_err("view is read-only");
    assert!(matches!(err, XenonError::InvalidStorageMode { .. }));
}

// test_clip_non_contiguous already exists at line 39 — not duplicated here.

#[test]
fn test_clip_invalid_parameters() {
    let tensor = Tensor1::<i32>::from_shape_vec([3], vec![1, 2, 3])
        .expect("valid construction");
    let err = tensor.clip(10, 5).expect_err("min > max should be rejected");
    assert!(matches!(err, XenonError::InvalidArgument { .. }));
}

#[test]
fn test_into_contiguous_reuses_owned_data() {
    let tensor = xenon::tensor::Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("valid construction");
    let contiguous = tensor.into_contiguous();
    assert!(contiguous.is_f_contiguous());
    assert_eq!(*contiguous.try_at((0, 0)).expect("valid index"), 1);
    assert_eq!(*contiguous.try_at((1, 0)).expect("valid index"), 2);
    assert_eq!(*contiguous.try_at((0, 1)).expect("valid index"), 3);
    assert_eq!(*contiguous.try_at((1, 1)).expect("valid index"), 4);
}

#[test]
fn test_into_contiguous_materializes_view() {
    // The view is non-contiguous; into_contiguous() copies raw physical storage
    // (StorageIntoOwned for ViewRepr) and wraps with canonical F-order strides.
    let tensor = xenon::tensor::Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
        .expect("valid construction");
    let view = tensor.transpose(); // shape [3, 2], non-contiguous.
    assert!(!view.is_f_contiguous());
    let contiguous = view.into_contiguous();
    assert!(contiguous.is_f_contiguous());
    assert_eq!(contiguous.shape(), &[3, 2]);
    // Physical storage is [1, 2, 3, 4, 5, 6]; canonical F-order strides for [3,2] are [1, 3].
    // Logical F-order matrix: [[1, 4], [2, 5], [3, 6]]
    assert_eq!(*contiguous.try_at((0, 0)).expect("valid index"), 1);
    assert_eq!(*contiguous.try_at((0, 1)).expect("valid index"), 4);
    assert_eq!(*contiguous.try_at((1, 0)).expect("valid index"), 2);
    assert_eq!(*contiguous.try_at((1, 1)).expect("valid index"), 5);
    assert_eq!(*contiguous.try_at((2, 0)).expect("valid index"), 3);
    assert_eq!(*contiguous.try_at((2, 1)).expect("valid index"), 6);
}
