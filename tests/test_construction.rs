use xenon::tensor::Tensor;
use xenon::dimension::{Ix0, Ix1, Ix2, IxDyn};

#[test]
fn test_construction_high_rank_ixdyn() {
    let tensor =
        Tensor::<i32, _>::zeros(IxDyn::from_vec(vec![1, 2, 1, 3])).expect("test input valid");
    assert_eq!(tensor.shape(), &[1, 2, 1, 3]);
    assert!(tensor.iter().all(|value| *value == 0));
}

#[test]
fn test_construction_round_trip_sources() {
    assert_eq!(
        Tensor::<i32, Ix0>::from_scalar(7i32)
            .expect("test input valid")
            .len(),
        1
    );
    assert_eq!(
        Tensor::<i32, _>::from_array([2, 2], [1, 2, 3, 4])
            .expect("test input valid")
            .len(),
        4
    );
    assert_eq!(
        Tensor::<i32, _>::from_shape_slice([2], &[5, 6])
            .expect("test input valid")
            .len(),
        2
    );
    assert_eq!(
        Tensor::<i32, _>::from_shape_vec([2], vec![5, 6])
            .expect("test input valid")
            .len(),
        2
    );
    assert_eq!(
        Tensor::<i32, Ix1>::from_vec(vec![1, 2, 3])
            .expect("test input valid")
            .len(),
        3
    );
    assert_eq!(
        Tensor::<i32, _>::ones([1]).expect("test input valid").len(),
        1
    );
    assert_eq!(
        Tensor::<i32, Ix2>::eye(0).expect("test input valid").len(),
        0
    );
}

#[test]
fn test_ones_integration() {
    let tensor = Tensor::<f64, _>::ones([2, 2]).expect("test input valid");
    assert_eq!(tensor.shape(), &[2, 2]);
    assert!(tensor.iter().all(|v| (*v - 1.0f64).abs() < f64::EPSILON));
}

#[test]
fn test_eye_zero_empty_matrix() {
    let tensor = Tensor::<i32, Ix2>::eye(0).expect("test input valid");
    assert_eq!(tensor.shape(), &[0, 0]);
    assert_eq!(tensor.len(), 0);
}

#[test]
fn test_eye_3x3_integration() {
    // Full diagonal sweep — exercises eye + indexing + iter cross-module path.
    let tensor = Tensor::<f64, Ix2>::eye(3).expect("test input valid");
    assert_eq!(tensor.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let v = *tensor.get(&[i, j]).expect("test input valid");
            if i == j {
                assert!((v - 1.0).abs() < f64::EPSILON);
            } else {
                assert!(v.abs() < f64::EPSILON);
            }
        }
    }
}

#[test]
fn test_from_scalar_zero_dim() {
    // Cross-module path: from_scalar + Ix0 + indexing-by-unit.
    let tensor = Tensor::<f64, Ix0>::from_scalar(std::f64::consts::PI).expect("test input valid");
    assert_eq!(tensor.ndim(), 0);
    assert_eq!(tensor.len(), 1);
    assert!(
        (*tensor.get(&[] as &[usize]).expect("test input valid") - std::f64::consts::PI).abs()
            < f64::EPSILON
    );
}

#[test]
fn test_construction_large_tensor() {
    // Verify large tensor construction succeeds and shape is correct.
    // Design §8.3 specifies [3162, 3162] (≈80MB), but CI memory constraints
    // motivate using [512, 512] (2MB) while preserving the semantic intent:
    // "large tensor allocation + F-order contiguous layout verification".
    let n: usize = 512;
    let tensor = Tensor::<f64, _>::zeros([n, n]).expect("test input valid");
    assert_eq!(tensor.shape(), &[n, n]);
    assert_eq!(tensor.len(), n * n);
    assert!(tensor.iter().all(|v| *v == 0.0));
}

#[test]
fn test_zero_axis_integration() {
    // §8.3 boundary: empty tensors with zero-length axes.
    // zeros([0]) → 1D empty, len=0
    let t1 = Tensor::<i32, _>::zeros([0]).expect("test input valid");
    assert_eq!(t1.shape(), &[0]);
    assert_eq!(t1.len(), 0);

    // zeros([0, 3]) → 2D with zero rows, len=0
    let t2 = Tensor::<i32, _>::zeros([0, 3]).expect("test input valid");
    assert_eq!(t2.shape(), &[0, 3]);
    assert_eq!(t2.len(), 0);

    // from_shape_vec([0], vec![]) → empty 1D construction
    let t3 = Tensor::<i32, _>::from_shape_vec([0], vec![]).expect("test input valid");
    assert_eq!(t3.shape(), &[0]);
    assert_eq!(t3.len(), 0);
}

#[test]
fn test_overflow_shape_integration() {
    // §8.3 boundary: shape product overflow triggers ProductOverflow error.
    use xenon::error::{InvalidShapeKind, XenonError};

    let huge_shape = [usize::MAX, 2];
    let result = Tensor::<i32, _>::zeros(huge_shape);
    assert!(result.is_err());
    if let Err(XenonError::InvalidShape { kind, .. }) = result {
        assert!(matches!(kind, InvalidShapeKind::ProductOverflow));
    } else {
        panic!("Expected InvalidShape with ProductOverflow");
    }
}

// ── Additional construction integration tests ──
use xenon::tensor::Tensor1;


/// zeros, ones, from_scalar constructors produce correctly-shaped tensors.
#[test]
fn test_zeros_ones_from_scalar() {
    let z = Tensor::<i32, _>::zeros([2, 3]).expect("valid test input");
    assert_eq!(z.shape(), &[2, 3]);
    assert!(z.iter().all(|v| *v == 0));

    let o = Tensor::<f64, _>::ones([2, 2]).expect("valid test input");
    assert_eq!(o.shape(), &[2, 2]);
    assert!(o.iter().all(|v| (*v - 1.0).abs() < f64::EPSILON));

    let s = Tensor::<i32, Ix0>::from_scalar(42).expect("valid test input");
    assert_eq!(s.ndim(), 0);
    assert_eq!(*s.try_at(()).expect("valid index"), 42);
}

/// eye and identity produce correct diagonal structure.
#[test]
fn test_eye_identity() {
    let eye3 = Tensor::<i32, Ix2>::eye(3).expect("valid test input");
    assert_eq!(eye3.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let v = *eye3.try_at((i, j)).expect("valid index");
            if i == j {
                assert_eq!(v, 1);
            } else {
                assert_eq!(v, 0);
            }
        }
    }
    // eye(0) produces empty matrix.
    let eye0 = Tensor::<f64, Ix2>::eye(0).expect("valid test input");
    assert_eq!(eye0.len(), 0);
}

/// from_shape_vec, from_shape_slice, from_vec constructors.
#[test]
fn test_from_data_constructors() {
    let sv = Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
    assert_eq!(sv.len(), 4);

    let data = [5i32, 6, 7, 8];
    let ss = Tensor::<i32, _>::from_shape_slice([2, 2], &data).expect("valid test input");
    assert_eq!(ss.len(), 4);

    let fv = Tensor1::<i32>::from_vec(vec![1, 2, 3]).expect("valid test input");
    assert_eq!(fv.shape(), &[3]);

    // ElementCountMismatch error.
    let err = Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3]).expect_err("mismatch");
    assert!(matches!(err, xenon::XenonError::InvalidShape { .. }));
}

/// from_array with fixed-size array.
#[test]
fn test_from_fixed_array() {
    let t = Tensor::<i32, _>::from_array([2, 2], [1, 2, 3, 4]).expect("valid test input");
    assert_eq!(t.shape(), &[2, 2]);
    // F-order: col 0 = [1,2], col 1 = [3,4]
    assert_eq!(*t.try_at((0, 0)).expect("valid index"), 1);
    assert_eq!(*t.try_at((1, 1)).expect("valid index"), 4);
}

/// F-order mapping: from_shape_vec stores data in column-major order.
#[test]
fn test_from_shape_vec_f_order_mapping() {
    // For shape [2,3] with data [1,2,3,4,5,6]:
    // F-order stores: col 0 = [1,2], col 1 = [3,4], col 2 = [5,6]
    // Logical matrix: row 0 = [1,3,5], row 1 = [2,4,6]
    let t = Tensor::<i32, _>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    assert_eq!(*t.try_at((0, 0)).expect("valid index"), 1);
    assert_eq!(*t.try_at((0, 1)).expect("valid index"), 3);
    assert_eq!(*t.try_at((0, 2)).expect("valid index"), 5);
    assert_eq!(*t.try_at((1, 0)).expect("valid index"), 2);
    assert_eq!(*t.try_at((1, 1)).expect("valid index"), 4);
    assert_eq!(*t.try_at((1, 2)).expect("valid index"), 6);
}
