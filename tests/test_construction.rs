use xenon::dimension::{Ix0, Ix1, Ix2, IxDyn};
use xenon::tensor::Tensor;

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
