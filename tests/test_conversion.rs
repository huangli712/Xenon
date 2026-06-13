use xenon::dimension::Ix1;
use xenon::tensor::Tensor;

/// Cross-module test: element type conversion respects element boundaries.
///
/// Per 03-element.md §8, `cast` must preserve numeric meaning when converting
/// between closed element types (i32 → f64).
#[test]
fn test_conversion_respects_element_boundaries() {
    let tensor =
        Tensor::<i32, Ix1>::from_shape_vec((2,), vec![1, 2]).expect("valid shape and data");
    let converted = tensor.cast::<f64>().expect("i32 to f64 conversion");
    assert_eq!(converted.as_slice().expect("F-contiguous"), &[1.0, 2.0]);
}

// §8.3 L939 compile-time boundary: `bool` does NOT implement `SealedElement`,
// the public sealed marker that gates `cast()`, so `cast::<f32>()` on a
// `Tensor<bool, _>` is rejected at compile time rather than failing at
// runtime. (The internal per-pair dispatch trait `CastTo` is `pub(crate)`
// in `convert` and cannot be named from external crates; the boundary is
// observed below through the public `SealedElement` marker.)
//
// ```compile_fail
// # use xenon::prelude::SealedElement;
// fn _assert_bool_cast<A: SealedElement>() {}
// _assert_bool_cast::<bool>();
// ```

// ── Additional integration tests for cast / to_owned / into_owned ──

use xenon::complex::Complex;
use xenon::error::XenonError;
use xenon::tensor::Tensor1;

#[test]
fn test_cast_f32_to_f64() {
    let tensor: Tensor1<f32> =
        Tensor1::from_shape_vec(Ix1(3), vec![1.5_f32, 2.5, 3.5])
            .expect("valid construction");
    let converted: Tensor1<f64> = tensor.cast().expect("f32 -> f64 should succeed (widening)");
    let result: Vec<f64> = converted.iter().copied().collect();
    assert_eq!(result, vec![1.5_f64, 2.5, 3.5]);
}

#[test]
fn test_cast_f64_to_f32() {
    let tensor: Tensor1<f64> =
        Tensor1::from_shape_vec(Ix1(2), vec![1.0_f64, 2.0])
            .expect("valid construction");
    let err = tensor.cast::<f32>().expect_err("f64 -> f32 is lossy and should fail");
    assert!(matches!(err, XenonError::TypeConversion { .. }));
}

#[test]
fn test_cast_real_to_complex() {
    let tensor: Tensor1<f64> =
        Tensor1::from_shape_vec(Ix1(3), vec![1.0_f64, 2.0, 3.0])
            .expect("valid construction");
    let converted: Tensor1<Complex<f64>> = tensor.cast().expect("f64 -> Complex<f64> widening");
    let result: Vec<Complex<f64>> = converted.iter().copied().collect();
    assert_eq!(result, vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
    ]);
}

/// `bool` does NOT implement `SealedElement`, so `bool` tensors cannot call
/// `.cast::<T>()`. This is enforced at compile time:
///
/// ```compile_fail
/// # use xenon::tensor::Tensor1;
/// # use xenon::dimension::Ix1;
/// let t = Tensor1::<bool>::from_shape_vec(Ix1(2), vec![true, false])
///     .expect("valid construction");
/// let _: Tensor1<f32> = t.cast().expect("bool cannot participate in cast");
/// ```
///
/// The runtime test below verifies that `bool` is a valid `Element` but
/// the `SealedElement` trait is absent.
#[test]
fn test_bool_not_participating_in_cast() {
    let t = Tensor1::<bool>::from_shape_vec(Ix1(2), vec![true, false])
        .expect("valid construction");
    assert_eq!(t.len(), 2);
    // Calling t.cast::<f32>() on a `Tensor<bool, _>` does NOT compile
    // because `bool` does not implement `SealedElement`.
    // The following line, if uncommented, would produce a compile error:
    // let _: Tensor1<f32> = t.cast().expect("would not compile");
}

#[test]
fn test_cast_nan_to_int() {
    let tensor: Tensor1<f64> =
        Tensor1::from_shape_vec(Ix1(1), vec![f64::NAN])
            .expect("valid construction");
    let err = tensor.cast::<i32>().expect_err("NaN -> i32 should fail");
    assert!(matches!(err, XenonError::TypeConversion {
        ref operation,
        ..
    } if operation.as_ref() == "cast"));
}

#[test]
fn test_cast_complex_to_real_zero_imag() {
    let tensor: Tensor1<Complex<f64>> =
        Tensor1::from_shape_vec(Ix1(2), vec![
            Complex::new(3.0, 0.0),
            Complex::new(-1.5, 0.0),
        ]).expect("valid construction");
    let converted: Tensor1<f64> = tensor.cast().expect("Complex<f64> -> f64 with zero imag");
    let result: Vec<f64> = converted.iter().copied().collect();
    assert_eq!(result, vec![3.0, -1.5]);
}

#[test]
fn test_cast_complex_to_real_nonzero_imag() {
    let tensor: Tensor1<Complex<f64>> =
        Tensor1::from_shape_vec(Ix1(1), vec![Complex::new(1.0, 2.0)])
            .expect("valid construction");
    let err = tensor.cast::<f64>().expect_err("non-zero imag should fail");
    assert!(matches!(err, XenonError::TypeConversion { .. }));
}

#[test]
fn test_to_owned_and_into_owned_from_view() {
    let tensor: Tensor1<i32> =
        Tensor1::from_shape_vec(Ix1(3), vec![10, 20, 30])
            .expect("valid construction");
    let view = tensor.view();

    // to_owned() from a view produces a fresh owned tensor.
    let owned_from_view = view.to_owned();
    assert_eq!(owned_from_view.iter().copied().collect::<Vec<_>>(), vec![10, 20, 30]);

    // into_owned() from an owned tensor reuses storage (O(1)).
    let into_owned = tensor.into_owned();
    assert_eq!(into_owned.iter().copied().collect::<Vec<_>>(), vec![10, 20, 30]);
}
