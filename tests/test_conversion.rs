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

// §8.3 L939 compile-time boundary: bool must NOT satisfy CastTo<f32>.
// This is verified at compile time — attempting to use `cast::<f32>()` on
// a `Tensor<bool, _>` fails because `bool: CastElement` but there is no
// `impl ConvertTo<f32> for bool`.
//
// ```compile_fail
// # use xenon::element::CastTo;
// fn _assert_bool_cast<A: CastTo<f32>>() {}
// _assert_bool_cast::<bool>();
// ```
