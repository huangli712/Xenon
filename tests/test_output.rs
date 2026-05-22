// tests/test_output.rs
//
// Integration coverage for tensor Display / Debug formatting
// per 22-output.md §8.

use xenon::complex::Complex;
use xenon::dimension::{Ix0, Ix1, Ix2};
use xenon::format::FormatConfig;
use xenon::tensor::{Tensor, Tensor1, Tensor2};

#[test]
fn test_display_small_tensor() {
    // 1-D tensor
    let t = Tensor1::from_shape_vec(Ix1(3), vec![1_i32, 2, 3])
        .expect("valid construction");
    assert_eq!(format!("{}", t), "[1, 2, 3]");

    // 2-D tensor (F-order: shape [2, 2], data [1, 2, 3, 4])
    // logical = [[1, 3], [2, 4]]
    let t2 = Tensor2::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4])
        .expect("valid construction");
    let text = format!("{}", t2);
    assert!(text.starts_with("[["));
    assert!(text.ends_with("]]"));
    // F-order logical: first row [1, 3], second row [2, 4]
    assert!(text.contains("[1, 3]"));
    assert!(text.contains("[2, 4]"));
}

#[test]
fn test_display_truncated() {
    // Create a tensor larger than default threshold (1000).
    let n = 1005_usize;
    let data: Vec<i32> = (0..n as i32).collect();
    let t = Tensor1::from_shape_vec(Ix1(n), data).expect("valid construction");
    let text = format!("{}", t);
    // Display truncation appends "... (N elements omitted)  shape=[...]".
    assert!(text.contains("elements omitted"), "truncation suffix missing");
    assert!(text.contains("shape=[1005]"), "shape in truncation suffix");
    // Head elements present.
    assert!(text.contains("0, 1, 2"));
    // Tail elements present.
    assert!(text.contains("1002, 1003, 1004"));
}

#[test]
fn test_debug_includes_metadata() {
    let t = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
        .expect("valid construction");
    let text = format!("{:?}", t);
    // Header: shape, strides, dtype, layout.
    assert!(text.contains("shape=[2, 2]"), "shape in debug header");
    assert!(text.contains("strides="), "strides in debug header");
    assert!(text.contains("dtype=i32"), "dtype in debug header");
    assert!(text.contains("layout="), "layout in debug header");
    // Data section present.
    assert!(text.contains("[1, 3]") || text.contains("[1, 2]"),
        "data section must contain tensor elements; got: {text}");
}

#[test]
fn test_output_complex() {
    let t = Tensor1::from_shape_vec(Ix1(2), vec![
        Complex::new(1.0_f64, 2.0),
        Complex::new(3.0_f64, 4.0),
    ]).expect("valid construction");
    let text = format!("{}", t);
    assert_eq!(text, "[1+2j, 3+4j]");
}

#[test]
fn test_output_complex_signed_special_values() {
    let t = Tensor1::from_shape_vec(Ix1(4), vec![
        Complex::new(1.0_f64, -2.0),
        Complex::new(-3.0_f64, -4.0),
        Complex::new(f64::NAN, 1.0),
        Complex::new(1.0, f64::NEG_INFINITY),
    ]).expect("valid construction");
    let text = format!("{}", t);
    assert_eq!(text, "[1-2j, -3-4j, NaN+1j, 1-infj]");
}

#[test]
fn test_scalar_vs_zero_dim_formatting() {
    // 0-D tensor: Display with Tensor0(...) wrapper.
    let scalar = Tensor::<i32, Ix0>::from_shape_vec(Ix0, vec![42])
        .expect("valid construction");
    assert_eq!(format!("{}", scalar), "Tensor0(42)");

    // Debug includes header plus Tensor0(...) data.
    let debug = format!("{:?}", scalar);
    assert!(debug.contains("Tensor0(42)"), "debug Tensor0 data: {debug}");
}
