//! Placeholder for W25/W8: type conversion across element boundaries.

#[test]
#[ignore = "Requires W8 (Tensor Core) and W25 (Type Conversion) to be completed first"]
fn test_conversion_respects_element_boundaries() {
    // Requires: W8 (Tensor), W3 (Dimension), W25 (Type Conversion)
    // W8/W25 will implement:
    //   let tensor = Tensor::<i32, Ix1>::from_shape_vec((2,), vec![1, 2]).unwrap();
    //   let converted = tensor.cast::<f64>().unwrap();
    //   assert_eq!(converted.to_vec(), vec![1.0, 2.0]);
    //
    // §8.3 L939 boundary: bool must NOT satisfy CastTo<f32> (compile-time check).
    // Uncommenting the line below MUST trigger a compile error:
    //   fn _assert_bool_cast<A: CastTo<f32>>() {}
    //   _assert_bool_cast::<bool>();
    //
    // TODO(post-W4): The commented-out code above currently exists as a comment
    // placeholder. Once a trybuild (or equivalent compile-fail framework) is
    // adopted, convert this into a formal compile-fail test case; see
    // 28-tests.md §9.2 for the compile-fail coverage strategy.
    unimplemented!("W8/W25 not yet available; replace body when Tensor and cast APIs are defined");
}
