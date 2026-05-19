//! Integration tests for dimension types used as Tensor construction inputs.
//!
//! Per `02-dimension.md` §8.5 line 1121, this file covers `IntoDimension` in
//! Tensor::from_shape_vec / zeros end-to-end paths. The actual Tensor type is
//! introduced in W8; tests below are split into:
//!
//! - **W3-runnable tests**: dimension-layer `IntoDimension` compatibility,
//!   verifying that tuples / arrays / slices / Vecs produce the expected
//!   `Dim` types. These run today.
//! - **W8 placeholder tests** marked `#[ignore]`: stubs for the Tensor
//!   end-to-end path. W8 implementation must remove `#[ignore]` and add the
//!   actual Tensor::from_shape_vec / zeros calls.

use xenon::dimension::{Dimension, IntoDimension, Ix0, Ix1, Ix2, Ix3, Ix6, IxDyn};

/// Tuple input → static Ix1-Ix6.
#[test]
fn test_tuple_inputs_for_tensor_construction() {
    let _: <(usize,) as IntoDimension>::Dim = (5,).into_dimension();
    let d2: Ix2 = (3, 4).into_dimension();
    assert_eq!(d2.slice(), &[3, 4]);
    let d6: Ix6 = (1, 2, 3, 4, 5, 6).into_dimension();
    assert_eq!(d6.slice(), &[1, 2, 3, 4, 5, 6]);
}

/// Array input preserves static dimensionality (per §5.6 line 504).
#[test]
fn test_array_inputs_for_tensor_construction() {
    let d0: Ix0 = [].into_dimension();
    let _ = d0;
    let d3: Ix3 = [2, 3, 4].into_dimension();
    assert_eq!(d3.slice(), &[2, 3, 4]);
}

/// Slice and Vec inputs produce IxDyn (dynamic dimension).
#[test]
fn test_dynamic_inputs_for_tensor_construction() {
    let dyn_dim: IxDyn = (&[2, 3, 4, 5][..]).into_dimension();
    assert_eq!(dyn_dim.ndim(), 4);
    assert_eq!(dyn_dim.checked_size(), Ok(120));
    let dyn_vec: IxDyn = vec![10, 20].into_dimension();
    assert_eq!(dyn_vec.slice(), &[10, 20]);
}

/// Zero-rank scalar input — Ix0 from unit tuple or empty array.
#[test]
fn test_scalar_input_for_tensor_construction() {
    let _: Ix0 = ().into_dimension();
    let _: Ix0 = [].into_dimension();
    assert_eq!(Ix0.checked_size(), Ok(1));
}

/// Zero-length axis is a valid Tensor shape (size = 0, not an error).
#[test]
fn test_zero_axis_input_for_tensor_construction() {
    let d = Ix1(0);
    assert_eq!(d.checked_size(), Ok(0));
    let dyn_d: IxDyn = vec![3, 0, 5].into_dimension();
    assert_eq!(dyn_d.checked_size(), Ok(0));
}

// ── W8 activation placeholders ──

/// Placeholder for W8: Tensor::from_shape_vec accepts tuple/array/slice via
/// IntoDimension. Activate by removing #[ignore] in W8 after Tensor type
/// exists.
#[test]
#[ignore = "W8 activation required: Tensor type not yet defined"]
fn test_tensor_from_shape_vec_accepts_intodimension() {
    // W8 will implement:
    //   let t = Tensor::<f64, _>::from_shape_vec((3, 4), vec![0.0; 12]).unwrap();
    //   assert_eq!(t.shape(), &[3, 4]);
    //   let t = Tensor::<f64, _>::from_shape_vec(&[2, 3, 4][..], vec![0.0; 24]).unwrap();
    //   assert_eq!(t.shape(), &[2, 3, 4]);
    unimplemented!("W8 Tensor type not yet available; replace body with the commented-out assertions above when Tensor is defined");
}

/// Placeholder for W8: Tensor::zeros via IntoDimension.
#[test]
#[ignore = "W8 activation required: Tensor type not yet defined"]
fn test_tensor_zeros_accepts_intodimension() {
    // W8 will implement:
    //   let t = Tensor::<f32, _>::zeros((10, 20));
    //   assert_eq!(t.dim().slice(), &[10, 20]);
    unimplemented!("W8 Tensor type not yet available; replace body with the commented-out assertions above when Tensor is defined");
}

/// Placeholder for W8: element type interaction via Tensor::from_shape_vec.
#[test]
#[ignore = "Requires W8 (Tensor Core) to be completed first"]
fn test_tensor_accepts_element_types() {
    // W8 will implement:
    //   let values = vec![1.0f64, 2.0, 3.0, 4.0];
    //   let tensor = Tensor::<f64, Ix2>::from_shape_vec((2, 2), values).unwrap();
    //   assert_eq!(tensor.shape(), &[2, 2]);
    unimplemented!("W8 Tensor type not yet available; replace body with the commented-out assertions above when Tensor is defined");
}
