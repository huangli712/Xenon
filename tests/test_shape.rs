//! Integration tests for dimension types in shape operation paths.
//!
//! Per `02-dimension.md` §8.5 line 1122, this file covers `Axis` /
//! `BroadcastDim` in reshape / transpose / broadcast paths. The high-level
//! shape operations are introduced in W11 (broadcast) and W16 (reshape /
//! transpose); tests here are split into:
//!
//! - **W3-runnable**: pure dimension-layer shape contracts (rank, slice,
//!   checked_size, Axis access, checked() validation, equality).
//! - **W3T22 active**: `test_broadcast_dim_compatibility` exercises the
//!   `BroadcastDim` trait (activated by W3T22).
//! - **W11/W16 placeholders** marked `#[ignore]`: stubs for broadcast,
//!   reshape, transpose paths. Activate by removing `#[ignore]` in the
//!   corresponding Wave.

use xenon::dimension::{Axis, BroadcastDim, Dimension, Ix0, Ix1, Ix2, Ix3, IxDyn};
use xenon::error::XenonError;

/// shape contract: rank/slice/checked_size for static dimensions.
#[test]
fn test_static_dimension_shape_contract() {
    let d = Ix3(2, 3, 4);
    assert_eq!(d.ndim(), 3);
    assert_eq!(d.slice(), &[2, 3, 4]);
    assert_eq!(d.checked_size(), Ok(24));
}

/// shape contract: rank/slice/checked_size for IxDyn.
#[test]
fn test_dynamic_dimension_shape_contract() {
    let d = IxDyn::from_vec(vec![2, 3, 4, 5]);
    assert_eq!(d.ndim(), 4);
    assert_eq!(d.checked_size(), Ok(120));
}

/// `Dimension::checked()` validates shape metadata without consuming size.
#[test]
fn test_dimension_checked_validates_shape() {
    assert_eq!(Ix3(2, 3, 4).checked(), Ok(()));
    assert_eq!(IxDyn::from_slice(&[1, 2, 3]).checked(), Ok(()));
    // Overflow case: checked() returns Err same kind as checked_size.
    assert!(Ix2(usize::MAX, 2).checked().is_err());
}

/// Shape equality: two dimensions of same type and same axis lengths are
/// equal.
#[test]
fn test_shape_equality() {
    assert_eq!(Ix3(2, 3, 4), Ix3(2, 3, 4));
    assert_ne!(Ix3(2, 3, 4), Ix3(2, 3, 5));
    assert_eq!(IxDyn::from_slice(&[1, 2, 3]), IxDyn::from_slice(&[1, 2, 3]));
    assert_ne!(IxDyn::from_slice(&[1, 2, 3]), IxDyn::from_slice(&[1, 2]));
}

/// Axis access on shape: legal axis returns Ok(length), out-of-range returns
/// InvalidAxis.
#[test]
fn test_shape_axis_access() {
    let d = Ix3(2, 3, 4);
    assert_eq!(d.axis(Axis::new(0)), Ok(2));
    assert_eq!(d.axis(Axis::new(1)), Ok(3));
    assert_eq!(d.axis(Axis::new(2)), Ok(4));
    assert!(matches!(
        d.axis(Axis::new(3)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

/// Single-element shape and zero-axis shape boundaries.
#[test]
fn test_shape_boundary_cases() {
    assert_eq!(Ix0.checked_size(), Ok(1));
    assert_eq!(Ix2(1, 1).checked_size(), Ok(1));
    assert_eq!(Ix2(0, 5).checked_size(), Ok(0));
}

// ── W3T22-gated: BroadcastDim type-level test ──

/// W3T22 type-level test: `BroadcastDim` output inference.
///
/// Activated after W3T22 lands: BroadcastDim trait + 64 impls now available.
#[test]
fn test_broadcast_dim_compatibility() {
    // Same-rank: Ix2 BroadcastDim Ix2 → Ix2.
    fn _check_same<D: BroadcastDim<D, Output = D>>() {}
    _check_same::<Ix2>();
    // Cross-rank: Ix1 BroadcastDim Ix3 → Ix3 (higher rank wins).
    fn _check_cross<A, B, O>()
    where
        A: BroadcastDim<B, Output = O>,
        B: Dimension,
        O: Dimension,
    {
    }
    _check_cross::<Ix1, Ix3, Ix3>();
    // Mixed with IxDyn: any side IxDyn ⇒ IxDyn.
    _check_cross::<Ix2, IxDyn, IxDyn>();
    _check_cross::<IxDyn, Ix2, IxDyn>();
}

// ── W11 activation placeholder ──

/// Placeholder for W11: runtime broadcast path (`broadcast_to` /
/// `broadcast_with`).
#[test]
#[ignore = "W11 activation required: broadcast_to / broadcast_with not yet defined"]
fn test_broadcast_runtime_with_dim() {
    // W11 will implement, e.g.:
    //   let t = Tensor2::<f64>::zeros([1, 4]).unwrap();
    //   let v = t.broadcast_to([3, 4]).unwrap();
    //   assert_eq!(v.shape(), &[3, 4]);
    panic!("W11 placeholder — must be replaced before W11 completion");
}

// ── W16 activation placeholders ──

/// Placeholder for W16: reshape path using `IntoDimension`.
#[test]
#[ignore = "W16 activation required: reshape not yet implemented"]
fn test_reshape_via_dimension() {
    // W16 will implement reshape on Tensor, e.g.:
    //   let t = Tensor::<f32, _>::zeros((6,));
    //   let reshaped = t.reshape((2, 3)).unwrap();
    //   assert_eq!(reshaped.dim().slice(), &[2, 3]);
    panic!("W16 placeholder — must be replaced before W16 completion");
}

/// Placeholder for W16: transpose via Axis.
#[test]
#[ignore = "W16 activation required: transpose not yet implemented"]
fn test_transpose_via_axes() {
    // W16 will implement transpose using Axis pairs.
    panic!("W16 placeholder — must be replaced before W16 completion");
}
