//! Integration tests for dimension types in indexing paths.
//!
//! Per `02-dimension.md` §8.5 line 1123, this file covers the `D: Dimension`
//! bound at indexing/slicing entry points. Tensor indexing/slicing is
//! introduced in W11+ (post-Tensor); this file is split into:
//!
//! - **W3-runnable**: dimension-layer indexing contracts:
//!     * `Index<usize>` for Ix1 and Ix2 only (per §5.4)
//!     * `Dimension::axis(Axis)` access for all dimensions
//!     * `Axis` navigation helpers in indexing context
//! - **W11+ placeholders** marked `#[ignore]`: stubs for Tensor indexing paths.

use xenon::dimension::{Axis, Dimension, Ix0, Ix1, Ix2, Ix3, IxDyn};
use xenon::error::XenonError;

/// `Index<usize>` impl on Ix1 (per W3T5 / §5.4).
#[test]
fn test_ix1_index_via_usize() {
    let d = Ix1(42);
    assert_eq!(d[0], 42);
}

/// `Index<usize>` impl on Ix2 (per W3T6 / §5.4).
#[test]
fn test_ix2_index_via_usize() {
    let d = Ix2(10, 20);
    assert_eq!(d[0], 10);
    assert_eq!(d[1], 20);
}

/// Ix1 out-of-bounds index panics per Index<usize> default semantics.
#[test]
#[should_panic(expected = "Ix1 index out of bounds")]
fn test_ix1_index_out_of_bounds_panics() {
    let d = Ix1(5);
    let _ = d[1];
}

/// Ix2 out-of-bounds index panics.
#[test]
#[should_panic(expected = "Ix2 index out of bounds")]
fn test_ix2_index_out_of_bounds_panics() {
    let d = Ix2(5, 6);
    let _ = d[2];
}

/// `Dimension::axis(Axis)` is the canonical API for axis access on all
/// dimension types (Ix0-Ix6, IxDyn). Higher-rank static types (Ix3-Ix6) do
/// NOT implement `Index<usize>` — verify via `axis()`.
#[test]
fn test_higher_rank_axis_access_uses_axis_method() {
    let d3 = Ix3(2, 3, 4);
    assert_eq!(d3.axis(Axis::new(0)), Ok(2));
    assert_eq!(d3.axis(Axis::new(1)), Ok(3));
    assert_eq!(d3.axis(Axis::new(2)), Ok(4));
    assert!(matches!(
        d3.axis(Axis::new(3)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

/// IxDyn axis access via `axis()` (Index<usize> not implemented on IxDyn).
#[test]
fn test_ixdyn_axis_access() {
    let d = IxDyn::from_vec(vec![7, 8, 9]);
    assert_eq!(d.axis(Axis::new(0)), Ok(7));
    assert_eq!(d.axis(Axis::new(1)), Ok(8));
    assert_eq!(d.axis(Axis::new(2)), Ok(9));
    assert!(matches!(
        d.axis(Axis::new(3)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

/// Ix0 axis access always fails: scalar has no axes.
#[test]
fn test_ix0_axis_access_always_invalid() {
    assert!(matches!(
        Ix0.axis(Axis::new(0)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

/// `Axis` navigation in indexing context: traversing axes of an Ix3.
#[test]
fn test_axis_navigation_traverses_dimensions() {
    let d = Ix3(2, 3, 4);
    let mut current = Axis::new(0);
    let mut visited: Vec<usize> = Vec::new();
    while let Ok(len) = d.axis(current) {
        visited.push(len);
        match current.next() {
            Some(n) => current = n,
            None => break,
        }
    }
    assert_eq!(visited, vec![2, 3, 4]);
}

// ── W11+ activation placeholders ──

/// Placeholder for W11+: Tensor element indexing via dimension-typed index.
#[test]
#[ignore = "W11+ activation required: Tensor type and indexing API not yet defined"]
fn test_tensor_element_indexing_via_dimension() {
    // W11+ will implement, e.g.:
    //   let t = Tensor::<f64, _>::zeros((3, 4));
    //   let idx: Ix2 = (1, 2).into_dimension();
    //   let elem = t[idx];
    //   assert_eq!(elem, 0.0);
    panic!("W11+ placeholder — must be replaced before that wave completion");
}

/// Placeholder for W11+: Tensor axis slicing via Axis.
#[test]
#[ignore = "W11+ activation required: slicing API not yet defined"]
fn test_tensor_axis_slicing_via_axis() {
    // W11+ will implement, e.g.:
    //   let t = Tensor::<f64, _>::zeros((3, 4));
    //   let row = t.index_axis(Axis::new(0), 1);
    //   assert_eq!(row.shape(), &[4]);
    panic!("W11+ placeholder — must be replaced before that wave completion");
}
