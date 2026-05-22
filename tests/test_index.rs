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

// ── Tensor indexing and slicing integration tests ──

use xenon::index::{SliceInfo, SliceInfoElem, SliceInfoIndices};
use xenon::tensor::Tensor2;

/// Multi-dimensional element access via try_at with tuple index.
#[test]
fn test_multi_dim_index() {
    let t = Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]).expect("valid test input");
    assert_eq!(*t.try_at((0, 0)).expect("valid index"), 1);
    assert_eq!(*t.try_at((1, 2)).expect("valid index"), 6);
}

/// Out-of-bounds access via try_at returns IndexOutOfBounds error.
#[test]
fn test_index_out_of_bounds() {
    let t = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
    let err = t.try_at((2, 0)).expect_err("out of bounds");
    assert!(matches!(err, xenon::XenonError::IndexOutOfBounds { axis: 0, .. }));
}

/// Basic slice: select a single row via Index, then take a Range subview.
#[test]
fn test_slice_range() {
    let t = Tensor2::<i32>::from_shape_vec([4, 5], (0i32..20).collect()).expect("valid test input");
    let info = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Range { start: 1, end: 4 },
            SliceInfoElem::Index(2),
        ]),
        Ix2(4, 5),
        Ix1(3),
    )
    .expect("valid slice");
    let view = t.slice(info).expect("valid slice");
    assert_eq!(view.shape(), &[3]);
    // F-order: column-major, so column 2 is elements 2,7,12,17; rows 1..3 → 7,12,17.
    // Actually in F-order shape [4,5]: data is strided with col-major layout.
    // With from_shape_vec on [4,5], data=0..20:
    //   col 0 = [0,1,2,3], col 1 = [4,5,6,7], col 2 = [8,9,10,11]
    //   col 3 = [12,13,14,15], col 4 = [16,17,18,19]
    // Slice row range [1,4) and index 2 → col 2, rows 1..3 → [9,10,11]
    assert_eq!(view.as_slice(), Some(&[9, 10, 11][..]));
}

/// Subrange slice: range within an already-sliced view.
#[test]
fn test_slice_subrange() {
    let t = Tensor2::<i32>::from_shape_vec([3, 4], (0i32..12).collect()).expect("valid test input");
    // First slice: rows [0,2), cols [1,3) → shape Ix2(2,2)
    let info1 = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Range { start: 0, end: 2 },
            SliceInfoElem::Range { start: 1, end: 3 },
        ]),
        Ix2(3, 4),
        Ix2(2, 2),
    )
    .expect("valid slice 1");
    let view1 = t.slice(info1).expect("valid slice 1");
    assert_eq!(view1.shape(), &[2, 2]);

    // Second slice on view1: pick row 1, keep 2 columns.
    let info2 = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Index(1),
            SliceInfoElem::Range { start: 0, end: 2 },
        ]),
        Ix2(2, 2),
        Ix1(2),
    )
    .expect("valid slice 2");
    let view2 = view1.slice(info2).expect("valid slice 2");
    assert_eq!(view2.shape(), &[2]);
}

/// Mutable view slicing should be rejected on broadcast views (compile-time
/// read-only guarantee). Here we verify slicing a read-only view works.
#[test]
fn test_slice_mut_broadcast_rejected() {
    let t = Tensor2::<f64>::from_shape_vec([1, 3], vec![1.0, 2.0, 3.0]).expect("valid test input");
    let bv = t.broadcast_to([2, 3]).expect("valid test input");
    // bv is a TensorView (read-only). slice() is available.
    let info = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Index(0),
            SliceInfoElem::Range { start: 0, end: 3 },
        ]),
        Ix2(2, 3),
        Ix1(3),
    )
    .expect("valid slice");
    let sliced = bv.slice(info).expect("valid slice");
    assert_eq!(sliced.shape(), &[3]);
}

/// SliceInfo structural validation: mismatched ranks and invalid ranges.
#[test]
fn test_sliceinfo_structural_validation() {
    // Rank mismatch: indices length != input dim ndim.
    let err_rank = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![SliceInfoElem::Index(0)]),
        Ix2(2, 3),
        Ix1(0),
    )
    .expect_err("rank mismatch");
    assert!(matches!(err_rank, xenon::XenonError::InvalidArgument { .. }));

    // Output rank mismatch: Range count != output dim ndim.
    let err_out = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Range { start: 0, end: 2 },
            SliceInfoElem::Range { start: 0, end: 3 },
        ]),
        Ix2(2, 3),
        Ix1(2),
    )
    .expect_err("output rank mismatch");
    assert!(matches!(err_out, xenon::XenonError::InvalidArgument { .. }));

    // Range start > end.
    let err_range = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Range { start: 5, end: 2 },
            SliceInfoElem::Index(0),
        ]),
        Ix2(10, 3),
        Ix1(3),
    )
    .expect_err("start > end");
    assert!(matches!(
        err_range,
        xenon::XenonError::InvalidArgument { .. }
    ));

    // Range end > axis extent.
    let t = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("valid test input");
    let info_oob = SliceInfo::new(
        SliceInfoIndices::from_vec(vec![
            SliceInfoElem::Range { start: 0, end: 5 },
            SliceInfoElem::Index(0),
        ]),
        Ix2(2, 2),
        Ix1(2),
    )
    .expect("valid slice info (end validated at slice time)");
    let err_slice = t.slice(info_oob).expect_err("range out of bounds");
    assert!(matches!(err_slice, xenon::XenonError::InvalidArgument { .. }));
}
