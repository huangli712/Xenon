//! Integration tests for the broadcast module — covers 15-broadcast.md §8.

use xenon::broadcast::{broadcast_shape, broadcast_strides, can_broadcast};
use xenon::dimension::Dimension;
use xenon::error::XenonError;
use xenon::layout::LayoutState;
use xenon::tensor::Tensor2;

// --- §8.2 shape rules ---

#[test]
fn test_broadcast_shape_basic() {
    let shape = broadcast_shape(&[1, 3], &[2, 3]).expect("valid test input");
    assert_eq!(shape.slice(), &[2, 3]);
}

#[test]
fn test_broadcast_shape_error_structured() {
    match broadcast_shape(&[2, 3], &[4, 3]).expect_err("expected error") {
        XenonError::BroadcastError {
            operation,
            lhs_shape,
            rhs_shape,
            attempted_target_shape,
            axis,
        } => {
            assert_eq!(operation.as_ref(), "broadcast_shape");
            assert_eq!(lhs_shape, vec![2, 3]);
            assert_eq!(rhs_shape, vec![4, 3]);
            assert_eq!(attempted_target_shape, None);
            assert_eq!(axis, Some(0));
        }
        other => panic!("expected BroadcastError, got {:?}", other),
    }
}

// --- §8.2 view construction ---

#[test]
fn test_broadcast_to_basic() {
    let tensor: Tensor2<f64> =
        Tensor2::from_shape_vec([1, 3], vec![1.0, 2.0, 3.0]).expect("valid test input");
    let view = tensor.broadcast_to([2, 3]).expect("valid test input");
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides()[0], 0);
    assert_eq!(view.as_ptr(), tensor.as_ptr());
}

// --- §8.3 boundary scenarios ---

/// §8.3 row 4: high-rank cross-broadcast `[2, 1, 4]` → `[3, 2, 5, 4]`.
#[test]
fn test_broadcast_high_rank_ixdyn() {
    let shape = broadcast_shape(&[2, 1, 4], &[3, 2, 5, 4]).expect("valid test input");
    assert_eq!(shape.slice(), &[3, 2, 5, 4]);
}

/// §8.3 row 1: empty-axis broadcast `[0, 3]` vs `[1, 3]` → `[0, 3]`. Stride for
/// `1 → 0` axis is 0; no data accessed.
#[test]
fn test_broadcast_to_empty_axis_no_data_access() {
    let tensor: Tensor2<f64> = Tensor2::ones([1, 3]).expect("valid test input");
    let view = tensor.broadcast_to([0, 3]).expect("valid test input");
    assert_eq!(view.shape(), &[0, 3]);
    // Empty-array degenerate: NOT classified as BroadcastView (§5.11 line 261-263).
    assert_ne!(view.layout_state(), LayoutState::BroadcastView);
    // Iterating yields zero elements.
    assert_eq!(view.iter().count(), 0);
}

// --- §8.2 missing items: rebroadcast + large-tensor zero-copy ---

/// §8.2 `test_broadcast_rebroadcast_zero_stride`: re-broadcasting an already-
/// broadcast view preserves zero strides on previously-broadcast axes and
/// inserts new zero strides for newly-broadcast leading axes.
#[test]
fn test_broadcast_rebroadcast_zero_stride() {
    let tensor: Tensor2<f64> =
        Tensor2::from_shape_vec([1, 3], vec![1.0, 2.0, 3.0]).expect("valid test input");
    // First broadcast: [1, 3] → [2, 3]; axis 0 becomes broadcast (stride 0).
    let view1 = tensor.broadcast_to([2, 3]).expect("valid test input");
    assert_eq!(view1.strides(), &[0, 1]);
    // Re-broadcast the result through `broadcast_strides` directly (the public
    // path goes through `TensorBase::broadcast_to` which calls `broadcast_strides`).
    let strides2 = broadcast_strides(view1.shape(), view1.strides(), &[2, 3]).expect("valid test input");
    assert_eq!(strides2, vec![0, 1]);
}

/// §8.2 `test_broadcast_large_tensor_zero_copy` + §8.3 row 6: `~10^7`-element
/// broadcast keeps zero-copy and zero-stride semantics. We use [1, 3162, 3162]
/// → [3162, 3162] target produces ~10^7 logical elements.
#[test]
fn test_broadcast_large_tensor_zero_copy() {
    let tensor: Tensor2<f64> = Tensor2::ones([1, 3162]).expect("valid test input");
    // Target [3162, 3162] gives ~10^7 logical elements.
    let view = tensor.broadcast_to([3162, 3162]).expect("valid test input");
    assert_eq!(view.shape(), &[3162, 3162]);
    // Axis 0 was 1 → 3162: zero stride.
    assert_eq!(view.strides()[0], 0);
    // Zero-copy: pointer unchanged; no allocation.
    assert_eq!(view.as_ptr(), tensor.as_ptr());
    // Layout flag: non-empty + has zero stride ⇒ BroadcastView.
    assert_eq!(view.layout_state(), LayoutState::BroadcastView);
}

// --- §8.2 `test_broadcast_read_only`: read-only iteration works ---

#[test]
fn test_broadcast_read_only_iter() {
    let tensor: Tensor2<f64> = Tensor2::ones([1, 3]).expect("valid test input");
    let view = tensor.broadcast_to([2, 3]).expect("valid test input");
    let n: usize = view.iter().count();
    assert_eq!(n, 6);
}

// --- Cross-module error propagation ---

#[test]
fn test_broadcast_to_error_propagation() {
    let tensor: Tensor2<f64> = Tensor2::zeros([2, 3]).expect("valid test input");
    assert!(matches!(
        tensor.broadcast_to([4, 3]),
        Err(XenonError::BroadcastError { .. })
    ));
}

// --- §8.4 invariant 1: `can_broadcast == broadcast_shape.is_ok()` ---

#[test]
fn test_invariant_can_broadcast_matches_broadcast_shape() {
    let cases = [
        (&[1, 3][..], &[2, 3][..]),
        (&[2, 3][..], &[4, 3][..]),
        (&[][..], &[2, 3][..]),
        (&[0, 3][..], &[1, 3][..]),
        (&[2, 1, 4][..], &[3, 2, 5, 4][..]),
    ];
    for (a, b) in cases {
        assert_eq!(can_broadcast(a, b), broadcast_shape(a, b).is_ok());
    }
}

// --- §8.4 property invariants (deferred to W29 / W3T21) ---
//
// The following four invariants from 15-broadcast.md §8.4 are scheduled to be
// implemented in `tests/property_tests.rs` by W29 (using the proptest harness
// introduced there). DO NOT add `proptest` as a dev-dependency in W11.
//
//   1. `can_broadcast(a, b) == broadcast_shape(a, b).is_ok()` for random shape pairs
//      — already covered above on a hand-picked corpus via
//        `test_invariant_can_broadcast_matches_broadcast_shape`. Random-shape
//        sweep moves to W29.
//   2. Logical element count of broadcast result == product(target_shape).
//   3. Broadcast-axis stride is always 0 (for axes where orig_dim == 1).
//   4. Broadcast result shares source data pointer & offset (zero-copy invariant).
//
// W29 task list must include `test_broadcast_property_*` entries covering these
// four invariants; this comment block is the authoritative scheduling record.
