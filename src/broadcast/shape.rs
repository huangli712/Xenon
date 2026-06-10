//! Shape and stride computation for numpy-style broadcasting.

use std::borrow::Cow;

use crate::error::{InvalidArgumentKind, XenonError};
use crate::dimension::IxDyn;

/// Numpy-style broadcast compatibility check. Equivalent to
/// `broadcast_shape(a, b).is_ok()`.
pub fn can_broadcast(shape_a: &[usize], shape_b: &[usize]) -> bool {
    broadcast_shape(shape_a, shape_b).is_ok()
}

/// Compute the broadcast result shape of two input shapes.
///
/// Algorithm (right-align):
///   1. Align dimensions from right to left.
///   2. Missing leading dimensions are treated as 1.
///   3. If two aligned dimensions differ and neither is 1, return BroadcastError.
///   4. Otherwise result = max(a, b): if one is 1, take the other; if both equal
///      (including both 1 or both 0 for empty-axis broadcast), take that value.
///   5. Return the resulting IxDyn shape.
///
/// # Errors
///
/// Returns `XenonError::BroadcastError` when two right-aligned axes differ
/// and neither equals 1. `attempted_target_shape` is `None` because
/// `broadcast_shape` compares two inputs bidirectionally — there is no single
/// target shape.
pub fn broadcast_shape(
    shape_a: &[usize],
    shape_b: &[usize]
) -> Result<IxDyn, XenonError> {
    let ndim = shape_a.len().max(shape_b.len());
    let mut out = vec![1usize; ndim];

    // Iterate over the result axes from index 0 (leading) to ndim-1 (trailing),
    // right-aligning the input shapes.
    for (out_axis, item) in out.iter_mut().enumerate() {
        let from_back = ndim - 1 - out_axis;
        // Missing leading axes are treated as length 1.
        let a = shape_a
            .len()
            .checked_sub(from_back + 1)
            .map(|i| shape_a[i])
            .unwrap_or(1);
        let b = shape_b
            .len()
            .checked_sub(from_back + 1)
            .map(|i| shape_b[i])
            .unwrap_or(1);

        *item = match (a, b) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(broadcast_error(
                    "broadcast_shape",
                    shape_a,
                    shape_b,
                    None,
                    out_axis,
                ));
            },
        };
    }

    Ok(IxDyn::from_vec(out))
}

/// Produce strides for a broadcast view from `orig` to `target`.
///
/// Algorithm:
///   1. Validate rank compatibility (`orig_shape.len() == orig_strides.len()`,
///      and `orig_shape.len() <= target_shape.len()`).
///   2. Right-align the original shape against the target shape.
///   3. For each result axis:
///      - missing leading axis → write stride 0;
///      - orig_dim == target_dim → keep orig_stride (preserves prior zero strides
///        for re-broadcasting);
///      - orig_dim == 1 → write stride 0 (covers empty-axis `1 -> 0`);
///      - otherwise → BroadcastError.
///   4. Return the stride vector.
///
/// # Errors
///
/// - `XenonError::InvalidArgument` — `orig_shape.len() != orig_strides.len()`
///   (caller precondition failure).
/// - `XenonError::BroadcastError` — either `orig_shape.len() > target_shape.len()`
///   (rank-excess, cannot right-align a higher-rank source into a lower-rank
///   target), or some right-aligned axis has `orig_dim != target_dim` and
///   `orig_dim != 1`. `attempted_target_shape` is always populated because this
///   path is single-directional (a specific target is always known).
pub fn broadcast_strides(
    orig_shape: &[usize],
    orig_strides: &[usize],
    target_shape: &[usize],
) -> Result<Vec<usize>, XenonError> {
    // Precondition failure (caller bug) → InvalidArgument.
    if orig_shape.len() != orig_strides.len() {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("broadcast_strides"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("orig_shape/orig_strides"),
                constraint: Cow::Borrowed("orig_shape.len() must equal orig_strides.len()"),
            },
        });
    }

    // Rank mismatch where orig has MORE axes than target is a broadcast
    // incompatibility (cannot right-align a higher-rank source into a
    // lower-rank target).
    if orig_shape.len() > target_shape.len() {
        return Err(broadcast_error(
            "broadcast_strides",
            orig_shape,
            target_shape,
            Some(target_shape),
            0,
        ));
    }

    let mut out = vec![0usize; target_shape.len()];
    let offset = target_shape.len() - orig_shape.len();

    for target_axis in 0..target_shape.len() {
        if target_axis < offset {
            // Missing leading axis on the source side → broadcast, stride 0.
            out[target_axis] = 0;
            continue;
        }

        let input_axis = target_axis - offset;
        let input_dim = orig_shape[input_axis];
        let input_stride = orig_strides[input_axis];
        let target_dim = target_shape[target_axis];

        out[target_axis] = match (input_dim, target_dim) {
            // Equal dim (including `0 == 0` empty axis): keep orig stride.
            (x, y) if x == y => input_stride,
            // Source dim is 1 → broadcast axis (covers `1 -> N` and `1 -> 0`).
            (1, _) => 0,
            // Otherwise incompatible.
            _ => {
                return Err(broadcast_error(
                    "broadcast_strides",
                    orig_shape,
                    target_shape,
                    Some(target_shape),
                    target_axis,
                ));
            },
        };
    }
    Ok(out)
}

/// Constructs `XenonError::BroadcastError` with all fields populated.
///
/// - `operation`: caller name (e.g. "broadcast_shape", "broadcast_strides").
/// - `lhs_shape` / `rhs_shape`: the two compared shapes. For `broadcast_strides`
///   (single-input broadcast), pass `orig_shape` and `target_shape` respectively.
/// - `attempted_target_shape`: caller-supplied target if known (only relevant in
///   `broadcast_to` / `broadcast_strides` paths); `None` for pure `broadcast_shape`.
/// - `axis`: the conflicting axis index (right-aligned, 0-based on the result rank).
pub(crate) fn broadcast_error(
    operation: &'static str,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    attempted_target_shape: Option<&[usize]>,
    axis: usize,
) -> XenonError {
    XenonError::BroadcastError {
        operation: Cow::Borrowed(operation),
        lhs_shape: lhs_shape.to_vec(),
        rhs_shape: rhs_shape.to_vec(),
        attempted_target_shape: attempted_target_shape.map(|s| s.to_vec()),
        axis: Some(axis),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Dimension;
    use crate::error::XenonError;

    /// Tests that `can_broadcast` returns `true` for compatible shape pairs,
    /// including same-shape, right-aligned singleton expansion, missing leading
    /// axes, and cross-rank scenarios.
    #[test]
    fn test_can_broadcast_compatible() {
        // Same shape.
        assert!(can_broadcast(&[2, 3], &[2, 3]));
        // Right-align broadcasting (`1` axis expands).
        assert!(can_broadcast(&[1, 3], &[2, 3]));
        assert!(can_broadcast(&[2, 3], &[1, 3]));
        // Missing leading axis treated as 1 (scalar to high-dim).
        assert!(can_broadcast(&[], &[2, 3]));
        assert!(can_broadcast(&[3], &[2, 3]));
        // Cross-rank scenario.
        assert!(can_broadcast(&[2, 1, 4], &[3, 2, 5, 4]));
    }

    /// Tests that `can_broadcast` returns `false` for incompatible shape pairs
    /// where same-rank axes differ with neither being 1.
    #[test]
    fn test_can_broadcast_incompatible() {
        assert!(!can_broadcast(&[2, 3], &[4, 3]));
        assert!(!can_broadcast(&[2, 3, 4], &[2, 3, 5]));
    }

    /// Tests compatibility for empty-axis broadcasting where one axis is length 0.
    /// Compatibility must not special-case axes of length 0.
    #[test]
    fn test_can_broadcast_empty_axis() {
        assert!(can_broadcast(&[0, 3], &[1, 3]));
    }

    /// Verifies the invariant: `can_broadcast(a, b) == broadcast_shape(a, b).is_ok()`.
    #[test]
    fn test_can_broadcast_matches_broadcast_shape() {
        let cases = [
            (&[1, 3][..], &[2, 3][..]),
            (&[2, 3][..], &[4, 3][..]),
            (&[][..], &[2, 3][..]),
            (&[0, 3][..], &[1, 3][..]),
            (&[2, 1, 4][..], &[3, 2, 5, 4][..]),
        ];
        for (a, b) in cases {
            assert_eq!(
                can_broadcast(a, b),
                broadcast_shape(a, b).is_ok(),
                "mismatch on {:?} vs {:?}",
                a,
                b
            );
        }
    }

    /// Tests basic `broadcast_shape` cases: singleton expansion, scalar to
    /// higher-dim, empty-axis broadcast, and cross-rank broadcast.
    #[test]
    fn test_broadcast_shape_basic() {
        // Right-align with `1` axis expansion.
        let r = broadcast_shape(&[1, 3], &[2, 3]).expect("compatible shapes");
        assert_eq!(r.slice(), &[2, 3]);
        // Scalar to high-dim: missing leading axes are 1.
        let r = broadcast_shape(&[], &[2, 3]).expect("scalar to high-dim");
        assert_eq!(r.slice(), &[2, 3]);
        // Empty-axis broadcast: [0,3] vs [1,3] → [0,3].
        let r = broadcast_shape(&[0, 3], &[1, 3]).expect("empty-axis broadcast");
        assert_eq!(r.slice(), &[0, 3]);
        // High-rank cross-broadcast: [2,1,4] → [3,2,5,4].
        let r = broadcast_shape(&[2, 1, 4], &[3, 2, 5, 4]).expect("cross-rank broadcast");
        assert_eq!(r.slice(), &[3, 2, 5, 4]);
    }

    /// Tests that `broadcast_shape` returns a structured `BroadcastError` with
    /// the correct fields when shapes are incompatible at the leading axis.
    #[test]
    fn test_broadcast_shape_error() {
        let err = broadcast_shape(&[2, 3], &[4, 3]).expect_err("incompatible shapes");
        match err {
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
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    /// Tests that a broadcast conflict at a non-leading (trailing) axis reports
    /// that axis index, verifying the right-aligned `out_axis` computation.
    #[test]
    fn test_broadcast_shape_error_non_leading_axis() {
        let err = broadcast_shape(&[2, 3], &[2, 4]).expect_err("axis-1 conflict");
        match err {
            XenonError::BroadcastError {
                operation,
                lhs_shape,
                rhs_shape,
                attempted_target_shape,
                axis,
            } => {
                assert_eq!(operation.as_ref(), "broadcast_shape");
                assert_eq!(lhs_shape, vec![2, 3]);
                assert_eq!(rhs_shape, vec![2, 4]);
                assert_eq!(attempted_target_shape, None);
                // Axis 0 matches (2 == 2); conflict surfaces at trailing axis 1.
                assert_eq!(axis, Some(1));
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    /// Tests that broadcasting is commutative: the result shape is independent
    /// of argument order.
    #[test]
    fn test_broadcast_shape_commutative() {
        let cases = [
            (&[1, 3][..], &[2, 3][..]),
            (&[][..], &[2, 3][..]),
            (&[0, 3][..], &[1, 3][..]),
            (&[2, 1, 4][..], &[3, 2, 5, 4][..]),
        ];
        for (a, b) in cases {
            let ab = broadcast_shape(a, b).expect("compatible");
            let ba = broadcast_shape(b, a).expect("compatible");
            assert_eq!(ab.slice(), ba.slice(), "asymmetry on {:?} vs {:?}", a, b);
        }
    }

    /// Scalar-to-scalar boundary: two rank-0 shapes broadcast to rank-0.
    #[test]
    fn test_broadcast_shape_scalar_scalar() {
        let r = broadcast_shape(&[], &[]).expect("scalar broadcast");
        assert!(r.slice().is_empty(), "rank-0 broadcast must be empty");
    }

    /// Tests that a singleton axis broadcasts to a larger target with stride 0.
    #[test]
    fn test_broadcast_strides_zero_stride() {
        // orig [1, 3] with strides [3, 1] → target [2, 3]: axis 0 broadcast, stride 0.
        let s = broadcast_strides(&[1, 3], &[3, 1], &[2, 3]).expect("compatible strides");
        assert_eq!(s, vec![0, 1]);
    }

    /// Tests that strides are preserved when no broadcast occurs (shapes match).
    #[test]
    fn test_broadcast_strides_non_negative() {
        // No broadcast occurs when shapes already match: orig stride is preserved.
        let s = broadcast_strides(&[2, 3], &[3, 1], &[2, 3]).expect("compatible strides");
        assert_eq!(s, vec![3, 1]);
    }

    /// Tests the re-broadcast rule: a previously-broadcast input keeps its
    /// existing zero strides when re-broadcast to a larger target.
    #[test]
    fn test_broadcast_strides_rebroadcast_zero_stride() {
        // Source: shape [4] with stride [0] (already a broadcast view).
        // Re-broadcast to [2, 4]: leading axis is new broadcast (stride 0),
        // trailing axis matches dim and keeps the existing zero stride.
        let s = broadcast_strides(&[4], &[0], &[2, 4]).expect("compatible strides");
        assert_eq!(s, vec![0, 0]);
    }

    /// Tests empty-axis broadcast: a source axis of length 1 broadcast to a
    /// target axis of length 0 writes stride 0.
    #[test]
    fn test_broadcast_strides_empty_axis() {
        let s = broadcast_strides(&[1, 3], &[3, 1], &[0, 3]).expect("compatible strides");
        assert_eq!(s, vec![0, 1]);
    }

    /// Tests that `broadcast_strides` returns an `InvalidArgument` error when
    /// `orig_shape.len() != orig_strides.len()`.
    #[test]
    fn test_broadcast_strides_invalid_argument_on_len_mismatch() {
        let err = broadcast_strides(&[1, 3], &[3], &[2, 3]).expect_err("incompatible strides");
        match err {
            XenonError::InvalidArgument { operation, kind } => {
                assert_eq!(operation.as_ref(), "broadcast_strides");
                match kind {
                    InvalidArgumentKind::OperationSpecific {
                        argument,
                        constraint,
                    } => {
                        assert_eq!(argument.as_ref(), "orig_shape/orig_strides");
                        assert!(constraint.as_ref().contains("must equal"));
                    },
                    other => panic!("expected OperationSpecific, got {:?}", other),
                }
            },
            other => panic!("expected InvalidArgument, got {:?}", other),
        }
    }

    /// Tests that `broadcast_strides` returns a `BroadcastError` with the correct
    /// fields when an axis conflict occurs at the leading axis.
    #[test]
    fn test_broadcast_strides_broadcast_error_on_axis_conflict() {
        let err = broadcast_strides(&[2, 3], &[3, 1], &[4, 3]).expect_err("incompatible strides");
        match err {
            XenonError::BroadcastError {
                operation,
                lhs_shape,
                rhs_shape,
                attempted_target_shape,
                axis,
            } => {
                assert_eq!(operation.as_ref(), "broadcast_strides");
                assert_eq!(lhs_shape, vec![2, 3]);
                assert_eq!(rhs_shape, vec![4, 3]);
                assert_eq!(attempted_target_shape, Some(vec![4, 3]));
                assert_eq!(axis, Some(0));
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    /// Tests that a non-leading (trailing) axis conflict reports the correct
    /// axis index.
    #[test]
    fn test_broadcast_strides_broadcast_error_non_leading_axis() {
        let err = broadcast_strides(&[2, 3], &[3, 1], &[2, 5]).expect_err("axis-1 conflict");
        match err {
            XenonError::BroadcastError {
                operation,
                lhs_shape,
                rhs_shape,
                attempted_target_shape,
                axis,
            } => {
                assert_eq!(operation.as_ref(), "broadcast_strides");
                assert_eq!(lhs_shape, vec![2, 3]);
                assert_eq!(rhs_shape, vec![2, 5]);
                assert_eq!(attempted_target_shape, Some(vec![2, 5]));
                // Axis 0 matches (2 == 2); conflict surfaces at trailing axis 1.
                assert_eq!(axis, Some(1));
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }

    /// Tests the equal-dim branch: when source and target axes are both 0
    /// (empty), the original stride is preserved rather than being forced to 0.
    #[test]
    fn test_broadcast_strides_equal_empty_axis_keeps_stride() {
        let s = broadcast_strides(&[0, 3], &[3, 1], &[0, 3]).expect("equal empty axis");
        assert_eq!(s, vec![3, 1]);
    }

    /// Tests that `broadcast_strides` returns a `BroadcastError` when the
    /// original shape has more axes than the target (rank excess).
    #[test]
    fn test_broadcast_strides_broadcast_error_on_rank_excess() {
        // orig rank 3 > target rank 2 → BroadcastError (right-align impossible).
        let err =
            broadcast_strides(&[2, 3, 4], &[12, 4, 1], &[3, 4]).expect_err("incompatible strides");
        assert!(matches!(err, XenonError::BroadcastError { .. }));
    }

    /// Tests that `broadcast_error` produces a `BroadcastError` with all five
    /// fields populated correctly.
    #[test]
    fn test_broadcast_error_has_complete_structured_fields() {
        let err = broadcast_error("broadcast_shape", &[2, 3], &[4, 3], None, 0);
        match err {
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
            },
            other => panic!("expected BroadcastError, got {:?}", other),
        }
    }
}
