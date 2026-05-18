//! Property-based tests for the dimension module.
//!
//! Covers `02-dimension.md` §8.4 invariants using `proptest`. The actual
//! property assertions exercise random shapes generated within bounds that
//! avoid `usize` overflow during product computation.

use proptest::prelude::*;
use xenon::dimension::{Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use xenon::error::InvalidShapeKind;
use xenon::error::XenonError;

/// §8.4: roundtrip invariant for Ix0 (trivial — Ix0 is a ZST).
#[test]
fn test_ix0_roundtrip() {
    let dim = Ix0;
    assert_eq!(Ix0::try_from_dyn(dim.into_dyn()), Ok(Ix0));
}

proptest! {
    /// §8.4: roundtrip invariant for Ix1.
    #[test]
    fn test_ix1_roundtrip(a in 0usize..1024) {
        let dim = Ix1(a);
        prop_assert_eq!(Ix1::try_from_dyn(dim.into_dyn()), Ok(dim));
    }

    /// §8.4: roundtrip invariant for Ix2.
    #[test]
    fn test_ix2_roundtrip(a in 0usize..1024, b in 0usize..1024) {
        let dim = Ix2(a, b);
        prop_assert_eq!(Ix2::try_from_dyn(dim.into_dyn()), Ok(dim));
    }

    /// §8.4: roundtrip invariant for Ix3.
    #[test]
    fn test_ix3_roundtrip(a in 0usize..512, b in 0usize..512, c in 0usize..512) {
        let dim = Ix3(a, b, c);
        prop_assert_eq!(Ix3::try_from_dyn(dim.into_dyn()), Ok(dim));
    }

    /// §8.4: roundtrip invariant for Ix4.
    #[test]
    fn test_ix4_roundtrip(
        a in 0usize..256, b in 0usize..256,
        c in 0usize..256, d in 0usize..256,
    ) {
        let dim = Ix4(a, b, c, d);
        prop_assert_eq!(Ix4::try_from_dyn(dim.into_dyn()), Ok(dim));
    }

    /// §8.4: roundtrip invariant for Ix5.
    #[test]
    fn test_ix5_roundtrip(
        a in 0usize..128, b in 0usize..128, c in 0usize..128,
        d in 0usize..128, e in 0usize..128,
    ) {
        let dim = Ix5(a, b, c, d, e);
        prop_assert_eq!(Ix5::try_from_dyn(dim.into_dyn()), Ok(dim));
    }

    /// §8.4: roundtrip invariant for Ix6.
    #[test]
    fn test_ix6_roundtrip(
        a in 0usize..64, b in 0usize..64, c in 0usize..64,
        d in 0usize..64, e in 0usize..64, f in 0usize..64,
    ) {
        let dim = Ix6(a, b, c, d, e, f);
        prop_assert_eq!(Ix6::try_from_dyn(dim.into_dyn()), Ok(dim));
    }
}

proptest! {
    /// §8.4: `dim.checked_size()? == dim.slice().iter().product()` for
    /// non-overflowing shapes.
    ///
    /// Strategy generates IxDyn shapes with rank ∈ [0, 6] and axis values that
    /// guarantee the product fits in `usize`.
    #[test]
    fn test_checked_size_equals_slice_product(
        shape in proptest::collection::vec(0usize..16, 0..7)
    ) {
        let dim = IxDyn::from_vec(shape.clone());
        let expected: Option<usize> =
            shape.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
        match (dim.checked_size(), expected) {
            (Ok(actual), Some(exp)) => prop_assert_eq!(actual, exp),
            (Err(XenonError::InvalidShape { .. }), None) => {
                // Strategy bound ensures product fits, so this branch should
                // not occur.  If reached, the test still passes (both report
                // overflow consistently).
            }
            other => prop_assert!(false, "size/product disagree: {:?}", other),
        }
    }
}

proptest! {
    /// Overflow invariant: if `checked_size()` returns Err, it must be
    /// `InvalidShape::ProductOverflow` with `offending_dim` filled.
    ///
    /// Strategy: place `usize::MAX` at a random axis to force overflow.
    #[test]
    fn test_overflow_includes_offending_dim(
        idx in 0usize..4,
        other in 2usize..16,
    ) {
        // Build a shape with `usize::MAX` at position `idx`, `other`
        // elsewhere.  Total rank 4; product at `idx` makes subsequent
        // axes overflow.
        let mut shape = vec![other; 4];
        shape[idx] = usize::MAX;
        let dim = IxDyn::from_vec(shape);
        match dim.checked_size() {
            Err(XenonError::InvalidShape {
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim,
                ..
            }) => {
                prop_assert!(
                    offending_dim.is_some(),
                    "offending_dim must be filled on overflow, got None"
                );
                let off = offending_dim.expect("offending_dim must be Some");
                prop_assert!(
                    off >= idx,
                    "offending_dim {} must be >= injection point {}",
                    off,
                    idx
                );
            }
            other_result => {
                // No overflow if MAX is at the last axis multiplied by 1,
                // but strategy ensures `other >= 2`, so overflow is always
                // triggered.
                prop_assert!(
                    false,
                    "expected ProductOverflow, got {:?}",
                    other_result
                );
            }
        }
    }
}
