//! Element-wise clip operation.
//!
//! Provides `clip()` as an inherent method on [`TensorBase`].
//! See `docs/design/20-utility.md` §5.1 / §6.1 / §6.4.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::element::OrderedCompareElement;
use crate::error::{InvalidArgumentKind, XenonError};
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Validate that `min <= max`; reject NaN bounds.
///
/// Returns `Err(XenonError::InvalidArgument)` when `min > max` or either
/// bound is `NaN` (for floating-point types).
fn validate_clip_bounds<A>(min: &A, max: &A) -> Result<(), XenonError>
where
    A: OrderedCompareElement,
{
    if min.partial_cmp(max).is_none() || min > max {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("clip"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("min/max"),
                constraint: Cow::Borrowed(
                    "min <= max; NaN bounds are invalid for floating-point inputs",
                ),
            },
        });
    }
    Ok(())
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: OrderedCompareElement + Clone,
{
    /// Clamp each logical element into `[min, max]`.
    ///
    /// Per `20-utility §5.1` / §6.4:
    /// - Bounds are validated **before** allocation.
    /// - `NaN` input values pass through unchanged (both `< min` and `> max`
    ///   are `false` under IEEE 754), matching NumPy `np.clip` semantics.
    /// - `NaN` *bounds* (or `min > max`) return `InvalidArgument`.
    ///
    /// Implementation note: this body uses the `iter()` + `Vec` +
    /// `from_shape_vec` path explicitly permitted by design §5.1
    /// ("may use MaybeUninit or equivalent internal uninitialized owned
    /// buffer"). Once the
    /// `uninit_like` / `iter_uninit_mut` / `assume_init` primitives land in
    /// a future wave, migrate to a single-pass MaybeUninit write per §5.1
    /// §5.1 / §6.1 algorithm sketch.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is correct generic pattern"
    )]
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError> {
        validate_clip_bounds(&min, &max)?;
        // Iterate logical F-order (10-iterator §5.5) and collect into a
        // single owned buffer. `from_shape_vec` (18-construction §5.3)
        // produces a canonical F-order owned tensor (`21-type §5.5`).
        let data: Vec<A> = self
            .iter()
            .map(|src| {
                if *src < min {
                    min.clone()
                } else if *src > max {
                    max.clone()
                } else {
                    // NaN inputs land here (both comparisons return false
                    // under IEEE 754) and are cloned through unchanged —
                    // exactly the §6.4 row "NaN input -> NaN".
                    src.clone()
                }
            })
            .collect();
        // SAFETY of `expect`: `iter()` yields exactly `product(shape)`
        // elements (10-iterator §5.5), which `from_shape_vec` requires.
        Tensor::from_shape_vec(self.raw_dim(), data)
    }
}

// ── Unit tests (§8.2 / §7 T2) ──

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::tensor::{Tensor1, Tensor2};

    // §8.2 — test_clip_basic
    #[test]
    fn test_clip_basic() {
        let tensor = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    }

    // §8.2 — test_clip_no_change
    #[test]
    fn test_clip_no_change() {
        let tensor = Tensor1::from_shape_vec([3], vec![0.5, 1.0, 1.5])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.5, 1.0, 1.5]);
    }

    // §8.2 — test_clip_nan (NaN inputs pass through unchanged, §6.4)
    #[test]
    fn test_clip_nan() {
        let tensor = Tensor1::from_shape_vec([3], vec![1.0_f64, f64::NAN, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 4.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
    }

    // §8.2 — test_clip_nan_bound (NaN as min or max → InvalidArgument)
    #[test]
    fn test_clip_nan_bound() {
        let tensor =
            Tensor1::from_shape_vec([1], vec![1.0_f64]).expect("from_shape_vec matching shape");
        assert!(matches!(
            tensor.clip(f64::NAN, 2.0),
            Err(XenonError::InvalidArgument { .. })
        ));
        assert!(matches!(
            tensor.clip(0.0, f64::NAN),
            Err(XenonError::InvalidArgument { .. })
        ));
    }

    // §8.2 — test_clip_integers
    #[test]
    fn test_clip_integers() {
        let tensor = Tensor1::from_shape_vec([4], vec![-5_i32, 0, 5, 10])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0, 7).expect("valid clip bounds");
        let values: Vec<i32> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0, 0, 5, 7]);
    }

    // §8.2 — test_clip_non_contiguous
    //
    // Construction is in F-order. `from_shape_vec([2, 3], [1, 4, 2, 5, 3, 6])`
    // produces the 2×3 matrix
    //     [ 1 2 3 ]
    //     [ 4 5 6 ]
    // (column-major: column 0 = [1, 4], column 1 = [2, 5], column 2 = [3, 6]).
    // After `transpose()` we get the 3×2 matrix
    //     [ 1 4 ]
    //     [ 2 5 ]
    //     [ 3 6 ]
    // and clip(2.0, 5.0) yields
    //     [ 2 4 ]
    //     [ 2 5 ]
    //     [ 3 5 ]
    // F-order to_vec of that result (column 0 then column 1) is
    //     [2.0, 2.0, 3.0, 4.0, 5.0, 5.0].
    #[test]
    fn test_clip_non_contiguous() {
        let tensor = Tensor2::from_shape_vec([2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        let clipped = transposed.clip(2.0, 5.0).expect("valid clip bounds");
        assert_eq!(clipped.shape(), &[3, 2]);
        // Per-element assertions are immune to to_vec()'s order question:
        assert_eq!(*clipped.get(&[0, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[0, 1]).expect("valid index"), 4.0);
        assert_eq!(*clipped.get(&[1, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[1, 1]).expect("valid index"), 5.0);
        assert_eq!(*clipped.get(&[2, 0]).expect("valid index"), 3.0);
        assert_eq!(*clipped.get(&[2, 1]).expect("valid index"), 5.0);
        // F-order iter consistency (column-major), kept as a regression
        // guard against any future iter() semantics drift:
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![2.0, 2.0, 3.0, 4.0, 5.0, 5.0]);
    }
}
