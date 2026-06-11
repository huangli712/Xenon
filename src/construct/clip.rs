//! Value clamping for tensors.
//!
//! Provides `TensorBase::clip()`: a method that returns a new owned tensor
//! with each element clamped between `min` and `max`.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::element::OrderedCompareElement;
use crate::error::{InvalidArgumentKind, XenonError};
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

/// Validate that `min <= max`; reject NaN bounds.
///
/// Returns [`XenonError::InvalidArgument`] when `min > max` or either
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
    /// Bounds are validated before allocation. `NaN` input values pass
    /// through unchanged (both `< min` and `> max` are `false` under
    /// IEEE 754), matching numpy `np.clip` semantics. `NaN` bounds or
    /// `min > max` return [`XenonError::InvalidArgument`].
    ///
    /// # Errors
    ///
    /// Returns [`XenonError::InvalidArgument`] when either bound is `NaN`
    /// or `min > max`.
    #[expect(clippy::clone_on_copy)]
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError> {
        validate_clip_bounds(&min, &max)?;
        let data: Vec<A> = self
            .iter()
            .map(|src| {
                if *src < min {
                    min.clone()
                } else if *src > max {
                    max.clone()
                } else {
                    src.clone()
                }
            })
            .collect();
        Tensor::from_shape_vec(self.raw_dim(), data)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::tensor::{Tensor1, Tensor2};

    /// Values outside [min, max] are clamped; values within pass through.
    #[test]
    fn test_clip_basic() {
        let tensor = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    }

    /// All values within bounds: output equals input.
    #[test]
    fn test_clip_no_change() {
        let tensor = Tensor1::from_shape_vec([3], vec![0.5, 1.0, 1.5])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.5, 1.0, 1.5]);
    }

    /// NaN input values pass through unchanged.
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

    /// NaN as min or max bound returns [`XenonError::InvalidArgument`].
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

    /// clip works correctly on integer element types.
    #[test]
    fn test_clip_integers() {
        let tensor = Tensor1::from_shape_vec([4], vec![-5_i32, 0, 5, 10])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0, 7).expect("valid clip bounds");
        let values: Vec<i32> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0, 0, 5, 7]);
    }

    /// clip works correctly on a transposed (non-contiguous) tensor.
    #[test]
    fn test_clip_non_contiguous() {
        let tensor = Tensor2::from_shape_vec([2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        let clipped = transposed.clip(2.0, 5.0).expect("valid clip bounds");
        assert_eq!(clipped.shape(), &[3, 2]);
        assert_eq!(*clipped.get(&[0, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[0, 1]).expect("valid index"), 4.0);
        assert_eq!(*clipped.get(&[1, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[1, 1]).expect("valid index"), 5.0);
        assert_eq!(*clipped.get(&[2, 0]).expect("valid index"), 3.0);
        assert_eq!(*clipped.get(&[2, 1]).expect("valid index"), 5.0);
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![2.0, 2.0, 3.0, 4.0, 5.0, 5.0]);
    }
}
