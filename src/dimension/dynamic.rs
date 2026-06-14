//! Dynamic dimension types.

use std::borrow::Cow;

use crate::error::XenonError;
use crate::dimension::{Axis, Dimension, RemoveAxis, Reverse};

/// Dynamic dimension type. Dimension count determined at runtime.
/// Dynamic rank is bounded only by `usize` representability and
/// available memory.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, IxDyn};
/// let dim = IxDyn::from_slice(&[2, 3, 4]);
/// assert_eq!(dim.ndim(), 3);
/// assert_eq!(dim.slice(), &[2, 3, 4]);
/// assert_eq!(dim.checked_size(), Ok(24));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct IxDyn {
    dims: Vec<usize>,
}

impl IxDyn {
    /// Creates an empty (0-dimensional) dynamic dimension.
    #[inline]
    pub fn new() -> Self {
        IxDyn { dims: Vec::new() }
    }

    /// Creates from a slice (clones into an owned Vec).
    #[inline]
    pub fn from_slice(slice: &[usize]) -> Self {
        IxDyn {
            dims: slice.to_vec(),
        }
    }

    /// Creates from a Vec (consumes ownership, zero-copy).
    #[inline]
    pub fn from_vec(dims: Vec<usize>) -> Self {
        IxDyn { dims }
    }

    /// Creates with all `ndim` axes set to the given `value`.
    #[inline]
    pub fn from_element(value: usize, ndim: usize) -> Self {
        IxDyn {
            dims: vec![value; ndim],
        }
    }

    /// Creates with all `ndim` axes set to 1.
    #[inline]
    pub fn ones(ndim: usize) -> Self {
        Self::from_element(1, ndim)
    }

    /// Creates with all `ndim` axes set to 0.
    #[inline]
    pub fn zeros(ndim: usize) -> Self {
        Self::from_element(0, ndim)
    }

    /// Consumes and returns the inner Vec.
    #[inline]
    pub fn into_vec(self) -> Vec<usize> {
        self.dims
    }
}

impl Dimension for IxDyn {
    /// Maximum number of static dimensions: `None` for dynamic dimension.
    const NDIM: Option<usize> = None;

    /// Returns the number of axes (rank).
    #[inline]
    fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the axis lengths as a slice.
    #[inline]
    fn slice(&self) -> &[usize] {
        &self.dims
    }

    /// Builds from a slice; any rank is accepted.
    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        Ok(IxDyn::from_slice(slice))
    }

    /// Identity conversion: returns `self` unchanged (zero-copy).
    #[inline]
    fn into_dyn(self) -> IxDyn {
        self
    }

    /// Identity conversion. Always succeeds; no rank check is needed.
    ///
    /// # Errors
    ///
    /// Infallible: always returns `Ok(dyn_dim)`. The fallible signature
    /// mirrors `IxN::try_from_dyn` (Ix0..Ix6) so generic callers share one
    /// uniform conversion API across every dimension type.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        Ok(dyn_dim)
    }
}

impl Reverse for IxDyn {
    /// Reverses the axis order in-place.
    fn reverse(self) -> Self {
        let mut dims = self.dims;
        dims.reverse();
        IxDyn { dims }
    }
}

impl RemoveAxis for IxDyn {
    /// The reduced-rank dimension type.
    type Smaller = IxDyn;

    /// Removes one axis at the given index, returning the smaller
    /// dimension and the removed axis length.
    fn remove_axis(&self, axis: Axis) -> Result<(Self::Smaller, usize), XenonError> {
        if axis.0 >= self.ndim() {
            return Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("IxDyn::remove_axis"),
                axis: axis.0,
                ndim: self.ndim(),
                shape: self.slice().to_vec(),
            });
        }
        let removed_len = self.dims[axis.0];
        let mut dims = self.dims.clone();
        dims.remove(axis.0);
        Ok((IxDyn { dims }, removed_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic construction from slice.
    #[test]
    fn test_ixdyn_from_slice() {
        let dim = IxDyn::from_slice(&[2, 3, 4]);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.slice(), &[2, 3, 4]);
        // Empty slice → 0-dimensional IxDyn (size 1).
        assert_eq!(IxDyn::from_slice(&[]).slice(), &[]);
    }

    /// `checked_size` returns `Result<usize, XenonError>`.
    #[test]
    fn test_ixdyn_size() {
        let dim = IxDyn::from_slice(&[2, 3, 4]);
        assert_eq!(dim.checked_size(), Ok(24));
        // 0-rank IxDyn (no axes) has size 1.
        assert_eq!(IxDyn::new().checked_size(), Ok(1));
        // Zero-length axis: total size is 0, not an error.
        assert_eq!(IxDyn::from_slice(&[3, 0, 5]).checked_size(), Ok(0));
    }

    /// Constructor coverage: from_vec / from_element / ones / zeros / into_vec.
    #[test]
    fn test_ixdyn_constructors() {
        assert_eq!(IxDyn::from_vec(vec![1, 2, 3]).slice(), &[1, 2, 3]);
        assert_eq!(IxDyn::from_element(5, 3).slice(), &[5, 5, 5]);
        assert_eq!(IxDyn::ones(2).slice(), &[1, 1]);
        assert_eq!(IxDyn::zeros(2).slice(), &[0, 0]);
        assert_eq!(IxDyn::from_slice(&[1, 2, 3]).into_vec(), vec![1, 2, 3]);
    }

    /// Overflow path: checked_size returns Err with offending_dim filled.
    #[test]
    fn test_ixdyn_size_overflow() {
        let dim = IxDyn::from_slice(&[usize::MAX, 2]);
        match dim.checked_size() {
            Err(XenonError::InvalidShape { offending_dim, .. }) => {
                assert_eq!(offending_dim, Some(1));
            },
            other => panic!("expected InvalidShape with offending_dim, got {:?}", other),
        }
    }

    /// `IxDyn::ones(0)` — zero-rank dynamic dimension is valid.
    #[test]
    fn test_ixdyn_ones_zero_rank() {
        let dim = IxDyn::ones(0);
        assert_eq!(dim.ndim(), 0);
        assert_eq!(dim.slice(), &[] as &[usize]);
        assert_eq!(dim.checked_size(), Ok(1));
    }

    /// Large dynamic dim with overflow.
    #[test]
    fn test_ixdyn_large_dim_overflow() {
        let dim = IxDyn::from_slice(&[usize::MAX, 2, 2]);
        match dim.checked_size() {
            Err(XenonError::InvalidShape {
                offending_dim: Some(idx),
                ..
            }) => {
                assert!(
                    idx >= 1,
                    "overflow should be detected at or after axis 1, got {}",
                    idx
                );
            },
            other => panic!("expected InvalidShape with offending_dim, got {:?}", other),
        }
    }

    /// `reverse()` reverses axis order for IxDyn.
    #[test]
    fn test_ixdyn_reverse() {
        let dim = IxDyn::from_slice(&[1, 2, 3, 4]);
        assert_eq!(dim.reverse().slice(), &[4, 3, 2, 1]);
        // Empty and single-element edge cases.
        assert_eq!(IxDyn::new().reverse().slice(), &[] as &[usize]);
        assert_eq!(IxDyn::from_slice(&[5]).reverse().slice(), &[5]);
    }

    /// `remove_axis` for IxDyn: success at each valid index.
    #[test]
    fn test_ixdyn_remove_axis() {
        let dim = IxDyn::from_slice(&[2, 3, 4]);
        assert_eq!(
            dim.remove_axis(Axis::new(0)),
            Ok((IxDyn::from_slice(&[3, 4]), 2))
        );
        assert_eq!(
            dim.remove_axis(Axis::new(1)),
            Ok((IxDyn::from_slice(&[2, 4]), 3))
        );
        assert_eq!(
            dim.remove_axis(Axis::new(2)),
            Ok((IxDyn::from_slice(&[2, 3]), 4))
        );
    }

    /// `remove_axis` for IxDyn: OOB and empty-dim return InvalidAxis.
    #[test]
    fn test_ixdyn_remove_axis_oob() {
        let dim = IxDyn::from_slice(&[2, 3, 4]);
        assert!(matches!(
            dim.remove_axis(Axis::new(3)),
            Err(XenonError::InvalidAxis { .. })
        ));
        // Empty IxDyn: every axis index is invalid.
        assert!(matches!(
            IxDyn::new().remove_axis(Axis::new(0)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }
}
