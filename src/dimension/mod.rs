pub mod axes;
pub mod dynamic;
pub mod fixed;
pub mod into;

use std::fmt::Debug;

use crate::dimension::axes::Axis;
use crate::error::XenonError;
use crate::private::Sealed;

/// Trait for array dimension types.
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// Implementations exist for `Ix0`, `Ix1`, ..., `Ix6` (static dimensions)
/// and `IxDyn` (dynamic dimension).
pub trait Dimension: Sealed + Clone + PartialEq + Eq + Debug + Send + Sync + 'static {
    /// Maximum number of dimensions for static dimension types.
    /// `Some(N)` for static dimensions (Ix0..Ix6), `None` for IxDyn.
    const NDIM: Option<usize>;

    /// Number of dimensions (rank).
    fn ndim(&self) -> usize;

    /// Shape as a slice of axis lengths.
    fn slice(&self) -> &[usize];

    /// Total number of elements, checking for overflow.
    /// Returns `XenonError::InvalidShape { kind: ProductOverflow }` on overflow.
    fn checked_size(&self) -> Result<usize, XenonError>;

    /// Validates dimension metadata without consuming the element count.
    /// The default contract is equivalent to `self.checked_size().map(|_| ())`.
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
    }

    /// Create a dimension from a slice.
    /// Returns `XenonError::DimensionMismatch` on rank mismatch.
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError>
    where
        Self: Sized;

    /// Returns the axis length at the given index.
    fn axis(&self, axis: Axis) -> Result<usize, XenonError> {
        self.slice()
            .get(axis.0)
            .copied()
            .ok_or(XenonError::InvalidAxis {
                operation: "Dimension::axis".into(),
                axis: axis.index(),
                ndim: self.ndim(),
                shape: self.slice().into(),
            })
    }
}

/// Maximum number of dimensions representable on this platform.
pub const MAX_DIMENSION: usize = usize::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time path check: ensure all four submodules are reachable
    /// via the public dimension module path. If any submodule is renamed
    /// or removed, these `use` statements fail to compile.
    #[test]
    fn test_dimension_submodules_reachable() {
        #[allow(unused_imports)]
        use crate::dimension::axes;
        #[allow(unused_imports)]
        use crate::dimension::dynamic;
        #[allow(unused_imports)]
        use crate::dimension::fixed;
        #[allow(unused_imports)]
        use crate::dimension::into;
    }

    /// §8.2: MAX_DIMENSION constant value.
    #[test]
    fn test_max_dimension_is_usize_max() {
        assert_eq!(MAX_DIMENSION, usize::MAX);
    }

    /// Compile-time check: verify `Dimension` trait bound is implementable.
    /// This function is never called but ensures the trait can be used as a
    /// generic bound — if any required supertrait is missing, this will fail
    /// to compile in downstream tasks (W3T4+) that instantiate it.
    #[allow(dead_code)]
    fn assert_dimension_bounds<D: Dimension>() {}
}
