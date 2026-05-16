pub mod axes;
pub mod dynamic;
pub mod fixed;
pub mod into;

use std::fmt::Debug;

use crate::error::XenonError;
use crate::private::Sealed;

/// Trait for array dimension types.
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// Implementations exist for `Ix0`, `Ix1`, ..., `Ix6` (static dimensions)
/// and `IxDyn` (dynamic dimension).
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix3};
/// let dim = Ix3(2, 3, 4);
/// assert_eq!(dim.ndim(), 3);
/// assert_eq!(dim.slice(), &[2, 3, 4]);
/// assert_eq!(dim.checked_size().unwrap(), 24);
/// ```
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

// Public re-exports — the canonical access path for dimension types.
pub use fixed::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6};
pub use dynamic::IxDyn;
pub use axes::Axis;
pub use into::IntoDimension;

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

    /// All 8 dimension types satisfy the `Dimension` trait bound
    /// (which requires `Sealed`). Compile-time check.
    #[test]
    fn test_sealed_dimension_bound_satisfied() {
        fn assert_dimension<D: Dimension>(_: D) {}
        assert_dimension(Ix0);
        assert_dimension(Ix1(1));
        assert_dimension(Ix2(2, 3));
        assert_dimension(Ix3(2, 3, 4));
        assert_dimension(Ix4(1, 2, 3, 4));
        assert_dimension(Ix5(1, 2, 3, 4, 5));
        assert_dimension(Ix6(1, 2, 3, 4, 5, 6));
        assert_dimension(IxDyn::from_slice(&[1, 2]));
    }

    /// Public re-exports are reachable via `crate::dimension::*`.
    #[test]
    fn test_public_exports_reachable() {
        let _: Ix0 = Ix0;
        let _: Ix1 = Ix1(1);
        let _: IxDyn = IxDyn::new();
        let _: Axis = Axis::new(0);
        // IntoDimension trait reachable: tuple → Ix3.
        let _: Ix3 = (1, 2, 3).into_dimension();
    }

    /// §8.2: Mirrors the canonical doc examples on Ix2 / Axis /
    /// IntoDimension to ensure they exercise real public API paths.
    #[test]
    fn test_public_doc_examples_execute() {
        // Ix2 example
        let dim = Ix2(10, 20);
        assert_eq!(dim.ndim(), 2);
        assert_eq!(dim.slice(), &[10, 20]);
        assert_eq!(dim.checked_size(), Ok(200));
        assert_eq!(dim[0], 10);

        // Axis example
        let ax = Axis::new(0);
        assert!(ax.is_first());

        // IntoDimension example
        let d3: Ix3 = (2, 3, 4).into_dimension();
        assert_eq!(d3.slice(), &[2, 3, 4]);
    }
}
