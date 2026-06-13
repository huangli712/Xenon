//! Dimension traits: [`Dimension`], [`Reverse`], [`RemoveAxis`].

use std::borrow::Cow;
use std::fmt::Debug;

use super::axes::Axis;
use crate::private::Sealed;
use crate::error::{InvalidShapeKind, XenonError};

/// Maximum number of dimensions representable on this platform.
pub const MAX_DIMENSION: usize = usize::MAX;

/// Trait for array dimension types.
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// Implementations exist for `Ix0`, `Ix1`, ..., `Ix6` (static dimensions)
/// and `IxDyn` (dynamic dimension).
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
/// External crates may name it in `where` clauses or trait bounds, but
/// adding new implementations is intentionally not supported.
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
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidShape { kind: InvalidShapeKind::ProductOverflow }`
    /// when the cumulative product of the per-axis lengths overflows `usize`.
    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = self.slice();
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc
                .checked_mul(dim)
                .ok_or_else(|| XenonError::InvalidShape {
                    operation: Cow::Borrowed("Dimension::checked_size"),
                    shape: dims.to_vec(),
                    kind: InvalidShapeKind::ProductOverflow,
                    offending_dim: Some(axis),
                })?;
        }
        Ok(acc)
    }

    /// Validates dimension metadata without consuming the element count.
    /// The default contract is equivalent to `self.checked_size().map(|_| ())`.
    ///
    /// # Errors
    ///
    /// Forwards every error from [`Self::checked_size`] — currently only
    /// `XenonError::InvalidShape { kind: InvalidShapeKind::ProductOverflow }`
    /// when the cumulative axis-length product overflows `usize`.
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
    }

    /// Create a dimension from a slice.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch` when `slice.len()` does not
    /// match the static rank of `Self` (e.g. `Ix3::try_from_slice(&[1, 2])`).
    /// For `IxDyn`, any slice length is accepted.
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError>
    where
        Self: Sized;

    /// Returns the axis length at the given index.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.0 >= self.ndim()`.
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

/// Sealed trait for reversing the axis order of a dimension.
///
/// Every concrete `D` Xenon supports (Ix0..Ix6, IxDyn) implements
/// `Reverse` returning `Self`, so the bound is satisfied for all
/// supported dimensions. The trait is `pub` so that public API
/// signatures can name `D: Reverse`, but sealed so that external
/// crates cannot add their own implementations.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait Reverse: Dimension + Sealed {
    /// Returns a new dimension with the axis order reversed.
    /// Preserves the static rank (e.g., `Ix2 → Ix2`).
    fn reverse(self) -> Self;
}

/// Sealed trait for removing one axis from a dimension, producing a
/// dimension of one rank lower (`D::Smaller`).
///
/// Used by `AxisIter` / `AxisIterMut` to describe the reduced-dimension
/// subview type yielded by each step. The trait is `pub` so that public
/// API signatures can name `D: RemoveAxis`, but sealed so that external
/// crates cannot add their own implementations.
///
/// For Ix0 the operation is always a runtime-recoverable error
/// (`XenonError::InvalidAxis`); `Ix0` still implements the trait with
/// `Smaller = Ix0` so type-level APIs compile, but `remove_axis` fails
/// at runtime.
pub trait RemoveAxis: Dimension + Sealed {
    /// The dimension type after removing one axis.
    type Smaller: Dimension;

    /// Remove the axis at the given index.
    ///
    /// Returns `(Smaller_dim, removed_axis_len)` on success.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidAxis` when `axis.0 >= self.ndim()`.
    /// For `Ix0` (rank 0), every `axis` is invalid so this always errors.
    fn remove_axis(&self, axis: Axis) -> Result<(Self::Smaller, usize), XenonError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
    use crate::dimension::IntoDimension;

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

    /// `MAX_DIMENSION` equals `usize::MAX`.
    #[test]
    fn test_max_dimension_is_usize_max() {
        assert_eq!(MAX_DIMENSION, usize::MAX);
    }

    /// Public re-exports are reachable via `crate::dimension::*`.
    #[test]
    fn test_public_exports_reachable() {
        let _: Ix0 = Ix0;
        let _: Ix1 = Ix1(1);
        let _: IxDyn = IxDyn::new();
        let _: Axis = Axis::new(0);
        let _: Ix3 = (1, 2, 3).into_dimension();
    }

    /// Canonical doc examples on Ix2 / Axis / IntoDimension.
    #[test]
    fn test_public_doc_examples_execute() {
        let dim = Ix2(10, 20);
        assert_eq!(dim.ndim(), 2);
        assert_eq!(dim.slice(), &[10, 20]);
        assert_eq!(dim.checked_size(), Ok(200));
        assert_eq!(dim[0], 10);

        let ax = Axis::new(0);
        assert!(ax.is_first());

        let d3: Ix3 = (2, 3, 4).into_dimension();
        assert_eq!(d3.slice(), &[2, 3, 4]);
    }
}
