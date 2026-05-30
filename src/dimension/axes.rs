//! Axis types.

/// Axis marker type. Provides type safety over raw `usize`.
///
/// # Examples
///
/// ```
/// use xenon::dimension::Axis;
/// let ax = Axis::new(0);
/// assert!(ax.is_first());
/// assert_eq!(ax.index(), 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Axis(pub usize);

impl Axis {
    /// Create a new Axis with the given index.
    #[inline]
    pub fn new(axis: usize) -> Self {
        Axis(axis)
    }

    /// Return the raw index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }

    /// Returns the next axis, or `None` if `self.0 == usize::MAX`.
    #[inline]
    pub fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Axis)
    }

    /// Returns the previous axis, or `None` if already at index 0.
    #[inline]
    pub fn prev(self) -> Option<Self> {
        self.0.checked_sub(1).map(Axis)
    }

    /// True if this is the first axis (index 0).
    #[inline]
    pub fn is_first(self) -> bool {
        self.0 == 0
    }

    /// True if this axis is the last axis in a dimension with `ndim` axes.
    /// Returns `false` when `ndim == 0` (no axes exist).
    #[inline]
    pub fn is_last(self, ndim: usize) -> bool {
        ndim > 0 && self.0 == ndim - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Normal next/prev navigation.
    #[test]
    fn test_axis_next_prev() {
        let axis = Axis::new(2);
        assert_eq!(axis.index(), 2);
        assert_eq!(axis.next().expect("next of 2").index(), 3);
        assert_eq!(axis.prev().expect("prev of 2").index(), 1);
        assert_eq!(Axis::new(0).prev(), None);
    }

    /// `next()` overflow returns `None`.
    #[test]
    fn test_axis_next_overflow() {
        assert_eq!(Axis::new(usize::MAX).next(), None);
    }

    /// Boundary semantics: `is_first`, `is_last`, and the `ndim == 0` edge case.
    #[test]
    fn test_axis_is_first_last() {
        assert!(Axis::new(0).is_first());
        assert!(!Axis::new(1).is_first());
        assert!(Axis::new(2).is_last(3));
        assert!(!Axis::new(0).is_last(0));
        assert!(!Axis::new(1).is_last(3));
    }
}
