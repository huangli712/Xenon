//! Fixed-size dimension types.

use std::borrow::Cow;
use std::ops::Index;
use std::slice;

use crate::error::XenonError;
use crate::dimension::{Axis, Dimension, RemoveAxis, Reverse, IxDyn};

// --- Ix0 --------------------------------------------------------------------

/// Zero-dimensional index (scalar). Always has rank 0, size 1.
/// This type is a ZST (Zero-Sized Type); `size_of::<Ix0>() == 0`.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix0};
/// let dim = Ix0;
/// assert_eq!(dim.ndim(), 0);
/// assert_eq!(dim.slice(), &[] as &[usize]);
/// assert_eq!(dim.checked_size(), Ok(1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix0;

impl Dimension for Ix0 {
    const NDIM: Option<usize> = Some(0);

    #[inline]
    fn ndim(&self) -> usize {
        0
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        &[]
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        Ok(1)
    }

    #[inline]
    fn checked(&self) -> Result<(), XenonError> {
        Ok(())
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.is_empty() {
            Ok(Ix0)
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix0::try_from_slice"),
                expected: 0,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 0`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 0 {
            Ok(Ix0)
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix0::try_from_dyn"),
                expected: 0,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix0 {
    /// Identity: 0-dimensional.
    fn reverse(self) -> Self {
        self
    }
}

impl RemoveAxis for Ix0 {
    type Smaller = Ix0;
    
    /// Always errors (Ix0 has no axes).
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        Err(XenonError::InvalidAxis {
            operation: Cow::Borrowed("Ix0::remove_axis"),
            axis: axis.0,
            ndim: 0,
            shape: vec![],
        })
    }
}

impl From<()> for Ix0 {
    /// Converts the unit tuple into Ix0 (the rank-0 dimension).
    #[inline]
    fn from(_: ()) -> Self {
        Ix0
    }
}

/// Index-based access to axis length.
///
/// # Panics
///
/// Always panics: `Ix0` is rank-0 and has no axes, so every index is out
/// of bounds.
impl Index<usize> for Ix0 {
    type Output = usize;

    /// Always panics, as `Ix0` has no axes to index.
    ///
    /// # Panics
    ///
    /// Panics for every `index`.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        panic!("Ix0 index out of bounds: {index} (rank-0 has no axes)");
    }
}

// --- Ix1 --------------------------------------------------------------------

/// One-dimensional index.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix1};
/// let dim = Ix1(5);
/// assert_eq!(dim.ndim(), 1);
/// assert_eq!(dim.slice(), &[5]);
/// assert_eq!(dim[0], 5);
/// assert_eq!(dim.checked_size().unwrap(), 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix1(pub usize);

impl Dimension for Ix1 {
    const NDIM: Option<usize> = Some(1);

    #[inline]
    fn ndim(&self) -> usize {
        1
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        slice::from_ref(&self.0)
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        Ok(self.0)
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 1 {
            Ok(Ix1(slice[0]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix1::try_from_slice"),
                expected: 1,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 1`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 1 {
            let s = dyn_dim.slice();
            Ok(Ix1(s[0]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix1::try_from_dyn"),
                expected: 1,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix1 {
    /// Identity: single axis.
    fn reverse(self) -> Self {
        self
    }
}

impl RemoveAxis for Ix1 {
    type Smaller = Ix0;

    /// Removes axis 0, returning Ix0 and the length.
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        if axis.0 != 0 {
            return Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("Ix1::remove_axis"),
                axis: axis.0,
                ndim: 1,
                shape: self.slice().to_vec(),
            });
        }
        Ok((Ix0, self.0))
    }
}

/// Index-based access to axis length.
///
/// # Panics
///
/// Panics if `index != 0`, as `Ix1` has only one axis.
impl Index<usize> for Ix1 {
    type Output = usize;

    /// Returns the axis length at index 0.
    ///
    /// # Panics
    ///
    /// Panics if `index != 0`.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        assert_eq!(index, 0, "Ix1 index out of bounds");
        &self.0
    }
}

impl From<(usize,)> for Ix1 {
    /// Converts a 1-tuple into Ix1.
    #[inline]
    fn from(t: (usize,)) -> Self {
        Ix1(t.0)
    }
}

// --- Ix2 --------------------------------------------------------------------

/// Two-dimensional index.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix2};
/// let dim = Ix2(10, 20);
/// assert_eq!(dim.ndim(), 2);
/// assert_eq!(dim.slice(), &[10, 20]);
/// assert_eq!(dim.checked_size().unwrap(), 200);
/// assert_eq!(dim[0], 10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix2(pub usize, pub usize);

impl Dimension for Ix2 {
    const NDIM: Option<usize> = Some(2);

    #[inline]
    fn ndim(&self) -> usize {
        2
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: Ix2 uses #[repr(C)] and contains exactly two `usize`
        // fields in declaration order. Reinterpreting &Ix2 as a
        // contiguous `*const usize` slice of length 2 preserves
        // provenance, alignment, and size.
        unsafe { slice::from_raw_parts(self as *const Self as *const usize, 2) }
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 2 {
            Ok(Ix2(slice[0], slice[1]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix2::try_from_slice"),
                expected: 2,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 2`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 2 {
            let s = dyn_dim.slice();
            Ok(Ix2(s[0], s[1]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix2::try_from_dyn"),
                expected: 2,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix2 {
    /// Reverses axis order: `(a, b) → (b, a)`.
    fn reverse(self) -> Self {
        Ix2(self.1, self.0)
    }
}

impl RemoveAxis for Ix2 {
    type Smaller = Ix1;
    
    /// Removes the given axis, returning Ix1.
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        match axis.0 {
            0 => Ok((Ix1(self.1), self.0)),
            1 => Ok((Ix1(self.0), self.1)),
            _ => Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("Ix2::remove_axis"),
                axis: axis.0,
                ndim: 2,
                shape: self.slice().to_vec(),
            }),
        }
    }
}

/// Index-based access to axis lengths.
///
/// # Panics
///
/// Panics if `index` is not 0 or 1, as `Ix2` has only two axes.
impl Index<usize> for Ix2 {
    type Output = usize;

    /// Returns the axis length at the given index (0 or 1).
    ///
    /// # Panics
    ///
    /// Panics if `index` is not 0 or 1.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("Ix2 index out of bounds: {index}"),
        }
    }
}

impl From<(usize, usize)> for Ix2 {
    /// Converts a 2-tuple into Ix2.
    #[inline]
    fn from(t: (usize, usize)) -> Self {
        Ix2(t.0, t.1)
    }
}

// --- Ix3 --------------------------------------------------------------------

/// Three-dimensional index.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix3};
/// let dim = Ix3(2, 3, 4);
/// assert_eq!(dim.ndim(), 3);
/// assert_eq!(dim.checked_size().unwrap(), 24);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix3(pub usize, pub usize, pub usize);

impl Dimension for Ix3 {
    const NDIM: Option<usize> = Some(3);

    #[inline]
    fn ndim(&self) -> usize {
        3
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: Ix3 uses #[repr(C)] and contains exactly three `usize`
        // fields in declaration order. Reinterpreting &Ix3 as a
        // contiguous `*const usize` slice of length 3 is valid per
        // repr(C) layout guarantee.
        unsafe { slice::from_raw_parts(self as *const Self as *const usize, 3) }
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 3 {
            Ok(Ix3(slice[0], slice[1], slice[2]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix3::try_from_slice"),
                expected: 3,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 3`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 3 {
            let s = dyn_dim.slice();
            Ok(Ix3(s[0], s[1], s[2]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix3::try_from_dyn"),
                expected: 3,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix3 {
    /// Reverses axis order: `(a, b, c) → (c, b, a)`.
    fn reverse(self) -> Self {
        Ix3(self.2, self.1, self.0)
    }
}

impl RemoveAxis for Ix3 {
    type Smaller = Ix2;

    /// Removes the given axis, returning Ix2.
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        match axis.0 {
            0 => Ok((Ix2(self.1, self.2), self.0)),
            1 => Ok((Ix2(self.0, self.2), self.1)),
            2 => Ok((Ix2(self.0, self.1), self.2)),
            _ => Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("Ix3::remove_axis"),
                axis: axis.0,
                ndim: 3,
                shape: self.slice().to_vec(),
            }),
        }
    }
}

impl From<(usize, usize, usize)> for Ix3 {
    /// Converts a 3-tuple into Ix3.
    #[inline]
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        Ix3(a, b, c)
    }
}

/// Index-based access to axis lengths.
///
/// # Panics
///
/// Panics if `index >= 3`, as `Ix3` has only three axes.
impl Index<usize> for Ix3 {
    type Output = usize;

    /// Returns the axis length at the given index (0, 1, or 2).
    ///
    /// # Panics
    ///
    /// Panics if `index >= 3`.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("Ix3 index out of bounds: {index}"),
        }
    }
}

// --- Ix4 --------------------------------------------------------------------

/// Four-dimensional index.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix4};
/// let dim = Ix4(2, 3, 4, 5);
/// assert_eq!(dim.ndim(), 4);
/// assert_eq!(dim.checked_size().unwrap(), 120);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix4(pub usize, pub usize, pub usize, pub usize);

impl Dimension for Ix4 {
    const NDIM: Option<usize> = Some(4);

    #[inline]
    fn ndim(&self) -> usize {
        4
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: Ix4 uses #[repr(C)] and contains exactly four `usize`
        // fields laid out contiguously starting at `self.0`.
        unsafe { slice::from_raw_parts(self as *const Self as *const usize, 4) }
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 4 {
            Ok(Ix4(slice[0], slice[1], slice[2], slice[3]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix4::try_from_slice"),
                expected: 4,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 4`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 4 {
            let s = dyn_dim.slice();
            Ok(Ix4(s[0], s[1], s[2], s[3]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix4::try_from_dyn"),
                expected: 4,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix4 {
    /// Reverses axis order.
    fn reverse(self) -> Self {
        Ix4(self.3, self.2, self.1, self.0)
    }
}

impl RemoveAxis for Ix4 {
    type Smaller = Ix3;

    /// Removes the given axis, returning Ix3.
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        match axis.0 {
            0 => Ok((Ix3(self.1, self.2, self.3), self.0)),
            1 => Ok((Ix3(self.0, self.2, self.3), self.1)),
            2 => Ok((Ix3(self.0, self.1, self.3), self.2)),
            3 => Ok((Ix3(self.0, self.1, self.2), self.3)),
            _ => Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("Ix4::remove_axis"),
                axis: axis.0,
                ndim: 4,
                shape: self.slice().to_vec(),
            }),
        }
    }
}

impl From<(usize, usize, usize, usize)> for Ix4 {
    /// Converts a 4-tuple into Ix4.
    #[inline]
    fn from(t: (usize, usize, usize, usize)) -> Self {
        Ix4(t.0, t.1, t.2, t.3)
    }
}

/// Index-based access to axis lengths.
///
/// # Panics
///
/// Panics if `index >= 4`, as `Ix4` has only four axes.
impl Index<usize> for Ix4 {
    type Output = usize;

    /// Returns the axis length at the given index (0..=3).
    ///
    /// # Panics
    ///
    /// Panics if `index >= 4`.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            3 => &self.3,
            _ => panic!("Ix4 index out of bounds: {index}"),
        }
    }
}

// --- Ix5 --------------------------------------------------------------------

/// Five-dimensional dimension.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as
/// `&[usize; 5]` via pointer cast; this is only safe because `repr(C)`
/// guarantees the `usize` fields are laid out contiguously starting at
/// offset 0.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix5};
/// let dim = Ix5(2, 3, 4, 5, 6);
/// assert_eq!(dim.ndim(), 5);
/// assert_eq!(dim.checked_size().unwrap(), 720);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix5(pub usize, pub usize, pub usize, pub usize, pub usize);

impl Dimension for Ix5 {
    const NDIM: Option<usize> = Some(5);

    #[inline]
    fn ndim(&self) -> usize {
        5
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: `Ix5` uses `#[repr(C)]` and contains exactly five `usize`
        // fields in declaration order. Reinterpreting `&Ix5` as a contiguous
        // `*const usize` slice of length 5 preserves provenance, alignment,
        // and size.
        unsafe { slice::from_raw_parts(self as *const Self as *const usize, 5) }
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 5 {
            Ok(Ix5(slice[0], slice[1], slice[2], slice[3], slice[4]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix5::try_from_slice"),
                expected: 5,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 5`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 5 {
            let s = dyn_dim.slice();
            Ok(Ix5(s[0], s[1], s[2], s[3], s[4]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix5::try_from_dyn"),
                expected: 5,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix5 {
    /// Reverses axis order.
    fn reverse(self) -> Self {
        Ix5(self.4, self.3, self.2, self.1, self.0)
    }
}

impl RemoveAxis for Ix5 {
    type Smaller = Ix4;

    /// Removes the given axis, returning Ix4.
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        match axis.0 {
            0 => Ok((Ix4(self.1, self.2, self.3, self.4), self.0)),
            1 => Ok((Ix4(self.0, self.2, self.3, self.4), self.1)),
            2 => Ok((Ix4(self.0, self.1, self.3, self.4), self.2)),
            3 => Ok((Ix4(self.0, self.1, self.2, self.4), self.3)),
            4 => Ok((Ix4(self.0, self.1, self.2, self.3), self.4)),
            _ => Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("Ix5::remove_axis"),
                axis: axis.0,
                ndim: 5,
                shape: self.slice().to_vec(),
            }),
        }
    }
}

impl From<(usize, usize, usize, usize, usize)> for Ix5 {
    /// Converts a 5-tuple into Ix5.
    #[inline]
    fn from(t: (usize, usize, usize, usize, usize)) -> Self {
        Ix5(t.0, t.1, t.2, t.3, t.4)
    }
}

/// Index-based access to axis lengths.
///
/// # Panics
///
/// Panics if `index >= 5`, as `Ix5` has only five axes.
impl Index<usize> for Ix5 {
    type Output = usize;

    /// Returns the axis length at the given index (0..=4).
    ///
    /// # Panics
    ///
    /// Panics if `index >= 5`.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            3 => &self.3,
            4 => &self.4,
            _ => panic!("Ix5 index out of bounds: {index}"),
        }
    }
}

// --- Ix6 --------------------------------------------------------------------

/// Six-dimensional dimension.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as
/// `&[usize; 6]` via pointer cast; this is only safe because `repr(C)`
/// guarantees the `usize` fields are laid out contiguously starting at
/// offset 0.
///
/// # Examples
///
/// ```
/// use xenon::dimension::{Dimension, Ix6};
/// let dim = Ix6(1, 2, 3, 4, 5, 6);
/// assert_eq!(dim.ndim(), 6);
/// assert_eq!(dim.checked_size().unwrap(), 720);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix6(pub usize, pub usize, pub usize, pub usize, pub usize, pub usize);

impl Dimension for Ix6 {
    const NDIM: Option<usize> = Some(6);

    #[inline]
    fn ndim(&self) -> usize {
        6
    }

    #[inline]
    fn slice(&self) -> &[usize] {
        // SAFETY: Ix6 uses #[repr(C)] and contains exactly six `usize`
        // fields in declaration order. Reinterpreting &Ix6 as a contiguous
        // *const usize slice of length 6 preserves provenance, alignment,
        // and size.
        unsafe { slice::from_raw_parts(self as *const Self as *const usize, 6) }
    }

    #[inline]
    fn try_from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        if slice.len() == 6 {
            Ok(Ix6(
                slice[0], slice[1], slice[2], slice[3], slice[4], slice[5],
            ))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix6::try_from_slice"),
                expected: 6,
                actual: slice.len(),
            })
        }
    }

    /// Attempts to convert from a dynamic dimension.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch { operation, expected, actual }`
    /// when `dyn_dim.ndim() != 6`.
    #[inline]
    fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
        if dyn_dim.ndim() == 6 {
            let s = dyn_dim.slice();
            Ok(Ix6(s[0], s[1], s[2], s[3], s[4], s[5]))
        } else {
            Err(XenonError::DimensionMismatch {
                operation: Cow::Borrowed("Ix6::try_from_dyn"),
                expected: 6,
                actual: dyn_dim.ndim(),
            })
        }
    }
}

impl Reverse for Ix6 {
    /// Reverses axis order.
    fn reverse(self) -> Self {
        Ix6(self.5, self.4, self.3, self.2, self.1, self.0)
    }
}

impl RemoveAxis for Ix6 {
    type Smaller = Ix5;

    /// Removes the given axis, returning Ix5.
    fn remove_axis(
        &self,
        axis: Axis
    ) -> Result<(Self::Smaller, usize), XenonError> {
        match axis.0 {
            0 => Ok((Ix5(self.1, self.2, self.3, self.4, self.5), self.0)),
            1 => Ok((Ix5(self.0, self.2, self.3, self.4, self.5), self.1)),
            2 => Ok((Ix5(self.0, self.1, self.3, self.4, self.5), self.2)),
            3 => Ok((Ix5(self.0, self.1, self.2, self.4, self.5), self.3)),
            4 => Ok((Ix5(self.0, self.1, self.2, self.3, self.5), self.4)),
            5 => Ok((Ix5(self.0, self.1, self.2, self.3, self.4), self.5)),
            _ => Err(XenonError::InvalidAxis {
                operation: Cow::Borrowed("Ix6::remove_axis"),
                axis: axis.0,
                ndim: 6,
                shape: self.slice().to_vec(),
            }),
        }
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Ix6 {
    /// Converts a 6-tuple into Ix6.
    #[inline]
    fn from(t: (usize, usize, usize, usize, usize, usize)) -> Self {
        Ix6(t.0, t.1, t.2, t.3, t.4, t.5)
    }
}

/// Index-based access to axis lengths.
///
/// # Panics
///
/// Panics if `index >= 6`, as `Ix6` has only six axes.
impl Index<usize> for Ix6 {
    type Output = usize;

    /// Returns the axis length at the given index (0..=5).
    ///
    /// # Panics
    ///
    /// Panics if `index >= 6`.
    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            3 => &self.3,
            4 => &self.4,
            5 => &self.5,
            _ => panic!("Ix6 index out of bounds: {index}"),
        }
    }
}

// --- Layout Assertions ------------------------------------------------------

/// Compile-time layout assertions for unsafe pointer casts in `slice()`.
///
/// Verifies `size_of`, `align_of`, and field offsets of each `IxN` type
/// against the corresponding `[usize; N]` array.  If any `#[repr(C)]`
/// attribute is removed or field types are altered, these assertions fail
/// at compile time instead of silently introducing UB.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    // Ix0 is a ZST — no repr(C) / no pointer cast needed.

    // Ix1
    assert!(size_of::<Ix1>() == size_of::<[usize; 1]>());
    assert!(align_of::<Ix1>() == align_of::<[usize; 1]>());
    assert!(offset_of!(Ix1, 0) == 0);

    // Ix2
    assert!(size_of::<Ix2>() == size_of::<[usize; 2]>());
    assert!(align_of::<Ix2>() == align_of::<[usize; 2]>());
    assert!(offset_of!(Ix2, 0) == 0);
    assert!(offset_of!(Ix2, 1) == size_of::<usize>());

    // Ix3
    assert!(size_of::<Ix3>() == size_of::<[usize; 3]>());
    assert!(align_of::<Ix3>() == align_of::<[usize; 3]>());
    assert!(offset_of!(Ix3, 0) == 0);
    assert!(offset_of!(Ix3, 1) == size_of::<usize>());
    assert!(offset_of!(Ix3, 2) == 2 * size_of::<usize>());

    // Ix4
    assert!(size_of::<Ix4>() == size_of::<[usize; 4]>());
    assert!(align_of::<Ix4>() == align_of::<[usize; 4]>());
    assert!(offset_of!(Ix4, 0) == 0);
    assert!(offset_of!(Ix4, 1) == size_of::<usize>());
    assert!(offset_of!(Ix4, 2) == 2 * size_of::<usize>());
    assert!(offset_of!(Ix4, 3) == 3 * size_of::<usize>());

    // Ix5
    assert!(size_of::<Ix5>() == size_of::<[usize; 5]>());
    assert!(align_of::<Ix5>() == align_of::<[usize; 5]>());
    assert!(offset_of!(Ix5, 0) == 0);
    assert!(offset_of!(Ix5, 1) == size_of::<usize>());
    assert!(offset_of!(Ix5, 2) == 2 * size_of::<usize>());
    assert!(offset_of!(Ix5, 3) == 3 * size_of::<usize>());
    assert!(offset_of!(Ix5, 4) == 4 * size_of::<usize>());

    // Ix6
    assert!(size_of::<Ix6>() == size_of::<[usize; 6]>());
    assert!(align_of::<Ix6>() == align_of::<[usize; 6]>());
    assert!(offset_of!(Ix6, 0) == 0);
    assert!(offset_of!(Ix6, 1) == size_of::<usize>());
    assert!(offset_of!(Ix6, 2) == 2 * size_of::<usize>());
    assert!(offset_of!(Ix6, 3) == 3 * size_of::<usize>());
    assert!(offset_of!(Ix6, 4) == 4 * size_of::<usize>());
    assert!(offset_of!(Ix6, 5) == 5 * size_of::<usize>());
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};
    use crate::error::InvalidShapeKind;

    // --- Per-type tests -----------------------------------------------------

    /// Ix0 is a Zero-Sized Type.
    #[test]
    fn test_ix0_is_zst() {
        assert_eq!(size_of::<Ix0>(), 0);
    }

    /// Ix0 has rank 0.
    #[test]
    fn test_ix0_ndim_is_zero() {
        let dim = Ix0;
        assert_eq!(dim.ndim(), 0);
        assert_eq!(dim.slice(), &[] as &[usize]);
    }

    /// Scalar has exactly one element.
    #[test]
    fn test_ix0_size_is_one() {
        assert_eq!(Ix0.checked_size(), Ok(1));
        assert_eq!(Ix0.checked(), Ok(()));
    }

    /// Ix1 shape/rank/Index verification.
    #[test]
    fn test_ix1_slice() {
        assert_eq!(Ix1(7).ndim(), 1);
        assert_eq!(Ix1(7).slice(), &[7]);
        assert_eq!(Ix1(7).checked_size(), Ok(7));
        assert_eq!(Ix1(7)[0], 7);
    }

    /// Ix2(3, 4).slice() == &[3, 4]
    #[test]
    fn test_ix2_slice() {
        let dim = Ix2(3, 4);
        assert_eq!(dim.ndim(), 2);
        assert_eq!(dim.slice(), &[3, 4]);
        assert_eq!(dim.checked_size(), Ok(12));
        assert_eq!(dim[0], 3);
        assert_eq!(dim[1], 4);
    }

    /// Overflow returns Err with offending_dim.
    #[test]
    fn test_ix2_size_overflow() {
        let dim = Ix2(usize::MAX, 2);
        let err = dim.checked_size().expect_err("should overflow");
        match err {
            XenonError::InvalidShape {
                offending_dim: Some(1),
                ..
            } => {},
            _ => panic!("expected InvalidShape with offending_dim: \
                         Some(1), got {err:?}"),
        }
    }

    /// Ix3(2, 3, 4).slice() == &[2, 3, 4]
    #[test]
    fn test_ix3_slice() {
        let dim = Ix3(2, 3, 4);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.slice(), &[2, 3, 4]);
        assert!(dim.checked().is_ok());
    }

    /// Ix3(2, 3, 4).checked_size() == Ok(24)
    #[test]
    fn test_ix3_size_calculation() {
        let dim = Ix3::from((2, 3, 4));
        assert_eq!(dim, Ix3(2, 3, 4));
        assert_eq!(dim.checked_size().expect("product should fit"), 24);
    }

    /// Overflow: product overflow correctly reports offending axis.
    #[test]
    fn test_ix3_overflow_reports_offending_axis() {
        let dim = Ix3(usize::MAX, 2, 3);
        let err = dim.checked_size().expect_err("should overflow");
        match err {
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(1),
                ..
            } => {},
            _ => panic!("expected InvalidShape with ProductOverflow at axis 1"),
        }
    }

    /// Ix4 slice, rank, and size calculation.
    #[test]
    fn test_ix4_basic() {
        let dim = Ix4(2, 3, 4, 5);
        assert_eq!(dim.ndim(), 4);
        assert_eq!(dim.slice(), &[2, 3, 4, 5]);
        assert_eq!(dim.checked_size(), Ok(120));
    }

    /// Ix4 tuple conversion.
    #[test]
    fn test_ix4_from_tuple() {
        let dim = Ix4::from((2, 3, 4, 5));
        assert_eq!(dim, Ix4(2, 3, 4, 5));
    }

    /// Overflow reports offending dimension.
    #[test]
    fn test_ix4_overflow_offending_dim() {
        let large = Ix4(usize::MAX, 2, 3, 4);
        let err = large.checked_size().expect_err("should overflow");
        match err {
            XenonError::InvalidShape {
                offending_dim: Some(1),
                ..
            } => {},
            _ => panic!("expected offending_dim=Some(1), got {err:?}"),
        }
    }

    /// Ix5 slice and size calculation.
    #[test]
    fn test_ix5_basic() {
        let dim = Ix5(2, 3, 4, 5, 6);
        assert_eq!(dim.slice(), &[2, 3, 4, 5, 6]);
        assert_eq!(dim.checked_size(), Ok(720));
        assert_eq!(Ix5::from((2, 3, 4, 5, 6)), dim);
    }

    /// Overflow reports offending_dim.
    #[test]
    fn test_ix5_overflow() {
        let big = Ix5(usize::MAX, 2, 1, 1, 1);
        let err = big.checked_size().expect_err("should overflow");
        if let XenonError::InvalidShape { offending_dim, .. } = err {
            assert_eq!(offending_dim, Some(1));
        } else {
            panic!("expected InvalidShape, got {:?}", err);
        }
    }

    /// Ix6 tuple construction yields correct fields.
    #[test]
    fn test_ix6_from_tuple() {
        let dim = Ix6::from((1, 2, 3, 4, 5, 6));
        assert_eq!(dim, Ix6(1, 2, 3, 4, 5, 6));
        assert_eq!(dim.ndim(), 6);
    }

    /// Ix6 with moderate dimensions computes size correctly.
    #[test]
    fn test_ix6_max_dimensions() {
        let dim = Ix6(10, 10, 10, 10, 10, 10);
        assert_eq!(dim.checked_size(), Ok(1_000_000));
        assert_eq!(dim.checked(), Ok(()));
    }

    /// Overflow in checked_size reports offending axis.
    #[test]
    fn test_ix6_overflow_offending_dim() {
        let large = usize::MAX / 2 + 1;
        let dim = Ix6(large, 2, 1, 1, 1, 1);
        let err = dim.checked_size().expect_err("should overflow");
        match &err {
            XenonError::InvalidShape {
                offending_dim: Some(1),
                ..
            } => { /* expected */ },
            _ => panic!("expected ProductOverflow at axis 1, got {err:?}"),
        }
    }

    // --- Cross-type tests ---------------------------------------------------

    /// `into_dyn` for each static rank.
    #[test]
    fn test_static_to_dyn() {
        assert_eq!(
            Ix0.into_dyn().slice(),
            &[] as &[usize]
        );
        assert_eq!(
            Ix1(2).into_dyn().slice(),
            &[2]
        );
        assert_eq!(
            Ix2(2, 3).into_dyn().slice(), 
            &[2, 3]
        );
        assert_eq!(
            Ix3(2, 3, 4).into_dyn().slice(),
            &[2, 3, 4]
        );
        assert_eq!(
            Ix4(2, 3, 4, 5).into_dyn().slice(), 
            &[2, 3, 4, 5]
        );
        assert_eq!(
            Ix5(2, 3, 4, 5, 6).into_dyn().slice(), 
            &[2, 3, 4, 5, 6]
        );
        assert_eq!(
            Ix6(2, 3, 4, 5, 6, 7).into_dyn().slice(),
            &[2, 3, 4, 5, 6, 7]
        );
    }

    /// `try_from_dyn` succeeds when rank matches.
    #[test]
    fn test_dyn_to_static_success() {
        assert_eq!(
            Ix0::try_from_dyn(IxDyn::new()),
            Ok(Ix0)
        );
        assert_eq!(
            Ix3::try_from_dyn(IxDyn::from_slice(&[2, 3, 4])),
            Ok(Ix3(2, 3, 4))
        );
    }

    /// `try_from_dyn` returns `DimensionMismatch` on rank mismatch.
    #[test]
    fn test_dyn_to_static_failure() {
        match Ix3::try_from_dyn(IxDyn::from_slice(&[2, 3, 4, 5])) {
            Err(XenonError::DimensionMismatch {
                operation,
                expected,
                actual,
            }) => {
                assert_eq!(operation, "Ix3::try_from_dyn");
                assert_eq!(expected, 3);
                assert_eq!(actual, 4);
            },
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    /// Roundtrip invariant:
    /// try_from_dyn(into_dyn(d)) == Ok(d) for all static d.
    #[test]
    fn test_static_dynamic_roundtrip() {
        let dim = Ix3(2, 3, 4);
        let dyn_dim = dim.into_dyn();
        assert_eq!(dyn_dim.slice(), &[2, 3, 4]);
        assert_eq!(Ix3::try_from_dyn(dyn_dim), Ok(dim));
    }

    /// IxDyn identity conversions.
    #[test]
    fn test_ixdyn_identity_conversions() {
        let d = IxDyn::from_slice(&[1, 2, 3]);
        assert_eq!(d.clone().into_dyn().slice(), &[1, 2, 3]);
        assert_eq!(IxDyn::try_from_dyn(d.clone()), Ok(d));
    }

    /// Ix1 Index out of bounds panics.
    #[test]
    #[should_panic(expected = "Ix1 index out of bounds")]
    fn test_ix1_index_oob_panics() {
        let _ = Ix1(5)[1];
    }

    /// Ix2 Index out of bounds panics.
    #[test]
    #[should_panic(expected = "Ix2 index out of bounds")]
    fn test_ix2_index_oob_panics() {
        let _ = Ix2(3, 4)[2];
    }

    /// `From<tuple>` for the low-rank types (Ix0/Ix1/Ix2) completes the
    /// family already covered for Ix3..Ix6.
    #[test]
    fn test_from_tuple_low_rank() {
        assert_eq!(Ix0::from(()), Ix0);
        assert_eq!(Ix1::from((5,)), Ix1(5));
        assert_eq!(Ix2::from((3, 4)), Ix2(3, 4));
    }

    /// `Index<usize>` works for the high-rank types (Ix3..Ix6), completing
    /// the family already covered for Ix1/Ix2.
    #[test]
    fn test_index_high_rank() {
        let d3 = Ix3(2, 3, 4);
        assert_eq!((d3[0], d3[1], d3[2]), (2, 3, 4));
        let d4 = Ix4(2, 3, 4, 5);
        assert_eq!((d4[0], d4[3]), (2, 5));
        let d5 = Ix5(2, 3, 4, 5, 6);
        assert_eq!((d5[0], d5[4]), (2, 6));
        let d6 = Ix6(1, 2, 3, 4, 5, 6);
        assert_eq!((d6[0], d6[5]), (1, 6));
    }

    /// Ix0 has no axes, so indexing always panics.
    #[test]
    #[should_panic(expected = "Ix0 index out of bounds")]
    fn test_ix0_index_panics() {
        let _ = Ix0[0];
    }

    /// Ix3 Index out of bounds panics.
    #[test]
    #[should_panic(expected = "Ix3 index out of bounds")]
    fn test_ix3_index_oob_panics() {
        let _ = Ix3(2, 3, 4)[3];
    }

    /// Ix6 Index out of bounds panics.
    #[test]
    #[should_panic(expected = "Ix6 index out of bounds")]
    fn test_ix6_index_oob_panics() {
        let _ = Ix6(1, 2, 3, 4, 5, 6)[6];
    }

    /// Zero-length axis case — size is `Ok(0)`, not an error.
    #[test]
    fn test_zero_length_axis_yields_zero_size() {
        let dim = Ix2(0, 5);
        assert_eq!(dim.slice(), &[0, 5]);
        assert_eq!(dim.checked_size(), Ok(0));
        let dim = Ix3(3, 0, 5);
        assert_eq!(dim.checked_size(), Ok(0));
    }

    /// Verify that the `const _` layout assertion block compiles
    /// for Ix1-Ix6.
    #[test]
    fn test_static_ix_layout_assertions_compile() {
        assert_eq!(size_of::<Ix1>(), size_of::<[usize; 1]>());
        assert_eq!(size_of::<Ix2>(), size_of::<[usize; 2]>());
        assert_eq!(size_of::<Ix3>(), size_of::<[usize; 3]>());
        assert_eq!(size_of::<Ix4>(), size_of::<[usize; 4]>());
        assert_eq!(size_of::<Ix5>(), size_of::<[usize; 5]>());
        assert_eq!(size_of::<Ix6>(), size_of::<[usize; 6]>());
        assert_eq!(align_of::<Ix1>(), align_of::<[usize; 1]>());
        assert_eq!(align_of::<Ix6>(), align_of::<[usize; 6]>());
    }

    /// Zero-dim axis ops return InvalidAxis (recoverable error).
    #[test]
    fn test_ix0_axis_returns_invalid_axis() {
        assert!(matches!(
            Ix0.axis(Axis::new(0)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }

    /// `axis()` returns the length for a valid axis index.
    #[test]
    fn test_axis_returns_length_for_valid_index() {
        let dim = Ix2(3, 4);
        assert_eq!(dim.axis(Axis::new(0)), Ok(3));
        assert_eq!(dim.axis(Axis::new(1)), Ok(4));
        let dim = Ix3(2, 3, 4);
        assert_eq!(dim.axis(Axis::new(2)), Ok(4));
    }

    // --- Reverse tests ------------------------------------------------------

    /// `reverse()` for Ix0 and Ix1 is identity.
    #[test]
    fn test_reverse_identity() {
        assert_eq!(Ix0.reverse(), Ix0);
        assert_eq!(Ix1(5).reverse(), Ix1(5));
    }

    /// `reverse()` swaps axis order correctly.
    #[test]
    fn test_reverse_swaps() {
        assert_eq!(Ix2(3, 4).reverse(), Ix2(4, 3));
        assert_eq!(Ix3(2, 3, 4).reverse(), Ix3(4, 3, 2));
        assert_eq!(Ix4(1, 2, 3, 4).reverse(), Ix4(4, 3, 2, 1));
        assert_eq!(Ix5(1, 2, 3, 4, 5).reverse(), Ix5(5, 4, 3, 2, 1));
        assert_eq!(Ix6(1, 2, 3, 4, 5, 6).reverse(), Ix6(6, 5, 4, 3, 2, 1));
    }

    /// `reverse().reverse()` is identity.
    #[test]
    fn test_reverse_roundtrip() {
        let dim = Ix4(2, 4, 6, 8);
        assert_eq!(dim.reverse().reverse(), dim);
    }

    // --- RemoveAxis tests ---------------------------------------------------

    /// Ix0 has no axes, so `remove_axis` always errors.
    #[test]
    fn test_remove_axis_ix0_errors() {
        assert!(matches!(
            Ix0.remove_axis(Axis::new(0)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }

    /// `remove_axis` for Ix1: only axis 0 is valid.
    #[test]
    fn test_remove_axis_ix1() {
        let dim = Ix1(5);
        assert_eq!(dim.remove_axis(Axis::new(0)), Ok((Ix0, 5)));
        assert!(dim.remove_axis(Axis::new(1)).is_err());
    }

    /// `remove_axis` for Ix2.
    #[test]
    fn test_remove_axis_ix2() {
        let dim = Ix2(3, 4);
        assert_eq!(dim.remove_axis(Axis::new(0)), Ok((Ix1(4), 3)));
        assert_eq!(dim.remove_axis(Axis::new(1)), Ok((Ix1(3), 4)));
        assert!(dim.remove_axis(Axis::new(2)).is_err());
    }

    /// `remove_axis` for Ix3.
    #[test]
    fn test_remove_axis_ix3() {
        let dim = Ix3(2, 3, 4);
        assert_eq!(dim.remove_axis(Axis::new(0)), Ok((Ix2(3, 4), 2)));
        assert_eq!(dim.remove_axis(Axis::new(1)), Ok((Ix2(2, 4), 3)));
        assert_eq!(dim.remove_axis(Axis::new(2)), Ok((Ix2(2, 3), 4)));
        assert!(dim.remove_axis(Axis::new(3)).is_err());
    }

    /// `remove_axis` for Ix4: spot-check first and last, plus OOB.
    #[test]
    fn test_remove_axis_ix4() {
        let dim = Ix4(2, 3, 4, 5);
        assert_eq!(dim.remove_axis(Axis::new(0)), Ok((Ix3(3, 4, 5), 2)));
        assert_eq!(dim.remove_axis(Axis::new(3)), Ok((Ix3(2, 3, 4), 5)));
        assert!(dim.remove_axis(Axis::new(4)).is_err());
    }

    /// `remove_axis` for Ix5 and Ix6: spot-check.
    #[test]
    fn test_remove_axis_ix5_ix6() {
        let dim5 = Ix5(1, 2, 3, 4, 5);
        assert_eq!(dim5.remove_axis(Axis::new(2)), Ok((Ix4(1, 2, 4, 5), 3)));
        assert!(dim5.remove_axis(Axis::new(5)).is_err());

        let dim6 = Ix6(1, 2, 3, 4, 5, 6);
        assert_eq!(dim6.remove_axis(Axis::new(0)), Ok((Ix5(2, 3, 4, 5, 6), 1)));
        assert_eq!(dim6.remove_axis(Axis::new(5)), Ok((Ix5(1, 2, 3, 4, 5), 6)));
        assert!(dim6.remove_axis(Axis::new(6)).is_err());
    }
}
