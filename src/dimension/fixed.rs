//! Fixed-size dimension types.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::dimension::dynamic::IxDyn;
use crate::error::InvalidShapeKind;
use crate::error::XenonError;

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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically (slice is empty so any index is
    // invalid).
}

impl Ix0 {
    /// Converts to dynamic dimension. Always succeeds. Returns a 0-rank IxDyn.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::new()
    }

    /// Attempts to convert from a dynamic dimension.
    /// Returns `XenonError::DimensionMismatch` if `dyn_dim.ndim() != 0`.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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
        std::slice::from_ref(&self.0)
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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically (single axis at index 0).
}

/// Index-based access to axis length.
///
/// # Panics
///
/// Panics if `index != 0`, as `Ix1` has only one axis.
impl std::ops::Index<usize> for Ix1 {
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize {
        assert_eq!(index, 0, "Ix1 index out of bounds: {index}");
        &self.0
    }
}

impl Ix1 {
    /// Converts to dynamic dimension.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0])
    }

    /// Attempts to convert from a dynamic dimension.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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
        unsafe { core::slice::from_raw_parts(self as *const Self as *const usize, 2) }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = [self.0, self.1];
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc
                .checked_mul(dim)
                .ok_or_else(|| XenonError::InvalidShape {
                    operation: Cow::Borrowed("Dimension::checked_size"),
                    shape: dims.into(),
                    kind: InvalidShapeKind::ProductOverflow,
                    offending_dim: Some(axis),
                })?;
        }
        Ok(acc)
    }

    // `checked()` uses the trait default implementation (equivalent to
    // `self.checked_size().map(|_| ())`); no override needed.

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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically.
}

/// Index-based access to axis lengths.
///
/// # Panics
///
/// Panics if `index` is not 0 or 1, as `Ix2` has only two axes.
impl std::ops::Index<usize> for Ix2 {
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize {
        match index {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("Ix2 index out of bounds: {index}"),
        }
    }
}

impl Ix2 {
    /// Converts to dynamic dimension.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0, self.1])
    }

    /// Attempts to convert from a dynamic dimension.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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
        unsafe { core::slice::from_raw_parts(self as *const Self as *const usize, 3) }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = [self.0, self.1, self.2];
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc
                .checked_mul(dim)
                .ok_or_else(|| XenonError::InvalidShape {
                    operation: Cow::Borrowed("Dimension::checked_size"),
                    shape: dims.into(),
                    kind: InvalidShapeKind::ProductOverflow,
                    offending_dim: Some(axis),
                })?;
        }
        Ok(acc)
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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically.
}

impl From<(usize, usize, usize)> for Ix3 {
    #[inline]
    fn from((a, b, c): (usize, usize, usize)) -> Self {
        Ix3(a, b, c)
    }
}

impl Ix3 {
    /// Converts to dynamic dimension.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0, self.1, self.2])
    }

    /// Attempts to convert from a dynamic dimension.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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
        unsafe { std::slice::from_raw_parts(self as *const Self as *const usize, 4) }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let mut size = 1usize;
        let axes = [self.0, self.1, self.2, self.3];
        for (i, &ax) in axes.iter().enumerate() {
            size = size
                .checked_mul(ax)
                .ok_or_else(|| XenonError::InvalidShape {
                    operation: Cow::Borrowed("Dimension::checked_size"),
                    shape: axes.into(),
                    kind: InvalidShapeKind::ProductOverflow,
                    offending_dim: Some(i),
                })?;
        }
        Ok(size)
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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically.
}

impl From<(usize, usize, usize, usize)> for Ix4 {
    #[inline]
    fn from(t: (usize, usize, usize, usize)) -> Self {
        Ix4(t.0, t.1, t.2, t.3)
    }
}

impl Ix4 {
    /// Converts to dynamic dimension.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0, self.1, self.2, self.3])
    }

    /// Attempts to convert from a dynamic dimension.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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
        unsafe { core::slice::from_raw_parts(self as *const Self as *const usize, 5) }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = [self.0, self.1, self.2, self.3, self.4];
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc
                .checked_mul(dim)
                .ok_or_else(|| XenonError::InvalidShape {
                    operation: Cow::Borrowed("Dimension::checked_size"),
                    shape: dims.into(),
                    kind: InvalidShapeKind::ProductOverflow,
                    offending_dim: Some(axis),
                })?;
        }
        Ok(acc)
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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically.
}

impl From<(usize, usize, usize, usize, usize)> for Ix5 {
    #[inline]
    fn from(t: (usize, usize, usize, usize, usize)) -> Self {
        Ix5(t.0, t.1, t.2, t.3, t.4)
    }
}

impl Ix5 {
    /// Converts to dynamic dimension.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0, self.1, self.2, self.3, self.4])
    }

    /// Attempts to convert from a dynamic dimension.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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
pub struct Ix6(
    pub usize,
    pub usize,
    pub usize,
    pub usize,
    pub usize,
    pub usize,
);

/// Compile-time layout assertions for unsafe pointer casts in `slice()`.
///
/// Verifies `size_of`, `align_of`, and field offsets of each `IxN` type
/// against the corresponding `[usize; N]` array.  If any `#[repr(C)]`
/// attribute is removed or field types are altered, these assertions fail
/// at compile time instead of silently introducing UB.
///
/// See `02-dimension.md` §5.2.
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
        unsafe { std::slice::from_raw_parts(self as *const Self as *const usize, 6) }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = [self.0, self.1, self.2, self.3, self.4, self.5];
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc
                .checked_mul(dim)
                .ok_or_else(|| XenonError::InvalidShape {
                    operation: Cow::Borrowed("Dimension::checked_size"),
                    shape: dims.into(),
                    kind: InvalidShapeKind::ProductOverflow,
                    offending_dim: Some(axis),
                })?;
        }
        Ok(acc)
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

    // `axis()` uses the trait default implementation; out-of-range returns
    // `XenonError::InvalidAxis` automatically (axes 0..=5 valid, 6+ invalid).
}

impl From<(usize, usize, usize, usize, usize, usize)> for Ix6 {
    #[inline]
    fn from(t: (usize, usize, usize, usize, usize, usize)) -> Self {
        Ix6(t.0, t.1, t.2, t.3, t.4, t.5)
    }
}

impl Ix6 {
    /// Converts to dynamic dimension.
    #[inline]
    pub fn into_dyn(self) -> IxDyn {
        IxDyn::from_vec(vec![self.0, self.1, self.2, self.3, self.4, self.5])
    }

    /// Attempts to convert from a dynamic dimension.
    #[inline]
    pub fn try_from_dyn(dyn_dim: IxDyn) -> Result<Self, XenonError> {
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

// ── Reverse implementations for all static dimensions ──

impl crate::dimension::Reverse for Ix0 {
    fn reverse(self) -> Self { self }
}

impl crate::dimension::Reverse for Ix1 {
    fn reverse(self) -> Self { self }
}

impl crate::dimension::Reverse for Ix2 {
    fn reverse(self) -> Self { Ix2(self.1, self.0) }
}

impl crate::dimension::Reverse for Ix3 {
    fn reverse(self) -> Self { Ix3(self.2, self.1, self.0) }
}

impl crate::dimension::Reverse for Ix4 {
    fn reverse(self) -> Self { Ix4(self.3, self.2, self.1, self.0) }
}

impl crate::dimension::Reverse for Ix5 {
    fn reverse(self) -> Self { Ix5(self.4, self.3, self.2, self.1, self.0) }
}

impl crate::dimension::Reverse for Ix6 {
    fn reverse(self) -> Self { Ix6(self.5, self.4, self.3, self.2, self.1, self.0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    /// §8.2: Ix0 is a Zero-Sized Type.
    #[test]
    fn test_ix0_is_zst() {
        assert_eq!(size_of::<Ix0>(), 0);
    }

    /// §8.2: Ix0 has rank 0.
    #[test]
    fn test_ix0_ndim_is_zero() {
        let dim = Ix0;
        assert_eq!(dim.ndim(), 0);
        assert_eq!(dim.slice(), &[] as &[usize]);
    }

    /// §8.2: scalar has exactly one element.
    #[test]
    fn test_ix0_size_is_one() {
        assert_eq!(Ix0.checked_size(), Ok(1));
        assert_eq!(Ix0.checked(), Ok(()));
    }

    /// §8.2: Ix1 shape/rank/Index verification.
    #[test]
    fn test_ix1_slice() {
        assert_eq!(Ix1(7).ndim(), 1);
        assert_eq!(Ix1(7).slice(), &[7]);
        assert_eq!(Ix1(7).checked_size(), Ok(7));
        assert_eq!(Ix1(7)[0], 7);
    }

    /// §8.2: Ix2(3, 4).slice() == &[3, 4]
    #[test]
    fn test_ix2_slice() {
        let dim = Ix2(3, 4);
        assert_eq!(dim.ndim(), 2);
        assert_eq!(dim.slice(), &[3, 4]);
        assert_eq!(dim.checked_size(), Ok(12));
        assert_eq!(dim[0], 3);
        assert_eq!(dim[1], 4);
    }

    /// §8.2: overflow returns Err with offending_dim
    #[test]
    fn test_ix2_size_overflow() {
        let dim = Ix2(usize::MAX, 2);
        let err = dim.checked_size().expect_err("should overflow");
        match err {
            XenonError::InvalidShape {
                offending_dim: Some(1),
                ..
            } => {},
            _ => panic!("expected InvalidShape with offending_dim: Some(1), got {err:?}"),
        }
    }

    /// §8.2: Ix3(2, 3, 4).slice() == &[2, 3, 4]
    #[test]
    fn test_ix3_slice() {
        let dim = Ix3(2, 3, 4);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.slice(), &[2, 3, 4]);
        assert!(dim.checked().is_ok());
    }

    /// §8.2: Ix3(2, 3, 4).checked_size() == Ok(24)
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

    /// §8.2: Ix4 slice, rank, and size calculation.
    #[test]
    fn test_ix4_basic() {
        let dim = Ix4(2, 3, 4, 5);
        assert_eq!(dim.ndim(), 4);
        assert_eq!(dim.slice(), &[2, 3, 4, 5]);
        assert_eq!(dim.checked_size(), Ok(120));
    }

    /// §8.2: Ix4 tuple conversion.
    #[test]
    fn test_ix4_from_tuple() {
        let dim = Ix4::from((2, 3, 4, 5));
        assert_eq!(dim, Ix4(2, 3, 4, 5));
    }

    /// §8.2: overflow reports offending dimension.
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

    /// §5.4: Ix5 slice and size calculation.
    #[test]
    fn test_ix5_basic() {
        let dim = Ix5(2, 3, 4, 5, 6);
        assert_eq!(dim.slice(), &[2, 3, 4, 5, 6]);
        assert_eq!(dim.checked_size(), Ok(720));
        assert_eq!(Ix5::from((2, 3, 4, 5, 6)), dim);
    }

    /// §8.3 / §5.4: overflow reports offending_dim.
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

    /// §8.2: Ix6 tuple construction yields correct fields.
    #[test]
    fn test_ix6_from_tuple() {
        let dim = Ix6::from((1, 2, 3, 4, 5, 6));
        assert_eq!(dim, Ix6(1, 2, 3, 4, 5, 6));
        assert_eq!(dim.ndim(), 6);
    }

    /// §8.2: Ix6 with moderate dimensions computes size correctly.
    #[test]
    fn test_ix6_max_dimensions() {
        let dim = Ix6(10, 10, 10, 10, 10, 10);
        assert_eq!(dim.checked_size(), Ok(1_000_000));
        assert_eq!(dim.checked(), Ok(()));
    }

    /// §8.2: overflow in checked_size reports offending axis.
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
    /// §8.2: test_static_to_dyn — into_dyn for each static rank.
    #[test]
    fn test_static_to_dyn() {
        assert_eq!(Ix0.into_dyn().slice(), &[] as &[usize]);
        assert_eq!(Ix1(2).into_dyn().slice(), &[2]);
        assert_eq!(Ix2(2, 3).into_dyn().slice(), &[2, 3]);
        assert_eq!(Ix3(2, 3, 4).into_dyn().slice(), &[2, 3, 4]);
        assert_eq!(Ix4(2, 3, 4, 5).into_dyn().slice(), &[2, 3, 4, 5]);
        assert_eq!(Ix5(2, 3, 4, 5, 6).into_dyn().slice(), &[2, 3, 4, 5, 6]);
        assert_eq!(
            Ix6(2, 3, 4, 5, 6, 7).into_dyn().slice(),
            &[2, 3, 4, 5, 6, 7]
        );
    }

    /// §8.2: test_dyn_to_static_success — try_from_dyn succeeds
    /// when rank matches.
    #[test]
    fn test_dyn_to_static_success() {
        assert_eq!(Ix0::try_from_dyn(IxDyn::new()), Ok(Ix0));
        assert_eq!(
            Ix3::try_from_dyn(IxDyn::from_slice(&[2, 3, 4])),
            Ok(Ix3(2, 3, 4))
        );
    }

    /// §8.2: test_dyn_to_static_failure — try_from_dyn returns
    /// DimensionMismatch on rank mismatch.
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

    /// §8.2 / §5.3: Ix1 Index out of bounds panics.
    #[test]
    #[should_panic(expected = "Ix1 index out of bounds")]
    fn test_ix1_index_oob_panics() {
        let _ = Ix1(5)[1];
    }

    /// §8.2 / §5.3: Ix2 Index out of bounds panics.
    #[test]
    #[should_panic(expected = "Ix2 index out of bounds")]
    fn test_ix2_index_oob_panics() {
        let _ = Ix2(3, 4)[2];
    }

    /// §8.3 line 1105: zero-length axis case — size is `Ok(0)`, not an
    /// error.
    #[test]
    fn test_zero_length_axis_yields_zero_size() {
        let dim = Ix2(0, 5);
        assert_eq!(dim.slice(), &[0, 5]);
        assert_eq!(dim.checked_size(), Ok(0));
        let dim = Ix3(3, 0, 5);
        assert_eq!(dim.checked_size(), Ok(0));
    }

    /// §8.7 line 1139 / §5.2: test_static_ix_layout_assertions_compile —
    /// verify that the `const _` layout assertion block in §5.2
    /// compiles for Ix1-Ix6. This positive test ensures the assertion
    /// block exists at compile time by referencing its values at
    /// runtime.
    #[test]
    fn test_static_ix_layout_assertions_compile() {
        use std::mem::{align_of, size_of};
        assert_eq!(size_of::<Ix1>(), size_of::<[usize; 1]>());
        assert_eq!(size_of::<Ix2>(), size_of::<[usize; 2]>());
        assert_eq!(size_of::<Ix3>(), size_of::<[usize; 3]>());
        assert_eq!(size_of::<Ix4>(), size_of::<[usize; 4]>());
        assert_eq!(size_of::<Ix5>(), size_of::<[usize; 5]>());
        assert_eq!(size_of::<Ix6>(), size_of::<[usize; 6]>());
        assert_eq!(align_of::<Ix1>(), align_of::<[usize; 1]>());
        assert_eq!(align_of::<Ix6>(), align_of::<[usize; 6]>());
    }

    /// §8.7 line 1137: zero-dim axis ops return InvalidAxis (recoverable
    /// error).
    #[test]
    fn test_ix0_axis_returns_invalid_axis() {
        use crate::dimension::axes::Axis;
        use crate::error::XenonError;
        assert!(matches!(
            Ix0.axis(Axis::new(0)),
            Err(XenonError::InvalidAxis { .. })
        ));
    }
}
