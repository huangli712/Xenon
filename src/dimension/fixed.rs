//! Fixed-size dimension types.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::error::XenonError;
use crate::error::InvalidShapeKind;

/// Zero-dimensional index (scalar). Always has rank 0, size 1.
/// This type is a ZST (Zero-Sized Type); `size_of::<Ix0>() == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix0;

impl crate::private::Sealed for Ix0 {}

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

/// One-dimensional index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix1(pub usize);

impl crate::private::Sealed for Ix1 {}

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
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
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

impl std::ops::Index<usize> for Ix1 {
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &usize {
        assert_eq!(index, 0, "Ix1 index out of bounds: {index}");
        &self.0
    }
}

/// Two-dimensional index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix2(pub usize, pub usize);

impl crate::private::Sealed for Ix2 {}

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
            acc = acc.checked_mul(dim).ok_or_else(|| XenonError::InvalidShape {
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

/// Three-dimensional index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix3(pub usize, pub usize, pub usize);

impl crate::private::Sealed for Ix3 {}

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
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const usize,
                3,
            )
        }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = [self.0, self.1, self.2];
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc.checked_mul(dim).ok_or_else(|| XenonError::InvalidShape {
                operation: Cow::Borrowed("Dimension::checked_size"),
                shape: dims.into(),
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(axis),
            })?;
        }
        Ok(acc)
    }

    #[inline]
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
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

/// Four-dimensional index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Ix4(pub usize, pub usize, pub usize, pub usize);

impl crate::private::Sealed for Ix4 {}

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
        unsafe {
            std::slice::from_raw_parts(self as *const Self as *const usize, 4)
        }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let mut size = 1usize;
        let axes = [self.0, self.1, self.2, self.3];
        for (i, &ax) in axes.iter().enumerate() {
            size = size.checked_mul(ax).ok_or_else(|| XenonError::InvalidShape {
                operation: Cow::Borrowed("Dimension::checked_size"),
                shape: axes.into(),
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(i),
            })?;
        }
        Ok(size)
    }

    #[inline]
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
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

/// Five-dimensional index.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as
/// `&[usize; 5]` via pointer cast; this is only safe because `repr(C)`
/// guarantees the `usize` fields are laid out contiguously starting at offset 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix5(pub usize, pub usize, pub usize, pub usize, pub usize);

impl crate::private::Sealed for Ix5 {}

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
        unsafe {
            core::slice::from_raw_parts(self as *const Self as *const usize, 5)
        }
    }

    #[inline]
    fn checked_size(&self) -> Result<usize, XenonError> {
        let dims = [self.0, self.1, self.2, self.3, self.4];
        let mut acc = 1usize;
        for (axis, &dim) in dims.iter().enumerate() {
            acc = acc.checked_mul(dim).ok_or_else(|| XenonError::InvalidShape {
                operation: Cow::Borrowed("Dimension::checked_size"),
                shape: dims.into(),
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(axis),
            })?;
        }
        Ok(acc)
    }

    #[inline]
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
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

/// Six-dimensional index.
///
/// `#[repr(C)]` is required because `slice()` reinterprets `&Self` as
/// `&[usize; 6]` via pointer cast; this is only safe because `repr(C)`
/// guarantees the `usize` fields are laid out contiguously starting at offset 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Ix6(
    pub usize, pub usize, pub usize, pub usize, pub usize, pub usize,
);

impl crate::private::Sealed for Ix6 {}

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
            acc = acc.checked_mul(dim).ok_or_else(|| XenonError::InvalidShape {
                operation: Cow::Borrowed("Dimension::checked_size"),
                shape: dims.into(),
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(axis),
            })?;
        }
        Ok(acc)
    }

    #[inline]
    fn checked(&self) -> Result<(), XenonError> {
        self.checked_size().map(|_| ())
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
                offending_dim: Some(1), ..
            } => {},
            _ => panic!(/* expected InvalidShape with offending_dim: Some(1), got {err:?} */),
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
            } => {}
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
                offending_dim: Some(1), ..
            } => {}
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
        if let XenonError::InvalidShape {
            offending_dim, ..
        } = err
        {
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
                offending_dim: Some(1), ..
            } => { /* expected */ }
            _ => panic!("expected ProductOverflow at axis 1, got {err:?}"),
        }
    }
}
