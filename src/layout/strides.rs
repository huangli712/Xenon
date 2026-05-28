//! Stride carrier, F-order stride computation, and zero-stride helpers.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::error::{InvalidShapeKind, XenonError};

/// Stride carrier; element-offset along each axis, same rank as `D`.
#[derive(Debug, Clone)]
pub struct Strides<D: Dimension> {
    strides: D,
}

impl<D: Dimension> Strides<D> {
    /// Construct strides from a dimension value. Zero stride is allowed
    /// and represents a broadcast dimension. This constructor only wraps
    /// the carrier and does **not** perform full layout validation.
    pub fn new(strides: D) -> Self {
        Self { strides }
    }

    /// Borrow the stride storage as a slice. Delegates to
    /// `<D as Dimension>::slice`.
    pub fn as_slice(&self) -> &[usize] {
        self.strides.slice()
    }

    /// Construct strides from a slice of `usize` stride values.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::DimensionMismatch` if `slice.len()` does not
    /// match the rank of `D`.
    pub fn from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        let dim = D::try_from_slice(slice)?;
        Ok(Self { strides: dim })
    }

    /// Compute default F-contiguous strides for the given shape.
    /// Convenience alias for `compute_f_strides`.
    ///
    /// # Errors
    ///
    /// Forwards `compute_f_strides` errors: returns
    /// `XenonError::InvalidShape { kind: ProductOverflow, .. }` if the cumulative
    /// stride product overflows `usize`.
    pub fn f_contiguous(shape: &D) -> Result<Self, XenonError> {
        compute_f_strides(shape)
    }

    /// Returns the stride for dimension `axis`.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::IndexOutOfBounds` if `axis >= self.as_slice().len()`.
    pub fn try_stride(&self, axis: usize) -> Result<usize, XenonError> {
        let strides = self.as_slice();
        strides
            .get(axis)
            .copied()
            .ok_or_else(|| XenonError::IndexOutOfBounds {
                operation: Cow::Borrowed("Strides::try_stride"),
                attempted_index: vec![axis],
                axis: 0,
                shape: vec![strides.len()],
            })
    }

    /// Returns an iterator over stride values.
    ///
    /// Delegates to `D::slice().iter()`.
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.as_slice().iter()
    }

    /// Returns `true` iff any stride value equals 0.
    ///
    /// **This is NOT the same as the `HAS_ZERO_STRIDE` flag value**: the
    /// flag is set only when `product(shape) > 0` additionally holds.
    /// Use `should_set_zero_stride_flag(shape)` for flag assignment in
    /// `compute_layout_flags`.
    pub fn has_zero_stride(&self) -> bool {
        self.as_slice().contains(&0)
    }

    /// Returns `true` iff `any(stride == 0) && product(shape) > 0`.
    /// Empty-array degenerate metadata (`product(shape) == 0`) is
    /// excluded by this guard, so `compute_layout_flags` MUST call this
    /// helper instead of bare `has_zero_stride` when writing the bit.
    pub(crate) fn should_set_zero_stride_flag(&self, shape: &D) -> bool {
        if !self.has_zero_stride() {
            return false;
        }
        shape.slice().iter().all(|&e| e > 0)
    }
}

/// Compute strides for an F-order contiguous layout from the given shape.
///
/// Algorithm:
/// ```text
/// strides[0] = 1;
/// for i in 1..N: strides[i] = strides[i-1].checked_mul(shape[i-1])?
/// ```
///
/// # Errors
///
/// Returns `XenonError::InvalidShape { kind: ProductOverflow, .. }` if the
/// cumulative product overflows `usize`. The error is recoverable; this
/// function MUST NOT panic.
pub(crate) fn compute_f_strides<D: Dimension>(shape: &D) -> Result<Strides<D>, XenonError> {
    let axes = shape.slice();
    let mut values = vec![0_usize; axes.len()];
    let mut cumulative: usize = 1;
    for (axis_idx, &extent) in axes.iter().enumerate() {
        values[axis_idx] = cumulative;
        cumulative = cumulative
            .checked_mul(extent)
            .ok_or_else(|| XenonError::InvalidShape {
                operation: Cow::Borrowed("layout::compute_f_strides"),
                shape: axes.to_vec(),
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(axis_idx),
            })?;
    }
    Strides::from_slice(&values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix0, Ix1, Ix2, Ix3};

    #[test]
    fn test_strides_new_ix2() {
        let strides = Strides::new(Ix2(1, 3));
        assert_eq!(strides.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_f_strides_1d() {
        // F-contiguous strides computed from shape dimensions.
        let strides = compute_f_strides(&Ix1(5)).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1]);
    }

    #[test]
    fn test_f_strides_2d() {
        // F-contiguous strides for [3, 4] are [1, 3].
        let strides = compute_f_strides(&Ix2(3, 4)).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_f_strides_3d() {
        // F-contiguous strides for [2, 3, 4] are [1, 2, 6].
        let strides = compute_f_strides(&Ix3(2, 3, 4)).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
    }

    #[test]
    fn test_f_strides_scalar() {
        // 0-D scalar produces empty strides.
        let strides = compute_f_strides(&Ix0).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[] as &[usize]);
    }

    #[test]
    fn test_f_strides_overflow() {
        // Overflow of cumulative product → InvalidShape::ProductOverflow.
        let shape = Ix2(usize::MAX, usize::MAX);
        let err = compute_f_strides(&shape).expect_err("expected overflow error");
        match err {
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ProductOverflow,
                ..
            } => {},
            other => panic!("expected InvalidShape::ProductOverflow, got {other:?}"),
        }
    }

    #[test]
    fn test_strides_try_stride_ok() {
        // try_stride returns Ok(value) for valid axis.
        let strides = compute_f_strides(&Ix3(2, 3, 4)).expect("valid test shape");
        assert_eq!(strides.try_stride(0).expect("valid axis"), 1);
        assert_eq!(strides.try_stride(1).expect("valid axis"), 2);
        assert_eq!(strides.try_stride(2).expect("valid axis"), 6);
    }

    #[test]
    fn test_strides_try_stride_out_of_bounds() {
        // try_stride returns Err(IndexOutOfBounds) for axis >= ndim.
        let strides = compute_f_strides(&Ix2(3, 4)).expect("valid test shape");
        let err = strides
            .try_stride(2)
            .expect_err("expected out-of-bounds error");
        match err {
            XenonError::IndexOutOfBounds { .. } => {},
            other => panic!("expected IndexOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn test_strides_iter() {
        // iter yields stride values in axis order.
        let strides = compute_f_strides(&Ix3(2, 3, 4)).expect("valid test shape");
        let collected: Vec<usize> = strides.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 6]);
    }

    // --- zero-stride tests ---

    #[test]
    fn test_zero_stride_detect() {
        assert!(Strides::new(Ix2(1, 0)).has_zero_stride());
        assert!(!Strides::new(Ix2(1, 2)).has_zero_stride());
    }

    #[test]
    fn test_should_set_zero_stride_flag_broadcast() {
        let shape = Ix2(5, 1);
        let strides = Strides::new(Ix2(1, 0));
        assert!(strides.should_set_zero_stride_flag(&shape));
    }

    #[test]
    fn test_should_set_zero_stride_flag_empty() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        assert!(!strides.should_set_zero_stride_flag(&shape));
    }

    #[test]
    fn test_should_set_zero_stride_flag_no_zero() {
        let shape = Ix2(5, 4);
        let strides = Strides::new(Ix2(1, 5));
        assert!(!strides.should_set_zero_stride_flag(&shape));
    }
}
