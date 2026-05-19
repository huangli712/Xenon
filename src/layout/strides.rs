//! Stride carrier and helpers.
//!
//! Full implementations:
//! - `compute_f_strides`      → here (W6T6)
//! - `has_zero_stride`        → W6T8
//! - `is_aligned` / `is_aligned_to` → W6T9

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::error::{InvalidShapeKind, XenonError};

/// Stride carrier; element-offset along each axis, same rank as `D`.
///
/// See `06-layout §5.5` for the full API contract.
#[derive(Debug)]
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
    /// Returns `XenonError::DimensionMismatch` if `slice.len()` does not
    /// match the rank of `D`. See `06-layout §5.5`.
    pub fn from_slice(slice: &[usize]) -> Result<Self, XenonError> {
        let dim = D::try_from_slice(slice)?;
        Ok(Self { strides: dim })
    }

    /// Compute default F-contiguous strides for the given shape.
    /// Convenience alias for the free function `compute_f_strides`
    /// (`06-layout §5.5`).
    pub fn f_contiguous(shape: &D) -> Result<Self, XenonError> {
        compute_f_strides(shape)
    }

    /// Returns the stride for dimension `axis`.
    ///
    /// Returns `XenonError::IndexOutOfBounds` if `axis >= self.as_slice().len()`.
    /// See `06-layout §5.5`.
    pub fn try_stride(&self, axis: usize) -> Result<usize, XenonError> {
        let strides = self.as_slice();
        strides.get(axis).copied().ok_or_else(|| {
            XenonError::IndexOutOfBounds {
                operation: Cow::Borrowed("Strides::try_stride"),
                attempted_index: vec![axis],
                axis: 0,
                shape: vec![strides.len()],
            }
        })
    }

    /// Returns an iterator over stride values.
    ///
    /// Delegates to `D::slice().iter()` so the iteration shares its
    /// representation with `Dimension::slice()`. See `06-layout §5.5`.
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.as_slice().iter()
    }
}

/// Compute strides for an F-order contiguous layout from the given shape.
///
/// See `06-layout §5.6` algorithm:
/// ```text
/// strides[0] = 1;
/// for i in 1..N: strides[i] = strides[i-1].checked_mul(shape[i-1])?
/// ```
///
/// Returns `XenonError::InvalidShape { kind: ProductOverflow, .. }` if the
/// cumulative product overflows `usize`. The error is recoverable; this
/// function MUST NOT panic.
pub fn compute_f_strides<D: Dimension>(shape: &D) -> Result<Strides<D>, XenonError> {
    let axes = shape.slice();
    let mut values = vec![0_usize; axes.len()];
    let mut cumulative: usize = 1;
    for (axis_idx, &extent) in axes.iter().enumerate() {
        values[axis_idx] = cumulative;
        cumulative = cumulative.checked_mul(extent).ok_or_else(|| {
            XenonError::InvalidShape {
                operation: Cow::Borrowed("layout::compute_f_strides"),
                shape: axes.to_vec(),
                kind: InvalidShapeKind::ProductOverflow,
                offending_dim: Some(axis_idx),
            }
        })?;
    }
    Strides::from_slice(&values)
}

// === Zero-stride detection (W6T8) and alignment checks (W6T9) ===

/// Raw zero-stride detector: returns `true` iff any stride value equals 0.
///
/// **This is NOT the same as the `HAS_ZERO_STRIDE` flag value**: the flag
/// is set only when `product(shape) > 0` additionally holds. Use
/// `should_set_zero_stride_flag(shape, strides)` for flag assignment in
/// `compute_layout_flags` (`06-layout §5.11`, §6.1).
pub fn has_zero_stride<D: Dimension>(strides: &Strides<D>) -> bool {
    strides.as_slice().contains(&0)
}

/// Flag-assignment guard for `HAS_ZERO_STRIDE`.
///
/// Returns `true` iff `any(stride == 0) && product(shape) > 0`, which is
/// the formal definition from `06-layout §5.11`. Empty-array degenerate
/// metadata (`product(shape) == 0`) is excluded by this guard, so
/// `compute_layout_flags` MUST call this helper instead of bare
/// `has_zero_stride` when writing the bit.
pub(crate) fn should_set_zero_stride_flag<D: Dimension>(
    shape: &D,
    strides: &Strides<D>,
) -> bool {
    if !has_zero_stride(strides) {
        return false;
    }
    shape.slice().iter().all(|&e| e > 0)
}

/// Check whether the logical-first pointer satisfies the alignment
/// requirement (`06-layout §5.9`).
///
/// Returns `false` for `align == 0` or non-power-of-two `align`; never
/// panics. The pointer is inspected only as an integer address (modulo
/// `align`); it is **not** dereferenced, and is permitted to be dangling
/// (e.g., for empty tensors; see §6.5).
#[inline]
pub fn is_aligned_to(ptr: *const u8, align: usize) -> bool {
    if align == 0 || !align.is_power_of_two() {
        return false;
    }
    (ptr as usize).is_multiple_of(align)
}

/// Check whether the logical first-element pointer is 64-byte aligned
/// (cache-line size; the minimum useful for most SIMD paths).
/// See `06-layout §5.9`.
#[inline]
pub fn is_aligned(ptr: *const u8) -> bool {
    is_aligned_to(ptr, 64)
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
        // §8.2 high priority: shape [5] → strides [1]
        let strides = compute_f_strides(&Ix1(5)).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1]);
    }

    #[test]
    fn test_f_strides_2d() {
        // §8.2 high priority: shape [3, 4] → strides [1, 3]
        let strides = compute_f_strides(&Ix2(3, 4)).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_f_strides_3d() {
        // §8.2 high priority: shape [2, 3, 4] → strides [1, 2, 6]
        let strides = compute_f_strides(&Ix3(2, 3, 4)).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
    }

    #[test]
    fn test_f_strides_scalar() {
        // §8.2 high priority: 0-D → empty strides.
        let strides = compute_f_strides(&Ix0).expect("valid test shape");
        assert_eq!(strides.as_slice(), &[] as &[usize]);
    }

    #[test]
    fn test_f_strides_overflow() {
        // §8.2 high priority: overflow → Err(InvalidShape::ProductOverflow).
        let shape = Ix2(usize::MAX, usize::MAX);
        let err = compute_f_strides(&shape).expect_err("expected overflow error");
        match err {
            XenonError::InvalidShape { kind: InvalidShapeKind::ProductOverflow, .. } => {}
            other => panic!("expected InvalidShape::ProductOverflow, got {other:?}"),
        }
    }

    #[test]
    fn test_strides_try_stride_ok() {
        // §5.5: try_stride returns Ok(value) for valid axis.
        let strides = compute_f_strides(&Ix3(2, 3, 4)).expect("valid test shape");
        assert_eq!(strides.try_stride(0).expect("valid axis"), 1);
        assert_eq!(strides.try_stride(1).expect("valid axis"), 2);
        assert_eq!(strides.try_stride(2).expect("valid axis"), 6);
    }

    #[test]
    fn test_strides_try_stride_out_of_bounds() {
        // §5.5: try_stride returns Err(IndexOutOfBounds) for axis >= ndim.
        let strides = compute_f_strides(&Ix2(3, 4)).expect("valid test shape");
        let err = strides.try_stride(2).expect_err("expected out-of-bounds error");
        match err {
            XenonError::IndexOutOfBounds { .. } => {}
            other => panic!("expected IndexOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn test_strides_iter() {
        // §5.5: iter yields stride values in axis order.
        let strides = compute_f_strides(&Ix3(2, 3, 4)).expect("valid test shape");
        let collected: Vec<usize> = strides.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 6]);
    }

    // --- §8.2 / §5.11 zero-stride tests (W6T8) ---

    #[test]
    fn test_zero_stride_detect() {
        assert!(has_zero_stride(&Strides::new(Ix2(1, 0))));
        assert!(!has_zero_stride(&Strides::new(Ix2(1, 2))));
    }

    #[test]
    fn test_should_set_zero_stride_flag_broadcast() {
        let shape = Ix2(5, 1);
        let strides = Strides::new(Ix2(1, 0));
        assert!(should_set_zero_stride_flag(&shape, &strides));
    }

    #[test]
    fn test_should_set_zero_stride_flag_empty() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        assert!(!should_set_zero_stride_flag(&shape, &strides));
    }

    #[test]
    fn test_should_set_zero_stride_flag_no_zero() {
        let shape = Ix2(5, 4);
        let strides = Strides::new(Ix2(1, 5));
        assert!(!should_set_zero_stride_flag(&shape, &strides));
    }

    // --- §8.2 alignment tests (W6T9) ---

    #[test]
    fn test_alignment_aligned() {
        use std::alloc::{alloc, dealloc, Layout};
        let layout = Layout::from_size_align(256, 64).expect("valid layout");
        // SAFETY: layout is non-zero size with valid align.
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocator returned null");
        assert!(is_aligned(ptr));
        assert!(is_aligned_to(ptr, 64));
        assert!(is_aligned_to(ptr, 32));
        assert!(is_aligned_to(ptr, 1));
        // SAFETY: ptr was obtained from `alloc(layout)`.
        unsafe { dealloc(ptr, layout); }
    }

    #[test]
    fn test_alignment_unaligned() {
        let values = [1_u8, 2, 3];
        // SAFETY: `.add(1)` stays within the allocation of `values`.
        let ptr = unsafe { values.as_ptr().add(1) };
        assert!(!is_aligned_to(ptr, 64));
        assert!(!is_aligned_to(values.as_ptr(), 0));
        assert!(!is_aligned_to(values.as_ptr(), 3));
    }
}
