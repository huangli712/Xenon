//! F-order contiguity detection.
//!
//! F-order matrix layout algorithm: axes with extent > 1 must have
//! strictly increasing strides, with stride[0] == 1.

use crate::dimension::Dimension;
use super::Strides;

/// Returns `true` if the tensor is F-contiguous.
///
/// An F-contiguous layout has `stride[0] == 1` and strictly increasing
/// strides for axes with extent > 1. Size-1 axes may have arbitrary
/// strides. Empty and single-element tensors are always contiguous.
pub(crate) fn is_f_contiguous<D: Dimension>(shape: &D, strides: &Strides<D>) -> bool {
    let shape = shape.slice();
    let strides = strides.as_slice();

    // Fast path: empty / scalar / single-element layouts are always
    // contiguous, regardless of stride values.
    let mut size: usize = 1;
    for &extent in shape.iter() {
        size = match size.checked_mul(extent) {
            Some(v) => v,
            None => break, // overflow ⇒ definitely > 1 ⇒ go to general path
        };
        if size == 0 {
            return true;
        }
    }
    if size <= 1 {
        return true;
    }

    // General path: expected stride accumulates the product(shape[0..i]);
    // axes with shape[i] == 1 are skipped (stride may be arbitrary).
    let mut expected: usize = 1;
    for (&extent, &stride) in shape.iter().zip(strides.iter()) {
        if extent != 1 && stride != expected {
            return false;
        }
        // `expected_stride` accumulator: overflow saturates conservatively;
        // any subsequent stride that has to equal a saturated value will
        // simply fail the equality check and short-circuit to `false`.
        expected = match expected.checked_mul(extent) {
            Some(v) => v,
            None => return false,
        };
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::dimension::{Ix0, Ix1, Ix2, Ix3};
    use super::*;

    /// Empty shape with zero stride is still F-contiguous.
    #[test]
    fn test_f_contig_empty() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// F-order strides for [2, 3] are [1, 2] ⇒ contiguous.
    #[test]
    fn test_f_contig_true() {
        let shape = Ix2(2, 3);
        let strides = Strides::new(Ix2(1, 2));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// C-order strides [3, 1] for [2, 3] ⇒ NOT F-contiguous.
    #[test]
    fn test_f_contig_false() {
        let shape = Ix2(2, 3);
        let strides = Strides::new(Ix2(3, 1));
        assert!(!is_f_contiguous(&shape, &strides));
    }

    /// 0-D scalar always F-contiguous.
    #[test]
    fn test_f_contig_scalar() {
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// Size-1 axis with arbitrary stride is still F-contiguous.
    #[test]
    fn test_f_contig_size1_axis() {
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// 1-D arrays are F-contiguous when stride[0] == 1.
    #[test]
    fn test_f_contig_1d() {
        let shape = Ix1(5);
        let strides = Strides::new(Ix1(1));
        assert!(is_f_contiguous(&shape, &strides));
    }

    /// `Strides::f_contiguous` output is always recognised as F-contiguous.
    #[test]
    fn test_f_contiguous_round_trip() {
        let shape = Ix3(2, 3, 4);
        let s = Strides::f_contiguous(&shape).expect("valid test shape");
        assert_eq!(s.as_slice(), &[1, 2, 6]);
        assert!(is_f_contiguous(&shape, &s));

        let shape = Ix2(4, 5);
        let s = Strides::f_contiguous(&shape).expect("valid test shape");
        assert_eq!(s.as_slice(), &[1, 4]);
        assert!(is_f_contiguous(&shape, &s));
    }
}
