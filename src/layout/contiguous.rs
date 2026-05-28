//! F-order contiguity detection.
//!
//! Full recognition algorithm per `06-layout §5.7`.

use crate::dimension::Dimension;

use super::Strides;

/// Returns `true` if the tensor is F-contiguous.
///
/// Uses the F-order recognition algorithm from `06-layout §5.7`:
/// - Empty / scalar / single-element layouts always return `true`.
/// - For general shapes: axis 0 → axis N-1, skipping `shape[i] == 1` axes.
/// - `expected_stride` accumulates with `checked_mul` to guard against
///   overflow in raw-part inputs.
pub fn is_f_contiguous<D: Dimension>(shape: &D, strides: &Strides<D>) -> bool {
    let shape = shape.slice();
    let strides = strides.as_slice();

    // §5.7 fast path: empty / scalar / single-element layouts are always
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

    // §5.7 general path: expected_stride accumulates product(shape[0..i]);
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
    use super::*;
    use crate::dimension::{Ix0, Ix1, Ix2, Ix3};

    #[test]
    fn test_f_contig_empty() {
        let shape = Ix2(0, 3);
        let strides = Strides::new(Ix2(1, 0));
        assert!(is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contig_true() {
        // §8.2 high: F-order strides for [2, 3] are [1, 2] ⇒ contiguous.
        let shape = Ix2(2, 3);
        let strides = Strides::new(Ix2(1, 2));
        assert!(is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contig_false() {
        // §8.2 high: C-order strides [3, 1] for [2, 3] ⇒ NOT F-contiguous.
        let shape = Ix2(2, 3);
        let strides = Strides::new(Ix2(3, 1));
        assert!(!is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contig_scalar() {
        // §8.2 high: 0-D scalar always F-contiguous.
        let shape = Ix0;
        let strides = Strides::new(Ix0);
        assert!(is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contig_size1_axis() {
        // §8.2 medium: size=1 axis is allowed to have arbitrary stride.
        let shape = Ix3(5, 1, 4);
        let strides = Strides::new(Ix3(1, 999, 5));
        assert!(is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contig_1d() {
        // 1-D arrays are F-contiguous when stride[0] == 1.
        let shape = Ix1(5);
        let strides = Strides::new(Ix1(1));
        assert!(is_f_contiguous(&shape, &strides));
    }
}
