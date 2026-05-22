use crate::complex::Complex;
use crate::dimension::Ix2;
use crate::element::Element;
use crate::error::XenonError;
use crate::private::Sealed;
use crate::storage::Owned;
use crate::tensor::TensorBase;

/// Sealed trait restricting `eye()` to the closed numeric set defined in
/// `需求说明书 §19` and `18-construction.md §5.2`.
///
/// Implementors: `i32`, `i64`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
/// `bool` and `usize`/`isize`/`u8` etc. are intentionally excluded.
pub trait EyeElement: Sealed + Element {}

impl EyeElement for i32 {}
impl EyeElement for i64 {}
impl EyeElement for f32 {}
impl EyeElement for f64 {}
impl EyeElement for Complex<f32> {}
impl EyeElement for Complex<f64> {}

impl<A> TensorBase<Owned<A>, Ix2>
where
    A: EyeElement,
{
    /// Create an n×n identity matrix.
    ///
    /// Diagonal elements are 1, all others are 0. F-order layout.
    pub fn eye(n: usize) -> Result<Self, XenonError> {
        let mut result = Self::zeros([n, n])?;
        for i in 0..n {
            // SAFETY: `i < n`, so `[i, i]` is always in-bounds for the validated
            // `[n, n]` shape created above. `eye()` uses unchecked indexing
            // internally and does not rely on the public `IndexMut` panic sugar.
            unsafe {
                *result.get_unchecked_mut(&[i, i]) = A::one();
            }
        }
        Ok(result)
    }
}

/// Compile-time rejection: `bool` does not implement `EyeElement`.
/// ```compile_fail
/// # use xenon::dimension::Ix2;
/// # use xenon::tensor::Tensor;
/// let _ = Tensor::<bool, Ix2>::eye(3);
/// ```
///
/// Compile-time rejection: `usize` does not implement `EyeElement` either
/// (despite implementing `Numeric` — `EyeElement` is strictly closed).
/// ```compile_fail
/// # use xenon::dimension::Ix2;
/// # use xenon::tensor::Tensor;
/// let _ = Tensor::<usize, Ix2>::eye(3);
/// ```
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{InvalidShapeKind, XenonError};
    use crate::tensor::Tensor;

    #[test]
    fn test_eye_3x3() {
        let tensor = Tensor::<i32, Ix2>::eye(3).expect("test input must be valid");
        assert_eq!(*tensor.get(&[0, 0]).expect("test input must be valid"), 1);
        assert_eq!(*tensor.get(&[1, 0]).expect("test input must be valid"), 0);
        assert_eq!(*tensor.get(&[2, 2]).expect("test input must be valid"), 1);
    }

    #[test]
    fn test_eye_zero() {
        // Empty identity matrix: eye(0) produces 0×0 tensor with len=0.
        let tensor = Tensor::<f64, Ix2>::eye(0).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[0, 0]);
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_eye_bool_not_supported() {
        // Verified via compile_fail doc tests above (type-level rejection).
        // The #[test] fn exists as a documentation anchor; the actual enforcement
        // is the `A: EyeElement` trait bound on `eye`.
    }

    #[test]
    fn test_eye_overflow() {
        // n * n overflows checked_size when n approaches usize::MAX.
        let err = Tensor::<i32, Ix2>::eye(usize::MAX).expect_err("usize::MAX overflows");
        assert!(matches!(
            err,
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ProductOverflow,
                ..
            }
        ));
    }
}
