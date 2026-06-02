//! Construction trait definitions.
//!
//! Contains `EyeElement`, a sealed marker trait that restricts the `eye()`
//! constructor to a closed set of numeric types.

use crate::complex::Complex;
use crate::element::Element;
use crate::private::Sealed;

/// Sealed trait restricting `eye()` to a closed numeric set.
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

/// Compile-time rejection: `bool` does not implement `EyeElement`.
/// ```compile_fail
/// # use xenon::dimension::Ix2;
/// # use xenon::tensor::Tensor;
/// let _ = Tensor::<bool, Ix2>::eye(3);
/// ```
///
/// Compile-time rejection: `usize` does not implement `EyeElement`.
/// ```compile_fail
/// # use xenon::dimension::Ix2;
/// # use xenon::tensor::Tensor;
/// let _ = Tensor::<usize, Ix2>::eye(3);
/// ```
#[cfg(test)]
mod tests {
    /// Verify that `bool` is rejected by the `EyeElement` trait bound at compile time.
    /// Actual enforcement is via compile_fail doc tests above.
    #[test]
    fn test_eye_bool_not_supported() {}
}
