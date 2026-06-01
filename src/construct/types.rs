use crate::complex::Complex;
use crate::element::Element;
use crate::private::Sealed;

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
    #[test]
    fn test_eye_bool_not_supported() {
        // Verified via compile_fail doc tests above (type-level rejection).
        // The #[test] fn exists as a documentation anchor; the actual enforcement
        // is the `A: EyeElement` trait bound on `eye`.
    }
}
