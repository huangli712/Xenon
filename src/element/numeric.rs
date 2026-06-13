//! Numeric element trait and primitive implementations.
//!
//! `Numeric` extends [`Element`](crate::element::Element) with arithmetic
//! operators (`Add`, `Sub`, `Mul`, `Div`, `Neg`) and a unified `conjugate()`
//! entry point.

use core::ops::{Add, Sub, Mul, Div, Neg};

use crate::complex::Complex;
use super::Element;

/// Numeric element trait.
///
/// `Numeric` extends [`Element`] with all five arithmetic operators and
/// a unified `conjugate()` method. For real and integer types, `conjugate()`
/// is the identity; for `Complex<f32>` and `Complex<f64>`, it returns the
/// mathematical conjugate (negating the imaginary part).
///
/// Implemented by `{i32, i64, f32, f64, Complex<f32>, Complex<f64>}`.
/// `bool` is excluded because it does not form a ring under addition.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait Numeric:
    Element
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Returns the conjugate of `self`.
    ///
    /// For real and integer types this is the identity operation.
    fn conjugate(self) -> Self;
}

impl Numeric for i32 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

impl Numeric for i64 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

impl Numeric for f32 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

impl Numeric for f64 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

/// `Complex<f32>`: conjugate negates the imaginary part.
impl Numeric for Complex<f32> {
    #[inline]
    fn conjugate(self) -> Self {
        self.conj()
    }
}

/// `Complex<f64>`: conjugate negates the imaginary part.
impl Numeric for Complex<f64> {
    #[inline]
    fn conjugate(self) -> Self {
        self.conj()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::element::ComplexScalar;

    /// Verifies that the `Numeric` API surface (all five arithmetic ops
    /// plus `conjugate`) is well‑formed through a generic `N: Numeric`
    /// bound. Exercises integer, float, and complex types.
    #[test]
    fn test_numeric_contract() {
        fn check<N: Numeric>(a: N, b: N) -> N {
            let _add = a + b;
            let _sub = a - b;
            let _mul = a * b;
            let _div = a / b;
            let _neg = -a;
            a.conjugate()
        }
        assert_eq!(check(10i32, 2), 10);
        assert_eq!(check(10i64, 2), 10);
        assert_eq!(check(10.0f32, 2.0f32), 10.0f32);
        assert_eq!(check(10.0f64, 2.0), 10.0);
        assert_eq!(
            check(Complex::<f64>::new(3.0, 4.0), Complex::new(1.0, 2.0)),
            Complex::new(3.0, -4.0),
        );
    }

    /// Verifies `i32` `Element::zero`, `Element::one`, and
    /// `Numeric::conjugate`.
    #[test]
    fn test_i32_zero_one() {
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
        assert_eq!(<i32 as Numeric>::conjugate(-7), -7);
    }

    /// Verifies `i32` arithmetic operators.
    #[test]
    fn test_i32_arithmetic() {
        let a = 10i32;
        let b = 3i32;
        assert_eq!(a + b, 13);
        assert_eq!(a - b, 7);
        assert_eq!(a * b, 30);
        assert_eq!(a / b, 3);
        assert_eq!(-a, -10);
    }

    /// Verifies `i64` `Element` and `Numeric` trait methods.
    #[test]
    fn test_i64_zero_one() {
        assert_eq!(i64::zero(), 0);
        assert_eq!(i64::one(), 1);
        assert_eq!(40i64 + 2, 42);
        assert_eq!(<i64 as Element>::zero(), 0);
        assert_eq!(<i64 as Element>::one(), 1);
        assert_eq!(<i64 as Numeric>::conjugate(-7), -7);
        assert_eq!(<i64 as Numeric>::conjugate(7), 7);
    }

    /// Verifies `Complex<f64>` `Numeric::conjugate`,
    /// `ComplexScalar::norm`, and `Element::zero`.
    #[test]
    fn test_complex_f64_conj_and_norm() {
        let value = Complex::new(3.0f64, 4.0f64);
        assert_eq!(
            <Complex<f64> as Numeric>::conjugate(value),
            Complex::new(3.0, -4.0)
        );
        assert_eq!(<Complex<f64> as ComplexScalar>::norm(value), 5.0);
        assert_eq!(Complex::<f64>::zero(), Complex::new(0.0, 0.0));
    }

    /// Verifies `conjugate()` concrete return values for `f32`, `f64`, and
    /// `Complex<f32>`: identity on the real types and imaginary-part
    /// negation on the complex type.
    #[test]
    fn test_conjugate_concrete_values() {
        assert_eq!(<f32 as Numeric>::conjugate(3.5_f32), 3.5_f32);
        assert_eq!(<f64 as Numeric>::conjugate(-2.25_f64), -2.25_f64);
        assert_eq!(
            <Complex<f32> as Numeric>::conjugate(Complex::new(3.0, 4.0)),
            Complex::new(3.0, -4.0)
        );
    }
}
