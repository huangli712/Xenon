//! Numeric element trait.
//!
//! `Numeric` extends [`Element`](crate::element::Element) with arithmetic
//! operators and a unified conjugate entry point.

use crate::element::Element;

/// Numeric element trait.
///
/// `Numeric` extends [`Element`] with arithmetic operators (`Add`, `Sub`,
/// `Mul`, `Div`, `Neg`) and a unified `conjugate()` entry point. Real and
/// integer types return `self` from `conjugate()`; complex types compute the
/// mathematical conjugate in their own impl blocks.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait Numeric:
    Element
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    /// Returns the conjugate of `self`.
    ///
    /// For real and integer types this is the identity operation.
    fn conjugate(self) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

    /// Verifies that the `Numeric` API surface (arithmetic ops + conjugate)
    /// is well-formed and reachable through a generic `N: Numeric` bound.
    /// Exercises integer, float, and complex implementations.
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
}
