//! Complex scalar trait (§5.4).
//!
//! `ComplexScalar` extends [`Numeric`](crate::element::Numeric) with
//! read-only accessors for complex number components.

use crate::element::{Numeric, RealScalar};
use crate::private::Sealed;

/// Complex scalar trait.
///
/// `ComplexScalar` extends [`Numeric`] with read-only accessors for the
/// real part, imaginary part, and modulus. The conjugate is already
/// provided by [`Numeric::conjugate`] and is not repeated here.
///
/// Only `Complex<f32>` and `Complex<f64>` implement this trait.
pub trait ComplexScalar: Numeric + Sealed {
    /// The real component type; must satisfy [`RealScalar`].
    type Real: RealScalar;

    /// Returns the real part.
    fn re(self) -> Self::Real;

    /// Returns the imaginary part.
    fn im(self) -> Self::Real;

    /// Returns the modulus |z|.
    fn norm(self) -> Self::Real;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;

    /// Verifies that the `ComplexScalar` API surface and associated type
    /// bound are well-formed and reachable through a generic `C: ComplexScalar`
    /// bound. Exercises both concrete implementations (`Complex<f32>` and
    /// `Complex<f64>`).
    #[test]
    fn test_complex_scalar_contract() {
        fn check<C: ComplexScalar>(v: C) -> C::Real {
            let _r: C::Real = v.re();
            let _i: C::Real = v.im();
            v.norm()
        }
        assert_eq!(check(Complex::<f32>::new(3.0, 4.0)), 5.0);
        assert_eq!(check(Complex::<f64>::new(3.0, 4.0)), 5.0);
    }
}
