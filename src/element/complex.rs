//! Complex scalar trait.
//!
//! `ComplexScalar` extends [`Numeric`](crate::element::Numeric) with
//! read-only accessors for complex number components.

use crate::complex::Complex;
use crate::element::{Numeric, RealScalar};
use crate::private::Sealed;

/// Complex scalar trait.
///
/// `ComplexScalar` extends [`Numeric`] with read-only accessors for the
/// real part, imaginary part, and modulus. The conjugate is already
/// provided by [`Numeric::conjugate`] and is not repeated here.
///
/// Only `Complex<f32>` and `Complex<f64>` implement this trait.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
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

impl ComplexScalar for Complex<f32> {
    type Real = f32;
    fn re(self) -> f32 {
        self.re
    }
    fn im(self) -> f32 {
        self.im
    }
    fn norm(self) -> f32 {
        self.norm()
    }
}

impl ComplexScalar for Complex<f64> {
    type Real = f64;
    fn re(self) -> f64 {
        self.re
    }
    fn im(self) -> f64 {
        self.im
    }
    fn norm(self) -> f64 {
        self.norm()
    }
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

    /// Verifies ComplexScalar::norm for Complex&lt;f32&gt;.
    #[test]
    fn test_complex_f32_norm() {
        let c = Complex::<f32>::new(3.0, 4.0);
        assert_eq!(<Complex<f32> as ComplexScalar>::norm(c), 5.0);
    }

    /// Boundary: ComplexScalar::norm with NaN component returns NaN.
    #[test]
    fn test_boundary_complex_nan_norm_is_nan() {
        let c = Complex::<f64>::new(f64::NAN, 0.0);
        let n = <Complex<f64> as ComplexScalar>::norm(c);
        assert!(f64::is_nan(n));
    }

    /// Compile-time: verifies ComplexScalar trait bounds.
    #[test]
    fn test_compile_positive_trait_bounds() {
        fn assert_complex<A: ComplexScalar>() {}
        assert_complex::<Complex<f32>>();
        assert_complex::<Complex<f64>>();
    }
}
