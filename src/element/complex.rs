//! Complex scalar trait and primitive implementations.
//!
//! `ComplexScalar` extends [`Numeric`](crate::element::Numeric) with
//! read-only accessors for complex number components (`re`, `im`,
//! `norm`). The conjugate is already provided by [`Numeric::conjugate`]
//! and is not repeated here.

use crate::complex::Complex;
use super::{Numeric, RealScalar};
use crate::private::Sealed;

/// Complex scalar trait.
///
/// `ComplexScalar` extends [`Numeric`] with read-only accessors for the
/// real part, imaginary part, and modulus.
///
/// Only `Complex<f32>` and `Complex<f64>` implement this trait. The
/// associated type `Real` maps `Complex<f32>` → `f32` and
/// `Complex<f64>` → `f64`, both satisfying [`RealScalar`].
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

    /// Returns the modulus |z| = √(re² + im²).
    fn norm(self) -> Self::Real;
}

// ── ComplexScalar impls ───────────────────────────────────────────────────

/// `Complex<f32>`: `Real = f32`, delegates to the inner `Complex` field.
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

/// `Complex<f64>`: `Real = f64`, delegates to the inner `Complex` field.
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

    /// Verifies that the `ComplexScalar` API surface (re, im, norm) and
    /// associated type `Real` are well-formed through a generic
    /// `C: ComplexScalar` bound. Exercises both `Complex<f32>` and
    /// `Complex<f64>`.
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

    /// Verifies `ComplexScalar::norm` for `Complex<f32>` with the
    /// classic 3-4-5 right triangle.
    #[test]
    fn test_complex_f32_norm() {
        let c = Complex::<f32>::new(3.0, 4.0);
        assert_eq!(<Complex<f32> as ComplexScalar>::norm(c), 5.0);
    }

    /// Boundary: `norm` propagates NaN from any component.
    #[test]
    fn test_boundary_complex_nan_norm_is_nan() {
        let c = Complex::<f64>::new(f64::NAN, 0.0);
        let n = <Complex<f64> as ComplexScalar>::norm(c);
        assert!(f64::is_nan(n));
    }

    /// Compile-time: verifies `ComplexScalar` trait bounds for both
    /// concrete complex types.
    #[test]
    fn test_compile_positive_trait_bounds() {
        fn assert_complex<A: ComplexScalar>() {}
        assert_complex::<Complex<f32>>();
        assert_complex::<Complex<f64>>();
    }
}
