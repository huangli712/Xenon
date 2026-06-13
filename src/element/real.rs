//! Real-valued scalar trait and primitive implementations.
//!
//! `RealScalar` exposes 8 IEEE‑754 math functions plus 3 predicates, sealed
//! to `f32` and `f64` only.

use crate::private::Sealed;
use super::Numeric;

/// Real-valued scalar trait.
///
/// `RealScalar` extends [`Numeric`] with IEEE‑754 math functions
/// (`abs`, `signum`, `sqrt`, `sin`, `exp`, `ln`, `floor`, `ceil`)
/// and predicates for detecting NaN, infinity, and finiteness.
///
/// Only `f32` and `f64` implement this trait. Integer types are excluded
/// because the IEEE‑754 math methods (`sqrt`, `sin`, `exp`, `ln`, ...) have
/// no integer equivalent. `Complex<f32>` / `Complex<f64>` are excluded by the
/// `PartialOrd` supertrait bound, which complex numbers do not implement.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait RealScalar: Numeric + PartialOrd + Sealed {
    /// Absolute value.
    fn abs(self) -> Self;

    /// Signum: `1.0` for positive, `-1.0` for negative, `0.0` for zero,
    /// `NaN` for NaN. Follows `f32` / `f64` semantics.
    fn signum(self) -> Self;

    /// Square root. Returns NaN for negative inputs.
    fn sqrt(self) -> Self;

    /// Sine (radians).
    fn sin(self) -> Self;

    /// Natural exponential function eˣ.
    fn exp(self) -> Self;

    /// Natural logarithm ln(x). Returns -infinity for 0, NaN for negative.
    fn ln(self) -> Self;

    /// Largest integer ≤ `self`.
    fn floor(self) -> Self;

    /// Smallest integer ≥ `self`.
    fn ceil(self) -> Self;

    /// Returns `true` if `self` is NaN.
    fn is_nan(self) -> bool;

    /// Returns `true` if `self` is positive or negative infinity.
    fn is_infinite(self) -> bool;

    /// Returns `true` if `self` is finite (not NaN and not infinite).
    fn is_finite(self) -> bool;
}

impl RealScalar for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }

    fn signum(self) -> Self {
        f32::signum(self)
    }
    
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    
    fn sin(self) -> Self {
        f32::sin(self)
    }
    
    fn exp(self) -> Self {
        f32::exp(self)
    }
    
    fn ln(self) -> Self {
        f32::ln(self)
    }
    
    fn floor(self) -> Self {
        f32::floor(self)
    }
    
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    
    fn is_infinite(self) -> bool {
        f32::is_infinite(self)
    }
    
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl RealScalar for f64 {
    fn abs(self) -> Self {
        f64::abs(self)
    }
   
    fn signum(self) -> Self {
        f64::signum(self)
    }
    
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    
    fn sin(self) -> Self {
        f64::sin(self)
    }
    
    fn exp(self) -> Self {
        f64::exp(self)
    }
    
    fn ln(self) -> Self {
        f64::ln(self)
    }
    
    fn floor(self) -> Self {
        f64::floor(self)
    }
    
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    
    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }
    
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Element;

    /// Verifies that the `RealScalar` API surface (8 math functions +
    /// 3 IEEE‑754 predicates) is well-formed and reachable through a
    /// generic `R: RealScalar` bound. Exercises both `f32` and `f64`.
    #[test]
    fn test_real_scalar_contract() {
        fn check<R: RealScalar>(v: R) {
            let _ = <R as RealScalar>::abs(v);
            let _ = <R as RealScalar>::signum(v);
            let _ = <R as RealScalar>::sqrt(v);
            let _ = <R as RealScalar>::sin(v);
            let _ = <R as RealScalar>::exp(v);
            let _ = <R as RealScalar>::ln(v);
            let _ = <R as RealScalar>::floor(v);
            let _ = <R as RealScalar>::ceil(v);
            let _: bool = <R as RealScalar>::is_nan(v);
            let _: bool = <R as RealScalar>::is_infinite(v);
            let _: bool = <R as RealScalar>::is_finite(v);
        }
        check(1.0f32);
        check(1.0f64);
    }

    /// Quick smoke test exercising `Element` and `RealScalar` methods
    /// together.
    #[test]
    fn test_f32_f64_real_scalar() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(<f64 as RealScalar>::sqrt(9.0), 3.0);
        assert!(<f32 as RealScalar>::is_nan(f32::NAN));
    }

    /// Verifies `RealScalar::sqrt` for `f64`.
    #[test]
    fn test_f64_sqrt() {
        assert_eq!(<f64 as RealScalar>::sqrt(4.0), 2.0);
        assert_eq!(<f64 as RealScalar>::sqrt(9.0), 3.0);
    }

    /// Verifies `RealScalar::sin` for `f64` at zero.
    #[test]
    fn test_f64_sin() {
        assert_eq!(<f64 as RealScalar>::sin(0.0), 0.0);
    }

    /// Verifies `f32` IEEE‑754 predicates: is_nan, is_infinite, is_finite.
    #[test]
    fn test_f32_nan_detection() {
        assert!(<f32 as RealScalar>::is_nan(f32::NAN));
        assert!(!<f32 as RealScalar>::is_nan(1.0f32));
        assert!(<f32 as RealScalar>::is_infinite(f32::INFINITY));
        assert!(<f32 as RealScalar>::is_finite(1.0f32));
    }

    /// Verifies exp(ln(x)) ≈ x (round‑trip identity) for `f64`.
    #[test]
    fn test_f64_exp_ln_inverse() {
        let tolerance = 1e-12_f64;
        for x in [1.0_f64, 2.0, std::f64::consts::E, 10.0, 100.0] {
            let round_trip = <f64 as RealScalar>::exp(<f64 as RealScalar>::ln(x));
            assert!(
                (round_trip - x).abs() < tolerance * x.max(1.0),
                "exp(ln({})) = {}, expected approx {}",
                x,
                round_trip,
                x,
            );
        }
    }

    /// Boundary: `is_nan` is `true` for NaN, `false` otherwise.
    #[test]
    fn test_boundary_f64_nan_is_nan() {
        assert!(<f64 as RealScalar>::is_nan(f64::NAN));
        assert!(!<f64 as RealScalar>::is_nan(1.0_f64));
    }

    /// Boundary: `is_finite` is `false` for infinity.
    #[test]
    fn test_boundary_f64_infinity_is_not_finite() {
        assert!(!<f64 as RealScalar>::is_finite(f64::INFINITY));
        assert!(<f64 as RealScalar>::is_finite(1.0_f64));
    }

    /// Boundary: `sqrt` of a negative number returns NaN.
    #[test]
    fn test_boundary_f64_sqrt_neg_is_nan() {
        assert!(<f64 as RealScalar>::is_nan(<f64 as RealScalar>::sqrt(-1.0)));
    }

    /// Boundary: `ln(0.0)` returns negative infinity.
    #[test]
    fn test_boundary_f64_ln_zero_is_neg_infinity() {
        let v = <f64 as RealScalar>::ln(0.0);
        assert!(<f64 as RealScalar>::is_infinite(v));
        assert!(v < 0.0);
    }

    /// Property: sqrt(a)² ≈ a for `f32` and `f64`.
    #[test]
    fn test_property_sqrt_square_inverse() {
        let tol64 = 1e-10_f64;
        for a in [0.0_f64, 0.25, 1.0, 2.0, 9.0, 100.0] {
            let y = <f64 as RealScalar>::sqrt(a);
            assert!((y * y - a).abs() < tol64 * a.max(1.0));
        }
        let tol32 = 1e-5_f32;
        for a in [0.0_f32, 0.25, 1.0, 9.0] {
            let y = <f32 as RealScalar>::sqrt(a);
            assert!((y * y - a).abs() < tol32 * a.max(1.0));
        }
    }

    /// Property: ln(exp(a)) ≈ a for `f32` and `f64`.
    #[test]
    fn test_property_exp_ln_inverse() {
        let tol64 = 1e-10_f64;
        for a in [-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            let round = <f64 as RealScalar>::ln(<f64 as RealScalar>::exp(a));
            assert!((round - a).abs() < tol64 * a.abs().max(1.0));
        }
        let tol32 = 1e-5_f32;
        for a in [-2.0_f32, 0.0, 2.0] {
            let round = <f32 as RealScalar>::ln(<f32 as RealScalar>::exp(a));
            assert!((round - a).abs() < tol32 * a.abs().max(1.0));
        }
    }

    /// Property: exp(ln(x)) ≈ x for `f32` and `f64`.
    #[test]
    fn test_property_ln_exp_inverse() {
        let tol64 = 1e-10_f64;
        for x in [0.5_f64, 1.0, std::f64::consts::E, 10.0] {
            let round = <f64 as RealScalar>::exp(<f64 as RealScalar>::ln(x));
            assert!((round - x).abs() < tol64 * x.max(1.0));
        }
        let tol32 = 1e-5_f32;
        for x in [0.5_f32, 1.0, 10.0] {
            let round = <f32 as RealScalar>::exp(<f32 as RealScalar>::ln(x));
            assert!((round - x).abs() < tol32 * x.max(1.0));
        }
    }

    /// Verifies `RealScalar::abs` for `f32`.
    #[test]
    fn test_f32_abs() {
        assert_eq!(<f32 as RealScalar>::abs(-3.5_f32), 3.5_f32);
    }

    /// Verifies `RealScalar::signum` for `f32`.
    #[test]
    fn test_f32_signum() {
        assert_eq!(<f32 as RealScalar>::signum(5.0_f32), 1.0_f32);
        assert_eq!(<f32 as RealScalar>::signum(-3.0_f32), -1.0_f32);
    }

    /// Verifies `RealScalar::sin` for `f32` at zero.
    #[test]
    fn test_f32_sin() {
        let val = <f32 as RealScalar>::sin(0.0_f32);
        assert!((val - 0.0_f32).abs() < 1e-6_f32);
    }

    /// Verifies `RealScalar::floor` for `f32`.
    #[test]
    fn test_f32_floor() {
        assert_eq!(<f32 as RealScalar>::floor(3.7_f32), 3.0_f32);
    }

    /// Verifies `RealScalar::ceil` for `f32`.
    #[test]
    fn test_f32_ceil() {
        assert_eq!(<f32 as RealScalar>::ceil(2.3_f32), 3.0_f32);
    }

    /// Exercises several `f64` math methods on a well‑defined input.
    #[test]
    fn test_real_scalar_boundary_methods() {
        let value = 4.0f64;
        assert_eq!(<f64 as RealScalar>::floor(value), 4.0);
        assert_eq!(<f64 as RealScalar>::ceil(value), 4.0);
        assert!(<f64 as RealScalar>::is_finite(value));
        assert_eq!(<f64 as RealScalar>::abs(-4.0), 4.0);
        assert_eq!(<f64 as RealScalar>::sqrt(9.0), 3.0);
        assert_eq!(<f64 as RealScalar>::sin(0.0), 0.0);
    }

    /// Compile‑time: verifies `RealScalar` trait bounds for both
    /// `f32` and `f64`.
    #[test]
    fn test_compile_positive_real_bounds() {
        fn assert_real<A: RealScalar>() {}
        assert_real::<f32>();
        assert_real::<f64>();
    }
}
