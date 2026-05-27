//! Element trait implementations for the 7 closed element types.
//!
//! Implements `Sealed`, `Element`, `Numeric`, `RealScalar`, and
//! `ComplexScalar` for the standard numeric types.

use crate::complex::Complex;
use crate::element::{
    ComplexScalar, Element, ElementType, Numeric, RealScalar,
};
use crate::private::Sealed;

impl Element for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    const ELEMENT_TYPE: ElementType = ElementType::I32;
    const ELEMENT_TYPE_NAME: &'static str = "i32";
}

impl Numeric for i32 {
    fn conjugate(self) -> Self {
        self
    }
}

impl Element for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    const ELEMENT_TYPE: ElementType = ElementType::I64;
    const ELEMENT_TYPE_NAME: &'static str = "i64";
}

impl Numeric for i64 {
    fn conjugate(self) -> Self {
        self
    }
}

impl Element for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    const ELEMENT_TYPE: ElementType = ElementType::F32;
    const ELEMENT_TYPE_NAME: &'static str = "f32";
}

impl Numeric for f32 {
    fn conjugate(self) -> Self {
        self
    }
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

impl Element for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    const ELEMENT_TYPE: ElementType = ElementType::F64;
    const ELEMENT_TYPE_NAME: &'static str = "f64";
}

impl Numeric for f64 {
    fn conjugate(self) -> Self {
        self
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

impl Sealed for bool {}

impl Element for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
    const ELEMENT_TYPE: ElementType = ElementType::Bool;
    const ELEMENT_TYPE_NAME: &'static str = "bool";
}

impl Sealed for Complex<f32> {}

impl Element for Complex<f32> {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    const ELEMENT_TYPE: ElementType = ElementType::Complex32;
    const ELEMENT_TYPE_NAME: &'static str = "Complex<f32>";
}

impl Numeric for Complex<f32> {
    fn conjugate(self) -> Self {
        self.conj()
    }
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

impl Sealed for Complex<f64> {}

impl Element for Complex<f64> {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    const ELEMENT_TYPE: ElementType = ElementType::Complex64;
    const ELEMENT_TYPE_NAME: &'static str = "Complex<f64>";
}

impl Numeric for Complex<f64> {
    fn conjugate(self) -> Self {
        self.conj()
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

    /// Exercises i32 Element::zero, Element::one, and Numeric::conjugate.
    #[test]
    fn test_i32_zero_one() {
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
        assert_eq!(<i32 as Numeric>::conjugate(-7), -7);
    }

    /// Verifies i32 arithmetic operators (add, sub, mul, div, neg).
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

    /// Exercises i64 Element and Numeric trait methods.
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

    /// Exercises f32 and f64 Element and RealScalar trait methods.
    #[test]
    fn test_f32_f64_real_scalar() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(<f64 as RealScalar>::sqrt(9.0), 3.0);
        assert!(<f32 as RealScalar>::is_nan(f32::NAN));
    }

    /// Verifies f32 Element::zero and Element::one.
    #[test]
    fn test_f32_zero_one() {
        assert_eq!(<f32 as Element>::zero(), 0.0_f32);
        assert_eq!(<f32 as Element>::one(), 1.0_f32);
    }

    /// Verifies f64 Element::zero and Element::one.
    #[test]
    fn test_f64_zero_one() {
        assert_eq!(<f64 as Element>::zero(), 0.0_f64);
        assert_eq!(<f64 as Element>::one(), 1.0_f64);
    }

    /// Verifies RealScalar::sqrt for f64.
    #[test]
    fn test_f64_sqrt() {
        assert_eq!(<f64 as RealScalar>::sqrt(4.0), 2.0);
        assert_eq!(<f64 as RealScalar>::sqrt(9.0), 3.0);
    }

    /// Verifies RealScalar::sin for f64 at zero.
    #[test]
    fn test_f64_sin() {
        assert_eq!(<f64 as RealScalar>::sin(0.0), 0.0);
    }

    /// Verifies f32 IEEE-754 predicates: is_nan, is_infinite, is_finite.
    #[test]
    fn test_f32_nan_detection() {
        assert!(<f32 as RealScalar>::is_nan(f32::NAN));
        assert!(!<f32 as RealScalar>::is_nan(1.0f32));
        assert!(<f32 as RealScalar>::is_infinite(f32::INFINITY));
        assert!(<f32 as RealScalar>::is_finite(1.0f32));
    }

    /// Verifies exp(ln(x)) ≈ x (round-trip identity) for f64.
    #[test]
    fn test_f64_exp_ln_inverse() {
        let tolerance = 1e-12_f64;
        for x in [1.0_f64, 2.0, std::f64::consts::E, 10.0, 100.0] {
            let round_trip = <f64 as RealScalar>::exp(<f64 as RealScalar>::ln(x));
            assert!(
                (round_trip - x).abs() < tolerance * x.max(1.0),
                "exp(ln({})) = {}, expected ≈ {}",
                x,
                round_trip,
                x,
            );
        }
    }

    /// Verifies bool Element impl: zero, one, and trait bound.
    #[test]
    fn test_bool_element_only() {
        fn assert_element<A: Element>() {}
        assert_element::<bool>();
        assert!(!bool::zero());
        assert!(bool::one());
    }

    /// Verifies Complex&lt;f64&gt; Numeric::conjugate, ComplexScalar::norm,
    /// and Element::zero.
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

    /// Verifies Complex&lt;f64&gt; Element::zero and Element::one.
    #[test]
    fn test_complex_f64_zero_one() {
        assert_eq!(<Complex<f64> as Element>::zero(), Complex::new(0.0, 0.0));
        assert_eq!(<Complex<f64> as Element>::one(), Complex::new(1.0, 0.0));
    }

    /// Verifies ComplexScalar::norm for Complex&lt;f32&gt;.
    #[test]
    fn test_complex_f32_norm() {
        let c = Complex::<f32>::new(3.0, 4.0);
        assert_eq!(<Complex<f32> as ComplexScalar>::norm(c), 5.0);
    }

    /// Boundary: RealScalar::is_nan for f64 NaN.
    #[test]
    fn test_boundary_f64_nan_is_nan() {
        assert!(<f64 as RealScalar>::is_nan(f64::NAN));
        assert!(!<f64 as RealScalar>::is_nan(1.0_f64));
    }

    /// Boundary: RealScalar::is_finite for f64 infinity.
    #[test]
    fn test_boundary_f64_infinity_is_not_finite() {
        assert!(!<f64 as RealScalar>::is_finite(f64::INFINITY));
        assert!(<f64 as RealScalar>::is_finite(1.0_f64));
    }

    /// Boundary: RealScalar::sqrt of negative f64 returns NaN.
    #[test]
    fn test_boundary_f64_sqrt_neg_is_nan() {
        assert!(<f64 as RealScalar>::is_nan(<f64 as RealScalar>::sqrt(-1.0)));
    }

    /// Boundary: RealScalar::ln(0.0) returns negative infinity.
    #[test]
    fn test_boundary_f64_ln_zero_is_neg_infinity() {
        let v = <f64 as RealScalar>::ln(0.0);
        assert!(<f64 as RealScalar>::is_infinite(v));
        assert!(v < 0.0);
    }

    /// Boundary: ComplexScalar::norm with NaN component returns NaN.
    #[test]
    fn test_boundary_complex_nan_norm_is_nan() {
        let c = Complex::<f64>::new(f64::NAN, 0.0);
        let n = <Complex<f64> as ComplexScalar>::norm(c);
        assert!(<f64 as RealScalar>::is_nan(n));
    }

    /// Property: zero() + a == a for all element types.
    #[test]
    fn test_property_zero_additive_identity() {
        for a in [-7_i32, 0, 42] {
            assert_eq!(<i32 as Element>::zero() + a, a);
        }
        for a in [-7_i64, 0, 42] {
            assert_eq!(<i64 as Element>::zero() + a, a);
        }
        for a in [-3.5_f32, 0.0, 7.25] {
            assert_eq!(<f32 as Element>::zero() + a, a);
        }
        for a in [-3.5_f64, 0.0, 7.25] {
            assert_eq!(<f64 as Element>::zero() + a, a);
        }
        let cz = <Complex<f64> as Element>::zero();
        for a in [Complex::<f64>::new(1.0, 2.0), Complex::new(-3.0, 4.0)] {
            assert_eq!(cz + a, a);
        }
    }

    /// Property: one() * a == a for all element types.
    #[test]
    fn test_property_one_multiplicative_identity() {
        for a in [-7_i32, 0, 42] {
            assert_eq!(<i32 as Element>::one() * a, a);
        }
        for a in [-3.5_f64, 0.0, 7.25] {
            assert_eq!(<f64 as Element>::one() * a, a);
        }
        let co = <Complex<f64> as Element>::one();
        for a in [Complex::<f64>::new(1.0, 2.0), Complex::new(-3.0, 4.0)] {
            assert_eq!(co * a, a);
        }
    }

    /// Property: sqrt(a)² ≈ a for f32 and f64.
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

    /// Property: ln(exp(a)) ≈ a for f32 and f64.
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

    /// Property: exp(ln(x)) ≈ x for f32 and f64.
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

    /// Verifies RealScalar::abs for f32.
    #[test]
    fn test_f32_abs() {
        assert_eq!(<f32 as RealScalar>::abs(-3.5_f32), 3.5_f32);
    }

    /// Verifies RealScalar::signum for f32.
    #[test]
    fn test_f32_signum() {
        assert_eq!(<f32 as RealScalar>::signum(5.0_f32), 1.0_f32);
        assert_eq!(<f32 as RealScalar>::signum(-3.0_f32), -1.0_f32);
    }

    /// Verifies RealScalar::sin for f32 at zero.
    #[test]
    fn test_f32_sin() {
        let val = <f32 as RealScalar>::sin(0.0_f32);
        assert!((val - 0.0_f32).abs() < 1e-6_f32);
    }

    /// Verifies RealScalar::floor for f32.
    #[test]
    fn test_f32_floor() {
        assert_eq!(<f32 as RealScalar>::floor(3.7_f32), 3.0_f32);
    }

    /// Verifies RealScalar::ceil for f32.
    #[test]
    fn test_f32_ceil() {
        assert_eq!(<f32 as RealScalar>::ceil(2.3_f32), 3.0_f32);
    }

    /// Exercises key f64 math methods and verifies they match primitive
    /// counterparts for well-defined inputs.
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

    /// Compile-time: verifies Numeric, RealScalar, and ComplexScalar
    /// trait bounds for all types.
    #[test]
    fn test_compile_positive_trait_bounds() {
        fn assert_numeric<A: Numeric>() {}
        fn assert_real<A: RealScalar>() {}
        fn assert_complex<A: ComplexScalar>() {}
        assert_numeric::<i32>();
        assert_numeric::<i64>();
        assert_real::<f32>();
        assert_real::<f64>();
        assert_complex::<Complex<f32>>();
        assert_complex::<Complex<f64>>();
    }
}
