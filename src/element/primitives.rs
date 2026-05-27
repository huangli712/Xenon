//! Element trait implementations for the 7 closed element types.
//!
//! Implements `Element` for the standard numeric types.

use crate::complex::Complex;
use crate::element::{Element, ElementType};

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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Verifies bool Element impl: zero, one, and trait bound.
    #[test]
    fn test_bool_element_only() {
        fn assert_element<A: Element>() {}
        assert_element::<bool>();
        assert!(!bool::zero());
        assert!(bool::one());
    }

    /// Verifies Complex&lt;f64&gt; Element::zero and Element::one.
    #[test]
    fn test_complex_f64_zero_one() {
        assert_eq!(<Complex<f64> as Element>::zero(), Complex::new(0.0, 0.0));
        assert_eq!(<Complex<f64> as Element>::one(), Complex::new(1.0, 0.0));
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
}
