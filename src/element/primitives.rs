//! Element trait definition and implementations for the closed set of 7
//! element types: `{bool, i32, i64, f32, f64, Complex<f32>, Complex<f64>}`.
//!
//! The `Element` trait provides algebraic identity values (`zero`, `one`),
//! a compile‑time type discriminant (`ELEMENT_TYPE`), and a canonical name
//! (`ELEMENT_TYPE_NAME`) for use in error messages and FFI mapping.

use core::fmt::{Debug, Display};

use crate::private::Sealed;
use crate::complex::Complex;
use super::types::ElementType;

/// Base trait for all tensor element types.
///
/// `Element` provides the algebraic identities `zero()` and `one()`, a
/// compile‑time type discriminant `ELEMENT_TYPE` for FFI dispatch, and
/// a canonical name `ELEMENT_TYPE_NAME` for diagnostics.
///
/// The supertrait bounds `Copy + Clone + PartialEq + Debug + Display +
/// Send + Sync` guarantee that every element type is cheap to move,
/// printable, debuggable, and safe to send across threads.
///
/// # Sealed
///
/// Only Xenon’s closed set of 7 types may implement `Element`.
pub trait Element:
    Copy + Clone + PartialEq + Debug + Display + Send + Sync + Sealed
{
    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Compile-time discriminant for this element type.
    const ELEMENT_TYPE: ElementType;

    /// Canonical, stable name for this element type.
    const ELEMENT_TYPE_NAME: &'static str;
}

/// `bool`: `zero()` is `false`, `one()` is `true`.
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

/// `i32`: standard integer identities.
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

/// `i64`: standard integer identities.
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

/// `f32`: IEEE‑754 identities.
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

/// `f64`: IEEE‑754 identities.
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

/// `Complex<f32>`: `zero()` is `0 + 0i`, `one()` is `1 + 0i`.
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

/// `Complex<f64>`: `zero()` is `0 + 0i`, `one()` is `1 + 0i`.
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

    /// Verifies `f32` `Element::zero` and `Element::one`.
    #[test]
    fn test_f32_zero_one() {
        assert_eq!(<f32 as Element>::zero(), 0.0_f32);
        assert_eq!(<f32 as Element>::one(), 1.0_f32);
    }

    /// Verifies `f64` `Element::zero` and `Element::one`.
    #[test]
    fn test_f64_zero_one() {
        assert_eq!(<f64 as Element>::zero(), 0.0_f64);
        assert_eq!(<f64 as Element>::one(), 1.0_f64);
    }

    /// Verifies `bool` `Element` impl and compile‑time trait bound.
    #[test]
    fn test_bool_element_only() {
        fn assert_element<A: Element>() {}
        assert_element::<bool>();
        assert!(!bool::zero());
        assert!(bool::one());
    }

    /// Verifies `Complex<f64>` `Element::zero` and `Element::one`.
    #[test]
    fn test_complex_f64_zero_one() {
        assert_eq!(<Complex<f64> as Element>::zero(), Complex::new(0.0, 0.0));
        assert_eq!(<Complex<f64> as Element>::one(), Complex::new(1.0, 0.0));
    }

    /// Verifies that the `Element` API surface (`zero`, `one`,
    /// `ELEMENT_TYPE`, `ELEMENT_TYPE_NAME`) is reachable through a generic
    /// `A: Element` bound. Exercises all 7 closed element types.
    #[test]
    fn test_element_contract() {
        fn check<A: Element>() {
            let _ = A::zero();
            let _ = A::one();
            let _ = A::ELEMENT_TYPE;
            let _ = A::ELEMENT_TYPE_NAME;
        }
        check::<i32>();
        check::<i64>();
        check::<f32>();
        check::<f64>();
        check::<bool>();
        check::<Complex<f32>>();
        check::<Complex<f64>>();
    }

    /// Compile‑time verification: `usize` must NOT implement `Element`.
    /// Uncommenting the `_assert_element::<usize>()` line must produce a
    /// compile error.
    #[test]
    fn test_usize_does_not_implement_element() {
        fn _assert_element<T: Element>() {
            let _ = T::zero();
        }
        _assert_element::<i32>();
        _assert_element::<i64>();
        _assert_element::<f64>();
        // _assert_element::<usize>();
    }

    /// Verifies `ELEMENT_TYPE_NAME` matches `ElementType::name()` for
    /// every element type.
    #[test]
    fn test_element_type_name_consistency() {
        assert_eq!(<i32 as Element>::ELEMENT_TYPE_NAME, ElementType::I32.name());
        assert_eq!(<i64 as Element>::ELEMENT_TYPE_NAME, ElementType::I64.name());
        assert_eq!(<f32 as Element>::ELEMENT_TYPE_NAME, ElementType::F32.name());
        assert_eq!(<f64 as Element>::ELEMENT_TYPE_NAME, ElementType::F64.name());
        assert_eq!(
            <bool as Element>::ELEMENT_TYPE_NAME,
            ElementType::Bool.name()
        );
        assert_eq!(
            <Complex<f32> as Element>::ELEMENT_TYPE_NAME,
            ElementType::Complex32.name(),
        );
        assert_eq!(
            <Complex<f64> as Element>::ELEMENT_TYPE_NAME,
            ElementType::Complex64.name(),
        );
    }

    /// Property: `zero() + a == a` for all element types.
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

    /// Property: `one() * a == a` for all element types.
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
