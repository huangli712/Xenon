//! `CastElement` sealed marker trait for the type conversion system.
//!
//! Defines the public `CastElement` trait that gates `cast()` at compile time,
//! restricting type conversion to the 6 numeric types in Xenon's element set.

use crate::complex::Complex;
use crate::element::Element;

/// Public sealed marker for element types in the cast matrix.
///
/// `cast()` public signature `where A: CastElement, T: CastElement` uses this
/// trait to exclude `bool` from conversion at compile time and narrow the
/// element set to the 6 numeric types.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait CastElement: Element {}

impl CastElement for i32 {}
impl CastElement for i64 {}
impl CastElement for f32 {}
impl CastElement for f64 {}
impl CastElement for Complex<f32> {}
impl CastElement for Complex<f64> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_element_trait_signature_accepts_cast_elements() {
        fn require_cast_element<A: CastElement>() {}
        require_cast_element::<i32>();
        require_cast_element::<i64>();
        require_cast_element::<f32>();
        require_cast_element::<f64>();
        require_cast_element::<Complex<f32>>();
        require_cast_element::<Complex<f64>>();
    }
}
