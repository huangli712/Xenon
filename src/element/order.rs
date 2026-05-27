//! Marker trait for element types that support ordered comparison.
//!
//! Only `i32`, `i64`, `f32`, `f64` implement this trait. Used by
//! `less()`, `greater()`, and related comparison functions.

use crate::element::primitives::Element;
use crate::private::Sealed;

/// Marker trait for element types that support ordered comparison.
///
/// Only `i32`, `i64`, `f32`, `f64` implement this trait.
pub trait OrderedCompareElement: Element + PartialOrd + Sealed {}

impl OrderedCompareElement for i32 {}
impl OrderedCompareElement for i64 {}
impl OrderedCompareElement for f32 {}
impl OrderedCompareElement for f64 {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies OrderedCompareElement trait bounds for concrete types.
    #[test]
    fn test_marker_trait_impls() {
        fn assert_ordered<T: OrderedCompareElement>() {}
        assert_ordered::<i32>();
        assert_ordered::<i64>();
        assert_ordered::<f32>();
        assert_ordered::<f64>();
    }
}
