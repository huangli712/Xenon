//! Ordered comparison marker trait.
//!
//! `OrderedCompareElement` constrains comparison operations (`less`,
//! `greater`, `less_equal`, `greater_equal`) to the closed set `{i32,
//! i64, f32, f64}`. `Complex<f32>` and `Complex<f64>` are excluded at
//! compile time because they do not implement `PartialOrd`.

use crate::private::Sealed;
use super::Element;

/// Marker trait for element types that support ordered comparison.
///
/// `OrderedCompareElement` is sealed to `{i32, i64, f32, f64}` via
/// `PartialOrd` — the only supertrait that `Complex<f32>` and
/// `Complex<f64>` lack. This allows comparison functions to use a
/// single generic bound while excluding complex types at compile time.
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait OrderedCompareElement: Element + PartialOrd + Sealed {}

impl OrderedCompareElement for i32 {}
impl OrderedCompareElement for i64 {}
impl OrderedCompareElement for f32 {}
impl OrderedCompareElement for f64 {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time verification that `{i32, i64, f32, f64}` satisfy
    /// the `OrderedCompareElement` trait.
    #[test]
    fn test_marker_trait_impls() {
        fn assert_ordered<T: OrderedCompareElement>() {}
        assert_ordered::<i32>();
        assert_ordered::<i64>();
        assert_ordered::<f32>();
        assert_ordered::<f64>();
    }
}
