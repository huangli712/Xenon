//! `SealedElement` sealed marker trait for numeric-only tensor operations.
//!
//! `SealedElement` is the shared compile-time gate for operations that are
//! defined over Xenon's six numeric element types but are **not** meaningful
//! for `bool`:
//!
//! ```text
//! {i32, i64, f32, f64, Complex<f32>, Complex<f64>}
//! ```
//!
//! # Relationship to `Numeric`
//!
//! `SealedElement` covers the same six types as `Numeric`, yet carries a
//! deliberately weaker contract: it does **not** require the arithmetic
//! operators (`Add`/`Sub`/`Mul`/`Div`/`Neg`). The operations gated by this
//! trait only need "a numeric element type that is not `bool`", never ring
//! arithmetic. Binding them to `Numeric` would over-constrain the public
//! signature and misstate intent, so the marker is kept independent.
//!
//! # Application scenarios
//!
//! Reach for `SealedElement` as the bound whenever an operation must accept
//! the full numeric set yet exclude `bool` at compile time, without depending
//! on arithmetic. Current consumers:
//!
//! | Operation       | Entry point          |
//! |-----------------|----------------------|
//! | Identity matrix | `TensorBase::eye`    |
//! | Deduplication   | `TensorBase::unique` |
//! | Type conversion | `TensorBase::cast`   |
//!
//! Future numeric-only, arithmetic-free operations (for example additional
//! set operations like `intersect`/`union`, or histogram/value-table builders)
//! should reuse this bound rather than re-declaring yet another identical
//! marker trait.
//!
//! # Sealed
//!
//! `SealedElement: Element` and `Element: Sealed`, so this trait is sealed
//! transitively and cannot be implemented outside of `Xenon`.

use super::Element;
use crate::complex::Complex;
use crate::private::Sealed;

/// Sealed marker for the six numeric element types, excluding `bool`.
///
/// Implementors: `i32`, `i64`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
/// `bool` is intentionally excluded; `usize`/`isize` and other primitives are
/// not `Element` types and therefore cannot implement this trait either.
///
/// This is a shared compile-time gate for numeric-only operations that do not
/// need arithmetic operators (`eye`, `unique`, `cast`).
///
/// # Sealed
///
/// This trait is sealed and cannot be implemented outside of `Xenon`.
pub trait SealedElement: Element + Sealed {}

impl SealedElement for i32 {}
impl SealedElement for i64 {}
impl SealedElement for f32 {}
impl SealedElement for f64 {}
impl SealedElement for Complex<f32> {}
impl SealedElement for Complex<f64> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time verification that all 6 numeric element types
    /// implement the `SealedElement` marker trait.
    #[test]
    fn test_marker_trait_impls() {
        fn assert_sealed<T: SealedElement>() {}
        assert_sealed::<i32>();
        assert_sealed::<i64>();
        assert_sealed::<f32>();
        assert_sealed::<f64>();
        assert_sealed::<Complex<f32>>();
        assert_sealed::<Complex<f64>>();
    }
}
