//! Real-valued scalar trait.
//!
//! The `RealScalar` trait exposes a deliberately minimal set of 11 real-valued
//! math functions (`abs`, `signum`, `sqrt`, `sin`, `exp`, `ln`, `floor`, `ceil`,
//! `is_nan`, `is_infinite`, `is_finite`). Adding new methods in future
//! versions constitutes a semi-breaking change (SemVer minor bump): new methods
//! do not break existing downstream user code, but they DO break any external
//! types that `impl RealScalar for MyType` — however, the `Sealed` supertrait
//! prevents any such external impls by design.
//!
//! For extended math functions (e.g., `cos`, `tan`, `log2`, `tanh`), use the
//! crate-internal extension traits in `src/element/real_extended.rs` (if added
//! in future waves). Do not expose those on the public `RealScalar` trait.

use crate::element::Numeric;
use crate::private::Sealed;

/// Real-valued scalar trait.
///
/// `RealScalar` extends [`Numeric`] with IEEE-754 math functions
/// (`abs`, `sqrt`, `sin`, `exp`, `ln`, `floor`, `ceil`), a `signum`
/// discriminant, and predicates for detecting NaN, infinity, and
/// finiteness. Only `f32` and `f64` implement this trait.
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

    /// Square root.
    fn sqrt(self) -> Self;

    /// Sine (radians).
    fn sin(self) -> Self;

    /// Natural exponential function e^x.
    fn exp(self) -> Self;

    /// Natural logarithm ln(x).
    fn ln(self) -> Self;

    /// Largest integer less than or equal to `self`.
    fn floor(self) -> Self;

    /// Smallest integer greater than or equal to `self`.
    fn ceil(self) -> Self;

    /// Returns `true` if `self` is NaN.
    fn is_nan(self) -> bool;

    /// Returns `true` if `self` is positive or negative infinity.
    fn is_infinite(self) -> bool;

    /// Returns `true` if `self` is finite (not NaN and not infinite).
    fn is_finite(self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the `RealScalar` API surface (11 math functions +
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
}
