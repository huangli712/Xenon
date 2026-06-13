//! Checked arithmetic traits for integer types.
//!
//! Each trait wraps a standard library `checked_*` method and returns
//! `Option` — `None` signals overflow or division-by-zero. Float and
//! complex types are intentionally excluded; they use ordinary operators
//! with NaN propagation instead.
//!
//! # Sealed
//!
//! Only `i32` and `i64` implement these traits. The `Sealed` supertrait
//! prevents external implementations.

use crate::private::Sealed;
use super::Numeric;

// ── CheckedAdd ────────────────────────────────────────────────────────────

/// Checked addition for integer types.
///
/// Returns `None` on overflow instead of wrapping. Callers translate
/// `None` to a panic with diagnostic context (element index, shape).
///
/// Only `i32` and `i64` implement this trait.
pub(crate) trait CheckedAdd: Numeric + Sealed {
    /// Returns `Some(self + rhs)` if no overflow, `None` otherwise.
    fn checked_add(self, rhs: Self) -> Option<Self>;
}

impl CheckedAdd for i32 {
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        i32::checked_add(self, rhs)
    }
}

impl CheckedAdd for i64 {
    #[inline]
    fn checked_add(self, rhs: Self) -> Option<Self> {
        i64::checked_add(self, rhs)
    }
}

// ── CheckedSub ────────────────────────────────────────────────────────────

/// Checked subtraction for integer types.
///
/// Returns `None` on overflow instead of wrapping. Used by element-wise
/// subtraction and related reductions.
///
/// Only `i32` and `i64` implement this trait.
pub(crate) trait CheckedSub: Numeric + Sealed {
    /// Returns `Some(self - rhs)` if no overflow, `None` otherwise.
    fn checked_sub(self, rhs: Self) -> Option<Self>;
}

impl CheckedSub for i32 {
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        i32::checked_sub(self, rhs)
    }
}

impl CheckedSub for i64 {
    #[inline]
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        i64::checked_sub(self, rhs)
    }
}

// ── CheckedMul ────────────────────────────────────────────────────────────

/// Checked multiplication for integer types.
///
/// Returns `None` on overflow instead of wrapping. Used by element-wise
/// multiplication and dot‑product accumulation.
///
/// Only `i32` and `i64` implement this trait.
pub(crate) trait CheckedMul: Numeric + Sealed {
    /// Returns `Some(self * rhs)` if no overflow, `None` otherwise.
    fn checked_mul(self, rhs: Self) -> Option<Self>;
}

impl CheckedMul for i32 {
    #[inline]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        i32::checked_mul(self, rhs)
    }
}

impl CheckedMul for i64 {
    #[inline]
    fn checked_mul(self, rhs: Self) -> Option<Self> {
        i64::checked_mul(self, rhs)
    }
}

// ── CheckedNeg ────────────────────────────────────────────────────────────

/// Checked negation for integer types.
///
/// Returns `None` when negating `i32::MIN` or `i64::MIN` (whose absolute
/// value cannot be represented).
///
/// Only `i32` and `i64` implement this trait.
pub(crate) trait CheckedNeg: Numeric + Sealed {
    /// Returns `Some(-self)` if no overflow, `None` otherwise.
    fn checked_neg(self) -> Option<Self>;
}

impl CheckedNeg for i32 {
    #[inline]
    fn checked_neg(self) -> Option<Self> {
        i32::checked_neg(self)
    }
}

impl CheckedNeg for i64 {
    #[inline]
    fn checked_neg(self) -> Option<Self> {
        i64::checked_neg(self)
    }
}

// ── CheckedDiv ────────────────────────────────────────────────────────────

/// Checked division for integer types.
///
/// Returns `None` for two cases: division by zero, and `MIN / -1` (which
/// overflows the signed integer range). Callers translate `None` to a
/// panic with a descriptive trigger label.
///
/// Only `i32` and `i64` implement this trait.
pub(crate) trait CheckedDiv: Numeric + Sealed {
    /// Returns `Some(self / rhs)` if the division is safe, `None` otherwise.
    fn checked_div(self, rhs: Self) -> Option<Self>;
}

impl CheckedDiv for i32 {
    #[inline]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        i32::checked_div(self, rhs)
    }
}

impl CheckedDiv for i64 {
    #[inline]
    fn checked_div(self, rhs: Self) -> Option<Self> {
        i64::checked_div(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies overflow detection on all five checked arithmetic traits,
    /// covering `i32` and `i64` for both success and failure paths.
    #[test]
    fn test_checked_arithmetic_traits() {
        // Success
        assert_eq!(<i32 as CheckedAdd>::checked_add(1, 2), Some(3));
        assert_eq!(<i64 as CheckedAdd>::checked_add(1, 2), Some(3));
        assert_eq!(<i32 as CheckedSub>::checked_sub(5, 3), Some(2));
        assert_eq!(<i64 as CheckedSub>::checked_sub(5, 3), Some(2));
        assert_eq!(<i32 as CheckedMul>::checked_mul(3, 4), Some(12));
        assert_eq!(<i64 as CheckedMul>::checked_mul(3, 4), Some(12));
        assert_eq!(<i32 as CheckedNeg>::checked_neg(7), Some(-7));
        assert_eq!(<i64 as CheckedNeg>::checked_neg(7), Some(-7));
        assert_eq!(<i32 as CheckedDiv>::checked_div(10, 3), Some(3));
        assert_eq!(<i64 as CheckedDiv>::checked_div(10, 3), Some(3));

        // Overflow / error
        assert_eq!(<i32 as CheckedAdd>::checked_add(i32::MAX, 1), None);
        assert_eq!(<i64 as CheckedAdd>::checked_add(i64::MAX, 1), None);
        assert_eq!(<i32 as CheckedSub>::checked_sub(i32::MIN, 1), None);
        assert_eq!(<i32 as CheckedMul>::checked_mul(i32::MAX, 2), None);
        assert_eq!(<i32 as CheckedNeg>::checked_neg(i32::MIN), None);
        assert_eq!(<i32 as CheckedDiv>::checked_div(1, 0), None);
        assert_eq!(<i32 as CheckedDiv>::checked_div(i32::MIN, -1), None);
    }

    /// Verifies the `i64` failure paths not exercised by
    /// `test_checked_arithmetic_traits` (which only covers `i32` for the
    /// Sub / Mul / Neg / Div failure branches).
    #[test]
    fn test_checked_i64_overflow_paths() {
        assert_eq!(<i64 as CheckedSub>::checked_sub(i64::MIN, 1), None);
        assert_eq!(<i64 as CheckedMul>::checked_mul(i64::MAX, 2), None);
        assert_eq!(<i64 as CheckedNeg>::checked_neg(i64::MIN), None);
        assert_eq!(<i64 as CheckedDiv>::checked_div(1, 0), None);
        assert_eq!(<i64 as CheckedDiv>::checked_div(i64::MIN, -1), None);
    }
}
