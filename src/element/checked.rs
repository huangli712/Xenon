//! Checked arithmetic traits (§5.10).
//!
//! Integer-only overflow-sensitive operations that return `Option`. Callers
//! translate `None` to a panic per the project-wide integer overflow policy
//! (see `26-error.md §6`). Float types use ordinary operators (NaN
//! propagation handles the semantics) and are intentionally not covered
//! here.

use crate::element::Numeric;
use crate::private::Sealed;

/// Checked addition for types that support it.
///
/// Returns `None` on overflow instead of wrapping.
/// Only implemented for integer types (`i32`, `i64`).
/// Float types use ordinary `+` (NaN propagation handles the semantics).
// TODO(W11/W18): remove when math/reduction/cast call sites land.
#[allow(dead_code)]
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

/// Checked subtraction for integer-only overflow-sensitive paths.
// TODO(W11/W18): remove when math/reduction/cast call sites land.
#[allow(dead_code)]
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

/// Checked multiplication for integer-only overflow-sensitive paths.
// TODO(W11/W18): remove when math/reduction/cast call sites land.
#[allow(dead_code)]
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

/// Checked negation for integer-only overflow-sensitive paths.
// TODO(W11/W18): remove when math/reduction/cast call sites land.
#[allow(dead_code)]
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

/// Checked division for integer-only overflow-sensitive paths.
///
/// Returns `None` for divisor zero or for the `MIN / -1` overflow case;
/// callers translate `None` to a panic per the project-wide integer
/// overflow policy (see `26-error.md §6`).
// TODO(W11/W18): remove when math/reduction/cast call sites land.
#[allow(dead_code)]
pub(crate) trait CheckedDiv: Numeric + Sealed {
    /// Returns `Some(self / rhs)` if no overflow or zero-divisor, `None` otherwise.
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

    #[test]
    fn test_checked_arithmetic_traits() {
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

        assert_eq!(<i32 as CheckedAdd>::checked_add(i32::MAX, 1), None);
        assert_eq!(<i64 as CheckedAdd>::checked_add(i64::MAX, 1), None);
        assert_eq!(<i32 as CheckedSub>::checked_sub(i32::MIN, 1), None);
        assert_eq!(<i32 as CheckedMul>::checked_mul(i32::MAX, 2), None);
        assert_eq!(<i32 as CheckedNeg>::checked_neg(i32::MIN), None);
        assert_eq!(<i32 as CheckedDiv>::checked_div(1, 0), None);
        assert_eq!(<i32 as CheckedDiv>::checked_div(i32::MIN, -1), None);
    }
}
