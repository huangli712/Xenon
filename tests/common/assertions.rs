//! Precision-aware floating-point comparison helpers for integration tests.
//!
//! Comparison helpers (subset of `28-tests §6.2`):
//! - Tier 1: same-path bitwise equality (`real_bits_eq` on `f32`/`f64`).
//! - Tier 2: cross-path tolerance (`MathTolerance` + `ulp_eq_f64_with_tolerance`).
//! - Integer tensor-level equality (`assert_tensor_exact_int`).
//!
//! Tier 3 (math-function tolerance) and most f32 tolerance helpers have been
//! removed as unused; reintroduce as needed.
//!
//! ULP distance computation follows Bruce Dawson (2012) sign-magnitude →
//! biased-integer monotonic mapping. ±0.0 have an ULP distance of exactly 1;
//! NaN inputs return `u64::MAX`.

use xenon::element::RealScalar;

// ---------------------------------------------------------------------------
// ULP distance primitives
// ---------------------------------------------------------------------------

/// Maps an `f64` bit pattern into a monotonic biased `u64` such that the
/// natural ordering of biased values matches the IEEE-754 total ordering.
#[inline]
fn bias_f64(bits: u64) -> u64 {
    if bits & 0x8000_0000_0000_0000 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    }
}

/// Returns the ULP distance between two `f64` values using the monotonic
/// sign-magnitude → biased-int mapping (Bruce Dawson 2012):
/// `bias(x) = if x.to_bits() >> 63 == 0 { x.to_bits() ^ 0x8000_0000_0000_0000 }
///            else { !x.to_bits() }`
/// after which `|bias(a) - bias(b)|` is the ULP distance.
///
/// Properties:
/// - `ulp_distance_f64(0.0, -0.0) == 1` (sign bit alone differs by one ULP).
/// - NaN inputs return `u64::MAX`.
/// - Any two finite same-sign values return their bitwise integer distance.
///
/// Note: a naive `to_bits().wrapping_sub` is incorrect — ±0.0 would return
/// `0x8000_0000_0000_0000` instead of 1.
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let a_bias = bias_f64(a.to_bits());
    let b_bias = bias_f64(b.to_bits());
    a_bias.abs_diff(b_bias)
}

// ---------------------------------------------------------------------------
// Tier 1 native-type primitives (28-tests §5.2 L354-359)
// ---------------------------------------------------------------------------

/// Crate-private extension trait covering `f32`/`f64` — `RealScalar`
/// (03-element.md §5.3) intentionally does NOT expose `to_bits()` as part
/// of its public surface. This trait bridges the gap for test assertions.
///
/// Public callers pass `f32`/`f64` directly — the extension trait is
/// implemented for both, so call sites using `f32`/`f64` remain valid.
pub(crate) trait RealScalarBits: RealScalar {
    type Bits: Eq;
    fn bits(self) -> Self::Bits;
}

impl RealScalarBits for f32 {
    type Bits = u32;
    fn bits(self) -> u32 {
        self.to_bits()
    }
}

impl RealScalarBits for f64 {
    type Bits = u64;
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

/// Tier 1 bitwise-equality helper. Requires `RealScalarBits` (i.e., `f32`
/// or `f64`) — `RealScalar` alone does not expose `to_bits`.
pub fn real_bits_eq<A: RealScalarBits>(a: A, b: A) -> bool {
    a.bits() == b.bits()
}

// ---------------------------------------------------------------------------
// MathTolerance — Tier 2 cross-path tolerance budget
// ---------------------------------------------------------------------------

/// Tolerance budget used by Tier 2 cross-path helpers
/// (`28-tests §5.2 / §6.2`).
#[derive(Debug, Clone, Copy)]
pub struct MathTolerance {
    /// Maximum allowed ULP distance.
    pub ulp: u64,
    /// Maximum allowed absolute difference (for near-zero values).
    pub abs: f64,
}

impl MathTolerance {
    /// Documented cross-path budget for reductions that traverse the same
    /// elements in different summation orders (Tier 2 — `28-tests §6.2.1`).
    /// Size `n` widens the budget linearly so that serial/parallel/SIMD
    /// sums of `n` summands remain within 2·n ULP of each other.
    #[allow(dead_code)]
    pub const fn cross_path_sum(n: usize) -> Self {
        Self {
            ulp: 2 * n as u64,
            abs: 0.0,
        }
    }

    /// Documented cross-path budget for dot products (Tier 2).
    /// Dot mixes multiply + reduce, giving it 4·n ULP of headroom.
    #[allow(dead_code)]
    pub const fn cross_path_dot(n: usize) -> Self {
        Self {
            ulp: 4 * n as u64,
            abs: 0.0,
        }
    }
}

/// Tier 2 cross-path equality on `f64` (`28-tests §5.2`).
pub fn ulp_eq_f64_with_tolerance(a: f64, b: f64, t: MathTolerance) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if (a - b).abs() <= t.abs {
        return true;
    }
    ulp_distance_f64(a, b) <= t.ulp
}

// ---------------------------------------------------------------------------
// Tensor-level assertion helpers
// ---------------------------------------------------------------------------

use xenon::dimension::Dimension;
use xenon::storage::Storage;
use xenon::tensor::TensorBase;

/// Integer-element same-path equality; thin wrapper over `assert_eq!`.
///
/// Uses `Numeric + PartialEq + std::fmt::Debug` as the element bound since
/// `03-element.md` provides no standalone `IntegerScalar` trait.
/// The bound covers i32/i64 (the two integer element types) while naturally
/// accepting `f32`/`f64` when callers choose to use this path.
#[allow(dead_code)]
pub fn assert_tensor_exact_int<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    msg: &str,
) where
    A: xenon::element::Element + PartialEq + std::fmt::Debug,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{msg}: shape mismatch"
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "{msg}: element {idx} differs");
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_bits_distinguishes_signed_zero() {
        assert!(!real_bits_eq(0.0, -0.0));
        assert!(real_bits_eq(1.0, 1.0));
    }

    #[test]
    fn test_ulp_distance_f64_positive_zero_vs_negative_zero() {
        assert_eq!(ulp_distance_f64(0.0, -0.0), 1);
        assert_eq!(ulp_distance_f64(1.0, 1.0), 0);
    }

    #[test]
    fn test_ulp_distance_f64_nan_returns_max() {
        assert_eq!(ulp_distance_f64(f64::NAN, 1.0), u64::MAX);
        assert_eq!(ulp_distance_f64(1.0, f64::NAN), u64::MAX);
    }

    #[test]
    fn test_ulp_eq_f64_with_tolerance_within_budget() {
        let t = MathTolerance { ulp: 2, abs: 0.0 };
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert!(ulp_eq_f64_with_tolerance(a, b, t));
    }
}