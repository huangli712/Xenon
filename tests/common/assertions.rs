//! Precision-aware floating-point comparison helpers for integration tests.
//!
//! Three-tier comparison model per `28-tests §6.2`:
//! - Tier 1: same execution path → bitwise equality or ULP == 0 (exact).
//! - Tier 2: cross-path (serial/SIMD/parallel) → documented ULP tolerance.
//! - Tier 3: math functions (sin/sqrt/exp/ln) → per-function tolerance.
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
    if a_bias >= b_bias {
        a_bias - b_bias
    } else {
        b_bias - a_bias
    }
}

/// Maps an `f32` bit pattern into a monotonic biased `u32`.
#[inline]
fn bias_f32(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

/// Returns the ULP distance between two `f32` values. NaN → `u64::MAX`.
pub fn ulp_distance_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let a_bias = bias_f32(a.to_bits());
    let b_bias = bias_f32(b.to_bits());
    let d = a_bias.abs_diff(b_bias);
    d as u64
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
trait RealScalarBits: RealScalar {
    type Bits: Eq;
    fn bits(self) -> Self::Bits;
    fn ulp(a: Self, b: Self) -> u64;
}

impl RealScalarBits for f32 {
    type Bits = u32;
    fn bits(self) -> u32 {
        self.to_bits()
    }
    fn ulp(a: f32, b: f32) -> u64 {
        ulp_distance_f32(a, b)
    }
}

impl RealScalarBits for f64 {
    type Bits = u64;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn ulp(a: f64, b: f64) -> u64 {
        ulp_distance_f64(a, b)
    }
}

/// Tier 1 bitwise-equality helper. Requires `RealScalarBits` (i.e., `f32`
/// or `f64`) — `RealScalar` alone does not expose `to_bits`.
pub fn real_bits_eq<A: RealScalarBits>(a: A, b: A) -> bool {
    a.bits() == b.bits()
}

/// Tier 1 ULP == 0 equality (NaN → false). Calls the ULP-distance function
/// associated with the native type via `RealScalarBits::ulp`.
pub fn real_ulp_eq<A: RealScalarBits>(a: A, b: A) -> bool {
    A::ulp(a, b) == 0
}

/// Tier 1 same-path ULP equality on `f64` (ULP == 0, NaN → false).
///
/// Design `28-tests §6.2.1` Tier 1 requires that two values produced by the
/// same code path compare bit-identical modulo ±0.0 treatment. This helper
/// is the canonical entry point used by downstream tasks (W29T19/T20/T22)
/// when they only need "is the result exactly the same?" without a
/// tolerance budget. `NaN` vs `NaN` returns `false` to stay consistent
/// with IEEE 754 ordering semantics.
pub fn ulp_eq_f64_exact(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    ulp_distance_f64(a, b) == 0
}

/// Tier 1 same-path ULP equality on `f32` (ULP == 0, NaN → false).
pub fn ulp_eq_f32_exact(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    ulp_distance_f32(a, b) == 0
}

// ---------------------------------------------------------------------------
// MathTolerance — Tier 2 / Tier 3 tolerance budget
// ---------------------------------------------------------------------------

/// Tolerance budget used by Tier 2 cross-path and Tier 3 math-function
/// helpers (`28-tests §5.2 / §6.2`).
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
    pub const fn cross_path_sum(n: usize) -> Self {
        Self {
            ulp: 2 * n as u64,
            abs: 0.0,
        }
    }

    /// Documented cross-path budget for dot products (Tier 2).
    /// Dot mixes multiply + reduce, giving it 4·n ULP of headroom.
    pub const fn cross_path_dot(n: usize) -> Self {
        Self {
            ulp: 4 * n as u64,
            abs: 0.0,
        }
    }
}

/// Documented cross-path tolerance for the default reduction input size.
/// Used by tests that do not know the reduction length at compile time
/// (`28-tests §5.2 / §6.2.1` Tier 2). Returns a conservative generic budget.
pub fn documented_cross_path_tolerance() -> MathTolerance {
    // 128 ULP covers sums up to a few thousand elements; tests that need
    // tighter or looser bounds should call `MathTolerance::cross_path_*`.
    MathTolerance { ulp: 128, abs: 0.0 }
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

/// Tier 3 math-function equality on `f64`; same semantics as the cross-path
/// helper but the tolerance is sourced from per-function documentation
/// (`28-tests §5.2 / §6.2.1` Tier 3).
pub fn math_eq_f64(a: f64, b: f64, t: MathTolerance) -> bool {
    ulp_eq_f64_with_tolerance(a, b, t)
}

/// Tier 2 cross-path equality on `f32` (`28-tests §5.2`; native-type
/// comparison per §6.2.1 mandates parallel f32/f64 helpers).
pub fn ulp_eq_f32_with_tolerance(a: f32, b: f32, t: MathTolerance) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if (a - b).abs() as f64 <= t.abs {
        return true;
    }
    ulp_distance_f32(a, b) <= t.ulp
}

/// Tier 3 math-function equality on `f32` (`28-tests §5.2 / §6.2.1` Tier 3).
pub fn math_eq_f32(a: f32, b: f32, t: MathTolerance) -> bool {
    ulp_eq_f32_with_tolerance(a, b, t)
}

// ---------------------------------------------------------------------------
// Tensor-level assertion helpers
// ---------------------------------------------------------------------------

use xenon::dimension::Dimension;
use xenon::element::{CastTo, ComplexScalar};
use xenon::storage::Storage;
use xenon::tensor::TensorBase;

/// Tier 1 same-path real-valued equality (`28-tests §5.2 L211-230`).
///
/// Element comparison stays in the native element type; no cross-precision
/// cast is performed. Integer elements should reach this helper via a
/// dedicated integer overload that uses `assert_eq!`.
pub fn assert_tensor_exact_real<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    msg: &str,
) where
    A: RealScalar + RealScalarBits,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{msg}: shape mismatch: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_native = *a;
        let e_native = *e;
        assert!(
            real_bits_eq(a_native, e_native) && real_ulp_eq(a_native, e_native),
            "{msg}: element {idx} differs: actual={a_native:?}, expected={e_native:?}, \
             comparison=strict native-type ULP==0"
        );
    }
}

/// Tier 1 same-path complex equality (`28-tests §5.2 L236-260`).
///
/// Real and imaginary components are compared independently in their native
/// component type with strict `ULP == 0`.
pub fn assert_tensor_exact_complex<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    msg: &str,
) where
    A: ComplexScalar,
    A::Real: RealScalarBits,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{msg}: shape mismatch: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_re = a.re();
        let a_im = a.im();
        let e_re = e.re();
        let e_im = e.im();
        assert!(
            real_bits_eq(a_re, e_re) && real_ulp_eq(a_re, e_re),
            "{msg}: element {idx} real part differs: actual={a_re:?}, expected={e_re:?}, \
             comparison=strict native-type ULP==0"
        );
        assert!(
            real_bits_eq(a_im, e_im) && real_ulp_eq(a_im, e_im),
            "{msg}: element {idx} imag part differs: actual={a_im:?}, expected={e_im:?}, \
             comparison=strict native-type ULP==0"
        );
    }
}

/// Tier 2 cross-path real equality (`28-tests §5.2 L262-293`).
///
/// Use only for scalar-vs-SIMD or serial-vs-parallel comparisons where
/// rounding is allowed. The tolerance must come from documented sources.
pub fn assert_tensor_close_real_cross_path<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    tolerance: MathTolerance,
    msg: &str,
) where
    A: RealScalar + CastTo<f64>,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{msg}: shape mismatch: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f: f64 = CastTo::<f64>::cast_to(*a)
            .expect("cross-path helper requires CastTo::<f64>::cast_to support");
        let e_f: f64 = CastTo::<f64>::cast_to(*e)
            .expect("cross-path helper requires CastTo::<f64>::cast_to support");
        assert!(
            ulp_eq_f64_with_tolerance(a_f, e_f, tolerance),
            "{msg}: element {idx} differs: actual={a_f}, expected={e_f}, \
             comparison=cross-path tolerance"
        );
    }
}

/// Tier 3 math-function real equality (`28-tests §5.2 L295-326`).
///
/// Use only for `sin/sqrt/exp/ln/floor/ceil` etc. with per-function tolerance.
pub fn assert_tensor_close_real_math<A, D>(
    actual: &TensorBase<impl Storage<Elem = A>, D>,
    expected: &TensorBase<impl Storage<Elem = A>, D>,
    tolerance: MathTolerance,
    msg: &str,
) where
    A: RealScalar + CastTo<f64>,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{msg}: shape mismatch: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a_f: f64 = CastTo::<f64>::cast_to(*a)
            .expect("math helper requires CastTo::<f64>::cast_to support");
        let e_f: f64 = CastTo::<f64>::cast_to(*e)
            .expect("math helper requires CastTo::<f64>::cast_to support");
        assert!(
            math_eq_f64(a_f, e_f, tolerance),
            "{msg}: element {idx} differs: actual={a_f}, expected={e_f}, \
             comparison=math-function tolerance"
        );
    }
}

/// Integer-element same-path equality; thin wrapper over `assert_eq!`.
///
/// Uses `Numeric + PartialEq + std::fmt::Debug` as the element bound since
/// `03-element.md` provides no standalone `IntegerScalar` trait.
/// The bound covers i32/i64 (the two integer element types) while naturally
/// accepting `f32`/`f64` when callers choose to use this path.
#[expect(dead_code)]
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