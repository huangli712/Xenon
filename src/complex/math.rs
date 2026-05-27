//! Concrete math methods for `Complex<f32>` and `Complex<f64>`.
//!
//! Hosts the modulus (`norm`, `norm_sqr`) and special-value predicates
//! (`is_nan`, `is_finite`) that are only meaningful on a real-floating
//! component type and therefore live outside the generic `Complex<T>` impl.

use super::Complex;

impl Complex<f64> {
    /// Modulus |z| = sqrt(re² + im²), via `hypot` to avoid overflow.
    #[inline]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared modulus |z|² = re² + im² (no sqrt).
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// True if either component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// True if both components are finite (not NaN and not infinite).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl Complex<f32> {
    /// Modulus |z| = sqrt(re² + im²), via `hypot` to avoid overflow.
    #[inline]
    pub fn norm(self) -> f32 {
        self.re.hypot(self.im)
    }

    /// Squared modulus |z|² = re² + im² (no sqrt).
    #[inline]
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// True if either component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// True if both components are finite (not NaN and not infinite).
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Classic 3-4-5 triangle: norm = 5, norm_sqr = 25.
    #[test]
    fn test_norm_3_4_5() {
        let z = Complex::new(3.0_f64, 4.0);
        assert_eq!(z.norm(), 5.0);
        assert_eq!(z.norm_sqr(), 25.0);
    }

    /// `norm_sqr` = re² + im².
    #[test]
    fn test_norm_sqr() {
        let z = Complex::new(3.0_f64, 4.0);
        assert_eq!(z.norm_sqr(), 3.0 * 3.0 + 4.0 * 4.0);
    }

    /// `hypot` avoids overflow with large values.
    #[test]
    fn test_norm_no_overflow() {
        let z = Complex::new(1.0e200_f64, 1.0e200);
        assert!(z.norm().is_finite());
    }

    /// Detects NaN in either component.
    #[test]
    fn test_is_nan() {
        assert!(Complex::new(f64::NAN, 0.0).is_nan());
        assert!(Complex::new(0.0_f64, f64::NAN).is_nan());
        assert!(!Complex::new(1.0_f64, 2.0).is_nan());
    }

    /// Detects non-finite (NaN or ∞) in either component.
    #[test]
    fn test_is_finite() {
        assert!(Complex::new(1.0_f64, 2.0).is_finite());
        assert!(!Complex::new(f64::INFINITY, 0.0).is_finite());
        assert!(!Complex::new(0.0_f64, f64::NAN).is_finite());
    }

    /// Norm works for `f32` as well.
    #[test]
    fn test_norm_f32() {
        let z = Complex::new(3.0_f32, 4.0);
        assert_eq!(z.norm(), 5.0);
    }
}
