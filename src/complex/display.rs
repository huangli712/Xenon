//! `Display` formatting for `Complex<T>`.
//!
//! NaN-aware, `-0.0`-preserving, precision-aware. Distinguishes IEEE-754
//! `+0.0` from `-0.0` via the crate-private [`PositiveZero`] helper.

use super::{Complex, ComplexFloat};

/// Crate-private helper: distinguishes IEEE-754 `+0.0` from `-0.0`.
///
/// This is an implementation detail, not a public extension point.
pub(crate) trait PositiveZero {
    fn is_positive_zero(&self) -> bool;
}

impl PositiveZero for f32 {
    #[inline]
    fn is_positive_zero(&self) -> bool {
        self.to_bits() == 0.0f32.to_bits()
    }
}

impl PositiveZero for f64 {
    #[inline]
    fn is_positive_zero(&self) -> bool {
        self.to_bits() == 0.0f64.to_bits()
    }
}

impl<T> core::fmt::Display for Complex<T>
where
    T: ComplexFloat + core::fmt::Display + PositiveZero,
{
    /// Formats the complex number in standard mathematical notation.
    ///
    /// # Formatting rules
    ///
    /// | Input | Output |
    /// |---|---|
    /// | `Complex::new(3.0, 4.0)` | `"3+4j"` |
    /// | `Complex::new(3.0, -4.0)` | `"3-4j"` |
    /// | `Complex::new(3.0, 0.0)` | `"3"` |
    /// | `Complex::new(3.0, -0.0)` | `"3-0j"` |
    /// | `Complex::new(0.0, 4.0)` | `"4j"` |
    /// | `Complex::new(0.0, 0.0)` | `"0"` |
    ///
    /// Negative zero (`-0.0`) is distinguished from positive zero via the
    /// crate-private `PositiveZero` helper, which checks the IEEE-754 bit
    /// pattern. NaN imaginary parts are rendered explicitly as `NaNj`.
    ///
    /// Precision (e.g. `{:.2}`) is propagated to every write branch.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let prec = f.precision();
        let zero = T::default();

        // Branch A: NaN imaginary part → always show "{re}+NaNj"
        #[expect(clippy::eq_op)]
        if self.im != self.im {
            return match prec {
                Some(p) => write!(f, "{:.p$}+NaNj", self.re),
                None => write!(f, "{}+NaNj", self.re),
            };
        }

        if self.im == zero {
            // Branch B: imaginary part is +0.0 → fold to pure real
            if self.im.is_positive_zero() {
                match prec {
                    Some(p) => write!(f, "{:.p$}", self.re),
                    None => write!(f, "{}", self.re),
                }
            } else {
                // Branch C: imaginary part is -0.0 → preserve sign explicitly
                match prec {
                    Some(p) => write!(f, "{:.p$}{:.p$}j", self.re, self.im),
                    None => write!(f, "{}{}j", self.re, self.im),
                }
            }
        } else if self.re == zero {
            // Branch D: real part is zero → "{im}j"
            match prec {
                Some(p) => write!(f, "{:.p$}j", self.im),
                None => write!(f, "{}j", self.im),
            }
        } else if self.im > zero {
            // Branch E: positive imaginary → need explicit '+'
            match prec {
                Some(p) => write!(f, "{:.p$}+{:.p$}j", self.re, self.im),
                None => write!(f, "{}+{}j", self.re, self.im),
            }
        } else {
            // Branch F: negative imaginary → '-' already in im's Display
            match prec {
                Some(p) => write!(f, "{:.p$}{:.p$}j", self.re, self.im),
                None => write!(f, "{}{}j", self.re, self.im),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Display: design §5.9 stability table ──

    #[test]
    fn test_display_pos_imag() {
        assert_eq!(Complex::new(3.0_f64, 4.0).to_string(), "3+4j");
    }

    #[test]
    fn test_display_neg_imag() {
        assert_eq!(Complex::new(3.0_f64, -4.0).to_string(), "3-4j");
    }

    #[test]
    fn test_display_pure_real_pos_zero() {
        // +0.0 imaginary part folds away
        assert_eq!(Complex::new(3.0_f64, 0.0).to_string(), "3");
    }

    #[test]
    fn test_display_pure_real_neg_zero_preserved() {
        // -0.0 must NOT fold away (design §5.9 stability rule)
        assert_eq!(Complex::new(3.0_f64, -0.0).to_string(), "3-0j");
    }

    #[test]
    fn test_display_pure_imag() {
        assert_eq!(Complex::new(0.0_f64, 4.0).to_string(), "4j");
    }

    #[test]
    fn test_display_zero() {
        assert_eq!(Complex::new(0.0_f64, 0.0).to_string(), "0");
    }

    #[test]
    fn test_display_nan_imag_shows_na_nj() {
        let s = format!("{}", Complex::new(1.0_f64, f64::NAN));
        assert_eq!(s, "1+NaNj");
    }

    #[test]
    fn test_display_precision_propagation() {
        // {:.2} should propagate to every write! branch
        assert_eq!(format!("{:.2}", Complex::new(1.0_f64, 2.0)), "1.00+2.00j");
        assert_eq!(format!("{:.2}", Complex::new(1.0_f64, -2.0)), "1.00-2.00j");
        assert_eq!(format!("{:.2}", Complex::new(1.0_f64, 0.0)), "1.00");
        assert_eq!(format!("{:.2}", Complex::new(0.0_f64, 2.0)), "2.00j");
    }

    // ── Edge-case Display coverage (audit P4 closure) ──

    #[test]
    fn test_display_neg_zero_real_zero_preserved() {
        assert_eq!(Complex::new(0.0_f64, -0.0).to_string(), "0-0j");
    }

    #[test]
    fn test_display_f32_nan_imag() {
        let s = format!("{}", Complex::new(1.0_f32, f32::NAN));
        assert_eq!(s, "1+NaNj");
    }

    #[test]
    fn test_display_f32_neg_zero_preserved() {
        assert_eq!(Complex::new(3.0_f32, -0.0f32).to_string(), "3-0j");
    }

    #[test]
    fn test_display_precision_nan() {
        let s = format!("{:.2}", Complex::new(1.0_f64, f64::NAN));
        assert_eq!(s, "1.00+NaNj");
    }

    #[test]
    fn test_display_precision_neg_zero() {
        let s = format!("{:.2}", Complex::new(1.0_f64, -0.0));
        assert_eq!(s, "1.00-0.00j");
    }

    #[test]
    fn test_display_pos_infinity() {
        assert_eq!(Complex::new(1.0_f64, f64::INFINITY).to_string(), "1+infj");
    }

    #[test]
    fn test_display_neg_infinity() {
        assert_eq!(
            Complex::new(1.0_f64, f64::NEG_INFINITY).to_string(),
            "1-infj"
        );
    }

    // ── Infinity + special-value Display combinations ──

    #[test]
    fn test_display_inf_nan_imag() {
        assert_eq!(
            Complex::new(f64::INFINITY, f64::NAN).to_string(),
            "inf+NaNj"
        );
    }

    #[test]
    fn test_display_inf_zero_imag() {
        assert_eq!(Complex::new(f64::INFINITY, 0.0).to_string(), "inf");
    }

    #[test]
    fn test_display_inf_neg_zero_imag() {
        assert_eq!(Complex::new(f64::INFINITY, -0.0).to_string(), "inf-0j");
    }

    #[test]
    fn test_display_nan_nan() {
        let s = format!("{}", Complex::new(f64::NAN, f64::NAN));
        assert_eq!(s, "NaN+NaNj");
    }

    // ── Second-audit P4 closure: NaN real, -0 real, {:.0} precision ──

    #[test]
    fn test_display_nan_real_pos_imag() {
        assert_eq!(Complex::new(f64::NAN, 1.0).to_string(), "NaN+1j");
    }

    #[test]
    fn test_display_nan_real_neg_imag() {
        assert_eq!(Complex::new(f64::NAN, -1.0).to_string(), "NaN-1j");
    }

    #[test]
    fn test_display_nan_real_zero_imag() {
        assert_eq!(Complex::new(f64::NAN, 0.0).to_string(), "NaN");
    }

    #[test]
    fn test_display_neg_zero_real() {
        assert_eq!(Complex::new(-0.0_f64, 0.0).to_string(), "-0");
    }

    #[test]
    fn test_display_neg_zero_both() {
        assert_eq!(Complex::new(-0.0_f64, -0.0).to_string(), "-0-0j");
    }

    #[test]
    fn test_display_precision_zero_pos_imag() {
        assert_eq!(format!("{:.0}", Complex::new(1.5_f64, 2.5)), "2+2j");
    }

    #[test]
    fn test_display_precision_zero_neg_imag() {
        assert_eq!(format!("{:.0}", Complex::new(1.5_f64, -2.5)), "2-2j");
    }

    #[test]
    fn test_display_precision_zero_pure_real() {
        assert_eq!(format!("{:.0}", Complex::new(1.5_f64, 0.0)), "2");
    }

    #[test]
    fn test_display_precision_zero_pure_imag() {
        assert_eq!(format!("{:.0}", Complex::new(0.5_f64, 2.5)), "0+2j");
    }
}
