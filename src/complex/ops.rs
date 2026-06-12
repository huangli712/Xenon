//! Arithmetic operator implementations for `Complex<T>`.
//!
//! Each operator follows the sealed `ComplexFloat` bound so only
//! `Complex<f32>` and `Complex<f64>` participate.

use core::ops::{Add, Div, Mul, Neg, Sub};

use super::{Complex, ComplexFloat};

/// Component-wise negation: -(a+bj) = -a-bj.
impl<T: ComplexFloat> Neg for Complex<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

/// Component-wise addition: (a+bj) + (c+dj) = (a+c)+(b+d)j.
impl<T: ComplexFloat> Add for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

/// Component-wise subtraction: (a+bj) - (c+dj) = (a-c)+(b-d)j.
impl<T: ComplexFloat> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

/// Complex multiplication: (ac-bd)+(ad+bc)j.
impl<T: ComplexFloat> Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// Complex division using the Smith algorithm in `f32` precision.
///
/// This is an independent implementation; it does **not** delegate to
/// `Complex<f64>`. The branch `|re| >= |im|` avoids intermediate overflow
/// within `f32` arithmetic.
impl Div for Complex<f32> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self::new(
                (self.re + self.im * r) / denom,
                (self.im - self.re * r) / denom,
            )
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.re * r + rhs.im;
            Self::new(
                (self.re * r + self.im) / denom,
                (self.im * r - self.re) / denom,
            )
        }
    }
}

/// Complex division using the Smith algorithm in `f64` precision.
///
/// The branch `|re| >= |im|` avoids forming `c² + d²` directly,
/// preventing intermediate overflow.
impl Div for Complex<f64> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self::new(
                (self.re + self.im * r) / denom,
                (self.im - self.re * r) / denom,
            )
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.re * r + rhs.im;
            Self::new(
                (self.re * r + self.im) / denom,
                (self.im * r - self.re) / denom,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f32: -(1+2j) = (-1-2j).
    #[test]
    fn test_neg_complex_f32() {
        assert_eq!(-Complex::new(1.0_f32, 2.0), Complex::new(-1.0, -2.0));
    }

    /// -(1+2j) = (-1-2j).
    #[test]
    fn test_neg_complex_f64() {
        assert_eq!(-Complex::new(1.0_f64, 2.0), Complex::new(-1.0, -2.0));
    }

    /// f32: (1+2j) + (3+4j) = (4+6j).
    #[test]
    fn test_add_complex_f32() {
        assert_eq!(
            Complex::new(1.0_f32, 2.0) + Complex::new(3.0, 4.0),
            Complex::new(4.0, 6.0)
        );
    }

    /// (1+2j) + (3+4j) = (4+6j).
    #[test]
    fn test_add_complex_f64() {
        assert_eq!(
            Complex::new(1.0_f64, 2.0) + Complex::new(3.0, 4.0),
            Complex::new(4.0, 6.0)
        );
    }

    /// f32: (5+7j) - (2+3j) = (3+4j).
    #[test]
    fn test_sub_complex_f32() {
        assert_eq!(
            Complex::new(5.0_f32, 7.0) - Complex::new(2.0, 3.0),
            Complex::new(3.0, 4.0)
        );
    }

    /// (5+7j) - (2+3j) = (3+4j).
    #[test]
    fn test_sub_complex_f64() {
        assert_eq!(
            Complex::new(5.0_f64, 7.0) - Complex::new(2.0, 3.0),
            Complex::new(3.0, 4.0)
        );
    }

    /// f32: (1+2j) * (3+4j) = (-5+10j).
    #[test]
    fn test_mul_complex_f32() {
        assert_eq!(
            Complex::new(1.0_f32, 2.0) * Complex::new(3.0, 4.0),
            Complex::new(-5.0, 10.0)
        );
    }

    /// (1+2j) * (3+4j) = (-5+10j).
    #[test]
    fn test_mul_complex_f64() {
        assert_eq!(
            Complex::new(1.0_f64, 2.0) * Complex::new(3.0, 4.0),
            Complex::new(-5.0, 10.0)
        );
    }

    /// `f32` basic division: (6+8j)/(3+4j) = 2+0j.
    #[test]
    fn test_div_basic_f32() {
        let z = Complex::new(6.0_f32, 8.0) / Complex::new(3.0_f32, 4.0);
        assert!((z.re - 2.0).abs() < 1e-5);
        assert!(z.im.abs() < 1e-5);
    }

    /// `f64` basic division: (6+8j)/(3+4j) = 2+0j.
    #[test]
    fn test_div_basic_f64() {
        let z = Complex::new(6.0_f64, 8.0) / Complex::new(3.0_f64, 4.0);
        assert!((z.re - 2.0).abs() < 1e-12);
        assert!(z.im.abs() < 1e-12);
    }

    /// `f32` division by zero propagates NaN or ∞ per IEEE 754.
    #[test]
    fn test_div_zero_f32() {
        let z = Complex::new(1.0_f32, 2.0) / Complex::new(0.0_f32, 0.0);
        assert!(z.re.is_nan() || z.re.is_infinite());
        assert!(z.im.is_nan() || z.im.is_infinite());
    }

    /// `f64` division by zero propagates NaN or ∞ per IEEE 754.
    #[test]
    fn test_div_zero_f64() {
        let z = Complex::new(1.0_f64, 2.0) / Complex::new(0.0_f64, 0.0);
        assert!(z.re.is_nan() || z.re.is_infinite());
        assert!(z.im.is_nan() || z.im.is_infinite());
    }

    /// `f32` division exercises the |im| > |re| Smith branch.
    #[test]
    fn test_div_large_im_f32() {
        let result = Complex::new(1.0_f32, 0.0) / Complex::new(0.0_f32, 1.0);
        assert!(result.re.abs() < 1e-5);
        assert!((result.im - (-1.0)).abs() < 1e-5);
    }

    /// `f64` division exercises the |im| > |re| Smith branch.
    #[test]
    fn test_div_large_im_f64() {
        let result = Complex::new(1.0_f64, 0.0) / Complex::new(0.0_f64, 1.0);
        assert!(result.re.abs() < 1e-12);
        assert!((result.im - (-1.0)).abs() < 1e-12);
    }

    /// Explicit real-to-complex promotion: `Complex::from(r)` works with all operators.
    #[test]
    fn test_explicit_real_to_complex_ops() {
        let c = Complex::new(1.0_f64, 2.0);
        let r = Complex::from(3.0_f64);

        assert_eq!(r + c, Complex::new(4.0, 2.0));
        assert_eq!(r - c, Complex::new(2.0, -2.0));
        assert_eq!(r * c, Complex::new(3.0, 6.0));

        let q = c / r;
        assert!((q.re - 1.0 / 3.0).abs() < 1e-12);
        assert!((q.im - 2.0 / 3.0).abs() < 1e-12);

        assert_eq!(-r, Complex::new(-3.0, 0.0));
    }
}
