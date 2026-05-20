//! Arithmetic operator implementations for `Complex<T>`.
//!
//! Each operator follows the sealed `ComplexFloat` bound so only
//! `Complex<f32>` and `Complex<f64>` participate.

use super::{Complex, ComplexFloat};

/// Component-wise addition: (a+bj) + (c+dj) = (a+c)+(b+d)j.
impl<T: ComplexFloat> core::ops::Add for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

/// Component-wise subtraction: (a+bj) - (c+dj) = (a-c)+(b-d)j.
impl<T: ComplexFloat> core::ops::Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

/// Complex multiplication: (ac-bd)+(ad+bc)j.
impl<T: ComplexFloat> core::ops::Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// Complex division using the Smith algorithm in `f64` precision.
///
/// The branch `|re| >= |im|` avoids forming `c² + d²` directly,
/// preventing intermediate overflow.
impl core::ops::Div for Complex<f64> {
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

/// Complex division using the Smith algorithm in `f32` precision.
///
/// This is an independent implementation; it does **not** delegate to
/// `Complex<f64>`. The branch `|re| >= |im|` avoids intermediate overflow
/// within `f32` arithmetic.
impl core::ops::Div for Complex<f32> {
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

/// Component-wise negation: -(a+bj) = -a-bj.
impl<T: ComplexFloat> core::ops::Neg for Complex<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_complex() {
        assert_eq!(
            Complex::new(1.0_f64, 2.0) + Complex::new(3.0, 4.0),
            Complex::new(4.0, 6.0)
        );
    }

    #[test]
    fn test_sub_complex() {
        assert_eq!(
            Complex::new(5.0_f64, 7.0) - Complex::new(2.0, 3.0),
            Complex::new(3.0, 4.0)
        );
    }

    #[test]
    fn test_mul_complex() {
        assert_eq!(
            Complex::new(1.0_f64, 2.0) * Complex::new(3.0, 4.0),
            Complex::new(-5.0, 10.0)
        );
    }

    #[test]
    fn test_div_complex_f64() {
        let z = Complex::new(6.0_f64, 8.0) / Complex::new(3.0_f64, 4.0);
        assert!((z.re - 2.0).abs() < 1e-12);
        assert!(z.im.abs() < 1e-12);
    }

    #[test]
    fn test_div_complex_f32() {
        let z = Complex::new(6.0_f32, 8.0) / Complex::new(3.0_f32, 4.0);
        assert!((z.re - 2.0).abs() < 1e-5);
        assert!(z.im.abs() < 1e-5);
    }

    #[test]
    fn test_div_zero_propagates_ieee754_f64() {
        let z = Complex::new(1.0_f64, 2.0) / Complex::new(0.0_f64, 0.0);
        assert!(z.re.is_nan() || z.re.is_infinite());
        assert!(z.im.is_nan() || z.im.is_infinite());
    }

    #[test]
    fn test_div_zero_propagates_ieee754_f32() {
        let z = Complex::new(1.0_f32, 2.0) / Complex::new(0.0_f32, 0.0);
        assert!(z.re.is_nan() || z.re.is_infinite());
        assert!(z.im.is_nan() || z.im.is_infinite());
    }

    #[test]
    fn test_div_branch_selection_large_im_f64() {
        let result = Complex::new(1.0_f64, 0.0) / Complex::new(0.0_f64, 1.0);
        assert!(result.re.abs() < 1e-12);
        assert!((result.im - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_div_branch_selection_large_im_f32() {
        let result = Complex::new(1.0_f32, 0.0) / Complex::new(0.0_f32, 1.0);
        assert!(result.re.abs() < 1e-5);
        assert!((result.im - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_neg_complex() {
        assert_eq!(-Complex::new(1.0_f64, 2.0), Complex::new(-1.0, -2.0));
    }

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
