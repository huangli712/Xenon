//! Complex number type `Complex<T>` with a sealed component bound.
//!
//! W5T1 provides the minimal skeleton: the `Complex<T>` struct, its `new()`
//! constructor, and the `ComplexFloat` sealed trait with `Sealed + Copy +
//! Default` supertraits. Subsequent Wave-5 tasks extend `ComplexFloat` and
//! add arithmetic, formatting, conversion, and math methods.

/// Public bound for `Complex<T>` — sealed to `f32` and `f64`.
///
/// The supertrait set captures every algebraic / ordering capability used by
/// the complex arithmetic and Display impls (W5T5+/W5T7+). `f32` and `f64`
/// from the standard library naturally satisfy every supertrait listed below.
///
/// # Sealed boundary
///
/// Downstream crates cannot implement `ComplexFloat` because of the
/// `Sealed` supertrait. The following doctest verifies that non-supported
/// types are rejected at compile time:
///
/// ```compile_fail
/// use xenon::complex::Complex;
/// // i32 does not implement ComplexFloat (Sealed); this declaration must
/// // fail at compile time.
/// let _: Complex<i32>;
/// ```
pub trait ComplexFloat:
    crate::private::Sealed
    + Copy
    + Default
    + PartialEq
    + PartialOrd
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{}

impl ComplexFloat for f32 {}
impl ComplexFloat for f64 {}

/// Complex number: a + bj.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex<T: ComplexFloat> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: ComplexFloat> Complex<T> {
    /// Creates a new complex number.
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Returns the real part.
    #[inline]
    pub fn re(self) -> T {
        self.re
    }

    /// Returns the imaginary part.
    #[inline]
    pub fn im(self) -> T {
        self.im
    }

    /// Returns true if imaginary part is zero.
    #[inline]
    pub fn is_real(self) -> bool {
        self.im == T::default()
    }

    /// Returns true if real part is zero.
    #[inline]
    pub fn is_imaginary(self) -> bool {
        self.re == T::default()
    }

    /// Creates a purely imaginary number (re = 0).
    #[inline]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::default(), im)
    }

    /// Returns the complex conjugate: conj(a + bj) = a - bj.
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
}

// Compile-time layout verification.  Protects the #[repr(C)] contract
// so that Complex<f32>/Complex<f64> remain layout-compatible with
// two-field C structs { T re; T im; }.
const _: () = {
    assert!(core::mem::size_of::<Complex<f32>>() == 2 * core::mem::size_of::<f32>());
    assert!(core::mem::align_of::<Complex<f32>>() == core::mem::align_of::<f32>());
    assert!(core::mem::size_of::<Complex<f64>>() == 2 * core::mem::size_of::<f64>());
    assert!(core::mem::align_of::<Complex<f64>>() == core::mem::align_of::<f64>());
};

// ── From<T>: explicit real-to-complex construction ──

impl<T: ComplexFloat> From<T> for Complex<T> {
    #[inline]
    fn from(re: T) -> Self {
        Self::new(re, T::default())
    }
}

// ── PositiveZero: crate-private helper for distinguishing +0.0 / -0.0 ──

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

// ── PartialEq: component-wise IEEE-754 equality ──

impl<T: ComplexFloat> PartialEq for Complex<T> {
    /// Component-wise IEEE-754 equality.  `NaN != NaN` is preserved.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}
// Intentionally NOT implementing Eq (NaN violates reflexivity)
// nor PartialOrd / Ord (complex numbers have no natural total order).

// ── Display: NaN-aware, -0.0-preserving, precision-aware ──

impl<T> core::fmt::Display for Complex<T>
where
    T: ComplexFloat + core::fmt::Display + PositiveZero,
{
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

    #[test]
    fn test_complex_new() {
        let z = Complex::new(3.0_f64, 4.0);
        assert_eq!(z.re, 3.0);
        assert_eq!(z.im, 4.0);
    }

    #[test]
    fn test_complex_float_accepts_f32_f64() {
        fn accepts_complex_float<T: ComplexFloat>(_value: Complex<T>) {}
        accepts_complex_float(Complex::new(1.0_f32, 2.0));
        accepts_complex_float(Complex::new(1.0_f64, 2.0));
    }

    #[test]
    fn test_complex_layout_f64() {
        assert_eq!(core::mem::size_of::<Complex<f64>>(), 16);
        assert_eq!(core::mem::align_of::<Complex<f64>>(), core::mem::align_of::<f64>());
    }

    #[test]
    fn test_complex_layout_f32() {
        assert_eq!(core::mem::size_of::<Complex<f32>>(), 8);
        assert_eq!(core::mem::align_of::<Complex<f32>>(), core::mem::align_of::<f32>());
    }

    #[test]
    fn test_complex_field_offsets_f64() {
        let z = Complex::<f64>::new(0.0, 0.0);
        let base = (&z) as *const _ as usize;
        let re_addr = (&z.re) as *const _ as usize;
        let im_addr = (&z.im) as *const _ as usize;
        assert_eq!(re_addr - base, 0);
        assert_eq!(im_addr - base, core::mem::size_of::<f64>());
    }

    #[test]
    fn test_complex_field_offsets_f32() {
        let z = Complex::<f32>::new(0.0, 0.0);
        let base = (&z) as *const _ as usize;
        let re_addr = (&z.re) as *const _ as usize;
        let im_addr = (&z.im) as *const _ as usize;
        assert_eq!(re_addr - base, 0);
        assert_eq!(im_addr - base, core::mem::size_of::<f32>());
    }

    #[test]
    fn test_complex_accessors() {
        let z = Complex::new(3.0_f64, 4.0);
        assert_eq!(z.re(), 3.0);
        assert_eq!(z.im(), 4.0);
    }

    #[test]
    fn test_is_real_imaginary() {
        assert!(Complex::new(3.0_f64, 0.0).is_real());
        assert!(!Complex::new(3.0_f64, 4.0).is_real());
        assert!(Complex::new(0.0, 3.0_f64).is_imaginary());
        assert!(!Complex::new(3.0_f64, 4.0).is_imaginary());
        assert!(Complex::new(0.0, 0.0).is_real() && Complex::new(0.0, 0.0).is_imaginary());
    }

    // ── PartialEq tests ──

    #[test]
    fn test_eq_nan() {
        let nan = Complex::new(f64::NAN, 0.0);
        assert_ne!(nan, nan);
    }

    #[test]
    fn test_eq_componentwise() {
        assert_eq!(Complex::new(1.0_f64, 2.0), Complex::new(1.0, 2.0));
        assert_ne!(Complex::new(1.0_f64, 2.0), Complex::new(1.0, 3.0));
    }

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

    #[test]
    fn test_from_imag_and_conj() {
        let z = Complex::from_imag(4.0_f64);
        assert_eq!(z, Complex::new(0.0, 4.0));
        assert_eq!(z.conj(), Complex::new(0.0, -4.0));
    }

    #[test]
    fn test_from_real() {
        assert_eq!(Complex::from(5.0_f64), Complex::new(5.0, 0.0));
        assert_eq!(Complex::from(-1.0_f64), Complex::new(-1.0, 0.0));
    }

    // ── Edge-case Display / predicate coverage (audit P4 closure) ──

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
    fn test_is_real_neg_zero_imag() {
        assert!(Complex::new(3.0_f64, -0.0).is_real());
    }

    #[test]
    fn test_display_pos_infinity() {
        assert_eq!(Complex::new(1.0_f64, f64::INFINITY).to_string(), "1+infj");
    }

    #[test]
    fn test_display_neg_infinity() {
        assert_eq!(Complex::new(1.0_f64, f64::NEG_INFINITY).to_string(), "1-infj");
    }

    #[test]
    fn test_display_nan_nan() {
        let s = format!("{}", Complex::new(f64::NAN, f64::NAN));
        assert_eq!(s, "NaN+NaNj");
    }

    #[test]
    fn test_complex_default() {
        assert_eq!(Complex::<f64>::default(), Complex::new(0.0, 0.0));
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
    fn test_is_imaginary_neg_zero_real() {
        assert!(Complex::new(-0.0_f64, 5.0).is_imaginary());
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
