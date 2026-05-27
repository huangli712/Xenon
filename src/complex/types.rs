//! Core types: [`ComplexFloat`] trait and [`Complex`] struct.

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
{
}

impl ComplexFloat for f32 {}
impl ComplexFloat for f64 {}

/// Complex number represented as `re + im*j`.
///
/// # Supported types
///
/// Only `Complex<f32>` and `Complex<f64>` are supported, enforced by
/// the sealed [`ComplexFloat`] trait.
///
/// # Memory layout
///
/// `#[repr(C)]` keeps this type layout-compatible with a two-field C struct
/// `{ T re; T im; }`. Size is `2 * size_of::<T>()`; alignment matches `T`.
///
/// # Examples
///
/// ```
/// use xenon::complex::Complex;
/// let z = Complex::new(3.0_f64, 4.0);
/// assert_eq!(z.norm(), 5.0);
/// ```
///
/// # Type safety: real-complex mixed arithmetic is rejected at compile time
///
/// Use [`Complex::from`] (provided by [`From<T>`] for [`Complex<T>`]) for
/// explicit promotion. The following blocks must all fail to compile.
///
/// ## `Complex + T` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = c + 3.0_f64;
/// ```
/// ## `T + Complex` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = 3.0_f64 + c;
/// ```
/// ## `Complex - T` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = c - 3.0_f64;
/// ```
/// ## `T - Complex` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = 3.0_f64 - c;
/// ```
/// ## `Complex * T` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = c * 3.0_f64;
/// ```
/// ## `T * Complex` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = 3.0_f64 * c;
/// ```
/// ## `Complex / T` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = c / 3.0_f64;
/// ```
/// ## `T / Complex` is rejected
/// ```compile_fail
/// use xenon::complex::Complex;
/// let c = Complex::new(1.0_f64, 2.0);
/// let _ = 3.0_f64 / c;
/// ```
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
    ///
    /// This is a `const fn` so it can be used in constant contexts.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::complex::Complex;
    /// let z = Complex::new(3.0_f64, 4.0);
    /// assert_eq!(z.re, 3.0);
    /// assert_eq!(z.im, 4.0);
    /// ```
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
    ///
    /// Note: `0` is simultaneously real and imaginary, so
    /// [`Complex::new(0.0, 0.0)`](Complex::new) satisfies both predicates.
    #[inline]
    pub fn is_real(self) -> bool {
        self.im == T::default()
    }

    /// Returns true if real part is zero.
    ///
    /// Note: `0` is simultaneously real and imaginary, so
    /// [`Complex::new(0.0, 0.0)`](Complex::new) satisfies both predicates.
    #[inline]
    pub fn is_imaginary(self) -> bool {
        self.re == T::default()
    }

    /// Creates a purely imaginary number (re = 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::complex::Complex;
    /// let z = Complex::from_imag(4.0_f64);
    /// assert_eq!(z, Complex::new(0.0, 4.0));
    /// ```
    #[inline]
    pub fn from_imag(im: T) -> Self {
        Self::new(T::default(), im)
    }

    /// Returns the complex conjugate: conj(a + bj) = a - bj.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::complex::Complex;
    /// let z = Complex::new(1.0_f64, 2.0);
    /// let conj = z.conj();
    /// assert_eq!(conj, Complex::new(1.0, -2.0));
    /// ```
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
    /// Converts a real number into a complex number with zero imaginary part.
    ///
    /// This is the only supported scalar-to-complex conversion path.
    ///
    /// # Examples
    ///
    /// ```
    /// use xenon::complex::Complex;
    /// assert_eq!(Complex::from(5.0_f64), Complex::new(5.0, 0.0));
    /// ```
    #[inline]
    fn from(re: T) -> Self {
        Self::new(re, T::default())
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
        assert_eq!(
            core::mem::align_of::<Complex<f64>>(),
            core::mem::align_of::<f64>()
        );
    }

    #[test]
    fn test_complex_layout_f32() {
        assert_eq!(core::mem::size_of::<Complex<f32>>(), 8);
        assert_eq!(
            core::mem::align_of::<Complex<f32>>(),
            core::mem::align_of::<f32>()
        );
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

    #[test]
    fn test_is_real_neg_zero_imag() {
        assert!(Complex::new(3.0_f64, -0.0).is_real());
    }

    #[test]
    fn test_complex_default() {
        assert_eq!(Complex::<f64>::default(), Complex::new(0.0, 0.0));
    }

    #[test]
    fn test_is_imaginary_neg_zero_real() {
        assert!(Complex::new(-0.0_f64, 5.0).is_imaginary());
    }

    /// Mirrors the doc example in the `Complex<T>` struct comment to
    /// guarantee the public usage stays compilable after W5T15.
    #[test]
    fn test_complex_docs_compile_example() {
        let z = Complex::new(3.0_f64, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-12);
        assert_eq!(format!("{}", z), "3+4j");
    }
}
