//! Complex number type `Complex<T>` with a sealed component bound.
//!
//! W5T1 provides the minimal skeleton: the `Complex<T>` struct, its `new()`
//! constructor, and the `ComplexFloat` sealed trait with `Sealed + Copy +
//! Default` supertraits. Subsequent Wave-5 tasks extend `ComplexFloat` and
//! add arithmetic, formatting, conversion, and math methods.

mod display;
mod math;
mod ops;

mod types;

pub use types::{Complex, ComplexFloat};

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
