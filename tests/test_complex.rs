use xenon::complex::Complex;

// ── Arithmetic integration ──

#[test]
fn test_add_sub_mul_div_neg() {
    let a = Complex::new(1.0_f64, 2.0);
    let b = Complex::new(3.0_f64, 4.0);
    assert_eq!(a + b, Complex::new(4.0, 6.0));
    assert_eq!(a - b, Complex::new(-2.0, -2.0));
    assert_eq!(a * b, Complex::new(-5.0, 10.0));
    let q = Complex::new(6.0_f64, 8.0) / Complex::new(3.0, 4.0);
    assert!((q.re - 2.0).abs() < 1e-12);
    assert!(q.im.abs() < 1e-12);
    assert_eq!(-a, Complex::new(-1.0, -2.0));
}

// ── Constructors and accessors ──

#[test]
fn test_constructors_and_accessors() {
    let z = Complex::new(3.0_f64, 4.0);
    assert_eq!(z.re(), 3.0);
    assert_eq!(z.im(), 4.0);
    assert_eq!(Complex::from(5.0_f64), Complex::new(5.0, 0.0));
    assert_eq!(Complex::from_imag(6.0_f64), Complex::new(0.0, 6.0));
    assert_eq!(z.conj(), Complex::new(3.0, -4.0));
}

// ── Predicates ──

#[test]
fn test_complex_predicates() {
    assert!(Complex::new(3.0_f64, 0.0).is_real());
    assert!(Complex::new(0.0, 3.0_f64).is_imaginary());
    assert!(!Complex::new(3.0_f64, 4.0).is_real());
    assert!(!Complex::new(3.0_f64, 4.0).is_imaginary());

    assert!(Complex::new(f64::NAN, 0.0).is_nan());
    assert!(!Complex::new(1.0_f64, 2.0).is_nan());
    assert!(Complex::new(1.0_f64, 2.0).is_finite());
    assert!(!Complex::new(f64::INFINITY, 0.0).is_finite());
}

// ── Math methods ──

#[test]
fn test_complex_norm() {
    let z = Complex::new(3.0_f64, 4.0);
    assert_eq!(z.norm(), 5.0);
    assert_eq!(z.norm_sqr(), 25.0);
}

// ── Display formatting ──

#[test]
fn test_complex_display() {
    assert_eq!(Complex::new(3.0_f64, 4.0).to_string(), "3+4j");
    assert_eq!(Complex::new(3.0_f64, -4.0).to_string(), "3-4j");
    assert_eq!(Complex::new(0.0_f64, 5.0).to_string(), "5j");
    assert_eq!(Complex::new(5.0_f64, 0.0).to_string(), "5");
    assert_eq!(Complex::new(0.0_f64, 0.0).to_string(), "0");
}

#[test]
fn test_display_neg_zero_preserved() {
    assert_eq!(Complex::new(3.0_f64, -0.0).to_string(), "3-0j");
}

#[test]
fn test_display_nan_imag() {
    assert_eq!(format!("{}", Complex::new(1.0_f64, f64::NAN)), "1+NaNj");
}

// ── Layout guarantees ──

#[test]
fn test_complex_layout() {
    assert_eq!(core::mem::size_of::<Complex<f32>>(), 8);
    assert_eq!(core::mem::size_of::<Complex<f64>>(), 16);
    assert_eq!(
        core::mem::align_of::<Complex<f32>>(),
        core::mem::align_of::<f32>()
    );
    assert_eq!(
        core::mem::align_of::<Complex<f64>>(),
        core::mem::align_of::<f64>()
    );
}

// ── Boundary: zero division ──

#[test]
fn test_div_by_zero() {
    let z = Complex::new(1.0_f64, 2.0);
    let zero = Complex::new(0.0_f64, 0.0);
    let result = z / zero;
    assert!(result.re.is_nan() || result.re.is_infinite());
    assert!(result.im.is_nan() || result.im.is_infinite());
}

// ── Boundary: NaN propagation ──

#[test]
fn test_nan_propagation() {
    let nan = Complex::new(f64::NAN, f64::NAN);
    let sum = nan + Complex::new(1.0_f64, 2.0);
    assert!(sum.is_nan());
    let prod = nan * Complex::new(1.0_f64, 2.0);
    assert!(prod.is_nan());
    assert_ne!(nan, nan);
}

// ── Boundary: large magnitude norm avoids overflow ──

#[test]
fn test_large_norm_no_overflow() {
    let big = 1.0e200_f64;
    let z = Complex::new(big, big);
    assert!(z.norm().is_finite());
}

// ── Boundary: subnormal magnitudes ──

/// `f64::MIN_POSITIVE²` underflows to 0 in plain f64 arithmetic.
/// A naive `sqrt(re²+im²)` implementation would return 0 here;
/// `hypot()` must keep the result non-zero and finite.
#[test]
fn test_subnormal_norm() {
    let tiny = Complex::new(f64::MIN_POSITIVE, f64::MIN_POSITIVE);
    assert!(tiny.norm().is_finite() && tiny.norm() > 0.0);
}

// ── Property: (z * z.conj()).re == norm_sqr  &&  .im == 0 ──

#[test]
fn test_conjugate_norm_invariant() {
    let cases = [
        Complex::new(1.0_f64, 2.0),
        Complex::new(-3.0_f64, 4.0),
        Complex::new(0.5_f64, -0.25),
        Complex::new(1.0e6_f64, 1.0e-6),
    ];
    for z in cases {
        let product = z * z.conj();
        assert!(
            (product.re - z.norm_sqr()).abs() < 1e-9 * z.norm_sqr().abs().max(1.0)
        );
        assert!(
            product.im.abs() < 1e-9 * z.norm_sqr().abs().max(1.0)
        );
    }
}

// ── Property: (z / w) * w ≈ z ──

#[test]
fn test_div_then_mul_roundtrip() {
    let cases = [
        (Complex::new(1.0_f64, 2.0), Complex::new(3.0_f64, 4.0)),
        (Complex::new(-1.0_f64, 0.5), Complex::new(2.0_f64, -1.0)),
        (Complex::new(7.0_f64, 11.0), Complex::new(-0.5_f64, 0.25)),
    ];
    for (z, w) in cases {
        let recovered = (z / w) * w;
        let tol = 1e-9 * z.norm().max(1.0);
        assert!(
            (recovered.re - z.re).abs() < tol,
            "z={z}, recovered={recovered}"
        );
        assert!(
            (recovered.im - z.im).abs() < tol,
            "z={z}, recovered={recovered}"
        );
    }
}
