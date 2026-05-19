use xenon::complex::Complex;
use xenon::element::{ComplexScalar, Numeric, RealScalar};

#[test]
fn test_math_uses_numeric_and_real_scalar() {
    let value = 4.0f64;
    assert_eq!(<f64 as RealScalar>::sqrt(value), 2.0);
    assert_eq!(<f64 as Numeric>::conjugate(value), value);

    // §8.5 ComplexScalar coverage (W4T10 provides Complex impls)
    let c = Complex::<f64>::new(3.0, 4.0);
    assert_eq!(<Complex<f64> as ComplexScalar>::norm(c), 5.0);
    assert_eq!(
        <Complex<f64> as Numeric>::conjugate(c),
        Complex::new(3.0, -4.0)
    );
}
