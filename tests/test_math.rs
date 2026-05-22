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

// ── Math operation integration tests ──

use xenon::tensor::{Tensor1, Tensor2};

#[test]
fn test_add_same_shape() {
    let a = Tensor2::<f64>::from_shape_vec([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid test input");
    let b = Tensor2::<f64>::from_shape_vec([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).expect("valid test input");
    let result = a.add(&b).expect("broadcast succeeds");
    assert_eq!(result.shape(), &[2, 3]);
    assert!((*result.try_at((0, 0)).expect("valid index") - 11.0).abs() < 1e-10);
    assert!((*result.try_at((1, 2)).expect("valid index") - 66.0).abs() < 1e-10);
}

#[test]
fn test_sub_mul_div() {
    let a = Tensor2::<f64>::from_shape_vec([2, 2], vec![8.0, 2.0, 9.0, 3.0]).expect("valid test input");
    let b = Tensor2::<f64>::from_shape_vec([2, 2], vec![2.0, 4.0, 3.0, 5.0]).expect("valid test input");
    let sub = a.sub(&b).expect("broadcast succeeds");
    let mul = a.mul(&b).expect("broadcast succeeds");
    let div = a.div(&b).expect("broadcast succeeds");
    assert!((*sub.try_at((0, 0)).expect("valid index") - 6.0).abs() < 1e-10);
    assert!((*mul.try_at((0, 0)).expect("valid index") - 16.0).abs() < 1e-10);
    assert!((*div.try_at((0, 0)).expect("valid index") - 4.0).abs() < 1e-10);
}

#[test]
fn test_add_scalar() {
    let t = Tensor1::<f64>::from_shape_vec([3], vec![1.0, 2.0, 3.0]).expect("valid test input");
    let r = t.add_scalar(10.0);
    assert!((*r.try_at((0,)).expect("valid index") - 11.0).abs() < 1e-10);
    assert!((*r.try_at((2,)).expect("valid index") - 13.0).abs() < 1e-10);
}

#[test]
fn test_abs_signum() {
    let t = Tensor1::<i32>::from_shape_vec([3], vec![-5, 0, 3]).expect("valid test input");
    let a = t.abs();
    assert_eq!(*a.try_at((0,)).expect("valid index"), 5);
    assert_eq!(*a.try_at((2,)).expect("valid index"), 3);
    let s = t.signum();
    assert_eq!(*s.try_at((0,)).expect("valid index"), -1);
    assert_eq!(*s.try_at((1,)).expect("valid index"), 0);
    assert_eq!(*s.try_at((2,)).expect("valid index"), 1);
}

#[test]
fn test_neg() {
    let t = Tensor1::<f64>::from_shape_vec([2], vec![1.0, -2.0]).expect("valid test input");
    let r = t.neg();
    assert_eq!(*r.try_at((0,)).expect("valid index"), -1.0);
    assert_eq!(*r.try_at((1,)).expect("valid index"), 2.0);
}

#[test]
fn test_sin_sqrt_exp_ln_floor_ceil() {
    let t = Tensor1::<f64>::from_shape_vec([3], vec![0.0, 4.0, 2.0]).expect("valid test input");
    assert!((*t.sin().try_at((0,)).expect("valid index") - 0.0).abs() < 1e-10);
    assert!((*t.sqrt().try_at((1,)).expect("valid index") - 2.0).abs() < 1e-10);
    assert!((*t.exp().try_at((0,)).expect("valid index") - 1.0).abs() < 1e-10);
    assert!((*t.ln().try_at((2,)).expect("valid index") - (2.0f64).ln()).abs() < 1e-10);
    let f = Tensor1::<f64>::from_shape_vec([2], vec![1.7, 1.3]).expect("valid test input");
    assert_eq!(*f.floor().try_at((0,)).expect("valid index"), 1.0);
    assert_eq!(*f.ceil().try_at((1,)).expect("valid index"), 2.0);
}

#[test]
fn test_complex_math() {
    let t = Tensor1::from_shape_vec([2], vec![Complex::new(3.0, 4.0), Complex::new(1.0, 2.0)]).expect("valid test input");
    let m = t.modulus();
    assert!((*m.try_at((0,)).expect("valid index") - 5.0).abs() < 1e-10);
    let c = t.conjugate();
    assert_eq!(c.try_at((1,)).expect("valid index").im(), -2.0);
    assert_eq!(c.try_at((1,)).expect("valid index").re(), 1.0);
}

#[test]
fn test_bool_not() {
    let t = Tensor1::from_shape_vec([3], vec![true, false, true]).expect("valid test input");
    let r = t.not();
    assert!(!*r.try_at((0,)).expect("valid index"));
    assert!(*r.try_at((1,)).expect("valid index"));
    assert!(!*r.try_at((2,)).expect("valid index"));
}

#[test]
fn test_compare_equal_not_equal() {
    let a = Tensor1::<i32>::from_shape_vec([3], vec![1, 2, 3]).expect("valid test input");
    let b = Tensor1::<i32>::from_shape_vec([3], vec![1, 2, 4]).expect("valid test input");
    let eq = a.equal(&b).expect("broadcast succeeds");
    assert!(*eq.try_at((0,)).expect("valid index"));
    assert!(!*eq.try_at((2,)).expect("valid index"));
    let neq = a.not_equal(&b).expect("broadcast succeeds");
    assert!(!*neq.try_at((0,)).expect("valid index"));
    assert!(*neq.try_at((2,)).expect("valid index"));
}

#[test]
fn test_compare_less_greater() {
    let a = Tensor1::<i32>::from_shape_vec([3], vec![1, 5, 10]).expect("valid test input");
    let b = Tensor1::<i32>::from_shape_vec([3], vec![2, 5, 8]).expect("valid test input");
    let less = a.less(&b).expect("broadcast succeeds");
    assert!(*less.try_at((0,)).expect("valid index"));
    assert!(!*less.try_at((1,)).expect("valid index"));
    let greater = a.greater(&b).expect("broadcast succeeds");
    assert!(!*greater.try_at((0,)).expect("valid index"));
    assert!(*greater.try_at((2,)).expect("valid index"));
}

#[test]
fn test_square() {
    let t = Tensor1::<f64>::from_shape_vec([3], vec![2.0, 3.0, 4.0]).expect("valid test input");
    let r = t.square();
    assert!((*r.try_at((0,)).expect("valid index") - 4.0).abs() < 1e-10);
    assert!((*r.try_at((2,)).expect("valid index") - 16.0).abs() < 1e-10);
}

#[test]
#[should_panic(expected = "overflow")]
fn test_integer_add_overflow_panics() {
    let a = Tensor1::<i32>::from_shape_vec([1], vec![i32::MAX]).expect("valid test input");
    let b = Tensor1::<i32>::from_shape_vec([1], vec![1]).expect("valid test input");
    let _ = a.add(&b);
}

#[test]
#[should_panic(expected = "div_by_zero")]
fn test_integer_divide_by_zero_panics() {
    let a = Tensor1::<i32>::from_shape_vec([1], vec![1]).expect("valid test input");
    let b = Tensor1::<i32>::from_shape_vec([1], vec![0]).expect("valid test input");
    let _ = a.div(&b);
}

#[test]
#[should_panic(expected = "overflow")]
fn test_integer_min_abs_panics() {
    let t = Tensor1::<i32>::from_shape_vec([1], vec![i32::MIN]).expect("valid test input");
    let _ = t.abs();
}

#[test]
#[should_panic(expected = "overflow")]
fn test_integer_min_div_neg_one_panics() {
    let a = Tensor1::<i32>::from_shape_vec([1], vec![i32::MIN]).expect("valid test input");
    let b = Tensor1::<i32>::from_shape_vec([1], vec![-1]).expect("valid test input");
    let _ = a.div(&b);
}

#[test]
#[should_panic(expected = "overflow")]
fn test_integer_neg_min_panics() {
    let t = Tensor1::<i32>::from_shape_vec([1], vec![i32::MIN]).expect("valid test input");
    let _ = t.neg();
}

#[test]
#[should_panic(expected = "overflow")]
fn test_integer_square_overflow_panics() {
    let t = Tensor1::<i32>::from_shape_vec([1], vec![i32::MAX]).expect("valid test input");
    let _ = t.square();
}
