//! Element-wise comparison operations producing boolean tensors:
//! equal, not_equal, less, less_equal, greater, greater_equal.

use crate::error::XenonError;
use crate::broadcast::broadcast_with;
use crate::dispatch::{ExecPath, select_exec_path};

use super::binary::apply_binary_serial;

use crate::dimension::{BroadcastDimension, Dimension, Ix0};
use crate::element::{Element, OrderedCompareElement};
use crate::storage::Storage;
use crate::tensor::{Tensor, TensorBase};

#[cfg(feature = "parallel")]
use crate::dispatch::ParallelExecStrategy;

#[cfg(feature = "parallel")]
use super::binary_parallel::par_zip_checked;

// ----------------------------------------------------------------------------
// equal / not_equal for Element + PartialEq types
// ----------------------------------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: Element + PartialEq,
{
    /// Element-wise equality comparison. Returns a bool tensor.
    ///
    /// NaN comparison follows IEEE 754: `equal(NaN, NaN)` is element-wise
    /// `false` (consistent with `f64::partial_cmp(NaN, NaN) == None`).
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn equal<S2, DB>(
        &self,
        other: &TensorBase<S2, DB>,
    ) -> Result<Tensor<bool, <D as BroadcastDimension<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDimension<DB>,
        DB: Dimension,
    {
        apply_compare_with_dispatch(self, other, |a, b| a == b)
    }

    /// Element-wise inequality comparison.
    /// `not_equal(NaN, NaN)` is element-wise `true` per IEEE 754.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn not_equal<S2, DB>(
        &self,
        other: &TensorBase<S2, DB>,
    ) -> Result<Tensor<bool, <D as BroadcastDimension<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDimension<DB>,
        DB: Dimension,
    {
        apply_compare_with_dispatch(self, other, |a, b| a != b)
    }
}

// scalar variants for equal/not_equal
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + BroadcastDimension<Ix0, Output = D>,
    A: Element + PartialEq,
{
    /// Element-wise equality comparison with a scalar right-hand side.
    ///
    /// **Type bound**: `A: Element + PartialEq`. Every element type
    /// implements equality; `bool` and `Complex` are also supported.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `from_scalar` is infallible for `Ix0`,
    /// and the `BroadcastDim<Ix0, Output = D>` bound on the `impl` block
    /// guarantees scalar broadcast always succeeds. The `expect` messages
    /// document the invariant for future refactors.
    pub fn equal_scalar(&self, scalar: A) -> Tensor<bool, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.equal(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }

    /// Element-wise not-equal comparison with a scalar right-hand side.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `from_scalar` is infallible for `Ix0`,
    /// and the `BroadcastDim<Ix0, Output = D>` bound on the `impl` block
    /// guarantees scalar broadcast always succeeds. The `expect` messages
    /// document the invariant for future refactors.
    pub fn not_equal_scalar(&self, scalar: A) -> Tensor<bool, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.not_equal(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }
}

// ----------------------------------------------------------------------------
// less / less_equal / greater / greater_equal for OrderedCompareElement types
// ----------------------------------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
    A: OrderedCompareElement,
{
    /// Element-wise less-than comparison.
    ///
    /// **Type bound**: `A: OrderedCompareElement`, which is sealed to
    /// `i32` / `i64` / `f32` / `f64`. Complex and bool tensors are
    /// excluded at compile time.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn less<S2, DB>(
        &self,
        other: &TensorBase<S2, DB>,
    ) -> Result<Tensor<bool, <D as BroadcastDimension<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDimension<DB>,
        DB: Dimension,
    {
        apply_compare_with_dispatch(self, other, |a, b| a < b)
    }

    /// Element-wise less-or-equal comparison.
    /// Uses a single broadcast traversal that evaluates `<=` directly,
    /// avoiding intermediate bool tensors.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn less_equal<S2, DB>(
        &self,
        other: &TensorBase<S2, DB>,
    ) -> Result<Tensor<bool, <D as BroadcastDimension<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDimension<DB>,
        DB: Dimension,
    {
        apply_compare_with_dispatch(self, other, |a, b| a <= b)
    }

    /// Element-wise greater-than comparison.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn greater<S2, DB>(
        &self,
        other: &TensorBase<S2, DB>,
    ) -> Result<Tensor<bool, <D as BroadcastDimension<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDimension<DB>,
        DB: Dimension,
    {
        apply_compare_with_dispatch(self, other, |a, b| a > b)
    }

    /// Element-wise greater-or-equal comparison.
    /// Uses a single broadcast traversal that evaluates `>=` directly,
    /// avoiding intermediate bool tensors.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::BroadcastError` if `self.shape()` and
    /// `other.shape()` are not broadcast-compatible.
    pub fn greater_equal<S2, DB>(
        &self,
        other: &TensorBase<S2, DB>,
    ) -> Result<Tensor<bool, <D as BroadcastDimension<DB>>::Output>, XenonError>
    where
        S2: Storage<Elem = A>,
        D: BroadcastDimension<DB>,
        DB: Dimension,
    {
        apply_compare_with_dispatch(self, other, |a, b| a >= b)
    }
}

// scalar variants for less/less_equal/greater/greater_equal
impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + BroadcastDimension<Ix0, Output = D>,
    A: OrderedCompareElement,
{
    /// Element-wise less-than comparison with a scalar right-hand side.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `from_scalar` is infallible for `Ix0`,
    /// and the `BroadcastDim<Ix0, Output = D>` bound on the `impl` block
    /// guarantees scalar broadcast always succeeds. The `expect` messages
    /// document the invariant for future refactors.
    pub fn less_scalar(&self, scalar: A) -> Tensor<bool, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.less(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }

    /// Element-wise less-or-equal comparison with a scalar right-hand side.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `from_scalar` is infallible for `Ix0`,
    /// and the `BroadcastDim<Ix0, Output = D>` bound on the `impl` block
    /// guarantees scalar broadcast always succeeds. The `expect` messages
    /// document the invariant for future refactors.
    pub fn less_equal_scalar(&self, scalar: A) -> Tensor<bool, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.less_equal(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }

    /// Element-wise greater-than comparison with a scalar right-hand side.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `from_scalar` is infallible for `Ix0`,
    /// and the `BroadcastDim<Ix0, Output = D>` bound on the `impl` block
    /// guarantees scalar broadcast always succeeds. The `expect` messages
    /// document the invariant for future refactors.
    pub fn greater_scalar(&self, scalar: A) -> Tensor<bool, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.greater(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }

    /// Element-wise greater-or-equal comparison with a scalar right-hand side.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: `from_scalar` is infallible for `Ix0`,
    /// and the `BroadcastDim<Ix0, Output = D>` bound on the `impl` block
    /// guarantees scalar broadcast always succeeds. The `expect` messages
    /// document the invariant for future refactors.
    pub fn greater_equal_scalar(&self, scalar: A) -> Tensor<bool, D> {
        let other = Tensor::<A, Ix0>::from_scalar(scalar)
            .expect("from_scalar never fails");
        self.greater_equal(&other)
            .expect("scalar broadcast cannot fail: BroadcastDim<Ix0> \
                     guarantees compatibility")
    }
}

// ----------------------------------------------------------------------------
// Dispatch-aware comparison helper
// ----------------------------------------------------------------------------

/// Dispatch-aware broadcast comparison helper used by all comparison
/// operators. Output is always `bool`. Comparison exposes no SIMD kernel,
/// so the SIMD execution path collapses to the scalar baseline
/// (`apply_binary_serial`); only the Serial and Parallel paths chosen by
/// `select_exec_path` are distinct.
pub(in crate::math) fn apply_compare_with_dispatch<A, S1, S2, D1, D2, F>(
    a: &TensorBase<S1, D1>,
    b: &TensorBase<S2, D2>,
    op: F,
) -> Result<Tensor<bool, <D1 as BroadcastDimension<D2>>::Output>, XenonError>
where
    A: Element,
    S1: Storage<Elem = A>,
    S2: Storage<Elem = A>,
    D1: Dimension + BroadcastDimension<D2>,
    D2: Dimension,
    F: Fn(A, A) -> bool + Copy + Send + Sync,
{
    let (a_view, b_view, out_dim) = broadcast_with(a, b)?;
    let len = out_dim.checked_size().expect("broadcast_shape validated");
    let both_contiguous = a_view.is_f_contiguous() && b_view.is_f_contiguous();
    let both_aligned = a_view.is_aligned() && b_view.is_aligned();
    let (path, guard) = select_exec_path(len, both_contiguous, both_aligned);

    let result = match path {
        // No SIMD kernel for comparisons; Serial and Simd both use the
        // scalar baseline.
        ExecPath::Serial | ExecPath::Simd => {
            let _ = guard;
            apply_binary_serial(&a_view, &b_view, op)
        },
        ExecPath::Parallel => {
            #[cfg(feature = "parallel")]
            {
                let strat = ParallelExecStrategy::auto();
                let g = guard
                    .expect("ExecPath::Parallel must carry a ParallelGuard");
                par_zip_checked(
                    a,
                    b,
                    &out_dim,
                    &strat, g, |a, b| Ok(op(*a, *b))
                )?
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = guard;
                apply_binary_serial(&a_view, &b_view, op)
            }
        },
    };
    Ok(result)
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::dimension::{Ix1, Ix2};
    use crate::tensor::Tensor;

    /// `equal` returns true for matching f64 elements and false for
    /// NaN == NaN per IEEE 754.
    #[test]
    fn test_equal_f64() {
        let a = Tensor::<f64, Ix1>::from_shape_vec(
            [3],
            vec![1.0, 2.0, f64::NAN]
        ).expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec(
            [3],
            vec![1.0, 2.0, f64::NAN]
        ).expect("valid tensor shape");
        let result = a.equal(&b)
            .expect("broadcast succeeds in test");
        assert!(
            *result.get(&[0]).expect("valid index"),
            "1.0 == 1.0 should be true"
        );
        assert!(
            *result.get(&[1]).expect("valid index"),
            "2.0 == 2.0 should be true"
        );
        assert!(
            !*result.get(&[2]).expect("valid index"),
            "NaN == NaN should be false per IEEE 754"
        );
    }

    /// `not_equal(NaN, NaN)` returns true per IEEE 754.
    #[test]
    fn test_not_equal_nan() {
        let a = Tensor::<f64, Ix1>::from_shape_vec(
            [1],
            vec![f64::NAN]
        ).expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec(
            [1],
            vec![f64::NAN]
        ).expect("valid tensor shape");
        let result = a.not_equal(&b)
            .expect("broadcast succeeds in test");
        assert!(
            *result.get(&[0]).expect("valid index"),
            "NaN != NaN should be true per IEEE 754"
        );
    }

    /// `equal_scalar` matches only the element equal to the scalar.
    #[test]
    fn test_equal_scalar() {
        let t = Tensor::<i32, Ix1>::from_shape_vec(
            [3],
            vec![1, 2, 3]
        ).expect("valid tensor shape");
        let r = t.equal_scalar(2);
        assert!(!*r.get(&[0]).expect("valid index"));
        assert!(*r.get(&[1]).expect("valid index"));
        assert!(!*r.get(&[2]).expect("valid index"));
    }

    /// `less` returns true only where the left lane is strictly less than
    /// the right lane.
    #[test]
    fn test_less_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 5, 10])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![2, 5, 8])
            .expect("valid tensor shape");
        let result = a.less(&b).expect("broadcast succeeds in test");
        assert!(
            *result.get(&[0]).expect("valid index"),
            "1 < 2 should be true"
        );
        assert!(
            !*result.get(&[1]).expect("valid index"),
            "5 < 5 should be false"
        );
        assert!(
            !*result.get(&[2]).expect("valid index"),
            "10 < 8 should be false"
        );
    }

    /// `less_equal` returns true for both strictly-less and equal lanes.
    #[test]
    fn test_less_equal_i32() {
        let a = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 5, 10])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec([3], vec![2, 5, 8])
            .expect("valid tensor shape");
        let r = a.less_equal(&b).expect("broadcast succeeds in test");
        assert!(r.get(&[0]).expect("valid index"));
        assert!(r.get(&[1]).expect("valid index"));
        assert!(!r.get(&[2]).expect("valid index"));
    }

    /// `greater` returns false for any comparison involving NaN per IEEE 754.
    #[test]
    fn test_nan_comparison() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([2], vec![f64::NAN, 1.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([2], vec![1.0, f64::NAN])
            .expect("valid tensor shape");
        let result = a.greater(&b).expect("broadcast succeeds in test");
        assert!(
            !*result.get(&[0]).expect("valid index"),
            "NaN > 1.0 should be false"
        );
        assert!(
            !*result.get(&[1]).expect("valid index"),
            "1.0 > NaN should be false"
        );
    }

    /// `greater_equal` returns true for both strictly-greater and equal lanes.
    #[test]
    fn test_greater_equal_f64() {
        let a = Tensor::<f64, Ix1>::from_shape_vec([3], vec![2.0, 5.0, 1.0])
            .expect("valid tensor shape");
        let b = Tensor::<f64, Ix1>::from_shape_vec([3], vec![1.0, 5.0, 9.0])
            .expect("valid tensor shape");
        let r = a.greater_equal(&b).expect("broadcast succeeds in test");
        assert!(r.get(&[0]).expect("valid index"));
        assert!(r.get(&[1]).expect("valid index"));
        assert!(!r.get(&[2]).expect("valid index"));
    }

    /// `not_equal_scalar` is true for every element NOT equal to the scalar.
    #[test]
    fn test_not_equal_scalar() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 2, 3])
            .expect("valid tensor shape");
        let r = t.not_equal_scalar(2);
        assert!(*r.get(&[0]).expect("valid index"));
        assert!(!*r.get(&[1]).expect("valid index"));
        assert!(*r.get(&[2]).expect("valid index"));
    }

    /// `less_scalar` is true where the element is strictly less than
    /// the scalar.
    #[test]
    fn test_less_scalar() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 5, 10])
            .expect("valid tensor shape");
        let r = t.less_scalar(5);
        assert!(*r.get(&[0]).expect("valid index"));
        assert!(!*r.get(&[1]).expect("valid index"));
        assert!(!*r.get(&[2]).expect("valid index"));
    }

    /// `less_equal_scalar` is true where the element is `<=` the scalar.
    #[test]
    fn test_less_equal_scalar() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 5, 10])
            .expect("valid tensor shape");
        let r = t.less_equal_scalar(5);
        assert!(*r.get(&[0]).expect("valid index"));
        assert!(*r.get(&[1]).expect("valid index"));
        assert!(!*r.get(&[2]).expect("valid index"));
    }

    /// `greater_scalar` is true where the element is strictly greater
    /// than the scalar.
    #[test]
    fn test_greater_scalar() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 5, 10])
            .expect("valid tensor shape");
        let r = t.greater_scalar(5);
        assert!(!*r.get(&[0]).expect("valid index"));
        assert!(!*r.get(&[1]).expect("valid index"));
        assert!(*r.get(&[2]).expect("valid index"));
    }

    /// `greater_equal_scalar` is true where the element is `>=` the scalar.
    #[test]
    fn test_greater_equal_scalar() {
        let t = Tensor::<i32, Ix1>::from_shape_vec([3], vec![1, 5, 10])
            .expect("valid tensor shape");
        let r = t.greater_equal_scalar(5);
        assert!(!*r.get(&[0]).expect("valid index"));
        assert!(*r.get(&[1]).expect("valid index"));
        assert!(*r.get(&[2]).expect("valid index"));
    }

    /// Comparison broadcasts rank-2 inputs `[3,1]` vs `[1,4]` to `[3,4]`.
    #[test]
    fn test_compare_broadcast() {
        let a = Tensor::<i32, Ix2>::from_shape_vec([3, 1], vec![1, 5, 9])
            .expect("valid tensor shape");
        let b = Tensor::<i32, Ix2>::from_shape_vec([1, 4], vec![2, 5, 8, 10])
            .expect("valid tensor shape");
        let r = a.less(&b).expect("broadcast succeeds in test");
        assert_eq!(r.shape(), &[3, 4]);
        
        // row 0 (a=1): 1<2,1<5,1<8,1<10 -> all true
        assert!(*r.get(&[0, 0]).expect("valid index"));
        assert!(*r.get(&[0, 3]).expect("valid index"));
        
        // row 1 (a=5): 5<2 false, 5<5 false, 5<8 true, 5<10 true
        assert!(!*r.get(&[1, 0]).expect("valid index"));
        assert!(!*r.get(&[1, 1]).expect("valid index"));
        assert!(*r.get(&[1, 2]).expect("valid index"));
        
        // row 2 (a=9): 9<2,9<5,9<8 false, 9<10 true
        assert!(!*r.get(&[2, 2]).expect("valid index"));
        assert!(*r.get(&[2, 3]).expect("valid index"));
    }

    /// `equal`/`less`/`greater` produce identical results on the Parallel path
    /// (forced via a threshold of 1) as on the Serial path (parallel disabled
    /// via the 0 sentinel). Guards the inlined `ExecPath::Parallel` arm of
    /// `apply_compare_with_dispatch`.
    #[cfg(feature = "parallel")]
    #[test]
    fn test_compare_parallel_matches_serial() {
        use crate::dispatch::ThresholdTestGuard;
        use crate::dispatch::set_parallel_threshold;

        let a = Tensor::<i32, Ix1>::from_shape_vec(
            [128],
            (0..128).collect()
        ).expect("valid tensor shape");
        let b = Tensor::<i32, Ix1>::from_shape_vec(
            [128],
            (0..128).map(|x| 64 - x).collect()
        ).expect("valid tensor shape");

        let _guard = ThresholdTestGuard::new();
        set_parallel_threshold(0);
        let eq_serial = a.equal(&b).expect("broadcast succeeds in test");
        let lt_serial = a.less(&b).expect("broadcast succeeds in test");
        let gt_serial = a.greater(&b).expect("broadcast succeeds in test");
        set_parallel_threshold(1);
        let eq_par = a.equal(&b).expect("broadcast succeeds in test");
        let lt_par = a.less(&b).expect("broadcast succeeds in test");
        let gt_par = a.greater(&b).expect("broadcast succeeds in test");

        for i in 0..128 {
            let ix = [i];
            assert_eq!(
                eq_par.get(&ix).expect("valid index"),
                eq_serial.get(&ix).expect("valid index"),
                "equal parallel/serial mismatch at {i}"
            );
            assert_eq!(
                lt_par.get(&ix).expect("valid index"),
                lt_serial.get(&ix).expect("valid index"),
                "less parallel/serial mismatch at {i}"
            );
            assert_eq!(
                gt_par.get(&ix).expect("valid index"),
                gt_serial.get(&ix).expect("valid index"),
                "greater parallel/serial mismatch at {i}"
            );
        }
    }
}
