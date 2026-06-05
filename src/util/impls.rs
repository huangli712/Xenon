//! Utility operations: `clip`, `fill` / `try_fill`, `to_contiguous` /
//! `into_contiguous`.
//!
//! Provides inherent methods on [`TensorBase`].
//! See `docs/design/20-utility.md`.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::element::{Element, OrderedCompareElement};
use crate::error::{InvalidArgumentKind, StorageKindTag, XenonError};
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{
    ArcRepr, Owned, RawStorage, Storage, StorageIntoOwned, StorageMut, ViewMutRepr, ViewRepr,
};
use crate::tensor::{StorageKind, StorageSemantics, Tensor, TensorBase};

// ── Free functions ──

/// Validate that `min <= max`; reject NaN bounds.
///
/// Returns `Err(XenonError::InvalidArgument)` when `min > max` or either
/// bound is `NaN` (for floating-point types).
fn validate_clip_bounds<A>(min: &A, max: &A) -> Result<(), XenonError>
where
    A: OrderedCompareElement,
{
    if min.partial_cmp(max).is_none() || min > max {
        return Err(XenonError::InvalidArgument {
            operation: Cow::Borrowed("clip"),
            kind: InvalidArgumentKind::OperationSpecific {
                argument: Cow::Borrowed("min/max"),
                constraint: Cow::Borrowed(
                    "min <= max; NaN bounds are invalid for floating-point inputs",
                ),
            },
        });
    }
    Ok(())
}

/// Read-only branch: §5.3 row "View / Shared → InvalidStorageMode".
/// `tag` is the [`StorageKindTag`] for the concrete storage in this arm.
pub(crate) fn fill_try_read_only_err(tag: StorageKindTag) -> XenonError {
    XenonError::InvalidStorageMode {
        operation: Cow::Borrowed("Tensor::try_fill"),
        expected: StorageKindTag::Owned,
        actual: tag,
        shape: None,
    }
}

/// Crate-internal canonical predicate from `20-utility §6.3`.
///
/// Returns `true` iff **all four** conditions from `20-utility §6.3` hold:
///   1. `is_f_contiguous()` — strides satisfy F-order pattern
///   2. `storage_kind() == Owned` — sole-ownership
///   3. `offset() == 0` — no head padding
///   4. `storage_len() == product(shape)` — no tail padding
fn is_canonical_f_contiguous_owned<S, D, A>(t: &TensorBase<S, D>) -> bool
where
    S: Storage<Elem = A> + StorageSemantics,
    D: Dimension,
{
    if !t.is_f_contiguous() {
        return false;
    }
    if t.storage_kind() != StorageKind::Owned {
        return false;
    }
    if t.offset() != 0 {
        return false;
    }
    let logical = match t.raw_dim().checked_size() {
        Ok(n) => n,
        Err(_) => return false,
    };
    t.storage_len() == logical
}

// ── impl blocks ──

// ── clip ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: OrderedCompareElement + Clone,
{
    /// Clamp each logical element into `[min, max]`.
    ///
    /// Per `20-utility §5.1` / §6.4:
    /// - Bounds are validated **before** allocation.
    /// - `NaN` input values pass through unchanged (both `< min` and `> max`
    ///   are `false` under IEEE 754), matching NumPy `np.clip` semantics.
    /// - `NaN` *bounds* (or `min > max`) return `InvalidArgument`.
    ///
    /// # Errors
    ///
    /// - `XenonError::InvalidArgument` when bounds are invalid: either bound
    ///   is `NaN`, or `min > max` (`20-utility §5.1` / §6.4). Validated before
    ///   allocation.
    /// - `XenonError::InvalidShape` propagated from `Tensor::from_shape_vec`
    ///   when the shape's element count overflows `usize` (`ProductOverflow`).
    ///   Unreachable in practice — `self` already holds a valid shape — but
    ///   the failure mode is preserved by `?` for completeness.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is correct generic pattern"
    )]
    pub fn clip(&self, min: A, max: A) -> Result<Tensor<A, D>, XenonError> {
        validate_clip_bounds(&min, &max)?;
        let data: Vec<A> = self
            .iter()
            .map(|src| {
                if *src < min {
                    min.clone()
                } else if *src > max {
                    max.clone()
                } else {
                    src.clone()
                }
            })
            .collect();
        Tensor::from_shape_vec(self.raw_dim(), data)
    }
}

// ── fill ──

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Fill all logical elements with `value` in place
    /// (`20-utility §5.2`, primary public entry point).
    ///
    /// Stride-aware: iterates via `iter_mut()` so non-contiguous layouts and
    /// tensors with internal padding only have their logical elements
    /// touched (`§5.4`).
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is the correct generic pattern"
    )]
    pub fn fill(&mut self, value: A) {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
    }
}

// ── to_contiguous ──

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Clone,
{
    /// Ensure the tensor's data is stored contiguously in canonical F-order
    /// (`20-utility §5.5`). Always returns a fresh owned tensor; the input
    /// borrow is never aliased into the result.
    ///
    /// # Panics
    ///
    /// Does not panic in practice: on the repack path, `Iter` is an
    /// `ExactSizeIterator` whose `len()` equals `product(shape)`, which is
    /// exactly what `from_shape_vec` requires (see `10-iterator §5.5`,
    /// `18-construction §5.6`). A mismatch would indicate an iterator-contract
    /// bug elsewhere in the crate.
    pub fn to_contiguous(&self) -> Tensor<A, D> {
        if self.is_f_contiguous() {
            self.to_owned()
        } else {
            let values: Vec<A> = self.iter().cloned().collect();
            Tensor::from_shape_vec(self.raw_dim(), values)
                .expect("logical iteration length equals shape product")
        }
    }
}

// ── into_contiguous ──

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A> + StorageSemantics,
    D: Dimension + Clone,
    A: Element + Clone,
{
    /// Consume `self` and produce an owned, canonical F-order tensor
    /// (`20-utility §5.5`, §6.3). Reuses backing storage only when the input
    /// is already a canonical F-contiguous `Owned` tensor (predicate below).
    ///
    /// # Panics
    ///
    /// Panics if `Strides::f_contiguous(&dim)` fails. This cannot happen on the
    /// reuse path because `is_canonical_f_contiguous_owned` already
    /// established `is_f_contiguous()`, which implies the shape's element
    /// count fits `usize` (a construction-time invariant of `TensorBase`).
    pub fn into_contiguous(self) -> Tensor<A, D> {
        if is_canonical_f_contiguous_owned(&self) {
            let dim = self.raw_dim();
            let strides =
                Strides::f_contiguous(&dim).expect("canonical predicate implies shape is valid");
            let owned = self.storage.into_owned_storage();
            let flags = compute_layout_flags::<A, D>(&dim, &strides, owned.as_ptr());
            // SAFETY: is_canonical_f_contiguous_owned verified F-order, owned,
            // offset==0, storage_len==shape product. D: Clone ensured raw_dim()
            // snapshot precedes the move.
            unsafe { TensorBase::new_unchecked(owned, dim, strides, 0, flags, false) }
        } else {
            self.into_owned()
        }
    }
}

// ── try_fill ──

impl<D, A> TensorBase<Owned<A>, D>
where
    D: Dimension,
    A: Element + Clone,
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on Owned).
    ///
    /// §5.3 dispatch arm: Owned → `iter_mut()` write path.
    ///
    /// # Errors
    ///
    /// Infallible: always returns `Ok(())`. The `Result` return type exists for
    /// API uniformity with the read-only `ViewRepr` / `ArcRepr` variants of
    /// `try_fill`, which return `XenonError::InvalidStorageMode`.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is the correct generic pattern"
    )]
    pub fn try_fill(&mut self, value: A) -> Result<(), XenonError> {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
        Ok(())
    }
}

impl<'a, D, A> TensorBase<ViewRepr<'a, A>, D>
where
    D: Dimension,
    A: Element + Clone,
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on View).
    ///
    /// §5.3 dispatch arm: View / SharedReadOnly → `InvalidStorageMode`.
    /// Covers BOTH the plain `ReadOnly` ViewRepr and the runtime-tagged
    /// `SharedReadOnly` ViewRepr cases (derived_from_view_mut and zero-stride
    /// broadcast — see W8T4 `access_semantics()`): both collapse to the
    /// same `InvalidStorageMode` outcome here.
    ///
    /// # Errors
    ///
    /// Always returns `XenonError::InvalidStorageMode` with
    /// `storage_kind: StorageKindTag::View` — a `View` (read-only) tensor
    /// cannot be mutated through `try_fill`.
    pub fn try_fill(&mut self, _value: A) -> Result<(), XenonError> {
        Err(fill_try_read_only_err(StorageKindTag::View))
    }
}

impl<'a, D, A> TensorBase<ViewMutRepr<'a, A>, D>
where
    D: Dimension,
    A: Element + Clone,
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on ViewMut).
    ///
    /// §5.3 dispatch arm: ViewMut → `iter_mut()` write path.
    ///
    /// # Errors
    ///
    /// Infallible: always returns `Ok(())`. The `Result` return type exists for
    /// API uniformity with the read-only `ViewRepr` / `ArcRepr` variants of
    /// `try_fill`, which return `XenonError::InvalidStorageMode`.
    #[expect(
        clippy::clone_on_copy,
        reason = "generic over Clone (not Copy); .clone() is the correct generic pattern"
    )]
    pub fn try_fill(&mut self, value: A) -> Result<(), XenonError> {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
        Ok(())
    }
}

impl<D, A> TensorBase<ArcRepr<A>, D>
where
    D: Dimension,
    A: Element + Clone,
{
    /// Fallible fill (`20-utility §5.2`, secondary entry on Arc).
    ///
    /// §5.3 dispatch arm: Shared (read-only) → `InvalidStorageMode`.
    ///
    /// # Errors
    ///
    /// Always returns `XenonError::InvalidStorageMode` with
    /// `storage_kind: StorageKindTag::Shared` — an `ArcRepr` (shared,
    /// read-only) tensor cannot be mutated through `try_fill`.
    pub fn try_fill(&mut self, _value: A) -> Result<(), XenonError> {
        Err(fill_try_read_only_err(StorageKindTag::Shared))
    }
}

// ── Unit tests (§8.2) ──

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::tensor::{StorageKind, Tensor1, Tensor2};

    // ── clip tests (§8.2 / §7 T2) ──

    #[test]
    fn test_clip_basic() {
        let tensor = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_clip_no_change() {
        let tensor = Tensor1::from_shape_vec([3], vec![0.5, 1.0, 1.5])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_clip_nan() {
        let tensor = Tensor1::from_shape_vec([3], vec![1.0_f64, f64::NAN, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 4.0).expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
    }

    #[test]
    fn test_clip_nan_bound() {
        let tensor =
            Tensor1::from_shape_vec([1], vec![1.0_f64]).expect("from_shape_vec matching shape");
        assert!(matches!(
            tensor.clip(f64::NAN, 2.0),
            Err(XenonError::InvalidArgument { .. })
        ));
        assert!(matches!(
            tensor.clip(0.0, f64::NAN),
            Err(XenonError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn test_clip_integers() {
        let tensor = Tensor1::from_shape_vec([4], vec![-5_i32, 0, 5, 10])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0, 7).expect("valid clip bounds");
        let values: Vec<i32> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0, 0, 5, 7]);
    }

    #[test]
    fn test_clip_non_contiguous() {
        let tensor = Tensor2::from_shape_vec([2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        let clipped = transposed.clip(2.0, 5.0).expect("valid clip bounds");
        assert_eq!(clipped.shape(), &[3, 2]);
        assert_eq!(*clipped.get(&[0, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[0, 1]).expect("valid index"), 4.0);
        assert_eq!(*clipped.get(&[1, 0]).expect("valid index"), 2.0);
        assert_eq!(*clipped.get(&[1, 1]).expect("valid index"), 5.0);
        assert_eq!(*clipped.get(&[2, 0]).expect("valid index"), 3.0);
        assert_eq!(*clipped.get(&[2, 1]).expect("valid index"), 5.0);
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![2.0, 2.0, 3.0, 4.0, 5.0, 5.0]);
    }

    // ── fill tests (§8.2 / §7 T1) ──

    #[test]
    fn test_fill_basic() {
        let mut tensor = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        tensor.fill(2.5);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            vec![2.5, 2.5, 2.5]
        );
    }

    #[test]
    fn test_try_fill_read_only_returns_error() {
        let tensor =
            Tensor1::from_shape_vec([2], vec![1_i32, 2]).expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    #[test]
    fn test_try_fill_read_only_returns_read_only_storage() {
        let tensor =
            Tensor1::from_shape_vec([2], vec![1_i32, 2]).expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    #[test]
    #[ignore = "needs writable non-contiguous view primitive"]
    fn test_fill_non_contiguous() {
        todo!("activate after writable non-contiguous view constructor lands");
    }

    #[test]
    #[ignore = "needs writable strided sub-view primitive"]
    fn test_fill_padded_writes_logical_only() {
        todo!("activate after writable strided slice constructor lands");
    }

    #[test]
    fn test_try_fill_writable_matches_fill() {
        let mut t1 = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        let mut t2 = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        t1.fill(2.71);
        t2.try_fill(2.71).expect("try_fill on owned is writable");
        let v1: Vec<_> = t1.iter().copied().collect();
        let v2: Vec<_> = t2.iter().copied().collect();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_fill_empty() {
        let mut tensor = Tensor1::<f64>::zeros([0]).expect("zeros(empty)");
        tensor.fill(1.0); // must not panic
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_fill_invariant_all_equal_value() {
        for &n in &[1_usize, 2, 3, 5, 8, 13] {
            let mut t = Tensor1::<i64>::zeros([n]).expect("zeros(valid n)");
            t.fill(42);
            assert!(t.iter().all(|&x| x == 42), "n={}", n);
        }
    }

    // ── contiguous tests (§8.2) ──

    #[test]
    fn test_to_contiguous_f_order() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("from_shape_vec matching shape");
        let contiguous = tensor.to_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
    }

    #[test]
    fn test_to_contiguous_transposed_becomes_f() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("from_shape_vec matching shape");
        let contiguous = tensor.transpose().to_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.shape(), &[3, 2]);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
        assert_eq!(*contiguous.get(&[2, 0]).expect("valid index"), 5);
        assert_eq!(*contiguous.get(&[2, 1]).expect("valid index"), 6);
    }

    #[test]
    fn test_into_contiguous_reuses_canonical_owned_data() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("from_shape_vec matching shape");
        let contiguous = tensor.into_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.storage_kind(), StorageKind::Owned);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
    }

    #[test]
    #[ignore = "needs non-canonical owned constructor (tail padding / non-zero offset, W6+W8)"]
    fn test_into_contiguous_repacks_noncanonical_f_contiguous_owned() {
        todo!("activate after padding-aware owned constructor lands (W6+W8)");
    }

    #[test]
    fn test_into_contiguous_repacks_arc_input() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 2], vec![1, 2, 3, 4])
            .expect("from_shape_vec matching shape");
        let arc = tensor.into_shared();
        let contiguous = arc.into_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.storage_kind(), StorageKind::Owned);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
    }

    #[test]
    fn test_to_contiguous_non_contiguous() {
        let tensor = Tensor2::<i32>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])
            .expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        assert!(!transposed.is_f_contiguous());
        let contiguous = transposed.to_contiguous();
        assert!(contiguous.is_f_contiguous());
        assert_eq!(contiguous.shape(), &[3, 2]);
        assert_eq!(*contiguous.get(&[0, 0]).expect("valid index"), 1);
        assert_eq!(*contiguous.get(&[0, 1]).expect("valid index"), 2);
        assert_eq!(*contiguous.get(&[1, 0]).expect("valid index"), 3);
        assert_eq!(*contiguous.get(&[1, 1]).expect("valid index"), 4);
        assert_eq!(*contiguous.get(&[2, 0]).expect("valid index"), 5);
        assert_eq!(*contiguous.get(&[2, 1]).expect("valid index"), 6);
    }
}
