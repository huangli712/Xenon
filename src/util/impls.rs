//! Utility operations: `clip`, `fill` / `try_fill`, `to_contiguous` /
//! `into_contiguous`.
//!
//! All operations are exposed as inherent methods on [`TensorBase`].

use std::borrow::Cow;

use crate::error::{InvalidArgumentKind, StorageKindTag, XenonError};
use crate::dimension::Dimension;
use crate::element::{Element, OrderedCompareElement};
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{Owned, ViewRepr, ViewMutRepr, ArcRepr};
use crate::storage::{RawStorage, Storage, StorageMut, StorageIntoOwned};
use crate::tensor::{StorageKind, StorageSemantics, Tensor, TensorBase};

// --- Free functions ---------------------------------------------------------

/// Validate that `min <= max`; reject NaN bounds.
///
/// Returns [`XenonError::InvalidArgument`] when `min > max` or either
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

/// Construct an [`XenonError::InvalidStorageMode`] error for the read-only
/// branch of `try_fill`.
///
/// `tag` indicates the concrete storage kind that was found
/// ([`StorageKindTag::View`] or [`StorageKindTag::Shared`]).
pub(crate) fn fill_try_read_only_err(tag: StorageKindTag) -> XenonError {
    XenonError::InvalidStorageMode {
        operation: Cow::Borrowed("Tensor::try_fill"),
        expected: StorageKindTag::Owned,
        actual: tag,
        shape: None,
    }
}

/// Check whether a tensor satisfies the canonical F-order owned predicate.
///
/// Returns `true` iff all four conditions hold:
///   1. `is_f_contiguous()` — strides satisfy F-order pattern.
///   2. `storage_kind() == Owned` — sole-ownership, not view-derived.
///   3. `offset() == 0` — no head padding.
///   4. `storage_len() == product(shape)` — no tail padding.
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

// --- clip -------------------------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: OrderedCompareElement + Clone,
{
    /// Clamp each logical element into `[min, max]`.
    ///
    /// Bounds are validated before allocation. `NaN` input values pass
    /// through unchanged (both `< min` and `> max` are `false` under
    /// IEEE 754), matching NumPy `np.clip` semantics. `NaN` bounds or
    /// `min > max` return [`XenonError::InvalidArgument`].
    ///
    /// # Errors
    ///
    /// Returns [`XenonError::InvalidArgument`] when either bound is `NaN`
    /// or `min > max`.
    #[expect(clippy::clone_on_copy)]
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

// --- fill -------------------------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: StorageMut<Elem = A>,
    D: Dimension,
    A: Element + Clone,
{
    /// Fill all logical elements with `value` in place.
    ///
    /// Stride-aware: iterates via `iter_mut()` so only logical elements
    /// are written, regardless of layout or internal padding.
    #[expect(clippy::clone_on_copy)]
    pub fn fill(&mut self, value: A) {
        for slot in self.iter_mut() {
            *slot = value.clone();
        }
    }
}

// --- to_contiguous ----------------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Clone,
    A: Element + Clone,
{
    /// Produce a canonical F-order owned tensor with the same logical data.
    ///
    /// Always returns a fresh owned tensor; the input borrow is never
    /// aliased into the result. If the input is already F-contiguous,
    /// delegates to `to_owned()`.
    ///
    /// # Panics
    ///
    /// Panics if the logical iteration length does not equal `product(shape)`.
    /// This indicates a contract violation in the iterator and should never
    /// occur for valid tensors.
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

// --- into_contiguous --------------------------------------------------------

impl<S, D, A> TensorBase<S, D>
where
    S: StorageIntoOwned<Elem = A> + StorageSemantics,
    D: Dimension + Clone,
    A: Element + Clone,
{
    /// Consume `self` and produce a canonical F-order owned tensor.
    ///
    /// Reuses backing storage when the input is already a canonical
    /// F-contiguous `Owned` tensor. Otherwise falls back to `into_owned()`.
    ///
    /// # Panics
    ///
    /// Panics if F-order strides cannot be derived from the shape. This
    /// cannot happen on the reuse path because the canonical predicate
    /// already verified `is_f_contiguous()`, which implies the shape's
    /// element count fits `usize`.
    pub fn into_contiguous(self) -> Tensor<A, D> {
        if is_canonical_f_contiguous_owned(&self) {
            let dim = self.raw_dim();
            let strides = Strides::f_contiguous(&dim)
                .expect("canonical predicate implies shape is valid");
            let owned = self.storage.into_owned_storage();
            let flags = compute_layout_flags::<A, D>(
                &dim,
                &strides,
                owned.as_ptr()
            );
            // SAFETY: is_canonical_f_contiguous_owned verified F-order,
            // owned, offset==0, storage_len==shape product. D: Clone
            // ensured raw_dim() snapshot precedes the move.
            unsafe {
                TensorBase::new_unchecked(
                    owned,
                    dim,
                    strides,
                    0,
                    flags,
                    false
                )
            }
        } else {
            self.into_owned()
        }
    }
}

// --- try_fill ---------------------------------------------------------------

impl<D, A> TensorBase<Owned<A>, D>
where
    D: Dimension,
    A: Element + Clone,
{
    /// Fallible fill. On `Owned` storage, writes `value` to all elements.
    ///
    /// # Errors
    ///
    /// Always returns `Ok(())`. The `Result` return type exists for API
    /// uniformity with the read-only variants.
    #[expect(clippy::clone_on_copy)]
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
    /// Fallible fill. On `ViewRepr` (read-only), always returns an error.
    ///
    /// Covers both plain read-only views and `SharedReadOnly` views
    /// (derived from `ViewMut` or zero-stride broadcast).
    ///
    /// # Errors
    ///
    /// Always returns [`XenonError::InvalidStorageMode`] with
    /// `storage_kind: View`.
    pub fn try_fill(&mut self, _value: A) -> Result<(), XenonError> {
        Err(fill_try_read_only_err(StorageKindTag::View))
    }
}

impl<'a, D, A> TensorBase<ViewMutRepr<'a, A>, D>
where
    D: Dimension,
    A: Element + Clone,
{
    /// Fallible fill. On `ViewMutRepr`, writes `value` to all elements.
    ///
    /// # Errors
    ///
    /// Always returns `Ok(())`. The `Result` return type exists for API
    /// uniformity with the read-only variants.
    #[expect(clippy::clone_on_copy)]
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
    /// Fallible fill. On `ArcRepr` (shared, read-only), always returns an
    /// error.
    ///
    /// # Errors
    ///
    /// Always returns [`XenonError::InvalidStorageMode`] with
    /// `storage_kind: Shared`.
    pub fn try_fill(&mut self, _value: A) -> Result<(), XenonError> {
        Err(fill_try_read_only_err(StorageKindTag::Shared))
    }
}

#[cfg(test)]
mod tests {
    use crate::error::XenonError;
    use crate::tensor::{StorageKind, Tensor1, Tensor2};

    // --- clip tests ---------------------------------------------------------

    /// Values outside [min, max] are clamped; values within pass through.
    #[test]
    fn test_clip_basic() {
        let tensor = Tensor1::from_shape_vec([5], vec![-1.0, 0.5, 1.0, 2.0, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0)
            .expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.0, 0.5, 1.0, 2.0, 2.0]);
    }

    /// All values within bounds: output equals input.
    #[test]
    fn test_clip_no_change() {
        let tensor = Tensor1::from_shape_vec([3], vec![0.5, 1.0, 1.5])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 2.0)
            .expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0.5, 1.0, 1.5]);
    }

    /// NaN input values pass through unchanged.
    #[test]
    fn test_clip_nan() {
        let tensor = Tensor1::from_shape_vec([3], vec![1.0_f64, f64::NAN, 3.0])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0.0, 4.0)
            .expect("valid clip bounds");
        let values: Vec<f64> = clipped.iter().copied().collect();
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
    }

    /// NaN as min or max bound returns [`XenonError::InvalidArgument`].
    #[test]
    fn test_clip_nan_bound() {
        let tensor = Tensor1::from_shape_vec([1], vec![1.0_f64])
            .expect("from_shape_vec matching shape");
        assert!(matches!(
            tensor.clip(f64::NAN, 2.0),
            Err(XenonError::InvalidArgument { .. })
        ));
        assert!(matches!(
            tensor.clip(0.0, f64::NAN),
            Err(XenonError::InvalidArgument { .. })
        ));
    }

    /// clip works correctly on integer element types.
    #[test]
    fn test_clip_integers() {
        let tensor = Tensor1::from_shape_vec([4], vec![-5_i32, 0, 5, 10])
            .expect("from_shape_vec matching shape");
        let clipped = tensor.clip(0, 7)
            .expect("valid clip bounds");
        let values: Vec<i32> = clipped.iter().copied().collect();
        assert_eq!(values, vec![0, 0, 5, 7]);
    }

    /// clip works correctly on a transposed (non-contiguous) tensor.
    #[test]
    fn test_clip_non_contiguous() {
        let tensor = Tensor2::from_shape_vec(
            [2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]
        ).expect("from_shape_vec matching shape");
        let transposed = tensor.transpose();
        let clipped = transposed.clip(2.0, 5.0)
            .expect("valid clip bounds");
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

    // --- fill tests ---------------------------------------------------------

    /// fill writes the given value to all elements.
    #[test]
    fn test_fill_basic() {
        let mut tensor = Tensor1::<f64>::zeros([3])
            .expect("zeros(valid shape)");
        tensor.fill(2.5);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            vec![2.5, 2.5, 2.5]
        );
    }

    /// try_fill on a read-only view returns [`XenonError::InvalidStorageMode`].
    #[test]
    fn test_try_fill_read_only_returns_error() {
        let tensor = Tensor1::from_shape_vec([2], vec![1_i32, 2])
            .expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    /// try_fill on read-only storage returns InvalidStorageMode.
    #[test]
    fn test_try_fill_read_only_returns_read_only_storage() {
        let tensor = Tensor1::from_shape_vec([2], vec![1_i32, 2])
            .expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    /// fill works on non-contiguous writable layouts (pending constructor).
    #[test]
    #[ignore = "needs writable non-contiguous view primitive"]
    fn test_fill_non_contiguous() {
        todo!("activate after writable non-contiguous view constructor lands");
    }

    /// fill does not touch padding elements (pending constructor).
    #[test]
    #[ignore = "needs writable strided sub-view primitive"]
    fn test_fill_padded_writes_logical_only() {
        todo!("activate after writable strided slice constructor lands");
    }

    /// try_fill on writable storage produces the same result as fill.
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

    /// fill on an empty tensor is a no-op.
    #[test]
    fn test_fill_empty() {
        let mut tensor = Tensor1::<f64>::zeros([0]).expect("zeros(empty)");
        tensor.fill(1.0);
        assert_eq!(tensor.len(), 0);
    }

    /// After fill(v), every element equals v.
    #[test]
    fn test_fill_invariant_all_equal_value() {
        for &n in &[1_usize, 2, 3, 5, 8, 13] {
            let mut t = Tensor1::<i64>::zeros([n]).expect("zeros(valid n)");
            t.fill(42);
            assert!(t.iter().all(|&x| x == 42), "n={}", n);
        }
    }

    // --- contiguous tests ---------------------------------------------------

    /// to_contiguous on an F-contiguous input preserves element values.
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

    /// to_contiguous on a transposed input produces F-order output.
    #[test]
    fn test_to_contiguous_transposed_becomes_f() {
        let tensor = Tensor2::<i32>::from_shape_vec(
            [2, 3],
            vec![1, 2, 3, 4, 5, 6]
        ).expect("from_shape_vec matching shape");
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

    /// into_contiguous preserves element data for canonical owned input.
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

    /// into_contiguous repacks when the input has tail padding
    /// (storage length exceeds shape product).
    #[test]
    fn test_into_contiguous_repacks_noncanonical_f_contiguous_owned() {
        use crate::dimension::Ix1;
        use crate::layout::{Strides, compute_layout_flags};
        use crate::storage::{Owned, RawStorage};
        use crate::tensor::TensorBase;

        // Owned storage with 5 elements, but shape [4] has product 4.
        let owned = Owned::from_vec(vec![1_i32, 2, 3, 4, 99])
            .expect("from_vec");
        let shape = Ix1(4);
        let strides = Strides::f_contiguous(&shape).expect("f_contiguous strides");
        let flags = compute_layout_flags::<i32, Ix1>(&shape, &strides, owned.as_ptr());
        // SAFETY: strides are F-order for shape [4]; owned storage holds 5
        // elements so the logical access range [0..4] is in bounds.
        let padded = unsafe {
            TensorBase::new_unchecked(owned, shape, strides, 0, flags, false)
        };
        assert!(padded.is_f_contiguous());
        assert_eq!(padded.storage_kind(), StorageKind::Owned);
        // Tail padding: storage_len (5) != product(shape) (4)
        assert_ne!(padded.storage_len(), padded.len());

        let canonical = padded.into_contiguous();
        assert!(canonical.is_f_contiguous());
        assert_eq!(canonical.storage_kind(), StorageKind::Owned);
        assert_eq!(canonical.storage_len(), canonical.len());
        assert_eq!(
            canonical.iter().copied().collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
    }

    /// into_contiguous repacks shared (Arc) input into owned F-order.
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

    /// to_contiguous on a non-contiguous tensor produces canonical F-order.
    #[test]
    fn test_to_contiguous_non_contiguous() {
        let tensor = Tensor2::<i32>::from_shape_vec(
            [2, 3],
            vec![1, 2, 3, 4, 5, 6]
        ).expect("from_shape_vec matching shape");
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
