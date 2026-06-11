//! In-place fill operations for tensors.
//!
//! Provides `TensorBase::fill()` and `TensorBase::try_fill()` for writing a
//! uniform value to every logical element.

use std::borrow::Cow;

use crate::dimension::Dimension;
use crate::element::Element;
use crate::error::{StorageKindTag, XenonError};
use crate::storage::StorageMut;
use crate::storage::{ArcRepr, Owned, ViewMutRepr, ViewRepr};
use crate::tensor::TensorBase;

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
    use crate::tensor::Tensor1;

    /// fill writes the given value to all elements.
    #[test]
    fn test_fill_basic() {
        let mut tensor = Tensor1::<f64>::zeros([3]).expect("zeros(valid shape)");
        tensor.fill(2.5);
        assert_eq!(
            tensor.iter().copied().collect::<Vec<_>>(),
            vec![2.5, 2.5, 2.5]
        );
    }

    /// try_fill on a read-only view returns [`XenonError::InvalidStorageMode`].
    #[test]
    fn test_try_fill_read_only_returns_error() {
        let tensor =
            Tensor1::from_shape_vec([2], vec![1_i32, 2]).expect("from_shape_vec matching shape");
        let mut view = tensor.view();
        let error = view.try_fill(7).expect_err("view is read-only");
        assert!(matches!(error, XenonError::InvalidStorageMode { .. }));
    }

    /// try_fill on read-only storage returns InvalidStorageMode.
    #[test]
    fn test_try_fill_read_only_returns_read_only_storage() {
        let tensor =
            Tensor1::from_shape_vec([2], vec![1_i32, 2]).expect("from_shape_vec matching shape");
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
}
