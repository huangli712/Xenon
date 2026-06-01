use std::borrow::Cow;

use crate::dimension::{Dimension, IntoDimension, Ix0, Ix1};
use crate::element::Element;
use crate::error::{InvalidShapeKind, XenonError};
use crate::layout;
use crate::storage::{Owned, RawStorage, StorageOwned};
use crate::tensor::TensorBase;

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Create a zero-initialized tensor (F-order).
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidShape { kind: ProductOverflow, .. }` if
    /// `shape`'s element count or stride product overflows `usize` (forwarded
    /// from `dim.checked_size()` / `compute_f_strides`).
    pub fn zeros<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.checked_size()?;
        let strides = layout::Strides::f_contiguous(&dim)?;
        let storage = <Owned<A> as StorageOwned>::from_elem(len, A::zero());
        let flags = layout::compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: shape validated by checked_size; strides from compute_f_strides;
        // flags from compute_layout_flags; storage length = len; offset 0;
        // derived_from_view_mut: false — zeros is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }

    /// Create a one-initialized tensor (F-order).
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidShape { kind: ProductOverflow, .. }` if
    /// `shape`'s element count or stride product overflows `usize` (forwarded
    /// from `dim.checked_size()` / `compute_f_strides`).
    pub fn ones<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.checked_size()?;
        let strides = layout::Strides::f_contiguous(&dim)?;
        let storage = <Owned<A> as StorageOwned>::from_elem(len, A::one());
        let flags = layout::compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: shape validated by checked_size; strides from compute_f_strides;
        // flags from compute_layout_flags; storage length = len; offset 0;
        // derived_from_view_mut: false — ones is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }
}

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Construct a tensor from `shape` and `data`, consuming the Vec.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidShape` aligned with `26-error.md §5.1`:
    /// - `kind: InvalidShapeKind::ProductOverflow` if `dim.checked_size()` overflows usize.
    /// - `kind: InvalidShapeKind::ElementCountMismatch { expected, actual }` if `data.len() != expected`.
    pub fn from_shape_vec<Sh>(shape: Sh, data: Vec<A>) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.checked_size()?;
        if data.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: Cow::Borrowed("from_shape_vec"),
                shape: dim.slice().to_vec(),
                kind: InvalidShapeKind::ElementCountMismatch {
                    expected,
                    actual: data.len(),
                },
                offending_dim: None,
            });
        }
        let strides = layout::Strides::f_contiguous(&dim)?;
        let storage = Owned::from_vec_aligned(data)?;
        let flags = layout::compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: `dim` validated by `IntoDimension` + `checked_size`; `data.len()
        // == expected` already enforced via ElementCountMismatch check; `strides`
        // from `compute_f_strides(&dim)?`; `flags` from `compute_layout_flags`;
        // storage length = expected; offset 0.
        // `derived_from_view_mut: false` — `from_shape_vec` is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }
}

impl<A> TensorBase<Owned<A>, Ix1>
where
    A: Element,
{
    /// Construct a 1-D tensor from `data`. Convenience wrapper around
    /// `from_shape_vec` with shape inferred as `[data.len()]`.
    ///
    /// # Errors
    ///
    /// Forwards `from_shape_vec` errors. Because the shape `[data.len()]` is
    /// derived from `data` itself, `ElementCountMismatch` is unreachable; in
    /// practice the only observable error is
    /// `XenonError::InvalidShape { kind: InvalidShapeKind::ProductOverflow, .. }`
    /// when `data.len()` overflows `usize` (impossible on 64-bit targets).
    pub fn from_vec(data: Vec<A>) -> Result<Self, XenonError> {
        Self::from_shape_vec([data.len()], data)
    }
}

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element + Clone,
    D: Dimension,
{
    /// Construct a tensor by copying data from a slice (F-order).
    /// The source slice is never borrowed — data is always copied.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidShape` aligned with `26-error.md §5.1`:
    /// - `kind: InvalidShapeKind::ProductOverflow` if `dim.checked_size()` overflows usize.
    /// - `kind: InvalidShapeKind::ElementCountMismatch { expected, actual }` if `slice.len() != expected`.
    ///
    /// The `operation` field is set to `"from_shape_slice"` so downstream
    /// diagnostics reflect the actual entry point rather than the internal
    /// `from_shape_vec` delegation target.
    pub fn from_shape_slice<Sh>(shape: Sh, slice: &[A]) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let expected = dim.checked_size()?;
        if slice.len() != expected {
            return Err(XenonError::InvalidShape {
                operation: Cow::Borrowed("from_shape_slice"),
                shape: dim.slice().to_vec(),
                kind: InvalidShapeKind::ElementCountMismatch {
                    expected,
                    actual: slice.len(),
                },
                offending_dim: None,
            });
        }
        // Length already validated; pass the already-normalized `dim` to avoid
        // a second `IntoDimension` traversal inside `from_shape_vec`.
        Self::from_shape_vec(dim, slice.to_vec())
    }
}

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Construct a tensor from a fixed-size array.
    ///
    /// The const generic `N` provides compile-time length for the input array;
    /// runtime shape validation (`shape.checked_size() == N`) is still performed
    /// inside `from_shape_vec`.
    ///
    /// # Errors
    ///
    /// Propagates from [`Self::from_shape_vec`]:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` if
    ///   `shape.checked_size()` overflows `usize`.
    /// - `XenonError::InvalidShape { kind: ElementCountMismatch { expected, actual } }`
    ///   if `shape.checked_size() != N`.
    /// - `XenonError::AllocationFailed` if the underlying aligned allocator
    ///   cannot satisfy the buffer request.
    pub fn from_array<Sh, const N: usize>(shape: Sh, arr: [A; N]) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        // Per 18-construction.md §5.3 line 432-452: the `into_iter().collect()`
        // hop introduces one temporary `Vec<A>` of length N. Combined with the
        // potential repack inside `Owned::from_vec_aligned`, `from_array` is
        // documented as "necessary data movement" rather than "avoidable
        // temporary allocation". Do NOT switch to `Vec::from(arr)` — the
        // design uses `into_iter().collect()` as its canonical form.
        Self::from_shape_vec(shape, arr.into_iter().collect())
    }
}

impl<A> TensorBase<Owned<A>, Ix0>
where
    A: Element,
{
    /// Construct a zero-dimensional tensor from a scalar.
    ///
    /// # Errors
    ///
    /// In practice this single-element path does not produce
    /// `InvalidShapeKind::ElementCountMismatch` (the `vec![scalar]` length is
    /// always 1, matching `Ix0::checked_size() == 1`). The `Result` return
    /// type is preserved for signature uniformity with the rest of the
    /// construction family and to leave room for future allocator-side
    /// failure paths surfaced by `Owned::from_vec_aligned`.
    pub fn from_scalar(scalar: A) -> Result<Self, XenonError> {
        let storage = Owned::from_vec_aligned(vec![scalar])?;
        let shape = Ix0;
        let strides = layout::Strides::f_contiguous(&shape)?;
        let flags = layout::compute_layout_flags(&shape, &strides, storage.as_ptr());
        // SAFETY: 0-D scalar; `shape = Ix0` (product = 1); `strides = []`;
        // `flags` from `compute_layout_flags`; storage length = 1; offset 0;
        // logical access trivially within storage.
        // `derived_from_view_mut: false` — `from_scalar` is not a downgrade path.
        Ok(unsafe { TensorBase::new_unchecked(storage, shape, strides, 0, flags, false) })
    }
}

#[cfg(test)]
mod tests {
    use crate::dimension::{Ix0, Ix1};
    use crate::error::{InvalidShapeKind, XenonError};
    use crate::tensor::Tensor;

    #[test]
    fn test_zeros_shape() {
        let tensor = Tensor::<f64, _>::zeros([3, 4]).expect("zeros [3,4]");
        assert_eq!(tensor.shape(), &[3, 4]);
    }

    #[test]
    fn test_zeros_values() {
        let tensor = Tensor::<i32, _>::zeros([2, 3]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.iter().all(|value| *value == 0));
    }

    #[test]
    fn test_zeros_empty() {
        // Zero-length axis produces valid empty tensor
        let tensor = Tensor::<f64, _>::zeros([0, 5]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[0, 5]);
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_ones_values() {
        let tensor = Tensor::<bool, _>::ones([2, 2]).expect("test input must be valid");
        assert_eq!(tensor.len(), 4);
        assert!(tensor.iter().all(|value| *value));
    }

    #[test]
    fn test_ones_zero_dim() {
        // Zero-dimensional tensor: shape = [], product = 1, len() == 1.
        // (Per 18-construction §8.3: `from_scalar(42)` / `ones([])` both yield
        // ndim=0 tensors of length 1; this is distinct from `ones([0])`.)
        let tensor = Tensor::<i32, _>::ones([]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[]);
        assert_eq!(tensor.ndim(), 0);
        assert_eq!(tensor.len(), 1);
        assert_eq!(
            *tensor
                .get(&[] as &[usize])
                .expect("test input must be valid"),
            1
        );
    }

    #[test]
    fn test_ones_empty() {
        // Zero-length axis produces a valid empty tensor with len() == 0.
        // This is the canonical "empty tensor" case from 18-construction §8.3.
        let tensor = Tensor::<i32, _>::ones([0, 5]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[0, 5]);
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_from_shape_vec_success() {
        let tensor =
            Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("test input valid");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn test_from_shape_vec_mismatch() {
        let err =
            Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3]).expect_err("mismatched shape");
        assert!(matches!(
            err,
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ElementCountMismatch {
                    expected: 4,
                    actual: 3
                },
                ..
            }
        ));
    }

    #[test]
    fn test_from_shape_vec_mismatch_operation_field() {
        // Verify `operation` field carries "from_shape_vec" — exercises the
        // structured-diagnostic contract from 26-error.md §5.1.
        let err =
            Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3]).expect_err("mismatched shape");
        if let XenonError::InvalidShape { operation, .. } = err {
            assert_eq!(operation.as_ref(), "from_shape_vec");
        } else {
            panic!("expected XenonError::InvalidShape");
        }
    }

    #[test]
    fn test_from_vec_success() {
        let tensor = Tensor::<i32, Ix1>::from_vec(vec![1i32, 2, 3]).expect("test input valid");
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.len(), 3);
    }

    #[test]
    fn test_from_shape_slice_success() {
        let source = [1.0f64, 2.0, 3.0, 4.0];
        let tensor = Tensor::<f64, _>::from_shape_slice([2, 2], &source).expect("test input valid");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(*tensor.get(&[0, 0]).expect("test input valid"), 1.0);
        // Source is not consumed — original array is untouched
        assert_eq!(source[0], 1.0);
    }

    #[test]
    fn test_from_shape_slice_mismatch() {
        let source = [1, 2, 3];
        let err =
            Tensor::<i32, _>::from_shape_slice([2, 2], &source).expect_err("mismatched shape");
        assert!(matches!(
            err,
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ElementCountMismatch {
                    expected: 4,
                    actual: 3
                },
                ..
            }
        ));
    }

    #[test]
    fn test_from_shape_slice_operation_field() {
        // `operation` must read "from_shape_slice", not "from_shape_vec",
        // even though the implementation eventually delegates.
        let source = [1, 2, 3];
        let err =
            Tensor::<i32, _>::from_shape_slice([2, 2], &source).expect_err("mismatched shape");
        if let XenonError::InvalidShape { operation, .. } = err {
            assert_eq!(operation.as_ref(), "from_shape_slice");
        } else {
            panic!("expected XenonError::InvalidShape");
        }
    }

    #[test]
    fn test_from_array_success() {
        let tensor =
            Tensor::<i32, _>::from_array([2, 2], [1i32, 2, 3, 4]).expect("test input valid");
        assert_eq!(tensor.len(), 4);
        assert_eq!(*tensor.get(&[0, 0]).expect("test input valid"), 1);
    }

    #[test]
    fn test_from_array_mismatch() {
        let err = Tensor::<i32, _>::from_array([3, 3], [1i32; 4]).expect_err("mismatched shape");
        assert!(matches!(
            err,
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ElementCountMismatch {
                    expected: 9,
                    actual: 4
                },
                ..
            }
        ));
    }

    #[test]
    fn test_from_scalar() {
        let tensor = Tensor::<i32, Ix0>::from_scalar(42i32).expect("test input must be valid");
        assert_eq!(tensor.ndim(), 0);
        assert_eq!(tensor.len(), 1);
        assert_eq!(
            *tensor
                .get(&[] as &[usize])
                .expect("test input must be valid"),
            42
        );
    }

    #[test]
    fn test_from_scalar_bool() {
        // from_scalar allows `Element`-bound types including `bool` (more
        // permissive than `eye`'s `EyeElement` constraint).
        let tensor = Tensor::<bool, Ix0>::from_scalar(true).expect("test input must be valid");
        assert_eq!(tensor.len(), 1);
        assert!(
            *tensor
                .get(&[] as &[usize])
                .expect("test input must be valid")
        );
    }
}
