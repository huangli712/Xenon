use std::borrow::Cow;

use crate::dimension::{Dimension, IntoDimension, Ix1};
use crate::element::Element;
use crate::error::{InvalidShapeKind, XenonError};
use crate::layout;
use crate::storage::Owned;
use crate::storage::RawStorage;
use crate::tensor::TensorBase;

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
        let strides = layout::compute_f_strides(&dim)?;
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

#[cfg(test)]
mod tests {
    use crate::dimension::Ix1;
    use crate::error::{InvalidShapeKind, XenonError};
    use crate::tensor::Tensor;

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
}
