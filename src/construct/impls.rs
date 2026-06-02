//! Tensor constructor implementations.
//!
//! Provides `TensorBase<Owned<A>, D>` construction methods: `zeros`, `ones`,
//! `eye`, `from_shape_vec`, `from_vec`, `from_shape_slice`, `from_array`,
//! and `from_scalar`. All constructors enforce shape validation and return
//! `Result<Self, XenonError>`.

use std::borrow::Cow;

use crate::error::{InvalidShapeKind, XenonError};
use crate::dimension::{Dimension, IntoDimension, Ix0, Ix1, Ix2};
use crate::storage::{Owned, RawStorage, StorageOwned};
use crate::layout::{Strides, compute_layout_flags};
use crate::element::Element;
use crate::tensor::TensorBase;
use super::EyeElement;

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
        let strides = Strides::f_contiguous(&shape)?;
        let flags = compute_layout_flags(&shape, &strides, storage.as_ptr());
        // SAFETY: 0-D scalar; `shape = Ix0` (product = 1); `strides = []`;
        // `flags` from `compute_layout_flags`; storage length = 1; offset 0;
        // logical access trivially within storage.
        // `derived_from_view_mut: false` — `from_scalar` is not a downgrade path.
        Ok(unsafe {
            TensorBase::new_unchecked(
                storage,
                shape,
                strides,
                0,
                flags,
                false
            )
        })
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

impl<A> TensorBase<Owned<A>, Ix2>
where
    A: EyeElement,
{
    /// Create an n×n identity matrix.
    ///
    /// Diagonal elements are 1, all others are 0. F-order layout.
    ///
    /// # Errors
    ///
    /// Propagates from `Self::zeros([n, n])`:
    /// - `XenonError::InvalidShape { kind: ProductOverflow }` — `n * n`
    ///   (or its byte size) overflows `usize` / `isize::MAX`.
    /// - `XenonError::AllocationFailed` — the underlying allocator could not
    ///   provide the requested zero-filled aligned buffer.
    pub fn eye(n: usize) -> Result<Self, XenonError> {
        let mut result = Self::zeros([n, n])?;
        for i in 0..n {
            // SAFETY: `i < n`, so `[i, i]` is always in-bounds for the validated
            // `[n, n]` shape created above. `eye()` uses unchecked indexing
            // internally and does not rely on the public `IndexMut` panic sugar.
            unsafe {
                *result.get_unchecked_mut(&[i, i]) = A::one();
            }
        }
        Ok(result)
    }
}

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
        let strides = Strides::f_contiguous(&dim)?;
        let storage = <Owned<A> as StorageOwned>::from_elem(len, A::zero());
        let flags = compute_layout_flags(&dim, &strides, storage.as_ptr());
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
        let strides = Strides::f_contiguous(&dim)?;
        let storage = <Owned<A> as StorageOwned>::from_elem(len, A::one());
        let flags = compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: shape validated by checked_size; strides from compute_f_strides;
        // flags from compute_layout_flags; storage length = len; offset 0;
        // derived_from_view_mut: false — ones is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }

    /// Construct a tensor from `shape` and `data`, consuming the Vec.
    ///
    /// # Errors
    ///
    /// Returns `XenonError::InvalidShape`:
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
        let strides = Strides::f_contiguous(&dim)?;
        let storage = Owned::from_vec_aligned(data)?;
        let flags = compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: `dim` validated by `IntoDimension` + `checked_size`; `data.len()
        // == expected` already enforced via ElementCountMismatch check; `strides`
        // from `compute_f_strides(&dim)?`; `flags` from `compute_layout_flags`;
        // storage length = expected; offset 0.
        // `derived_from_view_mut: false` — `from_shape_vec` is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }

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
        // The `into_iter().collect()` hop introduces one temporary `Vec<A>`
        // hop introduces one temporary `Vec<A>` of length N. Combined with the
        // potential repack inside `Owned::from_vec_aligned`, `from_array` is
        // documented as "necessary data movement" rather than "avoidable
        // temporary allocation". Do NOT switch to `Vec::from(arr)` — the
        // design uses `into_iter().collect()` as its canonical form.
        Self::from_shape_vec(shape, arr.into_iter().collect())
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
    /// Returns `XenonError::InvalidShape`:
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

#[cfg(test)]
mod tests {
    use crate::dimension::{Ix0, Ix1, Ix2};
    use crate::error::{InvalidShapeKind, XenonError};
    use crate::tensor::Tensor;

    /// `zeros` produces a tensor with the expected shape.
    #[test]
    fn test_zeros_shape() {
        let tensor = Tensor::<f64, _>::zeros([3, 4]).expect("zeros [3,4]");
        assert_eq!(tensor.shape(), &[3, 4]);
    }

    /// All elements of a `zeros` tensor evaluate to zero.
    #[test]
    fn test_zeros_values() {
        let tensor = Tensor::<i32, _>::zeros([2, 3]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.iter().all(|value| *value == 0));
    }

    /// Zero-length axis produces a valid empty tensor.
    #[test]
    fn test_zeros_empty() {
        let tensor = Tensor::<f64, _>::zeros([0, 5]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[0, 5]);
        assert_eq!(tensor.len(), 0);
    }

    /// All elements of a `ones` tensor evaluate to the multiplicative identity.
    #[test]
    fn test_ones_values() {
        let tensor = Tensor::<bool, _>::ones([2, 2]).expect("test input must be valid");
        assert_eq!(tensor.len(), 4);
        assert!(tensor.iter().all(|value| *value));
    }

    /// Zero-dimensional tensor: shape = [], product = 1, len() == 1,
    /// distinct from `ones([0])` which yields an empty tensor.
    #[test]
    fn test_ones_zero_dim() {
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

    /// Zero-length axis produces a valid empty tensor with len() == 0.
    #[test]
    fn test_ones_empty() {
        let tensor = Tensor::<i32, _>::ones([0, 5]).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[0, 5]);
        assert_eq!(tensor.len(), 0);
    }

    /// A 3×3 identity matrix has ones on the diagonal and zeros elsewhere.
    #[test]
    fn test_eye_3x3() {
        let tensor = Tensor::<i32, Ix2>::eye(3).expect("test input must be valid");
        assert_eq!(*tensor.get(&[0, 0]).expect("test input must be valid"), 1);
        assert_eq!(*tensor.get(&[1, 0]).expect("test input must be valid"), 0);
        assert_eq!(*tensor.get(&[2, 2]).expect("test input must be valid"), 1);
    }

    /// Empty identity matrix: eye(0) produces a 0×0 tensor with len() == 0.
    #[test]
    fn test_eye_zero() {
        let tensor = Tensor::<f64, Ix2>::eye(0).expect("test input must be valid");
        assert_eq!(tensor.shape(), &[0, 0]);
        assert_eq!(tensor.len(), 0);
    }

    /// n×n identity matrix overflows `checked_size` when `n` approaches `usize::MAX`.
    #[test]
    fn test_eye_overflow() {
        let err = Tensor::<i32, Ix2>::eye(usize::MAX).expect_err("usize::MAX overflows");
        assert!(matches!(
            err,
            XenonError::InvalidShape {
                kind: InvalidShapeKind::ProductOverflow,
                ..
            }
        ));
    }

    /// `from_shape_vec` constructs a tensor with the correct shape and element count.
    #[test]
    fn test_from_shape_vec_success() {
        let tensor =
            Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4]).expect("test input valid");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
    }

    /// Mismatched shape-element count returns `ElementCountMismatch` error.
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

    /// Verify that `InvalidShape` carries the correct `operation` field
    /// identifying the constructor (`"from_shape_vec"`).
    #[test]
    fn test_from_shape_vec_mismatch_operation_field() {
        let err =
            Tensor::<i32, _>::from_shape_vec([2, 2], vec![1, 2, 3]).expect_err("mismatched shape");
        if let XenonError::InvalidShape { operation, .. } = err {
            assert_eq!(operation.as_ref(), "from_shape_vec");
        } else {
            panic!("expected XenonError::InvalidShape");
        }
    }

    /// `from_vec` infers a 1-D shape `[data.len()]` from the input vec.
    #[test]
    fn test_from_vec_success() {
        let tensor = Tensor::<i32, Ix1>::from_vec(vec![1i32, 2, 3]).expect("test input valid");
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.len(), 3);
    }

    /// `from_shape_slice` copies data into an owned tensor and preserves the source.
    #[test]
    fn test_from_shape_slice_success() {
        let source = [1.0f64, 2.0, 3.0, 4.0];
        let tensor = Tensor::<f64, _>::from_shape_slice([2, 2], &source).expect("test input valid");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(*tensor.get(&[0, 0]).expect("test input valid"), 1.0);
        // Source is not consumed — original array is untouched.
        assert_eq!(source[0], 1.0);
    }

    /// Slice-element count mismatch returns `ElementCountMismatch` error.
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

    /// Verify `InvalidShape` carries `operation: "from_shape_slice"`,
    /// not the internal `"from_shape_vec"` delegation target.
    #[test]
    fn test_from_shape_slice_operation_field() {
        let source = [1, 2, 3];
        let err =
            Tensor::<i32, _>::from_shape_slice([2, 2], &source).expect_err("mismatched shape");
        if let XenonError::InvalidShape { operation, .. } = err {
            assert_eq!(operation.as_ref(), "from_shape_slice");
        } else {
            panic!("expected XenonError::InvalidShape");
        }
    }

    /// `from_array` constructs a tensor from a fixed-size array.
    #[test]
    fn test_from_array_success() {
        let tensor =
            Tensor::<i32, _>::from_array([2, 2], [1i32, 2, 3, 4]).expect("test input valid");
        assert_eq!(tensor.len(), 4);
        assert_eq!(*tensor.get(&[0, 0]).expect("test input valid"), 1);
    }

    /// Array length mismatch returns `ElementCountMismatch` error.
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

    /// `from_scalar` produces a 0-D tensor of length 1 containing the value.
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

    /// `from_scalar` accepts `bool` (bound is `A: Element`), unlike `eye`
    /// which requires the stricter `EyeElement` trait.
    #[test]
    fn test_from_scalar_bool() {
        let tensor = Tensor::<bool, Ix0>::from_scalar(true).expect("test input must be valid");
        assert_eq!(tensor.len(), 1);
        assert!(
            *tensor
                .get(&[] as &[usize])
                .expect("test input must be valid")
        );
    }
}
