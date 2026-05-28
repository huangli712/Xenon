use crate::dimension::Ix0;
use crate::element::Element;
use crate::error::XenonError;
use crate::layout;
use crate::storage::Owned;
use crate::storage::RawStorage;
use crate::tensor::TensorBase;

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
    use crate::dimension::Ix0;
    use crate::tensor::Tensor;

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
