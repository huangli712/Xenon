use crate::dimension::{Dimension, IntoDimension};
use crate::element::Element;
use crate::error::XenonError;
use crate::layout;
use crate::storage::{Owned, RawStorage, StorageOwned};
use crate::tensor::TensorBase;

impl<A, D> TensorBase<Owned<A>, D>
where
    A: Element,
    D: Dimension,
{
    /// Create a zero-initialized tensor (F-order).
    pub fn zeros<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.checked_size()?;
        let strides = layout::compute_f_strides(&dim)?;
        let storage = <Owned<A> as StorageOwned>::from_elem(len, A::zero());
        let flags = layout::compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: shape validated by checked_size; strides from compute_f_strides;
        // flags from compute_layout_flags; storage length = len; offset 0;
        // derived_from_view_mut: false — zeros is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }

    /// Create a one-initialized tensor (F-order).
    pub fn ones<Sh>(shape: Sh) -> Result<Self, XenonError>
    where
        Sh: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        let len = dim.checked_size()?;
        let strides = layout::compute_f_strides(&dim)?;
        let storage = <Owned<A> as StorageOwned>::from_elem(len, A::one());
        let flags = layout::compute_layout_flags(&dim, &strides, storage.as_ptr());
        // SAFETY: shape validated by checked_size; strides from compute_f_strides;
        // flags from compute_layout_flags; storage length = len; offset 0;
        // derived_from_view_mut: false — ones is not a downgrade path.
        Ok(unsafe { Self::new_unchecked(storage, dim, strides, 0, flags, false) })
    }
}

#[cfg(test)]
mod tests {
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
}
