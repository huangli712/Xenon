//! Re-layout operations: `to_contiguous` / `into_contiguous`.
//!
//! All operations are exposed as inherent methods on [`TensorBase`].

use crate::dimension::Dimension;
use crate::element::Element;
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{RawStorage, Storage, StorageIntoOwned};
use crate::tensor::{StorageKind, StorageSemantics, Tensor, TensorBase};

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
            let strides =
                Strides::f_contiguous(&dim).expect("canonical predicate implies shape is valid");
            let owned = self.storage.into_owned_storage();
            let flags = compute_layout_flags::<A, D>(&dim, &strides, owned.as_ptr());
            // SAFETY: is_canonical_f_contiguous_owned verified F-order,
            // owned, offset==0, storage_len==shape product. D: Clone
            // ensured raw_dim() snapshot precedes the move.
            unsafe { TensorBase::new_unchecked(owned, dim, strides, 0, flags, false) }
        } else {
            // Repack: into_owned() cannot be used because for Owned
            // storage it is O(1) and preserves tail padding / non-zero
            // offset. Always iterate logical F-order instead.
            let dim = self.raw_dim();
            let values: Vec<A> = self.iter().cloned().collect();
            Tensor::from_shape_vec(dim, values)
                .expect("logical iteration length equals shape product")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{StorageKind, Tensor2};

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

    /// to_contiguous on a non-contiguous tensor produces canonical F-order.
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

    /// into_contiguous repacks when the input has tail padding (storage
    /// length exceeds shape product).
    #[test]
    fn test_into_contiguous_repacks_noncanonical_f_contiguous_owned() {
        use crate::dimension::Ix1;
        use crate::layout::{Strides, compute_layout_flags};
        use crate::storage::{Owned, RawStorage};
        use crate::tensor::TensorBase;

        // Owned storage with 5 elements, but shape [4] has product 4.
        let owned = Owned::from_vec(vec![1_i32, 2, 3, 4, 99]).expect("from_vec");
        let shape = Ix1(4);
        let strides = Strides::f_contiguous(&shape).expect("f_contiguous strides");
        let flags = compute_layout_flags::<i32, Ix1>(&shape, &strides, owned.as_ptr());
        let padded = unsafe { TensorBase::new_unchecked(owned, shape, strides, 0, flags, false) };
        assert!(padded.is_f_contiguous());
        assert_eq!(padded.storage_kind(), StorageKind::Owned);
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
}
