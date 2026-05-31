//! Full-axis transpose implementation.

use crate::dimension::{Dimension, Reverse};
use crate::layout::{Strides, compute_layout_flags};
use crate::storage::{Storage, ViewRepr};
use crate::tensor::{TensorBase, TensorView};

/// Reverse the dimension, strides, and layout flags to produce a transposed view.
fn transpose_impl<S, D, A>(tensor: &TensorBase<S, D>) -> TensorView<'_, A, D>
where
    S: Storage<Elem = A>,
    D: Dimension + Reverse,
{
    // Reverse the dimension via the sealed `Reverse` trait.
    let new_shape: D = tensor.raw_dim().reverse();

    // Build reversed strides.
    let rev: Vec<usize> = tensor.strides().iter().rev().copied().collect();
    let new_strides: Strides<D> = Strides::<D>::from_slice(&rev)
        .expect("rank-preserving stride reverse cannot change slice length");

    // Recompute layout flags.
    let new_flags = compute_layout_flags::<A, D>(
        &new_shape,
        &new_strides,
        tensor.as_ptr()
    );

    // Build a ViewRepr borrowing the source storage.
    // SAFETY: as_storage_ptr() is a non-null aligned base pointer of
    // already-validated live storage. storage_len() is the correct extent.
    // Result lifetime is bound to &tensor.
    let view_storage: ViewRepr<'_, A> =
        unsafe {
            ViewRepr::from_raw_parts(
                tensor.as_storage_ptr(),
                tensor.storage_len()
            )
        };

    // Assemble the result via TensorBase::new_unchecked.
    // SAFETY: new_shape + new_strides + offset form a bijective reversal
    // of source metadata. new_flags computed by the authoritative entry
    // point. derived_from_view_mut forwarded from source.
    unsafe {
        TensorBase::<ViewRepr<'_, A>, D>::new_unchecked(
            view_storage,
            new_shape,
            new_strides,
            tensor.offset(),
            new_flags,
            tensor.derived_from_view_mut,
        )
    }
}

impl<S, D, A> TensorBase<S, D>
where
    S: Storage<Elem = A>,
    D: Dimension,
{
    /// Reverse the axis order.
    ///
    /// `Reverse` is bound at the method-level `where`-clause (not the
    /// `impl` header) so it constrains only this API — other methods on
    /// the same `impl` block are unaffected.
    pub fn transpose(&self) -> TensorView<'_, A, D>
    where
        D: Reverse,
    {
        transpose_impl(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::dimension::{Dimension, Ix0, Ix1, Ix2, Ix3};
    use crate::element::Element;
    use crate::layout::LayoutState;
    use crate::storage::Owned;
    use crate::tensor::{StorageKind, TensorBase};

    /// Construct a tensor using the internal fast path.
    unsafe fn make_tensor<A: Element, D: Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<Owned<A>, D> {
        // SAFETY: caller provides data with correct length matching shape.
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    /// Access element at logical index.
    unsafe fn read_at<'a, S, D, A>(tensor: &'a TensorBase<S, D>, indices: &[usize]) -> &'a A
    where
        S: crate::storage::Storage<Elem = A>,
        D: Dimension,
        A: Element,
    {
        debug_assert_eq!(indices.len(), tensor.ndim());
        let strides = tensor.strides();
        let mut rel_offset: isize = 0;
        for (axis, &idx) in indices.iter().enumerate() {
            rel_offset += (idx as isize) * (strides[axis] as isize);
        }
        unsafe { &*tensor.as_ptr().offset(rel_offset) }
    }

    /// Transpose swaps row/col order and keeps element values correct.
    #[test]
    fn test_transpose_2d() {
        let x = unsafe { make_tensor(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3)) };
        let y = x.transpose();
        assert_eq!(y.shape(), &[3, 2]);
        unsafe {
            assert_eq!(*read_at(&y, &[0, 0]), *read_at(&x, &[0, 0]));
            assert_eq!(*read_at(&y, &[2, 1]), *read_at(&x, &[1, 2]));
        }
    }

    /// Transpose reverses a 3D shape to `(4, 3, 2)`.
    #[test]
    fn test_transpose_3d() {
        let x = unsafe { make_tensor(Vec::<i32>::new(), Ix3(2, 3, 4)) };
        assert_eq!(x.transpose().shape(), &[4, 3, 2]);
    }

    /// Transposing a 1D tensor is a no‑op — shape and length unchanged.
    #[test]
    fn test_transpose_1d_noop() {
        let x = unsafe { make_tensor(vec![1_i32, 2, 3], Ix1(3)) };
        let y = x.transpose();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.len(), x.len());
    }

    /// Transposing a 0D tensor is a no‑op — shape stays empty.
    #[test]
    fn test_transpose_0d_noop() {
        let x = unsafe { make_tensor(vec![5], Ix0) };
        let y = x.transpose();
        assert_eq!(y.shape(), &[]);
    }

    /// Transpose breaks F-contiguity for 2D+ tensors.
    #[test]
    fn test_transpose_not_f_contiguous() {
        let x = unsafe { make_tensor(Vec::<i32>::new(), Ix2(2, 3)) };
        assert!(x.is_f_contiguous());
        assert!(!x.transpose().is_f_contiguous());
    }

    /// Transposing 0D or 1D tensors preserves F‑contiguity.
    #[test]
    fn test_transpose_0d_1d_preserves_contiguity() {
        let s = unsafe { make_tensor(vec![42.0_f64], Ix0) };
        let st = s.transpose();
        assert_eq!(st.is_f_contiguous(), s.is_f_contiguous());
        let v = unsafe { make_tensor(vec![1_i32, 2, 3, 4], Ix1(4)) };
        let vt = v.transpose();
        assert_eq!(vt.is_f_contiguous(), v.is_f_contiguous());
    }

    /// Transpose of a broadcast view preserves the `BroadcastView` layout
    /// state — the zero stride swaps axes but remains zero.
    #[test]
    fn test_transpose_broadcast_view_keeps_flag() {
        let t = unsafe { make_tensor(vec![1.0_f64, 2.0, 3.0], Ix2(1, 3)) };
        let b = t.broadcast_to([2, 3]).expect("compatible shapes");
        assert_eq!(b.layout_state(), LayoutState::BroadcastView);
        // Transpose: broadcast axis moves from [0] to [1], stride stays zero.
        let bt = b.transpose();
        assert_eq!(bt.strides(), &[1, 0]);
        assert_eq!(bt.layout_state(), LayoutState::BroadcastView);
    }

    /// Transpose of a view_mut yields a View storage kind.
    #[test]
    fn test_transpose_view_mut_returns_view_kind() {
        let mut x = unsafe { make_tensor(Vec::<i32>::new(), Ix2(2, 3)) };
        let v = x.view_mut();
        assert_eq!(v.transpose().storage_kind(), StorageKind::View);
    }

    /// Transpose swaps shape even for empty arrays.
    #[test]
    fn test_transpose_empty_array() {
        let x = unsafe { make_tensor(Vec::<i32>::new(), Ix2(0, 3)) };
        let y = x.transpose();
        assert_eq!(y.shape(), &[3, 0]);
        assert_eq!(y.len(), 0);
    }

    /// Double-transpose is the identity — shape and strides are restored.
    #[test]
    fn test_transpose_high_dim() {
        let x = unsafe { make_tensor(Vec::<i32>::new(), Ix3(2, 3, 4)) };
        assert_eq!(x.transpose().transpose().shape(), x.shape());
        assert_eq!(x.transpose().transpose().strides(), x.strides());
    }
}
