//! Integration tests for dimension types in shape operation paths.
//!
//! Per `02-dimension.md` §8.5 line 1122, this file covers `Axis` /
//! `BroadcastDim` in reshape / transpose / broadcast paths. The high-level
//! shape operations are introduced in W11 (broadcast) and W16 (reshape /
//! transpose); tests here are split into:
//!
//! - **W3-runnable**: pure dimension-layer shape contracts (rank, slice,
//!   checked_size, Axis access, checked() validation, equality).
//! - **W3T22 active**: `test_broadcast_dim_compatibility` exercises the
//!   `BroadcastDim` trait (activated by W3T22).
//! - **W11 active**: `test_broadcast_runtime_with_dim` exercises `broadcast_to`
//!   plus transpose of the broadcast view.
//! - **W20 integration**: transpose shape/data/view-kind/involution tests
//!   using `from_raw_vec_unchecked` (made `pub` in W20T4).

use xenon::dimension::{Axis, BroadcastDim, Dimension, Ix0, Ix1, Ix2, Ix3, Ix6, IxDyn};
use xenon::element::Element;
use xenon::error::XenonError;
use xenon::storage::Owned;
use xenon::tensor::{StorageKind, TensorBase};

/// Internal helper: construct tensor via fast path.
unsafe fn make_tensor<A: Element, D: Dimension>(data: Vec<A>, shape: D) -> TensorBase<Owned<A>, D> {
    // SAFETY: caller provides data with correct length matching shape.
    unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
}

/// Access element at logical index.
unsafe fn read_at<'a, S, D, A>(tensor: &'a TensorBase<S, D>, indices: &[usize]) -> &'a A
where
    S: xenon::storage::Storage<Elem = A>,
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

/// shape contract: rank/slice/checked_size for static dimensions.
#[test]
fn test_static_dimension_shape_contract() {
    let d = Ix3(2, 3, 4);
    assert_eq!(d.ndim(), 3);
    assert_eq!(d.slice(), &[2, 3, 4]);
    assert_eq!(d.checked_size(), Ok(24));
}

/// shape contract: rank/slice/checked_size for IxDyn.
#[test]
fn test_dynamic_dimension_shape_contract() {
    let d = IxDyn::from_vec(vec![2, 3, 4, 5]);
    assert_eq!(d.ndim(), 4);
    assert_eq!(d.checked_size(), Ok(120));
}

/// `Dimension::checked()` validates shape metadata without consuming size.
#[test]
fn test_dimension_checked_validates_shape() {
    assert_eq!(Ix3(2, 3, 4).checked(), Ok(()));
    assert_eq!(IxDyn::from_slice(&[1, 2, 3]).checked(), Ok(()));
    // Overflow case: checked() returns Err same kind as checked_size.
    assert!(Ix2(usize::MAX, 2).checked().is_err());
}

/// Shape equality: two dimensions of same type and same axis lengths are
/// equal.
#[test]
fn test_shape_equality() {
    assert_eq!(Ix3(2, 3, 4), Ix3(2, 3, 4));
    assert_ne!(Ix3(2, 3, 4), Ix3(2, 3, 5));
    assert_eq!(IxDyn::from_slice(&[1, 2, 3]), IxDyn::from_slice(&[1, 2, 3]));
    assert_ne!(IxDyn::from_slice(&[1, 2, 3]), IxDyn::from_slice(&[1, 2]));
}

/// Axis access on shape: legal axis returns Ok(length), out-of-range returns
/// InvalidAxis.
#[test]
fn test_shape_axis_access() {
    let d = Ix3(2, 3, 4);
    assert_eq!(d.axis(Axis::new(0)), Ok(2));
    assert_eq!(d.axis(Axis::new(1)), Ok(3));
    assert_eq!(d.axis(Axis::new(2)), Ok(4));
    assert!(matches!(
        d.axis(Axis::new(3)),
        Err(XenonError::InvalidAxis { .. })
    ));
}

/// Single-element shape and zero-axis shape boundaries.
#[test]
fn test_shape_boundary_cases() {
    assert_eq!(Ix0.checked_size(), Ok(1));
    assert_eq!(Ix2(1, 1).checked_size(), Ok(1));
    assert_eq!(Ix2(0, 5).checked_size(), Ok(0));
}

// ── W3T22-gated: BroadcastDim type-level test ──

/// W3T22 type-level test: `BroadcastDim` output inference.
///
/// Activated after W3T22 lands: BroadcastDim trait + 64 impls now available.
#[test]
fn test_broadcast_dim_compatibility() {
    // Same-rank: Ix2 BroadcastDim Ix2 → Ix2.
    fn _check_same<D: BroadcastDim<D, Output = D>>() {}
    _check_same::<Ix2>();
    // Cross-rank: Ix1 BroadcastDim Ix3 → Ix3 (higher rank wins).
    fn _check_cross<A, B, O>()
    where
        A: BroadcastDim<B, Output = O>,
        B: Dimension,
        O: Dimension,
    {
    }
    _check_cross::<Ix1, Ix3, Ix3>();
    // Mixed with IxDyn: any side IxDyn ⇒ IxDyn.
    _check_cross::<Ix2, IxDyn, IxDyn>();
    _check_cross::<IxDyn, Ix2, IxDyn>();
}

// ── W11 activation placeholder ──

/// W11 activated: broadcast path (`broadcast_to` and shape compatibility).
#[test]
fn test_broadcast_runtime_with_dim() {
    let t = xenon::tensor::Tensor2::<f64>::zeros([1, 4]).expect("valid shape");
    let v = t.broadcast_to([3, 4]).expect("compatible shapes");
    assert_eq!(v.shape(), &[3, 4]);
    // Transpose the broadcast view: shape swaps, broadcast stride moves.
    let vt = v.transpose();
    assert_eq!(vt.shape(), &[4, 3]);
    assert_eq!(vt.strides()[1], 0); // broadcast axis 0 → trailing after transpose.
}

// ── W20 transpose integration tests ──

#[test]
fn test_shape_integration_transpose_2d() {
    let x = unsafe { make_tensor(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3)) };
    let y = x.transpose();
    assert_eq!(y.shape(), &[3, 2]);
    unsafe {
        assert_eq!(*read_at(&y, &[0, 0]), *read_at(&x, &[0, 0]));
        assert_eq!(*read_at(&y, &[2, 1]), *read_at(&x, &[1, 2]));
    }
}

#[test]
fn test_shape_integration_transpose_3d() {
    let x = unsafe { make_tensor(Vec::<i32>::new(), Ix3(2, 3, 4)) };
    assert_eq!(x.transpose().shape(), &[4, 3, 2]);
}

#[test]
fn test_shape_integration_transpose_1d_noop() {
    let x = unsafe { make_tensor(vec![1_i32, 2, 3], Ix1(3)) };
    assert_eq!(x.transpose().shape(), &[3]);
}

#[test]
fn test_shape_integration_transpose_0d_noop() {
    let x = unsafe { make_tensor(vec![5], Ix0) };
    assert_eq!(x.transpose().shape(), &[]);
}

#[test]
fn test_shape_integration_transpose_not_f_contiguous() {
    let x = unsafe { make_tensor(Vec::<i32>::new(), Ix2(2, 3)) };
    assert!(!x.transpose().is_f_contiguous());
}

#[test]
fn test_shape_integration_transpose_view_kind() {
    let x = unsafe { make_tensor(Vec::<i32>::new(), Ix2(2, 3)) };
    assert_eq!(x.transpose().storage_kind(), StorageKind::View);
}

#[test]
#[ignore = "depends on ArcRepr constructor visibility (W18)"]
fn test_shape_integration_transpose_arc_tensor_view_kind() {}

#[test]
fn test_shape_integration_transpose_with_index() {
    let x = unsafe { make_tensor(vec![1, 2, 3, 4, 5, 6], Ix2(2, 3)) };
    let y = x.transpose();
    unsafe {
        for i in 0..2_usize {
            for j in 0..3_usize {
                assert_eq!(*read_at(&y, &[j, i]), *read_at(&x, &[i, j]));
            }
        }
    }
}

/// W11 activated: transpose of a broadcast view preserves the broadcast
/// layout with axes swapped.
#[test]
fn test_shape_integration_transpose_with_broadcast() {
    let t = unsafe { make_tensor(vec![1.0_f64, 2.0, 3.0], Ix2(1, 3)) };
    let b = t.broadcast_to([2, 3]).expect("compatible shapes");
    // Broadcast: shape [2, 3], strides [0, 1]
    assert_eq!(b.strides(), &[0, 1]);
    // Transpose: shape [3, 2], strides [1, 0]
    let bt = b.transpose();
    assert_eq!(bt.shape(), &[3, 2]);
    assert_eq!(bt.strides(), &[1, 0]);
    // Zero-copy through both operations: pointer unchanged.
    assert_eq!(bt.as_ptr(), t.as_ptr());
}

#[test]
fn test_shape_integration_transpose_6d() {
    let x = unsafe { make_tensor(Vec::<f64>::new(), Ix6(2, 3, 4, 5, 6, 7)) };
    let y = x.transpose();
    assert_eq!(y.shape(), &[7, 6, 5, 4, 3, 2]);
    assert_eq!(y.transpose().shape(), x.shape());
}

#[test]
fn test_shape_integration_transpose_dyn() {
    let x = unsafe { make_tensor(Vec::<i32>::new(), IxDyn::from_slice(&[2, 3, 4, 5, 6])) };
    let y = x.transpose();
    assert_eq!(y.shape(), &[6, 5, 4, 3, 2]);
    assert_eq!(y.len(), x.len());
    assert_eq!(y.transpose().strides(), x.strides());
}

#[test]
fn test_shape_integration_transpose_large_array_2d() {
    let x = unsafe { make_tensor(Vec::<f64>::new(), Ix2(3162, 3162)) };
    let src_ptr = x.as_ptr();
    let y = x.transpose();
    assert_eq!(y.storage_kind(), StorageKind::View);
    assert_eq!(y.shape(), &[3162, 3162]);
    assert_eq!(y.len(), x.len());
    assert_eq!(y.as_ptr(), src_ptr);
    assert_eq!(y.transpose().shape(), x.shape());
    assert_eq!(y.transpose().strides(), x.strides());
}

// ── Additional integration tests for transpose ──

mod tests {
    use xenon::dimension::{Ix2, Ix6};
    use xenon::storage::Owned;
    use xenon::tensor::{TensorBase, StorageKind};

    /// Helper: construct tensor via fast path.
    unsafe fn make_tensor<A: xenon::element::Element, D: xenon::dimension::Dimension>(
        data: Vec<A>,
        shape: D,
    ) -> TensorBase<Owned<A>, D> {
        unsafe { TensorBase::from_raw_vec_unchecked(data, shape) }
    }

    /// Access element at logical index.
    unsafe fn read_at<'a, S, D, A>(tensor: &'a TensorBase<S, D>, indices: &[usize]) -> &'a A
    where
        S: xenon::storage::Storage<Elem = A>,
        D: xenon::dimension::Dimension,
        A: xenon::element::Element,
    {
        debug_assert_eq!(indices.len(), tensor.ndim());
        let strides = tensor.strides();
        let mut rel_offset: isize = 0;
        for (axis, &idx) in indices.iter().enumerate() {
            rel_offset += (idx as isize) * (strides[axis] as isize);
        }
        unsafe { &*tensor.as_ptr().offset(rel_offset) }
    }

    #[test]
    fn test_transpose_2d() {
        // 2×3 F-order: data = [1, 2, 3, 4, 5, 6]
        //   logical = [[1, 3, 5], [2, 4, 6]]
        let x = unsafe { make_tensor(vec![1_i32, 2, 3, 4, 5, 6], Ix2(2, 3)) };
        let y = x.transpose();
        // Transpose swaps axes: shape [3, 2]
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.ndim(), 2);
        // Verify element access via logical indices (transposed order).
        unsafe {
            assert_eq!(*read_at(&y, &[0, 0]), *read_at(&x, &[0, 0])); // 1
            assert_eq!(*read_at(&y, &[0, 1]), *read_at(&x, &[1, 0])); // 2
            assert_eq!(*read_at(&y, &[1, 0]), *read_at(&x, &[0, 1])); // 3
            assert_eq!(*read_at(&y, &[2, 1]), *read_at(&x, &[1, 2])); // 6
        }
        // Transpose of transpose recovers original shape and strides.
        let z = y.transpose();
        assert_eq!(z.shape(), x.shape());
        assert_eq!(z.strides(), x.strides());
        // Transpose always produces a View.
        assert_eq!(y.storage_kind(), StorageKind::View);
    }

    #[test]
    fn test_transpose_high_dim() {
        // 6-D tensor: transpose reverses all axes.
        let x = unsafe { make_tensor(Vec::<f64>::new(), Ix6(2, 3, 4, 5, 6, 7)) };
        let y = x.transpose();
        assert_eq!(y.shape(), &[7, 6, 5, 4, 3, 2]);
        // Double transpose recovers original shape.
        assert_eq!(y.transpose().shape(), x.shape());
        // Storage kind is View.
        assert_eq!(y.storage_kind(), StorageKind::View);
    }
}

// ── Additional integration tests for transpose ──

#[test]
fn test_transpose_2d() {
    // 2×3 F-order: data = [1, 2, 3, 4, 5, 6]
    //   logical = [[1, 3, 5], [2, 4, 6]]
    let x = unsafe { make_tensor(vec![1_i32, 2, 3, 4, 5, 6], Ix2(2, 3)) };
    let y = x.transpose();
    // Transpose swaps axes: shape [3, 2]
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(y.ndim(), 2);
    // Verify element access via logical indices (transposed order).
    unsafe {
        assert_eq!(*read_at(&y, &[0, 0]), *read_at(&x, &[0, 0])); // 1
        assert_eq!(*read_at(&y, &[0, 1]), *read_at(&x, &[1, 0])); // 2
        assert_eq!(*read_at(&y, &[1, 0]), *read_at(&x, &[0, 1])); // 3
        assert_eq!(*read_at(&y, &[2, 1]), *read_at(&x, &[1, 2])); // 6
    }
    // Transpose of transpose recovers original shape and strides.
    let z = y.transpose();
    assert_eq!(z.shape(), x.shape());
    assert_eq!(z.strides(), x.strides());
    // Transpose always produces a View.
    assert_eq!(y.storage_kind(), StorageKind::View);
}

#[test]
fn test_transpose_high_dim() {
    // 6-D tensor: transpose reverses all axes.
    let x = unsafe { make_tensor(Vec::<f64>::new(), Ix6(2, 3, 4, 5, 6, 7)) };
    let y = x.transpose();
    assert_eq!(y.shape(), &[7, 6, 5, 4, 3, 2]);
    // Double transpose recovers original shape.
    assert_eq!(y.transpose().shape(), x.shape());
    // Storage kind is View.
    assert_eq!(y.storage_kind(), StorageKind::View);
}
